// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui.c
 * @brief   Central MVC controller, undo/redo ring, input dispatcher & event loop for cacaosim-CTRL
 *
 * Architectural Role in `cacaosim-CTRL`:
 *   Acts as the central Model-View-Controller (MVC) coordinator connecting the declarative `.bench`
 *   data model (`cacaosim_bench_cfg_t`), the zero-FFT optical/spectral analyzer
 *   (`cacaosim_bench_ctx_t`), the live `MILK_SHM_DIR` monitor (`cs_shm_mon_t`), and the cell-diffed
 *   terminal renderer (`cacaosim_tui_render.c` / `cacaosim_tui_views.c`). Specifically owns:
 *     1. Undo/Redo Snapshot Ring (`push_undo()`, `do_undo()`, `do_redo()`): 24-deep value-snapshot
 *        history of complete bench configurations and cursor selections.
 *     2. External File Persistence, Hot-Reload & `$EDITOR` Round-Trip (`save_current_file()`,
 *        `trigger_save()`, `trigger_external_editor()`, `stat_mtime_ns()`): Saves/loads `.bench`
 *        files, watches nanosecond `st_mtim` on unmodified files for live reload, and cleanly
 *        suspends/restores raw alternate-screen TTY mode around `$VISUAL` / `$EDITOR` subprocesses.
 *     3. Topological Arranger & Parameter Mutation Actions (`action_*()`, `start_param_edit()`):
 *        Executes smart optical splicing, cloning, deletion with beam bridging, upstream/downstream
 *        reordering, arm swapping, trial-validated beam rewiring, and inline parameter editing.
 *     4. Immediate-Mode Hit-Box Registry & Event Dispatcher (`cs_tui_add_hit()`,
 *        `handle_modal_event()`, `handle_mouse_event()`): Dispatches keyboard shortcuts, bracketed
 *        paste, SGR mouse clicks, pane-divider drags, scroll wheel ticks, and subprocess launchers
 *        (`cacaosim-bench` / `htsim` / `mkpupil` via `cs_tui_exec_*()`).
 *     5. Main Event Loop & Headless Verification API (`cacaosim_tui_run()`,
 *        `cacaosim_tui_render_headless()`): Runs the `poll`-driven frame loop (`120 ms` idle tick,
 *        instant input wakeup) and provides a TTY-free plain-text snapshot renderer for `ctest`.
 *
 * Design Rationale & Motivation:
 *   - Full-Struct Value Snapshots for Undo/Redo: Topological splicing operations (inserting a
 *     dichroic, removing a mirror and bridging its beams, or swapping splitter arms) simultaneously
 *     rewrite `in_beam` / `out_beam` across multiple nodes. Because `cacaosim_bench_cfg_t` is a
 *     self-contained flat POD struct with zero heap pointers, copying it by value into a 24-slot
 *     ring makes undo, redo, and failed-edit rollback O(1) and completely free of inverse-diff
 *     bugs.
 *   - Immediate-Mode Hit-Testing (`cs_tui_add_hit()`): Terminal layouts reflow continuously across
 *     `SIGWINCH` resizes, draggable pane dividers, scroll offsets, and floating modals. Rebuilding
 *     `app->hits[]` during each render pass and scanning back-to-front (`n_hits - 1` down to `0`)
 *     guarantees modals naturally occlude background panes and mouse coordinates never drift from
 *     visual glyph positions.
 *   - Gated Hot-Reload (`!app->dirty`): Watching nanosecond disk `mtime` only when the in-memory
 *     buffer has no unsaved edits lets `cacaosim-CTRL` double as a live-updating schematic viewer
 *     alongside external editors or scripts without risking data loss on active TUI edits.
 */

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "bench/cacaosim_bench_schema.h"
#include "bench/cacaosim_tui_internal.h"
#include "pupil/cacaosim_pupil.h"

/**
 * cacaosim_tui_opts_defaults() - Initialize `cacaosim_tui_opts_t` with safe interactive defaults
 * @opts: Launch options struct to populate (no-op if NULL)
 *
 * Role:
 *   Resets all TUI startup options to deterministic defaults (`color_level = -1` auto-detect,
 *   `CACAOSIM_THEME_DARK` palette, UTF-8 box-drawing enabled, initial tab `F2:BUILDER`, fused
 *   `htsim` deploy mode, atmospheric turbulence off with `0.7"` nominal seeing).
 *
 * Rationale:
 *   Centralizing default initialization ensures CLI callers (`cacaosim-CTRL`), embedded callers
 *   (`cacaosim-bench gui`), and headless unit tests (`cacaosim_tui_render_headless()`) start from
 *   identical baseline settings even as new fields are added to `cacaosim_tui_opts_t`.
 */
void cacaosim_tui_opts_defaults(cacaosim_tui_opts_t *opts)
{
    if (opts == NULL)
    {
        return;
    }
    memset(opts, 0, sizeof(*opts));
    opts->color_level   = -1;
    opts->theme         = CACAOSIM_THEME_DARK;
    opts->ascii_only    = 0;
    opts->initial_view  = CACAOSIM_TUI_VIEW_BUILDER;
    opts->deploy_mode   = 0;
    opts->with_atmturb  = 0;
    opts->seeing_arcsec = 0.8f;
}

/**
 * stat_mtime_ns() - Query a file's last-modification timestamp in integer nanoseconds
 * @path: Filesystem path to inspect (nullable / empty returns `0`)
 *
 * Role:
 *   Provides a single scalar `uint64_t` modification token (`tv_sec * 1e9 + tv_nsec`) used by the
 *   main loop's external `.bench` hot-reload watcher and updated after every local save.
 *
 * Rationale:
 *   Sub-second `st_mtim` resolution is essential on modern Linux filesystems because automated
 *   scripts or fast `$EDITOR` saves can modify a `.bench` file multiple times within the same
 *   wall-clock second; packing seconds and nanoseconds into `uint64_t` avoids missed reloads.
 *
 * Return: Modification timestamp in nanoseconds since the Unix epoch, or `0` if `stat()` fails.
 */
static uint64_t stat_mtime_ns(const char *path)
{
    if (path == NULL || path[0] == '\0')
    {
        return 0u;
    }
    struct stat st;
    if (stat(path, &st) != 0)
    {
        return 0u;
    }
    return ((uint64_t) st.st_mtim.tv_sec * 1000000000ULL) + (uint64_t) st.st_mtim.tv_nsec;
}

/**
 * cs_tui_add_hit() - Register an interactive screen rectangle for mouse clicks and hover tooltips
 * @app:  Master TUI application state (no-op if NULL or `CS_MAX_HITS` reached)
 * @kind: Semantic hit-target category (`cs_hit_kind_t`)
 * @arg:  Context-dependent index payload (e.g., tab ID, element row `0..N`, parameter index)
 * @x:    Leftmost 0-based screen column of the clickable region
 * @y:    Topmost 0-based screen row of the clickable region
 * @w:    Width in terminal columns (`<= 0` ignored)
 * @h:    Height in terminal rows (`<= 0` ignored)
 * @tip:  Optional hover tooltip text copied into the hit record (nullable)
 *
 * Role:
 *   Appends a screen-space hit box to `app->hits[]` during immediate-mode frame rendering so
 *   `handle_mouse_event()` and `render_hover_tooltip()` can map `(mouse_x, mouse_y)` coordinates
 *   back to logical UI actions.
 *
 * Rationale:
 *   Because widgets are drawn back-to-front (panes first, then header/footer bars, then floating
 *   modals on top), appending to a flat array and iterating in reverse (`n_hits - 1` down to `0`)
 *   naturally gives z-order priority to foreground modals and buttons without maintaining a
 *   retained-mode widget hierarchy.
 */
void cs_tui_add_hit(
    cs_tui_app_t *app,
    cs_hit_kind_t kind,
    int           arg,
    int           x,
    int           y,
    int           w,
    int           h,
    const char   *tip)
{
    if (app == NULL || app->n_hits >= CS_MAX_HITS || w <= 0 || h <= 0)
    {
        return;
    }
    cs_hit_t *ht = &app->hits[app->n_hits++];
    ht->kind     = kind;
    ht->arg      = arg;
    ht->x        = x;
    ht->y        = y;
    ht->w        = w;
    ht->h        = h;
    snprintf(ht->tip, sizeof(ht->tip), "%s", (tip != NULL) ? tip : "");
}

/**
 * cs_tui_log() - Post a printf-formatted status toast and append a timestamped activity log entry
 * @app:   Master TUI application state (no-op if NULL)
 * @level: Severity level (`0` = info, `1` = ok/green, `2` = warning/amber, `3` = error/red)
 * @fmt:   `printf`-style format string (no-op if NULL)
 *
 * Role:
 *   Simultaneously updates the transient 4-second bottom toast banner (`app->toast`) for immediate
 *   visual feedback on any tab and appends a `[HH:MM:SS]`-prefixed record to the 32-line scrolling
 *   activity history (`app->logs[]`) displayed in `F5:STATUS`.
 *
 * Rationale:
 *   Operators working in `F2:BUILDER` need instant confirmation or error diagnostics at the bottom
 *   of the screen without modal popups interrupting keyboard flow, while retaining a persistent
 *   chronological audit trail in `F5:STATUS` when troubleshooting deployment or parse errors.
 */
void cs_tui_log(cs_tui_app_t *app, int level, const char *fmt, ...)
{
    if (app == NULL || fmt == NULL)
    {
        return;
    }
    char    msg[220];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    /* Arm transient 4-second toast notification on row (rows - 2) */
    snprintf(app->toast, sizeof(app->toast), "%s", msg);
    app->toast_level   = level;
    app->toast_until_s = cs_tui_now_sec() + 4.0;

    time_t     now = time(NULL);
    struct tm *tmv = localtime(&now);
    char       stamped[160];
    if (tmv != NULL)
    {
        snprintf(stamped, sizeof(stamped), "[%02d:%02d:%02d] %.140s", tmv->tm_hour, tmv->tm_min,
                 tmv->tm_sec, msg);
    }
    else
    {
        snprintf(stamped, sizeof(stamped), "%.155s", msg);
    }

    /* Append to bounded FIFO log buffer, shifting older lines up when full */
    if (app->n_logs < CS_LOG_LINES)
    {
        snprintf(app->logs[app->n_logs++], sizeof(app->logs[0]), "%s", stamped);
    }
    else
    {
        memmove(app->logs[0], app->logs[1], (size_t) (CS_LOG_LINES - 1) * sizeof(app->logs[0]));
        snprintf(app->logs[CS_LOG_LINES - 1], sizeof(app->logs[0]), "%s", stamped);
    }
}

/**
 * cs_tui_refresh_analysis() - Lazily recompute the 16x16 pupil preview and zero-FFT DAG photometry
 * @app: Master TUI application state (no-op if NULL or `!app->ana_dirty`)
 *
 * Role:
 *   When `app->ana_dirty` is set after a configuration edit, preset load, undo/redo, or file
 *   reload, regenerates the anti-aliased `16x16` entrance pupil thumbnail (`app->pup16`) via
 *   `cacaosim_make_pupil()` and runs topological sorting + 32-bin (`0.40 .. 2.40 um`) spectral
 *   throughput propagation via `cacaosim_bench_analyze()`.
 *
 * Rationale:
 *   Gating pupil rasterization and spectral graph propagation behind the `ana_dirty` dirty flag
 *   keeps idle `120 ms` telemetry refreshes completely free of redundant math, while ensuring that
 *   every parameter step (e.g., changing `cobs`, `star_mag`, or a dichroic `lambda_cut`) updates
 *   the pupil glyph, DAG validity banner, and per-sensor photon flux (`ph/fr`) with sub-ms
 *   latency before the next frame renders.
 */
void cs_tui_refresh_analysis(cs_tui_app_t *app)
{
    if (app == NULL || !app->ana_dirty)
    {
        return;
    }
    cacaosim_pupil_cfg_t pcfg = {
        .nwf           = 16u,
        .pup_diam      = app->cfg.pup_diam,
        .cobs          = app->cfg.cobs,
        .nspider       = app->cfg.nspider,
        .spider_w      = (app->cfg.spider_w > 0.0f && app->cfg.spider_w < 0.05f)
                             ? 0.05f
                             : app->cfg.spider_w,
        .spider_deg    = app->cfg.spider_deg,
        .spider_offset = app->cfg.spider_offset,
    };
    app->pup16_sum = cacaosim_make_pupil(&pcfg, app->pup16);

    app->dag_err[0] = '\0';
    if (cacaosim_bench_analyze(&app->ctx, &app->cfg, app->dag_err, sizeof(app->dag_err)) == 0)
    {
        app->dag_valid = 1;
    }
    else
    {
        app->dag_valid = 0;
    }
    app->ana_dirty = 0;
}

/* ── Undo / Redo History Ring ─────────────────────────────────────────────── */

/**
 * push_undo() - Snapshot the current bench configuration and selection onto the undo history
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Saves `app->cfg` and `app->sel_item` at `app->undo.pos` prior to any mutating action,
 *   truncating any forward redo branch (`count = pos`) and evicting the oldest snapshot if the
 *   `CS_UNDO_CAP` (24-step) buffer is full.
 *
 * Rationale:
 *   Capturing full value snapshots of `cacaosim_bench_cfg_t` before mutations makes every
 *   topological and parameter edit atomically reversible without maintaining complex per-operation
 *   inverse deltas, and also allows callers (`action_swap_arms`, inline parameter parser) to
 *   cleanly pop a speculative undo entry if validation rejects the edit.
 */
static void push_undo(cs_tui_app_t *app)
{
    cs_undo_ring_t *u = &app->undo;
    /* Discard invalidated forward redo states when branching onto a new edit */
    if (u->pos < u->count)
    {
        u->count = u->pos;
    }
    if (u->count >= CS_UNDO_CAP)
    {
        memmove(&u->snaps[0], &u->snaps[1], (size_t) (CS_UNDO_CAP - 1) * sizeof(u->snaps[0]));
        memmove(&u->sel_item[0], &u->sel_item[1],
                (size_t) (CS_UNDO_CAP - 1) * sizeof(u->sel_item[0]));
        u->count = CS_UNDO_CAP - 1;
    }
    u->snaps[u->count]    = app->cfg;
    u->sel_item[u->count] = app->sel_item;
    u->count++;
    u->pos = u->count;
}

/**
 * do_undo() - Step backward one snapshot in the undo history ring
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Restores the preceding `cacaosim_bench_cfg_t` and `sel_item` from `app->undo`, stashing the
 *   live tip state at `snaps[u->count]` when undoing from the newest state so `do_redo()` can
 *   recover it.
 *
 * Rationale:
 *   Stashing the live tip lazily on the first `u` press avoids keeping a duplicate copy of the
 *   current state on every edit while guaranteeing lossless round-trip `Undo` <-> `Redo` (`Ctrl+R`
 *   / `U`) across both parameter tweaks and multi-node topological splices.
 */
static void do_undo(cs_tui_app_t *app)
{
    cs_undo_ring_t *u = &app->undo;
    if (u->pos <= 0)
    {
        cs_tui_log(app, 2, "Nothing to undo");
        return;
    }
    /* Preserve live post-edit state at the tail slot so Redo can return to it */
    if (u->pos == u->count && u->count < CS_UNDO_CAP)
    {
        u->snaps[u->count]    = app->cfg;
        u->sel_item[u->count] = app->sel_item;
    }
    u->pos--;
    app->cfg       = u->snaps[u->pos];
    app->sel_item  = u->sel_item[u->pos];
    app->dirty     = 1;
    app->ana_dirty = 1;
    cs_tui_log(app, 0, "Undo (%d/%d)", u->pos, u->count);
}

/**
 * do_redo() - Step forward one snapshot in the undo/redo history ring
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Advances `app->undo.pos` and restores the stashed forward `cacaosim_bench_cfg_t` and
 *   `sel_item`, marking both disk state and optical analysis dirty.
 *
 * Rationale:
 *   Complements `do_undo()` so operators experimenting with alternative optical layouts or WFS
 *   samplings can toggle back and forth (`u` / `Ctrl+R`) while comparing flux and pixel-scale
 *   readouts in real time.
 */
static void do_redo(cs_tui_app_t *app)
{
    cs_undo_ring_t *u = &app->undo;
    if (u->pos >= u->count)
    {
        cs_tui_log(app, 2, "Nothing to redo");
        return;
    }
    u->pos++;
    if (u->pos < CS_UNDO_CAP)
    {
        app->cfg       = u->snaps[u->pos];
        app->sel_item  = u->sel_item[u->pos];
        app->dirty     = 1;
        app->ana_dirty = 1;
        cs_tui_log(app, 0, "Redo (%d/%d)", u->pos, u->count);
    }
}

/* ── File Save / Load / External $EDITOR ──────────────────────────────────── */

/**
 * save_current_file() - Serialize and write `app->cfg` to disk and clear the modified flag
 * @app:  Master TUI application state (non-NULL)
 * @path: Destination `.bench` filesystem path (non-NULL)
 *
 * Role:
 *   Persists the active optical bench configuration via `cacaosim_bench_save_file()`, updates
 *   `app->file_path` and `app->disk_mtime_ns`, clears `app->dirty`, and logs a success or error
 *   toast.
 *
 * Rationale:
 *   Refreshing `app->disk_mtime_ns` immediately after writing prevents the main loop's external
 *   file watcher from misinterpreting the TUI's own save as an external modification.
 *
 * Return: 0 on success, -1 on file I/O or serialization failure.
 */
static int save_current_file(cs_tui_app_t *app, const char *path)
{
    char ebuf[160] = { 0 };
    if (cacaosim_bench_save_file(path, &app->cfg, ebuf, sizeof(ebuf)) != 0)
    {
        cs_tui_log(app, 3, "Save failed: %s", ebuf);
        return -1;
    }
    snprintf(app->file_path, sizeof(app->file_path), "%s", path);
    app->disk_mtime_ns = stat_mtime_ns(path);
    app->dirty         = 0;
    cs_tui_log(app, 1, "Saved '%s' (%u optical elements)", path, app->cfg.n_nodes);
    return 0;
}

/**
 * trigger_save() - Save directly to the bound `.bench` path or open the Save-As path prompt
 * @app:     Master TUI application state (non-NULL)
 * @save_as: Non-zero (`A` hotkey) to force the `CS_MODAL_SAVE` path dialog; 0 (`s` / `Ctrl+S`)
 *           to overwrite `app->file_path` directly when already bound
 *
 * Role:
 *   Implements both quick save (`s` / `Ctrl+S` / `[s:Save]`) and explicit Save-As (`A`), seeding
 *   the modal text field with `"custom_ao.bench"` when working from an unnamed in-memory preset.
 *
 * Rationale:
 *   Eliminates repetitive confirmation dialogs during iterative parameter tuning once a file path
 *   is established, while still prompting cleanly before creating the first file on disk.
 */
static void trigger_save(cs_tui_app_t *app, int save_as)
{
    if (!save_as && app->file_path[0] != '\0')
    {
        save_current_file(app, app->file_path);
        return;
    }
    app->modal = CS_MODAL_SAVE;
    snprintf(app->modal_input, sizeof(app->modal_input), "%s",
             app->file_path[0] ? app->file_path : "custom_ao.bench");
    app->modal_cursor = (int) strlen(app->modal_input);
}

/**
 * trigger_external_editor() - Hand off the `.bench` file to `$VISUAL` / `$EDITOR` and reload it
 * @app:    Master TUI application state (non-NULL)
 * @tty_fd: In/out pointer to the active controlling TTY file descriptor (reopened on return)
 *
 * Role:
 *   Flushes `app->cfg` to disk, exits raw alternate-screen terminal mode (`cs_tui_term_exit()`),
 *   invokes `$VISUAL`, `$EDITOR`, or `"vi"` synchronously on the `.bench` file, re-enters raw
 *   SGR-mouse TUI mode (`cs_tui_term_enter()`), and parses the edited file onto the undo stack.
 *
 * Rationale:
 *   Interactive GUI builders excel at visual wiring and stepping single parameters, whereas text
 *   editors excel at bulk search-and-replace or commenting sections. Cleanly releasing and
 *   reacquiring raw `termios` + alternate screen state lets both workflows coexist seamlessly,
 *   while keeping the pre-edit `app->cfg` intact if the user introduces a syntax error in Vim.
 */
static void trigger_external_editor(cs_tui_app_t *app, int *tty_fd)
{
    char target[512];
    if (app->file_path[0] != '\0')
    {
        snprintf(target, sizeof(target), "%s", app->file_path);
    }
    else
    {
        snprintf(target, sizeof(target), "custom_ao.bench");
    }
    if (save_current_file(app, target) != 0)
    {
        return;
    }

    const char *ed = getenv("VISUAL");
    if (ed == NULL || ed[0] == '\0')
    {
        ed = getenv("EDITOR");
    }
    if (ed == NULL || ed[0] == '\0')
    {
        ed = "vi";
    }

    /* Leave raw/alt-screen mode so the interactive text editor owns the terminal cleanly */
    cs_tui_term_exit();
    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "%.256s \"%.500s\"", ed, target);
    int rc = system(cmd);
    (void) rc;

    /* Re-enter raw TUI mode and query potentially resized terminal dimensions */
    int cols = app->scr.cols, rows = app->scr.rows;
    *tty_fd = cs_tui_term_enter(&cols, &rows);
    cs_scr_resize(&app->scr, cols, rows);

    cacaosim_bench_cfg_t loaded;
    char                 ebuf[160] = { 0 };
    if (cacaosim_bench_load_file(target, &loaded, ebuf, sizeof(ebuf)) == 0)
    {
        push_undo(app);
        app->cfg           = loaded;
        app->disk_mtime_ns = stat_mtime_ns(target);
        app->dirty         = 0;
        app->ana_dirty     = 1;
        if (app->sel_item > (int) app->cfg.n_nodes)
        {
            app->sel_item = (int) app->cfg.n_nodes;
        }
        cs_tui_log(app, 1, "Reloaded '%s' from editor", target);
    }
    else
    {
        cs_tui_log(app, 3, "Syntax error in edited file: %s", ebuf);
    }
}

/* ── Arranger & Inspector Actions ─────────────────────────────────────────── */

/**
 * action_open_add_modal() - Open the 9-type `ADD OPTICAL ELEMENT` wizard modal
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Activates `CS_MODAL_ADD` with the cursor on item `0` (`[dm]`) and `modal_beam_idx = -1`
 *   (`AUTO` smart topological splice downstream of the currently selected element).
 *
 * Rationale:
 *   Defaulting `modal_beam_idx` to `-1` (`AUTO`) enables two-keystroke hardware insertion (`a`
 *   followed by `1`..`9`) that splices inline optics into the active beam chain automatically,
 *   while still letting the user press `Tab` to target any explicit named beam arm.
 */
static void action_open_add_modal(cs_tui_app_t *app)
{
    app->modal          = CS_MODAL_ADD;
    app->modal_sel      = 0;
    app->modal_beam_idx = -1;
}

/**
 * action_commit_add() - Instantiate and splice a newly chosen optical element into the bench
 * @app:    Master TUI application state (non-NULL)
 * @choice: Catalog row index `[0..8]` selected in `CS_MODAL_ADD` (`1` selects 2-axis TTM mode)
 *
 * Role:
 *   Translates the 9-entry UI catalog index into `(cacaosim_node_kind_t, submode)`, records an
 *   undo snapshot, invokes `cacaosim_bench_add_node()` with either automatic downstream splicing
 *   or the user-selected override beam, and focuses the newly inserted element.
 *
 * Rationale:
 *   Exposing 2-axis Tip-Tilt Mirrors (`choice == 1`, `CACAOSIM_NODE_DM` with `submode = 1`) as a
 *   first-class peer alongside zonal DMs (`choice == 0`) matches physical AO bench mental models
 *   without requiring users to add a generic DM and manually toggle `dm_mode` afterward.
 */
static void action_commit_add(cs_tui_app_t *app, int choice)
{
    static const cacaosim_node_kind_t kinds[9] = {
        CACAOSIM_NODE_DM,     CACAOSIM_NODE_DM,     CACAOSIM_NODE_SPLITTER,
        CACAOSIM_NODE_CORO,   CACAOSIM_NODE_PYRWFS, CACAOSIM_NODE_SHWFS,
        CACAOSIM_NODE_ZWFS,   CACAOSIM_NODE_FPCAM,  CACAOSIM_NODE_WFE,
    };
    if (choice < 0 || choice >= 9)
    {
        return;
    }
    char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
    uint32_t nb    = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
    const char *tb = (app->modal_beam_idx >= 0 && (uint32_t) app->modal_beam_idx < nb)
                         ? beams[app->modal_beam_idx]
                         : NULL;

    push_undo(app);
    int after = (app->sel_item <= 0) ? -1 : (app->sel_item - 1);
    int idx   = cacaosim_bench_add_node(&app->cfg, kinds[choice], (choice == 1) ? 1 : 0, after, tb);
    app->modal = CS_MODAL_NONE;
    if (idx < 0)
    {
        cs_tui_log(app, 3, "Bench node table full (max %d elements)", CACAOSIM_BENCH_MAX_NODES);
        return;
    }
    app->sel_item  = idx + 1;
    app->dirty     = 1;
    app->ana_dirty = 1;
    cs_tui_log(app, 1, "Inserted [%s] (in=%s)", app->cfg.nodes[idx].id,
               app->cfg.nodes[idx].in_beam);
}

/**
 * action_clone_selected() - Duplicate the currently selected optical element with fresh IDs/streams
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Validates that a concrete node `[1..n_nodes]` (not `[0]` entrance telescope) is selected,
 *   pushes an undo snapshot, and delegates to `cacaosim_bench_clone_node()` to create an
 *   auto-numbered copy spliced in series (for inline elements) or parallel (for sensors).
 *
 * Rationale:
 *   Multi-stage benches frequently pair nearly identical woofer/tweeter DMs, dual science cameras,
 *   or twin WFS arms; cloning preserves custom detector/servo settings while automatically minting
 *   collision-free section IDs and `MILK_SHM_DIR` stream names.
 */
static void action_clone_selected(cs_tui_app_t *app)
{
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        cs_tui_log(app, 2, "Select an optical element [1..N] to duplicate");
        return;
    }
    push_undo(app);
    int idx = cacaosim_bench_clone_node(&app->cfg, (uint32_t) (app->sel_item - 1));
    if (idx >= 0)
    {
        app->sel_item  = idx + 1;
        app->dirty     = 1;
        app->ana_dirty = 1;
        cs_tui_log(app, 1, "Duplicated -> [%s]", app->cfg.nodes[idx].id);
    }
}

/**
 * action_delete_selected() - Remove the selected element and heal upstream/downstream beam links
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Guards the mandatory entrance `[bench]` root and the last remaining optical node against
 *   deletion, records an undo snapshot, and calls `cacaosim_bench_remove_node()` to delete the
 *   element while re-bridging downstream consumers onto its `in_beam`.
 *
 * Rationale:
 *   Without automatic beam bridging, deleting an inline DM or WFE map in the middle of an optical
 *   relay would orphan every downstream beamsplitter, coronagraph, and camera. Healing links
 *   inside `cacaosim_bench_remove_node()` keeps the bench DAG valid after single-key (`d` / `Del`)
 *   deletions.
 */
static void action_delete_selected(cs_tui_app_t *app)
{
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        cs_tui_log(app, 2, "Entrance [bench] telescope cannot be deleted");
        return;
    }
    if (app->cfg.n_nodes <= 1u)
    {
        cs_tui_log(app, 2, "Bench must retain at least 1 optical element");
        return;
    }
    char rem_id[64];
    snprintf(rem_id, sizeof(rem_id), "%s", app->cfg.nodes[app->sel_item - 1].id);
    push_undo(app);
    if (cacaosim_bench_remove_node(&app->cfg, (uint32_t) (app->sel_item - 1)) == 0)
    {
        if (app->sel_item > (int) app->cfg.n_nodes)
        {
            app->sel_item = (int) app->cfg.n_nodes;
        }
        app->dirty     = 1;
        app->ana_dirty = 1;
        cs_tui_log(app, 1, "Removed [%s] and bridged downstream beams", rem_id);
    }
}

/**
 * action_move_selected() - Reorder the selected element upstream (`-1`) or downstream (`+1`)
 * @app: Master TUI application state (non-NULL)
 * @dir: Direction to move (`< 0` for upstream `[K / Shift+Up]`, `> 0` for downstream `[J]`)
 *
 * Role:
 *   Attempts to move the selected optical node via `cacaosim_bench_move_node()` (which optically
 *   swaps inline serial neighbors' beam links as well as array order) and commits an undo entry
 *   only when a valid move actually occurred (`nidx >= 0`).
 *
 * Rationale:
 *   Holding a temporary `before` copy so `push_undo()` is recorded only when `nidx >= 0` prevents
 *   no-op boundary presses (e.g., pressing `Shift+Up` on the very first element) from polluting
 *   the 24-step undo ring with duplicate snapshots.
 */
static void action_move_selected(cs_tui_app_t *app, int dir)
{
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        return;
    }
    cacaosim_bench_cfg_t before = app->cfg;
    int                  prev_s = app->sel_item;
    int nidx = cacaosim_bench_move_node(&app->cfg, (uint32_t) (app->sel_item - 1), dir);
    if (nidx >= 0)
    {
        cacaosim_bench_cfg_t moved = app->cfg;
        app->cfg                   = before;
        app->sel_item              = prev_s;
        push_undo(app);
        app->cfg       = moved;
        app->sel_item  = nidx + 1;
        app->dirty     = 1;
        app->ana_dirty = 1;
        cs_tui_log(app, 1, "Moved [%s] %s", app->cfg.nodes[nidx].id,
                   (dir < 0) ? "upstream" : "downstream");
    }
}

/**
 * action_swap_arms() - Swap the two downstream output arms of a Beamsplitter or Coronagraph
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Invokes `cacaosim_bench_swap_branches()` on the selected node (`x` hotkey or `[Swp]` button)
 *   to exchange `out_refl` <-> `out_trans` on a splitter or `out_beam` <-> `out_lowfs` on a
 *   coronagraph, rolling back the speculative undo snapshot if the element is single-output.
 *
 * Rationale:
 *   When flipping a shortpass/longpass dichroic between WFS Reflection vs. Science Transmission,
 *   manually renaming two beam strings across multiple downstream nodes is tedious and error-prone
 *   due to intermediate collision states; atomic branch swapping completes it in one keystroke.
 */
static void action_swap_arms(cs_tui_app_t *app)
{
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        cs_tui_log(app, 2, "Select a Beamsplitter or Coronagraph to swap output arms");
        return;
    }
    push_undo(app);
    if (cacaosim_bench_swap_branches(&app->cfg, (uint32_t) (app->sel_item - 1)) == 0)
    {
        app->dirty     = 1;
        app->ana_dirty = 1;
        cs_tui_log(app, 1, "Swapped downstream output arms on [%s]",
                   app->cfg.nodes[app->sel_item - 1].id);
    }
    else
    {
        app->undo.pos--;
        app->undo.count--;
        cs_tui_log(app, 2, "Selected element has only one output arm");
    }
}

/**
 * action_open_wire_modal() - Open the `WIRE UPSTREAM OPTICAL BEAM` picker for the selected node
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Enumerates all available optical beams in `app->cfg`, pre-highlights the selected element's
 *   current `in_beam`, and activates `CS_MODAL_WIRE` (`w` hotkey or `[w:Wire Beam]` button).
 *
 * Rationale:
 *   Presenting every upstream beam alongside its live 32-bin spectrum bar, mean wavelength, and
 *   total photon flux (`ph/s`) eliminates typos when re-routing sensors across beamsplitter arms
 *   and immediately shows whether the target arm carries light in the sensor's passband.
 */
static void action_open_wire_modal(cs_tui_app_t *app)
{
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        cs_tui_log(app, 2, "Select an optical element [1..N] to rewire its input beam");
        return;
    }
    char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
    uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
    app->modal_sel = 0;
    for (uint32_t i = 0; i < nb; i++)
    {
        if (strcmp(beams[i], app->cfg.nodes[app->sel_item - 1].in_beam) == 0)
        {
            app->modal_sel = (int) i;
            break;
        }
    }
    app->modal = CS_MODAL_WIRE;
}

/**
 * start_param_edit() - Activate parameter editing, enum/boolean cycling, or beam wiring
 * @app: Master TUI application state (non-NULL)
 *
 * Role:
 *   Handles `Enter`, `Space`, or second-click activation on the highlighted row in the Parameter
 *   Inspector:
 *     - `CS_PTYPE_BEAM`: Opens `CS_MODAL_WIRE` via `action_open_wire_modal()`.
 *     - `CS_PTYPE_BOOL` / `CS_PTYPE_ENUM`: Immediately steps/toggles to the next valid discrete
 *       option with undo support.
 *     - Numeric / String / ID fields: Populates `app->edit_buf` with the formatted current value
 *       and enters inline text editing mode (`app->editing = 1`).
 *
 * Rationale:
 *   Dispatching by `cs_ptype_t` avoids forcing users to type `"true"` or memorize enum strings
 *   such as `"dichroic_lp"`, while giving full free-form numeric and SHM-name text entry where
 *   needed.
 */
static void start_param_edit(cs_tui_app_t *app)
{
    if (app->param_sel < 0 || app->param_sel >= app->n_params)
    {
        return;
    }
    const cs_param_desc_t *pd = &app->params[app->param_sel];
    if (pd->type == CS_PTYPE_BEAM)
    {
        action_open_wire_modal(app);
        return;
    }
    if (pd->type == CS_PTYPE_BOOL || pd->type == CS_PTYPE_ENUM)
    {
        char ebuf[128] = { 0 };
        push_undo(app);
        if (cs_param_step(&app->cfg, app->sel_item, pd, 1, 0, ebuf, sizeof(ebuf)) == 0)
        {
            app->dirty     = 1;
            app->ana_dirty = 1;
        }
        return;
    }
    cs_param_format_val(&app->cfg, app->sel_item, pd, app->edit_buf, sizeof(app->edit_buf));
    app->edit_cursor = (int) strlen(app->edit_buf);
    app->editing     = 1;
}

/* ── Modal & Keyboard / Mouse Event Dispatch ──────────────────────────────── */

/**
 * handle_modal_event() - Process keyboard and synthesized modal input while a popup is active
 * @app: Master TUI application state with `app->modal != CS_MODAL_NONE` (non-NULL)
 * @ev:  Decoded keyboard, paste, or right-click dismiss event (non-NULL)
 *
 * Role:
 *   Captures all input while a modal dialog (`CS_MODAL_ADD`, `CS_MODAL_WIRE`, `CS_MODAL_PRESET`,
 *   `CS_MODAL_SAVE`, `CS_MODAL_OPEN`, `CS_MODAL_THEME`, `CS_MODAL_CONFIRM`, or `CS_MODAL_HELP`)
 *   is open, handling `Esc` / right-click cancellation, path text entry + bracketed paste, `1..9`
 *   quick-select hotkeys, `Tab` target-beam cycling, and `Enter` confirmation.
 *
 * Rationale:
 *   Isolating modal event handling into a dedicated state-machine handler prevents modal keystrokes
 *   (such as typing `'d'` or `'q'` inside a `.bench` filename prompt) from leaking through to
 *   destructive global workspace shortcuts. In `CS_MODAL_WIRE`, candidate beam connections are
 *   trial-analyzed on a scratch `cacaosim_bench_cfg_t` copy via `cacaosim_bench_analyze()` before
 *   committing, actively rejecting topological feedback cycles with an explanatory error toast.
 */
static void handle_modal_event(cs_tui_app_t *app, const cs_input_ev_t *ev)
{
    /* Universal modal cancel: Escape key or SGR right mouse click */
    if (ev->key == CS_KEY_ESC || (ev->key == CS_KEY_MOUSE && ev->mbtn == CS_MOUSE_RIGHT))
    {
        app->modal = CS_MODAL_NONE;
        return;
    }

    /* Single-line path prompt modals: Save (.bench) and Open (.bench) */
    if (app->modal == CS_MODAL_SAVE || app->modal == CS_MODAL_OPEN)
    {
        if (ev->key == CS_KEY_ENTER)
        {
            if (app->modal == CS_MODAL_SAVE)
            {
                save_current_file(app, app->modal_input);
            }
            else
            {
                cacaosim_bench_cfg_t loaded;
                char                 ebuf[160] = { 0 };
                if (cacaosim_bench_load_file(app->modal_input, &loaded, ebuf, sizeof(ebuf)) == 0)
                {
                    push_undo(app);
                    app->cfg = loaded;
                    snprintf(app->file_path, sizeof(app->file_path), "%s", app->modal_input);
                    app->disk_mtime_ns = stat_mtime_ns(app->file_path);
                    app->dirty         = 0;
                    app->sel_item      = 0;
                    app->ana_dirty     = 1;
                    cs_tui_log(app, 1, "Loaded '%s'", app->file_path);
                }
                else
                {
                    cs_tui_log(app, 3, "Open failed: %s", ebuf);
                }
            }
            app->modal = CS_MODAL_NONE;
        }
        else if (ev->key == CS_KEY_BACKSPACE)
        {
            size_t l = strlen(app->modal_input);
            if (l > 0u)
            {
                app->modal_input[l - 1u] = '\0';
            }
        }
        else if (ev->key == CS_KEY_PASTE)
        {
            size_t l = strlen(app->modal_input);
            snprintf(app->modal_input + l, sizeof(app->modal_input) - l, "%s", ev->paste);
        }
        else if (ev->key >= 32 && ev->key < 127)
        {
            size_t l = strlen(app->modal_input);
            if (l + 1u < sizeof(app->modal_input))
            {
                app->modal_input[l]      = (char) ev->key;
                app->modal_input[l + 1u] = '\0';
            }
        }
        return;
    }

    /* Unsaved-changes confirmation gate before quitting (act=1) or loading a preset (act=2) */
    if (app->modal == CS_MODAL_CONFIRM)
    {
        if (ev->key == 'y' || ev->key == 'Y' || ev->key == CS_KEY_ENTER)
        {
            int act    = app->confirm_action;
            app->modal = CS_MODAL_NONE;
            if (act == 1)
            {
                app->running = 0;
            }
            else if (act == 2)
            {
                push_undo(app);
                cacaosim_bench_preset(&app->cfg, app->pending_preset);
                app->sel_item  = 0;
                app->dirty     = 1;
                app->ana_dirty = 1;
                cs_tui_log(app, 1, "Loaded preset '%s'",
                           cacaosim_bench_preset_name(app->pending_preset));
            }
        }
        else if (ev->key == 'n' || ev->key == 'N')
        {
            app->modal = CS_MODAL_NONE;
        }
        return;
    }

    if (app->modal == CS_MODAL_HELP)
    {
        if (ev->key == '?')
        {
            app->opt_search[0] = '\0';
            app->opt_scroll    = 0;
            app->opt_sel       = 0;
            app->modal         = CS_MODAL_OPTIONS;
        }
        else if (ev->key == 'h' || ev->key == 'H' || ev->key == 'q' || ev->key == CS_KEY_F1 ||
                 ev->key == CS_KEY_ENTER)
        {
            app->modal = CS_MODAL_NONE;
        }
        return;
    }

    if (app->modal == CS_MODAL_OPTIONS)
    {
        if (ev->key == CS_KEY_ENTER)
        {
            app->modal = CS_MODAL_NONE;
        }
        else if (ev->key == CS_KEY_UP)
        {
            if (app->opt_sel > 0)
            {
                app->opt_sel--;
            }
        }
        else if (ev->key == CS_KEY_DOWN)
        {
            app->opt_sel++;
        }
        else if (ev->key == CS_KEY_PGUP)
        {
            app->opt_sel = (app->opt_sel > 10) ? (app->opt_sel - 10) : 0;
        }
        else if (ev->key == CS_KEY_PGDN)
        {
            app->opt_sel += 10;
        }
        else if (ev->key == 21) /* Ctrl+U: clear filter */
        {
            app->opt_search[0] = '\0';
            app->opt_sel       = 0;
            app->opt_scroll    = 0;
        }
        else if (ev->key == CS_KEY_BACKSPACE)
        {
            size_t l = strlen(app->opt_search);
            if (l > 0u)
            {
                app->opt_search[l - 1u] = '\0';
                app->opt_sel            = 0;
                app->opt_scroll         = 0;
            }
        }
        else if (ev->key >= 32 && ev->key < 127)
        {
            size_t l = strlen(app->opt_search);
            if (l + 1u < sizeof(app->opt_search))
            {
                app->opt_search[l]      = (char) ev->key;
                app->opt_search[l + 1u] = '\0';
                app->opt_sel            = 0;
                app->opt_scroll         = 0;
            }
        }
        return;
    }

    /* List-selection modals: Add Element, Wire Beam, Preset Gallery, and Color Theme */
    int max_items = 1;
    if (app->modal == CS_MODAL_ADD)
    {
        max_items = 9;
    }
    else if (app->modal == CS_MODAL_PRESET)
    {
        max_items = CACAOSIM_PRESET_COUNT;
    }
    else if (app->modal == CS_MODAL_THEME)
    {
        max_items = CACAOSIM_THEME_COUNT;
    }
    else if (app->modal == CS_MODAL_WIRE)
    {
        char beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
        max_items = (int) cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
    }

    if (ev->key == CS_KEY_UP || ev->key == 'k')
    {
        app->modal_sel = (app->modal_sel + max_items - 1) % max_items;
    }
    else if (ev->key == CS_KEY_DOWN || ev->key == 'j')
    {
        app->modal_sel = (app->modal_sel + 1) % max_items;
    }
    else if (app->modal == CS_MODAL_ADD && ev->key == CS_KEY_TAB)
    {
        /* Cycle target input beam override: AUTO (-1) -> beam[0] -> ... -> beam[nb-1] -> AUTO */
        char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
        uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
        app->modal_beam_idx++;
        if (app->modal_beam_idx >= (int) nb)
        {
            app->modal_beam_idx = -1;
        }
    }
    else if (app->modal == CS_MODAL_THEME && (ev->key == 'a' || ev->key == 'A'))
    {
        app->scr.ascii_only = !app->scr.ascii_only;
        app->scr.force_full = 1;
    }
    else if (ev->key >= '1' && ev->key <= '9' && (ev->key - '1') < max_items)
    {
        /* Numeric accelerator 1..9 selects and immediately confirms the corresponding row */
        app->modal_sel = ev->key - '1';
        cs_input_ev_t enter_ev;
        memset(&enter_ev, 0, sizeof(enter_ev));
        enter_ev.key = CS_KEY_ENTER;
        handle_modal_event(app, &enter_ev);
    }
    else if (ev->key == CS_KEY_ENTER)
    {
        if (app->modal == CS_MODAL_ADD)
        {
            action_commit_add(app, app->modal_sel);
        }
        else if (app->modal == CS_MODAL_WIRE)
        {
            char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
            uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
            if (app->sel_item > 0 && (uint32_t) app->modal_sel < nb)
            {
                /* Trial-validate candidate beam connection so DAG cycles are caught cleanly */
                cacaosim_bench_cfg_t trial = app->cfg;
                snprintf(trial.nodes[app->sel_item - 1].in_beam,
                         sizeof(trial.nodes[app->sel_item - 1].in_beam), "%s",
                         beams[app->modal_sel]);
                cacaosim_bench_ctx_t tctx;
                char                 ebuf[160] = { 0 };
                if (cacaosim_bench_analyze(&tctx, &trial, ebuf, sizeof(ebuf)) != 0)
                {
                    cs_tui_log(app, 3, "Invalid wiring (%s)", ebuf);
                }
                else
                {
                    push_undo(app);
                    app->cfg       = trial;
                    app->dirty     = 1;
                    app->ana_dirty = 1;
                    cs_tui_log(app, 1, "Connected [%s].in -> '%s'",
                               app->cfg.nodes[app->sel_item - 1].id, beams[app->modal_sel]);
                }
            }
            app->modal = CS_MODAL_NONE;
        }
        else if (app->modal == CS_MODAL_PRESET)
        {
            int pid    = app->modal_sel;
            app->modal = CS_MODAL_NONE;
            if (app->dirty)
            {
                app->pending_preset = pid;
                app->confirm_action = 2;
                app->modal          = CS_MODAL_CONFIRM;
            }
            else
            {
                push_undo(app);
                cacaosim_bench_preset(&app->cfg, pid);
                app->sel_item  = 0;
                app->dirty     = 1;
                app->ana_dirty = 1;
                cs_tui_log(app, 1, "Loaded preset '%s'", cacaosim_bench_preset_name(pid));
            }
        }
        else if (app->modal == CS_MODAL_THEME)
        {
            app->opts.theme     = app->modal_sel;
            app->scr.force_full = 1;
            app->modal          = CS_MODAL_NONE;
            cs_tui_log(app, 0, "Theme set to '%s'", cs_tui_get_palette(app->opts.theme)->name);
        }
    }
}

/**
 * handle_mouse_event() - Dispatch SGR 1006 mouse motion, wheel scroll, drag, and click events
 * @app: Master TUI application state containing the latest frame's `app->hits[]` table (non-NULL)
 * @ev:  Decoded mouse event with 0-based `(mx, my)` cell coordinates and `mbtn` / `mrelease`
 *
 * Role:
 *   Updates hover coordinates `(app->mouse_x, app->mouse_y)` for tooltip rendering, resizes split
 *   pane dividers (`left_w`, `schem_h_pct`) during left-drag gestures, routes scroll-wheel ticks
 *   to whichever pane or modal lies beneath the pointer, and resolves left-button clicks against
 *   `app->hits[]` in front-to-back z-order.
 *
 * Rationale:
 *   Routing wheel scrolling by pointer position (`ev->mx < app->left_w` vs. Schematic vs.
 *   Inspector) lets users scroll any pane without clicking to change keyboard focus first.
 *   Scanning `app->hits[]` from `n_hits - 1` down to `0` ensures floating modal buttons and rows
 *   intercept clicks before underlying panes, and clicking outside modal buttons while a modal is
 *   open never triggers background actions.
 */
static void handle_mouse_event(cs_tui_app_t *app, const cs_input_ev_t *ev)
{
    app->mouse_x = ev->mx;
    app->mouse_y = ev->my;

    /* Pure pointer motion only updates hover tooltip coordinates */
    if (ev->mbtn == CS_MOUSE_MOVE)
    {
        return;
    }
    if (ev->mrelease)
    {
        app->drag_split = 0;
        return;
    }
    /* Active pane-splitter drag: clamp widths/percentages so no pane collapses below usability */
    if (ev->mbtn == CS_MOUSE_DRAG_LEFT && app->drag_split != 0)
    {
        if (app->drag_split == 1)
        {
            int nw = ev->mx + 1;
            if (nw >= 26 && nw <= app->scr.cols - 42)
            {
                app->left_w = nw;
            }
        }
        else if (app->drag_split == 2)
        {
            int bh  = app->scr.rows - 4;
            int pct = ((ev->my - 2) * 100) / (bh > 1 ? bh : 1);
            if (pct >= 28 && pct <= 72)
            {
                app->schem_h_pct = pct;
            }
        }
        return;
    }

    /* Scroll wheel: dispatch contextually to active modal, view, or pane under cursor */
    if (ev->mbtn == CS_MOUSE_WHEEL_UP || ev->mbtn == CS_MOUSE_WHEEL_DOWN)
    {
        int d = (ev->mbtn == CS_MOUSE_WHEEL_UP) ? -1 : 1;
        if (app->modal != CS_MODAL_NONE)
        {
            cs_input_ev_t kev = { .key = (d < 0) ? CS_KEY_UP : CS_KEY_DOWN };
            handle_modal_event(app, &kev);
            return;
        }
        if (app->view == CACAOSIM_TUI_VIEW_SOURCE)
        {
            app->source_scroll += d * 3;
            return;
        }
        if (app->view == CACAOSIM_TUI_VIEW_SCHEMATIC)
        {
            app->schem_scroll_y = (app->schem_scroll_y + d * 2 > 0) ? (app->schem_scroll_y + d * 2)
                                                                    : 0;
            return;
        }
        if (app->view == CACAOSIM_TUI_VIEW_BUILDER)
        {
            if (ev->mx < app->left_w)
            {
                int ni = app->sel_item + d;
                if (ni >= 0 && ni <= (int) app->cfg.n_nodes)
                {
                    app->sel_item = ni;
                }
            }
            else
            {
                int sh = ((app->scr.rows - 4) * app->schem_h_pct) / 100;
                if (ev->my < 2 + sh)
                {
                    int ny              = app->schem_scroll_y + d * 2;
                    app->schem_scroll_y = (ny > 0) ? ny : 0;
                }
                else if (app->n_params > 0)
                {
                    int np = app->param_sel + d;
                    if (np >= 0 && np < app->n_params)
                    {
                        app->param_sel = np;
                    }
                }
            }
        }
        return;
    }

    if (ev->mbtn != CS_MOUSE_LEFT)
    {
        return;
    }

    /* Hit-test from frontmost (latest drawn) to backmost (earliest drawn) */
    for (int i = app->n_hits - 1; i >= 0; i--)
    {
        const cs_hit_t *h = &app->hits[i];
        if (ev->mx < h->x || ev->mx >= h->x + h->w || ev->my < h->y || ev->my >= h->y + h->h)
        {
            continue;
        }

        if (app->modal != CS_MODAL_NONE)
        {
            if (h->kind == CS_HIT_MODAL_ROW)
            {
                if (app->modal == CS_MODAL_OPTIONS)
                {
                    app->opt_sel = h->arg;
                }
                else
                {
                    app->modal_sel = h->arg;
                    cs_input_ev_t kev = { .key = CS_KEY_ENTER };
                    handle_modal_event(app, &kev);
                }
            }
            else if (h->kind == CS_HIT_MODAL_OK)
            {
                cs_input_ev_t kev = { .key = CS_KEY_ENTER };
                handle_modal_event(app, &kev);
            }
            else if (h->kind == CS_HIT_MODAL_CANCEL)
            {
                app->modal = CS_MODAL_NONE;
            }
            else if (h->kind == CS_HIT_MODAL_EXTRA)
            {
                cs_input_ev_t kev = { .key = CS_KEY_TAB };
                handle_modal_event(app, &kev);
            }
            return;
        }

        switch (h->kind)
        {
        case CS_HIT_VIEW_TAB:
            app->editing = 0;
            app->view    = h->arg;
            break;
        case CS_HIT_HELP_BTN:
            app->modal = CS_MODAL_HELP;
            break;
        case CS_HIT_ACT_OPTREF:
            app->opt_search[0] = '\0';
            app->opt_scroll    = 0;
            app->opt_sel       = 0;
            app->modal         = CS_MODAL_OPTIONS;
            break;
        case CS_HIT_ACT_ADVANCED:
            app->editing       = 0;
            app->show_advanced = !app->show_advanced;
            app->param_sel     = 0;
            app->param_scroll  = 0;
            cs_tui_log(app, 0, "Inspector Advanced options %s",
                       app->show_advanced ? "SHOWN" : "HIDDEN");
            break;
        case CS_HIT_ELEM_ROW:
            app->editing  = 0;
            app->sel_item = h->arg;
            app->focus    = CS_FOCUS_ELEMS;
            break;
        case CS_HIT_SCHEM_NODE:
            app->editing  = 0;
            app->sel_item = h->arg;
            app->focus    = CS_FOCUS_SCHEM;
            break;
        case CS_HIT_SUBTAB:
            app->editing   = 0;
            app->subtab    = h->arg;
            app->param_sel = 0;
            app->focus     = CS_FOCUS_PARAMS;
            break;
        case CS_HIT_PARAM_ROW:
            /* First click selects parameter row; clicking an already-selected row edits/steps it */
            app->focus = CS_FOCUS_PARAMS;
            if (app->param_sel == h->arg && !app->editing)
            {
                start_param_edit(app);
            }
            else
            {
                app->editing   = 0;
                app->param_sel = h->arg;
            }
            break;
        case CS_HIT_ACT_ADD:
            action_open_add_modal(app);
            break;
        case CS_HIT_ACT_CLONE:
            action_clone_selected(app);
            break;
        case CS_HIT_ACT_UP:
            action_move_selected(app, -1);
            break;
        case CS_HIT_ACT_DOWN:
            action_move_selected(app, 1);
            break;
        case CS_HIT_ACT_SWAP:
            action_swap_arms(app);
            break;
        case CS_HIT_ACT_WIRE:
            action_open_wire_modal(app);
            break;
        case CS_HIT_ACT_DEL:
            action_delete_selected(app);
            break;
        case CS_HIT_ACT_PRESET:
            app->modal     = CS_MODAL_PRESET;
            app->modal_sel = 0;
            break;
        case CS_HIT_ACT_SAVE:
            trigger_save(app, 0);
            break;
        case CS_HIT_ACT_DEPLOY_F:
            cs_tui_exec_deploy(app, "fused");
            break;
        case CS_HIT_ACT_DEPLOY_M:
            cs_tui_exec_deploy(app, "modular");
            break;
        case CS_HIT_ACT_STOP:
            cs_tui_exec_stop(app);
            break;
        case CS_HIT_ACT_MKPUP:
            cs_tui_exec_mkpupil(app);
            break;
        case CS_HIT_ACT_ATMTURB:
            app->opts.with_atmturb = !app->opts.with_atmturb;
            cs_tui_log(app, 0, "AtmTurb %s for next deployment",
                       app->opts.with_atmturb ? "ENABLED" : "DISABLED");
            break;
        case CS_HIT_THEME_BTN:
            app->modal     = CS_MODAL_THEME;
            app->modal_sel = app->opts.theme;
            break;
        case CS_HIT_TOOLTIP_BTN:
            app->show_tooltips = !app->show_tooltips;
            break;
        case CS_HIT_EXIT_BTN:
            if (app->dirty)
            {
                app->confirm_action = 1;
                app->modal          = CS_MODAL_CONFIRM;
            }
            else
            {
                app->running = 0;
            }
            break;
        case CS_HIT_SPLIT_LEFT:
            app->drag_split = 1;
            break;
        case CS_HIT_SPLIT_SCHEM:
            app->drag_split = 2;
            break;
        default:
            break;
        }
        return;
    }
}

/* ── Main Interactive TUI Entrypoint & Headless Snapshot ──────────────────── */

/**
 * cacaosim_tui_run() - Run the interactive `milk-CTRL`-style AO hardware builder and status TUI
 * @bench_path: Optional `.bench` file path to load or create (NULL / `""` starts from the two-stage
 *              built-in template; non-existent path starts from single-PyWFS marked modified)
 * @opts:       Optional TUI launch settings (NULL uses `cacaosim_tui_opts_defaults()`)
 *
 * Role:
 *   Allocates `cs_tui_app_t`, loads or seeds the initial `.bench` configuration, enters raw
 *   alternate-screen TTY mode with SGR mouse and bracketed paste enabled, runs the `poll`-paced
 *   event loop (`120 ms` idle refresh for SHM telemetry, external `.bench` `mtime` hot-reload, and
 *   heartbeat animation; zero-latency wakeup on keystrokes/mouse), and guarantees clean terminal
 *   teardown on exit.
 *
 * Rationale:
 *   - Heap-allocating `cs_tui_app_t` avoids placing the 24-snapshot `cs_undo_ring_t` + stream
 *     tables on thread stacks.
 *   - A `120 ms` (`~8.3 Hz`) `poll` timeout strikes an ergonomic balance: `MILK_SHM_DIR` frame
 *     counters, `Hz` rates, Strehl ratios, and external file reloads update smoothly while idle
 *     SSH bandwidth drops near zero thanks to cell-diff flushing in `cs_scr_flush()`.
 *   - Intercepting `CS_HIT_ACT_EDITOR` clicks directly in the top-level loop (where `tty_fd` lives)
 *     lets `trigger_external_editor()` cleanly close and reopen the controlling terminal fd.
 *
 * Return: 0 on normal user exit, -1 on allocation or terminal screen-buffer initialization error.
 */
int cacaosim_tui_run(const char *bench_path, const cacaosim_tui_opts_t *opts)
{
    cs_tui_app_t *app = (cs_tui_app_t *) calloc(1u, sizeof(cs_tui_app_t));
    if (app == NULL)
    {
        return -1;
    }

    if (opts != NULL)
    {
        app->opts = *opts;
    }
    else
    {
        cacaosim_tui_opts_defaults(&app->opts);
    }

    app->view          = app->opts.initial_view;
    app->focus         = CS_FOCUS_ELEMS;
    app->left_w        = 38;
    app->schem_h_pct   = 48;
    app->show_tooltips = 1;
    app->mouse_x       = -1;
    app->mouse_y       = -1;
    app->running       = 1;
    app->ana_dirty     = 1;

    /* Load existing .bench file, prepare a new target path, or seed default two-stage preset */
    if (bench_path != NULL && bench_path[0] != '\0')
    {
        snprintf(app->file_path, sizeof(app->file_path), "%s", bench_path);
        char ebuf[160] = { 0 };
        if (access(bench_path, R_OK) == 0)
        {
            if (cacaosim_bench_load_file(bench_path, &app->cfg, ebuf, sizeof(ebuf)) == 0)
            {
                app->disk_mtime_ns = stat_mtime_ns(bench_path);
                cs_tui_log(app, 1, "Loaded '%s' (%u optical elements)", bench_path,
                           app->cfg.n_nodes);
            }
            else
            {
                cacaosim_bench_preset(&app->cfg, CACAOSIM_PRESET_SINGLE_PYRWFS);
                cs_tui_log(app, 3, "Parse error in '%s': %s (loaded default template)", bench_path,
                           ebuf);
            }
        }
        else
        {
            cacaosim_bench_preset(&app->cfg, CACAOSIM_PRESET_SINGLE_PYRWFS);
            app->dirty = 1;
            cs_tui_log(app, 0, "New file '%s' -- press 'P' for presets or 's' to save", bench_path);
        }
    }
    else
    {
        cacaosim_bench_preset(&app->cfg, CACAOSIM_PRESET_TWO_STAGE);
        cs_tui_log(app, 0, "cacaosim-CTRL ready -- press 'a' to add optics, 'P' for presets");
    }

    int cols = 120, rows = 38;
    int color_lvl = cs_tui_detect_color_level(app->opts.color_level);
    int ascii_m   = cs_tui_detect_ascii(app->opts.ascii_only);
    int tty_fd    = cs_tui_term_enter(&cols, &rows);
    if (cs_scr_init(&app->scr, cols, rows, color_lvl, ascii_m) != 0)
    {
        cs_tui_term_exit();
        free(app);
        return -1;
    }

    cs_shm_scan(&app->shm, &app->cfg, 1);

    while (app->running)
    {
        /* Auto-reload clean file if modified externally (never overwrites unsaved TUI edits) */
        if (!app->dirty && app->file_path[0] != '\0')
        {
            uint64_t mt = stat_mtime_ns(app->file_path);
            if (mt != 0u && app->disk_mtime_ns != 0u && mt != app->disk_mtime_ns)
            {
                cacaosim_bench_cfg_t reloaded;
                char                 ebuf[160] = { 0 };
                if (cacaosim_bench_load_file(app->file_path, &reloaded, ebuf, sizeof(ebuf)) == 0)
                {
                    app->cfg       = reloaded;
                    app->ana_dirty = 1;
                    cs_tui_log(app, 1, "Auto-reloaded '%s' from disk", app->file_path);
                }
                app->disk_mtime_ns = mt;
            }
        }

        /* Drain redirected library stdout/stderr into activity log so raw TTY stays uncorrupted */
        cs_tui_drain_captured_stdio(app);
        cs_shm_scan(&app->shm, &app->cfg, 0);
        cs_tui_render_frame(app);
        cs_scr_flush(&app->scr, tty_fd);

        cs_input_ev_t ev;
        if (!cs_tui_poll_input(&ev, 120))
        {
            app->frame_tick++;
            continue;
        }

        if (ev.key == CS_KEY_RESIZE)
        {
            cs_tui_term_query_size(&cols, &rows);
            cs_scr_resize(&app->scr, cols, rows);
            continue;
        }
        if (ev.key == 3) /* Ctrl+C */
        {
            break;
        }
        if (ev.key == CS_KEY_MOUSE)
        {
            /* Intercept [e:Open in $EDITOR] button clicks here where tty_fd can be updated */
            if (ev.mbtn == CS_MOUSE_LEFT)
            {
                for (int hi = app->n_hits - 1; hi >= 0; hi--)
                {
                    const cs_hit_t *h = &app->hits[hi];
                    if (h->kind == CS_HIT_ACT_EDITOR && ev.mx >= h->x && ev.mx < h->x + h->w &&
                        ev.my >= h->y && ev.my < h->y + h->h)
                    {
                        trigger_external_editor(app, &tty_fd);
                        ev.key = CS_KEY_NONE;
                        break;
                    }
                }
            }
            if (ev.key == CS_KEY_MOUSE)
            {
                handle_mouse_event(app, &ev);
            }
            continue;
        }

        if (app->modal != CS_MODAL_NONE)
        {
            handle_modal_event(app, &ev);
            continue;
        }

        /* Inline Parameter Editor Mode */
        if (app->editing)
        {
            if (ev.key == CS_KEY_ESC)
            {
                app->editing = 0;
            }
            else if (ev.key == CS_KEY_ENTER)
            {
                const cs_param_desc_t *pd = &app->params[app->param_sel];
                char                   ebuf[160] = { 0 };
                push_undo(app);
                if (cs_param_set_str(&app->cfg, app->sel_item, pd, app->edit_buf, ebuf,
                                     sizeof(ebuf)) == 0)
                {
                    app->dirty     = 1;
                    app->ana_dirty = 1;
                    cs_tui_log(app, 1, "Set %s = %s", pd->key, app->edit_buf);
                }
                else
                {
                    /* Roll back speculative undo snapshot if string validation/range check fails */
                    app->undo.pos--;
                    app->undo.count--;
                    cs_tui_log(app, 3, "%s", ebuf);
                }
                app->editing = 0;
            }
            else if (ev.key == CS_KEY_BACKSPACE)
            {
                size_t l = strlen(app->edit_buf);
                if (l > 0u)
                {
                    app->edit_buf[l - 1u] = '\0';
                }
            }
            else if (ev.key == CS_KEY_PASTE)
            {
                size_t l = strlen(app->edit_buf);
                snprintf(app->edit_buf + l, sizeof(app->edit_buf) - l, "%s", ev.paste);
            }
            else if (ev.key >= 32 && ev.key < 127)
            {
                size_t l = strlen(app->edit_buf);
                if (l + 1u < sizeof(app->edit_buf))
                {
                    app->edit_buf[l]      = (char) ev.key;
                    app->edit_buf[l + 1u] = '\0';
                }
            }
            continue;
        }

        /* Global Workspace Keys (active across all tabs when not in modal/inline-edit mode) */
        if (ev.key == 'q' || ev.key == 'Q')
        {
            if (app->dirty)
            {
                app->confirm_action = 1;
                app->modal          = CS_MODAL_CONFIRM;
            }
            else
            {
                app->running = 0;
            }
            continue;
        }
        if (ev.key == CS_KEY_F1 || ev.key == 'h')
        {
            app->modal = CS_MODAL_HELP;
            continue;
        }
        if (ev.key == '?')
        {
            app->opt_search[0] = '\0';
            app->opt_scroll    = 0;
            app->opt_sel       = 0;
            app->modal         = CS_MODAL_OPTIONS;
            continue;
        }
        if (ev.key == 'v')
        {
            app->show_advanced = !app->show_advanced;
            app->param_sel     = 0;
            app->param_scroll  = 0;
            cs_tui_log(app, 0, "Inspector Advanced options %s",
                       app->show_advanced ? "SHOWN" : "HIDDEN");
            continue;
        }
        if (ev.key >= CS_KEY_F2 && ev.key <= CS_KEY_F6)
        {
            app->view = ev.key - CS_KEY_F2;
            continue;
        }
        if (ev.key == CS_KEY_TAB)
        {
            app->focus = (cs_focus_pane_t) (((int) app->focus + 1) % 3);
            continue;
        }
        if (ev.key == CS_KEY_BTAB)
        {
            app->focus = (cs_focus_pane_t) (((int) app->focus + 2) % 3);
            continue;
        }
        if (ev.key == '1' || ev.key == '2' || ev.key == '3')
        {
            app->subtab    = ev.key - '1';
            app->param_sel = 0;
            app->focus     = CS_FOCUS_PARAMS;
            continue;
        }
        if (ev.key == 'a' || ev.key == '+')
        {
            action_open_add_modal(app);
            continue;
        }
        if (ev.key == 'c')
        {
            action_clone_selected(app);
            continue;
        }
        if (ev.key == 'd' || ev.key == CS_KEY_DEL)
        {
            action_delete_selected(app);
            continue;
        }
        if (ev.key == 'K' || ev.key == CS_KEY_SHIFT_UP)
        {
            action_move_selected(app, -1);
            continue;
        }
        if (ev.key == 'J' || ev.key == CS_KEY_SHIFT_DOWN)
        {
            action_move_selected(app, 1);
            continue;
        }
        if (ev.key == 'x')
        {
            action_swap_arms(app);
            continue;
        }
        if (ev.key == 'w')
        {
            action_open_wire_modal(app);
            continue;
        }
        if (ev.key == 'u')
        {
            do_undo(app);
            continue;
        }
        if (ev.key == 18 || ev.key == 'U') /* Ctrl+R or U */
        {
            do_redo(app);
            continue;
        }
        if (ev.key == 'P')
        {
            app->modal     = CS_MODAL_PRESET;
            app->modal_sel = 0;
            continue;
        }
        if (ev.key == 's' || ev.key == 19) /* s or Ctrl+S */
        {
            trigger_save(app, 0);
            continue;
        }
        if (ev.key == 'A')
        {
            trigger_save(app, 1);
            continue;
        }
        if (ev.key == 'o')
        {
            app->modal = CS_MODAL_OPEN;
            snprintf(app->modal_input, sizeof(app->modal_input), "%s", app->file_path);
            continue;
        }
        if (ev.key == 'e')
        {
            trigger_external_editor(app, &tty_fd);
            continue;
        }
        if (ev.key == 'f')
        {
            cs_tui_exec_deploy(app, "fused");
            continue;
        }
        if (ev.key == 'M')
        {
            cs_tui_exec_deploy(app, "modular");
            continue;
        }
        if (ev.key == 'S')
        {
            cs_tui_exec_stop(app);
            continue;
        }
        if (ev.key == 'g')
        {
            cs_tui_exec_mkpupil(app);
            continue;
        }
        if (ev.key == 't')
        {
            app->opts.with_atmturb = !app->opts.with_atmturb;
            cs_tui_log(app, 0, "AtmTurb %s for next deployment",
                       app->opts.with_atmturb ? "ENABLED" : "DISABLED");
            continue;
        }
        if (ev.key == 'T')
        {
            app->modal     = CS_MODAL_THEME;
            app->modal_sel = app->opts.theme;
            continue;
        }
        if (ev.key == 'm')
        {
            app->show_tooltips = !app->show_tooltips;
            continue;
        }
        if (ev.key == 12) /* Ctrl+L */
        {
            app->scr.force_full = 1;
            continue;
        }

        /* Pane-Sensitive Navigation & Stepping */
        if (app->view == CACAOSIM_TUI_VIEW_SOURCE)
        {
            if (ev.key == CS_KEY_UP || ev.key == 'k')
            {
                app->source_scroll--;
            }
            else if (ev.key == CS_KEY_DOWN || ev.key == 'j')
            {
                app->source_scroll++;
            }
            else if (ev.key == CS_KEY_PGUP)
            {
                app->source_scroll -= 15;
            }
            else if (ev.key == CS_KEY_PGDN)
            {
                app->source_scroll += 15;
            }
            continue;
        }

        if (app->view != CACAOSIM_TUI_VIEW_BUILDER)
        {
            if (ev.key == CS_KEY_UP || ev.key == 'k')
            {
                if (app->sel_item > 0)
                {
                    app->sel_item--;
                }
            }
            else if (ev.key == CS_KEY_DOWN || ev.key == 'j')
            {
                if (app->sel_item < (int) app->cfg.n_nodes)
                {
                    app->sel_item++;
                }
            }
            else if (ev.key == CS_KEY_LEFT && app->view == CACAOSIM_TUI_VIEW_SCHEMATIC)
            {
                if (app->schem_scroll_x >= 4)
                {
                    app->schem_scroll_x -= 4;
                }
            }
            else if (ev.key == CS_KEY_RIGHT && app->view == CACAOSIM_TUI_VIEW_SCHEMATIC)
            {
                app->schem_scroll_x += 4;
            }
            else if (ev.key == CS_KEY_ENTER)
            {
                app->view  = CACAOSIM_TUI_VIEW_BUILDER;
                app->focus = CS_FOCUS_PARAMS;
            }
            continue;
        }

        if (app->focus == CS_FOCUS_ELEMS)
        {
            if ((ev.key == CS_KEY_UP || ev.key == 'k') && app->sel_item > 0)
            {
                app->sel_item--;
                app->param_sel = 0;
            }
            else if ((ev.key == CS_KEY_DOWN || ev.key == 'j') &&
                     app->sel_item < (int) app->cfg.n_nodes)
            {
                app->sel_item++;
                app->param_sel = 0;
            }
            else if (ev.key == CS_KEY_ENTER || ev.key == CS_KEY_RIGHT)
            {
                app->focus = CS_FOCUS_PARAMS;
            }
        }
        else if (app->focus == CS_FOCUS_SCHEM)
        {
            if (ev.key == CS_KEY_UP && app->schem_scroll_y > 0)
            {
                app->schem_scroll_y--;
            }
            else if (ev.key == CS_KEY_DOWN)
            {
                app->schem_scroll_y++;
            }
            else if (ev.key == CS_KEY_LEFT && app->schem_scroll_x >= 3)
            {
                app->schem_scroll_x -= 3;
            }
            else if (ev.key == CS_KEY_RIGHT)
            {
                app->schem_scroll_x += 3;
            }
            else if (ev.key == CS_KEY_ENTER)
            {
                app->focus = CS_FOCUS_PARAMS;
            }
        }
        else if (app->focus == CS_FOCUS_PARAMS)
        {
            if ((ev.key == CS_KEY_UP || ev.key == 'k') && app->param_sel > 0)
            {
                app->param_sel--;
            }
            else if ((ev.key == CS_KEY_DOWN || ev.key == 'j') &&
                     app->param_sel + 1 < app->n_params)
            {
                app->param_sel++;
            }
            else if (ev.key == CS_KEY_ENTER || ev.key == ' ')
            {
                start_param_edit(app);
            }
            else if (ev.key == CS_KEY_LEFT || ev.key == CS_KEY_RIGHT || ev.key == '-' ||
                     ev.key == '=' || ev.key == '[' || ev.key == ']')
            {
                /* Fine step (Left/Right/-/=) vs. 5x coarse step ([/]) with rollback on error */
                if (app->param_sel >= 0 && app->param_sel < app->n_params)
                {
                    int dir = (ev.key == CS_KEY_LEFT || ev.key == '-' || ev.key == '[') ? -1 : 1;
                    int crs = (ev.key == '[' || ev.key == ']') ? 1 : 0;
                    char ebuf[128] = { 0 };
                    push_undo(app);
                    if (cs_param_step(&app->cfg, app->sel_item, &app->params[app->param_sel], dir,
                                      crs, ebuf, sizeof(ebuf)) == 0)
                    {
                        app->dirty     = 1;
                        app->ana_dirty = 1;
                    }
                    else
                    {
                        app->undo.pos--;
                        app->undo.count--;
                    }
                }
            }
        }
    }

    cs_tui_term_exit();
    cs_scr_free(&app->scr);
    free(app);
    return 0;
}

/**
 * cacaosim_tui_render_headless() - Render a single TUI workspace view to a plain-text buffer
 * @cfg:        Input bench configuration to visualize (non-NULL)
 * @view_id:    Target workspace tab (`cacaosim_tui_view_t`, clamped to `F2:BUILDER` if out of
 *              range)
 * @cols:       Virtual terminal width in columns
 * @rows:       Virtual terminal height in rows
 * @ascii_only: Non-zero to restrict borders/arrows/spectra to 7-bit ASCII; 0 for UTF-8 glyphs
 * @out_buf:    Destination NUL-terminated plain-text snapshot buffer (`'\n'`-joined rows,
 *              non-NULL)
 * @out_cap:    Capacity of @out_buf in bytes (`> 0`)
 *
 * Role:
 *   Executes the exact same `cs_tui_render_frame()` layout, zero-FFT photometry, optical scale,
 *   and schematic compositor pipeline used by the live interactive GUI, but dumps the virtual
 *   backbuffer cells to plain text (`cs_scr_dump_plain()`) without touching `/dev/tty`, `termios`,
 *   or ANSI escape streams.
 *
 * Rationale:
 *   Terminal UIs are notoriously fragile to test in headless CI runners that lack a controlling
 *   TTY. Exposing a deterministic, pure-memory snapshot path allows `ctest` suites to assert on
 *   rendered node chips, beam routing arrows, parameter rows, and optical scale strings across all
 *   five views (`F2`..`F6`) and arbitrary terminal dimensions.
 *
 * Return: Number of characters written to @out_buf (excluding NUL), or -1 on invalid arguments or
 *         allocation failure.
 */
int cacaosim_tui_render_headless(
    const cacaosim_bench_cfg_t *cfg,
    int                         view_id,
    int                         cols,
    int                         rows,
    int                         ascii_only,
    char                       *out_buf,
    size_t                      out_cap)
{
    if (cfg == NULL || out_buf == NULL || out_cap == 0u)
    {
        return -1;
    }
    cs_tui_app_t *app = (cs_tui_app_t *) calloc(1u, sizeof(cs_tui_app_t));
    if (app == NULL)
    {
        return -1;
    }
    cacaosim_tui_opts_defaults(&app->opts);
    app->opts.ascii_only = ascii_only;
    app->cfg             = *cfg;
    app->view            = (view_id >= 0 && view_id < CACAOSIM_TUI_VIEW_COUNT)
                               ? view_id
                               : CACAOSIM_TUI_VIEW_BUILDER;
    app->left_w          = 38;
    app->schem_h_pct     = 48;
    app->ana_dirty       = 1;
    snprintf(app->file_path, sizeof(app->file_path), "headless.bench");

    if (cs_scr_init(&app->scr, cols, rows, 0, ascii_only) != 0)
    {
        free(app);
        return -1;
    }
    cs_tui_render_frame(app);
    int n = cs_scr_dump_plain(&app->scr, out_buf, out_cap);
    cs_scr_free(&app->scr);
    free(app);
    return n;
}
