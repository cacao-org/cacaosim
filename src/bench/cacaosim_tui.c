// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui.c
 * @brief   Main event loop, undo/redo history, mouse/key dispatch, and headless TUI API
 */

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "bench/cacaosim_tui_internal.h"
#include "pupil/cacaosim_pupil.h"

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
    opts->seeing_arcsec = 0.7f;
}

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

void cs_tui_refresh_analysis(cs_tui_app_t *app)
{
    if (app == NULL || !app->ana_dirty)
    {
        return;
    }
    cacaosim_pupil_cfg_t pcfg = {
        .nwf        = 16u,
        .pup_diam   = app->cfg.pup_diam,
        .cobs       = app->cfg.cobs,
        .nspider    = app->cfg.nspider,
        .spider_w   = app->cfg.spider_w,
        .spider_deg = app->cfg.spider_deg,
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

static void push_undo(cs_tui_app_t *app)
{
    cs_undo_ring_t *u = &app->undo;
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

static void do_undo(cs_tui_app_t *app)
{
    cs_undo_ring_t *u = &app->undo;
    if (u->pos <= 0)
    {
        cs_tui_log(app, 2, "Nothing to undo");
        return;
    }
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

    cs_tui_term_exit();
    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "%.256s \"%.500s\"", ed, target);
    int rc = system(cmd);
    (void) rc;

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

static void action_open_add_modal(cs_tui_app_t *app)
{
    app->modal          = CS_MODAL_ADD;
    app->modal_sel      = 0;
    app->modal_beam_idx = -1;
}

static void action_commit_add(cs_tui_app_t *app, int choice)
{
    static const cacaosim_node_kind_t kinds[8] = {
        CACAOSIM_NODE_DM,     CACAOSIM_NODE_DM,     CACAOSIM_NODE_SPLITTER, CACAOSIM_NODE_CORO,
        CACAOSIM_NODE_PYRWFS, CACAOSIM_NODE_SHWFS,  CACAOSIM_NODE_ZWFS,     CACAOSIM_NODE_FPCAM,
    };
    if (choice < 0 || choice >= 8)
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

static void handle_modal_event(cs_tui_app_t *app, const cs_input_ev_t *ev)
{
    if (ev->key == CS_KEY_ESC || (ev->key == CS_KEY_MOUSE && ev->mbtn == CS_MOUSE_RIGHT))
    {
        app->modal = CS_MODAL_NONE;
        return;
    }

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
        if (ev->key == 'h' || ev->key == 'H' || ev->key == 'q' || ev->key == '?' ||
            ev->key == CS_KEY_F1 || ev->key == CS_KEY_ENTER)
        {
            app->modal = CS_MODAL_NONE;
        }
        return;
    }

    int max_items = 1;
    if (app->modal == CS_MODAL_ADD)
    {
        max_items = 8;
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
    else if (ev->key >= '1' && ev->key <= '8' && (ev->key - '1') < max_items)
    {
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

static void handle_mouse_event(cs_tui_app_t *app, const cs_input_ev_t *ev)
{
    app->mouse_x = ev->mx;
    app->mouse_y = ev->my;

    if (ev->mbtn == CS_MOUSE_MOVE)
    {
        return;
    }
    if (ev->mrelease)
    {
        app->drag_split = 0;
        return;
    }
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

    /* Hit-test from frontmost to backmost */
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
                app->modal_sel = h->arg;
                cs_input_ev_t kev = { .key = CS_KEY_ENTER };
                handle_modal_event(app, &kev);
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
        /* Auto-reload clean file if modified externally */
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

        /* Global Workspace Keys */
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
        if (ev.key == CS_KEY_F1 || ev.key == 'h' || ev.key == '?')
        {
            app->modal = CS_MODAL_HELP;
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
