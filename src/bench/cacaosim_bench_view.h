// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_view.h
 * @brief   Public API for one-shot ANSI/UTF-8/ASCII optical bench schematic rendering and live SHM
 *          stream status probing
 *
 * Architectural Role in cacaosim:
 *   Declares the non-interactive, stream-oriented visualization interface for declarative .bench
 *   multi-stage Adaptive Optics (AO) configurations in libmilkcacaosim.so. Given an analyzed or
 *   initialized cacaosim_bench_ctx_t (or a .bench file path on disk), this module renders:
 *     - A 16x16 anti-aliased 2D telescope pupil thumbnail, physical pupil/focal plate scales,
 *       master clock cadence, stellar blackbody photometry, and 32-bin (0.40..2.40 um) source
 *       spectrum bar chart.
 *     - A box-drawn 2D branching optical topology schematic tracing wavefronts through inline
 *       DMs, TTMs, and static WFE maps, splitting across gray/dichroic beamsplitters and
 *       coronagraphic focal/Lyot pickoffs, and terminating at multi-rate WFS and science cameras
 *       with per-arm spectral sparklines and 2D registration badges.
 *     - Hardware & photon-budget summary tables (Active Mirrors, WFS/Science Cameras, Auxiliary
 *       Optics) and structured syntax/DAG error diagnostic panels.
 *     - Live MILK_SHM_DIR ImageStreamIO (.im.shm) stream presence badges overlaid onto configured
 *       inputs, DM commands, coronagraph masks, and camera outputs.
 *   Shared color theme identifiers (cacaosim_view_theme_t) are also consumed by the interactive
 *   cacaosim-CTRL TUI (src/bench/cacaosim_tui.h) so CLI snapshots and interactive sessions share
 *   identical visual semantics.
 *
 * Design Rationale & Motivation:
 *   1. Why a stream-oriented renderer complements the full-screen cacaosim-CTRL TUI:
 *      Full-screen TUIs seize the terminal alternate screen buffer, require raw TTY input, and
 *      disappear when exited. Operators, automated launch scripts (cacaosim-bench start/status),
 *      SSH sessions, tmux scrollback buffers, CI verification logs, and piped pagers (less -R)
 *      need a stateless FILE* renderer that emits clean, self-contained schematics inline without
 *      requiring an interactive terminal or mutating terminal state.
 *   2. Zero-FFT sub-millisecond preview:
 *      Pairing cacaosim_bench_analyze() with cacaosim_bench_render() validates DAG topology and
 *      propagates 32-bin stellar photon budgets in < 1 ms without allocating FFTW/cuFFT plans,
 *      CUDA buffers, or shared-memory segments.
 *   3. Non-intrusive live SHM inspection:
 *      Live stream readiness is queried via lightweight filesystem existence checks in
 *      MILK_SHM_DIR rather than mapping or locking SHM segments, ensuring zero perturbation to
 *      real-time CACAO control loops while complementary tools (such as cacaosim-CTRL live SHM
 *      discovery) inspect modular benches launched with or without a .bench file.
 */

#ifndef CACAOSIM_BENCH_VIEW_H
#define CACAOSIM_BENCH_VIEW_H

#include <stdio.h>
#include "bench/cacaosim_bench.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * enum cacaosim_view_theme_t - Semantic color palette themes matching milk's overview TUI themes
 * @CACAOSIM_THEME_DARK:       Default milk dark slate / cyan / blue / gold palette for standard
 *                             dark-background developer and operator terminals.
 * @CACAOSIM_THEME_NIGHT:      Observatory Red scotopic-vision-safe palette restricting green/blue
 *                             channel emission for control-room night operations.
 * @CACAOSIM_THEME_ACCESSIBLE: Okabe-Ito colorblind-safe high-contrast palette engineered to remain
 *                             distinguishable under deuteranopia, protanopia, and tritanopia.
 * @CACAOSIM_THEME_LIGHT:      Paper Light high-contrast dark-ink palette for daylight terminals
 *                             and white-background documentation exports.
 * @CACAOSIM_THEME_NORDIC:     Cool Nordic frost / aurora low-glare palette for long monitoring
 *                             sessions.
 * @CACAOSIM_THEME_COUNT:      Total number of built-in semantic color palettes (sentinel bound).
 *
 * Role: Selects the unified 15-slot RGB palette applied to panel borders, status badges, SHM
 * stream tags, optical element pills, and wavelength-dependent spectral bar charts.
 *
 * Rationale: Keeping theme indices identical between cacaosim_bench_view, cacaosim-CTRL, and
 * milk's overview TUI ensures visual continuity across one-shot CLI diagnostics and interactive
 * control panels while accommodating real-world observatory lighting and accessibility needs.
 */
typedef enum
{
    CACAOSIM_THEME_DARK       = 0, /**< Default milk dark slate / cyan / blue / gold palette */
    CACAOSIM_THEME_NIGHT      = 1, /**< Observatory Red night-vision-safe palette */
    CACAOSIM_THEME_ACCESSIBLE = 2, /**< Okabe-Ito colorblind-safe high-contrast palette */
    CACAOSIM_THEME_LIGHT      = 3, /**< Paper Light palette for light terminal backgrounds */
    CACAOSIM_THEME_NORDIC     = 4, /**< Cool Nordic frost / aurora palette */
    CACAOSIM_THEME_COUNT      = 5  /**< Number of registered visualization color palettes */
} cacaosim_view_theme_t;

/**
 * struct cacaosim_view_opts_t - Rendering configuration for cacaosim_bench_render() and helpers
 * @color_level: ANSI color capability override: -1 = auto-detect from destination FD isatty(),
 *               NO_COLOR, TERM, and COLORTERM; 0 = plain monochrome (zero CSI escapes);
 *               1 = 16-color ANSI; 2 = xterm-256 palette; 3 = 24-bit TrueColor RGB.
 * @theme:       Semantic color palette index from enum cacaosim_view_theme_t
 *               (default: CACAOSIM_THEME_DARK; out-of-range values fall back to dark).
 * @ascii_only:  Glyph set selector: 1 forces pure 7-bit ASCII box-drawing (+, -, |), arrows,
 *               and spectrum bars; 0 (default) uses UTF-8 rounded borders, half-blocks, and
 *               8-level vertical block sparklines unless LC_ALL/LC_CTYPE/LANG is "C" or "POSIX".
 * @check_shm:   Live shared-memory probe toggle: 1 (default) checks MILK_SHM_DIR (or /milk/shm)
 *               for live <stream>.im.shm files and renders LIVE/STANDBY status indicators;
 *               0 disables filesystem probing for deterministic offline/CI snapshots.
 * @show_tables: Hardware summary table toggle: 1 (default) appends Panel 3 (Active Mirrors,
 *               WFS & Science Cameras, and Beamsplitters/Coronagraphs/WFE tables); 0 renders
 *               only the overview header and 2D optical path schematic (--compact).
 * @show_footer: Workflow hint footer toggle: 1 (default) appends a one-line CLI quick-reference
 *               footer below the panels; 0 suppresses the footer (used in watch/TUI overlays).
 * @width:       Target outer panel width in terminal columns: 0 (default) auto-detects terminal
 *               width via TIOCGWINSZ and clamps to [98..114] columns (104 when non-TTY);
 *               explicit values >= 76 are clamped to [76..128] columns.
 *
 * Role: Bundles caller-specified formatting, width, glyph, color, and SHM-inspection controls
 * passed into all cacaosim_bench_render*() entry points without requiring global state.
 *
 * Rationale: Passing a value-initialized options struct (with NULL accepted everywhere as a
 * shorthand for cacaosim_view_opts_defaults()) makes every rendering function reentrant,
 * testable against in-memory FILE* streams (open_memstream), and adaptable to both rich
 * interactive terminals and stripped-down log files.
 */
typedef struct
{
    int32_t color_level; /**< -1=auto (TTY+NO_COLOR), 0=mono, 1=16c, 2=256c, 3=truecolor */
    int32_t theme;       /**< Palette index (cacaosim_view_theme_t, default: CACAOSIM_THEME_DARK) */
    int32_t ascii_only;  /**< 1 for pure 7-bit ASCII boxes/bars, 0 for UTF-8 box-drawing & blocks */
    int32_t check_shm;   /**< 1 to probe MILK_SHM_DIR for live .im.shm streams (default: 1) */
    int32_t show_tables; /**< 1 to render Active Mirrors & Sensors hardware tables (default: 1) */
    int32_t show_footer; /**< 1 to render CLI edit/deploy workflow hints footer (default: 1) */
    int32_t width;       /**< Target panel width in columns (0 = auto-detect terminal [98..114]) */
} cacaosim_view_opts_t;

/**
 * cacaosim_view_opts_defaults() - Populate default milk-styled visualization options
 * @opts: Destination options struct to initialize (safe no-op if NULL)
 *
 * Role: Resets all fields of @opts to standard auto-detecting defaults (color_level = -1,
 * CACAOSIM_THEME_DARK, UTF-8 enabled, live MILK_SHM_DIR probing enabled, hardware tables and
 * workflow footer visible, and automatic terminal column width).
 *
 * Rationale: Centralizing default initialization guarantees forward compatibility whenever new
 * rendering options are added and ensures CLI callers, TUI callers, and NULL-opts library calls
 * start from identical baseline behavior.
 */
void cacaosim_view_opts_defaults(cacaosim_view_opts_t *opts);

/**
 * cacaosim_view_theme_name() - Return canonical lowercase identifier for a theme index
 * @theme: Palette index from enum cacaosim_view_theme_t ([0..CACAOSIM_THEME_COUNT-1])
 *
 * Role: Maps an integer theme enum value to its canonical CLI/config string ("dark", "night",
 * "accessible", "light", or "nordic") for status bars, help messages, and diagnostics.
 *
 * Rationale: Clamping out-of-bounds indices safely to "dark" prevents NULL dereferences when
 * formatting user-supplied or cycled theme indices in UI status lines.
 *
 * Return: Pointer to a static NUL-terminated theme name string (never NULL; defaults to "dark").
 */
const char *cacaosim_view_theme_name(int theme);

/**
 * cacaosim_view_theme_from_name() - Parse a human-readable theme name or alias into an enum index
 * @name: Case-insensitive theme name string ("dark", "night", "red", "accessible", "light",
 *        "nordic", or "nord"); NULL-safe
 *
 * Role: Translates CLI arguments (--theme <name>) into a validated cacaosim_view_theme_t index.
 *
 * Rationale: Case-insensitive matching plus intuitive shorthand aliases ("red" -> night,
 * "nord" -> nordic) reduce CLI friction while returning -1 on unknown names so callers can emit
 * actionable error diagnostics.
 *
 * Return: Theme enum index in [0..CACAOSIM_THEME_COUNT-1], or -1 if @name is NULL or unrecognized.
 */
int cacaosim_view_theme_from_name(const char *name);

/**
 * cacaosim_bench_render() - Render an analyzed or initialized optical bench DAG to a FILE stream
 * @ctx:         Analyzed (cacaosim_bench_analyze) or initialized (cacaosim_bench_init) optical
 *               bench context containing sorted nodes, beams, cadences, and 32-bin spectra;
 *               safe no-op if NULL
 * @source_path: Optional .bench file path displayed in Panel 1 metadata and workflow footer hints
 *               (NULL or "" renders "(in-memory .bench)" / "<file.bench>")
 * @opts:        Optional rendering options controlling color depth, theme, UTF-8/ASCII glyphs,
 *               SHM probing, tables, footer, and column width (NULL uses defaults)
 * @fp:          Destination output stream (e.g. stdout, stderr, tmpfile, or open_memstream);
 *               safe no-op if NULL
 *
 * Role: Primary schematic rendering entry point. Emits Panel 1 (overview, 16x16 pupil preview,
 * pupil/focal plate scales, stellar photon budget, 32-bin spectrum), Panel 2 (2D box-drawn
 * optical path & light-splitting tree), optional Panel 3 (DM/TTM, WFS/camera, and splitter/coro/
 * WFE hardware tables), and optional CLI workflow hints sequentially to @fp.
 *
 * Rationale: Accepting an already-analyzed cacaosim_bench_ctx_t decouples rendering from file I/O
 * so in-memory presets, programmatic DAG edits, live running engines, and disk .bench files can
 * all be rendered identically without re-parsing or allocating optical propagation buffers.
 */
void cacaosim_bench_render(
    const cacaosim_bench_ctx_t  *ctx,
    const char                  *source_path,
    const cacaosim_view_opts_t  *opts,
    FILE                        *fp);

/**
 * cacaosim_bench_render_error() - Render a milk-styled .bench configuration error diagnostic panel
 * @source_path: Optional path to the .bench file being parsed or validated (NULL or "" displays
 *               "(in-memory .bench)")
 * @partial_cfg: Optional partially parsed bench configuration dumped to show section wiring up to
 *               the point of failure (pass NULL when file open/read or syntax parsing failed)
 * @err_msg:     Human-readable syntax or topological DAG validation error string (NULL or ""
 *               displays a generic fallback message)
 * @opts:        Optional rendering options (NULL uses cacaosim_view_opts_defaults)
 * @fp:          Destination output stream (safe no-op if NULL)
 *
 * Role: Formats parse errors, missing upstream optical beams, duplicate beam producers, or cyclic
 * DAG wiring errors inside a high-visibility red/error-themed bordered panel on @fp.
 *
 * Rationale: During interactive editing or --watch live reload, typos in beam names are common.
 * Rendering a structured diagnostic panel—complete with the partially parsed node list and each
 * node's in/out/refl/trans/lowfs beam names—lets the user pinpoint broken beam links immediately
 * without losing visual layout consistency.
 */
void cacaosim_bench_render_error(
    const char                  *source_path,
    const cacaosim_bench_cfg_t  *partial_cfg,
    const char                  *err_msg,
    const cacaosim_view_opts_t  *opts,
    FILE                        *fp);

/**
 * cacaosim_bench_render_file() - Load, analyze, and render a .bench file in one zero-FFT call
 * @path: Filesystem path to the declarative .bench configuration file (non-NULL)
 * @opts: Optional rendering options (NULL uses cacaosim_view_opts_defaults)
 * @fp:   Destination output stream (non-NULL)
 *
 * Role: Convenience one-shot pipeline combining cacaosim_bench_load_file(),
 * cacaosim_bench_analyze(), and either cacaosim_bench_render() (on success) or
 * cacaosim_bench_render_error() (on I/O, syntax, or DAG topology failure).
 *
 * Rationale: Encapsulates the standard CLI and watch-loop sequence into a single stack-allocated,
 * zero-heap, zero-FFT call (< 1 ms) that always produces visual feedback on @fp—even when the
 * input file is missing or contains invalid beam wiring—while returning a crisp exit status code.
 *
 * Return: 0 if a valid .bench optical topology was rendered, -1 if @path/@fp is NULL or an error
 *         diagnostic panel was rendered.
 */
int cacaosim_bench_render_file(
    const char                 *path,
    const cacaosim_view_opts_t *opts,
    FILE                       *fp);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BENCH_VIEW_H */
