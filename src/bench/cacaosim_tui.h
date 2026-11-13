// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui.h
 * @brief   Interactive milk-CTRL-style terminal hardware configurator and live status GUI
 *
 * Architectural Role in cacaosim-CTRL:
 *   Defines the public C API contract for launching interactive `cacaosim-CTRL` (`cacaosim-bench
 *   gui`) terminal sessions and executing deterministic headless plain-text frame snapshots for
 *   automated self-tests. Callers interact exclusively with workspace view identifiers
 *   (`cacaosim_tui_view_t`), startup preferences (`cacaosim_tui_opts_t`), the modal event-loop
 *   entry point (`cacaosim_tui_run()`), and the TTY-free snapshot entry point
 *   (`cacaosim_tui_render_headless()`), while raw `termios` state, ANSI cell diffing, live
 *   `ImageStreamIO` shared-memory probing, and parameter reflection remain encapsulated in
 *   `cacaosim_tui_internal.h`.
 *
 * Design Rationale & Motivation:
 *   `cacaosim-CTRL` uses a custom double-buffered 2D cell canvas with DEC synchronized updates
 *   (`DECSET 2026`) and SGR 1006 mouse decoding rather than `ncurses` for five reasons:
 *   1. Native 24-bit TrueColor (`\033[38;2;R;G;Bm`) support with automatic perceptual fallback to
 *      256-color, 16-color, and `NO_COLOR` monochrome terminals, enabling physical wavelength
 *      colormaps (`0.40..2.40 um`) and scotopic Observatory Red night-vision palettes.
 *   2. Arbitrary multi-byte UTF-8 glyphs per cell (`U+2580`/`U+2584` half-blocks, `U+2581..U+2588`
 *      spectral sparklines, `U+2800..U+28FF` Braille patterns, and rounded/double box-drawing
 *      characters) for dense 2D pupil, wavefront, and spectral visualizations without `ncursesw`
 *      wide-character locale fragility.
 *   3. Zero external UI library dependencies (`libncurses` / `libtinfo`), ensuring effortless
 *      builds and deployment on minimal real-time AO servers, containers, and SSH sessions.
 *   4. Deterministic headless rendering (`cacaosim_tui_render_headless()`) directly from the
 *      virtual cell back-buffer into a plain C string without allocating a pseudo-terminal (`pty`)
 *      or relying on system `terminfo` databases during CI testing.
 *   5. Tear-free, low-bandwidth screen updates during 10-30 Hz live SHM telemetry polling by
 *      emitting only changed cells inside atomic synchronized-update escape brackets.
 */

#ifndef CACAOSIM_TUI_H
#define CACAOSIM_TUI_H

#include <stddef.h>
#include <stdint.h>

#include "bench/cacaosim_bench.h"
#include "bench/cacaosim_bench_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Primary top-level workspace tabs in `cacaosim-CTRL` (bound to function keys `F2`..`F6`).
 *
 * Separates interactive topology authoring (`BUILDER`), unobstructed 2D optical & spectral diagram
 * inspection (`SCHEMATIC`), compact hardware/photometry audit tables (`TABLES`), live runtime SHM
 * & FPS process control (`STATUS`), and transparent `.bench` file serialization preview (`SOURCE`)
 * so each workflow fits cleanly within standard `120x38` SSH terminal viewports.
 */
typedef enum
{
    CACAOSIM_TUI_VIEW_BUILDER   = 0, /**< F2: 3-pane Builder (Elements + Schematic + Inspector) */
    CACAOSIM_TUI_VIEW_SCHEMATIC = 1, /**< F3: Full-screen 2D Optical Schematic & 32-bin Spectra */
    CACAOSIM_TUI_VIEW_TABLES    = 2, /**< F4: Hardware Mirrors, Sensors & Optical Beams tables */
    CACAOSIM_TUI_VIEW_STATUS    = 3, /**< F5: Live MILK_SHM_DIR streams, FPS & deploy controls */
    CACAOSIM_TUI_VIEW_SOURCE    = 4, /**< F6: Live formatted .bench INI/TOML serialization view */
    CACAOSIM_TUI_VIEW_COUNT     = 5  /**< Total number of addressable top-level workspace views */
} cacaosim_tui_view_t;

/**
 * @brief Startup and runtime configuration options passed to `cacaosim_tui_run()`.
 *
 * Groups visual terminal capabilities (color depth, theme palette, UTF-8 vs 7-bit ASCII fallback),
 * initial workspace navigation, and child-process deployment defaults (`htsim` vs modular FPS,
 * optional atmospheric phase-screen generator, and binary search directory). Always initialize via
 * `cacaosim_tui_opts_defaults()` before overriding individual fields for forward compatibility.
 */
typedef struct
{
    int32_t color_level;   /**< SGR depth: -1=auto (env), 0=mono, 1=16c, 2=256c, 3=24-bit RGB */
    int32_t theme;         /**< Initial UI palette index (`cacaosim_view_theme_t` [0..4]) */
    int32_t ascii_only;    /**< 1 to force pure 7-bit ASCII borders/bars, 0 to allow UTF-8 */
    int32_t initial_view;  /**< Starting workspace tab (`cacaosim_tui_view_t` [0..4]) */
    int32_t deploy_mode;   /**< Default deploy backend: 0 = fused (`htsim`), 1 = modular FPS */
    int32_t with_atmturb;  /**< 1 to co-launch `milk-fpsexec-atmturb-mkwfs` on deploy, else 0 */
    float   seeing_arcsec; /**< Atmospheric Fried seeing [arcsec] passed when `with_atmturb == 1` */
    char    exe_dir[512];  /**< Directory of invoking executable for locating sibling CLI helpers */
} cacaosim_tui_opts_t;

/**
 * cacaosim_tui_opts_defaults() - Populate default startup options for `cacaosim-CTRL`
 * @opts: Destination options struct to initialize (safe no-op if NULL)
 *
 * Role:
 *   Establishes canonical baseline settings (`color_level = -1` auto-detect, Dark Slate theme,
 *   UTF-8 enabled when locale permits, `F2:BUILDER` initial tab, fused `htsim` deploy mode, and
 *   `0.70 arcsec` nominal seeing) prior to CLI flag overrides.
 *
 * Rationale:
 *   Centralizing option initialization guarantees deterministic behavior when callers pass a
 *   partially overridden struct or `NULL` to `cacaosim_tui_run()`, and isolates CLI parsers from
 *   future additions to `cacaosim_tui_opts_t`.
 */
void cacaosim_tui_opts_defaults(cacaosim_tui_opts_t *opts);

/**
 * cacaosim_tui_run() - Run the interactive `milk-CTRL`-style hardware builder and live status TUI
 * @bench_path: Optional `.bench` file path to load or create (NULL or missing path starts from a
 *              built-in preset template associated with @bench_path when saved)
 * @opts:       Optional startup preferences (NULL applies `cacaosim_tui_opts_defaults()`)
 *
 * Role:
 *   Top-level lifecycle driver for an interactive `cacaosim-CTRL` session. Loads or initializes the
 *   optical bench DAG, enters raw alternate-screen terminal mode with SGR 1006 mouse tracking and
 *   stdio capture, runs the poll-driven input / live-SHM-telemetry / delta-render loop until user
 *   exit or fatal signal, and restores original terminal state on return.
 *
 * Rationale:
 *   Encapsulating full TUI state allocation, signal registration, `MILK_SHM_DIR` polling, and
 *   terminal cleanup behind a single blocking call allows both standalone `cacaosim-CTRL` binaries
 *   and `cacaosim-bench gui` subcommands to invoke the configurator without managing terminal file
 *   descriptors or cleanup hooks.
 *
 * Return: 0 on clean user exit, or non-zero on allocation or terminal initialization failure.
 */
int cacaosim_tui_run(
    const char                *bench_path,
    const cacaosim_tui_opts_t *opts);

/**
 * cacaosim_tui_render_headless() - Render one TUI frame into plain text for headless verification
 * @cfg:        Parsed `.bench` configuration to analyze and render (must not be NULL)
 * @view_id:    Target workspace tab index (`cacaosim_tui_view_t`, clamped to valid range)
 * @cols:       Virtual terminal width in character columns (clamped to >= 40)
 * @rows:       Virtual terminal height in character rows (clamped to >= 12)
 * @ascii_only: 1 to emit pure 7-bit ASCII borders/glyphs, 0 to emit UTF-8 glyphs
 * @out_buf:    Destination NUL-terminated plain-text snapshot buffer (rows joined by `\n`)
 * @out_cap:    Capacity of @out_buf in bytes (must be > 0)
 *
 * Role:
 *   Executes the complete zero-FFT optical analysis (`cacaosim_bench_analyze()`) and full TUI frame
 *   layout/widget renderer for @view_id into an in-memory `cs_screen_t` canvas, then serializes the
 *   resulting character grid into @out_buf without touching `stdin`, `stdout`, `termios`, or ANSI
 *   escape sequences.
 *
 * Rationale:
 *   Interactive terminal UIs are notoriously hard to regression-test if rendering is coupled to a
 *   live TTY or `ncurses` global state. Exposing the exact production layout engine as a pure
 *   in-memory function lets unit tests verify every workspace tab, box border, node card, and
 *   parameter table deterministically across arbitrary viewport dimensions.
 *
 * Return: Number of bytes written to @out_buf (excluding the trailing NUL), or -1 on invalid args
 *         or out-of-memory failure.
 */
int cacaosim_tui_render_headless(
    const cacaosim_bench_cfg_t *cfg,
    int                         view_id,
    int                         cols,
    int                         rows,
    int                         ascii_only,
    char                       *out_buf,
    size_t                      out_cap);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_TUI_H */
