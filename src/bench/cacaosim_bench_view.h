// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_view.h
 * @brief   milk-styled ANSI / UTF-8 / ASCII graphical visualizer for .bench AO configurations
 */

#ifndef CACAOSIM_BENCH_VIEW_H
#define CACAOSIM_BENCH_VIEW_H

#include <stdio.h>
#include "bench/cacaosim_bench.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Color palette themes matching milk's overview TUI themes.
 */
typedef enum
{
    CACAOSIM_THEME_DARK       = 0, /**< Default milk dark slate / cyan / blue / gold palette */
    CACAOSIM_THEME_NIGHT      = 1, /**< Observatory Red night-vision-safe palette */
    CACAOSIM_THEME_ACCESSIBLE = 2, /**< Okabe-Ito colorblind-safe high-contrast palette */
    CACAOSIM_THEME_LIGHT      = 3, /**< Paper Light palette for light terminal backgrounds */
    CACAOSIM_THEME_NORDIC     = 4, /**< Cool Nordic frost / aurora palette */
    CACAOSIM_THEME_COUNT      = 5
} cacaosim_view_theme_t;

/**
 * @brief Rendering options for cacaosim_bench_render().
 */
typedef struct
{
    int32_t color_level; /**< -1=auto (TTY+NO_COLOR), 0=mono, 1=16c, 2=256c, 3=truecolor */
    int32_t theme;       /**< Palette index (cacaosim_view_theme_t, default: CACAOSIM_THEME_DARK) */
    int32_t ascii_only;  /**< 1 for pure 7-bit ASCII boxes/bars, 0 for UTF-8 box-drawing & blocks */
    int32_t check_shm;   /**< 1 to probe MILK_SHM_DIR for live .im.shm streams (default: 1) */
    int32_t show_tables; /**< 1 to render Active Mirrors & Sensors hardware tables (default: 1) */
    int32_t show_footer; /**< 1 to render CLI edit/deploy workflow hints footer (default: 1) */
    int32_t width;       /**< Target panel width in columns (0 = auto-detect terminal [88..112]) */
} cacaosim_view_opts_t;

/**
 * cacaosim_view_opts_defaults() - Populate default milk-styled visualization options
 */
void cacaosim_view_opts_defaults(cacaosim_view_opts_t *opts);

/**
 * cacaosim_view_theme_name() - Return human-readable name for a theme index
 */
const char *cacaosim_view_theme_name(int theme);

/**
 * cacaosim_view_theme_from_name() - Parse theme name ("dark", "night", "accessible", "light",
 * "nordic")
 *
 * Return: Theme enum index [0..4], or -1 if unrecognized.
 */
int cacaosim_view_theme_from_name(const char *name);

/**
 * cacaosim_bench_render() - Render a compiled/analyzed optical bench DAG as an ANSI/UTF-8 diagram
 * @ctx:         Analyzed or initialized optical bench context
 * @source_path: Optional .bench file path displayed in header and workflow hints
 * @opts:        Optional rendering options (NULL uses cacaosim_view_opts_defaults)
 * @fp:          Destination stream (e.g. stdout or open_memstream)
 */
void cacaosim_bench_render(
    const cacaosim_bench_ctx_t  *ctx,
    const char                  *source_path,
    const cacaosim_view_opts_t  *opts,
    FILE                        *fp);

/**
 * cacaosim_bench_render_error() - Render a milk-styled configuration error diagnostic panel
 * @source_path: Path to the .bench file being edited
 * @partial_cfg: Optional partially parsed bench config (NULL if file I/O failed)
 * @err_msg:     Syntax or DAG wiring error message
 * @opts:        Optional rendering options (NULL uses defaults)
 * @fp:          Destination stream
 */
void cacaosim_bench_render_error(
    const char                  *source_path,
    const cacaosim_bench_cfg_t  *partial_cfg,
    const char                  *err_msg,
    const cacaosim_view_opts_t  *opts,
    FILE                        *fp);

/**
 * cacaosim_bench_render_file() - Load, analyze, and render a .bench file in one zero-FFT call
 * @path: Path to .bench configuration file
 * @opts: Optional rendering options (NULL uses defaults)
 * @fp:   Destination stream
 *
 * Return: 0 if valid .bench topology was rendered, -1 if syntax/DAG error panel was rendered.
 */
int cacaosim_bench_render_file(
    const char                 *path,
    const cacaosim_view_opts_t *opts,
    FILE                       *fp);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BENCH_VIEW_H */
