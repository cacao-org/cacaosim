// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui.h
 * @brief   Interactive milk-CTRL-style terminal hardware configurator and live status GUI
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
 * @brief Primary workspace tabs in cacaosim-CTRL.
 */
typedef enum
{
    CACAOSIM_TUI_VIEW_BUILDER   = 0, /**< F2: 3-pane Builder (Elements + Schematic + Inspector) */
    CACAOSIM_TUI_VIEW_SCHEMATIC = 1, /**< F3: Full-screen 2D Optical Schematic & Spectra */
    CACAOSIM_TUI_VIEW_TABLES    = 2, /**< F4: Hardware Mirrors, Sensors & Optical Beams tables */
    CACAOSIM_TUI_VIEW_STATUS    = 3, /**< F5: Live MILK_SHM_DIR streams, FPS & deploy controls */
    CACAOSIM_TUI_VIEW_SOURCE    = 4, /**< F6: Live formatted .bench INI/TOML preview */
    CACAOSIM_TUI_VIEW_COUNT     = 5
} cacaosim_tui_view_t;

/**
 * @brief Launch options for cacaosim_tui_run().
 */
typedef struct
{
    int32_t color_level;   /**< -1=auto, 0=mono, 1=16c, 2=256c, 3=truecolor */
    int32_t theme;         /**< Palette index (cacaosim_view_theme_t) */
    int32_t ascii_only;    /**< 1 for pure 7-bit ASCII borders/glyphs, 0 for UTF-8 */
    int32_t initial_view;  /**< Starting tab (cacaosim_tui_view_t) */
    int32_t deploy_mode;   /**< Default deploy mode: 0 = fused (htsim), 1 = modular FPS */
    int32_t with_atmturb;  /**< 1 to launch milk-fpsexec-atmturb-mkwfs when deploying */
    float   seeing_arcsec; /**< Atmospheric seeing [arcsec] when with_atmturb == 1 */
    char    exe_dir[512];  /**< Directory of invoking binary for locating helper tools */
} cacaosim_tui_opts_t;

/**
 * cacaosim_tui_opts_defaults() - Populate default options for cacaosim-CTRL
 */
void cacaosim_tui_opts_defaults(cacaosim_tui_opts_t *opts);

/**
 * cacaosim_tui_run() - Run the interactive milk-CTRL-style hardware builder and status TUI
 * @bench_path: Optional .bench file path to load/edit (NULL or new path starts from template)
 * @opts:       Optional TUI configuration (NULL uses cacaosim_tui_opts_defaults)
 *
 * Return: 0 on normal exit, non-zero on fatal terminal error.
 */
int cacaosim_tui_run(
    const char                *bench_path,
    const cacaosim_tui_opts_t *opts);

/**
 * cacaosim_tui_render_headless() - Render one TUI frame into plain text for headless verification
 * @cfg:        Bench configuration to render
 * @view_id:    Workspace view index (cacaosim_tui_view_t)
 * @cols:       Virtual terminal width in columns
 * @rows:       Virtual terminal height in rows
 * @ascii_only: 1 for 7-bit ASCII glyphs, 0 for UTF-8
 * @out_buf:    Destination NUL-terminated plain-text snapshot (rows joined by '\n')
 * @out_cap:    Capacity of @out_buf in bytes
 *
 * Return: Number of bytes written to @out_buf (excluding NUL), or -1 on error.
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
