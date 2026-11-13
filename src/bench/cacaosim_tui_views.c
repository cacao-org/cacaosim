// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_views.c
 * @brief   Secondary full-screen workspace tabs (`F3:SCHEMATIC` .. `F6:SOURCE`) for `cacaosim-CTRL`
 *
 * Architectural Role in `cacaosim-CTRL`:
 *   Implements the four dedicated full-width workspace views rendered by `cs_tui_render_frame()`
 *   when the operator switches tabs away from the 3-pane `F2:BUILDER` configurator:
 *     - `F3:SCHEMATIC` (`cs_render_view_schematic()`): Full-screen 2D interactive optical bench
 *       schematic paired with an 11-row bottom deck showing the 16x16 anti-aliased telescope pupil,
 *       physical pupil/focal/sky pixel scales (`m/pix`, `l/D/pix`, `arcsec/pix`), and 32-bin
 *       polychromatic (`0.40 .. 2.40 um`) spectral & photon-flux budgets for every beam in the DAG.
 *     - `F4:TABLES` (`cs_render_view_tables()`): Structured engineering and photometry audit tables
 *       grouped into Active Mirrors (DM/TTM geometry, stroke, ZOH servo dynamics, frame delay),
 *       Wavefront & Focal-Plane Sensors (detector size, passband, rate, `navg`, effective flux in
 *       `ph/fr`, read noise), and Beamsplitters / Coronagraphs / Static WFE elements.
 *     - `F5:STATUS` (`cs_render_view_status()`): Live `MILK_SHM_DIR` shared-memory stream and
 *       `milk` FPS process monitor with an integrated runtime control bar (`[f:Deploy Fused]`,
 *       `[M:Deploy Modular]`, `[S:Stop Bench]`, `[g:Init Pupil+WFE]`, `[t:AtmTurb]`), stream frame
 *       counters (`cnt0`), measured frame rates (`Hz`), writer PIDs, Strehl telemetry, and a
 *       tailing deployment/activity event log.
 *     - `F6:SOURCE` (`cs_render_view_source()`): Scrollable, line-numbered, syntax-highlighted live
 *       preview of the serialized `.bench` INI/TOML configuration text with one-click disk save
 *       and `$VISUAL` / `$EDITOR` round-trip launch buttons.
 *
 * Design Rationale & Motivation:
 *   - Separating dedicated full-width views (`F3`..`F6`) from the 3-pane builder (`F2`) preserves
 *     legibility on `80x24` .. `120x38` SSH sessions while giving optical engineers full horizontal
 *     width to audit multi-branch polychromatic flux budgets, compare sensor noise/rate budgets
 *     side by side, monitor live RTC `ImageStreamIO` streams, and verify exact `.bench` output.
 *   - Registering clickable hit regions (`cs_tui_add_hit()`) directly during row and button
 *     rendering keeps `app->sel_item` synchronized across tabs: clicking a mirror or WFS row in
 *     `F4:TABLES` or a node box in `F3:SCHEMATIC` and pressing `Enter` jumps straight into its
 *     parameter inspector in `F2:BUILDER` with zero duplicate layout state.
 *   - Generating `F6:SOURCE` on demand from `cacaosim_bench_format_str(&app->cfg, ...)` guarantees
 *     the preview always matches the exact canonical serializer output consumed by `cacaosim.htsim`
 *     and `cacaosim-bench`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench/cacaosim_tui_internal.h"

/**
 * fmt_flux_v() - Format a photon rate or per-frame photon count into a compact metric string
 * @ph:  Photon count or rate (`[ph/s]` or `[ph/frame]`; `<= 0` prints `"0"`)
 * @out: Destination character buffer (non-NULL)
 * @cap: Capacity of @out in bytes
 *
 * Role:
 *   Formats wide-dynamic-range stellar and sensor photon budgets (`0` up to `> 10^9` photons)
 *   into fixed-width table cells across `F3:SCHEMATIC` and `F4:TABLES`.
 *
 * Rationale:
 *   AO benches span bright natural guide stars (`10^8 .. 10^10 ph/s`) down to faint coronagraphic
 *   leakage (`< 100 ph/fr`). Compact SI-style suffixes (`k`, `M`, `e9`) prevent table columns
 *   from overflowing 10-12 character cell widths without sacrificing 2-3 significant digits.
 */
static void fmt_flux_v(float ph, char *out, size_t cap)
{
    if (ph <= 0.0f)
    {
        snprintf(out, cap, "0");
    }
    else if (ph >= 1.0e9f)
    {
        snprintf(out, cap, "%.2fe9", ph * 1.0e-9f);
    }
    else if (ph >= 1.0e6f)
    {
        snprintf(out, cap, "%.2fM", ph * 1.0e-6f);
    }
    else if (ph >= 1.0e3f)
    {
        snprintf(out, cap, "%.1fk", ph * 1.0e-3f);
    }
    else
    {
        snprintf(out, cap, "%.0f", ph);
    }
}

/**
 * shm_dtype_label() - Map an `ImageStreamIO` pixel datatype code to a 3-char mnemonic
 * @dt: `ImageStreamIO` `_DATATYPE_*` enum byte read from shared-memory stream metadata
 *
 * Role:
 *   Supplies the compact `"DTYPE"` column label in the `F5:STATUS` live shared-memory stream
 *   table (`"u8"`, `"i16"`, `"u16"`, `"i32"`, `"u32"`, `"f32"`, `"f64"`, or fallback `"img"`).
 *
 * Rationale:
 *   AO hardware streams mix `f32` OPD/DM command maps with `u16` / `i32` camera readout frames;
 *   a 3-character type badge lets operators spot datatype mismatches between WFS outputs and
 *   CACAO calibration loops at a glance while using only 6 columns of horizontal space.
 *
 * Return: Pointer to a static NUL-terminated string literal.
 */
static const char *shm_dtype_label(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "u8";
    case _DATATYPE_INT16:
        return "i16";
    case _DATATYPE_UINT16:
        return "u16";
    case _DATATYPE_INT32:
        return "i32";
    case _DATATYPE_UINT32:
        return "u32";
    case _DATATYPE_FLOAT:
        return "f32";
    case _DATATYPE_DOUBLE:
        return "f64";
    default:
        return "img";
    }
}

/**
 * view_btn() - Render a clickable action pill button and register its SGR mouse hit box
 * @app: Master TUI application state (non-NULL)
 * @x:   Leftmost terminal column (0-based)
 * @y:   Terminal row (0-based)
 * @lbl: Button label text (e.g., `"[f:Deploy Fused]"`, non-NULL, clipped at 36 columns)
 * @hk:  Semantic hit-box action kind (`cs_hit_kind_t`) dispatched on left-click
 * @arg: Auxiliary integer payload stored in the hit entry
 * @fg:  Foreground RGB color
 * @bg:  Background RGB fill color
 * @tip: Hover tooltip message displayed in the status/tooltip overlay (nullable)
 *
 * Role:
 *   Draws toolbar buttons inside full-screen secondary views (`F5:STATUS` and `F6:SOURCE`) and
 *   atomically records their exact screen bounding box in `app->hits[]`.
 *
 * Rationale:
 *   Coupling `cs_put_str()` and `cs_tui_add_hit()` into a single immediate-mode call guarantees
 *   that clickable coordinates and hover tooltips never drift out of sync with button label widths
 *   when dynamic text (such as seeing `[t:AtmTurb ON 0.70"]`) changes length.
 *
 * Return: Number of terminal columns occupied by the rendered label.
 */
static int view_btn(
    cs_tui_app_t *app,
    int           x,
    int           y,
    const char   *lbl,
    cs_hit_kind_t hk,
    int           arg,
    cs_tui_rgb_t  fg,
    cs_tui_rgb_t  bg,
    const char   *tip)
{
    int len = cs_put_str(&app->scr, x, y, 36, lbl, fg, bg, CS_ATTR_BOLD, 1);
    cs_tui_add_hit(app, hk, arg, x, y, len, 1, tip);
    return len;
}

/* ── F3:SCHEMATIC — Full-Width Optical Schematic & 32-Bin Spectral Deck ───── */

/**
 * cs_render_view_schematic() - Render the `F3:SCHEMATIC` full-screen diagram and spectral deck
 * @app: Master TUI application state containing bench config, zero-FFT analysis, and pupil cache
 * @x:   Viewport left terminal column (0-based)
 * @y:   Viewport top terminal row (0-based)
 * @w:   Viewport width in columns
 * @h:   Viewport height in rows
 *
 * Role:
 *   Draws the dedicated full-width optical architecture view (`F3`), combining the interactive 2D
 *   DAG canvas (with wide 24-column terminal sensor sparklines enabled) on top with an 11-row
 *   optical summary deck on the bottom showing the 16x16 entrance pupil, pupil/focal/sky pixel
 *   scales, and 32-bin polychromatic (`0.40 .. 2.40 um`) spectral bars + integrated photon flux
 *   for every optical beam in `app->ctx.beams[]`.
 *
 * Rationale:
 *   In multi-stage benches with several dichroic splits and LOWFS arms, the right-hand schematic
 *   pane in `F2:BUILDER` may require horizontal scrolling. `F3:SCHEMATIC` gives the DAG the full
 *   terminal width while juxtaposing every intermediate beam's passband and flux budget below it,
 *   making missing throughput or misordered dichroic cutoffs immediately visible. On short
 *   terminals (`h < 24`), the bottom deck collapses gracefully (`deck_h = 0`) so the schematic
 *   remains usable.
 */
void cs_render_view_schematic(cs_tui_app_t *app, int x, int y, int w, int h)
{
    const cs_tui_pal_t *pal     = cs_tui_get_palette(app->opts.theme);
    /* Reserve 11 bottom rows for the pupil/scale/spectrum deck only when terminal height permits */
    int                 deck_h  = (h >= 24) ? 11 : 0;
    int                 schem_h = h - deck_h;

    cs_draw_box(&app->scr, x, y, w, schem_h, 1, "FULL-SCREEN 2D OPTICAL BENCH SCHEMATIC",
                "Click node + Enter -> F2:Builder", pal->fg_border_focus, pal->fg_title,
                pal->fg_muted, pal->bg_panel);
    cs_render_dag_canvas(app, x + 1, y + 1, w - 2, schem_h - 2, 1);

    if (deck_h == 0)
    {
        return;
    }

    int dy = y + schem_h;
    cs_draw_box(&app->scr, x, dy, w, deck_h, 0,
                "ENTRANCE PUPIL, PIXEL SCALES & 32-BIN POLYCHROMATIC SPECTRA (0.40 .. 2.40 um)",
                NULL, pal->fg_border, pal->fg_title, pal->fg_muted, pal->bg_panel);

    cs_optical_scales_t sc;
    cs_tui_compute_scales(&app->cfg, app->sel_item, &sc);

    /* Left & middle deck columns: 16x16 anti-aliased pupil glyph + physical sampling summary */
    cs_draw_pupil16(&app->scr, x + 2, dy + 1, app->pup16, pal);
    cs_printf_at(&app->scr, x + 20, dy + 1, 33, pal->fg_bright, pal->bg_panel, CS_ATTR_BOLD, 1,
                 "Telescope %.2f m (obs %.0f%%)", app->cfg.tel_diam_m, app->cfg.cobs * 100.0f);
    cs_printf_at(&app->scr, x + 20, dy + 2, 33, pal->fg_text, pal->bg_panel, 0u, 1,
                 "Array: %ux%u pix", app->cfg.nwf, app->cfg.nwf);
    cs_printf_at(&app->scr, x + 20, dy + 3, 33, pal->fg_text, pal->bg_panel, 0u, 1,
                 "Beam : %.1f pix (%.1f%%)", sc.d_beam_px, app->cfg.pup_diam * 100.0f);
    cs_printf_at(&app->scr, x + 20, dy + 4, 33, pal->fg_stream, pal->bg_panel, 0u, 1,
                 "Pupil: %.4f m/pix", sc.pup_m_pix);
    cs_printf_at(&app->scr, x + 20, dy + 5, 33, pal->fg_accent, pal->bg_panel, 0u, 1,
                 "Focal: %.3f l/D/pix (%.8s)", sc.foc_ld_pix, sc.foc_src);
    cs_printf_at(&app->scr, x + 20, dy + 6, 33, pal->fg_muted, pal->bg_panel, 0u, 1,
                 "Sky  : %.4f arcsec/pix @%.2fum", sc.foc_arcsec_pix, sc.ref_lam_um);
    cs_printf_at(&app->scr, x + 20, dy + 7, 33, pal->fg_accent, pal->bg_panel, 0u, 1,
                 "Star : mV=%.1f  %.0fK  %.0fHz", app->cfg.star_mag, app->cfg.star_teff,
                 app->cfg.master_hz);
    cs_printf_at(&app->scr, x + 20, dy + 8, 33, pal->fg_dim, pal->bg_panel, 0u, 1,
                 "SHM  : wf=%s pup=%s", app->cfg.in_wf, app->cfg.in_pup);

    /* Right deck columns: per-beam 32-bin wavelength-colored spectrum bars & photon flux */
    int sx = x + 55;
    int sw = w - 57;
    if (sw > 36 && app->dag_valid)
    {
        cs_printf_at(&app->scr, sx, dy + 1, sw, pal->fg_dim, pal->bg_panel, 0u, 1,
                     "%-16s %-32s %-10s %s", "BEAM", "0.4u        1.0u        1.6u    2.4u",
                     "MEAN LAM", "FLUX [ph/s]");
        int max_b = deck_h - 3;
        for (int bi = 0; bi < (int) app->ctx.n_beams && bi < max_b; bi++)
        {
            const cacaosim_beam_slot_t *bs = &app->ctx.beams[bi];
            int                         ry = dy + 2 + bi;
            char                        fl[24];
            fmt_flux_v(cacaosim_spec_total(&bs->spec), fl, sizeof(fl));
            cs_printf_at(&app->scr, sx, ry, 16, pal->fg_stream, pal->bg_panel, CS_ATTR_BOLD, 1,
                         "%-16.16s", bs->name);
            cs_draw_spectrum32(&app->scr, sx + 17, ry, 32, &bs->spec, app->opts.theme,
                               pal->bg_panel);
            if (sw > 54)
            {
                cs_printf_at(&app->scr, sx + 50, ry, sw - 50, pal->fg_text, pal->bg_panel, 0u, 1,
                             "%-9.2f  %s", cs_tui_spec_mean_lambda(&bs->spec), fl);
            }
        }
    }
}

/* ── F4:TABLES — Hardware Active Mirrors, Sensors & Beams Summary ─────────── */

/**
 * cs_render_view_tables() - Render the `F4:TABLES` hardware and photometry summary tables
 * @app: Master TUI application state (non-NULL)
 * @x:   Viewport left terminal column (0-based)
 * @y:   Viewport top terminal row (0-based)
 * @w:   Viewport width in columns
 * @h:   Viewport height in rows
 *
 * Role:
 *   Draws three stacked, category-specialized hardware audit tables inside the `F4` workspace tab:
 *     1. `ACTIVE MIRRORS (DM & TIP-TILT)`: Actuator grid or 2-axis geometry, stroke saturation
 *        (`+/- um`), discrete ZOH servo dynamics (`instant`, 1st-order `rc`, or 2nd-order `osc`),
 *        fractional frame delay, input command SHM stream, and `in_beam -> out_beam` chain.
 *     2. `WAVEFRONT & FOCAL-PLANE SENSORS`: Detector pixel dimensions (`ncam` or
 *        `n_sub * pix_per_sub`), central wavelength & fractional bandwidth, effective readout
 *        rate (`Hz` and `navg` divisor), zero-FFT integrated flux (`ph/fr`), detector read noise
 *        (`e-`), output camera SHM stream, and upstream input beam.
 *     3. `BEAMSPLITTERS, CORONAGRAPHS & STATIC WFE`: Gray/dichroic split specs, analytic/custom
 *        coronagraph mask modes + `fpm_refl`/`lyot_refl` LOWFS pickoff descriptors, and inline
 *        `[meter]` OPD static WFE stream gains.
 *
 * Rationale:
 *   Different optical subsystem classes have fundamentally incompatible parameter sets (servo
 *   pole frequencies and DAC stroke for DMs vs. read noise and `ph/frame` for WFS detectors vs.
 *   dual output arms for splitters/coronagraphs). Grouping them into three domain-specific tables
 *   with aligned columns lets system engineers cross-check multi-stage timing, latency, and SNR
 *   budgets in a single glance, while registering `CS_HIT_ELEM_ROW` on every row so clicking any
 *   row highlights the element globally and `Enter` opens it in `F2:BUILDER`.
 */
void cs_render_view_tables(cs_tui_app_t *app, int x, int y, int w, int h)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    cs_draw_box(&app->scr, x, y, w, h, 1, "OPTICAL BENCH HARDWARE & PHOTOMETRY TABLES",
                "Click row -> Select  Enter -> Edit in F2:Builder", pal->fg_border_focus,
                pal->fg_title, pal->fg_muted, pal->bg_panel);

    int ry  = y + 1;
    int bot = y + h - 1;

    /* Section 1: Active Mirrors (zonal DMs and 2-axis Tip-Tilt mirrors) */
    if (ry + 2 < bot)
    {
        cs_printf_at(&app->scr, x + 2, ry++, w - 4, pal->fg_accent, pal->bg_panel, CS_ATTR_BOLD, 1,
                     "ACTIVE MIRRORS (DM & TIP-TILT)");
        cs_printf_at(&app->scr, x + 2, ry++, w - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                     "%-12s %-6s %-10s %-10s %-14s %-8s %-14s %s", "ID", "TYPE", "GRID", "STROKE",
                     "SERVO DYN", "DELAY", "CMD SHM", "BEAMS");
    }

    for (uint32_t i = 0; i < app->cfg.n_nodes && ry < bot; i++)
    {
        const cacaosim_node_cfg_t *nd = &app->cfg.nodes[i];
        if (nd->kind != CACAOSIM_NODE_DM)
        {
            continue;
        }
        int          sel = (app->sel_item == (int) i + 1);
        cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, x + 1, ry, w - 2, 1, pal->fg_text, bg);

        char grid[16], strk[16], dyn[24], dly[16], bms[40];
        if (nd->dm.dm_mode == 1)
        {
            snprintf(grid, sizeof(grid), "2-axis");
        }
        else
        {
            snprintf(grid, sizeof(grid), "%ux%u", nd->dm.nact_x, nd->dm.nact_y);
        }
        snprintf(strk, sizeof(strk), "+/-%.1fu", nd->dm.max_stroke);
        if (nd->dm.dyn_mode == 0)
        {
            snprintf(dyn, sizeof(dyn), "instant");
        }
        else if (nd->dm.dyn_mode == 2)
        {
            snprintf(dyn, sizeof(dyn), "osc %.0fHz", nd->dm.f0_hz);
        }
        else
        {
            snprintf(dyn, sizeof(dyn), "rc %.0fus", nd->dm.tau_sec * 1.0e6f);
        }
        snprintf(dly, sizeof(dly), "%.2ffr", nd->dm.delay_frames);
        snprintf(bms, sizeof(bms), "%s -> %s", nd->in_beam, nd->out_beam);

        cs_printf_at(&app->scr, x + 2, ry, w - 4, sel ? pal->fg_bright : pal->fg_text, bg,
                     sel ? CS_ATTR_BOLD : 0u, 1, "%-12.12s %-6s %-10s %-10s %-14s %-8s %-14.14s %s",
                     nd->id, (nd->dm.dm_mode == 1) ? "TTM" : "DM", grid, strk, dyn, dly,
                     nd->in_cmd, bms);
        cs_tui_add_hit(app, CS_HIT_ELEM_ROW, (int) i + 1, x + 1, ry, w - 2, 1,
                       "Click to select active mirror");
        ry++;
    }

    ry++;
    /* Section 2: Wavefront & Focal-Plane Sensors (PYRWFS, SHWFS, ZWFS, FPCAM) */
    if (ry + 2 < bot)
    {
        cs_printf_at(&app->scr, x + 2, ry++, w - 4, pal->fg_stream, pal->bg_panel, CS_ATTR_BOLD, 1,
                     "WAVEFRONT & FOCAL-PLANE SENSORS");
        cs_printf_at(&app->scr, x + 2, ry++, w - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                     "%-12s %-6s %-10s %-12s %-10s %-12s %-8s %-14s %s", "ID", "KIND", "DETECTOR",
                     "BAND [um]", "RATE [Hz]", "FLUX [ph/fr]", "RON", "OUT SHM", "IN BEAM");
    }

    for (uint32_t i = 0; i < app->cfg.n_nodes && ry < bot; i++)
    {
        const cacaosim_node_cfg_t *nd = &app->cfg.nodes[i];
        if (nd->kind != CACAOSIM_NODE_PYRWFS && nd->kind != CACAOSIM_NODE_SHWFS &&
            nd->kind != CACAOSIM_NODE_ZWFS && nd->kind != CACAOSIM_NODE_FPCAM)
        {
            continue;
        }
        int          sel = (app->sel_item == (int) i + 1);
        cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, x + 1, ry, w - 2, 1, pal->fg_text, bg);

        /* Extract per-sensor detector dimensions, wavelength, RON, and effective rate */
        const char *kstr = (nd->kind == CACAOSIM_NODE_PYRWFS)
                               ? "PYRWFS"
                               : (nd->kind == CACAOSIM_NODE_SHWFS
                                      ? "SHWFS"
                                      : (nd->kind == CACAOSIM_NODE_ZWFS ? "ZWFS" : "FPCAM"));
        uint32_t ncam = (nd->kind == CACAOSIM_NODE_PYRWFS)
                            ? nd->pyrwfs.ncam
                            : (nd->kind == CACAOSIM_NODE_SHWFS
                                   ? (nd->shwfs.n_sub * nd->shwfs.pix_per_sub)
                                   : (nd->kind == CACAOSIM_NODE_ZWFS ? nd->zwfs.ncam
                                                                     : nd->fpcam.ncam));
        float lam = (nd->kind == CACAOSIM_NODE_PYRWFS)
                        ? nd->pyrwfs.lambda_um
                        : (nd->kind == CACAOSIM_NODE_SHWFS
                               ? nd->shwfs.lambda_um
                               : (nd->kind == CACAOSIM_NODE_ZWFS ? nd->zwfs.lambda_um
                                                                 : nd->fpcam.lambda_um));
        float ron = (nd->kind == CACAOSIM_NODE_PYRWFS)
                        ? nd->pyrwfs.det.ron_e
                        : (nd->kind == CACAOSIM_NODE_SHWFS
                               ? nd->shwfs.det.ron_e
                               : (nd->kind == CACAOSIM_NODE_ZWFS ? nd->zwfs.det.ron_e
                                                                 : nd->fpcam.det.ron_e));
        float mhz = (app->cfg.master_hz > 0.0f) ? app->cfg.master_hz : 2000.0f;
        float hz  = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));

        /* Look up precomputed per-frame photon flux from the zero-FFT spectral analyzer */
        float flux = 0.0f;
        if (app->dag_valid)
        {
            for (uint32_t si = 0; si < app->ctx.n_nodes; si++)
            {
                if (strcmp(app->ctx.nodes[si].cfg.id, nd->id) == 0)
                {
                    flux = app->ctx.nodes[si].eff_flux_ph;
                    break;
                }
            }
        }

        char dets[16], band[16], rstr[16], fstr[16], rons[12];
        snprintf(dets, sizeof(dets), "%ux%u", ncam, ncam);
        snprintf(band, sizeof(band), "%.2f (%.0f%%)", lam, nd->bw_frac * 100.0f);
        snprintf(rstr, sizeof(rstr), "%.0f (/%u)", hz, nd->navg ? nd->navg : 1u);
        fmt_flux_v(flux, fstr, sizeof(fstr));
        snprintf(rons, sizeof(rons), "%.1fe-", ron);

        cs_printf_at(&app->scr, x + 2, ry, w - 4, sel ? pal->fg_bright : pal->fg_text, bg,
                     sel ? CS_ATTR_BOLD : 0u, 1,
                     "%-12.12s %-6s %-10s %-12s %-10s %-12s %-8s %-14.14s %s", nd->id, kstr, dets,
                     band, rstr, fstr, rons, nd->out_cam, nd->in_beam);
        cs_tui_add_hit(app, CS_HIT_ELEM_ROW, (int) i + 1, x + 1, ry, w - 2, 1,
                       "Click to select wavefront/science sensor");
        ry++;
    }

    ry++;
    /* Section 3: Beamsplitters, Coronagraphs & Static WFE */
    if (ry + 2 < bot)
    {
        cs_printf_at(&app->scr, x + 2, ry++, w - 4, pal->fg_proc, pal->bg_panel, CS_ATTR_BOLD, 1,
                     "BEAMSPLITTERS, CORONAGRAPHS & STATIC WFE");
        cs_printf_at(&app->scr, x + 2, ry++, w - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                     "%-12s %-10s %-24s %-16s %s", "ID", "KIND", "MODE / SPEC", "IN BEAM",
                     "OUTPUT ARMS / SHM");
    }

    for (uint32_t i = 0; i < app->cfg.n_nodes && ry < bot; i++)
    {
        const cacaosim_node_cfg_t *nd = &app->cfg.nodes[i];
        if (nd->kind != CACAOSIM_NODE_SPLITTER && nd->kind != CACAOSIM_NODE_CORO &&
            nd->kind != CACAOSIM_NODE_WFE)
        {
            continue;
        }
        int          sel = (app->sel_item == (int) i + 1);
        cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, x + 1, ry, w - 2, 1, pal->fg_text, bg);

        const char *kstr = (nd->kind == CACAOSIM_NODE_SPLITTER)
                               ? "SPLITTER"
                               : (nd->kind == CACAOSIM_NODE_CORO ? "CORO" : "WFE");
        char mstr[36], arms[112];
        if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            if (nd->split.mode == CACAOSIM_SPLIT_GRAY)
            {
                snprintf(mstr, sizeof(mstr), "gray R=%.2f T=%.2f", nd->split.r_gray,
                         nd->split.t_gray);
            }
            else
            {
                snprintf(mstr, sizeof(mstr), "dichroic %.2f um", nd->split.lambda_cut);
            }
            snprintf(arms, sizeof(arms), "R:%s  T:%s", nd->out_refl, nd->out_trans);
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            /* Summarize coronagraph architecture, custom FITS/SHM mask count & LOWFS pickoff */
            static const char *const cm_names[6] = {
                "lyot_spot", "dimple", "vortex", "4qpm", "custom_mft", "custom_fft"
            };
            const char *cm = (nd->coro.mode >= 0 && nd->coro.mode < 6) ? cm_names[nd->coro.mode]
                                                                       : "coro";
            if (nd->coro.nlambda > 1)
            {
                snprintf(mstr, sizeof(mstr), "%s %dch %.2fu", cm, nd->coro.nlambda,
                         nd->coro.lambda_um);
            }
            else
            {
                snprintf(mstr, sizeof(mstr), "%s %.2fu", cm, nd->coro.lambda_um);
            }
            int n_masks = (nd->in_pup_amp[0] != '\0') + (nd->in_pup_pha[0] != '\0') +
                          (nd->in_fpm_amp[0] != '\0') + (nd->in_fpm_pha[0] != '\0') +
                          (nd->in_lyot[0] != '\0') + (nd->in_lyot_pha[0] != '\0');
            char ldesc[40] = "";
            if (nd->out_lowfs[0] != '\0')
            {
                if (nd->coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL)
                {
                    snprintf(ldesc, sizeof(ldesc), "lyot_refl");
                }
                else
                {
                    float ap = (nd->coro.lowfs_pup_diam > 0.0f) ? nd->coro.lowfs_pup_diam : 1.0f;
                    snprintf(ldesc, sizeof(ldesc), "fpm_refl ap=%.2fD", ap);
                }
            }
            if (n_masks > 0 && nd->out_lowfs[0] != '\0')
            {
                snprintf(arms, sizeof(arms), "Sci:%s  LOWFS:%s (%s)  +%d masks", nd->out_beam,
                         nd->out_lowfs, ldesc, n_masks);
            }
            else if (nd->out_lowfs[0] != '\0')
            {
                snprintf(arms, sizeof(arms), "Sci:%s  LOWFS:%s (%s)", nd->out_beam, nd->out_lowfs,
                         ldesc);
            }
            else if (n_masks > 0)
            {
                snprintf(arms, sizeof(arms), "Sci:%s  (+%d custom masks)", nd->out_beam, n_masks);
            }
            else
            {
                snprintf(arms, sizeof(arms), "Sci:%s", nd->out_beam);
            }
        }
        else
        {
            snprintf(mstr, sizeof(mstr), "%.0fnm x%.2f [OPD m]", nd->wfe_rms_nm, nd->wfe_gain);
            snprintf(arms, sizeof(arms), "Out:%s  SHM:%s", nd->out_beam, nd->in_wfe);
        }

        cs_printf_at(&app->scr, x + 2, ry, w - 4, sel ? pal->fg_bright : pal->fg_text, bg,
                     sel ? CS_ATTR_BOLD : 0u, 1, "%-12.12s %-10s %-24s %-16.16s %s", nd->id, kstr,
                     mstr, nd->in_beam, arms);
        cs_tui_add_hit(app, CS_HIT_ELEM_ROW, (int) i + 1, x + 1, ry, w - 2, 1,
                       "Click to select splitter / coronagraph / static WFE");
        ry++;
    }
}

/* ── F5:STATUS — Live MILK_SHM_DIR Streams, FPS Units & Deployment ────────── */

/**
 * cs_render_view_status() - Render the `F5:STATUS` live SHM monitor, FPS deck, and event log
 * @app: Master TUI application state containing `app->shm` telemetry and `app->logs` (non-NULL)
 * @x:   Viewport left terminal column (0-based)
 * @y:   Viewport top terminal row (0-based)
 * @w:   Viewport width in columns
 * @h:   Viewport height in rows
 *
 * Role:
 *   Draws the runtime operations and live telemetry screen (`F5`), split into:
 *     - Top 58% pane (`LIVE SHARED-MEMORY STREAMS & DEPLOYMENT`): Interactive process-control
 *       toolbar (`Deploy Fused`, `Deploy Modular`, `Stop Bench`, `Init Pupil+WFE`, `AtmTurb`
 *       toggle) above a table of every expected and discovered `ImageStreamIO` stream in
 *       `MILK_SHM_DIR`, displaying role, owner, cube dimensions, datatype, frame counter (`cnt0`),
 *       measured frame rate (`Hz`), writer PID, and live status (`STREAMING` / `READY` / `STATIC`
 *       / `OFFLINE`).
 *     - Bottom-left 42% pane (`ACTIVE MILK FPS UNITS`): Running `csim_*` and `atmturb` `.fps.shm`
 *       function parameter structures with `conf_pid`, `run_pid`, `RUNNING`/`IDLE` state, and
 *       instantaneous/averaged Strehl ratio readouts (`SR`).
 *     - Bottom-right 42% pane (`ACTIVITY & DEPLOYMENT LOG`): Auto-tailing chronological ring of
 *       timestamped TUI actions, subprocess messages, and captured stdio warnings.
 *
 * Rationale:
 *   Combining bench deployment triggers, expected-vs-actual SHM stream liveness, FPS process PIDs,
 *   and subprocess logs on one screen eliminates switching tmux panes to run `milk-CTRL` or check
 *   why a CACAO loop lacks an expected input stream (`OFFLINE` vs. `STATIC` vs. `STREAMING`).
 */
void cs_render_view_status(cs_tui_app_t *app, int x, int y, int w, int h)
{
    const cs_tui_pal_t *pal   = cs_tui_get_palette(app->opts.theme);
    /* Allocate ~58% height to the SHM stream table and ~42% to FPS + Activity Log below */
    int                 top_h = (h * 58) / 100;
    if (top_h < 8)
    {
        top_h = 8;
    }
    if (top_h > h - 7)
    {
        top_h = h - 7;
    }
    int bot_h = h - top_h;

    char sbadge[96];
    snprintf(sbadge, sizeof(sbadge), "MILK_SHM_DIR: %.78s", app->shm.shm_dir);
    cs_draw_box(&app->scr, x, y, w, top_h, 1, "LIVE SHARED-MEMORY STREAMS & DEPLOYMENT", sbadge,
                pal->fg_border_focus, pal->fg_title, pal->fg_stream, pal->bg_panel);

    /* Action buttons row for one-click bench launch, teardown, and SHM map initialization */
    int bx = x + 2;
    int by = y + 1;
    bx += view_btn(app, bx, by, "[f:Deploy Fused]", CS_HIT_ACT_DEPLOY_F, 0, pal->fg_active,
                   pal->bg_panel_alt, "Deploy & run single-process htsim DAG [f]") +
          1;
    bx += view_btn(app, bx, by, "[M:Deploy Modular]", CS_HIT_ACT_DEPLOY_M, 0, pal->fg_stream,
                   pal->bg_panel_alt, "Deploy separate per-element FPS units [M]") +
          1;
    bx += view_btn(app, bx, by, "[S:Stop Bench]", CS_HIT_ACT_STOP, 0, pal->fg_error,
                   pal->bg_panel_alt, "Stop running cacaosim / atmturb processes [S]") +
          1;
    bx += view_btn(app, bx, by, "[g:Init Pupil+WFE]", CS_HIT_ACT_MKPUP, 0, pal->fg_fps,
                   pal->bg_panel_alt,
                   "Initialize entrance pupil & static WFE [m] SHM maps [g]") +
          1;
    {
        char at_lbl[32];
        snprintf(at_lbl, sizeof(at_lbl), "[t:AtmTurb %s %.2f\"]",
                 app->opts.with_atmturb ? "ON" : "OFF", app->opts.seeing_arcsec);
        view_btn(app, bx, by, at_lbl, CS_HIT_ACT_ATMTURB, 0,
                 app->opts.with_atmturb ? pal->fg_accent : pal->fg_muted, pal->bg_panel_alt,
                 "Toggle atmospheric turbulence launch on next deploy [t]");
    }

    cs_printf_at(&app->scr, x + 2, y + 3, w - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                 "%-18s %-10s %-12s %-14s %-6s %-12s %-10s %-8s %s", "STREAM", "ROLE", "OWNER",
                 "DIMENSIONS", "DTYPE", "CNT0", "RATE [Hz]", "PID", "STATE");

    /* Render expected & discovered ImageStreamIO streams color-coded by update activity */
    int s_rows = top_h - 5;
    for (int i = 0; i < app->shm.n_streams && i < s_rows; i++)
    {
        const cs_mon_stream_t *s  = &app->shm.streams[i];
        int                    ry = y + 4 + i;
        if (!s->exists)
        {
            cs_printf_at(&app->scr, x + 2, ry, w - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                         "%-18.18s %-10.10s %-12.12s %-14s %-6s %-12s %-10s %-8s OFFLINE", s->name,
                         s->role, s->owner, "-", "-", "-", "-", "-");
            continue;
        }
        char dims[24], pids[16], hzs[16];
        if (s->naxis >= 3 && s->size[2] > 1u)
        {
            snprintf(dims, sizeof(dims), "%ux%ux%u", s->size[0], s->size[1], s->size[2]);
        }
        else if (s->naxis == 2)
        {
            snprintf(dims, sizeof(dims), "%ux%u", s->size[0], s->size[1]);
        }
        else
        {
            snprintf(dims, sizeof(dims), "%u", s->size[0]);
        }
        snprintf(pids, sizeof(pids), "%d", (int) s->writer_pid);
        snprintf(hzs, sizeof(hzs), "%.1f", s->hz);

        cs_tui_rgb_t sfg = (s->hz > 0.5) ? pal->fg_active
                                         : (s->alive ? pal->fg_stream : pal->fg_text);
        cs_printf_at(&app->scr, x + 2, ry, w - 4, sfg, pal->bg_panel,
                     (s->hz > 0.5) ? CS_ATTR_BOLD : 0u, 1,
                     "%-18.18s %-10.10s %-12.12s %-14s %-6s %-12llu %-10s %-8s %s", s->name,
                     s->role, s->owner, dims, shm_dtype_label(s->datatype),
                     (unsigned long long) s->cnt0, hzs, pids,
                     (s->hz > 0.5) ? "STREAMING" : (s->alive ? "READY" : "STATIC"));
    }

    /* Bottom split: Left = Active FPS Units, Right = Activity Log */
    int fy  = y + top_h;
    int fw  = w / 2;
    int lw2 = w - fw;
    cs_draw_box(&app->scr, x, fy, fw, bot_h, 0, "ACTIVE MILK FPS UNITS", NULL, pal->fg_border,
                pal->fg_title, pal->fg_muted, pal->bg_panel);
    cs_printf_at(&app->scr, x + 2, fy + 1, fw - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                 "%-18s %-8s %-8s %-8s %s", "FPS NAME", "CONF", "RUN PID", "STATE", "TELEMETRY");

    int f_rows = bot_h - 3;
    if (app->shm.n_fps == 0 && f_rows > 0)
    {
        cs_put_str(&app->scr, x + 2, fy + 2, fw - 4, "(no csim_* / atmturb .fps.shm active)",
                   pal->fg_dim, pal->bg_panel, 0u, 1);
    }
    for (int i = 0; i < app->shm.n_fps && i < f_rows; i++)
    {
        const cs_mon_fps_t *f   = &app->shm.fps[i];
        char                tel[32] = "";
        if (f->has_strehl)
        {
            snprintf(tel, sizeof(tel), "SR=%.3f (inst %.3f)", f->strehl_ave, f->strehl_inst);
        }
        cs_printf_at(&app->scr, x + 2, fy + 2 + i, fw - 4,
                     f->run_alive ? pal->fg_active : pal->fg_muted, pal->bg_panel,
                     f->run_alive ? CS_ATTR_BOLD : 0u, 1, "%-18.18s %-8d %-8d %-8s %s", f->name,
                     (int) f->conf_pid, (int) f->run_pid, f->run_alive ? "RUNNING" : "IDLE", tel);
    }

    /* Auto-tail the most recent l_rows entries from the timestamped ring buffer */
    cs_draw_box(&app->scr, x + fw, fy, lw2, bot_h, 0, "ACTIVITY & DEPLOYMENT LOG", NULL,
                pal->fg_border, pal->fg_title, pal->fg_muted, pal->bg_panel);
    int l_rows = bot_h - 2;
    int start  = (app->n_logs > l_rows) ? (app->n_logs - l_rows) : 0;
    for (int i = 0; i < l_rows && (start + i) < app->n_logs; i++)
    {
        cs_printf_at(&app->scr, x + fw + 2, fy + 1 + i, lw2 - 4, pal->fg_text, pal->bg_panel, 0u, 1,
                     "%s", app->logs[start + i]);
    }
}

/* ── F6:SOURCE — Live Formatted .bench INI/TOML Preview ───────────────────── */

/**
 * cs_render_view_source() - Render the `F6:SOURCE` syntax-highlighted `.bench` INI/TOML preview
 * @app: Master TUI application state (`app->cfg` serialized; `app->source_scroll` clamped)
 * @x:   Viewport left terminal column (0-based)
 * @y:   Viewport top terminal row (0-based)
 * @w:   Viewport width in columns
 * @h:   Viewport height in rows
 *
 * Role:
 *   Serializes the live in-memory `app->cfg` via `cacaosim_bench_format_str()` and renders a
 *   scrollable, line-numbered, syntax-colored `.bench` source listing alongside quick action
 *   buttons for saving (`[s:Save File]`) or editing in `$VISUAL` / `$EDITOR`
 *   (`[e:Open in $EDITOR]`).
 *
 * Rationale:
 *   Power users often want to verify exact key names, section headers (`[dm.tweeter]`), or comment
 *   annotations produced by GUI edits before committing `.bench` files to Git, or jump into Vim /
 *   Emacs for bulk text edits. Using two-pass `cacaosim_bench_format_str()` (`NULL` size query
 *   followed by exact buffer allocation;note this runs only in the interactive ~8 Hz UI tab, never
 *   in an AO RTC hot loop) guarantees arbitrary-length bench files never truncate, while simple
 *   lexical coloring (`#` comments, `[section]` headers, `key = value` splits) maximizes
 *   readability.
 */
void cs_render_view_source(cs_tui_app_t *app, int x, int y, int w, int h)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    cs_draw_box(&app->scr, x, y, w, h, 1, "DECLARATIVE .bench SOURCE PREVIEW (INI/TOML)",
                "Up/Down/PgUp/PgDn:Scroll", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                pal->bg_panel);

    int bx = x + 2;
    bx += view_btn(app, bx, y + 1, "[s:Save File]", CS_HIT_ACT_SAVE, 0, pal->fg_active,
                   pal->bg_panel_alt, "Save formatted .bench configuration to disk [s]") +
          2;
    view_btn(app, bx, y + 1, "[e:Open in $EDITOR]", CS_HIT_ACT_EDITOR, 0, pal->fg_accent,
             pal->bg_panel_alt, "Save and edit .bench text directly in $VISUAL / $EDITOR [e]");

    /* Query exact serialized byte length first so arbitrarily large benches never truncate */
    int need = cacaosim_bench_format_str(&app->cfg, NULL, 0u);
    if (need <= 0)
    {
        return;
    }
    char *src = (char *) malloc((size_t) need + 2u);
    if (src == NULL)
    {
        return;
    }
    cacaosim_bench_format_str(&app->cfg, src, (size_t) need + 2u);

    /* Count total lines to clamp vertical scroll offset cleanly */
    int total_lines = 1;
    for (const char *p = src; *p != '\0'; p++)
    {
        if (*p == '\n')
        {
            total_lines++;
        }
    }

    int view_rows = h - 3;
    if (app->source_scroll > total_lines - view_rows)
    {
        app->source_scroll = total_lines - view_rows;
    }
    if (app->source_scroll < 0)
    {
        app->source_scroll = 0;
    }

    /* Walk lines and apply lightweight INI/TOML lexical coloring (#, [section], key = val) */
    int         lineno = 0;
    const char *cur    = src;
    while (*cur != '\0')
    {
        const char *nl   = strchr(cur, '\n');
        size_t      llen = (nl != NULL) ? (size_t) (nl - cur) : strlen(cur);
        if (lineno >= app->source_scroll && lineno < app->source_scroll + view_rows)
        {
            int  ry = y + 2 + (lineno - app->source_scroll);
            char lbuf[256];
            if (llen >= sizeof(lbuf))
            {
                llen = sizeof(lbuf) - 1u;
            }
            memcpy(lbuf, cur, llen);
            lbuf[llen] = '\0';

            cs_printf_at(&app->scr, x + 2, ry, 5, pal->fg_dim, pal->bg_panel, 0u, 1, "%3d ",
                         lineno + 1);
            if (lbuf[0] == '#')
            {
                cs_put_str(&app->scr, x + 7, ry, w - 9, lbuf, pal->fg_dim, pal->bg_panel, 0u, 1);
            }
            else if (lbuf[0] == '[')
            {
                cs_put_str(&app->scr, x + 7, ry, w - 9, lbuf, pal->fg_accent, pal->bg_panel,
                           CS_ATTR_BOLD, 1);
            }
            else
            {
                char *eq = strchr(lbuf, '=');
                if (eq != NULL)
                {
                    *eq = '\0';
                    int kw = cs_put_str(&app->scr, x + 7, ry, 24, lbuf, pal->fg_fps, pal->bg_panel,
                                        0u, 1);
                    cs_put_str(&app->scr, x + 7 + kw, ry, 2, "=", pal->fg_muted, pal->bg_panel, 0u,
                               1);
                    cs_put_str(&app->scr, x + 8 + kw, ry, w - 10 - kw, eq + 1, pal->fg_stream,
                               pal->bg_panel, 0u, 1);
                }
                else
                {
                    cs_put_str(&app->scr, x + 7, ry, w - 9, lbuf, pal->fg_text, pal->bg_panel, 0u,
                               1);
                }
            }
        }
        lineno++;
        if (nl == NULL)
        {
            break;
        }
        cur = nl + 1;
    }

    free(src);
}
