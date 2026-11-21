// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_render.c
 * @brief   2D topological schematic router, optical scale engine, F2:BUILDER, chrome, and modals
 *
 * Architectural Role in `cacaosim-CTRL`:
 *   Implements the core visual compositor and optical layout engine for `cacaosim-CTRL`, turning
 *   declarative `cacaosim_bench_cfg_t` topologies and live `MILK_SHM_DIR` telemetry into an
 *   interactive `milk-CTRL`-style terminal workspace:
 *     1. Physical Pupil & Focal-Plane Scale Engine (`node_focal_scale()`,
 *        `cs_tui_compute_scales()`): Computes pupil spatial sampling (`d_beam_px`, `m/pix`) and
 *        per-instrument focal-plane angular sampling (`lambda/D/pix` and `arcsec/pix` on sky,
 *        where `1 arcsec = 1000 mas`), covering padded FFTs, Soummer Matrix-DFTs (FPCAM, Lyot /
 *        dimple / custom coronagraphs, ZWFS), and Shack-Hartmann subaperture lenslet arrays.
 *     2. Interactive 2D Topological DAG Auto-Layout & Orthogonal Beam Router (`dag_draw_t`,
 *        `dc_put()`, `dc_draw_node_chip()`, `dc_draw_harrow()`, `dc_draw_vbranch()`,
 *        `dc_trace_beam()`, `cs_render_dag_canvas()`): Places the `[TEL]` entrance source and
 *        traces named optical beams recursively onto a 2D logical character grid, keeping inline
 *        and transmitted paths colinear on horizontal rails while dropping reflective splitter
 *        arms (`out_refl`), coronagraphic LOWFS pickoffs (`out_lowfs`), and secondary fan-out
 *        consumers onto vertical branches, terminating each sensor with a 32-bin polychromatic
 *        (`0.40 .. 2.40 um`) spectrum sparkline and `ph/fr -> out_cam` flux readout.
 *     3. `F2:BUILDER` 3-Pane Configurator (`render_view_builder()`): Renders the resizable left
 *        `ELEMENTS & ARRANGER` list (with splicing toolbar, live SHM status dots, and docked
 *        16x16 anti-aliased pupil + physical scale summary), top-right `2D OPTICAL SCHEMATIC`,
 *        and bottom-right categorized `INSPECTOR` (`1:OPTICS`, `2:DYN/DET`, `3:SHM/WIRING`).
 *     4. Global Chrome, Floating Modals & Hover Tooltips (`render_header_and_tabs()`,
 *        `render_bottom_bars()`, `render_modal()`, `render_hover_tooltip()`,
 *        `cs_tui_render_frame()`): Draws the `milk-CTRL` header/tab/status bars, all 8 modal
 *        overlays (Add/Splice wizard, Beam Wire picker, Preset gallery, Theme selector, Save/Open
 *        file prompt, Unsaved confirmation, Help cheatsheet), and mouse hover tooltips.
 *
 * Design Rationale & Motivation:
 *   - Physical Optical-Table Layout Convention: Real AO benches propagate transmitted relays
 *     (`out_beam`, `out_trans`) colinearly along horizontal optical rails while folding dichroic
 *     reflections (`out_refl`) and focal/Lyot LOWFS pickoffs (`out_lowfs`) onto perpendicular
 *     arms. Encoding that exact convention in `dc_trace_beam()` yields deterministic, crossing-free
 *     schematics for complex multi-stage coronagraphic benches without iterative graph-layout
 *     oscillations.
 *   - Two-Pass `measure_only` Viewport Tracking: Running a zero-draw `measure_only = 1` layout
 *     pass before rendering automatically pans `(schem_scroll_x, schem_scroll_y)` to keep the
 *     selected element card and its output spectrum visible while navigating `ELEMENTS` or
 *     `INSPECTOR`, yet yields full manual panning when focus enters `CS_FOCUS_SCHEM`.
 *   - Design-Time Nyquist & Plate-Scale Verification: Surfacing `d_beam_px`, `m/pix`,
 *     `lambda/D/pix`, and `arcsec/pix` live alongside the 16x16 pupil preview lets engineers catch
 *     FFT Nyquist undersampling (`pup_diam * pad < 2`), coarse DM actuator sampling, or mismatched
 *     WFS/coronagraph/camera fields of view before launching heavy GPU/CPU simulations.
 *   - Immediate-Mode Hit-Box Registration (`cs_tui_add_hit()`): Recording viewport-clipped click
 *     and tooltip rectangles at draw time eliminates brittle inverse coordinate formulas across
 *     terminal resizes, draggable pane dividers, and scrolled 2D schematics.
 */

#include <ctype.h>
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bench/cacaosim_bench_schema.h"
#include "bench/cacaosim_tui_internal.h"

/* ── Compact Formatting & Node Metadata Helpers ───────────────────────────── */

/**
 * fmt_flux_compact() - Format a photon count or photon rate into a short human-readable string
 * @ph:  Photon count (`[ph/fr]`) or photon flux (`[ph/s]`; `<= 0` formats as `"0 ph"`)
 * @out: Destination character buffer (non-NULL)
 * @cap: Capacity of @out in bytes
 *
 * Role:
 *   Formats stellar entrance flux and per-sensor integrated photon counts into compact strings
 *   (`"0 ph"`, `"850"`, `"12.4k"`, `"3.45M"`, `"1.20e9"`) for 2D schematic sensor tails, pupil
 *   dock readouts, and the `CS_MODAL_WIRE` beam picker.
 *
 * Rationale:
 *   Schematic sensor annotations share horizontal space with node cards and 32-bin sparklines;
 *   4-6 character SI-scaled strings preserve 2-3 significant digits across 10 orders of magnitude
 *   without widening the schematic canvas.
 */
static void fmt_flux_compact(float ph, char *out, size_t cap)
{
    if (ph <= 0.0f)
    {
        snprintf(out, cap, "0 ph");
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
 * node_badge() - Return a short 2-to-4 character hardware category tag for an optical node
 * @nd: Optical element configuration descriptor (non-NULL)
 *
 * Role:
 *   Provides the uppercase type badge (`"DM"`, `"TTM"`, `"BS"`, `"CORO"`, `"PYR"`, `"SH"`,
 *   `"ZWFS"`, `"CAM"`, `"WFE"`) rendered in `ELEMENTS & ARRANGER` rows, 2D schematic chip headers,
 *   and hover tooltips.
 *
 * Rationale:
 *   Distinguishing `"TTM"` (`dm_mode == 1`) from zonal `"DM"` (`dm_mode == 0`) at the badge level
 *   reflects their distinct control roles (2-axis tip-tilt vs. high-order wavefront correction)
 *   even though both share `CACAOSIM_NODE_DM` (`cacaosim.dmsim`) underneath.
 *
 * Return: Pointer to a static NUL-terminated uppercase tag literal.
 */
static const char *node_badge(const cacaosim_node_cfg_t *nd)
{
    switch (nd->kind)
    {
    case CACAOSIM_NODE_DM:
        return (nd->dm.dm_mode == 1) ? "TTM" : "DM";
    case CACAOSIM_NODE_SPLITTER:
        return "BS";
    case CACAOSIM_NODE_CORO:
        return "CORO";
    case CACAOSIM_NODE_PYRWFS:
        return "PYR";
    case CACAOSIM_NODE_SHWFS:
        return "SH";
    case CACAOSIM_NODE_ZWFS:
        return "ZWFS";
    case CACAOSIM_NODE_FPCAM:
        return "CAM";
    case CACAOSIM_NODE_WFE:
        return "WFE";
    }
    return "NODE";
}

/**
 * node_color() - Select the semantic theme accent color for an optical element's category
 * @nd:  Optical element configuration descriptor (non-NULL)
 * @pal: Active `milk-CTRL` RGB theme palette (non-NULL)
 *
 * Role:
 *   Maps each subsystem family onto a consistent semantic color from @pal: active correctors
 *   (`DM`/`TTM` -> `fg_accent`), beamsplitters (`BS` -> `fg_proc`), phase/coronagraphic masks
 *   (`CORO`/`WFE` -> `fg_warn`), wavefront sensors (`PYR`/`SH`/`ZWFS` -> `fg_stream`), and focal
 *   science cameras (`CAM` -> `fg_active`).
 *
 * Rationale:
 *   Color-coding by physical role lets the eye immediately separate active mirrors, dichroic
 *   routing optics, WFS control sensors, and science cameras in both the 1D arranger list and the
 *   2D schematic across all five color themes (including colorblind-safe `accessible`).
 *
 * Return: RGB color value from @pal for borders and badge text.
 */
static cs_tui_rgb_t node_color(const cacaosim_node_cfg_t *nd, const cs_tui_pal_t *pal)
{
    switch (nd->kind)
    {
    case CACAOSIM_NODE_DM:
        return pal->fg_accent;
    case CACAOSIM_NODE_SPLITTER:
        return pal->fg_proc;
    case CACAOSIM_NODE_CORO:
    case CACAOSIM_NODE_WFE:
        return pal->fg_warn;
    case CACAOSIM_NODE_PYRWFS:
    case CACAOSIM_NODE_SHWFS:
    case CACAOSIM_NODE_ZWFS:
        return pal->fg_stream;
    case CACAOSIM_NODE_FPCAM:
        return pal->fg_active;
    }
    return pal->fg_text;
}

/**
 * node_short_spec() - Format a one-line hardware specification subtitle for an optical node
 * @cfg: Parent bench configuration supplying fallback `master_hz` (non-NULL)
 * @nd:  Optical element configuration to summarize (non-NULL)
 * @out: Destination character buffer (non-NULL)
 * @cap: Capacity of @out in bytes
 *
 * Role:
 *   Generates the concise (< 22 char) subtitle displayed on line 2 of each 2D schematic node chip
 *   and in the right column of `ELEMENTS & ARRANGER` rows (e.g., `"32x32 +/-5u"`, `"LP 0.95u"`,
 *   `"Vortex l=2+M"`, `"128px 0.75u 2000Hz"`, `"50nm [OPD m]"`).
 *
 * Rationale:
 *   Surfacing each element's defining physical quantities (actuator pitch/stroke, dichroic mode &
 *   cutoff wavelength, coronagraph mask/charge + `+M` custom mask indicator, sensor pixel count +
 *   wavelength + effective frame rate `rate_hz` or `master_hz / navg`, and scaled WFE RMS) allows
 *   operators to audit the entire bench directly from the schematic without clicking into every
 *   node's inspector.
 */
static void node_short_spec(
    const cacaosim_bench_cfg_t *cfg,
    const cacaosim_node_cfg_t  *nd,
    char                       *out,
    size_t                      cap)
{
    float mhz = (cfg->master_hz > 0.0f) ? cfg->master_hz : 2000.0f;
    switch (nd->kind)
    {
    case CACAOSIM_NODE_DM:
        if (nd->dm.dm_mode == 1)
        {
            snprintf(out, cap, "2-ax +/-%.0fu", nd->dm.max_stroke);
        }
        else
        {
            snprintf(out, cap, "%ux%u +/-%.0fu", nd->dm.nact_x, nd->dm.nact_y, nd->dm.max_stroke);
        }
        break;
    case CACAOSIM_NODE_SPLITTER:
        if (nd->split.mode == CACAOSIM_SPLIT_GRAY)
        {
            snprintf(out, cap, "%.0f/%.0f gray", nd->split.r_gray * 100.0f,
                     nd->split.t_gray * 100.0f);
        }
        else if (nd->split.mode == CACAOSIM_SPLIT_DICHROIC_BP)
        {
            snprintf(out, cap, "BP %.2fu", nd->split.lambda_cut);
        }
        else
        {
            snprintf(out, cap, "%s %.2fu",
                     (nd->split.mode == CACAOSIM_SPLIT_DICHROIC_SP) ? "SP" : "LP",
                     nd->split.lambda_cut);
        }
        break;
    case CACAOSIM_NODE_CORO:
    {
        /* Append "+M" whenever one or more custom FITS/SHM pupil, FPM, or Lyot masks are bound */
        int has_cmask = (nd->in_pup_amp[0] != '\0' || nd->in_pup_pha[0] != '\0' ||
                         nd->in_fpm_amp[0] != '\0' || nd->in_fpm_pha[0] != '\0' ||
                         nd->in_lyot[0] != '\0' || nd->in_lyot_pha[0] != '\0');
        if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            if (nd->coro.nlambda > 1)
            {
                snprintf(out, cap, "CustMFT %dch %.2fu", nd->coro.nlambda, nd->coro.lambda_um);
            }
            else
            {
                snprintf(out, cap, "CustMFT %.2fu", nd->coro.lambda_um);
            }
        }
        else if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_FFT)
        {
            if (nd->coro.nlambda > 1)
            {
                snprintf(out, cap, "CustFFT %dch %.2fu", nd->coro.nlambda, nd->coro.lambda_um);
            }
            else
            {
                snprintf(out, cap, "CustFFT %.2fu", nd->coro.lambda_um);
            }
        }
        else if (nd->coro.mode == CACAOSIM_CORO_VORTEX)
        {
            snprintf(out, cap, "Vortex l=%d%s", nd->coro.vortex_charge, has_cmask ? "+M" : "");
        }
        else if (nd->coro.mode == CACAOSIM_CORO_4QPM)
        {
            snprintf(out, cap, "4QPM %.2fu%s", nd->coro.lambda_um, has_cmask ? "+M" : "");
        }
        else if (nd->coro.mode == CACAOSIM_CORO_PHASE_DIMPLE)
        {
            snprintf(out, cap, "Dimple %.1fl/D%s", nd->coro.fpm_rad_ld, has_cmask ? "+M" : "");
        }
        else
        {
            snprintf(out, cap, "Lyot %.1fl/D%s", nd->coro.fpm_rad_ld, has_cmask ? "+M" : "");
        }
        break;
    }
    case CACAOSIM_NODE_PYRWFS:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%upx %.2fu %.0fHz", nd->pyrwfs.ncam, nd->pyrwfs.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_SHWFS:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%ux%u %.2fu %.0fHz", nd->shwfs.n_sub, nd->shwfs.n_sub,
                 nd->shwfs.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_ZWFS:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%upx %.2fu %.0fHz", nd->zwfs.ncam, nd->zwfs.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_FPCAM:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%upx %.2fu %.0fHz", nd->fpcam.ncam, nd->fpcam.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_WFE:
        if (fabsf(nd->wfe_defocus_nm) > 1.0e-3f && nd->wfe_rms_nm <= 1.0e-3f)
        {
            snprintf(out, cap, "Z4=%+.0fnm [OPD m]", nd->wfe_defocus_nm * nd->wfe_gain);
        }
        else if (fabsf(nd->wfe_defocus_nm) > 1.0e-3f)
        {
            snprintf(out, cap, "%.0fnm Z4=%+.0fnm", nd->wfe_rms_nm * fabsf(nd->wfe_gain),
                     nd->wfe_defocus_nm * nd->wfe_gain);
        }
        else
        {
            snprintf(out, cap, "%.0fnm [OPD m]", nd->wfe_rms_nm * fabsf(nd->wfe_gain));
        }
        break;
    }
}

/**
 * node_focal_scale() - Compute the focal-plane angular sampling (`lambda/D` per pixel) of a node
 * @cfg:        Parent bench configuration providing normalized pupil diameter `pup_diam` (non-NULL)
 * @nd:         Candidate optical element configuration (non-NULL)
 * @out_ld_pix: Output focal-plane plate scale in units of full-aperture `lambda/D` per pixel
 * @out_lam_um: Output reference central wavelength `[um]` at which @out_ld_pix is defined
 *
 * Role:
 *   Evaluates the exact numerical diffraction-propagator focal sampling for elements that form an
 *   internal or detector focal plane (`FPCAM`, `CORO`, `SHWFS`, `PYRWFS`, `ZWFS`), returning `0`
 *   for purely pupil-plane elements (`DM`, `SPLITTER`, `WFE`).
 *
 * Rationale:
 *   Each `cacaosim` propagator parameterizes its focal grid differently:
 *     - Padded full-pupil FFT (`FPCAM` FFT mode, `PYRWFS`, and `CORO` vortex/4QPM/custom_fft):
 *       A beam of diameter `D_px = pup_diam * N` zero-padded to `N_fft = pad_factor * N` yields
 *       `Delta_foc = 1 / (pup_diam * pad_factor)` `lambda/D` per pixel (Nyquist-sampled at `<= 0.5`
 *       `lambda/D/pix` whenever `pup_diam * pad_factor >= 2`).
 *     - Soummer Matrix-DFT (`FPCAM` MFT & `CORO` `custom_mft`): Configured directly in pixels per
 *       `lambda/D` (`pixscale_ld`), so `Delta_foc = 1 / pixscale_ld` `lambda/D/pix`.
 *     - Analytic Babinet MFT masks (`CORO` Lyot spot / phase dimple & `ZWFS`): Sample a compact
 *       focal box spanning `2.4 * fpm_rad_ld` (or `1.2 * dimple_ld` diameter) across `n_mft`
 *       pixels, giving `(2.4 * fpm_rad_ld) / n_mft` (or `(1.2 * dimple_ld) / n_mft`) in
 *       `lambda/D/pix`.
 *     - Shack-Hartmann lenslet FFT (`SHWFS`): Each subaperture has diameter `d_sub = D / n_sub`,
 *       so `1 / pad_factor` in subaperture `(lambda / d_sub)/pix` equals `n_sub / pad_factor` in
 *       full-telescope `(lambda / D)/pix`.
 *   Unifying all five conventions into full-telescope `lambda/D/pix` lets `cs_tui_compute_scales()`
 *   convert any selected instrument directly into on-sky arcseconds/milliarcseconds per pixel.
 *
 * Return: 1 if @nd defines a focal plane (@out_ld_pix and @out_lam_um written), 0 otherwise.
 */
static int node_focal_scale(
    const cacaosim_bench_cfg_t *cfg,
    const cacaosim_node_cfg_t  *nd,
    float                      *out_ld_pix,
    float                      *out_lam_um)
{
    float pdia = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 1.0f;
    switch (nd->kind)
    {
    case CACAOSIM_NODE_FPCAM:
        *out_lam_um = (nd->fpcam.lambda_um > 0.1f) ? nd->fpcam.lambda_um : 1.65f;
        if (nd->fpcam.prop_mode == CACAOSIM_FPCAM_PROP_FFT)
        {
            /* Padded FFT: 1 / (pup_diam * pad_factor) lambda/D per detector pixel */
            float pad   = (nd->fpcam.pad_factor >= 2u) ? (float) nd->fpcam.pad_factor : 2.0f;
            *out_ld_pix = 1.0f / (pdia * pad);
        }
        else
        {
            /* Matrix-DFT: user specifies pixels per lambda/D, invert to lambda/D per pixel */
            float px    = (nd->fpcam.pixscale_ld > 0.05f) ? nd->fpcam.pixscale_ld : 3.0f;
            *out_ld_pix = 1.0f / px;
        }
        return 1;
    case CACAOSIM_NODE_CORO:
        *out_lam_um = (nd->coro.lambda_um > 0.1f) ? nd->coro.lambda_um : 1.65f;
        if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            float px    = (nd->coro.fpm_pixscale_ld > 0.05f) ? nd->coro.fpm_pixscale_ld : 8.0f;
            *out_ld_pix = 1.0f / px;
        }
        else if (nd->coro.mode == CACAOSIM_CORO_VORTEX || nd->coro.mode == CACAOSIM_CORO_4QPM ||
                 nd->coro.mode == CACAOSIM_CORO_CUSTOM_FFT)
        {
            float pad   = (nd->coro.pad_factor >= 2u) ? (float) nd->coro.pad_factor : 4.0f;
            *out_ld_pix = 1.0f / (pdia * pad);
        }
        else
        {
            /* Analytic Babinet spot/dimple MFT grid spans 2.4 * fpm_rad_ld across n_mft pixels */
            float r_ld  = (nd->coro.fpm_rad_ld > 0.1f) ? nd->coro.fpm_rad_ld : 2.5f;
            float nmft  = (nd->coro.n_mft >= 8u) ? (float) nd->coro.n_mft : 32.0f;
            *out_ld_pix = (2.4f * r_ld) / nmft;
        }
        return 1;
    case CACAOSIM_NODE_SHWFS:
    {
        /* Lenslet d_sub = D / n_sub -> (1 / pad) (lambda / d_sub) = (n_sub / pad) (lambda / D) */
        float nsub  = (nd->shwfs.n_sub >= 1u) ? (float) nd->shwfs.n_sub : 16.0f;
        float pad   = (nd->shwfs.pad_factor >= 1u) ? (float) nd->shwfs.pad_factor : 2.0f;
        *out_lam_um = (nd->shwfs.lambda_um > 0.1f) ? nd->shwfs.lambda_um : 0.75f;
        *out_ld_pix = nsub / pad;
        return 1;
    }
    case CACAOSIM_NODE_PYRWFS:
    {
        float pad   = (nd->pyrwfs.pad_factor >= 2u) ? (float) nd->pyrwfs.pad_factor : 2.0f;
        *out_lam_um = (nd->pyrwfs.lambda_um > 0.1f) ? nd->pyrwfs.lambda_um : 0.75f;
        *out_ld_pix = 1.0f / (pdia * pad);
        return 1;
    }
    case CACAOSIM_NODE_ZWFS:
    {
        /* Babinet ZWFS MFT grid spans 1.2 * dimple_ld diameter across n_mft focal pixels */
        float d_ld  = (nd->zwfs.dimple_ld > 0.1f) ? nd->zwfs.dimple_ld : 1.06f;
        float nmft  = (nd->zwfs.n_mft >= 8u) ? (float) nd->zwfs.n_mft : 32.0f;
        *out_lam_um = (nd->zwfs.lambda_um > 0.1f) ? nd->zwfs.lambda_um : 0.85f;
        *out_ld_pix = (1.2f * d_ld) / nmft;
        return 1;
    }
    default:
        return 0;
    }
}

/**
 * cs_tui_compute_scales() - Compute pupil spatial and focal angular pixel scales for UI display
 * @cfg:      Bench configuration supplying `nwf`, `pup_diam`, `tel_diam_m`, and nodes (nullable)
 * @sel_item: Currently selected element (`0` = `[bench]`, `1..n_nodes` = `cfg->nodes[sel_item-1]`)
 * @out:      Destination `cs_optical_scales_t` struct receiving computed scales (no-op if NULL)
 *
 * Role:
 *   Populates `d_beam_px` (`[pix]` across primary diameter), `pup_m_pix` (`[m/pix]` in the entrance
 *   pupil), `foc_ld_pix` (`[lambda/D/pix]`), `ref_lam_um` (`[um]`), `foc_src` (source node ID or
 *   `"2xFFT"`), and `foc_arcsec_pix` (`[arcsec/pix]` on sky, i.e., `1e-3 * mas/pix`) for both the
 *   `F2:BUILDER` pupil dock and the `F3:SCHEMATIC` bottom deck.
 *
 * Rationale:
 *   When the user clicks a focal-plane node (`FPCAM`, `CORO`, `SHWFS`, `PYRWFS`, `ZWFS`), the
 *   readout dynamically tracks that specific node's focal plate scale (`foc_src = nd->id`) so the
 *   user can verify Nyquist sampling and detector FoV immediately while stepping `pad_factor`,
 *   `pixscale_ld`, or `n_sub`. When `[bench]` or a pure pupil-plane optic (`DM`, `BS`, `WFE`) is
 *   selected, a deterministic priority search (`FPCAM -> CORO -> SHWFS -> PYRWFS -> ZWFS -> 2xFFT`)
 *   anchors the focal readout to the bench's primary science camera or wavefront sensor.
 */
void cs_tui_compute_scales(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    cs_optical_scales_t        *out)
{
    if (out == NULL)
    {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (cfg == NULL)
    {
        return;
    }

    /* Pupil-plane geometry: beam diameter in pixels and physical meters per wavefront pixel */
    float nwf_f     = (cfg->nwf > 0u) ? (float) cfg->nwf : 128.0f;
    float pdia      = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 1.0f;
    float tel_d     = (cfg->tel_diam_m > 0.05f) ? cfg->tel_diam_m : 8.2f;
    out->d_beam_px  = pdia * nwf_f;
    out->pup_m_pix  = tel_d / out->d_beam_px;
    out->foc_ld_pix = 1.0f / (pdia * 2.0f);
    out->ref_lam_um = (cfg->ref_lambda > 0.1f) ? cfg->ref_lambda : 1.65f;
    snprintf(out->foc_src, sizeof(out->foc_src), "2xFFT");

    /* 1. Prefer the actively selected element if it forms an internal or detector focal plane */
    int found = 0;
    if (sel_item > 0 && (uint32_t) (sel_item - 1) < cfg->n_nodes)
    {
        const cacaosim_node_cfg_t *nd = &cfg->nodes[sel_item - 1];
        if (node_focal_scale(cfg, nd, &out->foc_ld_pix, &out->ref_lam_um))
        {
            snprintf(out->foc_src, sizeof(out->foc_src), "%s", nd->id);
            found = 1;
        }
    }

    /* 2. Otherwise fall back to the highest-priority focal element in the bench (SciCam first) */
    if (!found)
    {
        static const cacaosim_node_kind_t prio[5] = {
            CACAOSIM_NODE_FPCAM, CACAOSIM_NODE_CORO,   CACAOSIM_NODE_SHWFS,
            CACAOSIM_NODE_PYRWFS, CACAOSIM_NODE_ZWFS
        };
        for (int p = 0; p < 5 && !found; p++)
        {
            for (uint32_t i = 0; i < cfg->n_nodes; i++)
            {
                if (cfg->nodes[i].kind == prio[p] &&
                    node_focal_scale(cfg, &cfg->nodes[i], &out->foc_ld_pix, &out->ref_lam_um))
                {
                    snprintf(out->foc_src, sizeof(out->foc_src), "%s", cfg->nodes[i].id);
                    found = 1;
                    break;
                }
            }
        }
    }

    /* Convert diffraction units (lambda/D) to sky arcseconds (1 rad = 206264.8062 arcsec) */
    float ld_arcsec     = (out->ref_lam_um * 1.0e-6f / tel_d) * 206264.8062f;
    out->foc_arcsec_pix = out->foc_ld_pix * ld_arcsec;
}

/**
 * find_analyzed_node() - Look up a node's precomputed zero-FFT telemetry record by section ID
 * @app:     Master TUI application state holding `app->ctx` (non-NULL)
 * @node_id: Node identifier string to match (nullable)
 *
 * Role:
 *   Retrieves the `cacaosim_bench_node_t` produced by `cacaosim_bench_analyze()` so schematic
 *   sensor annotations can display `eff_flux_ph` (`[ph/fr]`) even though topological sort order in
 *   `app->ctx.nodes[]` may differ from user declaration order in `app->cfg.nodes[]`.
 *
 * Rationale:
 *   `app->cfg.nodes[]` preserves the user's editable arranger order, whereas `app->ctx.nodes[]`
 *   stores topologically sorted execution stages; matching by `node_id` bridges the two cleanly
 *   whenever `app->dag_valid` holds.
 *
 * Return: Pointer to the analyzed node in `app->ctx.nodes[]`, or NULL if invalid/not found.
 */
static const cacaosim_bench_node_t *find_analyzed_node(
    const cs_tui_app_t *app,
    const char         *node_id)
{
    if (!app->dag_valid || node_id == NULL)
    {
        return NULL;
    }
    for (uint32_t i = 0; i < app->ctx.n_nodes; i++)
    {
        if (strcmp(app->ctx.nodes[i].cfg.id, node_id) == 0)
        {
            return &app->ctx.nodes[i];
        }
    }
    return NULL;
}

/**
 * find_beam_spec() - Look up the 32-bin polychromatic photon spectrum of a named optical beam
 * @app:       Master TUI application state holding `app->ctx.beams[]` (non-NULL)
 * @beam_name: Optical beam identifier (e.g., `"b_pup"`, `"b_sci"`, nullable)
 *
 * Role:
 *   Fetches the `cacaosim_spec_t` (`0.40 .. 2.40 um` photon-rate histogram) propagated to
 *   @beam_name so terminal sensors in the 2D schematic and rows in `CS_MODAL_WIRE` can draw
 *   accurate wavelength-colored spectral sparklines.
 *
 * Rationale:
 *   Reading directly from the cached zero-FFT `app->ctx.beams[]` table lets both schematic
 *   rendering and interactive beam-wiring modals display instant spectral passbands with zero
 *   per-frame recomputation.
 *
 * Return: Pointer to the beam's `cacaosim_spec_t`, or NULL if the DAG is invalid or unmatched.
 */
static const cacaosim_spec_t *find_beam_spec(const cs_tui_app_t *app, const char *beam_name)
{
    if (!app->dag_valid || beam_name == NULL || beam_name[0] == '\0')
    {
        return NULL;
    }
    for (uint32_t i = 0; i < app->ctx.n_beams; i++)
    {
        if (strcmp(app->ctx.beams[i].name, beam_name) == 0)
        {
            return &app->ctx.beams[i].spec;
        }
    }
    return NULL;
}

/**
 * node_primary_shm_live() - Query the live `MILK_SHM_DIR` status of an element's primary stream
 * @app: Master TUI application state containing scanned `app->shm` entries (non-NULL)
 * @nd:  Optical element configuration to inspect (non-NULL)
 *
 * Role:
 *   Maps hardware elements that bind a primary external `ImageStreamIO` stream (`DM` -> `in_cmd`,
 *   `WFE` -> `in_wfe`, `PYRWFS`/`SHWFS`/`ZWFS`/`FPCAM` -> `out_cam`) to a tri-state liveness code
 *   rendered as a status dot (`●` / `○`) in `ELEMENTS & ARRANGER`.
 *
 * Rationale:
 *   Giving each mirror and detector a live indicator dot directly inside `F2:BUILDER` lets the
 *   operator verify at a glance whether `InitSHM` or `Deploy` has created and started streaming
 *   each element's SHM endpoint without leaving the builder tab.
 *
 * Return: `-1` if the node has no primary external SHM stream (`BS`, `CORO`), `0` if offline /
 *         missing, `1` if present on disk but static, or `2` if actively written (`alive`).
 */
static int node_primary_shm_live(const cs_tui_app_t *app, const cacaosim_node_cfg_t *nd)
{
    const char *s = NULL;
    if (nd->kind == CACAOSIM_NODE_DM)
    {
        s = nd->in_cmd;
    }
    else if (nd->kind == CACAOSIM_NODE_WFE)
    {
        s = nd->in_wfe;
    }
    else if (nd->kind == CACAOSIM_NODE_PYRWFS || nd->kind == CACAOSIM_NODE_SHWFS ||
             nd->kind == CACAOSIM_NODE_ZWFS || nd->kind == CACAOSIM_NODE_FPCAM)
    {
        s = nd->out_cam;
    }
    if (s == NULL || s[0] == '\0')
    {
        return -1;
    }
    int idx = cs_shm_find_stream(&app->shm, s);
    if (idx < 0 || !app->shm.streams[idx].exists)
    {
        return 0;
    }
    return app->shm.streams[idx].alive ? 2 : 1;
}

/* ── Interactive 2D Optical Schematic Canvas ──────────────────────────────── */

/**
 * struct dag_draw_t - Ephemeral layout and viewport-clipping context for 2D schematic rendering
 * @app:          Master TUI application state (provides scroll offsets, selection, and cell buffer)
 * @pal:          Active `milk-CTRL` RGB theme palette
 * @vx:           Viewport left screen column (0-based)
 * @vy:           Viewport top screen row (0-based)
 * @vw:           Viewport width in screen columns
 * @vh:           Viewport height in screen rows
 * @wide_spec:    Non-zero to render 24-column terminal sensor sparklines; 0 for compact 14-column
 * @measure_only: Non-zero during pass 1 to compute `(sel_lx, sel_ly, sel_bw)` without writing cells
 * @sel_found:    Set to 1 during traversal when `app->sel_item` is located on the logical canvas
 * @sel_lx:       Logical X column of the selected element chip
 * @sel_ly:       Logical Y row of the selected element chip
 * @sel_bw:       Width in columns of the selected element chip
 * @visited:      Per-node visited bitmap preventing duplicate placement or infinite loop cycles
 * @next_row:     Monotonically increasing logical optical-rail slot (`ly = row_slot * 5`) allocated
 *                whenever a beamsplitter reflection, LOWFS pickoff, or fan-out drops a new branch
 *
 * Role:
 *   Carries all coordinate transforms, cycle-guard bitmaps, vertical rail allocation counters, and
 *   pass-1 measurement outputs across the recursive `dc_trace_beam()` schematic layout traversal.
 */
typedef struct
{
    cs_tui_app_t       *app;
    const cs_tui_pal_t *pal;
    int                 vx;
    int                 vy;
    int                 vw;
    int                 vh;
    int                 wide_spec;
    int                 measure_only;
    int                 sel_found;
    int                 sel_lx;
    int                 sel_ly;
    int                 sel_bw;
    uint8_t             visited[CACAOSIM_BENCH_MAX_NODES];
    int                 next_row;
} dag_draw_t;

/**
 * dc_put() - Clip and blit a UTF-8 string from logical DAG coordinates onto the screen viewport
 * @d:     Active schematic drawing context (no-op when `d->measure_only` is non-zero)
 * @lx:    Logical canvas column (0-based, relative to schematic origin before scroll)
 * @ly:    Logical canvas row (0-based, relative to schematic origin before scroll)
 * @str:   UTF-8 / ASCII string to plot cell-by-cell (non-NULL)
 * @fg:    Foreground RGB color
 * @bg:    Background RGB fill color
 * @attrs: Cell attribute bitmask (`CS_ATTR_BOLD`, `CS_ATTR_DIM`, etc.)
 *
 * Role:
 *   Translates unbounded logical schematic coordinates `(lx, ly)` by `(-schem_scroll_x,
 *   -schem_scroll_y)` into screen cells inside `[vx .. vx+vw-1] x [vy .. vy+vh-1]`, decoding
 *   multi-byte UTF-8 sequences so each glyph occupies exactly one `cs_cell_t`.
 *
 * Rationale:
 *   Centralizing scroll translation and per-cell viewport clipping here lets all higher-level
 *   schematic primitives (`dc_draw_node_chip()`, `dc_draw_harrow()`, `dc_draw_vbranch()`) draw in
 *   clean, scroll-independent logical optical-rail coordinates without worrying about partial box
 *   clipping at pane borders.
 */
static void dc_put(
    dag_draw_t  *d,
    int          lx,
    int          ly,
    const char  *str,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs)
{
    if (d->measure_only)
    {
        return;
    }
    int sy = d->vy + (ly - d->app->schem_scroll_y);
    if (sy < d->vy || sy >= d->vy + d->vh)
    {
        return;
    }
    int         col = 0;
    const char *p   = str;
    while (*p != '\0')
    {
        /* Decode UTF-8 lead byte length (1..4 bytes) so box/arrow glyphs land in single cells */
        unsigned char b    = (unsigned char) *p;
        size_t        clen = ((b & 0x80u) == 0u)
                                 ? 1u
                                 : (((b & 0xE0u) == 0xC0u) ? 2u
                                                           : (((b & 0xF0u) == 0xE0u) ? 3u : 4u));
        char g[5] = { 0 };
        for (size_t i = 0u; i < clen && p[i] != '\0'; i++)
        {
            g[i] = p[i];
        }
        int sx = d->vx + (lx + col - d->app->schem_scroll_x);
        if (sx >= d->vx && sx < d->vx + d->vw)
        {
            cs_put_cell(&d->app->scr, sx, sy, g, fg, bg, attrs, 1);
        }
        p += clen;
        col++;
    }
}

/**
 * dc_draw_node_chip() - Measure, draw, and hit-register a 4-row hardware card on the DAG canvas
 * @d:        Active schematic drawing context (non-NULL)
 * @lx:       Logical left column of the card on the schematic canvas
 * @ly:       Logical top row of the card (`row_slot * 5`)
 * @item_idx: Selectable item index (`0` = `[bench]` telescope, `1..N` = `cfg.nodes[item_idx-1]`)
 * @tag:      Short hardware badge (`"TEL"`, `"DM"`, `"BS"`, `"CORO"`, etc., non-NULL)
 * @title:    Element section ID or bench name (non-NULL)
 * @sub:      One-line optical specification subtitle from `node_short_spec()` (non-NULL)
 * @accent:   Category border/title RGB color when unselected
 * @tip:      Hover tooltip registered on the card's visible rectangle (nullable)
 *
 * Role:
 *   Computes the auto-sized width (`bw` in `[15..28]` columns) of a 4-row optical component box,
 *   records its coordinates if selected (`sel_found`), and -- when not in `measure_only` mode --
 *   draws rounded (`╭─╮`) or double-line (`╔═╗` when selected) borders, `[TAG] id` header,
 *   hardware spec subtitle, and a viewport-clipped `CS_HIT_SCHEM_NODE` mouse hit region.
 *
 * Rationale:
 *   Switching from single rounded corners to bold double-line borders + highlighted fill on the
 *   active element makes keyboard/mouse selection unmistakable even on monochrome (`--text`) or
 *   7-bit ASCII (`--ascii`, `+===+` vs `+---+`) terminals, while clipping the registered hit box
 *   to `[vx..vx+vw, vy..vy+vh]` prevents partially scrolled-off chips from stealing clicks on
 *   neighboring panes.
 *
 * Return: Total outer width `bw` of the rendered card in logical columns.
 */
static int dc_draw_node_chip(
    dag_draw_t         *d,
    int                 lx,
    int                 ly,
    int                 item_idx,
    const char         *tag,
    const char         *title,
    const char         *sub,
    cs_tui_rgb_t        accent,
    const char         *tip)
{
    int          sel = (d->app->sel_item == item_idx);
    cs_tui_rgb_t bg  = sel ? d->pal->bg_selected : d->pal->bg_panel_alt;
    cs_tui_rgb_t bfg = sel ? d->pal->fg_bright : accent;
    uint8_t      at  = sel ? CS_ATTR_BOLD : 0u;

    char top_line[64];
    char bot_line[64];
    snprintf(top_line, sizeof(top_line), "[%s] %s", tag, title);
    snprintf(bot_line, sizeof(bot_line), "%s", sub);

    /* Clamp inner content width to [11..24] cols so cards stay compact yet readable */
    int inner = (int) strlen(top_line);
    int slen  = (int) strlen(bot_line);
    if (slen > inner)
    {
        inner = slen;
    }
    if (inner < 11)
    {
        inner = 11;
    }
    if (inner > 24)
    {
        inner = 24;
    }
    int bw = inner + 4;
    if (sel)
    {
        d->sel_found = 1;
        d->sel_lx    = lx;
        d->sel_ly    = ly;
        d->sel_bw    = bw;
    }
    if (d->measure_only)
    {
        return bw;
    }

    const char *tl = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x94" : "\xe2\x95\xad");
    const char *tr = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x97" : "\xe2\x95\xae");
    const char *bl = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x9a" : "\xe2\x95\xb0");
    const char *br = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x9d" : "\xe2\x95\xaf");
    const char *hz = d->app->scr.ascii_only ? (sel ? "=" : "-")
                                            : (sel ? "\xe2\x95\x90" : "\xe2\x94\x80");
    const char *vt = d->app->scr.ascii_only ? "|" : (sel ? "\xe2\x95\x91" : "\xe2\x94\x82");

    dc_put(d, lx, ly, tl, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly, tr, bfg, bg, at);
    dc_put(d, lx, ly + 3, bl, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly + 3, br, bfg, bg, at);
    for (int i = 1; i < bw - 1; i++)
    {
        dc_put(d, lx + i, ly, hz, bfg, bg, at);
        dc_put(d, lx + i, ly + 3, hz, bfg, bg, at);
        dc_put(d, lx + i, ly + 1, " ", d->pal->fg_text, bg, 0u);
        dc_put(d, lx + i, ly + 2, " ", d->pal->fg_muted, bg, 0u);
    }
    dc_put(d, lx, ly + 1, vt, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly + 1, vt, bfg, bg, at);
    dc_put(d, lx, ly + 2, vt, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly + 2, vt, bfg, bg, at);

    char pad_t[32], pad_b[32];
    snprintf(pad_t, sizeof(pad_t), "%.*s", inner, top_line);
    snprintf(pad_b, sizeof(pad_b), "%.*s", inner, bot_line);
    dc_put(d, lx + 2, ly + 1, pad_t, sel ? d->pal->fg_bright : accent, bg, CS_ATTR_BOLD);
    dc_put(d, lx + 2, ly + 2, pad_b, sel ? d->pal->fg_text : d->pal->fg_muted, bg, 0u);

    /* Register clickable hit region clipped to viewport */
    int sx = d->vx + (lx - d->app->schem_scroll_x);
    int sy = d->vy + (ly - d->app->schem_scroll_y);
    int x0 = (sx < d->vx) ? d->vx : sx;
    int y0 = (sy < d->vy) ? d->vy : sy;
    int x1 = (sx + bw > d->vx + d->vw) ? (d->vx + d->vw) : (sx + bw);
    int y1 = (sy + 4 > d->vy + d->vh) ? (d->vy + d->vh) : (sy + 4);
    if (x1 > x0 && y1 > y0)
    {
        cs_tui_add_hit(d->app, CS_HIT_SCHEM_NODE, item_idx, x0, y0, x1 - x0, y1 - y0, tip);
    }
    return bw;
}

/**
 * dc_draw_harrow() - Draw a 6-column horizontal colinear optical beam arrow with overhead label
 * @d:     Active schematic drawing context (non-NULL)
 * @lx:    Logical start column immediately to the right of the upstream node chip
 * @ly:    Logical top row of the current horizontal optical rail (`row_slot * 5`)
 * @label: Beam name or arm code (`"T"`, `"b_dm0"`, etc., nullable) drawn above the shaft
 * @fg:    Connector RGB color
 *
 * Role:
 *   Renders `─────▶` (or `----->` in `--ascii` mode) along row `ly + 1` to represent an inline
 *   or transmitted optical relay continuing along the same horizontal bench axis, placing an
 *   abbreviated 6-char beam/arm tag on row `ly`.
 *
 * Rationale:
 *   Stripping redundant `"b_"` prefixes on long beam identifiers (`strlen > 6`) keeps the 6-column
 *   inter-chip gap legible without pushing downstream optics unnecessarily far to the right.
 */
static void dc_draw_harrow(dag_draw_t *d, int lx, int ly, const char *label, cs_tui_rgb_t fg)
{
    const char *dash = d->app->scr.ascii_only ? "-" : "\xe2\x94\x80";
    const char *arr  = d->app->scr.ascii_only ? ">" : "\xe2\x96\xb6";
    if (label != NULL && label[0] != '\0')
    {
        const char *short_lbl = (strncmp(label, "b_", 2) == 0 && strlen(label) > 6u)
                                    ? (label + 2)
                                    : label;
        char        lbuf[16];
        snprintf(lbuf, sizeof(lbuf), "%.6s", short_lbl);
        dc_put(d, lx, ly, lbuf, d->pal->fg_dim, d->pal->bg_panel, 0u);
    }
    for (int i = 0; i < 5; i++)
    {
        dc_put(d, lx + i, ly + 1, dash, fg, d->pal->bg_panel, 0u);
    }
    dc_put(d, lx + 5, ly + 1, arr, fg, d->pal->bg_panel, CS_ATTR_BOLD);
}

/**
 * dc_draw_vbranch() - Draw an orthogonal vertical fold/pickoff drop-arm into a lower optical rail
 * @d:       Active schematic drawing context (non-NULL)
 * @lx:      Logical column of the vertical drop shaft (typically centered beneath a BS/CORO card)
 * @ly0:     Logical starting row where the folded beam exits the parent optic (`ly + 4`)
 * @ly1:     Logical top row of the newly allocated child rail (`child_row_slot * 5`)
 * @arm_lbl: Optional branch annotation (`"R"`, `"FPM(1.00D)"`, `"lowfs:Lyot"`, or fan-out beam)
 *
 * Role:
 *   Draws a vertical stem `│` from `ly0` down to `ly1 + 1`, turns `90 deg` rightward via
 *   `╰──▶` (or `` `--> `` in `--ascii` mode), and labels the elbow with @arm_lbl.
 *
 * Rationale:
 *   Dropping reflected (`out_refl`), coronagraphic LOWFS (`out_lowfs`), and multi-consumer fan-out
 *   beams vertically onto dedicated `5`-row-spaced horizontal rails guarantees that arbitrarily
 *   deep sub-branches never collide or cross existing horizontal arms above them.
 */
static void dc_draw_vbranch(dag_draw_t *d, int lx, int ly0, int ly1, const char *arm_lbl)
{
    const char *vt  = d->app->scr.ascii_only ? "|" : "\xe2\x94\x82";
    const char *cbl = d->app->scr.ascii_only ? "`" : "\xe2\x95\xb0";
    const char *hz  = d->app->scr.ascii_only ? "-" : "\xe2\x94\x80";
    const char *arr = d->app->scr.ascii_only ? ">" : "\xe2\x96\xb6";

    for (int ry = ly0; ry < ly1 + 1; ry++)
    {
        dc_put(d, lx, ry, vt, d->pal->fg_conn, d->pal->bg_panel, 0u);
    }
    dc_put(d, lx, ly1 + 1, cbl, d->pal->fg_conn, d->pal->bg_panel, 0u);
    dc_put(d, lx + 1, ly1 + 1, hz, d->pal->fg_conn, d->pal->bg_panel, 0u);
    dc_put(d, lx + 2, ly1 + 1, hz, d->pal->fg_conn, d->pal->bg_panel, 0u);
    dc_put(d, lx + 3, ly1 + 1, arr, d->pal->fg_conn, d->pal->bg_panel, CS_ATTR_BOLD);
    if (arm_lbl != NULL && arm_lbl[0] != '\0')
    {
        dc_put(d, lx + 2, (ly1 > 0) ? (ly1 - 1) : ly1, arm_lbl, d->pal->fg_dim, d->pal->bg_panel,
               0u);
    }
}

/**
 * dc_trace_beam() - Recursively lay out and render all downstream optical stages fed by @beam
 * @d:        Active schematic drawing context (updates `visited[]` and `next_row`, non-NULL)
 * @beam:     Upstream optical beam name to trace (nullable / empty returns immediately)
 * @lx:       Logical column at which the first consumer of @beam should be placed
 * @row_slot: Logical horizontal rail index (`ly = row_slot * 5`) for the primary colinear consumer
 * @depth:    Recursion depth guard (capped at `24` against malformed cyclic `.bench` files)
 *
 * Role:
 *   Finds every unvisited node whose `in_beam` matches @beam, places the primary consumer
 *   (`ci == 0`) colinearly on `row_slot` and any parallel fan-out consumers (`ci > 0`) on freshly
 *   allocated vertical drop rails (`d->next_row++`), and dispatches downstream recursion by class:
 *     - Inline pupil relays (`DM`, `TTM`, `WFE`): Continue colinearly along `use_row`.
 *     - Coronagraph (`CORO`): Science Lyot output (`out_beam`) continues colinearly right; optional
 *       reflective FPM or Lyot-stop pickoff (`out_lowfs`) drops vertically onto a child rail.
 *     - Beamsplitter (`SPLITTER`): Transmitted arm (`out_trans`, `"T"`) continues colinearly right;
 *       reflected arm (`out_refl`, `"R"`) drops vertically from the splitter's midpoint.
 *     - Terminal detectors (`PYRWFS`, `SHWFS`, `ZWFS`, `FPCAM`): Render a 32-bin polychromatic
 *       spectrum sparkline (`cs_draw_spectrum32()`) and `"<flux> ph/fr -> <out_cam>"` telemetry.
 *     - Unconsumed beams (`n_cons == 0`): Render a dimmed `"(beam_name)"` open-port stub.
 *
 * Rationale:
 *   Pre-order depth-first traversal (tracing the horizontal transmitted rail to completion before
 *   allocating `d->next_row++` for reflected/LOWFS arms, or vice versa in tree order) guarantees
 *   monotonic non-overlapping vertical rail indices in `O(N^2)` over `N <= 32` nodes with zero
 *   heap allocation.
 */
static void dc_trace_beam(dag_draw_t *d, const char *beam, int lx, int row_slot, int depth)
{
    if (beam == NULL || beam[0] == '\0' || depth > 24)
    {
        return;
    }

    int consumers[CACAOSIM_BENCH_MAX_NODES];
    int n_cons = 0;
    for (uint32_t i = 0; i < d->app->cfg.n_nodes; i++)
    {
        if (!d->visited[i] && strcmp(d->app->cfg.nodes[i].in_beam, beam) == 0)
        {
            consumers[n_cons++] = (int) i;
        }
    }

    /* Dangling optical beam with no downstream consumer: draw dimmed open-arm label */
    if (n_cons == 0)
    {
        int ly = row_slot * 5;
        char ubuf[32];
        snprintf(ubuf, sizeof(ubuf), "(%.20s)", beam);
        dc_put(d, lx, ly + 1, ubuf, d->pal->fg_dim, d->pal->bg_panel, CS_ATTR_DIM);
        return;
    }

    for (int ci = 0; ci < n_cons; ci++)
    {
        int idx = consumers[ci];
        if (d->visited[idx])
        {
            continue;
        }
        d->visited[idx] = 1u;

        /* Primary consumer stays colinear on row_slot; additional consumers drop onto new rails */
        int use_row = (ci == 0) ? row_slot : d->next_row++;
        int ly      = use_row * 5;
        int cur_x   = lx;

        if (ci > 0)
        {
            dc_draw_vbranch(d, lx - 3, row_slot * 5 + 2, ly, beam);
            cur_x = lx + 1;
        }

        const cacaosim_node_cfg_t *nd = &d->app->cfg.nodes[idx];
        char                       spec[40];
        char                       tip[96];
        node_short_spec(&d->app->cfg, nd, spec, sizeof(spec));
        snprintf(tip, sizeof(tip), "Click to inspect [%s.%s] (in=%s)", node_badge(nd), nd->id,
                 nd->in_beam);

        int bw = dc_draw_node_chip(d, cur_x, ly, idx + 1, node_badge(nd), nd->id, spec,
                                   node_color(nd, d->pal), tip);
        int nx = cur_x + bw;

        if (nd->kind == CACAOSIM_NODE_DM || nd->kind == CACAOSIM_NODE_WFE)
        {
            dc_draw_harrow(d, nx, ly, nd->out_beam, d->pal->fg_conn);
            dc_trace_beam(d, nd->out_beam, nx + 6, use_row, depth + 1);
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            /* Coronagraph: transmitted post-Lyot science beam right, reflective LOWFS arm down */
            dc_draw_harrow(d, nx, ly, nd->out_beam, d->pal->fg_conn);
            dc_trace_beam(d, nd->out_beam, nx + 6, use_row, depth + 1);
            if (nd->out_lowfs[0] != '\0')
            {
                int  lrow = d->next_row++;
                char llbl[24];
                if (nd->coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL)
                {
                    snprintf(llbl, sizeof(llbl), "lowfs:Lyot");
                }
                else
                {
                    float ap = (nd->coro.lowfs_pup_diam > 0.0f) ? nd->coro.lowfs_pup_diam : 1.0f;
                    snprintf(llbl, sizeof(llbl), "FPM(%.2fD)", ap);
                }
                dc_draw_vbranch(d, cur_x + bw / 2, ly + 4, lrow * 5, llbl);
                dc_trace_beam(d, nd->out_lowfs, cur_x + bw / 2 + 4, lrow, depth + 1);
            }
        }
        else if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            /* Beamsplitter: transmitted (T) arm continues right, reflected (R) arm folds down */
            if (nd->out_trans[0] != '\0')
            {
                dc_draw_harrow(d, nx, ly, "T", d->pal->fg_conn);
                dc_trace_beam(d, nd->out_trans, nx + 6, use_row, depth + 1);
            }
            if (nd->out_refl[0] != '\0')
            {
                int rrow = d->next_row++;
                dc_draw_vbranch(d, cur_x + bw / 2, ly + 4, rrow * 5, "R");
                dc_trace_beam(d, nd->out_refl, cur_x + bw / 2 + 4, rrow, depth + 1);
            }
        }
        else
        {
            /* Terminal sensor: draw incoming spectrum sparkline + live flux/shm readout */
            const cacaosim_spec_t       *sp  = find_beam_spec(d->app, nd->in_beam);
            const cacaosim_bench_node_t *an  = find_analyzed_node(d->app, nd->id);
            int                          sw  = d->wide_spec ? 24 : 14;
            int                          spx = d->vx + (nx + 1 - d->app->schem_scroll_x);
            int                          spy = d->vy + (ly + 1 - d->app->schem_scroll_y);

            if (!d->measure_only && sp != NULL && spy >= d->vy && spy < d->vy + d->vh &&
                spx >= d->vx && spx + sw <= d->vx + d->vw)
            {
                cs_draw_spectrum32(&d->app->scr, spx, spy, sw, sp, d->app->opts.theme,
                                   d->pal->bg_panel);
            }
            if (an != NULL)
            {
                char fstr[24], info[64];
                fmt_flux_compact(an->eff_flux_ph, fstr, sizeof(fstr));
                snprintf(info, sizeof(info), "%s ph/fr -> %s", fstr, nd->out_cam);
                dc_put(d, nx + 1, ly + 2, info, d->pal->fg_stream, d->pal->bg_panel, 0u);
            }
        }
    }
}

/**
 * cs_render_dag_canvas() - Auto-scroll and render the interactive 2D optical bench schematic
 * @app:          Master TUI application state (non-NULL)
 * @x:            Canvas left terminal column (0-based, inside pane border)
 * @y:            Canvas top terminal row (0-based, inside pane border)
 * @w:            Canvas width in columns (`< 8` returns early)
 * @h:            Canvas height in rows (`< 4` returns early)
 * @wide_spectra: Non-zero to draw 24-column spectral sparklines (used in wide panes / `F3`);
 *                0 for compact 14-column sparklines
 *
 * Role:
 *   Renders the complete 2D optical DAG inside both `F2:BUILDER` (top-right pane) and
 *   `F3:SCHEMATIC` (full-width upper pane):
 *     1. Displays a prominent red `[!] DAG WIRING WARNING` banner at the top of the canvas if
 *        `!app->dag_valid`.
 *     2. When keyboard focus is outside `CS_FOCUS_SCHEM`, executes a dry-run `measure_only = 1`
 *        pass to locate the selected node's logical coordinates and adjusts `schem_scroll_x` /
 *        `schem_scroll_y` with trailing lookahead (`+28` cols, `+9` rows) so both the chip and its
 *        downstream arrow/spectrum remain framed.
 *     3. Executes the live draw pass from `[TEL]` root beam (`b_pup`) through `dc_trace_beam()`,
 *        then renders any disconnected/unwired nodes (`!d.visited[i]`) below the main tree in
 *        `fg_error` red so broken elements never vanish and can be clicked and rewired via `w`.
 *
 * Rationale:
 *   Sharing a single parameterized canvas engine across `F2:BUILDER` and `F3:SCHEMATIC` guarantees
 *   identical topology visualization and mouse hit-testing everywhere, while rendering orphaned
 *   nodes explicitly at the bottom prevents hand-edited `.bench` typos from hiding elements from
 *   the GUI.
 */
void cs_render_dag_canvas(
    cs_tui_app_t *app,
    int           x,
    int           y,
    int           w,
    int           h,
    int           wide_spectra)
{
    if (w < 8 || h < 4)
    {
        return;
    }
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 cy  = y;
    int                 ch  = h;

    if (!app->dag_valid)
    {
        cs_fill_rect(&app->scr, x, cy, w, 1, pal->fg_error, pal->bg_selected);
        cs_printf_at(&app->scr, x + 1, cy, w - 2, pal->fg_error, pal->bg_selected, CS_ATTR_BOLD, 1,
                     "[!] DAG WIRING WARNING: %s (select node & press 'w' to rewire)",
                     app->dag_err);
        cy++;
        ch--;
    }

    char tel_sub[40];
    snprintf(tel_sub, sizeof(tel_sub), "%.1fm mV=%.1f %.0fHz", app->cfg.tel_diam_m,
             app->cfg.star_mag, app->cfg.master_hz);
    const char *root_beam = app->cfg.out_beam[0] ? app->cfg.out_beam : "b_pup";

    /* Auto-follow selected element when schematic pane is not being manually scrolled */
    if (app->focus != CS_FOCUS_SCHEM)
    {
        dag_draw_t md;
        memset(&md, 0, sizeof(md));
        md.app          = app;
        md.pal          = pal;
        md.vw           = w;
        md.vh           = ch;
        md.measure_only = 1;
        md.next_row     = 1;
        int mbw = dc_draw_node_chip(&md, 1, 0, 0, "TEL", app->cfg.name[0] ? app->cfg.name : "bench",
                                    tel_sub, pal->fg_title, "");
        dc_trace_beam(&md, root_beam, 1 + mbw + 6, 0, 0);
        for (uint32_t i = 0; i < app->cfg.n_nodes; i++)
        {
            if (!md.visited[i])
            {
                int                        urow = md.next_row++;
                const cacaosim_node_cfg_t *nd   = &app->cfg.nodes[i];
                dc_draw_node_chip(&md, 2, urow * 5, (int) i + 1, node_badge(nd), nd->id, "",
                                  pal->fg_error, "");
            }
        }
        if (md.sel_found)
        {
            /* Keep +28 cols horizontal and +9 rows vertical lookahead around selected card */
            int rx_need = md.sel_lx + md.sel_bw + 28;
            if (md.sel_lx < app->schem_scroll_x)
            {
                app->schem_scroll_x = (md.sel_lx > 2) ? (md.sel_lx - 2) : 0;
            }
            else if (rx_need > app->schem_scroll_x + w)
            {
                app->schem_scroll_x = rx_need - w;
                if (app->schem_scroll_x > md.sel_lx)
                {
                    app->schem_scroll_x = (md.sel_lx > 1) ? (md.sel_lx - 1) : 0;
                }
            }
            int ry_need = md.sel_ly + 9;
            if (md.sel_ly < app->schem_scroll_y)
            {
                app->schem_scroll_y = md.sel_ly;
            }
            else if (ry_need > app->schem_scroll_y + ch)
            {
                app->schem_scroll_y = ry_need - ch;
                if (app->schem_scroll_y > md.sel_ly)
                {
                    app->schem_scroll_y = md.sel_ly;
                }
            }
        }
    }

    dag_draw_t d;
    memset(&d, 0, sizeof(d));
    d.app       = app;
    d.pal       = pal;
    d.vx        = x;
    d.vy        = cy;
    d.vw        = w;
    d.vh        = ch;
    d.wide_spec = wide_spectra;
    d.next_row  = 1;

    int         bw        = dc_draw_node_chip(
        &d, 1, 0, 0, "TEL", app->cfg.name[0] ? app->cfg.name : "bench", tel_sub, pal->fg_title,
        "Click to inspect [bench] telescope, pupil & guide star");
    dc_draw_harrow(&d, 1 + bw, 0, root_beam, pal->fg_conn);
    dc_trace_beam(&d, root_beam, 1 + bw + 6, 0, 0);

    /* Render any disconnected/unwired nodes below main tree so user can click & rewire them */
    for (uint32_t i = 0; i < app->cfg.n_nodes; i++)
    {
        if (!d.visited[i])
        {
            int                        urow = d.next_row++;
            const cacaosim_node_cfg_t *nd   = &app->cfg.nodes[i];
            char                       sub[40];
            snprintf(sub, sizeof(sub), "UNWIRED in=%s", nd->in_beam);
            dc_draw_node_chip(&d, 2, urow * 5, (int) i + 1, node_badge(nd), nd->id, sub,
                              pal->fg_error, "Unwired optical element -- press 'w' to connect");
        }
    }
}

/* ── F2:BUILDER 3-Pane View (Elements + 2D Schematic + Inspector) ─────────── */

/**
 * draw_btn() - Render a bold clickable pill button and register its hit-test box
 * @app: Active TUI application state owning the screen backbuffer and hit table (non-NULL)
 * @x:   Left terminal column of the button in 0-based screen coordinates
 * @y:   Terminal row of the button in 0-based screen coordinates
 * @lbl: NUL-terminated button caption string (clamped to 24 columns)
 * @hk:  Semantic hit-box action kind dispatched on mouse click
 * @arg: Integer payload associated with @hk (e.g. tab index or 0)
 * @fg:  Foreground RGB color for the button label
 * @bg:  Background RGB color for the button pill
 * @tip: Optional NUL-terminated hover tooltip message (may be NULL)
 *
 * Role: Paints a single-row bold button widget onto the backbuffer and atomically records
 * its exact rendered width in `app->hits` for mouse click and hover-tooltip dispatch.
 *
 * Rationale: Co-locating `cs_put_str()` and `cs_tui_add_hit()` in one immediate-mode helper
 * guarantees that button click regions always match the true rendered glyph width across
 * dynamic label changes (e.g. `[T:dark]` vs. `[T:milk]`) and terminal resizes, eliminating
 * manual coordinate arithmetic bugs.
 *
 * Return: Number of terminal columns occupied by the rendered button label.
 */
static int draw_btn(
    cs_tui_app_t       *app,
    int                 x,
    int                 y,
    const char         *lbl,
    cs_hit_kind_t       hk,
    int                 arg,
    cs_tui_rgb_t        fg,
    cs_tui_rgb_t        bg,
    const char         *tip)
{
    int len = cs_put_str(&app->scr, x, y, 24, lbl, fg, bg, CS_ATTR_BOLD, 1);
    cs_tui_add_hit(app, hk, arg, x, y, len, 1, tip);
    return len;
}

/**
 * render_view_builder() - Render the primary 3-pane F2:BUILDER interactive workspace
 * @app: Active TUI application state (non-NULL)
 * @x:   Left column of the workspace body area in screen coordinates
 * @y:   Top row of the workspace body area in screen coordinates
 * @w:   Available workspace width in terminal columns
 * @h:   Available workspace height in terminal rows
 *
 * Role: Lays out and renders the primary `F2:BUILDER` configurator screen comprising three
 * synchronized, mouse-resizable panes:
 *   1. Left `ELEMENTS & ARRANGER` pane: topological toolbar (`+Add`, `Dup`, `Up`, `Dn`,
 *      `Swp`, `Del`), scrollable element list (`[0] TEL` entrance source plus all hardware
 *      nodes with live SHM activity dots), and an adaptive 16x16 pupil preview & physical
 *      optical scale dock (`m/pix`, `lambda/D/pix`, `arcsec/pix`, `[g:InitSHM]`).
 *   2. Top-right `2D OPTICAL SCHEMATIC & PHOTOMETRY` pane: interactive DAG canvas showing
 *      beams, splitters, coronagraph arms, and `[+]` optical splice targets.
 *   3. Bottom-right `INSPECTOR` pane: 3-subtab (`1:OPTICS`, `2:DYN/DET`, `3:SHM/WIRING`)
 *      declarative parameter table with inline text editing, `< enum >` stepper badges, and
 *      smart `.../basename` path truncation.
 *
 * Rationale: Combining the linear hardware list, 2D optical table schematic, physical
 * sampling dock, and categorized parameter inspector in a single split view gives AO
 * engineers immediate visual feedback whenever they splice a node or tweak an optical
 * parameter. Clamping `lw` and `sh` to safe minimum dimensions and switching the pupil dock
 * between compact (`h >= 24`) and tall (`h >= 28`) layouts keeps all three panes legible
 * from compact 80x24 SSH sessions up to ultrawide control-room monitors.
 */
static void render_view_builder(cs_tui_app_t *app, int x, int y, int w, int h)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    /*
     * Compute resizable split geometry: clamp the left Elements pane width so both the
     * element list (>= 28 cols when possible) and right Schematic/Inspector (>= 42 cols)
     * remain usable, and derive the Schematic height from `schem_h_pct` with safe bounds.
     */
    int                 lw  = app->left_w;
    if (lw < 28)
    {
        lw = 28;
    }
    if (lw > w - 42)
    {
        lw = w - 42;
    }
    if (lw < 24)
    {
        lw = w / 3;
    }
    int rx = x + lw;
    int rw = w - lw;

    int sh = (h * app->schem_h_pct) / 100;
    if (sh < 8)
    {
        sh = 8;
    }
    if (sh > h - 9)
    {
        sh = h - 9;
    }
    int ih = h - sh;

    /* 1. Left Pane: Optical Elements & Arranger + draggable vertical splitter handle */
    int f_el = (app->focus == CS_FOCUS_ELEMS);
    char nbadge[32];
    snprintf(nbadge, sizeof(nbadge), "%u nodes", app->cfg.n_nodes);
    cs_draw_box(&app->scr, x, y, lw, h, f_el, "ELEMENTS & ARRANGER", nbadge,
                f_el ? pal->fg_border_focus : pal->fg_border, pal->fg_title, pal->fg_muted,
                pal->bg_panel);
    cs_tui_add_hit(app, CS_HIT_SPLIT_LEFT, 0, x + lw - 1, y, 2, h,
                   "Drag vertical divider to resize Elements pane");

    /* Action toolbar inside Elements pane (progressively elides right pills on narrow panes) */
    int bx = x + 2;
    int by = y + 1;
    bx += draw_btn(app, bx, by, "[+Add]", CS_HIT_ACT_ADD, 0, pal->fg_active, pal->bg_panel_alt,
                   "Add optical element [a]") +
          1;
    bx += draw_btn(app, bx, by, "[Dup]", CS_HIT_ACT_CLONE, 0, pal->fg_stream, pal->bg_panel_alt,
                   "Duplicate selected element [c]") +
          1;
    bx += draw_btn(app, bx, by, "[Up]", CS_HIT_ACT_UP, 0, pal->fg_text, pal->bg_panel_alt,
                   "Move element upstream [K / Shift+Up]") +
          1;
    bx += draw_btn(app, bx, by, "[Dn]", CS_HIT_ACT_DOWN, 0, pal->fg_text, pal->bg_panel_alt,
                   "Move element downstream [J / Shift+Down]") +
          1;
    if (bx + 6 <= x + lw - 1)
    {
        bx += draw_btn(app, bx, by, "[Swp]", CS_HIT_ACT_SWAP, 0, pal->fg_proc, pal->bg_panel_alt,
                       "Swap splitter/coro output arms [x]") +
              1;
    }
    if (bx + 5 <= x + lw - 1)
    {
        draw_btn(app, bx, by, "[Del]", CS_HIT_ACT_DEL, 0, pal->fg_error, pal->bg_panel_alt,
                 "Remove selected element [d / Del]");
    }

    /* Reserve bottom rows of the left pane for the 16x16 pupil & optical scale dock */
    int show_pup_dock = (h >= 24 && lw >= 30);
    int tall_pup_dock = (h >= 28 && lw >= 30);
    int dock_rows     = tall_pup_dock ? 15 : 10;
    int list_top      = y + 3;
    int list_bot      = show_pup_dock ? (y + h - dock_rows) : (y + h - 1);
    int list_rows     = list_bot - list_top;
    int total_items   = 1 + (int) app->cfg.n_nodes;

    /* Auto-scroll element list viewport to keep `sel_item` visible */
    if (app->sel_item < app->elem_scroll)
    {
        app->elem_scroll = app->sel_item;
    }
    if (list_rows > 0 && app->sel_item >= app->elem_scroll + list_rows)
    {
        app->elem_scroll = app->sel_item - list_rows + 1;
    }

    cs_printf_at(&app->scr, x + 2, y + 2, lw - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                 " #  TYPE  ID / SPEC");

    for (int r = 0; r < list_rows; r++)
    {
        int item = app->elem_scroll + r;
        int ry   = list_top + r;
        if (item >= total_items)
        {
            break;
        }
        int          sel    = (item == app->sel_item);
        cs_tui_rgb_t row_bg = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, x + 1, ry, lw - 2, 1, pal->fg_text, row_bg);

        const char *ptr = sel ? (app->scr.ascii_only ? ">" : "\xe2\x96\xb6") : " ";
        cs_put_str(&app->scr, x + 1, ry, 1, ptr, pal->fg_accent, row_bg, CS_ATTR_BOLD, 1);

        if (item == 0)
        {
            /* Item 0 always represents the `[bench]` telescope pupil & stellar source */
            cs_printf_at(&app->scr, x + 3, ry, lw - 5, sel ? pal->fg_bright : pal->fg_title, row_bg,
                         CS_ATTR_BOLD, 1, "[0] TEL   %-10.10s %.1fm mV=%.1f",
                         app->cfg.name[0] ? app->cfg.name : "bench", app->cfg.tel_diam_m,
                         app->cfg.star_mag);
            cs_tui_add_hit(app, CS_HIT_ELEM_ROW, 0, x + 1, ry, lw - 2, 1,
                           "Select [bench] entrance telescope & star");
        }
        else
        {
            const cacaosim_node_cfg_t *nd = &app->cfg.nodes[item - 1];
            char                       spec[32];
            node_short_spec(&app->cfg, nd, spec, sizeof(spec));
            cs_tui_rgb_t nc = node_color(nd, pal);
            cs_printf_at(&app->scr, x + 3, ry, 9, sel ? pal->fg_bright : nc, row_bg, CS_ATTR_BOLD,
                         1, "[%d] %-4s", item, node_badge(nd));
            cs_printf_at(&app->scr, x + 12, ry, lw - 15, sel ? pal->fg_bright : pal->fg_text,
                         row_bg, sel ? CS_ATTR_BOLD : 0u, 1, "%-10.10s %s", nd->id, spec);

            /* Trailing SHM liveness indicator: active green (2), present cyan (1), missing (0) */
            int live = node_primary_shm_live(app, nd);
            if (live >= 0 && lw >= 28)
            {
                const char  *dot = (live > 0) ? (app->scr.ascii_only ? "*" : "\xe2\x97\x8f")
                                              : (app->scr.ascii_only ? "o" : "\xe2\x97\x8b");
                cs_tui_rgb_t dfg = (live == 2) ? pal->fg_active
                                               : (live == 1 ? pal->fg_stream : pal->fg_dim);
                cs_put_str(&app->scr, x + lw - 3, ry, 1, dot, dfg, row_bg, 0u, 1);
            }
            char rtip[96];
            snprintf(rtip, sizeof(rtip), "Select [%s.%s] (in=%s)", node_badge(nd), nd->id,
                     nd->in_beam);
            cs_tui_add_hit(app, CS_HIT_ELEM_ROW, item, x + 1, ry, lw - 2, 1, rtip);
        }
    }

    /*
     * Render the 16x16 anti-aliased pupil preview and physical optical scale dock.
     * In tall terminals (`h >= 28`), pupil/source summary lines sit beside the 16x8 half-block
     * aperture while full-width physical sampling (`m/pix`, `l/D/pix`, `arcsec/pix`) renders
     * underneath; in compact terminals (`24 <= h < 28`), sampling metrics pack beside the pupil.
     */
    if (show_pup_dock)
    {
        cs_optical_scales_t sc;
        cs_tui_compute_scales(&app->cfg, app->sel_item, &sc);

        int py = y + h - (tall_pup_dock ? 14 : 9);
        for (int cx = x + 1; cx < x + lw - 1; cx++)
        {
            cs_put_cell(&app->scr, cx, py - 1, app->scr.ascii_only ? "-" : "\xe2\x94\x80",
                        pal->fg_border, pal->bg_panel, 0u, 1);
        }
        cs_draw_pupil16(&app->scr, x + 2, py, app->pup16, pal);
        int tx = x + 19;
        int tw = lw - 20;
        if (tw > 6)
        {
            if (tall_pup_dock)
            {
                cs_printf_at(&app->scr, tx, py, tw, pal->fg_title, pal->bg_panel, CS_ATTR_BOLD, 1,
                             "PUPIL %ux%u", app->cfg.nwf, app->cfg.nwf);
                cs_printf_at(&app->scr, tx, py + 1, tw, pal->fg_text, pal->bg_panel, 0u, 1,
                             "D=%.1fm obs=%.0f%%", app->cfg.tel_diam_m, app->cfg.cobs * 100.0f);
                cs_printf_at(&app->scr, tx, py + 2, tw, pal->fg_muted, pal->bg_panel, 0u, 1,
                             "%d spiders", app->cfg.nspider);
                cs_printf_at(&app->scr, tx, py + 4, tw, pal->fg_accent, pal->bg_panel, 0u, 1,
                             "mV=%.1f %.0fK", app->cfg.star_mag, app->cfg.star_teff);
                if (app->dag_valid && app->ctx.n_beams > 0u)
                {
                    char f0[24];
                    fmt_flux_compact(cacaosim_spec_total(&app->ctx.beams[0].spec), f0, sizeof(f0));
                    cs_printf_at(&app->scr, tx, py + 5, tw, pal->fg_stream, pal->bg_panel, 0u, 1,
                                 "%s ph/s", f0);
                }
                draw_btn(app, tx, py + 7, "[g:InitSHM]", CS_HIT_ACT_MKPUP, 0, pal->fg_stream,
                         pal->bg_panel_alt,
                         "Generate pupil & static WFE streams in MILK_SHM_DIR [g]");
            }
            else
            {
                cs_printf_at(&app->scr, tx, py, tw, pal->fg_title, pal->bg_panel, CS_ATTR_BOLD, 1,
                             "%ux%u pix", app->cfg.nwf, app->cfg.nwf);
                cs_printf_at(&app->scr, tx, py + 1, tw, pal->fg_text, pal->bg_panel, 0u, 1,
                             "Beam %.1f pix", sc.d_beam_px);
                cs_printf_at(&app->scr, tx, py + 2, tw, pal->fg_stream, pal->bg_panel, 0u, 1,
                             "%.4f m/pix", sc.pup_m_pix);
                cs_printf_at(&app->scr, tx, py + 3, tw, pal->fg_accent, pal->bg_panel, 0u, 1,
                             "%.3f l/D/pix", sc.foc_ld_pix);
                cs_printf_at(&app->scr, tx, py + 4, tw, pal->fg_muted, pal->bg_panel, 0u, 1,
                             "%.4f arcsec/pix", sc.foc_arcsec_pix);
                cs_printf_at(&app->scr, tx, py + 5, tw, pal->fg_dim, pal->bg_panel, 0u, 1,
                             "@%.2fum D=%.1fm", sc.ref_lam_um, app->cfg.tel_diam_m);
                draw_btn(app, tx, py + 7, "[g:InitSHM]", CS_HIT_ACT_MKPUP, 0, pal->fg_stream,
                         pal->bg_panel_alt,
                         "Generate pupil & static WFE streams in MILK_SHM_DIR [g]");
            }
        }
        if (tall_pup_dock)
        {
            int fw = lw - 4;
            cs_printf_at(&app->scr, x + 2, py + 8, fw, pal->fg_text, pal->bg_panel, 0u, 1,
                         "Array: %ux%u pix", app->cfg.nwf, app->cfg.nwf);
            cs_printf_at(&app->scr, x + 2, py + 9, fw, pal->fg_text, pal->bg_panel, 0u, 1,
                         "Beam : %.1f pix (%.1f%%)", sc.d_beam_px, app->cfg.pup_diam * 100.0f);
            cs_printf_at(&app->scr, x + 2, py + 10, fw, pal->fg_stream, pal->bg_panel, 0u, 1,
                         "Pupil: %.4f m/pix", sc.pup_m_pix);
            cs_printf_at(&app->scr, x + 2, py + 11, fw, pal->fg_accent, pal->bg_panel, 0u, 1,
                         "Focal: %.3f l/D/pix (%.8s)", sc.foc_ld_pix, sc.foc_src);
            cs_printf_at(&app->scr, x + 2, py + 12, fw, pal->fg_muted, pal->bg_panel, 0u, 1,
                         "Sky  : %.4f arcsec/pix @%.2fum", sc.foc_arcsec_pix, sc.ref_lam_um);
        }
    }

    /* 2. Top-Right Pane: 2D Interactive Optical Schematic (compact mode if width < 88 cols) */
    int f_sc = (app->focus == CS_FOCUS_SCHEM);
    cs_draw_box(&app->scr, rx, y, rw, sh, f_sc, "2D OPTICAL SCHEMATIC & PHOTOMETRY",
                app->dag_valid ? "CLICK NODE TO INSPECT" : "CHECK WIRING",
                f_sc ? pal->fg_border_focus : pal->fg_border, pal->fg_title,
                app->dag_valid ? pal->fg_muted : pal->fg_error, pal->bg_panel);
    cs_render_dag_canvas(app, rx + 1, y + 1, rw - 2, sh - 2, rw >= 88);

    /* 3. Bottom-Right Pane: Declarative Parameter Inspector + horizontal resize handle */
    int iy   = y + sh;
    int f_pa = (app->focus == CS_FOCUS_PARAMS);
    char ititle[96];
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        snprintf(ititle, sizeof(ititle), "INSPECTOR: [bench] %s", app->cfg.name);
    }
    else
    {
        const cacaosim_node_cfg_t *nd    = &app->cfg.nodes[app->sel_item - 1];
        const char                *kname = "fpcam";
        if (nd->kind == CACAOSIM_NODE_DM)
        {
            kname = (nd->dm.dm_mode == 1) ? "ttm" : "dm";
        }
        else if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            kname = "splitter";
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            kname = "coro";
        }
        else if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            kname = "pyrwfs";
        }
        else if (nd->kind == CACAOSIM_NODE_SHWFS)
        {
            kname = "shwfs";
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            kname = "zwfs";
        }
        else if (nd->kind == CACAOSIM_NODE_WFE)
        {
            kname = "wfe";
        }
        snprintf(ititle, sizeof(ititle), "INSPECTOR: [%s.%s]", kname, nd->id);
    }
    cs_draw_box(&app->scr, rx, iy, rw, ih, f_pa, ititle, "Enter:Edit  <-/->:Step",
                f_pa ? pal->fg_border_focus : pal->fg_border, pal->fg_title, pal->fg_muted,
                pal->bg_panel);
    cs_tui_add_hit(app, CS_HIT_SPLIT_SCHEM, 0, rx, iy, rw, 1,
                   "Drag horizontal divider to resize Schematic / Inspector");

    /* Inspector category subtabs (`1:OPTICS`, `2:DYN/DET`, `3:SHM/WIRING`) + Wire shortcut */
    static const char *const st_names[3] = { " 1:OPTICS ", " 2:DYN/DET ", " 3:SHM/WIRING " };
    int                      stx         = rx + 2;
    for (int st = 0; st < 3; st++)
    {
        int          act = (app->subtab == st);
        cs_tui_rgb_t sfg = act ? pal->fg_bright : pal->fg_muted;
        cs_tui_rgb_t sbg = act ? pal->bg_selected : pal->bg_panel_alt;
        int          len = cs_put_str(&app->scr, stx, iy + 1, 20, st_names[st], sfg, sbg,
                                      act ? CS_ATTR_BOLD : 0u, 1);
        cs_tui_add_hit(app, CS_HIT_SUBTAB, st, stx, iy + 1, len, 1,
                       "Switch parameter category [1 / 2 / 3]");
        stx += len + 1;
    }
    if (stx + 11 < rx + rw - 1)
    {
        const char  *albl = app->show_advanced ? "[v:Adv:ON]" : "[v:Adv]";
        cs_tui_rgb_t afg  = app->show_advanced ? pal->fg_active : pal->fg_muted;
        cs_tui_rgb_t abg  = app->show_advanced ? pal->bg_selected : pal->bg_panel_alt;
        stx += draw_btn(app, stx + 1, iy + 1, albl, CS_HIT_ACT_ADVANCED, 0, afg, abg,
                        "Toggle Advanced .bench options in Inspector [v]") +
               1;
    }
    if (app->sel_item > 0 && stx + 14 < rx + rw - 1)
    {
        draw_btn(app, stx + 1, iy + 1, "[w:Wire Beam]", CS_HIT_ACT_WIRE, 0, pal->fg_stream,
                 pal->bg_panel_alt, "Pick upstream optical beam for this element [w]");
    }

    int p_top  = iy + 3;
    int p_rows = (iy + ih - 1) - p_top;
    cs_printf_at(&app->scr, rx + 2, iy + 2, rw - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                 "%-14s %-18s %-18s %-6s %s", "KEY", "PARAMETER", "VALUE", "UNIT", "NOTES");

    /* Clamp and auto-scroll parameter selection when switching nodes or subtabs */
    if (app->param_sel >= app->n_params && app->n_params > 0)
    {
        app->param_sel = app->n_params - 1;
    }
    if (app->param_sel < 0)
    {
        app->param_sel = 0;
    }
    if (app->param_sel < app->param_scroll)
    {
        app->param_scroll = app->param_sel;
    }
    if (p_rows > 0 && app->param_sel >= app->param_scroll + p_rows)
    {
        app->param_scroll = app->param_sel - p_rows + 1;
    }

    for (int r = 0; r < p_rows; r++)
    {
        int pi = app->param_scroll + r;
        int py = p_top + r;
        if (pi >= app->n_params)
        {
            break;
        }
        const cs_param_desc_t *pd  = &app->params[pi];
        int                    sel = (pi == app->param_sel && f_pa);
        cs_tui_rgb_t           rbg = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, rx + 1, py, rw - 2, 1, pal->fg_text, rbg);

        char vbuf[96];
        cs_param_format_val(&app->cfg, app->sel_item, pd, vbuf, sizeof(vbuf));

        cs_printf_at(&app->scr, rx + 2, py, 14, pal->fg_fps, rbg, 0u, 1, "%-14.14s", pd->key);
        cs_printf_at(&app->scr, rx + 17, py, 18, sel ? pal->fg_bright : pal->fg_text, rbg,
                     sel ? CS_ATTR_BOLD : 0u, 1, "%-18.18s", pd->label);

        /* Render value cell as live input cursor, `< discrete >` stepper, or scalar/path */
        if (sel && app->editing)
        {
            cs_fill_rect(&app->scr, rx + 36, py, 18, 1, pal->fg_bright, pal->bg_input);
            cs_printf_at(&app->scr, rx + 36, py, 17, pal->fg_accent, pal->bg_input, CS_ATTR_BOLD, 1,
                         "%s_", app->edit_buf);
        }
        else if (pd->type == CS_PTYPE_ENUM || pd->type == CS_PTYPE_BOOL ||
                 pd->type == CS_PTYPE_BEAM)
        {
            char disp[32];
            snprintf(disp, sizeof(disp), "< %.26s >", vbuf[0] ? vbuf : "-");
            cs_tui_rgb_t vfg = (pd->type == CS_PTYPE_BOOL && strcmp(vbuf, "ON") == 0)
                                   ? pal->fg_active
                                   : pal->fg_accent;
            cs_printf_at(&app->scr, rx + 36, py, 18, vfg, rbg, CS_ATTR_BOLD, 1, "%-18.18s", disp);
        }
        else
        {
            /* Preserve FITS/SHM filename suffix (`.../basename`) when long paths exceed 18 cols */
            char        pdisp[32];
            const char *vshow = vbuf[0] ? vbuf : "\"\"";
            if (strlen(vbuf) > 18u && strchr(vbuf, '/') != NULL)
            {
                const char *bn = strrchr(vbuf, '/');
                snprintf(pdisp, sizeof(pdisp), ".../%.14s", (bn != NULL) ? (bn + 1) : vbuf);
                vshow = pdisp;
            }
            cs_printf_at(&app->scr, rx + 36, py, 18, sel ? pal->fg_bright : pal->fg_stream, rbg,
                         sel ? CS_ATTR_BOLD : 0u, 1, "%-18.18s", vshow);
        }

        cs_printf_at(&app->scr, rx + 55, py, 6, pal->fg_muted, rbg, 0u, 1, "%-6.6s", pd->unit);
        if (rw > 66)
        {
            cs_printf_at(&app->scr, rx + 62, py, rw - 64, pal->fg_dim, rbg, 0u, 1, "%s", pd->tip);
        }
        cs_tui_add_hit(app, CS_HIT_PARAM_ROW, pi, rx + 1, py, rw - 2, 1, pd->tip);
    }
}

/* ── Top Header, Tab Bar, Bottom Status Bar, Modals & Tooltips ────────────── */

/**
 * render_header_and_tabs() - Render the 2-row milk-CTRL top telemetry header and tab bar
 * @app: Active TUI application state (non-NULL)
 *
 * Role: Paints Rows 0 and 1 of the full-screen terminal frame:
 *   - Row 0 (`milk-CTRL` banner): alternating heartbeat diamond, application badge, active
 *     `[bench]` name and `.bench` filename, `[*MODIFIED]` unsaved-edits indicator, and
 *     right-aligned live telemetry (`DAG:OK|ERR`, `SHM:live/total`, `FPS:running`, science
 *     `SR` Strehl ratio when available, and differential terminal I/O `tx_kbps`).
 *   - Row 1 (navigation & quick actions): clickable `F2:BUILDER` .. `F6:SOURCE` workspace
 *     tabs on the left and `[P:Presets]`, `[s:Save]`, `[f:Run]`, `[S:Stop]`, `[h:HELP]`
 *     action pills on the right.
 *
 * Rationale: Keeping DAG health, SHM/FPS liveness, science Strehl, and serial bandwidth pinned
 * to Row 0 across all five workspace tabs mirrors `milk-CTRL` operational ergonomics so the
 * user can spot broken wiring or stalled simulation processes while editing parameters or
 * reading `.bench` source, while the oscillating heartbeat confirms the TUI event loop itself
 * has not hung over a slow SSH link.
 */
static void render_header_and_tabs(cs_tui_app_t *app)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 W   = app->scr.cols;

    /* Row 0: milk-CTRL Header Bar with frame-tick heartbeat glyph */
    cs_fill_rect(&app->scr, 0, 0, W, 1, pal->fg_text, pal->bg_bar);
    const char *hb = ((app->frame_tick % 2u) == 0u)
                         ? (app->scr.ascii_only ? "*" : "\xe2\x97\x86")
                         : (app->scr.ascii_only ? "o" : "\xe2\x97\x87");
    cs_put_str(&app->scr, 1, 0, 2, hb, pal->fg_active, pal->bg_bar, CS_ATTR_BOLD, 1);
    cs_put_str(&app->scr, 3, 0, 14, "cacaosim-CTRL", pal->fg_title, pal->bg_bar, CS_ATTR_BOLD, 1);

    char path_copy[512];
    snprintf(path_copy, sizeof(path_copy), "%s",
             app->file_path[0] ? app->file_path : "unsaved.bench");
    const char *bname = basename(path_copy);
    int         cx    = 18;
    cx += cs_printf_at(&app->scr, cx, 0, 44, pal->fg_bright, pal->bg_bar, 0u, 1, "[%s] %s",
                       app->cfg.name[0] ? app->cfg.name : "bench", bname);
    if (app->dirty)
    {
        cx += cs_put_str(&app->scr, cx + 1, 0, 12, "[*MODIFIED]", pal->fg_warn, pal->bg_bar,
                         CS_ATTR_BOLD, 1) +
              1;
    }

    /* Right-aligned status pills on Row 0 (omitted if terminal is too narrow) */
    char rstat[128];
    if (app->shm.has_strehl)
    {
        snprintf(rstat, sizeof(rstat), "%s  SHM:%d/%d  FPS:%d  SR:%.2f  %4.1f kB/s ",
                 app->dag_valid ? "DAG:OK" : "DAG:ERR", app->shm.n_live_streams, app->shm.n_streams,
                 app->shm.n_running_fps, app->shm.strehl_ave, app->scr.tx_kbps);
    }
    else
    {
        snprintf(rstat, sizeof(rstat), "%s  SHM:%d/%d  FPS:%d  %4.1f kB/s ",
                 app->dag_valid ? "DAG:OK" : "DAG:ERR", app->shm.n_live_streams, app->shm.n_streams,
                 app->shm.n_running_fps, app->scr.tx_kbps);
    }
    int rlen = (int) strlen(rstat);
    if (W - rlen > cx + 2)
    {
        cs_put_str(&app->scr, W - rlen, 0, rlen, rstat,
                   !app->dag_valid
                       ? pal->fg_error
                       : (app->shm.n_running_fps > 0 ? pal->fg_active : pal->fg_muted),
                   pal->bg_bar, CS_ATTR_BOLD, 1);
    }

    /* Row 1: F2..F6 View Tabs + Action Pills */
    cs_fill_rect(&app->scr, 0, 1, W, 1, pal->fg_text, pal->bg_bar);
    static const char *const vtabs[CACAOSIM_TUI_VIEW_COUNT] = {
        " F2:BUILDER ", " F3:SCHEMATIC ", " F4:TABLES ", " F5:STATUS ", " F6:SOURCE "
    };
    static const char *const vtips[CACAOSIM_TUI_VIEW_COUNT] = {
        "Interactive 3-pane hardware builder & inspector [F2]",
        "Full-screen 2D optical schematic & 32-bin spectra [F3]",
        "Hardware mirrors, sensors & optical beams tables [F4]",
        "Live MILK_SHM_DIR streams, FPS units & deployment [F5]",
        "Formatted .bench INI/TOML configuration preview [F6]",
    };

    int tx = 1;
    for (int v = 0; v < CACAOSIM_TUI_VIEW_COUNT; v++)
    {
        int          act = (app->view == v);
        cs_tui_rgb_t fg  = act ? pal->fg_bright : pal->fg_muted;
        cs_tui_rgb_t bg  = act ? pal->bg_selected : pal->bg_panel_alt;
        int          len = cs_put_str(&app->scr, tx, 1, 20, vtabs[v], fg, bg,
                                      act ? CS_ATTR_BOLD : 0u, 1);
        cs_tui_add_hit(app, CS_HIT_VIEW_TAB, v, tx, 1, len, 1, vtips[v]);
        tx += len + 1;
    }

    /* Quick right-side action pills on Row 1 */
    int rx = W - 56;
    if (rx > tx + 2)
    {
        rx += draw_btn(app, rx, 1, "[P:Presets]", CS_HIT_ACT_PRESET, 0, pal->fg_accent,
                       pal->bg_panel_alt, "Load built-in optical bench template [P]") +
              1;
        rx += draw_btn(app, rx, 1, "[s:Save]", CS_HIT_ACT_SAVE, 0, pal->fg_stream,
                       pal->bg_panel_alt, "Save .bench configuration to disk [s]") +
              1;
        rx += draw_btn(app, rx, 1, "[f:Run]", CS_HIT_ACT_DEPLOY_F, 0, pal->fg_active,
                       pal->bg_panel_alt, "Deploy & start fused htsim bench [f]") +
              1;
        rx += draw_btn(app, rx, 1, "[S:Stop]", CS_HIT_ACT_STOP, 0, pal->fg_error,
                       pal->bg_panel_alt, "Stop running cacaosim FPS units [S]") +
              1;
        rx += draw_btn(app, rx, 1, "[?:Opts]", CS_HIT_ACT_OPTREF, 0, pal->fg_proc,
                       pal->bg_panel_alt, "Search all .bench & simulator options [?]") +
              1;
        draw_btn(app, rx, 1, "[h:HELP]", CS_HIT_HELP_BTN, 0, pal->fg_bright, pal->bg_selected,
                 "Show keyboard & mouse shortcuts [h / F1]");
    }
}

/**
 * render_bottom_bars() - Render the 2-row bottom notification banner and shortcut footer
 * @app: Active TUI application state (non-NULL)
 *
 * Role: Paints the final two terminal rows (`rows - 2` and `rows - 1`):
 *   - Row `rows - 2`: priority-cascaded context line displaying (1) severity-colored transient
 *     toast notifications while `now < toast_until_s`, else (2) the full description of the
 *     currently highlighted parameter in `F2:BUILDER`, else (3) the most recent session log
 *     entry.
 *   - Row `rows - 1`: mode-sensitive keyboard cheat-sheet (`editing` vs. normal navigation),
 *     clickable `[m:tips|off]`, `[T:<theme>]`, and `[q:exit]` pills, and a local `HH:MM:SS`
 *     wall-clock readout.
 *
 * Rationale: Dedicating a persistent single-line banner to toast alerts and full parameter
 * descriptions avoids disruptive modal popups for routine actions (saves, undo steps, stepping
 * bounds) and ensures long parameter documentation remains readable even when the Inspector's
 * inline `NOTES` column is clipped on narrow 80-column terminals.
 */
static void render_bottom_bars(cs_tui_app_t *app)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 W   = app->scr.cols;
    int                 ty  = app->scr.rows - 2;
    int                 sy  = app->scr.rows - 1;

    /* Row rows-2: Priority cascade (1: active toast -> 2: selected param tip -> 3: last log) */
    cs_fill_rect(&app->scr, 0, ty, W, 1, pal->fg_muted, pal->bg_panel_alt);
    double now = cs_tui_now_sec();
    if (app->toast[0] != '\0' && now < app->toast_until_s)
    {
        cs_tui_rgb_t tfg = (app->toast_level == 3)
                               ? pal->fg_error
                               : (app->toast_level == 2
                                      ? pal->fg_warn
                                      : (app->toast_level == 1 ? pal->fg_active : pal->fg_bright));
        cs_printf_at(&app->scr, 2, ty, W - 4, tfg, pal->bg_panel_alt, CS_ATTR_BOLD, 1, "%s",
                     app->toast);
    }
    else if (app->view == CACAOSIM_TUI_VIEW_BUILDER && app->param_sel >= 0 &&
             app->param_sel < app->n_params)
    {
        const cs_param_desc_t *pd = &app->params[app->param_sel];
        cs_printf_at(&app->scr, 2, ty, W - 4, pal->fg_muted, pal->bg_panel_alt, 0u, 1,
                     "%s: %s", pd->key, pd->tip);
    }
    else if (app->n_logs > 0)
    {
        cs_printf_at(&app->scr, 2, ty, W - 4, pal->fg_muted, pal->bg_panel_alt, 0u, 1, "%s",
                     app->logs[app->n_logs - 1]);
    }

    /* Row rows-1: milk-CTRL Status Bar with mode-aware key hints and utility toggles */
    cs_fill_rect(&app->scr, 0, sy, W, 1, pal->fg_text, pal->bg_bar);
    const char *hints =
        (app->editing)
            ? " Type value  Enter:Apply  Esc:Cancel  Bksp:Delete"
            : " Tab:Pane  a:Add  c:Dup  K/J:Move  x:Swap  w:Wire  d:Del  u:Undo  s:Save  f:Run";
    char m_lbl[16], t_lbl[20];
    snprintf(m_lbl, sizeof(m_lbl), "[m:%s]", app->show_tooltips ? "tips" : "off");
    snprintf(t_lbl, sizeof(t_lbl), "[T:%s]", pal->name);
    int rbar_w = (int) strlen(m_lbl) + 1 + (int) strlen(t_lbl) + 1 + 8 + 2 + 9;
    int rx     = W - rbar_w;
    cs_put_str(&app->scr, 0, sy, (rx > 2) ? (rx - 1) : (W - 2), hints, pal->fg_muted, pal->bg_bar,
               0u, 1);

    if (rx > 10)
    {
        rx += draw_btn(app, rx, sy, m_lbl, CS_HIT_TOOLTIP_BTN, 0,
                       app->show_tooltips ? pal->fg_active : pal->fg_dim, pal->bg_bar,
                       "Toggle mouse hover tooltips") +
              1;
        rx += draw_btn(app, rx, sy, t_lbl, CS_HIT_THEME_BTN, 0, pal->fg_accent, pal->bg_bar,
                       "Open color theme selector [T]") +
              1;
        rx += draw_btn(app, rx, sy, "[q:exit]", CS_HIT_EXIT_BTN, 0, pal->fg_error, pal->bg_bar,
                       "Quit cacaosim-CTRL [q]") +
              2;

        time_t     wall = time(NULL);
        struct tm *tmv  = localtime(&wall);
        if (tmv != NULL && rx + 8 <= W)
        {
            cs_printf_at(&app->scr, W - 9, sy, 9, pal->fg_dim, pal->bg_bar, 0u, 1, "%02d:%02d:%02d",
                         tmv->tm_hour, tmv->tm_min, tmv->tm_sec);
        }
    }
}

/**
 * render_modal() - Render the active centered modal dialog overlay onto the backbuffer
 * @app: Active TUI application state inspecting `app->modal` (non-NULL)
 *
 * Role: Draws a drop-shadowed centered popup dialog and registers modal hit-test boxes when
 * `app->modal != CS_MODAL_NONE`:
 *   - `CS_MODAL_ADD`: 9-kind optical hardware catalog with `Tab`-cyclable target input beam
 *     (`AUTO` smart splice vs. explicit upstream beam).
 *   - `CS_MODAL_WIRE`: upstream optical beam picker displaying photon flux, flux-weighted mean
 *     wavelength, and an inline 32-bin polychromatic spectrum bar per beam.
 *   - `CS_MODAL_PRESET`: built-in `.bench` hardware template gallery with summary blurbs.
 *   - `CS_MODAL_THEME`: `milk-CTRL` color palette selector with live RGB swatch blocks and
 *     7-bit ASCII border toggle status.
 *   - `CS_MODAL_SAVE` / `CS_MODAL_OPEN`: `.bench` file path text prompt.
 *   - `CS_MODAL_CONFIRM`: unsaved-changes guard dialog before destructive reload/preset/quit.
 *   - `CS_MODAL_HELP`: categorized keyboard and mouse reference card.
 *
 * Rationale: Rendering modals after the main workspace onto the same cell backbuffer preserves
 * the live schematic context visually dimmed behind a drop shadow, and because hit-test boxes
 * are scanned in reverse registration order (`n_hits - 1` down to `0`), modal rows and buttons
 * automatically capture clicks ahead of underlying workspace widgets while `handle_mouse_event`
 * blocks clicks outside the dialog. Showing live 32-bin spectra inside `CS_MODAL_WIRE` prevents
 * accidentally wiring an infrared sensor onto a visible-only reflected arm.
 */
static void render_modal(cs_tui_app_t *app)
{
    if (app->modal == CS_MODAL_NONE)
    {
        return;
    }
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 W   = app->scr.cols;
    int                 H   = app->scr.rows;

    /* 1. Add-Element & Splice Wizard (`a` / `+` or clicking `[+]` on a schematic beam) */
    if (app->modal == CS_MODAL_ADD)
    {
        static const char *const names[9] = {
            "1. [dm]       Zonal Deformable Mirror",
            "2. [ttm]      2-Axis Tip-Tilt Mirror",
            "3. [splitter] Dichroic / Gray Beamsplitter",
            "4. [coro]     Coronagraph + LOWFS Pickoff",
            "5. [pyrwfs]   4-Sided Pyramid WFS",
            "6. [shwfs]    Shack-Hartmann Lenslet WFS",
            "7. [zwfs]     Zernike Phase-Contrast WFS",
            "8. [fpcam]    Focal-Plane Science Camera",
            "9. [wfe]      Static Wavefront Error [OPD m]",
        };
        static const char *const descs[9] = {
            "Actuator grid with ZOH servo dynamics, stroke clip & DAC bits",
            "Fast steering tip-tilt mirror (dm_mode=1) with servo latency",
            "Splits beam into out_refl + out_trans (gray, LP, SP, or BP)",
            "Analytic/custom FPM & pupil masks + reflective FPM or Lyot LOWFS",
            "Modulated/unmodulated diffractive 4-quadrant Pyramid WFS",
            "Diffractive lenslet array Shack-Hartmann wavefront sensor",
            "Babinet Matrix-DFT Zernike phase-contrast wavefront sensor",
            "Polychromatic MFT/FFT science PSF camera + Strehl telemetry",
            "Inline static wavefront error SHM stream in unit of OPD [meter]",
        };

        char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
        uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
        int      mw = (W > 80) ? 76 : (W - 4);
        int      mh = 16;
        int      mx = (W - mw) / 2;
        int      my = (H - mh) / 2;

        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "ADD OPTICAL ELEMENT", "1..9 / Enter  Esc:Cancel",
                    pal->fg_border_focus, pal->fg_title, pal->fg_muted, pal->bg_popup);

        const char *b_lbl = (app->modal_beam_idx >= 0 && (uint32_t) app->modal_beam_idx < nb)
                                ? beams[app->modal_beam_idx]
                                : "AUTO (smart splice after selection)";
        cs_printf_at(&app->scr, mx + 2, my + 1, mw - 4, pal->fg_stream, pal->bg_popup, 0u, 1,
                     "Target Input Beam [Tab to cycle]: %s", b_lbl);
        cs_tui_add_hit(app, CS_HIT_MODAL_EXTRA, 0, mx + 2, my + 1, mw - 4, 1,
                       "Click or press Tab to cycle target input beam");

        for (int i = 0; i < 9; i++)
        {
            int          sel = (app->modal_sel == i);
            cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 3 + i, mw - 2, 1, pal->fg_text, bg);
            cs_printf_at(&app->scr, mx + 2, my + 3 + i, mw - 4,
                         sel ? pal->fg_bright : pal->fg_text, bg, sel ? CS_ATTR_BOLD : 0u, 1,
                         "%-44s", names[i]);
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 3 + i, mw - 2, 1, descs[i]);
        }
        cs_printf_at(&app->scr, mx + 2, my + 13, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1, "%s",
                     descs[(app->modal_sel >= 0 && app->modal_sel < 9) ? app->modal_sel : 0]);
        draw_btn(app, mx + mw - 22, my + 14, "[Enter:Insert]", CS_HIT_MODAL_OK, 0, pal->fg_active,
                 pal->bg_panel_alt, "Insert selected optical element");
        draw_btn(app, mx + mw - 7, my + 14, "[Esc]", CS_HIT_MODAL_CANCEL, 0, pal->fg_error,
                 pal->bg_panel_alt, "Cancel");
        return;
    }

    /* 2. Upstream Optical Beam Rewiring Picker (`w`) with inline 32-bin spectra */
    if (app->modal == CS_MODAL_WIRE)
    {
        char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
        uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
        int      mw = (W > 76) ? 72 : (W - 4);
        int      mh = (int) nb + 6;
        if (mh > H - 4)
        {
            mh = H - 4;
        }
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;

        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "WIRE UPSTREAM OPTICAL BEAM",
                    "Enter:Connect  Esc:Cancel", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        cs_printf_at(&app->scr, mx + 2, my + 1, mw - 4, pal->fg_dim, pal->bg_popup, 0u, 1,
                     "%-18s %-12s %-10s %s", "BEAM NAME", "FLUX [ph/s]", "MEAN LAM", "SPECTRUM");

        int max_r = mh - 4;
        for (int i = 0; i < (int) nb && i < max_r; i++)
        {
            int          sel = (app->modal_sel == i);
            cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 2 + i, mw - 2, 1, pal->fg_text, bg);

            const cacaosim_spec_t *sp = find_beam_spec(app, beams[i]);
            char                   fstr[24] = "-";
            float                  mlam     = 0.0f;
            if (sp != NULL)
            {
                fmt_flux_compact(cacaosim_spec_total(sp), fstr, sizeof(fstr));
                mlam = cs_tui_spec_mean_lambda(sp);
            }
            cs_printf_at(&app->scr, mx + 2, my + 2 + i, 44, sel ? pal->fg_bright : pal->fg_stream,
                         bg, sel ? CS_ATTR_BOLD : 0u, 1, "%-18.18s %-12s %.2f um", beams[i], fstr,
                         mlam);
            if (sp != NULL && mw >= 62)
            {
                cs_draw_spectrum32(&app->scr, mx + 44, my + 2 + i, 20, sp, app->opts.theme, bg);
            }
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 2 + i, mw - 2, 1,
                           "Connect element input to this optical beam");
        }
        return;
    }

    /* 3. Built-in Optical Bench Preset Gallery (`P`) */
    if (app->modal == CS_MODAL_PRESET)
    {
        int mw = (W > 84) ? 80 : (W - 4);
        int mh = CACAOSIM_PRESET_COUNT + 6;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "LOAD HARDWARE TEMPLATE (.bench PRESET)",
                    "Enter:Load  Esc:Cancel", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        for (int i = 0; i < CACAOSIM_PRESET_COUNT; i++)
        {
            int          sel = (app->modal_sel == i);
            cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 2 + i, mw - 2, 1, pal->fg_text, bg);
            cs_printf_at(&app->scr, mx + 2, my + 2 + i, mw - 4,
                         sel ? pal->fg_bright : pal->fg_accent, bg, sel ? CS_ATTR_BOLD : 0u, 1,
                         "%d. %s", i + 1, cacaosim_bench_preset_name(i));
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 2 + i, mw - 2, 1,
                           cacaosim_bench_preset_desc(i));
        }
        cs_printf_at(&app->scr, mx + 2, my + mh - 2, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1,
                     "%s", cacaosim_bench_preset_desc(app->modal_sel));
        return;
    }

    /* 4. Color Theme & 7-bit ASCII Selector (`T`) */
    if (app->modal == CS_MODAL_THEME)
    {
        int mw = 58;
        int mh = CACAOSIM_THEME_COUNT + 6;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "SELECT MILK COLOR THEME",
                    "Enter:Select  a:ASCII", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        for (int i = 0; i < CACAOSIM_THEME_COUNT; i++)
        {
            const cs_tui_pal_t *tp  = cs_tui_get_palette(i);
            int                 sel = (app->modal_sel == i);
            cs_tui_rgb_t        bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 2 + i, mw - 2, 1, pal->fg_text, bg);
            cs_printf_at(&app->scr, mx + 2, my + 2 + i, 38, sel ? pal->fg_bright : pal->fg_text, bg,
                         sel ? CS_ATTR_BOLD : 0u, 1, "%d. %-33s", i + 1, tp->label);
            /* Preview title, stream, active, and accent swatches against each theme's panel bg */
            cs_put_str(&app->scr, mx + 42, my + 2 + i, 3, "###", tp->fg_title, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_put_str(&app->scr, mx + 45, my + 2 + i, 3, "###", tp->fg_stream, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_put_str(&app->scr, mx + 48, my + 2 + i, 3, "###", tp->fg_active, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_put_str(&app->scr, mx + 51, my + 2 + i, 3, "###", tp->fg_accent, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 2 + i, mw - 2, 1, tp->label);
        }
        cs_printf_at(&app->scr, mx + 2, my + mh - 2, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1,
                     "7-bit ASCII borders [press 'a']: %s", app->scr.ascii_only ? "ON" : "OFF");
        return;
    }

    /* 5. File Path Prompt for Save As (`A` / first `s`) and Open (`o`) */
    if (app->modal == CS_MODAL_SAVE || app->modal == CS_MODAL_OPEN)
    {
        int mw = (W > 72) ? 68 : (W - 4);
        int mh = 7;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1,
                    (app->modal == CS_MODAL_SAVE) ? "SAVE .bench CONFIGURATION"
                                                  : "OPEN .bench CONFIGURATION",
                    "Enter:OK  Esc:Cancel", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        cs_put_str(&app->scr, mx + 2, my + 2, mw - 4, "File path:", pal->fg_muted, pal->bg_popup,
                   0u, 1);
        cs_fill_rect(&app->scr, mx + 2, my + 3, mw - 4, 1, pal->fg_bright, pal->bg_input);
        cs_printf_at(&app->scr, mx + 2, my + 3, mw - 5, pal->fg_accent, pal->bg_input, CS_ATTR_BOLD,
                     1, "%s_", app->modal_input);
        draw_btn(app, mx + mw - 18, my + 5, "[Enter:Confirm]", CS_HIT_MODAL_OK, 0, pal->fg_active,
                 pal->bg_panel_alt, "Confirm file operation");
        return;
    }

    /* 6. Unsaved Changes Confirmation Guard (`q`, `P`, `o` when `app->dirty != 0`) */
    if (app->modal == CS_MODAL_CONFIRM)
    {
        int mw = 56;
        int mh = 7;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "UNSAVED CONFIGURATION CHANGES", "y / n",
                    pal->fg_warn, pal->fg_warn, pal->fg_muted, pal->bg_popup);
        cs_put_str(&app->scr, mx + 2, my + 2, mw - 4,
                   "Discard unsaved optical bench edits and continue?", pal->fg_bright,
                   pal->bg_popup, 0u, 1);
        draw_btn(app, mx + 12, my + 4, "[y: Discard & Continue]", CS_HIT_MODAL_OK, 0, pal->fg_error,
                 pal->bg_panel_alt, "Proceed without saving");
        draw_btn(app, mx + 38, my + 4, "[n: Cancel]", CS_HIT_MODAL_CANCEL, 0, pal->fg_active,
                 pal->bg_panel_alt, "Return to builder");
        return;
    }

    /* 7. Interactive Keyboard & Mouse Reference Overlay (`h` / `?` / `F1`) */
    if (app->modal == CS_MODAL_HELP)
    {
        static const char *const hlines[] = {
            "NAVIGATION & VIEWS",
            "  F2..F6            Switch Builder / Schematic / Tables / Status / Source",
            "  Tab / Shift+Tab   Cycle focus between Elements, Schematic & Inspector",
            "  Up / Down / Click Select element, parameter, or click any 2D schematic box",
            "HARDWARE ARRANGER",
            "  a / +             Add element (DM, TTM, BS, Coro, PyWFS, SH, ZWFS, Cam, WFE)",
            "  c                 Duplicate (clone) selected element with auto-wiring",
            "  K / J (Shift+U/D) Move element upstream / downstream (optically splices chain!)",
            "  x                 Swap downstream arms on Beamsplitter (R/T) or Coro (Sci/LOWFS)",
            "  w                 Rewire selected element's input beam from upstream beams",
            "  d / Delete        Remove element & automatically bridge upstream/downstream beams",
            "  u / Ctrl+R        Undo / Redo configuration edits (24-step history)",
            "PARAMETER INSPECTOR & DEPLOYMENT",
            "  1 / 2 / 3         Switch Inspector subtab (Optics, Dyn/Det, SHM/Wiring)",
            "  v                 Toggle Advanced .bench options in Inspector",
            "  ?                 Open searchable Option Reference across all .bench keys",
            "  Enter / Space     Edit parameter inline or cycle enum/boolean value",
            "  Left / Right      Step numeric/enum value ([ / ] for 5x coarse step)",
            "  P / s / A / o / e Presets / Save / Save As / Open / Edit in $EDITOR",
            "  f / M / S / g / t Deploy Fused / Deploy Modular / Stop / InitSHM / AtmTurb",
            "  T / m / q         Theme selector / Toggle hover tooltips / Quit",
        };
        int n_hl = (int) (sizeof(hlines) / sizeof(hlines[0]));
        int mw   = (W > 86) ? 84 : (W - 2);
        int mh   = n_hl + 4;
        if (mh > H - 2)
        {
            mh = H - 2;
        }
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "cacaosim-CTRL KEYBOARD & MOUSE REFERENCE",
                    "?:Options  Esc/h:Close", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        for (int i = 0; i < n_hl && i < mh - 3; i++)
        {
            int hdr = (hlines[i][0] != ' ');
            cs_put_str(&app->scr, mx + 2, my + 2 + i, mw - 4, hlines[i],
                       hdr ? pal->fg_accent : pal->fg_text, pal->bg_popup, hdr ? CS_ATTR_BOLD : 0u,
                       1);
        }
        cs_tui_add_hit(app, CS_HIT_MODAL_CANCEL, 0, mx, my, mw, mh, "Click to close help");
        return;
    }

    /* 8. Searchable Option Reference Overlay (`?` or `[?:Opts]`) */
    if (app->modal == CS_MODAL_OPTIONS)
    {
        size_t total = cacaosim_schema_bench_count();
        size_t matches[256];
        size_t n_match = 0u;
        const char *q  = app->opt_search;
        size_t qlen    = strlen(q);
        for (size_t i = 0u; i < total && n_match < 256u; i++)
        {
            const cacaosim_bench_opt_t *o = cacaosim_schema_bench_get(i);
            if (o == NULL)
            {
                continue;
            }
            if (qlen == 0u)
            {
                matches[n_match++] = i;
                continue;
            }
            const char *fields[5] = { o->section, o->key, o->aliases, o->fps_param, o->description };
            int         hit       = 0;
            for (int f = 0; f < 5 && !hit; f++)
            {
                const char *h = fields[f];
                if (h == NULL)
                {
                    continue;
                }
                for (const char *p = h; *p != '\0'; p++)
                {
                    size_t k = 0u;
                    while (k < qlen && p[k] != '\0' &&
                           tolower((unsigned char) p[k]) == tolower((unsigned char) q[k]))
                    {
                        k++;
                    }
                    if (k == qlen)
                    {
                        hit = 1;
                        break;
                    }
                }
            }
            if (hit)
            {
                matches[n_match++] = i;
            }
        }

        int mw = (W > 108) ? 104 : (W - 4);
        int mh = (H > 26) ? (H - 4) : (H - 2);
        if (mw < 50)
        {
            mw = W - 2;
        }
        if (mh < 12)
        {
            mh = H - 2;
        }
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;

        if (n_match == 0u)
        {
            app->opt_sel    = 0;
            app->opt_scroll = 0;
        }
        else
        {
            if (app->opt_sel >= (int) n_match)
            {
                app->opt_sel = (int) n_match - 1;
            }
            if (app->opt_sel < 0)
            {
                app->opt_sel = 0;
            }
        }

        int list_top  = my + 4;
        int list_rows = mh - 8;
        if (list_rows < 2)
        {
            list_rows = 2;
        }
        if (app->opt_sel < app->opt_scroll)
        {
            app->opt_scroll = app->opt_sel;
        }
        if (app->opt_sel >= app->opt_scroll + list_rows)
        {
            app->opt_scroll = app->opt_sel - list_rows + 1;
        }
        if (app->opt_scroll < 0)
        {
            app->opt_scroll = 0;
        }

        char rbadge[48];
        snprintf(rbadge, sizeof(rbadge), "%zu/%zu  Esc:Close", n_match, total);
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "cacaosim .bench OPTION REFERENCE", rbadge,
                    pal->fg_border_focus, pal->fg_title, pal->fg_muted, pal->bg_popup);

        cs_put_str(&app->scr, mx + 2, my + 1, 9, "Search:", pal->fg_muted, pal->bg_popup, 0u, 1);
        cs_fill_rect(&app->scr, mx + 10, my + 1, mw - 28, 1, pal->fg_bright, pal->bg_input);
        cs_printf_at(&app->scr, mx + 11, my + 1, mw - 30, pal->fg_accent, pal->bg_input,
                     CS_ATTR_BOLD, 1, "%s_", app->opt_search);
        cs_put_str(&app->scr, mx + mw - 17, my + 1, 15, "Ctrl+U:Clear", pal->fg_dim, pal->bg_popup,
                   0u, 1);

        cs_printf_at(&app->scr, mx + 2, my + 3, mw - 4, pal->fg_dim, pal->bg_popup, 0u, 1,
                     "%-8s %-17s %-6s %-11s %-10s %s", "SECTION", "KEY", "TYPE", "DEFAULT",
                     "MODE", "DESCRIPTION");

        for (int r = 0; r < list_rows; r++)
        {
            size_t mi = (size_t) (app->opt_scroll + r);
            if (mi >= n_match)
            {
                break;
            }
            const cacaosim_bench_opt_t *o   = cacaosim_schema_bench_get(matches[mi]);
            int                         ry  = list_top + r;
            int                         sel = ((int) mi == app->opt_sel);
            cs_tui_rgb_t                rbg = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, ry, mw - 2, 1, pal->fg_text, rbg);
            const char *mstr = (o->mode == CACAOSIM_OPT_MODE_FUSED) ? "fused" : "fused+mod";
            cs_printf_at(&app->scr, mx + 2, ry, 8, pal->fg_proc, rbg, 0u, 1, "%-8.8s", o->section);
            cs_printf_at(&app->scr, mx + 11, ry, 17, sel ? pal->fg_bright : pal->fg_fps, rbg,
                         sel ? CS_ATTR_BOLD : 0u, 1, "%-17.17s", o->key);
            cs_printf_at(&app->scr, mx + 29, ry, 6, pal->fg_muted, rbg, 0u, 1, "%-6.6s", o->type);
            cs_printf_at(&app->scr, mx + 36, ry, 11, pal->fg_stream, rbg, 0u, 1, "%-11.11s",
                         o->default_str[0] ? o->default_str : "\"\"");
            cs_printf_at(&app->scr, mx + 48, ry, 10,
                         (o->mode == CACAOSIM_OPT_MODE_FUSED) ? pal->fg_warn : pal->fg_active, rbg,
                         0u, 1, "%-10.10s", mstr);
            if (mw > 61)
            {
                cs_printf_at(&app->scr, mx + 59, ry, mw - 61, sel ? pal->fg_bright : pal->fg_text,
                             rbg, 0u, 1, "%s", o->description);
            }
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, (int) mi, mx + 1, ry, mw - 2, 1, o->description);
        }

        /* Detailed footer for currently highlighted option */
        if (n_match > 0u && app->opt_sel >= 0 && (size_t) app->opt_sel < n_match)
        {
            const cacaosim_bench_opt_t *sel_o = cacaosim_schema_bench_get(matches[app->opt_sel]);
            int                         fy    = my + mh - 3;
            cs_printf_at(&app->scr, mx + 2, fy, mw - 4, pal->fg_accent, pal->bg_popup, CS_ATTR_BOLD,
                         1, "[%s] %s  unit:%s  aliases:%s  fps:%s", sel_o->applies_to, sel_o->key,
                         sel_o->unit[0] ? sel_o->unit : "-",
                         sel_o->aliases[0] ? sel_o->aliases : "-",
                         sel_o->fps_param[0] ? sel_o->fps_param : "-");
            cs_printf_at(&app->scr, mx + 2, fy + 1, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1,
                         "%s%s%s", sel_o->description, sel_o->enum_values[0] ? "  choices: " : "",
                         sel_o->enum_values);
        }
        else
        {
            cs_put_str(&app->scr, mx + 2, my + mh - 3, mw - 4,
                       "No matching .bench options (press Backspace or Ctrl+U to clear filter)",
                       pal->fg_warn, pal->bg_popup, 0u, 1);
        }
        draw_btn(app, mx + mw - 12, my + mh - 2, "[Esc:Close]", CS_HIT_MODAL_CANCEL, 0,
                 pal->fg_error, pal->bg_panel_alt, "Close option reference");
    }
}

/**
 * render_hover_tooltip() - Render a floating mouse-hover tooltip callout near the cursor
 * @app: Active TUI application state holding mouse coordinates and `app->hits` (non-NULL)
 *
 * Role: Scans `app->hits` from front to back (`n_hits - 1` down to `0`) for the topmost
 * interactive widget under `(mouse_x, mouse_y)` carrying a non-empty `tip` string, and paints
 * a screen-clamped single-line callout pill just above (or below, near Row 0..3) the cursor.
 *
 * Rationale: Reusing the exact immediate-mode hit-test boxes registered during widget drawing
 * provides zero-maintenance hover discovery for abbreviated schematic badges (`[LP 0.95]`,
 * `[+]`, SHM dots) and toolbar pills without maintaining parallel tooltip geometry tables.
 * Suppressing tooltips while a modal is open (`app->modal != CS_MODAL_NONE`) prevents background
 * callouts from bleeding over modal overlays.
 */
static void render_hover_tooltip(cs_tui_app_t *app)
{
    if (!app->show_tooltips || app->modal != CS_MODAL_NONE || app->mouse_x < 0 || app->mouse_y < 0)
    {
        return;
    }
    /* Reverse scan matches the topmost overlapping hit-box painted latest in the frame */
    const char *tip = NULL;
    for (int i = app->n_hits - 1; i >= 0; i--)
    {
        const cs_hit_t *h = &app->hits[i];
        if (app->mouse_x >= h->x && app->mouse_x < h->x + h->w && app->mouse_y >= h->y &&
            app->mouse_y < h->y + h->h && h->tip[0] != '\0')
        {
            tip = h->tip;
            break;
        }
    }
    if (tip == NULL)
    {
        return;
    }
    const cs_tui_pal_t *pal  = cs_tui_get_palette(app->opts.theme);
    int                 tlen = (int) strlen(tip) + 2;
    if (tlen > app->scr.cols - 4)
    {
        tlen = app->scr.cols - 4;
    }
    /* Flip below the cursor near the top header rows and clamp horizontally inside screen edges */
    int tx = app->mouse_x + 2;
    int ty = (app->mouse_y > 3) ? (app->mouse_y - 1) : (app->mouse_y + 1);
    if (tx + tlen >= app->scr.cols)
    {
        tx = app->scr.cols - tlen - 1;
    }
    if (tx < 1)
    {
        tx = 1;
    }
    cs_fill_rect(&app->scr, tx, ty, tlen, 1, pal->fg_bright, pal->bg_popup);
    cs_printf_at(&app->scr, tx + 1, ty, tlen - 2, pal->fg_accent, pal->bg_popup, 0u, 1, "%s", tip);
}

/**
 * cs_tui_render_frame() - Execute one complete immediate-mode TUI frame into the backbuffer
 * @app: Active TUI application state (safe no-op if NULL)
 *
 * Role: Master per-frame rendering entry point shared by both the interactive `cacaosim_tui_run()`
 * event loop and headless `cacaosim_tui_render_headless()` snapshots:
 *   1. Lazily refreshes zero-FFT spectral/DAG analysis and the 16x16 pupil mask if dirty.
 *   2. Rebuilds the active node/category `cs_param_desc_t` descriptor table (`app->params`).
 *   3. Resets the immediate-mode hit-test table (`app->n_hits = 0`) and clears `app->scr`.
 *   4. Paints top chrome (`render_header_and_tabs`), the active `F2..F6` workspace body,
 *      bottom status bars (`render_bottom_bars`), any active modal (`render_modal`), and the
 *      floating mouse tooltip (`render_hover_tooltip`).
 *
 * Rationale: Rebuilding parameter descriptors and hit-test rectangles unconditionally at the
 * start of every frame enforces strict immediate-mode UI invariants: the screen backbuffer,
 * mouse click targets, and keyboard parameter indices can never drift out of sync after undo,
 * hot-reload, terminal `SIGWINCH` resizes, or headless `--dump` captures, while `cs_scr_flush()`
 * subsequently diffs `cells` against `prev` so unchanged terminal cells emit zero ANSI bytes.
 */
void cs_tui_render_frame(cs_tui_app_t *app)
{
    if (app == NULL)
    {
        return;
    }
    cs_tui_refresh_analysis(app);
    app->n_params = cs_params_build(&app->cfg, app->sel_item, app->subtab, app->show_advanced,
                                    app->params, CS_MAX_PARAMS);
    app->n_hits   = 0;

    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    cs_scr_clear(&app->scr, pal->fg_text, pal->bg_panel);

    render_header_and_tabs(app);

    int bx = 0;
    int by = 2;
    int bw = app->scr.cols;
    int bh = app->scr.rows - 4;

    if (bh >= 6)
    {
        switch (app->view)
        {
        case CACAOSIM_TUI_VIEW_SCHEMATIC:
            cs_render_view_schematic(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_TABLES:
            cs_render_view_tables(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_STATUS:
            cs_render_view_status(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_SOURCE:
            cs_render_view_source(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_BUILDER:
        default:
            render_view_builder(app, bx, by, bw, bh);
            break;
        }
    }

    render_bottom_bars(app);
    render_modal(app);
    render_hover_tooltip(app);
}
