// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_params.c
 * @brief   Declarative parameter inspector tables, formatting, stepping, and validation
 *
 * Architectural Role in cacaosim-CTRL:
 *   Implements the schema-driven parameter reflection, string serialization, incremental stepping,
 *   and validated mutation layer backing the `F2:BUILDER` Inspector pane across `[bench]` entrance
 *   settings and all optical node types (`dm` / `ttm`, `wfe`, `splitter`, `coro`, `pyrwfs`,
 *   `shwfs`, `zwfs`, `fpcam`). Organizes every element's parameters into three consistent subtabs:
 *   - Subtab `0` (`1:OPTICS`): Aperture geometry, passbands, masks, actuator grids, WFS optics.
 *   - Subtab `1` (`2:DYN/DET`): Master/camera cadences, ZOH servo dynamics, photometry & RON.
 *   - Subtab `2` (`3:SHM/WIRING`): Section ID, beam connectivity, SHM streams, 2D shift/rot.
 *
 * Design Rationale & Motivation:
 *   1. Data-Driven Conditional Schema vs Static Offset Tables: Many AO node parameters are
 *      meaningful only in specific operating modes (e.g., `nact_x`/`nact_y` only for Zonal DM,
 *      `lambda_cut`/`band_width` only for dichroic splitters, `vortex_charge` vs `fpm_rad_ld` vs
 *      `fpm_pixscale` across coronagraph architectures, manual `flux_ph` only when
 *      `use_beam_flux == OFF`). Building `cs_param_desc_t` rows dynamically in `cs_params_build()`
 *      keeps the UI decluttered while allowing a single generic widget renderer and input loop in
 *      `cacaosim_tui_ansi.c` / `cacaosim_tui_render.c` (using the custom TrueColor + UTF-8 cell
 *      canvas, SGR 1006 mouse wheel/click regions, and `DECSET 2026` synchronized diffing) to edit
 *      over 80 distinct hardware parameters without bespoke per-node dialog code.
 *   2. Coupled Invariants & Topological Side Effects: Routing both inline text edits and
 *      arrow/wheel steps through `cs_param_set_str()` enforces cross-field invariants in one place:
 *      even-integer grid rounding (`nwf`, `vortex_charge`), microsecond-to-second servo scaling
 *      (`tau_us` <-> `tau_sec`), dual-struct registration/bandwidth mirroring, unique
 *      `[<type>.<id>]` renames (`cacaosim_bench_rename_node()`), and automatic downstream consumer
 *      rewiring (`cacaosim_bench_rewire_beam_out()`) whenever a produced optical beam is renamed.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bench/cacaosim_tui_internal.h"

/**
 * @brief Internal parameter dispatch identifiers (`cs_param_desc_t.id`).
 *
 * Partitioned into non-overlapping numeric blocks (`1..99` `[bench]`, `100..199` common node
 * wiring & 2D pupil registration, `200..299` DM/TTM hardware, `300..399` beamsplitter, `400..499`
 * coronagraph, `500..599` WFS & focal-plane cameras, `600..699` static WFE `[m]`) so a single
 * `switch (pd->id)` in `cs_param_format_val()` and `cs_param_set_str()` dispatches cleanly.
 */
enum
{
    /* [bench] global entrance telescope, star, atmosphere, and master clock parameters */
    PID_B_NAME = 1,
    PID_B_NWF,
    PID_B_PUP_DIAM,
    PID_B_COBS,
    PID_B_NSPIDER,
    PID_B_SPIDER_W,
    PID_B_SPIDER_DEG,
    PID_B_SPIDER_OFF,
    PID_B_TEL_DIAM,
    PID_B_STAR_MAG,
    PID_B_STAR_TEFF,
    PID_B_TPUT_TEL,
    PID_B_MASTER_HZ,
    PID_B_WF_UNIT,
    PID_B_REF_LAMBDA,
    PID_B_TURB_GAIN,
    PID_B_BACKEND,
    PID_B_OUT_BEAM,
    PID_B_IN_WF,
    PID_B_IN_PUP,
    PID_B_IN_AMP,
    PID_B_IN_NCPA,

    /* Node common section ID, beam/SHM wiring, throughput, and 2D affine pupil registration */
    PID_N_ID = 100,
    PID_N_IN_BEAM,
    PID_N_OUT_BEAM,
    PID_N_OUT_REFL,
    PID_N_OUT_TRANS,
    PID_N_OUT_LOWFS,
    PID_N_IN_CMD,
    PID_N_OUT_OPD,
    PID_N_IN_FLAT,
    PID_N_OUT_CAM,
    PID_N_OUT_LONG,
    PID_N_IN_NCPA,
    PID_N_IN_LYOT,
    PID_N_IN_WFE,
    PID_N_TPUT,
    PID_N_SHIFT_X,
    PID_N_SHIFT_Y,
    PID_N_ROT_DEG,
    PID_N_MAG_SCALE,

    /* Deformable Mirror / 2-Axis Tip-Tilt Mirror geometry & discrete ZOH servo dynamics */
    PID_DM_MODE = 200,
    PID_DM_NACT_X,
    PID_DM_NACT_Y,
    PID_DM_IF_COUPLING,
    PID_DM_MAX_STROKE,
    PID_DM_REFL_SCALE,
    PID_DM_DYN_MODE,
    PID_DM_TAU_US,
    PID_DM_F0_HZ,
    PID_DM_ZETA,
    PID_DM_DELAY,
    PID_DM_DAC_BITS,
    PID_DM_NONLIN,

    /* Gray & Dichroic (LP / SP / Band-Reflect) Beamsplitter parameters */
    PID_BS_MODE = 300,
    PID_BS_LAMBDA_CUT,
    PID_BS_BAND_WIDTH,
    PID_BS_ROLL_UM,
    PID_BS_R_GRAY,
    PID_BS_T_GRAY,
    PID_BS_EFF_REFL,
    PID_BS_EFF_TRANS,
    PID_BS_NCPA_REFL,
    PID_BS_NCPA_TRANS,

    /* Coronagraph focal/pupil/Lyot masks & dual-mode LOWFS pickoff parameters */
    PID_CO_MODE = 400,
    PID_CO_LAMBDA,
    PID_CO_BW_FRAC,
    PID_CO_NLAMBDA,
    PID_CO_PHA_UNIT,
    PID_CO_FPM_RAD,
    PID_CO_FPM_AMP,
    PID_CO_FPM_PHASE,
    PID_CO_FPM_PIXSCALE,
    PID_CO_PAD_FACTOR,
    PID_CO_VORTEX_Q,
    PID_CO_LOWFS_MODE,
    PID_CO_LOWFS_PUP,
    PID_CO_LOWFS_REFL,
    PID_CO_LYOT_OUTER,
    PID_CO_LYOT_OBS,
    PID_CO_LYOT_NSP,
    PID_CO_LYOT_SPW,
    PID_CO_LYOT_SPDEG,
    PID_CO_LYOT_SPOFF,
    PID_CO_IN_PUP_AMP,
    PID_CO_IN_PUP_PHA,
    PID_CO_IN_FPM_AMP,
    PID_CO_IN_FPM_PHA,
    PID_CO_IN_LYOT_PHA,

    /* Wavefront & Focal-Plane Sensors (PyWFS / SHWFS / ZWFS / FPCAM) + detector readout */
    PID_S_NCAM = 500,
    PID_S_LAMBDA,
    PID_S_BW_FRAC,
    PID_S_NLAMBDA,
    PID_S_NAVG,
    PID_S_RATE_HZ,
    PID_S_QE,
    PID_S_USE_BFLUX,
    PID_S_FLUX_PH,
    PID_S_RON_E,
    PID_S_PHOT_NOISE,
    PID_S_OUT_DTYPE,
    PID_S_PAD_FACTOR,
    PID_PYR_MOD_RAD,
    PID_PYR_MOD_NPTS,
    PID_PYR_PUP_SEP,
    PID_PYR_ROOF_LD,
    PID_SH_NSUB,
    PID_SH_PIX_SUB,
    PID_ZW_DIMPLE_LD,
    PID_ZW_DIMPLE_RAD,
    PID_ZW_NMFT,
    PID_FP_PROP_MODE,
    PID_FP_PIXSCALE,
    PID_FP_NLAMBDA,

    /* Static Wavefront Error ([wfe.<id>], OPD in [meter]) synthesis & scaling parameters */
    PID_WFE_RMS_NM = 600,
    PID_WFE_DEFOCUS_NM,
    PID_WFE_ALPHA,
    PID_WFE_GAIN,
    PID_WFE_SEED
};

/**
 * add_pd() - Append a scalar, boolean, string, beam, or ID parameter descriptor to @arr
 * @arr:    Destination descriptor array
 * @n:      In/out active count within @arr
 * @max_n:  Maximum capacity of @arr
 * @id:     Internal `PID_*` dispatch tag
 * @key:    `.bench` INI property name
 * @label:  Human-readable UI row title
 * @unit:   Physical/logical unit string (NULL normalized to `""`)
 * @type:   Value/widget classification (`cs_ptype_t`)
 * @min_v:  Inclusive lower clamp bound for numeric types
 * @max_v:  Inclusive upper clamp bound for numeric types
 * @step_v: Fine keyboard/wheel step increment
 * @tip:    Contextual help string displayed in status/hover tooltip (NULL normalized to `""`)
 *
 * Role:
 *   Bounds-checked builder helper used throughout `cs_params_build()` to populate `cs_param_desc_t`
 *   entries concisely.
 *
 * Rationale:
 *   Normalizing NULL `unit`/`tip` pointers to `""` and guarding `*n < max_n` keeps rendering code
 *   free of NULL checks and prevents buffer overruns if a subtab grows toward `CS_MAX_PARAMS`.
 */
static void add_pd(
    cs_param_desc_t *arr,
    int             *n,
    int              max_n,
    int              id,
    const char      *key,
    const char      *label,
    const char      *unit,
    cs_ptype_t       type,
    double           min_v,
    double           max_v,
    double           step_v,
    const char      *tip)
{
    if (*n >= max_n)
    {
        return;
    }
    cs_param_desc_t *p = &arr[(*n)++];
    memset(p, 0, sizeof(*p));
    p->id     = id;
    p->key    = key;
    p->label  = label;
    p->unit   = (unit != NULL) ? unit : "";
    p->tip    = (tip != NULL) ? tip : "";
    p->type   = type;
    p->min_v  = min_v;
    p->max_v  = max_v;
    p->step_v = step_v;
}

/**
 * add_enum_pd() - Append an enumerated-choice parameter descriptor (`<= 4` options) to @arr
 * @arr:    Destination descriptor array
 * @n:      In/out active count within @arr
 * @max_n:  Maximum capacity of @arr
 * @id:     Internal `PID_*` dispatch tag
 * @key:    `.bench` INI property name
 * @label:  Human-readable UI row title
 * @n_enum: Number of active option strings `[1..4]`
 * @e0:     Option 0 canonical string
 * @e1:     Option 1 canonical string (or NULL)
 * @e2:     Option 2 canonical string (or NULL)
 * @e3:     Option 3 canonical string (or NULL)
 * @tip:    Contextual help tooltip string
 *
 * Role:
 *   Registers discrete `CS_PTYPE_ENUM` rows (e.g., `wf_unit`, `backend`, `dm_mode`, `dyn_mode`,
 *   `splitter.mode`, `prop_mode`, `out_dtype`) supporting cyclic arrow/wheel stepping and
 *   case-insensitive string entry.
 *
 * Rationale:
 *   Embedding canonical option names directly in the descriptor lets `cs_param_step()` and
 *   `cs_param_set_str()` validate and cycle any enum generically without per-field option tables.
 */
static void add_enum_pd(
    cs_param_desc_t *arr,
    int             *n,
    int              max_n,
    int              id,
    const char      *key,
    const char      *label,
    int              n_enum,
    const char      *e0,
    const char      *e1,
    const char      *e2,
    const char      *e3,
    const char      *tip)
{
    if (*n >= max_n)
    {
        return;
    }
    cs_param_desc_t *p = &arr[(*n)++];
    memset(p, 0, sizeof(*p));
    p->id            = id;
    p->key           = key;
    p->label         = label;
    p->unit          = "";
    p->tip           = (tip != NULL) ? tip : "";
    p->type          = CS_PTYPE_ENUM;
    p->min_v         = 0.0;
    p->max_v         = (double) (n_enum - 1);
    p->step_v        = 1.0;
    p->n_enum        = n_enum;
    p->enum_names[0] = e0;
    p->enum_names[1] = e1;
    p->enum_names[2] = e2;
    p->enum_names[3] = e3;
}

/**
 * add_enum6_pd() - Append a 6-option enumerated parameter descriptor to @arr
 * @arr:   Destination descriptor array
 * @n:     In/out active count within @arr
 * @max_n: Maximum capacity of @arr
 * @id:    Internal `PID_*` dispatch tag
 * @key:   `.bench` INI property name
 * @label: Human-readable UI row title
 * @e0:    Option 0 canonical string
 * @e1:    Option 1 canonical string
 * @e2:    Option 2 canonical string
 * @e3:    Option 3 canonical string
 * @e4:    Option 4 canonical string
 * @e5:    Option 5 canonical string
 * @tip:   Contextual help tooltip string
 *
 * Role:
 *   Registers 6-way enums such as coronagraph FPM architecture (`lyot_spot`, `phase_dimple`,
 *   `vortex`, `4qpm`, `custom_mft`, `custom_fft`).
 *
 * Rationale:
 *   Keeps the common `<= 4`-option `add_enum_pd()` call signature concise while accommodating wider
 *   enums via `enum_names[4..5]`.
 */
static void add_enum6_pd(
    cs_param_desc_t *arr,
    int             *n,
    int              max_n,
    int              id,
    const char      *key,
    const char      *label,
    const char      *e0,
    const char      *e1,
    const char      *e2,
    const char      *e3,
    const char      *e4,
    const char      *e5,
    const char      *tip)
{
    add_enum_pd(arr, n, max_n, id, key, label, 6, e0, e1, e2, e3, tip);
    if (*n > 0)
    {
        arr[*n - 1].enum_names[4] = e4;
        arr[*n - 1].enum_names[5] = e5;
    }
}

/**
 * cs_params_build() - Construct active Inspector parameter descriptors for `(sel_item, subtab)`
 * @cfg:      Source `.bench` configuration (must not be NULL)
 * @sel_item: Selected item (`<= 0` or out-of-range selects `[bench]`; `1..n_nodes` selects node)
 * @subtab:   Inspector category (`0` = OPTICS, `1` = DYN/DET, `2` = SHM/WIRING)
 * @out:      Destination `cs_param_desc_t` array (must not be NULL)
 * @max_out:  Capacity of @out (`<= CS_MAX_PARAMS`)
 *
 * Role:
 *   Projects the currently selected `[bench]` or optical node configuration onto an ordered list of
 *   typed UI rows tailored to the active subtab and operating mode.
 *
 * Rationale:
 *   Evaluating conditional visibility rules live against `cfg` (e.g., hiding zonal `nact_x` and
 *   `nact_y` when `dm_mode == 1` TTM, showing `band_width` only for `dichroic_bp`, switching
 *   between analytic/vortex/custom coronagraph parameters, and hiding manual `flux_ph` when
 *   `use_beam_flux != 0`) guarantees operators only see parameters that actively affect physics.
 *
 * Return: Number of populated descriptors in @out `[0..max_out]`.
 */
int cs_params_build(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    int                         subtab,
    cs_param_desc_t            *out,
    int                         max_out)
{
    if (cfg == NULL || out == NULL || max_out <= 0)
    {
        return 0;
    }
    int n = 0;

    /* Entrance [bench] section: Subtab 0=Aperture, 1=Clock/Star/Atm/Backend, 2=Entrance SHM */
    if (sel_item <= 0 || (uint32_t) (sel_item - 1) >= cfg->n_nodes)
    {
        if (subtab == 0)
        {
            add_pd(out, &n, max_out, PID_B_NAME, "name", "Bench Name", "", CS_PTYPE_STR, 0, 0, 0,
                   "Bench configuration identifier");
            add_pd(out, &n, max_out, PID_B_NWF, "nwf", "Pupil Grid Size", "px", CS_PTYPE_UINT, 16,
                   1024, 16, "Square pupil wavefront grid dimension [pixels]");
            add_pd(out, &n, max_out, PID_B_TEL_DIAM, "tel_diam_m", "Telescope Diam", "m",
                   CS_PTYPE_FLOAT, 0.2, 50.0, 0.2, "Primary mirror outer diameter [meters]");
            add_pd(out, &n, max_out, PID_B_PUP_DIAM, "pup_diam", "Pupil Fraction", "nwf",
                   CS_PTYPE_FLOAT, 0.2, 1.0, 0.0125, "Outer pupil diameter as fraction of nwf");
            add_pd(out, &n, max_out, PID_B_COBS, "cobs", "Central Obs", "frac", CS_PTYPE_FLOAT,
                   0.0, 0.90, 0.02, "Central secondary obstruction ratio [0..0.90]");
            add_pd(out, &n, max_out, PID_B_NSPIDER, "nspider", "Spider Vanes", "", CS_PTYPE_INT, 0,
                   12, 1, "Number of secondary support spider vanes");
            add_pd(out, &n, max_out, PID_B_SPIDER_W, "spider_w", "Spider Width", "D",
                   CS_PTYPE_FLOAT, 0.0, 0.15, 0.002, "Spider vane width as fraction of pupil diam");
            add_pd(out, &n, max_out, PID_B_SPIDER_DEG, "spider_deg", "Spider Angle", "deg",
                   CS_PTYPE_FLOAT, 0.0, 180.0, 5.0, "Azimuth angle of first spider vane [deg]");
            add_pd(out, &n, max_out, PID_B_SPIDER_OFF, "spider_offset", "Spider Offset", "D",
                   CS_PTYPE_FLOAT, 0.0, 0.45, 0.005,
                   "V-spider X-vertex offset fraction of pupil diam (0 = radial)");
        }
        else if (subtab == 1)
        {
            add_pd(out, &n, max_out, PID_B_MASTER_HZ, "master_hz", "Master Clock", "Hz",
                   CS_PTYPE_FLOAT, 1.0, 20000.0, 100.0, "Master bench simulation loop rate [Hz]");
            add_pd(out, &n, max_out, PID_B_STAR_MAG, "star_mag", "Guide Star Mag", "mV",
                   CS_PTYPE_FLOAT, -2.0, 22.0, 0.5, "Apparent Vega V-band magnitude");
            add_pd(out, &n, max_out, PID_B_STAR_TEFF, "star_teff", "Star Teff", "K",
                   CS_PTYPE_FLOAT, 2000.0, 35000.0, 250.0, "Blackbody effective temperature [K]");
            add_pd(out, &n, max_out, PID_B_TPUT_TEL, "tput_tel", "Telescope Tput", "",
                   CS_PTYPE_FLOAT, 0.01, 1.0, 0.05, "Entrance telescope + relay transmission");
            add_pd(out, &n, max_out, PID_B_TURB_GAIN, "turb_gain", "Turbulence Gain", "x",
                   CS_PTYPE_FLOAT, 0.0, 10.0, 0.1, "Atmospheric phase amplitude multiplier");
            add_enum_pd(out, &n, max_out, PID_B_WF_UNIT, "wf_unit", "Input WF Unit", 2, "um",
                        "rad", NULL, NULL, "Units of input atmosphere stream (OPD um or rad)");
            add_pd(out, &n, max_out, PID_B_REF_LAMBDA, "ref_lambda", "Phase Ref Lam", "um",
                   CS_PTYPE_FLOAT, 0.30, 3.0, 0.05, "Reference wavelength when wf_unit == rad");
            add_enum_pd(out, &n, max_out, PID_B_BACKEND, "backend", "Compute Backend", 3, "auto",
                        "cpu", "cuda", NULL, "Execution engine (0=auto, 1=CPU, 2=CUDA GPU)");
        }
        else
        {
            add_pd(out, &n, max_out, PID_B_OUT_BEAM, "out_beam", "Entrance Beam", "", CS_PTYPE_STR,
                   0, 0, 0, "Root optical beam name emitted by telescope");
            add_pd(out, &n, max_out, PID_B_IN_WF, "in_wf", "Atm Phase SHM", "shm", CS_PTYPE_STR, 0,
                   0, 0, "Input atmospheric turbulence phase/OPD stream");
            add_pd(out, &n, max_out, PID_B_IN_PUP, "in_pup", "Pupil Amp SHM", "shm", CS_PTYPE_STR,
                   0, 0, 0, "Input anti-aliased pupil amplitude stream");
            add_pd(out, &n, max_out, PID_B_IN_AMP, "in_amp", "Scint Amp SHM", "shm", CS_PTYPE_STR,
                   0, 0, 0, "Optional scintillation amplitude stream");
            add_pd(out, &n, max_out, PID_B_IN_NCPA, "in_ncpa", "Entrance NCPA", "shm", CS_PTYPE_STR,
                   0, 0, 0, "Optional entrance static NCPA OPD map [um]");
        }
        return n;
    }

    const cacaosim_node_cfg_t *nd = &cfg->nodes[sel_item - 1];

    /* Subtab 2: SHM & Beam Wiring + 2D Pupil Registration (shared across node kinds) */
    if (subtab == 2)
    {
        add_pd(out, &n, max_out, PID_N_ID, "id", "Element ID", "", CS_PTYPE_ID, 0, 0, 0,
               "Unique section identifier [<type>.<id>]");
        add_pd(out, &n, max_out, PID_N_IN_BEAM, "in", "Input Beam [w]", "beam", CS_PTYPE_BEAM, 0, 0,
               0, "Upstream optical beam (press Enter or 'w' to pick)");

        if (nd->kind == CACAOSIM_NODE_DM)
        {
            add_pd(out, &n, max_out, PID_N_OUT_BEAM, "out", "Output Beam", "beam", CS_PTYPE_STR, 0,
                   0, 0, "Downstream reflected optical beam name");
            add_pd(out, &n, max_out, PID_N_IN_CMD, "in_cmd", "Command SHM", "shm", CS_PTYPE_STR, 0,
                   0, 0, "Actuator command input SHM stream [um]");
            add_pd(out, &n, max_out, PID_N_OUT_OPD, "out_opd", "Surface OPD SHM", "shm",
                   CS_PTYPE_STR, 0, 0, 0, "Optional output 2D DM surface OPD stream");
            add_pd(out, &n, max_out, PID_N_IN_FLAT, "in_flat", "Flat Offset SHM", "shm",
                   CS_PTYPE_STR, 0, 0, 0, "Optional static flat command offset stream");
        }
        else if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            add_pd(out, &n, max_out, PID_N_OUT_REFL, "out_refl", "Reflected Beam", "beam",
                   CS_PTYPE_STR, 0, 0, 0, "Reflected arm optical beam name");
            add_pd(out, &n, max_out, PID_N_OUT_TRANS, "out_trans", "Transmitted Beam", "beam",
                   CS_PTYPE_STR, 0, 0, 0, "Transmitted arm optical beam name");
            add_pd(out, &n, max_out, PID_BS_NCPA_REFL, "ncpa_refl", "Refl NCPA SHM", "shm",
                   CS_PTYPE_STR, 0, 0, 0, "Optional reflected arm static NCPA stream");
            add_pd(out, &n, max_out, PID_BS_NCPA_TRANS, "ncpa_trans", "Trans NCPA SHM", "shm",
                   CS_PTYPE_STR, 0, 0, 0, "Optional transmitted arm static NCPA stream");
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            add_pd(out, &n, max_out, PID_N_OUT_BEAM, "out", "Post-Lyot Beam", "beam", CS_PTYPE_STR,
                   0, 0, 0, "Post-Lyot coronagraphic output beam");
            add_pd(out, &n, max_out, PID_N_OUT_LOWFS, "out_lowfs", "LOWFS Pickoff", "beam",
                   CS_PTYPE_STR, 0, 0, 0, "Reflected FPM or Lyot stop pickoff beam for LOWFS");
            add_pd(out, &n, max_out, PID_CO_IN_PUP_AMP, "in_pup_amp", "Pupil Amp Mask", "mask",
                   CS_PTYPE_STR, 0, 0, 0, "Custom 2D/3D pupil apodizer amplitude (FITS or SHM)");
            add_pd(out, &n, max_out, PID_CO_IN_PUP_PHA, "in_pup_pha", "Pupil Pha Mask", "mask",
                   CS_PTYPE_STR, 0, 0, 0, "Custom 2D/3D pupil apodizer phase/OPD (FITS or SHM)");
            add_pd(out, &n, max_out, PID_CO_IN_FPM_AMP, "in_fpm_amp", "Focal Amp Mask", "mask",
                   CS_PTYPE_STR, 0, 0, 0, "Custom 2D/3D focal mask amplitude (FITS or SHM)");
            add_pd(out, &n, max_out, PID_CO_IN_FPM_PHA, "in_fpm_pha", "Focal Pha Mask", "mask",
                   CS_PTYPE_STR, 0, 0, 0, "Custom 2D/3D focal mask phase/OPD (FITS or SHM)");
            add_pd(out, &n, max_out, PID_N_IN_LYOT, "in_lyot", "Lyot Amp Mask", "mask",
                   CS_PTYPE_STR, 0, 0, 0, "Custom 2D/3D Lyot stop amplitude (FITS or SHM)");
            add_pd(out, &n, max_out, PID_CO_IN_LYOT_PHA, "in_lyot_pha", "Lyot Pha Mask", "mask",
                   CS_PTYPE_STR, 0, 0, 0, "Custom 2D/3D Lyot stop phase/OPD (FITS or SHM)");
        }
        else if (nd->kind == CACAOSIM_NODE_WFE)
        {
            add_pd(out, &n, max_out, PID_N_OUT_BEAM, "out", "Output Beam", "beam", CS_PTYPE_STR, 0,
                   0, 0, "Downstream optical beam with static WFE applied");
            add_pd(out, &n, max_out, PID_N_IN_WFE, "in_wfe", "Static WFE SHM", "m", CS_PTYPE_STR, 0,
                   0, 0, "Input 2D static wavefront error OPD stream [meter]");
        }
        else
        {
            add_pd(out, &n, max_out, PID_N_OUT_CAM, "out_cam", "Camera SHM", "shm", CS_PTYPE_STR, 0,
                   0, 0, "Output detector frame SHM stream name");
            if (nd->kind == CACAOSIM_NODE_FPCAM)
            {
                add_pd(out, &n, max_out, PID_N_OUT_LONG, "out_long", "Long-Exp SHM", "shm",
                       CS_PTYPE_STR, 0, 0, 0, "Cumulative long-exposure PSF stream");
            }
            add_pd(out, &n, max_out, PID_N_IN_NCPA, "in_ncpa", "Arm NCPA SHM", "shm", CS_PTYPE_STR,
                   0, 0, 0, "Optional sensor-arm static NCPA OPD stream");
        }

        /* Splitters carry per-arm registration; all single-input elements share 2D reg fields */
        if (nd->kind != CACAOSIM_NODE_SPLITTER)
        {
            add_pd(out, &n, max_out, PID_N_SHIFT_X, "shift_x", "Pupil Shift X", "px",
                   CS_PTYPE_FLOAT, -32.0, 32.0, 0.25, "Lateral X misregistration [pupil pixels]");
            add_pd(out, &n, max_out, PID_N_SHIFT_Y, "shift_y", "Pupil Shift Y", "px",
                   CS_PTYPE_FLOAT, -32.0, 32.0, 0.25, "Lateral Y misregistration [pupil pixels]");
            add_pd(out, &n, max_out, PID_N_ROT_DEG, "rot_deg", "Pupil Rotation", "deg",
                   CS_PTYPE_FLOAT, -180.0, 180.0, 1.0, "In-plane pupil clocking angle [degrees]");
            add_pd(out, &n, max_out, PID_N_MAG_SCALE, "mag_scale", "Magnification", "x",
                   CS_PTYPE_FLOAT, 0.50, 1.50, 0.01, "Radial pupil magnification factor");
        }
        return n;
    }

    /* Node-specific Subtab 0 (1:OPTICS) and Subtab 1 (2:DYN/DET) */
    if (nd->kind == CACAOSIM_NODE_DM)
    {
        if (subtab == 0)
        {
            add_enum_pd(out, &n, max_out, PID_DM_MODE, "dm_mode", "Mirror Type", 2, "Zonal DM",
                        "2-Axis TTM", NULL, NULL, "0 = Zonal actuator grid, 1 = Tip-Tilt mirror");
            /* Zonal actuator grid dimensions & Gaussian coupling apply only when dm_mode == 0 */
            if (nd->dm.dm_mode == 0)
            {
                add_pd(out, &n, max_out, PID_DM_NACT_X, "nact_x", "Actuators X", "act",
                       CS_PTYPE_UINT, 2, 128, 2, "Actuator count along X across pupil");
                add_pd(out, &n, max_out, PID_DM_NACT_Y, "nact_y", "Actuators Y", "act",
                       CS_PTYPE_UINT, 2, 128, 2, "Actuator count along Y across pupil");
                add_pd(out, &n, max_out, PID_DM_IF_COUPLING, "if_coupling", "IF Coupling", "",
                       CS_PTYPE_FLOAT, 0.02, 0.60, 0.02, "Gaussian inter-actuator coupling");
            }
            add_pd(out, &n, max_out, PID_DM_MAX_STROKE, "max_stroke", "Max Stroke", "um",
                   CS_PTYPE_FLOAT, 0.1, 100.0, 0.5, "Symmetric mechanical stroke clip [+/- um]");
            add_pd(out, &n, max_out, PID_DM_REFL_SCALE, "refl_scale", "OPD/Surf Scale", "x",
                   CS_PTYPE_FLOAT, 1.0, 4.0, 0.1, "OPD per surface displacement (2.0 = normal)");
            add_pd(out, &n, max_out, PID_N_TPUT, "tput", "Reflectivity", "", CS_PTYPE_FLOAT, 0.05,
                   1.0, 0.01, "Mirror optical intensity reflectivity [0..1]");
        }
        else
        {
            add_enum_pd(out, &n, max_out, PID_DM_DYN_MODE, "dyn_mode", "Servo Dynamics", 3,
                        "instant", "rc", "osc", NULL, "Discrete ZOH actuator temporal response");
            add_pd(out, &n, max_out, PID_DM_TAU_US, "tau_us", "RC Time Const", "us", CS_PTYPE_FLOAT,
                   0.0, 20000.0, 20.0, "1st-order RC settling time constant [microseconds]");
            add_pd(out, &n, max_out, PID_DM_F0_HZ, "f0_hz", "Resonance f0", "Hz", CS_PTYPE_FLOAT,
                   50.0, 50000.0, 250.0, "2nd-order natural resonance frequency [Hz]");
            add_pd(out, &n, max_out, PID_DM_ZETA, "zeta", "Damping Ratio", "", CS_PTYPE_FLOAT, 0.05,
                   2.0, 0.05, "2nd-order damping ratio (0.7 = critically damped)");
            add_pd(out, &n, max_out, PID_DM_DELAY, "delay_frames", "Frame Delay", "fr",
                   CS_PTYPE_FLOAT, 0.0, 7.0, 0.25, "Fractional master-clock latency [frames]");
            add_pd(out, &n, max_out, PID_DM_DAC_BITS, "dac_bits", "DAC Resolution", "bits",
                   CS_PTYPE_INT, 0, 24, 2, "Drive electronics quantization bits (0 = float)");
            add_pd(out, &n, max_out, PID_DM_NONLIN, "nonlin_quad", "Quad Nonlin", "",
                   CS_PTYPE_FLOAT, -0.5, 0.5, 0.01, "Quadratic electrostrictive nonlinearity");
        }
        return n;
    }

    if (nd->kind == CACAOSIM_NODE_SPLITTER)
    {
        if (subtab == 0)
        {
            add_enum_pd(out, &n, max_out, PID_BS_MODE, "mode", "Splitter Mode", 4, "gray",
                        "dichroic_lp", "dichroic_sp", "dichroic_bp",
                        "gray, longpass (refl<cut), shortpass (refl>cut), or band-reflect");
            /* Expose gray R/T ratios vs wavelength cutoff/bandwidth depending on splitter mode */
            if (nd->split.mode == CACAOSIM_SPLIT_GRAY)
            {
                add_pd(out, &n, max_out, PID_BS_R_GRAY, "r_gray", "Reflectance R", "",
                       CS_PTYPE_FLOAT, 0.0, 1.0, 0.05, "Achromatic reflected power fraction");
                add_pd(out, &n, max_out, PID_BS_T_GRAY, "t_gray", "Transmittance T", "",
                       CS_PTYPE_FLOAT, 0.0, 1.0, 0.05, "Achromatic transmitted power fraction");
            }
            else
            {
                add_pd(out, &n, max_out, PID_BS_LAMBDA_CUT, "lambda_cut", "Cut/Center Lam", "um",
                       CS_PTYPE_FLOAT, 0.42, 2.35, 0.05, "Dichroic cutoff or band center [um]");
                if (nd->split.mode == CACAOSIM_SPLIT_DICHROIC_BP)
                {
                    add_pd(out, &n, max_out, PID_BS_BAND_WIDTH, "band_width", "Refl Bandwidth",
                           "um", CS_PTYPE_FLOAT, 0.05, 1.20, 0.05,
                           "Reflected notch full width [um]");
                }
                add_pd(out, &n, max_out, PID_BS_ROLL_UM, "roll_um", "Edge Roll-Off", "um",
                       CS_PTYPE_FLOAT, 0.005, 0.25, 0.01, "Dichroic transition edge width [um]");
            }
        }
        else
        {
            add_pd(out, &n, max_out, PID_BS_EFF_REFL, "eff_refl", "Refl Coating Eff", "",
                   CS_PTYPE_FLOAT, 0.05, 1.0, 0.02, "Peak efficiency in reflected passband");
            add_pd(out, &n, max_out, PID_BS_EFF_TRANS, "eff_trans", "Trans Coating Eff", "",
                   CS_PTYPE_FLOAT, 0.05, 1.0, 0.02, "Peak efficiency in transmitted passband");
        }
        return n;
    }

    if (nd->kind == CACAOSIM_NODE_CORO)
    {
        if (subtab == 0)
        {
            add_enum6_pd(out, &n, max_out, PID_CO_MODE, "mode", "FPM Architecture", "lyot_spot",
                         "phase_dimple", "vortex", "4qpm", "custom_mft", "custom_fft",
                         "Analytic or custom FITS/SHM focal/pupil mask propagation");
            add_pd(out, &n, max_out, PID_CO_LAMBDA, "lambda_um", "Design Lambda", "um",
                   CS_PTYPE_FLOAT, 0.40, 2.40, 0.05, "Coronagraph reference wavelength [um]");
            add_pd(out, &n, max_out, PID_CO_BW_FRAC, "bw_frac", "Bandwidth dL/L", "",
                   CS_PTYPE_FLOAT, 0.0, 0.60, 0.02, "Polychromatic fractional bandwidth dLambda/L");
            add_pd(out, &n, max_out, PID_CO_NLAMBDA, "nlambda", "Spectral Slices", "ch",
                   CS_PTYPE_INT, 0, 32, 1,
                   "Polychromatic slices (0 = auto -> 2 if bw_frac > 0, or 3D mask)");
            add_enum_pd(out, &n, max_out, PID_CO_PHA_UNIT, "pha_unit", "Mask Phase Unit", 2, "rad",
                        "um", NULL, NULL, "Unit of custom phase masks: 0 = radians, 1 = OPD [um]");
            /* Tailor focal-mask geometry rows to active analytic/vortex/MFT/FFT propagator */
            if (nd->coro.mode <= CACAOSIM_CORO_PHASE_DIMPLE)
            {
                add_pd(out, &n, max_out, PID_CO_FPM_RAD, "fpm_rad_ld", "FPM Radius", "l/D",
                       CS_PTYPE_FLOAT, 0.5, 12.0, 0.25, "Focal mask radius [lambda/D]");
                add_pd(out, &n, max_out, PID_CO_FPM_AMP, "fpm_amp", "Core Amp Trans", "",
                       CS_PTYPE_FLOAT, 0.0, 1.0, 0.05, "Inner core field amplitude transmission");
                if (nd->coro.mode == CACAOSIM_CORO_PHASE_DIMPLE)
                {
                    add_pd(out, &n, max_out, PID_CO_FPM_PHASE, "fpm_phase", "Core Phase", "rad",
                           CS_PTYPE_FLOAT, 0.0, 6.2832, 0.157, "Phase shift inside dimple [rad]");
                }
            }
            else if (nd->coro.mode == CACAOSIM_CORO_VORTEX)
            {
                add_pd(out, &n, max_out, PID_CO_VORTEX_Q, "vortex_charge", "Vortex Charge", "l",
                       CS_PTYPE_INT, 2, 8, 2, "Topological charge l (even integer: 2, 4, 6)");
                add_pd(out, &n, max_out, PID_CO_PAD_FACTOR, "pad_factor", "FFT Pad Factor", "x",
                       CS_PTYPE_UINT, 2, 8, 1, "Zero-padded focal FFT oversampling factor");
            }
            else if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
            {
                add_pd(out, &n, max_out, PID_CO_FPM_PIXSCALE, "fpm_pixscale", "FPM Sampling",
                       "px/(l/D)", CS_PTYPE_FLOAT, 0.5, 64.0, 0.5,
                       "Custom MFT focal mask pixels per lambda_0/D");
            }
            else
            {
                add_pd(out, &n, max_out, PID_CO_PAD_FACTOR, "pad_factor", "FFT Pad Factor", "x",
                       CS_PTYPE_UINT, 2, 8, 1, "Zero-padded focal FFT oversampling factor");
            }
        }
        else
        {
            add_pd(out, &n, max_out, PID_CO_LYOT_OUTER, "lyot_outer", "Lyot Outer Stop", "D",
                   CS_PTYPE_FLOAT, 0.50, 1.0, 0.02, "Lyot stop outer diameter ratio vs pupil");
            add_pd(out, &n, max_out, PID_CO_LYOT_OBS, "lyot_obs", "Lyot Inner Obs", "D",
                   CS_PTYPE_FLOAT, 0.0, 0.80, 0.02, "Lyot stop central obstruction ratio");
            add_pd(out, &n, max_out, PID_CO_LYOT_NSP, "lyot_nspider", "Lyot Spiders", "",
                   CS_PTYPE_INT, 0, 12, 1, "Lyot spider mask vane count");
            add_pd(out, &n, max_out, PID_CO_LYOT_SPW, "lyot_spider_w", "Lyot Spider W", "D",
                   CS_PTYPE_FLOAT, 0.0, 0.15, 0.005, "Lyot spider vane width fraction");
            add_pd(out, &n, max_out, PID_CO_LYOT_SPDEG, "lyot_spider_deg", "Lyot Spid Ang", "deg",
                   CS_PTYPE_FLOAT, 0.0, 180.0, 5.0, "Lyot spider orientation angle [deg]");
            add_pd(out, &n, max_out, PID_CO_LYOT_SPOFF, "lyot_spider_offset", "Lyot Spid Off",
                   "D", CS_PTYPE_FLOAT, 0.0, 0.45, 0.005,
                   "Lyot V-spider X-vertex offset fraction of pup_diam");
            add_enum_pd(out, &n, max_out, PID_CO_LOWFS_MODE, "lowfs_mode", "LOWFS Pickoff", 2,
                        "fpm_refl", "lyot_refl", NULL, NULL,
                        "0 = FPM-absorbed light reflected, 1 = Reflective Lyot stop");
            /* Pupil-plane stop diameter on LOWFS arm applies only to FPM-reflection pickoff */
            if (nd->coro.lowfs_mode == CACAOSIM_LOWFS_FPM_REFL)
            {
                add_pd(out, &n, max_out, PID_CO_LOWFS_PUP, "lowfs_pup_diam", "LOWFS Pupil Ap",
                       "D_pup", CS_PTYPE_FLOAT, 0.05, 2.0, 0.05,
                       "Pupil-plane aperture diameter on FPM->LOWFS path [D_pup]");
            }
            add_pd(out, &n, max_out, PID_CO_LOWFS_REFL, "lowfs_refl", "LOWFS Pickoff R", "",
                   CS_PTYPE_FLOAT, 0.0, 1.0, 0.05,
                   "Amplitude reflectivity into LOWFS pickoff arm");
            add_pd(out, &n, max_out, PID_N_TPUT, "tput", "Relay Tput", "", CS_PTYPE_FLOAT, 0.05,
                   1.0, 0.02, "Coronagraph relay optical throughput");
        }
        return n;
    }

    if (nd->kind == CACAOSIM_NODE_WFE)
    {
        if (subtab == 0)
        {
            add_pd(out, &n, max_out, PID_N_IN_WFE, "in_wfe", "Static WFE SHM", "m", CS_PTYPE_STR, 0,
                   0, 0, "2D static wavefront error stream in unit of OPD [meter]");
            add_pd(out, &n, max_out, PID_WFE_RMS_NM, "rms_nm", "Init Pupil RMS", "nm",
                   CS_PTYPE_FLOAT, 0.0, 5000.0, 5.0,
                   "Target pupil-weighted RMS OPD when initializing map [nm]");
            add_pd(out, &n, max_out, PID_WFE_DEFOCUS_NM, "defocus_nm", "Z4 Defocus RMS", "nm",
                   CS_PTYPE_FLOAT, -5000.0, 5000.0, 10.0,
                   "Additive smooth Zernike Z4 defocus RMS across grid [nm]");
            add_pd(out, &n, max_out, PID_WFE_ALPHA, "psd_alpha", "PSD Slope Alpha", "",
                   CS_PTYPE_FLOAT, 0.5, 4.0, 0.1,
                   "Spatial amplitude power-law exponent k^-alpha");
            add_pd(out, &n, max_out, PID_WFE_GAIN, "wfe_gain", "OPD Scale Gain", "x",
                   CS_PTYPE_FLOAT, -10.0, 10.0, 0.05,
                   "Dimensionless multiplier on input [meter] OPD map");
            add_pd(out, &n, max_out, PID_WFE_SEED, "wfe_seed", "Random Seed", "", CS_PTYPE_UINT, 1,
                   9999999, 1, "Deterministic seed for map synthesis");
            add_pd(out, &n, max_out, PID_N_TPUT, "tput", "Throughput", "", CS_PTYPE_FLOAT, 0.05,
                   1.0, 0.01, "Optical intensity transmission factor [0..1]");
        }
        else
        {
            add_pd(out, &n, max_out, PID_N_SHIFT_X, "shift_x", "Pupil Shift X", "px",
                   CS_PTYPE_FLOAT, -32.0, 32.0, 0.25, "Lateral X misregistration [pupil pixels]");
            add_pd(out, &n, max_out, PID_N_SHIFT_Y, "shift_y", "Pupil Shift Y", "px",
                   CS_PTYPE_FLOAT, -32.0, 32.0, 0.25, "Lateral Y misregistration [pupil pixels]");
            add_pd(out, &n, max_out, PID_N_ROT_DEG, "rot_deg", "Pupil Rotation", "deg",
                   CS_PTYPE_FLOAT, -180.0, 180.0, 1.0, "In-plane pupil clocking angle [degrees]");
            add_pd(out, &n, max_out, PID_N_MAG_SCALE, "mag_scale", "Magnification", "x",
                   CS_PTYPE_FLOAT, 0.50, 1.50, 0.01, "Radial pupil magnification factor");
        }
        return n;
    }

    /* Wavefront & Focal-Plane Sensors: Subtab 0 = Diffractive Optics, Subtab 1 = Readout/Flux */
    if (subtab == 0)
    {
        if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            add_pd(out, &n, max_out, PID_S_NCAM, "ncam", "Detector Size", "px", CS_PTYPE_UINT, 16,
                   512, 16, "Square WFS camera frame size [pixels]");
            add_pd(out, &n, max_out, PID_S_LAMBDA, "lambda_um", "Center Lambda", "um",
                   CS_PTYPE_FLOAT, 0.40, 2.40, 0.05, "Sensing central wavelength [um]");
            add_pd(out, &n, max_out, PID_S_BW_FRAC, "bw_frac", "Bandwidth dL/L", "", CS_PTYPE_FLOAT,
                   0.0, 0.60, 0.02, "Fractional spectral filter width dLambda/Lambda (0 = mono)");
            add_pd(out, &n, max_out, PID_S_NLAMBDA, "nlambda", "Spectral Slices", "ch",
                   CS_PTYPE_INT, 0, 32, 1, "Polychromatic slices (0 = auto -> 2 when bw_frac > 0)");
            add_pd(out, &n, max_out, PID_PYR_MOD_RAD, "mod_rad", "Modulation Rad", "l/D",
                   CS_PTYPE_FLOAT, 0.0, 16.0, 0.5, "Tip-tilt circular modulation radius [l/D]");
            add_pd(out, &n, max_out, PID_PYR_MOD_NPTS, "mod_npts", "Modulation Pts", "pts",
                   CS_PTYPE_INT, 1, 64, 4, "Discrete points around modulation ring");
            add_pd(out, &n, max_out, PID_PYR_PUP_SEP, "pup_sep", "Pupil Separation", "ncam",
                   CS_PTYPE_FLOAT, 0.25, 0.75, 0.02, "Sub-pupil center separation fraction");
            add_pd(out, &n, max_out, PID_PYR_ROOF_LD, "roof_ld", "Roof Defect", "l/D",
                   CS_PTYPE_FLOAT, 0.0, 2.0, 0.05, "Pyramid apex roof width [lambda/D]");
        }
        else if (nd->kind == CACAOSIM_NODE_SHWFS)
        {
            add_pd(out, &n, max_out, PID_SH_NSUB, "n_sub", "Lenslets Across", "sub", CS_PTYPE_UINT,
                   2, 64, 2, "Linear lenslet count across pupil (ncam = n_sub * pix_per_sub)");
            add_pd(out, &n, max_out, PID_SH_PIX_SUB, "pix_per_sub", "Pixels / Subap", "px",
                   CS_PTYPE_UINT, 4, 32, 2, "Detector pixels per lenslet subaperture");
            add_pd(out, &n, max_out, PID_S_LAMBDA, "lambda_um", "Center Lambda", "um",
                   CS_PTYPE_FLOAT, 0.40, 2.40, 0.05, "Sensing central wavelength [um]");
            add_pd(out, &n, max_out, PID_S_BW_FRAC, "bw_frac", "Bandwidth dL/L", "", CS_PTYPE_FLOAT,
                   0.0, 0.60, 0.02, "Fractional spectral filter width dLambda/Lambda (0 = mono)");
            add_pd(out, &n, max_out, PID_S_NLAMBDA, "nlambda", "Spectral Slices", "ch",
                   CS_PTYPE_INT, 0, 32, 1, "Polychromatic slices (0 = auto -> 2 when bw_frac > 0)");
            add_pd(out, &n, max_out, PID_S_PAD_FACTOR, "pad_factor", "Subap FFT Pad", "x",
                   CS_PTYPE_UINT, 1, 4, 1, "Subaperture Nyquist FFT padding factor");
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            add_pd(out, &n, max_out, PID_S_NCAM, "ncam", "Detector Size", "px", CS_PTYPE_UINT, 16,
                   512, 16, "Square Zernike WFS camera frame size [pixels]");
            add_pd(out, &n, max_out, PID_S_LAMBDA, "lambda_um", "Center Lambda", "um",
                   CS_PTYPE_FLOAT, 0.40, 2.40, 0.05, "Sensing central wavelength [um]");
            add_pd(out, &n, max_out, PID_S_BW_FRAC, "bw_frac", "Bandwidth dL/L", "", CS_PTYPE_FLOAT,
                   0.0, 0.60, 0.02, "Fractional spectral filter width dLambda/Lambda (0 = mono)");
            add_pd(out, &n, max_out, PID_S_NLAMBDA, "nlambda", "Spectral Slices", "ch",
                   CS_PTYPE_INT, 0, 32, 1, "Polychromatic slices (0 = auto -> 2 when bw_frac > 0)");
            add_pd(out, &n, max_out, PID_ZW_DIMPLE_LD, "dimple_ld", "Dimple Diam", "l/D",
                   CS_PTYPE_FLOAT, 0.4, 4.0, 0.04, "Phase-contrast dimple diameter [lambda/D]");
            add_pd(out, &n, max_out, PID_ZW_DIMPLE_RAD, "dimple_rad", "Dimple Phase", "rad",
                   CS_PTYPE_FLOAT, 0.1, 3.1416, 0.05, "Dimple phase shift [rad] (pi/2 = 1.5708)");
            add_pd(out, &n, max_out, PID_ZW_NMFT, "n_mft", "MFT Grid Pts", "px", CS_PTYPE_UINT, 16,
                   96, 8, "Babinet Matrix-DFT sampling across dimple");
        }
        else if (nd->kind == CACAOSIM_NODE_FPCAM)
        {
            add_pd(out, &n, max_out, PID_S_NCAM, "ncam", "Detector Size", "px", CS_PTYPE_UINT, 16,
                   1024, 32, "Square science PSF detector size [pixels]");
            add_enum_pd(out, &n, max_out, PID_FP_PROP_MODE, "prop_mode", "Propagator", 2, "mft",
                        "fft", NULL, NULL, "Soummer Matrix-DFT (arbitrary pixscale) or padded FFT");
            add_pd(out, &n, max_out, PID_S_LAMBDA, "lambda_um", "Center Lambda", "um",
                   CS_PTYPE_FLOAT, 0.40, 2.40, 0.05, "Science filter central wavelength [um]");
            add_pd(out, &n, max_out, PID_S_BW_FRAC, "bw_frac", "Bandwidth dL/L", "", CS_PTYPE_FLOAT,
                   0.0, 0.60, 0.02, "Fractional filter bandwidth dLambda/Lambda (0 = mono)");
            add_pd(out, &n, max_out, PID_S_NLAMBDA, "nlambda", "Spectral Slices", "ch",
                   CS_PTYPE_INT, 0, 32, 1, "Polychromatic slices (0 = auto -> 2 when bw_frac > 0)");
            add_pd(out, &n, max_out, PID_FP_PIXSCALE, "pixscale_ld", "Sampling", "px/lD",
                   CS_PTYPE_FLOAT, 1.0, 10.0, 0.25, "Detector pixels per lambda/D (MFT mode)");
        }
    }
    else
    {
        add_pd(out, &n, max_out, PID_S_NAVG, "navg", "Subframe Avg", "ticks", CS_PTYPE_UINT, 1,
               1000, 1, "Master ticks co-added per detector readout");
        add_pd(out, &n, max_out, PID_S_RATE_HZ, "rate_hz", "Target Rate", "Hz", CS_PTYPE_FLOAT, 0.0,
               20000.0, 50.0, "Optional explicit camera rate [Hz] (0 = use navg)");
        add_pd(out, &n, max_out, PID_S_QE, "qe", "Quantum Eff", "", CS_PTYPE_FLOAT, 0.05, 1.0, 0.05,
               "Detector quantum efficiency [e-/photon]");
        add_pd(out, &n, max_out, PID_S_USE_BFLUX, "use_beam_flux", "Auto Star Flux", "",
               CS_PTYPE_BOOL, 0, 1, 1, "Derive flux from stellar spectrum & upstream optics");
        /* Manual photon budget row is exposed only when automatic stellar photometry is disabled */
        if (!nd->use_beam_flux)
        {
            add_pd(out, &n, max_out, PID_S_FLUX_PH, "flux_ph", "Manual Flux", "ph/fr",
                   CS_PTYPE_FLOAT, 0.0, 1.0e10, 1000.0, "Manual photon count per exposure");
        }
        add_pd(out, &n, max_out, PID_S_RON_E, "ron_e", "Readout Noise", "e-", CS_PTYPE_FLOAT, 0.0,
               100.0, 0.2, "Detector RMS readout noise [electrons/pixel]");
        add_pd(out, &n, max_out, PID_S_PHOT_NOISE, "photon_noise", "Photon Shot", "",
               CS_PTYPE_BOOL, 0, 1, 1, "Enable Poisson photon shot noise");
        add_enum_pd(out, &n, max_out, PID_S_OUT_DTYPE, "out_dtype", "Pixel Dtype", 2, "uint16",
                    "float32", NULL, NULL, "Output SHM pixel format (ADU uint16 or float32)");
    }

    return n;
}

/**
 * node_det_ro() - Obtain a const pointer to the embedded detector config of a sensor node
 * @nd: Sensor node (`PYRWFS`, `SHWFS`, `ZWFS`, or `FPCAM`)
 *
 * Role:
 *   Unifies read access to `flux_ph`, `ron_e`, `photon_noise`, and `out_dtype` across all four
 *   sensor structs inside `cs_param_format_val()`.
 *
 * Rationale:
 *   Each sensor compute unit embeds its own `cacaosim_det_cfg_t det` sub-struct; factoring out
 *   pointer resolution avoids duplicating four identical detector `switch` branches.
 *
 * Return: Non-NULL const pointer to the active `cacaosim_det_cfg_t` sub-struct.
 */
static const cacaosim_det_cfg_t *node_det_ro(const cacaosim_node_cfg_t *nd)
{
    if (nd->kind == CACAOSIM_NODE_PYRWFS)
    {
        return &nd->pyrwfs.det;
    }
    if (nd->kind == CACAOSIM_NODE_SHWFS)
    {
        return &nd->shwfs.det;
    }
    if (nd->kind == CACAOSIM_NODE_ZWFS)
    {
        return &nd->zwfs.det;
    }
    return &nd->fpcam.det;
}

/**
 * node_det_rw() - Obtain a writable pointer to the embedded detector config of a sensor node
 * @nd: Mutable sensor node (`PYRWFS`, `SHWFS`, `ZWFS`, or `FPCAM`)
 *
 * Role:
 *   Mutable counterpart to `node_det_ro()` used by `cs_param_set_str()` when updating detector
 *   readout noise, shot noise, manual flux, or output datatype.
 *
 * Rationale:
 *   Keeps `const`-correctness strict in `cs_param_format_val()` while sharing unified `PID_S_*`
 *   detector parameter IDs across all WFS and science camera node kinds.
 *
 * Return: Non-NULL writable pointer to the active `cacaosim_det_cfg_t` sub-struct.
 */
static cacaosim_det_cfg_t *node_det_rw(cacaosim_node_cfg_t *nd)
{
    if (nd->kind == CACAOSIM_NODE_PYRWFS)
    {
        return &nd->pyrwfs.det;
    }
    if (nd->kind == CACAOSIM_NODE_SHWFS)
    {
        return &nd->shwfs.det;
    }
    if (nd->kind == CACAOSIM_NODE_ZWFS)
    {
        return &nd->zwfs.det;
    }
    return &nd->fpcam.det;
}

/**
 * cs_param_format_val() - Format a parameter's current value in @cfg as a UI string
 * @cfg:      Optical bench configuration to inspect (ignored if NULL).
 * @sel_item: Selected topology item (`0` for `[bench]` globals, `1..n_nodes` for a node).
 * @pd:       Parameter descriptor identifying which field (`pd->id`) to read.
 * @buf:      Destination character buffer receiving the NUL-terminated formatted text.
 * @cap:      Capacity of @buf in bytes.
 *
 * Role: Read-side reflection projector converting typed fields in `cacaosim_bench_cfg_t`
 * or `cacaosim_node_cfg_t` into human-readable inspector cell strings and initial text
 * buffers for inline modal editing and incremental stepping (`cs_param_step()`).
 *
 * Rationale: Centralizing field-to-string formatting by `pd->id` ensures consistent
 * decimal precision across physical units (e.g., `%.4f` for pupil fractions, `%.1f` after
 * `1e6` scaling from `tau_sec` to microseconds, `ON`/`OFF` badges for booleans, and safe
 * fallback defaults when optional parameters such as `mag_scale` or `fpm_pixscale_ld`
 * hold zero sentinel values).
 */
void cs_param_format_val(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    const cs_param_desc_t      *pd,
    char                       *buf,
    size_t                      cap)
{
    if (cfg == NULL || pd == NULL || buf == NULL || cap == 0u)
    {
        return;
    }
    buf[0] = '\0';

    if (sel_item <= 0 || (uint32_t) (sel_item - 1) >= cfg->n_nodes)
    {
        switch (pd->id)
        {
        case PID_B_NAME:
            snprintf(buf, cap, "%s", cfg->name);
            break;
        case PID_B_NWF:
            snprintf(buf, cap, "%u", cfg->nwf);
            break;
        case PID_B_PUP_DIAM:
            snprintf(buf, cap, "%.4f", cfg->pup_diam);
            break;
        case PID_B_COBS:
            snprintf(buf, cap, "%.3f", cfg->cobs);
            break;
        case PID_B_NSPIDER:
            snprintf(buf, cap, "%d", cfg->nspider);
            break;
        case PID_B_SPIDER_W:
            snprintf(buf, cap, "%.4f", cfg->spider_w);
            break;
        case PID_B_SPIDER_DEG:
            snprintf(buf, cap, "%.2f", cfg->spider_deg);
            break;
        case PID_B_SPIDER_OFF:
            snprintf(buf, cap, "%.5f", cfg->spider_offset);
            break;
        case PID_B_TEL_DIAM:
            snprintf(buf, cap, "%.2f", cfg->tel_diam_m);
            break;
        case PID_B_STAR_MAG:
            snprintf(buf, cap, "%.2f", cfg->star_mag);
            break;
        case PID_B_STAR_TEFF:
            snprintf(buf, cap, "%.0f", cfg->star_teff);
            break;
        case PID_B_TPUT_TEL:
            snprintf(buf, cap, "%.3f", cfg->tput_tel);
            break;
        case PID_B_MASTER_HZ:
            snprintf(buf, cap, "%.1f", cfg->master_hz);
            break;
        case PID_B_WF_UNIT:
            snprintf(buf, cap, "%s", (cfg->wf_unit == CACAOSIM_WFUNIT_PHASE_RAD) ? "rad" : "um");
            break;
        case PID_B_REF_LAMBDA:
            snprintf(buf, cap, "%.3f", cfg->ref_lambda);
            break;
        case PID_B_TURB_GAIN:
            snprintf(buf, cap, "%.2f", cfg->turb_gain);
            break;
        case PID_B_BACKEND:
            snprintf(buf, cap, "%s",
                     (cfg->backend == 1) ? "cpu" : (cfg->backend == 2 ? "cuda" : "auto"));
            break;
        case PID_B_OUT_BEAM:
            snprintf(buf, cap, "%s", cfg->out_beam);
            break;
        case PID_B_IN_WF:
            snprintf(buf, cap, "%s", cfg->in_wf);
            break;
        case PID_B_IN_PUP:
            snprintf(buf, cap, "%s", cfg->in_pup);
            break;
        case PID_B_IN_AMP:
            snprintf(buf, cap, "%s", cfg->in_amp);
            break;
        case PID_B_IN_NCPA:
            snprintf(buf, cap, "%s", cfg->in_ncpa);
            break;
        default:
            break;
        }
        return;
    }

    const cacaosim_node_cfg_t *nd  = &cfg->nodes[sel_item - 1];
    const cacaosim_det_cfg_t  *det = node_det_ro(nd);

    switch (pd->id)
    {
    case PID_N_ID:
        snprintf(buf, cap, "%s", nd->id);
        break;
    case PID_N_IN_BEAM:
        snprintf(buf, cap, "%s", nd->in_beam);
        break;
    case PID_N_OUT_BEAM:
        snprintf(buf, cap, "%s", nd->out_beam);
        break;
    case PID_N_OUT_REFL:
        snprintf(buf, cap, "%s", nd->out_refl);
        break;
    case PID_N_OUT_TRANS:
        snprintf(buf, cap, "%s", nd->out_trans);
        break;
    case PID_N_OUT_LOWFS:
        snprintf(buf, cap, "%s", nd->out_lowfs);
        break;
    case PID_N_IN_CMD:
        snprintf(buf, cap, "%s", nd->in_cmd);
        break;
    case PID_N_OUT_OPD:
        snprintf(buf, cap, "%s", nd->out_opd);
        break;
    case PID_N_IN_FLAT:
        snprintf(buf, cap, "%s", nd->in_flat);
        break;
    case PID_N_OUT_CAM:
        snprintf(buf, cap, "%s", nd->out_cam);
        break;
    case PID_N_OUT_LONG:
        snprintf(buf, cap, "%s", nd->out_long);
        break;
    case PID_N_IN_NCPA:
        snprintf(buf, cap, "%s", nd->in_ncpa);
        break;
    case PID_N_IN_LYOT:
        snprintf(buf, cap, "%s", nd->in_lyot);
        break;
    case PID_N_IN_WFE:
        snprintf(buf, cap, "%s", nd->in_wfe);
        break;
    case PID_N_TPUT:
        snprintf(buf, cap, "%.3f", nd->tput);
        break;
    case PID_N_SHIFT_X:
        snprintf(buf, cap, "%.2f", nd->reg.shift_x);
        break;
    case PID_N_SHIFT_Y:
        snprintf(buf, cap, "%.2f", nd->reg.shift_y);
        break;
    case PID_N_ROT_DEG:
        snprintf(buf, cap, "%.2f", nd->reg.rot_deg);
        break;
    case PID_N_MAG_SCALE:
        snprintf(buf, cap, "%.4f", (nd->reg.scale > 0.0f) ? nd->reg.scale : 1.0f);
        break;

    case PID_DM_MODE:
        snprintf(buf, cap, "%s", (nd->dm.dm_mode == 1) ? "2-Axis TTM" : "Zonal DM");
        break;
    case PID_DM_NACT_X:
        snprintf(buf, cap, "%u", nd->dm.nact_x);
        break;
    case PID_DM_NACT_Y:
        snprintf(buf, cap, "%u", nd->dm.nact_y);
        break;
    case PID_DM_IF_COUPLING:
        snprintf(buf, cap, "%.3f", nd->dm.if_coupling);
        break;
    case PID_DM_MAX_STROKE:
        snprintf(buf, cap, "%.2f", nd->dm.max_stroke);
        break;
    case PID_DM_REFL_SCALE:
        snprintf(buf, cap, "%.2f", nd->dm.refl_scale);
        break;
    case PID_DM_DYN_MODE:
        snprintf(buf, cap, "%s",
                 (nd->dm.dyn_mode == 0) ? "instant" : (nd->dm.dyn_mode == 2 ? "osc" : "rc"));
        break;
    case PID_DM_TAU_US:
        snprintf(buf, cap, "%.1f", nd->dm.tau_sec * 1.0e6f);
        break;
    case PID_DM_F0_HZ:
        snprintf(buf, cap, "%.0f", nd->dm.f0_hz);
        break;
    case PID_DM_ZETA:
        snprintf(buf, cap, "%.2f", nd->dm.zeta);
        break;
    case PID_DM_DELAY:
        snprintf(buf, cap, "%.2f", nd->dm.delay_frames);
        break;
    case PID_DM_DAC_BITS:
        snprintf(buf, cap, "%d", nd->dm.dac_bits);
        break;
    case PID_DM_NONLIN:
        snprintf(buf, cap, "%.3f", nd->dm.nonlin_quad);
        break;

    case PID_BS_MODE:
        snprintf(buf, cap, "%s",
                 pd->enum_names[(nd->split.mode >= 0 && nd->split.mode < 4) ? nd->split.mode : 1]);
        break;
    case PID_BS_LAMBDA_CUT:
        snprintf(buf, cap, "%.3f", nd->split.lambda_cut);
        break;
    case PID_BS_BAND_WIDTH:
        snprintf(buf, cap, "%.3f", nd->split.band_width);
        break;
    case PID_BS_ROLL_UM:
        snprintf(buf, cap, "%.3f", nd->split.roll_um);
        break;
    case PID_BS_R_GRAY:
        snprintf(buf, cap, "%.3f", nd->split.r_gray);
        break;
    case PID_BS_T_GRAY:
        snprintf(buf, cap, "%.3f", nd->split.t_gray);
        break;
    case PID_BS_EFF_REFL:
        snprintf(buf, cap, "%.3f", nd->split.eff_refl);
        break;
    case PID_BS_EFF_TRANS:
        snprintf(buf, cap, "%.3f", nd->split.eff_trans);
        break;
    case PID_BS_NCPA_REFL:
        snprintf(buf, cap, "%s", nd->ncpa_refl);
        break;
    case PID_BS_NCPA_TRANS:
        snprintf(buf, cap, "%s", nd->ncpa_trans);
        break;

    case PID_CO_MODE:
        snprintf(buf, cap, "%s",
                 pd->enum_names[(nd->coro.mode >= 0 && nd->coro.mode < 6) ? nd->coro.mode : 0]);
        break;
    case PID_CO_LAMBDA:
        snprintf(buf, cap, "%.3f", nd->coro.lambda_um);
        break;
    case PID_CO_BW_FRAC:
        snprintf(buf, cap, "%.3f", nd->coro.bw_frac);
        break;
    case PID_CO_NLAMBDA:
        snprintf(buf, cap, "%d", (nd->coro.nlambda >= 0) ? nd->coro.nlambda : 0);
        break;
    case PID_CO_PHA_UNIT:
        snprintf(buf, cap, "%s", (nd->coro.mask_pha_unit == 1) ? "um" : "rad");
        break;
    case PID_CO_FPM_RAD:
        snprintf(buf, cap, "%.2f", nd->coro.fpm_rad_ld);
        break;
    case PID_CO_FPM_AMP:
        snprintf(buf, cap, "%.3f", nd->coro.fpm_amp);
        break;
    case PID_CO_FPM_PHASE:
        snprintf(buf, cap, "%.3f", nd->coro.fpm_phase_rad);
        break;
    case PID_CO_FPM_PIXSCALE:
        snprintf(buf, cap, "%.2f",
                 (nd->coro.fpm_pixscale_ld > 0.0f) ? nd->coro.fpm_pixscale_ld : 8.0f);
        break;
    case PID_CO_PAD_FACTOR:
        snprintf(buf, cap, "%u", (nd->coro.pad_factor >= 2u) ? nd->coro.pad_factor : 4u);
        break;
    case PID_CO_VORTEX_Q:
        snprintf(buf, cap, "%d", nd->coro.vortex_charge);
        break;
    case PID_CO_LOWFS_MODE:
        snprintf(buf, cap, "%s",
                 (nd->coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? "lyot_refl" : "fpm_refl");
        break;
    case PID_CO_LOWFS_PUP:
        snprintf(buf, cap, "%.3f",
                 (nd->coro.lowfs_pup_diam > 0.0f) ? nd->coro.lowfs_pup_diam : 1.0f);
        break;
    case PID_CO_LOWFS_REFL:
        snprintf(buf, cap, "%.3f", nd->coro.lowfs_refl);
        break;
    case PID_CO_LYOT_OUTER:
        snprintf(buf, cap, "%.3f", nd->coro.lyot_outer);
        break;
    case PID_CO_LYOT_OBS:
        snprintf(buf, cap, "%.3f", nd->coro.lyot_obs);
        break;
    case PID_CO_LYOT_NSP:
        snprintf(buf, cap, "%d", nd->coro.lyot_nspider);
        break;
    case PID_CO_LYOT_SPW:
        snprintf(buf, cap, "%.4f", nd->coro.lyot_spider_w);
        break;
    case PID_CO_LYOT_SPDEG:
        snprintf(buf, cap, "%.2f", nd->coro.lyot_spider_deg);
        break;
    case PID_CO_LYOT_SPOFF:
        snprintf(buf, cap, "%.5f", nd->coro.lyot_spider_offset);
        break;
    case PID_CO_IN_PUP_AMP:
        snprintf(buf, cap, "%s", nd->in_pup_amp);
        break;
    case PID_CO_IN_PUP_PHA:
        snprintf(buf, cap, "%s", nd->in_pup_pha);
        break;
    case PID_CO_IN_FPM_AMP:
        snprintf(buf, cap, "%s", nd->in_fpm_amp);
        break;
    case PID_CO_IN_FPM_PHA:
        snprintf(buf, cap, "%s", nd->in_fpm_pha);
        break;
    case PID_CO_IN_LYOT_PHA:
        snprintf(buf, cap, "%s", nd->in_lyot_pha);
        break;

    case PID_S_NCAM:
        snprintf(buf, cap, "%u",
                 (nd->kind == CACAOSIM_NODE_PYRWFS)
                     ? nd->pyrwfs.ncam
                     : (nd->kind == CACAOSIM_NODE_ZWFS ? nd->zwfs.ncam : nd->fpcam.ncam));
        break;
    case PID_S_LAMBDA:
        snprintf(buf, cap, "%.3f",
                 (nd->kind == CACAOSIM_NODE_PYRWFS)
                     ? nd->pyrwfs.lambda_um
                     : (nd->kind == CACAOSIM_NODE_SHWFS
                            ? nd->shwfs.lambda_um
                            : (nd->kind == CACAOSIM_NODE_ZWFS ? nd->zwfs.lambda_um
                                                              : nd->fpcam.lambda_um)));
        break;
    case PID_S_BW_FRAC:
        snprintf(buf, cap, "%.3f", nd->bw_frac);
        break;
    case PID_S_NLAMBDA:
    case PID_FP_NLAMBDA:
        snprintf(buf, cap, "%d",
                 (nd->kind == CACAOSIM_NODE_PYRWFS)
                     ? nd->pyrwfs.nlambda
                     : (nd->kind == CACAOSIM_NODE_SHWFS
                            ? nd->shwfs.nlambda
                            : (nd->kind == CACAOSIM_NODE_ZWFS ? nd->zwfs.nlambda
                                                              : nd->fpcam.nlambda)));
        break;
    case PID_S_NAVG:
        snprintf(buf, cap, "%u", nd->navg);
        break;
    case PID_S_RATE_HZ:
        snprintf(buf, cap, "%.1f", nd->rate_hz);
        break;
    case PID_S_QE:
        snprintf(buf, cap, "%.3f", nd->qe);
        break;
    case PID_S_USE_BFLUX:
        snprintf(buf, cap, "%s", nd->use_beam_flux ? "ON" : "OFF");
        break;
    case PID_S_FLUX_PH:
        snprintf(buf, cap, "%.4g", det->flux_ph);
        break;
    case PID_S_RON_E:
        snprintf(buf, cap, "%.2f", det->ron_e);
        break;
    case PID_S_PHOT_NOISE:
        snprintf(buf, cap, "%s", det->photon_noise ? "ON" : "OFF");
        break;
    case PID_S_OUT_DTYPE:
        snprintf(buf, cap, "%s", (det->out_dtype == CACAOSIM_DTYPE_F32) ? "float32" : "uint16");
        break;
    case PID_S_PAD_FACTOR:
        snprintf(buf, cap, "%u", nd->shwfs.pad_factor);
        break;
    case PID_PYR_MOD_RAD:
        snprintf(buf, cap, "%.2f", nd->pyrwfs.mod_rad_ld);
        break;
    case PID_PYR_MOD_NPTS:
        snprintf(buf, cap, "%d", nd->pyrwfs.mod_npts);
        break;
    case PID_PYR_PUP_SEP:
        snprintf(buf, cap, "%.3f", nd->pyrwfs.pup_sep);
        break;
    case PID_PYR_ROOF_LD:
        snprintf(buf, cap, "%.3f", nd->pyrwfs.roof_ld);
        break;
    case PID_SH_NSUB:
        snprintf(buf, cap, "%u", nd->shwfs.n_sub);
        break;
    case PID_SH_PIX_SUB:
        snprintf(buf, cap, "%u", nd->shwfs.pix_per_sub);
        break;
    case PID_ZW_DIMPLE_LD:
        snprintf(buf, cap, "%.3f", nd->zwfs.dimple_ld);
        break;
    case PID_ZW_DIMPLE_RAD:
        snprintf(buf, cap, "%.4f", nd->zwfs.dimple_rad);
        break;
    case PID_ZW_NMFT:
        snprintf(buf, cap, "%u", nd->zwfs.n_mft);
        break;
    case PID_FP_PROP_MODE:
        snprintf(buf, cap, "%s", (nd->fpcam.prop_mode == CACAOSIM_FPCAM_PROP_FFT) ? "fft" : "mft");
        break;
    case PID_FP_PIXSCALE:
        snprintf(buf, cap, "%.2f", nd->fpcam.pixscale_ld);
        break;
    case PID_WFE_RMS_NM:
        snprintf(buf, cap, "%.1f", nd->wfe_rms_nm);
        break;
    case PID_WFE_DEFOCUS_NM:
        snprintf(buf, cap, "%.1f", nd->wfe_defocus_nm);
        break;
    case PID_WFE_ALPHA:
        snprintf(buf, cap, "%.2f", nd->wfe_alpha);
        break;
    case PID_WFE_GAIN:
        snprintf(buf, cap, "%.3f", nd->wfe_gain);
        break;
    case PID_WFE_SEED:
        snprintf(buf, cap, "%llu", (unsigned long long) nd->wfe_seed);
        break;
    default:
        break;
    }
}

/**
 * clamp_d() - Clamp a double-precision scalar to the closed interval [@lo, @hi]
 * @v:  Input scalar value.
 * @lo: Inclusive lower bound.
 * @hi: Inclusive upper bound.
 *
 * Role: Numeric sanitizer applied to every user-entered or incrementally stepped scalar
 * before casting into target configuration struct fields.
 *
 * Rationale: Clamping in `double` precision prior to integer rounding (`llround()`) or
 * `float` narrowing avoids signed integer overflow and prevents out-of-domain physical
 * values (such as negative aperture sizes, zero FFT grid dimensions, or unphysical filter
 * roll-offs) from reaching downstream simulation kernels.
 *
 * Return: @v clamped to `[@lo, @hi]`.
 */
static double clamp_d(double v, double lo, double hi)
{
    if (v < lo)
    {
        return lo;
    }
    if (v > hi)
    {
        return hi;
    }
    return v;
}

/**
 * cs_param_set_str() - Parse, validate, clamp, and apply a string edit to @cfg
 * @cfg:      Optical bench configuration to mutate in place (must be non-NULL).
 * @sel_item: Selected topology item (`0` for `[bench]` globals, `1..n_nodes` for a node).
 * @pd:       Descriptor supplying type, `[min_v, max_v]` bounds, and enum vocabulary.
 * @val_str:  NUL-terminated user input string to parse and commit.
 * @err_buf:  Optional buffer receiving a human-readable diagnostic on parse/rename error.
 * @err_cap:  Capacity of @err_buf in bytes (`0` if @err_buf is NULL).
 *
 * Role: Write-side reflection mutator committing inline text edits, enum selections,
 * boolean toggles, and numeric step updates into `cacaosim_bench_cfg_t` while maintaining
 * cross-field and topological invariants.
 *
 * Rationale: Funneling all parameter mutations through a single string-accepting entry
 * point guarantees identical validation whether a change originates from keyboard text
 * entry, left/right arrow stepping, or mouse wheel scrolling. Beyond basic range clamping,
 * this mutator enforces structural bench invariants: unique node renaming via
 * `cacaosim_bench_rename_node()`, cascading downstream `in_beam` rewires when an output
 * stream name changes (`cacaosim_bench_rewire_beam_out()`), even-integer parity for FFT
 * grids and vortex charges, DM/TTM actuator geometry resets, and dual-field mirroring for
 * 2D registration and spectral bandwidth fields.
 *
 * Return: `0` on success, or `-1` if validation fails (leaving @cfg unmodified).
 */
int cs_param_set_str(
    cacaosim_bench_cfg_t  *cfg,
    int                    sel_item,
    const cs_param_desc_t *pd,
    const char            *val_str,
    char                  *err_buf,
    size_t                 err_cap)
{
    if (cfg == NULL || pd == NULL || val_str == NULL)
    {
        return -1;
    }

    /*
     * Node ID edits require collision checking against existing nodes and must preserve
     * valid INI section identifier characters; delegate to the bench topology helper.
     */
    if (pd->type == CS_PTYPE_ID)
    {
        if (sel_item <= 0 || (uint32_t) (sel_item - 1) >= cfg->n_nodes)
        {
            return -1;
        }
        return cacaosim_bench_rename_node(cfg, (uint32_t) (sel_item - 1), val_str, err_buf,
                                          err_cap);
    }

    /*
     * Parse numeric, boolean, and enumerated inputs upfront before touching @cfg so
     * malformed strings fail atomically without partially mutating bench state.
     */
    double dv = 0.0;
    int    iv = 0;
    if (pd->type == CS_PTYPE_FLOAT || pd->type == CS_PTYPE_INT || pd->type == CS_PTYPE_UINT)
    {
        char *endp = NULL;
        dv         = strtod(val_str, &endp);
        if (endp == val_str)
        {
            if (err_buf != NULL && err_cap > 0u)
            {
                snprintf(err_buf, err_cap, "Expected numeric value for '%s'", pd->key);
            }
            return -1;
        }
        dv = clamp_d(dv, pd->min_v, pd->max_v);
        iv = (int) llround(dv);
    }
    else if (pd->type == CS_PTYPE_BOOL)
    {
        /* Accept standard INI/CLI truthy synonyms case-insensitively */
        iv = (strcasecmp(val_str, "1") == 0 || strcasecmp(val_str, "on") == 0 ||
              strcasecmp(val_str, "true") == 0 || strcasecmp(val_str, "yes") == 0)
                 ? 1
                 : 0;
    }
    else if (pd->type == CS_PTYPE_ENUM)
    {
        iv = -1;
        for (int e = 0; e < pd->n_enum; e++)
        {
            if (pd->enum_names[e] != NULL && strcasecmp(val_str, pd->enum_names[e]) == 0)
            {
                iv = e;
                break;
            }
        }
        /* Allow numeric enum ordinal fallback (0..n_enum-1) for power-user quick entry */
        if (iv < 0)
        {
            char *endp = NULL;
            long  lv   = strtol(val_str, &endp, 10);
            if (endp != val_str && lv >= 0 && lv < pd->n_enum)
            {
                iv = (int) lv;
            }
            else
            {
                if (err_buf != NULL && err_cap > 0u)
                {
                    snprintf(err_buf, err_cap, "Unknown option '%s' for '%s'", val_str, pd->key);
                }
                return -1;
            }
        }
    }

    if (sel_item <= 0 || (uint32_t) (sel_item - 1) >= cfg->n_nodes)
    {
        switch (pd->id)
        {
        case PID_B_NAME:
            snprintf(cfg->name, sizeof(cfg->name), "%s", val_str);
            break;
        case PID_B_NWF:
            /* Round up to an even pixel count so pupil/FFT quadrants split symmetrically */
            cfg->nwf = (uint32_t) ((iv + 1) & ~1);
            break;
        case PID_B_PUP_DIAM:
            cfg->pup_diam = (float) dv;
            break;
        case PID_B_COBS:
            cfg->cobs = (float) dv;
            break;
        case PID_B_NSPIDER:
            cfg->nspider = iv;
            break;
        case PID_B_SPIDER_W:
            cfg->spider_w = (float) dv;
            break;
        case PID_B_SPIDER_DEG:
            cfg->spider_deg = (float) dv;
            break;
        case PID_B_SPIDER_OFF:
            cfg->spider_offset = (float) dv;
            break;
        case PID_B_TEL_DIAM:
            cfg->tel_diam_m = (float) dv;
            break;
        case PID_B_STAR_MAG:
            cfg->star_mag = (float) dv;
            break;
        case PID_B_STAR_TEFF:
            cfg->star_teff = (float) dv;
            break;
        case PID_B_TPUT_TEL:
            cfg->tput_tel = (float) dv;
            break;
        case PID_B_MASTER_HZ:
            cfg->master_hz = (float) dv;
            break;
        case PID_B_WF_UNIT:
            cfg->wf_unit = (iv == 1) ? CACAOSIM_WFUNIT_PHASE_RAD : CACAOSIM_WFUNIT_OPD_UM;
            break;
        case PID_B_REF_LAMBDA:
            cfg->ref_lambda = (float) dv;
            break;
        case PID_B_TURB_GAIN:
            cfg->turb_gain = (float) dv;
            break;
        case PID_B_BACKEND:
            cfg->backend = iv;
            break;
        case PID_B_OUT_BEAM:
            if (val_str[0] != '\0')
            {
                /* Rewire downstream consumers attached to the old source beam stream */
                cacaosim_bench_rewire_beam_out(cfg, cfg->out_beam, val_str);
                snprintf(cfg->out_beam, sizeof(cfg->out_beam), "%s", val_str);
            }
            break;
        case PID_B_IN_WF:
            snprintf(cfg->in_wf, sizeof(cfg->in_wf), "%s", val_str);
            break;
        case PID_B_IN_PUP:
            snprintf(cfg->in_pup, sizeof(cfg->in_pup), "%s", val_str);
            break;
        case PID_B_IN_AMP:
            snprintf(cfg->in_amp, sizeof(cfg->in_amp), "%s", val_str);
            break;
        case PID_B_IN_NCPA:
            snprintf(cfg->in_ncpa, sizeof(cfg->in_ncpa), "%s", val_str);
            break;
        default:
            break;
        }
        return 0;
    }

    cacaosim_node_cfg_t *nd  = &cfg->nodes[sel_item - 1];
    cacaosim_det_cfg_t  *det = node_det_rw(nd);

    switch (pd->id)
    {
    case PID_N_IN_BEAM:
        if (val_str[0] != '\0')
        {
            snprintf(nd->in_beam, sizeof(nd->in_beam), "%s", val_str);
        }
        break;
    /*
     * Renaming any outgoing optical beam stream cascades automatically to downstream
     * nodes currently wired to that stream so inline optical chains remain intact.
     */
    case PID_N_OUT_BEAM:
        if (val_str[0] != '\0')
        {
            cacaosim_bench_rewire_beam_out(cfg, nd->out_beam, val_str);
            snprintf(nd->out_beam, sizeof(nd->out_beam), "%s", val_str);
        }
        break;
    case PID_N_OUT_REFL:
        if (val_str[0] != '\0')
        {
            cacaosim_bench_rewire_beam_out(cfg, nd->out_refl, val_str);
            snprintf(nd->out_refl, sizeof(nd->out_refl), "%s", val_str);
        }
        break;
    case PID_N_OUT_TRANS:
        if (val_str[0] != '\0')
        {
            cacaosim_bench_rewire_beam_out(cfg, nd->out_trans, val_str);
            snprintf(nd->out_trans, sizeof(nd->out_trans), "%s", val_str);
        }
        break;
    case PID_N_OUT_LOWFS:
        if (val_str[0] != '\0')
        {
            cacaosim_bench_rewire_beam_out(cfg, nd->out_lowfs, val_str);
        }
        snprintf(nd->out_lowfs, sizeof(nd->out_lowfs), "%s", val_str);
        break;
    case PID_N_IN_CMD:
        snprintf(nd->in_cmd, sizeof(nd->in_cmd), "%s", val_str);
        break;
    case PID_N_OUT_OPD:
        snprintf(nd->out_opd, sizeof(nd->out_opd), "%s", val_str);
        break;
    case PID_N_IN_FLAT:
        snprintf(nd->in_flat, sizeof(nd->in_flat), "%s", val_str);
        break;
    case PID_N_OUT_CAM:
        snprintf(nd->out_cam, sizeof(nd->out_cam), "%s", val_str);
        break;
    case PID_N_OUT_LONG:
        snprintf(nd->out_long, sizeof(nd->out_long), "%s", val_str);
        break;
    case PID_N_IN_NCPA:
        snprintf(nd->in_ncpa, sizeof(nd->in_ncpa), "%s", val_str);
        break;
    case PID_N_IN_LYOT:
        snprintf(nd->in_lyot, sizeof(nd->in_lyot), "%s", val_str);
        break;
    case PID_N_IN_WFE:
        snprintf(nd->in_wfe, sizeof(nd->in_wfe), "%s", val_str);
        break;
    case PID_N_TPUT:
        nd->tput = (float) dv;
        break;
    /*
     * Keep generic 2D pupil registration (`nd->reg.*`) synchronized with the DM-specific
     * registration fields (`nd->dm.*`) used by standalone `cacaosim.dmsim` instances.
     */
    case PID_N_SHIFT_X:
        nd->reg.shift_x = (float) dv;
        nd->dm.shift_x  = nd->reg.shift_x;
        break;
    case PID_N_SHIFT_Y:
        nd->reg.shift_y = (float) dv;
        nd->dm.shift_y  = nd->reg.shift_y;
        break;
    case PID_N_ROT_DEG:
        nd->reg.rot_deg = (float) dv;
        nd->dm.rot_deg  = nd->reg.rot_deg;
        break;
    case PID_N_MAG_SCALE:
        nd->reg.scale    = (float) dv;
        nd->dm.mag_scale = nd->reg.scale;
        break;

    case PID_DM_MODE:
        nd->dm.dm_mode = iv;
        /* Switch between 2x1 tip-tilt channels and a default 16x16 zonal grid cleanly */
        if (iv == 1)
        {
            nd->dm.nact_x = 2u;
            nd->dm.nact_y = 1u;
        }
        else if (nd->dm.nact_x <= 2u && nd->dm.nact_y <= 1u)
        {
            nd->dm.nact_x = 16u;
            nd->dm.nact_y = 16u;
        }
        break;
    case PID_DM_NACT_X:
        nd->dm.nact_x = (uint32_t) iv;
        break;
    case PID_DM_NACT_Y:
        nd->dm.nact_y = (uint32_t) iv;
        break;
    case PID_DM_IF_COUPLING:
        nd->dm.if_coupling = (float) dv;
        break;
    case PID_DM_MAX_STROKE:
        nd->dm.max_stroke = (float) dv;
        break;
    case PID_DM_REFL_SCALE:
        nd->dm.refl_scale = (float) dv;
        break;
    case PID_DM_DYN_MODE:
        nd->dm.dyn_mode = iv;
        break;
    case PID_DM_TAU_US:
        /* Inspector displays microseconds for ergonomics; runtime kernel stores seconds */
        nd->dm.tau_sec = (float) dv * 1.0e-6f;
        break;
    case PID_DM_F0_HZ:
        nd->dm.f0_hz = (float) dv;
        break;
    case PID_DM_ZETA:
        nd->dm.zeta = (float) dv;
        break;
    case PID_DM_DELAY:
        nd->dm.delay_frames = (float) dv;
        break;
    case PID_DM_DAC_BITS:
        nd->dm.dac_bits = iv;
        break;
    case PID_DM_NONLIN:
        nd->dm.nonlin_quad = (float) dv;
        break;

    case PID_BS_MODE:
        nd->split.mode = iv;
        break;
    case PID_BS_LAMBDA_CUT:
        nd->split.lambda_cut = (float) dv;
        break;
    case PID_BS_BAND_WIDTH:
        nd->split.band_width = (float) dv;
        break;
    case PID_BS_ROLL_UM:
        nd->split.roll_um = (float) dv;
        break;
    case PID_BS_R_GRAY:
        nd->split.r_gray = (float) dv;
        break;
    case PID_BS_T_GRAY:
        nd->split.t_gray = (float) dv;
        break;
    case PID_BS_EFF_REFL:
        nd->split.eff_refl = (float) dv;
        break;
    case PID_BS_EFF_TRANS:
        nd->split.eff_trans = (float) dv;
        break;
    case PID_BS_NCPA_REFL:
        snprintf(nd->ncpa_refl, sizeof(nd->ncpa_refl), "%s", val_str);
        break;
    case PID_BS_NCPA_TRANS:
        snprintf(nd->ncpa_trans, sizeof(nd->ncpa_trans), "%s", val_str);
        break;

    case PID_CO_MODE:
        nd->coro.mode = iv;
        break;
    case PID_CO_LAMBDA:
        nd->coro.lambda_um = (float) dv;
        break;
    case PID_CO_BW_FRAC:
        /* Mirror coronagraph fractional bandwidth into the generic node photometry field */
        nd->coro.bw_frac = (float) dv;
        nd->bw_frac      = nd->coro.bw_frac;
        break;
    case PID_CO_NLAMBDA:
        nd->coro.nlambda = iv;
        break;
    case PID_CO_PHA_UNIT:
        nd->coro.mask_pha_unit = iv;
        break;
    case PID_CO_FPM_RAD:
        nd->coro.fpm_rad_ld = (float) dv;
        break;
    case PID_CO_FPM_AMP:
        nd->coro.fpm_amp = (float) dv;
        break;
    case PID_CO_FPM_PHASE:
        nd->coro.fpm_phase_rad = (float) dv;
        break;
    case PID_CO_FPM_PIXSCALE:
        nd->coro.fpm_pixscale_ld = (float) dv;
        break;
    case PID_CO_PAD_FACTOR:
        nd->coro.pad_factor = (uint32_t) iv;
        break;
    case PID_CO_VORTEX_Q:
        /* Enforce even topological charge (`2, 4, 6, ...`) for on-axis vortex nulling */
        nd->coro.vortex_charge = (iv + 1) & ~1;
        break;
    case PID_CO_LOWFS_MODE:
        nd->coro.lowfs_mode =
            (iv == 1) ? CACAOSIM_LOWFS_LYOT_REFL : CACAOSIM_LOWFS_FPM_REFL;
        break;
    case PID_CO_LOWFS_PUP:
        nd->coro.lowfs_pup_diam = (float) dv;
        break;
    case PID_CO_LOWFS_REFL:
        nd->coro.lowfs_refl = (float) dv;
        break;
    case PID_CO_LYOT_OUTER:
        nd->coro.lyot_outer = (float) dv;
        break;
    case PID_CO_LYOT_OBS:
        nd->coro.lyot_obs = (float) dv;
        break;
    case PID_CO_LYOT_NSP:
        nd->coro.lyot_nspider = iv;
        break;
    case PID_CO_LYOT_SPW:
        nd->coro.lyot_spider_w = (float) dv;
        break;
    case PID_CO_LYOT_SPDEG:
        nd->coro.lyot_spider_deg = (float) dv;
        break;
    case PID_CO_LYOT_SPOFF:
        nd->coro.lyot_spider_offset = (float) dv;
        break;
    case PID_CO_IN_PUP_AMP:
        snprintf(nd->in_pup_amp, sizeof(nd->in_pup_amp), "%s", val_str);
        break;
    case PID_CO_IN_PUP_PHA:
        snprintf(nd->in_pup_pha, sizeof(nd->in_pup_pha), "%s", val_str);
        break;
    case PID_CO_IN_FPM_AMP:
        snprintf(nd->in_fpm_amp, sizeof(nd->in_fpm_amp), "%s", val_str);
        break;
    case PID_CO_IN_FPM_PHA:
        snprintf(nd->in_fpm_pha, sizeof(nd->in_fpm_pha), "%s", val_str);
        break;
    case PID_CO_IN_LYOT_PHA:
        snprintf(nd->in_lyot_pha, sizeof(nd->in_lyot_pha), "%s", val_str);
        break;

    case PID_S_NCAM:
        if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            nd->pyrwfs.ncam = (uint32_t) iv;
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            nd->zwfs.ncam = (uint32_t) iv;
        }
        else
        {
            nd->fpcam.ncam = (uint32_t) iv;
        }
        break;
    case PID_S_LAMBDA:
        if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            nd->pyrwfs.lambda_um = (float) dv;
        }
        else if (nd->kind == CACAOSIM_NODE_SHWFS)
        {
            nd->shwfs.lambda_um = (float) dv;
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            nd->zwfs.lambda_um = (float) dv;
        }
        else
        {
            nd->fpcam.lambda_um = (float) dv;
        }
        break;
    case PID_S_BW_FRAC:
        /* Mirror sensor photometric bandpass into per-sensor polychromatic optical configs */
        nd->bw_frac        = (float) dv;
        nd->pyrwfs.bw_frac = nd->bw_frac;
        nd->shwfs.bw_frac  = nd->bw_frac;
        nd->zwfs.bw_frac   = nd->bw_frac;
        nd->fpcam.bw_frac  = nd->bw_frac;
        break;
    case PID_S_NLAMBDA:
    case PID_FP_NLAMBDA:
        if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            nd->pyrwfs.nlambda = iv;
        }
        else if (nd->kind == CACAOSIM_NODE_SHWFS)
        {
            nd->shwfs.nlambda = iv;
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            nd->zwfs.nlambda = iv;
        }
        else
        {
            nd->fpcam.nlambda = iv;
        }
        break;
    case PID_S_NAVG:
        nd->navg = (uint32_t) iv;
        break;
    case PID_S_RATE_HZ:
        nd->rate_hz = (float) dv;
        break;
    case PID_S_QE:
        nd->qe = (float) dv;
        break;
    case PID_S_USE_BFLUX:
        nd->use_beam_flux = iv;
        break;
    case PID_S_FLUX_PH:
        det->flux_ph = (float) dv;
        break;
    case PID_S_RON_E:
        det->ron_e = (float) dv;
        break;
    case PID_S_PHOT_NOISE:
        det->photon_noise = iv;
        break;
    case PID_S_OUT_DTYPE:
        det->out_dtype = (iv == 1) ? CACAOSIM_DTYPE_F32 : CACAOSIM_DTYPE_U16;
        break;
    case PID_S_PAD_FACTOR:
        nd->shwfs.pad_factor = (uint32_t) iv;
        break;
    case PID_PYR_MOD_RAD:
        nd->pyrwfs.mod_rad_ld = (float) dv;
        break;
    case PID_PYR_MOD_NPTS:
        nd->pyrwfs.mod_npts = iv;
        break;
    case PID_PYR_PUP_SEP:
        nd->pyrwfs.pup_sep = (float) dv;
        break;
    case PID_PYR_ROOF_LD:
        nd->pyrwfs.roof_ld = (float) dv;
        break;
    case PID_SH_NSUB:
        nd->shwfs.n_sub = (uint32_t) iv;
        break;
    case PID_SH_PIX_SUB:
        nd->shwfs.pix_per_sub = (uint32_t) iv;
        break;
    case PID_ZW_DIMPLE_LD:
        nd->zwfs.dimple_ld = (float) dv;
        break;
    case PID_ZW_DIMPLE_RAD:
        nd->zwfs.dimple_rad = (float) dv;
        break;
    case PID_ZW_NMFT:
        nd->zwfs.n_mft = (uint32_t) iv;
        break;
    case PID_FP_PROP_MODE:
        nd->fpcam.prop_mode = iv;
        break;
    case PID_FP_PIXSCALE:
        nd->fpcam.pixscale_ld = (float) dv;
        break;
    case PID_WFE_RMS_NM:
        nd->wfe_rms_nm = (float) dv;
        break;
    case PID_WFE_DEFOCUS_NM:
        nd->wfe_defocus_nm = (float) dv;
        break;
    case PID_WFE_ALPHA:
        nd->wfe_alpha = (float) dv;
        break;
    case PID_WFE_GAIN:
        nd->wfe_gain = (float) dv;
        break;
    case PID_WFE_SEED:
        nd->wfe_seed = (uint64_t) llround(dv);
        break;
    default:
        break;
    }
    return 0;
}

/**
 * cs_param_step() - Increment, decrement, toggle, or cycle a parameter in @cfg
 * @cfg:      Optical bench configuration to mutate in place (must be non-NULL).
 * @sel_item: Selected topology item (`0` for `[bench]` globals, `1..n_nodes` for a node).
 * @pd:       Parameter descriptor specifying type, `step_v`, `[min_v, max_v]`, and enums.
 * @dir:      Step direction (`+1` to increment/next, `-1` to decrement/previous).
 * @coarse:   Non-zero to multiply numeric `step_v` by `5.0` (e.g., Shift+Arrow / `[` `]`).
 * @err_buf:  Optional diagnostic buffer forwarded to `cs_param_set_str()`.
 * @err_cap:  Capacity of @err_buf in bytes.
 *
 * Role: Interactive jog controller driving Left/Right arrow adjustments, `[`/`]` coarse
 * steps, Spacebar toggles, and mouse-wheel tweaks in the inspector pane without opening a
 * modal text-entry dialog.
 *
 * Rationale: Reading the formatted current value via `cs_param_format_val()` and committing
 * the stepped value back through `cs_param_set_str()` avoids duplicating per-field read/write
 * switch tables while automatically inheriting every range clamp, even-parity rounding rule
 * (`nwf`, `vortex_charge`), DM/TTM mode geometry reset, and dual-field mirror invariant.
 * String (`CS_PTYPE_STR`) and node ID (`CS_PTYPE_ID`) rows intentionally return `-1`
 * because arbitrary identifiers cannot be meaningfully stepped.
 *
 * Return: `0` if @cfg was updated, or `-1` if @pd is non-steppable or arguments are invalid.
 */
int cs_param_step(
    cacaosim_bench_cfg_t  *cfg,
    int                    sel_item,
    const cs_param_desc_t *pd,
    int                    dir,
    int                    coarse,
    char                  *err_buf,
    size_t                 err_cap)
{
    if (cfg == NULL || pd == NULL || dir == 0)
    {
        return -1;
    }
    char cur[128];
    cs_param_format_val(cfg, sel_item, pd, cur, sizeof(cur));

    /* Booleans invert regardless of step direction */
    if (pd->type == CS_PTYPE_BOOL)
    {
        int on = (strcasecmp(cur, "ON") == 0) ? 0 : 1;
        return cs_param_set_str(cfg, sel_item, pd, on ? "ON" : "OFF", err_buf, err_cap);
    }
    /* Enumerations wrap cyclically modulo `n_enum` in either direction */
    if (pd->type == CS_PTYPE_ENUM && pd->n_enum > 0)
    {
        int idx = 0;
        for (int e = 0; e < pd->n_enum; e++)
        {
            if (pd->enum_names[e] != NULL && strcasecmp(cur, pd->enum_names[e]) == 0)
            {
                idx = e;
                break;
            }
        }
        idx = (idx + (dir > 0 ? 1 : pd->n_enum - 1)) % pd->n_enum;
        return cs_param_set_str(cfg, sel_item, pd, pd->enum_names[idx], err_buf, err_cap);
    }
    /* Numeric types apply `dir * step_v * (coarse ? 5 : 1)` and route through clamping */
    if (pd->type == CS_PTYPE_FLOAT || pd->type == CS_PTYPE_INT || pd->type == CS_PTYPE_UINT)
    {
        double v    = strtod(cur, NULL);
        double mult = coarse ? 5.0 : 1.0;
        double nv   = clamp_d(v + (double) dir * pd->step_v * mult, pd->min_v, pd->max_v);
        char   nbuf[64];
        if (pd->type == CS_PTYPE_FLOAT)
        {
            snprintf(nbuf, sizeof(nbuf), "%.8g", nv);
        }
        else
        {
            snprintf(nbuf, sizeof(nbuf), "%lld", (long long) llround(nv));
        }
        return cs_param_set_str(cfg, sel_item, pd, nbuf, err_buf, err_cap);
    }
    return -1;
}
