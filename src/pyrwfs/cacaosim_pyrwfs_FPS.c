// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs_FPS.c
 * @brief   MILK_RUNNABLE compute unit for 4-sided Pyramid WFS and detector readout simulation
 *
 * Architectural Role in cacaosim:
 *   Implements the standalone MILK Function Parameter Structure (FPS) process (`csim_pyrwfs` /
 *   `milk-fpsexec-cacaosim-pyrwfs`) wrapping `cacaosim_pyrwfs_ctx_t`. Connects to upstream CACAO /
 *   `cacaosim` shared-memory streams (either a unified 3D `[OPD_um, Amplitude]` optical beam
 *   `.in_beam` or split `.in_wf` + `.in_dm` + `.in_pup` streams), applies optional 2D pupil-plane
 *   affine registration and spectral bandpass photometry, integrates `.navg` subframes, and
 *   publishes noisy WFS camera frames to `.out_cam` (e.g. `aol0_wfsim`).
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Dual Input Topology (Modular Beam vs Split Legacy Streams):
 *      Supporting `.in_beam` enables plug-and-play multi-stage AO benches where dichroics, static
 *      WFE maps, and upstream DMs propagate spectral photon budgets (`CSIM_BM` / `S00..S31` SHM
 *      keywords) and dynamic pupil amplitudes automatically. When `.in_beam` is omitted, direct
 *      `.in_wf` / `.in_dm` / `.in_pup` binding (with automatic VLT/Subaru-like spiders fallback if
 *      `.in_pup` is absent) lets users drop `cacaosim.pyrwfs` into standard CACAO `milkatmturb`
 *      loops with zero upstream glue.
 *   2. Fourier-Shift `(1 + N_mod)` Optics + Live Runtime Tunability:
 *      Delegates optical propagation to `cacaosim_pyrwfs.h`, where precomputed shifted focal-plane
 *      pyramid masks reduce per-subframe FFT cost from `2 * N_mod` to `1 + N_mod` 2D FFTs (via
 *      L2-cached OpenMP FFTW3f or batched cuFFT + CUDA Graphs). Marking `.mod_rad`, `.turb_gain`,
 *      `.flux_ph`, `.ron_e`, `.gain_eadu`, `.shutter`, and `.navg` with `FPFLAG_WRITERUN` lets AO
 *      operators acquire darks, measure interaction matrices (`turb_gain = 0`), and ramp modulation
 *      or guide-star magnitude live without restarting the WFS process.
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_time.h"
#include "pupil/cacaosim_pupil.h"
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "pyrwfs/cacaosim_pyrwfs_FPS.h"

/*
 * File-scope FPS parameter backing variables bound by `CACAOSIM_PYRWFS_PARAMS(X)`.
 * Parameters carrying `FPFLAG_WRITERUN` are re-sampled on every loop iteration so live `milk-fps`
 * updates take effect on the very next subframe or camera readout without process restart.
 */
static char     param_out_cam[FUNCTION_PARAMETER_STRMAXLEN]  = "aol0_wfsim";
static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]    = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char     param_in_dm[FUNCTION_PARAMETER_STRMAXLEN]    = "csim_dmopd";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_pup";
static char     param_in_ncpa[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static uint32_t param_nwf                                    = 128;
static uint32_t param_pad_factor                             = 2;
static uint32_t param_ncam                                   = 64;
static float    param_pup_diam                               = 0.9375f;
static float    param_lambda_um                              = 0.75f;
static float    param_bw_frac                                = 0.20f;
static int32_t  param_nlambda                                = 0;
static float    param_qe                                     = 0.80f;
static float    param_dt_exp                                 = 0.001f;
static int64_t  param_use_beam_flux                          = 1;
static uint32_t param_navg                                   = 1;
static float    param_mod_rad                                = 3.0f;
static int32_t  param_mod_npts                               = 16;
static int32_t  param_mod_mode                               = 1;
static int32_t  param_mod_rev                                = 1;
static float    param_mod_phase_deg                          = 0.0f;
static float    param_pup_sep                                = 0.50f;
static float    param_roof_ld                                = 0.0f;
static float    param_shift_x                                = 0.0f;
static float    param_shift_y                                = 0.0f;
static float    param_rot_deg                                = 0.0f;
static float    param_mag_scale                              = 1.0f;
static int32_t  param_wf_unit                                = 1;
static float    param_ref_lambda                             = 0.50f;
static float    param_turb_gain                              = 1.0f;
static float    param_flux_ph                                = 100000.0f;
static float    param_dark_e                                 = 0.0f;
static float    param_ron_e                                  = 0.5f;
static float    param_gain_eadu                              = 1.0f;
static float    param_bias_adu                               = 100.0f;
static float    param_sat_adu                                = 65535.0f;
static int64_t  param_shutter                                = 1;
static int64_t  param_photon_noise                           = 1;
static int32_t  param_out_dtype                              = 0;
static float    param_frame_rate                             = 0.0f;
static uint64_t param_rng_seed                               = 12345;
static int32_t  param_backend                                = 0;

/**
 * @brief X-macro FPS parameter table for `cacaosim.pyrwfs` (`csim_pyrwfs`).
 *
 * Groups parameters into:
 *   - Stream bindings (`.out_cam`, `.in_dm` trigger, `.in_beam`, `.in_wf`, `.in_amp`, `.in_pup`,
 *     `.in_ncpa`),
 *   - Optical & pyramid prism geometry (`.nwf`, `.pad_factor`, `.ncam`, `.pup_diam`, `.lambda_um`,
 *     `.mod_rad`, `.mod_npts`, `.pup_sep`, `.roof_ld`),
 *   - Spectral bandpass photometry & subframe integration (`.bw_frac`, `.qe`, `.dt_exp`,
 *     `.use_beam_flux`, `.navg`),
 *   - 2D pupil-plane affine misregistration (`.shift_x`, `.shift_y`, `.rot_deg`, `.mag_scale`),
 *   - Wavefront scaling, detector readout electronics, pacing, and compute backend selection.
 */
#define CACAOSIM_PYRWFS_PARAMS(X)                                                       \
    X(".out_cam", param_out_cam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D WFS camera stream (e.g. aol0_wfsim)")                                  \
    X(".in_dm", param_in_dm, FPTYPE_STREAMNAME, 1,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 2D DM OPD stream [um] (optional trigger)")                                 \
    X(".in_beam", param_in_beam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional 3D [nwf x nwf x 2] input optical beam stream")                          \
    X(".in_wf", param_in_wf, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional atmospheric WF stream (e.g. outarraypha from milkatmturb)")             \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional atmospheric amplitude stream (e.g. outarrayamp)")                       \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".in_ncpa", param_in_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional static WFS-path NCPA OPD stream [um]")                                  \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".pad_factor", &param_pad_factor, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "Optical FFT oversampling factor (>=2, default: 2)")                              \
    X(".ncam", &param_ncam, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                     \
      "WFS detector grid size [pixels] (default: 64)")                                  \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Clear pupil diameter fraction of nwf (default: 0.9375)")                         \
    X(".lambda_um", &param_lambda_um, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "WFS sensing wavelength [um] (default: 0.75)")                                    \
    X(".bw_frac", &param_bw_frac, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Filter fractional bandwidth dlambda/lambda0 (default: 0.20)")                    \
    X(".nlambda", &param_nlambda, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Polychromatic spectral slices (0 = auto -> 2, 1..32, default: 0)")               \
    X(".qe", &param_qe, FPTYPE_FLOAT32, 0,                                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Detector quantum efficiency [0..1] for beam photometry (default: 0.80)")         \
    X(".dt_exp", &param_dt_exp, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Subframe step [s] for beam photometry when frame_rate == 0 (default: 0.001)")    \
    X(".use_beam_flux", &param_use_beam_flux, FPTYPE_ONOFF, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Compute flux_ph from in_beam photon spectrum when available")                    \
    X(".navg", &param_navg, FPTYPE_UINT32, 0,                                           \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Optical subframes integrated per WFS readout (>=1, default: 1)")                 \
    X(".mod_rad", &param_mod_rad, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Modulation radius [lambda/D] (0 = unmodulated, default: 3.0)")                   \
    X(".mod_npts", &param_mod_npts, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Number of modulation points around circle (default: 16)")                        \
    X(".mod_mode", &param_mod_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Subframe modulation mode: 0=full, 1=sync_arc, 2=interleaved (default: 1)")       \
    X(".mod_rev", &param_mod_rev, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Modulator revolutions per WFS exposure (default: 1)")                            \
    X(".mod_phase_deg", &param_mod_phase_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,  \
      "Initial modulator azimuth at exposure start [deg] (default: 0.0)")               \
    X(".pup_sep", &param_pup_sep, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Sub-pupil center separation fraction of FFT grid (default: 0.50)")               \
    X(".roof_ld", &param_roof_ld, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Pyramid vertex roof width [lambda/D] (default: 0.0)")                            \
    X(".shift_x", &param_shift_x, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "WFS pupil X registration shift [pixels] (default: 0)")                           \
    X(".shift_y", &param_shift_y, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "WFS pupil Y registration shift [pixels] (default: 0)")                           \
    X(".rot_deg", &param_rot_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "WFS pupil CCW rotation [deg] (default: 0)")                                      \
    X(".mag_scale", &param_mag_scale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "WFS pupil magnification (default: 1.0)")                                         \
    X(".wf_unit", &param_wf_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1)")                   \
    X(".ref_lambda", &param_ref_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reference wavelength [um] for phase input (default: 0.50)")                      \
    X(".turb_gain", &param_turb_gain, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live atmospheric turbulence multiplier (0 = off for calibration)")               \
    X(".flux_ph", &param_flux_ph, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Total guide-star photons per WFS frame (<0 raw intensity)")                      \
    X(".dark_e", &param_dark_e, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Dark + sky background [e-/pix/frame]")                                           \
    X(".ron_e", &param_ron_e, FPTYPE_FLOAT32, 0,                                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Detector readout noise [e- RMS] (default: 0.5)")                                 \
    X(".gain_eadu", &param_gain_eadu, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Camera gain [e-/ADU] (default: 1.0)")                                            \
    X(".bias_adu", &param_bias_adu, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Camera bias offset [ADU] (default: 100.0)")                                      \
    X(".sat_adu", &param_sat_adu, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Detector saturation ceiling [ADU] (default: 65535)")                             \
    X(".shutter", &param_shutter, FPTYPE_ONOFF, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Optical shutter (ON = open, OFF = dark frame)")                                  \
    X(".photon_noise", &param_photon_noise, FPTYPE_ONOFF, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Enable Poisson photon shot noise")                                               \
    X(".out_dtype", &param_out_dtype, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Output stream datatype: 0 = uint16, 1 = float32")                                \
    X(".frame_rate", &param_frame_rate, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Internal TIMER_ABSTIME pacing rate [Hz] (0 = use procinfo trigger)")             \
    X(".rng_seed", &param_rng_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,             \
      "Detector noise PRNG seed")                                                       \
    X(".backend", &param_backend, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Compute backend: 0=auto, 1=CPU, 2=CUDA")

MILK_PARAMS_DEFINE(cacaosim_pyrwfs, CACAOSIM_PYRWFS_PARAMS);

/**
 * pyrwfs_confcheck() - Validate FPS configuration and enforce live `FPFLAG_WRITERUN` mutability
 *
 * Role:
 *   Cold-path FPS `confcheck` callback executed during `confstart` / parameter updates.
 *
 * Rationale:
 *   Pre-existing FPS shared-memory files on disk may have been created by an older schema lacking
 *   `FPFLAG_WRITERUN` on live-tunable keys; explicitly re-applying `FPFLAG_WRITERUN` guarantees
 *   that runtime modulation, shutter, photometry, and noise adjustments always remain writable
 *   while `run` is active.
 *
 * Return: `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t pyrwfs_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".mod_rad",   ".turb_gain",     ".flux_ph",       ".dark_e",     ".ron_e",
            ".gain_eadu", ".bias_adu",      ".shutter",       ".photon_noise", ".frame_rate",
            ".navg",      ".qe",            ".dt_exp",        ".use_beam_flux"
        };
        for (size_t i = 0; i < sizeof(live_keys) / sizeof(live_keys[0]); i++)
        {
            long idx = functionparameter_GetParamIndex(dcfpsptr, live_keys[i]);
            if (idx >= 0)
            {
                dcfpsptr->parray[idx].fpflag |= FPFLAG_WRITERUN;
            }
        }
    }
    return RETURN_SUCCESS;
}

/**
 * pyrwfs_compute() - Main MILK `processinfo` real-time Pyramid WFS simulation loop
 * @self: Pointer to `runnable_cacaosim_pyrwfs` descriptor (non-NULL)
 *
 * Role:
 *   Connects/creates SHM input and output streams, initializes `cacaosim_pyrwfs_ctx_t`, executes
 *   paced or stream-triggered optical subframe integrations and WFS camera readouts inside a
 *   `MILK_PROCINFO` loop, and cleanly releases all host/GPU/SHM resources on stop.
 *
 * Rationale:
 *   - Auto-synthesizes an anti-aliased obstructed 4-spider telescope pupil if `.in_pup` is missing,
 *     allowing standalone WFS startup before upstream pupil generators run.
 *   - Pre-creates `.in_beam` with initialized `CSIM_BM` spectral keywords if configured before
 *     upstream `mkbeam`/`beamsplit` processes start, preventing race conditions in parallel bench
 *     launches.
 *   - Re-attempts `stream_connect()` lazily inside the loop for optional `.in_dm`, `.in_wf`, and
 *     `.in_amp` streams so processes can start in arbitrary order.
 *   - Honors the MILK SHM concurrency protocol (`img_cam.md->write = 1` before writing pixels,
 *     finalized by `processinfo_update_output_stream()` only when `subframe_cnt >= navg`) so
 *     downstream CACAO WFS acquisition sees atomic, tear-free camera frames.
 *
 * Return: `RETURN_SUCCESS` on normal loop exit, `RETURN_FAILURE` on invalid geometry or init error.
 */
static MILK_HOT errno_t pyrwfs_compute(const MILK_RUNNABLE *self)
{
    if (param_out_cam[0] == '\0' || param_nwf < 8 || param_ncam < 4)
    {
        return RETURN_FAILURE;
    }

    IMGID img_pup = { 0 };
    img_pup.ID    = -1;
    if (param_in_pup[0] != '\0')
    {
        img_pup = stream_connect(param_in_pup);
    }

    uint32_t     nwf          = (img_pup.ID != -1) ? img_pup.md->size[0] : param_nwf;
    uint64_t     npix         = (uint64_t) nwf * nwf;
    float       *fallback_pup = NULL;
    const float *pup_ptr      = NULL;

    /* Bind shared-memory pupil amplitude if available; otherwise synthesize default 4-spider pup */
    if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT)
    {
        pup_ptr = img_pup.im->array.F;
    }
    else
    {
        fallback_pup = (float *) cacaosim_aligned_calloc((size_t) npix, sizeof(float));
        if (fallback_pup == NULL)
        {
            return RETURN_FAILURE;
        }
        cacaosim_pupil_cfg_t pcfg = {
            .nwf        = nwf,
            .pup_diam   = param_pup_diam,
            .cobs       = 0.30f,
            .nspider    = 4,
            .spider_w   = 0.01f,
            .spider_deg = 45.0f,
        };
        cacaosim_make_pupil(&pcfg, fallback_pup);
        pup_ptr = fallback_pup;
    }

    IMGID img_beam = { 0 };
    IMGID img_dm   = { 0 };
    IMGID img_wf   = { 0 };
    IMGID img_amp  = { 0 };
    IMGID img_ncpa = { 0 };
    img_beam.ID    = -1;
    img_dm.ID      = -1;
    img_wf.ID      = -1;
    img_amp.ID     = -1;
    img_ncpa.ID    = -1;

    /* Connect or pre-allocate 3D [OPD_um, Amp] input beam stream for race-free bench startup */
    if (param_in_beam[0] != '\0')
    {
        img_beam = stream_connect(param_in_beam);
        if (img_beam.ID == -1)
        {
            imgid_free(&img_beam);
            img_beam = cacaosim_beam_shm_create(param_in_beam, nwf, 2);
            if (img_beam.ID != -1)
            {
                cacaosim_beam_shm_init_kw(img_beam.im);
            }
        }
    }
    if (param_in_dm[0] != '\0')
    {
        img_dm = stream_connect(param_in_dm);
    }
    if (param_in_wf[0] != '\0')
    {
        img_wf = stream_connect(param_in_wf);
    }
    if (param_in_amp[0] != '\0')
    {
        img_amp = stream_connect(param_in_amp);
    }
    if (param_in_ncpa[0] != '\0')
    {
        img_ncpa = stream_connect(param_in_ncpa);
    }

    uint8_t out_dt  = (param_out_dtype == 0) ? _DATATYPE_UINT16 : _DATATYPE_FLOAT;
    IMGID   img_cam = stream_connect_create_2D(param_out_cam, param_ncam, param_ncam, out_dt);
    if (img_cam.ID == -1)
    {
        free(fallback_pup);
        imgid_free(&img_cam);
        return RETURN_FAILURE;
    }

    const float *ncpa_ptr = (img_ncpa.ID != -1 && img_ncpa.md->datatype == _DATATYPE_FLOAT &&
                             img_ncpa.md->size[0] == nwf)
                                ? img_ncpa.im->array.F
                                : NULL;

    cacaosim_pyrwfs_cfg_t cfg = {
        .nwf               = nwf,
        .pad_factor        = param_pad_factor,
        .ncam              = param_ncam,
        .pup_diam          = param_pup_diam,
        .lambda_um         = param_lambda_um,
        .bw_frac           = param_bw_frac,
        .nlambda           = param_nlambda,
        .mod_rad_ld        = param_mod_rad,
        .mod_npts          = param_mod_npts,
        .mod_subframe_mode = param_mod_mode,
        .mod_rev           = param_mod_rev,
        .mod_phase_deg     = param_mod_phase_deg,
        .mod_subframes     = (param_navg >= 1u) ? param_navg : 1u,
        .pup_sep           = param_pup_sep,
        .roof_ld           = param_roof_ld,
        .fstop_frac        = 0.45f,
        .wf_unit           = param_wf_unit,
        .ref_lambda        = param_ref_lambda,
        .rng_seed          = param_rng_seed,
        .backend           = param_backend,
        .det               = {
            .flux_ph      = param_flux_ph,
            .dark_e       = param_dark_e,
            .ron_e        = param_ron_e,
            .gain_eadu    = param_gain_eadu,
            .bias_adu     = param_bias_adu,
            .sat_adu      = param_sat_adu,
            .shutter      = (int32_t) param_shutter,
            .photon_noise = (int32_t) param_photon_noise,
            .out_dtype    = param_out_dtype,
        },
    };

    float    beam_lam[CACAOSIM_MAX_LAMBDA] = { 0.0f };
    uint32_t init_b_nwl                    = 1u;
    if (img_beam.ID != -1)
    {
        float b_wl0 = 0.0f;
        float b_wl1 = 0.0f;
        if (cacaosim_beam_shm_read_poly_kw(img_beam.im, &init_b_nwl, &b_wl0, &b_wl1) &&
            init_b_nwl > 1u)
        {
            if (init_b_nwl > CACAOSIM_MAX_LAMBDA)
            {
                init_b_nwl = CACAOSIM_MAX_LAMBDA;
            }
            if (!cacaosim_beam_shm_read_poly_wls(img_beam.im, init_b_nwl, beam_lam))
            {
                for (uint32_t c = 0; c < init_b_nwl; c++)
                {
                    float t     = (float) c / (float) (init_b_nwl - 1u);
                    beam_lam[c] = (b_wl0 > 0.05f && b_wl1 > 0.05f) ? (b_wl0 + (b_wl1 - b_wl0) * t)
                                                                   : param_lambda_um;
                }
            }
            if (cfg.nlambda <= 0)
            {
                cfg.nlambda = (int32_t) init_b_nwl;
            }
        }
        cacaosim_spec_t init_spec;
        if (param_use_beam_flux && cacaosim_beam_shm_read_spec(img_beam.im, &init_spec))
        {
            int32_t nl       = cacaosim_resolve_nlambda(cfg.nlambda, init_b_nwl, cfg.bw_frac);
            cfg.nlambda      = nl;
            cfg.has_ext_chan = 1;
            cacaosim_spec_sample_band(&init_spec, cfg.lambda_um, cfg.bw_frac, nl, 0.001f, param_qe,
                                      cfg.ext_lam_um, cfg.ext_weight, NULL);
        }
    }

    cacaosim_pyrwfs_ctx_t wfs_ctx;
    if (cacaosim_pyrwfs_init(&wfs_ctx, &cfg, pup_ptr, ncpa_ptr, NULL) != 0)
    {
        free(fallback_pup);
        imgid_free(&img_cam);
        return RETURN_FAILURE;
    }

    /* Pre-allocate pupil registration scratch buffer only when non-identity shift/rot/mag is set */
    cacaosim_reg_t reg = {
        .shift_x = param_shift_x,
        .shift_y = param_shift_y,
        .rot_deg = param_rot_deg,
        .scale   = param_mag_scale,
    };
    float *reg_buf = NULL;
    if (!cacaosim_reg_is_identity(&reg))
    {
        reg_buf = (float *) cacaosim_aligned_calloc(
            (size_t) CACAOSIM_MAX_LAMBDA * npix * 2u, sizeof(float));
    }

    cacaosim_disp_t b_disp     = { 0 };
    cacaosim_disp_t b_disp_reg = { 0 };
    uint64_t        reg_dgen   = 0ULL;

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        /* Optional drift-free monotonic pacing when running without an external stream trigger */
        if (param_frame_rate > 0.1f)
        {
            uint64_t period_ns = (uint64_t) (1.0e9f / param_frame_rate);
            cacaosim_timespec_add_ns(&deadline, period_ns);
            cacaosim_sleep_until(&deadline);
        }

        /* Lazy stream discovery for late-starting DM or atmospheric turbulence producers */
        if (img_dm.ID == -1 && param_in_dm[0] != '\0')
        {
            img_dm = stream_connect(param_in_dm);
        }
        if (img_wf.ID == -1 && param_in_wf[0] != '\0')
        {
            img_wf = stream_connect(param_in_wf);
        }
        if (img_amp.ID == -1 && param_in_amp[0] != '\0')
        {
            img_amp = stream_connect(param_in_amp);
        }

        float eff_flux = param_flux_ph;
        if (img_beam.ID != -1 && cacaosim_beam_shm_is_valid(img_beam.im, nwf))
        {
            cacaosim_spec_t b_spec;
            if (param_use_beam_flux && param_flux_ph >= 0.0f &&
                cacaosim_beam_shm_read_spec(img_beam.im, &b_spec))
            {
                uint32_t nav = (param_navg >= 1u) ? param_navg : 1u;
                float    dts = (param_frame_rate > 0.1f)
                                   ? (1.0f / param_frame_rate)
                                   : ((param_dt_exp > 1e-7f) ? param_dt_exp : 0.001f);
                float    dyn_w[CACAOSIM_MAX_LAMBDA];
                eff_flux = cacaosim_spec_sample_band(&b_spec, param_lambda_um, param_bw_frac,
                                                     wfs_ctx.eff_nlam, dts * (float) nav, param_qe,
                                                     NULL, dyn_w, NULL);
                cacaosim_pyrwfs_set_weights(&wfs_ctx, dyn_w);
            }

            uint32_t b_nwl = (img_beam.md->size[2] >= 4u) ? (img_beam.md->size[2] / 2u) : 1u;
            if (b_nwl > CACAOSIM_MAX_LAMBDA)
            {
                b_nwl = CACAOSIM_MAX_LAMBDA;
            }
            if (b_nwl > 1u && cacaosim_beam_shm_read_poly_wls(img_beam.im, b_nwl, beam_lam))
            {
                init_b_nwl = b_nwl;
            }
            cacaosim_beam_shm_read_disp(img_beam.im, &b_disp);
            const cacaosim_disp_t *eff_disp = &b_disp;
            if (reg_buf != NULL)
            {
                if (b_disp.active)
                {
                    if (b_disp.gen != reg_dgen || !b_disp_reg.active)
                    {
                        cacaosim_disp_apply_reg(&b_disp, &reg, &b_disp_reg);
                        b_disp_reg.gen = b_disp.gen;
                        reg_dgen       = b_disp.gen;
                    }
                    eff_disp = &b_disp_reg;
                }
                else
                {
                    b_disp_reg.active = 0;
                }
            }
            float   b_diam  = cacaosim_beam_shm_read_diam(img_beam.im, 8.0f);
            int32_t b_pmode = cacaosim_beam_shm_read_poly_mode(img_beam.im, b_nwl);
            cacaosim_pyrwfs_set_disp(&wfs_ctx, eff_disp, b_diam, b_pmode);

            const float *b_opd    = img_beam.im->array.F;
            const float *b_amp    = img_beam.im->array.F + npix;
            uint64_t     b_stride = npix * 2u;
            if (reg_buf != NULL)
            {
                for (uint32_t c = 0; c < b_nwl; c++)
                {
                    cacaosim_beam_relay_cpu(b_opd + (uint64_t) c * b_stride,
                                            b_amp + (uint64_t) c * b_stride, NULL, NULL, &reg, nwf,
                                            reg_buf + (uint64_t) c * b_stride,
                                            reg_buf + (uint64_t) c * b_stride + npix);
                }
                b_opd = reg_buf;
                b_amp = reg_buf + npix;
            }
            cacaosim_pyrwfs_accum_beam_poly(&wfs_ctx, b_opd, b_amp,
                                            (init_b_nwl == b_nwl && b_nwl > 1u) ? beam_lam : NULL,
                                            b_nwl, b_stride, ncpa_ptr, NULL, NULL, param_mod_rad);
        }
        else
        {
            const float *dm_ptr  = (img_dm.ID != -1 && img_dm.md->datatype == _DATATYPE_FLOAT &&
                                    img_dm.md->size[0] == nwf)
                                       ? img_dm.im->array.F
                                       : NULL;
            const float *wf_ptr  = (img_wf.ID != -1 && img_wf.md->datatype == _DATATYPE_FLOAT &&
                                    img_wf.md->size[0] == nwf)
                                       ? img_wf.im->array.F
                                       : NULL;
            const float *amp_ptr = (img_amp.ID != -1 && img_amp.md->datatype == _DATATYPE_FLOAT &&
                                    img_amp.md->size[0] == nwf)
                                       ? img_amp.im->array.F
                                       : NULL;

            cacaosim_pyrwfs_accum_subframe(&wfs_ctx, pup_ptr, amp_ptr, wf_ptr, dm_ptr, ncpa_ptr,
                                           NULL, param_turb_gain, param_mod_rad);
        }

        /* Read out noisy detector pixels and post SHM semaphore once `navg` subframes integrate */
        uint32_t target_navg = (param_navg >= 1u) ? param_navg : 1u;
        if (wfs_ctx.subframe_cnt >= target_navg)
        {
            cacaosim_det_cfg_t det_live = {
                .flux_ph      = eff_flux,
                .dark_e       = param_dark_e,
                .ron_e        = param_ron_e,
                .gain_eadu    = param_gain_eadu,
                .bias_adu     = param_bias_adu,
                .sat_adu      = param_sat_adu,
                .shutter      = (int32_t) param_shutter,
                .photon_noise = (int32_t) param_photon_noise,
                .out_dtype    = param_out_dtype,
            };

            img_cam.md->write = 1;
            cacaosim_pyrwfs_readout(&wfs_ctx, &det_live,
                                    (param_out_dtype == 0) ? img_cam.im->array.UI16 : NULL,
                                    (param_out_dtype != 0) ? img_cam.im->array.F : NULL);

            IMAGE *trig_src = (img_beam.ID != -1)
                                  ? img_beam.im
                                  : ((img_dm.ID != -1)
                                         ? img_dm.im
                                         : ((img_wf.ID != -1) ? img_wf.im : NULL));
            processinfo_update_output_stream(processinfo, img_cam.im, trig_src);
        }
    }
    MILK_PROCINFO_END(self)

    free(reg_buf);
    cacaosim_pyrwfs_free(&wfs_ctx);
    free(fallback_pup);
    if (img_ncpa.ID != -1)
    {
        imgid_free(&img_ncpa);
    }
    if (img_amp.ID != -1)
    {
        imgid_free(&img_amp);
    }
    if (img_wf.ID != -1)
    {
        imgid_free(&img_wf);
    }
    if (img_dm.ID != -1)
    {
        imgid_free(&img_dm);
    }
    if (img_beam.ID != -1)
    {
        imgid_free(&img_beam);
    }
    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
    imgid_free(&img_cam);
    return RETURN_SUCCESS;
}

/**
 * @brief Registered MILK FPS runnable definition and standalone CLI export for `cacaosim.pyrwfs`.
 */
const MILK_RUNNABLE runnable_cacaosim_pyrwfs = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_pyrwfs",
              .cmdkey      = "pyrwfs",
              .description = "Simulate modulated/unmodulated 4-sided Pyramid WFS and detector" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_pyrwfs),
    .compute   = pyrwfs_compute,
    .confcheck = pyrwfs_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-pyrwfs", runnable_cacaosim_pyrwfs);
