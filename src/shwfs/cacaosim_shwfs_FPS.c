// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs_FPS.c
 * @brief   MILK_RUNNABLE compute unit for Diffractive Shack-Hartmann WFS and detector readout
 *
 * Architectural Role in cacaosim:
 *   Implements the standalone MILK Function Parameter Structure (FPS) process (`csim_shwfs` /
 *   `milk-fpsexec-cacaosim-shwfs`) wrapping `cacaosim_shwfs_ctx_t`. Binds upstream shared-memory
 *   optical inputs (unified 3D `[OPD_um, Amplitude]` `.in_beam` or split `.in_wf` + `.in_dm` +
 *   `.in_pup` streams), applies optional 2D pupil affine registration and spectral bandpass flux
 *   integration, accumulates `.navg` subframes, and publishes stitched `[n_sub*pix_per_sub]^2`
 *   detector frames to `.out_cam`.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Physical Diffractive Subaperture Spots in CACAO Loops:
 *      Provides CACAO wavefront control pipelines with realistic lenslet focal-plane images
 *      exhibiting diffraction-limited spot widths, high-WFE speckle breakup, spot elongation, and
 *      edge/obscuration/spider vignetting via L1-cached OpenMP FFTW3f or GPU batched
 *      `cufftPlanMany` propagation.
 *   2. Seamless Modular Bench & Calibration Support:
 *      Reads upstream 32-bin stellar photon spectra (`CSIM_BM` / `S00..S31`) when `.in_beam` and
 *      `.use_beam_flux` are active, while exposing live `FPFLAG_WRITERUN` controls on `.turb_gain`,
 *      `.flux_ph`, `.ron_e`, `.gain_eadu`, `.shutter`, and `.navg` so reference slopes, darks, and
 *      interaction matrices can be acquired without restarting the WFS process.
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_time.h"
#include "pupil/cacaosim_pupil.h"
#include "shwfs/cacaosim_shwfs.h"
#include "shwfs/cacaosim_shwfs_FPS.h"

/*
 * File-scope FPS parameter backing variables bound by `CACAOSIM_SHWFS_PARAMS(X)`.
 * Parameters carrying `FPFLAG_WRITERUN` are sampled dynamically inside the real-time loop.
 */
static char     param_out_cam[FUNCTION_PARAMETER_STRMAXLEN] = "csim_shwfs";
static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]   = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char     param_in_dm[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_dmopd";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_pup";
static char     param_in_ncpa[FUNCTION_PARAMETER_STRMAXLEN] = "";
static uint32_t param_nwf                                   = 128;
static uint32_t param_n_sub                                 = 16;
static uint32_t param_pix_per_sub                           = 8;
static uint32_t param_pad_factor                            = 2;
static float    param_pup_diam                              = 0.9375f;
static float    param_lambda_um                             = 0.75f;
static float    param_bw_frac                               = 0.20f;
static int32_t  param_nlambda                               = 0;
static float    param_qe                                    = 0.80f;
static float    param_dt_exp                                = 0.001f;
static int64_t  param_use_beam_flux                         = 1;
static uint32_t param_navg                                  = 1;
static float    param_shift_x                               = 0.0f;
static float    param_shift_y                               = 0.0f;
static float    param_rot_deg                               = 0.0f;
static float    param_mag_scale                             = 1.0f;
static int32_t  param_wf_unit                               = 1;
static float    param_ref_lambda                            = 0.50f;
static float    param_turb_gain                             = 1.0f;
static float    param_flux_ph                               = 100000.0f;
static float    param_dark_e                                = 0.0f;
static float    param_ron_e                                 = 0.5f;
static float    param_gain_eadu                             = 1.0f;
static float    param_bias_adu                              = 100.0f;
static float    param_sat_adu                               = 65535.0f;
static int64_t  param_shutter                               = 1;
static int64_t  param_photon_noise                          = 1;
static int32_t  param_out_dtype                             = 0;
static float    param_frame_rate                            = 0.0f;
static uint64_t param_rng_seed                              = 23456;
static int32_t  param_backend                               = 0;

/**
 * @brief X-macro FPS parameter table for `cacaosim.shwfs` (`csim_shwfs`).
 *
 * Defines SHM stream endpoints, lenslet array geometry (`.nwf`, `.n_sub`, `.pix_per_sub`,
 * `.pad_factor`), spectral filter photometry, `.navg` multi-subframe integration, 2D pupil
 * registration (`.shift_x`, `.shift_y`, `.rot_deg`, `.mag_scale`), detector noise/ADU settings,
 * monotonic frame pacing, and CPU/CUDA backend selection.
 */
#define CACAOSIM_SHWFS_PARAMS(X)                                                        \
    X(".out_cam", param_out_cam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D Shack-Hartmann WFS camera stream [n_sub*pix_per_sub]")                 \
    X(".in_dm", param_in_dm, FPTYPE_STREAMNAME, 1,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 2D DM OPD stream [um] (optional trigger)")                                 \
    X(".in_beam", param_in_beam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional 3D [nwf x nwf x 2] input optical beam stream")                          \
    X(".in_wf", param_in_wf, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional atmospheric WF stream (e.g. outarraypha)")                              \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional atmospheric amplitude stream")                                          \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".in_ncpa", param_in_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional static WFS-path NCPA OPD stream [um]")                                  \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".n_sub", &param_n_sub, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                   \
      "Number of lenslets across pupil (default: 16)")                                  \
    X(".pix_per_sub", &param_pix_per_sub, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,       \
      "Detector pixels per lenslet side (default: 8 -> 128x128 cam)")                   \
    X(".pad_factor", &param_pad_factor, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "Subaperture FFT oversampling factor (>=2, default: 2)")                          \
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
      "Live atmospheric turbulence multiplier (0 = off)")                               \
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

MILK_PARAMS_DEFINE(cacaosim_shwfs, CACAOSIM_SHWFS_PARAMS);

/**
 * shwfs_confcheck() - Validate FPS configuration and enforce `FPFLAG_WRITERUN` on live parameters
 *
 * Role:
 *   Cold-path FPS `confcheck` hook invoked during configuration updates.
 *
 * Rationale:
 *   Guarantees runtime mutability on turbulence gain, photometry, shutter, detector noise, pacing,
 *   and `navg` parameters even when reattaching to an existing persisted FPS SHM table.
 *
 * Return: `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t shwfs_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".turb_gain", ".flux_ph",       ".dark_e",    ".ron_e",  ".gain_eadu",
            ".bias_adu",  ".shutter",       ".photon_noise", ".frame_rate", ".navg",
            ".qe",        ".dt_exp",        ".use_beam_flux"
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
 * shwfs_compute() - Main MILK `processinfo` real-time Shack-Hartmann WFS simulation loop
 * @self: Pointer to `runnable_cacaosim_shwfs` descriptor (non-NULL)
 *
 * Role:
 *   Binds/creates SHM streams, initializes `cacaosim_shwfs_ctx_t`, runs stream-triggered or
 *   `TIMER_ABSTIME`-paced diffractive subaperture integrations and camera readouts, and frees all
 *   resources cleanly on termination.
 *
 * Rationale:
 *   - Synthesizes a fallback 4-spider obstructed pupil when `.in_pup` is absent so standalone
 *     launches work out of the box.
 *   - Supports lazy reconnection to `.in_dm`, `.in_wf`, and `.in_amp` inside the hot loop and
 *     optional 2D pupil affine registration (`cacaosim_beam_relay_cpu`) on `.in_beam`.
 *   - Publishes `.out_cam` atomically (`md->write = 1` -> `processinfo_update_output_stream`) once
 *     every `navg` optical subframes.
 *
 * Return: `RETURN_SUCCESS` on clean stop, `RETURN_FAILURE` on invalid geometry or init failure.
 */
static MILK_HOT errno_t shwfs_compute(const MILK_RUNNABLE *self)
{
    if (param_out_cam[0] == '\0' || param_nwf < 8 || param_n_sub < 1 || param_pix_per_sub < 2)
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
    uint32_t     ncam         = param_n_sub * param_pix_per_sub;
    float       *fallback_pup = NULL;
    const float *pup_ptr      = NULL;

    /* Use upstream SHM pupil amplitude if present, else synthesize default obstructed 4-spider */
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

    /* Connect or pre-create 3D [OPD_um, Amp] input beam stream with CSIM_BM spectral keywords */
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
    IMGID   img_cam = stream_connect_create_2D(param_out_cam, ncam, ncam, out_dt);
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

    cacaosim_shwfs_cfg_t cfg = {
        .nwf         = nwf,
        .n_sub       = param_n_sub,
        .pix_per_sub = param_pix_per_sub,
        .pad_factor  = param_pad_factor,
        .pup_diam    = param_pup_diam,
        .lambda_um   = param_lambda_um,
        .bw_frac     = param_bw_frac,
        .nlambda     = param_nlambda,
        .wf_unit     = param_wf_unit,
        .ref_lambda  = param_ref_lambda,
        .rng_seed    = param_rng_seed,
        .backend     = param_backend,
        .det         = {
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

    cacaosim_shwfs_ctx_t wfs_ctx;
    if (cacaosim_shwfs_init(&wfs_ctx, &cfg, pup_ptr, ncpa_ptr, NULL) != 0)
    {
        free(fallback_pup);
        imgid_free(&img_cam);
        return RETURN_FAILURE;
    }

    /* Allocate 2D pupil affine registration workspace only when non-identity transform is set */
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
        /* Optional absolute-deadline monotonic pacing when running internally clocked */
        if (param_frame_rate > 0.1f)
        {
            uint64_t period_ns = (uint64_t) (1.0e9f / param_frame_rate);
            cacaosim_timespec_add_ns(&deadline, period_ns);
            cacaosim_sleep_until(&deadline);
        }

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
                cacaosim_shwfs_set_weights(&wfs_ctx, dyn_w);
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
            cacaosim_shwfs_set_disp(&wfs_ctx, eff_disp, b_diam, b_pmode);

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
            cacaosim_shwfs_accum_beam_poly(&wfs_ctx, b_opd, b_amp,
                                           (init_b_nwl == b_nwl && b_nwl > 1u) ? beam_lam : NULL,
                                           b_nwl, b_stride, ncpa_ptr, NULL, NULL);
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

            cacaosim_shwfs_accum_subframe(&wfs_ctx, pup_ptr, amp_ptr, wf_ptr, dm_ptr, ncpa_ptr,
                                          NULL, param_turb_gain);
        }

        /* Complete WFS camera exposure once `navg` subframes have been integrated */
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
            cacaosim_shwfs_readout(&wfs_ctx, &det_live,
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
    cacaosim_shwfs_free(&wfs_ctx);
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
 * @brief Registered MILK FPS runnable definition and standalone CLI export for `cacaosim.shwfs`.
 */
const MILK_RUNNABLE runnable_cacaosim_shwfs = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_shwfs",
              .cmdkey      = "shwfs",
              .description = "Simulate diffractive Shack-Hartmann lenslet WFS and detector" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_shwfs),
    .compute   = shwfs_compute,
    .confcheck = shwfs_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-shwfs", runnable_cacaosim_shwfs);
