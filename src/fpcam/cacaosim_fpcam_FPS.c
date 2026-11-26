// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam_FPS.c
 * @brief   MILK_RUNNABLE FPS compute unit for focal-plane PSF & science camera simulation
 *
 * Architectural Role in `cacaosim`:
 *   Wraps the `cacaosim_fpcam` polychromatic Fraunhofer diffraction and detector readout engine
 *   (`cacaosim_fpcam_engine.c` / `cacaosim_fpcam_cuda.cu`) as a `milk` Function Parameter
 *   Structure (FPS) runnable (`cacaosim.fpcam`, standalone CLI `milk-fpsexec-cacaosim-fpcam`,
 *   default instance `csim_fpcam`). Connects `ImageStreamIO` shared-memory (SHM) streams in two
 *   interchangeable optical configurations:
 *     1. Beam-relayed / post-coronagraphic mode (`.in_beam` non-empty): Consumes dynamic 3D
 *        `[nwf x nwf x 2*N]` `[OPD_um, Amplitude]` SHM cubes (such as `out_beam` from
 *        `cacaosim.coro` or `cacaosim.beamsplit`), auto-discovers upstream spectral channel grids
 *        via `CSIM_NWL`/`CSIM_WL0`/`CSIM_WL1` keywords, optionally applies 2D affine pupil
 *        misregistration (`.shift_x`, `.shift_y`, `.rot_deg`, `.mag_scale`), and computes physical
 *        incident photon flux from the upstream 32-bin `CSIM_BM` photon-rate spectrum.
 *     2. Classical-AO stream mode (`.in_beam` empty): Combines static entrance pupil (`.in_pup`,
 *        or an auto-synthesized 4-spider aperture if absent), atmospheric wavefront (`.in_wf`),
 *        scintillation amplitude (`.in_amp`), deformable mirror OPD (`.in_dm`), and static
 *        science-path NCPA (`.in_ncpa`).
 *   Integrates `.navg` optical subframes per camera readout, publishes `uint16` or `float32`
 *   detector frames to `.out_cam` (with `md->write = 1` fencing and `processinfo` stream triggers)
 *   plus an optional cumulative Strehl-normalized long-exposure PSF to `.out_long`, and exports
 *   live `.strehl_inst` and `.strehl_ave` FPS telemetry.
 *
 * Design Rationale & Motivation:
 *   1. Clear-Pupil Reference Anchoring Downstream of Coronagraphs:
 *      - Even when `.in_beam` carries a heavily apodized post-Lyot coronagraphic field,
 *        `cacaosim_fpcam_init()` is initialized with the unocculted telescope pupil `.in_pup`.
 *        Consequently, raw contrast values in `.out_long`, `.strehl_inst`, and Parseval photon
 *        scaling (`inf_flux_norm`) remain rigorously normalized to the unocculted stellar peak and
 *        incident clear-pupil photon flux.
 *   2. Zero-Duplication Polychromatic & Radiometric Handshake:
 *      - Auto-reading `CSIM_NWL`/`CSIM_WL0`/`CSIM_WL1` at startup synchronizes camera MFT/FFT
 *        channels with upstream `cacaosim.coro` `poly_out` cubes automatically, while
 *        `cacaosim_spec_band_flux()` integrates upstream `CSIM_BM` spectral throughput over
 *        `(lambda_um, bw_frac, dts * navg, qe)` on every frame so dichroic losses, stellar
 *        magnitudes, and exposure changes propagate end-to-end without manual `flux_ph` math.
 *   3. Live `FPFLAG_WRITERUN` Camera Control & Monotonic Pacing:
 *      - Marking `.navg`, `.reset_long`, `.shutter`, `.flux_ph`, `.qe`, `.ron_e`, `.dark_e`,
 *        `.gain_eadu`, `.bias_adu`, `.photon_noise`, `.turb_gain`, and `.frame_rate` writable at
 *        runtime lets observers take darks, change integration times, or reset long exposures on
 *        the fly without tearing down precomputed MFT matrices or CUDA plans, while optional
 *        `CLOCK_MONOTONIC` `TIMER_ABSTIME` pacing prevents cumulative timing drift.
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_time.h"
#include "fpcam/cacaosim_fpcam.h"
#include "fpcam/cacaosim_fpcam_FPS.h"
#include "pupil/cacaosim_pupil.h"

/*
 * Static FPS parameter backing store bound by `CACAOSIM_FPCAM_PARAMS(X)`:
 *   - SHM I/O stream names (`param_out_cam`, `param_out_long`, `param_in_beam`, `param_in_dm`,
 *     `param_in_wf`, `param_in_amp`, `param_in_pup`, `param_in_ncpa`)
 *   - Optical grid, propagator, and spectral bandpass (`param_nwf`, `param_ncam`,
 *     `param_prop_mode`, `param_pad_factor`, `param_pup_diam`, `param_lambda_um`, `param_bw_frac`,
 *     `param_nlambda`, `param_pixscale_ld`)
 *   - Beam radiometry & 2D pupil affine registration (`param_qe`, `param_dt_exp`,
 *     `param_use_beam_flux`, `param_shift_x`, `param_shift_y`, `param_rot_deg`, `param_mag_scale`)
 *   - Classical-AO wavefront scaling, multi-subframe integration, detector electronics, monotonic
 *     pacing, backend selection, and live Strehl telemetry (`param_strehl_inst`,
 *     `param_strehl_ave`)
 */
static char     param_out_cam[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_fpcam";
static char     param_out_long[FUNCTION_PARAMETER_STRMAXLEN] = "csim_fplong";
static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char     param_in_dm[FUNCTION_PARAMETER_STRMAXLEN]    = "csim_dmopd";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]    = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_pup";
static char     param_in_ncpa[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static uint32_t param_nwf                                    = 128;
static uint32_t param_ncam                                   = 128;
static int32_t  param_prop_mode                              = 0;
static uint32_t param_pad_factor                             = 2;
static float    param_pup_diam                               = 0.9375f;
static float    param_lambda_um                              = 1.65f;
static float    param_bw_frac                                = 0.18f;
static int32_t  param_nlambda                                = 0;
static float    param_pixscale_ld                            = 3.0f;
static float    param_qe                                     = 0.85f;
static float    param_dt_exp                                 = 0.001f;
static int64_t  param_use_beam_flux                          = 1;
static float    param_shift_x                                = 0.0f;
static float    param_shift_y                                = 0.0f;
static float    param_rot_deg                                = 0.0f;
static float    param_mag_scale                              = 1.0f;
static int32_t  param_wf_unit                                = 1;
static float    param_ref_lambda                             = 0.50f;
static float    param_turb_gain                              = 1.0f;
static uint32_t param_navg                                   = 1;
static int64_t  param_reset_long                             = 0;
static float    param_flux_ph                                = 1000000.0f;
static float    param_dark_e                                 = 0.0f;
static float    param_ron_e                                  = 1.0f;
static float    param_gain_eadu                              = 1.0f;
static float    param_bias_adu                               = 100.0f;
static float    param_sat_adu                                = 65535.0f;
static int64_t  param_shutter                                = 1;
static int64_t  param_photon_noise                           = 1;
static int32_t  param_out_dtype                              = 0;
static float    param_frame_rate                             = 0.0f;
static uint64_t param_rng_seed                               = 54321;
static int32_t  param_backend                                = 0;
static float    param_strehl_inst                            = 0.0f;
static float    param_strehl_ave                             = 0.0f;

/**
 * CACAOSIM_FPCAM_PARAMS() - X-macro FPS parameter table for `cacaosim.fpcam`
 * @X: Visitor macro `(key, ptr, type, req, flags, desc)` expanded by `MILK_PARAMS_DEFINE`
 *
 * Declares all shared-memory stream endpoints, MFT/FFT optical propagation geometry, polychromatic
 * bandpass and beam-spectrum radiometry settings, 2D pupil misregistration controls, detector
 * noise/ADU electronics, pacing, and live Strehl output telemetry for `csim_fpcam`.
 */
#define CACAOSIM_FPCAM_PARAMS(X)                                                        \
    X(".out_cam", param_out_cam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D focal plane camera stream")                                            \
    X(".out_long", param_out_long, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,              \
      "Optional output float32 cumulative Strehl-normalized long-exposure PSF")         \
    X(".in_dm", param_in_dm, FPTYPE_STREAMNAME, 1,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 2D DM OPD stream [um] (optional trigger)")                                 \
    X(".in_beam", param_in_beam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional 3D [nwf x nwf x 2*N] input optical beam stream (e.g. post-coronagraph)")\
    X(".in_wf", param_in_wf, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional atmospheric WF stream (e.g. outarraypha from milkatmturb)")             \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional atmospheric amplitude stream (e.g. outarrayamp)")                       \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".in_ncpa", param_in_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional static science-path NCPA OPD stream [um]")                              \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".ncam", &param_ncam, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                     \
      "Science camera grid size [pixels] (default: 128)")                               \
    X(".prop_mode", &param_prop_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Propagation mode: 0 = Soummer MFT, 1 = padded 2D FFT (default: 0)")              \
    X(".pad_factor", &param_pad_factor, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "FFT oversampling factor when prop_mode == 1 (>=2, default: 2)")                  \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Clear pupil diameter fraction of nwf (default: 0.9375)")                         \
    X(".lambda_um", &param_lambda_um, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Central science wavelength [um] (default: 1.65)")                                \
    X(".bw_frac", &param_bw_frac, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Fractional bandpass dlambda/lambda0 (default: 0.18)")                            \
    X(".nlambda", &param_nlambda, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Polychromatic spectral slices (0 = auto -> 2 if bw>0, 1..32, default: 0)")       \
    X(".pixscale_ld", &param_pixscale_ld, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "Focal sampling [pixels per lambda0/D] in MFT mode (default: 3.0)")               \
    X(".qe", &param_qe, FPTYPE_FLOAT32, 0,                                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Detector quantum efficiency [0..1] for beam photometry (default: 0.85)")         \
    X(".dt_exp", &param_dt_exp, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Subframe step [s] for beam photometry when frame_rate == 0 (default: 0.001)")    \
    X(".use_beam_flux", &param_use_beam_flux, FPTYPE_ONOFF, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Compute flux_ph from in_beam photon spectrum when available")                    \
    X(".shift_x", &param_shift_x, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Camera pupil X registration shift [pixels] (default: 0)")                        \
    X(".shift_y", &param_shift_y, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Camera pupil Y registration shift [pixels] (default: 0)")                        \
    X(".rot_deg", &param_rot_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Camera pupil CCW rotation [deg] (default: 0)")                                   \
    X(".mag_scale", &param_mag_scale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Camera pupil magnification (default: 1.0)")                                      \
    X(".wf_unit", &param_wf_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1)")                   \
    X(".ref_lambda", &param_ref_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reference wavelength [um] for phase input (default: 0.50)")                      \
    X(".turb_gain", &param_turb_gain, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live atmospheric turbulence multiplier (0 = off)")                               \
    X(".navg", &param_navg, FPTYPE_UINT32, 0,                                           \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Optical subframes integrated per camera readout (>=1, default: 1)")              \
    X(".reset_long", &param_reset_long, FPTYPE_ONOFF, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Toggle ON to clear cumulative long-exposure PSF")                                \
    X(".flux_ph", &param_flux_ph, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Total incident starlight photons per camera exposure (<0 raw Strehl PSF)")       \
    X(".dark_e", &param_dark_e, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Dark + sky background [e-/pix/exposure]")                                        \
    X(".ron_e", &param_ron_e, FPTYPE_FLOAT32, 0,                                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Detector readout noise [e- RMS] (default: 1.0)")                                 \
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
      "Internal TIMER_ABSTIME subframe pacing rate [Hz] (0 = use trigger)")             \
    X(".rng_seed", &param_rng_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,             \
      "Detector noise PRNG seed")                                                       \
    X(".backend", &param_backend, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Compute backend: 0=auto, 1=CPU, 2=CUDA")                                         \
    X(".strehl_inst", &param_strehl_inst, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,     \
      "Instantaneous exposure peak optical Strehl ratio [0..1]")                        \
    X(".strehl_ave", &param_strehl_ave, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,       \
      "Cumulative long-exposure peak optical Strehl ratio [0..1]")

MILK_PARAMS_DEFINE(cacaosim_fpcam, CACAOSIM_FPCAM_PARAMS);

/**
 * fpcam_confcheck() - FPS configuration validation callback enforcing `FPFLAG_WRITERUN` flags
 *
 * Role: Invoked by the `milk` FPS state machine during `confstart` / `confcheck` transitions to
 * ensure all live-tunable camera, exposure, shutter, radiometry, and turbulence keys remain
 * writable while `fpcam_compute()` is running.
 *
 * Rationale: Re-asserting `FPFLAG_WRITERUN` on `dcfpsptr` guarantees live parameter mutability
 * (`milk-fps-set` / `cacaosim-CTRL`) even when an FPS shared-memory table was restored from an
 * older on-disk snapshot lacking runtime write flags.
 *
 * Return: `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t fpcam_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".turb_gain",  ".navg",       ".reset_long", ".flux_ph",       ".dark_e",
            ".ron_e",      ".gain_eadu",  ".bias_adu",   ".shutter",       ".photon_noise",
            ".frame_rate", ".qe",         ".dt_exp",     ".use_beam_flux"
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
 * fpcam_compute() - Main real-time FPS optical propagation and detector readout loop
 * @self: Pointer to the active `MILK_RUNNABLE` descriptor (`runnable_cacaosim_fpcam`)
 *
 * Role: Connects or synthesizes the clear reference pupil (`.in_pup`), connects optional upstream
 * optical beam (`.in_beam`) or classical-AO streams (`.in_dm`, `.in_wf`, `.in_amp`, `.in_ncpa`),
 * creates `.out_cam` and optional `.out_long` SHM streams, auto-discovers upstream polychromatic
 * wavelengths (`CSIM_NWL`), initializes `cacaosim_fpcam_ctx_t`, and executes the zero-allocation
 * real-time subframe integration and camera readout loop until processinfo stop.
 *
 * Rationale: Lazily reconnecting optional `.in_dm`/`.in_wf`/`.in_amp` streams inside the loop
 * eliminates process startup races across independent tmux/FPS daemons. Pre-allocating `reg_buf`
 * outside the loop only when `.shift_x`/`.shift_y`/`.rot_deg`/`.mag_scale` is non-identity avoids
 * dead memory and unnecessary copies in aligned benches while supporting sub-pixel bilinear pupil
 * misregistration per spectral slice when needed. Setting `md->write = 1` before readout and
 * finalizing via `processinfo_update_output_stream()` ensures race-free SHM semaphore delivery to
 * downstream dark-hole estimators and telemetry viewers.
 *
 * Return: `RETURN_SUCCESS` on clean loop exit, `RETURN_FAILURE` on invalid geometry or SHM error.
 */
static MILK_HOT errno_t fpcam_compute(const MILK_RUNNABLE *self)
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

    /* Reference clear entrance pupil anchors Strehl = 1.0 and Parseval `inf_flux_norm` */
    uint32_t     nwf          = (img_pup.ID != -1) ? img_pup.md->size[0] : param_nwf;
    uint64_t     npix         = (uint64_t) nwf * nwf;
    float       *fallback_pup = NULL;
    const float *pup_ptr      = NULL;

    if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT)
    {
        pup_ptr = img_pup.im->array.F;
    }
    else
    {
        /* Synthesize a default 30%-obscured 4-spider telescope pupil when `.in_pup` is absent */
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
    IMGID img_long = { 0 };
    img_beam.ID    = -1;
    img_dm.ID      = -1;
    img_wf.ID      = -1;
    img_amp.ID     = -1;
    img_ncpa.ID    = -1;
    img_long.ID    = -1;

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

    if (param_out_long[0] != '\0')
    {
        img_long = stream_connect_create_2Df32(param_out_long, param_ncam, param_ncam);
        if (img_long.ID != -1)
        {
            cacaosim_fpcam_shm_write_kw(img_long.im, 0.0f, 0.0f, 0.0f, 0ULL);
        }
    }

    const float *ncpa_ptr = (img_ncpa.ID != -1 && img_ncpa.md->datatype == _DATATYPE_FLOAT &&
                             img_ncpa.md->size[0] == nwf)
                                ? img_ncpa.im->array.F
                                : NULL;

    cacaosim_fpcam_cfg_t cfg = {
        .nwf         = nwf,
        .ncam        = param_ncam,
        .prop_mode   = param_prop_mode,
        .pad_factor  = param_pad_factor,
        .pup_diam    = param_pup_diam,
        .lambda_um   = param_lambda_um,
        .bw_frac     = param_bw_frac,
        .nlambda     = param_nlambda,
        .pixscale_ld = param_pixscale_ld,
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

    /*
     * Auto-discover polychromatic channel count and endpoints (`CSIM_NWL`, `CSIM_WL0`, `CSIM_WL1`)
     * plus upstream 32-bin stellar spectrum (`CSIM_BM`) when `in_beam` is connected.
     */
    if (img_beam.ID != -1)
    {
        uint32_t b_nwl = 1u;
        float    b_wl0 = 0.0f;
        float    b_wl1 = 0.0f;
        if (cacaosim_beam_shm_read_poly_kw(img_beam.im, &b_nwl, &b_wl0, &b_wl1) && b_nwl > 1u)
        {
            if (b_nwl > CACAOSIM_FPCAM_MAX_LAMBDA)
            {
                b_nwl = CACAOSIM_FPCAM_MAX_LAMBDA;
            }
            cfg.has_in_lam = 1;
            if (!cacaosim_beam_shm_read_poly_wls(img_beam.im, b_nwl, cfg.in_lam_um))
            {
                for (uint32_t c = 0; c < b_nwl; c++)
                {
                    float t          = (b_nwl > 1u) ? ((float) c / (float) (b_nwl - 1u)) : 0.0f;
                    cfg.in_lam_um[c] = (b_wl0 > 0.05f && b_wl1 > 0.05f)
                                           ? (b_wl0 + (b_wl1 - b_wl0) * t)
                                           : param_lambda_um;
                }
            }
            if (cfg.nlambda <= 0)
            {
                cfg.nlambda     = (int32_t) b_nwl;
                cfg.has_ext_lam = 1;
                for (uint32_t c = 0; c < b_nwl; c++)
                {
                    cfg.ext_lam_um[c] = cfg.in_lam_um[c];
                }
            }
            if (cfg.lambda_um <= 0.05f && b_wl0 > 0.05f && b_wl1 > 0.05f)
            {
                cfg.lambda_um = 0.5f * (b_wl0 + b_wl1);
            }
        }
        cacaosim_spec_t init_spec;
        if (param_use_beam_flux && cacaosim_beam_shm_read_spec(img_beam.im, &init_spec))
        {
            int32_t nl         = cacaosim_resolve_nlambda(cfg.nlambda, b_nwl, cfg.bw_frac);
            cfg.nlambda        = nl;
            cfg.has_ext_weight = 1;
            float *lam_dst     = cfg.has_ext_lam ? NULL : cfg.ext_lam_um;
            cacaosim_spec_sample_band(&init_spec, cfg.lambda_um, cfg.bw_frac, nl, 0.001f, param_qe,
                                      lam_dst, cfg.ext_weight, NULL);
            if (!cfg.has_ext_lam)
            {
                cfg.has_ext_lam = 1;
            }
        }
    }

    cacaosim_fpcam_ctx_t fp_ctx;
    if (cacaosim_fpcam_init(&fp_ctx, &cfg, pup_ptr, ncpa_ptr, NULL) != 0)
    {
        free(fallback_pup);
        if (img_long.ID != -1)
        {
            imgid_free(&img_long);
        }
        imgid_free(&img_cam);
        return RETURN_FAILURE;
    }

    /* Pre-allocate 2D affine pupil relay scratch buffer only when non-identity registration set */
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
            (size_t) CACAOSIM_FPCAM_MAX_LAMBDA * npix * 2u, sizeof(float));
    }

    cacaosim_disp_t b_disp     = { 0 };
    cacaosim_disp_t b_disp_reg = { 0 };
    uint64_t        reg_dgen   = 0ULL;

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        if (param_frame_rate > 0.1f)
        {
            uint64_t period_ns = (uint64_t) (1.0e9f / param_frame_rate);
            cacaosim_timespec_add_ns(&deadline, period_ns);
            cacaosim_sleep_until(&deadline);
        }

        if (param_reset_long)
        {
            cacaosim_fpcam_reset_long(&fp_ctx);
            if (img_long.ID != -1)
            {
                cacaosim_fpcam_shm_write_kw(img_long.im, 0.0f, 0.0f, 0.0f, 0ULL);
            }
            param_reset_long = 0;
        }

        /* Lazy stream reconnects tolerate asynchronous producer startup ordering */
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
            /*
             * Beam-relayed / post-coronagraphic path: SHM packs `b_nwl` interleaved `[OPD, Amp]`
             * pairs with channel stride `b_stride = 2 * npix`.
             */
            uint32_t b_nwl = (img_beam.md->size[2] >= 4u) ? (img_beam.md->size[2] / 2u) : 1u;
            if (b_nwl > CACAOSIM_FPCAM_MAX_LAMBDA)
            {
                b_nwl = CACAOSIM_FPCAM_MAX_LAMBDA;
            }
            if (b_nwl > 1u)
            {
                float live_wls[CACAOSIM_FPCAM_MAX_LAMBDA];
                if (cacaosim_beam_shm_read_poly_wls(img_beam.im, b_nwl, live_wls))
                {
                    cacaosim_fpcam_set_in_lambdas(&fp_ctx, live_wls, b_nwl);
                }
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
            cacaosim_fpcam_set_disp(&fp_ctx, eff_disp, b_diam, b_pmode);

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
            /* Integrate upstream 32-bin photon spectrum over camera filter, `navg * dts`, and QE */
            cacaosim_spec_t b_spec;
            if (param_use_beam_flux && param_flux_ph >= 0.0f &&
                cacaosim_beam_shm_read_spec(img_beam.im, &b_spec))
            {
                uint32_t nav = (param_navg >= 1u) ? param_navg : 1u;
                float    dts = (param_frame_rate > 0.1f)
                                   ? (1.0f / param_frame_rate)
                                   : ((param_dt_exp > 1e-7f) ? param_dt_exp : 0.001f);
                float    dyn_w[CACAOSIM_FPCAM_MAX_LAMBDA];
                eff_flux = cacaosim_spec_sample_band(&b_spec, param_lambda_um, param_bw_frac,
                                                     fp_ctx.eff_nlam, dts * (float) nav, param_qe,
                                                     NULL, dyn_w, NULL);
                cacaosim_fpcam_set_weights(&fp_ctx, dyn_w);
            }
            cacaosim_fpcam_accum_beam_poly(&fp_ctx, b_opd, b_amp, b_nwl, b_stride, ncpa_ptr, NULL,
                                           NULL);
        }
        else
        {
            /* Classical-AO stream path (`in_pup` + `in_wf` + `in_amp` - `in_dm` + `in_ncpa`) */
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

            cacaosim_fpcam_accum_subframe(&fp_ctx, pup_ptr, amp_ptr, wf_ptr, dm_ptr, ncpa_ptr,
                                          NULL, param_turb_gain);
        }

        /* Read out detector and publish SHM streams once `navg` subframes have been integrated */
        uint32_t target_navg = (param_navg >= 1u) ? param_navg : 1u;
        if (fp_ctx.subframe_cnt >= target_navg)
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

            if (img_long.ID != -1)
            {
                img_long.md->write = 1;
            }
            img_cam.md->write = 1;
            cacaosim_fpcam_readout(&fp_ctx, &det_live,
                                   (param_out_dtype == 0) ? img_cam.im->array.UI16 : NULL,
                                   (param_out_dtype != 0) ? img_cam.im->array.F : NULL,
                                   (img_long.ID != -1) ? img_long.im->array.F : NULL);

            param_strehl_inst = fp_ctx.last_strehl_peak;
            param_strehl_ave  = fp_ctx.long_strehl_peak;

            IMAGE *trig_src = (img_beam.ID != -1)
                                  ? img_beam.im
                                  : ((img_dm.ID != -1)
                                         ? img_dm.im
                                         : ((img_wf.ID != -1) ? img_wf.im : NULL));
            processinfo_update_output_stream(processinfo, img_cam.im, trig_src);
            if (img_long.ID != -1)
            {
                cacaosim_fpcam_shm_write_kw(img_long.im, fp_ctx.long_strehl_peak,
                                            fp_ctx.last_strehl_peak, fp_ctx.last_strehl_axis,
                                            fp_ctx.long_cnt);
                processinfo_update_output_stream(processinfo, img_long.im, trig_src);
            }
        }
    }
    MILK_PROCINFO_END(self)

    free(reg_buf);
    cacaosim_fpcam_free(&fp_ctx);
    free(fallback_pup);
    if (img_long.ID != -1)
    {
        imgid_free(&img_long);
    }
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
 * runnable_cacaosim_fpcam - Exported `MILK_RUNNABLE` descriptor for `cacaosim.fpcam`
 *
 * Role: Registers the `fpcam` command key (`csim_fpcam` default FPS name) inside
 * `libmilkcacaosim.so` and generates the standalone `milk-fpsexec-cacaosim-fpcam` binary via
 * `EXPORT_FPSEXEC`.
 */
const MILK_RUNNABLE runnable_cacaosim_fpcam = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_fpcam",
              .cmdkey      = "fpcam",
              .description = "Simulate polychromatic MFT/FFT focal plane PSF and detector" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_fpcam),
    .compute   = fpcam_compute,
    .confcheck = fpcam_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-fpcam", runnable_cacaosim_fpcam);
