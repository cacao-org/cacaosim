// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_beamsplit_FPS.c
 * @brief   MILK_RUNNABLE dichroic / gray beamsplitter with per-arm NCPA, masks, & 2D registration
 *
 * Architectural Role:
 *   Implements `cacaosim.beamsplit` (`milk-fpsexec-cacaosim-beamsplit`), a zero-FFT
 *   pupil-plane branching unit that splits an upstream 3D `[nwf x nwf x 2]`
 *   optical beam (`.in_beam`) into two independent downstream SHM optical beams
 *   (`.out_refl` and `.out_trans`). Supports four coating laws via
 *   `cacaosim_spec_split()`: achromatic gray (`0`), long-pass dichroic (`1`),
 *   short-pass dichroic (`2`), and band-reflect notch/filter dichroic (`3`).
 *
 * Design Rationale & Motivation:
 *   - Smooth thin-film transition modeling: Real multi-layer dielectric dichroics
 *     exhibit finite spectral transition edges (`roll_um`, default `30 nm`) and
 *     sub-unity peak efficiencies (`eff_refl`, `eff_trans`). Modeling smooth
 *     logistic roll-offs over the 32-bin spectrum (`0.40..2.40 um`) while clamping
 *     `R(lambda) + T(lambda) <= 1` guarantees photon conservation and realistic
 *     chromatic cross-bleed when sensor bandpasses straddle `.lambda_cut`.
 *   - Co-located differential relay physics: In real AO instruments, non-common
 *     path aberrations (`ncpa_refl`, `ncpa_trans`), cold stops (`pup_refl`,
 *     `pup_trans`), and pupil misregistrations (`dx, dy, rot, scale`) arise
 *     immediately downstream of the WFS/science pickoff. Folding them into
 *     `cacaosim_beam_relay_cpu()` inside `beamsplit` avoids spawning extra SHM
 *     relay processes per optical arm.
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "beam/cacaosim_beamsplit_FPS.h"
#include "common/cacaosim_beam.h"

/*
 * Module-static FPS parameter backing storage for cacaosim.beamsplit.
 * Defaults configure a 0.95 um long-pass dichroic (reflecting visible < 0.95 um to WFS and
 * transmitting NIR > 0.95 um to science/coronagraph) with 95% peak arm efficiencies.
 */
static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_beam0";
static char     param_out_refl[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_beam_refl";
static char     param_out_trans[FUNCTION_PARAMETER_STRMAXLEN] = "csim_beam_trans";
static char     param_ncpa_refl[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_ncpa_tran[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_pup_refl[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char     param_pup_tran[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static uint32_t param_nwf                                     = 128;
static int32_t  param_split_mode                              = 1;
static float    param_r_gray                                  = 0.50f;
static float    param_t_gray                                  = 0.50f;
static float    param_lambda_cut                              = 0.95f;
static float    param_band_width                              = 0.25f;
static float    param_roll_um                                 = 0.03f;
static float    param_eff_refl                                = 0.95f;
static float    param_eff_trans                               = 0.95f;
static float    param_refl_dx                                 = 0.0f;
static float    param_refl_dy                                 = 0.0f;
static float    param_refl_rot                                = 0.0f;
static float    param_refl_scale                              = 1.0f;
static float    param_tran_dx                                 = 0.0f;
static float    param_tran_dy                                 = 0.0f;
static float    param_tran_rot                                = 0.0f;
static float    param_tran_scale                              = 1.0f;

/**
 * CACAOSIM_BEAMSPLIT_PARAMS(X) - X-macro FPS parameter table for `cacaosim.beamsplit`
 *
 * Parameters:
 *   - `.in_beam`     (STREAMNAME, trigger): Upstream `[nwf x nwf x 2]` `FLOAT32` optical beam.
 *   - `.out_refl`    (STREAMNAME): Reflected arm `[nwf x nwf x 2]` `FLOAT32` beam (empty = skip).
 *   - `.out_trans`   (STREAMNAME): Transmitted arm `[nwf x nwf x 2]` `FLOAT32` beam (empty = skip).
 *   - `.nwf`         (UINT32): Fallback grid dimension [pixels] if `.in_beam` is auto-created.
 *   - `.split_mode`  (INT32, WRITERUN): `0` = achromatic gray, `1` = long-pass (`refl < cut`),
 *                    `2` = short-pass (`refl > cut`), `3` = band-reflect (`[cut-w/2, cut+w/2]`).
 *   - `.r_gray`      (FLOAT32, WRITERUN): Gray reflectance `[0..1]` when `.split_mode == 0`.
 *   - `.t_gray`      (FLOAT32, WRITERUN): Gray transmittance `[0..1]` when `.split_mode == 0`.
 *   - `.lambda_cut`  (FLOAT32, WRITERUN): Dichroic transition or band-center wavelength [um].
 *   - `.band_width`  (FLOAT32, WRITERUN): Full reflection bandwidth [um] when `.split_mode == 3`.
 *   - `.roll_um`     (FLOAT32): Logistic transition width (`10%..90%`-scale) [um] (default `0.03`).
 *   - `.eff_refl`    (FLOAT32, WRITERUN): Peak in-band reflectance efficiency `[0..1]`.
 *   - `.eff_trans`   (FLOAT32, WRITERUN): Peak in-band transmittance efficiency `[0..1]`.
 *   - `.ncpa_refl`   (STRING): Optional `[nwf x nwf]` `FLOAT32` static NCPA OPD [um] on refl arm.
 *   - `.ncpa_trans`  (STRING): Optional `[nwf x nwf]` `FLOAT32` static NCPA OPD [um] on trans arm.
 *   - `.pup_refl`    (STRING): Optional `[nwf x nwf]` `FLOAT32` pupil/Lyot mask on refl arm.
 *   - `.pup_trans`   (STRING): Optional `[nwf x nwf]` `FLOAT32` pupil/Lyot mask on trans arm.
 *   - `.refl_dx`, `.refl_dy`   (FLOAT32): Reflected arm lateral pupil shift [pixels].
 *   - `.refl_rot`    (FLOAT32): Reflected arm counter-clockwise pupil rotation [degrees].
 *   - `.refl_scale`  (FLOAT32): Reflected arm isotropic pupil magnification (`1.0` = unity).
 *   - `.trans_dx`, `.trans_dy` (FLOAT32): Transmitted arm lateral pupil shift [pixels].
 *   - `.trans_rot`   (FLOAT32): Transmitted arm counter-clockwise pupil rotation [degrees].
 *   - `.trans_scale` (FLOAT32): Transmitted arm isotropic pupil magnification (`1.0` = unity).
 */
#define CACAOSIM_BEAMSPLIT_PARAMS(X)                                                    \
    X(".in_beam", param_in_beam, FPTYPE_STREAMNAME, 1,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 3D [nwf x nwf x 2] optical beam stream")                                   \
    X(".out_refl", param_out_refl, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "Output reflected arm [nwf x nwf x 2] beam stream")                               \
    X(".out_trans", param_out_trans, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,        \
      "Output transmitted arm [nwf x nwf x 2] beam stream")                             \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Fallback pupil grid size [pixels] if in_beam is auto-created (default: 128)")    \
    X(".split_mode", &param_split_mode, FPTYPE_INT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "0=gray, 1=dichroic_LP (refl<cut), 2=dichroic_SP (refl>cut), 3=band_refl")        \
    X(".r_gray", &param_r_gray, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Gray reflectance [0..1] when split_mode == 0 (default: 0.50)")                   \
    X(".t_gray", &param_t_gray, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Gray transmittance [0..1] when split_mode == 0 (default: 0.50)")                 \
    X(".lambda_cut", &param_lambda_cut, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Dichroic cutoff or band center wavelength [um] (default: 0.95)")                 \
    X(".band_width", &param_band_width, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Band-reflect full width [um] when split_mode == 3 (default: 0.25)")              \
    X(".roll_um", &param_roll_um, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Dichroic transition edge width [um] (default: 0.03)")                            \
    X(".eff_refl", &param_eff_refl, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Peak reflectance efficiency [0..1] (default: 0.95)")                             \
    X(".eff_trans", &param_eff_trans, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Peak transmittance efficiency [0..1] (default: 0.95)")                           \
    X(".ncpa_refl", param_ncpa_refl, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Optional static NCPA OPD stream [um] on reflected arm")                          \
    X(".ncpa_trans", param_ncpa_tran, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,           \
      "Optional static NCPA OPD stream [um] on transmitted arm")                        \
    X(".pup_refl", param_pup_refl, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,              \
      "Optional pupil/Lyot mask stream on reflected arm")                               \
    X(".pup_trans", param_pup_tran, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,             \
      "Optional pupil/Lyot mask stream on transmitted arm")                             \
    X(".refl_dx", &param_refl_dx, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Reflected arm X shift [pixels] (default: 0)")                                    \
    X(".refl_dy", &param_refl_dy, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Reflected arm Y shift [pixels] (default: 0)")                                    \
    X(".refl_rot", &param_refl_rot, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Reflected arm rotation [deg] (default: 0)")                                      \
    X(".refl_scale", &param_refl_scale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reflected arm magnification (default: 1.0)")                                     \
    X(".trans_dx", &param_tran_dx, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,             \
      "Transmitted arm X shift [pixels] (default: 0)")                                  \
    X(".trans_dy", &param_tran_dy, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,             \
      "Transmitted arm Y shift [pixels] (default: 0)")                                  \
    X(".trans_rot", &param_tran_rot, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,           \
      "Transmitted arm rotation [deg] (default: 0)")                                    \
    X(".trans_scale", &param_tran_scale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,       \
      "Transmitted arm magnification (default: 1.0)")

MILK_PARAMS_DEFINE(cacaosim_beamsplit, CACAOSIM_BEAMSPLIT_PARAMS);

/**
 * beamsplit_confcheck() - Enforce `FPFLAG_WRITERUN` on live spectral beamsplitter parameters
 *
 * Role:
 *   FPS `confcheck` hook invoked during parameter validation so coating mode,
 *   cutoff wavelength, bandwidth, and arm efficiencies remain live-tunable while
 *   `beamsplit_compute()` is running.
 *
 * Rationale:
 *   Allows operators and automated trade-study scripts to sweep dichroic cutoffs
 *   or switch between gray and dichroic splitting in real time without restarting
 *   the optical pipeline or breaking downstream SHM semaphores.
 *
 * Return: Always `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t beamsplit_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".split_mode", ".r_gray",   ".t_gray",   ".lambda_cut",
            ".band_width", ".eff_refl", ".eff_trans"
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
 * beamsplit_compute() - Main real-time loop splitting `.in_beam` into `.out_refl` & `.out_trans`
 * @self: Pointer to `runnable_cacaosim_beamsplit` descriptor
 *
 * Role:
 *   Connects to `.in_beam` (auto-creating a `[nwf x nwf x 2]` placeholder stream
 *   if launched before upstream `mkbeam`/`dmsim`), allocates `.out_refl` and
 *   `.out_trans` with `CSIM_BM` spectral keywords, and on every trigger frame
 *   splits the 32-bin photon spectrum and relays `[OPD_um, Amplitude]` through
 *   each arm's NCPA, pupil stop, and 2D affine registration transform.
 *
 * Rationale:
 *   - Uses `cacaosim_reg_is_identity()` fast-path inside `cacaosim_beam_relay_cpu()`
 *     when an arm's registration is `(0, 0, 0, 1)`, reducing aligned relay to a
 *     streaming OpenMP SIMD pass, while seamlessly switching to bilinear 2D
 *     inverse-mapped interpolation when sub-pixel misregistration is configured.
 *   - Supplies a default `m_V = 8.0` G2V fallback spectrum if `.in_beam` lacks
 *     `CSIM_BM` keywords so standalone tests with raw OPD cubes still yield
 *     physically sensible downstream photon fluxes.
 *
 * Return: `RETURN_SUCCESS` on clean termination, or `RETURN_FAILURE` if `.in_beam`
 *         cannot be connected or created.
 */
static MILK_HOT errno_t beamsplit_compute(const MILK_RUNNABLE *self)
{
    /* Connect to upstream beam or create a placeholder cube to avoid startup ordering races */
    IMGID img_in = stream_connect(param_in_beam);
    if (img_in.ID == -1)
    {
        uint32_t fb_nwf = (param_nwf >= 8) ? param_nwf : 128;
        img_in          = stream_connect_create_3Df32(param_in_beam, fb_nwf, fb_nwf, 2);
        if (img_in.ID == -1)
        {
            return RETURN_FAILURE;
        }
        cacaosim_beam_shm_init_kw(img_in.im);
    }

    uint32_t nwf  = img_in.md->size[0];
    uint64_t npix = (uint64_t) nwf * nwf;

    /* Create reflected and transmitted output beam cubes (either arm may be omitted if empty) */
    IMGID img_refl  = { 0 };
    IMGID img_trans = { 0 };
    img_refl.ID     = -1;
    img_trans.ID    = -1;

    if (param_out_refl[0] != '\0')
    {
        img_refl = stream_connect_create_3Df32(param_out_refl, nwf, nwf, 2);
        if (img_refl.ID != -1)
        {
            cacaosim_beam_shm_init_kw(img_refl.im);
        }
    }
    if (param_out_trans[0] != '\0')
    {
        img_trans = stream_connect_create_3Df32(param_out_trans, nwf, nwf, 2);
        if (img_trans.ID != -1)
        {
            cacaosim_beam_shm_init_kw(img_trans.im);
        }
    }

    /* Connect optional per-arm static NCPA OPD [um] and pupil/Lyot amplitude mask streams */
    IMGID img_nr = { 0 };
    IMGID img_nt = { 0 };
    IMGID img_pr = { 0 };
    IMGID img_pt = { 0 };
    img_nr.ID    = -1;
    img_nt.ID    = -1;
    img_pr.ID    = -1;
    img_pt.ID    = -1;

    if (param_ncpa_refl[0] != '\0')
    {
        img_nr = stream_connect(param_ncpa_refl);
    }
    if (param_ncpa_tran[0] != '\0')
    {
        img_nt = stream_connect(param_ncpa_tran);
    }
    if (param_pup_refl[0] != '\0')
    {
        img_pr = stream_connect(param_pup_refl);
    }
    if (param_pup_tran[0] != '\0')
    {
        img_pt = stream_connect(param_pup_tran);
    }

    /* Precompute fallback G2V m_V = 8.0 spectrum for raw upstream cubes without CSIM_BM kw */
    cacaosim_spec_t fallback_spec;
    cacaosim_spec_star(&fallback_spec, 8.2f, 0.30f, 8.0f, 5778.0f, 0.75f);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        const float *in_opd = img_in.im->array.F;
        const float *in_amp = (img_in.md->naxis == 3 && img_in.md->size[2] >= 2)
                                  ? (img_in.im->array.F + npix)
                                  : NULL;

        cacaosim_spec_t in_spec;
        if (!cacaosim_beam_shm_read_spec(img_in.im, &in_spec))
        {
            in_spec = fallback_spec;
        }

        /* Partition 32-bin photon spectrum between reflected and transmitted arms */
        cacaosim_split_cfg_t scfg = {
            .mode       = param_split_mode,
            .r_gray     = param_r_gray,
            .t_gray     = param_t_gray,
            .lambda_cut = param_lambda_cut,
            .band_width = param_band_width,
            .roll_um    = param_roll_um,
            .eff_refl   = param_eff_refl,
            .eff_trans  = param_eff_trans,
        };

        cacaosim_spec_t spec_r, spec_t;
        cacaosim_spec_split(&in_spec, &scfg, &spec_r, &spec_t);

        const float *nr = (img_nr.ID != -1 && img_nr.md->size[0] == nwf) ? img_nr.im->array.F
                                                                         : NULL;
        const float *nt = (img_nt.ID != -1 && img_nt.md->size[0] == nwf) ? img_nt.im->array.F
                                                                         : NULL;
        const float *pr = (img_pr.ID != -1 && img_pr.md->size[0] == nwf) ? img_pr.im->array.F
                                                                         : NULL;
        const float *pt = (img_pt.ID != -1 && img_pt.md->size[0] == nwf) ? img_pt.im->array.F
                                                                         : NULL;

        /* Relay reflected arm: 2D affine registration + reflected NCPA + reflected pupil stop */
        if (img_refl.ID != -1)
        {
            cacaosim_reg_t reg_r = {
                .shift_x = param_refl_dx,
                .shift_y = param_refl_dy,
                .rot_deg = param_refl_rot,
                .scale   = param_refl_scale,
            };
            img_refl.md->write = 1;
            cacaosim_beam_relay_cpu(in_opd, in_amp, nr, pr, &reg_r, nwf, img_refl.im->array.F,
                                    img_refl.im->array.F + npix);
            cacaosim_beam_shm_write_spec(img_refl.im, &spec_r);
            processinfo_update_output_stream(processinfo, img_refl.im, img_in.im);
        }

        /* Relay transmitted arm: 2D affine registration + trans NCPA + trans pupil stop */
        if (img_trans.ID != -1)
        {
            cacaosim_reg_t reg_t = {
                .shift_x = param_tran_dx,
                .shift_y = param_tran_dy,
                .rot_deg = param_tran_rot,
                .scale   = param_tran_scale,
            };
            img_trans.md->write = 1;
            cacaosim_beam_relay_cpu(in_opd, in_amp, nt, pt, &reg_t, nwf, img_trans.im->array.F,
                                    img_trans.im->array.F + npix);
            cacaosim_beam_shm_write_spec(img_trans.im, &spec_t);
            processinfo_update_output_stream(processinfo, img_trans.im, img_in.im);
        }
    }
    MILK_PROCINFO_END(self)

    if (img_pt.ID != -1)
    {
        imgid_free(&img_pt);
    }
    if (img_pr.ID != -1)
    {
        imgid_free(&img_pr);
    }
    if (img_nt.ID != -1)
    {
        imgid_free(&img_nt);
    }
    if (img_nr.ID != -1)
    {
        imgid_free(&img_nr);
    }
    if (img_trans.ID != -1)
    {
        imgid_free(&img_trans);
    }
    if (img_refl.ID != -1)
    {
        imgid_free(&img_refl);
    }
    imgid_free(&img_in);
    return RETURN_SUCCESS;
}

/* Public MILK_RUNNABLE registration and standalone milk-fpsexec-cacaosim-beamsplit export */
const MILK_RUNNABLE runnable_cacaosim_beamsplit = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_beamsplit",
              .cmdkey      = "beamsplit",
              .description = "Split optical beam via dichroic or gray beamsplitter into two arms" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_beamsplit),
    .compute   = beamsplit_compute,
    .confcheck = beamsplit_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-beamsplit", runnable_cacaosim_beamsplit);
