// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfe_FPS.c
 * @brief   MILK_RUNNABLE inline static wavefront error optical element specified in OPD [meter]
 *
 * Architectural Role:
 *   Implements `cacaosim.wfe` (`milk-fpsexec-cacaosim-wfe`, bench element
 *   `[wfe.<id>]`), an inline optical aberration and pupil-registration relay.
 *   Reads a persistent 2D `FLOAT32` wavefront error map `.in_wfe` stored in SI
 *   meters `[m]` (auto-creating and populating it via `cacaosim_make_wfe_m()` if
 *   absent), adds `1e6 * wfe_gain * in_wfe` `[um]` across all `[OPD_c, Amp_c]`
 *   wavelength channels of `.in_beam`, scales the 32-bin photon spectrum by
 *   `.tput`, and writes `.out_beam`.
 *
 * Design Rationale & Motivation:
 *   - SI `[meter]` SHM convention with `[nm]` user ergonomics: Storing `.in_wfe`
 *     in meters (`BUNIT = "m"`, `WFE_RMSM`) makes the SHM map directly compatible
 *     with CACAO NCPA/calibration tools that inject or sense meter-scale OPD
 *     offsets, while exposing `.rms_nm` in nanometers (`50 nm -> 5e-8 m`) keeps
 *     interactive configuration intuitive and converting via `1.0e6f * wfe_gain`
 *     fuses unit scaling and live gain modulation into one scalar multiplier.
 *   - One-shot `.init_only` setup mode: When `.init_only` is enabled, `wfe`
 *     creates, populates, and tags `.in_wfe` in SHM and exits immediately without
 *     entering a real-time loop, allowing `cacaosim-bench` and `cacaosim.htsim`
 *     to initialize persistent WFE maps cleanly at bench startup.
 *   - Polychromatic cube transparency: Iterating over `nlam = size[2] / 2` slice
 *     pairs and forwarding `CSIM_NWL`/`CSIM_WL0`/`CSIM_WL1` keywords allows
 *     `cacaosim.wfe` to be spliced either upstream or downstream of polychromatic
 *     coronagraph relays (`cacaosim.coro`).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "beam/cacaosim_wfe_FPS.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_disp.h"
#include "pupil/cacaosim_pupil.h"

/*
 * Module-static FPS parameter backing storage for cacaosim.wfe.
 * Defaults synthesize a 50 nm RMS (5.0e-8 m) Zernike + k^-1.6 polishing-ripple static WFE map
 * with 98% optical surface throughput and identity 2D pupil registration.
 */
static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_beam0";
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN] = "csim_beam_wfe";
static char     param_in_wfe[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_wfe";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static uint32_t param_nwf                                    = 128;
static float    param_pup_diam                               = 0.9375f;
static float    param_rms_nm                                 = 50.0f;
static float    param_defocus_nm                             = 0.0f;
static float    param_psd_alpha                              = 1.6f;
static uint64_t param_wfe_seed                               = 101;
static float    param_wfe_gain                               = 1.0f;
static float    param_tput                                   = 0.98f;
static float    param_shift_x                                = 0.0f;
static float    param_shift_y                                = 0.0f;
static float    param_rot_deg                                = 0.0f;
static float    param_mag_scale                              = 1.0f;
static int64_t  param_init_only                              = 0;

/**
 * CACAOSIM_WFE_PARAMS(X) - X-macro FPS parameter table for `cacaosim.wfe`
 *
 * Parameters:
 *   - `.in_beam`   (STREAMNAME, trigger): Upstream `[nwf x nwf x 2*N]` `FLOAT32` optical beam.
 *   - `.out_beam`  (STREAMNAME, required): Downstream `[nwf x nwf x 2*N]` `FLOAT32` optical beam.
 *   - `.in_wfe`    (STREAMNAME, required): 2D `[nwf x nwf]` `FLOAT32` static WFE OPD map in SI
 *                  `[meter]`; auto-created and synthesized if absent or dimension-mismatched.
 *   - `.in_pup`    (STRING): Optional 2D pupil mask stream weighting initial `.in_wfe` RMS.
 *   - `.nwf`       (UINT32): Fallback grid size [pixels] when creating `.in_wfe` or `.in_beam`.
 *   - `.pup_diam`  (FLOAT32): Clear pupil diameter fraction of `nwf` for initial WFE synthesis.
 *   - `.rms_nm`    (FLOAT32): Target initial pupil-weighted WFE RMS [nm] (`1 nm = 1e-9 m`).
 *   - `.psd_alpha` (FLOAT32): Mid-spatial-frequency amplitude slope `k^-alpha` (default `1.6`).
 *   - `.wfe_seed`  (UINT64): Deterministic PRNG seed for initial `.in_wfe` synthesis.
 *   - `.wfe_gain`  (FLOAT32, WRITERUN): Live dimensionless gain multiplying `.in_wfe` (`0` = off,
 *                  `-1` = conjugate sign flip).
 *   - `.tput`      (FLOAT32, WRITERUN): Optical power transmission/reflectivity `[0..1]`.
 *   - `.shift_x`, `.shift_y` (FLOAT32, WRITERUN): Lateral beam shift before WFE addition [pixels].
 *   - `.rot_deg`   (FLOAT32, WRITERUN): Counter-clockwise beam rotation [degrees].
 *   - `.mag_scale` (FLOAT32, WRITERUN): Isotropic beam magnification (`1.0` = unity).
 *   - `.init_only` (ONOFF): If ON, create & populate `.in_wfe` `[meter]` in SHM and exit cleanly.
 */
#define CACAOSIM_WFE_PARAMS(X)                                                          \
    X(".in_beam", param_in_beam, FPTYPE_STREAMNAME, 1,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 3D [nwf x nwf x 2*N] optical beam stream")                                 \
    X(".out_beam", param_out_beam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "Output 3D [nwf x nwf x 2*N] optical beam stream")                                \
    X(".in_wfe", param_in_wfe, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,              \
      "2D static wavefront error OPD stream in unit of [meter]")                        \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional 2D pupil stream for initial WFE RMS normalization")                     \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Grid size [pixels] when creating .in_wfe / fallback beam (default: 128)")        \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Pupil diameter fraction of nwf for initial WFE map (default: 0.9375)")           \
    X(".rms_nm", &param_rms_nm, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Initial WFE pupil RMS [nm] (50 nm = 5e-8 m) when creating .in_wfe")              \
    X(".defocus_nm", &param_defocus_nm, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Additive smooth Zernike Z4 defocus RMS [nm] across full grid (default: 0.0)")    \
    X(".psd_alpha", &param_psd_alpha, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Spatial PSD amplitude slope k^-alpha for initial WFE map (default: 1.6)")        \
    X(".wfe_seed", &param_wfe_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,             \
      "Deterministic PRNG seed for initial WFE map synthesis (default: 101)")           \
    X(".wfe_gain", &param_wfe_gain, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Dimensionless multiplier applied to .in_wfe [meter] (default: 1.0)")             \
    X(".tput", &param_tput, FPTYPE_FLOAT32, 0,                                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Optical power throughput [0..1] (default: 0.98)")                                \
    X(".shift_x", &param_shift_x, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Pupil lateral X shift [pixels] (default: 0.0)")                                  \
    X(".shift_y", &param_shift_y, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Pupil lateral Y shift [pixels] (default: 0.0)")                                  \
    X(".rot_deg", &param_rot_deg, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Pupil rotation [deg] (default: 0.0)")                                            \
    X(".mag_scale", &param_mag_scale, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Pupil magnification factor (default: 1.0)")                                      \
    X(".init_only", &param_init_only, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,            \
      "ON: create & initialize .in_wfe [meter] stream map and exit immediately")

MILK_PARAMS_DEFINE(cacaosim_wfe, CACAOSIM_WFE_PARAMS);

/**
 * wfe_write_unit_kw() - Tag a 2D WFE SHM image with SI meter `BUNIT` and `WFE_RMSM` keywords
 * @im:    Target `ImageStreamIO` 2D image pointer (gracefully ignored if NULL or `NBkw < 1`)
 * @rms_m: Initial synthesized pupil-weighted RMS OPD in SI meters `[m]`
 *
 * Role:
 *   Cold-path SHM metadata helper called by `wfe_populate_stream()` immediately
 *   after synthesizing `.in_wfe`.
 *
 * Rationale:
 *   Recording `BUNIT = "m"` and `WFE_RMSM` directly in SHM FITS-style keywords
 *   makes the physical unit self-describing for `cacaosim-CTRL`, FITS archivers,
 *   and external CACAO calibration scripts inspecting the stream.
 */
static void wfe_write_unit_kw(IMAGE *im, float rms_m)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < 1)
    {
        return;
    }
    snprintf(im->kw[0].name, KEYWORD_MAX_STRING, "BUNIT");
    im->kw[0].type = 'S';
    snprintf(im->kw[0].value.valstr, KEYWORD_MAX_STRING, "m");
    snprintf(im->kw[0].comment, KEYWORD_MAX_COMMENT, "Static wavefront error OPD [meter]");

    if (im->md->NBkw >= 2)
    {
        snprintf(im->kw[1].name, KEYWORD_MAX_STRING, "WFE_RMSM");
        im->kw[1].type       = 'D';
        im->kw[1].value.numf = (double) rms_m;
        snprintf(im->kw[1].comment, KEYWORD_MAX_COMMENT, "Initial pupil RMS OPD [meter]");
    }
}

/**
 * wfe_populate_stream() - Synthesize and publish initial static WFE `[meter]` map into SHM
 * @img_wfe: Connected 2D `[nwf x nwf]` `FLOAT32` SHM stream descriptor to populate
 * @nwf:     Square grid dimension [pixels]
 *
 * Role:
 *   Cold-path initializer invoked either in `.init_only` one-shot mode or when
 *   `wfe_compute()` creates a missing/mismatched `.in_wfe` stream at startup.
 *
 * Rationale:
 *   Optionally attaches to `.in_pup` so initial RMS normalization accounts for
 *   the exact telescope central obstruction and spider vanes when available,
 *   converts `.rms_nm` (`[nm]`) to SI `[m]` (`* 1.0e-9f`), synthesizes the map
 *   via `cacaosim_make_wfe_m()`, writes `BUNIT` keywords, and posts SHM update
 *   notifications via `ImageStreamIO_UpdateIm()`.
 */
static void wfe_populate_stream(IMGID *img_wfe, uint32_t nwf)
{
    const float *pup_ptr = NULL;
    IMGID        img_pup = { 0 };
    img_pup.ID           = -1;

    if (param_in_pup[0] != '\0')
    {
        img_pup = stream_connect(param_in_pup);
        if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT &&
            img_pup.md->size[0] == nwf && img_pup.md->size[1] == nwf)
        {
            pup_ptr = img_pup.im->array.F;
        }
    }

    float rms_m        = (param_rms_nm > 0.0f) ? (param_rms_nm * 1.0e-9f) : 0.0f;
    float defoc_m      = param_defocus_nm * 1.0e-9f;
    img_wfe->md->write = 1;
    cacaosim_make_wfe_m(nwf, param_pup_diam, pup_ptr, rms_m, param_psd_alpha, param_wfe_seed,
                        img_wfe->im->array.F);
    if (fabsf(defoc_m) > 1e-15f)
    {
        cacaosim_add_defocus_m(nwf, param_pup_diam, defoc_m, img_wfe->im->array.F);
    }
    wfe_write_unit_kw(img_wfe->im, hypotf(rms_m, defoc_m));
    ImageStreamIO_UpdateIm(img_wfe->im);

    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
}

/**
 * wfe_confcheck() - Enforce `FPFLAG_WRITERUN` on live WFE gain, throughput, & registration keys
 *
 * Role:
 *   FPS `confcheck` callback ensuring `.wfe_gain`, `.tput`, `.shift_x`,
 *   `.shift_y`, `.rot_deg`, and `.mag_scale` remain modifiable during `RUN`.
 *
 * Rationale:
 *   Permits live injection/removal of static aberrations (`wfe_gain`) and dynamic
 *   pupil shear/rotation/magnification drift studies during closed-loop AO tests
 *   without stopping or re-initializing the optical bench.
 *
 * Return: Always `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t wfe_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".wfe_gain", ".tput", ".shift_x", ".shift_y", ".rot_deg", ".mag_scale"
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
 * wfe_compute() - One-shot `.in_wfe` initializer or real-time inline WFE optical relay loop
 * @self: Pointer to `runnable_cacaosim_wfe` descriptor
 *
 * Role:
 *   When `.init_only` is ON, allocates and populates `.in_wfe` `[meter]` in SHM
 *   and returns immediately. Otherwise, connects `.in_beam`, `.in_wfe` (preserving
 *   any user/CACAO-supplied `[meter]` map already matching `[nwf x nwf]`), and
 *   `.out_beam` (`[nwf x nwf x 2*N]`), applying 2D affine registration, meter-to-um
 *   scaled WFE addition, and spectral throughput scaling on every trigger frame.
 *
 * Rationale:
 *   - Preserves pre-existing `.in_wfe` SHM streams of matching `FLOAT32` `[nwf x nwf]`
 *     shape so external FITS loaders or CACAO phase-diversity loops can overwrite
 *     `.in_wfe` arbitrarily before or during execution.
 *   - Loops across all `nlam` `[OPD_c, Amp_c]` channel pairs and forwards
 *     `CSIM_NWL`/`CSIM_WL0`/`CSIM_WL1` keywords so polychromatic coronagraphic
 *     beams pass through `cacaosim.wfe` without losing spectral channel metadata.
 *
 * Return: `RETURN_SUCCESS` on completion, or `RETURN_FAILURE` on invalid stream
 *         names or SHM allocation failure.
 */
static MILK_HOT errno_t wfe_compute(const MILK_RUNNABLE *self)
{
    if (param_in_wfe[0] == '\0')
    {
        return RETURN_FAILURE;
    }

    /* One-shot setup mode: materialize the persistent .in_wfe [meter] SHM stream and exit */
    if (param_init_only)
    {
        uint32_t nwf = (param_nwf >= 8u) ? param_nwf : 128u;
        IMGID    iw  = stream_connect_create_2Df32(param_in_wfe, nwf, nwf);
        if (iw.ID == -1)
        {
            return RETURN_FAILURE;
        }
        wfe_populate_stream(&iw, nwf);
        imgid_free(&iw);
        return RETURN_SUCCESS;
    }

    /* Connect upstream beam (or create a 2-slice fallback cube to avoid startup ordering races) */
    IMGID img_in = stream_connect(param_in_beam);
    if (img_in.ID == -1)
    {
        imgid_free(&img_in);
        uint32_t fb_nwf = (param_nwf >= 8u) ? param_nwf : 128u;
        img_in          = cacaosim_beam_shm_create(param_in_beam, fb_nwf, 2u);
        if (img_in.ID == -1)
        {
            return RETURN_FAILURE;
        }
        cacaosim_beam_shm_init_kw(img_in.im);
    }

    uint32_t nwf   = img_in.md->size[0];
    uint64_t npix  = (uint64_t) nwf * nwf;
    uint32_t in_nz = (img_in.md->naxis == 3 && img_in.md->size[2] >= 2u &&
                      (img_in.md->size[2] % 2u) == 0u)
                         ? img_in.md->size[2]
                         : 2u;

    /* Reuse existing [nwf x nwf] FLOAT32 .in_wfe [meter] stream if valid, else synthesize it */
    IMGID img_wfe = stream_connect(param_in_wfe);
    if (img_wfe.ID == -1 || img_wfe.md->datatype != _DATATYPE_FLOAT ||
        img_wfe.md->size[0] != nwf || img_wfe.md->size[1] != nwf)
    {
        if (img_wfe.ID != -1)
        {
            imgid_free(&img_wfe);
        }
        img_wfe = stream_connect_create_2Df32(param_in_wfe, nwf, nwf);
        if (img_wfe.ID == -1)
        {
            imgid_free(&img_in);
            return RETURN_FAILURE;
        }
        wfe_populate_stream(&img_wfe, nwf);
    }

    /* Match output beam Z depth (2 * nlam) with NBkw = 80 for dispersion metadata */
    IMGID img_out = cacaosim_beam_shm_create(param_out_beam, nwf, in_nz);
    if (img_out.ID == -1)
    {
        imgid_free(&img_wfe);
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }
    cacaosim_beam_shm_init_kw(img_out.im);

    cacaosim_spec_t fallback_spec;
    cacaosim_spec_star(&fallback_spec, 8.2f, 0.30f, 8.0f, 5778.0f, 0.75f);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        /* Determine active coherent wavelength channel count clamped to output slice capacity */
        uint32_t nlam = (img_in.md->naxis == 3 && img_in.md->size[2] >= 2u &&
                         (img_in.md->size[2] % 2u) == 0u)
                            ? (img_in.md->size[2] / 2u)
                            : 1u;
        if (img_out.md->size[2] < 2u * nlam)
        {
            nlam = img_out.md->size[2] / 2u;
            if (nlam < 1u)
            {
                nlam = 1u;
            }
        }

        /* Attenuate 32-bin photon arrival rate spectrum by optical element throughput */
        cacaosim_spec_t in_spec;
        if (!cacaosim_beam_shm_read_spec(img_in.im, &in_spec))
        {
            in_spec = fallback_spec;
        }
        cacaosim_spec_t out_spec;
        cacaosim_spec_scale(&in_spec, param_tput, &out_spec);

        /* Convert SI [meter] WFE map to beam [um] with live gain: 1 m = 1e6 um */
        cacaosim_reg_t reg = {
            .shift_x = param_shift_x,
            .shift_y = param_shift_y,
            .rot_deg = param_rot_deg,
            .scale   = (param_mag_scale > 1e-4f) ? param_mag_scale : 1.0f,
        };
        const float wfe_to_um = 1.0e6f * param_wfe_gain;

        /* Apply registration and achromatic OPD error across every [OPD_c, Amp_c] channel */
        img_out.md->write = 1;
        for (uint32_t c = 0; c < nlam; c++)
        {
            uint64_t     off    = (uint64_t) c * 2u * npix;
            const float *in_opd = img_in.im->array.F + off;
            const float *in_amp = (img_in.md->naxis == 3 && img_in.md->size[2] >= 2u)
                                      ? (in_opd + npix)
                                      : NULL;
            float       *out_o  = img_out.im->array.F + off;
            float       *out_a  = out_o + npix;

            cacaosim_beam_apply_wfe_cpu(in_opd, in_amp, img_wfe.im->array.F, wfe_to_um, &reg, nwf,
                                        out_o, out_a);
        }

        /* Propagate 32-bin spectrum, dispersion, and optional multi-wavelength metadata */
        cacaosim_beam_shm_write_spec(img_out.im, &out_spec);
        cacaosim_disp_t in_disp, out_disp;
        memset(&in_disp, 0, sizeof(in_disp));
        memset(&out_disp, 0, sizeof(out_disp));
        cacaosim_beam_shm_read_disp(img_in.im, &in_disp);
        cacaosim_disp_apply_reg(&in_disp, &reg, &out_disp);
        cacaosim_beam_shm_write_disp(img_out.im, &out_disp);
        float in_diam = cacaosim_beam_shm_read_diam(img_in.im, 0.0f);
        if (in_diam > 0.0f)
        {
            cacaosim_beam_shm_write_diam(img_out.im, in_diam);
        }
        uint32_t nwl_kw = 1u;
        float    wl0_kw = 0.0f;
        float    wl1_kw = 0.0f;
        if (cacaosim_beam_shm_read_poly_kw(img_in.im, &nwl_kw, &wl0_kw, &wl1_kw))
        {
            cacaosim_beam_shm_write_poly_kw(img_out.im, nwl_kw, wl0_kw, wl1_kw);
            cacaosim_beam_shm_write_poly_mode(
                img_out.im, cacaosim_beam_shm_read_poly_mode(img_in.im, nwl_kw));
            float wls_kw[8];
            if (nwl_kw <= 8u && cacaosim_beam_shm_read_poly_wls(img_in.im, nwl_kw, wls_kw))
            {
                cacaosim_beam_shm_write_poly_wls(img_out.im, wls_kw, nwl_kw);
            }
        }
        processinfo_update_output_stream(processinfo, img_out.im, img_in.im);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&img_out);
    imgid_free(&img_wfe);
    imgid_free(&img_in);
    return RETURN_SUCCESS;
}

/* Public MILK_RUNNABLE registration and standalone milk-fpsexec-cacaosim-wfe export */
const MILK_RUNNABLE runnable_cacaosim_wfe = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_wfe",
              .cmdkey      = "wfe",
              .description = "Inline static wavefront error optical element specified in OPD [m]" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_wfe),
    .compute   = wfe_compute,
    .confcheck = wfe_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-wfe", runnable_cacaosim_wfe);
