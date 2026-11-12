// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_FPS.c
 * @brief   MILK_RUNNABLE compute unit for coronagraphic pupil/focal masks, Lyot, & LOWFS pickoff
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_mask.h"
#include "coro/cacaosim_coro.h"
#include "coro/cacaosim_coro_FPS.h"

static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]    = "csim_beam0";
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_beam_coro";
static char     param_out_lowfs[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char     param_in_pup_amp[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_pup_pha[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_fpm_amp[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_fpm_pha[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_lyot[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char     param_in_lyot_pha[FUNCTION_PARAMETER_STRMAXLEN] = "";
static uint32_t param_nwf                                      = 128;
static int32_t  param_coro_mode                                = 0;
static uint32_t param_n_mft                                    = 48;
static uint32_t param_pad_factor                               = 4;
static float    param_pup_diam                                 = 0.9375f;
static float    param_lambda_um                                = 1.65f;
static float    param_bw_frac                                  = 0.0f;
static int32_t  param_nlambda                                  = 1;
static float    param_fpm_rad_ld                               = 3.0f;
static float    param_fpm_amp                                  = 0.0f;
static float    param_fpm_phase                                = 3.14159265f;
static float    param_fpm_pixscale                             = 8.0f;
static int32_t  param_pha_unit                                 = 0;
static int32_t  param_vortex_l                                 = 2;
static int32_t  param_lowfs_mode                               = 0;
static float    param_lowfs_pup_diam                           = 1.0f;
static float    param_lowfs_refl                               = 0.95f;
static float    param_tput                                     = 0.92f;
static float    param_lyot_outer                               = 0.85f;
static float    param_lyot_obs                                 = 0.36f;
static int32_t  param_lyot_nspider                             = 4;
static float    param_lyot_spid_w                              = 0.02f;
static float    param_lyot_spid_deg                            = 45.0f;
static int32_t  param_backend                                  = 0;

#define CACAOSIM_CORO_PARAMS(X)                                                         \
    X(".in_beam", param_in_beam, FPTYPE_STREAMNAME, 1,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 3D [nwf x nwf x 2*N] optical beam stream")                                 \
    X(".out_beam", param_out_beam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "Output post-Lyot [nwf x nwf x 2*Nlam] coronagraphic beam stream")                \
    X(".out_lowfs", param_out_lowfs, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Optional FPM-reflected [nwf x nwf x 2] LOWFS pupil beam stream")                 \
    X(".in_pup_amp", param_in_pup_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional custom pre-FPM pupil amplitude mask (FITS or SHM, 2D/3D)")              \
    X(".in_pup_pha", param_in_pup_pha, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional custom pre-FPM pupil phase/OPD mask (FITS or SHM, 2D/3D)")              \
    X(".in_fpm_amp", param_in_fpm_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional custom focal-plane amplitude mask (FITS or SHM, 2D/3D)")                \
    X(".in_fpm_pha", param_in_fpm_pha, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional custom focal-plane phase/OPD mask (FITS or SHM, 2D/3D)")                \
    X(".in_lyot", param_in_lyot, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional custom Lyot stop amplitude mask (FITS or SHM, 2D/3D)")                  \
    X(".in_lyot_pha", param_in_lyot_pha, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,        \
      "Optional custom Lyot stop phase/OPD mask (FITS or SHM, 2D/3D)")                  \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Fallback pupil grid size [pixels] (default: 128)")                               \
    X(".coro_mode", &param_coro_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "0=Lyot, 1=dimple, 2=vortex, 3=4QPM, 4=custom MFT, 5=custom FFT")                 \
    X(".n_mft", &param_n_mft, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                   \
      "Babinet MFT core resolution for modes 0..1 (default: 48)")                       \
    X(".pad_factor", &param_pad_factor, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "2D FFT padding factor for modes 2, 3, 5 (>=2, default: 4)")                      \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Entrance pupil diameter fraction of nwf (default: 0.9375)")                      \
    X(".lambda_um", &param_lambda_um, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Coronagraph central wavelength [um] (default: 1.65)")                            \
    X(".bw_frac", &param_bw_frac, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Fractional bandpass dlambda/lambda0 (default: 0.0, 0.20 for 3D masks)")          \
    X(".nlambda", &param_nlambda, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Polychromatic channels (0/1=auto from 3D masks, >1=explicit)")                   \
    X(".fpm_rad_ld", &param_fpm_rad_ld, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Focal mask radius [lambda0/D] for modes 0..1 (default: 3.0)")                    \
    X(".fpm_amp", &param_fpm_amp, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Core amplitude transmission [0..1] (0 = opaque spot)")                           \
    X(".fpm_phase", &param_fpm_phase, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Core phase shift [rad] for mode 1 (default: pi)")                                \
    X(".fpm_pixscale", &param_fpm_pixscale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,    \
      "Custom MFT focal sampling [pixels per lambda0/D] (default: 8.0)")                \
    X(".pha_unit", &param_pha_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Custom *_pha mask unit: 0 = radians, 1 = OPD [um] (default: 0)")                 \
    X(".vortex_charge", &param_vortex_l, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "Optical vortex topological charge (2 or 4, default: 2)")                         \
    X(".lowfs_mode", &param_lowfs_mode, FPTYPE_INT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "LOWFS pickoff: 0=fpm_refl (FPM absorbed), 1=lyot_refl (Lyot stop reflected)")    \
    X(".lowfs_pup_diam", &param_lowfs_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,\
      "Pupil aperture size on FPM->LOWFS path [D_pup] (default: 1.0)")                  \
    X(".lowfs_refl", &param_lowfs_refl, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Amplitude reflectivity into out_lowfs port (default: 0.95)")                     \
    X(".tput", &param_tput, FPTYPE_FLOAT32, 0,                                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Coronagraph relay optical throughput [0..1] (default: 0.92)")                    \
    X(".lyot_outer", &param_lyot_outer, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Lyot stop outer diameter fraction of pup_diam (default: 0.85)")                  \
    X(".lyot_obs", &param_lyot_obs, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Lyot stop central obscuration fraction of pup_diam (default: 0.36)")             \
    X(".lyot_nspider", &param_lyot_nspider, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "Number of Lyot stop spider vanes (default: 4)")                                  \
    X(".lyot_spider_w", &param_lyot_spid_w, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,    \
      "Lyot spider width fraction of pup_diam (default: 0.02)")                         \
    X(".lyot_spider_deg", &param_lyot_spid_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,\
      "Lyot spider orientation [deg] (default: 45.0)")                                  \
    X(".backend", &param_backend, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Compute backend: 0=auto, 1=CPU, 2=CUDA")

MILK_PARAMS_DEFINE(cacaosim_coro, CACAOSIM_CORO_PARAMS);

static void coro_free_masks(cacaosim_coro_masks_t *m)
{
    cacaosim_mask_free(&m->pup_amp);
    cacaosim_mask_free(&m->pup_pha);
    cacaosim_mask_free(&m->fpm_amp);
    cacaosim_mask_free(&m->fpm_pha);
    cacaosim_mask_free(&m->lyot_amp);
    cacaosim_mask_free(&m->lyot_pha);
}

static int coro_load_opt_mask(const char *spec, cacaosim_mask_cube_t *out)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 0;
    }
    char err[256] = { 0 };
    if (cacaosim_mask_load(spec, 1, out, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "cacaosim.coro: failed to load mask '%s': %s\n", spec, err);
        return -1;
    }
    return 0;
}

static MILK_COLD errno_t coro_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = { ".lowfs_mode", ".lowfs_refl", ".tput" };
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

static MILK_HOT errno_t coro_compute(const MILK_RUNNABLE *self)
{
    IMGID img_in = stream_connect(param_in_beam);
    if (img_in.ID == -1)
    {
        imgid_free(&img_in);
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

    cacaosim_coro_masks_t masks = { 0 };
    if (coro_load_opt_mask(param_in_pup_amp, &masks.pup_amp) != 0 ||
        coro_load_opt_mask(param_in_pup_pha, &masks.pup_pha) != 0 ||
        coro_load_opt_mask(param_in_fpm_amp, &masks.fpm_amp) != 0 ||
        coro_load_opt_mask(param_in_fpm_pha, &masks.fpm_pha) != 0 ||
        coro_load_opt_mask(param_in_lyot, &masks.lyot_amp) != 0 ||
        coro_load_opt_mask(param_in_lyot_pha, &masks.lyot_pha) != 0)
    {
        coro_free_masks(&masks);
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }

    cacaosim_coro_cfg_t cfg = {
        .nwf             = nwf,
        .mode            = param_coro_mode,
        .n_mft           = param_n_mft,
        .pad_factor      = param_pad_factor,
        .pup_diam        = param_pup_diam,
        .lambda_um       = param_lambda_um,
        .fpm_rad_ld      = param_fpm_rad_ld,
        .fpm_amp         = param_fpm_amp,
        .fpm_phase_rad   = param_fpm_phase,
        .vortex_charge   = param_vortex_l,
        .lowfs_refl      = param_lowfs_refl,
        .lyot_outer      = param_lyot_outer,
        .lyot_obs        = param_lyot_obs,
        .lyot_nspider    = param_lyot_nspider,
        .lyot_spider_w   = param_lyot_spid_w,
        .lyot_spider_deg = param_lyot_spid_deg,
        .backend         = param_backend,
        .fpm_pixscale_ld = param_fpm_pixscale,
        .bw_frac         = param_bw_frac,
        .nlambda         = param_nlambda,
        .mask_pha_unit   = param_pha_unit,
        .lowfs_mode      = param_lowfs_mode,
        .lowfs_pup_diam  = param_lowfs_pup_diam,
    };

    cacaosim_coro_ctx_t coro_ctx;
    int                 init_rc = cacaosim_coro_init_masks(&coro_ctx, &cfg, &masks, NULL);
    coro_free_masks(&masks);
    if (init_rc != 0)
    {
        fprintf(stderr, "cacaosim.coro: coronagraph initialization failed (check mask sizes)\n");
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }

    uint32_t out_nz  = (uint32_t) (2 * coro_ctx.eff_nlam);
    IMGID    img_out = stream_connect_create_3Df32(param_out_beam, nwf, nwf, out_nz);
    if (img_out.ID == -1)
    {
        cacaosim_coro_free(&coro_ctx);
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }
    cacaosim_beam_shm_init_kw(img_out.im);
    cacaosim_beam_shm_write_poly_kw(img_out.im, (uint32_t) coro_ctx.eff_nlam, coro_ctx.lam_um[0],
                                    coro_ctx.lam_um[coro_ctx.eff_nlam - 1]);

    IMGID img_lowfs = { 0 };
    img_lowfs.ID    = -1;
    if (param_out_lowfs[0] != '\0')
    {
        img_lowfs = stream_connect_create_3Df32(param_out_lowfs, nwf, nwf, 2);
        if (img_lowfs.ID != -1)
        {
            cacaosim_beam_shm_init_kw(img_lowfs.im);
        }
    }

    cacaosim_spec_t fallback_spec;
    cacaosim_spec_star(&fallback_spec, 8.2f, 0.30f, 8.0f, 5778.0f, 0.75f);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        const float *in_opd  = img_in.im->array.F;
        const float *in_amp  = (img_in.md->naxis == 3 && img_in.md->size[2] >= 2)
                                   ? (img_in.im->array.F + npix)
                                   : NULL;
        uint32_t     in_nlam = (img_in.md->naxis == 3 && img_in.md->size[2] >= 4 &&
                                (img_in.md->size[2] % 2u) == 0u)
                                   ? (img_in.md->size[2] / 2u)
                                   : 1u;

        cacaosim_spec_t in_spec;
        if (!cacaosim_beam_shm_read_spec(img_in.im, &in_spec))
        {
            in_spec = fallback_spec;
        }

        cacaosim_spec_t out_spec, lo_spec;
        cacaosim_spec_scale(&in_spec, param_tput, &out_spec);
        cacaosim_spec_scale(&in_spec, param_tput, &lo_spec);

        coro_ctx.cfg.lowfs_mode = param_lowfs_mode;
        coro_ctx.cfg.lowfs_refl = param_lowfs_refl;

        img_out.md->write = 1;
        if (img_lowfs.ID != -1)
        {
            img_lowfs.md->write = 1;
        }

        cacaosim_coro_step_poly(
            &coro_ctx, in_opd, in_amp, in_nlam, npix * 2u, img_out.im->array.F,
            img_out.im->array.F + npix, npix * 2u,
            (img_lowfs.ID != -1) ? img_lowfs.im->array.F : NULL,
            (img_lowfs.ID != -1) ? (img_lowfs.im->array.F + npix) : NULL, NULL, NULL, NULL, NULL,
            NULL, NULL);

        cacaosim_beam_shm_write_spec(img_out.im, &out_spec);
        cacaosim_beam_shm_write_poly_kw(img_out.im, (uint32_t) coro_ctx.eff_nlam,
                                        coro_ctx.lam_um[0],
                                        coro_ctx.lam_um[coro_ctx.eff_nlam - 1]);
        processinfo_update_output_stream(processinfo, img_out.im, img_in.im);

        if (img_lowfs.ID != -1)
        {
            cacaosim_beam_shm_write_spec(img_lowfs.im, &lo_spec);
            processinfo_update_output_stream(processinfo, img_lowfs.im, img_in.im);
        }
    }
    MILK_PROCINFO_END(self)

    cacaosim_coro_free(&coro_ctx);
    if (img_lowfs.ID != -1)
    {
        imgid_free(&img_lowfs);
    }
    imgid_free(&img_out);
    imgid_free(&img_in);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_coro = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_coro",
              .cmdkey      = "coro",
              .description = "Simulate coronagraphic focal/pupil masks, Lyot, and LOWFS pickoff" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_coro),
    .compute   = coro_compute,
    .confcheck = coro_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-coro", runnable_cacaosim_coro);
