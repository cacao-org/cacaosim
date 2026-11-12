// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_FPS.c
 * @brief   MILK_RUNNABLE compute unit for coronagraphic FPM, Lyot stop, & LOWFS core pickoff
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "coro/cacaosim_coro.h"
#include "coro/cacaosim_coro_FPS.h"

static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_beam0";
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_beam_coro";
static char     param_out_lowfs[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_lyot[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static uint32_t param_nwf                                     = 128;
static int32_t  param_coro_mode                               = 0;
static uint32_t param_n_mft                                   = 48;
static uint32_t param_pad_factor                              = 4;
static float    param_pup_diam                                = 0.9375f;
static float    param_lambda_um                               = 1.65f;
static float    param_fpm_rad_ld                              = 3.0f;
static float    param_fpm_amp                                 = 0.0f;
static float    param_fpm_phase                               = 3.14159265f;
static int32_t  param_vortex_l                                = 2;
static float    param_lowfs_refl                              = 0.95f;
static float    param_tput                                    = 0.92f;
static float    param_lyot_outer                              = 0.85f;
static float    param_lyot_obs                                = 0.36f;
static int32_t  param_lyot_nspider                            = 4;
static float    param_lyot_spid_w                             = 0.02f;
static float    param_lyot_spid_deg                           = 45.0f;
static int32_t  param_backend                                 = 0;

#define CACAOSIM_CORO_PARAMS(X)                                                         \
    X(".in_beam", param_in_beam, FPTYPE_STREAMNAME, 1,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 3D [nwf x nwf x 2] optical beam stream")                                   \
    X(".out_beam", param_out_beam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "Output post-Lyot [nwf x nwf x 2] coronagraphic beam stream")                     \
    X(".out_lowfs", param_out_lowfs, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Optional FPM-reflected [nwf x nwf x 2] LOWFS pupil beam stream")                 \
    X(".in_lyot", param_in_lyot, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional external 2D Lyot stop amplitude stream")                                \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Fallback pupil grid size [pixels] (default: 128)")                               \
    X(".coro_mode", &param_coro_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "0=Lyot spot, 1=phase dimple, 2=vortex, 3=4QPM (default: 0)")                     \
    X(".n_mft", &param_n_mft, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                   \
      "Babinet MFT core resolution for modes 0..1 (default: 48)")                       \
    X(".pad_factor", &param_pad_factor, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "2D FFT padding factor for modes 2..3 (>=2, default: 4)")                         \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Entrance pupil diameter fraction of nwf (default: 0.9375)")                      \
    X(".lambda_um", &param_lambda_um, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Coronagraph central wavelength [um] (default: 1.65)")                            \
    X(".fpm_rad_ld", &param_fpm_rad_ld, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Focal mask radius [lambda/D] for modes 0..1 (default: 3.0)")                     \
    X(".fpm_amp", &param_fpm_amp, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Core amplitude transmission [0..1] (0 = opaque spot)")                           \
    X(".fpm_phase", &param_fpm_phase, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Core phase shift [rad] for mode 1 (default: pi)")                                \
    X(".vortex_charge", &param_vortex_l, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,         \
      "Optical vortex topological charge (2 or 4, default: 2)")                         \
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

static MILK_COLD errno_t coro_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = { ".lowfs_refl", ".tput" };
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

    IMGID img_out = stream_connect_create_3Df32(param_out_beam, nwf, nwf, 2);
    if (img_out.ID == -1)
    {
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }
    cacaosim_beam_shm_init_kw(img_out.im);

    IMGID img_lowfs = { 0 };
    IMGID img_lyot  = { 0 };
    img_lowfs.ID    = -1;
    img_lyot.ID     = -1;

    if (param_out_lowfs[0] != '\0')
    {
        img_lowfs = stream_connect_create_3Df32(param_out_lowfs, nwf, nwf, 2);
        if (img_lowfs.ID != -1)
        {
            cacaosim_beam_shm_init_kw(img_lowfs.im);
        }
    }
    if (param_in_lyot[0] != '\0')
    {
        img_lyot = stream_connect(param_in_lyot);
    }

    const float *ext_lyot = (img_lyot.ID != -1 && img_lyot.md->datatype == _DATATYPE_FLOAT &&
                             img_lyot.md->size[0] == nwf)
                                ? img_lyot.im->array.F
                                : NULL;

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
    };

    cacaosim_coro_ctx_t coro_ctx;
    if (cacaosim_coro_init(&coro_ctx, &cfg, ext_lyot, NULL) != 0)
    {
        if (img_lyot.ID != -1)
        {
            imgid_free(&img_lyot);
        }
        if (img_lowfs.ID != -1)
        {
            imgid_free(&img_lowfs);
        }
        imgid_free(&img_out);
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }

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

        cacaosim_spec_t out_spec, lo_spec;
        cacaosim_spec_scale(&in_spec, param_tput, &out_spec);
        cacaosim_spec_scale(&in_spec, param_tput, &lo_spec);

        coro_ctx.cfg.lowfs_refl = param_lowfs_refl;

        img_out.md->write = 1;
        if (img_lowfs.ID != -1)
        {
            img_lowfs.md->write = 1;
        }

        cacaosim_coro_step_beam(
            &coro_ctx, in_opd, in_amp, img_out.im->array.F, img_out.im->array.F + npix,
            (img_lowfs.ID != -1) ? img_lowfs.im->array.F : NULL,
            (img_lowfs.ID != -1) ? (img_lowfs.im->array.F + npix) : NULL,
            NULL, NULL, NULL, NULL, NULL, NULL);

        cacaosim_beam_shm_write_spec(img_out.im, &out_spec);
        processinfo_update_output_stream(processinfo, img_out.im, img_in.im);

        if (img_lowfs.ID != -1)
        {
            cacaosim_beam_shm_write_spec(img_lowfs.im, &lo_spec);
            processinfo_update_output_stream(processinfo, img_lowfs.im, img_in.im);
        }
    }
    MILK_PROCINFO_END(self)

    cacaosim_coro_free(&coro_ctx);
    if (img_lyot.ID != -1)
    {
        imgid_free(&img_lyot);
    }
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
              .description = "Simulate coronagraphic focal mask, Lyot stop, and LOWFS pickoff" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_coro),
    .compute   = coro_compute,
    .confcheck = coro_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-coro", runnable_cacaosim_coro);
