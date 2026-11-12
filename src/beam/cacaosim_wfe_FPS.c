// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfe_FPS.c
 * @brief   MILK_RUNNABLE inline static wavefront error optical element specified in OPD [meter]
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "beam/cacaosim_wfe_FPS.h"
#include "common/cacaosim_beam.h"
#include "pupil/cacaosim_pupil.h"

static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_beam0";
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN] = "csim_beam_wfe";
static char     param_in_wfe[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_wfe";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static uint32_t param_nwf                                    = 128;
static float    param_pup_diam                               = 0.9375f;
static float    param_rms_nm                                 = 50.0f;
static float    param_psd_alpha                              = 1.6f;
static uint64_t param_wfe_seed                               = 101;
static float    param_wfe_gain                               = 1.0f;
static float    param_tput                                   = 0.98f;
static float    param_shift_x                                = 0.0f;
static float    param_shift_y                                = 0.0f;
static float    param_rot_deg                                = 0.0f;
static float    param_mag_scale                              = 1.0f;
static int64_t  param_init_only                              = 0;

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

    float rms_m       = (param_rms_nm > 0.0f) ? (param_rms_nm * 1.0e-9f) : 0.0f;
    img_wfe->md->write = 1;
    cacaosim_make_wfe_m(nwf, param_pup_diam, pup_ptr, rms_m, param_psd_alpha, param_wfe_seed,
                        img_wfe->im->array.F);
    wfe_write_unit_kw(img_wfe->im, rms_m);
    ImageStreamIO_UpdateIm(img_wfe->im);

    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
}

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

static MILK_HOT errno_t wfe_compute(const MILK_RUNNABLE *self)
{
    if (param_in_wfe[0] == '\0')
    {
        return RETURN_FAILURE;
    }

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

    IMGID img_in = stream_connect(param_in_beam);
    if (img_in.ID == -1)
    {
        imgid_free(&img_in);
        uint32_t fb_nwf = (param_nwf >= 8u) ? param_nwf : 128u;
        img_in          = stream_connect_create_3Df32(param_in_beam, fb_nwf, fb_nwf, 2u);
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

    IMGID img_out = stream_connect_create_3Df32(param_out_beam, nwf, nwf, in_nz);
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

        cacaosim_spec_t in_spec;
        if (!cacaosim_beam_shm_read_spec(img_in.im, &in_spec))
        {
            in_spec = fallback_spec;
        }
        cacaosim_spec_t out_spec;
        cacaosim_spec_scale(&in_spec, param_tput, &out_spec);

        cacaosim_reg_t reg = {
            .shift_x = param_shift_x,
            .shift_y = param_shift_y,
            .rot_deg = param_rot_deg,
            .scale   = (param_mag_scale > 1e-4f) ? param_mag_scale : 1.0f,
        };
        const float wfe_to_um = 1.0e6f * param_wfe_gain;

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

        cacaosim_beam_shm_write_spec(img_out.im, &out_spec);
        uint32_t nwl_kw = 1u;
        float    wl0_kw = 0.0f;
        float    wl1_kw = 0.0f;
        if (cacaosim_beam_shm_read_poly_kw(img_in.im, &nwl_kw, &wl0_kw, &wl1_kw))
        {
            cacaosim_beam_shm_write_poly_kw(img_out.im, nwl_kw, wl0_kw, wl1_kw);
        }
        processinfo_update_output_stream(processinfo, img_out.im, img_in.im);
    }
    MILK_PROCINFO_END(self)

    imgid_free(&img_out);
    imgid_free(&img_wfe);
    imgid_free(&img_in);
    return RETURN_SUCCESS;
}

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
