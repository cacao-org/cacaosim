// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mkpupil_FPS.c
 * @brief   MILK_RUNNABLE compute unit generating telescope pupil mask and static NCPA streams
 */

#include <stdint.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "pupil/cacaosim_mkpupil_FPS.h"
#include "pupil/cacaosim_pupil.h"

static char     param_out_pup[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_pup";
static uint32_t param_nwf                                    = 128;
static float    param_pup_diam                               = 0.9375f;
static float    param_cobs                                   = 0.30f;
static int32_t  param_nspider                                = 4;
static float    param_spider_w                               = 0.01f;
static float    param_spider_deg                             = 45.0f;
static char     param_out_ncpa[FUNCTION_PARAMETER_STRMAXLEN] = "";
static float    param_ncpa_rms                               = 0.0f;
static uint64_t param_ncpa_seed                              = 42;

#define CACAOSIM_MKPUPIL_PARAMS(X)                                                      \
    X(".out_pup", param_out_pup, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D pupil amplitude stream name")                                          \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT,                       \
      "Square pupil grid size [pixels] (default: 128)")                                 \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Outer pupil diameter as fraction of nwf (default: 0.9375)")                      \
    X(".cobs", &param_cobs, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Central obstruction fraction of pup_diam (default: 0.30)")                       \
    X(".nspider", &param_nspider, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Number of radial spider vanes (default: 4)")                                     \
    X(".spider_w", &param_spider_w, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Spider vane width as fraction of pup_diam (default: 0.01)")                      \
    X(".spider_deg", &param_spider_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Spider vane azimuth offset [deg] (default: 45)")                                 \
    X(".out_ncpa", param_out_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,              \
      "Optional output static NCPA OPD stream [um] (empty = skip)")                     \
    X(".ncpa_rms", &param_ncpa_rms, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Static NCPA RMS OPD [um] (default: 0.0)")                                        \
    X(".ncpa_seed", &param_ncpa_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,           \
      "Random seed for NCPA Zernike coefficients")

MILK_PARAMS_DEFINE(cacaosim_mkpupil, CACAOSIM_MKPUPIL_PARAMS);

static MILK_HOT errno_t mkpupil_compute(const MILK_RUNNABLE *self)
{
    if (param_nwf < 8 || param_out_pup[0] == '\0')
    {
        return RETURN_FAILURE;
    }

    IMGID img_pup = stream_connect_create_2Df32(param_out_pup, param_nwf, param_nwf);
    if (img_pup.ID == -1)
    {
        imgid_free(&img_pup);
        return RETURN_FAILURE;
    }

    int   has_ncpa = (param_out_ncpa[0] != '\0');
    IMGID img_ncpa = { 0 };
    img_ncpa.ID    = -1;
    if (has_ncpa)
    {
        img_ncpa = stream_connect_create_2Df32(param_out_ncpa, param_nwf, param_nwf);
        if (img_ncpa.ID == -1)
        {
            imgid_free(&img_ncpa);
            imgid_free(&img_pup);
            return RETURN_FAILURE;
        }
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        cacaosim_pupil_cfg_t cfg = {
            .nwf        = param_nwf,
            .pup_diam   = param_pup_diam,
            .cobs       = param_cobs,
            .nspider    = param_nspider,
            .spider_w   = param_spider_w,
            .spider_deg = param_spider_deg,
        };

        img_pup.md->write = 1;
        cacaosim_make_pupil(&cfg, img_pup.im->array.F);
        processinfo_update_output_stream(processinfo, img_pup.im, NULL);

        if (has_ncpa)
        {
            img_ncpa.md->write = 1;
            cacaosim_make_ncpa(param_nwf, param_pup_diam, img_pup.im->array.F, param_ncpa_rms,
                               param_ncpa_seed, img_ncpa.im->array.F);
            processinfo_update_output_stream(processinfo, img_ncpa.im, NULL);
        }
    }
    MILK_PROCINFO_END(self)

    if (has_ncpa)
    {
        imgid_free(&img_ncpa);
    }
    imgid_free(&img_pup);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_mkpupil = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_mkpupil",
              .cmdkey      = "mkpupil",
              .description = "Generate anti-aliased telescope pupil and static NCPA streams" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_mkpupil),
    .compute = mkpupil_compute,
};
EXPORT_FPSEXEC(milk, "cacaosim-mkpupil", runnable_cacaosim_mkpupil);
