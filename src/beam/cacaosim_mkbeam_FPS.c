// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mkbeam_FPS.c
 * @brief   MILK_RUNNABLE compute unit synthesizing an [OPD, Amp] optical beam + stellar spectrum
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "beam/cacaosim_mkbeam_FPS.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_time.h"
#include "pupil/cacaosim_pupil.h"

static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN] = "csim_beam0";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]    = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_pup";
static char     param_in_ncpa[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static uint32_t param_nwf                                    = 128;
static float    param_pup_diam                               = 0.9375f;
static float    param_obs_frac                               = 0.30f;
static int32_t  param_n_spiders                              = 4;
static float    param_spider_w                               = 0.012f;
static float    param_tel_diam_m                             = 8.2f;
static float    param_star_mag                               = 8.0f;
static float    param_star_teff                              = 5778.0f;
static float    param_tel_tput                               = 0.75f;
static int32_t  param_wf_unit                                = 1;
static float    param_ref_lambda                             = 0.50f;
static float    param_turb_gain                              = 1.0f;
static float    param_frame_rate                             = 0.0f;

#define CACAOSIM_MKBEAM_PARAMS(X)                                                       \
    X(".out_beam", param_out_beam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "Output 3D [nwf x nwf x 2] optical beam stream (slice0=OPD_um, slice1=Amp)")      \
    X(".in_wf", param_in_wf, FPTYPE_STREAMNAME, 1,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Optional input atmospheric WF stream (e.g. outarraypha)")                        \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional input atmospheric amplitude stream (e.g. outarrayamp)")                 \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".in_ncpa", param_in_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional static telescope NCPA OPD stream [um]")                                 \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Clear pupil diameter fraction of nwf (default: 0.9375)")                         \
    X(".obs_frac", &param_obs_frac, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Central obscuration linear fraction (default: 0.30)")                            \
    X(".n_spiders", &param_n_spiders, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Spider count when auto-generating pupil (default: 4)")                           \
    X(".spider_w", &param_spider_w, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Spider width fraction of pupil diameter (default: 0.012)")                       \
    X(".tel_diam_m", &param_tel_diam_m, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Primary mirror diameter [m] for photon budget (default: 8.2)")                   \
    X(".star_mag", &param_star_mag, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Guide/target star V-band Vega magnitude (default: 8.0)")                         \
    X(".star_teff", &param_star_teff, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Stellar blackbody temperature [K] (default: 5778)")                              \
    X(".tel_tput", &param_tel_tput, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Atmosphere + telescope throughput [0..1] (default: 0.75)")                       \
    X(".wf_unit", &param_wf_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1)")                   \
    X(".ref_lambda", &param_ref_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reference wavelength [um] for phase input (default: 0.50)")                      \
    X(".turb_gain", &param_turb_gain, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live atmospheric turbulence multiplier (0 = flat WF)")                           \
    X(".frame_rate", &param_frame_rate, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Internal TIMER_ABSTIME pacing rate [Hz] (0 = use trigger)")

MILK_PARAMS_DEFINE(cacaosim_mkbeam, CACAOSIM_MKBEAM_PARAMS);

static MILK_COLD errno_t mkbeam_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".star_mag", ".star_teff", ".tel_tput", ".turb_gain", ".frame_rate"
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
    if (param_nwf < 8)
    {
        param_nwf = 8;
    }
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t mkbeam_compute(const MILK_RUNNABLE *self)
{
    uint32_t nwf  = (param_nwf >= 8) ? param_nwf : 128;
    uint64_t npix = (uint64_t) nwf * nwf;

    IMGID img_beam = stream_connect_create_3Df32(param_out_beam, nwf, nwf, 2);
    if (img_beam.ID == -1)
    {
        return RETURN_FAILURE;
    }
    cacaosim_beam_shm_init_kw(img_beam.im);

    IMGID  img_pup      = { 0 };
    float *fallback_pup = NULL;
    img_pup.ID          = -1;
    if (param_in_pup[0] != '\0')
    {
        img_pup = stream_connect(param_in_pup);
    }
    const float *pup_ptr = NULL;
    if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT &&
        img_pup.md->size[0] == nwf && img_pup.md->size[1] == nwf)
    {
        pup_ptr = img_pup.im->array.F;
    }
    else
    {
        fallback_pup = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        if (fallback_pup == NULL)
        {
            imgid_free(&img_beam);
            return RETURN_FAILURE;
        }
        cacaosim_pupil_cfg_t pcfg = {
            .nwf        = nwf,
            .pup_diam   = param_pup_diam,
            .cobs       = param_obs_frac,
            .nspider    = param_n_spiders,
            .spider_w   = param_spider_w,
            .spider_deg = 45.0f,
        };
        cacaosim_make_pupil(&pcfg, fallback_pup);
        pup_ptr = fallback_pup;
    }

    IMGID img_wf   = { 0 };
    IMGID img_amp  = { 0 };
    IMGID img_ncpa = { 0 };
    img_wf.ID      = -1;
    img_amp.ID     = -1;
    img_ncpa.ID    = -1;

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

        if (img_wf.ID == -1 && param_in_wf[0] != '\0')
        {
            img_wf = stream_connect(param_in_wf);
        }
        if (img_amp.ID == -1 && param_in_amp[0] != '\0')
        {
            img_amp = stream_connect(param_in_amp);
        }

        const float *wf_ptr = (img_wf.ID != -1 && img_wf.md->datatype == _DATATYPE_FLOAT &&
                               img_wf.md->size[0] == nwf && img_wf.md->size[1] == nwf)
                                  ? img_wf.im->array.F
                                  : NULL;
        const float *amp_ptr = (img_amp.ID != -1 && img_amp.md->datatype == _DATATYPE_FLOAT &&
                                img_amp.md->size[0] == nwf && img_amp.md->size[1] == nwf)
                                   ? img_amp.im->array.F
                                   : NULL;
        const float *ncpa_ptr = (img_ncpa.ID != -1 && img_ncpa.md->datatype == _DATATYPE_FLOAT &&
                                 img_ncpa.md->size[0] == nwf && img_ncpa.md->size[1] == nwf)
                                    ? img_ncpa.im->array.F
                                    : NULL;

        cacaosim_spec_t spec;
        cacaosim_spec_star(&spec, param_tel_diam_m, param_obs_frac, param_star_mag, param_star_teff,
                           param_tel_tput);

        float wf_to_opd =
            cacaosim_wf_to_opd_scale(param_wf_unit, param_ref_lambda, param_turb_gain);

        img_beam.md->write = 1;
        cacaosim_beam_init_cpu(pup_ptr, amp_ptr, wf_ptr, ncpa_ptr, wf_to_opd, nwf,
                               img_beam.im->array.F, img_beam.im->array.F + npix);
        cacaosim_beam_shm_write_spec(img_beam.im, &spec);

        IMAGE *trig_src = (img_wf.ID != -1) ? img_wf.im : NULL;
        processinfo_update_output_stream(processinfo, img_beam.im, trig_src);
    }
    MILK_PROCINFO_END(self)

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
    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
    imgid_free(&img_beam);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_mkbeam = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_mkbeam",
              .cmdkey      = "mkbeam",
              .description = "Combine telescope pupil, atmosphere, and star spectrum into beam" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_mkbeam),
    .compute   = mkbeam_compute,
    .confcheck = mkbeam_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-mkbeam", runnable_cacaosim_mkbeam);
