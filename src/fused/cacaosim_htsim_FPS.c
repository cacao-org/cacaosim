// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_htsim_FPS.c
 * @brief   Fused zero-hop High-Throughput AO hardware bench simulator (DM + PyWFS + FPcam)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "bench/cacaosim_bench.h"
#include "bench/cacaosim_bench_view.h"
#include "common/cacaosim_time.h"
#include "dm/cacaosim_dm.h"
#include "fpcam/cacaosim_fpcam.h"
#include "fused/cacaosim_htsim_FPS.h"
#include "pupil/cacaosim_pupil.h"
#include "pyrwfs/cacaosim_pyrwfs.h"

static char     param_bench_file[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_dmcmd[FUNCTION_PARAMETER_STRMAXLEN]   = "aol0_dmC";
static char     param_out_wfscam[FUNCTION_PARAMETER_STRMAXLEN] = "aol0_wfsim";
static char     param_out_dmopd[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static int64_t  param_enable_fpcam                             = 1;
static char     param_out_fpcam[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_fpcam";
static char     param_out_fplong[FUNCTION_PARAMETER_STRMAXLEN] = "csim_fplong";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]      = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]     = "csim_pup";
static uint32_t param_nwf                                      = 128;
static float    param_pup_diam                                 = 0.9375f;
static uint32_t param_nact_x                                   = 20;
static uint32_t param_nact_y                                   = 20;
static float    param_if_coupling                              = 0.20f;
static float    param_delay_frames                             = 1.0f;
static int32_t  param_dyn_mode                                 = 1;
static float    param_tau_us                                   = 120.0f;
static float    param_f0_hz                                    = 2500.0f;
static float    param_zeta                                     = 0.70f;
static float    param_max_stroke                               = 5.0f;
static int32_t  param_dac_bits                                 = 16;
static uint32_t param_wfs_ncam                                 = 64;
static float    param_wfs_lambda                               = 0.75f;
static float    param_mod_rad                                  = 3.0f;
static int32_t  param_mod_npts                                 = 16;
static float    param_wfs_flux                                 = 100000.0f;
static float    param_wfs_ron                                  = 0.5f;
static int64_t  param_wfs_shutter                              = 1;
static uint32_t param_fp_ncam                                  = 128;
static float    param_fp_lambda                                = 1.65f;
static float    param_fp_pixscale                              = 3.0f;
static uint32_t param_fp_navg                                  = 10;
static float    param_fp_flux                                  = 1000000.0f;
static float    param_fp_ron                                   = 1.0f;
static int64_t  param_reset_long                               = 0;
static int64_t  param_photon_noise                             = 1;
static int32_t  param_out_dtype                                = 0;
static int32_t  param_wf_unit                                  = 1;
static float    param_ref_lambda                               = 0.50f;
static float    param_turb_gain                                = 1.0f;
static float    param_frame_rate                               = 2000.0f;
static uint64_t param_rng_seed                                 = 98765;
static int32_t  param_backend                                  = 0;
static float    param_strehl_inst                              = 0.0f;
static float    param_strehl_ave                               = 0.0f;

#define CACAOSIM_HTSIM_PARAMS(X)                                                        \
    X(".bench_file", param_bench_file, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional declarative multi-stage .bench DAG file (empty = flat single-stage)")   \
    X(".in_dmcmd", param_in_dmcmd, FPTYPE_STREAMNAME, 1,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 2D DM actuator command stream (auto-created if missing)")                  \
    X(".out_wfscam", param_out_wfscam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,      \
      "Output 2D Pyramid WFS camera stream (e.g. aol0_wfsim)")                          \
    X(".out_dmopd", param_out_dmopd, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Optional output 2D DM OPD stream (empty = zero-copy GPU internal)")              \
    X(".enable_fpcam", &param_enable_fpcam, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,      \
      "Enable in-process focal plane science PSF camera")                               \
    X(".out_fpcam", param_out_fpcam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D focal plane science camera stream")                                    \
    X(".out_fplong", param_out_fplong, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional output float32 cumulative long-exposure PSF stream")                    \
    X(".in_wf", param_in_wf, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional atmospheric WF stream (e.g. outarraypha from milkatmturb)")             \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional atmospheric amplitude stream (e.g. outarrayamp)")                       \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Clear pupil diameter fraction of nwf (default: 0.9375)")                         \
    X(".nact_x", &param_nact_x, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "DM actuator grid X size (default: 20)")                                          \
    X(".nact_y", &param_nact_y, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "DM actuator grid Y size (default: 20)")                                          \
    X(".if_coupling", &param_if_coupling, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "DM inter-actuator mechanical coupling (default: 0.20)")                          \
    X(".delay_frames", &param_delay_frames, FPTYPE_FLOAT32, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Pure DM hardware latency [frames] (default: 1.0)")                               \
    X(".dyn_mode", &param_dyn_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM dynamics: 0=instant, 1=1st-order RC, 2=2nd-order resonance")                  \
    X(".tau_us", &param_tau_us, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "DM 1st-order RC time constant [us] (default: 120.0)")                            \
    X(".f0_hz", &param_f0_hz, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                  \
      "DM 2nd-order natural resonance [Hz] (default: 2500.0)")                          \
    X(".zeta", &param_zeta, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "DM 2nd-order damping ratio (default: 0.70)")                                     \
    X(".max_stroke", &param_max_stroke, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "DM mechanical stroke clip [um] (default: 5.0)")                                  \
    X(".dac_bits", &param_dac_bits, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM DAC bit depth (0 = floating point, default: 16)")                             \
    X(".wfs_ncam", &param_wfs_ncam, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,             \
      "WFS camera size [pixels] (default: 64)")                                         \
    X(".wfs_lambda", &param_wfs_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "WFS wavelength [um] (default: 0.75)")                                            \
    X(".mod_rad", &param_mod_rad, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "PyWFS modulation radius [lambda/D] (default: 3.0)")                              \
    X(".mod_npts", &param_mod_npts, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "PyWFS modulation points (default: 16)")                                          \
    X(".wfs_flux", &param_wfs_flux, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Guide-star flux [photons / WFS frame] (default: 100000)")                        \
    X(".wfs_ron", &param_wfs_ron, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "WFS camera readout noise [e- RMS] (default: 0.5)")                               \
    X(".wfs_shutter", &param_wfs_shutter, FPTYPE_ONOFF, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "WFS optical shutter (ON = open, OFF = dark frame)")                              \
    X(".fp_ncam", &param_fp_ncam, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,               \
      "Focal plane camera size [pixels] (default: 128)")                                \
    X(".fp_lambda", &param_fp_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Focal plane wavelength [um] (default: 1.65)")                                    \
    X(".fp_pixscale", &param_fp_pixscale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "Focal plane pixel scale [pixels per lambda/D] (default: 3.0)")                   \
    X(".fp_navg", &param_fp_navg, FPTYPE_UINT32, 0,                                     \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "WFS frames integrated per focal plane readout (default: 10)")                    \
    X(".fp_flux", &param_fp_flux, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Science starlight flux [photons / FP exposure] (default: 1e6)")                  \
    X(".fp_ron", &param_fp_ron, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Focal plane readout noise [e- RMS] (default: 1.0)")                              \
    X(".reset_long", &param_reset_long, FPTYPE_ONOFF, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Toggle ON to clear cumulative long-exposure PSF")                                \
    X(".photon_noise", &param_photon_noise, FPTYPE_ONOFF, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Enable Poisson photon shot noise on detectors")                                  \
    X(".out_dtype", &param_out_dtype, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Detector output datatype: 0 = uint16, 1 = float32")                              \
    X(".wf_unit", &param_wf_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1)")                   \
    X(".ref_lambda", &param_ref_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reference wavelength [um] for phase input (default: 0.50)")                      \
    X(".turb_gain", &param_turb_gain, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live turbulence multiplier (0 = off for calibration, 1 = nominal)")              \
    X(".frame_rate", &param_frame_rate, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Hardware master clock [Hz] via TIMER_ABSTIME (0 = use procinfo trigger)")        \
    X(".rng_seed", &param_rng_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,             \
      "Detector PRNG seed")                                                             \
    X(".backend", &param_backend, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Compute backend: 0=auto, 1=CPU, 2=CUDA")                                         \
    X(".strehl_inst", &param_strehl_inst, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,     \
      "Instantaneous focal plane peak Strehl ratio [0..1]")                             \
    X(".strehl_ave", &param_strehl_ave, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,       \
      "Cumulative long-exposure peak Strehl ratio [0..1]")

MILK_PARAMS_DEFINE(cacaosim_htsim, CACAOSIM_HTSIM_PARAMS);

static MILK_COLD errno_t htsim_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".delay_frames", ".tau_us",      ".max_stroke", ".mod_rad",   ".wfs_flux",
            ".wfs_ron",      ".wfs_shutter", ".fp_navg",    ".fp_flux",   ".fp_ron",
            ".reset_long",   ".photon_noise", ".turb_gain", ".frame_rate"
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

static void resolve_bench_path(const char *spec, char *out, size_t cap)
{
    if (strcmp(spec, "@env") == 0)
    {
        const char *ev = getenv("CACAOSIM_BENCH_FILE");
        if (ev != NULL && ev[0] != '\0')
        {
            snprintf(out, cap, "%s", ev);
            return;
        }
    }
    if (access(spec, R_OK) == 0)
    {
        snprintf(out, cap, "%s", spec);
        return;
    }
    if (spec[0] != '/')
    {
        const char *shm_dir = getenv("MILK_SHM_DIR");
        if (shm_dir != NULL && shm_dir[0] != '\0')
        {
            snprintf(out, cap, "%s/%s", shm_dir, spec);
            if (access(out, R_OK) == 0)
            {
                return;
            }
        }
        const char *milk_root = getenv("MILK_ROOT");
        if (milk_root != NULL && milk_root[0] != '\0')
        {
            snprintf(out, cap, "%s/share/cacaosim/examples/%s", milk_root, spec);
            if (access(out, R_OK) == 0)
            {
                return;
            }
        }
    }
    snprintf(out, cap, "%s", spec);
}

static MILK_HOT errno_t htsim_run_bench(const MILK_RUNNABLE *self)
{
    cacaosim_bench_cfg_t bcfg;
    cacaosim_bench_ctx_t bctx;
    char                 resolved[1024];
    char                 err_buf[256] = { 0 };

    resolve_bench_path(param_bench_file, resolved, sizeof(resolved));
    if (cacaosim_bench_load_file(resolved, &bcfg, err_buf, sizeof(err_buf)) != 0)
    {
        fprintf(stderr, "[cacaosim.htsim] .bench load error: %s\n", err_buf);
        return RETURN_FAILURE;
    }

    if (param_backend != 0)
    {
        bcfg.backend = param_backend;
    }
    if (param_frame_rate != 2000.0f)
    {
        bcfg.master_hz = param_frame_rate;
    }
    else
    {
        param_frame_rate = bcfg.master_hz;
    }
    if (param_turb_gain != 1.0f)
    {
        bcfg.turb_gain = param_turb_gain;
    }
    else
    {
        param_turb_gain = bcfg.turb_gain;
    }

    if (cacaosim_bench_init(&bctx, &bcfg, NULL, 1, err_buf, sizeof(err_buf)) != 0)
    {
        fprintf(stderr, "[cacaosim.htsim] .bench init error: %s\n", err_buf);
        return RETURN_FAILURE;
    }
    if (isatty(STDOUT_FILENO))
    {
        cacaosim_view_opts_t vopts;
        cacaosim_view_opts_defaults(&vopts);
        vopts.show_footer = 0;
        cacaosim_bench_render(&bctx, resolved, &vopts, stdout);
    }
    else
    {
        cacaosim_bench_print_summary(&bctx, stdout);
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

        if (param_reset_long)
        {
            for (uint32_t i = 0; i < bctx.n_nodes; i++)
            {
                if (bctx.nodes[i].cfg.kind == CACAOSIM_NODE_FPCAM)
                {
                    cacaosim_fpcam_reset_long(&bctx.nodes[i].fpcam_ctx);
                }
            }
            param_reset_long = 0;
        }

        cacaosim_bench_step(&bctx, NULL, NULL, param_turb_gain, processinfo);
        param_strehl_inst = bctx.last_strehl_inst;
        param_strehl_ave  = bctx.last_strehl_ave;
    }
    MILK_PROCINFO_END(self)

    cacaosim_bench_free(&bctx);
    return RETURN_SUCCESS;
}

static MILK_HOT errno_t htsim_compute(const MILK_RUNNABLE *self)
{
    if (param_bench_file[0] != '\0')
    {
        return htsim_run_bench(self);
    }

    if (param_in_dmcmd[0] == '\0' || param_out_wfscam[0] == '\0' || param_nwf < 8)
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
    float       *fallback_pup = NULL;
    const float *pup_ptr      = NULL;

    if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT)
    {
        pup_ptr = img_pup.im->array.F;
    }
    else
    {
        fallback_pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
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

    IMGID img_cmd = stream_connect_create_2Df32(param_in_dmcmd, param_nact_x, param_nact_y);
    if (img_cmd.ID == -1)
    {
        free(fallback_pup);
        imgid_free(&img_cmd);
        return RETURN_FAILURE;
    }

    uint8_t out_dt  = (param_out_dtype == 0) ? _DATATYPE_UINT16 : _DATATYPE_FLOAT;
    IMGID   img_wfs = stream_connect_create_2D(param_out_wfscam, param_wfs_ncam, param_wfs_ncam,
                                               out_dt);
    if (img_wfs.ID == -1)
    {
        free(fallback_pup);
        imgid_free(&img_cmd);
        imgid_free(&img_wfs);
        return RETURN_FAILURE;
    }

    IMGID img_dmopd  = { 0 };
    IMGID img_fpcam  = { 0 };
    IMGID img_fplong = { 0 };
    IMGID img_wf     = { 0 };
    IMGID img_amp    = { 0 };
    img_dmopd.ID     = -1;
    img_fpcam.ID     = -1;
    img_fplong.ID    = -1;
    img_wf.ID        = -1;
    img_amp.ID       = -1;

    if (param_out_dmopd[0] != '\0')
    {
        img_dmopd = stream_connect_create_2Df32(param_out_dmopd, nwf, nwf);
    }
    if (param_enable_fpcam && param_out_fpcam[0] != '\0')
    {
        img_fpcam = stream_connect_create_2D(param_out_fpcam, param_fp_ncam, param_fp_ncam, out_dt);
        if (param_out_fplong[0] != '\0')
        {
            img_fplong = stream_connect_create_2Df32(param_out_fplong, param_fp_ncam,
                                                     param_fp_ncam);
        }
    }
    if (param_in_wf[0] != '\0')
    {
        img_wf = stream_connect(param_in_wf);
    }
    if (param_in_amp[0] != '\0')
    {
        img_amp = stream_connect(param_in_amp);
    }

    float dt_sec = (param_frame_rate > 1.0f) ? (1.0f / param_frame_rate) : 0.0005f;

    cacaosim_dm_cfg_t dm_cfg = {
        .nact_x       = img_cmd.md->size[0],
        .nact_y       = img_cmd.md->size[1],
        .nwf          = nwf,
        .dm_diam      = param_pup_diam,
        .if_coupling  = param_if_coupling,
        .refl_scale   = 2.0f,
        .max_stroke   = param_max_stroke,
        .dac_bits     = param_dac_bits,
        .nonlin_quad  = 0.0f,
        .dyn_mode     = param_dyn_mode,
        .dt_sec       = dt_sec,
        .tau_sec      = param_tau_us * 1.0e-6f,
        .f0_hz        = param_f0_hz,
        .zeta         = param_zeta,
        .delay_frames = param_delay_frames,
        .backend      = param_backend,
    };

    cacaosim_dm_ctx_t     dm_ctx;
    cacaosim_pyrwfs_ctx_t wfs_ctx;
    cacaosim_fpcam_ctx_t  fp_ctx;
    int                   has_fp = 0;

    if (cacaosim_dm_init(&dm_ctx, &dm_cfg, NULL) != 0)
    {
        free(fallback_pup);
        imgid_free(&img_cmd);
        imgid_free(&img_wfs);
        return RETURN_FAILURE;
    }

    cacaosim_pyrwfs_cfg_t wfs_cfg = {
        .nwf        = nwf,
        .pad_factor = 2,
        .ncam       = param_wfs_ncam,
        .pup_diam   = param_pup_diam,
        .lambda_um  = param_wfs_lambda,
        .mod_rad_ld = param_mod_rad,
        .mod_npts   = param_mod_npts,
        .pup_sep    = 0.50f,
        .roof_ld    = 0.0f,
        .fstop_frac = 0.45f,
        .wf_unit    = param_wf_unit,
        .ref_lambda = param_ref_lambda,
        .rng_seed   = param_rng_seed,
        .backend    = param_backend,
        .det        = {
            .flux_ph      = param_wfs_flux,
            .dark_e       = 0.0f,
            .ron_e        = param_wfs_ron,
            .gain_eadu    = 1.0f,
            .bias_adu     = 100.0f,
            .sat_adu      = 65535.0f,
            .shutter      = param_wfs_shutter,
            .photon_noise = param_photon_noise,
            .out_dtype    = param_out_dtype,
        },
    };

    if (cacaosim_pyrwfs_init(&wfs_ctx, &wfs_cfg, pup_ptr, NULL, dm_ctx.cuda_stream) != 0)
    {
        cacaosim_dm_free(&dm_ctx);
        free(fallback_pup);
        imgid_free(&img_cmd);
        imgid_free(&img_wfs);
        return RETURN_FAILURE;
    }

    if (param_enable_fpcam && img_fpcam.ID != -1)
    {
        cacaosim_fpcam_cfg_t fp_cfg = {
            .nwf         = nwf,
            .ncam        = param_fp_ncam,
            .prop_mode   = CACAOSIM_FPCAM_PROP_MFT,
            .pad_factor  = 2,
            .pup_diam    = param_pup_diam,
            .lambda_um   = param_fp_lambda,
            .bw_frac     = 0.0f,
            .nlambda     = 1,
            .pixscale_ld = param_fp_pixscale,
            .wf_unit     = param_wf_unit,
            .ref_lambda  = param_ref_lambda,
            .rng_seed    = param_rng_seed + 7777ULL,
            .backend     = param_backend,
            .det         = {
                .flux_ph      = param_fp_flux,
                .dark_e       = 0.0f,
                .ron_e        = param_fp_ron,
                .gain_eadu    = 1.0f,
                .bias_adu     = 100.0f,
                .sat_adu      = 65535.0f,
                .shutter      = 1,
                .photon_noise = param_photon_noise,
                .out_dtype    = param_out_dtype,
            },
        };
        if (cacaosim_fpcam_init(&fp_ctx, &fp_cfg, pup_ptr, NULL, dm_ctx.cuda_stream) == 0)
        {
            has_fp = 1;
        }
    }

    float *cpu_dm_buf = NULL;
    if (!dm_ctx.use_cuda && img_dmopd.ID == -1)
    {
        cpu_dm_buf = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
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
                               img_wf.md->size[0] == nwf)
                                  ? img_wf.im->array.F
                                  : NULL;
        const float *amp_ptr = (img_amp.ID != -1 && img_amp.md->datatype == _DATATYPE_FLOAT &&
                                img_amp.md->size[0] == nwf)
                                   ? img_amp.im->array.F
                                   : NULL;

        float *host_dm_dst = (img_dmopd.ID != -1) ? img_dmopd.im->array.F : cpu_dm_buf;
        if (img_dmopd.ID != -1)
        {
            img_dmopd.md->write = 1;
        }

        cacaosim_dm_step(&dm_ctx, img_cmd.im->array.F, NULL, param_delay_frames,
                         param_tau_us * 1.0e-6f, param_max_stroke, host_dm_dst);

        if (img_dmopd.ID != -1)
        {
            processinfo_update_output_stream(NULL, img_dmopd.im, img_cmd.im);
        }

        const float *d_dm_ext    = dm_ctx.use_cuda ? dm_ctx.d_opd_out : NULL;
        const float *host_dm_src = dm_ctx.use_cuda ? NULL : host_dm_dst;

        cacaosim_det_cfg_t wfs_det = {
            .flux_ph      = param_wfs_flux,
            .dark_e       = 0.0f,
            .ron_e        = param_wfs_ron,
            .gain_eadu    = 1.0f,
            .bias_adu     = 100.0f,
            .sat_adu      = 65535.0f,
            .shutter      = param_wfs_shutter,
            .photon_noise = param_photon_noise,
            .out_dtype    = param_out_dtype,
        };

        img_wfs.md->write = 1;
        cacaosim_pyrwfs_step(&wfs_ctx, pup_ptr, amp_ptr, wf_ptr, host_dm_src, NULL, d_dm_ext,
                             param_turb_gain, param_mod_rad, &wfs_det,
                             (param_out_dtype == 0) ? img_wfs.im->array.UI16 : NULL,
                             (param_out_dtype != 0) ? img_wfs.im->array.F : NULL);
        processinfo_update_output_stream(processinfo, img_wfs.im, img_cmd.im);

        if (has_fp)
        {
            if (param_reset_long)
            {
                cacaosim_fpcam_reset_long(&fp_ctx);
                param_reset_long = 0;
            }
            cacaosim_fpcam_accum_subframe(&fp_ctx, pup_ptr, amp_ptr, wf_ptr, host_dm_src, NULL,
                                          d_dm_ext, param_turb_gain);

            uint32_t navg = (param_fp_navg >= 1u) ? param_fp_navg : 1u;
            if (fp_ctx.subframe_cnt >= navg)
            {
                cacaosim_det_cfg_t fp_det = {
                    .flux_ph      = param_fp_flux,
                    .dark_e       = 0.0f,
                    .ron_e        = param_fp_ron,
                    .gain_eadu    = 1.0f,
                    .bias_adu     = 100.0f,
                    .sat_adu      = 65535.0f,
                    .shutter      = 1,
                    .photon_noise = param_photon_noise,
                    .out_dtype    = param_out_dtype,
                };
                if (img_fplong.ID != -1)
                {
                    img_fplong.md->write = 1;
                }
                img_fpcam.md->write = 1;
                cacaosim_fpcam_readout(&fp_ctx, &fp_det,
                                       (param_out_dtype == 0) ? img_fpcam.im->array.UI16 : NULL,
                                       (param_out_dtype != 0) ? img_fpcam.im->array.F : NULL,
                                       (img_fplong.ID != -1) ? img_fplong.im->array.F : NULL);

                param_strehl_inst = fp_ctx.last_strehl_peak;
                param_strehl_ave  = fp_ctx.long_strehl_peak;

                processinfo_update_output_stream(NULL, img_fpcam.im, img_cmd.im);
                if (img_fplong.ID != -1)
                {
                    processinfo_update_output_stream(NULL, img_fplong.im, img_cmd.im);
                }
            }
        }
    }
    MILK_PROCINFO_END(self)

    if (has_fp)
    {
        cacaosim_fpcam_free(&fp_ctx);
    }
    cacaosim_pyrwfs_free(&wfs_ctx);
    cacaosim_dm_free(&dm_ctx);
    free(cpu_dm_buf);
    free(fallback_pup);

    if (img_amp.ID != -1)
    {
        imgid_free(&img_amp);
    }
    if (img_wf.ID != -1)
    {
        imgid_free(&img_wf);
    }
    if (img_fplong.ID != -1)
    {
        imgid_free(&img_fplong);
    }
    if (img_fpcam.ID != -1)
    {
        imgid_free(&img_fpcam);
    }
    if (img_dmopd.ID != -1)
    {
        imgid_free(&img_dmopd);
    }
    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
    imgid_free(&img_wfs);
    imgid_free(&img_cmd);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_htsim = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_htsim",
              .cmdkey      = "htsim",
              .description = "Fused zero-hop HT hardware simulator (DM + PyWFS + FPcam)" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_htsim),
    .compute   = htsim_compute,
    .confcheck = htsim_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-htsim", runnable_cacaosim_htsim);
