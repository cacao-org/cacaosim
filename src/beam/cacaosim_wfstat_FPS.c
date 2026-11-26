// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfstat_FPS.c
 * @brief   MILK_RUNNABLE semaphore-triggered pupil wavefront stats & FITS cube recorder
 *
 * Connects the upstream optical beams (.beam_raw, .beam_stg1, .beam_final) exported over SHM,
 * and feeds them to the shared wavefront statistics engine (common/cacaosim_wfstat_core) on
 * every trigger. Accumulated maps, sample cubes, and wf_stats.json are written on loop exit.
 *
 * In fused mode, cacaosim.htsim runs the same engine in-process on device beams
 * (.wfstat_enable), which avoids exporting beams to SHM; this unit remains the modular-mode
 * recorder and a cross-check of the in-DAG statistics.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "beam/cacaosim_wfstat_FPS.h"
#include "common/cacaosim_wfstat_core.h"

static char     param_trig_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "b_post_stg2";
static char     param_beam_raw[FUNCTION_PARAMETER_STRMAXLEN]   = "b_pup";
static char     param_beam_stg1[FUNCTION_PARAMETER_STRMAXLEN]  = "b_post_stg1";
static char     param_beam_final[FUNCTION_PARAMETER_STRMAXLEN] = "b_post_stg2";
static char     param_outdir[FUNCTION_PARAMETER_STRMAXLEN]     = ".";
static uint64_t param_target_ticks                             = 20000;
static float    param_dt_phys                                  = 0.0005f;
static int64_t  param_snap_extra                               = 1;
static uint64_t param_clip_start                               = 0;
static uint32_t param_clip_len                                 = 0;
static float    param_full_fps                                 = 0.0f;

#define CACAOSIM_WFSTAT_PARAMS(X)                                                      \
    X(".trig_beam", param_trig_beam, FPTYPE_STREAMNAME, 1,                             \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                    \
      "Trigger optical beam stream (usually post-Stage-2)")                            \
    X(".beam_raw", param_beam_raw, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,             \
      "Raw entrance pupil beam stream name")                                           \
    X(".beam_stg1", param_beam_stg1, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,           \
      "Post-Stage-1 optical beam stream name")                                         \
    X(".beam_final", param_beam_final, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,         \
      "Post-Stage-2 final optical beam stream name")                                   \
    X(".outdir", param_outdir, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Output directory (<64 chars; or set WFSTAT_OUTDIR env)")                         \
    X(".target_ticks", &param_target_ticks, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,    \
      "Planned integration tick count for 5 sample markers")                           \
    X(".dt_phys", &param_dt_phys, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,             \
      "Physical frame interval [s] (default: 0.0005)")                                 \
    X(".snap_extra", &param_snap_extra, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,         \
      "Capture synchronized WFS/DM/PSF 2D snapshot cubes")                             \
    X(".clip_start", &param_clip_start, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,        \
      "1-based frame start for short-timescale WF/DM video clip (0 = auto mid)")       \
    X(".clip_len", &param_clip_len, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Contiguous frame count for short-timescale WF/DM video clip (0 = off)")         \
    X(".full_fps", &param_full_fps, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,           \
      "Strided full-duration WF/DM capture rate [Hz] for 1x real-time video (0 = off)")

MILK_PARAMS_DEFINE(cacaosim_wfstat, CACAOSIM_WFSTAT_PARAMS);

static MILK_HOT errno_t wfstat_compute(const MILK_RUNNABLE *self)
{
    const char *env_out  = getenv("WFSTAT_OUTDIR");
    const char *base_out = (env_out && env_out[0] != '\0') ? env_out : param_outdir;
    if (!base_out || base_out[0] == '\0')
    {
        base_out = ".";
    }
    char out_dir[1024];
    snprintf(out_dir, sizeof(out_dir), "%s", base_out);

    const char *loc_snames[CACAOSIM_WFSTAT_NB_LOC] = {
        param_beam_raw,
        param_beam_stg1,
        param_beam_final,
    };
    IMGID img_loc[CACAOSIM_WFSTAT_NB_LOC];
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        img_loc[k] = stream_connect(loc_snames[k]);
        if (img_loc[k].ID == -1)
        {
            for (int j = 0; j < k; j++)
            {
                imgid_free(&img_loc[j]);
            }
            fprintf(stderr, "cacaosim.wfstat: failed to connect beam '%s'\n", loc_snames[k]);
            return RETURN_FAILURE;
        }
    }

    /* Optional separate trigger stream keeps its connection alive for processinfo triggering */
    IMGID img_trig = { 0 };
    img_trig.ID    = -1;
    if (param_trig_beam[0] != '\0' && strcmp(param_trig_beam, param_beam_raw) != 0 &&
        strcmp(param_trig_beam, param_beam_stg1) != 0 &&
        strcmp(param_trig_beam, param_beam_final) != 0)
    {
        img_trig = stream_connect(param_trig_beam);
    }

    uint32_t     nwf     = img_loc[0].md->size[0];
    const float *amp_src = (img_loc[0].md->naxis >= 3 && img_loc[0].md->size[2] >= 2)
                               ? (img_loc[0].im->array.F + (uint64_t) nwf * nwf)
                               : NULL;

    cacaosim_wfstat_t ws;
    if (cacaosim_wfstat_init(&ws, nwf, amp_src, param_target_ticks, (int) param_snap_extra, 0,
                             NULL) != 0)
    {
        for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
        {
            imgid_free(&img_loc[k]);
        }
        return RETURN_FAILURE;
    }
    if (param_clip_len > 0u)
    {
        uint64_t cstart = param_clip_start;
        if (cstart == 0ULL)
        {
            uint64_t tgt = (param_target_ticks > 10ULL) ? param_target_ticks : 20000ULL;
            uint64_t mid = tgt / 2ULL;
            if (mid + (uint64_t) param_clip_len > tgt)
            {
                cstart = (tgt > (uint64_t) param_clip_len)
                             ? ((tgt - (uint64_t) param_clip_len) / 2ULL + 1ULL)
                             : 1ULL;
            }
            else
            {
                cstart = (mid > 20ULL) ? (mid - 20ULL) : 1ULL;
            }
        }
        (void) cacaosim_wfstat_enable_clip(&ws, cstart, param_clip_len);
    }
    if (param_full_fps > 0.0f && param_dt_phys > 0.0f)
    {
        double step_fr = 1.0 / ((double) param_full_fps * (double) param_dt_phys);
        if (step_fr < 1.0)
        {
            step_fr = 1.0;
        }
        uint64_t tgt = (param_target_ticks > 10ULL) ? param_target_ticks : 20000ULL;
        uint32_t cap = (uint32_t) ((double) tgt / step_fr) + 8u;
        if (cap > 3600u)
        {
            cap = 3600u;
        }
        (void) cacaosim_wfstat_enable_full(&ws, step_fr, cap);
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        const float *opd[CACAOSIM_WFSTAT_NB_LOC];
        for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
        {
            opd[k] = img_loc[k].im->array.F;
        }
        cacaosim_wfstat_frame(&ws, opd);
    }
    MILK_PROCINFO_END(self)

    cacaosim_wfstat_write(&ws, out_dir, (double) param_dt_phys);
    cacaosim_wfstat_free(&ws);

    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        imgid_free(&img_loc[k]);
    }
    if (img_trig.ID != -1)
    {
        imgid_free(&img_trig);
    }
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_wfstat = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_wfstat",
              .cmdkey      = "wfstat",
              .description = "Real-time pupil wavefront stats & synchronized FITS cube recorder" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_wfstat),
    .compute = wfstat_compute,
};
EXPORT_FPSEXEC(milk, "cacaosim-wfstat", runnable_cacaosim_wfstat);
