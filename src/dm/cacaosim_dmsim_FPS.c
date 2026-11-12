// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dmsim_FPS.c
 * @brief   MILK_RUNNABLE compute unit for zonal DM / Tip-Tilt mirror hardware & in-line beam OPD
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "dm/cacaosim_dm.h"
#include "dm/cacaosim_dmsim_FPS.h"

static char     param_in_cmd[FUNCTION_PARAMETER_STRMAXLEN]   = "dm00disp";
static char     param_out_opd[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_dmopd";
static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN] = "";
static char     param_in_flat[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static int32_t  param_dm_mode                                = 0;
static uint32_t param_nact_x                                 = 16;
static uint32_t param_nact_y                                 = 16;
static uint32_t param_nwf                                    = 128;
static float    param_dm_diam                                = 0.9375f;
static float    param_if_coupling                            = 0.15f;
static float    param_refl_scale                             = 2.0f;
static float    param_tput                                   = 0.98f;
static float    param_max_stroke                             = 5.0f;
static int32_t  param_dac_bits                               = 16;
static float    param_nonlin_quad                            = 0.0f;
static int32_t  param_dyn_mode                               = 1;
static float    param_dt_sec                                 = 0.0005f;
static float    param_tau_sec                                = 0.00015f;
static float    param_f0_hz                                  = 3000.0f;
static float    param_zeta                                   = 0.7f;
static float    param_delay_fr                               = 0.0f;
static float    param_shift_x                                = 0.0f;
static float    param_shift_y                                = 0.0f;
static float    param_rot_deg                                = 0.0f;
static float    param_mag_scale                              = 1.0f;
static int32_t  param_backend                                = 0;

#define CACAOSIM_DMSIM_PARAMS(X)                                                        \
    X(".in_cmd", param_in_cmd, FPTYPE_STREAMNAME, 1,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input actuator command stream (2D zonal grid or 2x1 Tip-Tilt)")                  \
    X(".out_opd", param_out_opd, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D reflective DM OPD stream [um]")                                        \
    X(".in_beam", param_in_beam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional upstream 3D [nwf x nwf x 2] optical beam stream")                       \
    X(".out_beam", param_out_beam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,              \
      "Optional downstream 3D [nwf x nwf x 2] optical beam stream (in_beam - dm_opd)")  \
    X(".dm_mode", &param_dm_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Mirror type: 0 = 2D zonal DM, 1 = 2-axis Tip-Tilt mirror (2x1 cmd)")             \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Output pupil OPD grid dimension [pixels] (default: 128)")                        \
    X(".nact_x", &param_nact_x, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Fallback actuator X dimension if in_cmd is created")                             \
    X(".nact_y", &param_nact_y, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Fallback actuator Y dimension if in_cmd is created")                             \
    X(".dm_diam", &param_dm_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM active diameter fraction of nwf (default: 0.9375)")                           \
    X(".if_coupling", &param_if_coupling, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "Inter-actuator coupling at 1 pitch (default: 0.15)")                             \
    X(".refl_scale", &param_refl_scale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "OPD per mechanical stroke unit (default: 2.0)")                                  \
    X(".tput", &param_tput, FPTYPE_FLOAT32, 0,                                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Mirror optical reflectivity [0..1] applied to out_beam spectrum (default: 0.98)")\
    X(".max_stroke", &param_max_stroke, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Mechanical stroke saturation limit [um] (<=0 disables)")                         \
    X(".dac_bits", &param_dac_bits, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DAC quantization bit depth (0 = floating-point)")                                \
    X(".nonlin_quad", &param_nonlin_quad, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "Quadratic voltage/stroke non-linearity coefficient")                             \
    X(".dyn_mode", &param_dyn_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "Temporal mode: 0=instant, 1=1st-order RC, 2=2nd-order oscillator")               \
    X(".dt_sec", &param_dt_sec, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Physics time step [s] (default: 0.0005 = 2 kHz)")                                \
    X(".tau_sec", &param_tau_sec, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "1st-order 1/e time constant [s] (default: 0.00015)")                             \
    X(".f0_hz", &param_f0_hz, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                  \
      "2nd-order natural resonance frequency [Hz] (default: 3000)")                     \
    X(".zeta", &param_zeta, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "2nd-order damping ratio (default: 0.7)")                                         \
    X(".delay_frames", &param_delay_fr, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Transport latency [frames, fractional supported]")                               \
    X(".shift_x", &param_shift_x, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM-to-pupil X shift [pixels] (default: 0)")                                      \
    X(".shift_y", &param_shift_y, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM-to-pupil Y shift [pixels] (default: 0)")                                      \
    X(".rot_deg", &param_rot_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM-to-pupil CCW rotation [deg] (default: 0)")                                    \
    X(".mag_scale", &param_mag_scale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "DM-to-pupil magnification (default: 1.0)")                                       \
    X(".in_flat", param_in_flat, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional static DM flat/shape error OPD stream [um]")                            \
    X(".backend", &param_backend, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Compute backend: 0=auto, 1=CPU, 2=CUDA")

MILK_PARAMS_DEFINE(cacaosim_dmsim, CACAOSIM_DMSIM_PARAMS);

static MILK_COLD errno_t dmsim_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = { ".max_stroke", ".tau_sec", ".delay_frames", ".tput" };
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

static MILK_HOT errno_t dmsim_compute(const MILK_RUNNABLE *self)
{
    if (param_in_cmd[0] == '\0' || param_out_opd[0] == '\0' || param_nwf < 8)
    {
        return RETURN_FAILURE;
    }

    uint32_t req_nx = (param_dm_mode == 1) ? 2u : param_nact_x;
    uint32_t req_ny = (param_dm_mode == 1) ? 1u : param_nact_y;

    IMGID img_cmd = stream_connect(param_in_cmd);
    if (img_cmd.ID == -1)
    {
        imgid_free(&img_cmd);
        img_cmd = stream_connect_create_2Df32(param_in_cmd, req_nx, req_ny);
    }
    if (img_cmd.ID == -1)
    {
        imgid_free(&img_cmd);
        return RETURN_FAILURE;
    }

    uint32_t nx   = (param_dm_mode == 1) ? 2u : img_cmd.md->size[0];
    uint32_t ny   = (param_dm_mode == 1)
                        ? 1u
                        : ((img_cmd.md->naxis >= 2 && img_cmd.md->size[1] > 0)
                               ? img_cmd.md->size[1]
                               : 1u);
    uint32_t nact = nx * ny;
    uint32_t nwf  = param_nwf;
    uint64_t npix = (uint64_t) nwf * nwf;

    IMGID img_opd = stream_connect_create_2Df32(param_out_opd, nwf, nwf);
    if (img_opd.ID == -1)
    {
        imgid_free(&img_opd);
        imgid_free(&img_cmd);
        return RETURN_FAILURE;
    }

    IMGID img_in_beam  = { 0 };
    IMGID img_out_beam = { 0 };
    IMGID img_flat     = { 0 };
    img_in_beam.ID     = -1;
    img_out_beam.ID    = -1;
    img_flat.ID        = -1;

    if (param_in_beam[0] != '\0')
    {
        img_in_beam = stream_connect(param_in_beam);
        if (img_in_beam.ID == -1)
        {
            imgid_free(&img_in_beam);
            img_in_beam = stream_connect_create_3Df32(param_in_beam, nwf, nwf, 2);
            if (img_in_beam.ID != -1)
            {
                cacaosim_beam_shm_init_kw(img_in_beam.im);
            }
        }
    }
    if (param_out_beam[0] != '\0')
    {
        img_out_beam = stream_connect_create_3Df32(param_out_beam, nwf, nwf, 2);
        if (img_out_beam.ID != -1)
        {
            cacaosim_beam_shm_init_kw(img_out_beam.im);
        }
    }
    if (param_in_flat[0] != '\0')
    {
        img_flat = stream_connect(param_in_flat);
    }

    cacaosim_dm_cfg_t cfg = {
        .nact_x       = nx,
        .nact_y       = ny,
        .nwf          = nwf,
        .dm_diam      = param_dm_diam,
        .if_coupling  = param_if_coupling,
        .refl_scale   = param_refl_scale,
        .max_stroke   = param_max_stroke,
        .dac_bits     = param_dac_bits,
        .nonlin_quad  = param_nonlin_quad,
        .dyn_mode     = param_dyn_mode,
        .dt_sec       = param_dt_sec,
        .tau_sec      = param_tau_sec,
        .f0_hz        = param_f0_hz,
        .zeta         = param_zeta,
        .delay_frames = param_delay_fr,
        .dm_mode      = param_dm_mode,
        .shift_x      = param_shift_x,
        .shift_y      = param_shift_y,
        .rot_deg      = param_rot_deg,
        .mag_scale    = param_mag_scale,
        .backend      = param_backend,
    };

    cacaosim_dm_ctx_t dm_ctx;
    if (cacaosim_dm_init(&dm_ctx, &cfg, NULL) != 0)
    {
        if (img_flat.ID != -1)
        {
            imgid_free(&img_flat);
        }
        if (img_out_beam.ID != -1)
        {
            imgid_free(&img_out_beam);
        }
        if (img_in_beam.ID != -1)
        {
            imgid_free(&img_in_beam);
        }
        imgid_free(&img_opd);
        imgid_free(&img_cmd);
        return RETURN_FAILURE;
    }

    float *cmd_f32 = NULL;
    if (img_cmd.md->datatype != _DATATYPE_FLOAT)
    {
        cmd_f32 = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    }

    cacaosim_spec_t fallback_spec;
    cacaosim_spec_star(&fallback_spec, 8.2f, 0.30f, 8.0f, 5778.0f, 0.75f);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        if (img_in_beam.ID == -1 && param_in_beam[0] != '\0')
        {
            img_in_beam = stream_connect(param_in_beam);
        }

        const float *cmd_ptr = NULL;
        if (img_cmd.md->datatype == _DATATYPE_FLOAT)
        {
            cmd_ptr = img_cmd.im->array.F;
        }
        else if (cmd_f32 != NULL && img_cmd.md->datatype == _DATATYPE_UINT16)
        {
            for (uint32_t a = 0; a < nact; a++)
            {
                cmd_f32[a] = (float) img_cmd.im->array.UI16[a];
            }
            cmd_ptr = cmd_f32;
        }

        const float *flat_ptr = NULL;
        if (img_flat.ID != -1 && img_flat.md->datatype == _DATATYPE_FLOAT &&
            (uint64_t) img_flat.md->size[0] * img_flat.md->size[1] == dm_ctx.npix_wf)
        {
            flat_ptr = img_flat.im->array.F;
        }

        img_opd.md->write = 1;
        cacaosim_dm_step(&dm_ctx, cmd_ptr, flat_ptr, param_delay_fr, param_tau_sec,
                         param_max_stroke, img_opd.im->array.F);
        processinfo_update_output_stream(processinfo, img_opd.im, img_cmd.im);

        if (img_out_beam.ID != -1)
        {
            const float *b_opd = NULL;
            const float *b_amp = NULL;
            cacaosim_spec_t b_spec = fallback_spec;

            if (img_in_beam.ID != -1 && cacaosim_beam_shm_is_valid(img_in_beam.im, nwf))
            {
                b_opd = img_in_beam.im->array.F;
                b_amp = img_in_beam.im->array.F + npix;
                if (!cacaosim_beam_shm_read_spec(img_in_beam.im, &b_spec))
                {
                    b_spec = fallback_spec;
                }
            }
            cacaosim_spec_scale(&b_spec, param_tput, &b_spec);

            img_out_beam.md->write = 1;
            cacaosim_beam_apply_dm_cpu(b_opd, b_amp, img_opd.im->array.F, nwf,
                                       img_out_beam.im->array.F, img_out_beam.im->array.F + npix);
            cacaosim_beam_shm_write_spec(img_out_beam.im, &b_spec);
            IMAGE *b_trig = (img_in_beam.ID != -1) ? img_in_beam.im : img_cmd.im;
            processinfo_update_output_stream(processinfo, img_out_beam.im, b_trig);
        }
    }
    MILK_PROCINFO_END(self)

    free(cmd_f32);
    cacaosim_dm_free(&dm_ctx);
    if (img_flat.ID != -1)
    {
        imgid_free(&img_flat);
    }
    if (img_out_beam.ID != -1)
    {
        imgid_free(&img_out_beam);
    }
    if (img_in_beam.ID != -1)
    {
        imgid_free(&img_in_beam);
    }
    imgid_free(&img_opd);
    imgid_free(&img_cmd);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_dmsim = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_dmsim",
              .cmdkey      = "dmsim",
              .description = "Simulate DM / Tip-Tilt electronics, dynamics, OPD, & in-line beam" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_dmsim),
    .compute   = dmsim_compute,
    .confcheck = dmsim_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-dmsim", runnable_cacaosim_dmsim);
