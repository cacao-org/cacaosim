// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_adc_FPS.c
 * @brief   MILK_RUNNABLE FPS compute unit for inline Subaru AO188 ADC (`cacaosim.adc`)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "adc/cacaosim_adc.h"
#include "adc/cacaosim_adc_FPS.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_disp.h"
#include "cacaosim_types.h"

/* =====================================================================
 * 1. LOCAL FPS PARAMETER VARIABLES
 * ===================================================================== */

static char     param_in_beam[FUNCTION_PARAMETER_STRMAXLEN]   = "b_pup";
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "b_adc";
static char     param_in_cmd[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char     param_out_state[FUNCTION_PARAMETER_STRMAXLEN] = "";
static uint32_t param_nwf                                     = 256;
static char     param_preset[FUNCTION_PARAMETER_STRMAXLEN]    = "ao188";
static char     param_mode[FUNCTION_PARAMETER_STRMAXLEN]      = "lut";
static char     param_opt_band[FUNCTION_PARAMETER_STRMAXLEN]  = "H";
static float    param_opt_lam1                                = 1.50f;
static float    param_opt_lam2                                = 1.78f;
static float    param_angle1_deg                              = -90.0f;
static float    param_angle2_deg                              = 90.0f;
static int32_t  param_manual_use_psi                          = 0;
static float    param_manual_psi_deg                          = 0.0f;
static char     param_glass1[FUNCTION_PARAMETER_STRMAXLEN]    = "S-FPL51";
static char     param_glass2[FUNCTION_PARAMETER_STRMAXLEN]    = "S-NBH5";
static float    param_alpha1_deg                              = -1.526f;
static float    param_alpha2_deg                              = 1.085f;
static float    param_beam_mag                                = 88.333f;
static float    param_zenith_deg                              = 30.0f;
static float    param_disp_pa_deg                             = 0.0f;
static float    param_pressure_hpa                            = 615.0f;
static float    param_temp_c                                  = 2.0f;
static float    param_rh_pct                                  = 20.0f;
static float    param_ref_lam_um                              = 1.65f;
static float    param_lut_scale_err                           = 0.0f;
static float    param_lut_pa_err_deg                          = 0.0f;
static float    param_lut_p_err_hpa                           = 0.0f;
static float    param_lut_t_err_c                             = 0.0f;
static int32_t  param_lut_use_nominal                         = 0;
static int32_t  param_ideal_mech                              = 1;
static float    param_enc_step_deg                            = 0.087f;
static float    param_max_slew_dps                            = 3.750f;
static float    param_backlash_deg                            = 0.200f;
static float    param_prism_offset_deg                        = 0.0f;
static float    param_unit_rot_err_deg                        = 0.0f;
static int32_t  param_tput_enable                             = 1;
static float    param_tput_base                               = 0.96f;
static float    param_dt_sec                                  = 0.0005f;

/* =====================================================================
 * 2. UNIFIED FPS PARAMETER TABLE
 * ===================================================================== */

#define CACAOSIM_ADC_PARAMS(X)                                                                  \
    X(".in_beam", param_in_beam, FPTYPE_STREAMNAME, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Input 3D [OPD_um, Amp] optical beam SHM stream")                                         \
    X(".out_beam", param_out_beam, FPTYPE_STREAMNAME, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Output 3D [OPD_um, Amp] optical beam SHM stream")                                        \
    X(".in_cmd", param_in_cmd, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                          \
      "Optional [2x1] float32 SHM prism command stream (empty = none)")                         \
    X(".out_state", param_out_state, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional [10x1] float32 SHM telemetry output stream (empty = none)")                     \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                               \
      "Fallback wavefront grid size [pixels]")                                                  \
    X(".preset", param_preset, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                          \
      "ADC preset: ao188 (Table 11), ao188_nominal (Table 10), snell_doublet")                  \
    X(".mode", param_mode, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,            \
      "ADC mode: retracted, manual, lut, shm, lut_plus_shm")                                    \
    X(".opt_band", param_opt_band, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,    \
      "LUT band: H, whole, J, z, K', K, vis, custom")                                           \
    X(".opt_lam1_um", &param_opt_lam1, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Custom LUT band lower bound [um]")                                                       \
    X(".opt_lam2_um", &param_opt_lam2, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Custom LUT band upper bound [um]")                                                       \
    X(".angle1_deg", &param_angle1_deg, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Manual prism #1 angle [deg]")                    \
    X(".angle2_deg", &param_angle2_deg, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Manual prism #2 angle [deg]")                    \
    X(".manual_use_psi", &param_manual_use_psi, FPTYPE_INT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "1 = use manual_psi_deg in manual mode")          \
    X(".manual_psi_deg", &param_manual_psi_deg, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Manual Egner dispersion angle psi [deg]")        \
    X(".glass1", param_glass1, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                          \
      "Doublet glass #1 name (for snell_doublet)")                                              \
    X(".glass2", param_glass2, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                          \
      "Doublet glass #2 name (for snell_doublet)")                                              \
    X(".alpha1_deg", &param_alpha1_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Doublet wedge #1 apex angle [deg]")                                                      \
    X(".alpha2_deg", &param_alpha2_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Doublet wedge #2 apex angle [deg]")                                                      \
    X(".beam_mag", &param_beam_mag, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Pupil magnification D_tel / D_beam_adc (default: 88.333)")                               \
    X(".zenith_deg", &param_zenith_deg, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Target zenith distance for LUT [deg]")           \
    X(".disp_pa_deg", &param_disp_pa_deg, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Zenith dispersion azimuth on pupil [deg]")       \
    X(".atm_pressure_hpa", &param_pressure_hpa, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Ambient atmospheric pressure [hPa]")                                                     \
    X(".atm_temp_c", &param_temp_c, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Ambient air temperature [deg C]")                                                        \
    X(".atm_rh_pct", &param_rh_pct, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Ambient relative humidity [%]")                                                          \
    X(".atm_disp_ref_um", &param_ref_lam_um, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,           \
      "Reference pointing wavelength [um]")                                                     \
    X(".lut_scale_err", &param_lut_scale_err, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Fractional LUT dispersion gain error")           \
    X(".lut_pa_err_deg", &param_lut_pa_err_deg, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "Static LUT PA alignment error [deg]")            \
    X(".lut_p_err_hpa", &param_lut_p_err_hpa, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "LUT pressure bias [hPa]")                        \
    X(".lut_t_err_c", &param_lut_t_err_c, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN, "LUT temperature bias [deg C]")                   \
    X(".lut_use_nominal", &param_lut_use_nominal, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "1 = drive measured prisms with Table 10 nominal LUT")                                    \
    X(".ideal_mech", &param_ideal_mech, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                  \
      "1 = ideal stages; 0 = Table 29 step/slew/backlash")                                      \
    X(".enc_step_deg", &param_enc_step_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Rotation step quantization [deg] (Table 29: 0.087)")                                     \
    X(".max_slew_dps", &param_max_slew_dps, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Max prism rotation rate [deg/s] (Table 29: 3.75)")                                       \
    X(".backlash_deg", &param_backlash_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Rotational backlash [deg] (Table 29: 0.200)")                                            \
    X(".prism_offset_deg", &param_prism_offset_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,    \
      "Static relative rotational offset error between prisms [deg]")                           \
    X(".unit_rot_err_deg", &param_unit_rot_err_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,    \
      "Static complete-unit rotation error [deg]")                                              \
    X(".tput_enable", &param_tput_enable, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "1 = apply AR-coated doublet spectral throughput when deployed")                          \
    X(".tput_base", &param_tput_base, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Mid-band ADC transmission when deployed [0..1]")                                         \
    X(".dt_sec", &param_dt_sec, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                        \
      "Frame step duration [s] for stage slew integration")

MILK_PARAMS_DEFINE(cacaosim_adc, CACAOSIM_ADC_PARAMS);

static void sync_adc_cfg(cacaosim_adc_cfg_t *cfg)
{
    int32_t pr = cacaosim_adc_parse_preset(param_preset);
    int32_t md = cacaosim_adc_parse_mode(param_mode);
    int32_t ob = cacaosim_adc_parse_opt_band(param_opt_band);

    cfg->preset           = (pr >= 0) ? pr : CACAOSIM_ADC_PRESET_AO188;
    cfg->mode             = (md >= 0) ? md : CACAOSIM_ADC_MODE_LUT;
    cfg->opt_band         = (ob >= 0) ? ob : CACAOSIM_ADC_OPT_H;
    cfg->opt_lam1_um      = param_opt_lam1;
    cfg->opt_lam2_um      = param_opt_lam2;
    cfg->angle1_deg       = param_angle1_deg;
    cfg->angle2_deg       = param_angle2_deg;
    cfg->manual_use_psi   = param_manual_use_psi;
    cfg->manual_psi_deg   = param_manual_psi_deg;
    snprintf(cfg->glass1, sizeof(cfg->glass1), "%.31s", param_glass1);
    snprintf(cfg->glass2, sizeof(cfg->glass2), "%.31s", param_glass2);
    cfg->alpha1_deg       = param_alpha1_deg;
    cfg->alpha2_deg       = param_alpha2_deg;
    cfg->beam_mag         = param_beam_mag;
    cfg->lut_scale_err    = param_lut_scale_err;
    cfg->lut_pa_err_deg   = param_lut_pa_err_deg;
    cfg->lut_p_err_hpa    = param_lut_p_err_hpa;
    cfg->lut_t_err_c      = param_lut_t_err_c;
    cfg->lut_use_nominal  = param_lut_use_nominal;
    cfg->ideal_mech       = param_ideal_mech;
    cfg->enc_step_deg     = param_enc_step_deg;
    cfg->max_slew_dps     = param_max_slew_dps;
    cfg->backlash_deg     = param_backlash_deg;
    cfg->prism_offset_deg = param_prism_offset_deg;
    cfg->unit_rot_err_deg = param_unit_rot_err_deg;
    cfg->tput_enable      = param_tput_enable;
    cfg->tput_base        = param_tput_base;
}

static MILK_COLD errno_t adc_conf_check(void)
{
    if (dcfpsptr != NULL)
    {
        static const char *const wr[] = {
            ".mode", ".opt_band", ".angle1_deg", ".angle2_deg", ".manual_use_psi",
            ".manual_psi_deg", ".zenith_deg", ".disp_pa_deg", ".lut_scale_err",
            ".lut_pa_err_deg", ".lut_p_err_hpa", ".lut_t_err_c"
        };
        for (size_t i = 0; i < sizeof(wr) / sizeof(wr[0]); i++)
        {
            long idx = functionparameter_GetParamIndex(dcfpsptr, wr[i]);
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

static MILK_HOT errno_t adc_compute(const MILK_RUNNABLE *self)
{
    IMGID img_in = stream_connect(param_in_beam);
    if (img_in.ID == -1)
    {
        uint32_t fb_nwf = (param_nwf >= 8) ? param_nwf : 128;
        img_in          = cacaosim_beam_shm_create(param_in_beam, fb_nwf, 2u);
        if (img_in.ID == -1)
        {
            return RETURN_FAILURE;
        }
        cacaosim_beam_shm_init_kw(img_in.im);
    }

    uint32_t nwf   = img_in.md->size[0];
    uint32_t in_nz = (img_in.md->naxis >= 3 && img_in.md->size[2] >= 2u) ? img_in.md->size[2] : 2u;
    uint64_t npix  = (uint64_t) nwf * nwf;

    IMGID img_out = cacaosim_beam_shm_create(param_out_beam, nwf, in_nz);
    if (img_out.ID == -1)
    {
        imgid_free(&img_in);
        return RETURN_FAILURE;
    }
    cacaosim_beam_shm_init_kw(img_out.im);

    IMGID img_cmd = { 0 };
    img_cmd.ID    = -1;
    if (param_in_cmd[0] != '\0')
    {
        img_cmd = stream_connect(param_in_cmd);
    }

    IMGID img_state = { 0 };
    img_state.ID    = -1;
    if (param_out_state[0] != '\0')
    {
        img_state = stream_connect_create_2Df32(param_out_state, CACAOSIM_ADC_STATE_LEN, 1u);
    }

    cacaosim_adc_cfg_t acfg;
    cacaosim_adc_cfg_defaults(&acfg);
    sync_adc_cfg(&acfg);

    cacaosim_atm_disp_cfg_t atm;
    cacaosim_atm_disp_cfg_defaults(&atm);
    atm.enable       = 1;
    atm.zenith_deg   = param_zenith_deg;
    atm.disp_pa_deg  = param_disp_pa_deg;
    atm.pressure_hpa = param_pressure_hpa;
    atm.temp_c       = param_temp_c;
    atm.rh_pct       = param_rh_pct;
    atm.ref_lam_um   = param_ref_lam_um;

    cacaosim_adc_ctx_t actx;
    cacaosim_adc_init(&actx, &acfg, &atm);

    cacaosim_spec_t fallback_spec;
    cacaosim_spec_star(&fallback_spec, 7.95f, 0.307f, 6.0f, 5778.0f, 0.75f);

    cacaosim_disp_t in_disp;
    cacaosim_disp_t out_disp;
    memset(&in_disp, 0, sizeof(in_disp));
    memset(&out_disp, 0, sizeof(out_disp));

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        if (img_cmd.ID == -1 && param_in_cmd[0] != '\0')
        {
            img_cmd = stream_connect(param_in_cmd);
        }

        sync_adc_cfg(&actx.cfg);
        atm.zenith_deg   = param_zenith_deg;
        atm.disp_pa_deg  = param_disp_pa_deg;
        atm.pressure_hpa = param_pressure_hpa;
        atm.temp_c       = param_temp_c;
        atm.rh_pct       = param_rh_pct;
        atm.ref_lam_um   = param_ref_lam_um;

        const float *cmd_ptr =
            (img_cmd.ID != -1 && img_cmd.md->datatype == _DATATYPE_FLOAT &&
             (uint64_t) img_cmd.md->nelement >= 2ULL)
                ? img_cmd.im->array.F
                : NULL;

        cacaosim_spec_t in_spec;
        if (!cacaosim_beam_shm_read_spec(img_in.im, &in_spec))
        {
            in_spec = fallback_spec;
        }
        cacaosim_beam_shm_read_disp(img_in.im, &in_disp);

        cacaosim_spec_t out_spec;
        cacaosim_adc_step(&actx, (double) param_dt_sec, &atm, cmd_ptr, &in_disp, &in_spec,
                          &out_disp, &out_spec);

        /* Pass pupil OPD & Amp slices directly (ADC dispersion is carried in CSIM_DX/DY) */
        img_out.md->write = 1;
        memcpy(img_out.im->array.F, img_in.im->array.F, (size_t) in_nz * npix * sizeof(float));
        cacaosim_beam_shm_write_spec(img_out.im, &out_spec);
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

        if (img_state.ID != -1)
        {
            img_state.md->write = 1;
            memcpy(img_state.im->array.F, actx.state_vec, sizeof(actx.state_vec));
            processinfo_update_output_stream(NULL, img_state.im, img_in.im);
        }
    }
    MILK_PROCINFO_END(self)

    if (img_state.ID != -1)
    {
        imgid_free(&img_state);
    }
    if (img_cmd.ID != -1)
    {
        imgid_free(&img_cmd);
    }
    imgid_free(&img_out);
    imgid_free(&img_in);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_adc = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = {
        .fps_name         = "cacaosim_adc",
        .cmdkey           = "adc",
        .description      = "Subaru AO188 Atmospheric Dispersion Compensator (ADC) simulator",
        .description_long = "Simulates counter-rotating doublet-prism atmospheric dispersion "
                            "compensation calibrated to Subaru AO188 (Egner 2009 Tables 10/11 "
                            "& Table 29) or exact 3D Snell ray tracing, with LUT/manual/SHM modes."
    },
    .srcfile   = __FILE__,
    MILK_PARAMS_REF(cacaosim_adc),
    .confcheck = adc_conf_check,
    .compute   = adc_compute,
};

EXPORT_FPSEXEC(milk, "cacaosim-adc", runnable_cacaosim_adc);
