// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_adc.h
 * @brief   Subaru AO188 Atmospheric Dispersion Compensator (ADC) optical & mechanical engine
 *
 * Architectural Role:
 *   Implements the counter-rotating doublet-prism Atmospheric Dispersion Compensator calibrated
 *   directly against Sebastian Egner's Subaru AO188 ADC Specification Document (17 April 2009):
 *     - Calibrated `ao188` (Table 11 actual manufactured prisms) and `ao188_nominal` (Table 10
 *       nominal design) dispersion models matching all 7 IRCS/HiCIAO bands (`0.45..2.37 um`).
 *     - Exact 3D vector Snell's Law 6-interface ray tracer (`snell_doublet`) for arbitrary custom
 *       cemented-doublet glass pairs (Ohara/Schott Sellmeier catalog or custom coefficients).
 *     - Full Table 29 mechanical & tolerance model (`0.087 deg` step size, `3.75 deg/s` max slew,
 *       `0.200 deg` backlash hysteresis, `0.150 deg` absolute accuracy, prism-to-prism offset).
 *     - LUT solver supporting Egner's `whole` (0.45..2.20 um) and single-band (`H`, `J`, `z`, `K'`,
 *       `K`, `vis`, or custom `[opt_lam1_um, opt_lam2_um]`) 5-wavelength RMS optimization, plus
 *       live SHM command injection (`in_cmd`) and `[10 x 1]` float32 telemetry (`out_state`).
 */

#ifndef CACAOSIM_ADC_H
#define CACAOSIM_ADC_H

#include <stdint.h>

#include "common/cacaosim_beam.h"
#include "common/cacaosim_disp.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Number of float32 telemetry elements written to `out_state` SHM stream */
#define CACAOSIM_ADC_STATE_LEN 10

/** Optical prescription preset / model selector */
typedef enum
{
    CACAOSIM_ADC_PRESET_AO188         = 0, /**< Calibrated AO188 measured prisms (Egner Table 11) */
    CACAOSIM_ADC_PRESET_AO188_NOMINAL = 1, /**< Calibrated AO188 nominal design (Egner Table 10) */
    CACAOSIM_ADC_PRESET_SNELL_DOUBLET = 2  /**< Exact 3D vector Snell 6-interface cemented doublet */
} cacaosim_adc_preset_t;

/** ADC control operating mode */
typedef enum
{
    CACAOSIM_ADC_MODE_RETRACTED    = 0, /**< Physically retracted out of beam (zero disp, T=1) */
    CACAOSIM_ADC_MODE_MANUAL       = 1, /**< Fixed manual prism angles `(angle1_deg, angle2_deg)` */
    CACAOSIM_ADC_MODE_LUT          = 2, /**< Open-loop atmospheric LUT tracking `(z, PA)` */
    CACAOSIM_ADC_MODE_SHM          = 3, /**< External `[2x1]` SHM command `[angle1, angle2]` */
    CACAOSIM_ADC_MODE_LUT_PLUS_SHM = 4  /**< Open-loop LUT + SHM differential `[d_psi, d_pa]` */
} cacaosim_adc_mode_t;

/** LUT spectral optimization target band */
typedef enum
{
    CACAOSIM_ADC_OPT_H      = 0, /**< H-band (1.500..1.780 um; Egner HiCIAO/coro default) */
    CACAOSIM_ADC_OPT_WHOLE  = 1, /**< Simultaneous APDs + z/J/H/K' (Egner Table 10/11 whole) */
    CACAOSIM_ADC_OPT_J      = 2, /**< J-band (1.175..1.345 um) */
    CACAOSIM_ADC_OPT_Z      = 3, /**< z-band (0.970..1.070 um) */
    CACAOSIM_ADC_OPT_KP     = 4, /**< K'-band (1.985..2.295 um) */
    CACAOSIM_ADC_OPT_K      = 5, /**< K-band (2.030..2.370 um) */
    CACAOSIM_ADC_OPT_VIS    = 6, /**< Visible / APDs (0.450..0.900 um) */
    CACAOSIM_ADC_OPT_CUSTOM = 7  /**< Custom interval `[opt_lam1_um, opt_lam2_um]` */
} cacaosim_adc_opt_band_t;

/**
 * struct cacaosim_adc_cfg_t - Complete AO188 ADC optical, control, and mechanical configuration
 */
typedef struct
{
    int32_t preset;          /**< cacaosim_adc_preset_t (default: CACAOSIM_ADC_PRESET_AO188) */
    int32_t mode;            /**< cacaosim_adc_mode_t (default: CACAOSIM_ADC_MODE_LUT) */
    int32_t opt_band;        /**< cacaosim_adc_opt_band_t (default: CACAOSIM_ADC_OPT_H) */
    float   opt_lam1_um;     /**< Custom LUT lower bound [um] when opt_band == CUSTOM */
    float   opt_lam2_um;     /**< Custom LUT upper bound [um] when opt_band == CUSTOM */

    /* Manual / initial angles (Egner convention: theta1 = pa - 90 + psi, theta2 = pa + 90 - psi) */
    float   angle1_deg;      /**< Manual prism #1 rotation angle [deg] */
    float   angle2_deg;      /**< Manual prism #2 rotation angle [deg] */
    int32_t manual_use_psi;  /**< If 1 in MANUAL mode, derive theta1/2 from (psi_deg, disp_pa) */
    float   manual_psi_deg;  /**< Differential dispersion angle psi [deg] (0 = zero dispersion) */

    /* 3D Snell cemented-doublet prescription (used when preset == SNELL_DOUBLET) */
    char    glass1[32];      /**< First wedge glass name (e.g. "S-FPL51") */
    char    glass2[32];      /**< Second wedge glass name (e.g. "S-NBH5") */
    float   alpha1_deg;      /**< First wedge apex angle [deg] */
    float   alpha2_deg;      /**< Second wedge apex angle [deg] */
    float   beam_mag;        /**< Pupil magnification D_tel / D_beam_adc (Subaru ~ 88.33) */
    float   thick1_mm;       /**< Center thickness of wedge 1 [mm] (default: 15 mm) */
    float   thick2_mm;       /**< Center thickness of wedge 2 [mm] (default: 15 mm) */
    float   gap_mm;          /**< Air gap between prism 1 and prism 2 [mm] (default: 10 mm) */
    double  g1_sellmeier[6]; /**< Optional custom Sellmeier [B1,B2,B3,C1,C2,C3] if B1 > 0 */
    double  g2_sellmeier[6]; /**< Optional custom Sellmeier [B1,B2,B3,C1,C2,C3] if B1 > 0 */

    /* Realistic LUT calibration errors & nominal-vs-manufactured LUT mismatch */
    float   lut_scale_err;   /**< Fractional LUT dispersion gain error (e.g. 0.05 = +5%) */
    float   lut_pa_err_deg;  /**< Static PA alignment error in LUT [deg] */
    float   lut_p_err_hpa;   /**< Pressure sensor bias in LUT [hPa] */
    float   lut_t_err_c;     /**< Temperature sensor bias in LUT [deg C] */
    int32_t lut_use_nominal; /**< 1 = compute LUT from nominal Table 10 even on measured prisms */

    /* Mechanical & optomechanical realism (Egner 2009 Table 29 defaults) */
    int32_t ideal_mech;      /**< 1 = ideal instantaneous stages; 0 = Table 29 step/slew/backlash */
    float   enc_step_deg;    /**< Rotation step size [deg] (Table 29: 0.087 deg) */
    float   max_slew_dps;    /**< Max rotation velocity [deg/s] (Table 29: 3.750 deg/s) */
    float   backlash_deg;    /**< Rotational backlash [deg] (Table 29: 0.200 deg) */
    float   prism_offset_deg;/**< Static relative rotational offset error between prisms [deg] */
    float   unit_rot_err_deg;/**< Static complete-unit rotation error around optical axis [deg] */

    /* Broadband optical throughput when deployed */
    int32_t tput_enable;     /**< 1 = apply AR-coated doublet spectral throughput (0 = unity) */
    float   tput_base;       /**< Mid-band transmission [0..1] when deployed (default: 0.96) */
} cacaosim_adc_cfg_t;

/**
 * struct cacaosim_adc_ctx_t - Persistent runtime state for an ADC optical node
 */
typedef struct
{
    cacaosim_adc_cfg_t cfg;
    int                initialized;
    double             cur_th1_deg;    /**< Actual mechanical angle of prism #1 [deg] */
    double             cur_th2_deg;    /**< Actual mechanical angle of prism #2 [deg] */
    int                dir1;           /**< Last motion direction of prism #1 (-1, 0, +1) */
    int                dir2;           /**< Last motion direction of prism #2 (-1, 0, +1) */
    double             bl_off1_deg;    /**< Active backlash state offset on prism #1 [deg] */
    double             bl_off2_deg;    /**< Active backlash state offset on prism #2 [deg] */

    /* Cached dispersion descriptor to avoid redundant Chebyshev fits when angles are static */
    int                cache_valid;
    int                cache_deployed;
    double             cache_th1_deg;
    double             cache_th2_deg;
    uint64_t           cache_in_gen;
    cacaosim_disp_t    cached_adc_disp;
    cacaosim_disp_t    cached_out_disp;

    /* Most recent `[10 x 1]` telemetry snapshot */
    float              state_vec[CACAOSIM_ADC_STATE_LEN];
} cacaosim_adc_ctx_t;

/** Populate default AO188 ADC configuration (Table 11 measured prisms, H-band LUT, Table 29) */
void cacaosim_adc_cfg_defaults(cacaosim_adc_cfg_t *cfg);

/** Parse string preset name ("ao188", "ao188_nominal", "snell_doublet") -> enum (or -1) */
int32_t cacaosim_adc_parse_preset(const char *s);
const char *cacaosim_adc_preset_name(int32_t preset);

/** Parse string mode name ("retracted", "manual", "lut", "shm", "lut_plus_shm") -> enum (or -1) */
int32_t cacaosim_adc_parse_mode(const char *s);
const char *cacaosim_adc_mode_name(int32_t mode);

/** Parse string opt_band ("H", "whole", "J", "z", "K'", "K", "vis", "custom") -> enum (or -1) */
int32_t cacaosim_adc_parse_opt_band(const char *s);
const char *cacaosim_adc_opt_band_name(int32_t opt_band);

/** Evaluate refractive index of a named catalog glass or custom Sellmeier array at `lam_um` */
double cacaosim_adc_glass_n(
    const char   *name,
    const double *custom_sellmeier,
    double        lam_um);

/**
 * cacaosim_adc_eval_deflection_rad() - Evaluate exact ADC sky-projected chromatic tilt `[rad]`
 * @cfg:        ADC optical prescription
 * @lam_um:     Optical wavelength `[um]`
 * @theta1_deg: Rotation angle of prism #1 `[deg]`
 * @theta2_deg: Rotation angle of prism #2 `[deg]`
 * @tx_rad:     Output X angular deflection projected to sky `[rad]`
 * @ty_rad:     Output Y angular deflection projected to sky `[rad]`
 *
 * Uses calibrated Egner (2009) dispersion curves for `ao188` / `ao188_nominal` or exact 6-interface
 * 3D vector Snell ray tracing for `snell_doublet`.
 */
void cacaosim_adc_eval_deflection_rad(
    const cacaosim_adc_cfg_t *cfg,
    double                    lam_um,
    double                    theta1_deg,
    double                    theta2_deg,
    double                   *tx_rad,
    double                   *ty_rad);

/**
 * cacaosim_adc_solve_lut() - Solve optimal Egner rotation angle `psi` and prism angles `(th1, th2)`
 * @cfg:          ADC configuration (band selector, preset, optional LUT errors)
 * @atm:          Current atmospheric dispersion configuration
 * @out_th1_deg:  Output target angle for prism #1 `[deg]`
 * @out_th2_deg:  Output target angle for prism #2 `[deg]`
 * @out_psi_deg:  Optional output Egner differential rotation angle `psi` `[deg]` (`[0..90]`)
 *
 * Minimizes 5-wavelength `[1, 1, 2, 1, 1]` weighted RMS residual dispersion across the selected
 * optimization band (matching Egner 2009 Section 8).
 */
void cacaosim_adc_solve_lut(
    const cacaosim_adc_cfg_t      *cfg,
    const cacaosim_atm_disp_cfg_t *atm,
    double                        *out_th1_deg,
    double                        *out_th2_deg,
    double                        *out_psi_deg);

/** Initialize runtime ADC state (snapping stages to initial LUT/manual position) */
void cacaosim_adc_init(
    cacaosim_adc_ctx_t            *ctx,
    const cacaosim_adc_cfg_t      *cfg,
    const cacaosim_atm_disp_cfg_t *atm);

/**
 * cacaosim_adc_step() - Advance ADC stages by `@dt_s`, compose dispersion, & attenuate spectrum
 * @ctx:      Persistent ADC runtime context
 * @dt_s:     Time step `[seconds]` (used for Table 29 slew-rate clamping when `ideal_mech == 0`)
 * @atm:      Current atmospheric dispersion configuration (for LUT & telemetry)
 * @shm_cmd:  Optional `[2]` float32 command from `in_cmd` SHM (`NULL` if unattached)
 * @in_disp:  Upstream beam chromatic dispersion (`NULL` treated as zero)
 * @in_spec:  Upstream 32-bin photon spectrum (`NULL` skips spectral attenuation)
 * @out_disp: Output beam chromatic dispersion (`in_disp + adc_disp`)
 * @out_spec: Output 32-bin photon spectrum (safe if aliasing `@in_spec`)
 */
void cacaosim_adc_step(
    cacaosim_adc_ctx_t            *ctx,
    double                         dt_s,
    const cacaosim_atm_disp_cfg_t *atm,
    const float                   *shm_cmd,
    const cacaosim_disp_t         *in_disp,
    const cacaosim_spec_t         *in_spec,
    cacaosim_disp_t               *out_disp,
    cacaosim_spec_t               *out_spec);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_ADC_H */
