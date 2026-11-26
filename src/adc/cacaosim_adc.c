// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_adc.c
 * @brief   Subaru AO188 ADC optical solver (Egner 2009 + 3D vector Snell) and stage dynamics
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef HAVE_OPTICSMATERIALS
#include "OpticsMaterials/OpticsMaterials.h"
#endif

#include "cacaosim_adc.h"
#include "cacaosim_types.h"

#define ADC_PI      3.14159265358979323846
#define ADC_DEG2RAD (ADC_PI / 180.0)
#define ADC_RAD2DEG (180.0 / ADC_PI)
#define ADC_MAS2RAD (1.0 / 206264806.24709636)

/*
 * Calibrated AO188 ADC maximum differential dispersion D_max(lam) - D_max(0.70 um) [mas on sky]
 * reconstructed directly from S. Egner (17 April 2009) Tables 9, 10, 11, 13, 16, 19:
 *   D_max(l) - D_max(0.70) = c0*(l^-2 - 0.7^-2) + c1*(l^-4 - 0.7^-4)
 *                          + c2*(l^2  - 0.7^2)  + c3*(l^4  - 0.7^4)
 * Note: positive D_max(l) for l < 0.70, negative for l > 0.70; ADC applies -sin(psi) * D_max(l)
 * along the zenith dispersion axis so positive psi cancels positive atmospheric dispersion!
 */
static const double k_egner_nominal_mas[4] = {
    336.8307, 13.2420, 2.9890, -2.1854
};
static const double k_egner_measured_mas[4] = {
    317.5423, 15.9328, -2.6620, -1.8637
};

/* Built-in verified Ohara/Schott Sellmeier fallback table (matches OpticsMaterials optmat_solids) */
typedef struct
{
    const char *name;
    double      b1, b2, b3;
    double      c1, c2, c3;
} adc_glass_entry_t;

static const adc_glass_entry_t k_adc_glasses[] = {
    { "SiO2",    0.6961663,   0.4079426,   0.8974794,   0.004679148, 0.013512063, 97.9340025 },
    { "CaF2",    0.5675888,   0.4710914,   3.8484723,   0.002526430, 0.010078333, 1200.5560  },
    { "BaF2",    0.64335600,  0.50676200,  3.82610000,  0.003339569, 0.012029702, 2151.6981  },
    { "S-FPL51", 1.17010505,  0.047571078, 0.763832445, 0.006162039, 0.026337288, 141.882642 },
    { "S-FPL53", 0.983532327, 0.069568814, 1.11409238,  0.004922350, 0.019358109, 264.275294 },
    { "S-FSL5",  1.17447043,  0.014005615, 1.19272435,  0.008418552, -0.05817908, 129.599726 },
    { "S-BSL7",  1.15150190,  0.118583612, 1.26301359,  0.010598413, -0.01182252, 129.617662 },
    { "N-BK7",   1.03961212,  0.231792344, 1.01046945,  0.006000699, 0.020017914, 103.560653 },
    { "BK7",     1.03961212,  0.231792344, 1.01046945,  0.006000699, 0.020017914, 103.560653 },
    { "N-PK52A", 1.02960700,  0.188050600, 0.736488165, 0.005168002, 0.016665880, 138.964129 },
    { "S-BAL35", 0.941357273, 0.546174895, 1.16168917,  0.014033400, 0.000906636, 114.163758 },
    { "S-BSM14", 1.28286270,  0.247647429, 1.10383999,  0.012290240, -0.00613142, 106.883378 },
    { "S-LAL18", 1.50276318,  0.430224497, 1.34726060,  0.014546236, -0.00332784, 93.3508342 },
    { "N-LAK21", 1.22718116,  0.420783743, 1.01284843,  0.006020757, 0.019686289, 88.4370099 },
    { "N-F2",    1.39757037,  0.159201403, 1.26865430,  0.009959061, 0.054693175, 119.248346 },
    { "S-TIM2",  1.42193846,  0.133827968, 1.45060574,  0.010729151, 0.057258755, 145.381805 },
    { "S-TIM28", 1.54270810,  0.217113891, 1.81904459,  0.011392501, 0.057922457, 167.697189 },
    { "S-NBH5",  1.47544521,  0.193060095, 1.50939010,  0.009558367, 0.046043048, 126.422746 },
    { "N-SF2",   1.47343127,  0.163681849, 1.36920899,  0.010901910, 0.058568369, 127.404933 },
    { "N-SF6",   1.77931763,  0.338149866, 2.08734474,  0.013371418, 0.061753362, 174.017590 },
    { "S-TIH1",  1.60326759,  0.242980935, 1.81313592,  0.011801914, 0.059136366, 161.218747 },
    { "S-TIH53", 1.87904886,  0.369719775, 2.33730863,  0.014412177, 0.063881799, 182.668180 },
    { "N-KZFS4", 1.35055424,  0.197575506, 1.09962992,  0.008762821, 0.037176720, 90.3866994 },
    { "ZnSe",    4.45813734,  0.467216334, 2.89566290,  0.040344681, 0.153171390, 2221.82237 },
};

static int str_eq_ci(const char *a, const char *b)
{
    if (a == NULL || b == NULL)
    {
        return 0;
    }
    while (*a && *b)
    {
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b))
        {
            return 0;
        }
        a++;
        b++;
    }
    return (*a == '\0' && *b == '\0');
}

void cacaosim_adc_cfg_defaults(cacaosim_adc_cfg_t *cfg)
{
    if (cfg == NULL)
    {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->preset           = CACAOSIM_ADC_PRESET_AO188;
    cfg->mode             = CACAOSIM_ADC_MODE_LUT;
    cfg->opt_band         = CACAOSIM_ADC_OPT_H;
    cfg->opt_lam1_um      = 1.50f;
    cfg->opt_lam2_um      = 1.78f;
    cfg->angle1_deg       = -90.0f;
    cfg->angle2_deg       = 90.0f;
    cfg->manual_use_psi   = 0;
    cfg->manual_psi_deg   = 0.0f;
    snprintf(cfg->glass1, sizeof(cfg->glass1), "S-FPL51");
    snprintf(cfg->glass2, sizeof(cfg->glass2), "S-NBH5");
    cfg->alpha1_deg       = -1.526f;
    cfg->alpha2_deg       = 1.085f;
    cfg->beam_mag         = 88.333f; /* 7.95 m / 0.090 m collimated pupil */
    cfg->thick1_mm        = 15.0f;
    cfg->thick2_mm        = 15.0f;
    cfg->gap_mm           = 10.0f;
    cfg->lut_scale_err    = 0.0f;
    cfg->lut_pa_err_deg   = 0.0f;
    cfg->lut_p_err_hpa    = 0.0f;
    cfg->lut_t_err_c      = 0.0f;
    cfg->lut_use_nominal  = 0;
    cfg->ideal_mech       = 1;
    cfg->enc_step_deg     = 0.087f; /* Egner 2009 Table 29 */
    cfg->max_slew_dps     = 3.750f; /* Egner 2009 Table 29 */
    cfg->backlash_deg     = 0.200f; /* Egner 2009 Table 29 */
    cfg->prism_offset_deg = 0.0f;
    cfg->unit_rot_err_deg = 0.0f;
    cfg->tput_enable      = 1;
    cfg->tput_base        = 0.96f;
}

int32_t cacaosim_adc_parse_preset(const char *s)
{
    if (s == NULL)
    {
        return -1;
    }
    if (str_eq_ci(s, "ao188") || str_eq_ci(s, "ao188_measured") || str_eq_ci(s, "0"))
    {
        return CACAOSIM_ADC_PRESET_AO188;
    }
    if (str_eq_ci(s, "ao188_nominal") || str_eq_ci(s, "nominal") || str_eq_ci(s, "1"))
    {
        return CACAOSIM_ADC_PRESET_AO188_NOMINAL;
    }
    if (str_eq_ci(s, "snell_doublet") || str_eq_ci(s, "snell") || str_eq_ci(s, "doublet") ||
        str_eq_ci(s, "2"))
    {
        return CACAOSIM_ADC_PRESET_SNELL_DOUBLET;
    }
    return -1;
}

const char *cacaosim_adc_preset_name(int32_t preset)
{
    switch (preset)
    {
    case CACAOSIM_ADC_PRESET_AO188:
        return "ao188";
    case CACAOSIM_ADC_PRESET_AO188_NOMINAL:
        return "ao188_nominal";
    case CACAOSIM_ADC_PRESET_SNELL_DOUBLET:
        return "snell_doublet";
    default:
        return "ao188";
    }
}

int32_t cacaosim_adc_parse_mode(const char *s)
{
    if (s == NULL)
    {
        return -1;
    }
    if (str_eq_ci(s, "retracted") || str_eq_ci(s, "off") || str_eq_ci(s, "bypass") ||
        str_eq_ci(s, "0"))
    {
        return CACAOSIM_ADC_MODE_RETRACTED;
    }
    if (str_eq_ci(s, "manual") || str_eq_ci(s, "1"))
    {
        return CACAOSIM_ADC_MODE_MANUAL;
    }
    if (str_eq_ci(s, "lut") || str_eq_ci(s, "track") || str_eq_ci(s, "auto") || str_eq_ci(s, "2"))
    {
        return CACAOSIM_ADC_MODE_LUT;
    }
    if (str_eq_ci(s, "shm") || str_eq_ci(s, "3"))
    {
        return CACAOSIM_ADC_MODE_SHM;
    }
    if (str_eq_ci(s, "lut_plus_shm") || str_eq_ci(s, "lut+shm") || str_eq_ci(s, "4"))
    {
        return CACAOSIM_ADC_MODE_LUT_PLUS_SHM;
    }
    return -1;
}

const char *cacaosim_adc_mode_name(int32_t mode)
{
    switch (mode)
    {
    case CACAOSIM_ADC_MODE_RETRACTED:
        return "retracted";
    case CACAOSIM_ADC_MODE_MANUAL:
        return "manual";
    case CACAOSIM_ADC_MODE_LUT:
        return "lut";
    case CACAOSIM_ADC_MODE_SHM:
        return "shm";
    case CACAOSIM_ADC_MODE_LUT_PLUS_SHM:
        return "lut_plus_shm";
    default:
        return "lut";
    }
}

int32_t cacaosim_adc_parse_opt_band(const char *s)
{
    if (s == NULL)
    {
        return -1;
    }
    if (str_eq_ci(s, "h") || str_eq_ci(s, "h-band") || str_eq_ci(s, "0"))
    {
        return CACAOSIM_ADC_OPT_H;
    }
    if (str_eq_ci(s, "whole") || str_eq_ci(s, "broadband") || str_eq_ci(s, "all") ||
        str_eq_ci(s, "1"))
    {
        return CACAOSIM_ADC_OPT_WHOLE;
    }
    if (str_eq_ci(s, "j") || str_eq_ci(s, "j-band") || str_eq_ci(s, "2"))
    {
        return CACAOSIM_ADC_OPT_J;
    }
    if (str_eq_ci(s, "z") || str_eq_ci(s, "z-band") || str_eq_ci(s, "3"))
    {
        return CACAOSIM_ADC_OPT_Z;
    }
    if (str_eq_ci(s, "kp") || str_eq_ci(s, "k'") || str_eq_ci(s, "kprime") || str_eq_ci(s, "4"))
    {
        return CACAOSIM_ADC_OPT_KP;
    }
    if (str_eq_ci(s, "k") || str_eq_ci(s, "k-band") || str_eq_ci(s, "5"))
    {
        return CACAOSIM_ADC_OPT_K;
    }
    if (str_eq_ci(s, "vis") || str_eq_ci(s, "apds") || str_eq_ci(s, "apd") || str_eq_ci(s, "6"))
    {
        return CACAOSIM_ADC_OPT_VIS;
    }
    if (str_eq_ci(s, "custom") || str_eq_ci(s, "7"))
    {
        return CACAOSIM_ADC_OPT_CUSTOM;
    }
    return -1;
}

const char *cacaosim_adc_opt_band_name(int32_t opt_band)
{
    switch (opt_band)
    {
    case CACAOSIM_ADC_OPT_H:
        return "H";
    case CACAOSIM_ADC_OPT_WHOLE:
        return "whole";
    case CACAOSIM_ADC_OPT_J:
        return "J";
    case CACAOSIM_ADC_OPT_Z:
        return "z";
    case CACAOSIM_ADC_OPT_KP:
        return "K'";
    case CACAOSIM_ADC_OPT_K:
        return "K";
    case CACAOSIM_ADC_OPT_VIS:
        return "vis";
    case CACAOSIM_ADC_OPT_CUSTOM:
        return "custom";
    default:
        return "H";
    }
}

double cacaosim_adc_glass_n(
    const char   *name,
    const double *custom_sellmeier,
    double        lam_um)
{
    if (lam_um < 0.30)
    {
        lam_um = 0.30;
    }
    double l2 = lam_um * lam_um;

    if (custom_sellmeier != NULL && custom_sellmeier[0] > 0.0)
    {
        double n2 = 1.0 + (custom_sellmeier[0] * l2) / (l2 - custom_sellmeier[3]) +
                    (custom_sellmeier[1] * l2) / (l2 - custom_sellmeier[4]) +
                    (custom_sellmeier[2] * l2) / (l2 - custom_sellmeier[5]);
        return (n2 > 1.0) ? sqrt(n2) : 1.5;
    }

#ifdef HAVE_OPTICSMATERIALS
    if (name != NULL && name[0] != '\0')
    {
        int code = OPTICSMATERIALS_code(name);
        if (code >= 0)
        {
            double n = OPTICSMATERIALS_n(code, lam_um * 1.0e-6);
            if (n > 1.0)
            {
                return n;
            }
        }
    }
#endif

    const size_t n_tbl = sizeof(k_adc_glasses) / sizeof(k_adc_glasses[0]);
    for (size_t i = 0; i < n_tbl; i++)
    {
        if (str_eq_ci(name, k_adc_glasses[i].name))
        {
            const adc_glass_entry_t *g  = &k_adc_glasses[i];
            double                   n2 = 1.0 + (g->b1 * l2) / (l2 - g->c1) +
                        (g->b2 * l2) / (l2 - g->c2) + (g->b3 * l2) / (l2 - g->c3);
            return (n2 > 1.0) ? sqrt(n2) : 1.5;
        }
    }
    return 1.5;
}

/* 3D vector Snell refraction at interface with unit normal pointing AGAINST incoming ray (+Z) */
static void snell_refract_3d(
    double       kx,
    double       ky,
    double       kz,
    double       nx,
    double       ny,
    double       nz,
    double       n_in,
    double       n_out,
    double      *ox,
    double      *oy,
    double      *oz)
{
    double eta = n_in / n_out;
    /* Orient normal against ray so c1 = -(k . n) >= 0 */
    double dot = kx * nx + ky * ny + kz * nz;
    if (dot > 0.0)
    {
        nx  = -nx;
        ny  = -ny;
        nz  = -nz;
        dot = -dot;
    }
    double c1  = -dot;
    double k_t = 1.0 - eta * eta * (1.0 - c1 * c1);
    double c2  = (k_t > 0.0) ? sqrt(k_t) : 0.0;
    double fac = eta * c1 - c2;

    double rx  = eta * kx + fac * nx;
    double ry  = eta * ky + fac * ny;
    double rz  = eta * kz + fac * nz;
    double len = sqrt(rx * rx + ry * ry + rz * rz);
    if (len > 1e-12)
    {
        rx /= len;
        ry /= len;
        rz /= len;
    }
    *ox = rx;
    *oy = ry;
    *oz = rz;
}

/* Trace an axial ray (0,0,1) through two counter-rotating cemented doublets */
static void trace_snell_doublet_pair(
    const cacaosim_adc_cfg_t *cfg,
    double                    lam_um,
    double                    theta1_deg,
    double                    theta2_deg,
    double                   *tx_rad,
    double                   *ty_rad)
{
    double n1 = cacaosim_adc_glass_n(cfg->glass1, cfg->g1_sellmeier, lam_um);
    double n2 = cacaosim_adc_glass_n(cfg->glass2, cfg->g2_sellmeier, lam_um);

    double a1 = (double) cfg->alpha1_deg * ADC_DEG2RAD;
    double a2 = (double) cfg->alpha2_deg * ADC_DEG2RAD;

    /*
     * Each cemented doublet consists of:
     *   - Surface 1 (air -> glass1): tilted by +a1 along prism axis u(theta)
     *   - Surface 2 (glass1 -> glass2): normal along +Z (flat cemented interface)
     *   - Surface 3 (glass2 -> air): tilted by -a2 along prism axis u(theta)
     * (Equivalently, prism 2 is reversed air->glass2->glass1->air so at theta2 = theta1 + 180 deg
     * the pair is strictly symmetric and produces exact zero dispersion and zero deviation!)
     */
    double th1 = theta1_deg * ADC_DEG2RAD;
    double th2 = theta2_deg * ADC_DEG2RAD;
    double u1x = sin(th1), u1y = cos(th1);
    double u2x = sin(th2), u2y = cos(th2);

    double kx = 0.0, ky = 0.0, kz = 1.0;

    /* Prism 1: air -> glass1 (wedge a1) -> glass2 (flat) -> air (wedge a2) */
    snell_refract_3d(kx, ky, kz, sin(a1) * u1x, sin(a1) * u1y, cos(a1), 1.0, n1, &kx, &ky, &kz);
    snell_refract_3d(kx, ky, kz, 0.0, 0.0, 1.0, n1, n2, &kx, &ky, &kz);
    snell_refract_3d(kx, ky, kz, -sin(a2) * u1x, -sin(a2) * u1y, cos(a2), n2, 1.0, &kx, &ky, &kz);

    /* Prism 2 (reversed orientation): air -> glass2 (wedge a2) -> glass1 (flat) -> air (wedge a1) */
    snell_refract_3d(kx, ky, kz, sin(a2) * u2x, sin(a2) * u2y, cos(a2), 1.0, n2, &kx, &ky, &kz);
    snell_refract_3d(kx, ky, kz, 0.0, 0.0, 1.0, n2, n1, &kx, &ky, &kz);
    snell_refract_3d(kx, ky, kz, -sin(a1) * u2x, -sin(a1) * u2y, cos(a1), n1, 1.0, &kx, &ky, &kz);

    double mag = (cfg->beam_mag > 1e-3f) ? (double) cfg->beam_mag : 88.333;
    /* Project collimated-beam exit direction back onto sky angle (divided by pupil magnification) */
    *tx_rad = atan2(kx, kz) / mag;
    *ty_rad = atan2(ky, kz) / mag;
}

static double egner_dmax_rad(
    const double *c_mas,
    double        lam_um)
{
    if (lam_um < 0.40)
    {
        lam_um = 0.40;
    }
    if (lam_um > 2.40)
    {
        lam_um = 2.40;
    }
    const double l0   = 0.70;
    double       l2   = lam_um * lam_um;
    double       l02  = l0 * l0;
    double       il2  = 1.0 / l2;
    double       il02 = 1.0 / l02;
    double       dmas = c_mas[0] * (il2 - il02) + c_mas[1] * (il2 * il2 - il02 * il02) +
                        c_mas[2] * (l2 - l02) + c_mas[3] * (l2 * l2 - l02 * l02);
    return dmas * ADC_MAS2RAD;
}

void cacaosim_adc_eval_deflection_rad(
    const cacaosim_adc_cfg_t *cfg,
    double                    lam_um,
    double                    theta1_deg,
    double                    theta2_deg,
    double                   *tx_rad,
    double                   *ty_rad)
{
    if (cfg == NULL)
    {
        if (tx_rad != NULL)
        {
            *tx_rad = 0.0;
        }
        if (ty_rad != NULL)
        {
            *ty_rad = 0.0;
        }
        return;
    }

    double t1 = theta1_deg + (double) cfg->unit_rot_err_deg;
    double t2 = theta2_deg + (double) cfg->unit_rot_err_deg + (double) cfg->prism_offset_deg;

    if (cfg->preset == CACAOSIM_ADC_PRESET_SNELL_DOUBLET)
    {
        double raw_x = 0.0, raw_y = 0.0;
        double ref_x = 0.0, ref_y = 0.0;
        trace_snell_doublet_pair(cfg, lam_um, t1, t2, &raw_x, &raw_y);
        trace_snell_doublet_pair(cfg, 0.70, t1, t2, &ref_x, &ref_y);
        /* Note: keep relative chromatic dispersion anchored at 0.70 um + residual zero-dev */
        (void) ref_x;
        (void) ref_y;
        if (tx_rad != NULL)
        {
            *tx_rad = raw_x;
        }
        if (ty_rad != NULL)
        {
            *ty_rad = raw_y;
        }
        return;
    }

    /*
     * Calibrated Egner (2009) counter-rotating pair:
     *   At theta1 = pa - 90 + psi, theta2 = pa + 90 - psi:
     *   0.5 * [u(theta1) + u(theta2)] = sin(psi) * e(pa),
     *   where u(t) = (sin t, cos t).
     *   Applying -D_max(lam) * 0.5 * [u(t1) + u(t2)] cancels positive atmospheric dispersion!
     */
    const double *c_mas = (cfg->preset == CACAOSIM_ADC_PRESET_AO188_NOMINAL)
                              ? k_egner_nominal_mas
                              : k_egner_measured_mas;
    double        dmax  = egner_dmax_rad(c_mas, lam_um);
    double        r1    = t1 * ADC_DEG2RAD;
    double        r2    = t2 * ADC_DEG2RAD;
    double        vx    = 0.5 * (sin(r1) + sin(r2));
    double        vy    = 0.5 * (cos(r1) + cos(r2));

    if (tx_rad != NULL)
    {
        *tx_rad = -dmax * vx;
    }
    if (ty_rad != NULL)
    {
        *ty_rad = -dmax * vy;
    }
}

/* Fill Egner 5-wavelength grid + [1,1,2,1,1] weights for a band selector */
static int adc_opt_band_grid(
    const cacaosim_adc_cfg_t *cfg,
    double                   *wls_out,
    double                   *wts_out)
{
    static const double w5[5] = { 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };

    if (cfg->opt_band == CACAOSIM_ADC_OPT_WHOLE)
    {
        /* Egner Table 1 IRCS + APDs simultaneous broadband optimization (7 bands x 5 = 35 pts) */
        static const double bands[7][5] = {
            { 0.997, 1.015, 1.033, 1.051, 1.070 }, /* z */
            { 1.170, 1.210, 1.250, 1.290, 1.330 }, /* J */
            { 1.480, 1.555, 1.630, 1.705, 1.780 }, /* H */
            { 1.480, 1.685, 1.890, 2.095, 2.300 }, /* H+K' */
            { 1.945, 2.033, 2.120, 2.208, 2.295 }, /* K' */
            { 2.030, 2.115, 2.200, 2.285, 2.370 }, /* K */
            { 0.450, 0.563, 0.700, 0.813, 0.900 }, /* APDs */
        };
        int idx = 0;
        for (int b = 0; b < 7; b++)
        {
            for (int i = 0; i < 5; i++)
            {
                wls_out[idx] = bands[b][i];
                wts_out[idx] = w5[i];
                idx++;
            }
        }
        return idx;
    }

    double l0 = 1.500, l1 = 1.780;
    switch (cfg->opt_band)
    {
    case CACAOSIM_ADC_OPT_H:
        l0 = 1.500;
        l1 = 1.780;
        break;
    case CACAOSIM_ADC_OPT_J:
        l0 = 1.175;
        l1 = 1.345;
        break;
    case CACAOSIM_ADC_OPT_Z:
        l0 = 0.970;
        l1 = 1.070;
        break;
    case CACAOSIM_ADC_OPT_KP:
        l0 = 1.985;
        l1 = 2.295;
        break;
    case CACAOSIM_ADC_OPT_K:
        l0 = 2.030;
        l1 = 2.370;
        break;
    case CACAOSIM_ADC_OPT_VIS:
        wls_out[0] = 0.450;
        wls_out[1] = 0.563;
        wls_out[2] = 0.700;
        wls_out[3] = 0.813;
        wls_out[4] = 0.900;
        for (int i = 0; i < 5; i++)
        {
            wts_out[i] = w5[i];
        }
        return 5;
    case CACAOSIM_ADC_OPT_CUSTOM:
        l0 = (cfg->opt_lam1_um > 0.35f) ? (double) cfg->opt_lam1_um : 0.65;
        l1 = (cfg->opt_lam2_um > l0 + 0.01) ? (double) cfg->opt_lam2_um : (l0 + 0.25);
        break;
    default:
        break;
    }

    for (int i = 0; i < 5; i++)
    {
        wls_out[i] = l0 + (l1 - l0) * ((double) i / 4.0);
        wts_out[i] = w5[i];
    }
    return 5;
}

void cacaosim_adc_solve_lut(
    const cacaosim_adc_cfg_t      *cfg,
    const cacaosim_atm_disp_cfg_t *atm,
    double                        *out_th1_deg,
    double                        *out_th2_deg,
    double                        *out_psi_deg)
{
    double pa_deg = (atm != NULL ? (double) atm->disp_pa_deg : 0.0) +
                    (cfg != NULL ? (double) cfg->lut_pa_err_deg : 0.0);
    double psi_deg = 0.0;

    if (cfg != NULL && atm != NULL && atm->enable && atm->zenith_deg > 1e-6f)
    {
        /* Build LUT's believed atmospheric model (including optional P/T sensor bias) */
        cacaosim_atm_disp_cfg_t lut_atm = *atm;
        lut_atm.pressure_hpa += cfg->lut_p_err_hpa;
        lut_atm.temp_c += cfg->lut_t_err_c;

        /* Build LUT's believed ADC model (optionally nominal Table 10 even on measured prisms) */
        cacaosim_adc_cfg_t lut_cfg = *cfg;
        lut_cfg.prism_offset_deg   = 0.0f;
        lut_cfg.unit_rot_err_deg   = 0.0f;
        if (cfg->lut_use_nominal && cfg->preset == CACAOSIM_ADC_PRESET_AO188)
        {
            lut_cfg.preset = CACAOSIM_ADC_PRESET_AO188_NOMINAL;
        }

        double wls[40];
        double wts[40];
        int    npts    = adc_opt_band_grid(&lut_cfg, wls, wts);
        int    nbands  = (lut_cfg.opt_band == CACAOSIM_ADC_OPT_WHOLE) ? 7 : 1;
        double num_cov = 0.0;
        double den_var = 0.0;

        /*
         * Evaluate max-dispersion curve D_max(lam) along +Y (pa=0, psi=90 => th1=0, th2=0).
         * Optimal sin(psi) minimizes sum_b sum_i w_i * ((atm_i - mean_atm_b) - sin(psi)*(d_i - mean_d_b))^2
         * across each band b (matching Egner 2009 on-axis spot-size RMS minimization).
         */
        for (int b = 0; b < nbands; b++)
        {
            int    i0   = b * 5;
            int    n_b  = (nbands == 1) ? npts : 5;
            double a_b[5];
            double d_b[5];
            double ma = 0.0, md = 0.0;
            for (int i = 0; i < n_b; i++)
            {
                double lam = wls[i0 + i];
                double tx = 0.0, ty = 0.0;
                a_b[i] = cacaosim_atm_disp_rad(lam, &lut_atm);
                /* At th1=0, th2=0, eval_deflection returns -D_max(lam) along +Y -> negate */
                cacaosim_adc_eval_deflection_rad(&lut_cfg, lam, 0.0, 0.0, &tx, &ty);
                d_b[i] = -ty;
                ma += wts[i0 + i] * a_b[i];
                md += wts[i0 + i] * d_b[i];
            }
            for (int i = 0; i < n_b; i++)
            {
                double da = a_b[i] - ma;
                double dd = d_b[i] - md;
                num_cov += wts[i0 + i] * da * dd;
                den_var += wts[i0 + i] * dd * dd;
            }
        }

        double s_opt = (den_var > 1.0e-30) ? (num_cov / den_var) : 0.0;
        s_opt *= (1.0 + (double) cfg->lut_scale_err);
        if (s_opt > 1.0)
        {
            s_opt = 1.0;
        }
        if (s_opt < -1.0)
        {
            s_opt = -1.0;
        }
        psi_deg = asin(s_opt) * ADC_RAD2DEG;
    }

    if (out_th1_deg != NULL)
    {
        *out_th1_deg = pa_deg - 90.0 + psi_deg;
    }
    if (out_th2_deg != NULL)
    {
        *out_th2_deg = pa_deg + 90.0 - psi_deg;
    }
    if (out_psi_deg != NULL)
    {
        *out_psi_deg = psi_deg;
    }
}

static double wrap_delta_deg(double d)
{
    while (d > 180.0)
    {
        d -= 360.0;
    }
    while (d < -180.0)
    {
        d += 360.0;
    }
    return d;
}

/* Advance one rotational stage with Table 29 step quantization, slew-rate limit, and backlash */
static double advance_stage(
    double target_deg,
    double dt_s,
    int    ideal_mech,
    double step_deg,
    double max_slew_dps,
    double backlash_deg,
    double *cur_deg,
    int    *last_dir,
    double *bl_off_deg)
{
    if (ideal_mech)
    {
        *cur_deg    = target_deg;
        *bl_off_deg = 0.0;
        return target_deg;
    }

    /* Stepper quantization */
    double q_tgt = target_deg;
    if (step_deg > 1e-6)
    {
        q_tgt = round(target_deg / step_deg) * step_deg;
    }

    double diff = wrap_delta_deg(q_tgt - *cur_deg);
    if (max_slew_dps > 1e-6 && dt_s > 0.0)
    {
        double max_step = max_slew_dps * dt_s;
        if (diff > max_step)
        {
            diff = max_step;
        }
        else if (diff < -max_step)
        {
            diff = -max_step;
        }
    }

    int new_dir = (diff > 1e-7) ? 1 : ((diff < -1e-7) ? -1 : 0);
    if (new_dir != 0)
    {
        if (*last_dir != 0 && new_dir != *last_dir && backlash_deg > 0.0)
        {
            *bl_off_deg = -0.5 * backlash_deg * (double) new_dir;
        }
        *last_dir = new_dir;
    }
    *cur_deg += diff;
    return *cur_deg + *bl_off_deg;
}

typedef struct
{
    const cacaosim_adc_cfg_t *cfg;
    double                    th1_deg;
    double                    th2_deg;
} adc_fit_ctx_t;

static void adc_eval_cb(
    double  lam_um,
    void   *vctx,
    double *tx_rad,
    double *ty_rad)
{
    const adc_fit_ctx_t *c = (const adc_fit_ctx_t *) vctx;
    cacaosim_adc_eval_deflection_rad(c->cfg, lam_um, c->th1_deg, c->th2_deg, tx_rad, ty_rad);
}

void cacaosim_adc_init(
    cacaosim_adc_ctx_t            *ctx,
    const cacaosim_adc_cfg_t      *cfg,
    const cacaosim_atm_disp_cfg_t *atm)
{
    if (ctx == NULL)
    {
        return;
    }
    memset(ctx, 0, sizeof(*ctx));
    if (cfg != NULL)
    {
        ctx->cfg = *cfg;
    }
    else
    {
        cacaosim_adc_cfg_defaults(&ctx->cfg);
    }

    double th1 = (double) ctx->cfg.angle1_deg;
    double th2 = (double) ctx->cfg.angle2_deg;
    if (ctx->cfg.mode == CACAOSIM_ADC_MODE_LUT || ctx->cfg.mode == CACAOSIM_ADC_MODE_LUT_PLUS_SHM)
    {
        cacaosim_adc_solve_lut(&ctx->cfg, atm, &th1, &th2, NULL);
    }
    else if (ctx->cfg.mode == CACAOSIM_ADC_MODE_MANUAL && ctx->cfg.manual_use_psi)
    {
        double pa = (atm != NULL) ? (double) atm->disp_pa_deg : 0.0;
        th1       = pa - 90.0 + (double) ctx->cfg.manual_psi_deg;
        th2       = pa + 90.0 - (double) ctx->cfg.manual_psi_deg;
    }

    ctx->cur_th1_deg = th1;
    ctx->cur_th2_deg = th2;
    ctx->initialized = 1;
}

/* Spectral transmission curve of AR-coated cemented doublet (flat ~tput_base, mild UV/K roll) */
static float adc_bin_tput(float lam_um, float tput_base)
{
    float t = tput_base;
    if (lam_um < 0.50f)
    {
        float f = (0.50f - lam_um) / 0.10f;
        t *= (1.0f - 0.08f * f * f);
    }
    else if (lam_um > 2.10f)
    {
        float f = (lam_um - 2.10f) / 0.30f;
        t *= (1.0f - 0.06f * f * f);
    }
    if (t < 0.0f)
    {
        t = 0.0f;
    }
    if (t > 1.0f)
    {
        t = 1.0f;
    }
    return t;
}

void cacaosim_adc_step(
    cacaosim_adc_ctx_t            *ctx,
    double                         dt_s,
    const cacaosim_atm_disp_cfg_t *atm,
    const float                   *shm_cmd,
    const cacaosim_disp_t         *in_disp,
    const cacaosim_spec_t         *in_spec,
    cacaosim_disp_t               *out_disp,
    cacaosim_spec_t               *out_spec)
{
    if (ctx == NULL)
    {
        return;
    }
    if (!ctx->initialized)
    {
        cacaosim_adc_init(ctx, &ctx->cfg, atm);
    }

    const int deployed = (ctx->cfg.mode != CACAOSIM_ADC_MODE_RETRACTED) ? 1 : 0;
    double    tgt1     = ctx->cur_th1_deg;
    double    tgt2     = ctx->cur_th2_deg;

    if (deployed)
    {
        switch (ctx->cfg.mode)
        {
        case CACAOSIM_ADC_MODE_MANUAL:
            if (ctx->cfg.manual_use_psi)
            {
                double pa = (atm != NULL) ? (double) atm->disp_pa_deg : 0.0;
                tgt1      = pa - 90.0 + (double) ctx->cfg.manual_psi_deg;
                tgt2      = pa + 90.0 - (double) ctx->cfg.manual_psi_deg;
            }
            else
            {
                tgt1 = (double) ctx->cfg.angle1_deg;
                tgt2 = (double) ctx->cfg.angle2_deg;
            }
            break;

        case CACAOSIM_ADC_MODE_LUT:
            cacaosim_adc_solve_lut(&ctx->cfg, atm, &tgt1, &tgt2, NULL);
            break;

        case CACAOSIM_ADC_MODE_SHM:
            if (shm_cmd != NULL)
            {
                tgt1 = (double) shm_cmd[0];
                tgt2 = (double) shm_cmd[1];
            }
            else
            {
                tgt1 = (double) ctx->cfg.angle1_deg;
                tgt2 = (double) ctx->cfg.angle2_deg;
            }
            break;

        case CACAOSIM_ADC_MODE_LUT_PLUS_SHM:
        {
            double lut_psi = 0.0;
            cacaosim_adc_solve_lut(&ctx->cfg, atm, &tgt1, &tgt2, &lut_psi);
            if (shm_cmd != NULL)
            {
                /* shm_cmd = [d_psi_deg, d_pa_deg] differential closed-loop trim */
                double d_psi = (double) shm_cmd[0];
                double d_pa  = (double) shm_cmd[1];
                tgt1 += d_pa + d_psi;
                tgt2 += d_pa - d_psi;
            }
            break;
        }

        default:
            break;
        }
    }

    double eff1 = advance_stage(tgt1, dt_s, ctx->cfg.ideal_mech, (double) ctx->cfg.enc_step_deg,
                                (double) ctx->cfg.max_slew_dps, (double) ctx->cfg.backlash_deg,
                                &ctx->cur_th1_deg, &ctx->dir1, &ctx->bl_off1_deg);
    double eff2 = advance_stage(tgt2, dt_s, ctx->cfg.ideal_mech, (double) ctx->cfg.enc_step_deg,
                                (double) ctx->cfg.max_slew_dps, (double) ctx->cfg.backlash_deg,
                                &ctx->cur_th2_deg, &ctx->dir2, &ctx->bl_off2_deg);

    /* Rebuild cached ADC Chebyshev dispersion only when deployment or prism angles change */
    int need_refit = (!ctx->cache_valid || ctx->cache_deployed != deployed ||
                      fabs(eff1 - ctx->cache_th1_deg) > 1e-7 ||
                      fabs(eff2 - ctx->cache_th2_deg) > 1e-7);
    if (need_refit)
    {
        if (!deployed)
        {
            cacaosim_disp_zero(&ctx->cached_adc_disp);
        }
        else
        {
            adc_fit_ctx_t fctx = { .cfg = &ctx->cfg, .th1_deg = eff1, .th2_deg = eff2 };
            cacaosim_disp_fit_fn(&ctx->cached_adc_disp, adc_eval_cb, &fctx);
        }
        ctx->cache_valid    = 1;
        ctx->cache_deployed = deployed;
        ctx->cache_th1_deg  = eff1;
        ctx->cache_th2_deg  = eff2;
    }

    /* Compose output dispersion: out = in_disp + adc_disp (preserve gen when steady) */
    uint64_t in_gen     = (in_disp != NULL) ? in_disp->gen : 0ULL;
    int      need_sum   = (need_refit || !ctx->cache_valid || in_gen != ctx->cache_in_gen ||
                           ctx->cached_out_disp.gen == 0ULL);
    if (need_sum)
    {
        cacaosim_disp_add(&ctx->cached_out_disp, in_disp, &ctx->cached_adc_disp);
        ctx->cache_in_gen = in_gen;
    }
    cacaosim_disp_t *dst_d = (out_disp != NULL) ? out_disp : &ctx->cached_out_disp;
    if (out_disp != NULL)
    {
        *out_disp = ctx->cached_out_disp;
    }

    /* Apply optional spectral throughput attenuation when ADC is deployed in beam */
    if (in_spec != NULL && out_spec != NULL)
    {
        if (deployed && ctx->cfg.tput_enable)
        {
            for (int b = 0; b < CACAOSIM_NSPEC; b++)
            {
                float lam         = cacaosim_spec_bin_lambda(b);
                out_spec->ph_s[b] = in_spec->ph_s[b] * adc_bin_tput(lam, ctx->cfg.tput_base);
            }
        }
        else if (out_spec != in_spec)
        {
            *out_spec = *in_spec;
        }
    }

    /* Update `[10 x 1]` live telemetry snapshot */
    double psi_deg = 0.5 * wrap_delta_deg(eff1 - eff2 + 180.0);
    double pa_deg  = (atm != NULL) ? (double) atm->disp_pa_deg : 0.0;
    double z_deg   = (atm != NULL) ? (double) atm->zenith_deg : 0.0;

    ctx->state_vec[0] = (float) eff1;
    ctx->state_vec[1] = (float) eff2;
    ctx->state_vec[2] = (float) psi_deg;
    ctx->state_vec[3] = (float) pa_deg;
    ctx->state_vec[4] = (float) z_deg;
    ctx->state_vec[5] = (float) cacaosim_disp_band_span_mas(dst_d, 0.675, 0.825); /* Vis [mas] */
    ctx->state_vec[6] = (float) cacaosim_disp_band_span_mas(dst_d, 1.130, 1.310); /* J [mas] */
    ctx->state_vec[7] = (float) cacaosim_disp_band_span_mas(dst_d, 1.500, 1.780); /* H [mas] */
    ctx->state_vec[8] = (float) cacaosim_disp_shift_mas(dst_d, 0.750, 1.650);     /* Vis-H [mas] */
    ctx->state_vec[9] = (float) deployed;
}
