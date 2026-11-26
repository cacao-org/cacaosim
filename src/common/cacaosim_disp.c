// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_disp.c
 * @brief   Ciddor/Seidelmann atmospheric refraction, Chebyshev-12 dispersion math, and SHM I/O
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_disp.h"
#include "cacaosim_types.h"

#define CACAOSIM_RAD2MAS 206264806.24709636
#define CACAOSIM_DEG2RAD (3.14159265358979323846 / 180.0)
#define CACAOSIM_PI_D    3.14159265358979323846

/* Wavenumber domain bounds s = 1 / lambda [1/um] mapped linearly onto Chebyshev u in [-1, +1] */
static const double k_smin = 1.0 / CACAOSIM_DISP_LMAX_UM;
static const double k_smax = 1.0 / CACAOSIM_DISP_LMIN_UM;

static inline double lam_to_u(double lam_um)
{
    if (lam_um < CACAOSIM_DISP_LMIN_UM)
    {
        lam_um = CACAOSIM_DISP_LMIN_UM;
    }
    if (lam_um > CACAOSIM_DISP_LMAX_UM)
    {
        lam_um = CACAOSIM_DISP_LMAX_UM;
    }
    double s = 1.0 / lam_um;
    return (2.0 * s - (k_smin + k_smax)) / (k_smax - k_smin);
}

static inline double u_to_lam(double u)
{
    double s = 0.5 * ((k_smax - k_smin) * u + (k_smin + k_smax));
    return 1.0 / s;
}

void cacaosim_atm_disp_cfg_defaults(cacaosim_atm_disp_cfg_t *cfg)
{
    if (cfg == NULL)
    {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->enable       = 1;
    cfg->zenith_deg   = 0.0f;
    cfg->disp_pa_deg  = 0.0f;
    cfg->site_alt_m   = 4205.0f;
    cfg->pressure_hpa = 615.0f;
    cfg->temp_c       = 2.0f;
    cfg->rh_pct       = 20.0f;
    cfg->ref_lam_um   = 1.65f;
    cfg->sphere_corr  = 1;
}

/**
 * cacaosim_air_n_minus_1() - Ciddor (1996, Appl. Opt. 35, 1566) moist-air refractivity
 */
double cacaosim_air_n_minus_1(
    double lam_um,
    double pressure_hpa,
    double temp_c,
    double rh_pct)
{
    if (lam_um < 0.20)
    {
        lam_um = 0.20;
    }
    if (pressure_hpa <= 0.0)
    {
        return 0.0;
    }

    double sigma2 = 1.0 / (lam_um * lam_um); /* [um^-2] */
    double T_k    = temp_c + 273.15;
    if (T_k < 150.0)
    {
        T_k = 150.0;
    }
    double P_pa = pressure_hpa * 100.0;

    /* Standard dry air (15 C, 1013.25 hPa, 450 ppm CO2) refractivity * 1e8 (Ciddor eq. 1-2) */
    const double k0 = 238.0185;
    const double k1 = 5792105.0;
    const double k2 = 57.362;
    const double k3 = 167917.0;
    const double xc = 415.0; /* Maunakea CO2 [ppm] */

    double n_as_m1 = (k1 / (k0 - sigma2) + k3 / (k2 - sigma2)) * 1.0e-8;
    double n_ax_m1 = n_as_m1 * (1.0 + 0.534e-6 * (xc - 450.0));

    /* Standard water vapor refractivity (Ciddor eq. 3) */
    const double cf = 1.022;
    const double w0 = 295.235;
    const double w1 = 2.6422;
    const double w2 = -0.032380;
    const double w3 = 0.004028;
    double       n_ws_m1 =
        cf * (w0 + sigma2 * (w1 + sigma2 * (w2 + sigma2 * w3))) * 1.0e-8;

    /* Saturation vapor pressure (Davis 1992 / Ciddor Appendix A) */
    double rh = rh_pct;
    if (rh < 0.0)
    {
        rh = 0.0;
    }
    if (rh > 100.0)
    {
        rh = 100.0;
    }
    double svp = exp(1.2378847e-5 * T_k * T_k - 1.9121316e-2 * T_k + 33.93711047 -
                     6.3431645e3 / T_k);
    double f_enh = 1.00062 + 3.14e-8 * P_pa + 5.6e-7 * temp_c * temp_c;
    double p_v   = (rh * 0.01) * f_enh * svp;
    if (p_v > P_pa)
    {
        p_v = P_pa;
    }
    double x_w = p_v / P_pa;

    /* Compressibility factor Z (CIPM-81 / Ciddor eq. 12) */
    double a0 = 1.58123e-6;
    double a1 = -2.9331e-8;
    double a2 = 1.1043e-10;
    double b0 = 5.707e-6;
    double b1 = -2.051e-8;
    double c0 = 1.9898e-4;
    double c1 = -2.376e-6;
    double d0 = 1.83e-11;
    double e0 = -0.765e-8;
    double pt = P_pa / T_k;
    double Z  = 1.0 - pt * (a0 + a1 * temp_c + a2 * temp_c * temp_c +
                            (b0 + b1 * temp_c) * x_w + (c0 + c1 * temp_c) * x_w * x_w) +
                pt * pt * (d0 + e0 * x_w * x_w);

    /* Molar masses [kg/mol] & gas constant */
    const double R_gas = 8.314510;
    const double M_a   = 1.0e-3 * (28.9635 + 12.011e-6 * (xc - 400.0));
    const double M_w   = 0.018015;

    /* Reference densities of standard dry air (rho_axs) and water vapor (rho_ws) */
    const double Z_a0    = 0.9995922115;
    const double Z_w0    = 0.999621;
    double       rho_axs = (101325.0 * M_a) / (Z_a0 * R_gas * 288.15);
    double       rho_ws  = (1333.0 * M_w) / (Z_w0 * R_gas * 293.15);

    double rho_a = ((P_pa * M_a) / (Z * R_gas * T_k)) * (1.0 - x_w);
    double rho_w = ((P_pa * M_w) / (Z * R_gas * T_k)) * x_w;

    return (rho_a / rho_axs) * n_ax_m1 + (rho_w / rho_ws) * n_ws_m1;
}

double cacaosim_atm_disp_rad(
    double                         lam_um,
    const cacaosim_atm_disp_cfg_t *cfg)
{
    if (cfg == NULL || !cfg->enable || cfg->zenith_deg <= 1e-6f)
    {
        return 0.0;
    }

    double z_deg = (double) cfg->zenith_deg;
    if (z_deg > 80.0)
    {
        z_deg = 80.0;
    }
    double z_rad  = z_deg * CACAOSIM_DEG2RAD;
    double tan_z  = tan(z_rad);
    double factor = tan_z;
    if (cfg->sphere_corr)
    {
        /* Spherical-atmosphere correction (Seidelmann 1992 / Smart): beta = H_scale / R_earth */
        const double beta = 0.001254;
        factor            = tan_z * (1.0 - beta * (1.0 + tan_z * tan_z));
    }

    double ref_lam = (cfg->ref_lam_um > 0.1f) ? (double) cfg->ref_lam_um : 1.65;
    double n_lam   = cacaosim_air_n_minus_1(lam_um, cfg->pressure_hpa, cfg->temp_c, cfg->rh_pct);
    double n_ref   = cacaosim_air_n_minus_1(ref_lam, cfg->pressure_hpa, cfg->temp_c, cfg->rh_pct);
    return (n_lam - n_ref) * factor;
}

void cacaosim_disp_zero(cacaosim_disp_t *d)
{
    if (d == NULL)
    {
        return;
    }
    uint64_t next_gen = d->gen + 1ULL;
    memset(d, 0, sizeof(*d));
    d->active = 0;
    d->gen    = next_gen;
}

void cacaosim_disp_fit_fn(
    cacaosim_disp_t        *d,
    cacaosim_disp_eval_fn_t fn,
    void                   *ctx)
{
    if (d == NULL || fn == NULL)
    {
        return;
    }

    const int N = CACAOSIM_DISP_NCOEFF;
    double    fx[CACAOSIM_DISP_NCOEFF];
    double    fy[CACAOSIM_DISP_NCOEFF];
    double    max_abs = 0.0;

    /* Evaluate exact callback at the 12 Chebyshev-Gauss nodes u_j = cos((j + 0.5) * pi / N) */
    for (int j = 0; j < N; j++)
    {
        double u   = cos(CACAOSIM_PI_D * ((double) j + 0.5) / (double) N);
        double lam = u_to_lam(u);
        double tx  = 0.0;
        double ty  = 0.0;
        fn(lam, ctx, &tx, &ty);
        fx[j] = tx;
        fy[j] = ty;
        if (fabs(tx) > max_abs)
        {
            max_abs = fabs(tx);
        }
        if (fabs(ty) > max_abs)
        {
            max_abs = fabs(ty);
        }
    }

    uint64_t next_gen = d->gen + 1ULL;
    if (max_abs < 1.0e-15)
    {
        memset(d, 0, sizeof(*d));
        d->active = 0;
        d->gen    = next_gen;
        return;
    }

    /* Discrete Chebyshev Transform (DCT-II): c_k = (2/N) sum_j f(u_j) cos(k*(j+0.5)*pi/N) */
    const double inv_n = 2.0 / (double) N;
    for (int k = 0; k < N; k++)
    {
        double sx = 0.0;
        double sy = 0.0;
        for (int j = 0; j < N; j++)
        {
            double ang = CACAOSIM_PI_D * (double) k * ((double) j + 0.5) / (double) N;
            double w   = cos(ang);
            sx += fx[j] * w;
            sy += fy[j] * w;
        }
        double scale = (k == 0) ? (0.5 * inv_n) : inv_n;
        d->cx[k]     = sx * scale;
        d->cy[k]     = sy * scale;
    }

    d->active = 1;
    d->gen    = next_gen;
}

typedef struct
{
    const cacaosim_atm_disp_cfg_t *cfg;
    double                         ex;
    double                         ey;
} atm_fit_ctx_t;

static void atm_eval_cb(
    double  lam_um,
    void   *vctx,
    double *tx_rad,
    double *ty_rad)
{
    const atm_fit_ctx_t *c = (const atm_fit_ctx_t *) vctx;
    double               r = cacaosim_atm_disp_rad(lam_um, c->cfg);
    *tx_rad                = r * c->ex;
    *ty_rad                = r * c->ey;
}

void cacaosim_disp_set_atm(
    cacaosim_disp_t               *d,
    const cacaosim_atm_disp_cfg_t *cfg)
{
    if (d == NULL)
    {
        return;
    }
    if (cfg == NULL || !cfg->enable || cfg->zenith_deg <= 1e-6f)
    {
        cacaosim_disp_zero(d);
        return;
    }
    /* PA convention matching milkatmturb: PA=0 -> +Y (zenith up), PA=90 -> +X */
    double        pa  = (double) cfg->disp_pa_deg * CACAOSIM_DEG2RAD;
    atm_fit_ctx_t ctx = {
        .cfg = cfg,
        .ex  = sin(pa),
        .ey  = cos(pa),
    };
    cacaosim_disp_fit_fn(d, atm_eval_cb, &ctx);
}

void cacaosim_disp_add(
    cacaosim_disp_t       *dst,
    const cacaosim_disp_t *a,
    const cacaosim_disp_t *b)
{
    if (dst == NULL)
    {
        return;
    }
    uint64_t next_gen = dst->gen + 1ULL;
    int      act_a    = (a != NULL && a->active) ? 1 : 0;
    int      act_b    = (b != NULL && b->active) ? 1 : 0;
    if (!act_a && !act_b)
    {
        memset(dst, 0, sizeof(*dst));
        dst->active = 0;
        dst->gen    = next_gen;
        return;
    }
    double max_c = 0.0;
    for (int k = 0; k < CACAOSIM_DISP_NCOEFF; k++)
    {
        double vx = (act_a ? a->cx[k] : 0.0) + (act_b ? b->cx[k] : 0.0);
        double vy = (act_a ? a->cy[k] : 0.0) + (act_b ? b->cy[k] : 0.0);
        dst->cx[k] = vx;
        dst->cy[k] = vy;
        if (fabs(vx) > max_c)
        {
            max_c = fabs(vx);
        }
        if (fabs(vy) > max_c)
        {
            max_c = fabs(vy);
        }
    }
    dst->active = (max_c >= 1.0e-15) ? 1 : 0;
    dst->gen    = next_gen;
}

void cacaosim_disp_apply_reg(
    const cacaosim_disp_t *in,
    const cacaosim_reg_t  *reg,
    cacaosim_disp_t       *out)
{
    if (out == NULL)
    {
        return;
    }
    uint64_t next_gen = out->gen + 1ULL;
    if (in == NULL || !in->active)
    {
        memset(out, 0, sizeof(*out));
        out->active = 0;
        out->gen    = next_gen;
        return;
    }
    if (cacaosim_reg_is_identity(reg))
    {
        if (out != in)
        {
            memcpy(out->cx, in->cx, sizeof(out->cx));
            memcpy(out->cy, in->cy, sizeof(out->cy));
        }
        out->active = 1;
        out->gen    = next_gen;
        return;
    }

    double sc   = (reg->scale > 1e-4f) ? (double) reg->scale : 1.0;
    double invs = 1.0 / sc;
    double rad  = (double) reg->rot_deg * CACAOSIM_DEG2RAD;
    double cs   = cos(rad) * invs;
    double sn   = sin(rad) * invs;

    for (int k = 0; k < CACAOSIM_DISP_NCOEFF; k++)
    {
        double vx  = in->cx[k];
        double vy  = in->cy[k];
        out->cx[k] = cs * vx - sn * vy;
        out->cy[k] = sn * vx + cs * vy;
    }
    out->active = 1;
    out->gen    = next_gen;
}

void MILK_HOT cacaosim_disp_eval_rad(
    const cacaosim_disp_t *d,
    double                 lam_um,
    double                *tx_rad,
    double                *ty_rad)
{
    if (d == NULL || !d->active)
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

    /* Clenshaw's recurrence for sum_{k=0}^{N-1} c_k T_k(u) */
    double u    = lam_to_u(lam_um);
    double twoU = 2.0 * u;
    double bx1  = 0.0;
    double bx2  = 0.0;
    double by1  = 0.0;
    double by2  = 0.0;

    for (int k = CACAOSIM_DISP_NCOEFF - 1; k >= 1; k--)
    {
        double bx0 = twoU * bx1 - bx2 + d->cx[k];
        bx2        = bx1;
        bx1        = bx0;
        double by0 = twoU * by1 - by2 + d->cy[k];
        by2        = by1;
        by1        = by0;
    }

    if (tx_rad != NULL)
    {
        *tx_rad = u * bx1 - bx2 + d->cx[0];
    }
    if (ty_rad != NULL)
    {
        *ty_rad = u * by1 - by2 + d->cy[0];
    }
}

void cacaosim_disp_eval_slopes(
    const cacaosim_disp_t *d,
    const float           *lam_um,
    uint32_t               nlam,
    float                  tel_diam_m,
    uint32_t               nwf,
    float                  pup_diam,
    float                 *out_sx,
    float                 *out_sy)
{
    if (out_sx == NULL || out_sy == NULL || nlam == 0u)
    {
        return;
    }
    if (d == NULL || !d->active || lam_um == NULL || nwf == 0u)
    {
        memset(out_sx, 0, (size_t) nlam * sizeof(float));
        memset(out_sy, 0, (size_t) nlam * sizeof(float));
        return;
    }

    double diam_m = (tel_diam_m > 0.01f) ? (double) tel_diam_m : 8.0;
    double pfrac  = (pup_diam > 1e-4f) ? (double) pup_diam : 1.0;
    /* Physical pupil pixel pitch [um/pixel] = (D_tel * 1e6) / (nwf * pup_diam) */
    double um_per_pix = (diam_m * 1.0e6) / ((double) nwf * pfrac);

    for (uint32_t c = 0u; c < nlam; c++)
    {
        double tx = 0.0;
        double ty = 0.0;
        cacaosim_disp_eval_rad(d, (double) lam_um[c], &tx, &ty);
        out_sx[c] = (float) (tx * um_per_pix);
        out_sy[c] = (float) (ty * um_per_pix);
    }
}

double cacaosim_disp_band_span_mas(
    const cacaosim_disp_t *d,
    double                 lam0_um,
    double                 lam1_um)
{
    if (d == NULL || !d->active || lam1_um <= lam0_um)
    {
        return 0.0;
    }
    const int npts = 17;
    double    xs[17];
    double    ys[17];
    for (int i = 0; i < npts; i++)
    {
        double lam = lam0_um + (lam1_um - lam0_um) * ((double) i / (double) (npts - 1));
        cacaosim_disp_eval_rad(d, lam, &xs[i], &ys[i]);
    }
    double max_d2 = 0.0;
    for (int i = 0; i < npts; i++)
    {
        for (int j = i + 1; j < npts; j++)
        {
            double dx = xs[j] - xs[i];
            double dy = ys[j] - ys[i];
            double d2 = dx * dx + dy * dy;
            if (d2 > max_d2)
            {
                max_d2 = d2;
            }
        }
    }
    return sqrt(max_d2) * CACAOSIM_RAD2MAS;
}

double cacaosim_disp_shift_mas(
    const cacaosim_disp_t *d,
    double                 lam_a_um,
    double                 lam_b_um)
{
    if (d == NULL || !d->active)
    {
        return 0.0;
    }
    double xa = 0.0, ya = 0.0, xb = 0.0, yb = 0.0;
    cacaosim_disp_eval_rad(d, lam_a_um, &xa, &ya);
    cacaosim_disp_eval_rad(d, lam_b_um, &xb, &yb);
    return hypot(xb - xa, yb - ya) * CACAOSIM_RAD2MAS;
}

IMGID cacaosim_beam_shm_create(
    const char *name,
    uint32_t    nwf,
    uint32_t    nz)
{
    IMGID img;
    memset(&img, 0, sizeof(img));
    img.ID = -1;
    if (name == NULL || name[0] == '\0' || nwf == 0u || nz < 2u)
    {
        return img;
    }

    img = stream_connect(name);
    if (img.ID != -1)
    {
        if (img.md != NULL && img.md->naxis == 3 && img.md->size[0] == nwf &&
            img.md->size[1] == nwf && img.md->size[2] == nz &&
            img.md->datatype == _DATATYPE_FLOAT && img.md->NBkw >= CACAOSIM_BEAM_SHM_NBKW)
        {
            return img;
        }
        delete_image_ID(name, DELETE_IMAGE_ERRMODE_WARNING);
        img.ID = -1;
    }

    uint32_t dims[3] = { nwf, nwf, nz };
    create_image_ID(name, 3, dims, _DATATYPE_FLOAT, 1, CACAOSIM_BEAM_SHM_NBKW, 0, &img.ID);
    if (img.ID != -1 && dcimg[img.ID].md != NULL)
    {
        imageID id    = img.ID;
        img.im        = &dcimg[id];
        img.md        = dcimg[id].md;
        img.createcnt = dcimg[id].createcnt;
        imgid_update_creationparams(&img);
    }
    else
    {
        img.ID = -1;
        img.im = NULL;
        img.md = NULL;
    }
    return img;
}

void MILK_HOT cacaosim_beam_shm_write_disp(
    IMAGE                 *im,
    const cacaosim_disp_t *d)
{
    const int base = CACAOSIM_BEAM_KW_DISP_BASE;
    const int need = base + 1 + 2 * CACAOSIM_DISP_NCOEFF; /* 39 + 25 = 64 */
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < need)
    {
        return;
    }

    /* One-time lazy keyword name initialization at slots kw[39..63] */
    if (im->kw[base].type != 'L' || strcmp(im->kw[base].name, "CSIM_DSP") != 0)
    {
        snprintf(im->kw[base].name, KEYWORD_MAX_STRING, "CSIM_DSP");
        im->kw[base].type = 'L';
        snprintf(im->kw[base].comment, KEYWORD_MAX_COMMENT, "Chromatic dispersion active (0/1)");
        for (int k = 0; k < CACAOSIM_DISP_NCOEFF; k++)
        {
            int ix = base + 1 + k;
            int iy = base + 1 + CACAOSIM_DISP_NCOEFF + k;
            snprintf(im->kw[ix].name, KEYWORD_MAX_STRING, "CSIM_DX%02d", k);
            im->kw[ix].type = 'D';
            snprintf(im->kw[ix].comment, KEYWORD_MAX_COMMENT, "Cheb X coeff %d [rad]", k);
            snprintf(im->kw[iy].name, KEYWORD_MAX_STRING, "CSIM_DY%02d", k);
            im->kw[iy].type = 'D';
            snprintf(im->kw[iy].comment, KEYWORD_MAX_COMMENT, "Cheb Y coeff %d [rad]", k);
        }
    }

    if (d == NULL || !d->active)
    {
        im->kw[base].value.numl = 0;
        return;
    }

    im->kw[base].value.numl = 1;
    for (int k = 0; k < CACAOSIM_DISP_NCOEFF; k++)
    {
        im->kw[base + 1 + k].value.numf                        = d->cx[k];
        im->kw[base + 1 + CACAOSIM_DISP_NCOEFF + k].value.numf = d->cy[k];
    }
}

int cacaosim_beam_shm_read_disp(
    const IMAGE     *im,
    cacaosim_disp_t *d)
{
    if (d == NULL)
    {
        return 0;
    }
    const int base = CACAOSIM_BEAM_KW_DISP_BASE;
    const int need = base + 1 + 2 * CACAOSIM_DISP_NCOEFF;
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < need ||
        im->kw[base].type != 'L' || strcmp(im->kw[base].name, "CSIM_DSP") != 0 ||
        im->kw[base].value.numl != 1)
    {
        if (d->active)
        {
            cacaosim_disp_zero(d);
        }
        return 0;
    }

    int changed = (!d->active);
    for (int k = 0; k < CACAOSIM_DISP_NCOEFF; k++)
    {
        double vx = im->kw[base + 1 + k].value.numf;
        double vy = im->kw[base + 1 + CACAOSIM_DISP_NCOEFF + k].value.numf;
        if (vx != d->cx[k] || vy != d->cy[k])
        {
            d->cx[k] = vx;
            d->cy[k] = vy;
            changed  = 1;
        }
    }
    d->active = 1;
    if (changed)
    {
        d->gen++;
    }
    return 1;
}

void cacaosim_beam_shm_write_poly_mode(
    IMAGE  *im,
    int32_t poly_mode)
{
    const int slot = CACAOSIM_BEAM_KW_DISP_BASE + 1 + 2 * CACAOSIM_DISP_NCOEFF; /* kw[64] */
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw <= slot)
    {
        return;
    }
    if (im->kw[slot].type != 'L' || strcmp(im->kw[slot].name, "CSIM_PMD") != 0)
    {
        snprintf(im->kw[slot].name, KEYWORD_MAX_STRING, "CSIM_PMD");
        im->kw[slot].type = 'L';
        snprintf(im->kw[slot].comment, KEYWORD_MAX_COMMENT, "Poly mode (0=1wl,1=OPD,2=cplx)");
    }
    im->kw[slot].value.numl = (int64_t) poly_mode;
}

int32_t cacaosim_beam_shm_read_poly_mode(
    const IMAGE *im,
    uint32_t     nwl)
{
    if (nwl <= 1u)
    {
        return (int32_t) CACAOSIM_POLY_NONE;
    }
    const int slot = CACAOSIM_BEAM_KW_DISP_BASE + 1 + 2 * CACAOSIM_DISP_NCOEFF;
    if (im != NULL && im->md != NULL && im->kw != NULL && im->md->NBkw > slot &&
        im->kw[slot].type == 'L' && strcmp(im->kw[slot].name, "CSIM_PMD") == 0)
    {
        int64_t v = im->kw[slot].value.numl;
        if (v == CACAOSIM_POLY_UNWRAPPED_OPD || v == CACAOSIM_POLY_COMPLEX_FIELD)
        {
            return (int32_t) v;
        }
    }
    return (int32_t) CACAOSIM_POLY_COMPLEX_FIELD;
}

void cacaosim_beam_shm_write_diam(
    IMAGE *im,
    float  tel_diam_m)
{
    const int slot = CACAOSIM_BEAM_KW_DISP_BASE + 2 + 2 * CACAOSIM_DISP_NCOEFF; /* kw[65] */
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw <= slot)
    {
        return;
    }
    if (im->kw[slot].type != 'D' || strcmp(im->kw[slot].name, "CSIM_DIA") != 0)
    {
        snprintf(im->kw[slot].name, KEYWORD_MAX_STRING, "CSIM_DIA");
        im->kw[slot].type = 'D';
        snprintf(im->kw[slot].comment, KEYWORD_MAX_COMMENT, "Telescope primary diameter [m]");
    }
    im->kw[slot].value.numf = (double) ((tel_diam_m > 0.01f) ? tel_diam_m : 8.0f);
}

float cacaosim_beam_shm_read_diam(
    const IMAGE *im,
    float        fallback_m)
{
    const int slot = CACAOSIM_BEAM_KW_DISP_BASE + 2 + 2 * CACAOSIM_DISP_NCOEFF; /* kw[65] */
    if (im != NULL && im->md != NULL && im->kw != NULL && im->md->NBkw > slot &&
        im->kw[slot].type == 'D' && strcmp(im->kw[slot].name, "CSIM_DIA") == 0 &&
        im->kw[slot].value.numf > 0.01)
    {
        return (float) im->kw[slot].value.numf;
    }
    return (fallback_m > 0.01f) ? fallback_m : 8.0f;
}

void cacaosim_beam_shm_write_poly_wls(
    IMAGE       *im,
    const float *lam_um,
    uint32_t     nwl)
{
    const int base = CACAOSIM_BEAM_KW_DISP_BASE + 3 + 2 * CACAOSIM_DISP_NCOEFF; /* kw[66] */
    if (im == NULL || im->md == NULL || im->kw == NULL || lam_um == NULL || nwl == 0u)
    {
        return;
    }
    uint32_t cnt = (nwl <= 8u) ? nwl : 8u;
    if (im->md->NBkw < base + (int) cnt)
    {
        return;
    }
    for (uint32_t i = 0u; i < cnt; i++)
    {
        int  slot = base + (int) i;
        char kname[16];
        snprintf(kname, sizeof(kname), "CSIM_W%02u", i);
        if (im->kw[slot].type != 'D' || strcmp(im->kw[slot].name, kname) != 0)
        {
            snprintf(im->kw[slot].name, KEYWORD_MAX_STRING, "%s", kname);
            im->kw[slot].type = 'D';
            snprintf(im->kw[slot].comment, KEYWORD_MAX_COMMENT, "Beam slice %u wavelength [um]",
                     i);
        }
        im->kw[slot].value.numf = (double) lam_um[i];
    }
}

int cacaosim_beam_shm_read_poly_wls(
    const IMAGE *im,
    uint32_t     nwl,
    float       *out_lam_um)
{
    const int base = CACAOSIM_BEAM_KW_DISP_BASE + 3 + 2 * CACAOSIM_DISP_NCOEFF; /* kw[66] */
    if (im == NULL || im->md == NULL || im->kw == NULL || out_lam_um == NULL || nwl == 0u ||
        nwl > 8u || im->md->NBkw < base + (int) nwl)
    {
        return 0;
    }
    for (uint32_t i = 0u; i < nwl; i++)
    {
        int  slot = base + (int) i;
        char kname[16];
        snprintf(kname, sizeof(kname), "CSIM_W%02u", i);
        if (im->kw[slot].type != 'D' || strcmp(im->kw[slot].name, kname) != 0 ||
            im->kw[slot].value.numf <= 0.05)
        {
            return 0;
        }
        out_lam_um[i] = (float) im->kw[slot].value.numf;
    }
    return 1;
}

int cacaosim_wf_shm_read_chrom(
    const IMAGE *im,
    uint32_t    *out_nwl,
    int32_t     *out_unit,
    float       *out_wls)
{
    if (out_nwl != NULL)
    {
        *out_nwl = 1u;
    }
    if (out_unit != NULL)
    {
        *out_unit = 1;
    }
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->naxis != 3 ||
        im->md->size[2] < 2u || im->md->NBkw < 6)
    {
        return 0;
    }

    int64_t nwl  = 0;
    int64_t unit = 1;
    float   wls[8];
    int     have_wl[8];
    memset(wls, 0, sizeof(wls));
    memset(have_wl, 0, sizeof(have_wl));

    for (int i = 0; i < im->md->NBkw; i++)
    {
        const char *k = im->kw[i].name;
        if (k[0] == '\0')
        {
            continue;
        }
        if (im->kw[i].type == 'L' && strcmp(k, "ATMT_NWL") == 0)
        {
            nwl = im->kw[i].value.numl;
        }
        else if (im->kw[i].type == 'L' && strcmp(k, "ATMT_UNIT") == 0)
        {
            unit = im->kw[i].value.numl;
        }
        else if (im->kw[i].type == 'D' && strncmp(k, "ATMT_WL", 7) == 0 &&
                 k[7] >= '0' && k[7] <= '7' && k[8] == '\0')
        {
            int idx = k[7] - '0';
            if (im->kw[i].value.numf > 0.05)
            {
                wls[idx]     = (float) im->kw[i].value.numf;
                have_wl[idx] = 1;
            }
        }
    }

    if (nwl < 2 || nwl > 8 || (uint32_t) nwl > im->md->size[2])
    {
        return 0;
    }
    for (int m = 0; m < (int) nwl; m++)
    {
        if (!have_wl[m])
        {
            return 0;
        }
    }

    if (out_nwl != NULL)
    {
        *out_nwl = (uint32_t) nwl;
    }
    if (out_unit != NULL)
    {
        *out_unit = (int32_t) unit;
    }
    if (out_wls != NULL)
    {
        for (int m = 0; m < (int) nwl; m++)
        {
            out_wls[m] = wls[m];
        }
    }
    return 1;
}
