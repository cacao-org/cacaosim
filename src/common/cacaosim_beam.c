// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_beam.c
 * @brief   CPU implementation of spectral bandpass photometry, 2D beam relay, and SHM beam IO
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "milk_config.h"
#include "milk_types.h"
#include "common/cacaosim_beam.h"

int cacaosim_reg_is_identity(const cacaosim_reg_t *reg)
{
    if (reg == NULL)
    {
        return 1;
    }
    float sc = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
    return (fabsf(reg->shift_x) < 1e-5f && fabsf(reg->shift_y) < 1e-5f &&
            fabsf(reg->rot_deg) < 1e-5f && fabsf(sc - 1.0f) < 1e-5f)
               ? 1
               : 0;
}

/**
 * planck_photon_rel() - Planck photon spectral density N_lambda(lam) normalized to 1 at 0.55 um
 */
static float planck_photon_rel(
    float lam_um,
    float teff_k)
{
    /* h*c / k_B = 14387.77 um * K */
    const float hc_k = 14387.77f;
    float       T    = (teff_k > 500.0f) ? teff_k : 5778.0f;
    float       x    = hc_k / (lam_um * T);
    float       x0   = hc_k / (0.55f * T);
    if (x > 80.0f)
    {
        return 0.0f;
    }
    float n_lam = 1.0f / (lam_um * lam_um * lam_um * lam_um * expm1f(x));
    float n_v   = 1.0f / (0.55f * 0.55f * 0.55f * 0.55f * expm1f(x0));
    return (n_v > 0.0f) ? (n_lam / n_v) : 1.0f;
}

void MILK_COLD cacaosim_spec_star(
    cacaosim_spec_t *out,
    float            tel_diam_m,
    float            obs_frac,
    float            star_mag,
    float            teff_k,
    float            tput)
{
    if (out == NULL)
    {
        return;
    }
    float diam = (tel_diam_m > 0.01f) ? tel_diam_m : 8.0f;
    float obs  = (obs_frac > 0.0f && obs_frac < 0.99f) ? obs_frac : 0.0f;
    float area = 0.25f * CACAOSIM_PI_F * diam * diam * (1.0f - obs * obs);
    float tp   = (tput >= 0.0f) ? tput : 1.0f;

    /* Vega m_V = 0 zero-point: 1.0e11 photons / (s * m^2 * um) at 0.55 um */
    float f0_v     = 1.0e11f * powf(10.0f, -0.4f * star_mag);
    float base_flx = f0_v * area * tp * CACAOSIM_SPEC_DL_UM;

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        float lam      = cacaosim_spec_bin_lambda(b);
        out->ph_s[b]   = base_flx * planck_photon_rel(lam, teff_k);
    }
}

void cacaosim_spec_scale(
    const cacaosim_spec_t *in,
    float                  tput,
    cacaosim_spec_t       *out)
{
    if (in == NULL || out == NULL)
    {
        return;
    }
    float g = (tput > 0.0f) ? ((tput <= 1.0f) ? tput : 1.0f) : 0.0f;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        out->ph_s[b] = g * in->ph_s[b];
    }
}

float cacaosim_spec_total(const cacaosim_spec_t *spec)
{
    if (spec == NULL)
    {
        return 0.0f;
    }
    float sum = 0.0f;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        sum += spec->ph_s[b];
    }
    return sum;
}

static float smooth_step_lo(
    float lam,
    float cut,
    float roll)
{
    float w = (roll > 1e-4f) ? roll : 0.02f;
    float z = 4.0f * (lam - cut) / w;
    if (z > 40.0f)
    {
        return 0.0f;
    }
    if (z < -40.0f)
    {
        return 1.0f;
    }
    return 1.0f / (1.0f + expf(z));
}

void cacaosim_spec_split(
    const cacaosim_spec_t      *in,
    const cacaosim_split_cfg_t *cfg,
    cacaosim_spec_t            *out_refl,
    cacaosim_spec_t            *out_trans)
{
    if (in == NULL || cfg == NULL)
    {
        return;
    }
    cacaosim_spec_t src = *in;

    float er = (cfg->eff_refl > 0.0f) ? ((cfg->eff_refl <= 1.0f) ? cfg->eff_refl : 1.0f) : 0.0f;
    float et = (cfg->eff_trans > 0.0f) ? ((cfg->eff_trans <= 1.0f) ? cfg->eff_trans : 1.0f) : 0.0f;

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        float lam = cacaosim_spec_bin_lambda(b);
        float r   = 0.0f;
        float t   = 0.0f;

        if (cfg->mode == CACAOSIM_SPLIT_GRAY)
        {
            r = (cfg->r_gray > 0.0f) ? cfg->r_gray : 0.0f;
            t = (cfg->t_gray > 0.0f) ? cfg->t_gray : 0.0f;
        }
        else if (cfg->mode == CACAOSIM_SPLIT_DICHROIC_LP)
        {
            float s = smooth_step_lo(lam, cfg->lambda_cut, cfg->roll_um);
            r       = er * s;
            t       = et * (1.0f - s);
        }
        else if (cfg->mode == CACAOSIM_SPLIT_DICHROIC_SP)
        {
            float s = smooth_step_lo(lam, cfg->lambda_cut, cfg->roll_um);
            r       = er * (1.0f - s);
            t       = et * s;
        }
        else
        {
            float hw = 0.5f * ((cfg->band_width > 0.01f) ? cfg->band_width : 0.20f);
            float s1 = 1.0f - smooth_step_lo(lam, cfg->lambda_cut - hw, cfg->roll_um);
            float s2 = smooth_step_lo(lam, cfg->lambda_cut + hw, cfg->roll_um);
            float bp = s1 * s2;
            r        = er * bp;
            t        = et * (1.0f - bp);
        }

        if (r + t > 1.0f)
        {
            float inv = 1.0f / (r + t);
            r *= inv;
            t *= inv;
        }

        if (out_refl != NULL)
        {
            out_refl->ph_s[b] = r * src.ph_s[b];
        }
        if (out_trans != NULL)
        {
            out_trans->ph_s[b] = t * src.ph_s[b];
        }
    }
}

float cacaosim_spec_band_flux(
    const cacaosim_spec_t *spec,
    float                  lambda0_um,
    float                  bw_frac,
    float                  dt_exp_s,
    float                  qe,
    float                 *eff_lambda)
{
    if (spec == NULL)
    {
        if (eff_lambda != NULL)
        {
            *eff_lambda = lambda0_um;
        }
        return 0.0f;
    }
    float bw   = (bw_frac > 1e-3f) ? bw_frac : 0.15f;
    float lmin = lambda0_um * (1.0f - 0.5f * bw);
    float lmax = lambda0_um * (1.0f + 0.5f * bw);
    float dt   = (dt_exp_s > 0.0f) ? dt_exp_s : 0.0005f;
    float q    = (qe > 0.0f) ? ((qe <= 1.0f) ? qe : 1.0f) : 1.0f;

    float sum_rate = 0.0f;
    float sum_wlam = 0.0f;

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        float b0 = CACAOSIM_SPEC_L0_UM + (float) b * CACAOSIM_SPEC_DL_UM;
        float b1 = b0 + CACAOSIM_SPEC_DL_UM;
        float lo = (b0 > lmin) ? b0 : lmin;
        float hi = (b1 < lmax) ? b1 : lmax;
        if (hi > lo)
        {
            float frac = (hi - lo) / CACAOSIM_SPEC_DL_UM;
            float r    = frac * spec->ph_s[b];
            sum_rate += r;
            sum_wlam += r * 0.5f * (lo + hi);
        }
    }

    if (eff_lambda != NULL)
    {
        *eff_lambda = (sum_rate > 1e-20f) ? (sum_wlam / sum_rate) : lambda0_um;
    }
    return sum_rate * dt * q;
}

void MILK_HOT cacaosim_beam_init_cpu(
    const float *__restrict pup_amp,
    const float *__restrict atm_amp,
    const float *__restrict wf_in,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd,
    uint32_t                nwf,
    float *__restrict       out_opd,
    float *__restrict       out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

#pragma omp parallel for simd schedule(static) if (npix >= 8192)
    for (uint64_t p = 0; p < npix; p++)
    {
        float a = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
        if (atm_amp != NULL)
        {
            a *= atm_amp[p];
        }
        float o = (wf_in != NULL) ? (wf_to_opd * wf_in[p]) : 0.0f;
        if (ncpa_opd != NULL)
        {
            o += ncpa_opd[p];
        }
        out_opd[p] = o;
        out_amp[p] = a;
    }
}

void MILK_HOT cacaosim_beam_apply_dm_cpu(
    const float *__restrict in_opd,
    const float *__restrict in_amp,
    const float *__restrict dm_opd,
    uint32_t                nwf,
    float *__restrict       out_opd,
    float *__restrict       out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

#pragma omp parallel for simd schedule(static) if (npix >= 8192)
    for (uint64_t p = 0; p < npix; p++)
    {
        float o    = (in_opd != NULL) ? in_opd[p] : 0.0f;
        float d    = (dm_opd != NULL) ? dm_opd[p] : 0.0f;
        out_opd[p] = o - d;
        if (out_amp != NULL && out_amp != in_amp)
        {
            out_amp[p] = (in_amp != NULL) ? in_amp[p] : 1.0f;
        }
    }
}

void MILK_HOT cacaosim_beam_relay_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *ncpa_opd,
    const float          *pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

    if (cacaosim_reg_is_identity(reg))
    {
#pragma omp parallel for simd schedule(static) if (npix >= 8192)
        for (uint64_t p = 0; p < npix; p++)
        {
            float o = (in_opd != NULL) ? in_opd[p] : 0.0f;
            float a = (in_amp != NULL) ? in_amp[p] : 1.0f;
            if (ncpa_opd != NULL)
            {
                o += ncpa_opd[p];
            }
            if (pup_mask != NULL)
            {
                a *= pup_mask[p];
            }
            out_opd[p] = o;
            out_amp[p] = a;
        }
        return;
    }

    float sc   = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
    float invs = 1.0f / sc;
    float rad  = -reg->rot_deg * (CACAOSIM_PI_F / 180.0f);
    float c    = cosf(rad) * invs;
    float s    = sinf(rad) * invs;
    float ctr  = 0.5f * (float) (nwf - 1u);
    float dx   = reg->shift_x;
    float dy   = reg->shift_y;

    float *tmp_opd = NULL;
    float *tmp_amp = NULL;
    if (out_opd == in_opd || out_amp == in_amp)
    {
        tmp_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        tmp_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    }
    float *dst_o = (tmp_opd != NULL) ? tmp_opd : out_opd;
    float *dst_a = (tmp_amp != NULL) ? tmp_amp : out_amp;

#pragma omp parallel for schedule(static) if (npix >= 4096)
    for (uint32_t iy = 0; iy < nwf; iy++)
    {
        float vy = (float) iy - ctr - dy;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            float    vx = (float) ix - ctr - dx;
            float    sx = ctr + c * vx - s * vy;
            float    sy = ctr + s * vx + c * vy;
            uint64_t p  = (uint64_t) iy * nwf + ix;

            float o = 0.0f;
            float a = 0.0f;
            if (sx >= 0.0f && sy >= 0.0f && sx <= (float) (nwf - 1u) && sy <= (float) (nwf - 1u))
            {
                uint32_t x0 = (uint32_t) sx;
                uint32_t y0 = (uint32_t) sy;
                uint32_t x1 = (x0 + 1u < nwf) ? (x0 + 1u) : x0;
                uint32_t y1 = (y0 + 1u < nwf) ? (y0 + 1u) : y0;
                float    fx = sx - (float) x0;
                float    fy = sy - (float) y0;
                float    w00 = (1.0f - fx) * (1.0f - fy);
                float    w10 = fx * (1.0f - fy);
                float    w01 = (1.0f - fx) * fy;
                float    w11 = fx * fy;

                uint64_t i00 = (uint64_t) y0 * nwf + x0;
                uint64_t i10 = (uint64_t) y0 * nwf + x1;
                uint64_t i01 = (uint64_t) y1 * nwf + x0;
                uint64_t i11 = (uint64_t) y1 * nwf + x1;

                if (in_opd != NULL)
                {
                    o = w00 * in_opd[i00] + w10 * in_opd[i10] + w01 * in_opd[i01] +
                        w11 * in_opd[i11];
                }
                if (in_amp != NULL)
                {
                    a = w00 * in_amp[i00] + w10 * in_amp[i10] + w01 * in_amp[i01] +
                        w11 * in_amp[i11];
                }
                else
                {
                    a = 1.0f;
                }
            }
            if (ncpa_opd != NULL)
            {
                o += ncpa_opd[p];
            }
            if (pup_mask != NULL)
            {
                a *= pup_mask[p];
            }
            dst_o[p] = o;
            dst_a[p] = a;
        }
    }

    if (tmp_opd != NULL)
    {
        memcpy(out_opd, tmp_opd, npix * sizeof(float));
        memcpy(out_amp, tmp_amp, npix * sizeof(float));
        free(tmp_opd);
        free(tmp_amp);
    }
}

void MILK_HOT cacaosim_beam_apply_wfe_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *wfe_opd,
    float                 wfe_to_um,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

    if (cacaosim_reg_is_identity(reg))
    {
#pragma omp parallel for simd schedule(static) if (npix >= 8192)
        for (uint64_t p = 0; p < npix; p++)
        {
            float o    = (in_opd != NULL) ? in_opd[p] : 0.0f;
            float a    = (in_amp != NULL) ? in_amp[p] : 1.0f;
            float w    = (wfe_opd != NULL) ? (wfe_to_um * wfe_opd[p]) : 0.0f;
            out_opd[p] = o + w;
            if (out_amp != NULL && out_amp != in_amp)
            {
                out_amp[p] = a;
            }
        }
        return;
    }

    float sc   = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
    float invs = 1.0f / sc;
    float rad  = -reg->rot_deg * (CACAOSIM_PI_F / 180.0f);
    float c    = cosf(rad) * invs;
    float s    = sinf(rad) * invs;
    float ctr  = 0.5f * (float) (nwf - 1u);
    float dx   = reg->shift_x;
    float dy   = reg->shift_y;

#pragma omp parallel for schedule(static) if (npix >= 4096)
    for (uint32_t iy = 0; iy < nwf; iy++)
    {
        float vy = (float) iy - ctr - dy;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            float    vx = (float) ix - ctr - dx;
            float    sx = ctr + c * vx - s * vy;
            float    sy = ctr + s * vx + c * vy;
            uint64_t p  = (uint64_t) iy * nwf + ix;

            float o = 0.0f;
            float a = 0.0f;
            if (sx >= 0.0f && sy >= 0.0f && sx <= (float) (nwf - 1u) && sy <= (float) (nwf - 1u))
            {
                uint32_t x0  = (uint32_t) sx;
                uint32_t y0  = (uint32_t) sy;
                uint32_t x1  = (x0 + 1u < nwf) ? (x0 + 1u) : x0;
                uint32_t y1  = (y0 + 1u < nwf) ? (y0 + 1u) : y0;
                float    fx  = sx - (float) x0;
                float    fy  = sy - (float) y0;
                float    w00 = (1.0f - fx) * (1.0f - fy);
                float    w10 = fx * (1.0f - fy);
                float    w01 = (1.0f - fx) * fy;
                float    w11 = fx * fy;

                uint64_t i00 = (uint64_t) y0 * nwf + x0;
                uint64_t i10 = (uint64_t) y0 * nwf + x1;
                uint64_t i01 = (uint64_t) y1 * nwf + x0;
                uint64_t i11 = (uint64_t) y1 * nwf + x1;

                if (in_opd != NULL)
                {
                    o = w00 * in_opd[i00] + w10 * in_opd[i10] + w01 * in_opd[i01] +
                        w11 * in_opd[i11];
                }
                a = (in_amp != NULL)
                        ? (w00 * in_amp[i00] + w10 * in_amp[i10] + w01 * in_amp[i01] +
                           w11 * in_amp[i11])
                        : 1.0f;
            }
            if (wfe_opd != NULL)
            {
                o += wfe_to_um * wfe_opd[p];
            }
            out_opd[p] = o;
            if (out_amp != NULL)
            {
                out_amp[p] = a;
            }
        }
    }
}

void MILK_COLD cacaosim_beam_shm_init_kw(IMAGE *im)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < 4 + CACAOSIM_NSPEC)
    {
        return;
    }
    snprintf(im->kw[0].name, KEYWORD_MAX_STRING, "CSIM_BM");
    im->kw[0].type       = 'L';
    im->kw[0].value.numl = 1;
    snprintf(im->kw[0].comment, KEYWORD_MAX_COMMENT, "cacaosim [OPD_um, Amp] optical beam");

    snprintf(im->kw[1].name, KEYWORD_MAX_STRING, "CSIM_L0");
    im->kw[1].type       = 'D';
    im->kw[1].value.numf = (double) CACAOSIM_SPEC_L0_UM;
    snprintf(im->kw[1].comment, KEYWORD_MAX_COMMENT, "Min spectral wavelength [um]");

    snprintf(im->kw[2].name, KEYWORD_MAX_STRING, "CSIM_L1");
    im->kw[2].type       = 'D';
    im->kw[2].value.numf = (double) CACAOSIM_SPEC_L1_UM;
    snprintf(im->kw[2].comment, KEYWORD_MAX_COMMENT, "Max spectral wavelength [um]");

    snprintf(im->kw[3].name, KEYWORD_MAX_STRING, "CSIM_FLX");
    im->kw[3].type       = 'D';
    im->kw[3].value.numf = 0.0;
    snprintf(im->kw[3].comment, KEYWORD_MAX_COMMENT, "Total integrated photon rate [ph/s]");

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        int k = 4 + b;
        snprintf(im->kw[k].name, KEYWORD_MAX_STRING, "CSIM_S%02d", b);
        im->kw[k].type       = 'D';
        im->kw[k].value.numf = 0.0;
        snprintf(im->kw[k].comment, KEYWORD_MAX_COMMENT, "Bin %.3f um rate [ph/s]",
                 (double) cacaosim_spec_bin_lambda(b));
    }
}

void MILK_HOT cacaosim_beam_shm_write_spec(
    IMAGE                 *im,
    const cacaosim_spec_t *spec)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || spec == NULL ||
        im->md->NBkw < 4 + CACAOSIM_NSPEC)
    {
        return;
    }
    if (im->kw[0].type != 'L' || im->kw[0].value.numl != 1)
    {
        cacaosim_beam_shm_init_kw(im);
    }
    double tot = 0.0;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        double v             = (double) spec->ph_s[b];
        im->kw[4 + b].value.numf = v;
        tot += v;
    }
    im->kw[3].value.numf = tot;
}

int cacaosim_beam_shm_read_spec(
    const IMAGE     *im,
    cacaosim_spec_t *spec)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || spec == NULL ||
        im->md->NBkw < 4 + CACAOSIM_NSPEC)
    {
        return 0;
    }
    if (im->kw[0].type != 'L' || im->kw[0].value.numl != 1)
    {
        return 0;
    }
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        spec->ph_s[b] = (float) im->kw[4 + b].value.numf;
    }
    return 1;
}

int cacaosim_beam_shm_is_valid(
    const IMAGE *im,
    uint32_t     nwf)
{
    if (im == NULL || im->md == NULL)
    {
        return 0;
    }
    return (im->md->datatype == _DATATYPE_FLOAT && im->md->naxis == 3 &&
            im->md->size[0] == nwf && im->md->size[1] == nwf && im->md->size[2] >= 2)
               ? 1
               : 0;
}

void cacaosim_beam_shm_write_poly_kw(
    IMAGE   *im,
    uint32_t nwl,
    float    wl0_um,
    float    wl1_um)
{
    const int base = 4 + CACAOSIM_NSPEC;
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < base + 3)
    {
        return;
    }
    snprintf(im->kw[base].name, KEYWORD_MAX_STRING, "CSIM_NWL");
    im->kw[base].type       = 'L';
    im->kw[base].value.numl = (int64_t) ((nwl >= 1u) ? nwl : 1u);
    snprintf(im->kw[base].comment, KEYWORD_MAX_COMMENT, "Number of [OPD, Amp] spectral channels");

    snprintf(im->kw[base + 1].name, KEYWORD_MAX_STRING, "CSIM_WL0");
    im->kw[base + 1].type       = 'D';
    im->kw[base + 1].value.numf = (double) wl0_um;
    snprintf(im->kw[base + 1].comment, KEYWORD_MAX_COMMENT, "First channel wavelength [um]");

    snprintf(im->kw[base + 2].name, KEYWORD_MAX_STRING, "CSIM_WL1");
    im->kw[base + 2].type       = 'D';
    im->kw[base + 2].value.numf = (double) wl1_um;
    snprintf(im->kw[base + 2].comment, KEYWORD_MAX_COMMENT, "Last channel wavelength [um]");
}

int cacaosim_beam_shm_read_poly_kw(
    const IMAGE *im,
    uint32_t    *nwl_out,
    float       *wl0_out,
    float       *wl1_out)
{
    if (nwl_out != NULL)
    {
        *nwl_out = 1u;
    }
    if (wl0_out != NULL)
    {
        *wl0_out = 0.0f;
    }
    if (wl1_out != NULL)
    {
        *wl1_out = 0.0f;
    }
    if (im == NULL || im->md == NULL || im->md->naxis != 3 || im->md->size[2] < 4u)
    {
        return 0;
    }

    uint32_t z_nwl = im->md->size[2] / 2u;
    uint32_t kw_n  = z_nwl;
    float    l0    = 0.0f;
    float    l1    = 0.0f;

    const int base = 4 + CACAOSIM_NSPEC;
    if (im->kw != NULL && im->md->NBkw >= base + 3 && im->kw[base].type == 'L' &&
        strcmp(im->kw[base].name, "CSIM_NWL") == 0 && im->kw[base].value.numl >= 1)
    {
        uint32_t kn = (uint32_t) im->kw[base].value.numl;
        if (kn <= z_nwl)
        {
            kw_n = kn;
        }
        l0 = (float) im->kw[base + 1].value.numf;
        l1 = (float) im->kw[base + 2].value.numf;
    }

    if (nwl_out != NULL)
    {
        *nwl_out = kw_n;
    }
    if (wl0_out != NULL)
    {
        *wl0_out = l0;
    }
    if (wl1_out != NULL)
    {
        *wl1_out = l1;
    }
    return (kw_n > 1u) ? 1 : 0;
}
