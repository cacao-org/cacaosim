// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pupil.c
 * @brief   4x4 supersampled anti-aliased pupil amplitude and low-order NCPA generator
 */

#include <math.h>
#include <omp.h>
#include <stddef.h>

#include "milk_config.h"
#include "milk_types.h"
#include "cacaosim_types.h"
#include "common/cacaosim_rng.h"
#include "pupil/cacaosim_pupil.h"

double cacaosim_make_pupil(
    const cacaosim_pupil_cfg_t *cfg,
    float                      *pup_out)
{
    if (cfg == NULL || pup_out == NULL || cfg->nwf == 0)
    {
        return 0.0;
    }

    const uint32_t nwf    = cfg->nwf;
    const float    diam   = (cfg->pup_diam > 0.01f && cfg->pup_diam <= 1.0f) ? cfg->pup_diam : 0.9f;
    const float    r_out  = 0.5f * diam * (float) nwf;
    const float    r_out2 = r_out * r_out;
    const float    cobs   = (cfg->cobs >= 0.0f && cfg->cobs < 0.98f) ? cfg->cobs : 0.0f;
    const float    r_in2  = (cobs * r_out) * (cobs * r_out);
    const float    ctr    = 0.5f * (float) (nwf - 1u);
    const int32_t  nsp    = (cfg->nspider > 0 && cfg->nspider <= 16) ? cfg->nspider : 0;
    const float    half_w = 0.5f * cfg->spider_w * (2.0f * r_out);

    float sp_cos[16];
    float sp_sin[16];
    for (int32_t k = 0; k < nsp; k++)
    {
        float ang = cfg->spider_deg * (CACAOSIM_PI_F / 180.0f) +
                    (CACAOSIM_TWO_PI_F * (float) k) / (float) nsp;
        sp_cos[k] = cosf(ang);
        sp_sin[k] = sinf(ang);
    }

    double sum_amp = 0.0;

#pragma omp parallel for reduction(+ : sum_amp) schedule(static)
    for (uint32_t jy = 0; jy < nwf; jy++)
    {
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            int32_t hits = 0;
            for (int sy = 0; sy < 4; sy++)
            {
                float dy = ((float) jy - ctr) + ((float) sy - 1.5f) * 0.25f;
                for (int sx = 0; sx < 4; sx++)
                {
                    float dx = ((float) ix - ctr) + ((float) sx - 1.5f) * 0.25f;
                    float r2 = dx * dx + dy * dy;
                    if (r2 > r_out2 || r2 < r_in2)
                    {
                        continue;
                    }

                    int blocked = 0;
                    if (nsp > 0 && half_w > 0.0f)
                    {
                        for (int32_t k = 0; k < nsp; k++)
                        {
                            float along = dx * sp_cos[k] + dy * sp_sin[k];
                            if (along >= 0.0f)
                            {
                                float perp = fabsf(-dx * sp_sin[k] + dy * sp_cos[k]);
                                if (perp <= half_w)
                                {
                                    blocked = 1;
                                    break;
                                }
                            }
                        }
                    }
                    if (!blocked)
                    {
                        hits++;
                    }
                }
            }
            float val                = (float) hits * (1.0f / 16.0f);
            pup_out[jy * nwf + ix]   = val;
            sum_amp                 += (double) val;
        }
    }

    return sum_amp;
}

void cacaosim_make_ncpa(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        ncpa_rms_um,
    uint64_t     seed,
    float       *ncpa_out)
{
    if (ncpa_out == NULL || nwf == 0)
    {
        return;
    }

    uint64_t npix = (uint64_t) nwf * nwf;
    if (ncpa_rms_um <= 0.0f || pup_amp == NULL)
    {
        __builtin_memset(ncpa_out, 0, npix * sizeof(float));
        return;
    }

    cacaosim_rng_t rng;
    cacaosim_rng_seed(&rng, (seed != 0ULL) ? seed : 42ULL, 99ULL);

    /* Coefficients for defocus, astigmatism (x2), coma (x2), trefoil (x2), spherical */
    float coef[8];
    for (int m = 0; m < 8; m++)
    {
        float weight = (m < 3) ? 1.0f : ((m < 7) ? 0.6f : 0.4f);
        coef[m]      = weight * cacaosim_rng_gauss_f32(&rng);
    }

    const float diam  = (pup_diam > 0.01f && pup_diam <= 1.0f) ? pup_diam : 0.9f;
    const float r_out = 0.5f * diam * (float) nwf;
    const float inv_r = 1.0f / r_out;
    const float ctr   = 0.5f * (float) (nwf - 1u);

    double wsum = 0.0;
    double mean = 0.0;

    for (uint32_t jy = 0; jy < nwf; jy++)
    {
        float y = ((float) jy - ctr) * inv_r;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            uint64_t idx = (uint64_t) jy * nwf + ix;
            float    w   = pup_amp[idx];
            if (w <= 0.0f)
            {
                ncpa_out[idx] = 0.0f;
                continue;
            }
            float x  = ((float) ix - ctr) * inv_r;
            float r2 = x * x + y * y;
            float z  = coef[0] * (2.0f * r2 - 1.0f) +
                       coef[1] * (x * x - y * y) +
                       coef[2] * (2.0f * x * y) +
                       coef[3] * (3.0f * r2 - 2.0f) * x +
                       coef[4] * (3.0f * r2 - 2.0f) * y +
                       coef[5] * (x * x * x - 3.0f * x * y * y) +
                       coef[6] * (3.0f * x * x * y - y * y * y) +
                       coef[7] * (6.0f * r2 * r2 - 6.0f * r2 + 1.0f);

            ncpa_out[idx] = z;
            wsum += (double) w;
            mean += (double) w * (double) z;
        }
    }

    if (wsum <= 1e-12)
    {
        __builtin_memset(ncpa_out, 0, npix * sizeof(float));
        return;
    }
    mean /= wsum;

    double var = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (pup_amp[i] > 0.0f)
        {
            double dv = (double) ncpa_out[i] - mean;
            var += (double) pup_amp[i] * dv * dv;
        }
    }
    double rms   = sqrt(var / wsum);
    float  scale = (rms > 1e-12) ? (ncpa_rms_um / (float) rms) : 0.0f;

    for (uint64_t i = 0; i < npix; i++)
    {
        if (pup_amp[i] > 0.0f)
        {
            ncpa_out[i] = ((float) ((double) ncpa_out[i] - mean)) * scale;
        }
        else
        {
            ncpa_out[i] = 0.0f;
        }
    }
}

void MILK_COLD cacaosim_make_wfe_m(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        rms_m,
    float        psd_alpha,
    uint64_t     seed,
    float       *wfe_out_m)
{
    if (wfe_out_m == NULL || nwf == 0)
    {
        return;
    }

    const uint64_t npix = (uint64_t) nwf * nwf;
    memset(wfe_out_m, 0, npix * sizeof(float));

    if (rms_m <= 0.0f || pup_diam <= 0.0f)
    {
        return;
    }

    const float alpha = (psd_alpha > 0.2f) ? psd_alpha : 1.6f;
    const float cx    = 0.5f * (float) (nwf - 1u);
    const float cy    = 0.5f * (float) (nwf - 1u);
    const float d_px  = pup_diam * (float) nwf;
    const float r_out = 0.5f * d_px;

    cacaosim_rng_t rng;
    cacaosim_rng_seed(&rng, (seed != 0ULL) ? seed : 101ULL, 101ULL);

    /* Low-order Zernike polishing & alignment terms (focus, astig, coma, trefoil, spherical) */
    float       zc[8];
    const float z_w[8] = { 1.0f, 0.85f, 0.85f, 0.55f, 0.55f, 0.40f, 0.40f, 0.35f };
    for (int k = 0; k < 8; k++)
    {
        zc[k] = z_w[k] * cacaosim_rng_gauss_f32(&rng);
    }

    /* Mid-spatial-frequency 2D Fourier polishing ripple modes (2..8 cycles / pupil) */
    enum
    {
        KMAX   = 8,
        MAX_MD = 160
    };
    int   mkx[MAX_MD];
    int   mky[MAX_MD];
    float mamp[MAX_MD];
    float mphi[MAX_MD];
    int   n_md = 0;

    for (int ky = 0; ky <= KMAX; ky++)
    {
        for (int kx = -KMAX; kx <= KMAX; kx++)
        {
            if (ky == 0 && kx <= 0)
            {
                continue;
            }
            float kf = hypotf((float) kx, (float) ky);
            if (kf < 1.8f || kf > (float) KMAX || n_md >= MAX_MD)
            {
                continue;
            }
            mkx[n_md]  = kx;
            mky[n_md]  = ky;
            mamp[n_md] = 0.65f * powf(kf, -alpha) * (0.5f + 0.7f * cacaosim_rng_uniform_f32(&rng));
            mphi[n_md] = CACAOSIM_TWO_PI_F * cacaosim_rng_uniform_f32(&rng);
            n_md++;
        }
    }

    double w_sum = 0.0;
    double m_sum = 0.0;

    for (uint32_t iy = 0; iy < nwf; iy++)
    {
        const float y  = ((float) iy - cy) / r_out;
        const float uy = ((float) iy - cy) / d_px;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            const float x  = ((float) ix - cx) / r_out;
            const float ux = ((float) ix - cx) / d_px;
            const float r2 = x * x + y * y;
            if (r2 > 1.04f)
            {
                continue;
            }

            float r   = sqrtf(r2);
            float th  = atan2f(y, x);
            float val = 0.0f;

            val += zc[0] * 1.7320508f * (2.0f * r2 - 1.0f);
            val += zc[1] * 2.4494897f * r2 * cosf(2.0f * th);
            val += zc[2] * 2.4494897f * r2 * sinf(2.0f * th);
            val += zc[3] * 2.8284271f * (3.0f * r2 - 2.0f) * r * cosf(th);
            val += zc[4] * 2.8284271f * (3.0f * r2 - 2.0f) * r * sinf(th);
            val += zc[5] * 2.8284271f * r * r2 * cosf(3.0f * th);
            val += zc[6] * 2.8284271f * r * r2 * sinf(3.0f * th);
            val += zc[7] * 2.2360680f * (6.0f * r2 * r2 - 6.0f * r2 + 1.0f);

            for (int m = 0; m < n_md; m++)
            {
                float ang = CACAOSIM_TWO_PI_F * ((float) mkx[m] * ux + (float) mky[m] * uy) +
                            mphi[m];
                val += mamp[m] * cosf(ang);
            }

            uint64_t idx   = (uint64_t) iy * nwf + ix;
            wfe_out_m[idx] = val;

            double wt = (pup_amp != NULL) ? (double) pup_amp[idx] : ((r2 <= 1.0f) ? 1.0 : 0.0);
            if (wt > 0.0)
            {
                w_sum += wt;
                m_sum += wt * (double) val;
            }
        }
    }

    if (w_sum <= 0.0)
    {
        return;
    }

    const double mean = m_sum / w_sum;
    double       var  = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        double wt = 0.0;
        if (pup_amp != NULL)
        {
            wt = (double) pup_amp[i];
        }
        else if (wfe_out_m[i] != 0.0f)
        {
            uint32_t ix = (uint32_t) (i % nwf);
            uint32_t iy = (uint32_t) (i / nwf);
            float    x  = ((float) ix - cx) / r_out;
            float    y  = ((float) iy - cy) / r_out;
            wt          = (x * x + y * y <= 1.0f) ? 1.0 : 0.0;
        }
        if (wt > 0.0)
        {
            double dv = (double) wfe_out_m[i] - mean;
            var += wt * dv * dv;
        }
    }

    double rms   = sqrt(var / w_sum);
    float  scale = (rms > 1e-30) ? (rms_m / (float) rms) : 0.0f;

    for (uint64_t i = 0; i < npix; i++)
    {
        int in_pup = 0;
        if (pup_amp != NULL)
        {
            in_pup = (pup_amp[i] > 0.0f) ? 1 : 0;
        }
        else if (wfe_out_m[i] != 0.0f)
        {
            uint32_t ix = (uint32_t) (i % nwf);
            uint32_t iy = (uint32_t) (i / nwf);
            float    x  = ((float) ix - cx) / r_out;
            float    y  = ((float) iy - cy) / r_out;
            in_pup      = (x * x + y * y <= 1.04f) ? 1 : 0;
        }
        wfe_out_m[i] = in_pup ? (((float) ((double) wfe_out_m[i] - mean)) * scale) : 0.0f;
    }
}
