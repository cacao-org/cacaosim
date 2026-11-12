// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_engine.c
 * @brief   Coronagraphic focal-plane mask (Babinet MFT + padded FFT), Lyot stop, and LOWFS engine
 */

#define _GNU_SOURCE
#include <cblas.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_cuda.h"
#include "coro/cacaosim_coro.h"
#include "pupil/cacaosim_pupil.h"

extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

static void coro_precompute_babinet(cacaosim_coro_ctx_t *ctx)
{
    const uint32_t nwf  = ctx->cfg.nwf;
    const uint32_t m    = ctx->n_mft;
    const float    pdim = (ctx->cfg.pup_diam > 0.05f) ? ctx->cfg.pup_diam : 0.9375f;
    const float    dpix = pdim * (float) nwf;
    const float    p0   = 0.5f * (float) (nwf - 1u);
    const float    r_ld = (ctx->cfg.fpm_rad_ld > 0.05f) ? ctx->cfg.fpm_rad_ld : 3.0f;
    const float    fov  = (2.4f * r_ld > 2.0f) ? (2.4f * r_ld) : 2.0f;
    const float    du   = fov / (float) m;
    const float    nrm  = sqrtf(du / dpix);

    for (uint32_t k = 0; k < m; k++)
    {
        const float u = (((float) k + 0.5f) / (float) m - 0.5f) * fov;
        for (uint32_t p = 0; p < nwf; p++)
        {
            const float xt  = ((float) p - p0) / dpix;
            const float ang = -CACAOSIM_TWO_PI_F * xt * u;
            float       s   = 0.0f;
            float       co  = 1.0f;
            sincosf(ang, &s, &co);

            size_t idx_kp               = ((size_t) k * nwf + p) * 2u;
            size_t idx_pk               = ((size_t) p * m + k) * 2u;
            ctx->fwd_U_cplx[idx_kp]     = nrm * co;
            ctx->fwd_U_cplx[idx_kp + 1] = nrm * s;
            ctx->fwd_V_cplx[idx_pk]     = nrm * co;
            ctx->fwd_V_cplx[idx_pk + 1] = nrm * s;

            ctx->inv_U_cplx[idx_pk]     = nrm * co;
            ctx->inv_U_cplx[idx_pk + 1] = -nrm * s;
            ctx->inv_V_cplx[idx_kp]     = nrm * co;
            ctx->inv_V_cplx[idx_kp + 1] = -nrm * s;
        }
    }

    const int   nsub = 8;
    const float invs = 1.0f / (float) (nsub * nsub);
    const float r2   = r_ld * r_ld;

    for (uint32_t ky = 0; ky < m; ky++)
    {
        const float vc = (((float) ky + 0.5f) / (float) m - 0.5f) * fov;
        for (uint32_t kx = 0; kx < m; kx++)
        {
            const float uc   = (((float) kx + 0.5f) / (float) m - 0.5f) * fov;
            int         hits = 0;
            for (int sy = 0; sy < nsub; sy++)
            {
                float vs = vc + (((float) sy + 0.5f) / (float) nsub - 0.5f) * du;
                for (int sx = 0; sx < nsub; sx++)
                {
                    float us = uc + (((float) sx + 0.5f) / (float) nsub - 0.5f) * du;
                    if (us * us + vs * vs <= r2)
                    {
                        hits++;
                    }
                }
            }
            ctx->fpm_weight[(size_t) ky * m + kx] = (float) hits * invs;
        }
    }
}

static void coro_precompute_fft_mask(cacaosim_coro_ctx_t *ctx)
{
    const uint32_t nfft     = ctx->nfft;
    const int32_t  half     = (int32_t) (nfft / 2u);
    const float    inv_nfft2 = 1.0f / (float) ctx->npix_fft;
    const int32_t  charge   = (ctx->cfg.vortex_charge == 4) ? 4 : 2;

    for (uint32_t m = 0; m < nfft; m++)
    {
        float ang                      = -CACAOSIM_PI_F * (float) m / (float) nfft;
        ctx->shift_1d_cplx[2u * m]     = cosf(ang);
        ctx->shift_1d_cplx[2u * m + 1] = sinf(ang);
    }

    for (uint32_t jy = 0; jy < nfft; jy++)
    {
        int32_t iv = (jy < (uint32_t) half) ? (int32_t) jy : ((int32_t) jy - (int32_t) nfft);
        float   v  = (float) iv + 0.5f;
        for (uint32_t ix = 0; ix < nfft; ix++)
        {
            int32_t iu  = (ix < (uint32_t) half) ? (int32_t) ix : ((int32_t) ix - (int32_t) nfft);
            float   u   = (float) iu + 0.5f;
            size_t  idx = ((size_t) jy * nfft + ix) * 2u;
            float   mr  = 1.0f;
            float   mi  = 0.0f;

            if (ctx->cfg.mode == CACAOSIM_CORO_4QPM)
            {
                mr = ((u > 0.0f) ? 1.0f : -1.0f) * ((v > 0.0f) ? 1.0f : -1.0f);
                mi = 0.0f;
            }
            else
            {
                float th = (float) charge * atan2f(v, u);
                mr       = cosf(th);
                mi       = sinf(th);
            }

            ctx->fft_mask_cplx[idx]      = inv_nfft2 * mr;
            ctx->fft_mask_cplx[idx + 1u] = inv_nfft2 * mi;
        }
    }
}

void MILK_HOT cacaosim_coro_step_cpu(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    float               *out_opd,
    float               *out_amp,
    float               *lowfs_opd,
    float               *lowfs_amp)
{
    const uint32_t nwf        = ctx->cfg.nwf;
    const uint64_t npix_wf    = ctx->npix_wf;
    const float    lam        = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 1.65f;
    const float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    const float    rad_to_opd = lam / CACAOSIM_TWO_PI_F;

    if (ctx->is_babinet)
    {
        const uint32_t m      = ctx->n_mft;
        float *__restrict ep  = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);
        float *__restrict ec  = MILK_ASSUME_ALIGNED(ctx->cpu_ecore_cplx);

        for (uint64_t p = 0; p < npix_wf; p++)
        {
            float amp = (in_amp != NULL) ? in_amp[p] : 1.0f;
            if (amp <= 0.0f)
            {
                ep[2u * p]      = 0.0f;
                ep[2u * p + 1u] = 0.0f;
                continue;
            }
            float phi = ((in_opd != NULL) ? in_opd[p] : 0.0f) * opd_to_rad;
            float s   = 0.0f;
            float co  = 1.0f;
            sincosf(phi, &s, &co);
            ep[2u * p]      = amp * co;
            ep[2u * p + 1u] = amp * s;
        }

        const float c_one[2]  = { 1.0f, 0.0f };
        const float c_zero[2] = { 0.0f, 0.0f };

        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) nwf, (int) nwf,
                    c_one, ctx->fwd_U_cplx, (int) nwf, ep, (int) nwf, c_zero, ctx->cpu_tmp_cplx,
                    (int) nwf);
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) m, (int) nwf, c_one,
                    ctx->cpu_tmp_cplx, (int) nwf, ctx->fwd_V_cplx, (int) m, c_zero,
                    ctx->cpu_efoc_cplx, (int) m);

        float *__restrict       ef = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);
        const float *__restrict fw = MILK_ASSUME_ALIGNED(ctx->fpm_weight);
#pragma omp simd
        for (uint64_t i = 0; i < ctx->npix_mft; i++)
        {
            float w        = fw[i];
            ef[2u * i]    *= w;
            ef[2u * i + 1] *= w;
        }

        /* Reconstruct core pupil field ecore = inv_U * (M * efoc) * inv_V */
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) m, (int) m, c_one,
                    ctx->inv_U_cplx, (int) m, ef, (int) m, c_zero, ctx->cpu_tmp_cplx, (int) m);
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) nwf, (int) m,
                    c_one, ctx->cpu_tmp_cplx, (int) m, ctx->inv_V_cplx, (int) nwf, c_zero, ec,
                    (int) nwf);

        float t_amp = ctx->cfg.fpm_amp;
        float ph    = (ctx->cfg.mode == CACAOSIM_CORO_PHASE_DIMPLE) ? ctx->cfg.fpm_phase_rad : 0.0f;
        float c_re  = t_amp * cosf(ph) - 1.0f;
        float c_im  = t_amp * sinf(ph);
        float refl  = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;

        for (uint64_t p = 0; p < npix_wf; p++)
        {
            float pr = ep[2u * p];
            float pi = ep[2u * p + 1u];
            float cr = ec[2u * p];
            float ci = ec[2u * p + 1u];

            /* Post-FPM + Lyot stop field */
            float ly  = ctx->lyot_mask[p];
            float lr  = ly * (pr + (c_re * cr - c_im * ci));
            float li  = ly * (pi + (c_re * ci + c_im * cr));
            float amp = hypotf(lr, li);
            out_amp[p] = amp;
            out_opd[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

            if (lowfs_opd != NULL && lowfs_amp != NULL)
            {
                float la     = refl * hypotf(cr, ci);
                lowfs_amp[p] = la;
                lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ci, cr) * rad_to_opd) : 0.0f;
            }
        }
    }
    else
    {
        /* Full-plane padded 2D FFT coronagraph (Vortex / 4QPM) */
        const uint32_t          nfft    = ctx->nfft;
        const uint32_t          pad_off = (nfft - nwf) / 2u;
        float *__restrict       fbuf    = MILK_ASSUME_ALIGNED(ctx->cpu_fft_buf);
        const float *__restrict sh1d    = ctx->shift_1d_cplx;
        const float *__restrict fmsk    = MILK_ASSUME_ALIGNED(ctx->fft_mask_cplx);

        __builtin_memset(fbuf, 0, ctx->npix_fft * 2u * sizeof(float));

        for (uint32_t yp = 0; yp < nwf; yp++)
        {
            uint32_t gy    = yp + pad_off;
            float    sy_re = sh1d[2u * gy];
            float    sy_im = sh1d[2u * gy + 1u];
            float   *dst   = &fbuf[((size_t) gy * nfft + pad_off) * 2u];
            uint64_t pbase = (uint64_t) yp * nwf;

            for (uint32_t xp = 0; xp < nwf; xp++)
            {
                uint64_t p   = pbase + xp;
                float    amp = (in_amp != NULL) ? in_amp[p] : 1.0f;
                if (amp <= 0.0f)
                {
                    continue;
                }
                float    phi   = ((in_opd != NULL) ? in_opd[p] : 0.0f) * opd_to_rad;
                float    s     = 0.0f;
                float    co    = 1.0f;
                sincosf(phi, &s, &co);
                float    pr    = amp * co;
                float    pi    = amp * s;
                uint32_t gx    = xp + pad_off;
                float    sx_re = sh1d[2u * gx];
                float    sx_im = sh1d[2u * gx + 1u];
                float    sr    = sx_re * sy_re - sx_im * sy_im;
                float    si    = sx_re * sy_im + sx_im * sy_re;
                dst[2u * xp]      = pr * sr - pi * si;
                dst[2u * xp + 1u] = pr * si + pi * sr;
            }
        }

        fftwf_execute((fftwf_plan) ctx->fftw_fwd);

#pragma omp simd
        for (uint64_t i = 0; i < ctx->npix_fft; i++)
        {
            float er        = fbuf[2u * i];
            float ei        = fbuf[2u * i + 1u];
            float mr        = fmsk[2u * i];
            float mi        = fmsk[2u * i + 1u];
            fbuf[2u * i]    = er * mr - ei * mi;
            fbuf[2u * i + 1] = er * mi + ei * mr;
        }

        fftwf_execute((fftwf_plan) ctx->fftw_inv);

        float refl = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;
        for (uint32_t yp = 0; yp < nwf; yp++)
        {
            uint32_t    gy    = yp + pad_off;
            float       sy_re = sh1d[2u * gy];
            float       sy_im = -sh1d[2u * gy + 1u]; /* conjugate post-shift */
            const float *src  = &fbuf[((size_t) gy * nfft + pad_off) * 2u];
            uint64_t    pbase = (uint64_t) yp * nwf;

            for (uint32_t xp = 0; xp < nwf; xp++)
            {
                uint64_t p     = pbase + xp;
                uint32_t gx    = xp + pad_off;
                float    sx_re = sh1d[2u * gx];
                float    sx_im = -sh1d[2u * gx + 1u];
                float    sr    = sx_re * sy_re - sx_im * sy_im;
                float    si    = sx_re * sy_im + sx_im * sy_re;
                float    er    = src[2u * xp];
                float    ei    = src[2u * xp + 1u];
                float    ur    = er * sr - ei * si;
                float    ui    = er * si + ei * sr;

                float ly   = ctx->lyot_mask[p];
                float lr   = ly * ur;
                float li   = ly * ui;
                float amp  = hypotf(lr, li);
                out_amp[p] = amp;
                out_opd[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

                /* Reflective Lyot-stop ring pickoff for LOWFS when using phase masks */
                if (lowfs_opd != NULL && lowfs_amp != NULL)
                {
                    float r_ly   = refl * (1.0f - ly);
                    float la     = r_ly * hypotf(ur, ui);
                    lowfs_amp[p] = la;
                    lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                }
            }
        }
    }
}

int MILK_COLD cacaosim_coro_init(
    cacaosim_coro_ctx_t       *ctx,
    const cacaosim_coro_cfg_t *cfg,
    const float               *ext_lyot,
    void                      *ext_stream)
{
    if (ctx == NULL || cfg == NULL || cfg->nwf < 8)
    {
        return -1;
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg        = *cfg;
    ctx->npix_wf    = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->is_babinet = (cfg->mode == CACAOSIM_CORO_LYOT_SPOT ||
                       cfg->mode == CACAOSIM_CORO_PHASE_DIMPLE)
                          ? 1
                          : 0;

    if (openblas_set_num_threads != NULL)
    {
        openblas_set_num_threads(1);
    }

    ctx->lyot_mask = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    if (!ctx->lyot_mask)
    {
        return -1;
    }

    if (ext_lyot != NULL)
    {
        __builtin_memcpy(ctx->lyot_mask, ext_lyot, ctx->npix_wf * sizeof(float));
    }
    else
    {
        float pdim  = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 0.9375f;
        float outer = (cfg->lyot_outer > 0.05f) ? cfg->lyot_outer : 0.85f;
        float obs_p = (cfg->lyot_obs >= 0.0f) ? cfg->lyot_obs : 0.36f;
        cacaosim_pupil_cfg_t lcfg = {
            .nwf        = cfg->nwf,
            .pup_diam   = pdim * outer,
            .cobs       = (outer > 1e-4f) ? (obs_p / outer) : 0.0f,
            .nspider    = cfg->lyot_nspider,
            .spider_w   = (outer > 1e-4f) ? (cfg->lyot_spider_w / outer) : 0.0f,
            .spider_deg = cfg->lyot_spider_deg,
        };
        if (lcfg.cobs > 0.95f)
        {
            lcfg.cobs = 0.95f;
        }
        cacaosim_make_pupil(&lcfg, ctx->lyot_mask);
    }

    if (ctx->is_babinet)
    {
        ctx->n_mft      = (cfg->n_mft >= 16u) ? cfg->n_mft : 48u;
        ctx->npix_mft   = (uint64_t) ctx->n_mft * ctx->n_mft;
        size_t uv_elems = (size_t) ctx->n_mft * cfg->nwf * 2u;

        ctx->fwd_U_cplx     = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->fwd_V_cplx     = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->inv_U_cplx     = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->inv_V_cplx     = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->fpm_weight     = (float *) cacaosim_aligned_calloc(ctx->npix_mft, sizeof(float));
        ctx->cpu_epup_cplx  = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
        ctx->cpu_ecore_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
        ctx->cpu_tmp_cplx   = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->cpu_efoc_cplx  = (float *) cacaosim_aligned_calloc(ctx->npix_mft * 2u, sizeof(float));

        if (!ctx->fwd_U_cplx || !ctx->fwd_V_cplx || !ctx->inv_U_cplx || !ctx->inv_V_cplx ||
            !ctx->fpm_weight || !ctx->cpu_epup_cplx || !ctx->cpu_ecore_cplx ||
            !ctx->cpu_tmp_cplx || !ctx->cpu_efoc_cplx)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
        coro_precompute_babinet(ctx);
    }
    else
    {
        uint32_t pad       = (cfg->pad_factor >= 2u) ? cfg->pad_factor : 4u;
        ctx->nfft          = cfg->nwf * pad;
        ctx->npix_fft      = (uint64_t) ctx->nfft * ctx->nfft;
        ctx->shift_1d_cplx = (float *) cacaosim_aligned_calloc((size_t) ctx->nfft * 2u,
                                                               sizeof(float));
        ctx->fft_mask_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        ctx->cpu_fft_buf   = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));

        if (!ctx->shift_1d_cplx || !ctx->fft_mask_cplx || !ctx->cpu_fft_buf)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
        coro_precompute_fft_mask(ctx);

        int n         = (int) ctx->nfft;
        ctx->fftw_fwd = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_buf, (fftwf_complex *) ctx->cpu_fft_buf,
            FFTW_FORWARD, FFTW_ESTIMATE);
        ctx->fftw_inv = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_buf, (fftwf_complex *) ctx->cpu_fft_buf,
            FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!ctx->fftw_fwd || !ctx->fftw_inv)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
    }

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_coro_cuda_init(ctx, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

void MILK_COLD cacaosim_coro_free(cacaosim_coro_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_coro_cuda_free(ctx);
    }
    if (ctx->fftw_fwd)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_fwd);
    }
    if (ctx->fftw_inv)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_inv);
    }
    free(ctx->lyot_mask);
    free(ctx->fwd_U_cplx);
    free(ctx->fwd_V_cplx);
    free(ctx->inv_U_cplx);
    free(ctx->inv_V_cplx);
    free(ctx->fpm_weight);
    free(ctx->shift_1d_cplx);
    free(ctx->fft_mask_cplx);
    free(ctx->cpu_epup_cplx);
    free(ctx->cpu_ecore_cplx);
    free(ctx->cpu_tmp_cplx);
    free(ctx->cpu_efoc_cplx);
    free(ctx->cpu_fft_buf);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

void MILK_HOT cacaosim_coro_step_beam(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    float               *out_opd,
    float               *out_amp,
    float               *lowfs_opd,
    float               *lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp)
{
    if (ctx->use_cuda)
    {
        cacaosim_coro_cuda_step(ctx, in_opd, in_amp, out_opd, out_amp, lowfs_opd, lowfs_amp,
                                d_in_opd, d_in_amp, d_out_opd, d_out_amp, d_lowfs_opd,
                                d_lowfs_amp);
        return;
    }
    cacaosim_coro_step_cpu(ctx, in_opd, in_amp, out_opd, out_amp, lowfs_opd, lowfs_amp);
}
