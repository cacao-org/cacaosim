// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam_engine.c
 * @brief   CPU (OpenBLAS MFT + FFTW3f) and host orchestration for focal plane PSF/camera simulator
 */

#define _GNU_SOURCE
#include <cblas.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milk_config.h"
#include "milk_types.h"
#include "common/cacaosim_cuda.h"
#include "fpcam/cacaosim_fpcam.h"

extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

static void fpcam_precompute_mft(cacaosim_fpcam_ctx_t *ctx)
{
    const uint32_t nwf  = ctx->cfg.nwf;
    const uint32_t ncam = ctx->cfg.ncam;
    const float    lam0 = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 1.65f;
    const float    dpix = ctx->cfg.pup_diam * (float) nwf;
    const float    p0   = 0.5f * (float) (nwf - 1u);
    const float    f0   = (float) (ncam / 2u);
    const float    nrm  = 1.0f / sqrtf(ctx->pup_sum_amp);
    const float    ps0  = (ctx->cfg.pixscale_ld > 0.05f) ? ctx->cfg.pixscale_ld : 2.5f;

    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        const float sc = ps0 * (ctx->lam_um[c] / lam0);
        float      *U  = &ctx->mft_U_cplx[(size_t) c * ncam * nwf * 2u];
        float      *V  = &ctx->mft_V_cplx[(size_t) c * nwf * ncam * 2u];

        for (uint32_t yf = 0; yf < ncam; yf++)
        {
            const float eta = ((float) yf - f0) / sc;
            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                const float yt  = ((float) yp - p0) / dpix;
                const float ang = -CACAOSIM_TWO_PI_F * yt * eta;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);
                size_t idx  = ((size_t) yf * nwf + yp) * 2u;
                U[idx]      = nrm * co;
                U[idx + 1u] = nrm * s;
            }
        }

        for (uint32_t xp = 0; xp < nwf; xp++)
        {
            const float xt = ((float) xp - p0) / dpix;
            for (uint32_t xf = 0; xf < ncam; xf++)
            {
                const float xi  = ((float) xf - f0) / sc;
                const float ang = -CACAOSIM_TWO_PI_F * xt * xi;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);
                size_t idx  = ((size_t) xp * ncam + xf) * 2u;
                V[idx]      = nrm * co;
                V[idx + 1u] = nrm * s;
            }
        }
    }
}

static void MILK_HOT fpcam_build_epup_cpu(
    const float *__restrict pup_amp,
    const float *__restrict atm_amp,
    const float *__restrict wf_in,
    const float *__restrict dm_opd,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd,
    float                   opd_to_rad,
    uint64_t                npix_wf,
    float *__restrict       epup_cplx)
{
    const int nth = (npix_wf >= 16384ULL) ? omp_get_max_threads() : 1;

#pragma omp parallel for num_threads(nth) if (nth > 1) schedule(static)
    for (uint64_t p = 0; p < npix_wf; p++)
    {
        float amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
        if (atm_amp != NULL)
        {
            amp *= atm_amp[p];
        }
        if (amp <= 0.0f)
        {
            epup_cplx[2u * p]      = 0.0f;
            epup_cplx[2u * p + 1u] = 0.0f;
            continue;
        }
        float opd = 0.0f;
        if (wf_in != NULL)
        {
            opd += wf_to_opd * wf_in[p];
        }
        if (dm_opd != NULL)
        {
            opd -= dm_opd[p];
        }
        if (ncpa_opd != NULL)
        {
            opd += ncpa_opd[p];
        }
        float phi = opd * opd_to_rad;
        float s   = 0.0f;
        float co  = 1.0f;
        sincosf(phi, &s, &co);
        epup_cplx[2u * p]      = amp * co;
        epup_cplx[2u * p + 1u] = amp * s;
    }
}

static void MILK_HOT fpcam_accum_cpu(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    float                 wf_to_opd)
{
    const uint32_t nwf      = ctx->cfg.nwf;
    const uint32_t ncam     = ctx->cfg.ncam;
    const uint64_t npix_cam = ctx->npix_cam;
    const float    inv_nlam = 1.0f / (float) ctx->eff_nlam;
    float *__restrict acc   = MILK_ASSUME_ALIGNED(ctx->cpu_accum_int);

    if (ctx->cfg.prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        const float c_one[2]  = { 1.0f, 0.0f };
        const float c_zero[2] = { 0.0f, 0.0f };

        for (int32_t c = 0; c < ctx->eff_nlam; c++)
        {
            const float opd_to_rad = CACAOSIM_TWO_PI_F / ctx->lam_um[c];
            fpcam_build_epup_cpu(pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, wf_to_opd,
                                 opd_to_rad, ctx->npix_wf, ctx->cpu_epup_cplx);

            const float *Uc = &ctx->mft_U_cplx[(size_t) c * ncam * nwf * 2u];
            const float *Vc = &ctx->mft_V_cplx[(size_t) c * nwf * ncam * 2u];

            /* T [ncam x nwf] = Uc [ncam x nwf] * Epup [nwf x nwf] */
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) ncam, (int) nwf,
                        (int) nwf, c_one, Uc, (int) nwf, ctx->cpu_epup_cplx, (int) nwf, c_zero,
                        ctx->cpu_mft_tmp_cplx, (int) nwf);

            /* Efoc [ncam x ncam] = T [ncam x nwf] * Vc [nwf x ncam] */
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) ncam, (int) ncam,
                        (int) nwf, c_one, ctx->cpu_mft_tmp_cplx, (int) nwf, Vc, (int) ncam,
                        c_zero, ctx->cpu_efoc_cplx, (int) ncam);

            const float *__restrict ef = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);
#pragma omp simd
            for (uint64_t i = 0; i < npix_cam; i++)
            {
                float re = ef[2u * i];
                float im = ef[2u * i + 1u];
                acc[i] += inv_nlam * (re * re + im * im);
            }
        }
    }
    else
    {
        /* Zero-padded 2D FFT crop centered on DC at (ncam/2, ncam/2) */
        const uint32_t nfft   = ctx->nfft;
        const float    inv_s2 = inv_nlam / (ctx->pup_sum_amp * ctx->pup_sum_amp);
        const int32_t  half_c = (int32_t) (ncam / 2u);

        for (int32_t c = 0; c < ctx->eff_nlam; c++)
        {
            const float opd_to_rad = CACAOSIM_TWO_PI_F / ctx->lam_um[c];
            fpcam_build_epup_cpu(pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, wf_to_opd,
                                 opd_to_rad, ctx->npix_wf, ctx->cpu_epup_cplx);

            __builtin_memset(ctx->cpu_fft_in_cplx, 0, ctx->npix_fft * 2u * sizeof(float));
            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                __builtin_memcpy(&ctx->cpu_fft_in_cplx[(size_t) yp * nfft * 2u],
                                 &ctx->cpu_epup_cplx[(size_t) yp * nwf * 2u],
                                 (size_t) nwf * 2u * sizeof(float));
            }

            fftwf_execute((fftwf_plan) ctx->fftw_plan_fwd);

            const float *__restrict fo = ctx->cpu_fft_out_cplx;
            for (uint32_t yf = 0; yf < ncam; yf++)
            {
                int32_t  dy = (int32_t) yf - half_c;
                uint32_t ky = (uint32_t) ((dy % (int32_t) nfft + (int32_t) nfft) % (int32_t) nfft);
                for (uint32_t xf = 0; xf < ncam; xf++)
                {
                    int32_t  dx = (int32_t) xf - half_c;
                    uint32_t kx = (uint32_t) ((dx % (int32_t) nfft + (int32_t) nfft) %
                                              (int32_t) nfft);
                    size_t   k  = ((size_t) ky * nfft + kx) * 2u;
                    float    re = fo[k];
                    float    im = fo[k + 1u];
                    acc[(size_t) yf * ncam + xf] += inv_s2 * (re * re + im * im);
                }
            }
        }
    }
}

int MILK_COLD cacaosim_fpcam_init(
    cacaosim_fpcam_ctx_t       *ctx,
    const cacaosim_fpcam_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream)
{
    if (ctx == NULL || cfg == NULL || pup_amp == NULL || cfg->nwf < 8 || cfg->ncam < 4)
    {
        return -1;
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg      = *cfg;
    uint32_t pad  = (cfg->pad_factor >= 2u) ? cfg->pad_factor : 2u;
    ctx->nfft     = cfg->nwf * pad;
    ctx->npix_wf  = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->npix_fft = (uint64_t) ctx->nfft * ctx->nfft;
    ctx->npix_cam = (uint64_t) cfg->ncam * cfg->ncam;

    double sum_a  = 0.0;
    double sum_a2 = 0.0;
    for (uint64_t p = 0; p < ctx->npix_wf; p++)
    {
        double a = (double) pup_amp[p];
        sum_a += a;
        sum_a2 += a * a;
    }
    if (sum_a <= 1e-12)
    {
        return -1;
    }
    ctx->pup_sum_amp  = (float) sum_a;
    ctx->pup_sum_amp2 = (float) sum_a2;

    float   lam0 = (cfg->lambda_um > 0.05f) ? cfg->lambda_um : 1.65f;
    float   bw   = (cfg->bw_frac > 0.0f) ? cfg->bw_frac : 0.0f;
    int32_t nlam = cfg->nlambda;
    if (nlam < 1 || bw <= 1e-6f)
    {
        nlam = 1;
    }
    if (nlam > CACAOSIM_FPCAM_MAX_LAMBDA)
    {
        nlam = CACAOSIM_FPCAM_MAX_LAMBDA;
    }
    ctx->eff_nlam = nlam;

    for (int32_t c = 0; c < nlam; c++)
    {
        float u        = (nlam > 1) ? (((float) c + 0.5f) / (float) nlam - 0.5f) : 0.0f;
        ctx->lam_um[c] = lam0 * (1.0f + bw * u);
    }

    float dpix = cfg->pup_diam * (float) cfg->nwf;
    if (cfg->prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        ctx->eff_pixscale_ld = (cfg->pixscale_ld > 0.05f) ? cfg->pixscale_ld : 2.5f;
    }
    else
    {
        ctx->eff_pixscale_ld = (float) ctx->nfft / dpix;
    }

    /* Infinite-plane Parseval energy so flux_ph equals total clear-pupil incident photons */
    double inf_acc = 0.0;
    for (int32_t c = 0; c < nlam; c++)
    {
        double sc = (cfg->prop_mode == CACAOSIM_FPCAM_PROP_MFT)
                        ? ((double) ctx->eff_pixscale_ld * (double) ctx->lam_um[c] / (double) lam0)
                        : (double) ctx->eff_pixscale_ld;
        double n_per_axis = (double) dpix * sc;
        inf_acc += (n_per_axis * n_per_axis * sum_a2) / (sum_a * sum_a);
    }
    ctx->inf_flux_norm = (float) (inf_acc / (double) nlam);

    ctx->cpu_epup_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
    ctx->cpu_sub_int   = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_accum_int = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_exp_int   = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_long_int  = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));

    if (!ctx->cpu_epup_cplx || !ctx->cpu_sub_int || !ctx->cpu_accum_int || !ctx->cpu_exp_int ||
        !ctx->cpu_long_int || cacaosim_det_pool_init(&ctx->det_pool, 0, cfg->rng_seed) != 0)
    {
        cacaosim_fpcam_free(ctx);
        return -1;
    }

    if (openblas_set_num_threads != NULL)
    {
        /* Avoid OpenBLAS pthread pool colliding with OpenMP threads on small 128^2 CGEMMs */
        openblas_set_num_threads(1);
    }

    if (cfg->prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        size_t uv_elems       = (size_t) nlam * cfg->ncam * cfg->nwf * 2u;
        ctx->mft_U_cplx       = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->mft_V_cplx       = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->cpu_mft_tmp_cplx = (float *) cacaosim_aligned_calloc(
            (size_t) cfg->ncam * cfg->nwf * 2u, sizeof(float));
        ctx->cpu_efoc_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_cam * 2u, sizeof(float));
        if (!ctx->mft_U_cplx || !ctx->mft_V_cplx || !ctx->cpu_mft_tmp_cplx || !ctx->cpu_efoc_cplx)
        {
            cacaosim_fpcam_free(ctx);
            return -1;
        }
        fpcam_precompute_mft(ctx);
    }
    else
    {
        ctx->cpu_fft_in_cplx =
            (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        ctx->cpu_fft_out_cplx =
            (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        if (!ctx->cpu_fft_in_cplx || !ctx->cpu_fft_out_cplx)
        {
            cacaosim_fpcam_free(ctx);
            return -1;
        }
        int n = (int) ctx->nfft;
        ctx->fftw_plan_fwd = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_in_cplx, (fftwf_complex *) ctx->cpu_fft_out_cplx,
            FFTW_FORWARD, FFTW_ESTIMATE);
        if (ctx->fftw_plan_fwd == NULL)
        {
            cacaosim_fpcam_free(ctx);
            return -1;
        }
    }

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_fpcam_cuda_init(ctx, pup_amp, ncpa_opd, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

void MILK_COLD cacaosim_fpcam_free(cacaosim_fpcam_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_fpcam_cuda_free(ctx);
    }
    if (ctx->fftw_plan_fwd)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_plan_fwd);
    }
    cacaosim_det_pool_free(&ctx->det_pool);
    free(ctx->mft_U_cplx);
    free(ctx->mft_V_cplx);
    free(ctx->cpu_epup_cplx);
    free(ctx->cpu_mft_tmp_cplx);
    free(ctx->cpu_efoc_cplx);
    free(ctx->cpu_fft_in_cplx);
    free(ctx->cpu_fft_out_cplx);
    free(ctx->cpu_sub_int);
    free(ctx->cpu_accum_int);
    free(ctx->cpu_exp_int);
    free(ctx->cpu_long_int);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

void cacaosim_fpcam_reset_long(cacaosim_fpcam_ctx_t *ctx)
{
    if (ctx == NULL || ctx->cpu_long_int == NULL)
    {
        return;
    }
    __builtin_memset(ctx->cpu_long_int, 0, ctx->npix_cam * sizeof(float));
    ctx->long_cnt         = 0;
    ctx->long_strehl_peak = 0.0f;
}

void MILK_HOT cacaosim_fpcam_accum_subframe(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    const float          *d_dm_opd,
    float                 turb_gain)
{
    float        wf_to_opd = cacaosim_wf_to_opd_scale(ctx->cfg.wf_unit, ctx->cfg.ref_lambda,
                                                      turb_gain);
    const float *wf_eff    = (turb_gain != 0.0f) ? wf_in : NULL;

    if (ctx->use_cuda)
    {
        if (ctx->subframe_cnt == 0u)
        {
            cacaosim_fpcam_cuda_reset_accum(ctx);
        }
        cacaosim_fpcam_cuda_accum(ctx, NULL, atm_amp, wf_eff, dm_opd, NULL, NULL, d_dm_opd,
                                  wf_to_opd);
    }
    else
    {
        if (ctx->subframe_cnt == 0u)
        {
            __builtin_memset(ctx->cpu_accum_int, 0, ctx->npix_cam * sizeof(float));
        }
        fpcam_accum_cpu(ctx, pup_amp, atm_amp, wf_eff, dm_opd, ncpa_opd, wf_to_opd);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_fpcam_accum_beam(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    if (ctx->use_cuda)
    {
        if (ctx->subframe_cnt == 0u)
        {
            cacaosim_fpcam_cuda_reset_accum(ctx);
        }
        cacaosim_fpcam_cuda_accum(ctx, beam_amp, NULL, beam_opd, NULL, d_beam_amp, d_beam_opd,
                                  NULL, 1.0f);
    }
    else
    {
        if (ctx->subframe_cnt == 0u)
        {
            __builtin_memset(ctx->cpu_accum_int, 0, ctx->npix_cam * sizeof(float));
        }
        fpcam_accum_cpu(ctx, beam_amp, NULL, beam_opd, NULL, ncpa_opd, 1.0f);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_fpcam_readout(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long)
{
    if (ctx->subframe_cnt == 0u)
    {
        return;
    }

    const cacaosim_det_cfg_t *det      = (det_live != NULL) ? det_live : &ctx->cfg.det;
    const uint64_t            npix     = ctx->npix_cam;
    const uint32_t            c0       = ctx->cfg.ncam / 2u;
    const uint64_t            axis_idx = (uint64_t) c0 * ctx->cfg.ncam + c0;
    ctx->frame_seq++;

    if (ctx->use_cuda)
    {
        cacaosim_fpcam_cuda_readout(ctx, det, ctx->cpu_exp_int, out_u16, out_f32);
    }
    else
    {
        const float             inv_sub = 1.0f / (float) ctx->subframe_cnt;
        const float *__restrict acc     = MILK_ASSUME_ALIGNED(ctx->cpu_accum_int);
        float *__restrict       exp     = MILK_ASSUME_ALIGNED(ctx->cpu_exp_int);
#pragma omp simd
        for (uint64_t i = 0; i < npix; i++)
        {
            exp[i] = acc[i] * inv_sub;
        }
        cacaosim_detector_readout_cpu(&ctx->det_pool, det, exp, npix, ctx->inf_flux_norm, out_u16,
                                      out_f32);
    }

    /* Update Strehl telemetry and cumulative long-exposure PSF */
    const float *__restrict exp      = MILK_ASSUME_ALIGNED(ctx->cpu_exp_int);
    float *__restrict       lbuf     = MILK_ASSUME_ALIGNED(ctx->cpu_long_int);
    ctx->long_cnt++;
    const float             inv_long = 1.0f / (float) ctx->long_cnt;

    float max_exp  = 0.0f;
    float max_long = 0.0f;
    for (uint64_t i = 0; i < npix; i++)
    {
        float v = exp[i];
        if (v > max_exp)
        {
            max_exp = v;
        }
        lbuf[i] += v;
        float lv = lbuf[i] * inv_long;
        if (lv > max_long)
        {
            max_long = lv;
        }
        if (out_long != NULL)
        {
            out_long[i] = lv;
        }
    }

    ctx->last_strehl_axis = exp[axis_idx];
    ctx->last_strehl_peak = max_exp;
    ctx->long_strehl_peak = max_long;
    ctx->subframe_cnt     = 0u;
}

void MILK_HOT cacaosim_fpcam_step(
    cacaosim_fpcam_ctx_t     *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long)
{
    cacaosim_fpcam_accum_subframe(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, d_dm_opd,
                                  turb_gain);
    cacaosim_fpcam_readout(ctx, det_live, out_u16, out_f32, out_long);
}
