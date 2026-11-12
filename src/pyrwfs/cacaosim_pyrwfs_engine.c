// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs_engine.c
 * @brief   Shifted complex pyramid mask synthesis and L2-cached OpenMP + FFTW3f PyWFS engine
 *
 * Implements the (N_mod + 1) 2D FFT reduction: a single forward FFT transforms the pupil field
 * to the focal plane once per frame, followed by multiplication with N_mod shifted analytic
 * 4-sided pyramid masks and N_mod inverse FFTs accumulated in thread-private L2 buffers.
 */

#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_cuda.h"
#include "pyrwfs/cacaosim_pyrwfs.h"

/**
 * pyrwfs_bin_image_cpu() - Area-conserving resample/bin from [nfft x nfft] to [ncam x ncam]
 */
static void MILK_HOT pyrwfs_bin_image_cpu(
    const float *__restrict src,
    uint32_t                nfft,
    uint32_t                ncam,
    float *__restrict       dst)
{
    if (ncam == nfft)
    {
        __builtin_memcpy(dst, src, (size_t) nfft * nfft * sizeof(float));
        return;
    }

    /* Fast path: exact integer box binning */
    if (nfft > ncam && (nfft % ncam) == 0u)
    {
        const uint32_t bin = nfft / ncam;
        for (uint32_t cy = 0; cy < ncam; cy++)
        {
            for (uint32_t cx = 0; cx < ncam; cx++)
            {
                float    acc = 0.0f;
                uint32_t y0  = cy * bin;
                uint32_t x0  = cx * bin;
                for (uint32_t by = 0; by < bin; by++)
                {
                    const float *row = &src[(size_t) (y0 + by) * nfft + x0];
                    for (uint32_t bx = 0; bx < bin; bx++)
                    {
                        acc += row[bx];
                    }
                }
                dst[(size_t) cy * ncam + cx] = acc;
            }
        }
        return;
    }

    /* General fractional box integration */
    const float scale = (float) nfft / (float) ncam;
    for (uint32_t cy = 0; cy < ncam; cy++)
    {
        float fy0 = (float) cy * scale;
        float fy1 = (float) (cy + 1u) * scale;
        int   iy0 = (int) floorf(fy0);
        int   iy1 = (int) ceilf(fy1);
        for (uint32_t cx = 0; cx < ncam; cx++)
        {
            float fx0 = (float) cx * scale;
            float fx1 = (float) (cx + 1u) * scale;
            int   ix0 = (int) floorf(fx0);
            int   ix1 = (int) ceilf(fx1);
            float acc = 0.0f;

            for (int iy = iy0; iy < iy1; iy++)
            {
                if (iy < 0 || iy >= (int) nfft)
                {
                    continue;
                }
                float wy0 = (fy0 > (float) iy) ? fy0 : (float) iy;
                float wy1 = (fy1 < (float) (iy + 1)) ? fy1 : (float) (iy + 1);
                float wy  = wy1 - wy0;
                if (wy <= 0.0f)
                {
                    continue;
                }
                for (int ix = ix0; ix < ix1; ix++)
                {
                    if (ix < 0 || ix >= (int) nfft)
                    {
                        continue;
                    }
                    float wx0 = (fx0 > (float) ix) ? fx0 : (float) ix;
                    float wx1 = (fx1 < (float) (ix + 1)) ? fx1 : (float) (ix + 1);
                    float wx  = wx1 - wx0;
                    if (wx > 0.0f)
                    {
                        acc += wx * wy * src[(size_t) iy * nfft + (uint32_t) ix];
                    }
                }
            }
            dst[(size_t) cy * ncam + cx] = acc;
        }
    }
}

/**
 * pyr_1d_edge() - Exact subpixel-integrated 1D complex pyramid facet over [u - 0.5, u + 0.5]
 */
static inline void pyr_1d_edge(
    float  u,
    float  alpha,
    float  roof2,
    float *re_out,
    float *im_out)
{
    if (roof2 > 0.0f)
    {
        float a = sqrtf(u * u + roof2);
        *re_out = cosf(-alpha * a);
        *im_out = sinf(-alpha * a);
        return;
    }
    float au = fabsf(u);
    if (alpha < 1e-6f)
    {
        *re_out = 1.0f;
        *im_out = 0.0f;
        return;
    }
    if (au >= 0.5f)
    {
        float sinc_h = sinf(0.5f * alpha) / (0.5f * alpha);
        *re_out      = sinc_h * cosf(alpha * au);
        *im_out      = -sinc_h * sinf(alpha * au);
    }
    else
    {
        float cu = cosf(alpha * au);
        *re_out  = (2.0f * cu * sinf(0.5f * alpha)) / alpha;
        *im_out  = -(2.0f * (1.0f - cu * cosf(0.5f * alpha))) / alpha;
    }
}

void MILK_COLD cacaosim_pyrwfs_update_masks(
    cacaosim_pyrwfs_ctx_t *ctx,
    float                  mod_rad_ld)
{
    const uint32_t nfft   = ctx->nfft;
    const int32_t  npts   = ctx->eff_npts;
    const float    diam_p = ((ctx->cfg.pup_diam > 0.05f) ? ctx->cfg.pup_diam : 0.9375f) *
                            (float) ctx->cfg.nwf;
    const float    pix_ld = (float) nfft / diam_p; /* focal pixels per lambda/D */
    const float    sep    = (ctx->cfg.pup_sep > 0.05f && ctx->cfg.pup_sep < 0.9f)
                                ? ctx->cfg.pup_sep
                                : 0.5f;
    /* Shift each sub-pupil by +/- (sep/2 * nfft) pixels => slope alpha = pi * sep */
    const float    alpha  = CACAOSIM_PI_F * sep;
    const float    roof_p = (ctx->cfg.roof_ld > 0.0f) ? (ctx->cfg.roof_ld * pix_ld) : 0.0f;
    const float    roof2  = roof_p * roof_p;
    const float    fstop  = ((ctx->cfg.fstop_frac > 0.1f && ctx->cfg.fstop_frac < 0.49f)
                                 ? ctx->cfg.fstop_frac
                                 : 0.45f) *
                            (float) nfft;
    const float    r_mod  = (npts > 1 && mod_rad_ld > 0.0f) ? (mod_rad_ld * pix_ld) : 0.0f;

    /* Pre-include 1 / (nfft^2 * sqrt(npts)) amplitude normalization inside masks */
    const float    amp_norm = 1.0f / ((float) ((uint64_t) nfft * nfft) * sqrtf((float) npts));
    const int32_t  half     = (int32_t) (nfft / 2u);

#pragma omp parallel for schedule(static)
    for (int32_t k = 0; k < npts; k++)
    {
        float  ang = (CACAOSIM_TWO_PI_F * ((float) k + 0.5f)) / (float) npts;
        float  uk  = r_mod * cosf(ang);
        float  vk  = r_mod * sinf(ang);
        float *mk  = &ctx->masks_cplx[(size_t) k * ctx->npix_fft * 2u];

        for (uint32_t jy = 0; jy < nfft; jy++)
        {
            int32_t iv = (jy < (uint32_t) half) ? (int32_t) jy : ((int32_t) jy - (int32_t) nfft);
            for (uint32_t ix = 0; ix < nfft; ix++)
            {
                int32_t  iu  = (ix < (uint32_t) half) ? (int32_t) ix
                                                      : ((int32_t) ix - (int32_t) nfft);
                uint64_t idx = ((uint64_t) jy * nfft + ix) * 2u;

                /* Zero Nyquist rows/cols and points outside symmetric field stop */
                if (iu == -half || iv == -half || fabsf((float) iu) > fstop ||
                    fabsf((float) iv) > fstop)
                {
                    mk[idx]      = 0.0f;
                    mk[idx + 1u] = 0.0f;
                    continue;
                }

                float u = (float) iu + uk;
                float v = (float) iv + vk;
                float rx, ix_c, ry, iy_c;
                pyr_1d_edge(u, alpha, roof2, &rx, &ix_c);
                pyr_1d_edge(v, alpha, roof2, &ry, &iy_c);

                mk[idx]      = amp_norm * (rx * ry - ix_c * iy_c);
                mk[idx + 1u] = amp_norm * (rx * iy_c + ix_c * ry);
            }
        }
    }

    ctx->cached_mod_rad = mod_rad_ld;
    if (ctx->use_cuda)
    {
        cacaosim_pyrwfs_cuda_upload_masks(ctx);
    }
}

/**
 * pyrwfs_propagate_cpu() - Run 1 forward FFT + L2-local OpenMP N_mod inverse FFTs + binning
 */
static void MILK_HOT pyrwfs_propagate_cpu(
    cacaosim_pyrwfs_ctx_t   *ctx,
    const float *__restrict  pup_amp,
    const float *__restrict  atm_amp,
    const float *__restrict  wf_in,
    const float *__restrict  dm_opd,
    const float *__restrict  ncpa_opd,
    float                    wf_to_opd)
{
    const uint32_t nwf        = ctx->cfg.nwf;
    const uint32_t nfft       = ctx->nfft;
    const uint32_t pad_off    = (nfft - nwf) / 2u;
    const float    lam        = (ctx->cfg.lambda_um > 0.1f) ? ctx->cfg.lambda_um : 0.75f;
    const float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;

    __builtin_memset(ctx->cpu_pup_cplx, 0, (size_t) ctx->npix_fft * 2u * sizeof(float));

    for (uint32_t jy = 0; jy < nwf; jy++)
    {
        float   *dst_row = &ctx->cpu_pup_cplx[((size_t) (jy + pad_off) * nfft + pad_off) * 2u];
        uint64_t pbase   = (uint64_t) jy * nwf;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            uint64_t p   = pbase + ix;
            float    amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
            if (amp == 0.0f)
            {
                continue;
            }
            if (atm_amp != NULL)
            {
                amp *= atm_amp[p];
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
            float phi             = opd * opd_to_rad;
            dst_row[2u * ix]      = amp * cosf(phi);
            dst_row[2u * ix + 1u] = amp * sinf(phi);
        }
    }

    /* 1. Single 2D forward FFT: pupil -> focal plane */
    fftwf_execute_dft((fftwf_plan) ctx->fftw_plan_fwd,
                      (fftwf_complex *) ctx->cpu_pup_cplx,
                      (fftwf_complex *) ctx->cpu_foc_cplx);

    /* 2. Parallel modulation loop with thread-private L2 buffers */
    const int32_t  npts  = ctx->eff_npts;
    const uint64_t nfft2 = ctx->npix_fft;
    const int      nth   = (npts > 1 && ctx->cpu_nth > 1)
                               ? ((npts < ctx->cpu_nth) ? npts : ctx->cpu_nth)
                               : 1;

    for (int t = 0; t < nth; t++)
    {
        __builtin_memset(&ctx->cpu_thr_int[(size_t) t * nfft2], 0, nfft2 * sizeof(float));
    }

    const float *__restrict foc = MILK_ASSUME_ALIGNED(ctx->cpu_foc_cplx);

#pragma omp parallel num_threads(nth) if (nth > 1)
    {
        int        tid  = (nth > 1) ? omp_get_thread_num() : 0;
        float     *work = &ctx->cpu_thr_cplx[(size_t) tid * nfft2 * 2u];
        float     *acc  = &ctx->cpu_thr_int[(size_t) tid * nfft2];
        fftwf_plan pinv = (fftwf_plan) ctx->fftw_plans_inv[tid];

#pragma omp for schedule(static)
        for (int32_t k = 0; k < npts; k++)
        {
            const float *__restrict mk = &ctx->masks_cplx[(size_t) k * nfft2 * 2u];

#pragma omp simd
            for (uint64_t i = 0; i < nfft2; i++)
            {
                float er          = foc[2u * i];
                float ei          = foc[2u * i + 1u];
                float mr          = mk[2u * i];
                float mi          = mk[2u * i + 1u];
                work[2u * i]      = er * mr - ei * mi;
                work[2u * i + 1u] = er * mi + ei * mr;
            }

            fftwf_execute_dft(pinv, (fftwf_complex *) work, (fftwf_complex *) work);

#pragma omp simd
            for (uint64_t i = 0; i < nfft2; i++)
            {
                float dr = work[2u * i];
                float di = work[2u * i + 1u];
                acc[i]  += dr * dr + di * di;
            }
        }
    }

    /* 3. Reduce thread accumulators and bin onto camera pixels */
    float *__restrict det = MILK_ASSUME_ALIGNED(ctx->cpu_det_int);
    __builtin_memcpy(det, ctx->cpu_thr_int, nfft2 * sizeof(float));
    for (int t = 1; t < nth; t++)
    {
        const float *__restrict src_t = &ctx->cpu_thr_int[(size_t) t * nfft2];
#pragma omp simd
        for (uint64_t i = 0; i < nfft2; i++)
        {
            det[i] += src_t[i];
        }
    }

    pyrwfs_bin_image_cpu(det, nfft, ctx->cfg.ncam, ctx->cpu_cam_int);
}

static void MILK_HOT pyrwfs_accum_cpu_common(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *ncpa_opd,
    float                  wf_to_opd)
{
    if (ctx->subframe_cnt == 1u)
    {
        __builtin_memcpy(ctx->cpu_accum_int, ctx->cpu_cam_int, ctx->npix_cam * sizeof(float));
    }

    pyrwfs_propagate_cpu(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, wf_to_opd);

    if (ctx->subframe_cnt >= 1u)
    {
        float *__restrict       acc = MILK_ASSUME_ALIGNED(ctx->cpu_accum_int);
        const float *__restrict cam = MILK_ASSUME_ALIGNED(ctx->cpu_cam_int);
#pragma omp simd
        for (uint64_t i = 0; i < ctx->npix_cam; i++)
        {
            acc[i] += cam[i];
        }
    }
}

int MILK_COLD cacaosim_pyrwfs_init(
    cacaosim_pyrwfs_ctx_t       *ctx,
    const cacaosim_pyrwfs_cfg_t *cfg,
    const float                 *pup_amp,
    const float                 *ncpa_opd,
    void                        *ext_stream)
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
    ctx->eff_npts = (cfg->mod_npts >= 1 && cfg->mod_rad_ld > 0.0f) ? cfg->mod_npts : 1;
    if (cfg->mod_npts >= 1 && cfg->mod_rad_ld <= 0.0f)
    {
        /* Keep buffer capacity if user later raises mod_rad live */
        ctx->eff_npts = cfg->mod_npts;
    }

    int max_th   = omp_get_max_threads();
    ctx->cpu_nth = (max_th > 0) ? max_th : 1;
    if (ctx->cpu_nth > 8)
    {
        ctx->cpu_nth = 8;
    }

    ctx->masks_cplx = (float *) cacaosim_aligned_calloc(
        (size_t) ctx->eff_npts * ctx->npix_fft * 2u, sizeof(float));
    ctx->cpu_pup_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
    ctx->cpu_foc_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
    ctx->cpu_thr_cplx = (float *) cacaosim_aligned_calloc(
        (size_t) ctx->cpu_nth * ctx->npix_fft * 2u, sizeof(float));
    ctx->cpu_thr_int = (float *) cacaosim_aligned_calloc(
        (size_t) ctx->cpu_nth * ctx->npix_fft, sizeof(float));
    ctx->cpu_det_int    = (float *) cacaosim_aligned_calloc(ctx->npix_fft, sizeof(float));
    ctx->cpu_cam_int    = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_accum_int  = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->fftw_plans_inv = (void **) calloc((size_t) ctx->cpu_nth, sizeof(void *));

    if (!ctx->masks_cplx || !ctx->cpu_pup_cplx || !ctx->cpu_foc_cplx || !ctx->cpu_thr_cplx ||
        !ctx->cpu_thr_int || !ctx->cpu_det_int || !ctx->cpu_cam_int || !ctx->cpu_accum_int ||
        !ctx->fftw_plans_inv ||
        cacaosim_det_pool_init(&ctx->det_pool, ctx->cpu_nth, cfg->rng_seed) != 0)
    {
        cacaosim_pyrwfs_free(ctx);
        return -1;
    }

    int n = (int) ctx->nfft;
    ctx->fftw_plan_fwd = (void *) fftwf_plan_dft_2d(
        n, n, (fftwf_complex *) ctx->cpu_pup_cplx, (fftwf_complex *) ctx->cpu_foc_cplx,
        FFTW_FORWARD, FFTW_ESTIMATE);
    if (ctx->fftw_plan_fwd == NULL)
    {
        cacaosim_pyrwfs_free(ctx);
        return -1;
    }

    for (int t = 0; t < ctx->cpu_nth; t++)
    {
        fftwf_complex *buf = (fftwf_complex *) &ctx->cpu_thr_cplx[(size_t) t * ctx->npix_fft * 2u];
        ctx->fftw_plans_inv[t] = (void *) fftwf_plan_dft_2d(
            n, n, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (ctx->fftw_plans_inv[t] == NULL)
        {
            cacaosim_pyrwfs_free(ctx);
            return -1;
        }
    }

    cacaosim_pyrwfs_update_masks(ctx, cfg->mod_rad_ld);

    /* Measure flat-wavefront total detector intensity so flux_ph equals total detected photons */
    pyrwfs_propagate_cpu(ctx, pup_amp, NULL, NULL, NULL, NULL, 1.0f);
    double sum_flat = 0.0;
    for (uint64_t i = 0; i < ctx->npix_cam; i++)
    {
        sum_flat += (double) ctx->cpu_cam_int[i];
    }
    ctx->flat_flux_norm = (sum_flat > 1e-20) ? (float) sum_flat : 1.0f;

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_pyrwfs_cuda_init(ctx, pup_amp, ncpa_opd, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

void MILK_COLD cacaosim_pyrwfs_free(cacaosim_pyrwfs_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
    }
    if (ctx->fftw_plan_fwd)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_plan_fwd);
    }
    if (ctx->fftw_plans_inv)
    {
        for (int t = 0; t < ctx->cpu_nth; t++)
        {
            if (ctx->fftw_plans_inv[t])
            {
                fftwf_destroy_plan((fftwf_plan) ctx->fftw_plans_inv[t]);
            }
        }
        free(ctx->fftw_plans_inv);
    }
    cacaosim_det_pool_free(&ctx->det_pool);
    free(ctx->masks_cplx);
    free(ctx->cpu_pup_cplx);
    free(ctx->cpu_foc_cplx);
    free(ctx->cpu_thr_cplx);
    free(ctx->cpu_thr_int);
    free(ctx->cpu_det_int);
    free(ctx->cpu_cam_int);
    free(ctx->cpu_accum_int);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

void MILK_HOT cacaosim_pyrwfs_accum_subframe(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *ncpa_opd,
    const float           *d_dm_opd,
    float                  turb_gain,
    float                  mod_rad)
{
    if (fabsf(mod_rad - ctx->cached_mod_rad) > 1e-5f)
    {
        cacaosim_pyrwfs_update_masks(ctx, mod_rad);
    }

    float        wf_to_opd = cacaosim_wf_to_opd_scale(ctx->cfg.wf_unit, ctx->cfg.ref_lambda,
                                                      turb_gain);
    const float *wf_eff    = (turb_gain != 0.0f) ? wf_in : NULL;

    if (ctx->use_cuda)
    {
        cacaosim_pyrwfs_cuda_accum(ctx, NULL, atm_amp, wf_eff, dm_opd, NULL, NULL, d_dm_opd,
                                   wf_to_opd);
    }
    else
    {
        pyrwfs_accum_cpu_common(ctx, pup_amp, atm_amp, wf_eff, dm_opd, ncpa_opd, wf_to_opd);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_pyrwfs_accum_beam(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *ncpa_opd,
    const float           *d_beam_opd,
    const float           *d_beam_amp,
    float                  mod_rad)
{
    if (fabsf(mod_rad - ctx->cached_mod_rad) > 1e-5f)
    {
        cacaosim_pyrwfs_update_masks(ctx, mod_rad);
    }

    if (ctx->use_cuda)
    {
        cacaosim_pyrwfs_cuda_accum(ctx, beam_amp, NULL, beam_opd, NULL, d_beam_amp, d_beam_opd,
                                   NULL, 1.0f);
    }
    else
    {
        pyrwfs_accum_cpu_common(ctx, beam_amp, NULL, beam_opd, NULL, ncpa_opd, 1.0f);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_pyrwfs_readout(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    if (ctx->subframe_cnt == 0u)
    {
        return;
    }

    const cacaosim_det_cfg_t *det = (det_live != NULL) ? det_live : &ctx->cfg.det;
    ctx->frame_seq++;

    if (ctx->use_cuda)
    {
        cacaosim_pyrwfs_cuda_readout(ctx, det, out_u16, out_f32);
        ctx->subframe_cnt = 0u;
        return;
    }

    if (ctx->subframe_cnt > 1u)
    {
        const float             inv_n = 1.0f / (float) ctx->subframe_cnt;
        const float *__restrict acc   = MILK_ASSUME_ALIGNED(ctx->cpu_accum_int);
        float *__restrict       cam   = MILK_ASSUME_ALIGNED(ctx->cpu_cam_int);
#pragma omp simd
        for (uint64_t i = 0; i < ctx->npix_cam; i++)
        {
            cam[i] = acc[i] * inv_n;
        }
    }

    cacaosim_detector_readout_cpu(&ctx->det_pool, det, ctx->cpu_cam_int, ctx->npix_cam,
                                  ctx->flat_flux_norm, out_u16, out_f32);
    ctx->subframe_cnt = 0u;
}

void MILK_HOT cacaosim_pyrwfs_step(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    float                     mod_rad,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    cacaosim_pyrwfs_accum_subframe(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, d_dm_opd,
                                   turb_gain, mod_rad);
    cacaosim_pyrwfs_readout(ctx, det_live, out_u16, out_f32);
}
