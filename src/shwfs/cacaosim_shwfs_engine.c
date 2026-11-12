// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs_engine.c
 * @brief   OpenMP L1-cached FFTW3f Diffractive Shack-Hartmann WFS engine
 */

#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_cuda.h"
#include "shwfs/cacaosim_shwfs.h"

static void MILK_HOT shwfs_propagate_cpu(
    cacaosim_shwfs_ctx_t   *ctx,
    const float *__restrict pup_amp,
    const float *__restrict atm_amp,
    const float *__restrict wf_in,
    const float *__restrict dm_opd,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd)
{
    const uint32_t nwf        = ctx->cfg.nwf;
    const uint32_t n_sub      = ctx->cfg.n_sub;
    const uint32_t psub       = ctx->cfg.pix_per_sub;
    const uint32_t ncam       = ctx->ncam;
    const uint32_t nfft       = ctx->nfft_sub;
    const uint64_t npix_sfft  = ctx->npix_sub_fft;
    const int32_t  half_p     = (int32_t) (psub / 2u);
    const float    lam        = (ctx->cfg.lambda_um > 0.1f) ? ctx->cfg.lambda_um : 0.75f;
    const float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    const float    inv_nfft2  = 1.0f / (float) npix_sfft;
    const int      nth        = (ctx->nbatch >= 16u && ctx->cpu_nth > 1) ? ctx->cpu_nth : 1;

    const float *__restrict sh1d = ctx->shift_1d_cplx;

#pragma omp parallel num_threads(nth) if (nth > 1)
    {
        int        tid  = (nth > 1) ? omp_get_thread_num() : 0;
        float     *work = &ctx->cpu_thr_cplx[(size_t) tid * npix_sfft * 2u];
        fftwf_plan plan = (fftwf_plan) ctx->fftw_plans[tid];

#pragma omp for schedule(static)
        for (uint32_t b = 0; b < ctx->nbatch; b++)
        {
            uint32_t sx = b % n_sub;
            uint32_t sy = b / n_sub;
            uint32_t x0 = ctx->sub_off[sx];
            uint32_t y0 = ctx->sub_off[sy];
            uint32_t nx = ctx->sub_len[sx];
            uint32_t ny = ctx->sub_len[sy];

            __builtin_memset(work, 0, (size_t) npix_sfft * 2u * sizeof(float));

            int has_flux = 0;
            for (uint32_t ly = 0; ly < ny; ly++)
            {
                float    sy_re = sh1d[2u * ly];
                float    sy_im = sh1d[2u * ly + 1u];
                uint64_t p_row = (uint64_t) (y0 + ly) * nwf + x0;
                float   *w_row = &work[(size_t) ly * nfft * 2u];

                for (uint32_t lx = 0; lx < nx; lx++)
                {
                    uint64_t p   = p_row + lx;
                    float    amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
                    if (atm_amp != NULL)
                    {
                        amp *= atm_amp[p];
                    }
                    if (amp <= 0.0f)
                    {
                        continue;
                    }
                    has_flux = 1;

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

                    float phi   = opd * opd_to_rad;
                    float pr    = amp * cosf(phi);
                    float pi    = amp * sinf(phi);
                    float sx_re = sh1d[2u * lx];
                    float sx_im = sh1d[2u * lx + 1u];

                    /* Multiply by separable 2D half-bin centering phasor s_x * s_y */
                    float sr            = sx_re * sy_re - sx_im * sy_im;
                    float si            = sx_re * sy_im + sx_im * sy_re;
                    w_row[2u * lx]      = pr * sr - pi * si;
                    w_row[2u * lx + 1u] = pr * si + pi * sr;
                }
            }

            uint32_t cam_x0 = sx * psub;
            uint32_t cam_y0 = sy * psub;

            if (!has_flux)
            {
                for (uint32_t cy = 0; cy < psub; cy++)
                {
                    __builtin_memset(&ctx->cpu_cam_int[(size_t) (cam_y0 + cy) * ncam + cam_x0], 0,
                                     (size_t) psub * sizeof(float));
                }
                continue;
            }

            fftwf_execute_dft(plan, (fftwf_complex *) work, (fftwf_complex *) work);

            for (uint32_t cy = 0; cy < psub; cy++)
            {
                int32_t  dy  = (int32_t) cy - half_p;
                uint32_t ky  = (uint32_t) ((dy % (int32_t) nfft + (int32_t) nfft) % (int32_t) nfft);
                float   *dst = &ctx->cpu_cam_int[(size_t) (cam_y0 + cy) * ncam + cam_x0];
                for (uint32_t cx = 0; cx < psub; cx++)
                {
                    int32_t  dx = (int32_t) cx - half_p;
                    uint32_t kx = (uint32_t) ((dx % (int32_t) nfft + (int32_t) nfft) %
                                              (int32_t) nfft);
                    size_t   k  = ((size_t) ky * nfft + kx) * 2u;
                    float    re = work[k];
                    float    im = work[k + 1u];
                    dst[cx]     = inv_nfft2 * (re * re + im * im);
                }
            }
        }
    }
}

static void MILK_HOT shwfs_accum_cpu_common(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    float                 wf_to_opd)
{
    if (ctx->subframe_cnt == 1u)
    {
        __builtin_memcpy(ctx->cpu_accum_int, ctx->cpu_cam_int, ctx->npix_cam * sizeof(float));
    }

    shwfs_propagate_cpu(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, wf_to_opd);

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

int MILK_COLD cacaosim_shwfs_init(
    cacaosim_shwfs_ctx_t       *ctx,
    const cacaosim_shwfs_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream)
{
    if (ctx == NULL || cfg == NULL || pup_amp == NULL || cfg->nwf < 8 || cfg->n_sub < 1 ||
        cfg->pix_per_sub < 2)
    {
        return -1;
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg     = *cfg;
    ctx->ncam    = cfg->n_sub * cfg->pix_per_sub;
    ctx->nbatch  = cfg->n_sub * cfg->n_sub;
    ctx->npix_wf = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->npix_cam = (uint64_t) ctx->ncam * ctx->ncam;

    ctx->sub_off = (uint32_t *) calloc(cfg->n_sub, sizeof(uint32_t));
    ctx->sub_len = (uint32_t *) calloc(cfg->n_sub, sizeof(uint32_t));
    if (!ctx->sub_off || !ctx->sub_len)
    {
        cacaosim_shwfs_free(ctx);
        return -1;
    }

    uint32_t w_max = 1u;
    for (uint32_t s = 0; s < cfg->n_sub; s++)
    {
        uint32_t x0      = (s * cfg->nwf) / cfg->n_sub;
        uint32_t x1      = ((s + 1u) * cfg->nwf) / cfg->n_sub;
        uint32_t len     = (x1 > x0) ? (x1 - x0) : 1u;
        ctx->sub_off[s]  = x0;
        ctx->sub_len[s]  = len;
        if (len > w_max)
        {
            w_max = len;
        }
    }
    ctx->w_sub_max = w_max;

    uint32_t pad  = (cfg->pad_factor >= 2u) ? cfg->pad_factor : 2u;
    uint32_t nfft = w_max * pad;
    if (nfft < cfg->pix_per_sub)
    {
        nfft = cfg->pix_per_sub;
    }
    if ((nfft & 1u) != 0u)
    {
        nfft++;
    }
    ctx->nfft_sub     = nfft;
    ctx->npix_sub_fft = (uint64_t) nfft * nfft;

    ctx->shift_1d_cplx = (float *) cacaosim_aligned_calloc((size_t) nfft * 2u, sizeof(float));
    if (!ctx->shift_1d_cplx)
    {
        cacaosim_shwfs_free(ctx);
        return -1;
    }

    /* Half-bin shift aligns zero-tilt spot symmetrically at (pix_per_sub - 1) / 2.0 */
    float half_bin = ((cfg->pix_per_sub & 1u) == 0u) ? 0.5f : 0.0f;
    for (uint32_t m = 0; m < nfft; m++)
    {
        float ang                      = -CACAOSIM_TWO_PI_F * half_bin * (float) m / (float) nfft;
        ctx->shift_1d_cplx[2u * m]     = cosf(ang);
        ctx->shift_1d_cplx[2u * m + 1] = sinf(ang);
    }

    int max_th   = omp_get_max_threads();
    ctx->cpu_nth = (max_th > 0) ? max_th : 1;
    if (ctx->cpu_nth > 16)
    {
        ctx->cpu_nth = 16;
    }

    ctx->cpu_thr_cplx = (float *) cacaosim_aligned_calloc(
        (size_t) ctx->cpu_nth * ctx->npix_sub_fft * 2u, sizeof(float));
    ctx->cpu_cam_int   = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_accum_int = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->fftw_plans    = (void **) calloc((size_t) ctx->cpu_nth, sizeof(void *));

    if (!ctx->cpu_thr_cplx || !ctx->cpu_cam_int || !ctx->cpu_accum_int || !ctx->fftw_plans ||
        cacaosim_det_pool_init(&ctx->det_pool, ctx->cpu_nth, cfg->rng_seed) != 0)
    {
        cacaosim_shwfs_free(ctx);
        return -1;
    }

    int nf = (int) nfft;
    for (int t = 0; t < ctx->cpu_nth; t++)
    {
        fftwf_complex *buf =
            (fftwf_complex *) &ctx->cpu_thr_cplx[(size_t) t * ctx->npix_sub_fft * 2u];
        ctx->fftw_plans[t] =
            (void *) fftwf_plan_dft_2d(nf, nf, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
        if (ctx->fftw_plans[t] == NULL)
        {
            cacaosim_shwfs_free(ctx);
            return -1;
        }
    }

    /* Calibrate flat-wavefront flux normalization */
    shwfs_propagate_cpu(ctx, pup_amp, NULL, NULL, NULL, NULL, 1.0f);
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
        if (cacaosim_shwfs_cuda_init(ctx, pup_amp, ncpa_opd, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

void MILK_COLD cacaosim_shwfs_free(cacaosim_shwfs_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_shwfs_cuda_free(ctx);
    }
    if (ctx->fftw_plans)
    {
        for (int t = 0; t < ctx->cpu_nth; t++)
        {
            if (ctx->fftw_plans[t])
            {
                fftwf_destroy_plan((fftwf_plan) ctx->fftw_plans[t]);
            }
        }
        free(ctx->fftw_plans);
    }
    cacaosim_det_pool_free(&ctx->det_pool);
    free(ctx->sub_off);
    free(ctx->sub_len);
    free(ctx->shift_1d_cplx);
    free(ctx->cpu_thr_cplx);
    free(ctx->cpu_cam_int);
    free(ctx->cpu_accum_int);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

void MILK_HOT cacaosim_shwfs_accum_subframe(
    cacaosim_shwfs_ctx_t *ctx,
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
        cacaosim_shwfs_cuda_accum(ctx, NULL, atm_amp, wf_eff, dm_opd, NULL, NULL, d_dm_opd,
                                  wf_to_opd);
    }
    else
    {
        shwfs_accum_cpu_common(ctx, pup_amp, atm_amp, wf_eff, dm_opd, ncpa_opd, wf_to_opd);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_shwfs_accum_beam(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    if (ctx->use_cuda)
    {
        cacaosim_shwfs_cuda_accum(ctx, beam_amp, NULL, beam_opd, NULL, d_beam_amp, d_beam_opd,
                                  NULL, 1.0f);
    }
    else
    {
        shwfs_accum_cpu_common(ctx, beam_amp, NULL, beam_opd, NULL, ncpa_opd, 1.0f);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_shwfs_readout(
    cacaosim_shwfs_ctx_t     *ctx,
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
        cacaosim_shwfs_cuda_readout(ctx, det, out_u16, out_f32);
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

void MILK_HOT cacaosim_shwfs_step(
    cacaosim_shwfs_ctx_t     *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    cacaosim_shwfs_accum_subframe(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, d_dm_opd,
                                  turb_gain);
    cacaosim_shwfs_readout(ctx, det_live, out_u16, out_f32);
}
