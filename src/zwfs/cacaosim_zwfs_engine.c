// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs_engine.c
 * @brief   Babinet Matrix-DFT Zernike phase-contrast WFS CPU engine and host orchestration
 */

#define _GNU_SOURCE
#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_cuda.h"
#include "zwfs/cacaosim_zwfs.h"

extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

static void MILK_HOT zwfs_bin_image_cpu(
    const float *__restrict src,
    uint32_t                nsrc,
    uint32_t                ncam,
    float *__restrict       dst)
{
    if (ncam == nsrc)
    {
        __builtin_memcpy(dst, src, (size_t) nsrc * nsrc * sizeof(float));
        return;
    }

    if (nsrc > ncam && (nsrc % ncam) == 0u)
    {
        const uint32_t bin = nsrc / ncam;
        for (uint32_t cy = 0; cy < ncam; cy++)
        {
            for (uint32_t cx = 0; cx < ncam; cx++)
            {
                float    acc = 0.0f;
                uint32_t y0  = cy * bin;
                uint32_t x0  = cx * bin;
                for (uint32_t by = 0; by < bin; by++)
                {
                    const float *row = &src[(size_t) (y0 + by) * nsrc + x0];
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

    const float scale = (float) nsrc / (float) ncam;
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
                if (iy < 0 || iy >= (int) nsrc)
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
                    if (ix < 0 || ix >= (int) nsrc)
                    {
                        continue;
                    }
                    float wx0 = (fx0 > (float) ix) ? fx0 : (float) ix;
                    float wx1 = (fx1 < (float) (ix + 1)) ? fx1 : (float) (ix + 1);
                    float wx  = wx1 - wx0;
                    if (wx > 0.0f)
                    {
                        acc += wx * wy * src[(size_t) iy * nsrc + (uint32_t) ix];
                    }
                }
            }
            dst[(size_t) cy * ncam + cx] = acc;
        }
    }
}

static void zwfs_precompute_babinet(cacaosim_zwfs_ctx_t *ctx)
{
    const uint32_t nwf   = ctx->cfg.nwf;
    const uint32_t m     = ctx->n_mft;
    const float    pdim  = (ctx->cfg.pup_diam > 0.05f) ? ctx->cfg.pup_diam : 0.9375f;
    const float    dpix  = pdim * (float) nwf;
    const float    p0    = 0.5f * (float) (nwf - 1u);
    const float    d_ld  = (ctx->cfg.dimple_ld > 0.05f) ? ctx->cfg.dimple_ld : 1.06f;
    const float    r_ld  = 0.5f * d_ld;
    const float    fov   = (1.4f * d_ld > 2.0f) ? (1.4f * d_ld) : 2.0f;
    const float    du    = fov / (float) m;
    const float    nrm   = sqrtf(du / dpix);

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

            size_t idx_kp             = ((size_t) k * nwf + p) * 2u;
            size_t idx_pk             = ((size_t) p * m + k) * 2u;
            ctx->fwd_U_cplx[idx_kp]   = nrm * co;
            ctx->fwd_U_cplx[idx_kp + 1] = nrm * s;
            ctx->fwd_V_cplx[idx_pk]   = nrm * co;
            ctx->fwd_V_cplx[idx_pk + 1] = nrm * s;

            ctx->inv_U_cplx[idx_pk]   = nrm * co;
            ctx->inv_U_cplx[idx_pk + 1] = -nrm * s;
            ctx->inv_V_cplx[idx_kp]   = nrm * co;
            ctx->inv_V_cplx[idx_kp + 1] = -nrm * s;
        }
    }

    float t_amp = (ctx->cfg.dimple_amp >= 0.0f && ctx->cfg.dimple_amp <= 1.0f)
                      ? ctx->cfg.dimple_amp
                      : 1.0f;
    float c_re  = t_amp * cosf(ctx->cfg.dimple_rad) - 1.0f;
    float c_im  = t_amp * sinf(ctx->cfg.dimple_rad);

    /* 8x8 subpixel area integration of circular focal dimple */
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
            float  w   = (float) hits * invs;
            size_t idx = ((size_t) ky * m + kx) * 2u;
            ctx->mask_bab_cplx[idx]      = w * c_re;
            ctx->mask_bab_cplx[idx + 1u] = w * c_im;
        }
    }
}

static void MILK_HOT zwfs_propagate_cpu(
    cacaosim_zwfs_ctx_t    *ctx,
    const float *__restrict pup_amp,
    const float *__restrict atm_amp,
    const float *__restrict wf_in,
    const float *__restrict dm_opd,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd)
{
    const uint32_t nwf        = ctx->cfg.nwf;
    const uint32_t m          = ctx->n_mft;
    const uint64_t npix_wf    = ctx->npix_wf;
    const float    lam        = (ctx->cfg.lambda_um > 0.1f) ? ctx->cfg.lambda_um : 0.75f;
    const float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    const int      nth        = (npix_wf >= 16384ULL) ? omp_get_max_threads() : 1;

    float *__restrict epup = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);

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
            epup[2u * p]      = 0.0f;
            epup[2u * p + 1u] = 0.0f;
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
        epup[2u * p]      = amp * co;
        epup[2u * p + 1u] = amp * s;
    }

    const float c_one[2]  = { 1.0f, 0.0f };
    const float c_zero[2] = { 0.0f, 0.0f };

    /* Forward MFT onto focal dimple patch: Efoc [m x m] = fwd_U * Epup * fwd_V */
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) nwf, (int) nwf, c_one,
                ctx->fwd_U_cplx, (int) nwf, epup, (int) nwf, c_zero, ctx->cpu_tmp_cplx, (int) nwf);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) m, (int) nwf, c_one,
                ctx->cpu_tmp_cplx, (int) nwf, ctx->fwd_V_cplx, (int) m, c_zero, ctx->cpu_efoc_cplx,
                (int) m);

    /* Apply Babinet complex mask (t_d * exp(i*theta) - 1) * M(u, v) */
    float *__restrict       ef = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);
    const float *__restrict mb = MILK_ASSUME_ALIGNED(ctx->mask_bab_cplx);
#pragma omp simd
    for (uint64_t i = 0; i < ctx->npix_mft; i++)
    {
        float er       = ef[2u * i];
        float ei       = ef[2u * i + 1u];
        float mr       = mb[2u * i];
        float mi       = mb[2u * i + 1u];
        ef[2u * i]     = er * mr - ei * mi;
        ef[2u * i + 1] = er * mi + ei * mr;
    }

    /* Inverse MFT + Babinet superposition: Epup += inv_U * Efoc * inv_V */
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) m, (int) m, c_one,
                ctx->inv_U_cplx, (int) m, ef, (int) m, c_zero, ctx->cpu_tmp_cplx, (int) m);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) nwf, (int) m, c_one,
                ctx->cpu_tmp_cplx, (int) m, ctx->inv_V_cplx, (int) nwf, c_one, epup, (int) nwf);

    float *__restrict pint = MILK_ASSUME_ALIGNED(ctx->cpu_pup_int);
#pragma omp simd
    for (uint64_t p = 0; p < npix_wf; p++)
    {
        float re = epup[2u * p];
        float im = epup[2u * p + 1u];
        pint[p]  = re * re + im * im;
    }

    zwfs_bin_image_cpu(pint, nwf, ctx->cfg.ncam, ctx->cpu_cam_int);
}

static void MILK_HOT zwfs_accum_cpu_common(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *pup_amp,
    const float         *atm_amp,
    const float         *wf_in,
    const float         *dm_opd,
    const float         *ncpa_opd,
    float                wf_to_opd)
{
    if (ctx->subframe_cnt == 1u)
    {
        __builtin_memcpy(ctx->cpu_accum_int, ctx->cpu_cam_int, ctx->npix_cam * sizeof(float));
    }

    zwfs_propagate_cpu(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, wf_to_opd);

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

int MILK_COLD cacaosim_zwfs_init(
    cacaosim_zwfs_ctx_t       *ctx,
    const cacaosim_zwfs_cfg_t *cfg,
    const float               *pup_amp,
    const float               *ncpa_opd,
    void                      *ext_stream)
{
    if (ctx == NULL || cfg == NULL || pup_amp == NULL || cfg->nwf < 8 || cfg->ncam < 4)
    {
        return -1;
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg      = *cfg;
    ctx->n_mft    = (cfg->n_mft >= 16u) ? cfg->n_mft : 32u;
    ctx->npix_wf  = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->npix_mft = (uint64_t) ctx->n_mft * ctx->n_mft;
    ctx->npix_cam = (uint64_t) cfg->ncam * cfg->ncam;

    if (openblas_set_num_threads != NULL)
    {
        openblas_set_num_threads(1);
    }

    size_t uv_elems    = (size_t) ctx->n_mft * cfg->nwf * 2u;
    ctx->fwd_U_cplx    = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
    ctx->fwd_V_cplx    = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
    ctx->inv_U_cplx    = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
    ctx->inv_V_cplx    = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
    ctx->mask_bab_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_mft * 2u, sizeof(float));
    ctx->cpu_epup_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
    ctx->cpu_tmp_cplx  = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
    ctx->cpu_efoc_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_mft * 2u, sizeof(float));
    ctx->cpu_pup_int   = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    ctx->cpu_cam_int   = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_accum_int = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));

    if (!ctx->fwd_U_cplx || !ctx->fwd_V_cplx || !ctx->inv_U_cplx || !ctx->inv_V_cplx ||
        !ctx->mask_bab_cplx || !ctx->cpu_epup_cplx || !ctx->cpu_tmp_cplx || !ctx->cpu_efoc_cplx ||
        !ctx->cpu_pup_int || !ctx->cpu_cam_int || !ctx->cpu_accum_int ||
        cacaosim_det_pool_init(&ctx->det_pool, 0, cfg->rng_seed) != 0)
    {
        cacaosim_zwfs_free(ctx);
        return -1;
    }

    zwfs_precompute_babinet(ctx);

    /* Normalize flux against open-pupil Parseval intensity so dimple absorption is preserved */
    double sum_open = 0.0;
    for (uint64_t p = 0; p < ctx->npix_wf; p++)
    {
        double a = (double) pup_amp[p];
        sum_open += a * a;
    }
    ctx->flat_flux_norm = (sum_open > 1e-20) ? (float) sum_open : 1.0f;

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_zwfs_cuda_init(ctx, pup_amp, ncpa_opd, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

void MILK_COLD cacaosim_zwfs_free(cacaosim_zwfs_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_zwfs_cuda_free(ctx);
    }
    cacaosim_det_pool_free(&ctx->det_pool);
    free(ctx->fwd_U_cplx);
    free(ctx->fwd_V_cplx);
    free(ctx->inv_U_cplx);
    free(ctx->inv_V_cplx);
    free(ctx->mask_bab_cplx);
    free(ctx->cpu_epup_cplx);
    free(ctx->cpu_tmp_cplx);
    free(ctx->cpu_efoc_cplx);
    free(ctx->cpu_pup_int);
    free(ctx->cpu_cam_int);
    free(ctx->cpu_accum_int);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

void MILK_HOT cacaosim_zwfs_accum_subframe(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *pup_amp,
    const float         *atm_amp,
    const float         *wf_in,
    const float         *dm_opd,
    const float         *ncpa_opd,
    const float         *d_dm_opd,
    float                turb_gain)
{
    float        wf_to_opd = cacaosim_wf_to_opd_scale(ctx->cfg.wf_unit, ctx->cfg.ref_lambda,
                                                      turb_gain);
    const float *wf_eff    = (turb_gain != 0.0f) ? wf_in : NULL;

    if (ctx->use_cuda)
    {
        cacaosim_zwfs_cuda_accum(ctx, NULL, atm_amp, wf_eff, dm_opd, NULL, NULL, d_dm_opd,
                                 wf_to_opd);
    }
    else
    {
        zwfs_accum_cpu_common(ctx, pup_amp, atm_amp, wf_eff, dm_opd, ncpa_opd, wf_to_opd);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_zwfs_accum_beam(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *beam_opd,
    const float         *beam_amp,
    const float         *ncpa_opd,
    const float         *d_beam_opd,
    const float         *d_beam_amp)
{
    if (ctx->use_cuda)
    {
        cacaosim_zwfs_cuda_accum(ctx, beam_amp, NULL, beam_opd, NULL, d_beam_amp, d_beam_opd,
                                 NULL, 1.0f);
    }
    else
    {
        zwfs_accum_cpu_common(ctx, beam_amp, NULL, beam_opd, NULL, ncpa_opd, 1.0f);
    }
    ctx->subframe_cnt++;
}

void MILK_HOT cacaosim_zwfs_readout(
    cacaosim_zwfs_ctx_t      *ctx,
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
        cacaosim_zwfs_cuda_readout(ctx, det, out_u16, out_f32);
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

void MILK_HOT cacaosim_zwfs_step(
    cacaosim_zwfs_ctx_t      *ctx,
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
    cacaosim_zwfs_accum_subframe(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, d_dm_opd,
                                 turb_gain);
    cacaosim_zwfs_readout(ctx, det_live, out_u16, out_f32);
}
