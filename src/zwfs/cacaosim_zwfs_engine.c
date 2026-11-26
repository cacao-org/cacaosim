// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs_engine.c
 * @brief   Babinet Matrix-DFT Zernike phase-contrast WFS CPU engine and host orchestration
 *
 * Architectural Role in cacaosim:
 *   Implements precomputation of separable Soummer Matrix-DFT (`MFT`) operators and subpixel-
 *   anti-aliased Babinet focal dimple masks, OpenBLAS `cblas_cgemm` CPU optical propagation,
 *   area-conserving detector binning, multi-subframe (`navg`) exposure accumulation, and backend
 *   dispatch to `cacaosim_zwfs_cuda.cu`.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Babinet Decomposition & Fine-Sampled Matrix-DFT:
 *      Discretizing Zernike's `~1.06 lambda/D` phase-shifting dimple on a direct 2D FFT grid
 *      suffers severe staircase aliasing unless padded to thousands of pixels. By Babinet's
 *      principle, writing the focal mask transmission as `1 + (t_d * exp(i*theta) - 1) * D(u, v)`
 *      (equivalently `1 - (1 - t_d * exp(i*theta)) * D(u, v)`) restricts the Fourier integral
 *      exclusively to the compact `~2 lambda/D` dimple patch:
 *        `Psi_out = Psi_pup + MFT^-1[(t_d * exp(i*theta) - 1) * D * MFT[Psi_pup]]`
 *      Pre-multiplying `(t_d * exp(i*theta) - 1)` into `mask_bab_cplx` lets the final inverse MFT
 *      `cblas_cgemm` superpose the dimple-diffracted reference wave onto `epup` in-place via
 *      `beta = +1` (`Epup := 1 * (inv_U * Efoc) * inv_V + 1 * Epup`).
 *   2. Cache-Hot Micro-GEMMs & Thread Control:
 *      For `nwf = 128` and `n_mft = 32`, each MFT matrix occupies only `32 KiB`, fitting entirely
 *      inside CPU L1/L2 cache so four `cblas_cgemm` calls finish in microseconds. Pinning OpenBLAS
 *      to 1 thread via weak symbol `openblas_set_num_threads(1)` avoids BLAS thread-pool spin-wait
 *      contention inside real-time multi-process AO loops.
 */

#define _GNU_SOURCE
#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "zwfs/cacaosim_zwfs.h"

/*
 * Optional weak symbol for OpenBLAS thread-count control: resolves to NULL when linked against
 * Netlib BLAS, Intel MKL, or BLIS without requiring build-time BLAS vendor conditionals.
 */
extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

/**
 * zwfs_bin_image_cpu() - Flux-conserving 2D rebinning from pupil grid onto detector pixels
 * @src:  Source 2D pupil-plane intensity map `[nsrc x nsrc]` (`|Psi_out|^2`)
 * @nsrc: Linear dimension of the source pupil grid [pixels]
 * @ncam: Linear dimension of the target WFS camera grid [pixels]
 * @dst:  Output 2D detector intensity map `[ncam x ncam]`
 *
 * Role:
 *   Hot-loop CPU helper that resamples the `nsrc x nsrc` Zernike output pupil intensity onto the
 *   `ncam x ncam` detector pixel array while strictly conserving total integrated optical flux.
 *
 * Rationale:
 *   Decouples optical wavefront sampling (`nwf`, chosen to resolve fine spiders, segments, and
 *   high-order aberrations) from physical camera geometry (`ncam`). Specializes three regimes:
 *   1. Direct `memcpy` fast path when `ncam == nsrc` (1:1 pixel mapping).
 *   2. Exact integer `bin x bin` superpixel box-summation when `nsrc` is an integer multiple of
 *      `ncam` (e.g., `128 -> 64`), avoiding floating-point overlap bounds inside the inner loop.
 *   3. Exact 2D fractional pixel-overlap area integration (`wx * wy`) for arbitrary non-integer
 *      pupil-to-camera magnification ratios or upsampling (`ncam > nsrc`).
 */
static void MILK_HOT zwfs_bin_image_cpu(
    const float *__restrict src,
    uint32_t                nsrc,
    uint32_t                ncam,
    float *__restrict       dst)
{
    /* Fast path 1: 1:1 pupil-to-detector pixel match */
    if (ncam == nsrc)
    {
        __builtin_memcpy(dst, src, (size_t) nsrc * nsrc * sizeof(float));
        return;
    }

    /* Fast path 2: exact integer superpixel box binning (e.g. 128 -> 64 with bin = 2) */
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

    /* General path: exact 2D geometric overlap area integration across fractional source pixels */
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

/**
 * zwfs_precompute_babinet() - Build separable Soummer MFT matrices and anti-aliased Babinet mask
 * @ctx: ZWFS engine context with allocated MFT and mask buffers
 *
 * Role:
 *   Cold-path setup routine called once by `cacaosim_zwfs_init()` to precompute the forward and
 *   inverse 1D Fourier projection matrices (`fwd_U_cplx`, `fwd_V_cplx`, `inv_U_cplx`,
 *   `inv_V_cplx`) and the 2D complex Babinet focal-plane dimple mask (`mask_bab_cplx`).
 *
 * Rationale:
 *   Sampling Zernike's `~1.06 lambda/D` phase-shifting dimple accurately on a conventional FFT
 *   grid would demand massive `20x..32x` zero-padding (`~4096 x 4096` 2D FFTs). Decomposing the
 *   focal transmission via Babinet's principle as `T(u, v) = 1 + (t_d * exp(i*theta) - 1) * D(u,v)`
 *   confines the Fourier integral to a compact square focal window `fov = max(1.4 * dimple_ld, 2)`
 *   [lambda/D] sampled on `m x m` (`n_mft x n_mft`, default `32 x 32`) fine grid points:
 *   - Symmetric factor `nrm = sqrt(du / dpix)` split across forward and inverse MFT kernels
 *     enforces unitary Parseval energy conservation between pupil pixels and `[lambda/D]` units.
 *   - `8 x 8` (`64`-point) subpixel supersampling of circular dimple boundary `u^2 + v^2 <= r^2`
 *     eliminates staircase discretization errors in dimple diameter sensitivity.
 *   - Pre-multiplying the complex Babinet scalar `(t_amp * exp(i * dimple_rad) - 1)` into
 *     `mask_bab_cplx` eliminates an arithmetic subtraction stage at runtime so the final inverse
 *     MFT GEMM accumulates directly onto `epup` with `beta = +1`.
 */
static void zwfs_precompute_babinet(cacaosim_zwfs_ctx_t *ctx)
{
    const uint32_t nwf      = ctx->cfg.nwf;
    const uint32_t m        = ctx->n_mft;
    const int32_t  nlam     = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    const size_t   uv_elems = (size_t) m * nwf * 2u;
    const float    lam0     = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 0.75f;
    const float    pdim     = (ctx->cfg.pup_diam > 0.05f) ? ctx->cfg.pup_diam : 0.9375f;
    const float    dpix     = pdim * (float) nwf;
    const float    p0       = 0.5f * (float) (nwf - 1u);
    const float    d_ld     = (ctx->cfg.dimple_ld > 0.05f) ? ctx->cfg.dimple_ld : 1.06f;
    const float    r_ld     = 0.5f * d_ld;
    const float    fov0     = (1.4f * d_ld > 2.0f) ? (1.4f * d_ld) : 2.0f;
    const float    du0      = fov0 / (float) m;
    const float    t_amp    = (ctx->cfg.dimple_amp >= 0.0f && ctx->cfg.dimple_amp <= 1.0f)
                                  ? ctx->cfg.dimple_amp
                                  : 1.0f;

    float bab_re[CACAOSIM_MAX_LAMBDA];
    float bab_im[CACAOSIM_MAX_LAMBDA];

    for (int32_t c = 0; c < nlam; c++)
    {
        const float scale = lam0 / ctx->lam_um[c];
        const float fov_c = fov0 * scale;
        const float du_c  = fov_c / (float) m;
        const float nrm_c = sqrtf(du_c / dpix);

        float *fU = &ctx->fwd_U_cplx[(size_t) c * uv_elems];
        float *fV = &ctx->fwd_V_cplx[(size_t) c * uv_elems];
        float *iU = &ctx->inv_U_cplx[(size_t) c * uv_elems];
        float *iV = &ctx->inv_V_cplx[(size_t) c * uv_elems];

        for (uint32_t k = 0; k < m; k++)
        {
            const float u = (((float) k + 0.5f) / (float) m - 0.5f) * fov_c;
            for (uint32_t p = 0; p < nwf; p++)
            {
                const float xt  = ((float) p - p0) / dpix;
                const float ang = -CACAOSIM_TWO_PI_F * xt * u;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);

                size_t idx_kp  = ((size_t) k * nwf + p) * 2u;
                size_t idx_pk  = ((size_t) p * m + k) * 2u;
                fU[idx_kp]     = nrm_c * co;
                fU[idx_kp + 1] = nrm_c * s;
                fV[idx_pk]     = nrm_c * co;
                fV[idx_pk + 1] = nrm_c * s;

                iU[idx_pk]     = nrm_c * co;
                iU[idx_pk + 1] = -nrm_c * s;
                iV[idx_kp]     = nrm_c * co;
                iV[idx_kp + 1] = -nrm_c * s;
            }
        }

        float theta_c = ctx->cfg.achrom_dimple ? ctx->cfg.dimple_rad
                                               : (ctx->cfg.dimple_rad * scale);
        bab_re[c]     = t_amp * cosf(theta_c) - 1.0f;
        bab_im[c]     = t_amp * sinf(theta_c);
    }

    /* 8x8 subpixel area integration of physical focal dimple (invariant in normalized MFT box) */
    const int   nsub = 8;
    const float invs = 1.0f / (float) (nsub * nsub);
    const float r2   = r_ld * r_ld;

    for (uint32_t ky = 0; ky < m; ky++)
    {
        const float vc = (((float) ky + 0.5f) / (float) m - 0.5f) * fov0;
        for (uint32_t kx = 0; kx < m; kx++)
        {
            const float uc   = (((float) kx + 0.5f) / (float) m - 0.5f) * fov0;
            int         hits = 0;
            for (int sy = 0; sy < nsub; sy++)
            {
                float vs = vc + (((float) sy + 0.5f) / (float) nsub - 0.5f) * du0;
                for (int sx = 0; sx < nsub; sx++)
                {
                    float us = uc + (((float) sx + 0.5f) / (float) nsub - 0.5f) * du0;
                    if (us * us + vs * vs <= r2)
                    {
                        hits++;
                    }
                }
            }
            float  w   = (float) hits * invs;
            size_t pix = (size_t) ky * m + kx;
            for (int32_t c = 0; c < nlam; c++)
            {
                size_t idx                   = ((size_t) c * ctx->npix_mft + pix) * 2u;
                ctx->mask_bab_cplx[idx]      = w * bab_re[c];
                ctx->mask_bab_cplx[idx + 1u] = w * bab_im[c];
            }
        }
    }
}

static void MILK_HOT zwfs_eval_babinet_chan_cpu(
    cacaosim_zwfs_ctx_t *ctx,
    int32_t              c)
{
    const uint32_t nwf      = ctx->cfg.nwf;
    const uint32_t m        = ctx->n_mft;
    const uint64_t npix_wf  = ctx->npix_wf;
    const size_t   uv_elems = (size_t) m * nwf * 2u;
    const float    wc       = ctx->chan_w[c];
    const float    c_one[2] = { 1.0f, 0.0f };
    const float    c_zero[2] = { 0.0f, 0.0f };

    float *__restrict       epup = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);
    float *__restrict       ef   = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);
    const float *__restrict fU   = &ctx->fwd_U_cplx[(size_t) c * uv_elems];
    const float *__restrict fV   = &ctx->fwd_V_cplx[(size_t) c * uv_elems];
    const float *__restrict iU   = &ctx->inv_U_cplx[(size_t) c * uv_elems];
    const float *__restrict iV   = &ctx->inv_V_cplx[(size_t) c * uv_elems];
    const float *__restrict mb   = &ctx->mask_bab_cplx[(size_t) c * ctx->npix_mft * 2u];

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) nwf, (int) nwf, c_one,
                fU, (int) nwf, epup, (int) nwf, c_zero, ctx->cpu_tmp_cplx, (int) nwf);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) m, (int) m, (int) nwf, c_one,
                ctx->cpu_tmp_cplx, (int) nwf, fV, (int) m, c_zero, ef, (int) m);

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

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) m, (int) m, c_one,
                iU, (int) m, ef, (int) m, c_zero, ctx->cpu_tmp_cplx, (int) m);
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) nwf, (int) m, c_one,
                ctx->cpu_tmp_cplx, (int) m, iV, (int) nwf, c_one, epup, (int) nwf);

    float *__restrict pint = MILK_ASSUME_ALIGNED(ctx->cpu_pup_int);
    if (c == 0)
    {
#pragma omp simd
        for (uint64_t p = 0; p < npix_wf; p++)
        {
            float re = epup[2u * p];
            float im = epup[2u * p + 1u];
            pint[p]  = wc * (re * re + im * im);
        }
    }
    else
    {
#pragma omp simd
        for (uint64_t p = 0; p < npix_wf; p++)
        {
            float re = epup[2u * p];
            float im = epup[2u * p + 1u];
            pint[p] += wc * (re * re + im * im);
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
    const uint32_t nwf     = ctx->cfg.nwf;
    const uint64_t npix_wf = ctx->npix_wf;
    const int32_t  nlam    = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    const int      nth     = (npix_wf >= 16384ULL) ? omp_get_max_threads() : 1;
    const float    p0_xy   = 0.5f * (float) (nwf - 1u);

    float *__restrict epup = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);

    for (int32_t c = 0; c < nlam; c++)
    {
        const float opd_to_rad = ctx->k_rad[c];
        const float sx_c       = ctx->disp_sx[c];
        const float sy_c       = ctx->disp_sy[c];
        const int   has_disp   = (sx_c != 0.0f || sy_c != 0.0f) ? 1 : 0;

#pragma omp parallel for num_threads(nth) if (nth > 1) schedule(static)
        for (uint32_t iy = 0; iy < nwf; iy++)
        {
            const float    dy    = has_disp ? (sy_c * ((float) iy - p0_xy)) : 0.0f;
            const uint64_t pbase = (uint64_t) iy * nwf;
            for (uint32_t ix = 0; ix < nwf; ix++)
            {
                uint64_t p   = pbase + ix;
                float    amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
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
                float opd = has_disp ? (sx_c * ((float) ix - p0_xy) + dy) : 0.0f;
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
        }

        zwfs_eval_babinet_chan_cpu(ctx, c);
    }

    zwfs_bin_image_cpu(ctx->cpu_pup_int, nwf, ctx->cfg.ncam, ctx->cpu_cam_int);
}

static void MILK_HOT zwfs_propagate_poly_cpu(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *beam_opd,
    const float         *beam_amp,
    const float         *beam_lam_um,
    uint32_t             beam_nlam,
    uint64_t             chan_stride,
    const float         *ncpa_opd)
{
    const uint32_t nwf       = ctx->cfg.nwf;
    const uint64_t npix_wf   = ctx->npix_wf;
    const int32_t  nlam      = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    const int      nth       = (npix_wf >= 16384ULL) ? omp_get_max_threads() : 1;
    const float    lam0      = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 0.75f;
    const float    bw        = (ctx->cfg.bw_frac > 0.0f) ? ctx->cfg.bw_frac : 0.20f;
    const float    def_l0    = lam0 * (1.0f - 0.5f * bw);
    const float    def_l1    = lam0 * (1.0f + 0.5f * bw);
    const float    p0_xy     = 0.5f * (float) (nwf - 1u);
    const int32_t  unwrapped = (ctx->in_poly_mode == CACAOSIM_POLY_UNWRAPPED_OPD) ? 1 : 0;

    float *__restrict epup = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);

    for (int32_t c = 0; c < nlam; c++)
    {
        uint32_t m0 = 0u;
        uint32_t m1 = 0u;
        float    w1 = 0.0f;
        cacaosim_poly_interp_bracket_ex(ctx->lam_um[c], beam_lam_um, beam_nlam, def_l0, def_l1,
                                        unwrapped, &m0, &m1, &w1);
        float        w0       = 1.0f - w1;
        float        kc       = ctx->k_rad[c];
        float        sx_c     = ctx->disp_sx[c];
        float        sy_c     = ctx->disp_sy[c];
        int          has_disp = (sx_c != 0.0f || sy_c != 0.0f) ? 1 : 0;
        const float *o0       = (beam_opd != NULL) ? (beam_opd + (uint64_t) m0 * chan_stride) : NULL;
        const float *a0       = (beam_amp != NULL) ? (beam_amp + (uint64_t) m0 * chan_stride) : NULL;
        const float *o1       = (beam_opd != NULL) ? (beam_opd + (uint64_t) m1 * chan_stride) : NULL;
        const float *a1       = (beam_amp != NULL) ? (beam_amp + (uint64_t) m1 * chan_stride) : NULL;

#pragma omp parallel for num_threads(nth) if (nth > 1) schedule(static)
        for (uint32_t iy = 0; iy < nwf; iy++)
        {
            const float    dy    = has_disp ? (sy_c * ((float) iy - p0_xy)) : 0.0f;
            const uint64_t pbase = (uint64_t) iy * nwf;
            for (uint32_t ix = 0; ix < nwf; ix++)
            {
                uint64_t p    = pbase + ix;
                float    amp0 = (a0 != NULL) ? a0[p] : 1.0f;
                float    amp1 = (a1 != NULL) ? a1[p] : 1.0f;
                if (amp0 <= 0.0f && amp1 <= 0.0f)
                {
                    epup[2u * p]      = 0.0f;
                    epup[2u * p + 1u] = 0.0f;
                    continue;
                }
                float disp_opd = has_disp ? (sx_c * ((float) ix - p0_xy) + dy) : 0.0f;
                float nc       = ((ncpa_opd != NULL) ? ncpa_opd[p] : 0.0f) + disp_opd;
                if (unwrapped && m1 != m0 && w1 > 0.0f)
                {
                    float amp = w0 * amp0 + w1 * amp1;
                    float opd = w0 * ((o0 != NULL) ? o0[p] : 0.0f) +
                                w1 * ((o1 != NULL) ? o1[p] : 0.0f) + nc;
                    float s  = 0.0f;
                    float co = 1.0f;
                    sincosf(opd * kc, &s, &co);
                    epup[2u * p]      = amp * co;
                    epup[2u * p + 1u] = amp * s;
                }
                else
                {
                    float p0  = (((o0 != NULL) ? o0[p] : 0.0f) + nc) * kc;
                    float s0  = 0.0f;
                    float co0 = 1.0f;
                    sincosf(p0, &s0, &co0);
                    float re = w0 * amp0 * co0;
                    float im = w0 * amp0 * s0;

                    if (m1 != m0 && w1 > 0.0f)
                    {
                        float p1  = (((o1 != NULL) ? o1[p] : 0.0f) + nc) * kc;
                        float s1  = 0.0f;
                        float co1 = 1.0f;
                        sincosf(p1, &s1, &co1);
                        re += w1 * amp1 * co1;
                        im += w1 * amp1 * s1;
                    }
                    epup[2u * p]      = re;
                    epup[2u * p + 1u] = im;
                }
            }
        }

        zwfs_eval_babinet_chan_cpu(ctx, c);
    }

    zwfs_bin_image_cpu(ctx->cpu_pup_int, nwf, ctx->cfg.ncam, ctx->cpu_cam_int);
}

/**
 * zwfs_accum_cpu_common() - Propagate one CPU subframe and lazily accumulate multi-subframe flux
 * @ctx:       Initialized ZWFS context
 * @pup_amp:   2D entrance pupil amplitude `[nwf x nwf]`
 * @atm_amp:   Optional scintillation amplitude `[nwf x nwf]`, or NULL
 * @wf_in:     Optional incident wavefront map `[nwf x nwf]`, or NULL
 * @dm_opd:    Optional DM correction OPD `[nwf x nwf]` [um], or NULL
 * @ncpa_opd:  Optional static WFS-path NCPA OPD `[nwf x nwf]` [um], or NULL
 * @wf_to_opd: Scale factor converting `wf_in` units to OPD [um]
 *
 * Role:
 *   Shared CPU subframe integration wrapper called by `cacaosim_zwfs_accum_subframe()` and
 *   `cacaosim_zwfs_accum_beam()` to support `navg >= 1` temporal turbulence averaging prior to
 *   detector readout.
 *
 * Rationale:
 *   Uses lazy accumulator initialization (`subframe_cnt == 1u`): for the common single-subframe
 *   case (`navg == 1`), `subframe_cnt` is `0` here and `1` at `cacaosim_zwfs_readout()`, so
 *   `cpu_cam_int` is read out directly without ever touching `cpu_accum_int`. Only if a second
 *   subframe is integrated within the same camera exposure is subframe 0 copied into
 *   `cpu_accum_int` and summed with subsequent subframes.
 */
static void MILK_HOT zwfs_accum_cpu_common(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *pup_amp,
    const float         *atm_amp,
    const float         *wf_in,
    const float         *dm_opd,
    const float         *ncpa_opd,
    float                wf_to_opd)
{
    /* Lazily seed multi-subframe accumulator from subframe 0 only when subframe 1 arrives */
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

/**
 * cacaosim_zwfs_init() - Initialize Zernike phase-contrast WFS context, MFT operators, and GPU
 * @ctx:        Uninitialized ZWFS engine context to populate (non-NULL)
 * @cfg:        ZWFS optical, MFT, and detector configuration parameters (non-NULL)
 * @pup_amp:    Static 2D telescope or coronagraphic LOWFS pupil amplitude `[nwf x nwf]` (non-NULL)
 * @ncpa_opd:   Optional static WFS-path NCPA OPD `[nwf x nwf]` [um], or NULL
 * @ext_stream: Optional caller-owned `cudaStream_t` (`void *`) for fused GPU execution, or NULL
 *
 * Role:
 *   Cold-path lifecycle constructor that validates geometry, pins OpenBLAS to single-threaded
 *   mode, allocates cache-aligned workspace buffers, precomputes the Babinet MFT operators and
 *   anti-aliased dimple mask, calibrates open-pupil photometric normalization, and initializes the
 *   optional `cuBLAS` GPU backend.
 *
 * Rationale:
 *   1. Pre-allocating all aligned CPU and GPU workspaces upfront guarantees strictly zero dynamic
 *      memory allocations (`malloc`/`cudaMalloc`) inside the real-time kHz AO loop.
 *   2. Calibrating `flat_flux_norm = sum(|pup_amp|^2)` against the open entrance pupil (without
 *      the dimple mask) via Parseval's identity ensures `flux_ph` represents true incident photons
 *      entering the WFS while physically preserving throughput attenuation when a partially
 *      absorbing dimple (`dimple_amp < 1.0`) is configured.
 *   3. Gracefully falls back to the OpenBLAS CPU engine if `CACAOSIM_BACKEND_AUTO` is requested on
 *      a host without a usable CUDA device.
 *
 * Return: `0` on success, or `-1` on invalid geometry or memory allocation failure.
 */
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

    /* Prevent OpenBLAS thread-pool spin-wait contention on tiny cache-resident micro-GEMMs */
    if (openblas_set_num_threads != NULL)
    {
        openblas_set_num_threads(1);
    }

    float lam0    = (cfg->lambda_um > 0.05f) ? cfg->lambda_um : 0.75f;
    ctx->eff_nlam = cacaosim_resolve_nlambda(cfg->nlambda, 1u, cfg->bw_frac);
    if (cfg->has_ext_chan)
    {
        double wsum = 0.0;
        for (int32_t c = 0; c < ctx->eff_nlam; c++)
        {
            ctx->lam_um[c] = (cfg->ext_lam_um[c] > 0.05f) ? cfg->ext_lam_um[c] : lam0;
            ctx->chan_w[c] = (cfg->ext_weight[c] > 0.0f) ? cfg->ext_weight[c] : 0.0f;
            wsum += (double) ctx->chan_w[c];
        }
        if (wsum > 1e-20)
        {
            for (int32_t c = 0; c < ctx->eff_nlam; c++)
            {
                ctx->chan_w[c] = (float) ((double) ctx->chan_w[c] / wsum);
            }
        }
        else
        {
            float inv_n = 1.0f / (float) ctx->eff_nlam;
            for (int32_t c = 0; c < ctx->eff_nlam; c++)
            {
                ctx->chan_w[c] = inv_n;
            }
        }
    }
    else
    {
        cacaosim_spec_sample_band(NULL, lam0, cfg->bw_frac, ctx->eff_nlam, 0.001f, 1.0f,
                                  ctx->lam_um, ctx->chan_w, NULL);
    }
    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        ctx->k_rad[c] = CACAOSIM_TWO_PI_F / ctx->lam_um[c];
    }

    size_t nlam_sz     = (size_t) ctx->eff_nlam;
    size_t uv_elems    = (size_t) ctx->n_mft * cfg->nwf * 2u;
    ctx->fwd_U_cplx    = (float *) cacaosim_aligned_calloc(nlam_sz * uv_elems, sizeof(float));
    ctx->fwd_V_cplx    = (float *) cacaosim_aligned_calloc(nlam_sz * uv_elems, sizeof(float));
    ctx->inv_U_cplx    = (float *) cacaosim_aligned_calloc(nlam_sz * uv_elems, sizeof(float));
    ctx->inv_V_cplx    = (float *) cacaosim_aligned_calloc(nlam_sz * uv_elems, sizeof(float));
    ctx->mask_bab_cplx =
        (float *) cacaosim_aligned_calloc(nlam_sz * ctx->npix_mft * 2u, sizeof(float));
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

/**
 * cacaosim_zwfs_free() - Release all CPU, detector PRNG, and CUDA resources owned by a ZWFS context
 * @ctx: ZWFS context to tear down (safe to pass NULL or partially initialized context)
 *
 * Role:
 *   Cold-path destructor that frees `cuBLAS` handles, device buffers, PRNG state, and aligned host
 *   arrays, then zeros `*ctx` to prevent dangling-pointer reuse.
 *
 * Rationale:
 *   Centralizes cleanup so both normal shutdown and partial `cacaosim_zwfs_init()` rollback share
 *   a single leak-free code path.
 */
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

/**
 * cacaosim_zwfs_accum_subframe() - Propagate and integrate one split-stream optical subframe
 * @ctx:       Initialized ZWFS context
 * @pup_amp:   Host 2D pupil amplitude `[nwf x nwf]` (used on CPU; cached on GPU at init)
 * @atm_amp:   Optional host atmospheric scintillation amplitude `[nwf x nwf]`, or NULL
 * @wf_in:     Optional host atmospheric wavefront `[nwf x nwf]`, or NULL
 * @dm_opd:    Optional host DM correction OPD `[nwf x nwf]` [um], or NULL
 * @ncpa_opd:  Optional host static NCPA OPD `[nwf x nwf]` [um] (cached on GPU at init), or NULL
 * @d_dm_opd:  Optional zero-copy device pointer to DM OPD `[nwf x nwf]` [um] on GPU, or NULL
 * @turb_gain: Live atmospheric turbulence multiplier (`0.0f` disables `wf_in`)
 *
 * Role:
 *   Hot-loop split-stream subframe entry point that scales `wf_in` to microns OPD, dispatches to
 *   the active CUDA or CPU Babinet MFT backend, and increments `ctx->subframe_cnt`.
 *
 * Rationale:
 *   Decoupling optical subframe accumulation from `cacaosim_zwfs_readout()` enables multi-rate AO
 *   benches (`navg > 1`) where fast atmospheric/DM physics evolve multiple steps per WFS camera
 *   exposure, capturing realistic phase-contrast fringe smearing before noise injection. Setting
 *   `wf_eff = NULL` when `turb_gain == 0.0f` skips redundant PCIe H2D transfers during closed-loop
 *   calibration and interaction-matrix acquisition.
 */
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

/**
 * cacaosim_zwfs_accum_beam() - Propagate and integrate one unified `[OPD_um, Amp]` beam subframe
 * @ctx:        Initialized ZWFS context
 * @beam_opd:   Host 2D beam OPD slice `[nwf x nwf]` [um], or NULL when using `d_beam_opd`
 * @beam_amp:   Host 2D beam amplitude slice `[nwf x nwf]`, or NULL when using `d_beam_amp`
 * @ncpa_opd:   Optional host static WFS-path NCPA OPD `[nwf x nwf]` [um], or NULL
 * @d_beam_opd: Optional zero-copy device beam OPD slice `[nwf x nwf]` [um], or NULL
 * @d_beam_amp: Optional zero-copy device beam amplitude slice `[nwf x nwf]`, or NULL
 *
 * Role:
 *   Hot-loop optical-bench entry point for propagating a 3D `[nwf x nwf x 2]` beam stream (such as
 *   a coronagraphic `out_lowfs` reflective focal-mask or Lyot-stop pickoff beam, or a dichroic
 *   beamsplitter arm) through the Zernike phase-contrast WFS.
 *
 * Rationale:
 *   Unlike `cacaosim_zwfs_accum_subframe()`, which assumes a static entrance pupil, `in_beam`
 *   carries a dynamic per-frame amplitude slice `beam_amp` (crucial for coronagraphic LOWFS
 *   pickoffs where upstream tip-tilt shifts diffraction rings across the reflective focal stop).
 *   Accepting optional device pointers (`d_beam_opd`, `d_beam_amp`) allows `cacaosim.htsim` to
 *   chain coronagraph and LOWFS stages entirely in GPU VRAM with zero host round-trips.
 */
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

void cacaosim_zwfs_set_weights(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *weights)
{
    if (ctx == NULL || weights == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    double wsum = 0.0;
    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        wsum += (weights[c] > 0.0f) ? (double) weights[c] : 0.0;
    }
    int changed = 0;
    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        float nw = (wsum > 1e-20)
                       ? (float) (((weights[c] > 0.0f) ? (double) weights[c] : 0.0) / wsum)
                       : (1.0f / (float) ctx->eff_nlam);
        if (fabsf(nw - ctx->chan_w[c]) > 1e-6f)
        {
            changed = 1;
        }
        ctx->chan_w[c] = nw;
    }
    if (changed && ctx->use_cuda)
    {
        cacaosim_zwfs_cuda_upload_weights(ctx);
    }
}

void cacaosim_zwfs_set_disp(
    cacaosim_zwfs_ctx_t   *ctx,
    const cacaosim_disp_t *disp,
    float                  tel_diam_m,
    int32_t                in_poly_mode)
{
    if (ctx == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    ctx->in_poly_mode = in_poly_mode;

    if (disp == NULL || !disp->active)
    {
        if (ctx->disp_active)
        {
            memset(ctx->disp_sx, 0, sizeof(ctx->disp_sx));
            memset(ctx->disp_sy, 0, sizeof(ctx->disp_sy));
            ctx->disp_active = 0;
            ctx->disp_gen    = 0ULL;
            if (ctx->use_cuda)
            {
                cacaosim_zwfs_cuda_upload_disp(ctx);
            }
        }
        return;
    }

    float diam = (tel_diam_m > 0.01f) ? tel_diam_m : 8.0f;
    if (!ctx->disp_active || disp->gen != ctx->disp_gen ||
        fabsf(diam - ctx->disp_tel_diam) > 1e-4f)
    {
        cacaosim_disp_eval_slopes(disp, ctx->lam_um, (uint32_t) ctx->eff_nlam, diam, ctx->cfg.nwf,
                                  ctx->cfg.pup_diam, ctx->disp_sx, ctx->disp_sy);
        ctx->disp_active   = 1;
        ctx->disp_gen      = disp->gen;
        ctx->disp_tel_diam = diam;
        if (ctx->use_cuda)
        {
            cacaosim_zwfs_cuda_upload_disp(ctx);
        }
    }
}

void MILK_HOT cacaosim_zwfs_accum_beam_poly(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *beam_opd,
    const float         *beam_amp,
    const float         *beam_lam_um,
    uint32_t             beam_nlam,
    uint64_t             chan_stride,
    const float         *ncpa_opd,
    const float         *d_beam_opd,
    const float         *d_beam_amp)
{
    if (beam_nlam <= 1u)
    {
        cacaosim_zwfs_accum_beam(ctx, beam_opd, beam_amp, ncpa_opd, d_beam_opd, d_beam_amp);
        return;
    }

    if (ctx->use_cuda)
    {
        cacaosim_zwfs_cuda_accum_poly(ctx, beam_opd, beam_amp, beam_lam_um, beam_nlam,
                                      chan_stride, d_beam_opd, d_beam_amp);
    }
    else
    {
        if (ctx->subframe_cnt == 1u)
        {
            __builtin_memcpy(ctx->cpu_accum_int, ctx->cpu_cam_int, ctx->npix_cam * sizeof(float));
        }
        zwfs_propagate_poly_cpu(ctx, beam_opd, beam_amp, beam_lam_um, beam_nlam, chan_stride,
                                ncpa_opd);
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
    ctx->subframe_cnt++;
}

/**
 * cacaosim_zwfs_readout() - Finalize exposure average, apply detector physics, and reset counter
 * @ctx:      Initialized ZWFS context with `subframe_cnt >= 1` integrated subframes
 * @det_live: Optional live detector parameters (flux, RON, gain, shutter), or NULL for `ctx->cfg`
 * @out_u16:  Destination `uint16_t` camera buffer `[ncam x ncam]` when `out_dtype == 0`, or NULL
 * @out_f32:  Destination `float` camera buffer `[ncam x ncam]` when `out_dtype != 0`, or NULL
 *
 * Role:
 *   Hot-loop exposure completion routine that normalizes the accumulated optical subframes by
 *   `1 / subframe_cnt`, applies photometric scaling, Poisson shot noise, dark current, Gaussian
 *   readout noise, gain/bias quantization, and saturation clipping, and resets `subframe_cnt = 0`.
 *
 * Rationale:
 *   Applying detector readout physics strictly once after averaging `navg` optical subframes
 *   accurately models physical charge integration in a CCD/CMOS/sCMOS pixel well (where photon
 *   arrival statistics depend on integrated exposure flux while readout noise `ron_e` is paid once
 *   per frame readout).
 */
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

    /* Normalize multi-subframe sum into mean optical intensity per frame prior to readout */
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

/**
 * cacaosim_zwfs_step() - Convenience single-subframe (`navg = 1`) ZWFS propagation and readout
 * @ctx:       Initialized ZWFS context
 * @pup_amp:   Host 2D pupil amplitude `[nwf x nwf]`
 * @atm_amp:   Optional host atmospheric scintillation amplitude `[nwf x nwf]`, or NULL
 * @wf_in:     Optional host atmospheric wavefront `[nwf x nwf]`, or NULL
 * @dm_opd:    Optional host DM correction OPD `[nwf x nwf]` [um], or NULL
 * @ncpa_opd:  Optional host static WFS-path NCPA OPD `[nwf x nwf]` [um], or NULL
 * @d_dm_opd:  Optional zero-copy device DM OPD `[nwf x nwf]` [um], or NULL
 * @turb_gain: Live atmospheric turbulence multiplier
 * @det_live:  Optional live detector configuration override, or NULL
 * @out_u16:   Destination `uint16_t` camera frame `[ncam x ncam]`, or NULL
 * @out_f32:   Destination `float` camera frame `[ncam x ncam]`, or NULL
 *
 * Role:
 *   Single-call wrapper combining `cacaosim_zwfs_accum_subframe()` and `cacaosim_zwfs_readout()`
 *   for synchronous 1:1 WFS simulation benchmarks and unit tests.
 *
 * Rationale:
 *   Provides an ergonomic atomic frame step API while delegating all optical and noise physics to
 *   the exact same subframe accumulation and readout pipeline used in multi-rate benches.
 */
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
