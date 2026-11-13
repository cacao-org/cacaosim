// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs_engine.c
 * @brief   Shifted complex pyramid mask synthesis and L2-cached OpenMP + FFTW3f PyWFS engine
 *
 * Architectural Role in cacaosim:
 *   Implements the host-side Pyramid WFS lifecycle, analytic subpixel-integrated 4-sided pyramid
 *   phase-mask synthesis, L2-cache-resident OpenMP + FFTW3f CPU optical propagation, area-
 *   conserving detector binning, multi-subframe (`navg`) turbulence accumulation, and transparent
 *   dispatch to the GPU accelerator (`cacaosim_pyrwfs_cuda.cu`) when CUDA is enabled.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Fourier-Shift (1 + N_mod) 2D FFT Reduction:
 *      Steering a beam around a modulation circle in the pupil plane via `N_mod` tip-tilt phase
 *      ramps shifts the focal-plane PSF by `(u_k, v_k)` relative to the pyramid apex. By the
 *      Fourier shift theorem, shifting the 4-quadrant focal pyramid mask by `(-u_k, -v_k)` against
 *      a single stationary PSF produces the exact same detector-plane intensity distribution (the
 *      residual pupil-plane linear phase tilt cancels identically under `|.|^2`). Transforming the
 *      zero-padded pupil field to the focal plane once per subframe (`1` forward FFT) and applying
 *      `N_mod` precomputed shifted complex masks followed by `N_mod` inverse FFTs reduces total
 *      FFT work from `2 * N_mod` down to `1 + N_mod` 2D FFTs—nearly halving per-frame FFT cost.
 *   2. Separable Analytic Subpixel Facet Integration:
 *      A square-base 4-sided pyramid phase `exp(-i * alpha * (|u| + |v|))` separates into two 1D
 *      roof factors. Evaluating point-sampled phases near the sharp apex (`u = 0`) introduces
 *      subpixel modulation aliasing; integrating `exp(-i * alpha * |u'|)` analytically across each
 *      pixel interval `[u - 0.5, u + 0.5]` eliminates apex ringing without massive FFT padding.
 *   3. Contention-Free L2-Resident OpenMP Parallelism:
 *      Each OpenMP worker thread owns a dedicated aligned complex scratch buffer (`cpu_thr_cplx`),
 *      intensity accumulator (`cpu_thr_int`), and pre-planned backward `fftwf_plan`. For typical
 *      `256 x 256` FFT grids (`512 KiB` complex), each thread's working set stays entirely inside
 *      per-core L2 cache with zero atomic operations or lock contention across modulation steps.
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
 * @src:  Input unbinned optical detector-plane intensity [nfft * nfft] (non-NULL)
 * @nfft: Optical FFT grid dimension [pixels]
 * @ncam: Target square WFS camera grid dimension [pixels]
 * @dst:  Output binned camera intensity buffer [ncam * ncam] (non-NULL, distinct from `@src`)
 *
 * Role:
 *   Hot-loop spatial flux integration mapping the oversampled `nfft x nfft` optical pupil image
 *   onto physical `ncam x ncam` WFS detector pixels while strictly conserving total integrated
 *   photon flux (`sum(dst) == sum(src)`).
 *
 * Rationale:
 *   Optical propagation requires `nfft = nwf * pad_factor` (typically `>= 256`) to prevent spatial
 *   wrap-around between the four separated sub-pupils, whereas real Pyramid WFS detectors use much
 *   coarser pixelation (e.g. `40x40` to `120x120`) for low readout noise and high frame rates.
 *   Dispatching an exact integer box-sum fast path when `nfft % ncam == 0` avoids floating-point
 *   bounds math in the common power-of-two case, while general 2D pixel-overlap area weighting
 *   supports arbitrary non-integer `nfft / ncam` ratios (e.g. `256 -> 60` or `240 -> 96`) without
 *   flux loss or Moire aliasing.
 */
static void MILK_HOT pyrwfs_bin_image_cpu(
    const float *__restrict src,
    uint32_t                nfft,
    uint32_t                ncam,
    float *__restrict       dst)
{
    /* Identity fast path when optical FFT grid matches physical camera resolution */
    if (ncam == nfft)
    {
        __builtin_memcpy(dst, src, (size_t) nfft * nfft * sizeof(float));
        return;
    }

    /* Fast path: exact integer box binning (e.g. 256x256 FFT -> 64x64 camera, 4x4 sum) */
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

    /*
     * General fractional box integration: each camera pixel covers `[cx*scale, (cx+1)*scale] x
     * [cy*scale, (cy+1)*scale]` in FFT-grid coordinates. Accumulating exact 1D overlap widths
     * `wx * wy` preserves total optical energy across arbitrary camera geometries.
     */
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
 * @u:      Shifted 1D focal-plane pixel coordinate relative to pyramid roof ridge [focal pixels]
 * @alpha:  1D prism phase slope `pi * pup_sep` [radians / focal pixel]
 * @roof2:  Squared pyramid vertex roof rounding width [focal pixels^2] (0.0 for ideal sharp ridge)
 * @re_out: Output real part of the 1D pixel-averaged complex transmission (non-NULL)
 * @im_out: Output imaginary part of the 1D pixel-averaged complex transmission (non-NULL)
 *
 * Role:
 *   Cold/warm-path helper computing one separable 1D axis of the 4-sided pyramid focal phase mask
 *   `exp(-i * alpha * |u|)` (or its smooth hyperbolic roof-rounded counterpart).
 *
 * Rationale:
 *   A 4-sided square pyramid has separable 2D transmission `T_2D(u, v) = T_1D(u) * T_1D(v)`.
 *   When `roof2 > 0`, physical glass prism roof grinding defects smooth `|u|` into the hyperbola
 *   `sqrt(u^2 + roof2)`. When `roof2 == 0` (ideal sharp apex), naive center-point sampling causes
 *   abrupt phase jumps whenever a modulation step crosses `u = 0` on a discrete focal grid.
 *   Evaluating the exact analytical integral `int_{u-0.5}^{u+0.5} exp(-i * alpha * |u'|) du'`
 *   yields `sinc(alpha/2) * exp(-i * alpha * |u|)` outside the ridge (`|u| >= 0.5`) and a smooth
 *   split-interval integral across the ridge (`|u| < 0.5`), eliminating subpixel modulation
 *   discontinuities without expensive focal-plane supersampling.
 */
static inline void pyr_1d_edge(
    float  u,
    float  alpha,
    float  roof2,
    float *re_out,
    float *im_out)
{
    /* Smooth hyperbolic vertex roof defect when physical roof width is non-zero */
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
        /* Pixel lies strictly on one prism facet: integral reduces to sinc(alpha/2)*e^(-i*a*|u|) */
        float sinc_h = sinf(0.5f * alpha) / (0.5f * alpha);
        *re_out      = sinc_h * cosf(alpha * au);
        *im_out      = -sinc_h * sinf(alpha * au);
    }
    else
    {
        /* Pixel straddles the ridge at u' = 0: closed-form sum of [-0.5+u, 0] and [0, 0.5+u] */
        float cu = cosf(alpha * au);
        *re_out  = (2.0f * cu * sinf(0.5f * alpha)) / alpha;
        *im_out  = -(2.0f * (1.0f - cu * cosf(0.5f * alpha))) / alpha;
    }
}

/**
 * cacaosim_pyrwfs_update_masks() - Rebuild shifted focal-plane pyramid masks for new modulation
 * @ctx:        Initialized Pyramid WFS context (non-NULL)
 * @mod_rad_ld: Desired circular modulation radius [lambda/D] (0 = unmodulated static apex)
 *
 * Role:
 *   Synthesizes the stack of `eff_npts` FFT-ordered complex 2D focal-plane pyramid masks in
 *   `ctx->masks_cplx` for modulation circle radius `@mod_rad_ld` and mirrors them to GPU memory
 *   when `ctx->use_cuda` is active.
 *
 * Rationale:
 *   Each modulation point `k` sits at half-step azimuth `2*pi*(k + 0.5)/npts` so even small `npts`
 *   (e.g. 4 or 8) straddle the orthogonal prism ridges symmetrically rather than landing directly
 *   on the roof edges. Folding `amp_norm = 1 / (nfft^2 * sqrt(npts))` directly into the mask
 *   weights absorbs both the unnormalized FFTW/cuFFT round-trip scale `1 / nfft^2` and the `1/npts`
 *   modulation intensity average (`(1/sqrt(npts))^2`), saving `N_mod` array-wide multiplications
 *   every subframe. Masking the Nyquist row/column and enforcing a symmetric square field stop
 *   (`fstop_frac * nfft`) suppresses wrap-around aliasing between the four re-imaged sub-pupils.
 */
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
    /*
     * A linear phase ramp exp(-i * alpha * u) shifts the inverse-FFT pupil image by
     * +/-(alpha / (2*pi)) * nfft pixels. Choosing alpha = pi * sep places the 4 quadrant sub-pupil
     * centers at (+/- sep/2 * nfft, +/- sep/2 * nfft), separated by `sep * nfft` pixels.
     */
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
        /* Half-step azimuth offset avoids sampling directly along the X/Y prism roof ridges */
        float  ang = (CACAOSIM_TWO_PI_F * ((float) k + 0.5f)) / (float) npts;
        float  uk  = r_mod * cosf(ang);
        float  vk  = r_mod * sinf(ang);
        float *mk  = &ctx->masks_cplx[(size_t) k * ctx->npix_fft * 2u];

        for (uint32_t jy = 0; jy < nfft; jy++)
        {
            /* Convert unshifted FFT index jy in [0, nfft) to signed focal frequency iv */
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

                /* Separable 2D complex product M(u + uk, v + vk) = T_1D(u + uk) * T_1D(v + vk) */
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
 * @ctx:       Initialized Pyramid WFS context (non-NULL)
 * @pup_amp:   Optional 2D pupil amplitude [nwf * nwf] (NULL -> uniform unit aperture)
 * @atm_amp:   Optional 2D atmospheric scintillation amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional 2D input wavefront [nwf * nwf] (NULL -> 0.0)
 * @dm_opd:    Optional 2D DM corrective OPD [nwf * nwf] in microns, subtracted (NULL -> 0.0)
 * @ncpa_opd:  Optional 2D WFS-arm static NCPA OPD [nwf * nwf] in microns, added (NULL -> 0.0)
 * @wf_to_opd: Scale factor converting `@wf_in` into microns OPD
 *
 * Role:
 *   Core CPU hot-loop optical engine computing a single coherent Pyramid WFS subframe intensity
 *   image in `ctx->cpu_cam_int` via the `(1 + N_mod)` FFTW3f reduction.
 *
 * Rationale:
 *   1. Embeds the centered `nwf x nwf` complex pupil phasor `A * exp(i * 2*pi/lambda * OPD)` into
 *      the zero-padded `nfft x nfft` buffer, skipping zero-amplitude pixels outside the pupil to
 *      avoid unnecessary `cosf`/`sinf` evaluations (~45% of pixels in a circular aperture).
 *   2. Executes one 2D forward FFT into `cpu_foc_cplx`, shared read-only across all OpenMP threads.
 *   3. Distributes the `eff_npts` modulation steps statically across up to 8 OpenMP threads, each
 *      multiplying `foc` by `masks_cplx[k]` into its L2-resident private buffer `work`, running an
 *      in-place 2D backward FFT, and accumulating `|work|^2` into `cpu_thr_int[tid]`.
 *   4. Reduces thread accumulators via SIMD vector addition and bins onto `ncam x ncam`.
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

    /* Zero-fill padded complex pupil plane, then populate central [nwf x nwf] aperture */
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

    /* 1. Single 2D forward FFT: pupil -> focal plane (shared across all N_mod steps) */
    fftwf_execute_dft((fftwf_plan) ctx->fftw_plan_fwd,
                      (fftwf_complex *) ctx->cpu_pup_cplx,
                      (fftwf_complex *) ctx->cpu_foc_cplx);

    /* 2. Parallel modulation loop with thread-private L2 buffers (unmodulated uses 1 thread) */
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

            /* Complex focal multiplication: E_mod(u,v) = E_foc(u,v) * M_k(u,v) */
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

            /* Inverse FFT back to re-imaged 4-pupil plane */
            fftwf_execute_dft(pinv, (fftwf_complex *) work, (fftwf_complex *) work);

            /* Incoherent intensity sum across modulation positions around the circle */
#pragma omp simd
            for (uint64_t i = 0; i < nfft2; i++)
            {
                float dr = work[2u * i];
                float di = work[2u * i + 1u];
                acc[i]  += dr * dr + di * di;
            }
        }
    }

    /* 3. Reduce thread accumulators and area-bin onto physical WFS camera pixels */
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

/**
 * pyrwfs_accum_cpu_common() - Propagate one CPU subframe and lazily accumulate when navg > 1
 * @ctx:       Initialized Pyramid WFS context (non-NULL)
 * @pup_amp:   2D pupil or beam amplitude [nwf * nwf]
 * @atm_amp:   Optional 2D atmospheric amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional 2D wavefront or beam OPD [nwf * nwf]
 * @dm_opd:    Optional 2D DM OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @ncpa_opd:  Optional 2D NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @wf_to_opd: Scale converting `@wf_in` to microns OPD
 *
 * Role:
 *   Shared CPU subframe driver invoked by both `cacaosim_pyrwfs_accum_subframe()` and
 *   `cacaosim_pyrwfs_accum_beam()`.
 *
 * Rationale:
 *   Uses a lazy accumulation scheme to keep the `navg == 1` common case completely zero-overhead:
 *   when `subframe_cnt == 0` on entry, `pyrwfs_propagate_cpu()` writes directly into `cpu_cam_int`
 *   without touching `cpu_accum_int`. Only if a second subframe arrives (`subframe_cnt == 1` on
 *   entry) is subframe 0 copied into `cpu_accum_int` and summed with subsequent subframes.
 */
static void MILK_HOT pyrwfs_accum_cpu_common(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *ncpa_opd,
    float                  wf_to_opd)
{
    /* Lazily seed accumulator with subframe 0 only when a 2nd subframe actually arrives */
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

/**
 * cacaosim_pyrwfs_init() - Initialize Pyramid WFS masks, FFT plans, and flat-wavefront flux norm
 * @ctx:        Output Pyramid WFS context to initialize (non-NULL)
 * @cfg:        Optical and detector configuration (non-NULL)
 * @pup_amp:    Static 2D telescope pupil amplitude [nwf * nwf] (non-NULL)
 * @ncpa_opd:   Optional static WFS-path NCPA OPD [nwf * nwf] in microns (NULL -> 0)
 * @ext_stream: Optional external `cudaStream_t` to share in fused mode (NULL creates private)
 *
 * Role:
 *   Cold-path initialization of host FFTW3f plans, aligned buffers, shifted focal pyramid masks,
 *   flat-wavefront photometric reference (`flat_flux_norm`), and optional CUDA GPU backend.
 *
 * Rationale:
 *   Caps `cpu_nth` at 8 threads so `cpu_nth * nfft^2` working buffers stay within multi-core L2/L3
 *   cache without excessive memory footprint, reserves `eff_npts = cfg->mod_npts` even if initial
 *   `mod_rad_ld == 0` so the operator can enable modulation live without re-allocating buffers,
 *   and measures `flat_flux_norm` directly through `pyrwfs_propagate_cpu()` so field-stop
 *   diffraction losses are calibrated out of flat-wavefront photon counts.
 *
 * Return: 0 on success, -1 on invalid arguments or allocation/plan failure.
 */
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

/**
 * cacaosim_pyrwfs_free() - Destroy FFT plans and free all host/device Pyramid WFS buffers
 * @ctx: Pyramid WFS context to release (NULL-safe)
 *
 * Role:
 *   Cold-path cleanup releasing CUDA state, FFTW3f plans, PRNG pool, and aligned host memory.
 *
 * Rationale:
 *   Centralizes teardown for both normal exit and partial-allocation rollback during init, leaving
 *   `@ctx` zeroed so accidental double-free calls are harmless.
 */
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

/**
 * cacaosim_pyrwfs_accum_subframe() - Propagate one wavefront+DM subframe into WFS accumulator
 * @ctx:       Initialized Pyramid WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf]
 * @atm_amp:   Optional host 2D atmospheric amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] (NULL -> 0.0)
 * @dm_opd:    Optional host 2D DM OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D static NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD [nwf * nwf] for GPU zero-copy fusion (NULL -> H2D)
 * @turb_gain: Live atmospheric turbulence scale factor (0.0 bypasses `@wf_in` completely)
 * @mod_rad:   Live modulation radius [lambda/D]
 *
 * Role:
 *   Hot-loop split-stream optical subframe entry point; updates shifted masks on live `@mod_rad`
 *   changes, converts `@wf_in` units to microns OPD, dispatches to CUDA or CPU, and increments
 *   `ctx->subframe_cnt`.
 *
 * Rationale:
 *   Short-circuiting `wf_eff = NULL` when `turb_gain == 0.0f` skips both H2D PCIe transfers and
 *   host/device memory reads of the turbulence stream during interaction-matrix calibration.
 */
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

/**
 * cacaosim_pyrwfs_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into WFS
 * @ctx:        Initialized Pyramid WFS context (non-NULL)
 * @beam_opd:   Optional host 2D beam OPD [nwf * nwf] in microns
 * @beam_amp:   Optional host 2D beam amplitude [nwf * nwf]
 * @ncpa_opd:   Optional host 2D WFS-arm static NCPA OPD [nwf * nwf] in microns
 * @d_beam_opd: Optional device 2D beam OPD [nwf * nwf] in microns
 * @d_beam_amp: Optional device 2D beam amplitude [nwf * nwf]
 * @mod_rad:    Live modulation radius [lambda/D]
 *
 * Role:
 *   Hot-loop optical-bench beam subframe entry point consuming dynamic `[OPD_um, Amplitude]`
 *   planes from upstream optical elements and incrementing `ctx->subframe_cnt`.
 *
 * Rationale:
 *   Unlike `accum_subframe()`, beam OPD is already expressed in physical microns (`wf_to_opd = 1`)
 *   and beam amplitude may evolve dynamically due to upstream stops, spiders, or registration.
 */
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

/**
 * cacaosim_pyrwfs_readout() - Finalize accumulated WFS exposure and perform detector readout
 * @ctx:      Initialized Pyramid WFS context (non-NULL)
 * @det_live: Optional live detector configuration override (NULL uses `ctx->cfg.det`)
 * @out_u16:  Destination `uint16_t` camera buffer [ncam * ncam] when `out_dtype == 0`
 * @out_f32:  Destination `float` camera buffer [ncam * ncam] when `out_dtype != 0`
 *
 * Role:
 *   Hot-loop exposure normalization and electronic detector readout dispatcher.
 *
 * Rationale:
 *   Normalizes the summed multi-subframe optical intensity by `1 / subframe_cnt` so `det->flux_ph`
 *   represents the total integrated guide-star photons over the entire WFS camera exposure
 *   regardless of `navg`, advances `frame_seq` for independent noise realizations, and resets
 *   `subframe_cnt` to begin the next exposure cleanly.
 */
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

    /* Average accumulated subframes before applying per-exposure photon & readout noise */
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
 * cacaosim_pyrwfs_step() - Single-subframe Pyramid WFS propagation and camera readout
 * @ctx:       Initialized Pyramid WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf]
 * @atm_amp:   Optional host 2D atmospheric amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] (NULL -> 0.0)
 * @dm_opd:    Optional host 2D DM OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD [nwf * nwf] (NULL -> H2D copy)
 * @turb_gain: Live atmospheric turbulence gain
 * @mod_rad:   Circular modulation radius [lambda/D]
 * @det_live:  Optional live detector readout configuration (NULL uses `ctx->cfg.det`)
 * @out_u16:   Output `uint16_t` frame [ncam * ncam] when `out_dtype == 0`
 * @out_f32:   Output `float` frame [ncam * ncam] when `out_dtype != 0`
 *
 * Role:
 *   Single-call synchronous wrapper executing one optical subframe propagation and camera readout.
 *
 * Rationale:
 *   Composes `accum_subframe` + `readout` directly, benefiting from the zero-copy `subframe_cnt`
 *   fast path (`navg == 1`) without code duplication.
 */
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
