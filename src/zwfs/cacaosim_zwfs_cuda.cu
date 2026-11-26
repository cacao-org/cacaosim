// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs_cuda.cu
 * @brief   cuBLAS Babinet Matrix-DFT Zernike phase-contrast WFS GPU backend
 *
 * Architectural Role in cacaosim:
 *   Implements the CUDA-accelerated optical propagation, detector binning, multi-subframe (`navg`)
 *   exposure integration, and Philox noisy readout backend for `cacaosim.zwfs` (used both as a
 *   primary Zernike wavefront sensor and as a coronagraphic Low-Order WFS / `LOWFS` sensor).
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Babinet Matrix-DFT (`MFT`) vs Zero-Padded 2D FFT:
 *      Zernike's phase-shifting focal dimple has diameter `~1.06 lambda/D`, spanning ~1 pixel on
 *      an unpadded pupil FFT grid. Resolving its circular boundary with 2D FFTs would require
 *      `20x..32x` zero-padding (`~4096 x 4096` complex grids). By Babinet's principle:
 *        `Psi_out = Psi_pup + MFT^-1[(t_d * exp(i*theta) - 1) * D * MFT[Psi_pup]]`
 *      restricting the Fourier transform to a fine-sampled `n_mft x n_mft` (`32 x 32`) patch over
 *      the dimple and executing four tiny `cublasCgemm` matrix multiplications in microseconds.
 *   2. Zero-Transpose Row-Major `cuBLAS` Identity:
 *      `cuBLAS` assumes Fortran column-major matrices, whereas `cacaosim` stores all 2D optical
 *      arrays in C row-major layout. Using `(A * B)^T = B^T * A^T`, every `cublasCgemm` call passes
 *      row-major device pointers directly with `CUBLAS_OP_N` and reversed operand order `(B, A)`,
 *      avoiding explicit transpose kernels. Passing `beta = +1` to the final `cublasCgemm` fuses
 *      the Babinet pupil-field superposition directly onto `d_epup` in-place.
 *   3. Fused Modulus-Squared Binning & Zero-Copy Bench Chaining:
 *      `k_zwfs_bin_cam` computes `|Psi_out|^2` on the fly while integrating onto `ncam x ncam`
 *      camera pixels (1 thread per output pixel, zero atomics, zero full-pupil float scratch), and
 *      `cacaosim_zwfs_cuda_accum()` accepts device pointers (`d_pup_amp_ext`, `d_wf_opd_ext`,
 *      `d_dm_opd_ext`) so `cacaosim.htsim` keeps multi-stage beams entirely in GPU VRAM.
 */

#ifdef HAVE_CUDA

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "zwfs/cacaosim_zwfs.h"

typedef struct
{
    uint32_t m0[CACAOSIM_MAX_LAMBDA];
    uint32_t m1[CACAOSIM_MAX_LAMBDA];
    float    alpha[CACAOSIM_MAX_LAMBDA];
} zwfs_poly_map_t;

typedef struct
{
    cudaStream_t   stream;
    int            owns_stream;
    cublasHandle_t cublas;

    float         *d_pup_amp;
    float         *d_dyn_pup;
    float         *d_ncpa_opd;
    float         *d_atm_amp;
    float         *d_wf_in;
    float         *d_dm_opd;
    float         *d_k_rad;
    float         *d_chan_w;
    float         *d_disp_sx;
    float         *d_disp_sy;
    float         *d_poly_opd;
    float         *d_poly_amp;

    cuComplex     *d_fwd_U;
    cuComplex     *d_fwd_V;
    cuComplex     *d_inv_U;
    cuComplex     *d_inv_V;
    cuComplex     *d_mask_bab;
    cuComplex     *d_epup;
    cuComplex     *d_tmp;
    cuComplex     *d_efoc;

    float         *d_cam_int;
    float         *d_accum_int;
    uint16_t      *d_out_u16;
    float         *d_out_f32;
} zwfs_cuda_state_t;

__global__ static void k_zwfs_build_epup(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ k_rad,
    const float *__restrict__ disp_sx,
    const float *__restrict__ disp_sy,
    float                     wf_to_opd,
    uint32_t                  nwf,
    uint64_t                  npix_wf,
    int32_t                   nlam,
    cuComplex *__restrict__   epup)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix_wf)
    {
        return;
    }

    float amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
    if (atm_amp != NULL)
    {
        amp *= atm_amp[p];
    }
    if (amp <= 0.0f)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            epup[(uint64_t) c * npix_wf + p] = make_cuComplex(0.0f, 0.0f);
        }
        return;
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

    float p0_xy = 0.5f * (float) (nwf - 1u);
    float dx    = (float) (p % nwf) - p0_xy;
    float dy    = (float) (p / nwf) - p0_xy;

    for (int32_t c = 0; c < nlam; c++)
    {
        float phi = (opd + disp_sx[c] * dx + disp_sy[c] * dy) * k_rad[c];
        float s   = 0.0f;
        float co  = 1.0f;
        __sincosf(phi, &s, &co);
        epup[(uint64_t) c * npix_wf + p] = make_cuComplex(amp * co, amp * s);
    }
}

__global__ static void k_zwfs_build_epup_poly(
    const float *__restrict__ beam_opd,
    const float *__restrict__ beam_amp,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ k_rad,
    const float *__restrict__ disp_sx,
    const float *__restrict__ disp_sy,
    zwfs_poly_map_t           map,
    uint64_t                  chan_stride,
    uint32_t                  nwf,
    uint64_t                  npix_wf,
    int32_t                   nlam,
    int32_t                   poly_mode,
    cuComplex *__restrict__   epup)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix_wf)
    {
        return;
    }

    float nc    = (ncpa_opd != NULL) ? ncpa_opd[p] : 0.0f;
    float p0_xy = 0.5f * (float) (nwf - 1u);
    float dx    = (float) (p % nwf) - p0_xy;
    float dy    = (float) (p / nwf) - p0_xy;
    for (int32_t c = 0; c < nlam; c++)
    {
        uint32_t m0   = map.m0[c];
        uint32_t m1   = map.m1[c];
        float    w1   = map.alpha[c];
        float    w0   = 1.0f - w1;
        float    kc   = k_rad[c];
        uint64_t off0 = (uint64_t) m0 * chan_stride + p;
        uint64_t off1 = (uint64_t) m1 * chan_stride + p;
        float    amp0 = (beam_amp != NULL) ? beam_amp[off0] : 1.0f;
        float    amp1 = (beam_amp != NULL) ? beam_amp[off1] : 1.0f;
        if (amp0 <= 0.0f && (m1 == m0 || amp1 <= 0.0f))
        {
            epup[(uint64_t) c * npix_wf + p] = make_cuComplex(0.0f, 0.0f);
            continue;
        }
        float nc_c = nc + disp_sx[c] * dx + disp_sy[c] * dy;
        if (poly_mode == CACAOSIM_POLY_UNWRAPPED_OPD && m1 != m0 && w1 > 0.0f)
        {
            float amp = w0 * amp0 + w1 * amp1;
            float opd = w0 * ((beam_opd != NULL) ? beam_opd[off0] : 0.0f) +
                        w1 * ((beam_opd != NULL) ? beam_opd[off1] : 0.0f) + nc_c;
            float s   = 0.0f;
            float co  = 1.0f;
            __sincosf(opd * kc, &s, &co);
            epup[(uint64_t) c * npix_wf + p] = make_cuComplex(amp * co, amp * s);
            continue;
        }
        float p0  = (((beam_opd != NULL) ? beam_opd[off0] : 0.0f) + nc_c) * kc;
        float s0  = 0.0f;
        float co0 = 1.0f;
        __sincosf(p0, &s0, &co0);
        float re = w0 * amp0 * co0;
        float im = w0 * amp0 * s0;
        if (m1 != m0 && w1 > 0.0f)
        {
            float p1  = (((beam_opd != NULL) ? beam_opd[off1] : 0.0f) + nc_c) * kc;
            float s1  = 0.0f;
            float co1 = 1.0f;
            __sincosf(p1, &s1, &co1);
            re += w1 * amp1 * co1;
            im += w1 * amp1 * s1;
        }
        epup[(uint64_t) c * npix_wf + p] = make_cuComplex(re, im);
    }
}

__global__ static void k_zwfs_mul_mask(
    cuComplex *__restrict__       efoc,
    const cuComplex *__restrict__ mask_bab,
    uint64_t                      npix_mft)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix_mft)
    {
        cuComplex e = efoc[i];
        cuComplex m = mask_bab[i];
        efoc[i]     = make_cuComplex(e.x * m.x - e.y * m.y, e.x * m.y + e.y * m.x);
    }
}

__global__ static void k_zwfs_bin_cam(
    const cuComplex *__restrict__ epup,
    const float *__restrict__     chan_w,
    uint32_t                      nwf,
    uint32_t                      ncam,
    int32_t                       nlam,
    float *__restrict__           cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx      = cid % ncam;
    uint32_t cy      = cid / ncam;
    uint64_t npix_wf = (uint64_t) nwf * nwf;

    if (nwf == ncam)
    {
        float sum = 0.0f;
        for (int32_t c = 0; c < nlam; c++)
        {
            cuComplex z = epup[(uint64_t) c * npix_wf + cid];
            sum += chan_w[c] * (z.x * z.x + z.y * z.y);
        }
        cam_int[cid] = sum;
        return;
    }

    if (nwf > ncam && (nwf % ncam) == 0u)
    {
        uint32_t bin = nwf / ncam;
        uint32_t x0  = cx * bin;
        uint32_t y0  = cy * bin;
        float    acc = 0.0f;
        for (uint32_t by = 0; by < bin; by++)
        {
            uint64_t roff = (uint64_t) (y0 + by) * nwf + x0;
            for (uint32_t bx = 0; bx < bin; bx++)
            {
                uint64_t p     = roff + bx;
                float    p_int = 0.0f;
                for (int32_t c = 0; c < nlam; c++)
                {
                    cuComplex z = epup[(uint64_t) c * npix_wf + p];
                    p_int += chan_w[c] * (z.x * z.x + z.y * z.y);
                }
                acc += p_int;
            }
        }
        cam_int[cid] = acc;
        return;
    }

    float scale = (float) nwf / (float) ncam;
    float fx0   = (float) cx * scale;
    float fx1   = (float) (cx + 1u) * scale;
    float fy0   = (float) cy * scale;
    float fy1   = (float) (cy + 1u) * scale;
    int   ix0   = (int) floorf(fx0);
    int   ix1   = (int) ceilf(fx1);
    int   iy0   = (int) floorf(fy0);
    int   iy1   = (int) ceilf(fy1);
    float acc   = 0.0f;

    for (int iy = iy0; iy < iy1; iy++)
    {
        if (iy < 0 || iy >= (int) nwf)
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
            if (ix < 0 || ix >= (int) nwf)
            {
                continue;
            }
            float wx0 = (fx0 > (float) ix) ? fx0 : (float) ix;
            float wx1 = (fx1 < (float) (ix + 1)) ? fx1 : (float) (ix + 1);
            float wx  = wx1 - wx0;
            if (wx > 0.0f)
            {
                uint64_t p     = (uint64_t) iy * nwf + (uint32_t) ix;
                float    p_int = 0.0f;
                for (int32_t c = 0; c < nlam; c++)
                {
                    cuComplex z = epup[(uint64_t) c * npix_wf + p];
                    p_int += chan_w[c] * (z.x * z.x + z.y * z.y);
                }
                acc += wx * wy * p_int;
            }
        }
    }
    cam_int[cid] = acc;
}

/**
 * k_zwfs_add_accum() - Accumulate one optical subframe intensity into multi-subframe exposure sum
 * @cam_int:   Current subframe binned camera intensity `[npix]`
 * @npix:      Total camera pixel count (`ncam * ncam`)
 * @accum_int: In/out running multi-subframe exposure sum `[npix]`
 *
 * Role:
 *   Integrates successive optical subframes (`subframe_cnt >= 1`) when `navg > 1`.
 *
 * Rationale:
 *   Invoked only when multiple subframes are averaged per exposure, preserving zero accumulation
 *   overhead for single-subframe (`navg == 1`) operation.
 */
__global__ static void k_zwfs_add_accum(
    const float *__restrict__ cam_int,
    uint64_t                  npix,
    float *__restrict__       accum_int)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        accum_int[i] += cam_int[i];
    }
}

/**
 * k_zwfs_scale_accum() - Normalize multi-subframe intensity sum by `1 / subframe_cnt`
 * @accum_int: Integrated multi-subframe intensity sum `[npix]`
 * @inv_n:     Reciprocal subframe count `1.0f / subframe_cnt`
 * @npix:      Total camera pixel count (`ncam * ncam`)
 * @cam_int:   Output mean per-frame optical intensity `[npix]` ready for detector readout
 *
 * Role:
 *   Divides the accumulated subframe intensity by `subframe_cnt` prior to detector noise readout.
 *
 * Rationale:
 *   Using pre-inverted `@inv_n` replaces per-pixel floating-point division with multiplication and
 *   normalizes the mean optical distribution so `flux_ph` remains invariant to `navg`.
 */
__global__ static void k_zwfs_scale_accum(
    const float *__restrict__ accum_int,
    float                     inv_n,
    uint64_t                  npix,
    float *__restrict__       cam_int)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        cam_int[i] = accum_int[i] * inv_n;
    }
}

/**
 * cacaosim_zwfs_cuda_init() - Allocate GPU buffers, bind cuBLAS, and upload Babinet MFT operators
 * @ctx:        Host-initialized ZWFS context containing precomputed MFT matrices and Babinet mask
 * @pup_amp:    Static host 2D entrance pupil amplitude `[nwf x nwf]`
 * @ncpa_opd:   Optional static host 2D WFS-path NCPA OPD `[nwf x nwf]` [um], or NULL
 * @ext_stream: Optional caller-owned `cudaStream_t` (`void *`) for fused bench execution, or NULL
 *
 * Role:
 *   Cold-path GPU backend constructor invoked by `cacaosim_zwfs_init()` to provision the `cuBLAS`
 *   handle, CUDA stream, and all persistent device workspaces, and to upload static optical tables.
 *
 * Rationale:
 *   Front-loading every `cudaMalloc()`, `cublasCreate()`, and static H2D copy guarantees zero
 *   runtime GPU memory management overhead or implicit device synchronization during real-time AO
 *   operation. Sharing `@ext_stream` when supplied by `cacaosim.htsim` enables asynchronous GPU
 *   pipelining across upstream coronagraph / DM stages and ZWFS sensing.
 *
 * Return: `0` on success, or `-1` on CUDA stream, `cuBLAS`, or VRAM allocation failure.
 */
static void zwfs_cuda_run_babinet_batched(
    cacaosim_zwfs_ctx_t *ctx,
    zwfs_cuda_state_t   *cs,
    int32_t              nlam)
{
    int             nwf      = (int) ctx->cfg.nwf;
    int             m        = (int) ctx->n_mft;
    long long       uv_cplx  = (long long) ctx->n_mft * ctx->cfg.nwf;
    long long       wf_cplx  = (long long) ctx->npix_wf;
    long long       mft_cplx = (long long) ctx->npix_mft;
    const cuComplex one      = make_cuComplex(1.0f, 0.0f);
    const cuComplex zero     = make_cuComplex(0.0f, 0.0f);

    cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, m, nwf, &one, cs->d_epup,
                              nwf, wf_cplx, cs->d_fwd_U, nwf, uv_cplx, &zero, cs->d_tmp, nwf,
                              uv_cplx, nlam);
    cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, m, nwf, &one, cs->d_fwd_V, m,
                              uv_cplx, cs->d_tmp, nwf, uv_cplx, &zero, cs->d_efoc, m, mft_cplx,
                              nlam);

    uint64_t total_mft = (uint64_t) nlam * ctx->npix_mft;
    uint32_t bl_mft    = (uint32_t) ((total_mft + 255ULL) / 256ULL);
    k_zwfs_mul_mask<<<bl_mft, 256, 0, cs->stream>>>(cs->d_efoc, cs->d_mask_bab, total_mft);

    cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, nwf, m, &one, cs->d_efoc, m,
                              mft_cplx, cs->d_inv_U, m, uv_cplx, &zero, cs->d_tmp, m, uv_cplx,
                              nlam);
    cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, m, &one, cs->d_inv_V,
                              nwf, uv_cplx, cs->d_tmp, m, uv_cplx, &one, cs->d_epup, nwf, wf_cplx,
                              nlam);

    uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + 255ULL) / 256ULL);
    k_zwfs_bin_cam<<<bl_cam, 256, 0, cs->stream>>>(cs->d_epup, cs->d_chan_w, ctx->cfg.nwf,
                                                   ctx->cfg.ncam, nlam, cs->d_cam_int);

    if (ctx->subframe_cnt >= 1u)
    {
        k_zwfs_add_accum<<<bl_cam, 256, 0, cs->stream>>>(
            cs->d_cam_int, ctx->npix_cam, cs->d_accum_int);
    }
}

extern "C" int cacaosim_zwfs_cuda_init(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *pup_amp,
    const float         *ncpa_opd,
    void                *ext_stream)
{
    zwfs_cuda_state_t *cs = (zwfs_cuda_state_t *) calloc(1, sizeof(*cs));
    if (cs == NULL)
    {
        return -1;
    }
    ctx->cuda_priv = cs;

    if (ext_stream != NULL)
    {
        cs->stream      = (cudaStream_t) ext_stream;
        cs->owns_stream = 0;
    }
    else
    {
        if (cudaStreamCreateWithFlags(&cs->stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            cacaosim_zwfs_cuda_free(ctx);
            return -1;
        }
        cs->owns_stream = 1;
    }

    if (cublasCreate(&cs->cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(cs->cublas, cs->stream) != CUBLAS_STATUS_SUCCESS)
    {
        cacaosim_zwfs_cuda_free(ctx);
        return -1;
    }

    int32_t nlam       = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    size_t  nlam_sz    = (size_t) nlam;
    size_t  wf_bytes   = (size_t) ctx->npix_wf * sizeof(float);
    size_t  lam_bytes  = nlam_sz * sizeof(float);
    size_t  poly_bytes = (size_t) (2 * nlam) * wf_bytes;
    size_t  uv_bytes   = nlam_sz * ctx->n_mft * ctx->cfg.nwf * sizeof(cuComplex);
    size_t  mft_bytes  = nlam_sz * ctx->npix_mft * sizeof(cuComplex);
    size_t  ep_bytes   = nlam_sz * ctx->npix_wf * sizeof(cuComplex);
    size_t  cam_f      = (size_t) ctx->npix_cam * sizeof(float);
    size_t  cam_u      = (size_t) ctx->npix_cam * sizeof(uint16_t);

    if (cudaMalloc((void **) &cs->d_pup_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dyn_pup, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_atm_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_wf_in, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dm_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_k_rad, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_chan_w, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_disp_sx, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_disp_sy, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_poly_opd, poly_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_poly_amp, poly_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_fwd_U, uv_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_fwd_V, uv_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_inv_U, uv_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_inv_V, uv_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_mask_bab, mft_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_epup, ep_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_tmp, uv_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_efoc, mft_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_cam_int, cam_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_accum_int, cam_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_u16, cam_u) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_f32, cam_f) != cudaSuccess)
    {
        cacaosim_zwfs_cuda_free(ctx);
        return -1;
    }

    if (ncpa_opd != NULL)
    {
        if (cudaMalloc((void **) &cs->d_ncpa_opd, wf_bytes) != cudaSuccess)
        {
            cacaosim_zwfs_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_ncpa_opd, ncpa_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    }

    cudaMemcpyAsync(cs->d_pup_amp, pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_k_rad, ctx->k_rad, lam_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_chan_w, ctx->chan_w, lam_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_disp_sx, ctx->disp_sx, lam_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_disp_sy, ctx->disp_sy, lam_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_fwd_U, ctx->fwd_U_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_fwd_V, ctx->fwd_V_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_inv_U, ctx->inv_U_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_inv_V, ctx->inv_V_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_mask_bab, ctx->mask_bab_cplx, mft_bytes, cudaMemcpyHostToDevice,
                    cs->stream);

    cudaStreamSynchronize(cs->stream);
    return 0;
}

extern "C" void cacaosim_zwfs_cuda_free(cacaosim_zwfs_ctx_t *ctx)
{
    zwfs_cuda_state_t *cs = (zwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL)
    {
        return;
    }
    if (cs->cublas)
    {
        cublasDestroy(cs->cublas);
    }
    cudaFree(cs->d_pup_amp);
    cudaFree(cs->d_dyn_pup);
    cudaFree(cs->d_ncpa_opd);
    cudaFree(cs->d_atm_amp);
    cudaFree(cs->d_wf_in);
    cudaFree(cs->d_dm_opd);
    cudaFree(cs->d_k_rad);
    cudaFree(cs->d_chan_w);
    cudaFree(cs->d_disp_sx);
    cudaFree(cs->d_disp_sy);
    cudaFree(cs->d_poly_opd);
    cudaFree(cs->d_poly_amp);
    cudaFree(cs->d_fwd_U);
    cudaFree(cs->d_fwd_V);
    cudaFree(cs->d_inv_U);
    cudaFree(cs->d_inv_V);
    cudaFree(cs->d_mask_bab);
    cudaFree(cs->d_epup);
    cudaFree(cs->d_tmp);
    cudaFree(cs->d_efoc);
    cudaFree(cs->d_cam_int);
    cudaFree(cs->d_accum_int);
    cudaFree(cs->d_out_u16);
    cudaFree(cs->d_out_f32);
    if (cs->owns_stream && cs->stream)
    {
        cudaStreamDestroy(cs->stream);
    }
    free(cs);
    ctx->cuda_priv = NULL;
}

extern "C" void cacaosim_zwfs_cuda_upload_weights(cacaosim_zwfs_ctx_t *ctx)
{
    zwfs_cuda_state_t *cs = (zwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    cudaMemcpyAsync(cs->d_chan_w, ctx->chan_w, (size_t) ctx->eff_nlam * sizeof(float),
                    cudaMemcpyHostToDevice, cs->stream);
}

extern "C" void cacaosim_zwfs_cuda_upload_disp(cacaosim_zwfs_ctx_t *ctx)
{
    zwfs_cuda_state_t *cs = (zwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    size_t lam_bytes = (size_t) ctx->eff_nlam * sizeof(float);
    cudaMemcpyAsync(cs->d_disp_sx, ctx->disp_sx, lam_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_disp_sy, ctx->disp_sy, lam_bytes, cudaMemcpyHostToDevice, cs->stream);
}

extern "C" void cacaosim_zwfs_cuda_accum(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *dyn_pup_amp,
    const float         *atm_amp,
    const float         *wf_in,
    const float         *dm_opd,
    const float         *d_pup_amp_ext,
    const float         *d_wf_opd_ext,
    const float         *d_dm_opd_ext,
    float                wf_to_opd)
{
    zwfs_cuda_state_t *cs       = (zwfs_cuda_state_t *) ctx->cuda_priv;
    size_t             wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    int32_t            nlam     = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;

    const float *d_pup = cs->d_pup_amp;
    if (d_pup_amp_ext != NULL)
    {
        d_pup = d_pup_amp_ext;
    }
    else if (dyn_pup_amp != NULL)
    {
        cudaMemcpyAsync(cs->d_dyn_pup, dyn_pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_pup = cs->d_dyn_pup;
    }

    const float *d_amp = NULL;
    if (atm_amp != NULL)
    {
        cudaMemcpyAsync(cs->d_atm_amp, atm_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_amp = cs->d_atm_amp;
    }

    const float *d_wf = NULL;
    if (d_wf_opd_ext != NULL)
    {
        d_wf = d_wf_opd_ext;
    }
    else if (wf_in != NULL && wf_to_opd != 0.0f)
    {
        cudaMemcpyAsync(cs->d_wf_in, wf_in, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_wf = cs->d_wf_in;
    }

    const float *d_dm = NULL;
    if (d_dm_opd_ext != NULL)
    {
        d_dm = d_dm_opd_ext;
    }
    else if (dm_opd != NULL)
    {
        cudaMemcpyAsync(cs->d_dm_opd, dm_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_dm = cs->d_dm_opd;
    }

    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    uint32_t bl_wf = (uint32_t) ((ctx->npix_wf + 255ULL) / 256ULL);
    k_zwfs_build_epup<<<bl_wf, 256, 0, cs->stream>>>(
        d_pup, d_amp, d_wf, d_dm, cs->d_ncpa_opd, cs->d_k_rad, cs->d_disp_sx, cs->d_disp_sy,
        wf_to_opd, ctx->cfg.nwf, ctx->npix_wf, nlam, cs->d_epup);

    zwfs_cuda_run_babinet_batched(ctx, cs, nlam);
}

extern "C" void cacaosim_zwfs_cuda_accum_poly(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *beam_opd,
    const float         *beam_amp,
    const float         *beam_lam_um,
    uint32_t             beam_nlam,
    uint64_t             chan_stride,
    const float         *d_beam_opd,
    const float         *d_beam_amp)
{
    zwfs_cuda_state_t *cs       = (zwfs_cuda_state_t *) ctx->cuda_priv;
    size_t             wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    int32_t            nlam     = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;

    float lam0   = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 0.75f;
    float bw     = (ctx->cfg.bw_frac > 0.0f) ? ctx->cfg.bw_frac : 0.20f;
    float def_l0 = lam0 * (1.0f - 0.5f * bw);
    float def_l1 = lam0 * (1.0f + 0.5f * bw);

    zwfs_poly_map_t map        = { { 0 }, { 0 }, { 0.0f } };
    const float    *d_opd_use  = d_beam_opd;
    const float    *d_amp_use  = d_beam_amp;
    uint64_t        stride_use = chan_stride;

    int32_t unwrapped = (ctx->in_poly_mode == CACAOSIM_POLY_UNWRAPPED_OPD) ? 1 : 0;
    for (int32_t c = 0; c < nlam; c++)
    {
        uint32_t m0 = 0u;
        uint32_t m1 = 0u;
        float    al = 0.0f;
        cacaosim_poly_interp_bracket_ex(ctx->lam_um[c], beam_lam_um, beam_nlam, def_l0, def_l1,
                                        unwrapped, &m0, &m1, &al);
        if (d_beam_opd == NULL && beam_opd != NULL)
        {
            uint32_t s0 = (uint32_t) (2 * c);
            uint32_t s1 = (m1 == m0) ? s0 : (s0 + 1u);
            cudaMemcpyAsync(cs->d_poly_opd + (uint64_t) s0 * ctx->npix_wf,
                            beam_opd + (uint64_t) m0 * chan_stride, wf_bytes,
                            cudaMemcpyHostToDevice, cs->stream);
            if (s1 != s0)
            {
                cudaMemcpyAsync(cs->d_poly_opd + (uint64_t) s1 * ctx->npix_wf,
                                beam_opd + (uint64_t) m1 * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
            if (beam_amp != NULL)
            {
                cudaMemcpyAsync(cs->d_poly_amp + (uint64_t) s0 * ctx->npix_wf,
                                beam_amp + (uint64_t) m0 * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
                if (s1 != s0)
                {
                    cudaMemcpyAsync(cs->d_poly_amp + (uint64_t) s1 * ctx->npix_wf,
                                    beam_amp + (uint64_t) m1 * chan_stride, wf_bytes,
                                    cudaMemcpyHostToDevice, cs->stream);
                }
            }
            map.m0[c]    = s0;
            map.m1[c]    = s1;
            map.alpha[c] = al;
            d_opd_use    = cs->d_poly_opd;
            d_amp_use    = (beam_amp != NULL) ? cs->d_poly_amp : NULL;
            stride_use   = ctx->npix_wf;
        }
        else
        {
            map.m0[c]    = m0;
            map.m1[c]    = m1;
            map.alpha[c] = al;
        }
    }

    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    uint32_t bl_wf = (uint32_t) ((ctx->npix_wf + 255ULL) / 256ULL);
    k_zwfs_build_epup_poly<<<bl_wf, 256, 0, cs->stream>>>(
        d_opd_use, d_amp_use, cs->d_ncpa_opd, cs->d_k_rad, cs->d_disp_sx, cs->d_disp_sy, map,
        stride_use, ctx->cfg.nwf, ctx->npix_wf, nlam, ctx->in_poly_mode, cs->d_epup);

    zwfs_cuda_run_babinet_batched(ctx, cs, nlam);
}

/**
 * cacaosim_zwfs_cuda_readout() - Average GPU exposure, synthesize detector noise, and copy to host
 * @ctx:      Initialized ZWFS context with `subframe_cnt >= 1` integrated subframes
 * @det_live: Active detector readout parameters (flux, RON, gain, bias, saturation, shutter)
 * @out_u16:  Host destination `uint16_t` frame `[ncam x ncam]` when `out_dtype == 0`, or NULL
 * @out_f32:  Host destination `float` frame `[ncam x ncam]` when `out_dtype != 0`, or NULL
 *
 * Role:
 *   Hot-loop GPU exposure finalizer that normalizes multi-subframe accumulation, runs the Philox
 *   CUDA detector readout kernel, transfers only the tiny `ncam x ncam` camera image to host
 *   shared memory, and synchronizes `cs->stream`.
 *
 * Rationale:
 *   Executing Poisson photon shot noise, Gaussian readout noise, and ADU quantization on the GPU
 *   keeps the entire optical-to-detector pipeline on-device and restricts D2H PCIe traffic to a
 *   single compact camera frame (`8 KiB` for `64 x 64` `uint16_t`).
 */
extern "C" void cacaosim_zwfs_cuda_readout(
    cacaosim_zwfs_ctx_t      *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    zwfs_cuda_state_t *cs     = (zwfs_cuda_state_t *) ctx->cuda_priv;
    uint32_t           bl_cam = (uint32_t) ((ctx->npix_cam + 255ULL) / 256ULL);

    if (ctx->subframe_cnt > 1u)
    {
        k_zwfs_scale_accum<<<bl_cam, 256, 0, cs->stream>>>(
            cs->d_accum_int, 1.0f / (float) ctx->subframe_cnt, ctx->npix_cam, cs->d_cam_int);
    }

    cacaosim_cuda_detector_readout(det_live, cs->d_cam_int, ctx->npix_cam, ctx->flat_flux_norm,
                                   ctx->cfg.rng_seed, ctx->frame_seq, cs->d_out_u16, cs->d_out_f32,
                                   cs->stream);

    if (det_live->out_dtype == 0 && out_u16 != NULL)
    {
        cudaMemcpyAsync(out_u16, cs->d_out_u16, (size_t) ctx->npix_cam * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, cs->stream);
    }
    else if (out_f32 != NULL)
    {
        cudaMemcpyAsync(out_f32, cs->d_out_f32, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToHost, cs->stream);
    }
    cudaStreamSynchronize(cs->stream);
}

#endif /* HAVE_CUDA */
