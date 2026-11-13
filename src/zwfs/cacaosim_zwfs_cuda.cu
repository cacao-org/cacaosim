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

#include "common/cacaosim_cuda.h"
#include "zwfs/cacaosim_zwfs.h"

/**
 * struct zwfs_cuda_state_t - Private GPU state and pre-allocated VRAM buffers for ZWFS propagation
 * @stream:      Active CUDA execution stream (owned internally or borrowed from `cacaosim.htsim`)
 * @owns_stream: `1` if `stream` was created by `cacaosim_zwfs_cuda_init()` and must be destroyed
 * @cublas:      `cuBLAS` handle bound to `@stream` for separable Matrix-DFT `cublasCgemm` calls
 * @d_pup_amp:   Static entrance pupil amplitude `[nwf x nwf]` uploaded at initialization
 * @d_dyn_pup:   Staging buffer `[nwf x nwf]` for dynamic host `in_beam` pupil amplitude slices
 * @d_ncpa_opd:  Optional static WFS-path NCPA OPD `[nwf x nwf]` [um] on device, or NULL
 * @d_atm_amp:   Staging buffer `[nwf x nwf]` for host atmospheric scintillation maps
 * @d_wf_in:     Staging buffer `[nwf x nwf]` for host incident wavefront phase/OPD maps
 * @d_dm_opd:    Staging buffer `[nwf x nwf]` for host DM correction OPD maps [um]
 * @d_fwd_U:     Row-major forward vertical MFT matrix `[n_mft x nwf]` (`cuComplex`)
 * @d_fwd_V:     Row-major forward horizontal MFT matrix `[nwf x n_mft]` (`cuComplex`)
 * @d_inv_U:     Row-major inverse vertical MFT matrix `[nwf x n_mft]` (`cuComplex`)
 * @d_inv_V:     Row-major inverse horizontal MFT matrix `[n_mft x nwf]` (`cuComplex`)
 * @d_mask_bab:  Pre-baked Babinet dimple mask `[n_mft x n_mft]` (`(t_d * exp(i*theta) - 1) * D`)
 * @d_epup:      Complex pupil field `[nwf x nwf]` (entrance `Psi_pup` -> interfered `Psi_out`)
 * @d_tmp:       Intermediate separable MFT workspace `[n_mft x nwf]` / `[nwf x n_mft]`
 * @d_efoc:      Fine-sampled complex focal-plane dimple field `[n_mft x n_mft]`
 * @d_cam_int:   Single-subframe binned noiseless detector intensity `[ncam x ncam]`
 * @d_accum_int: Multi-subframe (`navg > 1`) exposure intensity accumulator `[ncam x ncam]`
 * @d_out_u16:   Device quantized `uint16_t` detector readout frame `[ncam x ncam]`
 * @d_out_f32:   Device `float` detector readout frame `[ncam x ncam]`
 */
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

/**
 * k_zwfs_build_epup() - Synthesize the complex entrance pupil field `Psi_pup(x, y)` on GPU
 * @pup_amp:    2D pupil amplitude `[npix_wf]`, or NULL for uniform unit transmission
 * @atm_amp:    Optional multiplicative atmospheric scintillation amplitude `[npix_wf]`, or NULL
 * @wf_in:      Optional incident wavefront map `[npix_wf]`, or NULL
 * @dm_opd:     Optional subtracted DM surface OPD `[npix_wf]` [um], or NULL
 * @ncpa_opd:   Optional additive static WFS-path NCPA OPD `[npix_wf]` [um], or NULL
 * @wf_to_opd:  Scale factor converting `@wf_in` into microns of OPD
 * @opd_to_rad: Wavenumber `2 * pi / lambda_um` [rad/um]
 * @npix_wf:    Total pupil grid pixel count (`nwf * nwf`)
 * @epup:       Output complex pupil field `[npix_wf]`
 *
 * Role:
 *   Fused 1D elementwise CUDA kernel that combines amplitude apodization, turbulence/beam OPD, DM
 *   correction, and static NCPA into `A(x, y) * exp(i * 2*pi/lambda * OPD)` in a single pass.
 *
 * Rationale:
 *   Fusing all optional optical terms into one memory-bandwidth-bound kernel avoids intermediate
 *   VRAM reads/writes, while early-exiting on masked pixels (`amp <= 0`) skips SFU `__sincosf()`
 *   instructions outside the clear telescope aperture.
 */
__global__ static void k_zwfs_build_epup(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    float                     wf_to_opd,
    float                     opd_to_rad,
    uint64_t                  npix_wf,
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
        epup[p] = make_cuComplex(0.0f, 0.0f);
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

    float phi = opd * opd_to_rad;
    float s   = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &s, &co);
    epup[p] = make_cuComplex(amp * co, amp * s);
}

/**
 * k_zwfs_mul_mask() - Multiply focal-plane dimple patch field in-place by Babinet complex mask
 * @efoc:     In/out complex focal-plane field `[npix_mft]` over the fine-sampled dimple window
 * @mask_bab: Precomputed complex Babinet mask `(t_d * exp(i * dimple_rad) - 1) * D(u, v)`
 * @npix_mft: Total focal MFT sample count (`n_mft * n_mft`, typically `1024`)
 *
 * Role:
 *   Applies the subpixel-anti-aliased Zernike phase-contrast dimple transfer function between the
 *   forward and inverse `cublasCgemm` Matrix-DFT stages.
 *
 * Rationale:
 *   Because `(t_d * exp(i * dimple_rad) - 1)` is pre-multiplied into `@mask_bab` during cold init,
 *   this single `n_mft^2` complex multiply prepares the exact Babinet perturbation wave for direct
 *   `beta = +1` accumulation onto `d_epup`.
 */
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

/**
 * k_zwfs_bin_cam() - Compute output pupil intensity `|Psi_out|^2` and bin onto detector pixels
 * @epup:    Interfered output pupil complex field `[nwf x nwf]`
 * @nwf:     Source pupil grid linear size [pixels]
 * @ncam:    Destination WFS camera grid linear size [pixels]
 * @cam_int: Output flux-conserving 2D detector intensity `[ncam x ncam]`
 *
 * Role:
 *   Converts the Babinet-interfered pupil complex amplitude `@epup` into optical intensity and
 *   resamples it from `nwf x nwf` onto the `ncam x ncam` camera pixel array.
 *
 * Rationale:
 *   Fusing modulus-squared evaluation (`z.x^2 + z.y^2`) directly into gather-based detector
 *   binning (1 thread per output camera pixel) eliminates an intermediate `nwf x nwf` float buffer
 *   and avoids GPU atomic contention. Provides specialized fast paths for `nwf == ncam` (1:1) and
 *   integer superpixel factors (`nwf % ncam == 0`), plus exact 2D fractional overlap integration
 *   for arbitrary non-integer pupil-to-detector pitch ratios.
 */
__global__ static void k_zwfs_bin_cam(
    const cuComplex *__restrict__ epup,
    uint32_t                      nwf,
    uint32_t                      ncam,
    float *__restrict__           cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx = cid % ncam;
    uint32_t cy = cid / ncam;

    /* Fast path 1: 1:1 pupil-to-camera sampling */
    if (nwf == ncam)
    {
        cuComplex z  = epup[cid];
        cam_int[cid] = z.x * z.x + z.y * z.y;
        return;
    }

    /* Fast path 2: exact integer bin x bin superpixel summation */
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
                cuComplex z = epup[roff + bx];
                acc += z.x * z.x + z.y * z.y;
            }
        }
        cam_int[cid] = acc;
        return;
    }

    /* General path: exact 2D geometric pixel-overlap area weighting */
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
                cuComplex z = epup[(uint64_t) iy * nwf + (uint32_t) ix];
                acc += wx * wy * (z.x * z.x + z.y * z.y);
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

    size_t wf_bytes  = (size_t) ctx->npix_wf * sizeof(float);
    size_t uv_bytes  = (size_t) ctx->n_mft * ctx->cfg.nwf * sizeof(cuComplex);
    size_t mft_bytes = (size_t) ctx->npix_mft * sizeof(cuComplex);
    size_t ep_bytes  = (size_t) ctx->npix_wf * sizeof(cuComplex);
    size_t cam_f     = (size_t) ctx->npix_cam * sizeof(float);
    size_t cam_u     = (size_t) ctx->npix_cam * sizeof(uint16_t);

    if (cudaMalloc((void **) &cs->d_pup_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dyn_pup, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_atm_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_wf_in, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dm_opd, wf_bytes) != cudaSuccess ||
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
    cudaMemcpyAsync(cs->d_fwd_U, ctx->fwd_U_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_fwd_V, ctx->fwd_V_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_inv_U, ctx->inv_U_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_inv_V, ctx->inv_V_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_mask_bab, ctx->mask_bab_cplx, mft_bytes, cudaMemcpyHostToDevice,
                    cs->stream);

    cudaStreamSynchronize(cs->stream);
    return 0;
}

/**
 * cacaosim_zwfs_cuda_free() - Destroy cuBLAS handle, owned stream, and all ZWFS device buffers
 * @ctx: ZWFS context owning `cuda_priv` (safe when `cuda_priv == NULL` or partially allocated)
 *
 * Role:
 *   Cold-path GPU teardown routine invoked by `cacaosim_zwfs_free()` or on init rollback.
 *
 * Rationale:
 *   Releases owned CUDA streams while leaving externally borrowed streams (`owns_stream == 0`)
 *   intact for the parent `cacaosim.htsim` orchestrator.
 */
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

/**
 * cacaosim_zwfs_cuda_accum() - Enqueue one asynchronous GPU Babinet MFT ZWFS subframe propagation
 * @ctx:           Initialized ZWFS context with active CUDA backend
 * @dyn_pup_amp:   Optional dynamic host pupil amplitude `[nwf x nwf]` (`in_beam`), or NULL
 * @atm_amp:       Optional host atmospheric scintillation amplitude `[nwf x nwf]`, or NULL
 * @wf_in:         Optional host incident wavefront `[nwf x nwf]`, or NULL
 * @dm_opd:        Optional host DM correction OPD `[nwf x nwf]` [um], or NULL
 * @d_pup_amp_ext: Optional zero-copy device dynamic pupil amplitude `[nwf x nwf]`, or NULL
 * @d_wf_opd_ext:  Optional zero-copy device beam/wavefront OPD `[nwf x nwf]` [um], or NULL
 * @d_dm_opd_ext:  Optional zero-copy device DM correction OPD `[nwf x nwf]` [um], or NULL
 * @wf_to_opd:     Scale factor converting `@wf_in` / `@d_wf_opd_ext` into microns of OPD
 *
 * Role:
 *   Hot-loop asynchronous GPU subframe propagator that stages any host inputs, synthesizes the
 *   complex pupil field, runs the 4-GEMM `cuBLAS` Babinet Matrix-DFT, bins intensity onto
 *   `d_cam_int`, and lazily accumulates into `d_accum_int` when `navg > 1`.
 *
 * Rationale:
 *   - Prioritizes zero-copy device pointers (`*_ext`) over host H2D copies so fused GPU benches
 *     incur zero PCIe traffic between upstream optical elements and the ZWFS.
 *   - Enqueues all transfers, `cublasCgemm` calls, and kernels asynchronously onto `cs->stream`
 *     without host blocking; synchronization occurs once per camera frame at readout.
 */
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
    int                nwf      = (int) ctx->cfg.nwf;
    int                m        = (int) ctx->n_mft;

    /* Select zero-copy device pupil, dynamic host beam amplitude, or static cached pupil */
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

    /* Lazily seed multi-subframe accumulator from subframe 0 only when subframe 1 arrives */
    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    float    lam        = (ctx->cfg.lambda_um > 0.1f) ? ctx->cfg.lambda_um : 0.75f;
    float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    uint32_t bl_wf      = (uint32_t) ((ctx->npix_wf + 255ULL) / 256ULL);
    k_zwfs_build_epup<<<bl_wf, 256, 0, cs->stream>>>(
        d_pup, d_amp, d_wf, d_dm, cs->d_ncpa_opd, wf_to_opd, opd_to_rad, ctx->npix_wf,
        cs->d_epup);

    const cuComplex one  = make_cuComplex(1.0f, 0.0f);
    const cuComplex zero = make_cuComplex(0.0f, 0.0f);

    /* Row-major tmp[m x nwf] = fwd_U[m x nwf] * epup[nwf x nwf] => col-major tmp^T = epup^T*U^T */
    cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, m, nwf, &one, cs->d_epup, nwf,
                cs->d_fwd_U, nwf, &zero, cs->d_tmp, nwf);
    /* Row-major efoc[m x m] = tmp[m x nwf] * fwd_V[nwf x m] => col-major efoc^T = V^T * tmp^T */
    cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, m, nwf, &one, cs->d_fwd_V, m, cs->d_tmp,
                nwf, &zero, cs->d_efoc, m);

    uint32_t bl_mft = (uint32_t) ((ctx->npix_mft + 255ULL) / 256ULL);
    k_zwfs_mul_mask<<<bl_mft, 256, 0, cs->stream>>>(cs->d_efoc, cs->d_mask_bab, ctx->npix_mft);

    /* Row-major tmp[nwf x m] = inv_U * efoc => col-major tmp^T = efoc^T * inv_U^T */
    cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, nwf, m, &one, cs->d_efoc, m, cs->d_inv_U,
                m, &zero, cs->d_tmp, m);
    /* Row-major epup += tmp[nwf x m] * inv_V[m x nwf] => col-major epup^T += inv_V^T * tmp^T */
    cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, m, &one, cs->d_inv_V, nwf,
                cs->d_tmp, m, &one, cs->d_epup, nwf);

    uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + 255ULL) / 256ULL);
    k_zwfs_bin_cam<<<bl_cam, 256, 0, cs->stream>>>(cs->d_epup, ctx->cfg.nwf, ctx->cfg.ncam,
                                                   cs->d_cam_int);

    if (ctx->subframe_cnt >= 1u)
    {
        k_zwfs_add_accum<<<bl_cam, 256, 0, cs->stream>>>(
            cs->d_cam_int, ctx->npix_cam, cs->d_accum_int);
    }
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
