// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs_cuda.cu
 * @brief   Batched cuFFT Diffractive Shack-Hartmann WFS GPU backend
 *
 * Architectural Role in cacaosim:
 *   Implements the CUDA GPU accelerator (`ctx->cuda_priv`) for `cacaosim.shwfs`, executing massive
 *   parallel subaperture complex patch assembly, batched 2D Fraunhofer cuFFT propagation across all
 *   `n_sub * n_sub` lenslets, FFT-shifted focal spot extraction into the monolithic camera mosaic,
 *   multi-subframe (`navg`) accumulation, and Philox Poisson/Gaussian detector readout on-device.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Single Batched 2D `cufftPlanMany` Over All Subapertures:
 *      A high-order Shack-Hartmann WFS (e.g. `16x16 = 256` up to `64x64 = 4096` lenslets) requires
 *      hundreds or thousands of tiny (`16x16`..`32x32`) 2D FFTs per subframe. Launching separate
 *      cuFFT calls per lenslet would serialize on host launch overhead and leave GPU SMs idle.
 *      Packing all `nbatch = n_sub * n_sub` zero-padded, half-bin shift-centered subaperture tiles
 *      into a flat `[nbatch * nfft_sub * nfft_sub]` buffer (`d_patches`) lets `cufftPlanMany` fuse
 *      the entire lenslet array into a single high-occupancy GPU kernel launch, faithfully modeling
 *      diffraction, speckle breakup, elongation, and spider/obscuration edge vignetting.
 *   2. Coalesced Gather / Scatter Tiling Without Atomic Contention:
 *      `k_shwfs_build_patches` maps one thread per padded batch element so global writes to
 *      `d_patches` are 100% coalesced, while `k_shwfs_extract_spots` maps one thread per output
 *      detector pixel `cid in [0, ncam * ncam)` so global writes to `d_cam_int` are likewise 100%
 *      coalesced with zero atomic operations or shared-memory bank conflicts.
 */

#ifdef HAVE_CUDA

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>

#include "common/cacaosim_cuda.h"
#include "shwfs/cacaosim_shwfs.h"

/**
 * @brief Private GPU execution state (`ctx->cuda_priv`) for the Diffractive Shack-Hartmann WFS.
 *
 * Holds the CUDA execution stream, batched 2D `cufftHandle`, uploaded lenslet pupil partitions and
 * 1D half-bin centering phasor, contiguous subaperture complex batch buffer, camera accumulators,
 * and typed ADU readout buffers.
 */
typedef struct
{
    cudaStream_t    stream;      /**< Execution CUDA stream (private or shared `htsim` stream) */
    int             owns_stream; /**< 1 if `stream` was created here and must be destroyed */
    cufftHandle     plan_batch;  /**< Batched 2D C2C forward cuFFT plan (`nbatch` x `nfft_sub^2`) */
    int             has_plan;    /**< Non-zero once `plan_batch` has been created */

    float          *d_pup_amp;   /**< Static uploaded 2D pupil amplitude `[npix_wf]` */
    float          *d_dyn_pup;   /**< Staging buffer for dynamic host beam amplitude `[npix_wf]` */
    float          *d_ncpa_opd;  /**< Optional static WFS-arm NCPA OPD `[npix_wf]` [um] */
    float          *d_atm_amp;   /**< Staging buffer for host atmospheric amplitude `[npix_wf]` */
    float          *d_wf_in;     /**< Staging buffer for host wavefront/beam OPD `[npix_wf]` */
    float          *d_dm_opd;    /**< Staging buffer for host DM OPD `[npix_wf]` [um] */
    uint32_t       *d_sub_off;   /**< Uploaded subaperture start indices `[n_sub]` */
    uint32_t       *d_sub_len;   /**< Uploaded subaperture widths `[n_sub]` */
    cuFloatComplex *d_shift_1d;  /**< Uploaded 1D half-bin centering phasor `[nfft_sub]` */
    cuFloatComplex *d_patches;   /**< Batched lenslet complex planes `[nbatch * npix_sub_fft]` */
    float          *d_cam_int;   /**< Single-subframe stitched camera intensity `[npix_cam]` */
    float          *d_accum_int; /**< Multi-subframe (`navg > 1`) intensity accumulator */
    uint16_t       *d_out_u16;   /**< Device uint16 ADU readout buffer `[npix_cam]` */
    float          *d_out_f32;   /**< Device float32 readout buffer `[npix_cam]` */
} shwfs_cuda_state_t;

/**
 * k_shwfs_build_patches() - Assemble zero-padded, half-bin centered complex lenslet patches on GPU
 * @pup_amp:    Device 2D pupil/beam amplitude `[nwf * nwf]` (NULL -> 1.0)
 * @atm_amp:    Optional device 2D atmospheric amplitude `[nwf * nwf]` (NULL -> 1.0)
 * @wf_in:      Optional device 2D wavefront/beam OPD `[nwf * nwf]` (NULL -> 0.0)
 * @dm_opd:     Optional device 2D DM OPD `[nwf * nwf]` in microns, subtracted (NULL -> 0.0)
 * @ncpa_opd:   Optional device 2D static NCPA OPD `[nwf * nwf]` in microns, added (NULL -> 0.0)
 * @sub_off:    Device subaperture start offsets `[n_sub]`
 * @sub_len:    Device subaperture pixel lengths `[n_sub]`
 * @shift_1d:   Device 1D half-bin centering phasor `[nfft]`
 * @wf_to_opd:  Scale converting `@wf_in` to microns OPD
 * @opd_to_rad: Optical wavenumber `2 * pi / lambda_um` [rad / um]
 * @nwf:        Input pupil grid dimension [pixels]
 * @n_sub:      Number of lenslets per pupil side
 * @nfft:       Padded subaperture 2D FFT dimension (`nfft_sub`)
 * @nbatch:     Total number of lenslets (`n_sub * n_sub`)
 * @patches:    Output contiguous complex batch `[nbatch * nfft * nfft]`
 *
 * Role:
 *   First hot-loop GPU stage slicing the 2D pupil complex field into `nbatch` independent
 *   zero-padded subaperture FFT tiles in a single kernel launch.
 *
 * Rationale:
 *   Threads outside `[0..nx) x [0..ny)` or on dark pupil pixels write `(0, 0)` directly without
 *   needing a separate `cudaMemsetAsync` pass, while illuminated threads fuse `__sincosf` phasor
 *   synthesis with separable `shift_1d[lx] * shift_1d[ly]` spot centering in registers.
 */
__global__ static void k_shwfs_build_patches(
    const float *__restrict__          pup_amp,
    const float *__restrict__          atm_amp,
    const float *__restrict__          wf_in,
    const float *__restrict__          dm_opd,
    const float *__restrict__          ncpa_opd,
    const uint32_t *__restrict__       sub_off,
    const uint32_t *__restrict__       sub_len,
    const cuFloatComplex *__restrict__ shift_1d,
    float                              wf_to_opd,
    float                              opd_to_rad,
    uint32_t                           nwf,
    uint32_t                           n_sub,
    uint32_t                           nfft,
    uint32_t                           nbatch,
    cuFloatComplex *__restrict__       patches)
{
    uint64_t idx   = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sfft2 = (uint64_t) nfft * nfft;
    uint64_t total = (uint64_t) nbatch * sfft2;
    if (idx >= total)
    {
        return;
    }

    /* Decode flat batch index into lenslet (sx, sy) and local subaperture coordinate (lx, ly) */
    uint32_t b   = (uint32_t) (idx / sfft2);
    uint32_t rem = (uint32_t) (idx % sfft2);
    uint32_t lx  = rem % nfft;
    uint32_t ly  = rem / nfft;
    uint32_t sx  = b % n_sub;
    uint32_t sy  = b / n_sub;

    /* Zero-pad samples outside the subaperture's [nx x ny] pupil footprint */
    uint32_t nx = sub_len[sx];
    uint32_t ny = sub_len[sy];
    if (lx >= nx || ly >= ny)
    {
        patches[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    uint32_t px  = sub_off[sx] + lx;
    uint32_t py  = sub_off[sy] + ly;
    uint64_t p   = (uint64_t) py * nwf + px;
    float    amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
    if (atm_amp != NULL)
    {
        amp *= atm_amp[p];
    }
    if (amp <= 0.0f)
    {
        patches[idx] = make_cuFloatComplex(0.0f, 0.0f);
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
    float si  = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &si, &co);
    float pr = amp * co;
    float pi = amp * si;

    /* Apply separable 2D half-bin Fourier shift ramp to center flat-WF spots on quad-cells */
    cuFloatComplex sx_c = shift_1d[lx];
    cuFloatComplex sy_c = shift_1d[ly];
    float          sr   = sx_c.x * sy_c.x - sx_c.y * sy_c.y;
    float          sm   = sx_c.x * sy_c.y + sx_c.y * sy_c.x;

    patches[idx] = make_cuFloatComplex(pr * sr - pi * sm, pr * sm + pi * sr);
}

/**
 * k_shwfs_extract_spots() - FFT-shift, crop, and tile subaperture focal spots into camera mosaic
 * @patches:   Post-cuFFT complex focal subaperture batch `[nbatch * nfft * nfft]`
 * @n_sub:     Number of lenslets per side
 * @psub:      Extracted detector pixels per lenslet side (`pix_per_sub`)
 * @ncam:      Output square camera dimension (`n_sub * psub`)
 * @nfft:      Subaperture FFT grid dimension (`nfft_sub`)
 * @inv_nfft2: Parseval normalization factor `1.0f / (nfft * nfft)`
 * @cam_int:   Output stitched 2D camera optical intensity `[ncam * ncam]`
 *
 * Role:
 *   Final optical GPU stage converting unshifted complex subaperture DFTs into centered `psub x
 *   psub` intensity sub-windows tiled across `cam_int`.
 *
 * Rationale:
 *   Indexing threads by output camera pixel `cid` maps consecutive threads in a warp to adjacent
 *   columns `cx` of `cam_int` for coalesced float32 writes while wrapping signed focal offsets
 *   `(dx, dy) in [-psub/2, +psub/2)` modulo `nfft` in registers.
 */
__global__ static void k_shwfs_extract_spots(
    const cuFloatComplex *__restrict__ patches,
    uint32_t                           n_sub,
    uint32_t                           psub,
    uint32_t                           ncam,
    uint32_t                           nfft,
    float                              inv_nfft2,
    float *__restrict__                cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx     = cid % ncam;
    uint32_t cy     = cid / ncam;
    uint32_t sx     = cx / psub;
    uint32_t sy     = cy / psub;
    uint32_t lx     = cx % psub;
    uint32_t ly     = cy % psub;
    int32_t  half_p = (int32_t) (psub / 2u);
    int32_t  dx     = (int32_t) lx - half_p;
    int32_t  dy     = (int32_t) ly - half_p;
    uint32_t kx     = (uint32_t) ((dx % (int32_t) nfft + (int32_t) nfft) % (int32_t) nfft);
    uint32_t ky     = (uint32_t) ((dy % (int32_t) nfft + (int32_t) nfft) % (int32_t) nfft);
    uint32_t b      = sy * n_sub + sx;

    cuFloatComplex z = patches[((uint64_t) b * nfft + ky) * nfft + kx];
    cam_int[cid]     = inv_nfft2 * (z.x * z.x + z.y * z.y);
}

/**
 * k_shwfs_add_accum() - Accumulate single-subframe camera intensity into multi-subframe buffer
 * @cam_int:   Current subframe stitched camera intensity `[npix]`
 * @npix:      Total camera pixels (`ncam * ncam`)
 * @accum_int: In-place multi-subframe accumulator `[npix]`
 *
 * Role:
 *   Hot-loop `navg > 1` temporal intensity summation kernel.
 *
 * Rationale:
 *   Launched only when `subframe_cnt >= 1`, keeping single-subframe exposures (`navg == 1`) free of
 *   extra memory passes.
 */
__global__ static void k_shwfs_add_accum(
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
 * k_shwfs_scale_accum() - Divide accumulated multi-subframe intensity by `subframe_cnt`
 * @accum_int: Summed multi-subframe camera intensity `[npix]`
 * @inv_n:     Reciprocal subframe count (`1.0f / subframe_cnt`)
 * @npix:      Total camera pixels (`ncam * ncam`)
 * @cam_int:   Normalized output camera intensity `[npix]`
 *
 * Role:
 *   Normalizes temporal subframe sum prior to `cacaosim_cuda_detector_readout()`.
 *
 * Rationale:
 *   Ensures `det_live->flux_ph` represents total detected guide-star photons per exposure
 *   independent of `navg`.
 */
__global__ static void k_shwfs_scale_accum(
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
 * cacaosim_shwfs_cuda_init() - Allocate GPU buffers, upload lenslet bounds, and plan batched cuFFT
 * @ctx:        Host SHWFS context with geometry and `shift_1d_cplx` precomputed
 * @pup_amp:    Static host 2D pupil amplitude `[npix_wf]`
 * @ncpa_opd:   Optional static host 2D NCPA OPD `[npix_wf]` in microns (NULL -> none)
 * @ext_stream: Optional external `cudaStream_t` (NULL creates a private non-blocking stream)
 *
 * Role:
 *   Cold-path GPU backend constructor allocating VRAM workspaces, uploading static subaperture
 *   lookup tables, and planning the `nbatch` 2D C2C `cufftPlanMany` transform.
 *
 * Rationale:
 *   Binding `cufftSetStream` to `cs->stream` during initialization enables asynchronous multi-unit
 *   execution inside `cacaosim.htsim` without implicit default-stream synchronization barriers.
 *
 * Return: 0 on success, -1 on CUDA or cuFFT failure.
 */
extern "C" int cacaosim_shwfs_cuda_init(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *pup_amp,
    const float          *ncpa_opd,
    void                 *ext_stream)
{
    shwfs_cuda_state_t *cs = (shwfs_cuda_state_t *) calloc(1, sizeof(*cs));
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
            cacaosim_shwfs_cuda_free(ctx);
            return -1;
        }
        cs->owns_stream = 1;
    }

    size_t wf_bytes  = (size_t) ctx->npix_wf * sizeof(float);
    size_t sub_bytes = (size_t) ctx->cfg.n_sub * sizeof(uint32_t);
    size_t sh_bytes  = (size_t) ctx->nfft_sub * sizeof(cuFloatComplex);
    size_t bat_bytes = (size_t) ctx->nbatch * ctx->npix_sub_fft * sizeof(cuFloatComplex);
    size_t cam_f     = (size_t) ctx->npix_cam * sizeof(float);
    size_t cam_u     = (size_t) ctx->npix_cam * sizeof(uint16_t);

    if (cudaMalloc((void **) &cs->d_pup_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dyn_pup, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_atm_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_wf_in, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dm_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_sub_off, sub_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_sub_len, sub_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_shift_1d, sh_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_patches, bat_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_cam_int, cam_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_accum_int, cam_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_u16, cam_u) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_f32, cam_f) != cudaSuccess)
    {
        cacaosim_shwfs_cuda_free(ctx);
        return -1;
    }

    if (ncpa_opd != NULL)
    {
        if (cudaMalloc((void **) &cs->d_ncpa_opd, wf_bytes) != cudaSuccess)
        {
            cacaosim_shwfs_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_ncpa_opd, ncpa_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    }

    cudaMemcpyAsync(cs->d_pup_amp, pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_sub_off, ctx->sub_off, sub_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_sub_len, ctx->sub_len, sub_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_shift_1d, ctx->shift_1d_cplx, sh_bytes, cudaMemcpyHostToDevice,
                    cs->stream);

    int n[2]       = { (int) ctx->nfft_sub, (int) ctx->nfft_sub };
    int inembed[2] = { n[0], n[1] };
    int idist      = (int) ctx->npix_sub_fft;
    if (cufftPlanMany(&cs->plan_batch, 2, n, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C,
                      (int) ctx->nbatch) != CUFFT_SUCCESS ||
        cufftSetStream(cs->plan_batch, cs->stream) != CUFFT_SUCCESS)
    {
        cacaosim_shwfs_cuda_free(ctx);
        return -1;
    }
    cs->has_plan = 1;

    cudaStreamSynchronize(cs->stream);
    return 0;
}

/**
 * cacaosim_shwfs_cuda_free() - Release batched cuFFT plan, device buffers, and owned CUDA stream
 * @ctx: Shack-Hartmann WFS context whose `cuda_priv` state will be destroyed
 *
 * Role:
 *   Cold-path GPU teardown routine releasing all device memory and cuFFT state owned by `@ctx`.
 *
 * Rationale:
 *   Leaves borrowed `ext_stream` handles intact while destroying privately created streams.
 */
extern "C" void cacaosim_shwfs_cuda_free(cacaosim_shwfs_ctx_t *ctx)
{
    shwfs_cuda_state_t *cs = (shwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL)
    {
        return;
    }
    if (cs->has_plan)
    {
        cufftDestroy(cs->plan_batch);
    }
    cudaFree(cs->d_pup_amp);
    cudaFree(cs->d_dyn_pup);
    cudaFree(cs->d_ncpa_opd);
    cudaFree(cs->d_atm_amp);
    cudaFree(cs->d_wf_in);
    cudaFree(cs->d_dm_opd);
    cudaFree(cs->d_sub_off);
    cudaFree(cs->d_sub_len);
    cudaFree(cs->d_shift_1d);
    cudaFree(cs->d_patches);
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
 * cacaosim_shwfs_cuda_accum() - Run batched GPU lenslet patch synthesis, cuFFT, and spot tiling
 * @ctx:           Active SHWFS context (`use_cuda == 1`)
 * @dyn_pup_amp:   Optional dynamic host pupil/beam amplitude `[npix_wf]`
 * @atm_amp:       Optional host atmospheric amplitude `[npix_wf]`
 * @wf_in:         Optional host wavefront or beam OPD `[npix_wf]`
 * @dm_opd:        Optional host DM OPD `[npix_wf]` in microns
 * @d_pup_amp_ext: Optional resident device pupil/beam amplitude pointer `[npix_wf]`
 * @d_wf_opd_ext:  Optional resident device wavefront/beam OPD pointer `[npix_wf]`
 * @d_dm_opd_ext:  Optional resident device DM OPD pointer `[npix_wf]`
 * @wf_to_opd:     Scale converting `@wf_in` / `@d_wf_opd_ext` to microns OPD
 *
 * Role:
 *   Hot-loop GPU subframe propagator queuing optional H2D staging, `k_shwfs_build_patches`,
 *   in-place batched `cufftExecC2C`, `k_shwfs_extract_spots`, and lazy `navg > 1` accumulation
 *   asynchronously on `cs->stream`.
 *
 * Rationale:
 *   Supports zero-PCIe-copy device pointer forwarding in fused `htsim` benches while pipelining
 *   multiple `navg` subframes asynchronously without host synchronization until camera readout.
 */
extern "C" void cacaosim_shwfs_cuda_accum(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *dyn_pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *d_pup_amp_ext,
    const float          *d_wf_opd_ext,
    const float          *d_dm_opd_ext,
    float                 wf_to_opd)
{
    shwfs_cuda_state_t *cs       = (shwfs_cuda_state_t *) ctx->cuda_priv;
    size_t              wf_bytes = (size_t) ctx->npix_wf * sizeof(float);

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

    /* Lazily seed multi-subframe accumulator only when a second subframe arrives (navg > 1) */
    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    float    lam        = (ctx->cfg.lambda_um > 0.1f) ? ctx->cfg.lambda_um : 0.75f;
    float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    uint64_t total_pat  = (uint64_t) ctx->nbatch * ctx->npix_sub_fft;
    uint32_t bl_pat     = (uint32_t) ((total_pat + 255ULL) / 256ULL);

    k_shwfs_build_patches<<<bl_pat, 256, 0, cs->stream>>>(
        d_pup, d_amp, d_wf, d_dm, cs->d_ncpa_opd, cs->d_sub_off, cs->d_sub_len, cs->d_shift_1d,
        wf_to_opd, opd_to_rad, ctx->cfg.nwf, ctx->cfg.n_sub, ctx->nfft_sub, ctx->nbatch,
        cs->d_patches);

    cufftExecC2C(cs->plan_batch, cs->d_patches, cs->d_patches, CUFFT_FORWARD);

    uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + 255ULL) / 256ULL);
    k_shwfs_extract_spots<<<bl_cam, 256, 0, cs->stream>>>(
        cs->d_patches, ctx->cfg.n_sub, ctx->cfg.pix_per_sub, ctx->ncam, ctx->nfft_sub,
        1.0f / (float) ctx->npix_sub_fft, cs->d_cam_int);

    if (ctx->subframe_cnt >= 1u)
    {
        k_shwfs_add_accum<<<bl_cam, 256, 0, cs->stream>>>(
            cs->d_cam_int, ctx->npix_cam, cs->d_accum_int);
    }
}

/**
 * cacaosim_shwfs_cuda_readout() - Average GPU subframes, apply detector noise, and copy to host
 * @ctx:      Active SHWFS context (`use_cuda == 1`)
 * @det_live: Active detector readout parameters (non-NULL)
 * @out_u16:  Destination host `uint16_t` camera buffer `[npix_cam]` when `out_dtype == 0`
 * @out_f32:  Destination host `float` camera buffer `[npix_cam]` when `out_dtype != 0`
 *
 * Role:
 *   Completes the GPU WFS exposure via optional `1 / subframe_cnt` averaging, Philox Poisson +
 *   Gaussian readout quantization, and synchronized D2H transfer into output memory.
 *
 * Rationale:
 *   Single-point stream synchronization at the end of camera readout ensures host SHM buffers
 *   receive complete frames before `processinfo_update_output_stream()` posts semaphores.
 */
extern "C" void cacaosim_shwfs_cuda_readout(
    cacaosim_shwfs_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    shwfs_cuda_state_t *cs     = (shwfs_cuda_state_t *) ctx->cuda_priv;
    uint32_t            bl_cam = (uint32_t) ((ctx->npix_cam + 255ULL) / 256ULL);

    if (ctx->subframe_cnt > 1u)
    {
        k_shwfs_scale_accum<<<bl_cam, 256, 0, cs->stream>>>(
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
