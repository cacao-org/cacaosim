// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm_cuda.cu
 * @brief   CUDA 16-tap gather-stencil kernel for sub-5-microsecond 2D DM OPD synthesis
 *
 * Architectural Role:
 *   Implements the GPU device memory lifecycle (`cacaosim_dm_cuda_alloc`,
 *   `cacaosim_dm_cuda_free`) and per-frame CUDA OPD synthesis launcher
 *   (`cacaosim_dm_cuda_render` + `kernel_dm_gather_opd`) backing `cacaosim_dm.h`.
 *
 * Design Rationale & Motivation:
 *   - Minimal H2D payload (`act_pos` only): Rather than pushing 2D wavefront maps
 *     across PCIe or running tiny sequential ring-buffer updates on GPU, the host
 *     uploads only the `nact_total`-element float actuator displacement vector
 *     (e.g., `16x16 = 1 KB`, `64x64 = 16 KB`) via `cudaMemcpyAsync` on a
 *     non-blocking stream, completing in `~1..2 us`.
 *   - Coalesced gather + read-only texture/L1 broadcast (`__ldg`): Each GPU thread
 *     computes one output pixel's OPD by unrolling all 16 stencil taps. Adjacent
 *     threads in a 32-thread warp share nearly identical 4x4 actuator neighborhoods,
 *     so `__ldg(&act_pos[a])` hits the read-only data cache with near-100% locality
 *     and zero atomic write contention.
 *   - Asynchronous GPU pipelining: When `host_opd_out == NULL` (as in `cacaosim.htsim`),
 *     `cacaosim_dm_cuda_render()` enqueues H2D + kernel onto `ctx->cuda_stream`
 *     and returns immediately without synchronizing, letting downstream GPU WFS
 *     and coronagraph stages consume `ctx->d_opd_out` directly in VRAM.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dm/cacaosim_dm.h"

/**
 * kernel_dm_gather_opd() - Evaluate 16-tap gather stencil per wavefront pixel on GPU
 * @act_pos:     Device actuator mechanical position vector `[nact_total]` in `[um]`
 * @stencil_idx: Device gather index table `[npix * 16]` mapping taps to `[0, nact_total)`
 * @stencil_w:   Device gather weight table `[npix * 16]` in `[OPD_um / stroke_um]`
 * @npix:        Total 2D wavefront pixel count `nwf * nwf`
 * @opd_out:     Device output 2D reflective OPD buffer `[npix]` in microns `[um]`
 *
 * Role:
 *   Hot-path CUDA `__global__` kernel launched once per DM step by
 *   `cacaosim_dm_cuda_render()` over a 1D grid of 256-thread blocks.
 *
 * Rationale:
 *   Full `#pragma unroll` of `CACAOSIM_DM_STENCIL` (`16`) keeps the accumulator
 *   `acc` in a register and overlaps independent stencil memory loads, while
 *   `__ldg()` routes actuator position gathers through the read-only cache.
 */
__global__ void kernel_dm_gather_opd(
    const float *__restrict__   act_pos,
    const int32_t *__restrict__ stencil_idx,
    const float *__restrict__   stencil_w,
    uint64_t                    npix,
    float *__restrict__         opd_out)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix)
    {
        return;
    }

    uint64_t base = p * CACAOSIM_DM_STENCIL;
    float    acc  = 0.0f;

#pragma unroll
    for (uint32_t k = 0; k < CACAOSIM_DM_STENCIL; k++)
    {
        int32_t a = stencil_idx[base + k];
        float   w = stencil_w[base + k];
        acc      += w * __ldg(&act_pos[a]);
    }
    opd_out[p] = acc;
}

/**
 * cacaosim_dm_cuda_alloc() - Allocate CUDA device buffers and upload static 16-tap stencil
 * @ctx:        DM context with precomputed host `stencil_idx` and `stencil_w` arrays
 * @ext_stream: Optional external `cudaStream_t` (`void *`) to borrow; NULL creates a private
 *              `cudaStreamNonBlocking` stream
 *
 * Role:
 *   Cold-path GPU resource allocator called from `cacaosim_dm_init()`.
 *
 * Rationale:
 *   Allocates `d_act_pos`, `d_stencil_idx`, `d_stencil_w`, and `d_opd_out` up
 *   front, zeroes dynamic state, uploads the static stencil tables asynchronously,
 *   and synchronizes `st` once so the context is immediately ready for real-time
 *   execution with zero runtime `cudaMalloc` calls. Rolls back cleanly via
 *   `cacaosim_dm_cuda_free()` if any allocation fails.
 *
 * Return: `0` on success, `-1` on CUDA stream or memory allocation failure.
 */
extern "C" int cacaosim_dm_cuda_alloc(
    cacaosim_dm_ctx_t *ctx,
    void              *ext_stream)
{
    size_t nact_bytes = (size_t) ctx->nact_total * sizeof(float);
    size_t npix_bytes = (size_t) ctx->npix_wf * sizeof(float);
    size_t idx_bytes  = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL * sizeof(int32_t);
    size_t w_bytes    = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL * sizeof(float);

    /* Borrow external pipeline stream if supplied, else create an owned non-blocking stream */
    if (ext_stream != NULL)
    {
        ctx->cuda_stream = ext_stream;
        ctx->owns_stream = 0;
    }
    else
    {
        cudaStream_t st = NULL;
        if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
        {
            return -1;
        }
        ctx->cuda_stream = (void *) st;
        ctx->owns_stream = 1;
    }

    if (cudaMalloc((void **) &ctx->d_act_pos, nact_bytes) != cudaSuccess ||
        cudaMalloc((void **) &ctx->d_stencil_idx, idx_bytes) != cudaSuccess ||
        cudaMalloc((void **) &ctx->d_stencil_w, w_bytes) != cudaSuccess ||
        cudaMalloc((void **) &ctx->d_opd_out, npix_bytes) != cudaSuccess)
    {
        cacaosim_dm_cuda_free(ctx);
        return -1;
    }

    /* Zero device buffers and upload immutable 16-tap stencil index/weight tables once */
    cudaStream_t st = (cudaStream_t) ctx->cuda_stream;
    cudaMemsetAsync(ctx->d_act_pos, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_opd_out, 0, npix_bytes, st);
    cudaMemcpyAsync(ctx->d_stencil_idx, ctx->stencil_idx, idx_bytes, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(ctx->d_stencil_w, ctx->stencil_w, w_bytes, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
    return 0;
}

/**
 * cacaosim_dm_cuda_free() - Release CUDA device buffers and owned stream in a DM context
 * @ctx: DM context whose GPU allocations should be freed
 *
 * Role:
 *   Cold-path GPU teardown routine invoked by `cacaosim_dm_free()` and error paths.
 *
 * Rationale:
 *   Guards every pointer individually and checks `owns_stream` before calling
 *   `cudaStreamDestroy()` so borrowed `htsim` streams are never invalidated.
 */
extern "C" void cacaosim_dm_cuda_free(cacaosim_dm_ctx_t *ctx)
{
    if (ctx->d_act_pos)
    {
        cudaFree(ctx->d_act_pos);
        ctx->d_act_pos = NULL;
    }
    if (ctx->d_stencil_idx)
    {
        cudaFree(ctx->d_stencil_idx);
        ctx->d_stencil_idx = NULL;
    }
    if (ctx->d_stencil_w)
    {
        cudaFree(ctx->d_stencil_w);
        ctx->d_stencil_w = NULL;
    }
    if (ctx->d_opd_out)
    {
        cudaFree(ctx->d_opd_out);
        ctx->d_opd_out = NULL;
    }
    if (ctx->owns_stream && ctx->cuda_stream)
    {
        cudaStreamDestroy((cudaStream_t) ctx->cuda_stream);
    }
    ctx->cuda_stream = NULL;
    ctx->owns_stream = 0;
}

/**
 * cacaosim_dm_cuda_render() - Asynchronously upload `act_pos_render` and render 2D DM OPD on GPU
 * @ctx:          Active CUDA-enabled DM context with updated `ctx->act_pos_render`
 * @host_opd_out: Optional host destination buffer `[npix_wf]` [um]; NULL skips D2H & sync
 *
 * Role:
 *   Hot-path CUDA execution wrapper called per frame from `cacaosim_dm_step()`.
 *
 * Rationale:
 *   Skips H2D upload and `kernel_dm_gather_opd` altogether when `ctx->opd_valid` is set and the
 *   rendered actuator stroke vector has settled (`max |act_pos_render - prev_pos_render| <= 1e-7`),
 *   reusing `ctx->d_opd_out` in VRAM. Only performs a D2H copy and `cudaStreamSynchronize()` when
 *   a host SHM buffer (`host_opd_out != NULL`) requires immediate CPU visibility.
 */
extern "C" void cacaosim_dm_cuda_render(
    cacaosim_dm_ctx_t *ctx,
    float             *host_opd_out)
{
    cudaStream_t st         = (cudaStream_t) ctx->cuda_stream;
    const float *pos_ren    = ctx->act_pos_render;
    uint32_t     nact       = ctx->nact_total;
    int          need_k     = !ctx->opd_valid;

    if (!need_k)
    {
        float max_d = 0.0f;
        for (uint32_t a = 0; a < nact; a++)
        {
            float d = fabsf(pos_ren[a] - ctx->prev_pos_render[a]);
            if (d > max_d)
            {
                max_d = d;
            }
        }
        if (max_d > 1.0e-7f)
        {
            need_k = 1;
        }
    }

    if (need_k)
    {
        size_t   nact_bytes = (size_t) nact * sizeof(float);
        uint32_t threads    = 256u;
        uint32_t blocks     = (uint32_t) ((ctx->npix_wf + threads - 1ULL) / threads);

        memcpy(ctx->prev_pos_render, pos_ren, nact_bytes);
        cudaMemcpyAsync(ctx->d_act_pos, pos_ren, nact_bytes, cudaMemcpyHostToDevice, st);
        kernel_dm_gather_opd<<<blocks, threads, 0, st>>>(
            ctx->d_act_pos, ctx->d_stencil_idx, ctx->d_stencil_w, ctx->npix_wf, ctx->d_opd_out);
        ctx->opd_valid = 1;
    }

    if (host_opd_out != NULL)
    {
        cudaMemcpyAsync(host_opd_out, ctx->d_opd_out, (size_t) ctx->npix_wf * sizeof(float),
                        cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
}
