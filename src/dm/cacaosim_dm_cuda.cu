// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm_cuda.cu
 * @brief   CUDA 16-tap gather-stencil kernel for sub-5-microsecond 2D DM OPD synthesis
 */

#include <cuda_runtime.h>
#include <stdint.h>

#include "dm/cacaosim_dm.h"

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

extern "C" int cacaosim_dm_cuda_alloc(
    cacaosim_dm_ctx_t *ctx,
    void              *ext_stream)
{
    size_t nact_bytes = (size_t) ctx->nact_total * sizeof(float);
    size_t npix_bytes = (size_t) ctx->npix_wf * sizeof(float);
    size_t idx_bytes  = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL * sizeof(int32_t);
    size_t w_bytes    = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL * sizeof(float);

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

    cudaStream_t st = (cudaStream_t) ctx->cuda_stream;
    cudaMemsetAsync(ctx->d_act_pos, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_opd_out, 0, npix_bytes, st);
    cudaMemcpyAsync(ctx->d_stencil_idx, ctx->stencil_idx, idx_bytes, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(ctx->d_stencil_w, ctx->stencil_w, w_bytes, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);
    return 0;
}

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

extern "C" void cacaosim_dm_cuda_render(
    cacaosim_dm_ctx_t *ctx,
    float             *host_opd_out)
{
    cudaStream_t st         = (cudaStream_t) ctx->cuda_stream;
    size_t       nact_bytes = (size_t) ctx->nact_total * sizeof(float);
    uint32_t     threads    = 256u;
    uint32_t     blocks     = (uint32_t) ((ctx->npix_wf + threads - 1ULL) / threads);

    cudaMemcpyAsync(ctx->d_act_pos, ctx->act_pos, nact_bytes, cudaMemcpyHostToDevice, st);

    kernel_dm_gather_opd<<<blocks, threads, 0, st>>>(
        ctx->d_act_pos, ctx->d_stencil_idx, ctx->d_stencil_w, ctx->npix_wf, ctx->d_opd_out);

    if (host_opd_out != NULL)
    {
        cudaMemcpyAsync(host_opd_out, ctx->d_opd_out, (size_t) ctx->npix_wf * sizeof(float),
                        cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
}
