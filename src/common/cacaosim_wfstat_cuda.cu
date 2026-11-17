// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfstat_cuda.cu
 * @brief   CUDA backend for the shared pupil wavefront statistics engine (cacaosim_wfstat_core)
 *
 * Per frame and per optical plane k, on the producer's CUDA stream (3 launches for all planes):
 *   1. k_pist : weighted piston  p_k = sum_i w_i * 1000 * opd_k[i]           (1 block / plane)
 *   2. k_pix  : v = 1000 * opd_k[i] - p_k inside the mask; sum += v, sum2 += v^2, tmp = v
 *   3. k_stat : svar = sum_i w_i v^2, min/max(v); acc[srms] += sqrt(svar), acc[svar] += svar
 * All accumulators are FP64 so results match the CPU path of cacaosim_wfstat_core.c. Host
 * transfers happen only on sample ticks (maps + per-frame scalars) and at finalization.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define WFG_MAX_LOC 4
#define WFG_RED_THREADS 1024

typedef struct
{
    const float *opd[WFG_MAX_LOC];
    double      *sum[WFG_MAX_LOC];
    double      *sum2[WFG_MAX_LOC];
    float       *tmp[WFG_MAX_LOC];
} wfg_ptrs_t;

typedef struct
{
    uint64_t     npix;
    uint32_t     nloc;
    cudaStream_t stream;
    double      *d_w;
    double      *d_frame; /* [nloc * 4]: pist, svar, min, max of the latest frame */
    double      *d_acc;   /* [nloc * 2]: sum srms, sum svar */
    wfg_ptrs_t   p;
} wfg_ctx_t;

__device__ static double block_reduce_sum(double v, double *sh)
{
    sh[threadIdx.x] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    double r = sh[0];
    __syncthreads();
    return r;
}

__global__ void k_wfg_pist(const double *w, wfg_ptrs_t P, uint64_t npix, double *frame)
{
    __shared__ double sh[WFG_RED_THREADS];
    const float      *o = P.opd[blockIdx.x];
    double            s = 0.0;
    for (uint64_t i = threadIdx.x; i < npix; i += blockDim.x)
    {
        double wi = w[i];
        if (wi > 0.0)
        {
            s += wi * ((double) o[i] * 1000.0);
        }
    }
    double tot = block_reduce_sum(s, sh);
    if (threadIdx.x == 0)
    {
        frame[4 * blockIdx.x + 0] = tot;
    }
}

__global__ void k_wfg_pix(const double *w, wfg_ptrs_t P, uint64_t npix, const double *frame)
{
    uint32_t k = blockIdx.y;
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix)
    {
        return;
    }
    if (w[i] > 0.0)
    {
        double v = (double) P.opd[k][i] * 1000.0 - frame[4 * k + 0];
        P.sum[k][i] += v;
        P.sum2[k][i] += v * v;
        P.tmp[k][i] = (float) v;
    }
    else
    {
        P.tmp[k][i] = 0.0f;
    }
}

__global__ void k_wfg_stat(const double *w, wfg_ptrs_t P, uint64_t npix, double *frame,
                           double *acc)
{
    __shared__ double sh[WFG_RED_THREADS];
    uint32_t          k    = blockIdx.x;
    const float      *o    = P.opd[k];
    double            pist = frame[4 * k + 0];
    double            sv = 0.0, mn = 1.0e30, mx = -1.0e30;
    for (uint64_t i = threadIdx.x; i < npix; i += blockDim.x)
    {
        double wi = w[i];
        if (wi > 0.0)
        {
            double v = (double) o[i] * 1000.0 - pist;
            sv += wi * v * v;
            mn = fmin(mn, v);
            mx = fmax(mx, v);
        }
    }
    double svar = block_reduce_sum(sv, sh);

    sh[threadIdx.x] = mn;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sh[threadIdx.x] = fmin(sh[threadIdx.x], sh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    double vmin = sh[0];
    __syncthreads();

    sh[threadIdx.x] = mx;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    double vmax = sh[0];

    if (threadIdx.x == 0)
    {
        frame[4 * k + 1] = svar;
        frame[4 * k + 2] = vmin;
        frame[4 * k + 3] = vmax;
        acc[2 * k + 0] += sqrt(svar > 0.0 ? svar : 0.0);
        acc[2 * k + 1] += svar;
    }
}

static void wfg_release(wfg_ctx_t *g)
{
    cudaFree(g->d_w);
    cudaFree(g->d_frame);
    cudaFree(g->d_acc);
    for (uint32_t k = 0; k < WFG_MAX_LOC; k++)
    {
        cudaFree(g->p.sum[k]);
        cudaFree(g->p.sum2[k]);
        cudaFree(g->p.tmp[k]);
    }
    free(g);
}

extern "C" void *cacaosim_wfstat_gpu_create(
    const double *h_weights,
    uint64_t      npix,
    uint32_t      nloc,
    void         *cuda_stream)
{
    if (h_weights == NULL || npix == 0 || nloc == 0 || nloc > WFG_MAX_LOC)
    {
        return NULL;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) calloc(1, sizeof(wfg_ctx_t));
    if (g == NULL)
    {
        return NULL;
    }
    g->npix   = npix;
    g->nloc   = nloc;
    g->stream = (cudaStream_t) cuda_stream;

    int ok = (cudaMalloc((void **) &g->d_w, npix * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_frame, 4 * nloc * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_acc, 2 * nloc * sizeof(double)) == cudaSuccess);
    for (uint32_t k = 0; ok && k < nloc; k++)
    {
        ok = (cudaMalloc((void **) &g->p.sum[k], npix * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->p.sum2[k], npix * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->p.tmp[k], npix * sizeof(float)) == cudaSuccess);
        if (ok)
        {
            ok = (cudaMemsetAsync(g->p.sum[k], 0, npix * sizeof(double), g->stream) ==
                      cudaSuccess &&
                  cudaMemsetAsync(g->p.sum2[k], 0, npix * sizeof(double), g->stream) ==
                      cudaSuccess);
        }
    }
    if (ok)
    {
        ok = (cudaMemcpyAsync(g->d_w, h_weights, npix * sizeof(double), cudaMemcpyHostToDevice,
                              g->stream) == cudaSuccess &&
              cudaMemsetAsync(g->d_acc, 0, 2 * nloc * sizeof(double), g->stream) == cudaSuccess &&
              cudaStreamSynchronize(g->stream) == cudaSuccess);
    }
    if (!ok)
    {
        wfg_release(g);
        return NULL;
    }
    return g;
}

extern "C" void cacaosim_wfstat_gpu_free(void *gpu)
{
    if (gpu != NULL)
    {
        wfg_ctx_t *g = (wfg_ctx_t *) gpu;
        cudaStreamSynchronize(g->stream);
        wfg_release(g);
    }
}

extern "C" void cacaosim_wfstat_gpu_accum(void *gpu, const float *const d_opd[])
{
    if (gpu == NULL || d_opd == NULL)
    {
        return;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) gpu;
    for (uint32_t k = 0; k < g->nloc; k++)
    {
        g->p.opd[k] = d_opd[k];
    }
    dim3 grid_pix((unsigned) ((g->npix + 255) / 256), g->nloc);
    k_wfg_pist<<<g->nloc, WFG_RED_THREADS, 0, g->stream>>>(g->d_w, g->p, g->npix, g->d_frame);
    k_wfg_pix<<<grid_pix, 256, 0, g->stream>>>(g->d_w, g->p, g->npix, g->d_frame);
    k_wfg_stat<<<g->nloc, WFG_RED_THREADS, 0, g->stream>>>(g->d_w, g->p, g->npix, g->d_frame,
                                                           g->d_acc);
}

extern "C" int cacaosim_wfstat_gpu_fetch_sample(void *gpu, float *const h_maps[],
                                                double *h_frame)
{
    if (gpu == NULL)
    {
        return -1;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) gpu;
    for (uint32_t k = 0; k < g->nloc; k++)
    {
        cudaMemcpyAsync(h_maps[k], g->p.tmp[k], g->npix * sizeof(float), cudaMemcpyDeviceToHost,
                        g->stream);
    }
    cudaMemcpyAsync(h_frame, g->d_frame, 4 * g->nloc * sizeof(double), cudaMemcpyDeviceToHost,
                    g->stream);
    return (cudaStreamSynchronize(g->stream) == cudaSuccess) ? 0 : -1;
}

extern "C" int cacaosim_wfstat_gpu_fetch_sums(
    void         *gpu,
    double *const h_sum[],
    double *const h_sum2[],
    double       *h_acc)
{
    if (gpu == NULL)
    {
        return -1;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) gpu;
    for (uint32_t k = 0; k < g->nloc; k++)
    {
        cudaMemcpyAsync(h_sum[k], g->p.sum[k], g->npix * sizeof(double), cudaMemcpyDeviceToHost,
                        g->stream);
        cudaMemcpyAsync(h_sum2[k], g->p.sum2[k], g->npix * sizeof(double),
                        cudaMemcpyDeviceToHost, g->stream);
    }
    cudaMemcpyAsync(h_acc, g->d_acc, 2 * g->nloc * sizeof(double), cudaMemcpyDeviceToHost,
                    g->stream);
    return (cudaStreamSynchronize(g->stream) == cudaSuccess) ? 0 : -1;
}
