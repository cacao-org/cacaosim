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

#include <cublas_v2.h>
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
    float       *ts[WFG_MAX_LOC];
    float       *ts_in[WFG_MAX_LOC];
    float       *ts_out[WFG_MAX_LOC];
} wfg_ptrs_t;

typedef struct
{
    uint64_t       npix;
    uint32_t       nloc;
    uint64_t       ts_cap;
    uint64_t       ts_len;
    cudaStream_t   stream;
    double        *d_w;
    double        *d_frame; /* [nloc * 4]: pist, svar, min, max of the latest frame */
    double        *d_acc;   /* [nloc * 2]: sum srms, sum svar */
    wfg_ptrs_t     p;

    int            ctrl_active;
    uint64_t       act_pix;
    uint32_t       ctrl_n_stg1;
    uint32_t       ctrl_n_tot;
    cublasHandle_t cublas;
    uint32_t      *d_act_idx;
    float         *d_act_sw;
    float         *d_ctrl_q;   /* [act_pix * ctrl_n_tot] col-major */
    float         *d_ctrl_v;   /* [act_pix * nloc] col-major */
    float         *d_ctrl_c;   /* [ctrl_n_tot * nloc] col-major */
    double        *d_acc_ctrl; /* [nloc * 4]: sum_in_rms, sum_in_var, sum_out_rms, sum_out_var */
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
                           double *acc, uint64_t ts_slot, uint64_t ts_cap)
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
        double srms      = sqrt(svar > 0.0 ? svar : 0.0);
        frame[4 * k + 1] = svar;
        frame[4 * k + 2] = vmin;
        frame[4 * k + 3] = vmax;
        acc[2 * k + 0] += srms;
        acc[2 * k + 1] += svar;
        if (ts_slot < ts_cap && P.ts[k] != NULL)
        {
            P.ts[k][ts_slot] = (float) srms;
        }
    }
}

__global__ void k_wfg_pack_ctrl(wfg_ptrs_t P, const uint32_t *act_idx, const float *act_sw,
                                uint64_t act_pix, float *d_v)
{
    uint32_t k = blockIdx.y;
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= act_pix)
    {
        return;
    }
    uint32_t idx         = act_idx[p];
    d_v[k * act_pix + p] = act_sw[p] * P.tmp[k][idx];
}

__global__ void k_wfg_ctrl_stat(const float *d_c, uint32_t n_stg1, uint32_t n_tot,
                                const double *frame, double *acc_ctrl, wfg_ptrs_t P,
                                uint64_t ts_slot, uint64_t ts_cap)
{
    __shared__ double sh[WFG_RED_THREADS];
    uint32_t          k     = blockIdx.x;
    uint32_t          m_lim = (k == 1 && n_stg1 > 0) ? n_stg1 : n_tot;
    const float      *ck    = d_c + (uint64_t) k * n_tot;
    double            s     = 0.0;
    for (uint32_t m = threadIdx.x; m < m_lim; m += blockDim.x)
    {
        double v = (double) ck[m];
        s += v * v;
    }
    double in_var = block_reduce_sum(s, sh);
    if (threadIdx.x == 0)
    {
        double tot_var = frame[4 * k + 1];
        if (tot_var < 0.0)
        {
            tot_var = 0.0;
        }
        if (in_var > tot_var)
        {
            in_var = tot_var;
        }
        double out_var = tot_var - in_var;
        double in_rms  = sqrt(in_var > 0.0 ? in_var : 0.0);
        double out_rms = sqrt(out_var > 0.0 ? out_var : 0.0);
        acc_ctrl[4 * k + 0] += in_rms;
        acc_ctrl[4 * k + 1] += in_var;
        acc_ctrl[4 * k + 2] += out_rms;
        acc_ctrl[4 * k + 3] += out_var;
        if (ts_slot < ts_cap)
        {
            if (P.ts_in[k] != NULL)
            {
                P.ts_in[k][ts_slot] = (float) in_rms;
            }
            if (P.ts_out[k] != NULL)
            {
                P.ts_out[k][ts_slot] = (float) out_rms;
            }
        }
    }
}

static void wfg_release_ctrl(wfg_ctx_t *g)
{
    if (g->cublas != NULL)
    {
        cublasDestroy(g->cublas);
        g->cublas = NULL;
    }
    cudaFree(g->d_act_idx);
    cudaFree(g->d_act_sw);
    cudaFree(g->d_ctrl_q);
    cudaFree(g->d_ctrl_v);
    cudaFree(g->d_ctrl_c);
    cudaFree(g->d_acc_ctrl);
    g->d_act_idx  = NULL;
    g->d_act_sw   = NULL;
    g->d_ctrl_q   = NULL;
    g->d_ctrl_v   = NULL;
    g->d_ctrl_c   = NULL;
    g->d_acc_ctrl = NULL;
    for (uint32_t k = 0; k < WFG_MAX_LOC; k++)
    {
        cudaFree(g->p.ts_in[k]);
        cudaFree(g->p.ts_out[k]);
        g->p.ts_in[k]  = NULL;
        g->p.ts_out[k] = NULL;
    }
    g->ctrl_active = 0;
}

static void wfg_release(wfg_ctx_t *g)
{
    wfg_release_ctrl(g);
    cudaFree(g->d_w);
    cudaFree(g->d_frame);
    cudaFree(g->d_acc);
    for (uint32_t k = 0; k < WFG_MAX_LOC; k++)
    {
        cudaFree(g->p.sum[k]);
        cudaFree(g->p.sum2[k]);
        cudaFree(g->p.tmp[k]);
        cudaFree(g->p.ts[k]);
    }
    free(g);
}

extern "C" void *cacaosim_wfstat_gpu_create(
    const double *h_weights,
    uint64_t      npix,
    uint32_t      nloc,
    uint64_t      ts_cap,
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
    g->ts_cap = ts_cap;
    g->ts_len = 0;
    g->stream = (cudaStream_t) cuda_stream;

    int ok = (cudaMalloc((void **) &g->d_w, npix * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_frame, 4 * nloc * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_acc, 2 * nloc * sizeof(double)) == cudaSuccess);
    for (uint32_t k = 0; ok && k < nloc; k++)
    {
        ok = (cudaMalloc((void **) &g->p.sum[k], npix * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->p.sum2[k], npix * sizeof(double)) == cudaSuccess &&
              cudaMalloc((void **) &g->p.tmp[k], npix * sizeof(float)) == cudaSuccess);
        if (ok && ts_cap > 0)
        {
            ok = (cudaMalloc((void **) &g->p.ts[k], ts_cap * sizeof(float)) == cudaSuccess);
        }
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
    k_wfg_stat<<<g->nloc, WFG_RED_THREADS, 0, g->stream>>>(
        g->d_w, g->p, g->npix, g->d_frame, g->d_acc, g->ts_len, g->ts_cap);
    if (g->ctrl_active)
    {
        dim3        grid_pack((unsigned) ((g->act_pix + 255) / 256), g->nloc);
        const float one  = 1.0f;
        const float zero = 0.0f;
        k_wfg_pack_ctrl<<<grid_pack, 256, 0, g->stream>>>(
            g->p, g->d_act_idx, g->d_act_sw, g->act_pix, g->d_ctrl_v);
        cublasSgemm(g->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int) g->ctrl_n_tot, (int) g->nloc,
                    (int) g->act_pix, &one, g->d_ctrl_q, (int) g->act_pix, g->d_ctrl_v,
                    (int) g->act_pix, &zero, g->d_ctrl_c, (int) g->ctrl_n_tot);
        k_wfg_ctrl_stat<<<g->nloc, WFG_RED_THREADS, 0, g->stream>>>(
            g->d_ctrl_c, g->ctrl_n_stg1, g->ctrl_n_tot, g->d_frame, g->d_acc_ctrl, g->p,
            g->ts_len, g->ts_cap);
    }
    if (g->ts_len < g->ts_cap)
    {
        g->ts_len++;
    }
}

extern "C" int cacaosim_wfstat_gpu_set_ctrl_basis(
    void           *gpu,
    const uint32_t *h_act_idx,
    const float    *h_act_sw,
    uint64_t        act_pix,
    const float    *h_q_act,
    uint32_t        n_stg1,
    uint32_t        n_tot)
{
    if (gpu == NULL || h_act_idx == NULL || h_act_sw == NULL || h_q_act == NULL ||
        act_pix == 0 || n_tot == 0 || n_stg1 > n_tot)
    {
        return -1;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) gpu;
    wfg_release_ctrl(g);
    g->act_pix     = act_pix;
    g->ctrl_n_stg1 = n_stg1;
    g->ctrl_n_tot  = n_tot;

    int ok = (cublasCreate(&g->cublas) == CUBLAS_STATUS_SUCCESS &&
              cublasSetStream(g->cublas, g->stream) == CUBLAS_STATUS_SUCCESS &&
              cudaMalloc((void **) &g->d_act_idx, act_pix * sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_act_sw, act_pix * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_ctrl_q, act_pix * n_tot * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_ctrl_v, act_pix * g->nloc * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void **) &g->d_ctrl_c, (uint64_t) n_tot * g->nloc * sizeof(float)) ==
                  cudaSuccess &&
              cudaMalloc((void **) &g->d_acc_ctrl, 4 * g->nloc * sizeof(double)) == cudaSuccess);
    for (uint32_t k = 0; ok && k < g->nloc; k++)
    {
        if (g->ts_cap > 0)
        {
            ok = (cudaMalloc((void **) &g->p.ts_in[k], g->ts_cap * sizeof(float)) == cudaSuccess &&
                  cudaMalloc((void **) &g->p.ts_out[k], g->ts_cap * sizeof(float)) == cudaSuccess);
        }
    }
    if (ok)
    {
        ok = (cudaMemcpyAsync(g->d_act_idx, h_act_idx, act_pix * sizeof(uint32_t),
                              cudaMemcpyHostToDevice, g->stream) == cudaSuccess &&
              cudaMemcpyAsync(g->d_act_sw, h_act_sw, act_pix * sizeof(float),
                              cudaMemcpyHostToDevice, g->stream) == cudaSuccess &&
              cudaMemcpyAsync(g->d_ctrl_q, h_q_act, act_pix * n_tot * sizeof(float),
                              cudaMemcpyHostToDevice, g->stream) == cudaSuccess &&
              cudaMemsetAsync(g->d_acc_ctrl, 0, 4 * g->nloc * sizeof(double), g->stream) ==
                  cudaSuccess &&
              cudaStreamSynchronize(g->stream) == cudaSuccess);
    }
    if (!ok)
    {
        wfg_release_ctrl(g);
        return -1;
    }
    g->ctrl_active = 1;
    return 0;
}

extern "C" int cacaosim_wfstat_gpu_fetch_ctrl(
    void         *gpu,
    double       *h_acc_ctrl,
    float *const  h_ts_in[],
    float *const  h_ts_out[])
{
    if (gpu == NULL)
    {
        return -1;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) gpu;
    if (!g->ctrl_active)
    {
        return 0;
    }
    if (h_acc_ctrl != NULL)
    {
        cudaMemcpyAsync(h_acc_ctrl, g->d_acc_ctrl, 4 * g->nloc * sizeof(double),
                        cudaMemcpyDeviceToHost, g->stream);
    }
    for (uint32_t k = 0; k < g->nloc; k++)
    {
        if (h_ts_in != NULL && h_ts_in[k] != NULL && g->p.ts_in[k] != NULL && g->ts_len > 0)
        {
            cudaMemcpyAsync(h_ts_in[k], g->p.ts_in[k], g->ts_len * sizeof(float),
                            cudaMemcpyDeviceToHost, g->stream);
        }
        if (h_ts_out != NULL && h_ts_out[k] != NULL && g->p.ts_out[k] != NULL && g->ts_len > 0)
        {
            cudaMemcpyAsync(h_ts_out[k], g->p.ts_out[k], g->ts_len * sizeof(float),
                            cudaMemcpyDeviceToHost, g->stream);
        }
    }
    return (cudaStreamSynchronize(g->stream) == cudaSuccess) ? 0 : -1;
}

extern "C" int cacaosim_wfstat_gpu_fetch_sample(void *gpu, float *const h_maps[],
                                                double *h_frame)
{
    if (gpu == NULL)
    {
        return -1;
    }
    wfg_ctx_t *g = (wfg_ctx_t *) gpu;
    if (h_maps != NULL)
    {
        for (uint32_t k = 0; k < g->nloc; k++)
        {
            if (h_maps[k] != NULL)
            {
                cudaMemcpyAsync(h_maps[k], g->p.tmp[k], g->npix * sizeof(float),
                                cudaMemcpyDeviceToHost, g->stream);
            }
        }
    }
    if (h_frame != NULL)
    {
        cudaMemcpyAsync(h_frame, g->d_frame, 4 * g->nloc * sizeof(double), cudaMemcpyDeviceToHost,
                        g->stream);
    }
    return (cudaStreamSynchronize(g->stream) == cudaSuccess) ? 0 : -1;
}

extern "C" int cacaosim_wfstat_gpu_fetch_sums(
    void         *gpu,
    double *const h_sum[],
    double *const h_sum2[],
    double       *h_acc,
    float *const  h_ts_rms[],
    uint64_t     *out_ts_len)
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
        if (h_ts_rms != NULL && h_ts_rms[k] != NULL && g->p.ts[k] != NULL && g->ts_len > 0)
        {
            cudaMemcpyAsync(h_ts_rms[k], g->p.ts[k], g->ts_len * sizeof(float),
                            cudaMemcpyDeviceToHost, g->stream);
        }
    }
    cudaMemcpyAsync(h_acc, g->d_acc, 2 * g->nloc * sizeof(double), cudaMemcpyDeviceToHost,
                    g->stream);
    if (out_ts_len != NULL)
    {
        *out_ts_len = g->ts_len;
    }
    return (cudaStreamSynchronize(g->stream) == cudaSuccess) ? 0 : -1;
}
