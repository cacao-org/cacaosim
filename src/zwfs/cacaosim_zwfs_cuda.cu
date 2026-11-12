// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs_cuda.cu
 * @brief   cuBLAS Babinet Matrix-DFT Zernike phase-contrast WFS GPU backend
 */

#ifdef HAVE_CUDA

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "common/cacaosim_cuda.h"
#include "zwfs/cacaosim_zwfs.h"

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

    if (nwf == ncam)
    {
        cuComplex z  = epup[cid];
        cam_int[cid] = z.x * z.x + z.y * z.y;
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
                cuComplex z = epup[roff + bx];
                acc += z.x * z.x + z.y * z.y;
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
                cuComplex z = epup[(uint64_t) iy * nwf + (uint32_t) ix];
                acc += wx * wy * (z.x * z.x + z.y * z.y);
            }
        }
    }
    cam_int[cid] = acc;
}

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
