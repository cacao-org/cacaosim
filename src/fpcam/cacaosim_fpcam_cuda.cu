// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam_cuda.cu
 * @brief   cuBLAS StridedBatched MFT + cuFFT focal plane camera GPU backend
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/cacaosim_cuda.h"
#include "fpcam/cacaosim_fpcam.h"

typedef struct
{
    cudaStream_t   stream;
    int            owns_stream;
    cublasHandle_t cublas;
    cufftHandle    fft_fwd;
    int            has_fft;

    float         *d_pup_amp;
    float         *d_dyn_pup;
    float         *d_ncpa_opd;
    float         *d_atm_amp;
    float         *d_wf_in;
    float         *d_dm_opd;
    float         *d_lam_um;

    cuComplex     *d_mft_U;
    cuComplex     *d_mft_V;
    cuComplex     *d_epup_batch;
    cuComplex     *d_mft_tmp;
    cuComplex     *d_efoc_batch;

    cufftComplex  *d_fft_in;
    cufftComplex  *d_fft_out;

    float         *d_accum_int;
    float         *d_exp_int;
    uint16_t      *d_out_u16;
    float         *d_out_f32;
} fpcam_cuda_priv_t;

__global__ static void k_fpcam_build_epup_batch(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ lam_um,
    float                     wf_to_opd,
    uint64_t                  npix_wf,
    int32_t                   nlam,
    cuComplex *__restrict__   epup_batch)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= nlam)
    {
        return;
    }

    float amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
    if (atm_amp != NULL)
    {
        amp *= atm_amp[p];
    }
    uint64_t out_idx = (uint64_t) c * npix_wf + p;
    if (amp <= 0.0f)
    {
        epup_batch[out_idx] = make_cuComplex(0.0f, 0.0f);
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

    float phi = opd * (CACAOSIM_TWO_PI_F / lam_um[c]);
    float s   = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &s, &co);
    epup_batch[out_idx] = make_cuComplex(amp * co, amp * s);
}

__global__ static void k_fpcam_accum_mft_batch(
    const cuComplex *__restrict__ efoc_batch,
    uint64_t                      npix_cam,
    int32_t                       nlam,
    float                         inv_nlam,
    float *__restrict__           accum_int)
{
    uint64_t idx = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix_cam)
    {
        return;
    }
    float sum = 0.0f;
    for (int32_t c = 0; c < nlam; c++)
    {
        cuComplex z = efoc_batch[(uint64_t) c * npix_cam + idx];
        sum += z.x * z.x + z.y * z.y;
    }
    accum_int[idx] += inv_nlam * sum;
}

__global__ static void k_fpcam_accum_fft_crop(
    const cufftComplex *__restrict__ fft_out,
    uint32_t                         nfft,
    uint32_t                         ncam,
    float                            scale_i,
    float *__restrict__              accum_int)
{
    uint32_t xf = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t yf = blockIdx.y * blockDim.y + threadIdx.y;
    if (xf >= ncam || yf >= ncam)
    {
        return;
    }
    int32_t      half_c = (int32_t) (ncam / 2u);
    int32_t      dx     = (int32_t) xf - half_c;
    int32_t      dy     = (int32_t) yf - half_c;
    uint32_t     kx     = (uint32_t) ((dx % (int32_t) nfft + (int32_t) nfft) % (int32_t) nfft);
    uint32_t     ky     = (uint32_t) ((dy % (int32_t) nfft + (int32_t) nfft) % (int32_t) nfft);
    cufftComplex z      = fft_out[(uint64_t) ky * nfft + kx];
    accum_int[(uint64_t) yf * ncam + xf] += scale_i * (z.x * z.x + z.y * z.y);
}

__global__ static void k_fpcam_scale_exposure(
    const float *__restrict__ accum_int,
    float *__restrict__       exp_int,
    uint64_t                  npix,
    float                     inv_sub)
{
    uint64_t idx = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npix)
    {
        exp_int[idx] = accum_int[idx] * inv_sub;
    }
}

extern "C" int cacaosim_fpcam_cuda_init(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *ncpa_opd,
    void                 *ext_stream)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) calloc(1, sizeof(*p));
    if (p == NULL)
    {
        return -1;
    }
    ctx->cuda_priv = p;

    if (ext_stream != NULL)
    {
        p->stream      = (cudaStream_t) ext_stream;
        p->owns_stream = 0;
    }
    else
    {
        if (cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
        p->owns_stream = 1;
    }

    const uint64_t npix_wf  = ctx->npix_wf;
    const uint64_t npix_cam = ctx->npix_cam;
    const int32_t  nlam     = ctx->eff_nlam;

    if (cudaMalloc((void **) &p->d_pup_amp, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_dyn_pup, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_atm_amp, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_wf_in, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_dm_opd, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_lam_um, (size_t) nlam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_accum_int, npix_cam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_exp_int, npix_cam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_out_u16, npix_cam * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_out_f32, npix_cam * sizeof(float)) != cudaSuccess)
    {
        cacaosim_fpcam_cuda_free(ctx);
        return -1;
    }

    cudaMemcpyAsync(p->d_pup_amp, pup_amp, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                    p->stream);
    cudaMemcpyAsync(p->d_lam_um, ctx->lam_um, (size_t) nlam * sizeof(float),
                    cudaMemcpyHostToDevice, p->stream);
    cudaMemsetAsync(p->d_accum_int, 0, npix_cam * sizeof(float), p->stream);

    if (ncpa_opd != NULL)
    {
        if (cudaMalloc((void **) &p->d_ncpa_opd, npix_wf * sizeof(float)) != cudaSuccess)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(p->d_ncpa_opd, ncpa_opd, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                        p->stream);
    }

    if (ctx->cfg.prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        if (cublasCreate(&p->cublas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(p->cublas, p->stream) != CUBLAS_STATUS_SUCCESS)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }

        size_t uv_bytes = (size_t) nlam * ctx->cfg.ncam * ctx->cfg.nwf * sizeof(cuComplex);
        size_t ep_bytes = (size_t) nlam * npix_wf * sizeof(cuComplex);
        size_t tm_bytes = (size_t) nlam * ctx->cfg.ncam * ctx->cfg.nwf * sizeof(cuComplex);
        size_t ef_bytes = (size_t) nlam * npix_cam * sizeof(cuComplex);

        if (cudaMalloc((void **) &p->d_mft_U, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &p->d_mft_V, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &p->d_epup_batch, ep_bytes) != cudaSuccess ||
            cudaMalloc((void **) &p->d_mft_tmp, tm_bytes) != cudaSuccess ||
            cudaMalloc((void **) &p->d_efoc_batch, ef_bytes) != cudaSuccess)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }

        cudaMemcpyAsync(p->d_mft_U, ctx->mft_U_cplx, uv_bytes, cudaMemcpyHostToDevice, p->stream);
        cudaMemcpyAsync(p->d_mft_V, ctx->mft_V_cplx, uv_bytes, cudaMemcpyHostToDevice, p->stream);
    }
    else
    {
        size_t fft_bytes = ctx->npix_fft * sizeof(cufftComplex);
        if (cudaMalloc((void **) &p->d_fft_in, fft_bytes) != cudaSuccess ||
            cudaMalloc((void **) &p->d_fft_out, fft_bytes) != cudaSuccess)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
        int n = (int) ctx->nfft;
        if (cufftPlan2d(&p->fft_fwd, n, n, CUFFT_C2C) != CUFFT_SUCCESS ||
            cufftSetStream(p->fft_fwd, p->stream) != CUFFT_SUCCESS)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
        p->has_fft = 1;
    }

    cudaStreamSynchronize(p->stream);
    return 0;
}

extern "C" void cacaosim_fpcam_cuda_free(cacaosim_fpcam_ctx_t *ctx)
{
    if (ctx == NULL || ctx->cuda_priv == NULL)
    {
        return;
    }
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    if (p->cublas)
    {
        cublasDestroy(p->cublas);
    }
    if (p->has_fft)
    {
        cufftDestroy(p->fft_fwd);
    }
    cudaFree(p->d_pup_amp);
    cudaFree(p->d_dyn_pup);
    cudaFree(p->d_ncpa_opd);
    cudaFree(p->d_atm_amp);
    cudaFree(p->d_wf_in);
    cudaFree(p->d_dm_opd);
    cudaFree(p->d_lam_um);
    cudaFree(p->d_mft_U);
    cudaFree(p->d_mft_V);
    cudaFree(p->d_epup_batch);
    cudaFree(p->d_mft_tmp);
    cudaFree(p->d_efoc_batch);
    cudaFree(p->d_fft_in);
    cudaFree(p->d_fft_out);
    cudaFree(p->d_accum_int);
    cudaFree(p->d_exp_int);
    cudaFree(p->d_out_u16);
    cudaFree(p->d_out_f32);
    if (p->owns_stream && p->stream)
    {
        cudaStreamDestroy(p->stream);
    }
    free(p);
    ctx->cuda_priv = NULL;
    ctx->use_cuda  = 0;
}

extern "C" void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    cudaMemsetAsync(p->d_accum_int, 0, ctx->npix_cam * sizeof(float), p->stream);
}

extern "C" void cacaosim_fpcam_cuda_accum(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *dyn_pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *d_pup_amp_ext,
    const float          *d_wf_opd_ext,
    const float          *d_dm_opd_ext,
    float                 wf_to_opd)
{
    fpcam_cuda_priv_t *p        = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    const uint64_t     npix_wf  = ctx->npix_wf;
    const uint64_t     npix_cam = ctx->npix_cam;
    const uint32_t     nwf      = ctx->cfg.nwf;
    const uint32_t     ncam     = ctx->cfg.ncam;
    const int32_t      nlam     = ctx->eff_nlam;

    const float *d_pup = p->d_pup_amp;
    const float *d_amp = NULL;
    const float *d_wf  = NULL;
    const float *d_dm  = NULL;

    if (d_pup_amp_ext != NULL)
    {
        d_pup = d_pup_amp_ext;
    }
    else if (dyn_pup_amp != NULL)
    {
        cudaMemcpyAsync(p->d_dyn_pup, dyn_pup_amp, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                        p->stream);
        d_pup = p->d_dyn_pup;
    }

    if (atm_amp != NULL)
    {
        cudaMemcpyAsync(p->d_atm_amp, atm_amp, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                        p->stream);
        d_amp = p->d_atm_amp;
    }
    if (d_wf_opd_ext != NULL)
    {
        d_wf = d_wf_opd_ext;
    }
    else if (wf_in != NULL && wf_to_opd != 0.0f)
    {
        cudaMemcpyAsync(p->d_wf_in, wf_in, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                        p->stream);
        d_wf = p->d_wf_in;
    }
    if (d_dm_opd_ext != NULL)
    {
        d_dm = d_dm_opd_ext;
    }
    else if (dm_opd != NULL)
    {
        cudaMemcpyAsync(p->d_dm_opd, dm_opd, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                        p->stream);
        d_dm = p->d_dm_opd;
    }

    if (ctx->cfg.prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        dim3 blk(256, 1, 1);
        dim3 grd((unsigned int) ((npix_wf + 255u) / 256u), (unsigned int) nlam, 1);
        k_fpcam_build_epup_batch<<<grd, blk, 0, p->stream>>>(
            d_pup, d_amp, d_wf, d_dm, p->d_ncpa_opd, p->d_lam_um, wf_to_opd, npix_wf, nlam,
            p->d_epup_batch);

        const cuComplex one       = make_cuComplex(1.0f, 0.0f);
        const cuComplex zero      = make_cuComplex(0.0f, 0.0f);
        long long int   stride_ep = (long long int) npix_wf;
        long long int   stride_uv = (long long int) ncam * nwf;
        long long int   stride_ef = (long long int) npix_cam;

        /* Row-major T[ncam x nwf] = U * Epup -> col-major T^T = Epup^T * U^T */
        cublasCgemmStridedBatched(p->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int) nwf, (int) ncam,
                                  (int) nwf, &one, p->d_epup_batch, (int) nwf, stride_ep,
                                  p->d_mft_U, (int) nwf, stride_uv, &zero, p->d_mft_tmp, (int) nwf,
                                  stride_uv, nlam);

        /* Row-major Efoc[ncam x ncam] = T * V -> col-major Efoc^T = V^T * T^T */
        cublasCgemmStridedBatched(p->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int) ncam, (int) ncam,
                                  (int) nwf, &one, p->d_mft_V, (int) ncam, stride_uv, p->d_mft_tmp,
                                  (int) nwf, stride_uv, &zero, p->d_efoc_batch, (int) ncam,
                                  stride_ef, nlam);

        uint32_t g1d = (uint32_t) ((npix_cam + 255u) / 256u);
        k_fpcam_accum_mft_batch<<<g1d, 256, 0, p->stream>>>(p->d_efoc_batch, npix_cam, nlam,
                                                            1.0f / (float) nlam, p->d_accum_int);
    }
    else
    {
        const float scale_i = 1.0f / ((float) nlam * ctx->pup_sum_amp * ctx->pup_sum_amp);
        dim3        b2d(16, 16);
        dim3        g2d((ncam + 15u) / 16u, (ncam + 15u) / 16u);

        for (int32_t c = 0; c < nlam; c++)
        {
            float opd_to_rad = CACAOSIM_TWO_PI_F / ctx->lam_um[c];
            cacaosim_cuda_build_pupil_padded(d_pup, d_amp, d_wf, d_dm, p->d_ncpa_opd,
                                             wf_to_opd, opd_to_rad, nwf, ctx->nfft, p->d_fft_in,
                                             p->stream);
            cufftExecC2C(p->fft_fwd, p->d_fft_in, p->d_fft_out, CUFFT_FORWARD);
            k_fpcam_accum_fft_crop<<<g2d, b2d, 0, p->stream>>>(p->d_fft_out, ctx->nfft, ncam,
                                                               scale_i, p->d_accum_int);
        }
    }
}

extern "C" void cacaosim_fpcam_cuda_readout(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    float                    *host_exp_int,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    fpcam_cuda_priv_t *p    = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    const uint64_t     npix = ctx->npix_cam;
    const float        invs = 1.0f / (float) ctx->subframe_cnt;

    uint32_t g1d = (uint32_t) ((npix + 255u) / 256u);
    k_fpcam_scale_exposure<<<g1d, 256, 0, p->stream>>>(p->d_accum_int, p->d_exp_int, npix, invs);

    cacaosim_cuda_detector_readout(det_live, p->d_exp_int, npix, ctx->inf_flux_norm,
                                   ctx->cfg.rng_seed, ctx->frame_seq, p->d_out_u16, p->d_out_f32,
                                   p->stream);

    if (host_exp_int != NULL)
    {
        cudaMemcpyAsync(host_exp_int, p->d_exp_int, npix * sizeof(float), cudaMemcpyDeviceToHost,
                        p->stream);
    }
    if (det_live->out_dtype == 0 && out_u16 != NULL)
    {
        cudaMemcpyAsync(out_u16, p->d_out_u16, npix * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                        p->stream);
    }
    else if (det_live->out_dtype != 0 && out_f32 != NULL)
    {
        cudaMemcpyAsync(out_f32, p->d_out_f32, npix * sizeof(float), cudaMemcpyDeviceToHost,
                        p->stream);
    }
    cudaStreamSynchronize(p->stream);
}

#endif /* HAVE_CUDA */
