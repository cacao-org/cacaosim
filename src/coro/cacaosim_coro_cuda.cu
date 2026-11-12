// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_cuda.cu
 * @brief   CUDA cuBLAS Babinet MFT + cuFFT Vortex/4QPM coronagraph and LOWFS pickoff backend
 */

#ifdef HAVE_CUDA

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>

#include "coro/cacaosim_coro.h"

typedef struct
{
    cudaStream_t   stream;
    int            owns_stream;
    cublasHandle_t cublas;
    cufftHandle    fft_plan;
    int            has_fft;

    float         *d_lyot;
    float         *d_in_opd;
    float         *d_in_amp;
    float         *d_out_opd;
    float         *d_out_amp;
    float         *d_lowfs_opd;
    float         *d_lowfs_amp;

    /* Babinet MFT buffers */
    cuComplex     *d_fwd_U;
    cuComplex     *d_fwd_V;
    cuComplex     *d_inv_U;
    cuComplex     *d_inv_V;
    float         *d_fpm_w;
    cuComplex     *d_epup;
    cuComplex     *d_ecore;
    cuComplex     *d_tmp;
    cuComplex     *d_efoc;

    /* Full-plane FFT buffers */
    cuComplex     *d_shift_1d;
    cuComplex     *d_fft_mask;
    cuComplex     *d_fft_buf;
} coro_cuda_state_t;

__global__ static void k_coro_build_epup(
    const float *__restrict__ opd,
    const float *__restrict__ amp,
    float                     opd_to_rad,
    uint64_t                  npix,
    cuComplex *__restrict__   epup)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix)
    {
        return;
    }
    float a = (amp != NULL) ? amp[p] : 1.0f;
    if (a <= 0.0f)
    {
        epup[p] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    float phi = ((opd != NULL) ? opd[p] : 0.0f) * opd_to_rad;
    float s   = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &s, &co);
    epup[p] = make_cuComplex(a * co, a * s);
}

__global__ static void k_coro_scale_real(
    cuComplex *__restrict__   efoc,
    const float *__restrict__ w,
    uint64_t                  npix)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        float wi  = w[i];
        efoc[i].x *= wi;
        efoc[i].y *= wi;
    }
}

__global__ static void k_coro_babinet_finalize(
    const cuComplex *__restrict__ epup,
    const cuComplex *__restrict__ ecore,
    const float *__restrict__     lyot,
    float                         c_re,
    float                         c_im,
    float                         refl,
    float                         rad_to_opd,
    uint64_t                      npix,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix)
    {
        return;
    }

    cuComplex ep = epup[p];
    cuComplex ec = ecore[p];
    float     ly = lyot[p];

    float lr   = ly * (ep.x + (c_re * ec.x - c_im * ec.y));
    float li   = ly * (ep.y + (c_re * ec.y + c_im * ec.x));
    float amp  = hypotf(lr, li);
    out_amp[p] = amp;
    out_opd[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (lowfs_opd != NULL && lowfs_amp != NULL)
    {
        float la     = refl * hypotf(ec.x, ec.y);
        lowfs_amp[p] = la;
        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ec.y, ec.x) * rad_to_opd) : 0.0f;
    }
}

__global__ static void k_coro_fft_embed(
    const float *__restrict__     opd,
    const float *__restrict__     amp,
    const cuComplex *__restrict__ shift_1d,
    float                         opd_to_rad,
    uint32_t                      nwf,
    uint32_t                      nfft,
    cuComplex *__restrict__       fft_buf)
{
    uint64_t idx   = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t) nfft * nfft;
    if (idx >= total)
    {
        return;
    }

    uint32_t gx      = (uint32_t) (idx % nfft);
    uint32_t gy      = (uint32_t) (idx / nfft);
    uint32_t pad_off = (nfft - nwf) / 2u;

    if (gx < pad_off || gx >= pad_off + nwf || gy < pad_off || gy >= pad_off + nwf)
    {
        fft_buf[idx] = make_cuComplex(0.0f, 0.0f);
        return;
    }

    uint32_t xp = gx - pad_off;
    uint32_t yp = gy - pad_off;
    uint64_t p  = (uint64_t) yp * nwf + xp;
    float    a  = (amp != NULL) ? amp[p] : 1.0f;
    if (a <= 0.0f)
    {
        fft_buf[idx] = make_cuComplex(0.0f, 0.0f);
        return;
    }

    float phi = ((opd != NULL) ? opd[p] : 0.0f) * opd_to_rad;
    float s   = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &s, &co);
    float pr = a * co;
    float pi = a * s;

    cuComplex sx = shift_1d[gx];
    cuComplex sy = shift_1d[gy];
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    fft_buf[idx] = make_cuComplex(pr * sr - pi * si, pr * si + pi * sr);
}

__global__ static void k_coro_fft_mul_mask(
    cuComplex *__restrict__       fft_buf,
    const cuComplex *__restrict__ mask,
    uint64_t                      npix_fft)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix_fft)
    {
        cuComplex e = fft_buf[i];
        cuComplex m = mask[i];
        fft_buf[i]  = make_cuComplex(e.x * m.x - e.y * m.y, e.x * m.y + e.y * m.x);
    }
}

__global__ static void k_coro_fft_extract(
    const cuComplex *__restrict__ fft_buf,
    const cuComplex *__restrict__ shift_1d,
    const float *__restrict__     lyot,
    float                         refl,
    float                         rad_to_opd,
    uint32_t                      nwf,
    uint32_t                      nfft,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p     = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t) nwf * nwf;
    if (p >= total)
    {
        return;
    }

    uint32_t xp      = (uint32_t) (p % nwf);
    uint32_t yp      = (uint32_t) (p / nwf);
    uint32_t pad_off = (nfft - nwf) / 2u;
    uint32_t gx      = xp + pad_off;
    uint32_t gy      = yp + pad_off;

    cuComplex sx = make_cuComplex(shift_1d[gx].x, -shift_1d[gx].y);
    cuComplex sy = make_cuComplex(shift_1d[gy].x, -shift_1d[gy].y);
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    cuComplex e  = fft_buf[(uint64_t) gy * nfft + gx];
    float     ur = e.x * sr - e.y * si;
    float     ui = e.x * si + e.y * sr;
    float     ly = lyot[p];

    float lr   = ly * ur;
    float li   = ly * ui;
    float amp  = hypotf(lr, li);
    out_amp[p] = amp;
    out_opd[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (lowfs_opd != NULL && lowfs_amp != NULL)
    {
        float la     = refl * (1.0f - ly) * hypotf(ur, ui);
        lowfs_amp[p] = la;
        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
    }
}

extern "C" int cacaosim_coro_cuda_init(cacaosim_coro_ctx_t *ctx, void *ext_stream)
{
    coro_cuda_state_t *cs = (coro_cuda_state_t *) calloc(1, sizeof(*cs));
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
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cs->owns_stream = 1;
    }

    size_t wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    if (cudaMalloc((void **) &cs->d_lyot, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_in_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_in_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_amp, wf_bytes) != cudaSuccess)
    {
        cacaosim_coro_cuda_free(ctx);
        return -1;
    }
    cudaMemcpyAsync(cs->d_lyot, ctx->lyot_mask, wf_bytes, cudaMemcpyHostToDevice, cs->stream);

    if (ctx->is_babinet)
    {
        if (cublasCreate(&cs->cublas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(cs->cublas, cs->stream) != CUBLAS_STATUS_SUCCESS)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }

        size_t uv_bytes  = (size_t) ctx->n_mft * ctx->cfg.nwf * sizeof(cuComplex);
        size_t mft_c     = (size_t) ctx->npix_mft * sizeof(cuComplex);
        size_t mft_f     = (size_t) ctx->npix_mft * sizeof(float);
        size_t ep_bytes  = (size_t) ctx->npix_wf * sizeof(cuComplex);

        if (cudaMalloc((void **) &cs->d_fwd_U, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fwd_V, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_inv_U, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_inv_V, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fpm_w, mft_f) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_epup, ep_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_ecore, ep_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_tmp, uv_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_efoc, mft_c) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }

        cudaMemcpyAsync(cs->d_fwd_U, ctx->fwd_U_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_fwd_V, ctx->fwd_V_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_inv_U, ctx->inv_U_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_inv_V, ctx->inv_V_cplx, uv_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_fpm_w, ctx->fpm_weight, mft_f, cudaMemcpyHostToDevice, cs->stream);
    }
    else
    {
        size_t sh_bytes  = (size_t) ctx->nfft * sizeof(cuComplex);
        size_t fft_bytes = (size_t) ctx->npix_fft * sizeof(cuComplex);
        if (cudaMalloc((void **) &cs->d_shift_1d, sh_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_mask, fft_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_buf, fft_bytes) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_shift_1d, ctx->shift_1d_cplx, sh_bytes, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fft_mask, ctx->fft_mask_cplx, fft_bytes, cudaMemcpyHostToDevice,
                        cs->stream);

        int n = (int) ctx->nfft;
        if (cufftPlan2d(&cs->fft_plan, n, n, CUFFT_C2C) != CUFFT_SUCCESS ||
            cufftSetStream(cs->fft_plan, cs->stream) != CUFFT_SUCCESS)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cs->has_fft = 1;
    }

    cudaStreamSynchronize(cs->stream);
    return 0;
}

extern "C" void cacaosim_coro_cuda_free(cacaosim_coro_ctx_t *ctx)
{
    coro_cuda_state_t *cs = (coro_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL)
    {
        return;
    }
    if (cs->cublas)
    {
        cublasDestroy(cs->cublas);
    }
    if (cs->has_fft)
    {
        cufftDestroy(cs->fft_plan);
    }
    cudaFree(cs->d_lyot);
    cudaFree(cs->d_in_opd);
    cudaFree(cs->d_in_amp);
    cudaFree(cs->d_out_opd);
    cudaFree(cs->d_out_amp);
    cudaFree(cs->d_lowfs_opd);
    cudaFree(cs->d_lowfs_amp);
    cudaFree(cs->d_fwd_U);
    cudaFree(cs->d_fwd_V);
    cudaFree(cs->d_inv_U);
    cudaFree(cs->d_inv_V);
    cudaFree(cs->d_fpm_w);
    cudaFree(cs->d_epup);
    cudaFree(cs->d_ecore);
    cudaFree(cs->d_tmp);
    cudaFree(cs->d_efoc);
    cudaFree(cs->d_shift_1d);
    cudaFree(cs->d_fft_mask);
    cudaFree(cs->d_fft_buf);
    if (cs->owns_stream && cs->stream)
    {
        cudaStreamDestroy(cs->stream);
    }
    free(cs);
    ctx->cuda_priv = NULL;
}

extern "C" void cacaosim_coro_cuda_step(
    cacaosim_coro_ctx_t *ctx,
    const float         *h_in_opd,
    const float         *h_in_amp,
    float               *h_out_opd,
    float               *h_out_amp,
    float               *h_lowfs_opd,
    float               *h_lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp)
{
    coro_cuda_state_t *cs       = (coro_cuda_state_t *) ctx->cuda_priv;
    size_t             wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    int                dev_mode = (d_in_opd != NULL && d_out_opd != NULL) ? 1 : 0;

    const float *din_opd = d_in_opd;
    const float *din_amp = d_in_amp;
    float       *dout_o  = (d_out_opd != NULL) ? d_out_opd : cs->d_out_opd;
    float       *dout_a  = (d_out_amp != NULL) ? d_out_amp : cs->d_out_amp;
    int          want_lo = (d_lowfs_opd != NULL || h_lowfs_opd != NULL) ? 1 : 0;
    float       *dlo_o   = want_lo ? ((d_lowfs_opd != NULL) ? d_lowfs_opd : cs->d_lowfs_opd) : NULL;
    float       *dlo_a   = want_lo ? ((d_lowfs_amp != NULL) ? d_lowfs_amp : cs->d_lowfs_amp) : NULL;

    if (!dev_mode)
    {
        if (h_in_opd != NULL)
        {
            cudaMemcpyAsync(cs->d_in_opd, h_in_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
            din_opd = cs->d_in_opd;
        }
        if (h_in_amp != NULL)
        {
            cudaMemcpyAsync(cs->d_in_amp, h_in_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
            din_amp = cs->d_in_amp;
        }
    }

    float    lam        = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 1.65f;
    float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    float    rad_to_opd = lam / CACAOSIM_TWO_PI_F;
    float    refl       = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;
    uint32_t bl_wf      = (uint32_t) ((ctx->npix_wf + 255ULL) / 256ULL);

    if (ctx->is_babinet)
    {
        int nwf = (int) ctx->cfg.nwf;
        int m   = (int) ctx->n_mft;

        k_coro_build_epup<<<bl_wf, 256, 0, cs->stream>>>(din_opd, din_amp, opd_to_rad, ctx->npix_wf,
                                                         cs->d_epup);

        const cuComplex one  = make_cuComplex(1.0f, 0.0f);
        const cuComplex zero = make_cuComplex(0.0f, 0.0f);

        cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, m, nwf, &one, cs->d_epup, nwf,
                    cs->d_fwd_U, nwf, &zero, cs->d_tmp, nwf);
        cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, m, nwf, &one, cs->d_fwd_V, m,
                    cs->d_tmp, nwf, &zero, cs->d_efoc, m);

        uint32_t bl_mft = (uint32_t) ((ctx->npix_mft + 255ULL) / 256ULL);
        k_coro_scale_real<<<bl_mft, 256, 0, cs->stream>>>(cs->d_efoc, cs->d_fpm_w, ctx->npix_mft);

        cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m, nwf, m, &one, cs->d_efoc, m,
                    cs->d_inv_U, m, &zero, cs->d_tmp, m);
        cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, m, &one, cs->d_inv_V, nwf,
                    cs->d_tmp, m, &zero, cs->d_ecore, nwf);

        float ph   = (ctx->cfg.mode == CACAOSIM_CORO_PHASE_DIMPLE) ? ctx->cfg.fpm_phase_rad : 0.0f;
        float c_re = ctx->cfg.fpm_amp * cosf(ph) - 1.0f;
        float c_im = ctx->cfg.fpm_amp * sinf(ph);

        k_coro_babinet_finalize<<<bl_wf, 256, 0, cs->stream>>>(
            cs->d_epup, cs->d_ecore, cs->d_lyot, c_re, c_im, refl, rad_to_opd, ctx->npix_wf,
            dout_o, dout_a, dlo_o, dlo_a);
    }
    else
    {
        uint32_t bl_fft = (uint32_t) ((ctx->npix_fft + 255ULL) / 256ULL);
        k_coro_fft_embed<<<bl_fft, 256, 0, cs->stream>>>(
            din_opd, din_amp, cs->d_shift_1d, opd_to_rad, ctx->cfg.nwf, ctx->nfft, cs->d_fft_buf);
        cufftExecC2C(cs->fft_plan, cs->d_fft_buf, cs->d_fft_buf, CUFFT_FORWARD);
        k_coro_fft_mul_mask<<<bl_fft, 256, 0, cs->stream>>>(
            cs->d_fft_buf, cs->d_fft_mask, ctx->npix_fft);
        cufftExecC2C(cs->fft_plan, cs->d_fft_buf, cs->d_fft_buf, CUFFT_INVERSE);
        k_coro_fft_extract<<<bl_wf, 256, 0, cs->stream>>>(
            cs->d_fft_buf, cs->d_shift_1d, cs->d_lyot, refl, rad_to_opd, ctx->cfg.nwf, ctx->nfft,
            dout_o, dout_a, dlo_o, dlo_a);
    }

    if (!dev_mode)
    {
        if (h_out_opd != NULL)
        {
            cudaMemcpyAsync(h_out_opd, dout_o, wf_bytes, cudaMemcpyDeviceToHost, cs->stream);
        }
        if (h_out_amp != NULL)
        {
            cudaMemcpyAsync(h_out_amp, dout_a, wf_bytes, cudaMemcpyDeviceToHost, cs->stream);
        }
        if (h_lowfs_opd != NULL && dlo_o != NULL)
        {
            cudaMemcpyAsync(h_lowfs_opd, dlo_o, wf_bytes, cudaMemcpyDeviceToHost, cs->stream);
        }
        if (h_lowfs_amp != NULL && dlo_a != NULL)
        {
            cudaMemcpyAsync(h_lowfs_amp, dlo_a, wf_bytes, cudaMemcpyDeviceToHost, cs->stream);
        }
        cudaStreamSynchronize(cs->stream);
    }
}

#endif /* HAVE_CUDA */
