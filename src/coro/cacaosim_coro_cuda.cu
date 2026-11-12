// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_cuda.cu
 * @brief   CUDA cuBLAS StridedBatched MFT + cuFFT polychromatic custom-mask coronagraph backend
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

    float         *d_lam_um;
    float         *d_lowfs_pup_mask;
    cuComplex     *d_pup_mask;
    cuComplex     *d_lyot_cplx;
    float         *d_in_opd;
    float         *d_in_amp;
    float         *d_out_opd;
    float         *d_out_amp;
    float         *d_lowfs_opd;
    float         *d_lowfs_amp;
    cuComplex     *d_epup;

    /* Babinet MFT buffers */
    cuComplex     *d_fwd_U;
    cuComplex     *d_fwd_V;
    cuComplex     *d_inv_U;
    cuComplex     *d_inv_V;
    cuComplex     *d_fpm_delta;
    float         *d_fpm_refl_w;
    cuComplex     *d_ecore;
    cuComplex     *d_tmp;
    cuComplex     *d_efoc;
    cuComplex     *d_efoc_r;

    /* Full-plane FFT buffers */
    cuComplex     *d_shift_1d;
    cuComplex     *d_fft_mask;
    float         *d_fft_refl_w;
    cuComplex     *d_fft_buf;
    cuComplex     *d_fft_lo_buf;
} coro_cuda_state_t;

__global__ static void k_coro_build_epup_batch(
    const float *__restrict__     opd,
    const float *__restrict__     amp,
    uint32_t                      in_nlam,
    uint64_t                      in_stride,
    const cuComplex *__restrict__ pup_mask,
    const float *__restrict__     lam_um,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    cuComplex *__restrict__       epup)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= eff_nlam)
    {
        return;
    }

    uint32_t ic      = (in_nlam > 1u)
                           ? (((uint32_t) c < in_nlam) ? (uint32_t) c : (in_nlam - 1u))
                           : 0u;
    uint64_t in_idx  = (uint64_t) ic * in_stride + p;
    uint64_t out_idx = (uint64_t) c * npix_wf + p;
    float    a       = (amp != NULL) ? amp[in_idx] : 1.0f;
    if (a <= 0.0f)
    {
        epup[out_idx] = make_cuComplex(0.0f, 0.0f);
        return;
    }

    float phi = ((opd != NULL) ? opd[in_idx] : 0.0f) * (CACAOSIM_TWO_PI_F / lam_um[c]);
    float s   = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &s, &co);
    float pr = a * co;
    float pi = a * s;

    if (pup_mask != NULL)
    {
        cuComplex pm = pup_mask[out_idx];
        float     nr = pr * pm.x - pi * pm.y;
        float     ni = pr * pm.y + pi * pm.x;
        pr           = nr;
        pi           = ni;
    }
    epup[out_idx] = make_cuComplex(pr, pi);
}

__global__ static void k_coro_apply_lyot_batch(
    const cuComplex *__restrict__ epup,
    const cuComplex *__restrict__ lyot_cplx,
    const float *__restrict__     lam_um,
    float                         refl,
    int32_t                       lo_lyot,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    int32_t                       c_mid,
    uint64_t                      out_stride,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= eff_nlam)
    {
        return;
    }

    uint64_t  cidx       = (uint64_t) c * npix_wf + p;
    uint64_t  oidx       = (uint64_t) c * out_stride + p;
    cuComplex u          = epup[cidx];
    cuComplex ly         = lyot_cplx[cidx];
    float     rad_to_opd = lam_um[c] / CACAOSIM_TWO_PI_F;

    float lr      = u.x * ly.x - u.y * ly.y;
    float li      = u.x * ly.y + u.y * ly.x;
    float amp     = hypotf(lr, li);
    out_amp[oidx] = amp;
    out_opd[oidx] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (c == c_mid && lowfs_opd != NULL && lowfs_amp != NULL)
    {
        if (lo_lyot)
        {
            float lya2   = ly.x * ly.x + ly.y * ly.y;
            float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
            float la     = r_ly * hypotf(u.x, u.y);
            lowfs_amp[p] = la;
            lowfs_opd[p] = (la > 1e-12f) ? (atan2f(u.y, u.x) * rad_to_opd) : 0.0f;
        }
        else
        {
            lowfs_amp[p] = 0.0f;
            lowfs_opd[p] = 0.0f;
        }
    }
}

__global__ static void k_coro_scale_real_copy(
    const cuComplex *__restrict__ src,
    const float *__restrict__     w,
    cuComplex *__restrict__       dst,
    uint64_t                      npix)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        float     wi = w[i];
        cuComplex s  = src[i];
        dst[i]       = make_cuComplex(s.x * wi, s.y * wi);
    }
}

__global__ static void k_coro_write_lowfs(
    const cuComplex *__restrict__ ecore,
    const float *__restrict__     lowfs_pup_mask,
    float                         refl,
    float                         rad_to_opd,
    uint64_t                      npix,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p < npix)
    {
        cuComplex ec = ecore[p];
        float     pm = (lowfs_pup_mask != NULL) ? lowfs_pup_mask[p] : 1.0f;
        float     la = refl * pm * hypotf(ec.x, ec.y);
        lowfs_amp[p] = la;
        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ec.y, ec.x) * rad_to_opd) : 0.0f;
    }
}

__global__ static void k_coro_mul_cplx(
    cuComplex *__restrict__       efoc,
    const cuComplex *__restrict__ delta,
    uint64_t                      total)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        cuComplex e = efoc[i];
        cuComplex d = delta[i];
        efoc[i]     = make_cuComplex(e.x * d.x - e.y * d.y, e.x * d.y + e.y * d.x);
    }
}

__global__ static void k_coro_babinet_finalize_batch(
    const cuComplex *__restrict__ epup,
    const cuComplex *__restrict__ ecore,
    const cuComplex *__restrict__ lyot_cplx,
    const float *__restrict__     lam_um,
    float                         refl,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    int32_t                       c_mid,
    uint64_t                      out_stride,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= eff_nlam)
    {
        return;
    }

    uint64_t  cidx       = (uint64_t) c * npix_wf + p;
    uint64_t  oidx       = (uint64_t) c * out_stride + p;
    cuComplex ep         = epup[cidx];
    cuComplex ec         = ecore[cidx];
    cuComplex ly         = lyot_cplx[cidx];
    float     ur         = ep.x + ec.x;
    float     ui         = ep.y + ec.y;
    float     lr         = ur * ly.x - ui * ly.y;
    float     li         = ur * ly.y + ui * ly.x;
    float     rad_to_opd = lam_um[c] / CACAOSIM_TWO_PI_F;
    float     amp        = hypotf(lr, li);

    out_amp[oidx] = amp;
    out_opd[oidx] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (c == c_mid && lowfs_opd != NULL && lowfs_amp != NULL)
    {
        float lya2   = ly.x * ly.x + ly.y * ly.y;
        float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
        float la     = r_ly * hypotf(ur, ui);
        lowfs_amp[p] = la;
        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
    }
}

__global__ static void k_coro_fft_embed_epup(
    const cuComplex *__restrict__ epup_c,
    const cuComplex *__restrict__ shift_x,
    const cuComplex *__restrict__ shift_y,
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

    uint32_t  xp = gx - pad_off;
    uint32_t  yp = gy - pad_off;
    cuComplex ep = epup_c[(uint64_t) yp * nwf + xp];
    cuComplex sx = shift_x[gx];
    cuComplex sy = shift_y[gy];
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    fft_buf[idx] = make_cuComplex(ep.x * sr - ep.y * si, ep.x * si + ep.y * sr);
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

__global__ static void k_coro_fft_extract_lowfs_fpm(
    const cuComplex *__restrict__ fft_lo_buf,
    const cuComplex *__restrict__ shift_x,
    const cuComplex *__restrict__ shift_y,
    const float *__restrict__     lowfs_pup_mask,
    float                         refl,
    float                         rad_to_opd,
    uint32_t                      nwf,
    uint32_t                      nfft,
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

    cuComplex sx = make_cuComplex(shift_x[gx].x, -shift_x[gx].y);
    cuComplex sy = make_cuComplex(shift_y[gy].x, -shift_y[gy].y);
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    cuComplex e    = fft_lo_buf[(uint64_t) gy * nfft + gx];
    float     ur   = e.x * sr - e.y * si;
    float     ui   = e.x * si + e.y * sr;
    float     pm   = (lowfs_pup_mask != NULL) ? lowfs_pup_mask[p] : 1.0f;
    float     la   = refl * pm * hypotf(ur, ui);
    lowfs_amp[p]   = la;
    lowfs_opd[p]   = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
}

__global__ static void k_coro_fft_extract_cplx(
    const cuComplex *__restrict__ fft_buf,
    const cuComplex *__restrict__ shift_x,
    const cuComplex *__restrict__ shift_y,
    const cuComplex *__restrict__ lyot_c,
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

    cuComplex sx = make_cuComplex(shift_x[gx].x, -shift_x[gx].y);
    cuComplex sy = make_cuComplex(shift_y[gy].x, -shift_y[gy].y);
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    cuComplex e  = fft_buf[(uint64_t) gy * nfft + gx];
    float     ur = e.x * sr - e.y * si;
    float     ui = e.x * si + e.y * sr;
    cuComplex ly = lyot_c[p];

    float lr   = ur * ly.x - ui * ly.y;
    float li   = ur * ly.y + ui * ly.x;
    float amp  = hypotf(lr, li);
    out_amp[p] = amp;
    out_opd[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (lowfs_opd != NULL && lowfs_amp != NULL)
    {
        float lya2   = ly.x * ly.x + ly.y * ly.y;
        float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
        float la     = r_ly * hypotf(ur, ui);
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

    const size_t nlam     = (size_t) ctx->eff_nlam;
    const size_t wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    const size_t poly_f   = nlam * wf_bytes;
    const size_t poly_c   = nlam * (size_t) ctx->npix_wf * sizeof(cuComplex);

    if (cudaMalloc((void **) &cs->d_lam_um, nlam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_pup_mask, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lyot_cplx, poly_c) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_in_opd, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_in_amp, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_opd, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_amp, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_epup, poly_c) != cudaSuccess)
    {
        cacaosim_coro_cuda_free(ctx);
        return -1;
    }

    cudaMemcpyAsync(cs->d_lam_um, ctx->lam_um, nlam * sizeof(float), cudaMemcpyHostToDevice,
                    cs->stream);
    cudaMemcpyAsync(cs->d_lowfs_pup_mask, ctx->lowfs_pup_mask, wf_bytes, cudaMemcpyHostToDevice,
                    cs->stream);
    cudaMemcpyAsync(cs->d_lyot_cplx, ctx->lyot_cplx, poly_c, cudaMemcpyHostToDevice, cs->stream);

    if (ctx->has_pup_mask && ctx->pup_mask_cplx != NULL)
    {
        if (cudaMalloc((void **) &cs->d_pup_mask, poly_c) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_pup_mask, ctx->pup_mask_cplx, poly_c, cudaMemcpyHostToDevice,
                        cs->stream);
    }

    if (ctx->is_babinet && !ctx->skip_fpm)
    {
        if (cublasCreate(&cs->cublas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(cs->cublas, cs->stream) != CUBLAS_STATUS_SUCCESS)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }

        uint32_t max_m   = (ctx->nx_mft > ctx->ny_mft) ? ctx->nx_mft : ctx->ny_mft;
        size_t   u_bytes = nlam * (size_t) ctx->ny_mft * ctx->cfg.nwf * sizeof(cuComplex);
        size_t   v_bytes = nlam * (size_t) ctx->nx_mft * ctx->cfg.nwf * sizeof(cuComplex);
        size_t   t_bytes = nlam * (size_t) max_m * ctx->cfg.nwf * sizeof(cuComplex);
        size_t   mft_c   = nlam * (size_t) ctx->npix_mft * sizeof(cuComplex);
        size_t   mft_f   = nlam * (size_t) ctx->npix_mft * sizeof(float);
        size_t   mft_1c  = (size_t) ctx->npix_mft * sizeof(cuComplex);

        if (cudaMalloc((void **) &cs->d_fwd_U, u_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_inv_U, u_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fwd_V, v_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_inv_V, v_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fpm_delta, mft_c) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fpm_refl_w, mft_f) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_ecore, poly_c) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_tmp, t_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_efoc, mft_c) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_efoc_r, mft_1c) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }

        cudaMemcpyAsync(cs->d_fwd_U, ctx->fwd_U_cplx, u_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_inv_U, ctx->inv_U_cplx, u_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_fwd_V, ctx->fwd_V_cplx, v_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_inv_V, ctx->inv_V_cplx, v_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_fpm_delta, ctx->fpm_delta_cplx, mft_c, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fpm_refl_w, ctx->fpm_refl_w, mft_f, cudaMemcpyHostToDevice,
                        cs->stream);
    }
    else if (!ctx->is_babinet)
    {
        size_t sh_bytes  = (size_t) ctx->nfft * 2u * sizeof(cuComplex);
        size_t msk_bytes = nlam * (size_t) ctx->npix_fft * sizeof(cuComplex);
        size_t rw_bytes  = (size_t) ctx->npix_fft * sizeof(float);
        size_t fft_bytes = (size_t) ctx->npix_fft * sizeof(cuComplex);
        if (cudaMalloc((void **) &cs->d_shift_1d, sh_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_mask, msk_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_refl_w, rw_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_buf, fft_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_lo_buf, fft_bytes) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_shift_1d, ctx->shift_1d_cplx, sh_bytes, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fft_mask, ctx->fft_mask_cplx, msk_bytes, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fft_refl_w, ctx->fft_refl_w, rw_bytes, cudaMemcpyHostToDevice,
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
    cudaFree(cs->d_lam_um);
    cudaFree(cs->d_lowfs_pup_mask);
    cudaFree(cs->d_pup_mask);
    cudaFree(cs->d_lyot_cplx);
    cudaFree(cs->d_in_opd);
    cudaFree(cs->d_in_amp);
    cudaFree(cs->d_out_opd);
    cudaFree(cs->d_out_amp);
    cudaFree(cs->d_lowfs_opd);
    cudaFree(cs->d_lowfs_amp);
    cudaFree(cs->d_epup);
    cudaFree(cs->d_fwd_U);
    cudaFree(cs->d_fwd_V);
    cudaFree(cs->d_inv_U);
    cudaFree(cs->d_inv_V);
    cudaFree(cs->d_fpm_delta);
    cudaFree(cs->d_fpm_refl_w);
    cudaFree(cs->d_ecore);
    cudaFree(cs->d_tmp);
    cudaFree(cs->d_efoc);
    cudaFree(cs->d_efoc_r);
    cudaFree(cs->d_shift_1d);
    cudaFree(cs->d_fft_mask);
    cudaFree(cs->d_fft_refl_w);
    cudaFree(cs->d_fft_buf);
    cudaFree(cs->d_fft_lo_buf);
    if (cs->owns_stream && cs->stream)
    {
        cudaStreamDestroy(cs->stream);
    }
    free(cs);
    ctx->cuda_priv = NULL;
}

extern "C" void cacaosim_coro_cuda_step_poly(
    cacaosim_coro_ctx_t *ctx,
    const float         *h_in_opd,
    const float         *h_in_amp,
    uint32_t             in_nlam,
    uint64_t             in_stride,
    float               *h_out_opd,
    float               *h_out_amp,
    uint64_t             out_stride,
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
    const uint64_t     npix_wf  = ctx->npix_wf;
    const int32_t      eff_nlam = ctx->eff_nlam;
    const int32_t      c_mid    = eff_nlam / 2;
    const size_t       wf_bytes = (size_t) npix_wf * sizeof(float);
    const int          dev_mode = (d_in_opd != NULL && d_out_opd != NULL) ? 1 : 0;

    uint32_t copy_in_nlam = (in_nlam >= 1u)
                                ? ((in_nlam <= (uint32_t) eff_nlam) ? in_nlam : (uint32_t) eff_nlam)
                                : 1u;

    const float *din_opd       = d_in_opd;
    const float *din_amp       = d_in_amp;
    uint64_t     dev_in_stride = in_stride;
    float       *dout_o        = (d_out_opd != NULL) ? d_out_opd : cs->d_out_opd;
    float       *dout_a        = (d_out_amp != NULL) ? d_out_amp : cs->d_out_amp;
    uint64_t     dev_out_str   = (d_out_opd != NULL) ? out_stride : npix_wf;
    int          want_lo       = (d_lowfs_opd != NULL || h_lowfs_opd != NULL) ? 1 : 0;
    float       *dlo_o = want_lo ? ((d_lowfs_opd != NULL) ? d_lowfs_opd : cs->d_lowfs_opd) : NULL;
    float       *dlo_a = want_lo ? ((d_lowfs_amp != NULL) ? d_lowfs_amp : cs->d_lowfs_amp) : NULL;

    if (!dev_mode)
    {
        dev_in_stride = npix_wf;
        if (h_in_opd != NULL)
        {
            if (copy_in_nlam == 1u || in_stride == npix_wf)
            {
                cudaMemcpyAsync(cs->d_in_opd, h_in_opd, (size_t) copy_in_nlam * wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(cs->d_in_opd, wf_bytes, h_in_opd,
                                  (size_t) in_stride * sizeof(float), wf_bytes, copy_in_nlam,
                                  cudaMemcpyHostToDevice, cs->stream);
            }
            din_opd = cs->d_in_opd;
        }
        if (h_in_amp != NULL)
        {
            if (copy_in_nlam == 1u || in_stride == npix_wf)
            {
                cudaMemcpyAsync(cs->d_in_amp, h_in_amp, (size_t) copy_in_nlam * wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(cs->d_in_amp, wf_bytes, h_in_amp,
                                  (size_t) in_stride * sizeof(float), wf_bytes, copy_in_nlam,
                                  cudaMemcpyHostToDevice, cs->stream);
            }
            din_amp = cs->d_in_amp;
        }
    }

    const int32_t  lo_lyot = (ctx->cfg.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? 1 : 0;
    const float    refl    = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;
    const uint32_t bl_wf   = (uint32_t) ((npix_wf + 255ULL) / 256ULL);
    dim3           grd_wf(bl_wf, (unsigned int) eff_nlam, 1);

    k_coro_build_epup_batch<<<grd_wf, 256, 0, cs->stream>>>(
        din_opd, din_amp, copy_in_nlam, dev_in_stride, cs->d_pup_mask, cs->d_lam_um, npix_wf,
        eff_nlam, cs->d_epup);

    if (ctx->skip_fpm)
    {
        k_coro_apply_lyot_batch<<<grd_wf, 256, 0, cs->stream>>>(
            cs->d_epup, cs->d_lyot_cplx, cs->d_lam_um, refl, lo_lyot, npix_wf, eff_nlam, c_mid,
            dev_out_str, dout_o, dout_a, dlo_o, dlo_a);
    }
    else if (ctx->is_babinet)
    {
        const int           nwf   = (int) ctx->cfg.nwf;
        const int           mx    = (int) ctx->nx_mft;
        const int           my    = (int) ctx->ny_mft;
        const int           max_m = (mx > my) ? mx : my;
        const cuComplex     one   = make_cuComplex(1.0f, 0.0f);
        const cuComplex     zero  = make_cuComplex(0.0f, 0.0f);
        const long long int str_ep = (long long int) npix_wf;
        const long long int str_u  = (long long int) my * nwf;
        const long long int str_v  = (long long int) mx * nwf;
        const long long int str_t  = (long long int) max_m * nwf;
        const long long int str_ef = (long long int) ctx->npix_mft;

        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, my, nwf, &one,
                                  cs->d_epup, nwf, str_ep, cs->d_fwd_U, nwf, str_u, &zero,
                                  cs->d_tmp, nwf, str_t, eff_nlam);
        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, mx, my, nwf, &one,
                                  cs->d_fwd_V, mx, str_v, cs->d_tmp, nwf, str_t, &zero, cs->d_efoc,
                                  mx, str_ef, eff_nlam);

        if (want_lo && !lo_lyot && dlo_o != NULL && dlo_a != NULL)
        {
            uint32_t bl_m1 = (uint32_t) ((ctx->npix_mft + 255ULL) / 256ULL);
            k_coro_scale_real_copy<<<bl_m1, 256, 0, cs->stream>>>(
                cs->d_efoc + (size_t) c_mid * ctx->npix_mft,
                cs->d_fpm_refl_w + (size_t) c_mid * ctx->npix_mft, cs->d_efoc_r, ctx->npix_mft);
            cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, mx, nwf, my, &one, cs->d_efoc_r, mx,
                        cs->d_inv_U + (size_t) c_mid * str_u, my, &zero, cs->d_tmp, mx);
            cublasCgemm(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, mx, &one,
                        cs->d_inv_V + (size_t) c_mid * str_v, nwf, cs->d_tmp, mx, &zero,
                        cs->d_ecore, nwf);
            float rad_to_opd_mid = ctx->lam_um[c_mid] / CACAOSIM_TWO_PI_F;
            k_coro_write_lowfs<<<bl_wf, 256, 0, cs->stream>>>(
                cs->d_ecore, cs->d_lowfs_pup_mask, refl, rad_to_opd_mid, npix_wf, dlo_o, dlo_a);
        }

        uint64_t total_mft = (uint64_t) eff_nlam * ctx->npix_mft;
        uint32_t bl_mft    = (uint32_t) ((total_mft + 255ULL) / 256ULL);
        k_coro_mul_cplx<<<bl_mft, 256, 0, cs->stream>>>(cs->d_efoc, cs->d_fpm_delta, total_mft);

        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, mx, nwf, my, &one,
                                  cs->d_efoc, mx, str_ef, cs->d_inv_U, my, str_u, &zero, cs->d_tmp,
                                  mx, str_t, eff_nlam);
        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, mx, &one,
                                  cs->d_inv_V, nwf, str_v, cs->d_tmp, mx, str_t, &zero,
                                  cs->d_ecore, nwf, str_ep, eff_nlam);

        k_coro_babinet_finalize_batch<<<grd_wf, 256, 0, cs->stream>>>(
            cs->d_epup, cs->d_ecore, cs->d_lyot_cplx, cs->d_lam_um, refl, npix_wf, eff_nlam, c_mid,
            dev_out_str, dout_o, dout_a, (want_lo && lo_lyot) ? dlo_o : NULL,
            (want_lo && lo_lyot) ? dlo_a : NULL);
    }
    else
    {
        const uint32_t   bl_fft = (uint32_t) ((ctx->npix_fft + 255ULL) / 256ULL);
        const cuComplex *sh_x   = cs->d_shift_1d;
        const cuComplex *sh_y   = cs->d_shift_1d + ctx->nfft;

        for (int32_t c = 0; c < eff_nlam; c++)
        {
            int    is_mid = (c == c_mid && want_lo && dlo_o != NULL && dlo_a != NULL) ? 1 : 0;
            float *lo_o_c = (is_mid && lo_lyot) ? dlo_o : NULL;
            float *lo_a_c = (is_mid && lo_lyot) ? dlo_a : NULL;
            float  r2o_c  = ctx->lam_um[c] / CACAOSIM_TWO_PI_F;

            k_coro_fft_embed_epup<<<bl_fft, 256, 0, cs->stream>>>(
                cs->d_epup + (size_t) c * npix_wf, sh_x, sh_y, ctx->cfg.nwf, ctx->nfft,
                cs->d_fft_buf);
            cufftExecC2C(cs->fft_plan, cs->d_fft_buf, cs->d_fft_buf, CUFFT_FORWARD);

            if (is_mid && !lo_lyot)
            {
                k_coro_scale_real_copy<<<bl_fft, 256, 0, cs->stream>>>(
                    cs->d_fft_buf, cs->d_fft_refl_w, cs->d_fft_lo_buf, ctx->npix_fft);
                cufftExecC2C(cs->fft_plan, cs->d_fft_lo_buf, cs->d_fft_lo_buf, CUFFT_INVERSE);
                k_coro_fft_extract_lowfs_fpm<<<bl_wf, 256, 0, cs->stream>>>(
                    cs->d_fft_lo_buf, sh_x, sh_y, cs->d_lowfs_pup_mask, refl, r2o_c, ctx->cfg.nwf,
                    ctx->nfft, dlo_o, dlo_a);
            }

            k_coro_fft_mul_mask<<<bl_fft, 256, 0, cs->stream>>>(
                cs->d_fft_buf, cs->d_fft_mask + (size_t) c * ctx->npix_fft, ctx->npix_fft);
            cufftExecC2C(cs->fft_plan, cs->d_fft_buf, cs->d_fft_buf, CUFFT_INVERSE);
            k_coro_fft_extract_cplx<<<bl_wf, 256, 0, cs->stream>>>(
                cs->d_fft_buf, sh_x, sh_y, cs->d_lyot_cplx + (size_t) c * npix_wf, refl, r2o_c,
                ctx->cfg.nwf, ctx->nfft, dout_o + (size_t) c * dev_out_str,
                dout_a + (size_t) c * dev_out_str, lo_o_c, lo_a_c);
        }
    }

    if (!dev_mode)
    {
        if (h_out_opd != NULL)
        {
            if (eff_nlam == 1 || out_stride == npix_wf)
            {
                cudaMemcpyAsync(h_out_opd, dout_o, (size_t) eff_nlam * wf_bytes,
                                cudaMemcpyDeviceToHost, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(h_out_opd, (size_t) out_stride * sizeof(float), dout_o, wf_bytes,
                                  wf_bytes, (size_t) eff_nlam, cudaMemcpyDeviceToHost, cs->stream);
            }
        }
        if (h_out_amp != NULL)
        {
            if (eff_nlam == 1 || out_stride == npix_wf)
            {
                cudaMemcpyAsync(h_out_amp, dout_a, (size_t) eff_nlam * wf_bytes,
                                cudaMemcpyDeviceToHost, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(h_out_amp, (size_t) out_stride * sizeof(float), dout_a, wf_bytes,
                                  wf_bytes, (size_t) eff_nlam, cudaMemcpyDeviceToHost, cs->stream);
            }
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
