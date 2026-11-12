// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs_cuda.cu
 * @brief   Batched cuFFT Diffractive Shack-Hartmann WFS GPU backend
 */

#ifdef HAVE_CUDA

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>

#include "common/cacaosim_cuda.h"
#include "shwfs/cacaosim_shwfs.h"

typedef struct
{
    cudaStream_t    stream;
    int             owns_stream;
    cufftHandle     plan_batch;
    int             has_plan;

    float          *d_pup_amp;
    float          *d_dyn_pup;
    float          *d_ncpa_opd;
    float          *d_atm_amp;
    float          *d_wf_in;
    float          *d_dm_opd;
    uint32_t       *d_sub_off;
    uint32_t       *d_sub_len;
    cuFloatComplex *d_shift_1d;
    cuFloatComplex *d_patches;
    float          *d_cam_int;
    float          *d_accum_int;
    uint16_t       *d_out_u16;
    float          *d_out_f32;
} shwfs_cuda_state_t;

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

    uint32_t b   = (uint32_t) (idx / sfft2);
    uint32_t rem = (uint32_t) (idx % sfft2);
    uint32_t lx  = rem % nfft;
    uint32_t ly  = rem / nfft;
    uint32_t sx  = b % n_sub;
    uint32_t sy  = b / n_sub;

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

    cuFloatComplex sx_c = shift_1d[lx];
    cuFloatComplex sy_c = shift_1d[ly];
    float          sr   = sx_c.x * sy_c.x - sx_c.y * sy_c.y;
    float          sm   = sx_c.x * sy_c.y + sx_c.y * sy_c.x;

    patches[idx] = make_cuFloatComplex(pr * sr - pi * sm, pr * sm + pi * sr);
}

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
