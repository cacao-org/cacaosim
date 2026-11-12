// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs_cuda.cu
 * @brief   CUDA cuFFT + CUDA Graph Pyramid WFS propagation and multi-subframe integration kernels
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>

#include "common/cacaosim_cuda.h"
#include "pyrwfs/cacaosim_pyrwfs.h"

typedef struct
{
    cudaStream_t     stream;
    int              owns_stream;
    cufftHandle      plan_fwd;
    cufftHandle      plan_inv_batch;
    int              has_fwd;
    int              has_inv;

    float           *d_pup_amp;
    float           *d_dyn_pup;
    float           *d_ncpa_opd;
    float           *d_atm_amp;
    float           *d_wf_in;
    float           *d_dm_opd;
    cuFloatComplex  *d_masks;
    cuFloatComplex  *d_pup_cplx;
    cuFloatComplex  *d_foc_cplx;
    cuFloatComplex  *d_mod_batch;
    float           *d_cam_int;
    float           *d_accum_int;
    uint16_t        *d_out_u16;
    float           *d_out_f32;

    /* CUDA Graph cached for steady-state optical propagation */
    cudaGraph_t      graph;
    cudaGraphExec_t  graph_exec;
    int              graph_valid;
    int              graph_tried;
    const float     *graph_pup_ptr;
    const float     *graph_amp_ptr;
    const float     *graph_wf_ptr;
    const float     *graph_dm_ptr;
    float            graph_wf_scale;
} pyrwfs_cuda_state_t;

__global__ void kernel_pyrwfs_mod_mul(
    const cuFloatComplex *__restrict__ foc,
    const cuFloatComplex *__restrict__ masks,
    uint64_t                           npix_fft,
    int32_t                            npts,
    cuFloatComplex *__restrict__       batch_out)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix_fft)
    {
        return;
    }

    cuFloatComplex e = foc[p];
    for (int32_t k = 0; k < npts; k++)
    {
        uint64_t       idx = (uint64_t) k * npix_fft + p;
        cuFloatComplex m   = masks[idx];
        batch_out[idx] = make_cuFloatComplex(e.x * m.x - e.y * m.y, e.x * m.y + e.y * m.x);
    }
}

__global__ void kernel_pyrwfs_accum_bin_int(
    const cuFloatComplex *__restrict__ batch_det,
    uint32_t                           nfft,
    uint32_t                           ncam,
    uint32_t                           bin,
    int32_t                            npts,
    float *__restrict__                cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx       = cid % ncam;
    uint32_t cy       = cid / ncam;
    uint32_t x0       = cx * bin;
    uint32_t y0       = cy * bin;
    uint64_t npix_fft = (uint64_t) nfft * nfft;
    float    acc      = 0.0f;

    for (uint32_t by = 0; by < bin; by++)
    {
        uint64_t row_off = (uint64_t) (y0 + by) * nfft + x0;
        for (uint32_t bx = 0; bx < bin; bx++)
        {
            uint64_t pix = row_off + bx;
            for (int32_t k = 0; k < npts; k++)
            {
                cuFloatComplex v = batch_det[(uint64_t) k * npix_fft + pix];
                acc += v.x * v.x + v.y * v.y;
            }
        }
    }
    cam_int[cid] = acc;
}

__global__ void kernel_pyrwfs_accum_bin_frac(
    const cuFloatComplex *__restrict__ batch_det,
    uint32_t                           nfft,
    uint32_t                           ncam,
    int32_t                            npts,
    float *__restrict__                cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx       = cid % ncam;
    uint32_t cy       = cid / ncam;
    float    scale    = (float) nfft / (float) ncam;
    float    fx0      = (float) cx * scale;
    float    fx1      = (float) (cx + 1u) * scale;
    float    fy0      = (float) cy * scale;
    float    fy1      = (float) (cy + 1u) * scale;
    int      ix0      = (int) floorf(fx0);
    int      ix1      = (int) ceilf(fx1);
    int      iy0      = (int) floorf(fy0);
    int      iy1      = (int) ceilf(fy1);
    uint64_t npix_fft = (uint64_t) nfft * nfft;
    float    acc      = 0.0f;

    for (int iy = iy0; iy < iy1; iy++)
    {
        if (iy < 0 || iy >= (int) nfft)
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
            if (ix < 0 || ix >= (int) nfft)
            {
                continue;
            }
            float wx0 = (fx0 > (float) ix) ? fx0 : (float) ix;
            float wx1 = (fx1 < (float) (ix + 1)) ? fx1 : (float) (ix + 1);
            float wx  = wx1 - wx0;
            if (wx > 0.0f)
            {
                uint64_t pix = (uint64_t) iy * nfft + (uint32_t) ix;
                float    p_i = 0.0f;
                for (int32_t k = 0; k < npts; k++)
                {
                    cuFloatComplex v = batch_det[(uint64_t) k * npix_fft + pix];
                    p_i += v.x * v.x + v.y * v.y;
                }
                acc += wx * wy * p_i;
            }
        }
    }
    cam_int[cid] = acc;
}

__global__ void kernel_pyrwfs_add_accum(
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

__global__ void kernel_pyrwfs_scale_accum(
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

static void pyrwfs_cuda_invalidate_graph(pyrwfs_cuda_state_t *cs)
{
    if (cs->graph_valid)
    {
        cudaGraphExecDestroy(cs->graph_exec);
        cudaGraphDestroy(cs->graph);
        cs->graph_valid = 0;
    }
    cs->graph_tried = 0;
}

static void pyrwfs_cuda_launch_optics(
    cacaosim_pyrwfs_ctx_t *ctx,
    pyrwfs_cuda_state_t   *cs,
    const float           *d_pup_ptr,
    const float           *d_atm_amp_ptr,
    const float           *d_wf_ptr,
    const float           *d_dm_ptr,
    float                  wf_to_opd)
{
    float    lam        = (ctx->cfg.lambda_um > 0.1f) ? ctx->cfg.lambda_um : 0.75f;
    float    opd_to_rad = CACAOSIM_TWO_PI_F / lam;
    uint32_t nfft       = ctx->nfft;
    uint32_t ncam       = ctx->cfg.ncam;
    int32_t  npts       = ctx->eff_npts;

    cacaosim_cuda_build_pupil_padded(
        d_pup_ptr, d_atm_amp_ptr, d_wf_ptr, d_dm_ptr, cs->d_ncpa_opd, wf_to_opd, opd_to_rad,
        ctx->cfg.nwf, nfft, cs->d_pup_cplx, cs->stream);

    cufftExecC2C(cs->plan_fwd, cs->d_pup_cplx, cs->d_foc_cplx, CUFFT_FORWARD);

    uint32_t th_fft = 256u;
    uint32_t bl_fft = (uint32_t) ((ctx->npix_fft + th_fft - 1ULL) / th_fft);
    kernel_pyrwfs_mod_mul<<<bl_fft, th_fft, 0, cs->stream>>>(
        cs->d_foc_cplx, cs->d_masks, ctx->npix_fft, npts, cs->d_mod_batch);

    cufftExecC2C(cs->plan_inv_batch, cs->d_mod_batch, cs->d_mod_batch, CUFFT_INVERSE);

    uint32_t th_cam = 256u;
    uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
    if (nfft >= ncam && (nfft % ncam) == 0u)
    {
        kernel_pyrwfs_accum_bin_int<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_mod_batch, nfft, ncam, nfft / ncam, npts, cs->d_cam_int);
    }
    else
    {
        kernel_pyrwfs_accum_bin_frac<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_mod_batch, nfft, ncam, npts, cs->d_cam_int);
    }
}

extern "C" int cacaosim_pyrwfs_cuda_init(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *ncpa_opd,
    void                  *ext_stream)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) calloc(1, sizeof(pyrwfs_cuda_state_t));
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
            cacaosim_pyrwfs_cuda_free(ctx);
            return -1;
        }
        cs->owns_stream = 1;
    }

    size_t wf_bytes    = (size_t) ctx->npix_wf * sizeof(float);
    size_t fft_c_bytes = (size_t) ctx->npix_fft * sizeof(cuFloatComplex);
    size_t bat_c_bytes = (size_t) ctx->eff_npts * fft_c_bytes;
    size_t cam_f_bytes = (size_t) ctx->npix_cam * sizeof(float);
    size_t cam_u_bytes = (size_t) ctx->npix_cam * sizeof(uint16_t);

    if (cudaMalloc((void **) &cs->d_pup_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dyn_pup, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_atm_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_wf_in, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dm_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_masks, bat_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_pup_cplx, fft_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_foc_cplx, fft_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_mod_batch, bat_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_cam_int, cam_f_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_accum_int, cam_f_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_u16, cam_u_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_f32, cam_f_bytes) != cudaSuccess)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
        return -1;
    }

    if (ncpa_opd != NULL)
    {
        if (cudaMalloc((void **) &cs->d_ncpa_opd, wf_bytes) != cudaSuccess)
        {
            cacaosim_pyrwfs_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_ncpa_opd, ncpa_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    }

    cudaMemcpyAsync(cs->d_pup_amp, pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_masks, ctx->masks_cplx, bat_c_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemsetAsync(cs->d_wf_in, 0, wf_bytes, cs->stream);
    cudaMemsetAsync(cs->d_dm_opd, 0, wf_bytes, cs->stream);

    int n[2] = { (int) ctx->nfft, (int) ctx->nfft };
    if (cufftPlan2d(&cs->plan_fwd, n[0], n[1], CUFFT_C2C) != CUFFT_SUCCESS)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
        return -1;
    }
    cs->has_fwd = 1;
    cufftSetStream(cs->plan_fwd, cs->stream);

    int inembed[2] = { n[0], n[1] };
    int idist      = (int) ctx->npix_fft;
    if (cufftPlanMany(&cs->plan_inv_batch, 2, n, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C,
                      ctx->eff_npts) != CUFFT_SUCCESS)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
        return -1;
    }
    cs->has_inv = 1;
    cufftSetStream(cs->plan_inv_batch, cs->stream);

    /* Warmup run so cuFFT internal workspaces are bound before any CUDA Graph capture */
    pyrwfs_cuda_launch_optics(ctx, cs, cs->d_pup_amp, NULL, NULL, NULL, 1.0f);
    cudaStreamSynchronize(cs->stream);
    return 0;
}

extern "C" void cacaosim_pyrwfs_cuda_upload_masks(cacaosim_pyrwfs_ctx_t *ctx)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL || cs->d_masks == NULL)
    {
        return;
    }
    pyrwfs_cuda_invalidate_graph(cs);
    size_t bat_c_bytes = (size_t) ctx->eff_npts * ctx->npix_fft * sizeof(cuFloatComplex);
    cudaMemcpyAsync(cs->d_masks, ctx->masks_cplx, bat_c_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaStreamSynchronize(cs->stream);
}

extern "C" void cacaosim_pyrwfs_cuda_free(cacaosim_pyrwfs_ctx_t *ctx)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL)
    {
        return;
    }
    pyrwfs_cuda_invalidate_graph(cs);
    if (cs->has_fwd)
    {
        cufftDestroy(cs->plan_fwd);
    }
    if (cs->has_inv)
    {
        cufftDestroy(cs->plan_inv_batch);
    }
    cudaFree(cs->d_pup_amp);
    cudaFree(cs->d_dyn_pup);
    cudaFree(cs->d_ncpa_opd);
    cudaFree(cs->d_atm_amp);
    cudaFree(cs->d_wf_in);
    cudaFree(cs->d_dm_opd);
    cudaFree(cs->d_masks);
    cudaFree(cs->d_pup_cplx);
    cudaFree(cs->d_foc_cplx);
    cudaFree(cs->d_mod_batch);
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

extern "C" void cacaosim_pyrwfs_cuda_accum(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *dyn_pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *d_pup_amp_ext,
    const float           *d_wf_opd_ext,
    const float           *d_dm_opd_ext,
    float                  wf_to_opd)
{
    pyrwfs_cuda_state_t *cs       = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    size_t               wf_bytes = (size_t) ctx->npix_wf * sizeof(float);

    const float *d_pup_ptr = cs->d_pup_amp;
    if (d_pup_amp_ext != NULL)
    {
        d_pup_ptr = d_pup_amp_ext;
    }
    else if (dyn_pup_amp != NULL)
    {
        cudaMemcpyAsync(cs->d_dyn_pup, dyn_pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_pup_ptr = cs->d_dyn_pup;
    }

    const float *d_amp_ptr = NULL;
    if (atm_amp != NULL)
    {
        cudaMemcpyAsync(cs->d_atm_amp, atm_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_amp_ptr = cs->d_atm_amp;
    }

    const float *d_wf_ptr = NULL;
    if (d_wf_opd_ext != NULL)
    {
        d_wf_ptr = d_wf_opd_ext;
    }
    else if (wf_in != NULL && wf_to_opd != 0.0f)
    {
        cudaMemcpyAsync(cs->d_wf_in, wf_in, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_wf_ptr = cs->d_wf_in;
    }

    const float *d_dm_ptr = NULL;
    if (d_dm_opd_ext != NULL)
    {
        d_dm_ptr = d_dm_opd_ext;
    }
    else if (dm_opd != NULL)
    {
        cudaMemcpyAsync(cs->d_dm_opd, dm_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_dm_ptr = cs->d_dm_opd;
    }

    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    /* Re-validate or capture steady-state optical CUDA Graph when pointer signature is fixed */
    if (cs->graph_valid &&
        (cs->graph_pup_ptr != d_pup_ptr || cs->graph_amp_ptr != d_amp_ptr ||
         cs->graph_wf_ptr != d_wf_ptr || cs->graph_dm_ptr != d_dm_ptr ||
         fabsf(cs->graph_wf_scale - wf_to_opd) > 1e-7f))
    {
        pyrwfs_cuda_invalidate_graph(cs);
    }

    if (!cs->graph_valid && !cs->graph_tried)
    {
        pyrwfs_cuda_launch_optics(ctx, cs, d_pup_ptr, d_amp_ptr, d_wf_ptr, d_dm_ptr, wf_to_opd);
        cudaStreamSynchronize(cs->stream);

        cs->graph_tried = 1;
        if (cudaStreamBeginCapture(cs->stream, cudaStreamCaptureModeRelaxed) == cudaSuccess)
        {
            pyrwfs_cuda_launch_optics(ctx, cs, d_pup_ptr, d_amp_ptr, d_wf_ptr, d_dm_ptr, wf_to_opd);
            if (cudaStreamEndCapture(cs->stream, &cs->graph) == cudaSuccess)
            {
                if (cudaGraphInstantiate(&cs->graph_exec, cs->graph, NULL, NULL, 0) == cudaSuccess)
                {
                    cs->graph_valid    = 1;
                    cs->graph_pup_ptr  = d_pup_ptr;
                    cs->graph_amp_ptr  = d_amp_ptr;
                    cs->graph_wf_ptr   = d_wf_ptr;
                    cs->graph_dm_ptr   = d_dm_ptr;
                    cs->graph_wf_scale = wf_to_opd;
                }
                else
                {
                    cudaGraphDestroy(cs->graph);
                    cudaGetLastError();
                }
            }
            else
            {
                cudaGetLastError();
            }
        }
        else
        {
            cudaGetLastError();
        }
    }
    else if (cs->graph_valid)
    {
        cudaGraphLaunch(cs->graph_exec, cs->stream);
    }
    else
    {
        pyrwfs_cuda_launch_optics(ctx, cs, d_pup_ptr, d_amp_ptr, d_wf_ptr, d_dm_ptr, wf_to_opd);
    }

    if (ctx->subframe_cnt >= 1u)
    {
        uint32_t th_cam = 256u;
        uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
        kernel_pyrwfs_add_accum<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_cam_int, ctx->npix_cam, cs->d_accum_int);
    }
}

extern "C" void cacaosim_pyrwfs_cuda_readout(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;

    if (ctx->subframe_cnt > 1u)
    {
        uint32_t th_cam = 256u;
        uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
        kernel_pyrwfs_scale_accum<<<bl_cam, th_cam, 0, cs->stream>>>(
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
