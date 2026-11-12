// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_cuda.cu
 * @brief   CUDA kernels for stateless Philox PRNG, detector readout, and complex pupil synthesis
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdint.h>

#include "common/cacaosim_cuda.h"

/**
 * philox2x32_10() - Counter-based 10-round Philox 2x32 stateless PRNG on GPU
 */
__device__ __forceinline__ uint2 philox2x32_10(
    uint2    ctr,
    uint32_t key)
{
    const uint32_t M = 0xd256d193u;
    const uint32_t W = 0x9e3779b9u;

#pragma unroll
    for (int r = 0; r < 10; r++)
    {
        uint32_t hi = __umulhi(M, ctr.x);
        uint32_t lo = M * ctr.x;
        ctr.x       = hi ^ key ^ ctr.y;
        ctr.y       = lo;
        key += W;
    }
    return ctr;
}

__device__ __forceinline__ float gpu_u32_to_uniform(uint32_t u)
{
    return ((float) (u >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float2 gpu_box_muller(uint2 u)
{
    float u1  = gpu_u32_to_uniform(u.x);
    float u2  = gpu_u32_to_uniform(u.y);
    float rad = sqrtf(-2.0f * __logf(u1));
    float s, c;
    __sincosf(CACAOSIM_TWO_PI_F * u2, &s, &c);
    return make_float2(rad * c, rad * s);
}

__device__ __forceinline__ float gpu_sample_poisson(
    float    mu,
    uint64_t pixel_id,
    uint64_t frame_seq,
    uint32_t base_key,
    float    z_norm)
{
    if (mu <= 0.0f)
    {
        return 0.0f;
    }
    if (mu < 24.0f)
    {
        float    L    = __expf(-mu);
        float    p    = 1.0f;
        uint32_t k    = 0u;
        uint32_t step = 1u;
        do
        {
            uint2 ctr = make_uint2((uint32_t) pixel_id ^ (step * 0x85ebca6bu),
                                   (uint32_t) frame_seq + step);
            uint2 rnd = philox2x32_10(ctr, base_key);
            k++;
            p *= gpu_u32_to_uniform(rnd.x);
            if (p <= L)
            {
                break;
            }
            k++;
            p *= gpu_u32_to_uniform(rnd.y);
            step++;
        } while (p > L && step < 128u);
        return (float) (k - 1u);
    }

    float val = mu + sqrtf(mu) * z_norm + (1.0f / 6.0f) * (z_norm * z_norm - 1.0f) + 0.5f;
    return (val > 0.0f) ? floorf(val) : 0.0f;
}

__global__ void kernel_build_pupil_padded(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    float                     wf_to_opd,
    float                     opd_to_rad,
    uint32_t                  nwf,
    uint32_t                  nfft,
    uint32_t                  pad_off,
    cuFloatComplex *__restrict__ out_cplx)
{
    uint32_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = nfft * nfft;
    if (idx >= total)
    {
        return;
    }

    uint32_t ox = idx % nfft;
    uint32_t oy = idx / nfft;

    if (ox < pad_off || oy < pad_off || ox >= pad_off + nwf || oy >= pad_off + nwf)
    {
        out_cplx[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    uint32_t pix = (oy - pad_off) * nwf + (ox - pad_off);
    float    amp = pup_amp[pix];
    if (amp == 0.0f)
    {
        out_cplx[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }
    if (atm_amp != NULL)
    {
        amp *= atm_amp[pix];
    }

    float opd = 0.0f;
    if (wf_in != NULL)
    {
        opd += wf_to_opd * wf_in[pix];
    }
    if (dm_opd != NULL)
    {
        opd -= dm_opd[pix];
    }
    if (ncpa_opd != NULL)
    {
        opd += ncpa_opd[pix];
    }

    float s, c;
    __sincosf(opd * opd_to_rad, &s, &c);
    out_cplx[idx] = make_cuFloatComplex(amp * c, amp * s);
}

__global__ void kernel_detector_readout(
    const float *__restrict__ opt_int,
    uint64_t                  npix,
    float                     flux_scale,
    float                     dark_e,
    float                     ron_e,
    float                     inv_gain,
    float                     bias_adu,
    float                     sat_adu,
    int32_t                   raw_passthrough,
    int32_t                   do_poiss,
    int32_t                   do_ron,
    int32_t                   out_dtype,
    uint32_t                  base_key,
    uint64_t                  frame_seq,
    uint16_t *__restrict__    out_u16,
    float *__restrict__       out_f32)
{
    uint64_t idx = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix)
    {
        return;
    }

    float sig = opt_int[idx];
    if (!raw_passthrough && sig < 0.0f)
    {
        sig = 0.0f;
    }

    float phe = sig * flux_scale + dark_e;

    if (do_poiss || do_ron)
    {
        uint2  ctr = make_uint2((uint32_t) idx,
                                (uint32_t) (frame_seq * 0x9e3779b9u) ^ (uint32_t) (idx >> 32));
        uint2  rnd = philox2x32_10(ctr, base_key);
        float2 g2  = gpu_box_muller(rnd);

        if (do_poiss)
        {
            phe = gpu_sample_poisson(phe, idx, frame_seq, base_key ^ 0xa5a5a5a5u, g2.x);
        }
        if (do_ron)
        {
            phe += ron_e * g2.y;
        }
    }

    float adu = phe * inv_gain + bias_adu;

    if (out_dtype == 0 && out_u16 != NULL)
    {
        float max_u16 = (sat_adu < 65535.0f) ? sat_adu : 65535.0f;
        if (adu < 0.0f)
        {
            adu = 0.0f;
        }
        else if (adu > max_u16)
        {
            adu = max_u16;
        }
        out_u16[idx] = (uint16_t) (adu + 0.5f);
    }
    else if (out_f32 != NULL)
    {
        if (!raw_passthrough)
        {
            if (adu < 0.0f)
            {
                adu = 0.0f;
            }
            else if (adu > sat_adu)
            {
                adu = sat_adu;
            }
        }
        out_f32[idx] = adu;
    }
}

extern "C" int cacaosim_cuda_available(void)
{
    static int cached = -1;
    if (cached >= 0)
    {
        return cached;
    }
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    cached          = (err == cudaSuccess && count > 0) ? 1 : 0;
    return cached;
}

extern "C" void *cacaosim_cuda_stream_create(void)
{
    if (!cacaosim_cuda_available())
    {
        return NULL;
    }
    cudaStream_t st = NULL;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
    {
        return NULL;
    }
    return (void *) st;
}

extern "C" void cacaosim_cuda_stream_destroy(void *cuda_stream)
{
    if (cuda_stream != NULL)
    {
        cudaStreamDestroy((cudaStream_t) cuda_stream);
    }
}

extern "C" void cacaosim_cuda_stream_sync(void *cuda_stream)
{
    cudaStreamSynchronize((cudaStream_t) cuda_stream);
}

extern "C" float *cacaosim_cuda_alloc_f32(uint64_t nelem)
{
    if (nelem == 0 || !cacaosim_cuda_available())
    {
        return NULL;
    }
    float *d_ptr = NULL;
    size_t bytes = (size_t) nelem * sizeof(float);
    if (cudaMalloc((void **) &d_ptr, bytes) != cudaSuccess)
    {
        return NULL;
    }
    cudaMemset(d_ptr, 0, bytes);
    return d_ptr;
}

extern "C" void cacaosim_cuda_free_f32(float *d_ptr)
{
    if (d_ptr != NULL)
    {
        cudaFree(d_ptr);
    }
}

extern "C" void cacaosim_cuda_h2d_f32(
    float       *d_dst,
    const float *h_src,
    uint64_t     nelem,
    void        *cuda_stream)
{
    if (d_dst != NULL && h_src != NULL && nelem > 0)
    {
        cudaMemcpyAsync(d_dst, h_src, (size_t) nelem * sizeof(float), cudaMemcpyHostToDevice,
                        (cudaStream_t) cuda_stream);
    }
}

extern "C" void cacaosim_cuda_d2h_f32(
    float       *h_dst,
    const float *d_src,
    uint64_t     nelem,
    void        *cuda_stream)
{
    if (h_dst != NULL && d_src != NULL && nelem > 0)
    {
        cudaMemcpyAsync(h_dst, d_src, (size_t) nelem * sizeof(float), cudaMemcpyDeviceToHost,
                        (cudaStream_t) cuda_stream);
    }
}

extern "C" void cacaosim_cuda_build_pupil_padded(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_dm_opd,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    float        opd_to_rad,
    uint32_t     nwf,
    uint32_t     nfft,
    void        *d_out_cplx,
    void        *cuda_stream)
{
    uint32_t     total   = nfft * nfft;
    uint32_t     pad_off = (nfft >= nwf) ? ((nfft - nwf) / 2u) : 0u;
    uint32_t     threads = 256u;
    uint32_t     blocks  = (total + threads - 1u) / threads;
    cudaStream_t st      = (cudaStream_t) cuda_stream;

    kernel_build_pupil_padded<<<blocks, threads, 0, st>>>(
        d_pup_amp, d_atm_amp, d_wf_in, d_dm_opd, d_ncpa_opd, wf_to_opd, opd_to_rad, nwf, nfft,
        pad_off, (cuFloatComplex *) d_out_cplx);
}

extern "C" void cacaosim_cuda_detector_readout(
    const cacaosim_det_cfg_t *cfg,
    const float              *d_opt_int,
    uint64_t                  npix,
    float                     norm_total,
    uint64_t                  seed,
    uint64_t                  frame_seq,
    uint16_t                 *d_out_u16,
    float                    *d_out_f32,
    void                     *cuda_stream)
{
    int32_t raw_pass = (cfg->flux_ph < 0.0f) ? 1 : 0;
    int32_t do_poiss = (cfg->photon_noise != 0 && !raw_pass) ? 1 : 0;
    int32_t do_ron   = (cfg->ron_e > 0.0f && !raw_pass) ? 1 : 0;
    float   dark_e   = (!raw_pass) ? cfg->dark_e : 0.0f;
    float   inv_gain = (cfg->gain_eadu > 1e-6f && !raw_pass) ? (1.0f / cfg->gain_eadu) : 1.0f;
    float   bias_adu = (!raw_pass) ? cfg->bias_adu : 0.0f;
    float   sat_adu  = (cfg->sat_adu > 0.0f) ? cfg->sat_adu : 65535.0f;

    float flux_scale = 1.0f;
    if (!cfg->shutter)
    {
        flux_scale = 0.0f;
    }
    else if (!raw_pass)
    {
        flux_scale = (norm_total > 1e-20f) ? (cfg->flux_ph / norm_total) : cfg->flux_ph;
    }

    uint32_t     base_key = (uint32_t) (seed ^ (seed >> 32) ^ 0x517cc1b7u);
    uint32_t     threads  = 256u;
    uint32_t     blocks   = (uint32_t) ((npix + threads - 1ULL) / threads);
    cudaStream_t st       = (cudaStream_t) cuda_stream;

    kernel_detector_readout<<<blocks, threads, 0, st>>>(
        d_opt_int, npix, flux_scale, dark_e, cfg->ron_e, inv_gain, bias_adu, sat_adu, raw_pass,
        do_poiss, do_ron, cfg->out_dtype, base_key, frame_seq, d_out_u16, d_out_f32);
}

__global__ void kernel_beam_init(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    const float *__restrict__ ncpa_opd,
    float                     wf_to_opd,
    uint32_t                  npix,
    float *__restrict__       out_opd,
    float *__restrict__       out_amp)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix)
    {
        return;
    }
    float a = (pup_amp != NULL) ? pup_amp[idx] : 1.0f;
    if (atm_amp != NULL)
    {
        a *= atm_amp[idx];
    }
    float o = (wf_in != NULL) ? (wf_to_opd * wf_in[idx]) : 0.0f;
    if (ncpa_opd != NULL)
    {
        o += ncpa_opd[idx];
    }
    out_opd[idx] = o;
    out_amp[idx] = a;
}

__global__ void kernel_beam_apply_dm(
    const float *__restrict__ in_opd,
    const float *__restrict__ in_amp,
    const float *__restrict__ dm_opd,
    uint32_t                  npix,
    float *__restrict__       out_opd,
    float *__restrict__       out_amp)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix)
    {
        return;
    }
    float o      = (in_opd != NULL) ? in_opd[idx] : 0.0f;
    float d      = (dm_opd != NULL) ? dm_opd[idx] : 0.0f;
    out_opd[idx] = o - d;
    if (out_amp != NULL && out_amp != in_amp)
    {
        out_amp[idx] = (in_amp != NULL) ? in_amp[idx] : 1.0f;
    }
}

__global__ void kernel_beam_relay_ident(
    const float *__restrict__ in_opd,
    const float *__restrict__ in_amp,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ pup_mask,
    uint32_t                  npix,
    float *__restrict__       out_opd,
    float *__restrict__       out_amp)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix)
    {
        return;
    }
    float o = (in_opd != NULL) ? in_opd[idx] : 0.0f;
    float a = (in_amp != NULL) ? in_amp[idx] : 1.0f;
    if (ncpa_opd != NULL)
    {
        o += ncpa_opd[idx];
    }
    if (pup_mask != NULL)
    {
        a *= pup_mask[idx];
    }
    out_opd[idx] = o;
    out_amp[idx] = a;
}

__global__ void kernel_beam_relay_warp(
    const float *__restrict__ in_opd,
    const float *__restrict__ in_amp,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ pup_mask,
    float                     c,
    float                     s,
    float                     dx,
    float                     dy,
    uint32_t                  nwf,
    float *__restrict__       out_opd,
    float *__restrict__       out_amp)
{
    uint32_t idx  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t npix = nwf * nwf;
    if (idx >= npix)
    {
        return;
    }
    uint32_t ix  = idx % nwf;
    uint32_t iy  = idx / nwf;
    float    ctr = 0.5f * (float) (nwf - 1u);
    float    vx  = (float) ix - ctr - dx;
    float    vy  = (float) iy - ctr - dy;
    float    sx  = ctr + c * vx - s * vy;
    float    sy  = ctr + s * vx + c * vy;

    float o = 0.0f;
    float a = 0.0f;
    if (sx >= 0.0f && sy >= 0.0f && sx <= (float) (nwf - 1u) && sy <= (float) (nwf - 1u))
    {
        uint32_t x0  = (uint32_t) sx;
        uint32_t y0  = (uint32_t) sy;
        uint32_t x1  = (x0 + 1u < nwf) ? (x0 + 1u) : x0;
        uint32_t y1  = (y0 + 1u < nwf) ? (y0 + 1u) : y0;
        float    fx  = sx - (float) x0;
        float    fy  = sy - (float) y0;
        float    w00 = (1.0f - fx) * (1.0f - fy);
        float    w10 = fx * (1.0f - fy);
        float    w01 = (1.0f - fx) * fy;
        float    w11 = fx * fy;

        uint32_t i00 = y0 * nwf + x0;
        uint32_t i10 = y0 * nwf + x1;
        uint32_t i01 = y1 * nwf + x0;
        uint32_t i11 = y1 * nwf + x1;

        if (in_opd != NULL)
        {
            o = w00 * in_opd[i00] + w10 * in_opd[i10] + w01 * in_opd[i01] + w11 * in_opd[i11];
        }
        a = (in_amp != NULL)
                ? (w00 * in_amp[i00] + w10 * in_amp[i10] + w01 * in_amp[i01] + w11 * in_amp[i11])
                : 1.0f;
    }
    if (ncpa_opd != NULL)
    {
        o += ncpa_opd[idx];
    }
    if (pup_mask != NULL)
    {
        a *= pup_mask[idx];
    }
    out_opd[idx] = o;
    out_amp[idx] = a;
}

extern "C" void cacaosim_cuda_beam_init(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream)
{
    uint32_t     npix    = nwf * nwf;
    uint32_t     threads = 256u;
    uint32_t     blocks  = (npix + threads - 1u) / threads;
    cudaStream_t st      = (cudaStream_t) cuda_stream;

    kernel_beam_init<<<blocks, threads, 0, st>>>(
        d_pup_amp, d_atm_amp, d_wf_in, d_ncpa_opd, wf_to_opd, npix, d_out_opd, d_out_amp);
}

extern "C" void cacaosim_cuda_beam_apply_dm(
    const float *d_in_opd,
    const float *d_in_amp,
    const float *d_dm_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream)
{
    uint32_t     npix    = nwf * nwf;
    uint32_t     threads = 256u;
    uint32_t     blocks  = (npix + threads - 1u) / threads;
    cudaStream_t st      = (cudaStream_t) cuda_stream;

    kernel_beam_apply_dm<<<blocks, threads, 0, st>>>(
        d_in_opd, d_in_amp, d_dm_opd, npix, d_out_opd, d_out_amp);
}

extern "C" void cacaosim_cuda_beam_relay(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_ncpa_opd,
    const float          *d_pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *d_out_opd,
    float                *d_out_amp,
    void                 *cuda_stream)
{
    uint32_t     npix    = nwf * nwf;
    uint32_t     threads = 256u;
    uint32_t     blocks  = (npix + threads - 1u) / threads;
    cudaStream_t st      = (cudaStream_t) cuda_stream;

    if (cacaosim_reg_is_identity(reg))
    {
        kernel_beam_relay_ident<<<blocks, threads, 0, st>>>(
            d_in_opd, d_in_amp, d_ncpa_opd, d_pup_mask, npix, d_out_opd, d_out_amp);
    }
    else
    {
        float sc   = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
        float invs = 1.0f / sc;
        float rad  = -reg->rot_deg * (CACAOSIM_PI_F / 180.0f);
        float c    = cosf(rad) * invs;
        float s    = sinf(rad) * invs;
        kernel_beam_relay_warp<<<blocks, threads, 0, st>>>(
            d_in_opd, d_in_amp, d_ncpa_opd, d_pup_mask, c, s, reg->shift_x, reg->shift_y, nwf,
            d_out_opd, d_out_amp);
    }
}
