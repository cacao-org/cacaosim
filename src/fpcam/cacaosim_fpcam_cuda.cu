// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam_cuda.cu
 * @brief   cuBLAS StridedBatched MFT + cuFFT 2D focal-plane camera GPU backend
 *
 * Architectural Role in `cacaosim`:
 *   Implements the CUDA GPU compute backend for the polychromatic focal-plane science camera
 *   (`cacaosim.fpcam`). Executes batched pupil complex phasor synthesis, Soummer Matrix-DFT
 *   (`cublasCgemmStridedBatched`) or zero-padded 2D FFT (`cufftExecC2C`) Fraunhofer propagation,
 *   multi-subframe incoherent broadband intensity integration, and GPU Philox-PRNG detector
 *   readout (`cacaosim_cuda_detector_readout`). Supports both standalone SHM execution via
 *   asynchronous H2D transfers and zero-copy fused GPU benches (`cacaosim.htsim`) via borrowed
 *   `cudaStream_t` (`ext_stream`) and resident device `[OPD, Amp]` pointers.
 *
 * Design Rationale & GPU Numerical Strategy:
 *   1. Zero-Transpose Row-Major MFT via Column-Major `cublasCgemmStridedBatched`:
 *      - Host-precomputed MFT basis matrices `U_c [ncam x nwf]` and `V_c [nwf x ncam]` and pupil
 *        fields `Epup_c [nwf x nwf]` are stored in C row-major memory layout, whereas cuBLAS
 *        interprets pointers as Fortran column-major matrices.
 *      - Using the exact transpose equivalence `(A * B)_row == (B^T * A^T)_col`, row-major
 *        `T = U * Epup` maps directly to column-major `T^T [nwf x ncam] = Epup^T * U^T`, and
 *        `Efoc = T * V` maps to `Efoc^T [ncam x ncam] = V^T * T^T` with `CUBLAS_OP_N` and zero
 *        matrix transpose kernels.
 *      - Batching all `eff_nlam` spectral channels into just two `cublasCgemmStridedBatched`
 *        invocations saturates GPU streaming multiprocessors on compact `128 x 128` grids where
 *        serial per-wavelength GEMM launches would be bottlenecked by host launch overhead.
 *   2. Shift-Invariant Modular FFT Cropping (`PROP_FFT`):
 *      - In wide-field `PROP_FFT` mode, `cacaosim_cuda_build_pupil_padded()` places `Epup` in the
 *        upper-left `[nwf x nwf]` quadrant of `[nfft x nfft]`, and `k_fpcam_accum_fft_crop` wraps
 *        signed camera coordinates `(dx, dy)` modularly around DFT DC bin `(0, 0)` onto camera
 *        center `(ncam / 2, ncam / 2)`. Because `|F(kx, ky)|^2` is invariant to spatial shifts of
 *        the pupil origin, no pre- or post-FFT checkerboard phase modulation is needed.
 *   3. Strided Polychromatic DMA Staging (`cudaMemcpy2DAsync`):
 *      - Upstream polychromatic SHM beams pack `[OPD_0, Amp_0, OPD_1, Amp_1, ...]` with channel
 *        stride `2 * npix_wf`. When `beam_nlam == eff_nlam`, `cudaMemcpy2DAsync` gathers all `OPD`
 *        or `Amp` slices into contiguous `[nlam x npix_wf]` device memory in a single PCIe DMA
 *        command rather than issuing `nlam` separate `cudaMemcpyAsync` calls.
 */

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "fpcam/cacaosim_fpcam.h"

typedef struct
{
    uint32_t m0[CACAOSIM_FPCAM_MAX_LAMBDA];
    uint32_t m1[CACAOSIM_FPCAM_MAX_LAMBDA];
    float    alpha[CACAOSIM_FPCAM_MAX_LAMBDA];
} fpcam_poly_map_t;

/**
 * struct fpcam_cuda_priv_t - Opaque resident GPU state owned by `cacaosim_fpcam_ctx_t::cuda_priv`
 * @stream:       Active CUDA execution stream (owned non-blocking stream or borrowed `ext_stream`)
 * @owns_stream:  Non-zero when `@stream` was created internally and must be destroyed on teardown
 * @cublas:       cuBLAS handle bound to `@stream` for `PROP_MFT` batched complex GEMMs
 * @fft_fwd:      2D `[nfft x nfft]` `CUFFT_C2C` plan bound to `@stream` for `PROP_FFT` mode
 * @has_fft:      Non-zero when `@fft_fwd` holds a valid cuFFT plan requiring `cufftDestroy()`
 * @d_pup_amp:    Resident clear reference pupil amplitude `[npix_wf]`
 * @d_dyn_pup:    Staging buffer `[eff_nlam * npix_wf]` for dynamic host beam amplitudes
 * @d_ncpa_opd:   Optional resident static science-arm NCPA OPD `[npix_wf]` [um], or NULL
 * @d_atm_amp:    Staging buffer `[npix_wf]` for host atmospheric scintillation amplitude
 * @d_wf_in:      Staging buffer `[eff_nlam * npix_wf]` for host wavefront or beam OPD slices
 * @d_dm_opd:     Staging buffer `[npix_wf]` for host DM OPD [um]
 * @d_lam_um:     Resident spectral channel wavelengths `[eff_nlam]` [um]
 * @d_mft_U:      Resident row-major 1D MFT Y-basis `[eff_nlam * ncam * nwf]` (`cuComplex`)
 * @d_mft_V:      Resident row-major 1D MFT X-basis `[eff_nlam * nwf * ncam]` (`cuComplex`)
 * @d_epup_batch: Batched complex pupil fields `[eff_nlam * npix_wf]` for `PROP_MFT`
 * @d_mft_tmp:    Batched intermediate `U * Epup` workspace `[eff_nlam * ncam * nwf]`
 * @d_efoc_batch: Batched complex focal fields `[eff_nlam * npix_cam]` output by `T * V`
 * @d_fft_in:     Zero-padded `[nfft * nfft]` complex input workspace for `PROP_FFT`
 * @d_fft_out:    Transformed `[nfft * nfft]` complex output workspace for `PROP_FFT`
 * @d_accum_int:  Running multi-subframe Strehl-normalized optical intensity sum `[npix_cam]`
 * @d_exp_int:    Subframe-averaged noise-free exposure PSF `[npix_cam]` in Strehl units `[0..1]`
 * @d_out_u16:    Device `uint16_t` ADU readout buffer `[npix_cam]` when `out_dtype == 0`
 * @d_out_f32:    Device `float` readout buffer `[npix_cam]` when `out_dtype != 0`
 *
 * Pre-allocated once during `cacaosim_fpcam_cuda_init()` so hot per-subframe accumulation and
 * camera readout perform zero `cudaMalloc()` or plan creation calls.
 */
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
    float         *d_strehl_w;
    float         *d_scale_c;
    float         *d_disp_sx;
    float         *d_disp_sy;

    cuComplex     *d_mft_U;
    cuComplex     *d_mft_V;
    cuComplex     *d_epup_batch;
    cuComplex     *d_mft_tmp;
    cuComplex     *d_efoc_batch;

    cufftComplex  *d_fft_in;
    cufftComplex  *d_fft_out;

    float           *d_accum_int;
    float           *d_exp_int;
    float           *d_long_int;
    float           *d_long_out;
    uint32_t        *d_stats_u32;
    uint32_t        *h_stats_u32;
    int              pinned_stats;
    uint16_t        *d_out_u16;
    float           *d_out_f32;

    cudaGraph_t      graph;
    cudaGraphExec_t  graph_exec;
    int              graph_valid;
    int              graph_tried;
    int              has_sig;
    const float     *sig_pup;
    const float     *sig_amp;
    const float     *sig_wf;
    const float     *sig_dm;
    uint64_t         sig_stride;
    float            sig_scale;
    int32_t          sig_poly_mode;
    fpcam_poly_map_t sig_map;
} fpcam_cuda_priv_t;

__global__ static void k_fpcam_build_epup_batch(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    fpcam_poly_map_t          map,
    uint64_t                  chan_stride,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ lam_um,
    const float *__restrict__ disp_sx,
    const float *__restrict__ disp_sy,
    float                     wf_to_opd,
    uint32_t                  nwf,
    uint64_t                  npix_wf,
    int32_t                   nlam,
    int32_t                   poly_mode,
    cuComplex *__restrict__   epup_batch)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= nlam)
    {
        return;
    }

    uint32_t m0      = map.m0[c];
    uint32_t m1      = map.m1[c];
    float    w1      = map.alpha[c];
    float    w0      = 1.0f - w1;
    uint64_t b0      = (uint64_t) m0 * chan_stride + p;
    uint64_t b1      = (uint64_t) m1 * chan_stride + p;
    float    at      = (atm_amp != NULL) ? atm_amp[p] : 1.0f;
    float    amp0    = ((pup_amp != NULL) ? pup_amp[b0] : 1.0f) * at;
    float    amp1    = ((pup_amp != NULL) ? pup_amp[b1] : 1.0f) * at;
    uint64_t out_idx = (uint64_t) c * npix_wf + p;
    if (amp0 <= 0.0f && (m1 == m0 || amp1 <= 0.0f))
    {
        epup_batch[out_idx] = make_cuComplex(0.0f, 0.0f);
        return;
    }

    uint32_t x        = (uint32_t) (p % nwf);
    uint32_t y        = (uint32_t) (p / nwf);
    float    p0_xy    = 0.5f * (float) (nwf - 1u);
    float    disp_opd = disp_sx[c] * ((float) x - p0_xy) + disp_sy[c] * ((float) y - p0_xy);
    float    common   = ((ncpa_opd != NULL) ? ncpa_opd[p] : 0.0f) -
                        ((dm_opd != NULL) ? dm_opd[p] : 0.0f) + disp_opd;
    float    kc       = CACAOSIM_TWO_PI_F / lam_um[c];

    if (poly_mode == CACAOSIM_POLY_UNWRAPPED_OPD && m1 != m0 && w1 > 0.0f)
    {
        float amp = w0 * amp0 + w1 * amp1;
        if (amp <= 0.0f)
        {
            epup_batch[out_idx] = make_cuComplex(0.0f, 0.0f);
            return;
        }
        float opd0 = (wf_in != NULL) ? (wf_to_opd * wf_in[b0]) : 0.0f;
        float opd1 = (wf_in != NULL) ? (wf_to_opd * wf_in[b1]) : 0.0f;
        float s    = 0.0f;
        float co   = 1.0f;
        __sincosf((w0 * opd0 + w1 * opd1 + common) * kc, &s, &co);
        epup_batch[out_idx] = make_cuComplex(amp * co, amp * s);
        return;
    }

    float p0  = (((wf_in != NULL) ? (wf_to_opd * wf_in[b0]) : 0.0f) + common) * kc;
    float s0  = 0.0f;
    float co0 = 1.0f;
    __sincosf(p0, &s0, &co0);
    float re = w0 * amp0 * co0;
    float im = w0 * amp0 * s0;

    if (m1 != m0 && w1 > 0.0f)
    {
        float p1  = (((wf_in != NULL) ? (wf_to_opd * wf_in[b1]) : 0.0f) + common) * kc;
        float s1  = 0.0f;
        float co1 = 1.0f;
        __sincosf(p1, &s1, &co1);
        re += w1 * amp1 * co1;
        im += w1 * amp1 * s1;
    }
    epup_batch[out_idx] = make_cuComplex(re, im);
}

__global__ static void k_fpcam_build_fft_batch(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    fpcam_poly_map_t          map,
    uint64_t                  chan_stride,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ lam_um,
    const float *__restrict__ disp_sx,
    const float *__restrict__ disp_sy,
    float                     wf_to_opd,
    uint32_t                  nwf,
    uint32_t                  nfft,
    int32_t                   nlam,
    int32_t                   poly_mode,
    cufftComplex *__restrict__ fft_in)
{
    uint64_t idx      = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c        = (int32_t) blockIdx.y;
    uint64_t npix_fft = (uint64_t) nfft * nfft;
    if (idx >= npix_fft || c >= nlam)
    {
        return;
    }

    uint64_t out_idx = (uint64_t) c * npix_fft + idx;
    uint32_t x       = (uint32_t) (idx % nfft);
    uint32_t y       = (uint32_t) (idx / nfft);
    if (x >= nwf || y >= nwf)
    {
        fft_in[out_idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    uint64_t p    = (uint64_t) y * nwf + x;
    uint32_t m0   = map.m0[c];
    uint32_t m1   = map.m1[c];
    float    w1   = map.alpha[c];
    float    w0   = 1.0f - w1;
    uint64_t b0   = (uint64_t) m0 * chan_stride + p;
    uint64_t b1   = (uint64_t) m1 * chan_stride + p;
    float    at   = (atm_amp != NULL) ? atm_amp[p] : 1.0f;
    float    amp0 = ((pup_amp != NULL) ? pup_amp[b0] : 1.0f) * at;
    float    amp1 = ((pup_amp != NULL) ? pup_amp[b1] : 1.0f) * at;
    if (amp0 <= 0.0f && (m1 == m0 || amp1 <= 0.0f))
    {
        fft_in[out_idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    float p0_xy    = 0.5f * (float) (nwf - 1u);
    float disp_opd = disp_sx[c] * ((float) x - p0_xy) + disp_sy[c] * ((float) y - p0_xy);
    float common   = ((ncpa_opd != NULL) ? ncpa_opd[p] : 0.0f) -
                     ((dm_opd != NULL) ? dm_opd[p] : 0.0f) + disp_opd;
    float kc       = CACAOSIM_TWO_PI_F / lam_um[c];

    if (poly_mode == CACAOSIM_POLY_UNWRAPPED_OPD && m1 != m0 && w1 > 0.0f)
    {
        float amp = w0 * amp0 + w1 * amp1;
        if (amp <= 0.0f)
        {
            fft_in[out_idx] = make_cuFloatComplex(0.0f, 0.0f);
            return;
        }
        float opd0 = (wf_in != NULL) ? (wf_to_opd * wf_in[b0]) : 0.0f;
        float opd1 = (wf_in != NULL) ? (wf_to_opd * wf_in[b1]) : 0.0f;
        float s    = 0.0f;
        float co   = 1.0f;
        __sincosf((w0 * opd0 + w1 * opd1 + common) * kc, &s, &co);
        fft_in[out_idx] = make_cuFloatComplex(amp * co, amp * s);
        return;
    }

    float p0  = (((wf_in != NULL) ? (wf_to_opd * wf_in[b0]) : 0.0f) + common) * kc;
    float s0  = 0.0f;
    float co0 = 1.0f;
    __sincosf(p0, &s0, &co0);
    float re = w0 * amp0 * co0;
    float im = w0 * amp0 * s0;

    if (m1 != m0 && w1 > 0.0f)
    {
        float p1  = (((wf_in != NULL) ? (wf_to_opd * wf_in[b1]) : 0.0f) + common) * kc;
        float s1  = 0.0f;
        float co1 = 1.0f;
        __sincosf(p1, &s1, &co1);
        re += w1 * amp1 * co1;
        im += w1 * amp1 * s1;
    }
    fft_in[out_idx] = make_cuFloatComplex(re, im);
}

__global__ static void k_fpcam_accum_mft_batch(
    const cuComplex *__restrict__ efoc_batch,
    const float *__restrict__     strehl_w,
    uint64_t                      npix_cam,
    int32_t                       nlam,
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
        sum += strehl_w[c] * (z.x * z.x + z.y * z.y);
    }
    accum_int[idx] += sum;
}

__global__ static void k_fpcam_accum_fft_crop(
    const cufftComplex *__restrict__ fft_out,
    const float *__restrict__        scale_c,
    const float *__restrict__        strehl_w,
    uint32_t                         nfft,
    uint32_t                         ncam,
    int32_t                          nlam,
    float                            inv_s2,
    float *__restrict__              accum_int)
{
    uint32_t xf = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t yf = blockIdx.y * blockDim.y + threadIdx.y;
    if (xf >= ncam || yf >= ncam)
    {
        return;
    }
    int32_t  half_c   = (int32_t) (ncam / 2u);
    int32_t  dx       = (int32_t) xf - half_c;
    int32_t  dy       = (int32_t) yf - half_c;
    int32_t  nf       = (int32_t) nfft;
    uint64_t npix_fft = (uint64_t) nfft * nfft;
    float    sum      = 0.0f;

    for (int32_t c = 0; c < nlam; c++)
    {
        float    sc   = scale_c[c];
        float    sx_f = (float) dx * sc;
        float    sy_f = (float) dy * sc;
        int32_t  ix0  = (int32_t) floorf(sx_f);
        int32_t  iy0  = (int32_t) floorf(sy_f);
        float    fx   = sx_f - (float) ix0;
        float    fy   = sy_f - (float) iy0;
        uint32_t kx0  = (uint32_t) ((ix0 % nf + nf) % nf);
        uint32_t kx1  = (uint32_t) (((ix0 + 1) % nf + nf) % nf);
        uint32_t ky0  = (uint32_t) ((iy0 % nf + nf) % nf);
        uint32_t ky1  = (uint32_t) (((iy0 + 1) % nf + nf) % nf);

        const cufftComplex *fo  = &fft_out[(uint64_t) c * npix_fft];
        cufftComplex        z00 = fo[(uint64_t) ky0 * nfft + kx0];
        cufftComplex        z10 = fo[(uint64_t) ky0 * nfft + kx1];
        cufftComplex        z01 = fo[(uint64_t) ky1 * nfft + kx0];
        cufftComplex        z11 = fo[(uint64_t) ky1 * nfft + kx1];

        float i00 = z00.x * z00.x + z00.y * z00.y;
        float i10 = z10.x * z10.x + z10.y * z10.y;
        float i01 = z01.x * z01.x + z01.y * z01.y;
        float i11 = z11.x * z11.x + z11.y * z11.y;

        float v0 = (1.0f - fx) * i00 + fx * i10;
        float v1 = (1.0f - fx) * i01 + fx * i11;
        sum += strehl_w[c] * inv_s2 * ((1.0f - fy) * v0 + fy * v1);
    }
    accum_int[(uint64_t) yf * ncam + xf] += sum;
}

/**
 * k_fpcam_scale_exposure() - Normalize multi-subframe intensity sum by `1 / subframe_cnt`
 * @accum_int: Summed optical intensity `[npix]` over `subframe_cnt` subframes
 * @exp_int:   Output mean exposure optical intensity `[npix]` in Strehl units `[0..1]`
 * @npix:      Total camera pixels (`ncam * ncam`)
 * @inv_sub:   Reciprocal subframe count `1.0f / subframe_cnt`
 *
 * Role: Divides `accum_int` by `subframe_cnt` into `exp_int` immediately before GPU detector
 * readout and host Strehl telemetry copyback.
 *
 * Rationale: Writing to a separate `exp_int` buffer preserves the exact noise-free Strehl-scaled
 * exposure PSF for asynchronous D2H copyback while `cacaosim_cuda_detector_readout()` consumes
 * `exp_int` concurrently on device.
 */
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

__global__ static void k_fpcam_scale_long_stats(
    const float *__restrict__ accum_int,
    float *__restrict__       exp_int,
    float *__restrict__       long_int,
    float *__restrict__       long_out,
    uint32_t *__restrict__    stats_u32,
    uint64_t                  npix,
    uint64_t                  axis_idx,
    float                     inv_sub,
    float                     inv_long)
{
    __shared__ float s_exp[256];
    __shared__ float s_lng[256];

    uint32_t tid = threadIdx.x;
    uint64_t idx = (uint64_t) blockIdx.x * blockDim.x + tid;
    float    ve  = 0.0f;
    float    vl  = 0.0f;

    if (idx < npix)
    {
        ve = accum_int[idx] * inv_sub;
        if (ve < 0.0f)
        {
            ve = 0.0f;
        }
        exp_int[idx] = ve;

        float lsum    = long_int[idx] + ve;
        long_int[idx] = lsum;
        vl            = lsum * inv_long;
        if (vl < 0.0f)
        {
            vl = 0.0f;
        }
        if (long_out != NULL)
        {
            long_out[idx] = vl;
        }
        if (idx == axis_idx)
        {
            stats_u32[0] = __float_as_uint(ve);
        }
    }

    s_exp[tid] = ve;
    s_lng[tid] = vl;
    __syncthreads();

    for (uint32_t s = 128u; s > 0u; s >>= 1u)
    {
        if (tid < s)
        {
            if (s_exp[tid + s] > s_exp[tid])
            {
                s_exp[tid] = s_exp[tid + s];
            }
            if (s_lng[tid + s] > s_lng[tid])
            {
                s_lng[tid] = s_lng[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0u)
    {
        atomicMax(&stats_u32[1], __float_as_uint(s_exp[0]));
        atomicMax(&stats_u32[2], __float_as_uint(s_lng[0]));
    }
}

static void fpcam_cuda_invalidate_graph(fpcam_cuda_priv_t *p)
{
    if (p->graph_valid)
    {
        cudaGraphExecDestroy(p->graph_exec);
        cudaGraphDestroy(p->graph);
        p->graph_valid = 0;
    }
    p->graph_tried = 0;
    p->has_sig     = 0;
}

static void fpcam_cuda_launch_prop_kernels(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *d_pup,
    const float          *d_amp,
    const float          *d_wf,
    fpcam_poly_map_t      map,
    uint64_t              dev_stride,
    const float          *d_dm,
    float                 wf_to_opd,
    int32_t               poly_mode);

/**
 * cacaosim_fpcam_cuda_init() - Allocate device buffers, upload MFT/pupil data, and bind cuBLAS/FFT
 * @ctx:        Host-initialized focal-plane camera context
 * @pup_amp:    Host clear reference pupil amplitude `[npix_wf]`
 * @ncpa_opd:   Optional host static science-arm NCPA OPD `[npix_wf]` [um], or NULL
 * @ext_stream: Optional borrowed `cudaStream_t` (`void *`) for fused benches, or NULL
 *
 * Role: Cold GPU setup routine called from `cacaosim_fpcam_init()` to provision `fpcam_cuda_priv_t`
 * device memory, upload `pup_amp`, `lam_um`, `ncpa_opd`, and MFT basis matrices (`d_mft_U`,
 * `d_mft_V`) or create the 2D `CUFFT_C2C` plan, and synchronize `@stream` before returning.
 *
 * Rationale: Uploading static clear-pupil, wavelength, NCPA, and MFT basis arrays once at init
 * leaves only dynamic per-subframe wavefront/beam planes on the H2D path (or zero H2D bytes when
 * running on resident GPU pointers inside `cacaosim.htsim`).
 *
 * Return: `0` on success, `-1` on any CUDA/cuBLAS/cuFFT allocation or plan failure (auto-cleaned).
 */
static void fpcam_cuda_resolve_map(
    const cacaosim_fpcam_ctx_t *ctx,
    uint32_t                    beam_nlam,
    fpcam_poly_map_t           *map,
    int                        *identity_match)
{
    int32_t nlam = ctx->eff_nlam;
    int     id   = (beam_nlam == (uint32_t) nlam) ? 1 : 0;
    if (beam_nlam <= 1u)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            map->m0[c]    = 0u;
            map->m1[c]    = 0u;
            map->alpha[c] = 0.0f;
        }
        if (identity_match != NULL)
        {
            *identity_match = 0;
        }
        return;
    }

    const float *in_lam = NULL;
    if (ctx->cfg.has_in_lam)
    {
        in_lam = ctx->cfg.in_lam_um;
    }
    else if (ctx->cfg.has_ext_lam && beam_nlam == (uint32_t) nlam)
    {
        in_lam = ctx->lam_um;
    }
    float lam0   = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 1.65f;
    float bw     = (ctx->cfg.bw_frac > 0.0f) ? ctx->cfg.bw_frac : 0.20f;
    float span   = 0.5f * bw * ((float) (beam_nlam - 1u) / (float) beam_nlam);
    float def_l0 = lam0 * (1.0f - span);
    float def_l1 = lam0 * (1.0f + span);

    int32_t inv2 = (ctx->in_poly_mode == CACAOSIM_POLY_UNWRAPPED_OPD) ? 1 : 0;
    for (int32_t c = 0; c < nlam; c++)
    {
        cacaosim_poly_interp_bracket_ex(ctx->lam_um[c], in_lam, beam_nlam, def_l0, def_l1, inv2,
                                        &map->m0[c], &map->m1[c], &map->alpha[c]);
        if (map->m0[c] != (uint32_t) c || map->m1[c] != (uint32_t) c || map->alpha[c] > 1e-6f)
        {
            id = 0;
        }
    }
    if (identity_match != NULL)
    {
        *identity_match = id;
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

    const uint64_t npix_wf   = ctx->npix_wf;
    const uint64_t npix_cam  = ctx->npix_cam;
    const int32_t  nlam      = ctx->eff_nlam;
    const size_t   lam_bytes = (size_t) nlam * sizeof(float);
    const size_t   stg_bytes = (size_t) (2 * nlam) * npix_wf * sizeof(float);

    if (cudaMalloc((void **) &p->d_pup_amp, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_dyn_pup, stg_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_atm_amp, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_wf_in, stg_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_dm_opd, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_lam_um, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_strehl_w, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_scale_c, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_disp_sx, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_disp_sy, lam_bytes) != cudaSuccess ||
        cudaMalloc((void **) &p->d_accum_int, npix_cam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_exp_int, npix_cam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_long_int, npix_cam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_long_out, npix_cam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_stats_u32, 3u * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_out_u16, npix_cam * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_out_f32, npix_cam * sizeof(float)) != cudaSuccess)
    {
        cacaosim_fpcam_cuda_free(ctx);
        return -1;
    }

    if (cudaHostAlloc((void **) &p->h_stats_u32, 3u * sizeof(uint32_t), cudaHostAllocDefault) ==
        cudaSuccess)
    {
        p->pinned_stats = 1;
        memset(p->h_stats_u32, 0, 3u * sizeof(uint32_t));
    }
    else
    {
        cudaGetLastError();
        p->h_stats_u32 = (uint32_t *) calloc(3u, sizeof(uint32_t));
        if (p->h_stats_u32 == NULL)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
    }

    cudaMemcpyAsync(p->d_pup_amp, pup_amp, npix_wf * sizeof(float), cudaMemcpyHostToDevice,
                    p->stream);
    cudaMemcpyAsync(p->d_lam_um, ctx->lam_um, lam_bytes, cudaMemcpyHostToDevice, p->stream);
    cudaMemcpyAsync(p->d_strehl_w, ctx->strehl_w, lam_bytes, cudaMemcpyHostToDevice, p->stream);
    cudaMemcpyAsync(p->d_scale_c, ctx->scale_c, lam_bytes, cudaMemcpyHostToDevice, p->stream);
    cudaMemcpyAsync(p->d_disp_sx, ctx->disp_sx, lam_bytes, cudaMemcpyHostToDevice, p->stream);
    cudaMemcpyAsync(p->d_disp_sy, ctx->disp_sy, lam_bytes, cudaMemcpyHostToDevice, p->stream);
    cudaMemsetAsync(p->d_accum_int, 0, npix_cam * sizeof(float), p->stream);
    cudaMemsetAsync(p->d_long_int, 0, npix_cam * sizeof(float), p->stream);
    cudaMemsetAsync(p->d_stats_u32, 0, 3u * sizeof(uint32_t), p->stream);

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
        size_t fft_bytes = (size_t) nlam * ctx->npix_fft * sizeof(cufftComplex);
        if (cudaMalloc((void **) &p->d_fft_in, fft_bytes) != cudaSuccess ||
            cudaMalloc((void **) &p->d_fft_out, fft_bytes) != cudaSuccess)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
        int n_arr[2]   = { (int) ctx->nfft, (int) ctx->nfft };
        int inembed[2] = { n_arr[0], n_arr[1] };
        int idist      = (int) ctx->npix_fft;
        if (cufftPlanMany(&p->fft_fwd, 2, n_arr, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C,
                          nlam) != CUFFT_SUCCESS ||
            cufftSetStream(p->fft_fwd, p->stream) != CUFFT_SUCCESS)
        {
            cacaosim_fpcam_cuda_free(ctx);
            return -1;
        }
        p->has_fft = 1;
    }

    /* Bind cuBLAS / cuFFT internal workspaces once outside CUDA Graph capture, then zero accum */
    {
        fpcam_poly_map_t zero_map = { { 0 }, { 0 }, { 0.0f } };
        fpcam_cuda_launch_prop_kernels(ctx, p->d_pup_amp, NULL, NULL, zero_map, npix_wf, NULL,
                                       1.0f, 0);
        cudaMemsetAsync(p->d_accum_int, 0, npix_cam * sizeof(float), p->stream);
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
    fpcam_cuda_invalidate_graph(p);
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
    cudaFree(p->d_strehl_w);
    cudaFree(p->d_scale_c);
    cudaFree(p->d_disp_sx);
    cudaFree(p->d_disp_sy);
    cudaFree(p->d_mft_U);
    cudaFree(p->d_mft_V);
    cudaFree(p->d_epup_batch);
    cudaFree(p->d_mft_tmp);
    cudaFree(p->d_efoc_batch);
    cudaFree(p->d_fft_in);
    cudaFree(p->d_fft_out);
    cudaFree(p->d_accum_int);
    cudaFree(p->d_exp_int);
    cudaFree(p->d_long_int);
    cudaFree(p->d_long_out);
    cudaFree(p->d_stats_u32);
    if (p->h_stats_u32 != NULL)
    {
        if (p->pinned_stats)
        {
            cudaFreeHost(p->h_stats_u32);
        }
        else
        {
            free(p->h_stats_u32);
        }
    }
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

extern "C" void cacaosim_fpcam_cuda_upload_weights(cacaosim_fpcam_ctx_t *ctx)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    if (p == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    cudaMemcpyAsync(p->d_strehl_w, ctx->strehl_w, (size_t) ctx->eff_nlam * sizeof(float),
                    cudaMemcpyHostToDevice, p->stream);
}

extern "C" void cacaosim_fpcam_cuda_upload_disp(cacaosim_fpcam_ctx_t *ctx)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    if (p == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    size_t lam_bytes = (size_t) ctx->eff_nlam * sizeof(float);
    cudaMemcpyAsync(p->d_disp_sx, ctx->disp_sx, lam_bytes, cudaMemcpyHostToDevice, p->stream);
    cudaMemcpyAsync(p->d_disp_sy, ctx->disp_sy, lam_bytes, cudaMemcpyHostToDevice, p->stream);
}

extern "C" void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    cudaMemsetAsync(p->d_accum_int, 0, ctx->npix_cam * sizeof(float), p->stream);
}

extern "C" void cacaosim_fpcam_cuda_reset_long(cacaosim_fpcam_ctx_t *ctx)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    if (p == NULL)
    {
        return;
    }
    cudaMemsetAsync(p->d_long_int, 0, ctx->npix_cam * sizeof(float), p->stream);
    if (p->h_stats_u32 != NULL)
    {
        memset(p->h_stats_u32, 0, 3u * sizeof(uint32_t));
    }
}

static void fpcam_cuda_launch_prop_kernels(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *d_pup,
    const float          *d_amp,
    const float          *d_wf,
    fpcam_poly_map_t      map,
    uint64_t              dev_stride,
    const float          *d_dm,
    float                 wf_to_opd,
    int32_t               poly_mode)
{
    fpcam_cuda_priv_t *p        = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    const uint64_t     npix_wf  = ctx->npix_wf;
    const uint64_t     npix_cam = ctx->npix_cam;
    const uint32_t     nwf      = ctx->cfg.nwf;
    const uint32_t     ncam     = ctx->cfg.ncam;
    const int32_t      nlam     = ctx->eff_nlam;

    if (ctx->cfg.prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        dim3 blk(256, 1, 1);
        dim3 grd((unsigned int) ((npix_wf + 255u) / 256u), (unsigned int) nlam, 1);
        k_fpcam_build_epup_batch<<<grd, blk, 0, p->stream>>>(
            d_pup, d_amp, d_wf, map, dev_stride, d_dm, p->d_ncpa_opd, p->d_lam_um, p->d_disp_sx,
            p->d_disp_sy, wf_to_opd, nwf, npix_wf, nlam, poly_mode, p->d_epup_batch);

        const cuComplex one       = make_cuComplex(1.0f, 0.0f);
        const cuComplex zero      = make_cuComplex(0.0f, 0.0f);
        long long int   stride_ep = (long long int) npix_wf;
        long long int   stride_uv = (long long int) ncam * nwf;
        long long int   stride_ef = (long long int) npix_cam;

        cublasCgemmStridedBatched(p->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int) nwf, (int) ncam,
                                  (int) nwf, &one, p->d_epup_batch, (int) nwf, stride_ep,
                                  p->d_mft_U, (int) nwf, stride_uv, &zero, p->d_mft_tmp, (int) nwf,
                                  stride_uv, nlam);

        cublasCgemmStridedBatched(p->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int) ncam, (int) ncam,
                                  (int) nwf, &one, p->d_mft_V, (int) ncam, stride_uv, p->d_mft_tmp,
                                  (int) nwf, stride_uv, &zero, p->d_efoc_batch, (int) ncam,
                                  stride_ef, nlam);

        uint32_t g1d = (uint32_t) ((npix_cam + 255u) / 256u);
        k_fpcam_accum_mft_batch<<<g1d, 256, 0, p->stream>>>(
            p->d_efoc_batch, p->d_strehl_w, npix_cam, nlam, p->d_accum_int);
    }
    else
    {
        const float inv_s2 = 1.0f / (ctx->pup_sum_amp * ctx->pup_sum_amp);
        dim3        blk1(256, 1, 1);
        dim3        grd1((unsigned int) ((ctx->npix_fft + 255u) / 256u), (unsigned int) nlam, 1);

        k_fpcam_build_fft_batch<<<grd1, blk1, 0, p->stream>>>(
            d_pup, d_amp, d_wf, map, dev_stride, d_dm, p->d_ncpa_opd, p->d_lam_um, p->d_disp_sx,
            p->d_disp_sy, wf_to_opd, nwf, ctx->nfft, nlam, poly_mode, p->d_fft_in);

        cufftExecC2C(p->fft_fwd, p->d_fft_in, p->d_fft_out, CUFFT_FORWARD);

        dim3 b2d(16, 16);
        dim3 g2d((ncam + 15u) / 16u, (ncam + 15u) / 16u);
        k_fpcam_accum_fft_crop<<<g2d, b2d, 0, p->stream>>>(
            p->d_fft_out, p->d_scale_c, p->d_strehl_w, ctx->nfft, ncam, nlam, inv_s2,
            p->d_accum_int);
    }
}

static void fpcam_cuda_run_prop(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *d_pup,
    const float          *d_amp,
    const float          *d_wf,
    fpcam_poly_map_t      map,
    uint64_t              dev_stride,
    const float          *d_dm,
    float                 wf_to_opd,
    int32_t               poly_mode)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;

    if (p->has_sig &&
        (p->sig_pup != d_pup || p->sig_amp != d_amp || p->sig_wf != d_wf || p->sig_dm != d_dm ||
         p->sig_stride != dev_stride || fabsf(p->sig_scale - wf_to_opd) > 1e-7f ||
         p->sig_poly_mode != poly_mode || memcmp(&p->sig_map, &map, sizeof(map)) != 0))
    {
        fpcam_cuda_invalidate_graph(p);
    }

    if (!p->graph_valid && !p->graph_tried)
    {
        p->graph_tried = 1;
        if (cudaStreamBeginCapture(p->stream, cudaStreamCaptureModeRelaxed) == cudaSuccess)
        {
            fpcam_cuda_launch_prop_kernels(ctx, d_pup, d_amp, d_wf, map, dev_stride, d_dm,
                                           wf_to_opd, poly_mode);
            if (cudaStreamEndCapture(p->stream, &p->graph) == cudaSuccess)
            {
                if (cudaGraphInstantiate(&p->graph_exec, p->graph, NULL, NULL, 0) == cudaSuccess)
                {
                    p->graph_valid   = 1;
                    p->has_sig       = 1;
                    p->sig_pup       = d_pup;
                    p->sig_amp       = d_amp;
                    p->sig_wf        = d_wf;
                    p->sig_dm        = d_dm;
                    p->sig_stride    = dev_stride;
                    p->sig_scale     = wf_to_opd;
                    p->sig_poly_mode = poly_mode;
                    p->sig_map       = map;
                }
                else
                {
                    cudaGraphDestroy(p->graph);
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

    if (p->graph_valid)
    {
        cudaGraphLaunch(p->graph_exec, p->stream);
    }
    else
    {
        fpcam_cuda_launch_prop_kernels(ctx, d_pup, d_amp, d_wf, map, dev_stride, d_dm, wf_to_opd,
                                       poly_mode);
    }
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
    fpcam_cuda_priv_t *p       = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    const uint64_t     npix_wf = ctx->npix_wf;
    fpcam_poly_map_t   map     = { { 0 }, { 0 }, { 0.0f } };

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

    fpcam_cuda_run_prop(ctx, d_pup, d_amp, d_wf, map, npix_wf, d_dm, wf_to_opd, 0);
}

extern "C" void cacaosim_fpcam_cuda_accum_poly(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    fpcam_cuda_priv_t *p        = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    const uint64_t     npix_wf  = ctx->npix_wf;
    const size_t       wf_bytes = (size_t) npix_wf * sizeof(float);
    const int32_t      nlam     = ctx->eff_nlam;

    fpcam_poly_map_t map            = { { 0 }, { 0 }, { 0.0f } };
    int              identity_match = 0;
    uint32_t         bn             = (beam_nlam >= 1u) ? beam_nlam : 1u;
    fpcam_cuda_resolve_map(ctx, bn, &map, &identity_match);

    if (d_beam_opd != NULL || d_beam_amp != NULL)
    {
        const float *d_pup = (d_beam_amp != NULL) ? d_beam_amp : p->d_pup_amp;
        fpcam_cuda_run_prop(ctx, d_pup, NULL, d_beam_opd, map, chan_stride, NULL, 1.0f,
                            ctx->in_poly_mode);
        return;
    }

    const float     *d_pup      = p->d_pup_amp;
    const float     *d_wf       = NULL;
    fpcam_poly_map_t staged_map = map;

    if (bn <= 1u)
    {
        if (beam_amp != NULL)
        {
            cudaMemcpyAsync(p->d_dyn_pup, beam_amp, wf_bytes, cudaMemcpyHostToDevice, p->stream);
            d_pup = p->d_dyn_pup;
        }
        if (beam_opd != NULL)
        {
            cudaMemcpyAsync(p->d_wf_in, beam_opd, wf_bytes, cudaMemcpyHostToDevice, p->stream);
            d_wf = p->d_wf_in;
        }
    }
    else if (identity_match)
    {
        if (beam_amp != NULL)
        {
            if (chan_stride == npix_wf)
            {
                cudaMemcpyAsync(p->d_dyn_pup, beam_amp, (size_t) nlam * wf_bytes,
                                cudaMemcpyHostToDevice, p->stream);
            }
            else
            {
                cudaMemcpy2DAsync(p->d_dyn_pup, wf_bytes, beam_amp,
                                  (size_t) chan_stride * sizeof(float), wf_bytes, (size_t) nlam,
                                  cudaMemcpyHostToDevice, p->stream);
            }
            d_pup = p->d_dyn_pup;
        }
        if (beam_opd != NULL)
        {
            if (chan_stride == npix_wf)
            {
                cudaMemcpyAsync(p->d_wf_in, beam_opd, (size_t) nlam * wf_bytes,
                                cudaMemcpyHostToDevice, p->stream);
            }
            else
            {
                cudaMemcpy2DAsync(p->d_wf_in, wf_bytes, beam_opd,
                                  (size_t) chan_stride * sizeof(float), wf_bytes, (size_t) nlam,
                                  cudaMemcpyHostToDevice, p->stream);
            }
            d_wf = p->d_wf_in;
        }
    }
    else
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            uint32_t m0 = map.m0[c];
            uint32_t m1 = map.m1[c];
            uint32_t s0 = (uint32_t) (2 * c);
            uint32_t s1 = (m1 == m0) ? s0 : (s0 + 1u);
            if (beam_amp != NULL)
            {
                cudaMemcpyAsync(p->d_dyn_pup + (uint64_t) s0 * npix_wf,
                                beam_amp + (uint64_t) m0 * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, p->stream);
                if (s1 != s0)
                {
                    cudaMemcpyAsync(p->d_dyn_pup + (uint64_t) s1 * npix_wf,
                                    beam_amp + (uint64_t) m1 * chan_stride, wf_bytes,
                                    cudaMemcpyHostToDevice, p->stream);
                }
                d_pup = p->d_dyn_pup;
            }
            if (beam_opd != NULL)
            {
                cudaMemcpyAsync(p->d_wf_in + (uint64_t) s0 * npix_wf,
                                beam_opd + (uint64_t) m0 * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, p->stream);
                if (s1 != s0)
                {
                    cudaMemcpyAsync(p->d_wf_in + (uint64_t) s1 * npix_wf,
                                    beam_opd + (uint64_t) m1 * chan_stride, wf_bytes,
                                    cudaMemcpyHostToDevice, p->stream);
                }
                d_wf = p->d_wf_in;
            }
            staged_map.m0[c] = s0;
            staged_map.m1[c] = s1;
        }
    }

    fpcam_cuda_run_prop(ctx, d_pup, NULL, d_wf, staged_map, npix_wf, NULL, 1.0f,
                        ctx->in_poly_mode);
}

/**
 * cacaosim_fpcam_cuda_readout() - Average subframes, synthesize GPU detector noise, and copy D2H
 * @ctx:          Initialized focal-plane camera context (`subframe_cnt >= 1`)
 * @det_live:     Active detector readout configuration (flux, dark, RON, gain, bias, saturation)
 * @host_exp_int: Optional host destination `[npix_cam]` for noise-free Strehl-scaled exposure PSF
 * @out_u16:      Optional host destination `[npix_cam]` `uint16_t` frame when `out_dtype == 0`
 * @out_f32:      Optional host destination `[npix_cam]` `float` frame when `out_dtype != 0`
 *
 * Role: Completes a GPU science camera exposure by scaling `d_accum_int` by `1 / subframe_cnt`
 * into `d_exp_int`, executing Philox-PRNG Poisson + Gaussian detector readout on GPU via
 * `cacaosim_cuda_detector_readout()`, queueing D2H copies for both `host_exp_int` and the active
 * ADU buffer, and synchronizing `p->stream`.
 *
 * Rationale: Queueing both `d_exp_int -> host_exp_int` and `d_out_* -> out_*` before a single
 * `cudaStreamSynchronize()` overlaps D2H DMA and delivers both the noisy science image for SHM
 * consumers and the exact noise-free Strehl PSF for host `out_long` / `.strehl_*` telemetry in one
 * stream barrier.
 */
extern "C" void cacaosim_fpcam_cuda_readout_full(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long,
    int                       sync_stream)
{
    fpcam_cuda_priv_t *p        = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    const uint64_t     npix     = ctx->npix_cam;
    const uint32_t     c0       = ctx->cfg.ncam / 2u;
    const uint64_t     axis_idx = (uint64_t) c0 * ctx->cfg.ncam + c0;
    const float        invs     = 1.0f / (float) ctx->subframe_cnt;
    const float        invl     = 1.0f / (float) ctx->long_cnt;
    float             *d_lout   = (out_long != NULL) ? p->d_long_out : NULL;

    cudaMemsetAsync(p->d_stats_u32, 0, 3u * sizeof(uint32_t), p->stream);

    uint32_t g1d = (uint32_t) ((npix + 255u) / 256u);
    k_fpcam_scale_long_stats<<<g1d, 256, 0, p->stream>>>(
        p->d_accum_int, p->d_exp_int, p->d_long_int, d_lout, p->d_stats_u32, npix, axis_idx, invs,
        invl);

    cacaosim_cuda_detector_readout(det_live, p->d_exp_int, npix, ctx->inf_flux_norm,
                                   ctx->cfg.rng_seed, ctx->frame_seq, p->d_out_u16, p->d_out_f32,
                                   p->stream);

    cudaMemcpyAsync(p->h_stats_u32, p->d_stats_u32, 3u * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                    p->stream);

    if (out_long != NULL)
    {
        cudaMemcpyAsync(out_long, p->d_long_out, npix * sizeof(float), cudaMemcpyDeviceToHost,
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
    if (sync_stream)
    {
        cudaStreamSynchronize(p->stream);
        cacaosim_fpcam_cuda_readout_finish(ctx);
    }
}

extern "C" void cacaosim_fpcam_cuda_readout_finish(cacaosim_fpcam_ctx_t *ctx)
{
    if (ctx == NULL || ctx->cuda_priv == NULL)
    {
        return;
    }
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    float              s_axis = 0.0f, s_peak = 0.0f, s_long = 0.0f;
    memcpy(&s_axis, &p->h_stats_u32[0], sizeof(float));
    memcpy(&s_peak, &p->h_stats_u32[1], sizeof(float));
    memcpy(&s_long, &p->h_stats_u32[2], sizeof(float));
    ctx->last_strehl_axis = s_axis;
    ctx->last_strehl_peak = s_peak;
    ctx->long_strehl_peak = s_long;
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
