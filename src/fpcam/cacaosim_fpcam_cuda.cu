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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/cacaosim_cuda.h"
#include "fpcam/cacaosim_fpcam.h"

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

/**
 * k_fpcam_build_epup_batch() - Synthesize `[nlam x npix_wf]` complex pupil phasors across channels
 * @pup_amp:     Device pupil/beam amplitude (`[beam_nlam]` at `chan_stride`), or NULL for `1.0f`
 * @atm_amp:     Optional achromatic atmospheric amplitude `[npix_wf]`, or NULL
 * @wf_in:       Optional wavefront/beam OPD (`[beam_nlam]` at `chan_stride`), or NULL
 * @beam_nlam:   Number of spectral slices in `pup_amp` and `wf_in` (`1` = achromatic)
 * @chan_stride: Float element stride between consecutive input spectral slices
 * @dm_opd:      Optional DM OPD `[npix_wf]` [um] (subtracted), or NULL
 * @ncpa_opd:    Optional static science-arm NCPA OPD `[npix_wf]` [um] (added), or NULL
 * @lam_um:      Device spectral channel wavelengths `[nlam]` [um]
 * @wf_to_opd:   Conversion factor mapping `wf_in` units to OPD [um]
 * @npix_wf:     Pupil pixel count (`nwf * nwf`)
 * @nlam:        Number of active camera spectral channels (`eff_nlam`)
 * @epup_batch:  Output contiguous complex pupil batch `[nlam * npix_wf]`
 *
 * Role: 2D-grid GPU kernel (`blockIdx.x` = pixel block, `blockIdx.y` = wavelength channel `c`)
 * launched once per subframe in `PROP_MFT` mode to populate all `nlam` complex pupil slices prior
 * to `cublasCgemmStridedBatched`.
 *
 * Rationale: Fusing all `nlam` spectral channels into a single 2D launch eliminates per-channel
 * kernel launch latency, reuses shared L2 cache lines for achromatic maps (`atm_amp`, `dm_opd`,
 * `ncpa_opd`), and uses hardware SFU `__sincosf()` while early-exiting masked (`amp <= 0`) pixels.
 */
__global__ static void k_fpcam_build_epup_batch(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    uint32_t                  beam_nlam,
    uint64_t                  chan_stride,
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

    /* Map camera wavelength slice `c` onto corresponding upstream beam slice `bc` */
    uint32_t bc = (beam_nlam <= 1u)
                      ? 0u
                      : ((beam_nlam == (uint32_t) nlam)
                             ? (uint32_t) c
                             : (((uint32_t) c * beam_nlam) / (uint32_t) nlam));
    uint64_t b_idx = (uint64_t) bc * chan_stride + p;

    float amp = (pup_amp != NULL) ? pup_amp[b_idx] : 1.0f;
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
        opd += wf_to_opd * wf_in[b_idx];
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

/**
 * k_fpcam_accum_mft_batch() - Reduce polychromatic MFT focal amplitudes into broadband intensity
 * @efoc_batch: Batched complex focal fields `[nlam * npix_cam]` from `cublasCgemmStridedBatched`
 * @npix_cam:   Detector pixel count (`ncam * ncam`)
 * @nlam:       Number of spectral slices in `@efoc_batch`
 * @inv_nlam:   Reciprocal channel count `1.0f / nlam`
 * @accum_int:  In/out running Strehl-normalized subframe intensity accumulator `[npix_cam]`
 *
 * Role: Post-MFT reduction kernel that sums `|Efoc_c(idx)|^2` incoherently across `[0 .. nlam - 1]`
 * and accumulates `inv_nlam * sum` into `accum_int[idx]`.
 *
 * Rationale: Assigning one thread per focal pixel `idx` avoids atomic contention across spectral
 * channels and performs only a single read-modify-write to `accum_int[idx]` per subframe.
 */
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

/**
 * k_fpcam_accum_fft_crop() - Modularly center, crop, and accumulate 2D FFT focal intensity
 * @fft_out:   Unshifted `[nfft x nfft]` complex output from `cufftExecC2C`
 * @nfft:      Zero-padded 2D FFT dimension (`nwf * pad_factor`)
 * @ncam:      Science camera grid dimension
 * @scale_i:   Strehl + spectral weight `1.0f / (nlam * pup_sum_amp^2)`
 * @accum_int: In/out running Strehl-normalized subframe intensity accumulator `[ncam * ncam]`
 *
 * Role: 2D focal-plane kernel launched per channel in `PROP_FFT` mode to map unshifted cuFFT bins
 * `(kx, ky)` around DC `(0, 0)` onto camera center `(ncam / 2, ncam / 2)` and accumulate scaled
 * squared modulus `scale_i * |fft_out(kx, ky)|^2`.
 *
 * Rationale: Modular index wrapping `((d % nfft) + nfft) % nfft` combines `fftshift`, central ROI
 * cropping (or periodic wrapping if `ncam > nfft`), modulus-squared evaluation, and Strehl
 * normalization into one memory-coalesced pass over `[ncam x ncam]`.
 */
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
        cudaMalloc((void **) &p->d_dyn_pup, (size_t) nlam * npix_wf * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc((void **) &p->d_atm_amp, npix_wf * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &p->d_wf_in, (size_t) nlam * npix_wf * sizeof(float)) !=
            cudaSuccess ||
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

/**
 * cacaosim_fpcam_cuda_free() - Destroy cuBLAS/cuFFT handles, device buffers, and owned CUDA stream
 * @ctx: Focal-plane camera context owning `cuda_priv` (safe on NULL or partially allocated state)
 *
 * Role: Cold GPU teardown helper invoked by `cacaosim_fpcam_free()` and partial-init rollback
 * paths inside `cacaosim_fpcam_cuda_init()`.
 *
 * Rationale: `cudaFree(NULL)` is a well-defined no-op in the CUDA Runtime API, while `owns_stream`
 * ensures borrowed external streams (`cacaosim.htsim`) are never prematurely destroyed.
 */
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

/**
 * cacaosim_fpcam_cuda_reset_accum() - Asynchronously zero device subframe intensity accumulator
 * @ctx: Initialized focal-plane camera context with active `cuda_priv`
 *
 * Role: Enqueues `cudaMemsetAsync` on `p->stream` at the start of a new camera integration
 * (`subframe_cnt == 0`) to clear `d_accum_int`.
 *
 * Rationale: Stream-ordered `cudaMemsetAsync` avoids host-device synchronization stalls between
 * exposures while guaranteeing subsequent `+=` kernel accumulations start cleanly from zero.
 */
extern "C" void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx)
{
    fpcam_cuda_priv_t *p = (fpcam_cuda_priv_t *) ctx->cuda_priv;
    cudaMemsetAsync(p->d_accum_int, 0, ctx->npix_cam * sizeof(float), p->stream);
}

/**
 * fpcam_cuda_run_prop() - Enqueue GPU polychromatic Fraunhofer diffraction and intensity sum
 * @ctx:        Initialized focal-plane camera context
 * @d_pup:      Device pupil or dynamic beam amplitude (`[dev_nlam]` at `dev_stride`), or NULL
 * @d_amp:      Optional device atmospheric amplitude `[npix_wf]`, or NULL
 * @d_wf:       Optional device wavefront or beam OPD (`[dev_nlam]` at `dev_stride`), or NULL
 * @dev_nlam:   Number of spectral slices present in `d_pup` and `d_wf` (`1` = achromatic)
 * @dev_stride: Float element stride between consecutive device spectral slices
 * @d_dm:       Optional device DM OPD `[npix_wf]` [um], or NULL
 * @wf_to_opd:  Scale factor mapping `d_wf` units to OPD [um]
 *
 * Role: Shared device-pointer propagation core invoked by both `cacaosim_fpcam_cuda_accum()` and
 * `cacaosim_fpcam_cuda_accum_poly()` once inputs are resident in GPU memory.
 *
 * Rationale: Centralizing MFT vs FFT dispatch over device pointers lets standalone host-staged
 * SHM loops and zero-copy fused `cacaosim.htsim` pipelines execute the exact same sequence of
 * CUDA kernels and cuBLAS/cuFFT operations without code duplication.
 */
static void fpcam_cuda_run_prop(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *d_pup,
    const float          *d_amp,
    const float          *d_wf,
    uint32_t              dev_nlam,
    uint64_t              dev_stride,
    const float          *d_dm,
    float                 wf_to_opd)
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
            d_pup, d_amp, d_wf, dev_nlam, dev_stride, d_dm, p->d_ncpa_opd, p->d_lam_um, wf_to_opd,
            npix_wf, nlam, p->d_epup_batch);

        const cuComplex one       = make_cuComplex(1.0f, 0.0f);
        const cuComplex zero      = make_cuComplex(0.0f, 0.0f);
        long long int   stride_ep = (long long int) npix_wf;
        long long int   stride_uv = (long long int) ncam * nwf;
        long long int   stride_ef = (long long int) npix_cam;

        /*
         * Row-major T[ncam x nwf] = U[ncam x nwf] * Epup[nwf x nwf] is identical in memory to
         * column-major T^T[nwf x ncam] = Epup^T[nwf x nwf] * U^T[nwf x ncam] (lda=nwf, ldb=nwf).
         */
        cublasCgemmStridedBatched(p->cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int) nwf, (int) ncam,
                                  (int) nwf, &one, p->d_epup_batch, (int) nwf, stride_ep,
                                  p->d_mft_U, (int) nwf, stride_uv, &zero, p->d_mft_tmp, (int) nwf,
                                  stride_uv, nlam);

        /*
         * Row-major Efoc[ncam x ncam] = T[ncam x nwf] * V[nwf x ncam] is identical in memory to
         * column-major Efoc^T[ncam x ncam] = V^T[ncam x nwf] * T^T[nwf x ncam] (lda=ncam, ldb=nwf).
         */
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
        /* Normalize unnormalized 2D cuFFT peak (`pup_sum_amp`) into Strehl units `[0..1]` */
        const float scale_i = 1.0f / ((float) nlam * ctx->pup_sum_amp * ctx->pup_sum_amp);
        dim3        b2d(16, 16);
        dim3        g2d((ncam + 15u) / 16u, (ncam + 15u) / 16u);

        for (int32_t c = 0; c < nlam; c++)
        {
            uint32_t bc = (dev_nlam <= 1u)
                              ? 0u
                              : ((dev_nlam == (uint32_t) nlam)
                                     ? (uint32_t) c
                                     : (((uint32_t) c * dev_nlam) / (uint32_t) nlam));
            const float *d_pup_c = (d_pup != NULL) ? (d_pup + (uint64_t) bc * dev_stride) : NULL;
            const float *d_wf_c  = (d_wf != NULL) ? (d_wf + (uint64_t) bc * dev_stride) : NULL;
            float        opd_to_rad = CACAOSIM_TWO_PI_F / ctx->lam_um[c];
            cacaosim_cuda_build_pupil_padded(d_pup_c, d_amp, d_wf_c, d_dm, p->d_ncpa_opd,
                                             wf_to_opd, opd_to_rad, nwf, ctx->nfft, p->d_fft_in,
                                             p->stream);
            cufftExecC2C(p->fft_fwd, p->d_fft_in, p->d_fft_out, CUFFT_FORWARD);
            k_fpcam_accum_fft_crop<<<g2d, b2d, 0, p->stream>>>(p->d_fft_out, ctx->nfft, ncam,
                                                               scale_i, p->d_accum_int);
        }
    }
}

/**
 * cacaosim_fpcam_cuda_accum() - Stage optional host/device AO planes and accumulate one subframe
 * @ctx:           Initialized focal-plane camera context
 * @dyn_pup_amp:   Optional host dynamic pupil amplitude `[npix_wf]`, or NULL for resident pupil
 * @atm_amp:       Optional host atmospheric scintillation amplitude `[npix_wf]`, or NULL
 * @wf_in:         Optional host wavefront map `[npix_wf]`, or NULL
 * @dm_opd:        Optional host DM OPD `[npix_wf]` [um], or NULL
 * @d_pup_amp_ext: Optional resident device pupil amplitude `[npix_wf]`, or NULL
 * @d_wf_opd_ext:  Optional resident device wavefront/OPD `[npix_wf]`, or NULL
 * @d_dm_opd_ext:  Optional resident device DM OPD `[npix_wf]` [um], or NULL
 * @wf_to_opd:     Scale factor mapping `wf_in` / `d_wf_opd_ext` units to OPD [um]
 *
 * Role: GPU accumulation entry point for classical achromatic AO wavefronts, prioritizing external
 * resident device pointers (`d_*_ext`) when non-NULL and falling back to asynchronous H2D staging
 * (`cudaMemcpyAsync`) for host SHM pointers.
 *
 * Rationale: Allows hybrid benches (e.g., host SHM turbulence combined with GPU-resident DM OPD)
 * to copy only host-sourced arrays across PCIe while leaving resident device arrays untouched.
 */
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

    fpcam_cuda_run_prop(ctx, d_pup, d_amp, d_wf, 1u, npix_wf, d_dm, wf_to_opd);
}

/**
 * cacaosim_fpcam_cuda_accum_poly() - Stage or consume polychromatic `[OPD, Amp]` beam on GPU
 * @ctx:         Initialized focal-plane camera context
 * @beam_opd:    Host OPD slices `[beam_nlam]` [um] at `chan_stride`, or NULL
 * @beam_amp:    Host amplitude slices `[beam_nlam]` at `chan_stride`, or NULL
 * @beam_nlam:   Number of spectral channels in the input beam (`1` = collapsed, `>= 2` = poly)
 * @chan_stride: Float element stride between consecutive input spectral slices
 * @d_beam_opd:  Optional resident device OPD slices (`[beam_nlam]` at `chan_stride`), or NULL
 * @d_beam_amp:  Optional resident device amplitude slices (`[beam_nlam]` at `chan_stride`), or NULL
 *
 * Role: GPU entry point for dynamic `[OPD_um, Amplitude]` optical beams (e.g., post-coronagraphic
 * `out_beam` cubes from `cacaosim.coro`). Bypasses all H2D copies when `d_beam_opd` or
 * `d_beam_amp` is provided, or packs strided host SHM slices into contiguous `d_dyn_pup` /
 * `d_wf_in` device buffers before calling `fpcam_cuda_run_prop()`.
 *
 * Rationale: Selects the fastest DMA transfer topology for every host layout: single-block
 * `cudaMemcpyAsync` when contiguous (`beam_nlam == 1` or `chan_stride == npix_wf`), pitched 2D
 * `cudaMemcpy2DAsync` when interleaved SHM `[OPD_c, Amp_c]` matches `nlam` (`chan_stride = 2 *
 * npix_wf`), and nearest-slice gather copies when upstream `beam_nlam` differs from `nlam`.
 */
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

    /* Fast zero-copy path for fused GPU benches (`cacaosim.htsim`) */
    if (d_beam_opd != NULL || d_beam_amp != NULL)
    {
        const float *d_pup = (d_beam_amp != NULL) ? d_beam_amp : p->d_pup_amp;
        uint32_t     dn    = (beam_nlam >= 1u) ? beam_nlam : 1u;
        fpcam_cuda_run_prop(ctx, d_pup, NULL, d_beam_opd, dn, chan_stride, NULL, 1.0f);
        return;
    }

    const float *d_pup    = p->d_pup_amp;
    const float *d_wf     = NULL;
    uint32_t     dev_nlam = (beam_nlam <= 1u) ? 1u : (uint32_t) nlam;

    if (beam_amp != NULL)
    {
        if (beam_nlam <= 1u)
        {
            cudaMemcpyAsync(p->d_dyn_pup, beam_amp, wf_bytes, cudaMemcpyHostToDevice, p->stream);
        }
        else if (beam_nlam == (uint32_t) nlam && chan_stride == npix_wf)
        {
            cudaMemcpyAsync(p->d_dyn_pup, beam_amp, (size_t) nlam * wf_bytes,
                            cudaMemcpyHostToDevice, p->stream);
        }
        else if (beam_nlam == (uint32_t) nlam)
        {
            /* Gather strided SHM amplitude slices `[0 .. nlam-1]` in a single 2D DMA transaction */
            cudaMemcpy2DAsync(p->d_dyn_pup, wf_bytes, beam_amp,
                              (size_t) chan_stride * sizeof(float), wf_bytes, (size_t) nlam,
                              cudaMemcpyHostToDevice, p->stream);
        }
        else
        {
            for (int32_t c = 0; c < nlam; c++)
            {
                uint32_t bc = ((uint32_t) c * beam_nlam) / (uint32_t) nlam;
                cudaMemcpyAsync(p->d_dyn_pup + (size_t) c * npix_wf,
                                beam_amp + (uint64_t) bc * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, p->stream);
            }
        }
        d_pup = p->d_dyn_pup;
    }

    if (beam_opd != NULL)
    {
        if (beam_nlam <= 1u)
        {
            cudaMemcpyAsync(p->d_wf_in, beam_opd, wf_bytes, cudaMemcpyHostToDevice, p->stream);
        }
        else if (beam_nlam == (uint32_t) nlam && chan_stride == npix_wf)
        {
            cudaMemcpyAsync(p->d_wf_in, beam_opd, (size_t) nlam * wf_bytes, cudaMemcpyHostToDevice,
                            p->stream);
        }
        else if (beam_nlam == (uint32_t) nlam)
        {
            /* Gather strided SHM OPD slices `[0 .. nlam-1]` in a single 2D DMA transaction */
            cudaMemcpy2DAsync(p->d_wf_in, wf_bytes, beam_opd,
                              (size_t) chan_stride * sizeof(float), wf_bytes, (size_t) nlam,
                              cudaMemcpyHostToDevice, p->stream);
        }
        else
        {
            for (int32_t c = 0; c < nlam; c++)
            {
                uint32_t bc = ((uint32_t) c * beam_nlam) / (uint32_t) nlam;
                cudaMemcpyAsync(p->d_wf_in + (size_t) c * npix_wf,
                                beam_opd + (uint64_t) bc * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, p->stream);
            }
        }
        d_wf = p->d_wf_in;
    }

    fpcam_cuda_run_prop(ctx, d_pup, NULL, d_wf, dev_nlam, npix_wf, NULL, 1.0f);
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
