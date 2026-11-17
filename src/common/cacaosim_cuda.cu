// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_cuda.cu
 * @brief   CUDA kernels and launchers for Philox PRNG, detector readout, pupil, and beam algebra
 *
 * Architectural Role:
 *   Implements all GPU runtime helpers and shared optical/detector device kernels declared in
 *   `cacaosim_cuda.h`. Serves as the foundational CUDA compute layer beneath `pyrwfs`, `shwfs`,
 *   `zwfs`, `fpcam`, `coro`, `dmsim`, `mkbeam`, `wfe`, `beamsplit`, and fused `htsim`.
 *
 * Design Rationale & Motivation:
 *   1. Stateless counter-based Philox-2x32-10 PRNG (`philox2x32_10`): Conventional GPU PRNGs
 *      (`curandStateXORWOW`) store 48 bytes of mutable state per pixel, adding tens of megabytes
 *      of global memory read/write traffic per frame. Salmon et al.'s (SC11) 10-round Philox-2x32
 *      bijection computes independent Crush-resistant random pairs purely in registers from
 *      `(pixel_id, frame_seq, key)` using fast GPU `__umulhi` instructions—requiring zero VRAM
 *      state and guaranteeing identical results independent of thread-block scheduling.
 *   2. GPU Special Function Unit (`SFU`) intrinsics (`__sincosf`, `__logf`, `__expf`): Complex
 *      phasor assembly `A * exp(i * phi)` and Box-Muller transforms execute in a single SFU
 *      instruction sequence rather than multi-branch software range-reduction routines, turning
 *      pupil construction and camera noise into purely memory-bandwidth-bound kernels.
 *   3. Fused zero-padding + phasor synthesis (`kernel_build_pupil_padded`): Writing zeros outside
 *      the `[pad_off, pad_off + nwf)` window and `A * exp(i * phi)` inside in one pass eliminates
 *      a separate `cudaMemsetAsync` over the `nfft x nfft` complex FFT input plane every subframe.
 *   4. Specialized identity vs. warp beam kernels (`*_ident` vs. `*_warp`): Separating aligned
 *      1-to-1 kernels from 4-tap bilinear inverse-affine warp kernels ensures nominal aligned arms
 *      enjoy fully coalesced 128-byte warp memory transactions with zero trigonometric overhead.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdint.h>

#include "common/cacaosim_cuda.h"

/**
 * philox2x32_10() - Evaluate 10-round counter-based Philox-2x32 stateless bijection on GPU
 * @ctr: 64-bit input counter (`uint2`) encoding pixel index and frame/sub-step sequence
 * @key: 32-bit key derived from the simulation random seed
 *
 * Role:
 *   Core register-only `__device__` PRNG primitive driving all GPU Box-Muller Gaussian and Knuth
 *   Poisson detector noise draws in `kernel_detector_readout`.
 *
 * Rationale:
 *   Applies 10 Feistel-network rounds mixing the 64-bit product `(hi, lo) = M * ctr.x` (`M =
 *   0xd256d193`, Salmon et al. SC11 optimal multiplier) with Weyl key schedule `key += 0x9e3779b9`
 *   (golden-ratio constant). Unrolled completely (`#pragma unroll`) into register ALU ops
 *   (`__umulhi` + integer multiply + XOR), passing TestU01 BigCrush with zero global memory
 *   accesses.
 *
 * Return: Statistically uniform, independent pair of 32-bit unsigned integers (`uint2`).
 */
__device__ __forceinline__ uint2 philox2x32_10(
    uint2    ctr,
    uint32_t key)
{
    /* Salmon et al. (2011) Philox-2x32 multiplier and 32-bit golden-ratio Weyl increment */
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

/**
 * gpu_u32_to_uniform() - Map a 32-bit random word onto the open float32 interval `(0, 1)`
 * @u: Uniformly distributed 32-bit unsigned integer
 *
 * Role:
 *   Converts raw Philox-2x32 output words into singularity-free `(0, 1)` floats for GPU
 *   Box-Muller and Knuth Poisson sampling.
 *
 * Rationale:
 *   Matches `cacaosim_rng_uniform_f32()` on CPU: keeps the upper 24 mantissa bits (`u >> 8`) and
 *   shifts by `+0.5f * 2^-24` so `0.0f` and `1.0f` can never occur (`__logf(u1)` is always finite).
 *
 * Return: Uniform `float` strictly inside `[2^-25, 1 - 2^-25]`.
 */
__device__ __forceinline__ float gpu_u32_to_uniform(uint32_t u)
{
    return ((float) (u >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

/**
 * gpu_box_muller() - Transform two uniform `uint32_t` words into two independent `N(0, 1)` floats
 * @u: Pair of independent 32-bit random words from `philox2x32_10()`
 *
 * Role:
 *   Produces two orthogonal standard normal variates (`g2.x` for high-flux Cornish-Fisher Poisson
 *   and `g2.y` for Gaussian readout noise) in a single non-divergent step per pixel.
 *
 * Rationale:
 *   Whereas CPU code uses rejection-based Marsaglia polar to avoid slow scalar `sinf`/`cosf`, SIMT
 *   GPU warps suffer branch divergence on rejection `while` loops and possess dedicated hardware
 *   `__sincosf` SFU units. Direct trigonometric Box-Muller is 100% branchless across all 32 lanes
 *   of a warp and yields both normal variates from a single Philox draw.
 *
 * Return: `float2` containing two independent `N(0, 1)` standard normal samples.
 */
__device__ __forceinline__ float2 gpu_box_muller(uint2 u)
{
    float u1  = gpu_u32_to_uniform(u.x);
    float u2  = gpu_u32_to_uniform(u.y);
    float rad = sqrtf(-2.0f * __logf(u1));
    float s, c;
    __sincosf(CACAOSIM_TWO_PI_F * u2, &s, &c);
    return make_float2(rad * c, rad * s);
}

/**
 * gpu_sample_poisson() - Sample `Poisson(mu)` photoelectrons on GPU using stateless Philox draws
 * @mu:        Expected mean photoelectron count `[e-]` (`<= 0` returns `0.0f`)
 * @pixel_id:  Linear 64-bit detector pixel index (`0 .. npix - 1`)
 * @frame_seq: Monotonic exposure sequence counter
 * @base_key:  32-bit Philox key (XOR-decorrelated from the outer Box-Muller key)
 * @z_norm:    Pre-drawn `N(0, 1)` variate (`g2.x`) used when `@mu >= 24.0f`
 *
 * Role:
 *   Device shot-noise sampler invoked per pixel by `kernel_detector_readout` when `do_poiss != 0`.
 *
 * Rationale:
 *   Mirrors CPU `cacaosim_rng_poisson_f32()` at the exact `mu = 24.0f` boundary:
 *     - For `0 < mu < 24`: Runs Knuth's exact multiplicative algorithm consuming both outputs
 *       (`rnd.x` and `rnd.y`) of each `philox2x32_10` call (halving Philox invocations to ~`mu/2`),
 *       advancing `step` up to a safety cap of `128` (`256` draws, `< 10^-70` tail probability at
 *       `mu < 24`) to guarantee bounded warp termination.
 *     - For `mu >= 24`: Evaluates the branchless `O(1)` Cornish-Fisher skewness-corrected normal
 *       formula reusing the already-computed `@z_norm` variate with zero extra Philox rounds.
 *
 * Return: Non-negative integer photoelectron count as `float`.
 */
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
        /* Exact Knuth multiplicative sampler consuming two uniform variates per Philox call */
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

    /* High-flux O(1) Cornish-Fisher third-moment skewness-corrected normal approximation */
    float val = mu + sqrtf(mu) * z_norm + (1.0f / 6.0f) * (z_norm * z_norm - 1.0f) + 0.5f;
    return (val > 0.0f) ? floorf(val) : 0.0f;
}

/**
 * kernel_build_pupil_padded() - Assemble zero-padded `[nfft x nfft]` complex pupil field on GPU
 * @pup_amp:    Device pupil amplitude `[nwf * nwf]` (non-NULL)
 * @atm_amp:    Optional device atmospheric scintillation amplitude `[nwf * nwf]` (`NULL` -> `1`)
 * @wf_in:      Optional device external wavefront `[nwf * nwf]` (`NULL` -> `0`)
 * @dm_opd:     Optional device DM surface OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0`)
 * @ncpa_opd:   Optional device static NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0`)
 * @wf_to_opd:  Scale factor converting `@wf_in` to `[um]` OPD
 * @opd_to_rad: Wavenumber `2 * pi / lambda_um` `[rad / um]`
 * @nwf:        Input pupil grid width `[pixels]`
 * @nfft:       Output padded complex FFT grid width `[pixels]`
 * @pad_off:    Centered window offset `(nfft - nwf) / 2` `[pixels]`
 * @out_cplx:   Output `cuFloatComplex` array `[nfft * nfft]`
 *
 * Role:
 *   1D-grid CUDA kernel launched by `cacaosim_cuda_build_pupil_padded()` with one thread per
 *   output FFT grid cell (`total = nfft * nfft`).
 *
 * Rationale:
 *   Mapping threads over the destination `[nfft x nfft]` plane writes every element of `out_cplx`
 *   coalesced in linear order—emitting `(0, 0)` in the guard band and early-exiting when
 *   `pup_amp[pix] == 0.0f` to skip OPD reads and `__sincosf` outside the clear telescope aperture.
 */
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

    /* Zero-fill oversampling guard band outside the centered [nwf x nwf] pupil window */
    if (ox < pad_off || oy < pad_off || ox >= pad_off + nwf || oy >= pad_off + nwf)
    {
        out_cplx[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    uint32_t pix = (oy - pad_off) * nwf + (ox - pad_off);
    float    amp = pup_amp[pix];
    /* Short-circuit masked-out pixels outside the telescope pupil / behind spiders */
    if (amp == 0.0f)
    {
        out_cplx[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }
    if (atm_amp != NULL)
    {
        amp *= atm_amp[pix];
    }

    /* Combine turbulence, reflective DM correction (-dm_opd), and static path NCPA in [um] */
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

/**
 * kernel_detector_readout() - Per-pixel GPU radiometric scaling, shot/RON noise, and ADC clamping
 * @opt_int:         Device input optical intensity `[npix]`
 * @npix:            Total detector pixel count
 * @flux_scale:      Precomputed optical intensity-to-photoelectron multiplier (`0` if shutter=0)
 * @dark_e:          Dark + background electrons `[e-/pixel/frame]` (`0` in raw passthrough)
 * @ron_e:           Readout noise standard deviation `[e- RMS]`
 * @inv_gain:        Reciprocal conversion gain `1 / gain_eadu` `[ADU/e-]`
 * @bias_adu:        Electronic pedestal offset `[ADU]`
 * @sat_adu:         Full-well / ADC saturation ceiling `[ADU]`
 * @raw_passthrough: `1` when `cfg->flux_ph < 0` (preserves signed/unclipped intensity in float32)
 * @do_poiss:        `1` to apply quantum Poisson shot noise
 * @do_ron:          `1` to add Gaussian electronics readout noise
 * @out_dtype:       Output format (`0` = `uint16`, non-zero = `float32`)
 * @base_key:        32-bit scrambled seed key for Philox-2x32
 * @frame_seq:       Monotonic exposure counter decorrelating successive frames
 * @out_u16:         Destination `uint16_t` buffer `[npix]` when `@out_dtype == 0`
 * @out_f32:         Destination `float` buffer `[npix]` when `@out_dtype != 0`
 *
 * Role:
 *   CUDA kernel launched by `cacaosim_cuda_detector_readout()` across all `@npix` camera pixels.
 *
 * Rationale:
 *   Skips `philox2x32_10` and `gpu_box_muller` entirely when `!do_poiss && !do_ron` (noiseless
 *   calibration mode), and when stochastic noise is enabled, derives a unique 64-bit counter from
 *   `(idx, frame_seq * 0x9e3779b9 ^ (idx >> 32))` and XORs `base_key` with `0xa5a5a5a5` inside
 *   `gpu_sample_poisson` so low-flux Knuth draws never collide with the outer Box-Muller stream.
 */
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
        /* Unique per-(pixel, frame) 64-bit counter; g2.x feeds high-mu Poisson, g2.y feeds RON */
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
        /* Nearest-integer rounding before uint16 ADC conversion */
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

/**
 * cacaosim_cuda_available() - Probe and memoize whether at least one CUDA GPU is accessible
 *
 * Role:
 *   Host capability query called during module initialization (`AUTO` / `CUDA` backend selection).
 *
 * Rationale:
 *   Caches the first `cudaGetDeviceCount()` result in a `static int` (`-1` = unprobed) so
 *   subsequent checks execute in a single branch without querying the CUDA driver API repeatedly.
 *
 * Return: `1` if `cudaGetDeviceCount` succeeds with `count > 0`, `0` otherwise.
 */
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

/**
 * cacaosim_cuda_stream_create() - Allocate a `cudaStreamNonBlocking` CUDA stream handle
 *
 * Role:
 *   Creates an independent asynchronous GPU work queue for standalone or fused `cacaosim` units.
 *
 * Rationale:
 *   Using `cudaStreamNonBlocking` prevents implicit synchronization barriers against default
 *   stream `0`, allowing concurrent overlap across independent AO arms.
 *
 * Return: `cudaStream_t` cast to `void *`, or `NULL` on failure / no GPU.
 */
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

/**
 * cacaosim_cuda_stream_destroy() - Destroy a non-NULL CUDA stream handle
 * @cuda_stream: Opaque `cudaStream_t` (`NULL` is safely ignored)
 *
 * Role:
 *   Releases GPU stream resources during module teardown.
 *
 * Rationale:
 *   Guards `NULL` so callers do not need conditional checks around uninitialized streams.
 */
extern "C" void cacaosim_cuda_stream_destroy(void *cuda_stream)
{
    if (cuda_stream != NULL)
    {
        cudaStreamDestroy((cudaStream_t) cuda_stream);
    }
}

/**
 * cacaosim_cuda_stream_sync() - Block calling CPU thread until `@cuda_stream` drains
 * @cuda_stream: Opaque `cudaStream_t` (`NULL` synchronizes default stream `0`)
 *
 * Role:
 *   Host-side completion barrier ensuring async D2H transfers have landed in host/SHM memory.
 *
 * Rationale:
 *   Thin C-ABI wrapper around `cudaStreamSynchronize()`.
 */
extern "C" void cacaosim_cuda_stream_sync(void *cuda_stream)
{
    cudaStreamSynchronize((cudaStream_t) cuda_stream);
}

/**
 * cacaosim_cuda_alloc_f32() - Allocate and zero-clear a `float32` array in GPU device memory
 * @nelem: Element count (`0` returns `NULL`)
 *
 * Role:
 *   Cold-path VRAM buffer allocator for shared beam planes and device workspaces.
 *
 * Rationale:
 *   Synchronizes default stream `0` after `cudaMemset` so any non-blocking stream immediately
 *   observing the returned pointer sees deterministically zeroed memory without a race condition.
 *
 * Return: Zeroed device `float *` pointer, or `NULL` on error.
 */
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
    cudaStreamSynchronize((cudaStream_t) 0);
    return d_ptr;
}

/**
 * cacaosim_cuda_free_f32() - Free a `float32` device buffer if non-NULL
 * @d_ptr: Device pointer to release (`NULL` is safely ignored)
 *
 * Role:
 *   Cold-path VRAM destructor paired with `cacaosim_cuda_alloc_f32()`.
 *
 * Rationale:
 *   Safe `NULL`-tolerant wrapper around `cudaFree()`.
 */
extern "C" void cacaosim_cuda_free_f32(float *d_ptr)
{
    if (d_ptr != NULL)
    {
        cudaFree(d_ptr);
    }
}

/**
 * cacaosim_cuda_h2d_f32() - Enqueue an asynchronous Host-to-Device `float32` memory copy
 * @d_dst:       Destination device pointer (`NULL` -> no-op)
 * @h_src:       Source host pointer (`NULL` -> no-op)
 * @nelem:       Number of floats to copy (`0` -> no-op)
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Uploads host float buffers onto `@cuda_stream`.
 *
 * Rationale:
 *   Validates non-NULL endpoints and non-zero size before calling `cudaMemcpyAsync`.
 */
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

/**
 * cacaosim_cuda_d2h_f32() - Enqueue an asynchronous Device-to-Host `float32` memory copy
 * @h_dst:       Destination host pointer (`NULL` -> no-op)
 * @d_src:       Source device pointer (`NULL` -> no-op)
 * @nelem:       Number of floats to copy (`0` -> no-op)
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Downloads GPU float results back to host / SHM buffers on `@cuda_stream`.
 *
 * Rationale:
 *   Validates non-NULL endpoints and non-zero size before calling `cudaMemcpyAsync`.
 */
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

/**
 * cacaosim_cuda_build_pupil_padded() - Host launcher for `kernel_build_pupil_padded`
 * @d_pup_amp:   Device pupil amplitude `[nwf * nwf]`
 * @d_atm_amp:   Optional device scintillation amplitude `[nwf * nwf]` (`NULL` -> `1`)
 * @d_wf_in:     Optional device external wavefront `[nwf * nwf]` (`NULL` -> `0`)
 * @d_dm_opd:    Optional device DM OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0`)
 * @d_ncpa_opd:  Optional device static NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0`)
 * @wf_to_opd:   Wavefront-to-microns OPD multiplier
 * @opd_to_rad:  `2 * pi / lambda_um` `[rad / um]`
 * @nwf:         Input pupil grid width `[pixels]`
 * @nfft:        Output padded FFT grid width `[pixels]`
 * @d_out_cplx:  Destination `cuFloatComplex` array `[nfft * nfft]`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Launches complex pupil field construction on `@cuda_stream` with 256-thread blocks.
 *
 * Rationale:
 *   Centers the `[nwf x nwf]` pupil at `pad_off = (nfft - nwf) / 2` and uses 256 threads/block
 *   (`8` warps) for high SM occupancy across Pascal through Hopper/Blackwell GPUs.
 */
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

/**
 * cacaosim_cuda_detector_readout() - Host launcher for `kernel_detector_readout`
 * @cfg:         Detector physics and ADC configuration
 * @d_opt_int:   Device input optical intensity `[npix]`
 * @npix:        Total camera pixel count
 * @norm_total:  Reference flux normalization integral
 * @seed:        64-bit random seed
 * @frame_seq:   Monotonic frame sequence number
 * @d_out_u16:   Device `uint16_t` output `[npix]` when `cfg->out_dtype == 0`
 * @d_out_f32:   Device `float` output `[npix]` when `cfg->out_dtype != 0`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Precomputes scalar radiometric invariants on host and launches `kernel_detector_readout`.
 *
 * Rationale:
 *   Hoists divisions (`1.0f / gain_eadu`, `flux_ph / norm_total`), shutter gating, and 64-to-32
 *   seed folding (`seed ^ (seed >> 32) ^ 0x517cc1b7u`) out of the GPU kernel into scalar kernel
 *   arguments stored in fast constant-memory bank `c[0]`.
 */
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

    /* Fold 64-bit seed with golden-ratio-derived constant into 32-bit Philox key */
    uint32_t     base_key = (uint32_t) (seed ^ (seed >> 32) ^ 0x517cc1b7u);
    uint32_t     threads  = 256u;
    uint32_t     blocks   = (uint32_t) ((npix + threads - 1ULL) / threads);
    cudaStream_t st       = (cudaStream_t) cuda_stream;

    kernel_detector_readout<<<blocks, threads, 0, st>>>(
        d_opt_int, npix, flux_scale, dark_e, cfg->ron_e, inv_gain, bias_adu, sat_adu, raw_pass,
        do_poiss, do_ron, cfg->out_dtype, base_key, frame_seq, d_out_u16, d_out_f32);
}

/**
 * kernel_beam_init() - Synthesize initial `[OPD, Amp]` pupil beam arrays elementwise on GPU
 * @pup_amp:   Device static pupil amplitude `[npix]` (`NULL` -> `1.0f`)
 * @atm_amp:   Optional device scintillation amplitude `[npix]` (`NULL` -> `1.0f`)
 * @wf_in:     Optional device input wavefront `[npix]` (`NULL` -> `0.0f`)
 * @ncpa_opd:  Optional device entrance NCPA OPD `[npix]` in `[um]` (`NULL` -> `0.0f`)
 * @wf_to_opd: Scale converting `@wf_in` to `[um]` OPD
 * @npix:      Total pupil pixel count (`nwf * nwf`)
 * @out_opd:   Device output OPD `[npix]` in `[um]`
 * @out_amp:   Device output amplitude `[npix]`
 *
 * Role:
 *   CUDA kernel launched by `cacaosim_cuda_beam_init()`.
 *
 * Rationale:
 *   Coalesced 1D elementwise kernel computing `out_amp = pup_amp * atm_amp` and
 *   `out_opd = wf_to_opd * wf_in + ncpa_opd` in a single memory pass.
 */
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

/**
 * kernel_beam_apply_dm() - Subtract DM OPD surface from an optical beam elementwise on GPU
 * @in_opd:  Device input OPD `[npix]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:  Device input amplitude `[npix]` (`NULL` -> `1.0f`)
 * @dm_opd:  Device DM OPD `[npix]` in `[um]` (`NULL` -> `0.0f`)
 * @npix:    Total pupil pixel count (`nwf * nwf`)
 * @out_opd: Device output residual OPD `[npix]` in `[um]` (may alias `@in_opd`)
 * @out_amp: Device output amplitude `[npix]` (skipped if `NULL` or aliasing `@in_amp`)
 *
 * Role:
 *   CUDA kernel launched by `cacaosim_cuda_beam_apply_dm()`.
 *
 * Rationale:
 *   Computes closed-loop subtraction `out_opd = in_opd - dm_opd` and avoids redundant global
 *   memory writes when `out_amp == in_amp`.
 */
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

/**
 * kernel_beam_relay_ident() - Fast coalesced GPU beam relay when 2D registration is identity
 * @in_opd:   Device input OPD `[npix]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:   Device input amplitude `[npix]` (`NULL` -> `1.0f`)
 * @ncpa_opd: Optional device arm NCPA OPD `[npix]` in `[um]` (`NULL` -> `0.0f`)
 * @pup_mask: Optional device arm pupil mask `[npix]` (`NULL` -> `1.0f`)
 * @npix:     Total pupil pixel count (`nwf * nwf`)
 * @out_opd:  Device output OPD `[npix]` in `[um]`
 * @out_amp:  Device output amplitude `[npix]`
 *
 * Role:
 *   Identity-registration fast path launched by `cacaosim_cuda_beam_relay()`.
 *
 * Rationale:
 *   Preserves 1-to-1 coalesced memory loads/stores and supports safe in-place operation without
 *   bilinear gather overhead when optical arms are nominally aligned.
 */
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

/**
 * kernel_beam_relay_warp() - Inverse-affine bilinear 2D pupil warp + NCPA + pupil mask on GPU
 * @in_opd:   Device input OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:   Device input amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @ncpa_opd: Optional device post-warp arm NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @pup_mask: Optional device post-warp arm pupil stop `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @c:        Precomputed `cos(-rot_rad) / scale` inverse affine matrix term
 * @s:        Precomputed `sin(-rot_rad) / scale` inverse affine matrix term
 * @dx:       Forward pupil X translation `[pixels]`
 * @dy:       Forward pupil Y translation `[pixels]`
 * @nwf:      Square pupil grid dimension `[pixels]`
 * @out_opd:  Device output OPD `[nwf * nwf]` in `[um]` (distinct from `@in_opd`)
 * @out_amp:  Device output amplitude `[nwf * nwf]` (distinct from `@in_amp`)
 *
 * Role:
 *   General 2D misregistered relay kernel launched by `cacaosim_cuda_beam_relay()`.
 *
 * Rationale:
 *   Matches `cacaosim_beam_relay_cpu()` to floating-point rounding precision: maps each output
 *   pixel `(ix, iy)` backward around `ctr = 0.5 * (nwf - 1)` into source coordinates `(sx, sy)`,
 *   performs 2x2 bilinear gather (zero-padded outside `[0, nwf - 1]^2`), and applies downstream
 *   `ncpa_opd` and `pup_mask` at `idx`.
 */
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
    /* Inverse similarity mapping from destination arm pixel (ix, iy) to source (sx, sy) */
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

/**
 * cacaosim_cuda_beam_init() - Host launcher for `kernel_beam_init`
 * @d_pup_amp:   Device static pupil amplitude `[nwf * nwf]`
 * @d_atm_amp:   Optional device scintillation amplitude `[nwf * nwf]`
 * @d_wf_in:     Optional device external wavefront `[nwf * nwf]`
 * @d_ncpa_opd:  Optional device entrance NCPA OPD `[nwf * nwf]` in `[um]`
 * @wf_to_opd:   Scale converting `@d_wf_in` to `[um]` OPD
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Device output amplitude `[nwf * nwf]`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Enqueues GPU beam synthesis onto `@cuda_stream`.
 *
 * Rationale:
 *   Uses 256-thread 1D grid geometry covering `nwf * nwf` pixels.
 */
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

/**
 * kernel_beam_init_lerp() - GPU pupil beam synthesis with linear two-screen WF interpolation
 * @pup_amp:  Device static pupil amplitude `[npix]` (`NULL` -> `1.0f`)
 * @atm_amp:  Optional device scintillation amplitude `[npix]` (`NULL` -> `1.0f`)
 * @wf_prev:  Optional device previous wavefront screen `[npix]`
 * @wf_curr:  Optional device current wavefront screen `[npix]`
 * @w0:       Pre-scaled `(1 - wf_alpha) * wf_to_opd` weight
 * @w1:       Pre-scaled `wf_alpha * wf_to_opd` weight
 * @wf_scale: Raw `wf_to_opd` scale when only one screen pointer is non-NULL
 * @ncpa_opd: Optional device static NCPA OPD `[npix]` in `[um]`
 * @npix:     Total pupil pixels (`nwf * nwf`)
 * @out_opd:  Device output OPD `[npix]` in `[um]`
 * @out_amp:  Device output amplitude `[npix]`
 */
__global__ void kernel_beam_init_lerp(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *             wf_prev,
    const float *             wf_curr,
    float                     w0,
    float                     w1,
    float                     wf_scale,
    const float *__restrict__ ncpa_opd,
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
    float o = 0.0f;
    if (wf_prev != NULL && wf_curr != NULL)
    {
        o = w0 * wf_prev[idx] + w1 * wf_curr[idx];
    }
    else if (wf_curr != NULL)
    {
        o = wf_scale * wf_curr[idx];
    }
    else if (wf_prev != NULL)
    {
        o = wf_scale * wf_prev[idx];
    }
    if (ncpa_opd != NULL)
    {
        o += ncpa_opd[idx];
    }
    out_opd[idx] = o;
    out_amp[idx] = a;
}

/**
 * cacaosim_cuda_beam_init_lerp() - Host launcher for `kernel_beam_init_lerp`
 * @d_pup_amp:   Device static pupil amplitude `[nwf * nwf]`
 * @d_atm_amp:   Optional device scintillation amplitude `[nwf * nwf]`
 * @d_wf_prev:   Optional device previous wavefront `[nwf * nwf]`
 * @d_wf_curr:   Optional device current wavefront `[nwf * nwf]`
 * @wf_alpha:    Sub-tick linear blend weight in `[0, 1]`
 * @d_ncpa_opd:  Optional device entrance NCPA OPD `[nwf * nwf]` in `[um]`
 * @wf_to_opd:   Scale converting wavefront units to `[um]` OPD
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Device output amplitude `[nwf * nwf]`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 */
extern "C" void cacaosim_cuda_beam_init_lerp(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_prev,
    const float *d_wf_curr,
    float        wf_alpha,
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
    float        w0      = (1.0f - wf_alpha) * wf_to_opd;
    float        w1      = wf_alpha * wf_to_opd;

    kernel_beam_init_lerp<<<blocks, threads, 0, st>>>(
        d_pup_amp, d_atm_amp, d_wf_prev, d_wf_curr, w0, w1, wf_to_opd, d_ncpa_opd, npix,
        d_out_opd, d_out_amp);
}

/**
 * cacaosim_cuda_beam_apply_dm() - Host launcher for `kernel_beam_apply_dm`
 * @d_in_opd:    Device input OPD `[nwf * nwf]` in `[um]`
 * @d_in_amp:    Device input amplitude `[nwf * nwf]`
 * @d_dm_opd:    Device DM OPD `[nwf * nwf]` in `[um]`
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output residual OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Device output amplitude `[nwf * nwf]`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Enqueues GPU DM OPD subtraction onto `@cuda_stream`.
 *
 * Rationale:
 *   Single-launch 256-thread 1D dispatch keeping DM correction completely device-resident.
 */
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

/**
 * cacaosim_cuda_beam_relay() - Host dispatcher for `kernel_beam_relay_ident` / `warp`
 * @d_in_opd:    Device input OPD `[nwf * nwf]` in `[um]`
 * @d_in_amp:    Device input amplitude `[nwf * nwf]`
 * @d_ncpa_opd:  Optional device arm NCPA OPD `[nwf * nwf]` in `[um]`
 * @d_pup_mask:  Optional device arm pupil stop `[nwf * nwf]`
 * @reg:         Optional host 2D registration parameters (`NULL` -> identity)
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Device output amplitude `[nwf * nwf]`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Evaluates `cacaosim_reg_is_identity(reg)` on host and launches either the coalesced identity
 *   relay kernel or the precomputed inverse-affine bilinear warp kernel.
 *
 * Rationale:
 *   Computing `cosf`/`sinf` and `1.0f / scale` once on the host CPU avoids redundant per-thread
 *   trigonometric setup across `nwf^2` GPU threads.
 */
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

/**
 * kernel_beam_wfe_ident() - Coalesced GPU static WFE OPD addition for identity registration
 * @in_opd:    Device input OPD `[npix]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:    Device input amplitude `[npix]` (`NULL` -> `1.0f`)
 * @wfe_opd:   Device static WFE map `[npix]` (`NULL` -> `0.0f`)
 * @wfe_to_um: Scale converting `@wfe_opd` + gain to `[um]`
 * @npix:      Total pupil pixel count (`nwf * nwf`)
 * @out_opd:   Device output OPD `[npix]` in `[um]`
 * @out_amp:   Optional device output amplitude `[npix]`
 *
 * Role:
 *   Aligned fast-path kernel launched by `cacaosim_cuda_beam_apply_wfe()`.
 *
 * Rationale:
 *   Evaluates `out_opd = in_opd + wfe_to_um * wfe_opd` in one coalesced pass over `npix` elements.
 */
__global__ void kernel_beam_wfe_ident(
    const float *__restrict__ in_opd,
    const float *__restrict__ in_amp,
    const float *__restrict__ wfe_opd,
    float                     wfe_to_um,
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
    float a      = (in_amp != NULL) ? in_amp[idx] : 1.0f;
    float w      = (wfe_opd != NULL) ? (wfe_to_um * wfe_opd[idx]) : 0.0f;
    out_opd[idx] = o + w;
    if (out_amp != NULL)
    {
        out_amp[idx] = a;
    }
}

/**
 * kernel_beam_wfe_warp() - Inverse-affine bilinear 2D pupil warp + scaled static WFE OPD on GPU
 * @in_opd:    Device input OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:    Device input amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @wfe_opd:   Device static WFE map `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @wfe_to_um: Scale converting `@wfe_opd` + gain to `[um]`
 * @c:         Precomputed `cos(-rot_rad) / scale` inverse affine matrix term
 * @s:         Precomputed `sin(-rot_rad) / scale` inverse affine matrix term
 * @dx:        Forward pupil X translation `[pixels]`
 * @dy:        Forward pupil Y translation `[pixels]`
 * @nwf:       Square pupil grid dimension `[pixels]`
 * @out_opd:   Device output OPD `[nwf * nwf]` in `[um]` (distinct from `@in_opd`)
 * @out_amp:   Optional device output amplitude `[nwf * nwf]` (distinct from `@in_amp`)
 *
 * Role:
 *   Misregistered static WFE kernel launched by `cacaosim_cuda_beam_apply_wfe()`.
 *
 * Rationale:
 *   Mirrors `cacaosim_beam_apply_wfe_cpu()`: resamples the incoming `[OPD, Amp]` beam into the WFE
 *   optic's local frame via 2x2 bilinear interpolation before adding `wfe_to_um * wfe_opd[idx]`.
 */
__global__ void kernel_beam_wfe_warp(
    const float *__restrict__ in_opd,
    const float *__restrict__ in_amp,
    const float *__restrict__ wfe_opd,
    float                     wfe_to_um,
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
    if (wfe_opd != NULL)
    {
        o += wfe_to_um * wfe_opd[idx];
    }
    out_opd[idx] = o;
    if (out_amp != NULL)
    {
        out_amp[idx] = a;
    }
}

/**
 * cacaosim_cuda_beam_apply_wfe() - Host dispatcher for `kernel_beam_wfe_ident` / `warp`
 * @d_in_opd:    Device input OPD `[nwf * nwf]` in `[um]`
 * @d_in_amp:    Device input amplitude `[nwf * nwf]`
 * @d_wfe_opd:   Device static WFE map `[nwf * nwf]`
 * @wfe_to_um:   Scale converting `@d_wfe_opd` + gain into `[um]`
 * @reg:         Optional host 2D affine registration (`NULL` -> identity)
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Optional device output amplitude `[nwf * nwf]`
 * @cuda_stream: Target `cudaStream_t` (`NULL` -> stream `0`)
 *
 * Role:
 *   Dispatches GPU static WFE injection onto `@cuda_stream` for `cacaosim.wfe` and `htsim`.
 *
 * Rationale:
 *   Branches on `cacaosim_reg_is_identity(reg)` on host so aligned static WFE maps avoid coordinate
 *   transforms and 4-tap gathers on device.
 */
extern "C" void cacaosim_cuda_beam_apply_wfe(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_wfe_opd,
    float                 wfe_to_um,
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
        kernel_beam_wfe_ident<<<blocks, threads, 0, st>>>(
            d_in_opd, d_in_amp, d_wfe_opd, wfe_to_um, npix, d_out_opd, d_out_amp);
    }
    else
    {
        float sc   = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
        float invs = 1.0f / sc;
        float rad  = -reg->rot_deg * (CACAOSIM_PI_F / 180.0f);
        float c    = cosf(rad) * invs;
        float s    = sinf(rad) * invs;
        kernel_beam_wfe_warp<<<blocks, threads, 0, st>>>(
            d_in_opd, d_in_amp, d_wfe_opd, wfe_to_um, c, s, reg->shift_x, reg->shift_y, nwf,
            d_out_opd, d_out_amp);
    }
}
