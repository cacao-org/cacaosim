// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_cuda.h
 * @brief   Pure-C host wrappers for common CUDA runtime, PRNG, detector, pupil, and beam kernels
 *
 * Architectural Role:
 *   Declares the CUDA-header-free C11 ABI (`void *cuda_stream`, `void *d_out_cplx`) shared by all
 *   `cacaosim` compute modules (`mkbeam`, `wfe`, `dmsim`, `beamsplit`, `coro`, `pyrwfs`, `shwfs`,
 *   `zwfs`, `fpcam`, and fused `htsim`). Backed by `cacaosim_cuda.cu` when built with CUDA support
 *   (`HAVE_CUDA`) and by `cacaosim_cuda_stub.c` in CPU-only builds.
 *
 * Design Rationale & Motivation:
 *   1. Opaque `void *` handles isolate C11 host code from CUDA Toolkit headers: Passing
 *      `cudaStream_t` and `cuFloatComplex *` as `void *` allows standard `gcc`/`clang` C11 units
 *      to orchestrate GPU memory transfers, stream synchronization, and kernel launches without
 *      requiring `cuda_runtime.h` or `#ifdef HAVE_CUDA` guards at call sites.
 *   2. Shared non-blocking stream pipelines: Exposing explicit stream creation and passing
 *      `cuda_stream` into every H2D/D2H copy and kernel launch enables `cacaosim.htsim` to bind
 *      cascaded optical stages (`mkbeam -> dm -> beamsplit -> coro -> wfs/fpcam`) onto a single
 *      `cudaStreamNonBlocking` queue so intermediate `[OPD, Amp]` buffers stay resident in VRAM
 *      with zero host synchronization hops.
 *   3. Exact physical and numerical parity with CPU primitives: Every pupil, beam, registration,
 *      and camera readout kernel mirrors the exact unit, sign, and radiometric conventions of
 *      `cacaosim_beam.c` and `cacaosim_detector.c`.
 */

#ifndef CACAOSIM_CUDA_H
#define CACAOSIM_CUDA_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_beam.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * cacaosim_cuda_available() - Query whether CUDA support is compiled in and a GPU is accessible
 *
 * Role:
 *   Cold-path capability probe called during `init()` of every GPU-capable `cacaosim` module when
 *   resolving `CACAOSIM_BACKEND_AUTO` or `CACAOSIM_BACKEND_CUDA`.
 *
 * Rationale:
 *   Caches the result of `cudaGetDeviceCount()` after the first query so repeated subsystem
 *   initializations in multi-stage `.bench` topologies avoid redundant driver round-trips, and
 *   returns `0` cleanly in CPU-only builds (`cacaosim_cuda_stub.c`).
 *
 * Return: `1` if at least one CUDA-capable GPU is available at runtime, `0` otherwise.
 */
int cacaosim_cuda_available(void);

/**
 * cacaosim_cuda_stream_create() - Create a non-blocking CUDA execution stream
 *
 * Role:
 *   Cold-path resource constructor used by standalone GPU modules and fused `htsim` to allocate
 *   an asynchronous `cudaStream_t` queue.
 *
 * Rationale:
 *   Creates streams with `cudaStreamNonBlocking` so multi-stream optical arms (e.g., parallel WFS
 *   and science branches) never implicitly serialize against legacy default stream `0`.
 *
 * Return: Opaque `cudaStream_t` cast to `void *`, or `NULL` if CUDA is unavailable or creation
 *         fails.
 */
void *cacaosim_cuda_stream_create(void);

/**
 * cacaosim_cuda_stream_destroy() - Destroy a CUDA stream created by `cacaosim_cuda_stream_create()`
 * @cuda_stream: Opaque `cudaStream_t` handle (`NULL` is safely ignored)
 *
 * Role:
 *   Cold-path destructor releasing private module CUDA streams during `free()`.
 *
 * Rationale:
 *   Accepts `NULL` as a no-op so teardown paths remain safe after failed or CPU-fallback inits.
 */
void cacaosim_cuda_stream_destroy(void *cuda_stream);

/**
 * cacaosim_cuda_stream_sync() - Block host thread until all work queued on `@cuda_stream` finishes
 * @cuda_stream: Opaque `cudaStream_t` handle (`NULL` synchronizes default stream `0`)
 *
 * Role:
 *   Synchronization barrier invoked after asynchronous D2H readouts before publishing output
 *   frames to `ImageStreamIO` shared memory (`processinfo_update_output_stream()`).
 *
 * Rationale:
 *   Scoping synchronization to a specific stream rather than `cudaDeviceSynchronize()` avoids
 *   stalling unrelated GPU tasks (such as `cacao` matrix-vector RTC reconstruction) sharing the
 *   same GPU.
 */
void cacaosim_cuda_stream_sync(void *cuda_stream);

/**
 * cacaosim_cuda_alloc_f32() - Allocate a zero-initialized `float32` buffer in GPU device memory
 * @nelem: Number of `float` elements to allocate (`0` returns `NULL`)
 *
 * Role:
 *   Cold-path VRAM allocator used by beam relays, fused `htsim`, and GPU test harnesses.
 *
 * Rationale:
 *   Performs `cudaMalloc` followed by `cudaMemset(..., 0)` + synchronization during setup so newly
 *   allocated device accumulators and OPD/amplitude planes start deterministically cleared without
 *   needing per-frame zeroing kernels.
 *
 * Return: Device `float *` pointer on success, or `NULL` if `@nelem == 0`, CUDA is unavailable, or
 *         VRAM allocation fails.
 */
float *cacaosim_cuda_alloc_f32(uint64_t nelem);

/**
 * cacaosim_cuda_free_f32() - Release a `float32` device buffer allocated on GPU
 * @d_ptr: Device pointer to free (`NULL` is safely ignored)
 *
 * Role:
 *   Cold-path VRAM deallocator paired with `cacaosim_cuda_alloc_f32()`.
 *
 * Rationale:
 *   Guards `NULL` pointers before `cudaFree()` for clean, idempotent error and teardown paths.
 */
void cacaosim_cuda_free_f32(float *d_ptr);

/**
 * cacaosim_cuda_h2d_f32() - Enqueue an asynchronous Host-to-Device `float32` copy
 * @d_dst:       Destination GPU device buffer `[nelem]` (no-op if `NULL`)
 * @h_src:       Source CPU host buffer `[nelem]` (no-op if `NULL`)
 * @nelem:       Number of `float` elements to transfer (`0` is a no-op)
 * @cuda_stream: Optional `cudaStream_t` (`NULL` uses default stream `0`)
 *
 * Role:
 *   Uploads external SHM turbulence screens, static masks, or host beam slices onto `@cuda_stream`.
 *
 * Rationale:
 *   Uses `cudaMemcpyAsync` on `@cuda_stream` to pipeline PCIe uploads ahead of dependent GPU
 *   kernels without intermediate host barriers.
 */
void cacaosim_cuda_h2d_f32(
    float       *d_dst,
    const float *h_src,
    uint64_t     nelem,
    void        *cuda_stream);

/**
 * cacaosim_cuda_d2h_f32() - Enqueue an asynchronous Device-to-Host `float32` copy
 * @h_dst:       Destination CPU host buffer `[nelem]` (no-op if `NULL`)
 * @d_src:       Source GPU device buffer `[nelem]` (no-op if `NULL`)
 * @nelem:       Number of `float` elements to transfer (`0` is a no-op)
 * @cuda_stream: Optional `cudaStream_t` (`NULL` uses default stream `0`)
 *
 * Role:
 *   Downloads GPU-computed `[OPD, Amp]` beams, DM surfaces, or float32 detector frames back to
 *   host / SHM memory on `@cuda_stream`.
 *
 * Rationale:
 *   Enqueues asynchronously so multiple slices (e.g., `OPD` and `Amp`) queue back-to-back before
 *   a single `cacaosim_cuda_stream_sync()` completes the transaction.
 */
void cacaosim_cuda_d2h_f32(
    float       *h_dst,
    const float *d_src,
    uint64_t     nelem,
    void        *cuda_stream);

/**
 * cacaosim_cuda_build_pupil_padded() - Synthesize zero-padded complex pupil electric field on GPU
 * @d_pup_amp:   Device telescope/beam pupil amplitude `[nwf * nwf]` (non-NULL)
 * @d_atm_amp:   Optional device atmospheric scintillation amplitude `[nwf * nwf]` (`NULL` -> `1.0`)
 * @d_wf_in:     Optional device input wavefront `[nwf * nwf]` (`NULL` -> `0.0`)
 * @d_dm_opd:    Optional device DM surface OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0`)
 * @d_ncpa_opd:  Optional device static NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0`)
 * @wf_to_opd:   Scale factor converting `@d_wf_in` into `[um]` OPD
 * @opd_to_rad:  Wavenumber `2 * pi / lambda_um` `[rad / um]` converting OPD into optical phase
 * @nwf:         Square input pupil grid dimension `[pixels]`
 * @nfft:        Square output FFT complex grid dimension `[pixels]` (`>= nwf`)
 * @d_out_cplx:  Destination `cuFloatComplex` (`float2`) buffer `[nfft * nfft]`
 * @cuda_stream: Optional `cudaStream_t` handle (`NULL` for default stream)
 *
 * Role:
 *   Hot-path GPU pupil-plane constructor shared by FFT-based optical propagators (`pyrwfs`,
 *   `fpcam` FFT mode, etc.) prior to `cufftExecC2C`.
 *
 * Rationale:
 *   Fuses window zero-padding (`nwf -> nfft` centered at `(nfft - nwf) / 2`), amplitude product,
 *   net residual OPD (`wf_in * wf_to_opd - dm_opd + ncpa_opd`), and hardware `__sincosf` Euler
 *   phasor evaluation `A * exp(i * phi)` into a single global-memory write pass, eliminating a
 *   separate `cudaMemsetAsync` over `nfft^2` complex elements on every subframe.
 */
void cacaosim_cuda_build_pupil_padded(
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
    void        *cuda_stream);

/**
 * cacaosim_cuda_detector_readout() - Run stateless GPU camera readout, noise, and ADU quantization
 * @cfg:         Detector radiometric, shutter, noise, gain, bias, and datatype configuration
 * @d_opt_int:   Device input optical intensity `[npix]`
 * @npix:        Total number of detector pixels
 * @norm_total:  Normalization divisor when `> 1e-20f` (`flux_scale = cfg->flux_ph / norm_total`)
 * @seed:        Base 64-bit random seed
 * @frame_seq:   Monotonic frame sequence counter ensuring independent noise across exposures
 * @d_out_u16:   Device `uint16_t` output buffer `[npix]` when `cfg->out_dtype == 0` (else `NULL`)
 * @d_out_f32:   Device `float` output buffer `[npix]` when `cfg->out_dtype != 0` (else `NULL`)
 * @cuda_stream: Optional `cudaStream_t` handle
 *
 * Role:
 *   Hot-path GPU counterpart to `cacaosim_detector_readout_cpu()`, executed at the end of every
 *   CUDA sensor exposure (`pyrwfs`, `shwfs`, `zwfs`, `fpcam`).
 *
 * Rationale:
 *   Uses a stateless counter-based 10-round Philox-2x32 PRNG keyed by `(pixel_idx, frame_seq,
 *   seed)` rather than stateful `curandState` arrays, requiring zero bytes of persistent PRNG VRAM,
 *   incurring zero state-load/store memory traffic, and remaining fully compatible with CUDA Graph
 *   capture/replay when `frame_seq` or noiseless mode is used.
 */
void cacaosim_cuda_detector_readout(
    const cacaosim_det_cfg_t *cfg,
    const float              *d_opt_int,
    uint64_t                  npix,
    float                     norm_total,
    uint64_t                  seed,
    uint64_t                  frame_seq,
    uint16_t                 *d_out_u16,
    float                    *d_out_f32,
    void                     *cuda_stream);

/**
 * cacaosim_cuda_beam_init() - Synthesize `[OPD, Amp]` pupil beam on GPU from atmosphere & pupil
 * @d_pup_amp:   Device static pupil amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_atm_amp:   Optional device scintillation amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_wf_in:     Optional device external wavefront `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @d_ncpa_opd:  Optional device entrance NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @wf_to_opd:   Scale converting `@d_wf_in` to `[um]` OPD
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Device output amplitude `[nwf * nwf]`
 * @cuda_stream: Optional `cudaStream_t` handle
 *
 * Role:
 *   GPU counterpart to `cacaosim_beam_init_cpu()`, assembling initial device-resident beams in
 *   `mkbeam` and fused `htsim`.
 *
 * Rationale:
 *   Keeps beam synthesis entirely on device so fused GPU benches avoid host round-trips between
 *   turbulence ingestion and downstream DM/coronagraph/WFS stages.
 */
void cacaosim_cuda_beam_init(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream);

/**
 * cacaosim_cuda_beam_init_lerp() - Synthesize `[OPD, Amp]` beam on GPU with linear WF interpolation
 * @d_pup_amp:   Device static pupil amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_atm_amp:   Optional device scintillation amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_wf_prev:   Optional device previous base-rate wavefront `[nwf * nwf]`
 * @d_wf_curr:   Optional device current base-rate wavefront `[nwf * nwf]`
 * @wf_alpha:    Sub-tick linear blend weight in `[0, 1]` (`0` = `@d_wf_prev`, `1` = `@d_wf_curr`)
 * @d_ncpa_opd:  Optional device entrance NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @wf_to_opd:   Scale converting wavefront units to `[um]` OPD
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Device output amplitude `[nwf * nwf]`
 * @cuda_stream: Optional `cudaStream_t` handle
 *
 * Role:
 *   GPU counterpart to `cacaosim_beam_init_lerp_cpu()`, blending two device-resident `1x`
 *   turbulence screens across `K` oversampled sub-ticks with zero extra PCIe transfers.
 *
 * Rationale:
 *   Uploads each new `outarraypha` screen once per base period and evaluates all `K` sub-ticks
 *   purely in VRAM bandwidth (`< 2 us` for `256 x 256`).
 */
void cacaosim_cuda_beam_init_lerp(
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
    void        *cuda_stream);

/**
 * cacaosim_cuda_beam_apply_dm() - Subtract reflective DM OPD surface from a beam on GPU
 * @d_in_opd:    Device input beam OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @d_in_amp:    Device input beam amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_dm_opd:    Device rendered DM OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output residual OPD `[nwf * nwf]` in `[um]` (may alias `@d_in_opd`)
 * @d_out_amp:   Device output amplitude `[nwf * nwf]` (may alias `@d_in_amp` or be `NULL`)
 * @cuda_stream: Optional `cudaStream_t` handle
 *
 * Role:
 *   GPU counterpart to `cacaosim_beam_apply_dm_cpu()`, subtracting `ctx->d_opd_out` directly in
 *   VRAM inside `dmsim` and fused `htsim`.
 *
 * Rationale:
 *   Avoids copying the `nwf x nwf` DM surface back to host CPU just to correct an inline beam,
 *   and skips amplitude writes when `d_out_amp == d_in_amp`.
 */
void cacaosim_cuda_beam_apply_dm(
    const float *d_in_opd,
    const float *d_in_amp,
    const float *d_dm_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream);

/**
 * cacaosim_cuda_beam_relay() - Apply 2D registration, arm NCPA, and arm pupil mask on GPU
 * @d_in_opd:    Device input OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @d_in_amp:    Device input amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_ncpa_opd:  Optional device arm NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @d_pup_mask:  Optional device arm pupil stop mask `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @reg:         Host pointer to 2D affine registration parameters (`NULL` -> identity)
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]` (distinct from `@d_in_opd` when warping)
 * @d_out_amp:   Device output amplitude `[nwf * nwf]` (distinct from `@d_in_amp` when warping)
 * @cuda_stream: Optional `cudaStream_t` handle
 *
 * Role:
 *   GPU counterpart to `cacaosim_beam_relay_cpu()` for dichroic/beamsplitter arms in `beamsplit`
 *   and fused `htsim`.
 *
 * Rationale:
 *   Selects between a coalesced 1-to-1 elementwise kernel (`kernel_beam_relay_ident`) when `@reg`
 *   is identity and a parallel 2D inverse-affine bilinear gather kernel (`kernel_beam_relay_warp`)
 *   when pupil shift, rotation, or magnification is active.
 */
void cacaosim_cuda_beam_relay(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_ncpa_opd,
    const float          *d_pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *d_out_opd,
    float                *d_out_amp,
    void                 *cuda_stream);

/**
 * cacaosim_cuda_beam_apply_wfe() - Add scaled static WFE OPD (+ optional registration) on GPU
 * @d_in_opd:    Device input OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @d_in_amp:    Device input amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @d_wfe_opd:   Device static WFE map `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @wfe_to_um:   Conversion & gain multiplier to `[um]` (`1.0e6f * wfe_gain` for `[m] -> [um]`)
 * @reg:         Optional host 2D affine pupil registration (`NULL` -> identity)
 * @nwf:         Square pupil grid dimension `[pixels]`
 * @d_out_opd:   Device output OPD `[nwf * nwf]` in `[um]`
 * @d_out_amp:   Optional device output amplitude `[nwf * nwf]`
 * @cuda_stream: Optional `cudaStream_t` handle
 *
 * Role:
 *   GPU counterpart to `cacaosim_beam_apply_wfe_cpu()` for `cacaosim.wfe` and fused `htsim`.
 *
 * Rationale:
 *   Dispatches `kernel_beam_wfe_ident` for aligned optics and `kernel_beam_wfe_warp` when `@reg`
 *   is non-trivial, keeping inline WFE perturbations zero-copy inside GPU optical trains.
 */
void cacaosim_cuda_beam_apply_wfe(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_wfe_opd,
    float                 wfe_to_um,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *d_out_opd,
    float                *d_out_amp,
    void                 *cuda_stream);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_CUDA_H */
