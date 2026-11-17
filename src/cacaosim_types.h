// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_types.h
 * @brief   Shared constants, enums, detector config, and aligned memory helpers for cacaosim
 *
 * Architectural Role:
 *   Foundational leaf header included by every CPU compute unit, CUDA `.cu` kernel file, `.bench`
 *   orchestrator, and FPS wrapper in `cacaosim`. Establishes uniform physical unit conventions,
 *   compute backend selectors, camera readout configuration layouts, 64-byte SIMD/cache-line
 *   aligned memory allocation, and single-precision wavefront scaling.
 *
 * Design Rationale & Motivation:
 *   1. Zero external header coupling: Depends solely on standard C headers (`<stddef.h>`,
 *      `<stdint.h>`, `<stdlib.h>`), keeping `ImageStreamIO`, `milk`, `fftw3`, and `cuda_runtime.h`
 *      out of foundational type definitions so C11 and `nvcc` C++ translation units share exact
 *      binary struct layouts without namespace collisions.
 *   2. 64-byte alignment (`CACAOSIM_ALIGN_BYTES = 64u`): Simultaneously satisfies 64-byte CPU
 *      L1/L2 cache-line boundaries (eliminating OpenMP false sharing across per-thread states),
 *      512-bit AVX-512 / 256-bit AVX2 aligned vector load/store instructions, and FFTW3f optimal
 *      SIMD planner alignment.
 *   3. Single-precision (`float`) physical constants: Explicit `f`-suffixed constants
 *      (`CACAOSIM_PI_F`, `CACAOSIM_TWO_PI_F`) guarantee IEEE-754 binary32 evaluation inside hot
 *      inner loops, preventing silent C promotion to `double` that would halve CPU SIMD lane
 *      width and throttle GPU FP32 execution units by up to 32x--64x on consumer/workstation GPUs.
 *   4. Achromatic internal OPD convention: Internally representing wavefront perturbations as
 *      optical path difference in micrometers (`[um]`) makes reflective mirror surfaces and
 *      geometric atmospheric screens invariant across wavelength, while `cacaosim_wf_to_opd_scale`
 *      bridges external turbulence generators supplying phase in radians.
 */

#ifndef CACAOSIM_TYPES_H
#define CACAOSIM_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Memory alignment quantum [bytes] matching x86-64/ARM64 cache lines and AVX-512 vector width,
 * plus single-precision pi and 2*pi constants avoiding double-precision promotion in float kernels.
 */
#define CACAOSIM_ALIGN_BYTES 64u
#define CACAOSIM_MAX_LAMBDA  32
#define CACAOSIM_PI_F        3.14159265358979323846f
#define CACAOSIM_TWO_PI_F    6.28318530717958647692f

/**
 * enum cacaosim_wfunit_t - Input atmospheric/external wavefront physical unit convention
 * @CACAOSIM_WFUNIT_OPD_UM:    Input array values represent optical path difference in microns
 *                             `[um]`; achromatic across sensing and science wavelengths.
 * @CACAOSIM_WFUNIT_PHASE_RAD: Input array values represent monochromatic phase in radians `[rad]`
 *                             defined at a specified reference wavelength `ref_lambda` `[um]`.
 *
 * Rationale: Different `cacao` / `milk` turbulence generators emit either geometric OPD `[um]` or
 * phase `[rad]` at 0.50 um; supporting both at zero runtime cost via a scalar pre-multiplier
 * avoids inserting an extra stream-conversion process in real-time simulations.
 */
typedef enum
{
    CACAOSIM_WFUNIT_OPD_UM    = 0, /**< Optical path difference in microns */
    CACAOSIM_WFUNIT_PHASE_RAD = 1  /**< Phase in radians at reference wavelength */
} cacaosim_wfunit_t;

/**
 * enum cacaosim_backend_t - Compute execution backend policy for optical and hardware engines
 * @CACAOSIM_BACKEND_AUTO: Query `cacaosim_cuda_available()` at init time; run on CUDA GPU when
 *                         present and automatically fall back to CPU OpenMP+SIMD otherwise.
 * @CACAOSIM_BACKEND_CPU:  Force multi-threaded CPU execution (OpenMP + SIMD + FFTW3f / OpenBLAS)
 *                         even when a CUDA GPU is installed (useful for deterministic parity tests
 *                         or reserving the GPU for RTC reconstruction).
 * @CACAOSIM_BACKEND_CUDA: Request CUDA GPU execution; gracefully falls back to CPU if built
 *                         without CUDA or if no compatible GPU device is accessible at runtime.
 */
typedef enum
{
    CACAOSIM_BACKEND_AUTO = 0, /**< Prefer CUDA GPU when available, else CPU */
    CACAOSIM_BACKEND_CPU  = 1, /**< Force CPU OpenMP + SIMD + FFTW3f/BLAS */
    CACAOSIM_BACKEND_CUDA = 2  /**< Force CUDA GPU (falls back if unavailable) */
} cacaosim_backend_t;

/**
 * enum cacaosim_dtype_t - Detector output stream pixel datatype selector
 * @CACAOSIM_DTYPE_U16: 16-bit unsigned integer (`_DATATYPE_UINT16`, `[0..65535]` ADU), emulating
 *                      raw CameraLink / CoaXPress sCMOS, EMCCD, and InGaAs hardware grabbers fed
 *                      into `cacao` real-time WFS acquisition pipelines.
 * @CACAOSIM_DTYPE_F32: 32-bit IEEE-754 float (`_DATATYPE_FLOAT`), supporting high-dynamic-range
 *                      coronagraphic PSFs, calibrated float ADU frames, or unquantized raw optical
 *                      intensity passthrough (`flux_ph < 0`).
 */
typedef enum
{
    CACAOSIM_DTYPE_U16 = 0, /**< 16-bit unsigned integer ADU */
    CACAOSIM_DTYPE_F32 = 1  /**< 32-bit IEEE floating-point ADU / intensity */
} cacaosim_dtype_t;

/**
 * struct cacaosim_det_cfg_t - Shared radiometric and electronics readout model for all cameras
 * @flux_ph:      Total incident/detected optical photoevents per exposure across the full aperture
 *                `[photons/frame]` (or `[e-/frame]` when QE is folded into
 *                `cacaosim_spec_band_flux`). Sentinel `< 0.0f` activates raw intensity passthrough
 *                mode: bypasses dark current, Poisson shot noise, RON, inverse gain, bias, and
 *                `[0, sat_adu]` clamping so raw optical intensity is written directly to float32
 *                output.
 * @dark_e:       Mean thermal dark current plus diffuse sky/thermal background `[e-/pixel/frame]`,
 *                added prior to Poisson shot-noise sampling so background photon/dark shot noise
 *                is physically included.
 * @ron_e:         Additive zero-mean Gaussian readout electronics noise `[e- RMS / pixel / frame]`;
 *                disabled when `<= 0.0f` or in raw passthrough mode.
 * @gain_eadu:    Detector inverse conversion gain `[e-/ADU]`; photoelectrons are divided by
 *                `gain_eadu` (when `> 1e-6f`) before adding `bias_adu`.
 * @bias_adu:     Electronic pedestal / black-level offset `[ADU]` added after gain conversion to
 *                prevent negative Gaussian RON excursions from clipping at zero ADU.
 * @sat_adu:      Full-well / ADC saturation ceiling `[ADU]` (`<= 0.0f` defaults to `65535.0f`);
 *                `uint16` output additionally clamps at `65535.0f` to prevent integer wrap-around.
 * @shutter:      Optical shutter state (`1` = open / illuminated, `0` = closed / dark frame where
 *                optical flux is zeroed while preserving `dark_e`, Poisson dark noise, RON, and
 *                `bias_adu` for realistic `cacao` dark-calibration sequences).
 * @photon_noise: Quantum shot-noise toggle (`1` = sample `Poisson(phe)`, `0` = noiseless mean).
 * @out_dtype:    Destination pixel format (`cacaosim_dtype_t`: `0` = uint16, `1` = float32).
 *
 * Rationale: Embedding this identical POD struct inside `pyrwfs`, `shwfs`, `zwfs`, and `fpcam`
 * guarantees uniform camera physics and identical CPU/CUDA readout behavior across the bench.
 */
typedef struct
{
    float    flux_ph;      /**< Total optical flux per frame [photons] (<0: raw intensity) */
    float    dark_e;       /**< Per-pixel dark + background level [e-/frame] */
    float    ron_e;        /**< Gaussian readout noise [e- RMS] */
    float    gain_eadu;    /**< Detector conversion gain [e-/ADU] */
    float    bias_adu;     /**< Detector zero-flux bias offset [ADU] */
    float    sat_adu;      /**< Saturation ceiling [ADU] */
    int32_t  shutter;      /**< Optical shutter (1 = open, 0 = closed / dark frame) */
    int32_t  photon_noise; /**< Enable Poisson shot noise (1 = on, 0 = off) */
    int32_t  out_dtype;    /**< Output format: 0 = uint16, 1 = float32 */
} cacaosim_det_cfg_t;

/**
 * cacaosim_aligned_calloc() - Allocate zero-initialized 64-byte aligned heap memory
 * @count: Number of elements to allocate (if 0, safely allocates one 64-byte block)
 * @size:  Size in bytes of each element (if 0, safely allocates one 64-byte block)
 *
 * Role:
 *   Cold-path memory allocator used during module `init()` across all `cacaosim` subsystems to
 *   allocate pupil grids, FFTW3f complex planes, BLAS MFT matrices, sparse DM gather stencils,
 *   and cache-line-padded OpenMP PRNG state pools. Must NEVER be called inside hot frame loops.
 *
 * Rationale:
 *   Standard `calloc()` only guarantees `max_align_t` (typically 16 bytes), which causes unaligned
 *   penalties or faults with AVX-512 / FFTW SIMD instructions and allows adjacent OpenMP thread
 *   structures to straddle the same 64-byte cache line. `posix_memalign()` is used instead of C11
 *   `aligned_alloc()` for universal availability across POSIX C11 and C++ (`nvcc` host) standard
 *   libraries; padding `total` to an integer multiple of `CACAOSIM_ALIGN_BYTES` satisfies strict
 *   vectorized tail loops and `aligned_alloc`-style invariants, followed by deterministic zeroing.
 *
 * Return: 64-byte aligned, zero-filled heap pointer (freeable with standard `free()`), or `NULL`
 *         if `posix_memalign()` fails.
 */
static inline void *cacaosim_aligned_calloc(
    size_t count,
    size_t size)
{
    size_t total = count * size;
    /* Guard zero-sized requests so posix_memalign always receives a valid non-zero multiple */
    if (total == 0)
    {
        total = CACAOSIM_ALIGN_BYTES;
    }
    /* Pad allocation up to a full 64-byte cache-line / AVX-512 register multiple */
    size_t rem = total % CACAOSIM_ALIGN_BYTES;
    if (rem != 0)
    {
        total += (CACAOSIM_ALIGN_BYTES - rem);
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, CACAOSIM_ALIGN_BYTES, total) != 0 || ptr == NULL)
    {
        return NULL;
    }
    __builtin_memset(ptr, 0, total);
    return ptr;
}

/**
 * cacaosim_wf_to_opd_scale() - Compute multiplicative factor converting input WF map to microns OPD
 * @wf_unit:    Input wavefront unit (`cacaosim_wfunit_t`: `0` = OPD [um], `1` = phase [rad])
 * @ref_lambda: Reference wavelength `[um]` at which phase is defined when `@wf_unit == 1`
 * @turb_gain:  Dimensionless user turbulence amplitude multiplier (e.g., `1.0` nominal, `0.0` flat)
 *
 * Role:
 *   Evaluated once per subframe at the entry of `mkbeam`, `pyrwfs`, `shwfs`, `zwfs`, `fpcam`, and
 *   `htsim` so inner pixel kernels convert arbitrary external turbulence streams into microns of
 *   OPD using a single fused multiply (`opd_um = wf_to_opd * wf_in[p]`).
 *
 * Rationale:
 *   Folding both unit conversion ($\text{OPD} = \phi \cdot \lambda_{\text{ref}} / 2\pi$) and live
 *   `turb_gain` scaling into one scalar hoists divisions and unit branches completely out of the
 *   $O(N_{\text{wf}}^2)$ SIMD/CUDA pixel loops. Clamping degenerate `ref_lambda <= 1e-4f` to
 *   `0.50f` um (astronomical V-band) prevents silent zeroing if an uninitialized parameter is
 *   passed while `CACAOSIM_WFUNIT_PHASE_RAD` is selected.
 *
 * Return: Multiplicative scale factor in `[um / input_unit]`.
 */
static inline float cacaosim_wf_to_opd_scale(
    int32_t wf_unit,
    float   ref_lambda,
    float   turb_gain)
{
    if (wf_unit == CACAOSIM_WFUNIT_PHASE_RAD)
    {
        /* Convert radians at ref_lambda [um] to geometric OPD [um]: OPD = gain * phi * lam / 2pi */
        float lam = (ref_lambda > 1e-4f) ? ref_lambda : 0.50f;
        return turb_gain * (lam / CACAOSIM_TWO_PI_F);
    }
    return turb_gain;
}

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_TYPES_H */
