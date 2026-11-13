// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_detector.h
 * @brief   CPU detector PRNG pool and radiometric-to-ADU camera readout interface
 *
 * Architectural Role:
 *   Defines the persistent multi-threaded CPU PRNG pool (`cacaosim_det_pool_t`) and the unified
 *   CPU camera digitization entry point (`cacaosim_detector_readout_cpu`) shared by all four
 *   `cacaosim` optical sensors (`pyrwfs`, `shwfs`, `zwfs`, and `fpcam`). Transforms accumulated
 *   optical intensity maps into realistic `uint16` or `float32` camera frames modeling shutter
 *   state, radiometric photon scaling, dark/background electrons, quantum Poisson shot noise,
 *   Gaussian readout noise (RON), conversion gain (`e-/ADU`), bias offset, and full-well clipping.
 *
 * Design Rationale & Motivation:
 *   1. Pre-allocated per-sensor PRNG pool: Allocating cache-line-aligned `cacaosim_rng_t` states
 *      once during sensor `init()` (`cacaosim_det_pool_init`) upholds the strict zero-allocation
 *      hot-loop rule while giving each sensor instance an isolated, reproducible noise stream.
 *   2. Flexible flux normalization (`norm_total`): Wavefront sensors calibrate `norm_total` to the
 *      unaberrated pupil integrated intensity so focal-stop or Lyot vignetting naturally attenuates
 *      detected flux, whereas normalized PSFs pass `norm_total = 1.0` or `0.0`; supporting both
 *      via a single parameter avoids rescaling intensity buffers in a separate memory pass.
 *   3. Exact behavioral symmetry with `cacaosim_cuda_detector_readout`: Sharing
 *      `cacaosim_det_cfg_t` and identical radiometric equations guarantees CPU and CUDA backends
 *      produce statistically equivalent camera frames across dark, calibration (`flux_ph < 0`),
 *      and closed-loop modes.
 */

#ifndef CACAOSIM_DETECTOR_H
#define CACAOSIM_DETECTOR_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_rng.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * struct cacaosim_det_pool_t - Persistent OpenMP thread-local PRNG state pool for CPU readout
 * @states:     64-byte aligned array of `@nb_threads` independent `cacaosim_rng_t` states
 *              (`64` bytes per slot, eliminating false sharing across OpenMP workers).
 * @nb_threads: Number of pre-allocated OpenMP thread slots (`>= 1` when initialized, `0` freed).
 * @frame_seq:  Monotonically incremented stochastic readout frame counter for diagnostics/parity.
 */
typedef struct
{
    cacaosim_rng_t * states;
    int32_t          nb_threads;
    uint64_t         frame_seq;
} cacaosim_det_pool_t;

/**
 * cacaosim_det_pool_init() - Allocate and seed per-thread PRNG states for CPU detector readout
 * @pool:        Output detector PRNG pool structure to initialize (non-NULL)
 * @max_threads: Maximum OpenMP worker count (`<= 0` automatically queries `omp_get_max_threads()`)
 * @seed:        Base 64-bit random seed (`0` substitutes a deterministic non-zero default seed)
 *
 * Role:
 *   Cold-path setup routine called during `cacaosim_{pyrwfs,shwfs,zwfs,fpcam}_init()` to prepare
 *   lock-free stochastic noise generation before real-time frame acquisition begins.
 *
 * Rationale:
 *   Allocates all thread PRNG states up front via `cacaosim_aligned_calloc()` so each 64-byte
 *   `cacaosim_rng_t` sits on a dedicated CPU cache line, and seeds slot `t` with `(eff_seed, t)`
 *   so OpenMP threads never contend on shared state or locks during kHz frame readouts.
 *
 * Return: `0` on success, `-1` if `@pool == NULL` or aligned heap allocation fails.
 */
int cacaosim_det_pool_init(
    cacaosim_det_pool_t *pool,
    int32_t              max_threads,
    uint64_t             seed);

/**
 * cacaosim_det_pool_free() - Release heap memory owned by a CPU detector PRNG pool
 * @pool: Detector PRNG pool to tear down (safe if `NULL` or already zeroed/freed)
 *
 * Role:
 *   Cold-path cleanup counterpart to `cacaosim_det_pool_init()`, invoked from sensor `free()`
 *   routines when shutting down or reconfiguring an optical bench.
 *
 * Rationale:
 *   Frees `pool->states` and zeroes `states` and `nb_threads` to make repeated teardown calls
 *   idempotent and prevent dangling pointer reuse.
 */
void cacaosim_det_pool_free(cacaosim_det_pool_t *pool);

/**
 * cacaosim_detector_readout_cpu() - Digitize 2D optical intensity into `uint16` or `float32` ADUs
 * @pool:       Initialized per-thread PRNG pool (may be `NULL` for noiseless or single-thread use)
 * @cfg:        Detector radiometric, noise, gain, bias, shutter, and datatype configuration
 * @opt_int:    Input 2D optical intensity array (`@npix` floats, non-aliasing with outputs)
 * @npix:       Total number of camera pixels (`ncam * ncam`)
 * @norm_total: Reference intensity divisor: when `> 1e-20f` and `cfg->flux_ph >= 0`, scales
 *              `@opt_int` by `cfg->flux_ph / norm_total`; otherwise scales directly by `flux_ph`
 * @out_u16:    Destination `uint16_t` pixel buffer when `cfg->out_dtype == 0` (else `NULL`)
 * @out_f32:    Destination `float` pixel buffer when `cfg->out_dtype != 0` (else `NULL`)
 *
 * Role:
 *   Hot-path terminal stage of CPU sensor exposures (`pyrwfs`, `shwfs`, `zwfs`, `fpcam`), writing
 *   directly into either local buffers or mapped `ImageStreamIO` shared-memory camera arrays.
 *
 * Rationale:
 *   Fuses flux scaling, dark current addition, optional Poisson shot noise, optional Gaussian RON,
 *   gain conversion, pedestal bias, saturation clamping, and nearest-integer `uint16` rounding
 *   into a single cache-streaming pass over `@npix` pixels, dispatching a pure SIMD fast path when
 *   noise is disabled and a lock-free OpenMP thread-local PRNG path when stochastic noise is on.
 */
void cacaosim_detector_readout_cpu(
    cacaosim_det_pool_t       *pool,
    const cacaosim_det_cfg_t  *cfg,
    const float *__restrict    opt_int,
    uint64_t                   npix,
    float                      norm_total,
    uint16_t *__restrict       out_u16,
    float *__restrict          out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_DETECTOR_H */
