// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_detector.h
 * @brief   CPU and CUDA detector readout engine (flux, shutter, Poisson, RON, gain, bias, ADU)
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
 * @brief Reusable detector PRNG pool across OpenMP threads.
 */
typedef struct
{
    cacaosim_rng_t * states;
    int32_t          nb_threads;
    uint64_t         frame_seq;
} cacaosim_det_pool_t;

/**
 * cacaosim_det_pool_init() - Allocate per-thread PRNG states for CPU detector readout
 * @pool:        Output detector pool
 * @max_threads: Maximum OpenMP thread count (<= 0 auto-queries omp_get_max_threads)
 * @seed:        Base 64-bit random seed
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int cacaosim_det_pool_init(
    cacaosim_det_pool_t *pool,
    int32_t              max_threads,
    uint64_t             seed);

/**
 * cacaosim_det_pool_free() - Release per-thread PRNG pool memory
 * @pool: Detector pool to release
 */
void cacaosim_det_pool_free(cacaosim_det_pool_t *pool);

/**
 * cacaosim_detector_readout_cpu() - Convert normalized/raw optical intensity to camera pixels
 * @pool:       Initialized PRNG pool
 * @cfg:        Detector physics configuration
 * @opt_int:    Input 2D optical intensity buffer (npix floats; normalized to sum=1 or raw)
 * @npix:       Total pixel count
 * @norm_total: If > 0 and cfg->flux_ph >= 0, multiplies opt_int by (cfg->flux_ph / norm_total);
 *              if <= 0 and cfg->flux_ph >= 0, multiplies opt_int directly by cfg->flux_ph
 * @out_u16:    Destination uint16 buffer when cfg->out_dtype == 0 (else NULL)
 * @out_f32:    Destination float32 buffer when cfg->out_dtype != 0 (else NULL)
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
