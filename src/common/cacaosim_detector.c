// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_detector.c
 * @brief   Multi-threaded SIMD-friendly CPU detector readout implementation
 */

#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_detector.h"

int MILK_COLD cacaosim_det_pool_init(
    cacaosim_det_pool_t *pool,
    int32_t              max_threads,
    uint64_t             seed)
{
    if (pool == NULL)
    {
        return -1;
    }
    int32_t nth = (max_threads > 0) ? max_threads : omp_get_max_threads();
    if (nth < 1)
    {
        nth = 1;
    }

    pool->states = (cacaosim_rng_t *) cacaosim_aligned_calloc((size_t) nth, sizeof(cacaosim_rng_t));
    if (pool->states == NULL)
    {
        pool->nb_threads = 0;
        return -1;
    }

    pool->nb_threads = nth;
    pool->frame_seq  = 0;

    uint64_t eff_seed = (seed != 0ULL) ? seed : 0x434143414f53494dULL;
    for (int32_t t = 0; t < nth; t++)
    {
        cacaosim_rng_seed(&pool->states[t], eff_seed, (uint64_t) t);
    }
    return 0;
}

void MILK_COLD cacaosim_det_pool_free(cacaosim_det_pool_t *pool)
{
    if (pool != NULL)
    {
        free(pool->states);
        pool->states     = NULL;
        pool->nb_threads = 0;
    }
}

void MILK_HOT cacaosim_detector_readout_cpu(
    cacaosim_det_pool_t       *pool,
    const cacaosim_det_cfg_t  *cfg,
    const float *__restrict    opt_int,
    uint64_t                   npix,
    float                      norm_total,
    uint16_t *__restrict       out_u16,
    float *__restrict          out_f32)
{
    const int32_t shutter  = cfg->shutter;
    const int32_t do_poiss = (cfg->photon_noise != 0 && cfg->flux_ph >= 0.0f);
    const int32_t do_ron   = (cfg->ron_e > 0.0f && cfg->flux_ph >= 0.0f);
    const float   ron_e    = cfg->ron_e;
    const float   dark_e   = (cfg->flux_ph >= 0.0f) ? cfg->dark_e : 0.0f;
    const float   inv_gain = (cfg->gain_eadu > 1e-6f && cfg->flux_ph >= 0.0f)
                                 ? (1.0f / cfg->gain_eadu)
                                 : 1.0f;
    const float   bias_adu = (cfg->flux_ph >= 0.0f) ? cfg->bias_adu : 0.0f;
    const float   sat_adu  = (cfg->sat_adu > 0.0f) ? cfg->sat_adu : 65535.0f;

    float flux_scale = 1.0f;
    if (!shutter)
    {
        flux_scale = 0.0f;
    }
    else if (cfg->flux_ph >= 0.0f)
    {
        flux_scale = (norm_total > 1e-20f) ? (cfg->flux_ph / norm_total) : cfg->flux_ph;
    }

    /* Noiseless deterministic fast path: pure SIMD vectorization */
    if (!do_poiss && !do_ron)
    {
        int nth = (npix >= 16384ULL && pool != NULL) ? pool->nb_threads : 1;
        if (cfg->out_dtype == 0 && out_u16 != NULL)
        {
            const float max_u16 = (sat_adu < 65535.0f) ? sat_adu : 65535.0f;
#pragma omp parallel for simd num_threads(nth) if (parallel: nth > 1)
            for (uint64_t i = 0; i < npix; i++)
            {
                float sig = opt_int[i];
                if (sig < 0.0f)
                {
                    sig = 0.0f;
                }
                float adu = (sig * flux_scale + dark_e) * inv_gain + bias_adu;
                if (adu < 0.0f)
                {
                    adu = 0.0f;
                }
                else if (adu > max_u16)
                {
                    adu = max_u16;
                }
                out_u16[i] = (uint16_t) (adu + 0.5f);
            }
        }
        else if (out_f32 != NULL)
        {
#pragma omp parallel for simd num_threads(nth) if (parallel: nth > 1)
            for (uint64_t i = 0; i < npix; i++)
            {
                float sig = opt_int[i];
                if (sig < 0.0f && cfg->flux_ph >= 0.0f)
                {
                    sig = 0.0f;
                }
                float adu = (sig * flux_scale + dark_e) * inv_gain + bias_adu;
                if (cfg->flux_ph >= 0.0f)
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
                out_f32[i] = adu;
            }
        }
        return;
    }

    /* Stochastic path with per-thread lock-free xoshiro128++ state */
    int nth = (pool != NULL && npix >= 8192ULL) ? pool->nb_threads : 1;
    if (pool != NULL)
    {
        pool->frame_seq++;
    }

#pragma omp parallel num_threads(nth) if (nth > 1)
    {
        int             tid = omp_get_thread_num();
        cacaosim_rng_t  local_rng;
        cacaosim_rng_t *rng = &local_rng;

        if (pool != NULL && tid < pool->nb_threads)
        {
            local_rng = pool->states[tid];
        }
        else
        {
            cacaosim_rng_seed(&local_rng, 0x12345678ULL, (uint64_t) tid);
        }

        const float max_u16 = (sat_adu < 65535.0f) ? sat_adu : 65535.0f;

#pragma omp for schedule(static)
        for (uint64_t i = 0; i < npix; i++)
        {
            float sig = opt_int[i];
            if (sig < 0.0f)
            {
                sig = 0.0f;
            }
            float phe = sig * flux_scale + dark_e;
            if (do_poiss)
            {
                phe = cacaosim_rng_poisson_f32(rng, phe);
            }
            if (do_ron)
            {
                phe += ron_e * cacaosim_rng_gauss_f32(rng);
            }
            float adu = phe * inv_gain + bias_adu;

            if (cfg->out_dtype == 0 && out_u16 != NULL)
            {
                if (adu < 0.0f)
                {
                    adu = 0.0f;
                }
                else if (adu > max_u16)
                {
                    adu = max_u16;
                }
                out_u16[i] = (uint16_t) (adu + 0.5f);
            }
            else if (out_f32 != NULL)
            {
                if (adu < 0.0f)
                {
                    adu = 0.0f;
                }
                else if (adu > sat_adu)
                {
                    adu = sat_adu;
                }
                out_f32[i] = adu;
            }
        }

        if (pool != NULL && tid < pool->nb_threads)
        {
            pool->states[tid] = local_rng;
        }
    }
}
