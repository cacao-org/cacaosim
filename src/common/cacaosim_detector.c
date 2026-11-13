// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_detector.c
 * @brief   Multi-threaded OpenMP and SIMD CPU camera readout engine implementation
 *
 * Architectural Role:
 *   Implements `cacaosim_det_pool_init()`, `cacaosim_det_pool_free()`, and the hot-loop CPU
 *   camera digitization pipeline `cacaosim_detector_readout_cpu()` used by `pyrwfs`, `shwfs`,
 *   `zwfs`, and `fpcam` whenever executing on the CPU backend.
 *
 * Design Rationale & Motivation:
 *   1. Dual-path execution (noiseless SIMD vs. stochastic OpenMP): Calibration workflows in
 *      `cacao` (response-matrix pokes, reference slope capture, raw PSF verification) disable
 *      photon and readout noise (`!do_poiss && !do_ron`). Separating that case into dedicated
 *      `#pragma omp parallel for simd` loops lets the compiler emit branchless AVX2/AVX-512 packed
 *      FMA/min/max/cvt instructions unhindered by data-dependent Knuth/Marsaglia `while` loops.
 *   2. Register-promoted PRNG state in stochastic mode: Inside the OpenMP parallel region, each
 *      worker copies `pool->states[tid]` into a stack-local `local_rng` before entering the static
 *      pixel loop and writes the advanced state back once on exit. This proves to the C compiler
 *      that `rng->s[0..3]` cannot alias output buffers or neighbor threads, keeping the 128-bit
 *      state in scalar CPU registers across thousands of pixels.
 *   3. Workload-gated OpenMP threading (`npix >= 16384` noiseless, `npix >= 8192` noisy): Small
 *      WFS detectors (e.g., `32x32` or `64x64` pixels) complete in `< 2 us` on a single core—less
 *      than an OpenMP barrier fork/join. Gating `nth` on `npix` avoids multi-threading slowdown on
 *      compact WFS formats while scaling across cores on large (`256x256`..`2048x2048`) cameras.
 *   4. Physical signal chain ordering: Evaluates `phe = max(opt_int, 0) * flux_scale + dark_e`
 *      first so both optical photons and thermal dark/sky background undergo quantum Poisson
 *      fluctuations prior to additive electronics RON, inverse gain, pedestal bias, and ADC clip.
 */

#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_detector.h"

/**
 * cacaosim_det_pool_init() - Allocate and seed cache-line-aligned per-thread CPU PRNG states
 * @pool:        Destination pool descriptor (non-NULL)
 * @max_threads: Requested thread count (`<= 0` auto-detects via `omp_get_max_threads()`)
 * @seed:        64-bit base seed (`0` falls back to ASCII `"CACAOSIM"` = `0x434143414f53494dULL`)
 *
 * Role:
 *   Cold-path resource initializer invoked once when constructing a CPU-capable WFS or science
 *   camera context.
 *
 * Rationale:
 *   Marked `MILK_COLD` to keep setup error branches out of instruction caches shared with hot
 *   readout code. Uses `cacaosim_aligned_calloc()` with `64`-byte `cacaosim_rng_t` elements so
 *   `pool->states[t]` never shares a cache line with `pool->states[t + 1]`, and substitutes a
 *   fixed non-zero ASCII mnemonic seed when `seed == 0` for reproducible default runs.
 *
 * Return: `0` on success, `-1` on `NULL` input or memory exhaustion.
 */
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

    /* Allocate 64-byte aligned array; each 64-byte element occupies its own CPU cache line */
    pool->states = (cacaosim_rng_t *) cacaosim_aligned_calloc((size_t) nth, sizeof(cacaosim_rng_t));
    if (pool->states == NULL)
    {
        pool->nb_threads = 0;
        return -1;
    }

    pool->nb_threads = nth;
    pool->frame_seq  = 0;

    /* Default seed 0x434143414f53494dULL encodes ASCII "CACAOSIM" for deterministic defaults */
    uint64_t eff_seed = (seed != 0ULL) ? seed : 0x434143414f53494dULL;
    for (int32_t t = 0; t < nth; t++)
    {
        cacaosim_rng_seed(&pool->states[t], eff_seed, (uint64_t) t);
    }
    return 0;
}

/**
 * cacaosim_det_pool_free() - Free per-thread PRNG state buffer and reset pool metadata
 * @pool: Detector PRNG pool to release (safe if `NULL` or uninitialized)
 *
 * Role:
 *   Cold-path destructor paired with `cacaosim_det_pool_init()`.
 *
 * Rationale:
 *   Nulls `pool->states` and resets `pool->nb_threads = 0` after `free()` so accidental double
 *   cleanup or subsequent inspection sees an empty, safe descriptor.
 */
void MILK_COLD cacaosim_det_pool_free(cacaosim_det_pool_t *pool)
{
    if (pool != NULL)
    {
        free(pool->states);
        pool->states     = NULL;
        pool->nb_threads = 0;
    }
}

/**
 * cacaosim_detector_readout_cpu() - Execute radiometric scaling, noise, and ADU quantization on CPU
 * @pool:       Optional initialized per-thread PRNG pool (`NULL` falls back to single-threaded)
 * @cfg:        Active detector model configuration (non-NULL)
 * @opt_int:    Input 2D optical intensity buffer `[npix]` (`restrict`-qualified)
 * @npix:       Total detector pixel count
 * @norm_total: Reference flux normalization integral (`> 1e-20f` divides `cfg->flux_ph`)
 * @out_u16:    Output `uint16_t` ADU frame `[npix]` when `cfg->out_dtype == 0`
 * @out_f32:    Output `float` ADU / raw intensity frame `[npix]` when `cfg->out_dtype != 0`
 *
 * Role:
 *   Hot-path (`MILK_HOT`) CPU camera readout engine invoked at the end of every sensor exposure.
 *
 * Rationale:
 *   Hoists all invariant radiometric factors (`flux_scale`, `inv_gain = 1 / gain_eadu`, `dark_e`,
 *   `bias_adu`, `sat_adu`, and raw-passthrough flags) before looping so per-pixel arithmetic
 *   uses only fast FMA/multiply-add operations without divisions. Preserves signed/unclipped
 *   values in `out_f32` when `cfg->flux_ph < 0.0f` (raw intensity passthrough), and applies `+0.5f`
 *   half-ADU bias before `(uint16_t)` truncation for unbiased nearest-integer ADC quantization.
 */
void MILK_HOT cacaosim_detector_readout_cpu(
    cacaosim_det_pool_t       *pool,
    const cacaosim_det_cfg_t  *cfg,
    const float *__restrict    opt_int,
    uint64_t                   npix,
    float                      norm_total,
    uint16_t *__restrict       out_u16,
    float *__restrict          out_f32)
{
    /* Hoist configuration flags; negative flux_ph selects raw intensity passthrough mode */
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

    /* Closed shutter blocks optical light while keeping dark_e, RON, and bias active */
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
        /* Only spawn OpenMP threads when >= 16k pixels amortize barrier overhead */
        int nth = (npix >= 16384ULL && pool != NULL) ? pool->nb_threads : 1;
        if (cfg->out_dtype == 0 && out_u16 != NULL)
        {
            const float max_u16 = (sat_adu < 65535.0f) ? sat_adu : 65535.0f;
#pragma omp parallel for simd num_threads(nth) if (parallel: nth > 1)
            for (uint64_t i = 0; i < npix; i++)
            {
                /* Guard small negative FFT/MFT numerical ringing before radiometric scaling */
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
                /* Nearest-integer ADC quantization over [0, min(sat_adu, 65535)] */
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
                /* Preserve raw unclipped intensity when flux_ph < 0; else enforce [0, sat_adu] */
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

        /* Promote thread's cache-line-isolated PRNG state to stack/registers for the loop */
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
            /* Mean photoelectrons: optical signal + dark/sky background prior to shot noise */
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

        /* Commit advanced PRNG state back to pool so the next frame continues the sequence */
        if (pool != NULL && tid < pool->nb_threads)
        {
            pool->states[tid] = local_rng;
        }
    }
}
