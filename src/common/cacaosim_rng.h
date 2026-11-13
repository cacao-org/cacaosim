// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_rng.h
 * @brief   Lock-free cache-line-isolated `xoshiro128++` PRNG for CPU Poisson and Gaussian noise
 *
 * Architectural Role:
 *   Provides fast, deterministic, thread-local pseudo-random number generation and stochastic
 *   variate samplers (open-interval uniform, Marsaglia polar standard normal, and hybrid
 *   Knuth / Cornish-Fisher Poisson) for CPU camera readout (`cacaosim_detector.c`) across all
 *   wavefront sensors (`pyrwfs`, `shwfs`, `zwfs`) and focal-plane cameras (`fpcam`).
 *
 * Design Rationale & Motivation:
 *   1. Lock-free multi-threaded throughput without false sharing: Standard `rand()` serializes on
 *      a global mutex and POSIX `rand_r()` uses a weak 31-bit LCG. Blackman & Vigna's
 *      `xoshiro128++` holds 128 bits of state (`4 x uint32_t`), passes TestU01 BigCrush, has
 *      period `2^128 - 1`, and executes in a few register bit-ops. Padding `cacaosim_rng_t` to
 *      exactly 64 bytes (`16 + 4 + 4 + 40 = 64`) ensures that when `cacaosim_det_pool_t` allocates
 *      an aligned array of per-thread states, every OpenMP worker owns an exclusive L1 cache line
 *      with zero MESI cache-coherency invalidations.
 *   2. Singularity-free `(0, 1)` uniform floats: Extracting 24 mantissa bits (`u >> 8`) and
 *      evaluating `(u + 0.5f) * 2^-24` maps strictly inside `(2^-25, 1 - 2^-25)`, guaranteeing
 *      `logf()` never encounters `0.0f` (`-inf`) without runtime clamping branches.
 *   3. Transcendental-light Marsaglia polar normal sampler: Rejection sampling inside the unit
 *      disk (`pi/4 ~ 78.5%` acceptance) avoids expensive CPU `sinf`/`cosf` evaluations and yields
 *      two independent `N(0, 1)` variates per `logf`/`sqrtf` pair, caching the second variate in
 *      `spare_gauss` to halve amortized transcendental cost per pixel.
 *   4. Hybrid exact Knuth / skewness-corrected Cornish-Fisher Poisson (`mu = 24` crossover):
 *      At low photon counts (`mu < 24`), discrete integer quantization and skewness dominate WFS
 *      subaperture noise, and Knuth's exact multiplicative inter-arrival loop terminates in small
 *      `O(mu)` steps. Above `mu = 24` (e.g., bright PSF cores with `10^4..10^6` photons), `O(mu)`
 *      loops would stall kHz cameras; switching to a 3rd-cumulant Cornish-Fisher expansion
 *      `floor(mu + sqrt(mu)*z + (z^2 - 1)/6 + 0.5)` runs in `O(1)` constant time while preserving
 *      Poisson mean `mu`, variance `mu`, and `mu^(-1/2)` skewness smoothly across the crossover.
 */

#ifndef CACAOSIM_RNG_H
#define CACAOSIM_RNG_H

#include <math.h>
#include <stdint.h>

#include "cacaosim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * struct cacaosim_rng_t - Cache-line-padded (64-byte) per-thread `xoshiro128++` PRNG state
 * @s:           128-bit (`4 x 32-bit`) non-zero linear engine state words `[16 bytes]`.
 * @spare_gauss: Cached second `N(0, 1)` variate produced by the previous Marsaglia polar pair
 *               `[4 bytes]`.
 * @has_spare:   Boolean flag (`1` = `@spare_gauss` holds an unconsumed sample, `0` = empty)
 *               `[4 bytes]`.
 * @pad:         Explicit 40-byte padding (`10 x uint32_t`) rounding `sizeof(cacaosim_rng_t)` to
 *               `CACAOSIM_ALIGN_BYTES` (`64` bytes) to prevent false sharing in OpenMP thread
 *               arrays.
 */
typedef struct
{
    uint32_t s[4];
    float    spare_gauss;
    int32_t  has_spare;
    uint32_t pad[10];
} cacaosim_rng_t;

/**
 * cacaosim_rng_seed() - Deterministically initialize a `xoshiro128++` state from seed & stream ID
 * @rng:       Output PRNG state structure to initialize (non-NULL)
 * @seed:      Base 64-bit simulation seed
 * @stream_id: Zero-based OpenMP thread or independent stream index
 *
 * Role:
 *   Cold-path initializer called once per worker thread during `cacaosim_det_pool_init()` (or
 *   fallback local seeding) to establish decorrelated random sequences across parallel threads.
 *
 * Rationale:
 *   Scrambles `(seed, stream_id)` through `SplitMix64` (in `cacaosim_rng.c`) so adjacent thread
 *   IDs (`0, 1, 2, ...`) or simple integer seeds (`0, 1, ...`) start at uncorrelated points in the
 *   `2^128 - 1` state space, clears any stale `spare_gauss`, and guards against the all-zero
 *   absorbing state.
 */
void cacaosim_rng_seed(
    cacaosim_rng_t *rng,
    uint64_t        seed,
    uint64_t        stream_id);

/**
 * cacaosim_rng_u32() - Advance `xoshiro128++` state and draw the next 32-bit unsigned integer
 * @rng: Active per-thread PRNG state (non-NULL)
 *
 * Role:
 *   Core hot-path bit generator underlying all uniform, Gaussian, and Poisson CPU noise draws.
 *
 * Rationale:
 *   Implements Blackman & Vigna's `xoshiro128++` (`rotl(s0 + s3, 7) + s0` scrambler with shift
 *   `A = 9` and rotate `B = 11`). Marked `static inline` so the compiler keeps `s[0..3]` in scalar
 *   CPU registers across an entire pixel loop without function-call overhead.
 *
 * Return: Uniformly distributed `uint32_t` over `[0, 2^32 - 1]`.
 */
static inline uint32_t cacaosim_rng_u32(cacaosim_rng_t *rng)
{
    /* `++` scrambler: rotl(s0 + s3, 7) + s0 eliminates low-bit linear artifacts */
    const uint32_t s0  = rng->s[0];
    uint32_t       s1  = rng->s[1];
    const uint32_t s3  = rng->s[3];
    const uint32_t sum = s0 + s3;
    const uint32_t rot = (sum << 7) | (sum >> 25);
    const uint32_t res = rot + s0;
    const uint32_t t   = s1 << 9;

    /* 128-bit xoroshiro-family linear state transition */
    rng->s[2] ^= s0;
    rng->s[3] ^= s1;
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    s1        = rng->s[3];
    rng->s[3] = (s1 << 11) | (s1 >> 21);

    return res;
}

/**
 * cacaosim_rng_uniform_f32() - Draw a uniform single-precision float strictly inside `(0, 1)`
 * @rng: Active per-thread PRNG state (non-NULL)
 *
 * Role:
 *   Supplies uniform variates to the Marsaglia polar Gaussian and Knuth Poisson samplers.
 *
 * Rationale:
 *   Discarding the low 8 bits (`>> 8`) leaves 24 uniform bits that fit losslessly in an IEEE-754
 *   binary32 significand (`[0, 16777215]`). Offsetting by `+0.5f` centers each bin inside
 *   `[0, 1]` and guarantees the result is strictly bounded in `[2^-25, 1 - 2^-25]`, preventing
 *   `0.0f` or `1.0f` boundary singularities in logarithmic transforms.
 *
 * Return: Uniform `float` strictly on the open interval `(0, 1)`.
 */
static inline float cacaosim_rng_uniform_f32(cacaosim_rng_t *rng)
{
    /* Keep upper 24 bits and shift by half-ULP into open interval (0, 1) */
    uint32_t u = cacaosim_rng_u32(rng) >> 8;
    return ((float) u + 0.5f) * (1.0f / 16777216.0f);
}

/**
 * cacaosim_rng_gauss_f32() - Sample a standard normal variate `N(0, 1)` via Marsaglia polar method
 * @rng: Active per-thread PRNG state (non-NULL)
 *
 * Role:
 *   Hot-path Gaussian sampler modeling detector electronics readout noise (`ron_e * z`) and
 *   driving the high-flux Cornish-Fisher Poisson approximation.
 *
 * Rationale:
 *   Unlike Box-Muller (`sqrt(-2 ln u1) cos(2 pi u2)`), which evaluates expensive CPU `sinf`/`cosf`
 *   functions, Marsaglia's polar method draws `(u1, u2)` in `(-1, 1)^2` until `s = u1^2 + u2^2`
 *   falls inside `(1e-12, 1)`, then scales both axes by `sqrt(-2 ln(s) / s)`. Caching `u2 * mul`
 *   in `rng->spare_gauss` returns every second sample with a single branch and zero math library
 *   calls. The `s < 1e-12f` guard prevents `logf` / division overflow near the origin.
 *
 * Return: Standard normal `float` sample with zero mean and unit variance.
 */
static inline float cacaosim_rng_gauss_f32(cacaosim_rng_t *rng)
{
    /* Fast path: return the second orthogonal variate from the previous polar draw */
    if (rng->has_spare)
    {
        rng->has_spare = 0;
        return rng->spare_gauss;
    }

    /* Rejection-sample uniform point inside punctured unit disk: 1e-12 <= u1^2 + u2^2 < 1 */
    float u1;
    float u2;
    float s;
    do
    {
        u1 = 2.0f * cacaosim_rng_uniform_f32(rng) - 1.0f;
        u2 = 2.0f * cacaosim_rng_uniform_f32(rng) - 1.0f;
        s  = u1 * u1 + u2 * u2;
    } while (s >= 1.0f || s < 1e-12f);

    /* Radial Box-Muller projection yielding two independent N(0, 1) variates */
    float mul        = sqrtf(-2.0f * logf(s) / s);
    rng->spare_gauss = u2 * mul;
    rng->has_spare   = 1;
    return u1 * mul;
}

/**
 * cacaosim_rng_poisson_f32() - Sample quantum `Poisson(mu)` count as a single-precision float
 * @rng: Active per-thread PRNG state (non-NULL)
 * @mu:  Expected mean photoelectron count `[e-]` (`<= 0` deterministically returns `0.0f`)
 *
 * Role:
 *   Hot-path photon/dark-current shot-noise generator invoked per illuminated pixel in CPU
 *   detector readout (`cacaosim_detector_readout_cpu`) when `photon_noise != 0`.
 *
 * Rationale:
 *   Combines two regimes chosen for numerical fidelity and real-time kHz bound:
 *     - Low-flux regime (`0 < mu < 24`): Knuth's exact exponential inter-arrival product
 *       (`prod_{i=1}^k U_i <= exp(-mu)`) reproduces exact discrete `0, 1, 2, ...` photon-counting
 *       statistics critical for photon-starved WFS subapertures and coronagraphic dark holes,
 *       requiring on average `mu + 1 <= 25` fast register-resident uniform draws.
 *     - High-flux regime (`mu >= 24`): Evaluates the 1st-order Cornish-Fisher skewness-corrected
 *       normal transformation `floor(mu + sqrt(mu)*z + (z^2 - 1)/6 + 0.5)` clamped at `0.0f`.
 *       Adding `(z^2 - 1)/6` matches the exact third cumulant (`kappa_3 = mu`) of the Poisson
 *       distribution, eliminating the systematic median/tail bias of plain Gaussian approximations
 *       while guaranteeing strictly `O(1)` execution even at `mu > 10^7` photons/pixel.
 *
 * Return: Non-negative integer photoelectron realization represented as `float`.
 */
static inline float cacaosim_rng_poisson_f32(
    cacaosim_rng_t *rng,
    float           mu)
{
    if (mu <= 0.0f)
    {
        return 0.0f;
    }
    if (mu < 24.0f)
    {
        /* Exact Knuth multiplicative Poisson algorithm for low-photon-count regime */
        float    L = expf(-mu);
        float    p = 1.0f;
        uint32_t k = 0;
        do
        {
            k++;
            p *= cacaosim_rng_uniform_f32(rng);
        } while (p > L);
        return (float) (k - 1u);
    }

    /* O(1) skewness-corrected Cornish-Fisher normal mapping + 0.5 continuity correction */
    float z   = cacaosim_rng_gauss_f32(rng);
    float val = mu + sqrtf(mu) * z + (1.0f / 6.0f) * (z * z - 1.0f) + 0.5f;
    if (val < 0.0f)
    {
        return 0.0f;
    }
    return floorf(val);
}

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_RNG_H */
