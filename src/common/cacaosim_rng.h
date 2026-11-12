// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_rng.h
 * @brief   Lock-free per-thread xoshiro128++ PRNG for Poisson and Gaussian detector noise
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
 * @brief 128-bit state for xoshiro128++ generator (cache-line padded for OpenMP arrays).
 */
typedef struct
{
    uint32_t s[4];
    float    spare_gauss;
    int32_t  has_spare;
    uint32_t pad[10];
} cacaosim_rng_t;

/**
 * cacaosim_rng_seed() - Initialize a xoshiro128++ state from a 64-bit seed and stream index
 * @rng:       Output PRNG state
 * @seed:      Base 64-bit seed
 * @stream_id: Thread or stream index for independent sequences
 */
void cacaosim_rng_seed(
    cacaosim_rng_t *rng,
    uint64_t        seed,
    uint64_t        stream_id);

/**
 * cacaosim_rng_u32() - Next 32-bit unsigned random integer (xoshiro128++)
 * @rng: Active PRNG state
 *
 * Return: Uniformly distributed uint32_t.
 */
static inline uint32_t cacaosim_rng_u32(cacaosim_rng_t *rng)
{
    const uint32_t s0  = rng->s[0];
    uint32_t       s1  = rng->s[1];
    const uint32_t s3  = rng->s[3];
    const uint32_t sum = s0 + s3;
    const uint32_t rot = (sum << 7) | (sum >> 25);
    const uint32_t res = rot + s0;
    const uint32_t t   = s1 << 9;

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
 * cacaosim_rng_uniform_f32() - Uniform float in (0, 1)
 * @rng: Active PRNG state
 *
 * Return: Single-precision float strictly inside (0, 1).
 */
static inline float cacaosim_rng_uniform_f32(cacaosim_rng_t *rng)
{
    uint32_t u = cacaosim_rng_u32(rng) >> 8;
    return ((float) u + 0.5f) * (1.0f / 16777216.0f);
}

/**
 * cacaosim_rng_gauss_f32() - Standard normal variate N(0, 1) via Marsaglia polar method
 * @rng: Active PRNG state
 *
 * Return: Standard normal float32 sample.
 */
static inline float cacaosim_rng_gauss_f32(cacaosim_rng_t *rng)
{
    if (rng->has_spare)
    {
        rng->has_spare = 0;
        return rng->spare_gauss;
    }

    float u1;
    float u2;
    float s;
    do
    {
        u1 = 2.0f * cacaosim_rng_uniform_f32(rng) - 1.0f;
        u2 = 2.0f * cacaosim_rng_uniform_f32(rng) - 1.0f;
        s  = u1 * u1 + u2 * u2;
    } while (s >= 1.0f || s < 1e-12f);

    float mul        = sqrtf(-2.0f * logf(s) / s);
    rng->spare_gauss = u2 * mul;
    rng->has_spare   = 1;
    return u1 * mul;
}

/**
 * cacaosim_rng_poisson_f32() - Sample Poisson(mu) returning float photoelectron count
 * @rng: Active PRNG state
 * @mu:  Expected mean count [photoelectrons]
 *
 * Uses exact Knuth multiplicative sampling for mu < 24 and third-moment skewness-corrected
 * Cornish-Fisher normal mapping for mu >= 24.
 *
 * Return: Non-negative integer count cast to float.
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
