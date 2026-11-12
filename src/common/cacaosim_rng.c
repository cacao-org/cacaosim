// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_rng.c
 * @brief   Seed mixing (SplitMix64) for lock-free xoshiro128++ PRNG states
 */

#include "common/cacaosim_rng.h"

/**
 * splitmix64_next() - Scramble a 64-bit state to initialize xoshiro128++ words
 * @state: Pointer to mutable 64-bit seed state
 *
 * Return: Mixed 64-bit unsigned integer.
 */
static uint64_t splitmix64_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void cacaosim_rng_seed(
    cacaosim_rng_t *rng,
    uint64_t        seed,
    uint64_t        stream_id)
{
    uint64_t sm = seed ^ (stream_id * 0xd1b54a32d192ed03ULL + 0x9e3779b97f4a7c15ULL);
    uint64_t w0 = splitmix64_next(&sm);
    uint64_t w1 = splitmix64_next(&sm);

    rng->s[0] = (uint32_t) (w0 & 0xffffffffu);
    rng->s[1] = (uint32_t) (w0 >> 32);
    rng->s[2] = (uint32_t) (w1 & 0xffffffffu);
    rng->s[3] = (uint32_t) (w1 >> 32);

    if ((rng->s[0] | rng->s[1] | rng->s[2] | rng->s[3]) == 0u)
    {
        rng->s[0] = 1u;
    }

    rng->spare_gauss = 0.0f;
    rng->has_spare   = 0;
}
