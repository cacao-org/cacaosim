// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_rng.c
 * @brief   `SplitMix64` seed diffuser for lock-free per-thread `xoshiro128++` PRNG initialization
 *
 * Architectural Role:
 *   Implements the cold-path state-seeding routine (`cacaosim_rng_seed`) declared in
 *   `cacaosim_rng.h`, expanding a `(seed, stream_id)` tuple into four statistically independent,
 *   non-zero 32-bit state words for each OpenMP worker thread in `cacaosim_det_pool_t`.
 *
 * Design Rationale & Motivation:
 *   `F_2`-linear generators such as `xoshiro128++` have high throughput and long period, but if
 *   two threads are seeded directly with low-Hamming-weight or adjacent integers (e.g., `seed = 1`,
 *   `stream_id = 0, 1, 2`), their initial bit states share sparse structure that can leak subtle
 *   spatial stripes across OpenMP static pixel chunks during early camera frames. Furthermore, the
 *   all-zero 128-bit state is a fixed point that outputs zeros forever. Hashing `(seed, stream_id)`
 *   through Stafford's Mix13 `SplitMix64` bijection (the canonical initializer recommended by
 *   Blackman & Vigna) achieves full 64-bit avalanche diffusion before populating `s[0..3]`, and an
 *   explicit bitwise-OR check guarantees the state can never land on zero.
 */

#include "common/cacaosim_rng.h"

/**
 * splitmix64_next() - Advance a 64-bit Weyl state and apply Stafford Mix13 bit-avalanche mixing
 * @state: In/out mutable 64-bit Weyl-sequence accumulator (non-NULL)
 *
 * Role:
 *   Internal helper called twice by `cacaosim_rng_seed()` to synthesize 128 bits of high-entropy
 *   initial state from a single 64-bit scrambled seed.
 *
 * Rationale:
 *   Adds the odd golden-ratio Weyl increment `0x9e3779b97f4a7c15ULL` (`floor(2^64 / phi)`) and
 *   applies David Stafford's Mix13 xorshift-multiply cascade, which flips every output bit with
 *   probability ~0.5 for any single-bit change in `@state` and passes BigCrush on its own.
 *
 * Return: Well-diffused 64-bit unsigned integer.
 */
static uint64_t splitmix64_next(uint64_t *state)
{
    /* Golden-ratio Weyl step followed by two Mix13 xorshift-multiply rounds */
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * cacaosim_rng_seed() - Initialize a `xoshiro128++` PRNG state from a base seed and stream index
 * @rng:       Output per-thread PRNG state to populate (non-NULL)
 * @seed:      Base 64-bit user or fallback simulation seed
 * @stream_id: Thread or logical stream index (`0 .. nb_threads - 1`) for independent sequences
 *
 * Role:
 *   Cold-path setup entry point invoked per OpenMP worker slot during detector pool creation so
 *   parallel CPU readout threads generate uncorrelated noise without synchronization.
 *
 * Rationale:
 *   Pre-mixes `@stream_id` using an odd 64-bit Pelle Evensen / Murmur-style multiplier plus the
 *   golden-ratio constant before XORing with `@seed`, ensuring `(seed = 0, stream_id = 0)` still
 *   enters `splitmix64_next()` with a non-trivial state and preventing collisions between shifted
 *   `(seed, stream_id)` pairs. Also resets the Marsaglia polar Gaussian cache (`has_spare = 0`) so
 *   re-seeding produces strictly reproducible trajectories.
 */
void cacaosim_rng_seed(
    cacaosim_rng_t *rng,
    uint64_t        seed,
    uint64_t        stream_id)
{
    /* Inject stream_id via odd multiplicative + additive Weyl offset prior to SplitMix64 */
    uint64_t sm = seed ^ (stream_id * 0xd1b54a32d192ed03ULL + 0x9e3779b97f4a7c15ULL);
    uint64_t w0 = splitmix64_next(&sm);
    uint64_t w1 = splitmix64_next(&sm);

    /* Unpack two 64-bit SplitMix64 words into the four 32-bit xoshiro128++ state registers */
    rng->s[0] = (uint32_t) (w0 & 0xffffffffu);
    rng->s[1] = (uint32_t) (w0 >> 32);
    rng->s[2] = (uint32_t) (w1 & 0xffffffffu);
    rng->s[3] = (uint32_t) (w1 >> 32);

    /* Escape the degenerate 2^-128 all-zero fixed point of F_2-linear state transitions */
    if ((rng->s[0] | rng->s[1] | rng->s[2] | rng->s[3]) == 0u)
    {
        rng->s[0] = 1u;
    }

    /* Clear cached second Gaussian variate so re-seeding is 100% deterministic */
    rng->spare_gauss = 0.0f;
    rng->has_spare   = 0;
}
