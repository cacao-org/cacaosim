// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_time.h
 * @brief   Drift-free monotonic high-precision timer helpers for HT hardware pacing
 */

#ifndef CACAOSIM_TIME_H
#define CACAOSIM_TIME_H

#include <errno.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * cacaosim_now_ns() - Read current monotonic clock in nanoseconds
 *
 * Return: Monotonic timestamp in nanoseconds.
 */
static inline uint64_t cacaosim_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

/**
 * cacaosim_timespec_add_ns() - Advance a timespec deadline by delta_ns without drift
 * @ts:       In/out absolute deadline timespec
 * @delta_ns: Step interval in nanoseconds
 */
static inline void cacaosim_timespec_add_ns(
    struct timespec *ts,
    uint64_t         delta_ns)
{
    uint64_t nsec = (uint64_t) ts->tv_nsec + delta_ns;
    ts->tv_sec += (time_t) (nsec / 1000000000ULL);
    ts->tv_nsec = (long) (nsec % 1000000000ULL);
}

/**
 * cacaosim_sleep_until() - Sleep until absolute CLOCK_MONOTONIC deadline
 * @deadline: Target monotonic timespec; if already in the past, resyncs to now
 */
static inline void cacaosim_sleep_until(struct timespec *deadline)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (now.tv_sec > deadline->tv_sec ||
        (now.tv_sec == deadline->tv_sec && now.tv_nsec >= deadline->tv_nsec))
    {
        *deadline = now;
        return;
    }

    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, deadline, NULL) == EINTR)
    {
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_TIME_H */
