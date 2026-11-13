// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_time.h
 * @brief   Drift-free monotonic nanosecond timer utilities for real-time AO hardware pacing
 *
 * Architectural Role:
 *   Provides deterministic frame-rate pacing (`framerate_hz` / `dt_sec`) and high-resolution
 *   elapsed-time telemetry to all real-time `cacaosim` FPS hardware loops (`dmsim`, `mkbeam`,
 *   `wfe`, `beamsplit`, `coro`, `pyrwfs`, `shwfs`, `zwfs`, `fpcam`, and fused `htsim`) as well as
 *   wall-clock throughput measurement in `cacaosim-benchmark`.
 *
 * Design Rationale & Motivation:
 *   1. `CLOCK_MONOTONIC` immunity to wall-clock steps: Closed-loop AO simulations frequently run
 *      on observatory servers alongside NTP/chrony or PTP daemons. Unlike `CLOCK_REALTIME`,
 *      `CLOCK_MONOTONIC` is guaranteed never to jump backward or slew discontinuously when system
 *      wall time is adjusted.
 *   2. Absolute-deadline pacing (`TIMER_ABSTIME`) eliminates cumulative loop-rate drift: Naive
 *      relative sleeps (`usleep(dt)` or `nanosleep(&rel)`) add frame compute time plus OS wakeup
 *      slack onto every period ($T_{\text{actual}} = T_{\text{target}} + t_{\text{compute}} +
 *      t_{\text{jitter}}$), causing a 2 kHz (`500 us`) WFS loop taking `80 us` of compute to sag
 *      to ~1.72 kHz. Advancing an exact integer-nanosecond `struct timespec` target via
 *      `cacaosim_timespec_add_ns()` and sleeping until `TIMER_ABSTIME` absorbs compute time and
 *      scheduler jitter inside each period, locking long-term frame rate to machine-crystal
 *      accuracy.
 *   3. Overrun clamping prevents catch-up bursts: If a frame exceeds its budget (e.g., first-frame
 *      lazy FFT/GPU warmup or transient OS preemption), `cacaosim_sleep_until()` resynchronizes
 *      `*deadline = now` rather than leaving `deadline` in the past, preventing an unpaced burst
 *      of back-to-back frames that would corrupt simulated discrete-time DM ZOH physics and flood
 *      downstream `cacao` real-time control queues.
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
 * cacaosim_now_ns() - Read the current `CLOCK_MONOTONIC` timestamp as a 64-bit nanosecond count
 *
 * Role:
 *   Used by benchmark harnesses, FPS telemetry updaters, and rate monitors to measure exact
 *   sub-microsecond elapsed intervals via simple unsigned 64-bit integer subtraction.
 *
 * Rationale:
 *   Flattening `struct timespec` into a single `uint64_t` nanosecond counter simplifies interval
 *   math (`t1 - t0`) without borrow handling across `tv_sec`/`tv_nsec` fields, and `2^64` ns spans
 *   ~584.9 years of uptime without wrap-around.
 *
 * Return: Monotonic timestamp in nanoseconds since an unspecified boot epoch.
 */
static inline uint64_t cacaosim_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

/**
 * cacaosim_timespec_add_ns() - Advance an absolute `timespec` deadline by `delta_ns` without drift
 * @ts:       In/out normalized `CLOCK_MONOTONIC` deadline (`0 <= tv_nsec < 10^9`), updated in place
 * @delta_ns: Step interval to add `[nanoseconds]` (may exceed 1 second)
 *
 * Role:
 *   Called once per real-time simulation tick prior to `cacaosim_sleep_until()` to compute the
 *   next frame's exact wakeup deadline.
 *
 * Rationale:
 *   Performing deadline arithmetic in integer nanoseconds rather than floating-point seconds
 *   avoids IEEE-754 mantissa rounding accumulation over tens of millions of kHz AO loop frames,
 *   while explicit `/ 10^9` and `% 10^9` normalization keeps `ts->tv_nsec` strictly inside the
 *   POSIX `[0, 999999999]` range required by `clock_nanosleep()`.
 */
static inline void cacaosim_timespec_add_ns(
    struct timespec *ts,
    uint64_t         delta_ns)
{
    /* Sum nanoseconds in 64-bit unsigned arithmetic and carry whole seconds into tv_sec */
    uint64_t nsec = (uint64_t) ts->tv_nsec + delta_ns;
    ts->tv_sec += (time_t) (nsec / 1000000000ULL);
    ts->tv_nsec = (long) (nsec % 1000000000ULL);
}

/**
 * cacaosim_sleep_until() - Block calling thread until an absolute `CLOCK_MONOTONIC` deadline
 * @deadline: In/out target monotonic `timespec`; if already elapsed (`now >= *deadline`),
 *            overwritten with `now` and returns immediately without sleeping
 *
 * Role:
 *   Hot-loop frame pacer invoked at the end of every timed hardware simulation step to enforce
 *   the configured sensor or mirror update rate (`1 / dt_sec`).
 *
 * Rationale:
 *   Checking `now >= *deadline` upfront handles compute overruns gracefully by re-anchoring the
 *   schedule to the current instant instead of bursting catch-up frames. When sleeping, passing
 *   `TIMER_ABSTIME` to `clock_nanosleep()` allows transparent `EINTR` signal-interruption retries
 *   in a tight `while` loop without recomputing remaining relative sleep duration or introducing
 *   restart drift.
 */
static inline void cacaosim_sleep_until(struct timespec *deadline)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    /* Overrun guard: if compute exceeded the frame budget, resync schedule to `now` (no burst) */
    if (now.tv_sec > deadline->tv_sec ||
        (now.tv_sec == deadline->tv_sec && now.tv_nsec >= deadline->tv_nsec))
    {
        *deadline = now;
        return;
    }

    /* Sleep until exact absolute deadline; TIMER_ABSTIME makes EINTR retries drift-free */
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, deadline, NULL) == EINTR)
    {
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_TIME_H */
