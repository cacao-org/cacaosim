// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_benchmark.c
 * @brief   Latency (mean / p50 / p99 us) and throughput (kHz) benchmark suite for cacaosim
 *
 * Architectural Overview & Benchmarking Philosophy
 * ------------------------------------------------
 * Adaptive Optics real-time control (RTC) loops operate at kilohertz frame rates (1..4 kHz) under
 * strict per-frame jitter budgets: a single tail-latency spike caused by dynamic heap allocation,
 * lazy FFT plan creation, cache thrashing, or uncaptured GPU kernel launch overhead stalls the WFS
 * telemetry stream and injects servo lag into the closed loop.
 *
 * Consequently, reporting mean frame rate alone is insufficient for qualifying AO simulation
 * kernels. This benchmark suite characterizes both standalone optical/hardware primitives and
 * complete multi-rate optical bench DAGs across CPU (OpenMP SIMD + FFTW3f + CBLAS) and GPU (CUDA +
 * cuFFT + cuBLAS + CUDA Graphs) backends using monotonic nanosecond timing (cacaosim_now_ns()):
 *
 *   1. Untimed Warm-Up Protocol: Every benchmark runs 20..25 untimed warm-up steps before sampling
 *      to fault in aligned pages, spin up OpenMP worker threads, populate instruction/data caches,
 *      and trigger first-step CUDA Graph stream capture/instantiation so timed samples reflect true
 *      steady-state closed-loop execution.
 *   2. Order-Statistic Telemetry: Reports arithmetic Mean [us], Median p50 [us], 99th-percentile
 *      tail latency p99 [us], and sustained throughput [kHz = 1000 / mean_us] side by side.
 *   3. End-to-End Parity: Measures individual components (DM, PyWFS, SHWFS, ZWFS, Coronagraph,
 *      FPcam), zero-IPC fused single-stage pipelines (DM + PyWFS + decimated FPcam sharing one
 *      CUDA stream and device OPD pointer), and declarative multi-stage .bench topologies loaded
 *      from CACAOSIM_SRC_DIR (falling back to ".").
 *   4. Graceful Hardware Fallback: When compiled without CUDA or executed on a host lacking a
 *      CUDA-capable GPU, CUDA benchmark rows cleanly self-skip without failing the suite.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "milkcacaosim.h"

/**
 * @brief Ascending qsort() comparator for double-precision latency arrays.
 *
 * Role:
 *   Orders per-iteration latency measurements [us] from smallest to largest so non-parametric
 *   empirical quantiles (median p50 and tail p99) can be indexed directly.
 *
 * Rationale:
 *   Uses the branchless three-way idiom `(da > db) - (da < db)` rather than casting `da - db` to
 *   `int`, preventing truncation-to-zero bugs when comparing sub-microsecond latency differences
 *   (`|da - db| < 1.0 us`).
 *
 * @param[in] a  Pointer to first double sample.
 * @param[in] b  Pointer to second double sample.
 * @return -1 if `*a < *b`, 0 if `*a == *b`, or +1 if `*a > *b`.
 */
static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *) a;
    double db = *(const double *) b;
    return (da > db) - (da < db);
}

/**
 * @brief Compute and print mean, p50, p99 latency [us] and throughput [kHz] for a benchmark run.
 *
 * Role:
 *   Sorts `lat_us` in-place, evaluates the sample mean, median (`n / 2`), 99th percentile
 *   (`0.99 * (n - 1)`), and sustained frame rate (`1000 / mean`), and formats one table row.
 *
 * Rationale:
 *   Pairing `p50` and `p99` alongside `mean` immediately exposes OS scheduling jitter, OpenMP
 *   barrier skew, or GPU synchronization stalls that would violate deterministic `TIMER_ABSTIME`
 *   pacing in real-time `cacao` hardware-in-the-loop operation.
 *
 * @param[in]     label    Human-readable pipeline configuration description.
 * @param[in]     backend  Active execution backend tag (`"CPU"` or `"CUDA"`).
 * @param[in,out] lat_us   Array of `n` per-frame wall-clock latencies in microseconds (sorted
 *                         in-place by this function).
 * @param[in]     n        Number of timed iterations in `lat_us`.
 */
static void print_stats(const char *label, const char *backend, double *lat_us, int n)
{
    qsort(lat_us, (size_t) n, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += lat_us[i];
    }
    double mean = sum / (double) n;
    double p50  = lat_us[n / 2];
    double p99  = lat_us[(int) ((double) (n - 1) * 0.99)];
    double khz  = 1000.0 / mean;
    printf("  %-38s | %-4s | %8.2f | %8.2f | %8.2f | %8.2f kHz\n", label, backend, mean, p50,
           p99, khz);
}

/**
 * @brief Benchmark deformable mirror temporal dynamics and gather-stencil 2D OPD rendering.
 *
 * Role:
 *   Measures full-step latency of `cacaosim_dm_step()`—including fractional command ring-buffer
 *   lookup (`delay_frames = 1.0`), 16-bit DAC quantization, stroke saturation, exact 1st-order ZOH
 *   actuator state update (`tau = 120 us`, `dt = 500 us`), and compact influence-function surface
 *   evaluation (plus D2H transfer into host `opd` when running on CUDA).
 *
 * Rationale:
 *   Evaluated at both `20x20 -> 128x128` (classical 8 m AO / Woofer scale) and `50x50 -> 256x256`
 *   (2,500-actuator ExAO Tweeter scale) to verify that compact gather-stencil evaluation scales as
 *   `O(N_wf^2)` independent of total actuator count `N_act^2`, sustaining multi-kHz rates. Twenty-
 *   five untimed warm-up iterations prime the command delay ring and thread/GPU state.
 *
 * @param[in] nact     Linear actuator grid dimension (`nact x nact` actuators).
 * @param[in] nwf      Wavefront OPD grid dimension (`nwf x nwf` pixels).
 * @param[in] backend  Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter    Number of timed iterations to record after warm-up.
 */
static void bench_dm(uint32_t nact, uint32_t nwf, int32_t backend, int niter)
{
    cacaosim_dm_cfg_t cfg = {
        .nact_x = nact, .nact_y = nact, .nwf = nwf, .dm_diam = 0.9375f, .if_coupling = 0.20f,
        .refl_scale = 2.0f, .max_stroke = 5.0f, .dac_bits = 16, .nonlin_quad = 0.0f,
        .dyn_mode = 1, .dt_sec = 0.0005f, .tau_sec = 0.00012f, .f0_hz = 2500.0f, .zeta = 0.7f,
        .delay_frames = 1.0f, .backend = backend
    };
    cacaosim_dm_ctx_t ctx;
    if (cacaosim_dm_init(&ctx, &cfg, NULL) != 0)
    {
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_dm_free(&ctx);
        return;
    }

    float  *cmd = (float *) cacaosim_aligned_calloc((size_t) nact * nact, sizeof(float));
    float  *opd = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (uint32_t i = 0; i < nact * nact; i++)
    {
        cmd[i] = 0.2f;
    }

    /* Warm up OpenMP threads / GPU context and fill the 1-frame command delay ring */
    for (int w = 0; w < 25; w++)
    {
        cacaosim_dm_step(&ctx, cmd, NULL, 1.0f, 0.00012f, 5.0f, opd);
    }
    for (int k = 0; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_dm_step(&ctx, cmd, NULL, 1.0f, 0.00012f, 5.0f, opd);
        lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
    }

    char label[64];
    snprintf(label, sizeof(label), "DM %ux%u -> %ux%u OPD", nact, nact, nwf, nwf);
    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_dm_free(&ctx);
    free(lat);
    free(opd);
    free(cmd);
}

/**
 * @brief Benchmark 4-sided Pyramid WFS diffractive propagation and noisy EMCCD readout.
 *
 * Role:
 *   Measures per-frame latency of `cacaosim_pyrwfs_step()` across unmodulated (`mod_r = 0`) and
 *   ring-modulated (`mod_r = 3 lambda/D`, `mod_n = 16` tip-tilt points) configurations, including
 *   `2x`-padded FFT propagation, 4-quadrant intensity accumulation, Poisson photon shot noise,
 *   sub-electron Gaussian readout noise (`0.5 e-`), and `uint16` quantization.
 *
 * Rationale:
 *   Modulated Pyramid WFS dominates single-sensor compute cost in ExAO simulation because our
 *   precomputed shifted-focal-mask reduction executes `1` forward pupil FFT plus `N_mod` inverse
 *   FFTs (`17` 2D FFTs of size `(2 * nwf)^2` per frame when `mod_n = 16`). Running 20 warm-up
 *   frames ensures one-time CUDA Graph capture completes before timing begins, isolating steady-
 *   state `cudaGraphLaunch` replay performance against multi-threaded `FFTW3f` on CPU.
 *
 * @param[in] nwf      Input wavefront grid size (`nwf x nwf` pixels).
 * @param[in] ncam     Output WFS detector size (`ncam x ncam` pixels).
 * @param[in] mod_r    Circular modulation radius in units of `lambda / D` (`0.0` = unmodulated).
 * @param[in] mod_n    Number of discrete modulation points around the ring when `mod_r > 0`.
 * @param[in] backend  Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter    Number of timed iterations to record after warm-up.
 */
static void bench_pyrwfs(uint32_t nwf, uint32_t ncam, float mod_r, int32_t mod_n, int32_t backend,
                         int niter)
{
    float *pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *wf  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    uint16_t *out = (uint16_t *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(uint16_t));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.15f, 77ULL, wf);

    cacaosim_pyrwfs_cfg_t cfg = {
        .nwf = nwf, .pad_factor = 2, .ncam = ncam, .pup_diam = 0.9375f, .lambda_um = 0.75f,
        .mod_rad_ld = mod_r, .mod_npts = mod_n, .pup_sep = 0.50f, .roof_ld = 0.0f,
        .fstop_frac = 0.45f, .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 0.75f,
        .rng_seed = 42ULL, .backend = backend,
        .det = { .flux_ph = 100000.0f, .dark_e = 0.0f, .ron_e = 0.5f, .gain_eadu = 1.0f,
                 .bias_adu = 100.0f, .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1,
                 .out_dtype = 0 }
    };

    cacaosim_pyrwfs_ctx_t ctx;
    if (cacaosim_pyrwfs_init(&ctx, &cfg, pup, NULL, NULL) != 0)
    {
        free(out);
        free(wf);
        free(pup);
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_pyrwfs_free(&ctx);
        free(out);
        free(wf);
        free(pup);
        return;
    }

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    /* Warm-up triggers initial CUDA Graph capture/instantiation on GPU and warms FFTW caches */
    for (int w = 0; w < 20; w++)
    {
        cacaosim_pyrwfs_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, mod_r, &cfg.det, out,
                             NULL);
    }
    for (int k = 0; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_pyrwfs_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, mod_r, &cfg.det, out,
                             NULL);
        lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
    }

    char label[64];
    snprintf(label, sizeof(label), "PyWFS %u->%u (nmod=%d)", nwf, ncam,
             (mod_r > 0.0f) ? mod_n : 1);
    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_pyrwfs_free(&ctx);
    free(lat);
    free(out);
    free(wf);
    free(pup);
}

/**
 * @brief Benchmark focal-plane science PSF camera propagation (MFT vs FFT, mono vs broadband).
 *
 * Role:
 *   Measures per-frame execution time of `cacaosim_fpcam_step()` (complex pupil phasor synthesis,
 *   Fraunhofer propagation to focal plane, Strehl ratio telemetry, and Poisson + Gaussian `uint16`
 *   detector readout) across Soummer Matrix-DFT (`prop_mode = 0`) and 2x-padded FFT
 *   (`prop_mode = 1`), for both monochromatic (`nlam = 1`) and 20%-bandwidth polychromatic
 *   (`nlam = 5`) H-band (`1.65 um`) configurations.
 *
 * Rationale:
 *   Directly compares separable two-step complex GEMM (`cblas_cgemm` / strided-batched
 *   `cublasCgemmStridedBatched`) at arbitrary `3.0 px / (lambda/D)` sampling against padded 2D
 *   FFTs, and quantifies polychromatic batching efficiency when synthesizing broadband chromatic
 *   speckle elongation across `nlam = 5` spectral channels.
 *
 * @param[in] nwf        Input pupil wavefront grid size (`nwf x nwf` pixels).
 * @param[in] ncam       Output focal-plane detector size (`ncam x ncam` pixels).
 * @param[in] prop_mode  Optical propagator (`0` = Soummer Matrix-DFT, `1` = padded 2D FFT).
 * @param[in] nlam       Number of discrete spectral channels (`1` = mono, `> 1` = 20% broadband).
 * @param[in] backend    Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter      Number of timed iterations to record after warm-up.
 */
static void bench_fpcam(uint32_t nwf, uint32_t ncam, int32_t prop_mode, int32_t nlam,
                        int32_t backend, int niter)
{
    float    *pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float    *wf  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    uint16_t *out = (uint16_t *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(uint16_t));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.12f, 55ULL, wf);

    cacaosim_fpcam_cfg_t cfg = {
        .nwf = nwf, .ncam = ncam, .prop_mode = prop_mode, .pad_factor = 2, .pup_diam = 0.9375f,
        .lambda_um = 1.65f, .bw_frac = (nlam > 1) ? 0.20f : 0.0f, .nlambda = nlam,
        .pixscale_ld = 3.0f, .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 1.65f,
        .rng_seed = 88ULL, .backend = backend,
        .det = { .flux_ph = 1000000.0f, .dark_e = 0.0f, .ron_e = 1.0f, .gain_eadu = 1.0f,
                 .bias_adu = 100.0f, .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1,
                 .out_dtype = 0 }
    };

    cacaosim_fpcam_ctx_t ctx;
    if (cacaosim_fpcam_init(&ctx, &cfg, pup, NULL, NULL) != 0)
    {
        free(out);
        free(wf);
        free(pup);
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_fpcam_free(&ctx);
        free(out);
        free(wf);
        free(pup);
        return;
    }

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int w = 0; w < 20; w++)
    {
        cacaosim_fpcam_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &cfg.det, out, NULL, NULL);
    }
    for (int k = 0; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_fpcam_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &cfg.det, out, NULL, NULL);
        lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
    }

    char label[64];
    snprintf(label, sizeof(label), "FPcam %u->%u (%s, %d-lam)", nwf, ncam,
             (prop_mode == 0) ? "MFT" : "FFT", nlam);
    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_fpcam_free(&ctx);
    free(lat);
    free(out);
    free(wf);
    free(pup);
}

/**
 * @brief Benchmark the zero-IPC fused single-stage AO pipeline (DM + PyWFS + decimated FPcam).
 *
 * Role:
 *   Measures per-master-tick latency of the core `cacaosim.htsim` fused loop: stepping the DM,
 *   evaluating a 16-point modulated Pyramid WFS at 1:1 rate, accumulating instantaneous H-band
 *   focal-plane subframes every tick, and executing noisy science detector readout every 10th tick
 *   (`fp.subframe_cnt >= 10u`).
 *
 * Rationale:
 *   Demonstrates the latency reduction achieved by sharing `dm.cuda_stream` across all three
 *   engines and passing the device-resident `dm.d_opd_out` pointer directly into `pyrwfs` and
 *   `fpcam` (`opd = NULL` on CUDA), completely eliminating intermediate host-device PCIe copies
 *   and inter-process SHM semaphore round-trips on the critical WFS path.
 *
 * @param[in] nwf      Wavefront grid size (`nwf x nwf` pixels).
 * @param[in] nact     Linear DM actuator count (`nact x nact` actuators).
 * @param[in] nwfs     Pyramid WFS camera size (`nwfs x nwfs` pixels).
 * @param[in] backend  Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter    Number of timed master ticks to record after 20 warm-up ticks.
 */
static void bench_fused_htsim(uint32_t nwf, uint32_t nact, uint32_t nwfs, int32_t backend,
                              int niter)
{
    float    *pup  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float    *wf   = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float    *cmd  = (float *) cacaosim_aligned_calloc((size_t) nact * nact, sizeof(float));
    float    *opd  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    uint16_t *wout = (uint16_t *) cacaosim_aligned_calloc((size_t) nwfs * nwfs, sizeof(uint16_t));
    uint16_t *fout = (uint16_t *) cacaosim_aligned_calloc(128 * 128, sizeof(uint16_t));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.15f, 12ULL, wf);

    cacaosim_dm_cfg_t dcfg = {
        .nact_x = nact, .nact_y = nact, .nwf = nwf, .dm_diam = 0.9375f, .if_coupling = 0.20f,
        .refl_scale = 2.0f, .max_stroke = 5.0f, .dac_bits = 16, .nonlin_quad = 0.0f,
        .dyn_mode = 1, .dt_sec = 0.0005f, .tau_sec = 0.00012f, .f0_hz = 2500.0f, .zeta = 0.7f,
        .delay_frames = 1.0f, .backend = backend
    };
    cacaosim_pyrwfs_cfg_t wcfg = {
        .nwf = nwf, .pad_factor = 2, .ncam = nwfs, .pup_diam = 0.9375f, .lambda_um = 0.75f,
        .mod_rad_ld = 3.0f, .mod_npts = 16, .pup_sep = 0.50f, .roof_ld = 0.0f, .fstop_frac = 0.45f,
        .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 0.75f, .rng_seed = 42ULL,
        .backend = backend,
        .det = { .flux_ph = 100000.0f, .dark_e = 0.0f, .ron_e = 0.5f, .gain_eadu = 1.0f,
                 .bias_adu = 100.0f, .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1,
                 .out_dtype = 0 }
    };
    cacaosim_fpcam_cfg_t fcfg = {
        .nwf = nwf, .ncam = 128, .prop_mode = CACAOSIM_FPCAM_PROP_MFT, .pad_factor = 2,
        .pup_diam = 0.9375f, .lambda_um = 1.65f, .bw_frac = 0.0f, .nlambda = 1, .pixscale_ld = 3.0f,
        .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 1.65f, .rng_seed = 99ULL,
        .backend = backend,
        .det = { .flux_ph = 1000000.0f, .dark_e = 0.0f, .ron_e = 1.0f, .gain_eadu = 1.0f,
                 .bias_adu = 100.0f, .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1,
                 .out_dtype = 0 }
    };

    cacaosim_dm_ctx_t     dm;
    cacaosim_pyrwfs_ctx_t wfs;
    cacaosim_fpcam_ctx_t  fp;
    cacaosim_dm_init(&dm, &dcfg, NULL);
    if (backend == CACAOSIM_BACKEND_CUDA && !dm.use_cuda)
    {
        cacaosim_dm_free(&dm);
        free(fout);
        free(wout);
        free(opd);
        free(cmd);
        free(wf);
        free(pup);
        return;
    }
    /* Bind PyWFS and FPcam onto dm.cuda_stream for zero-sync GPU pipeline chaining */
    cacaosim_pyrwfs_init(&wfs, &wcfg, pup, NULL, dm.cuda_stream);
    cacaosim_fpcam_init(&fp, &fcfg, pup, NULL, dm.cuda_stream);

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int k = -20; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        /* Keep DM OPD strictly in VRAM on CUDA (host opd = NULL) and pass dm.d_opd_out directly */
        cacaosim_dm_step(&dm, cmd, NULL, 1.0f, 0.00012f, 5.0f, dm.use_cuda ? NULL : opd);
        cacaosim_pyrwfs_step(&wfs, pup, NULL, wf, dm.use_cuda ? NULL : opd, NULL,
                             dm.use_cuda ? dm.d_opd_out : NULL, 1.0f, 3.0f, &wcfg.det, wout, NULL);
        cacaosim_fpcam_accum_subframe(&fp, pup, NULL, wf, dm.use_cuda ? NULL : opd, NULL,
                                      dm.use_cuda ? dm.d_opd_out : NULL, 1.0f);
        if (fp.subframe_cnt >= 10u)
        {
            cacaosim_fpcam_readout(&fp, &fcfg.det, fout, NULL, NULL);
        }
        if (k >= 0)
        {
            lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
        }
    }

    char label[64];
    snprintf(label, sizeof(label), "Fused HT (%u^2 DM + %u^2 PyWFS + FP)", nact, nwf);
    print_stats(label, dm.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_fpcam_free(&fp);
    cacaosim_pyrwfs_free(&wfs);
    cacaosim_dm_free(&dm);
    free(lat);
    free(fout);
    free(wout);
    free(opd);
    free(cmd);
    free(wf);
    free(pup);
}

/**
 * @brief Benchmark diffractive Shack-Hartmann lenslet array WFS propagation and detector readout.
 *
 * Role:
 *   Measures per-frame latency of `cacaosim_shwfs_step()` across `n_sub x n_sub` lenslets (`pps`
 *   detector pixels per lenslet, `pad_factor = 2` per-subaperture FFTs) including Poisson photon
 *   shot noise, `0.5 e-` readout noise, and `uint16` detector quantization.
 *
 * Rationale:
 *   Contrasts OpenMP thread-local L1-cache-resident `FFTW3f` subaperture FFTs on CPU against a
 *   single batched `cufftPlanMany` execution across all `n_sub^2` (`256`) subapertures on GPU,
 *   verifying multi-kilohertz suitability for classical and ExAO Shack-Hartmann loops.
 *
 * @param[in] nwf      Input wavefront grid size (`nwf x nwf` pixels).
 * @param[in] n_sub    Linear subaperture count across the pupil (`n_sub x n_sub` lenslets).
 * @param[in] pps      Detector pixels per subaperture (`ncam = n_sub * pps`).
 * @param[in] backend  Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter    Number of timed iterations to record after 20 warm-up steps.
 */
static void bench_shwfs(uint32_t nwf, uint32_t n_sub, uint32_t pps, int32_t backend, int niter)
{
    uint32_t  ncam = n_sub * pps;
    float    *pup  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float    *wf   = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    uint16_t *out  = (uint16_t *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(uint16_t));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.15f, 33ULL, wf);

    cacaosim_shwfs_cfg_t cfg = {
        .nwf = nwf, .n_sub = n_sub, .pix_per_sub = pps, .pad_factor = 2, .pup_diam = 0.9375f,
        .lambda_um = 0.75f, .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 0.75f,
        .rng_seed = 44ULL, .backend = backend,
        .det = { .flux_ph = 100000.0f, .ron_e = 0.5f, .gain_eadu = 1.0f, .bias_adu = 100.0f,
                 .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1, .out_dtype = 0 }
    };

    cacaosim_shwfs_ctx_t ctx;
    if (cacaosim_shwfs_init(&ctx, &cfg, pup, NULL, NULL) != 0)
    {
        free(out);
        free(wf);
        free(pup);
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_shwfs_free(&ctx);
        free(out);
        free(wf);
        free(pup);
        return;
    }

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int w = 0; w < 20; w++)
    {
        cacaosim_shwfs_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &cfg.det, out, NULL);
    }
    for (int k = 0; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_shwfs_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &cfg.det, out, NULL);
        lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
    }

    char label[64];
    snprintf(label, sizeof(label), "SHWFS %u->%u (%ux%u lenslets)", nwf, ncam, n_sub, n_sub);
    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_shwfs_free(&ctx);
    free(lat);
    free(out);
    free(wf);
    free(pup);
}

/**
 * @brief Benchmark Babinet Matrix-DFT Zernike phase-contrast WFS and detector readout.
 *
 * Role:
 *   Measures per-frame execution latency of `cacaosim_zwfs_step()` evaluating a `1.06 lambda/D`,
 *   `pi/2`-radian focal phase dimple via a compact `32x32` Soummer Matrix-DFT (`cblas_cgemm` /
 *   `cublasCgemm`), Babinet pupil-plane interference, and noisy `uint16` readout.
 *
 * Rationale:
 *   Illustrates why Babinet decomposition over the compact `1.06 lambda/D` dimple support achieves
 *   sub-100-microsecond CPU and ~10-kHz GPU latencies—avoiding `8x..16x` zero-padded full-plane
 *   FFTs otherwise required to sample a `1 lambda/D` focal mask without edge aliasing.
 *
 * @param[in] nwf      Input wavefront grid size (`nwf x nwf` pixels).
 * @param[in] ncam     Output pupil-imaging camera size (`ncam x ncam` pixels).
 * @param[in] backend  Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter    Number of timed iterations to record after 20 warm-up steps.
 */
static void bench_zwfs(uint32_t nwf, uint32_t ncam, int32_t backend, int niter)
{
    float    *pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float    *wf  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    uint16_t *out = (uint16_t *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(uint16_t));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.04f, 66ULL, wf);

    cacaosim_zwfs_cfg_t cfg = {
        .nwf = nwf, .ncam = ncam, .n_mft = 32, .pup_diam = 0.9375f, .lambda_um = 1.65f,
        .dimple_ld = 1.06f, .dimple_rad = 0.5f * CACAOSIM_PI_F, .dimple_amp = 1.0f,
        .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 1.65f, .rng_seed = 77ULL,
        .backend = backend,
        .det = { .flux_ph = 100000.0f, .ron_e = 0.5f, .gain_eadu = 1.0f, .bias_adu = 100.0f,
                 .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1, .out_dtype = 0 }
    };

    cacaosim_zwfs_ctx_t ctx;
    if (cacaosim_zwfs_init(&ctx, &cfg, pup, NULL, NULL) != 0)
    {
        free(out);
        free(wf);
        free(pup);
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_zwfs_free(&ctx);
        free(out);
        free(wf);
        free(pup);
        return;
    }

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int w = 0; w < 20; w++)
    {
        cacaosim_zwfs_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &cfg.det, out, NULL);
    }
    for (int k = 0; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_zwfs_step(&ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &cfg.det, out, NULL);
        lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
    }

    char label[64];
    snprintf(label, sizeof(label), "ZWFS %u->%u (32x32 Babinet MFT)", nwf, ncam);
    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_zwfs_free(&ctx);
    free(lat);
    free(out);
    free(wf);
    free(pup);
}

/**
 * @brief Benchmark dual-output coronagraphic beam propagation (post-Lyot + LOWFS pickoff).
 *
 * Role:
 *   Measures `cacaosim_coro_step_beam()` latency for two complementary coronagraphic propagators:
 *     - `CACAOSIM_CORO_LYOT_SPOT`: Compact `48x48` Babinet Matrix-DFT over a `3.0 lambda/D` spot.
 *     - `CACAOSIM_CORO_VORTEX`: Full-plane `4x`-padded 2D FFT for an unbounded charge-2 vortex.
 *   Both modes simultaneously synthesize the post-Lyot `[OPD, Amplitude]` science beam and the
 *   reflected `[OPD, Amplitude]` LOWFS beam.
 *
 * Rationale:
 *   On CUDA, input/output beam planes (`d_iopd..d_lamp`) are pre-allocated in device memory so the
 *   timed region (`cacaosim_coro_step_beam` + `cacaosim_cuda_stream_sync`) accurately reflects
 *   in-VRAM execution inside a fused ExAO optical bench DAG without artificial PCIe transfer bias.
 *
 * @param[in] nwf      Pupil wavefront grid dimension (`nwf x nwf` pixels).
 * @param[in] mode     Coronagraph mode (`CACAOSIM_CORO_LYOT_SPOT` or `CACAOSIM_CORO_VORTEX`).
 * @param[in] backend  Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter    Number of timed iterations to record after 20 warm-up steps.
 */
static void bench_coro(uint32_t nwf, int32_t mode, int32_t backend, int niter)
{
    uint64_t np      = (uint64_t) nwf * nwf;
    float   *pup     = (float *) cacaosim_aligned_calloc(np, sizeof(float));
    float   *wf      = (float *) cacaosim_aligned_calloc(np, sizeof(float));
    float   *out_opd = (float *) cacaosim_aligned_calloc(np, sizeof(float));
    float   *out_amp = (float *) cacaosim_aligned_calloc(np, sizeof(float));
    float   *low_opd = (float *) cacaosim_aligned_calloc(np, sizeof(float));
    float   *low_amp = (float *) cacaosim_aligned_calloc(np, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.05f, 88ULL, wf);

    cacaosim_coro_cfg_t cfg = {
        .nwf = nwf, .mode = mode, .n_mft = 48, .pad_factor = 4, .pup_diam = 0.9375f,
        .lambda_um = 1.65f, .fpm_rad_ld = 3.0f, .fpm_amp = 0.0f, .fpm_phase_rad = CACAOSIM_PI_F,
        .vortex_charge = 2, .lowfs_refl = 0.95f, .lyot_outer = 0.85f, .lyot_obs = 0.36f,
        .backend = backend
    };

    cacaosim_coro_ctx_t ctx;
    if (cacaosim_coro_init(&ctx, &cfg, NULL, NULL) != 0)
    {
        free(low_amp);
        free(low_opd);
        free(out_amp);
        free(out_opd);
        free(wf);
        free(pup);
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_coro_free(&ctx);
        free(low_amp);
        free(low_opd);
        free(out_amp);
        free(out_opd);
        free(wf);
        free(pup);
        return;
    }

    /* Pre-stage input/output pupil planes in GPU VRAM to isolate pure kernel execution latency */
    float *d_iopd = ctx.use_cuda ? cacaosim_cuda_alloc_f32(np) : NULL;
    float *d_iamp = ctx.use_cuda ? cacaosim_cuda_alloc_f32(np) : NULL;
    float *d_oopd = ctx.use_cuda ? cacaosim_cuda_alloc_f32(np) : NULL;
    float *d_oamp = ctx.use_cuda ? cacaosim_cuda_alloc_f32(np) : NULL;
    float *d_lopd = ctx.use_cuda ? cacaosim_cuda_alloc_f32(np) : NULL;
    float *d_lamp = ctx.use_cuda ? cacaosim_cuda_alloc_f32(np) : NULL;
    if (ctx.use_cuda)
    {
        cacaosim_cuda_h2d_f32(d_iopd, wf, np, NULL);
        cacaosim_cuda_h2d_f32(d_iamp, pup, np, NULL);
        cacaosim_cuda_stream_sync(NULL);
    }

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int k = -20; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_coro_step_beam(&ctx, wf, pup, out_opd, out_amp, low_opd, low_amp, d_iopd, d_iamp,
                                d_oopd, d_oamp, d_lopd, d_lamp);
        if (ctx.use_cuda)
        {
            cacaosim_cuda_stream_sync(NULL);
        }
        if (k >= 0)
        {
            lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
        }
    }

    char label[64];
    snprintf(label, sizeof(label), "Coro %u->%u (%s + LOWFS)", nwf, nwf,
             (mode == CACAOSIM_CORO_LYOT_SPOT) ? "Lyot MFT" : "Vortex 4x");
    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    cacaosim_cuda_free_f32(d_lamp);
    cacaosim_cuda_free_f32(d_lopd);
    cacaosim_cuda_free_f32(d_oamp);
    cacaosim_cuda_free_f32(d_oopd);
    cacaosim_cuda_free_f32(d_iamp);
    cacaosim_cuda_free_f32(d_iopd);
    cacaosim_coro_free(&ctx);
    free(lat);
    free(low_amp);
    free(low_opd);
    free(out_amp);
    free(out_opd);
    free(wf);
    free(pup);
}

/**
 * @brief Benchmark end-to-end master-tick execution of a declarative multi-stage .bench DAG file.
 *
 * Role:
 *   Loads `<CACAOSIM_SRC_DIR>/<rel_path>` via `cacaosim_bench_load_file()`, overrides `cfg.backend`
 *   to CPU or CUDA, initializes the full topological optical graph (`cacaosim_bench_init()`), and
 *   measures per-master-tick wall-clock latency of `cacaosim_bench_step()`.
 *
 * Rationale:
 *   Qualifies real-world system frame rates on complete multi-arm observatory topologies (such as
 *   2-stage Woofer+Tweeter cascaded AO and ExAO TTM+DM+SHWFS+Coronagraph+Z-LOWFS+FPcam), capturing
 *   beam-relay overhead, dichroic routing, and periodic `navg` detector readouts in `p99` tail
 *   statistics so users can verify margin above each bench's target `master_hz` (`2..3 kHz`).
 *
 * @param[in] rel_path  Path to `.bench` file relative to `$CACAOSIM_SRC_DIR` (or `.`).
 * @param[in] label     Row label printed in the benchmark summary table.
 * @param[in] backend   Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] niter     Number of timed master ticks to record after 20 warm-up ticks.
 */
static void bench_dag_file(const char *rel_path, const char *label, int32_t backend, int niter)
{
    const char *src_dir = getenv("CACAOSIM_SRC_DIR");
    if (src_dir == NULL || src_dir[0] == '\0')
    {
        src_dir = ".";
    }
    char path[512];
    char err[256];
    snprintf(path, sizeof(path), "%s/%s", src_dir, rel_path);

    cacaosim_bench_cfg_t cfg;
    if (cacaosim_bench_load_file(path, &cfg, err, sizeof(err)) != 0)
    {
        return;
    }
    cfg.backend = backend;

    cacaosim_bench_ctx_t ctx;
    if (cacaosim_bench_init(&ctx, &cfg, NULL, 0, err, sizeof(err)) != 0)
    {
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_bench_free(&ctx);
        return;
    }

    float *wf = (float *) cacaosim_aligned_calloc(ctx.npix_wf, sizeof(float));
    cacaosim_make_ncpa(ctx.nwf, cfg.pup_diam, ctx.pup_ptr, 0.25f, 123ULL, wf);

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int k = -20; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
        cacaosim_bench_step(&ctx, wf, NULL, 1.0f, NULL);
        if (ctx.use_cuda)
        {
            cacaosim_cuda_stream_sync(ctx.cuda_stream);
        }
        if (k >= 0)
        {
            lat[k] = (double) (cacaosim_now_ns() - t0) * 1e-3;
        }
    }

    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, niter);

    free(lat);
    free(wf);
    cacaosim_bench_free(&ctx);
}

/**
 * @brief Benchmark per-base-WFS-frame latency of an INI bench across 1x/2x/4x oversampling.
 *
 * Role:
 *   Parses `@ini_text` (or loads `@rel_path` when `@ini_text == NULL`), overrides `cfg.oversample`
 *   and `cfg.backend`, and times `K = ctx.eff_oversample` consecutive `cacaosim_bench_step()`
 *   sub-ticks per sample so reported latencies and kHz throughputs represent full base-AO WFS
 *   frames on an apples-to-apples basis across `K = 1, 2, 4`.
 *
 * Rationale:
 *   Directly quantifies the net wall-clock cost of `2..4x` temporal oversampling when combined
 *   with `sync_arc` Pyramid modulation slicing, reverse DAG liveness pruning, `FPCAM` subframe
 *   striding, and unchanged-DM OPD skip.
 *
 * @param[in] rel_path    Relative `.bench` path under `$CACAOSIM_SRC_DIR`, or `NULL`.
 * @param[in] ini_text    Inline `.bench` INI string when `@rel_path == NULL`.
 * @param[in] label       Row label (`<= 38` chars) printed in the benchmark summary table.
 * @param[in] oversample  Master temporal oversampling factor (`1`, `2`, or `4`).
 * @param[in] backend     Requested compute backend (`CACAOSIM_BACKEND_CPU` or `CUDA`).
 * @param[in] nframes     Number of timed base WFS exposures to record after warm-up.
 */
static void bench_dag_oversample(const char *rel_path, const char *ini_text, const char *label,
                                 uint32_t oversample, int32_t backend, int nframes)
{
    char                 err[256];
    cacaosim_bench_cfg_t cfg;
    if (rel_path != NULL)
    {
        const char *src_dir = getenv("CACAOSIM_SRC_DIR");
        if (src_dir == NULL || src_dir[0] == '\0')
        {
            src_dir = ".";
        }
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", src_dir, rel_path);
        if (cacaosim_bench_load_file(path, &cfg, err, sizeof(err)) != 0)
        {
            return;
        }
    }
    else if (ini_text != NULL)
    {
        if (cacaosim_bench_parse_str(ini_text, &cfg, err, sizeof(err)) != 0)
        {
            return;
        }
    }
    else
    {
        return;
    }

    cfg.oversample = oversample;
    cfg.backend    = backend;

    cacaosim_bench_ctx_t ctx;
    if (cacaosim_bench_init(&ctx, &cfg, NULL, 0, err, sizeof(err)) != 0)
    {
        return;
    }
    if (backend == CACAOSIM_BACKEND_CUDA && !ctx.use_cuda)
    {
        cacaosim_bench_free(&ctx);
        return;
    }

    float *wf   = (float *) cacaosim_aligned_calloc(ctx.npix_wf, sizeof(float));
    float *d_wf = NULL;
    cacaosim_make_ncpa(ctx.nwf, cfg.pup_diam, ctx.pup_ptr, 0.25f, 123ULL, wf);
    if (ctx.use_cuda)
    {
        d_wf = cacaosim_cuda_alloc_f32(ctx.npix_wf);
        cacaosim_cuda_h2d_f32(d_wf, wf, ctx.npix_wf, ctx.cuda_stream);
        cacaosim_cuda_stream_sync(ctx.cuda_stream);
    }

    uint32_t K   = (ctx.eff_oversample >= 1u) ? ctx.eff_oversample : 1u;
    double  *lat = (double *) calloc((size_t) nframes, sizeof(double));
    for (int f = -20; f < nframes; f++)
    {
        uint64_t t0 = cacaosim_now_ns();
        for (uint32_t s = 0; s < K; s++)
        {
            cacaosim_bench_step(&ctx, ctx.use_cuda ? NULL : wf, d_wf, 1.0f, NULL);
        }
        if (ctx.use_cuda)
        {
            cacaosim_cuda_stream_sync(ctx.cuda_stream);
        }
        if (f >= 0)
        {
            lat[f] = (double) (cacaosim_now_ns() - t0) * 1e-3;
        }
    }

    print_stats(label, ctx.use_cuda ? "CUDA" : "CPU", lat, nframes);

    cacaosim_cuda_free_f32(d_wf);
    free(lat);
    free(wf);
    cacaosim_bench_free(&ctx);
}

/** Inline 256x256 2-stage AO3k (64^2, 1 kHz) + SCExAO (50^2, 2 kHz) + 6x broadband MFT FPcam. */
static const char *const g_ao3k_scexao_bench_ini =
    "[bench]\n"
    "name = ao3k_scexao_256\n"
    "nwf = 256\n"
    "pup_diam = 0.9375\n"
    "cobs = 0.30\n"
    "nspider = 4\n"
    "spider_w = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag = 6.0\n"
    "master_hz = 2000.0\n"
    "out_beam = b_pup\n"
    "[fpcam.vis_raw]\n"
    "in = b_pup\n"
    "out_cam = psf_vis_raw\n"
    "ncam = 256\n"
    "prop_mode = mft\n"
    "lambda_um = 0.75\n"
    "bw_frac = 0.15\n"
    "nlambda = 3\n"
    "rate_hz = 100.0\n"
    "out_dtype = float32\n"
    "[fpcam.nir_raw]\n"
    "in = b_pup\n"
    "out_cam = psf_nir_raw\n"
    "ncam = 256\n"
    "prop_mode = mft\n"
    "lambda_um = 1.65\n"
    "bw_frac = 0.18\n"
    "nlambda = 3\n"
    "rate_hz = 100.0\n"
    "out_dtype = float32\n"
    "[dm.ao3k]\n"
    "in = b_pup\n"
    "out = b_post_stg1\n"
    "in_cmd = dm01disp\n"
    "nact_x = 64\n"
    "nact_y = 64\n"
    "dyn_mode = 1\n"
    "tau_us = 200.0\n"
    "delay_us = 1000.0\n"
    "[fpcam.vis_stg1]\n"
    "in = b_post_stg1\n"
    "out_cam = psf_vis_stg1\n"
    "ncam = 256\n"
    "prop_mode = mft\n"
    "lambda_um = 0.75\n"
    "bw_frac = 0.15\n"
    "nlambda = 3\n"
    "rate_hz = 100.0\n"
    "out_dtype = float32\n"
    "[fpcam.nir_stg1]\n"
    "in = b_post_stg1\n"
    "out_cam = psf_nir_stg1\n"
    "ncam = 256\n"
    "prop_mode = mft\n"
    "lambda_um = 1.65\n"
    "bw_frac = 0.18\n"
    "nlambda = 3\n"
    "rate_hz = 100.0\n"
    "out_dtype = float32\n"
    "[splitter.dichroic_stg1]\n"
    "in = b_post_stg1\n"
    "out_refl = b_nir_wfs\n"
    "out_trans = b_stg2_in\n"
    "mode = dichroic_bp\n"
    "lambda_cut = 1.22\n"
    "band_width = 0.30\n"
    "[pyrwfs.wfs_nir]\n"
    "in = b_nir_wfs\n"
    "out_cam = csim_wfs_nir\n"
    "ncam = 160\n"
    "lambda_um = 1.22\n"
    "rate_hz = 1000.0\n"
    "mod_rad = 3.0\n"
    "mod_npts = 16\n"
    "out_dtype = float32\n"
    "[dm.scexao]\n"
    "in = b_stg2_in\n"
    "out = b_post_stg2\n"
    "in_cmd = dm02disp\n"
    "nact_x = 50\n"
    "nact_y = 50\n"
    "dyn_mode = 1\n"
    "tau_us = 60.0\n"
    "delay_us = 500.0\n"
    "[splitter.dichroic_stg2]\n"
    "in = b_post_stg2\n"
    "out_refl = b_vis_arm\n"
    "out_trans = b_nir_sci\n"
    "mode = dichroic_lp\n"
    "lambda_cut = 1.00\n"
    "[splitter.vis_pickoff]\n"
    "in = b_vis_arm\n"
    "out_refl = b_vis_wfs\n"
    "out_trans = b_vis_sci\n"
    "mode = gray\n"
    "r_gray = 0.90\n"
    "t_gray = 0.10\n"
    "[pyrwfs.wfs_vis]\n"
    "in = b_vis_wfs\n"
    "out_cam = csim_wfs_vis\n"
    "ncam = 120\n"
    "lambda_um = 0.75\n"
    "rate_hz = 2000.0\n"
    "mod_rad = 3.0\n"
    "mod_npts = 16\n"
    "out_dtype = float32\n"
    "[fpcam.vis_final]\n"
    "in = b_vis_sci\n"
    "out_cam = psf_vis_final\n"
    "ncam = 256\n"
    "prop_mode = mft\n"
    "lambda_um = 0.75\n"
    "bw_frac = 0.15\n"
    "nlambda = 3\n"
    "rate_hz = 100.0\n"
    "out_dtype = float32\n"
    "[fpcam.nir_final]\n"
    "in = b_nir_sci\n"
    "out_cam = psf_nir_final\n"
    "ncam = 256\n"
    "prop_mode = mft\n"
    "lambda_um = 1.65\n"
    "bw_frac = 0.18\n"
    "nlambda = 3\n"
    "rate_hz = 100.0\n"
    "out_dtype = float32\n";

/**
 * @brief Entry point running all unit and full-DAG latency/throughput benchmarks.
 *
 * Role:
 *   Prints the formatted table header and sequentially executes CPU and CUDA benchmarks across DM
 *   surfaces, Pyramid/Shack-Hartmann/Zernike WFSs, Lyot/Vortex coronagraphs, MFT/FFT/broadband
 *   science cameras, fused single-stage HT loops, and multi-stage `.bench` topologies.
 *
 * Rationale:
 *   Iteration counts (`50..500`) balance statistical convergence of `p99` quantiles against total
 *   wall-clock runtime on CPU-heavy configurations (such as `256x256` 16-point modulated PyWFS),
 *   allowing interactive regression profiling in a few seconds.
 *
 * @return Always `0`.
 */
int main(void)
{
    printf("==================================================================================\n");
    printf("  cacaosim Runtime Latency & Throughput Benchmark\n");
    printf("==================================================================================\n");
    printf("  %-38s | %-4s | %8s | %8s | %8s | %s\n", "Compute Pipeline", "Back", "Mean us",
           "p50 us", "p99 us", "Throughput");
    printf("----------------------------------------------------------------------------------\n");

    /* 1. Deformable Mirror dynamics + gather-stencil OPD rendering (classical & ExAO scales) */
    bench_dm(20, 128, CACAOSIM_BACKEND_CPU, 500);
    bench_dm(20, 128, CACAOSIM_BACKEND_CUDA, 500);
    bench_dm(50, 256, CACAOSIM_BACKEND_CPU, 300);
    bench_dm(50, 256, CACAOSIM_BACKEND_CUDA, 500);

    /* 2. 4-sided Pyramid WFS (unmodulated vs 16-point ring modulation, 128 & 256 grids) */
    bench_pyrwfs(128, 64, 0.0f, 1, CACAOSIM_BACKEND_CPU, 300);
    bench_pyrwfs(128, 64, 0.0f, 1, CACAOSIM_BACKEND_CUDA, 500);
    bench_pyrwfs(128, 64, 3.0f, 16, CACAOSIM_BACKEND_CPU, 150);
    bench_pyrwfs(128, 64, 3.0f, 16, CACAOSIM_BACKEND_CUDA, 500);
    bench_pyrwfs(256, 128, 3.0f, 16, CACAOSIM_BACKEND_CPU, 50);
    bench_pyrwfs(256, 128, 3.0f, 16, CACAOSIM_BACKEND_CUDA, 300);

    /* 3. Diffractive Shack-Hartmann WFS & Babinet MFT Zernike phase-contrast WFS */
    bench_shwfs(128, 16, 8, CACAOSIM_BACKEND_CPU, 200);
    bench_shwfs(128, 16, 8, CACAOSIM_BACKEND_CUDA, 500);

    bench_zwfs(128, 64, CACAOSIM_BACKEND_CPU, 300);
    bench_zwfs(128, 64, CACAOSIM_BACKEND_CUDA, 500);

    /* 4. Coronagraphic starlight suppression + LOWFS pickoff (Babinet MFT vs 4x-padded FFT) */
    bench_coro(128, CACAOSIM_CORO_LYOT_SPOT, CACAOSIM_BACKEND_CPU, 200);
    bench_coro(128, CACAOSIM_CORO_LYOT_SPOT, CACAOSIM_BACKEND_CUDA, 500);
    bench_coro(128, CACAOSIM_CORO_VORTEX, CACAOSIM_BACKEND_CPU, 100);
    bench_coro(128, CACAOSIM_CORO_VORTEX, CACAOSIM_BACKEND_CUDA, 500);

    /* 5. Focal-plane science PSF camera (Soummer MFT mono/5-lambda broadband vs 2x FFT) */
    bench_fpcam(128, 128, 0, 1, CACAOSIM_BACKEND_CPU, 300);
    bench_fpcam(128, 128, 0, 1, CACAOSIM_BACKEND_CUDA, 500);
    bench_fpcam(128, 128, 0, 5, CACAOSIM_BACKEND_CPU, 150);
    bench_fpcam(128, 128, 0, 5, CACAOSIM_BACKEND_CUDA, 500);
    bench_fpcam(128, 128, 1, 1, CACAOSIM_BACKEND_CPU, 300);
    bench_fpcam(128, 128, 1, 1, CACAOSIM_BACKEND_CUDA, 500);

    /* 6. Zero-IPC fused single-stage HT bench & declarative multi-stage .bench DAGs */
    bench_fused_htsim(128, 20, 64, CACAOSIM_BACKEND_CPU, 150);
    bench_fused_htsim(128, 20, 64, CACAOSIM_BACKEND_CUDA, 500);
    bench_fused_htsim(256, 50, 128, CACAOSIM_BACKEND_CUDA, 300);

    bench_dag_file("examples/two_stage_ao.bench", "2-Stage Bench DAG (Woofer+Tweeter)",
                   CACAOSIM_BACKEND_CPU, 120);
    bench_dag_file("examples/two_stage_ao.bench", "2-Stage Bench DAG (Woofer+Tweeter)",
                   CACAOSIM_BACKEND_CUDA, 400);
    bench_dag_file("examples/exao_coro_lowfs.bench", "ExAO DAG (TTM+DM+SH+Coro+ZWFS+FP)",
                   CACAOSIM_BACKEND_CPU, 120);
    bench_dag_file("examples/exao_coro_lowfs.bench", "ExAO DAG (TTM+DM+SH+Coro+ZWFS+FP)",
                   CACAOSIM_BACKEND_CUDA, 400);

    /* 7. Multi-rate temporal oversampling (1x vs 2x vs 4x per base WFS exposure) */
    printf("----------------------------------------------------------------------------------\n");
    bench_dag_oversample("examples/two_stage_ao.bench", NULL, "2-Stage 128^2 (1x / base WFS frame)",
                         1u, CACAOSIM_BACKEND_CPU, 90);
    bench_dag_oversample("examples/two_stage_ao.bench", NULL, "2-Stage 128^2 (2x oversample / WFS)",
                         2u, CACAOSIM_BACKEND_CPU, 90);
    bench_dag_oversample("examples/two_stage_ao.bench", NULL, "2-Stage 128^2 (4x oversample / WFS)",
                         4u, CACAOSIM_BACKEND_CPU, 90);
    bench_dag_oversample("examples/two_stage_ao.bench", NULL, "2-Stage 128^2 (1x / base WFS frame)",
                         1u, CACAOSIM_BACKEND_CUDA, 300);
    bench_dag_oversample("examples/two_stage_ao.bench", NULL, "2-Stage 128^2 (2x oversample / WFS)",
                         2u, CACAOSIM_BACKEND_CUDA, 300);
    bench_dag_oversample("examples/two_stage_ao.bench", NULL, "2-Stage 128^2 (4x oversample / WFS)",
                         4u, CACAOSIM_BACKEND_CUDA, 300);
    bench_dag_oversample(NULL, g_ao3k_scexao_bench_ini, "AO3k+SCExAO 256^2 (1x / 2kHz frame)", 1u,
                         CACAOSIM_BACKEND_CUDA, 200);
    bench_dag_oversample(NULL, g_ao3k_scexao_bench_ini, "AO3k+SCExAO 256^2 (2x / 2kHz frame)", 2u,
                         CACAOSIM_BACKEND_CUDA, 200);
    bench_dag_oversample(NULL, g_ao3k_scexao_bench_ini, "AO3k+SCExAO 256^2 (4x / 2kHz frame)", 4u,
                         CACAOSIM_BACKEND_CUDA, 200);

    printf("==================================================================================\n");
    return 0;
}
