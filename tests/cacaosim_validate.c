// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_validate.c
 * @brief   Physical, statistical, CPU/CUDA parity, and closed-loop validation suite for cacaosim
 *
 * Verification Philosophy & Four-Pillar Methodology
 * -------------------------------------------------
 * High-contrast Adaptive Optics simulation is uniquely vulnerable to subtle numerical bugs --
 * half-pixel FFT centering shifts, missing `2 * pi / lambda` vs `1e6` unit factors, inaccurate
 * Euler vs Zero-Order-Hold (ZOH) actuator discretization, non-conservative dichroic splitting, or
 * divergent CPU/GPU floating-point kernels -- that do not crash the program yet silently invalidate
 * closed-loop stability margins, Strehl ratios, or coronagraphic raw contrast.
 *
 * To guarantee end-to-end physical and software fidelity, `cacaosim_validate` combines four
 * orthogonal verification pillars inside a single deterministic test executable:
 *
 *   1. Exact Analytical Physics Invariants:
 *      Every optical and dynamical engine is checked against closed-form continuum theory:
 *      sub-pixel annular area `pi * (R^2 - r^2)`, zero-mean Zernike/PSD RMS wavefront
 *      normalization, ZOH step response `1 - exp(-dt / tau)` and underdamped resonance overshoot
 *      `1 + exp(-pi * zeta / sqrt(1 - zeta^2))`, Maréchal Strehl degradation `exp(-sigma_phi^2)`,
 *      Pogson's `100x` per `5 mag` photometric scaling, Pyramid/SHWFS linear slope response,
 *      Babinet Zernike phase-contrast antisymmetry, Lyot/Vortex stellar suppression, and two-port
 *      coronagraphic LOWFS energy conservation `|A_sci|^2 + |A_lowfs|^2 = |A_in|^2`.
 *
 *   2. Large-Sample Statistical Noise Laws:
 *      Detector readout physics (`cacaosim_detector_readout_cpu`) is tested over `N = 262,144`
 *      pixels so Monte Carlo standard errors shrink below `0.3%`, verifying closed-shutter dark
 *      isolation (`sigma_ADU = ron_e / gain`) and open-shutter compound Poisson + dark + readout
 *      variance `Var(ADU) = (flux_e + dark_e + ron_e^2) / gain^2`.
 *
 *   3. CPU vs CUDA Numerical Parity:
 *      Whenever a CUDA GPU is present (`cacaosim_cuda_available()`), every compute unit—DM gather
 *      stencil, Pyramid CUDA Graph replay, 5-wavelength strided-batched cuBLAS FPcam, affine pupil
 *      relay, batched cuFFT SHWFS, cuBLAS ZWFS, MFT/FFT Coronagraph + dual LOWFS pickoffs, 3D
 *      polychromatic FITS mask cubes, and SI `[meter]` WFE nodes—is compared directly against the
 *      CPU reference to `1e-6 .. 5e-5` pointwise or relative `L_2` tolerance.
 *
 *   4. Closed-Loop AO Convergence & Declarative Bench Tooling:
 *      Calibrates push-pull interaction matrices, computes Tikhonov-regularized reconstructors,
 *      and closes leaky-integrator feedback loops under 1-frame delay + ZOH lag + photon/readout
 *      noise on both single-stage and 2-stage cascaded (Woofer + Tweeter) benches, while also
 *      verifying `.bench` cycle rejection, lossless INI/TOML round-trips, topological node
 *      splicing, and headless UTF-8 / 7-bit ASCII `cacaosim-CTRL` & `cacaosim-bench-view` renders.
 */

#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milkcacaosim.h"
#include "bench/cacaosim_tui_internal.h"

/** Cumulative count of passing validation assertions across all test sections. */
static int g_pass = 0;

/** Cumulative count of failing validation assertions across all test sections. */
static int g_fail = 0;

/**
 * @brief Record and print the outcome of a single quantitative validation assertion.
 *
 * Role:
 *   Formats a standardized `[PASS]` or `[FAIL]` line displaying both the invariant description
 *   (`label`) and measured vs theoretical telemetry (`detail`), updating `g_pass` or `g_fail`.
 *
 * Rationale:
 *   Accumulating failures rather than calling `abort()` on the first mismatch allows the entire
 *   15-section suite to run in one shot so engineers can immediately see whether a regression is
 *   isolated to one optical mode or systemic across CPU/CUDA backends, while `main()` still returns
 *   exit code `1` to CTest if `g_fail > 0`.
 *
 * @param[in] ok      Non-zero if the physical/numerical invariant passed; `0` on failure.
 * @param[in] label   Short description of the tested invariant (`<= 46` chars for alignment).
 * @param[in] detail  Formatted quantitative telemetry string (measured vs reference values).
 */
static void check_cond(int ok, const char *label, const char *detail)
{
    if (ok)
    {
        printf("  [PASS] %-46s (%s)\n", label, detail);
        g_pass++;
    }
    else
    {
        printf("  [FAIL] %-46s (%s)\n", label, detail);
        g_fail++;
    }
}

/**
 * @brief Section 1: Validate anti-aliased annular pupil geometry and static NCPA RMS calibration.
 *
 * Role:
 *   Verifies `cacaosim_make_pupil()` grayscale aperture integration against exact continuum annulus
 *   area `pi * R_out^2 * (1 - cobs^2)` and checks `cacaosim_make_ncpa()` pupil-weighted mean and
 *   RMS wavefront amplitude (`target_rms = 0.085 um`).
 *
 * Rationale:
 *   Binary (non-anti-aliased) pupil masks suffer `O(1 / N_wf)` perimeter staircase errors (~1..2%
 *   area bias at `N_wf = 128`) and inject spurious high-spatial-frequency diffraction ripples into
 *   coronagraphic Lyot planes and Zernike/Pyramid WFS pupils. Requiring `< 0.4%` area error proves
 *   sub-pixel edge supersampling is active, while `< 1e-6 um` piston residual and `< 1e-4 um` RMS
 *   error guarantee exact wavefront scaling for downstream Maréchal Strehl tests.
 */
static void test_pupil_and_ncpa(void)
{
    printf("\n=== 1. Pupil Anti-Aliasing & Static NCPA ===\n");
    const uint32_t nwf  = 128;
    const float    diam = 0.90f;
    const float    cobs = 0.30f;
    float         *pup  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float         *ncpa = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf        = nwf,
        .pup_diam   = diam,
        .cobs       = cobs,
        .nspider    = 0,
        .spider_w   = 0.0f,
        .spider_deg = 0.0f,
    };
    double area_meas = cacaosim_make_pupil(&pcfg, pup);
    double r_out     = 0.5 * (double) diam * (double) nwf;
    double area_ref  = M_PI * r_out * r_out * (1.0 - (double) cobs * (double) cobs);
    double rel_err   = fabs(area_meas - area_ref) / area_ref;

    char buf[128];
    snprintf(buf, sizeof(buf), "meas=%.1f px, ref=%.1f px, err=%.4f%%", area_meas, area_ref,
             100.0 * rel_err);
    check_cond(rel_err < 0.004, "Annular pupil area matches pi*(R^2-r^2)", buf);

    const float target_rms = 0.085f;
    cacaosim_make_ncpa(nwf, diam, pup, target_rms, 42ULL, ncpa);

    double wsum  = 0.0;
    double wmean = 0.0;
    for (uint32_t i = 0; i < nwf * nwf; i++)
    {
        wsum += pup[i];
        wmean += pup[i] * ncpa[i];
    }
    wmean /= wsum;
    double wvar = 0.0;
    for (uint32_t i = 0; i < nwf * nwf; i++)
    {
        double d = ncpa[i] - wmean;
        wvar += pup[i] * d * d;
    }
    double meas_rms = sqrt(wvar / wsum);
    snprintf(buf, sizeof(buf), "mean=%.2e um, rms=%.5f um vs %.5f um", wmean, meas_rms, target_rms);
    check_cond(fabs(wmean) < 1e-6 && fabs(meas_rms - target_rms) < 1e-4,
               "NCPA Zernike map is zero-mean & exact RMS", buf);

    free(ncpa);
    free(pup);
}

/**
 * @brief Section 2: Validate detector readout noise statistics (shutter darks, Poisson + RON).
 *
 * Role:
 *   Tests `cacaosim_detector_readout_cpu()` across `262,144` uniform pixels in two physical modes:
 *     1. Closed shutter (`shutter = 0`): verifies optical flux isolation, pedestal `bias_adu`, and
 *        Gaussian readout standard deviation `sigma_ADU = ron_e / gain_eadu`.
 *     2. Open shutter (`shutter = 1`): verifies compound Poisson photo-electron + dark current +
 *        Gaussian read-noise mean `bias + (flux + dark) / gain` and variance
 *        `Var(ADU) = (flux + dark + ron_e^2) / gain^2`.
 *
 * Rationale:
 *   WFS thresholded center-of-gravity and optimal linear reconstructor weighting depend directly on
 *   accurate photon-to-ADU gain scaling and uncorrelated Poisson + Gaussian variance addition.
 *   Sampling `N = 2^18 = 262,144` pixels reduces 1-sigma Monte Carlo uncertainty on the sample
 *   mean to `sigma / sqrt(N) < 0.01 ADU` and on sample variance to `sqrt(2 / N) ~ 0.28%`, enabling
 *   tight deterministic tolerances (`0.05..0.35`) with a fixed PRNG seed (`77777ULL`).
 */
static void test_detector_stats(void)
{
    printf("\n=== 2. Detector Readout Physics (Poisson + RON + Shutter) ===\n");
    const uint64_t      npix = 262144ULL;
    cacaosim_det_pool_t pool;
    cacaosim_det_pool_init(&pool, 4, 77777ULL);

    float *opt = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *out = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    for (uint64_t i = 0; i < npix; i++)
    {
        opt[i] = 1.0f;
    }

    /* Closed shutter dark frame: bias=200 ADU, RON=2.5 e-, gain=0.5 e-/ADU -> sigma_ADU = 5.0 */
    cacaosim_det_cfg_t dcfg = {
        .flux_ph      = 1000.0f,
        .dark_e       = 0.0f,
        .ron_e        = 2.5f,
        .gain_eadu    = 0.5f,
        .bias_adu     = 200.0f,
        .sat_adu      = 65535.0f,
        .shutter      = 0,
        .photon_noise = 1,
        .out_dtype    = 1,
    };
    cacaosim_detector_readout_cpu(&pool, &dcfg, opt, npix, 0.0f, NULL, out);

    double mean = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        mean += out[i];
    }
    mean /= (double) npix;
    double var = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        double d = out[i] - mean;
        var += d * d;
    }
    double std_adu = sqrt(var / (double) npix);
    char   buf[128];
    snprintf(buf, sizeof(buf), "mean=%.2f ADU (200), std=%.3f ADU (5.000)", mean, std_adu);
    check_cond(fabs(mean - 200.0) < 0.05 && fabs(std_adu - 5.0) < 0.06,
               "Closed-shutter dark frame bias & RON sigma", buf);

    /* Open shutter: 100 ph/pix + dark=4 e-, RON=3 e-, gain=2 e-/ADU -> var_e = 104+9 = 113 */
    dcfg.shutter   = 1;
    dcfg.flux_ph   = 100.0f;
    dcfg.dark_e    = 4.0f;
    dcfg.ron_e     = 3.0f;
    dcfg.gain_eadu = 2.0f;
    dcfg.bias_adu  = 100.0f;
    cacaosim_detector_readout_cpu(&pool, &dcfg, opt, npix, 0.0f, NULL, out);

    mean = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        mean += out[i];
    }
    mean /= (double) npix;
    var = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        double d = out[i] - mean;
        var += d * d;
    }
    var /= (double) npix;
    double exp_mean = 100.0 + 104.0 / 2.0;
    double exp_var  = (104.0 + 9.0) / 4.0;
    snprintf(buf, sizeof(buf), "mean=%.2f (%.2f), var=%.2f (%.2f)", mean, exp_mean, var, exp_var);
    check_cond(fabs(mean - exp_mean) < 0.08 && fabs(var - exp_var) < 0.35,
               "Poisson + dark + RON mean and variance", buf);

    cacaosim_det_pool_free(&pool);
    free(out);
    free(opt);
}

/**
 * @brief Section 3: Validate DM fractional delay, piston normalization, ZOH dynamics & GPU parity.
 *
 * Role:
 *   Validates five hardware invariants of `cacaosim_dm_*`:
 *     1. Fractional `2.5`-frame command ring-buffer interpolation (`[0, 0, 0.5, 1.0]` step).
 *     2. Gather-stencil partition-of-unity interior piston (`1.0 um` stroke -> `2.0 um` OPD).
 *     3. Exact 1st-order Zero-Order-Hold (ZOH) exponential step `1 - exp(-dt / tau)` at `dt = tau`.
 *     4. Exact 2nd-order ZOH underdamped resonance peak overshoot `1 + exp(-pi*zeta/sqrt(1-z^2))`.
 *     5. Pointwise CPU vs CUDA OPD surface agreement (`max |CPU - CUDA| < 1e-6 um`).
 *
 * Rationale:
 *   - Fractional frame delays (`delay_frames = 2.5`) occur whenever RTC compute + camera transfer
 *     time is not an integer multiple of `1 / master_hz`; linear ring interpolation reproduces the
 *     exact phase margin degradation in Bode stability tests.
 *   - Partition-of-unity normalization ensures uniform actuator commands produce pure optical
 *     piston without artificial actuator-pitch print-through ripples.
 *   - Exact matrix-exponential ZOH state updates remain unconditionally stable and exact even when
 *     `dt >= tau` or near resonance `f_0`, where forward-Euler integration diverges or distorts Q.
 */
static void test_dm_physics(void)
{
    printf("\n=== 3. DM Hardware Simulator (Delay, ZOH Dynamics, Gather IF) ===\n");
    cacaosim_dm_cfg_t cfg = {
        .nact_x       = 20,
        .nact_y       = 20,
        .nwf          = 128,
        .dm_diam      = 0.9375f,
        .if_coupling  = 0.22f,
        .refl_scale   = 2.0f,
        .max_stroke   = 5.0f,
        .dac_bits     = 0,
        .nonlin_quad  = 0.0f,
        .dyn_mode     = 0,
        .dt_sec       = 0.0005f,
        .tau_sec      = 0.0005f,
        .f0_hz        = 1000.0f,
        .zeta         = 0.50f,
        .delay_frames = 2.5f,
        .backend      = CACAOSIM_BACKEND_CPU,
    };

    cacaosim_dm_ctx_t ctx;
    cacaosim_dm_init(&ctx, &cfg, NULL);

    float *cmd     = (float *) cacaosim_aligned_calloc(400, sizeof(float));
    float *opd_cpu = (float *) cacaosim_aligned_calloc(128 * 128, sizeof(float));
    for (int i = 0; i < 400; i++)
    {
        cmd[i] = 1.0f;
    }

    /* Fractional 2.5-frame pure delay on unit step: k=0->0, k=1->0, k=2->0.5, k=3->1.0 */
    float seq[4];
    for (int k = 0; k < 4; k++)
    {
        cacaosim_dm_step_actuators(&ctx, cmd, 2.5f, 0.0f, 5.0f);
        seq[k] = ctx.act_pos[0];
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "y=[%.2f, %.2f, %.2f, %.2f] vs [0, 0, 0.5, 1]", seq[0], seq[1],
             seq[2], seq[3]);
    check_cond(fabsf(seq[0]) < 1e-6f && fabsf(seq[1]) < 1e-6f && fabsf(seq[2] - 0.5f) < 1e-6f &&
                   fabsf(seq[3] - 1.0f) < 1e-6f,
               "Fractional 2.5-frame command ring delay", buf);

    /* Interior piston normalization: cmd=1.0 um * refl_scale=2.0 -> OPD=2.0 um */
    cacaosim_dm_render_opd_cpu(&ctx, NULL, opd_cpu);
    float center_opd = opd_cpu[64 * 128 + 64];
    snprintf(buf, sizeof(buf), "center OPD=%.6f um vs 2.000000 um", center_opd);
    check_cond(fabsf(center_opd - 2.0f) < 1e-4f,
               "Gather-stencil interior piston = 2.0 * stroke", buf);

    /* 1st-order ZOH step response: after 1 step with dt == tau, pos = 1 - exp(-1) */
    cacaosim_dm_reset(&ctx);
    ctx.cfg.dyn_mode = 1;
    cacaosim_dm_step_actuators(&ctx, cmd, 0.0f, 0.0005f, 5.0f);
    float exp_rc = 1.0f - expf(-1.0f);
    snprintf(buf, sizeof(buf), "pos(dt=tau)=%.6f vs 1-1/e=%.6f", ctx.act_pos[0], exp_rc);
    check_cond(fabsf(ctx.act_pos[0] - exp_rc) < 1e-6f,
               "1st-order ZOH exact exponential step", buf);

    /* 2nd-order ZOH underdamped peak overshoot: Mp = exp(-pi*zeta/sqrt(1-zeta^2)) */
    cacaosim_dm_reset(&ctx);
    ctx.cfg.dyn_mode = 2;
    ctx.cfg.dt_sec   = 0.00005f;
    ctx.cfg.f0_hz    = 1000.0f;
    ctx.cfg.zeta     = 0.50f;
    float peak       = 0.0f;
    for (int k = 0; k < 400; k++)
    {
        cacaosim_dm_step_actuators(&ctx, cmd, 0.0f, 0.0f, 5.0f);
        if (ctx.act_pos[0] > peak)
        {
            peak = ctx.act_pos[0];
        }
    }
    float th_peak = 1.0f + expf(-CACAOSIM_PI_F * 0.5f / sqrtf(1.0f - 0.25f));
    snprintf(buf, sizeof(buf), "peak=%.5f vs analytic %.5f", peak, th_peak);
    check_cond(fabsf(peak - th_peak) < 0.003f,
               "2nd-order ZOH resonance overshoot", buf);

    if (cacaosim_cuda_available())
    {
        for (int i = 0; i < 400; i++)
        {
            cmd[i] = 0.4f * sinf(0.37f * (float) i);
        }
        ctx.cfg.dyn_mode = 0;
        cacaosim_dm_reset(&ctx);
        cacaosim_dm_step(&ctx, cmd, NULL, 0.0f, 0.0f, 5.0f, opd_cpu);

        cacaosim_dm_cfg_t gcfg = cfg;
        gcfg.delay_frames      = 0.0f;
        gcfg.backend           = CACAOSIM_BACKEND_CUDA;
        cacaosim_dm_ctx_t gctx;
        cacaosim_dm_init(&gctx, &gcfg, NULL);
        float *opd_gpu = (float *) cacaosim_aligned_calloc(128 * 128, sizeof(float));
        cacaosim_dm_step(&gctx, cmd, NULL, 0.0f, 0.0f, 5.0f, opd_gpu);

        double max_diff = 0.0;
        for (int p = 0; p < 128 * 128; p++)
        {
            double d = fabs((double) opd_cpu[p] - (double) opd_gpu[p]);
            if (d > max_diff)
            {
                max_diff = d;
            }
        }
        snprintf(buf, sizeof(buf), "max |CPU - CUDA| = %.3e um", max_diff);
        check_cond(gctx.use_cuda && max_diff < 1e-6, "DM OPD CPU vs CUDA parity", buf);
        cacaosim_dm_free(&gctx);
        free(opd_gpu);
    }

    /* 3b. Coupled 2D circular-rim flexural+membrane plate PDE (dyn_mode == 3, ALPAO 64x64) */
    {
        const uint32_t pnact = 64u;
        const uint32_t pnwf  = 128u;
        cacaosim_dm_cfg_t pcfg = {
            .nact_x        = pnact,
            .nact_y        = pnact,
            .nwf           = pnwf,
            .dm_diam       = 0.984375f,
            .if_coupling   = 0.20f,
            .refl_scale    = 2.0f,
            .max_stroke    = 8.0f,
            .dac_bits      = 0,
            .nonlin_quad   = 0.0f,
            .dyn_mode      = 3,
            .dt_sec        = 5.0e-5f, /* 20 kHz */
            .tau_sec       = 2.0e-5f, /* 20 us voice-coil current step */
            .f0_hz         = 1000.0f,
            .act_pitch_mm  = 1.5f,
            .wave_c_ms     = 14.0f,
            .bend_beta_m2s = 0.0085f,
            .damp_g0       = 195.0f,
            .damp_g2       = 1.5f,
            .plate_drive   = 1,
            .rim_bc        = 1,
            .delay_frames  = 0.0f,
            .backend       = CACAOSIM_BACKEND_CPU,
        };
        cacaosim_dm_ctx_t pctx;
        int init_ok = (cacaosim_dm_init(&pctx, &pcfg, NULL) == 0);
        float *pcmd = (float *) cacaosim_aligned_calloc(pnact * pnact, sizeof(float));
        float *popd = (float *) cacaosim_aligned_calloc(pnwf * pnwf, sizeof(float));

        /* Store 440 frames (22 ms at 20 kHz) of 64x64 actuator displacement; poke at fr = 40 */
        const uint32_t nfr = 440u;
        float *hist        = (float *) calloc((size_t) nfr * pnact * pnact, sizeof(float));
        const uint32_t px  = 19u;
        const uint32_t py  = 42u;

        for (uint32_t fr = 0; fr < nfr; fr++)
        {
            if (fr == 40u)
            {
                pcmd[py * pnact + px] = 5.0f; /* ~0.91 um peak physical poke (force drive) */
            }
            cacaosim_dm_step_actuators(&pctx, pcmd, 0.0f, 2.0e-5f, 8.0f);
            memcpy(&hist[(size_t) fr * pnact * pnact], pctx.act_pos_render,
                   pnact * pnact * sizeof(float));
        }

        /* Measure 1 ms consecutive-exposure boxcar diff in far ring (18..36 act from poke) */
        double peak_5ms_nm  = 0.0;
        double peak_18ms_nm = 0.0;
        for (uint32_t ay = 0; ay < pnact; ay++)
        {
            for (uint32_t ax = 0; ax < pnact; ax++)
            {
                float rc = hypotf((float) ax - 31.5f, (float) ay - 31.5f);
                float rp = hypotf((float) ax - (float) px, (float) ay - (float) py);
                if (rc <= 30.5f && rp >= 18.0f && rp <= 36.0f)
                {
                    uint32_t id = ay * pnact + ax;
                    double   c5 = 0.0, p5 = 0.0, c18 = 0.0, p18 = 0.0;
                    for (uint32_t j = 0; j < 20u; j++)
                    {
                        c5  += (double) hist[(140u - j) * pnact * pnact + id];
                        p5  += (double) hist[(120u - j) * pnact * pnact + id];
                        c18 += (double) hist[(400u - j) * pnact * pnact + id];
                        p18 += (double) hist[(380u - j) * pnact * pnact + id];
                    }
                    double d5  = fabs(c5 - p5) * (1000.0 / 20.0);
                    double d18 = fabs(c18 - p18) * (1000.0 / 20.0);
                    if (d5 > peak_5ms_nm)
                    {
                        peak_5ms_nm = d5;
                    }
                    if (d18 > peak_18ms_nm)
                    {
                        peak_18ms_nm = d18;
                    }
                }
            }
        }
        /* Continue 800 more steps (60 ms total) to check continuous-facesheet static coupling */
        for (uint32_t k = 0; k < 800u; k++)
        {
            cacaosim_dm_step_actuators(&pctx, pcmd, 0.0f, 2.0e-5f, 8.0f);
        }
        float poke_dc  = pctx.act_pos[py * pnact + px];
        float nb_ratio = pctx.act_pos[py * pnact + px + 1u] / fmaxf(poke_dc, 1e-6f);

        /* Verify static CG solver (calib_fast) matches t->inf dynamic plate PDE equilibrium */
        float *stat_ref = (float *) calloc(pnact * pnact, sizeof(float));
        memcpy(stat_ref, pctx.act_pos, pnact * pnact * sizeof(float));
        cacaosim_dm_reset(&pctx);
        cacaosim_dm_step_static_plate(&pctx, pcmd, NULL, 8.0f, popd);
        float max_stat_err = 0.0f;
        for (uint32_t a = 0; a < pnact * pnact; a++)
        {
            float e = fabsf(pctx.act_pos_render[a] - stat_ref[a]);
            if (e > max_stat_err)
            {
                max_stat_err = e;
            }
        }
        free(stat_ref);

        /* Verify exact uniform piston DC = 1.0000 with force drive + free circular rim */
        cacaosim_dm_reset(&pctx);
        for (uint32_t a = 0; a < pnact * pnact; a++)
        {
            pcmd[a] = 1.0f;
        }
        cacaosim_dm_step_static_plate(&pctx, pcmd, NULL, 8.0f, popd);
        float piston_stat = pctx.act_pos_render[32u * pnact + 32u];
        cacaosim_dm_reset(&pctx);
        for (uint32_t k = 0; k < 1200u; k++)
        {
            cacaosim_dm_step_actuators(&pctx, pcmd, 0.0f, 2.0e-5f, 8.0f);
        }
        float piston_dc = pctx.act_pos[32u * pnact + 32u];

        snprintf(buf, sizeof(buf),
                 "5ms=%.2fnm, 18ms=%.2fnm, nb/ctr=%.3f, piston=%.5f, cg_err=%.2e, nsub=%u",
                 peak_5ms_nm, peak_18ms_nm, nb_ratio, piston_dc, max_stat_err, pctx.plate_nsub);
        check_cond(init_ok && peak_5ms_nm >= 2.0 && peak_5ms_nm <= 15.0 && peak_18ms_nm < 1.2 &&
                       peak_18ms_nm < 0.30 * peak_5ms_nm && nb_ratio >= 0.25f &&
                       nb_ratio <= 0.50f && fabsf(piston_dc - 1.0f) < 1e-3f &&
                       fabsf(piston_stat - 1.0f) < 1e-5f && max_stat_err < 2.0e-4f,
                   "DM plate PDE 64x64 ripples, decay & static DC", buf);

        /* Verify CPU vs CUDA parity for coupled 64x64 plate PDE + 16-tap gather */
        if (cacaosim_cuda_available())
        {
            cacaosim_dm_reset(&pctx);
            cacaosim_dm_cfg_t pgcfg = pcfg;
            pgcfg.backend           = CACAOSIM_BACKEND_CUDA;
            cacaosim_dm_ctx_t pgctx;
            int               g_ok = (cacaosim_dm_init(&pgctx, &pgcfg, NULL) == 0);
            float            *g_opd = (float *) cacaosim_aligned_calloc(pnwf * pnwf, sizeof(float));

            memset(pcmd, 0, pnact * pnact * sizeof(float));
            pcmd[py * pnact + px] = 5.0f;
            for (uint32_t fr = 0; fr < 80u; fr++)
            {
                cacaosim_dm_step(&pctx, pcmd, NULL, 0.0f, 2.0e-5f, 8.0f, popd);
                cacaosim_dm_step(&pgctx, pcmd, NULL, 0.0f, 2.0e-5f, 8.0f, g_opd);
            }
            double max_d = 0.0;
            for (uint32_t p = 0; p < pnwf * pnwf; p++)
            {
                double d = fabs((double) popd[p] - (double) g_opd[p]);
                if (d > max_d)
                {
                    max_d = d;
                }
            }
            snprintf(buf, sizeof(buf), "max |CPU - CUDA| = %.3e um (plate_cuda=%d)", max_d,
                     pgctx.plate_use_cuda);
            check_cond(g_ok && pgctx.plate_use_cuda && max_d < 1e-5,
                       "DM plate PDE 64x64 CPU vs CUDA parity", buf);
            cacaosim_dm_free(&pgctx);
            free(g_opd);
        }

        /* Verify .bench parse/format round-trip for dyn_mode = "plate" */
        static const char *const plate_ini =
            "[bench]\n"
            "nwf = 64\n"
            "master_hz = 2000.0\n"
            "out_beam = b_in\n"
            "[dm.ao3k]\n"
            "in = b_in\n"
            "out = b_out\n"
            "in_cmd = dm01disp\n"
            "nact_x = 64\n"
            "nact_y = 64\n"
            "dyn_mode = \"plate\"\n"
            "tau_us = 200\n"
            "f0_hz = 950\n"
            "act_pitch_mm = 1.5\n"
            "wave_c = 22\n"
            "bend_beta = 0.025\n"
            "damp_g0 = 150\n"
            "damp_g2 = 12\n"
            "plate_drive = \"force\"\n"
            "rim_bc = \"free\"\n";
        cacaosim_bench_cfg_t b1, b2;
        char                 berr[128] = "";
        char                 bfmt[4096];
        int                  rt_ok = 0;
        if (cacaosim_bench_parse_str(plate_ini, &b1, berr, sizeof(berr)) == 0 &&
            cacaosim_bench_format_str(&b1, bfmt, sizeof(bfmt)) > 0 &&
            cacaosim_bench_parse_str(bfmt, &b2, berr, sizeof(berr)) == 0)
        {
            const cacaosim_dm_cfg_t *d2 = &b2.nodes[0].dm;
            rt_ok = (d2->dyn_mode == 3 && fabsf(d2->f0_hz - 950.0f) < 1e-2f &&
                     fabsf(d2->wave_c_ms - 22.0f) < 1e-2f &&
                     fabsf(d2->bend_beta_m2s - 0.025f) < 1e-4f &&
                     fabsf(d2->damp_g0 - 150.0f) < 1e-2f &&
                     fabsf(d2->damp_g2 - 12.0f) < 1e-2f && d2->plate_drive == 1 &&
                     d2->rim_bc == 1);
        }
        check_cond(rt_ok, ".bench dyn_mode=plate parse/format round-trip",
                   rt_ok ? "plate_drive=force, rim_bc=free" : berr);

        cacaosim_dm_free(&pctx);
        free(hist);
        free(popd);
        free(pcmd);
    }

    cacaosim_dm_free(&ctx);
    free(opd_cpu);
    free(cmd);
}

/**
 * @brief Section 4: Validate 4-sided Pyramid WFS symmetry, slope linearity, gain & CUDA Graphs.
 *
 * Role:
 *   Verifies four physical and algorithmic properties of `cacaosim_pyrwfs_*`:
 *     1. Flat-wavefront total photon flux conservation (`100,000 ph`) and 4-quadrant symmetry
 *        (`quad_asym < 1e-5`).
 *     2. Small-signal tip slope linearity (`S(+0.2) / S(+0.1) == 2.0`) and antisymmetry
 *        (`S(-0.2) + S(+0.2) == 0`).
 *     3. Optical sensitivity gain boost in unmodulated (`mod = 0`) vs modulated (`mod = 3 l/D`)
 *        operation (`g_0 > 1.8 * g_3`).
 *     4. CPU vs CUDA Graph execution parity (`rel_L2 < 1e-5` after graph capture and replay).
 *
 * Rationale:
 *   Even a 0.5-pixel misalignment between the discrete FFT origin and the pyramid apex splits a
 *   flat-wavefront PSF unequally across quadrants (`asym >> 1e-2`), injecting a huge false tip-tilt
 *   offset in closed loop. Testing both modulation regimes confirms that `mod_rad_ld` properly
 *   trades sensitivity against linear range, while stepping the CUDA context twice verifies that
 *   cached `cudaGraphExec_t` replay produces identical pixels to the initial stream-capture pass.
 */
static void test_pyrwfs_physics(void)
{
    printf("\n=== 4. Pyramid WFS Optical & Detector Simulator ===\n");
    const uint32_t nwf  = 128;
    const uint32_t ncam = 64;
    float         *pup  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float         *wf   = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float         *cam0 = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
    float         *cam1 = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.25f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);

    cacaosim_pyrwfs_cfg_t wcfg = {
        .nwf        = nwf,
        .pad_factor = 2,
        .ncam       = ncam,
        .pup_diam   = 0.9375f,
        .lambda_um  = 0.75f,
        .mod_rad_ld = 3.0f,
        .mod_npts   = 24,
        .pup_sep    = 0.50f,
        .roof_ld    = 0.0f,
        .fstop_frac = 0.45f,
        .wf_unit    = CACAOSIM_WFUNIT_OPD_UM,
        .ref_lambda = 0.75f,
        .rng_seed   = 101ULL,
        .backend    = CACAOSIM_BACKEND_CPU,
        .det        = {
            .flux_ph = 100000.0f, .dark_e = 0.0f, .ron_e = 0.0f, .gain_eadu = 1.0f,
            .bias_adu = 0.0f, .sat_adu = 1e9f, .shutter = 1, .photon_noise = 0, .out_dtype = 1
        },
    };

    cacaosim_pyrwfs_ctx_t cpu_wfs;
    cacaosim_pyrwfs_init(&cpu_wfs, &wcfg, pup, NULL, NULL);
    cacaosim_pyrwfs_step(&cpu_wfs, pup, NULL, NULL, NULL, NULL, NULL, 0.0f, 3.0f, &wcfg.det, NULL,
                         cam0);

    double q00 = 0.0, q01 = 0.0, q10 = 0.0, q11 = 0.0;
    for (uint32_t y = 0; y < ncam; y++)
    {
        for (uint32_t x = 0; x < ncam; x++)
        {
            double v = cam0[y * ncam + x];
            if (y < ncam / 2)
            {
                if (x < ncam / 2)
                {
                    q00 += v;
                }
                else
                {
                    q01 += v;
                }
            }
            else
            {
                if (x < ncam / 2)
                {
                    q10 += v;
                }
                else
                {
                    q11 += v;
                }
            }
        }
    }
    double qtot = q00 + q01 + q10 + q11;
    double asym = (fabs(q00 - q01) + fabs(q00 - q10) + fabs(q00 - q11)) / qtot;
    char   buf[128];
    snprintf(buf, sizeof(buf), "total=%.1f ph, quad_asym=%.2e", qtot, asym);
    check_cond(fabs(qtot - 100000.0) < 5.0 && asym < 1e-5,
               "Flat-WF total flux & 4-quadrant symmetry", buf);

    /* Tip ramp slope linearity & unmodulated vs modulated sensitivity gain */
    float dpix = 0.9375f * (float) nwf;
    float p0   = 0.5f * (float) (nwf - 1u);
    float sx_mod[3];
    float alphas[3] = { -0.2f, 0.1f, 0.2f }; /* small-signal linear regime in lambda/D */

    for (int m = 0; m < 3; m++)
    {
        for (uint32_t y = 0; y < nwf; y++)
        {
            for (uint32_t x = 0; x < nwf; x++)
            {
                wf[y * nwf + x] = alphas[m] * 0.75f * (((float) x - p0) / dpix);
            }
        }
        cacaosim_pyrwfs_step(&cpu_wfs, pup, NULL, wf, NULL, NULL, NULL, 1.0f, 3.0f, &wcfg.det,
                             NULL, cam1);
        double left = 0.0, right = 0.0;
        for (uint32_t y = 0; y < ncam; y++)
        {
            for (uint32_t x = 0; x < ncam; x++)
            {
                if (x < ncam / 2)
                {
                    left += cam1[y * ncam + x];
                }
                else
                {
                    right += cam1[y * ncam + x];
                }
            }
        }
        sx_mod[m] = (float) ((right - left) / (right + left));
    }

    /* Small tip (0.15 lambda/D) sensitivity: unmodulated (mod=0) > modulated (mod=3) */
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            wf[y * nwf + x] = 0.15f * 0.75f * (((float) x - p0) / dpix);
        }
    }
    cacaosim_pyrwfs_step(&cpu_wfs, pup, NULL, wf, NULL, NULL, NULL, 1.0f, 3.0f, &wcfg.det, NULL,
                         cam0);
    cacaosim_pyrwfs_step(&cpu_wfs, pup, NULL, wf, NULL, NULL, NULL, 1.0f, 0.0f, &wcfg.det, NULL,
                         cam1);
    double l3 = 0.0, r3 = 0.0, l0 = 0.0, r0 = 0.0;
    for (uint32_t i = 0; i < ncam * ncam; i++)
    {
        if ((i % ncam) < ncam / 2)
        {
            l3 += cam0[i];
            l0 += cam1[i];
        }
        else
        {
            r3 += cam0[i];
            r0 += cam1[i];
        }
    }
    double g3 = fabs((r3 - l3) / (r3 + l3));
    double g0 = fabs((r0 - l0) / (r0 + l0));
    float  lin_ratio = fabsf(sx_mod[2] / sx_mod[1]);

    snprintf(buf, sizeof(buf), "S(+0.2)/S(+0.1)=%.3f, S(-0.2)+S(+0.2)=%.2e, g0/g3=%.2f",
             lin_ratio, sx_mod[0] + sx_mod[2], g0 / g3);
    check_cond(fabsf(lin_ratio - 2.0f) < 0.08f && fabsf(sx_mod[0] + sx_mod[2]) < 1e-5f &&
                   g0 > 1.8 * g3,
               "PyWFS linear tip response & modulation gain", buf);

    if (cacaosim_cuda_available())
    {
        cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.08f, 99ULL, wf);
        cacaosim_pyrwfs_step(&cpu_wfs, pup, NULL, wf, NULL, NULL, NULL, 1.0f, 3.0f, &wcfg.det,
                             NULL, cam0);

        cacaosim_pyrwfs_cfg_t gcfg = wcfg;
        gcfg.backend               = CACAOSIM_BACKEND_CUDA;
        cacaosim_pyrwfs_ctx_t gpu_wfs;
        cacaosim_pyrwfs_init(&gpu_wfs, &gcfg, pup, NULL, NULL);
        /* Run twice to exercise both warmup/capture and CUDA Graph replay! */
        cacaosim_pyrwfs_step(&gpu_wfs, pup, NULL, wf, NULL, NULL, NULL, 1.0f, 3.0f, &gcfg.det,
                             NULL, cam1);
        cacaosim_pyrwfs_step(&gpu_wfs, pup, NULL, wf, NULL, NULL, NULL, 1.0f, 3.0f, &gcfg.det,
                             NULL, cam1);

        double n_diff = 0.0, n_ref = 0.0;
        for (uint32_t i = 0; i < ncam * ncam; i++)
        {
            double d = (double) cam0[i] - (double) cam1[i];
            n_diff += d * d;
            n_ref += (double) cam0[i] * (double) cam0[i];
        }
        double rel_l2 = sqrt(n_diff / n_ref);
        snprintf(buf, sizeof(buf), "rel_L2(CPU, CUDA+Graph) = %.3e", rel_l2);
        check_cond(gpu_wfs.use_cuda && rel_l2 < 1e-5, "PyWFS CPU vs CUDA Graph parity", buf);
        cacaosim_pyrwfs_free(&gpu_wfs);
    }

    cacaosim_pyrwfs_free(&cpu_wfs);
    free(cam1);
    free(cam0);
    free(wf);
    free(pup);
}

/**
 * @brief Section 5: Validate focal-plane camera MFT vs FFT, Strehl, Maréchal & broadband PSFs.
 *
 * Role:
 *   Validates four optical invariants of `cacaosim_fpcam_*`:
 *     1. Exact numerical equivalence (`max_diff < 1e-5`) between Soummer Matrix-DFT and 2x-padded
 *        FFT at matched focal sampling (`pixscale_ld = pad_factor / pup_diam = 2.5`), with exact
 *        diffraction-limited flat-wavefront `Strehl = 1.000000`.
 *     2. Absolute on-axis Strehl calibration against Maréchal's extended approximation
 *        `S ~ exp(-sigma_phi^2) = 0.8847` for `sigma_phi = 0.35 rad` RMS NCPA at `1.65 um`.
 *     3. Polychromatic radial speckle dispersion (`20%` bandwidth across `5` spectral channels)
 *        attenuating a `10`-cycle/pupil cosine satellite speckle peak by `> 1.35x` relative to
 *        monochromatic illumination.
 *     4. Pointwise parity (`< 1e-5`) between CPU OpenMP/CBLAS MFT and GPU strided-batched cuBLAS.
 *
 * Rationale:
 *   Science PSF Strehl telemetry and post-coronagraphic speckle morphology require exact energy
 *   and spatial-scale consistency between MFT and FFT propagators, accurate `OPD -> rad` phase
 *   conversion (`2 * pi / lambda_c`), and physical `1 / lambda_c` angular scaling per channel.
 */
static void test_fpcam_physics(void)
{
    printf("\n=== 5. Focal Plane Camera (Soummer MFT, FFT, Broadband, Marechal) ===\n");
    const uint32_t nwf  = 128;
    const uint32_t ncam = 128;
    float         *pup  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float         *wf   = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float         *psf0 = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
    float         *psf1 = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.80f, .cobs = 0.20f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 0.0f
    };
    cacaosim_make_pupil(&pcfg, pup);

    /* Match MFT pixscale_ld to 2x-padded FFT (pad_factor / pup_diam = 2.0 / 0.80 = 2.5) */
    cacaosim_fpcam_cfg_t mcfg = {
        .nwf         = nwf,
        .ncam        = ncam,
        .prop_mode   = CACAOSIM_FPCAM_PROP_MFT,
        .pad_factor  = 2,
        .pup_diam    = 0.80f,
        .lambda_um   = 1.65f,
        .bw_frac     = 0.0f,
        .nlambda     = 1,
        .pixscale_ld = 2.5f,
        .wf_unit     = CACAOSIM_WFUNIT_OPD_UM,
        .ref_lambda  = 1.65f,
        .rng_seed    = 1ULL,
        .backend     = CACAOSIM_BACKEND_CPU,
        .det         = {
            .flux_ph = -1.0f, .dark_e = 0.0f, .ron_e = 0.0f, .gain_eadu = 1.0f,
            .bias_adu = 0.0f, .sat_adu = 1e9f, .shutter = 1, .photon_noise = 0, .out_dtype = 1
        },
    };
    cacaosim_fpcam_cfg_t fcfg = mcfg;
    fcfg.prop_mode            = CACAOSIM_FPCAM_PROP_FFT;

    cacaosim_fpcam_ctx_t mft_ctx, fft_ctx;
    cacaosim_fpcam_init(&mft_ctx, &mcfg, pup, NULL, NULL);
    cacaosim_fpcam_init(&fft_ctx, &fcfg, pup, NULL, NULL);

    cacaosim_fpcam_step(&mft_ctx, pup, NULL, NULL, NULL, NULL, NULL, 0.0f, &mcfg.det, NULL, psf0,
                        NULL);
    cacaosim_fpcam_step(&fft_ctx, pup, NULL, NULL, NULL, NULL, NULL, 0.0f, &fcfg.det, NULL, psf1,
                        NULL);

    double max_mft_fft = 0.0;
    for (uint32_t i = 0; i < ncam * ncam; i++)
    {
        double d = fabs((double) psf0[i] - (double) psf1[i]);
        if (d > max_mft_fft)
        {
            max_mft_fft = d;
        }
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "Strehl_MFT=%.6f, Strehl_FFT=%.6f, max_diff=%.2e",
             mft_ctx.last_strehl_peak, fft_ctx.last_strehl_peak, max_mft_fft);
    check_cond(fabsf(mft_ctx.last_strehl_peak - 1.0f) < 1e-5f &&
                   fabsf(fft_ctx.last_strehl_peak - 1.0f) < 1e-5f && max_mft_fft < 1e-5,
               "Flat-WF Strehl=1.000000 & MFT == FFT exactness", buf);

    /* Marechal Strehl test: sigma_phi = 0.35 rad -> S ~ exp(-0.1225) = 0.8847 */
    const float sigma_rad = 0.35f;
    const float rms_um    = sigma_rad * (1.65f / CACAOSIM_TWO_PI_F);
    cacaosim_make_ncpa(nwf, 0.80f, pup, rms_um, 1234ULL, wf);
    cacaosim_fpcam_step(&mft_ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &mcfg.det, NULL, psf0,
                        NULL);
    float s_marechal = expf(-sigma_rad * sigma_rad);
    snprintf(buf, sizeof(buf), "Strehl_meas=%.4f vs exp(-sigma^2)=%.4f", mft_ctx.last_strehl_axis,
             s_marechal);
    check_cond(fabsf(mft_ctx.last_strehl_axis - s_marechal) < 0.012f,
               "Marechal Strehl ratio exp(-sigma_phi^2)", buf);

    /* Broadband 20% 5-channel MFT CPU vs CUDA parity & chromatic speckle broadening */
    cacaosim_fpcam_cfg_t bcfg = mcfg;
    bcfg.bw_frac              = 0.20f;
    bcfg.nlambda              = 5;
    bcfg.pixscale_ld          = 3.0f;
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            /* 10 cycles/pupil cosine ripple puts satellite speckles at +/- 10 lambda/D */
            float xt        = ((float) x - 63.5f) / (0.80f * (float) nwf);
            wf[y * nwf + x] = 0.08f * cosf(CACAOSIM_TWO_PI_F * 10.0f * xt);
        }
    }

    cacaosim_fpcam_ctx_t mono_ctx, poly_cpu;
    cacaosim_fpcam_cfg_t m3cfg = mcfg;
    m3cfg.pixscale_ld          = 3.0f;
    cacaosim_fpcam_init(&mono_ctx, &m3cfg, pup, NULL, NULL);
    cacaosim_fpcam_init(&poly_cpu, &bcfg, pup, NULL, NULL);

    cacaosim_fpcam_step(&mono_ctx, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &m3cfg.det, NULL, psf0,
                        NULL);
    cacaosim_fpcam_step(&poly_cpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &bcfg.det, NULL, psf1,
                        NULL);

    /* Satellite peak at (y=64, x=64+30=94) drops in peak height when dispersed over 20% BW */
    float spk_mono = psf0[64 * ncam + 94];
    float spk_poly = psf1[64 * ncam + 94];
    snprintf(buf, sizeof(buf), "speckle_peak mono=%.4e -> 20%%_BW=%.4e", spk_mono, spk_poly);
    check_cond(spk_mono > 1.35f * spk_poly && spk_poly > 0.0f,
               "Broadband chromatic speckle radial dispersion", buf);

    if (cacaosim_cuda_available())
    {
        cacaosim_fpcam_cfg_t gcfg = bcfg;
        gcfg.backend              = CACAOSIM_BACKEND_CUDA;
        cacaosim_fpcam_ctx_t poly_gpu;
        cacaosim_fpcam_init(&poly_gpu, &gcfg, pup, NULL, NULL);
        cacaosim_fpcam_step(&poly_gpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &gcfg.det, NULL,
                            psf0, NULL);

        double max_d = 0.0;
        for (uint32_t i = 0; i < ncam * ncam; i++)
        {
            double d = fabs((double) psf0[i] - (double) psf1[i]);
            if (d > max_d)
            {
                max_d = d;
            }
        }
        snprintf(buf, sizeof(buf), "max |CPU - cuBLAS_Batched| = %.3e", max_d);
        check_cond(poly_gpu.use_cuda && max_d < 1e-5,
                   "FPcam 5-lambda MFT CPU vs CUDA parity", buf);
        cacaosim_fpcam_free(&poly_gpu);
    }

    cacaosim_fpcam_free(&poly_cpu);
    cacaosim_fpcam_free(&mono_ctx);
    cacaosim_fpcam_free(&fft_ctx);
    cacaosim_fpcam_free(&mft_ctx);
    free(psf1);
    free(psf0);
    free(wf);
    free(pup);
}

/**
 * @brief Section 6: Validate end-to-end single-stage closed-loop AO calibration and control.
 *
 * Role:
 *   Couples `cacaosim_dm` (`8x8`), `cacaosim_pyrwfs` (`32x32`, `3 lambda/D` modulation), and
 *   `cacaosim_fpcam` (`64x64` H-band) through a complete AO calibration + real-time control cycle:
 *     1. Push-pull (`+/- 0.02 um`) interaction matrix `I` measurement in noiseless static mode.
 *     2. `2%`-trace Tikhonov-regularized Gauss-Jordan reconstructor
 *        `C = (I^T I + reg * Id)^(-1) I^T`.
 *     3. `120`-frame closed-loop leaky-integrator (`g = 0.35`, leak `0.998`, piston projection)
 *        under `1.0`-frame hardware delay, `100 us` ZOH lag, photon shot noise, and read noise on
 *        a drifting `0.35 um` RMS aberration.
 *
 * Rationale:
 *   Component-level tests cannot detect subtle coordinate-transpose, sign-convention, or scaling
 *   mismatches between DM commands and WFS intensities. Demonstrating H-band Strehl improvement
 *   from degraded open-loop (`< 0.40`) to diffraction-limited closed-loop (`> 0.85`) proves full
 *   physical closed-loop compatibility with `cacao`-style linear matrix reconstruction.
 */
static void test_closed_loop_ao(void)
{
    printf("\n=== 6. Closed-Loop AO Calibration & Turbulence Rejection ===\n");
    const uint32_t nwf   = 64;
    const uint32_t nact  = 8;
    const uint32_t nacts = nact * nact;
    const uint32_t nwfs  = 32;
    const uint32_t npwfs = nwfs * nwfs;
    const uint32_t nfp   = 64;

    float *pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.90f, .cobs = 0.20f, .nspider = 0, .spider_w = 0.0f,
        .spider_deg = 0.0f
    };
    cacaosim_make_pupil(&pcfg, pup);

    cacaosim_dm_cfg_t dcfg = {
        .nact_x = nact, .nact_y = nact, .nwf = nwf, .dm_diam = 0.90f, .if_coupling = 0.22f,
        .refl_scale = 2.0f, .max_stroke = 5.0f, .dac_bits = 16, .nonlin_quad = 0.0f,
        .dyn_mode = 1, .dt_sec = 0.0005f, .tau_sec = 0.0001f, .f0_hz = 2500.0f, .zeta = 0.7f,
        .delay_frames = 1.0f, .backend = CACAOSIM_BACKEND_AUTO
    };
    cacaosim_pyrwfs_cfg_t wcfg = {
        .nwf = nwf, .pad_factor = 2, .ncam = nwfs, .pup_diam = 0.90f, .lambda_um = 0.80f,
        .mod_rad_ld = 3.0f, .mod_npts = 16, .pup_sep = 0.50f, .roof_ld = 0.0f, .fstop_frac = 0.45f,
        .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 0.80f, .rng_seed = 42ULL,
        .backend = CACAOSIM_BACKEND_AUTO,
        .det = { .flux_ph = 200000.0f, .dark_e = 0.0f, .ron_e = 0.2f, .gain_eadu = 1.0f,
                 .bias_adu = 0.0f, .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1,
                 .out_dtype = 1 }
    };
    cacaosim_fpcam_cfg_t fcfg = {
        .nwf = nwf, .ncam = nfp, .prop_mode = CACAOSIM_FPCAM_PROP_MFT, .pad_factor = 2,
        .pup_diam = 0.90f, .lambda_um = 1.65f, .bw_frac = 0.0f, .nlambda = 1, .pixscale_ld = 3.0f,
        .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 1.65f, .rng_seed = 99ULL,
        .backend = CACAOSIM_BACKEND_AUTO,
        .det = { .flux_ph = 500000.0f, .dark_e = 0.0f, .ron_e = 0.5f, .gain_eadu = 1.0f,
                 .bias_adu = 0.0f, .sat_adu = 65535.0f, .shutter = 1, .photon_noise = 1,
                 .out_dtype = 1 }
    };

    cacaosim_dm_ctx_t     dm;
    cacaosim_pyrwfs_ctx_t wfs;
    cacaosim_fpcam_ctx_t  fp;
    cacaosim_dm_init(&dm, &dcfg, NULL);
    cacaosim_pyrwfs_init(&wfs, &wcfg, pup, NULL, dm.cuda_stream);
    cacaosim_fpcam_init(&fp, &fcfg, pup, NULL, dm.cuda_stream);

    float *cmd     = (float *) cacaosim_aligned_calloc(nacts, sizeof(float));
    float *dm_opd  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *wfs_ref = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    float *wfs_p   = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    float *wfs_m   = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    float *imat    = (float *) cacaosim_aligned_calloc((size_t) npwfs * nacts, sizeof(float));
    float *ata     = (float *) cacaosim_aligned_calloc((size_t) nacts * nacts, sizeof(float));
    float *cmat    = (float *) cacaosim_aligned_calloc((size_t) nacts * npwfs, sizeof(float));

    /* Push-pull calibration with noiseless instantaneous DM */
    cacaosim_det_cfg_t cal_det = wcfg.det;
    cal_det.photon_noise       = 0;
    cal_det.ron_e              = 0.0f;
    dm.cfg.dyn_mode            = 0;

    cacaosim_dm_reset(&dm);
    cacaosim_dm_step(&dm, cmd, NULL, 0.0f, 0.0f, 5.0f, dm_opd);
    cacaosim_pyrwfs_step(&wfs, pup, NULL, NULL, dm.use_cuda ? NULL : dm_opd, NULL,
                         dm.use_cuda ? dm.d_opd_out : NULL, 0.0f, 3.0f, &cal_det, NULL, wfs_ref);

    const float poke = 0.02f;
    for (uint32_t a = 0; a < nacts; a++)
    {
        __builtin_memset(cmd, 0, nacts * sizeof(float));
        cmd[a] = poke;
        cacaosim_dm_step(&dm, cmd, NULL, 0.0f, 0.0f, 5.0f, dm_opd);
        cacaosim_pyrwfs_step(&wfs, pup, NULL, NULL, dm.use_cuda ? NULL : dm_opd, NULL,
                             dm.use_cuda ? dm.d_opd_out : NULL, 0.0f, 3.0f, &cal_det, NULL, wfs_p);

        cmd[a] = -poke;
        cacaosim_dm_step(&dm, cmd, NULL, 0.0f, 0.0f, 5.0f, dm_opd);
        cacaosim_pyrwfs_step(&wfs, pup, NULL, NULL, dm.use_cuda ? NULL : dm_opd, NULL,
                             dm.use_cuda ? dm.d_opd_out : NULL, 0.0f, 3.0f, &cal_det, NULL, wfs_m);

        for (uint32_t p = 0; p < npwfs; p++)
        {
            imat[(size_t) p * nacts + a] = (wfs_p[p] - wfs_m[p]) / (2.0f * poke);
        }
    }

    /* Tikhonov regularized reconstructor: C = (I^T I + reg * Id)^(-1) I^T via Gauss-Jordan */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int) nacts, (int) nacts, (int) npwfs,
                1.0f, imat, (int) nacts, imat, (int) nacts, 0.0f, ata, (int) nacts);
    float trace = 0.0f;
    for (uint32_t a = 0; a < nacts; a++)
    {
        trace += ata[(size_t) a * nacts + a];
    }
    float reg = 0.02f * (trace / (float) nacts);
    for (uint32_t a = 0; a < nacts; a++)
    {
        ata[(size_t) a * nacts + a] += reg;
    }

    float *inv = (float *) cacaosim_aligned_calloc((size_t) nacts * nacts, sizeof(float));
    for (uint32_t a = 0; a < nacts; a++)
    {
        inv[(size_t) a * nacts + a] = 1.0f;
    }
    for (uint32_t col = 0; col < nacts; col++)
    {
        float piv = ata[(size_t) col * nacts + col];
        for (uint32_t j = 0; j < nacts; j++)
        {
            ata[(size_t) col * nacts + j] /= piv;
            inv[(size_t) col * nacts + j] /= piv;
        }
        for (uint32_t row = 0; row < nacts; row++)
        {
            if (row == col)
            {
                continue;
            }
            float f = ata[(size_t) row * nacts + col];
            for (uint32_t j = 0; j < nacts; j++)
            {
                ata[(size_t) row * nacts + j] -= f * ata[(size_t) col * nacts + j];
                inv[(size_t) row * nacts + j] -= f * inv[(size_t) col * nacts + j];
            }
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) nacts, (int) npwfs, (int) nacts,
                1.0f, inv, (int) nacts, imat, (int) nacts, 0.0f, cmat, (int) npwfs);

    /* Create 0.35 um RMS evolving low-order aberration & compare open vs closed loop Strehl */
    float *turb_base = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *turb_cur  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *fp_img    = (float *) cacaosim_aligned_calloc((size_t) nfp * nfp, sizeof(float));
    float *ds        = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    float *dcmd      = (float *) cacaosim_aligned_calloc(nacts, sizeof(float));
    cacaosim_make_ncpa(nwf, 0.90f, pup, 0.35f, 2026ULL, turb_base);

    /* Open-loop Strehl */
    cacaosim_fpcam_reset_long(&fp);
    cacaosim_fpcam_step(&fp, pup, NULL, turb_base, NULL, NULL, NULL, 1.0f, &fcfg.det, NULL, fp_img,
                        NULL);
    float strehl_open = fp.last_strehl_peak;

    /* Closed-loop with realistic 1-frame hardware delay + 100 us RC + photon & readout noise */
    dm.cfg.dyn_mode = 1;
    cacaosim_dm_reset(&dm);
    __builtin_memset(cmd, 0, nacts * sizeof(float));
    cacaosim_fpcam_reset_long(&fp);

    for (int iter = 0; iter < 120; iter++)
    {
        float drift = 1.0f + 0.05f * sinf(0.04f * (float) iter);
        for (uint32_t i = 0; i < nwf * nwf; i++)
        {
            turb_cur[i] = drift * turb_base[i];
        }

        cacaosim_dm_step(&dm, cmd, NULL, 1.0f, 0.0001f, 5.0f, dm_opd);
        cacaosim_pyrwfs_step(&wfs, pup, NULL, turb_cur, dm.use_cuda ? NULL : dm_opd, NULL,
                             dm.use_cuda ? dm.d_opd_out : NULL, 1.0f, 3.0f, &wcfg.det, NULL, wfs_p);

        for (uint32_t p = 0; p < npwfs; p++)
        {
            ds[p] = wfs_p[p] - wfs_ref[p];
        }
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) nacts, (int) npwfs, 1.0f, cmat, (int) npwfs,
                    ds, 1, 0.0f, dcmd, 1);

        float mean_c = 0.0f;
        for (uint32_t a = 0; a < nacts; a++)
        {
            cmd[a] = 0.998f * cmd[a] - 0.35f * dcmd[a];
            mean_c += cmd[a];
        }
        mean_c /= (float) nacts;
        for (uint32_t a = 0; a < nacts; a++)
        {
            cmd[a] -= mean_c;
        }

        if (iter >= 40)
        {
            cacaosim_fpcam_step(&fp, pup, NULL, turb_cur, dm.use_cuda ? NULL : dm_opd, NULL,
                                dm.use_cuda ? dm.d_opd_out : NULL, 1.0f, &fcfg.det, NULL, fp_img,
                                NULL);
        }
    }

    float strehl_closed = fp.long_strehl_peak;
    char  buf[128];
    snprintf(buf, sizeof(buf), "Strehl open=%.3f -> closed=%.3f (1-frame delay + noise)",
             strehl_open, strehl_closed);
    check_cond(strehl_open < 0.40f && strehl_closed > 0.85f,
               "Closed-loop AO calibration & wavefront control", buf);

    cacaosim_fpcam_free(&fp);
    cacaosim_pyrwfs_free(&wfs);
    cacaosim_dm_free(&dm);
    free(dcmd);
    free(ds);
    free(fp_img);
    free(turb_cur);
    free(turb_base);
    free(inv);
    free(cmat);
    free(ata);
    free(imat);
    free(wfs_m);
    free(wfs_p);
    free(wfs_ref);
    free(dm_opd);
    free(cmd);
    free(pup);
}

/**
 * @brief Compute scale-invariant relative Euclidean (`L_2`) error between two float32 buffers.
 *
 * Role:
 *   Evaluates `||a - b||_2 / ||a||_2` (or `||a - b||_2` when `||a||_2 == 0`) over `n` elements
 *   using `double` accumulators.
 *
 * Rationale:
 *   Single-precision CPU (`FFTW3f` / `cblas_cgemm`) and GPU (`cuFFT` / `cublasCgemm`) reductions
 *   differ slightly in summation order (`~1e-7..1e-5` relative error). Normalizing by `||a||_2` in
 *   `double` provides a flux-invariant parity metric across unit-amplitude pupil masks and `10^5`-
 *   photon WFS detector frames alike without accumulator rounding drift.
 *
 * @param[in] a  Reference `float` array (typically CPU golden output) of length `n`.
 * @param[in] b  Candidate `float` array (typically CUDA output) of length `n`.
 * @param[in] n  Number of elements to compare.
 * @return Relative `L_2` difference `sqrt(sum((a - b)^2) / sum(a^2))`.
 */
static double rel_l2_f32(const float *a, const float *b, uint64_t n)
{
    double num = 0.0;
    double den = 0.0;
    for (uint64_t i = 0; i < n; i++)
    {
        double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) a[i] * (double) a[i];
    }
    return (den > 0.0) ? sqrt(num / den) : sqrt(num);
}

/**
 * @brief Section 7: Validate stellar photometry, dichroic splitters, 2-axis TTM & 2D registration.
 *
 * Role:
 *   Verifies four common-optics and beam-routing subsystems:
 *     1. 32-bin (`0.40..2.40 um`) blackbody/Vega photometry (`cacaosim_spec_star()`): exact `100x`
 *        photon rate ratio across a `5.0`-magnitude step (`m_V = 8` vs `m_V = 13`).
 *     2. Spectral photon conservation (`R + T <= 1`, `R, T >= 0`) across gray, LP, SP, and J-notch
 *        band-reflect (`dichroic_bp` at `1.22 um`) splitters (`> 85%` J refl, `> 90%` V & H trans).
 *     3. 2-axis Tip-Tilt mirror (`dm_mode = 1`): exact planar `X/Y` OPD slopes `2 * stroke / R`.
 *     4. 2D affine pupil relay (`cacaosim_beam_relay_*`): exact sub-pixel translation shift and
 *        rotated/scaled bilinear interpolation parity (`rel_L2 < 1e-5`) between CPU and CUDA.
 *
 * Rationale:
 *   Multi-arm AO benches rely on radiometrically consistent dichroic photon budgets across Visible,
 *   J, and H bands to model realistic WFS SNR limits, and require accurate sub-pixel shift,
 *   rotation, and magnification relays to simulate DM-to-WFS misregistration effects.
 */
static void test_beam_photometry_and_ttm(void)
{
    printf("\n=== 7. Spectral Photometry, Beamsplitters, TTM & Registration ===\n");
    char buf[128];

    cacaosim_spec_t s8, s13;
    cacaosim_spec_star(&s8, 8.2f, 0.30f, 8.0f, 5778.0f, 0.75f);
    cacaosim_spec_star(&s13, 8.2f, 0.30f, 13.0f, 5778.0f, 0.75f);
    double tot8  = cacaosim_spec_total(&s8);
    double tot13 = cacaosim_spec_total(&s13);
    double ratio = tot8 / tot13;
    snprintf(buf, sizeof(buf), "mV=8 (%.2e ph/s) / mV=13 = %.5f", tot8, ratio);
    check_cond(tot8 > 1e8 && fabs(ratio - 100.0) < 1e-3,
               "Vega 5-mag step equals exact 100x photon rate", buf);

    cacaosim_split_cfg_t modes[4] = {
        { .mode = CACAOSIM_SPLIT_GRAY, .r_gray = 0.35f, .t_gray = 0.62f },
        { .mode = CACAOSIM_SPLIT_DICHROIC_LP, .lambda_cut = 1.00f, .roll_um = 0.03f,
          .eff_refl = 0.95f, .eff_trans = 0.95f },
        { .mode = CACAOSIM_SPLIT_DICHROIC_SP, .lambda_cut = 1.00f, .roll_um = 0.03f,
          .eff_refl = 0.95f, .eff_trans = 0.95f },
        { .mode = CACAOSIM_SPLIT_DICHROIC_BP, .lambda_cut = 1.22f, .band_width = 0.30f,
          .roll_um = 0.03f, .eff_refl = 0.95f, .eff_trans = 0.95f },
    };
    int conserve_ok = 1;
    for (int m = 0; m < 4; m++)
    {
        cacaosim_spec_t sr, st;
        cacaosim_spec_split(&s8, &modes[m], &sr, &st);
        for (int b = 0; b < CACAOSIM_NSPEC; b++)
        {
            if (sr.ph_s[b] < -1e-6f || st.ph_s[b] < -1e-6f ||
                (sr.ph_s[b] + st.ph_s[b]) > s8.ph_s[b] * 1.00001f)
            {
                conserve_ok = 0;
            }
        }
    }

    cacaosim_spec_t bp_r, bp_t;
    cacaosim_spec_split(&s8, &modes[3], &bp_r, &bp_t);
    float j_in    = cacaosim_spec_band_flux(&s8, 1.22f, 0.16f, 0.001f, 1.0f, NULL);
    float j_refl  = cacaosim_spec_band_flux(&bp_r, 1.22f, 0.16f, 0.001f, 1.0f, NULL);
    float v_in    = cacaosim_spec_band_flux(&s8, 0.75f, 0.20f, 0.001f, 1.0f, NULL);
    float v_trans = cacaosim_spec_band_flux(&bp_t, 0.75f, 0.20f, 0.001f, 1.0f, NULL);
    float h_in    = cacaosim_spec_band_flux(&s8, 1.65f, 0.18f, 0.001f, 1.0f, NULL);
    float h_trans = cacaosim_spec_band_flux(&bp_t, 1.65f, 0.18f, 0.001f, 1.0f, NULL);
    snprintf(buf, sizeof(buf), "J_refl=%.1f%%, V_trans=%.1f%%, H_trans=%.1f%%",
             100.0f * j_refl / j_in, 100.0f * v_trans / v_in, 100.0f * h_trans / h_in);
    check_cond(conserve_ok && (j_refl / j_in) > 0.85f && (v_trans / v_in) > 0.90f &&
                   (h_trans / h_in) > 0.90f,
               "Splitter photon conservation & J-notch dichroic", buf);

    /* 2-axis Tip-Tilt mirror (dm_mode = 1) */
    const uint32_t    nwf  = 128;
    cacaosim_dm_cfg_t tcfg = {
        .nact_x = 2, .nact_y = 1, .nwf = nwf, .dm_diam = 0.9375f, .if_coupling = 0.20f,
        .refl_scale = 2.0f, .max_stroke = 10.0f, .dac_bits = 0, .dyn_mode = 0, .dt_sec = 0.0005f,
        .dm_mode = 1, .backend = CACAOSIM_BACKEND_CPU
    };
    cacaosim_dm_ctx_t ttm;
    cacaosim_dm_init(&ttm, &tcfg, NULL);
    float  tt_cmd[2] = { 0.40f, -0.25f };
    float *tt_opd    = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    cacaosim_dm_step(&ttm, tt_cmd, NULL, 0.0f, 0.0f, 10.0f, tt_opd);

    float r_pup  = 0.5f * 0.9375f * (float) nwf;
    float dx_opd = tt_opd[64 * nwf + 94] - tt_opd[64 * nwf + 34];
    float dy_opd = tt_opd[94 * nwf + 64] - tt_opd[34 * nwf + 64];
    float ref_dx = 2.0f * 0.40f * (60.0f / r_pup);
    float ref_dy = 2.0f * (-0.25f) * (60.0f / r_pup);
    snprintf(buf, sizeof(buf), "dOPD_x=%.5f vs %.5f, dOPD_y=%.5f vs %.5f", dx_opd, ref_dx, dy_opd,
             ref_dy);
    check_cond(fabsf(dx_opd - ref_dx) < 1e-5f && fabsf(dy_opd - ref_dy) < 1e-5f,
               "2-axis Tip-Tilt mirror linear X/Y OPD plane", buf);

    /* 2D beam relay registration + CUDA parity */
    float *in_opd  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *in_amp  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *out_opd = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *out_amp = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            float dx             = (float) x - 60.0f;
            float dy             = (float) y - 66.0f;
            in_opd[y * nwf + x]  = expf(-0.08f * (dx * dx + dy * dy));
            in_amp[y * nwf + x]  = 1.0f;
        }
    }
    cacaosim_reg_t reg = { .shift_x = 4.0f, .shift_y = -3.0f, .rot_deg = 0.0f, .scale = 1.0f };
    cacaosim_beam_relay_cpu(in_opd, in_amp, NULL, NULL, &reg, nwf, out_opd, out_amp);
    float shifted_peak = out_opd[63 * nwf + 64];
    snprintf(buf, sizeof(buf), "peak@(64,63)=%.6f after (+4,-3) shift", shifted_peak);
    check_cond(fabsf(shifted_peak - 1.0f) < 1e-5f, "2D pupil registration translation relay", buf);

    if (cacaosim_cuda_available())
    {
        cacaosim_reg_t rot_reg = {
            .shift_x = 2.25f, .shift_y = -1.5f, .rot_deg = 12.0f, .scale = 0.98f
        };
        cacaosim_beam_relay_cpu(tt_opd, in_amp, in_opd, NULL, &rot_reg, nwf, out_opd, out_amp);

        uint64_t np      = (uint64_t) nwf * nwf;
        float   *d_iopd  = cacaosim_cuda_alloc_f32(np);
        float   *d_iamp  = cacaosim_cuda_alloc_f32(np);
        float   *d_ncpa  = cacaosim_cuda_alloc_f32(np);
        float   *d_oopd  = cacaosim_cuda_alloc_f32(np);
        float   *d_oamp  = cacaosim_cuda_alloc_f32(np);
        float   *gpu_opd = (float *) cacaosim_aligned_calloc(np, sizeof(float));
        cacaosim_cuda_h2d_f32(d_iopd, tt_opd, np, NULL);
        cacaosim_cuda_h2d_f32(d_iamp, in_amp, np, NULL);
        cacaosim_cuda_h2d_f32(d_ncpa, in_opd, np, NULL);
        cacaosim_cuda_beam_relay(d_iopd, d_iamp, d_ncpa, NULL, &rot_reg, nwf, d_oopd, d_oamp, NULL);
        cacaosim_cuda_d2h_f32(gpu_opd, d_oopd, np, NULL);
        cacaosim_cuda_stream_sync(NULL);

        double err = rel_l2_f32(out_opd, gpu_opd, np);
        snprintf(buf, sizeof(buf), "rel_L2(CPU, CUDA) = %.3e", err);
        check_cond(err < 1e-5, "Beam affine relay CPU vs CUDA parity", buf);

        free(gpu_opd);
        cacaosim_cuda_free_f32(d_oamp);
        cacaosim_cuda_free_f32(d_oopd);
        cacaosim_cuda_free_f32(d_ncpa);
        cacaosim_cuda_free_f32(d_iamp);
        cacaosim_cuda_free_f32(d_iopd);
    }

    free(out_amp);
    free(out_opd);
    free(in_amp);
    free(in_opd);
    free(tt_opd);
    cacaosim_dm_free(&ttm);
}

/**
 * @brief Compute mean subaperture Center-of-Gravity (CoG) spot offset on a Shack-Hartmann frame.
 *
 * Role:
 *   Iterates over an `n_sub x n_sub` grid of `pps x pps`-pixel subapertures, selects well-lit
 *   lenslets exceeding `65%` of the brightest subaperture's integrated flux, and averages their
 *   intensity-weighted `(x, y)` centroids relative to geometric subaperture center `(pps - 1) / 2`.
 *
 * Rationale:
 *   Obstructed annular pupils with spiders partially vignette perimeter, central, and vane-crossing
 *   lenslets, skewing their diffraction spots radially/azimuthally even on a flat wavefront. Flux-
 *   gating at `0.65 * max_sf` isolates unobstructed full-flux lenslets so optical centering and
 *   tilt linearity can be verified to `< 1e-4` pixel precision.
 *
 * @param[in]  img     Full Shack-Hartmann detector image of size `(n_sub * pps)^2`.
 * @param[in]  n_sub   Number of subapertures along each pupil axis.
 * @param[in]  pps     Detector pixels per subaperture axis.
 * @param[out] out_cx  Mean X spot centroid offset [pixels] across illuminated subapertures.
 * @param[out] out_cy  Mean Y spot centroid offset [pixels] across illuminated subapertures.
 */
static void shwfs_mean_cog(
    const float *img,
    uint32_t     n_sub,
    uint32_t     pps,
    double      *out_cx,
    double      *out_cy)
{
    uint32_t ncam   = n_sub * pps;
    double   max_sf = 0.0;
    for (uint32_t sy = 0; sy < n_sub; sy++)
    {
        for (uint32_t sx = 0; sx < n_sub; sx++)
        {
            double sf = 0.0;
            for (uint32_t py = 0; py < pps; py++)
            {
                for (uint32_t px = 0; px < pps; px++)
                {
                    sf += img[(sy * pps + py) * ncam + (sx * pps + px)];
                }
            }
            if (sf > max_sf)
            {
                max_sf = sf;
            }
        }
    }

    double sum_cx = 0.0;
    double sum_cy = 0.0;
    int    n_val  = 0;
    double c0     = 0.5 * (double) (pps - 1);
    for (uint32_t sy = 0; sy < n_sub; sy++)
    {
        for (uint32_t sx = 0; sx < n_sub; sx++)
        {
            double sf = 0.0;
            double mx = 0.0;
            double my = 0.0;
            for (uint32_t py = 0; py < pps; py++)
            {
                for (uint32_t px = 0; px < pps; px++)
                {
                    double v = img[(sy * pps + py) * ncam + (sx * pps + px)];
                    sf += v;
                    mx += v * ((double) px - c0);
                    my += v * ((double) py - c0);
                }
            }
            if (sf > 0.65 * max_sf)
            {
                sum_cx += mx / sf;
                sum_cy += my / sf;
                n_val++;
            }
        }
    }
    *out_cx = (n_val > 0) ? (sum_cx / (double) n_val) : 0.0;
    *out_cy = (n_val > 0) ? (sum_cy / (double) n_val) : 0.0;
}

/**
 * @brief Section 8: Validate diffractive Shack-Hartmann WFS centering, tilt linearity & GPU parity.
 *
 * Role:
 *   Validates `cacaosim_shwfs_*` (`16x16` lenslets, `8x8` px/subap, `2x` FFT padding) across:
 *     1. Flat-wavefront total photon conservation (`100,000 ph`) and sub-pixel spot centering
 *        symmetry (`|<cx>|, |<cy>| < 1e-4 px`).
 *     2. Pure X-tilt spot displacement `2:1` linearity (`cx(2x) / cx(1x) = 2.0 +/- 0.02`) with
 *        zero orthogonal Y cross-talk (`|<cy>| < 1e-4 px`).
 *     3. Numerical parity (`rel_L2 < 1e-5`) between OpenMP `FFTW3f` and batched `cufftPlanMany`.
 *
 * Rationale:
 *   Diffractive lenslet propagation onto an even-sized (`8x8`) pixel grid requires an exact half-
 *   pixel focal-plane shift so un-tilted spots straddle the central `2x2` quad-cell symmetrically;
 *   any sign or phase-ramp error breaks CoG nulling and introduces static centroid bias.
 */
static void test_shwfs_physics(void)
{
    printf("\n=== 8. Diffractive Shack-Hartmann WFS Simulator ===\n");
    const uint32_t nwf   = 128;
    const uint32_t n_sub = 16;
    const uint32_t pps   = 8;
    const uint32_t ncam  = n_sub * pps;
    char           buf[128];

    float *pup  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *wf   = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *cam0 = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
    float *cam1 = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.25f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);

    cacaosim_shwfs_cfg_t scfg = {
        .nwf         = nwf,
        .n_sub       = n_sub,
        .pix_per_sub = pps,
        .pad_factor  = 2,
        .pup_diam    = 0.9375f,
        .lambda_um   = 0.75f,
        .wf_unit     = CACAOSIM_WFUNIT_OPD_UM,
        .ref_lambda  = 0.75f,
        .rng_seed    = 202ULL,
        .backend     = CACAOSIM_BACKEND_CPU,
        .det         = {
            .flux_ph = 100000.0f, .dark_e = 0.0f, .ron_e = 0.0f, .gain_eadu = 1.0f,
            .bias_adu = 0.0f, .sat_adu = 1e9f, .shutter = 1, .photon_noise = 0, .out_dtype = 1
        },
    };

    cacaosim_shwfs_ctx_t sh_cpu;
    cacaosim_shwfs_init(&sh_cpu, &scfg, pup, NULL, NULL);
    cacaosim_shwfs_step(&sh_cpu, pup, NULL, NULL, NULL, NULL, NULL, 0.0f, &scfg.det, NULL, cam0);

    double tot_flux = 0.0;
    for (uint32_t i = 0; i < ncam * ncam; i++)
    {
        tot_flux += cam0[i];
    }
    double cx0 = 0.0, cy0 = 0.0;
    shwfs_mean_cog(cam0, n_sub, pps, &cx0, &cy0);
    snprintf(buf, sizeof(buf), "flux=%.1f ph, <cx>=%.2e, <cy>=%.2e px", tot_flux, cx0, cy0);
    check_cond(fabs(tot_flux - 100000.0) < 100.0 && fabs(cx0) < 1e-4 && fabs(cy0) < 1e-4,
               "SHWFS flat wavefront symmetry & flux norm", buf);

    /* X-tilt spot centroid linearity */
    float r_pup = 0.5f * 0.9375f * (float) nwf;
    float c_wf  = 0.5f * (float) (nwf - 1);
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            wf[y * nwf + x] = 0.25f * ((float) x - c_wf) / r_pup;
        }
    }
    double cx1 = 0.0, cy1 = 0.0, cx2 = 0.0, cy2 = 0.0;
    cacaosim_shwfs_step(&sh_cpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &scfg.det, NULL, cam0);
    shwfs_mean_cog(cam0, n_sub, pps, &cx1, &cy1);
    cacaosim_shwfs_step(&sh_cpu, pup, NULL, wf, NULL, NULL, NULL, 2.0f, &scfg.det, NULL, cam1);
    shwfs_mean_cog(cam1, n_sub, pps, &cx2, &cy2);

    double lin_ratio = cx2 / cx1;
    snprintf(buf, sizeof(buf), "cx(1x)=%.4f, cx(2x)=%.4f (ratio=%.4f), cy=%.2e", cx1, cx2,
             lin_ratio, cy1);
    check_cond(fabs(cx1) > 0.05 && fabs(lin_ratio - 2.0) < 0.02 && fabs(cy1) < 1e-4,
               "SHWFS spot displacement linearity vs tilt", buf);

    if (cacaosim_cuda_available())
    {
        cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.14f, 303ULL, wf);
        cacaosim_shwfs_step(&sh_cpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &scfg.det, NULL, cam0);

        cacaosim_shwfs_cfg_t gcfg = scfg;
        gcfg.backend              = CACAOSIM_BACKEND_CUDA;
        cacaosim_shwfs_ctx_t sh_gpu;
        cacaosim_shwfs_init(&sh_gpu, &gcfg, pup, NULL, NULL);
        cacaosim_shwfs_step(&sh_gpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &gcfg.det, NULL, cam1);

        double err = rel_l2_f32(cam0, cam1, (uint64_t) ncam * ncam);
        snprintf(buf, sizeof(buf), "rel_L2(CPU, CUDA) = %.3e", err);
        check_cond(sh_gpu.use_cuda && err < 1e-5, "SHWFS batched cuFFT vs CPU parity", buf);
        cacaosim_shwfs_free(&sh_gpu);
    }

    cacaosim_shwfs_free(&sh_cpu);
    free(cam1);
    free(cam0);
    free(wf);
    free(pup);
}

/**
 * @brief Section 9: Validate Babinet Zernike WFS & analytic Lyot/Vortex coronagraphs + GPU parity.
 *
 * Role:
 *   Validates two complementary phase-contrast / starlight-suppression subsystems:
 *     1. Zernike WFS (`cacaosim_zwfs_*`): verifies small-phase (`5 nm` RMS) Babinet MFT linear
 *        antisymmetry (`corr(dI(+phi), -dI(-phi)) > 0.995`) for a `1.06 lambda/D`, `pi/2` dimple,
 *        plus CPU vs cuBLAS MFT parity (`rel_L2 < 1e-5`).
 *     2. Coronagraph (`cacaosim_coro_*`):
 *        - `3 lambda/D` Lyot spot: verifies `> 500x` on-axis raw peak suppression (`< 2e-3`),
 *          high off-axis companion throughput at `6 lambda/D` (`Strehl > 0.85`), and `> 40%` core
 *          energy reflection into `out_lowfs`.
 *        - Charge-2 Optical Vortex on an unobscured circular pupil: verifies `> 98%` total
 *          starlight rejection inside the post-Lyot pupil (`< 0.02` on 4x-padded FFT grid).
 *        - CPU vs CUDA parity (`< 1e-5 .. 5e-5`) across Lyot, LOWFS, and Vortex propagators.
 *
 * Rationale:
 *   Zernike LOWFS closed-loop convergence requires odd-symmetric linear intensity response around
 *   the flat-wavefront reference, while exoplanet coronagraph simulation must simultaneously reject
 *   on-axis starlight, preserve off-axis planet Strehl, and conserve reflected LOWFS core photons.
 */
static void test_zwfs_and_coro_physics(void)
{
    printf("\n=== 9. Zernike WFS & Coronagraphic Starlight Suppression ===\n");
    const uint32_t nwf  = 128;
    const uint32_t ncam = 64;
    char           buf[128];

    float *pup   = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *wf    = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *z_ref = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
    float *z_pos = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
    float *z_neg = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = {
        .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f, .nspider = 4, .spider_w = 0.01f,
        .spider_deg = 45.0f
    };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, 0.9375f, pup, 0.005f, 404ULL, wf);

    cacaosim_zwfs_cfg_t zcfg = {
        .nwf        = nwf,
        .ncam       = ncam,
        .n_mft      = 32,
        .pup_diam   = 0.9375f,
        .lambda_um  = 1.65f,
        .dimple_ld  = 1.06f,
        .dimple_rad = 0.5f * CACAOSIM_PI_F,
        .dimple_amp = 1.0f,
        .wf_unit    = CACAOSIM_WFUNIT_OPD_UM,
        .ref_lambda = 1.65f,
        .rng_seed   = 505ULL,
        .backend    = CACAOSIM_BACKEND_CPU,
        .det        = {
            .flux_ph = 100000.0f, .dark_e = 0.0f, .ron_e = 0.0f, .gain_eadu = 1.0f,
            .bias_adu = 0.0f, .sat_adu = 1e9f, .shutter = 1, .photon_noise = 0, .out_dtype = 1
        },
    };

    cacaosim_zwfs_ctx_t zw_cpu;
    cacaosim_zwfs_init(&zw_cpu, &zcfg, pup, NULL, NULL);
    cacaosim_zwfs_step(&zw_cpu, pup, NULL, NULL, NULL, NULL, NULL, 0.0f, &zcfg.det, NULL, z_ref);
    cacaosim_zwfs_step(&zw_cpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &zcfg.det, NULL, z_pos);
    cacaosim_zwfs_step(&zw_cpu, pup, NULL, wf, NULL, NULL, NULL, -1.0f, &zcfg.det, NULL, z_neg);

    double dot_pn = 0.0, n_p = 0.0, n_n = 0.0;
    for (uint32_t i = 0; i < ncam * ncam; i++)
    {
        double dp = (double) z_pos[i] - (double) z_ref[i];
        double dn = -((double) z_neg[i] - (double) z_ref[i]);
        dot_pn += dp * dn;
        n_p += dp * dp;
        n_n += dn * dn;
    }
    double z_corr = dot_pn / sqrt(n_p * n_n);
    snprintf(buf, sizeof(buf), "corr(dI(+phi), -dI(-phi)) = %.6f", z_corr);
    check_cond(n_p > 100.0 && z_corr > 0.995, "ZWFS Babinet MFT small-phase linear contrast", buf);

    if (cacaosim_cuda_available())
    {
        cacaosim_zwfs_cfg_t gzcfg = zcfg;
        gzcfg.backend             = CACAOSIM_BACKEND_CUDA;
        cacaosim_zwfs_ctx_t zw_gpu;
        cacaosim_zwfs_init(&zw_gpu, &gzcfg, pup, NULL, NULL);
        cacaosim_zwfs_step(&zw_gpu, pup, NULL, wf, NULL, NULL, NULL, 1.0f, &gzcfg.det, NULL, z_neg);

        double err = rel_l2_f32(z_pos, z_neg, (uint64_t) ncam * ncam);
        snprintf(buf, sizeof(buf), "rel_L2(CPU, CUDA) = %.3e", err);
        check_cond(zw_gpu.use_cuda && err < 1e-5, "ZWFS cuBLAS MFT vs CPU parity", buf);
        cacaosim_zwfs_free(&zw_gpu);
    }
    cacaosim_zwfs_free(&zw_cpu);

    /* Lyot-spot coronagraph: on-axis suppression, off-axis companion, and LOWFS core energy */
    cacaosim_coro_cfg_t ccfg = {
        .nwf             = nwf,
        .mode            = CACAOSIM_CORO_LYOT_SPOT,
        .n_mft           = 48,
        .pad_factor      = 4,
        .pup_diam        = 0.9375f,
        .lambda_um       = 1.65f,
        .fpm_rad_ld      = 3.0f,
        .fpm_amp         = 0.0f,
        .fpm_phase_rad   = CACAOSIM_PI_F,
        .vortex_charge   = 2,
        .lowfs_refl      = 0.95f,
        .lyot_outer      = 0.85f,
        .lyot_obs        = 0.36f,
        .lyot_nspider    = 4,
        .lyot_spider_w   = 0.02f,
        .lyot_spider_deg = 45.0f,
        .backend         = CACAOSIM_BACKEND_CPU,
    };

    cacaosim_coro_ctx_t coro_cpu;
    cacaosim_coro_init(&coro_cpu, &ccfg, NULL, NULL);

    float *zero_opd = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *tilt_opd = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *lyot_opd = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *lyot_amp = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *low_opd  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *low_amp  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    float *psf_img  = (float *) cacaosim_aligned_calloc(128 * 128, sizeof(float));

    /* Initialize FPcam normalized to unocculted Lyot pupil to measure raw contrast */
    float *lyot_pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    double pup_e    = 0.0;
    for (uint32_t i = 0; i < nwf * nwf; i++)
    {
        lyot_pup[i] = pup[i] * coro_cpu.lyot_mask[i];
        pup_e += (double) pup[i] * pup[i];
    }

    cacaosim_fpcam_cfg_t fcfg = {
        .nwf = nwf, .ncam = 128, .prop_mode = CACAOSIM_FPCAM_PROP_MFT, .pad_factor = 2,
        .pup_diam = 0.9375f, .lambda_um = 1.65f, .bw_frac = 0.0f, .nlambda = 1, .pixscale_ld = 3.0f,
        .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 1.65f, .rng_seed = 606ULL,
        .backend = CACAOSIM_BACKEND_CPU,
        .det = { .flux_ph = 1e6f, .shutter = 1, .photon_noise = 0, .gain_eadu = 1.0f,
                 .sat_adu = 1e12f, .out_dtype = 1 }
    };
    cacaosim_fpcam_ctx_t fp_coro;
    cacaosim_fpcam_init(&fp_coro, &fcfg, lyot_pup, NULL, NULL);

    /* On-axis star through Lyot spot */
    cacaosim_coro_step_cpu(&coro_cpu, zero_opd, pup, lyot_opd, lyot_amp, low_opd, low_amp);
    cacaosim_fpcam_accum_beam(&fp_coro, lyot_opd, lyot_amp, NULL, NULL, NULL);
    cacaosim_fpcam_readout(&fp_coro, &fcfg.det, NULL, psf_img, NULL);
    float onaxis_peak = fp_coro.last_strehl_peak;

    double low_e = 0.0;
    for (uint32_t i = 0; i < nwf * nwf; i++)
    {
        low_e += (double) low_amp[i] * low_amp[i];
    }

    /* Off-axis companion at 6.0 lambda/D */
    float d_pup = 0.9375f * (float) nwf;
    float c_wf  = 0.5f * (float) (nwf - 1);
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            tilt_opd[y * nwf + x] = 6.0f * 1.65f * ((float) x - c_wf) / d_pup;
        }
    }
    cacaosim_coro_step_cpu(&coro_cpu, tilt_opd, pup, lyot_opd, lyot_amp, NULL, NULL);
    cacaosim_fpcam_accum_beam(&fp_coro, lyot_opd, lyot_amp, NULL, NULL, NULL);
    cacaosim_fpcam_readout(&fp_coro, &fcfg.det, NULL, psf_img, NULL);
    float offaxis_peak = fp_coro.last_strehl_peak;

    snprintf(buf, sizeof(buf), "on-axis=%.2e, 6l/D=%.3f, LOWFS_E=%.1f%%", onaxis_peak,
             offaxis_peak, 100.0 * low_e / pup_e);
    check_cond(onaxis_peak < 2e-3f && offaxis_peak > 0.85f && (low_e / pup_e) > 0.40,
               "Lyot spot rejection, 6l/D throughput & LOWFS port", buf);

    /* Charge-2 Vortex coronagraph on clear circular aperture */
    float               *clear_pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf,
                                                                       sizeof(float));
    cacaosim_pupil_cfg_t cpcfg     = { .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.0f };
    cacaosim_make_pupil(&cpcfg, clear_pup);

    cacaosim_coro_cfg_t vcfg = ccfg;
    vcfg.mode                = CACAOSIM_CORO_VORTEX;
    vcfg.vortex_charge       = 2;
    vcfg.lyot_obs            = 0.0f;
    vcfg.lyot_nspider        = 0;
    cacaosim_coro_ctx_t vcoro;
    cacaosim_coro_init(&vcoro, &vcfg, NULL, NULL);
    cacaosim_coro_step_cpu(&vcoro, zero_opd, clear_pup, lyot_opd, lyot_amp, NULL, NULL);

    double clear_lyot_e = 0.0, vort_res_e = 0.0;
    for (uint32_t i = 0; i < nwf * nwf; i++)
    {
        double lp = (double) clear_pup[i] * vcoro.lyot_mask[i];
        clear_lyot_e += lp * lp;
        vort_res_e += (double) lyot_amp[i] * lyot_amp[i];
    }
    snprintf(buf, sizeof(buf), "post-Lyot total starlight energy = %.3e",
             vort_res_e / clear_lyot_e);
    check_cond((vort_res_e / clear_lyot_e) < 0.02,
               "Charge-2 Optical Vortex Lyot pupil suppression", buf);

    if (cacaosim_cuda_available())
    {
        cacaosim_coro_step_cpu(&coro_cpu, wf, pup, lyot_opd, lyot_amp, low_opd, low_amp);

        cacaosim_coro_cfg_t gccfg = ccfg;
        gccfg.backend             = CACAOSIM_BACKEND_CUDA;
        cacaosim_coro_ctx_t coro_gpu;
        cacaosim_coro_init(&coro_gpu, &gccfg, NULL, NULL);

        float *g_lyot_amp = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
        float *g_lyot_opd = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
        float *g_low_amp  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
        float *g_low_opd  = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
        cacaosim_coro_step_beam(&coro_gpu, wf, pup, g_lyot_opd, g_lyot_amp, g_low_opd, g_low_amp,
                                NULL, NULL, NULL, NULL, NULL, NULL);

        double err_lyot = rel_l2_f32(lyot_amp, g_lyot_amp, (uint64_t) nwf * nwf);
        double err_low  = rel_l2_f32(low_amp, g_low_amp, (uint64_t) nwf * nwf);
        snprintf(buf, sizeof(buf), "rel_L2 lyot=%.2e, lowfs=%.2e", err_lyot, err_low);
        check_cond(coro_gpu.use_cuda && err_lyot < 1e-5 && err_low < 1e-5,
                   "Coronagraph Lyot + LOWFS CPU vs CUDA parity", buf);

        cacaosim_coro_cfg_t gvcfg = vcfg;
        gvcfg.backend             = CACAOSIM_BACKEND_CUDA;
        cacaosim_coro_ctx_t vcoro_gpu;
        cacaosim_coro_init(&vcoro_gpu, &gvcfg, NULL, NULL);
        cacaosim_coro_step_cpu(&vcoro, wf, clear_pup, lyot_opd, lyot_amp, NULL, NULL);
        cacaosim_coro_step_beam(&vcoro_gpu, wf, clear_pup, g_lyot_opd, g_lyot_amp, NULL, NULL,
                                NULL, NULL, NULL, NULL, NULL, NULL);
        double err_vort = rel_l2_f32(lyot_amp, g_lyot_amp, (uint64_t) nwf * nwf);
        snprintf(buf, sizeof(buf), "rel_L2 vortex_amp=%.2e", err_vort);
        check_cond(vcoro_gpu.use_cuda && err_vort < 5e-5,
                   "Vortex coronagraph FFT CPU vs CUDA parity", buf);

        cacaosim_coro_free(&vcoro_gpu);
        cacaosim_coro_free(&coro_gpu);
        free(g_low_opd);
        free(g_low_amp);
        free(g_lyot_opd);
        free(g_lyot_amp);
    }

    cacaosim_coro_free(&vcoro);
    cacaosim_fpcam_free(&fp_coro);
    cacaosim_coro_free(&coro_cpu);
    free(clear_pup);
    free(lyot_pup);
    free(psf_img);
    free(low_amp);
    free(low_opd);
    free(lyot_amp);
    free(lyot_opd);
    free(tilt_opd);
    free(zero_opd);
    free(z_neg);
    free(z_pos);
    free(z_ref);
    free(wf);
    free(pup);
}

/**
 * @brief Compute a Tikhonov-regularized linear AO control matrix via Gauss-Jordan inversion.
 *
 * Role:
 *   Forms the normal matrix `A = I * I^T + reg * Id` (for actuator-major `imat` of shape
 *   `[nacts x npix]` where `reg = reg_frac * max_diag(I * I^T)`), inverts `A` in-place via
 *   row-normalized Gauss-Jordan elimination, and computes `cmat_out = A^(-1) * I`
 *   (`[nacts x npix]`) using single-precision BLAS (`cblas_sgemm`).
 *
 * Rationale:
 *   Provides a self-contained, LAPACK-free regularized pseudo-inverse suitable for calibrating
 *   arbitrary actuator counts (`12x12 = 144` Woofer and `24x24 = 576` Tweeter) while damping
 *   weakly illuminated edge/obscured actuators and null-space piston/waffle modes.
 *
 * @param[in]  imat      Actuator-major interaction matrix `[nacts x npix]`.
 * @param[in]  nacts     Number of DM actuators (rows of `imat` and `cmat_out`).
 * @param[in]  npix      Number of WFS detector pixels (columns of `imat` and `cmat_out`).
 * @param[in]  reg_frac  Dimensionless Tikhonov ridge fraction relative to `max_diag(I * I^T)`.
 * @param[out] cmat_out  Output regularized reconstructor matrix `[nacts x npix]`.
 */
static void build_tikhonov_cmat(
    const float *imat,
    uint32_t     nacts,
    uint32_t     npix,
    float        reg_frac,
    float       *cmat_out)
{
    float *ata = (float *) cacaosim_aligned_calloc((size_t) nacts * nacts, sizeof(float));
    float *inv = (float *) cacaosim_aligned_calloc((size_t) nacts * nacts, sizeof(float));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int) nacts, (int) nacts, (int) npix,
                1.0f, imat, (int) npix, imat, (int) npix, 0.0f, ata, (int) nacts);

    float max_d = 0.0f;
    for (uint32_t a = 0; a < nacts; a++)
    {
        if (ata[a * nacts + a] > max_d)
        {
            max_d = ata[a * nacts + a];
        }
    }
    for (uint32_t a = 0; a < nacts; a++)
    {
        ata[a * nacts + a] += reg_frac * max_d;
        inv[a * nacts + a] = 1.0f;
    }

    for (uint32_t k = 0; k < nacts; k++)
    {
        float inv_piv = 1.0f / ata[k * nacts + k];
        for (uint32_t j = 0; j < nacts; j++)
        {
            ata[k * nacts + j] *= inv_piv;
            inv[k * nacts + j] *= inv_piv;
        }
        for (uint32_t i = 0; i < nacts; i++)
        {
            if (i == k)
            {
                continue;
            }
            float f = ata[i * nacts + k];
            if (fabsf(f) < 1e-12f)
            {
                continue;
            }
            for (uint32_t j = 0; j < nacts; j++)
            {
                ata[i * nacts + j] -= f * ata[k * nacts + j];
                inv[i * nacts + j] -= f * inv[k * nacts + j];
            }
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nacts, (int) npix, (int) nacts,
                1.0f, inv, (int) nacts, imat, (int) npix, 0.0f, cmat_out, (int) npix);

    free(inv);
    free(ata);
}

/**
 * @brief Sections 10 & 11: Validate multi-stage .bench DAG, 2-stage AO & schematic visualizer.
 *
 * Role:
 *   Exercises the declarative `.bench` engine (`cacaosim_bench_*`) and schematic renderer
 *   (`cacaosim_bench_render_*`) across five end-to-end tests:
 *     - 10a (Topological Guardrails): Rejects cyclic/duplicate optical beam graphs (`b1->b2->b1`).
 *     - 10b (Multi-Rate Cadence): Verifies exact `navg` decimation ratios (`30` Vis @ `3 kHz`,
 *       `10` NIR @ `1 kHz`, `1` H-band science @ `100 Hz` over `30` master ticks).
 *     - 10c (Optical Causality & 2-Stage Cascaded AO): Verifies downstream Tweeter pokes produce
 *       zero upstream NIR WFS crosstalk (`< 1e-7`), calibrates both stages, and demonstrates
 *       monotonic H-band Strehl gain (`open < 0.35 -> Woofer > 0.55 -> Woofer+Tweeter > 0.88`) on
 *       composite low-order + `(7, 4)`-cycle/D high-spatial-frequency turbulence.
 *     - 10d (ExAO Coronagraphic Bench): Steps `examples/exao_coro_lowfs.bench` verifying `2 kHz`
 *       SHWFS, `500 Hz` Z-LOWFS, `100 Hz` CoroCam cadence, and `< 2e-3` raw stellar core peak.
 *     - 11 (Zero-FFT Analyzer & ANSI/ASCII Visualizer): Verifies `cacaosim_bench_analyze()` matches
 *       full-init sensor fluxes without FFT allocation, and tests TrueColor ANSI, pure 7-bit ASCII
 *       (zero bytes `> 0x7F` or `ESC`), and diagnostic `INVALID CONFIG` error panels.
 *
 * Rationale:
 *   Injecting `(7, 4)` cycles/D ripple exceeds the `12x12` Woofer's `6`-cycle/D Nyquist limit, so
 *   Stage 1 alone is physically fitting-error limited while Stage 2 (`24x24`, `12` cycles/D) locks
 *   onto the residual and recovers diffraction-limited Strehl—proving physical multi-rate, multi-
 *   stage DAG scheduling end-to-end.
 */
static void test_multistage_bench_dag(void)
{
    printf("\n=== 10. Multi-Stage .bench DAG & 2-Stage Cascaded Closed-Loop AO ===\n");
    char err[256];
    char buf[128];

    /* 10a. Topological cycle & missing-beam rejection */
    const char *bad_cycle =
        "[bench]\nnwf = 64\nout_beam = \"b0\"\n"
        "[splitter.s0]\nin = \"b0\"\nout_refl = \"b1\"\n"
        "[dm.d1]\nin = \"b1\"\nout = \"b2\"\n"
        "[dm.d2]\nin = \"b2\"\nout = \"b1\"\n";
    cacaosim_bench_cfg_t bad_cfg;
    cacaosim_bench_ctx_t bad_ctx;
    int prc = cacaosim_bench_parse_str(bad_cycle, &bad_cfg, err, sizeof(err));
    int irc = (prc == 0) ? cacaosim_bench_init(&bad_ctx, &bad_cfg, NULL, 0, err, sizeof(err)) : -1;
    snprintf(buf, sizeof(buf), "rejected with: %.60s", err);
    check_cond(irc != 0, "DAG rejects duplicate/cyclic optical beam graph", buf);

    const char *src_dir = getenv("CACAOSIM_SRC_DIR");
    if (src_dir == NULL || src_dir[0] == '\0')
    {
        src_dir = ".";
    }
    char path_2stg[512];
    char path_exao[512];
    snprintf(path_2stg, sizeof(path_2stg), "%s/examples/two_stage_ao.bench", src_dir);
    snprintf(path_exao, sizeof(path_exao), "%s/examples/exao_coro_lowfs.bench", src_dir);

    /* 10b. Load two_stage_ao.bench & verify multi-rate cadence (3 kHz / 1 kHz / 100 Hz) */
    cacaosim_bench_cfg_t cfg2;
    if (cacaosim_bench_load_file(path_2stg, &cfg2, err, sizeof(err)) != 0)
    {
        check_cond(0, "Load examples/two_stage_ao.bench", err);
        return;
    }

    cacaosim_bench_ctx_t ctx_rate;
    if (cacaosim_bench_init(&ctx_rate, &cfg2, NULL, 0, err, sizeof(err)) != 0)
    {
        check_cond(0, "Init examples/two_stage_ao.bench", err);
        return;
    }

    int s_nir = cacaosim_bench_find_sensor(&ctx_rate, "wfs_nir");
    int s_vis = cacaosim_bench_find_sensor(&ctx_rate, "wfs_vis");
    int s_sci = cacaosim_bench_find_sensor(&ctx_rate, "sci_h");
    int cnt_nir = 0, cnt_vis = 0, cnt_sci = 0;
    for (int t = 0; t < 30; t++)
    {
        cacaosim_bench_step(&ctx_rate, NULL, NULL, 0.0f, NULL);
        cnt_nir += ctx_rate.nodes[ctx_rate.sensor_node_idx[s_nir]].ready_this_tick;
        cnt_vis += ctx_rate.nodes[ctx_rate.sensor_node_idx[s_vis]].ready_this_tick;
        cnt_sci += ctx_rate.nodes[ctx_rate.sensor_node_idx[s_sci]].ready_this_tick;
    }
    snprintf(buf, sizeof(buf), "30 ticks -> vis=%d, nir=%d, sci=%d readouts", cnt_vis, cnt_nir,
             cnt_sci);
    check_cond(cnt_vis == 30 && cnt_nir == 10 && cnt_sci == 1,
               "Multi-rate sensor cadence (3kHz / 1kHz / 100Hz)", buf);
    cacaosim_bench_free(&ctx_rate);

    /* 10c. Optical causality & 2-stage cascaded closed-loop AO */
    cfg2.wf_unit = CACAOSIM_WFUNIT_OPD_UM;
    for (uint32_t i = 0; i < cfg2.n_nodes; i++)
    {
        cacaosim_node_cfg_t *nd = &cfg2.nodes[i];
        if (nd->kind == CACAOSIM_NODE_DM)
        {
            nd->dm.dyn_mode     = 0;
            nd->dm.delay_frames = 0.0f;
            nd->dm.dac_bits     = 0;
        }
        else if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            nd->navg                    = 1;
            nd->pyrwfs.mod_npts         = 8;
            nd->pyrwfs.det.photon_noise = 0;
            nd->pyrwfs.det.ron_e        = 0.0f;
            nd->pyrwfs.det.bias_adu     = 0.0f;
            nd->pyrwfs.det.sat_adu      = 1e12f;
            nd->pyrwfs.det.out_dtype    = 1;
        }
        else if (nd->kind == CACAOSIM_NODE_FPCAM)
        {
            nd->navg                   = 1;
            nd->fpcam.det.photon_noise = 0;
            nd->fpcam.det.ron_e        = 0.0f;
            nd->fpcam.det.bias_adu     = 0.0f;
            nd->fpcam.det.sat_adu      = 1e12f;
            nd->fpcam.det.out_dtype    = 1;
        }
    }

    cacaosim_bench_ctx_t ctx;
    if (cacaosim_bench_init(&ctx, &cfg2, NULL, 0, err, sizeof(err)) != 0)
    {
        check_cond(0, "Init calibration 2-stage bench", err);
        return;
    }

    int    d_woof   = cacaosim_bench_find_dm(&ctx, "woofer");
    int    d_twet   = cacaosim_bench_find_dm(&ctx, "tweeter");
    s_nir           = cacaosim_bench_find_sensor(&ctx, "wfs_nir");
    s_vis           = cacaosim_bench_find_sensor(&ctx, "wfs_vis");
    s_sci           = cacaosim_bench_find_sensor(&ctx, "sci_h");
    float *cmd_woof = cacaosim_bench_dm_cmd_ptr(&ctx, d_woof);
    float *cmd_twet = cacaosim_bench_dm_cmd_ptr(&ctx, d_twet);

    const uint32_t nw_act = 12 * 12;
    const uint32_t nt_act = 24 * 24;
    const uint32_t npwfs  = 64 * 64;

    float *ref_nir = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    float *ref_vis = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
    memcpy(ref_nir, cacaosim_bench_sensor_f32(&ctx, s_nir), npwfs * sizeof(float));
    memcpy(ref_vis, cacaosim_bench_sensor_f32(&ctx, s_vis), npwfs * sizeof(float));

    /* Optical causality check: downstream tweeter must NOT affect upstream NIR WFS */
    cmd_twet[12 * 24 + 17] = 0.20f;
    cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
    double twet_on_nir = rel_l2_f32(ref_nir, cacaosim_bench_sensor_f32(&ctx, s_nir), npwfs);
    double twet_on_vis = rel_l2_f32(ref_vis, cacaosim_bench_sensor_f32(&ctx, s_vis), npwfs);
    cmd_twet[12 * 24 + 17] = 0.0f;

    /* Upstream woofer MUST affect BOTH NIR WFS and downstream Visible WFS */
    cmd_woof[6 * 12 + 8] = 0.20f;
    cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
    double woof_on_nir = rel_l2_f32(ref_nir, cacaosim_bench_sensor_f32(&ctx, s_nir), npwfs);
    double woof_on_vis = rel_l2_f32(ref_vis, cacaosim_bench_sensor_f32(&ctx, s_vis), npwfs);
    cmd_woof[6 * 12 + 8] = 0.0f;

    snprintf(buf, sizeof(buf), "twet->(nir=%.1e,vis=%.2e) woof->(nir=%.2e,vis=%.2e)", twet_on_nir,
             twet_on_vis, woof_on_nir, woof_on_vis);
    check_cond(twet_on_nir < 1e-7 && twet_on_vis > 1e-2 && woof_on_nir > 1e-2 && woof_on_vis > 1e-2,
               "2-stage optical causality (Woofer -> NIR -> Tweeter -> Vis)", buf);

    /* Calibrate Stage-1 (Woofer -> NIR WFS) and Stage-2 (Tweeter -> Vis WFS) control matrices */
    float *imat1 = (float *) cacaosim_aligned_calloc((size_t) nw_act * npwfs, sizeof(float));
    float *cmat1 = (float *) cacaosim_aligned_calloc((size_t) nw_act * npwfs, sizeof(float));
    float *imat2 = (float *) cacaosim_aligned_calloc((size_t) nt_act * npwfs, sizeof(float));
    float *cmat2 = (float *) cacaosim_aligned_calloc((size_t) nt_act * npwfs, sizeof(float));

    const float poke1 = 0.04f;
    for (uint32_t a = 0; a < nw_act; a++)
    {
        cmd_woof[a] = poke1;
        cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
        const float *im_p = cacaosim_bench_sensor_f32(&ctx, s_nir);
        for (uint32_t p = 0; p < npwfs; p++)
        {
            imat1[(size_t) a * npwfs + p] = (im_p[p] - ref_nir[p]) / poke1;
        }
        cmd_woof[a] = 0.0f;
    }
    build_tikhonov_cmat(imat1, nw_act, npwfs, 0.02f, cmat1);

    const float poke2 = 0.02f;
    for (uint32_t a = 0; a < nt_act; a++)
    {
        cmd_twet[a] = poke2;
        cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
        const float *im_p = cacaosim_bench_sensor_f32(&ctx, s_vis);
        for (uint32_t p = 0; p < npwfs; p++)
        {
            imat2[(size_t) a * npwfs + p] = (im_p[p] - ref_vis[p]) / poke2;
        }
        cmd_twet[a] = 0.0f;
    }
    build_tikhonov_cmat(imat2, nt_act, npwfs, 0.03f, cmat2);

    /* Composite aberration: 0.45 um low-order + 0.14 um 7-cycle/D high-spatial-freq ripple */
    const uint32_t nwf  = ctx.nwf;
    float         *turb = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    cacaosim_make_ncpa(nwf, cfg2.pup_diam, ctx.pup_ptr, 0.45f, 707ULL, turb);
    float d_pup = cfg2.pup_diam * (float) nwf;
    float c_wf  = 0.5f * (float) (nwf - 1);
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            float ux = ((float) x - c_wf) / d_pup;
            float uy = ((float) y - c_wf) / d_pup;
            turb[y * nwf + x] += 0.16f * sinf(2.0f * CACAOSIM_PI_F * 7.0f * ux) *
                                 cosf(2.0f * CACAOSIM_PI_F * 4.0f * uy);
        }
    }

    /* 1) Open-loop H-band Strehl */
    cacaosim_bench_step(&ctx, turb, NULL, 1.0f, NULL);
    float strehl_open = ctx.last_strehl_inst;

    /* Enable realistic multi-rate cadence (1 kHz Stage 1, 3 kHz Stage 2) + 1-frame delay + RC */
    cacaosim_bench_node_t *nd_woof = &ctx.nodes[ctx.dm_node_idx[d_woof]];
    cacaosim_bench_node_t *nd_twet = &ctx.nodes[ctx.dm_node_idx[d_twet]];
    cacaosim_bench_node_t *nd_nir  = &ctx.nodes[ctx.sensor_node_idx[s_nir]];
    cacaosim_bench_node_t *nd_vis  = &ctx.nodes[ctx.sensor_node_idx[s_vis]];

    nd_woof->cfg.dm.dyn_mode     = 1;
    nd_woof->cfg.dm.delay_frames = 1.0f;
    nd_woof->dm_ctx.cfg.dyn_mode = 1;
    nd_twet->cfg.dm.dyn_mode     = 1;
    nd_twet->cfg.dm.delay_frames = 1.0f;
    nd_twet->dm_ctx.cfg.dyn_mode = 1;
    nd_nir->eff_navg             = 3;
    nd_vis->eff_navg             = 1;

    float *ds   = (float *) cacaosim_aligned_calloc(npwfs, sizeof(float));
    float *dw   = (float *) cacaosim_aligned_calloc(nw_act, sizeof(float));
    float *dt_c = (float *) cacaosim_aligned_calloc(nt_act, sizeof(float));

    /* 2) Stage-1 Woofer loop ONLY (12x12 @ 1 kHz, Tweeter OFF) */
    cacaosim_dm_reset(&nd_woof->dm_ctx);
    cacaosim_dm_reset(&nd_twet->dm_ctx);
    memset(cmd_woof, 0, nw_act * sizeof(float));
    memset(cmd_twet, 0, nt_act * sizeof(float));
    nd_nir->pyrwfs_ctx.subframe_cnt = 0;

    for (int t = 0; t < 90; t++)
    {
        cacaosim_bench_step(&ctx, turb, NULL, 1.0f, NULL);
        if (nd_nir->ready_this_tick)
        {
            const float *im_n = cacaosim_bench_sensor_f32(&ctx, s_nir);
            for (uint32_t p = 0; p < npwfs; p++)
            {
                ds[p] = (im_n[p] / 3.0f) - ref_nir[p];
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) nw_act, (int) npwfs, 1.0f, cmat1,
                        (int) npwfs, ds, 1, 0.0f, dw, 1);
            float mean_w = 0.0f;
            for (uint32_t a = 0; a < nw_act; a++)
            {
                cmd_woof[a] = 0.998f * cmd_woof[a] - 0.45f * dw[a];
                mean_w += cmd_woof[a];
            }
            mean_w /= (float) nw_act;
            for (uint32_t a = 0; a < nw_act; a++)
            {
                cmd_woof[a] -= mean_w;
            }
        }
    }
    float strehl_stg1 = ctx.last_strehl_inst;

    /* 3) Full 2-Stage Cascaded Loop (Woofer @ 1 kHz + Tweeter @ 3 kHz) */
    cacaosim_dm_reset(&nd_woof->dm_ctx);
    cacaosim_dm_reset(&nd_twet->dm_ctx);
    memset(cmd_woof, 0, nw_act * sizeof(float));
    memset(cmd_twet, 0, nt_act * sizeof(float));
    nd_nir->pyrwfs_ctx.subframe_cnt = 0;
    nd_vis->pyrwfs_ctx.subframe_cnt = 0;

    for (int t = 0; t < 90; t++)
    {
        cacaosim_bench_step(&ctx, turb, NULL, 1.0f, NULL);
        if (nd_nir->ready_this_tick)
        {
            const float *im_n = cacaosim_bench_sensor_f32(&ctx, s_nir);
            for (uint32_t p = 0; p < npwfs; p++)
            {
                ds[p] = (im_n[p] / 3.0f) - ref_nir[p];
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) nw_act, (int) npwfs, 1.0f, cmat1,
                        (int) npwfs, ds, 1, 0.0f, dw, 1);
            float mean_w = 0.0f;
            for (uint32_t a = 0; a < nw_act; a++)
            {
                cmd_woof[a] = 0.998f * cmd_woof[a] - 0.45f * dw[a];
                mean_w += cmd_woof[a];
            }
            mean_w /= (float) nw_act;
            for (uint32_t a = 0; a < nw_act; a++)
            {
                cmd_woof[a] -= mean_w;
            }
        }
        if (t >= 15 && nd_vis->ready_this_tick)
        {
            const float *im_v = cacaosim_bench_sensor_f32(&ctx, s_vis);
            for (uint32_t p = 0; p < npwfs; p++)
            {
                ds[p] = im_v[p] - ref_vis[p];
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) nt_act, (int) npwfs, 1.0f, cmat2,
                        (int) npwfs, ds, 1, 0.0f, dt_c, 1);
            float mean_t = 0.0f;
            for (uint32_t a = 0; a < nt_act; a++)
            {
                cmd_twet[a] = 0.998f * cmd_twet[a] - 0.35f * dt_c[a];
                mean_t += cmd_twet[a];
            }
            mean_t /= (float) nt_act;
            for (uint32_t a = 0; a < nt_act; a++)
            {
                cmd_twet[a] -= mean_t;
            }
        }
    }
    float strehl_2stg = ctx.last_strehl_inst;

    snprintf(buf, sizeof(buf), "H-Strehl: open=%.3f -> stg1=%.3f -> 2-stage=%.3f", strehl_open,
             strehl_stg1, strehl_2stg);
    check_cond(strehl_open < 0.35f && strehl_stg1 > 0.55f && strehl_2stg > strehl_stg1 + 0.08f &&
                   strehl_2stg > 0.88f,
               "2-stage cascaded AO loop (Woofer + Tweeter)", buf);

    free(dt_c);
    free(dw);
    free(ds);
    free(turb);
    free(cmat2);
    free(imat2);
    free(cmat1);
    free(imat1);
    free(ref_vis);
    free(ref_nir);
    cacaosim_bench_free(&ctx);

    /* 10d. Load & step examples/exao_coro_lowfs.bench (TTM + DM + SHWFS + Coro + ZWFS + FPcam) */
    cacaosim_bench_cfg_t cfg_exao;
    if (cacaosim_bench_load_file(path_exao, &cfg_exao, err, sizeof(err)) != 0)
    {
        check_cond(0, "Load examples/exao_coro_lowfs.bench", err);
        return;
    }
    cacaosim_bench_ctx_t ctx_exao;
    if (cacaosim_bench_init(&ctx_exao, &cfg_exao, NULL, 0, err, sizeof(err)) != 0)
    {
        check_cond(0, "Init examples/exao_coro_lowfs.bench", err);
        return;
    }

    int s_ho   = cacaosim_bench_find_sensor(&ctx_exao, "wfs_ho");
    int s_low  = cacaosim_bench_find_sensor(&ctx_exao, "lowfs_z");
    int s_ccam = cacaosim_bench_find_sensor(&ctx_exao, "coro_cam");
    int n_ho = 0, n_low = 0, n_ccam = 0;
    for (int t = 0; t < 20; t++)
    {
        cacaosim_bench_step(&ctx_exao, NULL, NULL, 0.0f, NULL);
        n_ho += ctx_exao.nodes[ctx_exao.sensor_node_idx[s_ho]].ready_this_tick;
        n_low += ctx_exao.nodes[ctx_exao.sensor_node_idx[s_low]].ready_this_tick;
        n_ccam += ctx_exao.nodes[ctx_exao.sensor_node_idx[s_ccam]].ready_this_tick;
    }
    snprintf(buf, sizeof(buf), "SHWFS=%d, Z-LOWFS=%d, CoroCam=%d, rawPeak=%.2e", n_ho, n_low,
             n_ccam, ctx_exao.last_strehl_inst);
    check_cond(n_ho == 20 && n_low == 5 && n_ccam == 1 && ctx_exao.last_strehl_inst < 2e-3f,
               "ExAO Coro + TTM + SHWFS + Z-LOWFS bench execution", buf);

    /* 11. Zero-FFT bench analyzer & milk-styled ANSI / UTF-8 / ASCII visualizer */
    printf("\n=== 11. Optical Bench ANSI / UTF-8 / ASCII Visualizer ===\n");
    cacaosim_bench_ctx_t dry_exao;
    int                  dry_rc = cacaosim_bench_analyze(&dry_exao, &cfg_exao, err, sizeof(err));
    int                  flux_ok = (dry_rc == 0 && dry_exao.n_beams == ctx_exao.n_beams &&
                                    dry_exao.n_nodes == ctx_exao.n_nodes &&
                                    dry_exao.n_sensors == ctx_exao.n_sensors);
    for (uint32_t s = 0; flux_ok && s < dry_exao.n_sensors; s++)
    {
        float f_dry  = dry_exao.nodes[dry_exao.sensor_node_idx[s]].eff_flux_ph;
        float f_init = ctx_exao.nodes[ctx_exao.sensor_node_idx[s]].eff_flux_ph;
        if (fabsf(f_dry - f_init) > 1e-3f * (f_init + 1.0f))
        {
            flux_ok = 0;
        }
    }
    snprintf(buf, sizeof(buf), "beams=%u, nodes=%u, sensors=%u, zero-FFT flux match",
             dry_exao.n_beams, dry_exao.n_nodes, dry_exao.n_sensors);
    check_cond(flux_ok, "Zero-FFT cacaosim_bench_analyze() parity", buf);
    cacaosim_bench_free(&ctx_exao);

    /* Render TrueColor, Monochrome UTF-8, Pure 7-bit ASCII, and Invalid Config Error Panel */
    char  *vbuf   = NULL;
    size_t vlen   = 0;
    FILE  *mfp    = open_memstream(&vbuf, &vlen);
    int    tc_ok  = 0;
    if (mfp != NULL)
    {
        cacaosim_view_opts_t vopts;
        cacaosim_view_opts_defaults(&vopts);
        vopts.color_level = 3;
        vopts.theme       = CACAOSIM_THEME_DARK;
        vopts.width       = 104;
        int rrc           = cacaosim_bench_render_file(path_2stg, &vopts, mfp);
        fflush(mfp);
        fclose(mfp);
        tc_ok = (rrc == 0 && vbuf != NULL && strstr(vbuf, "\033[38;2;") != NULL &&
                 strstr(vbuf, "dichroic_stg1") != NULL && strstr(vbuf, "aol1_wfsim") != NULL &&
                 strstr(vbuf, "aol2_wfsim") != NULL);
        snprintf(buf, sizeof(buf), "bytes=%zu, TrueColor CSI + 2-stage schematic verified", vlen);
        free(vbuf);
    }
    check_cond(tc_ok, "TrueColor ANSI optical schematic render", buf);

    vbuf = NULL;
    vlen = 0;
    mfp  = open_memstream(&vbuf, &vlen);
    int as_ok = 0;
    if (mfp != NULL)
    {
        cacaosim_view_opts_t vopts;
        cacaosim_view_opts_defaults(&vopts);
        vopts.color_level = 0;
        vopts.ascii_only  = 1;
        vopts.width       = 104;
        int rrc           = cacaosim_bench_render_file(path_exao, &vopts, mfp);
        fflush(mfp);
        fclose(mfp);
        int pure_7bit = (rrc == 0 && vbuf != NULL && vlen > 200);
        for (size_t k = 0; pure_7bit && k < vlen; k++)
        {
            unsigned char uc = (unsigned char) vbuf[k];
            if (uc > 0x7Fu || uc == 0x1Bu)
            {
                pure_7bit = 0;
            }
        }
        as_ok = pure_7bit && strstr(vbuf, "coro_h") != NULL && strstr(vbuf, "lowfs_z") != NULL;
        snprintf(buf, sizeof(buf), "bytes=%zu, 100%% 7-bit ASCII, zero ESC bytes", vlen);
        free(vbuf);
    }
    check_cond(as_ok, "Monochrome 7-bit ASCII fallback render", buf);

    vbuf = NULL;
    vlen = 0;
    mfp  = open_memstream(&vbuf, &vlen);
    int er_ok = 0;
    if (mfp != NULL)
    {
        cacaosim_bench_cfg_t bad_cfg;
        const char          *bad_txt = "[bench]\nname=\"bad\"\n[pyrwfs.w]\nin=\"b_missing\"\n";
        cacaosim_bench_parse_str(bad_txt, &bad_cfg, err, sizeof(err));
        cacaosim_bench_ctx_t bad_ctx;
        int                  arc = cacaosim_bench_analyze(&bad_ctx, &bad_cfg, err, sizeof(err));
        cacaosim_view_opts_t vopts;
        cacaosim_view_opts_defaults(&vopts);
        vopts.color_level = 0;
        cacaosim_bench_render_error("bad.bench", &bad_cfg, err, &vopts, mfp);
        fflush(mfp);
        fclose(mfp);
        er_ok = (arc != 0 && vbuf != NULL && strstr(vbuf, "INVALID CONFIG") != NULL &&
                 strstr(vbuf, "b_missing") != NULL);
        snprintf(buf, sizeof(buf), "detected missing beam & rendered error panel (%zu B)", vlen);
        free(vbuf);
    }
    check_cond(er_ok, "Diagnostic error panel on invalid .bench", buf);
}

/**
 * @brief Section 12: Validate .bench presets, topological splicing editor & headless TUI renderer.
 *
 * Role:
 *   Validates the interactive bench-building backend (`cacaosim_bench_*` & `cacaosim_tui_*`):
 *     - 12a: Formats and re-parses all `CACAOSIM_PRESET_COUNT` built-in presets, checking lossless
 *       node/beam/sensor topology and `< 0.1%` photometric flux agreement.
 *     - 12b: Constructs a 7-node ExAO bench (`TTM -> DM -> Splitter -> PyWFS + Coro -> ZWFS + FP`)
 *       via smart auto-wiring (`cacaosim_bench_add_node`), then exercises upstream/downstream node
 *       moves across both inline elements and branch splitters, branch swapping, cloning, renaming,
 *       and bridge-healing node deletion (`cacaosim_bench_remove_node`).
 *     - 12c: Renders `cacaosim-CTRL` headlessly into memory (`cacaosim_tui_render_headless`),
 *       verifying UTF-8 box-drawing 3-pane Builder layout and pure 7-bit ASCII across all 5 views.
 *
 * Rationale:
 *   Interactive optical splicing rewrites upstream/downstream beam names across splitters and
 *   coronagraphs; automated invariants ensure no sequence of GUI edits can orphan a sensor or
 *   corrupt `.bench` serialization, and headless rendering tests TUI layout geometry in CI without
 *   requiring an interactive terminal.
 */
static void test_bench_configurator_and_tui(void)
{
    printf("\n=== 12. Hardware Configurator, Optical Splicing & cacaosim-CTRL TUI ===\n");
    char err[256];
    char detail[128];

    /* 12a. All 6 built-in presets + format_str / parse_str lossless round-trip */
    int presets_ok = 1;
    for (int p = 0; p < CACAOSIM_PRESET_COUNT; p++)
    {
        cacaosim_bench_cfg_t cfg0, cfg1;
        cacaosim_bench_ctx_t ctx0, ctx1;
        if (cacaosim_bench_preset(&cfg0, p) != 0 ||
            cacaosim_bench_analyze(&ctx0, &cfg0, err, sizeof(err)) != 0)
        {
            presets_ok = 0;
            break;
        }
        char txt[16384];
        int  flen = cacaosim_bench_format_str(&cfg0, txt, sizeof(txt));
        if (flen <= 0 || flen >= (int) sizeof(txt) ||
            cacaosim_bench_parse_str(txt, &cfg1, err, sizeof(err)) != 0 ||
            cacaosim_bench_analyze(&ctx1, &cfg1, err, sizeof(err)) != 0 ||
            ctx0.n_nodes != ctx1.n_nodes || ctx0.n_beams != ctx1.n_beams ||
            ctx0.n_sensors != ctx1.n_sensors)
        {
            presets_ok = 0;
            break;
        }
        for (uint32_t s = 0; s < ctx0.n_sensors; s++)
        {
            float f0 = ctx0.nodes[ctx0.sensor_node_idx[s]].eff_flux_ph;
            float f1 = ctx1.nodes[ctx1.sensor_node_idx[s]].eff_flux_ph;
            if (fabsf(f0 - f1) > 1e-3f * (f0 + 1.0f))
            {
                presets_ok = 0;
                break;
            }
        }
    }
    snprintf(detail, sizeof(detail), "%d presets formatted, re-parsed & flux-matched",
             CACAOSIM_PRESET_COUNT);
    check_cond(presets_ok, "All .bench presets & INI/TOML round-trip", detail);

    /* 12b. Smart element insertion, optical chain splicing, branch swap, clone & removal */
    cacaosim_bench_cfg_t bcfg;
    cacaosim_bench_ctx_t bctx;
    cacaosim_bench_cfg_defaults(&bcfg);

    int i_ttm  = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_DM, 1, -1, NULL);
    int i_dm   = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_DM, 0, i_ttm, NULL);
    int i_bs   = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_SPLITTER, 0, i_dm, NULL);
    int i_pyr  = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_PYRWFS, 0, i_bs, NULL);
    int i_coro = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_CORO, 0, i_bs, NULL);
    int i_zwfs = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_ZWFS, 0, i_coro, NULL);
    int i_sci  = cacaosim_bench_add_node(&bcfg, CACAOSIM_NODE_FPCAM, 0, i_coro, NULL);

    int build_ok = (i_ttm >= 0 && i_dm >= 0 && i_bs >= 0 && i_pyr >= 0 && i_coro >= 0 &&
                    i_zwfs >= 0 && i_sci >= 0 && bcfg.n_nodes == 7u &&
                    cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0 &&
                    bctx.n_dms == 2u && bctx.n_sensors == 3u);
    snprintf(detail, sizeof(detail), "nodes=%u, dms=%u, sensors=%u, beams=%u", bcfg.n_nodes,
             bctx.n_dms, bctx.n_sensors, bctx.n_beams);
    check_cond(build_ok, "Smart auto-wired ExAO bench construction", detail);

    /* Reorder DM upstream across TTM, back downstream, across Splitter, and back */
    int mv1 = cacaosim_bench_move_node(&bcfg, 1u, -1);
    int ok1 = (mv1 == 0 && strcmp(bcfg.nodes[0].id, "dm0") == 0 &&
               cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0);
    int mv2 = cacaosim_bench_move_node(&bcfg, 0u, +1);
    int ok2 = (mv2 == 1 && strcmp(bcfg.nodes[1].id, "dm0") == 0 &&
               cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0);
    int mv3 = cacaosim_bench_move_node(&bcfg, 1u, +1);
    int ok3 = (mv3 >= 2 && cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0);
    int mv4 = (mv3 >= 0) ? cacaosim_bench_move_node(&bcfg, (uint32_t) mv3, -1) : -1;
    int ok4 = (mv4 >= 0 && cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0);

    int sw_ok = (cacaosim_bench_swap_branches(&bcfg, 2u) == 0 &&
                 cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0 &&
                 cacaosim_bench_swap_branches(&bcfg, 2u) == 0);
    int cl_idx = cacaosim_bench_clone_node(&bcfg, 1u);
    int rn_ok  = (cl_idx >= 0 &&
                  cacaosim_bench_rename_node(&bcfg, (uint32_t) cl_idx, "tweeter_hi", err,
                                             sizeof(err)) == 0);
    int rm_ok  = (cacaosim_bench_remove_node(&bcfg, 0u) == 0 &&
                  cacaosim_bench_analyze(&bctx, &bcfg, err, sizeof(err)) == 0);

    snprintf(detail, sizeof(detail), "move=%d/%d/%d/%d, swap=%d, clone+rename=%d, bridge_del=%d",
             ok1, ok2, ok3, ok4, sw_ok, rn_ok, rm_ok);
    check_cond(ok1 && ok2 && ok3 && ok4 && sw_ok && rn_ok && rm_ok,
               "Optical chain splicing, swap, clone & delete", detail);

    /* 12c. Headless cacaosim-CTRL frame renderer across all 5 views (UTF-8 + 7-bit ASCII) */
    char *hbuf = (char *) malloc(65536u);
    int   tui_utf8_ok = 0;
    if (hbuf != NULL)
    {
        int n = cacaosim_tui_render_headless(&bcfg, CACAOSIM_TUI_VIEW_BUILDER, 128, 40, 0, hbuf,
                                             65536u);
        tui_utf8_ok = (n > 2000 && strstr(hbuf, "cacaosim-CTRL") != NULL &&
                       strstr(hbuf, "ELEMENTS & ARRANGER") != NULL &&
                       strstr(hbuf, "2D OPTICAL SCHEMATIC") != NULL &&
                       strstr(hbuf, "INSPECTOR") != NULL && strstr(hbuf, "tweeter_hi") != NULL &&
                       strstr(hbuf, "\xe2\x95") != NULL);
        snprintf(detail, sizeof(detail), "bytes=%d, UTF-8 3-pane Builder + DAG verified", n);
    }
    check_cond(tui_utf8_ok, "cacaosim-CTRL UTF-8 3-pane Builder snapshot", detail);

    int    tui_ascii_ok      = (hbuf != NULL);
    size_t total_ascii_bytes = 0u;
    for (int v = 0; tui_ascii_ok && v < CACAOSIM_TUI_VIEW_COUNT; v++)
    {
        int n = cacaosim_tui_render_headless(&bcfg, v, 120, 36, 1, hbuf, 65536u);
        if (n < 1000)
        {
            tui_ascii_ok = 0;
            break;
        }
        for (int k = 0; k < n; k++)
        {
            unsigned char uc = (unsigned char) hbuf[k];
            if (uc > 0x7Fu || uc == 0x1Bu)
            {
                tui_ascii_ok = 0;
                break;
            }
        }
        total_ascii_bytes += (size_t) n;
    }
    free(hbuf);
    snprintf(detail, sizeof(detail), "all %d views rendered (%zu B), 100%% 7-bit ASCII",
             CACAOSIM_TUI_VIEW_COUNT, total_ascii_bytes);
    check_cond(tui_ascii_ok, "cacaosim-CTRL 5 views 7-bit ASCII render", detail);
}

/**
 * @brief Section 13: Validate custom 2D & 3D polychromatic pupil/focal/Lyot amplitude/phase masks.
 *
 * Role:
 *   Validates user-supplied FITS/memory coronagraphic masks (`cacaosim_mask_*` & `CUSTOM_MFT/FFT`):
 *     - 13a: 3D `[32 x 32 x 5]` FITS cube writer/reader round-trip including spectral WCS axis-3
 *       cards (`CRPIX3`, `CRVAL3`, `CDELT3`, `CUNIT3 = 'um'`).
 *     - 13b: Exact equivalence (`rel_L2 < 1e-5`) between 2D `CUSTOM_MFT` (`8x8` supersampled
 *       complex dimple) and analytic `PHASE_DIMPLE`, and between `CUSTOM_FFT` and analytic `4QPM`.
 *     - 13c: Pupil-only + Lyot-only complex `[amp, OPD_um]` apodizer relay (`skip_fpm == 1`),
 *       checking multiplicative amplitude `pup * p_amp * l_amp` and additive OPD
 *       `wf + p_opd + l_opd`.
 *     - 13d: Broadband (`20%` BW, `5` channels) stellar core suppression comparing an achromatic
 *       3D half-wave 4QPM OPD cube (`0.5 * lambda_c` per slice) against a chromatic 2D half-wave
 *       plate (`0.5 * lambda_0`), verifying `> 25x` core leakage reduction.
 *     - 13e: 3D polychromatic `coro + fpcam` CPU vs CUDA parity (`rel_L2 < 5e-5`) for both FFT/MFT.
 *
 * Rationale:
 *   Modern coronagraph designs (shaped-pupil/APLC apodizers, scalar/vector vortex or multi-zone
 *   masks) exhibit wavelength-dependent amplitude/OPD responses stored as 3D FITS cubes; comparing
 *   achromatic 3D OPD slices against chromatic 2D OPD physically proves per-wavelength slice
 *   indexing and phase conversion (`2 * pi * OPD / lambda_c`).
 */
static void test_custom_coro_masks_and_poly_cube(void)
{
    printf("\n=== 13. Custom Pupil/Focal Amplitude & Phase Masks + 3D Polychromatic Cubes ===\n");
    char detail[160];
    char err[256];

    const uint32_t nwf  = 64u;
    const uint64_t npix = (uint64_t) nwf * nwf;
    const float    pdia = 0.875f;

    /* 13a. 3D FITS cube writer/reader round-trip with spectral WCS keywords */
    const uint32_t nz_test     = 5u;
    const float    lam_wcs[5]  = { 1.485f, 1.5675f, 1.650f, 1.7325f, 1.815f };
    const size_t   cube_n      = 32u * 32u * nz_test;
    float         *cube_src    = (float *) cacaosim_aligned_calloc(cube_n, sizeof(float));
    const char    *fits_rt_pth = "/tmp/cacaosim_val_3dcube_rt.fits";

    for (size_t i = 0; i < cube_n; i++)
    {
        cube_src[i] = 0.5f + 0.4f * sinf(0.13f * (float) i);
    }
    int save_ok = (cacaosim_mask_save_fits(fits_rt_pth, cube_src, 32u, 32u, nz_test, lam_wcs) == 0);
    cacaosim_mask_cube_t loaded = { 0 };
    int load_ok = save_ok && (cacaosim_mask_load(fits_rt_pth, 0, &loaded, err, sizeof(err)) == 0);
    double rt_err  = (load_ok && loaded.data) ? rel_l2_f32(cube_src, loaded.data, cube_n) : 1.0;
    float  lam_err = 0.0f;
    if (load_ok && loaded.has_wcs_lam && loaded.nz == nz_test)
    {
        for (uint32_t c = 0; c < nz_test; c++)
        {
            float de = fabsf(loaded.lam_um[c] - lam_wcs[c]);
            if (de > lam_err)
            {
                lam_err = de;
            }
        }
    }
    else
    {
        lam_err = 1.0f;
    }
    snprintf(detail, sizeof(detail), "32x32x%u FITS rel_L2=%.2e, max_dlam=%.2e um", loaded.nz,
             rt_err, lam_err);
    check_cond(load_ok && loaded.nx == 32u && loaded.ny == 32u && loaded.nz == nz_test &&
                   loaded.has_wcs_lam && rt_err < 1e-7 && lam_err < 1e-5f,
               "3D polychromatic FITS mask WCS round-trip", detail);
    cacaosim_mask_free(&loaded);
    remove(fits_rt_pth);
    free(cube_src);

    /* Common entrance pupil and test aberration */
    float *pup = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *wf  = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = pdia, .cobs = 0.0f };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, pdia, pup, 0.06f, 2026ULL, wf);

    /* 13b. Custom 2D focal amp+phase MFT vs analytic PHASE_DIMPLE & custom FFT vs analytic 4QPM */
    const uint32_t nmft   = 48u;
    const float    r_ld   = 2.5f;
    const float    fov    = 2.4f * r_ld;
    const float    px_ld  = (float) nmft / fov;
    float         *c_famp = (float *) cacaosim_aligned_calloc((size_t) nmft * nmft, sizeof(float));
    float         *c_fpha = (float *) cacaosim_aligned_calloc((size_t) nmft * nmft, sizeof(float));
    const float    du_mft = 1.0f / px_ld;
    const float    c_mft  = 0.5f * (float) (nmft - 1u);

    for (uint32_t v = 0; v < nmft; v++)
    {
        float vc = ((float) v - c_mft) * du_mft;
        for (uint32_t u = 0; u < nmft; u++)
        {
            float uc   = ((float) u - c_mft) * du_mft;
            int   hits = 0;
            for (int sy = 0; sy < 8; sy++)
            {
                float vs = vc + (((float) sy + 0.5f) / 8.0f - 0.5f) * du_mft;
                for (int sx = 0; sx < 8; sx++)
                {
                    float us = uc + (((float) sx + 0.5f) / 8.0f - 0.5f) * du_mft;
                    if (us * us + vs * vs <= r_ld * r_ld)
                    {
                        hits++;
                    }
                }
            }
            float w              = (float) hits / 64.0f;
            float re             = 1.0f + w * (-0.30f - 1.0f);
            c_famp[v * nmft + u] = fabsf(re);
            c_fpha[v * nmft + u] = (re < 0.0f) ? CACAOSIM_PI_F : 0.0f;
        }
    }

    cacaosim_coro_cfg_t ref_dcfg = {
        .nwf           = nwf,
        .mode          = CACAOSIM_CORO_PHASE_DIMPLE,
        .n_mft         = nmft,
        .pup_diam      = pdia,
        .lambda_um     = 1.65f,
        .fpm_rad_ld    = r_ld,
        .fpm_amp       = 0.30f,
        .fpm_phase_rad = CACAOSIM_PI_F,
        .lowfs_refl    = 0.90f,
        .lyot_outer    = 0.85f,
        .backend       = CACAOSIM_BACKEND_CPU,
    };
    cacaosim_coro_cfg_t cus_mcfg  = ref_dcfg;
    cus_mcfg.mode                 = CACAOSIM_CORO_CUSTOM_MFT;
    cus_mcfg.fpm_pixscale_ld      = px_ld;
    cacaosim_coro_masks_t m_dimp  = { 0 };
    m_dimp.fpm_amp                = cacaosim_mask_view_2d(c_famp, nmft, nmft);
    m_dimp.fpm_pha                = cacaosim_mask_view_2d(c_fpha, nmft, nmft);

    cacaosim_coro_ctx_t ctx_ref_d, ctx_cus_m;
    cacaosim_coro_init(&ctx_ref_d, &ref_dcfg, NULL, NULL);
    cacaosim_coro_init_masks(&ctx_cus_m, &cus_mcfg, &m_dimp, NULL);

    float *o_opd1 = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *o_amp1 = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *o_opd2 = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *o_amp2 = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    cacaosim_coro_step_cpu(&ctx_ref_d, wf, pup, o_opd1, o_amp1, NULL, NULL);
    cacaosim_coro_step_cpu(&ctx_cus_m, wf, pup, o_opd2, o_amp2, NULL, NULL);
    double err_mft = rel_l2_f32(o_amp1, o_amp2, npix);

    /* Phase-only custom FFT mask reproducing analytic 4QPM */
    const uint32_t nfft    = nwf * 4u;
    float         *qpm_pha = (float *) cacaosim_aligned_calloc((size_t) nfft * nfft, sizeof(float));
    const float    c_fft   = 0.5f * (float) (nfft - 1u);
    for (uint32_t v = 0; v < nfft; v++)
    {
        float fy = (float) v - c_fft;
        for (uint32_t u = 0; u < nfft; u++)
        {
            float fx              = (float) u - c_fft;
            qpm_pha[v * nfft + u] = (fx * fy < 0.0f) ? CACAOSIM_PI_F : 0.0f;
        }
    }
    cacaosim_coro_cfg_t ref_qcfg = {
        .nwf        = nwf,
        .mode       = CACAOSIM_CORO_4QPM,
        .pad_factor = 4u,
        .pup_diam   = pdia,
        .lambda_um  = 1.65f,
        .lyot_outer = 0.85f,
        .backend    = CACAOSIM_BACKEND_CPU,
    };
    cacaosim_coro_cfg_t   cus_fcfg = ref_qcfg;
    cus_fcfg.mode                  = CACAOSIM_CORO_CUSTOM_FFT;
    cacaosim_coro_masks_t m_qpm    = { 0 };
    m_qpm.fpm_pha                  = cacaosim_mask_view_2d(qpm_pha, nfft, nfft);

    cacaosim_coro_ctx_t ctx_ref_q, ctx_cus_f;
    cacaosim_coro_init(&ctx_ref_q, &ref_qcfg, NULL, NULL);
    cacaosim_coro_init_masks(&ctx_cus_f, &cus_fcfg, &m_qpm, NULL);
    cacaosim_coro_step_cpu(&ctx_ref_q, wf, pup, o_opd1, o_amp1, NULL, NULL);
    cacaosim_coro_step_cpu(&ctx_cus_f, wf, pup, o_opd2, o_amp2, NULL, NULL);
    double err_fft = rel_l2_f32(o_amp1, o_amp2, npix);

    snprintf(detail, sizeof(detail), "MFT vs dimple=%.2e, FFT phase-only vs 4QPM=%.2e", err_mft,
             err_fft);
    check_cond(err_mft < 1e-5 && err_fft < 1e-5,
               "Custom 2D focal amp+phase MFT & FFT exact match", detail);

    cacaosim_coro_free(&ctx_cus_f);
    cacaosim_coro_free(&ctx_ref_q);
    cacaosim_coro_free(&ctx_cus_m);
    cacaosim_coro_free(&ctx_ref_d);
    free(qpm_pha);
    free(c_fpha);
    free(c_famp);

    /* 13c. Custom pupil-plane apodizer (amp + OPD) + Lyot (amp + OPD) in skip_fpm mode */
    float *p_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *p_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *l_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *l_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float  c_wf  = 0.5f * (float) (nwf - 1u);
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            float rx            = ((float) x - c_wf) / (0.5f * pdia * (float) nwf);
            float ry            = ((float) y - c_wf) / (0.5f * pdia * (float) nwf);
            float r2            = rx * rx + ry * ry;
            size_t idx          = (size_t) y * nwf + x;
            p_amp[idx]          = expf(-0.6f * r2);
            p_opd[idx]          = 0.08f * rx;
            l_amp[idx]          = (r2 <= 0.75f) ? 0.90f : 0.0f;
            l_opd[idx]          = -0.05f * ry;
        }
    }
    cacaosim_coro_cfg_t ponly_cfg = {
        .nwf           = nwf,
        .mode          = CACAOSIM_CORO_CUSTOM_MFT,
        .pup_diam      = pdia,
        .lambda_um     = 1.65f,
        .mask_pha_unit = 1, /* OPD [um] */
        .backend       = CACAOSIM_BACKEND_CPU,
    };
    cacaosim_coro_masks_t m_ponly = { 0 };
    m_ponly.pup_amp               = cacaosim_mask_view_2d(p_amp, nwf, nwf);
    m_ponly.pup_pha               = cacaosim_mask_view_2d(p_opd, nwf, nwf);
    m_ponly.lyot_amp              = cacaosim_mask_view_2d(l_amp, nwf, nwf);
    m_ponly.lyot_pha              = cacaosim_mask_view_2d(l_opd, nwf, nwf);

    cacaosim_coro_ctx_t ctx_ponly;
    cacaosim_coro_init_masks(&ctx_ponly, &ponly_cfg, &m_ponly, NULL);
    cacaosim_coro_step_cpu(&ctx_ponly, wf, pup, o_opd1, o_amp1, NULL, NULL);

    double err_pamp = 0.0;
    double err_popd = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        float exp_a = pup[i] * p_amp[i] * l_amp[i];
        err_pamp += (double) (o_amp1[i] - exp_a) * (double) (o_amp1[i] - exp_a);
        if (exp_a > 0.1f)
        {
            float exp_o = wf[i] + p_opd[i] + l_opd[i];
            err_popd += (double) (o_opd1[i] - exp_o) * (double) (o_opd1[i] - exp_o);
        }
    }
    err_pamp = sqrt(err_pamp / (double) npix);
    err_popd = sqrt(err_popd / (double) npix);
    snprintf(detail, sizeof(detail), "skip_fpm=%d, rmse_amp=%.2e, rmse_opd=%.2e um",
             ctx_ponly.skip_fpm, err_pamp, err_popd);
    check_cond(ctx_ponly.skip_fpm == 1 && err_pamp < 1e-6 && err_popd < 1e-6,
               "Custom pupil & Lyot amp+OPD apodizer (skip_fpm)", detail);
    cacaosim_coro_free(&ctx_ponly);

    /* 13d. Polychromatic 3D [Nx, Ny, Nlambda] pupil + focal cubes in .bench DAG (CPU & CUDA) */
    const uint32_t nl3          = 5u;
    const uint32_t nfft3        = nwf * 2u;
    const float    lam3[5]      = { 1.485f, 1.5675f, 1.650f, 1.7325f, 1.815f };
    float         *pup3_amp     = (float *) cacaosim_aligned_calloc(npix * nl3, sizeof(float));
    float         *fpm3_achrom  = (float *) cacaosim_aligned_calloc(
        (size_t) nfft3 * nfft3 * nl3, sizeof(float));
    float         *fpm2_chrom   = (float *) cacaosim_aligned_calloc(
        (size_t) nfft3 * nfft3, sizeof(float));
    float         *fpm3_mft_amp = (float *) cacaosim_aligned_calloc(32u * 32u * nl3, sizeof(float));
    float         *fpm3_mft_pha = (float *) cacaosim_aligned_calloc(32u * 32u * nl3, sizeof(float));

    const float c_f3 = 0.5f * (float) (nfft3 - 1u);
    for (uint32_t v = 0; v < nfft3; v++)
    {
        float fy = (float) v - c_f3;
        for (uint32_t u = 0; u < nfft3; u++)
        {
            float fx = (float) u - c_f3;
            /* Chromatic 2D half-wave plate OPD: 0.5 * lambda0 -> phase = pi * (lambda0 / lambda) */
            fpm2_chrom[v * nfft3 + u] = (fx * fy < 0.0f) ? (0.5f * 1.65f) : 0.0f;
            for (uint32_t c = 0; c < nl3; c++)
            {
                /* Achromatic 3D cube: exact half-wave OPD 0.5 * lambda_c in every slice! */
                fpm3_achrom[((size_t) c * nfft3 + v) * nfft3 + u] =
                    (fx * fy < 0.0f) ? (0.5f * lam3[c]) : 0.0f;
            }
        }
    }
    for (uint32_t c = 0; c < nl3; c++)
    {
        /* Wavelength-dependent pupil apodization cube across 3rd axis */
        float alpha = 0.15f + 0.03f * (float) c;
        for (uint32_t y = 0; y < nwf; y++)
        {
            for (uint32_t x = 0; x < nwf; x++)
            {
                float rx = ((float) x - c_wf) / (0.5f * pdia * (float) nwf);
                float ry = ((float) y - c_wf) / (0.5f * pdia * (float) nwf);
                pup3_amp[(size_t) c * npix + (size_t) y * nwf + x] =
                    1.0f - alpha * 0.2f * (rx * rx + ry * ry);
            }
        }
        /* 3D MFT focal mask cube for CPU/CUDA parity test */
        for (uint32_t v = 0; v < 32u; v++)
        {
            float vy = ((float) v - 15.5f) / 6.0f;
            for (uint32_t u = 0; u < 32u; u++)
            {
                float ux = ((float) u - 15.5f) / 6.0f;
                float r  = hypotf(ux, vy);
                size_t idx = ((size_t) c * 32u + v) * 32u + u;
                fpm3_mft_amp[idx] = (r < 2.2f) ? (0.15f + 0.04f * (float) c) : 1.0f;
                fpm3_mft_pha[idx] = (r < 2.2f) ? (0.5f * lam3[c]) : 0.0f;
            }
        }
    }

    const char *pth_p3   = "/tmp/cacaosim_val_pup3d.fits";
    const char *pth_f3   = "/tmp/cacaosim_val_fpm3d_achrom.fits";
    const char *pth_f2   = "/tmp/cacaosim_val_fpm2d_chrom.fits";
    const char *pth_m3a  = "/tmp/cacaosim_val_mft3d_amp.fits";
    const char *pth_m3p  = "/tmp/cacaosim_val_mft3d_pha.fits";
    cacaosim_mask_save_fits(pth_p3, pup3_amp, nwf, nwf, nl3, lam3);
    cacaosim_mask_save_fits(pth_f3, fpm3_achrom, nfft3, nfft3, nl3, lam3);
    cacaosim_mask_save_fits(pth_f2, fpm2_chrom, nfft3, nfft3, 1u, NULL);
    cacaosim_mask_save_fits(pth_m3a, fpm3_mft_amp, 32u, 32u, nl3, lam3);
    cacaosim_mask_save_fits(pth_m3p, fpm3_mft_pha, 32u, 32u, nl3, lam3);

    char bench_achrom[2048];
    char bench_chrom[2048];
    snprintf(bench_achrom, sizeof(bench_achrom),
             "[bench]\nname = \"poly_3d_coro\"\nnwf = %u\npup_diam = %.4f\ncobs = 0.0\n"
             "nspider = 0\nbackend = \"cpu\"\nout_beam = \"b0\"\n"
             "[coro.c0]\nin = \"b0\"\nout = \"b1\"\nmode = \"custom_fft\"\npad_factor = 2\n"
             "lambda_um = 1.65\nbw_frac = 0.20\npha_unit = \"um\"\n"
             "in_pup_amp = \"%s\"\nin_fpm_pha = \"%s\"\nlyot_outer = 0.82\nlyot_obs = 0.0\n"
             "lyot_nspider = 0\n"
             "[fpcam.sci]\nin = \"b1\"\nout_cam = \"sci_cam\"\nncam = 64\nlambda_um = 1.65\n"
             "bw_frac = 0.20\npixscale_ld = 3.0\nphoton_noise = 0\nron_e = 0.0\nbias_adu = 0.0\n"
             "out_dtype = \"float32\"\n",
             nwf, pdia, pth_p3, pth_f3);

    snprintf(bench_chrom, sizeof(bench_chrom),
             "[bench]\nname = \"poly_2d_chrom\"\nnwf = %u\npup_diam = %.4f\ncobs = 0.0\n"
             "nspider = 0\nbackend = \"cpu\"\nout_beam = \"b0\"\n"
             "[coro.c0]\nin = \"b0\"\nout = \"b1\"\nmode = \"custom_fft\"\npad_factor = 2\n"
             "lambda_um = 1.65\nbw_frac = 0.20\nnlambda = 5\npha_unit = \"um\"\n"
             "in_pup_amp = \"%s\"\nin_fpm_pha = \"%s\"\nlyot_outer = 0.82\nlyot_obs = 0.0\n"
             "lyot_nspider = 0\n"
             "[fpcam.sci]\nin = \"b1\"\nout_cam = \"sci_cam\"\nncam = 64\nlambda_um = 1.65\n"
             "bw_frac = 0.20\npixscale_ld = 3.0\nphoton_noise = 0\nron_e = 0.0\nbias_adu = 0.0\n"
             "out_dtype = \"float32\"\n",
             nwf, pdia, pth_p3, pth_f2);

    cacaosim_bench_cfg_t cfg_3d, cfg_2d;
    cacaosim_bench_ctx_t ctx_3d, ctx_2d;
    cacaosim_bench_parse_str(bench_achrom, &cfg_3d, err, sizeof(err));
    cacaosim_bench_parse_str(bench_chrom, &cfg_2d, err, sizeof(err));

    int ok_3d = (cacaosim_bench_init(&ctx_3d, &cfg_3d, NULL, 0, err, sizeof(err)) == 0);
    int ok_2d = (cacaosim_bench_init(&ctx_2d, &cfg_2d, NULL, 0, err, sizeof(err)) == 0);

    double core_3d = 1.0, core_2d = 0.0;
    if (ok_3d && ok_2d)
    {
        cacaosim_bench_step(&ctx_3d, NULL, NULL, 0.0f, NULL);
        cacaosim_bench_step(&ctx_2d, NULL, NULL, 0.0f, NULL);
        const float *psf_3d = cacaosim_bench_sensor_f32(&ctx_3d, 0);
        const float *psf_2d = cacaosim_bench_sensor_f32(&ctx_2d, 0);
        core_3d             = 0.0;
        core_2d             = 0.0;
        /* Integrate inner 2x2 central PSF core around optical axis (31..32, 31..32) */
        for (int yy = 30; yy <= 33; yy++)
        {
            for (int xx = 30; xx <= 33; xx++)
            {
                core_3d += (double) psf_3d[yy * 64 + xx];
                core_2d += (double) psf_2d[yy * 64 + xx];
            }
        }
    }
    double gain_achrom = (core_3d > 0.0) ? (core_2d / core_3d) : 0.0;
    snprintf(detail, sizeof(detail), "eff_nlam=%d, broadband core leakage ratio 2D/3D=%.1fx",
             ok_3d ? ctx_3d.nodes[0].coro_ctx.eff_nlam : 0, gain_achrom);
    check_cond(ok_3d && ok_2d && ctx_3d.nodes[0].coro_ctx.eff_nlam == 5 &&
                   ctx_3d.nodes[1].fpcam_ctx.cfg.nlambda == 5 && gain_achrom > 25.0,
               "3D polychromatic FITS mask broadband suppression", detail);

    /* 13e. 3D polychromatic CPU vs CUDA parity for both CUSTOM_FFT and CUSTOM_MFT */
    if (cacaosim_cuda_available() && ok_3d)
    {
        cacaosim_bench_cfg_t cfg_3d_gpu = cfg_3d;
        cfg_3d_gpu.backend              = CACAOSIM_BACKEND_CUDA;
        cacaosim_bench_ctx_t ctx_3d_gpu;
        int ok_g1 = (cacaosim_bench_init(&ctx_3d_gpu, &cfg_3d_gpu, NULL, 0, err, sizeof(err)) == 0);
        if (ok_g1)
        {
            cacaosim_bench_step(&ctx_3d, wf, NULL, 1.0f, NULL);
            cacaosim_bench_step(&ctx_3d_gpu, wf, NULL, 1.0f, NULL);
            double err_psf_fft = rel_l2_f32(cacaosim_bench_sensor_f32(&ctx_3d, 0),
                                            cacaosim_bench_sensor_f32(&ctx_3d_gpu, 0), 64u * 64u);
            cacaosim_bench_free(&ctx_3d_gpu);

            /* Also test 3D CUSTOM_MFT (3D pupil amp + 3D focal amp + 3D focal phase) CPU vs GPU */
            cacaosim_bench_cfg_t cfg_m3_cpu = cfg_3d;
            cfg_m3_cpu.nodes[0].coro.mode            = CACAOSIM_CORO_CUSTOM_MFT;
            cfg_m3_cpu.nodes[0].coro.fpm_pixscale_ld = 6.0f;
            snprintf(cfg_m3_cpu.nodes[0].in_fpm_amp, sizeof(cfg_m3_cpu.nodes[0].in_fpm_amp), "%s",
                     pth_m3a);
            snprintf(cfg_m3_cpu.nodes[0].in_fpm_pha, sizeof(cfg_m3_cpu.nodes[0].in_fpm_pha), "%s",
                     pth_m3p);
            cacaosim_bench_cfg_t cfg_m3_gpu = cfg_m3_cpu;
            cfg_m3_gpu.backend              = CACAOSIM_BACKEND_CUDA;

            cacaosim_bench_ctx_t ctx_mc, ctx_mg;
            cacaosim_bench_init(&ctx_mc, &cfg_m3_cpu, NULL, 0, err, sizeof(err));
            cacaosim_bench_init(&ctx_mg, &cfg_m3_gpu, NULL, 0, err, sizeof(err));
            cacaosim_bench_step(&ctx_mc, wf, NULL, 1.0f, NULL);
            cacaosim_bench_step(&ctx_mg, wf, NULL, 1.0f, NULL);
            double err_psf_mft = rel_l2_f32(cacaosim_bench_sensor_f32(&ctx_mc, 0),
                                            cacaosim_bench_sensor_f32(&ctx_mg, 0), 64u * 64u);
            cacaosim_bench_free(&ctx_mg);
            cacaosim_bench_free(&ctx_mc);

            snprintf(detail, sizeof(detail), "3D PSF rel_L2: FFT=%.2e, MFT=%.2e", err_psf_fft,
                     err_psf_mft);
            check_cond(err_psf_fft < 5e-5 && err_psf_mft < 5e-5,
                       "3D polychromatic coro+fpcam CPU vs CUDA parity", detail);
        }
    }

    if (ok_3d)
    {
        cacaosim_bench_free(&ctx_3d);
    }
    if (ok_2d)
    {
        cacaosim_bench_free(&ctx_2d);
    }
    remove(pth_p3);
    remove(pth_f3);
    remove(pth_f2);
    remove(pth_m3a);
    remove(pth_m3p);
    free(fpm3_mft_pha);
    free(fpm3_mft_amp);
    free(fpm2_chrom);
    free(fpm3_achrom);
    free(pup3_amp);
    free(l_opd);
    free(l_amp);
    free(p_opd);
    free(p_amp);
    free(o_amp2);
    free(o_opd2);
    free(o_amp1);
    free(o_opd1);
    free(wf);
    free(pup);
}

/**
 * @brief Section 14: Validate dual LOWFS pickoffs, SI [meter] static WFE & TUI pixel scales.
 *
 * Role:
 *   Validates three ExAO hardware/UI subsystems:
 *     - 14a (`CACAOSIM_LOWFS_LYOT_REFL`): Confirms pure-phase Vortex FPM reflects zero focal light
 *       (`< 1e-8`) under `fpm_refl` while `lyot_refl` captures `> 25%` of total pupil energy from
 *       the rejected outer Lyot ring, and verifies exact two-port power conservation
 *       `|A_sci|^2 + |A_lowfs|^2 == |A_in|^2` (`rmse < 1e-6`).
 *     - 14b (`CACAOSIM_LOWFS_FPM_REFL` + `lowfs_pup_diam`): Confirms monotonic energy filtering
 *       (`E(1.00D) > E(0.75D) > E(0.45D)`) and zero exterior leakage (`< 1e-12`) when stopping down
 *       the LOWFS relay pupil stop on a partially transmitting (`fpm_amp = 0.20`) Lyot spot.
 *     - 14c: CPU vs CUDA parity (`rel_L2 < 5e-5`) across MFT `fpm_refl`, FFT `fpm_refl`, and FFT
 *       `lyot_refl` modes.
 *     - 14d (`cacaosim_make_wfe_m` & `[wfe.<id>]`): Verifies SI `[meter]` power-law WFE synthesis
 *       (`80 nm = 8.0e-8 m` RMS), `.bench` round-trip serialization, Maréchal Strehl agreement
 *       (`exp(-(2*pi*0.080/1.65)^2) = 0.911` vs `1.000` at `wfe_gain = 0`), CUDA parity, and
 *       topological add/clone/move/remove editing.
 *     - 14e: Verifies `cacaosim-CTRL` pupil `[m/pix]` and focal `[l/D/pix]` & `[arcsec/pix]` panel.
 *
 * Rationale:
 *   Supports both reflective-spot LOWFS benches (SCExAO Lyot/alc) and reflective-Lyot-stop vortex
 *   benches (KPIC/MagAO-X/SCExAO VVC), ensures error-free SI `[meter]` WFE interoperability with
 *   external optical modeling tools, and locks down physical plate-scale readouts in the TUI.
 */
static void test_coro_lowfs_wfe_and_scales(void)
{
    printf("\n=== 14. CORO+LOWFS Dual Pickoff, Static WFE [OPD m] & TUI Pixel Scales ===\n");
    char detail[160];
    char err[256];

    const uint32_t nwf  = 64u;
    const uint64_t npix = (uint64_t) nwf * nwf;
    const float    pdia = 0.875f;

    float *pup    = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *wf     = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *sc_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *sc_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *lo_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float *lo_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = pdia, .cobs = 0.0f };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, pdia, pup, 0.05f, 4040ULL, wf);

    /* 14a. Option (1): Reflective Lyot stop (lyot_refl) vs Option (2) on phase-only Vortex */
    cacaosim_coro_cfg_t vcfg_fpm = {
        .nwf            = nwf,
        .mode           = CACAOSIM_CORO_VORTEX,
        .pad_factor     = 4u,
        .pup_diam       = pdia,
        .lambda_um      = 1.65f,
        .vortex_charge  = 2,
        .lowfs_mode     = CACAOSIM_LOWFS_FPM_REFL,
        .lowfs_pup_diam = 1.0f,
        .lowfs_refl     = 0.90f,
        .lyot_outer     = 0.82f,
        .backend        = CACAOSIM_BACKEND_CPU,
    };
    cacaosim_coro_cfg_t vcfg_lyot = vcfg_fpm;
    vcfg_lyot.lowfs_mode          = CACAOSIM_LOWFS_LYOT_REFL;

    cacaosim_coro_ctx_t vctx_fpm, vctx_lyot;
    cacaosim_coro_init(&vctx_fpm, &vcfg_fpm, NULL, NULL);
    cacaosim_coro_init(&vctx_lyot, &vcfg_lyot, NULL, NULL);

    memset(wf, 0, npix * sizeof(float));
    cacaosim_coro_step_cpu(&vctx_fpm, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
    double e_vortex_fpm = 0.0;
    for (uint64_t p = 0; p < npix; p++)
    {
        e_vortex_fpm += (double) lo_amp[p] * (double) lo_amp[p];
    }

    cacaosim_coro_step_cpu(&vctx_lyot, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
    double e_vortex_lyot = 0.0;
    double e_pup_in      = 0.0;
    for (uint64_t p = 0; p < npix; p++)
    {
        e_vortex_lyot += (double) lo_amp[p] * (double) lo_amp[p];
        e_pup_in += (double) pup[p] * (double) pup[p];
    }
    cacaosim_coro_free(&vctx_fpm);
    cacaosim_coro_free(&vctx_lyot);

    /* Also verify exact Lyot power partition |L|^2 + (1 - |L|^2) == 1 in skip_fpm mode */
    float               *p_apod     = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    cacaosim_coro_cfg_t  sk_cfg     = vcfg_lyot;
    sk_cfg.mode                     = CACAOSIM_CORO_CUSTOM_MFT;
    sk_cfg.lowfs_refl               = 1.0f;
    for (uint64_t p = 0; p < npix; p++)
    {
        p_apod[p] = 1.0f;
    }
    cacaosim_coro_masks_t sk_m = { 0 };
    sk_m.pup_amp               = cacaosim_mask_view_2d(p_apod, nwf, nwf);
    cacaosim_coro_ctx_t sk_ctx;
    cacaosim_coro_init_masks(&sk_ctx, &sk_cfg, &sk_m, NULL);
    cacaosim_make_ncpa(nwf, pdia, pup, 0.05f, 4040ULL, wf);
    cacaosim_coro_step_cpu(&sk_ctx, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
    double lyot_part_err = 0.0;
    for (uint64_t p = 0; p < npix; p++)
    {
        double ein  = (double) pup[p] * (double) pup[p];
        double esum = (double) sc_amp[p] * (double) sc_amp[p] +
                      (double) lo_amp[p] * (double) lo_amp[p];
        lyot_part_err += (esum - ein) * (esum - ein);
    }
    lyot_part_err = sqrt(lyot_part_err / (double) npix);
    cacaosim_coro_free(&sk_ctx);
    free(p_apod);

    snprintf(detail, sizeof(detail),
             "Vortex fpm/Ein=%.1e, lyot_refl/Ein=%.3f, part_rmse=%.2e",
             e_vortex_fpm / e_pup_in, e_vortex_lyot / e_pup_in, lyot_part_err);
    check_cond((e_vortex_fpm / e_pup_in) < 1e-8 && (e_vortex_lyot / e_pup_in) > 0.25 &&
                   lyot_part_err < 1e-6,
               "CORO Option 1: Reflective Lyot stop LOWFS", detail);

    /* 14b. Option (2): Reflective FPM (absorbed light reflected) + lowfs_pup_diam stop */
    cacaosim_coro_cfg_t lcfg_full = {
        .nwf            = nwf,
        .mode           = CACAOSIM_CORO_LYOT_SPOT,
        .n_mft          = 40u,
        .pup_diam       = pdia,
        .lambda_um      = 1.65f,
        .fpm_rad_ld     = 3.0f,
        .fpm_amp        = 0.20f,
        .lowfs_mode     = CACAOSIM_LOWFS_FPM_REFL,
        .lowfs_pup_diam = 1.00f,
        .lowfs_refl     = 1.00f,
        .lyot_outer     = 0.85f,
        .backend        = CACAOSIM_BACKEND_CPU,
    };
    cacaosim_coro_cfg_t lcfg_mid  = lcfg_full;
    lcfg_mid.lowfs_pup_diam       = 0.75f;
    cacaosim_coro_cfg_t lcfg_stop = lcfg_full;
    lcfg_stop.lowfs_pup_diam      = 0.45f;

    cacaosim_coro_ctx_t lctx_full, lctx_mid, lctx_stop;
    cacaosim_coro_init(&lctx_full, &lcfg_full, NULL, NULL);
    cacaosim_coro_init(&lctx_mid, &lcfg_mid, NULL, NULL);
    cacaosim_coro_init(&lctx_stop, &lcfg_stop, NULL, NULL);

    cacaosim_coro_step_cpu(&lctx_full, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
    double e_lo_100 = 0.0;
    for (uint64_t p = 0; p < npix; p++)
    {
        e_lo_100 += (double) lo_amp[p] * (double) lo_amp[p];
    }

    cacaosim_coro_step_cpu(&lctx_mid, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
    double e_lo_075 = 0.0;
    for (uint64_t p = 0; p < npix; p++)
    {
        e_lo_075 += (double) lo_amp[p] * (double) lo_amp[p];
    }

    cacaosim_coro_step_cpu(&lctx_stop, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
    double e_lo_045    = 0.0;
    double leak_outside = 0.0;
    float  c_wf        = 0.5f * (float) (nwf - 1u);
    float  r_stop_px   = 0.5f * pdia * 0.45f * (float) nwf + 1.5f;
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            uint64_t p = (uint64_t) y * nwf + x;
            double   e = (double) lo_amp[p] * (double) lo_amp[p];
            e_lo_045 += e;
            if (hypotf((float) x - c_wf, (float) y - c_wf) > r_stop_px)
            {
                leak_outside += e;
            }
        }
    }
    cacaosim_coro_free(&lctx_stop);
    cacaosim_coro_free(&lctx_mid);
    cacaosim_coro_free(&lctx_full);

    snprintf(detail, sizeof(detail), "E(1.00D)=%.1f > E(0.75D)=%.1f > E(0.45D)=%.1f, leak=%.1e",
             e_lo_100, e_lo_075, e_lo_045, leak_outside);
    check_cond(e_lo_100 > e_lo_075 && e_lo_075 > e_lo_045 && e_lo_045 > 10.0 &&
                   leak_outside < 1e-12,
               "CORO Option 2: FPM-refl + lowfs_pup_diam stop", detail);

    /* 14c. CPU vs CUDA parity on both LOWFS modes (MFT fpm_refl, FFT fpm_refl, FFT lyot_refl) */
    if (cacaosim_cuda_available())
    {
        float *g_sc_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *g_sc_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *g_lo_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *g_lo_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));

        /* 1) MFT Lyot spot + fpm_refl + lowfs_pup_diam=0.72 */
        cacaosim_coro_cfg_t c1_cpu = lcfg_mid;
        c1_cpu.lowfs_pup_diam      = 0.72f;
        cacaosim_coro_cfg_t c1_gpu = c1_cpu;
        c1_gpu.backend             = CACAOSIM_BACKEND_CUDA;
        cacaosim_coro_ctx_t x1_c, x1_g;
        cacaosim_coro_init(&x1_c, &c1_cpu, NULL, NULL);
        cacaosim_coro_init(&x1_g, &c1_gpu, NULL, NULL);
        cacaosim_coro_step_cpu(&x1_c, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
        cacaosim_coro_step_beam(&x1_g, wf, pup, g_sc_opd, g_sc_amp, g_lo_opd, g_lo_amp, NULL, NULL,
                                NULL, NULL, NULL, NULL);
        double err_mft_fpm = rel_l2_f32(lo_amp, g_lo_amp, npix);
        cacaosim_coro_free(&x1_g);
        cacaosim_coro_free(&x1_c);

        /* 2) Custom FFT focal amplitude spot + fpm_refl + lowfs_pup_diam=0.75 */
        const uint32_t nfft2 = nwf * 2u;
        float         *fft_a = (float *) cacaosim_aligned_calloc(
            (size_t) nfft2 * nfft2, sizeof(float));
        float cf2 = 0.5f * (float) (nfft2 - 1u);
        for (uint32_t v = 0; v < nfft2; v++)
        {
            for (uint32_t u = 0; u < nfft2; u++)
            {
                float r               = hypotf((float) u - cf2, (float) v - cf2);
                fft_a[v * nfft2 + u]  = (r <= 5.5f) ? 0.15f : 1.0f;
            }
        }
        cacaosim_coro_cfg_t c2_cpu = {
            .nwf            = nwf,
            .mode           = CACAOSIM_CORO_CUSTOM_FFT,
            .pad_factor     = 2u,
            .pup_diam       = pdia,
            .lambda_um      = 1.65f,
            .lowfs_mode     = CACAOSIM_LOWFS_FPM_REFL,
            .lowfs_pup_diam = 0.75f,
            .lowfs_refl     = 0.95f,
            .lyot_outer     = 0.84f,
            .backend        = CACAOSIM_BACKEND_CPU,
        };
        cacaosim_coro_cfg_t   c2_gpu = c2_cpu;
        c2_gpu.backend               = CACAOSIM_BACKEND_CUDA;
        cacaosim_coro_masks_t m2     = { 0 };
        m2.fpm_amp                   = cacaosim_mask_view_2d(fft_a, nfft2, nfft2);
        cacaosim_coro_ctx_t x2_c, x2_g;
        cacaosim_coro_init_masks(&x2_c, &c2_cpu, &m2, NULL);
        cacaosim_coro_init_masks(&x2_g, &c2_gpu, &m2, NULL);
        cacaosim_coro_step_cpu(&x2_c, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
        cacaosim_coro_step_beam(&x2_g, wf, pup, g_sc_opd, g_sc_amp, g_lo_opd, g_lo_amp, NULL, NULL,
                                NULL, NULL, NULL, NULL);
        double err_fft_fpm = rel_l2_f32(lo_amp, g_lo_amp, npix);
        cacaosim_coro_free(&x2_g);
        cacaosim_coro_free(&x2_c);
        free(fft_a);

        /* 3) Vortex FFT + lyot_refl */
        cacaosim_coro_cfg_t c3_cpu = vcfg_lyot;
        cacaosim_coro_cfg_t c3_gpu = c3_cpu;
        c3_gpu.backend             = CACAOSIM_BACKEND_CUDA;
        cacaosim_coro_ctx_t x3_c, x3_g;
        cacaosim_coro_init(&x3_c, &c3_cpu, NULL, NULL);
        cacaosim_coro_init(&x3_g, &c3_gpu, NULL, NULL);
        cacaosim_coro_step_cpu(&x3_c, wf, pup, sc_opd, sc_amp, lo_opd, lo_amp);
        cacaosim_coro_step_beam(&x3_g, wf, pup, g_sc_opd, g_sc_amp, g_lo_opd, g_lo_amp, NULL, NULL,
                                NULL, NULL, NULL, NULL);
        double err_fft_lyot = rel_l2_f32(lo_amp, g_lo_amp, npix);
        cacaosim_coro_free(&x3_g);
        cacaosim_coro_free(&x3_c);

        snprintf(detail, sizeof(detail), "MFT_fpm=%.2e, FFT_fpm=%.2e, FFT_lyot=%.2e",
                 err_mft_fpm, err_fft_fpm, err_fft_lyot);
        check_cond(err_mft_fpm < 5e-5 && err_fft_fpm < 5e-5 && err_fft_lyot < 5e-5,
                   "CORO+LOWFS dual modes CPU vs CUDA parity", detail);

        free(g_lo_amp);
        free(g_lo_opd);
        free(g_sc_amp);
        free(g_sc_opd);
    }

    /* 14d. Static Wavefront Error [OPD m]: synthesis, Maréchal Strehl, .bench round-trip & CUDA */
    float *wfe_m = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    const float target_rms_m = 80.0e-9f;
    cacaosim_make_wfe_m(nwf, pdia, pup, target_rms_m, 1.6f, 2026ULL, wfe_m);

    double w_sum = 0.0, w_mean = 0.0, w_var = 0.0;
    for (uint64_t p = 0; p < npix; p++)
    {
        double w = (double) pup[p];
        w_sum += w;
        w_mean += w * (double) wfe_m[p];
    }
    w_mean /= w_sum;
    for (uint64_t p = 0; p < npix; p++)
    {
        double w = (double) pup[p];
        double d = (double) wfe_m[p] - w_mean;
        w_var += w * d * d;
    }
    double meas_rms_m = sqrt(w_var / w_sum);
    double rms_rel    = fabs(meas_rms_m - (double) target_rms_m) / (double) target_rms_m;
    snprintf(detail, sizeof(detail), "target=%.2e m, measured=%.2e m, rel_err=%.2e",
             (double) target_rms_m, meas_rms_m, rms_rel);
    check_cond(rms_rel < 1e-5 && fabs(w_mean) < 1e-12,
               "cacaosim_make_wfe_m pupil RMS in [meter]", detail);

    const char *wfe_bench_txt =
        "[bench]\n"
        "name = \"wfe_opd_meter_test\"\n"
        "nwf = 64\n"
        "pup_diam = 0.875\n"
        "cobs = 0.0\n"
        "nspider = 0\n"
        "backend = \"cpu\"\n"
        "out_beam = \"b_in\"\n"
        "[wfe.stat0]\n"
        "in = \"b_in\"\n"
        "out = \"b_wfe\"\n"
        "in_wfe = \"csim_val_wfe_m\"\n"
        "rms_nm = 80.0\n"
        "psd_alpha = 1.6\n"
        "wfe_gain = 1.0\n"
        "wfe_seed = 2026\n"
        "[coro.c0]\n"
        "in = \"b_wfe\"\n"
        "out = \"b_sci\"\n"
        "out_lowfs = \"b_lo\"\n"
        "mode = \"lyot_spot\"\n"
        "lowfs_mode = \"lyot_refl\"\n"
        "lowfs_pup_diam = 0.80\n"
        "[zwfs.lo]\n"
        "in = \"b_lo\"\n"
        "out_cam = \"lo_cam\"\n"
        "ncam = 32\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "out_dtype = \"float32\"\n"
        "[fpcam.sci]\n"
        "in = \"b_wfe\"\n"
        "out_cam = \"sci_cam\"\n"
        "ncam = 64\n"
        "lambda_um = 1.65\n"
        "bw_frac = 0.01\n"
        "nlambda = 1\n"
        "pixscale_ld = 3.0\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n";

    cacaosim_bench_cfg_t wcfg0, wcfg1;
    char                 wfmt[8192];
    int wp_ok = (cacaosim_bench_parse_str(wfe_bench_txt, &wcfg0, err, sizeof(err)) == 0);
    int wf_ln = wp_ok ? cacaosim_bench_format_str(&wcfg0, wfmt, sizeof(wfmt)) : -1;
    int wr_ok = (wf_ln > 0 && cacaosim_bench_parse_str(wfmt, &wcfg1, err, sizeof(err)) == 0 &&
                 wcfg1.nodes[0].kind == CACAOSIM_NODE_WFE &&
                 fabsf(wcfg1.nodes[0].wfe_rms_nm - 80.0f) < 1e-4f &&
                 wcfg1.nodes[1].coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL &&
                 fabsf(wcfg1.nodes[1].coro.lowfs_pup_diam - 0.80f) < 1e-4f);

    cacaosim_bench_ctx_t wctx;
    int wi_ok = wr_ok && (cacaosim_bench_init(&wctx, &wcfg1, NULL, 0, err, sizeof(err)) == 0);
    float sr_80nm = 0.0f;
    float sr_zero = 0.0f;
    double sr_mar = exp(-pow(2.0 * M_PI * 0.080 / 1.65, 2.0));
    double wfe_gpu_err = 0.0;

    if (wi_ok)
    {
        cacaosim_bench_step(&wctx, NULL, NULL, 0.0f, NULL);
        sr_80nm = wctx.last_strehl_inst;

        wctx.nodes[0].cfg.wfe_gain = 0.0f;
        cacaosim_bench_step(&wctx, NULL, NULL, 0.0f, NULL);
        sr_zero = wctx.last_strehl_inst;
        wctx.nodes[0].cfg.wfe_gain = 1.0f;
        cacaosim_bench_step(&wctx, NULL, NULL, 0.0f, NULL);

        if (cacaosim_cuda_available())
        {
            cacaosim_bench_cfg_t wcfg_gpu = wcfg1;
            wcfg_gpu.backend              = CACAOSIM_BACKEND_CUDA;
            cacaosim_bench_ctx_t wctx_gpu;
            if (cacaosim_bench_init(&wctx_gpu, &wcfg_gpu, NULL, 0, err, sizeof(err)) == 0)
            {
                cacaosim_bench_step(&wctx_gpu, NULL, NULL, 0.0f, NULL);
                int s_sci = cacaosim_bench_find_sensor(&wctx, "sci");
                wfe_gpu_err = rel_l2_f32(cacaosim_bench_sensor_f32(&wctx, s_sci),
                                         cacaosim_bench_sensor_f32(&wctx_gpu, s_sci), 64u * 64u);
                cacaosim_bench_free(&wctx_gpu);
            }
        }
        cacaosim_bench_free(&wctx);
    }

    /* Also test topological add/move/clone/remove of CACAOSIM_NODE_WFE */
    cacaosim_bench_cfg_t ecfg;
    cacaosim_bench_ctx_t ectx;
    cacaosim_bench_preset(&ecfg, CACAOSIM_PRESET_SINGLE_PYRWFS);
    int iw   = cacaosim_bench_add_node(&ecfg, CACAOSIM_NODE_WFE, 0, 0, NULL);
    int ic   = (iw >= 0) ? cacaosim_bench_clone_node(&ecfg, (uint32_t) iw) : -1;
    int im   = (iw >= 0) ? cacaosim_bench_move_node(&ecfg, (uint32_t) iw, -1) : -1;
    int ed_ok = (iw >= 0 && ic >= 0 && im == 0 &&
                 cacaosim_bench_analyze(&ectx, &ecfg, err, sizeof(err)) == 0 &&
                 cacaosim_bench_remove_node(&ecfg, 0u) == 0 &&
                 cacaosim_bench_analyze(&ectx, &ecfg, err, sizeof(err)) == 0);

    snprintf(detail, sizeof(detail), "SR(0)=%.3f, SR(80nm)=%.3f (Mar=%.3f), gpu_err=%.2e, ed=%d",
             sr_zero, sr_80nm, sr_mar, wfe_gpu_err, ed_ok);
    check_cond(wi_ok && ed_ok && fabsf(sr_zero - 1.0f) < 1e-4f &&
                   fabs((double) sr_80nm - sr_mar) < 0.015 && wfe_gpu_err < 5e-5,
               "Static WFE [wfe.<id>] OPD [m] DAG, CUDA & edit", detail);

    /* 14e. cacaosim-CTRL lower-left pupil panel: Array, Beam [pix], [m/pix], [l/D/pix], ["/pix] */
    cacaosim_bench_cfg_t scfg;
    cacaosim_bench_preset(&scfg, CACAOSIM_PRESET_TWO_STAGE);
    /* TWO_STAGE: nwf=128, pup_diam=0.9375 (120.0 pix), tel_diam_m=8.2 (0.0683 m/pix),
     * sci_h: lambda_um=1.65, pixscale_ld=3.0 -> 0.333 l/D/pix, 0.0138 arcsec/pix @1.65um */
    char *sbuf = (char *) malloc(65536u);
    int   sc_ok = 0;
    if (sbuf != NULL)
    {
        int n_b = cacaosim_tui_render_headless(&scfg, CACAOSIM_TUI_VIEW_BUILDER, 128, 38, 1, sbuf,
                                               65536u);
        int ok_b = (n_b > 1500 && strstr(sbuf, "128x128 pix") != NULL &&
                    strstr(sbuf, "120.0 pix") != NULL && strstr(sbuf, "0.0683 m/pix") != NULL &&
                    strstr(sbuf, "0.333 l/D/pix") != NULL &&
                    strstr(sbuf, "0.0138 arcsec/pix @1.65um") != NULL);

        int n_s = cacaosim_tui_render_headless(&scfg, CACAOSIM_TUI_VIEW_SCHEMATIC, 128, 38, 1, sbuf,
                                               65536u);
        int ok_s = (n_s > 1500 && strstr(sbuf, "128x128 pix") != NULL &&
                    strstr(sbuf, "120.0 pix") != NULL && strstr(sbuf, "0.0683 m/pix") != NULL &&
                    strstr(sbuf, "0.333 l/D/pix") != NULL &&
                    strstr(sbuf, "0.0138 arcsec/pix @1.65um") != NULL);
        sc_ok = ok_b && ok_s;
        snprintf(detail, sizeof(detail),
                 "128x128 pix, 120.0 pix, 0.0683 m/pix, 0.333 l/D/pix, 0.0138 arcsec/pix");
        free(sbuf);
    }
    check_cond(sc_ok, "cacaosim-CTRL pupil & focal pixel scale panel", detail);

    free(wfe_m);
    free(lo_amp);
    free(lo_opd);
    free(sc_amp);
    free(sc_opd);
    free(wf);
    free(pup);
}

/**
 * @brief Entry point executing all 14 cacaosim physical, statistical, and closed-loop test suites.
 *
 * Role:
 *   Reports runtime CUDA availability, invokes every `test_*()` verification stage in logical
 *   bottom-up order (primitives -> sensors -> single-stage closed loop -> multi-stage DAGs -> TUI
 *   & custom 3D masks), prints aggregate pass/fail totals, and returns CTest-compatible status.
 *
 * Rationale:
 *   Ordering tests from low-level aperture/detector/DM primitives up to full multi-stage cascaded
 *   closed-loop AO makes root-cause localization immediate whenever any assertion fails.
 *
 * @return `0` if every assertion passed (`g_fail == 0`); `1` if one or more assertions failed.
 */
/**
 * @brief Section 14: CPU vs CUDA parity of the shared wavefront statistics engine.
 *
 * Feeds identical deterministic pseudo-random OPD frames (with per-frame piston offsets) to the
 * CPU path (`cacaosim.wfstat`) and the CUDA in-DAG path (`csim_htsim.wfstat_enable`) of
 * `cacaosim_wfstat_core`, then compares FP64 per-pixel sums, per-frame RMS accumulators, and
 * the 5 captured sample maps. Both paths must agree to round-off since htsim replaces csim_wfstat
 * in fused mode.
 */
static void test_wfstat_cpu_gpu_parity(void)
{
    printf("\n[14] Wavefront statistics engine: CPU vs CUDA parity\n");
    if (!cacaosim_cuda_available())
    {
        printf("  [SKIP] No CUDA GPU available\n");
        return;
    }
    const uint32_t nwf = 128, nframes = 200;
    const uint64_t npix = (uint64_t) nwf * nwf;
    float         *pup  = (float *) calloc(npix, sizeof(float));
    float         *h_fr = (float *) calloc(3 * npix, sizeof(float));
    cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = 0.9375f, .cobs = 0.30f,
                                  .nspider = 4, .spider_w = 0.01f, .spider_deg = 45.0f };
    cacaosim_make_pupil(&pcfg, pup);

    void  *stream = cacaosim_cuda_stream_create();
    float *d_fr[3];
    for (int k = 0; k < 3; k++)
    {
        d_fr[k] = cacaosim_cuda_alloc_f32(npix);
    }
    cacaosim_wfstat_t wc, wg;
    int rc_c = cacaosim_wfstat_init(&wc, nwf, pup, nframes, 0, 0, NULL);
    int rc_g = cacaosim_wfstat_init(&wg, nwf, pup, nframes, 0, 1, stream);
    const uint64_t clip_s0  = 95ULL;
    const uint32_t clip_len = 16u;
    const double   full_stp = 10.0;
    const uint32_t full_cap = 32u;
    if (rc_c == 0)
    {
        rc_c = cacaosim_wfstat_enable_clip(&wc, clip_s0, clip_len);
    }
    if (rc_c == 0)
    {
        rc_c = cacaosim_wfstat_enable_full(&wc, full_stp, full_cap);
    }
    if (rc_g == 0)
    {
        rc_g = cacaosim_wfstat_enable_clip(&wg, clip_s0, clip_len);
    }
    if (rc_g == 0)
    {
        rc_g = cacaosim_wfstat_enable_full(&wg, full_stp, full_cap);
    }
    check_cond(rc_c == 0 && rc_g == 0 && d_fr[2] != NULL, "wfstat CPU + CUDA engines initialize",
               rc_g == 0 ? "ok" : "gpu init failed");
    if (rc_c != 0 || rc_g != 0)
    {
        free(pup);
        free(h_fr);
        return;
    }

    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    for (uint32_t f = 0; f < nframes; f++)
    {
        for (int k = 0; k < 3; k++)
        {
            float piston = 0.05f * (float) (k + 1) * sinf(0.1f * (float) f);
            for (uint64_t i = 0; i < npix; i++)
            {
                seed            = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                float u         = (float) ((seed >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f;
                h_fr[k * npix + i] = piston + (0.6f / (float) (k + 1)) * u;
            }
            cacaosim_cuda_h2d_f32(d_fr[k], h_fr + k * npix, npix, stream);
        }
        const float *hc[3] = { h_fr, h_fr + npix, h_fr + 2 * npix };
        const float *dg[3] = { d_fr[0], d_fr[1], d_fr[2] };
        cacaosim_wfstat_frame(&wc, hc);
        cacaosim_wfstat_frame(&wg, dg);
        cacaosim_cuda_stream_sync(stream);
    }

    double   acc[6];
    uint64_t gpu_ts_len = 0;
    int      rc_f = cacaosim_wfstat_gpu_fetch_sums(wg.gpu, wg.sum_wf, wg.sum2_wf, acc,
                                                   wg.ts_rms, &gpu_ts_len);
    wg.ts_len     = gpu_ts_len;
    double max_rel_sum = 0.0, max_rel_acc = 0.0, max_abs_samp = 0.0, max_rel_ptv = 0.0;
    double max_rel_ts  = 0.0;
    for (int k = 0; k < 3; k++)
    {
        for (uint64_t i = 0; i < npix; i++)
        {
            double ref = fabs(wc.sum2_wf[k][i]) + 1e-12;
            double d1  = fabs(wg.sum_wf[k][i] - wc.sum_wf[k][i]) / sqrt(ref * nframes);
            double d2  = fabs(wg.sum2_wf[k][i] - wc.sum2_wf[k][i]) / ref;
            max_rel_sum = fmax(max_rel_sum, fmax(d1, d2));
        }
        max_rel_acc = fmax(max_rel_acc, fabs(acc[2 * k] - wc.sum_srms[k]) / wc.sum_srms[k]);
        max_rel_acc = fmax(max_rel_acc, fabs(acc[2 * k + 1] - wc.sum_svar[k]) / wc.sum_svar[k]);
        for (uint32_t f = 0; f < nframes; f++)
        {
            double dt = fabs((double) wg.ts_rms[k][f] - (double) wc.ts_rms[k][f]) /
                        fmax((double) wc.ts_rms[k][f], 1e-9);
            max_rel_ts = fmax(max_rel_ts, dt);
        }
        for (int s = 0; s < wc.nsamp; s++)
        {
            max_rel_ptv = fmax(max_rel_ptv, fabs(wg.samp_sptv[s][k] - wc.samp_sptv[s][k]) /
                                                wc.samp_sptv[s][k]);
            for (uint64_t i = 0; i < npix; i++)
            {
                double d = fabs((double) wg.samp_wf[k][s * npix + i] -
                                (double) wc.samp_wf[k][s * npix + i]);
                max_abs_samp = fmax(max_abs_samp, d);
            }
        }
    }
    char det[160];
    snprintf(det, sizeof(det), "frames %lu/%lu, samples %d/%d, ts %lu/%lu",
             (unsigned long) wg.n_frames, (unsigned long) wc.n_frames, wg.nsamp, wc.nsamp,
             (unsigned long) wg.ts_len, (unsigned long) wc.ts_len);
    check_cond(rc_f == 0 && wg.n_frames == nframes && wg.nsamp == 5 && wc.nsamp == 5 &&
                   wg.ts_len == nframes && wc.ts_len == nframes,
               "wfstat GPU/CPU frame + sample + ts bookkeeping", det);
    snprintf(det, sizeof(det), "max rel = %.2e", max_rel_sum);
    check_cond(max_rel_sum < 1e-9, "wfstat per-pixel FP64 sum/sum2 parity", det);
    snprintf(det, sizeof(det), "max rel = %.2e", max_rel_acc);
    check_cond(max_rel_acc < 1e-9, "wfstat per-frame RMS/variance accumulators", det);
    snprintf(det, sizeof(det), "max rel = %.2e", max_rel_ts);
    check_cond(max_rel_ts < 1e-6, "wfstat per-frame RMS time-series GPU/CPU parity", det);
    snprintf(det, sizeof(det), "max |dmap| = %.2e nm, max rel dPTV = %.2e", max_abs_samp,
             max_rel_ptv);
    check_cond(max_abs_samp < 1e-3 && max_rel_ptv < 1e-9, "wfstat sample maps + PTV parity", det);

    double max_abs_clip = 0.0;
    if (wc.clip_n == clip_len && wg.clip_n == clip_len)
    {
        for (uint32_t ci = 0; ci < clip_len; ci++)
        {
            for (uint64_t i = 0; i < npix; i++)
            {
                double dr = fabs((double) wg.clip_wf_raw[ci * npix + i] -
                                 (double) wc.clip_wf_raw[ci * npix + i]);
                double df = fabs((double) wg.clip_wf_final[ci * npix + i] -
                                 (double) wc.clip_wf_final[ci * npix + i]);
                max_abs_clip = fmax(max_abs_clip, fmax(dr, df));
            }
        }
    }
    snprintf(det, sizeof(det), "clip_n=%u/%u (expect %u), max |dclip| = %.2e nm", wg.clip_n,
             wc.clip_n, clip_len, max_abs_clip);
    check_cond(wc.clip_n == clip_len && wg.clip_n == clip_len && max_abs_clip < 1e-3 &&
                   wg.clip_frame[0] == clip_s0 && wg.clip_frame[clip_len - 1] == clip_s0 + clip_len - 1,
               "wfstat contiguous short-timescale clip GPU/CPU parity", det);

    double max_abs_full = 0.0;
    if (wc.full_n > 0u && wc.full_n == wg.full_n)
    {
        for (uint32_t fi = 0; fi < wg.full_n; fi++)
        {
            for (uint64_t i = 0; i < npix; i++)
            {
                double dr = fabs((double) wg.full_wf_raw[fi * npix + i] -
                                 (double) wc.full_wf_raw[fi * npix + i]);
                double df = fabs((double) wg.full_wf_final[fi * npix + i] -
                                 (double) wc.full_wf_final[fi * npix + i]);
                max_abs_full = fmax(max_abs_full, fmax(dr, df));
            }
        }
    }
    snprintf(det, sizeof(det), "full_n=%u/%u, first=%llu, max |dfull| = %.2e nm", wg.full_n,
             wc.full_n, (unsigned long long) ((wg.full_n > 0u) ? wg.full_frame[0] : 0ULL),
             max_abs_full);
    check_cond(wc.full_n >= 10u && wg.full_n == wc.full_n && max_abs_full < 1e-3 &&
                   wg.full_frame[0] == 1ULL,
               "wfstat strided full-duration capture GPU/CPU parity", det);

    char tmpd[128], pbuf[256];
    snprintf(tmpd, sizeof(tmpd), "/tmp/cacaosim_wfstat_ut_%ld", (long) getpid());
    int rc_w = cacaosim_wfstat_write(&wg, tmpd, 0.0005);
    snprintf(pbuf, sizeof(pbuf), "%s/wf_timeseries.csv", tmpd);
    FILE    *cfp   = fopen(pbuf, "r");
    uint32_t lines = 0;
    if (cfp != NULL)
    {
        char lbuf[256];
        while (fgets(lbuf, sizeof(lbuf), cfp) != NULL)
        {
            lines++;
        }
        fclose(cfp);
    }
    snprintf(pbuf, sizeof(pbuf), "%s/wf_stats.json", tmpd);
    FILE *jfp  = fopen(pbuf, "r");
    int   j_ok = (jfp != NULL);
    if (jfp != NULL)
    {
        fclose(jfp);
    }
    snprintf(pbuf, sizeof(pbuf), "%s/wf_clip_meta.json", tmpd);
    FILE *cmfp  = fopen(pbuf, "r");
    int   cm_ok = (cmfp != NULL);
    if (cmfp != NULL)
    {
        fclose(cmfp);
    }
    snprintf(pbuf, sizeof(pbuf), "%s/wf_full_meta.json", tmpd);
    FILE *fmfp  = fopen(pbuf, "r");
    int   fm_ok = (fmfp != NULL);
    if (fmfp != NULL)
    {
        fclose(fmfp);
    }
    snprintf(det, sizeof(det), "write rc=%d, csv lines=%u (expect %u), clip=%d, full=%d", rc_w,
             lines, nframes + 1, cm_ok, fm_ok);
    check_cond(rc_w == 0 && j_ok && cm_ok && fm_ok && lines == nframes + 1,
               "wfstat writes wf_stats.json + csv + wf_clip_meta + wf_full_meta", det);

    static const char *const ut_files[] = {
        "wf_pupil_mask.fits", "wf_stats.json",     "wf_timeseries.csv",
        "wf_mean_raw.fits",   "wf_std_raw.fits",   "wf_samples_raw.fits",
        "wf_mean_stg1.fits",  "wf_std_stg1.fits",  "wf_samples_stg1.fits",
        "wf_mean_final.fits", "wf_std_final.fits", "wf_samples_final.fits",
        "wf_clip_raw.fits",   "wf_clip_final.fits", "wf_clip_meta.json",
        "wf_full_raw.fits",   "wf_full_final.fits", "wf_full_meta.json",
    };
    for (size_t i = 0; i < sizeof(ut_files) / sizeof(ut_files[0]); i++)
    {
        snprintf(pbuf, sizeof(pbuf), "%s/%s", tmpd, ut_files[i]);
        remove(pbuf);
    }
    rmdir(tmpd);

    cacaosim_wfstat_free(&wc);
    cacaosim_wfstat_free(&wg);

    /* Control-mode orthogonal projection parity (CPU vs CUDA + Parseval identity) */
    cacaosim_wfstat_t wcc, wcg;
    int rc_cc = cacaosim_wfstat_init(&wcc, nwf, pup, nframes, 0, 0, NULL);
    int rc_cg = cacaosim_wfstat_init(&wcg, nwf, pup, nframes, 0, 1, stream);
    if (rc_cc == 0 && rc_cg == 0 && wcc.act_pix > 16)
    {
        uint64_t ap   = wcc.act_pix;
        uint32_t n1   = 4;
        uint32_t ntot = 8;
        float   *Q    = (float *) calloc((size_t) ap * ntot, sizeof(float));
        /* Build 8 orthonormal columns on active pixels via modified Gram-Schmidt */
        for (uint32_t m = 0; m < ntot; m++)
        {
            double mean = 0.0;
            for (uint64_t p = 0; p < ap; p++)
            {
                uint32_t pix = wcc.act_idx[p];
                double   x   = ((double) (pix % nwf) / (double) nwf) - 0.5;
                double   y   = ((double) (pix / nwf) / (double) nwf) - 0.5;
                double   v   = cos((double) (m + 1) * 3.14159 * x + 0.7 * (double) m * y);
                Q[m * ap + p] = (float) v;
                mean += wcc.weights[pix] * v;
            }
            for (uint64_t p = 0; p < ap; p++)
            {
                Q[m * ap + p] = (float) (((double) Q[m * ap + p] - mean) * (double) wcc.act_sw[p]);
            }
            for (uint32_t prev = 0; prev < m; prev++)
            {
                double dot = 0.0;
                for (uint64_t p = 0; p < ap; p++)
                {
                    dot += (double) Q[prev * ap + p] * (double) Q[m * ap + p];
                }
                for (uint64_t p = 0; p < ap; p++)
                {
                    Q[m * ap + p] -= (float) (dot * (double) Q[prev * ap + p]);
                }
            }
            double nrm2 = 0.0;
            for (uint64_t p = 0; p < ap; p++)
            {
                nrm2 += (double) Q[m * ap + p] * (double) Q[m * ap + p];
            }
            double invn = 1.0 / sqrt(fmax(nrm2, 1e-24));
            for (uint64_t p = 0; p < ap; p++)
            {
                Q[m * ap + p] = (float) ((double) Q[m * ap + p] * invn);
            }
        }
        int rb_c = cacaosim_wfstat_set_ctrl_basis(&wcc, Q, n1, ntot, n1, ntot);
        int rb_g = cacaosim_wfstat_set_ctrl_basis(&wcg, Q, n1, ntot, n1, ntot);
        for (uint32_t f = 0; f < nframes && rb_c == 0 && rb_g == 0; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                for (uint64_t i = 0; i < npix; i++)
                {
                    seed               = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                    float u            = (float) ((seed >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f;
                    h_fr[k * npix + i] = (0.5f / (float) (k + 1)) * u;
                }
                /* Inject known in-space mode 0 (within Stg1) and mode 6 (only in Stg2) */
                for (uint64_t p = 0; p < ap; p++)
                {
                    uint32_t pix = wcc.act_idx[p];
                    double   inv = 1.0 / fmax((double) wcc.act_sw[p], 1e-12);
                    h_fr[k * npix + pix] += (float) (0.04 * (double) Q[0 * ap + p] * inv +
                                                     0.03 * (double) Q[6 * ap + p] * inv);
                }
                cacaosim_cuda_h2d_f32(d_fr[k], h_fr + k * npix, npix, stream);
            }
            const float *hc[3] = { h_fr, h_fr + npix, h_fr + 2 * npix };
            const float *dg[3] = { d_fr[0], d_fr[1], d_fr[2] };
            cacaosim_wfstat_frame(&wcc, hc);
            cacaosim_wfstat_frame(&wcg, dg);
            cacaosim_cuda_stream_sync(stream);
        }
        cacaosim_wfstat_sync_gpu(&wcg);
        double max_rel_ctrl = 0.0, max_parseval = 0.0;
        for (int k = 0; k < 3; k++)
        {
            for (uint32_t f = 0; f < nframes; f++)
            {
                double di = fabs((double) wcg.ts_in[k][f] - (double) wcc.ts_in[k][f]) /
                            fmax((double) wcc.ts_in[k][f], 1e-6);
                double do_ = fabs((double) wcg.ts_out[k][f] - (double) wcc.ts_out[k][f]) /
                             fmax((double) wcc.ts_out[k][f], 1e-6);
                max_rel_ctrl = fmax(max_rel_ctrl, fmax(di, do_));
                double tot2  = (double) wcc.ts_rms[k][f] * (double) wcc.ts_rms[k][f];
                double sum2  = (double) wcc.ts_in[k][f] * (double) wcc.ts_in[k][f] +
                               (double) wcc.ts_out[k][f] * (double) wcc.ts_out[k][f];
                max_parseval = fmax(max_parseval, fabs(sum2 - tot2) / fmax(tot2, 1e-9));
            }
        }
        snprintf(det, sizeof(det), "max rel CPU/GPU = %.2e, Parseval = %.2e, stg1_in=%.1f < fin_in=%.1f",
                 max_rel_ctrl, max_parseval, wcc.sum_in_rms[1] / nframes,
                 wcc.sum_in_rms[2] / nframes);
        check_cond(rb_c == 0 && rb_g == 0 && max_rel_ctrl < 1e-4 && max_parseval < 1e-4,
                   "wfstat control-mode in/out projection CPU/GPU parity & Parseval", det);
        free(Q);
    }
    cacaosim_wfstat_free(&wcc);
    cacaosim_wfstat_free(&wcg);
    for (int k = 0; k < 3; k++)
    {
        cacaosim_cuda_free_f32(d_fr[k]);
    }
    cacaosim_cuda_stream_destroy(stream);
    free(pup);
    free(h_fr);
}

/**
 * @brief Section 15: Validate multi-rate temporal oversampling physics across WFS, DM, & clocks.
 *
 * Role:
 *   Verifies all four temporal-fidelity mechanisms enabled when running the optical simulation at
 *   `oversample = 2..4x` above the fastest WFS frame rate:
 *     1. Sub-frame WFS exposure integration: exact `100%, 75%, 50%, 25%, 0%` step-arrival weights
 *        and `sinc(pi * f * T_exp)` high-frequency temporal roll-off vs `1x` aliasing.
 *     2. DM analytical ZOH subframe-mean (`act_pos_render`) matching continuum `1 - (tau/T) *
 *        (1 - exp(-T/tau))` across `K = 1, 2, 4` (eliminating right-endpoint stroke bias) plus
 *        fractional-frame `delay_us` transport latency (`625 us = 1.25` base frames).
 *     3. Pyramid WFS subframe modulation (`sync_arc` & `interleaved` vs `full` + `calib_fast`
 *        static parity on CPU and CUDA, plus physical mid-exposure modulation-phase cross-talk).
 *     4. Stage-1 / Stage-2 fractional clock offset (`clock_offset_us`), clock drift
 *        (`clock_drift_ppm`), and CPU/CUDA linear wavefront interpolation (`beam_init_lerp`).
 *
 * Rationale:
 *   Operating at `1x` WFS frame rate collapses detector boxcar integration to an impulse sample,
 *   evaluates right-endpoint DM settling, quantizes pure delays to integer frames, and forces
 *   synchronous inter-stage clocks. These deterministic analytical tests guarantee that `2..4x`
 *   oversampling reproduces exact continuum temporal physics without steady-state bias.
 */
static void test_temporal_oversampling_physics(void)
{
    printf("\n=== 15. Multi-Rate Temporal Oversampling Physics ===\n");

    char det[192];
    char err[256];

    /* --- 15a. WFS mid-frame step arrival partitioning & sinc(pi * f * T_exp) roll-off --- */
    {
        static const char *const ini_step =
            "[bench]\n"
            "nwf = 64\n"
            "pup_diam = 0.90\n"
            "master_hz = 1000.0\n"
            "oversample = 4\n"
            "atm_interp = 0\n"
            "out_beam = b_in\n"
            "wf_unit = 0\n"
            "backend = cpu\n"
            "[pyrwfs.wfs]\n"
            "in = b_in\n"
            "out_cam = csim_wfs\n"
            "ncam = 32\n"
            "lambda_um = 0.75\n"
            "mod_rad = 3.0\n"
            "mod_npts = 8\n"
            "mod_mode = full\n"
            "flux_ph = 1.0e6\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "bias_adu = 0.0\n"
            "out_dtype = float32\n";

        cacaosim_bench_cfg_t bcfg;
        cacaosim_bench_ctx_t bctx;
        int rc = cacaosim_bench_parse_str(ini_step, &bcfg, err, sizeof(err));
        if (rc == 0)
        {
            rc = cacaosim_bench_init(&bctx, &bcfg, NULL, 0, err, sizeof(err));
        }

        const uint32_t nwf  = 64;
        const size_t   npix = (size_t) nwf * nwf;
        const size_t   ncam = 32u * 32u;
        float *wf_zero      = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *wf_tilt      = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *wf_osc       = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *im_ref       = (float *) calloc(ncam, sizeof(float));
        float *im_full      = (float *) calloc(ncam, sizeof(float));

        for (uint32_t y = 0; y < nwf; y++)
        {
            for (uint32_t x = 0; x < nwf; x++)
            {
                float xn                = ((float) x - 31.5f) / 28.8f;
                wf_tilt[y * nwf + x]    = 0.015f * xn;
            }
        }

        int s_wfs = (rc == 0) ? cacaosim_bench_find_sensor(&bctx, "wfs") : -1;
        if (rc == 0 && s_wfs >= 0)
        {
            for (int k = 0; k < 4; k++)
            {
                cacaosim_bench_step(&bctx, wf_zero, NULL, 1.0f, NULL);
            }
            memcpy(im_ref, cacaosim_bench_sensor_f32(&bctx, s_wfs), ncam * sizeof(float));

            for (int k = 0; k < 4; k++)
            {
                cacaosim_bench_step(&bctx, wf_tilt, NULL, 1.0f, NULL);
            }
            memcpy(im_full, cacaosim_bench_sensor_f32(&bctx, s_wfs), ncam * sizeof(float));

            double norm2     = 0.0;
            double max_w_err = 0.0;
            for (size_t i = 0; i < ncam; i++)
            {
                double di = (double) im_full[i] - (double) im_ref[i];
                norm2 += di * di;
            }

            double meas_w[5] = { 0 };
            for (int m = 0; m <= 4; m++)
            {
                for (int k = 0; k < 4; k++)
                {
                    const float *w = (k >= m) ? wf_tilt : wf_zero;
                    cacaosim_bench_step(&bctx, w, NULL, 1.0f, NULL);
                }
                const float *im_m = cacaosim_bench_sensor_f32(&bctx, s_wfs);
                double       dot  = 0.0;
                for (size_t i = 0; i < ncam; i++)
                {
                    dot += ((double) im_m[i] - (double) im_ref[i]) *
                           ((double) im_full[i] - (double) im_ref[i]);
                }
                meas_w[m]   = (norm2 > 1e-20) ? (dot / norm2) : -1.0;
                double w_th = 1.0 - 0.25 * (double) m;
                max_w_err   = fmax(max_w_err, fabs(meas_w[m] - w_th));
            }
            snprintf(det, sizeof(det), "weights=[%.2f,%.2f,%.2f,%.2f,%.2f], max|err|=%.2e",
                     meas_w[0], meas_w[1], meas_w[2], meas_w[3], meas_w[4], max_w_err);
            check_cond(max_w_err < 1e-5, "WFS 4x subframe step-arrival weights", det);

            /*
             * 1 cycle / T_exp sinusoid sampled at midpoint phases pi/4, 3pi/4, 5pi/4, 7pi/4:
             * Project onto antisymmetric push-pull slope mode 0.5*(I(+w) - I(-w)) so quadratic
             * intra-frame optical-gain drop (even in w) is separated from linear slope (odd in w).
             */
            float *im_neg = (float *) calloc(ncam, sizeof(float));
            for (size_t i = 0; i < npix; i++)
            {
                wf_osc[i] = -wf_tilt[i];
            }
            for (int k = 0; k < 4; k++)
            {
                cacaosim_bench_step(&bctx, wf_osc, NULL, 1.0f, NULL);
            }
            memcpy(im_neg, cacaosim_bench_sensor_f32(&bctx, s_wfs), ncam * sizeof(float));

            double lin_n2 = 0.0;
            for (size_t i = 0; i < ncam; i++)
            {
                double dl = 0.5 * ((double) im_full[i] - (double) im_neg[i]);
                lin_n2 += dl * dl;
            }

            for (int k = 0; k < 4; k++)
            {
                float s = sinf(2.0f * CACAOSIM_PI_F * ((float) k + 0.5f) * 0.25f);
                for (size_t i = 0; i < npix; i++)
                {
                    wf_osc[i] = s * wf_tilt[i];
                }
                cacaosim_bench_step(&bctx, wf_osc, NULL, 1.0f, NULL);
            }
            const float *im_osc = cacaosim_bench_sensor_f32(&bctx, s_wfs);
            double       dot_o  = 0.0;
            for (size_t i = 0; i < ncam; i++)
            {
                double dl = 0.5 * ((double) im_full[i] - (double) im_neg[i]);
                dot_o += ((double) im_osc[i] - (double) im_ref[i]) * dl;
            }
            double gain_4x = (lin_n2 > 1e-20) ? fabs(dot_o / lin_n2) : 1.0;

            /* Compare against 1x impulse sampling at t_0 (phase pi/4 -> sin = 0.7071 alias) */
            bcfg.oversample = 1;
            cacaosim_bench_ctx_t bctx_1x;
            cacaosim_bench_init(&bctx_1x, &bcfg, NULL, 0, err, sizeof(err));
            float s0 = sinf(0.25f * CACAOSIM_PI_F);
            for (size_t i = 0; i < npix; i++)
            {
                wf_osc[i] = s0 * wf_tilt[i];
            }
            cacaosim_bench_step(&bctx_1x, wf_osc, NULL, 1.0f, NULL);
            const float *im_1x = cacaosim_bench_sensor_f32(&bctx_1x, 0);
            double       dot_1 = 0.0;
            for (size_t i = 0; i < ncam; i++)
            {
                double dl = 0.5 * ((double) im_full[i] - (double) im_neg[i]);
                dot_1 += ((double) im_1x[i] - (double) im_ref[i]) * dl;
            }
            double gain_1x = (lin_n2 > 1e-20) ? fabs(dot_1 / lin_n2) : 0.0;
            cacaosim_bench_free(&bctx_1x);
            free(im_neg);

            snprintf(det, sizeof(det), "|G(f_exp)|: 4x=%.2e vs 1x_alias=%.3f", gain_4x, gain_1x);
            check_cond(gain_4x < 1e-5 && gain_1x > 0.65,
                       "WFS sinc(pi*f*T) null vs 1x Nyquist aliasing", det);
            cacaosim_bench_free(&bctx);
        }
        else
        {
            check_cond(0, "WFS 4x subframe step-arrival weights", err);
        }

        free(im_full);
        free(im_ref);
        free(wf_osc);
        free(wf_tilt);
        free(wf_zero);
    }

    /* --- 15b. DM analytical ZOH subframe-mean & fractional-frame delay_us invariance --- */
    {
        const double T_sec   = 0.0005;
        const double tau_sec = 0.0002;
        const double exact_m = 1.0 - (tau_sec / T_sec) * (1.0 - exp(-T_sec / tau_sec));
        const double r_end   = 1.0 - exp(-T_sec / tau_sec);

        double max_zoh_err = 0.0;
        double no_mean_1x  = 0.0;
        const int ks[3]    = { 1, 2, 4 };
        for (int idx = 0; idx < 3; idx++)
        {
            int               K    = ks[idx];
            cacaosim_dm_cfg_t dcfg = {
                .nact_x = 4, .nact_y = 4, .nwf = 32, .dm_diam = 0.9f, .if_coupling = 0.2f,
                .refl_scale = 1.0f, .max_stroke = 10.0f, .dac_bits = 0, .dyn_mode = 1,
                .dt_sec = (float) (T_sec / (double) K), .tau_sec = (float) tau_sec,
                .delay_sec = 0.0f, .no_zoh_mean = 0, .backend = CACAOSIM_BACKEND_CPU
            };
            cacaosim_dm_ctx_t dm;
            cacaosim_dm_init(&dm, &dcfg, NULL);
            float cmd[16];
            float opd[32 * 32];
            for (int a = 0; a < 16; a++)
            {
                cmd[a] = 1.0f;
            }
            double sum_render = 0.0;
            for (int k = 0; k < K; k++)
            {
                cacaosim_dm_step(&dm, cmd, NULL, 0.0f, (float) tau_sec, 10.0f, opd);
                sum_render += (double) dm.act_pos_render[5];
            }
            sum_render /= (double) K;
            max_zoh_err = fmax(max_zoh_err, fabs(sum_render - exact_m));
            cacaosim_dm_free(&dm);
        }

        /* Verify legacy right-endpoint mode (`no_zoh_mean = 1`) reproduces `1 - exp(-T/tau)` */
        {
            cacaosim_dm_cfg_t dcfg = {
                .nact_x = 4, .nact_y = 4, .nwf = 32, .dm_diam = 0.9f, .if_coupling = 0.2f,
                .refl_scale = 1.0f, .max_stroke = 10.0f, .dac_bits = 0, .dyn_mode = 1,
                .dt_sec = (float) T_sec, .tau_sec = (float) tau_sec, .no_zoh_mean = 1,
                .backend = CACAOSIM_BACKEND_CPU
            };
            cacaosim_dm_ctx_t dm;
            cacaosim_dm_init(&dm, &dcfg, NULL);
            float cmd[16];
            float opd[32 * 32];
            for (int a = 0; a < 16; a++)
            {
                cmd[a] = 1.0f;
            }
            cacaosim_dm_step(&dm, cmd, NULL, 0.0f, (float) tau_sec, 10.0f, opd);
            no_mean_1x = (double) dm.act_pos_render[5];
            cacaosim_dm_free(&dm);
        }

        snprintf(det, sizeof(det), "exact=%.5f, max|err(K=1,2,4)|=%.2e (no_mean=%.5f)", exact_m,
                 max_zoh_err, no_mean_1x);
        check_cond(max_zoh_err < 1e-5 && fabs(no_mean_1x - r_end) < 1e-5,
                   "DM analytical ZOH subframe-mean (K=1,2,4)", det);

        /* Verify 2nd-order underdamped ZOH subframe-mean matches fine K=64 quadrature */
        double m2_k[2]       = { 0.0, 0.0 };
        const int ks2[2]     = { 1, 64 };
        for (int idx = 0; idx < 2; idx++)
        {
            int               K    = ks2[idx];
            cacaosim_dm_cfg_t dcfg = {
                .nact_x = 4, .nact_y = 4, .nwf = 32, .dm_diam = 0.9f, .if_coupling = 0.2f,
                .refl_scale = 1.0f, .max_stroke = 10.0f, .dac_bits = 0, .dyn_mode = 2,
                .dt_sec = (float) (T_sec / (double) K), .f0_hz = 2200.0f, .zeta = 0.55f,
                .no_zoh_mean = 0, .backend = CACAOSIM_BACKEND_CPU
            };
            cacaosim_dm_ctx_t dm;
            cacaosim_dm_init(&dm, &dcfg, NULL);
            float cmd[16];
            float opd[32 * 32];
            for (int a = 0; a < 16; a++)
            {
                cmd[a] = 1.0f;
            }
            for (int k = 0; k < K; k++)
            {
                cacaosim_dm_step(&dm, cmd, NULL, 0.0f, 0.0001f, 10.0f, opd);
                m2_k[idx] += (double) dm.act_pos_render[5];
            }
            m2_k[idx] /= (double) K;
            cacaosim_dm_free(&dm);
        }
        snprintf(det, sizeof(det), "K=1: %.6f vs K=64: %.6f (|diff|=%.2e)", m2_k[0], m2_k[1],
                 fabs(m2_k[0] - m2_k[1]));
        check_cond(fabs(m2_k[0] - m2_k[1]) < 1e-5,
                   "DM 2nd-order ZOH exact subframe-mean integral", det);

        /* Fractional-frame delay_us = 625 us at 2 kHz x 4 = 8 kHz (5.0 sub-ticks = 1.25 frames) */
        static const char *const ini_dly =
            "[bench]\n"
            "nwf = 64\n"
            "pup_diam = 0.90\n"
            "master_hz = 2000.0\n"
            "oversample = 4\n"
            "out_beam = b_in\n"
            "backend = cpu\n"
            "[dm.dm0]\n"
            "in = b_in\n"
            "out = b_dm\n"
            "in_cmd = dm00disp\n"
            "nact_x = 8\n"
            "nact_y = 8\n"
            "dac_bits = 0\n"
            "dyn_mode = 0\n"
            "delay_us = 625.0\n"
            "[pyrwfs.wfs]\n"
            "in = b_dm\n"
            "out_cam = csim_wfs\n"
            "ncam = 32\n"
            "lambda_um = 0.75\n"
            "mod_rad = 3.0\n"
            "mod_npts = 8\n"
            "mod_mode = full\n"
            "flux_ph = 1.0e6\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "bias_adu = 0.0\n"
            "out_dtype = float32\n";

        cacaosim_bench_cfg_t dly_cfg;
        cacaosim_bench_ctx_t dly_ctx;
        if (cacaosim_bench_parse_str(ini_dly, &dly_cfg, err, sizeof(err)) == 0 &&
            cacaosim_bench_init(&dly_ctx, &dly_cfg, NULL, 0, err, sizeof(err)) == 0)
        {
            const size_t ncam = 32u * 32u;
            float *im0        = (float *) calloc(ncam, sizeof(float));
            float *im_ss      = (float *) calloc(ncam, sizeof(float));
            float *cmd        = cacaosim_bench_dm_cmd_ptr(&dly_ctx, 0);

            /* Settle zero baseline */
            for (int k = 0; k < 12; k++)
            {
                cacaosim_bench_step(&dly_ctx, NULL, NULL, 0.0f, NULL);
            }
            memcpy(im0, cacaosim_bench_sensor_f32(&dly_ctx, 0), ncam * sizeof(float));

            /* Apply small poke at start of frame 0 (sub-tick 0); delay = 5 sub-ticks */
            cmd[27] = 0.01f;
            double resp[3] = { 0.0, 0.0, 0.0 };
            float *fr_buf[3];
            for (int f = 0; f < 3; f++)
            {
                fr_buf[f] = (float *) calloc(ncam, sizeof(float));
                for (int k = 0; k < 4; k++)
                {
                    cacaosim_bench_step(&dly_ctx, NULL, NULL, 0.0f, NULL);
                }
                memcpy(fr_buf[f], cacaosim_bench_sensor_f32(&dly_ctx, 0), ncam * sizeof(float));
            }
            memcpy(im_ss, fr_buf[2], ncam * sizeof(float));

            double n2 = 0.0;
            for (size_t i = 0; i < ncam; i++)
            {
                double d = (double) im_ss[i] - (double) im0[i];
                n2 += d * d;
            }
            for (int f = 0; f < 3; f++)
            {
                double dot = 0.0;
                for (size_t i = 0; i < ncam; i++)
                {
                    dot += ((double) fr_buf[f][i] - (double) im0[i]) *
                           ((double) im_ss[i] - (double) im0[i]);
                }
                resp[f] = (n2 > 1e-20) ? (dot / n2) : -1.0;
                free(fr_buf[f]);
            }
            snprintf(det, sizeof(det), "fr[0..2]=[%.3f, %.3f, %.3f] (expect 0, 0.75, 1)", resp[0],
                     resp[1], resp[2]);
            check_cond(fabs(resp[0]) < 1e-5 && fabs(resp[1] - 0.75) < 1e-4 &&
                           fabs(resp[2] - 1.0) < 1e-5,
                       "DM delay_us=625us fractional 1.25-frame step", det);

            free(im_ss);
            free(im0);
            cacaosim_bench_free(&dly_ctx);
        }
    }

    /* --- 15c. Pyramid WFS subframe modulation (`sync_arc`, `interleaved`, `calib_fast`) --- */
    {
        const uint32_t nwf  = 64;
        const uint32_t ncam = 32;
        const size_t   npix = (size_t) nwf * nwf;
        const size_t   nc2  = (size_t) ncam * ncam;
        float *pup          = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *wf_ab        = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        float *wf_zero      = (float *) cacaosim_aligned_calloc(npix, sizeof(float));

        cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = 0.90f, .cobs = 0.30f };
        cacaosim_make_pupil(&pcfg, pup);
        cacaosim_make_ncpa(nwf, 0.90f, pup, 0.07f, 314ULL, wf_ab);

        float *im_ref_full = (float *) calloc(nc2, sizeof(float));
        float *im_tmp      = (float *) calloc(nc2, sizeof(float));

        int      has_gpu     = cacaosim_cuda_available();
        int      nb          = has_gpu ? 2 : 1;
        double   max_rel_mod = 0.0;
        for (int bi = 0; bi < nb; bi++)
        {
            int32_t backend = (bi == 0) ? CACAOSIM_BACKEND_CPU : CACAOSIM_BACKEND_CUDA;
            for (int mode = 0; mode <= 2; mode++)
            {
                cacaosim_pyrwfs_cfg_t p_cfg = {
                    .nwf = nwf, .pad_factor = 2, .ncam = ncam, .pup_diam = 0.90f,
                    .lambda_um = 0.75f, .mod_rad_ld = 3.0f, .mod_npts = 16, .pup_sep = 0.50f,
                    .fstop_frac = 0.45f, .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 0.75f,
                    .mod_subframe_mode = mode, .mod_rev = 1, .mod_subframes = 4u,
                    .backend = backend,
                    .det = { .flux_ph = 1.0e5f, .shutter = 1, .photon_noise = 0, .ron_e = 0.0f,
                             .gain_eadu = 1.0f, .bias_adu = 0.0f, .sat_adu = 1.0e9f,
                             .out_dtype = 1 }
                };
                cacaosim_pyrwfs_ctx_t p_ctx;
                cacaosim_pyrwfs_init(&p_ctx, &p_cfg, pup, NULL, NULL);
                for (int k = 0; k < 4; k++)
                {
                    cacaosim_pyrwfs_accum_subframe(&p_ctx, pup, NULL, wf_ab, NULL, NULL, NULL,
                                                   1.0f, 3.0f);
                }
                cacaosim_pyrwfs_readout(&p_ctx, &p_cfg.det, NULL, im_tmp);

                if (bi == 0 && mode == 0)
                {
                    memcpy(im_ref_full, im_tmp, nc2 * sizeof(float));
                }
                else
                {
                    double d2 = 0.0, r2 = 0.0;
                    for (size_t i = 0; i < nc2; i++)
                    {
                        double d = (double) im_tmp[i] - (double) im_ref_full[i];
                        d2 += d * d;
                        r2 += (double) im_ref_full[i] * (double) im_ref_full[i];
                    }
                    max_rel_mod = fmax(max_rel_mod, sqrt(d2 / fmax(r2, 1e-20)));
                }

                /* Also verify 1-tick `force_full_mod = 1` (used by `.calib_fast`) on `sync_arc` */
                if (mode == CACAOSIM_PYRMOD_SYNC_ARC)
                {
                    p_ctx.force_full_mod = 1;
                    cacaosim_pyrwfs_accum_subframe(&p_ctx, pup, NULL, wf_ab, NULL, NULL, NULL,
                                                   1.0f, 3.0f);
                    cacaosim_pyrwfs_readout(&p_ctx, &p_cfg.det, NULL, im_tmp);
                    double d2 = 0.0, r2 = 0.0;
                    for (size_t i = 0; i < nc2; i++)
                    {
                        double d = (double) im_tmp[i] - (double) im_ref_full[i];
                        d2 += d * d;
                        r2 += (double) im_ref_full[i] * (double) im_ref_full[i];
                    }
                    max_rel_mod = fmax(max_rel_mod, sqrt(d2 / fmax(r2, 1e-20)));
                }
                cacaosim_pyrwfs_free(&p_ctx);
            }
        }
        snprintf(det, sizeof(det), "max rel L2 (sync_arc/interleaved/calib_fast%s) = %.2e",
                 has_gpu ? " CPU+CUDA" : " CPU", max_rel_mod);
        check_cond(max_rel_mod < 1e-5,
                   "PyWFS sync_arc/interleaved static parity", det);

        /* Half-exposure transient poke: `sync_arc` captures physical piezo-arc cross-talk */
        float *im_flat = (float *) calloc(nc2, sizeof(float));
        float *im_sync = (float *) calloc(nc2, sizeof(float));
        float *im_fhalf = (float *) calloc(nc2, sizeof(float));
        for (int mode = 0; mode <= 1; mode++)
        {
            cacaosim_pyrwfs_cfg_t p_cfg = {
                .nwf = nwf, .pad_factor = 2, .ncam = ncam, .pup_diam = 0.90f, .lambda_um = 0.75f,
                .mod_rad_ld = 3.0f, .mod_npts = 16, .pup_sep = 0.50f, .fstop_frac = 0.45f,
                .wf_unit = CACAOSIM_WFUNIT_OPD_UM, .ref_lambda = 0.75f,
                .mod_subframe_mode = mode, .mod_rev = 1, .mod_subframes = 4u,
                .backend = CACAOSIM_BACKEND_CPU,
                .det = { .flux_ph = 1.0e5f, .shutter = 1, .photon_noise = 0, .ron_e = 0.0f,
                         .gain_eadu = 1.0f, .bias_adu = 0.0f, .sat_adu = 1.0e9f, .out_dtype = 1 }
            };
            cacaosim_pyrwfs_ctx_t p_ctx;
            cacaosim_pyrwfs_init(&p_ctx, &p_cfg, pup, NULL, NULL);
            if (mode == 0)
            {
                for (int k = 0; k < 4; k++)
                {
                    cacaosim_pyrwfs_accum_subframe(&p_ctx, pup, NULL, wf_zero, NULL, NULL, NULL,
                                                   0.0f, 3.0f);
                }
                cacaosim_pyrwfs_readout(&p_ctx, &p_cfg.det, NULL, im_flat);
            }
            for (int k = 0; k < 4; k++)
            {
                const float *w = (k < 2) ? wf_ab : wf_zero;
                cacaosim_pyrwfs_accum_subframe(&p_ctx, pup, NULL, w, NULL, NULL, NULL, 1.0f, 3.0f);
            }
            cacaosim_pyrwfs_readout(&p_ctx, &p_cfg.det, NULL, (mode == 0) ? im_fhalf : im_sync);
            cacaosim_pyrwfs_free(&p_ctx);
        }
        double diff_xtalk = 0.0, ref_sig = 0.0;
        for (size_t i = 0; i < nc2; i++)
        {
            double ds = (double) im_sync[i] - (double) im_flat[i];
            double df = (double) im_fhalf[i] - (double) im_flat[i];
            diff_xtalk += (ds - df) * (ds - df);
            ref_sig    += df * df;
        }
        double rel_xtalk = sqrt(diff_xtalk / fmax(ref_sig, 1e-20));
        snprintf(det, sizeof(det), "half-frame arc vs full-ring rel L2 = %.3f (> 0.05)",
                 rel_xtalk);
        check_cond(rel_xtalk > 0.05, "PyWFS sync_arc captures mid-frame arc cross-talk", det);

        free(im_fhalf);
        free(im_sync);
        free(im_flat);
        free(im_tmp);
        free(im_ref_full);
        free(wf_zero);
        free(wf_ab);
        free(pup);
    }

    /* --- 15d. Stage-1/Stage-2 clock skew (`clock_offset_us`), drift, & lerp parity --- */
    {
        static const char *const ini_clk =
            "[bench]\n"
            "nwf = 64\n"
            "pup_diam = 0.90\n"
            "master_hz = 2000.0\n"
            "oversample = 4\n"
            "out_beam = b_in\n"
            "backend = cpu\n"
            "[splitter.sp]\n"
            "in = b_in\n"
            "out_refl = b1\n"
            "out_trans = b2\n"
            "mode = gray\n"
            "r_gray = 0.5\n"
            "t_gray = 0.5\n"
            "[pyrwfs.wfs1]\n"
            "in = b1\n"
            "out_cam = csim_wfs1\n"
            "ncam = 32\n"
            "mod_rad = 0.0\n"
            "clock_offset_us = 0.0\n"
            "clock_drift_ppm = 0.0\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "out_dtype = float32\n"
            "[pyrwfs.wfs2]\n"
            "in = b2\n"
            "out_cam = csim_wfs2\n"
            "ncam = 32\n"
            "mod_rad = 0.0\n"
            "clock_offset_us = 250.0\n"
            "clock_drift_ppm = 10000.0\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "out_dtype = float32\n";

        cacaosim_bench_cfg_t ccfg;
        cacaosim_bench_ctx_t cctx;
        if (cacaosim_bench_parse_str(ini_clk, &ccfg, err, sizeof(err)) == 0 &&
            cacaosim_bench_init(&cctx, &ccfg, NULL, 0, err, sizeof(err)) == 0)
        {
            int s1 = cacaosim_bench_find_sensor(&cctx, "wfs1");
            int s2 = cacaosim_bench_find_sensor(&cctx, "wfs2");
            int first_rd1 = -1, first_rd2 = -1;
            for (int t = 0; t < 400; t++)
            {
                uint64_t p1 = cctx.nodes[cctx.sensor_node_idx[s1]].pyrwfs_ctx.frame_seq;
                uint64_t p2 = cctx.nodes[cctx.sensor_node_idx[s2]].pyrwfs_ctx.frame_seq;
                cacaosim_bench_step(&cctx, NULL, NULL, 0.0f, NULL);
                if (first_rd1 < 0 &&
                    cctx.nodes[cctx.sensor_node_idx[s1]].pyrwfs_ctx.frame_seq > p1)
                {
                    first_rd1 = t;
                }
                if (first_rd2 < 0 &&
                    cctx.nodes[cctx.sensor_node_idx[s2]].pyrwfs_ctx.frame_seq > p2)
                {
                    first_rd2 = t;
                }
            }
            uint64_t tot1 = cctx.nodes[cctx.sensor_node_idx[s1]].pyrwfs_ctx.frame_seq;
            uint64_t tot2 = cctx.nodes[cctx.sensor_node_idx[s2]].pyrwfs_ctx.frame_seq;
            snprintf(det, sizeof(det), "first_tick=(%d vs %d), readouts_100fr=(%lu vs %lu)",
                     first_rd1, first_rd2, (unsigned long) tot1, (unsigned long) tot2);
            check_cond(first_rd1 == 3 && first_rd2 == 1 && tot1 == 100u && tot2 == 101u,
                       "Inter-stage clock_offset_us & clock_drift_ppm", det);
            cacaosim_bench_free(&cctx);
        }

        /* Linear wavefront interpolation CPU vs CUDA parity */
        const uint32_t nwf_l = 64u;
        const size_t   np    = (size_t) nwf_l * nwf_l;
        float *pup = (float *) cacaosim_aligned_calloc(np, sizeof(float));
        float *w0  = (float *) cacaosim_aligned_calloc(np, sizeof(float));
        float *w1  = (float *) cacaosim_aligned_calloc(np, sizeof(float));
        float *oc  = (float *) cacaosim_aligned_calloc(np, sizeof(float));
        float *ac  = (float *) cacaosim_aligned_calloc(np, sizeof(float));
        for (size_t i = 0; i < np; i++)
        {
            pup[i] = 0.85f;
            w0[i]  = 0.1f * sinf((float) i * 0.03f);
            w1[i]  = -0.2f * cosf((float) i * 0.05f);
        }
        cacaosim_beam_init_lerp_cpu(pup, NULL, w0, w1, 0.375f, NULL, 0.5f, nwf_l, oc, ac);
        double max_lerp_err = 0.0;
        for (size_t i = 0; i < np; i++)
        {
            double ref = 0.5 * (0.625 * (double) w0[i] + 0.375 * (double) w1[i]);
            max_lerp_err = fmax(max_lerp_err, fabs((double) oc[i] - ref));
        }
        if (cacaosim_cuda_available())
        {
            float *dp  = cacaosim_cuda_alloc_f32(np);
            float *d0  = cacaosim_cuda_alloc_f32(np);
            float *d1  = cacaosim_cuda_alloc_f32(np);
            float *do_ = cacaosim_cuda_alloc_f32(np);
            float *da  = cacaosim_cuda_alloc_f32(np);
            float *hg  = (float *) calloc(np, sizeof(float));
            cacaosim_cuda_h2d_f32(dp, pup, np, NULL);
            cacaosim_cuda_h2d_f32(d0, w0, np, NULL);
            cacaosim_cuda_h2d_f32(d1, w1, np, NULL);
            cacaosim_cuda_beam_init_lerp(dp, NULL, d0, d1, 0.375f, NULL, 0.5f, nwf_l, do_, da,
                                         NULL);
            cacaosim_cuda_d2h_f32(hg, do_, np, NULL);
            cacaosim_cuda_stream_sync(NULL);
            for (size_t i = 0; i < np; i++)
            {
                max_lerp_err = fmax(max_lerp_err, fabs((double) hg[i] - (double) oc[i]));
            }
            free(hg);
            cacaosim_cuda_free_f32(da);
            cacaosim_cuda_free_f32(do_);
            cacaosim_cuda_free_f32(d1);
            cacaosim_cuda_free_f32(d0);
            cacaosim_cuda_free_f32(dp);
        }
        snprintf(det, sizeof(det), "max |err| = %.2e", max_lerp_err);
        check_cond(max_lerp_err < 1e-6, "Sub-tick linear WF interpolation parity", det);
        free(ac);
        free(oc);
        free(w1);
        free(w0);
        free(pup);
    }
}

/**
 * @brief Section 16: Validate polychromatic WFS/camera bandwidths, lambda/D scaling & CUDA parity.
 *
 * Role:
 *   Verifies end-to-end chromatic physics across photometry, all four diffractive sensors
 *   (`pyrwfs`, `shwfs`, `zwfs`, `fpcam` MFT & FFT), polychromatic upstream beam linear phasor
 *   interpolation, broadband coronagraph + LOWFS pickoff, and CPU vs CUDA parity:
 *     - 16a: `cacaosim_resolve_nlambda()` (`0 -> 2` default when `bw_frac > 0`, `1` at `0`),
 *       strict narrowband flux linearity (`bw_frac = 1e-4` vs `0.20`, no `0.15` clamp), and exact
 *       bolometric sub-band photon conservation + dichroic centroid shift in
 *       `cacaosim_spec_sample_band()`.
 *     - 16b: `fpcam` broadband (`bw_frac = 0.30, N = 5`) unaberrated Strehl `== 1.0` in both
 *       `PROP_MFT` and `PROP_FFT`, `PROP_FFT` 2D bilinear `lambda/D` PSF width scaling (`1.50x`),
 *       and achromatic geometric tilt pixel landing.
 *     - 16c: `shwfs` subaperture 2D bilinear `lambda/D` spot width scaling (`1.50x` for
 *       `0.90 / 0.60 um`) alongside achromatic geometric mirror-tilt spot centroids (`< 0.03 px`).
 *     - 16d: `pyrwfs` polychromatic default (`eff_nlam == 2`), `nlambda = 1` override with
 *       identical bandpass photometry, achromatic double-pyramid sub-pupil registration
 *       (`chrom_sep = 0`), and single-prism chromatic sub-pupil shift (`chrom_sep = 0.20`).
 *     - 16e: `zwfs` dispersive (`achrom_dimple = 0`) vs achromatic (`achrom_dimple = 1`) dimple
 *       phase response, and 1D wavelength linear complex-phasor interpolation (`accum_beam_poly`)
 *       at midpoint `alpha = 0.5` between upstream polychromatic beam channels.
 *     - 16f: CPU vs CUDA parity (`rel_L2 < 3e-5`) across polychromatic `pyrwfs`, `shwfs`, `zwfs`,
 *       `fpcam` (MFT & FFT), and a full broadband `coro + out_lowfs + 4-sensor` `.bench` DAG.
 */
static void test_chromaticity_and_poly_wfs(void)
{
    printf("\n=== 16. Polychromatic Bandwidths, Focal lambda/D Scaling & Interp ===\n");
    char det[192];

    /* 16a. Channel resolution, narrowband flux linearity & sub-band conservation */
    {
        int ok_res = (cacaosim_resolve_nlambda(0, 1u, 0.20f) == 2) &&
                     (cacaosim_resolve_nlambda(0, 1u, 0.0f) == 1) &&
                     (cacaosim_resolve_nlambda(0, 5u, 0.20f) == 5) &&
                     (cacaosim_resolve_nlambda(1, 5u, 0.20f) == 1) &&
                     (cacaosim_resolve_nlambda(3, 5u, 0.20f) == 3);

        cacaosim_spec_t star_sp;
        cacaosim_spec_star(&star_sp, 8.2f, 0.30f, 8.0f, 5778.0f, 0.75f);

        float f_zero   = cacaosim_spec_band_flux(&star_sp, 0.75f, 0.0f, 1.0e-3f, 0.80f, NULL);
        float f_narrow = cacaosim_spec_band_flux(&star_sp, 0.75f, 1.0e-4f, 1.0e-3f, 0.80f, NULL);
        float f_broad  = cacaosim_spec_band_flux(&star_sp, 0.75f, 0.20f, 1.0e-3f, 0.80f, NULL);
        double bw_ratio = (f_narrow > 0.0f) ? ((double) f_broad / (double) f_narrow) : 0.0;

        float flat_lam[2] = { 0.0f, 0.0f };
        float flat_w[2]   = { 0.0f, 0.0f };
        (void) cacaosim_spec_sample_band(NULL, 0.80f, 0.25f, 2, 1.0e-3f, 1.0f, flat_lam, flat_w,
                                         NULL);
        int ok_flat = (fabsf(flat_lam[0] - 0.75f) < 1e-6f) &&
                      (fabsf(flat_lam[1] - 0.85f) < 1e-6f) &&
                      (fabsf(flat_w[0] - 0.5f) < 1e-6f) &&
                      (fabsf(flat_w[1] - 0.5f) < 1e-6f);

        /* Dichroic short-pass reflection (< 0.80 um) sampled across [0.70, 0.90] um */
        cacaosim_split_cfg_t sp_cfg = {
            .mode       = CACAOSIM_SPLIT_DICHROIC_LP,
            .lambda_cut = 0.80f,
            .roll_um    = 0.01f,
            .eff_refl   = 0.95f,
            .eff_trans  = 0.95f,
        };
        cacaosim_spec_t vis_sp, nir_sp;
        cacaosim_spec_split(&star_sp, &sp_cfg, &vis_sp, &nir_sp);

        float    dich_lam[2] = { 0.0f, 0.0f };
        float    dich_w[2]   = { 0.0f, 0.0f };
        float    eff_lam     = 0.0f;
        float    f_tot_ref   = cacaosim_spec_band_flux(&vis_sp, 0.80f, 0.25f, 1.0e-3f, 0.80f, NULL);
        double   max_part_err = 0.0;
        const int n_list[4]   = { 1, 2, 5, 9 };
        for (int k = 0; k < 4; k++)
        {
            float lk[9], wk[9];
            float fk = cacaosim_spec_sample_band(&vis_sp, 0.80f, 0.25f, n_list[k], 1.0e-3f, 0.80f,
                                                 lk, wk, (n_list[k] == 2) ? &eff_lam : NULL);
            double wsum = 0.0;
            for (int c = 0; c < n_list[k]; c++)
            {
                wsum += (double) wk[c];
            }
            double rel_f = fabs((double) fk - (double) f_tot_ref) / (double) f_tot_ref;
            max_part_err = fmax(max_part_err, fmax(rel_f, fabs(wsum - 1.0)));
            if (n_list[k] == 2)
            {
                dich_lam[0] = lk[0];
                dich_lam[1] = lk[1];
                dich_w[0]   = wk[0];
                dich_w[1]   = wk[1];
            }
        }
        (void) dich_lam;

        snprintf(det, sizeof(det),
                 "f0=%.1e, f(0.20)/f(1e-4)=%.1f, part_err=%.2e, dich_w=(%.3f,%.3f), lam_eff=%.3fum",
                 f_zero, bw_ratio, max_part_err, dich_w[0], dich_w[1], eff_lam);
        check_cond(ok_res && ok_flat && f_zero == 0.0f && fabs(bw_ratio - 2000.0) < 25.0 &&
                       max_part_err < 1e-6 && dich_w[0] > 0.90f && eff_lam < 0.765f,
                   "Sub-band photometry, narrowband & dichroic", det);
    }

    /* Shared pupil & achromatic linear tilt OPD screen */
    const uint32_t nwf  = 64u;
    const uint64_t npix = (uint64_t) nwf * nwf;
    const float    pdia = 0.875f;
    float         *pup  = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float         *tilt = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    float         *ab   = (float *) cacaosim_aligned_calloc(npix, sizeof(float));

    cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = pdia, .cobs = 0.0f, .nspider = 0 };
    cacaosim_make_pupil(&pcfg, pup);
    cacaosim_make_ncpa(nwf, pdia, pup, 0.09f, 9091ULL, ab);

    const float dpix = pdia * (float) nwf;
    const float c_wf = 0.5f * (float) (nwf - 1u);
    /* 1.5 lambda0/D peak-to-valley tilt at lambda0 = 1.65 um -> OPD(x) = 1.5 * 1.65 * (x/D) */
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            tilt[(size_t) y * nwf + x] = 1.5f * 1.65f * (((float) x - c_wf) / dpix);
        }
    }

    /* 16b. FPCAM (MFT & FFT) broadband Strehl == 1.0, lambda/D scaling & achromatic tilt */
    {
        const uint32_t ncam = 96u;
        float         *psf_b = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam,
                                                                 sizeof(float));
        float         *psf_r = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam,
                                                                 sizeof(float));
        cacaosim_det_cfg_t clean_det = {
            .flux_ph      = 1.0e6f,
            .gain_eadu    = 1.0f,
            .shutter      = 1,
            .photon_noise = 0,
            .sat_adu      = 1.0e12f,
            .out_dtype    = CACAOSIM_DTYPE_F32,
        };
        cacaosim_fpcam_cfg_t fcfg_mft = {
            .nwf         = nwf,
            .ncam        = ncam,
            .prop_mode   = CACAOSIM_FPCAM_PROP_MFT,
            .pad_factor  = 4u,
            .pup_diam    = pdia,
            .lambda_um   = 1.65f,
            .bw_frac     = 0.30f,
            .nlambda     = 5,
            .pixscale_ld = 4.0f,
            .wf_unit     = CACAOSIM_WFUNIT_OPD_UM,
            .det         = clean_det,
            .backend     = CACAOSIM_BACKEND_CPU,
        };
        cacaosim_fpcam_cfg_t fcfg_fft = fcfg_mft;
        fcfg_fft.prop_mode            = CACAOSIM_FPCAM_PROP_FFT;

        cacaosim_fpcam_ctx_t fctx_m, fctx_f;
        cacaosim_fpcam_init(&fctx_m, &fcfg_mft, pup, NULL, NULL);
        cacaosim_fpcam_init(&fctx_f, &fcfg_fft, pup, NULL, NULL);
        cacaosim_fpcam_accum_beam(&fctx_m, NULL, pup, NULL, NULL, NULL);
        cacaosim_fpcam_readout(&fctx_m, &clean_det, NULL, psf_b, NULL);
        cacaosim_fpcam_accum_beam(&fctx_f, NULL, pup, NULL, NULL, NULL);
        cacaosim_fpcam_readout(&fctx_f, &clean_det, NULL, psf_r, NULL);
        float sr_mft = fctx_m.last_strehl_peak;
        float sr_fft = fctx_f.last_strehl_peak;
        cacaosim_fpcam_free(&fctx_m);
        cacaosim_fpcam_free(&fctx_f);

        /* Compare blue (1.32 um) vs red (1.98 um) inside PROP_FFT camera defined at lam0=1.65 */
        cacaosim_fpcam_cfg_t fblue = fcfg_fft;
        fblue.nlambda              = 1;
        fblue.has_ext_lam          = 1;
        fblue.ext_lam_um[0]        = 1.32f;
        cacaosim_fpcam_cfg_t fred  = fblue;
        fred.ext_lam_um[0]         = 1.98f;

        cacaosim_fpcam_ctx_t xb, xr;
        cacaosim_fpcam_init(&xb, &fblue, pup, NULL, NULL);
        cacaosim_fpcam_init(&xr, &fred, pup, NULL, NULL);

        /* Flat-wavefront main-lobe width scaling (I >= 10% peak): w(1.98)/w(1.32) == 1.50 */
        cacaosim_fpcam_accum_beam(&xb, NULL, pup, NULL, NULL, NULL);
        cacaosim_fpcam_readout(&xb, &clean_det, NULL, psf_b, NULL);
        cacaosim_fpcam_accum_beam(&xr, NULL, pup, NULL, NULL, NULL);
        cacaosim_fpcam_readout(&xr, &clean_det, NULL, psf_r, NULL);

        float pk_b = 0.0f, pk_r = 0.0f;
        for (size_t k = 0; k < (size_t) ncam * ncam; k++)
        {
            if (psf_b[k] > pk_b)
            {
                pk_b = psf_b[k];
            }
            if (psf_r[k] > pk_r)
            {
                pk_r = psf_r[k];
            }
        }
        double       sb = 0.0, sr = 0.0, r2b = 0.0, r2r = 0.0;
        const double c0 = 0.5 * (double) ncam;
        for (uint32_t y = 0; y < ncam; y++)
        {
            for (uint32_t x = 0; x < ncam; x++)
            {
                double dx = (double) x - c0;
                double dy = (double) y - c0;
                double r2 = dx * dx + dy * dy;
                float  vb = psf_b[(size_t) y * ncam + x];
                float  vr = psf_r[(size_t) y * ncam + x];
                if (vb >= 0.10f * pk_b)
                {
                    sb += (double) vb;
                    r2b += (double) vb * r2;
                }
                if (vr >= 0.10f * pk_r)
                {
                    sr += (double) vr;
                    r2r += (double) vr * r2;
                }
            }
        }
        double w_ratio = sqrt((r2r / sr) / (r2b / sb));

        /* Achromatic mirror tilt: identical focal centroid pixel across 1.32 um and 1.98 um */
        cacaosim_fpcam_accum_beam(&xb, tilt, pup, NULL, NULL, NULL);
        cacaosim_fpcam_readout(&xb, &clean_det, NULL, psf_b, NULL);
        cacaosim_fpcam_accum_beam(&xr, tilt, pup, NULL, NULL, NULL);
        cacaosim_fpcam_readout(&xr, &clean_det, NULL, psf_r, NULL);

        pk_b = 0.0f;
        pk_r = 0.0f;
        for (size_t k = 0; k < (size_t) ncam * ncam; k++)
        {
            if (psf_b[k] > pk_b)
            {
                pk_b = psf_b[k];
            }
            if (psf_r[k] > pk_r)
            {
                pk_r = psf_r[k];
            }
        }
        double fb = 0.0, fr = 0.0, wtb = 0.0, wtr = 0.0, cxb = 0.0, cxr = 0.0;
        for (uint32_t y = 0; y < ncam; y++)
        {
            for (uint32_t x = 0; x < ncam; x++)
            {
                float vb = psf_b[(size_t) y * ncam + x];
                float vr = psf_r[(size_t) y * ncam + x];
                fb += (double) vb;
                fr += (double) vr;
                if (vb >= 0.05f * pk_b)
                {
                    wtb += (double) vb;
                    cxb += (double) vb * (double) x;
                }
                if (vr >= 0.05f * pk_r)
                {
                    wtr += (double) vr;
                    cxr += (double) vr * (double) x;
                }
            }
        }
        cxb /= wtb;
        cxr /= wtr;
        double tilt_drift = fabs(cxb - cxr);
        double flux_diff  = fabs(fb - fr) / (0.5 * (fb + fr));

        cacaosim_fpcam_free(&xr);
        cacaosim_fpcam_free(&xb);
        free(psf_r);
        free(psf_b);

        snprintf(det, sizeof(det),
                 "SR=(%.5f,%.5f), w_red/w_blue=%.3f (ref 1.50), |dx|=%.3fpx, dFlux=%.3f",
                 sr_mft, sr_fft, w_ratio, tilt_drift, flux_diff);
        check_cond(fabsf(sr_mft - 1.0f) < 1e-5f && fabsf(sr_fft - 1.0f) < 1e-5f &&
                       fabs(w_ratio - 1.50) < 0.08 && tilt_drift < 0.03 && flux_diff < 0.02,
                   "FPCAM MFT/FFT Strehl=1, l/D scale & tilt", det);
    }

    /* 16c. SHWFS subaperture 2D bilinear lambda/D spot scaling & achromatic geometric slope */
    {
        const uint32_t nsub = 4u;
        const uint32_t psub = 16u;
        const uint32_t ncam = nsub * psub;
        float         *im_b = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam,
                                                                sizeof(float));
        float         *im_r = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam,
                                                                sizeof(float));
        cacaosim_det_cfg_t clean_det = {
            .flux_ph      = 1.0e5f,
            .gain_eadu    = 1.0f,
            .shutter      = 1,
            .photon_noise = 0,
            .sat_adu      = 1.0e12f,
            .out_dtype    = CACAOSIM_DTYPE_F32,
        };
        cacaosim_shwfs_cfg_t scfg_b = {
            .nwf          = nwf,
            .n_sub        = nsub,
            .pix_per_sub  = psub,
            .pad_factor   = 4u,
            .pup_diam     = pdia,
            .lambda_um    = 0.75f,
            .bw_frac      = 0.0f,
            .nlambda      = 1,
            .has_ext_chan = 1,
            .wf_unit      = CACAOSIM_WFUNIT_OPD_UM,
            .det          = clean_det,
            .backend      = CACAOSIM_BACKEND_CPU,
        };
        scfg_b.ext_lam_um[0]        = 0.60f;
        scfg_b.ext_weight[0]        = 1.0f;
        cacaosim_shwfs_cfg_t scfg_r = scfg_b;
        scfg_r.ext_lam_um[0]        = 0.90f;

        cacaosim_shwfs_ctx_t sb_ctx, sr_ctx;
        cacaosim_shwfs_init(&sb_ctx, &scfg_b, pup, NULL, NULL);
        cacaosim_shwfs_init(&sr_ctx, &scfg_r, pup, NULL, NULL);

        /* Measure main-lobe spot width (I >= 10% peak) on interior subaperture (sx=1, sy=1) */
        cacaosim_shwfs_accum_beam(&sb_ctx, NULL, pup, NULL, NULL, NULL);
        cacaosim_shwfs_readout(&sb_ctx, &clean_det, NULL, im_b);
        cacaosim_shwfs_accum_beam(&sr_ctx, NULL, pup, NULL, NULL, NULL);
        cacaosim_shwfs_readout(&sr_ctx, &clean_det, NULL, im_r);

        float spk_b = 0.0f, spk_r = 0.0f;
        for (uint32_t ly = 0; ly < psub; ly++)
        {
            for (uint32_t lx = 0; lx < psub; lx++)
            {
                float vb = im_b[(size_t) (psub + ly) * ncam + (psub + lx)];
                float vr = im_r[(size_t) (psub + ly) * ncam + (psub + lx)];
                if (vb > spk_b)
                {
                    spk_b = vb;
                }
                if (vr > spk_r)
                {
                    spk_r = vr;
                }
            }
        }
        double       sum_b = 0.0, sum_r = 0.0, m2_b = 0.0, m2_r = 0.0;
        const double sc0   = 0.5 * (double) (psub - 1u);
        for (uint32_t ly = 0; ly < psub; ly++)
        {
            for (uint32_t lx = 0; lx < psub; lx++)
            {
                double dx = (double) lx - sc0;
                double dy = (double) ly - sc0;
                double r2 = dx * dx + dy * dy;
                float  vb = im_b[(size_t) (psub + ly) * ncam + (psub + lx)];
                float  vr = im_r[(size_t) (psub + ly) * ncam + (psub + lx)];
                if (vb >= 0.10f * spk_b)
                {
                    sum_b += (double) vb;
                    m2_b += (double) vb * r2;
                }
                if (vr >= 0.10f * spk_r)
                {
                    sum_r += (double) vr;
                    m2_r += (double) vr * r2;
                }
            }
        }
        double sh_w_ratio = sqrt((m2_r / sum_r) / (m2_b / sum_b));

        /* Measure subaperture spot centroid shift under achromatic mirror tilt */
        cacaosim_shwfs_accum_beam(&sb_ctx, tilt, pup, NULL, NULL, NULL);
        cacaosim_shwfs_readout(&sb_ctx, &clean_det, NULL, im_b);
        cacaosim_shwfs_accum_beam(&sr_ctx, tilt, pup, NULL, NULL, NULL);
        cacaosim_shwfs_readout(&sr_ctx, &clean_det, NULL, im_r);

        double tb = 0.0, tr = 0.0, cxb = 0.0, cxr = 0.0;
        for (uint32_t ly = 0; ly < psub; ly++)
        {
            for (uint32_t lx = 0; lx < psub; lx++)
            {
                double vb = (double) im_b[(size_t) (psub + ly) * ncam + (psub + lx)];
                double vr = (double) im_r[(size_t) (psub + ly) * ncam + (psub + lx)];
                tb += vb;
                tr += vr;
                cxb += vb * ((double) lx - sc0);
                cxr += vr * ((double) lx - sc0);
            }
        }
        cxb /= tb;
        cxr /= tr;
        double sh_drift = fabs(cxb - cxr);

        cacaosim_shwfs_free(&sr_ctx);
        cacaosim_shwfs_free(&sb_ctx);
        free(im_r);
        free(im_b);

        snprintf(det, sizeof(det), "w_red/w_blue=%.3f (ref 1.50), cx=(%.3f,%.3f) |dcx|=%.3fpx",
                 sh_w_ratio, cxb, cxr, sh_drift);
        check_cond(fabs(sh_w_ratio - 1.50) < 0.12 && fabs(cxb) > 0.4 && sh_drift < 0.03,
                   "SHWFS 2D bilinear l/D spot scale & tilt", det);
    }

    /* 16d. Pyramid WFS default N=2, focal modulation/roof l/D scaling & prism chrom_sep */
    {
        const uint32_t ncam = 64u;
        float *p_ach_b = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
        float *p_ach_r = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
        float *p_chr_b = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
        float *p_chr_r = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));

        cacaosim_det_cfg_t clean_det = {
            .flux_ph      = 1.0e5f,
            .gain_eadu    = 1.0f,
            .shutter      = 1,
            .photon_noise = 0,
            .sat_adu      = 1.0e12f,
            .out_dtype    = CACAOSIM_DTYPE_F32,
        };
        cacaosim_pyrwfs_cfg_t pcfg_def = {
            .nwf        = nwf,
            .pad_factor = 2u,
            .ncam       = ncam,
            .pup_diam   = pdia,
            .lambda_um  = 0.75f,
            .bw_frac    = 0.25f,
            .nlambda    = 0,
            .mod_rad_ld = 2.0f,
            .mod_npts   = 12,
            .pup_sep    = 0.50f,
            .roof_ld    = 0.60f,
            .fstop_frac = 0.45f,
            .chrom_sep  = 0.0f,
            .wf_unit    = CACAOSIM_WFUNIT_OPD_UM,
            .det        = clean_det,
            .backend    = CACAOSIM_BACKEND_CPU,
        };
        cacaosim_pyrwfs_cfg_t pcfg_m1 = pcfg_def;
        pcfg_m1.nlambda               = 1;

        cacaosim_pyrwfs_ctx_t pctx_2, pctx_1;
        cacaosim_pyrwfs_init(&pctx_2, &pcfg_def, pup, NULL, NULL);
        cacaosim_pyrwfs_init(&pctx_1, &pcfg_m1, pup, NULL, NULL);
        int ok_pyr_n = (pctx_2.eff_nlam == 2 && pctx_1.eff_nlam == 1);
        cacaosim_pyrwfs_accum_beam(&pctx_2, ab, pup, NULL, NULL, NULL, 2.0f);
        cacaosim_pyrwfs_readout(&pctx_2, &clean_det, NULL, p_ach_b);
        cacaosim_pyrwfs_accum_beam(&pctx_1, ab, pup, NULL, NULL, NULL, 2.0f);
        cacaosim_pyrwfs_readout(&pctx_1, &clean_det, NULL, p_ach_r);
        double pyr_poly_diff = rel_l2_f32(p_ach_b, p_ach_r, (size_t) ncam * ncam);
        cacaosim_pyrwfs_free(&pctx_1);
        cacaosim_pyrwfs_free(&pctx_2);

        /* Compare radial sub-pupil offset between 0.60 um and 0.90 um for chrom_sep = 0 vs 0.20 */
        cacaosim_pyrwfs_cfg_t pb_ach = pcfg_def;
        pb_ach.nlambda               = 1;
        pb_ach.roof_ld               = 0.0f;
        pb_ach.has_ext_chan          = 1;
        pb_ach.ext_lam_um[0]         = 0.60f;
        pb_ach.ext_weight[0]         = 1.0f;
        cacaosim_pyrwfs_cfg_t pr_ach = pb_ach;
        pr_ach.ext_lam_um[0]         = 0.90f;

        cacaosim_pyrwfs_cfg_t pb_chr = pb_ach;
        pb_chr.chrom_sep             = 0.20f;
        cacaosim_pyrwfs_cfg_t pr_chr = pr_ach;
        pr_chr.chrom_sep             = 0.20f;

        cacaosim_pyrwfs_ctx_t xba, xra, xbc, xrc;
        cacaosim_pyrwfs_init(&xba, &pb_ach, pup, NULL, NULL);
        cacaosim_pyrwfs_init(&xra, &pr_ach, pup, NULL, NULL);
        cacaosim_pyrwfs_init(&xbc, &pb_chr, pup, NULL, NULL);
        cacaosim_pyrwfs_init(&xrc, &pr_chr, pup, NULL, NULL);

        cacaosim_pyrwfs_accum_beam(&xba, NULL, pup, NULL, NULL, NULL, 3.0f);
        cacaosim_pyrwfs_readout(&xba, &clean_det, NULL, p_ach_b);
        cacaosim_pyrwfs_accum_beam(&xra, NULL, pup, NULL, NULL, NULL, 3.0f);
        cacaosim_pyrwfs_readout(&xra, &clean_det, NULL, p_ach_r);
        cacaosim_pyrwfs_accum_beam(&xbc, NULL, pup, NULL, NULL, NULL, 3.0f);
        cacaosim_pyrwfs_readout(&xbc, &clean_det, NULL, p_chr_b);
        cacaosim_pyrwfs_accum_beam(&xrc, NULL, pup, NULL, NULL, NULL, 3.0f);
        cacaosim_pyrwfs_readout(&xrc, &clean_det, NULL, p_chr_r);

        float pk_ab = 0.0f, pk_ar = 0.0f, pk_cb = 0.0f, pk_cr = 0.0f;
        for (size_t k = 0; k < (size_t) ncam * ncam; k++)
        {
            if (p_ach_b[k] > pk_ab)
            {
                pk_ab = p_ach_b[k];
            }
            if (p_ach_r[k] > pk_ar)
            {
                pk_ar = p_ach_r[k];
            }
            if (p_chr_b[k] > pk_cb)
            {
                pk_cb = p_chr_b[k];
            }
            if (p_chr_r[k] > pk_cr)
            {
                pk_cr = p_chr_r[k];
            }
        }
        const double pc = 0.5 * (double) (ncam - 1u);
        double       sab = 0.0, sar = 0.0, scb = 0.0, scr = 0.0;
        double       rab = 0.0, rar = 0.0, rcb = 0.0, rcr = 0.0;
        for (uint32_t y = 0; y < ncam; y++)
        {
            for (uint32_t x = 0; x < ncam; x++)
            {
                double r = hypot((double) x - pc, (double) y - pc);
                size_t k = (size_t) y * ncam + x;
                if (p_ach_b[k] >= 0.25f * pk_ab)
                {
                    sab += 1.0;
                    rab += r;
                }
                if (p_ach_r[k] >= 0.25f * pk_ar)
                {
                    sar += 1.0;
                    rar += r;
                }
                if (p_chr_b[k] >= 0.25f * pk_cb)
                {
                    scb += 1.0;
                    rcb += r;
                }
                if (p_chr_r[k] >= 0.25f * pk_cr)
                {
                    scr += 1.0;
                    rcr += r;
                }
            }
        }
        double dr_ach = fabs((rar / sar) - (rab / sab));
        double dr_chr = (rcr / scr) - (rcb / scb);

        cacaosim_pyrwfs_free(&xrc);
        cacaosim_pyrwfs_free(&xbc);
        cacaosim_pyrwfs_free(&xra);
        cacaosim_pyrwfs_free(&xba);
        free(p_chr_r);
        free(p_chr_b);
        free(p_ach_r);
        free(p_ach_b);

        snprintf(det, sizeof(det),
                 "eff_nlam=%d, poly_vs_mono=%.2e, |dr_achrom|=%.3fpx, dr_chrom=%.2fpx",
                 ok_pyr_n ? 2 : 0, pyr_poly_diff, dr_ach, dr_chr);
        check_cond(ok_pyr_n && pyr_poly_diff > 1e-3 && dr_ach < 0.06 && dr_chr > 0.80,
                   "PyrWFS N=2 default, focal l/D & chrom_sep", det);
    }

    /* 16e. ZWFS dispersive vs achromatic dimple & 1D lambda linear phasor interpolation */
    {
        const uint32_t ncam = 64u;
        float *z_disp = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
        float *z_achr = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));
        float *z_ref  = (float *) cacaosim_aligned_calloc((size_t) ncam * ncam, sizeof(float));

        cacaosim_det_cfg_t clean_det = {
            .flux_ph      = 1.0e5f,
            .gain_eadu    = 1.0f,
            .shutter      = 1,
            .photon_noise = 0,
            .sat_adu      = 1.0e12f,
            .out_dtype    = CACAOSIM_DTYPE_F32,
        };
        cacaosim_zwfs_cfg_t zcfg_d = {
            .nwf           = nwf,
            .ncam          = ncam,
            .n_mft         = 32u,
            .pup_diam      = pdia,
            .lambda_um     = 0.75f,
            .bw_frac       = 0.40f,
            .nlambda       = 5,
            .dimple_ld     = 1.06f,
            .dimple_rad    = 0.5f * CACAOSIM_PI_F,
            .dimple_amp    = 1.0f,
            .achrom_dimple = 0,
            .wf_unit       = CACAOSIM_WFUNIT_OPD_UM,
            .det           = clean_det,
            .backend       = CACAOSIM_BACKEND_CPU,
        };
        cacaosim_zwfs_cfg_t zcfg_a = zcfg_d;
        zcfg_a.achrom_dimple       = 1;

        cacaosim_zwfs_ctx_t zctx_d, zctx_a;
        cacaosim_zwfs_init(&zctx_d, &zcfg_d, pup, NULL, NULL);
        cacaosim_zwfs_init(&zctx_a, &zcfg_a, pup, NULL, NULL);
        cacaosim_zwfs_accum_beam(&zctx_d, ab, pup, NULL, NULL, NULL);
        cacaosim_zwfs_readout(&zctx_d, &clean_det, NULL, z_disp);
        cacaosim_zwfs_accum_beam(&zctx_a, ab, pup, NULL, NULL, NULL);
        cacaosim_zwfs_readout(&zctx_a, &clean_det, NULL, z_achr);
        double zw_disp_diff = rel_l2_f32(z_disp, z_achr, (size_t) ncam * ncam);
        cacaosim_zwfs_free(&zctx_a);
        cacaosim_zwfs_free(&zctx_d);

        /* 3-channel upstream polychromatic beam at {1.50, 1.65, 1.80} um interpolated at 1.575 */
        const float beam_lam[3] = { 1.50f, 1.65f, 1.80f };
        const float amp_s[3]    = { 0.60f, 1.00f, 1.40f };
        float      *b_opd       = (float *) cacaosim_aligned_calloc(npix * 3u, sizeof(float));
        float      *b_amp       = (float *) cacaosim_aligned_calloc(npix * 3u, sizeof(float));
        float      *m_amp       = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        for (uint32_t c = 0; c < 3u; c++)
        {
            for (uint64_t p = 0; p < npix; p++)
            {
                b_opd[(size_t) c * npix + p] = ab[p];
                b_amp[(size_t) c * npix + p] = pup[p] * amp_s[c];
            }
        }
        /* Exact midpoint alpha = 0.5 between 1.50 (0.60) and 1.65 (1.00) -> 0.80 * pup */
        for (uint64_t p = 0; p < npix; p++)
        {
            m_amp[p] = pup[p] * 0.80f;
        }

        cacaosim_zwfs_cfg_t zcfg_i = zcfg_d;
        zcfg_i.lambda_um           = 1.575f;
        zcfg_i.bw_frac             = 0.0f;
        zcfg_i.nlambda             = 1;
        cacaosim_zwfs_ctx_t zctx_i;
        cacaosim_zwfs_init(&zctx_i, &zcfg_i, pup, NULL, NULL);

        cacaosim_zwfs_accum_beam_poly(&zctx_i, b_opd, b_amp, beam_lam, 3u, npix, NULL, NULL, NULL);
        cacaosim_zwfs_readout(&zctx_i, &clean_det, NULL, z_disp);
        cacaosim_zwfs_accum_beam(&zctx_i, ab, m_amp, NULL, NULL, NULL);
        cacaosim_zwfs_readout(&zctx_i, &clean_det, NULL, z_ref);
        double interp_err = rel_l2_f32(z_disp, z_ref, (size_t) ncam * ncam);

        cacaosim_zwfs_free(&zctx_i);
        free(m_amp);
        free(b_amp);
        free(b_opd);
        free(z_ref);
        free(z_achr);
        free(z_disp);

        snprintf(det, sizeof(det), "disp_vs_achrom rel_L2=%.3e, poly_lerp_err=%.2e",
                 zw_disp_diff, interp_err);
        check_cond(zw_disp_diff > 1e-2 && interp_err < 1e-6,
                   "ZWFS dispersive dimple & 1D lambda phasor lerp", det);
    }

    /* 16f. Polychromatic CPU vs CUDA parity across all 4 sensors + broadband .bench DAG */
    {
        const char *bench_txt =
            "[bench]\n"
            "name = \"chrom_dag\"\n"
            "nwf = 64\n"
            "pup_diam = 0.875\n"
            "cobs = 0.28\n"
            "nspider = 4\n"
            "tel_diam_m = 8.2\n"
            "star_mag = 7.0\n"
            "master_hz = 1000\n"
            "out_beam = \"b_in\"\n"
            "[splitter.sp0]\n"
            "in = \"b_in\"\n"
            "out_refl = \"b_vis\"\n"
            "out_trans = \"b_nir\"\n"
            "mode = \"dichroic_lp\"\n"
            "lambda_cut = 1.00\n"
            "[pyrwfs.ho]\n"
            "in = \"b_vis\"\n"
            "out_cam = \"pyr_cam\"\n"
            "ncam = 64\n"
            "lambda_um = 0.75\n"
            "bw_frac = 0.22\n"
            "mod_rad = 2.5\n"
            "mod_npts = 12\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "bias_adu = 0.0\n"
            "out_dtype = \"float32\"\n"
            "[coro.c0]\n"
            "in = \"b_nir\"\n"
            "out = \"b_sci\"\n"
            "out_lowfs = \"b_lo\"\n"
            "mode = \"lyot_spot\"\n"
            "lambda_um = 1.65\n"
            "dlambda_um = 0.33\n"
            "fpm_rad_ld = 2.8\n"
            "fpm_amp = 0.10\n"
            "lowfs_mode = \"fpm_refl\"\n"
            "lowfs_refl = 0.95\n"
            "[zwfs.lo]\n"
            "in = \"b_lo\"\n"
            "out_cam = \"lo_cam\"\n"
            "ncam = 64\n"
            "lambda_um = 1.65\n"
            "bw_frac = 0.20\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "bias_adu = 0.0\n"
            "out_dtype = \"float32\"\n"
            "[shwfs.aux]\n"
            "in = \"b_vis\"\n"
            "out_cam = \"sh_cam\"\n"
            "n_sub = 8\n"
            "pix_per_sub = 8\n"
            "lambda_um = 0.80\n"
            "dlambda_um = 0.16\n"
            "nlambda = 3\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "bias_adu = 0.0\n"
            "out_dtype = \"float32\"\n"
            "[fpcam.sci]\n"
            "in = \"b_sci\"\n"
            "out_cam = \"sci_cam\"\n"
            "ncam = 64\n"
            "prop_mode = \"fft\"\n"
            "pad_factor = 2\n"
            "lambda_um = 1.65\n"
            "bw_frac = 0.20\n"
            "nlambda = 3\n"
            "photon_noise = 0\n"
            "ron_e = 0.0\n"
            "bias_adu = 0.0\n"
            "out_dtype = \"float32\"\n";

        char                 err[256];
        cacaosim_bench_cfg_t cfg_c, cfg_rt;
        int ok_p = (cacaosim_bench_parse_str(bench_txt, &cfg_c, err, sizeof(err)) == 0);
        cfg_c.backend = CACAOSIM_BACKEND_CPU;

        char fmt_buf[8192];
        cacaosim_bench_format_str(&cfg_c, fmt_buf, sizeof(fmt_buf));
        int ok_rt = (cacaosim_bench_parse_str(fmt_buf, &cfg_rt, err, sizeof(err)) == 0);

        cacaosim_bench_ctx_t ctx_c;
        int ok_ic = ok_p && (cacaosim_bench_init(&ctx_c, &cfg_c, pup, 0, err, sizeof(err)) == 0);
        int ok_ch = 0;
        if (ok_ic)
        {
            /* Verify auto N=2 on pyrwfs, coro, zwfs and explicit N=3 on shwfs, fpcam */
            ok_ch = 1;
            for (uint32_t i = 0; i < ctx_c.n_nodes; i++)
            {
                const cacaosim_bench_node_t *nd = &ctx_c.nodes[i];
                if (strcmp(nd->cfg.id, "ho") == 0 && nd->eff_nlam != 2)
                {
                    ok_ch = 0;
                }
                else if (strcmp(nd->cfg.id, "c0") == 0 &&
                         (nd->coro_ctx.eff_nlam != 2 ||
                          ctx_c.beams[nd->lowfs_idx].nlam != 2u))
                {
                    ok_ch = 0;
                }
                else if (strcmp(nd->cfg.id, "lo") == 0 && nd->eff_nlam != 2)
                {
                    ok_ch = 0;
                }
                else if (strcmp(nd->cfg.id, "aux") == 0 && nd->eff_nlam != 3)
                {
                    ok_ch = 0;
                }
                else if (strcmp(nd->cfg.id, "sci") == 0 && nd->eff_nlam != 3)
                {
                    ok_ch = 0;
                }
            }
            cacaosim_bench_step(&ctx_c, ab, NULL, 1.0f, NULL);
        }

        double max_gpu_err = 0.0;
        double mft_gpu_err = 0.0;
        if (cacaosim_cuda_available() && ok_ic)
        {
            cacaosim_bench_cfg_t cfg_g = cfg_rt;
            cfg_g.backend              = CACAOSIM_BACKEND_CUDA;
            cacaosim_bench_ctx_t ctx_g;
            if (cacaosim_bench_init(&ctx_g, &cfg_g, pup, 0, err, sizeof(err)) == 0)
            {
                cacaosim_bench_step(&ctx_g, ab, NULL, 1.0f, NULL);
                for (uint32_t s = 0; s < ctx_c.n_sensors; s++)
                {
                    double e = rel_l2_f32(cacaosim_bench_sensor_f32(&ctx_c, s),
                                          cacaosim_bench_sensor_f32(&ctx_g, s), 64u * 64u);
                    max_gpu_err = fmax(max_gpu_err, e);
                }
                cacaosim_bench_free(&ctx_g);
            }
            else
            {
                max_gpu_err = 1.0;
            }

            /* Also check FPCAM PROP_MFT 3-channel broadband CPU vs CUDA parity */
            cfg_c.nodes[5].fpcam.prop_mode = CACAOSIM_FPCAM_PROP_MFT;
            cfg_g                          = cfg_c;
            cfg_g.backend                  = CACAOSIM_BACKEND_CUDA;
            cacaosim_bench_ctx_t mc, mg;
            if (cacaosim_bench_init(&mc, &cfg_c, pup, 0, err, sizeof(err)) == 0 &&
                cacaosim_bench_init(&mg, &cfg_g, pup, 0, err, sizeof(err)) == 0)
            {
                cacaosim_bench_step(&mc, ab, NULL, 1.0f, NULL);
                cacaosim_bench_step(&mg, ab, NULL, 1.0f, NULL);
                mft_gpu_err = rel_l2_f32(cacaosim_bench_sensor_f32(&mc, 3),
                                         cacaosim_bench_sensor_f32(&mg, 3), 64u * 64u);
                cacaosim_bench_free(&mg);
                cacaosim_bench_free(&mc);
            }
            else
            {
                mft_gpu_err = 1.0;
            }
        }

        if (ok_ic)
        {
            cacaosim_bench_free(&ctx_c);
        }

        snprintf(det, sizeof(det),
                 "ok_ch=%d, ok_rt=%d, max_4sens_GPU_L2=%.2e, fpcam_MFT_L2=%.2e",
                 ok_ch, ok_rt, max_gpu_err, mft_gpu_err);
        check_cond(ok_ic && ok_rt && ok_ch && max_gpu_err < 3e-5 && mft_gpu_err < 3e-5,
                   "Polychromatic 4-sensor DAG & CPU/CUDA parity", det);
    }

    free(ab);
    free(tilt);
    free(pup);
}

static void test_subaru_lyot_coro_and_llowfs(void)
{
    printf("\n[17] Subaru Telescope Pupil, Lyot Coronagraph & Defocused Lyot-Refl LLOWFS\n");

    const uint32_t nwf  = 128u;
    const uint64_t npix = (uint64_t) nwf * nwf;
    float         *pup  = (float *) calloc(npix, sizeof(float));
    float         *opd  = (float *) calloc(npix, sizeof(float));
    char           err[256] = { 0 };
    char           det[256] = { 0 };

    /* 1. Subaru Nasmyth-IR offset-V spider geometry & 4-quadrant mirror symmetry */
    cacaosim_pupil_cfg_t scfg = {
        .nwf           = nwf,
        .pup_diam      = 0.9375f,
        .cobs          = 0.307f,
        .nspider       = 4,
        .spider_w      = 0.02314f,
        .spider_deg    = 51.75f,
        .spider_offset = 0.08289f,
    };
    double clear_area = cacaosim_make_pupil(&scfg, pup);

    double max_sym_err = 0.0;
    for (uint32_t y = 0; y < nwf; y++)
    {
        uint32_t ym = nwf - 1u - y;
        for (uint32_t x = 0; x < nwf; x++)
        {
            uint32_t xm = nwf - 1u - x;
            float    v  = pup[y * nwf + x];
            max_sym_err = fmax(max_sym_err, fabs((double) v - (double) pup[y * nwf + xm]));
            max_sym_err = fmax(max_sym_err, fabs((double) v - (double) pup[ym * nwf + x]));
        }
    }

    const float ctr    = 0.5f * (float) (nwf - 1u);
    const float d_px   = scfg.pup_diam * (float) nwf;
    const float ang    = 51.75f * (CACAOSIM_PI_F / 180.0f);
    /* Sample midpoint along the Q1 Subaru V-vane: (off_px + s*cos(51.75), s*sin(51.75)) */
    const float s_test = 0.32f * d_px;
    uint32_t    vx     = (uint32_t) lroundf(ctr + 0.08289f * d_px + s_test * cosf(ang));
    uint32_t    vy     = (uint32_t) lroundf(ctr + s_test * sinf(ang));
    /* Sample radial 45-deg point at r = 0.25*D (clear on Subaru because V-vane is offset +X) */
    uint32_t    dx45   = (uint32_t) lroundf(ctr + 0.175f * d_px);
    uint32_t    dy45   = (uint32_t) lroundf(ctr + 0.175f * d_px);
    float       v_vane = pup[vy * nwf + vx];
    float       v_diag = pup[dy45 * nwf + dx45];

    snprintf(det, sizeof(det), "area=%.1f px, sym_err=%.1e, amp_subaru_vane=%.3f, amp_45deg=%.3f",
             clear_area, max_sym_err, v_vane, v_diag);
    check_cond(clear_area > 8000.0 && max_sym_err < 1e-6 && v_vane < 0.05f && v_diag > 0.99f,
               "Subaru Nasmyth-IR offset-V spider pupil geometry", det);

    /* 2. .bench round-trip, skip_fpcam & calib_fast focal-plane WFS, and LLOWFS sensitivity */
    const char *bench_txt =
        "[bench]\n"
        "name          = \"subaru_coro_llowfs\"\n"
        "nwf           = 128\n"
        "pup_diam      = 0.9375\n"
        "pupil_geom    = \"subaru\"\n"
        "tput_tel      = 0.20\n"
        "star_mag      = 5.0\n"
        "master_hz     = 2000.0\n"
        "backend       = \"cpu\"\n"
        "\n"
        "[coro.coro_nir]\n"
        "in                 = \"b_pup\"\n"
        "out                = \"b_coro\"\n"
        "out_lowfs          = \"b_lowfs\"\n"
        "out_pre_lyot       = \"coro_pre_lyot\"\n"
        "out_pre_lyot_long  = \"coro_pre_lyot_long\"\n"
        "mode               = \"lyot_spot\"\n"
        "lambda_um          = 1.65\n"
        "bw_frac            = 0.18\n"
        "nlambda            = 3\n"
        "fpm_rad_ld         = 3.0\n"
        "fpm_amp            = 0.0\n"
        "lowfs_mode         = \"lyot_refl\"\n"
        "lowfs_refl         = 0.95\n"
        "tput               = 0.92\n"
        "lyot_outer         = 0.85\n"
        "lyot_obs           = 0.38\n"
        "lyot_nspider       = 4\n"
        "lyot_spider_w      = 0.040\n"
        "lyot_spider_deg    = 51.75\n"
        "lyot_spider_offset = 0.08289\n"
        "\n"
        "[fpcam.nir_coro]\n"
        "in            = \"b_coro\"\n"
        "out_cam       = \"psf_nir_coro\"\n"
        "out_long      = \"psf_nir_coro_long\"\n"
        "ncam          = 64\n"
        "prop_mode     = \"mft\"\n"
        "lambda_um     = 1.65\n"
        "bw_frac       = 0.18\n"
        "nlambda       = 3\n"
        "pixscale_ld   = 2.5\n"
        "photon_noise  = 0\n"
        "ron_e         = 0.0\n"
        "bias_adu      = 0.0\n"
        "out_dtype     = \"float32\"\n"
        "\n"
        "[wfe.lowfs_defoc]\n"
        "in         = \"b_lowfs\"\n"
        "out        = \"b_lowfs_defoc\"\n"
        "in_wfe     = \"csim_lowfs_defoc\"\n"
        "rms_nm     = 0.0\n"
        "defocus_nm = 200.0\n"
        "wfe_gain   = 1.0\n"
        "tput       = 0.98\n"
        "\n"
        "[fpcam.wfs_lowfs]\n"
        "in            = \"b_lowfs_defoc\"\n"
        "out_cam       = \"csim_wfs_lowfs\"\n"
        "is_wfs        = 1\n"
        "ncam          = 64\n"
        "prop_mode     = \"mft\"\n"
        "lambda_um     = 1.65\n"
        "bw_frac       = 0.18\n"
        "nlambda       = 3\n"
        "pixscale_ld   = 2.0\n"
        "photon_noise  = 0\n"
        "ron_e         = 0.0\n"
        "bias_adu      = 0.0\n"
        "out_dtype     = \"float32\"\n";

    cacaosim_bench_cfg_t b0, b1;
    char                 fmt_buf[8192];
    int ok_p0 = (cacaosim_bench_parse_str(bench_txt, &b0, err, sizeof(err)) == 0);
    int flen  = ok_p0 ? cacaosim_bench_format_str(&b0, fmt_buf, sizeof(fmt_buf)) : -1;
    int ok_rt = (flen > 0 && cacaosim_bench_parse_str(fmt_buf, &b1, err, sizeof(err)) == 0 &&
                 fabsf(b1.tel_diam_m - 7.95f) < 1e-4f &&
                 fabsf(b1.cobs - 0.307f) < 1e-4f &&
                 fabsf(b1.spider_offset - 0.08289f) < 1e-4f &&
                 fabsf(b1.tput_tel - 0.20f) < 1e-4f &&
                 fabsf(b1.nodes[0].coro.lyot_spider_offset - 0.08289f) < 1e-4f &&
                 strcmp(b1.nodes[0].out_pre_lyot, "coro_pre_lyot") == 0 &&
                 strcmp(b1.nodes[0].out_pre_lyot_long, "coro_pre_lyot_long") == 0 &&
                 fabsf(b1.nodes[2].wfe_defocus_nm - 200.0f) < 1e-3f &&
                 b1.nodes[3].is_wfs == 1);

    cacaosim_bench_ctx_t ctx;
    int ok_init = ok_rt && (cacaosim_bench_init(&ctx, &b1, NULL, 0, err, sizeof(err)) == 0);
    if (ok_init)
    {
        /* Verify oversized Lyot stop masks 100% of entrance spider vanes inside [0.39, 0.84] D */
        const float *lyot = ctx.nodes[0].coro_ctx.lyot_mask;
        float        leak = 0.0f;
        for (uint32_t y = 0; y < nwf; y++)
        {
            float dy = ((float) y - ctr) / (0.5f * d_px);
            for (uint32_t x = 0; x < nwf; x++)
            {
                float dx = ((float) x - ctr) / (0.5f * d_px);
                float r  = hypotf(dx, dy);
                if (r > 0.39f && r < 0.84f)
                {
                    float blocked_in = 1.0f - ctx.pup_ptr[y * nwf + x];
                    if (blocked_in > 0.5f && lyot[y * nwf + x] > leak)
                    {
                        leak = lyot[y * nwf + x];
                    }
                }
            }
        }

        /* Nominal flat wavefront step: check coronagraph rejection & skip_fpcam WFS preservation */
        cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
        float  coro_sr    = ctx.last_strehl_inst;
        float  f_pre_dl   = ctx.nodes[0].coro_ctx.last_frac_pre_lyot;
        float  f_tr_dl    = ctx.nodes[0].coro_ctx.last_frac_lyot_trans;
        float  f_lo_dl    = ctx.nodes[0].coro_ctx.last_frac_lowfs;
        double e_part_err = fabs((double) f_tr_dl + (double) f_lo_dl / (0.95 * 0.95) -
                                 (double) f_pre_dl);

        int s_coro  = cacaosim_bench_find_sensor(&ctx, "nir_coro");
        int s_lowfs = cacaosim_bench_find_sensor(&ctx, "wfs_lowfs");
        float *i0   = (float *) calloc(64u * 64u, sizeof(float));
        float *ip   = (float *) calloc(64u * 64u, sizeof(float));
        float *im   = (float *) calloc(64u * 64u, sizeof(float));
        memcpy(i0, cacaosim_bench_sensor_f32(&ctx, s_lowfs), 64u * 64u * sizeof(float));

        ctx.skip_fpcam = 1;
        cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);
        int skip_ok = (ctx.nodes[ctx.sensor_node_idx[s_coro]].ready_this_tick == 0 &&
                       ctx.nodes[ctx.sensor_node_idx[s_lowfs]].ready_this_tick == 1 &&
                       fabsf(ctx.last_strehl_inst - coro_sr) < 1e-7f);
        ctx.skip_fpcam = 0;

        /* Test +/-10 nm tip and +/-10 nm defocus push-pull linearity in calib_fast mode */
        ctx.calib_fast = 1;
        snprintf(ctx.calib_wfs_out, sizeof(ctx.calib_wfs_out), "csim_wfs_lowfs");
        b1.wf_unit = CACAOSIM_WFUNIT_OPD_UM;
        ctx.cfg.wf_unit = CACAOSIM_WFUNIT_OPD_UM;

        for (uint32_t y = 0; y < nwf; y++)
        {
            for (uint32_t x = 0; x < nwf; x++)
            {
                float xn         = ((float) x - ctr) / (0.5f * d_px);
                opd[y * nwf + x] = 0.010f * 2.0f * xn; /* +10 nm RMS Tip */
            }
        }
        cacaosim_bench_step(&ctx, opd, NULL, 1.0f, NULL);
        memcpy(ip, cacaosim_bench_sensor_f32(&ctx, s_lowfs), 64u * 64u * sizeof(float));
        cacaosim_bench_step(&ctx, opd, NULL, -1.0f, NULL);
        memcpy(im, cacaosim_bench_sensor_f32(&ctx, s_lowfs), 64u * 64u * sizeof(float));

        double tip_dot = 0.0, tip_np = 0.0, tip_nm = 0.0;
        for (size_t k = 0; k < 64u * 64u; k++)
        {
            double dp = (double) ip[k] - (double) i0[k];
            double dm = (double) im[k] - (double) i0[k];
            tip_dot += dp * dm;
            tip_np += dp * dp;
            tip_nm += dm * dm;
        }
        double tip_cos = tip_dot / (sqrt(tip_np * tip_nm) + 1e-30);

        for (uint32_t y = 0; y < nwf; y++)
        {
            float yn = ((float) y - ctr) / (0.5f * d_px);
            for (uint32_t x = 0; x < nwf; x++)
            {
                float xn         = ((float) x - ctr) / (0.5f * d_px);
                float r2         = xn * xn + yn * yn;
                opd[y * nwf + x] = 0.010f * 1.7320508f * (2.0f * r2 - 1.0f); /* +10 nm Z4 */
            }
        }
        cacaosim_bench_step(&ctx, opd, NULL, 1.0f, NULL);
        memcpy(ip, cacaosim_bench_sensor_f32(&ctx, s_lowfs), 64u * 64u * sizeof(float));
        cacaosim_bench_step(&ctx, opd, NULL, -1.0f, NULL);
        memcpy(im, cacaosim_bench_sensor_f32(&ctx, s_lowfs), 64u * 64u * sizeof(float));

        double foc_dot = 0.0, foc_np = 0.0, foc_nm = 0.0;
        for (size_t k = 0; k < 64u * 64u; k++)
        {
            double dp = (double) ip[k] - (double) i0[k];
            double dm = (double) im[k] - (double) i0[k];
            foc_dot += dp * dm;
            foc_np += dp * dp;
            foc_nm += dm * dm;
        }
        double foc_cos = foc_dot / (sqrt(foc_np * foc_nm) + 1e-30);

        double gpu_err = 0.0;
        if (cacaosim_cuda_available())
        {
            ctx.calib_fast       = 0;
            ctx.calib_wfs_out[0] = '\0';
            cacaosim_bench_step(&ctx, opd, NULL, 1.0f, NULL);

            cacaosim_bench_cfg_t bg = b1;
            bg.backend              = CACAOSIM_BACKEND_CUDA;
            bg.wf_unit              = CACAOSIM_WFUNIT_OPD_UM;
            cacaosim_bench_ctx_t cg;
            if (cacaosim_bench_init(&cg, &bg, NULL, 0, err, sizeof(err)) == 0)
            {
                cacaosim_bench_step(&cg, opd, NULL, 1.0f, NULL);
                gpu_err = fmax(rel_l2_f32(cacaosim_bench_sensor_f32(&ctx, s_coro),
                                          cacaosim_bench_sensor_f32(&cg, s_coro), 64u * 64u),
                               rel_l2_f32(cacaosim_bench_sensor_f32(&ctx, s_lowfs),
                                          cacaosim_bench_sensor_f32(&cg, s_lowfs), 64u * 64u));
                gpu_err = fmax(gpu_err,
                               rel_l2_f32(ctx.nodes[0].coro_ctx.cpu_pre_lyot_int,
                                          cg.nodes[0].coro_ctx.cpu_pre_lyot_int, nwf * nwf));
                gpu_err = fmax(gpu_err,
                               fabs((double) ctx.nodes[0].coro_ctx.last_frac_lowfs -
                                    (double) cg.nodes[0].coro_ctx.last_frac_lowfs));
                cacaosim_bench_free(&cg);
            }
            else
            {
                gpu_err = 1.0;
            }
        }

        snprintf(det, sizeof(det),
                 "f_pre=%.3f, f_lo=%.3f, part=%.2e, coro_sr=%.2e, tip=%.3f, foc=%.3f, gpu=%.2e",
                 f_pre_dl, f_lo_dl, e_part_err, coro_sr, tip_cos, foc_cos, gpu_err);
        check_cond(leak < 1e-6f && coro_sr < 3e-3f && f_pre_dl > 0.08f && f_pre_dl < 0.20f &&
                       f_lo_dl > 0.07f && f_lo_dl < 0.18f && e_part_err < 1e-5 && skip_ok &&
                       tip_np > 1.0 && tip_cos < -0.98 && foc_np > 1.0 && foc_cos < -0.95 &&
                       gpu_err < 5e-5,
                   "Subaru Lyot coro rejection, pre-Lyot/LOWFS budget & GPU parity", det);

        free(im);
        free(ip);
        free(i0);
        cacaosim_bench_free(&ctx);
    }
    else
    {
        check_cond(0, "Subaru Lyot coro .bench parse/init", err);
    }

    free(opd);
    free(pup);
}

static const char *sample_val_for_opt(const cacaosim_bench_opt_t *o, char *buf, size_t cap)
{
    if (strcmp(o->key, "nwf") == 0)
    {
        return "64";
    }
    if (o->enum_values[0] != '\0')
    {
        const char *bar = strchr(o->enum_values, '|');
        size_t      len = (bar != NULL) ? (size_t) (bar - o->enum_values) : strlen(o->enum_values);
        if (len >= cap)
        {
            len = cap - 1u;
        }
        memcpy(buf, o->enum_values, len);
        buf[len] = '\0';
        return buf;
    }
    if (strcmp(o->type, "string") == 0)
    {
        return "test_tok";
    }
    if (strcmp(o->type, "bool") == 0)
    {
        return "1";
    }
    return "1";
}

static int tui_has_key_for_item(const cacaosim_bench_cfg_t *cfg, int sel_item, const char *key)
{
    cs_param_desc_t pds[CS_MAX_PARAMS];
    for (int st = 0; st < 3; st++)
    {
        int n = cs_params_build(cfg, sel_item, st, 1, pds, CS_MAX_PARAMS);
        for (int i = 0; i < n; i++)
        {
            if (pds[i].key != NULL && strcmp(pds[i].key, key) == 0)
            {
                return 1;
            }
        }
    }
    return 0;
}

static void test_options_schema_and_tui_coverage(void)
{
    printf("\n[18] Options Schema SSOT, FPS Introspection & TUI Advanced Coverage\n");

    /* 1. Verify every canonical .bench key and alias parses cleanly through cacaosim_bench_parse */
    size_t n_opts      = cacaosim_schema_bench_count();
    int    parse_fails = 0;
    char   first_fail[256] = { 0 };

    for (size_t i = 0u; i < n_opts; i++)
    {
        const cacaosim_bench_opt_t *o = cacaosim_schema_bench_get(i);
        char                        sec_tok[32];
        const char                 *comma = strchr(o->applies_to, ',');
        size_t                      slen =
            (comma != NULL) ? (size_t) (comma - o->applies_to) : strlen(o->applies_to);
        if (slen >= sizeof(sec_tok))
        {
            slen = sizeof(sec_tok) - 1u;
        }
        memcpy(sec_tok, o->applies_to, slen);
        sec_tok[slen] = '\0';

        char        sval_buf[64];
        const char *sval = sample_val_for_opt(o, sval_buf, sizeof(sval_buf));

        /* Test canonical key + each comma-separated alias */
        char key_list[128];
        if (o->aliases[0] != '\0')
        {
            snprintf(key_list, sizeof(key_list), "%s,%s", o->key, o->aliases);
        }
        else
        {
            snprintf(key_list, sizeof(key_list), "%s", o->key);
        }
        char *p = key_list;
        while (*p != '\0')
        {
            char *next = strchr(p, ',');
            if (next != NULL)
            {
                *next = '\0';
            }
            char                 doc[512];
            cacaosim_bench_cfg_t tcfg;
            char                 err[160] = { 0 };
            if (strcmp(sec_tok, "bench") == 0)
            {
                snprintf(doc, sizeof(doc),
                         "[bench]\n%s = \"%s\"\n[fpcam.d0]\nin = \"b_pup\"\n", p, sval);
            }
            else
            {
                snprintf(doc, sizeof(doc), "[bench]\nnwf = 64\n[%s.t0]\nin = \"b_pup\"\n%s = \"%s\"\n",
                         sec_tok, p, sval);
            }
            if (cacaosim_bench_parse_str(doc, &tcfg, err, sizeof(err)) != 0)
            {
                if (parse_fails == 0)
                {
                    snprintf(first_fail, sizeof(first_fail), "%s.%s: %s", sec_tok, p, err);
                }
                parse_fails++;
            }
            p = (next != NULL) ? (next + 1) : (p + strlen(p));
        }
    }
    char det[256];
    snprintf(det, sizeof(det), "checked=%zu keys+aliases, fails=%d %s", n_opts, parse_fails,
             first_fail);
    check_cond(n_opts >= 120u && parse_fails == 0,
               "Every schema .bench key & alias accepted by parser", det);

    /* 2. Build exhaustive non-default .bench covering every node/mode & check schema + TUI */
    cacaosim_bench_cfg_t ecfg;
    cacaosim_bench_cfg_defaults(&ecfg);
    ecfg.spider_offset    = 0.08f;
    ecfg.oversample       = 4u;
    ecfg.atm_interp       = 0;
    ecfg.telemetry_stride = 2u;
    snprintf(ecfg.in_amp, sizeof(ecfg.in_amp), "atm_amp");
    snprintf(ecfg.in_ncpa, sizeof(ecfg.in_ncpa), "ent_ncpa");
    ecfg.wf_unit              = CACAOSIM_WFUNIT_OPD_UM;
    ecfg.ref_lambda           = 0.65f;
    ecfg.turb_gain            = 0.8f;
    ecfg.perturb.vib_enable   = 1;
    ecfg.perturb.vib_rms_nm   = 35.0f;
    ecfg.perturb.vib_seed     = 111ULL;
    ecfg.perturb.petal_enable = 1;
    ecfg.perturb.petal_rms_nm = 180.0f;
    ecfg.perturb.petal_tau_s  = 2.5f;
    ecfg.perturb.petal_seed   = 222ULL;
    ecfg.backend              = CACAOSIM_BACKEND_CPU;

    cacaosim_node_cfg_t *nd;
    /* 0: DM zonal RC + delay_frames + reg */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_DM, "dm_rc", 0, 0);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_pup");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_1");
    snprintf(nd->out_opd, sizeof(nd->out_opd), "dm_opd");
    snprintf(nd->in_flat, sizeof(nd->in_flat), "dm_flat");
    nd->dm.dm_diam     = 14.0f;
    nd->dm.refl_scale  = 1.8f;
    nd->dm.nonlin_quad = 0.05f;
    nd->reg.shift_x    = 0.5f;
    nd->reg.shift_y    = -0.5f;
    nd->reg.rot_deg    = 2.0f;
    nd->reg.scale      = 1.02f;

    /* 1: DM zonal OSC + delay_us + zoh_mean=0 */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_DM, "dm_osc", 0, 1);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_1");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_2");
    nd->dm.dyn_mode    = 2;
    nd->dm.delay_sec   = 2.5e-4f;
    nd->dm.no_zoh_mean = 1;

    /* 1b: DM zonal PLATE PDE (ALPAO 64x64 coupled facesheet) */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_DM, "dm_plt", 0, 1);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_2");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_2p");
    nd->dm.dyn_mode = 3;

    /* 2: TTM */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_DM, "ttm0", 1, 2);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_2p");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_3");

    /* 3: Splitter dichroic_bp with all per-arm reg/ncpa/pup */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_SPLITTER, "bs_bp", 0, 3);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_3");
    snprintf(nd->out_refl, sizeof(nd->out_refl), "b_r1");
    snprintf(nd->out_trans, sizeof(nd->out_trans), "b_t1");
    nd->split.mode        = CACAOSIM_SPLIT_DICHROIC_BP;
    nd->split.roll_um     = 0.05f;
    nd->reg_refl.shift_x  = 0.2f;
    nd->reg_refl.shift_y  = 0.3f;
    nd->reg_refl.rot_deg  = 1.0f;
    nd->reg_refl.scale    = 0.99f;
    nd->reg_trans.shift_x = -0.2f;
    nd->reg_trans.shift_y = -0.3f;
    nd->reg_trans.rot_deg = -1.0f;
    nd->reg_trans.scale   = 1.01f;
    snprintf(nd->ncpa_refl, sizeof(nd->ncpa_refl), "nr");
    snprintf(nd->ncpa_trans, sizeof(nd->ncpa_trans), "nt");
    snprintf(nd->pup_refl, sizeof(nd->pup_refl), "pr");
    snprintf(nd->pup_trans, sizeof(nd->pup_trans), "pt");

    /* 4: Splitter gray */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_SPLITTER, "bs_g", 0, 4);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_t1");
    snprintf(nd->out_refl, sizeof(nd->out_refl), "b_r2");
    snprintf(nd->out_trans, sizeof(nd->out_trans), "b_t2");
    nd->split.mode = CACAOSIM_SPLIT_GRAY;

    /* 5: Coro phase_dimple + pre_lyot + custom masks + lyot_spider_offset */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_CORO, "co_pd", 0, 5);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_t2");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_c1");
    snprintf(nd->out_lowfs, sizeof(nd->out_lowfs), "b_lo1");
    snprintf(nd->out_pre_lyot, sizeof(nd->out_pre_lyot), "pl_s");
    snprintf(nd->out_pre_lyot_long, sizeof(nd->out_pre_lyot_long), "pl_l");
    snprintf(nd->in_pup_amp, sizeof(nd->in_pup_amp), "pa");
    snprintf(nd->in_pup_pha, sizeof(nd->in_pup_pha), "pp");
    snprintf(nd->in_fpm_amp, sizeof(nd->in_fpm_amp), "fa");
    snprintf(nd->in_fpm_pha, sizeof(nd->in_fpm_pha), "fp");
    snprintf(nd->in_lyot, sizeof(nd->in_lyot), "la");
    snprintf(nd->in_lyot_pha, sizeof(nd->in_lyot_pha), "lp");
    nd->coro.mode               = CACAOSIM_CORO_PHASE_DIMPLE;
    nd->coro.bw_frac            = 0.15f;
    nd->coro.nlambda            = 3;
    nd->coro.mask_pha_unit      = 1;
    nd->coro.n_mft              = 64u;
    nd->coro.lowfs_pup_diam     = 0.9f;
    nd->coro.lyot_spider_deg    = 51.0f;
    nd->coro.lyot_spider_offset = 0.08f;

    /* 6: Coro vortex + lyot_refl + pad_factor */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_CORO, "co_vx", 0, 6);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_c1");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_c2");
    nd->coro.mode       = CACAOSIM_CORO_VORTEX;
    nd->coro.pad_factor = 6u;
    nd->coro.lowfs_mode = CACAOSIM_LOWFS_LYOT_REFL;

    /* 7: Coro custom_mft */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_CORO, "co_cm", 0, 7);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_c2");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_c3");
    nd->coro.mode = CACAOSIM_CORO_CUSTOM_MFT;

    /* 8: WFE */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_WFE, "wfe0", 0, 8);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_c3");
    snprintf(nd->out_beam, sizeof(nd->out_beam), "b_w1");
    nd->wfe_defocus_nm = 25.0f;
    nd->wfe_alpha      = 2.2f;

    /* 9: PyWFS with all advanced fields + det + timing + sim_hz */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_PYRWFS, "pyr0", 0, 9);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_r1");
    snprintf(nd->in_ncpa, sizeof(nd->in_ncpa), "p_ncpa");
    nd->tput                     = 0.90f;
    nd->rate_hz                  = 1000.0f;
    nd->clock_offset_us          = 125.0f;
    nd->clock_drift_ppm          = 15.0f;
    nd->sim_hz                   = 4000.0f;
    nd->use_beam_flux            = 0;
    nd->pyrwfs.pad_factor        = 4u;
    nd->pyrwfs.mod_subframe_mode = CACAOSIM_PYRMOD_FULL;
    nd->pyrwfs.mod_rev           = 2;
    nd->pyrwfs.mod_phase_deg     = 30.0f;
    nd->pyrwfs.pup_sep           = 0.45f;
    nd->pyrwfs.roof_ld           = 0.2f;
    nd->pyrwfs.fstop_frac        = 0.40f;
    nd->pyrwfs.chrom_sep         = 0.05f;
    nd->pyrwfs.nlambda           = 3;
    nd->pyrwfs.det.dark_e        = 1.5f;
    nd->pyrwfs.det.gain_eadu     = 2.0f;
    nd->pyrwfs.det.bias_adu      = 200.0f;
    nd->pyrwfs.det.sat_adu       = 32000.0f;
    nd->pyrwfs.det.shutter       = 0;

    /* 10: SHWFS with subframe_stride & navg */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_SHWFS, "sh0", 0, 10);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_r2");
    nd->subframe_stride  = 2u;
    nd->shwfs.pad_factor = 4u;
    nd->shwfs.nlambda    = 2;

    /* 11: ZWFS with dimple_amp & achrom_dimple */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_ZWFS, "zw0", 0, 11);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_lo1");
    nd->zwfs.n_mft         = 48u;
    nd->zwfs.dimple_rad    = 1.2f;
    nd->zwfs.dimple_amp    = 0.8f;
    nd->zwfs.achrom_dimple = 1;
    nd->zwfs.nlambda       = 2;

    /* 12: FPCAM fft + is_wfs + out_long */
    nd = &ecfg.nodes[ecfg.n_nodes++];
    cacaosim_bench_node_defaults(nd, CACAOSIM_NODE_FPCAM, "fp0", 0, 12);
    snprintf(nd->in_beam, sizeof(nd->in_beam), "b_w1");
    snprintf(nd->out_long, sizeof(nd->out_long), "fp0_long");
    nd->is_wfs           = 1;
    nd->fpcam.prop_mode  = CACAOSIM_FPCAM_PROP_FFT;
    nd->fpcam.pad_factor = 4u;
    nd->fpcam.nlambda    = 3;

    int   need = cacaosim_bench_format_str(&ecfg, NULL, 0u);
    char *sbuf = (char *) malloc((size_t) need + 16u);
    cacaosim_bench_format_str(&ecfg, sbuf, (size_t) need + 16u);

    int  emitted_keys  = 0;
    int  schema_misses = 0;
    int  tui_misses    = 0;
    char miss_msg[160] = { 0 };
    char cur_sec[32]   = { 0 };
    int  cur_sel       = 0;

    char *lp = sbuf;
    while (*lp != '\0')
    {
        char *nl = strchr(lp, '\n');
        if (nl != NULL)
        {
            *nl = '\0';
        }
        while (*lp == ' ' || *lp == '\t')
        {
            lp++;
        }
        if (lp[0] == '[')
        {
            char *rb = strchr(lp, ']');
            if (rb != NULL)
            {
                *rb = '\0';
                char *dot = strchr(lp + 1, '.');
                if (dot == NULL)
                {
                    snprintf(cur_sec, sizeof(cur_sec), "%s", lp + 1);
                    cur_sel = 0;
                }
                else
                {
                    *dot = '\0';
                    snprintf(cur_sec, sizeof(cur_sec), "%s", lp + 1);
                    const char *nid = dot + 1;
                    cur_sel         = 0;
                    for (uint32_t ni = 0; ni < ecfg.n_nodes; ni++)
                    {
                        if (strcmp(ecfg.nodes[ni].id, nid) == 0)
                        {
                            cur_sel = (int) ni + 1;
                            break;
                        }
                    }
                }
            }
        }
        else if (lp[0] != '\0' && lp[0] != '#')
        {
            char *eq = strchr(lp, '=');
            if (eq != NULL)
            {
                *eq = '\0';
                char *ke = eq;
                while (ke > lp && (ke[-1] == ' ' || ke[-1] == '\t'))
                {
                    *--ke = '\0';
                }
                emitted_keys++;
                if (cacaosim_schema_find_bench_opt(cur_sec, lp) == NULL)
                {
                    if (schema_misses == 0)
                    {
                        snprintf(miss_msg, sizeof(miss_msg), "schema miss: %s.%s", cur_sec, lp);
                    }
                    schema_misses++;
                }
                if (!tui_has_key_for_item(&ecfg, cur_sel, lp))
                {
                    if (tui_misses == 0 && miss_msg[0] == '\0')
                    {
                        snprintf(miss_msg, sizeof(miss_msg), "tui miss: %s.%s (sel=%d)", cur_sec,
                                 lp, cur_sel);
                    }
                    tui_misses++;
                }
            }
        }
        lp = (nl != NULL) ? (nl + 1) : (lp + strlen(lp));
    }
    free(sbuf);

    snprintf(det, sizeof(det), "emitted=%d, schema_miss=%d, tui_miss=%d %s", emitted_keys,
             schema_misses, tui_misses, miss_msg);
    check_cond(emitted_keys >= 140 && schema_misses == 0 && tui_misses == 0,
               "100% serializer -> schema & TUI Advanced inspector coverage", det);

    /* 3. Verify runtime introspection of all 12 MILK_RUNNABLE FPS units & exporters */
    size_t                      n_units   = 0u;
    const MILK_RUNNABLE *const *units     = cacaosim_list_runnables(&n_units);
    size_t                      n_bindings = 0u;
    for (size_t u = 0u; u < n_units; u++)
    {
        n_bindings += (size_t) units[u]->nb_bindings;
    }

    FILE  *jf       = tmpfile();
    int    jrc      = (jf != NULL) ? cacaosim_schema_render_json(NULL, jf) : -1;
    long   jsz      = 0;
    int    j_has_ok = 0;
    if (jf != NULL && jrc == 0)
    {
        fflush(jf);
        jsz = ftell(jf);
        rewind(jf);
        char *jb = (char *) malloc((size_t) jsz + 1u);
        if (jb != NULL)
        {
            size_t nr = fread(jb, 1u, (size_t) jsz, jf);
            jb[nr]    = '\0';
            j_has_ok  = (strstr(jb, "\"bench_options\"") != NULL &&
                         strstr(jb, "\"fps_units\"") != NULL &&
                         strstr(jb, "\"wfstat\"") != NULL &&
                         strstr(jb, "\"adc\"") != NULL &&
                         strstr(jb, "\"clock_offset_us\"") != NULL);
            free(jb);
        }
        fclose(jf);
    }

    FILE *mf  = tmpfile();
    int   mrc = (mf != NULL) ? cacaosim_schema_render_markdown(NULL, mf) : -1;
    long  msz = (mf != NULL) ? (fflush(mf), ftell(mf)) : 0;
    if (mf != NULL)
    {
        fclose(mf);
    }

    FILE *tf  = tmpfile();
    int   trc = (tf != NULL) ? cacaosim_schema_render_table(NULL, 0, tf) : -1;
    long  tsz = (tf != NULL) ? (fflush(tf), ftell(tf)) : 0;
    if (tf != NULL)
    {
        fclose(tf);
    }

    snprintf(det, sizeof(det), "units=%zu, fps_params=%zu, json=%ldB, md=%ldB, tbl=%ldB", n_units,
             n_bindings, jsz, msz, tsz);
    check_cond(n_units == 13u && n_bindings >= 380u && jrc == 0 && j_has_ok && mrc == 0 &&
                   msz > 10000L && trc == 0 && tsz > 10000L,
               "13 MILK_RUNNABLE FPS units introspected & JSON/MD/Table exported", det);
}

static void test_entrance_perturbations(void)
{
    printf("\n[19] Entrance-Pupil Vibrations & Thermally Induced Petaling (LWE)\n");

    const uint32_t       nwf  = 128u;
    const uint64_t       npix = (uint64_t) nwf * nwf;
    float               *pup  = (float *) calloc(npix, sizeof(float));
    float               *opd  = (float *) calloc(npix, sizeof(float));
    cacaosim_pupil_cfg_t pcfg = {
        .nwf           = nwf,
        .pup_diam      = 0.9375f,
        .cobs          = 0.307f,
        .nspider       = 4,
        .spider_w      = 0.02314f,
        .spider_deg    = 51.75f,
        .spider_offset = 0.08289f,
    };
    cacaosim_make_pupil(&pcfg, pup);

    cacaosim_perturb_cfg_t cfg;
    cacaosim_perturb_cfg_defaults(&cfg);
    cfg.vib_enable   = 1;
    cfg.vib_rms_nm   = 45.0f;
    cfg.petal_enable = 1;
    cfg.petal_rms_nm = 250.0f;
    cfg.petal_tau_s  = 2.0f;

    cacaosim_perturb_ctx_t ctx;
    int                    irc = cacaosim_perturb_init(&ctx, nwf, pcfg.pup_diam, pup, &cfg, 0,
                                                       NULL);
    char                   det[256];
    snprintf(det, sizeof(det), "irc=%d, n_petals=%d, n_modes=%d", irc, ctx.n_petals, ctx.n_modes);
    check_cond(irc == 0 && ctx.n_petals == 4 && ctx.n_modes == 17,
               "Subaru 4-V-spider pupil segmented into 4 petals (17 modes)", det);

    /* Scale == 0 must leave OPD identically zero (flat calibration safety) */
    memset(opd, 0, npix * sizeof(float));
    cacaosim_perturb_step_apply(&ctx, &cfg, 0.0005, 0.0f, opd, NULL, NULL);
    double zmax = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (fabs((double) opd[i]) > zmax)
        {
            zmax = fabs((double) opd[i]);
        }
    }
    snprintf(det, sizeof(det), "max |opd| = %.2e um", zmax);
    check_cond(zmax == 0.0, "Zero perturbation injected when turb_gain == 0", det);

    /* Measure vibration RMS & 49.5 Hz resonant tone over 4.0 s @ 2 kHz */
    cfg.petal_enable = 0;
    const int nstep  = 8000;
    double    dt_s   = 0.0005;
    double    sum_v2 = 0.0;
    double    re49 = 0.0, im49 = 0.0, re200 = 0.0, im200 = 0.0;
    for (int t = 0; t < nstep; t++)
    {
        memset(opd, 0, npix * sizeof(float));
        cacaosim_perturb_step_apply(&ctx, &cfg, dt_s, 1.0f, opd, NULL, NULL);
        sum_v2 += (double) ctx.tlm.vib_rms_nm * (double) ctx.tlm.vib_rms_nm;
        double ang49  = 2.0 * 3.14159265358979323846 * 49.5 * (double) t * dt_s;
        double ang200 = 2.0 * 3.14159265358979323846 * 200.0 * (double) t * dt_s;
        re49 += (double) ctx.tlm.vib_tip_nm * cos(ang49);
        im49 += (double) ctx.tlm.vib_tip_nm * sin(ang49);
        re200 += (double) ctx.tlm.vib_tip_nm * cos(ang200);
        im200 += (double) ctx.tlm.vib_tip_nm * sin(ang200);
    }
    double vib_rms = sqrt(sum_v2 / (double) nstep);
    double p49     = re49 * re49 + im49 * im49;
    double p200    = re200 * re200 + im200 * im200;
    snprintf(det, sizeof(det), "rms=%.1f nm (target 45), P(49.5Hz)/P(200Hz)=%.1fx", vib_rms,
             (p200 > 1e-12) ? (p49 / p200) : 9999.0);
    check_cond(vib_rms > 32.0 && vib_rms < 58.0 && p49 > 10.0 * p200,
               "Stochastic vibrations match target RMS and 49.5 Hz resonant PSD peak", det);

    /* Measure thermal petaling: zero global mean, slow temporal evolution, realistic RMS */
    cfg.vib_enable   = 0;
    cfg.petal_enable = 1;
    double prev_p0 = 0.0, sum_dp2 = 0.0, sum_p02 = 0.0, mean_abs = 0.0;
    for (int t = 0; t < 2000; t++)
    {
        memset(opd, 0, npix * sizeof(float));
        cacaosim_perturb_step_apply(&ctx, &cfg, dt_s, 1.0f, opd, NULL, NULL);
        if (t == 0)
        {
            double s = 0.0, cnt = 0.0;
            for (uint64_t i = 0; i < npix; i++)
            {
                if (pup[i] > 0.5f)
                {
                    s += (double) opd[i];
                    cnt += 1.0;
                }
            }
            mean_abs = (cnt > 0.0) ? fabs(s / cnt) : 0.0;
        }
        else
        {
            double d = (double) ctx.tlm.petal_p_nm[0] - prev_p0;
            sum_dp2 += d * d;
        }
        prev_p0 = (double) ctx.tlm.petal_p_nm[0];
        sum_p02 += prev_p0 * prev_p0;
    }
    double step_rms = sqrt(sum_dp2 / 1999.0);
    double p0_rms   = sqrt(sum_p02 / 2000.0);
    snprintf(det, sizeof(det), "|mean|=%.2e um, step_rms=%.3f nm vs p0_rms=%.1f nm", mean_abs,
             step_rms, p0_rms);
    check_cond(mean_abs < 1.0e-3 && step_rms < 0.02 * (p0_rms + 1.0) &&
                   ctx.tlm.petal_rms_nm > 10.0f,
               "Thermal petaling has zero global piston & smooth slow temporal evolution", det);

    cacaosim_perturb_free(&ctx);
    free(pup);
    free(opd);
}

/**
 * @brief Section 20: Maunakea atmospheric dispersion, AO188 ADC (Egner 2009), 3D Snell doublet,
 *        5-engine CPU/CUDA chromatic parity, coronagraph no-double-count, and bench round-trip.
 */
static void test_atm_disp_and_ao188_adc(void)
{
    printf("\n=== 20. Atmospheric Dispersion, AO188 ADC (Egner 2009) & Chromatic Parity ===\n");
    char det[220];
    char err[256];

    /* 20a. Maunakea Ciddor dispersion (z=30, 60 deg) + Chebyshev-12 fit accuracy */
    cacaosim_atm_disp_cfg_t atm;
    cacaosim_atm_disp_cfg_defaults(&atm);
    atm.enable       = 1;
    atm.zenith_deg   = 30.0f;
    atm.disp_pa_deg  = 0.0f;
    atm.pressure_hpa = 615.0f;
    atm.temp_c       = 2.0f;
    atm.rh_pct       = 20.0f;
    atm.ref_lam_um   = 1.65f;
    atm.sphere_corr  = 1;

    cacaosim_disp_t d_atm30;
    memset(&d_atm30, 0, sizeof(d_atm30));
    cacaosim_disp_set_atm(&d_atm30, &atm);

    double max_cheb_mas = 0.0;
    for (int i = 0; i <= 40; i++)
    {
        double lam  = 0.42 + (2.38 - 0.42) * ((double) i / 40.0);
        double ex_r = cacaosim_atm_disp_rad(lam, &atm);
        double tx = 0.0, ty = 0.0;
        cacaosim_disp_eval_rad(&d_atm30, lam, &tx, &ty);
        double em = hypot(tx, ty - ex_r) * 206264806.24709636;
        if (em > max_cheb_mas)
        {
            max_cheb_mas = em;
        }
    }
    double span_h30   = cacaosim_disp_band_span_mas(&d_atm30, 1.500, 1.780);
    double span_vis30 = cacaosim_disp_band_span_mas(&d_atm30, 0.675, 0.825);
    double span_bb30  = cacaosim_disp_band_span_mas(&d_atm30, 0.500, 2.200);
    snprintf(det, sizeof(det), "z=30: H=%.2f, Vis=%.1f, 0.5-2.2=%.1f mas, Cheb_err=%.2e mas",
             span_h30, span_vis30, span_bb30, max_cheb_mas);
    check_cond(span_h30 > 15.0 && span_h30 < 25.0 && span_vis30 > 80.0 && span_vis30 < 120.0 &&
                   span_bb30 > 450.0 && span_bb30 < 620.0 && max_cheb_mas < 1e-3,
               "Maunakea Ciddor dispersion & Chebyshev-12 fit", det);

    /* 20b. AO188 Egner (2009) Table 11 measured & nominal LUT at z=60 deg */
    cacaosim_atm_disp_cfg_t atm60 = atm;
    atm60.zenith_deg              = 60.0f;
    cacaosim_disp_t d_atm60;
    memset(&d_atm60, 0, sizeof(d_atm60));
    cacaosim_disp_set_atm(&d_atm60, &atm60);

    cacaosim_adc_cfg_t adc_h;
    cacaosim_adc_cfg_defaults(&adc_h);
    adc_h.preset     = CACAOSIM_ADC_PRESET_AO188;
    adc_h.mode       = CACAOSIM_ADC_MODE_LUT;
    adc_h.opt_band   = CACAOSIM_ADC_OPT_H;
    adc_h.ideal_mech = 1;

    cacaosim_adc_ctx_t actx_h;
    cacaosim_disp_t    d_out_h;
    memset(&d_out_h, 0, sizeof(d_out_h));
    cacaosim_adc_init(&actx_h, &adc_h, &atm60);
    cacaosim_adc_step(&actx_h, 0.001, &atm60, NULL, &d_atm60, NULL, &d_out_h, NULL);
    double res_h60 = cacaosim_disp_band_span_mas(&d_out_h, 1.500, 1.780);

    cacaosim_adc_cfg_t adc_w = adc_h;
    adc_w.opt_band           = CACAOSIM_ADC_OPT_WHOLE;
    cacaosim_adc_ctx_t actx_w;
    cacaosim_disp_t    d_out_w;
    memset(&d_out_w, 0, sizeof(d_out_w));
    cacaosim_adc_init(&actx_w, &adc_w, &atm60);
    cacaosim_adc_step(&actx_w, 0.001, &atm60, NULL, &d_atm60, NULL, &d_out_w, NULL);
    double res_bb60  = cacaosim_disp_band_span_mas(&d_out_w, 0.500, 2.200);
    double res_h60_w = cacaosim_disp_band_span_mas(&d_out_w, 1.500, 1.780);
    double res_v60_w = cacaosim_disp_band_span_mas(&d_out_w, 0.675, 0.825);

    cacaosim_adc_cfg_t adc_nom_mis = adc_h;
    adc_nom_mis.lut_use_nominal    = 1;
    cacaosim_adc_ctx_t actx_nm;
    cacaosim_disp_t    d_out_nm;
    memset(&d_out_nm, 0, sizeof(d_out_nm));
    cacaosim_adc_init(&actx_nm, &adc_nom_mis, &atm60);
    cacaosim_adc_step(&actx_nm, 0.001, &atm60, NULL, &d_atm60, NULL, &d_out_nm, NULL);
    double res_h60_nom = cacaosim_disp_band_span_mas(&d_out_nm, 1.500, 1.780);

    snprintf(det, sizeof(det),
             "z=60: opt(H) resH=%.2f, opt(whole) bb=%.1f H=%.2f Vis=%.1f, nom-err=%.2f mas",
             res_h60, res_bb60, res_h60_w, res_v60_w, res_h60_nom);
    check_cond(res_h60 < 1.5 && res_bb60 < 80.0 && res_h60_w < 6.0 && res_v60_w < 25.0 &&
                   res_h60_nom > res_h60 + 0.4 &&
                   fabsf(actx_h.state_vec[CACAOSIM_ADC_ST_RES_H] - (float) res_h60) < 1e-3f &&
                   actx_h.state_vec[CACAOSIM_ADC_ST_DEPLOYED] > 0.5f,
               "AO188 Egner Table 11 LUT & nominal mismatch", det);

    /* 20c. Exact 3D vector Snell doublet: zero disp at psi=0 + Chebyshev fit accuracy */
    cacaosim_adc_cfg_t adc_sn;
    cacaosim_adc_cfg_defaults(&adc_sn);
    adc_sn.preset     = CACAOSIM_ADC_PRESET_SNELL_DOUBLET;
    adc_sn.mode       = CACAOSIM_ADC_MODE_MANUAL;
    adc_sn.angle1_deg = -90.0f;
    adc_sn.angle2_deg = 90.0f;
    double sx0 = 0.0, sy0 = 0.0, sx1 = 0.0, sy1 = 0.0;
    cacaosim_adc_eval_deflection_rad(&adc_sn, 0.70, -90.0, 90.0, &sx0, &sy0);
    cacaosim_adc_eval_deflection_rad(&adc_sn, 1.65, -90.0, 90.0, &sx1, &sy1);
    double zero_psi_mas = hypot(sx0 - sx1, sy0 - sy1) * 206264806.24709636;

    adc_sn.angle1_deg = -60.0f; /* psi = 30 deg */
    adc_sn.angle2_deg = 60.0f;
    cacaosim_adc_ctx_t actx_sn;
    cacaosim_disp_t    d_sn;
    memset(&d_sn, 0, sizeof(d_sn));
    cacaosim_adc_init(&actx_sn, &adc_sn, &atm);
    cacaosim_adc_step(&actx_sn, 0.001, &atm, NULL, NULL, NULL, &d_sn, NULL);
    double sn_span = cacaosim_disp_band_span_mas(&d_sn, 0.50, 2.20);
    double sn_cheb = 0.0;
    double ex0 = 0.0, ey0 = 0.0;
    cacaosim_adc_eval_deflection_rad(&adc_sn, (double) atm.ref_lam_um, -60.0, 60.0, &ex0, &ey0);
    for (int i = 0; i <= 20; i++)
    {
        double lam = 0.45 + (2.35 - 0.45) * ((double) i / 20.0);
        double ex = 0.0, ey = 0.0, cx = 0.0, cy = 0.0;
        cacaosim_adc_eval_deflection_rad(&adc_sn, lam, -60.0, 60.0, &ex, &ey);
        cacaosim_disp_eval_rad(&d_sn, lam, &cx, &cy);
        double em = hypot(cx - (ex - ex0), cy - (ey - ey0)) * 206264806.24709636;
        if (em > sn_cheb)
        {
            sn_cheb = em;
        }
    }
    snprintf(det, sizeof(det), "psi=0 span=%.2e mas, psi=30 span=%.1f mas, Cheb=%.2e mas",
             zero_psi_mas, sn_span, sn_cheb);
    check_cond(zero_psi_mas < 1e-8 && sn_span > 50.0 && sn_cheb < 0.25,
               "3D vector Snell doublet zero-psi & Chebyshev fit", det);

    /* 20d. Egner Table 29 mechanical realism: encoder quantization, slew rate, backlash */
    cacaosim_adc_cfg_t adc_m;
    cacaosim_adc_cfg_defaults(&adc_m);
    adc_m.mode       = CACAOSIM_ADC_MODE_SHM;
    adc_m.angle1_deg = 0.0f;
    adc_m.angle2_deg = 0.0f;
    adc_m.ideal_mech = 0;
    cacaosim_adc_ctx_t actx_m;
    cacaosim_adc_init(&actx_m, &adc_m, &atm);
    float cmd_fwd[2] = { 10.0f, -10.0f };
    cacaosim_adc_step(&actx_m, 0.1, &atm, cmd_fwd, NULL, NULL, NULL, NULL);
    double step_slew = actx_m.cur_th1_deg; /* max_slew=3.75 deg/s * 0.1s = 0.375 deg */
    float  cmd_rev[2] = { -10.0f, 10.0f };
    cacaosim_adc_step(&actx_m, 0.1, &atm, cmd_rev, NULL, NULL, NULL, NULL);
    double bl_seen = fabs(actx_m.bl_off1_deg);
    snprintf(det, sizeof(det), "0.1s slew=%.4f deg (cap 0.375), rev backlash=%.3f deg", step_slew,
             bl_seen);
    check_cond(fabs(step_slew - 0.375) < 1e-4 && fabs(bl_seen - 0.100) < 1e-4,
               "Egner Table 29 slew cap & backlash hysteresis", det);

    /* 20e. 5-engine chromatic dispersion + 3-wl real-OPD CPU vs CUDA parity & no-double-count */
    const uint32_t nwf  = 64u;
    const uint64_t npix = (uint64_t) nwf * nwf;
    const char    *bench_tmpl =
        "[bench]\n"
        "name = \"adc_val\"\n"
        "nwf = 64\n"
        "pup_diam = 0.875\n"
        "tel_diam_m = 7.95\n"
        "cobs = 0.30\n"
        "wf_unit = 0\n"
        "atm_disp_enable = 1\n"
        "zenith_deg = 45.0\n"
        "disp_pa_deg = 25.0\n"
        "atm_pressure_hpa = 615.0\n"
        "atm_temp_c = 2.0\n"
        "atm_rh_pct = 20.0\n"
        "atm_disp_ref_um = 1.65\n"
        "chrom_turb = 1\n"
        "chrom_lambdas = \"0.75,1.22,1.65\"\n"
        "out_beam = \"b_pup\"\n"
        "\n"
        "[adc.ao188]\n"
        "in = \"b_pup\"\n"
        "out = \"b_adc\"\n"
        "preset = \"ao188\"\n"
        "mode = \"%s\"\n"
        "opt_band = \"H\"\n"
        "ideal_mech = 1\n"
        "tput_enable = 0\n"
        "\n"
        "[fpcam.cam_vis]\n"
        "in = \"b_pup\"\n"
        "ncam = 64\n"
        "lambda_um = 1.22\n"
        "bw_frac = 0.02\n"
        "nlambda = 3\n"
        "pixscale_ld = 3.0\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n"
        "\n"
        "[fpcam.cam_h]\n"
        "in = \"b_adc\"\n"
        "ncam = 64\n"
        "lambda_um = 1.65\n"
        "bw_frac = 0.18\n"
        "nlambda = 5\n"
        "pixscale_ld = 3.0\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n"
        "\n"
        "[pyrwfs.pyr]\n"
        "in = \"b_adc\"\n"
        "ncam = 64\n"
        "pad_factor = 2\n"
        "lambda_um = 0.75\n"
        "bw_frac = 0.16\n"
        "nlambda = 3\n"
        "mod_rad = 3.0\n"
        "mod_npts = 8\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n"
        "\n"
        "[shwfs.sh]\n"
        "in = \"b_adc\"\n"
        "n_sub = 4\n"
        "pix_per_sub = 8\n"
        "lambda_um = 1.22\n"
        "bw_frac = 0.18\n"
        "nlambda = 3\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n"
        "\n"
        "[zwfs.zw]\n"
        "in = \"b_adc\"\n"
        "ncam = 64\n"
        "lambda_um = 1.65\n"
        "bw_frac = 0.18\n"
        "nlambda = 3\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n"
        "\n"
        "[coro.cor]\n"
        "in = \"b_adc\"\n"
        "out = \"b_cor\"\n"
        "out_lowfs = \"b_lo\"\n"
        "mode = \"lyot_spot\"\n"
        "lambda_um = 1.65\n"
        "bw_frac = 0.18\n"
        "nlambda = 3\n"
        "fpm_rad_ld = 1.5\n"
        "lyot_outer = 0.85\n"
        "lyot_obs = 0.36\n"
        "lowfs_mode = \"lyot_refl\"\n"
        "\n"
        "[fpcam.cam_cor]\n"
        "in = \"b_cor\"\n"
        "ncam = 64\n"
        "lambda_um = 1.65\n"
        "bw_frac = 0.18\n"
        "nlambda = 3\n"
        "pixscale_ld = 3.0\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n"
        "\n"
        "[zwfs.zw_lo]\n"
        "in = \"b_lo\"\n"
        "ncam = 64\n"
        "lambda_um = 1.65\n"
        "bw_frac = 0.18\n"
        "nlambda = 3\n"
        "photon_noise = 0\n"
        "ron_e = 0.0\n"
        "bias_adu = 0.0\n"
        "out_dtype = \"float32\"\n"
        "use_beam_flux = 0\n"
        "flux_ph = 100000\n";

    char bench_ret[4096];
    char bench_lut[4096];
    snprintf(bench_ret, sizeof(bench_ret), bench_tmpl, "retracted");
    snprintf(bench_lut, sizeof(bench_lut), bench_tmpl, "lut");

    cacaosim_bench_cfg_t cfg_ret, cfg_lut;
    int p1 = cacaosim_bench_parse_str(bench_ret, &cfg_ret, err, sizeof(err));
    int p2 = cacaosim_bench_parse_str(bench_lut, &cfg_lut, err, sizeof(err));
    cfg_ret.backend = CACAOSIM_BACKEND_CPU;
    cfg_lut.backend = CACAOSIM_BACKEND_CPU;

    /* 3-slice distinct chromatic OPD cube [0.75, 1.22, 1.65] um */
    float *wf3 = (float *) cacaosim_aligned_calloc(3u * npix, sizeof(float));
    float *pup_tmp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    cacaosim_pupil_cfg_t pcfg = { .nwf = nwf, .pup_diam = 0.875f, .cobs = 0.30f };
    cacaosim_make_pupil(&pcfg, pup_tmp);
    cacaosim_make_ncpa(nwf, 0.875f, pup_tmp, 0.06f, 7701ULL, wf3);
    cacaosim_make_ncpa(nwf, 0.875f, pup_tmp, 0.06f, 7702ULL, wf3 + npix);
    cacaosim_make_ncpa(nwf, 0.875f, pup_tmp, 0.06f, 7703ULL, wf3 + 2u * npix);

    /* Centroid test on pure flat WF at z=45, PA=25: cam_vis shifts vs ref 1.65 um */
    float *wf_flat3 = (float *) cacaosim_aligned_calloc(3u * npix, sizeof(float));
    cacaosim_bench_ctx_t ctx_ret_c, ctx_lut_c;
    int i1 = (p1 == 0) ? cacaosim_bench_init(&ctx_ret_c, &cfg_ret, NULL, 0, err, sizeof(err)) : -1;
    int i2 = (p2 == 0) ? cacaosim_bench_init(&ctx_lut_c, &cfg_lut, NULL, 0, err, sizeof(err)) : -1;

    snprintf(det, sizeof(det), "p1=%d p2=%d i1=%d i2=%d err=%.100s", p1, p2, i1, i2,
             (p1 || p2 || i1 || i2) ? err : "ok");
    check_cond(p1 == 0 && p2 == 0 && i1 == 0 && i2 == 0,
               "5-engine chrom+ADC bench parse & init", det);
    if (i1 == 0 && i2 == 0)
    {
        /* Step with flat 3-slice WF to measure pure chromatic PSF shift and coronagraph leakage */
        cacaosim_bench_step(&ctx_ret_c, wf_flat3, NULL, 1.0f, NULL);
        cacaosim_bench_step(&ctx_lut_c, wf_flat3, NULL, 1.0f, NULL);

        int          s_vis = cacaosim_bench_find_sensor(&ctx_ret_c, "cam_vis");
        const float *im_v  = cacaosim_bench_sensor_f32(&ctx_ret_c, s_vis);
        double       sum_i = 0.0, cx = 0.0, cy = 0.0;
        for (uint32_t y = 0; y < 64u; y++)
        {
            for (uint32_t x = 0; x < 64u; x++)
            {
                double v = (double) im_v[(size_t) y * 64u + x];
                sum_i += v;
                cx += v * (double) x;
                cy += v * (double) y;
            }
        }
        cx /= sum_i;
        cy /= sum_i;
        /* Expected shift of J=1.22 um (ref 1.65 um) at z=45, PA=25, D=7.95, 3 px/(l/D) */
        cacaosim_disp_t d_ref;
        memset(&d_ref, 0, sizeof(d_ref));
        cacaosim_disp_set_atm(&d_ref, &cfg_ret.atm_disp);
        double tx = 0.0, ty = 0.0;
        cacaosim_disp_eval_rad(&d_ref, 1.22, &tx, &ty);
        double ld_rad = 1.22e-6 / 7.95;
        double ex_dx  = (tx / ld_rad) * 3.0;
        double ex_dy  = (ty / ld_rad) * 3.0;
        double meas_r = hypot(cx - 32.0, cy - 32.0);
        double exp_r  = hypot(ex_dx, ex_dy);
        double r_err  = fabs(meas_r - exp_r);
        snprintf(det, sizeof(det), "centroid=(%.2f,%.2f), |dr|=%.2f px vs analytic %.2f px (err=%.3f)",
                 cx, cy, meas_r, exp_r, r_err);
        check_cond(r_err < 0.15 && meas_r > 3.0,
                   "FPCAM chromatic PSF shift matches Ciddor dispersion", det);

        /* Coronagraph no-double-count: post-coro beams clear disp.active & ADC recovers contrast */
        int    cor_b_idx = -1;
        int    lo_b_idx  = -1;
        for (uint32_t b = 0; b < ctx_ret_c.n_beams; b++)
        {
            if (strcmp(ctx_ret_c.beams[b].name, "b_cor") == 0)
            {
                cor_b_idx = (int) b;
            }
            if (strcmp(ctx_ret_c.beams[b].name, "b_lo") == 0)
            {
                lo_b_idx = (int) b;
            }
        }
        int          s_h_r   = cacaosim_bench_find_sensor(&ctx_ret_c, "cam_h");
        int          s_h_l   = cacaosim_bench_find_sensor(&ctx_lut_c, "cam_h");
        const float *im_h_r  = cacaosim_bench_sensor_f32(&ctx_ret_c, s_h_r);
        const float *im_h_l  = cacaosim_bench_sensor_f32(&ctx_lut_c, s_h_l);
        float        pk_h_r  = 0.0f;
        float        pk_h_l  = 0.0f;
        for (size_t p = 0; p < 64u * 64u; p++)
        {
            if (im_h_r[p] > pk_h_r)
            {
                pk_h_r = im_h_r[p];
            }
            if (im_h_l[p] > pk_h_l)
            {
                pk_h_l = im_h_l[p];
            }
        }
        int          s_cr_r  = cacaosim_bench_find_sensor(&ctx_ret_c, "cam_cor");
        int          s_cr_l  = cacaosim_bench_find_sensor(&ctx_lut_c, "cam_cor");
        const float *im_cr_r = cacaosim_bench_sensor_f32(&ctx_ret_c, s_cr_r);
        const float *im_cr_l = cacaosim_bench_sensor_f32(&ctx_lut_c, s_cr_l);
        double       sum_cr_r = 0.0, sum_cr_l = 0.0;
        for (size_t p = 0; p < 64u * 64u; p++)
        {
            sum_cr_r += (double) im_cr_r[p];
            sum_cr_l += (double) im_cr_l[p];
        }
        int disp_cleared = (cor_b_idx >= 0 && ctx_ret_c.beams[cor_b_idx].disp.active == 0 &&
                            lo_b_idx >= 0 && ctx_ret_c.beams[lo_b_idx].disp.active == 0);
        snprintf(det, sizeof(det), "disp_cleared=%d, H_peak LUT/Ret=%.3f, Coro_sum Ret/LUT=%.2fx",
                 disp_cleared, (pk_h_r > 0.0f) ? (pk_h_l / pk_h_r) : 0.0f,
                 (sum_cr_l > 0.0) ? (sum_cr_r / sum_cr_l) : 0.0);
        check_cond(disp_cleared && pk_h_l > pk_h_r * 1.05f && sum_cr_r > sum_cr_l * 1.015,
                   "Coro clears downstream disp & ADC restores H PSF", det);

        /* Steady-state disp.gen invariance across consecutive ticks with static (z, PA) */
        uint64_t gen_b0_1  = ctx_lut_c.beams[0].disp.gen;
        uint64_t gen_adc_1 = ctx_lut_c.beams[1].disp.gen;
        cacaosim_bench_step(&ctx_lut_c, wf3, NULL, 1.0f, NULL);
        uint64_t gen_b0_2  = ctx_lut_c.beams[0].disp.gen;
        uint64_t gen_adc_2 = ctx_lut_c.beams[1].disp.gen;
        snprintf(det, sizeof(det), "b0 gen %llu->%llu, adc gen %llu->%llu",
                 (unsigned long long) gen_b0_1, (unsigned long long) gen_b0_2,
                 (unsigned long long) gen_adc_1, (unsigned long long) gen_adc_2);
        check_cond(gen_b0_1 == gen_b0_2 && gen_adc_1 == gen_adc_2 && gen_b0_1 > 0ULL,
                   "Steady-state disp.gen stays invariant in hot loop", det);

        /* CPU vs CUDA parity across ALL 5 engines (fpcam, pyrwfs, shwfs, zwfs, coro+lowfs) */
        if (cacaosim_cuda_available())
        {
            cacaosim_bench_cfg_t cfg_gpu = cfg_lut;
            cfg_gpu.backend              = CACAOSIM_BACKEND_CUDA;
            cacaosim_bench_ctx_t ctx_lut_g;
            if (cacaosim_bench_init(&ctx_lut_g, &cfg_gpu, NULL, 0, err, sizeof(err)) == 0)
            {
                /* Sync both CPU and GPU to fresh tick with 3-slice chromatic wf3 */
                cacaosim_bench_step(&ctx_lut_c, wf3, NULL, 1.0f, NULL);
                cacaosim_bench_step(&ctx_lut_g, wf3, NULL, 1.0f, NULL);

                const char *snames[6] = { "cam_vis", "cam_h", "pyr", "sh", "zw", "cam_cor" };
                double      errs[6]   = { 0 };
                double      max_e     = 0.0;
                for (int k = 0; k < 6; k++)
                {
                    int    sc  = cacaosim_bench_find_sensor(&ctx_lut_c, snames[k]);
                    int    sg  = cacaosim_bench_find_sensor(&ctx_lut_g, snames[k]);
                    size_t npx = (strcmp(snames[k], "sh") == 0) ? (32u * 32u) : (64u * 64u);
                    errs[k]    = rel_l2_f32(cacaosim_bench_sensor_f32(&ctx_lut_c, sc),
                                            cacaosim_bench_sensor_f32(&ctx_lut_g, sg), npx);
                    if (errs[k] > max_e)
                    {
                        max_e = errs[k];
                    }
                }
                int    s_zl_c = cacaosim_bench_find_sensor(&ctx_lut_c, "zw_lo");
                int    s_zl_g = cacaosim_bench_find_sensor(&ctx_lut_g, "zw_lo");
                double e_zlo  = rel_l2_f32(cacaosim_bench_sensor_f32(&ctx_lut_c, s_zl_c),
                                           cacaosim_bench_sensor_f32(&ctx_lut_g, s_zl_g),
                                           64u * 64u);
                if (e_zlo > max_e)
                {
                    max_e = e_zlo;
                }
                snprintf(det, sizeof(det),
                         "vis=%.1e H=%.1e pyr=%.1e sh=%.1e zw=%.1e cor=%.1e lo=%.1e", errs[0],
                         errs[1], errs[2], errs[3], errs[4], errs[5], e_zlo);
                check_cond(max_e < 5e-5, "5-engine chrom+disp+3wl OPD CPU vs CUDA parity", det);
                cacaosim_bench_free(&ctx_lut_g);
            }
        }
    }
    if (i1 == 0)
    {
        cacaosim_bench_free(&ctx_ret_c);
    }
    if (i2 == 0)
    {
        cacaosim_bench_free(&ctx_lut_c);
    }

    /* 20f. .bench format -> re-parse round-trip preserving atmospheric & ADC fields */
    char                 rt_buf[8192];
    int                  rt_len = cacaosim_bench_format_str(&cfg_lut, rt_buf, sizeof(rt_buf));
    cacaosim_bench_cfg_t cfg_rt;
    int rt_ok = (rt_len > 0 && cacaosim_bench_parse_str(rt_buf, &cfg_rt, err, sizeof(err)) == 0);
    int fields_ok =
        (rt_ok && fabsf(cfg_rt.atm_disp.zenith_deg - 45.0f) < 1e-4f &&
         fabsf(cfg_rt.atm_disp.disp_pa_deg - 25.0f) < 1e-4f && cfg_rt.chrom_turb == 1 &&
         cfg_rt.n_nodes == cfg_lut.n_nodes && cfg_rt.nodes[0].kind == CACAOSIM_NODE_ADC &&
         cfg_rt.nodes[0].adc.preset == CACAOSIM_ADC_PRESET_AO188 &&
         cfg_rt.nodes[0].adc.mode == CACAOSIM_ADC_MODE_LUT &&
         cfg_rt.nodes[0].adc.opt_band == CACAOSIM_ADC_OPT_H);
    snprintf(det, sizeof(det), "len=%d, z=%.1f, pa=%.1f, chrom=%d, adc_mode=%s", rt_len,
             rt_ok ? cfg_rt.atm_disp.zenith_deg : -1.0f,
             rt_ok ? cfg_rt.atm_disp.disp_pa_deg : -1.0f, rt_ok ? cfg_rt.chrom_turb : -1,
             rt_ok ? cacaosim_adc_mode_name(cfg_rt.nodes[0].adc.mode) : "ERR");
    check_cond(fields_ok, ".bench round-trip preserves atm_disp, chrom & ADC", det);

    free(wf_flat3);
    free(pup_tmp);
    free(wf3);
}

int main(void)
{
    printf("======================================================================\n");
    printf("  cacaosim Physical, Statistical & Closed-Loop Validation Suite\n");
    printf("  CUDA GPU available: %s\n", cacaosim_cuda_available() ? "YES" : "NO");
    printf("======================================================================\n");

    test_pupil_and_ncpa();
    test_detector_stats();
    test_dm_physics();
    test_pyrwfs_physics();
    test_fpcam_physics();
    test_closed_loop_ao();
    test_beam_photometry_and_ttm();
    test_shwfs_physics();
    test_zwfs_and_coro_physics();
    test_multistage_bench_dag();
    test_bench_configurator_and_tui();
    test_custom_coro_masks_and_poly_cube();
    test_coro_lowfs_wfe_and_scales();
    test_wfstat_cpu_gpu_parity();
    test_temporal_oversampling_physics();
    test_chromaticity_and_poly_wfs();
    test_subaru_lyot_coro_and_llowfs();
    test_options_schema_and_tui_coverage();
    test_entrance_perturbations();
    test_atm_disp_and_ao188_adc();

    printf("\n======================================================================\n");
    printf("  Summary: %d passed, %d failed\n", g_pass, g_fail);
    printf("======================================================================\n");
    return (g_fail == 0) ? 0 : 1;
}
