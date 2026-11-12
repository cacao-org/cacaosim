// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_validate.c
 * @brief   Physical, statistical, CPU/CUDA parity, and closed-loop validation suite for cacaosim
 */

#include <cblas.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milkcacaosim.h"

static int g_pass = 0;
static int g_fail = 0;

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

/* 1. Pupil geometry & NCPA RMS validation */
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

/* 2. Detector readout statistics (shutter darks, Poisson + RON variance) */
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

/* 3. DM hardware kinematics, delay, piston normalization, and CPU/CUDA parity */
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

    cacaosim_dm_free(&ctx);
    free(opd_cpu);
    free(cmd);
}

/* 4. Pyramid WFS symmetry, tip-tilt slope linearity, modulation gain, CPU/CUDA parity */
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

/* 5. Focal plane camera MFT vs FFT, Strehl=1, Marechal law, broadband, CPU/CUDA parity */
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

/* 6. End-to-end closed-loop AO calibration + control test (DM + PyWFS + FPcam) */
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

/* 7. Spectral photometry, beamsplitters/dichroics, 2-axis TTM & 2D registration */
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

/* 8. Diffractive Shack-Hartmann WFS symmetry, slope linearity & CPU/CUDA parity */
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

/* 9. Zernike phase-contrast WFS & Coronagraph (Lyot spot + Vortex + LOWFS pickoff) */
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

/* 10. Multi-stage .bench DAG validation, optical causality, multi-rate & 2-stage cascaded AO */
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

    printf("\n======================================================================\n");
    printf("  Summary: %d passed, %d failed\n", g_pass, g_fail);
    printf("======================================================================\n");
    return (g_fail == 0) ? 0 : 1;
}
