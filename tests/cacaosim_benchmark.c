// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_benchmark.c
 * @brief   Latency (mean / p50 / p99 us) and throughput (kHz) benchmark suite for cacaosim
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "milkcacaosim.h"

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *) a;
    double db = *(const double *) b;
    return (da > db) - (da < db);
}

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
    cacaosim_pyrwfs_init(&wfs, &wcfg, pup, NULL, dm.cuda_stream);
    cacaosim_fpcam_init(&fp, &fcfg, pup, NULL, dm.cuda_stream);

    double *lat = (double *) calloc((size_t) niter, sizeof(double));
    for (int k = -20; k < niter; k++)
    {
        uint64_t t0 = cacaosim_now_ns();
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

int main(void)
{
    printf("==================================================================================\n");
    printf("  cacaosim Runtime Latency & Throughput Benchmark\n");
    printf("==================================================================================\n");
    printf("  %-38s | %-4s | %8s | %8s | %8s | %s\n", "Compute Pipeline", "Back", "Mean us",
           "p50 us", "p99 us", "Throughput");
    printf("----------------------------------------------------------------------------------\n");

    bench_dm(20, 128, CACAOSIM_BACKEND_CPU, 500);
    bench_dm(20, 128, CACAOSIM_BACKEND_CUDA, 500);
    bench_dm(50, 256, CACAOSIM_BACKEND_CPU, 300);
    bench_dm(50, 256, CACAOSIM_BACKEND_CUDA, 500);

    bench_pyrwfs(128, 64, 0.0f, 1, CACAOSIM_BACKEND_CPU, 300);
    bench_pyrwfs(128, 64, 0.0f, 1, CACAOSIM_BACKEND_CUDA, 500);
    bench_pyrwfs(128, 64, 3.0f, 16, CACAOSIM_BACKEND_CPU, 150);
    bench_pyrwfs(128, 64, 3.0f, 16, CACAOSIM_BACKEND_CUDA, 500);
    bench_pyrwfs(256, 128, 3.0f, 16, CACAOSIM_BACKEND_CPU, 50);
    bench_pyrwfs(256, 128, 3.0f, 16, CACAOSIM_BACKEND_CUDA, 300);

    bench_shwfs(128, 16, 8, CACAOSIM_BACKEND_CPU, 200);
    bench_shwfs(128, 16, 8, CACAOSIM_BACKEND_CUDA, 500);

    bench_zwfs(128, 64, CACAOSIM_BACKEND_CPU, 300);
    bench_zwfs(128, 64, CACAOSIM_BACKEND_CUDA, 500);

    bench_coro(128, CACAOSIM_CORO_LYOT_SPOT, CACAOSIM_BACKEND_CPU, 200);
    bench_coro(128, CACAOSIM_CORO_LYOT_SPOT, CACAOSIM_BACKEND_CUDA, 500);
    bench_coro(128, CACAOSIM_CORO_VORTEX, CACAOSIM_BACKEND_CPU, 100);
    bench_coro(128, CACAOSIM_CORO_VORTEX, CACAOSIM_BACKEND_CUDA, 500);

    bench_fpcam(128, 128, 0, 1, CACAOSIM_BACKEND_CPU, 300);
    bench_fpcam(128, 128, 0, 1, CACAOSIM_BACKEND_CUDA, 500);
    bench_fpcam(128, 128, 0, 5, CACAOSIM_BACKEND_CPU, 150);
    bench_fpcam(128, 128, 0, 5, CACAOSIM_BACKEND_CUDA, 500);
    bench_fpcam(128, 128, 1, 1, CACAOSIM_BACKEND_CPU, 300);
    bench_fpcam(128, 128, 1, 1, CACAOSIM_BACKEND_CUDA, 500);

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

    printf("==================================================================================\n");
    return 0;
}
