// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm_poke_sim.c
 * @brief   20 kHz ALPAO 64x64 DM single-actuator poke simulator (DM OPD + NIR Pyramid WFS)
 *
 * Architectural Role:
 *   Standalone high-speed diagnostic runner exercising the compiled `cacaosim` optical bench
 *   engine (`cacaosim_bench_init` + `cacaosim_bench_step`) with a 64x64 ALPAO plate-dynamics DM
 *   (`dyn_mode = "plate"`) feeding a 160x160 4-sided Near-IR Pyramid WFS at 20 kHz (`dt = 50 us`).
 *   Dumps raw float32 cubes (`dm_opd_20khz.f32`, `pyrwfs_20khz.f32`, `pup.f32`) plus `meta.json`
 *   for rendering side-by-side comparison videos against the SCExAO measurement (`LepICSaBL-E`).
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "milkcacaosim.h"
#include "common/cacaosim_cuda.h"

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --outdir <dir>       Output directory (required)\n"
            "  --poke-x <int>       Poked actuator X [0..63] (default: 19)\n"
            "  --poke-y <int>       Poked actuator Y [0..63] (default: 42)\n"
            "  --poke-um <float>    Actuator command [um] (default: 6.0 -> ~0.80 um peak surf)\n"
            "  --rate-hz <float>    Sampling rate [Hz] (default: 20000)\n"
            "  --duration-ms <f>    Post-baseline video duration [ms] (default: 20.0)\n"
            "  --pre-ms <float>     Pre-poke baseline [ms] (default: 0.8)\n"
            "  --f0-hz <float>      Foundation resonance f0 [Hz] (default: 1000)\n"
            "  --wave-c <float>     Membrane wave speed [m/s] (default: 14.0)\n"
            "  --bend-beta <float>  Flexural dispersion beta [m^2/s] (default: 0.0085)\n"
            "  --damp-g0 <float>    Viscous decay rate gamma_0 [1/s] (default: 195)\n"
            "  --damp-g2 <float>    Kelvin-Voigt damping gamma_2 (default: 1.5)\n"
            "  --tau-us <float>     Coil RC pre-filter [us] (default: 20)\n"
            "  --drive <disp|force> Plate drive mode (default: force)\n"
            "  --rim <clamped|free> Circular rim BC (default: free)\n"
            "  --mod-rad <float>    NIR PyWFS modulation radius [l/D] (default: 2.0)\n"
            "  --subaru-pupil       Use Subaru Nasmyth-IR obscuration & spiders (default: disk)\n"
            "  --photon-noise       Enable photon shot + 0.8e- read noise (default: off)\n"
            "  --backend <auto|cpu|cuda> Compute backend (default: auto)\n",
            prog);
}

static int ensure_dir(const char *path)
{
    if (mkdir(path, 0775) == 0 || errno == EEXIST)
    {
        return 0;
    }
    return -1;
}

int main(int argc, char **argv)
{
    const char *outdir       = NULL;
    uint32_t    poke_x       = 19u;
    uint32_t    poke_y       = 42u;
    float       poke_um      = 6.0f;
    float       rate_hz      = 20000.0f;
    float       dur_ms       = 20.0f;
    float       pre_ms       = 0.8f;
    float       f0_hz        = 1000.0f;
    float       wave_c       = 14.0f;
    float       bend_beta    = 0.0085f;
    float       damp_g0      = 195.0f;
    float       damp_g2      = 1.5f;
    float       tau_us       = 20.0f;
    const char *drive_str    = "force";
    const char *rim_str      = "free";
    float       mod_rad      = 2.0f;
    int         subaru_pupil = 0;
    int         photon_noise = 0;
    const char *backend_str  = "auto";

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--outdir") == 0 && i + 1 < argc)
        {
            outdir = argv[++i];
        }
        else if (strcmp(argv[i], "--poke-x") == 0 && i + 1 < argc)
        {
            poke_x = (uint32_t) atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--poke-y") == 0 && i + 1 < argc)
        {
            poke_y = (uint32_t) atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--poke-um") == 0 && i + 1 < argc)
        {
            poke_um = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--rate-hz") == 0 && i + 1 < argc)
        {
            rate_hz = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--duration-ms") == 0 && i + 1 < argc)
        {
            dur_ms = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--pre-ms") == 0 && i + 1 < argc)
        {
            pre_ms = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--f0-hz") == 0 && i + 1 < argc)
        {
            f0_hz = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--wave-c") == 0 && i + 1 < argc)
        {
            wave_c = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--bend-beta") == 0 && i + 1 < argc)
        {
            bend_beta = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--damp-g0") == 0 && i + 1 < argc)
        {
            damp_g0 = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--damp-g2") == 0 && i + 1 < argc)
        {
            damp_g2 = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--tau-us") == 0 && i + 1 < argc)
        {
            tau_us = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--drive") == 0 && i + 1 < argc)
        {
            drive_str = argv[++i];
        }
        else if (strcmp(argv[i], "--rim") == 0 && i + 1 < argc)
        {
            rim_str = argv[++i];
        }
        else if (strcmp(argv[i], "--mod-rad") == 0 && i + 1 < argc)
        {
            mod_rad = strtof(argv[++i], NULL);
        }
        else if (strcmp(argv[i], "--subaru-pupil") == 0)
        {
            subaru_pupil = 1;
        }
        else if (strcmp(argv[i], "--photon-noise") == 0)
        {
            photon_noise = 1;
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc)
        {
            backend_str = argv[++i];
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (outdir == NULL || outdir[0] == '\0')
    {
        usage(argv[0]);
        return 2;
    }
    if (ensure_dir(outdir) != 0)
    {
        perror("mkdir");
        return 1;
    }

    const uint32_t nwf      = 256u;
    const uint32_t ncam     = 160u;
    const uint32_t nact     = 64u;
    const float    cobs     = subaru_pupil ? 0.307f : 0.0f;
    const int      nsp      = subaru_pupil ? 4 : 0;
    const float    sp_w     = subaru_pupil ? 0.02314f : 0.0f;
    const float    sp_deg   = subaru_pupil ? 51.75f : 0.0f;
    const float    sp_off   = subaru_pupil ? 0.08289f : 0.0f;

    char ini[4096];
    snprintf(ini, sizeof(ini),
             "[bench]\n"
             "name          = \"dm64_poke_20khz\"\n"
             "nwf           = %u\n"
             "pup_diam      = 0.96\n"
             "cobs          = %.5f\n"
             "nspider       = %d\n"
             "spider_w      = %.5f\n"
             "spider_deg    = %.5f\n"
             "spider_offset = %.5f\n"
             "tel_diam_m    = 7.95\n"
             "star_mag      = 0.0\n"
             "master_hz     = %.2f\n"
             "oversample    = 1\n"
             "out_beam      = \"b_pup\"\n"
             "turb_gain     = 0.0\n"
             "backend       = \"%s\"\n"
             "\n"
             "[dm.ao3k]\n"
             "in            = \"b_pup\"\n"
             "out           = \"b_post_dm\"\n"
             "in_cmd        = \"dm01disp\"\n"
             "nact_x        = %u\n"
             "nact_y        = %u\n"
             "dm_diam       = 0.984375\n"
             "if_coupling   = %.3f\n"
             "tput          = 0.98\n"
             "max_stroke    = 12.0\n"
             "dac_bits      = 16\n"
             "dyn_mode      = \"plate\"\n"
             "tau_us        = %.4f\n"
             "f0_hz         = %.4f\n"
             "act_pitch_mm  = 1.5\n"
             "wave_c        = %.4f\n"
             "bend_beta     = %.6f\n"
             "damp_g0       = %.4f\n"
             "damp_g2       = %.4f\n"
             "plate_drive   = \"%s\"\n"
             "rim_bc        = \"%s\"\n"
             "delay_frames  = 0.0\n"
             "\n"
             "[pyrwfs.wfs_nir]\n"
             "in            = \"b_post_dm\"\n"
             "out_cam       = \"csim_wfs_nir\"\n"
             "ncam          = %u\n"
             "lambda_um     = 1.65\n"
             "bw_frac       = 0.0\n"
             "use_beam_flux = 0\n"
             "flux_ph       = 1.0e8\n"
             "navg          = 1\n"
             "mod_rad       = %.4f\n"
             "mod_npts      = 16\n"
             "mod_mode      = \"full\"\n"
             "pup_sep       = 0.48\n"
             "photon_noise  = %d\n"
             "ron_e         = %.2f\n"
             "bias_adu      = 0.0\n"
             "sat_adu       = 1.0e12\n"
             "out_dtype     = \"float32\"\n",
             nwf, cobs, nsp, sp_w, sp_deg, sp_off, rate_hz, backend_str, nact, nact,
             (strcmp(drive_str, "force") == 0) ? 0.08f : 0.20f, tau_us,
             f0_hz, wave_c, bend_beta, damp_g0, damp_g2, drive_str, rim_str, ncam, mod_rad,
             photon_noise, photon_noise ? 0.8f : 0.0f);

    cacaosim_bench_cfg_t cfg;
    cacaosim_bench_ctx_t ctx;
    char                 err[256] = "";
    if (cacaosim_bench_parse_str(ini, &cfg, err, sizeof(err)) != 0 ||
        cacaosim_bench_init(&ctx, &cfg, NULL, 0, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "cacaosim_bench_init failed: %s\n", err);
        return 1;
    }

    const int      dm_slot  = cacaosim_bench_find_dm(&ctx, "ao3k");
    const int      wfs_slot = cacaosim_bench_find_sensor(&ctx, "wfs_nir");
    const int32_t  dm_ni    = ctx.dm_node_idx[dm_slot];
    float         *cmd      = cacaosim_bench_dm_cmd_ptr(&ctx, dm_slot);
    const uint32_t n_diff   = (uint32_t) lroundf(rate_hz * 1.0e-3f); /* 1 ms exposure = 20 fr */
    const uint32_t n_pre    = (uint32_t) lroundf(rate_hz * pre_ms * 1.0e-3f);
    const uint32_t n_out    = (uint32_t) lroundf(rate_hz * dur_ms * 1.0e-3f);
    const uint32_t n_total  = 2u * n_diff + n_out; /* full 2*1ms lead-in for boxcar exposure diff */
    const uint32_t fr_poke  = 2u * n_diff + n_pre;

    char path_opd[512], path_wfs[512], path_pup[512], path_meta[512];
    snprintf(path_opd, sizeof(path_opd), "%s/dm_opd_20khz.f32", outdir);
    snprintf(path_wfs, sizeof(path_wfs), "%s/pyrwfs_20khz.f32", outdir);
    snprintf(path_pup, sizeof(path_pup), "%s/pup.f32", outdir);
    snprintf(path_meta, sizeof(path_meta), "%s/meta.json", outdir);

    FILE *f_opd = fopen(path_opd, "wb");
    FILE *f_wfs = fopen(path_wfs, "wb");
    FILE *f_pup = fopen(path_pup, "wb");
    if (!f_opd || !f_wfs || !f_pup)
    {
        perror("fopen");
        cacaosim_bench_free(&ctx);
        return 1;
    }

    fwrite(ctx.pup_ptr, sizeof(float), (size_t) nwf * nwf, f_pup);
    fclose(f_pup);

    float *opd_host = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    uint64_t t0_ns  = cacaosim_now_ns();

    for (uint32_t fr = 0; fr < n_total; fr++)
    {
        if (fr == fr_poke && poke_x < nact && poke_y < nact)
        {
            cmd[poke_y * nact + poke_x] = poke_um;
        }
        cacaosim_bench_step(&ctx, NULL, NULL, 0.0f, NULL);

        const float *wfs_im = cacaosim_bench_sensor_f32(&ctx, wfs_slot);
        fwrite(wfs_im, sizeof(float), (size_t) ncam * ncam, f_wfs);

        cacaosim_bench_node_t *dm_nd = &ctx.nodes[dm_ni];
        if (ctx.use_cuda)
        {
            cacaosim_cuda_d2h_f32(opd_host, dm_nd->dm_ctx.d_opd_out, (uint64_t) nwf * nwf,
                                  ctx.cuda_stream);
            cacaosim_cuda_stream_sync(ctx.cuda_stream);
            fwrite(opd_host, sizeof(float), (size_t) nwf * nwf, f_opd);
        }
        else
        {
            fwrite(dm_nd->dm_opd_buf, sizeof(float), (size_t) nwf * nwf, f_opd);
        }
    }

    double elapsed_s = (double) (cacaosim_now_ns() - t0_ns) * 1e-9;
    fclose(f_opd);
    fclose(f_wfs);

    FILE *f_meta = fopen(path_meta, "w");
    if (f_meta)
    {
        fprintf(f_meta,
                "{\n"
                "  \"n_total\": %u,\n"
                "  \"n_diff\": %u,\n"
                "  \"n_out\": %u,\n"
                "  \"fr_poke\": %u,\n"
                "  \"nwf\": %u,\n"
                "  \"ncam\": %u,\n"
                "  \"nact\": %u,\n"
                "  \"poke_x\": %u,\n"
                "  \"poke_y\": %u,\n"
                "  \"poke_um\": %.4f,\n"
                "  \"rate_hz\": %.2f,\n"
                "  \"pre_ms\": %.4f,\n"
                "  \"duration_ms\": %.4f,\n"
                "  \"f0_hz\": %.4f,\n"
                "  \"wave_c_ms\": %.4f,\n"
                "  \"bend_beta_m2s\": %.6f,\n"
                "  \"damp_g0\": %.4f,\n"
                "  \"damp_g2\": %.4f,\n"
                "  \"tau_us\": %.4f,\n"
                "  \"plate_drive\": \"%s\",\n"
                "  \"rim_bc\": \"%s\",\n"
                "  \"plate_nsub\": %u,\n"
                "  \"backend\": \"%s\",\n"
                "  \"plate_use_cuda\": %d,\n"
                "  \"wall_sec\": %.4f\n"
                "}\n",
                n_total, n_diff, n_out, fr_poke, nwf, ncam, nact, poke_x, poke_y, poke_um,
                rate_hz, pre_ms, dur_ms, f0_hz, wave_c, bend_beta, damp_g0, damp_g2, tau_us,
                drive_str, rim_str, ctx.nodes[dm_ni].dm_ctx.plate_nsub,
                ctx.use_cuda ? "cuda" : "cpu", ctx.nodes[dm_ni].dm_ctx.plate_use_cuda, elapsed_s);
        fclose(f_meta);
    }

    printf("[cacaosim-dm-poke-sim] %u frames @ %.0f Hz (%s, plate_cuda=%d, nsub=%u) in %.3f s -> "
           "%s\n",
           n_total, rate_hz, ctx.use_cuda ? "CUDA" : "CPU", ctx.nodes[dm_ni].dm_ctx.plate_use_cuda,
           ctx.nodes[dm_ni].dm_ctx.plate_nsub, elapsed_s, outdir);

    free(opd_host);
    cacaosim_bench_free(&ctx);
    return 0;
}
