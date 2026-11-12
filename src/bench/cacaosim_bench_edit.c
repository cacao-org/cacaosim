// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_edit.c
 * @brief   Optical bench (.bench) presets and topological node editing/splicing helpers
 */

#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "bench/cacaosim_bench.h"

static void edit_copy_str(char *dst, size_t cap, const char *src)
{
    if (cap == 0u)
    {
        return;
    }
    if (src == NULL)
    {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, cap, "%s", src);
}

static void edit_set_err(char *err_buf, size_t err_cap, const char *msg)
{
    if (err_buf != NULL && err_cap > 0u)
    {
        snprintf(err_buf, err_cap, "%s", (msg != NULL) ? msg : "Error");
    }
}

static int node_id_exists(const cacaosim_bench_cfg_t *cfg, const char *id, int skip_idx)
{
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        if ((int) i == skip_idx)
        {
            continue;
        }
        if (strcmp(cfg->nodes[i].id, id) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static int beam_name_exists(const cacaosim_bench_cfg_t *cfg, const char *beam)
{
    if (beam == NULL || beam[0] == '\0')
    {
        return 0;
    }
    if (strcmp(cfg->out_beam, beam) == 0)
    {
        return 1;
    }
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *nd = &cfg->nodes[i];
        if (strcmp(nd->in_beam, beam) == 0 || strcmp(nd->out_beam, beam) == 0 ||
            strcmp(nd->out_refl, beam) == 0 || strcmp(nd->out_trans, beam) == 0 ||
            strcmp(nd->out_lowfs, beam) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static int shm_name_exists(const cacaosim_bench_cfg_t *cfg, const char *shm)
{
    if (shm == NULL || shm[0] == '\0')
    {
        return 0;
    }
    if (strcmp(cfg->in_wf, shm) == 0 || strcmp(cfg->in_pup, shm) == 0 ||
        strcmp(cfg->in_amp, shm) == 0 || strcmp(cfg->in_ncpa, shm) == 0)
    {
        return 1;
    }
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *nd = &cfg->nodes[i];
        if (strcmp(nd->in_cmd, shm) == 0 || strcmp(nd->out_opd, shm) == 0 ||
            strcmp(nd->out_cam, shm) == 0 || strcmp(nd->out_long, shm) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static void make_unique_id(
    const cacaosim_bench_cfg_t *cfg,
    const char                 *prefix,
    char                       *out,
    size_t                      cap)
{
    for (uint32_t n = 0; n < 256u; n++)
    {
        snprintf(out, cap, "%s%u", prefix, n);
        if (!node_id_exists(cfg, out, -1))
        {
            return;
        }
    }
    snprintf(out, cap, "%s_x", prefix);
}

static void make_unique_beam(
    const cacaosim_bench_cfg_t *cfg,
    const char                 *base,
    char                       *out,
    size_t                      cap)
{
    snprintf(out, cap, "%s", base);
    if (!beam_name_exists(cfg, out))
    {
        return;
    }
    for (uint32_t n = 1; n < 256u; n++)
    {
        snprintf(out, cap, "%.48s_%u", base, n);
        if (!beam_name_exists(cfg, out))
        {
            return;
        }
    }
}

static void make_unique_shm(
    const cacaosim_bench_cfg_t *cfg,
    const char                 *base,
    char                       *out,
    size_t                      cap)
{
    snprintf(out, cap, "%s", base);
    if (!shm_name_exists(cfg, out))
    {
        return;
    }
    for (uint32_t n = 1; n < 256u; n++)
    {
        snprintf(out, cap, "%.48s_%u", base, n);
        if (!shm_name_exists(cfg, out))
        {
            return;
        }
    }
}

static uint32_t count_beam_consumers(
    const cacaosim_bench_cfg_t *cfg,
    const char                 *beam,
    int                         skip_idx)
{
    if (beam == NULL || beam[0] == '\0')
    {
        return 0u;
    }
    uint32_t cnt = 0u;
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        if ((int) i == skip_idx)
        {
            continue;
        }
        if (strcmp(cfg->nodes[i].in_beam, beam) == 0)
        {
            cnt++;
        }
    }
    return cnt;
}

static void rewire_consumers_except(
    cacaosim_bench_cfg_t *cfg,
    const char           *old_beam,
    const char           *new_beam,
    int                   skip_a,
    int                   skip_b)
{
    if (old_beam == NULL || old_beam[0] == '\0' || new_beam == NULL || new_beam[0] == '\0')
    {
        return;
    }
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        if ((int) i == skip_a || (int) i == skip_b)
        {
            continue;
        }
        if (strcmp(cfg->nodes[i].in_beam, old_beam) == 0)
        {
            edit_copy_str(cfg->nodes[i].in_beam, sizeof(cfg->nodes[i].in_beam), new_beam);
        }
    }
}

static int is_wfs_kind(cacaosim_node_kind_t k)
{
    return (k == CACAOSIM_NODE_PYRWFS || k == CACAOSIM_NODE_SHWFS || k == CACAOSIM_NODE_ZWFS);
}

static int is_inline_single_kind(cacaosim_node_kind_t k)
{
    return (k == CACAOSIM_NODE_DM || k == CACAOSIM_NODE_CORO);
}

/* Suggest a sensible sensor wavelength based on upstream dichroic arm if applicable */
static void auto_tune_sensor_wavelength(
    const cacaosim_bench_cfg_t *cfg,
    cacaosim_node_cfg_t        *sensor)
{
    const char *b = sensor->in_beam;
    if (b == NULL || b[0] == '\0')
    {
        return;
    }

    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *up = &cfg->nodes[i];
        if (up->kind == CACAOSIM_NODE_SPLITTER && up->split.mode != CACAOSIM_SPLIT_GRAY)
        {
            int is_refl  = (strcmp(up->out_refl, b) == 0);
            int is_trans = (strcmp(up->out_trans, b) == 0);
            if (!is_refl && !is_trans)
            {
                continue;
            }

            float lcut = up->split.lambda_cut;
            float lam  = 0.75f;
            if (up->split.mode == CACAOSIM_SPLIT_DICHROIC_BP)
            {
                lam = is_refl ? lcut : (lcut < 1.3f ? 1.65f : 0.75f);
            }
            else if (up->split.mode == CACAOSIM_SPLIT_DICHROIC_LP)
            {
                lam = is_refl ? (lcut > 0.65f ? 0.75f : 0.55f)
                              : (lcut < 1.4f ? 1.65f : lcut + 0.3f);
                if (is_refl && lam >= lcut)
                {
                    lam = (lcut > 0.50f) ? (lcut - 0.15f) : 0.45f;
                }
            }
            else if (up->split.mode == CACAOSIM_SPLIT_DICHROIC_SP)
            {
                lam = is_trans ? (lcut > 0.65f ? 0.75f : 0.55f)
                               : (lcut < 1.4f ? 1.65f : lcut + 0.3f);
            }

            if (sensor->kind == CACAOSIM_NODE_PYRWFS)
            {
                sensor->pyrwfs.lambda_um = lam;
            }
            else if (sensor->kind == CACAOSIM_NODE_SHWFS)
            {
                sensor->shwfs.lambda_um = lam;
            }
            else if (sensor->kind == CACAOSIM_NODE_ZWFS)
            {
                sensor->zwfs.lambda_um = lam;
            }
            else if (sensor->kind == CACAOSIM_NODE_FPCAM)
            {
                sensor->fpcam.lambda_um = lam;
            }
            return;
        }
        if (up->kind == CACAOSIM_NODE_CORO &&
            (strcmp(up->out_beam, b) == 0 || strcmp(up->out_lowfs, b) == 0))
        {
            float lam = up->coro.lambda_um;
            if (sensor->kind == CACAOSIM_NODE_ZWFS)
            {
                sensor->zwfs.lambda_um = lam;
            }
            else if (sensor->kind == CACAOSIM_NODE_FPCAM)
            {
                sensor->fpcam.lambda_um = lam;
            }
            return;
        }
    }
}

const char *cacaosim_bench_preset_name(int preset_id)
{
    switch (preset_id)
    {
    case CACAOSIM_PRESET_SINGLE_PYRWFS:
        return "Single-Stage PyWFS AO";
    case CACAOSIM_PRESET_SINGLE_SHWFS:
        return "Single-Stage TTM + DM + SHWFS";
    case CACAOSIM_PRESET_TWO_STAGE:
        return "Two-Stage Cascaded AO (Woofer + Tweeter)";
    case CACAOSIM_PRESET_EXAO_CORO:
        return "ExAO Coronagraph + Zernike LOWFS";
    case CACAOSIM_PRESET_TRUTH_DUAL_WFS:
        return "Dual-WFS (Fast PyWFS + Truth SHWFS)";
    case CACAOSIM_PRESET_MINIMAL:
        return "Minimal Telescope + Science Camera";
    default:
        return "Unknown Preset";
    }
}

const char *cacaosim_bench_preset_desc(int preset_id)
{
    switch (preset_id)
    {
    case CACAOSIM_PRESET_SINGLE_PYRWFS:
        return "20x20 DM -> 0.95 um LP Dichroic -> Vis PyWFS (2 kHz) + H-band Sci Cam";
    case CACAOSIM_PRESET_SINGLE_SHWFS:
        return "TTM + 16x16 DM -> 1.00 um LP Dichroic -> 16x16 SHWFS + H-band Sci Cam";
    case CACAOSIM_PRESET_TWO_STAGE:
        return "12x12 Woofer -> J-band PyWFS -> 24x24 Tweeter -> Vis PyWFS + H-band Cam";
    case CACAOSIM_PRESET_EXAO_CORO:
        return "TTM + 24x24 DM -> Vis SHWFS + H-band Lyot Coronagraph + ZWFS LOWFS + Cam";
    case CACAOSIM_PRESET_TRUTH_DUAL_WFS:
        return "20x20 DM -> Dichroic -> 80/20 Gray BS -> 2 kHz PyWFS + 100 Hz Truth SHWFS";
    case CACAOSIM_PRESET_MINIMAL:
        return "8.2 m Entrance Pupil -> Direct H-band Focal-Plane Science Camera";
    default:
        return "";
    }
}

static const char *const g_preset_ini[CACAOSIM_PRESET_COUNT] = {
    /* CACAOSIM_PRESET_SINGLE_PYRWFS */
    "[bench]\n"
    "name       = \"single_stage_pyrwfs\"\n"
    "nwf        = 128\n"
    "pup_diam   = 0.9375\n"
    "cobs       = 0.30\n"
    "nspider    = 4\n"
    "spider_w   = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag   = 8.0\n"
    "master_hz  = 2000.0\n"
    "in_wf      = \"outarraypha\"\n"
    "in_pup     = \"csim_pup\"\n"
    "out_beam   = \"b_pup\"\n"
    "[dm.dm0]\n"
    "in         = \"b_pup\"\n"
    "out        = \"b_dm0\"\n"
    "in_cmd     = \"aol0_dmC\"\n"
    "out_opd    = \"csim_dm_opd\"\n"
    "nact_x     = 20\n"
    "nact_y     = 20\n"
    "max_stroke = 5.0\n"
    "tau_us     = 120.0\n"
    "[splitter.dichroic]\n"
    "in         = \"b_dm0\"\n"
    "out_refl   = \"b_wfs\"\n"
    "out_trans  = \"b_sci\"\n"
    "mode       = \"dichroic_lp\"\n"
    "lambda_cut = 0.95\n"
    "[pyrwfs.wfs0]\n"
    "in         = \"b_wfs\"\n"
    "out_cam    = \"aol0_wfsim\"\n"
    "ncam       = 64\n"
    "lambda_um  = 0.75\n"
    "mod_rad    = 3.0\n"
    "navg       = 1\n"
    "[fpcam.sci]\n"
    "in         = \"b_sci\"\n"
    "out_cam    = \"csim_fpcam\"\n"
    "out_long   = \"csim_fplong\"\n"
    "ncam       = 128\n"
    "lambda_um  = 1.65\n"
    "navg       = 10\n",

    /* CACAOSIM_PRESET_SINGLE_SHWFS */
    "[bench]\n"
    "name       = \"single_stage_shwfs\"\n"
    "nwf        = 128\n"
    "pup_diam   = 0.9375\n"
    "cobs       = 0.30\n"
    "nspider    = 4\n"
    "spider_w   = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag   = 8.0\n"
    "master_hz  = 2000.0\n"
    "in_wf      = \"outarraypha\"\n"
    "in_pup     = \"csim_pup\"\n"
    "out_beam   = \"b_pup\"\n"
    "[ttm.ttm0]\n"
    "in         = \"b_pup\"\n"
    "out        = \"b_ttm\"\n"
    "in_cmd     = \"ttm0_cmd\"\n"
    "max_stroke = 10.0\n"
    "tau_us     = 400.0\n"
    "[dm.dm0]\n"
    "in         = \"b_ttm\"\n"
    "out        = \"b_dm0\"\n"
    "in_cmd     = \"aol0_dmC\"\n"
    "nact_x     = 16\n"
    "nact_y     = 16\n"
    "max_stroke = 5.0\n"
    "tau_us     = 120.0\n"
    "[splitter.dichroic]\n"
    "in         = \"b_dm0\"\n"
    "out_refl   = \"b_wfs\"\n"
    "out_trans  = \"b_sci\"\n"
    "mode       = \"dichroic_lp\"\n"
    "lambda_cut = 1.00\n"
    "[shwfs.sh0]\n"
    "in          = \"b_wfs\"\n"
    "out_cam     = \"aol0_wfsim\"\n"
    "n_sub       = 16\n"
    "pix_per_sub = 8\n"
    "lambda_um   = 0.75\n"
    "navg        = 1\n"
    "[fpcam.sci]\n"
    "in         = \"b_sci\"\n"
    "out_cam    = \"csim_fpcam\"\n"
    "out_long   = \"csim_fplong\"\n"
    "ncam       = 128\n"
    "lambda_um  = 1.65\n"
    "navg       = 10\n",

    /* CACAOSIM_PRESET_TWO_STAGE */
    "[bench]\n"
    "name       = \"two_stage_cascaded_ao\"\n"
    "nwf        = 128\n"
    "pup_diam   = 0.9375\n"
    "cobs       = 0.30\n"
    "nspider    = 4\n"
    "spider_w   = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag   = 8.0\n"
    "master_hz  = 2000.0\n"
    "in_wf      = \"outarraypha\"\n"
    "in_pup     = \"csim_pup\"\n"
    "out_beam   = \"b_pup\"\n"
    "[dm.woofer]\n"
    "in         = \"b_pup\"\n"
    "out        = \"b_post_dm1\"\n"
    "in_cmd     = \"dm01disp\"\n"
    "nact_x     = 12\n"
    "nact_y     = 12\n"
    "max_stroke = 8.0\n"
    "tau_us     = 250.0\n"
    "[splitter.dichroic_stg1]\n"
    "in         = \"b_post_dm1\"\n"
    "out_refl   = \"b_nir_wfs\"\n"
    "out_trans  = \"b_stg2_in\"\n"
    "mode       = \"dichroic_bp\"\n"
    "lambda_cut = 1.22\n"
    "band_width = 0.30\n"
    "[pyrwfs.wfs_nir]\n"
    "in         = \"b_nir_wfs\"\n"
    "out_cam    = \"aol1_wfsim\"\n"
    "ncam       = 64\n"
    "lambda_um  = 1.22\n"
    "navg       = 2\n"
    "mod_rad    = 3.0\n"
    "[dm.tweeter]\n"
    "in         = \"b_stg2_in\"\n"
    "out        = \"b_post_dm2\"\n"
    "in_cmd     = \"dm02disp\"\n"
    "nact_x     = 24\n"
    "nact_y     = 24\n"
    "max_stroke = 3.0\n"
    "tau_us     = 80.0\n"
    "[splitter.dichroic_stg2]\n"
    "in         = \"b_post_dm2\"\n"
    "out_refl   = \"b_vis_wfs\"\n"
    "out_trans  = \"b_sci\"\n"
    "mode       = \"dichroic_lp\"\n"
    "lambda_cut = 1.00\n"
    "[pyrwfs.wfs_vis]\n"
    "in         = \"b_vis_wfs\"\n"
    "out_cam    = \"aol2_wfsim\"\n"
    "ncam       = 64\n"
    "lambda_um  = 0.75\n"
    "navg       = 1\n"
    "mod_rad    = 2.0\n"
    "[fpcam.sci_h]\n"
    "in         = \"b_sci\"\n"
    "out_cam    = \"csim_fpcam\"\n"
    "out_long   = \"csim_fplong\"\n"
    "ncam       = 128\n"
    "lambda_um  = 1.65\n"
    "navg       = 20\n",

    /* CACAOSIM_PRESET_EXAO_CORO */
    "[bench]\n"
    "name       = \"exao_coro_lowfs\"\n"
    "nwf        = 128\n"
    "pup_diam   = 0.9375\n"
    "cobs       = 0.30\n"
    "nspider    = 4\n"
    "spider_w   = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag   = 6.0\n"
    "master_hz  = 2000.0\n"
    "in_wf      = \"outarraypha\"\n"
    "in_pup     = \"csim_pup\"\n"
    "out_beam   = \"b_pup\"\n"
    "[ttm.ttm_fast]\n"
    "in         = \"b_pup\"\n"
    "out        = \"b_after_ttm\"\n"
    "in_cmd     = \"ttm_cmd\"\n"
    "max_stroke = 10.0\n"
    "tau_us     = 300.0\n"
    "[dm.tweeter]\n"
    "in         = \"b_after_ttm\"\n"
    "out        = \"b_after_dm\"\n"
    "in_cmd     = \"dm01disp\"\n"
    "nact_x     = 24\n"
    "nact_y     = 24\n"
    "max_stroke = 3.5\n"
    "tau_us     = 60.0\n"
    "[splitter.dichroic]\n"
    "in         = \"b_after_dm\"\n"
    "out_refl   = \"b_vis\"\n"
    "out_trans  = \"b_nir\"\n"
    "mode       = \"dichroic_lp\"\n"
    "lambda_cut = 0.95\n"
    "[shwfs.howfs]\n"
    "in          = \"b_vis\"\n"
    "out_cam     = \"aol0_wfsim\"\n"
    "n_sub       = 16\n"
    "pix_per_sub = 8\n"
    "lambda_um   = 0.75\n"
    "navg        = 1\n"
    "[coro.lyot_h]\n"
    "in          = \"b_nir\"\n"
    "out         = \"b_coro_sci\"\n"
    "out_lowfs   = \"b_lowfs\"\n"
    "mode        = \"lyot_spot\"\n"
    "lambda_um   = 1.65\n"
    "fpm_rad_ld  = 3.0\n"
    "lowfs_refl  = 0.95\n"
    "lyot_outer  = 0.85\n"
    "lyot_obs    = 0.36\n"
    "[zwfs.lowfs]\n"
    "in          = \"b_lowfs\"\n"
    "out_cam     = \"lowfs_im\"\n"
    "ncam        = 64\n"
    "lambda_um   = 1.65\n"
    "navg        = 4\n"
    "[fpcam.coro_cam]\n"
    "in          = \"b_coro_sci\"\n"
    "out_cam     = \"csim_fpcam\"\n"
    "out_long    = \"csim_fplong\"\n"
    "ncam        = 128\n"
    "lambda_um   = 1.65\n"
    "navg        = 20\n",

    /* CACAOSIM_PRESET_TRUTH_DUAL_WFS */
    "[bench]\n"
    "name       = \"dual_wfs_truth_ao\"\n"
    "nwf        = 128\n"
    "pup_diam   = 0.9375\n"
    "cobs       = 0.30\n"
    "nspider    = 4\n"
    "spider_w   = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag   = 7.5\n"
    "master_hz  = 2000.0\n"
    "in_wf      = \"outarraypha\"\n"
    "in_pup     = \"csim_pup\"\n"
    "out_beam   = \"b_pup\"\n"
    "[dm.dm0]\n"
    "in         = \"b_pup\"\n"
    "out        = \"b_dm0\"\n"
    "in_cmd     = \"aol0_dmC\"\n"
    "nact_x     = 20\n"
    "nact_y     = 20\n"
    "tau_us     = 100.0\n"
    "[splitter.dichroic]\n"
    "in         = \"b_dm0\"\n"
    "out_refl   = \"b_wfs_arm\"\n"
    "out_trans  = \"b_sci\"\n"
    "mode       = \"dichroic_lp\"\n"
    "lambda_cut = 0.95\n"
    "[splitter.wfs_bs]\n"
    "in         = \"b_wfs_arm\"\n"
    "out_refl   = \"b_truth\"\n"
    "out_trans  = \"b_fast\"\n"
    "mode       = \"gray\"\n"
    "r_gray     = 0.20\n"
    "t_gray     = 0.80\n"
    "[pyrwfs.fast_pyr]\n"
    "in         = \"b_fast\"\n"
    "out_cam    = \"aol0_wfsim\"\n"
    "ncam       = 64\n"
    "lambda_um  = 0.75\n"
    "navg       = 1\n"
    "[shwfs.truth_sh]\n"
    "in          = \"b_truth\"\n"
    "out_cam     = \"truth_wfsim\"\n"
    "n_sub       = 16\n"
    "pix_per_sub = 8\n"
    "lambda_um   = 0.75\n"
    "navg        = 20\n"
    "[fpcam.sci]\n"
    "in         = \"b_sci\"\n"
    "out_cam    = \"csim_fpcam\"\n"
    "out_long   = \"csim_fplong\"\n"
    "ncam       = 128\n"
    "lambda_um  = 1.65\n"
    "navg       = 10\n",

    /* CACAOSIM_PRESET_MINIMAL */
    "[bench]\n"
    "name       = \"minimal_bench\"\n"
    "nwf        = 128\n"
    "pup_diam   = 0.9375\n"
    "cobs       = 0.30\n"
    "nspider    = 4\n"
    "spider_w   = 0.012\n"
    "tel_diam_m = 8.2\n"
    "star_mag   = 8.0\n"
    "master_hz  = 1000.0\n"
    "in_wf      = \"outarraypha\"\n"
    "in_pup     = \"csim_pup\"\n"
    "out_beam   = \"b_pup\"\n"
    "[fpcam.sci]\n"
    "in         = \"b_pup\"\n"
    "out_cam    = \"csim_fpcam\"\n"
    "out_long   = \"csim_fplong\"\n"
    "ncam       = 128\n"
    "lambda_um  = 1.65\n"
    "navg       = 10\n",
};

int cacaosim_bench_preset(cacaosim_bench_cfg_t *cfg, int preset_id)
{
    if (cfg == NULL || preset_id < 0 || preset_id >= CACAOSIM_PRESET_COUNT)
    {
        return -1;
    }
    return cacaosim_bench_parse_str(g_preset_ini[preset_id], cfg, NULL, 0u);
}

static void push_unique_beam(
    char        beams[][CACAOSIM_BENCH_NAME_LEN],
    uint32_t   *n,
    uint32_t    cap,
    const char *name)
{
    if (name == NULL || name[0] == '\0' || *n >= cap)
    {
        return;
    }
    for (uint32_t i = 0; i < *n; i++)
    {
        if (strcmp(beams[i], name) == 0)
        {
            return;
        }
    }
    edit_copy_str(beams[*n], CACAOSIM_BENCH_NAME_LEN, name);
    (*n)++;
}

uint32_t cacaosim_bench_list_beams(
    const cacaosim_bench_cfg_t *cfg,
    char                        beams[][CACAOSIM_BENCH_NAME_LEN],
    uint32_t                    max_beams)
{
    if (cfg == NULL || beams == NULL || max_beams == 0u)
    {
        return 0u;
    }
    uint32_t n = 0u;
    push_unique_beam(beams, &n, max_beams, cfg->out_beam[0] ? cfg->out_beam : "b_pup");
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *nd = &cfg->nodes[i];
        push_unique_beam(beams, &n, max_beams, nd->out_beam);
        push_unique_beam(beams, &n, max_beams, nd->out_refl);
        push_unique_beam(beams, &n, max_beams, nd->out_trans);
        push_unique_beam(beams, &n, max_beams, nd->out_lowfs);
    }
    return n;
}

static void derive_auto_target_beam(
    const cacaosim_bench_cfg_t *cfg,
    cacaosim_node_kind_t        new_kind,
    int                         after_node_idx,
    char                       *out_beam,
    size_t                      cap)
{
    const char *root = cfg->out_beam[0] ? cfg->out_beam : "b_pup";
    if (cfg->n_nodes == 0u || after_node_idx < 0)
    {
        edit_copy_str(out_beam, cap, root);
        return;
    }
    if (after_node_idx >= (int) cfg->n_nodes)
    {
        after_node_idx = (int) cfg->n_nodes - 1;
    }

    const cacaosim_node_cfg_t *prev = &cfg->nodes[after_node_idx];
    if (prev->kind == CACAOSIM_NODE_DM)
    {
        edit_copy_str(out_beam, cap, prev->out_beam[0] ? prev->out_beam : root);
        return;
    }
    if (prev->kind == CACAOSIM_NODE_CORO)
    {
        if (is_wfs_kind(new_kind) && prev->out_lowfs[0] != '\0' &&
            count_beam_consumers(cfg, prev->out_lowfs, -1) == 0u)
        {
            edit_copy_str(out_beam, cap, prev->out_lowfs);
            return;
        }
        edit_copy_str(out_beam, cap, prev->out_beam[0] ? prev->out_beam : root);
        return;
    }
    if (prev->kind == CACAOSIM_NODE_SPLITTER)
    {
        uint32_t n_refl  = count_beam_consumers(cfg, prev->out_refl, -1);
        uint32_t n_trans = count_beam_consumers(cfg, prev->out_trans, -1);
        if (is_wfs_kind(new_kind))
        {
            if (prev->out_refl[0] != '\0' && (n_refl == 0u || n_trans > 0u))
            {
                edit_copy_str(out_beam, cap, prev->out_refl);
                return;
            }
            if (prev->out_trans[0] != '\0')
            {
                edit_copy_str(out_beam, cap, prev->out_trans);
                return;
            }
        }
        else
        {
            if (prev->out_trans[0] != '\0' && (n_trans == 0u || n_refl > 0u))
            {
                edit_copy_str(out_beam, cap, prev->out_trans);
                return;
            }
            if (prev->out_refl[0] != '\0')
            {
                edit_copy_str(out_beam, cap, prev->out_refl);
                return;
            }
        }
    }
    edit_copy_str(out_beam, cap, prev->in_beam[0] ? prev->in_beam : root);
}

int cacaosim_bench_add_node(
    cacaosim_bench_cfg_t *cfg,
    cacaosim_node_kind_t  kind,
    int                   is_ttm,
    int                   after_node_idx,
    const char           *target_beam)
{
    if (cfg == NULL || cfg->n_nodes >= CACAOSIM_BENCH_MAX_NODES)
    {
        return -1;
    }
    if (cfg->out_beam[0] == '\0')
    {
        edit_copy_str(cfg->out_beam, sizeof(cfg->out_beam), "b_pup");
    }

    const char *prefix = "node";
    switch (kind)
    {
    case CACAOSIM_NODE_DM:
        prefix = is_ttm ? "ttm" : "dm";
        break;
    case CACAOSIM_NODE_SPLITTER:
        prefix = "bs";
        break;
    case CACAOSIM_NODE_CORO:
        prefix = "coro";
        break;
    case CACAOSIM_NODE_PYRWFS:
        prefix = "pyr";
        break;
    case CACAOSIM_NODE_SHWFS:
        prefix = "sh";
        break;
    case CACAOSIM_NODE_ZWFS:
        prefix = "zwfs";
        break;
    case CACAOSIM_NODE_FPCAM:
        prefix = "sci";
        break;
    }

    char id[CACAOSIM_BENCH_NAME_LEN];
    make_unique_id(cfg, prefix, id, sizeof(id));

    char in_beam[CACAOSIM_BENCH_NAME_LEN];
    if (target_beam != NULL && target_beam[0] != '\0')
    {
        edit_copy_str(in_beam, sizeof(in_beam), target_beam);
    }
    else
    {
        derive_auto_target_beam(cfg, kind, after_node_idx, in_beam, sizeof(in_beam));
    }

    cacaosim_node_cfg_t nd;
    cacaosim_bench_node_defaults(&nd, kind, id, is_ttm, cfg->n_nodes);
    edit_copy_str(nd.in_beam, sizeof(nd.in_beam), in_beam);

    char base_buf[CACAOSIM_BENCH_NAME_LEN];
    if (kind == CACAOSIM_NODE_DM)
    {
        snprintf(base_buf, sizeof(base_buf), "b_%.48s", id);
        make_unique_beam(cfg, base_buf, nd.out_beam, sizeof(nd.out_beam));
        snprintf(base_buf, sizeof(base_buf), "%.48s_cmd", id);
        make_unique_shm(cfg, base_buf, nd.in_cmd, sizeof(nd.in_cmd));
        rewire_consumers_except(cfg, in_beam, nd.out_beam, -1, -1);
    }
    else if (kind == CACAOSIM_NODE_CORO)
    {
        snprintf(base_buf, sizeof(base_buf), "b_%.48s_sci", id);
        make_unique_beam(cfg, base_buf, nd.out_beam, sizeof(nd.out_beam));
        snprintf(base_buf, sizeof(base_buf), "b_%.48s_lowfs", id);
        make_unique_beam(cfg, base_buf, nd.out_lowfs, sizeof(nd.out_lowfs));
        rewire_consumers_except(cfg, in_beam, nd.out_beam, -1, -1);
    }
    else if (kind == CACAOSIM_NODE_SPLITTER)
    {
        snprintf(base_buf, sizeof(base_buf), "b_%.48s_refl", id);
        make_unique_beam(cfg, base_buf, nd.out_refl, sizeof(nd.out_refl));
        snprintf(base_buf, sizeof(base_buf), "b_%.48s_trans", id);
        make_unique_beam(cfg, base_buf, nd.out_trans, sizeof(nd.out_trans));
        rewire_consumers_except(cfg, in_beam, nd.out_trans, -1, -1);
    }
    else
    {
        if (kind == CACAOSIM_NODE_FPCAM)
        {
            make_unique_shm(cfg, "csim_fpcam", nd.out_cam, sizeof(nd.out_cam));
            make_unique_shm(cfg, "csim_fplong", nd.out_long, sizeof(nd.out_long));
        }
        else
        {
            make_unique_shm(cfg, "aol0_wfsim", nd.out_cam, sizeof(nd.out_cam));
        }
        auto_tune_sensor_wavelength(cfg, &nd);
    }

    int ins = (after_node_idx < 0) ? 0 : (after_node_idx + 1);
    if (ins > (int) cfg->n_nodes)
    {
        ins = (int) cfg->n_nodes;
    }
    /* If inserting after a terminal sensor, place inline optics right before it if spliced */
    if (after_node_idx >= 0 && after_node_idx < (int) cfg->n_nodes &&
        !is_inline_single_kind(cfg->nodes[after_node_idx].kind) &&
        cfg->nodes[after_node_idx].kind != CACAOSIM_NODE_SPLITTER &&
        (is_inline_single_kind(kind) || kind == CACAOSIM_NODE_SPLITTER))
    {
        ins = after_node_idx;
    }

    for (int i = (int) cfg->n_nodes; i > ins; i--)
    {
        cfg->nodes[i] = cfg->nodes[i - 1];
    }
    cfg->nodes[ins] = nd;
    cfg->n_nodes++;
    return ins;
}

int cacaosim_bench_clone_node(cacaosim_bench_cfg_t *cfg, uint32_t src_idx)
{
    if (cfg == NULL || src_idx >= cfg->n_nodes || cfg->n_nodes >= CACAOSIM_BENCH_MAX_NODES)
    {
        return -1;
    }
    const cacaosim_node_cfg_t src    = cfg->nodes[src_idx];
    int                       is_ttm = (src.kind == CACAOSIM_NODE_DM && src.dm.dm_mode == 1);
    int                       idx    = cacaosim_bench_add_node(cfg, src.kind, is_ttm,
                                                               (int) src_idx, NULL);
    if (idx < 0)
    {
        return -1;
    }

    cacaosim_node_cfg_t *dst = &cfg->nodes[idx];
    char                 saved_id[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_in[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_out[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_refl[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_trans[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_lowfs[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_cmd[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_cam[CACAOSIM_BENCH_NAME_LEN];
    char                 saved_long[CACAOSIM_BENCH_NAME_LEN];

    edit_copy_str(saved_id, sizeof(saved_id), dst->id);
    edit_copy_str(saved_in, sizeof(saved_in), dst->in_beam);
    edit_copy_str(saved_out, sizeof(saved_out), dst->out_beam);
    edit_copy_str(saved_refl, sizeof(saved_refl), dst->out_refl);
    edit_copy_str(saved_trans, sizeof(saved_trans), dst->out_trans);
    edit_copy_str(saved_lowfs, sizeof(saved_lowfs), dst->out_lowfs);
    edit_copy_str(saved_cmd, sizeof(saved_cmd), dst->in_cmd);
    edit_copy_str(saved_cam, sizeof(saved_cam), dst->out_cam);
    edit_copy_str(saved_long, sizeof(saved_long), dst->out_long);

    *dst = src;
    edit_copy_str(dst->id, sizeof(dst->id), saved_id);
    edit_copy_str(dst->in_beam, sizeof(dst->in_beam), saved_in);
    edit_copy_str(dst->out_beam, sizeof(dst->out_beam), saved_out);
    edit_copy_str(dst->out_refl, sizeof(dst->out_refl), saved_refl);
    edit_copy_str(dst->out_trans, sizeof(dst->out_trans), saved_trans);
    edit_copy_str(dst->out_lowfs, sizeof(dst->out_lowfs), saved_lowfs);
    edit_copy_str(dst->in_cmd, sizeof(dst->in_cmd), saved_cmd);
    edit_copy_str(dst->out_cam, sizeof(dst->out_cam), saved_cam);
    edit_copy_str(dst->out_long, sizeof(dst->out_long), saved_long);
    dst->out_opd[0] = '\0';
    return idx;
}

int cacaosim_bench_remove_node(cacaosim_bench_cfg_t *cfg, uint32_t node_idx)
{
    if (cfg == NULL || node_idx >= cfg->n_nodes)
    {
        return -1;
    }

    cacaosim_node_cfg_t rem    = cfg->nodes[node_idx];
    const char         *bridge = rem.in_beam[0] ? rem.in_beam
                                                : (cfg->out_beam[0] ? cfg->out_beam : "b_pup");

    if (rem.out_beam[0] != '\0')
    {
        rewire_consumers_except(cfg, rem.out_beam, bridge, (int) node_idx, -1);
    }
    if (rem.out_trans[0] != '\0')
    {
        rewire_consumers_except(cfg, rem.out_trans, bridge, (int) node_idx, -1);
    }
    if (rem.out_refl[0] != '\0')
    {
        rewire_consumers_except(cfg, rem.out_refl, bridge, (int) node_idx, -1);
    }
    if (rem.out_lowfs[0] != '\0')
    {
        rewire_consumers_except(cfg, rem.out_lowfs, bridge, (int) node_idx, -1);
    }

    for (uint32_t i = node_idx; i + 1u < cfg->n_nodes; i++)
    {
        cfg->nodes[i] = cfg->nodes[i + 1u];
    }
    cfg->n_nodes--;
    return 0;
}

int cacaosim_bench_move_node(cacaosim_bench_cfg_t *cfg, uint32_t node_idx, int dir)
{
    if (cfg == NULL || node_idx >= cfg->n_nodes || (dir != -1 && dir != 1))
    {
        return -1;
    }
    int dst_idx = (int) node_idx + dir;
    if (dst_idx < 0 || dst_idx >= (int) cfg->n_nodes)
    {
        return -1;
    }

    int                  a_idx = ((int) node_idx < dst_idx) ? (int) node_idx : dst_idx;
    int                  b_idx = ((int) node_idx < dst_idx) ? dst_idx : (int) node_idx;
    cacaosim_node_cfg_t *a     = &cfg->nodes[a_idx];
    cacaosim_node_cfg_t *b     = &cfg->nodes[b_idx];

    /* Case 1: Inline A -> Inline B along same beam chain: swap physical optical order */
    if (is_inline_single_kind(a->kind) && is_inline_single_kind(b->kind) &&
        a->out_beam[0] != '\0' && strcmp(b->in_beam, a->out_beam) == 0)
    {
        char in0[CACAOSIM_BENCH_NAME_LEN];
        char out_a[CACAOSIM_BENCH_NAME_LEN];
        char out_b[CACAOSIM_BENCH_NAME_LEN];
        edit_copy_str(in0, sizeof(in0), a->in_beam);
        edit_copy_str(out_a, sizeof(out_a), a->out_beam);
        edit_copy_str(out_b, sizeof(out_b), b->out_beam);

        rewire_consumers_except(cfg, out_b, out_a, a_idx, b_idx);
        edit_copy_str(b->in_beam, sizeof(b->in_beam), in0);
        edit_copy_str(a->in_beam, sizeof(a->in_beam), out_b);
    }
    /* Case 2: Inline A -> Splitter B: move A downstream onto B's transmitted arm */
    else if (is_inline_single_kind(a->kind) && b->kind == CACAOSIM_NODE_SPLITTER &&
             a->out_beam[0] != '\0' && strcmp(b->in_beam, a->out_beam) == 0 &&
             b->out_trans[0] != '\0')
    {
        char in0[CACAOSIM_BENCH_NAME_LEN];
        char out_a[CACAOSIM_BENCH_NAME_LEN];
        char arm_t[CACAOSIM_BENCH_NAME_LEN];
        edit_copy_str(in0, sizeof(in0), a->in_beam);
        edit_copy_str(out_a, sizeof(out_a), a->out_beam);
        edit_copy_str(arm_t, sizeof(arm_t), b->out_trans);

        rewire_consumers_except(cfg, arm_t, out_a, a_idx, b_idx);
        edit_copy_str(b->in_beam, sizeof(b->in_beam), in0);
        edit_copy_str(a->in_beam, sizeof(a->in_beam), arm_t);
    }
    /* Case 3: Splitter A -> Inline B (on refl or trans arm): hoist B upstream into common path */
    else if (a->kind == CACAOSIM_NODE_SPLITTER && is_inline_single_kind(b->kind) &&
             b->out_beam[0] != '\0' &&
             ((a->out_trans[0] != '\0' && strcmp(b->in_beam, a->out_trans) == 0) ||
              (a->out_refl[0] != '\0' && strcmp(b->in_beam, a->out_refl) == 0)))
    {
        char in0[CACAOSIM_BENCH_NAME_LEN];
        char arm[CACAOSIM_BENCH_NAME_LEN];
        char out_b[CACAOSIM_BENCH_NAME_LEN];
        edit_copy_str(in0, sizeof(in0), a->in_beam);
        edit_copy_str(arm, sizeof(arm), b->in_beam);
        edit_copy_str(out_b, sizeof(out_b), b->out_beam);

        rewire_consumers_except(cfg, out_b, arm, a_idx, b_idx);
        edit_copy_str(b->in_beam, sizeof(b->in_beam), in0);
        edit_copy_str(a->in_beam, sizeof(a->in_beam), out_b);
    }

    cacaosim_node_cfg_t tmp = cfg->nodes[a_idx];
    cfg->nodes[a_idx]       = cfg->nodes[b_idx];
    cfg->nodes[b_idx]       = tmp;
    return dst_idx;
}

int cacaosim_bench_swap_branches(cacaosim_bench_cfg_t *cfg, uint32_t node_idx)
{
    if (cfg == NULL || node_idx >= cfg->n_nodes)
    {
        return -1;
    }
    const cacaosim_node_cfg_t *nd = &cfg->nodes[node_idx];
    const char                *b1 = NULL;
    const char                *b2 = NULL;

    if (nd->kind == CACAOSIM_NODE_SPLITTER && nd->out_refl[0] != '\0' && nd->out_trans[0] != '\0')
    {
        b1 = nd->out_refl;
        b2 = nd->out_trans;
    }
    else if (nd->kind == CACAOSIM_NODE_CORO && nd->out_beam[0] != '\0' && nd->out_lowfs[0] != '\0')
    {
        b1 = nd->out_beam;
        b2 = nd->out_lowfs;
    }
    else
    {
        return -1;
    }

    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        if (i == node_idx)
        {
            continue;
        }
        if (strcmp(cfg->nodes[i].in_beam, b1) == 0)
        {
            edit_copy_str(cfg->nodes[i].in_beam, sizeof(cfg->nodes[i].in_beam), b2);
        }
        else if (strcmp(cfg->nodes[i].in_beam, b2) == 0)
        {
            edit_copy_str(cfg->nodes[i].in_beam, sizeof(cfg->nodes[i].in_beam), b1);
        }
    }
    return 0;
}

int cacaosim_bench_rename_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx,
    const char           *new_id,
    char                 *err_buf,
    size_t                err_cap)
{
    if (cfg == NULL || node_idx >= cfg->n_nodes || new_id == NULL)
    {
        edit_set_err(err_buf, err_cap, "Invalid node or identifier");
        return -1;
    }
    size_t len = strlen(new_id);
    if (len == 0u || len >= CACAOSIM_BENCH_NAME_LEN)
    {
        edit_set_err(err_buf, err_cap, "Identifier must be 1..63 characters");
        return -1;
    }
    for (size_t i = 0; i < len; i++)
    {
        unsigned char c = (unsigned char) new_id[i];
        if (!isalnum(c) && c != '_' && c != '-')
        {
            edit_set_err(err_buf, err_cap, "Identifier may only contain [A-Za-z0-9_-]");
            return -1;
        }
    }
    if (node_id_exists(cfg, new_id, (int) node_idx))
    {
        edit_set_err(err_buf, err_cap, "Duplicate node identifier");
        return -1;
    }
    edit_copy_str(cfg->nodes[node_idx].id, sizeof(cfg->nodes[node_idx].id), new_id);
    return 0;
}

int cacaosim_bench_rewire_beam_out(
    cacaosim_bench_cfg_t *cfg,
    const char           *old_beam,
    const char           *new_beam)
{
    if (cfg == NULL || old_beam == NULL || new_beam == NULL || new_beam[0] == '\0')
    {
        return -1;
    }
    if (strcmp(old_beam, new_beam) == 0)
    {
        return 0;
    }
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        if (strcmp(cfg->nodes[i].in_beam, old_beam) == 0)
        {
            edit_copy_str(cfg->nodes[i].in_beam, sizeof(cfg->nodes[i].in_beam), new_beam);
        }
    }
    return 0;
}
