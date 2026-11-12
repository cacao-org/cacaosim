// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_cfg.c
 * @brief   INI/TOML-style parser for declarative multi-stage .bench optical configurations
 */

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bench/cacaosim_bench.h"

static void set_err(char *buf, size_t cap, const char *fmt, ...)
{
    if (buf == NULL || cap == 0)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
}

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (dst == NULL || cap == 0)
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

static char *trim_inplace(char *s)
{
    while (*s != '\0' && isspace((unsigned char) *s))
    {
        s++;
    }
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char) s[len - 1]))
    {
        s[--len] = '\0';
    }
    if (len >= 2 && ((s[0] == '"' && s[len - 1] == '"') || (s[0] == '\'' && s[len - 1] == '\'')))
    {
        s[len - 1] = '\0';
        s++;
    }
    return s;
}

static void strip_comment(char *line)
{
    int in_dq = 0;
    int in_sq = 0;
    for (char *p = line; *p != '\0'; p++)
    {
        if (*p == '"' && !in_sq)
        {
            in_dq = !in_dq;
        }
        else if (*p == '\'' && !in_dq)
        {
            in_sq = !in_sq;
        }
        else if (!in_dq && !in_sq && (*p == '#' || *p == ';'))
        {
            *p = '\0';
            return;
        }
    }
}

static int parse_bool(const char *v)
{
    if (strcasecmp(v, "1") == 0 || strcasecmp(v, "true") == 0 || strcasecmp(v, "on") == 0 ||
        strcasecmp(v, "yes") == 0)
    {
        return 1;
    }
    return 0;
}

static int parse_backend(const char *v)
{
    if (strcasecmp(v, "cpu") == 0)
    {
        return CACAOSIM_BACKEND_CPU;
    }
    if (strcasecmp(v, "cuda") == 0 || strcasecmp(v, "gpu") == 0)
    {
        return CACAOSIM_BACKEND_CUDA;
    }
    if (strcasecmp(v, "auto") == 0)
    {
        return CACAOSIM_BACKEND_AUTO;
    }
    return atoi(v);
}

static int parse_dtype(const char *v)
{
    if (strcasecmp(v, "float32") == 0 || strcasecmp(v, "f32") == 0 || strcasecmp(v, "float") == 0)
    {
        return CACAOSIM_DTYPE_F32;
    }
    if (strcasecmp(v, "uint16") == 0 || strcasecmp(v, "u16") == 0)
    {
        return CACAOSIM_DTYPE_U16;
    }
    return atoi(v);
}

static int parse_split_mode(const char *v)
{
    if (strcasecmp(v, "gray") == 0)
    {
        return CACAOSIM_SPLIT_GRAY;
    }
    if (strcasecmp(v, "dichroic_lp") == 0 || strcasecmp(v, "lp") == 0 ||
        strcasecmp(v, "longpass") == 0)
    {
        return CACAOSIM_SPLIT_DICHROIC_LP;
    }
    if (strcasecmp(v, "dichroic_sp") == 0 || strcasecmp(v, "sp") == 0 ||
        strcasecmp(v, "shortpass") == 0)
    {
        return CACAOSIM_SPLIT_DICHROIC_SP;
    }
    if (strcasecmp(v, "dichroic_bp") == 0 || strcasecmp(v, "bp") == 0 ||
        strcasecmp(v, "band_reflect") == 0)
    {
        return CACAOSIM_SPLIT_DICHROIC_BP;
    }
    return atoi(v);
}

static int parse_coro_mode(const char *v)
{
    if (strcasecmp(v, "lyot_spot") == 0 || strcasecmp(v, "lyot") == 0)
    {
        return CACAOSIM_CORO_LYOT_SPOT;
    }
    if (strcasecmp(v, "phase_dimple") == 0 || strcasecmp(v, "roddier") == 0)
    {
        return CACAOSIM_CORO_PHASE_DIMPLE;
    }
    if (strcasecmp(v, "vortex") == 0)
    {
        return CACAOSIM_CORO_VORTEX;
    }
    if (strcasecmp(v, "4qpm") == 0)
    {
        return CACAOSIM_CORO_4QPM;
    }
    return atoi(v);
}

static void init_det_defaults(cacaosim_det_cfg_t *det, float flux, float ron)
{
    memset(det, 0, sizeof(*det));
    det->flux_ph      = flux;
    det->dark_e       = 0.0f;
    det->ron_e        = ron;
    det->gain_eadu    = 1.0f;
    det->bias_adu     = 100.0f;
    det->sat_adu      = 65535.0f;
    det->shutter      = 1;
    det->photon_noise = 1;
    det->out_dtype    = CACAOSIM_DTYPE_U16;
}

static void init_node_defaults(
    cacaosim_node_cfg_t *nd,
    cacaosim_node_kind_t kind,
    const char          *id,
    int                  is_ttm,
    uint32_t             idx)
{
    memset(nd, 0, sizeof(*nd));
    nd->kind            = kind;
    copy_str(nd->id, sizeof(nd->id), id);
    nd->tput            = 1.0f;
    nd->navg            = 1u;
    nd->rate_hz         = 0.0f;
    nd->bw_frac         = 0.20f;
    nd->qe              = 0.80f;
    nd->use_beam_flux   = 1;
    nd->reg.scale       = 1.0f;
    nd->reg_refl.scale  = 1.0f;
    nd->reg_trans.scale = 1.0f;

    uint64_t seed_base = 10007ULL + 7919ULL * (uint64_t) (idx + 1u);

    if (kind == CACAOSIM_NODE_DM)
    {
        snprintf(nd->in_cmd, sizeof(nd->in_cmd), "%s_cmd", id);
        nd->tput            = 0.98f;
        nd->dm.dm_mode      = is_ttm ? 1 : 0;
        nd->dm.nact_x       = is_ttm ? 2u : 16u;
        nd->dm.nact_y       = is_ttm ? 1u : 16u;
        nd->dm.dm_diam      = 0.0f;
        nd->dm.if_coupling  = 0.20f;
        nd->dm.refl_scale   = 2.0f;
        nd->dm.max_stroke   = 5.0f;
        nd->dm.dac_bits     = 16;
        nd->dm.dyn_mode     = 1;
        nd->dm.tau_sec      = 0.00012f;
        nd->dm.f0_hz        = 2500.0f;
        nd->dm.zeta         = 0.70f;
        nd->dm.delay_frames = 1.0f;
        nd->dm.mag_scale    = 1.0f;
    }
    else if (kind == CACAOSIM_NODE_SPLITTER)
    {
        nd->split.mode       = CACAOSIM_SPLIT_DICHROIC_LP;
        nd->split.r_gray     = 0.50f;
        nd->split.t_gray     = 0.50f;
        nd->split.lambda_cut = 0.95f;
        nd->split.band_width = 0.25f;
        nd->split.roll_um    = 0.03f;
        nd->split.eff_refl   = 0.95f;
        nd->split.eff_trans  = 0.95f;
    }
    else if (kind == CACAOSIM_NODE_CORO)
    {
        nd->tput                 = 0.92f;
        nd->coro.mode            = CACAOSIM_CORO_LYOT_SPOT;
        nd->coro.n_mft           = 48u;
        nd->coro.pad_factor      = 4u;
        nd->coro.lambda_um       = 1.65f;
        nd->coro.fpm_rad_ld      = 3.0f;
        nd->coro.fpm_amp         = 0.0f;
        nd->coro.fpm_phase_rad   = CACAOSIM_PI_F;
        nd->coro.vortex_charge   = 2;
        nd->coro.lowfs_refl      = 0.95f;
        nd->coro.lyot_outer      = 0.85f;
        nd->coro.lyot_obs        = 0.36f;
        nd->coro.lyot_nspider    = 4;
        nd->coro.lyot_spider_w   = 0.02f;
        nd->coro.lyot_spider_deg = 45.0f;
    }
    else if (kind == CACAOSIM_NODE_PYRWFS)
    {
        snprintf(nd->out_cam, sizeof(nd->out_cam), "%s_cam", id);
        nd->pyrwfs.pad_factor = 2u;
        nd->pyrwfs.ncam       = 64u;
        nd->pyrwfs.lambda_um  = 0.75f;
        nd->pyrwfs.mod_rad_ld = 3.0f;
        nd->pyrwfs.mod_npts   = 16;
        nd->pyrwfs.pup_sep    = 0.50f;
        nd->pyrwfs.roof_ld    = 0.0f;
        nd->pyrwfs.fstop_frac = 0.45f;
        nd->pyrwfs.rng_seed   = seed_base;
        init_det_defaults(&nd->pyrwfs.det, 100000.0f, 0.5f);
    }
    else if (kind == CACAOSIM_NODE_SHWFS)
    {
        snprintf(nd->out_cam, sizeof(nd->out_cam), "%s_cam", id);
        nd->shwfs.n_sub       = 16u;
        nd->shwfs.pix_per_sub = 8u;
        nd->shwfs.pad_factor  = 2u;
        nd->shwfs.lambda_um   = 0.75f;
        nd->shwfs.rng_seed    = seed_base;
        init_det_defaults(&nd->shwfs.det, 100000.0f, 0.5f);
    }
    else if (kind == CACAOSIM_NODE_ZWFS)
    {
        snprintf(nd->out_cam, sizeof(nd->out_cam), "%s_cam", id);
        nd->zwfs.ncam       = 64u;
        nd->zwfs.n_mft      = 32u;
        nd->zwfs.lambda_um  = 0.75f;
        nd->zwfs.dimple_ld  = 1.06f;
        nd->zwfs.dimple_rad = 0.5f * CACAOSIM_PI_F;
        nd->zwfs.dimple_amp = 1.0f;
        nd->zwfs.rng_seed   = seed_base;
        init_det_defaults(&nd->zwfs.det, 100000.0f, 0.5f);
    }
    else if (kind == CACAOSIM_NODE_FPCAM)
    {
        snprintf(nd->out_cam, sizeof(nd->out_cam), "%s_cam", id);
        nd->bw_frac           = 0.18f;
        nd->qe                = 0.85f;
        nd->fpcam.ncam        = 128u;
        nd->fpcam.prop_mode   = CACAOSIM_FPCAM_PROP_MFT;
        nd->fpcam.pad_factor  = 2u;
        nd->fpcam.lambda_um   = 1.65f;
        nd->fpcam.bw_frac     = 0.0f;
        nd->fpcam.nlambda     = 1;
        nd->fpcam.pixscale_ld = 3.0f;
        nd->fpcam.rng_seed    = seed_base;
        init_det_defaults(&nd->fpcam.det, 1000000.0f, 1.0f);
    }
}

void cacaosim_bench_cfg_defaults(cacaosim_bench_cfg_t *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    copy_str(cfg->name, sizeof(cfg->name), "ao_bench");
    cfg->nwf        = 128u;
    cfg->pup_diam   = 0.9375f;
    cfg->cobs       = 0.30f;
    cfg->nspider    = 4;
    cfg->spider_w   = 0.012f;
    cfg->spider_deg = 45.0f;
    cfg->tel_diam_m = 8.2f;
    cfg->star_mag   = 8.0f;
    cfg->star_teff  = 5778.0f;
    cfg->tput_tel   = 0.75f;
    cfg->master_hz  = 2000.0f;
    copy_str(cfg->in_wf, sizeof(cfg->in_wf), "outarraypha");
    copy_str(cfg->in_pup, sizeof(cfg->in_pup), "csim_pup");
    copy_str(cfg->out_beam, sizeof(cfg->out_beam), "b_pup");
    cfg->wf_unit    = 1;
    cfg->ref_lambda = 0.50f;
    cfg->turb_gain  = 1.0f;
    cfg->backend    = CACAOSIM_BACKEND_AUTO;
}

static int apply_det_kv(cacaosim_det_cfg_t *det, uint64_t *seed, const char *k, const char *v)
{
    if (strcmp(k, "flux_ph") == 0)
    {
        det->flux_ph = strtof(v, NULL);
        return 1;
    }
    if (strcmp(k, "dark_e") == 0)
    {
        det->dark_e = strtof(v, NULL);
        return 1;
    }
    if (strcmp(k, "ron_e") == 0 || strcmp(k, "ron") == 0)
    {
        det->ron_e = strtof(v, NULL);
        return 1;
    }
    if (strcmp(k, "gain_eadu") == 0)
    {
        det->gain_eadu = strtof(v, NULL);
        return 1;
    }
    if (strcmp(k, "bias_adu") == 0)
    {
        det->bias_adu = strtof(v, NULL);
        return 1;
    }
    if (strcmp(k, "sat_adu") == 0)
    {
        det->sat_adu = strtof(v, NULL);
        return 1;
    }
    if (strcmp(k, "shutter") == 0)
    {
        det->shutter = parse_bool(v);
        return 1;
    }
    if (strcmp(k, "photon_noise") == 0)
    {
        det->photon_noise = parse_bool(v);
        return 1;
    }
    if (strcmp(k, "out_dtype") == 0)
    {
        det->out_dtype = parse_dtype(v);
        return 1;
    }
    if (strcmp(k, "rng_seed") == 0 && seed != NULL)
    {
        *seed = (uint64_t) strtoull(v, NULL, 10);
        return 1;
    }
    return 0;
}

static int apply_bench_kv(cacaosim_bench_cfg_t *cfg, const char *k, const char *v)
{
    if (strcmp(k, "name") == 0)
    {
        copy_str(cfg->name, sizeof(cfg->name), v);
    }
    else if (strcmp(k, "nwf") == 0)
    {
        cfg->nwf = (uint32_t) strtoul(v, NULL, 10);
    }
    else if (strcmp(k, "pup_diam") == 0)
    {
        cfg->pup_diam = strtof(v, NULL);
    }
    else if (strcmp(k, "cobs") == 0 || strcmp(k, "obs_frac") == 0)
    {
        cfg->cobs = strtof(v, NULL);
    }
    else if (strcmp(k, "nspider") == 0 || strcmp(k, "n_spiders") == 0)
    {
        cfg->nspider = atoi(v);
    }
    else if (strcmp(k, "spider_w") == 0)
    {
        cfg->spider_w = strtof(v, NULL);
    }
    else if (strcmp(k, "spider_deg") == 0)
    {
        cfg->spider_deg = strtof(v, NULL);
    }
    else if (strcmp(k, "tel_diam_m") == 0)
    {
        cfg->tel_diam_m = strtof(v, NULL);
    }
    else if (strcmp(k, "star_mag") == 0)
    {
        cfg->star_mag = strtof(v, NULL);
    }
    else if (strcmp(k, "star_teff") == 0)
    {
        cfg->star_teff = strtof(v, NULL);
    }
    else if (strcmp(k, "tput_tel") == 0 || strcmp(k, "tel_tput") == 0)
    {
        cfg->tput_tel = strtof(v, NULL);
    }
    else if (strcmp(k, "master_hz") == 0 || strcmp(k, "frame_rate") == 0)
    {
        cfg->master_hz = strtof(v, NULL);
    }
    else if (strcmp(k, "in_wf") == 0)
    {
        copy_str(cfg->in_wf, sizeof(cfg->in_wf), v);
    }
    else if (strcmp(k, "in_amp") == 0)
    {
        copy_str(cfg->in_amp, sizeof(cfg->in_amp), v);
    }
    else if (strcmp(k, "in_pup") == 0)
    {
        copy_str(cfg->in_pup, sizeof(cfg->in_pup), v);
    }
    else if (strcmp(k, "in_ncpa") == 0)
    {
        copy_str(cfg->in_ncpa, sizeof(cfg->in_ncpa), v);
    }
    else if (strcmp(k, "out_beam") == 0 || strcmp(k, "source_beam") == 0)
    {
        copy_str(cfg->out_beam, sizeof(cfg->out_beam), v);
    }
    else if (strcmp(k, "wf_unit") == 0)
    {
        cfg->wf_unit = (strcasecmp(v, "opd") == 0 || strcasecmp(v, "um") == 0) ? 0 : atoi(v);
        if (strcasecmp(v, "rad") == 0 || strcasecmp(v, "phase") == 0)
        {
            cfg->wf_unit = 1;
        }
    }
    else if (strcmp(k, "ref_lambda") == 0)
    {
        cfg->ref_lambda = strtof(v, NULL);
    }
    else if (strcmp(k, "turb_gain") == 0)
    {
        cfg->turb_gain = strtof(v, NULL);
    }
    else if (strcmp(k, "backend") == 0)
    {
        cfg->backend = parse_backend(v);
    }
    else
    {
        return -1;
    }
    return 0;
}

static int apply_node_kv(cacaosim_node_cfg_t *nd, const char *k, const char *v)
{
    if (strcmp(k, "in") == 0 || strcmp(k, "in_beam") == 0)
    {
        copy_str(nd->in_beam, sizeof(nd->in_beam), v);
        return 0;
    }
    if (strcmp(k, "out") == 0 || strcmp(k, "out_beam") == 0)
    {
        copy_str(nd->out_beam, sizeof(nd->out_beam), v);
        return 0;
    }
    if (strcmp(k, "tput") == 0)
    {
        nd->tput = strtof(v, NULL);
        return 0;
    }
    if (strcmp(k, "shift_x") == 0)
    {
        nd->reg.shift_x = strtof(v, NULL);
        nd->dm.shift_x  = nd->reg.shift_x;
        return 0;
    }
    if (strcmp(k, "shift_y") == 0)
    {
        nd->reg.shift_y = strtof(v, NULL);
        nd->dm.shift_y  = nd->reg.shift_y;
        return 0;
    }
    if (strcmp(k, "rot_deg") == 0)
    {
        nd->reg.rot_deg = strtof(v, NULL);
        nd->dm.rot_deg  = nd->reg.rot_deg;
        return 0;
    }
    if (strcmp(k, "mag_scale") == 0)
    {
        nd->reg.scale    = strtof(v, NULL);
        nd->dm.mag_scale = nd->reg.scale;
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_DM)
    {
        if (strcmp(k, "in_cmd") == 0)
        {
            copy_str(nd->in_cmd, sizeof(nd->in_cmd), v);
        }
        else if (strcmp(k, "out_opd") == 0)
        {
            copy_str(nd->out_opd, sizeof(nd->out_opd), v);
        }
        else if (strcmp(k, "in_flat") == 0)
        {
            copy_str(nd->in_flat, sizeof(nd->in_flat), v);
        }
        else if (strcmp(k, "dm_mode") == 0)
        {
            nd->dm.dm_mode = atoi(v);
        }
        else if (strcmp(k, "nact_x") == 0)
        {
            nd->dm.nact_x = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "nact_y") == 0)
        {
            nd->dm.nact_y = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "nact") == 0)
        {
            nd->dm.nact_x = (uint32_t) strtoul(v, NULL, 10);
            nd->dm.nact_y = nd->dm.nact_x;
        }
        else if (strcmp(k, "dm_diam") == 0)
        {
            nd->dm.dm_diam = strtof(v, NULL);
        }
        else if (strcmp(k, "if_coupling") == 0)
        {
            nd->dm.if_coupling = strtof(v, NULL);
        }
        else if (strcmp(k, "refl_scale") == 0)
        {
            nd->dm.refl_scale = strtof(v, NULL);
        }
        else if (strcmp(k, "max_stroke") == 0)
        {
            nd->dm.max_stroke = strtof(v, NULL);
        }
        else if (strcmp(k, "dac_bits") == 0)
        {
            nd->dm.dac_bits = atoi(v);
        }
        else if (strcmp(k, "nonlin_quad") == 0)
        {
            nd->dm.nonlin_quad = strtof(v, NULL);
        }
        else if (strcmp(k, "dyn_mode") == 0)
        {
            if (strcasecmp(v, "instant") == 0)
            {
                nd->dm.dyn_mode = 0;
            }
            else if (strcasecmp(v, "rc") == 0)
            {
                nd->dm.dyn_mode = 1;
            }
            else if (strcasecmp(v, "osc") == 0)
            {
                nd->dm.dyn_mode = 2;
            }
            else
            {
                nd->dm.dyn_mode = atoi(v);
            }
        }
        else if (strcmp(k, "tau_sec") == 0)
        {
            nd->dm.tau_sec = strtof(v, NULL);
        }
        else if (strcmp(k, "tau_us") == 0)
        {
            nd->dm.tau_sec = strtof(v, NULL) * 1.0e-6f;
        }
        else if (strcmp(k, "f0_hz") == 0)
        {
            nd->dm.f0_hz = strtof(v, NULL);
        }
        else if (strcmp(k, "zeta") == 0)
        {
            nd->dm.zeta = strtof(v, NULL);
        }
        else if (strcmp(k, "delay_frames") == 0)
        {
            nd->dm.delay_frames = strtof(v, NULL);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_SPLITTER)
    {
        if (strcmp(k, "out_refl") == 0)
        {
            copy_str(nd->out_refl, sizeof(nd->out_refl), v);
        }
        else if (strcmp(k, "out_trans") == 0)
        {
            copy_str(nd->out_trans, sizeof(nd->out_trans), v);
        }
        else if (strcmp(k, "mode") == 0 || strcmp(k, "split_mode") == 0)
        {
            nd->split.mode = parse_split_mode(v);
        }
        else if (strcmp(k, "r_gray") == 0)
        {
            nd->split.r_gray = strtof(v, NULL);
        }
        else if (strcmp(k, "t_gray") == 0)
        {
            nd->split.t_gray = strtof(v, NULL);
        }
        else if (strcmp(k, "lambda_cut") == 0)
        {
            nd->split.lambda_cut = strtof(v, NULL);
        }
        else if (strcmp(k, "band_width") == 0)
        {
            nd->split.band_width = strtof(v, NULL);
        }
        else if (strcmp(k, "roll_um") == 0)
        {
            nd->split.roll_um = strtof(v, NULL);
        }
        else if (strcmp(k, "eff_refl") == 0)
        {
            nd->split.eff_refl = strtof(v, NULL);
        }
        else if (strcmp(k, "eff_trans") == 0)
        {
            nd->split.eff_trans = strtof(v, NULL);
        }
        else if (strcmp(k, "refl_dx") == 0)
        {
            nd->reg_refl.shift_x = strtof(v, NULL);
        }
        else if (strcmp(k, "refl_dy") == 0)
        {
            nd->reg_refl.shift_y = strtof(v, NULL);
        }
        else if (strcmp(k, "refl_rot") == 0)
        {
            nd->reg_refl.rot_deg = strtof(v, NULL);
        }
        else if (strcmp(k, "refl_scale") == 0)
        {
            nd->reg_refl.scale = strtof(v, NULL);
        }
        else if (strcmp(k, "trans_dx") == 0)
        {
            nd->reg_trans.shift_x = strtof(v, NULL);
        }
        else if (strcmp(k, "trans_dy") == 0)
        {
            nd->reg_trans.shift_y = strtof(v, NULL);
        }
        else if (strcmp(k, "trans_rot") == 0)
        {
            nd->reg_trans.rot_deg = strtof(v, NULL);
        }
        else if (strcmp(k, "trans_scale") == 0)
        {
            nd->reg_trans.scale = strtof(v, NULL);
        }
        else if (strcmp(k, "ncpa_refl") == 0)
        {
            copy_str(nd->ncpa_refl, sizeof(nd->ncpa_refl), v);
        }
        else if (strcmp(k, "ncpa_trans") == 0)
        {
            copy_str(nd->ncpa_trans, sizeof(nd->ncpa_trans), v);
        }
        else if (strcmp(k, "pup_refl") == 0)
        {
            copy_str(nd->pup_refl, sizeof(nd->pup_refl), v);
        }
        else if (strcmp(k, "pup_trans") == 0)
        {
            copy_str(nd->pup_trans, sizeof(nd->pup_trans), v);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_CORO)
    {
        if (strcmp(k, "out_lowfs") == 0)
        {
            copy_str(nd->out_lowfs, sizeof(nd->out_lowfs), v);
        }
        else if (strcmp(k, "in_lyot") == 0)
        {
            copy_str(nd->in_lyot, sizeof(nd->in_lyot), v);
        }
        else if (strcmp(k, "mode") == 0 || strcmp(k, "coro_mode") == 0)
        {
            nd->coro.mode = parse_coro_mode(v);
        }
        else if (strcmp(k, "n_mft") == 0)
        {
            nd->coro.n_mft = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "pad_factor") == 0)
        {
            nd->coro.pad_factor = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "lambda_um") == 0)
        {
            nd->coro.lambda_um = strtof(v, NULL);
        }
        else if (strcmp(k, "fpm_rad_ld") == 0)
        {
            nd->coro.fpm_rad_ld = strtof(v, NULL);
        }
        else if (strcmp(k, "fpm_amp") == 0)
        {
            nd->coro.fpm_amp = strtof(v, NULL);
        }
        else if (strcmp(k, "fpm_phase") == 0)
        {
            nd->coro.fpm_phase_rad = strtof(v, NULL);
        }
        else if (strcmp(k, "vortex_charge") == 0)
        {
            nd->coro.vortex_charge = atoi(v);
        }
        else if (strcmp(k, "lowfs_refl") == 0)
        {
            nd->coro.lowfs_refl = strtof(v, NULL);
        }
        else if (strcmp(k, "lyot_outer") == 0)
        {
            nd->coro.lyot_outer = strtof(v, NULL);
        }
        else if (strcmp(k, "lyot_obs") == 0)
        {
            nd->coro.lyot_obs = strtof(v, NULL);
        }
        else if (strcmp(k, "lyot_nspider") == 0)
        {
            nd->coro.lyot_nspider = atoi(v);
        }
        else if (strcmp(k, "lyot_spider_w") == 0)
        {
            nd->coro.lyot_spider_w = strtof(v, NULL);
        }
        else if (strcmp(k, "lyot_spider_deg") == 0)
        {
            nd->coro.lyot_spider_deg = strtof(v, NULL);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    /* Sensor common parameters */
    if (strcmp(k, "out_cam") == 0)
    {
        copy_str(nd->out_cam, sizeof(nd->out_cam), v);
        return 0;
    }
    if (strcmp(k, "in_ncpa") == 0)
    {
        copy_str(nd->in_ncpa, sizeof(nd->in_ncpa), v);
        return 0;
    }
    if (strcmp(k, "navg") == 0)
    {
        nd->navg = (uint32_t) strtoul(v, NULL, 10);
        return 0;
    }
    if (strcmp(k, "rate_hz") == 0)
    {
        nd->rate_hz = strtof(v, NULL);
        return 0;
    }
    if (strcmp(k, "bw_frac") == 0)
    {
        nd->bw_frac       = strtof(v, NULL);
        nd->fpcam.bw_frac = nd->bw_frac;
        return 0;
    }
    if (strcmp(k, "qe") == 0)
    {
        nd->qe = strtof(v, NULL);
        return 0;
    }
    if (strcmp(k, "use_beam_flux") == 0)
    {
        nd->use_beam_flux = parse_bool(v);
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_PYRWFS)
    {
        if (apply_det_kv(&nd->pyrwfs.det, &nd->pyrwfs.rng_seed, k, v))
        {
            return 0;
        }
        if (strcmp(k, "ncam") == 0)
        {
            nd->pyrwfs.ncam = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "pad_factor") == 0)
        {
            nd->pyrwfs.pad_factor = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "lambda_um") == 0)
        {
            nd->pyrwfs.lambda_um = strtof(v, NULL);
        }
        else if (strcmp(k, "mod_rad") == 0 || strcmp(k, "mod_rad_ld") == 0)
        {
            nd->pyrwfs.mod_rad_ld = strtof(v, NULL);
        }
        else if (strcmp(k, "mod_npts") == 0)
        {
            nd->pyrwfs.mod_npts = atoi(v);
        }
        else if (strcmp(k, "pup_sep") == 0)
        {
            nd->pyrwfs.pup_sep = strtof(v, NULL);
        }
        else if (strcmp(k, "roof_ld") == 0)
        {
            nd->pyrwfs.roof_ld = strtof(v, NULL);
        }
        else if (strcmp(k, "fstop_frac") == 0)
        {
            nd->pyrwfs.fstop_frac = strtof(v, NULL);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_SHWFS)
    {
        if (apply_det_kv(&nd->shwfs.det, &nd->shwfs.rng_seed, k, v))
        {
            return 0;
        }
        if (strcmp(k, "n_sub") == 0)
        {
            nd->shwfs.n_sub = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "pix_per_sub") == 0)
        {
            nd->shwfs.pix_per_sub = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "pad_factor") == 0)
        {
            nd->shwfs.pad_factor = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "lambda_um") == 0)
        {
            nd->shwfs.lambda_um = strtof(v, NULL);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_ZWFS)
    {
        if (apply_det_kv(&nd->zwfs.det, &nd->zwfs.rng_seed, k, v))
        {
            return 0;
        }
        if (strcmp(k, "ncam") == 0)
        {
            nd->zwfs.ncam = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "n_mft") == 0)
        {
            nd->zwfs.n_mft = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "lambda_um") == 0)
        {
            nd->zwfs.lambda_um = strtof(v, NULL);
        }
        else if (strcmp(k, "dimple_ld") == 0)
        {
            nd->zwfs.dimple_ld = strtof(v, NULL);
        }
        else if (strcmp(k, "dimple_rad") == 0)
        {
            nd->zwfs.dimple_rad = strtof(v, NULL);
        }
        else if (strcmp(k, "dimple_amp") == 0)
        {
            nd->zwfs.dimple_amp = strtof(v, NULL);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    if (nd->kind == CACAOSIM_NODE_FPCAM)
    {
        if (apply_det_kv(&nd->fpcam.det, &nd->fpcam.rng_seed, k, v))
        {
            return 0;
        }
        if (strcmp(k, "out_long") == 0)
        {
            copy_str(nd->out_long, sizeof(nd->out_long), v);
        }
        else if (strcmp(k, "ncam") == 0)
        {
            nd->fpcam.ncam = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "prop_mode") == 0)
        {
            nd->fpcam.prop_mode =
                (strcasecmp(v, "fft") == 0) ? CACAOSIM_FPCAM_PROP_FFT : atoi(v);
        }
        else if (strcmp(k, "pad_factor") == 0)
        {
            nd->fpcam.pad_factor = (uint32_t) strtoul(v, NULL, 10);
        }
        else if (strcmp(k, "lambda_um") == 0)
        {
            nd->fpcam.lambda_um = strtof(v, NULL);
        }
        else if (strcmp(k, "nlambda") == 0)
        {
            nd->fpcam.nlambda = atoi(v);
        }
        else if (strcmp(k, "pixscale_ld") == 0 || strcmp(k, "pixscale") == 0)
        {
            nd->fpcam.pixscale_ld = strtof(v, NULL);
        }
        else
        {
            return -1;
        }
        return 0;
    }

    return -1;
}

int cacaosim_bench_parse_str(
    const char           *text,
    cacaosim_bench_cfg_t *cfg,
    char                 *err_buf,
    size_t                err_cap)
{
    if (text == NULL || cfg == NULL)
    {
        set_err(err_buf, err_cap, "NULL configuration pointer");
        return -1;
    }
    cacaosim_bench_cfg_defaults(cfg);

    enum
    {
        SEC_NONE  = 0,
        SEC_BENCH = 1,
        SEC_NODE  = 2
    } cur_sec = SEC_NONE;

    cacaosim_node_cfg_t *cur_nd  = NULL;
    uint32_t             line_no = 0;
    const char          *p       = text;

    while (*p != '\0')
    {
        const char *nl  = strchr(p, '\n');
        size_t      len = (nl != NULL) ? (size_t) (nl - p) : strlen(p);
        char        line[512];

        line_no++;
        if (len >= sizeof(line))
        {
            set_err(err_buf, err_cap, "Line %u exceeds %zu chars", line_no, sizeof(line) - 1u);
            return -1;
        }
        memcpy(line, p, len);
        line[len] = '\0';
        p         = (nl != NULL) ? (nl + 1) : (p + len);

        strip_comment(line);
        char *s = trim_inplace(line);
        if (*s == '\0')
        {
            continue;
        }

        if (s[0] == '[')
        {
            char *rb = strchr(s, ']');
            if (rb == NULL)
            {
                set_err(err_buf, err_cap, "Line %u: unclosed section header", line_no);
                return -1;
            }
            *rb        = '\0';
            char *sec  = trim_inplace(s + 1);
            if (strcasecmp(sec, "bench") == 0)
            {
                cur_sec = SEC_BENCH;
                cur_nd  = NULL;
                continue;
            }

            char *dot = strchr(sec, '.');
            if (dot == NULL || dot[1] == '\0')
            {
                set_err(err_buf, err_cap, "Line %u: expected [<type>.<id>], got '[%s]'", line_no,
                        sec);
                return -1;
            }
            *dot           = '\0';
            char *type_str = trim_inplace(sec);
            char *id_str   = trim_inplace(dot + 1);

            if (cfg->n_nodes >= CACAOSIM_BENCH_MAX_NODES)
            {
                set_err(err_buf, err_cap, "Line %u: exceeded max nodes (%d)", line_no,
                        CACAOSIM_BENCH_MAX_NODES);
                return -1;
            }

            cacaosim_node_kind_t kind;
            int                  is_ttm = 0;
            if (strcasecmp(type_str, "dm") == 0)
            {
                kind = CACAOSIM_NODE_DM;
            }
            else if (strcasecmp(type_str, "ttm") == 0)
            {
                kind   = CACAOSIM_NODE_DM;
                is_ttm = 1;
            }
            else if (strcasecmp(type_str, "splitter") == 0)
            {
                kind = CACAOSIM_NODE_SPLITTER;
            }
            else if (strcasecmp(type_str, "coro") == 0)
            {
                kind = CACAOSIM_NODE_CORO;
            }
            else if (strcasecmp(type_str, "pyrwfs") == 0)
            {
                kind = CACAOSIM_NODE_PYRWFS;
            }
            else if (strcasecmp(type_str, "shwfs") == 0)
            {
                kind = CACAOSIM_NODE_SHWFS;
            }
            else if (strcasecmp(type_str, "zwfs") == 0)
            {
                kind = CACAOSIM_NODE_ZWFS;
            }
            else if (strcasecmp(type_str, "fpcam") == 0)
            {
                kind = CACAOSIM_NODE_FPCAM;
            }
            else
            {
                set_err(err_buf, err_cap, "Line %u: unknown node type '%s'", line_no, type_str);
                return -1;
            }

            for (uint32_t i = 0; i < cfg->n_nodes; i++)
            {
                if (strcmp(cfg->nodes[i].id, id_str) == 0)
                {
                    set_err(err_buf, err_cap, "Line %u: duplicate node id '%s'", line_no, id_str);
                    return -1;
                }
            }

            cur_nd = &cfg->nodes[cfg->n_nodes];
            init_node_defaults(cur_nd, kind, id_str, is_ttm, cfg->n_nodes);
            cfg->n_nodes++;
            cur_sec = SEC_NODE;
            continue;
        }

        char *eq = strchr(s, '=');
        if (eq == NULL)
        {
            set_err(err_buf, err_cap, "Line %u: expected 'key = value'", line_no);
            return -1;
        }
        *eq     = '\0';
        char *k = trim_inplace(s);
        char *v = trim_inplace(eq + 1);

        if (cur_sec == SEC_BENCH)
        {
            if (apply_bench_kv(cfg, k, v) != 0)
            {
                set_err(err_buf, err_cap, "Line %u: unknown [bench] key '%s'", line_no, k);
                return -1;
            }
        }
        else if (cur_sec == SEC_NODE && cur_nd != NULL)
        {
            if (apply_node_kv(cur_nd, k, v) != 0)
            {
                set_err(err_buf, err_cap, "Line %u: unknown key '%s' in node '%s'", line_no, k,
                        cur_nd->id);
                return -1;
            }
        }
        else
        {
            set_err(err_buf, err_cap, "Line %u: key '%s' outside any section", line_no, k);
            return -1;
        }
    }

    if (cfg->nwf < 8u)
    {
        set_err(err_buf, err_cap, "Invalid [bench] nwf = %u (must be >= 8)", cfg->nwf);
        return -1;
    }
    if (cfg->n_nodes == 0u)
    {
        set_err(err_buf, err_cap, "Configuration defines zero optical nodes");
        return -1;
    }
    return 0;
}

int cacaosim_bench_load_file(
    const char           *path,
    cacaosim_bench_cfg_t *cfg,
    char                 *err_buf,
    size_t                err_cap)
{
    if (path == NULL || path[0] == '\0')
    {
        set_err(err_buf, err_cap, "Empty .bench file path");
        return -1;
    }
    FILE *fp = fopen(path, "rb");
    if (fp == NULL)
    {
        set_err(err_buf, err_cap, "Cannot open '%s'", path);
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        set_err(err_buf, err_cap, "fseek failed on '%s'", path);
        return -1;
    }
    long fsz = ftell(fp);
    if (fsz < 0 || fsz > 1024L * 1024L)
    {
        fclose(fp);
        set_err(err_buf, err_cap, "Invalid .bench file size (%ld bytes)", fsz);
        return -1;
    }
    rewind(fp);

    char *buf = (char *) malloc((size_t) fsz + 1u);
    if (buf == NULL)
    {
        fclose(fp);
        set_err(err_buf, err_cap, "Out of memory reading '%s'", path);
        return -1;
    }
    size_t nread = fread(buf, 1u, (size_t) fsz, fp);
    fclose(fp);
    buf[nread] = '\0';

    int rc = cacaosim_bench_parse_str(buf, cfg, err_buf, err_cap);
    free(buf);
    return rc;
}
