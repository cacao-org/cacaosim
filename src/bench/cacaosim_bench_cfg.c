// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_cfg.c
 * @brief   INI/TOML-style parser for declarative multi-stage .bench optical configurations
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
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
    if (strcasecmp(v, "custom_mft") == 0 || strcasecmp(v, "custom") == 0)
    {
        return CACAOSIM_CORO_CUSTOM_MFT;
    }
    if (strcasecmp(v, "custom_fft") == 0)
    {
        return CACAOSIM_CORO_CUSTOM_FFT;
    }
    return atoi(v);
}

static int parse_lowfs_mode(const char *v)
{
    if (strcasecmp(v, "fpm_refl") == 0 || strcasecmp(v, "fpm") == 0)
    {
        return CACAOSIM_LOWFS_FPM_REFL;
    }
    if (strcasecmp(v, "lyot_refl") == 0 || strcasecmp(v, "lyot") == 0)
    {
        return CACAOSIM_LOWFS_LYOT_REFL;
    }
    return atoi(v) ? CACAOSIM_LOWFS_LYOT_REFL : CACAOSIM_LOWFS_FPM_REFL;
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
        nd->coro.lowfs_mode      = CACAOSIM_LOWFS_FPM_REFL;
        nd->coro.lowfs_pup_diam  = 1.0f;
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
        nd->coro.fpm_pixscale_ld = 8.0f;
        nd->coro.bw_frac         = 0.0f;
        nd->coro.nlambda         = 1;
        nd->coro.mask_pha_unit   = 0;
    }
    else if (kind == CACAOSIM_NODE_WFE)
    {
        snprintf(nd->in_wfe, sizeof(nd->in_wfe), "%s_wfe", id);
        nd->tput       = 0.98f;
        nd->wfe_rms_nm = 50.0f;
        nd->wfe_alpha  = 1.6f;
        nd->wfe_gain   = 1.0f;
        nd->wfe_seed   = seed_base;
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

void cacaosim_bench_node_defaults(
    cacaosim_node_cfg_t *nd,
    cacaosim_node_kind_t kind,
    const char          *id,
    int                  is_ttm,
    uint32_t             idx)
{
    if (nd == NULL)
    {
        return;
    }
    init_node_defaults(nd, kind, (id != NULL) ? id : "node", is_ttm, idx);
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
        else if (strcmp(k, "in_pup_amp") == 0)
        {
            copy_str(nd->in_pup_amp, sizeof(nd->in_pup_amp), v);
        }
        else if (strcmp(k, "in_pup_pha") == 0)
        {
            copy_str(nd->in_pup_pha, sizeof(nd->in_pup_pha), v);
        }
        else if (strcmp(k, "in_fpm_amp") == 0)
        {
            copy_str(nd->in_fpm_amp, sizeof(nd->in_fpm_amp), v);
        }
        else if (strcmp(k, "in_fpm_pha") == 0)
        {
            copy_str(nd->in_fpm_pha, sizeof(nd->in_fpm_pha), v);
        }
        else if (strcmp(k, "in_lyot") == 0 || strcmp(k, "in_lyot_amp") == 0)
        {
            copy_str(nd->in_lyot, sizeof(nd->in_lyot), v);
        }
        else if (strcmp(k, "in_lyot_pha") == 0)
        {
            copy_str(nd->in_lyot_pha, sizeof(nd->in_lyot_pha), v);
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
        else if (strcmp(k, "bw_frac") == 0)
        {
            nd->coro.bw_frac = strtof(v, NULL);
            nd->bw_frac      = nd->coro.bw_frac;
        }
        else if (strcmp(k, "nlambda") == 0)
        {
            nd->coro.nlambda = atoi(v);
        }
        else if (strcmp(k, "fpm_pixscale") == 0 || strcmp(k, "fpm_pixscale_ld") == 0)
        {
            nd->coro.fpm_pixscale_ld = strtof(v, NULL);
        }
        else if (strcmp(k, "pha_unit") == 0 || strcmp(k, "mask_pha_unit") == 0)
        {
            if (strcasecmp(v, "opd") == 0 || strcasecmp(v, "um") == 0)
            {
                nd->coro.mask_pha_unit = 1;
            }
            else if (strcasecmp(v, "rad") == 0 || strcasecmp(v, "phase") == 0)
            {
                nd->coro.mask_pha_unit = 0;
            }
            else
            {
                nd->coro.mask_pha_unit = atoi(v);
            }
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
        else if (strcmp(k, "lowfs_mode") == 0)
        {
            nd->coro.lowfs_mode = parse_lowfs_mode(v);
        }
        else if (strcmp(k, "lowfs_pup_diam") == 0 || strcmp(k, "lowfs_pup") == 0)
        {
            nd->coro.lowfs_pup_diam = strtof(v, NULL);
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

    if (nd->kind == CACAOSIM_NODE_WFE)
    {
        if (strcmp(k, "in_wfe") == 0 || strcmp(k, "stream") == 0)
        {
            copy_str(nd->in_wfe, sizeof(nd->in_wfe), v);
        }
        else if (strcmp(k, "rms_nm") == 0)
        {
            nd->wfe_rms_nm = strtof(v, NULL);
        }
        else if (strcmp(k, "rms_m") == 0)
        {
            nd->wfe_rms_nm = strtof(v, NULL) * 1.0e9f;
        }
        else if (strcmp(k, "psd_alpha") == 0 || strcmp(k, "alpha") == 0)
        {
            nd->wfe_alpha = strtof(v, NULL);
        }
        else if (strcmp(k, "wfe_gain") == 0 || strcmp(k, "gain") == 0)
        {
            nd->wfe_gain = strtof(v, NULL);
        }
        else if (strcmp(k, "wfe_seed") == 0 || strcmp(k, "seed") == 0)
        {
            nd->wfe_seed = (uint64_t) strtoull(v, NULL, 10);
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
            else if (strcasecmp(type_str, "wfe") == 0 || strcasecmp(type_str, "static_wfe") == 0)
            {
                kind = CACAOSIM_NODE_WFE;
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

typedef struct
{
    char  *buf;
    size_t cap;
    size_t len;
} cfg_writer_t;

static void cw_printf(cfg_writer_t *w, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

static void cw_printf(cfg_writer_t *w, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char   *dst = (w->buf != NULL && w->len < w->cap) ? (w->buf + w->len) : NULL;
    size_t  rem = (w->buf != NULL && w->len < w->cap) ? (w->cap - w->len) : 0u;
    int     n   = vsnprintf(dst, rem, fmt, ap);
    va_end(ap);
    if (n > 0)
    {
        w->len += (size_t) n;
    }
}

static const char *split_mode_str(int32_t m)
{
    switch (m)
    {
    case CACAOSIM_SPLIT_GRAY:
        return "gray";
    case CACAOSIM_SPLIT_DICHROIC_LP:
        return "dichroic_lp";
    case CACAOSIM_SPLIT_DICHROIC_SP:
        return "dichroic_sp";
    case CACAOSIM_SPLIT_DICHROIC_BP:
        return "dichroic_bp";
    default:
        return "dichroic_lp";
    }
}

static const char *coro_mode_str(int32_t m)
{
    switch (m)
    {
    case CACAOSIM_CORO_LYOT_SPOT:
        return "lyot_spot";
    case CACAOSIM_CORO_PHASE_DIMPLE:
        return "phase_dimple";
    case CACAOSIM_CORO_VORTEX:
        return "vortex";
    case CACAOSIM_CORO_4QPM:
        return "4qpm";
    case CACAOSIM_CORO_CUSTOM_MFT:
        return "custom_mft";
    case CACAOSIM_CORO_CUSTOM_FFT:
        return "custom_fft";
    default:
        return "lyot_spot";
    }
}

static const char *dyn_mode_str(int32_t m)
{
    switch (m)
    {
    case 0:
        return "instant";
    case 1:
        return "rc";
    case 2:
        return "osc";
    default:
        return "rc";
    }
}

static void cw_write_reg(cfg_writer_t *w, const cacaosim_reg_t *reg)
{
    if (fabsf(reg->shift_x) > 1e-6f)
    {
        cw_printf(w, "shift_x       = %.4g\n", reg->shift_x);
    }
    if (fabsf(reg->shift_y) > 1e-6f)
    {
        cw_printf(w, "shift_y       = %.4g\n", reg->shift_y);
    }
    if (fabsf(reg->rot_deg) > 1e-6f)
    {
        cw_printf(w, "rot_deg       = %.4g\n", reg->rot_deg);
    }
    if (reg->scale > 0.0f && fabsf(reg->scale - 1.0f) > 1e-5f)
    {
        cw_printf(w, "mag_scale     = %.5g\n", reg->scale);
    }
}

static void cw_write_det(
    cfg_writer_t             *w,
    const cacaosim_det_cfg_t *det,
    int32_t                   use_beam_flux)
{
    if (!use_beam_flux)
    {
        cw_printf(w, "flux_ph       = %.6g\n", det->flux_ph);
    }
    cw_printf(w, "ron_e         = %.4g\n", det->ron_e);
    if (fabsf(det->dark_e) > 1e-6f)
    {
        cw_printf(w, "dark_e        = %.4g\n", det->dark_e);
    }
    if (fabsf(det->gain_eadu - 1.0f) > 1e-5f)
    {
        cw_printf(w, "gain_eadu     = %.4g\n", det->gain_eadu);
    }
    if (fabsf(det->bias_adu - 100.0f) > 1e-4f)
    {
        cw_printf(w, "bias_adu      = %.5g\n", det->bias_adu);
    }
    if (fabsf(det->sat_adu - 65535.0f) > 1e-2f)
    {
        cw_printf(w, "sat_adu       = %.6g\n", det->sat_adu);
    }
    if (det->shutter == 0)
    {
        cw_printf(w, "shutter       = 0\n");
    }
    cw_printf(w, "photon_noise  = %d\n", det->photon_noise ? 1 : 0);
    cw_printf(w, "out_dtype     = \"%s\"\n",
              (det->out_dtype == CACAOSIM_DTYPE_F32) ? "float32" : "uint16");
}

int cacaosim_bench_format_str(
    const cacaosim_bench_cfg_t *cfg,
    char                       *buf,
    size_t                      cap)
{
    if (cfg == NULL)
    {
        return -1;
    }

    cfg_writer_t w = { .buf = buf, .cap = cap, .len = 0u };

    cw_printf(&w,
              "# Declarative cacaosim Multi-Stage Optical Bench Configuration (.bench)\n"
              "# Interactive builder/status GUI: cacaosim-CTRL <file.bench>\n"
              "# Deploy & run:                   cacaosim-bench start --config <file.bench>\n\n");

    cw_printf(&w, "[bench]\n");
    cw_printf(&w, "name       = \"%s\"\n", cfg->name[0] ? cfg->name : "ao_bench");
    cw_printf(&w, "nwf        = %u\n", cfg->nwf);
    cw_printf(&w, "pup_diam   = %.6g\n", cfg->pup_diam);
    cw_printf(&w, "cobs       = %.4g\n", cfg->cobs);
    cw_printf(&w, "nspider    = %d\n", cfg->nspider);
    cw_printf(&w, "spider_w   = %.4g\n", cfg->spider_w);
    cw_printf(&w, "spider_deg = %.4g\n", cfg->spider_deg);
    cw_printf(&w, "tel_diam_m = %.4g\n", cfg->tel_diam_m);
    cw_printf(&w, "star_mag   = %.4g\n", cfg->star_mag);
    cw_printf(&w, "star_teff  = %.5g\n", cfg->star_teff);
    cw_printf(&w, "tput_tel   = %.4g\n", cfg->tput_tel);
    cw_printf(&w, "master_hz  = %.6g\n", cfg->master_hz);
    cw_printf(&w, "in_wf      = \"%s\"\n", cfg->in_wf);
    if (cfg->in_amp[0] != '\0')
    {
        cw_printf(&w, "in_amp     = \"%s\"\n", cfg->in_amp);
    }
    cw_printf(&w, "in_pup     = \"%s\"\n", cfg->in_pup);
    if (cfg->in_ncpa[0] != '\0')
    {
        cw_printf(&w, "in_ncpa    = \"%s\"\n", cfg->in_ncpa);
    }
    cw_printf(&w, "out_beam   = \"%s\"\n", cfg->out_beam[0] ? cfg->out_beam : "b_pup");
    if (cfg->wf_unit != CACAOSIM_WFUNIT_PHASE_RAD)
    {
        cw_printf(&w, "wf_unit    = \"um\"\n");
    }
    if (fabsf(cfg->ref_lambda - 0.50f) > 1e-5f)
    {
        cw_printf(&w, "ref_lambda = %.4g\n", cfg->ref_lambda);
    }
    if (fabsf(cfg->turb_gain - 1.0f) > 1e-5f)
    {
        cw_printf(&w, "turb_gain  = %.4g\n", cfg->turb_gain);
    }
    if (cfg->backend == CACAOSIM_BACKEND_CPU)
    {
        cw_printf(&w, "backend    = \"cpu\"\n");
    }
    else if (cfg->backend == CACAOSIM_BACKEND_CUDA)
    {
        cw_printf(&w, "backend    = \"cuda\"\n");
    }

    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *nd = &cfg->nodes[i];
        cw_printf(&w, "\n");

        if (nd->kind == CACAOSIM_NODE_DM)
        {
            int is_ttm = (nd->dm.dm_mode == 1);
            cw_printf(&w, "[%s.%s]\n", is_ttm ? "ttm" : "dm", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out           = \"%s\"\n", nd->out_beam);
            cw_printf(&w, "in_cmd        = \"%s\"\n", nd->in_cmd);
            if (nd->out_opd[0] != '\0')
            {
                cw_printf(&w, "out_opd       = \"%s\"\n", nd->out_opd);
            }
            if (nd->in_flat[0] != '\0')
            {
                cw_printf(&w, "in_flat       = \"%s\"\n", nd->in_flat);
            }
            if (!is_ttm)
            {
                cw_printf(&w, "nact_x        = %u\n", nd->dm.nact_x);
                cw_printf(&w, "nact_y        = %u\n", nd->dm.nact_y);
                cw_printf(&w, "if_coupling   = %.4g\n", nd->dm.if_coupling);
                if (nd->dm.dm_diam > 0.0f)
                {
                    cw_printf(&w, "dm_diam       = %.5g\n", nd->dm.dm_diam);
                }
            }
            cw_printf(&w, "tput          = %.4g\n", nd->tput);
            if (fabsf(nd->dm.refl_scale - 2.0f) > 1e-5f)
            {
                cw_printf(&w, "refl_scale    = %.4g\n", nd->dm.refl_scale);
            }
            cw_printf(&w, "max_stroke    = %.4g\n", nd->dm.max_stroke);
            cw_printf(&w, "dac_bits      = %d\n", nd->dm.dac_bits);
            if (fabsf(nd->dm.nonlin_quad) > 1e-6f)
            {
                cw_printf(&w, "nonlin_quad   = %.4g\n", nd->dm.nonlin_quad);
            }
            cw_printf(&w, "dyn_mode      = \"%s\"\n", dyn_mode_str(nd->dm.dyn_mode));
            if (nd->dm.dyn_mode == 1)
            {
                cw_printf(&w, "tau_us        = %.5g\n", nd->dm.tau_sec * 1.0e6f);
            }
            else if (nd->dm.dyn_mode == 2)
            {
                cw_printf(&w, "f0_hz         = %.5g\n", nd->dm.f0_hz);
                cw_printf(&w, "zeta          = %.4g\n", nd->dm.zeta);
            }
            cw_printf(&w, "delay_frames  = %.4g\n", nd->dm.delay_frames);
            cw_write_reg(&w, &nd->reg);
        }
        else if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            cw_printf(&w, "[splitter.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            if (nd->out_refl[0] != '\0')
            {
                cw_printf(&w, "out_refl      = \"%s\"\n", nd->out_refl);
            }
            if (nd->out_trans[0] != '\0')
            {
                cw_printf(&w, "out_trans     = \"%s\"\n", nd->out_trans);
            }
            cw_printf(&w, "mode          = \"%s\"\n", split_mode_str(nd->split.mode));
            if (nd->split.mode == CACAOSIM_SPLIT_GRAY)
            {
                cw_printf(&w, "r_gray        = %.4g\n", nd->split.r_gray);
                cw_printf(&w, "t_gray        = %.4g\n", nd->split.t_gray);
            }
            else
            {
                cw_printf(&w, "lambda_cut    = %.4g\n", nd->split.lambda_cut);
                if (nd->split.mode == CACAOSIM_SPLIT_DICHROIC_BP)
                {
                    cw_printf(&w, "band_width    = %.4g\n", nd->split.band_width);
                }
                if (fabsf(nd->split.roll_um - 0.03f) > 1e-5f)
                {
                    cw_printf(&w, "roll_um       = %.4g\n", nd->split.roll_um);
                }
                cw_printf(&w, "eff_refl      = %.4g\n", nd->split.eff_refl);
                cw_printf(&w, "eff_trans     = %.4g\n", nd->split.eff_trans);
            }
            if (fabsf(nd->reg_refl.shift_x) > 1e-6f)
            {
                cw_printf(&w, "refl_dx       = %.4g\n", nd->reg_refl.shift_x);
            }
            if (fabsf(nd->reg_refl.shift_y) > 1e-6f)
            {
                cw_printf(&w, "refl_dy       = %.4g\n", nd->reg_refl.shift_y);
            }
            if (fabsf(nd->reg_refl.rot_deg) > 1e-6f)
            {
                cw_printf(&w, "refl_rot      = %.4g\n", nd->reg_refl.rot_deg);
            }
            if (nd->reg_refl.scale > 0.0f && fabsf(nd->reg_refl.scale - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "refl_scale    = %.5g\n", nd->reg_refl.scale);
            }
            if (fabsf(nd->reg_trans.shift_x) > 1e-6f)
            {
                cw_printf(&w, "trans_dx      = %.4g\n", nd->reg_trans.shift_x);
            }
            if (fabsf(nd->reg_trans.shift_y) > 1e-6f)
            {
                cw_printf(&w, "trans_dy      = %.4g\n", nd->reg_trans.shift_y);
            }
            if (fabsf(nd->reg_trans.rot_deg) > 1e-6f)
            {
                cw_printf(&w, "trans_rot     = %.4g\n", nd->reg_trans.rot_deg);
            }
            if (nd->reg_trans.scale > 0.0f && fabsf(nd->reg_trans.scale - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "trans_scale   = %.5g\n", nd->reg_trans.scale);
            }
            if (nd->ncpa_refl[0] != '\0')
            {
                cw_printf(&w, "ncpa_refl     = \"%s\"\n", nd->ncpa_refl);
            }
            if (nd->ncpa_trans[0] != '\0')
            {
                cw_printf(&w, "ncpa_trans    = \"%s\"\n", nd->ncpa_trans);
            }
            if (nd->pup_refl[0] != '\0')
            {
                cw_printf(&w, "pup_refl      = \"%s\"\n", nd->pup_refl);
            }
            if (nd->pup_trans[0] != '\0')
            {
                cw_printf(&w, "pup_trans     = \"%s\"\n", nd->pup_trans);
            }
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            cw_printf(&w, "[coro.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out           = \"%s\"\n", nd->out_beam);
            if (nd->out_lowfs[0] != '\0')
            {
                cw_printf(&w, "out_lowfs     = \"%s\"\n", nd->out_lowfs);
            }
            if (nd->in_pup_amp[0] != '\0')
            {
                cw_printf(&w, "in_pup_amp    = \"%s\"\n", nd->in_pup_amp);
            }
            if (nd->in_pup_pha[0] != '\0')
            {
                cw_printf(&w, "in_pup_pha    = \"%s\"\n", nd->in_pup_pha);
            }
            if (nd->in_fpm_amp[0] != '\0')
            {
                cw_printf(&w, "in_fpm_amp    = \"%s\"\n", nd->in_fpm_amp);
            }
            if (nd->in_fpm_pha[0] != '\0')
            {
                cw_printf(&w, "in_fpm_pha    = \"%s\"\n", nd->in_fpm_pha);
            }
            if (nd->in_lyot[0] != '\0')
            {
                cw_printf(&w, "in_lyot       = \"%s\"\n", nd->in_lyot);
            }
            if (nd->in_lyot_pha[0] != '\0')
            {
                cw_printf(&w, "in_lyot_pha   = \"%s\"\n", nd->in_lyot_pha);
            }
            cw_printf(&w, "mode          = \"%s\"\n", coro_mode_str(nd->coro.mode));
            cw_printf(&w, "lambda_um     = %.4g\n", nd->coro.lambda_um);
            if (nd->coro.bw_frac > 1e-6f)
            {
                cw_printf(&w, "bw_frac       = %.4g\n", nd->coro.bw_frac);
            }
            if (nd->coro.nlambda > 1)
            {
                cw_printf(&w, "nlambda       = %d\n", nd->coro.nlambda);
            }
            if (nd->coro.mask_pha_unit != 0)
            {
                cw_printf(&w, "pha_unit      = \"um\"\n");
            }
            if (nd->coro.mode <= CACAOSIM_CORO_PHASE_DIMPLE)
            {
                cw_printf(&w, "fpm_rad_ld    = %.4g\n", nd->coro.fpm_rad_ld);
                cw_printf(&w, "fpm_amp       = %.4g\n", nd->coro.fpm_amp);
                if (nd->coro.mode == CACAOSIM_CORO_PHASE_DIMPLE)
                {
                    cw_printf(&w, "fpm_phase     = %.5g\n", nd->coro.fpm_phase_rad);
                }
                if (nd->coro.n_mft != 48u)
                {
                    cw_printf(&w, "n_mft         = %u\n", nd->coro.n_mft);
                }
            }
            else if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
            {
                cw_printf(&w, "fpm_pixscale  = %.4g\n", nd->coro.fpm_pixscale_ld);
            }
            else
            {
                if (nd->coro.mode == CACAOSIM_CORO_VORTEX)
                {
                    cw_printf(&w, "vortex_charge = %d\n", nd->coro.vortex_charge);
                }
                if (nd->coro.pad_factor != 4u)
                {
                    cw_printf(&w, "pad_factor    = %u\n", nd->coro.pad_factor);
                }
            }
            cw_printf(&w, "lowfs_refl    = %.4g\n", nd->coro.lowfs_refl);
            if (nd->out_lowfs[0] != '\0' || nd->coro.lowfs_mode != CACAOSIM_LOWFS_FPM_REFL)
            {
                cw_printf(&w, "lowfs_mode    = \"%s\"\n",
                          (nd->coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? "lyot_refl"
                                                                            : "fpm_refl");
            }
            if (nd->coro.lowfs_pup_diam > 0.0f &&
                ((nd->coro.lowfs_mode == CACAOSIM_LOWFS_FPM_REFL && nd->out_lowfs[0] != '\0') ||
                 fabsf(nd->coro.lowfs_pup_diam - 1.0f) > 1e-4f))
            {
                cw_printf(&w, "lowfs_pup_diam = %.4g\n", nd->coro.lowfs_pup_diam);
            }
            cw_printf(&w, "tput          = %.4g\n", nd->tput);
            cw_printf(&w, "lyot_outer    = %.4g\n", nd->coro.lyot_outer);
            cw_printf(&w, "lyot_obs      = %.4g\n", nd->coro.lyot_obs);
            cw_printf(&w, "lyot_nspider  = %d\n", nd->coro.lyot_nspider);
            cw_printf(&w, "lyot_spider_w = %.4g\n", nd->coro.lyot_spider_w);
            if (fabsf(nd->coro.lyot_spider_deg - 45.0f) > 1e-4f)
            {
                cw_printf(&w, "lyot_spider_deg = %.4g\n", nd->coro.lyot_spider_deg);
            }
            cw_write_reg(&w, &nd->reg);
        }
        else if (nd->kind == CACAOSIM_NODE_WFE)
        {
            cw_printf(&w, "[wfe.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out           = \"%s\"\n", nd->out_beam);
            cw_printf(&w, "in_wfe        = \"%s\"\n", nd->in_wfe);
            cw_printf(&w, "rms_nm        = %.4g\n", nd->wfe_rms_nm);
            if (fabsf(nd->wfe_alpha - 1.6f) > 1e-4f)
            {
                cw_printf(&w, "psd_alpha     = %.4g\n", nd->wfe_alpha);
            }
            cw_printf(&w, "wfe_gain      = %.4g\n", nd->wfe_gain);
            cw_printf(&w, "wfe_seed      = %llu\n", (unsigned long long) nd->wfe_seed);
            cw_printf(&w, "tput          = %.4g\n", nd->tput);
            cw_write_reg(&w, &nd->reg);
        }
        else if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            cw_printf(&w, "[pyrwfs.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out_cam       = \"%s\"\n", nd->out_cam);
            if (nd->in_ncpa[0] != '\0')
            {
                cw_printf(&w, "in_ncpa       = \"%s\"\n", nd->in_ncpa);
            }
            cw_printf(&w, "ncam          = %u\n", nd->pyrwfs.ncam);
            if (nd->pyrwfs.pad_factor != 2u)
            {
                cw_printf(&w, "pad_factor    = %u\n", nd->pyrwfs.pad_factor);
            }
            cw_printf(&w, "lambda_um     = %.4g\n", nd->pyrwfs.lambda_um);
            cw_printf(&w, "mod_rad       = %.4g\n", nd->pyrwfs.mod_rad_ld);
            cw_printf(&w, "mod_npts      = %d\n", nd->pyrwfs.mod_npts);
            if (fabsf(nd->pyrwfs.pup_sep - 0.50f) > 1e-5f)
            {
                cw_printf(&w, "pup_sep       = %.4g\n", nd->pyrwfs.pup_sep);
            }
            if (fabsf(nd->pyrwfs.roof_ld) > 1e-5f)
            {
                cw_printf(&w, "roof_ld       = %.4g\n", nd->pyrwfs.roof_ld);
            }
            if (fabsf(nd->pyrwfs.fstop_frac - 0.45f) > 1e-5f)
            {
                cw_printf(&w, "fstop_frac    = %.4g\n", nd->pyrwfs.fstop_frac);
            }
            cw_printf(&w, "bw_frac       = %.4g\n", nd->bw_frac);
            cw_printf(&w, "qe            = %.4g\n", nd->qe);
            cw_printf(&w, "use_beam_flux = %d\n", nd->use_beam_flux ? 1 : 0);
            if (nd->rate_hz > 0.0f)
            {
                cw_printf(&w, "rate_hz       = %.6g\n", nd->rate_hz);
            }
            else
            {
                cw_printf(&w, "navg          = %u\n", nd->navg);
            }
            if (fabsf(nd->tput - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "tput          = %.4g\n", nd->tput);
            }
            cw_write_det(&w, &nd->pyrwfs.det, nd->use_beam_flux);
            cw_write_reg(&w, &nd->reg);
        }
        else if (nd->kind == CACAOSIM_NODE_SHWFS)
        {
            cw_printf(&w, "[shwfs.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out_cam       = \"%s\"\n", nd->out_cam);
            if (nd->in_ncpa[0] != '\0')
            {
                cw_printf(&w, "in_ncpa       = \"%s\"\n", nd->in_ncpa);
            }
            cw_printf(&w, "n_sub         = %u\n", nd->shwfs.n_sub);
            cw_printf(&w, "pix_per_sub   = %u\n", nd->shwfs.pix_per_sub);
            if (nd->shwfs.pad_factor != 2u)
            {
                cw_printf(&w, "pad_factor    = %u\n", nd->shwfs.pad_factor);
            }
            cw_printf(&w, "lambda_um     = %.4g\n", nd->shwfs.lambda_um);
            cw_printf(&w, "bw_frac       = %.4g\n", nd->bw_frac);
            cw_printf(&w, "qe            = %.4g\n", nd->qe);
            cw_printf(&w, "use_beam_flux = %d\n", nd->use_beam_flux ? 1 : 0);
            if (nd->rate_hz > 0.0f)
            {
                cw_printf(&w, "rate_hz       = %.6g\n", nd->rate_hz);
            }
            else
            {
                cw_printf(&w, "navg          = %u\n", nd->navg);
            }
            if (fabsf(nd->tput - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "tput          = %.4g\n", nd->tput);
            }
            cw_write_det(&w, &nd->shwfs.det, nd->use_beam_flux);
            cw_write_reg(&w, &nd->reg);
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            cw_printf(&w, "[zwfs.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out_cam       = \"%s\"\n", nd->out_cam);
            if (nd->in_ncpa[0] != '\0')
            {
                cw_printf(&w, "in_ncpa       = \"%s\"\n", nd->in_ncpa);
            }
            cw_printf(&w, "ncam          = %u\n", nd->zwfs.ncam);
            if (nd->zwfs.n_mft != 32u)
            {
                cw_printf(&w, "n_mft         = %u\n", nd->zwfs.n_mft);
            }
            cw_printf(&w, "lambda_um     = %.4g\n", nd->zwfs.lambda_um);
            cw_printf(&w, "dimple_ld     = %.4g\n", nd->zwfs.dimple_ld);
            if (fabsf(nd->zwfs.dimple_rad - 0.5f * CACAOSIM_PI_F) > 1e-4f)
            {
                cw_printf(&w, "dimple_rad    = %.5g\n", nd->zwfs.dimple_rad);
            }
            if (fabsf(nd->zwfs.dimple_amp - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "dimple_amp    = %.4g\n", nd->zwfs.dimple_amp);
            }
            cw_printf(&w, "bw_frac       = %.4g\n", nd->bw_frac);
            cw_printf(&w, "qe            = %.4g\n", nd->qe);
            cw_printf(&w, "use_beam_flux = %d\n", nd->use_beam_flux ? 1 : 0);
            if (nd->rate_hz > 0.0f)
            {
                cw_printf(&w, "rate_hz       = %.6g\n", nd->rate_hz);
            }
            else
            {
                cw_printf(&w, "navg          = %u\n", nd->navg);
            }
            if (fabsf(nd->tput - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "tput          = %.4g\n", nd->tput);
            }
            cw_write_det(&w, &nd->zwfs.det, nd->use_beam_flux);
            cw_write_reg(&w, &nd->reg);
        }
        else if (nd->kind == CACAOSIM_NODE_FPCAM)
        {
            cw_printf(&w, "[fpcam.%s]\n", nd->id);
            cw_printf(&w, "in            = \"%s\"\n", nd->in_beam);
            cw_printf(&w, "out_cam       = \"%s\"\n", nd->out_cam);
            if (nd->out_long[0] != '\0')
            {
                cw_printf(&w, "out_long      = \"%s\"\n", nd->out_long);
            }
            if (nd->in_ncpa[0] != '\0')
            {
                cw_printf(&w, "in_ncpa       = \"%s\"\n", nd->in_ncpa);
            }
            cw_printf(&w, "ncam          = %u\n", nd->fpcam.ncam);
            cw_printf(&w, "prop_mode     = \"%s\"\n",
                      (nd->fpcam.prop_mode == CACAOSIM_FPCAM_PROP_FFT) ? "fft" : "mft");
            if (nd->fpcam.prop_mode == CACAOSIM_FPCAM_PROP_FFT && nd->fpcam.pad_factor != 2u)
            {
                cw_printf(&w, "pad_factor    = %u\n", nd->fpcam.pad_factor);
            }
            cw_printf(&w, "lambda_um     = %.4g\n", nd->fpcam.lambda_um);
            if (nd->fpcam.nlambda > 1)
            {
                cw_printf(&w, "nlambda       = %d\n", nd->fpcam.nlambda);
            }
            cw_printf(&w, "pixscale_ld   = %.4g\n", nd->fpcam.pixscale_ld);
            cw_printf(&w, "bw_frac       = %.4g\n", nd->bw_frac);
            cw_printf(&w, "qe            = %.4g\n", nd->qe);
            cw_printf(&w, "use_beam_flux = %d\n", nd->use_beam_flux ? 1 : 0);
            if (nd->rate_hz > 0.0f)
            {
                cw_printf(&w, "rate_hz       = %.6g\n", nd->rate_hz);
            }
            else
            {
                cw_printf(&w, "navg          = %u\n", nd->navg);
            }
            if (fabsf(nd->tput - 1.0f) > 1e-5f)
            {
                cw_printf(&w, "tput          = %.4g\n", nd->tput);
            }
            cw_write_det(&w, &nd->fpcam.det, nd->use_beam_flux);
            cw_write_reg(&w, &nd->reg);
        }
    }

    return (int) w.len;
}

int cacaosim_bench_save_file(
    const char                 *path,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap)
{
    if (path == NULL || path[0] == '\0' || cfg == NULL)
    {
        set_err(err_buf, err_cap, "Invalid path or configuration pointer");
        return -1;
    }
    int need = cacaosim_bench_format_str(cfg, NULL, 0u);
    if (need <= 0)
    {
        set_err(err_buf, err_cap, "Failed to format .bench configuration");
        return -1;
    }
    char *buf = (char *) malloc((size_t) need + 2u);
    if (buf == NULL)
    {
        set_err(err_buf, err_cap, "Out of memory formatting '%s'", path);
        return -1;
    }
    cacaosim_bench_format_str(cfg, buf, (size_t) need + 2u);

    FILE *fp = fopen(path, "wb");
    if (fp == NULL)
    {
        set_err(err_buf, err_cap, "Cannot write '%s': %s", path, strerror(errno));
        free(buf);
        return -1;
    }
    size_t nw = fwrite(buf, 1u, (size_t) need, fp);
    int    rc = fclose(fp);
    free(buf);
    if (nw != (size_t) need || rc != 0)
    {
        set_err(err_buf, err_cap, "Short write saving '%s'", path);
        return -1;
    }
    return 0;
}
