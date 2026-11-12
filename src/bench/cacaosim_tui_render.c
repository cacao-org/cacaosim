// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_render.c
 * @brief   milk-CTRL chrome, 2D interactive optical schematic, F2:BUILDER, modals, and tooltips
 */

#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bench/cacaosim_tui_internal.h"

/* ── Compact Formatting & Node Metadata Helpers ───────────────────────────── */

static void fmt_flux_compact(float ph, char *out, size_t cap)
{
    if (ph <= 0.0f)
    {
        snprintf(out, cap, "0 ph");
    }
    else if (ph >= 1.0e9f)
    {
        snprintf(out, cap, "%.2fe9", ph * 1.0e-9f);
    }
    else if (ph >= 1.0e6f)
    {
        snprintf(out, cap, "%.2fM", ph * 1.0e-6f);
    }
    else if (ph >= 1.0e3f)
    {
        snprintf(out, cap, "%.1fk", ph * 1.0e-3f);
    }
    else
    {
        snprintf(out, cap, "%.0f", ph);
    }
}

static const char *node_badge(const cacaosim_node_cfg_t *nd)
{
    switch (nd->kind)
    {
    case CACAOSIM_NODE_DM:
        return (nd->dm.dm_mode == 1) ? "TTM" : "DM";
    case CACAOSIM_NODE_SPLITTER:
        return "BS";
    case CACAOSIM_NODE_CORO:
        return "CORO";
    case CACAOSIM_NODE_PYRWFS:
        return "PYR";
    case CACAOSIM_NODE_SHWFS:
        return "SH";
    case CACAOSIM_NODE_ZWFS:
        return "ZWFS";
    case CACAOSIM_NODE_FPCAM:
        return "CAM";
    case CACAOSIM_NODE_WFE:
        return "WFE";
    }
    return "NODE";
}

static cs_tui_rgb_t node_color(const cacaosim_node_cfg_t *nd, const cs_tui_pal_t *pal)
{
    switch (nd->kind)
    {
    case CACAOSIM_NODE_DM:
        return pal->fg_accent;
    case CACAOSIM_NODE_SPLITTER:
        return pal->fg_proc;
    case CACAOSIM_NODE_CORO:
    case CACAOSIM_NODE_WFE:
        return pal->fg_warn;
    case CACAOSIM_NODE_PYRWFS:
    case CACAOSIM_NODE_SHWFS:
    case CACAOSIM_NODE_ZWFS:
        return pal->fg_stream;
    case CACAOSIM_NODE_FPCAM:
        return pal->fg_active;
    }
    return pal->fg_text;
}

static void node_short_spec(
    const cacaosim_bench_cfg_t *cfg,
    const cacaosim_node_cfg_t  *nd,
    char                       *out,
    size_t                      cap)
{
    float mhz = (cfg->master_hz > 0.0f) ? cfg->master_hz : 2000.0f;
    switch (nd->kind)
    {
    case CACAOSIM_NODE_DM:
        if (nd->dm.dm_mode == 1)
        {
            snprintf(out, cap, "2-ax +/-%.0fu", nd->dm.max_stroke);
        }
        else
        {
            snprintf(out, cap, "%ux%u +/-%.0fu", nd->dm.nact_x, nd->dm.nact_y, nd->dm.max_stroke);
        }
        break;
    case CACAOSIM_NODE_SPLITTER:
        if (nd->split.mode == CACAOSIM_SPLIT_GRAY)
        {
            snprintf(out, cap, "%.0f/%.0f gray", nd->split.r_gray * 100.0f,
                     nd->split.t_gray * 100.0f);
        }
        else if (nd->split.mode == CACAOSIM_SPLIT_DICHROIC_BP)
        {
            snprintf(out, cap, "BP %.2fu", nd->split.lambda_cut);
        }
        else
        {
            snprintf(out, cap, "%s %.2fu",
                     (nd->split.mode == CACAOSIM_SPLIT_DICHROIC_SP) ? "SP" : "LP",
                     nd->split.lambda_cut);
        }
        break;
    case CACAOSIM_NODE_CORO:
    {
        int has_cmask = (nd->in_pup_amp[0] != '\0' || nd->in_pup_pha[0] != '\0' ||
                         nd->in_fpm_amp[0] != '\0' || nd->in_fpm_pha[0] != '\0' ||
                         nd->in_lyot[0] != '\0' || nd->in_lyot_pha[0] != '\0');
        if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            if (nd->coro.nlambda > 1)
            {
                snprintf(out, cap, "CustMFT %dch %.2fu", nd->coro.nlambda, nd->coro.lambda_um);
            }
            else
            {
                snprintf(out, cap, "CustMFT %.2fu", nd->coro.lambda_um);
            }
        }
        else if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_FFT)
        {
            if (nd->coro.nlambda > 1)
            {
                snprintf(out, cap, "CustFFT %dch %.2fu", nd->coro.nlambda, nd->coro.lambda_um);
            }
            else
            {
                snprintf(out, cap, "CustFFT %.2fu", nd->coro.lambda_um);
            }
        }
        else if (nd->coro.mode == CACAOSIM_CORO_VORTEX)
        {
            snprintf(out, cap, "Vortex l=%d%s", nd->coro.vortex_charge, has_cmask ? "+M" : "");
        }
        else if (nd->coro.mode == CACAOSIM_CORO_4QPM)
        {
            snprintf(out, cap, "4QPM %.2fu%s", nd->coro.lambda_um, has_cmask ? "+M" : "");
        }
        else if (nd->coro.mode == CACAOSIM_CORO_PHASE_DIMPLE)
        {
            snprintf(out, cap, "Dimple %.1fl/D%s", nd->coro.fpm_rad_ld, has_cmask ? "+M" : "");
        }
        else
        {
            snprintf(out, cap, "Lyot %.1fl/D%s", nd->coro.fpm_rad_ld, has_cmask ? "+M" : "");
        }
        break;
    }
    case CACAOSIM_NODE_PYRWFS:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%upx %.2fu %.0fHz", nd->pyrwfs.ncam, nd->pyrwfs.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_SHWFS:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%ux%u %.2fu %.0fHz", nd->shwfs.n_sub, nd->shwfs.n_sub,
                 nd->shwfs.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_ZWFS:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%upx %.2fu %.0fHz", nd->zwfs.ncam, nd->zwfs.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_FPCAM:
    {
        float hz = (nd->rate_hz > 0.0f) ? nd->rate_hz : (mhz / (float) (nd->navg ? nd->navg : 1u));
        snprintf(out, cap, "%upx %.2fu %.0fHz", nd->fpcam.ncam, nd->fpcam.lambda_um, hz);
        break;
    }
    case CACAOSIM_NODE_WFE:
        snprintf(out, cap, "%.0fnm [OPD m]", nd->wfe_rms_nm * fabsf(nd->wfe_gain));
        break;
    }
}

static int node_focal_scale(
    const cacaosim_bench_cfg_t *cfg,
    const cacaosim_node_cfg_t  *nd,
    float                      *out_ld_pix,
    float                      *out_lam_um)
{
    float pdia = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 1.0f;
    switch (nd->kind)
    {
    case CACAOSIM_NODE_FPCAM:
        *out_lam_um = (nd->fpcam.lambda_um > 0.1f) ? nd->fpcam.lambda_um : 1.65f;
        if (nd->fpcam.prop_mode == CACAOSIM_FPCAM_PROP_FFT)
        {
            float pad   = (nd->fpcam.pad_factor >= 2u) ? (float) nd->fpcam.pad_factor : 2.0f;
            *out_ld_pix = 1.0f / (pdia * pad);
        }
        else
        {
            float px    = (nd->fpcam.pixscale_ld > 0.05f) ? nd->fpcam.pixscale_ld : 3.0f;
            *out_ld_pix = 1.0f / px;
        }
        return 1;
    case CACAOSIM_NODE_CORO:
        *out_lam_um = (nd->coro.lambda_um > 0.1f) ? nd->coro.lambda_um : 1.65f;
        if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            float px    = (nd->coro.fpm_pixscale_ld > 0.05f) ? nd->coro.fpm_pixscale_ld : 8.0f;
            *out_ld_pix = 1.0f / px;
        }
        else if (nd->coro.mode == CACAOSIM_CORO_VORTEX || nd->coro.mode == CACAOSIM_CORO_4QPM ||
                 nd->coro.mode == CACAOSIM_CORO_CUSTOM_FFT)
        {
            float pad   = (nd->coro.pad_factor >= 2u) ? (float) nd->coro.pad_factor : 4.0f;
            *out_ld_pix = 1.0f / (pdia * pad);
        }
        else
        {
            float r_ld  = (nd->coro.fpm_rad_ld > 0.1f) ? nd->coro.fpm_rad_ld : 2.5f;
            float nmft  = (nd->coro.n_mft >= 8u) ? (float) nd->coro.n_mft : 32.0f;
            *out_ld_pix = (2.4f * r_ld) / nmft;
        }
        return 1;
    case CACAOSIM_NODE_SHWFS:
    {
        float nsub  = (nd->shwfs.n_sub >= 1u) ? (float) nd->shwfs.n_sub : 16.0f;
        float pad   = (nd->shwfs.pad_factor >= 1u) ? (float) nd->shwfs.pad_factor : 2.0f;
        *out_lam_um = (nd->shwfs.lambda_um > 0.1f) ? nd->shwfs.lambda_um : 0.75f;
        *out_ld_pix = nsub / pad;
        return 1;
    }
    case CACAOSIM_NODE_PYRWFS:
    {
        float pad   = (nd->pyrwfs.pad_factor >= 2u) ? (float) nd->pyrwfs.pad_factor : 2.0f;
        *out_lam_um = (nd->pyrwfs.lambda_um > 0.1f) ? nd->pyrwfs.lambda_um : 0.75f;
        *out_ld_pix = 1.0f / (pdia * pad);
        return 1;
    }
    case CACAOSIM_NODE_ZWFS:
    {
        float d_ld  = (nd->zwfs.dimple_ld > 0.1f) ? nd->zwfs.dimple_ld : 1.06f;
        float nmft  = (nd->zwfs.n_mft >= 8u) ? (float) nd->zwfs.n_mft : 32.0f;
        *out_lam_um = (nd->zwfs.lambda_um > 0.1f) ? nd->zwfs.lambda_um : 0.85f;
        *out_ld_pix = (1.2f * d_ld) / nmft;
        return 1;
    }
    default:
        return 0;
    }
}

void cs_tui_compute_scales(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    cs_optical_scales_t        *out)
{
    if (out == NULL)
    {
        return;
    }
    memset(out, 0, sizeof(*out));
    if (cfg == NULL)
    {
        return;
    }

    float nwf_f     = (cfg->nwf > 0u) ? (float) cfg->nwf : 128.0f;
    float pdia      = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 1.0f;
    float tel_d     = (cfg->tel_diam_m > 0.05f) ? cfg->tel_diam_m : 8.2f;
    out->d_beam_px  = pdia * nwf_f;
    out->pup_m_pix  = tel_d / out->d_beam_px;
    out->foc_ld_pix = 1.0f / (pdia * 2.0f);
    out->ref_lam_um = (cfg->ref_lambda > 0.1f) ? cfg->ref_lambda : 1.65f;
    snprintf(out->foc_src, sizeof(out->foc_src), "2xFFT");

    int found = 0;
    if (sel_item > 0 && (uint32_t) (sel_item - 1) < cfg->n_nodes)
    {
        const cacaosim_node_cfg_t *nd = &cfg->nodes[sel_item - 1];
        if (node_focal_scale(cfg, nd, &out->foc_ld_pix, &out->ref_lam_um))
        {
            snprintf(out->foc_src, sizeof(out->foc_src), "%s", nd->id);
            found = 1;
        }
    }

    if (!found)
    {
        static const cacaosim_node_kind_t prio[5] = {
            CACAOSIM_NODE_FPCAM, CACAOSIM_NODE_CORO,   CACAOSIM_NODE_SHWFS,
            CACAOSIM_NODE_PYRWFS, CACAOSIM_NODE_ZWFS
        };
        for (int p = 0; p < 5 && !found; p++)
        {
            for (uint32_t i = 0; i < cfg->n_nodes; i++)
            {
                if (cfg->nodes[i].kind == prio[p] &&
                    node_focal_scale(cfg, &cfg->nodes[i], &out->foc_ld_pix, &out->ref_lam_um))
                {
                    snprintf(out->foc_src, sizeof(out->foc_src), "%s", cfg->nodes[i].id);
                    found = 1;
                    break;
                }
            }
        }
    }

    float ld_arcsec     = (out->ref_lam_um * 1.0e-6f / tel_d) * 206264.8062f;
    out->foc_arcsec_pix = out->foc_ld_pix * ld_arcsec;
}

static const cacaosim_bench_node_t *find_analyzed_node(
    const cs_tui_app_t *app,
    const char         *node_id)
{
    if (!app->dag_valid || node_id == NULL)
    {
        return NULL;
    }
    for (uint32_t i = 0; i < app->ctx.n_nodes; i++)
    {
        if (strcmp(app->ctx.nodes[i].cfg.id, node_id) == 0)
        {
            return &app->ctx.nodes[i];
        }
    }
    return NULL;
}

static const cacaosim_spec_t *find_beam_spec(const cs_tui_app_t *app, const char *beam_name)
{
    if (!app->dag_valid || beam_name == NULL || beam_name[0] == '\0')
    {
        return NULL;
    }
    for (uint32_t i = 0; i < app->ctx.n_beams; i++)
    {
        if (strcmp(app->ctx.beams[i].name, beam_name) == 0)
        {
            return &app->ctx.beams[i].spec;
        }
    }
    return NULL;
}

static int node_primary_shm_live(const cs_tui_app_t *app, const cacaosim_node_cfg_t *nd)
{
    const char *s = NULL;
    if (nd->kind == CACAOSIM_NODE_DM)
    {
        s = nd->in_cmd;
    }
    else if (nd->kind == CACAOSIM_NODE_WFE)
    {
        s = nd->in_wfe;
    }
    else if (nd->kind == CACAOSIM_NODE_PYRWFS || nd->kind == CACAOSIM_NODE_SHWFS ||
             nd->kind == CACAOSIM_NODE_ZWFS || nd->kind == CACAOSIM_NODE_FPCAM)
    {
        s = nd->out_cam;
    }
    if (s == NULL || s[0] == '\0')
    {
        return -1;
    }
    int idx = cs_shm_find_stream(&app->shm, s);
    if (idx < 0 || !app->shm.streams[idx].exists)
    {
        return 0;
    }
    return app->shm.streams[idx].alive ? 2 : 1;
}

/* ── Interactive 2D Optical Schematic Canvas ──────────────────────────────── */

typedef struct
{
    cs_tui_app_t       *app;
    const cs_tui_pal_t *pal;
    int                 vx;
    int                 vy;
    int                 vw;
    int                 vh;
    int                 wide_spec;
    int                 measure_only;
    int                 sel_found;
    int                 sel_lx;
    int                 sel_ly;
    int                 sel_bw;
    uint8_t             visited[CACAOSIM_BENCH_MAX_NODES];
    int                 next_row;
} dag_draw_t;

static void dc_put(
    dag_draw_t  *d,
    int          lx,
    int          ly,
    const char  *str,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs)
{
    if (d->measure_only)
    {
        return;
    }
    int sy = d->vy + (ly - d->app->schem_scroll_y);
    if (sy < d->vy || sy >= d->vy + d->vh)
    {
        return;
    }
    int         col = 0;
    const char *p   = str;
    while (*p != '\0')
    {
        unsigned char b    = (unsigned char) *p;
        size_t        clen = ((b & 0x80u) == 0u)
                                 ? 1u
                                 : (((b & 0xE0u) == 0xC0u) ? 2u
                                                           : (((b & 0xF0u) == 0xE0u) ? 3u : 4u));
        char g[5] = { 0 };
        for (size_t i = 0u; i < clen && p[i] != '\0'; i++)
        {
            g[i] = p[i];
        }
        int sx = d->vx + (lx + col - d->app->schem_scroll_x);
        if (sx >= d->vx && sx < d->vx + d->vw)
        {
            cs_put_cell(&d->app->scr, sx, sy, g, fg, bg, attrs, 1);
        }
        p += clen;
        col++;
    }
}

static int dc_draw_node_chip(
    dag_draw_t         *d,
    int                 lx,
    int                 ly,
    int                 item_idx,
    const char         *tag,
    const char         *title,
    const char         *sub,
    cs_tui_rgb_t        accent,
    const char         *tip)
{
    int          sel = (d->app->sel_item == item_idx);
    cs_tui_rgb_t bg  = sel ? d->pal->bg_selected : d->pal->bg_panel_alt;
    cs_tui_rgb_t bfg = sel ? d->pal->fg_bright : accent;
    uint8_t      at  = sel ? CS_ATTR_BOLD : 0u;

    char top_line[64];
    char bot_line[64];
    snprintf(top_line, sizeof(top_line), "[%s] %s", tag, title);
    snprintf(bot_line, sizeof(bot_line), "%s", sub);

    int inner = (int) strlen(top_line);
    int slen  = (int) strlen(bot_line);
    if (slen > inner)
    {
        inner = slen;
    }
    if (inner < 11)
    {
        inner = 11;
    }
    if (inner > 24)
    {
        inner = 24;
    }
    int bw = inner + 4;
    if (sel)
    {
        d->sel_found = 1;
        d->sel_lx    = lx;
        d->sel_ly    = ly;
        d->sel_bw    = bw;
    }
    if (d->measure_only)
    {
        return bw;
    }

    const char *tl = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x94" : "\xe2\x95\xad");
    const char *tr = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x97" : "\xe2\x95\xae");
    const char *bl = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x9a" : "\xe2\x95\xb0");
    const char *br = d->app->scr.ascii_only ? "+" : (sel ? "\xe2\x95\x9d" : "\xe2\x95\xaf");
    const char *hz = d->app->scr.ascii_only ? (sel ? "=" : "-")
                                            : (sel ? "\xe2\x95\x90" : "\xe2\x94\x80");
    const char *vt = d->app->scr.ascii_only ? "|" : (sel ? "\xe2\x95\x91" : "\xe2\x94\x82");

    dc_put(d, lx, ly, tl, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly, tr, bfg, bg, at);
    dc_put(d, lx, ly + 3, bl, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly + 3, br, bfg, bg, at);
    for (int i = 1; i < bw - 1; i++)
    {
        dc_put(d, lx + i, ly, hz, bfg, bg, at);
        dc_put(d, lx + i, ly + 3, hz, bfg, bg, at);
        dc_put(d, lx + i, ly + 1, " ", d->pal->fg_text, bg, 0u);
        dc_put(d, lx + i, ly + 2, " ", d->pal->fg_muted, bg, 0u);
    }
    dc_put(d, lx, ly + 1, vt, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly + 1, vt, bfg, bg, at);
    dc_put(d, lx, ly + 2, vt, bfg, bg, at);
    dc_put(d, lx + bw - 1, ly + 2, vt, bfg, bg, at);

    char pad_t[32], pad_b[32];
    snprintf(pad_t, sizeof(pad_t), "%.*s", inner, top_line);
    snprintf(pad_b, sizeof(pad_b), "%.*s", inner, bot_line);
    dc_put(d, lx + 2, ly + 1, pad_t, sel ? d->pal->fg_bright : accent, bg, CS_ATTR_BOLD);
    dc_put(d, lx + 2, ly + 2, pad_b, sel ? d->pal->fg_text : d->pal->fg_muted, bg, 0u);

    /* Register clickable hit region clipped to viewport */
    int sx = d->vx + (lx - d->app->schem_scroll_x);
    int sy = d->vy + (ly - d->app->schem_scroll_y);
    int x0 = (sx < d->vx) ? d->vx : sx;
    int y0 = (sy < d->vy) ? d->vy : sy;
    int x1 = (sx + bw > d->vx + d->vw) ? (d->vx + d->vw) : (sx + bw);
    int y1 = (sy + 4 > d->vy + d->vh) ? (d->vy + d->vh) : (sy + 4);
    if (x1 > x0 && y1 > y0)
    {
        cs_tui_add_hit(d->app, CS_HIT_SCHEM_NODE, item_idx, x0, y0, x1 - x0, y1 - y0, tip);
    }
    return bw;
}

static void dc_draw_harrow(dag_draw_t *d, int lx, int ly, const char *label, cs_tui_rgb_t fg)
{
    const char *dash = d->app->scr.ascii_only ? "-" : "\xe2\x94\x80";
    const char *arr  = d->app->scr.ascii_only ? ">" : "\xe2\x96\xb6";
    if (label != NULL && label[0] != '\0')
    {
        const char *short_lbl = (strncmp(label, "b_", 2) == 0 && strlen(label) > 6u)
                                    ? (label + 2)
                                    : label;
        char        lbuf[16];
        snprintf(lbuf, sizeof(lbuf), "%.6s", short_lbl);
        dc_put(d, lx, ly, lbuf, d->pal->fg_dim, d->pal->bg_panel, 0u);
    }
    for (int i = 0; i < 5; i++)
    {
        dc_put(d, lx + i, ly + 1, dash, fg, d->pal->bg_panel, 0u);
    }
    dc_put(d, lx + 5, ly + 1, arr, fg, d->pal->bg_panel, CS_ATTR_BOLD);
}

static void dc_draw_vbranch(dag_draw_t *d, int lx, int ly0, int ly1, const char *arm_lbl)
{
    const char *vt  = d->app->scr.ascii_only ? "|" : "\xe2\x94\x82";
    const char *cbl = d->app->scr.ascii_only ? "`" : "\xe2\x95\xb0";
    const char *hz  = d->app->scr.ascii_only ? "-" : "\xe2\x94\x80";
    const char *arr = d->app->scr.ascii_only ? ">" : "\xe2\x96\xb6";

    for (int ry = ly0; ry < ly1 + 1; ry++)
    {
        dc_put(d, lx, ry, vt, d->pal->fg_conn, d->pal->bg_panel, 0u);
    }
    dc_put(d, lx, ly1 + 1, cbl, d->pal->fg_conn, d->pal->bg_panel, 0u);
    dc_put(d, lx + 1, ly1 + 1, hz, d->pal->fg_conn, d->pal->bg_panel, 0u);
    dc_put(d, lx + 2, ly1 + 1, hz, d->pal->fg_conn, d->pal->bg_panel, 0u);
    dc_put(d, lx + 3, ly1 + 1, arr, d->pal->fg_conn, d->pal->bg_panel, CS_ATTR_BOLD);
    if (arm_lbl != NULL && arm_lbl[0] != '\0')
    {
        dc_put(d, lx + 2, (ly1 > 0) ? (ly1 - 1) : ly1, arm_lbl, d->pal->fg_dim, d->pal->bg_panel,
               0u);
    }
}

static void dc_trace_beam(dag_draw_t *d, const char *beam, int lx, int row_slot, int depth)
{
    if (beam == NULL || beam[0] == '\0' || depth > 24)
    {
        return;
    }

    int consumers[CACAOSIM_BENCH_MAX_NODES];
    int n_cons = 0;
    for (uint32_t i = 0; i < d->app->cfg.n_nodes; i++)
    {
        if (!d->visited[i] && strcmp(d->app->cfg.nodes[i].in_beam, beam) == 0)
        {
            consumers[n_cons++] = (int) i;
        }
    }

    if (n_cons == 0)
    {
        int ly = row_slot * 5;
        char ubuf[32];
        snprintf(ubuf, sizeof(ubuf), "(%.20s)", beam);
        dc_put(d, lx, ly + 1, ubuf, d->pal->fg_dim, d->pal->bg_panel, CS_ATTR_DIM);
        return;
    }

    for (int ci = 0; ci < n_cons; ci++)
    {
        int idx = consumers[ci];
        if (d->visited[idx])
        {
            continue;
        }
        d->visited[idx] = 1u;

        int use_row = (ci == 0) ? row_slot : d->next_row++;
        int ly      = use_row * 5;
        int cur_x   = lx;

        if (ci > 0)
        {
            dc_draw_vbranch(d, lx - 3, row_slot * 5 + 2, ly, beam);
            cur_x = lx + 1;
        }

        const cacaosim_node_cfg_t *nd = &d->app->cfg.nodes[idx];
        char                       spec[40];
        char                       tip[96];
        node_short_spec(&d->app->cfg, nd, spec, sizeof(spec));
        snprintf(tip, sizeof(tip), "Click to inspect [%s.%s] (in=%s)", node_badge(nd), nd->id,
                 nd->in_beam);

        int bw = dc_draw_node_chip(d, cur_x, ly, idx + 1, node_badge(nd), nd->id, spec,
                                   node_color(nd, d->pal), tip);
        int nx = cur_x + bw;

        if (nd->kind == CACAOSIM_NODE_DM || nd->kind == CACAOSIM_NODE_WFE)
        {
            dc_draw_harrow(d, nx, ly, nd->out_beam, d->pal->fg_conn);
            dc_trace_beam(d, nd->out_beam, nx + 6, use_row, depth + 1);
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            dc_draw_harrow(d, nx, ly, nd->out_beam, d->pal->fg_conn);
            dc_trace_beam(d, nd->out_beam, nx + 6, use_row, depth + 1);
            if (nd->out_lowfs[0] != '\0')
            {
                int  lrow = d->next_row++;
                char llbl[24];
                if (nd->coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL)
                {
                    snprintf(llbl, sizeof(llbl), "lowfs:Lyot");
                }
                else
                {
                    float ap = (nd->coro.lowfs_pup_diam > 0.0f) ? nd->coro.lowfs_pup_diam : 1.0f;
                    snprintf(llbl, sizeof(llbl), "FPM(%.2fD)", ap);
                }
                dc_draw_vbranch(d, cur_x + bw / 2, ly + 4, lrow * 5, llbl);
                dc_trace_beam(d, nd->out_lowfs, cur_x + bw / 2 + 4, lrow, depth + 1);
            }
        }
        else if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            if (nd->out_trans[0] != '\0')
            {
                dc_draw_harrow(d, nx, ly, "T", d->pal->fg_conn);
                dc_trace_beam(d, nd->out_trans, nx + 6, use_row, depth + 1);
            }
            if (nd->out_refl[0] != '\0')
            {
                int rrow = d->next_row++;
                dc_draw_vbranch(d, cur_x + bw / 2, ly + 4, rrow * 5, "R");
                dc_trace_beam(d, nd->out_refl, cur_x + bw / 2 + 4, rrow, depth + 1);
            }
        }
        else
        {
            /* Terminal sensor: draw incoming spectrum sparkline + live flux/shm readout */
            const cacaosim_spec_t       *sp  = find_beam_spec(d->app, nd->in_beam);
            const cacaosim_bench_node_t *an  = find_analyzed_node(d->app, nd->id);
            int                          sw  = d->wide_spec ? 24 : 14;
            int                          spx = d->vx + (nx + 1 - d->app->schem_scroll_x);
            int                          spy = d->vy + (ly + 1 - d->app->schem_scroll_y);

            if (!d->measure_only && sp != NULL && spy >= d->vy && spy < d->vy + d->vh &&
                spx >= d->vx && spx + sw <= d->vx + d->vw)
            {
                cs_draw_spectrum32(&d->app->scr, spx, spy, sw, sp, d->app->opts.theme,
                                   d->pal->bg_panel);
            }
            if (an != NULL)
            {
                char fstr[24], info[64];
                fmt_flux_compact(an->eff_flux_ph, fstr, sizeof(fstr));
                snprintf(info, sizeof(info), "%s ph/fr -> %s", fstr, nd->out_cam);
                dc_put(d, nx + 1, ly + 2, info, d->pal->fg_stream, d->pal->bg_panel, 0u);
            }
        }
    }
}

void cs_render_dag_canvas(
    cs_tui_app_t *app,
    int           x,
    int           y,
    int           w,
    int           h,
    int           wide_spectra)
{
    if (w < 8 || h < 4)
    {
        return;
    }
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 cy  = y;
    int                 ch  = h;

    if (!app->dag_valid)
    {
        cs_fill_rect(&app->scr, x, cy, w, 1, pal->fg_error, pal->bg_selected);
        cs_printf_at(&app->scr, x + 1, cy, w - 2, pal->fg_error, pal->bg_selected, CS_ATTR_BOLD, 1,
                     "[!] DAG WIRING WARNING: %s (select node & press 'w' to rewire)",
                     app->dag_err);
        cy++;
        ch--;
    }

    char tel_sub[40];
    snprintf(tel_sub, sizeof(tel_sub), "%.1fm mV=%.1f %.0fHz", app->cfg.tel_diam_m,
             app->cfg.star_mag, app->cfg.master_hz);
    const char *root_beam = app->cfg.out_beam[0] ? app->cfg.out_beam : "b_pup";

    /* Auto-follow selected element when schematic pane is not being manually scrolled */
    if (app->focus != CS_FOCUS_SCHEM)
    {
        dag_draw_t md;
        memset(&md, 0, sizeof(md));
        md.app          = app;
        md.pal          = pal;
        md.vw           = w;
        md.vh           = ch;
        md.measure_only = 1;
        md.next_row     = 1;
        int mbw = dc_draw_node_chip(&md, 1, 0, 0, "TEL", app->cfg.name[0] ? app->cfg.name : "bench",
                                    tel_sub, pal->fg_title, "");
        dc_trace_beam(&md, root_beam, 1 + mbw + 6, 0, 0);
        for (uint32_t i = 0; i < app->cfg.n_nodes; i++)
        {
            if (!md.visited[i])
            {
                int                        urow = md.next_row++;
                const cacaosim_node_cfg_t *nd   = &app->cfg.nodes[i];
                dc_draw_node_chip(&md, 2, urow * 5, (int) i + 1, node_badge(nd), nd->id, "",
                                  pal->fg_error, "");
            }
        }
        if (md.sel_found)
        {
            int rx_need = md.sel_lx + md.sel_bw + 28;
            if (md.sel_lx < app->schem_scroll_x)
            {
                app->schem_scroll_x = (md.sel_lx > 2) ? (md.sel_lx - 2) : 0;
            }
            else if (rx_need > app->schem_scroll_x + w)
            {
                app->schem_scroll_x = rx_need - w;
                if (app->schem_scroll_x > md.sel_lx)
                {
                    app->schem_scroll_x = (md.sel_lx > 1) ? (md.sel_lx - 1) : 0;
                }
            }
            int ry_need = md.sel_ly + 9;
            if (md.sel_ly < app->schem_scroll_y)
            {
                app->schem_scroll_y = md.sel_ly;
            }
            else if (ry_need > app->schem_scroll_y + ch)
            {
                app->schem_scroll_y = ry_need - ch;
                if (app->schem_scroll_y > md.sel_ly)
                {
                    app->schem_scroll_y = md.sel_ly;
                }
            }
        }
    }

    dag_draw_t d;
    memset(&d, 0, sizeof(d));
    d.app       = app;
    d.pal       = pal;
    d.vx        = x;
    d.vy        = cy;
    d.vw        = w;
    d.vh        = ch;
    d.wide_spec = wide_spectra;
    d.next_row  = 1;

    int         bw        = dc_draw_node_chip(
        &d, 1, 0, 0, "TEL", app->cfg.name[0] ? app->cfg.name : "bench", tel_sub, pal->fg_title,
        "Click to inspect [bench] telescope, pupil & guide star");
    dc_draw_harrow(&d, 1 + bw, 0, root_beam, pal->fg_conn);
    dc_trace_beam(&d, root_beam, 1 + bw + 6, 0, 0);

    /* Render any disconnected/unwired nodes below main tree so user can click & rewire them */
    for (uint32_t i = 0; i < app->cfg.n_nodes; i++)
    {
        if (!d.visited[i])
        {
            int                        urow = d.next_row++;
            const cacaosim_node_cfg_t *nd   = &app->cfg.nodes[i];
            char                       sub[40];
            snprintf(sub, sizeof(sub), "UNWIRED in=%s", nd->in_beam);
            dc_draw_node_chip(&d, 2, urow * 5, (int) i + 1, node_badge(nd), nd->id, sub,
                              pal->fg_error, "Unwired optical element -- press 'w' to connect");
        }
    }
}

/* ── F2:BUILDER 3-Pane View (Elements + 2D Schematic + Inspector) ─────────── */

static int draw_btn(
    cs_tui_app_t       *app,
    int                 x,
    int                 y,
    const char         *lbl,
    cs_hit_kind_t       hk,
    int                 arg,
    cs_tui_rgb_t        fg,
    cs_tui_rgb_t        bg,
    const char         *tip)
{
    int len = cs_put_str(&app->scr, x, y, 24, lbl, fg, bg, CS_ATTR_BOLD, 1);
    cs_tui_add_hit(app, hk, arg, x, y, len, 1, tip);
    return len;
}

static void render_view_builder(cs_tui_app_t *app, int x, int y, int w, int h)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 lw  = app->left_w;
    if (lw < 28)
    {
        lw = 28;
    }
    if (lw > w - 42)
    {
        lw = w - 42;
    }
    if (lw < 24)
    {
        lw = w / 3;
    }
    int rx = x + lw;
    int rw = w - lw;

    int sh = (h * app->schem_h_pct) / 100;
    if (sh < 8)
    {
        sh = 8;
    }
    if (sh > h - 9)
    {
        sh = h - 9;
    }
    int ih = h - sh;

    /* 1. Left Pane: Optical Elements & Arranger */
    int f_el = (app->focus == CS_FOCUS_ELEMS);
    char nbadge[32];
    snprintf(nbadge, sizeof(nbadge), "%u nodes", app->cfg.n_nodes);
    cs_draw_box(&app->scr, x, y, lw, h, f_el, "ELEMENTS & ARRANGER", nbadge,
                f_el ? pal->fg_border_focus : pal->fg_border, pal->fg_title, pal->fg_muted,
                pal->bg_panel);
    cs_tui_add_hit(app, CS_HIT_SPLIT_LEFT, 0, x + lw - 1, y, 2, h,
                   "Drag vertical divider to resize Elements pane");

    /* Action toolbar inside Elements pane */
    int bx = x + 2;
    int by = y + 1;
    bx += draw_btn(app, bx, by, "[+Add]", CS_HIT_ACT_ADD, 0, pal->fg_active, pal->bg_panel_alt,
                   "Add optical element [a]") +
          1;
    bx += draw_btn(app, bx, by, "[Dup]", CS_HIT_ACT_CLONE, 0, pal->fg_stream, pal->bg_panel_alt,
                   "Duplicate selected element [c]") +
          1;
    bx += draw_btn(app, bx, by, "[Up]", CS_HIT_ACT_UP, 0, pal->fg_text, pal->bg_panel_alt,
                   "Move element upstream [K / Shift+Up]") +
          1;
    bx += draw_btn(app, bx, by, "[Dn]", CS_HIT_ACT_DOWN, 0, pal->fg_text, pal->bg_panel_alt,
                   "Move element downstream [J / Shift+Down]") +
          1;
    if (bx + 6 <= x + lw - 1)
    {
        bx += draw_btn(app, bx, by, "[Swp]", CS_HIT_ACT_SWAP, 0, pal->fg_proc, pal->bg_panel_alt,
                       "Swap splitter/coro output arms [x]") +
              1;
    }
    if (bx + 5 <= x + lw - 1)
    {
        draw_btn(app, bx, by, "[Del]", CS_HIT_ACT_DEL, 0, pal->fg_error, pal->bg_panel_alt,
                 "Remove selected element [d / Del]");
    }

    int show_pup_dock = (h >= 24 && lw >= 30);
    int tall_pup_dock = (h >= 28 && lw >= 30);
    int dock_rows     = tall_pup_dock ? 15 : 10;
    int list_top      = y + 3;
    int list_bot      = show_pup_dock ? (y + h - dock_rows) : (y + h - 1);
    int list_rows     = list_bot - list_top;
    int total_items   = 1 + (int) app->cfg.n_nodes;

    if (app->sel_item < app->elem_scroll)
    {
        app->elem_scroll = app->sel_item;
    }
    if (list_rows > 0 && app->sel_item >= app->elem_scroll + list_rows)
    {
        app->elem_scroll = app->sel_item - list_rows + 1;
    }

    cs_printf_at(&app->scr, x + 2, y + 2, lw - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                 " #  TYPE  ID / SPEC");

    for (int r = 0; r < list_rows; r++)
    {
        int item = app->elem_scroll + r;
        int ry   = list_top + r;
        if (item >= total_items)
        {
            break;
        }
        int          sel    = (item == app->sel_item);
        cs_tui_rgb_t row_bg = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, x + 1, ry, lw - 2, 1, pal->fg_text, row_bg);

        const char *ptr = sel ? (app->scr.ascii_only ? ">" : "\xe2\x96\xb6") : " ";
        cs_put_str(&app->scr, x + 1, ry, 1, ptr, pal->fg_accent, row_bg, CS_ATTR_BOLD, 1);

        if (item == 0)
        {
            cs_printf_at(&app->scr, x + 3, ry, lw - 5, sel ? pal->fg_bright : pal->fg_title, row_bg,
                         CS_ATTR_BOLD, 1, "[0] TEL   %-10.10s %.1fm mV=%.1f",
                         app->cfg.name[0] ? app->cfg.name : "bench", app->cfg.tel_diam_m,
                         app->cfg.star_mag);
            cs_tui_add_hit(app, CS_HIT_ELEM_ROW, 0, x + 1, ry, lw - 2, 1,
                           "Select [bench] entrance telescope & star");
        }
        else
        {
            const cacaosim_node_cfg_t *nd = &app->cfg.nodes[item - 1];
            char                       spec[32];
            node_short_spec(&app->cfg, nd, spec, sizeof(spec));
            cs_tui_rgb_t nc = node_color(nd, pal);
            cs_printf_at(&app->scr, x + 3, ry, 9, sel ? pal->fg_bright : nc, row_bg, CS_ATTR_BOLD,
                         1, "[%d] %-4s", item, node_badge(nd));
            cs_printf_at(&app->scr, x + 12, ry, lw - 15, sel ? pal->fg_bright : pal->fg_text,
                         row_bg, sel ? CS_ATTR_BOLD : 0u, 1, "%-10.10s %s", nd->id, spec);

            int live = node_primary_shm_live(app, nd);
            if (live >= 0 && lw >= 28)
            {
                const char  *dot = (live > 0) ? (app->scr.ascii_only ? "*" : "\xe2\x97\x8f")
                                              : (app->scr.ascii_only ? "o" : "\xe2\x97\x8b");
                cs_tui_rgb_t dfg = (live == 2) ? pal->fg_active
                                               : (live == 1 ? pal->fg_stream : pal->fg_dim);
                cs_put_str(&app->scr, x + lw - 3, ry, 1, dot, dfg, row_bg, 0u, 1);
            }
            char rtip[96];
            snprintf(rtip, sizeof(rtip), "Select [%s.%s] (in=%s)", node_badge(nd), nd->id,
                     nd->in_beam);
            cs_tui_add_hit(app, CS_HIT_ELEM_ROW, item, x + 1, ry, lw - 2, 1, rtip);
        }
    }

    if (show_pup_dock)
    {
        cs_optical_scales_t sc;
        cs_tui_compute_scales(&app->cfg, app->sel_item, &sc);

        int py = y + h - (tall_pup_dock ? 14 : 9);
        for (int cx = x + 1; cx < x + lw - 1; cx++)
        {
            cs_put_cell(&app->scr, cx, py - 1, app->scr.ascii_only ? "-" : "\xe2\x94\x80",
                        pal->fg_border, pal->bg_panel, 0u, 1);
        }
        cs_draw_pupil16(&app->scr, x + 2, py, app->pup16, pal);
        int tx = x + 19;
        int tw = lw - 20;
        if (tw > 6)
        {
            if (tall_pup_dock)
            {
                cs_printf_at(&app->scr, tx, py, tw, pal->fg_title, pal->bg_panel, CS_ATTR_BOLD, 1,
                             "PUPIL %ux%u", app->cfg.nwf, app->cfg.nwf);
                cs_printf_at(&app->scr, tx, py + 1, tw, pal->fg_text, pal->bg_panel, 0u, 1,
                             "D=%.1fm obs=%.0f%%", app->cfg.tel_diam_m, app->cfg.cobs * 100.0f);
                cs_printf_at(&app->scr, tx, py + 2, tw, pal->fg_muted, pal->bg_panel, 0u, 1,
                             "%d spiders", app->cfg.nspider);
                cs_printf_at(&app->scr, tx, py + 4, tw, pal->fg_accent, pal->bg_panel, 0u, 1,
                             "mV=%.1f %.0fK", app->cfg.star_mag, app->cfg.star_teff);
                if (app->dag_valid && app->ctx.n_beams > 0u)
                {
                    char f0[24];
                    fmt_flux_compact(cacaosim_spec_total(&app->ctx.beams[0].spec), f0, sizeof(f0));
                    cs_printf_at(&app->scr, tx, py + 5, tw, pal->fg_stream, pal->bg_panel, 0u, 1,
                                 "%s ph/s", f0);
                }
                draw_btn(app, tx, py + 7, "[g:InitSHM]", CS_HIT_ACT_MKPUP, 0, pal->fg_stream,
                         pal->bg_panel_alt,
                         "Generate pupil & static WFE streams in MILK_SHM_DIR [g]");
            }
            else
            {
                cs_printf_at(&app->scr, tx, py, tw, pal->fg_title, pal->bg_panel, CS_ATTR_BOLD, 1,
                             "%ux%u pix", app->cfg.nwf, app->cfg.nwf);
                cs_printf_at(&app->scr, tx, py + 1, tw, pal->fg_text, pal->bg_panel, 0u, 1,
                             "Beam %.1f pix", sc.d_beam_px);
                cs_printf_at(&app->scr, tx, py + 2, tw, pal->fg_stream, pal->bg_panel, 0u, 1,
                             "%.4f m/pix", sc.pup_m_pix);
                cs_printf_at(&app->scr, tx, py + 3, tw, pal->fg_accent, pal->bg_panel, 0u, 1,
                             "%.3f l/D/pix", sc.foc_ld_pix);
                cs_printf_at(&app->scr, tx, py + 4, tw, pal->fg_muted, pal->bg_panel, 0u, 1,
                             "%.4f arcsec/pix", sc.foc_arcsec_pix);
                cs_printf_at(&app->scr, tx, py + 5, tw, pal->fg_dim, pal->bg_panel, 0u, 1,
                             "@%.2fum D=%.1fm", sc.ref_lam_um, app->cfg.tel_diam_m);
                draw_btn(app, tx, py + 7, "[g:InitSHM]", CS_HIT_ACT_MKPUP, 0, pal->fg_stream,
                         pal->bg_panel_alt,
                         "Generate pupil & static WFE streams in MILK_SHM_DIR [g]");
            }
        }
        if (tall_pup_dock)
        {
            int fw = lw - 4;
            cs_printf_at(&app->scr, x + 2, py + 8, fw, pal->fg_text, pal->bg_panel, 0u, 1,
                         "Array: %ux%u pix", app->cfg.nwf, app->cfg.nwf);
            cs_printf_at(&app->scr, x + 2, py + 9, fw, pal->fg_text, pal->bg_panel, 0u, 1,
                         "Beam : %.1f pix (%.1f%%)", sc.d_beam_px, app->cfg.pup_diam * 100.0f);
            cs_printf_at(&app->scr, x + 2, py + 10, fw, pal->fg_stream, pal->bg_panel, 0u, 1,
                         "Pupil: %.4f m/pix", sc.pup_m_pix);
            cs_printf_at(&app->scr, x + 2, py + 11, fw, pal->fg_accent, pal->bg_panel, 0u, 1,
                         "Focal: %.3f l/D/pix (%.8s)", sc.foc_ld_pix, sc.foc_src);
            cs_printf_at(&app->scr, x + 2, py + 12, fw, pal->fg_muted, pal->bg_panel, 0u, 1,
                         "Sky  : %.4f arcsec/pix @%.2fum", sc.foc_arcsec_pix, sc.ref_lam_um);
        }
    }

    /* 2. Top-Right Pane: 2D Interactive Optical Schematic */
    int f_sc = (app->focus == CS_FOCUS_SCHEM);
    cs_draw_box(&app->scr, rx, y, rw, sh, f_sc, "2D OPTICAL SCHEMATIC & PHOTOMETRY",
                app->dag_valid ? "CLICK NODE TO INSPECT" : "CHECK WIRING",
                f_sc ? pal->fg_border_focus : pal->fg_border, pal->fg_title,
                app->dag_valid ? pal->fg_muted : pal->fg_error, pal->bg_panel);
    cs_render_dag_canvas(app, rx + 1, y + 1, rw - 2, sh - 2, rw >= 88);

    /* 3. Bottom-Right Pane: Declarative Parameter Inspector */
    int iy   = y + sh;
    int f_pa = (app->focus == CS_FOCUS_PARAMS);
    char ititle[96];
    if (app->sel_item <= 0 || (uint32_t) (app->sel_item - 1) >= app->cfg.n_nodes)
    {
        snprintf(ititle, sizeof(ititle), "INSPECTOR: [bench] %s", app->cfg.name);
    }
    else
    {
        const cacaosim_node_cfg_t *nd    = &app->cfg.nodes[app->sel_item - 1];
        const char                *kname = "fpcam";
        if (nd->kind == CACAOSIM_NODE_DM)
        {
            kname = (nd->dm.dm_mode == 1) ? "ttm" : "dm";
        }
        else if (nd->kind == CACAOSIM_NODE_SPLITTER)
        {
            kname = "splitter";
        }
        else if (nd->kind == CACAOSIM_NODE_CORO)
        {
            kname = "coro";
        }
        else if (nd->kind == CACAOSIM_NODE_PYRWFS)
        {
            kname = "pyrwfs";
        }
        else if (nd->kind == CACAOSIM_NODE_SHWFS)
        {
            kname = "shwfs";
        }
        else if (nd->kind == CACAOSIM_NODE_ZWFS)
        {
            kname = "zwfs";
        }
        else if (nd->kind == CACAOSIM_NODE_WFE)
        {
            kname = "wfe";
        }
        snprintf(ititle, sizeof(ititle), "INSPECTOR: [%s.%s]", kname, nd->id);
    }
    cs_draw_box(&app->scr, rx, iy, rw, ih, f_pa, ititle, "Enter:Edit  <-/->:Step",
                f_pa ? pal->fg_border_focus : pal->fg_border, pal->fg_title, pal->fg_muted,
                pal->bg_panel);
    cs_tui_add_hit(app, CS_HIT_SPLIT_SCHEM, 0, rx, iy, rw, 1,
                   "Drag horizontal divider to resize Schematic / Inspector");

    /* Inspector subtabs */
    static const char *const st_names[3] = { " 1:OPTICS ", " 2:DYN/DET ", " 3:SHM/WIRING " };
    int                      stx         = rx + 2;
    for (int st = 0; st < 3; st++)
    {
        int          act = (app->subtab == st);
        cs_tui_rgb_t sfg = act ? pal->fg_bright : pal->fg_muted;
        cs_tui_rgb_t sbg = act ? pal->bg_selected : pal->bg_panel_alt;
        int          len = cs_put_str(&app->scr, stx, iy + 1, 20, st_names[st], sfg, sbg,
                                      act ? CS_ATTR_BOLD : 0u, 1);
        cs_tui_add_hit(app, CS_HIT_SUBTAB, st, stx, iy + 1, len, 1,
                       "Switch parameter category [1 / 2 / 3]");
        stx += len + 1;
    }
    if (app->sel_item > 0 && stx + 14 < rx + rw - 1)
    {
        draw_btn(app, stx + 1, iy + 1, "[w:Wire Beam]", CS_HIT_ACT_WIRE, 0, pal->fg_stream,
                 pal->bg_panel_alt, "Pick upstream optical beam for this element [w]");
    }

    int p_top  = iy + 3;
    int p_rows = (iy + ih - 1) - p_top;
    cs_printf_at(&app->scr, rx + 2, iy + 2, rw - 4, pal->fg_dim, pal->bg_panel, 0u, 1,
                 "%-14s %-18s %-18s %-6s %s", "KEY", "PARAMETER", "VALUE", "UNIT", "NOTES");

    if (app->param_sel >= app->n_params && app->n_params > 0)
    {
        app->param_sel = app->n_params - 1;
    }
    if (app->param_sel < 0)
    {
        app->param_sel = 0;
    }
    if (app->param_sel < app->param_scroll)
    {
        app->param_scroll = app->param_sel;
    }
    if (p_rows > 0 && app->param_sel >= app->param_scroll + p_rows)
    {
        app->param_scroll = app->param_sel - p_rows + 1;
    }

    for (int r = 0; r < p_rows; r++)
    {
        int pi = app->param_scroll + r;
        int py = p_top + r;
        if (pi >= app->n_params)
        {
            break;
        }
        const cs_param_desc_t *pd  = &app->params[pi];
        int                    sel = (pi == app->param_sel && f_pa);
        cs_tui_rgb_t           rbg = sel ? pal->bg_selected : pal->bg_panel;
        cs_fill_rect(&app->scr, rx + 1, py, rw - 2, 1, pal->fg_text, rbg);

        char vbuf[96];
        cs_param_format_val(&app->cfg, app->sel_item, pd, vbuf, sizeof(vbuf));

        cs_printf_at(&app->scr, rx + 2, py, 14, pal->fg_fps, rbg, 0u, 1, "%-14.14s", pd->key);
        cs_printf_at(&app->scr, rx + 17, py, 18, sel ? pal->fg_bright : pal->fg_text, rbg,
                     sel ? CS_ATTR_BOLD : 0u, 1, "%-18.18s", pd->label);

        if (sel && app->editing)
        {
            cs_fill_rect(&app->scr, rx + 36, py, 18, 1, pal->fg_bright, pal->bg_input);
            cs_printf_at(&app->scr, rx + 36, py, 17, pal->fg_accent, pal->bg_input, CS_ATTR_BOLD, 1,
                         "%s_", app->edit_buf);
        }
        else if (pd->type == CS_PTYPE_ENUM || pd->type == CS_PTYPE_BOOL ||
                 pd->type == CS_PTYPE_BEAM)
        {
            char disp[32];
            snprintf(disp, sizeof(disp), "< %.26s >", vbuf[0] ? vbuf : "-");
            cs_tui_rgb_t vfg = (pd->type == CS_PTYPE_BOOL && strcmp(vbuf, "ON") == 0)
                                   ? pal->fg_active
                                   : pal->fg_accent;
            cs_printf_at(&app->scr, rx + 36, py, 18, vfg, rbg, CS_ATTR_BOLD, 1, "%-18.18s", disp);
        }
        else
        {
            char        pdisp[32];
            const char *vshow = vbuf[0] ? vbuf : "\"\"";
            if (strlen(vbuf) > 18u && strchr(vbuf, '/') != NULL)
            {
                const char *bn = strrchr(vbuf, '/');
                snprintf(pdisp, sizeof(pdisp), ".../%.14s", (bn != NULL) ? (bn + 1) : vbuf);
                vshow = pdisp;
            }
            cs_printf_at(&app->scr, rx + 36, py, 18, sel ? pal->fg_bright : pal->fg_stream, rbg,
                         sel ? CS_ATTR_BOLD : 0u, 1, "%-18.18s", vshow);
        }

        cs_printf_at(&app->scr, rx + 55, py, 6, pal->fg_muted, rbg, 0u, 1, "%-6.6s", pd->unit);
        if (rw > 66)
        {
            cs_printf_at(&app->scr, rx + 62, py, rw - 64, pal->fg_dim, rbg, 0u, 1, "%s", pd->tip);
        }
        cs_tui_add_hit(app, CS_HIT_PARAM_ROW, pi, rx + 1, py, rw - 2, 1, pd->tip);
    }
}

/* ── Top Header, Tab Bar, Bottom Status Bar, Modals & Tooltips ────────────── */

static void render_header_and_tabs(cs_tui_app_t *app)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 W   = app->scr.cols;

    /* Row 0: milk-CTRL Header Bar */
    cs_fill_rect(&app->scr, 0, 0, W, 1, pal->fg_text, pal->bg_bar);
    const char *hb = ((app->frame_tick % 2u) == 0u)
                         ? (app->scr.ascii_only ? "*" : "\xe2\x97\x86")
                         : (app->scr.ascii_only ? "o" : "\xe2\x97\x87");
    cs_put_str(&app->scr, 1, 0, 2, hb, pal->fg_active, pal->bg_bar, CS_ATTR_BOLD, 1);
    cs_put_str(&app->scr, 3, 0, 14, "cacaosim-CTRL", pal->fg_title, pal->bg_bar, CS_ATTR_BOLD, 1);

    char path_copy[512];
    snprintf(path_copy, sizeof(path_copy), "%s",
             app->file_path[0] ? app->file_path : "unsaved.bench");
    const char *bname = basename(path_copy);
    int         cx    = 18;
    cx += cs_printf_at(&app->scr, cx, 0, 44, pal->fg_bright, pal->bg_bar, 0u, 1, "[%s] %s",
                       app->cfg.name[0] ? app->cfg.name : "bench", bname);
    if (app->dirty)
    {
        cx += cs_put_str(&app->scr, cx + 1, 0, 12, "[*MODIFIED]", pal->fg_warn, pal->bg_bar,
                         CS_ATTR_BOLD, 1) +
              1;
    }

    /* Right-aligned status pills on Row 0 */
    char rstat[128];
    if (app->shm.has_strehl)
    {
        snprintf(rstat, sizeof(rstat), "%s  SHM:%d/%d  FPS:%d  SR:%.2f  %4.1f kB/s ",
                 app->dag_valid ? "DAG:OK" : "DAG:ERR", app->shm.n_live_streams, app->shm.n_streams,
                 app->shm.n_running_fps, app->shm.strehl_ave, app->scr.tx_kbps);
    }
    else
    {
        snprintf(rstat, sizeof(rstat), "%s  SHM:%d/%d  FPS:%d  %4.1f kB/s ",
                 app->dag_valid ? "DAG:OK" : "DAG:ERR", app->shm.n_live_streams, app->shm.n_streams,
                 app->shm.n_running_fps, app->scr.tx_kbps);
    }
    int rlen = (int) strlen(rstat);
    if (W - rlen > cx + 2)
    {
        cs_put_str(&app->scr, W - rlen, 0, rlen, rstat,
                   !app->dag_valid
                       ? pal->fg_error
                       : (app->shm.n_running_fps > 0 ? pal->fg_active : pal->fg_muted),
                   pal->bg_bar, CS_ATTR_BOLD, 1);
    }

    /* Row 1: F2..F6 View Tabs + Action Pills */
    cs_fill_rect(&app->scr, 0, 1, W, 1, pal->fg_text, pal->bg_bar);
    static const char *const vtabs[CACAOSIM_TUI_VIEW_COUNT] = {
        " F2:BUILDER ", " F3:SCHEMATIC ", " F4:TABLES ", " F5:STATUS ", " F6:SOURCE "
    };
    static const char *const vtips[CACAOSIM_TUI_VIEW_COUNT] = {
        "Interactive 3-pane hardware builder & inspector [F2]",
        "Full-screen 2D optical schematic & 32-bin spectra [F3]",
        "Hardware mirrors, sensors & optical beams tables [F4]",
        "Live MILK_SHM_DIR streams, FPS units & deployment [F5]",
        "Formatted .bench INI/TOML configuration preview [F6]",
    };

    int tx = 1;
    for (int v = 0; v < CACAOSIM_TUI_VIEW_COUNT; v++)
    {
        int          act = (app->view == v);
        cs_tui_rgb_t fg  = act ? pal->fg_bright : pal->fg_muted;
        cs_tui_rgb_t bg  = act ? pal->bg_selected : pal->bg_panel_alt;
        int          len = cs_put_str(&app->scr, tx, 1, 20, vtabs[v], fg, bg,
                                      act ? CS_ATTR_BOLD : 0u, 1);
        cs_tui_add_hit(app, CS_HIT_VIEW_TAB, v, tx, 1, len, 1, vtips[v]);
        tx += len + 1;
    }

    /* Quick right-side action pills on Row 1 */
    int rx = W - 47;
    if (rx > tx + 2)
    {
        rx += draw_btn(app, rx, 1, "[P:Presets]", CS_HIT_ACT_PRESET, 0, pal->fg_accent,
                       pal->bg_panel_alt, "Load built-in optical bench template [P]") +
              1;
        rx += draw_btn(app, rx, 1, "[s:Save]", CS_HIT_ACT_SAVE, 0, pal->fg_stream,
                       pal->bg_panel_alt, "Save .bench configuration to disk [s]") +
              1;
        rx += draw_btn(app, rx, 1, "[f:Run]", CS_HIT_ACT_DEPLOY_F, 0, pal->fg_active,
                       pal->bg_panel_alt, "Deploy & start fused htsim bench [f]") +
              1;
        rx += draw_btn(app, rx, 1, "[S:Stop]", CS_HIT_ACT_STOP, 0, pal->fg_error,
                       pal->bg_panel_alt, "Stop running cacaosim FPS units [S]") +
              1;
        draw_btn(app, rx, 1, "[h:HELP]", CS_HIT_HELP_BTN, 0, pal->fg_bright, pal->bg_selected,
                 "Show keyboard & mouse shortcuts [h / F1]");
    }
}

static void render_bottom_bars(cs_tui_app_t *app)
{
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 W   = app->scr.cols;
    int                 ty  = app->scr.rows - 2;
    int                 sy  = app->scr.rows - 1;

    /* Row rows-2: Toast / Active hint line */
    cs_fill_rect(&app->scr, 0, ty, W, 1, pal->fg_muted, pal->bg_panel_alt);
    double now = cs_tui_now_sec();
    if (app->toast[0] != '\0' && now < app->toast_until_s)
    {
        cs_tui_rgb_t tfg = (app->toast_level == 3)
                               ? pal->fg_error
                               : (app->toast_level == 2
                                      ? pal->fg_warn
                                      : (app->toast_level == 1 ? pal->fg_active : pal->fg_bright));
        cs_printf_at(&app->scr, 2, ty, W - 4, tfg, pal->bg_panel_alt, CS_ATTR_BOLD, 1, "%s",
                     app->toast);
    }
    else if (app->view == CACAOSIM_TUI_VIEW_BUILDER && app->param_sel >= 0 &&
             app->param_sel < app->n_params)
    {
        const cs_param_desc_t *pd = &app->params[app->param_sel];
        cs_printf_at(&app->scr, 2, ty, W - 4, pal->fg_muted, pal->bg_panel_alt, 0u, 1,
                     "%s: %s", pd->key, pd->tip);
    }
    else if (app->n_logs > 0)
    {
        cs_printf_at(&app->scr, 2, ty, W - 4, pal->fg_muted, pal->bg_panel_alt, 0u, 1, "%s",
                     app->logs[app->n_logs - 1]);
    }

    /* Row rows-1: milk-CTRL Status Bar */
    cs_fill_rect(&app->scr, 0, sy, W, 1, pal->fg_text, pal->bg_bar);
    const char *hints =
        (app->editing)
            ? " Type value  Enter:Apply  Esc:Cancel  Bksp:Delete"
            : " Tab:Pane  a:Add  c:Dup  K/J:Move  x:Swap  w:Wire  d:Del  u:Undo  s:Save  f:Run";
    char m_lbl[16], t_lbl[20];
    snprintf(m_lbl, sizeof(m_lbl), "[m:%s]", app->show_tooltips ? "tips" : "off");
    snprintf(t_lbl, sizeof(t_lbl), "[T:%s]", pal->name);
    int rbar_w = (int) strlen(m_lbl) + 1 + (int) strlen(t_lbl) + 1 + 8 + 2 + 9;
    int rx     = W - rbar_w;
    cs_put_str(&app->scr, 0, sy, (rx > 2) ? (rx - 1) : (W - 2), hints, pal->fg_muted, pal->bg_bar,
               0u, 1);

    if (rx > 10)
    {
        rx += draw_btn(app, rx, sy, m_lbl, CS_HIT_TOOLTIP_BTN, 0,
                       app->show_tooltips ? pal->fg_active : pal->fg_dim, pal->bg_bar,
                       "Toggle mouse hover tooltips") +
              1;
        rx += draw_btn(app, rx, sy, t_lbl, CS_HIT_THEME_BTN, 0, pal->fg_accent, pal->bg_bar,
                       "Open color theme selector [T]") +
              1;
        rx += draw_btn(app, rx, sy, "[q:exit]", CS_HIT_EXIT_BTN, 0, pal->fg_error, pal->bg_bar,
                       "Quit cacaosim-CTRL [q]") +
              2;

        time_t     wall = time(NULL);
        struct tm *tmv  = localtime(&wall);
        if (tmv != NULL && rx + 8 <= W)
        {
            cs_printf_at(&app->scr, W - 9, sy, 9, pal->fg_dim, pal->bg_bar, 0u, 1, "%02d:%02d:%02d",
                         tmv->tm_hour, tmv->tm_min, tmv->tm_sec);
        }
    }
}

static void render_modal(cs_tui_app_t *app)
{
    if (app->modal == CS_MODAL_NONE)
    {
        return;
    }
    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    int                 W   = app->scr.cols;
    int                 H   = app->scr.rows;

    if (app->modal == CS_MODAL_ADD)
    {
        static const char *const names[9] = {
            "1. [dm]       Zonal Deformable Mirror",
            "2. [ttm]      2-Axis Tip-Tilt Mirror",
            "3. [splitter] Dichroic / Gray Beamsplitter",
            "4. [coro]     Coronagraph + LOWFS Pickoff",
            "5. [pyrwfs]   4-Sided Pyramid WFS",
            "6. [shwfs]    Shack-Hartmann Lenslet WFS",
            "7. [zwfs]     Zernike Phase-Contrast WFS",
            "8. [fpcam]    Focal-Plane Science Camera",
            "9. [wfe]      Static Wavefront Error [OPD m]",
        };
        static const char *const descs[9] = {
            "Actuator grid with ZOH servo dynamics, stroke clip & DAC bits",
            "Fast steering tip-tilt mirror (dm_mode=1) with servo latency",
            "Splits beam into out_refl + out_trans (gray, LP, SP, or BP)",
            "Analytic/custom FPM & pupil masks + reflective FPM or Lyot LOWFS",
            "Modulated/unmodulated diffractive 4-quadrant Pyramid WFS",
            "Diffractive lenslet array Shack-Hartmann wavefront sensor",
            "Babinet Matrix-DFT Zernike phase-contrast wavefront sensor",
            "Polychromatic MFT/FFT science PSF camera + Strehl telemetry",
            "Inline static wavefront error SHM stream in unit of OPD [meter]",
        };

        char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
        uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
        int      mw = (W > 80) ? 76 : (W - 4);
        int      mh = 16;
        int      mx = (W - mw) / 2;
        int      my = (H - mh) / 2;

        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "ADD OPTICAL ELEMENT", "1..9 / Enter  Esc:Cancel",
                    pal->fg_border_focus, pal->fg_title, pal->fg_muted, pal->bg_popup);

        const char *b_lbl = (app->modal_beam_idx >= 0 && (uint32_t) app->modal_beam_idx < nb)
                                ? beams[app->modal_beam_idx]
                                : "AUTO (smart splice after selection)";
        cs_printf_at(&app->scr, mx + 2, my + 1, mw - 4, pal->fg_stream, pal->bg_popup, 0u, 1,
                     "Target Input Beam [Tab to cycle]: %s", b_lbl);
        cs_tui_add_hit(app, CS_HIT_MODAL_EXTRA, 0, mx + 2, my + 1, mw - 4, 1,
                       "Click or press Tab to cycle target input beam");

        for (int i = 0; i < 9; i++)
        {
            int          sel = (app->modal_sel == i);
            cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 3 + i, mw - 2, 1, pal->fg_text, bg);
            cs_printf_at(&app->scr, mx + 2, my + 3 + i, mw - 4,
                         sel ? pal->fg_bright : pal->fg_text, bg, sel ? CS_ATTR_BOLD : 0u, 1,
                         "%-44s", names[i]);
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 3 + i, mw - 2, 1, descs[i]);
        }
        cs_printf_at(&app->scr, mx + 2, my + 13, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1, "%s",
                     descs[(app->modal_sel >= 0 && app->modal_sel < 9) ? app->modal_sel : 0]);
        draw_btn(app, mx + mw - 22, my + 14, "[Enter:Insert]", CS_HIT_MODAL_OK, 0, pal->fg_active,
                 pal->bg_panel_alt, "Insert selected optical element");
        draw_btn(app, mx + mw - 7, my + 14, "[Esc]", CS_HIT_MODAL_CANCEL, 0, pal->fg_error,
                 pal->bg_panel_alt, "Cancel");
        return;
    }

    if (app->modal == CS_MODAL_WIRE)
    {
        char     beams[CACAOSIM_BENCH_MAX_BEAMS][CACAOSIM_BENCH_NAME_LEN];
        uint32_t nb = cacaosim_bench_list_beams(&app->cfg, beams, CACAOSIM_BENCH_MAX_BEAMS);
        int      mw = (W > 76) ? 72 : (W - 4);
        int      mh = (int) nb + 6;
        if (mh > H - 4)
        {
            mh = H - 4;
        }
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;

        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "WIRE UPSTREAM OPTICAL BEAM",
                    "Enter:Connect  Esc:Cancel", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        cs_printf_at(&app->scr, mx + 2, my + 1, mw - 4, pal->fg_dim, pal->bg_popup, 0u, 1,
                     "%-18s %-12s %-10s %s", "BEAM NAME", "FLUX [ph/s]", "MEAN LAM", "SPECTRUM");

        int max_r = mh - 4;
        for (int i = 0; i < (int) nb && i < max_r; i++)
        {
            int          sel = (app->modal_sel == i);
            cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 2 + i, mw - 2, 1, pal->fg_text, bg);

            const cacaosim_spec_t *sp = find_beam_spec(app, beams[i]);
            char                   fstr[24] = "-";
            float                  mlam     = 0.0f;
            if (sp != NULL)
            {
                fmt_flux_compact(cacaosim_spec_total(sp), fstr, sizeof(fstr));
                mlam = cs_tui_spec_mean_lambda(sp);
            }
            cs_printf_at(&app->scr, mx + 2, my + 2 + i, 44, sel ? pal->fg_bright : pal->fg_stream,
                         bg, sel ? CS_ATTR_BOLD : 0u, 1, "%-18.18s %-12s %.2f um", beams[i], fstr,
                         mlam);
            if (sp != NULL && mw >= 62)
            {
                cs_draw_spectrum32(&app->scr, mx + 44, my + 2 + i, 20, sp, app->opts.theme, bg);
            }
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 2 + i, mw - 2, 1,
                           "Connect element input to this optical beam");
        }
        return;
    }

    if (app->modal == CS_MODAL_PRESET)
    {
        int mw = (W > 84) ? 80 : (W - 4);
        int mh = CACAOSIM_PRESET_COUNT + 6;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "LOAD HARDWARE TEMPLATE (.bench PRESET)",
                    "Enter:Load  Esc:Cancel", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        for (int i = 0; i < CACAOSIM_PRESET_COUNT; i++)
        {
            int          sel = (app->modal_sel == i);
            cs_tui_rgb_t bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 2 + i, mw - 2, 1, pal->fg_text, bg);
            cs_printf_at(&app->scr, mx + 2, my + 2 + i, mw - 4,
                         sel ? pal->fg_bright : pal->fg_accent, bg, sel ? CS_ATTR_BOLD : 0u, 1,
                         "%d. %s", i + 1, cacaosim_bench_preset_name(i));
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 2 + i, mw - 2, 1,
                           cacaosim_bench_preset_desc(i));
        }
        cs_printf_at(&app->scr, mx + 2, my + mh - 2, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1,
                     "%s", cacaosim_bench_preset_desc(app->modal_sel));
        return;
    }

    if (app->modal == CS_MODAL_THEME)
    {
        int mw = 58;
        int mh = CACAOSIM_THEME_COUNT + 6;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "SELECT MILK COLOR THEME",
                    "Enter:Select  a:ASCII", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        for (int i = 0; i < CACAOSIM_THEME_COUNT; i++)
        {
            const cs_tui_pal_t *tp  = cs_tui_get_palette(i);
            int                 sel = (app->modal_sel == i);
            cs_tui_rgb_t        bg  = sel ? pal->bg_selected : pal->bg_popup;
            cs_fill_rect(&app->scr, mx + 1, my + 2 + i, mw - 2, 1, pal->fg_text, bg);
            cs_printf_at(&app->scr, mx + 2, my + 2 + i, 38, sel ? pal->fg_bright : pal->fg_text, bg,
                         sel ? CS_ATTR_BOLD : 0u, 1, "%d. %-33s", i + 1, tp->label);
            cs_put_str(&app->scr, mx + 42, my + 2 + i, 3, "###", tp->fg_title, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_put_str(&app->scr, mx + 45, my + 2 + i, 3, "###", tp->fg_stream, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_put_str(&app->scr, mx + 48, my + 2 + i, 3, "###", tp->fg_active, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_put_str(&app->scr, mx + 51, my + 2 + i, 3, "###", tp->fg_accent, tp->bg_panel,
                       CS_ATTR_BOLD, 1);
            cs_tui_add_hit(app, CS_HIT_MODAL_ROW, i, mx + 1, my + 2 + i, mw - 2, 1, tp->label);
        }
        cs_printf_at(&app->scr, mx + 2, my + mh - 2, mw - 4, pal->fg_muted, pal->bg_popup, 0u, 1,
                     "7-bit ASCII borders [press 'a']: %s", app->scr.ascii_only ? "ON" : "OFF");
        return;
    }

    if (app->modal == CS_MODAL_SAVE || app->modal == CS_MODAL_OPEN)
    {
        int mw = (W > 72) ? 68 : (W - 4);
        int mh = 7;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1,
                    (app->modal == CS_MODAL_SAVE) ? "SAVE .bench CONFIGURATION"
                                                  : "OPEN .bench CONFIGURATION",
                    "Enter:OK  Esc:Cancel", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        cs_put_str(&app->scr, mx + 2, my + 2, mw - 4, "File path:", pal->fg_muted, pal->bg_popup,
                   0u, 1);
        cs_fill_rect(&app->scr, mx + 2, my + 3, mw - 4, 1, pal->fg_bright, pal->bg_input);
        cs_printf_at(&app->scr, mx + 2, my + 3, mw - 5, pal->fg_accent, pal->bg_input, CS_ATTR_BOLD,
                     1, "%s_", app->modal_input);
        draw_btn(app, mx + mw - 18, my + 5, "[Enter:Confirm]", CS_HIT_MODAL_OK, 0, pal->fg_active,
                 pal->bg_panel_alt, "Confirm file operation");
        return;
    }

    if (app->modal == CS_MODAL_CONFIRM)
    {
        int mw = 56;
        int mh = 7;
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "UNSAVED CONFIGURATION CHANGES", "y / n",
                    pal->fg_warn, pal->fg_warn, pal->fg_muted, pal->bg_popup);
        cs_put_str(&app->scr, mx + 2, my + 2, mw - 4,
                   "Discard unsaved optical bench edits and continue?", pal->fg_bright,
                   pal->bg_popup, 0u, 1);
        draw_btn(app, mx + 12, my + 4, "[y: Discard & Continue]", CS_HIT_MODAL_OK, 0, pal->fg_error,
                 pal->bg_panel_alt, "Proceed without saving");
        draw_btn(app, mx + 38, my + 4, "[n: Cancel]", CS_HIT_MODAL_CANCEL, 0, pal->fg_active,
                 pal->bg_panel_alt, "Return to builder");
        return;
    }

    if (app->modal == CS_MODAL_HELP)
    {
        static const char *const hlines[] = {
            "NAVIGATION & VIEWS",
            "  F2..F6            Switch Builder / Schematic / Tables / Status / Source",
            "  Tab / Shift+Tab   Cycle focus between Elements, Schematic & Inspector",
            "  Up / Down / Click Select element, parameter, or click any 2D schematic box",
            "HARDWARE ARRANGER",
            "  a / +             Add element (DM, TTM, BS, Coro, PyWFS, SH, ZWFS, Cam, WFE)",
            "  c                 Duplicate (clone) selected element with auto-wiring",
            "  K / J (Shift+U/D) Move element upstream / downstream (optically splices chain!)",
            "  x                 Swap downstream arms on Beamsplitter (R/T) or Coro (Sci/LOWFS)",
            "  w                 Rewire selected element's input beam from upstream beams",
            "  d / Delete        Remove element & automatically bridge upstream/downstream beams",
            "  u / Ctrl+R        Undo / Redo configuration edits (24-step history)",
            "PARAMETER INSPECTOR & DEPLOYMENT",
            "  1 / 2 / 3         Switch Inspector subtab (Optics, Dyn/Det, SHM/Wiring)",
            "  Enter / Space     Edit parameter inline or cycle enum/boolean value",
            "  Left / Right      Step numeric/enum value ([ / ] for 5x coarse step)",
            "  P / s / A / o / e Presets / Save / Save As / Open / Edit in $EDITOR",
            "  f / M / S / g / t Deploy Fused / Deploy Modular / Stop / InitSHM / AtmTurb",
            "  T / m / q         Theme selector / Toggle hover tooltips / Quit",
        };
        int n_hl = (int) (sizeof(hlines) / sizeof(hlines[0]));
        int mw   = (W > 86) ? 84 : (W - 2);
        int mh   = n_hl + 4;
        if (mh > H - 2)
        {
            mh = H - 2;
        }
        int mx = (W - mw) / 2;
        int my = (H - mh) / 2;
        cs_draw_shadow(&app->scr, mx, my, mw, mh, pal);
        cs_draw_box(&app->scr, mx, my, mw, mh, 1, "cacaosim-CTRL KEYBOARD & MOUSE REFERENCE",
                    "Esc / h: Close", pal->fg_border_focus, pal->fg_title, pal->fg_muted,
                    pal->bg_popup);
        for (int i = 0; i < n_hl && i < mh - 3; i++)
        {
            int hdr = (hlines[i][0] != ' ');
            cs_put_str(&app->scr, mx + 2, my + 2 + i, mw - 4, hlines[i],
                       hdr ? pal->fg_accent : pal->fg_text, pal->bg_popup, hdr ? CS_ATTR_BOLD : 0u,
                       1);
        }
        cs_tui_add_hit(app, CS_HIT_MODAL_CANCEL, 0, mx, my, mw, mh, "Click to close help");
    }
}

static void render_hover_tooltip(cs_tui_app_t *app)
{
    if (!app->show_tooltips || app->modal != CS_MODAL_NONE || app->mouse_x < 0 || app->mouse_y < 0)
    {
        return;
    }
    const char *tip = NULL;
    for (int i = app->n_hits - 1; i >= 0; i--)
    {
        const cs_hit_t *h = &app->hits[i];
        if (app->mouse_x >= h->x && app->mouse_x < h->x + h->w && app->mouse_y >= h->y &&
            app->mouse_y < h->y + h->h && h->tip[0] != '\0')
        {
            tip = h->tip;
            break;
        }
    }
    if (tip == NULL)
    {
        return;
    }
    const cs_tui_pal_t *pal  = cs_tui_get_palette(app->opts.theme);
    int                 tlen = (int) strlen(tip) + 2;
    if (tlen > app->scr.cols - 4)
    {
        tlen = app->scr.cols - 4;
    }
    int tx = app->mouse_x + 2;
    int ty = (app->mouse_y > 3) ? (app->mouse_y - 1) : (app->mouse_y + 1);
    if (tx + tlen >= app->scr.cols)
    {
        tx = app->scr.cols - tlen - 1;
    }
    if (tx < 1)
    {
        tx = 1;
    }
    cs_fill_rect(&app->scr, tx, ty, tlen, 1, pal->fg_bright, pal->bg_popup);
    cs_printf_at(&app->scr, tx + 1, ty, tlen - 2, pal->fg_accent, pal->bg_popup, 0u, 1, "%s", tip);
}

void cs_tui_render_frame(cs_tui_app_t *app)
{
    if (app == NULL)
    {
        return;
    }
    cs_tui_refresh_analysis(app);
    app->n_params = cs_params_build(&app->cfg, app->sel_item, app->subtab, app->params,
                                    CS_MAX_PARAMS);
    app->n_hits   = 0;

    const cs_tui_pal_t *pal = cs_tui_get_palette(app->opts.theme);
    cs_scr_clear(&app->scr, pal->fg_text, pal->bg_panel);

    render_header_and_tabs(app);

    int bx = 0;
    int by = 2;
    int bw = app->scr.cols;
    int bh = app->scr.rows - 4;

    if (bh >= 6)
    {
        switch (app->view)
        {
        case CACAOSIM_TUI_VIEW_SCHEMATIC:
            cs_render_view_schematic(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_TABLES:
            cs_render_view_tables(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_STATUS:
            cs_render_view_status(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_SOURCE:
            cs_render_view_source(app, bx, by, bw, bh);
            break;
        case CACAOSIM_TUI_VIEW_BUILDER:
        default:
            render_view_builder(app, bx, by, bw, bh);
            break;
        }
    }

    render_bottom_bars(app);
    render_modal(app);
    render_hover_tooltip(app);
}
