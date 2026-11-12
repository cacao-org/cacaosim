// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_dag.c
 * @brief   Topological compiler, spectral photon budget, and CPU/CUDA multi-rate DAG runtime
 */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "bench/cacaosim_bench.h"
#include "common/cacaosim_cuda.h"
#include "common/cacaosim_mask.h"
#include "pupil/cacaosim_pupil.h"

static void dag_err(char *buf, size_t cap, const char *fmt, ...)
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

static int find_beam_name(const cacaosim_bench_ctx_t *ctx, const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    for (uint32_t i = 0; i < ctx->n_beams; i++)
    {
        if (strcmp(ctx->beams[i].name, name) == 0)
        {
            return (int) i;
        }
    }
    return -1;
}

static int register_beam_name(
    cacaosim_bench_ctx_t *ctx,
    const char           *name,
    char                 *err_buf,
    size_t                err_cap)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    if (find_beam_name(ctx, name) >= 0)
    {
        dag_err(err_buf, err_cap, "Duplicate producer for optical beam '%s'", name);
        return -2;
    }
    if (ctx->n_beams >= CACAOSIM_BENCH_MAX_BEAMS)
    {
        dag_err(err_buf, err_cap, "Exceeded max optical beams (%d)", CACAOSIM_BENCH_MAX_BEAMS);
        return -2;
    }
    int idx = (int) ctx->n_beams++;
    snprintf(ctx->beams[idx].name, sizeof(ctx->beams[idx].name), "%s", name);
    return idx;
}

static void init_node_imgids(cacaosim_bench_node_t *nd)
{
    nd->img_cmd.ID        = -1;
    nd->img_opd.ID        = -1;
    nd->img_flat.ID       = -1;
    nd->img_wfe.ID        = -1;
    nd->img_cam.ID        = -1;
    nd->img_long.ID       = -1;
    nd->img_ncpa.ID       = -1;
    nd->img_lyot.ID       = -1;
    nd->img_ncpa_refl.ID  = -1;
    nd->img_ncpa_trans.ID = -1;
    nd->img_pup_refl.ID   = -1;
    nd->img_pup_trans.ID  = -1;
}

static void free_imgid_if_valid(IMGID *img)
{
    if (img->ID != -1)
    {
        imgid_free(img);
        img->ID = -1;
    }
}

static const float *connect_opt_2df32(const char *name, uint32_t nwf, IMGID *out_img)
{
    out_img->ID = -1;
    if (name == NULL || name[0] == '\0')
    {
        return NULL;
    }
    *out_img = stream_connect(name);
    if (out_img->ID != -1 && out_img->md->datatype == _DATATYPE_FLOAT &&
        out_img->md->size[0] == nwf && out_img->md->size[1] == nwf)
    {
        return out_img->im->array.F;
    }
    return NULL;
}

static int load_opt_coro_mask(
    const char           *spec,
    int                   connect_shm,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap)
{
    memset(out, 0, sizeof(*out));
    if (spec == NULL || spec[0] == '\0')
    {
        return 0;
    }
    if (cacaosim_mask_is_fits_path(spec))
    {
        return cacaosim_mask_load(spec, 0, out, err_buf, err_cap);
    }
    if (connect_shm)
    {
        (void) cacaosim_mask_load(spec, 1, out, NULL, 0);
    }
    return 0;
}

static int upgrade_beam_poly(
    cacaosim_bench_ctx_t *ctx,
    cacaosim_beam_slot_t *b,
    uint32_t              nlam,
    const float          *lam_um,
    char                 *err_buf,
    size_t                err_cap)
{
    if (nlam < 1u)
    {
        nlam = 1u;
    }
    if (nlam > CACAOSIM_CORO_MAX_LAMBDA)
    {
        nlam = CACAOSIM_CORO_MAX_LAMBDA;
    }
    if (lam_um != NULL)
    {
        for (uint32_t c = 0; c < nlam; c++)
        {
            b->lam_um[c] = lam_um[c];
        }
    }
    if (nlam <= b->nlam)
    {
        return 0;
    }

    size_t tot = (size_t) nlam * ctx->npix_wf;
    free(b->opd);
    free(b->amp);
    b->opd = (float *) cacaosim_aligned_calloc(tot, sizeof(float));
    b->amp = (float *) cacaosim_aligned_calloc(tot, sizeof(float));
    if (!b->opd || !b->amp)
    {
        dag_err(err_buf, err_cap, "Out of memory allocating %u-lambda beam '%s'", nlam, b->name);
        return -1;
    }
    if (ctx->use_cuda)
    {
        cacaosim_cuda_free_f32(b->d_opd);
        cacaosim_cuda_free_f32(b->d_amp);
        b->d_opd = cacaosim_cuda_alloc_f32(tot);
        b->d_amp = cacaosim_cuda_alloc_f32(tot);
        if (!b->d_opd || !b->d_amp)
        {
            dag_err(err_buf, err_cap, "CUDA alloc failed for %u-lambda beam '%s'", nlam, b->name);
            return -1;
        }
    }
    b->nlam = nlam;
    return 0;
}

static int topo_sort_and_wire_beams(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap)
{
    const char *src_name = (cfg->out_beam[0] != '\0') ? cfg->out_beam : "b_pup";
    if (register_beam_name(ctx, src_name, err_buf, err_cap) < 0)
    {
        return -1;
    }

    int32_t beam_prod[CACAOSIM_BENCH_MAX_BEAMS];
    for (int i = 0; i < CACAOSIM_BENCH_MAX_BEAMS; i++)
    {
        beam_prod[i] = -1;
    }

    int32_t raw_out[CACAOSIM_BENCH_MAX_NODES];
    int32_t raw_refl[CACAOSIM_BENCH_MAX_NODES];
    int32_t raw_trans[CACAOSIM_BENCH_MAX_NODES];
    int32_t raw_lowfs[CACAOSIM_BENCH_MAX_NODES];

    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *nc = &cfg->nodes[i];
        raw_out[i]                    = -1;
        raw_refl[i]                   = -1;
        raw_trans[i]                  = -1;
        raw_lowfs[i]                  = -1;

        if (nc->in_beam[0] == '\0')
        {
            dag_err(err_buf, err_cap, "Node '%s' missing input beam ('in = ...')", nc->id);
            return -1;
        }

        if (nc->kind == CACAOSIM_NODE_DM || nc->kind == CACAOSIM_NODE_CORO ||
            nc->kind == CACAOSIM_NODE_WFE)
        {
            if (nc->out_beam[0] == '\0')
            {
                dag_err(err_buf, err_cap, "Node '%s' missing output beam ('out = ...')", nc->id);
                return -1;
            }
            int b = register_beam_name(ctx, nc->out_beam, err_buf, err_cap);
            if (b < 0)
            {
                return -1;
            }
            raw_out[i]   = b;
            beam_prod[b] = (int32_t) i;

            if (nc->kind == CACAOSIM_NODE_CORO && nc->out_lowfs[0] != '\0')
            {
                int bl = register_beam_name(ctx, nc->out_lowfs, err_buf, err_cap);
                if (bl < 0)
                {
                    return -1;
                }
                raw_lowfs[i]  = bl;
                beam_prod[bl] = (int32_t) i;
            }
        }
        else if (nc->kind == CACAOSIM_NODE_SPLITTER)
        {
            if (nc->out_refl[0] == '\0' && nc->out_trans[0] == '\0')
            {
                dag_err(err_buf, err_cap, "Splitter '%s' needs out_refl and/or out_trans", nc->id);
                return -1;
            }
            if (nc->out_refl[0] != '\0')
            {
                int br = register_beam_name(ctx, nc->out_refl, err_buf, err_cap);
                if (br < 0)
                {
                    return -1;
                }
                raw_refl[i]   = br;
                beam_prod[br] = (int32_t) i;
            }
            if (nc->out_trans[0] != '\0')
            {
                int bt = register_beam_name(ctx, nc->out_trans, err_buf, err_cap);
                if (bt < 0)
                {
                    return -1;
                }
                raw_trans[i]  = bt;
                beam_prod[bt] = (int32_t) i;
            }
        }
    }

    int32_t raw_in[CACAOSIM_BENCH_MAX_NODES];
    int32_t in_deg[CACAOSIM_BENCH_MAX_NODES];
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        int bi = find_beam_name(ctx, cfg->nodes[i].in_beam);
        if (bi < 0)
        {
            dag_err(err_buf, err_cap, "Node '%s' references unknown beam '%s'", cfg->nodes[i].id,
                    cfg->nodes[i].in_beam);
            return -1;
        }
        raw_in[i] = bi;
        in_deg[i] = (beam_prod[bi] >= 0) ? 1 : 0;
    }

    /* Kahn's topological sort preserving declaration order among ready nodes */
    uint8_t visited[CACAOSIM_BENCH_MAX_NODES] = { 0 };
    uint32_t sorted_cnt                       = 0;

    while (sorted_cnt < cfg->n_nodes)
    {
        int found = -1;
        for (uint32_t i = 0; i < cfg->n_nodes; i++)
        {
            if (!visited[i] && in_deg[i] == 0)
            {
                found = (int) i;
                break;
            }
        }
        if (found < 0)
        {
            dag_err(err_buf, err_cap, "Optical cycle detected in .bench beam graph");
            return -1;
        }

        visited[found]                 = 1;
        cacaosim_bench_node_t *dst     = &ctx->nodes[sorted_cnt];
        memset(dst, 0, sizeof(*dst));
        init_node_imgids(dst);
        dst->cfg         = cfg->nodes[found];
        dst->in_idx      = raw_in[found];
        dst->out_idx     = raw_out[found];
        dst->refl_idx    = raw_refl[found];
        dst->trans_idx   = raw_trans[found];
        dst->lowfs_idx   = raw_lowfs[found];
        dst->dm_slot     = -1;
        dst->sensor_slot = -1;
        sorted_cnt++;

        for (uint32_t j = 0; j < cfg->n_nodes; j++)
        {
            if (!visited[j] && beam_prod[raw_in[j]] == found)
            {
                in_deg[j]--;
            }
        }
    }

    ctx->n_nodes = sorted_cnt;
    return 0;
}

static void propagate_spectra(cacaosim_bench_ctx_t *ctx)
{
    cacaosim_spec_star(&ctx->beams[0].spec, ctx->cfg.tel_diam_m, ctx->cfg.cobs, ctx->cfg.star_mag,
                       ctx->cfg.star_teff, ctx->cfg.tput_tel);

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd      = &ctx->nodes[i];
        const cacaosim_spec_t *in_spec = &ctx->beams[nd->in_idx].spec;

        if (nd->cfg.kind == CACAOSIM_NODE_DM || nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            cacaosim_spec_scale(in_spec, nd->cfg.tput, &ctx->beams[nd->out_idx].spec);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            cacaosim_spec_t *r_sp = (nd->refl_idx >= 0) ? &ctx->beams[nd->refl_idx].spec : NULL;
            cacaosim_spec_t *t_sp = (nd->trans_idx >= 0) ? &ctx->beams[nd->trans_idx].spec : NULL;
            cacaosim_spec_split(in_spec, &nd->cfg.split, r_sp, t_sp);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            cacaosim_spec_scale(in_spec, nd->cfg.tput, &ctx->beams[nd->out_idx].spec);
            if (nd->lowfs_idx >= 0)
            {
                float r2 = nd->cfg.coro.lowfs_refl * nd->cfg.coro.lowfs_refl;
                cacaosim_spec_scale(in_spec, nd->cfg.tput * r2, &ctx->beams[nd->lowfs_idx].spec);
            }
        }
        else
        {
            float lam = 0.75f;
            float flx = 100000.0f;
            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                lam = nd->cfg.pyrwfs.lambda_um;
                flx = nd->cfg.pyrwfs.det.flux_ph;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                lam = nd->cfg.shwfs.lambda_um;
                flx = nd->cfg.shwfs.det.flux_ph;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                lam = nd->cfg.zwfs.lambda_um;
                flx = nd->cfg.zwfs.det.flux_ph;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                lam = nd->cfg.fpcam.lambda_um;
                flx = nd->cfg.fpcam.det.flux_ph;
            }

            float dt_exp      = (float) nd->eff_navg * ctx->dt_master_s;
            nd->eff_lambda_um = lam;
            if (nd->cfg.use_beam_flux && flx >= 0.0f)
            {
                nd->eff_flux_ph = cacaosim_spec_band_flux(in_spec, lam, nd->cfg.bw_frac, dt_exp,
                                                          nd->cfg.qe, &nd->eff_lambda_um);
            }
            else
            {
                nd->eff_flux_ph = flx;
            }
        }
    }
}

int cacaosim_bench_analyze(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap)
{
    if (ctx == NULL || cfg == NULL)
    {
        dag_err(err_buf, err_cap, "NULL bench context or config");
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->img_pup.ID  = -1;
    ctx->img_ncpa.ID = -1;
    ctx->img_wf.ID   = -1;
    ctx->img_amp.ID  = -1;
    ctx->cfg         = *cfg;
    ctx->nwf         = cfg->nwf;
    ctx->npix_wf     = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->dt_master_s = (cfg->master_hz > 1.0f) ? (1.0f / cfg->master_hz) : 0.0005f;

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    ctx->use_cuda = (want_cuda && cacaosim_cuda_available()) ? 1 : 0;

    if (topo_sort_and_wire_beams(ctx, cfg, err_buf, err_cap) != 0)
    {
        return -1;
    }

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        uint32_t navg             = (nd->cfg.navg >= 1u) ? nd->cfg.navg : 1u;
        if (nd->cfg.rate_hz > 0.1f && cfg->master_hz > 0.1f)
        {
            long r = lroundf(cfg->master_hz / nd->cfg.rate_hz);
            navg   = (r >= 1L) ? (uint32_t) r : 1u;
        }
        nd->eff_navg = navg;

        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            if (ctx->n_dms >= CACAOSIM_BENCH_MAX_DMS)
            {
                dag_err(err_buf, err_cap, "Exceeded max DMs (%d)", CACAOSIM_BENCH_MAX_DMS);
                return -1;
            }
            nd->dm_slot                    = (int32_t) ctx->n_dms;
            ctx->dm_node_idx[ctx->n_dms++] = (int32_t) i;
            if (nd->cfg.dm.dm_diam <= 0.0f)
            {
                nd->cfg.dm.dm_diam = cfg->pup_diam;
            }
            if (nd->cfg.dm.dm_mode == 1)
            {
                nd->cfg.dm.nact_x = 2u;
                nd->cfg.dm.nact_y = 1u;
            }
        }
        else if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS && nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
        {
            if (ctx->n_sensors >= CACAOSIM_BENCH_MAX_SENSORS)
            {
                dag_err(err_buf, err_cap, "Exceeded max sensors (%d)", CACAOSIM_BENCH_MAX_SENSORS);
                return -1;
            }
            nd->sensor_slot                        = (int32_t) ctx->n_sensors;
            ctx->sensor_node_idx[ctx->n_sensors++] = (int32_t) i;

            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                nd->ncam      = nd->cfg.pyrwfs.ncam;
                nd->out_dtype = nd->cfg.pyrwfs.det.out_dtype;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                nd->ncam      = (uint32_t) (nd->cfg.shwfs.n_sub * nd->cfg.shwfs.pix_per_sub);
                nd->out_dtype = nd->cfg.shwfs.det.out_dtype;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                nd->ncam      = nd->cfg.zwfs.ncam;
                nd->out_dtype = nd->cfg.zwfs.det.out_dtype;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                nd->ncam      = nd->cfg.fpcam.ncam;
                nd->out_dtype = nd->cfg.fpcam.det.out_dtype;
            }
        }
    }

    propagate_spectra(ctx);
    return 0;
}

int cacaosim_bench_init(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    const float                *ext_pup,
    int32_t                     connect_shm,
    char                       *err_buf,
    size_t                      err_cap)
{
    if (ctx == NULL || cfg == NULL)
    {
        dag_err(err_buf, err_cap, "NULL bench context or config");
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->img_pup.ID  = -1;
    ctx->img_ncpa.ID = -1;
    ctx->img_wf.ID   = -1;
    ctx->img_amp.ID  = -1;
    ctx->cfg         = *cfg;
    ctx->connect_shm = connect_shm;
    ctx->nwf         = cfg->nwf;
    ctx->npix_wf     = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->dt_master_s = (cfg->master_hz > 1.0f) ? (1.0f / cfg->master_hz) : 0.0005f;

    if (topo_sort_and_wire_beams(ctx, cfg, err_buf, err_cap) != 0)
    {
        return -1;
    }

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        ctx->cuda_stream = cacaosim_cuda_stream_create();
        if (ctx->cuda_stream != NULL)
        {
            ctx->use_cuda = 1;
        }
    }
    int32_t eff_backend = ctx->use_cuda ? CACAOSIM_BACKEND_CUDA : CACAOSIM_BACKEND_CPU;

    /* Entrance pupil and optional entrance NCPA / atmosphere streams */
    if (ext_pup != NULL)
    {
        ctx->pup_ptr = ext_pup;
    }
    else if (connect_shm && cfg->in_pup[0] != '\0')
    {
        ctx->pup_ptr = connect_opt_2df32(cfg->in_pup, ctx->nwf, &ctx->img_pup);
    }

    if (ctx->pup_ptr == NULL)
    {
        ctx->fallback_pup = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
        if (ctx->fallback_pup == NULL)
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "Out of memory allocating pupil");
            return -1;
        }
        cacaosim_pupil_cfg_t pcfg = {
            .nwf        = ctx->nwf,
            .pup_diam   = cfg->pup_diam,
            .cobs       = cfg->cobs,
            .nspider    = cfg->nspider,
            .spider_w   = cfg->spider_w,
            .spider_deg = cfg->spider_deg,
        };
        cacaosim_make_pupil(&pcfg, ctx->fallback_pup);
        ctx->pup_ptr = ctx->fallback_pup;
    }

    if (connect_shm)
    {
        ctx->ncpa_ptr = connect_opt_2df32(cfg->in_ncpa, ctx->nwf, &ctx->img_ncpa);
        if (cfg->in_wf[0] != '\0')
        {
            ctx->img_wf = stream_connect(cfg->in_wf);
        }
        if (cfg->in_amp[0] != '\0')
        {
            ctx->img_amp = stream_connect(cfg->in_amp);
        }
    }

    if (ctx->use_cuda)
    {
        ctx->d_pup_amp = cacaosim_cuda_alloc_f32(ctx->npix_wf);
        ctx->d_wf_in   = cacaosim_cuda_alloc_f32(ctx->npix_wf);
        ctx->d_atm_amp = cacaosim_cuda_alloc_f32(ctx->npix_wf);
        if (!ctx->d_pup_amp || !ctx->d_wf_in || !ctx->d_atm_amp)
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "CUDA allocation failed for entrance buffers");
            return -1;
        }
        cacaosim_cuda_h2d_f32(ctx->d_pup_amp, ctx->pup_ptr, ctx->npix_wf, ctx->cuda_stream);
        if (ctx->ncpa_ptr != NULL)
        {
            ctx->d_ncpa_opd = cacaosim_cuda_alloc_f32(ctx->npix_wf);
            cacaosim_cuda_h2d_f32(ctx->d_ncpa_opd, ctx->ncpa_ptr, ctx->npix_wf, ctx->cuda_stream);
        }
    }

    /* Allocate named optical beam buffers */
    for (uint32_t b = 0; b < ctx->n_beams; b++)
    {
        ctx->beams[b].nlam = 1u;
        ctx->beams[b].opd  = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
        ctx->beams[b].amp  = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
        if (!ctx->beams[b].opd || !ctx->beams[b].amp)
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "Out of memory allocating beam '%s'", ctx->beams[b].name);
            return -1;
        }
        if (ctx->use_cuda)
        {
            ctx->beams[b].d_opd = cacaosim_cuda_alloc_f32(ctx->npix_wf);
            ctx->beams[b].d_amp = cacaosim_cuda_alloc_f32(ctx->npix_wf);
            if (!ctx->beams[b].d_opd || !ctx->beams[b].d_amp)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "CUDA alloc failed for beam '%s'", ctx->beams[b].name);
                return -1;
            }
        }
    }

    /* Resolve sensor cadences and precompute steady-state spectral photometry */
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        uint32_t navg             = (nd->cfg.navg >= 1u) ? nd->cfg.navg : 1u;
        if (nd->cfg.rate_hz > 0.1f && cfg->master_hz > 0.1f)
        {
            long r = lroundf(cfg->master_hz / nd->cfg.rate_hz);
            navg   = (r >= 1L) ? (uint32_t) r : 1u;
        }
        nd->eff_navg = navg;
    }
    propagate_spectra(ctx);

    /* Initialize each compute unit in topological order */
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];

        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            if (ctx->n_dms >= CACAOSIM_BENCH_MAX_DMS)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Exceeded max DMs (%d)", CACAOSIM_BENCH_MAX_DMS);
                return -1;
            }
            nd->dm_slot                    = (int32_t) ctx->n_dms;
            ctx->dm_node_idx[ctx->n_dms++] = (int32_t) i;

            nd->cfg.dm.nwf     = ctx->nwf;
            nd->cfg.dm.dt_sec  = ctx->dt_master_s;
            nd->cfg.dm.backend = eff_backend;
            if (nd->cfg.dm.dm_diam <= 0.0f)
            {
                nd->cfg.dm.dm_diam = cfg->pup_diam;
            }
            if (nd->cfg.dm.dm_mode == 1)
            {
                nd->cfg.dm.nact_x = 2u;
                nd->cfg.dm.nact_y = 1u;
            }

            if (connect_shm)
            {
                nd->img_cmd = stream_connect_create_2Df32(nd->cfg.in_cmd, nd->cfg.dm.nact_x,
                                                          nd->cfg.dm.nact_y);
                if (nd->img_cmd.ID == -1)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "Failed creating DM stream '%s'", nd->cfg.in_cmd);
                    return -1;
                }
                nd->cfg.dm.nact_x = nd->img_cmd.md->size[0];
                nd->cfg.dm.nact_y = nd->img_cmd.md->size[1];

                if (nd->cfg.out_opd[0] != '\0')
                {
                    nd->img_opd = stream_connect_create_2Df32(nd->cfg.out_opd, ctx->nwf, ctx->nwf);
                }
                connect_opt_2df32(nd->cfg.in_flat, ctx->nwf, &nd->img_flat);
            }
            else
            {
                size_t nact = (size_t) nd->cfg.dm.nact_x * nd->cfg.dm.nact_y;
                nd->cmd_buf = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
            }

            if (!ctx->use_cuda)
            {
                nd->dm_opd_buf = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
            }

            if (cacaosim_dm_init(&nd->dm_ctx, &nd->cfg.dm, ctx->cuda_stream) != 0)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Failed initializing DM '%s'", nd->cfg.id);
                return -1;
            }
            const cacaosim_beam_slot_t *bin_dm = &ctx->beams[nd->in_idx];
            if (bin_dm->nlam > 1u &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], bin_dm->nlam, bin_dm->lam_um,
                                  err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            if (connect_shm)
            {
                nd->h_ncpa_refl =
                    connect_opt_2df32(nd->cfg.ncpa_refl, ctx->nwf, &nd->img_ncpa_refl);
                nd->h_ncpa_trans =
                    connect_opt_2df32(nd->cfg.ncpa_trans, ctx->nwf, &nd->img_ncpa_trans);
                nd->h_pup_refl  = connect_opt_2df32(nd->cfg.pup_refl, ctx->nwf, &nd->img_pup_refl);
                nd->h_pup_trans =
                    connect_opt_2df32(nd->cfg.pup_trans, ctx->nwf, &nd->img_pup_trans);
            }
            if (ctx->use_cuda)
            {
                if (nd->h_ncpa_refl)
                {
                    nd->d_ncpa_refl = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_ncpa_refl, nd->h_ncpa_refl, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
                if (nd->h_ncpa_trans)
                {
                    nd->d_ncpa_trans = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_ncpa_trans, nd->h_ncpa_trans, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
                if (nd->h_pup_refl)
                {
                    nd->d_pup_refl = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_pup_refl, nd->h_pup_refl, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
                if (nd->h_pup_trans)
                {
                    nd->d_pup_trans = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_pup_trans, nd->h_pup_trans, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
            }
            const cacaosim_beam_slot_t *bin_sp = &ctx->beams[nd->in_idx];
            if (bin_sp->nlam > 1u)
            {
                if (nd->refl_idx >= 0 &&
                    upgrade_beam_poly(ctx, &ctx->beams[nd->refl_idx], bin_sp->nlam, bin_sp->lam_um,
                                      err_buf, err_cap) != 0)
                {
                    cacaosim_bench_free(ctx);
                    return -1;
                }
                if (nd->trans_idx >= 0 &&
                    upgrade_beam_poly(ctx, &ctx->beams[nd->trans_idx], bin_sp->nlam, bin_sp->lam_um,
                                      err_buf, err_cap) != 0)
                {
                    cacaosim_bench_free(ctx);
                    return -1;
                }
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            nd->cfg.coro.nwf      = ctx->nwf;
            nd->cfg.coro.pup_diam = cfg->pup_diam;
            nd->cfg.coro.backend  = eff_backend;
            if (ctx->beams[nd->in_idx].nlam > 1u && nd->cfg.coro.nlambda <= 1)
            {
                nd->cfg.coro.nlambda = (int32_t) ctx->beams[nd->in_idx].nlam;
            }

            cacaosim_coro_masks_t masks = { 0 };
            int                   m_rc  = 0;
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_pup_amp, connect_shm, &masks.pup_amp, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_pup_pha, connect_shm, &masks.pup_pha, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_fpm_amp, connect_shm, &masks.fpm_amp, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_fpm_pha, connect_shm, &masks.fpm_pha, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_lyot, connect_shm, &masks.lyot_amp, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_lyot_pha, connect_shm, &masks.lyot_pha,
                                          err_buf, err_cap);
            }

            int init_rc = (m_rc == 0) ? cacaosim_coro_init_masks(&nd->coro_ctx, &nd->cfg.coro,
                                                                 &masks, ctx->cuda_stream)
                                      : -1;
            cacaosim_mask_free(&masks.pup_amp);
            cacaosim_mask_free(&masks.pup_pha);
            cacaosim_mask_free(&masks.fpm_amp);
            cacaosim_mask_free(&masks.fpm_pha);
            cacaosim_mask_free(&masks.lyot_amp);
            cacaosim_mask_free(&masks.lyot_pha);

            if (init_rc != 0)
            {
                cacaosim_bench_free(ctx);
                if (m_rc == 0)
                {
                    dag_err(err_buf, err_cap, "Failed initializing coronagraph '%s'", nd->cfg.id);
                }
                return -1;
            }
            if (nd->coro_ctx.eff_nlam > 1 &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], (uint32_t) nd->coro_ctx.eff_nlam,
                                  nd->coro_ctx.lam_um, err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            float rms_m = (nd->cfg.wfe_rms_nm > 0.0f) ? (nd->cfg.wfe_rms_nm * 1.0e-9f) : 0.0f;
            if (connect_shm)
            {
                nd->img_wfe = stream_connect(nd->cfg.in_wfe);
                if (nd->img_wfe.ID == -1 || nd->img_wfe.md->datatype != _DATATYPE_FLOAT ||
                    nd->img_wfe.md->size[0] != ctx->nwf || nd->img_wfe.md->size[1] != ctx->nwf)
                {
                    free_imgid_if_valid(&nd->img_wfe);
                    nd->img_wfe =
                        stream_connect_create_2Df32(nd->cfg.in_wfe, ctx->nwf, ctx->nwf);
                    if (nd->img_wfe.ID == -1)
                    {
                        cacaosim_bench_free(ctx);
                        dag_err(err_buf, err_cap, "Failed creating WFE stream '%s'",
                                nd->cfg.in_wfe);
                        return -1;
                    }
                    nd->img_wfe.md->write = 1;
                    cacaosim_make_wfe_m(ctx->nwf, cfg->pup_diam, ctx->pup_ptr, rms_m,
                                        nd->cfg.wfe_alpha, nd->cfg.wfe_seed,
                                        nd->img_wfe.im->array.F);
                    if (nd->img_wfe.im->kw != NULL && nd->img_wfe.md->NBkw >= 1)
                    {
                        snprintf(nd->img_wfe.im->kw[0].name, KEYWORD_MAX_STRING, "BUNIT");
                        nd->img_wfe.im->kw[0].type = 'S';
                        snprintf(nd->img_wfe.im->kw[0].value.valstr, KEYWORD_MAX_STRING, "m");
                        snprintf(nd->img_wfe.im->kw[0].comment, KEYWORD_MAX_COMMENT,
                                 "Static wavefront error OPD [meter]");
                    }
                    ImageStreamIO_UpdateIm(nd->img_wfe.im);
                }
                nd->wfe_cnt0 = nd->img_wfe.md->cnt0;
            }
            else
            {
                nd->wfe_m_buf = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
                if (nd->wfe_m_buf == NULL)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "Out of memory allocating WFE map '%s'", nd->cfg.id);
                    return -1;
                }
                cacaosim_make_wfe_m(ctx->nwf, cfg->pup_diam, ctx->pup_ptr, rms_m,
                                    nd->cfg.wfe_alpha, nd->cfg.wfe_seed, nd->wfe_m_buf);
            }

            if (ctx->use_cuda)
            {
                const float *src_wfe = (connect_shm && nd->img_wfe.ID != -1)
                                           ? nd->img_wfe.im->array.F
                                           : nd->wfe_m_buf;
                nd->d_wfe_m          = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                if (nd->d_wfe_m == NULL)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "CUDA alloc failed for WFE map '%s'", nd->cfg.id);
                    return -1;
                }
                cacaosim_cuda_h2d_f32(nd->d_wfe_m, src_wfe, ctx->npix_wf, ctx->cuda_stream);
            }

            const cacaosim_beam_slot_t *bin_wfe = &ctx->beams[nd->in_idx];
            if (bin_wfe->nlam > 1u &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], bin_wfe->nlam, bin_wfe->lam_um,
                                  err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
        }
        else
        {
            if (ctx->n_sensors >= CACAOSIM_BENCH_MAX_SENSORS)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Exceeded max sensors (%d)", CACAOSIM_BENCH_MAX_SENSORS);
                return -1;
            }
            nd->sensor_slot                        = (int32_t) ctx->n_sensors;
            ctx->sensor_node_idx[ctx->n_sensors++] = (int32_t) i;

            const float *s_ncpa = NULL;
            if (connect_shm)
            {
                s_ncpa = connect_opt_2df32(nd->cfg.in_ncpa, ctx->nwf, &nd->img_ncpa);
            }

            const cacaosim_beam_slot_t *bin_sens = &ctx->beams[nd->in_idx];
            uint32_t in_nlam                     = (bin_sens->nlam >= 1u) ? bin_sens->nlam : 1u;

            nd->has_reg = !cacaosim_reg_is_identity(&nd->cfg.reg);
            if (nd->has_reg)
            {
                size_t tot_reg = (size_t) in_nlam * ctx->npix_wf;
                nd->reg_opd    = (float *) cacaosim_aligned_calloc(tot_reg, sizeof(float));
                nd->reg_amp    = (float *) cacaosim_aligned_calloc(tot_reg, sizeof(float));
                if (ctx->use_cuda)
                {
                    nd->d_reg_opd = cacaosim_cuda_alloc_f32(tot_reg);
                    nd->d_reg_amp = cacaosim_cuda_alloc_f32(tot_reg);
                }
            }

            int rc = -1;
            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                nd->cfg.pyrwfs.nwf          = ctx->nwf;
                nd->cfg.pyrwfs.pup_diam     = cfg->pup_diam;
                nd->cfg.pyrwfs.wf_unit      = 0;
                nd->cfg.pyrwfs.backend      = eff_backend;
                nd->cfg.pyrwfs.det.flux_ph  = nd->eff_flux_ph;
                nd->ncam                    = nd->cfg.pyrwfs.ncam;
                nd->out_dtype               = nd->cfg.pyrwfs.det.out_dtype;
                rc = cacaosim_pyrwfs_init(&nd->pyrwfs_ctx, &nd->cfg.pyrwfs, ctx->pup_ptr, s_ncpa,
                                          ctx->cuda_stream);
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                nd->cfg.shwfs.nwf          = ctx->nwf;
                nd->cfg.shwfs.pup_diam     = cfg->pup_diam;
                nd->cfg.shwfs.wf_unit      = 0;
                nd->cfg.shwfs.backend      = eff_backend;
                nd->cfg.shwfs.det.flux_ph  = nd->eff_flux_ph;
                nd->ncam                   = nd->cfg.shwfs.n_sub * nd->cfg.shwfs.pix_per_sub;
                nd->out_dtype              = nd->cfg.shwfs.det.out_dtype;
                rc = cacaosim_shwfs_init(&nd->shwfs_ctx, &nd->cfg.shwfs, ctx->pup_ptr, s_ncpa,
                                         ctx->cuda_stream);
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                nd->cfg.zwfs.nwf          = ctx->nwf;
                nd->cfg.zwfs.pup_diam     = cfg->pup_diam;
                nd->cfg.zwfs.wf_unit      = 0;
                nd->cfg.zwfs.backend      = eff_backend;
                nd->cfg.zwfs.det.flux_ph  = nd->eff_flux_ph;
                nd->ncam                  = nd->cfg.zwfs.ncam;
                nd->out_dtype             = nd->cfg.zwfs.det.out_dtype;
                rc = cacaosim_zwfs_init(&nd->zwfs_ctx, &nd->cfg.zwfs, ctx->pup_ptr, s_ncpa,
                                        ctx->cuda_stream);
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                nd->cfg.fpcam.nwf          = ctx->nwf;
                nd->cfg.fpcam.pup_diam     = cfg->pup_diam;
                nd->cfg.fpcam.wf_unit      = 0;
                nd->cfg.fpcam.backend      = eff_backend;
                nd->cfg.fpcam.det.flux_ph  = nd->eff_flux_ph;
                nd->ncam                   = nd->cfg.fpcam.ncam;
                nd->out_dtype              = nd->cfg.fpcam.det.out_dtype;
                if (in_nlam > 1u)
                {
                    nd->cfg.fpcam.has_ext_lam = 1;
                    nd->cfg.fpcam.nlambda     = (int32_t) in_nlam;
                    for (uint32_t c = 0; c < in_nlam; c++)
                    {
                        nd->cfg.fpcam.ext_lam_um[c] = bin_sens->lam_um[c];
                    }
                }
                rc = cacaosim_fpcam_init(&nd->fpcam_ctx, &nd->cfg.fpcam, ctx->pup_ptr, s_ncpa,
                                         ctx->cuda_stream);
            }

            if (rc != 0)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Failed initializing sensor '%s'", nd->cfg.id);
                return -1;
            }

            size_t npix_cam = (size_t) nd->ncam * nd->ncam;
            if (connect_shm)
            {
                uint8_t dt  = (nd->out_dtype == 0) ? _DATATYPE_UINT16 : _DATATYPE_FLOAT;
                nd->img_cam = stream_connect_create_2D(nd->cfg.out_cam, nd->ncam, nd->ncam, dt);
                if (nd->img_cam.ID == -1)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "Failed creating camera stream '%s'",
                            nd->cfg.out_cam);
                    return -1;
                }
                if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && nd->cfg.out_long[0] != '\0')
                {
                    nd->img_long =
                        stream_connect_create_2Df32(nd->cfg.out_long, nd->ncam, nd->ncam);
                }
            }
            else
            {
                nd->cam_u16 = (uint16_t *) cacaosim_aligned_calloc(npix_cam, sizeof(uint16_t));
                nd->cam_f32 = (float *) cacaosim_aligned_calloc(npix_cam, sizeof(float));
                if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
                {
                    nd->long_f32 = (float *) cacaosim_aligned_calloc(npix_cam, sizeof(float));
                }
            }
        }
    }

    if (ctx->use_cuda)
    {
        cacaosim_cuda_stream_sync(ctx->cuda_stream);
    }
    return 0;
}

void cacaosim_bench_free(cacaosim_bench_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            cacaosim_dm_free(&nd->dm_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            cacaosim_coro_free(&nd->coro_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
        {
            cacaosim_pyrwfs_free(&nd->pyrwfs_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
        {
            cacaosim_shwfs_free(&nd->shwfs_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
        {
            cacaosim_zwfs_free(&nd->zwfs_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
        {
            cacaosim_fpcam_free(&nd->fpcam_ctx);
        }

        free(nd->reg_opd);
        free(nd->reg_amp);
        free(nd->cmd_buf);
        free(nd->dm_opd_buf);
        free(nd->wfe_m_buf);
        free(nd->cam_u16);
        free(nd->cam_f32);
        free(nd->long_f32);

        cacaosim_cuda_free_f32(nd->d_wfe_m);
        cacaosim_cuda_free_f32(nd->d_reg_opd);
        cacaosim_cuda_free_f32(nd->d_reg_amp);
        cacaosim_cuda_free_f32(nd->d_ncpa_refl);
        cacaosim_cuda_free_f32(nd->d_ncpa_trans);
        cacaosim_cuda_free_f32(nd->d_pup_refl);
        cacaosim_cuda_free_f32(nd->d_pup_trans);

        free_imgid_if_valid(&nd->img_cmd);
        free_imgid_if_valid(&nd->img_opd);
        free_imgid_if_valid(&nd->img_flat);
        free_imgid_if_valid(&nd->img_wfe);
        free_imgid_if_valid(&nd->img_cam);
        free_imgid_if_valid(&nd->img_long);
        free_imgid_if_valid(&nd->img_ncpa);
        free_imgid_if_valid(&nd->img_lyot);
        free_imgid_if_valid(&nd->img_ncpa_refl);
        free_imgid_if_valid(&nd->img_ncpa_trans);
        free_imgid_if_valid(&nd->img_pup_refl);
        free_imgid_if_valid(&nd->img_pup_trans);
    }

    for (uint32_t b = 0; b < ctx->n_beams; b++)
    {
        free(ctx->beams[b].opd);
        free(ctx->beams[b].amp);
        cacaosim_cuda_free_f32(ctx->beams[b].d_opd);
        cacaosim_cuda_free_f32(ctx->beams[b].d_amp);
    }

    cacaosim_cuda_free_f32(ctx->d_pup_amp);
    cacaosim_cuda_free_f32(ctx->d_ncpa_opd);
    cacaosim_cuda_free_f32(ctx->d_wf_in);
    cacaosim_cuda_free_f32(ctx->d_atm_amp);
    cacaosim_cuda_stream_destroy(ctx->cuda_stream);

    free(ctx->fallback_pup);
    free_imgid_if_valid(&ctx->img_pup);
    free_imgid_if_valid(&ctx->img_ncpa);
    free_imgid_if_valid(&ctx->img_wf);
    free_imgid_if_valid(&ctx->img_amp);
    memset(ctx, 0, sizeof(*ctx));
}

static const char *node_kind_str(cacaosim_node_kind_t k, int dm_mode)
{
    switch (k)
    {
    case CACAOSIM_NODE_DM:
        return (dm_mode == 1) ? "TTM" : "DM";
    case CACAOSIM_NODE_SPLITTER:
        return "SPLITTER";
    case CACAOSIM_NODE_CORO:
        return "CORO";
    case CACAOSIM_NODE_PYRWFS:
        return "PYRWFS";
    case CACAOSIM_NODE_SHWFS:
        return "SHWFS";
    case CACAOSIM_NODE_ZWFS:
        return "ZWFS";
    case CACAOSIM_NODE_FPCAM:
        return "FPCAM";
    case CACAOSIM_NODE_WFE:
        return "WFE";
    default:
        return "UNKNOWN";
    }
}

void cacaosim_bench_print_summary(const cacaosim_bench_ctx_t *ctx, FILE *fp)
{
    if (ctx == NULL || fp == NULL)
    {
        return;
    }
    fprintf(fp, "=== cacaosim Bench '%s' (%ux%u, D=%.2fm, mV=%.1f, %.0f Hz, %s) ===\n",
            ctx->cfg.name, ctx->nwf, ctx->nwf, ctx->cfg.tel_diam_m, ctx->cfg.star_mag,
            ctx->cfg.master_hz, ctx->use_cuda ? "CUDA" : "CPU");
    fprintf(fp, "  Entrance beam '%s': %.3e ph/s (0.40..2.40 um)\n", ctx->beams[0].name,
            cacaosim_spec_total(&ctx->beams[0].spec));

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[i];
        const char                  *ks = node_kind_str(nd->cfg.kind, nd->cfg.dm.dm_mode);
        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> %s (%ux%u, cmd=%s)\n", i, ks, nd->cfg.id,
                    nd->cfg.in_beam, nd->cfg.out_beam, nd->cfg.dm.nact_x, nd->cfg.dm.nact_y,
                    nd->cfg.in_cmd);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> %s (wfe=%s [m], rms=%.1fnm, gain=%.2f)\n",
                    i, ks, nd->cfg.id, nd->cfg.in_beam, nd->cfg.out_beam, nd->cfg.in_wfe,
                    nd->cfg.wfe_rms_nm, nd->cfg.wfe_gain);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> refl=%s, trans=%s (cut=%.2fum)\n", i, ks,
                    nd->cfg.id, nd->cfg.in_beam,
                    nd->cfg.out_refl[0] ? nd->cfg.out_refl : "-",
                    nd->cfg.out_trans[0] ? nd->cfg.out_trans : "-", nd->cfg.split.lambda_cut);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> %s (lowfs=%s [%s], lam=%.2fum, nlam=%d)\n", i,
                    ks, nd->cfg.id, nd->cfg.in_beam, nd->cfg.out_beam,
                    nd->cfg.out_lowfs[0] ? nd->cfg.out_lowfs : "-",
                    (nd->cfg.coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? "lyot_refl"
                                                                          : "fpm_refl",
                    nd->cfg.coro.lambda_um,
                    (nd->coro_ctx.eff_nlam > 1) ? nd->coro_ctx.eff_nlam : 1);
        }
        else
        {
            float hz = ctx->cfg.master_hz / (float) nd->eff_navg;
            fprintf(fp,
                    "  [%u] %-8s %-12s : in=%s -> %s (%ux%u, %.2fum, %.0fHz navg=%u, %.2e ph/fr)\n",
                    i, ks, nd->cfg.id, nd->cfg.in_beam, nd->cfg.out_cam, nd->ncam, nd->ncam,
                    nd->eff_lambda_um, hz, nd->eff_navg, nd->eff_flux_ph);
        }
    }
}

MILK_HOT void cacaosim_bench_step(
    cacaosim_bench_ctx_t *ctx,
    const float          *ext_wf,
    const float          *ext_amp,
    float                 turb_gain,
    void                 *procinfo)
{
    if (ctx == NULL)
    {
        return;
    }

    const float *wf_ptr  = ext_wf;
    const float *amp_ptr = ext_amp;

    if (ctx->connect_shm)
    {
        if (ctx->img_wf.ID == -1 && ctx->cfg.in_wf[0] != '\0')
        {
            ctx->img_wf = stream_connect(ctx->cfg.in_wf);
        }
        if (ctx->img_amp.ID == -1 && ctx->cfg.in_amp[0] != '\0')
        {
            ctx->img_amp = stream_connect(ctx->cfg.in_amp);
        }
        if (wf_ptr == NULL && ctx->img_wf.ID != -1 && ctx->img_wf.md->datatype == _DATATYPE_FLOAT &&
            ctx->img_wf.md->size[0] == ctx->nwf)
        {
            wf_ptr = ctx->img_wf.im->array.F;
        }
        if (amp_ptr == NULL && ctx->img_amp.ID != -1 &&
            ctx->img_amp.md->datatype == _DATATYPE_FLOAT && ctx->img_amp.md->size[0] == ctx->nwf)
        {
            amp_ptr = ctx->img_amp.im->array.F;
        }
    }

    float wf_scale = (ctx->cfg.wf_unit == 1)
                         ? (turb_gain * ctx->cfg.ref_lambda / (2.0f * CACAOSIM_PI_F))
                         : turb_gain;

    /* 1. Synthesize entrance beam [OPD_um, Amplitude] */
    cacaosim_beam_slot_t *b0 = &ctx->beams[0];
    if (ctx->use_cuda)
    {
        const float *d_wf  = NULL;
        const float *d_amp = NULL;
        if (wf_ptr != NULL && fabsf(wf_scale) > 0.0f)
        {
            cacaosim_cuda_h2d_f32(ctx->d_wf_in, wf_ptr, ctx->npix_wf, ctx->cuda_stream);
            d_wf = ctx->d_wf_in;
        }
        if (amp_ptr != NULL)
        {
            cacaosim_cuda_h2d_f32(ctx->d_atm_amp, amp_ptr, ctx->npix_wf, ctx->cuda_stream);
            d_amp = ctx->d_atm_amp;
        }
        cacaosim_cuda_beam_init(ctx->d_pup_amp, d_amp, d_wf, ctx->d_ncpa_opd, wf_scale, ctx->nwf,
                                b0->d_opd, b0->d_amp, ctx->cuda_stream);
    }
    else
    {
        cacaosim_beam_init_cpu(ctx->pup_ptr, amp_ptr, wf_ptr, ctx->ncpa_ptr, wf_scale, ctx->nwf,
                               b0->opd, b0->amp);
    }

    IMAGE       *trig_im = (ctx->connect_shm && ctx->n_dms > 0 &&
                            ctx->nodes[ctx->dm_node_idx[0]].img_cmd.ID != -1)
                               ? ctx->nodes[ctx->dm_node_idx[0]].img_cmd.im
                               : NULL;
    PROCESSINFO *pinfo   = (PROCESSINFO *) procinfo;

    /* 2. Traverse optical DAG in topological order */
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t      *nd  = &ctx->nodes[i];
        const cacaosim_beam_slot_t *bin = &ctx->beams[nd->in_idx];
        nd->ready_this_tick             = 0;

        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            cacaosim_beam_slot_t *bout = &ctx->beams[nd->out_idx];
            const float *cmd_in =
                (ctx->connect_shm && nd->img_cmd.ID != -1) ? nd->img_cmd.im->array.F : nd->cmd_buf;
            const float *flat_in =
                (ctx->connect_shm && nd->img_flat.ID != -1) ? nd->img_flat.im->array.F : NULL;

            float *host_opd = (ctx->connect_shm && nd->img_opd.ID != -1) ? nd->img_opd.im->array.F
                                                                         : nd->dm_opd_buf;
            if (ctx->connect_shm && nd->img_opd.ID != -1)
            {
                nd->img_opd.md->write = 1;
            }

            cacaosim_dm_step(&nd->dm_ctx, cmd_in, flat_in, nd->cfg.dm.delay_frames,
                             nd->cfg.dm.tau_sec, nd->cfg.dm.max_stroke, host_opd);

            if (ctx->connect_shm && nd->img_opd.ID != -1)
            {
                processinfo_update_output_stream(pinfo, nd->img_opd.im, nd->img_cmd.im);
            }

            uint32_t nl = (bin->nlam >= 1u) ? bin->nlam : 1u;
            for (uint32_t c = 0; c < nl; c++)
            {
                size_t off = (size_t) c * ctx->npix_wf;
                if (ctx->use_cuda)
                {
                    cacaosim_cuda_beam_apply_dm(bin->d_opd + off, bin->d_amp + off,
                                                nd->dm_ctx.d_opd_out, ctx->nwf, bout->d_opd + off,
                                                bout->d_amp + off, ctx->cuda_stream);
                }
                else
                {
                    cacaosim_beam_apply_dm_cpu(bin->opd + off, bin->amp + off, host_opd, ctx->nwf,
                                               bout->opd + off, bout->amp + off);
                }
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            uint32_t nl = (bin->nlam >= 1u) ? bin->nlam : 1u;
            if (nd->refl_idx >= 0)
            {
                cacaosim_beam_slot_t *br = &ctx->beams[nd->refl_idx];
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_relay(bin->d_opd + off, bin->d_amp + off,
                                                 nd->d_ncpa_refl, nd->d_pup_refl,
                                                 &nd->cfg.reg_refl, ctx->nwf, br->d_opd + off,
                                                 br->d_amp + off, ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_relay_cpu(bin->opd + off, bin->amp + off, nd->h_ncpa_refl,
                                                nd->h_pup_refl, &nd->cfg.reg_refl, ctx->nwf,
                                                br->opd + off, br->amp + off);
                    }
                }
            }
            if (nd->trans_idx >= 0)
            {
                cacaosim_beam_slot_t *bt = &ctx->beams[nd->trans_idx];
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_relay(bin->d_opd + off, bin->d_amp + off,
                                                 nd->d_ncpa_trans, nd->d_pup_trans,
                                                 &nd->cfg.reg_trans, ctx->nwf, bt->d_opd + off,
                                                 bt->d_amp + off, ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_relay_cpu(bin->opd + off, bin->amp + off, nd->h_ncpa_trans,
                                                nd->h_pup_trans, &nd->cfg.reg_trans, ctx->nwf,
                                                bt->opd + off, bt->amp + off);
                    }
                }
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            cacaosim_beam_slot_t *bout = &ctx->beams[nd->out_idx];
            cacaosim_beam_slot_t *blow = (nd->lowfs_idx >= 0) ? &ctx->beams[nd->lowfs_idx] : NULL;
            uint32_t              nl   = (bin->nlam >= 1u) ? bin->nlam : 1u;

            cacaosim_coro_step_poly(&nd->coro_ctx, bin->opd, bin->amp, nl, ctx->npix_wf, bout->opd,
                                    bout->amp, ctx->npix_wf, blow ? blow->opd : NULL,
                                    blow ? blow->amp : NULL, bin->d_opd, bin->d_amp, bout->d_opd,
                                    bout->d_amp, blow ? blow->d_opd : NULL,
                                    blow ? blow->d_amp : NULL);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            cacaosim_beam_slot_t *bout  = &ctx->beams[nd->out_idx];
            const float          *h_wfe = (ctx->connect_shm && nd->img_wfe.ID != -1)
                                              ? nd->img_wfe.im->array.F
                                              : nd->wfe_m_buf;
            if (ctx->use_cuda && ctx->connect_shm && nd->img_wfe.ID != -1 &&
                nd->img_wfe.md->cnt0 != nd->wfe_cnt0)
            {
                cacaosim_cuda_h2d_f32(nd->d_wfe_m, h_wfe, ctx->npix_wf, ctx->cuda_stream);
                nd->wfe_cnt0 = nd->img_wfe.md->cnt0;
            }

            float    wfe_to_um = 1.0e6f * nd->cfg.wfe_gain;
            uint32_t nl        = (bin->nlam >= 1u) ? bin->nlam : 1u;
            for (uint32_t c = 0; c < nl; c++)
            {
                size_t off = (size_t) c * ctx->npix_wf;
                if (ctx->use_cuda)
                {
                    cacaosim_cuda_beam_apply_wfe(bin->d_opd + off, bin->d_amp + off, nd->d_wfe_m,
                                                 wfe_to_um, &nd->cfg.reg, ctx->nwf,
                                                 bout->d_opd + off, bout->d_amp + off,
                                                 ctx->cuda_stream);
                }
                else
                {
                    cacaosim_beam_apply_wfe_cpu(bin->opd + off, bin->amp + off, h_wfe, wfe_to_um,
                                                &nd->cfg.reg, ctx->nwf, bout->opd + off,
                                                bout->amp + off);
                }
            }
        }
        else
        {
            const float *h_opd = bin->opd;
            const float *h_amp = bin->amp;
            const float *d_opd = bin->d_opd;
            const float *d_amp = bin->d_amp;
            uint32_t     nl    = (bin->nlam >= 1u) ? bin->nlam : 1u;

            if (nd->has_reg)
            {
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_relay(bin->d_opd + off, bin->d_amp + off, NULL, NULL,
                                                 &nd->cfg.reg, ctx->nwf, nd->d_reg_opd + off,
                                                 nd->d_reg_amp + off, ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_relay_cpu(bin->opd + off, bin->amp + off, NULL, NULL,
                                                &nd->cfg.reg, ctx->nwf, nd->reg_opd + off,
                                                nd->reg_amp + off);
                    }
                }
                if (ctx->use_cuda)
                {
                    d_opd = nd->d_reg_opd;
                    d_amp = nd->d_reg_amp;
                }
                else
                {
                    h_opd = nd->reg_opd;
                    h_amp = nd->reg_amp;
                }
            }

            uint16_t *dst_u16 = (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype == 0)
                                    ? nd->img_cam.im->array.UI16
                                    : nd->cam_u16;
            float    *dst_f32 = (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype != 0)
                                    ? nd->img_cam.im->array.F
                                    : nd->cam_f32;
            const float *s_ncpa =
                (ctx->connect_shm && nd->img_ncpa.ID != -1) ? nd->img_ncpa.im->array.F : NULL;

            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                cacaosim_pyrwfs_accum_beam(&nd->pyrwfs_ctx, h_opd, h_amp, s_ncpa, d_opd, d_amp,
                                           nd->cfg.pyrwfs.mod_rad_ld);
                if (nd->pyrwfs_ctx.subframe_cnt >= nd->eff_navg)
                {
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    cacaosim_pyrwfs_readout(&nd->pyrwfs_ctx, &nd->cfg.pyrwfs.det, dst_u16, dst_f32);
                    nd->ready_this_tick = 1;
                }
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                cacaosim_shwfs_accum_beam(&nd->shwfs_ctx, h_opd, h_amp, s_ncpa, d_opd, d_amp);
                if (nd->shwfs_ctx.subframe_cnt >= nd->eff_navg)
                {
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    cacaosim_shwfs_readout(&nd->shwfs_ctx, &nd->cfg.shwfs.det, dst_u16, dst_f32);
                    nd->ready_this_tick = 1;
                }
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                cacaosim_zwfs_accum_beam(&nd->zwfs_ctx, h_opd, h_amp, s_ncpa, d_opd, d_amp);
                if (nd->zwfs_ctx.subframe_cnt >= nd->eff_navg)
                {
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    cacaosim_zwfs_readout(&nd->zwfs_ctx, &nd->cfg.zwfs.det, dst_u16, dst_f32);
                    nd->ready_this_tick = 1;
                }
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                cacaosim_fpcam_accum_beam_poly(&nd->fpcam_ctx, h_opd, h_amp, nl, ctx->npix_wf,
                                               s_ncpa, d_opd, d_amp);
                if (nd->fpcam_ctx.subframe_cnt >= nd->eff_navg)
                {
                    float *dst_long = (ctx->connect_shm && nd->img_long.ID != -1)
                                          ? nd->img_long.im->array.F
                                          : nd->long_f32;
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    if (ctx->connect_shm && nd->img_long.ID != -1)
                    {
                        nd->img_long.md->write = 1;
                    }
                    cacaosim_fpcam_readout(&nd->fpcam_ctx, &nd->cfg.fpcam.det, dst_u16, dst_f32,
                                           dst_long);
                    ctx->last_strehl_inst = nd->fpcam_ctx.last_strehl_peak;
                    ctx->last_strehl_ave  = nd->fpcam_ctx.long_strehl_peak;
                    nd->ready_this_tick   = 1;

                    if (ctx->connect_shm && nd->img_long.ID != -1)
                    {
                        processinfo_update_output_stream(pinfo, nd->img_long.im, trig_im);
                    }
                }
            }

            if (nd->ready_this_tick && ctx->connect_shm && nd->img_cam.ID != -1)
            {
                processinfo_update_output_stream(pinfo, nd->img_cam.im, trig_im);
            }
        }
    }

    ctx->tick_cnt++;
}

int cacaosim_bench_find_dm(const cacaosim_bench_ctx_t *ctx, const char *id)
{
    if (ctx == NULL || id == NULL)
    {
        return -1;
    }
    for (uint32_t d = 0; d < ctx->n_dms; d++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->dm_node_idx[d]];
        if (strcmp(nd->cfg.id, id) == 0)
        {
            return (int) d;
        }
    }
    return -1;
}

int cacaosim_bench_find_sensor(const cacaosim_bench_ctx_t *ctx, const char *id)
{
    if (ctx == NULL || id == NULL)
    {
        return -1;
    }
    for (uint32_t s = 0; s < ctx->n_sensors; s++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->sensor_node_idx[s]];
        if (strcmp(nd->cfg.id, id) == 0)
        {
            return (int) s;
        }
    }
    return -1;
}

float *cacaosim_bench_dm_cmd_ptr(cacaosim_bench_ctx_t *ctx, int dm_slot)
{
    if (ctx == NULL || dm_slot < 0 || (uint32_t) dm_slot >= ctx->n_dms)
    {
        return NULL;
    }
    cacaosim_bench_node_t *nd = &ctx->nodes[ctx->dm_node_idx[dm_slot]];
    if (ctx->connect_shm && nd->img_cmd.ID != -1)
    {
        return nd->img_cmd.im->array.F;
    }
    return nd->cmd_buf;
}

const float *cacaosim_bench_sensor_f32(const cacaosim_bench_ctx_t *ctx, int sensor_slot)
{
    if (ctx == NULL || sensor_slot < 0 || (uint32_t) sensor_slot >= ctx->n_sensors)
    {
        return NULL;
    }
    const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->sensor_node_idx[sensor_slot]];
    if (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype != 0)
    {
        return nd->img_cam.im->array.F;
    }
    return nd->cam_f32;
}

const uint16_t *cacaosim_bench_sensor_u16(const cacaosim_bench_ctx_t *ctx, int sensor_slot)
{
    if (ctx == NULL || sensor_slot < 0 || (uint32_t) sensor_slot >= ctx->n_sensors)
    {
        return NULL;
    }
    const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->sensor_node_idx[sensor_slot]];
    if (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype == 0)
    {
        return nd->img_cam.im->array.UI16;
    }
    return nd->cam_u16;
}
