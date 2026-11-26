// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfstat_core.c
 * @brief   Shared pupil wavefront statistics engine: CPU accumulation, extras, and file writer
 *
 * See cacaosim_wfstat_core.h. The CUDA accumulation kernels live in cacaosim_wfstat_cuda.cu.
 */

#include <cblas.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_mask.h"
#include "common/cacaosim_wfstat_core.h"

typedef struct
{
    const char *key;
    const char *stream_name;
    int         kind; /* 0=dm_opd, 1=dm_cmd, 2=wfs_raw, 3=wfs_sig, 4=psf_inst */
} wfstat_extra_def_t;

typedef struct
{
    const char *key;
    const char *stream_name;
} wfstat_strehl_def_t;

static const wfstat_extra_def_t s_extra_defs[CACAOSIM_WFSTAT_NB_EXTRA] = {
    { "wfs_raw_stg1", "csim_wfs_nir", 2 },        { "wfs_sig_stg1", "aol1_imWFS2", 3 },
    { "wfs_raw_stg2", "csim_wfs_vis", 2 },        { "wfs_sig_stg2", "aol2_imWFS2", 3 },
    { "dm_cmd_stg1", "dm01disp", 1 },             { "dm_opd_stg1", "csim_dm01_opd", 0 },
    { "dm_cmd_stg2", "dm02disp", 1 },             { "dm_opd_stg2", "csim_dm02_opd", 0 },
    { "psf_inst_vis_raw", "psf_vis_raw", 4 },     { "psf_inst_vis_stg1", "psf_vis_stg1", 4 },
    { "psf_inst_vis_final", "psf_vis_final", 4 }, { "psf_inst_nir_raw", "psf_nir_raw", 4 },
    { "psf_inst_nir_stg1", "psf_nir_stg1", 4 },   { "psf_inst_nir_final", "psf_nir_final", 4 },
    { "wfs_raw_stg3", "csim_wfs_lowfs", 2 },      { "wfs_sig_stg3", "aol3_imWFS2", 3 },
    { "dm_cmd_stg3", "aol3_dmC", 1 },             { "psf_inst_nir_coro", "psf_nir_coro", 4 },
    { "coro_pre_lyot", "coro_pre_lyot", 5 },
};

#define WFSTAT_EXTRA_WFS2_IDX    2
#define WFSTAT_EXTRA_DM1_OPD_IDX 5
#define WFSTAT_EXTRA_DM2_OPD_IDX 7

static const wfstat_strehl_def_t s_strehl_defs[CACAOSIM_WFSTAT_NB_STREHL] = {
    { "vis_raw", "psf_vis_raw_long" },     { "vis_stg1", "psf_vis_stg1_long" },
    { "vis_final", "psf_vis_final_long" }, { "nir_raw", "psf_nir_raw_long" },
    { "nir_stg1", "psf_nir_stg1_long" },   { "nir_final", "psf_nir_final_long" },
    { "sci", "csim_fplong" },              { "nir_coro", "psf_nir_coro_long" },
};

static const char *s_loc_keys[CACAOSIM_WFSTAT_NB_LOC] = { "raw", "stg1", "final" };
static const char *s_loc_desc[CACAOSIM_WFSTAT_NB_LOC] = {
    "Before Stage 1 (Entrance Pupil)",
    "Between Stages 1 & 2 (Post-AO3k)",
    "After Stage 2 (Post-SCExAO)",
};

static int cmp_dbl(const void *a, const void *b)
{
    double da = *(const double *) a;
    double db = *(const double *) b;
    return (da > db) - (da < db);
}

static void mkdir_p(const char *path)
{
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++)
    {
        if (*p == '/')
        {
            *p = '\0';
            mkdir(tmp, 0775);
            *p = '/';
        }
    }
    mkdir(tmp, 0775);
}

int cacaosim_wfstat_init(
    cacaosim_wfstat_t *ws,
    uint32_t           nwf,
    const float       *amp,
    uint64_t           target_ticks,
    int                snap_extra,
    int                use_cuda,
    void              *cuda_stream)
{
    memset(ws, 0, sizeof(*ws));
    for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
    {
        ws->extras[e].img.ID = -1;
    }
    for (int c = 0; c < CACAOSIM_WFSTAT_NB_STREHL; c++)
    {
        ws->strehl[c].img.ID = -1;
    }
    ws->coro_ts.img.ID = -1;
    ws->nwf        = nwf;
    ws->npix       = (uint64_t) nwf * nwf;
    ws->snap_extra = snap_extra;
    ws->ts_cap     = (target_ticks > 10) ? (target_ticks + 4096ULL) : 262144ULL;
    uint64_t npix  = ws->npix;

    ws->mask    = (uint8_t *) calloc(npix, sizeof(uint8_t));
    ws->weights = (double *) calloc(npix, sizeof(double));
    ws->pup_amp = (float *) calloc(npix, sizeof(float));
    ws->wf_tmp  = (float *) calloc(npix, sizeof(float));
    ws->sortbuf = (double *) calloc(npix, sizeof(double));
    if (!ws->mask || !ws->weights || !ws->pup_amp || !ws->wf_tmp || !ws->sortbuf)
    {
        cacaosim_wfstat_free(ws);
        return -1;
    }
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        ws->sum_wf[k]  = (double *) calloc(npix, sizeof(double));
        ws->sum2_wf[k] = (double *) calloc(npix, sizeof(double));
        ws->samp_wf[k] = (float *) calloc(CACAOSIM_WFSTAT_NB_SAMP * npix, sizeof(float));
        ws->ts_rms[k]  = (float *) calloc(ws->ts_cap, sizeof(float));
        if (!ws->sum_wf[k] || !ws->sum2_wf[k] || !ws->samp_wf[k] || !ws->ts_rms[k])
        {
            cacaosim_wfstat_free(ws);
            return -1;
        }
    }

    double w_sum = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        float a        = amp ? amp[i] : 1.0f;
        ws->pup_amp[i] = a;
        if (a > 0.5f)
        {
            ws->mask[i]    = 1;
            ws->weights[i] = (double) a * (double) a;
            w_sum += ws->weights[i];
            ws->act_pix++;
        }
    }
    if (w_sum <= 0.0)
    {
        ws->act_pix = npix;
        for (uint64_t i = 0; i < npix; i++)
        {
            ws->mask[i]    = 1;
            ws->pup_amp[i] = 1.0f;
            ws->weights[i] = 1.0 / (double) npix;
        }
    }
    else
    {
        for (uint64_t i = 0; i < npix; i++)
        {
            ws->weights[i] /= w_sum;
        }
    }

    ws->act_idx = (uint32_t *) calloc(ws->act_pix, sizeof(uint32_t));
    ws->act_sw  = (float *) calloc(ws->act_pix, sizeof(float));
    if (!ws->act_idx || !ws->act_sw)
    {
        cacaosim_wfstat_free(ws);
        return -1;
    }
    uint64_t ap = 0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (ws->mask[i])
        {
            ws->act_idx[ap] = (uint32_t) i;
            ws->act_sw[ap]  = (float) sqrt(ws->weights[i]);
            ap++;
        }
    }

    const double fracs[CACAOSIM_WFSTAT_NB_SAMP] = { 0.10, 0.30, 0.50, 0.70, 0.95 };
    uint64_t     tgt = (target_ticks > 10) ? target_ticks : 20000ULL;
    for (int s = 0; s < CACAOSIM_WFSTAT_NB_SAMP; s++)
    {
        uint64_t v        = (uint64_t) llround(fracs[s] * (double) tgt);
        ws->samp_ticks[s] = (v < 1) ? 1 : v;
    }

    if (snap_extra)
    {
        for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
        {
            cacaosim_wfstat_extra_t *es = &ws->extras[e];
            es->img                     = stream_connect(s_extra_defs[e].stream_name);
            if (es->img.ID != -1)
            {
                es->nx        = es->img.md->size[0];
                es->ny        = es->img.md->size[1];
                uint64_t epix = (uint64_t) es->nx * es->ny;
                es->cube   = (float *) calloc(CACAOSIM_WFSTAT_NB_SAMP * epix, sizeof(float));
                es->active = (es->cube != NULL);
            }
        }
        for (int c = 0; c < CACAOSIM_WFSTAT_NB_STREHL; c++)
        {
            cacaosim_wfstat_strehl_t *st = &ws->strehl[c];
            st->img                      = stream_connect(s_strehl_defs[c].stream_name);
            if (st->img.ID != -1 && st->img.md != NULL)
            {
                st->cap     = ws->ts_cap;
                st->frame   = (uint64_t *) calloc(st->cap, sizeof(uint64_t));
                st->sr_inst = (float *) calloc(st->cap, sizeof(float));
                st->sr_long = (float *) calloc(st->cap, sizeof(float));
                st->sr_axis = (float *) calloc(st->cap, sizeof(float));
                if (!st->frame || !st->sr_inst || !st->sr_long || !st->sr_axis)
                {
                    cacaosim_wfstat_free(ws);
                    return -1;
                }
                st->last_cnt0 = st->img.md->cnt0;
                st->active    = 1;
            }
        }
        ws->coro_ts.img = stream_connect("coro_pre_lyot_long");
        if (ws->coro_ts.img.ID != -1 && ws->coro_ts.img.md != NULL)
        {
            ws->coro_ts.cap          = ws->ts_cap;
            ws->coro_ts.frame        = (uint64_t *) calloc(ws->ts_cap, sizeof(uint64_t));
            ws->coro_ts.f_lowfs_inst = (float *) calloc(ws->ts_cap, sizeof(float));
            ws->coro_ts.f_lowfs_long = (float *) calloc(ws->ts_cap, sizeof(float));
            ws->coro_ts.f_pre_inst   = (float *) calloc(ws->ts_cap, sizeof(float));
            ws->coro_ts.f_trans_inst = (float *) calloc(ws->ts_cap, sizeof(float));
            if (!ws->coro_ts.frame || !ws->coro_ts.f_lowfs_inst || !ws->coro_ts.f_lowfs_long ||
                !ws->coro_ts.f_pre_inst || !ws->coro_ts.f_trans_inst)
            {
                cacaosim_wfstat_free(ws);
                return -1;
            }
            ws->coro_ts.last_cnt0 = ws->coro_ts.img.md->cnt0;
            ws->coro_ts.active    = 1;
        }
        ws->pert_ts = (cacaosim_wfstat_perturb_sample_t *) calloc(
            ws->ts_cap, sizeof(cacaosim_wfstat_perturb_sample_t));
    }

    if (use_cuda)
    {
        ws->gpu = cacaosim_wfstat_gpu_create(ws->weights, npix, CACAOSIM_WFSTAT_NB_LOC,
                                             ws->ts_cap, cuda_stream);
        if (ws->gpu == NULL)
        {
            cacaosim_wfstat_free(ws);
            return -1;
        }
        ws->use_cuda = 1;
    }
    return 0;
}

int cacaosim_wfstat_enable_clip(
    cacaosim_wfstat_t *ws,
    uint64_t           clip_start,
    uint32_t           clip_len)
{
    if (ws == NULL)
    {
        return -1;
    }
    free(ws->clip_wf_raw);
    free(ws->clip_wf_final);
    free(ws->clip_dm1_opd);
    free(ws->clip_dm2_opd);
    free(ws->clip_frame);
    free(ws->clip_wfs2_cnt0);
    ws->clip_wf_raw    = NULL;
    ws->clip_wf_final  = NULL;
    ws->clip_dm1_opd   = NULL;
    ws->clip_dm2_opd   = NULL;
    ws->clip_frame     = NULL;
    ws->clip_wfs2_cnt0 = NULL;
    ws->clip_start     = 0ULL;
    ws->clip_len       = 0u;
    ws->clip_n         = 0u;
    ws->clip_has_dm1   = 0;
    ws->clip_has_dm2   = 0;

    if (clip_len == 0u)
    {
        return 0;
    }
    if (clip_start < 1ULL)
    {
        clip_start = 1ULL;
    }

    size_t nfloats = (size_t) clip_len * (size_t) ws->npix;
    ws->clip_wf_raw    = (float *) calloc(nfloats, sizeof(float));
    ws->clip_wf_final  = (float *) calloc(nfloats, sizeof(float));
    ws->clip_frame     = (uint64_t *) calloc(clip_len, sizeof(uint64_t));
    ws->clip_wfs2_cnt0 = (uint64_t *) calloc(clip_len, sizeof(uint64_t));
    if (!ws->clip_wf_raw || !ws->clip_wf_final || !ws->clip_frame || !ws->clip_wfs2_cnt0)
    {
        free(ws->clip_wf_raw);
        free(ws->clip_wf_final);
        free(ws->clip_frame);
        free(ws->clip_wfs2_cnt0);
        ws->clip_wf_raw    = NULL;
        ws->clip_wf_final  = NULL;
        ws->clip_frame     = NULL;
        ws->clip_wfs2_cnt0 = NULL;
        return -1;
    }

    /* Lazily attach WFS2 counter and DM OPD SHM streams even if snap_extra == 0 */
    const int needed_idx[3] = {
        WFSTAT_EXTRA_WFS2_IDX,
        WFSTAT_EXTRA_DM1_OPD_IDX,
        WFSTAT_EXTRA_DM2_OPD_IDX,
    };
    for (int j = 0; j < 3; j++)
    {
        cacaosim_wfstat_extra_t *es = &ws->extras[needed_idx[j]];
        if (es->img.ID == -1)
        {
            es->img = stream_connect(s_extra_defs[needed_idx[j]].stream_name);
            if (es->img.ID != -1 && es->img.md != NULL)
            {
                es->nx = es->img.md->size[0];
                es->ny = es->img.md->size[1];
            }
        }
    }

    cacaosim_wfstat_extra_t *d1 = &ws->extras[WFSTAT_EXTRA_DM1_OPD_IDX];
    if (d1->img.ID != -1 && (uint64_t) d1->nx * d1->ny == ws->npix)
    {
        ws->clip_dm1_opd = (float *) calloc(nfloats, sizeof(float));
        ws->clip_has_dm1 = (ws->clip_dm1_opd != NULL) ? 1 : 0;
    }
    cacaosim_wfstat_extra_t *d2 = &ws->extras[WFSTAT_EXTRA_DM2_OPD_IDX];
    if (d2->img.ID != -1 && (uint64_t) d2->nx * d2->ny == ws->npix)
    {
        ws->clip_dm2_opd = (float *) calloc(nfloats, sizeof(float));
        ws->clip_has_dm2 = (ws->clip_dm2_opd != NULL) ? 1 : 0;
    }

    ws->clip_start = clip_start;
    ws->clip_len   = clip_len;
    ws->clip_n     = 0u;
    return 0;
}

int cacaosim_wfstat_enable_full(
    cacaosim_wfstat_t *ws,
    double             step_frames,
    uint32_t           max_frames)
{
    if (ws == NULL)
    {
        return -1;
    }
    free(ws->full_wf_raw);
    free(ws->full_wf_final);
    free(ws->full_dm1_opd);
    free(ws->full_dm2_opd);
    free(ws->full_frame);
    free(ws->full_wfs2_cnt0);
    ws->full_wf_raw     = NULL;
    ws->full_wf_final   = NULL;
    ws->full_dm1_opd    = NULL;
    ws->full_dm2_opd    = NULL;
    ws->full_frame      = NULL;
    ws->full_wfs2_cnt0  = NULL;
    ws->full_step       = 0.0;
    ws->full_next_frame = 0ULL;
    ws->full_cap        = 0u;
    ws->full_n          = 0u;
    ws->full_has_dm1    = 0;
    ws->full_has_dm2    = 0;

    if (max_frames == 0u || step_frames <= 0.0)
    {
        return 0;
    }
    if (step_frames < 1.0)
    {
        step_frames = 1.0;
    }

    size_t nfloats     = (size_t) max_frames * (size_t) ws->npix;
    ws->full_wf_raw    = (float *) calloc(nfloats, sizeof(float));
    ws->full_wf_final  = (float *) calloc(nfloats, sizeof(float));
    ws->full_frame     = (uint64_t *) calloc(max_frames, sizeof(uint64_t));
    ws->full_wfs2_cnt0 = (uint64_t *) calloc(max_frames, sizeof(uint64_t));
    if (!ws->full_wf_raw || !ws->full_wf_final || !ws->full_frame || !ws->full_wfs2_cnt0)
    {
        free(ws->full_wf_raw);
        free(ws->full_wf_final);
        free(ws->full_frame);
        free(ws->full_wfs2_cnt0);
        ws->full_wf_raw    = NULL;
        ws->full_wf_final  = NULL;
        ws->full_frame     = NULL;
        ws->full_wfs2_cnt0 = NULL;
        return -1;
    }

    /* Lazily attach WFS2 counter and DM OPD SHM streams even if snap_extra == 0 */
    const int needed_idx[3] = {
        WFSTAT_EXTRA_WFS2_IDX,
        WFSTAT_EXTRA_DM1_OPD_IDX,
        WFSTAT_EXTRA_DM2_OPD_IDX,
    };
    for (int j = 0; j < 3; j++)
    {
        cacaosim_wfstat_extra_t *es = &ws->extras[needed_idx[j]];
        if (es->img.ID == -1)
        {
            es->img = stream_connect(s_extra_defs[needed_idx[j]].stream_name);
            if (es->img.ID != -1 && es->img.md != NULL)
            {
                es->nx = es->img.md->size[0];
                es->ny = es->img.md->size[1];
            }
        }
    }

    cacaosim_wfstat_extra_t *d1 = &ws->extras[WFSTAT_EXTRA_DM1_OPD_IDX];
    if (d1->img.ID != -1 && (uint64_t) d1->nx * d1->ny == ws->npix)
    {
        ws->full_dm1_opd = (float *) calloc(nfloats, sizeof(float));
        ws->full_has_dm1 = (ws->full_dm1_opd != NULL) ? 1 : 0;
    }
    cacaosim_wfstat_extra_t *d2 = &ws->extras[WFSTAT_EXTRA_DM2_OPD_IDX];
    if (d2->img.ID != -1 && (uint64_t) d2->nx * d2->ny == ws->npix)
    {
        ws->full_dm2_opd = (float *) calloc(nfloats, sizeof(float));
        ws->full_has_dm2 = (ws->full_dm2_opd != NULL) ? 1 : 0;
    }

    ws->full_step       = step_frames;
    ws->full_next_frame = 1ULL;
    ws->full_cap        = max_frames;
    ws->full_n          = 0u;
    return 0;
}

/* Sequential rank-revealing Cholesky-QR with 2-pass re-orthogonalization (col-major FP64) */
static int orthonormalize_cols_d(
    double   *M,
    uint64_t  nrows,
    uint32_t  ncols,
    double    rel_tol,
    uint32_t *out_rank)
{
    *out_rank = 0;
    if (M == NULL || nrows == 0 || ncols == 0)
    {
        return 0;
    }
    double   *G     = (double *) calloc((size_t) ncols * ncols, sizeof(double));
    double   *R     = (double *) calloc((size_t) ncols * ncols, sizeof(double));
    double   *col_r = (double *) calloc(ncols, sizeof(double));
    uint32_t *keep  = (uint32_t *) calloc(ncols, sizeof(uint32_t));
    if (!G || !R || !col_r || !keep)
    {
        free(G);
        free(R);
        free(col_r);
        free(keep);
        return -1;
    }

    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, (int) ncols, (int) nrows, 1.0, M,
                (int) nrows, 0.0, G, (int) ncols);

    uint32_t r = 0;
    for (uint32_t j = 0; j < ncols; j++)
    {
        double diag0 = G[(size_t) j * ncols + j];
        if (diag0 <= 1.0e-24)
        {
            continue;
        }
        double rem = diag0;
        for (uint32_t i = 0; i < r; i++)
        {
            double s = G[(size_t) j * ncols + keep[i]];
            for (uint32_t p = 0; p < i; p++)
            {
                s -= R[(size_t) i * ncols + p] * col_r[p];
            }
            double rij = s / R[(size_t) i * ncols + i];
            col_r[i]   = rij;
            rem -= rij * rij;
        }
        if (rem > rel_tol * diag0)
        {
            keep[r] = j;
            for (uint32_t i = 0; i < r; i++)
            {
                R[(size_t) r * ncols + i] = col_r[i];
            }
            R[(size_t) r * ncols + r] = sqrt(rem);
            r++;
        }
    }

    if (r > 0)
    {
        for (uint32_t i = 0; i < r; i++)
        {
            if (keep[i] != i)
            {
                memcpy(M + (size_t) i * nrows, M + (size_t) keep[i] * nrows,
                       nrows * sizeof(double));
            }
        }
        double *R_sub = (double *) calloc((size_t) r * r, sizeof(double));
        if (!R_sub)
        {
            free(G);
            free(R);
            free(col_r);
            free(keep);
            return -1;
        }
        for (uint32_t j = 0; j < r; j++)
        {
            for (uint32_t i = 0; i <= j; i++)
            {
                R_sub[(size_t) j * r + i] = R[(size_t) j * ncols + i];
            }
        }
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, (int) nrows,
                    (int) r, 1.0, R_sub, (int) r, M, (int) nrows);

        /* Second pass: re-orthogonalize Q to machine precision */
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, (int) r, (int) nrows, 1.0, M,
                    (int) nrows, 0.0, G, (int) r);
        int ok2 = 1;
        for (uint32_t j = 0; j < r && ok2; j++)
        {
            double rem = G[(size_t) j * r + j];
            for (uint32_t i = 0; i < j; i++)
            {
                double s = G[(size_t) j * r + i];
                for (uint32_t p = 0; p < i; p++)
                {
                    s -= R_sub[(size_t) i * r + p] * R_sub[(size_t) j * r + p];
                }
                double rij                = s / R_sub[(size_t) i * r + i];
                R_sub[(size_t) j * r + i] = rij;
                rem -= rij * rij;
            }
            if (rem <= 1.0e-12)
            {
                ok2 = 0;
            }
            else
            {
                R_sub[(size_t) j * r + j] = sqrt(rem);
            }
        }
        if (ok2)
        {
            cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                        (int) nrows, (int) r, 1.0, R_sub, (int) r, M, (int) nrows);
        }
        free(R_sub);
    }

    free(G);
    free(R);
    free(col_r);
    free(keep);
    *out_rank = r;
    return 0;
}

/* Load a 3D float32 CMmodesDM SHM stream and propagate onto active pupil pixels via 16-tap stencil */
static double *gather_dm_modes_d(
    const cacaosim_wfstat_t *ws,
    const char              *cm_stream,
    const int32_t           *stencil_idx,
    const float             *stencil_w,
    uint32_t                 dm_nact,
    uint32_t                *out_nmodes)
{
    *out_nmodes = 0;
    if (ws == NULL || cm_stream == NULL || cm_stream[0] == '\0' || stencil_idx == NULL ||
        stencil_w == NULL || dm_nact == 0 || ws->act_pix == 0)
    {
        return NULL;
    }
    IMGID img = stream_connect(cm_stream);
    if (img.ID == -1 || img.md == NULL || img.im == NULL)
    {
        return NULL;
    }
    if (img.md->datatype != _DATATYPE_FLOAT)
    {
        imgid_free(&img);
        return NULL;
    }
    uint32_t nx     = img.md->size[0];
    uint32_t ny     = (img.md->naxis >= 2) ? img.md->size[1] : 1u;
    uint32_t nmodes = (img.md->naxis >= 3) ? img.md->size[2] : 1u;
    if ((uint64_t) nx * ny != (uint64_t) dm_nact || nmodes == 0)
    {
        imgid_free(&img);
        return NULL;
    }
    uint64_t act_pix = ws->act_pix;
    double  *M       = (double *) malloc(act_pix * (size_t) nmodes * sizeof(double));
    if (M == NULL)
    {
        imgid_free(&img);
        return NULL;
    }
    const float *src_all = img.im->array.F;
#pragma omp parallel for schedule(static)
    for (int64_t mi = 0; mi < (int64_t) nmodes; mi++)
    {
        const float *cmd  = src_all + (size_t) mi * dm_nact;
        double      *col  = M + (size_t) mi * act_pix;
        double       mean = 0.0;
        for (uint64_t p = 0; p < act_pix; p++)
        {
            uint32_t       pix  = ws->act_idx[p];
            const int32_t *sidx = stencil_idx + (size_t) pix * 16u;
            const float   *sw   = stencil_w + (size_t) pix * 16u;
            double         val  = 0.0;
            for (int t = 0; t < 16; t++)
            {
                val += (double) sw[t] * (double) cmd[sidx[t]];
            }
            col[p] = val;
            mean += ws->weights[pix] * val;
        }
        for (uint64_t p = 0; p < act_pix; p++)
        {
            col[p] = (col[p] - mean) * (double) ws->act_sw[p];
        }
    }
    imgid_free(&img);
    *out_nmodes = nmodes;
    return M;
}

int cacaosim_wfstat_set_ctrl_basis(
    cacaosim_wfstat_t *ws,
    const float       *q_act,
    uint32_t           n_stg1,
    uint32_t           n_tot,
    uint32_t           raw_n_stg1,
    uint32_t           raw_n_stg2)
{
    if (ws == NULL || q_act == NULL || ws->act_pix == 0 || n_tot == 0 || n_stg1 > n_tot)
    {
        return -1;
    }
    free(ws->ctrl_q);
    free(ws->ctrl_v_tmp);
    free(ws->ctrl_c_tmp);
    ws->ctrl_q     = NULL;
    ws->ctrl_v_tmp = NULL;
    ws->ctrl_c_tmp = NULL;
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        free(ws->ts_in[k]);
        free(ws->ts_out[k]);
        ws->ts_in[k]      = (float *) calloc(ws->ts_cap, sizeof(float));
        ws->ts_out[k]     = (float *) calloc(ws->ts_cap, sizeof(float));
        ws->sum_in_rms[k] = 0.0;
        ws->sum_in_var[k] = 0.0;
        ws->sum_out_rms[k] = 0.0;
        ws->sum_out_var[k] = 0.0;
        if (!ws->ts_in[k] || !ws->ts_out[k])
        {
            return -1;
        }
    }
    ws->ctrl_n_stg1     = n_stg1;
    ws->ctrl_n_tot      = n_tot;
    ws->ctrl_raw_n_stg1 = raw_n_stg1;
    ws->ctrl_raw_n_stg2 = raw_n_stg2;

    if (ws->use_cuda && ws->gpu != NULL)
    {
        if (cacaosim_wfstat_gpu_set_ctrl_basis(ws->gpu, ws->act_idx, ws->act_sw, ws->act_pix, q_act,
                                               n_stg1, n_tot) != 0)
        {
            return -1;
        }
    }
    else
    {
        size_t q_cnt   = (size_t) ws->act_pix * n_tot;
        ws->ctrl_q     = (float *) malloc(q_cnt * sizeof(float));
        ws->ctrl_v_tmp = (float *) malloc((size_t) ws->act_pix * CACAOSIM_WFSTAT_NB_LOC *
                                          sizeof(float));
        ws->ctrl_c_tmp = (float *) malloc((size_t) n_tot * CACAOSIM_WFSTAT_NB_LOC * sizeof(float));
        if (!ws->ctrl_q || !ws->ctrl_v_tmp || !ws->ctrl_c_tmp)
        {
            return -1;
        }
        memcpy(ws->ctrl_q, q_act, q_cnt * sizeof(float));
    }
    ws->ctrl_active = 1;
    return 0;
}

int cacaosim_wfstat_load_ctrl_modes(
    cacaosim_wfstat_t *ws,
    const char        *cm1_stream,
    const int32_t     *dm1_stencil_idx,
    const float       *dm1_stencil_w,
    uint32_t           dm1_nact,
    const char        *cm2_stream,
    const int32_t     *dm2_stencil_idx,
    const float       *dm2_stencil_w,
    uint32_t           dm2_nact)
{
    if (ws == NULL || ws->act_pix == 0)
    {
        return -1;
    }
    uint64_t act_pix = ws->act_pix;
    uint32_t raw_n1 = 0, raw_n2 = 0;
    double  *M1 = gather_dm_modes_d(ws, cm1_stream, dm1_stencil_idx, dm1_stencil_w, dm1_nact,
                                    &raw_n1);
    double  *M2 = gather_dm_modes_d(ws, cm2_stream, dm2_stencil_idx, dm2_stencil_w, dm2_nact,
                                    &raw_n2);
    if (M1 == NULL && M2 == NULL)
    {
        return 0;
    }

    uint32_t r1 = 0, r2 = 0;
    if (M1 != NULL && orthonormalize_cols_d(M1, act_pix, raw_n1, 1.0e-4, &r1) != 0)
    {
        free(M1);
        free(M2);
        return -1;
    }

    if (M2 != NULL)
    {
        double *orig_n2 = (double *) calloc(raw_n2, sizeof(double));
        if (!orig_n2)
        {
            free(M1);
            free(M2);
            return -1;
        }
        for (uint32_t j = 0; j < raw_n2; j++)
        {
            const double *col = M2 + (size_t) j * act_pix;
            double        s   = 0.0;
            for (uint64_t p = 0; p < act_pix; p++)
            {
                s += col[p] * col[p];
            }
            orig_n2[j] = s;
        }
        if (r1 > 0)
        {
            double *C = (double *) malloc((size_t) r1 * raw_n2 * sizeof(double));
            if (!C)
            {
                free(orig_n2);
                free(M1);
                free(M2);
                return -1;
            }
            /* Two-pass orthogonal projection of M2 out of span(Q1) */
            for (int pass = 0; pass < 2; pass++)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) r1, (int) raw_n2,
                            (int) act_pix, 1.0, M1, (int) act_pix, M2, (int) act_pix, 0.0, C,
                            (int) r1);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int) act_pix, (int) raw_n2,
                            (int) r1, -1.0, M1, (int) act_pix, C, (int) r1, 1.0, M2, (int) act_pix);
            }
            free(C);
        }
        /* Keep Stage-2 columns with >=25% energy orthogonal to Stage 1 (skip DM grid mismatch) */
        uint32_t kept2 = 0;
        for (uint32_t j = 0; j < raw_n2; j++)
        {
            const double *col = M2 + (size_t) j * act_pix;
            double        s   = 0.0;
            for (uint64_t p = 0; p < act_pix; p++)
            {
                s += col[p] * col[p];
            }
            if (orig_n2[j] > 1.0e-24 && s > 0.25 * orig_n2[j])
            {
                if (kept2 != j)
                {
                    memcpy(M2 + (size_t) kept2 * act_pix, col, act_pix * sizeof(double));
                }
                kept2++;
            }
        }
        free(orig_n2);
        if (kept2 > 0 && orthonormalize_cols_d(M2, act_pix, kept2, 1.0e-4, &r2) != 0)
        {
            free(M1);
            free(M2);
            return -1;
        }
        /* Final re-projection of Q2_perp against Q1 to eliminate any round-off drift */
        if (r1 > 0 && r2 > 0)
        {
            double *C = (double *) malloc((size_t) r1 * r2 * sizeof(double));
            if (C != NULL)
            {
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) r1, (int) r2,
                            (int) act_pix, 1.0, M1, (int) act_pix, M2, (int) act_pix, 0.0, C,
                            (int) r1);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int) act_pix, (int) r2,
                            (int) r1, -1.0, M1, (int) act_pix, C, (int) r1, 1.0, M2, (int) act_pix);
                free(C);
                uint32_t r2_final = 0;
                if (orthonormalize_cols_d(M2, act_pix, r2, 1.0e-4, &r2_final) == 0)
                {
                    r2 = r2_final;
                }
            }
        }
    }

    uint32_t n_tot = r1 + r2;
    int      rc    = 0;
    if (n_tot > 0)
    {
        float *Q_f32 = (float *) malloc(act_pix * (size_t) n_tot * sizeof(float));
        if (!Q_f32)
        {
            free(M1);
            free(M2);
            return -1;
        }
        for (uint32_t j = 0; j < r1; j++)
        {
            const double *src = M1 + (size_t) j * act_pix;
            float        *dst = Q_f32 + (size_t) j * act_pix;
            for (uint64_t p = 0; p < act_pix; p++)
            {
                dst[p] = (float) src[p];
            }
        }
        for (uint32_t j = 0; j < r2; j++)
        {
            const double *src = M2 + (size_t) j * act_pix;
            float        *dst = Q_f32 + (size_t) (r1 + j) * act_pix;
            for (uint64_t p = 0; p < act_pix; p++)
            {
                dst[p] = (float) src[p];
            }
        }
        rc = cacaosim_wfstat_set_ctrl_basis(ws, Q_f32, r1, n_tot, raw_n1, raw_n2);
        free(Q_f32);
    }
    free(M1);
    free(M2);
    return rc;
}

/* Piston-remove one host plane, accumulate sums, and optionally capture the sample map */
static void accum_host_plane(
    cacaosim_wfstat_t *ws,
    int                k,
    const float       *opd,
    int                is_samp,
    double            *out_svar)
{
    uint64_t npix = ws->npix;
    double   pist = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (ws->mask[i])
        {
            pist += ws->weights[i] * ((double) opd[i] * 1000.0);
        }
    }
    double svar = 0.0;
    double vmin = 1.0e30;
    double vmax = -1.0e30;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (ws->mask[i])
        {
            double w = (double) opd[i] * 1000.0 - pist;
            ws->sum_wf[k][i] += w;
            ws->sum2_wf[k][i] += w * w;
            svar += ws->weights[i] * w * w;
            ws->wf_tmp[i] = (float) w;
            vmin          = (w < vmin) ? w : vmin;
            vmax          = (w > vmax) ? w : vmax;
        }
        else
        {
            ws->wf_tmp[i] = 0.0f;
        }
    }
    if (out_svar != NULL)
    {
        *out_svar = svar;
    }
    if (ws->ctrl_active && ws->ctrl_v_tmp != NULL)
    {
        float *vk = ws->ctrl_v_tmp + (size_t) k * ws->act_pix;
        for (uint64_t p = 0; p < ws->act_pix; p++)
        {
            vk[p] = ws->act_sw[p] * ws->wf_tmp[ws->act_idx[p]];
        }
    }
    double srms = sqrt((svar > 0.0) ? svar : 0.0);
    ws->sum_svar[k] += svar;
    ws->sum_srms[k] += srms;
    uint64_t slot = ws->n_frames - 1ULL;
    if (slot < ws->ts_cap && ws->ts_rms[k] != NULL)
    {
        ws->ts_rms[k][slot] = (float) srms;
    }
    if (is_samp)
    {
        memcpy(ws->samp_wf[k] + (uint64_t) ws->nsamp * npix, ws->wf_tmp, npix * sizeof(float));
        ws->samp_srms[ws->nsamp][k] = srms;
        ws->samp_sptv[ws->nsamp][k] = (ws->act_pix > 0) ? (vmax - vmin) : 0.0;
    }
}

/* Snapshot the 14 optional diagnostic streams into sample slot ws->nsamp */
static void snapshot_extras(cacaosim_wfstat_t *ws)
{
    int s = ws->nsamp;
    for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
    {
        cacaosim_wfstat_extra_t *es = &ws->extras[e];
        if (!es->active)
        {
            continue;
        }
        uint64_t     epix = (uint64_t) es->nx * es->ny;
        const float *src  = es->img.im->array.F;
        float       *dst  = es->cube + (uint64_t) s * epix;
        int          kind = s_extra_defs[e].kind;
        if (kind == 0 && epix == ws->npix)
        {
            double pist = 0.0;
            for (uint64_t i = 0; i < epix; i++)
            {
                if (ws->mask[i])
                {
                    pist += ws->weights[i] * ((double) src[i] * 1000.0);
                }
            }
            double sv = 0.0, mn = 1.0e30, mx = -1.0e30;
            for (uint64_t i = 0; i < epix; i++)
            {
                if (ws->mask[i])
                {
                    double w = (double) src[i] * 1000.0 - pist;
                    dst[i]   = (float) w;
                    sv += ws->weights[i] * w * w;
                    mn = (w < mn) ? w : mn;
                    mx = (w > mx) ? w : mx;
                }
                else
                {
                    dst[i] = 0.0f;
                }
            }
            es->m1[s] = sqrt((sv > 0.0) ? sv : 0.0);
            es->m2[s] = mx - mn;
        }
        else if (kind == 1 || kind == 3)
        {
            double s2 = 0.0, mn = 1.0e30, mx = -1.0e30;
            for (uint64_t i = 0; i < epix; i++)
            {
                float v = src[i];
                dst[i]  = v;
                s2 += (double) v * (double) v;
                mn = (v < mn) ? v : mn;
                mx = (v > mx) ? v : mx;
            }
            es->m1[s] = sqrt(s2 / (double) (epix ? epix : 1));
            es->m2[s] = mx - mn;
        }
        else if (kind == 2)
        {
            double sm = 0.0, mx = -1.0e30;
            for (uint64_t i = 0; i < epix; i++)
            {
                float v = src[i];
                dst[i]  = v;
                sm += (double) v;
                mx = (v > mx) ? v : mx;
            }
            es->m1[s] = sm;
            es->m2[s] = mx;
        }
        else if (kind == 5)
        {
            double sm = 0.0, mx = 0.0;
            for (uint64_t i = 0; i < epix; i++)
            {
                float v = src[i];
                dst[i]  = v;
                sm += (double) v;
                if ((double) v > mx)
                {
                    mx = (double) v;
                }
            }
            es->m1[s] = mx;
            es->m2[s] = sm;
        }
        else
        {
            double sm = 0.0, mx = 0.0;
            for (uint64_t i = 0; i < epix; i++)
            {
                float  v = src[i];
                double d = ((double) v - 100.0 > 0.0) ? ((double) v - 100.0) : 0.0;
                dst[i]   = v;
                sm += d;
                mx = (d > mx) ? d : mx;
            }
            es->m1[s] = mx;
            es->m2[s] = sm;
        }
    }
}

/* Piston-remove a pupil OPD map [um] across active pupil mask -> dst_nm [nm] */
static void piston_remove_opd_nm(
    const cacaosim_wfstat_t *ws,
    const float             *src_um,
    float                   *dst_nm)
{
    uint64_t npix = ws->npix;
    double   pist = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (ws->mask[i])
        {
            pist += ws->weights[i] * ((double) src_um[i] * 1000.0);
        }
    }
    for (uint64_t i = 0; i < npix; i++)
    {
        dst_nm[i] = ws->mask[i] ? (float) ((double) src_um[i] * 1000.0 - pist) : 0.0f;
    }
}

void cacaosim_wfstat_frame(cacaosim_wfstat_t *ws, const float *const opd_um[])
{
    ws->n_frames++;
    int is_samp = (ws->nsamp < CACAOSIM_WFSTAT_NB_SAMP &&
                   ws->n_frames >= ws->samp_ticks[ws->nsamp]);
    int is_clip = (ws->clip_len > 0u && ws->n_frames >= ws->clip_start &&
                   ws->clip_n < ws->clip_len);
    int is_full = (ws->full_cap > 0u && ws->full_n < ws->full_cap &&
                   ws->n_frames >= ws->full_next_frame);
    if (is_samp)
    {
        ws->samp_frame[ws->nsamp] = ws->n_frames;
    }

    if (ws->use_cuda)
    {
        cacaosim_wfstat_gpu_accum(ws->gpu, opd_um);
        if (is_samp)
        {
            float *maps[CACAOSIM_WFSTAT_NB_LOC];
            double frame[4 * CACAOSIM_WFSTAT_NB_LOC];
            for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
            {
                maps[k] = ws->samp_wf[k] + (uint64_t) ws->nsamp * ws->npix;
            }
            if (cacaosim_wfstat_gpu_fetch_sample(ws->gpu, maps, frame) == 0)
            {
                for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
                {
                    double svar                 = frame[4 * k + 1];
                    ws->samp_srms[ws->nsamp][k] = sqrt((svar > 0.0) ? svar : 0.0);
                    ws->samp_sptv[ws->nsamp][k] =
                        (ws->act_pix > 0) ? (frame[4 * k + 3] - frame[4 * k + 2]) : 0.0;
                }
                if (is_clip)
                {
                    uint64_t coff = (uint64_t) ws->clip_n * ws->npix;
                    memcpy(ws->clip_wf_raw + coff, maps[0], ws->npix * sizeof(float));
                    memcpy(ws->clip_wf_final + coff, maps[2], ws->npix * sizeof(float));
                }
                if (is_full)
                {
                    uint64_t foff = (uint64_t) ws->full_n * ws->npix;
                    memcpy(ws->full_wf_raw + foff, maps[0], ws->npix * sizeof(float));
                    memcpy(ws->full_wf_final + foff, maps[2], ws->npix * sizeof(float));
                }
            }
        }
        else if (is_clip)
        {
            uint64_t coff                          = (uint64_t) ws->clip_n * ws->npix;
            float   *cmaps[CACAOSIM_WFSTAT_NB_LOC] = {
                ws->clip_wf_raw + coff,
                NULL,
                ws->clip_wf_final + coff,
            };
            if (cacaosim_wfstat_gpu_fetch_sample(ws->gpu, cmaps, NULL) == 0 && is_full)
            {
                uint64_t foff = (uint64_t) ws->full_n * ws->npix;
                memcpy(ws->full_wf_raw + foff, ws->clip_wf_raw + coff, ws->npix * sizeof(float));
                memcpy(ws->full_wf_final + foff, ws->clip_wf_final + coff,
                       ws->npix * sizeof(float));
            }
        }
        else if (is_full)
        {
            uint64_t foff                          = (uint64_t) ws->full_n * ws->npix;
            float   *fmaps[CACAOSIM_WFSTAT_NB_LOC] = {
                ws->full_wf_raw + foff,
                NULL,
                ws->full_wf_final + foff,
            };
            cacaosim_wfstat_gpu_fetch_sample(ws->gpu, fmaps, NULL);
        }
    }
    else
    {
        double host_svar[CACAOSIM_WFSTAT_NB_LOC] = { 0.0, 0.0, 0.0 };
        for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
        {
            accum_host_plane(ws, k, opd_um[k], is_samp, &host_svar[k]);
            if (is_clip && k == 0 && ws->clip_wf_raw != NULL)
            {
                memcpy(ws->clip_wf_raw + (uint64_t) ws->clip_n * ws->npix, ws->wf_tmp,
                       ws->npix * sizeof(float));
            }
            else if (is_clip && k == 2 && ws->clip_wf_final != NULL)
            {
                memcpy(ws->clip_wf_final + (uint64_t) ws->clip_n * ws->npix, ws->wf_tmp,
                       ws->npix * sizeof(float));
            }
            if (is_full && k == 0 && ws->full_wf_raw != NULL)
            {
                memcpy(ws->full_wf_raw + (uint64_t) ws->full_n * ws->npix, ws->wf_tmp,
                       ws->npix * sizeof(float));
            }
            else if (is_full && k == 2 && ws->full_wf_final != NULL)
            {
                memcpy(ws->full_wf_final + (uint64_t) ws->full_n * ws->npix, ws->wf_tmp,
                       ws->npix * sizeof(float));
            }
        }
        if (ws->ctrl_active && ws->ctrl_q != NULL && ws->ctrl_v_tmp != NULL &&
            ws->ctrl_c_tmp != NULL)
        {
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, (int) ws->ctrl_n_tot,
                        CACAOSIM_WFSTAT_NB_LOC, (int) ws->act_pix, 1.0f, ws->ctrl_q,
                        (int) ws->act_pix, ws->ctrl_v_tmp, (int) ws->act_pix, 0.0f, ws->ctrl_c_tmp,
                        (int) ws->ctrl_n_tot);
            uint64_t slot = ws->n_frames - 1ULL;
            for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
            {
                uint32_t     m_lim = (k == 1 && ws->ctrl_n_stg1 > 0) ? ws->ctrl_n_stg1
                                                                     : ws->ctrl_n_tot;
                const float *ck    = ws->ctrl_c_tmp + (size_t) k * ws->ctrl_n_tot;
                double       in_v  = 0.0;
                for (uint32_t m = 0; m < m_lim; m++)
                {
                    double c = (double) ck[m];
                    in_v += c * c;
                }
                double tot_v = (host_svar[k] > 0.0) ? host_svar[k] : 0.0;
                if (in_v > tot_v)
                {
                    in_v = tot_v;
                }
                double out_v = tot_v - in_v;
                double in_r  = sqrt(in_v > 0.0 ? in_v : 0.0);
                double out_r = sqrt(out_v > 0.0 ? out_v : 0.0);
                ws->sum_in_rms[k] += in_r;
                ws->sum_in_var[k] += in_v;
                ws->sum_out_rms[k] += out_r;
                ws->sum_out_var[k] += out_v;
                if (slot < ws->ts_cap)
                {
                    if (ws->ts_in[k] != NULL)
                    {
                        ws->ts_in[k][slot] = (float) in_r;
                    }
                    if (ws->ts_out[k] != NULL)
                    {
                        ws->ts_out[k][slot] = (float) out_r;
                    }
                }
            }
        }
        ws->ts_len = (ws->n_frames <= ws->ts_cap) ? ws->n_frames : ws->ts_cap;
    }

    if (is_samp)
    {
        snapshot_extras(ws);
        ws->nsamp++;
    }

    if (is_clip)
    {
        uint32_t ci   = ws->clip_n;
        uint64_t coff = (uint64_t) ci * ws->npix;
        ws->clip_frame[ci] = ws->n_frames;
        const cacaosim_wfstat_extra_t *w2 = &ws->extras[WFSTAT_EXTRA_WFS2_IDX];
        ws->clip_wfs2_cnt0[ci]            = (w2->img.ID != -1 && w2->img.md != NULL)
                                                ? w2->img.md->cnt0
                                                : 0ULL;
        const cacaosim_wfstat_extra_t *d1 = &ws->extras[WFSTAT_EXTRA_DM1_OPD_IDX];
        if (ws->clip_has_dm1 && ws->clip_dm1_opd != NULL && d1->img.ID != -1 &&
            d1->img.im != NULL && (uint64_t) d1->nx * d1->ny == ws->npix)
        {
            piston_remove_opd_nm(ws, d1->img.im->array.F, ws->clip_dm1_opd + coff);
        }
        const cacaosim_wfstat_extra_t *d2 = &ws->extras[WFSTAT_EXTRA_DM2_OPD_IDX];
        if (ws->clip_has_dm2 && ws->clip_dm2_opd != NULL && d2->img.ID != -1 &&
            d2->img.im != NULL && (uint64_t) d2->nx * d2->ny == ws->npix)
        {
            piston_remove_opd_nm(ws, d2->img.im->array.F, ws->clip_dm2_opd + coff);
        }
        ws->clip_n++;
    }

    if (is_full)
    {
        uint32_t fi   = ws->full_n;
        uint64_t foff = (uint64_t) fi * ws->npix;
        ws->full_frame[fi] = ws->n_frames;
        const cacaosim_wfstat_extra_t *w2 = &ws->extras[WFSTAT_EXTRA_WFS2_IDX];
        ws->full_wfs2_cnt0[fi]            = (w2->img.ID != -1 && w2->img.md != NULL)
                                                ? w2->img.md->cnt0
                                                : 0ULL;
        const cacaosim_wfstat_extra_t *d1 = &ws->extras[WFSTAT_EXTRA_DM1_OPD_IDX];
        if (ws->full_has_dm1 && ws->full_dm1_opd != NULL && d1->img.ID != -1 &&
            d1->img.im != NULL && (uint64_t) d1->nx * d1->ny == ws->npix)
        {
            if (is_clip && ws->clip_has_dm1 && ws->clip_dm1_opd != NULL)
            {
                uint64_t coff = (uint64_t) (ws->clip_n - 1u) * ws->npix;
                memcpy(ws->full_dm1_opd + foff, ws->clip_dm1_opd + coff,
                       ws->npix * sizeof(float));
            }
            else
            {
                piston_remove_opd_nm(ws, d1->img.im->array.F, ws->full_dm1_opd + foff);
            }
        }
        const cacaosim_wfstat_extra_t *d2 = &ws->extras[WFSTAT_EXTRA_DM2_OPD_IDX];
        if (ws->full_has_dm2 && ws->full_dm2_opd != NULL && d2->img.ID != -1 &&
            d2->img.im != NULL && (uint64_t) d2->nx * d2->ny == ws->npix)
        {
            if (is_clip && ws->clip_has_dm2 && ws->clip_dm2_opd != NULL)
            {
                uint64_t coff = (uint64_t) (ws->clip_n - 1u) * ws->npix;
                memcpy(ws->full_dm2_opd + foff, ws->clip_dm2_opd + coff,
                       ws->npix * sizeof(float));
            }
            else
            {
                piston_remove_opd_nm(ws, d2->img.im->array.F, ws->full_dm2_opd + foff);
            }
        }
        ws->full_n++;
        ws->full_next_frame = 1ULL + (uint64_t) ((double) ws->full_n * ws->full_step + 0.5);
    }

    if (ws->snap_extra)
    {
        for (int c = 0; c < CACAOSIM_WFSTAT_NB_STREHL; c++)
        {
            cacaosim_wfstat_strehl_t *st = &ws->strehl[c];
            if (!st->active || st->img.md == NULL)
            {
                continue;
            }
            uint64_t cnt0 = st->img.md->cnt0;
            if (cnt0 != st->last_cnt0)
            {
                st->last_cnt0 = cnt0;
                double srl = 0.0, sri = 0.0, sra = 0.0;
                if (cacaosim_fpcam_shm_read_kw(st->img.im, &srl, &sri, &sra) &&
                    st->len < st->cap)
                {
                    st->frame[st->len]   = ws->n_frames;
                    st->sr_inst[st->len] = (float) sri;
                    st->sr_long[st->len] = (float) srl;
                    st->sr_axis[st->len] = (float) sra;
                    st->len++;
                }
            }
        }
        if (ws->coro_ts.active && ws->coro_ts.img.md != NULL)
        {
            uint64_t cnt0 = ws->coro_ts.img.md->cnt0;
            if (cnt0 != ws->coro_ts.last_cnt0)
            {
                ws->coro_ts.last_cnt0 = cnt0;
                double fl_l = 0.0, fl_i = 0.0, fp_l = 0.0, fp_i = 0.0, ft_l = 0.0, ft_i = 0.0;
                if (cacaosim_coro_shm_read_kw(ws->coro_ts.img.im, &fl_l, &fl_i, &fp_l, &fp_i,
                                              &ft_l, &ft_i) &&
                    ws->coro_ts.len < ws->coro_ts.cap)
                {
                    uint64_t idx                  = ws->coro_ts.len++;
                    ws->coro_ts.frame[idx]        = ws->n_frames;
                    ws->coro_ts.f_lowfs_inst[idx] = (float) fl_i;
                    ws->coro_ts.f_lowfs_long[idx] = (float) fl_l;
                    ws->coro_ts.f_pre_inst[idx]   = (float) fp_i;
                    ws->coro_ts.f_trans_inst[idx] = (float) ft_i;
                }
            }
        }
    }
}

static void write_loc_json(FILE *jfp, int k, const cacaosim_wfstat_t *ws, double denom,
                           double mean_rms, double quad_rms, double mm_rms, double mm_ptv,
                           double sm_rms, double sd_med, double sd_p95, double sd_max)
{
    double pv_vis = 2.0 * M_PI * (quad_rms * 1.0e-3) / 0.75;
    double pv_nir = 2.0 * M_PI * (quad_rms * 1.0e-3) / 1.65;
    fprintf(jfp, "    \"%s\": {\n", s_loc_keys[k]);
    fprintf(jfp, "      \"label\": \"%s\",\n", s_loc_desc[k]);
    fprintf(jfp, "      \"mean_spatial_rms_nm\": %.4f,\n", mean_rms);
    fprintf(jfp, "      \"quad_total_rms_nm\": %.4f,\n", quad_rms);
    fprintf(jfp, "      \"mean_map_rms_nm\": %.4f,\n", mm_rms);
    fprintf(jfp, "      \"mean_map_ptv_nm\": %.4f,\n", (ws->act_pix > 0) ? mm_ptv : 0.0);
    fprintf(jfp, "      \"std_map_rms_nm\": %.4f,\n", sm_rms);
    fprintf(jfp, "      \"std_map_median_nm\": %.4f,\n", sd_med);
    fprintf(jfp, "      \"std_map_p95_nm\": %.4f,\n", sd_p95);
    fprintf(jfp, "      \"std_map_max_nm\": %.4f,\n", sd_max);
    if (ws->ctrl_active)
    {
        uint32_t m_lim = (k == 1 && ws->ctrl_n_stg1 > 0) ? ws->ctrl_n_stg1 : ws->ctrl_n_tot;
        fprintf(jfp, "      \"ctrl_basis_rank\": %u,\n", m_lim);
        fprintf(jfp, "      \"in_ctrl_mean_rms_nm\": %.4f,\n", ws->sum_in_rms[k] / denom);
        fprintf(jfp, "      \"in_ctrl_quad_rms_nm\": %.4f,\n", sqrt(ws->sum_in_var[k] / denom));
        fprintf(jfp, "      \"out_ctrl_mean_rms_nm\": %.4f,\n", ws->sum_out_rms[k] / denom);
        fprintf(jfp, "      \"out_ctrl_quad_rms_nm\": %.4f,\n", sqrt(ws->sum_out_var[k] / denom));
    }
    fprintf(jfp, "      \"marechal_sr_vis_750nm\": %.6f,\n", exp(-(pv_vis * pv_vis)));
    fprintf(jfp, "      \"marechal_sr_nir_1650nm\": %.6f\n", exp(-(pv_nir * pv_nir)));
    fprintf(jfp, "    }%s\n", (k + 1 < CACAOSIM_WFSTAT_NB_LOC) ? "," : "");
}

static void write_samples_json(FILE *jfp, const cacaosim_wfstat_t *ws, double dt_phys)
{
    fprintf(jfp, "  \"samples\": [\n");
    for (int s = 0; s < ws->nsamp; s++)
    {
        fprintf(jfp, "    {\n");
        fprintf(jfp, "      \"sample_index\": %d,\n", s);
        fprintf(jfp, "      \"frame\": %" PRIu64 ",\n", ws->samp_frame[s]);
        fprintf(jfp, "      \"t_phys_s\": %.6f,\n", (double) ws->samp_frame[s] * dt_phys);
        for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
        {
            fprintf(jfp, "      \"%s\": {\"rms_nm\": %.3f, \"ptv_nm\": %.3f},\n", s_loc_keys[k],
                    ws->samp_srms[s][k], ws->samp_sptv[s][k]);
        }
        fprintf(jfp, "      \"extra\": {\n");
        int first_ex = 1;
        for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
        {
            if (!ws->extras[e].active)
            {
                continue;
            }
            const char *k1 = "rms", *k2 = "ptv";
            switch (s_extra_defs[e].kind)
            {
            case 0:
                k1 = "rms_nm";
                k2 = "ptv_nm";
                break;
            case 1:
                k1 = "rms_um";
                k2 = "ptv_um";
                break;
            case 2:
                k1 = "sum_adu";
                k2 = "max_adu";
                break;
            case 4:
                k1 = "peak_adu";
                k2 = "sum_adu";
                break;
            case 5:
                k1 = "peak_int";
                k2 = "sum_int";
                break;
            default:
                break;
            }
            fprintf(jfp, "%s        \"%s\": {\"%s\": %.4f, \"%s\": %.4f}", first_ex ? "" : ",\n",
                    s_extra_defs[e].key, k1, ws->extras[e].m1[s], k2, ws->extras[e].m2[s]);
            first_ex = 0;
        }
        fprintf(jfp, "\n      }\n");
        fprintf(jfp, "    }%s\n", (s + 1 < ws->nsamp) ? "," : "");
    }
    fprintf(jfp, "  ]\n");
}

static uint64_t write_wf_timeseries_csv(
    const cacaosim_wfstat_t *ws,
    const char              *out_dir,
    double                   dt_phys)
{
    if (ws->ts_len == 0)
    {
        return 0;
    }
    char tmp[1280], dst[1280];
    snprintf(tmp, sizeof(tmp), "%s/.wf_timeseries.csv.tmp", out_dir);
    snprintf(dst, sizeof(dst), "%s/wf_timeseries.csv", out_dir);
    FILE *fp = fopen(tmp, "w");
    if (fp == NULL)
    {
        return 0;
    }
    if (ws->ctrl_active && ws->ts_in[0] != NULL && ws->ts_out[0] != NULL)
    {
        fprintf(fp,
                "frame,t_phys_s,rms_raw_nm,rms_stg1_nm,rms_final_nm,"
                "rms_raw_in_nm,rms_raw_out_nm,rms_stg1_in_nm,rms_stg1_out_nm,"
                "rms_final_in_nm,rms_final_out_nm\n");
        for (uint64_t i = 0; i < ws->ts_len; i++)
        {
            uint64_t fr = i + 1ULL;
            fprintf(fp,
                    "%" PRIu64 ",%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", fr,
                    (double) fr * dt_phys, (double) ws->ts_rms[0][i], (double) ws->ts_rms[1][i],
                    (double) ws->ts_rms[2][i], (double) ws->ts_in[0][i], (double) ws->ts_out[0][i],
                    (double) ws->ts_in[1][i], (double) ws->ts_out[1][i], (double) ws->ts_in[2][i],
                    (double) ws->ts_out[2][i]);
        }
    }
    else
    {
        fprintf(fp, "frame,t_phys_s,rms_raw_nm,rms_stg1_nm,rms_final_nm\n");
        for (uint64_t i = 0; i < ws->ts_len; i++)
        {
            uint64_t fr = i + 1ULL;
            fprintf(fp, "%" PRIu64 ",%.6f,%.4f,%.4f,%.4f\n", fr, (double) fr * dt_phys,
                    (double) ws->ts_rms[0][i], (double) ws->ts_rms[1][i],
                    (double) ws->ts_rms[2][i]);
        }
    }
    fclose(fp);
    return (rename(tmp, dst) == 0) ? ws->ts_len : 0;
}

static uint64_t write_strehl_timeseries_csv(
    const cacaosim_wfstat_t *ws,
    const char              *out_dir,
    double                   dt_phys)
{
    uint64_t ao3k_max_len = 0;
    for (int c = 0; c < 6; c++)
    {
        if (ws->strehl[c].active && ws->strehl[c].len > ao3k_max_len)
        {
            ao3k_max_len = ws->strehl[c].len;
        }
    }
    if (ao3k_max_len > 0)
    {
        static const int pref[6] = { 2, 5, 1, 4, 0, 3 };
        int              ref_c   = 0;
        for (int p = 0; p < 6; p++)
        {
            int c = pref[p];
            if (ws->strehl[c].active && ws->strehl[c].len == ao3k_max_len)
            {
                ref_c = c;
                break;
            }
        }
        char tmp[1280], dst[1280];
        snprintf(tmp, sizeof(tmp), "%s/.strehl_timeseries.csv.tmp", out_dir);
        snprintf(dst, sizeof(dst), "%s/strehl_timeseries.csv", out_dir);
        FILE *fp = fopen(tmp, "w");
        if (fp == NULL)
        {
            return 0;
        }
        int has_coro = (ws->strehl[7].active && ws->strehl[7].len > 0) ? 1 : 0;
        fprintf(fp, "exposure,frame,t_phys_s");
        for (int c = 0; c < 6; c++)
        {
            fprintf(fp, ",%s_inst,%s_long", s_strehl_defs[c].key, s_strehl_defs[c].key);
        }
        if (has_coro)
        {
            fprintf(fp, ",nir_coro_inst,nir_coro_long,nir_coro_axis");
        }
        fprintf(fp, "\n");
        uint64_t cur[6]   = { 0, 0, 0, 0, 0, 0 };
        uint64_t cur_coro = 0;
        for (uint64_t r = 0; r < ao3k_max_len; r++)
        {
            uint64_t exp_id = r + 1U;
            uint64_t fr     = ws->strehl[ref_c].frame[r];
            fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%.6f", exp_id, fr, (double) fr * dt_phys);
            for (int c = 0; c < 6; c++)
            {
                const cacaosim_wfstat_strehl_t *st = &ws->strehl[c];
                if (!st->active)
                {
                    fprintf(fp, ",nan,nan");
                    continue;
                }
                while (cur[c] + 1U < st->len && st->frame[cur[c]] < fr)
                {
                    cur[c]++;
                }
                if (cur[c] < st->len && st->frame[cur[c]] == fr)
                {
                    fprintf(fp, ",%.6f,%.6f", (double) st->sr_inst[cur[c]],
                            (double) st->sr_long[cur[c]]);
                    cur[c]++;
                }
                else
                {
                    fprintf(fp, ",nan,nan");
                }
            }
            if (has_coro)
            {
                const cacaosim_wfstat_strehl_t *sc = &ws->strehl[7];
                while (cur_coro + 1U < sc->len && sc->frame[cur_coro] < fr)
                {
                    cur_coro++;
                }
                if (cur_coro < sc->len && sc->frame[cur_coro] == fr)
                {
                    fprintf(fp, ",%.6e,%.6e,%.6e", (double) sc->sr_inst[cur_coro],
                            (double) sc->sr_long[cur_coro], (double) sc->sr_axis[cur_coro]);
                    cur_coro++;
                }
                else
                {
                    fprintf(fp, ",nan,nan,nan");
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        return (rename(tmp, dst) == 0) ? ao3k_max_len : 0;
    }

    const cacaosim_wfstat_strehl_t *sci = &ws->strehl[6];
    if (sci->active && sci->len > 0)
    {
        char tmp[1280], dst[1280];
        snprintf(tmp, sizeof(tmp), "%s/.strehl_timeseries.csv.tmp", out_dir);
        snprintf(dst, sizeof(dst), "%s/strehl_timeseries.csv", out_dir);
        FILE *fp = fopen(tmp, "w");
        if (fp == NULL)
        {
            return 0;
        }
        fprintf(fp, "exposure,frame,t_phys_s,sci_inst,sci_long\n");
        for (uint64_t r = 0; r < sci->len; r++)
        {
            uint64_t exp_id = r + 1U;
            uint64_t fr     = sci->frame[r];
            fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%.6f,%.6f,%.6f\n", exp_id, fr,
                    (double) fr * dt_phys, (double) sci->sr_inst[r], (double) sci->sr_long[r]);
        }
        fclose(fp);
        return (rename(tmp, dst) == 0) ? sci->len : 0;
    }
    return 0;
}

void cacaosim_wfstat_sync_gpu(cacaosim_wfstat_t *ws)
{
    if (ws == NULL)
    {
        return;
    }
    if (ws->use_cuda && ws->gpu != NULL)
    {
        double   acc[2 * CACAOSIM_WFSTAT_NB_LOC];
        uint64_t gpu_ts_len = 0;
        if (cacaosim_wfstat_gpu_fetch_sums(ws->gpu, ws->sum_wf, ws->sum2_wf, acc, ws->ts_rms,
                                           &gpu_ts_len) == 0)
        {
            for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
            {
                ws->sum_srms[k] = acc[2 * k + 0];
                ws->sum_svar[k] = acc[2 * k + 1];
            }
            ws->ts_len = gpu_ts_len;
        }
        if (ws->ctrl_active)
        {
            double acc_ctrl[4 * CACAOSIM_WFSTAT_NB_LOC];
            memset(acc_ctrl, 0, sizeof(acc_ctrl));
            if (cacaosim_wfstat_gpu_fetch_ctrl(ws->gpu, acc_ctrl, ws->ts_in, ws->ts_out) == 0)
            {
                for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
                {
                    ws->sum_in_rms[k]  = acc_ctrl[4 * k + 0];
                    ws->sum_in_var[k]  = acc_ctrl[4 * k + 1];
                    ws->sum_out_rms[k] = acc_ctrl[4 * k + 2];
                    ws->sum_out_var[k] = acc_ctrl[4 * k + 3];
                }
            }
        }
        cacaosim_wfstat_gpu_free(ws->gpu);
        ws->gpu      = NULL;
        ws->use_cuda = 0;
    }
    /* Release process-global MILK SHM handles on caller thread so background disk write is pure */
    for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
    {
        if (ws->extras[e].img.ID != -1)
        {
            imgid_free(&ws->extras[e].img);
            ws->extras[e].img.ID = -1;
        }
    }
    for (int c = 0; c < CACAOSIM_WFSTAT_NB_STREHL; c++)
    {
        if (ws->strehl[c].img.ID != -1)
        {
            imgid_free(&ws->strehl[c].img);
            ws->strehl[c].img.ID = -1;
        }
    }
    if (ws->coro_ts.img.ID != -1)
    {
        imgid_free(&ws->coro_ts.img);
        ws->coro_ts.img.ID = -1;
    }
}

void cacaosim_wfstat_record_perturb(
    cacaosim_wfstat_t                      *ws,
    const cacaosim_wfstat_perturb_sample_t *samp)
{
    if (ws == NULL || samp == NULL || ws->pert_ts == NULL || ws->pert_len >= ws->ts_cap)
    {
        return;
    }
    ws->pert_ts[ws->pert_len++] = *samp;
    if (fabsf(samp->wind_scale - 1.0f) > 1.0e-4f || fabsf(samp->wind_deg) > 1.0e-3f ||
        samp->burst_env > 1.0e-4f || fabsf(samp->cn2_scale - 1.0f) > 1.0e-4f ||
        samp->vib_rms_nm > 1.0e-3f || samp->petal_rms_nm > 1.0e-3f)
    {
        ws->pert_any_active = 1;
    }
}

static uint64_t write_perturb_timeseries_csv(
    const cacaosim_wfstat_t *ws,
    const char              *out_dir,
    double                   dt_phys)
{
    if (ws->pert_ts == NULL || ws->pert_len == 0 || !ws->pert_any_active)
    {
        return 0;
    }
    char tmp[1280], dst[1280];
    snprintf(tmp, sizeof(tmp), "%s/.perturb_timeseries.csv.tmp", out_dir);
    snprintf(dst, sizeof(dst), "%s/perturb_timeseries.csv", out_dir);
    FILE *fp = fopen(tmp, "w");
    if (fp == NULL)
    {
        return 0;
    }
    fprintf(fp, "frame,t_phys_s,wind_scale,wind_deg,burst_env,cn2_scale,"
                "vib_rms_nm,vib_tip_nm,vib_tilt_nm,vib_foc_nm,"
                "petal_rms_nm,petal_ptv_nm,petal_p0_nm,petal_p1_nm,petal_p2_nm,petal_p3_nm\n");
    for (uint64_t r = 0; r < ws->pert_len; r++)
    {
        const cacaosim_wfstat_perturb_sample_t *p = &ws->pert_ts[r];
        fprintf(fp,
                "%" PRIu64 ",%.6f,%.4f,%.3f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,"
                "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                r + 1U, (double) (r + 1U) * dt_phys, (double) p->wind_scale, (double) p->wind_deg,
                (double) p->burst_env, (double) p->cn2_scale, (double) p->vib_rms_nm,
                (double) p->vib_tip_nm, (double) p->vib_tilt_nm, (double) p->vib_foc_nm,
                (double) p->petal_rms_nm, (double) p->petal_ptv_nm, (double) p->petal_p_nm[0],
                (double) p->petal_p_nm[1], (double) p->petal_p_nm[2], (double) p->petal_p_nm[3]);
    }
    fclose(fp);
    return (rename(tmp, dst) == 0) ? ws->pert_len : 0;
}

static uint64_t write_coro_light_timeseries_csv(
    const cacaosim_wfstat_t *ws,
    const char              *out_dir,
    double                   dt_phys)
{
    if (!ws->coro_ts.active || ws->coro_ts.len == 0)
    {
        return 0;
    }
    char tmp[1280], dst[1280];
    snprintf(tmp, sizeof(tmp), "%s/.lowfs_light_timeseries.csv.tmp", out_dir);
    snprintf(dst, sizeof(dst), "%s/lowfs_light_timeseries.csv", out_dir);
    FILE *fp = fopen(tmp, "w");
    if (fp == NULL)
    {
        return 0;
    }
    fprintf(fp, "sample,frame,t_phys_s,frac_lowfs_inst,frac_lowfs_long,"
                "frac_pre_lyot_inst,frac_lyot_trans_inst,frac_fpm_abs_inst\n");
    for (uint64_t r = 0; r < ws->coro_ts.len; r++)
    {
        uint64_t fr    = ws->coro_ts.frame[r];
        double   f_pre = (double) ws->coro_ts.f_pre_inst[r];
        double   f_abs = (1.0 - f_pre > 0.0) ? (1.0 - f_pre) : 0.0;
        fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", r + 1U, fr,
                (double) fr * dt_phys, (double) ws->coro_ts.f_lowfs_inst[r],
                (double) ws->coro_ts.f_lowfs_long[r], f_pre,
                (double) ws->coro_ts.f_trans_inst[r], f_abs);
    }
    fclose(fp);
    return (rename(tmp, dst) == 0) ? ws->coro_ts.len : 0;
}

int cacaosim_wfstat_write(cacaosim_wfstat_t *ws, const char *out_dir, double dt_phys)
{
    uint64_t npix = ws->npix;
    if (ws->use_cuda)
    {
        cacaosim_wfstat_sync_gpu(ws);
    }

    mkdir_p(out_dir);
    double denom = (ws->n_frames > 0) ? (double) ws->n_frames : 1.0;
    char   fpath[1280];

    snprintf(fpath, sizeof(fpath), "%s/wf_pupil_mask.fits", out_dir);
    cacaosim_mask_save_fits(fpath, ws->pup_amp, ws->nwf, ws->nwf, 1, NULL);

    float *mean_f = (float *) calloc(npix, sizeof(float));
    float *std_f  = (float *) calloc(npix, sizeof(float));

    /* Write to a temp file and rename so pollers never read a partial wf_stats.json */
    char jtmp[1280];
    snprintf(jtmp, sizeof(jtmp), "%s/.wf_stats.json.tmp", out_dir);
    FILE *jfp = fopen(jtmp, "w");
    if (jfp)
    {
        fprintf(jfp, "{\n");
        fprintf(jfp, "  \"n_frames\": %" PRIu64 ",\n", ws->n_frames);
        fprintf(jfp, "  \"phys_time_recorded_s\": %.6f,\n", (double) ws->n_frames * dt_phys);
        fprintf(jfp, "  \"active_pupil_pixels\": %" PRIu64 ",\n", ws->act_pix);
        fprintf(jfp, "  \"units\": \"nm\",\n");
        fprintf(jfp, "  \"locations\": {\n");
    }

    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        double   w_mean2 = 0.0, w_std2 = 0.0;
        double   m_min = 1e30, m_max = -1e30;
        uint64_t nsort = 0;
        for (uint64_t i = 0; i < npix; i++)
        {
            if (ws->mask[i] && mean_f && std_f)
            {
                double mn = ws->sum_wf[k][i] / denom;
                double vr = ws->sum2_wf[k][i] / denom - mn * mn;
                double sd = sqrt((vr > 0.0) ? vr : 0.0);
                mean_f[i] = (float) mn;
                std_f[i]  = (float) sd;
                w_mean2 += ws->weights[i] * mn * mn;
                w_std2 += ws->weights[i] * sd * sd;
                m_min               = (mn < m_min) ? mn : m_min;
                m_max               = (mn > m_max) ? mn : m_max;
                ws->sortbuf[nsort++] = sd;
            }
            else if (mean_f && std_f)
            {
                mean_f[i] = 0.0f;
                std_f[i]  = 0.0f;
            }
        }
        snprintf(fpath, sizeof(fpath), "%s/wf_mean_%s.fits", out_dir, s_loc_keys[k]);
        cacaosim_mask_save_fits(fpath, mean_f, ws->nwf, ws->nwf, 1, NULL);
        snprintf(fpath, sizeof(fpath), "%s/wf_std_%s.fits", out_dir, s_loc_keys[k]);
        cacaosim_mask_save_fits(fpath, std_f, ws->nwf, ws->nwf, 1, NULL);
        if (ws->nsamp > 0)
        {
            snprintf(fpath, sizeof(fpath), "%s/wf_samples_%s.fits", out_dir, s_loc_keys[k]);
            cacaosim_mask_save_fits(fpath, ws->samp_wf[k], ws->nwf, ws->nwf,
                                    (uint32_t) ws->nsamp, NULL);
        }

        double sd_med = 0.0, sd_p95 = 0.0, sd_max = 0.0;
        if (nsort > 0)
        {
            qsort(ws->sortbuf, nsort, sizeof(double), cmp_dbl);
            sd_med = ws->sortbuf[nsort / 2];
            sd_p95 = ws->sortbuf[(nsort * 95) / 100];
            sd_max = ws->sortbuf[nsort - 1];
        }
        if (jfp)
        {
            write_loc_json(jfp, k, ws, denom, ws->sum_srms[k] / denom,
                           sqrt(ws->sum_svar[k] / denom), sqrt(w_mean2), m_max - m_min,
                           sqrt(w_std2), sd_med, sd_p95, sd_max);
        }
    }

    for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
    {
        cacaosim_wfstat_extra_t *es = &ws->extras[e];
        if (es->active && ws->nsamp > 0)
        {
            snprintf(fpath, sizeof(fpath), "%s/%s_samples.fits", out_dir, s_extra_defs[e].key);
            cacaosim_mask_save_fits(fpath, es->cube, es->nx, es->ny, (uint32_t) ws->nsamp, NULL);
        }
    }

    /* Flush CSV time series before publishing wf_stats.json so pollers see complete artifacts */
    uint64_t wf_rows   = write_wf_timeseries_csv(ws, out_dir, dt_phys);
    uint64_t sr_rows   = write_strehl_timeseries_csv(ws, out_dir, dt_phys);
    uint64_t coro_rows = write_coro_light_timeseries_csv(ws, out_dir, dt_phys);
    uint64_t pert_rows = write_perturb_timeseries_csv(ws, out_dir, dt_phys);

    if (ws->clip_n > 0u)
    {
        if (ws->clip_wf_raw != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/wf_clip_raw.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->clip_wf_raw, ws->nwf, ws->nwf, ws->clip_n, NULL);
        }
        if (ws->clip_wf_final != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/wf_clip_final.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->clip_wf_final, ws->nwf, ws->nwf, ws->clip_n, NULL);
        }
        if (ws->clip_has_dm1 && ws->clip_dm1_opd != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/dm_clip_opd_stg1.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->clip_dm1_opd, ws->nwf, ws->nwf, ws->clip_n, NULL);
        }
        if (ws->clip_has_dm2 && ws->clip_dm2_opd != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/dm_clip_opd_stg2.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->clip_dm2_opd, ws->nwf, ws->nwf, ws->clip_n, NULL);
        }
        char ctmp[1280];
        snprintf(ctmp, sizeof(ctmp), "%s/.wf_clip_meta.json.tmp", out_dir);
        FILE *cfp = fopen(ctmp, "w");
        if (cfp != NULL)
        {
            fprintf(cfp, "{\n");
            fprintf(cfp, "  \"n_frames\": %u,\n", ws->clip_n);
            fprintf(cfp, "  \"nwf\": %u,\n", ws->nwf);
            fprintf(cfp, "  \"dt_phys_s\": %.8f,\n", dt_phys);
            fprintf(cfp, "  \"clip_start_frame\": %" PRIu64 ",\n", ws->clip_start);
            fprintf(cfp, "  \"has_dm1_opd\": %s,\n", ws->clip_has_dm1 ? "true" : "false");
            fprintf(cfp, "  \"has_dm2_opd\": %s,\n", ws->clip_has_dm2 ? "true" : "false");
            fprintf(cfp, "  \"frames\": [");
            for (uint32_t ci = 0; ci < ws->clip_n; ci++)
            {
                fprintf(cfp, "%s%" PRIu64, (ci > 0u) ? "," : "", ws->clip_frame[ci]);
            }
            fprintf(cfp, "],\n");
            fprintf(cfp, "  \"wfs2_cnt0\": [");
            for (uint32_t ci = 0; ci < ws->clip_n; ci++)
            {
                fprintf(cfp, "%s%" PRIu64, (ci > 0u) ? "," : "", ws->clip_wfs2_cnt0[ci]);
            }
            fprintf(cfp, "]\n}\n");
            fclose(cfp);
            snprintf(fpath, sizeof(fpath), "%s/wf_clip_meta.json", out_dir);
            (void) rename(ctmp, fpath);
        }
    }

    if (ws->full_n > 0u)
    {
        if (ws->full_wf_raw != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/wf_full_raw.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->full_wf_raw, ws->nwf, ws->nwf, ws->full_n, NULL);
        }
        if (ws->full_wf_final != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/wf_full_final.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->full_wf_final, ws->nwf, ws->nwf, ws->full_n, NULL);
        }
        if (ws->full_has_dm1 && ws->full_dm1_opd != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/dm_full_opd_stg1.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->full_dm1_opd, ws->nwf, ws->nwf, ws->full_n, NULL);
        }
        if (ws->full_has_dm2 && ws->full_dm2_opd != NULL)
        {
            snprintf(fpath, sizeof(fpath), "%s/dm_full_opd_stg2.fits", out_dir);
            cacaosim_mask_save_fits(fpath, ws->full_dm2_opd, ws->nwf, ws->nwf, ws->full_n, NULL);
        }
        char ftmp[1280];
        snprintf(ftmp, sizeof(ftmp), "%s/.wf_full_meta.json.tmp", out_dir);
        FILE *ffp = fopen(ftmp, "w");
        if (ffp != NULL)
        {
            fprintf(ffp, "{\n");
            fprintf(ffp, "  \"n_frames\": %u,\n", ws->full_n);
            fprintf(ffp, "  \"nwf\": %u,\n", ws->nwf);
            fprintf(ffp, "  \"dt_phys_s\": %.8f,\n", dt_phys);
            fprintf(ffp, "  \"full_step_frames\": %.6f,\n", ws->full_step);
            fprintf(ffp, "  \"has_dm1_opd\": %s,\n", ws->full_has_dm1 ? "true" : "false");
            fprintf(ffp, "  \"has_dm2_opd\": %s,\n", ws->full_has_dm2 ? "true" : "false");
            fprintf(ffp, "  \"frames\": [");
            for (uint32_t fi = 0; fi < ws->full_n; fi++)
            {
                fprintf(ffp, "%s%" PRIu64, (fi > 0u) ? "," : "", ws->full_frame[fi]);
            }
            fprintf(ffp, "],\n");
            fprintf(ffp, "  \"wfs2_cnt0\": [");
            for (uint32_t fi = 0; fi < ws->full_n; fi++)
            {
                fprintf(ffp, "%s%" PRIu64, (fi > 0u) ? "," : "", ws->full_wfs2_cnt0[fi]);
            }
            fprintf(ffp, "]\n}\n");
            fclose(ffp);
            snprintf(fpath, sizeof(fpath), "%s/wf_full_meta.json", out_dir);
            (void) rename(ftmp, fpath);
        }
    }

    int rc = -1;
    if (jfp)
    {
        fprintf(jfp, "  },\n");
        fprintf(jfp, "  \"timeseries_files\": {\n");
        fprintf(jfp, "    \"wf_rms_csv\": %s,\n",
                (wf_rows > 0) ? "\"wf_timeseries.csv\"" : "null");
        fprintf(jfp, "    \"wf_rms_rows\": %" PRIu64 ",\n", wf_rows);
        fprintf(jfp, "    \"strehl_csv\": %s,\n",
                (sr_rows > 0) ? "\"strehl_timeseries.csv\"" : "null");
        fprintf(jfp, "    \"strehl_rows\": %" PRIu64 ",\n", sr_rows);
        fprintf(jfp, "    \"lowfs_light_csv\": %s,\n",
                (coro_rows > 0) ? "\"lowfs_light_timeseries.csv\"" : "null");
        fprintf(jfp, "    \"lowfs_light_rows\": %" PRIu64 ",\n", coro_rows);
        fprintf(jfp, "    \"perturb_csv\": %s,\n",
                (pert_rows > 0) ? "\"perturb_timeseries.csv\"" : "null");
        fprintf(jfp, "    \"perturb_rows\": %" PRIu64 "\n", pert_rows);
        fprintf(jfp, "  },\n");
        if (ws->ctrl_active)
        {
            fprintf(jfp, "  \"control_modes\": {\n");
            fprintf(jfp, "    \"stg1_cm_modes\": %u,\n", ws->ctrl_raw_n_stg1);
            fprintf(jfp, "    \"stg2_cm_modes\": %u,\n", ws->ctrl_raw_n_stg2);
            fprintf(jfp, "    \"stg1_basis_rank\": %u,\n", ws->ctrl_n_stg1);
            fprintf(jfp, "    \"final_basis_rank\": %u\n", ws->ctrl_n_tot);
            fprintf(jfp, "  },\n");
        }
        write_samples_json(jfp, ws, dt_phys);
        fprintf(jfp, "}\n");
        fclose(jfp);
        snprintf(fpath, sizeof(fpath), "%s/wf_stats.json", out_dir);
        rc = (rename(jtmp, fpath) == 0) ? 0 : -1;
    }
    free(mean_f);
    free(std_f);
    return rc;
}

void cacaosim_wfstat_free(cacaosim_wfstat_t *ws)
{
    if (ws == NULL)
    {
        return;
    }
    if (ws->gpu != NULL)
    {
        cacaosim_wfstat_gpu_free(ws->gpu);
        ws->gpu = NULL;
    }
    for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
    {
        free(ws->extras[e].cube);
        if (ws->extras[e].img.ID != -1)
        {
            imgid_free(&ws->extras[e].img);
            ws->extras[e].img.ID = -1;
        }
    }
    for (int c = 0; c < CACAOSIM_WFSTAT_NB_STREHL; c++)
    {
        free(ws->strehl[c].frame);
        free(ws->strehl[c].sr_inst);
        free(ws->strehl[c].sr_long);
        free(ws->strehl[c].sr_axis);
        if (ws->strehl[c].img.ID != -1)
        {
            imgid_free(&ws->strehl[c].img);
            ws->strehl[c].img.ID = -1;
        }
    }
    free(ws->coro_ts.frame);
    free(ws->coro_ts.f_lowfs_inst);
    free(ws->coro_ts.f_lowfs_long);
    free(ws->coro_ts.f_pre_inst);
    free(ws->coro_ts.f_trans_inst);
    free(ws->pert_ts);
    if (ws->coro_ts.img.ID != -1)
    {
        imgid_free(&ws->coro_ts.img);
        ws->coro_ts.img.ID = -1;
    }
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        free(ws->sum_wf[k]);
        free(ws->sum2_wf[k]);
        free(ws->samp_wf[k]);
        free(ws->ts_rms[k]);
        free(ws->ts_in[k]);
        free(ws->ts_out[k]);
    }
    free(ws->clip_wf_raw);
    free(ws->clip_wf_final);
    free(ws->clip_dm1_opd);
    free(ws->clip_dm2_opd);
    free(ws->clip_frame);
    free(ws->clip_wfs2_cnt0);
    free(ws->full_wf_raw);
    free(ws->full_wf_final);
    free(ws->full_dm1_opd);
    free(ws->full_dm2_opd);
    free(ws->full_frame);
    free(ws->full_wfs2_cnt0);
    free(ws->act_idx);
    free(ws->act_sw);
    free(ws->ctrl_q);
    free(ws->ctrl_v_tmp);
    free(ws->ctrl_c_tmp);
    free(ws->mask);
    free(ws->weights);
    free(ws->pup_amp);
    free(ws->wf_tmp);
    free(ws->sortbuf);
    memset(ws, 0, sizeof(*ws));
    for (int e = 0; e < CACAOSIM_WFSTAT_NB_EXTRA; e++)
    {
        ws->extras[e].img.ID = -1;
    }
    for (int c = 0; c < CACAOSIM_WFSTAT_NB_STREHL; c++)
    {
        ws->strehl[c].img.ID = -1;
    }
    ws->coro_ts.img.ID = -1;
}
