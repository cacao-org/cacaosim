// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfstat_core.c
 * @brief   Shared pupil wavefront statistics engine: CPU accumulation, extras, and file writer
 *
 * See cacaosim_wfstat_core.h. The CUDA accumulation kernels live in cacaosim_wfstat_cuda.cu.
 */

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "common/cacaosim_mask.h"
#include "common/cacaosim_wfstat_core.h"

typedef struct
{
    const char *key;
    const char *stream_name;
    int         kind; /* 0=dm_opd, 1=dm_cmd, 2=wfs_raw, 3=wfs_sig, 4=psf_inst */
} wfstat_extra_def_t;

static const wfstat_extra_def_t s_extra_defs[CACAOSIM_WFSTAT_NB_EXTRA] = {
    { "wfs_raw_stg1", "csim_wfs_nir", 2 },        { "wfs_sig_stg1", "aol1_imWFS2", 3 },
    { "wfs_raw_stg2", "csim_wfs_vis", 2 },        { "wfs_sig_stg2", "aol2_imWFS2", 3 },
    { "dm_cmd_stg1", "dm01disp", 1 },             { "dm_opd_stg1", "csim_dm01_opd", 0 },
    { "dm_cmd_stg2", "dm02disp", 1 },             { "dm_opd_stg2", "csim_dm02_opd", 0 },
    { "psf_inst_vis_raw", "psf_vis_raw", 4 },     { "psf_inst_vis_stg1", "psf_vis_stg1", 4 },
    { "psf_inst_vis_final", "psf_vis_final", 4 }, { "psf_inst_nir_raw", "psf_nir_raw", 4 },
    { "psf_inst_nir_stg1", "psf_nir_stg1", 4 },   { "psf_inst_nir_final", "psf_nir_final", 4 },
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
    ws->nwf        = nwf;
    ws->npix       = (uint64_t) nwf * nwf;
    ws->snap_extra = snap_extra;
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
        if (!ws->sum_wf[k] || !ws->sum2_wf[k] || !ws->samp_wf[k])
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
    }

    if (use_cuda)
    {
        ws->gpu = cacaosim_wfstat_gpu_create(ws->weights, npix, CACAOSIM_WFSTAT_NB_LOC,
                                             cuda_stream);
        if (ws->gpu == NULL)
        {
            cacaosim_wfstat_free(ws);
            return -1;
        }
        ws->use_cuda = 1;
    }
    return 0;
}

/* Piston-remove one host plane, accumulate sums, and optionally capture the sample map */
static void accum_host_plane(cacaosim_wfstat_t *ws, int k, const float *opd, int is_samp)
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
    double srms = sqrt((svar > 0.0) ? svar : 0.0);
    ws->sum_svar[k] += svar;
    ws->sum_srms[k] += srms;
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

void cacaosim_wfstat_frame(cacaosim_wfstat_t *ws, const float *const opd_um[])
{
    ws->n_frames++;
    int is_samp = (ws->nsamp < CACAOSIM_WFSTAT_NB_SAMP &&
                   ws->n_frames >= ws->samp_ticks[ws->nsamp]);
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
            }
        }
    }
    else
    {
        for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
        {
            accum_host_plane(ws, k, opd_um[k], is_samp);
        }
    }

    if (is_samp)
    {
        snapshot_extras(ws);
        ws->nsamp++;
    }
}

static void write_loc_json(FILE *jfp, int k, const cacaosim_wfstat_t *ws, double mean_rms,
                           double quad_rms, double mm_rms, double mm_ptv, double sm_rms,
                           double sd_med, double sd_p95, double sd_max)
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

int cacaosim_wfstat_write(cacaosim_wfstat_t *ws, const char *out_dir, double dt_phys)
{
    uint64_t npix = ws->npix;
    if (ws->use_cuda)
    {
        double acc[2 * CACAOSIM_WFSTAT_NB_LOC];
        if (cacaosim_wfstat_gpu_fetch_sums(ws->gpu, ws->sum_wf, ws->sum2_wf, acc) == 0)
        {
            for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
            {
                ws->sum_srms[k] = acc[2 * k + 0];
                ws->sum_svar[k] = acc[2 * k + 1];
            }
        }
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
            write_loc_json(jfp, k, ws, ws->sum_srms[k] / denom, sqrt(ws->sum_svar[k] / denom),
                           sqrt(w_mean2), m_max - m_min, sqrt(w_std2), sd_med, sd_p95, sd_max);
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

    int rc = -1;
    if (jfp)
    {
        fprintf(jfp, "  },\n");
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
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        free(ws->sum_wf[k]);
        free(ws->sum2_wf[k]);
        free(ws->samp_wf[k]);
    }
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
}
