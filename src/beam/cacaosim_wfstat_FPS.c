// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfstat_FPS.c
 * @brief   MILK_RUNNABLE semaphore-triggered pupil wavefront stats & FITS cube recorder
 *
 * Connects the upstream optical beams (.beam_raw, .beam_stg1, .beam_final) plus optional
 * WFS/DM/PSF 2D diagnostic streams, accumulates pupil-piston-removed mean/std OPD maps in
 * nanometers [nm], captures synchronized 5-slice sample FITS cubes, and writes wf_stats.json
 * on loop exit via cacaosim_mask_save_fits().
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
#include "beam/cacaosim_wfstat_FPS.h"
#include "common/cacaosim_mask.h"

#define WFSTAT_NB_LOC 3
#define WFSTAT_NB_SAMP 5
#define WFSTAT_NB_EXTRA 14

static char     param_trig_beam[FUNCTION_PARAMETER_STRMAXLEN]  = "b_post_stg2";
static char     param_beam_raw[FUNCTION_PARAMETER_STRMAXLEN]   = "b_pup";
static char     param_beam_stg1[FUNCTION_PARAMETER_STRMAXLEN]  = "b_post_stg1";
static char     param_beam_final[FUNCTION_PARAMETER_STRMAXLEN] = "b_post_stg2";
static char     param_outdir[FUNCTION_PARAMETER_STRMAXLEN]     = ".";
static uint64_t param_target_ticks                             = 20000;
static float    param_dt_phys                                  = 0.0005f;
static int64_t  param_snap_extra                               = 1;

#define CACAOSIM_WFSTAT_PARAMS(X)                                                      \
    X(".trig_beam", param_trig_beam, FPTYPE_STREAMNAME, 1,                             \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                    \
      "Trigger optical beam stream (usually post-Stage-2)")                            \
    X(".beam_raw", param_beam_raw, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,             \
      "Raw entrance pupil beam stream name")                                           \
    X(".beam_stg1", param_beam_stg1, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,           \
      "Post-Stage-1 optical beam stream name")                                         \
    X(".beam_final", param_beam_final, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,         \
      "Post-Stage-2 final optical beam stream name")                                   \
    X(".outdir", param_outdir, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                 \
      "Output directory (<64 chars; or set WFSTAT_OUTDIR env)")                         \
    X(".target_ticks", &param_target_ticks, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,    \
      "Planned integration tick count for 5 sample markers")                           \
    X(".dt_phys", &param_dt_phys, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,             \
      "Physical frame interval [s] (default: 0.0005)")                                 \
    X(".snap_extra", &param_snap_extra, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,         \
      "Capture synchronized WFS/DM/PSF 2D snapshot cubes")

MILK_PARAMS_DEFINE(cacaosim_wfstat, CACAOSIM_WFSTAT_PARAMS);

typedef struct
{
    const char *key;
    const char *stream_name;
    const char *desc;
    int         kind; /* 0=dm_opd, 1=dm_cmd, 2=wfs_raw, 3=wfs_sig, 4=psf_inst */
} wfstat_extra_def_t;

static const wfstat_extra_def_t s_extra_defs[WFSTAT_NB_EXTRA] = {
    { "wfs_raw_stg1", "csim_wfs_nir", "AO3k J-band PyWFS raw detector", 2 },
    { "wfs_sig_stg1", "aol1_imWFS2", "AO3k PyWFS ref-subtracted signal", 3 },
    { "wfs_raw_stg2", "csim_wfs_vis", "SCExAO Vis PyWFS raw detector", 2 },
    { "wfs_sig_stg2", "aol2_imWFS2", "SCExAO PyWFS ref-subtracted signal", 3 },
    { "dm_cmd_stg1", "dm01disp", "AO3k 64x64 DM stroke command", 1 },
    { "dm_opd_stg1", "csim_dm01_opd", "AO3k 256x256 reflective DM OPD", 0 },
    { "dm_cmd_stg2", "dm02disp", "SCExAO 50x50 DM stroke command", 1 },
    { "dm_opd_stg2", "csim_dm02_opd", "SCExAO 256x256 reflective DM OPD", 0 },
    { "psf_inst_vis_raw", "psf_vis_raw", "Inst Vis PSF Before Stage 1", 4 },
    { "psf_inst_vis_stg1", "psf_vis_stg1", "Inst Vis PSF Post-Stage 1", 4 },
    { "psf_inst_vis_final", "psf_vis_final", "Inst Vis PSF Post-Stage 2", 4 },
    { "psf_inst_nir_raw", "psf_nir_raw", "Inst NIR PSF Before Stage 1", 4 },
    { "psf_inst_nir_stg1", "psf_nir_stg1", "Inst NIR PSF Post-Stage 1", 4 },
    { "psf_inst_nir_final", "psf_nir_final", "Inst NIR PSF Post-Stage 2", 4 },
};

typedef struct
{
    IMGID    img;
    uint32_t nx;
    uint32_t ny;
    float   *cube;
    double   m1[WFSTAT_NB_SAMP];
    double   m2[WFSTAT_NB_SAMP];
    int      active;
} wfstat_extra_state_t;

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

static MILK_HOT errno_t wfstat_compute(const MILK_RUNNABLE *self)
{
    const char *env_out  = getenv("WFSTAT_OUTDIR");
    const char *base_out = (env_out && env_out[0] != '\0') ? env_out : param_outdir;
    if (!base_out || base_out[0] == '\0')
    {
        base_out = ".";
    }
    char out_dir[1024];
    snprintf(out_dir, sizeof(out_dir), "%s", base_out);
    mkdir_p(out_dir);

    const char *loc_keys[WFSTAT_NB_LOC] = { "raw", "stg1", "final" };
    const char *loc_desc[WFSTAT_NB_LOC] = {
        "Before Stage 1 (Entrance Pupil)",
        "Between Stages 1 & 2 (Post-AO3k)",
        "After Stage 2 (Post-SCExAO)",
    };
    const char *loc_snames[WFSTAT_NB_LOC] = {
        param_beam_raw,
        param_beam_stg1,
        param_beam_final,
    };

    IMGID img_loc[WFSTAT_NB_LOC];
    for (int k = 0; k < WFSTAT_NB_LOC; k++)
    {
        memset(&img_loc[k], 0, sizeof(IMGID));
        img_loc[k].ID = -1;
        img_loc[k]    = stream_connect(loc_snames[k]);
        if (img_loc[k].ID == -1)
        {
            for (int j = 0; j < k; j++)
            {
                imgid_free(&img_loc[j]);
            }
            fprintf(stderr, "cacaosim.wfstat: failed to connect beam '%s'\n", loc_snames[k]);
            return RETURN_FAILURE;
        }
    }

    IMGID img_trig = { 0 };
    img_trig.ID    = -1;
    if (param_trig_beam[0] != '\0' && strcmp(param_trig_beam, param_beam_raw) != 0 &&
        strcmp(param_trig_beam, param_beam_stg1) != 0 &&
        strcmp(param_trig_beam, param_beam_final) != 0)
    {
        img_trig = stream_connect(param_trig_beam);
    }

    uint32_t nwf  = img_loc[0].md->size[0];
    uint64_t npix = (uint64_t) nwf * nwf;

    uint8_t *mask    = (uint8_t *) calloc(npix, sizeof(uint8_t));
    double  *weights = (double *) calloc(npix, sizeof(double));
    float   *pup_amp = (float *) calloc(npix, sizeof(float));
    double  *sum_wf[WFSTAT_NB_LOC];
    double  *sum2_wf[WFSTAT_NB_LOC];
    float   *samp_wf[WFSTAT_NB_LOC];
    float   *wf_tmp  = (float *) calloc(npix, sizeof(float));
    double  *sortbuf = (double *) calloc(npix, sizeof(double));
    if (!mask || !weights || !pup_amp || !wf_tmp || !sortbuf)
    {
        return RETURN_FAILURE;
    }

    for (int k = 0; k < WFSTAT_NB_LOC; k++)
    {
        sum_wf[k]  = (double *) calloc(npix, sizeof(double));
        sum2_wf[k] = (double *) calloc(npix, sizeof(double));
        samp_wf[k] = (float *) calloc(WFSTAT_NB_SAMP * npix, sizeof(float));
    }

    const float *amp_src = (img_loc[0].md->naxis >= 3 && img_loc[0].md->size[2] >= 2)
                               ? (img_loc[0].im->array.F + npix)
                               : NULL;
    double   w_sum    = 0.0;
    uint64_t act_pix  = 0;
    for (uint64_t i = 0; i < npix; i++)
    {
        float a    = amp_src ? amp_src[i] : 1.0f;
        pup_amp[i] = a;
        if (a > 0.5f)
        {
            mask[i]    = 1;
            weights[i] = (double) a * (double) a;
            w_sum += weights[i];
            act_pix++;
        }
    }
    if (w_sum <= 0.0)
    {
        act_pix = npix;
        for (uint64_t i = 0; i < npix; i++)
        {
            mask[i]    = 1;
            weights[i] = 1.0 / (double) npix;
        }
    }
    else
    {
        for (uint64_t i = 0; i < npix; i++)
        {
            weights[i] /= w_sum;
        }
    }

    wfstat_extra_state_t extras[WFSTAT_NB_EXTRA];
    memset(extras, 0, sizeof(extras));
    for (int e = 0; e < WFSTAT_NB_EXTRA; e++)
    {
        extras[e].img.ID = -1;
        if (!param_snap_extra)
        {
            continue;
        }
        extras[e].img = stream_connect(s_extra_defs[e].stream_name);
        if (extras[e].img.ID != -1)
        {
            extras[e].nx     = extras[e].img.md->size[0];
            extras[e].ny     = extras[e].img.md->size[1];
            uint64_t epix    = (uint64_t) extras[e].nx * extras[e].ny;
            extras[e].cube   = (float *) calloc(WFSTAT_NB_SAMP * epix, sizeof(float));
            extras[e].active = (extras[e].cube != NULL);
        }
    }

    const double fracs[WFSTAT_NB_SAMP] = { 0.10, 0.30, 0.50, 0.70, 0.95 };
    uint64_t     samp_ticks[WFSTAT_NB_SAMP];
    uint64_t     tgt = (param_target_ticks > 10) ? param_target_ticks : 20000ULL;
    for (int s = 0; s < WFSTAT_NB_SAMP; s++)
    {
        uint64_t v = (uint64_t) llround(fracs[s] * (double) tgt);
        if (v < 1)
        {
            v = 1;
        }
        samp_ticks[s] = v;
    }

    double   sum_srms[WFSTAT_NB_LOC] = { 0.0, 0.0, 0.0 };
    double   sum_svar[WFSTAT_NB_LOC] = { 0.0, 0.0, 0.0 };
    double   samp_srms[WFSTAT_NB_SAMP][WFSTAT_NB_LOC];
    double   samp_sptv[WFSTAT_NB_SAMP][WFSTAT_NB_LOC];
    uint64_t samp_frame[WFSTAT_NB_SAMP];
    memset(samp_srms, 0, sizeof(samp_srms));
    memset(samp_sptv, 0, sizeof(samp_sptv));
    memset(samp_frame, 0, sizeof(samp_frame));

    uint64_t n_frames = 0;
    int      nsamp    = 0;

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        n_frames++;
        int is_samp = (nsamp < WFSTAT_NB_SAMP && n_frames >= samp_ticks[nsamp]);
        if (is_samp)
        {
            samp_frame[nsamp] = n_frames;
        }

        for (int k = 0; k < WFSTAT_NB_LOC; k++)
        {
            const float *opd = img_loc[k].im->array.F;
            double       pist = 0.0;
            for (uint64_t i = 0; i < npix; i++)
            {
                if (mask[i])
                {
                    pist += weights[i] * ((double) opd[i] * 1000.0);
                }
            }
            double svar = 0.0;
            double vmin = 1.0e30;
            double vmax = -1.0e30;
            for (uint64_t i = 0; i < npix; i++)
            {
                if (mask[i])
                {
                    double w = (double) opd[i] * 1000.0 - pist;
                    sum_wf[k][i] += w;
                    sum2_wf[k][i] += w * w;
                    svar += weights[i] * w * w;
                    wf_tmp[i] = (float) w;
                    if (w < vmin)
                    {
                        vmin = w;
                    }
                    if (w > vmax)
                    {
                        vmax = w;
                    }
                }
                else
                {
                    wf_tmp[i] = 0.0f;
                }
            }
            double srms = sqrt((svar > 0.0) ? svar : 0.0);
            sum_svar[k] += svar;
            sum_srms[k] += srms;

            if (is_samp)
            {
                memcpy(samp_wf[k] + (uint64_t) nsamp * npix, wf_tmp, npix * sizeof(float));
                samp_srms[nsamp][k] = srms;
                samp_sptv[nsamp][k] = (act_pix > 0) ? (vmax - vmin) : 0.0;
            }
        }

        if (is_samp)
        {
            for (int e = 0; e < WFSTAT_NB_EXTRA; e++)
            {
                wfstat_extra_state_t *es = &extras[e];
                if (!es->active)
                {
                    continue;
                }
                uint64_t     epix = (uint64_t) es->nx * es->ny;
                const float *src  = es->img.im->array.F;
                float       *dst  = es->cube + (uint64_t) nsamp * epix;
                if (s_extra_defs[e].kind == 0 && epix == npix)
                {
                    double pist = 0.0;
                    for (uint64_t i = 0; i < epix; i++)
                    {
                        if (mask[i])
                        {
                            pist += weights[i] * ((double) src[i] * 1000.0);
                        }
                    }
                    double sv = 0.0, mn = 1.0e30, mx = -1.0e30;
                    for (uint64_t i = 0; i < epix; i++)
                    {
                        if (mask[i])
                        {
                            double w = (double) src[i] * 1000.0 - pist;
                            dst[i]   = (float) w;
                            sv += weights[i] * w * w;
                            if (w < mn)
                            {
                                mn = w;
                            }
                            if (w > mx)
                            {
                                mx = w;
                            }
                        }
                        else
                        {
                            dst[i] = 0.0f;
                        }
                    }
                    es->m1[nsamp] = sqrt((sv > 0.0) ? sv : 0.0);
                    es->m2[nsamp] = mx - mn;
                }
                else if (s_extra_defs[e].kind == 1 || s_extra_defs[e].kind == 3)
                {
                    double s2 = 0.0, mn = 1.0e30, mx = -1.0e30;
                    for (uint64_t i = 0; i < epix; i++)
                    {
                        float v = src[i];
                        dst[i]  = v;
                        s2 += (double) v * (double) v;
                        if (v < mn)
                        {
                            mn = v;
                        }
                        if (v > mx)
                        {
                            mx = v;
                        }
                    }
                    es->m1[nsamp] = sqrt(s2 / (double) (epix ? epix : 1));
                    es->m2[nsamp] = mx - mn;
                }
                else if (s_extra_defs[e].kind == 2)
                {
                    double sm = 0.0, mx = -1.0e30;
                    for (uint64_t i = 0; i < epix; i++)
                    {
                        float v = src[i];
                        dst[i]  = v;
                        sm += (double) v;
                        if (v > mx)
                        {
                            mx = v;
                        }
                    }
                    es->m1[nsamp] = sm;
                    es->m2[nsamp] = mx;
                }
                else
                {
                    double sm = 0.0, mx = 0.0;
                    for (uint64_t i = 0; i < epix; i++)
                    {
                        float  v = src[i];
                        double s = ((double) v - 100.0 > 0.0) ? ((double) v - 100.0) : 0.0;
                        dst[i]   = v;
                        sm += s;
                        if (s > mx)
                        {
                            mx = s;
                        }
                    }
                    es->m1[nsamp] = mx;
                    es->m2[nsamp] = sm;
                }
            }
            nsamp++;
        }
    }
    MILK_PROCINFO_END(self)

    /* Flush accumulated maps, sample cubes, & wf_stats.json cleanly on loop exit */
    double denom  = (n_frames > 0) ? (double) n_frames : 1.0;
    double phys_s = (double) n_frames * (double) param_dt_phys;
    char   fpath[1280];

    snprintf(fpath, sizeof(fpath), "%s/wf_pupil_mask.fits", out_dir);
    cacaosim_mask_save_fits(fpath, pup_amp, nwf, nwf, 1, NULL);

    float *mean_f = (float *) calloc(npix, sizeof(float));
    float *std_f  = (float *) calloc(npix, sizeof(float));

    snprintf(fpath, sizeof(fpath), "%s/wf_stats.json", out_dir);
    FILE *jfp = fopen(fpath, "w");
    if (jfp)
    {
        fprintf(jfp, "{\n");
        fprintf(jfp, "  \"n_frames\": %" PRIu64 ",\n", n_frames);
        fprintf(jfp, "  \"phys_time_recorded_s\": %.6f,\n", phys_s);
        fprintf(jfp, "  \"active_pupil_pixels\": %" PRIu64 ",\n", act_pix);
        fprintf(jfp, "  \"units\": \"nm\",\n");
        fprintf(jfp, "  \"locations\": {\n");
    }

    for (int k = 0; k < WFSTAT_NB_LOC; k++)
    {
        double   w_mean2 = 0.0, w_std2 = 0.0;
        double   m_min = 1e30, m_max = -1e30;
        uint64_t nsort = 0;
        for (uint64_t i = 0; i < npix; i++)
        {
            if (mask[i])
            {
                double mn = sum_wf[k][i] / denom;
                double vr = sum2_wf[k][i] / denom - mn * mn;
                double sd = sqrt((vr > 0.0) ? vr : 0.0);
                mean_f[i] = (float) mn;
                std_f[i]  = (float) sd;
                w_mean2 += weights[i] * mn * mn;
                w_std2 += weights[i] * sd * sd;
                if (mn < m_min)
                {
                    m_min = mn;
                }
                if (mn > m_max)
                {
                    m_max = mn;
                }
                sortbuf[nsort++] = sd;
            }
            else
            {
                mean_f[i] = 0.0f;
                std_f[i]  = 0.0f;
            }
        }
        snprintf(fpath, sizeof(fpath), "%s/wf_mean_%s.fits", out_dir, loc_keys[k]);
        cacaosim_mask_save_fits(fpath, mean_f, nwf, nwf, 1, NULL);
        snprintf(fpath, sizeof(fpath), "%s/wf_std_%s.fits", out_dir, loc_keys[k]);
        cacaosim_mask_save_fits(fpath, std_f, nwf, nwf, 1, NULL);
        if (nsamp > 0)
        {
            snprintf(fpath, sizeof(fpath), "%s/wf_samples_%s.fits", out_dir, loc_keys[k]);
            cacaosim_mask_save_fits(fpath, samp_wf[k], nwf, nwf, (uint32_t) nsamp, NULL);
        }

        double sd_med = 0.0, sd_p95 = 0.0, sd_max = 0.0;
        if (nsort > 0)
        {
            qsort(sortbuf, nsort, sizeof(double), cmp_dbl);
            sd_med = sortbuf[nsort / 2];
            sd_p95 = sortbuf[(nsort * 95) / 100];
            sd_max = sortbuf[nsort - 1];
        }
        double quad_rms = sqrt(sum_svar[k] / denom);
        double mean_rms = sum_srms[k] / denom;
        double mm_rms   = sqrt(w_mean2);
        double sm_rms   = sqrt(w_std2);
        double pv_vis   = 2.0 * M_PI * (quad_rms * 1.0e-3) / 0.75;
        double pv_nir   = 2.0 * M_PI * (quad_rms * 1.0e-3) / 1.65;
        double sr_vis   = exp(-(pv_vis * pv_vis));
        double sr_nir   = exp(-(pv_nir * pv_nir));

        if (jfp)
        {
            fprintf(jfp, "    \"%s\": {\n", loc_keys[k]);
            fprintf(jfp, "      \"label\": \"%s\",\n", loc_desc[k]);
            fprintf(jfp, "      \"mean_spatial_rms_nm\": %.4f,\n", mean_rms);
            fprintf(jfp, "      \"quad_total_rms_nm\": %.4f,\n", quad_rms);
            fprintf(jfp, "      \"mean_map_rms_nm\": %.4f,\n", mm_rms);
            fprintf(jfp, "      \"mean_map_ptv_nm\": %.4f,\n",
                    (act_pix > 0) ? (m_max - m_min) : 0.0);
            fprintf(jfp, "      \"std_map_rms_nm\": %.4f,\n", sm_rms);
            fprintf(jfp, "      \"std_map_median_nm\": %.4f,\n", sd_med);
            fprintf(jfp, "      \"std_map_p95_nm\": %.4f,\n", sd_p95);
            fprintf(jfp, "      \"std_map_max_nm\": %.4f,\n", sd_max);
            fprintf(jfp, "      \"marechal_sr_vis_750nm\": %.6f,\n", sr_vis);
            fprintf(jfp, "      \"marechal_sr_nir_1650nm\": %.6f\n", sr_nir);
            fprintf(jfp, "    }%s\n", (k + 1 < WFSTAT_NB_LOC) ? "," : "");
        }
    }

    for (int e = 0; e < WFSTAT_NB_EXTRA; e++)
    {
        wfstat_extra_state_t *es = &extras[e];
        if (es->active && nsamp > 0)
        {
            snprintf(fpath, sizeof(fpath), "%s/%s_samples.fits", out_dir, s_extra_defs[e].key);
            cacaosim_mask_save_fits(fpath, es->cube, es->nx, es->ny, (uint32_t) nsamp, NULL);
        }
    }

    if (jfp)
    {
        fprintf(jfp, "  },\n");
        fprintf(jfp, "  \"samples\": [\n");
        for (int s = 0; s < nsamp; s++)
        {
            fprintf(jfp, "    {\n");
            fprintf(jfp, "      \"sample_index\": %d,\n", s);
            fprintf(jfp, "      \"frame\": %" PRIu64 ",\n", samp_frame[s]);
            fprintf(jfp, "      \"t_phys_s\": %.6f,\n",
                    (double) samp_frame[s] * (double) param_dt_phys);
            for (int k = 0; k < WFSTAT_NB_LOC; k++)
            {
                fprintf(jfp, "      \"%s\": {\"rms_nm\": %.3f, \"ptv_nm\": %.3f},\n", loc_keys[k],
                        samp_srms[s][k], samp_sptv[s][k]);
            }
            fprintf(jfp, "      \"extra\": {\n");
            int first_ex = 1;
            for (int e = 0; e < WFSTAT_NB_EXTRA; e++)
            {
                if (!extras[e].active)
                {
                    continue;
                }
                const char *k1 = "rms", *k2 = "ptv";
                if (s_extra_defs[e].kind == 0)
                {
                    k1 = "rms_nm";
                    k2 = "ptv_nm";
                }
                else if (s_extra_defs[e].kind == 1)
                {
                    k1 = "rms_um";
                    k2 = "ptv_um";
                }
                else if (s_extra_defs[e].kind == 2)
                {
                    k1 = "sum_adu";
                    k2 = "max_adu";
                }
                else if (s_extra_defs[e].kind == 4)
                {
                    k1 = "peak_adu";
                    k2 = "sum_adu";
                }
                fprintf(jfp, "%s        \"%s\": {\"%s\": %.4f, \"%s\": %.4f}",
                        first_ex ? "" : ",\n", s_extra_defs[e].key, k1, extras[e].m1[s], k2,
                        extras[e].m2[s]);
                first_ex = 0;
            }
            fprintf(jfp, "\n      }\n");
            fprintf(jfp, "    }%s\n", (s + 1 < nsamp) ? "," : "");
        }
        fprintf(jfp, "  ]\n}\n");
        fclose(jfp);
    }

    for (int e = 0; e < WFSTAT_NB_EXTRA; e++)
    {
        free(extras[e].cube);
        if (extras[e].img.ID != -1)
        {
            imgid_free(&extras[e].img);
        }
    }
    for (int k = 0; k < WFSTAT_NB_LOC; k++)
    {
        free(sum_wf[k]);
        free(sum2_wf[k]);
        free(samp_wf[k]);
        imgid_free(&img_loc[k]);
    }
    if (img_trig.ID != -1)
    {
        imgid_free(&img_trig);
    }
    free(mean_f);
    free(std_f);
    free(sortbuf);
    free(wf_tmp);
    free(pup_amp);
    free(weights);
    free(mask);
    return RETURN_SUCCESS;
}

const MILK_RUNNABLE runnable_cacaosim_wfstat = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_wfstat",
              .cmdkey      = "wfstat",
              .description = "Real-time pupil wavefront stats & synchronized FITS cube recorder" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_wfstat),
    .compute = wfstat_compute,
};
EXPORT_FPSEXEC(milk, "cacaosim-wfstat", runnable_cacaosim_wfstat);
