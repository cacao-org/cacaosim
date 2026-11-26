// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_perturb.c
 * @brief   Entrance-pupil telescope vibration and thermally induced petaling (LWE) engine
 *
 * Implements automatic spider-island pupil segmentation, pupil-orthonormalized Zernike/petal
 * basis synthesis, exact stationary AR(2) damped vibration resonators, and smooth Matern-3/2
 * thermal petaling evolution in physical simulation time.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "common/cacaosim_perturb.h"

#define PERTURB_TWO_PI 6.28318530717958647692

/* Realistic 8m-class (Subaru AO3k/SCExAO) telescope & bench mechanical vibration modes */
typedef struct
{
    double f0_hz;
    double zeta;
    double share;
    double z_wt[CACAOSIM_PERTURB_NVIB_ZERN]; /* Unit-norm coupling into [Z2, Z3, Z4, Z5, Z6] */
} perturb_vib_mode_t;

static const perturb_vib_mode_t k_vib_modes[CACAOSIM_PERTURB_NVIB_LINES] = {
    /* 14.5 Hz: telescope truss / M2 wind-shake (broader Q ~ 14, elliptical tip/tilt) */
    { 14.5, 0.035, 0.22, { 0.85, 0.5268, 0.0, 0.0, 0.0 } },
    /* 31.0 Hz: Nasmyth platform / M1 cell support (Q ~ 33, near-orthogonal tip/tilt) */
    { 31.0, 0.015, 0.16, { -0.45, 0.8930, 0.0, 0.0, 0.0 } },
    /* 49.5 Hz: cryocooler / bench fundamental (Q ~ 62, tip/tilt + small defocus) */
    { 49.5, 0.008, 0.38, { 0.78, 0.5492, 0.30, 0.0, 0.0 } },
    /* 99.0 Hz: cryocooler 2nd harmonic / fan (Q ~ 62, tip/tilt + astigmatism) */
    { 99.0, 0.008, 0.15, { 0.60, -0.65, 0.0, 0.35, 0.3082 } },
    /* 138.0 Hz: high-frequency optical-bench / M2 structural mode (Q ~ 50, tip/tilt) */
    { 138.0, 0.010, 0.09, { 0.70, 0.7141, 0.0, 0.0, 0.0 } },
};

static inline uint64_t perturb_splitmix64(uint64_t *st)
{
    uint64_t z = (*st += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline void perturb_gauss_pair(uint64_t *st, double *g0, double *g1)
{
    double u1 = ((double) (perturb_splitmix64(st) >> 11) + 1.0) * (1.0 / 9007199254740992.0);
    double u2 = (double) (perturb_splitmix64(st) >> 11) * (1.0 / 9007199254740992.0);
    double r  = sqrt(-2.0 * log(u1));
    double th = PERTURB_TWO_PI * u2;
    *g0       = r * cos(th);
    *g1       = r * sin(th);
}

static void perturb_matern32_step(
    double   *u,
    double   *v,
    double    dt_s,
    double    tau_s,
    uint64_t *rng)
{
    if (!(dt_s > 0.0) || !(tau_s > 0.0))
    {
        return;
    }
    double lam = dt_s / tau_s;
    double em  = exp(-lam);
    double e2  = em * em;
    double p00 = em * (1.0 + lam);
    double p01 = em * lam;
    double p10 = -em * lam;
    double p11 = em * (1.0 - lam);

    double q00, q01, q11;
    if (lam < 1.0e-3)
    {
        double l2 = lam * lam;
        double l3 = l2 * lam;
        q00       = (4.0 / 3.0) * l3 - 2.0 * l3 * lam + 1.6 * l3 * l2;
        q01       = 2.0 * l2 * e2;
        q11       = 4.0 * lam - 4.0 * l2 + (8.0 / 3.0) * l3;
    }
    else
    {
        q00 = 1.0 - e2 * (1.0 + 2.0 * lam + 2.0 * lam * lam);
        q01 = 2.0 * lam * lam * e2;
        q11 = 1.0 - e2 * (1.0 - 2.0 * lam + 2.0 * lam * lam);
    }
    if (q00 < 0.0)
    {
        q00 = 0.0;
    }
    if (q11 < 0.0)
    {
        q11 = 0.0;
    }

    double l00 = sqrt(q00);
    double l10 = (l00 > 1.0e-15) ? (q01 / l00) : 0.0;
    double rem = q11 - l10 * l10;
    double l11 = (rem > 0.0) ? sqrt(rem) : 0.0;

    double g0 = 0.0, g1 = 0.0;
    perturb_gauss_pair(rng, &g0, &g1);
    double un = p00 * (*u) + p01 * (*v) + l00 * g0;
    double vn = p10 * (*u) + p11 * (*v) + l10 * g0 + l11 * g1;
    *u        = un;
    *v        = vn;
}

/**
 * segment_pupil_islands() - 4-connected BFS segmentation of spider-bounded pupil petals
 *
 * Labels each connected component of `pup_amp > 0.5` with > 1% of pupil pixels, sorts islands
 * counter-clockwise by centroid angle starting from -45 deg (so Subaru's 4 V-spider petals map
 * deterministically to 0=East, 1=North, 2=West, 3=South), and falls back to 4 quadrants on
 * unspidered pupils.
 */
static int segment_pupil_islands(
    uint32_t     nwf,
    const float *pup_amp,
    int32_t     *labels)
{
    uint64_t npix = (uint64_t) nwf * nwf;
    int32_t *raw  = (int32_t *) malloc(npix * sizeof(int32_t));
    uint32_t *q   = (uint32_t *) malloc(npix * sizeof(uint32_t));
    if (!raw || !q)
    {
        free(raw);
        free(q);
        return 0;
    }
    uint64_t act_tot = 0;
    for (uint64_t i = 0; i < npix; i++)
    {
        raw[i]    = -1;
        labels[i] = -1;
        if (pup_amp == NULL || pup_amp[i] > 0.5f)
        {
            act_tot++;
        }
    }

    int32_t  comp_id = 0;
    uint64_t comp_cnt[64];
    double   comp_sx[64], comp_sy[64];
    memset(comp_cnt, 0, sizeof(comp_cnt));
    memset(comp_sx, 0, sizeof(comp_sx));
    memset(comp_sy, 0, sizeof(comp_sy));

    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            uint64_t idx = (uint64_t) y * nwf + x;
            if (raw[idx] >= 0 || (pup_amp != NULL && pup_amp[idx] <= 0.5f))
            {
                continue;
            }
            int32_t  cid  = (comp_id < 64) ? comp_id : 63;
            uint64_t head = 0, tail = 0;
            raw[idx]  = cid;
            q[tail++] = (uint32_t) idx;
            while (head < tail)
            {
                uint32_t cur = q[head++];
                uint32_t cx  = cur % nwf;
                uint32_t cy  = cur / nwf;
                comp_cnt[cid]++;
                comp_sx[cid] += (double) cx;
                comp_sy[cid] += (double) cy;

                const int dx[4] = { -1, 1, 0, 0 };
                const int dy[4] = { 0, 0, -1, 1 };
                for (int d = 0; d < 4; d++)
                {
                    int nx = (int) cx + dx[d];
                    int ny = (int) cy + dy[d];
                    if (nx < 0 || ny < 0 || nx >= (int) nwf || ny >= (int) nwf)
                    {
                        continue;
                    }
                    uint64_t nidx = (uint64_t) ny * nwf + (uint64_t) nx;
                    if (raw[nidx] < 0 && (pup_amp == NULL || pup_amp[nidx] > 0.5f))
                    {
                        raw[nidx] = cid;
                        q[tail++] = (uint32_t) nidx;
                    }
                }
            }
            if (comp_id < 64)
            {
                comp_id++;
            }
        }
    }

    int32_t kept[CACAOSIM_PERTURB_MAX_PETALS];
    double  ang[CACAOSIM_PERTURB_MAX_PETALS];
    int     n_kept  = 0;
    double  mid     = 0.5 * ((double) nwf - 1.0);
    uint64_t min_px = (act_tot > 100) ? (act_tot / 100) : 1;

    for (int c = 0; c < comp_id && n_kept < CACAOSIM_PERTURB_MAX_PETALS; c++)
    {
        if (comp_cnt[c] >= min_px)
        {
            double mx = comp_sx[c] / (double) comp_cnt[c] - mid;
            double my = comp_sy[c] / (double) comp_cnt[c] - mid;
            double a  = atan2(my, mx);
            if (a < -0.7853981633974483)
            {
                a += PERTURB_TWO_PI;
            }
            kept[n_kept] = c;
            ang[n_kept]  = a;
            n_kept++;
        }
    }

    if (n_kept >= 2)
    {
        /* Insertion sort by angle for deterministic petal ordering (E, N, W, S on Subaru) */
        for (int i = 1; i < n_kept; i++)
        {
            int32_t kc = kept[i];
            double  ka = ang[i];
            int     j  = i - 1;
            while (j >= 0 && ang[j] > ka)
            {
                kept[j + 1] = kept[j];
                ang[j + 1]  = ang[j];
                j--;
            }
            kept[j + 1] = kc;
            ang[j + 1]  = ka;
        }
        int32_t remap[64];
        for (int c = 0; c < 64; c++)
        {
            remap[c] = -1;
        }
        for (int p = 0; p < n_kept; p++)
        {
            remap[kept[p]] = p;
        }
        for (uint64_t i = 0; i < npix; i++)
        {
            if (raw[i] >= 0 && raw[i] < 64)
            {
                labels[i] = remap[raw[i]];
            }
        }
        free(raw);
        free(q);
        return n_kept;
    }

    /* Fallback for unspidered circular pupil: split into 4 geometric quadrants */
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            uint64_t idx = (uint64_t) y * nwf + x;
            if (pup_amp != NULL && pup_amp[idx] <= 0.5f)
            {
                continue;
            }
            double rx = (double) x - mid;
            double ry = (double) y - mid;
            if (rx >= 0.0 && ry >= 0.0)
            {
                labels[idx] = 0;
            }
            else if (rx < 0.0 && ry >= 0.0)
            {
                labels[idx] = 1;
            }
            else if (rx < 0.0 && ry < 0.0)
            {
                labels[idx] = 2;
            }
            else
            {
                labels[idx] = 3;
            }
        }
    }
    free(raw);
    free(q);
    return 4;
}

static void build_zernike_basis(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float       *basis)
{
    uint64_t npix = (uint64_t) nwf * nwf;
    double   mid  = 0.5 * ((double) nwf - 1.0);
    double   rad  = 0.5 * (double) pup_diam * (double) nwf;
    if (rad < 1.0)
    {
        rad = 0.5 * (double) nwf;
    }
    double inv_r = 1.0 / rad;

    for (uint32_t y = 0; y < nwf; y++)
    {
        double yn = ((double) y - mid) * inv_r;
        for (uint32_t x = 0; x < nwf; x++)
        {
            uint64_t idx = (uint64_t) y * nwf + x;
            if (pup_amp != NULL && pup_amp[idx] <= 0.5f)
            {
                continue;
            }
            double xn                   = ((double) x - mid) * inv_r;
            double r2                   = xn * xn + yn * yn;
            basis[0 * npix + idx]       = (float) xn;                  /* Z2: Tip */
            basis[1 * npix + idx]       = (float) yn;                  /* Z3: Tilt */
            basis[2 * npix + idx]       = (float) (2.0 * r2 - 1.0);    /* Z4: Defocus */
            basis[3 * npix + idx]       = (float) (2.0 * xn * yn);     /* Z5: Oblique astig */
            basis[4 * npix + idx]       = (float) (xn * xn - yn * yn); /* Z6: Vertical astig */
        }
    }

    /* Gram-Schmidt orthonormalization over active pupil (piston-removed, unit RMS) */
    double wsum = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (pup_amp == NULL || pup_amp[i] > 0.5f)
        {
            wsum += 1.0;
        }
    }
    if (wsum <= 0.0)
    {
        return;
    }

    for (int m = 0; m < CACAOSIM_PERTURB_NVIB_ZERN; m++)
    {
        float *bm   = basis + (uint64_t) m * npix;
        double mean = 0.0;
        for (uint64_t i = 0; i < npix; i++)
        {
            if (pup_amp == NULL || pup_amp[i] > 0.5f)
            {
                mean += (double) bm[i];
            }
        }
        mean /= wsum;
        for (uint64_t i = 0; i < npix; i++)
        {
            if (pup_amp == NULL || pup_amp[i] > 0.5f)
            {
                bm[i] -= (float) mean;
            }
        }
        for (int p = 0; p < m; p++)
        {
            const float *bp  = basis + (uint64_t) p * npix;
            double       dot = 0.0;
            for (uint64_t i = 0; i < npix; i++)
            {
                if (pup_amp == NULL || pup_amp[i] > 0.5f)
                {
                    dot += (double) bm[i] * (double) bp[i];
                }
            }
            dot /= wsum;
            for (uint64_t i = 0; i < npix; i++)
            {
                if (pup_amp == NULL || pup_amp[i] > 0.5f)
                {
                    bm[i] -= (float) (dot * (double) bp[i]);
                }
            }
        }
        double s2 = 0.0;
        for (uint64_t i = 0; i < npix; i++)
        {
            if (pup_amp == NULL || pup_amp[i] > 0.5f)
            {
                s2 += (double) bm[i] * (double) bm[i];
            }
        }
        double rms = sqrt(s2 / wsum);
        if (rms > 1.0e-12)
        {
            float inv = (float) (1.0 / rms);
            for (uint64_t i = 0; i < npix; i++)
            {
                bm[i] *= inv;
            }
        }
    }
}

static void build_petal_basis(
    uint32_t       nwf,
    const int32_t *labels,
    int            n_petals,
    float         *basis_petals,
    double        *petal_frac)
{
    uint64_t npix = (uint64_t) nwf * nwf;
    uint64_t cnt[CACAOSIM_PERTURB_MAX_PETALS];
    double   cx[CACAOSIM_PERTURB_MAX_PETALS], cy[CACAOSIM_PERTURB_MAX_PETALS];
    memset(cnt, 0, sizeof(cnt));
    memset(cx, 0, sizeof(cx));
    memset(cy, 0, sizeof(cy));

    uint64_t tot = 0;
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            int32_t p = labels[(uint64_t) y * nwf + x];
            if (p >= 0 && p < n_petals)
            {
                cnt[p]++;
                cx[p] += (double) x;
                cy[p] += (double) y;
                tot++;
            }
        }
    }

    for (int p = 0; p < n_petals; p++)
    {
        petal_frac[p] = (tot > 0) ? ((double) cnt[p] / (double) tot) : (1.0 / (double) n_petals);
        if (cnt[p] > 0)
        {
            cx[p] /= (double) cnt[p];
            cy[p] /= (double) cnt[p];
        }
    }

    double sx2[CACAOSIM_PERTURB_MAX_PETALS], sy2[CACAOSIM_PERTURB_MAX_PETALS];
    memset(sx2, 0, sizeof(sx2));
    memset(sy2, 0, sizeof(sy2));
    for (uint32_t y = 0; y < nwf; y++)
    {
        for (uint32_t x = 0; x < nwf; x++)
        {
            int32_t p = labels[(uint64_t) y * nwf + x];
            if (p >= 0 && p < n_petals)
            {
                double dx = (double) x - cx[p];
                double dy = (double) y - cy[p];
                sx2[p] += dx * dx;
                sy2[p] += dy * dy;
            }
        }
    }

    for (int p = 0; p < n_petals; p++)
    {
        double rx = (cnt[p] > 0 && sx2[p] > 0.0) ? sqrt(sx2[p] / (double) cnt[p]) : 1.0;
        double ry = (cnt[p] > 0 && sy2[p] > 0.0) ? sqrt(sy2[p] / (double) cnt[p]) : 1.0;
        float *bp = basis_petals + (uint64_t) (3 * p + 0) * npix;
        float *bx = basis_petals + (uint64_t) (3 * p + 1) * npix;
        float *by = basis_petals + (uint64_t) (3 * p + 2) * npix;
        for (uint32_t y = 0; y < nwf; y++)
        {
            for (uint32_t x = 0; x < nwf; x++)
            {
                uint64_t idx = (uint64_t) y * nwf + x;
                if (labels[idx] == p)
                {
                    bp[idx] = 1.0f;
                    bx[idx] = (float) (((double) x - cx[p]) / rx);
                    by[idx] = (float) (((double) y - cy[p]) / ry);
                }
            }
        }
    }
}

void cacaosim_perturb_cfg_defaults(cacaosim_perturb_cfg_t *cfg)
{
    if (cfg == NULL)
    {
        return;
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->vib_enable   = 0;
    cfg->vib_rms_nm   = 45.0f;
    cfg->vib_seed     = 101ULL;
    cfg->petal_enable = 0;
    cfg->petal_rms_nm = 1000.0f;
    cfg->petal_tau_s  = 3.0f;
    cfg->petal_seed   = 202ULL;
}

int cacaosim_perturb_init(
    cacaosim_perturb_ctx_t       *ctx,
    uint32_t                      nwf,
    float                         pup_diam,
    const float                  *pup_amp,
    const cacaosim_perturb_cfg_t *cfg,
    int                           use_cuda,
    void                         *cuda_stream)
{
    if (ctx == NULL || nwf < 8u)
    {
        return -1;
    }
    memset(ctx, 0, sizeof(*ctx));
    ctx->nwf      = nwf;
    ctx->npix     = (uint64_t) nwf * nwf;
    ctx->use_cuda = use_cuda ? 1 : 0;

    int32_t *labels = (int32_t *) malloc(ctx->npix * sizeof(int32_t));
    if (labels == NULL)
    {
        return -1;
    }
    ctx->n_petals = segment_pupil_islands(nwf, pup_amp, labels);
    if (ctx->n_petals < 2)
    {
        ctx->n_petals = 4;
    }
    if (ctx->n_petals > CACAOSIM_PERTURB_MAX_PETALS)
    {
        ctx->n_petals = CACAOSIM_PERTURB_MAX_PETALS;
    }
    ctx->n_modes      = CACAOSIM_PERTURB_NVIB_ZERN + 3 * ctx->n_petals;
    ctx->tlm.n_petals = ctx->n_petals;

    size_t tot   = (size_t) ctx->n_modes * (size_t) ctx->npix;
    ctx->h_basis = (float *) cacaosim_aligned_calloc(tot, sizeof(float));
    if (ctx->h_basis == NULL)
    {
        free(labels);
        return -1;
    }
    build_zernike_basis(nwf, pup_diam, pup_amp, ctx->h_basis);
    build_petal_basis(nwf, labels, ctx->n_petals,
                      ctx->h_basis + (uint64_t) CACAOSIM_PERTURB_NVIB_ZERN * ctx->npix,
                      ctx->petal_frac);
    free(labels);

    if (ctx->use_cuda)
    {
        ctx->d_basis = cacaosim_cuda_alloc_f32((uint64_t) tot);
        if (ctx->d_basis == NULL)
        {
            cacaosim_perturb_free(ctx);
            return -1;
        }
        cacaosim_cuda_h2d_f32(ctx->d_basis, ctx->h_basis, (uint64_t) tot, cuda_stream);
    }

    uint64_t vseed = (cfg && cfg->vib_seed != 0ULL) ? cfg->vib_seed : 101ULL;
    uint64_t pseed = (cfg && cfg->petal_seed != 0ULL) ? cfg->petal_seed : 202ULL;
    ctx->vib_rng   = vseed ^ 0xa0761d6478bd642fULL;
    ctx->petal_rng = pseed ^ 0xe7037ed1a0b428dbULL;
    (void) perturb_splitmix64(&ctx->vib_rng);
    (void) perturb_splitmix64(&ctx->petal_rng);

    /* Prime stationary unit-variance thermal petal states so LWE is active from tick 1 */
    for (int i = 0; i < 3 * ctx->n_petals; i++)
    {
        perturb_gauss_pair(&ctx->petal_rng, &ctx->petal_u[i], &ctx->petal_v[i]);
    }
    ctx->petal_primed = 1;
    ctx->initialized  = 1;
    return 0;
}

static void step_vibrations(
    cacaosim_perturb_ctx_t *ctx,
    double                  dt_s,
    double                  rms_um,
    float                  *out_z_um)
{
    double z_acc[CACAOSIM_PERTURB_NVIB_ZERN] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (int j = 0; j < CACAOSIM_PERTURB_NVIB_LINES; j++)
    {
        double w0   = PERTURB_TWO_PI * k_vib_modes[j].f0_hz;
        double zeta = k_vib_modes[j].zeta;
        double wd   = w0 * sqrt(1.0 - zeta * zeta);
        double r    = exp(-zeta * w0 * dt_s);
        double a1   = 2.0 * r * cos(wd * dt_s);
        double a2   = -(r * r);
        double num  = (1.0 + a2) * ((1.0 - a2) * (1.0 - a2) - a1 * a1);
        double den  = 1.0 - a2;
        double b    = (num > 0.0 && den > 0.0) ? sqrt(num / den) : 0.0;

        if (!ctx->vib_primed)
        {
            double g0 = 0.0, g1 = 0.0;
            perturb_gauss_pair(&ctx->vib_rng, &g0, &g1);
            double rho1 = (den > 1.0e-12) ? (a1 / den) : 0.0;
            if (rho1 > 0.999999)
            {
                rho1 = 0.999999;
            }
            if (rho1 < -0.999999)
            {
                rho1 = -0.999999;
            }
            ctx->vib_x2[j] = g0;
            ctx->vib_x1[j] = rho1 * g0 + sqrt(1.0 - rho1 * rho1) * g1;
        }
        else
        {
            double g0 = 0.0, g1 = 0.0;
            perturb_gauss_pair(&ctx->vib_rng, &g0, &g1);
            double xn      = a1 * ctx->vib_x1[j] + a2 * ctx->vib_x2[j] + b * g0;
            ctx->vib_x2[j] = ctx->vib_x1[j];
            ctx->vib_x1[j] = xn;
        }

        double amp = rms_um * sqrt(k_vib_modes[j].share) * ctx->vib_x1[j];
        for (int m = 0; m < CACAOSIM_PERTURB_NVIB_ZERN; m++)
        {
            z_acc[m] += amp * k_vib_modes[j].z_wt[m];
        }
    }
    ctx->vib_primed = 1;

    double s2 = 0.0;
    for (int m = 0; m < CACAOSIM_PERTURB_NVIB_ZERN; m++)
    {
        out_z_um[m] = (float) z_acc[m];
        s2 += z_acc[m] * z_acc[m];
    }
    ctx->tlm.vib_rms_nm  = (float) (sqrt(s2) * 1000.0);
    ctx->tlm.vib_tip_nm  = (float) (z_acc[0] * 1000.0);
    ctx->tlm.vib_tilt_nm = (float) (z_acc[1] * 1000.0);
    ctx->tlm.vib_foc_nm  = (float) (z_acc[2] * 1000.0);
}

static void step_petaling(
    cacaosim_perturb_ctx_t *ctx,
    double                  dt_s,
    double                  rms_um,
    double                  tau_s,
    float                  *out_p_um)
{
    int np = ctx->n_petals;
    if (tau_s < 0.05)
    {
        tau_s = 0.05;
    }

    /* 75% variance in differential island pistons, 25% split equally in local tip & tilt */
    double np_d    = (double) np;
    double sig_pis = rms_um * sqrt(0.75 * np_d / ((np > 1) ? (np_d - 1.0) : 1.0));
    double sig_tt  = rms_um * sqrt(0.125);

    double pis_raw[CACAOSIM_PERTURB_MAX_PETALS];
    double mean_p = 0.0;
    for (int p = 0; p < np; p++)
    {
        int i0 = 3 * p + 0;
        int i1 = 3 * p + 1;
        int i2 = 3 * p + 2;
        perturb_matern32_step(&ctx->petal_u[i0], &ctx->petal_v[i0], dt_s, tau_s, &ctx->petal_rng);
        perturb_matern32_step(&ctx->petal_u[i1], &ctx->petal_v[i1], dt_s, tau_s, &ctx->petal_rng);
        perturb_matern32_step(&ctx->petal_u[i2], &ctx->petal_v[i2], dt_s, tau_s, &ctx->petal_rng);
        pis_raw[p] = sig_pis * ctx->petal_u[i0];
        mean_p += ctx->petal_frac[p] * pis_raw[p];
    }

    double var_tot = 0.0;
    double p_min   = 1.0e30;
    double p_max   = -1.0e30;
    for (int p = 0; p < np; p++)
    {
        double dp = pis_raw[p] - mean_p;
        double tx = sig_tt * ctx->petal_u[3 * p + 1];
        double ty = sig_tt * ctx->petal_u[3 * p + 2];

        out_p_um[3 * p + 0] = (float) dp;
        out_p_um[3 * p + 1] = (float) tx;
        out_p_um[3 * p + 2] = (float) ty;

        var_tot += ctx->petal_frac[p] * (dp * dp + tx * tx + ty * ty);
        if (dp < p_min)
        {
            p_min = dp;
        }
        if (dp > p_max)
        {
            p_max = dp;
        }
        if (p < 4)
        {
            ctx->tlm.petal_p_nm[p] = (float) (dp * 1000.0);
        }
    }
    ctx->tlm.petal_rms_nm = (float) (sqrt((var_tot > 0.0) ? var_tot : 0.0) * 1000.0);
    ctx->tlm.petal_ptv_nm = (np > 0) ? (float) ((p_max - p_min) * 1000.0) : 0.0f;
}

void cacaosim_perturb_apply_current(
    const cacaosim_perturb_ctx_t *ctx,
    float                        *h_opd,
    float                        *d_opd,
    void                         *cuda_stream)
{
    if (ctx == NULL || !ctx->initialized)
    {
        return;
    }
    if (ctx->use_cuda && d_opd != NULL && ctx->d_basis != NULL)
    {
        cacaosim_cuda_beam_add_modes(d_opd, ctx->d_basis, ctx->coeff_um, (uint32_t) ctx->n_modes,
                                     ctx->npix, cuda_stream);
    }
    else if (h_opd != NULL && ctx->h_basis != NULL)
    {
        uint64_t npix = ctx->npix;
        for (int m = 0; m < ctx->n_modes; m++)
        {
            float c = ctx->coeff_um[m];
            if (fabsf(c) <= 1.0e-12f)
            {
                continue;
            }
            const float *bm = ctx->h_basis + (uint64_t) m * npix;
            for (uint64_t i = 0; i < npix; i++)
            {
                h_opd[i] += c * bm[i];
            }
        }
    }
}

void cacaosim_perturb_step_apply(
    cacaosim_perturb_ctx_t       *ctx,
    const cacaosim_perturb_cfg_t *cfg,
    double                        dt_s,
    float                         scale,
    float                        *h_opd,
    float                        *d_opd,
    void                         *cuda_stream)
{
    if (ctx == NULL || !ctx->initialized || cfg == NULL)
    {
        return;
    }

    memset(ctx->coeff_um, 0, sizeof(ctx->coeff_um));
    ctx->tlm.vib_rms_nm   = 0.0f;
    ctx->tlm.vib_tip_nm   = 0.0f;
    ctx->tlm.vib_tilt_nm  = 0.0f;
    ctx->tlm.vib_foc_nm   = 0.0f;
    ctx->tlm.petal_rms_nm = 0.0f;
    ctx->tlm.petal_ptv_nm = 0.0f;
    memset(ctx->tlm.petal_p_nm, 0, sizeof(ctx->tlm.petal_p_nm));

    float asc = fabsf(scale);
    if (asc <= 0.0f || (!cfg->vib_enable && !cfg->petal_enable))
    {
        return;
    }

    int any = 0;
    if (cfg->vib_enable && cfg->vib_rms_nm > 0.0f)
    {
        step_vibrations(ctx, dt_s, (double) cfg->vib_rms_nm * 1.0e-3 * (double) scale,
                        ctx->coeff_um);
        any = 1;
    }
    if (cfg->petal_enable && cfg->petal_rms_nm > 0.0f)
    {
        step_petaling(ctx, dt_s, (double) cfg->petal_rms_nm * 1.0e-3 * (double) scale,
                      (double) cfg->petal_tau_s, ctx->coeff_um + CACAOSIM_PERTURB_NVIB_ZERN);
        any = 1;
    }
    if (!any)
    {
        return;
    }

    cacaosim_perturb_apply_current(ctx, h_opd, d_opd, cuda_stream);
}

void cacaosim_perturb_free(cacaosim_perturb_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    free(ctx->h_basis);
    cacaosim_cuda_free_f32(ctx->d_basis);
    memset(ctx, 0, sizeof(*ctx));
}
