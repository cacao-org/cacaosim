// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_engine.c
 * @brief   Coronagraphic pupil/focal custom masks, polychromatic MFT/FFT, Lyot, & LOWFS engine
 */

#define _GNU_SOURCE
#include <cblas.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_cuda.h"
#include "coro/cacaosim_coro.h"
#include "pupil/cacaosim_pupil.h"

extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

static inline int coro_mask_present(const cacaosim_mask_cube_t *m)
{
    return (m != NULL && m->data != NULL && m->nx >= 1u && m->ny >= 1u && m->nz >= 1u) ? 1 : 0;
}

static inline void coro_mask_eval_cplx(
    const cacaosim_mask_cube_t *amp_m,
    const cacaosim_mask_cube_t *pha_m,
    int32_t                     c,
    uint64_t                    npix_plane,
    uint64_t                    idx,
    float                       lam_c,
    int32_t                     pha_unit,
    float                      *out_re,
    float                      *out_im,
    float                      *out_amp)
{
    float a   = 1.0f;
    float phi = 0.0f;

    if (coro_mask_present(amp_m))
    {
        uint32_t zc = (amp_m->nz > 1u) ? (uint32_t) c : 0u;
        if (zc >= amp_m->nz)
        {
            zc = amp_m->nz - 1u;
        }
        a = amp_m->data[(uint64_t) zc * npix_plane + idx];
    }

    if (coro_mask_present(pha_m))
    {
        uint32_t zc = (pha_m->nz > 1u) ? (uint32_t) c : 0u;
        if (zc >= pha_m->nz)
        {
            zc = pha_m->nz - 1u;
        }
        float raw = pha_m->data[(uint64_t) zc * npix_plane + idx];
        phi       = (pha_unit == 1) ? (raw * (CACAOSIM_TWO_PI_F / lam_c)) : raw;
    }

    float s  = 0.0f;
    float co = 1.0f;
    sincosf(phi, &s, &co);
    *out_re = a * co;
    *out_im = a * s;
    if (out_amp != NULL)
    {
        *out_amp = a;
    }
}

static void coro_precompute_babinet(
    cacaosim_coro_ctx_t         *ctx,
    const cacaosim_coro_masks_t *masks)
{
    const uint32_t nwf  = ctx->cfg.nwf;
    const uint32_t mx   = ctx->nx_mft;
    const uint32_t my   = ctx->ny_mft;
    const float    pdim = (ctx->cfg.pup_diam > 0.05f) ? ctx->cfg.pup_diam : 0.9375f;
    const float    dpix = pdim * (float) nwf;
    const float    p0   = 0.5f * (float) (nwf - 1u);
    const float    fx0  = 0.5f * (float) (mx - 1u);
    const float    fy0  = 0.5f * (float) (my - 1u);
    const float    lam0 = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um
                                                       : ctx->lam_um[ctx->eff_nlam / 2];

    float r_ld = (ctx->cfg.fpm_rad_ld > 0.05f) ? ctx->cfg.fpm_rad_ld : 3.0f;
    float du0  = 0.0f;
    if (ctx->cfg.mode == CACAOSIM_CORO_CUSTOM_MFT)
    {
        float ps0 = (ctx->cfg.fpm_pixscale_ld > 0.05f) ? ctx->cfg.fpm_pixscale_ld : 8.0f;
        du0       = 1.0f / ps0;
    }
    else
    {
        float fov = (2.4f * r_ld > 2.0f) ? (2.4f * r_ld) : 2.0f;
        du0       = fov / (float) mx;
    }

    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        const float du_c = du0 * (lam0 / ctx->lam_um[c]);
        const float nrm  = sqrtf(du_c / dpix);

        float *fwd_U = &ctx->fwd_U_cplx[(size_t) c * my * nwf * 2u];
        float *inv_U = &ctx->inv_U_cplx[(size_t) c * nwf * my * 2u];
        float *fwd_V = &ctx->fwd_V_cplx[(size_t) c * nwf * mx * 2u];
        float *inv_V = &ctx->inv_V_cplx[(size_t) c * mx * nwf * 2u];

        for (uint32_t ky = 0; ky < my; ky++)
        {
            const float v = ((float) ky - fy0) * du_c;
            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                const float yt  = ((float) yp - p0) / dpix;
                const float ang = -CACAOSIM_TWO_PI_F * yt * v;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);

                size_t idx_kp     = ((size_t) ky * nwf + yp) * 2u;
                size_t idx_pk     = ((size_t) yp * my + ky) * 2u;
                fwd_U[idx_kp]     = nrm * co;
                fwd_U[idx_kp + 1] = nrm * s;
                inv_U[idx_pk]     = nrm * co;
                inv_U[idx_pk + 1] = -nrm * s;
            }
        }

        for (uint32_t kx = 0; kx < mx; kx++)
        {
            const float u = ((float) kx - fx0) * du_c;
            for (uint32_t xp = 0; xp < nwf; xp++)
            {
                const float xt  = ((float) xp - p0) / dpix;
                const float ang = -CACAOSIM_TWO_PI_F * xt * u;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);

                size_t idx_pk     = ((size_t) xp * mx + kx) * 2u;
                size_t idx_kp     = ((size_t) kx * nwf + xp) * 2u;
                fwd_V[idx_pk]     = nrm * co;
                fwd_V[idx_pk + 1] = nrm * s;
                inv_V[idx_kp]     = nrm * co;
                inv_V[idx_kp + 1] = -nrm * s;
            }
        }

        float *fdel  = &ctx->fpm_delta_cplx[(size_t) c * ctx->npix_mft * 2u];
        float *frefl = &ctx->fpm_refl_w[(size_t) c * ctx->npix_mft];

        if (ctx->cfg.mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            const cacaosim_mask_cube_t *fa = (masks != NULL) ? &masks->fpm_amp : NULL;
            const cacaosim_mask_cube_t *fp = (masks != NULL) ? &masks->fpm_pha : NULL;
            for (uint64_t k = 0; k < ctx->npix_mft; k++)
            {
                float tr = 1.0f;
                float ti = 0.0f;
                float ta = 1.0f;
                coro_mask_eval_cplx(fa, fp, c, ctx->npix_mft, k, ctx->lam_um[c],
                                    ctx->cfg.mask_pha_unit, &tr, &ti, &ta);
                fdel[2u * k]      = tr - 1.0f;
                fdel[2u * k + 1u] = ti;
                float ac          = (ta < 0.0f) ? 0.0f : ((ta > 1.0f) ? 1.0f : ta);
                frefl[k]          = sqrtf(1.0f - ac * ac);
            }
        }
        else
        {
            const int   nsub   = 8;
            const float invs   = 1.0f / (float) (nsub * nsub);
            const float r2     = r_ld * r_ld;
            const float ph     = (ctx->cfg.mode == CACAOSIM_CORO_PHASE_DIMPLE)
                                     ? (ctx->cfg.fpm_phase_rad * (lam0 / ctx->lam_um[c]))
                                     : 0.0f;
            const float a0     = (ctx->cfg.fpm_amp < 0.0f)
                                     ? 0.0f
                                     : ((ctx->cfg.fpm_amp > 1.0f) ? 1.0f : ctx->cfg.fpm_amp);
            const float refl_c = sqrtf(1.0f - a0 * a0);
            const float c_re   = ctx->cfg.fpm_amp * cosf(ph) - 1.0f;
            const float c_im   = ctx->cfg.fpm_amp * sinf(ph);

            for (uint32_t ky = 0; ky < my; ky++)
            {
                const float vc = ((float) ky - fy0) * du0;
                for (uint32_t kx = 0; kx < mx; kx++)
                {
                    const float uc   = ((float) kx - fx0) * du0;
                    int         hits = 0;
                    for (int sy = 0; sy < nsub; sy++)
                    {
                        float vs = vc + (((float) sy + 0.5f) / (float) nsub - 0.5f) * du0;
                        for (int sx = 0; sx < nsub; sx++)
                        {
                            float us = uc + (((float) sx + 0.5f) / (float) nsub - 0.5f) * du0;
                            if (us * us + vs * vs <= r2)
                            {
                                hits++;
                            }
                        }
                    }
                    float    w        = (float) hits * invs;
                    uint64_t k        = (uint64_t) ky * mx + kx;
                    fdel[2u * k]      = w * c_re;
                    fdel[2u * k + 1u] = w * c_im;
                    frefl[k]          = w * refl_c;
                }
            }
        }
    }
}

static void coro_precompute_fft_mask(
    cacaosim_coro_ctx_t         *ctx,
    const cacaosim_coro_masks_t *masks)
{
    const uint32_t nfft      = ctx->nfft;
    const int32_t  half      = (int32_t) (nfft / 2u);
    const float    inv_nfft2 = 1.0f / (float) ctx->npix_fft;
    const int32_t  charge    = (ctx->cfg.vortex_charge == 4) ? 4 : 2;

    const cacaosim_mask_cube_t *fa = (masks != NULL && coro_mask_present(&masks->fpm_amp))
                                         ? &masks->fpm_amp
                                         : NULL;
    const cacaosim_mask_cube_t *fp = (masks != NULL && coro_mask_present(&masks->fpm_pha))
                                         ? &masks->fpm_pha
                                         : NULL;
    uint32_t fx = (fa != NULL) ? fa->nx : ((fp != NULL) ? fp->nx : nfft);
    uint32_t fy = (fa != NULL) ? fa->ny : ((fp != NULL) ? fp->ny : nfft);

    int half_x = 1;
    int half_y = 1;
    if (ctx->cfg.mode == CACAOSIM_CORO_CUSTOM_FFT && (fa != NULL || fp != NULL))
    {
        half_x = ((fx & 1u) == 0u) ? 1 : 0;
        half_y = ((fy & 1u) == 0u) ? 1 : 0;
    }

    float *sh_x = &ctx->shift_1d_cplx[0];
    float *sh_y = &ctx->shift_1d_cplx[(size_t) nfft * 2u];
    for (uint32_t m = 0; m < nfft; m++)
    {
        float ax         = half_x ? (-CACAOSIM_PI_F * (float) m / (float) nfft) : 0.0f;
        float ay         = half_y ? (-CACAOSIM_PI_F * (float) m / (float) nfft) : 0.0f;
        sh_x[2u * m]     = cosf(ax);
        sh_x[2u * m + 1] = sinf(ax);
        sh_y[2u * m]     = cosf(ay);
        sh_y[2u * m + 1] = sinf(ay);
    }

    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        float *fmsk = &ctx->fft_mask_cplx[(size_t) c * ctx->npix_fft * 2u];

        for (uint32_t jy = 0; jy < nfft; jy++)
        {
            int32_t iv = (jy < (uint32_t) half) ? (int32_t) jy : ((int32_t) jy - (int32_t) nfft);
            float   v  = (float) iv + 0.5f;
            for (uint32_t ix = 0; ix < nfft; ix++)
            {
                int32_t iu  = (ix < (uint32_t) half) ? (int32_t) ix
                                                     : ((int32_t) ix - (int32_t) nfft);
                float   u   = (float) iu + 0.5f;
                size_t  idx = ((size_t) jy * nfft + ix) * 2u;
                float   mr  = 1.0f;
                float   mi  = 0.0f;

                if (ctx->cfg.mode == CACAOSIM_CORO_CUSTOM_FFT)
                {
                    if (fa != NULL || fp != NULL)
                    {
                        int32_t mx = iu + (int32_t) (fx / 2u);
                        int32_t my = iv + (int32_t) (fy / 2u);
                        if (mx >= 0 && mx < (int32_t) fx && my >= 0 && my < (int32_t) fy)
                        {
                            uint64_t midx = (uint64_t) my * fx + (uint32_t) mx;
                            coro_mask_eval_cplx(fa, fp, c, (uint64_t) fx * fy, midx,
                                                ctx->lam_um[c], ctx->cfg.mask_pha_unit, &mr, &mi,
                                                NULL);
                        }
                    }
                }
                else if (ctx->cfg.mode == CACAOSIM_CORO_4QPM)
                {
                    mr = ((u > 0.0f) ? 1.0f : -1.0f) * ((v > 0.0f) ? 1.0f : -1.0f);
                    mi = 0.0f;
                }
                else
                {
                    float th = (float) charge * atan2f(v, u);
                    mr       = cosf(th);
                    mi       = sinf(th);
                }

                fmsk[idx]      = inv_nfft2 * mr;
                fmsk[idx + 1u] = inv_nfft2 * mi;
                if (c == (ctx->eff_nlam / 2) && ctx->fft_refl_w != NULL)
                {
                    float t2 = mr * mr + mi * mi;
                    float rw = (t2 < 1.0f) ? sqrtf(fmaxf(0.0f, 1.0f - t2)) : 0.0f;
                    ctx->fft_refl_w[(size_t) jy * nfft + ix] = inv_nfft2 * rw;
                }
            }
        }
    }
}

void MILK_HOT cacaosim_coro_step_poly_cpu(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    uint32_t             in_nlam,
    uint64_t             in_stride,
    float               *out_opd,
    float               *out_amp,
    uint64_t             out_stride,
    float               *lowfs_opd,
    float               *lowfs_amp)
{
    const uint32_t nwf      = ctx->cfg.nwf;
    const uint64_t npix_wf  = ctx->npix_wf;
    const int32_t  nlam     = ctx->eff_nlam;
    const int32_t  c_mid    = nlam / 2;
    const int      lo_lyot  = (ctx->cfg.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? 1 : 0;
    const float    refl     = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;
    const float    c_one[2]  = { 1.0f, 0.0f };
    const float    c_zero[2] = { 0.0f, 0.0f };

    const float *__restrict lp_msk = MILK_ASSUME_ALIGNED(ctx->lowfs_pup_mask);
    float *__restrict       ep     = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);

    for (int32_t c = 0; c < nlam; c++)
    {
        uint32_t ic = (in_nlam > 1u)
                          ? (((uint32_t) c < in_nlam) ? (uint32_t) c : (in_nlam - 1u))
                          : 0u;
        const float *opd_c     = (in_opd != NULL) ? (in_opd + (uint64_t) ic * in_stride) : NULL;
        const float *amp_c     = (in_amp != NULL) ? (in_amp + (uint64_t) ic * in_stride) : NULL;
        float       *out_opd_c = out_opd + (uint64_t) c * out_stride;
        float       *out_amp_c = out_amp + (uint64_t) c * out_stride;
        const int    do_lo     = (c == c_mid && lowfs_opd != NULL && lowfs_amp != NULL) ? 1 : 0;

        const float  lam_c      = ctx->lam_um[c];
        const float  opd_to_rad = CACAOSIM_TWO_PI_F / lam_c;
        const float  rad_to_opd = lam_c / CACAOSIM_TWO_PI_F;
        const float *pm_c       = ctx->has_pup_mask
                                      ? &ctx->pup_mask_cplx[(size_t) c * npix_wf * 2u]
                                      : NULL;
        const float *ly_c       = &ctx->lyot_cplx[(size_t) c * npix_wf * 2u];

        /* Build entrance + custom pupil-mask complex field */
        for (uint64_t p = 0; p < npix_wf; p++)
        {
            float amp = (amp_c != NULL) ? amp_c[p] : 1.0f;
            if (amp <= 0.0f)
            {
                ep[2u * p]      = 0.0f;
                ep[2u * p + 1u] = 0.0f;
                continue;
            }
            float phi = ((opd_c != NULL) ? opd_c[p] : 0.0f) * opd_to_rad;
            float s   = 0.0f;
            float co  = 1.0f;
            sincosf(phi, &s, &co);
            float pr = amp * co;
            float pi = amp * s;
            if (pm_c != NULL)
            {
                float mr = pm_c[2u * p];
                float mi = pm_c[2u * p + 1u];
                float nr = pr * mr - pi * mi;
                float ni = pr * mi + pi * mr;
                pr       = nr;
                pi       = ni;
            }
            ep[2u * p]      = pr;
            ep[2u * p + 1u] = pi;
        }

        if (ctx->skip_fpm)
        {
            for (uint64_t p = 0; p < npix_wf; p++)
            {
                float ur  = ep[2u * p];
                float ui  = ep[2u * p + 1u];
                float lyr = ly_c[2u * p];
                float lyi = ly_c[2u * p + 1u];
                float lr  = ur * lyr - ui * lyi;
                float li  = ur * lyi + ui * lyr;
                float amp = hypotf(lr, li);
                out_amp_c[p] = amp;
                out_opd_c[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

                if (do_lo)
                {
                    if (lo_lyot)
                    {
                        float lya2   = lyr * lyr + lyi * lyi;
                        float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
                        float la     = r_ly * hypotf(ur, ui);
                        lowfs_amp[p] = la;
                        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                    }
                    else
                    {
                        lowfs_amp[p] = 0.0f;
                        lowfs_opd[p] = 0.0f;
                    }
                }
            }
        }
        else if (ctx->is_babinet)
        {
            const uint32_t mx    = ctx->nx_mft;
            const uint32_t my    = ctx->ny_mft;
            const float   *fwd_U = &ctx->fwd_U_cplx[(size_t) c * my * nwf * 2u];
            const float   *inv_U = &ctx->inv_U_cplx[(size_t) c * nwf * my * 2u];
            const float   *fwd_V = &ctx->fwd_V_cplx[(size_t) c * nwf * mx * 2u];
            const float   *inv_V = &ctx->inv_V_cplx[(size_t) c * mx * nwf * 2u];
            float *__restrict ec = MILK_ASSUME_ALIGNED(ctx->cpu_ecore_cplx);
            float *__restrict ef = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);

            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) my, (int) nwf, (int) nwf,
                        c_one, fwd_U, (int) nwf, ep, (int) nwf, c_zero, ctx->cpu_tmp_cplx,
                        (int) nwf);
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) my, (int) mx, (int) nwf,
                        c_one, ctx->cpu_tmp_cplx, (int) nwf, fwd_V, (int) mx, c_zero, ef,
                        (int) mx);

            if (do_lo && !lo_lyot)
            {
                const float *__restrict rw = &ctx->fpm_refl_w[(size_t) c * ctx->npix_mft];
                float *__restrict       er = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_r_cplx);
#pragma omp simd
                for (uint64_t i = 0; i < ctx->npix_mft; i++)
                {
                    float w         = rw[i];
                    er[2u * i]      = ef[2u * i] * w;
                    er[2u * i + 1u] = ef[2u * i + 1u] * w;
                }
                cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) mx,
                            (int) my, c_one, inv_U, (int) my, er, (int) mx, c_zero,
                            ctx->cpu_tmp_cplx, (int) mx);
                cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) nwf,
                            (int) mx, c_one, ctx->cpu_tmp_cplx, (int) mx, inv_V, (int) nwf, c_zero,
                            ec, (int) nwf);
                for (uint64_t p = 0; p < npix_wf; p++)
                {
                    float cr     = ec[2u * p];
                    float ci     = ec[2u * p + 1u];
                    float pm     = (lp_msk != NULL) ? lp_msk[p] : 1.0f;
                    float la     = refl * pm * hypotf(cr, ci);
                    lowfs_amp[p] = la;
                    lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ci, cr) * rad_to_opd) : 0.0f;
                }
            }

            const float *__restrict fd = &ctx->fpm_delta_cplx[(size_t) c * ctx->npix_mft * 2u];
#pragma omp simd
            for (uint64_t i = 0; i < ctx->npix_mft; i++)
            {
                float er        = ef[2u * i];
                float ei        = ef[2u * i + 1u];
                float dr        = fd[2u * i];
                float di        = fd[2u * i + 1u];
                ef[2u * i]      = er * dr - ei * di;
                ef[2u * i + 1u] = er * di + ei * dr;
            }

            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) mx, (int) my,
                        c_one, inv_U, (int) my, ef, (int) mx, c_zero, ctx->cpu_tmp_cplx, (int) mx);
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) nwf, (int) mx,
                        c_one, ctx->cpu_tmp_cplx, (int) mx, inv_V, (int) nwf, c_zero, ec,
                        (int) nwf);

            for (uint64_t p = 0; p < npix_wf; p++)
            {
                float ur  = ep[2u * p] + ec[2u * p];
                float ui  = ep[2u * p + 1u] + ec[2u * p + 1u];
                float lyr = ly_c[2u * p];
                float lyi = ly_c[2u * p + 1u];
                float lr  = ur * lyr - ui * lyi;
                float li  = ur * lyi + ui * lyr;
                float amp = hypotf(lr, li);
                out_amp_c[p] = amp;
                out_opd_c[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

                if (do_lo && lo_lyot)
                {
                    float lya2   = lyr * lyr + lyi * lyi;
                    float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
                    float la     = r_ly * hypotf(ur, ui);
                    lowfs_amp[p] = la;
                    lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                }
            }
        }
        else
        {
            const uint32_t          nfft    = ctx->nfft;
            const uint32_t          pad_off = (nfft - nwf) / 2u;
            float *__restrict       fbuf    = MILK_ASSUME_ALIGNED(ctx->cpu_fft_buf);
            const float *__restrict sh_x    = &ctx->shift_1d_cplx[0];
            const float *__restrict sh_y    = &ctx->shift_1d_cplx[(size_t) nfft * 2u];
            const float *__restrict fmsk    =
                MILK_ASSUME_ALIGNED(&ctx->fft_mask_cplx[(size_t) c * ctx->npix_fft * 2u]);

            __builtin_memset(fbuf, 0, ctx->npix_fft * 2u * sizeof(float));

            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                uint32_t gy    = yp + pad_off;
                float    sy_re = sh_y[2u * gy];
                float    sy_im = sh_y[2u * gy + 1u];
                float   *dst   = &fbuf[((size_t) gy * nfft + pad_off) * 2u];
                uint64_t pbase = (uint64_t) yp * nwf;

                for (uint32_t xp = 0; xp < nwf; xp++)
                {
                    uint64_t p     = pbase + xp;
                    float    pr    = ep[2u * p];
                    float    pi    = ep[2u * p + 1u];
                    uint32_t gx    = xp + pad_off;
                    float    sx_re = sh_x[2u * gx];
                    float    sx_im = sh_x[2u * gx + 1u];
                    float    sr    = sx_re * sy_re - sx_im * sy_im;
                    float    si    = sx_re * sy_im + sx_im * sy_re;
                    dst[2u * xp]      = pr * sr - pi * si;
                    dst[2u * xp + 1u] = pr * si + pi * sr;
                }
            }

            fftwf_execute((fftwf_plan) ctx->fftw_fwd);

            if (do_lo && !lo_lyot && ctx->cpu_fft_lo_buf != NULL && ctx->fft_refl_w != NULL)
            {
                float *__restrict       flob = MILK_ASSUME_ALIGNED(ctx->cpu_fft_lo_buf);
                const float *__restrict frw  = MILK_ASSUME_ALIGNED(ctx->fft_refl_w);
#pragma omp simd
                for (uint64_t i = 0; i < ctx->npix_fft; i++)
                {
                    float w          = frw[i];
                    flob[2u * i]     = fbuf[2u * i] * w;
                    flob[2u * i + 1] = fbuf[2u * i + 1u] * w;
                }
                fftwf_execute((fftwf_plan) ctx->fftw_lo_inv);

                for (uint32_t yp = 0; yp < nwf; yp++)
                {
                    uint32_t     gy    = yp + pad_off;
                    float        sy_re = sh_y[2u * gy];
                    float        sy_im = -sh_y[2u * gy + 1u];
                    const float *src   = &flob[((size_t) gy * nfft + pad_off) * 2u];
                    uint64_t     pbase = (uint64_t) yp * nwf;

                    for (uint32_t xp = 0; xp < nwf; xp++)
                    {
                        uint64_t p     = pbase + xp;
                        uint32_t gx    = xp + pad_off;
                        float    sx_re = sh_x[2u * gx];
                        float    sx_im = -sh_x[2u * gx + 1u];
                        float    sr    = sx_re * sy_re - sx_im * sy_im;
                        float    si    = sx_re * sy_im + sx_im * sy_re;
                        float    er    = src[2u * xp];
                        float    ei    = src[2u * xp + 1u];
                        float    ur    = er * sr - ei * si;
                        float    ui    = er * si + ei * sr;
                        float    pm    = (lp_msk != NULL) ? lp_msk[p] : 1.0f;
                        float    la    = refl * pm * hypotf(ur, ui);
                        lowfs_amp[p]   = la;
                        lowfs_opd[p]   = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                    }
                }
            }

#pragma omp simd
            for (uint64_t i = 0; i < ctx->npix_fft; i++)
            {
                float er        = fbuf[2u * i];
                float ei        = fbuf[2u * i + 1u];
                float mr        = fmsk[2u * i];
                float mi        = fmsk[2u * i + 1u];
                fbuf[2u * i]    = er * mr - ei * mi;
                fbuf[2u * i + 1] = er * mi + ei * mr;
            }

            fftwf_execute((fftwf_plan) ctx->fftw_inv);

            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                uint32_t     gy    = yp + pad_off;
                float        sy_re = sh_y[2u * gy];
                float        sy_im = -sh_y[2u * gy + 1u];
                const float *src   = &fbuf[((size_t) gy * nfft + pad_off) * 2u];
                uint64_t     pbase = (uint64_t) yp * nwf;

                for (uint32_t xp = 0; xp < nwf; xp++)
                {
                    uint64_t p     = pbase + xp;
                    uint32_t gx    = xp + pad_off;
                    float    sx_re = sh_x[2u * gx];
                    float    sx_im = -sh_x[2u * gx + 1u];
                    float    sr    = sx_re * sy_re - sx_im * sy_im;
                    float    si    = sx_re * sy_im + sx_im * sy_re;
                    float    er    = src[2u * xp];
                    float    ei    = src[2u * xp + 1u];
                    float    ur    = er * sr - ei * si;
                    float    ui    = er * si + ei * sr;

                    float lyr    = ly_c[2u * p];
                    float lyi    = ly_c[2u * p + 1u];
                    float lr     = ur * lyr - ui * lyi;
                    float li     = ur * lyi + ui * lyr;
                    float amp    = hypotf(lr, li);
                    out_amp_c[p] = amp;
                    out_opd_c[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

                    if (do_lo && lo_lyot)
                    {
                        float lya2   = lyr * lyr + lyi * lyi;
                        float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
                        float la     = r_ly * hypotf(ur, ui);
                        lowfs_amp[p] = la;
                        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                    }
                }
            }
        }
    }
}

void MILK_HOT cacaosim_coro_step_cpu(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    float               *out_opd,
    float               *out_amp,
    float               *lowfs_opd,
    float               *lowfs_amp)
{
    if (ctx->eff_nlam > 1 && ctx->cpu_poly_opd != NULL && ctx->cpu_poly_amp != NULL)
    {
        cacaosim_coro_step_poly_cpu(ctx, in_opd, in_amp, 1u, ctx->npix_wf, ctx->cpu_poly_opd,
                                    ctx->cpu_poly_amp, ctx->npix_wf, lowfs_opd, lowfs_amp);
        size_t off = (size_t) (ctx->eff_nlam / 2) * ctx->npix_wf;
        __builtin_memcpy(out_opd, ctx->cpu_poly_opd + off, ctx->npix_wf * sizeof(float));
        __builtin_memcpy(out_amp, ctx->cpu_poly_amp + off, ctx->npix_wf * sizeof(float));
        return;
    }
    cacaosim_coro_step_poly_cpu(ctx, in_opd, in_amp, 1u, ctx->npix_wf, out_opd, out_amp,
                                ctx->npix_wf, lowfs_opd, lowfs_amp);
}

int MILK_COLD cacaosim_coro_init_masks(
    cacaosim_coro_ctx_t         *ctx,
    const cacaosim_coro_cfg_t   *cfg,
    const cacaosim_coro_masks_t *masks,
    void                        *ext_stream)
{
    if (ctx == NULL || cfg == NULL || cfg->nwf < 8)
    {
        return -1;
    }

    const cacaosim_mask_cube_t *pa = (masks != NULL && coro_mask_present(&masks->pup_amp))
                                         ? &masks->pup_amp
                                         : NULL;
    const cacaosim_mask_cube_t *pp = (masks != NULL && coro_mask_present(&masks->pup_pha))
                                         ? &masks->pup_pha
                                         : NULL;
    const cacaosim_mask_cube_t *fa = (masks != NULL && coro_mask_present(&masks->fpm_amp))
                                         ? &masks->fpm_amp
                                         : NULL;
    const cacaosim_mask_cube_t *fp = (masks != NULL && coro_mask_present(&masks->fpm_pha))
                                         ? &masks->fpm_pha
                                         : NULL;
    const cacaosim_mask_cube_t *la = (masks != NULL && coro_mask_present(&masks->lyot_amp))
                                         ? &masks->lyot_amp
                                         : NULL;
    const cacaosim_mask_cube_t *lp = (masks != NULL && coro_mask_present(&masks->lyot_pha))
                                         ? &masks->lyot_pha
                                         : NULL;

    if ((pa != NULL && (pa->nx != cfg->nwf || pa->ny != cfg->nwf)) ||
        (pp != NULL && (pp->nx != cfg->nwf || pp->ny != cfg->nwf)) ||
        (la != NULL && (la->nx != cfg->nwf || la->ny != cfg->nwf)) ||
        (lp != NULL && (lp->nx != cfg->nwf || lp->ny != cfg->nwf)))
    {
        return -1;
    }
    if (fa != NULL && fp != NULL && (fa->nx != fp->nx || fa->ny != fp->ny))
    {
        return -1;
    }

    const cacaosim_mask_cube_t *all_m[6] = { pa, pp, fa, fp, la, lp };
    uint32_t                    mask_nz  = 1u;
    const cacaosim_mask_cube_t *wcs_src  = NULL;
    for (int i = 0; i < 6; i++)
    {
        if (all_m[i] != NULL && all_m[i]->nz > 1u)
        {
            if (all_m[i]->nz > CACAOSIM_CORO_MAX_LAMBDA)
            {
                return -1;
            }
            if (mask_nz > 1u && all_m[i]->nz != mask_nz)
            {
                return -1;
            }
            mask_nz = all_m[i]->nz;
            if (wcs_src == NULL && all_m[i]->has_wcs_lam)
            {
                wcs_src = all_m[i];
            }
        }
        else if (all_m[i] != NULL && wcs_src == NULL && all_m[i]->has_wcs_lam)
        {
            wcs_src = all_m[i];
        }
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg     = *cfg;
    ctx->npix_wf = (uint64_t) cfg->nwf * cfg->nwf;

    int32_t eff_mode = cfg->mode;
    if ((fa != NULL || fp != NULL) && eff_mode < CACAOSIM_CORO_CUSTOM_MFT)
    {
        eff_mode = (eff_mode == CACAOSIM_CORO_LYOT_SPOT ||
                    eff_mode == CACAOSIM_CORO_PHASE_DIMPLE)
                       ? CACAOSIM_CORO_CUSTOM_MFT
                       : CACAOSIM_CORO_CUSTOM_FFT;
    }
    ctx->cfg.mode   = eff_mode;
    ctx->is_babinet = (eff_mode == CACAOSIM_CORO_LYOT_SPOT ||
                       eff_mode == CACAOSIM_CORO_PHASE_DIMPLE ||
                       eff_mode == CACAOSIM_CORO_CUSTOM_MFT)
                          ? 1
                          : 0;
    ctx->skip_fpm   = (eff_mode == CACAOSIM_CORO_CUSTOM_MFT && fa == NULL && fp == NULL) ? 1 : 0;

    int32_t nlam = 1;
    if (mask_nz > 1u)
    {
        if (cfg->nlambda > 1 && (uint32_t) cfg->nlambda != mask_nz)
        {
            return -1;
        }
        nlam = (int32_t) mask_nz;
    }
    else if (cfg->nlambda > 1 && cfg->bw_frac > 1e-6f)
    {
        nlam = (cfg->nlambda <= CACAOSIM_CORO_MAX_LAMBDA) ? cfg->nlambda
                                                          : CACAOSIM_CORO_MAX_LAMBDA;
    }
    ctx->eff_nlam = nlam;

    float lam0 = (cfg->lambda_um > 0.05f) ? cfg->lambda_um : 1.65f;
    if (wcs_src != NULL && wcs_src->nz == (uint32_t) nlam && wcs_src->lam_um[0] > 0.05f)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            ctx->lam_um[c] = wcs_src->lam_um[c];
        }
        if (cfg->lambda_um <= 0.05f)
        {
            lam0               = 0.5f * (ctx->lam_um[0] + ctx->lam_um[nlam - 1]);
            ctx->cfg.lambda_um = lam0;
        }
    }
    else
    {
        float bw = (cfg->bw_frac > 1e-6f) ? cfg->bw_frac : ((nlam > 1) ? 0.20f : 0.0f);
        for (int32_t c = 0; c < nlam; c++)
        {
            float u        = (nlam > 1) ? (((float) c + 0.5f) / (float) nlam - 0.5f) : 0.0f;
            ctx->lam_um[c] = lam0 * (1.0f + bw * u);
        }
    }

    if (openblas_set_num_threads != NULL)
    {
        openblas_set_num_threads(1);
    }

    ctx->lyot_mask      = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    ctx->lowfs_pup_mask = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    ctx->lyot_cplx      = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf * 2u,
                                                            sizeof(float));
    ctx->cpu_epup_cplx  = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
    if (!ctx->lyot_mask || !ctx->lowfs_pup_mask || !ctx->lyot_cplx || !ctx->cpu_epup_cplx)
    {
        cacaosim_coro_free(ctx);
        return -1;
    }

    {
        float pdim    = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 0.9375f;
        float lp_rel  = (cfg->lowfs_pup_diam > 0.0f) ? cfg->lowfs_pup_diam : 1.0f;
        float lp_diam = pdim * lp_rel;
        if (lp_diam < 0.01f)
        {
            lp_diam = 0.01f;
        }
        if (lp_diam > 1.0f)
        {
            lp_diam = 1.0f;
        }
        cacaosim_pupil_cfg_t lpcfg = {
            .nwf      = cfg->nwf,
            .pup_diam = lp_diam,
            .cobs     = 0.0f,
            .nspider  = 0,
        };
        cacaosim_make_pupil(&lpcfg, ctx->lowfs_pup_mask);
    }

    if (nlam > 1)
    {
        ctx->cpu_poly_opd = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf,
                                                              sizeof(float));
        ctx->cpu_poly_amp = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf,
                                                              sizeof(float));
        if (!ctx->cpu_poly_opd || !ctx->cpu_poly_amp)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
    }

    ctx->has_pup_mask = (pa != NULL || pp != NULL) ? 1 : 0;
    if (ctx->has_pup_mask)
    {
        ctx->pup_mask_cplx =
            (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf * 2u, sizeof(float));
        if (!ctx->pup_mask_cplx)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
        for (int32_t c = 0; c < nlam; c++)
        {
            float *dst = &ctx->pup_mask_cplx[(size_t) c * ctx->npix_wf * 2u];
            for (uint64_t p = 0; p < ctx->npix_wf; p++)
            {
                coro_mask_eval_cplx(pa, pp, c, ctx->npix_wf, p, ctx->lam_um[c],
                                    cfg->mask_pha_unit, &dst[2u * p], &dst[2u * p + 1u], NULL);
            }
        }
    }

    cacaosim_mask_cube_t geom_lyot_view = { 0 };
    const cacaosim_mask_cube_t *lyot_amp_src = la;
    if (la != NULL)
    {
        __builtin_memcpy(ctx->lyot_mask, la->data, ctx->npix_wf * sizeof(float));
    }
    else
    {
        float pdim  = (cfg->pup_diam > 0.05f) ? cfg->pup_diam : 0.9375f;
        float outer = (cfg->lyot_outer > 0.05f) ? cfg->lyot_outer : 0.85f;
        float obs_p = (cfg->lyot_obs >= 0.0f) ? cfg->lyot_obs : 0.36f;
        cacaosim_pupil_cfg_t lcfg = {
            .nwf        = cfg->nwf,
            .pup_diam   = pdim * outer,
            .cobs       = (outer > 1e-4f) ? (obs_p / outer) : 0.0f,
            .nspider    = cfg->lyot_nspider,
            .spider_w   = (outer > 1e-4f) ? (cfg->lyot_spider_w / outer) : 0.0f,
            .spider_deg = cfg->lyot_spider_deg,
        };
        if (lcfg.cobs > 0.95f)
        {
            lcfg.cobs = 0.95f;
        }
        cacaosim_make_pupil(&lcfg, ctx->lyot_mask);
        geom_lyot_view = cacaosim_mask_view_2d(ctx->lyot_mask, cfg->nwf, cfg->nwf);
        lyot_amp_src   = &geom_lyot_view;
    }

    for (int32_t c = 0; c < nlam; c++)
    {
        float *dst = &ctx->lyot_cplx[(size_t) c * ctx->npix_wf * 2u];
        for (uint64_t p = 0; p < ctx->npix_wf; p++)
        {
            coro_mask_eval_cplx(lyot_amp_src, lp, c, ctx->npix_wf, p, ctx->lam_um[c],
                                cfg->mask_pha_unit, &dst[2u * p], &dst[2u * p + 1u], NULL);
        }
    }

    if (ctx->is_babinet && !ctx->skip_fpm)
    {
        if (eff_mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            ctx->nx_mft = (fa != NULL) ? fa->nx : fp->nx;
            ctx->ny_mft = (fa != NULL) ? fa->ny : fp->ny;
            if (ctx->nx_mft < 2u || ctx->ny_mft < 2u)
            {
                cacaosim_coro_free(ctx);
                return -1;
            }
        }
        else
        {
            uint32_t m  = (cfg->n_mft >= 16u) ? cfg->n_mft : 48u;
            ctx->nx_mft = m;
            ctx->ny_mft = m;
        }
        ctx->n_mft    = ctx->nx_mft;
        ctx->npix_mft = (uint64_t) ctx->nx_mft * ctx->ny_mft;

        uint32_t max_m   = (ctx->nx_mft > ctx->ny_mft) ? ctx->nx_mft : ctx->ny_mft;
        size_t   u_elems = (size_t) nlam * ctx->ny_mft * cfg->nwf * 2u;
        size_t   v_elems = (size_t) nlam * ctx->nx_mft * cfg->nwf * 2u;
        size_t   t_elems = (size_t) max_m * cfg->nwf * 2u;

        ctx->fwd_U_cplx     = (float *) cacaosim_aligned_calloc(u_elems, sizeof(float));
        ctx->inv_U_cplx     = (float *) cacaosim_aligned_calloc(u_elems, sizeof(float));
        ctx->fwd_V_cplx     = (float *) cacaosim_aligned_calloc(v_elems, sizeof(float));
        ctx->inv_V_cplx     = (float *) cacaosim_aligned_calloc(v_elems, sizeof(float));
        ctx->fpm_delta_cplx = (float *) cacaosim_aligned_calloc(
            (size_t) nlam * ctx->npix_mft * 2u, sizeof(float));
        ctx->fpm_refl_w     = (float *) cacaosim_aligned_calloc(
            (size_t) nlam * ctx->npix_mft, sizeof(float));
        ctx->cpu_ecore_cplx  = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
        ctx->cpu_tmp_cplx    = (float *) cacaosim_aligned_calloc(t_elems, sizeof(float));
        ctx->cpu_efoc_cplx   = (float *) cacaosim_aligned_calloc(ctx->npix_mft * 2u, sizeof(float));
        ctx->cpu_efoc_r_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_mft * 2u, sizeof(float));

        if (!ctx->fwd_U_cplx || !ctx->inv_U_cplx || !ctx->fwd_V_cplx || !ctx->inv_V_cplx ||
            !ctx->fpm_delta_cplx || !ctx->fpm_refl_w || !ctx->cpu_ecore_cplx ||
            !ctx->cpu_tmp_cplx || !ctx->cpu_efoc_cplx || !ctx->cpu_efoc_r_cplx)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
        coro_precompute_babinet(ctx, masks);
    }
    else if (!ctx->is_babinet)
    {
        uint32_t pad  = (cfg->pad_factor >= 2u) ? cfg->pad_factor : 4u;
        ctx->nfft     = cfg->nwf * pad;
        ctx->npix_fft = (uint64_t) ctx->nfft * ctx->nfft;

        if ((fa != NULL && (fa->nx > ctx->nfft || fa->ny > ctx->nfft)) ||
            (fp != NULL && (fp->nx > ctx->nfft || fp->ny > ctx->nfft)))
        {
            cacaosim_coro_free(ctx);
            return -1;
        }

        ctx->shift_1d_cplx  = (float *) cacaosim_aligned_calloc((size_t) ctx->nfft * 4u,
                                                                sizeof(float));
        ctx->fft_mask_cplx  = (float *) cacaosim_aligned_calloc(
            (size_t) nlam * ctx->npix_fft * 2u, sizeof(float));
        ctx->fft_refl_w     = (float *) cacaosim_aligned_calloc(ctx->npix_fft, sizeof(float));
        ctx->cpu_fft_buf    = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        ctx->cpu_fft_lo_buf = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));

        if (!ctx->shift_1d_cplx || !ctx->fft_mask_cplx || !ctx->fft_refl_w || !ctx->cpu_fft_buf ||
            !ctx->cpu_fft_lo_buf)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
        coro_precompute_fft_mask(ctx, masks);

        int n            = (int) ctx->nfft;
        ctx->fftw_fwd    = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_buf, (fftwf_complex *) ctx->cpu_fft_buf,
            FFTW_FORWARD, FFTW_ESTIMATE);
        ctx->fftw_inv    = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_buf, (fftwf_complex *) ctx->cpu_fft_buf,
            FFTW_BACKWARD, FFTW_ESTIMATE);
        ctx->fftw_lo_inv = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_lo_buf, (fftwf_complex *) ctx->cpu_fft_lo_buf,
            FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!ctx->fftw_fwd || !ctx->fftw_inv || !ctx->fftw_lo_inv)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
    }

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_coro_cuda_init(ctx, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

int MILK_COLD cacaosim_coro_init(
    cacaosim_coro_ctx_t       *ctx,
    const cacaosim_coro_cfg_t *cfg,
    const float               *ext_lyot,
    void                      *ext_stream)
{
    cacaosim_coro_masks_t masks = { 0 };
    if (ext_lyot != NULL && cfg != NULL)
    {
        masks.lyot_amp = cacaosim_mask_view_2d(ext_lyot, cfg->nwf, cfg->nwf);
    }
    return cacaosim_coro_init_masks(ctx, cfg, &masks, ext_stream);
}

void MILK_COLD cacaosim_coro_free(cacaosim_coro_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_coro_cuda_free(ctx);
    }
    if (ctx->fftw_fwd)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_fwd);
    }
    if (ctx->fftw_inv)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_inv);
    }
    if (ctx->fftw_lo_inv)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_lo_inv);
    }
    free(ctx->lyot_mask);
    free(ctx->lowfs_pup_mask);
    free(ctx->pup_mask_cplx);
    free(ctx->lyot_cplx);
    free(ctx->fwd_U_cplx);
    free(ctx->fwd_V_cplx);
    free(ctx->inv_U_cplx);
    free(ctx->inv_V_cplx);
    free(ctx->fpm_delta_cplx);
    free(ctx->fpm_refl_w);
    free(ctx->shift_1d_cplx);
    free(ctx->fft_mask_cplx);
    free(ctx->fft_refl_w);
    free(ctx->cpu_epup_cplx);
    free(ctx->cpu_ecore_cplx);
    free(ctx->cpu_tmp_cplx);
    free(ctx->cpu_efoc_cplx);
    free(ctx->cpu_efoc_r_cplx);
    free(ctx->cpu_fft_buf);
    free(ctx->cpu_fft_lo_buf);
    free(ctx->cpu_poly_opd);
    free(ctx->cpu_poly_amp);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

void MILK_HOT cacaosim_coro_step_poly(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    uint32_t             in_nlam,
    uint64_t             in_stride,
    float               *out_opd,
    float               *out_amp,
    uint64_t             out_stride,
    float               *lowfs_opd,
    float               *lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp)
{
    if (ctx->use_cuda)
    {
        cacaosim_coro_cuda_step_poly(ctx, in_opd, in_amp, in_nlam, in_stride, out_opd, out_amp,
                                     out_stride, lowfs_opd, lowfs_amp, d_in_opd, d_in_amp,
                                     d_out_opd, d_out_amp, d_lowfs_opd, d_lowfs_amp);
        return;
    }
    cacaosim_coro_step_poly_cpu(ctx, in_opd, in_amp, in_nlam, in_stride, out_opd, out_amp,
                                out_stride, lowfs_opd, lowfs_amp);
}

void MILK_HOT cacaosim_coro_step_beam(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    float               *out_opd,
    float               *out_amp,
    float               *lowfs_opd,
    float               *lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp)
{
    if (ctx->eff_nlam > 1 && d_out_opd == NULL)
    {
        if (ctx->use_cuda && ctx->cpu_poly_opd != NULL && ctx->cpu_poly_amp != NULL)
        {
            cacaosim_coro_cuda_step_poly(ctx, in_opd, in_amp, 1u, ctx->npix_wf, ctx->cpu_poly_opd,
                                         ctx->cpu_poly_amp, ctx->npix_wf, lowfs_opd, lowfs_amp,
                                         d_in_opd, d_in_amp, NULL, NULL, d_lowfs_opd, d_lowfs_amp);
            size_t off = (size_t) (ctx->eff_nlam / 2) * ctx->npix_wf;
            __builtin_memcpy(out_opd, ctx->cpu_poly_opd + off, ctx->npix_wf * sizeof(float));
            __builtin_memcpy(out_amp, ctx->cpu_poly_amp + off, ctx->npix_wf * sizeof(float));
            return;
        }
        cacaosim_coro_step_cpu(ctx, in_opd, in_amp, out_opd, out_amp, lowfs_opd, lowfs_amp);
        return;
    }

    cacaosim_coro_step_poly(ctx, in_opd, in_amp, 1u, ctx->npix_wf, out_opd, out_amp, ctx->npix_wf,
                            lowfs_opd, lowfs_amp, d_in_opd, d_in_amp, d_out_opd, d_out_amp,
                            d_lowfs_opd, d_lowfs_amp);
}
