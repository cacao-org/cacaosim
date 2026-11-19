// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_engine.c
 * @brief   Coronagraphic pupil/focal custom masks, polychromatic MFT/FFT, Lyot, & LOWFS engine
 *
 * Architectural Role in cacaosim:
 *   Implements the host cold-path initialization, precomputation of polychromatic Fourier optical
 *   operators, hot-path CPU diffraction kernels (OpenBLAS `cblas_cgemm` Matrix-DFT and FFTW3f
 *   padded 2D FFTs), and unified CPU/CUDA backend dispatchers for `cacaosim.coro`. Supports all
 *   six coronagraph modes (`lyot_spot`, `phase_dimple`, `vortex`, `fqpm`, `custom_mft`,
 *   `custom_fft`), arbitrary 2D/3D `[Nx, Ny, Nlambda]` pupil/focal/Lyot amplitude and phase/OPD
 *   mask cubes, complex post-Lyot `[OPD_um, Amplitude]` extraction, and dual `out_lowfs` pickoff
 *   physics (`fpm_refl` with relay pupil stop `lowfs_pup_diam` vs `lyot_refl` reflective Lyot).
 *
 * Design Rationale & Optical Physics:
 *   - Soummer Babinet Matrix-DFT (`is_babinet == 1`):
 *     Expresses compact focal-plane masks as `T(u,v,lam) = 1 + Delta_M(u,v,lam)` where
 *     `Delta_M = T - 1` vanishes outside a small `nx_mft x ny_mft` focal box. By Babinet's
 *     principle, the pre-Lyot complex pupil amplitude is:
 *       Psi_Lyot_in = Psi_pup + MFT^-1[ (T - 1) * MFT[Psi_pup] ]
 *     Each 2D MFT factors into two tiny dense matrix products (`Efoc = fwd_U * Epup * fwd_V`),
 *     achieving arbitrary sub-pixel focal sampling (e.g. 8..16 pix per lambda0/D) and exact
 *     physical chromatic beam magnification `du_c = du0 * (lam0 / lam_c)` without massive
 *     8x..16x zero-padded 2D FFTs.
 *   - Shifted Padded 2D FFT (`is_babinet == 0`):
 *     Extended scale-invariant masks (Optical Vortex and 4QPM) have infinite spatial support and
 *     cannot be truncated to a compact Babinet box. A 1D linear pupil phase ramp
 *     `exp(-i * pi * m / nfft)` shifts the FFT focal grid by `(-0.5, -0.5)` bins so `(u,v)=(0,0)`
 *     falls symmetrically at the inter-pixel vertex between the central four pixels rather than
 *     landing on a singular pixel center, and the `1 / nfft^2` round-trip FFT normalization is
 *     pre-multiplied into `fft_mask_cplx` and `fft_refl_w` at initialization.
 *   - OpenBLAS Thread Pinning:
 *     `openblas_set_num_threads(1)` (resolved via weak linkage) pins internal BLAS worker pools to
 *     a single thread so small `48 x 128` complex GEMMs do not incur pthread barrier contention or
 *     oversubscribe CPU cores alongside sibling real-time AO processes.
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
#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "coro/cacaosim_coro.h"
#include "pupil/cacaosim_pupil.h"

/*
 * Weak-linked OpenBLAS thread-count setter: resolves at runtime when linked against OpenBLAS and
 * evaluates to NULL safely when linked against MKL, BLIS, or Netlib CBLAS.
 */
extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

/**
 * coro_mask_present() - Check whether an optional mask cube descriptor holds valid pixel data
 * @m: Candidate mask cube pointer (may be NULL or zero-initialized).
 *
 * Role:
 *   Lightweight cold-path predicate used when inspecting optional pupil, focal, and Lyot masks.
 *
 * Rationale:
 *   Treats NULL pointers, unloaded descriptors (`data == NULL`), and degenerate zero-sized axes
 *   uniformly so callers can pass partially populated `cacaosim_coro_masks_t` structs cleanly.
 *
 * Return: 1 if `@m` contains at least a 1x1x1 float array, 0 otherwise.
 */
static inline int coro_mask_present(const cacaosim_mask_cube_t *m)
{
    return (m != NULL && m->data != NULL && m->nx >= 1u && m->ny >= 1u && m->nz >= 1u) ? 1 : 0;
}

/**
 * coro_mask_eval_cplx() - Evaluate complex mask phasor `A * exp(i * phi)` at pixel and wavelength
 * @amp_m:      Optional real amplitude mask cube (NULL or empty defaults to `A = 1.0`).
 * @pha_m:      Optional phase `[rad]` or OPD `[um]` mask cube (NULL or empty defaults to `0.0`).
 * @c:          Spectral channel index (`0 .. eff_nlam - 1`).
 * @npix_plane: Number of 2D pixels per spectral slice (`nx * ny`).
 * @idx:        2D linear pixel index within the slice (`0 .. npix_plane - 1`).
 * @lam_c:      Wavelength `[um]` of spectral channel `@c`.
 * @pha_unit:   Phase convention (`0` = phase `[rad]` directly, `1` = OPD `[um]` converted via
 *              `2 * pi * OPD / lam_c`).
 * @out_re:     Output real part `A * cos(phi)` (non-NULL).
 * @out_im:     Output imaginary part `A * sin(phi)` (non-NULL).
 * @out_amp:    Optional output scalar amplitude `A` (may be NULL).
 *
 * Role:
 *   Cold-path helper converting user 2D/3D amplitude and phase/OPD mask cubes into precomputed
 *   per-channel complex transmission arrays (`pup_mask_cplx`, `fpm_delta_cplx`, `fft_mask_cplx`,
 *   and `lyot_cplx`).
 *
 * Rationale:
 *   Supports mixing an achromatic 2D amplitude mask (`nz == 1`) with a chromatic 3D phase cube
 *   (`nz == eff_nlam`) or vice versa, and handles both geometric-phase achromatic masks (`[rad]`)
 *   and dispersive substrate masks (`OPD [um]`) in one unified evaluation step.
 */
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

    /* Broadcast slice 0 when the amplitude mask is 2D (nz == 1); otherwise index slice c */
    if (coro_mask_present(amp_m))
    {
        uint32_t zc = (amp_m->nz > 1u) ? (uint32_t) c : 0u;
        if (zc >= amp_m->nz)
        {
            zc = amp_m->nz - 1u;
        }
        a = amp_m->data[(uint64_t) zc * npix_plane + idx];
    }

    /* Convert OPD [um] to chromatic phase 2*pi*OPD/lam_c when pha_unit == 1 */
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

/**
 * coro_precompute_babinet() - Build per-channel Soummer MFT basis matrices and FPM core weights
 * @ctx:   Coronagraph context with allocated Babinet MFT arrays (`nx_mft`, `ny_mft`, `eff_nlam`).
 * @masks: Optional custom focal-plane mask cubes used when `cfg.mode == CACAOSIM_CORO_CUSTOM_MFT`.
 *
 * Role:
 *   Cold-path precomputation of forward/inverse 1D Fourier basis matrices (`fwd_U_cplx`,
 *   `fwd_V_cplx`, `inv_U_cplx`, `inv_V_cplx`), complex Babinet focal perturbation
 *   `fpm_delta_cplx = T(u,v,lam) - 1`, and lossless LOWFS reflection weights
 *   `fpm_refl_w = sqrt(1 - |T(u,v,lam)|^2)` across all `ctx->eff_nlam` spectral channels.
 *
 * Rationale:
 *   1. Physical Chromatic Scaling: A physical focal mask has fixed metric pitch in microns, which
 *      corresponds to angular pitch `du0` in units of `lambda0 / D` at the reference wavelength.
 *      At channel wavelength `lam_c`, the diffraction scale `lambda_c / D` grows with `lam_c`, so
 *      the same physical mask pixel subtends `du_c = du0 * (lam0 / lam_c)` in `[lambda_c / D]`.
 *   2. Unitary MFT Normalization: With normalized pupil coordinate `xt = (x - p0) / dpix`
 *      (`dx_tilde = 1 / dpix`) and focal frequency `u = (kx - fx0) * du_c` (`du = du_c`), the
 *      round-trip 2D Riemann-sum measure per axis is `du_c / dpix`. Assigning
 *      `nrm = sqrt(du_c / dpix)` to each of the four 1D basis matrices makes `MFT^-1[MFT[Psi]]`
 *      exact and unitary without any hot-loop scalar rescaling.
 *   3. Sub-Pixel Edge Anti-Aliasing: Analytic `LYOT_SPOT` and `PHASE_DIMPLE` masks supersample
 *      each focal pixel on an `8 x 8` sub-grid so circular spot boundaries vary smoothly with
 *      mask radius rather than jumping in discrete pixel steps.
 */
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

    /* Focal grid spacing du0 [lambda0/D per focal pixel] at reference wavelength lam0 */
    float r_ld = (ctx->cfg.fpm_rad_ld > 0.05f) ? ctx->cfg.fpm_rad_ld : 3.0f;
    float du0  = 0.0f;
    if (ctx->cfg.mode == CACAOSIM_CORO_CUSTOM_MFT)
    {
        float ps0 = (ctx->cfg.fpm_pixscale_ld > 0.05f) ? ctx->cfg.fpm_pixscale_ld : 8.0f;
        du0       = 1.0f / ps0;
    }
    else
    {
        /* Enclose the circular spot diameter (2 * r_ld) with a 20% guard margin */
        float fov = (2.4f * r_ld > 2.0f) ? (2.4f * r_ld) : 2.0f;
        du0       = fov / (float) mx;
    }

    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        /* Chromatic focal pitch in [lambda_c / D] and symmetric 1D Fourier norm sqrt(du_c/dpix) */
        const float du_c = du0 * (lam0 / ctx->lam_um[c]);
        const float nrm  = sqrtf(du_c / dpix);

        float *fwd_U = &ctx->fwd_U_cplx[(size_t) c * my * nwf * 2u];
        float *inv_U = &ctx->inv_U_cplx[(size_t) c * nwf * my * 2u];
        float *fwd_V = &ctx->fwd_V_cplx[(size_t) c * nwf * mx * 2u];
        float *inv_V = &ctx->inv_V_cplx[(size_t) c * mx * nwf * 2u];

        /* Y-axis 1D Fourier kernel fwd_U[my x nwf] and conjugate transpose inv_U[nwf x my] */
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

        /* X-axis 1D Fourier kernel fwd_V[nwf x mx] and conjugate transpose inv_V[mx x nwf] */
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
            /* Custom MFT focal mask: Babinet delta = T - 1, LOWFS refl = sqrt(1 - |T|^2) */
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
            /*
             * Analytic Lyot spot or Roddier phase dimple: evaluate 8x8 sub-pixel fill factor w in
             * fixed physical focal coordinates [lambda0/D], with dispersive phase step lam0/lam_c.
             */
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

/**
 * coro_precompute_fft_mask() - Precompute half-bin pupil shift phasors and padded FFT focal masks
 * @ctx:   Coronagraph context with allocated `nfft x nfft` FFT buffers (`is_babinet == 0`).
 * @masks: Optional custom focal-plane mask cubes used when `cfg.mode == CACAOSIM_CORO_CUSTOM_FFT`.
 *
 * Role:
 *   Cold-path precomputation of 1D pupil half-bin shift vectors (`shift_1d_cplx`), per-channel
 *   pre-normalized complex FFT focal masks (`fft_mask_cplx`), and central-channel LOWFS reflection
 *   weights (`fft_refl_w`) for Optical Vortex, 4QPM, and custom full-plane FFT coronagraphs.
 *
 * Rationale:
 *   - Inter-Pixel Singularity Centering: Standard unshifted 2D FFTs place DC `(0, 0)` directly on
 *     array element `[0, 0]`, where `atan2(0, 0)` (Vortex) and quadrant sign transitions (4QPM)
 *     are ill-defined and break four-fold symmetry. Multiplying the padded pupil by
 *     `exp(-i * pi * m / nfft)` shifts the focal grid by `-0.5` bins so the optical axis sits
 *     cleanly at `(u, v) = (iu + 0.5, iv + 0.5)` midway between bins `-1` and `0`. For
 *     `CUSTOM_FFT`, each axis enables the half-bin shift only when the custom mask dimension is
 *     even (`(fx & 1) == 0`), preserving exact center-pixel alignment for odd-sized masks.
 *   - Zero-Cost FFT Normalization: Unnormalized forward + backward `nfft x nfft` FFTs scale field
 *     amplitudes by `nfft^2`. Folding `inv_nfft2 = 1 / nfft^2` into `fft_mask_cplx` and
 *     `fft_refl_w` eliminates an extra `O(nfft^2)` scaling pass on every hot-loop channel.
 */
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

    /* Apply half-bin focal shift except along odd-dimensioned custom FFT mask axes */
    int half_x = 1;
    int half_y = 1;
    if (ctx->cfg.mode == CACAOSIM_CORO_CUSTOM_FFT && (fa != NULL || fp != NULL))
    {
        half_x = ((fx & 1u) == 0u) ? 1 : 0;
        half_y = ((fy & 1u) == 0u) ? 1 : 0;
    }

    /* 1D linear pupil phase ramps exp(-i * pi * m / nfft) for X and Y half-bin focal centering */
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
            /* Wrap FFT bin index [0..nfft-1] to signed focal offset [-nfft/2 .. nfft/2 - 1] */
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
                    /* Embed custom [fx x fy] focal mask centered at DC; outside defaults to 1+0i */
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
                    /* Rouan 4QPM: alternating {+1, -1} (0 / pi rad) across focal quadrants */
                    mr = ((u > 0.0f) ? 1.0f : -1.0f) * ((v > 0.0f) ? 1.0f : -1.0f);
                    mi = 0.0f;
                }
                else
                {
                    /* Optical Vortex: helical phase ramp exp(i * l * atan2(v, u)) */
                    float th = (float) charge * atan2f(v, u);
                    mr       = cosf(th);
                    mi       = sinf(th);
                }

                /* Fold 1/nfft^2 round-trip inverse-FFT normalization into focal mask weights */
                fmsk[idx]      = inv_nfft2 * mr;
                fmsk[idx + 1u] = inv_nfft2 * mi;
                if (ctx->fft_refl_w != NULL)
                {
                    float t2 = mr * mr + mi * mi;
                    float rw = (t2 < 1.0f) ? sqrtf(fmaxf(0.0f, 1.0f - t2)) : 0.0f;
                    ctx->fft_refl_w[((size_t) c * ctx->npix_fft) + (size_t) jy * nfft + ix] =
                        inv_nfft2 * rw;
                }
            }
        }
    }
}

/**
 * cacaosim_coro_step_poly_cpu() - Propagate 1 or N_lambda channels through coronagraph on CPU
 * @ctx:        Initialized coronagraph context (`eff_nlam` post-Lyot channels produced).
 * @in_opd:     Host input pupil OPD `[um]` (`in_nlam` slices separated by `@in_stride`; NULL -> 0).
 * @in_amp:     Host input pupil amplitude (`in_nlam` slices separated by `@in_stride`; NULL -> 1).
 * @in_nlam:    Number of coherent wavelength channels in `@in_opd`/`@in_amp` (`1` = broadcast).
 * @in_stride:  Float element stride between consecutive input wavelength channels.
 * @out_opd:    Host output post-Lyot OPD `[um]` (`eff_nlam` slices separated by `@out_stride`).
 * @out_amp:    Host output post-Lyot amplitude (`eff_nlam` slices separated by `@out_stride`).
 * @out_stride: Float element stride between consecutive output wavelength channels.
 * @lowfs_opd:  Optional host 2D LOWFS pupil OPD `[um]` `[npix_wf]` at central channel `eff_nlam/2`.
 * @lowfs_amp:  Optional host 2D LOWFS pupil amplitude `[npix_wf]`.
 *
 * Role:
 *   Hot-path CPU coronagraphic propagation kernel executing entrance pupil complex field synthesis,
 *   pre-FPM apodization, focal-plane mask diffraction (compact Babinet MFT or half-bin shifted
 *   padded 2D FFT), LOWFS pickoff (`fpm_refl` or `lyot_refl`), and post-FPM Lyot stop filtering
 *   across all `ctx->eff_nlam` spectral channels without a single heap allocation.
 *
 * Rationale:
 *   - Separates propagation into three specialized fast paths (`skip_fpm`, `is_babinet` OpenBLAS
 *     `cblas_cgemm`, and `!is_babinet` FFTW3f `fftwf_execute`) so each coronagraph architecture
 *     uses the minimum possible floating-point work.
 *   - Evaluates the `out_lowfs` beam at the central channel `c_mid = eff_nlam / 2` to supply a
 *     representative 2D `[OPD_um, Amplitude]` wavefront stream to downstream LOWFS units (`zwfs`,
 *     `shwfs`, or `pyrwfs`) without polychromatic phase cancellation.
 *   - Converts complex post-Lyot phasors `lr + i*li` back into `[OPD_um, Amplitude]` using
 *     `atan2f(li, lr) * (lam_c / 2*pi)` guarded by `amp > 1e-12f` to zero out dark regions cleanly.
 */
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
    const int      lo_lyot  = (ctx->cfg.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? 1 : 0;
    const float    refl     = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;
    const float    c_one[2]  = { 1.0f, 0.0f };
    const float    c_zero[2] = { 0.0f, 0.0f };

    const float *__restrict lp_msk  = MILK_ASSUME_ALIGNED(ctx->lowfs_pup_mask);
    float *__restrict       ep      = MILK_ASSUME_ALIGNED(ctx->cpu_epup_cplx);
    float *__restrict       pre_int = MILK_ASSUME_ALIGNED(ctx->cpu_pre_lyot_int);
    const float             w_c     = 1.0f / (float) ((nlam >= 1) ? nlam : 1);
    double                  step_e_in   = 0.0;
    double                  step_e_pre  = 0.0;
    double                  step_e_post = 0.0;
    double                  step_e_lo   = 0.0;

    if (pre_int != NULL)
    {
        __builtin_memset(pre_int, 0, npix_wf * sizeof(float));
    }

    for (int32_t c = 0; c < nlam; c++)
    {
        double ch_e_in   = 0.0;
        double ch_e_pre  = 0.0;
        double ch_e_post = 0.0;
        double ch_e_lo   = 0.0;

        /* Broadcast slice 0 when monochromatic input (in_nlam == 1) feeds a polychromatic coro */
        uint32_t ic = (in_nlam > 1u)
                          ? (((uint32_t) c < in_nlam) ? (uint32_t) c : (in_nlam - 1u))
                          : 0u;
        const float *opd_c     = (in_opd != NULL) ? (in_opd + (uint64_t) ic * in_stride) : NULL;
        const float *amp_c     = (in_amp != NULL) ? (in_amp + (uint64_t) ic * in_stride) : NULL;
        float       *out_opd_c = out_opd + (uint64_t) c * out_stride;
        float       *out_amp_c = out_amp + (uint64_t) c * out_stride;
        const int    do_lo     = (lowfs_opd != NULL && lowfs_amp != NULL) ? 1 : 0;
        float       *lo_opd_c  = do_lo ? (lowfs_opd + (uint64_t) c * out_stride) : NULL;
        float       *lo_amp_c  = do_lo ? (lowfs_amp + (uint64_t) c * out_stride) : NULL;

        const float  lam_c      = ctx->lam_um[c];
        const float  opd_to_rad = CACAOSIM_TWO_PI_F / lam_c;
        const float  rad_to_opd = lam_c / CACAOSIM_TWO_PI_F;
        const float *pm_c       = ctx->has_pup_mask
                                      ? &ctx->pup_mask_cplx[(size_t) c * npix_wf * 2u]
                                      : NULL;
        const float *ly_c       = &ctx->lyot_cplx[(size_t) c * npix_wf * 2u];

        /* Build entrance complex pupil field A * exp(i*2*pi*OPD/lam_c) * pup_mask_cplx */
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
            ch_e_in += (double) pr * pr + (double) pi * pi;
        }

        if (ctx->skip_fpm)
        {
            /* Fast path 1: pure pupil/Lyot apodizer with no focal-plane mask */
            for (uint64_t p = 0; p < npix_wf; p++)
            {
                float  ur   = ep[2u * p];
                float  ui   = ep[2u * p + 1u];
                double u2   = (double) ur * ur + (double) ui * ui;
                float  lyr  = ly_c[2u * p];
                float  lyi  = ly_c[2u * p + 1u];
                float  lr   = ur * lyr - ui * lyi;
                float  li   = ur * lyi + ui * lyr;
                float  amp  = hypotf(lr, li);
                out_amp_c[p] = amp;
                out_opd_c[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;
                if (pre_int != NULL)
                {
                    pre_int[p] += w_c * (float) u2;
                }
                ch_e_pre += u2;
                ch_e_post += (double) amp * amp;

                if (do_lo)
                {
                    if (lo_lyot)
                    {
                        /* Reflective Lyot stop: reflect light blocked where |Lyot|^2 < 1 */
                        float lya2   = lyr * lyr + lyi * lyi;
                        float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
                        float la     = r_ly * hypotf(ur, ui);
                        lo_amp_c[p]  = la;
                        lo_opd_c[p]  = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                        ch_e_lo += (double) la * la;
                    }
                    else
                    {
                        lo_amp_c[p] = 0.0f;
                        lo_opd_c[p] = 0.0f;
                    }
                }
            }
        }
        else if (ctx->is_babinet)
        {
            /* Fast path 2: Soummer Babinet Matrix-DFT on compact [my x mx] focal patch */
            const uint32_t mx    = ctx->nx_mft;
            const uint32_t my    = ctx->ny_mft;
            const float   *fwd_U = &ctx->fwd_U_cplx[(size_t) c * my * nwf * 2u];
            const float   *inv_U = &ctx->inv_U_cplx[(size_t) c * nwf * my * 2u];
            const float   *fwd_V = &ctx->fwd_V_cplx[(size_t) c * nwf * mx * 2u];
            const float   *inv_V = &ctx->inv_V_cplx[(size_t) c * mx * nwf * 2u];
            float *__restrict ec = MILK_ASSUME_ALIGNED(ctx->cpu_ecore_cplx);
            float *__restrict ef = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);

            /* Forward 2D MFT: tmp[my x nwf] = fwd_U * ep; ef[my x mx] = tmp * fwd_V */
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) my, (int) nwf, (int) nwf,
                        c_one, fwd_U, (int) nwf, ep, (int) nwf, c_zero, ctx->cpu_tmp_cplx,
                        (int) nwf);
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) my, (int) mx, (int) nwf,
                        c_one, ctx->cpu_tmp_cplx, (int) nwf, fwd_V, (int) mx, c_zero, ef,
                        (int) mx);

            /* fpm_refl LOWFS pickoff: inverse-MFT reflected core field sqrt(1 - |T|^2) * Efoc */
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
                    float cr    = ec[2u * p];
                    float ci    = ec[2u * p + 1u];
                    float pm    = (lp_msk != NULL) ? lp_msk[p] : 1.0f;
                    float la    = refl * pm * hypotf(cr, ci);
                    lo_amp_c[p] = la;
                    lo_opd_c[p] = (la > 1e-12f) ? (atan2f(ci, cr) * rad_to_opd) : 0.0f;
                    ch_e_lo += (double) la * la;
                }
            }

            /* Multiply focal field by Babinet core perturbation Delta_M = T(u,v,lam) - 1 */
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

            /* Inverse 2D MFT back to Lyot pupil: ec[nwf x nwf] = inv_U * ef * inv_V */
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) mx, (int) my,
                        c_one, inv_U, (int) my, ef, (int) mx, c_zero, ctx->cpu_tmp_cplx, (int) mx);
            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) nwf, (int) nwf, (int) mx,
                        c_one, ctx->cpu_tmp_cplx, (int) mx, inv_V, (int) nwf, c_zero, ec,
                        (int) nwf);

            /* Babinet superposition U_Lyot_in = ep + ec, followed by complex Lyot stop L_c */
            for (uint64_t p = 0; p < npix_wf; p++)
            {
                float  ur   = ep[2u * p] + ec[2u * p];
                float  ui   = ep[2u * p + 1u] + ec[2u * p + 1u];
                double u2   = (double) ur * ur + (double) ui * ui;
                float  lyr  = ly_c[2u * p];
                float  lyi  = ly_c[2u * p + 1u];
                float  lr   = ur * lyr - ui * lyi;
                float  li   = ur * lyi + ui * lyr;
                float  amp  = hypotf(lr, li);
                out_amp_c[p] = amp;
                out_opd_c[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;
                if (pre_int != NULL)
                {
                    pre_int[p] += w_c * (float) u2;
                }
                ch_e_pre += u2;
                ch_e_post += (double) amp * amp;

                if (do_lo && lo_lyot)
                {
                    float lya2  = lyr * lyr + lyi * lyi;
                    float r_ly  = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
                    float la    = r_ly * hypotf(ur, ui);
                    lo_amp_c[p] = la;
                    lo_opd_c[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                    ch_e_lo += (double) la * la;
                }
            }
        }
        else
        {
            /* Fast path 3: Zero-padded [nfft x nfft] 2D FFT with half-bin pupil shift */
            const uint32_t          nfft    = ctx->nfft;
            const uint32_t          pad_off = (nfft - nwf) / 2u;
            float *__restrict       fbuf    = MILK_ASSUME_ALIGNED(ctx->cpu_fft_buf);
            const float *__restrict sh_x    = &ctx->shift_1d_cplx[0];
            const float *__restrict sh_y    = &ctx->shift_1d_cplx[(size_t) nfft * 2u];
            const float *__restrict fmsk    =
                MILK_ASSUME_ALIGNED(&ctx->fft_mask_cplx[(size_t) c * ctx->npix_fft * 2u]);

            __builtin_memset(fbuf, 0, ctx->npix_fft * 2u * sizeof(float));

            /* Center-embed [nwf x nwf] pupil and apply forward half-bin shift ramp sh_x * sh_y */
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

            /* Optional fpm_refl LOWFS branch: inverse-FFT reflected focal amplitude to pupil */
            if (do_lo && !lo_lyot && ctx->cpu_fft_lo_buf != NULL && ctx->fft_refl_w != NULL)
            {
                float *__restrict       flob = MILK_ASSUME_ALIGNED(ctx->cpu_fft_lo_buf);
                const float *__restrict frw =
                    MILK_ASSUME_ALIGNED(&ctx->fft_refl_w[(size_t) c * ctx->npix_fft]);
#pragma omp simd
                for (uint64_t i = 0; i < ctx->npix_fft; i++)
                {
                    float w          = frw[i];
                    flob[2u * i]     = fbuf[2u * i] * w;
                    flob[2u * i + 1] = fbuf[2u * i + 1u] * w;
                }
                fftwf_execute((fftwf_plan) ctx->fftw_lo_inv);

                /* Undo half-bin shift via conj(sh_x) * conj(sh_y) and clip by lowfs_pup_mask */
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
                        lo_amp_c[p]    = la;
                        lo_opd_c[p]    = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                        ch_e_lo += (double) la * la;
                    }
                }
            }

            /* Multiply focal spectrum by pre-normalized (1/nfft^2) complex focal mask */
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

            /* Crop [nwf x nwf] Lyot pupil, undo half-bin phase ramp, and apply Lyot stop */
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
                    double   u2    = (double) ur * ur + (double) ui * ui;

                    float lyr    = ly_c[2u * p];
                    float lyi    = ly_c[2u * p + 1u];
                    float lr     = ur * lyr - ui * lyi;
                    float li     = ur * lyi + ui * lyr;
                    float amp    = hypotf(lr, li);
                    out_amp_c[p] = amp;
                    out_opd_c[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;
                    if (pre_int != NULL)
                    {
                        pre_int[p] += w_c * (float) u2;
                    }
                    ch_e_pre += u2;
                    ch_e_post += (double) amp * amp;

                    if (do_lo && lo_lyot)
                    {
                        float lya2  = lyr * lyr + lyi * lyi;
                        float r_ly  = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
                        float la    = r_ly * hypotf(ur, ui);
                        lo_amp_c[p] = la;
                        lo_opd_c[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
                        ch_e_lo += (double) la * la;
                    }
                }
            }
        }

        step_e_in += (double) w_c * ch_e_in;
        step_e_pre += (double) w_c * ch_e_pre;
        step_e_post += (double) w_c * ch_e_post;
        step_e_lo += (double) w_c * ch_e_lo;
    }

    if (ctx->cpu_pre_lyot_long != NULL && pre_int != NULL)
    {
        float *__restrict pre_long = MILK_ASSUME_ALIGNED(ctx->cpu_pre_lyot_long);
#pragma omp simd
        for (uint64_t p = 0; p < npix_wf; p++)
        {
            pre_long[p] += pre_int[p];
        }
    }
    ctx->long_cnt++;
    ctx->sum_e_in += step_e_in;
    ctx->sum_e_pre_lyot += step_e_pre;
    ctx->sum_e_post_lyot += step_e_post;
    ctx->sum_e_lowfs += step_e_lo;

    double den_i              = (step_e_in > 1e-30) ? step_e_in : 1.0;
    double den_l              = (ctx->sum_e_in > 1e-30) ? ctx->sum_e_in : 1.0;
    ctx->last_frac_pre_lyot   = (float) (step_e_pre / den_i);
    ctx->last_frac_lyot_trans = (float) (step_e_post / den_i);
    ctx->last_frac_lowfs      = (float) (step_e_lo / den_i);
    ctx->long_frac_pre_lyot   = (float) (ctx->sum_e_pre_lyot / den_l);
    ctx->long_frac_lyot_trans = (float) (ctx->sum_e_post_lyot / den_l);
    ctx->long_frac_lowfs      = (float) (ctx->sum_e_lowfs / den_l);
}

/**
 * cacaosim_coro_step_cpu() - Propagate single-plane [in_opd, in_amp] -> [out_opd, out_amp] on CPU
 * @ctx:       Initialized coronagraph context.
 * @in_opd:    Host 2D input pupil OPD `[um]` `[npix_wf]` (NULL -> 0).
 * @in_amp:    Host 2D input pupil amplitude `[npix_wf]` (NULL -> 1).
 * @out_opd:   Host 2D output post-Lyot OPD `[um]` `[npix_wf]`.
 * @out_amp:   Host 2D output post-Lyot amplitude `[npix_wf]`.
 * @lowfs_opd: Optional host 2D output LOWFS pupil OPD `[um]` `[npix_wf]`.
 * @lowfs_amp: Optional host 2D output LOWFS pupil amplitude `[npix_wf]`.
 *
 * Role:
 *   Hot-path single-plane CPU wrapper around `cacaosim_coro_step_poly_cpu()`.
 *
 * Rationale:
 *   When `@ctx` holds multiple spectral channels (`eff_nlam > 1`), writes into internal scratch
 *   cubes `cpu_poly_opd`/`cpu_poly_amp` (and `cpu_poly_lo_opd`/`cpu_poly_lo_amp` if LOWFS is
 *   requested) and copies out the central channel `eff_nlam / 2` so single-slice `[npix_wf]`
 *   output buffers are never overrun.
 */
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
        float *lo_o = (lowfs_opd != NULL && lowfs_amp != NULL) ? ctx->cpu_poly_lo_opd : NULL;
        float *lo_a = (lowfs_opd != NULL && lowfs_amp != NULL) ? ctx->cpu_poly_lo_amp : NULL;
        cacaosim_coro_step_poly_cpu(ctx, in_opd, in_amp, 1u, ctx->npix_wf, ctx->cpu_poly_opd,
                                    ctx->cpu_poly_amp, ctx->npix_wf, lo_o, lo_a);
        size_t off = (size_t) (ctx->eff_nlam / 2) * ctx->npix_wf;
        __builtin_memcpy(out_opd, ctx->cpu_poly_opd + off, ctx->npix_wf * sizeof(float));
        __builtin_memcpy(out_amp, ctx->cpu_poly_amp + off, ctx->npix_wf * sizeof(float));
        if (lo_o != NULL && lo_a != NULL)
        {
            __builtin_memcpy(lowfs_opd, lo_o + off, ctx->npix_wf * sizeof(float));
            __builtin_memcpy(lowfs_amp, lo_a + off, ctx->npix_wf * sizeof(float));
        }
        return;
    }
    cacaosim_coro_step_poly_cpu(ctx, in_opd, in_amp, 1u, ctx->npix_wf, out_opd, out_amp,
                                ctx->npix_wf, lowfs_opd, lowfs_amp);
}

/**
 * cacaosim_coro_init_masks() - Initialize coronagraph with custom 2D/3D pupil, FPM, & Lyot masks
 * @ctx:        Output coronagraph context to initialize (non-NULL).
 * @cfg:        Coronagraph optical and numerical configuration (non-NULL, `cfg->nwf >= 8`).
 * @masks:      Optional custom amplitude/phase mask cubes across pupil, focal, and Lyot planes.
 * @ext_stream: Optional external `cudaStream_t` handle for fused GPU bench execution.
 *
 * Role:
 *   Cold-path master setup routine for `cacaosim.coro`. Validates mask dimensions and spectral
 *   cube depths, resolves channel wavelengths from FITS/SHM WCS metadata or uniform `bw_frac`
 *   binning, synthesizes fallback anti-aliased Lyot and LOWFS relay stops, builds complex
 *   propagator tables (`coro_precompute_babinet()` or `coro_precompute_fft_mask()` + FFTW3f
 *   plans), and initializes the optional CUDA backend (`cacaosim_coro_cuda_init()`).
 *
 * Rationale:
 *   - Automatic Mode Promotion: If a caller supplies a custom focal mask (`fpm_amp` or `fpm_pha`)
 *     while leaving `cfg->mode` on an analytic default (`0..3`), `eff_mode` automatically promotes
 *     compact modes (`LYOT_SPOT`, `PHASE_DIMPLE`) to `CUSTOM_MFT` and extended modes (`VORTEX`,
 *     `4QPM`) to `CUSTOM_FFT`. Conversely, `CUSTOM_MFT` without an FPM sets `skip_fpm = 1` for
 *     zero-FFT Shaped-Pupil / vAPP operation.
 *   - Geometric Lyot Stop Scaling: User parameters `lyot_obs` and `lyot_spider_w` are expressed as
 *     fractions of the full entrance pupil diameter `pup_diam`, whereas `cacaosim_make_pupil()`
 *     expects ratios relative to the stop's own outer diameter `pup_diam * outer`; dividing both
 *     ratios by `outer` preserves exact physical obscuration and spider widths.
 *
 * Return: 0 on success, -1 on dimension mismatch, conflicting 3D slice counts, or OOM.
 */
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

    /* Enforce pupil/Lyot [nwf x nwf] grid match and consistent [fx x fy] FPM amp/pha sizes */
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

    /* Verify all 3D (>1 slice) mask cubes agree on nz and locate any WCS wavelength table */
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

    /* Auto-promote analytic mode to CUSTOM_MFT / CUSTOM_FFT when custom FPM cubes are provided */
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

    /* Resolve effective polychromatic channel count from 3D masks or bandpass config */
    int32_t nlam = 1;
    if (mask_nz > 1u)
    {
        if (cfg->nlambda > 1 && (uint32_t) cfg->nlambda != mask_nz)
        {
            return -1;
        }
        nlam = (int32_t) mask_nz;
    }
    else
    {
        nlam = cacaosim_resolve_nlambda(cfg->nlambda, 1u, cfg->bw_frac);
    }
    ctx->eff_nlam = nlam;

    /* Populate per-channel wavelength table [um] from mask WCS keywords or midpoint-rule bins */
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

    /* Pin OpenBLAS to 1 thread to prevent internal pthread contention on small MFT matrices */
    if (openblas_set_num_threads != NULL)
    {
        openblas_set_num_threads(1);
    }

    ctx->lyot_mask         = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    ctx->lowfs_pup_mask    = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    ctx->lyot_cplx         = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf * 2u,
                                                               sizeof(float));
    ctx->cpu_epup_cplx     = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
    ctx->cpu_pre_lyot_int  = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    ctx->cpu_pre_lyot_long = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
    if (!ctx->lyot_mask || !ctx->lowfs_pup_mask || !ctx->lyot_cplx || !ctx->cpu_epup_cplx ||
        !ctx->cpu_pre_lyot_int || !ctx->cpu_pre_lyot_long)
    {
        cacaosim_coro_free(ctx);
        return -1;
    }

    /* Synthesize clear circular relay pupil stop on the FPM->LOWFS reflective arm */
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
        ctx->cpu_poly_opd    = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf,
                                                                 sizeof(float));
        ctx->cpu_poly_amp    = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf,
                                                                 sizeof(float));
        ctx->cpu_poly_lo_opd = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf,
                                                                 sizeof(float));
        ctx->cpu_poly_lo_amp = (float *) cacaosim_aligned_calloc((size_t) nlam * ctx->npix_wf,
                                                                 sizeof(float));
        if (!ctx->cpu_poly_opd || !ctx->cpu_poly_amp || !ctx->cpu_poly_lo_opd ||
            !ctx->cpu_poly_lo_amp)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
    }

    /* Pre-evaluate optional custom pre-FPM entrance pupil complex apodizer cube */
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

    /* Build Lyot stop amplitude (custom 2D/3D cube or anti-aliased geometric pupil) + phase */
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
            .nwf           = cfg->nwf,
            .pup_diam      = pdim * outer,
            .cobs          = (outer > 1e-4f) ? (obs_p / outer) : 0.0f,
            .nspider       = cfg->lyot_nspider,
            .spider_w      = (outer > 1e-4f) ? (cfg->lyot_spider_w / outer) : 0.0f,
            .spider_deg    = cfg->lyot_spider_deg,
            .spider_offset = (outer > 1e-4f) ? (cfg->lyot_spider_offset / outer) : 0.0f,
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
        ctx->fft_refl_w     = (float *) cacaosim_aligned_calloc(
            (size_t) nlam * ctx->npix_fft, sizeof(float));
        ctx->cpu_fft_buf    = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        ctx->cpu_fft_lo_buf = (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));

        if (!ctx->shift_1d_cplx || !ctx->fft_mask_cplx || !ctx->fft_refl_w || !ctx->cpu_fft_buf ||
            !ctx->cpu_fft_lo_buf)
        {
            cacaosim_coro_free(ctx);
            return -1;
        }
        coro_precompute_fft_mask(ctx, masks);

        /* FFTW_ESTIMATE avoids multi-second planner tuning while keeping deterministic buffers */
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

/**
 * cacaosim_coro_init() - Initialize coronagraphic focal mask, Lyot stop, and CPU/CUDA plans
 * @ctx:        Output coronagraph context (non-NULL).
 * @cfg:        Coronagraph optical configuration (non-NULL).
 * @ext_lyot:   Optional external 2D Lyot stop amplitude map `[nwf * nwf]` (NULL synthesizes from
 *              `@cfg`).
 * @ext_stream: Optional external `cudaStream_t` handle for fused GPU pipelines.
 *
 * Role:
 *   Cold-path convenience wrapper around `cacaosim_coro_init_masks()` for analytic focal masks.
 *
 * Rationale:
 *   Wraps `@ext_lyot` in a non-owning 2D `cacaosim_mask_cube_t` view so callers without custom
 *   3D FITS/SHM cubes can initialize `cacaosim.coro` in a single call.
 *
 * Return: 0 on success, -1 on invalid dimensions or allocation failure.
 */
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

/**
 * cacaosim_coro_free() - Release all host and device coronagraph buffers
 * @ctx: Coronagraph context to clean up (NULL-safe).
 *
 * Role:
 *   Cold-path destructor tearing down CUDA device state, FFTW3f plans, and all aligned host
 *   propagator/scratch arrays.
 *
 * Rationale:
 *   Zeros `@ctx` after freeing every pointer so repeated `cacaosim_coro_free()` calls or error
 *   cleanups inside `cacaosim_coro_init_masks()` cannot double-free memory.
 */
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
    free(ctx->cpu_poly_lo_opd);
    free(ctx->cpu_poly_lo_amp);
    free(ctx->cpu_pre_lyot_int);
    free(ctx->cpu_pre_lyot_long);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

/**
 * cacaosim_coro_step_poly() - Propagate 1 or N_lambda channels on active backend (CPU or CUDA)
 * @ctx:         Active coronagraph context.
 * @in_opd:      Host input pupil OPD `[um]` (used when `@d_in_opd == NULL`).
 * @in_amp:      Host input pupil amplitude (used when `@d_in_opd == NULL`).
 * @in_nlam:     Number of input wavelength channels (`1` = monochromatic broadcast).
 * @in_stride:   Float element stride between input channels.
 * @out_opd:     Host output post-Lyot OPD `[um]` (`eff_nlam` slices separated by `@out_stride`).
 * @out_amp:     Host output post-Lyot amplitude (`eff_nlam` slices separated by `@out_stride`).
 * @out_stride:  Float element stride between output channels.
 * @lowfs_opd:   Optional host 2D LOWFS OPD `[um]` `[npix_wf]`.
 * @lowfs_amp:   Optional host 2D LOWFS amplitude `[npix_wf]`.
 * @d_in_opd:    Optional resident device input OPD.
 * @d_in_amp:    Optional resident device input amplitude.
 * @d_out_opd:   Optional resident device output post-Lyot OPD.
 * @d_out_amp:   Optional resident device output post-Lyot amplitude.
 * @d_lowfs_opd: Optional resident device 2D LOWFS OPD.
 * @d_lowfs_amp: Optional resident device 2D LOWFS amplitude.
 *
 * Role:
 *   Hot-path backend router dispatching polychromatic coronagraph propagation to CUDA or CPU.
 *
 * Rationale:
 *   Keeps FPS and fused-bench callers agnostic to whether CUDA was compiled/selected at runtime.
 */
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

/**
 * cacaosim_coro_step_beam() - Propagate single-plane beam on active backend (host or device)
 * @ctx:         Active coronagraph context.
 * @in_opd:      Host 2D input pupil OPD `[um]` `[npix_wf]`.
 * @in_amp:      Host 2D input pupil amplitude `[npix_wf]`.
 * @out_opd:     Host 2D output post-Lyot OPD `[um]` `[npix_wf]`.
 * @out_amp:     Host 2D output post-Lyot amplitude `[npix_wf]`.
 * @lowfs_opd:   Optional host 2D LOWFS OPD `[um]` `[npix_wf]`.
 * @lowfs_amp:   Optional host 2D LOWFS amplitude `[npix_wf]`.
 * @d_in_opd:    Optional device 2D input OPD `[npix_wf]`.
 * @d_in_amp:    Optional device 2D input amplitude `[npix_wf]`.
 * @d_out_opd:   Optional device output post-Lyot OPD.
 * @d_out_amp:   Optional device output post-Lyot amplitude.
 * @d_lowfs_opd: Optional device 2D LOWFS OPD `[npix_wf]`.
 * @d_lowfs_amp: Optional device 2D LOWFS amplitude `[npix_wf]`.
 *
 * Role:
 *   Hot-path single-beam wrapper routing monochromatic `[npix_wf]` beams through CPU or CUDA.
 *
 * Rationale:
 *   When `ctx->eff_nlam > 1` and output is requested on host (`d_out_opd == NULL`), stages all
 *   `eff_nlam` slices into `cpu_poly_opd`/`cpu_poly_amp` and extracts the central slice
 *   `eff_nlam / 2` so callers passing 2D `[npix_wf]` host buffers remain memory-safe on both CPU
 *   and GPU backends.
 */
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
            float *lo_o = (lowfs_opd != NULL && lowfs_amp != NULL && d_lowfs_opd == NULL)
                              ? ctx->cpu_poly_lo_opd
                              : lowfs_opd;
            float *lo_a = (lowfs_opd != NULL && lowfs_amp != NULL && d_lowfs_opd == NULL)
                              ? ctx->cpu_poly_lo_amp
                              : lowfs_amp;
            cacaosim_coro_cuda_step_poly(ctx, in_opd, in_amp, 1u, ctx->npix_wf, ctx->cpu_poly_opd,
                                         ctx->cpu_poly_amp, ctx->npix_wf, lo_o, lo_a, d_in_opd,
                                         d_in_amp, NULL, NULL, d_lowfs_opd, d_lowfs_amp);
            size_t off = (size_t) (ctx->eff_nlam / 2) * ctx->npix_wf;
            __builtin_memcpy(out_opd, ctx->cpu_poly_opd + off, ctx->npix_wf * sizeof(float));
            __builtin_memcpy(out_amp, ctx->cpu_poly_amp + off, ctx->npix_wf * sizeof(float));
            if (lowfs_opd != NULL && lowfs_amp != NULL && d_lowfs_opd == NULL && lo_o != NULL &&
                lo_a != NULL)
            {
                __builtin_memcpy(lowfs_opd, lo_o + off, ctx->npix_wf * sizeof(float));
                __builtin_memcpy(lowfs_amp, lo_a + off, ctx->npix_wf * sizeof(float));
            }
            return;
        }
        cacaosim_coro_step_cpu(ctx, in_opd, in_amp, out_opd, out_amp, lowfs_opd, lowfs_amp);
        return;
    }

    cacaosim_coro_step_poly(ctx, in_opd, in_amp, 1u, ctx->npix_wf, out_opd, out_amp, ctx->npix_wf,
                            lowfs_opd, lowfs_amp, d_in_opd, d_in_amp, d_out_opd, d_out_amp,
                            d_lowfs_opd, d_lowfs_amp);
}

/**
 * cacaosim_coro_reset_long() - Clear cumulative pre-Lyot intensity and LOWFS power accumulators
 */
void cacaosim_coro_reset_long(cacaosim_coro_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->cpu_pre_lyot_long != NULL)
    {
        __builtin_memset(ctx->cpu_pre_lyot_long, 0, ctx->npix_wf * sizeof(float));
    }
    if (ctx->use_cuda)
    {
        cacaosim_coro_cuda_reset_long(ctx);
    }
    ctx->long_cnt             = 0ULL;
    ctx->sum_e_in             = 0.0;
    ctx->sum_e_pre_lyot       = 0.0;
    ctx->sum_e_post_lyot      = 0.0;
    ctx->sum_e_lowfs          = 0.0;
    ctx->long_frac_pre_lyot   = 0.0f;
    ctx->long_frac_lyot_trans = 0.0f;
    ctx->long_frac_lowfs      = 0.0f;
}

/**
 * cacaosim_coro_readout_telemetry_async() - Stage pre-Lyot intensity maps and energy fractions
 */
void MILK_HOT cacaosim_coro_readout_telemetry_async(
    cacaosim_coro_ctx_t *ctx,
    float               *out_pre_lyot,
    float               *out_pre_lyot_long,
    int                  sync_stream)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda)
    {
        cacaosim_coro_cuda_readout_telemetry(ctx, out_pre_lyot, out_pre_lyot_long, sync_stream);
        return;
    }
    if (out_pre_lyot != NULL && ctx->cpu_pre_lyot_int != NULL)
    {
        __builtin_memcpy(out_pre_lyot, ctx->cpu_pre_lyot_int, ctx->npix_wf * sizeof(float));
    }
    if (out_pre_lyot_long != NULL && ctx->cpu_pre_lyot_long != NULL)
    {
        const float inv = (ctx->long_cnt > 0ULL) ? (1.0f / (float) ctx->long_cnt) : 0.0f;
        const float *__restrict src = MILK_ASSUME_ALIGNED(ctx->cpu_pre_lyot_long);
        for (uint64_t p = 0; p < ctx->npix_wf; p++)
        {
            out_pre_lyot_long[p] = src[p] * inv;
        }
    }
}

/**
 * cacaosim_coro_readout_telemetry_finish() - Finalize host power-budget fractions after GPU sync
 */
void MILK_HOT cacaosim_coro_readout_telemetry_finish(cacaosim_coro_ctx_t *ctx)
{
    if (ctx != NULL && ctx->use_cuda)
    {
        cacaosim_coro_cuda_readout_finish(ctx);
    }
}
