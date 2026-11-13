// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pupil.c
 * @brief   4x4 supersampled anti-aliased pupil amplitude, Zernike NCPA, and PSD WFE synthesis
 *
 * Architectural Role:
 *   Implements the cold-path CPU synthesis kernels declared in `cacaosim_pupil.h`
 *   for telescope aperture masks (`cacaosim_make_pupil`), static Zernike NCPA maps
 *   in microns (`cacaosim_make_ncpa`), and combined low-order Zernike plus
 *   power-law polishing-ripple static WFE maps in SI meters (`cacaosim_make_wfe_m`).
 *
 * Design Rationale & Motivation:
 *   - Stratified 4x4 sub-pixel anti-aliasing: Sampling 16 sub-pixel points per
 *     grid cell converts discontinuous circular/linear aperture edges into smooth
 *     fractional transmission weights (`0/16 .. 16/16`), eliminating high-frequency
 *     stair-step diffraction artifacts in downstream coronagraphs and WFSs.
 *   - Reproducible platform-independent PRNG: All stochastic modal coefficients use
 *     explicitly seeded stack-local `cacaosim_rng_t` (`xoshiro128++`) states rather
 *     than libc `rand()`, guaranteeing bit-identical NCPA and WFE maps across OSes,
 *     compilers, and thread counts.
 *   - Mixed-precision statistics: Wavefront maps are evaluated in fast `float`
 *     arithmetic, while area, weighted piston mean, and weighted variance reductions
 *     use `double` accumulators so large grids (`512x512` and above) suffer zero
 *     catastrophic cancellation before final RMS rescaling.
 */

#include <math.h>
#include <omp.h>
#include <stddef.h>

#include "milk_config.h"
#include "milk_types.h"
#include "cacaosim_types.h"
#include "common/cacaosim_rng.h"
#include "pupil/cacaosim_pupil.h"

/**
 * cacaosim_make_pupil() - Generate 4x4 sub-pixel anti-aliased 2D pupil amplitude array
 * @cfg:     Pointer to pupil geometry configuration (`nwf > 0`)
 * @pup_out: Destination row-major `float` amplitude buffer `[cfg->nwf * cfg->nwf]`
 *
 * Role:
 *   Synthesizes the static 2D pupil amplitude mask for `cacaosim.mkpupil`,
 *   `cacaosim.mkbeam`, `cacaosim.coro`, and `cacaosim.htsim`, returning the
 *   integrated clear aperture area in pixel units for flux/photometry diagnostics.
 *
 * Rationale:
 *   Precomputes squared inner/outer radii and spider vane unit vectors outside the
 *   OpenMP pixel loop so the `16 * nwf^2` inner sub-pixel loop executes purely
 *   algebraic dot/cross-product tests without `sqrtf()` or trigonometric calls.
 *   Spider vanes are modeled as outward half-rays (`along >= 0`) from the pupil
 *   center so odd vane counts (e.g., 3-vane tripods) do not spuriously mirror
 *   across the origin.
 *
 * Return: Integrated pupil amplitude sum (effective clear area in pixels), or `0.0` on error.
 */
double cacaosim_make_pupil(
    const cacaosim_pupil_cfg_t *cfg,
    float                      *pup_out)
{
    if (cfg == NULL || pup_out == NULL || cfg->nwf == 0)
    {
        return 0.0;
    }

    /* Sanitize aperture bounds and convert normalized fractions into pixel radii */
    const uint32_t nwf    = cfg->nwf;
    const float    diam   = (cfg->pup_diam > 0.01f && cfg->pup_diam <= 1.0f) ? cfg->pup_diam : 0.9f;
    const float    r_out  = 0.5f * diam * (float) nwf;
    const float    r_out2 = r_out * r_out;
    const float    cobs   = (cfg->cobs >= 0.0f && cfg->cobs < 0.98f) ? cfg->cobs : 0.0f;
    const float    r_in2  = (cobs * r_out) * (cobs * r_out);
    /* Grid center sits at half-pixel midpoint for even nwf and center pixel for odd nwf */
    const float    ctr    = 0.5f * (float) (nwf - 1u);
    const int32_t  nsp    = (cfg->nspider > 0 && cfg->nspider <= 16) ? cfg->nspider : 0;
    const float    half_w = 0.5f * cfg->spider_w * (2.0f * r_out);

    /* Precompute outward unit direction vectors (cos, sin) for each radial spider vane */
    float sp_cos[16];
    float sp_sin[16];
    for (int32_t k = 0; k < nsp; k++)
    {
        float ang = cfg->spider_deg * (CACAOSIM_PI_F / 180.0f) +
                    (CACAOSIM_TWO_PI_F * (float) k) / (float) nsp;
        sp_cos[k] = cosf(ang);
        sp_sin[k] = sinf(ang);
    }

    double sum_amp = 0.0;

    /* Evaluate 4x4 stratified sub-pixel rays per pixel at offsets {-3/8, -1/8, +1/8, +3/8} px */
#pragma omp parallel for reduction(+ : sum_amp) schedule(static)
    for (uint32_t jy = 0; jy < nwf; jy++)
    {
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            int32_t hits = 0;
            for (int sy = 0; sy < 4; sy++)
            {
                float dy = ((float) jy - ctr) + ((float) sy - 1.5f) * 0.25f;
                for (int sx = 0; sx < 4; sx++)
                {
                    float dx = ((float) ix - ctr) + ((float) sx - 1.5f) * 0.25f;
                    float r2 = dx * dx + dy * dy;
                    /* Reject sub-pixels outside primary mirror or inside central obstruction */
                    if (r2 > r_out2 || r2 < r_in2)
                    {
                        continue;
                    }

                    /* Signed-distance test against each outward half-ray spider vane */
                    int blocked = 0;
                    if (nsp > 0 && half_w > 0.0f)
                    {
                        for (int32_t k = 0; k < nsp; k++)
                        {
                            /* Radial projection along vane k (>= 0 selects outward half-line) */
                            float along = dx * sp_cos[k] + dy * sp_sin[k];
                            if (along >= 0.0f)
                            {
                                /* Perpendicular distance from sub-pixel to spider centerline */
                                float perp = fabsf(-dx * sp_sin[k] + dy * sp_cos[k]);
                                if (perp <= half_w)
                                {
                                    blocked = 1;
                                    break;
                                }
                            }
                        }
                    }
                    if (!blocked)
                    {
                        hits++;
                    }
                }
            }
            /* Sub-pixel areal fill factor in [0, 1] serves as anti-aliased amplitude weight */
            float val                = (float) hits * (1.0f / 16.0f);
            pup_out[jy * nwf + ix]   = val;
            sum_amp                 += (double) val;
        }
    }

    return sum_amp;
}

/**
 * cacaosim_make_ncpa() - Generate deterministic low-order Zernike static NCPA OPD map
 * @nwf:         Square pupil grid dimension [pixels]
 * @pup_diam:    Outer pupil diameter fraction of `nwf`
 * @pup_amp:     Pupil amplitude weights `[nwf * nwf]` for support masking and RMS weighting
 * @ncpa_rms_um: Target pupil-weighted RMS OPD [microns] (`<= 0` clears `ncpa_out` to zero)
 * @seed:        Deterministic PRNG seed (`0` uses fallback seed `42`)
 * @ncpa_out:    Destination row-major `float` OPD buffer `[nwf * nwf]` in microns `[um]`
 *
 * Role:
 *   Generates static Non-Common Path Aberration (NCPA) OPD screens in microns for
 *   `cacaosim.mkpupil` and bench configurations, excluding tip/tilt so nominal
 *   image pointing is preserved while higher-order focal/coronagraphic speckles
 *   and WFS zero-point offsets are exercised.
 *
 * Rationale:
 *   Evaluates the 8 classic low-order aberrations (radial orders `n = 2..4`:
 *   defocus, 0/45-deg astigmatism, X/Y coma, X/Y trefoil, primary spherical) directly
 *   in normalized Cartesian pupil coordinates `(x, y)` to avoid per-pixel polar
 *   trigonometry (`atan2f`, `sinf`, `cosf`). Because obscured/spidered apertures
 *   break analytical Zernike orthogonality, a two-pass `double` reduction removes
 *   the exact `pup_amp`-weighted piston and rescales the RMS to `ncpa_rms_um`.
 */
void cacaosim_make_ncpa(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        ncpa_rms_um,
    uint64_t     seed,
    float       *ncpa_out)
{
    if (ncpa_out == NULL || nwf == 0)
    {
        return;
    }

    uint64_t npix = (uint64_t) nwf * nwf;
    if (ncpa_rms_um <= 0.0f || pup_amp == NULL)
    {
        __builtin_memset(ncpa_out, 0, npix * sizeof(float));
        return;
    }

    cacaosim_rng_t rng;
    cacaosim_rng_seed(&rng, (seed != 0ULL) ? seed : 42ULL, 99ULL);

    /*
     * Draw Gaussian modal amplitudes with radial-order decay:
     *   m = 0..2 (n=2: defocus, astigmatism x2) -> weight 1.0
     *   m = 3..6 (n=3: coma x2, trefoil x2)     -> weight 0.6
     *   m = 7    (n=4: primary spherical)       -> weight 0.4
     */
    float coef[8];
    for (int m = 0; m < 8; m++)
    {
        float weight = (m < 3) ? 1.0f : ((m < 7) ? 0.6f : 0.4f);
        coef[m]      = weight * cacaosim_rng_gauss_f32(&rng);
    }

    const float diam  = (pup_diam > 0.01f && pup_diam <= 1.0f) ? pup_diam : 0.9f;
    const float r_out = 0.5f * diam * (float) nwf;
    const float inv_r = 1.0f / r_out;
    const float ctr   = 0.5f * (float) (nwf - 1u);

    double wsum = 0.0;
    double mean = 0.0;

    /* Pass 1: Evaluate Cartesian Zernikes on illuminated pixels and accumulate weighted mean */
    for (uint32_t jy = 0; jy < nwf; jy++)
    {
        float y = ((float) jy - ctr) * inv_r;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            uint64_t idx = (uint64_t) jy * nwf + ix;
            float    w   = pup_amp[idx];
            if (w <= 0.0f)
            {
                ncpa_out[idx] = 0.0f;
                continue;
            }
            float x  = ((float) ix - ctr) * inv_r;
            float r2 = x * x + y * y;
            float z  = coef[0] * (2.0f * r2 - 1.0f) +
                       coef[1] * (x * x - y * y) +
                       coef[2] * (2.0f * x * y) +
                       coef[3] * (3.0f * r2 - 2.0f) * x +
                       coef[4] * (3.0f * r2 - 2.0f) * y +
                       coef[5] * (x * x * x - 3.0f * x * y * y) +
                       coef[6] * (3.0f * x * x * y - y * y * y) +
                       coef[7] * (6.0f * r2 * r2 - 6.0f * r2 + 1.0f);

            ncpa_out[idx] = z;
            wsum += (double) w;
            mean += (double) w * (double) z;
        }
    }

    if (wsum <= 1e-12)
    {
        __builtin_memset(ncpa_out, 0, npix * sizeof(float));
        return;
    }
    mean /= wsum;

    /* Pass 2: Compute exact pup_amp-weighted variance around the discrete piston mean */
    double var = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        if (pup_amp[i] > 0.0f)
        {
            double dv = (double) ncpa_out[i] - mean;
            var += (double) pup_amp[i] * dv * dv;
        }
    }
    double rms   = sqrt(var / wsum);
    float  scale = (rms > 1e-12) ? (ncpa_rms_um / (float) rms) : 0.0f;

    /* Pass 3: Subtract weighted piston and rescale to exact target ncpa_rms_um [microns] */
    for (uint64_t i = 0; i < npix; i++)
    {
        if (pup_amp[i] > 0.0f)
        {
            ncpa_out[i] = ((float) ((double) ncpa_out[i] - mean)) * scale;
        }
        else
        {
            ncpa_out[i] = 0.0f;
        }
    }
}

/**
 * cacaosim_make_wfe_m() - Generate static optical wavefront error map in OPD [meter]
 * @nwf:       Square pupil grid dimension [pixels]
 * @pup_diam:  Outer pupil diameter fraction of `nwf`
 * @pup_amp:   Optional pupil amplitude weights `[nwf * nwf]` (NULL uses circular `r <= r_out`)
 * @rms_m:     Target pupil-weighted RMS OPD [meter] (`<= 0` clears `wfe_out_m` to zero)
 * @psd_alpha: Mid-spatial-frequency amplitude slope `k^-alpha` (`<= 0.2` defaults to `1.6`)
 * @seed:      Deterministic PRNG seed (`0` uses fallback seed `101`)
 * @wfe_out_m: Destination row-major `float` OPD buffer `[nwf * nwf]` in SI meters `[m]`
 *
 * Role:
 *   Cold-path static aberration map synthesizer for `cacaosim.wfe`, producing
 *   2D optical surface/wavefront OPD error maps in SI `[meter]` units matching
 *   standard CACAO shared-memory and FITS calibration conventions.
 *
 * Rationale:
 *   Superimposes 8 unit-RMS polar Zernike modes (capturing optical mount, alignment,
 *   and figure errors) with up to `MAX_MD = 160` 2D Fourier polishing-ripple modes
 *   across spatial frequencies `1.8 <= |k| <= 8` cycles per pupil diameter weighted
 *   by `|k|^-alpha` (corresponding to an isotropic 2D power spectral density
 *   `PSD_2D(k) ~ k^(-2*alpha)`). Only the upper half-plane (`ky >= 0`, excluding
 *   `ky == 0 && kx <= 0`) is enumerated to avoid conjugate mode duplication.
 *   Evaluation extends slightly past the nominal pupil edge (`r^2 <= 1.04`, ~2%
 *   radial guard margin) so small sub-pixel shifts in `cacaosim.wfe` do not sample
 *   an artificial discontinuity at `r = 1`, while weighted piston and RMS
 *   statistics are computed strictly over the clear aperture (`pup_amp` or `r^2 <= 1`).
 */
void MILK_COLD cacaosim_make_wfe_m(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        rms_m,
    float        psd_alpha,
    uint64_t     seed,
    float       *wfe_out_m)
{
    if (wfe_out_m == NULL || nwf == 0)
    {
        return;
    }

    const uint64_t npix = (uint64_t) nwf * nwf;
    memset(wfe_out_m, 0, npix * sizeof(float));

    if (rms_m <= 0.0f || pup_diam <= 0.0f)
    {
        return;
    }

    const float alpha = (psd_alpha > 0.2f) ? psd_alpha : 1.6f;
    const float cx    = 0.5f * (float) (nwf - 1u);
    const float cy    = 0.5f * (float) (nwf - 1u);
    const float d_px  = pup_diam * (float) nwf;
    const float r_out = 0.5f * d_px;

    cacaosim_rng_t rng;
    cacaosim_rng_seed(&rng, (seed != 0ULL) ? seed : 101ULL, 101ULL);

    /* Low-order Zernike polishing & alignment terms (focus, astig, coma, trefoil, spherical) */
    float       zc[8];
    const float z_w[8] = { 1.0f, 0.85f, 0.85f, 0.55f, 0.55f, 0.40f, 0.40f, 0.35f };
    for (int k = 0; k < 8; k++)
    {
        zc[k] = z_w[k] * cacaosim_rng_gauss_f32(&rng);
    }

    /*
     * Mid-spatial-frequency 2D Fourier polishing ripple modes (1.8..8 cycles / pupil):
     * traverse unique half-plane wavevectors (kx, ky) and draw k^-alpha amplitudes & phases.
     */
    enum
    {
        KMAX   = 8,
        MAX_MD = 160
    };
    int   mkx[MAX_MD];
    int   mky[MAX_MD];
    float mamp[MAX_MD];
    float mphi[MAX_MD];
    int   n_md = 0;

    for (int ky = 0; ky <= KMAX; ky++)
    {
        for (int kx = -KMAX; kx <= KMAX; kx++)
        {
            if (ky == 0 && kx <= 0)
            {
                continue;
            }
            float kf = hypotf((float) kx, (float) ky);
            if (kf < 1.8f || kf > (float) KMAX || n_md >= MAX_MD)
            {
                continue;
            }
            mkx[n_md]  = kx;
            mky[n_md]  = ky;
            mamp[n_md] = 0.65f * powf(kf, -alpha) * (0.5f + 0.7f * cacaosim_rng_uniform_f32(&rng));
            mphi[n_md] = CACAOSIM_TWO_PI_F * cacaosim_rng_uniform_f32(&rng);
            n_md++;
        }
    }

    double w_sum = 0.0;
    double m_sum = 0.0;

    /* Pass 1: Sum Noll-normalized Zernikes (sqrt(3), sqrt(6), sqrt(8), sqrt(5)) + Fourier modes */
    for (uint32_t iy = 0; iy < nwf; iy++)
    {
        const float y  = ((float) iy - cy) / r_out;
        const float uy = ((float) iy - cy) / d_px;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            const float x  = ((float) ix - cx) / r_out;
            const float ux = ((float) ix - cx) / d_px;
            const float r2 = x * x + y * y;
            /* Populate out to a 2% radial guard margin (r <= sqrt(1.04)) for smooth edge shifts */
            if (r2 > 1.04f)
            {
                continue;
            }

            float r   = sqrtf(r2);
            float th  = atan2f(y, x);
            float val = 0.0f;

            val += zc[0] * 1.7320508f * (2.0f * r2 - 1.0f);
            val += zc[1] * 2.4494897f * r2 * cosf(2.0f * th);
            val += zc[2] * 2.4494897f * r2 * sinf(2.0f * th);
            val += zc[3] * 2.8284271f * (3.0f * r2 - 2.0f) * r * cosf(th);
            val += zc[4] * 2.8284271f * (3.0f * r2 - 2.0f) * r * sinf(th);
            val += zc[5] * 2.8284271f * r * r2 * cosf(3.0f * th);
            val += zc[6] * 2.8284271f * r * r2 * sinf(3.0f * th);
            val += zc[7] * 2.2360680f * (6.0f * r2 * r2 - 6.0f * r2 + 1.0f);

            for (int m = 0; m < n_md; m++)
            {
                float ang = CACAOSIM_TWO_PI_F * ((float) mkx[m] * ux + (float) mky[m] * uy) +
                            mphi[m];
                val += mamp[m] * cosf(ang);
            }

            uint64_t idx   = (uint64_t) iy * nwf + ix;
            wfe_out_m[idx] = val;

            /* Restrict piston/variance weighting strictly to the nominal pupil (pup_amp or r<=1) */
            double wt = (pup_amp != NULL) ? (double) pup_amp[idx] : ((r2 <= 1.0f) ? 1.0 : 0.0);
            if (wt > 0.0)
            {
                w_sum += wt;
                m_sum += wt * (double) val;
            }
        }
    }

    if (w_sum <= 0.0)
    {
        return;
    }

    /* Pass 2: Accumulate pupil-weighted variance around piston mean in double precision */
    const double mean = m_sum / w_sum;
    double       var  = 0.0;
    for (uint64_t i = 0; i < npix; i++)
    {
        double wt = 0.0;
        if (pup_amp != NULL)
        {
            wt = (double) pup_amp[i];
        }
        else if (wfe_out_m[i] != 0.0f)
        {
            uint32_t ix = (uint32_t) (i % nwf);
            uint32_t iy = (uint32_t) (i / nwf);
            float    x  = ((float) ix - cx) / r_out;
            float    y  = ((float) iy - cy) / r_out;
            wt          = (x * x + y * y <= 1.0f) ? 1.0 : 0.0;
        }
        if (wt > 0.0)
        {
            double dv = (double) wfe_out_m[i] - mean;
            var += wt * dv * dv;
        }
    }

    double rms   = sqrt(var / w_sum);
    float  scale = (rms > 1e-30) ? (rms_m / (float) rms) : 0.0f;

    /* Pass 3: Remove piston and rescale supported pixels (including 2% guard ring if unmasked) */
    for (uint64_t i = 0; i < npix; i++)
    {
        int in_pup = 0;
        if (pup_amp != NULL)
        {
            in_pup = (pup_amp[i] > 0.0f) ? 1 : 0;
        }
        else if (wfe_out_m[i] != 0.0f)
        {
            uint32_t ix = (uint32_t) (i % nwf);
            uint32_t iy = (uint32_t) (i / nwf);
            float    x  = ((float) ix - cx) / r_out;
            float    y  = ((float) iy - cy) / r_out;
            in_pup      = (x * x + y * y <= 1.04f) ? 1 : 0;
        }
        wfe_out_m[i] = in_pup ? (((float) ((double) wfe_out_m[i] - mean)) * scale) : 0.0f;
    }
}
