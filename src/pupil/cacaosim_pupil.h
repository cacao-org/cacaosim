// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pupil.h
 * @brief   Anti-aliased telescope pupil mask, static Zernike NCPA, and PSD WFE synthesis
 *
 * Architectural Role:
 *   Declares the cold-path pupil-plane geometry and static optical aberration
 *   generators for `cacaosim`. These routines are invoked once at configuration
 *   or stream-initialization time by standalone FPS units (`cacaosim.mkpupil`,
 *   `cacaosim.mkbeam`, `cacaosim.wfe`), by the fused multi-rate optical bench
 *   engine (`cacaosim.htsim`), and by validation benchmarks to populate 2D
 *   `FLOAT32` shared-memory or host-resident maps prior to real-time execution.
 *
 * Design Rationale & Motivation:
 *   - Sub-pixel anti-aliasing (`cacaosim_make_pupil`): Evaluating hard circular
 *     apertures and thin spider vanes solely at pixel centers creates binary
 *     stair-step artifacts along oblique edges. In downstream Fourier optics,
 *     those artificial high-spatial-frequency steps scatter spurious diffraction
 *     into coronagraphic dark holes (`cacaosim.coro`) and bias edge subapertures
 *     in Shack-Hartmann, Pyramid, and Zernike WFS models. Stratified 4x4
 *     sub-pixel supersampling yields 17 smooth gray-edge amplitude levels in
 *     `[0, 1]`, suppressing discretization ringing at negligible startup cost.
 *   - Exact discrete pupil-weighted normalization: Analytical Zernike modes are
 *     orthogonal only over a continuous, unobscured unit disk. Central
 *     obstructions, spider vanes, and finite pixel sampling break continuous
 *     orthogonality. Both `cacaosim_make_ncpa()` and `cacaosim_make_wfe_m()`
 *     explicitly subtract the discrete pupil-weighted piston mean and rescale by
 *     the discrete pupil-weighted RMS in `double` precision so the output map
 *     matches the requested aberration budget on arbitrary aperture geometries.
 *   - Dual unit conventions (`[um]` vs `[m]`): Internal `cacaosim` beams store
 *     OPD in microns (`[um]`) to keep `float` phase values well-conditioned,
 *     whereas CACAO calibration pipelines and `cacaosim.wfe` streams follow the
 *     SI `[meter]` OPD convention. Providing dedicated synthesis entry points
 *     for both conventions prevents ad-hoc unit conversion errors in callers.
 */

#ifndef CACAOSIM_PUPIL_H
#define CACAOSIM_PUPIL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Geometric parameters for 4x4 anti-aliased telescope pupil synthesis.
 *
 * Coordinate conventions:
 *   - The pupil is centered symmetrically at `(ctr, ctr)` where
 *     `ctr = 0.5 * (nwf - 1)` in 0-indexed pixel coordinates `(ix, jy)`.
 *   - Outer clear diameter spans `pup_diam * nwf` pixels (`r_out = 0.5 * pup_diam * nwf`).
 *   - Secondary mirror central obstruction spans `cobs * pup_diam * nwf` pixels.
 *   - Radial spider vanes are half-rays originating at `(ctr, ctr)` spaced
 *     uniformly by `360 / nspider` degrees counter-clockwise starting from
 *     `spider_deg` relative to the `+X` axis, with full width
 *     `spider_w * pup_diam * nwf` pixels.
 */
typedef struct
{
    uint32_t nwf;        /**< Square pupil grid dimension [pixels] (> 0) */
    float    pup_diam;   /**< Outer clear diameter as fraction of nwf (0.01 < pup_diam <= 1.0) */
    float    cobs;       /**< Linear central obstruction ratio D_sec / D_tel [0.0 .. 0.98) */
    int32_t  nspider;    /**< Number of radial spider support vanes [0 .. 16] (0 = none) */
    float    spider_w;   /**< Full spider vane thickness as fraction of pup_diam (>= 0) */
    float    spider_deg; /**< Counter-clockwise azimuth of first spider vane from +X [degrees] */
} cacaosim_pupil_cfg_t;

/**
 * cacaosim_make_pupil() - Generate 4x4 sub-pixel anti-aliased 2D pupil amplitude array
 * @cfg:     Pointer to pupil geometry configuration (must be non-NULL with `nwf > 0`)
 * @pup_out: Destination row-major `float` amplitude buffer `[cfg->nwf * cfg->nwf]` in `[0, 1]`
 *
 * Role:
 *   Cold-path telescope aperture synthesizer invoked by `cacaosim.mkpupil`,
 *   `cacaosim.mkbeam` (fallback pupil), `cacaosim.coro` (default pupil/Lyot
 *   masks), and `cacaosim.htsim` to construct the 2D pupil amplitude mask.
 *
 * Rationale:
 *   Evaluates an analytic annular aperture intersected with `nspider` radial
 *   support vanes on a 4x4 stratified sub-pixel grid per wavefront pixel. Taking
 *   the sub-pixel areal fill fraction as the pixel amplitude mitigates jagged
 *   boundary aliasing in downstream Fresnel/Fraunhofer FFT and Matrix-DFT
 *   propagations without requiring oversized wavefront grids.
 *
 * Return: Sum of pupil amplitude values (effective clear area in pixels), or `0.0` on invalid
 *         arguments (`cfg == NULL`, `pup_out == NULL`, or `cfg->nwf == 0`).
 */
double cacaosim_make_pupil(
    const cacaosim_pupil_cfg_t *cfg,
    float                      *pup_out);

/**
 * cacaosim_make_ncpa() - Generate deterministic low-order Zernike static NCPA OPD map
 * @nwf:         Square pupil grid dimension [pixels] (`> 0`)
 * @pup_diam:    Outer pupil diameter as fraction of `nwf` (`0.01 < pup_diam <= 1.0`)
 * @pup_amp:     Pupil amplitude weights `[nwf * nwf]` used for support masking and weighted
 *               zero-mean / RMS normalization (if NULL or `ncpa_rms_um <= 0`, outputs zeros)
 * @ncpa_rms_um: Target pupil-weighted RMS OPD [microns] (`<= 0.0` produces an all-zero map)
 * @seed:        Deterministic `xoshiro128++` PRNG seed (`0` maps to default seed `42`)
 * @ncpa_out:    Destination row-major `float` OPD buffer `[nwf * nwf]` in microns `[um]`
 *
 * Role:
 *   Cold-path generator for static Non-Common Path Aberration (NCPA) maps
 *   consumed by `cacaosim.mkpupil`, `cacaosim.mkbeam`, `cacaosim.beamsplit`
 *   (differential reflected/transmitted arm NCPA), and `cacaosim.htsim`.
 *
 * Rationale:
 *   Differential static optical aberrations between WFS and science arms are
 *   dominated by low-order alignment and mount modes (defocus, astigmatism,
 *   coma, trefoil, spherical). Synthesizing an 8-mode Cartesian Zernike
 *   combination with radially decaying modal weights and exact discrete
 *   `pup_amp`-weighted piston removal and RMS normalization guarantees
 *   reproducible, zero-piston NCPA realizations across arbitrary pupil masks.
 */
void cacaosim_make_ncpa(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        ncpa_rms_um,
    uint64_t     seed,
    float       *ncpa_out);

/**
 * cacaosim_make_wfe_m() - Generate static optical wavefront error map in OPD [meter]
 * @nwf:       Square pupil grid dimension [pixels] (`> 0`)
 * @pup_diam:  Outer pupil diameter as fraction of `nwf` (`> 0.0`)
 * @pup_amp:   Optional pupil amplitude weights `[nwf * nwf]` for piston/RMS weighting
 *             (NULL falls back to an analytic circular disk of diameter `pup_diam * nwf`)
 * @rms_m:     Target pupil-weighted RMS OPD [meter] (e.g. `5.0e-8f` for 50 nm; `<= 0` -> zeros)
 * @psd_alpha: Mid-spatial-frequency amplitude decay exponent `k^-alpha` (`<= 0.2` -> `1.6`)
 * @seed:      Deterministic `xoshiro128++` PRNG seed (`0` maps to default seed `101`)
 * @wfe_out_m: Destination row-major `float` OPD buffer `[nwf * nwf]` in SI meters `[m]`
 *
 * Role:
 *   Cold-path static wavefront error generator backing `cacaosim.wfe` and bench
 *   initialization tools, producing persistent 2D OPD maps in SI meters `[m]`
 *   compatible with CACAO calibration and focal-plane WFS conventions.
 *
 * Rationale:
 *   Real optical surfaces exhibit both low-order mount/figure errors (Zernikes)
 *   and mid-spatial-frequency polishing ripples that scatter stellar halo speckles
 *   into coronagraphic control radii (`2..8` cycles/pupil). Combining 8 polar
 *   Zernike modes with a randomized `k^-alpha` 2D Fourier mode bank — and
 *   extending unmasked support to a `2%` radial guard band (`r^2 <= 1.04`) while
 *   normalizing RMS strictly over `r^2 <= 1.0` — prevents edge cliffs under
 *   downstream sub-pixel registration shifts while delivering exact SI `[meter]`
 *   RMS scaling.
 */
void cacaosim_make_wfe_m(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        rms_m,
    float        psd_alpha,
    uint64_t     seed,
    float       *wfe_out_m);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_PUPIL_H */
