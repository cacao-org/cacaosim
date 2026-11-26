// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_disp.h
 * @brief   Chromatic angular dispersion descriptor (`cacaosim_disp_t`), Maunakea air model, & SHM
 *
 * Architectural Role:
 *   Carries exact broadband sky-projected angular dispersion `(theta_x(lambda), theta_y(lambda))`
 *   [rad] across optical beams in `0.40..2.40 um` without expanding upstream pupil grids into
 *   dense 3D wavelength cubes. Uses a 12-term Chebyshev expansion in wavenumber `s = 1 / lambda`
 *   that represents both Ciddor/Seidelmann atmospheric refraction and exact 3D Snell refraction
 *   through high-index doublet ADC prisms to `< 1e-5 mas`, composes additively across optical
 *   elements, transforms analytically under 2D pupil affine registration (`cacaosim_reg_t`), and
 *   serializes into fixed O(1) ImageStreamIO SHM keywords (`kw[39..63]`, `NBkw >= 64`).
 */

#ifndef CACAOSIM_DISP_H
#define CACAOSIM_DISP_H

#include <stdint.h>

#ifndef __CUDACC__
#include "milk_runnable.h"
#include "ImageStreamIO.h"
#endif
#include "common/cacaosim_beam.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Number of Chebyshev wavenumber modes per axis (covers 0.40..2.40 um to < 1e-5 mas) */
#define CACAOSIM_DISP_NCOEFF 12

/** Minimum / maximum wavelength bounds [um] matching the 32-bin beam spectrum grid */
#define CACAOSIM_DISP_LMIN_UM 0.40
#define CACAOSIM_DISP_LMAX_UM 2.40

/** Minimum SHM keyword count required to store spec (0..35) + poly (36..38) + disp (39..63) */
#define CACAOSIM_BEAM_SHM_NBKW 80

/** Fixed SHM keyword slot offset for `CSIM_DSP`, `CSIM_DX00..11`, `CSIM_DY00..11` */
#define CACAOSIM_BEAM_KW_DISP_BASE (4 + CACAOSIM_NSPEC + 3)

/**
 * enum cacaosim_beam_poly_mode_t - Meaning of multi-wavelength slices in `cacaosim_beam_slot_t`
 * @CACAOSIM_POLY_NONE:          Single-slice monochromatic/achromatic `[1, nwf, nwf]` beam.
 * @CACAOSIM_POLY_UNWRAPPED_OPD: Multi-band unwrapped real `OPD [um]` + real `Amp` (e.g.
 *                               chromatic multi-layer turbulence from `milkatmturb`); consumers
 *                               interpolate real OPD & Amp linearly in `1/lam^2` before forming
 *                               the complex entrance-pupil phasor.
 * @CACAOSIM_POLY_COMPLEX_FIELD: Post-coronagraph diffracted complex pupil field encoded as
 *                               wrapped `(OPD, Amp)` per wavelength channel; consumers use
 *                               complex phasor interpolation.
 */
typedef enum
{
    CACAOSIM_POLY_NONE          = 0,
    CACAOSIM_POLY_UNWRAPPED_OPD = 1,
    CACAOSIM_POLY_COMPLEX_FIELD = 2
} cacaosim_beam_poly_mode_t;

/**
 * struct cacaosim_disp_t - Compact 12-mode Chebyshev chromatic angular dispersion descriptor
 * @active: `0` = achromatic zero-dispersion fast path; `1` = chromatic tilt active.
 * @gen:    Monotonic revision counter bumped whenever coefficients change (for GPU slope caches).
 * @cx:     Chebyshev coefficients of `theta_x(1/lambda)` `[rad on sky]`.
 * @cy:     Chebyshev coefficients of `theta_y(1/lambda)` `[rad on sky]`.
 */
typedef struct
{
    int      active;
    uint64_t gen;
    double   cx[CACAOSIM_DISP_NCOEFF];
    double   cy[CACAOSIM_DISP_NCOEFF];
} cacaosim_disp_t;

/**
 * struct cacaosim_atm_disp_cfg_t - Observing & atmospheric parameters for Maunakea refraction
 * @enable:       `0` = disabled (e.g. internal calibration source), `1` = enabled on sky.
 * @zenith_deg:   Observing zenith distance `[deg]` (0 = zenith, default 30 deg for SCExAO).
 * @disp_pa_deg:  Azimuth angle of the zenith dispersion vector on the bench pupil `[deg]`.
 * @site_alt_m:   Observatory altitude above sea level `[m]` (Maunakea = 4205 m).
 * @pressure_hpa: Ambient atmospheric pressure `[hPa]` (Maunakea ~ 615 hPa; Egner Zemax ~ 602.4).
 * @temp_c:       Ambient air temperature `[deg C]` (Maunakea ~ 2.0 C).
 * @rh_pct:       Ambient relative humidity `[%]` (Maunakea ~ 20%).
 * @ref_lam_um:   Telescope pointing / guiding reference wavelength `[um]` (zero-shift anchor).
 * @sphere_corr:  `1` = include spherical-atmosphere `-beta * tan^3(z)` term (`0` = plane-parallel).
 */
typedef struct
{
    int32_t enable;
    float   zenith_deg;
    float   disp_pa_deg;
    float   site_alt_m;
    float   pressure_hpa;
    float   temp_c;
    float   rh_pct;
    float   ref_lam_um;
    int32_t sphere_corr;
} cacaosim_atm_disp_cfg_t;

/** Callback type for sampling an exact dispersion curve `(tx_rad, ty_rad)` at `lam_um` */
typedef void (*cacaosim_disp_eval_fn_t)(
    double  lam_um,
    void   *ctx,
    double *tx_rad,
    double *ty_rad);

/** Initialize atmospheric dispersion defaults for Maunakea (disabled at z = 0 by default) */
void cacaosim_atm_disp_cfg_defaults(cacaosim_atm_disp_cfg_t *cfg);

/**
 * cacaosim_air_n_minus_1() - Ciddor (1996) moist-air refractivity `(n - 1)`
 * @lam_um:       Vacuum optical wavelength `[um]` (valid 0.23..2.50 um)
 * @pressure_hpa: Atmospheric pressure `[hPa]`
 * @temp_c:       Air temperature `[deg C]`
 * @rh_pct:       Relative humidity `[0..100 %]`
 *
 * Return: Dimensionless refractivity `n_air - 1`.
 */
double cacaosim_air_n_minus_1(
    double lam_um,
    double pressure_hpa,
    double temp_c,
    double rh_pct);

/**
 * cacaosim_atm_disp_rad() - Differential atmospheric refraction `[rad]` at `@lam_um`
 * @lam_um: Optical wavelength `[um]`
 * @cfg:    Observing & atmospheric parameters
 *
 * Positive values bend shorter wavelengths toward zenith relative to `cfg->ref_lam_um`.
 *
 * Return: Angular differential refraction `[rad]` (0.0 if disabled or `zenith_deg <= 0`).
 */
double cacaosim_atm_disp_rad(
    double                         lam_um,
    const cacaosim_atm_disp_cfg_t *cfg);

/** Reset dispersion descriptor to inactive zero (preserving/incrementing generation counter) */
void cacaosim_disp_zero(cacaosim_disp_t *d);

/** Fit 12-term Chebyshev wavenumber expansion from exact callback via Gauss-Chebyshev DCT */
void cacaosim_disp_fit_fn(
    cacaosim_disp_t        *d,
    cacaosim_disp_eval_fn_t fn,
    void                   *ctx);

/** Populate dispersion descriptor directly from atmospheric refraction parameters */
void cacaosim_disp_set_atm(
    cacaosim_disp_t               *d,
    const cacaosim_atm_disp_cfg_t *cfg);

/** Compose two dispersion stages additively (`dst = a + b`, safe if aliasing) */
void cacaosim_disp_add(
    cacaosim_disp_t       *dst,
    const cacaosim_disp_t *a,
    const cacaosim_disp_t *b);

/** Transform dispersion coefficients analytically under 2D pupil registration `(rot_deg, scale)` */
void cacaosim_disp_apply_reg(
    const cacaosim_disp_t *in,
    const cacaosim_reg_t  *reg,
    cacaosim_disp_t       *out);

/** Evaluate sky-projected angular dispersion `(tx_rad, ty_rad)` `[rad]` at `@lam_um` */
void cacaosim_disp_eval_rad(
    const cacaosim_disp_t *d,
    double                 lam_um,
    double                *tx_rad,
    double                *ty_rad);

/**
 * cacaosim_disp_eval_slopes() - Convert chromatic sky tilt into pupil OPD slopes `[um / pixel]`
 * @d:          Active dispersion descriptor (`NULL` or inactive yields zeros)
 * @lam_um:     Array of `nlam` exact channel wavelengths `[um]`
 * @nlam:       Number of channels to evaluate
 * @tel_diam_m: Telescope primary mirror diameter `[m]`
 * @nwf:        Square pupil grid dimension `[pixels]`
 * @pup_diam:   Pupil diameter fraction of `@nwf`
 * @out_sx:     Output array `[nlam]` of X pupil OPD slopes `[um / pixel]`
 * @out_sy:     Output array `[nlam]` of Y pupil OPD slopes `[um / pixel]`
 *
 * Sign convention: positive `(tx, ty)` introduces OPD tilt `+(sx*(ix-p0) + sy*(iy-p0))` `[um]`,
 * shifting the focal-plane PSF centroid proportionally to `(tx, ty)`.
 */
void cacaosim_disp_eval_slopes(
    const cacaosim_disp_t *d,
    const float           *lam_um,
    uint32_t               nlam,
    float                  tel_diam_m,
    uint32_t               nwf,
    float                  pup_diam,
    float                 *out_sx,
    float                 *out_sy);

/** Compute peak-to-valley angular dispersion span `[mas]` across `[lam0_um, lam1_um]` */
double cacaosim_disp_band_span_mas(
    const cacaosim_disp_t *d,
    double                 lam0_um,
    double                 lam1_um);

/** Compute vector centroid shift `[mas]` between two wavelengths `lam_a_um` and `lam_b_um` */
double cacaosim_disp_shift_mas(
    const cacaosim_disp_t *d,
    double                 lam_a_um,
    double                 lam_b_um);

#ifndef __CUDACC__
/** Create/connect a 3D float32 optical beam SHM cube with `NBkw = CACAOSIM_BEAM_SHM_NBKW` (80) */
IMGID cacaosim_beam_shm_create(
    const char *name,
    uint32_t    nwf,
    uint32_t    nz);
#endif

/** Write dispersion descriptor into fixed SHM keywords `kw[39..63]` (`NBkw >= 64`) */
void cacaosim_beam_shm_write_disp(
    IMAGE                 *im,
    const cacaosim_disp_t *d);

/** Read dispersion descriptor from SHM keywords `kw[39..63]` (returns `1` if active, `0` else) */
int cacaosim_beam_shm_read_disp(
    const IMAGE     *im,
    cacaosim_disp_t *d);

/** Write `CSIM_PMD` poly_mode + per-slice wavelengths for unwrapped multi-band turbulence beams */
void cacaosim_beam_shm_write_poly_mode(
    IMAGE   *im,
    int32_t  poly_mode);

/** Read `CSIM_PMD` poly_mode from SHM beam (`CACAOSIM_POLY_COMPLEX_FIELD` default when nwl > 1) */
int32_t cacaosim_beam_shm_read_poly_mode(
    const IMAGE *im,
    uint32_t     nwl);

/** Write telescope primary diameter `CSIM_DIA` [m] at fixed slot `kw[65]` */
void cacaosim_beam_shm_write_diam(
    IMAGE *im,
    float  tel_diam_m);

/** Read telescope primary diameter `CSIM_DIA` [m] from `kw[65]` (returns `fallback_m` if unset) */
float cacaosim_beam_shm_read_diam(
    const IMAGE *im,
    float        fallback_m);

/** Write up to 8 explicit per-slice wavelengths `CSIM_W00..07` [um] at `kw[66..73]` */
void cacaosim_beam_shm_write_poly_wls(
    IMAGE       *im,
    const float *lam_um,
    uint32_t     nwl);

/** Read up to `nwl` per-slice wavelengths from `CSIM_W00..07` (returns 1 if present, 0 else) */
int cacaosim_beam_shm_read_poly_wls(
    const IMAGE *im,
    uint32_t     nwl,
    float       *out_lam_um);

/**
 * cacaosim_wf_shm_read_chrom() - Detect 3D chromatic milkatmturb SHM cube (`ATMT_NWL >= 2`)
 * @im:       Input `outarraypha` SHM image from `milkatmturb`
 * @out_nwl:  Output slice count (`2..8`)
 * @out_unit: Output unit (`0` = OPD [um], `1` = phase [rad])
 * @out_wls:  Output array `[8]` of slice wavelengths `[um]`
 *
 * Return: `1` if `@im` is a valid 3D multi-wavelength chromatic turbulence cube, `0` otherwise.
 */
int cacaosim_wf_shm_read_chrom(
    const IMAGE *im,
    uint32_t    *out_nwl,
    int32_t     *out_unit,
    float       *out_wls);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_DISP_H */
