// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_beam.h
 * @brief   Optical `[OPD, Amp]` beam algebra, 32-bin photometry, 2D registration, and SHM beam I/O
 *
 * Architectural Role:
 *   Defines the standardized optical beam representation, spectral radiometry engine, pupil-plane
 *   affine registration primitives, and `ImageStreamIO` shared-memory metadata protocol connecting
 *   all optical components in `cacaosim`:
 *     - Upstream sources & modifiers: `mkbeam`, `wfe`, `dmsim`, `beamsplit`, `coro`
 *     - Downstream sensors: `pyrwfs`, `shwfs`, `zwfs`, `fpcam`
 *     - Orchestration & analytics: fused `htsim`, `cacaosim-CTRL` TUI, and `cacaosim-bench-view`
 *
 * Design Rationale & Motivation:
 *   1. Factored `[OPD_um, Amplitude]` pupil beam cube (`[nwf x nwf x 2]` or `[nwf x nwf x 2*N]`):
 *      Storing geometric optical path difference in microns (`slice 0`) and real scalar amplitude
 *      (`slice 1`) rather than a monochromatic complex electric field `A * exp(i * phi(lambda))`
 *      keeps reflective DMs, atmospheric turbulence, and static mirror WFE strictly achromatic and
 *      immune to `2*pi` phase-wrapping discontinuities across Visible/NIR arms, while extending
 *      seamlessly to `2 * N_lambda` slices (`CSIM_NWL`) downstream of dispersive/coronagraphic
 *      elements.
 *   2. Zero-FFT 32-bin `[0.40..2.40 um]` spectral vector in SHM keywords (`CSIM_S00..S31`):
 *      Multi-stage AO benches cascade gray, long-pass, short-pass, and band-reflect dichroics
 *      spanning visible (`0.40 um`) through K-band (`2.40 um`, `dlambda = 0.0625 um` per bin).
 *      Propagating per-bin photon arrival rates `[photons/s]` in SHM keywords lets beamsplitters
 *      and camera filters compute exact radiometric flux and flux-weighted effective wavelength
 *      (`eff_lambda`) algebraically in microseconds—without running 32 optical FFTs per element—
 *      and lets offline `.bench` tools audit photon budgets instantaneously.
 *   3. Fixed-slot `O(1)` SHM keyword layout: Reserving deterministic keyword indices `kw[0..35]`
 *      for monochromatic/spectral metadata and `kw[36..38]` for polychromatic channel bounds
 *      avoids `O(NBkw)` string-comparison loops inside kHz real-time frame writers and readers.
 *   4. Inverse-affine 2D pupil registration with identity fast-path: Models lateral shift, pupil
 *      clocking rotation, and magnification mismatches between optical conjugate planes via
 *      bilinear inverse warping while bypassing interpolation completely when `reg` is identity.
 */

#ifndef CACAOSIM_BEAM_H
#define CACAOSIM_BEAM_H

#include <stdint.h>

#include "ImageStruct.h"
#include "cacaosim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 32-bin uniform spectral grid covering visible through K-band (`[0.40, 2.40] um`, bin width
 * `CACAOSIM_SPEC_DL_UM = 0.0625 um`), chosen to resolve standard astronomical Johnson-Cousins /
 * Maunakea `B, V, R, I, z, Y, J, H, K` bandpasses and typical AO dichroic transition edges while
 * fitting compactly inside standard `ImageStreamIO` keyword tables (`NBkw >= 39`).
 */
#define CACAOSIM_NSPEC      32
#define CACAOSIM_SPEC_L0_UM 0.40f
#define CACAOSIM_SPEC_L1_UM 2.40f
#define CACAOSIM_SPEC_DL_UM ((CACAOSIM_SPEC_L1_UM - CACAOSIM_SPEC_L0_UM) / (float) CACAOSIM_NSPEC)

/**
 * enum cacaosim_split_mode_t - Beamsplitter / dichroic spectral splitting law
 * @CACAOSIM_SPLIT_GRAY:        Achromatic neutral splitter with constant reflectance `r_gray` and
 *                              transmittance `t_gray` across all 32 wavelength bins.
 * @CACAOSIM_SPLIT_DICHROIC_LP: Long-pass dichroic: reflects short wavelengths (`< lambda_cut`,
 *                              e.g., visible WFS pickoff) and transmits long wavelengths
 *                              (`> lambda_cut`, e.g., NIR science path).
 * @CACAOSIM_SPLIT_DICHROIC_SP: Short-pass dichroic: reflects long wavelengths (`> lambda_cut`)
 *                              and transmits short wavelengths (`< lambda_cut`).
 * @CACAOSIM_SPLIT_DICHROIC_BP: Notch / band-reflect dichroic: reflects `[lambda_cut - w/2,
 *                              lambda_cut + w/2]` and transmits wavelengths outside that band.
 */
typedef enum
{
    CACAOSIM_SPLIT_GRAY        = 0, /**< Achromatic gray splitter (R_gray, T_gray) */
    CACAOSIM_SPLIT_DICHROIC_LP = 1, /**< Long-pass: refl < lambda_cut, trans > lambda_cut */
    CACAOSIM_SPLIT_DICHROIC_SP = 2, /**< Short-pass: refl > lambda_cut, trans < lambda_cut */
    CACAOSIM_SPLIT_DICHROIC_BP = 3  /**< Band-reflect: refl [cut - w/2, cut + w/2], passes rest */
} cacaosim_split_mode_t;

/**
 * struct cacaosim_spec_t - 32-bin multi-wavelength photon arrival rate spectrum
 * @ph_s: Integrated photon arrival rate `[photons / second]` within each `0.0625 um` spectral bin
 *        `b = 0 .. CACAOSIM_NSPEC - 1` spanning `[0.40, 2.40] um` across the clear telescope area.
 */
typedef struct
{
    float ph_s[CACAOSIM_NSPEC]; /**< Photon arrival rate [photons/s] in each spectral bin */
} cacaosim_spec_t;

/**
 * struct cacaosim_split_cfg_t - Gray or dichroic beamsplitter coating configuration
 * @mode:       Spectral splitting law (`cacaosim_split_mode_t`).
 * @r_gray:     Achromatic reflectance `[0..1]` used when `mode == CACAOSIM_SPLIT_GRAY`.
 * @t_gray:     Achromatic transmittance `[0..1]` used when `mode == CACAOSIM_SPLIT_GRAY`.
 * @lambda_cut: Dichroic 50% transition edge (LP/SP) or reflected band center (BP) `[um]`.
 * @band_width: Full width `[um]` of the reflected band when `mode == CACAOSIM_SPLIT_DICHROIC_BP`.
 * @roll_um:    10%--90%-scale logistic transition width `[um]` modeling finite dielectric coating
 *              edge steepness (`<= 1e-4f` defaults internally to `0.02 um`).
 * @eff_refl:   Peak coating reflectance `[0..1]` inside the nominal reflection band (dichroic).
 * @eff_trans:  Peak coating transmittance `[0..1]` inside the nominal passband (dichroic).
 */
typedef struct
{
    int32_t mode;       /**< Split mode (cacaosim_split_mode_t) */
    float   r_gray;     /**< Gray reflectance [0..1] when mode == CACAOSIM_SPLIT_GRAY */
    float   t_gray;     /**< Gray transmittance [0..1] when mode == CACAOSIM_SPLIT_GRAY */
    float   lambda_cut; /**< Dichroic cutoff or band center wavelength [um] */
    float   band_width; /**< Band-reflect full width [um] when mode == CACAOSIM_SPLIT_DICHROIC_BP */
    float   roll_um;    /**< Dichroic transition edge width [um] (default: 0.03) */
    float   eff_refl;   /**< Peak reflectance efficiency in reflection band [0..1] */
    float   eff_trans;  /**< Peak transmittance efficiency in pass band [0..1] */
} cacaosim_split_cfg_t;

/**
 * struct cacaosim_reg_t - 2D pupil-plane affine misregistration transform parameters
 * @shift_x: Lateral translation of the beam along +X `[pupil pixels]` relative to grid center.
 * @shift_y: Lateral translation of the beam along +Y `[pupil pixels]` relative to grid center.
 * @rot_deg: Counter-clockwise in-plane pupil rotation angle `[degrees]` about `(nwf - 1) / 2`.
 * @scale:   Isotropic pupil magnification factor (`<= 1e-4f` or `1.0f` denotes unity scale `1.0`).
 *
 * Rationale: Parameterizes forward optical displacement `(shift, rot, scale)` of an incoming beam
 * relative to a downstream pupil plane; relay kernels evaluate the exact analytical inverse map.
 */
typedef struct
{
    float shift_x; /**< Translation along X [pupil pixels] */
    float shift_y; /**< Translation along Y [pupil pixels] */
    float rot_deg; /**< Counter-clockwise pupil rotation [degrees] */
    float scale;   /**< Isotropic pupil magnification (0 or 1 -> 1.0) */
} cacaosim_reg_t;

/**
 * cacaosim_spec_bin_lambda() - Compute center wavelength `[um]` of spectral bin `@b`
 * @b: Zero-based bin index (`0 .. CACAOSIM_NSPEC - 1`)
 *
 * Role:
 *   Maps discrete bin index `b` to its midpoint wavelength in `[0.43125 .. 2.36875] um` for
 *   Planck blackbody evaluation, dichroic transmission curves, and FITS/SHM keyword comments.
 *
 * Rationale:
 *   Midpoint quadrature `(b + 0.5) * dlambda` is second-order accurate for smooth stellar continua
 *   and inline-folds to a constant when `b` is unrolled.
 *
 * Return: Bin center wavelength in micrometers `[um]`.
 */
static inline float cacaosim_spec_bin_lambda(int b)
{
    return CACAOSIM_SPEC_L0_UM + ((float) b + 0.5f) * CACAOSIM_SPEC_DL_UM;
}

/**
 * cacaosim_reg_is_identity() - Test whether a 2D pupil registration is an exact or near identity
 * @reg: Pointer to registration descriptor (`NULL` is treated as identity)
 *
 * Role:
 *   Guards the fast SIMD / coalesced-GPU elementwise path in `cacaosim_beam_relay_*` and
 *   `cacaosim_beam_apply_wfe_*` so aligned optical arms skip bilinear coordinate warping.
 *
 * Rationale:
 *   Bilinear resampling introduces slight numerical smoothing of high-spatial-frequency turbulence
 *   and requires 4-tap gather reads (plus temporary bounce buffers when operating in place). A
 *   `1e-5` tolerance on shift, rotation, and `|scale - 1|` guarantees nominal zero-misregistration
 *   configs take the lossless 1-to-1 copy/accumulate path.
 *
 * Return: `1` if `@reg` is `NULL` or represents an identity transform, `0` otherwise.
 */
int cacaosim_reg_is_identity(const cacaosim_reg_t *reg);

/**
 * cacaosim_spec_star() - Synthesize a 32-bin stellar blackbody photon spectrum scaled to V mag
 * @out:        Output 32-bin photon arrival rate spectrum `[photons/s]` (safe no-op if `NULL`)
 * @tel_diam_m: Primary mirror outer diameter `[meters]` (`<= 0.01f` defaults to `8.0 m`)
 * @obs_frac:   Linear central obscuration diameter ratio `[0..0.99)` (reduces area by `1 - obs^2`)
 * @star_mag:   Apparent Vega V-band magnitude `m_V` (at `0.55 um`)
 * @teff_k:     Stellar effective Planck blackbody temperature `[K]` (e.g., `5778` G2V, `3200` M4V)
 * @tput:       Combined atmosphere + telescope + pre-AO end-to-end throughput `[0..1]`
 *
 * Role:
 *   Cold-path radiometric source model invoked by `mkbeam`, `htsim`, and the `.bench` spectral
 *   analyzer (`cacaosim-CTRL` / `cacaosim-bench-view`) to compute incident stellar photons/s
 *   entering the AO system across `0.40..2.40 um`.
 *
 * Rationale:
 *   Anchoring a Planck photon distribution `N_lambda(T_eff)` to the Vega `m_V = 0` zero-point at
 *   `0.55 um` accurately captures realistic Visible-to-NIR color ratios (e.g., extreme red photon
 *   advantage of cool M-dwarfs on NIR WFS vs. solar-type stars) from just `(star_mag, teff_k)`
 *   without requiring external stellar spectral library files.
 */
void cacaosim_spec_star(
    cacaosim_spec_t *out,
    float            tel_diam_m,
    float            obs_frac,
    float            star_mag,
    float            teff_k,
    float            tput);

/**
 * cacaosim_spec_scale() - Attenuate all 32 spectral bins by an achromatic scalar throughput
 * @in:   Input photon-rate spectrum (safe no-op if `NULL`)
 * @tput: Multiplicative scalar transmission clamped to `[0.0, 1.0]`
 * @out:  Output attenuated spectrum (may alias `@in`; safe no-op if `NULL`)
 *
 * Role:
 *   Applies gray optical relay losses (mirror coatings, entrance windows, neutral-density filters)
 *   to a beam's spectral vector.
 *
 * Rationale:
 *   Clamping `@tput` to `[0, 1]` enforces passive physical energy conservation even if user
 *   parameters momentarily exceed unity.
 */
void cacaosim_spec_scale(
    const cacaosim_spec_t *in,
    float                  tput,
    cacaosim_spec_t       *out);

/**
 * cacaosim_spec_total() - Sum integrated photon arrival rate `[photons/s]` over `0.40..2.40 um`
 * @spec: Input 32-bin photon spectrum (`NULL` returns `0.0f`)
 *
 * Role:
 *   Computes broadband bolometric photon flux for SHM `CSIM_FLX` telemetry and TUI bench summaries.
 *
 * Rationale:
 *   Since each bin `ph_s[b]` already stores bin-integrated `[photons/s]` (pre-multiplied by
 *   `CACAOSIM_SPEC_DL_UM`), total rate is an exact unweighted sum across all 32 bins.
 *
 * Return: Total photon arrival rate `[photons/s]` across `[0.40, 2.40] um`.
 */
float cacaosim_spec_total(const cacaosim_spec_t *spec);

/**
 * cacaosim_spec_split() - Partition an incident photon spectrum into reflected and transmitted arms
 * @in:        Incident 32-bin photon-rate spectrum (safe no-op if `NULL`)
 * @cfg:       Beamsplitter / dichroic coating configuration (safe no-op if `NULL`)
 * @out_refl:  Destination reflected-arm spectrum (may alias `@in`, or `NULL` if unused)
 * @out_trans: Destination transmitted-arm spectrum (may alias `@in`, or `NULL` if unused)
 *
 * Role:
 *   Core spectral router used by `cacaosim.beamsplit`, fused `htsim`, and the zero-FFT `.bench`
 *   spectral analyzer to propagate stellar spectra across arbitrary cascades of gray splitters and
 *   LP / SP / BP dichroics.
 *
 * Rationale:
 *   Copies `*in` to a local struct upfront so `@out_refl` or `@out_trans` can safely alias `@in`
 *   in place, models realistic finite coating roll-off via smooth logistic edges, and normalizes
 *   `(r, t)` whenever `r + t > 1` to guarantee strict per-bin photon conservation:
 *   `out_refl->ph_s[b] + out_trans->ph_s[b] <= in->ph_s[b]`.
 */
void cacaosim_spec_split(
    const cacaosim_spec_t      *in,
    const cacaosim_split_cfg_t *cfg,
    cacaosim_spec_t            *out_refl,
    cacaosim_spec_t            *out_trans);

/**
 * cacaosim_spec_band_flux() - Integrate photoevents and centroid wavelength over a sensor bandpass
 * @spec:       Incident beam photon-rate spectrum `[photons/s]` (`NULL` returns `0.0f`)
 * @lambda0_um: Sensor filter central wavelength `[um]`
 * @bw_frac:    Fractional top-hat filter full width `dlambda / lambda0` (`<= 0.0f` returns `0.0f`)
 * @dt_exp_s:   Camera exposure / integration duration `[seconds]` (`<= 0` -> `0.0005 s`)
 * @qe:         Detector quantum efficiency `[0..1]` (`<= 0` -> `1.0`, `> 1` -> `1.0`)
 * @eff_lambda: Optional output receiving flux-weighted effective wavelength `[um]` (may be `NULL`)
 *
 * Role:
 *   Bridges upstream beam spectra (`CSIM_S00..S31`) to downstream WFS and science camera detector
 *   flux (`det.flux_ph`) and effective sensing wavelength (`eff_lambda`), automatically accounting
 *   for dichroic cutoffs, stellar color, exposure time, and QE.
 *
 * Rationale:
 *   Integrates the exact bolometric-conserving piecewise-linear spectral photon density
 *   `S_lambda(lam)` across `[lambda0 * (1 - bw/2), lambda0 * (1 + bw/2)]`, ensuring photon flux
 *   scales strictly linearly with `bw_frac` down to zero for narrowband filters and varies
 *   smoothly across dichroic transition edges without bin-quantization steps.
 *
 * Return: Integrated detected photoevents `[e- / exposure]` (or incident photons when `qe == 1`).
 */
float cacaosim_spec_band_flux(
    const cacaosim_spec_t *spec,
    float                  lambda0_um,
    float                  bw_frac,
    float                  dt_exp_s,
    float                  qe,
    float                 *eff_lambda);

/**
 * cacaosim_spec_sample_band() - Partition a sensor bandpass into flux-consistent spectral channels
 * @spec:           Optional incident 32-bin photon-rate spectrum (`NULL` -> flat uniform SED)
 * @lambda0_um:     Filter central wavelength `[um]`
 * @bw_frac:        Fractional filter full width `dlambda / lambda0` (`<= 0` -> monochromatic)
 * @nlambda:        Number of discrete propagation wavelengths (`1 .. CACAOSIM_MAX_LAMBDA`)
 * @dt_exp_s:       Camera exposure duration `[seconds]` (`<= 0` -> `0.0005 s`)
 * @qe:             Detector quantum efficiency `[0..1]` (`<= 0` -> `1.0`, `> 1` -> `1.0`)
 * @out_lam_um:     Optional output channel flux-centroid wavelengths `[nlambda]` in `[um]`
 * @out_weights:    Optional output normalized channel optical weights `[nlambda]` (`sum == 1.0f`)
 * @out_eff_lambda: Optional output overall flux-weighted effective wavelength `[um]`
 *
 * Role:
 *   Unified spectral quadrature routine shared by `pyrwfs`, `shwfs`, `zwfs`, `fpcam`, and `.bench`
 *   analysis. Partitions `[lmin, lmax]` into `@nlambda` contiguous sub-bands of width
 *   `(lmax - lmin) / nlambda`, integrating detected photoevents `Phi_c` and first moment on each
 *   sub-band so `Phi_tot = sum_c Phi_c` matches `cacaosim_spec_band_flux()` identically while
 *   producing flux-consistent optical channel weights `w_c = Phi_c / Phi_tot` and centroids.
 *
 * Rationale:
 *   On a flat spectrum (`@spec == NULL`) with `nlambda = 2`, places channels at
 *   `lambda0 +- 0.25 * dlambda` with weights `(0.5, 0.5)`. When a dichroic cuts off part of a
 *   sensor's passband or a cool M-dwarf skews photon arrival toward redder wavelengths, both
 *   `w_c` and `out_lam_um[c]` adapt automatically to the actual detected photons.
 *
 * Return: Total detected photoevents `Phi_tot` across the full passband (`0.0f` if `@spec == NULL`
 *         or `@bw_frac <= 0.0f`).
 */
float cacaosim_spec_sample_band(
    const cacaosim_spec_t *spec,
    float                  lambda0_um,
    float                  bw_frac,
    int32_t                nlambda,
    float                  dt_exp_s,
    float                  qe,
    float                 *out_lam_um,
    float                 *out_weights,
    float                 *out_eff_lambda);

/**
 * cacaosim_resolve_nlambda() - Resolve effective discrete wavelength count (`0` = auto -> `2`)
 * @cfg_nlambda: Explicit user channel count (`>= 1` overrides auto; `<= 0` selects auto)
 * @in_nlam:     Upstream polychromatic beam channel count (`1` for standard `[OPD, Amp]` beams)
 * @bw_frac:     Fractional sensor/element spectral bandwidth `dlambda / lambda0`
 *
 * Role:
 *   Enforces the uniform `N = 2` default polychromatic sampling rule across `pyrwfs`, `shwfs`,
 *   `zwfs`, `fpcam`, `coro`, and `.bench` nodes while preserving upstream multi-channel beam counts
 *   when `cfg_nlambda <= 0` and backward-compatible `N = 1` behavior when `bw_frac <= 1e-6f`.
 *
 * Return: Clamped channel count in `[1 .. CACAOSIM_MAX_LAMBDA]`.
 */
static inline int32_t cacaosim_resolve_nlambda(
    int32_t  cfg_nlambda,
    uint32_t in_nlam,
    float    bw_frac)
{
    int32_t nl = cfg_nlambda;
    if (nl <= 0)
    {
        if (in_nlam > 1u)
        {
            nl = (int32_t) in_nlam;
        }
        else if (bw_frac > 1e-6f)
        {
            nl = 2;
        }
        else
        {
            nl = 1;
        }
    }
    if (nl < 1)
    {
        nl = 1;
    }
    if (nl > CACAOSIM_MAX_LAMBDA)
    {
        nl = CACAOSIM_MAX_LAMBDA;
    }
    return nl;
}

/**
 * cacaosim_poly_interp_bracket() - Compute 1D linear wavelength interpolation bracket `(m0, m1, a)`
 * @target_lam:  Target sensor channel wavelength `[um]`
 * @beam_lam_um: Optional monotonic array `[beam_nlam]` of upstream beam wavelengths `[um]`
 * @beam_nlam:   Number of coherent channels in upstream beam (`<= 1` -> `(0, 0, 0.0f)`)
 * @def_l0:      Fallback shortest wavelength `[um]` when `@beam_lam_um == NULL`
 * @def_l1:      Fallback longest wavelength `[um]` when `@beam_lam_um == NULL`
 * @out_m0:      Output lower channel index in `[0 .. beam_nlam - 1]`
 * @out_m1:      Output upper channel index in `[0 .. beam_nlam - 1]`
 * @out_alpha:   Output linear blend weight in `[0.0f, 1.0f]`
 *
 * Role:
 *   Maps each sensor propagation wavelength `@target_lam` onto two bracketing slices `(m0, m1)` of
 *   an upstream polychromatic `[OPD_m, Amp_m]` cube so complex pupil phasors are linearly
 *   interpolated in wavelength without index-truncation steps or `atan2` phase-wrap artifacts.
 */
static inline void cacaosim_poly_interp_bracket_ex(
    float        target_lam,
    const float *beam_lam_um,
    uint32_t     beam_nlam,
    float        def_l0,
    float        def_l1,
    int32_t      use_invlam2,
    uint32_t    *out_m0,
    uint32_t    *out_m1,
    float       *out_alpha)
{
    if (beam_nlam <= 1u)
    {
        *out_m0    = 0u;
        *out_m1    = 0u;
        *out_alpha = 0.0f;
        return;
    }
    if (beam_lam_um != NULL && beam_lam_um[beam_nlam - 1u] > beam_lam_um[0])
    {
        if (target_lam <= beam_lam_um[0])
        {
            *out_m0    = 0u;
            *out_m1    = 0u;
            *out_alpha = 0.0f;
            return;
        }
        if (target_lam >= beam_lam_um[beam_nlam - 1u])
        {
            *out_m0    = beam_nlam - 1u;
            *out_m1    = beam_nlam - 1u;
            *out_alpha = 0.0f;
            return;
        }
        for (uint32_t m = 0u; m + 1u < beam_nlam; m++)
        {
            float l0 = beam_lam_um[m];
            float l1 = beam_lam_um[m + 1u];
            if (target_lam >= l0 && target_lam <= l1)
            {
                float alpha = 0.0f;
                if (use_invlam2 && l0 > 1e-4f && l1 > 1e-4f && target_lam > 1e-4f)
                {
                    float inv0 = 1.0f / (l0 * l0);
                    float inv1 = 1.0f / (l1 * l1);
                    float invt = 1.0f / (target_lam * target_lam);
                    float den  = inv0 - inv1;
                    alpha      = (den > 1e-12f) ? ((inv0 - invt) / den) : 0.0f;
                }
                else
                {
                    float dl = l1 - l0;
                    alpha    = (dl > 1e-12f) ? ((target_lam - l0) / dl) : 0.0f;
                }
                if (alpha <= 1e-6f)
                {
                    *out_m0    = m;
                    *out_m1    = m;
                    *out_alpha = 0.0f;
                }
                else if (alpha >= 1.0f - 1e-6f)
                {
                    *out_m0    = m + 1u;
                    *out_m1    = m + 1u;
                    *out_alpha = 0.0f;
                }
                else
                {
                    *out_m0    = m;
                    *out_m1    = m + 1u;
                    *out_alpha = alpha;
                }
                return;
            }
        }
    }
    if (def_l1 > def_l0 + 1e-12f)
    {
        float u = ((target_lam - def_l0) / (def_l1 - def_l0)) * (float) (beam_nlam - 1u);
        if (u <= 0.0f)
        {
            *out_m0    = 0u;
            *out_m1    = 0u;
            *out_alpha = 0.0f;
        }
        else if (u >= (float) (beam_nlam - 1u))
        {
            *out_m0    = beam_nlam - 1u;
            *out_m1    = beam_nlam - 1u;
            *out_alpha = 0.0f;
        }
        else
        {
            uint32_t m0 = (uint32_t) u;
            uint32_t m1 = (m0 + 1u < beam_nlam) ? (m0 + 1u) : m0;
            *out_m0     = m0;
            *out_m1     = m1;
            *out_alpha  = u - (float) m0;
        }
        return;
    }
    *out_m0    = 0u;
    *out_m1    = 0u;
    *out_alpha = 0.0f;
}

static inline void cacaosim_poly_interp_bracket(
    float        target_lam,
    const float *beam_lam_um,
    uint32_t     beam_nlam,
    float        def_l0,
    float        def_l1,
    uint32_t    *out_m0,
    uint32_t    *out_m1,
    float       *out_alpha)
{
    cacaosim_poly_interp_bracket_ex(target_lam, beam_lam_um, beam_nlam, def_l0, def_l1, 0, out_m0,
                                    out_m1, out_alpha);
}

/**
 * cacaosim_beam_init_cpu() - Synthesize initial `[OPD, Amp]` pupil beam on CPU from inputs
 * @pup_amp:   Static telescope pupil amplitude `[nwf * nwf]` (`NULL` defaults to uniform `1.0f`)
 * @atm_amp:   Optional atmospheric scintillation amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @wf_in:     Optional external turbulence map `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @ncpa_opd:  Optional static entrance NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @wf_to_opd: Scale factor converting `@wf_in` to `[um]` OPD (`cacaosim_wf_to_opd_scale`)
 * @nwf:       Square pupil grid dimension `[pixels]`
 * @out_opd:   Output beam OPD slice `[nwf * nwf]` in `[um]` (non-NULL)
 * @out_amp:   Output beam amplitude slice `[nwf * nwf]` (non-NULL)
 *
 * Role:
 *   Hot-path CPU beam constructor used by `cacaosim.mkbeam` and fused `htsim` to assemble the
 *   initial `[OPD_um, Amplitude]` optical beam at the telescope entrance pupil.
 *
 * Rationale:
 *   Fuses amplitude product (`pup_amp * atm_amp`) and OPD linear combination
 *   (`wf_to_opd * wf_in + ncpa_opd`) into a single OpenMP SIMD loop over `nwf^2` pixels to
 *   minimize memory bandwidth consumption.
 */
void cacaosim_beam_init_cpu(
    const float *pup_amp,
    const float *atm_amp,
    const float *wf_in,
    const float *ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *out_opd,
    float       *out_amp);

/**
 * cacaosim_beam_init_lerp_cpu() - Synthesize `[OPD, Amp]` beam with linear temporal WF lerp on CPU
 * @pup_amp:   Static telescope pupil amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @atm_amp:   Optional atmospheric scintillation amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @wf_prev:   Previous base-rate turbulence screen `[nwf * nwf]` (`NULL` -> fallback to `@wf_curr`)
 * @wf_curr:   Current base-rate turbulence screen `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @wf_alpha:  Sub-tick linear blend weight in `[0, 1]` (`0` = `@wf_prev`, `1` = `@wf_curr`)
 * @ncpa_opd:  Optional static entrance NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @wf_to_opd: Scale factor converting wavefront units to `[um]` OPD
 * @nwf:       Square pupil grid dimension `[pixels]`
 * @out_opd:   Output beam OPD slice `[nwf * nwf]` in `[um]` (non-NULL)
 * @out_amp:   Output beam amplitude slice `[nwf * nwf]` (non-NULL)
 *
 * Role:
 *   Hot-path CPU beam constructor for multi-rate oversampled benches (`oversample > 1`,
 *   `atm_interp == 1`), smoothly interpolating `1x` atmospheric screens across `K` sub-ticks.
 *
 * Rationale:
 *   Eliminates zero-order-hold screen stepping artifacts during sub-frame WFS exposure integration
 *   without increasing upstream `milkatmturb` compute or SHM bandwidth.
 */
void cacaosim_beam_init_lerp_cpu(
    const float *pup_amp,
    const float *atm_amp,
    const float *wf_prev,
    const float *wf_curr,
    float        wf_alpha,
    const float *ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *out_opd,
    float       *out_amp);

/**
 * cacaosim_beam_apply_dm_cpu() - Subtract reflective DM OPD from an optical beam on CPU
 * @in_opd:  Input beam OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:  Input beam amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @dm_opd:  Rendered DM optical path difference `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @nwf:     Square pupil grid dimension `[pixels]`
 * @out_opd: Output corrected beam OPD `[nwf * nwf]` in `[um]` (may alias `@in_opd`)
 * @out_amp: Output beam amplitude `[nwf * nwf]` (may alias `@in_amp` or be `NULL`)
 *
 * Role:
 *   Hot-path CPU inline corrector executed by `cacaosim.dmsim` (when operating on `in_beam ->
 *   out_beam`) and `htsim` to apply closed-loop DM wavefront correction.
 *
 * Rationale:
 *   Uses the standard AO negative-feedback sign convention (`out_opd = in_opd - dm_opd`, where
 *   `dm_opd` already incorporates `refl_scale`, typically `2.0` for normal-incidence reflection)
 *   and skips copying amplitude when `out_amp == in_amp` to halve memory traffic for in-place
 *   relay buffers.
 */
void cacaosim_beam_apply_dm_cpu(
    const float *in_opd,
    const float *in_amp,
    const float *dm_opd,
    uint32_t     nwf,
    float       *out_opd,
    float       *out_amp);

/**
 * cacaosim_beam_relay_cpu() - Apply 2D pupil registration, arm NCPA, and arm pupil stop on CPU
 * @in_opd:   Input beam OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:   Input beam amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @ncpa_opd: Optional post-relay static arm NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @pup_mask: Optional post-relay arm pupil/cold-stop mask `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @reg:      Optional 2D affine pupil registration (`NULL` or identity uses SIMD fast path)
 * @nwf:      Square pupil grid dimension `[pixels]`
 * @out_opd:  Output relayed OPD `[nwf * nwf]` in `[um]` (safe if aliasing `@in_opd`)
 * @out_amp:  Output relayed amplitude `[nwf * nwf]` (safe if aliasing `@in_amp`)
 *
 * Role:
 *   Hot-path CPU optical arm propagator invoked per output port of `cacaosim.beamsplit` and
 *   `htsim` relays to model misregistration, differential NCPA, and pupil stops between arms.
 *
 * Rationale:
 *   Dispatches a zero-overhead OpenMP SIMD loop when `@reg` is identity, and an OpenMP-parallel
 *   inverse-affine bilinear sampler otherwise (allocating temporary bounce buffers only when
 *   in-place aliasing is detected so parallel threads never overwrite source neighbors).
 */
void cacaosim_beam_relay_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *ncpa_opd,
    const float          *pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp);

/**
 * cacaosim_beam_apply_wfe_cpu() - Add scaled static WFE OPD map (+ optional registration) on CPU
 * @in_opd:    Input pupil OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:    Input pupil amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @wfe_opd:   Static wavefront error map `[nwf * nwf]` (e.g., in `[meter]`, `NULL` -> `0.0f`)
 * @wfe_to_um: Multiplicative conversion & gain to microns (`1.0e6f * wfe_gain` for `[m] -> [um]`)
 * @reg:       Optional 2D affine pupil registration of incoming beam (`NULL` = identity)
 * @nwf:       Square pupil grid size `[pixels]`
 * @out_opd:   Output pupil OPD `[nwf * nwf]` in `[um]` (non-NULL; distinct from `@in_opd` if warp)
 * @out_amp:   Optional output pupil amplitude `[nwf * nwf]` (may be `NULL` or alias `@in_amp` on
 *             identity)
 *
 * Role:
 *   Hot-path CPU optical element kernel backing `cacaosim.wfe` (`[wfe.<id>]`) and fused `htsim`
 *   inline aberration injection.
 *
 * Rationale:
 *   External optical metrology / Zernike WFE FITS maps in `cacao` are conventionally specified in
 *   SI `[meter]` OPD; folding `1e6 * gain` into `@wfe_to_um` converts and scales the aberration in
 *   a single FMA on top of optional pupil registration without mutating the cached static WFE map.
 */
void cacaosim_beam_apply_wfe_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *wfe_opd,
    float                 wfe_to_um,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp);

/**
 * cacaosim_beam_shm_init_kw() - Initialize fixed-slot `CSIM_BM`/`S00..S31` keywords on SHM stream
 * @im: Target 3D `IMAGE` descriptor with `im->md->NBkw >= 36` (`4 + CACAOSIM_NSPEC`)
 *
 * Role:
 *   Cold-path setup helper called when creating an output optical beam SHM stream (`mkbeam`,
 *   `wfe`, `dmsim`, `beamsplit`, `coro`) to format keyword names, types, units, and comments at
 *   fixed indices `kw[0..35]`.
 *
 * Rationale:
 *   Formatting `snprintf` keyword names and comments once during stream creation allows
 *   `cacaosim_beam_shm_write_spec()` to update only `.value.numf` doubles at `O(1)` offsets during
 *   kHz hot-loop operation with zero string formatting overhead.
 */
void cacaosim_beam_shm_init_kw(IMAGE *im);

/**
 * cacaosim_beam_shm_write_spec() - Fast `O(1)` per-frame spectral vector write into SHM keywords
 * @im:   Destination optical beam SHM `IMAGE` (`NBkw >= 36`; safe no-op if `NULL` or undersized)
 * @spec: 32-bin photon arrival rate spectrum `[photons/s]` to serialize (safe no-op if `NULL`)
 *
 * Role:
 *   Hot-path metadata updater called before `processinfo_update_output_stream()` by beam-producing
 *   runnables so downstream processes see live stellar/dichroic photometry alongside pixel updates.
 *
 * Rationale:
 *   Lazily calls `cacaosim_beam_shm_init_kw()` if `CSIM_BM` sentinel (`kw[0]`) is not yet set,
 *   then writes `kw[4..35].value.numf` and total rate `kw[3].value.numf` (`CSIM_FLX`) in 32 direct
 *   indexed stores without `snprintf` or keyword-table searches.
 */
void cacaosim_beam_shm_write_spec(
    IMAGE                 *im,
    const cacaosim_spec_t *spec);

/**
 * cacaosim_beam_shm_read_spec() - Read 32-bin photon spectrum from SHM keywords if `CSIM_BM == 1`
 * @im:   Source upstream SHM `IMAGE` (`NULL` or non-beam stream returns `0`)
 * @spec: Output 32-bin photon-rate spectrum populated on success
 *
 * Role:
 *   Called by downstream beam consumers (`wfe`, `dmsim`, `beamsplit`, `coro`, `pyrwfs`, `shwfs`,
 *   `zwfs`, `fpcam`) to inherit upstream spectral photometry automatically over SHM.
 *
 * Rationale:
 *   Verifies `NBkw >= 36` and sentinel `kw[0]` (`CSIM_BM == 1` of type `'L'`) before reading
 *   `kw[4..35]`, safely distinguishing true `cacaosim` beam streams from arbitrary 3D SHM cubes
 *   so sensors fall back cleanly to their explicit `det.flux_ph` parameter when unattached.
 *
 * Return: `1` if a valid `cacaosim` beam spectrum was read into `@spec`, `0` otherwise.
 */
int cacaosim_beam_shm_read_spec(
    const IMAGE     *im,
    cacaosim_spec_t *spec);

/**
 * cacaosim_beam_shm_is_valid() - Validate that `@im` is a compatible 3D `FLOAT32` `[OPD, Amp]` beam
 * @im:  Candidate `ImageStreamIO` `IMAGE` pointer (`NULL` returns `0`)
 * @nwf: Expected square spatial pupil grid dimension `[pixels]` along axes 0 and 1
 *
 * Role:
 *   Input stream sanity check executed by beam-consuming modules prior to dereferencing slice
 *   pointers `im->array.F + 0` (`OPD`) and `im->array.F + nwf*nwf` (`Amp`).
 *
 * Rationale:
 *   Requires `_DATATYPE_FLOAT`, `naxis == 3`, `size[0] == nwf`, `size[1] == nwf`, and
 *   `size[2] >= 2` (accepting both `[nwf, nwf, 2]` monochromatic beams and `[nwf, nwf, 2*N_lambda]`
 *   polychromatic post-coronagraph cubes) to prevent out-of-bounds SHM reads on mismatched grids.
 *
 * Return: `1` if `@im` satisfies all spatial, axis, and datatype invariants, `0` otherwise.
 */
int cacaosim_beam_shm_is_valid(
    const IMAGE *im,
    uint32_t     nwf);

/**
 * cacaosim_beam_shm_write_poly_kw() - Write multi-wavelength `[OPD_c, Amp_c]` cube keywords to SHM
 * @im:     Target 3D beam `IMAGE` with `NBkw >= 39` (`4 + CACAOSIM_NSPEC + 3`)
 * @nwl:    Number of coherent wavelength channels `N_lambda` stored along Z (`size[2] == 2 * nwl`)
 * @wl0_um: First channel (`c = 0`) wavelength `[um]`
 * @wl1_um: Last channel (`c = nwl - 1`) wavelength `[um]`
 *
 * Role:
 *   Cold/config-path metadata writer used by polychromatic `cacaosim.coro` to advertise the
 *   channel count and `[wl0_um, wl1_um]` spectral grid of `[nwf x nwf x 2*N_lambda]` output beams
 *   at fixed slots `kw[36..38]` (`CSIM_NWL`, `CSIM_WL0`, `CSIM_WL1`).
 *
 * Rationale:
 *   Chromatic focal-plane or pupil-plane coronagraphs impart wavelength-dependent amplitude and
 *   phase residuals after the Lyot stop; tagging `CSIM_NWL`/`WL0`/`WL1` right after `CSIM_S31`
 *   lets `cacaosim.fpcam` ingest all `N_lambda` channels coherently at their exact wavelengths.
 */
void cacaosim_beam_shm_write_poly_kw(
    IMAGE   *im,
    uint32_t nwl,
    float    wl0_um,
    float    wl1_um);

/**
 * cacaosim_beam_shm_read_poly_kw() - Query multi-wavelength channel count and bounds from SHM beam
 * @im:      Source 3D beam `IMAGE` (`NULL` or `size[2] < 4` is treated as monochromatic)
 * @nwl_out: Optional output receiving channel count (`>= 1`; set to `1` when monochromatic)
 * @wl0_out: Optional output receiving first channel wavelength `[um]` (`0.0f` if unspecified)
 * @wl1_out: Optional output receiving last channel wavelength `[um]` (`0.0f` if unspecified)
 *
 * Role:
 *   Queried by polychromatic-capable downstream modules (`cacaosim.coro`, `cacaosim.fpcam`) to
 *   detect whether an input SHM beam carries `N_lambda > 1` `[OPD_c, Amp_c]` slice pairs and to
 *   recover their wavelength endpoints.
 *
 * Rationale:
 *   Defaults cleanly to `nwl = 1` for standard `[nwf, nwf, 2]` beams, infers `size[2] / 2` even if
 *   keywords were omitted, and clamps `CSIM_NWL` against `size[2] / 2` so corrupt or stale keyword
 *   metadata can never cause slice over-reads past allocated SHM memory.
 *
 * Return: `1` if `@im` carries `> 1` coherent wavelength channels, `0` if monochromatic.
 */
int cacaosim_beam_shm_read_poly_kw(
    const IMAGE *im,
    uint32_t    *nwl_out,
    float       *wl0_out,
    float       *wl1_out);

/**
 * cacaosim_fpcam_shm_write_kw() - Write fixed-slot Strehl keywords to an `out_long` SHM image
 * @im:       Target 2D long-exposure PSF `IMAGE` (`NBkw >= 4`; safe no-op if NULL or undersized)
 * @sr_long:  Cumulative long-exposure peak Strehl ratio `[0..1]` (`kw[0]`: `STREHL`)
 * @sr_inst:  Most recent short-exposure peak Strehl ratio `[0..1]` (`kw[1]`: `SR_INST`)
 * @sr_axis:  Most recent short-exposure on-axis Strehl ratio `[0..1]` (`kw[2]`: `SR_AXIS`)
 * @long_cnt: Cumulative long-exposure frame count (`kw[3]`: `LONGC`)
 *
 * Role:
 *   Hot-path metadata updater called by `cacaosim.fpcam`, `cacaosim.htsim`, and the bench DAG
 *   before `processinfo_update_output_stream()` on cumulative long-exposure PSF SHM streams.
 *
 * Rationale:
 *   Assigning deterministic keyword slots `kw[0..3]` enables O(1) keyword writes inside kHz loops
 *   while preserving `kw[0]` (`STREHL`) compatibility with live script pollers and allowing
 *   `cacaosim_wfstat_core` to sample both `SR_INST(t)` and `STREHL(t)` lock-free from SHM.
 */
void cacaosim_fpcam_shm_write_kw(
    IMAGE   *im,
    float    sr_long,
    float    sr_inst,
    float    sr_axis,
    uint64_t long_cnt);

/**
 * cacaosim_fpcam_shm_read_kw() - Read fixed-slot Strehl telemetry keywords from an `out_long` SHM
 * @im:      Source 2D long-exposure PSF `IMAGE` (`NULL` or uninitialized returns `0`)
 * @sr_long: Optional output receiving cumulative long-exposure peak Strehl ratio
 * @sr_inst: Optional output receiving short-exposure peak Strehl ratio (falls back to `@sr_long`)
 * @sr_axis: Optional output receiving short-exposure on-axis Strehl (falls back to `@sr_long`)
 *
 * Role:
 *   Queried by `cacaosim_wfstat_frame()` when `img->md->cnt0` advances to record synchronized
 *   instantaneous and cumulative Strehl time series without copying full 2D PSF images.
 *
 * Return: `1` if valid `STREHL` metadata was found, `0` otherwise.
 */
int cacaosim_fpcam_shm_read_kw(
    const IMAGE *im,
    double      *sr_long,
    double      *sr_inst,
    double      *sr_axis);

/**
 * cacaosim_coro_shm_write_kw() - Write fixed-slot power-budget keywords to a coronagraph 2D SHM
 * @im:           Target 2D pre-Lyot intensity `IMAGE` (`NBkw >= 7`; no-op if NULL or undersized)
 * @f_lowfs_long: Cumulative fraction of coronagraph input entering `out_lowfs` (`kw[0]`: `F_LOWFS`)
 * @f_lowfs_inst: Instantaneous fraction entering `out_lowfs` (`kw[1]`: `F_LOWFSI`)
 * @f_pre_long:   Cumulative fraction incident on the Lyot stop (`kw[2]`: `F_PRELYO`)
 * @f_pre_inst:   Instantaneous fraction incident on the Lyot stop (`kw[3]`: `F_PRELYI`)
 * @f_trans_long: Cumulative fraction transmitted past the Lyot stop (`kw[4]`: `F_LYOTTR`)
 * @f_trans_inst: Instantaneous fraction transmitted past the Lyot stop (`kw[5]`: `F_LYOTTI`)
 * @long_cnt:     Cumulative coronagraph frame count (`kw[6]`: `LONGC`)
 */
void cacaosim_coro_shm_write_kw(
    IMAGE   *im,
    float    f_lowfs_long,
    float    f_lowfs_inst,
    float    f_pre_long,
    float    f_pre_inst,
    float    f_trans_long,
    float    f_trans_inst,
    uint64_t long_cnt);

/**
 * cacaosim_coro_shm_read_kw() - Read fixed-slot coronagraph & LOWFS power-budget SHM keywords
 * @im:           Source 2D pre-Lyot intensity `IMAGE` (`NULL` returns `0`)
 * @f_lowfs_long: Optional output for cumulative LOWFS beam power fraction
 * @f_lowfs_inst: Optional output for instantaneous LOWFS beam power fraction
 * @f_pre_long:   Optional output for cumulative pre-Lyot incident power fraction
 * @f_pre_inst:   Optional output for instantaneous pre-Lyot incident power fraction
 * @f_trans_long: Optional output for cumulative post-Lyot transmitted power fraction
 * @f_trans_inst: Optional output for instantaneous post-Lyot transmitted power fraction
 *
 * Return: `1` if valid `F_LOWFS` metadata was read, `0` otherwise.
 */
int cacaosim_coro_shm_read_kw(
    const IMAGE *im,
    double      *f_lowfs_long,
    double      *f_lowfs_inst,
    double      *f_pre_long,
    double      *f_pre_inst,
    double      *f_trans_long,
    double      *f_trans_inst);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BEAM_H */
