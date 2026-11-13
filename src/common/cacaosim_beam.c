// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_beam.c
 * @brief   CPU implementation of Planck photometry, dichroic splitting, 2D beam relay, and SHM I/O
 *
 * Architectural Role:
 *   Implements the CPU execution and metadata routines declared in `cacaosim_beam.h`:
 *     1. Analytical stellar Planck photon radiometry (`cacaosim_spec_star`), gray attenuation,
 *        dichroic/beamsplitter spectral partitioning (`cacaosim_spec_split`), and sub-bin sensor
 *        filter integration (`cacaosim_spec_band_flux`).
 *     2. OpenMP + SIMD pupil-plane `[OPD_um, Amplitude]` synthesis, DM subtraction, inverse-affine
 *        bilinear 2D relay registration, and scaled static WFE injection.
 *     3. Fixed-slot `O(1)` `ImageStreamIO` SHM keyword serialization (`CSIM_BM`, `CSIM_L0`,
 *        `CSIM_L1`, `CSIM_FLX`, `CSIM_S00..S31`, `CSIM_NWL`, `CSIM_WL0`, `CSIM_WL1`).
 *
 * Design Rationale & Motivation:
 *   - Numerically stable Planck photometry: Evaluates photon spectral density `1 / (lambda^4 *
 *     (exp(hc / (lambda * k_B * T)) - 1))` using `expm1f(x)` so long-wavelength / high-temperature
 *     Rayleigh-Jeans evaluations (`x << 1`) do not suffer catastrophic cancellation, while clamping
 *     deep Wien-tail arguments (`x > 80`) to zero to avoid `expf` overflow (`Inf`).
 *   - Realistic smooth dichroic coatings: Models dielectric cutoff edges with a logistic sigmoid
 *     `1 / (1 + exp(4 * (lam - cut) / roll))` rather than discontinuous step functions, capturing
 *     spectral cross-talk near the dichroic transition while strictly enforcing `R(lam) + T(lam)
 *     <= 1` per bin.
 *   - Race-free OpenMP bilinear pupil warping: Inverse destination-to-source mapping (`dst -> src`)
 *     guarantees every output pixel is written by exactly one thread without holes or atomic
 *     splats; when `out` aliases `in` (`cacaosim_beam_relay_cpu`), temporary aligned bounce
 *     buffers prevent parallel read-after-write hazards.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "milk_config.h"
#include "milk_types.h"
#include "common/cacaosim_beam.h"

/**
 * cacaosim_reg_is_identity() - Determine whether a 2D affine pupil registration is a no-op
 * @reg: Candidate registration parameters (`NULL` is treated as identity)
 *
 * Role:
 *   Fast-path predicate queried by both CPU (`cacaosim_beam_relay_cpu`,
 *   `cacaosim_beam_apply_wfe_cpu`) and CUDA (`cacaosim_cuda_beam_relay`,
 *   `cacaosim_cuda_beam_apply_wfe`) dispatchers before choosing between 1-to-1 elementwise and
 *   bilinear warp kernels.
 *
 * Rationale:
 *   Treats `scale <= 1e-4f` (e.g., default zero-initialized structs where `scale == 0.0f`) as
 *   unity magnification `1.0f`, and compares translation, rotation, and scale deviation against
 *   `1e-5f` so floating-point config round-trips do not accidentally trigger expensive warping.
 *
 * Return: `1` if identity within `1e-5`, `0` if non-trivial warping is required.
 */
int cacaosim_reg_is_identity(const cacaosim_reg_t *reg)
{
    if (reg == NULL)
    {
        return 1;
    }
    /* Uninitialized / zero scale maps to 1.0f by convention across .bench and FPS configs */
    float sc = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
    return (fabsf(reg->shift_x) < 1e-5f && fabsf(reg->shift_y) < 1e-5f &&
            fabsf(reg->rot_deg) < 1e-5f && fabsf(sc - 1.0f) < 1e-5f)
               ? 1
               : 0;
}

/**
 * planck_photon_rel() - Relative Planck photon spectral density `N_lambda(lam) / N_lambda(0.55 um)`
 * @lam_um: Wavelength at which to evaluate photon density `[um]` (`> 0`)
 * @teff_k: Stellar effective blackbody temperature `[K]` (`<= 500 K` defaults to solar `5778 K`)
 *
 * Role:
 *   Internal helper called by `cacaosim_spec_star()` to compute the dimensionless spectral shape
 *   normalized to unity at astronomical V-band (`0.55 um`).
 *
 * Rationale:
 *   Planck energy radiance `B_lambda ~ lambda^-5 / (exp(x) - 1)` divided by photon energy
 *   `hc / lambda` gives photon spectral density `N_lambda ~ lambda^-4 / (exp(x) - 1)` with second
 *   radiation constant `hc / k_B = 14387.77 um * K`. Normalizing by `N_lambda(0.55 um)` cancels
 *   all leading physical constants (`2*c`, solid angle) so V-band Vega zero-point scaling applies
 *   directly.
 *
 * Return: Dimensionless photon-rate ratio relative to `0.55 um` (`0.0f` in extreme Wien tail).
 */
static float planck_photon_rel(
    float lam_um,
    float teff_k)
{
    /* Second radiation constant h*c / k_B = 14387.77 um * K */
    const float hc_k = 14387.77f;
    float       T    = (teff_k > 500.0f) ? teff_k : 5778.0f;
    float       x    = hc_k / (lam_um * T);
    float       x0   = hc_k / (0.55f * T);
    /* Prevent expm1f float32 overflow (~88.7) in the deep UV/cool-star Wien regime */
    if (x > 80.0f)
    {
        return 0.0f;
    }
    float n_lam = 1.0f / (lam_um * lam_um * lam_um * lam_um * expm1f(x));
    float n_v   = 1.0f / (0.55f * 0.55f * 0.55f * 0.55f * expm1f(x0));
    return (n_v > 0.0f) ? (n_lam / n_v) : 1.0f;
}

/**
 * cacaosim_spec_star() - Populate 32-bin stellar blackbody photon spectrum from V-band magnitude
 * @out:        Output 32-bin photon arrival rate spectrum `[photons/s]` (no-op if `NULL`)
 * @tel_diam_m: Primary mirror outer diameter `[m]` (`<= 0.01f` -> `8.0 m` default)
 * @obs_frac:   Linear central obscuration ratio `[0..0.99)`
 * @star_mag:   Apparent Vega V-band magnitude `m_V`
 * @teff_k:     Effective stellar temperature `[K]`
 * @tput:       End-to-end pre-bench throughput `[0..1]` (`< 0` -> `1.0`)
 *
 * Role:
 *   Cold-path (`MILK_COLD`) stellar source generator populating the initial `cacaosim_spec_t`
 *   vector at the entrance of an optical bench.
 *
 * Rationale:
 *   Combines annular collecting area `pi/4 * D^2 * (1 - obs^2)`, Pogson magnitude scaling
 *   `10^(-0.4 * m_V)` anchored to Vega's `1.0e11 ph / (s * m^2 * um)` V-band zero-point, bin width
 *   `CACAOSIM_SPEC_DL_UM`, and `planck_photon_rel()` so every bin stores directly usable
 *   integrated `[photons / second]`.
 */
void MILK_COLD cacaosim_spec_star(
    cacaosim_spec_t *out,
    float            tel_diam_m,
    float            obs_frac,
    float            star_mag,
    float            teff_k,
    float            tput)
{
    if (out == NULL)
    {
        return;
    }
    float diam = (tel_diam_m > 0.01f) ? tel_diam_m : 8.0f;
    float obs  = (obs_frac > 0.0f && obs_frac < 0.99f) ? obs_frac : 0.0f;
    /* Annular telescope collecting area [m^2] */
    float area = 0.25f * CACAOSIM_PI_F * diam * diam * (1.0f - obs * obs);
    float tp   = (tput >= 0.0f) ? tput : 1.0f;

    /* Vega m_V = 0 zero-point: 1.0e11 photons / (s * m^2 * um) at 0.55 um */
    float f0_v     = 1.0e11f * powf(10.0f, -0.4f * star_mag);
    float base_flx = f0_v * area * tp * CACAOSIM_SPEC_DL_UM;

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        float lam      = cacaosim_spec_bin_lambda(b);
        out->ph_s[b]   = base_flx * planck_photon_rel(lam, teff_k);
    }
}

/**
 * cacaosim_spec_scale() - Scale all 32 spectral bins by an achromatic scalar throughput
 * @in:   Source photon-rate spectrum (no-op if `NULL`)
 * @tput: Scalar transmission factor (clamped to `[0.0f, 1.0f]`)
 * @out:  Destination photon-rate spectrum (may alias `@in`; no-op if `NULL`)
 *
 * Role:
 *   Applies gray transmission efficiency across all spectral channels (e.g., reflective/coro loss).
 *
 * Rationale:
 *   Enforces passive optical bounds `0 <= g <= 1` and supports safe in-place updates (`out == in`).
 */
void cacaosim_spec_scale(
    const cacaosim_spec_t *in,
    float                  tput,
    cacaosim_spec_t       *out)
{
    if (in == NULL || out == NULL)
    {
        return;
    }
    float g = (tput > 0.0f) ? ((tput <= 1.0f) ? tput : 1.0f) : 0.0f;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        out->ph_s[b] = g * in->ph_s[b];
    }
}

/**
 * cacaosim_spec_total() - Compute integrated broadband photon rate across `0.40..2.40 um`
 * @spec: Source 32-bin photon spectrum (`NULL` returns `0.0f`)
 *
 * Role:
 *   Aggregates total `[photons/s]` in a beam for telemetry (`CSIM_FLX`) and TUI bench inspection.
 *
 * Rationale:
 *   Direct linear summation over the 32 pre-integrated `0.0625 um` photon-rate bins.
 *
 * Return: Total photon arrival rate `[photons/s]`.
 */
float cacaosim_spec_total(const cacaosim_spec_t *spec)
{
    if (spec == NULL)
    {
        return 0.0f;
    }
    float sum = 0.0f;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        sum += spec->ph_s[b];
    }
    return sum;
}

/**
 * smooth_step_lo() - Smooth logistic short-wavelength pass profile `1 / (1 + exp(4*(lam - cut)/w))`
 * @lam:  Evaluation wavelength `[um]`
 * @cut:  50% transition cutoff wavelength `[um]`
 * @roll: Characteristic transition edge width `[um]` (`<= 1e-4f` defaults to `0.02 um`)
 *
 * Role:
 *   Internal spectral edge primitive composed by `cacaosim_spec_split()` to build LP, SP, and BP
 *   dichroic reflectance/transmittance curves.
 *
 * Rationale:
 *   With slope factor `4 / w`, the logistic curve drops from `~0.88` at `cut - w/2` to `~0.12` at
 *   `cut + w/2`, closely matching multi-layer dielectric dichroic transition slopes while remaining
 *   infinitely differentiable and clamped for `|z| > 40` to avoid `expf` overflow/underflow.
 *
 * Return: Smooth short-wavelength weight in `[0.0f, 1.0f]` (`1` for `lam << cut`, `0` for `>>`).
 */
static float smooth_step_lo(
    float lam,
    float cut,
    float roll)
{
    float w = (roll > 1e-4f) ? roll : 0.02f;
    float z = 4.0f * (lam - cut) / w;
    if (z > 40.0f)
    {
        return 0.0f;
    }
    if (z < -40.0f)
    {
        return 1.0f;
    }
    return 1.0f / (1.0f + expf(z));
}

/**
 * cacaosim_spec_split() - Split an input photon spectrum into reflected and transmitted arms
 * @in:        Incident 32-bin photon spectrum (no-op if `NULL`)
 * @cfg:       Beamsplitter / dichroic coating configuration (no-op if `NULL`)
 * @out_refl:  Optional output reflected-arm spectrum (may alias `@in`)
 * @out_trans: Optional output transmitted-arm spectrum (may alias `@in`)
 *
 * Role:
 *   Evaluates per-bin reflectance `r(lam)` and transmittance `t(lam)` for gray, LP, SP, or BP
 *   beamsplitters in `cacaosim.beamsplit`, `htsim`, and `.bench` photometry analysis.
 *
 * Rationale:
 *   Snapshots `*in` into `src` first so callers may overwrite `@in` in place with either output
 *   arm. For band-reflect (`BP`), multiplies complementary rising and falling logistic edges
 *   `s1 * s2` around `[lambda_cut - hw, lambda_cut + hw]`. Finally enforces `r + t <= 1.0f` via
 *   proportional renormalization so misconfigured user gray values (e.g., `0.6 + 0.6`) can never
 *   create unphysical photons out of nothing.
 */
void cacaosim_spec_split(
    const cacaosim_spec_t      *in,
    const cacaosim_split_cfg_t *cfg,
    cacaosim_spec_t            *out_refl,
    cacaosim_spec_t            *out_trans)
{
    if (in == NULL || cfg == NULL)
    {
        return;
    }
    /* Copy input upfront so out_refl or out_trans may safely alias `in` */
    cacaosim_spec_t src = *in;

    float er = (cfg->eff_refl > 0.0f) ? ((cfg->eff_refl <= 1.0f) ? cfg->eff_refl : 1.0f) : 0.0f;
    float et = (cfg->eff_trans > 0.0f) ? ((cfg->eff_trans <= 1.0f) ? cfg->eff_trans : 1.0f) : 0.0f;

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        float lam = cacaosim_spec_bin_lambda(b);
        float r   = 0.0f;
        float t   = 0.0f;

        if (cfg->mode == CACAOSIM_SPLIT_GRAY)
        {
            r = (cfg->r_gray > 0.0f) ? cfg->r_gray : 0.0f;
            t = (cfg->t_gray > 0.0f) ? cfg->t_gray : 0.0f;
        }
        else if (cfg->mode == CACAOSIM_SPLIT_DICHROIC_LP)
        {
            /* Long-pass: short wavelengths reflect, long wavelengths transmit */
            float s = smooth_step_lo(lam, cfg->lambda_cut, cfg->roll_um);
            r       = er * s;
            t       = et * (1.0f - s);
        }
        else if (cfg->mode == CACAOSIM_SPLIT_DICHROIC_SP)
        {
            /* Short-pass: long wavelengths reflect, short wavelengths transmit */
            float s = smooth_step_lo(lam, cfg->lambda_cut, cfg->roll_um);
            r       = er * (1.0f - s);
            t       = et * s;
        }
        else
        {
            /* Band-reflect: product of rising edge at (cut - hw) and falling edge at (cut + hw) */
            float hw = 0.5f * ((cfg->band_width > 0.01f) ? cfg->band_width : 0.20f);
            float s1 = 1.0f - smooth_step_lo(lam, cfg->lambda_cut - hw, cfg->roll_um);
            float s2 = smooth_step_lo(lam, cfg->lambda_cut + hw, cfg->roll_um);
            float bp = s1 * s2;
            r        = er * bp;
            t        = et * (1.0f - bp);
        }

        /* Guarantee strict per-bin passive energy conservation: r + t <= 1 */
        if (r + t > 1.0f)
        {
            float inv = 1.0f / (r + t);
            r *= inv;
            t *= inv;
        }

        if (out_refl != NULL)
        {
            out_refl->ph_s[b] = r * src.ph_s[b];
        }
        if (out_trans != NULL)
        {
            out_trans->ph_s[b] = t * src.ph_s[b];
        }
    }
}

/**
 * cacaosim_spec_band_flux() - Integrate detected photoevents and centroid wavelength in a bandpass
 * @spec:       Incident 32-bin photon-rate spectrum `[photons/s]` (`NULL` returns `0.0f`)
 * @lambda0_um: Filter central wavelength `[um]`
 * @bw_frac:    Fractional filter width `dlambda / lambda0` (`<= 1e-3f` defaults to `0.15`)
 * @dt_exp_s:   Exposure duration `[s]` (`<= 0` defaults to `0.0005 s` = 2 kHz)
 * @qe:         Detector quantum efficiency `[0..1]` (`<= 0` defaults to `1.0`)
 * @eff_lambda: Optional output receiving flux-weighted effective wavelength `[um]`
 *
 * Role:
 *   Computes per-frame camera `flux_ph` and `eff_lambda` for WFS and science sensors fed by a
 *   spectral beam, as well as photon-per-frame tables in `.bench` TUI/CLI diagnostics.
 *
 * Rationale:
 *   Intersects `[lmin, lmax]` continuously with each bin `[b0, b1]` (`frac = (hi - lo) / dl`) and
 *   weights each sub-interval by its exact midpoint `0.5 * (lo + hi)`, preventing quantization
 *   steps when sweeping narrowband filters across the `0.0625 um` spectral grid.
 *
 * Return: Integrated photoevents (`sum_rate * dt_exp_s * qe`) over the exposure.
 */
float cacaosim_spec_band_flux(
    const cacaosim_spec_t *spec,
    float                  lambda0_um,
    float                  bw_frac,
    float                  dt_exp_s,
    float                  qe,
    float                 *eff_lambda)
{
    if (spec == NULL)
    {
        if (eff_lambda != NULL)
        {
            *eff_lambda = lambda0_um;
        }
        return 0.0f;
    }
    float bw   = (bw_frac > 1e-3f) ? bw_frac : 0.15f;
    float lmin = lambda0_um * (1.0f - 0.5f * bw);
    float lmax = lambda0_um * (1.0f + 0.5f * bw);
    float dt   = (dt_exp_s > 0.0f) ? dt_exp_s : 0.0005f;
    float q    = (qe > 0.0f) ? ((qe <= 1.0f) ? qe : 1.0f) : 1.0f;

    float sum_rate = 0.0f;
    float sum_wlam = 0.0f;

    /* Accumulate fractional bin overlap and first wavelength moment across all 32 bins */
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        float b0 = CACAOSIM_SPEC_L0_UM + (float) b * CACAOSIM_SPEC_DL_UM;
        float b1 = b0 + CACAOSIM_SPEC_DL_UM;
        float lo = (b0 > lmin) ? b0 : lmin;
        float hi = (b1 < lmax) ? b1 : lmax;
        if (hi > lo)
        {
            float frac = (hi - lo) / CACAOSIM_SPEC_DL_UM;
            float r    = frac * spec->ph_s[b];
            sum_rate += r;
            sum_wlam += r * 0.5f * (lo + hi);
        }
    }

    if (eff_lambda != NULL)
    {
        *eff_lambda = (sum_rate > 1e-20f) ? (sum_wlam / sum_rate) : lambda0_um;
    }
    return sum_rate * dt * q;
}

/**
 * cacaosim_beam_init_cpu() - Assemble initial `[OPD, Amp]` pupil beam on CPU with OpenMP + SIMD
 * @pup_amp:   Static telescope pupil amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @atm_amp:   Optional scintillation amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @wf_in:     Optional input turbulence map `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @ncpa_opd:  Optional static entrance NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @wf_to_opd: Scale converting `@wf_in` to `[um]` OPD
 * @nwf:       Pupil grid dimension `[pixels]`
 * @out_opd:   Destination OPD array `[nwf * nwf]` in `[um]` (`restrict`-qualified)
 * @out_amp:   Destination amplitude array `[nwf * nwf]` (`restrict`-qualified)
 *
 * Role:
 *   Hot-path (`MILK_HOT`) CPU beam synthesizer invoked per frame in `mkbeam` and `htsim`.
 *
 * Rationale:
 *   Gating OpenMP parallelization on `npix >= 8192` (`> 64x64`) avoids thread-barrier overhead on
 *   small test grids while saturating memory bandwidth via SIMD vectorization on ELT-scale grids.
 */
void MILK_HOT cacaosim_beam_init_cpu(
    const float *__restrict pup_amp,
    const float *__restrict atm_amp,
    const float *__restrict wf_in,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd,
    uint32_t                nwf,
    float *__restrict       out_opd,
    float *__restrict       out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

#pragma omp parallel for simd schedule(static) if (npix >= 8192)
    for (uint64_t p = 0; p < npix; p++)
    {
        float a = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
        if (atm_amp != NULL)
        {
            a *= atm_amp[p];
        }
        float o = (wf_in != NULL) ? (wf_to_opd * wf_in[p]) : 0.0f;
        if (ncpa_opd != NULL)
        {
            o += ncpa_opd[p];
        }
        out_opd[p] = o;
        out_amp[p] = a;
    }
}

/**
 * cacaosim_beam_apply_dm_cpu() - Subtract rendered DM OPD surface from a pupil beam on CPU
 * @in_opd:  Input beam OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:  Input beam amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @dm_opd:  DM surface OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @nwf:     Pupil grid dimension `[pixels]`
 * @out_opd: Output residual OPD `[nwf * nwf]` in `[um]` (may alias `@in_opd`)
 * @out_amp: Output amplitude `[nwf * nwf]` (skipped if `NULL` or aliasing `@in_amp`)
 *
 * Role:
 *   Hot-path (`MILK_HOT`) CPU wavefront correction step in `cacaosim.dmsim` and fused `htsim`.
 *
 * Rationale:
 *   Applies negative-feedback subtraction `out_opd = in_opd - dm_opd` in a single OpenMP SIMD
 *   sweep and elides redundant amplitude memory traffic when operating in-place (`out_amp ==
 *   in_amp`).
 */
void MILK_HOT cacaosim_beam_apply_dm_cpu(
    const float *__restrict in_opd,
    const float *__restrict in_amp,
    const float *__restrict dm_opd,
    uint32_t                nwf,
    float *__restrict       out_opd,
    float *__restrict       out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

#pragma omp parallel for simd schedule(static) if (npix >= 8192)
    for (uint64_t p = 0; p < npix; p++)
    {
        float o    = (in_opd != NULL) ? in_opd[p] : 0.0f;
        float d    = (dm_opd != NULL) ? dm_opd[p] : 0.0f;
        out_opd[p] = o - d;
        if (out_amp != NULL && out_amp != in_amp)
        {
            out_amp[p] = (in_amp != NULL) ? in_amp[p] : 1.0f;
        }
    }
}

/**
 * cacaosim_beam_relay_cpu() - Propagate a pupil beam across an arm with registration, NCPA, & mask
 * @in_opd:   Input pupil OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:   Input pupil amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @ncpa_opd: Optional post-relay static arm NCPA OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @pup_mask: Optional post-relay arm pupil stop mask `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @reg:      Optional 2D affine pupil misregistration (`NULL` -> identity)
 * @nwf:      Pupil grid dimension `[pixels]`
 * @out_opd:  Destination pupil OPD `[nwf * nwf]` in `[um]` (aliasing-safe)
 * @out_amp:  Destination pupil amplitude `[nwf * nwf]` (aliasing-safe)
 *
 * Role:
 *   Hot-path (`MILK_HOT`) CPU arm relay used by `cacaosim.beamsplit` and `htsim` to apply per-arm
 *   alignment shifts/rotations/magnification, differential NCPA, and cold-stop vignetting.
 *
 * Rationale:
 *   When `@reg` is non-identity, evaluates the exact inverse similarity transform around grid
 *   center `ctr = 0.5 * (nwf - 1)`:
 *     `v = dst - ctr - shift`,  `src = ctr + (1 / scale) * R(-rot_deg) * v`
 *   followed by 4-point bilinear interpolation with zero-padding outside `[0, nwf - 1]^2` (cold
 *   field stop) and post-warp addition/multiplication of downstream `ncpa_opd` and `pup_mask`
 *   fixed in the destination arm's coordinate frame.
 */
void MILK_HOT cacaosim_beam_relay_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *ncpa_opd,
    const float          *pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

    /* Fast path: aligned pupil grids execute a single coalesced SIMD pass */
    if (cacaosim_reg_is_identity(reg))
    {
#pragma omp parallel for simd schedule(static) if (npix >= 8192)
        for (uint64_t p = 0; p < npix; p++)
        {
            float o = (in_opd != NULL) ? in_opd[p] : 0.0f;
            float a = (in_amp != NULL) ? in_amp[p] : 1.0f;
            if (ncpa_opd != NULL)
            {
                o += ncpa_opd[p];
            }
            if (pup_mask != NULL)
            {
                a *= pup_mask[p];
            }
            out_opd[p] = o;
            out_amp[p] = a;
        }
        return;
    }

    /* Precompute inverse rotation + inverse magnification matrix around pupil grid midpoint */
    float sc   = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
    float invs = 1.0f / sc;
    float rad  = -reg->rot_deg * (CACAOSIM_PI_F / 180.0f);
    float c    = cosf(rad) * invs;
    float s    = sinf(rad) * invs;
    float ctr  = 0.5f * (float) (nwf - 1u);
    float dx   = reg->shift_x;
    float dy   = reg->shift_y;

    /* Allocate temporary bounce buffers only if caller requested in-place warping */
    float *tmp_opd = NULL;
    float *tmp_amp = NULL;
    if (out_opd == in_opd || out_amp == in_amp)
    {
        tmp_opd = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        tmp_amp = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
    }
    float *dst_o = (tmp_opd != NULL) ? tmp_opd : out_opd;
    float *dst_a = (tmp_amp != NULL) ? tmp_amp : out_amp;

#pragma omp parallel for schedule(static) if (npix >= 4096)
    for (uint32_t iy = 0; iy < nwf; iy++)
    {
        float vy = (float) iy - ctr - dy;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            /* Map destination pixel (ix, iy) back into upstream source coordinates (sx, sy) */
            float    vx = (float) ix - ctr - dx;
            float    sx = ctr + c * vx - s * vy;
            float    sy = ctr + s * vx + c * vy;
            uint64_t p  = (uint64_t) iy * nwf + ix;

            float o = 0.0f;
            float a = 0.0f;
            if (sx >= 0.0f && sy >= 0.0f && sx <= (float) (nwf - 1u) && sy <= (float) (nwf - 1u))
            {
                /* 2x2 bilinear interpolation weights within source pupil bounds */
                uint32_t x0 = (uint32_t) sx;
                uint32_t y0 = (uint32_t) sy;
                uint32_t x1 = (x0 + 1u < nwf) ? (x0 + 1u) : x0;
                uint32_t y1 = (y0 + 1u < nwf) ? (y0 + 1u) : y0;
                float    fx = sx - (float) x0;
                float    fy = sy - (float) y0;
                float    w00 = (1.0f - fx) * (1.0f - fy);
                float    w10 = fx * (1.0f - fy);
                float    w01 = (1.0f - fx) * fy;
                float    w11 = fx * fy;

                uint64_t i00 = (uint64_t) y0 * nwf + x0;
                uint64_t i10 = (uint64_t) y0 * nwf + x1;
                uint64_t i01 = (uint64_t) y1 * nwf + x0;
                uint64_t i11 = (uint64_t) y1 * nwf + x1;

                if (in_opd != NULL)
                {
                    o = w00 * in_opd[i00] + w10 * in_opd[i10] + w01 * in_opd[i01] +
                        w11 * in_opd[i11];
                }
                if (in_amp != NULL)
                {
                    a = w00 * in_amp[i00] + w10 * in_amp[i10] + w01 * in_amp[i01] +
                        w11 * in_amp[i11];
                }
                else
                {
                    a = 1.0f;
                }
            }
            /* Apply downstream arm's static NCPA and pupil stop in destination coordinates */
            if (ncpa_opd != NULL)
            {
                o += ncpa_opd[p];
            }
            if (pup_mask != NULL)
            {
                a *= pup_mask[p];
            }
            dst_o[p] = o;
            dst_a[p] = a;
        }
    }

    if (tmp_opd != NULL)
    {
        memcpy(out_opd, tmp_opd, npix * sizeof(float));
        memcpy(out_amp, tmp_amp, npix * sizeof(float));
        free(tmp_opd);
        free(tmp_amp);
    }
}

/**
 * cacaosim_beam_apply_wfe_cpu() - Add scaled static WFE OPD (+ optional 2D registration) on CPU
 * @in_opd:    Input beam OPD `[nwf * nwf]` in `[um]` (`NULL` -> `0.0f`)
 * @in_amp:    Input beam amplitude `[nwf * nwf]` (`NULL` -> `1.0f`)
 * @wfe_opd:   Static wavefront error map `[nwf * nwf]` (`NULL` -> `0.0f`)
 * @wfe_to_um: Scale factor converting `@wfe_opd` units (e.g., `[meter]`) + gain into `[um]`
 * @reg:       Optional 2D affine pupil registration (`NULL` = identity)
 * @nwf:       Pupil grid dimension `[pixels]`
 * @out_opd:   Output beam OPD `[nwf * nwf]` in `[um]` (non-NULL)
 * @out_amp:   Optional output beam amplitude `[nwf * nwf]`
 *
 * Role:
 *   Hot-path (`MILK_HOT`) CPU wavefront error injector backing `cacaosim.wfe` and `htsim` `[wfe]`
 *   inline optical elements.
 *
 * Rationale:
 *   Separates the common registered case (`cacaosim_reg_is_identity(reg)`) into a pure OpenMP SIMD
 *   FMA pass (`o + wfe_to_um * wfe_opd[p]`) that avoids rewriting `out_amp` when `out_amp ==
 *   in_amp`, while supporting full inverse-affine bilinear resampling of the incoming beam onto
 *   the WFE optic's coordinate grid when misregistered.
 */
void MILK_HOT cacaosim_beam_apply_wfe_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *wfe_opd,
    float                 wfe_to_um,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp)
{
    const uint64_t npix = (uint64_t) nwf * nwf;

    /* Identity registration fast path: pure OpenMP SIMD elementwise accumulation */
    if (cacaosim_reg_is_identity(reg))
    {
#pragma omp parallel for simd schedule(static) if (npix >= 8192)
        for (uint64_t p = 0; p < npix; p++)
        {
            float o    = (in_opd != NULL) ? in_opd[p] : 0.0f;
            float a    = (in_amp != NULL) ? in_amp[p] : 1.0f;
            float w    = (wfe_opd != NULL) ? (wfe_to_um * wfe_opd[p]) : 0.0f;
            out_opd[p] = o + w;
            if (out_amp != NULL && out_amp != in_amp)
            {
                out_amp[p] = a;
            }
        }
        return;
    }

    /* Inverse similarity transform from WFE element grid back to upstream beam coordinates */
    float sc   = (reg->scale > 1e-4f) ? reg->scale : 1.0f;
    float invs = 1.0f / sc;
    float rad  = -reg->rot_deg * (CACAOSIM_PI_F / 180.0f);
    float c    = cosf(rad) * invs;
    float s    = sinf(rad) * invs;
    float ctr  = 0.5f * (float) (nwf - 1u);
    float dx   = reg->shift_x;
    float dy   = reg->shift_y;

#pragma omp parallel for schedule(static) if (npix >= 4096)
    for (uint32_t iy = 0; iy < nwf; iy++)
    {
        float vy = (float) iy - ctr - dy;
        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            float    vx = (float) ix - ctr - dx;
            float    sx = ctr + c * vx - s * vy;
            float    sy = ctr + s * vx + c * vy;
            uint64_t p  = (uint64_t) iy * nwf + ix;

            float o = 0.0f;
            float a = 0.0f;
            if (sx >= 0.0f && sy >= 0.0f && sx <= (float) (nwf - 1u) && sy <= (float) (nwf - 1u))
            {
                uint32_t x0  = (uint32_t) sx;
                uint32_t y0  = (uint32_t) sy;
                uint32_t x1  = (x0 + 1u < nwf) ? (x0 + 1u) : x0;
                uint32_t y1  = (y0 + 1u < nwf) ? (y0 + 1u) : y0;
                float    fx  = sx - (float) x0;
                float    fy  = sy - (float) y0;
                float    w00 = (1.0f - fx) * (1.0f - fy);
                float    w10 = fx * (1.0f - fy);
                float    w01 = (1.0f - fx) * fy;
                float    w11 = fx * fy;

                uint64_t i00 = (uint64_t) y0 * nwf + x0;
                uint64_t i10 = (uint64_t) y0 * nwf + x1;
                uint64_t i01 = (uint64_t) y1 * nwf + x0;
                uint64_t i11 = (uint64_t) y1 * nwf + x1;

                if (in_opd != NULL)
                {
                    o = w00 * in_opd[i00] + w10 * in_opd[i10] + w01 * in_opd[i01] +
                        w11 * in_opd[i11];
                }
                a = (in_amp != NULL)
                        ? (w00 * in_amp[i00] + w10 * in_amp[i10] + w01 * in_amp[i01] +
                           w11 * in_amp[i11])
                        : 1.0f;
            }
            /* Superimpose scaled WFE map defined in the optical element's local pupil frame */
            if (wfe_opd != NULL)
            {
                o += wfe_to_um * wfe_opd[p];
            }
            out_opd[p] = o;
            if (out_amp != NULL)
            {
                out_amp[p] = a;
            }
        }
    }
}

/**
 * cacaosim_beam_shm_init_kw() - Populate fixed-slot `CSIM_*` keyword headers in a 3D SHM image
 * @im: Target `ImageStreamIO` beam image with `NBkw >= 36` (no-op if `NULL` or fewer keywords)
 *
 * Role:
 *   Cold-path (`MILK_COLD`) SHM header formatter laying out slots `kw[0..35]`:
 *     - `kw[0]`    : `CSIM_BM`  (`int64` `'L'` = `1`, magic marker identifying a cacaosim beam)
 *     - `kw[1]`    : `CSIM_L0`  (`double` `'D'` = `0.40`, minimum spectral wavelength `[um]`)
 *     - `kw[2]`    : `CSIM_L1`  (`double` `'D'` = `2.40`, maximum spectral wavelength `[um]`)
 *     - `kw[3]`    : `CSIM_FLX` (`double` `'D'`, integrated broadband photon rate `[ph/s]`)
 *     - `kw[4..35]`: `CSIM_S00`..`CSIM_S31` (`double` `'D'`, per-bin photon rates `[ph/s]`)
 *
 * Rationale:
 *   Establishing deterministic slot indices once at stream initialization lets per-frame writers
 *   and readers index `im->kw[4 + b]` in `O(1)` time without string formatting or linear scans,
 *   while remaining human-readable in `milk-shmim-info` / FITS exports.
 */
void MILK_COLD cacaosim_beam_shm_init_kw(IMAGE *im)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < 4 + CACAOSIM_NSPEC)
    {
        return;
    }
    snprintf(im->kw[0].name, KEYWORD_MAX_STRING, "CSIM_BM");
    im->kw[0].type       = 'L';
    im->kw[0].value.numl = 1;
    snprintf(im->kw[0].comment, KEYWORD_MAX_COMMENT, "cacaosim [OPD_um, Amp] optical beam");

    snprintf(im->kw[1].name, KEYWORD_MAX_STRING, "CSIM_L0");
    im->kw[1].type       = 'D';
    im->kw[1].value.numf = (double) CACAOSIM_SPEC_L0_UM;
    snprintf(im->kw[1].comment, KEYWORD_MAX_COMMENT, "Min spectral wavelength [um]");

    snprintf(im->kw[2].name, KEYWORD_MAX_STRING, "CSIM_L1");
    im->kw[2].type       = 'D';
    im->kw[2].value.numf = (double) CACAOSIM_SPEC_L1_UM;
    snprintf(im->kw[2].comment, KEYWORD_MAX_COMMENT, "Max spectral wavelength [um]");

    snprintf(im->kw[3].name, KEYWORD_MAX_STRING, "CSIM_FLX");
    im->kw[3].type       = 'D';
    im->kw[3].value.numf = 0.0;
    snprintf(im->kw[3].comment, KEYWORD_MAX_COMMENT, "Total integrated photon rate [ph/s]");

    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        int k = 4 + b;
        snprintf(im->kw[k].name, KEYWORD_MAX_STRING, "CSIM_S%02d", b);
        im->kw[k].type       = 'D';
        im->kw[k].value.numf = 0.0;
        snprintf(im->kw[k].comment, KEYWORD_MAX_COMMENT, "Bin %.3f um rate [ph/s]",
                 (double) cacaosim_spec_bin_lambda(b));
    }
}

/**
 * cacaosim_beam_shm_write_spec() - Write 32-bin photon spectrum into fixed SHM keyword slots
 * @im:   Target SHM beam `IMAGE` (`NBkw >= 36`; no-op if `NULL` or undersized)
 * @spec: Source 32-bin photon arrival rate spectrum `[photons/s]` (no-op if `NULL`)
 *
 * Role:
 *   Hot-path (`MILK_HOT`) per-frame serializer updating `CSIM_S00..S31` and `CSIM_FLX` in SHM.
 *
 * Rationale:
 *   Auto-initializes keyword headers on the first call if `kw[0]` is not yet marked `CSIM_BM = 1`,
 *   then performs 32 direct double-precision stores and accumulates `tot` in `double` to preserve
 *   precision across wide dynamic-range spectra.
 */
void MILK_HOT cacaosim_beam_shm_write_spec(
    IMAGE                 *im,
    const cacaosim_spec_t *spec)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || spec == NULL ||
        im->md->NBkw < 4 + CACAOSIM_NSPEC)
    {
        return;
    }
    /* Self-heal keyword table if stream was recreated without cacaosim_beam_shm_init_kw */
    if (im->kw[0].type != 'L' || im->kw[0].value.numl != 1)
    {
        cacaosim_beam_shm_init_kw(im);
    }
    double tot = 0.0;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        double v             = (double) spec->ph_s[b];
        im->kw[4 + b].value.numf = v;
        tot += v;
    }
    im->kw[3].value.numf = tot;
}

/**
 * cacaosim_beam_shm_read_spec() - Extract 32-bin photon spectrum from SHM beam keywords
 * @im:   Source SHM `IMAGE` to inspect
 * @spec: Destination spectrum populated when `CSIM_BM == 1`
 *
 * Role:
 *   Reads upstream spectral radiometry from an attached input beam SHM stream in `O(1)` time.
 *
 * Rationale:
 *   Checks both keyword capacity (`NBkw >= 36`) and the `kw[0]` `CSIM_BM == 1` tag before copying
 *   `kw[4..35]`, preventing uninitialized or unrelated SHM keywords from corrupting sensor flux.
 *
 * Return: `1` if valid beam spectral keywords were found and copied, `0` otherwise.
 */
int cacaosim_beam_shm_read_spec(
    const IMAGE     *im,
    cacaosim_spec_t *spec)
{
    if (im == NULL || im->md == NULL || im->kw == NULL || spec == NULL ||
        im->md->NBkw < 4 + CACAOSIM_NSPEC)
    {
        return 0;
    }
    if (im->kw[0].type != 'L' || im->kw[0].value.numl != 1)
    {
        return 0;
    }
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        spec->ph_s[b] = (float) im->kw[4 + b].value.numf;
    }
    return 1;
}

/**
 * cacaosim_beam_shm_is_valid() - Check whether `@im` is a valid `[nwf x nwf x >=2]` float32 beam
 * @im:  Candidate SHM `IMAGE` (`NULL` returns `0`)
 * @nwf: Required pupil spatial dimension `[pixels]` on axes 0 and 1
 *
 * Role:
 *   Fast structural validator guarding SHM pointer arithmetic on `[OPD, Amp]` beam cubes.
 *
 * Rationale:
 *   Accepts `size[2] >= 2` so both standard 2-slice beams (`[OPD, Amp]`) and `2 * N_lambda`-slice
 *   polychromatic beams pass spatial/datatype validation uniformly.
 *
 * Return: `1` if `@im` is a valid 3D `_DATATYPE_FLOAT` beam matching `@nwf`, `0` otherwise.
 */
int cacaosim_beam_shm_is_valid(
    const IMAGE *im,
    uint32_t     nwf)
{
    if (im == NULL || im->md == NULL)
    {
        return 0;
    }
    return (im->md->datatype == _DATATYPE_FLOAT && im->md->naxis == 3 &&
            im->md->size[0] == nwf && im->md->size[1] == nwf && im->md->size[2] >= 2)
               ? 1
               : 0;
}

/**
 * cacaosim_beam_shm_write_poly_kw() - Serialize polychromatic channel count and bounds to SHM
 * @im:     Target SHM beam `IMAGE` (`NBkw >= 39`; no-op if `NULL` or fewer keywords)
 * @nwl:    Number of `[OPD_c, Amp_c]` wavelength channel pairs (`< 1` clamped to `1`)
 * @wl0_um: Shortest / channel-0 wavelength `[um]`
 * @wl1_um: Longest / channel-`(nwl - 1)` wavelength `[um]`
 *
 * Role:
 *   Records `CSIM_NWL`, `CSIM_WL0`, and `CSIM_WL1` at slots `kw[36..38]` immediately following the
 *   32 spectral photometry bins (`kw[4..35]`).
 *
 * Rationale:
 *   Keeping polychromatic cube geometry at fixed offset `base = 4 + CACAOSIM_NSPEC` (`36`) avoids
 *   colliding with `CSIM_S00..S31` while maintaining `O(1)` lookup in
 *   `cacaosim_beam_shm_read_poly_kw()`.
 */
void cacaosim_beam_shm_write_poly_kw(
    IMAGE   *im,
    uint32_t nwl,
    float    wl0_um,
    float    wl1_um)
{
    const int base = 4 + CACAOSIM_NSPEC;
    if (im == NULL || im->md == NULL || im->kw == NULL || im->md->NBkw < base + 3)
    {
        return;
    }
    snprintf(im->kw[base].name, KEYWORD_MAX_STRING, "CSIM_NWL");
    im->kw[base].type       = 'L';
    im->kw[base].value.numl = (int64_t) ((nwl >= 1u) ? nwl : 1u);
    snprintf(im->kw[base].comment, KEYWORD_MAX_COMMENT, "Number of [OPD, Amp] spectral channels");

    snprintf(im->kw[base + 1].name, KEYWORD_MAX_STRING, "CSIM_WL0");
    im->kw[base + 1].type       = 'D';
    im->kw[base + 1].value.numf = (double) wl0_um;
    snprintf(im->kw[base + 1].comment, KEYWORD_MAX_COMMENT, "First channel wavelength [um]");

    snprintf(im->kw[base + 2].name, KEYWORD_MAX_STRING, "CSIM_WL1");
    im->kw[base + 2].type       = 'D';
    im->kw[base + 2].value.numf = (double) wl1_um;
    snprintf(im->kw[base + 2].comment, KEYWORD_MAX_COMMENT, "Last channel wavelength [um]");
}

/**
 * cacaosim_beam_shm_read_poly_kw() - Read polychromatic channel count and wavelength span from SHM
 * @im:      Source 3D SHM beam `IMAGE`
 * @nwl_out: Optional output receiving resolved channel count (`>= 1`)
 * @wl0_out: Optional output receiving channel-0 wavelength `[um]` (`0.0f` if unavailable)
 * @wl1_out: Optional output receiving last-channel wavelength `[um]` (`0.0f` if unavailable)
 *
 * Role:
 *   Inspects an input SHM beam stream (`cacaosim.coro`, `cacaosim.fpcam`) to determine whether it
 *   encodes multiple coherent wavelength slices `[OPD_c, Amp_c]` (`size[2] >= 4`) and extracts
 *   `CSIM_NWL`, `CSIM_WL0`, and `CSIM_WL1`.
 *
 * Rationale:
 *   Initializes outputs to safe monochromatic defaults (`1, 0, 0`) before checking `size[2] >= 4`,
 *   derives a hard upper bound `z_nwl = size[2] / 2` from actual allocated SHM cube depth, and
 *   only trusts `CSIM_NWL` when `kn <= z_nwl` so malformed keywords cannot trigger buffer overruns.
 *
 * Return: `1` when resolved channel count `> 1`, `0` when monochromatic or invalid.
 */
int cacaosim_beam_shm_read_poly_kw(
    const IMAGE *im,
    uint32_t    *nwl_out,
    float       *wl0_out,
    float       *wl1_out)
{
    if (nwl_out != NULL)
    {
        *nwl_out = 1u;
    }
    if (wl0_out != NULL)
    {
        *wl0_out = 0.0f;
    }
    if (wl1_out != NULL)
    {
        *wl1_out = 0.0f;
    }
    /* Need at least 4 Z-slices (2 [OPD, Amp] pairs) to carry > 1 wavelength channel */
    if (im == NULL || im->md == NULL || im->md->naxis != 3 || im->md->size[2] < 4u)
    {
        return 0;
    }

    uint32_t z_nwl = im->md->size[2] / 2u;
    uint32_t kw_n  = z_nwl;
    float    l0    = 0.0f;
    float    l1    = 0.0f;

    const int base = 4 + CACAOSIM_NSPEC;
    if (im->kw != NULL && im->md->NBkw >= base + 3 && im->kw[base].type == 'L' &&
        strcmp(im->kw[base].name, "CSIM_NWL") == 0 && im->kw[base].value.numl >= 1)
    {
        uint32_t kn = (uint32_t) im->kw[base].value.numl;
        if (kn <= z_nwl)
        {
            kw_n = kn;
        }
        l0 = (float) im->kw[base + 1].value.numf;
        l1 = (float) im->kw[base + 2].value.numf;
    }

    if (nwl_out != NULL)
    {
        *nwl_out = kw_n;
    }
    if (wl0_out != NULL)
    {
        *wl0_out = l0;
    }
    if (wl1_out != NULL)
    {
        *wl1_out = l1;
    }
    return (kw_n > 1u) ? 1 : 0;
}
