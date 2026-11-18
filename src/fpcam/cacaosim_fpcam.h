// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam.h
 * @brief   High-throughput focal plane PSF & camera simulator (Soummer MFT + padded FFT, CPU/CUDA)
 *
 * Architectural Role in cacaosim:
 *   Defines the public C API, optical/detector configuration structs, and persistent CPU/CUDA
 *   context for the polychromatic focal-plane science PSF camera (`cacaosim.fpcam`). Serves both
 *   classical Adaptive Optics imaging benches (combining static pupil `pup_amp`, atmospheric
 *   wavefront `wf_in`, scintillation `atm_amp`, DM surface `dm_opd`, and science-arm `ncpa_opd`)
 *   and post-coronagraphic high-contrast dark-hole benches (consuming dynamic per-frame,
 *   per-channel `[OPD_um, Amplitude]` beams from `cacaosim.coro` or `cacaosim.beamsplit`).
 *   Integrates `navg` optical subframes per detector exposure, applies realistic sensor physics
 *   (bandpass photon flux, Poisson shot noise, dark current, readout noise, e-/ADU gain, bias,
 *   saturation), maintains a cumulative long-exposure PSF, and reports live Strehl ratio telemetry.
 *
 * Design Rationale & Optical Physics:
 *   - Soummer Matrix-DFT (`PROP_MFT`) vs Zero-Padded 2D FFT (`PROP_FFT`):
 *     * `PROP_MFT` evaluates the exact 2D discrete Fourier transform onto an `ncam x ncam` focal
 *       grid at arbitrary continuous plate scale `pixscale_ld` (`pixels per lambda0/D`) via two
 *       separable matrix products `Efoc = U_c * Epup * V_c` (`cblas_cgemm` /
 *       `cublasCgemmStridedBatched`). In polychromatic mode, channel `c` scales focal sampling as
 *       `sc = pixscale_ld * (lam_c / lam0)`, faithfully modeling physical radial chromatic speckle
 *       elongation on a fixed-pitch detector without 8x..16x FFT zero-padding.
 *     * `PROP_FFT` zero-pads the `[nwf x nwf]` pupil to `nfft = nwf * pad_factor` and crops the
 *       central `[ncam x ncam]` field via `FFTW3f` / `cuFFT`, providing `O(N^2 log N)` throughput
 *       for wide-field survey cameras where `ncam` is large.
 *   - Dynamic Per-Frame Pupil Amplitude & OPD (`accum_beam` / `accum_beam_poly`):
 *     Downstream of a coronagraphic focal mask and Lyot stop, residual aberrations modulate both
 *     the pupil-plane amplitude and phase on every single AO timestep and vary chromatically
 *     across spectral channels. Rebuilding `Epup(x, y, lam_c)` dynamically from `[beam_opd,
 *     beam_amp]` on each subframe captures coherent speckle pinning, amplitude-induced contrast
 *     floors, and post-Lyot dark-hole digging accurately.
 *   - Dual Strehl + Parseval Photometric Normalization:
 *     Optical intensities are normalized by `(sum A_pup)^2` (`1 / sqrt(pup_sum_amp)` per 1D MFT
 *     matrix), so an unaberrated clear entrance pupil yields peak intensity `1.0` — making every
 *     internal exposure pixel a direct optical Strehl ratio `[0..1]`. At detector readout,
 *     `inf_flux_norm` (computed via Parseval's theorem over the infinite focal plane) converts
 *     Strehl intensity into physical photo-electrons such that `flux_ph` equals total clear-pupil
 *     incident photons regardless of field-of-view cropping (`ncam`) or upstream coronagraphic
 *     starlight rejection.
 */

#ifndef CACAOSIM_FPCAM_H
#define CACAOSIM_FPCAM_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of coherent spectral channels supported in polychromatic PSF synthesis.
 */
#define CACAOSIM_FPCAM_MAX_LAMBDA 32

/**
 * enum cacaosim_fpcam_prop_t - Focal-plane Fraunhofer optical propagation algorithm
 * @CACAOSIM_FPCAM_PROP_MFT: Exact Soummer et al. (2007) separable Matrix-DFT (`U_c * Epup * V_c`)
 *                           with arbitrary continuous `pixscale_ld` and physical chromatic PSF
 *                           magnification (`lam_c / lam0`) via OpenBLAS / cuBLAS.
 * @CACAOSIM_FPCAM_PROP_FFT: Zero-padded `nfft = nwf * pad_factor` 2D FFT followed by DC-centered
 *                           `[ncam x ncam]` crop via FFTW3f / cuFFT for wide-field imaging.
 */
typedef enum
{
    CACAOSIM_FPCAM_PROP_MFT = 0, /**< Exact Soummer (2007) Matrix-DFT via BLAS/cuBLAS CGEMM */
    CACAOSIM_FPCAM_PROP_FFT = 1  /**< Zero-padded 2D FFT crop via FFTW3f/cuFFT */
} cacaosim_fpcam_prop_t;

/**
 * struct cacaosim_fpcam_cfg_t - Optical and detector configuration for `cacaosim.fpcam`
 * @nwf:         Input square pupil grid dimension `[pixels]` (`>= 8`).
 * @ncam:        Output square focal-plane detector dimension `[pixels]` (`>= 4`).
 * @prop_mode:   Propagator choice (`cacaosim_fpcam_prop_t`: `0` = MFT, `1` = padded FFT).
 * @pad_factor:  Zero-padding factor `nfft / nwf` when `prop_mode == 1` (`>= 2`, default: `2`).
 * @pup_diam:    Clear entrance pupil diameter `D` as a fraction of `nwf` (default: `0.9375`).
 * @lambda_um:   Central science filter wavelength `lambda0` `[um]` (default: `1.65`).
 * @bw_frac:     Fractional filter bandwidth `dlambda / lambda0` (`[0..0.5]`, `0` = monochromatic).
 * @nlambda:     Number of coherent spectral channels (`1..32`, default: `1`).
 * @has_ext_lam: Non-zero to take channel wavelengths from `ext_lam_um[0..nlambda-1]` (e.g. matched
 *               to an upstream polychromatic coronagraph SHM stream).
 * @ext_lam_um:  Explicit per-channel wavelength table `[um]` used when `has_ext_lam != 0`.
 * @pixscale_ld: Focal sampling `[pixels per lambda0/D]` in MFT mode (default: `2.5`..`3.0`).
 * @wf_unit:     Physical unit of standalone `wf_in` (`0` = OPD `[um]`, `1` = phase `[rad]`).
 * @ref_lambda:  Reference wavelength `[um]` converting phase `[rad]` to OPD when `wf_unit == 1`.
 * @rng_seed:    Base 64-bit PRNG seed for deterministic Poisson photon and Gaussian readout noise.
 * @backend:     Compute backend (`cacaosim_backend_t`: `0` = auto, `1` = CPU, `2` = CUDA).
 * @det:         Baseline science detector readout configuration (flux, dark, RON, gain, bias, ADU).
 */
typedef struct
{
    uint32_t           nwf;            /**< Input pupil grid dimension [pixels] */
    uint32_t           ncam;           /**< Output square camera dimension [pixels] */
    int32_t            prop_mode;      /**< 0 = MFT (arbitrary pixscale_ld), 1 = padded FFT */
    uint32_t           pad_factor;     /**< FFT zero-padding factor when prop_mode == 1 (>= 2) */
    float              pup_diam;       /**< Clear pupil diameter fraction of nwf (def: 0.9375) */
    float              lambda_um;      /**< Central science wavelength [microns] (default: 1.65) */
    float              bw_frac;        /**< Fractional bandpass dlambda / lambda0 [0..0.5] */
    int32_t            nlambda;        /**< Spectral slices (0 = auto -> 2 if bw>0, 1..32) */
    int32_t            has_ext_lam;    /**< 1 to use explicit ext_lam_um[0..nlambda-1] */
    float              ext_lam_um[CACAOSIM_FPCAM_MAX_LAMBDA]; /**< Explicit channel wavelengths */
    int32_t            has_ext_weight; /**< 1 to use explicit ext_weight[0..nlambda-1] */
    float              ext_weight[CACAOSIM_FPCAM_MAX_LAMBDA]; /**< Optional channel photon wts */
    int32_t            has_in_lam;     /**< 1 when upstream beam wavelengths in_lam_um are set */
    float              in_lam_um[CACAOSIM_FPCAM_MAX_LAMBDA];  /**< Upstream beam channel [um] */
    float              pixscale_ld;    /**< Focal sampling [pixels per lambda0/D] (MFT mode) */
    int32_t            wf_unit;        /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda;     /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;       /**< Base PRNG seed for photon/readout noise */
    int32_t            backend;        /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;            /**< Integrated science detector readout parameters */
} cacaosim_fpcam_cfg_t;

/**
 * struct cacaosim_fpcam_ctx_t - Persistent CPU (OpenBLAS/FFTW3f) and CUDA (cuBLAS/cuFFT) state
 * @cfg:              Copy of active camera configuration.
 * @nfft:             Padded FFT grid dimension `nwf * pad_factor`.
 * @npix_wf:          Total pupil pixels (`nwf * nwf`).
 * @npix_fft:         Total padded FFT pixels (`nfft * nfft`).
 * @npix_cam:         Total focal detector pixels (`ncam * ncam`).
 * @eff_nlam:         Resolved number of spectral channels (`1..CACAOSIM_FPCAM_MAX_LAMBDA`).
 * @eff_pixscale_ld:  Effective focal sampling `[pixels per lambda0/D]` (`pixscale_ld` or
 *                    `nfft / dpix`).
 * @pup_sum_amp:      Reference clear-pupil amplitude sum `sum(A_pup)` defining Strehl = 1.0 peak.
 * @pup_sum_amp2:     Reference clear-pupil energy sum `sum(A_pup^2)` for Parseval photometry.
 * @inf_flux_norm:    Infinite-plane Parseval sum of Strehl-normalized PSF so `flux_ph /
 *                    inf_flux_norm` converts Strehl intensity into true incident photons/pixel.
 * @use_cuda:         Non-zero when GPU execution via `cuda_priv` is active.
 * @frame_seq:        Monotonic exposure readout counter driving independent GPU noise realizations.
 * @lam_um:           Resolved per-channel wavelengths `[eff_nlam]` in `[um]`.
 * @mft_U_cplx:       Precomputed MFT Y-axis matrices `[eff_nlam * ncam * nwf * 2]`
 *                    (`1/sqrt(pup_sum_amp)` normalized).
 * @mft_V_cplx:       Precomputed MFT X-axis matrices `[eff_nlam * nwf * ncam * 2]`.
 * @cpu_epup_cplx:    Scratch complex entrance pupil buffer `[npix_wf * 2]`.
 * @cpu_mft_tmp_cplx: Scratch intermediate MFT product `[ncam * nwf * 2]`.
 * @cpu_efoc_cplx:    Scratch complex focal field `[npix_cam * 2]`.
 * @cpu_fft_in_cplx:  Zero-padded FFTW3f input complex pupil buffer `[npix_fft * 2]`.
 * @cpu_fft_out_cplx: FFTW3f output complex focal spectrum buffer `[npix_fft * 2]`.
 * @fftw_plan_fwd:    Opaque `fftwf_plan` for 2D forward FFT (`cpu_fft_in_cplx -> out_cplx`).
 * @cpu_sub_int:      Reserved scratch subframe intensity buffer `[npix_cam]`.
 * @cpu_accum_int:    Running sum of Strehl-normalized subframe PSFs `[npix_cam]` across `navg`.
 * @cpu_exp_int:      Subframe-averaged Strehl-normalized exposure PSF `[npix_cam]`.
 * @cpu_long_int:     Cumulative sum of exposure PSFs `[npix_cam]` since init or last reset.
 * @subframe_cnt:     Number of optical subframes accumulated into the current exposure.
 * @long_cnt:         Number of completed exposures accumulated into `cpu_long_int`.
 * @last_strehl_axis: Most recent exposure Strehl ratio `[0..1]` at nominal optical axis `(c0, c0)`.
 * @last_strehl_peak: Most recent exposure peak Strehl ratio `[0..1]` (`max` over detector).
 * @long_strehl_peak: Cumulative long-exposure peak Strehl ratio `[0..1]`.
 * @det_pool:         Per-thread `xoshiro128++` PRNG pool for CPU detector readout noise.
 * @cuda_priv:        Opaque pointer to `fpcam_cuda_priv_t` allocated by `cacaosim_fpcam_cuda_init`.
 */
typedef struct
{
    cacaosim_fpcam_cfg_t cfg;
    uint32_t             nfft;
    uint64_t             npix_wf;
    uint64_t             npix_fft;
    uint64_t             npix_cam;
    int32_t              eff_nlam;
    float                eff_pixscale_ld;
    float                pup_sum_amp;
    float                pup_sum_amp2;
    float                base_inf_norm;
    float                inf_flux_norm;
    int32_t              use_cuda;
    uint64_t             frame_seq;

    /* Spectral channel wavelengths [eff_nlam], weights, scales, and wavenumbers */
    float                lam_um[CACAOSIM_FPCAM_MAX_LAMBDA];
    float                chan_w[CACAOSIM_FPCAM_MAX_LAMBDA];
    float                strehl_w[CACAOSIM_FPCAM_MAX_LAMBDA];
    float                scale_c[CACAOSIM_FPCAM_MAX_LAMBDA];
    float                k_rad[CACAOSIM_FPCAM_MAX_LAMBDA];

    /* Precomputed MFT basis matrices: U [eff_nlam * ncam * nwf], V [eff_nlam * nwf * ncam] */
    float               *mft_U_cplx;
    float               *mft_V_cplx;

    /* CPU work buffers */
    float               *cpu_epup_cplx;
    float               *cpu_mft_tmp_cplx;
    float               *cpu_efoc_cplx;
    float               *cpu_fft_in_cplx;
    float               *cpu_fft_out_cplx;
    void                *fftw_plan_fwd;
    float               *cpu_sub_int;
    float               *cpu_accum_int;
    float               *cpu_exp_int;
    float               *cpu_long_int;
    uint32_t             subframe_cnt;
    uint64_t             long_cnt;
    float                last_strehl_axis;
    float                last_strehl_peak;
    float                long_strehl_peak;
    cacaosim_det_pool_t  det_pool;

    /* Opaque CUDA state allocated by cacaosim_fpcam_cuda.cu */
    void                *cuda_priv;
} cacaosim_fpcam_ctx_t;

/**
 * cacaosim_fpcam_init() - Initialize MFT/FFT optical propagation plans and detector state
 * @ctx:        Output focal-plane camera context to populate (non-NULL).
 * @cfg:        Camera optical and detector configuration (`nwf >= 8`, `ncam >= 4`).
 * @pup_amp:    Reference clear entrance pupil amplitude `[nwf * nwf]` (non-NULL, `sum > 0`).
 * @ncpa_opd:   Optional static science-arm NCPA OPD `[um]` `[nwf * nwf]` (uploaded to GPU if used).
 * @ext_stream: Optional shared `cudaStream_t` handle for fused GPU benches (NULL creates private).
 *
 * Role:
 *   Cold-path initializer computing clear-pupil Strehl/Parseval photometric constants, resolving
 *   polychromatic wavelengths, precomputing MFT matrices or FFTW3f plans, initializing detector
 *   PRNG states, and setting up the optional CUDA GPU backend.
 *
 * Rationale:
 *   Measures `pup_sum_amp` and `pup_sum_amp2` from the unocculted entrance `@pup_amp` at startup
 *   so that even when downstream coronagraphic `beam_amp` frames have lower Lyot transmission,
 *   Strehl ratio and photon throughput remain referenced to the true clear telescope aperture.
 *
 * Return: 0 on success, -1 on invalid dimensions, zero pupil transmission, or allocation failure.
 */
int cacaosim_fpcam_init(
    cacaosim_fpcam_ctx_t       *ctx,
    const cacaosim_fpcam_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream);

/**
 * cacaosim_fpcam_free() - Destroy plans and release all host/device focal plane camera buffers
 * @ctx: Camera context to release (NULL-safe and idempotent).
 *
 * Role:
 *   Cold-path teardown freeing CUDA state, FFTW3f plans, PRNG pools, and aligned host arrays.
 *
 * Rationale:
 *   Zeros `@ctx` on exit so error-recovery paths and clean shutdowns share one safe destructor.
 */
void cacaosim_fpcam_free(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_reset_long() - Clear cumulative long-exposure PSF accumulator
 * @ctx: Active focal-plane camera context (NULL-safe).
 *
 * Role:
 *   Resets `cpu_long_int`, `long_cnt`, and `long_strehl_peak` to zero between AO experiments.
 *
 * Rationale:
 *   Allows live interactive or scripted clearing of long-exposure Strehl/contrast integrations
 *   after closing the AO loop without tearing down FFT/cuBLAS plans or SHM streams.
 */
void cacaosim_fpcam_reset_long(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_set_weights() - Update incoherent channel photon weights and Parseval flux norm
 */
void cacaosim_fpcam_set_weights(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *weights);

/**
 * cacaosim_fpcam_accum_subframe() - Propagate one optical subframe into exposure accumulator
 * @ctx:       Active camera context (increments `ctx->subframe_cnt`).
 * @pup_amp:   Host static entrance pupil amplitude `[npix_wf]` (used on CPU; GPU uses init copy).
 * @atm_amp:   Optional host atmospheric scintillation amplitude `[npix_wf]` (NULL -> 1.0).
 * @wf_in:     Optional host input wavefront `[npix_wf]` in `cfg.wf_unit` (NULL -> 0.0).
 * @dm_opd:    Optional host DM OPD `[um]` `[npix_wf]` subtracted from `wf_in` (NULL -> 0.0).
 * @ncpa_opd:  Optional host static NCPA OPD `[um]` `[npix_wf]` added to wavefront (CPU path).
 * @d_dm_opd:  Optional resident device DM OPD `[um]` `[npix_wf]` for zero-copy GPU benches.
 * @turb_gain: Dimensionless multiplier on `@wf_in` (`0.0f` disables turbulence cleanly).
 *
 * Role:
 *   Hot-path subframe propagator for standalone / classical AO benches where the pupil stop is
 *   static and wavefront components arrive on separate streams (`in_wf`, `in_dm`, `in_amp`).
 *
 * Rationale:
 *   Automatically clears the exposure accumulator (`cpu_accum_int` / `d_accum_int`) when
 *   `subframe_cnt == 0` before coherently propagating all `eff_nlam` spectral channels and
 *   summing their intensities incoherently, enabling multi-subframe temporal smear (`navg > 1`).
 */
void cacaosim_fpcam_accum_subframe(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    const float          *d_dm_opd,
    float                 turb_gain);

/**
 * cacaosim_fpcam_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into accumulator
 * @ctx:        Active camera context (increments `ctx->subframe_cnt`).
 * @beam_opd:   Host 2D beam OPD `[um]` `[npix_wf]` (used when `@d_beam_opd == NULL`).
 * @beam_amp:   Host 2D dynamic beam amplitude `[npix_wf]` (used when `@d_beam_amp == NULL`).
 * @ncpa_opd:   Optional host static science-arm NCPA OPD `[um]` `[npix_wf]`.
 * @d_beam_opd: Optional resident device 2D beam OPD `[um]` `[npix_wf]`.
 * @d_beam_amp: Optional resident device 2D dynamic beam amplitude `[npix_wf]`.
 *
 * Role:
 *   Hot-path single-channel beam subframe accumulator for inline optical benches (`htsim`, FPS).
 *
 * Rationale:
 *   Convenience wrapper around `cacaosim_fpcam_accum_beam_poly()` with `beam_nlam = 1`, updating
 *   both pupil amplitude and OPD dynamically on every subframe.
 */
void cacaosim_fpcam_accum_beam(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_fpcam_accum_beam_poly() - Propagate 1 or N_lambda [OPD_c, Amp_c] channels into camera
 * @ctx:         Active camera context (increments `ctx->subframe_cnt`).
 * @beam_opd:    Host beam OPD `[um]` (`@beam_nlam` slices separated by `@chan_stride`).
 * @beam_amp:    Host dynamic beam amplitude (`@beam_nlam` slices separated by `@chan_stride`).
 * @beam_nlam:   Number of wavelength channels in input beam (`1` broadcasts across `eff_nlam`).
 * @chan_stride: Float stride between consecutive beam channels (`npix_wf` or `2 * npix_wf`).
 * @ncpa_opd:    Optional host static science-arm NCPA OPD `[um]` `[npix_wf]`.
 * @d_beam_opd:  Optional resident device beam OPD (`@beam_nlam` slices at `@chan_stride`).
 * @d_beam_amp:  Optional resident device dynamic beam amplitude.
 *
 * Role:
 *   Primary hot-path polychromatic beam accumulator for post-coronagraphic dark-hole imaging.
 *
 * Rationale:
 *   Consumes dynamic per-channel `[OPD_c, Amp_c]` cubes emitted by `cacaosim.coro` so chromatic
 *   pupil amplitude apodization, Lyot diffraction residuals, and wrapped post-FPM phase are
 *   propagated coherently within each spectral channel and summed incoherently on the focal plane.
 */
void cacaosim_fpcam_accum_beam_poly(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_fpcam_readout() - Finalize exposure, update Strehl/long-exp, and run detector readout
 * @ctx:      Active camera context with `subframe_cnt >= 1` (resets `subframe_cnt` to 0).
 * @det_live: Optional live detector parameter override (NULL uses `ctx->cfg.det`).
 * @out_u16:  Destination `uint16` camera frame `[npix_cam]` when `out_dtype == 0` (else NULL).
 * @out_f32:  Destination `float32` camera frame `[npix_cam]` when `out_dtype != 0` (else NULL).
 * @out_long: Optional destination `float32` Strehl-normalized long-exposure PSF `[npix_cam]`.
 *
 * Role:
 *   Completes a `navg`-subframe camera integration, runs CPU or CUDA detector noise/quantization,
 *   updates `cpu_long_int`, and records `last_strehl_axis`, `last_strehl_peak`, and
 *   `long_strehl_peak`.
 *
 * Rationale:
 *   Decoupling `readout()` from `accum_*()` lets fast optical simulations run at multi-kHz DM/WFS
 *   rates while reading out noisy science exposures at realistic slower camera frame rates.
 */
void cacaosim_fpcam_readout(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long);

void cacaosim_fpcam_readout_async(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long);

void cacaosim_fpcam_readout_finish(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_step() - Single-subframe optical propagation + detector readout convenience call
 * @ctx:       Active camera context.
 * @pup_amp:   Host entrance pupil amplitude `[npix_wf]`.
 * @atm_amp:   Optional host atmospheric scintillation amplitude `[npix_wf]`.
 * @wf_in:     Optional host input wavefront `[npix_wf]`.
 * @dm_opd:    Optional host DM OPD `[um]` `[npix_wf]`.
 * @ncpa_opd:  Optional host static NCPA OPD `[um]` `[npix_wf]`.
 * @d_dm_opd:  Optional device DM OPD `[um]` `[npix_wf]`.
 * @turb_gain: Turbulence multiplier on `@wf_in`.
 * @det_live:  Optional live detector readout configuration (NULL uses `ctx->cfg.det`).
 * @out_u16:   Destination `uint16` frame `[npix_cam]` when `out_dtype == 0`.
 * @out_f32:   Destination `float32` frame `[npix_cam]` when `out_dtype != 0`.
 * @out_long:  Optional destination `float32` cumulative long-exposure PSF `[npix_cam]`.
 *
 * Role:
 *   Single-call convenience wrapper executing `cacaosim_fpcam_accum_subframe()` immediately
 *   followed by `cacaosim_fpcam_readout()` (`navg = 1`).
 *
 * Rationale:
 *   Simplifies synchronous 1-to-1 frame benchmarks and unit tests that do not require subframe
 *   oversampling.
 */
void cacaosim_fpcam_step(
    cacaosim_fpcam_ctx_t     *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long);

/**
 * cacaosim_fpcam_cuda_init() - Allocate GPU buffers, cuBLAS/cuFFT plans, and upload static arrays
 * @ctx:        Host-initialized camera context with precomputed wavelengths and MFT tables.
 * @pup_amp:    Static reference pupil amplitude `[npix_wf]` uploaded to `d_pup_amp`.
 * @ncpa_opd:   Optional static science-arm NCPA OPD `[um]` `[npix_wf]` (may be NULL).
 * @ext_stream: Optional external `cudaStream_t` handle (NULL creates a private stream).
 *
 * Role:
 *   Internal cold-path CUDA initializer invoked by `cacaosim_fpcam_init()`.
 *
 * Rationale:
 *   Pre-allocates batched polychromatic complex buffers and cuBLAS/cuFFT handles once at startup
 *   so hot GPU subframe accumulation performs zero allocations.
 *
 * Return: 0 on success, -1 on CUDA/cuBLAS/cuFFT allocation or plan failure.
 */
int  cacaosim_fpcam_cuda_init(cacaosim_fpcam_ctx_t *ctx,
                              const float          *pup_amp,
                              const float          *ncpa_opd,
                              void                 *ext_stream);

/**
 * cacaosim_fpcam_cuda_free() - Destroy cuBLAS/cuFFT handles and release GPU camera buffers
 * @ctx: Camera context owning `cuda_priv` (safe when NULL or uninitialized).
 *
 * Role:
 *   Internal cold-path CUDA teardown invoked by `cacaosim_fpcam_free()`.
 *
 * Rationale:
 *   Preserves externally borrowed CUDA streams (`owns_stream == 0`) while freeing all owned VRAM.
 */
void cacaosim_fpcam_cuda_free(cacaosim_fpcam_ctx_t *ctx);
void cacaosim_fpcam_cuda_upload_weights(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_cuda_reset_accum() - Zero the GPU focal-plane subframe intensity accumulator
 * @ctx: Active camera context with initialized `cuda_priv`.
 *
 * Role:
 *   Asynchronously clears `d_accum_int` on `p->stream` at the start of each new exposure
 *   (`subframe_cnt == 0`).
 *
 * Rationale:
 *   Uses `cudaMemsetAsync` on the camera stream to avoid host synchronization between exposures.
 */
void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx);
void cacaosim_fpcam_cuda_reset_long(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_cuda_accum() - Accumulate one monochromatic-input optical subframe on CUDA GPU
 * @ctx:           Active camera context.
 * @dyn_pup_amp:   Optional host dynamic 2D pupil amplitude `[npix_wf]` (NULL uses static pupil).
 * @atm_amp:       Optional host 2D atmospheric amplitude `[npix_wf]`.
 * @wf_in:         Optional host 2D input wavefront `[npix_wf]`.
 * @dm_opd:        Optional host 2D DM OPD `[um]` `[npix_wf]`.
 * @d_pup_amp_ext: Optional resident device 2D pupil amplitude `[npix_wf]`.
 * @d_wf_opd_ext:  Optional resident device 2D wavefront `[npix_wf]`.
 * @d_dm_opd_ext:  Optional resident device 2D DM OPD `[um]` `[npix_wf]`.
 * @wf_to_opd:     Scale converting `@wf_in` / `@d_wf_opd_ext` to microns OPD.
 *
 * Role:
 *   Internal hot-path GPU subframe propagator for classical AO (`accum_subframe`) inputs.
 *
 * Rationale:
 *   Stages only non-NULL host arrays while consuming resident device pointers (`d_*_ext`) directly
 *   without PCIe copies.
 */
void cacaosim_fpcam_cuda_accum(cacaosim_fpcam_ctx_t *ctx,
                               const float          *dyn_pup_amp,
                               const float          *atm_amp,
                               const float          *wf_in,
                               const float          *dm_opd,
                               const float          *d_pup_amp_ext,
                               const float          *d_wf_opd_ext,
                               const float          *d_dm_opd_ext,
                               float                 wf_to_opd);

/**
 * cacaosim_fpcam_cuda_accum_poly() - Accumulate 1 or N_lambda dynamic `[OPD, Amp]` channels on GPU
 * @ctx:         Active camera context.
 * @beam_opd:    Host beam OPD `[um]` (`@beam_nlam` slices separated by `@chan_stride`).
 * @beam_amp:    Host dynamic beam amplitude (`@beam_nlam` slices separated by `@chan_stride`).
 * @beam_nlam:   Number of input beam wavelength channels (`1` broadcasts across `eff_nlam`).
 * @chan_stride: Float stride between consecutive beam slices.
 * @d_beam_opd:  Optional resident device beam OPD.
 * @d_beam_amp:  Optional resident device dynamic beam amplitude.
 *
 * Role:
 *   Internal hot-path GPU propagator for dynamic monochromatic or polychromatic optical beams.
 *
 * Rationale:
 *   Executes purely on GPU when `@d_beam_opd` or `@d_beam_amp` is non-NULL, or packs strided SHM
 *   cubes via `cudaMemcpy2DAsync` before running batched cuBLAS MFT or cuFFT propagation.
 */
void cacaosim_fpcam_cuda_accum_poly(cacaosim_fpcam_ctx_t *ctx,
                                    const float          *beam_opd,
                                    const float          *beam_amp,
                                    uint32_t              beam_nlam,
                                    uint64_t              chan_stride,
                                    const float          *d_beam_opd,
                                    const float          *d_beam_amp);

/**
 * cacaosim_fpcam_cuda_readout() - Normalize GPU exposure, add detector noise, and copy frame to CPU
 * @ctx:          Active camera context (`subframe_cnt >= 1`).
 * @det_live:     Detector readout configuration (non-NULL).
 * @host_exp_int: Optional host destination `[npix_cam]` for noiseless Strehl-normalized PSF.
 * @out_u16:      Optional host `uint16` output frame `[npix_cam]` when `out_dtype == 0`.
 * @out_f32:      Optional host `float32` output frame `[npix_cam]` when `out_dtype != 0`.
 *
 * Role:
 *   Internal hot-path GPU exposure readout invoked by `cacaosim_fpcam_readout()`.
 *
 * Rationale:
 *   Runs Philox-based Poisson/Gaussian detector noise and ADU quantization on the GPU, then copies
 *   both the noiseless `d_exp_int` (for exact host Strehl/long-exposure tracking) and quantized
 *   pixel buffer back to host memory in a single synchronized stream flush.
 */
void cacaosim_fpcam_cuda_readout(cacaosim_fpcam_ctx_t     *ctx,
                                 const cacaosim_det_cfg_t *det_live,
                                 float                    *host_exp_int,
                                 uint16_t                 *out_u16,
                                 float                    *out_f32);

void cacaosim_fpcam_cuda_readout_full(cacaosim_fpcam_ctx_t     *ctx,
                                      const cacaosim_det_cfg_t *det_live,
                                      uint16_t                 *out_u16,
                                      float                    *out_f32,
                                      float                    *out_long,
                                      int                       sync_stream);

void cacaosim_fpcam_cuda_readout_finish(cacaosim_fpcam_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_FPCAM_H */
