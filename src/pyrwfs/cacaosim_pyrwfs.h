// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs.h
 * @brief   High-throughput 4-sided Pyramid WFS optical & detector simulator (FFTW3f + cuFFT/Graph)
 *
 * Architectural Role in cacaosim:
 *   Public C API, configuration schema, and persistent dual-backend (CPU FFTW3f + OpenMP and GPU
 *   cuFFT + CUDA Graph) execution context for the diffractive 4-sided Pyramid Wavefront Sensor
 *   compute unit (`cacaosim.pyrwfs`). Consumed both by the standalone MILK FPS process
 *   (`cacaosim_pyrwfs_FPS.c`) and by the multi-rate fused optical bench engine (`cacaosim.htsim`).
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Fourier-Shift (1 + N_mod) FFT Reduction:
 *      In a physical modulated Pyramid WFS, a fast tip-tilt mirror steers the pupil beam around a
 *      circle of radius `mod_rad_ld` [lambda/D] over `N_mod` discrete steps to sweep the PSF core
 *      across the 4-faceted prism apex, linearizing high-WFE response while preserving diffraction-
 *      limited sensitivity. Simulating pupil tip-tilt directly requires `2 * N_mod` 2D FFTs per
 *      subframe (`N_mod` forward + `N_mod` inverse). By the Fourier shift theorem, applying a
 *      linear tip-tilt phase ramp on the pupil is mathematically equivalent to translating the
 *      focal-plane pyramid phase mask by `(-u_k, -v_k)` against a single static focal complex
 *      amplitude (up to a global pupil-plane linear phase ramp that vanishes identically when
 *      taking detector squared modulus `|Psi_det|^2`). Precomputing `N_mod` shifted complex focal
 *      masks on the cold path reduces the hot-loop cost from `2 * N_mod` 2D FFTs down to `1 +
 *      N_mod` 2D FFTs (1 forward pupil FFT + `N_mod` modulated inverse FFTs), cutting FFT work
 *      nearly in half.
 *   2. Subpixel-Integrated Analytic Facets & Roof Defect:
 *      The 4-sided square pyramid phase `exp(-i * alpha * (|u| + |v|))` factors into two 1D roof
 *      edges with slope `alpha = pi * pup_sep` [rad / focal_pixel], separating opposite re-imaged
 *      sub-pupils by `pup_sep * nfft` pixels. Sharp prism ridges (`roof_ld == 0`) are integrated
 *      analytically across each focal pixel `[u - 0.5, u + 0.5]` to prevent subpixel modulation
 *      aliasing, while finite vertex roof rounding (`roof_ld > 0`) uses a hyperbolic profile.
 *   3. CUDA Graph Capture & Multi-Subframe Integration (`navg`):
 *      On GPU, the steady-state pupil synthesis, forward 2D cuFFT, broadcast mask multiplication,
 *      batched `N_mod` inverse `cufftPlanMany`, and intensity binning are captured into a single
 *      `cudaGraphExec_t`, eliminating host launch overhead at multi-kHz AO rates. Subframe
 *      accumulation (`accum_subframe` / `accum_beam`) integrates `navg` atmospheric/DM steps
 *      incoherently before a single camera readout (`readout`) applies Poisson and readout noise.
 */

#ifndef CACAOSIM_PYRWFS_H
#define CACAOSIM_PYRWFS_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"
#include "common/cacaosim_disp.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum subframe modulation arc/interleaved slices cached in CUDA Graph slots. */
#define CACAOSIM_PYRWFS_MAX_SLICES 16

/**
 * enum cacaosim_pyrmod_mode_t - Subframe tip-tilt modulation partitioning across an exposure
 * @CACAOSIM_PYRMOD_FULL:        Evaluate all `eff_npts` ring points on every subframe (models a
 *                               simultaneous static ring diffuser or `navg == 1`).
 * @CACAOSIM_PYRMOD_SYNC_ARC:    Sweep a contiguous `2*pi / n_slices` piezo arc per subframe over
 *                               `mod_rev` revolutions per exposure (`K + N_mod` FFTs/exposure;
 *                               models real synchronous kHz piezo tip-tilt modulators).
 * @CACAOSIM_PYRMOD_INTERLEAVED: Evaluate a quadrature-strided subset `s + j * n_slices` per
 *                               subframe (`K + N_mod` FFTs/exposure, isotropic per-subframe).
 */
typedef enum
{
    CACAOSIM_PYRMOD_FULL        = 0,
    CACAOSIM_PYRMOD_SYNC_ARC    = 1,
    CACAOSIM_PYRMOD_INTERLEAVED = 2
} cacaosim_pyrmod_mode_t;

/**
 * @brief Optical and detector configuration for the 4-sided Pyramid WFS engine.
 *
 * Defines spatial grid dimensions (`nwf`, `pad_factor`, `ncam`), clear pupil diameter, sensing
 * wavelength, circular tip-tilt modulation geometry, 4-quadrant prism slope and vertex roof defect,
 * focal field stop, wavefront physical unit scaling, compute backend policy, and camera readout
 * electronics parameters.
 */
typedef struct
{
    uint32_t           nwf;               /**< Input square pupil grid dimension [pixels] (>= 8) */
    uint32_t           pad_factor;        /**< Optical FFT oversampling factor (>= 2, def: 2) */
    uint32_t           ncam;              /**< Output square WFS camera dimension [pixels] (>= 4) */
    float              pup_diam;          /**< Clear pupil diameter fraction of nwf (def: 0.9375) */
    float              lambda_um;         /**< Central sensing wavelength [um] (def: 0.75) */
    float              bw_frac;           /**< Fractional bandpass dlambda/lambda0 (0 = mono) */
    int32_t            nlambda;           /**< Spectral slices (0 = auto -> 2 if bw_frac > 0) */
    float              chrom_sep;         /**< Prism dispersion d(pup_sep)/d(lam/lam0) (def: 0) */
    int32_t            has_ext_chan;      /**< 1 to use ext_lam_um / ext_weight channel tables */
    float              ext_lam_um[CACAOSIM_MAX_LAMBDA]; /**< Optional channel wavelengths [um] */
    float              ext_weight[CACAOSIM_MAX_LAMBDA]; /**< Optional channel photon weights */
    float              mod_rad_ld;        /**< Circular modulation radius [lambda0/D] (0 = unmod) */
    int32_t            mod_npts;          /**< Discrete modulation points around circle (>= 1) */
    int32_t            mod_subframe_mode; /**< cacaosim_pyrmod_mode_t: 0=full, 1=sync_arc, 2=int */
    int32_t            mod_rev;           /**< Modulator revolutions per WFS exposure (def: 1) */
    float              mod_phase_deg;     /**< Initial modulator azimuth at t=0 of exposure [deg] */
    uint32_t           mod_subframes;     /**< Subframes integrated per WFS exposure (0/1 = full) */
    float              pup_sep;           /**< Sub-pupil center separation fraction of nfft */
    float              roof_ld;           /**< Pyramid vertex roof rounding width [lambda/D] */
    float              fstop_frac;        /**< Focal field-stop half-width fraction of nfft */
    int32_t            wf_unit;           /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda;        /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;          /**< Base 64-bit PRNG seed for photon/readout noise */
    int32_t            backend;           /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;               /**< Integrated WFS camera readout electronics params */
} cacaosim_pyrwfs_cfg_t;

/**
 * @brief Persistent CPU (FFTW3f + L2 OpenMP) and CUDA (cuFFT + CUDA Graph) Pyramid WFS state.
 *
 * Holds precomputed shifted focal-plane pyramid complex masks, flat-wavefront photometric
 * calibration, subframe exposure counters, per-thread L2-cache-resident CPU FFTW3f buffers/plans,
 * PRNG pool state, and an opaque pointer to GPU device buffers/cuFFT plans/CUDA Graph handles.
 * Zero heap allocations or plan creations occur after `cacaosim_pyrwfs_init()`.
 */
typedef struct
{
    cacaosim_pyrwfs_cfg_t cfg;            /**< Copy of initialization optical/detector config */
    uint32_t              nfft;           /**< Padded 2D optical FFT grid dimension (nwf * pad) */
    uint64_t              npix_wf;        /**< Total input pupil pixels (nwf * nwf) */
    uint64_t              npix_fft;       /**< Total optical FFT plane pixels (nfft * nfft) */
    uint64_t              npix_cam;       /**< Total output detector pixels (ncam * ncam) */
    int32_t               eff_npts;       /**< Allocated modulation point count (>= 1) */
    int32_t               eff_nlam;       /**< Active spectral slices [1..CACAOSIM_MAX_LAMBDA] */
    int32_t               n_slices;       /**< Modulation slices per revolution [1..MAX_SLICES] */
    int32_t               pts_per_slice;  /**< Inverse FFT batch size per subframe slice */
    int32_t               slice_cursor;   /**< Current modulation slice index [0..n_slices-1] */
    int32_t               force_full_mod; /**< 1 to integrate all n_slices in 1 call (calib_fast) */
    float                 lam_um[CACAOSIM_MAX_LAMBDA];         /**< Channel wavelengths [um] */
    float                 chan_w[CACAOSIM_MAX_LAMBDA];         /**< Normalized channel weights */
    float                 k_rad[CACAOSIM_MAX_LAMBDA];          /**< 2*pi / lam_um[c] [rad/um] */
    float                 chan_flat_norm[CACAOSIM_MAX_LAMBDA]; /**< Per-channel flat intensity */
    int32_t               disp_active;                         /**< 1 if chromatic disp active */
    uint64_t              disp_gen;                            /**< Cached dispersion gen */
    float                 disp_tel_diam;                       /**< Cached tel_diam_m [m] */
    int32_t               in_poly_mode;                        /**< cacaosim_beam_poly_mode_t */
    float                 disp_sx[CACAOSIM_MAX_LAMBDA];        /**< X OPD slope [um/pix] */
    float                 disp_sy[CACAOSIM_MAX_LAMBDA];        /**< Y OPD slope [um/pix] */
    float                 cached_mod_rad; /**< Modulation radius [lambda/D] baked into masks_cplx */
    float                 flat_flux_norm; /**< Unaberrated total camera intensity for photometry */
    int32_t               use_cuda;       /**< 1 when CUDA backend is active, 0 for CPU FFTW3f */
    uint32_t              subframe_cnt;   /**< Optical subframes accumulated in current exposure */
    uint64_t              frame_seq;      /**< Monotonic camera readout counter for PRNG seeding */

    /* Shifted complex pyramid masks [n_slices * eff_nlam * pts_per_slice * npix_fft * 2] */
    float                *masks_cplx;     /**< Host shifted complex focal pyramid mask stack */

    /* CPU FFTW3f state (1 forward buffer + per-thread L2-resident inverse buffers) */
    float                *cpu_pup_cplx;   /**< Padded complex pupil field [npix_fft * 2] */
    float                *cpu_foc_cplx;   /**< Forward-FFT focal complex field [npix_fft * 2] */
    float                *cpu_thr_cplx;   /**< Per-thread IFFT scratch [cpu_nth * npix_fft * 2] */
    float                *cpu_thr_int;    /**< Per-thread detector intensity [cpu_nth * nfft^2] */
    float                *cpu_det_int;    /**< Reduced unbinned detector intensity [npix_fft] */
    float                *cpu_cam_int;    /**< Binned single-subframe camera intensity [npix_cam] */
    float                *cpu_accum_int;  /**< Multi-subframe intensity accumulator [npix_cam] */
    int32_t               cpu_nth;        /**< Capped OpenMP thread count for modulation loop */
    void                 *fftw_plan_fwd;  /**< 2D forward fftwf_plan (cpu_pup_cplx -> cpu_foc) */
    void                **fftw_plans_inv; /**< Per-thread in-place 2D backward fftwf_plan array */
    cacaosim_det_pool_t   det_pool;       /**< Thread-local xoshiro128++ detector PRNG pool */

    /* Opaque CUDA state allocated by cacaosim_pyrwfs_cuda.cu */
    void                 *cuda_priv;      /**< Pointer to pyrwfs_cuda_state_t (NULL if CPU-only) */
} cacaosim_pyrwfs_ctx_t;

/**
 * cacaosim_pyrwfs_init() - Initialize Pyramid WFS masks, FFT plans, and flat-wavefront flux norm
 * @ctx:        Unallocated output Pyramid WFS context to populate (non-NULL)
 * @cfg:        Optical and detector configuration parameters (non-NULL)
 * @pup_amp:    Static 2D telescope pupil amplitude [nwf * nwf] (non-NULL)
 * @ncpa_opd:   Optional static WFS-path NCPA OPD map [nwf * nwf] in microns (NULL -> zero NCPA)
 * @ext_stream: Optional external `cudaStream_t` to share in fused mode (NULL creates private)
 *
 * Role:
 *   Cold-path setup entry point for the Pyramid WFS engine. Allocates 64-byte aligned CPU buffers,
 *   plans 2D forward/backward FFTW3f transforms, synthesizes the initial shifted complex pyramid
 *   mask stack, calibrates `flat_flux_norm` on an unaberrated wavefront, and optionally initializes
 *   the CUDA cuFFT + CUDA Graph backend.
 *
 * Rationale:
 *   Concentrating all heap allocations, FFTW planning, mask trigonometry, flat-field photometric
 *   calibration, and GPU cuFFT workspace warmup in initialization guarantees deterministic,
 *   allocation-free hot-loop execution. Calibrating `flat_flux_norm` empirically through the exact
 *   field-stop and pyramid diffraction operator ensures `det.flux_ph` maps directly to detected
 *   guide-star photo-events regardless of `pad_factor`, `pup_sep`, or `ncam` binning.
 *
 * Return: 0 on success, -1 on invalid dimensions or memory/plan allocation failure.
 */
int cacaosim_pyrwfs_init(
    cacaosim_pyrwfs_ctx_t       *ctx,
    const cacaosim_pyrwfs_cfg_t *cfg,
    const float                 *pup_amp,
    const float                 *ncpa_opd,
    void                        *ext_stream);

/**
 * cacaosim_pyrwfs_free() - Destroy FFT plans and free all host/device Pyramid WFS buffers
 * @ctx: Pyramid WFS context to release (safe on NULL or partially initialized contexts)
 *
 * Role:
 *   Cold-path teardown routine releasing GPU CUDA Graph/cuFFT/device state, CPU FFTW3f plans,
 *   detector PRNG state, and aligned host buffers, then zeroing `@ctx`.
 *
 * Rationale:
 *   Provides idempotent cleanup for both normal shutdown and mid-initialization rollback paths in
 *   `cacaosim_pyrwfs_init()`.
 */
void cacaosim_pyrwfs_free(cacaosim_pyrwfs_ctx_t *ctx);

/**
 * cacaosim_pyrwfs_update_masks() - Rebuild shifted focal-plane pyramid masks for new modulation
 * @ctx:        Initialized Pyramid WFS context (non-NULL)
 * @mod_rad_ld: Updated circular tip-tilt modulation radius [lambda/D] (0 = unmodulated)
 *
 * Role:
 *   Cold/warm-path mask synthesis routine that recomputes all `eff_npts` shifted complex 4-sided
 *   pyramid focal masks on host memory and uploads them to the GPU (invalidating any cached CUDA
 *   Graph) when `mod_rad_ld` changes at runtime.
 *
 * Rationale:
 *   AO operators routinely adjust Pyramid modulation radius live (e.g., wide modulation during
 *   loop acquisition, tighter or zero modulation after loop closure, or dynamic calibration).
 *   Pre-baking the Fourier shift, separable subpixel-integrated prism slopes, field stop, and
 *   `1 / (nfft^2 * sqrt(npts))` amplitude normalization into `masks_cplx` keeps steady-state frames
 *   free of trigonometric mask evaluation while still supporting live `mod_rad` updates.
 */
void cacaosim_pyrwfs_update_masks(
    cacaosim_pyrwfs_ctx_t *ctx,
    float                  mod_rad_ld);

/**
 * cacaosim_pyrwfs_accum_subframe() - Propagate one wavefront+DM subframe into WFS accumulator
 * @ctx:       Initialized Pyramid WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf] (used on CPU; GPU uses uploaded `d_pup_amp`)
 * @atm_amp:   Optional host 2D atmospheric scintillation amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] in `cfg.wf_unit` (NULL -> 0.0)
 * @dm_opd:    Optional host 2D reflective DM OPD [nwf * nwf] in microns, subtracted (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D static NCPA OPD [nwf * nwf] in microns (CPU path; NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD pointer [nwf * nwf] for zero-copy GPU fusion (NULL -> H2D)
 * @turb_gain: Live scalar multiplier applied to `@wf_in` (0.0 disables turbulence for calibration)
 * @mod_rad:   Live circular modulation radius [lambda/D] (triggers mask rebuild if changed)
 *
 * Role:
 *   Hot-loop optical propagation step for standalone/split-stream mode (`wf_in` - `dm_opd` +
 *   `ncpa_opd`). Evaluates one coherent diffraction subframe across all modulation points and
 *   accumulates the resulting binned detector intensity into `@ctx` without triggering readout.
 *
 * Rationale:
 *   Atmospheric turbulence and DM hardware often evolve at a finer simulation rate (e.g. 2-4 kHz)
 *   than the WFS camera exposure (`navg > 1`). Separating coherent subframe intensity integration
 *   from noisy electronic readout models physical temporal smearing of speckles and modulation
 *   during a finite camera exposure while paying detector noise/transfer cost only once per frame.
 */
void cacaosim_pyrwfs_accum_subframe(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *ncpa_opd,
    const float           *d_dm_opd,
    float                  turb_gain,
    float                  mod_rad);

/**
 * cacaosim_pyrwfs_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into WFS
 * @ctx:        Initialized Pyramid WFS context (non-NULL)
 * @beam_opd:   Optional host 2D beam OPD slice [nwf * nwf] in microns (NULL if `@d_beam_opd` set)
 * @beam_amp:   Optional host 2D beam amplitude slice [nwf * nwf] (NULL if `@d_beam_amp` set)
 * @ncpa_opd:   Optional host 2D WFS-arm static NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_beam_opd: Optional device 2D beam OPD pointer [nwf * nwf] for zero-copy GPU bench pipelines
 * @d_beam_amp: Optional device 2D beam amplitude pointer [nwf * nwf] for zero-copy GPU pipelines
 * @mod_rad:    Live circular modulation radius [lambda/D] (triggers mask rebuild if changed)
 *
 * Role:
 *   Hot-loop optical propagation step for modular optical-bench mode, consuming a unified
 *   `[OPD_um, Amplitude]` beam produced upstream by `mkbeam`, `dmsim`, `wfe`, or `beamsplit`.
 *
 * Rationale:
 *   In multi-stage or dichroic-split AO benches, upstream pupil masks, spiders, registration
 *   shifts, and DM residuals are already encoded into the propagating beam's dynamic amplitude and
 *   OPD slices. Accepting both host SHM pointers and resident GPU device pointers enables either
 *   inter-process SHM modularity or zero-PCIe-copy execution inside `cacaosim.htsim`.
 */
void cacaosim_pyrwfs_accum_beam(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *ncpa_opd,
    const float           *d_beam_opd,
    const float           *d_beam_amp,
    float                  mod_rad);

/**
 * cacaosim_pyrwfs_accum_beam_poly() - Propagate 1 or M polychromatic beam channels into PyWFS
 * @ctx:         Initialized Pyramid WFS context (non-NULL)
 * @beam_opd:    Host beam OPD cube (`@beam_nlam` planes separated by `@chan_stride`)
 * @beam_amp:    Host beam amplitude cube (`@beam_nlam` planes separated by `@chan_stride`)
 * @beam_lam_um: Optional array `[beam_nlam]` of upstream channel wavelengths `[um]`
 * @beam_nlam:   Number of coherent upstream channels (`1` delegates to `accum_beam`)
 * @chan_stride: Float stride between consecutive input beam channels
 * @ncpa_opd:    Optional host 2D static WFS-arm NCPA OPD `[nwf * nwf]` in `[um]`
 * @d_beam_opd:  Optional device beam OPD cube (`@beam_nlam` planes at `@chan_stride`)
 * @d_beam_amp:  Optional device beam amplitude cube
 * @mod_rad:     Live circular modulation radius `[lambda0/D]`
 */
void cacaosim_pyrwfs_accum_beam_poly(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *beam_lam_um,
    uint32_t               beam_nlam,
    uint64_t               chan_stride,
    const float           *ncpa_opd,
    const float           *d_beam_opd,
    const float           *d_beam_amp,
    float                  mod_rad);

/**
 * cacaosim_pyrwfs_set_weights() - Update incoherent channel weights from live beam spectrum
 * @ctx:     Initialized Pyramid WFS context (non-NULL)
 * @weights: Array `[ctx->eff_nlam]` of positive photon weights (renormalized internally)
 */
void cacaosim_pyrwfs_set_weights(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *weights);

/**
 * cacaosim_pyrwfs_set_disp() - Update chromatic angular dispersion OPD slopes & input poly mode
 */
void cacaosim_pyrwfs_set_disp(
    cacaosim_pyrwfs_ctx_t *ctx,
    const cacaosim_disp_t *disp,
    float                  tel_diam_m,
    int32_t                in_poly_mode);

/**
 * cacaosim_pyrwfs_readout() - Finalize accumulated WFS exposure and perform detector readout
 * @ctx:      Initialized Pyramid WFS context with `subframe_cnt >= 1` (no-op if 0)
 * @det_live: Optional live detector configuration override (NULL uses `ctx->cfg.det`)
 * @out_u16:  Output host `uint16_t` camera buffer [ncam * ncam] when `out_dtype == 0` (else NULL)
 * @out_f32:  Output host `float` camera buffer [ncam * ncam] when `out_dtype != 0` (else NULL)
 *
 * Role:
 *   Hot-loop camera frame completion step. Averages the `subframe_cnt` accumulated optical
 *   intensities, applies photometric scaling (`flux_ph / flat_flux_norm`), shutter gating, Poisson
 *   shot noise, dark current, Gaussian readout noise, gain/bias conversion, and ADU saturation,
 *   writes the output pixels, and resets `subframe_cnt` to zero.
 *
 * Rationale:
 *   Accepting a `@det_live` struct per readout allows live runtime updates to guide-star flux,
 *   spectral bandpass photometry, RON, EMCCD/CMOS gain, bias, and dark-frame shutter state without
 *   re-planning FFTs or touching optical state.
 */
void cacaosim_pyrwfs_readout(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

void cacaosim_pyrwfs_readout_async(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/**
 * cacaosim_pyrwfs_step() - Single-subframe Pyramid WFS propagation and camera readout
 * @ctx:       Initialized Pyramid WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf]
 * @atm_amp:   Optional host 2D atmospheric amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] (NULL -> 0.0)
 * @dm_opd:    Optional host 2D DM OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD [nwf * nwf] (NULL -> H2D copy of `@dm_opd`)
 * @turb_gain: Turbulence amplitude multiplier
 * @mod_rad:   Circular modulation radius [lambda/D]
 * @det_live:  Optional live detector readout parameters (NULL uses `ctx->cfg.det`)
 * @out_u16:   Output `uint16_t` frame [ncam * ncam] when `out_dtype == 0`
 * @out_f32:   Output `float` frame [ncam * ncam] when `out_dtype != 0`
 *
 * Role:
 *   Convenience hot-loop wrapper executing one `cacaosim_pyrwfs_accum_subframe()` immediately
 *   followed by `cacaosim_pyrwfs_readout()` (`navg == 1`).
 *
 * Rationale:
 *   Provides a concise single-call API for unit tests, calibration matrix measurement, and
 *   single-rate synchronous AO loops where one optical step equals one WFS camera exposure.
 */
void cacaosim_pyrwfs_step(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    float                     mod_rad,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/*
 * Internal CUDA backend entry points implemented in `cacaosim_pyrwfs_cuda.cu`.
 * Invoked exclusively by `cacaosim_pyrwfs_engine.c` when `ctx->use_cuda` is active.
 */

/**
 * cacaosim_pyrwfs_cuda_init() - Allocate GPU buffers, upload static maps/masks, and plan cuFFTs
 * @ctx:        Host Pyramid WFS context with CPU masks and dimensions initialized
 * @pup_amp:    Static 2D pupil amplitude [nwf * nwf] to upload to device
 * @ncpa_opd:   Optional static 2D NCPA OPD [nwf * nwf] in microns to upload (NULL -> none)
 * @ext_stream: Optional external `cudaStream_t` (NULL creates a private non-blocking stream)
 *
 * Role:
 *   Cold-path GPU backend initializer called by `cacaosim_pyrwfs_init()`.
 *
 * Rationale:
 *   Pre-allocates all device memory, creates the 2D forward `cufftPlan2d` and `eff_npts`-batched
 *   inverse `cufftPlanMany`, and runs one synchronous warmup pass so cuFFT internal workspaces are
 *   allocated prior to subsequent `cudaStreamBeginCapture` graph recording.
 *
 * Return: 0 on success, -1 on any CUDA or cuFFT failure.
 */
int  cacaosim_pyrwfs_cuda_init(cacaosim_pyrwfs_ctx_t *ctx,
                               const float           *pup_amp,
                               const float           *ncpa_opd,
                               void                  *ext_stream);

/**
 * cacaosim_pyrwfs_cuda_upload_masks() - Copy rebuilt host pyramid mask stack to GPU memory
 * @ctx: Pyramid WFS context with updated `ctx->masks_cplx`
 *
 * Role:
 *   Synchronizes updated shifted pyramid focal masks to `d_masks` and invalidates any captured
 *   CUDA Graph whenever modulation radius changes.
 *
 * Rationale:
 *   Separates occasional live `mod_rad` updates from per-frame steady-state execution while
 *   keeping device mask contents coherent with host state.
 */
void cacaosim_pyrwfs_cuda_upload_masks(cacaosim_pyrwfs_ctx_t *ctx);
void cacaosim_pyrwfs_cuda_upload_weights(cacaosim_pyrwfs_ctx_t *ctx);
void cacaosim_pyrwfs_cuda_upload_disp(cacaosim_pyrwfs_ctx_t *ctx);
void cacaosim_pyrwfs_cuda_accum_poly(cacaosim_pyrwfs_ctx_t *ctx,
                                     const float           *beam_opd,
                                     const float           *beam_amp,
                                     const float           *beam_lam_um,
                                     uint32_t               beam_nlam,
                                     uint64_t               chan_stride,
                                     const float           *d_beam_opd,
                                     const float           *d_beam_amp);

void cacaosim_pyrwfs_cuda_accum_full(cacaosim_pyrwfs_ctx_t *ctx,
                                     const float           *dyn_pup_amp,
                                     const float           *atm_amp,
                                     const float           *wf_in,
                                     const float           *dm_opd,
                                     const float           *d_pup_amp_ext,
                                     const float           *d_wf_opd_ext,
                                     const float           *d_dm_opd_ext,
                                     float                  wf_to_opd);

void cacaosim_pyrwfs_cuda_accum_poly_full(cacaosim_pyrwfs_ctx_t *ctx,
                                          const float           *beam_opd,
                                          const float           *beam_amp,
                                          const float           *beam_lam_um,
                                          uint32_t               beam_nlam,
                                          uint64_t               chan_stride,
                                          const float           *d_beam_opd,
                                          const float           *d_beam_amp);

/**
 * cacaosim_pyrwfs_cuda_free() - Release all GPU plans, CUDA Graphs, device buffers, and streams
 * @ctx: Pyramid WFS context whose `cuda_priv` state will be destroyed
 *
 * Role:
 *   Cold-path GPU resource teardown called by `cacaosim_pyrwfs_free()` or init failure rollback.
 *
 * Rationale:
 *   Ensures deterministic cleanup of cuFFT plans, instantiated CUDA Graphs, device allocations,
 *   and privately owned CUDA streams without leaking VRAM across bench reconfigurations.
 */
void cacaosim_pyrwfs_cuda_free(cacaosim_pyrwfs_ctx_t *ctx);

/**
 * cacaosim_pyrwfs_cuda_accum() - Execute or replay GPU Pyramid WFS subframe propagation
 * @ctx:           Active Pyramid WFS context (`use_cuda == 1`)
 * @dyn_pup_amp:   Optional dynamic host pupil amplitude [nwf * nwf] (H2D copied if non-NULL)
 * @atm_amp:       Optional host atmospheric amplitude [nwf * nwf] (H2D copied if non-NULL)
 * @wf_in:         Optional host input wavefront [nwf * nwf] (H2D copied if non-NULL)
 * @dm_opd:        Optional host DM OPD [nwf * nwf] (H2D copied if non-NULL)
 * @d_pup_amp_ext: Optional zero-copy device pupil/beam amplitude pointer [nwf * nwf]
 * @d_wf_opd_ext:  Optional zero-copy device wavefront/beam OPD pointer [nwf * nwf]
 * @d_dm_opd_ext:  Optional zero-copy device DM OPD pointer [nwf * nwf]
 * @wf_to_opd:     Multiplicative scale converting `@wf_in` / `@d_wf_opd_ext` to microns OPD
 *
 * Role:
 *   Hot-loop GPU optical propagation and multi-subframe accumulation dispatcher.
 *
 * Rationale:
 *   Transparently selects between resident device pointers (zero PCIe traffic in fused `htsim`) and
 *   asynchronous H2D staging (standalone FPS mode), replays a cached `cudaGraphExec_t` when pointer
 *   signatures match, and lazily accumulates into `d_accum_int` only when `navg > 1`.
 */
void cacaosim_pyrwfs_cuda_accum(cacaosim_pyrwfs_ctx_t *ctx,
                                const float           *dyn_pup_amp,
                                const float           *atm_amp,
                                const float           *wf_in,
                                const float           *dm_opd,
                                const float           *d_pup_amp_ext,
                                const float           *d_wf_opd_ext,
                                const float           *d_dm_opd_ext,
                                float                  wf_to_opd);

/**
 * cacaosim_pyrwfs_cuda_readout() - Normalize GPU exposure, apply detector noise, and copy to host
 * @ctx:      Active Pyramid WFS context (`use_cuda == 1`)
 * @det_live: Active detector configuration for this readout
 * @out_u16:  Host `uint16_t` destination [ncam * ncam] when `det_live->out_dtype == 0`
 * @out_f32:  Host `float` destination [ncam * ncam] when `det_live->out_dtype != 0`
 *
 * Role:
 *   Hot-loop GPU frame completion and Device-to-Host readout transfer.
 *
 * Rationale:
 *   Executes Philox-based parallel Poisson + Gaussian readout noise and ADU quantization entirely
 *   on GPU (`cacaosim_cuda_detector_readout`), transferring only the compact `ncam x ncam` output
 *   image across PCIe once per WFS frame.
 */
void cacaosim_pyrwfs_cuda_readout(cacaosim_pyrwfs_ctx_t    *ctx,
                                  const cacaosim_det_cfg_t *det_live,
                                  uint16_t                 *out_u16,
                                  float                    *out_f32);

void cacaosim_pyrwfs_cuda_readout_ex(cacaosim_pyrwfs_ctx_t    *ctx,
                                     const cacaosim_det_cfg_t *det_live,
                                     uint16_t                 *out_u16,
                                     float                    *out_f32,
                                     int                       sync_stream);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_PYRWFS_H */
