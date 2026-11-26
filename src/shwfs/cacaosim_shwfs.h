// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs.h
 * @brief   Diffractive Shack-Hartmann WFS simulator (OpenMP L1 FFTW3f + batched cuFFT)
 *
 * Architectural Role in cacaosim:
 *   Public C API, lenslet-array configuration schema, and persistent CPU/GPU execution state for
 *   the Diffractive Shack-Hartmann Wavefront Sensor compute unit (`cacaosim.shwfs`). Shared by the
 *   standalone MILK FPS wrapper (`cacaosim_shwfs_FPS.c`) and the fused multi-rate optical bench
 *   simulator (`cacaosim.htsim`).
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Full Diffractive Subaperture Propagation vs Geometric Gradient Models:
 *      Geometric Shack-Hartmann simulators approximate each lenslet spot as a rigid shifted
 *      analytical template driven by the subaperture-averaged phase gradient. That approximation
 *      breaks down in realistic AO regimes: it misses diffraction-limited spot broadening on small
 *      subapertures, multi-speckle breakup and non-linear centroid response at high wavefront
 *      error, scintillation-induced spot distortion, and asymmetric spot elongation/flux loss on
 *      subapertures partially vignetted by the outer pupil stop, central obscuration, or spiders.
 *      `cacaosim_shwfs` instead performs exact Fraunhofer focal-plane diffraction on every
 *      subaperture `(sx, sy)` across the `n_sub x n_sub` lenslet array.
 *   2. L1-Cache-Resident OpenMP CPU Engine & Batched `cufftPlanMany` GPU Engine:
 *      Because each subaperture FFT (`nfft_sub`, typically 16..32) occupies only `2..8 KiB`, the
 *      CPU backend gives each OpenMP thread a private complex tile and pre-planned 2D `fftwf_plan`
 *      that remain resident in L1 data cache with zero inter-thread contention, skipping dark
 *      subapertures outside the pupil altogether. On GPU, where launching hundreds or thousands of
 *      tiny 2D FFTs individually would starve SMs and saturate launch queues, all `n_sub * n_sub`
 *      oversampled subaperture patches are gathered in parallel and transformed in one batched 2D
 *      `cufftPlanMany` call.
 *   3. Quad-Cell Half-Bin Spot Centering & Multi-Subframe Integration (`navg`):
 *      When `pix_per_sub` is even (e.g. `2x2` or `4x4` quad-cells, `8x8` CoG windows), a separable
 *      1D pupil phase ramp shifts each subaperture DFT by `-0.5` focal bin so an unaberrated spot
 *      centers symmetrically on `(pix_per_sub - 1) / 2.0` at the intersection of the central `2x2`
 *      pixels. Multi-subframe (`navg`) accumulation averages fast turbulence/DM evolution prior to
 *      single-frame Poisson and Gaussian readout noise injection.
 */

#ifndef CACAOSIM_SHWFS_H
#define CACAOSIM_SHWFS_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"
#include "common/cacaosim_disp.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optical and detector configuration for the Diffractive Shack-Hartmann WFS.
 *
 * Specifies input pupil sampling (`nwf`), lenslet array geometry (`n_sub`, `pix_per_sub`,
 * `pad_factor`), sensing wavelength, input wavefront physical unit convention, compute backend
 * selection, and camera readout electronics parameters. Output camera dimensions equal
 * `(n_sub * pix_per_sub) x (n_sub * pix_per_sub)`.
 */
typedef struct
{
    uint32_t           nwf;         /**< Input square pupil grid dimension [pixels] (>= 8) */
    uint32_t           n_sub;       /**< Number of lenslets across pupil grid (e.g. 16, >= 1) */
    uint32_t           pix_per_sub; /**< Detector pixels per lenslet side (>= 2; ncam = n_sub*P) */
    uint32_t           pad_factor;  /**< Subaperture FFT oversampling factor (>= 2, default: 2) */
    float              pup_diam;    /**< Clear pupil diameter fraction of nwf (default: 0.9375) */
    float              lambda_um;   /**< Central sensing wavelength [um] (default: 0.75) */
    float              bw_frac;     /**< Fractional bandpass dlambda/lambda0 (0 = monochromatic) */
    int32_t            nlambda;     /**< Spectral slices (0 = auto -> 2 if bw_frac > 0) */
    int32_t            has_ext_chan;/**< 1 to use ext_lam_um / ext_weight channel tables */
    float              ext_lam_um[CACAOSIM_MAX_LAMBDA]; /**< Optional channel wavelengths [um] */
    float              ext_weight[CACAOSIM_MAX_LAMBDA]; /**< Optional channel photon weights */
    int32_t            wf_unit;     /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda;  /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;    /**< Base 64-bit PRNG seed for photon/readout noise */
    int32_t            backend;     /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;         /**< Integrated WFS camera readout electronics parameters */
} cacaosim_shwfs_cfg_t;

/**
 * @brief Persistent CPU and CUDA state for the Diffractive Shack-Hartmann WFS.
 *
 * Caches precomputed integer subaperture pupil bounds (`sub_off`, `sub_len`), the separable 1D
 * half-bin spot-centering phasor (`shift_1d_cplx`), flat-wavefront photometric normalization,
 * L1-resident per-thread CPU FFTW3f plans/buffers, thread-local detector PRNG states, and an
 * opaque pointer to the batched GPU cuFFT state (`shwfs_cuda_state_t`).
 */
typedef struct
{
    cacaosim_shwfs_cfg_t cfg;           /**< Copy of initialization optical/detector config */
    uint32_t             ncam;          /**< Monolithic square camera size (`n_sub*pix_per_sub`) */
    uint32_t             w_sub_max;     /**< Maximum pupil pixels spanned by any 1D lenslet */
    uint32_t             nfft_sub;      /**< Even padded 2D FFT dimension per lenslet patch */
    uint32_t             nbatch;        /**< Total 2D lenslet count (`n_sub * n_sub`) */
    uint64_t             npix_wf;       /**< Total input pupil pixels (`nwf * nwf`) */
    uint64_t             npix_sub_fft;  /**< Complex samples per padded lenslet (`nfft_sub^2`) */
    uint64_t             npix_cam;      /**< Total output camera pixels (`ncam * ncam`) */
    int32_t              eff_nlam;      /**< Active spectral slices [1..CACAOSIM_MAX_LAMBDA] */
    float                lam_um[CACAOSIM_MAX_LAMBDA];        /**< Channel wavelengths [um] */
    float                chan_w[CACAOSIM_MAX_LAMBDA];        /**< Normalized channel weights */
    float                k_rad[CACAOSIM_MAX_LAMBDA];         /**< 2*pi / lam_um[c] [rad/um] */
    float                scale_c[CACAOSIM_MAX_LAMBDA];       /**< Focal scale lam0 / lam_um[c] */
    float                chan_w_jac[CACAOSIM_MAX_LAMBDA];    /**< chan_w[c] * scale_c[c]^2 */
    float                chan_flat_norm[CACAOSIM_MAX_LAMBDA];/**< Per-channel flat intensity */
    int32_t              disp_active;                         /**< 1 if chromatic disp active */
    uint64_t             disp_gen;                            /**< Cached dispersion gen */
    float                disp_tel_diam;                       /**< Cached tel_diam_m [m] */
    int32_t              in_poly_mode;                        /**< cacaosim_beam_poly_mode_t */
    float                disp_sx[CACAOSIM_MAX_LAMBDA];        /**< X OPD slope [um/pix] */
    float                disp_sy[CACAOSIM_MAX_LAMBDA];        /**< Y OPD slope [um/pix] */
    float                flat_flux_norm;/**< Unaberrated total camera intensity for photometry */
    int32_t              use_cuda;      /**< 1 when batched cuFFT GPU backend is active, else 0 */
    uint32_t             subframe_cnt;  /**< Optical subframes accumulated in current exposure */
    uint64_t             frame_seq;     /**< Monotonic camera readout counter for PRNG seeding */

    /* Precomputed subaperture pupil bounds [n_sub] and half-bin centering phasor [nfft_sub * 2] */
    uint32_t            *sub_off;       /**< Starting pupil index along X/Y per lenslet [n_sub] */
    uint32_t            *sub_len;       /**< Pupil pixel width along X/Y per lenslet [n_sub] */
    float               *shift_1d_cplx; /**< 1D complex half-bin centering ramp [nfft_sub * 2] */

    /* CPU per-thread L1 FFT buffers and plans */
    int32_t              cpu_nth;       /**< Capped OpenMP thread count (<= 16) */
    float               *cpu_thr_cplx;  /**< L1-resident thread FFT tiles [cpu_nth*nfft_sub^2*2] */
    float               *cpu_thr_int;   /**< L1-resident thread spot intensity [cpu_nth*nfft^2] */
    void               **fftw_plans;    /**< Per-thread in-place 2D forward `fftwf_plan` handles */
    float               *cpu_cam_int;   /**< Single-subframe stitched spot image [npix_cam] */
    float               *cpu_accum_int; /**< Multi-subframe intensity accumulator [npix_cam] */
    cacaosim_det_pool_t  det_pool;      /**< Thread-local xoshiro128++ detector PRNG pool */

    /* Opaque CUDA state allocated by cacaosim_shwfs_cuda.cu */
    void                *cuda_priv;     /**< Pointer to `shwfs_cuda_state_t` (NULL if CPU-only) */
} cacaosim_shwfs_ctx_t;

/**
 * cacaosim_shwfs_init() - Initialize Shack-Hartmann lenslet geometry, FFT plans, and flux norm
 * @ctx:        Unallocated output Shack-Hartmann WFS context to initialize (non-NULL)
 * @cfg:        Optical and detector configuration parameters (non-NULL)
 * @pup_amp:    Static 2D telescope pupil amplitude [nwf * nwf] (non-NULL)
 * @ncpa_opd:   Optional static WFS-path NCPA OPD [nwf * nwf] in microns (NULL -> zero NCPA)
 * @ext_stream: Optional external `cudaStream_t` to share in fused mode (NULL creates private)
 *
 * Role:
 *   Cold-path constructor computing subaperture pupil partitions, half-bin centering phasors,
 *   per-thread CPU FFTW3f plans, flat-wavefront photometric calibration (`flat_flux_norm`), and
 *   optional GPU batched `cufftPlanMany` resources.
 *
 * Rationale:
 *   Precomputing exact integer subaperture bounds `[(s*nwf)/n_sub, ((s+1)*nwf)/n_sub)` supports
 *   arbitrary `nwf` and `n_sub` ratios without runtime division, while calibrating `flat_flux_norm`
 *   on the cropped `pix_per_sub x pix_per_sub` focal windows ensures `det.flux_ph` accurately sets
 *   detected photons inside the subaperture readout boxes.
 *
 * Return: 0 on success, -1 on invalid geometry or memory/plan allocation failure.
 */
int cacaosim_shwfs_init(
    cacaosim_shwfs_ctx_t       *ctx,
    const cacaosim_shwfs_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream);

/**
 * cacaosim_shwfs_free() - Release all host and device Shack-Hartmann WFS resources
 * @ctx: Shack-Hartmann WFS context to release (safe on NULL or partially initialized contexts)
 *
 * Role:
 *   Cold-path destructor tearing down GPU batched cuFFT buffers, CPU FFTW3f plans, PRNG state, and
 *   aligned host memory, leaving `@ctx` zeroed.
 *
 * Rationale:
 *   Guarantees leak-free cleanup across both normal process shutdown and error rollback paths in
 *   `cacaosim_shwfs_init()`.
 */
void cacaosim_shwfs_free(cacaosim_shwfs_ctx_t *ctx);

/**
 * cacaosim_shwfs_accum_subframe() - Propagate one wavefront+DM subframe into WFS accumulator
 * @ctx:       Initialized Shack-Hartmann WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf] (CPU path; GPU uses uploaded `d_pup_amp`)
 * @atm_amp:   Optional host 2D atmospheric scintillation amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] in `cfg.wf_unit` (NULL -> 0.0)
 * @dm_opd:    Optional host 2D reflective DM OPD [nwf * nwf] in microns, subtracted (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D static NCPA OPD [nwf * nwf] in microns (CPU path; NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD pointer [nwf * nwf] for GPU zero-copy fusion (NULL -> H2D)
 * @turb_gain: Live scalar multiplier applied to `@wf_in` (0.0 disables turbulence for calibration)
 *
 * Role:
 *   Hot-loop split-stream optical propagation step computing diffractive lenslet spots for one
 *   atmospheric/DM timestep and accumulating them into `@ctx` without electronic readout.
 *
 * Rationale:
 *   Enables physical multi-subframe (`navg > 1`) spot smearing when turbulence evolves faster than
 *   the WFS camera frame rate while keeping `navg == 1` free of extra buffer copies.
 */
void cacaosim_shwfs_accum_subframe(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    const float          *d_dm_opd,
    float                 turb_gain);

/**
 * cacaosim_shwfs_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into SHWFS
 * @ctx:        Initialized Shack-Hartmann WFS context (non-NULL)
 * @beam_opd:   Optional host 2D beam OPD slice [nwf * nwf] in microns (NULL if `@d_beam_opd` set)
 * @beam_amp:   Optional host 2D beam amplitude slice [nwf * nwf] (NULL if `@d_beam_amp` set)
 * @ncpa_opd:   Optional host 2D WFS-arm static NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_beam_opd: Optional device 2D beam OPD pointer [nwf * nwf] for zero-copy GPU bench pipelines
 * @d_beam_amp: Optional device 2D beam amplitude pointer [nwf * nwf] for zero-copy GPU pipelines
 *
 * Role:
 *   Hot-loop optical-bench propagation step consuming a unified `[OPD_um, Amplitude]` beam from
 *   upstream `mkbeam`, `dmsim`, `wfe`, or `beamsplit` units.
 *
 * Rationale:
 *   Propagates dynamic upstream pupil stops, spider rotation/registration, and multi-DM residuals
 *   directly into the subaperture diffraction integrals without intermediate SHM round-trips when
 *   device pointers are provided.
 */
void cacaosim_shwfs_accum_beam(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_shwfs_accum_beam_poly() - Propagate 1 or M polychromatic beam channels into SHWFS
 * @ctx:         Initialized Shack-Hartmann WFS context (non-NULL)
 * @beam_opd:    Host beam OPD cube (`@beam_nlam` planes separated by `@chan_stride`)
 * @beam_amp:    Host beam amplitude cube (`@beam_nlam` planes separated by `@chan_stride`)
 * @beam_lam_um: Optional array `[beam_nlam]` of upstream channel wavelengths `[um]`
 * @beam_nlam:   Number of coherent upstream channels (`1` delegates to `accum_beam`)
 * @chan_stride: Float stride between consecutive input beam channels
 * @ncpa_opd:    Optional host 2D static WFS-arm NCPA OPD `[nwf * nwf]` in `[um]`
 * @d_beam_opd:  Optional device beam OPD cube (`@beam_nlam` planes at `@chan_stride`)
 * @d_beam_amp:  Optional device beam amplitude cube
 */
void cacaosim_shwfs_accum_beam_poly(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *beam_lam_um,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_shwfs_set_weights() - Update incoherent channel weights from live beam spectrum
 * @ctx:     Initialized Shack-Hartmann WFS context (non-NULL)
 * @weights: Array `[ctx->eff_nlam]` of positive photon weights (renormalized internally)
 */
void cacaosim_shwfs_set_weights(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *weights);

/**
 * cacaosim_shwfs_set_disp() - Update chromatic angular dispersion OPD slopes & input poly mode
 */
void cacaosim_shwfs_set_disp(
    cacaosim_shwfs_ctx_t  *ctx,
    const cacaosim_disp_t *disp,
    float                  tel_diam_m,
    int32_t                in_poly_mode);

/**
 * cacaosim_shwfs_readout() - Finalize accumulated SHWFS exposure and run detector readout
 * @ctx:      Initialized Shack-Hartmann WFS context with `subframe_cnt >= 1` (no-op if 0)
 * @det_live: Optional live detector configuration override (NULL uses `ctx->cfg.det`)
 * @out_u16:  Output host `uint16_t` camera buffer [ncam * ncam] when `out_dtype == 0` (else NULL)
 * @out_f32:  Output host `float` camera buffer [ncam * ncam] when `out_dtype != 0` (else NULL)
 *
 * Role:
 *   Hot-loop camera frame completion step averaging `subframe_cnt` accumulated spot images,
 *   applying photon scaling, Poisson shot noise, dark current, Gaussian RON, gain/bias, and ADU
 *   saturation, and resetting `subframe_cnt` to zero.
 *
 * Rationale:
 *   Decouples electronic camera parameters (`@det_live`) from optical lenslet diffraction so flux,
 *   noise, and shutter settings can change on the fly at runtime.
 */
void cacaosim_shwfs_readout(
    cacaosim_shwfs_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/**
 * cacaosim_shwfs_step() - Single-subframe SHWFS propagation and camera readout
 * @ctx:       Initialized Shack-Hartmann WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf]
 * @atm_amp:   Optional host 2D atmospheric amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] (NULL -> 0.0)
 * @dm_opd:    Optional host 2D DM OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD [nwf * nwf] (NULL -> H2D copy)
 * @turb_gain: Turbulence amplitude multiplier
 * @det_live:  Optional live detector readout parameters (NULL uses `ctx->cfg.det`)
 * @out_u16:   Output `uint16_t` frame [ncam * ncam] when `out_dtype == 0`
 * @out_f32:   Output `float` frame [ncam * ncam] when `out_dtype != 0`
 *
 * Role:
 *   Single-call convenience wrapper performing one `cacaosim_shwfs_accum_subframe()` followed
 *   immediately by `cacaosim_shwfs_readout()`.
 *
 * Rationale:
 *   Streamlines synchronous `navg == 1` loops, slope-response calibration, and unit tests.
 */
void cacaosim_shwfs_step(
    cacaosim_shwfs_ctx_t     *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/*
 * Internal CUDA backend entry points implemented in `cacaosim_shwfs_cuda.cu`.
 * Invoked exclusively by `cacaosim_shwfs_engine.c` when `ctx->use_cuda` is active.
 */

/**
 * cacaosim_shwfs_cuda_init() - Allocate GPU buffers, upload lenslet bounds, and plan batched cuFFT
 * @ctx:        Host SHWFS context with subaperture geometry and `shift_1d_cplx` initialized
 * @pup_amp:    Static 2D pupil amplitude [nwf * nwf] to upload to device
 * @ncpa_opd:   Optional static 2D NCPA OPD [nwf * nwf] in microns to upload (NULL -> none)
 * @ext_stream: Optional external `cudaStream_t` (NULL creates a private non-blocking stream)
 *
 * Role:
 *   Cold-path GPU initializer creating the `nbatch = n_sub * n_sub` 2D `cufftPlanMany` handle and
 *   uploading static pupil, NCPA, subaperture bounds, and centering phasors to VRAM.
 *
 * Rationale:
 *   Front-loading `cufftPlanMany` and device buffer allocations guarantees zero runtime memory
 *   management on the multi-kHz GPU hot path.
 *
 * Return: 0 on success, -1 on any CUDA or cuFFT failure.
 */
int  cacaosim_shwfs_cuda_init(cacaosim_shwfs_ctx_t *ctx,
                              const float          *pup_amp,
                              const float          *ncpa_opd,
                              void                 *ext_stream);

/**
 * cacaosim_shwfs_cuda_free() - Release batched cuFFT plan, device buffers, and owned CUDA stream
 * @ctx: Shack-Hartmann WFS context whose `cuda_priv` state will be destroyed
 *
 * Role:
 *   Cold-path GPU teardown routine called by `cacaosim_shwfs_free()`.
 *
 * Rationale:
 *   Frees all GPU allocations while preserving externally borrowed streams in fused bench mode.
 */
void cacaosim_shwfs_cuda_free(cacaosim_shwfs_ctx_t *ctx);
void cacaosim_shwfs_cuda_upload_weights(cacaosim_shwfs_ctx_t *ctx);
void cacaosim_shwfs_cuda_upload_disp(cacaosim_shwfs_ctx_t *ctx);
void cacaosim_shwfs_cuda_accum_poly(cacaosim_shwfs_ctx_t *ctx,
                                    const float          *beam_opd,
                                    const float          *beam_amp,
                                    const float          *beam_lam_um,
                                    uint32_t              beam_nlam,
                                    uint64_t              chan_stride,
                                    const float          *d_beam_opd,
                                    const float          *d_beam_amp);

/**
 * cacaosim_shwfs_cuda_accum() - Run batched GPU lenslet patch synthesis, cuFFT, and spot tiling
 * @ctx:           Active SHWFS context (`use_cuda == 1`)
 * @dyn_pup_amp:   Optional dynamic host pupil/beam amplitude [nwf * nwf]
 * @atm_amp:       Optional host atmospheric amplitude [nwf * nwf]
 * @wf_in:         Optional host wavefront or beam OPD [nwf * nwf]
 * @dm_opd:        Optional host DM OPD [nwf * nwf] in microns
 * @d_pup_amp_ext: Optional zero-copy device pupil/beam amplitude pointer [nwf * nwf]
 * @d_wf_opd_ext:  Optional zero-copy device wavefront/beam OPD pointer [nwf * nwf]
 * @d_dm_opd_ext:  Optional zero-copy device DM OPD pointer [nwf * nwf]
 * @wf_to_opd:     Multiplicative scale converting `@wf_in` / `@d_wf_opd_ext` to microns OPD
 *
 * Role:
 *   Hot-loop GPU optical subframe propagator and lazy multi-subframe accumulator.
 *
 * Rationale:
 *   Executes all `n_sub * n_sub` subaperture diffractive propagations in parallel across three
 *   device stages (patch gather -> batched 2D forward cuFFT -> FFT-shifted spot tile extraction).
 */
void cacaosim_shwfs_cuda_accum(cacaosim_shwfs_ctx_t *ctx,
                               const float          *dyn_pup_amp,
                               const float          *atm_amp,
                               const float          *wf_in,
                               const float          *dm_opd,
                               const float          *d_pup_amp_ext,
                               const float          *d_wf_opd_ext,
                               const float          *d_dm_opd_ext,
                               float                 wf_to_opd);

/**
 * cacaosim_shwfs_cuda_readout() - Average GPU subframes, apply detector noise, and copy to host
 * @ctx:      Active SHWFS context (`use_cuda == 1`)
 * @det_live: Active detector configuration for this readout
 * @out_u16:  Host `uint16_t` destination [ncam * ncam] when `det_live->out_dtype == 0`
 * @out_f32:  Host `float` destination [ncam * ncam] when `det_live->out_dtype != 0`
 *
 * Role:
 *   Hot-loop GPU exposure normalization, Philox noise readout, and synchronous D2H pixel transfer.
 *
 * Rationale:
 *   Keeps all pixel-level Poisson/Gaussian random number generation on the GPU and transfers only
 *   the final quantized `ncam x ncam` detector image over PCIe.
 */
void cacaosim_shwfs_cuda_readout(cacaosim_shwfs_ctx_t     *ctx,
                                 const cacaosim_det_cfg_t *det_live,
                                 uint16_t                 *out_u16,
                                 float                    *out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_SHWFS_H */
