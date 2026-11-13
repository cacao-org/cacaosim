// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs.h
 * @brief   Babinet Matrix-DFT Zernike phase-contrast WFS simulator (OpenBLAS + cuBLAS)
 *
 * Architectural Role in cacaosim:
 *   Public C API, optical/dimple configuration schema, and persistent CPU (OpenBLAS `cblas_cgemm`)
 *   and GPU (cuBLAS `cublasCgemm`) state for the diffractive Zernike Wavefront Sensor /
 *   coronagraphic Low-Order WFS (LOWFS) compute unit (`cacaosim.zwfs`). Shared by the standalone
 *   MILK FPS wrapper (`cacaosim_zwfs_FPS.c`) and the fused multi-rate optical bench engine
 *   (`cacaosim.htsim`).
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Why Standard FFTs Fail on the Zernike Dimple:
 *      A Zernike phase-contrast WFS places a tiny circular focal-plane dimple of diameter
 *      `~1.06 lambda/D` at the PSF core to phase-shift the unaberrated core wave by `theta`
 *      (nominally `pi/2` rad), interfering it against the unshifted aberration halo to convert
 *      pupil phase directly into linear re-imaged pupil intensity at the fundamental photon-noise
 *      sensitivity limit. On an unpadded `nwf x nwf` FFT grid, `1 lambda/D` spans ~1 pixel;
 *      resolving the circular `1.06 lambda/D` dimple boundary to ~20+ pixels via standard FFTs
 *      would require `20x..32x` zero-padding (`~4096 x 4096` 2D FFTs), costing milliseconds.
 *   2. Exact Microsecond Babinet Matrix-DFT (`cblas_cgemm` / `cublasCgemm`):
 *      Decomposing the focal mask transmission via Babinet's principle as
 *      `T(u, v) = 1 - (1 - t_d * exp(i * depth)) * D(u, v)` yields the exact output pupil field:
 *        `Psi_out = Psi_pup - (1 - t_d * exp(i * depth)) * MFT^-1[D * MFT[Psi_pup]]`
 *      Because the dimple indicator `D(u, v)` has compact support strictly inside a tiny `~2
 *      lambda/D` patch, Soummer's separable 2D Matrix-DFT (`MFT`) evaluates `n_mft x n_mft`
 *      (default `32 x 32`, `8x8` subpixel anti-aliased) fine focal samples across the dimple via
 *      four tiny `[n_mft x nwf]` complex matrix multiplications (`cblas_cgemm` / `cublasCgemm`),
 *      completing exact diffractive propagation in microseconds with zero FFT padding.
 */

#ifndef CACAOSIM_ZWFS_H
#define CACAOSIM_ZWFS_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optical and detector configuration for the Zernike WFS engine.
 *
 * Configures pupil and camera grid sizes (`nwf`, `ncam`), focal-core Matrix-DFT sampling (`n_mft`),
 * Zernike phase-dimple diameter (`dimple_ld`), phase depth (`dimple_rad`), and amplitude
 * transmission (`dimple_amp`), sensing wavelength, wavefront unit scaling, compute backend, and
 * camera readout electronics.
 */
typedef struct
{
    uint32_t           nwf;        /**< Input square pupil grid dimension [pixels] (>= 8) */
    uint32_t           ncam;       /**< Output square WFS camera dimension [pixels] (>= 4) */
    uint32_t           n_mft;      /**< Focal-core MFT grid size across dimple (>= 16, def: 32) */
    float              pup_diam;   /**< Clear pupil diameter fraction of nwf (default: 0.9375) */
    float              lambda_um;  /**< Monochromatic sensing wavelength [um] (default: 0.75) */
    float              dimple_ld;  /**< Phase dimple diameter [lambda/D] (default: 1.06) */
    float              dimple_rad; /**< Phase shift inside dimple [radians] (default: pi/2) */
    float              dimple_amp; /**< Amplitude transmission inside dimple [0..1] (def: 1.0) */
    int32_t            wf_unit;    /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda; /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;   /**< Base 64-bit PRNG seed for photon/readout noise */
    int32_t            backend;    /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;        /**< Integrated WFS camera readout electronics parameters */
} cacaosim_zwfs_cfg_t;

/**
 * @brief Persistent CPU (OpenBLAS) and CUDA (cuBLAS) Zernike WFS state.
 *
 * Stores precomputed separable forward/inverse Matrix-DFT bases (`fwd_U/V`, `inv_U/V`), the
 * subpixel-integrated complex Babinet dimple mask `(t_d * exp(i*theta) - 1) * D(u, v)`, open-pupil
 * Parseval flux normalization, aligned CPU GEMM workspaces, detector PRNG pool, and an opaque
 * pointer to the GPU cuBLAS backend (`zwfs_cuda_state_t`).
 */
typedef struct
{
    cacaosim_zwfs_cfg_t cfg;            /**< Copy of initialization optical/detector config */
    uint32_t            n_mft;          /**< Clamped focal-core MFT dimension (`>= 16`) */
    uint64_t            npix_wf;        /**< Total input pupil pixels (`nwf * nwf`) */
    uint64_t            npix_mft;       /**< Total focal-core MFT samples (`n_mft * n_mft`) */
    uint64_t            npix_cam;       /**< Total output WFS camera pixels (`ncam * ncam`) */
    float               flat_flux_norm; /**< Open-pupil Parseval intensity `sum(|pup_amp|^2)` */
    int32_t             use_cuda;       /**< 1 when cuBLAS GPU backend is active, 0 for CPU BLAS */
    uint32_t            subframe_cnt;   /**< Optical subframes accumulated in current exposure */
    uint64_t            frame_seq;      /**< Monotonic camera readout counter for PRNG seeding */

    /* Precomputed MFT matrices & complex Babinet focal mask (t_d*exp(i*theta) - 1)*M(u,v) */
    float              *fwd_U_cplx;     /**< Left forward MFT matrix `[n_mft * nwf * 2]` */
    float              *fwd_V_cplx;     /**< Right forward MFT matrix `[nwf * n_mft * 2]` */
    float              *inv_U_cplx;     /**< Left inverse MFT matrix `[nwf * n_mft * 2]` */
    float              *inv_V_cplx;     /**< Right inverse MFT matrix `[n_mft * nwf * 2]` */
    float              *mask_bab_cplx;  /**< Anti-aliased Babinet dimple `[n_mft * n_mft * 2]` */

    /* CPU work buffers */
    float              *cpu_epup_cplx;  /**< Complex pupil/output field `[nwf * nwf * 2]` */
    float              *cpu_tmp_cplx;   /**< Half-transformed MFT scratch `[n_mft * nwf * 2]` */
    float              *cpu_efoc_cplx;  /**< Complex focal-dimple field `[n_mft * n_mft * 2]` */
    float              *cpu_pup_int;    /**< Unbinned re-imaged pupil intensity `[nwf * nwf]` */
    float              *cpu_cam_int;    /**< Binned single-subframe camera intensity `[ncam^2]` */
    float              *cpu_accum_int;  /**< Multi-subframe intensity accumulator `[ncam^2]` */
    cacaosim_det_pool_t det_pool;       /**< Thread-local xoshiro128++ detector PRNG pool */

    /* Opaque CUDA state allocated by cacaosim_zwfs_cuda.cu */
    void               *cuda_priv;      /**< Pointer to `zwfs_cuda_state_t` (NULL if CPU-only) */
} cacaosim_zwfs_ctx_t;

/**
 * cacaosim_zwfs_init() - Initialize Babinet MFT Zernike WFS matrices, mask, and flux norm
 * @ctx:        Unallocated output Zernike WFS context to initialize (non-NULL)
 * @cfg:        Optical, dimple, and detector configuration parameters (non-NULL)
 * @pup_amp:    Static 2D telescope or LOWFS pupil amplitude [nwf * nwf] (non-NULL)
 * @ncpa_opd:   Optional static WFS-path NCPA OPD [nwf * nwf] in microns (NULL -> zero NCPA)
 * @ext_stream: Optional external `cudaStream_t` to share in fused mode (NULL creates private)
 *
 * Role:
 *   Cold-path constructor allocating aligned MFT matrices and work buffers, synthesizing the
 *   `8x8` subpixel-integrated Babinet dimple mask, computing open-pupil photometric normalization,
 *   and optionally initializing the GPU cuBLAS backend.
 *
 * Rationale:
 *   Calibrating `flat_flux_norm` against the incident open-pupil energy `sum(|pup_amp|^2)` (rather
 *   than post-dimple detector energy) ensures physical dimple absorption (`dimple_amp < 1.0`) and
 *   wide-angle halo scattering outside the finite re-imaged pupil reduce detected photon counts
 *   realistically.
 *
 * Return: 0 on success, -1 on invalid dimensions or memory/cuBLAS initialization failure.
 */
int cacaosim_zwfs_init(
    cacaosim_zwfs_ctx_t       *ctx,
    const cacaosim_zwfs_cfg_t *cfg,
    const float               *pup_amp,
    const float               *ncpa_opd,
    void                      *ext_stream);

/**
 * cacaosim_zwfs_free() - Release all host and device Zernike WFS resources
 * @ctx: Zernike WFS context to release (safe on NULL or partially initialized contexts)
 *
 * Role:
 *   Cold-path destructor releasing cuBLAS/VRAM resources, detector PRNG state, and aligned host
 *   MFT matrices/buffers, then zeroing `@ctx`.
 *
 * Rationale:
 *   Provides idempotent cleanup for both normal shutdown and initialization error rollback.
 */
void cacaosim_zwfs_free(cacaosim_zwfs_ctx_t *ctx);

/**
 * cacaosim_zwfs_accum_subframe() - Propagate one wavefront+DM subframe into ZWFS accumulator
 * @ctx:       Initialized Zernike WFS context (non-NULL)
 * @pup_amp:   Host 2D pupil amplitude [nwf * nwf] (CPU path; GPU uses uploaded `d_pup_amp`)
 * @atm_amp:   Optional host 2D atmospheric scintillation amplitude [nwf * nwf] (NULL -> 1.0)
 * @wf_in:     Optional host 2D input wavefront [nwf * nwf] in `cfg.wf_unit` (NULL -> 0.0)
 * @dm_opd:    Optional host 2D reflective DM OPD [nwf * nwf] in microns, subtracted (NULL -> 0.0)
 * @ncpa_opd:  Optional host 2D static NCPA OPD [nwf * nwf] in microns (CPU path; NULL -> 0.0)
 * @d_dm_opd:  Optional device 2D DM OPD pointer [nwf * nwf] for GPU zero-copy fusion (NULL -> H2D)
 * @turb_gain: Live scalar multiplier applied to `@wf_in` (0.0 disables turbulence for calibration)
 *
 * Role:
 *   Hot-loop split-stream optical propagation step evaluating one Babinet MFT Zernike phase-
 *   contrast subframe and accumulating binned intensity into `@ctx` without electronic readout.
 *
 * Rationale:
 *   Supports multi-subframe (`navg > 1`) temporal integration of fast residual jitter/turbulence
 *   on slower LOWFS or high-order ZWFS cameras while keeping `navg == 1` copy-free.
 */
void cacaosim_zwfs_accum_subframe(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *pup_amp,
    const float         *atm_amp,
    const float         *wf_in,
    const float         *dm_opd,
    const float         *ncpa_opd,
    const float         *d_dm_opd,
    float                turb_gain);

/**
 * cacaosim_zwfs_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into ZWFS
 * @ctx:        Initialized Zernike WFS context (non-NULL)
 * @beam_opd:   Optional host 2D beam OPD slice [nwf * nwf] in microns (NULL if `@d_beam_opd` set)
 * @beam_amp:   Optional host 2D beam amplitude slice [nwf * nwf] (NULL if `@d_beam_amp` set)
 * @ncpa_opd:   Optional host 2D WFS-arm static NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_beam_opd: Optional device 2D beam OPD pointer [nwf * nwf] for zero-copy GPU bench pipelines
 * @d_beam_amp: Optional device 2D beam amplitude pointer [nwf * nwf] for zero-copy GPU pipelines
 *
 * Role:
 *   Hot-loop optical-bench propagation step consuming a unified `[OPD_um, Amplitude]` beam—such as
 *   a coronagraphic reflective focal-plane or Lyot-stop LOWFS pickoff (`coro.out_lowfs`) or a
 *   dichroic WFS arm (`beamsplit`).
 *
 * Rationale:
 *   Coronagraphic LOWFS beams carry strongly non-uniform, dynamically evolving pupil amplitudes
 *   (e.g. light reflected off a central Lyot/FPM spot or annular reflective Lyot stop); passing
 *   both dynamic amplitude and OPD directly into the Babinet MFT preserves exact diffraction.
 */
void cacaosim_zwfs_accum_beam(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *beam_opd,
    const float         *beam_amp,
    const float         *ncpa_opd,
    const float         *d_beam_opd,
    const float         *d_beam_amp);

/**
 * cacaosim_zwfs_readout() - Finalize accumulated ZWFS exposure and run detector readout
 * @ctx:      Initialized Zernike WFS context with `subframe_cnt >= 1` (no-op if 0)
 * @det_live: Optional live detector configuration override (NULL uses `ctx->cfg.det`)
 * @out_u16:  Output host `uint16_t` camera buffer [ncam * ncam] when `out_dtype == 0` (else NULL)
 * @out_f32:  Output host `float` camera buffer [ncam * ncam] when `out_dtype != 0` (else NULL)
 *
 * Role:
 *   Hot-loop camera frame completion step normalizing accumulated subframes, applying photon flux
 *   scaling, Poisson shot noise, dark current, Gaussian RON, gain/bias, and ADU saturation, and
 *   resetting `subframe_cnt` to zero.
 *
 * Rationale:
 *   Allows runtime updates to guide-star flux, shutter state, and detector electronics without
 *   rebuilding MFT matrices.
 */
void cacaosim_zwfs_readout(
    cacaosim_zwfs_ctx_t      *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/**
 * cacaosim_zwfs_step() - Single-subframe Zernike WFS propagation and camera readout
 * @ctx:       Initialized Zernike WFS context (non-NULL)
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
 *   Single-call convenience wrapper performing one `cacaosim_zwfs_accum_subframe()` followed
 *   immediately by `cacaosim_zwfs_readout()`.
 *
 * Rationale:
 *   Provides a compact single-frame entry point for unit tests, phase reconstruction benchmarks,
 *   and synchronous `navg == 1` loops.
 */
void cacaosim_zwfs_step(
    cacaosim_zwfs_ctx_t      *ctx,
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
 * Internal CUDA backend entry points implemented in `cacaosim_zwfs_cuda.cu`.
 * Invoked exclusively by `cacaosim_zwfs_engine.c` when `ctx->use_cuda` is active.
 */

/**
 * cacaosim_zwfs_cuda_init() - Create cuBLAS handle, allocate VRAM, and upload Babinet MFT matrices
 * @ctx:        Host ZWFS context with `fwd_U/V`, `inv_U/V`, and `mask_bab_cplx` precomputed
 * @pup_amp:    Static 2D pupil amplitude [nwf * nwf] to upload to device
 * @ncpa_opd:   Optional static 2D NCPA OPD [nwf * nwf] in microns to upload (NULL -> none)
 * @ext_stream: Optional external `cudaStream_t` (NULL creates a private non-blocking stream)
 *
 * Role:
 *   Cold-path GPU backend constructor binding `cublasHandle_t` to the execution stream and
 *   uploading all precomputed MFT operators and static pupil maps.
 *
 * Rationale:
 *   Pre-uploading the compact `[n_mft x nwf]` MFT matrices keeps them resident in GPU L2 cache
 *   across every `cublasCgemm` call in the real-time loop.
 *
 * Return: 0 on success, -1 on CUDA or cuBLAS initialization failure.
 */
int  cacaosim_zwfs_cuda_init(cacaosim_zwfs_ctx_t *ctx,
                             const float         *pup_amp,
                             const float         *ncpa_opd,
                             void                *ext_stream);

/**
 * cacaosim_zwfs_cuda_free() - Destroy cuBLAS handle, free VRAM buffers, and release owned stream
 * @ctx: Zernike WFS context whose `cuda_priv` state will be destroyed
 *
 * Role:
 *   Cold-path GPU teardown routine called by `cacaosim_zwfs_free()`.
 *
 * Rationale:
 *   Releases all cuBLAS and device memory cleanly while preserving shared external streams.
 */
void cacaosim_zwfs_cuda_free(cacaosim_zwfs_ctx_t *ctx);

/**
 * cacaosim_zwfs_cuda_accum() - Execute GPU cuBLAS Babinet MFT propagation and camera binning
 * @ctx:           Active ZWFS context (`use_cuda == 1`)
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
 *   Hot-loop GPU optical subframe propagator running four `cublasCgemm` calls, Babinet mask
 *   multiplication, area-conserving camera binning, and lazy `navg > 1` accumulation.
 *
 * Rationale:
 *   Exploits row-major/column-major GEMM transpose duality so C row-major buffers require zero
 *   explicit matrix transposes on GPU.
 */
void cacaosim_zwfs_cuda_accum(cacaosim_zwfs_ctx_t *ctx,
                              const float         *dyn_pup_amp,
                              const float         *atm_amp,
                              const float         *wf_in,
                              const float         *dm_opd,
                              const float         *d_pup_amp_ext,
                              const float         *d_wf_opd_ext,
                              const float         *d_dm_opd_ext,
                              float                wf_to_opd);

/**
 * cacaosim_zwfs_cuda_readout() - Average GPU subframes, apply detector noise, and copy to host
 * @ctx:      Active ZWFS context (`use_cuda == 1`)
 * @det_live: Active detector configuration for this readout
 * @out_u16:  Host `uint16_t` destination [ncam * ncam] when `det_live->out_dtype == 0`
 * @out_f32:  Host `float` destination [ncam * ncam] when `det_live->out_dtype != 0`
 *
 * Role:
 *   Hot-loop GPU exposure normalization, Philox detector noise synthesis, and synchronous D2H copy.
 *
 * Rationale:
 *   Executes Poisson/Gaussian noise entirely on-device and copies only the binned `ncam x ncam`
 *   output pixels back to host memory once per exposure.
 */
void cacaosim_zwfs_cuda_readout(cacaosim_zwfs_ctx_t      *ctx,
                                const cacaosim_det_cfg_t *det_live,
                                uint16_t                 *out_u16,
                                float                    *out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_ZWFS_H */
