// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm.h
 * @brief   Deformable mirror hardware physics and gather-stencil OPD engine (CPU + CUDA)
 *
 * Architectural Role:
 *   Declares the hardware-accurate Deformable Mirror (DM) and 2-axis Tip-Tilt
 *   Mirror (TTM) physics engine (`cacaosim_dm_ctx_t`) shared by the standalone
 *   `cacaosim.dmsim` FPS process, the fused multi-rate `cacaosim.htsim` optical
 *   bench engine, and latency/throughput benchmarks. Models drive electronics
 *   (stroke clipping, finite-bit DAC quantization, electrostrictive quadratic
 *   nonlinearity, dead-actuator masking), fractional-frame ring-buffer transport
 *   delay, exact discrete Zero-Order Hold (ZOH) 1st/2nd-order actuator dynamics,
 *   and 2D reflective OPD synthesis on both CPU (OpenMP/SIMD) and CUDA GPU.
 *
 * Design Rationale & Motivation:
 *   - Hybrid actuator-CPU / pixel-GPU partitioning: Actuator state dimension
 *     `N_act` (`2` for TTM up to `~4096` for XAO DMs) is tiny compared to 2D
 *     wavefront pixel count `N_wf^2` and carries stateful temporal recurrences.
 *     Stepping actuator electronics and kinematics in L1-resident SIMD vectors on
 *     CPU (`cacaosim_dm_step_actuators`), then uploading only `act_pos` (`<= 16 KB`)
 *     asynchronously to GPU for 2D OPD synthesis avoids kernel-launch overhead on
 *     tiny actuator vectors and delivers sub-5-microsecond CUDA OPD rendering.
 *   - Pixel-centric 16-tap gather stencil vs actuator-centric scatter: Scattering
 *     each actuator's 2D Gaussian influence function onto the wavefront grid
 *     requires slow atomic floats on GPU or thread-local 2D scratch buffers on CPU.
 *     Precomputing a fixed `4x4 = 16`-tap nearest-actuator gather stencil per
 *     wavefront pixel turns 2D OPD rendering into a branchless, coalesced 16-term
 *     dot product (`sum_k w[p,k] * pos[idx[p,k]]`) that bakes in 2D affine
 *     DM-to-pupil registration (`shift_x`, `shift_y`, `rot_deg`, `mag_scale`),
 *     partition-of-unity piston flatness, and even 2-axis Tip-Tilt geometry at
 *     zero per-frame computational cost.
 *   - Analytical ZOH temporal integration: Closed-form discrete matrix exponentials
 *     for 1st-order RC lag and 2nd-order damped harmonic oscillators remain
 *     unconditionally stable and exact for piecewise-constant DAC steps even when
 *     `dt_sec >= tau_sec` or near kHz structural resonances, avoiding Euler blow-up.
 */

#ifndef CACAOSIM_DM_H
#define CACAOSIM_DM_H

#include <stdint.h>

#include "cacaosim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fixed gather-stencil tap count per wavefront pixel (4x4 nearest-actuator neighborhood
 * spanning +-2 actuator pitches, where Gaussian coupling <= 0.45^4 < 4.1% at boundary)
 * and circular command history depth supporting up to 254 sub-ticks of fractional transport delay.
 */
#define CACAOSIM_DM_STENCIL      16u
#define CACAOSIM_DM_MAX_DELAY    256u

/**
 * @brief Static geometry, drive electronics, and temporal dynamics configuration for a DM/TTM.
 */
typedef struct
{
    uint32_t nact_x;       /**< Actuator grid X count (overridden to 2 when dm_mode == 1) */
    uint32_t nact_y;       /**< Actuator grid Y count (overridden to 1 when dm_mode == 1) */
    uint32_t nwf;          /**< Square output 2D wavefront OPD grid dimension [pixels] */
    float    dm_diam;      /**< Active actuator array span `(nx - 1)*pitch` as fraction of nwf */
    float    if_coupling;  /**< Gaussian influence coupling at 1 actuator pitch [0.005..0.45] */
    float    refl_scale;   /**< Optical OPD [um] per mechanical stroke [um] (2.0 at normal inc.) */
    float    max_stroke;   /**< Bipolar mechanical stroke rail [um] (`[-max, +max]`, <=0 = off) */
    int32_t  dac_bits;     /**< Uniform DAC resolution [4..24 bits] over 2*max_stroke (0 = off) */
    float    nonlin_quad;  /**< Electrostrictive quadratic term k_nl in `c + k_nl*c*|c|` [1/um] */
    int32_t  dyn_mode;     /**< Temporal model: 0 = instant, 1 = 1st-order RC, 2 = 2nd-order osc */
    float    dt_sec;       /**< Discrete physics integration step interval [s] (> 0) */
    float    tau_sec;      /**< 1st-order exponential 1/e settling time constant [s] (mode == 1) */
    float    f0_hz;        /**< 2nd-order undamped natural resonance frequency [Hz] (mode == 2) */
    float    zeta;         /**< 2nd-order viscous damping ratio (mode == 2; <1, ==1, >1 valid) */
    float    delay_frames; /**< Pure transport latency [frames] in `[0, 254]`, fractional OK */
    float    delay_sec;    /**< Physical pure transport latency [s] (>0 overrides delay_frames) */
    int32_t  no_zoh_mean;  /**< 0 (default) = render analytical ZOH subframe mean; 1 = endpoint */
    int32_t  dm_mode;      /**< Mirror architecture: 0 = 2D zonal DM, 1 = 2-axis Tip-Tilt (2x1) */
    float    shift_x;      /**< Lateral DM-to-pupil X shift [wavefront pixels] */
    float    shift_y;      /**< Lateral DM-to-pupil Y shift [wavefront pixels] */
    float    rot_deg;      /**< Counter-clockwise DM-to-pupil rotation [degrees] */
    float    mag_scale;    /**< Isotropic DM-to-pupil magnification (`<= 1e-4` -> `1.0`) */
    int32_t  backend;      /**< Compute backend (`cacaosim_backend_t`: 0=auto, 1=CPU, 2=CUDA) */
} cacaosim_dm_cfg_t;

/**
 * @brief Pre-allocated runtime state, kinematics buffers, and 16-tap gather stencil for a DM/TTM.
 *
 * All host arrays are 64-byte cache-line aligned via `cacaosim_aligned_calloc()`
 * during `cacaosim_dm_init()` so per-frame execution (`cacaosim_dm_step()`)
 * performs zero heap allocations.
 */
typedef struct
{
    cacaosim_dm_cfg_t cfg;        /**< Sanitized copy of initial DM hardware configuration */
    uint32_t          nact_total; /**< Total actuator count `nact_x * nact_y` (`2` in TTM mode) */
    uint64_t          npix_wf;    /**< Total 2D wavefront pixel count `nwf * nwf` */
    int32_t           use_cuda;   /**< Non-zero when CUDA device buffers & stream are active */
    int32_t           opd_valid;  /**< 1 once d_opd_out matches prev_pos_render (skip static) */

    /* Ring buffer [CACAOSIM_DM_MAX_DELAY * nact_total] + actuator kinematics */
    float            *cmd_ring;        /**< Circular history of conditioned commands [um] */
    uint32_t          ring_head;       /**< Next write slot index in `[0, CACAOSIM_DM_MAX_DELAY)` */
    uint32_t          ring_filled;     /**< Number of populated history slots (up to MAX_DELAY) */
    float            *act_cmd_delayed; /**< Fractionally delayed command driving ZOH plant [um] */
    float            *act_pos;         /**< End-of-step mechanical actuator displacement [um] */
    float            *act_pos_render;  /**< Subframe-averaged (ZOH mean) stroke rendered [um] */
    float            *prev_pos_render; /**< Last actuator stroke vector rendered to d_opd_out */
    float            *act_vel;         /**< Instantaneous actuator velocity [um/s] (dyn_mode==2) */
    uint8_t          *dead_mask;       /**< Per-actuator stuck-zero fault mask (`!= 0` -> 0 um) */

    /* 16-tap pixel-centric gather stencil [npix_wf * 16] */
    int32_t          *stencil_idx;     /**< Actuator indices `[0, nact_total)` per pixel tap */
    float            *stencil_w;       /**< Pre-scaled OPD influence weights [OPD_um / stroke_um] */

    /* Optional CUDA device pointers */
    float            *d_act_pos;       /**< Device copy of `act_pos` `[nact_total]` [um] */
    int32_t          *d_stencil_idx;   /**< Device copy of `stencil_idx` `[npix_wf * 16]` */
    float            *d_stencil_w;     /**< Device copy of `stencil_w` `[npix_wf * 16]` */
    float            *d_opd_out;       /**< Device synthesized 2D OPD surface `[npix_wf]` [um] */
    void             *cuda_stream;     /**< Opaque `cudaStream_t` used for async H2D/kernel/D2H */
    int               owns_stream;     /**< `1` if `cuda_stream` was created privately, `0` ext */
} cacaosim_dm_ctx_t;

/**
 * cacaosim_dm_init() - Precompute gather stencil and allocate CPU/CUDA DM state
 * @ctx:         Destination DM context to initialize (must be non-NULL)
 * @cfg:         Pointer to DM geometry, electronics, and dynamics configuration
 * @ext_stream:  Optional external `cudaStream_t` (`void *`) to share with a fused pipeline
 *               such as `cacaosim.htsim`; pass NULL to create a private non-blocking stream
 *
 * Role:
 *   Cold-path constructor called once during `cacaosim.dmsim` or `cacaosim.htsim`
 *   startup. Allocates 64-byte aligned host ring/kinematics/stencil buffers,
 *   precomputes the 16-tap 2D affine gather stencil, and optionally allocates and
 *   uploads immutable stencil tables to CUDA device memory.
 *
 * Rationale:
 *   Front-loading all `expf()`, `cosf()`, `sinf()`, inverse affine coordinate
 *   transforms, and partition-of-unity normalizations into `cacaosim_dm_init()`
 *   guarantees zero dynamic allocation and purely linear multiply-accumulate math
 *   inside the real-time kHz control loop.
 *
 * Return: `0` on success, `-1` on invalid dimensions or host allocation failure
 *         (falls back gracefully to CPU if CUDA allocation fails).
 */
int cacaosim_dm_init(
    cacaosim_dm_ctx_t       *ctx,
    const cacaosim_dm_cfg_t *cfg,
    void                    *ext_stream);

/**
 * cacaosim_dm_reset() - Zero actuator history, positions, and velocities
 * @ctx: Initialized DM context (gracefully ignored if NULL)
 *
 * Role:
 *   Resets all dynamic temporal state (`cmd_ring`, `act_cmd_delayed`, `act_pos`,
 *   `act_vel`, `ring_head`, `ring_filled`) to zero at rest without deallocating
 *   buffers or rebuilding the 16-tap spatial stencil.
 *
 * Rationale:
 *   Allows closed-loop AO benchmarks and `cacaosim.htsim` resets to clear
 *   transient actuator ringing or latency pipelines instantaneously while
 *   preserving static calibration state (`stencil_idx`, `stencil_w`, `dead_mask`).
 */
void cacaosim_dm_reset(cacaosim_dm_ctx_t *ctx);

/**
 * cacaosim_dm_free() - Release all host and device buffers owned by DM context
 * @ctx: DM context to destroy (safe to call on partially initialized, zeroed, or NULL `ctx`)
 *
 * Role:
 *   Cold-path destructor releasing CUDA device buffers (and private CUDA stream if
 *   `owns_stream != 0`), freeing 64-byte aligned host allocations, and zeroing `ctx`.
 *
 * Rationale:
 *   Centralizes deterministic cleanup across both normal process shutdown and
 *   mid-initialization rollback paths.
 */
void cacaosim_dm_free(cacaosim_dm_ctx_t *ctx);

/**
 * cacaosim_dm_step_actuators() - Advance DAC, stroke clip, delay ring, and ZOH dynamics
 * @ctx:          Active DM context
 * @cmd_in:       Host actuator command vector `[nact_total]` in mechanical stroke `[um]`
 *                (if NULL, commands zero stroke on all actuators)
 * @delay_frames: Live pure transport latency [frames] in `[0, CACAOSIM_DM_MAX_DELAY - 2]`
 * @tau_sec:      Live 1st-order RC time constant [s] used when `cfg.dyn_mode == 1`
 * @max_stroke:   Live bipolar mechanical stroke limit [um] (`<= 0` disables rail & DAC clip)
 *
 * Role:
 *   Hot-path temporal physics stepper executed once per DM physics tick prior to
 *   2D spatial OPD synthesis. Updates `ctx->act_pos` and `ctx->act_vel` in place.
 *
 * Rationale:
 *   Accepts `delay_frames`, `tau_sec`, and `max_stroke` as per-step scalar
 *   arguments rather than reading frozen `ctx->cfg` fields so operators can sweep
 *   loop latency, bandwidth, and saturation limits live (`FPFLAG_WRITERUN`)
 *   without rebuilding the DM context.
 */
void cacaosim_dm_step_actuators(
    cacaosim_dm_ctx_t       *ctx,
    const float *__restrict  cmd_in,
    float                    delay_frames,
    float                    tau_sec,
    float                    max_stroke);

/**
 * cacaosim_dm_render_opd_cpu() - Synthesize 2D DM OPD surface on CPU via 16-tap gather
 * @ctx:      Active DM context with updated `ctx->act_pos`
 * @flat_opd: Optional host static DM flat/figure error map `[npix_wf]` in `[um]` (NULL -> `0`)
 * @opd_out:  Destination host 2D reflective OPD buffer `[npix_wf]` in microns `[um]`
 *
 * Role:
 *   Hot-path CPU spatial renderer evaluating the precomputed 16-tap gather
 *   stencil across all `nwf * nwf` wavefront pixels, adding optional `flat_opd`.
 *
 * Rationale:
 *   Each output pixel independently accumulates 16 contiguous `(idx, weight)` pairs
 *   against L1-cached `act_pos`, enabling lock-free static OpenMP threading on
 *   grids `>= 128x128` with zero write contention.
 */
void cacaosim_dm_render_opd_cpu(
    const cacaosim_dm_ctx_t *ctx,
    const float *__restrict  flat_opd,
    float *__restrict        opd_out);

/**
 * cacaosim_dm_step() - Full DM step (actuators + 2D OPD surface on active backend)
 * @ctx:          Active DM context
 * @cmd_in:       Host actuator command array `[nact_total]` in mechanical stroke `[um]`
 * @flat_opd:     Optional host static DM flat OPD `[npix_wf]` in `[um]` (NULL -> `0`)
 * @delay_frames: Live transport delay [frames]
 * @tau_sec:      Live 1st-order time constant [s]
 * @max_stroke:   Live mechanical stroke limit [um]
 * @host_opd_out: Destination host OPD buffer `[npix_wf]` in `[um]` (may be NULL on CUDA
 *                when downstream GPU stages read `ctx->d_opd_out` directly)
 *
 * Role:
 *   Unified per-frame DM entry point called by `cacaosim.dmsim` and
 *   `cacaosim.htsim` to advance actuator physics and synthesize the 2D OPD map.
 *
 * Rationale:
 *   Dispatches to `cacaosim_dm_cuda_render()` when `ctx->use_cuda` is active and
 *   no host-only `flat_opd` overlay is requested, or transparently falls back to
 *   `cacaosim_dm_render_opd_cpu()` otherwise.
 */
void cacaosim_dm_step(
    cacaosim_dm_ctx_t       *ctx,
    const float *__restrict  cmd_in,
    const float *__restrict  flat_opd,
    float                    delay_frames,
    float                    tau_sec,
    float                    max_stroke,
    float *__restrict        host_opd_out);

/**
 * cacaosim_dm_cuda_alloc() - Allocate CUDA device buffers and upload static 16-tap stencil
 * @ctx:        DM context with populated host `stencil_idx` and `stencil_w` arrays
 * @ext_stream: Optional borrowed `cudaStream_t` (`void *`); NULL creates a private stream
 *
 * Role:
 *   Internal CUDA backend initializer implemented in `cacaosim_dm_cuda.cu` and
 *   invoked by `cacaosim_dm_init()` when GPU execution is requested and available.
 *
 * Rationale:
 *   Uploads the large `[npix_wf * 16]` index and weight tables to VRAM once at
 *   init time so per-frame GPU rendering only transfers `nact_total` floats.
 *
 * Return: `0` on success, `-1` on CUDA stream or device memory allocation error.
 */
int  cacaosim_dm_cuda_alloc(cacaosim_dm_ctx_t *ctx, void *ext_stream);

/**
 * cacaosim_dm_cuda_free() - Free CUDA device buffers and owned stream in a DM context
 * @ctx: DM context whose GPU resources should be released
 *
 * Role:
 *   Internal CUDA teardown helper called by `cacaosim_dm_free()` (and on partial
 *   allocation failure inside `cacaosim_dm_cuda_alloc()`).
 *
 * Rationale:
 *   Destroys `ctx->cuda_stream` only when `ctx->owns_stream != 0` so shared
 *   streams owned by `cacaosim.htsim` remain intact.
 */
void cacaosim_dm_cuda_free(cacaosim_dm_ctx_t *ctx);

/**
 * cacaosim_dm_cuda_render() - Upload `act_pos` and launch CUDA 16-tap gather OPD kernel
 * @ctx:          Active CUDA-enabled DM context
 * @host_opd_out: Optional host destination buffer `[npix_wf]` [um]; if NULL, leaves output
 *                resident in `ctx->d_opd_out` asynchronously without stream synchronization
 *
 * Role:
 *   Internal hot-path GPU renderer called by `cacaosim_dm_step()`.
 *
 * Rationale:
 *   Making the D2H copy and `cudaStreamSynchronize()` conditional on
 *   `host_opd_out != NULL` allows standalone SHM units (`cacaosim.dmsim`) to
 *   materialize host SHM frames while allowing fused GPU pipelines
 *   (`cacaosim.htsim`) to chain DM -> WFS/coronagraph kernels asynchronously on
 *   VRAM with zero PCIe round-trip.
 */
void cacaosim_dm_cuda_render(cacaosim_dm_ctx_t *ctx, float *host_opd_out);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_DM_H */
