// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm.h
 * @brief   Deformable mirror hardware physics and gather-stencil OPD engine (CPU + CUDA)
 */

#ifndef CACAOSIM_DM_H
#define CACAOSIM_DM_H

#include <stdint.h>

#include "cacaosim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_DM_STENCIL      16u
#define CACAOSIM_DM_MAX_DELAY    64u

/**
 * @brief Configuration for the DM hardware simulator engine.
 */
typedef struct
{
    uint32_t nact_x;       /**< Actuator grid X dimension */
    uint32_t nact_y;       /**< Actuator grid Y dimension */
    uint32_t nwf;          /**< Output 2D OPD grid dimension [pixels] */
    float    dm_diam;      /**< DM active span as fraction of nwf (default: 0.9375) */
    float    if_coupling;  /**< Inter-actuator mechanical coupling at 1 pitch (0.01..0.45) */
    float    refl_scale;   /**< OPD per unit actuator stroke (2.0 for normal reflection) */
    float    max_stroke;   /**< Mechanical stroke saturation limit [um] (<=0 disables) */
    int32_t  dac_bits;     /**< DAC quantization bits over [-max_stroke, +max_stroke] (0=off) */
    float    nonlin_quad;  /**< Quadratic electrostrictive coefficient (0 = linear) */
    int32_t  dyn_mode;     /**< 0 = instantaneous, 1 = 1st-order RC, 2 = 2nd-order oscillator */
    float    dt_sec;       /**< Simulation step interval [s] */
    float    tau_sec;      /**< 1st-order 1/e time constant [s] (dyn_mode == 1) */
    float    f0_hz;        /**< 2nd-order natural resonance frequency [Hz] (dyn_mode == 2) */
    float    zeta;         /**< 2nd-order damping ratio (dyn_mode == 2) */
    float    delay_frames; /**< Pure hardware transport latency [frames, >= 0, fractional OK] */
    int32_t  dm_mode;      /**< 0 = 2D zonal DM, 1 = 2-axis Tip-Tilt mirror (2x1 cmd) */
    float    shift_x;      /**< DM-to-pupil X shift [pixels] (default: 0) */
    float    shift_y;      /**< DM-to-pupil Y shift [pixels] (default: 0) */
    float    rot_deg;      /**< DM-to-pupil CCW rotation [degrees] (default: 0) */
    float    mag_scale;    /**< DM-to-pupil magnification (0 or 1 -> 1.0) */
    int32_t  backend;      /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
} cacaosim_dm_cfg_t;

/**
 * @brief Pre-allocated state and sparse gather stencil for DM simulation.
 */
typedef struct
{
    cacaosim_dm_cfg_t cfg;
    uint32_t          nact_total;
    uint64_t          npix_wf;
    int32_t           use_cuda;

    /* Ring buffer [CACAOSIM_DM_MAX_DELAY * nact_total] + actuator kinematics */
    float            *cmd_ring;
    uint32_t          ring_head;
    uint32_t          ring_filled;
    float            *act_cmd_delayed;
    float            *act_pos;
    float            *act_vel;
    uint8_t          *dead_mask;

    /* 16-tap pixel-centric gather stencil [npix_wf * 16] */
    int32_t          *stencil_idx;
    float            *stencil_w;

    /* Optional CUDA device pointers */
    float            *d_act_pos;
    int32_t          *d_stencil_idx;
    float            *d_stencil_w;
    float            *d_opd_out;
    void             *cuda_stream;
    int               owns_stream;
} cacaosim_dm_ctx_t;

/**
 * cacaosim_dm_init() - Precompute gather stencil and allocate CPU/CUDA DM state
 * @ctx:         Output DM context
 * @cfg:         DM geometry and hardware configuration
 * @ext_stream:  Optional external cudaStream_t to share (NULL creates private stream)
 *
 * Return: 0 on success, -1 on failure.
 */
int cacaosim_dm_init(
    cacaosim_dm_ctx_t       *ctx,
    const cacaosim_dm_cfg_t *cfg,
    void                    *ext_stream);

/**
 * cacaosim_dm_reset() - Zero actuator history, positions, and velocities
 * @ctx: Initialized DM context
 */
void cacaosim_dm_reset(cacaosim_dm_ctx_t *ctx);

/**
 * cacaosim_dm_free() - Release all host and device buffers owned by DM context
 * @ctx: DM context to destroy
 */
void cacaosim_dm_free(cacaosim_dm_ctx_t *ctx);

/**
 * cacaosim_dm_step_actuators() - Advance DAC, stroke clip, delay ring, and ZOH dynamics
 * @ctx:          Active DM context
 * @cmd_in:       Host actuator command array [nact_x * nact_y]
 * @delay_frames: Live transport delay [frames] (overrides cfg.delay_frames)
 * @tau_sec:      Live 1st-order time constant [s]
 * @max_stroke:   Live mechanical stroke limit [um]
 */
void cacaosim_dm_step_actuators(
    cacaosim_dm_ctx_t       *ctx,
    const float *__restrict  cmd_in,
    float                    delay_frames,
    float                    tau_sec,
    float                    max_stroke);

/**
 * cacaosim_dm_render_opd_cpu() - Synthesize 2D DM OPD surface on CPU via 16-tap gather
 * @ctx:      Active DM context
 * @flat_opd: Optional static DM flat/error OPD [npix_wf] (NULL -> 0)
 * @opd_out:  Destination host OPD buffer [npix_wf] in microns
 */
void cacaosim_dm_render_opd_cpu(
    const cacaosim_dm_ctx_t *ctx,
    const float *__restrict  flat_opd,
    float *__restrict        opd_out);

/**
 * cacaosim_dm_step() - Full DM step (actuators + 2D OPD surface on active backend)
 * @ctx:          Active DM context
 * @cmd_in:       Host actuator command array [nact_x * nact_y]
 * @flat_opd:     Optional host static DM flat OPD [npix_wf] (NULL -> 0)
 * @delay_frames: Live transport delay [frames]
 * @tau_sec:      Live 1st-order time constant [s]
 * @max_stroke:   Live mechanical stroke limit [um]
 * @host_opd_out: Destination host OPD buffer [npix_wf] (may be NULL if only d_opd_out needed)
 */
void cacaosim_dm_step(
    cacaosim_dm_ctx_t       *ctx,
    const float *__restrict  cmd_in,
    const float *__restrict  flat_opd,
    float                    delay_frames,
    float                    tau_sec,
    float                    max_stroke,
    float *__restrict        host_opd_out);

/* Internal CUDA entry points implemented in cacaosim_dm_cuda.cu */
int  cacaosim_dm_cuda_alloc(cacaosim_dm_ctx_t *ctx, void *ext_stream);
void cacaosim_dm_cuda_free(cacaosim_dm_ctx_t *ctx);
void cacaosim_dm_cuda_render(cacaosim_dm_ctx_t *ctx, float *host_opd_out);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_DM_H */
