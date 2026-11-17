// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm_engine.c
 * @brief   DM actuator electronics, exact discrete ZOH kinematics, and 16-tap gather OPD engine
 *
 * Architectural Role:
 *   Implements the hardware-physics state machine and CPU 2D OPD surface
 *   synthesizer declared in `cacaosim_dm.h` for both 2D zonal Deformable Mirrors
 *   (`dm_mode == 0`) and 2-axis Tip-Tilt Mirrors (`dm_mode == 1`).
 *
 * Design Rationale & Motivation:
 *   - Unified 16-tap gather representation: Encoding both 2D Gaussian actuator
 *     influence functions (with inverse affine DM-to-pupil registration) and
 *     2-axis Tip-Tilt linear ramps into the same `[npix_wf * 16]` `(idx, weight)`
 *     table lets CPU and CUDA hot loops execute a single branchless 16-tap dot
 *     product per pixel regardless of mirror type or misregistration angle.
 *   - Unclipped 4x4 lattice partition-of-unity normalization: Discrete Gaussian
 *     actuator lattices exhibit periodic "print-through" ripple under uniform
 *     piston commands unless normalized by the local lattice sum. Evaluating the
 *     normalization denominator over the unclipped 4x4 infinite-lattice stencil
 *     eliminates interior print-through completely while preserving natural
 *     free-boundary roll-off outside the outermost actuator ring.
 *   - Exact analytical Zero-Order Hold (ZOH) kinematics: Using closed-form 1st-
 *     and 2nd-order state-transition matrices (handling underdamped, critically
 *     damped, and overdamped regimes analytically) guarantees unconditional
 *     numerical stability and exact phase/amplitude response at arbitrary ratios
 *     of `dt_sec` to `tau_sec` or resonance period `1 / f0_hz`.
 */

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milkDebugTools.h"
#include "milk_compiler.h"
#include "common/cacaosim_cuda.h"
#include "dm/cacaosim_dm.h"

/**
 * dm_build_gather_stencil() - Precompute 4x4 nearest-actuator or 2-axis TTM weights per WF pixel
 * @ctx: DM context with allocated `stencil_idx` and `stencil_w` arrays (`[npix_wf * 16]`)
 *
 * Role:
 *   Cold-path geometry builder called once by `cacaosim_dm_init()` before any
 *   optional H2D upload to CUDA. Maps every 2D wavefront pixel `(ix, jy)` through
 *   the inverse 2D affine DM-to-pupil registration transform and populates its
 *   16 gather indices and `refl_scale`-weighted influence coefficients.
 *
 * Rationale:
 *   - For `dm_mode == 1` (2-axis Tip-Tilt mirror), maps transformed pupil
 *     coordinates `(ux, uy)` normalized to `[-1, +1]` at radius `span / 2` into
 *     taps `0` and `1` (referencing actuators `0` and `1`) and zeros taps `2..15`,
 *     so `pos[0]` and `pos[1]` directly command X and Y edge stroke in `[um]`.
 *   - For `dm_mode == 0` (2D zonal DM), uses a Gaussian influence profile
 *     `exp(-alpha * r^2)` with `alpha = -ln(if_coupling)` so inter-actuator
 *     cross-coupling at `r = 1` pitch equals `if_coupling`. Normalizing by the
 *     unclipped 4x4 lattice sum `lat_sum` enforces exact partition of unity
 *     (`sum_k w_k = refl_scale`) across the interior so uniform actuator piston
 *     yields zero spatial ripple, while zeroing out-of-bounds taps `(ax, ay)`
 *     outside `[0, nx) x [0, ny)` preserves smooth free-edge mechanical roll-off.
 */
static void MILK_COLD dm_build_gather_stencil(cacaosim_dm_ctx_t *ctx)
{
    const uint32_t nx    = ctx->cfg.nact_x;
    const uint32_t ny    = ctx->cfg.nact_y;
    const uint32_t nwf   = ctx->cfg.nwf;
    const float    diam  = (ctx->cfg.dm_diam > 0.05f && ctx->cfg.dm_diam <= 1.2f)
                               ? ctx->cfg.dm_diam
                               : 0.9375f;
    const float    span  = diam * (float) nwf;
    const float    ctr   = 0.5f * (float) (nwf - 1u);
    /* Precompute inverse 2D affine transform: pupil pixel (ix, jy) -> DM frame */
    const float    sc    = (ctx->cfg.mag_scale > 1e-4f) ? ctx->cfg.mag_scale : 1.0f;
    const float    invs  = 1.0f / sc;
    const float    rad   = -ctx->cfg.rot_deg * (CACAOSIM_PI_F / 180.0f);
    const float    cr    = cosf(rad) * invs;
    const float    sr    = sinf(rad) * invs;
    const float    shx   = ctx->cfg.shift_x;
    const float    shy   = ctx->cfg.shift_y;

    if (ctx->cfg.dm_mode == 1)
    {
        /* 2-Axis Tip-Tilt Mirror: pos[0] = X edge stroke [um], pos[1] = Y edge stroke [um] */
        const float inv_r = (span > 1e-4f) ? (2.0f / span) : 1.0f;
#pragma omp parallel for schedule(static)
        for (uint32_t jy = 0; jy < nwf; jy++)
        {
            float vy = (float) jy - ctr - shy;
            for (uint32_t ix = 0; ix < nwf; ix++)
            {
                float    vx   = (float) ix - ctr - shx;
                float    ux   = (cr * vx - sr * vy) * inv_r;
                float    uy   = (sr * vx + cr * vy) * inv_r;
                uint64_t base = ((uint64_t) jy * nwf + ix) * CACAOSIM_DM_STENCIL;

                /* Pack X and Y linear tilt planes into taps 0 & 1; pad taps 2..15 with zeros */
                ctx->stencil_idx[base + 0] = 0;
                ctx->stencil_w[base + 0]   = ctx->cfg.refl_scale * ux;
                ctx->stencil_idx[base + 1] = (ctx->nact_total >= 2u) ? 1 : 0;
                ctx->stencil_w[base + 1]   = (ctx->nact_total >= 2u) ? (ctx->cfg.refl_scale * uy)
                                                                     : 0.0f;
                for (uint32_t t = 2u; t < CACAOSIM_DM_STENCIL; t++)
                {
                    ctx->stencil_idx[base + t] = 0;
                    ctx->stencil_w[base + t]   = 0.0f;
                }
            }
        }
        return;
    }

    /* Zonal 2D actuator grid pitch (px, py) and origin (x0, y0) of actuator (0, 0) in WF pixels */
    const float px = (nx > 1u) ? (span / (float) (nx - 1u)) : span;
    const float py = (ny > 1u) ? (span / (float) (ny - 1u)) : span;
    const float x0 = ctr - 0.5f * (float) (nx - 1u) * px;
    const float y0 = ctr - 0.5f * (float) (ny - 1u) * py;

    float coupl = ctx->cfg.if_coupling;
    if (coupl < 0.005f)
    {
        coupl = 0.005f;
    }
    else if (coupl > 0.45f)
    {
        coupl = 0.45f;
    }
    /* Gaussian: exp(-d^2 * alpha) = coupl at d = 1 pitch => alpha = -ln(coupl) */
    const float alpha = -logf(coupl);

#pragma omp parallel for schedule(static)
    for (uint32_t jy = 0; jy < nwf; jy++)
    {
        float vy = (float) jy - ctr - shy;

        for (uint32_t ix = 0; ix < nwf; ix++)
        {
            /* Map pupil pixel (ix, jy) into fractional actuator-grid coordinates (fx, fy) */
            float    vx   = (float) ix - ctr - shx;
            float    mx   = ctr + cr * vx - sr * vy;
            float    my   = ctr + sr * vx + cr * vy;
            float    fx   = (mx - x0) / px;
            float    fy   = (my - y0) / py;
            int      ax0  = (int) floorf(fx) - 1;
            int      ay0  = (int) floorf(fy) - 1;
            uint64_t base = ((uint64_t) jy * nwf + ix) * CACAOSIM_DM_STENCIL;

            /*
             * Unclipped 4x4 lattice partition-of-unity denominator for flat piston response:
             * summing over the unbounded 4x4 grid prevents print-through ripple in the interior,
             * while zeroing out-of-bounds taps below yields physical free-edge roll-off.
             */
            float lat_sum = 0.0f;
            for (int dy = 0; dy < 4; dy++)
            {
                float ry = fy - (float) (ay0 + dy);
                for (int dx = 0; dx < 4; dx++)
                {
                    float rx = fx - (float) (ax0 + dx);
                    lat_sum += expf(-alpha * (rx * rx + ry * ry));
                }
            }
            float w_scale = (lat_sum > 1e-8f) ? (ctx->cfg.refl_scale / lat_sum)
                                              : ctx->cfg.refl_scale;

            int tap = 0;
            for (int dy = 0; dy < 4; dy++)
            {
                int ay = ay0 + dy;
                for (int dx = 0; dx < 4; dx++)
                {
                    int ax = ax0 + dx;
                    if (ax >= 0 && ax < (int) nx && ay >= 0 && ay < (int) ny)
                    {
                        float rx  = fx - (float) ax;
                        float ry  = fy - (float) ay;
                        float r2  = rx * rx + ry * ry;
                        float w   = w_scale * expf(-alpha * r2);
                        ctx->stencil_idx[base + tap] = ay * (int) nx + ax;
                        ctx->stencil_w[base + tap]   = w;
                    }
                    else
                    {
                        /* Out-of-bounds tap safely indexes actuator 0 with zero weight */
                        ctx->stencil_idx[base + tap] = 0;
                        ctx->stencil_w[base + tap]   = 0.0f;
                    }
                    tap++;
                }
            }
        }
    }
}

/**
 * cacaosim_dm_init() - Precompute gather stencil and allocate CPU/CUDA DM state
 * @ctx:         Destination DM context to initialize
 * @cfg:         DM geometry, electronics, and kinematics configuration
 * @ext_stream:  Optional external `cudaStream_t` (`void *`) to borrow (NULL = private stream)
 *
 * Role:
 *   Cold-path constructor allocating all 64-byte aligned host buffers, building
 *   the 16-tap gather stencil, and initializing CUDA device buffers when
 *   `cfg->backend` permits GPU execution and a CUDA device is present.
 *
 * Rationale:
 *   Enforces `2x1` actuator geometry in Tip-Tilt mode (`dm_mode == 1`), defaults
 *   `refl_scale` to `2.0` (normal-incidence optical path doubling) and `dt_sec`
 *   to `0.0005` s (`2 kHz`), and cleanly rolls back partial allocations on OOM.
 *
 * Return: `0` on success, `-1` on invalid arguments or host memory allocation failure.
 */
int MILK_COLD cacaosim_dm_init(
    cacaosim_dm_ctx_t       *ctx,
    const cacaosim_dm_cfg_t *cfg,
    void                    *ext_stream)
{
    if (ctx == NULL || cfg == NULL || cfg->nact_x == 0 || cfg->nact_y == 0 || cfg->nwf == 0)
    {
        return -1;
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg = *cfg;
    if (ctx->cfg.dm_mode == 1)
    {
        ctx->cfg.nact_x = 2u;
        ctx->cfg.nact_y = 1u;
    }
    ctx->nact_total = ctx->cfg.nact_x * ctx->cfg.nact_y;
    ctx->npix_wf    = (uint64_t) cfg->nwf * cfg->nwf;

    if (ctx->cfg.refl_scale == 0.0f)
    {
        ctx->cfg.refl_scale = 2.0f;
    }
    if (ctx->cfg.dt_sec <= 0.0f)
    {
        ctx->cfg.dt_sec = 0.0005f;
    }

    size_t nact = ctx->nact_total;
    size_t ntap = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL;

    /* Allocate 64-byte cache-line aligned host buffers for AVX2/AVX-512 SIMD hot loops */
    ctx->cmd_ring =
        (float *) cacaosim_aligned_calloc(CACAOSIM_DM_MAX_DELAY * nact, sizeof(float));
    ctx->act_cmd_delayed = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->act_pos         = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->act_pos_render  = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->prev_pos_render = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->act_vel         = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->dead_mask       = (uint8_t *) cacaosim_aligned_calloc(nact, sizeof(uint8_t));
    ctx->stencil_idx     = (int32_t *) cacaosim_aligned_calloc(ntap, sizeof(int32_t));
    ctx->stencil_w       = (float *) cacaosim_aligned_calloc(ntap, sizeof(float));

    if (!ctx->cmd_ring || !ctx->act_cmd_delayed || !ctx->act_pos || !ctx->act_pos_render ||
        !ctx->prev_pos_render || !ctx->act_vel || !ctx->dead_mask || !ctx->stencil_idx ||
        !ctx->stencil_w)
    {
        cacaosim_dm_free(ctx);
        return -1;
    }

    dm_build_gather_stencil(ctx);

    /* Probe and initialize CUDA backend when requested (gracefully falls back to CPU on failure) */
    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_dm_cuda_alloc(ctx, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

/**
 * cacaosim_dm_reset() - Zero actuator history, positions, and velocities
 * @ctx: Initialized DM context (ignored if NULL)
 *
 * Role:
 *   Clears the circular command delay ring buffer and actuator displacement/velocity
 *   vectors back to zero rest state while keeping precomputed stencils and
 *   dead-actuator masks intact.
 *
 * Rationale:
 *   Provides fast deterministic state re-initialization between closed-loop AO
 *   runs without incurring stencil recomputation or CUDA memory reallocation.
 */
void MILK_COLD cacaosim_dm_reset(cacaosim_dm_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    size_t nact = ctx->nact_total;
    __builtin_memset(ctx->cmd_ring, 0, CACAOSIM_DM_MAX_DELAY * nact * sizeof(float));
    __builtin_memset(ctx->act_cmd_delayed, 0, nact * sizeof(float));
    __builtin_memset(ctx->act_pos, 0, nact * sizeof(float));
    __builtin_memset(ctx->act_pos_render, 0, nact * sizeof(float));
    __builtin_memset(ctx->prev_pos_render, 0, nact * sizeof(float));
    __builtin_memset(ctx->act_vel, 0, nact * sizeof(float));
    ctx->ring_head   = 0;
    ctx->ring_filled = 0;
    ctx->opd_valid   = 0;
}

/**
 * cacaosim_dm_free() - Release all host and device buffers owned by DM context
 * @ctx: DM context to destroy (ignored if NULL)
 *
 * Role:
 *   Tears down optional CUDA device allocations, frees all aligned host arrays,
 *   and zeroes `*ctx` to prevent dangling pointer reuse.
 *
 * Rationale:
 *   Safe to invoke both at normal teardown and from `cacaosim_dm_init()` error
 *   recovery when only a subset of buffers succeeded.
 */
void MILK_COLD cacaosim_dm_free(cacaosim_dm_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda)
    {
        cacaosim_dm_cuda_free(ctx);
    }
    free(ctx->cmd_ring);
    free(ctx->act_cmd_delayed);
    free(ctx->act_pos);
    free(ctx->act_pos_render);
    free(ctx->prev_pos_render);
    free(ctx->act_vel);
    free(ctx->dead_mask);
    free(ctx->stencil_idx);
    free(ctx->stencil_w);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

/**
 * cacaosim_dm_step_actuators() - Advance DAC, stroke clip, delay ring, and ZOH dynamics
 * @ctx:          Active DM context
 * @cmd_in:       Host actuator command array `[nact_total]` in mechanical stroke `[um]`
 * @delay_frames: Live pure transport latency [frames] (`[0, CACAOSIM_DM_MAX_DELAY - 2]`)
 * @tau_sec:      Live 1st-order RC time constant [s] (`dyn_mode == 1`)
 * @max_stroke:   Live mechanical stroke rail [um] (`<= 0` disables clipping & DAC quantization)
 *
 * Role:
 *   Hot-path actuator state-space integrator advancing drive electronics, pure
 *   transport latency, and mechanical kinematics by one discrete step `cfg.dt_sec`.
 *
 * Rationale:
 *   - Fractional frame delay via linear interpolation between ring-buffer slots
 *     `t - floor(d)` and `t - floor(d) - 1` reproduces exact phase lag in AO
 *     servo Bode responses without forcing fine sub-frame time steps.
 *   - Analytical ZOH matrix exponentials (`[e(dt); v(dt)] = A(dt) [e(0); v(0)]`
 *     where `e = pos - u`) compute scalar transition coefficients once outside the
 *     `#pragma omp simd` loop, updating every actuator in 4 vector FMAs with zero
 *     numerical damping or Courant/Euler step-size restrictions.
 */
void MILK_HOT cacaosim_dm_step_actuators(
    cacaosim_dm_ctx_t       *ctx,
    const float *__restrict  cmd_in,
    float                    delay_frames,
    float                    tau_sec,
    float                    max_stroke)
{
    const uint32_t nact     = ctx->nact_total;
    const uint32_t slot     = ctx->ring_head;
    float *__restrict ring_slot = &ctx->cmd_ring[(size_t) slot * nact];
    const uint8_t *__restrict dead = ctx->dead_mask;

    const int32_t do_clip   = (max_stroke > 0.0f);
    const int32_t dac_bits  = ctx->cfg.dac_bits;
    const int32_t do_dac    = (dac_bits >= 4 && dac_bits <= 24 && max_stroke > 0.0f);
    const float   dac_q     = do_dac ? ((2.0f * max_stroke) / (float) ((1u << dac_bits) - 1u))
                                     : 0.0f;
    const float   inv_dac_q = do_dac ? (1.0f / dac_q) : 0.0f;
    const float   k_nl      = ctx->cfg.nonlin_quad;

    /* 1. Apply stroke saturation, DAC quantization, non-linearity, and dead-actuator mask */
#pragma omp simd
    for (uint32_t a = 0; a < nact; a++)
    {
        float c = (cmd_in != NULL && !dead[a]) ? cmd_in[a] : 0.0f;
        if (do_clip)
        {
            if (c > max_stroke)
            {
                c = max_stroke;
            }
            else if (c < -max_stroke)
            {
                c = -max_stroke;
            }
        }
        if (do_dac)
        {
            /* Uniform mid-round quantization over [-max_stroke, +max_stroke] in 2^B - 1 steps */
            float code = floorf((c + max_stroke) * inv_dac_q + 0.5f);
            c          = code * dac_q - max_stroke;
        }
        if (k_nl != 0.0f)
        {
            /* Symmetric odd-function electrostrictive nonlinearity preserving stroke sign */
            c += k_nl * c * fabsf(c);
        }
        ring_slot[a] = c;
    }

    /* 2. Fractional transport delay interpolation from circular history */
    float d_clamped = delay_frames;
    if (d_clamped < 0.0f)
    {
        d_clamped = 0.0f;
    }
    else if (d_clamped > (float) (CACAOSIM_DM_MAX_DELAY - 2u))
    {
        d_clamped = (float) (CACAOSIM_DM_MAX_DELAY - 2u);
    }

    uint32_t d0   = (uint32_t) floorf(d_clamped);
    float    frac = d_clamped - (float) d0;
    uint32_t idx0 = (slot + CACAOSIM_DM_MAX_DELAY - d0) % CACAOSIM_DM_MAX_DELAY;
    uint32_t idx1 = (slot + CACAOSIM_DM_MAX_DELAY - (d0 + 1u)) % CACAOSIM_DM_MAX_DELAY;

    const float *__restrict r0  = &ctx->cmd_ring[(size_t) idx0 * nact];
    const float *__restrict r1  = &ctx->cmd_ring[(size_t) idx1 * nact];
    float *__restrict       del = ctx->act_cmd_delayed;

    if (frac <= 1e-6f)
    {
        __builtin_memcpy(del, r0, nact * sizeof(float));
    }
    else
    {
        const float w0 = 1.0f - frac;
#pragma omp simd
        for (uint32_t a = 0; a < nact; a++)
        {
            del[a] = w0 * r0[a] + frac * r1[a];
        }
    }

    ctx->ring_head = (slot + 1u) % CACAOSIM_DM_MAX_DELAY;
    if (ctx->ring_filled < CACAOSIM_DM_MAX_DELAY)
    {
        ctx->ring_filled++;
    }

    /* 3. Exact discrete Zero-Order Hold (ZOH) actuator kinematics & subframe-mean stroke */
    float *__restrict pos     = MILK_ASSUME_ALIGNED(ctx->act_pos);
    float *__restrict pos_ren = MILK_ASSUME_ALIGNED(ctx->act_pos_render);
    float *__restrict vel     = MILK_ASSUME_ALIGNED(ctx->act_vel);
    const int32_t     mode    = ctx->cfg.dyn_mode;
    const int32_t     do_mean = !ctx->cfg.no_zoh_mean;
    const float       dt      = ctx->cfg.dt_sec;

    if (mode == 0)
    {
        /* Mode 0: Infinite-bandwidth instantaneous settling */
        __builtin_memcpy(pos, del, nact * sizeof(float));
        __builtin_memcpy(pos_ren, del, nact * sizeof(float));
    }
    else if (mode == 1)
    {
        /*
         * Mode 1: Exact 1st-order RC step response x(t+dt) = u + exp(-dt/tau)*(x0 - u), paired
         * with analytical ZOH subframe mean x_bar = u + (tau/dt)*(1 - exp(-dt/tau))*(x0 - u).
         */
        float alpha  = 0.0f;
        float mean_c = 0.0f;
        if (tau_sec > 1e-7f)
        {
            float r = dt / tau_sec;
            alpha   = expf(-r);
            if (!do_mean)
            {
                mean_c = alpha;
            }
            else if (r < 1e-3f)
            {
                mean_c = 1.0f - 0.5f * r + (1.0f / 6.0f) * r * r;
            }
            else
            {
                mean_c = (1.0f - alpha) / r;
            }
        }
#pragma omp simd
        for (uint32_t a = 0; a < nact; a++)
        {
            float u    = del[a];
            float e0   = pos[a] - u;
            pos_ren[a] = u + mean_c * e0;
            pos[a]     = u + alpha * e0;
        }
    }
    else
    {
        /*
         * Mode 2: Exact 2nd-order damped oscillator e'' + 2*zeta*w0*e' + w0^2*e = 0
         * in tracking-error state coordinates e(t) = pos(t) - u, v(t) = vel(t).
         * Integrating e'' = -w0^2*e - 2*zeta*w0*e' over [0, dt] yields exact subframe mean:
         *   e_bar = -((v(dt) - v0) + 2*zeta*w0*(e(dt) - e0)) / (w0^2 * dt).
         */
        const float f0   = (ctx->cfg.f0_hz > 1.0f) ? ctx->cfg.f0_hz : 2500.0f;
        const float zeta = (ctx->cfg.zeta > 0.001f) ? ctx->cfg.zeta : 0.7f;
        const float w0   = CACAOSIM_TWO_PI_F * f0;

        float a_ee = 0.0f;
        float a_ev = 0.0f;
        float a_ve = 0.0f;
        float a_vv = 0.0f;

        if (zeta < 0.9999f)
        {
            /* Underdamped regime (0 < zeta < 1): complex-conjugate poles -zeta*w0 +- i*wd */
            float wd = w0 * sqrtf(1.0f - zeta * zeta);
            float E  = expf(-zeta * w0 * dt);
            float C  = cosf(wd * dt);
            float S  = sinf(wd * dt);

            a_ee = E * (C + (zeta * w0 / wd) * S);
            a_ev = E * (S / wd);
            a_ve = -E * ((w0 * w0 / wd) * S);
            a_vv = E * (C - (zeta * w0 / wd) * S);
        }
        else if (zeta <= 1.0001f)
        {
            /* Critically damped regime (zeta ~ 1): repeated real pole -w0, avoiding wd -> 0 */
            float E = expf(-w0 * dt);
            a_ee    = E * (1.0f + w0 * dt);
            a_ev    = E * dt;
            a_ve    = -E * w0 * w0 * dt;
            a_vv    = E * (1.0f - w0 * dt);
        }
        else
        {
            /* Overdamped regime (zeta > 1): distinct negative real poles r1v, r2v */
            float gam = w0 * sqrtf(zeta * zeta - 1.0f);
            float r1v = -w0 * zeta + gam;
            float r2v = -w0 * zeta - gam;
            float E1  = expf(r1v * dt);
            float E2  = expf(r2v * dt);
            float inv = 0.5f / gam;

            a_ee = (-r2v * E1 + r1v * E2) * inv;
            a_ev = (E1 - E2) * inv;
            a_ve = r1v * r2v * (E2 - E1) * inv;
            a_vv = (r1v * E1 - r2v * E2) * inv;
        }

        float m_ee = a_ee;
        float m_ev = a_ev;
        if (do_mean)
        {
            float wdt = w0 * dt;
            if (wdt < 0.02f)
            {
                /* 3rd-order Taylor series avoids catastrophic cancellation when w0*dt << 1 */
                m_ee = 1.0f - (1.0f / 6.0f) * wdt * wdt +
                       (1.0f / 12.0f) * zeta * wdt * wdt * wdt;
                m_ev = dt * (0.5f - (1.0f / 3.0f) * zeta * wdt +
                             (1.0f / 24.0f) * (4.0f * zeta * zeta - 1.0f) * wdt * wdt);
            }
            else
            {
                float inv_denom = 1.0f / (w0 * w0 * dt);
                m_ee            = -(a_ve + 2.0f * zeta * w0 * (a_ee - 1.0f)) * inv_denom;
                m_ev            = -((a_vv - 1.0f) + 2.0f * zeta * w0 * a_ev) * inv_denom;
            }
        }

#pragma omp simd
        for (uint32_t a = 0; a < nact; a++)
        {
            float u    = del[a];
            float e0   = pos[a] - u;
            float v0   = vel[a];
            pos_ren[a] = u + m_ee * e0 + m_ev * v0;
            pos[a]     = u + a_ee * e0 + a_ev * v0;
            vel[a]     = a_ve * e0 + a_vv * v0;
        }
    }
}

/**
 * cacaosim_dm_render_opd_cpu() - Synthesize 2D DM OPD surface on CPU via 16-tap gather
 * @ctx:      Active DM context with updated `ctx->act_pos_render`
 * @flat_opd: Optional static DM flat/figure OPD map `[npix_wf]` in `[um]` (NULL -> `0`)
 * @opd_out:  Destination host 2D reflective OPD buffer `[npix_wf]` in microns `[um]`
 *
 * Role:
 *   Hot-path CPU spatial surface synthesizer evaluating the 16-tap gather stencil
 *   across all `nwf * nwf` wavefront pixels.
 *
 * Rationale:
 *   Gates OpenMP multi-threading on `npix >= 16384` (`>= 128x128`) so small grids
 *   avoid thread-fork barrier latency while larger XAO grids scale linearly across
 *   CPU cores with contiguous 64-byte aligned stencil reads.
 */
void MILK_HOT cacaosim_dm_render_opd_cpu(
    const cacaosim_dm_ctx_t *ctx,
    const float *__restrict  flat_opd,
    float *__restrict        opd_out)
{
    if (opd_out == NULL)
    {
        return;
    }

    const uint64_t          npix = ctx->npix_wf;
    const float *__restrict pos  = MILK_ASSUME_ALIGNED(ctx->act_pos_render);
    const int32_t *__restrict sidx = MILK_ASSUME_ALIGNED(ctx->stencil_idx);
    const float *__restrict   sw   = MILK_ASSUME_ALIGNED(ctx->stencil_w);
    const int nth = (npix >= 16384ULL) ? omp_get_max_threads() : 1;

#pragma omp parallel for num_threads(nth) if (nth > 1) schedule(static)
    for (uint64_t p = 0; p < npix; p++)
    {
        const uint64_t          base = p * CACAOSIM_DM_STENCIL;
        const int32_t *__restrict idx = &sidx[base];
        const float *__restrict   w   = &sw[base];
        float                     acc = (flat_opd != NULL) ? flat_opd[p] : 0.0f;

        for (uint32_t k = 0; k < CACAOSIM_DM_STENCIL; k++)
        {
            acc += w[k] * pos[idx[k]];
        }
        opd_out[p] = acc;
    }
}

/**
 * cacaosim_dm_step() - Full DM step (actuators + 2D OPD surface on active backend)
 * @ctx:          Active DM context
 * @cmd_in:       Host actuator command array `[nact_total]` in mechanical stroke `[um]`
 * @flat_opd:     Optional host static DM flat OPD `[npix_wf]` in `[um]` (NULL -> `0`)
 * @delay_frames: Live transport delay [frames]
 * @tau_sec:      Live 1st-order time constant [s]
 * @max_stroke:   Live mechanical stroke limit [um]
 * @host_opd_out: Destination host OPD buffer `[npix_wf]` [um] (NULL = keep in `ctx->d_opd_out`)
 *
 * Role:
 *   Top-level per-frame DM physics and OPD synthesis entry point.
 *
 * Rationale:
 *   Always advances actuator kinematics on CPU (`cacaosim_dm_step_actuators`), then
 *   routes 2D spatial rendering to CUDA (`cacaosim_dm_cuda_render`) when `use_cuda`
 *   is enabled and `flat_opd == NULL`, or to `cacaosim_dm_render_opd_cpu` when CPU
 *   execution or a host-resident `flat_opd` overlay is requested.
 */
void MILK_HOT cacaosim_dm_step(
    cacaosim_dm_ctx_t       *ctx,
    const float *__restrict  cmd_in,
    const float *__restrict  flat_opd,
    float                    delay_frames,
    float                    tau_sec,
    float                    max_stroke,
    float *__restrict        host_opd_out)
{
    cacaosim_dm_step_actuators(ctx, cmd_in, delay_frames, tau_sec, max_stroke);

    if (ctx->use_cuda && flat_opd == NULL)
    {
        cacaosim_dm_cuda_render(ctx, host_opd_out);
    }
    else
    {
        cacaosim_dm_render_opd_cpu(ctx, flat_opd, host_opd_out);
    }
}
