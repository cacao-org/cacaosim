// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm_engine.c
 * @brief   DM actuator electronics, exact discrete ZOH kinematics, and 16-tap gather OPD engine
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
            float    vx   = (float) ix - ctr - shx;
            float    mx   = ctr + cr * vx - sr * vy;
            float    my   = ctr + sr * vx + cr * vy;
            float    fx   = (mx - x0) / px;
            float    fy   = (my - y0) / py;
            int      ax0  = (int) floorf(fx) - 1;
            int      ay0  = (int) floorf(fy) - 1;
            uint64_t base = ((uint64_t) jy * nwf + ix) * CACAOSIM_DM_STENCIL;

            /* Unclipped 4x4 lattice partition-of-unity denominator for flat piston response */
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
                        ctx->stencil_idx[base + tap] = 0;
                        ctx->stencil_w[base + tap]   = 0.0f;
                    }
                    tap++;
                }
            }
        }
    }
}

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

    ctx->cmd_ring =
        (float *) cacaosim_aligned_calloc(CACAOSIM_DM_MAX_DELAY * nact, sizeof(float));
    ctx->act_cmd_delayed = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->act_pos         = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->act_vel         = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
    ctx->dead_mask       = (uint8_t *) cacaosim_aligned_calloc(nact, sizeof(uint8_t));
    ctx->stencil_idx     = (int32_t *) cacaosim_aligned_calloc(ntap, sizeof(int32_t));
    ctx->stencil_w       = (float *) cacaosim_aligned_calloc(ntap, sizeof(float));

    if (!ctx->cmd_ring || !ctx->act_cmd_delayed || !ctx->act_pos || !ctx->act_vel ||
        !ctx->dead_mask || !ctx->stencil_idx || !ctx->stencil_w)
    {
        cacaosim_dm_free(ctx);
        return -1;
    }

    dm_build_gather_stencil(ctx);

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
    __builtin_memset(ctx->act_vel, 0, nact * sizeof(float));
    ctx->ring_head   = 0;
    ctx->ring_filled = 0;
}

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
    free(ctx->act_vel);
    free(ctx->dead_mask);
    free(ctx->stencil_idx);
    free(ctx->stencil_w);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

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
            float code = floorf((c + max_stroke) * inv_dac_q + 0.5f);
            c          = code * dac_q - max_stroke;
        }
        if (k_nl != 0.0f)
        {
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

    /* 3. Exact discrete Zero-Order Hold (ZOH) actuator kinematics */
    float *__restrict pos = MILK_ASSUME_ALIGNED(ctx->act_pos);
    float *__restrict vel = MILK_ASSUME_ALIGNED(ctx->act_vel);
    const int32_t mode    = ctx->cfg.dyn_mode;
    const float   dt      = ctx->cfg.dt_sec;

    if (mode == 0)
    {
        __builtin_memcpy(pos, del, nact * sizeof(float));
    }
    else if (mode == 1)
    {
        float alpha = (tau_sec > 1e-7f) ? expf(-dt / tau_sec) : 0.0f;
#pragma omp simd
        for (uint32_t a = 0; a < nact; a++)
        {
            pos[a] = del[a] + alpha * (pos[a] - del[a]);
        }
    }
    else
    {
        const float f0   = (ctx->cfg.f0_hz > 1.0f) ? ctx->cfg.f0_hz : 2500.0f;
        const float zeta = (ctx->cfg.zeta > 0.001f) ? ctx->cfg.zeta : 0.7f;
        const float w0   = CACAOSIM_TWO_PI_F * f0;

        float a_ee = 0.0f;
        float a_ev = 0.0f;
        float a_ve = 0.0f;
        float a_vv = 0.0f;

        if (zeta < 0.9999f)
        {
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
            float E = expf(-w0 * dt);
            a_ee    = E * (1.0f + w0 * dt);
            a_ev    = E * dt;
            a_ve    = -E * w0 * w0 * dt;
            a_vv    = E * (1.0f - w0 * dt);
        }
        else
        {
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

#pragma omp simd
        for (uint32_t a = 0; a < nact; a++)
        {
            float u  = del[a];
            float e0 = pos[a] - u;
            float v0 = vel[a];
            pos[a]   = u + a_ee * e0 + a_ev * v0;
            vel[a]   = a_ve * e0 + a_vv * v0;
        }
    }
}

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
    const float *__restrict pos  = MILK_ASSUME_ALIGNED(ctx->act_pos);
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
