// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dm_cuda.cu
 * @brief   CUDA 16-tap gather-stencil kernel for sub-5-microsecond 2D DM OPD synthesis
 *
 * Architectural Role:
 *   Implements the GPU device memory lifecycle (`cacaosim_dm_cuda_alloc`,
 *   `cacaosim_dm_cuda_free`) and per-frame CUDA OPD synthesis launcher
 *   (`cacaosim_dm_cuda_render` + `kernel_dm_gather_opd`) backing `cacaosim_dm.h`.
 *
 * Design Rationale & Motivation:
 *   - Minimal H2D payload (`act_pos` only): Rather than pushing 2D wavefront maps
 *     across PCIe or running tiny sequential ring-buffer updates on GPU, the host
 *     uploads only the `nact_total`-element float actuator displacement vector
 *     (e.g., `16x16 = 1 KB`, `64x64 = 16 KB`) via `cudaMemcpyAsync` on a
 *     non-blocking stream, completing in `~1..2 us`.
 *   - Coalesced gather + read-only texture/L1 broadcast (`__ldg`): Each GPU thread
 *     computes one output pixel's OPD by unrolling all 16 stencil taps. Adjacent
 *     threads in a 32-thread warp share nearly identical 4x4 actuator neighborhoods,
 *     so `__ldg(&act_pos[a])` hits the read-only data cache with near-100% locality
 *     and zero atomic write contention.
 *   - Asynchronous GPU pipelining: When `host_opd_out == NULL` (as in `cacaosim.htsim`),
 *     `cacaosim_dm_cuda_render()` enqueues H2D + kernel onto `ctx->cuda_stream`
 *     and returns immediately without synchronizing, letting downstream GPU WFS
 *     and coronagraph stages consume `ctx->d_opd_out` directly in VRAM.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dm/cacaosim_dm.h"

/**
 * kernel_dm_gather_opd() - Evaluate 16-tap gather stencil per wavefront pixel on GPU
 * @act_pos:     Device actuator mechanical position vector `[nact_total]` in `[um]`
 * @stencil_idx: Device gather index table `[npix * 16]` mapping taps to `[0, nact_total)`
 * @stencil_w:   Device gather weight table `[npix * 16]` in `[OPD_um / stroke_um]`
 * @npix:        Total 2D wavefront pixel count `nwf * nwf`
 * @opd_out:     Device output 2D reflective OPD buffer `[npix]` in microns `[um]`
 *
 * Role:
 *   Hot-path CUDA `__global__` kernel launched once per DM step by
 *   `cacaosim_dm_cuda_render()` over a 1D grid of 256-thread blocks.
 *
 * Rationale:
 *   Full `#pragma unroll` of `CACAOSIM_DM_STENCIL` (`16`) keeps the accumulator
 *   `acc` in a register and overlaps independent stencil memory loads, while
 *   `__ldg()` routes actuator position gathers through the read-only cache.
 */
__global__ void kernel_dm_gather_opd(
    const float *__restrict__   act_pos,
    const int32_t *__restrict__ stencil_idx,
    const float *__restrict__   stencil_w,
    uint64_t                    npix,
    float *__restrict__         opd_out)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix)
    {
        return;
    }

    uint64_t base = p * CACAOSIM_DM_STENCIL;
    float    acc  = 0.0f;

#pragma unroll
    for (uint32_t k = 0; k < CACAOSIM_DM_STENCIL; k++)
    {
        int32_t a = stencil_idx[base + k];
        float   w = stencil_w[base + k];
        acc      += w * __ldg(&act_pos[a]);
    }
    opd_out[p] = acc;
}

/**
 * dev_plate_lap() - Evaluate discrete 5-point Laplacian on shared-memory circular DM lattice
 */
static __device__ __forceinline__ float dev_plate_lap(
    const float   *a,
    const uint8_t *mask,
    uint32_t       nx,
    uint32_t       ny,
    uint32_t       ax,
    uint32_t       ay,
    int32_t        rim_bc)
{
    const uint32_t idx = ay * nx + ax;
    const float    c   = a[idx];
    const float    bc  = (rim_bc == 1) ? c : 0.0f;
    const float    l   = (ax > 0u && mask[idx - 1u]) ? a[idx - 1u] : bc;
    const float    r   = (ax + 1u < nx && mask[idx + 1u]) ? a[idx + 1u] : bc;
    const float    u   = (ay > 0u && mask[idx - nx]) ? a[idx - nx] : bc;
    const float    d   = (ay + 1u < ny && mask[idx + nx]) ? a[idx + nx] : bc;
    return (l + r + u + d) - 4.0f * c;
}

/**
 * kernel_dm_plate_step() - Single-SM shared-memory coupled 2D plate PDE integrator (dyn_mode == 3)
 *
 * Role:
 *   Executes all `nsub` CFL-safe symplectic Verlet sub-steps inside SM shared memory for circular
 *   actuator lattices up to `64x64` (`4096` actuators, `1024` threads x `4` cells/thread), with
 *   precomputed coordinates (zero integer divisions in the inner loop) and 2 barriers per substep.
 */
__global__ void kernel_dm_plate_step(
    const float *__restrict__   d_cmd_del,
    const uint8_t *__restrict__ d_mask,
    float *__restrict__         d_w,
    float *__restrict__         d_v,
    float *__restrict__         d_uf,
    float *__restrict__         d_act_pos,
    uint32_t                    nx,
    uint32_t                    ny,
    uint32_t                    nsub,
    float                       dt_sub,
    float                       a_rc,
    float                       w02,
    float                       cm2,
    float                       beta2,
    float                       two_g0,
    float                       two_g2,
    int32_t                     drive,
    int32_t                     rim_bc,
    int32_t                     do_mean)
{
    extern __shared__ unsigned char smem[];
    const uint32_t                  nact   = nx * ny;
    float                          *s_w    = (float *) smem;
    float                          *s_v    = s_w + nact;
    float                          *s_e    = s_v + nact;
    float                          *s_lap  = s_e + nact;
    uint8_t                        *s_mask = (uint8_t *) (s_lap + nact);

    const uint32_t tid = threadIdx.x;
    const uint32_t bsz = blockDim.x;

    /* Per-thread register caches for up to 4 grid cells (4096 / 1024 == 4) */
    float    u_del[4];
    float    uf_reg[4];
    float    mean_sum[4];
    float    lv_reg[4];
    uint16_t ax_reg[4];
    uint16_t ay_reg[4];
    uint8_t  mask_reg[4];

    uint32_t n_own = 0u;
    for (uint32_t idx = tid; idx < nact; idx += bsz)
    {
        uint32_t ay      = idx / nx;
        uint32_t ax      = idx - ay * nx;
        uint8_t  m       = d_mask[idx];
        float    w0      = d_w[idx];
        float    ud      = d_cmd_del[idx];
        float    uf0     = d_uf[idx];
        float    un      = uf0 + a_rc * (ud - uf0);
        s_w[idx]         = w0;
        s_v[idx]         = d_v[idx];
        s_mask[idx]      = m;
        s_e[idx]         = m ? ((drive == 0) ? (w0 - un) : w0) : 0.0f;
        u_del[n_own]     = ud;
        uf_reg[n_own]    = un;
        mean_sum[n_own]  = 0.0f;
        ax_reg[n_own]    = (uint16_t) ax;
        ay_reg[n_own]    = (uint16_t) ay;
        mask_reg[n_own]  = m;
        n_own++;
    }
    __syncthreads();

    for (uint32_t s = 0u; s < nsub; s++)
    {
        /* Pass A: Discrete 5-pt Laplacian of e (shared) and v (register), Jacobi-symmetric */
        for (uint32_t j = 0u; j < n_own; j++)
        {
            uint32_t idx = tid + j * bsz;
            if (mask_reg[j])
            {
                uint32_t ax = (uint32_t) ax_reg[j];
                uint32_t ay = (uint32_t) ay_reg[j];
                s_lap[idx]  = dev_plate_lap(s_e, s_mask, nx, ny, ax, ay, rim_bc);
                lv_reg[j]   = (two_g2 > 0.0f) ? dev_plate_lap(s_v, s_mask, nx, ny, ax, ay, rim_bc)
                                              : 0.0f;
            }
            else
            {
                s_lap[idx] = 0.0f;
                lv_reg[j]  = 0.0f;
            }
        }
        __syncthreads();

        /* Pass B: Biharmonic lap(lap(e)), Verlet update, and fused RC/error prep for s + 1 */
        for (uint32_t j = 0u; j < n_own; j++)
        {
            uint32_t idx = tid + j * bsz;
            float    wn;
            if (mask_reg[j])
            {
                uint32_t ax  = (uint32_t) ax_reg[j];
                uint32_t ay  = (uint32_t) ay_reg[j];
                float    lle = (beta2 > 0.0f)
                                   ? dev_plate_lap(s_lap, s_mask, nx, ny, ax, ay, rim_bc)
                                   : 0.0f;
                float    Ke  = w02 * s_e[idx] - cm2 * s_lap[idx] + beta2 * lle;
                float    Cv  = two_g0 * s_v[idx] - two_g2 * lv_reg[j];
                float    rhs = (drive == 1) ? (w02 * uf_reg[j]) : 0.0f;
                float    vn  = s_v[idx] + dt_sub * (rhs - Ke - Cv);
                wn           = s_w[idx] + dt_sub * vn;
                s_v[idx]     = vn;
                s_w[idx]     = wn;
            }
            else
            {
                s_v[idx] = 0.0f;
                wn       = (drive == 0) ? uf_reg[j] : 0.0f;
                s_w[idx] = wn;
            }
            mean_sum[j] += wn;
            if (s + 1u < nsub)
            {
                float un  = uf_reg[j] + a_rc * (u_del[j] - uf_reg[j]);
                uf_reg[j] = un;
                s_e[idx]  = mask_reg[j] ? ((drive == 0) ? (wn - un) : wn) : 0.0f;
            }
        }
        __syncthreads();
    }

    const float inv_sub = 1.0f / (float) nsub;
    for (uint32_t j = 0u; j < n_own; j++)
    {
        uint32_t idx   = tid + j * bsz;
        d_w[idx]       = s_w[idx];
        d_v[idx]       = s_v[idx];
        d_uf[idx]      = uf_reg[j];
        d_act_pos[idx] = do_mean ? (mean_sum[j] * inv_sub) : s_w[idx];
    }
}

/**
 * cacaosim_dm_cuda_alloc() - Allocate CUDA device buffers and upload static 16-tap stencil
 * @ctx:        DM context with precomputed host `stencil_idx` and `stencil_w` arrays
 * @ext_stream: Optional external `cudaStream_t` (`void *`) to borrow; NULL creates a private
 *              `cudaStreamNonBlocking` stream
 *
 * Role:
 *   Cold-path GPU resource allocator called from `cacaosim_dm_init()`.
 *
 * Rationale:
 *   Allocates `d_act_pos`, `d_stencil_idx`, `d_stencil_w`, and `d_opd_out` up
 *   front, zeroes dynamic state, uploads the static stencil tables asynchronously,
 *   and synchronizes `st` once so the context is immediately ready for real-time
 *   execution with zero runtime `cudaMalloc` calls. Rolls back cleanly via
 *   `cacaosim_dm_cuda_free()` if any allocation fails.
 *
 * Return: `0` on success, `-1` on CUDA stream or memory allocation failure.
 */
extern "C" int cacaosim_dm_cuda_alloc(
    cacaosim_dm_ctx_t *ctx,
    void              *ext_stream)
{
    size_t nact_bytes = (size_t) ctx->nact_total * sizeof(float);
    size_t npix_bytes = (size_t) ctx->npix_wf * sizeof(float);
    size_t idx_bytes  = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL * sizeof(int32_t);
    size_t w_bytes    = (size_t) ctx->npix_wf * CACAOSIM_DM_STENCIL * sizeof(float);

    /* Borrow external pipeline stream if supplied, else create an owned non-blocking stream */
    if (ext_stream != NULL)
    {
        ctx->cuda_stream = ext_stream;
        ctx->owns_stream = 0;
    }
    else
    {
        cudaStream_t st = NULL;
        if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess)
        {
            return -1;
        }
        ctx->cuda_stream = (void *) st;
        ctx->owns_stream = 1;
    }

    if (cudaMalloc((void **) &ctx->d_act_pos, nact_bytes) != cudaSuccess ||
        cudaMalloc((void **) &ctx->d_stencil_idx, idx_bytes) != cudaSuccess ||
        cudaMalloc((void **) &ctx->d_stencil_w, w_bytes) != cudaSuccess ||
        cudaMalloc((void **) &ctx->d_opd_out, npix_bytes) != cudaSuccess)
    {
        cacaosim_dm_cuda_free(ctx);
        return -1;
    }

    /* Zero device buffers and upload immutable 16-tap stencil index/weight tables once */
    cudaStream_t st = (cudaStream_t) ctx->cuda_stream;
    cudaMemsetAsync(ctx->d_act_pos, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_opd_out, 0, npix_bytes, st);
    cudaMemcpyAsync(ctx->d_stencil_idx, ctx->stencil_idx, idx_bytes, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(ctx->d_stencil_w, ctx->stencil_w, w_bytes, cudaMemcpyHostToDevice, st);

    /* Optional single-SM shared-memory PDE buffers when dyn_mode == 3 and nact <= 4096 */
    if (ctx->cfg.dyn_mode == 3 && ctx->plate_mask != NULL && ctx->nact_total <= 4096u)
    {
        size_t smem_bytes = 4u * nact_bytes + (size_t) ctx->nact_total;
        int    dev        = 0;
        int    max_smem   = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
                cudaSuccess &&
            (size_t) max_smem >= smem_bytes &&
            cudaFuncSetAttribute(kernel_dm_plate_step,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int) smem_bytes) == cudaSuccess &&
            cudaMalloc((void **) &ctx->d_act_cmd_del, nact_bytes) == cudaSuccess &&
            cudaMalloc((void **) &ctx->d_plate_w, nact_bytes) == cudaSuccess &&
            cudaMalloc((void **) &ctx->d_plate_v, nact_bytes) == cudaSuccess &&
            cudaMalloc((void **) &ctx->d_plate_uf, nact_bytes) == cudaSuccess &&
            cudaMalloc((void **) &ctx->d_plate_mask, (size_t) ctx->nact_total) == cudaSuccess)
        {
            cudaMemsetAsync(ctx->d_act_cmd_del, 0, nact_bytes, st);
            cudaMemsetAsync(ctx->d_plate_w, 0, nact_bytes, st);
            cudaMemsetAsync(ctx->d_plate_v, 0, nact_bytes, st);
            cudaMemsetAsync(ctx->d_plate_uf, 0, nact_bytes, st);
            cudaMemcpyAsync(ctx->d_plate_mask, ctx->plate_mask, (size_t) ctx->nact_total,
                            cudaMemcpyHostToDevice, st);
            ctx->plate_use_cuda = 1;
        }
        else
        {
            /* Gracefully fall back to CPU PDE stepper while keeping CUDA 16-tap gather */
            if (ctx->d_act_cmd_del) { cudaFree(ctx->d_act_cmd_del); ctx->d_act_cmd_del = NULL; }
            if (ctx->d_plate_w) { cudaFree(ctx->d_plate_w); ctx->d_plate_w = NULL; }
            if (ctx->d_plate_v) { cudaFree(ctx->d_plate_v); ctx->d_plate_v = NULL; }
            if (ctx->d_plate_uf) { cudaFree(ctx->d_plate_uf); ctx->d_plate_uf = NULL; }
            if (ctx->d_plate_mask) { cudaFree(ctx->d_plate_mask); ctx->d_plate_mask = NULL; }
            ctx->plate_use_cuda = 0;
        }
    }

    cudaStreamSynchronize(st);
    return 0;
}

/**
 * cacaosim_dm_cuda_free() - Release CUDA device buffers and owned stream in a DM context
 * @ctx: DM context whose GPU allocations should be freed
 *
 * Role:
 *   Cold-path GPU teardown routine invoked by `cacaosim_dm_free()` and error paths.
 *
 * Rationale:
 *   Guards every pointer individually and checks `owns_stream` before calling
 *   `cudaStreamDestroy()` so borrowed `htsim` streams are never invalidated.
 */
extern "C" void cacaosim_dm_cuda_free(cacaosim_dm_ctx_t *ctx)
{
    if (ctx->d_act_pos)
    {
        cudaFree(ctx->d_act_pos);
        ctx->d_act_pos = NULL;
    }
    if (ctx->d_act_cmd_del)
    {
        cudaFree(ctx->d_act_cmd_del);
        ctx->d_act_cmd_del = NULL;
    }
    if (ctx->d_plate_w)
    {
        cudaFree(ctx->d_plate_w);
        ctx->d_plate_w = NULL;
    }
    if (ctx->d_plate_v)
    {
        cudaFree(ctx->d_plate_v);
        ctx->d_plate_v = NULL;
    }
    if (ctx->d_plate_uf)
    {
        cudaFree(ctx->d_plate_uf);
        ctx->d_plate_uf = NULL;
    }
    if (ctx->d_plate_mask)
    {
        cudaFree(ctx->d_plate_mask);
        ctx->d_plate_mask = NULL;
    }
    ctx->plate_use_cuda = 0;
    if (ctx->d_stencil_idx)
    {
        cudaFree(ctx->d_stencil_idx);
        ctx->d_stencil_idx = NULL;
    }
    if (ctx->d_stencil_w)
    {
        cudaFree(ctx->d_stencil_w);
        ctx->d_stencil_w = NULL;
    }
    if (ctx->d_opd_out)
    {
        cudaFree(ctx->d_opd_out);
        ctx->d_opd_out = NULL;
    }
    if (ctx->owns_stream && ctx->cuda_stream)
    {
        cudaStreamDestroy((cudaStream_t) ctx->cuda_stream);
    }
    ctx->cuda_stream = NULL;
    ctx->owns_stream = 0;
}

/**
 * cacaosim_dm_cuda_render() - Asynchronously upload `act_pos_render` and render 2D DM OPD on GPU
 * @ctx:          Active CUDA-enabled DM context with updated `ctx->act_pos_render`
 * @host_opd_out: Optional host destination buffer `[npix_wf]` [um]; NULL skips D2H & sync
 *
 * Role:
 *   Hot-path CUDA execution wrapper called per frame from `cacaosim_dm_step()`.
 *
 * Rationale:
 *   Skips H2D upload and `kernel_dm_gather_opd` altogether when `ctx->opd_valid` is set and the
 *   rendered actuator stroke vector has settled (`max |act_pos_render - prev_pos_render| <= 1e-7`),
 *   reusing `ctx->d_opd_out` in VRAM. Only performs a D2H copy and `cudaStreamSynchronize()` when
 *   a host SHM buffer (`host_opd_out != NULL`) requires immediate CPU visibility.
 */
extern "C" void cacaosim_dm_cuda_plate_step(
    cacaosim_dm_ctx_t *ctx,
    float              tau_sec)
{
    cudaStream_t st         = (cudaStream_t) ctx->cuda_stream;
    const uint32_t nact     = ctx->nact_total;
    const size_t nact_bytes = (size_t) nact * sizeof(float);
    const uint32_t nsub     = (ctx->plate_nsub >= 1u) ? ctx->plate_nsub : 1u;
    const float    dt_sub   = ctx->cfg.dt_sec / (float) nsub;
    const float    a_rc     = (tau_sec > 1e-7f) ? (1.0f - expf(-dt_sub / tau_sec)) : 1.0f;
    const size_t   smem     = 4u * nact_bytes + (size_t) nact;

    const uint32_t nthr = (nact <= 1024u) ? 256u : 1024u;
    cudaMemcpyAsync(ctx->d_act_cmd_del, ctx->act_cmd_delayed, nact_bytes, cudaMemcpyHostToDevice,
                    st);
    kernel_dm_plate_step<<<1, nthr, smem, st>>>(
        ctx->d_act_cmd_del, ctx->d_plate_mask, ctx->d_plate_w, ctx->d_plate_v, ctx->d_plate_uf,
        ctx->d_act_pos, ctx->cfg.nact_x, ctx->cfg.nact_y, nsub, dt_sub, a_rc,
        ctx->plate_w0 * ctx->plate_w0, ctx->plate_cm_p * ctx->plate_cm_p,
        ctx->plate_beta_p * ctx->plate_beta_p, 2.0f * ctx->plate_g0, 2.0f * ctx->plate_g2,
        ctx->cfg.plate_drive, ctx->cfg.rim_bc, !ctx->cfg.no_zoh_mean);
    ctx->opd_valid = 0;
}

extern "C" void cacaosim_dm_cuda_plate_reset(cacaosim_dm_ctx_t *ctx)
{
    if (!ctx->plate_use_cuda)
    {
        return;
    }
    cudaStream_t st         = (cudaStream_t) ctx->cuda_stream;
    size_t       nact_bytes = (size_t) ctx->nact_total * sizeof(float);
    cudaMemsetAsync(ctx->d_act_cmd_del, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_plate_w, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_plate_v, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_plate_uf, 0, nact_bytes, st);
    cudaMemsetAsync(ctx->d_act_pos, 0, nact_bytes, st);
}

extern "C" void cacaosim_dm_cuda_plate_sync_host(cacaosim_dm_ctx_t *ctx)
{
    if (!ctx->plate_use_cuda)
    {
        return;
    }
    cudaStream_t st         = (cudaStream_t) ctx->cuda_stream;
    size_t       nact_bytes = (size_t) ctx->nact_total * sizeof(float);
    cudaMemcpyAsync(ctx->act_pos_render, ctx->d_act_pos, nact_bytes, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(ctx->act_pos, ctx->d_plate_w, nact_bytes, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(ctx->act_vel, ctx->d_plate_v, nact_bytes, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}

extern "C" void cacaosim_dm_cuda_render(
    cacaosim_dm_ctx_t *ctx,
    float             *host_opd_out)
{
    cudaStream_t st         = (cudaStream_t) ctx->cuda_stream;
    const float *pos_ren    = ctx->act_pos_render;
    uint32_t     nact       = ctx->nact_total;
    int          plate_gpu  = (ctx->plate_use_cuda && ctx->cfg.dyn_mode == 3);

    if (plate_gpu)
    {
        /* d_act_pos was already written on `st` by kernel_dm_plate_step; gather directly */
        uint32_t threads = 256u;
        uint32_t blocks  = (uint32_t) ((ctx->npix_wf + threads - 1ULL) / threads);
        kernel_dm_gather_opd<<<blocks, threads, 0, st>>>(
            ctx->d_act_pos, ctx->d_stencil_idx, ctx->d_stencil_w, ctx->npix_wf, ctx->d_opd_out);
        ctx->opd_valid = 1;
    }
    else
    {
        int need_k = !ctx->opd_valid;
        if (!need_k)
        {
            float max_d = 0.0f;
            for (uint32_t a = 0; a < nact; a++)
            {
                float d = fabsf(pos_ren[a] - ctx->prev_pos_render[a]);
                if (d > max_d)
                {
                    max_d = d;
                }
            }
            if (max_d > 1.0e-7f)
            {
                need_k = 1;
            }
        }

        if (need_k)
        {
            size_t   nact_bytes = (size_t) nact * sizeof(float);
            uint32_t threads    = 256u;
            uint32_t blocks     = (uint32_t) ((ctx->npix_wf + threads - 1ULL) / threads);

            memcpy(ctx->prev_pos_render, pos_ren, nact_bytes);
            cudaMemcpyAsync(ctx->d_act_pos, pos_ren, nact_bytes, cudaMemcpyHostToDevice, st);
            kernel_dm_gather_opd<<<blocks, threads, 0, st>>>(
                ctx->d_act_pos, ctx->d_stencil_idx, ctx->d_stencil_w, ctx->npix_wf,
                ctx->d_opd_out);
            ctx->opd_valid = 1;
        }
    }

    if (host_opd_out != NULL)
    {
        cudaMemcpyAsync(host_opd_out, ctx->d_opd_out, (size_t) ctx->npix_wf * sizeof(float),
                        cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
}
