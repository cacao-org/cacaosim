// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs_cuda.cu
 * @brief   CUDA cuFFT + CUDA Graph Pyramid WFS propagation and multi-subframe integration kernels
 *
 * Architectural Role in cacaosim:
 *   Provides the high-throughput GPU accelerator backend (`ctx->cuda_priv`) for `cacaosim.pyrwfs`,
 *   executing pupil complex-field synthesis, 2D cuFFT focal propagation, batched modulated pyramid
 *   inverse cuFFTs, area-conserving detector binning, multi-subframe (`navg`) accumulation, and
 *   parallel Philox Poisson/readout detector noise entirely on the GPU.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   1. Fourier-Shift (1 + N_mod) Batched GPU FFT Reduction:
 *      Instead of running `2 * N_mod` sequential 2D FFTs for `N_mod` pupil tip-tilt positions, a
 *      single `cufftPlan2d` transforms the zero-padded pupil field to the focal plane once per
 *      subframe (`d_foc_cplx`). `kernel_pyrwfs_mod_mul` then broadcasts `d_foc_cplx` across all
 *      `eff_npts` precomputed shifted complex pyramid masks in one L2-cached read pass, and a
 *      single `cufftPlanMany` (`plan_inv_batch`) executes all `eff_npts` 2D inverse FFTs
 *      simultaneously across GPU SMs—maximizing occupancy even on large GPUs.
 *   2. Fused Modulation Intensity Summation & Spatial Binning:
 *      Rather than materializing an intermediate `[nfft x nfft]` summed intensity buffer in global
 *      VRAM, `kernel_pyrwfs_accum_bin_int` and `kernel_pyrwfs_accum_bin_frac` fuse squared-modulus
 *      reduction across the `eff_npts` modulation planes directly with `[nfft x nfft] -> [ncam x
 *      ncam]` area-conserving box integration, halving global memory write/read traffic.
 *   3. Steady-State CUDA Graph Capture (`cudaStreamBeginCapture`):
 *      At multi-kHz AO loop rates (`2-5 kHz`), launching 5+ small kernels and internal cuFFT
 *      sub-kernels per subframe incurs tens of microseconds of CPU driver overhead. After a single
 *      warmup pass binds cuFFT workspaces, `cacaosim_pyrwfs_cuda_accum()` records the entire
 *      optical DAG into an executable `cudaGraphExec_t` and replays it with a single
 *      `cudaGraphLaunch()`, automatically invalidating and recapturing if device input pointers,
 *      wavefront unit scaling, or pyramid modulation masks change.
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>

#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "pyrwfs/cacaosim_pyrwfs.h"

typedef struct
{
    uint32_t m0[CACAOSIM_MAX_LAMBDA];
    uint32_t m1[CACAOSIM_MAX_LAMBDA];
    float    alpha[CACAOSIM_MAX_LAMBDA];
} pyrwfs_poly_map_t;

/**
 * @brief Private GPU execution state (`ctx->cuda_priv`) for the CUDA Pyramid WFS backend.
 *
 * Owns (or borrows) the execution `cudaStream_t`, forward and batched-inverse cuFFT plans,
 * persistent device buffers for static/dynamic pupil inputs and modulated complex planes, and the
 * cached `cudaGraphExec_t` along with the exact device pointer/scalar signature for which the
 * graph was recorded.
 */
typedef struct
{
    cudaStream_t     stream;         /**< Execution CUDA stream (private or shared `htsim`) */
    int              owns_stream;    /**< 1 if `stream` was created here and must be destroyed */
    cufftHandle      plan_fwd;       /**< Single 2D C2C forward FFT plan `[nfft x nfft]` */
    cufftHandle      plan_inv_batch; /**< Batched 2D C2C inverse FFT plan `[eff_npts*nfft^2]` */
    int              has_fwd;        /**< Non-zero once `plan_fwd` is valid */
    int              has_inv;        /**< Non-zero once `plan_inv_batch` is valid */

    float           *d_pup_amp;      /**< Static uploaded 2D pupil amplitude `[npix_wf]` */
    float           *d_dyn_pup;      /**< Staging buffer for dynamic host beam amplitude */
    float           *d_ncpa_opd;     /**< Optional static WFS-arm NCPA OPD `[npix_wf]` [um] */
    float           *d_atm_amp;      /**< Staging buffer for host atmospheric amplitude */
    float           *d_wf_in;        /**< Staging buffer for host wavefront/beam OPD */
    float           *d_dm_opd;       /**< Staging buffer for host DM OPD `[npix_wf]` [um] */
    float           *d_k_rad;        /**< Per-channel 2*pi/lam_c wavenumbers `[eff_nlam]` */
    float           *d_chan_w;       /**< Per-channel photon weights `[eff_nlam]` */
    float           *d_poly_opd;     /**< Host-cube staging OPD `[2 * eff_nlam * npix_wf]` */
    float           *d_poly_amp;     /**< Host-cube staging Amp `[2 * eff_nlam * npix_wf]` */
    cuFloatComplex  *d_masks;        /**< Shifted pyramid masks `[n_slices*nlam*pps*npix_fft]` */
    cuFloatComplex  *d_pup_cplx;     /**< Zero-padded pupil batch `[eff_nlam * npix_fft]` */
    cuFloatComplex  *d_foc_cplx;     /**< Unmodulated focal batch `[eff_nlam * npix_fft]` */
    cuFloatComplex  *d_mod_batch;    /**< Modulated slice batch `[eff_nlam * pps * npix_fft]` */
    float           *d_cam_int;      /**< Single-subframe binned camera intensity `[npix_cam]` */
    float           *d_accum_int;    /**< Multi-subframe (`navg > 1`) intensity accumulator */
    uint16_t        *d_out_u16;      /**< Device uint16 ADU readout buffer `[npix_cam]` */
    float           *d_out_f32;      /**< Device float32 readout buffer `[npix_cam]` */

    /* Per-slice CUDA Graphs cached for steady-state optical propagation */
    cudaGraph_t      graph[CACAOSIM_PYRWFS_MAX_SLICES];       /**< Per-slice captured graph */
    cudaGraphExec_t  graph_exec[CACAOSIM_PYRWFS_MAX_SLICES];  /**< Per-slice executable graph */
    uint8_t          graph_valid[CACAOSIM_PYRWFS_MAX_SLICES]; /**< 1 when slice graph is valid */
    uint8_t          graph_tried[CACAOSIM_PYRWFS_MAX_SLICES]; /**< 1 once slice capture tried */
    int              has_sig;                                 /**< 1 once pointer sig recorded */
    const float     *graph_pup_ptr;  /**< Bound pupil amplitude device pointer in `graph_exec` */
    const float     *graph_amp_ptr;  /**< Bound atmospheric amplitude device pointer in graph */
    const float     *graph_wf_ptr;   /**< Bound wavefront OPD device pointer in `graph_exec` */
    const float     *graph_dm_ptr;   /**< Bound DM OPD device pointer in `graph_exec` */
    float            graph_wf_scale; /**< Bound `wf_to_opd` scalar baked into `graph_exec` */
} pyrwfs_cuda_state_t;

/**
 * kernel_pyrwfs_mod_mul() - Broadcast-multiply focal field against all N_mod shifted pyramid masks
 * @foc:       Input 2D focal-plane complex amplitude `[npix_fft]`
 * @masks:     Precomputed shifted complex pyramid masks `[npts * npix_fft]`
 * @npix_fft:  Number of complex pixels per 2D FFT plane (`nfft * nfft`)
 * @npts:      Number of active modulation steps (`eff_npts`)
 * @batch_out: Output modulated focal complex batch `[npts * npix_fft]`
 *
 * Role:
 *   GPU focal-plane modulation kernel bridging the single forward cuFFT and the batched `npts`
 *   inverse cuFFT.
 *
 * Rationale:
 *   Mapping one GPU thread to each spatial frequency `p` loads `foc[p]` into registers once and
 *   loops over the `npts` strided mask planes, saving `(npts - 1) * npix_fft` global reads of the
 *   focal field while maintaining fully coalesced 8-byte `cuFloatComplex` accesses across warps.
 */
__global__ void kernel_pyrwfs_build_pupil_batch(
    const float *__restrict__ pup_amp,
    const float *__restrict__ atm_amp,
    const float *__restrict__ wf_in,
    const float *__restrict__ dm_opd,
    const float *__restrict__ ncpa_opd,
    float                     wf_to_opd,
    const float *__restrict__ k_rad,
    uint32_t                  nwf,
    uint32_t                  nfft,
    uint32_t                  pad_off,
    int32_t                   nlam,
    cuFloatComplex *__restrict__ out_cplx)
{
    uint32_t idx      = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t npix_fft = nfft * nfft;
    if (idx >= npix_fft)
    {
        return;
    }

    uint32_t ox = idx % nfft;
    uint32_t oy = idx / nfft;
    if (ox < pad_off || oy < pad_off || ox >= pad_off + nwf || oy >= pad_off + nwf)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            out_cplx[(uint64_t) c * npix_fft + idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
        return;
    }

    uint32_t pix = (oy - pad_off) * nwf + (ox - pad_off);
    float    amp = pup_amp[pix];
    if (amp == 0.0f)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            out_cplx[(uint64_t) c * npix_fft + idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
        return;
    }
    if (atm_amp != NULL)
    {
        amp *= atm_amp[pix];
    }

    float opd = 0.0f;
    if (wf_in != NULL)
    {
        opd += wf_to_opd * wf_in[pix];
    }
    if (dm_opd != NULL)
    {
        opd -= dm_opd[pix];
    }
    if (ncpa_opd != NULL)
    {
        opd += ncpa_opd[pix];
    }

    for (int32_t c = 0; c < nlam; c++)
    {
        float s, co;
        __sincosf(opd * k_rad[c], &s, &co);
        out_cplx[(uint64_t) c * npix_fft + idx] = make_cuFloatComplex(amp * co, amp * s);
    }
}

__global__ void kernel_pyrwfs_build_pupil_poly(
    const float *__restrict__ beam_opd,
    const float *__restrict__ beam_amp,
    uint64_t                  chan_stride,
    const float *__restrict__ ncpa_opd,
    const float *__restrict__ k_rad,
    pyrwfs_poly_map_t         pmap,
    uint32_t                  nwf,
    uint32_t                  nfft,
    uint32_t                  pad_off,
    int32_t                   nlam,
    cuFloatComplex *__restrict__ out_cplx)
{
    uint32_t idx      = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t npix_fft = nfft * nfft;
    if (idx >= npix_fft)
    {
        return;
    }

    uint32_t ox = idx % nfft;
    uint32_t oy = idx / nfft;
    if (ox < pad_off || oy < pad_off || ox >= pad_off + nwf || oy >= pad_off + nwf)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            out_cplx[(uint64_t) c * npix_fft + idx] = make_cuFloatComplex(0.0f, 0.0f);
        }
        return;
    }

    uint32_t pix = (oy - pad_off) * nwf + (ox - pad_off);
    float    nc  = (ncpa_opd != NULL) ? ncpa_opd[pix] : 0.0f;

    for (int32_t c = 0; c < nlam; c++)
    {
        uint32_t m0   = pmap.m0[c];
        uint32_t m1   = pmap.m1[c];
        float    w1   = pmap.alpha[c];
        float    w0   = 1.0f - w1;
        float    kc   = k_rad[c];
        uint64_t off0 = (uint64_t) m0 * chan_stride + pix;
        uint64_t off1 = (uint64_t) m1 * chan_stride + pix;

        float a0 = (beam_amp != NULL) ? beam_amp[off0] : 1.0f;
        float a1 = (beam_amp != NULL) ? beam_amp[off1] : 1.0f;
        if (a0 == 0.0f && a1 == 0.0f)
        {
            out_cplx[(uint64_t) c * npix_fft + idx] = make_cuFloatComplex(0.0f, 0.0f);
            continue;
        }

        float o0 = ((beam_opd != NULL) ? beam_opd[off0] : 0.0f) + nc;
        float s0, c0;
        __sincosf(o0 * kc, &s0, &c0);
        float re = w0 * a0 * c0;
        float im = w0 * a0 * s0;

        if (m1 != m0 && w1 > 0.0f)
        {
            float o1 = ((beam_opd != NULL) ? beam_opd[off1] : 0.0f) + nc;
            float s1, c1;
            __sincosf(o1 * kc, &s1, &c1);
            re += w1 * a1 * c1;
            im += w1 * a1 * s1;
        }
        out_cplx[(uint64_t) c * npix_fft + idx] = make_cuFloatComplex(re, im);
    }
}

__global__ void kernel_pyrwfs_mod_mul(
    const cuFloatComplex *__restrict__ foc,
    const cuFloatComplex *__restrict__ masks,
    uint64_t                           npix_fft,
    int32_t                            pps,
    cuFloatComplex *__restrict__       batch_out)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix_fft)
    {
        return;
    }

    uint32_t       c    = blockIdx.y;
    cuFloatComplex e    = foc[(uint64_t) c * npix_fft + p];
    uint64_t       base = (uint64_t) c * (uint64_t) pps * npix_fft + p;
    for (int32_t j = 0; j < pps; j++)
    {
        uint64_t       idx = base + (uint64_t) j * npix_fft;
        cuFloatComplex m   = masks[idx];
        batch_out[idx]     = make_cuFloatComplex(e.x * m.x - e.y * m.y, e.x * m.y + e.y * m.x);
    }
}

__global__ void kernel_pyrwfs_accum_bin_int(
    const cuFloatComplex *__restrict__ batch_det,
    const float *__restrict__          chan_w,
    uint32_t                           nfft,
    uint32_t                           ncam,
    uint32_t                           bin,
    int32_t                            nlam,
    int32_t                            pps,
    float *__restrict__                cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx       = cid % ncam;
    uint32_t cy       = cid / ncam;
    uint32_t x0       = cx * bin;
    uint32_t y0       = cy * bin;
    uint64_t npix_fft = (uint64_t) nfft * nfft;
    float    acc      = 0.0f;

    for (uint32_t by = 0; by < bin; by++)
    {
        uint64_t row_off = (uint64_t) (y0 + by) * nfft + x0;
        for (uint32_t bx = 0; bx < bin; bx++)
        {
            uint64_t pix = row_off + bx;
            for (int32_t c = 0; c < nlam; c++)
            {
                float    wc    = chan_w[c];
                float    sub   = 0.0f;
                uint64_t c_off = (uint64_t) (c * pps) * npix_fft + pix;
                for (int32_t j = 0; j < pps; j++)
                {
                    cuFloatComplex v = batch_det[c_off + (uint64_t) j * npix_fft];
                    sub += v.x * v.x + v.y * v.y;
                }
                acc += wc * sub;
            }
        }
    }
    cam_int[cid] = acc;
}

__global__ void kernel_pyrwfs_accum_bin_frac(
    const cuFloatComplex *__restrict__ batch_det,
    const float *__restrict__          chan_w,
    uint32_t                           nfft,
    uint32_t                           ncam,
    int32_t                            nlam,
    int32_t                            pps,
    float *__restrict__                cam_int)
{
    uint32_t cid   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncam * ncam;
    if (cid >= total)
    {
        return;
    }

    uint32_t cx       = cid % ncam;
    uint32_t cy       = cid / ncam;
    float    scale    = (float) nfft / (float) ncam;
    float    fx0      = (float) cx * scale;
    float    fx1      = (float) (cx + 1u) * scale;
    float    fy0      = (float) cy * scale;
    float    fy1      = (float) (cy + 1u) * scale;
    int      ix0      = (int) floorf(fx0);
    int      ix1      = (int) ceilf(fx1);
    int      iy0      = (int) floorf(fy0);
    int      iy1      = (int) ceilf(fy1);
    uint64_t npix_fft = (uint64_t) nfft * nfft;
    float    acc      = 0.0f;

    for (int iy = iy0; iy < iy1; iy++)
    {
        if (iy < 0 || iy >= (int) nfft)
        {
            continue;
        }
        float wy0 = (fy0 > (float) iy) ? fy0 : (float) iy;
        float wy1 = (fy1 < (float) (iy + 1)) ? fy1 : (float) (iy + 1);
        float wy  = wy1 - wy0;
        if (wy <= 0.0f)
        {
            continue;
        }
        for (int ix = ix0; ix < ix1; ix++)
        {
            if (ix < 0 || ix >= (int) nfft)
            {
                continue;
            }
            float wx0 = (fx0 > (float) ix) ? fx0 : (float) ix;
            float wx1 = (fx1 < (float) (ix + 1)) ? fx1 : (float) (ix + 1);
            float wx  = wx1 - wx0;
            if (wx > 0.0f)
            {
                uint64_t pix = (uint64_t) iy * nfft + (uint32_t) ix;
                float    p_i = 0.0f;
                for (int32_t c = 0; c < nlam; c++)
                {
                    float    wc    = chan_w[c];
                    float    sub   = 0.0f;
                    uint64_t c_off = (uint64_t) (c * pps) * npix_fft + pix;
                    for (int32_t j = 0; j < pps; j++)
                    {
                        cuFloatComplex v = batch_det[c_off + (uint64_t) j * npix_fft];
                        sub += v.x * v.x + v.y * v.y;
                    }
                    p_i += wc * sub;
                }
                acc += wx * wy * p_i;
            }
        }
    }
    cam_int[cid] = acc;
}

/**
 * kernel_pyrwfs_add_accum() - Accumulate single-subframe camera intensity into exposure buffer
 * @cam_int:   Current subframe binned intensity `[npix]`
 * @npix:      Total camera pixels (`ncam * ncam`)
 * @accum_int: In-place multi-subframe accumulator `[npix]`
 *
 * Role:
 *   Hot-loop multi-subframe (`navg > 1`) incoherent intensity summation kernel.
 *
 * Rationale:
 *   Kept outside the captured steady-state optical CUDA Graph so subframe 0 (`navg == 1`) skips
 *   accumulation entirely without needing two separate graph instantiations.
 */
__global__ void kernel_pyrwfs_add_accum(
    const float *__restrict__ cam_int,
    uint64_t                  npix,
    float *__restrict__       accum_int)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        accum_int[i] += cam_int[i];
    }
}

/**
 * kernel_pyrwfs_scale_accum() - Normalize multi-subframe exposure sum by `1 / subframe_cnt`
 * @accum_int: Summed multi-subframe camera intensity `[npix]`
 * @inv_n:     Reciprocal subframe count (`1.0f / subframe_cnt`)
 * @npix:      Total camera pixels (`ncam * ncam`)
 * @cam_int:   Normalized output camera intensity `[npix]` ready for detector readout
 *
 * Role:
 *   Normalizes accumulated `navg > 1` optical intensity prior to `cacaosim_cuda_detector_readout`.
 *
 * Rationale:
 *   Preserves the invariant that `det_live->flux_ph` specifies total photo-events per WFS camera
 *   readout independent of how many atmospheric subframes were averaged during the exposure.
 */
__global__ void kernel_pyrwfs_scale_accum(
    const float *__restrict__ accum_int,
    float                     inv_n,
    uint64_t                  npix,
    float *__restrict__       cam_int)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        cam_int[i] = accum_int[i] * inv_n;
    }
}

/**
 * pyrwfs_cuda_invalidate_graph() - Destroy cached CUDA Graph so next frame recaptures cleanly
 * @cs: Private CUDA Pyramid WFS state (non-NULL)
 *
 * Role:
 *   Invalidates `cs->graph_exec` and resets `cs->graph_tried` whenever bound device pointers,
 *   wavefront scale, or pyramid masks change, or during teardown.
 *
 * Rationale:
 *   CUDA Graphs bake kernel pointer arguments and scalar parameters at capture time; explicit
 *   invalidation guarantees correctness across dynamic stream connections or `turb_gain` toggles.
 */
static void pyrwfs_cuda_invalidate_graph(pyrwfs_cuda_state_t *cs)
{
    for (int s = 0; s < CACAOSIM_PYRWFS_MAX_SLICES; s++)
    {
        if (cs->graph_valid[s])
        {
            cudaGraphExecDestroy(cs->graph_exec[s]);
            cudaGraphDestroy(cs->graph[s]);
            cs->graph_valid[s] = 0;
        }
        cs->graph_tried[s] = 0;
    }
    cs->has_sig = 0;
}

/**
 * pyrwfs_cuda_launch_optics() - Enqueue 5-stage GPU Pyramid WFS optical pipeline on `cs->stream`
 * @ctx:           Parent Pyramid WFS context (non-NULL)
 * @cs:            Private CUDA state (non-NULL)
 * @d_pup_ptr:     Device pupil/beam amplitude `[npix_wf]`
 * @d_atm_amp_ptr: Optional device atmospheric amplitude `[npix_wf]` (NULL -> 1.0)
 * @d_wf_ptr:      Optional device wavefront/beam OPD `[npix_wf]` (NULL -> 0.0)
 * @d_dm_ptr:      Optional device DM OPD `[npix_wf]` in microns (NULL -> 0.0)
 * @wf_to_opd:     Scale converting `@d_wf_ptr` to microns OPD
 * @slice_idx:     Active modulation slice index `[0 .. n_slices - 1]`
 *
 * Role:
 *   Enqueues (or records during stream capture) the complete single-subframe optical sequence:
 *   padded pupil phasor synthesis -> 2D forward cuFFT -> shifted mask broadcast multiply ->
 *   batched 2D inverse cuFFT -> fused intensity sum & camera binning.
 *
 * Rationale:
 *   Factoring the pure device-to-device optical pipeline into one helper allows identical code to
 *   serve initialization warmup, per-slice `cudaStreamBeginCapture` recording, and un-graphed
 *   fallback.
 */
static void pyrwfs_cuda_run_fft_and_bin(
    cacaosim_pyrwfs_ctx_t *ctx,
    pyrwfs_cuda_state_t   *cs,
    int32_t                slice_idx)
{
    uint32_t nfft      = ctx->nfft;
    uint32_t ncam      = ctx->cfg.ncam;
    int32_t  nlam      = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    int32_t  pps       = (ctx->pts_per_slice >= 1) ? ctx->pts_per_slice : ctx->eff_npts;
    int32_t  s_clamped = (ctx->n_slices > 1) ? (slice_idx % ctx->n_slices) : 0;
    const cuFloatComplex *d_slice_masks =
        cs->d_masks + (size_t) s_clamped * (size_t) nlam * (size_t) pps * ctx->npix_fft;

    cufftExecC2C(cs->plan_fwd, cs->d_pup_cplx, cs->d_foc_cplx, CUFFT_FORWARD);

    uint32_t th_fft = 256u;
    dim3     bl_mod((uint32_t) ((ctx->npix_fft + th_fft - 1ULL) / th_fft), (uint32_t) nlam, 1u);
    kernel_pyrwfs_mod_mul<<<bl_mod, th_fft, 0, cs->stream>>>(
        cs->d_foc_cplx, d_slice_masks, ctx->npix_fft, pps, cs->d_mod_batch);

    cufftExecC2C(cs->plan_inv_batch, cs->d_mod_batch, cs->d_mod_batch, CUFFT_INVERSE);

    uint32_t th_cam = 256u;
    uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
    if (nfft >= ncam && (nfft % ncam) == 0u)
    {
        kernel_pyrwfs_accum_bin_int<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_mod_batch, cs->d_chan_w, nfft, ncam, nfft / ncam, nlam, pps, cs->d_cam_int);
    }
    else
    {
        kernel_pyrwfs_accum_bin_frac<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_mod_batch, cs->d_chan_w, nfft, ncam, nlam, pps, cs->d_cam_int);
    }
}

static void pyrwfs_cuda_launch_optics(
    cacaosim_pyrwfs_ctx_t *ctx,
    pyrwfs_cuda_state_t   *cs,
    const float           *d_pup_ptr,
    const float           *d_atm_amp_ptr,
    const float           *d_wf_ptr,
    const float           *d_dm_ptr,
    float                  wf_to_opd,
    int32_t                slice_idx)
{
    uint32_t nfft    = ctx->nfft;
    uint32_t nwf     = ctx->cfg.nwf;
    uint32_t pad_off = (nfft >= nwf) ? ((nfft - nwf) / 2u) : 0u;
    int32_t  nlam    = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    uint32_t th_fft  = 256u;
    uint32_t bl_fft  = (uint32_t) ((ctx->npix_fft + th_fft - 1ULL) / th_fft);

    kernel_pyrwfs_build_pupil_batch<<<bl_fft, th_fft, 0, cs->stream>>>(
        d_pup_ptr, d_atm_amp_ptr, d_wf_ptr, d_dm_ptr, cs->d_ncpa_opd, wf_to_opd, cs->d_k_rad,
        nwf, nfft, pad_off, nlam, cs->d_pup_cplx);

    pyrwfs_cuda_run_fft_and_bin(ctx, cs, slice_idx);
}

/**
 * cacaosim_pyrwfs_cuda_init() - Allocate GPU buffers, upload static maps/masks, and plan cuFFTs
 * @ctx:        Host Pyramid WFS context with dimensions and `masks_cplx` populated
 * @pup_amp:    Static host 2D pupil amplitude `[npix_wf]`
 * @ncpa_opd:   Optional static host 2D NCPA OPD `[npix_wf]` in microns (NULL -> none)
 * @ext_stream: Optional external `cudaStream_t` to share in fused benches (NULL creates private)
 *
 * Role:
 *   Cold-path GPU backend constructor allocating all VRAM buffers, uploading static maps/masks,
 *   planning single forward + batched inverse 2D cuFFTs, and warming up cuFFT workspaces.
 *
 * Rationale:
 *   Executing one synchronous warmup pass at the end of `init` forces cuFFT lazy internal device
 *   allocations to complete outside stream capture, preventing `cudaErrorStreamCaptureUnsupported`
 *   when `cacaosim_pyrwfs_cuda_accum()` later records its CUDA Graph.
 *
 * Return: 0 on success, -1 on CUDA/cuFFT allocation or planning failure.
 */
extern "C" int cacaosim_pyrwfs_cuda_init(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *ncpa_opd,
    void                  *ext_stream)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) calloc(1, sizeof(pyrwfs_cuda_state_t));
    if (cs == NULL)
    {
        return -1;
    }
    ctx->cuda_priv = cs;

    if (ext_stream != NULL)
    {
        cs->stream      = (cudaStream_t) ext_stream;
        cs->owns_stream = 0;
    }
    else
    {
        if (cudaStreamCreateWithFlags(&cs->stream, cudaStreamNonBlocking) != cudaSuccess)
        {
            cacaosim_pyrwfs_cuda_free(ctx);
            return -1;
        }
        cs->owns_stream = 1;
    }

    int32_t nlam         = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    int32_t pps          = (ctx->pts_per_slice >= 1) ? ctx->pts_per_slice : ctx->eff_npts;
    size_t  wf_bytes     = (size_t) ctx->npix_wf * sizeof(float);
    size_t  poly_bytes   = (size_t) (2 * nlam) * wf_bytes;
    size_t  lam_f_bytes  = (size_t) nlam * sizeof(float);
    size_t  fft_c_bytes  = (size_t) ctx->npix_fft * sizeof(cuFloatComplex);
    size_t  fwd_c_bytes  = (size_t) nlam * fft_c_bytes;
    size_t  mask_c_bytes = (size_t) ctx->eff_npts * (size_t) nlam * fft_c_bytes;
    size_t  bat_c_bytes  = (size_t) nlam * (size_t) pps * fft_c_bytes;
    size_t  cam_f_bytes  = (size_t) ctx->npix_cam * sizeof(float);
    size_t  cam_u_bytes  = (size_t) ctx->npix_cam * sizeof(uint16_t);

    if (cudaMalloc((void **) &cs->d_pup_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dyn_pup, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_atm_amp, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_wf_in, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_dm_opd, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_k_rad, lam_f_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_chan_w, lam_f_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_poly_opd, poly_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_poly_amp, poly_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_masks, mask_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_pup_cplx, fwd_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_foc_cplx, fwd_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_mod_batch, bat_c_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_cam_int, cam_f_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_accum_int, cam_f_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_u16, cam_u_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_f32, cam_f_bytes) != cudaSuccess)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
        return -1;
    }

    if (ncpa_opd != NULL)
    {
        if (cudaMalloc((void **) &cs->d_ncpa_opd, wf_bytes) != cudaSuccess)
        {
            cacaosim_pyrwfs_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_ncpa_opd, ncpa_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    }

    cudaMemcpyAsync(cs->d_pup_amp, pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_k_rad, ctx->k_rad, lam_f_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_chan_w, ctx->chan_w, lam_f_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaMemcpyAsync(cs->d_masks, ctx->masks_cplx, mask_c_bytes, cudaMemcpyHostToDevice,
                    cs->stream);
    cudaMemsetAsync(cs->d_wf_in, 0, wf_bytes, cs->stream);
    cudaMemsetAsync(cs->d_dm_opd, 0, wf_bytes, cs->stream);

    int n[2]       = { (int) ctx->nfft, (int) ctx->nfft };
    int inembed[2] = { n[0], n[1] };
    int idist      = (int) ctx->npix_fft;
    if (cufftPlanMany(&cs->plan_fwd, 2, n, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C,
                      nlam) != CUFFT_SUCCESS)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
        return -1;
    }
    cs->has_fwd = 1;
    cufftSetStream(cs->plan_fwd, cs->stream);

    if (cufftPlanMany(&cs->plan_inv_batch, 2, n, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C,
                      nlam * pps) != CUFFT_SUCCESS)
    {
        cacaosim_pyrwfs_cuda_free(ctx);
        return -1;
    }
    cs->has_inv = 1;
    cufftSetStream(cs->plan_inv_batch, cs->stream);

    /* Warmup run so cuFFT internal workspaces are bound before any CUDA Graph capture */
    pyrwfs_cuda_launch_optics(ctx, cs, cs->d_pup_amp, NULL, NULL, NULL, 1.0f, 0);
    cudaStreamSynchronize(cs->stream);
    return 0;
}

/**
 * cacaosim_pyrwfs_cuda_upload_masks() - Copy rebuilt host pyramid mask stack to GPU memory
 * @ctx: Pyramid WFS context with updated `ctx->masks_cplx`
 *
 * Role:
 *   Uploads `ctx->masks_cplx` to `cs->d_masks` and invalidates any cached CUDA Graph.
 *
 * Rationale:
 *   Called only when `mod_rad` changes at runtime; synchronizing the stream ensures host `masks`
 *   memory is safe to reuse or inspect immediately upon return.
 */
extern "C" void cacaosim_pyrwfs_cuda_upload_masks(cacaosim_pyrwfs_ctx_t *ctx)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL || cs->d_masks == NULL)
    {
        return;
    }
    pyrwfs_cuda_invalidate_graph(cs);
    int32_t nlam       = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    size_t  bat_c_bytes =
        (size_t) ctx->eff_npts * (size_t) nlam * ctx->npix_fft * sizeof(cuFloatComplex);
    cudaMemcpyAsync(cs->d_masks, ctx->masks_cplx, bat_c_bytes, cudaMemcpyHostToDevice, cs->stream);
    cudaStreamSynchronize(cs->stream);
}

extern "C" void cacaosim_pyrwfs_cuda_upload_weights(cacaosim_pyrwfs_ctx_t *ctx)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL || cs->d_chan_w == NULL)
    {
        return;
    }
    int32_t nlam = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    cudaMemcpyAsync(cs->d_chan_w, ctx->chan_w, (size_t) nlam * sizeof(float),
                    cudaMemcpyHostToDevice, cs->stream);
}

/**
 * cacaosim_pyrwfs_cuda_free() - Release all GPU plans, CUDA Graphs, device buffers, and streams
 * @ctx: Pyramid WFS context whose `cuda_priv` state will be destroyed
 *
 * Role:
 *   Cold-path GPU teardown routine freeing all CUDA resources owned by `@ctx`.
 *
 * Rationale:
 *   Destroys `cs->stream` only when `cs->owns_stream == 1` so shared streams managed by the fused
 *   `cacaosim.htsim` engine remain valid for sibling compute units.
 */
extern "C" void cacaosim_pyrwfs_cuda_free(cacaosim_pyrwfs_ctx_t *ctx)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL)
    {
        return;
    }
    pyrwfs_cuda_invalidate_graph(cs);
    if (cs->has_fwd)
    {
        cufftDestroy(cs->plan_fwd);
    }
    if (cs->has_inv)
    {
        cufftDestroy(cs->plan_inv_batch);
    }
    cudaFree(cs->d_pup_amp);
    cudaFree(cs->d_dyn_pup);
    cudaFree(cs->d_ncpa_opd);
    cudaFree(cs->d_atm_amp);
    cudaFree(cs->d_wf_in);
    cudaFree(cs->d_dm_opd);
    cudaFree(cs->d_k_rad);
    cudaFree(cs->d_chan_w);
    cudaFree(cs->d_poly_opd);
    cudaFree(cs->d_poly_amp);
    cudaFree(cs->d_masks);
    cudaFree(cs->d_pup_cplx);
    cudaFree(cs->d_foc_cplx);
    cudaFree(cs->d_mod_batch);
    cudaFree(cs->d_cam_int);
    cudaFree(cs->d_accum_int);
    cudaFree(cs->d_out_u16);
    cudaFree(cs->d_out_f32);
    if (cs->owns_stream && cs->stream)
    {
        cudaStreamDestroy(cs->stream);
    }
    free(cs);
    ctx->cuda_priv = NULL;
}

/**
 * cacaosim_pyrwfs_cuda_accum() - Execute or replay GPU Pyramid WFS subframe propagation
 * @ctx:           Active Pyramid WFS context (`use_cuda == 1`)
 * @dyn_pup_amp:   Optional dynamic host pupil/beam amplitude `[npix_wf]`
 * @atm_amp:       Optional host atmospheric scintillation amplitude `[npix_wf]`
 * @wf_in:         Optional host wavefront or beam OPD `[npix_wf]`
 * @dm_opd:        Optional host DM OPD `[npix_wf]` in microns
 * @d_pup_amp_ext: Optional resident device pupil/beam amplitude pointer `[npix_wf]`
 * @d_wf_opd_ext:  Optional resident device wavefront/beam OPD pointer `[npix_wf]`
 * @d_dm_opd_ext:  Optional resident device DM OPD pointer `[npix_wf]`
 * @wf_to_opd:     Scale converting `@wf_in` / `@d_wf_opd_ext` to microns OPD
 *
 * Role:
 *   Hot-loop GPU subframe dispatcher handling optional H2D input staging, lazy `navg > 1`
 *   accumulator seeding, CUDA Graph capture/replay of the 5-stage optical pipeline, and multi-
 *   subframe intensity summation.
 *
 * Rationale:
 *   When external device pointers (`d_*_ext`) are supplied by `cacaosim.htsim`, zero H2D copies
 *   occur and the entire optical propagation executes via a single `cudaGraphLaunch()`. Even in
 *   standalone SHM mode, H2D staging into fixed `cs->d_*` buffers preserves constant device
 *   addresses across frames, allowing `cs->graph_exec` to remain valid indefinitely. If stream
 *   capture is rejected by a driver/profiler configuration, `cs->graph_tried` prevents repeated
 *   capture attempts and falls back cleanly to direct stream launches.
 */
extern "C" void cacaosim_pyrwfs_cuda_accum(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *dyn_pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *d_pup_amp_ext,
    const float           *d_wf_opd_ext,
    const float           *d_dm_opd_ext,
    float                  wf_to_opd)
{
    pyrwfs_cuda_state_t *cs       = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    size_t               wf_bytes = (size_t) ctx->npix_wf * sizeof(float);

    const float *d_pup_ptr = cs->d_pup_amp;
    if (d_pup_amp_ext != NULL)
    {
        d_pup_ptr = d_pup_amp_ext;
    }
    else if (dyn_pup_amp != NULL)
    {
        cudaMemcpyAsync(cs->d_dyn_pup, dyn_pup_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_pup_ptr = cs->d_dyn_pup;
    }

    const float *d_amp_ptr = NULL;
    if (atm_amp != NULL)
    {
        cudaMemcpyAsync(cs->d_atm_amp, atm_amp, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_amp_ptr = cs->d_atm_amp;
    }

    const float *d_wf_ptr = NULL;
    if (d_wf_opd_ext != NULL)
    {
        d_wf_ptr = d_wf_opd_ext;
    }
    else if (wf_in != NULL && wf_to_opd != 0.0f)
    {
        cudaMemcpyAsync(cs->d_wf_in, wf_in, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_wf_ptr = cs->d_wf_in;
    }

    const float *d_dm_ptr = NULL;
    if (d_dm_opd_ext != NULL)
    {
        d_dm_ptr = d_dm_opd_ext;
    }
    else if (dm_opd != NULL)
    {
        cudaMemcpyAsync(cs->d_dm_opd, dm_opd, wf_bytes, cudaMemcpyHostToDevice, cs->stream);
        d_dm_ptr = cs->d_dm_opd;
    }

    /* Lazily seed multi-subframe accumulator only when a second subframe arrives (navg > 1) */
    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    int32_t s = (ctx->n_slices > 1) ? (ctx->slice_cursor % ctx->n_slices) : 0;
    if (s < 0 || s >= CACAOSIM_PYRWFS_MAX_SLICES)
    {
        s = 0;
    }

    /* Re-validate or capture steady-state optical CUDA Graph when pointer signature is fixed */
    if (cs->has_sig &&
        (cs->graph_pup_ptr != d_pup_ptr || cs->graph_amp_ptr != d_amp_ptr ||
         cs->graph_wf_ptr != d_wf_ptr || cs->graph_dm_ptr != d_dm_ptr ||
         fabsf(cs->graph_wf_scale - wf_to_opd) > 1e-7f))
    {
        pyrwfs_cuda_invalidate_graph(cs);
    }

    if (!cs->graph_valid[s] && !cs->graph_tried[s])
    {
        pyrwfs_cuda_launch_optics(ctx, cs, d_pup_ptr, d_amp_ptr, d_wf_ptr, d_dm_ptr, wf_to_opd, s);
        cudaStreamSynchronize(cs->stream);

        cs->graph_tried[s] = 1;
        if (cudaStreamBeginCapture(cs->stream, cudaStreamCaptureModeRelaxed) == cudaSuccess)
        {
            pyrwfs_cuda_launch_optics(ctx, cs, d_pup_ptr, d_amp_ptr, d_wf_ptr, d_dm_ptr, wf_to_opd,
                                      s);
            if (cudaStreamEndCapture(cs->stream, &cs->graph[s]) == cudaSuccess)
            {
                if (cudaGraphInstantiate(&cs->graph_exec[s], cs->graph[s], NULL, NULL, 0) ==
                    cudaSuccess)
                {
                    cs->graph_valid[s] = 1;
                    cs->has_sig        = 1;
                    cs->graph_pup_ptr  = d_pup_ptr;
                    cs->graph_amp_ptr  = d_amp_ptr;
                    cs->graph_wf_ptr   = d_wf_ptr;
                    cs->graph_dm_ptr   = d_dm_ptr;
                    cs->graph_wf_scale = wf_to_opd;
                }
                else
                {
                    cudaGraphDestroy(cs->graph[s]);
                    cudaGetLastError();
                }
            }
            else
            {
                cudaGetLastError();
            }
        }
        else
        {
            cudaGetLastError();
        }
    }
    else if (cs->graph_valid[s])
    {
        cudaGraphLaunch(cs->graph_exec[s], cs->stream);
    }
    else
    {
        pyrwfs_cuda_launch_optics(ctx, cs, d_pup_ptr, d_amp_ptr, d_wf_ptr, d_dm_ptr, wf_to_opd, s);
    }

    if (ctx->subframe_cnt >= 1u)
    {
        uint32_t th_cam = 256u;
        uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
        kernel_pyrwfs_add_accum<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_cam_int, ctx->npix_cam, cs->d_accum_int);
    }
}

extern "C" void cacaosim_pyrwfs_cuda_accum_poly(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *beam_lam_um,
    uint32_t               beam_nlam,
    uint64_t               chan_stride,
    const float           *d_beam_opd,
    const float           *d_beam_amp)
{
    pyrwfs_cuda_state_t *cs       = (pyrwfs_cuda_state_t *) ctx->cuda_priv;
    int32_t              nlam     = (ctx->eff_nlam >= 1) ? ctx->eff_nlam : 1;
    size_t               wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    float                lam0     = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 0.75f;
    float                bw       = (ctx->cfg.bw_frac > 0.0f) ? ctx->cfg.bw_frac : 0.20f;
    float                def_l0   = lam0 * (1.0f - 0.5f * bw);
    float                def_l1   = lam0 * (1.0f + 0.5f * bw);

    pyrwfs_poly_map_t pmap = { { 0 }, { 0 }, { 0.0f } };
    for (int32_t c = 0; c < nlam; c++)
    {
        cacaosim_poly_interp_bracket(ctx->lam_um[c], beam_lam_um, beam_nlam, def_l0, def_l1,
                                     &pmap.m0[c], &pmap.m1[c], &pmap.alpha[c]);
    }

    const float *d_opd_src  = d_beam_opd;
    const float *d_amp_src  = d_beam_amp;
    uint64_t     eff_stride = chan_stride;

    if (d_opd_src == NULL && beam_opd != NULL)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            uint32_t m0 = pmap.m0[c];
            uint32_t m1 = pmap.m1[c];
            cudaMemcpyAsync(cs->d_poly_opd + (size_t) (2 * c) * ctx->npix_wf,
                            beam_opd + (size_t) m0 * chan_stride, wf_bytes,
                            cudaMemcpyHostToDevice, cs->stream);
            if (m1 != m0)
            {
                cudaMemcpyAsync(cs->d_poly_opd + (size_t) (2 * c + 1) * ctx->npix_wf,
                                beam_opd + (size_t) m1 * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
        }
        d_opd_src = cs->d_poly_opd;
    }
    if (d_amp_src == NULL && beam_amp != NULL)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            uint32_t m0 = pmap.m0[c];
            uint32_t m1 = pmap.m1[c];
            cudaMemcpyAsync(cs->d_poly_amp + (size_t) (2 * c) * ctx->npix_wf,
                            beam_amp + (size_t) m0 * chan_stride, wf_bytes,
                            cudaMemcpyHostToDevice, cs->stream);
            if (m1 != m0)
            {
                cudaMemcpyAsync(cs->d_poly_amp + (size_t) (2 * c + 1) * ctx->npix_wf,
                                beam_amp + (size_t) m1 * chan_stride, wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
        }
        d_amp_src = cs->d_poly_amp;
    }
    if (d_beam_opd == NULL && d_beam_amp == NULL)
    {
        for (int32_t c = 0; c < nlam; c++)
        {
            uint32_t orig_m0 = pmap.m0[c];
            uint32_t orig_m1 = pmap.m1[c];
            pmap.m0[c]       = (uint32_t) (2 * c);
            pmap.m1[c]       = (orig_m1 != orig_m0) ? (uint32_t) (2 * c + 1) : (uint32_t) (2 * c);
        }
        eff_stride = ctx->npix_wf;
    }

    if (ctx->subframe_cnt == 1u)
    {
        cudaMemcpyAsync(cs->d_accum_int, cs->d_cam_int, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs->stream);
    }

    uint32_t nfft    = ctx->nfft;
    uint32_t nwf     = ctx->cfg.nwf;
    uint32_t pad_off = (nfft >= nwf) ? ((nfft - nwf) / 2u) : 0u;
    uint32_t th_fft  = 256u;
    uint32_t bl_fft  = (uint32_t) ((ctx->npix_fft + th_fft - 1ULL) / th_fft);
    int32_t  s       = (ctx->n_slices > 1) ? (ctx->slice_cursor % ctx->n_slices) : 0;

    kernel_pyrwfs_build_pupil_poly<<<bl_fft, th_fft, 0, cs->stream>>>(
        d_opd_src, d_amp_src, eff_stride, cs->d_ncpa_opd, cs->d_k_rad, pmap, nwf, nfft, pad_off,
        nlam, cs->d_pup_cplx);

    pyrwfs_cuda_run_fft_and_bin(ctx, cs, s);

    if (ctx->subframe_cnt >= 1u)
    {
        uint32_t th_cam = 256u;
        uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
        kernel_pyrwfs_add_accum<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_cam_int, ctx->npix_cam, cs->d_accum_int);
    }
}

/**
 * cacaosim_pyrwfs_cuda_readout() - Normalize GPU exposure, apply detector noise, and copy to host
 * @ctx:      Active Pyramid WFS context (`use_cuda == 1`)
 * @det_live: Active detector readout parameters (non-NULL)
 * @out_u16:  Destination host `uint16_t` camera buffer `[npix_cam]` when `out_dtype == 0`
 * @out_f32:  Destination host `float` camera buffer `[npix_cam]` when `out_dtype != 0`
 *
 * Role:
 *   Finalizes a GPU WFS camera exposure by averaging `subframe_cnt` accumulated subframes,
 *   synthesizing noisy ADU pixels on-device, and synchronizing the D2H copy into SHM/host memory.
 *
 * Rationale:
 *   Synchronizing `cs->stream` only once at camera readout allows multiple `navg` subframes to
 *   pipeline asynchronously back-to-back on the GPU without host stalls between subframes.
 */
extern "C" void cacaosim_pyrwfs_cuda_readout(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    pyrwfs_cuda_state_t *cs = (pyrwfs_cuda_state_t *) ctx->cuda_priv;

    if (ctx->subframe_cnt > 1u)
    {
        uint32_t th_cam = 256u;
        uint32_t bl_cam = (uint32_t) ((ctx->npix_cam + th_cam - 1ULL) / th_cam);
        kernel_pyrwfs_scale_accum<<<bl_cam, th_cam, 0, cs->stream>>>(
            cs->d_accum_int, 1.0f / (float) ctx->subframe_cnt, ctx->npix_cam, cs->d_cam_int);
    }

    cacaosim_cuda_detector_readout(det_live, cs->d_cam_int, ctx->npix_cam, ctx->flat_flux_norm,
                                   ctx->cfg.rng_seed, ctx->frame_seq, cs->d_out_u16, cs->d_out_f32,
                                   cs->stream);

    if (det_live->out_dtype == 0 && out_u16 != NULL)
    {
        cudaMemcpyAsync(out_u16, cs->d_out_u16, (size_t) ctx->npix_cam * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, cs->stream);
    }
    else if (out_f32 != NULL)
    {
        cudaMemcpyAsync(out_f32, cs->d_out_f32, (size_t) ctx->npix_cam * sizeof(float),
                        cudaMemcpyDeviceToHost, cs->stream);
    }
    cudaStreamSynchronize(cs->stream);
}
