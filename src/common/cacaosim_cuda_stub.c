// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_cuda_stub.c
 * @brief   Link-time no-op fallback stubs for all CUDA entry points when built without `HAVE_CUDA`
 *
 * Architectural Role:
 *   Provides pure-C fallback definitions for every symbol normally compiled from `.cu` translation
 *   units (`cacaosim_cuda.cu`, `cacaosim_dm_cuda.cu`, `cacaosim_pyrwfs_cuda.cu`,
 *   `cacaosim_fpcam_cuda.cu`, `cacaosim_shwfs_cuda.cu`, `cacaosim_zwfs_cuda.cu`, and
 *   `cacaosim_coro_cuda.cu`) whenever `cacaosim` is configured with `-DCACAOSIM_ENABLE_CUDA=OFF`
 *   or built on a host without a CUDA Toolkit (`#ifndef HAVE_CUDA`).
 *
 * Design Rationale & Motivation:
 *   1. Eliminates `#ifdef HAVE_CUDA` preprocessor sprawl across physics engines: Every `.c` module
 *      (`dmsim`, `pyrwfs`, `fpcam`, `shwfs`, `zwfs`, `coro`, `mkbeam`, `wfe`, `beamsplit`, `htsim`)
 *      compiles the exact same backend-dispatch logic regardless of whether CUDA is enabled at
 *      CMake configure time, preventing dead-code bit-rot between CPU-only and GPU builds.
 *   2. Fail-safe CPU fallback by construction: `cacaosim_cuda_available()` returns `0`, stream and
 *      VRAM allocators return `NULL`, and every `cacaosim_<unit>_cuda_init()` / `alloc()` returns
 *      `-1`. Consequently, every context leaves `ctx->use_cuda = 0` at initialization and routes
 *      all hot-loop frames exclusively through the OpenMP + SIMD + FFTW3f / OpenBLAS CPU backend,
 *      while the `void` per-frame stubs satisfy the linker cleanly with zero `-Wunused-parameter`
 *      warnings.
 */

#include <stddef.h>

#include "common/cacaosim_cuda.h"
#include "common/cacaosim_wfstat_core.h"
#include "coro/cacaosim_coro.h"
#include "dm/cacaosim_dm.h"
#include "fpcam/cacaosim_fpcam.h"
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "shwfs/cacaosim_shwfs.h"
#include "zwfs/cacaosim_zwfs.h"

#ifndef HAVE_CUDA

/**
 * cacaosim_cuda_available() - Report that CUDA GPU acceleration is absent in CPU-only builds
 *
 * Role:
 *   CPU-only stub for `cacaosim_cuda.h` capability query.
 *
 * Rationale:
 *   Returning `0` causes all `CACAOSIM_BACKEND_AUTO` and `CACAOSIM_BACKEND_CUDA` initializers to
 *   select the CPU execution path immediately without attempting device allocations.
 *
 * Return: Always `0`.
 */
int cacaosim_cuda_available(void)
{
    return 0;
}

/**
 * cacaosim_cuda_stream_create() - Stub returning `NULL` when CUDA streams are unavailable
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_stream_create()` link references.
 *
 * Rationale:
 *   Callers treat a `NULL` stream return alongside `cacaosim_cuda_available() == 0` as CPU mode.
 *
 * Return: Always `NULL`.
 */
void *cacaosim_cuda_stream_create(void)
{
    return NULL;
}

/**
 * cacaosim_cuda_stream_destroy() - No-op CUDA stream destructor stub for CPU-only builds
 * @cuda_stream: Unused stream pointer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_stream_destroy()` teardown calls.
 *
 * Rationale:
 *   Allows unconditional stream cleanup in `free()` routines without `#ifdef HAVE_CUDA` guards.
 */
void cacaosim_cuda_stream_destroy(void *cuda_stream)
{
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_stream_sync() - No-op CUDA stream barrier stub for CPU-only builds
 * @cuda_stream: Unused stream pointer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_stream_sync()` link references.
 *
 * Rationale:
 *   Host execution is already synchronous when `use_cuda == 0`.
 */
void cacaosim_cuda_stream_sync(void *cuda_stream)
{
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_alloc_f32() - Stub returning `NULL` for device float32 allocations without CUDA
 * @nelem: Unused requested element count
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_alloc_f32()` link references.
 *
 * Rationale:
 *   Returning `NULL` cleanly signals unavailability of device memory to any caller.
 *
 * Return: Always `NULL`.
 */
float *cacaosim_cuda_alloc_f32(uint64_t nelem)
{
    (void) nelem;
    return NULL;
}

/**
 * cacaosim_cuda_free_f32() - No-op device float32 deallocator stub for CPU-only builds
 * @d_ptr: Unused device buffer pointer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_free_f32()` in shared cleanup paths.
 *
 * Rationale:
 *   Permits unconditional device-pointer cleanup during module teardown.
 */
void cacaosim_cuda_free_f32(float *d_ptr)
{
    (void) d_ptr;
}

/**
 * cacaosim_cuda_h2d_f32() - No-op Host-to-Device float32 copy stub for CPU-only builds
 * @d_dst:       Unused destination device pointer
 * @h_src:       Unused source host pointer
 * @nelem:       Unused element count
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_h2d_f32()` link references.
 *
 * Rationale:
 *   Unreached at runtime because `use_cuda == 0`, kept to satisfy link-time symbol resolution.
 */
void cacaosim_cuda_h2d_f32(
    float       *d_dst,
    const float *h_src,
    uint64_t     nelem,
    void        *cuda_stream)
{
    (void) d_dst;
    (void) h_src;
    (void) nelem;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_d2h_f32() - No-op Device-to-Host float32 copy stub for CPU-only builds
 * @h_dst:       Unused destination host pointer
 * @d_src:       Unused source device pointer
 * @nelem:       Unused element count
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_d2h_f32()` link references.
 *
 * Rationale:
 *   Unreached at runtime because `use_cuda == 0`, kept to satisfy link-time symbol resolution.
 */
void cacaosim_cuda_d2h_f32(
    float       *h_dst,
    const float *d_src,
    uint64_t     nelem,
    void        *cuda_stream)
{
    (void) h_dst;
    (void) d_src;
    (void) nelem;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_build_pupil_padded() - No-op complex pupil assembly stub for CPU-only builds
 * @d_pup_amp:   Unused device pupil amplitude pointer
 * @d_atm_amp:   Unused device atmospheric amplitude pointer
 * @d_wf_in:     Unused device wavefront pointer
 * @d_dm_opd:    Unused device DM OPD pointer
 * @d_ncpa_opd:  Unused device NCPA OPD pointer
 * @wf_to_opd:   Unused wavefront-to-OPD scale
 * @opd_to_rad:  Unused OPD-to-phase scale
 * @nwf:         Unused pupil grid width
 * @nfft:        Unused FFT grid width
 * @d_out_cplx:  Unused complex output buffer
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_build_pupil_padded()` link references.
 *
 * Rationale:
 *   Avoids `#ifdef HAVE_CUDA` around shared pupil helper declarations when linking without `nvcc`.
 */
void cacaosim_cuda_build_pupil_padded(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_dm_opd,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    float        opd_to_rad,
    uint32_t     nwf,
    uint32_t     nfft,
    void        *d_out_cplx,
    void        *cuda_stream)
{
    (void) d_pup_amp;
    (void) d_atm_amp;
    (void) d_wf_in;
    (void) d_dm_opd;
    (void) d_ncpa_opd;
    (void) wf_to_opd;
    (void) opd_to_rad;
    (void) nwf;
    (void) nfft;
    (void) d_out_cplx;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_detector_readout() - No-op GPU detector readout stub for CPU-only builds
 * @cfg:         Unused detector configuration
 * @d_opt_int:   Unused device optical intensity pointer
 * @npix:        Unused pixel count
 * @norm_total:  Unused flux normalization
 * @seed:        Unused PRNG seed
 * @frame_seq:   Unused frame sequence counter
 * @d_out_u16:   Unused uint16 device output pointer
 * @d_out_f32:   Unused float32 device output pointer
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_cuda_detector_readout()` link references; sensors use
 *   `cacaosim_detector_readout_cpu()` instead.
 *
 * Rationale:
 *   Maintains link-time parity with `cacaosim_cuda.cu` when CUDA compilation is disabled.
 */
void cacaosim_cuda_detector_readout(
    const cacaosim_det_cfg_t *cfg,
    const float              *d_opt_int,
    uint64_t                  npix,
    float                     norm_total,
    uint64_t                  seed,
    uint64_t                  frame_seq,
    uint16_t                 *d_out_u16,
    float                    *d_out_f32,
    void                     *cuda_stream)
{
    (void) cfg;
    (void) d_opt_int;
    (void) npix;
    (void) norm_total;
    (void) seed;
    (void) frame_seq;
    (void) d_out_u16;
    (void) d_out_f32;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_beam_init() - No-op GPU beam synthesis stub for CPU-only builds
 * @d_pup_amp:   Unused device pupil amplitude pointer
 * @d_atm_amp:   Unused device scintillation amplitude pointer
 * @d_wf_in:     Unused device wavefront pointer
 * @d_ncpa_opd:  Unused device NCPA OPD pointer
 * @wf_to_opd:   Unused wavefront-to-OPD scale
 * @nwf:         Unused pupil grid width
 * @d_out_opd:   Unused device output OPD pointer
 * @d_out_amp:   Unused device output amplitude pointer
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub for `cacaosim_cuda_beam_init()`; CPU builds execute `cacaosim_beam_init_cpu()`.
 *
 * Rationale:
 *   Lets `mkbeam` and `htsim` compile identical C dispatch code without `#ifdef HAVE_CUDA`.
 */
void cacaosim_cuda_beam_init(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream)
{
    (void) d_pup_amp;
    (void) d_atm_amp;
    (void) d_wf_in;
    (void) d_ncpa_opd;
    (void) wf_to_opd;
    (void) nwf;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_beam_init_lerp() - No-op GPU linear-lerp beam synthesis stub for CPU-only builds
 * @d_pup_amp:   Unused device pupil amplitude pointer
 * @d_atm_amp:   Unused device scintillation amplitude pointer
 * @d_wf_prev:   Unused device previous wavefront pointer
 * @d_wf_curr:   Unused device current wavefront pointer
 * @wf_alpha:    Unused blend weight
 * @d_ncpa_opd:  Unused device NCPA OPD pointer
 * @wf_to_opd:   Unused wavefront-to-OPD scale
 * @nwf:         Unused pupil grid width
 * @d_out_opd:   Unused device output OPD pointer
 * @d_out_amp:   Unused device output amplitude pointer
 * @cuda_stream: Unused CUDA stream handle
 */
void cacaosim_cuda_beam_init_lerp(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_prev,
    const float *d_wf_curr,
    float        wf_alpha,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream)
{
    (void) d_pup_amp;
    (void) d_atm_amp;
    (void) d_wf_prev;
    (void) d_wf_curr;
    (void) wf_alpha;
    (void) d_ncpa_opd;
    (void) wf_to_opd;
    (void) nwf;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_beam_apply_dm() - No-op GPU DM subtraction stub for CPU-only builds
 * @d_in_opd:    Unused device input OPD pointer
 * @d_in_amp:    Unused device input amplitude pointer
 * @d_dm_opd:    Unused device DM OPD pointer
 * @nwf:         Unused pupil grid width
 * @d_out_opd:   Unused device output OPD pointer
 * @d_out_amp:   Unused device output amplitude pointer
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub for `cacaosim_cuda_beam_apply_dm()`; CPU builds run `cacaosim_beam_apply_dm_cpu`.
 *
 * Rationale:
 *   Resolves GPU DM-beam symbols in `dmsim` and `htsim` when built without CUDA.
 */
void cacaosim_cuda_beam_apply_dm(
    const float *d_in_opd,
    const float *d_in_amp,
    const float *d_dm_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream)
{
    (void) d_in_opd;
    (void) d_in_amp;
    (void) d_dm_opd;
    (void) nwf;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) cuda_stream;
}

/**
 * cacaosim_cuda_beam_relay() - No-op GPU beam relay stub for CPU-only builds
 * @d_in_opd:    Unused device input OPD pointer
 * @d_in_amp:    Unused device input amplitude pointer
 * @d_ncpa_opd:  Unused device arm NCPA pointer
 * @d_pup_mask:  Unused device arm pupil mask pointer
 * @reg:         Unused registration descriptor
 * @nwf:         Unused pupil grid width
 * @d_out_opd:   Unused device output OPD pointer
 * @d_out_amp:   Unused device output amplitude pointer
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub for `cacaosim_cuda_beam_relay()`; CPU builds run `cacaosim_beam_relay_cpu()`.
 *
 * Rationale:
 *   Resolves GPU relay symbols in `beamsplit` and `htsim` without conditional compilation.
 */
void cacaosim_cuda_beam_relay(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_ncpa_opd,
    const float          *d_pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *d_out_opd,
    float                *d_out_amp,
    void                 *cuda_stream)
{
    (void) d_in_opd;
    (void) d_in_amp;
    (void) d_ncpa_opd;
    (void) d_pup_mask;
    (void) reg;
    (void) nwf;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) cuda_stream;
}

/**
 * cacaosim_dm_cuda_alloc() - Fail-safe DM CUDA allocator stub returning `-1` without CUDA
 * @ctx:        Unused DM context
 * @ext_stream: Unused external CUDA stream handle
 *
 * Role:
 *   CPU-only stub for `cacaosim_dm_cuda.cu` initialization called from `cacaosim_dm_init()`.
 *
 * Rationale:
 *   Returning `-1` instructs `cacaosim_dm_init()` to keep `ctx->use_cuda = 0` and execute the
 *   16-tap gather stencil on CPU (`cacaosim_dm_render_opd_cpu`).
 *
 * Return: Always `-1`.
 */
int cacaosim_dm_cuda_alloc(cacaosim_dm_ctx_t *ctx, void *ext_stream)
{
    (void) ctx;
    (void) ext_stream;
    return -1;
}

/**
 * cacaosim_dm_cuda_free() - No-op DM CUDA destructor stub for CPU-only builds
 * @ctx: Unused DM context
 *
 * Role:
 *   CPU-only stub called unconditionally by `cacaosim_dm_free()`.
 *
 * Rationale:
 *   Keeps `cacaosim_dm.c` teardown free of `#ifdef HAVE_CUDA` directives.
 */
void cacaosim_dm_cuda_free(cacaosim_dm_ctx_t *ctx)
{
    (void) ctx;
}

/**
 * cacaosim_dm_cuda_render() - No-op DM CUDA gather-stencil OPD renderer stub for CPU-only builds
 * @ctx:          Unused DM context
 * @host_opd_out: Unused host destination OPD buffer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_dm_step()` link references when `use_cuda == 0`.
 *
 * Rationale:
 *   Unreached at runtime because `cacaosim_dm_cuda_alloc()` returns `-1`.
 */
void cacaosim_dm_cuda_render(cacaosim_dm_ctx_t *ctx, float *host_opd_out)
{
    (void) ctx;
    (void) host_opd_out;
}

/**
 * cacaosim_pyrwfs_cuda_init() - Fail-safe Pyramid WFS CUDA init stub returning `-1` without CUDA
 * @ctx:        Unused Pyramid WFS context
 * @pup_amp:    Unused static pupil amplitude
 * @ncpa_opd:   Unused static NCPA OPD
 * @ext_stream: Unused external CUDA stream
 *
 * Role:
 *   CPU-only stub for `cacaosim_pyrwfs_cuda.cu` initialization called from `cacaosim_pyrwfs_init`.
 *
 * Rationale:
 *   Returning `-1` triggers automatic fallback to the OpenMP L2-resident FFTW3f CPU backend.
 *
 * Return: Always `-1`.
 */
int cacaosim_pyrwfs_cuda_init(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *ncpa_opd,
    void                  *ext_stream)
{
    (void) ctx;
    (void) pup_amp;
    (void) ncpa_opd;
    (void) ext_stream;
    return -1;
}

/**
 * cacaosim_pyrwfs_cuda_upload_masks() - No-op Pyramid WFS GPU mask-upload stub for CPU-only builds
 * @ctx: Unused Pyramid WFS context
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_pyrwfs_update_masks()` when modulation radius changes live.
 *
 * Rationale:
 *   Host `ctx->masks_cplx` is already updated in place for FFTW3f; no VRAM upload is needed.
 */
void cacaosim_pyrwfs_cuda_upload_masks(cacaosim_pyrwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_pyrwfs_cuda_upload_weights(cacaosim_pyrwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_pyrwfs_cuda_accum_poly(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *beam_lam_um,
    uint32_t               beam_nlam,
    uint64_t               chan_stride,
    const float           *d_beam_opd,
    const float           *d_beam_amp)
{
    (void) ctx;
    (void) beam_opd;
    (void) beam_amp;
    (void) beam_lam_um;
    (void) beam_nlam;
    (void) chan_stride;
    (void) d_beam_opd;
    (void) d_beam_amp;
}

void cacaosim_pyrwfs_cuda_accum_full(
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
    (void) ctx;
    (void) dyn_pup_amp;
    (void) atm_amp;
    (void) wf_in;
    (void) dm_opd;
    (void) d_pup_amp_ext;
    (void) d_wf_opd_ext;
    (void) d_dm_opd_ext;
    (void) wf_to_opd;
}

void cacaosim_pyrwfs_cuda_accum_poly_full(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *beam_lam_um,
    uint32_t               beam_nlam,
    uint64_t               chan_stride,
    const float           *d_beam_opd,
    const float           *d_beam_amp)
{
    (void) ctx;
    (void) beam_opd;
    (void) beam_amp;
    (void) beam_lam_um;
    (void) beam_nlam;
    (void) chan_stride;
    (void) d_beam_opd;
    (void) d_beam_amp;
}

/**
 * cacaosim_pyrwfs_cuda_free() - No-op Pyramid WFS CUDA destructor stub for CPU-only builds
 * @ctx: Unused Pyramid WFS context
 *
 * Role:
 *   CPU-only stub called by `cacaosim_pyrwfs_free()`.
 *
 * Rationale:
 *   Satisfies link-time teardown references without `#ifdef` guards in `cacaosim_pyrwfs.c`.
 */
void cacaosim_pyrwfs_cuda_free(cacaosim_pyrwfs_ctx_t *ctx)
{
    (void) ctx;
}

/**
 * cacaosim_pyrwfs_cuda_accum() - No-op Pyramid WFS CUDA subframe accumulator stub
 * @ctx:           Unused Pyramid WFS context
 * @dyn_pup_amp:   Unused dynamic pupil amplitude
 * @atm_amp:       Unused scintillation amplitude
 * @wf_in:         Unused input wavefront
 * @dm_opd:        Unused host DM OPD
 * @d_pup_amp_ext: Unused external device pupil amplitude
 * @d_wf_opd_ext:  Unused external device wavefront OPD
 * @d_dm_opd_ext:  Unused external device DM OPD
 * @wf_to_opd:     Unused wavefront-to-OPD scale
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_pyrwfs_accum_*()` GPU branch references.
 *
 * Rationale:
 *   Never invoked at runtime because `cacaosim_pyrwfs_cuda_init()` forces `ctx->use_cuda = 0`.
 */
void cacaosim_pyrwfs_cuda_accum(
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
    (void) ctx;
    (void) dyn_pup_amp;
    (void) atm_amp;
    (void) wf_in;
    (void) dm_opd;
    (void) d_pup_amp_ext;
    (void) d_wf_opd_ext;
    (void) d_dm_opd_ext;
    (void) wf_to_opd;
}

/**
 * cacaosim_pyrwfs_cuda_readout() - No-op Pyramid WFS CUDA camera readout stub for CPU-only builds
 * @ctx:      Unused Pyramid WFS context
 * @det_live: Unused live detector parameters
 * @out_u16:  Unused uint16 destination buffer
 * @out_f32:  Unused float32 destination buffer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_pyrwfs_readout()` GPU branch references.
 *
 * Rationale:
 *   Unreached when `ctx->use_cuda == 0`; preserves `#ifdef`-free WFS readout dispatch.
 */
void cacaosim_pyrwfs_cuda_readout(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    (void) ctx;
    (void) det_live;
    (void) out_u16;
    (void) out_f32;
}

void cacaosim_pyrwfs_cuda_readout_ex(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    int                       sync_stream)
{
    (void) ctx;
    (void) det_live;
    (void) out_u16;
    (void) out_f32;
    (void) sync_stream;
}

/**
 * cacaosim_fpcam_cuda_init() - Fail-safe focal-plane camera CUDA init stub returning `-1`
 * @ctx:        Unused focal-plane camera context
 * @pup_amp:    Unused static pupil amplitude
 * @ncpa_opd:   Unused static NCPA OPD
 * @ext_stream: Unused external CUDA stream
 *
 * Role:
 *   CPU-only stub for `cacaosim_fpcam_cuda.cu` initialization called from `cacaosim_fpcam_init()`.
 *
 * Rationale:
 *   Returning `-1` selects the OpenBLAS `cblas_cgemm` MFT / FFTW3f CPU camera backend cleanly.
 *
 * Return: Always `-1`.
 */
int cacaosim_fpcam_cuda_init(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *ncpa_opd,
    void                 *ext_stream)
{
    (void) ctx;
    (void) pup_amp;
    (void) ncpa_opd;
    (void) ext_stream;
    return -1;
}

/**
 * cacaosim_fpcam_cuda_free() - No-op focal-plane camera CUDA destructor stub for CPU-only builds
 * @ctx: Unused focal-plane camera context
 *
 * Role:
 *   CPU-only stub called by `cacaosim_fpcam_free()`.
 *
 * Rationale:
 *   Resolves GPU teardown symbols without `#ifdef HAVE_CUDA` in `cacaosim_fpcam.c`.
 */
void cacaosim_fpcam_cuda_free(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_fpcam_cuda_upload_weights(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

/**
 * cacaosim_fpcam_cuda_reset_accum() - No-op focal-plane camera GPU accumulator reset stub
 * @ctx: Unused focal-plane camera context
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_fpcam_cuda_reset_accum()` link references.
 *
 * Rationale:
 *   Unreached when `ctx->use_cuda == 0`.
 */
void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_fpcam_cuda_reset_long(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

/**
 * cacaosim_fpcam_cuda_accum() - No-op focal-plane camera CUDA monochromatic subframe stub
 * @ctx:           Unused focal-plane camera context
 * @dyn_pup_amp:   Unused dynamic pupil amplitude
 * @atm_amp:       Unused scintillation amplitude
 * @wf_in:         Unused input wavefront
 * @dm_opd:        Unused host DM OPD
 * @d_pup_amp_ext: Unused external device pupil amplitude
 * @d_wf_opd_ext:  Unused external device wavefront OPD
 * @d_dm_opd_ext:  Unused external device DM OPD
 * @wf_to_opd:     Unused wavefront-to-OPD scale
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_fpcam_accum_*()` GPU branch references.
 *
 * Rationale:
 *   Unreached at runtime because `cacaosim_fpcam_cuda_init()` forces `ctx->use_cuda = 0`.
 */
void cacaosim_fpcam_cuda_accum(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *dyn_pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *d_pup_amp_ext,
    const float          *d_wf_opd_ext,
    const float          *d_dm_opd_ext,
    float                 wf_to_opd)
{
    (void) ctx;
    (void) dyn_pup_amp;
    (void) atm_amp;
    (void) wf_in;
    (void) dm_opd;
    (void) d_pup_amp_ext;
    (void) d_wf_opd_ext;
    (void) d_dm_opd_ext;
    (void) wf_to_opd;
}

/**
 * cacaosim_fpcam_cuda_accum_poly() - No-op focal-plane camera CUDA polychromatic cube stub
 * @ctx:         Unused focal-plane camera context
 * @beam_opd:    Unused host multi-wavelength OPD cube
 * @beam_amp:    Unused host multi-wavelength amplitude cube
 * @beam_nlam:   Unused number of input wavelength channels
 * @chan_stride: Unused element stride between channels
 * @d_beam_opd:  Unused device multi-wavelength OPD cube
 * @d_beam_amp:  Unused device multi-wavelength amplitude cube
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_fpcam_accum_beam_poly()` GPU branch references.
 *
 * Rationale:
 *   Unreached when `ctx->use_cuda == 0`; polychromatic propagation runs via OpenBLAS/FFTW3f on CPU.
 */
void cacaosim_fpcam_cuda_accum_poly(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    (void) ctx;
    (void) beam_opd;
    (void) beam_amp;
    (void) beam_nlam;
    (void) chan_stride;
    (void) d_beam_opd;
    (void) d_beam_amp;
}

/**
 * cacaosim_fpcam_cuda_readout() - No-op focal-plane camera CUDA readout stub for CPU-only builds
 * @ctx:          Unused focal-plane camera context
 * @det_live:     Unused live detector configuration
 * @host_exp_int: Unused host exposure intensity buffer for Strehl/long-exposure telemetry
 * @out_u16:      Unused uint16 destination buffer
 * @out_f32:      Unused float32 destination buffer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_fpcam_readout()` GPU branch references.
 *
 * Rationale:
 *   Unreached when `ctx->use_cuda == 0`.
 */
void cacaosim_fpcam_cuda_readout(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    float                    *host_exp_int,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    (void) ctx;
    (void) det_live;
    (void) host_exp_int;
    (void) out_u16;
    (void) out_f32;
}

void cacaosim_fpcam_cuda_readout_full(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long,
    int                       sync_stream)
{
    (void) ctx;
    (void) det_live;
    (void) out_u16;
    (void) out_f32;
    (void) out_long;
    (void) sync_stream;
}

void cacaosim_fpcam_cuda_readout_finish(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

/**
 * cacaosim_shwfs_cuda_init() - Fail-safe Shack-Hartmann WFS CUDA init stub returning `-1`
 * @ctx:        Unused Shack-Hartmann WFS context
 * @pup_amp:    Unused static pupil amplitude
 * @ncpa_opd:   Unused static NCPA OPD
 * @ext_stream: Unused external CUDA stream
 *
 * Role:
 *   CPU-only stub for `cacaosim_shwfs_cuda.cu` initialization called from `cacaosim_shwfs_init()`.
 *
 * Rationale:
 *   Returning `-1` selects the OpenMP per-thread L1-resident FFTW3f CPU lenslet propagator.
 *
 * Return: Always `-1`.
 */
int cacaosim_shwfs_cuda_init(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *pup_amp,
    const float          *ncpa_opd,
    void                 *ext_stream)
{
    (void) ctx;
    (void) pup_amp;
    (void) ncpa_opd;
    (void) ext_stream;
    return -1;
}

/**
 * cacaosim_shwfs_cuda_free() - No-op Shack-Hartmann WFS CUDA destructor stub for CPU-only builds
 * @ctx: Unused Shack-Hartmann WFS context
 *
 * Role:
 *   CPU-only stub called by `cacaosim_shwfs_free()`.
 *
 * Rationale:
 *   Resolves GPU teardown symbols without `#ifdef HAVE_CUDA` in `cacaosim_shwfs.c`.
 */
void cacaosim_shwfs_cuda_free(cacaosim_shwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_shwfs_cuda_upload_weights(cacaosim_shwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_shwfs_cuda_accum_poly(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *beam_lam_um,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    (void) ctx;
    (void) beam_opd;
    (void) beam_amp;
    (void) beam_lam_um;
    (void) beam_nlam;
    (void) chan_stride;
    (void) d_beam_opd;
    (void) d_beam_amp;
}

/**
 * cacaosim_shwfs_cuda_accum() - No-op Shack-Hartmann WFS CUDA subframe accumulator stub
 * @ctx:           Unused Shack-Hartmann WFS context
 * @dyn_pup_amp:   Unused dynamic pupil amplitude
 * @atm_amp:       Unused scintillation amplitude
 * @wf_in:         Unused input wavefront
 * @dm_opd:        Unused host DM OPD
 * @d_pup_amp_ext: Unused external device pupil amplitude
 * @d_wf_opd_ext:  Unused external device wavefront OPD
 * @d_dm_opd_ext:  Unused external device DM OPD
 * @wf_to_opd:     Unused wavefront-to-OPD scale
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_shwfs_accum_*()` GPU branch references.
 *
 * Rationale:
 *   Unreached at runtime because `cacaosim_shwfs_cuda_init()` forces `ctx->use_cuda = 0`.
 */
void cacaosim_shwfs_cuda_accum(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *dyn_pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *d_pup_amp_ext,
    const float          *d_wf_opd_ext,
    const float          *d_dm_opd_ext,
    float                 wf_to_opd)
{
    (void) ctx;
    (void) dyn_pup_amp;
    (void) atm_amp;
    (void) wf_in;
    (void) dm_opd;
    (void) d_pup_amp_ext;
    (void) d_wf_opd_ext;
    (void) d_dm_opd_ext;
    (void) wf_to_opd;
}

/**
 * cacaosim_shwfs_cuda_readout() - No-op Shack-Hartmann WFS CUDA readout stub for CPU-only builds
 * @ctx:      Unused Shack-Hartmann WFS context
 * @det_live: Unused live detector configuration
 * @out_u16:  Unused uint16 destination buffer
 * @out_f32:  Unused float32 destination buffer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_shwfs_readout()` GPU branch references.
 *
 * Rationale:
 *   Unreached when `ctx->use_cuda == 0`.
 */
void cacaosim_shwfs_cuda_readout(
    cacaosim_shwfs_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    (void) ctx;
    (void) det_live;
    (void) out_u16;
    (void) out_f32;
}

/**
 * cacaosim_zwfs_cuda_init() - Fail-safe Zernike WFS CUDA init stub returning `-1` without CUDA
 * @ctx:        Unused Zernike WFS context
 * @pup_amp:    Unused static pupil amplitude
 * @ncpa_opd:   Unused static NCPA OPD
 * @ext_stream: Unused external CUDA stream
 *
 * Role:
 *   CPU-only stub for `cacaosim_zwfs_cuda.cu` initialization called from `cacaosim_zwfs_init()`.
 *
 * Rationale:
 *   Returning `-1` selects the OpenBLAS `cblas_cgemm` Babinet Matrix-DFT CPU backend cleanly.
 *
 * Return: Always `-1`.
 */
int cacaosim_zwfs_cuda_init(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *pup_amp,
    const float         *ncpa_opd,
    void                *ext_stream)
{
    (void) ctx;
    (void) pup_amp;
    (void) ncpa_opd;
    (void) ext_stream;
    return -1;
}

/**
 * cacaosim_zwfs_cuda_free() - No-op Zernike WFS CUDA destructor stub for CPU-only builds
 * @ctx: Unused Zernike WFS context
 *
 * Role:
 *   CPU-only stub called by `cacaosim_zwfs_free()`.
 *
 * Rationale:
 *   Resolves GPU teardown symbols without `#ifdef HAVE_CUDA` in `cacaosim_zwfs.c`.
 */
void cacaosim_zwfs_cuda_free(cacaosim_zwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_zwfs_cuda_upload_weights(cacaosim_zwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_zwfs_cuda_accum_poly(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *beam_opd,
    const float         *beam_amp,
    const float         *beam_lam_um,
    uint32_t             beam_nlam,
    uint64_t             chan_stride,
    const float         *d_beam_opd,
    const float         *d_beam_amp)
{
    (void) ctx;
    (void) beam_opd;
    (void) beam_amp;
    (void) beam_lam_um;
    (void) beam_nlam;
    (void) chan_stride;
    (void) d_beam_opd;
    (void) d_beam_amp;
}

/**
 * cacaosim_zwfs_cuda_accum() - No-op Zernike WFS CUDA subframe accumulator stub
 * @ctx:           Unused Zernike WFS context
 * @dyn_pup_amp:   Unused dynamic pupil amplitude
 * @atm_amp:       Unused scintillation amplitude
 * @wf_in:         Unused input wavefront
 * @dm_opd:        Unused host DM OPD
 * @d_pup_amp_ext: Unused external device pupil amplitude
 * @d_wf_opd_ext:  Unused external device wavefront OPD
 * @d_dm_opd_ext:  Unused external device DM OPD
 * @wf_to_opd:     Unused wavefront-to-OPD scale
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_zwfs_accum_*()` GPU branch references.
 *
 * Rationale:
 *   Unreached at runtime because `cacaosim_zwfs_cuda_init()` forces `ctx->use_cuda = 0`.
 */
void cacaosim_zwfs_cuda_accum(
    cacaosim_zwfs_ctx_t *ctx,
    const float         *dyn_pup_amp,
    const float         *atm_amp,
    const float         *wf_in,
    const float         *dm_opd,
    const float         *d_pup_amp_ext,
    const float         *d_wf_opd_ext,
    const float         *d_dm_opd_ext,
    float                wf_to_opd)
{
    (void) ctx;
    (void) dyn_pup_amp;
    (void) atm_amp;
    (void) wf_in;
    (void) dm_opd;
    (void) d_pup_amp_ext;
    (void) d_wf_opd_ext;
    (void) d_dm_opd_ext;
    (void) wf_to_opd;
}

/**
 * cacaosim_zwfs_cuda_readout() - No-op Zernike WFS CUDA camera readout stub for CPU-only builds
 * @ctx:      Unused Zernike WFS context
 * @det_live: Unused live detector configuration
 * @out_u16:  Unused uint16 destination buffer
 * @out_f32:  Unused float32 destination buffer
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_zwfs_readout()` GPU branch references.
 *
 * Rationale:
 *   Unreached when `ctx->use_cuda == 0`.
 */
void cacaosim_zwfs_cuda_readout(
    cacaosim_zwfs_ctx_t      *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32)
{
    (void) ctx;
    (void) det_live;
    (void) out_u16;
    (void) out_f32;
}

/**
 * cacaosim_coro_cuda_init() - Fail-safe coronagraph CUDA init stub returning `-1` without CUDA
 * @ctx:        Unused coronagraph context
 * @ext_stream: Unused external CUDA stream
 *
 * Role:
 *   CPU-only stub for `cacaosim_coro_cuda.cu` initialization called from
 *   `cacaosim_coro_init_masks()`.
 *
 * Rationale:
 *   Returning `-1` selects the OpenBLAS MFT / FFTW3f CPU coronagraph engine cleanly.
 *
 * Return: Always `-1`.
 */
int cacaosim_coro_cuda_init(cacaosim_coro_ctx_t *ctx, void *ext_stream)
{
    (void) ctx;
    (void) ext_stream;
    return -1;
}

/**
 * cacaosim_coro_cuda_free() - No-op coronagraph CUDA destructor stub for CPU-only builds
 * @ctx: Unused coronagraph context
 *
 * Role:
 *   CPU-only stub called by `cacaosim_coro_free()`.
 *
 * Rationale:
 *   Resolves GPU teardown symbols without `#ifdef HAVE_CUDA` in `cacaosim_coro.c`.
 */
void cacaosim_coro_cuda_free(cacaosim_coro_ctx_t *ctx)
{
    (void) ctx;
}

/**
 * cacaosim_coro_cuda_step_poly() - No-op coronagraph CUDA polychromatic propagation stub
 * @ctx:         Unused coronagraph context
 * @h_in_opd:    Unused host input OPD
 * @h_in_amp:    Unused host input amplitude
 * @in_nlam:     Unused input channel count
 * @in_stride:   Unused input channel stride
 * @h_out_opd:   Unused host post-Lyot output OPD
 * @h_out_amp:   Unused host post-Lyot output amplitude
 * @out_stride:  Unused output channel stride
 * @h_lowfs_opd: Unused host LOWFS pickoff OPD
 * @h_lowfs_amp: Unused host LOWFS pickoff amplitude
 * @d_in_opd:    Unused device input OPD
 * @d_in_amp:    Unused device input amplitude
 * @d_out_opd:   Unused device post-Lyot output OPD
 * @d_out_amp:   Unused device post-Lyot output amplitude
 * @d_lowfs_opd: Unused device LOWFS pickoff OPD
 * @d_lowfs_amp: Unused device LOWFS pickoff amplitude
 *
 * Role:
 *   CPU-only stub satisfying `cacaosim_coro_step_poly()` GPU branch references.
 *
 * Rationale:
 *   Unreached at runtime because `cacaosim_coro_cuda_init()` forces `ctx->use_cuda = 0` so all
 *   coronagraph steps dispatch to `cacaosim_coro_step_poly_cpu()`.
 */
void cacaosim_coro_cuda_step_poly(
    cacaosim_coro_ctx_t *ctx,
    const float         *h_in_opd,
    const float         *h_in_amp,
    uint32_t             in_nlam,
    uint64_t             in_stride,
    float               *h_out_opd,
    float               *h_out_amp,
    uint64_t             out_stride,
    float               *h_lowfs_opd,
    float               *h_lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp)
{
    (void) ctx;
    (void) h_in_opd;
    (void) h_in_amp;
    (void) in_nlam;
    (void) in_stride;
    (void) h_out_opd;
    (void) h_out_amp;
    (void) out_stride;
    (void) h_lowfs_opd;
    (void) h_lowfs_amp;
    (void) d_in_opd;
    (void) d_in_amp;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) d_lowfs_opd;
    (void) d_lowfs_amp;
}

/**
 * cacaosim_cuda_beam_apply_wfe() - No-op GPU static WFE injection stub for CPU-only builds
 * @d_in_opd:    Unused device input OPD
 * @d_in_amp:    Unused device input amplitude
 * @d_wfe_opd:   Unused device static WFE map
 * @wfe_to_um:   Unused WFE-to-microns scale
 * @reg:         Unused 2D registration parameters
 * @nwf:         Unused pupil grid width
 * @d_out_opd:   Unused device output OPD
 * @d_out_amp:   Unused device output amplitude
 * @cuda_stream: Unused CUDA stream handle
 *
 * Role:
 *   CPU-only stub for `cacaosim_cuda_beam_apply_wfe()`; CPU builds run
 *   `cacaosim_beam_apply_wfe_cpu()`.
 *
 * Rationale:
 *   Resolves GPU static-WFE symbols in `cacaosim.wfe` and `cacaosim.htsim` without `#ifdef` guards.
 */
void cacaosim_cuda_beam_apply_wfe(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_wfe_opd,
    float                 wfe_to_um,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *d_out_opd,
    float                *d_out_amp,
    void                 *cuda_stream)
{
    (void) d_in_opd;
    (void) d_in_amp;
    (void) d_wfe_opd;
    (void) wfe_to_um;
    (void) reg;
    (void) nwf;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) cuda_stream;
}

/* Wavefront statistics engine (cacaosim_wfstat_core): GPU path unavailable without CUDA */
void *cacaosim_wfstat_gpu_create(
    const double *h_weights,
    uint64_t      npix,
    uint32_t      nloc,
    uint64_t      ts_cap,
    void         *cuda_stream)
{
    (void) h_weights;
    (void) npix;
    (void) nloc;
    (void) ts_cap;
    (void) cuda_stream;
    return NULL;
}

void cacaosim_wfstat_gpu_free(void *gpu)
{
    (void) gpu;
}

void cacaosim_wfstat_gpu_accum(void *gpu, const float *const d_opd[])
{
    (void) gpu;
    (void) d_opd;
}

int cacaosim_wfstat_gpu_fetch_sample(void *gpu, float *const h_maps[], double *h_frame)
{
    (void) gpu;
    (void) h_maps;
    (void) h_frame;
    return -1;
}

int cacaosim_wfstat_gpu_fetch_sums(
    void         *gpu,
    double *const h_sum[],
    double *const h_sum2[],
    double       *h_acc,
    float *const  h_ts_rms[],
    uint64_t     *out_ts_len)
{
    (void) gpu;
    (void) h_sum;
    (void) h_sum2;
    (void) h_acc;
    (void) h_ts_rms;
    (void) out_ts_len;
    return -1;
}

#endif /* !HAVE_CUDA */
