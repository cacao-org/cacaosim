// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_cuda_stub.c
 * @brief   No-op stubs for CUDA entry points when built without CUDA support
 */

#include <stddef.h>

#include "common/cacaosim_cuda.h"
#include "coro/cacaosim_coro.h"
#include "dm/cacaosim_dm.h"
#include "fpcam/cacaosim_fpcam.h"
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "shwfs/cacaosim_shwfs.h"
#include "zwfs/cacaosim_zwfs.h"

#ifndef HAVE_CUDA

int cacaosim_cuda_available(void)
{
    return 0;
}

void *cacaosim_cuda_stream_create(void)
{
    return NULL;
}

void cacaosim_cuda_stream_destroy(void *cuda_stream)
{
    (void) cuda_stream;
}

void cacaosim_cuda_stream_sync(void *cuda_stream)
{
    (void) cuda_stream;
}

float *cacaosim_cuda_alloc_f32(uint64_t nelem)
{
    (void) nelem;
    return NULL;
}

void cacaosim_cuda_free_f32(float *d_ptr)
{
    (void) d_ptr;
}

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

int cacaosim_dm_cuda_alloc(cacaosim_dm_ctx_t *ctx, void *ext_stream)
{
    (void) ctx;
    (void) ext_stream;
    return -1;
}

void cacaosim_dm_cuda_free(cacaosim_dm_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_dm_cuda_render(cacaosim_dm_ctx_t *ctx, float *host_opd_out)
{
    (void) ctx;
    (void) host_opd_out;
}

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

void cacaosim_pyrwfs_cuda_upload_masks(cacaosim_pyrwfs_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_pyrwfs_cuda_free(cacaosim_pyrwfs_ctx_t *ctx)
{
    (void) ctx;
}

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

void cacaosim_fpcam_cuda_free(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx)
{
    (void) ctx;
}

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

void cacaosim_shwfs_cuda_free(cacaosim_shwfs_ctx_t *ctx)
{
    (void) ctx;
}

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

void cacaosim_zwfs_cuda_free(cacaosim_zwfs_ctx_t *ctx)
{
    (void) ctx;
}

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

int cacaosim_coro_cuda_init(cacaosim_coro_ctx_t *ctx, void *ext_stream)
{
    (void) ctx;
    (void) ext_stream;
    return -1;
}

void cacaosim_coro_cuda_free(cacaosim_coro_ctx_t *ctx)
{
    (void) ctx;
}

void cacaosim_coro_cuda_step(
    cacaosim_coro_ctx_t *ctx,
    const float         *h_in_opd,
    const float         *h_in_amp,
    float               *h_out_opd,
    float               *h_out_amp,
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
    (void) h_out_opd;
    (void) h_out_amp;
    (void) h_lowfs_opd;
    (void) h_lowfs_amp;
    (void) d_in_opd;
    (void) d_in_amp;
    (void) d_out_opd;
    (void) d_out_amp;
    (void) d_lowfs_opd;
    (void) d_lowfs_amp;
}

#endif /* !HAVE_CUDA */
