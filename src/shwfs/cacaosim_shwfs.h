// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs.h
 * @brief   Diffractive Shack-Hartmann WFS simulator (OpenMP L1 FFTW3f + batched cuFFT)
 */

#ifndef CACAOSIM_SHWFS_H
#define CACAOSIM_SHWFS_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optical and detector configuration for the Diffractive Shack-Hartmann WFS.
 */
typedef struct
{
    uint32_t           nwf;         /**< Input pupil grid dimension [pixels] */
    uint32_t           n_sub;       /**< Number of lenslets across pupil diameter (e.g. 16) */
    uint32_t           pix_per_sub; /**< Detector pixels per lenslet side (ncam = n_sub * P) */
    uint32_t           pad_factor;  /**< Subaperture FFT oversampling factor (>= 2, default: 2) */
    float              pup_diam;    /**< Clear pupil diameter fraction of nwf (default: 0.9375) */
    float              lambda_um;   /**< Sensing wavelength [microns] (default: 0.75) */
    int32_t            wf_unit;     /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda;  /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;    /**< Base PRNG seed for photon/readout noise */
    int32_t            backend;     /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;         /**< Integrated WFS camera readout parameters */
} cacaosim_shwfs_cfg_t;

/**
 * @brief Persistent CPU and CUDA state for the Diffractive Shack-Hartmann WFS.
 */
typedef struct
{
    cacaosim_shwfs_cfg_t cfg;
    uint32_t             ncam;
    uint32_t             w_sub_max;
    uint32_t             nfft_sub;
    uint32_t             nbatch;
    uint64_t             npix_wf;
    uint64_t             npix_sub_fft;
    uint64_t             npix_cam;
    float                flat_flux_norm;
    int32_t              use_cuda;
    uint32_t             subframe_cnt;
    uint64_t             frame_seq;

    /* Precomputed subaperture pupil bounds [n_sub] and half-bin centering phasor [nfft_sub * 2] */
    uint32_t            *sub_off;
    uint32_t            *sub_len;
    float               *shift_1d_cplx;

    /* CPU per-thread L1 FFT buffers and plans */
    int32_t              cpu_nth;
    float               *cpu_thr_cplx;
    void               **fftw_plans;
    float               *cpu_cam_int;
    float               *cpu_accum_int;
    cacaosim_det_pool_t  det_pool;

    /* Opaque CUDA state allocated by cacaosim_shwfs_cuda.cu */
    void                *cuda_priv;
} cacaosim_shwfs_ctx_t;

/**
 * cacaosim_shwfs_init() - Initialize Shack-Hartmann lenslet geometry, FFT plans, and flux norm
 */
int cacaosim_shwfs_init(
    cacaosim_shwfs_ctx_t       *ctx,
    const cacaosim_shwfs_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream);

/**
 * cacaosim_shwfs_free() - Release all host and device Shack-Hartmann WFS resources
 */
void cacaosim_shwfs_free(cacaosim_shwfs_ctx_t *ctx);

/**
 * cacaosim_shwfs_accum_subframe() - Propagate one wavefront+DM subframe into WFS accumulator
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
 */
void cacaosim_shwfs_accum_beam(
    cacaosim_shwfs_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_shwfs_readout() - Finalize accumulated SHWFS exposure and run detector readout
 */
void cacaosim_shwfs_readout(
    cacaosim_shwfs_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/**
 * cacaosim_shwfs_step() - Single-subframe SHWFS propagation and camera readout
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

/* Internal CUDA entry points implemented in cacaosim_shwfs_cuda.cu */
int  cacaosim_shwfs_cuda_init(cacaosim_shwfs_ctx_t *ctx,
                              const float          *pup_amp,
                              const float          *ncpa_opd,
                              void                 *ext_stream);
void cacaosim_shwfs_cuda_free(cacaosim_shwfs_ctx_t *ctx);
void cacaosim_shwfs_cuda_accum(cacaosim_shwfs_ctx_t *ctx,
                               const float          *dyn_pup_amp,
                               const float          *atm_amp,
                               const float          *wf_in,
                               const float          *dm_opd,
                               const float          *d_pup_amp_ext,
                               const float          *d_wf_opd_ext,
                               const float          *d_dm_opd_ext,
                               float                 wf_to_opd);
void cacaosim_shwfs_cuda_readout(cacaosim_shwfs_ctx_t     *ctx,
                                 const cacaosim_det_cfg_t *det_live,
                                 uint16_t                 *out_u16,
                                 float                    *out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_SHWFS_H */
