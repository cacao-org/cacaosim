// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs.h
 * @brief   Babinet Matrix-DFT Zernike phase-contrast WFS simulator (OpenBLAS + cuBLAS)
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
 */
typedef struct
{
    uint32_t           nwf;        /**< Input pupil grid dimension [pixels] */
    uint32_t           ncam;       /**< Output square WFS camera dimension [pixels] */
    uint32_t           n_mft;      /**< Focal-core MFT grid size (>= 16, default: 32) */
    float              pup_diam;   /**< Clear pupil diameter fraction of nwf (default: 0.9375) */
    float              lambda_um;  /**< Sensing wavelength [microns] (default: 0.75) */
    float              dimple_ld;  /**< Phase dimple diameter [lambda/D] (default: 1.06) */
    float              dimple_rad; /**< Phase shift inside dimple [radians] (default: pi/2) */
    float              dimple_amp; /**< Amplitude transmission inside dimple [0..1] (def: 1.0) */
    int32_t            wf_unit;    /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda; /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;   /**< Base PRNG seed for photon/readout noise */
    int32_t            backend;    /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;        /**< Integrated WFS camera readout parameters */
} cacaosim_zwfs_cfg_t;

/**
 * @brief Persistent CPU (OpenBLAS) and CUDA (cuBLAS) Zernike WFS state.
 */
typedef struct
{
    cacaosim_zwfs_cfg_t cfg;
    uint32_t            n_mft;
    uint64_t            npix_wf;
    uint64_t            npix_mft;
    uint64_t            npix_cam;
    float               flat_flux_norm;
    int32_t             use_cuda;
    uint32_t            subframe_cnt;
    uint64_t            frame_seq;

    /* Precomputed MFT matrices & complex Babinet focal mask (t_d*exp(i*theta) - 1)*M(u,v) */
    float              *fwd_U_cplx;     /**< [n_mft * nwf * 2] */
    float              *fwd_V_cplx;     /**< [nwf * n_mft * 2] */
    float              *inv_U_cplx;     /**< [nwf * n_mft * 2] */
    float              *inv_V_cplx;     /**< [n_mft * nwf * 2] */
    float              *mask_bab_cplx;  /**< [n_mft * n_mft * 2] */

    /* CPU work buffers */
    float              *cpu_epup_cplx;  /**< [nwf * nwf * 2] */
    float              *cpu_tmp_cplx;   /**< [n_mft * nwf * 2] */
    float              *cpu_efoc_cplx;  /**< [n_mft * n_mft * 2] */
    float              *cpu_pup_int;    /**< [nwf * nwf] */
    float              *cpu_cam_int;    /**< [ncam * ncam] */
    float              *cpu_accum_int;  /**< [ncam * ncam] */
    cacaosim_det_pool_t det_pool;

    /* Opaque CUDA state allocated by cacaosim_zwfs_cuda.cu */
    void               *cuda_priv;
} cacaosim_zwfs_ctx_t;

/**
 * cacaosim_zwfs_init() - Initialize Babinet MFT Zernike WFS matrices, mask, and flux norm
 */
int cacaosim_zwfs_init(
    cacaosim_zwfs_ctx_t       *ctx,
    const cacaosim_zwfs_cfg_t *cfg,
    const float               *pup_amp,
    const float               *ncpa_opd,
    void                      *ext_stream);

/**
 * cacaosim_zwfs_free() - Release all host and device Zernike WFS resources
 */
void cacaosim_zwfs_free(cacaosim_zwfs_ctx_t *ctx);

/**
 * cacaosim_zwfs_accum_subframe() - Propagate one wavefront+DM subframe into ZWFS accumulator
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
 */
void cacaosim_zwfs_readout(
    cacaosim_zwfs_ctx_t      *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/**
 * cacaosim_zwfs_step() - Single-subframe Zernike WFS propagation and camera readout
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

/* Internal CUDA entry points implemented in cacaosim_zwfs_cuda.cu */
int  cacaosim_zwfs_cuda_init(cacaosim_zwfs_ctx_t *ctx,
                             const float         *pup_amp,
                             const float         *ncpa_opd,
                             void                *ext_stream);
void cacaosim_zwfs_cuda_free(cacaosim_zwfs_ctx_t *ctx);
void cacaosim_zwfs_cuda_accum(cacaosim_zwfs_ctx_t *ctx,
                              const float         *dyn_pup_amp,
                              const float         *atm_amp,
                              const float         *wf_in,
                              const float         *dm_opd,
                              const float         *d_pup_amp_ext,
                              const float         *d_wf_opd_ext,
                              const float         *d_dm_opd_ext,
                              float                wf_to_opd);
void cacaosim_zwfs_cuda_readout(cacaosim_zwfs_ctx_t      *ctx,
                                const cacaosim_det_cfg_t *det_live,
                                uint16_t                 *out_u16,
                                float                    *out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_ZWFS_H */
