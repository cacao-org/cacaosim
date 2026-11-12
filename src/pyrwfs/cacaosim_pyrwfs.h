// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs.h
 * @brief   High-throughput 4-sided Pyramid WFS optical & detector simulator (FFTW3f + cuFFT/Graph)
 */

#ifndef CACAOSIM_PYRWFS_H
#define CACAOSIM_PYRWFS_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Optical and detector configuration for the Pyramid WFS engine.
 */
typedef struct
{
    uint32_t           nwf;        /**< Input pupil grid dimension [pixels] */
    uint32_t           pad_factor; /**< Optical FFT oversampling factor (>= 2, default: 2) */
    uint32_t           ncam;       /**< Output square WFS camera dimension [pixels] */
    float              pup_diam;   /**< Clear pupil diameter fraction of nwf (default: 0.9375) */
    float              lambda_um;  /**< Sensing wavelength [microns] (default: 0.75) */
    float              mod_rad_ld; /**< Modulation radius [lambda/D] (0 = unmodulated) */
    int32_t            mod_npts;   /**< Discrete modulation points around circle (>= 1) */
    float              pup_sep;    /**< Sub-pupil center separation fraction of nfft (def: 0.5) */
    float              roof_ld;    /**< Pyramid vertex roof rounding width [lambda/D] (0 = sharp) */
    float              fstop_frac; /**< Focal field-stop half-width fraction of nfft (def: 0.45) */
    int32_t            wf_unit;    /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda; /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;   /**< Base PRNG seed for photon/readout noise */
    int32_t            backend;    /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;        /**< Integrated WFS camera readout parameters */
} cacaosim_pyrwfs_cfg_t;

/**
 * @brief Persistent CPU (FFTW3f + L2 OpenMP) and CUDA (cuFFT + CUDA Graph) Pyramid WFS state.
 */
typedef struct
{
    cacaosim_pyrwfs_cfg_t cfg;
    uint32_t              nfft;
    uint64_t              npix_wf;
    uint64_t              npix_fft;
    uint64_t              npix_cam;
    int32_t               eff_npts;
    float                 cached_mod_rad;
    float                 flat_flux_norm;
    int32_t               use_cuda;
    uint32_t              subframe_cnt;
    uint64_t              frame_seq;

    /* Precomputed shifted complex pyramid masks [eff_npts * npix_fft * 2] (interleaved re,im) */
    float                *masks_cplx;

    /* CPU FFTW3f state (1 forward buffer + per-thread L2-resident inverse buffers) */
    float                *cpu_pup_cplx;
    float                *cpu_foc_cplx;
    float                *cpu_thr_cplx;
    float                *cpu_thr_int;
    float                *cpu_det_int;
    float                *cpu_cam_int;
    float                *cpu_accum_int;
    int32_t               cpu_nth;
    void                 *fftw_plan_fwd;
    void                **fftw_plans_inv;
    cacaosim_det_pool_t   det_pool;

    /* Opaque CUDA state allocated by cacaosim_pyrwfs_cuda.cu */
    void                 *cuda_priv;
} cacaosim_pyrwfs_ctx_t;

/**
 * cacaosim_pyrwfs_init() - Initialize Pyramid WFS masks, FFT plans, and flat-wavefront flux norm
 * @ctx:        Output Pyramid WFS context
 * @cfg:        Optical and detector configuration
 * @pup_amp:    2D telescope pupil amplitude [nwf * nwf]
 * @ncpa_opd:   Optional static WFS-path NCPA OPD [nwf * nwf] in microns (NULL -> 0)
 * @ext_stream: Optional external cudaStream_t to share in fused mode (NULL creates private)
 *
 * Return: 0 on success, -1 on failure.
 */
int cacaosim_pyrwfs_init(
    cacaosim_pyrwfs_ctx_t       *ctx,
    const cacaosim_pyrwfs_cfg_t *cfg,
    const float                 *pup_amp,
    const float                 *ncpa_opd,
    void                        *ext_stream);

/**
 * cacaosim_pyrwfs_free() - Destroy FFT plans and free all host/device Pyramid WFS buffers
 * @ctx: Pyramid WFS context to release
 */
void cacaosim_pyrwfs_free(cacaosim_pyrwfs_ctx_t *ctx);

/**
 * cacaosim_pyrwfs_update_masks() - Rebuild shifted focal-plane pyramid masks for new modulation
 * @ctx:        Initialized Pyramid WFS context
 * @mod_rad_ld: Updated modulation radius [lambda/D]
 */
void cacaosim_pyrwfs_update_masks(
    cacaosim_pyrwfs_ctx_t *ctx,
    float                  mod_rad_ld);

/**
 * cacaosim_pyrwfs_accum_subframe() - Propagate one wavefront+DM subframe into WFS accumulator
 */
void cacaosim_pyrwfs_accum_subframe(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *pup_amp,
    const float           *atm_amp,
    const float           *wf_in,
    const float           *dm_opd,
    const float           *ncpa_opd,
    const float           *d_dm_opd,
    float                  turb_gain,
    float                  mod_rad);

/**
 * cacaosim_pyrwfs_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into WFS
 */
void cacaosim_pyrwfs_accum_beam(
    cacaosim_pyrwfs_ctx_t *ctx,
    const float           *beam_opd,
    const float           *beam_amp,
    const float           *ncpa_opd,
    const float           *d_beam_opd,
    const float           *d_beam_amp,
    float                  mod_rad);

/**
 * cacaosim_pyrwfs_readout() - Finalize accumulated WFS exposure and perform detector readout
 */
void cacaosim_pyrwfs_readout(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/**
 * cacaosim_pyrwfs_step() - Single-subframe Pyramid WFS propagation and camera readout
 */
void cacaosim_pyrwfs_step(
    cacaosim_pyrwfs_ctx_t    *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    float                     mod_rad,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32);

/* Internal CUDA entry points in cacaosim_pyrwfs_cuda.cu */
int  cacaosim_pyrwfs_cuda_init(cacaosim_pyrwfs_ctx_t *ctx,
                               const float           *pup_amp,
                               const float           *ncpa_opd,
                               void                  *ext_stream);
void cacaosim_pyrwfs_cuda_upload_masks(cacaosim_pyrwfs_ctx_t *ctx);
void cacaosim_pyrwfs_cuda_free(cacaosim_pyrwfs_ctx_t *ctx);
void cacaosim_pyrwfs_cuda_accum(cacaosim_pyrwfs_ctx_t *ctx,
                                const float           *dyn_pup_amp,
                                const float           *atm_amp,
                                const float           *wf_in,
                                const float           *dm_opd,
                                const float           *d_pup_amp_ext,
                                const float           *d_wf_opd_ext,
                                const float           *d_dm_opd_ext,
                                float                  wf_to_opd);
void cacaosim_pyrwfs_cuda_readout(cacaosim_pyrwfs_ctx_t    *ctx,
                                  const cacaosim_det_cfg_t *det_live,
                                  uint16_t                 *out_u16,
                                  float                    *out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_PYRWFS_H */
