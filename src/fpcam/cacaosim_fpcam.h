// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam.h
 * @brief   High-throughput focal plane PSF & camera simulator (Soummer MFT + padded FFT, CPU/CUDA)
 */

#ifndef CACAOSIM_FPCAM_H
#define CACAOSIM_FPCAM_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_FPCAM_MAX_LAMBDA 32

/**
 * @brief Focal plane propagation mode.
 */
typedef enum
{
    CACAOSIM_FPCAM_PROP_MFT = 0, /**< Exact Soummer (2007) Matrix-DFT via BLAS/cuBLAS CGEMM */
    CACAOSIM_FPCAM_PROP_FFT = 1  /**< Zero-padded 2D FFT crop via FFTW3f/cuFFT */
} cacaosim_fpcam_prop_t;

/**
 * @brief Optical and detector configuration for the focal plane camera engine.
 */
typedef struct
{
    uint32_t           nwf;         /**< Input pupil grid dimension [pixels] */
    uint32_t           ncam;        /**< Output square camera dimension [pixels] */
    int32_t            prop_mode;   /**< 0 = MFT (arbitrary pixscale_ld), 1 = padded FFT */
    uint32_t           pad_factor;  /**< FFT zero-padding factor when prop_mode == 1 (>= 2) */
    float              pup_diam;    /**< Clear pupil diameter fraction of nwf (default: 0.9375) */
    float              lambda_um;   /**< Central science wavelength [microns] (default: 1.65) */
    float              bw_frac;     /**< Fractional bandpass dlambda / lambda0 [0..0.5] */
    int32_t            nlambda;     /**< Polychromatic spectral channels (1..32, default: 1) */
    int32_t            has_ext_lam; /**< 1 to use explicit ext_lam_um[0..nlambda-1] */
    float              ext_lam_um[CACAOSIM_FPCAM_MAX_LAMBDA]; /**< Explicit channel wavelengths */
    float              pixscale_ld; /**< Focal sampling [pixels per lambda0/D] (MFT mode) */
    int32_t            wf_unit;     /**< Input WF unit: 0 = OPD [um], 1 = phase [rad] */
    float              ref_lambda;  /**< Reference wavelength [um] when wf_unit == 1 */
    uint64_t           rng_seed;    /**< Base PRNG seed for photon/readout noise */
    int32_t            backend;     /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    cacaosim_det_cfg_t det;         /**< Integrated science detector readout parameters */
} cacaosim_fpcam_cfg_t;

/**
 * @brief Persistent CPU (OpenBLAS/FFTW3f) and CUDA (cuBLAS/cuFFT) focal plane camera context.
 */
typedef struct
{
    cacaosim_fpcam_cfg_t cfg;
    uint32_t             nfft;
    uint64_t             npix_wf;
    uint64_t             npix_fft;
    uint64_t             npix_cam;
    int32_t              eff_nlam;
    float                eff_pixscale_ld;
    float                pup_sum_amp;
    float                pup_sum_amp2;
    float                inf_flux_norm;
    int32_t              use_cuda;
    uint64_t             frame_seq;

    /* Spectral channel wavelengths [eff_nlam] */
    float                lam_um[CACAOSIM_FPCAM_MAX_LAMBDA];

    /* Precomputed MFT basis matrices: U [eff_nlam * ncam * nwf], V [eff_nlam * nwf * ncam] */
    float               *mft_U_cplx;
    float               *mft_V_cplx;

    /* CPU work buffers */
    float               *cpu_epup_cplx;
    float               *cpu_mft_tmp_cplx;
    float               *cpu_efoc_cplx;
    float               *cpu_fft_in_cplx;
    float               *cpu_fft_out_cplx;
    void                *fftw_plan_fwd;
    float               *cpu_sub_int;
    float               *cpu_accum_int;
    float               *cpu_exp_int;
    float               *cpu_long_int;
    uint32_t             subframe_cnt;
    uint64_t             long_cnt;
    float                last_strehl_axis;
    float                last_strehl_peak;
    float                long_strehl_peak;
    cacaosim_det_pool_t  det_pool;

    /* Opaque CUDA state allocated by cacaosim_fpcam_cuda.cu */
    void                *cuda_priv;
} cacaosim_fpcam_ctx_t;

/**
 * cacaosim_fpcam_init() - Initialize MFT/FFT optical propagation plans and detector state
 */
int cacaosim_fpcam_init(
    cacaosim_fpcam_ctx_t       *ctx,
    const cacaosim_fpcam_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream);

/**
 * cacaosim_fpcam_free() - Destroy plans and release all host/device focal plane camera buffers
 */
void cacaosim_fpcam_free(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_reset_long() - Clear cumulative long-exposure PSF accumulator
 */
void cacaosim_fpcam_reset_long(cacaosim_fpcam_ctx_t *ctx);

/**
 * cacaosim_fpcam_accum_subframe() - Propagate one optical subframe into exposure accumulator
 */
void cacaosim_fpcam_accum_subframe(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    const float          *d_dm_opd,
    float                 turb_gain);

/**
 * cacaosim_fpcam_accum_beam() - Propagate one [OPD_um, Amp] optical beam subframe into accumulator
 */
void cacaosim_fpcam_accum_beam(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_fpcam_accum_beam_poly() - Propagate 1 or N_lambda [OPD_c, Amp_c] channels into camera
 */
void cacaosim_fpcam_accum_beam_poly(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp);

/**
 * cacaosim_fpcam_readout() - Finalize exposure, update Strehl/long-exp, and run detector readout
 */
void cacaosim_fpcam_readout(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long);

/**
 * cacaosim_fpcam_step() - Single-subframe optical propagation + detector readout convenience call
 */
void cacaosim_fpcam_step(
    cacaosim_fpcam_ctx_t     *ctx,
    const float              *pup_amp,
    const float              *atm_amp,
    const float              *wf_in,
    const float              *dm_opd,
    const float              *ncpa_opd,
    const float              *d_dm_opd,
    float                     turb_gain,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long);

/* Internal CUDA entry points implemented in cacaosim_fpcam_cuda.cu */
int  cacaosim_fpcam_cuda_init(cacaosim_fpcam_ctx_t *ctx,
                              const float          *pup_amp,
                              const float          *ncpa_opd,
                              void                 *ext_stream);
void cacaosim_fpcam_cuda_free(cacaosim_fpcam_ctx_t *ctx);
void cacaosim_fpcam_cuda_reset_accum(cacaosim_fpcam_ctx_t *ctx);
void cacaosim_fpcam_cuda_accum(cacaosim_fpcam_ctx_t *ctx,
                               const float          *dyn_pup_amp,
                               const float          *atm_amp,
                               const float          *wf_in,
                               const float          *dm_opd,
                               const float          *d_pup_amp_ext,
                               const float          *d_wf_opd_ext,
                               const float          *d_dm_opd_ext,
                               float                 wf_to_opd);
void cacaosim_fpcam_cuda_accum_poly(cacaosim_fpcam_ctx_t *ctx,
                                    const float          *beam_opd,
                                    const float          *beam_amp,
                                    uint32_t              beam_nlam,
                                    uint64_t              chan_stride,
                                    const float          *d_beam_opd,
                                    const float          *d_beam_amp);
void cacaosim_fpcam_cuda_readout(cacaosim_fpcam_ctx_t     *ctx,
                                 const cacaosim_det_cfg_t *det_live,
                                 float                    *host_exp_int,
                                 uint16_t                 *out_u16,
                                 float                    *out_f32);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_FPCAM_H */
