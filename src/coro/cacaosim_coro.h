// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro.h
 * @brief   Coronagraphic pupil/focal custom masks, polychromatic MFT/FFT, Lyot, & LOWFS (CPU/CUDA)
 */

#ifndef CACAOSIM_CORO_H
#define CACAOSIM_CORO_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_mask.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_CORO_MAX_LAMBDA 32

/**
 * @brief Coronagraphic focal-plane mask architecture.
 */
typedef enum
{
    CACAOSIM_CORO_LYOT_SPOT    = 0, /**< Subpixel Babinet MFT circular Lyot occulting spot */
    CACAOSIM_CORO_PHASE_DIMPLE = 1, /**< Subpixel Babinet MFT Roddier phase disk */
    CACAOSIM_CORO_VORTEX       = 2, /**< Padded 2D FFT optical vortex mask (charge 2 or 4) */
    CACAOSIM_CORO_4QPM         = 3, /**< Padded 2D FFT four-quadrant phase mask */
    CACAOSIM_CORO_CUSTOM_MFT   = 4, /**< User custom focal/pupil mask via Babinet Matrix-DFT */
    CACAOSIM_CORO_CUSTOM_FFT   = 5  /**< User custom full-plane focal mask via padded 2D FFT */
} cacaosim_coro_mode_t;

/**
 * @brief Coronagraphic LOWFS pickoff architecture.
 */
typedef enum
{
    CACAOSIM_LOWFS_FPM_REFL  = 0, /**< FPM reflective: light absorbed by FPM re-imaged to LOWFS */
    CACAOSIM_LOWFS_LYOT_REFL = 1  /**< Lyot stop reflective: light blocked by Lyot to LOWFS */
} cacaosim_lowfs_mode_t;

/**
 * @brief Optional custom user-provided pupil-plane, focal-plane, and Lyot-plane masks.
 *
 * Each plane supports amplitude only, phase only, or both amplitude and phase.
 * Any mask may be 2D [nx, ny] (achromatic, nz = 1) or a 3D [nx, ny, Nlambda] cube whose 3rd axis
 * encodes wavelength across the coronagraph bandpass.
 */
typedef struct
{
    cacaosim_mask_cube_t pup_amp;  /**< Pre-FPM pupil apodizer amplitude [nwf x nwf (x Nlam)] */
    cacaosim_mask_cube_t pup_pha;  /**< Pre-FPM pupil apodizer phase/OPD [nwf x nwf (x Nlam)] */
    cacaosim_mask_cube_t fpm_amp;  /**< Focal-plane mask amplitude [nx_f x ny_f (x Nlam)] */
    cacaosim_mask_cube_t fpm_pha;  /**< Focal-plane mask phase/OPD [nx_f x ny_f (x Nlam)] */
    cacaosim_mask_cube_t lyot_amp; /**< Post-FPM Lyot stop amplitude [nwf x nwf (x Nlam)] */
    cacaosim_mask_cube_t lyot_pha; /**< Post-FPM Lyot stop phase/OPD [nwf x nwf (x Nlam)] */
} cacaosim_coro_masks_t;

/**
 * @brief Optical configuration for the coronagraphic propagation unit.
 */
typedef struct
{
    uint32_t nwf;             /**< Pupil grid dimension [pixels] */
    int32_t  mode;            /**< cacaosim_coro_mode_t (0..5) */
    uint32_t n_mft;           /**< Babinet MFT focal grid size for modes 0..1 (default: 48) */
    uint32_t pad_factor;      /**< FFT oversampling factor for modes 2, 3, 5 (>= 2, default: 4) */
    float    pup_diam;        /**< Entrance pupil clear diameter fraction of nwf (def: 0.9375) */
    float    lambda_um;       /**< Coronagraph design / central wavelength [um] (default: 1.65) */
    float    fpm_rad_ld;      /**< Focal mask radius [lambda0/D] for modes 0..1 (default: 3.0) */
    float    fpm_amp;         /**< Core amplitude transmission [0..1] (0 = opaque spot) */
    float    fpm_phase_rad;   /**< Core phase shift [rad] for mode 1 (default: pi) */
    int32_t  vortex_charge;   /**< Topological charge l (even integer) for mode 2 (default: 2) */
    float    lowfs_refl;      /**< Amplitude reflectivity of FPM/Lyot to LOWFS (default: 0.95) */
    float    lyot_outer;      /**< Lyot stop outer diameter fraction of pup_diam (default: 0.85) */
    float    lyot_obs;        /**< Lyot stop central obscuration fraction of pup_diam (def: 0.36) */
    int32_t  lyot_nspider;    /**< Number of Lyot spider vanes (default: 0 or 4) */
    float    lyot_spider_w;   /**< Lyot spider width fraction of pup_diam (default: 0.02) */
    float    lyot_spider_deg; /**< Lyot spider orientation [deg] (default: 45.0) */
    int32_t  backend;         /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    float    fpm_pixscale_ld; /**< Custom MFT focal sampling [pixels per lambda0/D] (def: 8.0) */
    float    bw_frac;         /**< Fractional bandpass dlambda/lambda0 (def: 0.20 for 3D masks) */
    int32_t  nlambda;         /**< Spectral channels (0/1 = auto from 3D masks, >1 = explicit) */
    int32_t  mask_pha_unit;   /**< Custom *_pha mask unit: 0 = phase [rad], 1 = OPD [um] */
    int32_t  lowfs_mode;      /**< cacaosim_lowfs_mode_t: 0 = fpm_refl, 1 = lyot_refl */
    float    lowfs_pup_diam;  /**< Pupil stop size on FPM->LOWFS path [D_pup] (<=0 -> 1.0) */
} cacaosim_coro_cfg_t;

/**
 * @brief Persistent CPU (OpenBLAS/FFTW3f) and CUDA (cuBLAS/cuFFT) coronagraph state.
 */
typedef struct
{
    cacaosim_coro_cfg_t cfg;
    uint32_t            n_mft;
    uint32_t            nx_mft;
    uint32_t            ny_mft;
    uint32_t            nfft;
    uint64_t            npix_wf;
    uint64_t            npix_mft;
    uint64_t            npix_fft;
    int32_t             eff_nlam;
    int32_t             is_babinet;
    int32_t             skip_fpm;
    int32_t             has_pup_mask;
    int32_t             use_cuda;

    /* Per-channel wavelengths [eff_nlam] in microns */
    float               lam_um[CACAOSIM_CORO_MAX_LAMBDA];

    /* 2D slice-0 Lyot amplitude [npix_wf] + full complex pupil/Lyot cubes [eff_nlam * npix_wf] */
    float              *lyot_mask;
    float              *lowfs_pup_mask;
    float              *pup_mask_cplx;
    float              *lyot_cplx;

    /* Babinet MFT matrices [eff_nlam * n_mft * nwf] & focal weights [eff_nlam * npix_mft] */
    float              *fwd_U_cplx;
    float              *fwd_V_cplx;
    float              *inv_U_cplx;
    float              *inv_V_cplx;
    float              *fpm_delta_cplx; /**< Complex (T(u,v,lam) - 1) [eff_nlam * npix_mft * 2] */
    float              *fpm_refl_w;     /**< Real reflected core weight [eff_nlam * npix_mft] */

    /* Full-plane FFT half-bin shift [nfft] & focal mask cube [eff_nlam * npix_fft] */
    float              *shift_1d_cplx;
    float              *fft_mask_cplx;
    float              *fft_refl_w;     /**< Central-lam FFT FPM reflection weight [npix_fft] */

    /* CPU work buffers */
    float              *cpu_epup_cplx;
    float              *cpu_ecore_cplx;
    float              *cpu_tmp_cplx;
    float              *cpu_efoc_cplx;
    float              *cpu_efoc_r_cplx;
    float              *cpu_fft_buf;
    float              *cpu_fft_lo_buf;
    float              *cpu_poly_opd;
    float              *cpu_poly_amp;
    void               *fftw_fwd;
    void               *fftw_inv;
    void               *fftw_lo_inv;

    /* Opaque CUDA state allocated by cacaosim_coro_cuda.cu */
    void               *cuda_priv;
} cacaosim_coro_ctx_t;

/**
 * cacaosim_coro_init_masks() - Initialize coronagraph with custom 2D/3D pupil, FPM, & Lyot masks
 * @ctx:        Output coronagraph context
 * @cfg:        Coronagraph optical configuration
 * @masks:      Optional custom amplitude/phase mask cubes (may be NULL)
 * @ext_stream: Optional shared cudaStream_t for fused GPU pipelines (NULL creates private)
 *
 * Return: 0 on success, -1 on invalid dimensions or allocation failure.
 */
int cacaosim_coro_init_masks(
    cacaosim_coro_ctx_t         *ctx,
    const cacaosim_coro_cfg_t   *cfg,
    const cacaosim_coro_masks_t *masks,
    void                        *ext_stream);

/**
 * cacaosim_coro_init() - Initialize coronagraphic focal mask, Lyot stop, and CPU/CUDA plans
 * @ctx:        Output coronagraph context
 * @cfg:        Coronagraph optical configuration
 * @ext_lyot:   Optional external 2D Lyot stop mask [nwf * nwf] (NULL synthesizes from cfg)
 * @ext_stream: Optional shared cudaStream_t for fused GPU pipelines (NULL creates private)
 */
int cacaosim_coro_init(
    cacaosim_coro_ctx_t       *ctx,
    const cacaosim_coro_cfg_t *cfg,
    const float               *ext_lyot,
    void                      *ext_stream);

/**
 * cacaosim_coro_free() - Release all host and device coronagraph buffers
 */
void cacaosim_coro_free(cacaosim_coro_ctx_t *ctx);

/**
 * cacaosim_coro_step_poly_cpu() - Propagate 1 or N_lambda channels through coronagraph on CPU
 */
void cacaosim_coro_step_poly_cpu(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    uint32_t             in_nlam,
    uint64_t             in_stride,
    float               *out_opd,
    float               *out_amp,
    uint64_t             out_stride,
    float               *lowfs_opd,
    float               *lowfs_amp);

/**
 * cacaosim_coro_step_poly() - Propagate 1 or N_lambda channels on active backend (CPU or CUDA)
 * @ctx:         Active coronagraph context (writes ctx->eff_nlam post-Lyot channels)
 * @in_opd:      Host input pupil OPD [um]
 * @in_amp:      Host input pupil amplitude
 * @in_nlam:     Number of wavelength channels in input beam (1 = monochromatic)
 * @in_stride:   Element stride between input channels (e.g. npix_wf or 2 * npix_wf)
 * @out_opd:     Host output post-Lyot OPD [eff_nlam channels separated by @out_stride]
 * @out_amp:     Host output post-Lyot amplitude [eff_nlam channels separated by @out_stride]
 * @out_stride:  Element stride between output channels (npix_wf, or 2 * npix_wf for SHM cubes)
 * @lowfs_opd:   Optional 2D FPM/Lyot-reflected LOWFS pupil OPD [npix_wf] (central channel)
 * @lowfs_amp:   Optional 2D FPM/Lyot-reflected LOWFS pupil amplitude [npix_wf]
 * @d_in_opd:    Optional device input OPD (when non-NULL with @d_out_opd, runs purely on GPU)
 * @d_in_amp:    Optional device input amplitude
 * @d_out_opd:   Optional device output post-Lyot OPD
 * @d_out_amp:   Optional device output post-Lyot amplitude
 * @d_lowfs_opd: Optional device output LOWFS OPD
 * @d_lowfs_amp: Optional device output LOWFS amplitude
 */
void cacaosim_coro_step_poly(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    uint32_t             in_nlam,
    uint64_t             in_stride,
    float               *out_opd,
    float               *out_amp,
    uint64_t             out_stride,
    float               *lowfs_opd,
    float               *lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp);

/**
 * cacaosim_coro_step_cpu() - Propagate single-plane [in_opd, in_amp] -> [out_opd, out_amp] on CPU
 */
void cacaosim_coro_step_cpu(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    float               *out_opd,
    float               *out_amp,
    float               *lowfs_opd,
    float               *lowfs_amp);

/**
 * cacaosim_coro_step_beam() - Propagate single-plane beam on active backend (host or device)
 */
void cacaosim_coro_step_beam(
    cacaosim_coro_ctx_t *ctx,
    const float         *in_opd,
    const float         *in_amp,
    float               *out_opd,
    float               *out_amp,
    float               *lowfs_opd,
    float               *lowfs_amp,
    const float         *d_in_opd,
    const float         *d_in_amp,
    float               *d_out_opd,
    float               *d_out_amp,
    float               *d_lowfs_opd,
    float               *d_lowfs_amp);

/* Internal CUDA entry points implemented in cacaosim_coro_cuda.cu */
int  cacaosim_coro_cuda_init(cacaosim_coro_ctx_t *ctx, void *ext_stream);
void cacaosim_coro_cuda_free(cacaosim_coro_ctx_t *ctx);
void cacaosim_coro_cuda_step_poly(cacaosim_coro_ctx_t *ctx,
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
                                  float               *d_lowfs_amp);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_CORO_H */
