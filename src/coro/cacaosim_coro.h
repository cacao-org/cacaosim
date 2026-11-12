// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro.h
 * @brief   Coronagraphic focal-plane mask, Lyot stop, and LOWFS core pickoff simulator (CPU/CUDA)
 */

#ifndef CACAOSIM_CORO_H
#define CACAOSIM_CORO_H

#include <stdint.h>

#include "cacaosim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Coronagraphic focal-plane mask architecture.
 */
typedef enum
{
    CACAOSIM_CORO_LYOT_SPOT    = 0, /**< Subpixel Babinet MFT circular Lyot occulting spot */
    CACAOSIM_CORO_PHASE_DIMPLE = 1, /**< Subpixel Babinet MFT Roddier pi-phase disk */
    CACAOSIM_CORO_VORTEX       = 2, /**< Padded 2D FFT optical vortex mask (charge 2 or 4) */
    CACAOSIM_CORO_4QPM         = 3  /**< Padded 2D FFT four-quadrant phase mask */
} cacaosim_coro_mode_t;

/**
 * @brief Optical configuration for the coronagraphic propagation unit.
 */
typedef struct
{
    uint32_t nwf;           /**< Pupil grid dimension [pixels] */
    int32_t  mode;          /**< cacaosim_coro_mode_t (0=Lyot, 1=phase disk, 2=vortex, 3=4QPM) */
    uint32_t n_mft;         /**< Babinet MFT focal grid size for modes 0..1 (default: 48) */
    uint32_t pad_factor;    /**< FFT oversampling factor for modes 2..3 (>= 2, default: 4) */
    float    pup_diam;      /**< Entrance pupil clear diameter fraction of nwf (default: 0.9375) */
    float    lambda_um;     /**< Coronagraph design wavelength [microns] (default: 1.65) */
    float    fpm_rad_ld;    /**< Focal mask radius [lambda/D] for modes 0..1 (default: 3.0) */
    float    fpm_amp;       /**< Core amplitude transmission [0..1] (0 = opaque spot) */
    float    fpm_phase_rad; /**< Core phase shift [rad] for mode 1 (default: pi) */
    int32_t  vortex_charge; /**< Topological charge l (2 or 4) for mode 2 (default: 2) */
    float    lowfs_refl;    /**< Amplitude reflectivity of FPM core to LOWFS port (default: 0.95) */
    float    lyot_outer;    /**< Lyot stop outer diameter fraction of pup_diam (default: 0.85) */
    float    lyot_obs;      /**< Lyot stop central obscuration fraction of pup_diam (def: 0.36) */
    int32_t  lyot_nspider;  /**< Number of Lyot spider vanes (default: 0 or 4) */
    float    lyot_spider_w; /**< Lyot spider width fraction of pup_diam (default: 0.02) */
    float    lyot_spider_deg; /**< Lyot spider orientation [deg] (default: 45.0) */
    int32_t  backend;       /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
} cacaosim_coro_cfg_t;

/**
 * @brief Persistent CPU (OpenBLAS/FFTW3f) and CUDA (cuBLAS/cuFFT) coronagraph state.
 */
typedef struct
{
    cacaosim_coro_cfg_t cfg;
    uint32_t            n_mft;
    uint32_t            nfft;
    uint64_t            npix_wf;
    uint64_t            npix_mft;
    uint64_t            npix_fft;
    int32_t             is_babinet;
    int32_t             use_cuda;

    /* Lyot stop amplitude mask [nwf * nwf] */
    float              *lyot_mask;

    /* Babinet MFT matrices & focal mask (modes 0..1) */
    float              *fwd_U_cplx;
    float              *fwd_V_cplx;
    float              *inv_U_cplx;
    float              *inv_V_cplx;
    float              *fpm_weight; /**< Real subpixel circular mask M(u,v) in [0..1] */

    /* Full-plane FFT half-bin shift & focal mask (modes 2..3) */
    float              *shift_1d_cplx;
    float              *fft_mask_cplx;

    /* CPU work buffers */
    float              *cpu_epup_cplx;
    float              *cpu_ecore_cplx;
    float              *cpu_tmp_cplx;
    float              *cpu_efoc_cplx;
    float              *cpu_fft_buf;
    void               *fftw_fwd;
    void               *fftw_inv;

    /* Opaque CUDA state allocated by cacaosim_coro_cuda.cu */
    void               *cuda_priv;
} cacaosim_coro_ctx_t;

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
 * cacaosim_coro_step_cpu() - Propagate [in_opd, in_amp] through FPM + Lyot stop on CPU
 * @ctx:       Active coronagraph context
 * @in_opd:    Input pupil OPD [nwf * nwf] in microns
 * @in_amp:    Input pupil amplitude [nwf * nwf]
 * @out_opd:   Output post-Lyot OPD [nwf * nwf] in microns
 * @out_amp:   Output post-Lyot amplitude [nwf * nwf]
 * @lowfs_opd: Optional output FPM-reflected LOWFS pupil OPD [nwf * nwf] in microns (or NULL)
 * @lowfs_amp: Optional output FPM-reflected LOWFS pupil amplitude [nwf * nwf] (or NULL)
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
 * cacaosim_coro_step_beam() - Propagate beam on active backend (host or device pointers)
 *
 * When @d_in_opd and @d_out_opd are non-NULL and CUDA is active, executes entirely in GPU memory
 * without host transfers. Otherwise uses @in_opd / @out_opd host buffers.
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
void cacaosim_coro_cuda_step(cacaosim_coro_ctx_t *ctx,
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
                             float               *d_lowfs_amp);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_CORO_H */
