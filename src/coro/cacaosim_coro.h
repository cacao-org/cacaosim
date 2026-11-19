// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro.h
 * @brief   Coronagraphic pupil/focal custom masks, polychromatic MFT/FFT, Lyot, & LOWFS (CPU/CUDA)
 *
 * Architectural Role in cacaosim:
 *   Defines the public C API, configuration structures, and persistent CPU/CUDA execution context
 *   for the polychromatic coronagraphic diffraction unit (`cacaosim.coro`). Positioned inline on
 *   an Adaptive Optics optical bench between upstream wavefront control (`mkbeam`, `dmsim`, `wfe`,
 *   `beamsplit`) and downstream science cameras (`fpcam`) or post-coronagraphic wavefront sensors
 *   (`zwfs`, `pyrwfs`, `shwfs`), this module propagates 1 to 32 coherent spectral channels across
 *   three optical planes:
 *     1. Entrance pupil plane (with optional 2D/3D custom complex apodizer `pup_amp` / `pup_pha`),
 *     2. Coronagraphic focal plane (analytic or custom 2D/3D masks via Babinet MFT or padded FFT),
 *     3. Post-FPM Lyot stop plane (geometric pupil stop or custom 2D/3D `lyot_amp` / `lyot_pha`),
 *   while simultaneously extracting a reflected Low-Order Wavefront Sensor (`out_lowfs`) pupil
 *   beam from either the focal-plane mask (`fpm_refl`, with relay pupil stop `lowfs_pup_diam`) or
 *   a reflective Lyot stop (`lyot_refl`).
 *
 * Design Rationale & Optical Physics:
 *   - Compact Focal Masks via Babinet Matrix-DFT (`LYOT_SPOT`, `PHASE_DIMPLE`, `CUSTOM_MFT`):
 *     Small occulting spots and phase dimples (radius ~1..6 lambda0/D) require fine sub-pixel
 *     focal sampling (8..16 pixels per lambda/D) to suppress staircase edge diffraction. Achieving
 *     this resolution with a standard 2D FFT would require 8x..16x zero-padding across every
 *     spectral channel. Decomposing the focal transmission as T(u,v) = 1 + (T(u,v) - 1) via
 *     Babinet's principle restricts the discrete Fourier transform to the compact support of
 *     Delta_M = T - 1:
 *       Psi_Lyot = L * (Psi_pup + MFT^-1[ (T - 1) * MFT[Psi_pup] ])
 *     Evaluating the forward and inverse MFTs as separable two-step `cblas_cgemm` /
 *     `cublasCgemmStridedBatched` matrix products on a compact Mx x My focal patch (e.g. 48x48)
 *     delivers exact sub-pixel diffraction at a fraction of padded-FFT memory and FLOPs, while
 *     naturally applying exact physical chromatic magnification du_c = du0 * (lam0 / lam_c).
 *   - Extended Focal Masks via Padded 2D FFT (`VORTEX`, `4QPM`, `CUSTOM_FFT`):
 *     Scale-invariant phase masks such as Optical Vortex exp(i * l * theta) and Four-Quadrant
 *     Phase Masks sgn(u)*sgn(v) modify phase out to infinite focal radius and therefore cannot be
 *     truncated to a compact Babinet box. These modes use zero-padded 2D FFTs (`nfft = nwf * pad`)
 *     combined with half-pixel pupil phase ramps exp(-i * pi * m / nfft) so the optical axis lands
 *     symmetrically between four central focal bins, avoiding the (0, 0) phase singularity.
 *   - Complex Lyot Field to `[OPD_um, Amplitude]` Stream Encoding:
 *     Post-Lyot and LOWFS complex fields E = A * exp(i * phi) are exported as `[OPD_um, Amplitude]`
 *     slice pairs (`OPD_um = atan2(Im, Re) * lam_c / (2*pi)`) compatible with all downstream
 *     `cacaosim` beam consumers. Downstream coherent propagators rebuild `A * exp(i*2*pi*OPD/lam)`
 *     per channel, preserving both coronagraphic amplitude apodization and wrapped phase exactly.
 */

#ifndef CACAOSIM_CORO_H
#define CACAOSIM_CORO_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_mask.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of coherent spectral wavelength channels supported across the coronagraph.
 *
 * Matches `CACAOSIM_MASK_MAX_LAMBDA` and the 32-bin photometric grid so 3D `.fits` / SHM mask
 * cubes and polychromatic SHM beam streams `[nwf x nwf x (2 * Nlambda)]` map 1-to-1.
 */
#define CACAOSIM_CORO_MAX_LAMBDA 32

/**
 * enum cacaosim_coro_mode_t - Coronagraphic focal-plane mask diffraction architecture
 * @CACAOSIM_CORO_LYOT_SPOT:    Analytic circular Lyot occulting spot (opaque or partially
 *                              transmissive `fpm_amp`) evaluated with 8x8 sub-pixel anti-aliasing
 *                              via compact Soummer Babinet Matrix-DFT.
 * @CACAOSIM_CORO_PHASE_DIMPLE: Analytic circular Roddier / dual-zone phase-shifting disk with
 *                              core amplitude `fpm_amp` and chromatic phase step
 *                              `fpm_phase_rad * (lambda0 / lambda_c)` via Babinet Matrix-DFT.
 * @CACAOSIM_CORO_VORTEX:       Extended azimuthal optical vortex phase ramp `exp(i * l * theta)`
 *                              (even topological charge `l` in {2, 4}) evaluated via zero-padded
 *                              2D FFT with half-pixel inter-bin centering.
 * @CACAOSIM_CORO_4QPM:         Extended Rouan Four-Quadrant Phase Mask `sgn(u) * sgn(v)` with pi
 *                              phase steps along focal axes via half-pixel shifted 2D FFT.
 * @CACAOSIM_CORO_CUSTOM_MFT:   User-supplied compact 2D/3D focal-plane mask (`fpm_amp`/`fpm_pha`)
 *                              propagated at `fpm_pixscale_ld` pixels per `lambda0/D` via Babinet
 *                              MFT; also supports pure pupil-only apodizers when FPM is omitted.
 * @CACAOSIM_CORO_CUSTOM_FFT:   User-supplied wide-field 2D/3D focal-plane mask (`fpm_amp`/
 *                              `fpm_pha`) embedded into the `nfft x nfft` padded 2D FFT grid.
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
 * enum cacaosim_lowfs_mode_t - Coronagraphic Low-Order Wavefront Sensor (LOWFS) pickoff physics
 * @CACAOSIM_LOWFS_FPM_REFL:  Reflective focal-plane mask pickoff (e.g. reflective Lyot spot /
 *                            LLOWFS): starlight rejected by the FPM core with amplitude weight
 *                            `sqrt(1 - |T(u,v)|^2)` is inverse-transformed to a relay pupil plane
 *                            and clipped by `lowfs_pup_diam`.
 * @CACAOSIM_LOWFS_LYOT_REFL: Reflective Lyot stop pickoff (classical LLOWFS): light diffracted
 *                            outside the Lyot stop clear aperture (`|L(x,y)|^2 < 1`) is reflected
 *                            directly in the Lyot pupil with weight `sqrt(1 - |L(x,y)|^2)`.
 */
typedef enum
{
    CACAOSIM_LOWFS_FPM_REFL  = 0, /**< FPM reflective: light absorbed by FPM re-imaged to LOWFS */
    CACAOSIM_LOWFS_LYOT_REFL = 1  /**< Lyot stop reflective: light blocked by Lyot to LOWFS */
} cacaosim_lowfs_mode_t;

/**
 * struct cacaosim_coro_masks_t - Optional user-provided pupil, focal, and Lyot mask cubes
 * @pup_amp:  Pre-FPM entrance pupil apodizer amplitude transmission `[nwf x nwf (x Nlam)]`
 *            (e.g. shaped pupil or prolate spheroidal apodizer; omitted -> uniform 1.0).
 * @pup_pha:  Pre-FPM entrance pupil apodizer phase `[rad]` or OPD `[um]` `[nwf x nwf (x Nlam)]`
 *            (e.g. vAPP or PIAA phase plate; omitted -> 0.0).
 * @fpm_amp:  Focal-plane mask amplitude transmission `[nx_f x ny_f (x Nlam)]` for `CUSTOM_MFT` or
 *            `CUSTOM_FFT` modes (omitted -> 1.0 everywhere).
 * @fpm_pha:  Focal-plane mask phase `[rad]` or OPD `[um]` `[nx_f x ny_f (x Nlam)]` for
 *            `CUSTOM_MFT` or `CUSTOM_FFT` modes (omitted -> 0.0 everywhere).
 * @lyot_amp: Post-FPM Lyot stop amplitude transmission `[nwf x nwf (x Nlam)]`; overrides the
 *            analytic geometric Lyot stop when present.
 * @lyot_pha: Post-FPM Lyot stop phase `[rad]` or OPD `[um]` `[nwf x nwf (x Nlam)]` (e.g. static
 *            Lyot-plane phase mask or aberration; omitted -> 0.0).
 *
 * Each plane independently supports amplitude only, phase only, or combined complex transmission.
 * Any mask may be 2D `[nx, ny]` (achromatic across channels, `nz = 1`) or a 3D `[nx, ny, Nlambda]`
 * cube whose 3rd axis encodes wavelength-dependent complex transmission across the bandpass.
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
 * struct cacaosim_coro_cfg_t - Optical and numerical configuration for `cacaosim.coro`
 * @nwf:             Square pupil wavefront grid dimension `[pixels]` (`>= 8`).
 * @mode:            Focal-plane coronagraph mode (`cacaosim_coro_mode_t`, `0..5`).
 * @n_mft:           Square Babinet MFT focal grid dimension `[pixels]` for analytic modes `0..1`
 *                   (`>= 16`, default: `48`).
 * @pad_factor:      Zero-padding oversampling factor `nfft / nwf` for FFT modes `2, 3, 5`
 *                   (`>= 2`, default: `4`).
 * @pup_diam:        Clear entrance pupil diameter `D` as a fraction of `nwf` (default: `0.9375`).
 * @lambda_um:       Coronagraph design / central reference wavelength `lambda0` `[um]`
 *                   (default: `1.65`).
 * @fpm_rad_ld:      Analytic focal mask radius `[lambda0 / D]` for modes `0..1` (default: `3.0`).
 * @fpm_amp:         Analytic focal mask core amplitude transmission `[0..1]` (`0` = opaque spot).
 * @fpm_phase_rad:   Analytic core phase shift `[rad]` at `lambda0` for `PHASE_DIMPLE` (mode `1`,
 *                   default: `pi`; scales chromatically as `lambda0 / lambda_c`).
 * @vortex_charge:   Optical vortex topological charge `l` (`2` or `4`) for mode `2` (def: `2`).
 * @lowfs_refl:      Scalar amplitude reflectivity `[0..1]` into `out_lowfs` port (default: `0.95`).
 * @lyot_outer:      Analytic Lyot stop outer diameter fraction of `pup_diam` (default: `0.85`).
 * @lyot_obs:        Analytic Lyot stop central obscuration fraction of `pup_diam` (def: `0.36`).
 * @lyot_nspider:    Number of analytic Lyot stop spider vanes (default: `0` or `4`).
 * @lyot_spider_w:   Analytic Lyot spider vane width as a fraction of `pup_diam` (default: `0.02`).
 * @lyot_spider_deg: Analytic Lyot spider orientation angle `[deg]` (default: `45.0`).
 * @backend:         Compute backend (`cacaosim_backend_t`: `0` = auto, `1` = CPU, `2` = CUDA).
 * @fpm_pixscale_ld: Focal sampling `[pixels per lambda0/D]` for `CUSTOM_MFT` (default: `8.0`).
 * @bw_frac:         Fractional spectral bandwidth `dlambda / lambda0` (default: `0.20` for 3D).
 * @nlambda:         Number of coherent spectral channels (`0/1` = auto from 3D masks, `>1` = set).
 * @mask_pha_unit:   Physical unit of custom `*_pha` masks: `0` = phase `[rad]` (achromatic),
 *                   `1` = OPD `[um]` (chromatic `2*pi*OPD/lambda_c`).
 * @lowfs_mode:      LOWFS pickoff (`cacaosim_lowfs_mode_t`: `0` = `fpm_refl`, `1` = `lyot_refl`).
 * @lowfs_pup_diam:  Relay circular pupil stop diameter on `fpm_refl` LOWFS path in units of
 *                   `D_pup` (`<= 0` defaults to `1.0`).
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
    int32_t  lyot_nspider;       /**< Number of Lyot spider vanes (default: 0 or 4) */
    float    lyot_spider_w;      /**< Lyot spider width fraction of pup_diam (default: 0.02) */
    float    lyot_spider_deg;    /**< Lyot spider orientation [deg] (default: 45.0) */
    float    lyot_spider_offset; /**< Lyot V-spider X-vertex offset fraction of pup_diam (0=rad) */
    int32_t  backend;            /**< cacaosim_backend_t: 0 = auto, 1 = CPU, 2 = CUDA */
    float    fpm_pixscale_ld; /**< Custom MFT focal sampling [pixels per lambda0/D] (def: 8.0) */
    float    bw_frac;         /**< Fractional bandpass dlambda/lambda0 (def: 0.20 for 3D masks) */
    int32_t  nlambda;         /**< Spectral channels (0/1 = auto from 3D masks, >1 = explicit) */
    int32_t  mask_pha_unit;   /**< Custom *_pha mask unit: 0 = phase [rad], 1 = OPD [um] */
    int32_t  lowfs_mode;      /**< cacaosim_lowfs_mode_t: 0 = fpm_refl, 1 = lyot_refl */
    float    lowfs_pup_diam;  /**< Pupil stop size on FPM->LOWFS path [D_pup] (<=0 -> 1.0) */
} cacaosim_coro_cfg_t;

/**
 * struct cacaosim_coro_ctx_t - Persistent CPU (OpenBLAS/FFTW3f) and CUDA (cuBLAS/cuFFT) state
 * @cfg:             Active optical configuration (with promoted `mode` and resolved `lambda_um`).
 * @n_mft:           Nominal MFT focal width (`== nx_mft`) when `is_babinet == 1`.
 * @nx_mft:          Babinet MFT focal X dimension `[pixels]`.
 * @ny_mft:          Babinet MFT focal Y dimension `[pixels]`.
 * @nfft:            Padded 2D FFT grid dimension `nwf * pad_factor` when `is_babinet == 0`.
 * @npix_wf:         Total pupil pixels per slice (`nwf * nwf`).
 * @npix_mft:        Total Babinet focal pixels per slice (`nx_mft * ny_mft`).
 * @npix_fft:        Total padded FFT pixels per slice (`nfft * nfft`).
 * @eff_nlam:        Resolved number of coherent spectral channels (`1..CACAOSIM_CORO_MAX_LAMBDA`).
 * @is_babinet:      Non-zero if using compact Babinet Matrix-DFT; 0 if using padded 2D FFT.
 * @skip_fpm:        Non-zero when `CUSTOM_MFT` has no FPM (`fa == NULL && fp == NULL`), bypassing
 *                   focal diffraction for pure pupil/Lyot apodizer configurations.
 * @has_pup_mask:    Non-zero if `pup_mask_cplx` is populated from `pup_amp` and/or `pup_pha`.
 * @use_cuda:        Non-zero when GPU execution via `cuda_priv` is active.
 * @lam_um:          Per-channel wavelength table `[eff_nlam]` in `[um]`.
 * @lyot_mask:       Slice-0 real Lyot stop amplitude `[npix_wf]` for inspection/telemetry.
 * @lowfs_pup_mask:  Anti-aliased circular pupil stop `[npix_wf]` on the `fpm_refl` LOWFS relay.
 * @pup_mask_cplx:   Interleaved `[re, im]` pre-FPM pupil apodizer cube `[eff_nlam * npix_wf * 2]`.
 * @lyot_cplx:       Interleaved `[re, im]` post-FPM Lyot stop cube `[eff_nlam * npix_wf * 2]`.
 * @fwd_U_cplx:      Forward MFT Y-axis basis `[eff_nlam * ny_mft * nwf * 2]` (`sqrt(du_c/dpix)`).
 * @fwd_V_cplx:      Forward MFT X-axis basis `[eff_nlam * nwf * nx_mft * 2]` (`sqrt(du_c/dpix)`).
 * @inv_U_cplx:      Inverse MFT Y-axis conjugate transpose `[eff_nlam * nwf * ny_mft * 2]`.
 * @inv_V_cplx:      Inverse MFT X-axis conjugate transpose `[eff_nlam * nx_mft * nwf * 2]`.
 * @fpm_delta_cplx:  Precomputed Babinet core perturbation `T(u,v,lam) - 1`
 *                   `[eff_nlam * npix_mft * 2]`.
 * @fpm_refl_w:      Real FPM reflected amplitude weight `sqrt(1 - |T|^2)` `[eff_nlam * npix_mft]`.
 * @shift_1d_cplx:   Concatenated X and Y 1D half-bin pupil shift phasors `[2 * nfft * 2]`.
 * @fft_mask_cplx:   Pre-normalized (`1/nfft^2`) complex FFT focal mask `[eff_nlam * npix_fft * 2]`.
 * @fft_refl_w:      Pre-normalized (`1/nfft^2`) FPM reflection weight `[eff_nlam * npix_fft]`.
 * @cpu_epup_cplx:   Scratch complex entrance pupil field `[npix_wf * 2]`.
 * @cpu_ecore_cplx:  Scratch complex inverse-MFT pupil perturbation `[npix_wf * 2]`.
 * @cpu_tmp_cplx:    Scratch intermediate MFT matrix product `[max(nx_mft, ny_mft) * nwf * 2]`.
 * @cpu_efoc_cplx:   Scratch complex MFT focal field `[npix_mft * 2]`.
 * @cpu_efoc_r_cplx: Scratch complex reflected MFT focal field for LOWFS `[npix_mft * 2]`.
 * @cpu_fft_buf:     In-place FFTW3f forward/inverse complex buffer `[npix_fft * 2]`.
 * @cpu_fft_lo_buf:  In-place FFTW3f inverse complex buffer for `fpm_refl` LOWFS `[npix_fft * 2]`.
 * @cpu_poly_opd:    Scratch polychromatic OPD cube `[eff_nlam * npix_wf]` for single-beam collapse.
 * @cpu_poly_amp:    Scratch polychromatic Amp cube `[eff_nlam * npix_wf]` for single-beam collapse.
 * @cpu_poly_lo_opd: Scratch polychromatic LOWFS OPD cube `[eff_nlam * npix_wf]` for 2D collapse.
 * @cpu_poly_lo_amp: Scratch polychromatic LOWFS Amp cube `[eff_nlam * npix_wf]` for 2D collapse.
 * @fftw_fwd:        Opaque `fftwf_plan` for in-place 2D forward FFT on `cpu_fft_buf`.
 * @fftw_inv:        Opaque `fftwf_plan` for in-place 2D backward FFT on `cpu_fft_buf`.
 * @fftw_lo_inv:     Opaque `fftwf_plan` for in-place 2D backward FFT on `cpu_fft_lo_buf`.
 * @cuda_priv:       Opaque pointer to `coro_cuda_state_t` allocated by `cacaosim_coro_cuda_init()`.
 *
 * Pre-allocates all host aligned buffers, FFTW3f plans, MFT Fourier basis matrices, and optional
 * device memory during initialization so hot per-frame propagation performs zero dynamic memory
 * allocations or plan creations.
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
    float              *fft_refl_w;     /**< Per-lam FFT FPM refl weight [eff_nlam * npix_fft] */

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
    float              *cpu_poly_lo_opd;
    float              *cpu_poly_lo_amp;
    float              *cpu_pre_lyot_int;  /**< Broadband intensity incident on Lyot [npix_wf] */
    float              *cpu_pre_lyot_long; /**< Cumulative pre-Lyot intensity sum [npix_wf] */
    void               *fftw_fwd;
    void               *fftw_inv;
    void               *fftw_lo_inv;

    /* Instantaneous & cumulative coronagraph / LOWFS optical power-budget tracking */
    uint64_t            long_cnt;
    double              sum_e_in;
    double              sum_e_pre_lyot;
    double              sum_e_post_lyot;
    double              sum_e_lowfs;
    float               last_frac_pre_lyot;
    float               last_frac_lyot_trans;
    float               last_frac_lowfs;
    float               long_frac_pre_lyot;
    float               long_frac_lyot_trans;
    float               long_frac_lowfs;

    /* Opaque CUDA state allocated by cacaosim_coro_cuda.cu */
    void               *cuda_priv;
} cacaosim_coro_ctx_t;

/**
 * cacaosim_coro_init_masks() - Initialize coronagraph with custom 2D/3D pupil, FPM, & Lyot masks
 * @ctx:        Output coronagraph context to populate (caller-allocated, non-NULL).
 * @cfg:        Coronagraph optical and numerical configuration (non-NULL, `cfg->nwf >= 8`).
 * @masks:      Optional custom amplitude/phase mask cubes in pupil, focal, and Lyot planes
 *              (may be NULL or have individual empty `cacaosim_mask_cube_t` entries).
 * @ext_stream: Optional shared `cudaStream_t` handle for zero-copy fused GPU bench execution
 *              (NULL creates a private non-blocking CUDA stream when CUDA is active).
 *
 * Role:
 *   Cold-path master initializer for `cacaosim.coro`. Validates mask geometries and spectral slice
 *   counts, resolves the channel wavelength grid, synthesizes fallback geometric Lyot and LOWFS
 *   stops, precomputes polychromatic Babinet MFT basis matrices or half-bin shifted 2D FFT plans,
 *   and uploads static optical tables to the GPU when CUDA is enabled.
 *
 * Rationale:
 *   Centralizing both analytic and user-supplied 2D/3D FITS/SHM mask initialization in one cold
 *   routine ensures strict consistency between spectral WCS headers (`lam_um`), chromatic phase
 *   conversion (`mask_pha_unit`), MFT/FFT normalization constants, and host/device memory layouts,
 *   keeping hot per-frame propagation completely free of allocations and trigonometric setup.
 *
 * Return: 0 on success, -1 on invalid dimensions, mismatched cube depths, or allocation failure.
 */
int cacaosim_coro_init_masks(
    cacaosim_coro_ctx_t         *ctx,
    const cacaosim_coro_cfg_t   *cfg,
    const cacaosim_coro_masks_t *masks,
    void                        *ext_stream);

/**
 * cacaosim_coro_init() - Initialize coronagraphic focal mask, Lyot stop, and CPU/CUDA plans
 * @ctx:        Output coronagraph context to populate (non-NULL).
 * @cfg:        Coronagraph optical configuration (non-NULL).
 * @ext_lyot:   Optional caller-owned 2D real Lyot stop amplitude map `[nwf * nwf]` (NULL
 *              synthesizes an anti-aliased geometric Lyot stop from `cfg`).
 * @ext_stream: Optional shared `cudaStream_t` for fused GPU pipelines (NULL creates private).
 *
 * Role:
 *   Convenience cold-path initializer for analytic coronagraph modes with an optional external 2D
 *   Lyot amplitude map.
 *
 * Rationale:
 *   Wraps `@ext_lyot` in a zero-copy 2D `cacaosim_mask_cube_t` view and delegates directly to
 *   `cacaosim_coro_init_masks()` so standalone benchmarks, tests, and the fused `htsim` engine do
 *   not need to construct a full `cacaosim_coro_masks_t` struct when only a 2D Lyot stop is used.
 *
 * Return: 0 on success, -1 on invalid parameters or allocation failure.
 */
int cacaosim_coro_init(
    cacaosim_coro_ctx_t       *ctx,
    const cacaosim_coro_cfg_t *cfg,
    const float               *ext_lyot,
    void                      *ext_stream);

/**
 * cacaosim_coro_free() - Release all host and device coronagraph buffers
 * @ctx: Coronagraph context to tear down (safe on NULL or partially initialized contexts).
 *
 * Role:
 *   Cold-path teardown routine destroying GPU handles/buffers, FFTW3f plans, and 64-byte aligned
 *   host arrays before zeroing `@ctx`.
 *
 * Rationale:
 *   Provides idempotent cleanup for both normal shutdown and mid-initialization error rollback.
 */
void cacaosim_coro_free(cacaosim_coro_ctx_t *ctx);

/**
 * cacaosim_coro_step_poly_cpu() - Propagate 1 or N_lambda channels through coronagraph on CPU
 * @ctx:        Initialized coronagraph context (`eff_nlam` post-Lyot channels produced).
 * @in_opd:     Host input pupil OPD `[um]` (`in_nlam` slices separated by `@in_stride`; NULL -> 0).
 * @in_amp:     Host input pupil amplitude (`in_nlam` slices separated by `@in_stride`; NULL -> 1).
 * @in_nlam:    Number of coherent wavelength channels in `@in_opd`/`@in_amp` (`1` = monochromatic
 *              input broadcast across all `ctx->eff_nlam` coronagraph channels).
 * @in_stride:  Float element stride between consecutive input wavelength channels (e.g. `npix_wf`
 *              for planar arrays or `2 * npix_wf` for interleaved `[OPD_c, Amp_c]` SHM cubes).
 * @out_opd:    Host output post-Lyot OPD `[um]` (`eff_nlam` slices separated by `@out_stride`).
 * @out_amp:    Host output post-Lyot amplitude (`eff_nlam` slices separated by `@out_stride`).
 * @out_stride: Float element stride between consecutive output channels (`npix_wf` or `2*npix_wf`).
 * @lowfs_opd:  Optional host 2D LOWFS pupil OPD `[um]` `[npix_wf]` evaluated at central channel
 *              `eff_nlam / 2` (may be NULL if LOWFS output is unused).
 * @lowfs_amp:  Optional host 2D LOWFS pupil amplitude `[npix_wf]` (may be NULL).
 *
 * Role:
 *   Hot-path CPU diffraction engine executing entrance pupil apodization, Babinet MFT
 *   (`cblas_cgemm`) or padded 2D FFT (`fftwf_execute`), LOWFS pickoff (`fpm_refl` or `lyot_refl`),
 *   and Lyot stop filtering across all `ctx->eff_nlam` spectral channels.
 *
 * Rationale:
 *   Supports arbitrary input/output channel strides (`in_stride`, `out_stride`) so callers can
 *   operate directly in-place on `ImageStreamIO` `[nwf x nwf x (2 * Nlambda)]` shared-memory cubes
 *   without intermediate de-interleaving copies.
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
 * @ctx:         Active coronagraph context (writes `ctx->eff_nlam` post-Lyot channels).
 * @in_opd:      Host input pupil OPD `[um]` (used when `@d_in_opd == NULL`).
 * @in_amp:      Host input pupil amplitude (used when `@d_in_opd == NULL`).
 * @in_nlam:     Number of wavelength channels in input beam (`1` = monochromatic broadcast).
 * @in_stride:   Float element stride between input channels (`npix_wf` or `2 * npix_wf`).
 * @out_opd:     Host output post-Lyot OPD `[um]` (`eff_nlam` channels separated by `@out_stride`).
 * @out_amp:     Host output post-Lyot amplitude (`eff_nlam` channels separated by `@out_stride`).
 * @out_stride:  Float element stride between output channels (`npix_wf` or `2 * npix_wf`).
 * @lowfs_opd:   Optional host 2D FPM/Lyot-reflected LOWFS pupil OPD `[npix_wf]` (central channel).
 * @lowfs_amp:   Optional host 2D FPM/Lyot-reflected LOWFS pupil amplitude `[npix_wf]`.
 * @d_in_opd:    Optional device input OPD (when non-NULL with `@d_out_opd`, runs purely on GPU
 *               without PCIe transfers or stream synchronization).
 * @d_in_amp:    Optional device input amplitude.
 * @d_out_opd:   Optional device output post-Lyot OPD.
 * @d_out_amp:   Optional device output post-Lyot amplitude.
 * @d_lowfs_opd: Optional device output LOWFS OPD `[npix_wf]`.
 * @d_lowfs_amp: Optional device output LOWFS amplitude `[npix_wf]`.
 *
 * Role:
 *   Unified hot-path polychromatic dispatcher routing execution to `cacaosim_coro_cuda_step_poly()`
 *   when `ctx->use_cuda` is active or `cacaosim_coro_step_poly_cpu()` otherwise.
 *
 * Rationale:
 *   Accepts both host SHM pointers (for standalone FPS processes) and resident device pointers
 *   (for the fused multi-rate `cacaosim.htsim` GPU pipeline), avoiding PCIe round-trips between
 *   upstream DMs, the coronagraph, and downstream `fpcam` / `zwfs` sensors.
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
 * @ctx:       Initialized coronagraph context.
 * @in_opd:    Host 2D input pupil OPD `[um]` `[npix_wf]` (NULL -> 0).
 * @in_amp:    Host 2D input pupil amplitude `[npix_wf]` (NULL -> 1).
 * @out_opd:   Host 2D output post-Lyot OPD `[um]` `[npix_wf]`.
 * @out_amp:   Host 2D output post-Lyot amplitude `[npix_wf]`.
 * @lowfs_opd: Optional host 2D output LOWFS pupil OPD `[um]` `[npix_wf]`.
 * @lowfs_amp: Optional host 2D output LOWFS pupil amplitude `[npix_wf]`.
 *
 * Role:
 *   Single-channel 2D beam wrapper around `cacaosim_coro_step_poly_cpu()`.
 *
 * Rationale:
 *   When `ctx->eff_nlam > 1` (e.g. a 3D chromatic mask is loaded), propagates into internal
 *   polychromatic scratch buffers and extracts the central spectral slice (`eff_nlam / 2`) so
 *   callers expecting a single 2D `[OPD, Amp]` output buffer never overflow `@out_opd`/`@out_amp`.
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
 * @ctx:         Initialized coronagraph context.
 * @in_opd:      Host 2D input pupil OPD `[um]` `[npix_wf]` (used when `@d_in_opd == NULL`).
 * @in_amp:      Host 2D input pupil amplitude `[npix_wf]`.
 * @out_opd:     Host 2D output post-Lyot OPD `[um]` `[npix_wf]`.
 * @out_amp:     Host 2D output post-Lyot amplitude `[npix_wf]`.
 * @lowfs_opd:   Optional host 2D output LOWFS OPD `[um]` `[npix_wf]`.
 * @lowfs_amp:   Optional host 2D output LOWFS amplitude `[npix_wf]`.
 * @d_in_opd:    Optional device 2D input OPD `[npix_wf]`.
 * @d_in_amp:    Optional device 2D input amplitude `[npix_wf]`.
 * @d_out_opd:   Optional device output OPD (`[eff_nlam * npix_wf]` or `[npix_wf]`).
 * @d_out_amp:   Optional device output amplitude.
 * @d_lowfs_opd: Optional device 2D output LOWFS OPD `[npix_wf]`.
 * @d_lowfs_amp: Optional device 2D output LOWFS amplitude `[npix_wf]`.
 *
 * Role:
 *   Hot-path single-beam convenience entry point for CPU/CUDA callers (`htsim` and unit tests)
 *   supplying a monochromatic 2D input beam.
 *
 * Rationale:
 *   Safely extracts the central spectral channel on host when `ctx->eff_nlam > 1` and `@d_out_opd`
 *   is NULL, while forwarding directly to `cacaosim_coro_step_poly()` for zero-copy device or
 *   monochromatic pipelines.
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

/**
 * cacaosim_coro_cuda_init() - Allocate device buffers, cuBLAS/cuFFT plans, and upload static masks
 * @ctx:        Host-initialized coronagraph context with precomputed MFT/FFT tables.
 * @ext_stream: Optional external `cudaStream_t` (NULL creates a private non-blocking stream).
 *
 * Role:
 *   Internal cold-path CUDA backend initializer invoked by `cacaosim_coro_init_masks()`.
 *
 * Rationale:
 *   Uploads all static per-channel Fourier matrices, complex pupil/Lyot masks, and FPM weights once
 *   at startup so `cacaosim_coro_cuda_step_poly()` executes purely in device memory.
 *
 * Return: 0 on success, -1 on CUDA/cuBLAS/cuFFT allocation or plan failure.
 */
int  cacaosim_coro_cuda_init(cacaosim_coro_ctx_t *ctx, void *ext_stream);

/**
 * cacaosim_coro_cuda_free() - Destroy cuBLAS/cuFFT handles and release device coronagraph memory
 * @ctx: Coronagraph context owning `cuda_priv` (safe when `cuda_priv == NULL`).
 *
 * Role:
 *   Internal cold-path CUDA teardown invoked by `cacaosim_coro_free()` or failed initialization.
 *
 * Rationale:
 *   Destroys `stream` only when `owns_stream != 0` so shared streams owned by `cacaosim.htsim`
 *   remain intact across individual optical element teardowns.
 */
void cacaosim_coro_cuda_free(cacaosim_coro_ctx_t *ctx);

/**
 * cacaosim_coro_cuda_step_poly() - Execute polychromatic coronagraph propagation on CUDA GPU
 * @ctx:         Active coronagraph context with initialized `cuda_priv`.
 * @h_in_opd:    Host input OPD `[um]` (copied H2D when `@d_in_opd == NULL`).
 * @h_in_amp:    Host input amplitude (copied H2D when `@d_in_opd == NULL`).
 * @in_nlam:     Number of input wavelength channels (`1` = broadcast across `eff_nlam`).
 * @in_stride:   Float stride between input channels.
 * @h_out_opd:   Host output post-Lyot OPD `[um]` (copied D2H when `@d_out_opd == NULL`).
 * @h_out_amp:   Host output post-Lyot amplitude (copied D2H when `@d_out_opd == NULL`).
 * @out_stride:  Float stride between output channels.
 * @h_lowfs_opd: Optional host 2D LOWFS OPD `[um]` `[npix_wf]`.
 * @h_lowfs_amp: Optional host 2D LOWFS amplitude `[npix_wf]`.
 * @d_in_opd:    Optional resident device input OPD.
 * @d_in_amp:    Optional resident device input amplitude.
 * @d_out_opd:   Optional resident device output post-Lyot OPD.
 * @d_out_amp:   Optional resident device output post-Lyot amplitude.
 * @d_lowfs_opd: Optional resident device 2D LOWFS OPD.
 * @d_lowfs_amp: Optional resident device 2D LOWFS amplitude.
 *
 * Role:
 *   Internal hot-path GPU propagator driving strided-batched cuBLAS MFTs or per-channel cuFFT 2D
 *   transforms plus fused complex pupil/Lyot/LOWFS CUDA kernels.
 *
 * Rationale:
 *   Supports both standalone host-SHM staging (with `cudaMemcpy2DAsync` strided H2D/D2H transfers
 *   and stream sync) and asynchronous device-resident execution (`dev_mode`) for fused GPU benches.
 */
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

/**
 * cacaosim_coro_reset_long() - Clear cumulative pre-Lyot intensity and LOWFS power accumulators
 */
void cacaosim_coro_reset_long(cacaosim_coro_ctx_t *ctx);

/**
 * cacaosim_coro_readout_telemetry_async() - Stage pre-Lyot intensity maps and energy fractions
 */
void cacaosim_coro_readout_telemetry_async(
    cacaosim_coro_ctx_t *ctx,
    float               *out_pre_lyot,
    float               *out_pre_lyot_long,
    int                  sync_stream);

/**
 * cacaosim_coro_readout_telemetry_finish() - Finalize host power-budget fractions after GPU sync
 */
void cacaosim_coro_readout_telemetry_finish(cacaosim_coro_ctx_t *ctx);

void cacaosim_coro_cuda_reset_long(cacaosim_coro_ctx_t *ctx);
void cacaosim_coro_cuda_readout_telemetry(
    cacaosim_coro_ctx_t *ctx,
    float               *out_pre_lyot,
    float               *out_pre_lyot_long,
    int                  sync_stream);
void cacaosim_coro_cuda_readout_finish(cacaosim_coro_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_CORO_H */
