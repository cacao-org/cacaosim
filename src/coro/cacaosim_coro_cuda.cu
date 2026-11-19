// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_cuda.cu
 * @brief   CUDA cuBLAS StridedBatched MFT + cuFFT polychromatic custom-mask coronagraph backend
 *
 * Architectural Role in cacaosim:
 *   Provides the GPU-accelerated polychromatic optical diffraction backend for `cacaosim.coro`.
 *   Mirrors every optical capability of `cacaosim_coro_engine.c` — custom 2D/3D pupil apodizers,
 *   compact Soummer Babinet Matrix-DFT focal masks (`lyot_spot`, `phase_dimple`, `custom_mft`),
 *   shifted full-plane padded 2D FFT focal masks (`vortex`, `fqpm`, `custom_fft`), complex Lyot
 *   stops, and dual `out_lowfs` pickoffs (`fpm_refl` and `lyot_refl`) — while keeping all static
 *   optical tables and intermediate complex fields resident in GPU VRAM.
 *
 * Design Rationale & GPU Execution Strategy:
 *   - Zero-Transpose Row-Major to Column-Major cuBLAS Mapping:
 *     Host MFT basis matrices (`fwd_U`, `fwd_V`, `inv_U`, `inv_V`) are stored in row-major C
 *     order. Because a row-major `[M x N]` matrix in memory is identical to its `[N x M]`
 *     transpose `M^T` in cuBLAS's native column-major convention, the row-major product
 *     `C = A * B` maps directly to `C^T = B^T * A^T` with `CUBLAS_OP_N` on both operands and
 *     swapped argument order. Using `cublasCgemmStridedBatched`, all `eff_nlam` spectral channels
 *     execute in parallel in just four batched GEMM launches per frame.
 *   - Fused Device-Resident Pipeline vs Host SHM Staging:
 *     When invoked with device pointers (`d_in_opd != NULL && d_out_opd != NULL`) inside the
 *     multi-rate `cacaosim.htsim` bench engine, kernels enqueue asynchronously onto the shared
 *     `cudaStream_t` with zero PCIe copies or host stalls. When invoked from standalone SHM FPS
 *     processes, `cudaMemcpyAsync` / `cudaMemcpy2DAsync` gather and scatter strided `[OPD, Amp]`
 *     slices transparently.
 */

#ifdef HAVE_CUDA

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>

#include "coro/cacaosim_coro.h"

/**
 * struct coro_cuda_state_t - Private device buffers and library handles for `cacaosim.coro`
 * @stream:           Active CUDA stream (private non-blocking or borrowed from `ext_stream`).
 * @owns_stream:      Non-zero if `stream` was created locally and must be destroyed on free.
 * @cublas:           cuBLAS handle bound to `stream` for Babinet MFT `CgemmStridedBatched`.
 * @fft_plan:         2D `CUFFT_C2C` handle of size `nfft x nfft` bound to `stream`.
 * @has_fft:          Non-zero when `fft_plan` has been initialized (`!is_babinet`).
 * @d_lam_um:         Device wavelength array `[eff_nlam]` in `[um]`.
 * @d_lowfs_pup_mask: Device 2D relay pupil stop `[npix_wf]` for `fpm_refl` LOWFS pickoff.
 * @d_pup_mask:       Optional device pre-FPM complex pupil apodizer `[eff_nlam * npix_wf]`.
 * @d_lyot_cplx:      Device post-FPM complex Lyot stop cube `[eff_nlam * npix_wf]`.
 * @d_in_opd:         Staging buffer for host-to-device input OPD `[eff_nlam * npix_wf]`.
 * @d_in_amp:         Staging buffer for host-to-device input amplitude `[eff_nlam * npix_wf]`.
 * @d_out_opd:        Staging buffer for device-to-host output OPD `[eff_nlam * npix_wf]`.
 * @d_out_amp:        Staging buffer for device-to-host output amplitude `[eff_nlam * npix_wf]`.
 * @d_lowfs_opd:      Staging buffer for device-to-host 2D LOWFS OPD `[npix_wf]`.
 * @d_lowfs_amp:      Staging buffer for device-to-host 2D LOWFS amplitude `[npix_wf]`.
 * @d_epup:           Batched complex entrance pupil field `[eff_nlam * npix_wf]`.
 * @d_fwd_U:          Device forward MFT Y-axis matrices `[eff_nlam * ny_mft * nwf]`.
 * @d_fwd_V:          Device forward MFT X-axis matrices `[eff_nlam * nwf * nx_mft]`.
 * @d_inv_U:          Device inverse MFT Y-axis matrices `[eff_nlam * nwf * ny_mft]`.
 * @d_inv_V:          Device inverse MFT X-axis matrices `[eff_nlam * nx_mft * nwf]`.
 * @d_fpm_delta:      Device Babinet core perturbation `T - 1` `[eff_nlam * npix_mft]`.
 * @d_fpm_refl_w:     Device real FPM reflection weights `sqrt(1 - |T|^2)` `[eff_nlam * npix_mft]`.
 * @d_ecore:          Batched inverse-MFT pupil perturbation `[eff_nlam * npix_wf]`.
 * @d_tmp:            Batched intermediate MFT product `[eff_nlam * max(nx_mft, ny_mft) * nwf]`.
 * @d_efoc:           Batched complex MFT focal field `[eff_nlam * npix_mft]`.
 * @d_efoc_r:         Single-channel reflected MFT focal field `[npix_mft]` for `fpm_refl` LOWFS.
 * @d_shift_1d:       Device 1D half-bin pupil shift phasors `[2 * nfft]` (`sh_x` then `sh_y`).
 * @d_fft_mask:       Device pre-normalized (`1/nfft^2`) complex FFT mask `[eff_nlam * npix_fft]`.
 * @d_fft_refl_w:     Device pre-normalized (`1/nfft^2`) FFT LOWFS reflection weight `[npix_fft]`.
 * @d_fft_buf:        In-place `nfft x nfft` complex FFT buffer `[npix_fft]`.
 * @d_fft_lo_buf:     In-place `nfft x nfft` complex FFT buffer `[npix_fft]` for `fpm_refl` LOWFS.
 */
typedef struct
{
    cudaStream_t   stream;
    int            owns_stream;
    cublasHandle_t cublas;
    cufftHandle    fft_plan;
    int            has_fft;

    float         *d_lam_um;
    float         *d_lowfs_pup_mask;
    cuComplex     *d_pup_mask;
    cuComplex     *d_lyot_cplx;
    float         *d_in_opd;
    float         *d_in_amp;
    float         *d_out_opd;
    float         *d_out_amp;
    float         *d_lowfs_opd;
    float         *d_lowfs_amp;
    cuComplex     *d_epup;

    /* Babinet MFT buffers */
    cuComplex     *d_fwd_U;
    cuComplex     *d_fwd_V;
    cuComplex     *d_inv_U;
    cuComplex     *d_inv_V;
    cuComplex     *d_fpm_delta;
    float         *d_fpm_refl_w;
    cuComplex     *d_ecore;
    cuComplex     *d_tmp;
    cuComplex     *d_efoc;
    cuComplex     *d_efoc_r;

    /* Full-plane FFT buffers */
    cuComplex     *d_shift_1d;
    cuComplex     *d_fft_mask;
    float         *d_fft_refl_w;
    cuComplex     *d_fft_buf;
    cuComplex     *d_fft_lo_buf;

    /* Pre-Lyot intensity & optical power-budget telemetry buffers */
    float         *d_pre_u2;
    float         *d_pre_lyot_inst;
    float         *d_pre_lyot_accum;
    float         *d_pre_lyot_norm;
    double        *d_energy_stats;
    double         h_energy_stage[8];
} coro_cuda_state_t;

/**
 * k_coro_build_epup_batch() - Synthesize batched complex entrance pupil across spectral channels
 * @opd:       Device input OPD `[um]` (`in_nlam` slices at `@in_stride`; NULL -> 0).
 * @amp:       Device input amplitude (`in_nlam` slices at `@in_stride`; NULL -> 1).
 * @in_nlam:   Number of input channels (`1` broadcasts slice 0 to all `@eff_nlam` channels).
 * @in_stride: Float stride between consecutive input slices.
 * @pup_mask:  Optional pre-FPM complex pupil apodizer `[eff_nlam * npix_wf]` (may be NULL).
 * @lam_um:    Device channel wavelengths `[eff_nlam]` in `[um]`.
 * @npix_wf:   Pupil pixel count (`nwf * nwf`).
 * @eff_nlam:  Number of output coronagraph channels.
 * @epup:      Output complex pupil field `[eff_nlam * npix_wf]`.
 *
 * Role:
 *   Hot 2D-grid `(pixel, channel)` CUDA kernel converting `[OPD_um, Amplitude]` inputs into
 *   complex pupil phasors `A * exp(i * 2*pi * OPD / lam_c)` multiplied by optional `pup_mask`.
 *
 * Rationale:
 *   Uses hardware fast intrinsics (`__sincosf`) and early-exits on zero-amplitude pixels outside
 *   the clear aperture while preparing all `eff_nlam` slices in a single kernel launch.
 */
__global__ static void k_coro_build_epup_batch(
    const float *__restrict__     opd,
    const float *__restrict__     amp,
    uint32_t                      in_nlam,
    uint64_t                      in_stride,
    const cuComplex *__restrict__ pup_mask,
    const float *__restrict__     lam_um,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    cuComplex *__restrict__       epup)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= eff_nlam)
    {
        return;
    }

    uint32_t ic      = (in_nlam > 1u)
                           ? (((uint32_t) c < in_nlam) ? (uint32_t) c : (in_nlam - 1u))
                           : 0u;
    uint64_t in_idx  = (uint64_t) ic * in_stride + p;
    uint64_t out_idx = (uint64_t) c * npix_wf + p;
    float    a       = (amp != NULL) ? amp[in_idx] : 1.0f;
    if (a <= 0.0f)
    {
        epup[out_idx] = make_cuComplex(0.0f, 0.0f);
        return;
    }

    float phi = ((opd != NULL) ? opd[in_idx] : 0.0f) * (CACAOSIM_TWO_PI_F / lam_um[c]);
    float s   = 0.0f;
    float co  = 1.0f;
    __sincosf(phi, &s, &co);
    float pr = a * co;
    float pi = a * s;

    if (pup_mask != NULL)
    {
        cuComplex pm = pup_mask[out_idx];
        float     nr = pr * pm.x - pi * pm.y;
        float     ni = pr * pm.y + pi * pm.x;
        pr           = nr;
        pi           = ni;
    }
    epup[out_idx] = make_cuComplex(pr, pi);
}

/**
 * k_coro_apply_lyot_batch() - Apply complex Lyot stop directly when `skip_fpm == 1`
 * @epup:       Batched complex entrance pupil field `[eff_nlam * npix_wf]`.
 * @lyot_cplx:  Batched complex Lyot stop cube `[eff_nlam * npix_wf]`.
 * @lam_um:     Device channel wavelengths `[eff_nlam]` in `[um]`.
 * @refl:       Scalar amplitude reflectivity into LOWFS arm.
 * @lo_lyot:    Non-zero when `lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL`.
 * @npix_wf:    Pupil pixel count (`nwf * nwf`).
 * @eff_nlam:   Number of spectral channels.
 * @c_mid:      Central channel index `eff_nlam / 2` used for 2D LOWFS extraction.
 * @out_stride: Float stride between output wavelength slices.
 * @out_opd:    Output post-Lyot OPD `[um]` (`eff_nlam` slices at `@out_stride`).
 * @out_amp:    Output post-Lyot amplitude (`eff_nlam` slices at `@out_stride`).
 * @lowfs_opd:  Optional output 2D LOWFS OPD `[um]` `[npix_wf]`.
 * @lowfs_amp:  Optional output 2D LOWFS amplitude `[npix_wf]`.
 *
 * Role:
 *   Hot FPM-bypass kernel for pure pupil/Lyot apodizers (e.g. Shaped Pupil / vAPP).
 *
 * Rationale:
 *   Multiplies `epup` directly by `lyot_cplx`, extracts `[OPD_um, Amplitude]`, and computes the
 *   reflective-Lyot (`lyot_refl`) LOWFS beam in a single fused memory pass.
 */
__global__ static void k_coro_apply_lyot_batch(
    const cuComplex *__restrict__ epup,
    const cuComplex *__restrict__ lyot_cplx,
    const float *__restrict__     lam_um,
    float                         refl,
    int32_t                       lo_lyot,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    uint64_t                      out_stride,
    uint64_t                      lo_stride,
    float *__restrict__           pre_u2,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= eff_nlam)
    {
        return;
    }

    uint64_t  cidx       = (uint64_t) c * npix_wf + p;
    uint64_t  oidx       = (uint64_t) c * out_stride + p;
    cuComplex u          = epup[cidx];
    cuComplex ly         = lyot_cplx[cidx];
    float     rad_to_opd = lam_um[c] / CACAOSIM_TWO_PI_F;

    pre_u2[cidx]  = u.x * u.x + u.y * u.y;
    float lr      = u.x * ly.x - u.y * ly.y;
    float li      = u.x * ly.y + u.y * ly.x;
    float amp     = hypotf(lr, li);
    out_amp[oidx] = amp;
    out_opd[oidx] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (lowfs_opd != NULL && lowfs_amp != NULL)
    {
        uint64_t loidx = (uint64_t) c * lo_stride + p;
        if (lo_lyot)
        {
            float lya2       = ly.x * ly.x + ly.y * ly.y;
            float r_ly       = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
            float la         = r_ly * hypotf(u.x, u.y);
            lowfs_amp[loidx] = la;
            lowfs_opd[loidx] = (la > 1e-12f) ? (atan2f(u.y, u.x) * rad_to_opd) : 0.0f;
        }
        else
        {
            lowfs_amp[loidx] = 0.0f;
            lowfs_opd[loidx] = 0.0f;
        }
    }
}

/**
 * k_coro_scale_real_copy() - Multiply complex focal field by real FPM reflection weight into `dst`
 * @src:  Input complex focal field `[npix]`.
 * @w:    Real amplitude reflection weight `sqrt(1 - |T|^2)` `[npix]`.
 * @dst:  Output weighted complex focal field `[npix]`.
 * @npix: Number of focal pixels (`npix_mft` or `npix_fft`).
 *
 * Role:
 *   Prepares the reflected focal-plane field `w * E_foc` non-destructively for `fpm_refl` LOWFS
 *   inverse propagation while preserving `@src` for the transmitted science Babinet/FFT branch.
 *
 * Rationale:
 *   Writing to a separate `@dst` buffer avoids re-running the forward MFT/FFT for the science arm.
 */
__global__ static void k_coro_scale_real_copy(
    const cuComplex *__restrict__ src,
    const float *__restrict__     w,
    cuComplex *__restrict__       dst,
    uint64_t                      npix)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix)
    {
        float     wi = w[i];
        cuComplex s  = src[i];
        dst[i]       = make_cuComplex(s.x * wi, s.y * wi);
    }
}

/**
 * k_coro_write_lowfs() - Extract `[OPD_um, Amplitude]` from inverse-MFT LOWFS relay pupil field
 * @ecore:          Complex pupil field inverse-transformed from FPM reflection `[npix]`.
 * @lowfs_pup_mask: Relay circular pupil stop amplitude `[npix]` (may be NULL).
 * @refl:           Scalar LOWFS coating amplitude reflectivity.
 * @rad_to_opd:     Conversion factor `lam_mid / (2 * pi)` `[um / rad]`.
 * @npix:           Pupil pixel count (`npix_wf`).
 * @lowfs_opd:      Output 2D LOWFS OPD `[um]` `[npix]`.
 * @lowfs_amp:      Output 2D LOWFS amplitude `[npix]`.
 *
 * Role:
 *   Finalizes the `fpm_refl` LOWFS pupil output in Babinet MFT mode.
 *
 * Rationale:
 *   Applies both scalar mirror reflectivity `@refl` and the relay pupil stop `@lowfs_pup_mask`
 *   before converting complex phasors into `[OPD_um, Amplitude]`.
 */
__global__ static void k_coro_write_lowfs_batch(
    const cuComplex *__restrict__ ecore,
    const float *__restrict__     lowfs_pup_mask,
    const float *__restrict__     lam_um,
    float                         refl,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    uint64_t                      lo_stride,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p < npix_wf && c < eff_nlam)
    {
        uint64_t  cidx       = (uint64_t) c * npix_wf + p;
        uint64_t  loidx      = (uint64_t) c * lo_stride + p;
        cuComplex ec         = ecore[cidx];
        float     pm         = (lowfs_pup_mask != NULL) ? lowfs_pup_mask[p] : 1.0f;
        float     la         = refl * pm * hypotf(ec.x, ec.y);
        float     rad_to_opd = lam_um[c] / CACAOSIM_TWO_PI_F;
        lowfs_amp[loidx]     = la;
        lowfs_opd[loidx]     = (la > 1e-12f) ? (atan2f(ec.y, ec.x) * rad_to_opd) : 0.0f;
    }
}

/**
 * k_coro_mul_cplx() - In-place elementwise complex multiplication `efoc[i] *= delta[i]`
 * @efoc:  Batched complex MFT focal field `[total]` updated in place.
 * @delta: Batched complex Babinet perturbation `T(u,v,lam) - 1` `[total]`.
 * @total: Total complex elements across all channels (`eff_nlam * npix_mft`).
 *
 * Role:
 *   Applies the polychromatic Babinet focal core mask across all spectral channels in one 1D grid.
 *
 * Rationale:
 *   Flattening `(eff_nlam, npix_mft)` into a single contiguous 1D launch maximizes GPU occupancy
 *   even when `npix_mft` per channel is small (e.g. `48 * 48 = 2304`).
 */
__global__ static void k_coro_mul_cplx(
    cuComplex *__restrict__       efoc,
    const cuComplex *__restrict__ delta,
    uint64_t                      total)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
    {
        cuComplex e = efoc[i];
        cuComplex d = delta[i];
        efoc[i]     = make_cuComplex(e.x * d.x - e.y * d.y, e.x * d.y + e.y * d.x);
    }
}

/**
 * k_coro_babinet_finalize_batch() - Sum `epup + ecore`, apply Lyot stop, and extract OPD/Amp
 * @epup:       Batched complex entrance pupil field `[eff_nlam * npix_wf]`.
 * @ecore:      Batched inverse-MFT Babinet core perturbation `[eff_nlam * npix_wf]`.
 * @lyot_cplx:  Batched complex Lyot stop cube `[eff_nlam * npix_wf]`.
 * @lam_um:     Device channel wavelengths `[eff_nlam]` in `[um]`.
 * @refl:       Scalar amplitude reflectivity for `lyot_refl` LOWFS pickoff.
 * @npix_wf:    Pupil pixel count (`nwf * nwf`).
 * @eff_nlam:   Number of spectral channels.
 * @c_mid:      Central channel index `eff_nlam / 2` for LOWFS extraction.
 * @out_stride: Float stride between output wavelength slices.
 * @out_opd:    Output post-Lyot OPD `[um]` (`eff_nlam` slices at `@out_stride`).
 * @out_amp:    Output post-Lyot amplitude (`eff_nlam` slices at `@out_stride`).
 * @lowfs_opd:  Optional output 2D `lyot_refl` LOWFS OPD `[um]` `[npix_wf]` (NULL when inactive).
 * @lowfs_amp:  Optional output 2D `lyot_refl` LOWFS amplitude `[npix_wf]` (NULL when inactive).
 *
 * Role:
 *   Fused finalization kernel for the GPU Babinet MFT pipeline.
 *
 * Rationale:
 *   Combines Babinet superposition `U = epup + ecore`, complex Lyot filtering `L * U`, polar
 *   `[OPD_um, Amplitude]` conversion, and optional reflective-Lyot (`lyot_refl`) LOWFS pickoff
 *   into one bandwidth-bound kernel pass over `[eff_nlam * npix_wf]`.
 */
__global__ static void k_coro_babinet_finalize_batch(
    const cuComplex *__restrict__ epup,
    const cuComplex *__restrict__ ecore,
    const cuComplex *__restrict__ lyot_cplx,
    const float *__restrict__     lam_um,
    float                         refl,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    uint64_t                      out_stride,
    uint64_t                      lo_stride,
    float *__restrict__           pre_u2,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int32_t  c = (int32_t) blockIdx.y;
    if (p >= npix_wf || c >= eff_nlam)
    {
        return;
    }

    uint64_t  cidx       = (uint64_t) c * npix_wf + p;
    uint64_t  oidx       = (uint64_t) c * out_stride + p;
    cuComplex ep         = epup[cidx];
    cuComplex ec         = ecore[cidx];
    cuComplex ly         = lyot_cplx[cidx];
    float     ur         = ep.x + ec.x;
    float     ui         = ep.y + ec.y;
    float     lr         = ur * ly.x - ui * ly.y;
    float     li         = ur * ly.y + ui * ly.x;
    float     rad_to_opd = lam_um[c] / CACAOSIM_TWO_PI_F;
    float     amp        = hypotf(lr, li);

    pre_u2[cidx]  = ur * ur + ui * ui;
    out_amp[oidx] = amp;
    out_opd[oidx] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (lowfs_opd != NULL && lowfs_amp != NULL)
    {
        uint64_t loidx   = (uint64_t) c * lo_stride + p;
        float    lya2    = ly.x * ly.x + ly.y * ly.y;
        float    r_ly    = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
        float    la      = r_ly * hypotf(ur, ui);
        lowfs_amp[loidx] = la;
        lowfs_opd[loidx] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
    }
}

/**
 * k_coro_fft_embed_epup() - Zero-pad `[nwf x nwf]` pupil into `[nfft x nfft]` with half-bin shift
 * @epup_c:  Single-channel complex entrance pupil `[nwf * nwf]`.
 * @shift_x: 1D X half-bin shift phasor `exp(-i * pi * gx / nfft)` `[nfft]`.
 * @shift_y: 1D Y half-bin shift phasor `exp(-i * pi * gy / nfft)` `[nfft]`.
 * @nwf:     Pupil grid width `[pixels]`.
 * @nfft:    Padded FFT grid width `[pixels]`.
 * @fft_buf: Output complex padded FFT grid `[nfft * nfft]`.
 *
 * Role:
 *   Prepares the zero-padded, phase-ramped input array for forward 2D `cufftExecC2C`.
 *
 * Rationale:
 *   Fuses zero-clearing the outer guard band and center-embedding `epup_c * shift_x * shift_y`
 *   into one coalesced write pass over `nfft * nfft`, eliminating a separate `cudaMemsetAsync`.
 */
__global__ static void k_coro_fft_embed_epup(
    const cuComplex *__restrict__ epup_c,
    const cuComplex *__restrict__ shift_x,
    const cuComplex *__restrict__ shift_y,
    uint32_t                      nwf,
    uint32_t                      nfft,
    cuComplex *__restrict__       fft_buf)
{
    uint64_t idx   = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t) nfft * nfft;
    if (idx >= total)
    {
        return;
    }

    uint32_t gx      = (uint32_t) (idx % nfft);
    uint32_t gy      = (uint32_t) (idx / nfft);
    uint32_t pad_off = (nfft - nwf) / 2u;

    if (gx < pad_off || gx >= pad_off + nwf || gy < pad_off || gy >= pad_off + nwf)
    {
        fft_buf[idx] = make_cuComplex(0.0f, 0.0f);
        return;
    }

    uint32_t  xp = gx - pad_off;
    uint32_t  yp = gy - pad_off;
    cuComplex ep = epup_c[(uint64_t) yp * nwf + xp];
    cuComplex sx = shift_x[gx];
    cuComplex sy = shift_y[gy];
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    fft_buf[idx] = make_cuComplex(ep.x * sr - ep.y * si, ep.x * si + ep.y * sr);
}

/**
 * k_coro_fft_mul_mask() - Multiply complex focal spectrum by pre-normalized FFT coronagraph mask
 * @fft_buf:  In-place complex focal spectrum `[npix_fft]`.
 * @mask:     Pre-normalized (`1/nfft^2`) complex focal mask slice `[npix_fft]`.
 * @npix_fft: Total FFT pixels (`nfft * nfft`).
 *
 * Role:
 *   Applies Vortex, 4QPM, or `CUSTOM_FFT` focal phase/amplitude transmission in the Fourier domain.
 *
 * Rationale:
 *   Because `1 / nfft^2` was baked into `@mask` during `coro_precompute_fft_mask()`, no extra
 *   normalization pass is needed after the subsequent `CUFFT_INVERSE` transform.
 */
__global__ static void k_coro_fft_mul_mask(
    cuComplex *__restrict__       fft_buf,
    const cuComplex *__restrict__ mask,
    uint64_t                      npix_fft)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix_fft)
    {
        cuComplex e = fft_buf[i];
        cuComplex m = mask[i];
        fft_buf[i]  = make_cuComplex(e.x * m.x - e.y * m.y, e.x * m.y + e.y * m.x);
    }
}

/**
 * k_coro_fft_extract_lowfs_fpm() - Crop inverse-FFT reflected field, unshift, and write LOWFS beam
 * @fft_lo_buf:     Inverse-FFT reflected pupil buffer `[nfft * nfft]`.
 * @shift_x:        1D X half-bin forward shift phasor `[nfft]` (conjugated in kernel).
 * @shift_y:        1D Y half-bin forward shift phasor `[nfft]` (conjugated in kernel).
 * @lowfs_pup_mask: Relay circular pupil stop `[nwf * nwf]` (may be NULL).
 * @refl:           Scalar LOWFS coating amplitude reflectivity.
 * @rad_to_opd:     Conversion factor `lam_mid / (2 * pi)` `[um / rad]`.
 * @nwf:            Pupil grid width `[pixels]`.
 * @nfft:           Padded FFT grid width `[pixels]`.
 * @lowfs_opd:      Output 2D LOWFS OPD `[um]` `[nwf * nwf]`.
 * @lowfs_amp:      Output 2D LOWFS amplitude `[nwf * nwf]`.
 *
 * Role:
 *   Extracts the central `[nwf x nwf]` pupil window on the `fpm_refl` LOWFS branch in FFT modes.
 *
 * Rationale:
 *   Multiplies by `conj(shift_x) * conj(shift_y)` to remove the linear pupil phase ramp introduced
 *   by `k_coro_fft_embed_epup()` so downstream LOWFS tip/tilt sensing sees zero artificial tilt.
 */
__global__ static void k_coro_fft_extract_lowfs_fpm(
    const cuComplex *__restrict__ fft_lo_buf,
    const cuComplex *__restrict__ shift_x,
    const cuComplex *__restrict__ shift_y,
    const float *__restrict__     lowfs_pup_mask,
    float                         refl,
    float                         rad_to_opd,
    uint32_t                      nwf,
    uint32_t                      nfft,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p     = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t) nwf * nwf;
    if (p >= total)
    {
        return;
    }

    uint32_t xp      = (uint32_t) (p % nwf);
    uint32_t yp      = (uint32_t) (p / nwf);
    uint32_t pad_off = (nfft - nwf) / 2u;
    uint32_t gx      = xp + pad_off;
    uint32_t gy      = yp + pad_off;

    cuComplex sx = make_cuComplex(shift_x[gx].x, -shift_x[gx].y);
    cuComplex sy = make_cuComplex(shift_y[gy].x, -shift_y[gy].y);
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    cuComplex e    = fft_lo_buf[(uint64_t) gy * nfft + gx];
    float     ur   = e.x * sr - e.y * si;
    float     ui   = e.x * si + e.y * sr;
    float     pm   = (lowfs_pup_mask != NULL) ? lowfs_pup_mask[p] : 1.0f;
    float     la   = refl * pm * hypotf(ur, ui);
    lowfs_amp[p]   = la;
    lowfs_opd[p]   = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
}

/**
 * k_coro_fft_extract_cplx() - Crop inverse-FFT Lyot pupil, unshift, apply Lyot stop, & write OPD
 * @fft_buf:    Inverse-FFT science buffer `[nfft * nfft]`.
 * @shift_x:    1D X half-bin forward shift phasor `[nfft]` (conjugated in kernel).
 * @shift_y:    1D Y half-bin forward shift phasor `[nfft]` (conjugated in kernel).
 * @lyot_c:     Single-channel complex Lyot stop `[nwf * nwf]`.
 * @refl:       Scalar amplitude reflectivity for `lyot_refl` LOWFS pickoff.
 * @rad_to_opd: Conversion factor `lam_c / (2 * pi)` `[um / rad]`.
 * @nwf:        Pupil grid width `[pixels]`.
 * @nfft:       Padded FFT grid width `[pixels]`.
 * @out_opd:    Output channel post-Lyot OPD `[um]` `[nwf * nwf]`.
 * @out_amp:    Output channel post-Lyot amplitude `[nwf * nwf]`.
 * @lowfs_opd:  Optional output 2D `lyot_refl` LOWFS OPD `[um]` `[nwf * nwf]` (NULL otherwise).
 * @lowfs_amp:  Optional output 2D `lyot_refl` LOWFS amplitude `[nwf * nwf]` (NULL otherwise).
 *
 * Role:
 *   Finalizes one spectral channel of the padded-FFT coronagraph pipeline.
 *
 * Rationale:
 *   Crops the central `[nwf x nwf]` pupil sub-grid, cancels the half-bin shift ramp via
 *   `conj(shift_x) * conj(shift_y)`, applies `lyot_c`, and optionally extracts the reflective-Lyot
 *   LOWFS field in one pass.
 */
__global__ static void k_coro_fft_extract_cplx(
    const cuComplex *__restrict__ fft_buf,
    const cuComplex *__restrict__ shift_x,
    const cuComplex *__restrict__ shift_y,
    const cuComplex *__restrict__ lyot_c,
    float                         refl,
    float                         rad_to_opd,
    uint32_t                      nwf,
    uint32_t                      nfft,
    float *__restrict__           pre_u2,
    float *__restrict__           out_opd,
    float *__restrict__           out_amp,
    float *__restrict__           lowfs_opd,
    float *__restrict__           lowfs_amp)
{
    uint64_t p     = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t) nwf * nwf;
    if (p >= total)
    {
        return;
    }

    uint32_t xp      = (uint32_t) (p % nwf);
    uint32_t yp      = (uint32_t) (p / nwf);
    uint32_t pad_off = (nfft - nwf) / 2u;
    uint32_t gx      = xp + pad_off;
    uint32_t gy      = yp + pad_off;

    cuComplex sx = make_cuComplex(shift_x[gx].x, -shift_x[gx].y);
    cuComplex sy = make_cuComplex(shift_y[gy].x, -shift_y[gy].y);
    float     sr = sx.x * sy.x - sx.y * sy.y;
    float     si = sx.x * sy.y + sx.y * sy.x;

    cuComplex e  = fft_buf[(uint64_t) gy * nfft + gx];
    float     ur = e.x * sr - e.y * si;
    float     ui = e.x * si + e.y * sr;
    cuComplex ly = lyot_c[p];

    pre_u2[p]  = ur * ur + ui * ui;
    float lr   = ur * ly.x - ui * ly.y;
    float li   = ur * ly.y + ui * ly.x;
    float amp  = hypotf(lr, li);
    out_amp[p] = amp;
    out_opd[p] = (amp > 1e-12f) ? (atan2f(li, lr) * rad_to_opd) : 0.0f;

    if (lowfs_opd != NULL && lowfs_amp != NULL)
    {
        float lya2   = ly.x * ly.x + ly.y * ly.y;
        float r_ly   = refl * ((lya2 < 1.0f) ? sqrtf(1.0f - lya2) : 0.0f);
        float la     = r_ly * hypotf(ur, ui);
        lowfs_amp[p] = la;
        lowfs_opd[p] = (la > 1e-12f) ? (atan2f(ui, ur) * rad_to_opd) : 0.0f;
    }
}

__global__ static void k_coro_pre_lyot_accum(
    const float *__restrict__ pre_u2,
    uint64_t                  npix_wf,
    int32_t                   eff_nlam,
    float *__restrict__       pre_inst,
    float *__restrict__       pre_accum)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npix_wf)
    {
        return;
    }
    float w_c = 1.0f / (float) ((eff_nlam >= 1) ? eff_nlam : 1);
    float acc = 0.0f;
    for (int32_t c = 0; c < eff_nlam; c++)
    {
        acc += w_c * pre_u2[(uint64_t) c * npix_wf + p];
    }
    pre_inst[p] = acc;
    pre_accum[p] += acc;
}

__global__ static void k_coro_scale_long(
    const float *__restrict__ pre_accum,
    float                     inv_cnt,
    uint64_t                  npix_wf,
    float *__restrict__       pre_norm)
{
    uint64_t p = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p < npix_wf)
    {
        pre_norm[p] = pre_accum[p] * inv_cnt;
    }
}

__global__ static void k_coro_energy_reduce(
    const cuComplex *__restrict__ epup,
    const float *__restrict__     pre_inst,
    const float *__restrict__     out_amp,
    uint64_t                      out_stride,
    const float *__restrict__     lo_amp,
    uint64_t                      lo_stride,
    uint64_t                      npix_wf,
    int32_t                       eff_nlam,
    double *__restrict__          energy_stats)
{
    __shared__ double sh[4][256];
    const double      w_c  = 1.0 / (double) ((eff_nlam >= 1) ? eff_nlam : 1);
    double            s_in = 0.0;
    double            s_pr = 0.0;
    double            s_po = 0.0;
    double            s_lo = 0.0;

    for (uint64_t p = threadIdx.x; p < npix_wf; p += blockDim.x)
    {
        s_pr += (double) pre_inst[p];
        for (int32_t c = 0; c < eff_nlam; c++)
        {
            cuComplex ep = epup[(uint64_t) c * npix_wf + p];
            float     oa = out_amp[(uint64_t) c * out_stride + p];
            s_in += w_c * ((double) ep.x * ep.x + (double) ep.y * ep.y);
            s_po += w_c * ((double) oa * oa);
            if (lo_amp != NULL)
            {
                float la = lo_amp[(uint64_t) c * lo_stride + p];
                s_lo += w_c * ((double) la * la);
            }
        }
    }

    sh[0][threadIdx.x] = s_in;
    sh[1][threadIdx.x] = s_pr;
    sh[2][threadIdx.x] = s_po;
    sh[3][threadIdx.x] = s_lo;
    __syncthreads();

    for (unsigned s = blockDim.x / 2u; s > 0u; s >>= 1u)
    {
        if (threadIdx.x < s)
        {
            sh[0][threadIdx.x] += sh[0][threadIdx.x + s];
            sh[1][threadIdx.x] += sh[1][threadIdx.x + s];
            sh[2][threadIdx.x] += sh[2][threadIdx.x + s];
            sh[3][threadIdx.x] += sh[3][threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u)
    {
        for (int k = 0; k < 4; k++)
        {
            energy_stats[k] = sh[k][0];
            energy_stats[4 + k] += sh[k][0];
        }
    }
}

/**
 * cacaosim_coro_cuda_init() - Allocate device buffers, cuBLAS/cuFFT plans, and upload static masks
 * @ctx:        Host-initialized coronagraph context with precomputed MFT or FFT tables.
 * @ext_stream: Optional external `cudaStream_t` (NULL creates a private non-blocking stream).
 *
 * Role:
 *   Cold-path CUDA backend setup allocating GPU memory for polychromatic pupil/Lyot masks,
 *   staging I/O buffers, and either `cublasHandle_t` + Babinet MFT basis cubes or `cufftHandle` +
 *   2D FFT focal masks.
 *
 * Rationale:
 *   Synchronizes `cs->stream` once at the end of initialization after queueing asynchronous H2D
 *   transfers of all static optical arrays, guaranteeing zero host-to-device mask traffic during
 *   the real-time loop.
 *
 * Return: 0 on success, -1 on CUDA/cuBLAS/cuFFT error (with partial allocations rolled back).
 */
extern "C" int cacaosim_coro_cuda_init(cacaosim_coro_ctx_t *ctx, void *ext_stream)
{
    coro_cuda_state_t *cs = (coro_cuda_state_t *) calloc(1, sizeof(*cs));
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
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cs->owns_stream = 1;
    }

    const size_t nlam     = (size_t) ctx->eff_nlam;
    const size_t wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    const size_t poly_f   = nlam * wf_bytes;
    const size_t poly_c   = nlam * (size_t) ctx->npix_wf * sizeof(cuComplex);

    if (cudaMalloc((void **) &cs->d_lam_um, nlam * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_pup_mask, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lyot_cplx, poly_c) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_in_opd, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_in_amp, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_opd, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_out_amp, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_opd, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_lowfs_amp, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_epup, poly_c) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_pre_u2, poly_f) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_pre_lyot_inst, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_pre_lyot_accum, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_pre_lyot_norm, wf_bytes) != cudaSuccess ||
        cudaMalloc((void **) &cs->d_energy_stats, 8u * sizeof(double)) != cudaSuccess)
    {
        cacaosim_coro_cuda_free(ctx);
        return -1;
    }

    cudaMemsetAsync(cs->d_pre_lyot_accum, 0, wf_bytes, cs->stream);
    cudaMemsetAsync(cs->d_energy_stats, 0, 8u * sizeof(double), cs->stream);

    cudaMemcpyAsync(cs->d_lam_um, ctx->lam_um, nlam * sizeof(float), cudaMemcpyHostToDevice,
                    cs->stream);
    cudaMemcpyAsync(cs->d_lowfs_pup_mask, ctx->lowfs_pup_mask, wf_bytes, cudaMemcpyHostToDevice,
                    cs->stream);
    cudaMemcpyAsync(cs->d_lyot_cplx, ctx->lyot_cplx, poly_c, cudaMemcpyHostToDevice, cs->stream);

    if (ctx->has_pup_mask && ctx->pup_mask_cplx != NULL)
    {
        if (cudaMalloc((void **) &cs->d_pup_mask, poly_c) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_pup_mask, ctx->pup_mask_cplx, poly_c, cudaMemcpyHostToDevice,
                        cs->stream);
    }

    if (ctx->is_babinet && !ctx->skip_fpm)
    {
        if (cublasCreate(&cs->cublas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(cs->cublas, cs->stream) != CUBLAS_STATUS_SUCCESS)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }

        uint32_t max_m   = (ctx->nx_mft > ctx->ny_mft) ? ctx->nx_mft : ctx->ny_mft;
        size_t   u_bytes = nlam * (size_t) ctx->ny_mft * ctx->cfg.nwf * sizeof(cuComplex);
        size_t   v_bytes = nlam * (size_t) ctx->nx_mft * ctx->cfg.nwf * sizeof(cuComplex);
        size_t   t_bytes = nlam * (size_t) max_m * ctx->cfg.nwf * sizeof(cuComplex);
        size_t   mft_c   = nlam * (size_t) ctx->npix_mft * sizeof(cuComplex);
        size_t   mft_f   = nlam * (size_t) ctx->npix_mft * sizeof(float);

        if (cudaMalloc((void **) &cs->d_fwd_U, u_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_inv_U, u_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fwd_V, v_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_inv_V, v_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fpm_delta, mft_c) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fpm_refl_w, mft_f) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_ecore, poly_c) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_tmp, t_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_efoc, mft_c) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_efoc_r, mft_c) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }

        cudaMemcpyAsync(cs->d_fwd_U, ctx->fwd_U_cplx, u_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_inv_U, ctx->inv_U_cplx, u_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_fwd_V, ctx->fwd_V_cplx, v_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_inv_V, ctx->inv_V_cplx, v_bytes, cudaMemcpyHostToDevice, cs->stream);
        cudaMemcpyAsync(cs->d_fpm_delta, ctx->fpm_delta_cplx, mft_c, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fpm_refl_w, ctx->fpm_refl_w, mft_f, cudaMemcpyHostToDevice,
                        cs->stream);
    }
    else if (!ctx->is_babinet)
    {
        size_t sh_bytes  = (size_t) ctx->nfft * 2u * sizeof(cuComplex);
        size_t msk_bytes = nlam * (size_t) ctx->npix_fft * sizeof(cuComplex);
        size_t rw_bytes  = nlam * (size_t) ctx->npix_fft * sizeof(float);
        size_t fft_bytes = (size_t) ctx->npix_fft * sizeof(cuComplex);
        if (cudaMalloc((void **) &cs->d_shift_1d, sh_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_mask, msk_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_refl_w, rw_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_buf, fft_bytes) != cudaSuccess ||
            cudaMalloc((void **) &cs->d_fft_lo_buf, fft_bytes) != cudaSuccess)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cudaMemcpyAsync(cs->d_shift_1d, ctx->shift_1d_cplx, sh_bytes, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fft_mask, ctx->fft_mask_cplx, msk_bytes, cudaMemcpyHostToDevice,
                        cs->stream);
        cudaMemcpyAsync(cs->d_fft_refl_w, ctx->fft_refl_w, rw_bytes, cudaMemcpyHostToDevice,
                        cs->stream);

        int n = (int) ctx->nfft;
        if (cufftPlan2d(&cs->fft_plan, n, n, CUFFT_C2C) != CUFFT_SUCCESS ||
            cufftSetStream(cs->fft_plan, cs->stream) != CUFFT_SUCCESS)
        {
            cacaosim_coro_cuda_free(ctx);
            return -1;
        }
        cs->has_fft = 1;
    }

    cudaStreamSynchronize(cs->stream);
    return 0;
}

/**
 * cacaosim_coro_cuda_free() - Destroy cuBLAS/cuFFT handles and release device coronagraph memory
 * @ctx: Coronagraph context owning `cuda_priv` (safe when `cuda_priv == NULL`).
 *
 * Role:
 *   Cold-path teardown freeing all device buffers, library plans, and private CUDA stream.
 *
 * Rationale:
 *   `cudaFree(NULL)` is a safe no-op in the CUDA runtime, allowing partial allocations from failed
 *   `cacaosim_coro_cuda_init()` calls to be cleaned up safely through a single code path.
 */
extern "C" void cacaosim_coro_cuda_free(cacaosim_coro_ctx_t *ctx)
{
    coro_cuda_state_t *cs = (coro_cuda_state_t *) ctx->cuda_priv;
    if (cs == NULL)
    {
        return;
    }
    if (cs->cublas)
    {
        cublasDestroy(cs->cublas);
    }
    if (cs->has_fft)
    {
        cufftDestroy(cs->fft_plan);
    }
    cudaFree(cs->d_lam_um);
    cudaFree(cs->d_lowfs_pup_mask);
    cudaFree(cs->d_pup_mask);
    cudaFree(cs->d_lyot_cplx);
    cudaFree(cs->d_in_opd);
    cudaFree(cs->d_in_amp);
    cudaFree(cs->d_out_opd);
    cudaFree(cs->d_out_amp);
    cudaFree(cs->d_lowfs_opd);
    cudaFree(cs->d_lowfs_amp);
    cudaFree(cs->d_epup);
    cudaFree(cs->d_fwd_U);
    cudaFree(cs->d_fwd_V);
    cudaFree(cs->d_inv_U);
    cudaFree(cs->d_inv_V);
    cudaFree(cs->d_fpm_delta);
    cudaFree(cs->d_fpm_refl_w);
    cudaFree(cs->d_ecore);
    cudaFree(cs->d_tmp);
    cudaFree(cs->d_efoc);
    cudaFree(cs->d_efoc_r);
    cudaFree(cs->d_shift_1d);
    cudaFree(cs->d_fft_mask);
    cudaFree(cs->d_fft_refl_w);
    cudaFree(cs->d_fft_buf);
    cudaFree(cs->d_fft_lo_buf);
    cudaFree(cs->d_pre_u2);
    cudaFree(cs->d_pre_lyot_inst);
    cudaFree(cs->d_pre_lyot_accum);
    cudaFree(cs->d_pre_lyot_norm);
    cudaFree(cs->d_energy_stats);
    if (cs->owns_stream && cs->stream)
    {
        cudaStreamDestroy(cs->stream);
    }
    free(cs);
    ctx->cuda_priv = NULL;
}

/**
 * cacaosim_coro_cuda_step_poly() - Execute polychromatic coronagraph propagation on CUDA GPU
 * @ctx:         Active coronagraph context with initialized `cuda_priv`.
 * @h_in_opd:    Host input OPD `[um]` (used when `!dev_mode`).
 * @h_in_amp:    Host input amplitude (used when `!dev_mode`).
 * @in_nlam:     Number of input wavelength channels (`1` = monochromatic broadcast).
 * @in_stride:   Float stride between input channels.
 * @h_out_opd:   Host output post-Lyot OPD `[um]` (populated when `!dev_mode`).
 * @h_out_amp:   Host output post-Lyot amplitude (populated when `!dev_mode`).
 * @out_stride:  Float stride between output channels.
 * @h_lowfs_opd: Optional host 2D LOWFS OPD `[um]` `[npix_wf]`.
 * @h_lowfs_amp: Optional host 2D LOWFS amplitude `[npix_wf]`.
 * @d_in_opd:    Optional resident device input OPD (`dev_mode` when paired with `@d_out_opd`).
 * @d_in_amp:    Optional resident device input amplitude.
 * @d_out_opd:   Optional resident device output post-Lyot OPD.
 * @d_out_amp:   Optional resident device output post-Lyot amplitude.
 * @d_lowfs_opd: Optional resident device 2D LOWFS OPD.
 * @d_lowfs_amp: Optional resident device 2D LOWFS amplitude.
 *
 * Role:
 *   Hot-path GPU propagator orchestrating optional strided H2D staging, batched pupil construction,
 *   Babinet `cublasCgemmStridedBatched` MFT or per-channel `cufftExecC2C` 2D FFT diffraction,
 *   `fpm_refl` / `lyot_refl` LOWFS pickoffs, and optional strided D2H writeback.
 *
 * Rationale:
 *   Uses `cudaMemcpy2DAsync` whenever host SHM cubes interleave `[OPD_c, Amp_c]` slices
 *   (`in_stride == 2 * npix_wf`), packing slices contiguously into `cs->d_in_opd`/`cs->d_in_amp`
 *   in a single DMA transaction without per-channel host loops.
 */
extern "C" void cacaosim_coro_cuda_step_poly(
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
    coro_cuda_state_t *cs       = (coro_cuda_state_t *) ctx->cuda_priv;
    const uint64_t     npix_wf  = ctx->npix_wf;
    const int32_t      eff_nlam = ctx->eff_nlam;
    const size_t       wf_bytes = (size_t) npix_wf * sizeof(float);
    const int          dev_mode = (d_in_opd != NULL && d_out_opd != NULL) ? 1 : 0;

    uint32_t copy_in_nlam = (in_nlam >= 1u)
                                ? ((in_nlam <= (uint32_t) eff_nlam) ? in_nlam : (uint32_t) eff_nlam)
                                : 1u;

    const float *din_opd       = d_in_opd;
    const float *din_amp       = d_in_amp;
    uint64_t     dev_in_stride = in_stride;
    float       *dout_o        = (d_out_opd != NULL) ? d_out_opd : cs->d_out_opd;
    float       *dout_a        = (d_out_amp != NULL) ? d_out_amp : cs->d_out_amp;
    uint64_t     dev_out_str   = (d_out_opd != NULL) ? out_stride : npix_wf;
    int          want_lo       = (d_lowfs_opd != NULL || h_lowfs_opd != NULL) ? 1 : 0;
    float       *dlo_o = want_lo ? ((d_lowfs_opd != NULL) ? d_lowfs_opd : cs->d_lowfs_opd) : NULL;
    float       *dlo_a = want_lo ? ((d_lowfs_amp != NULL) ? d_lowfs_amp : cs->d_lowfs_amp) : NULL;
    uint64_t     dev_lo_str    = (d_lowfs_opd != NULL) ? out_stride : npix_wf;

    /* Stage host SHM slices to contiguous GPU buffers when not running purely device-resident */
    if (!dev_mode)
    {
        dev_in_stride = npix_wf;
        if (h_in_opd != NULL)
        {
            if (copy_in_nlam == 1u || in_stride == npix_wf)
            {
                cudaMemcpyAsync(cs->d_in_opd, h_in_opd, (size_t) copy_in_nlam * wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(cs->d_in_opd, wf_bytes, h_in_opd,
                                  (size_t) in_stride * sizeof(float), wf_bytes, copy_in_nlam,
                                  cudaMemcpyHostToDevice, cs->stream);
            }
            din_opd = cs->d_in_opd;
        }
        if (h_in_amp != NULL)
        {
            if (copy_in_nlam == 1u || in_stride == npix_wf)
            {
                cudaMemcpyAsync(cs->d_in_amp, h_in_amp, (size_t) copy_in_nlam * wf_bytes,
                                cudaMemcpyHostToDevice, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(cs->d_in_amp, wf_bytes, h_in_amp,
                                  (size_t) in_stride * sizeof(float), wf_bytes, copy_in_nlam,
                                  cudaMemcpyHostToDevice, cs->stream);
            }
            din_amp = cs->d_in_amp;
        }
    }

    const int32_t  lo_lyot = (ctx->cfg.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? 1 : 0;
    const float    refl    = (ctx->cfg.lowfs_refl >= 0.0f) ? ctx->cfg.lowfs_refl : 0.95f;
    const uint32_t bl_wf   = (uint32_t) ((npix_wf + 255ULL) / 256ULL);
    dim3           grd_wf(bl_wf, (unsigned int) eff_nlam, 1);

    k_coro_build_epup_batch<<<grd_wf, 256, 0, cs->stream>>>(
        din_opd, din_amp, copy_in_nlam, dev_in_stride, cs->d_pup_mask, cs->d_lam_um, npix_wf,
        eff_nlam, cs->d_epup);

    if (ctx->skip_fpm)
    {
        k_coro_apply_lyot_batch<<<grd_wf, 256, 0, cs->stream>>>(
            cs->d_epup, cs->d_lyot_cplx, cs->d_lam_um, refl, lo_lyot, npix_wf, eff_nlam,
            dev_out_str, dev_lo_str, cs->d_pre_u2, dout_o, dout_a, dlo_o, dlo_a);
    }
    else if (ctx->is_babinet)
    {
        const int           nwf       = (int) ctx->cfg.nwf;
        const int           mx        = (int) ctx->nx_mft;
        const int           my        = (int) ctx->ny_mft;
        const int           max_m     = (mx > my) ? mx : my;
        const cuComplex     one       = make_cuComplex(1.0f, 0.0f);
        const cuComplex     zero      = make_cuComplex(0.0f, 0.0f);
        const long long int str_ep    = (long long int) npix_wf;
        const long long int str_u     = (long long int) my * nwf;
        const long long int str_v     = (long long int) mx * nwf;
        const long long int str_t     = (long long int) max_m * nwf;
        const long long int str_ef    = (long long int) ctx->npix_mft;
        const uint64_t      total_mft = (uint64_t) eff_nlam * ctx->npix_mft;
        const uint32_t      bl_mft    = (uint32_t) ((total_mft + 255ULL) / 256ULL);

        /*
         * Batched forward 2D MFT via column-major transpose identity (A*B)^T = B^T * A^T:
         *   Row-major tmp[my x nwf] = fwd_U[my x nwf] * epup[nwf x nwf]
         *   Row-major efoc[my x mx] = tmp[my x nwf]   * fwd_V[nwf x mx]
         */
        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, my, nwf, &one,
                                  cs->d_epup, nwf, str_ep, cs->d_fwd_U, nwf, str_u, &zero,
                                  cs->d_tmp, nwf, str_t, eff_nlam);
        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, mx, my, nwf, &one,
                                  cs->d_fwd_V, mx, str_v, cs->d_tmp, nwf, str_t, &zero, cs->d_efoc,
                                  mx, str_ef, eff_nlam);

        /* Batched fpm_refl LOWFS inverse MFT: ecore = inv_U * (w * efoc) * inv_V */
        if (want_lo && !lo_lyot && dlo_o != NULL && dlo_a != NULL)
        {
            k_coro_scale_real_copy<<<bl_mft, 256, 0, cs->stream>>>(
                cs->d_efoc, cs->d_fpm_refl_w, cs->d_efoc_r, total_mft);
            cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, mx, nwf, my, &one,
                                      cs->d_efoc_r, mx, str_ef, cs->d_inv_U, my, str_u, &zero,
                                      cs->d_tmp, mx, str_t, eff_nlam);
            cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, mx, &one,
                                      cs->d_inv_V, nwf, str_v, cs->d_tmp, mx, str_t, &zero,
                                      cs->d_ecore, nwf, str_ep, eff_nlam);
            k_coro_write_lowfs_batch<<<grd_wf, 256, 0, cs->stream>>>(
                cs->d_ecore, cs->d_lowfs_pup_mask, cs->d_lam_um, refl, npix_wf, eff_nlam,
                dev_lo_str, dlo_o, dlo_a);
        }

        /* Apply Babinet focal perturbation (T - 1) and batched inverse 2D MFT back to pupil */
        k_coro_mul_cplx<<<bl_mft, 256, 0, cs->stream>>>(cs->d_efoc, cs->d_fpm_delta, total_mft);

        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, mx, nwf, my, &one,
                                  cs->d_efoc, mx, str_ef, cs->d_inv_U, my, str_u, &zero, cs->d_tmp,
                                  mx, str_t, eff_nlam);
        cublasCgemmStridedBatched(cs->cublas, CUBLAS_OP_N, CUBLAS_OP_N, nwf, nwf, mx, &one,
                                  cs->d_inv_V, nwf, str_v, cs->d_tmp, mx, str_t, &zero,
                                  cs->d_ecore, nwf, str_ep, eff_nlam);

        k_coro_babinet_finalize_batch<<<grd_wf, 256, 0, cs->stream>>>(
            cs->d_epup, cs->d_ecore, cs->d_lyot_cplx, cs->d_lam_um, refl, npix_wf, eff_nlam,
            dev_out_str, dev_lo_str, cs->d_pre_u2, dout_o, dout_a,
            (want_lo && lo_lyot) ? dlo_o : NULL, (want_lo && lo_lyot) ? dlo_a : NULL);
    }
    else
    {
        /* Per-channel padded 2D cuFFT pipeline (Vortex, 4QPM, and CUSTOM_FFT) */
        const uint32_t   bl_fft = (uint32_t) ((ctx->npix_fft + 255ULL) / 256ULL);
        const cuComplex *sh_x   = cs->d_shift_1d;
        const cuComplex *sh_y   = cs->d_shift_1d + ctx->nfft;

        for (int32_t c = 0; c < eff_nlam; c++)
        {
            int    do_lo  = (want_lo && dlo_o != NULL && dlo_a != NULL) ? 1 : 0;
            float *lo_o_c = (do_lo && lo_lyot) ? (dlo_o + (size_t) c * dev_lo_str) : NULL;
            float *lo_a_c = (do_lo && lo_lyot) ? (dlo_a + (size_t) c * dev_lo_str) : NULL;
            float  r2o_c  = ctx->lam_um[c] / CACAOSIM_TWO_PI_F;

            k_coro_fft_embed_epup<<<bl_fft, 256, 0, cs->stream>>>(
                cs->d_epup + (size_t) c * npix_wf, sh_x, sh_y, ctx->cfg.nwf, ctx->nfft,
                cs->d_fft_buf);
            cufftExecC2C(cs->fft_plan, cs->d_fft_buf, cs->d_fft_buf, CUFFT_FORWARD);

            if (do_lo && !lo_lyot)
            {
                k_coro_scale_real_copy<<<bl_fft, 256, 0, cs->stream>>>(
                    cs->d_fft_buf, cs->d_fft_refl_w + (size_t) c * ctx->npix_fft, cs->d_fft_lo_buf,
                    ctx->npix_fft);
                cufftExecC2C(cs->fft_plan, cs->d_fft_lo_buf, cs->d_fft_lo_buf, CUFFT_INVERSE);
                k_coro_fft_extract_lowfs_fpm<<<bl_wf, 256, 0, cs->stream>>>(
                    cs->d_fft_lo_buf, sh_x, sh_y, cs->d_lowfs_pup_mask, refl, r2o_c, ctx->cfg.nwf,
                    ctx->nfft, dlo_o + (size_t) c * dev_lo_str, dlo_a + (size_t) c * dev_lo_str);
            }

            k_coro_fft_mul_mask<<<bl_fft, 256, 0, cs->stream>>>(
                cs->d_fft_buf, cs->d_fft_mask + (size_t) c * ctx->npix_fft, ctx->npix_fft);
            cufftExecC2C(cs->fft_plan, cs->d_fft_buf, cs->d_fft_buf, CUFFT_INVERSE);
            k_coro_fft_extract_cplx<<<bl_wf, 256, 0, cs->stream>>>(
                cs->d_fft_buf, sh_x, sh_y, cs->d_lyot_cplx + (size_t) c * npix_wf, refl, r2o_c,
                ctx->cfg.nwf, ctx->nfft, cs->d_pre_u2 + (size_t) c * npix_wf,
                dout_o + (size_t) c * dev_out_str, dout_a + (size_t) c * dev_out_str, lo_o_c,
                lo_a_c);
        }
    }

    /* Accumulate pre-Lyot intensity map and exact FP64 optical power budget on GPU */
    k_coro_pre_lyot_accum<<<bl_wf, 256, 0, cs->stream>>>(
        cs->d_pre_u2, npix_wf, eff_nlam, cs->d_pre_lyot_inst, cs->d_pre_lyot_accum);
    k_coro_energy_reduce<<<1, 256, 0, cs->stream>>>(
        cs->d_epup, cs->d_pre_lyot_inst, dout_a, dev_out_str, want_lo ? dlo_a : NULL, dev_lo_str,
        npix_wf, eff_nlam, cs->d_energy_stats);
    ctx->long_cnt++;

    /* Scatter GPU outputs back to host SHM buffers and synchronize when in host staging mode */
    if (!dev_mode)
    {
        if (h_out_opd != NULL)
        {
            if (eff_nlam == 1 || out_stride == npix_wf)
            {
                cudaMemcpyAsync(h_out_opd, dout_o, (size_t) eff_nlam * wf_bytes,
                                cudaMemcpyDeviceToHost, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(h_out_opd, (size_t) out_stride * sizeof(float), dout_o, wf_bytes,
                                  wf_bytes, (size_t) eff_nlam, cudaMemcpyDeviceToHost, cs->stream);
            }
        }
        if (h_out_amp != NULL)
        {
            if (eff_nlam == 1 || out_stride == npix_wf)
            {
                cudaMemcpyAsync(h_out_amp, dout_a, (size_t) eff_nlam * wf_bytes,
                                cudaMemcpyDeviceToHost, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(h_out_amp, (size_t) out_stride * sizeof(float), dout_a, wf_bytes,
                                  wf_bytes, (size_t) eff_nlam, cudaMemcpyDeviceToHost, cs->stream);
            }
        }
        if (h_lowfs_opd != NULL && dlo_o != NULL)
        {
            if (eff_nlam == 1 || out_stride == npix_wf)
            {
                cudaMemcpyAsync(h_lowfs_opd, dlo_o, (size_t) eff_nlam * wf_bytes,
                                cudaMemcpyDeviceToHost, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(h_lowfs_opd, (size_t) out_stride * sizeof(float), dlo_o,
                                  wf_bytes, wf_bytes, (size_t) eff_nlam, cudaMemcpyDeviceToHost,
                                  cs->stream);
            }
        }
        if (h_lowfs_amp != NULL && dlo_a != NULL)
        {
            if (eff_nlam == 1 || out_stride == npix_wf)
            {
                cudaMemcpyAsync(h_lowfs_amp, dlo_a, (size_t) eff_nlam * wf_bytes,
                                cudaMemcpyDeviceToHost, cs->stream);
            }
            else
            {
                cudaMemcpy2DAsync(h_lowfs_amp, (size_t) out_stride * sizeof(float), dlo_a,
                                  wf_bytes, wf_bytes, (size_t) eff_nlam, cudaMemcpyDeviceToHost,
                                  cs->stream);
            }
        }
        if (ctx->cpu_pre_lyot_int != NULL)
        {
            cudaMemcpyAsync(ctx->cpu_pre_lyot_int, cs->d_pre_lyot_inst, wf_bytes,
                            cudaMemcpyDeviceToHost, cs->stream);
        }
        cudaMemcpyAsync(cs->h_energy_stage, cs->d_energy_stats, 8u * sizeof(double),
                        cudaMemcpyDeviceToHost, cs->stream);
        cudaStreamSynchronize(cs->stream);
        cacaosim_coro_cuda_readout_finish(ctx);
    }
}

extern "C" void cacaosim_coro_cuda_reset_long(cacaosim_coro_ctx_t *ctx)
{
    coro_cuda_state_t *cs = (ctx != NULL) ? (coro_cuda_state_t *) ctx->cuda_priv : NULL;
    if (cs == NULL)
    {
        return;
    }
    cudaMemsetAsync(cs->d_pre_lyot_accum, 0, (size_t) ctx->npix_wf * sizeof(float), cs->stream);
    cudaMemsetAsync(cs->d_energy_stats, 0, 8u * sizeof(double), cs->stream);
}

extern "C" void cacaosim_coro_cuda_readout_telemetry(
    cacaosim_coro_ctx_t *ctx,
    float               *out_pre_lyot,
    float               *out_pre_lyot_long,
    int                  sync_stream)
{
    coro_cuda_state_t *cs = (ctx != NULL) ? (coro_cuda_state_t *) ctx->cuda_priv : NULL;
    if (cs == NULL)
    {
        return;
    }
    const size_t   wf_bytes = (size_t) ctx->npix_wf * sizeof(float);
    const uint32_t bl_wf    = (uint32_t) ((ctx->npix_wf + 255ULL) / 256ULL);

    cudaMemcpyAsync(cs->h_energy_stage, cs->d_energy_stats, 8u * sizeof(double),
                    cudaMemcpyDeviceToHost, cs->stream);
    if (out_pre_lyot != NULL)
    {
        cudaMemcpyAsync(out_pre_lyot, cs->d_pre_lyot_inst, wf_bytes, cudaMemcpyDeviceToHost,
                        cs->stream);
    }
    if (out_pre_lyot_long != NULL)
    {
        float inv = (ctx->long_cnt > 0ULL) ? (1.0f / (float) ctx->long_cnt) : 0.0f;
        k_coro_scale_long<<<bl_wf, 256, 0, cs->stream>>>(
            cs->d_pre_lyot_accum, inv, ctx->npix_wf, cs->d_pre_lyot_norm);
        cudaMemcpyAsync(out_pre_lyot_long, cs->d_pre_lyot_norm, wf_bytes, cudaMemcpyDeviceToHost,
                        cs->stream);
    }
    if (sync_stream)
    {
        cudaStreamSynchronize(cs->stream);
        cacaosim_coro_cuda_readout_finish(ctx);
    }
}

extern "C" void cacaosim_coro_cuda_readout_finish(cacaosim_coro_ctx_t *ctx)
{
    coro_cuda_state_t *cs = (ctx != NULL) ? (coro_cuda_state_t *) ctx->cuda_priv : NULL;
    if (cs == NULL)
    {
        return;
    }
    double e_in   = cs->h_energy_stage[0];
    double e_pre  = cs->h_energy_stage[1];
    double e_post = cs->h_energy_stage[2];
    double e_lo   = cs->h_energy_stage[3];
    double s_in   = cs->h_energy_stage[4];
    double s_pre  = cs->h_energy_stage[5];
    double s_post = cs->h_energy_stage[6];
    double s_lo   = cs->h_energy_stage[7];

    ctx->sum_e_in             = s_in;
    ctx->sum_e_pre_lyot       = s_pre;
    ctx->sum_e_post_lyot      = s_post;
    ctx->sum_e_lowfs          = s_lo;
    double den_i              = (e_in > 1e-30) ? e_in : 1.0;
    double den_l              = (s_in > 1e-30) ? s_in : 1.0;
    ctx->last_frac_pre_lyot   = (float) (e_pre / den_i);
    ctx->last_frac_lyot_trans = (float) (e_post / den_i);
    ctx->last_frac_lowfs      = (float) (e_lo / den_i);
    ctx->long_frac_pre_lyot   = (float) (s_pre / den_l);
    ctx->long_frac_lyot_trans = (float) (s_post / den_l);
    ctx->long_frac_lowfs      = (float) (s_lo / den_l);
}

#endif /* HAVE_CUDA */
