// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam_engine.c
 * @brief   CPU (OpenBLAS MFT + FFTW3f) and host orchestration for focal-plane PSF camera
 *
 * Architectural Role in `cacaosim`:
 *   Implements the CPU Fraunhofer diffraction backend and unified host/GPU lifecycle and
 *   stepping entry points for the polychromatic focal-plane science camera (`cacaosim.fpcam`).
 *   Supports cold precomputation (`cacaosim_fpcam_init`), teardown (`cacaosim_fpcam_free`),
 *   multi-subframe polychromatic optical intensity integration (`cacaosim_fpcam_accum_subframe`,
 *   `cacaosim_fpcam_accum_beam_poly`, `cacaosim_fpcam_accum_beam`), detector readout with
 *   instantaneous and long-exposure Strehl telemetry (`cacaosim_fpcam_readout`), and single-call
 *   stepping (`cacaosim_fpcam_step`). Dispatches transparently to `cacaosim_fpcam_cuda.cu` when
 *   CUDA acceleration is active (`ctx->use_cuda == 1`).
 *
 * Design Rationale & Optical Physics Motivation:
 *   1. Separable Soummer Matrix-DFT (`PROP_MFT`) vs Zero-Padded 2D FFT (`PROP_FFT`):
 *      - High-contrast PSF and dark-hole imaging typically require a fine, non-integer detector
 *        plate scale (`pixscale_ld`, e.g., `2.5..5.0` pixels per `lambda0 / D`) over a moderate
 *        field (`ncam = 64..256`). Soummer et al. (2007) express the 2D Fraunhofer integral as a
 *        separable complex matrix product `Efoc = U_c * Epup * V_c`, evaluated via two level-3
 *        BLAS `cblas_cgemm` calls per spectral slice (`O(ncam * nwf^2 + ncam^2 * nwf)`) with zero
 *        FFT padding overhead or interpolation ringing.
 *      - Wide-field halo cameras (`ncam >= 512`) instead select `PROP_FFT`, embedding `Epup` in a
 *        `nfft = nwf * pad_factor` grid (`FFTW3f` 2D C2C forward DFT) for `O(N^2 log N)` scaling.
 *   2. Chromatic Fixed-Detector Plate Scale (`PROP_MFT`) vs Fixed-Grid FFT (`PROP_FFT`):
 *      - On a physical detector with fixed pixel pitch `p_det`, diffraction units expand linearly
 *        with wavelength, giving `sc = pixscale_ld * (lam_c / lam0)` pixels per `lambda_c / D`.
 *        Precomputing `U_c` and `V_c` with `sc` per channel reproduces exact radial chromatic
 *        speckle elongation across broad science filters without image-plane warping.
 *      - In `PROP_FFT`, `nfft / dpix` pixels per `lambda_c / D` is constant across channels; DC
 *        at `(0, 0)` is wrapped modularly onto `(ncam / 2, ncam / 2)` without a pupil phase ramp
 *        because only phase-invariant intensity `|Efoc|^2` is accumulated.
 *   3. Dynamic Per-Frame Pupil Amplitude & OPD:
 *      - Downstream of `cacaosim.coro`, post-Lyot pupil amplitude and phase evolve every AO step
 *        and vary chromatically across `[0 .. beam_nlam - 1]`. Rebuilding `Epup_c` from dynamic
 *        `[OPD_um, Amplitude]` slices on every subframe faithfully propagates residual stellar
 *        leakage, Lyot diffraction, and speckle pinning.
 *   4. Dual Clear-Pupil Strehl + Infinite-Plane Parseval Photometric Calibration:
 *      - Normalizing `U_c` and `V_c` each by `1 / sqrt(pup_sum_amp)` (and FFT intensities by
 *        `1 / pup_sum_amp^2`) places accumulated optical images directly in **Strehl ratio units**
 *        (`[0..1]`, where `1.0` equals the unaberrated clear-pupil on-axis peak).
 *      - Precomputing `inf_flux_norm` via discrete Parseval's theorem maps `flux_ph` (total
 *        clear-pupil incident photons) to photoelectrons per unit Strehl without summing the
 *        cropped `ncam x ncam` window, preserving absolute stellar/coronagraphic throughput even
 *        when outer Airy rings fall outside the sensor or a coronagraph blocks `99.99%` of light.
 *   5. OpenBLAS vs OpenMP Thread Coordination:
 *      - Weak-symbol `openblas_set_num_threads(1)` keeps OpenBLAS single-threaded so its internal
 *        pthread pool never contends with OpenMP (`fpcam_build_epup_cpu` enables OpenMP only when
 *        `npix_wf >= 16384`, i.e., `nwf >= 128`) or sibling real-time `cacao` control threads.
 */

#define _GNU_SOURCE
#include <cblas.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "milk_config.h"
#include "milk_types.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "fpcam/cacaosim_fpcam.h"

/*
 * Optional weak symbol resolved at link/load time when linked against OpenBLAS; NULL when linked
 * against reference Netlib CBLAS, Intel MKL, or Apple Accelerate.
 */
extern void openblas_set_num_threads(int num_threads) __attribute__((weak));

/**
 * fpcam_precompute_mft() - Precompute polychromatic 1D Soummer Matrix-DFT basis matrices U and V
 * @ctx: Focal-plane camera context with `cfg`, `eff_nlam`, `lam_um`, `pup_sum_amp`, and allocated
 *       `mft_U_cplx` / `mft_V_cplx` buffers
 *
 * Role: Cold initialization helper invoked once by `cacaosim_fpcam_init()` when `prop_mode` is
 * `CACAOSIM_FPCAM_PROP_MFT` to populate all `[eff_nlam]` spectral slices of `mft_U_cplx`
 * (`[ncam x nwf]` row/Y transform) and `mft_V_cplx` (`[nwf x ncam]` column/X transform).
 *
 * Rationale: Factoring the 2D Fraunhofer integral over normalized pupil coordinates
 * `(xt, yt) in [-D/2, +D/2]` (in telescope diameters `dpix = pup_diam * nwf`) and focal
 * coordinates `(xi, eta)` (in `lambda_c / D` units centered on camera pixel `f0 = ncam / 2`)
 * yields two small 1D Fourier kernels per channel. Scaling the angular sampling per channel as
 * `sc = ps0 * (lam_um[c] / lam0)` pixels per `lambda_c / D` models a physical fixed-pitch
 * detector array where longer wavelengths produce proportionally wider diffraction rings and
 * radially elongated speckles. Multiplying both `U` and `V` by `nrm = 1 / sqrt(pup_sum_amp)`
 * divides the 2D product `Efoc = U * Epup * V` by `pup_sum_amp`, guaranteeing that an unaberrated
 * clear pupil (`Epup = pup_amp`) produces an on-axis intensity peak `|Efoc(f0, f0)|^2 == 1.0`
 * directly in Strehl ratio units.
 */
static void fpcam_precompute_mft(cacaosim_fpcam_ctx_t *ctx)
{
    const uint32_t nwf  = ctx->cfg.nwf;
    const uint32_t ncam = ctx->cfg.ncam;
    const float    lam0 = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 1.65f;
    const float    dpix = ctx->cfg.pup_diam * (float) nwf;
    /* Symmetric pixel-centered pupil origin `(nwf - 1)/2`; integer camera DC pixel `ncam / 2` */
    const float    p0   = 0.5f * (float) (nwf - 1u);
    const float    f0   = (float) (ncam / 2u);
    /* Split `1 / pup_sum_amp` equally across U and V so `|U * pup_amp * V|^2` peaks at 1.0 */
    const float    nrm  = 1.0f / sqrtf(ctx->pup_sum_amp);
    const float    ps0  = (ctx->cfg.pixscale_ld > 0.05f) ? ctx->cfg.pixscale_ld : 2.5f;

    for (int32_t c = 0; c < ctx->eff_nlam; c++)
    {
        /* Fixed physical pixel pitch -> `sc` pixels per (lambda_c / D) grows linearly with lam_c */
        const float sc = ps0 * (ctx->lam_um[c] / lam0);
        float      *U  = &ctx->mft_U_cplx[(size_t) c * ncam * nwf * 2u];
        float      *V  = &ctx->mft_V_cplx[(size_t) c * nwf * ncam * 2u];

        /* Row-transform U_c[yf, yp] = nrm * exp(-i * 2*pi * yt * eta) */
        for (uint32_t yf = 0; yf < ncam; yf++)
        {
            const float eta = ((float) yf - f0) / sc;
            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                const float yt  = ((float) yp - p0) / dpix;
                const float ang = -CACAOSIM_TWO_PI_F * yt * eta;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);
                size_t idx  = ((size_t) yf * nwf + yp) * 2u;
                U[idx]      = nrm * co;
                U[idx + 1u] = nrm * s;
            }
        }

        /* Column-transform V_c[xp, xf] = nrm * exp(-i * 2*pi * xt * xi) */
        for (uint32_t xp = 0; xp < nwf; xp++)
        {
            const float xt = ((float) xp - p0) / dpix;
            for (uint32_t xf = 0; xf < ncam; xf++)
            {
                const float xi  = ((float) xf - f0) / sc;
                const float ang = -CACAOSIM_TWO_PI_F * xt * xi;
                float       s   = 0.0f;
                float       co  = 1.0f;
                sincosf(ang, &s, &co);
                size_t idx  = ((size_t) xp * ncam + xf) * 2u;
                V[idx]      = nrm * co;
                V[idx + 1u] = nrm * s;
            }
        }
    }
}

/**
 * fpcam_build_epup_cpu() - Synthesize 2D complex pupil phasor field for a single spectral slice
 * @pup_amp:    Pupil or dynamic beam amplitude `[npix_wf]`, or NULL for unit amplitude (`1.0`)
 * @atm_amp:    Optional atmospheric scintillation amplitude multiplier `[npix_wf]`, or NULL
 * @wf_in:      Optional input wavefront error or beam OPD `[npix_wf]`, or NULL
 * @dm_opd:     Optional deformable-mirror OPD correction `[npix_wf]` [um] (subtracted), or NULL
 * @ncpa_opd:   Optional static science-arm NCPA OPD `[npix_wf]` [um] (added), or NULL
 * @wf_to_opd:  Conversion scale factor mapping `wf_in` units to OPD [um]
 * @opd_to_rad: Monochromatic wavenumber `2 * pi / lam_um[c]` [rad/um]
 * @npix_wf:    Total number of pupil grid pixels (`nwf * nwf`)
 * @epup_cplx:  Output interleaved `[Re, Im]` complex pupil field `[2 * npix_wf]`
 *
 * Role: Hot-loop CPU helper called once per spectral channel per subframe to assemble
 * `Epup(p) = amp(p) * exp(i * (2*pi / lam_c) * (wf_to_opd * wf_in - dm_opd + ncpa_opd))`.
 *
 * Rationale: Combining static/dynamic pupil amplitude, scintillation, atmospheric turbulence,
 * DM correction, and science-arm NCPA in a single streaming pass maximizes L1/L2 cache reuse and
 * skips `sincosf` outside the illuminated pupil (`amp <= 0`). OpenMP threading is gated on
 * `npix_wf >= 16384` (`nwf >= 128`) so small grids avoid fork/join barrier overhead.
 */
static void MILK_HOT fpcam_build_epup_cpu(
    const float *__restrict pup_amp,
    const float *__restrict atm_amp,
    const float *__restrict wf_in,
    const float *__restrict dm_opd,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd,
    float                   opd_to_rad,
    uint64_t                npix_wf,
    float *__restrict       epup_cplx)
{
    const int nth = (npix_wf >= 16384ULL) ? omp_get_max_threads() : 1;

#pragma omp parallel for num_threads(nth) if (nth > 1) schedule(static)
    for (uint64_t p = 0; p < npix_wf; p++)
    {
        float amp = (pup_amp != NULL) ? pup_amp[p] : 1.0f;
        if (atm_amp != NULL)
        {
            amp *= atm_amp[p];
        }
        /* Short-circuit masked pixels outside pupil/Lyot support to skip transcendental math */
        if (amp <= 0.0f)
        {
            epup_cplx[2u * p]      = 0.0f;
            epup_cplx[2u * p + 1u] = 0.0f;
            continue;
        }
        float opd = 0.0f;
        if (wf_in != NULL)
        {
            opd += wf_to_opd * wf_in[p];
        }
        if (dm_opd != NULL)
        {
            opd -= dm_opd[p];
        }
        if (ncpa_opd != NULL)
        {
            opd += ncpa_opd[p];
        }
        float phi = opd * opd_to_rad;
        float s   = 0.0f;
        float co  = 1.0f;
        sincosf(phi, &s, &co);
        epup_cplx[2u * p]      = amp * co;
        epup_cplx[2u * p + 1u] = amp * s;
    }
}

/**
 * fpcam_accum_poly_cpu() - Propagate and incoherently sum polychromatic PSF channels on CPU
 * @ctx:         Initialized focal-plane camera context
 * @pup_amp:     Host pupil/beam amplitude slices (`[beam_nlam]` at `chan_stride`), or NULL
 * @atm_amp:     Optional achromatic atmospheric amplitude `[npix_wf]`, or NULL
 * @wf_in:       Optional wavefront/beam OPD slices (`[beam_nlam]` at `chan_stride`), or NULL
 * @beam_nlam:   Number of spectral slices in `pup_amp` / `wf_in` (`1` = achromatic)
 * @chan_stride: Float element stride between adjacent input spectral slices
 * @dm_opd:      Optional DM OPD `[npix_wf]` [um], or NULL
 * @ncpa_opd:    Optional static NCPA OPD `[npix_wf]` [um], or NULL
 * @wf_to_opd:   Scale factor mapping `wf_in` units to OPD [um]
 *
 * Role: Core hot-path CPU propagator that evaluates Fraunhofer diffraction across all `eff_nlam`
 * spectral channels for one optical subframe and accumulates `(1 / eff_nlam) * |Efoc_c|^2` into
 * `ctx->cpu_accum_int` in Strehl-normalized units.
 *
 * Rationale: Supports both monochrome/achromatic wavefront inputs (`beam_nlam == 1`) and
 * multi-wavelength post-coronagraphic `[OPD, Amp]` cubes (`beam_nlam >= 1`) via proportional
 * slice mapping `bc = (c * beam_nlam) / eff_nlam`. In `PROP_MFT` mode, two `cblas_cgemm` calls
 * compute `Efoc = U_c * Epup * V_c` directly onto the `[ncam x ncam]` grid. In `PROP_FFT` mode,
 * `Epup` is zero-padded into `[nfft x nfft]`, transformed via `fftwf_execute`, and cropped
 * modularly around index `(0, 0) -> (ncam / 2, ncam / 2)` without needing an `fftshift` phase
 * ramp because `|F(kx, ky)|^2` is invariant to pupil origin translation.
 */
static void MILK_HOT fpcam_build_epup_poly_cpu(
    const float *__restrict beam_amp,
    const float *__restrict atm_amp,
    const float *__restrict beam_opd,
    uint32_t                m0,
    uint32_t                m1,
    float                   alpha,
    uint64_t                chan_stride,
    const float *__restrict dm_opd,
    const float *__restrict ncpa_opd,
    float                   wf_to_opd,
    float                   opd_to_rad,
    uint64_t                npix_wf,
    float *__restrict       epup_cplx)
{
    if (m1 == m0 || alpha <= 0.0f)
    {
        const float *pa = (beam_amp != NULL) ? (beam_amp + (uint64_t) m0 * chan_stride) : NULL;
        const float *po = (beam_opd != NULL) ? (beam_opd + (uint64_t) m0 * chan_stride) : NULL;
        fpcam_build_epup_cpu(pa, atm_amp, po, dm_opd, ncpa_opd, wf_to_opd, opd_to_rad, npix_wf,
                             epup_cplx);
        return;
    }

    const int    nth = (npix_wf >= 16384ULL) ? omp_get_max_threads() : 1;
    const float  w1  = alpha;
    const float  w0  = 1.0f - w1;
    const float *a0  = (beam_amp != NULL) ? (beam_amp + (uint64_t) m0 * chan_stride) : NULL;
    const float *a1  = (beam_amp != NULL) ? (beam_amp + (uint64_t) m1 * chan_stride) : NULL;
    const float *o0  = (beam_opd != NULL) ? (beam_opd + (uint64_t) m0 * chan_stride) : NULL;
    const float *o1  = (beam_opd != NULL) ? (beam_opd + (uint64_t) m1 * chan_stride) : NULL;

#pragma omp parallel for num_threads(nth) if (nth > 1) schedule(static)
    for (uint64_t p = 0; p < npix_wf; p++)
    {
        float at   = (atm_amp != NULL) ? atm_amp[p] : 1.0f;
        float amp0 = ((a0 != NULL) ? a0[p] : 1.0f) * at;
        float amp1 = ((a1 != NULL) ? a1[p] : 1.0f) * at;
        if (amp0 <= 0.0f && amp1 <= 0.0f)
        {
            epup_cplx[2u * p]      = 0.0f;
            epup_cplx[2u * p + 1u] = 0.0f;
            continue;
        }
        float common = ((ncpa_opd != NULL) ? ncpa_opd[p] : 0.0f) -
                       ((dm_opd != NULL) ? dm_opd[p] : 0.0f);
        float p0  = (((o0 != NULL) ? (wf_to_opd * o0[p]) : 0.0f) + common) * opd_to_rad;
        float p1  = (((o1 != NULL) ? (wf_to_opd * o1[p]) : 0.0f) + common) * opd_to_rad;
        float s0  = 0.0f;
        float co0 = 1.0f;
        float s1  = 0.0f;
        float co1 = 1.0f;
        sincosf(p0, &s0, &co0);
        sincosf(p1, &s1, &co1);
        epup_cplx[2u * p]      = w0 * amp0 * co0 + w1 * amp1 * co1;
        epup_cplx[2u * p + 1u] = w0 * amp0 * s0 + w1 * amp1 * s1;
    }
}

static void fpcam_resolve_poly_bracket(
    const cacaosim_fpcam_ctx_t *ctx,
    int32_t                     c,
    uint32_t                    beam_nlam,
    uint32_t                   *m0,
    uint32_t                   *m1,
    float                      *alpha)
{
    if (beam_nlam <= 1u)
    {
        *m0    = 0u;
        *m1    = 0u;
        *alpha = 0.0f;
        return;
    }
    const float *in_lam = NULL;
    if (ctx->cfg.has_in_lam)
    {
        in_lam = ctx->cfg.in_lam_um;
    }
    else if (ctx->cfg.has_ext_lam && beam_nlam == (uint32_t) ctx->eff_nlam)
    {
        in_lam = ctx->lam_um;
    }
    float lam0   = (ctx->cfg.lambda_um > 0.05f) ? ctx->cfg.lambda_um : 1.65f;
    float bw     = (ctx->cfg.bw_frac > 0.0f) ? ctx->cfg.bw_frac : 0.20f;
    float span   = 0.5f * bw * ((float) (beam_nlam - 1u) / (float) beam_nlam);
    float def_l0 = lam0 * (1.0f - span);
    float def_l1 = lam0 * (1.0f + span);
    cacaosim_poly_interp_bracket(ctx->lam_um[c], in_lam, beam_nlam, def_l0, def_l1, m0, m1, alpha);
}

static void MILK_HOT fpcam_accum_poly_cpu(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *dm_opd,
    const float          *ncpa_opd,
    float                 wf_to_opd)
{
    const uint32_t nwf      = ctx->cfg.nwf;
    const uint32_t ncam     = ctx->cfg.ncam;
    const uint64_t npix_cam = ctx->npix_cam;
    float *__restrict acc   = MILK_ASSUME_ALIGNED(ctx->cpu_accum_int);

    if (ctx->cfg.prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        const float c_one[2]  = { 1.0f, 0.0f };
        const float c_zero[2] = { 0.0f, 0.0f };

        for (int32_t c = 0; c < ctx->eff_nlam; c++)
        {
            uint32_t m0 = 0u;
            uint32_t m1 = 0u;
            float    al = 0.0f;
            fpcam_resolve_poly_bracket(ctx, c, beam_nlam, &m0, &m1, &al);

            fpcam_build_epup_poly_cpu(pup_amp, atm_amp, wf_in, m0, m1, al, chan_stride, dm_opd,
                                      ncpa_opd, wf_to_opd, ctx->k_rad[c], ctx->npix_wf,
                                      ctx->cpu_epup_cplx);

            const float *Uc = &ctx->mft_U_cplx[(size_t) c * ncam * nwf * 2u];
            const float *Vc = &ctx->mft_V_cplx[(size_t) c * nwf * ncam * 2u];
            const float  wc = ctx->strehl_w[c];

            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) ncam, (int) nwf,
                        (int) nwf, c_one, Uc, (int) nwf, ctx->cpu_epup_cplx, (int) nwf, c_zero,
                        ctx->cpu_mft_tmp_cplx, (int) nwf);

            cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) ncam, (int) ncam,
                        (int) nwf, c_one, ctx->cpu_mft_tmp_cplx, (int) nwf, Vc, (int) ncam,
                        c_zero, ctx->cpu_efoc_cplx, (int) ncam);

            const float *__restrict ef = MILK_ASSUME_ALIGNED(ctx->cpu_efoc_cplx);
#pragma omp simd
            for (uint64_t i = 0; i < npix_cam; i++)
            {
                float re = ef[2u * i];
                float im = ef[2u * i + 1u];
                acc[i] += wc * (re * re + im * im);
            }
        }
    }
    else
    {
        /*
         * Zero-padded 2D FFT with 2D bilinear lambda/D rescaling around bore-sight `(ncam/2,
         * ncam/2)` so geometric tilts stay achromatic while diffraction rings scale by `lam_c /
         * lam0` identically to `PROP_MFT`.
         */
        const uint32_t nfft   = ctx->nfft;
        const int32_t  nf     = (int32_t) nfft;
        const float    inv_s2 = 1.0f / (ctx->pup_sum_amp * ctx->pup_sum_amp);
        const int32_t  half_c = (int32_t) (ncam / 2u);

        for (int32_t c = 0; c < ctx->eff_nlam; c++)
        {
            uint32_t m0 = 0u;
            uint32_t m1 = 0u;
            float    al = 0.0f;
            fpcam_resolve_poly_bracket(ctx, c, beam_nlam, &m0, &m1, &al);

            fpcam_build_epup_poly_cpu(pup_amp, atm_amp, wf_in, m0, m1, al, chan_stride, dm_opd,
                                      ncpa_opd, wf_to_opd, ctx->k_rad[c], ctx->npix_wf,
                                      ctx->cpu_epup_cplx);

            __builtin_memset(ctx->cpu_fft_in_cplx, 0, ctx->npix_fft * 2u * sizeof(float));
            for (uint32_t yp = 0; yp < nwf; yp++)
            {
                __builtin_memcpy(&ctx->cpu_fft_in_cplx[(size_t) yp * nfft * 2u],
                                 &ctx->cpu_epup_cplx[(size_t) yp * nwf * 2u],
                                 (size_t) nwf * 2u * sizeof(float));
            }

            fftwf_execute((fftwf_plan) ctx->fftw_plan_fwd);

            const float *__restrict fo = ctx->cpu_fft_out_cplx;
            const float             sc = ctx->scale_c[c];
            const float             wc = ctx->strehl_w[c] * inv_s2;

            for (uint32_t yf = 0; yf < ncam; yf++)
            {
                int32_t  dy   = (int32_t) yf - half_c;
                float    sy_f = (float) dy * sc;
                int32_t  iy0  = (int32_t) floorf(sy_f);
                float    fy   = sy_f - (float) iy0;
                uint32_t ky0  = (uint32_t) ((iy0 % nf + nf) % nf);
                uint32_t ky1  = (uint32_t) (((iy0 + 1) % nf + nf) % nf);
                const float *r0 = &fo[(size_t) ky0 * nfft * 2u];
                const float *r1 = &fo[(size_t) ky1 * nfft * 2u];

                for (uint32_t xf = 0; xf < ncam; xf++)
                {
                    int32_t  dx   = (int32_t) xf - half_c;
                    float    sx_f = (float) dx * sc;
                    int32_t  ix0  = (int32_t) floorf(sx_f);
                    float    fx   = sx_f - (float) ix0;
                    uint32_t kx0  = (uint32_t) ((ix0 % nf + nf) % nf);
                    uint32_t kx1  = (uint32_t) (((ix0 + 1) % nf + nf) % nf);

                    float i00 = r0[2u * kx0] * r0[2u * kx0] + r0[2u * kx0 + 1u] * r0[2u * kx0 + 1u];
                    float i10 = r0[2u * kx1] * r0[2u * kx1] + r0[2u * kx1 + 1u] * r0[2u * kx1 + 1u];
                    float i01 = r1[2u * kx0] * r1[2u * kx0] + r1[2u * kx0 + 1u] * r1[2u * kx0 + 1u];
                    float i11 = r1[2u * kx1] * r1[2u * kx1] + r1[2u * kx1 + 1u] * r1[2u * kx1 + 1u];

                    float v0 = (1.0f - fx) * i00 + fx * i10;
                    float v1 = (1.0f - fx) * i01 + fx * i11;
                    acc[(size_t) yf * ncam + xf] += wc * ((1.0f - fy) * v0 + fy * v1);
                }
            }
        }
    }
}

/**
 * fpcam_accum_cpu() - Convenience wrapper for single-slice achromatic CPU PSF accumulation
 * @ctx:       Initialized focal-plane camera context
 * @pup_amp:   Pupil amplitude `[npix_wf]`, or NULL
 * @atm_amp:   Optional atmospheric amplitude `[npix_wf]`, or NULL
 * @wf_in:     Optional wavefront map `[npix_wf]`, or NULL
 * @dm_opd:    Optional DM OPD `[npix_wf]` [um], or NULL
 * @ncpa_opd:  Optional static NCPA OPD `[npix_wf]` [um], or NULL
 * @wf_to_opd: Scale factor mapping `wf_in` units to OPD [um]
 *
 * Role: Forwards classical single-plane `(pup_amp, atm_amp, wf_in, dm_opd)` inputs to
 * `fpcam_accum_poly_cpu()` with `beam_nlam = 1`.
 *
 * Rationale: Keeps a single unified diffraction code path for both classical AO wavefront streams
 * and polychromatic `[OPD, Amp]` optical beam cubes.
 */
static void MILK_HOT fpcam_accum_cpu(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    float                 wf_to_opd)
{
    fpcam_accum_poly_cpu(ctx, pup_amp, atm_amp, wf_in, 1u, ctx->npix_wf, dm_opd, ncpa_opd,
                         wf_to_opd);
}

/**
 * cacaosim_fpcam_init() - Precompute MFT/FFTW plans, Parseval norm, and optional CUDA context
 * @ctx:        Caller-allocated focal-plane camera context to initialize
 * @cfg:        Immutable geometric, spectral, propagation, and baseline detector configuration
 * @pup_amp:    Static clear reference pupil amplitude `[nwf * nwf]` (required, non-NULL)
 * @ncpa_opd:   Optional static science-arm NCPA OPD `[nwf * nwf]` [um], or NULL
 * @ext_stream: Optional borrowed `cudaStream_t` (`void *`) for fused pipelines, or NULL
 *
 * Role: Cold setup routine invoked once prior to real-time execution. Computes clear-pupil
 * amplitude integrals (`pup_sum_amp`, `pup_sum_amp2`), resolves the polychromatic wavelength grid,
 * derives the infinite-plane Parseval photometric norm (`inf_flux_norm`), allocates 64-byte
 * aligned CPU buffers and PRNG pools, precomputes `mft_U_cplx`/`mft_V_cplx` or the `fftwf` 2D
 * forward plan, and initializes the CUDA backend when requested and available.
 *
 * Rationale: Isolating all heap allocations, trigonometric MFT basis construction, FFTW planning,
 * and `cudaMalloc` calls into `cacaosim_fpcam_init()` guarantees deterministic zero-allocation
 * real-time subframes. Computing `pup_sum_amp` and `inf_flux_norm = mean_c((dpix * sc)^2 * sum_a2
 * / sum_a^2)` from the clear entrance pupil `pup_amp` anchors both Strehl ratios (`1.0` = clear
 * diffraction peak) and photon calibration (`flux_ph` = total incident clear-pupil photons) to a
 * physical, field-of-view-independent reference even when `in_beam` is attenuated by an upstream
 * coronagraph or cropped by a small `ncam` detector window.
 *
 * Return: `0` on success, `-1` on invalid geometry/empty pupil or resource allocation failure.
 */
int MILK_COLD cacaosim_fpcam_init(
    cacaosim_fpcam_ctx_t       *ctx,
    const cacaosim_fpcam_cfg_t *cfg,
    const float                *pup_amp,
    const float                *ncpa_opd,
    void                       *ext_stream)
{
    if (ctx == NULL || cfg == NULL || pup_amp == NULL || cfg->nwf < 8 || cfg->ncam < 4)
    {
        return -1;
    }

    __builtin_memset(ctx, 0, sizeof(*ctx));
    ctx->cfg      = *cfg;
    uint32_t pad  = (cfg->pad_factor >= 2u) ? cfg->pad_factor : 2u;
    ctx->nfft     = cfg->nwf * pad;
    ctx->npix_wf  = (uint64_t) cfg->nwf * cfg->nwf;
    ctx->npix_fft = (uint64_t) ctx->nfft * ctx->nfft;
    ctx->npix_cam = (uint64_t) cfg->ncam * cfg->ncam;

    /* Clear-pupil L1 (`sum_a` -> Strehl peak) and L2 (`sum_a2` -> Parseval total energy) norms */
    double sum_a  = 0.0;
    double sum_a2 = 0.0;
    for (uint64_t p = 0; p < ctx->npix_wf; p++)
    {
        double a = (double) pup_amp[p];
        sum_a += a;
        sum_a2 += a * a;
    }
    if (sum_a <= 1e-12)
    {
        return -1;
    }
    ctx->pup_sum_amp  = (float) sum_a;
    ctx->pup_sum_amp2 = (float) sum_a2;

    /* Resolve spectral channel centers and photon weights from explicit tables or bandpass */
    float   lam0 = (cfg->lambda_um > 0.05f) ? cfg->lambda_um : 1.65f;
    float   bw   = (cfg->bw_frac > 0.0f) ? cfg->bw_frac : 0.0f;
    int32_t nlam = cacaosim_resolve_nlambda(cfg->nlambda, 1u, bw);
    if (nlam > CACAOSIM_FPCAM_MAX_LAMBDA)
    {
        nlam = CACAOSIM_FPCAM_MAX_LAMBDA;
    }

    float init_w[CACAOSIM_FPCAM_MAX_LAMBDA];
    if (cfg->has_ext_lam && nlam >= 1)
    {
        ctx->eff_nlam = nlam;
        for (int32_t c = 0; c < nlam; c++)
        {
            ctx->lam_um[c] = (cfg->ext_lam_um[c] > 0.05f) ? cfg->ext_lam_um[c] : lam0;
            init_w[c]      = cfg->has_ext_weight ? cfg->ext_weight[c] : 1.0f;
        }
    }
    else
    {
        if (bw <= 1e-6f)
        {
            nlam = 1;
        }
        ctx->eff_nlam = nlam;
        cacaosim_spec_sample_band(NULL, lam0, bw, nlam, 0.001f, 1.0f, ctx->lam_um, init_w, NULL);
        if (cfg->has_ext_weight)
        {
            for (int32_t c = 0; c < nlam; c++)
            {
                init_w[c] = cfg->ext_weight[c];
            }
        }
    }

    for (int32_t c = 0; c < nlam; c++)
    {
        ctx->scale_c[c] = lam0 / ctx->lam_um[c];
        ctx->k_rad[c]   = CACAOSIM_TWO_PI_F / ctx->lam_um[c];
    }

    float dpix = cfg->pup_diam * (float) cfg->nwf;
    if (cfg->prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        ctx->eff_pixscale_ld = (cfg->pixscale_ld > 0.05f) ? cfg->pixscale_ld : 2.5f;
    }
    else
    {
        ctx->eff_pixscale_ld = (float) ctx->nfft / dpix;
    }

    /*
     * Dual Strehl + Infinite-Plane Parseval calibration:
     * Base clear-pupil Strehl integral at `lam0` is `E0 = (dpix * ps0)^2 * sum_a2 / sum_a^2`.
     * Weighting channel `c` by `strehl_w[c] = w_c * (lam0/lam_c)^2 / sum_j(w_j * (lam0/lam_j)^2)`
     * and setting `inf_flux_norm = E0 / sum_j(w_j * (lam0/lam_j)^2)` keeps broadband clear-pupil
     * peak identically `1.0` while depositing exact photon fraction `w_c` per sub-band.
     */
    double n_axis0     = (double) dpix * (double) ctx->eff_pixscale_ld;
    ctx->base_inf_norm = (float) ((n_axis0 * n_axis0 * sum_a2) / (sum_a * sum_a));
    cacaosim_fpcam_set_weights(ctx, init_w);

    ctx->cpu_epup_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_wf * 2u, sizeof(float));
    ctx->cpu_sub_int   = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_accum_int = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_exp_int   = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));
    ctx->cpu_long_int  = (float *) cacaosim_aligned_calloc(ctx->npix_cam, sizeof(float));

    if (!ctx->cpu_epup_cplx || !ctx->cpu_sub_int || !ctx->cpu_accum_int || !ctx->cpu_exp_int ||
        !ctx->cpu_long_int || cacaosim_det_pool_init(&ctx->det_pool, 0, cfg->rng_seed) != 0)
    {
        cacaosim_fpcam_free(ctx);
        return -1;
    }

    if (openblas_set_num_threads != NULL)
    {
        /* Avoid OpenBLAS pthread pool colliding with OpenMP threads on small 128^2 CGEMMs */
        openblas_set_num_threads(1);
    }

    if (cfg->prop_mode == CACAOSIM_FPCAM_PROP_MFT)
    {
        size_t uv_elems       = (size_t) nlam * cfg->ncam * cfg->nwf * 2u;
        ctx->mft_U_cplx       = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->mft_V_cplx       = (float *) cacaosim_aligned_calloc(uv_elems, sizeof(float));
        ctx->cpu_mft_tmp_cplx = (float *) cacaosim_aligned_calloc(
            (size_t) cfg->ncam * cfg->nwf * 2u, sizeof(float));
        ctx->cpu_efoc_cplx = (float *) cacaosim_aligned_calloc(ctx->npix_cam * 2u, sizeof(float));
        if (!ctx->mft_U_cplx || !ctx->mft_V_cplx || !ctx->cpu_mft_tmp_cplx || !ctx->cpu_efoc_cplx)
        {
            cacaosim_fpcam_free(ctx);
            return -1;
        }
        fpcam_precompute_mft(ctx);
    }
    else
    {
        ctx->cpu_fft_in_cplx =
            (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        ctx->cpu_fft_out_cplx =
            (float *) cacaosim_aligned_calloc(ctx->npix_fft * 2u, sizeof(float));
        if (!ctx->cpu_fft_in_cplx || !ctx->cpu_fft_out_cplx)
        {
            cacaosim_fpcam_free(ctx);
            return -1;
        }
        int n = (int) ctx->nfft;
        ctx->fftw_plan_fwd = (void *) fftwf_plan_dft_2d(
            n, n, (fftwf_complex *) ctx->cpu_fft_in_cplx, (fftwf_complex *) ctx->cpu_fft_out_cplx,
            FFTW_FORWARD, FFTW_ESTIMATE);
        if (ctx->fftw_plan_fwd == NULL)
        {
            cacaosim_fpcam_free(ctx);
            return -1;
        }
    }

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        if (cacaosim_fpcam_cuda_init(ctx, pup_amp, ncpa_opd, ext_stream) == 0)
        {
            ctx->use_cuda = 1;
        }
    }

    return 0;
}

/**
 * cacaosim_fpcam_free() - Release all CPU, FFTW, PRNG, and CUDA resources owned by @ctx
 * @ctx: Focal-plane camera context to tear down (safe on NULL or partially initialized @ctx)
 *
 * Role: Cold teardown counterpart to `cacaosim_fpcam_init()` that destroys GPU handles, FFTW3f
 * plans, detector PRNG state, and aligned host buffers before zeroing `@ctx`.
 *
 * Rationale: Safe to call from partial-init error paths inside `cacaosim_fpcam_init()` as well as
 * normal FPS loop shutdown without risking double-frees or leaked device allocations.
 */
void MILK_COLD cacaosim_fpcam_free(cacaosim_fpcam_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->use_cuda || ctx->cuda_priv != NULL)
    {
        cacaosim_fpcam_cuda_free(ctx);
    }
    if (ctx->fftw_plan_fwd)
    {
        fftwf_destroy_plan((fftwf_plan) ctx->fftw_plan_fwd);
    }
    cacaosim_det_pool_free(&ctx->det_pool);
    free(ctx->mft_U_cplx);
    free(ctx->mft_V_cplx);
    free(ctx->cpu_epup_cplx);
    free(ctx->cpu_mft_tmp_cplx);
    free(ctx->cpu_efoc_cplx);
    free(ctx->cpu_fft_in_cplx);
    free(ctx->cpu_fft_out_cplx);
    free(ctx->cpu_sub_int);
    free(ctx->cpu_accum_int);
    free(ctx->cpu_exp_int);
    free(ctx->cpu_long_int);
    __builtin_memset(ctx, 0, sizeof(*ctx));
}

/**
 * cacaosim_fpcam_reset_long() - Clear cumulative long-exposure PSF buffer and Strehl counter
 * @ctx: Focal-plane camera context (safe on NULL)
 *
 * Role: Resets `cpu_long_int`, `long_cnt`, and `long_strehl_peak` to zero without disturbing any
 * in-progress short-exposure `subframe_cnt` integration or GPU plans.
 *
 * Rationale: Allows operators and automated AO/coronagraphic calibration scripts to restart
 * long-exposure science PSF and Strehl integration cleanly the instant an AO loop closes or dark
 * hole digging converges, via the live `.reset_long` FPS parameter.
 */
void cacaosim_fpcam_reset_long(cacaosim_fpcam_ctx_t *ctx)
{
    if (ctx == NULL || ctx->cpu_long_int == NULL)
    {
        return;
    }
    __builtin_memset(ctx->cpu_long_int, 0, ctx->npix_cam * sizeof(float));
    ctx->long_cnt         = 0;
    ctx->long_strehl_peak = 0.0f;
}

void cacaosim_fpcam_set_weights(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *weights)
{
    if (ctx == NULL || weights == NULL || ctx->eff_nlam < 1)
    {
        return;
    }
    int32_t nlam = ctx->eff_nlam;
    double  wsum = 0.0;
    for (int32_t c = 0; c < nlam; c++)
    {
        wsum += (weights[c] > 0.0f) ? (double) weights[c] : 0.0;
    }
    double jsum = 0.0;
    for (int32_t c = 0; c < nlam; c++)
    {
        float nw      = (wsum > 1e-20)
                            ? (float) (((weights[c] > 0.0f) ? (double) weights[c] : 0.0) / wsum)
                            : (1.0f / (float) nlam);
        ctx->chan_w[c] = nw;
        double sc      = (double) ctx->scale_c[c];
        jsum += (double) nw * sc * sc;
    }
    if (jsum <= 1e-20)
    {
        jsum = 1.0;
    }
    int changed = 0;
    for (int32_t c = 0; c < nlam; c++)
    {
        double sc = (double) ctx->scale_c[c];
        float  sw = (float) (((double) ctx->chan_w[c] * sc * sc) / jsum);
        if (fabsf(sw - ctx->strehl_w[c]) > 1e-6f)
        {
            changed = 1;
        }
        ctx->strehl_w[c] = sw;
    }
    ctx->inf_flux_norm = (float) ((double) ctx->base_inf_norm / jsum);
    if (changed && ctx->use_cuda)
    {
        cacaosim_fpcam_cuda_upload_weights(ctx);
    }
}

/**
 * cacaosim_fpcam_accum_subframe() - Propagate and accumulate one classical-AO optical subframe
 * @ctx:       Initialized focal-plane camera context
 * @pup_amp:   Host static pupil amplitude `[nwf * nwf]` (used on CPU; GPU uses resident copy)
 * @atm_amp:   Optional host atmospheric scintillation amplitude `[nwf * nwf]`, or NULL
 * @wf_in:     Optional host atmospheric wavefront map `[nwf * nwf]`, or NULL
 * @dm_opd:    Optional host DM OPD map `[nwf * nwf]` [um], or NULL
 * @ncpa_opd:  Optional host static NCPA OPD `[nwf * nwf]` [um] (CPU path; GPU uses resident copy)
 * @d_dm_opd:  Optional resident device DM OPD `[nwf * nwf]` [um] for zero-copy GPU benches, or NULL
 * @turb_gain: Live atmospheric turbulence multiplier (`0.0` bypasses `wf_in`)
 *
 * Role: Hot-loop classical-AO accumulator invoked once per simulation tick when `cacaosim.fpcam`
 * operates directly from `(pup_amp, atm_amp, wf_in, dm_opd)` streams rather than `in_beam`.
 * Lazily zeroes the exposure accumulator on the first subframe (`subframe_cnt == 0`) and
 * increments `subframe_cnt`.
 *
 * Rationale: Decoupling optical subframe diffraction (`accum_subframe`) from camera readout
 * (`readout`) models realistic multi-frame science integrations (`navg > 1`) over rapidly evolving
 * kHz turbulence and DM residuals while applying detector readout noise only once per exposure.
 */
void MILK_HOT cacaosim_fpcam_accum_subframe(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *pup_amp,
    const float          *atm_amp,
    const float          *wf_in,
    const float          *dm_opd,
    const float          *ncpa_opd,
    const float          *d_dm_opd,
    float                 turb_gain)
{
    float        wf_to_opd = cacaosim_wf_to_opd_scale(ctx->cfg.wf_unit, ctx->cfg.ref_lambda,
                                                      turb_gain);
    const float *wf_eff    = (turb_gain != 0.0f) ? wf_in : NULL;

    if (ctx->use_cuda)
    {
        if (ctx->subframe_cnt == 0u)
        {
            cacaosim_fpcam_cuda_reset_accum(ctx);
        }
        cacaosim_fpcam_cuda_accum(ctx, NULL, atm_amp, wf_eff, dm_opd, NULL, NULL, d_dm_opd,
                                  wf_to_opd);
    }
    else
    {
        if (ctx->subframe_cnt == 0u)
        {
            __builtin_memset(ctx->cpu_accum_int, 0, ctx->npix_cam * sizeof(float));
        }
        fpcam_accum_cpu(ctx, pup_amp, atm_amp, wf_eff, dm_opd, ncpa_opd, wf_to_opd);
    }
    ctx->subframe_cnt++;
}

/**
 * cacaosim_fpcam_accum_beam_poly() - Accumulate one polychromatic dynamic `[OPD, Amp]` beam frame
 * @ctx:         Initialized focal-plane camera context
 * @beam_opd:    Host OPD slices `[beam_nlam]` [um] spaced by `chan_stride`, or NULL
 * @beam_amp:    Host amplitude slices `[beam_nlam]` spaced by `chan_stride`, or NULL
 * @beam_nlam:   Number of spectral channels in the input beam (`1` = collapsed, `>= 2` = poly)
 * @chan_stride: Float stride between consecutive spectral slices (e.g., `2 * npix_wf` for SHM)
 * @ncpa_opd:    Optional host static science-arm NCPA OPD `[nwf * nwf]` [um], or NULL
 * @d_beam_opd:  Optional resident device OPD slices (`[beam_nlam]` at `chan_stride`), or NULL
 * @d_beam_amp:  Optional resident device amplitude slices (`[beam_nlam]` at `chan_stride`), or NULL
 *
 * Role: Primary hot-loop entry point for post-coronagraphic and beam-relayed science imaging,
 * propagating dynamic per-frame, per-wavelength `[OPD_um, Amplitude]` slices into the exposure
 * accumulator (`cpu_accum_int` or GPU `d_accum_int`).
 *
 * Rationale: Downstream of `cacaosim.coro`, the post-Lyot electric field exhibits strong chromatic
 * amplitude apodization, stellar diffraction rings, and phase jumps across focal mask transitions
 * that change on every AO step. Accepting strided polychromatic `[OPD, Amp]` cubes directly from
 * SHM (`chan_stride = 2 * npix_wf`) or fused GPU buffers (`chan_stride = npix_wf`) preserves exact
 * broadband dark-hole contrast without extra buffer repacking.
 */
void MILK_HOT cacaosim_fpcam_accum_beam_poly(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    uint32_t              beam_nlam,
    uint64_t              chan_stride,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    if (ctx->use_cuda)
    {
        if (ctx->subframe_cnt == 0u)
        {
            cacaosim_fpcam_cuda_reset_accum(ctx);
        }
        cacaosim_fpcam_cuda_accum_poly(ctx, beam_opd, beam_amp, beam_nlam, chan_stride, d_beam_opd,
                                       d_beam_amp);
    }
    else
    {
        if (ctx->subframe_cnt == 0u)
        {
            __builtin_memset(ctx->cpu_accum_int, 0, ctx->npix_cam * sizeof(float));
        }
        fpcam_accum_poly_cpu(ctx, beam_amp, NULL, beam_opd, beam_nlam, chan_stride, NULL, ncpa_opd,
                             1.0f);
    }
    ctx->subframe_cnt++;
}

/**
 * cacaosim_fpcam_accum_beam() - Accumulate one collapsed 2-slice `[OPD, Amp]` optical beam frame
 * @ctx:        Initialized focal-plane camera context
 * @beam_opd:   Host beam OPD plane `[nwf * nwf]` [um], or NULL
 * @beam_amp:   Host beam amplitude plane `[nwf * nwf]`, or NULL
 * @ncpa_opd:   Optional host static science-arm NCPA OPD `[nwf * nwf]` [um], or NULL
 * @d_beam_opd: Optional resident device beam OPD `[nwf * nwf]` [um], or NULL
 * @d_beam_amp: Optional resident device beam amplitude `[nwf * nwf]`, or NULL
 *
 * Role: Single-slice convenience wrapper around `cacaosim_fpcam_accum_beam_poly()` with
 * `beam_nlam = 1` and `chan_stride = ctx->npix_wf`.
 *
 * Rationale: Simplifies callers that relay an achromatic 2-plane `[OPD_um, Amplitude]` beam (such
 * as `cacaosim.mkbeam`, `cacaosim.beamsplit`, or `cacaosim.coro` with `poly_out = OFF`) while
 * still propagating through all `ctx->eff_nlam` camera spectral channels if `bw_frac > 0`.
 */
void MILK_HOT cacaosim_fpcam_accum_beam(
    cacaosim_fpcam_ctx_t *ctx,
    const float          *beam_opd,
    const float          *beam_amp,
    const float          *ncpa_opd,
    const float          *d_beam_opd,
    const float          *d_beam_amp)
{
    cacaosim_fpcam_accum_beam_poly(ctx, beam_opd, beam_amp, 1u, ctx->npix_wf, ncpa_opd, d_beam_opd,
                                   d_beam_amp);
}

/**
 * cacaosim_fpcam_readout() - Finalize exposure average, apply detector physics, and update Strehl
 * @ctx:      Initialized focal-plane camera context (`subframe_cnt >= 1`; no-op if `0`)
 * @det_live: Optional live detector override (flux, dark, RON, gain, shutter), or NULL for `cfg`
 * @out_u16:  Destination `[ncam * ncam]` `uint16_t` ADU frame when `out_dtype == 0`, or NULL
 * @out_f32:  Destination `[ncam * ncam]` `float` frame when `out_dtype != 0`, or NULL
 * @out_long: Optional destination `[ncam * ncam]` `float` cumulative Strehl-normalized long PSF
 *
 * Role: Completes one science camera exposure after `navg` accumulated subframes: divides the
 * summed optical intensity by `subframe_cnt`, applies Poisson photon noise, dark current, Gaussian
 * readout noise, gain, bias, and ADU saturation via `inf_flux_norm`, updates cumulative
 * long-exposure PSF `cpu_long_int` and Strehl metrics, and resets `subframe_cnt = 0`.
 *
 * Rationale: Keeping `cpu_exp_int` and `cpu_long_int` noise-free in optical Strehl units `[0..1]`
 * (fetched back from GPU on CUDA readout) enables exact, unbiased telemetry (`last_strehl_axis`
 * at bore-sight `(ncam/2, ncam/2)`, `last_strehl_peak` invariant to residual tip/tilt, and
 * `long_strehl_peak`) even when the detector output is heavily photon-starved, saturated, or
 * shuttered dark.
 */
void MILK_HOT cacaosim_fpcam_readout(
    cacaosim_fpcam_ctx_t     *ctx,
    const cacaosim_det_cfg_t *det_live,
    uint16_t                 *out_u16,
    float                    *out_f32,
    float                    *out_long)
{
    if (ctx->subframe_cnt == 0u)
    {
        return;
    }

    const cacaosim_det_cfg_t *det      = (det_live != NULL) ? det_live : &ctx->cfg.det;
    const uint64_t            npix     = ctx->npix_cam;
    const uint32_t            c0       = ctx->cfg.ncam / 2u;
    const uint64_t            axis_idx = (uint64_t) c0 * ctx->cfg.ncam + c0;
    ctx->frame_seq++;

    if (ctx->use_cuda)
    {
        cacaosim_fpcam_cuda_readout(ctx, det, ctx->cpu_exp_int, out_u16, out_f32);
    }
    else
    {
        const float             inv_sub = 1.0f / (float) ctx->subframe_cnt;
        const float *__restrict acc     = MILK_ASSUME_ALIGNED(ctx->cpu_accum_int);
        float *__restrict       exp     = MILK_ASSUME_ALIGNED(ctx->cpu_exp_int);
#pragma omp simd
        for (uint64_t i = 0; i < npix; i++)
        {
            exp[i] = acc[i] * inv_sub;
        }
        cacaosim_detector_readout_cpu(&ctx->det_pool, det, exp, npix, ctx->inf_flux_norm, out_u16,
                                      out_f32);
    }

    /* Update Strehl telemetry and cumulative long-exposure PSF from noise-free optical `exp` */
    const float *__restrict exp      = MILK_ASSUME_ALIGNED(ctx->cpu_exp_int);
    float *__restrict       lbuf     = MILK_ASSUME_ALIGNED(ctx->cpu_long_int);
    ctx->long_cnt++;
    const float             inv_long = 1.0f / (float) ctx->long_cnt;

    float max_exp  = 0.0f;
    float max_long = 0.0f;
    for (uint64_t i = 0; i < npix; i++)
    {
        float v = exp[i];
        if (v > max_exp)
        {
            max_exp = v;
        }
        lbuf[i] += v;
        float lv = lbuf[i] * inv_long;
        if (lv > max_long)
        {
            max_long = lv;
        }
        if (out_long != NULL)
        {
            out_long[i] = lv;
        }
    }

    ctx->last_strehl_axis = exp[axis_idx];
    ctx->last_strehl_peak = max_exp;
    ctx->long_strehl_peak = max_long;
    ctx->subframe_cnt     = 0u;
}

/**
 * cacaosim_fpcam_step() - Execute a single-subframe (`navg == 1`) classical-AO camera exposure
 * @ctx:       Initialized focal-plane camera context
 * @pup_amp:   Host static pupil amplitude `[nwf * nwf]`
 * @atm_amp:   Optional host atmospheric amplitude `[nwf * nwf]`, or NULL
 * @wf_in:     Optional host atmospheric wavefront `[nwf * nwf]`, or NULL
 * @dm_opd:    Optional host DM OPD `[nwf * nwf]` [um], or NULL
 * @ncpa_opd:  Optional host static science-arm NCPA OPD `[nwf * nwf]` [um], or NULL
 * @d_dm_opd:  Optional resident device DM OPD `[nwf * nwf]` [um], or NULL
 * @turb_gain: Live atmospheric turbulence multiplier
 * @det_live:  Optional live detector override, or NULL
 * @out_u16:   Destination `[ncam * ncam]` `uint16_t` frame when `out_dtype == 0`, or NULL
 * @out_f32:   Destination `[ncam * ncam]` `float` frame when `out_dtype != 0`, or NULL
 * @out_long:  Optional destination `[ncam * ncam]` `float` cumulative long-exposure PSF, or NULL
 *
 * Role: Convenience single-call wrapper combining `cacaosim_fpcam_accum_subframe()` followed
 * immediately by `cacaosim_fpcam_readout()`.
 *
 * Rationale: Streamlines unit tests, throughput benchmarks (`cacaosim-benchmark`), and 1:1
 * synchronous simulation loops that read out the camera on every simulation step.
 */
void MILK_HOT cacaosim_fpcam_step(
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
    float                    *out_long)
{
    cacaosim_fpcam_accum_subframe(ctx, pup_amp, atm_amp, wf_in, dm_opd, ncpa_opd, d_dm_opd,
                                  turb_gain);
    cacaosim_fpcam_readout(ctx, det_live, out_u16, out_f32, out_long);
}
