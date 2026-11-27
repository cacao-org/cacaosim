// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_perturb.h
 * @brief   Entrance-pupil telescope vibration and thermally induced petaling (LWE) engine
 *
 * Architectural Role:
 *   Synthesizes physics-realistic, deterministic, physical-time entrance-pupil OPD perturbations
 *   applied to the root optical beam (`b_pup`) before Stage-1 AO correction:
 *     1. Multi-mode telescope & optical-bench vibrations (5 stochastic AR(2) damped harmonic
 *        oscillators spanning 14.5..138 Hz coupled to pupil-orthonormalized Z2..Z6 modes).
 *     2. Thermally induced slow-varying petaling / Low-Wind Effect (LWE): automatic 4-connected
 *        pupil island segmentation (e.g. Subaru 4 offset V-spider quadrants) with independent
 *        smooth Matern-3/2 differential island pistons (~75% variance) and local island tip/tilt
 *        (~25% variance) evolving on physical thermal correlation timescale `petal_tau_s`.
 *
 * Design Rationale:
 *   - Precomputes static pupil-plane modal basis maps (`<= 29` modes) once at init and uploads
 *     them once to GPU VRAM when `use_cuda == 1`. Per-tick evolution updates scalar modal
 *     coefficients in O(1) on CPU (~50 ns) and accumulates the linear combination directly on
 *     device via `cacaosim_cuda_beam_add_modes()` with zero per-tick PCIe bandwidth.
 *   - Scaled by `turb_gain` so perturbations are strictly zero during flat calibration
 *     (`turb_gain == 0`) and ramp smoothly with atmospheric turbulence during closed-loop lock.
 */

#ifndef CACAOSIM_PERTURB_H
#define CACAOSIM_PERTURB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Number of low-order pupil Zernike modes used for telescope/bench vibrations (Z2..Z6). */
#define CACAOSIM_PERTURB_NVIB_ZERN  5
/** Number of stochastic damped resonant vibration lines (14.5, 31.0, 49.5, 99.0, 138.0 Hz). */
#define CACAOSIM_PERTURB_NVIB_LINES 5
/** Maximum number of distinct spider-bounded pupil islands (petals) segmented automatically. */
#define CACAOSIM_PERTURB_MAX_PETALS 8
/** Maximum total linear modal basis maps (5 Zernikes + 3 * MAX_PETALS). */
#define CACAOSIM_PERTURB_MAX_MODES  (CACAOSIM_PERTURB_NVIB_ZERN + 3 * CACAOSIM_PERTURB_MAX_PETALS)

/**
 * struct cacaosim_perturb_cfg_t - Configuration for entrance-pupil vibrations & thermal petaling
 * @vib_enable:   Enable multi-mode stochastic telescope/bench vibrations (0=off, 1=on).
 * @vib_rms_nm:   Target total pupil-plane vibration OPD RMS [nm] (default: 45.0).
 * @vib_seed:     Deterministic PRNG seed for vibration resonators.
 * @petal_enable: Enable thermally induced slow-varying island petaling / LWE (0=off, 1=on).
 * @petal_rms_nm: Target total pupil-plane LWE OPD RMS [nm] (default: 1000.0).
 * @petal_tau_s:  Thermal LWE correlation timescale [s] (default: 3.0).
 * @petal_seed:   Deterministic PRNG seed for thermal petal evolution.
 * @petal_offset_nm: Optional live additive piston step offset per island 0..3 [nm OPD].
 */
typedef struct
{
    int32_t  vib_enable;
    float    vib_rms_nm;
    uint64_t vib_seed;
    int32_t  petal_enable;
    float    petal_rms_nm;
    float    petal_tau_s;
    uint64_t petal_seed;
    float    petal_offset_nm[4];
} cacaosim_perturb_cfg_t;

static inline int cacaosim_perturb_has_petal_offset(const cacaosim_perturb_cfg_t *cfg)
{
    if (cfg == NULL)
    {
        return 0;
    }
    for (int p = 0; p < 4; p++)
    {
        if (cfg->petal_offset_nm[p] > 1.0e-6f || cfg->petal_offset_nm[p] < -1.0e-6f)
        {
            return 1;
        }
    }
    return 0;
}

static inline int cacaosim_perturb_any_active(const cacaosim_perturb_cfg_t *cfg)
{
    if (cfg == NULL)
    {
        return 0;
    }
    return (cfg->vib_enable || cfg->petal_enable || cacaosim_perturb_has_petal_offset(cfg)) ? 1 : 0;
}

/**
 * struct cacaosim_perturb_tlm_t - Instantaneous ground-truth perturbation telemetry snapshot
 * @vib_rms_nm:   Instantaneous vibration OPD RMS on active pupil [nm].
 * @vib_tip_nm:   Instantaneous vibration Tip (Z2) modal amplitude [nm].
 * @vib_tilt_nm:  Instantaneous vibration Tilt (Z3) modal amplitude [nm].
 * @vib_foc_nm:   Instantaneous vibration Defocus (Z4) modal amplitude [nm].
 * @petal_rms_nm: Instantaneous thermal petaling OPD RMS on active pupil [nm].
 * @petal_ptv_nm: Instantaneous island-piston peak-to-valley across petals [nm].
 * @petal_p_nm:   Instantaneous differential island piston for petals 0..3 [nm].
 * @n_petals:     Number of detected pupil islands (4 on Subaru V-spider pupil).
 */
typedef struct
{
    float   vib_rms_nm;
    float   vib_tip_nm;
    float   vib_tilt_nm;
    float   vib_foc_nm;
    float   petal_rms_nm;
    float   petal_ptv_nm;
    float   petal_p_nm[4];
    int32_t n_petals;
} cacaosim_perturb_tlm_t;

/**
 * struct cacaosim_perturb_ctx_t - Runtime basis & stochastic oscillator state for perturbations
 */
typedef struct
{
    int32_t  initialized;
    int32_t  use_cuda;
    uint32_t nwf;
    uint64_t npix;
    int32_t  n_petals;
    int32_t  n_modes;
    double   petal_frac[CACAOSIM_PERTURB_MAX_PETALS];

    float *h_basis; /* Host [n_modes * npix] static pupil basis maps (unit RMS or indicator) */
    float *d_basis; /* Optional GPU copy [n_modes * npix] when use_cuda == 1 */
    float  coeff_um[CACAOSIM_PERTURB_MAX_MODES]; /* Current modal amplitudes [um OPD] */

    /* Stochastic AR(2) vibration resonator states (unit stationary variance each) */
    uint64_t vib_rng;
    double   vib_x1[CACAOSIM_PERTURB_NVIB_LINES];
    double   vib_x2[CACAOSIM_PERTURB_NVIB_LINES];
    int32_t  vib_primed;

    /* Smooth Matern-3/2 thermal petaling states [u, v] for piston/tip/tilt per island */
    uint64_t petal_rng;
    double   petal_u[3 * CACAOSIM_PERTURB_MAX_PETALS];
    double   petal_v[3 * CACAOSIM_PERTURB_MAX_PETALS];
    int32_t  petal_primed;

    cacaosim_perturb_tlm_t tlm;
} cacaosim_perturb_ctx_t;

/**
 * cacaosim_perturb_cfg_defaults() - Populate default perturbation settings (off by default)
 */
void cacaosim_perturb_cfg_defaults(cacaosim_perturb_cfg_t *cfg);

/**
 * cacaosim_perturb_init() - Build pupil Zernike & island-petal basis maps and initialize RNG states
 * @ctx:         Destination runtime context (zeroed on entry).
 * @nwf:         Square pupil grid dimension [pixels].
 * @pup_diam:    Clear pupil diameter fraction of @nwf.
 * @pup_amp:     Entrance pupil amplitude map `[nwf * nwf]` (active mask is `pup_amp > 0.5`).
 * @cfg:         Initial perturbation configuration (seeds used here; enables/RMS may vary live).
 * @use_cuda:    Non-zero to upload static basis maps to GPU VRAM.
 * @cuda_stream: Optional `cudaStream_t` used for one-time basis upload.
 *
 * Return: 0 on success, -1 on allocation failure.
 */
int cacaosim_perturb_init(
    cacaosim_perturb_ctx_t       *ctx,
    uint32_t                      nwf,
    float                         pup_diam,
    const float                  *pup_amp,
    const cacaosim_perturb_cfg_t *cfg,
    int                           use_cuda,
    void                         *cuda_stream);

/**
 * cacaosim_perturb_step_apply() - Advance stochastic states by dt_s and add OPD onto entrance beam
 * @ctx:         Initialized perturbation context.
 * @cfg:         Active live perturbation configuration (enables, RMS, tau).
 * @dt_s:        Physical time step for this tick [s].
 * @scale:       Dimensionless global scale factor (e.g. `turb_gain`; 0 skips evolution & inject).
 * @h_opd:       Host beam OPD `[npix]` in `[um]` (used when `!use_cuda` or `d_opd == NULL`).
 * @d_opd:       Device beam OPD `[npix]` in `[um]` (used when `use_cuda && d_opd != NULL`).
 * @cuda_stream: Optional `cudaStream_t` for GPU accumulation kernel.
 */
void cacaosim_perturb_step_apply(
    cacaosim_perturb_ctx_t       *ctx,
    const cacaosim_perturb_cfg_t *cfg,
    double                        dt_s,
    float                         scale,
    float                        *h_opd,
    float                        *d_opd,
    void                         *cuda_stream);

/**
 * cacaosim_perturb_apply_current() - Add already-stepped modal OPD onto an additional OPD slice
 * @ctx:         Perturbation context containing current-tick `coeff_um[]`.
 * @h_opd:       Host beam OPD `[npix]` in `[um]` (used when `!use_cuda` or `d_opd == NULL`).
 * @d_opd:       Device beam OPD `[npix]` in `[um]` (used when `use_cuda && d_opd != NULL`).
 * @cuda_stream: Optional `cudaStream_t` for GPU accumulation kernel.
 */
void cacaosim_perturb_apply_current(
    const cacaosim_perturb_ctx_t *ctx,
    float                        *h_opd,
    float                        *d_opd,
    void                         *cuda_stream);

/**
 * cacaosim_perturb_free() - Release host and GPU basis buffers held by @ctx
 */
void cacaosim_perturb_free(cacaosim_perturb_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_PERTURB_H */
