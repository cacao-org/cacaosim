// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_types.h
 * @brief   Shared constants, aligned memory helpers, and configuration structs for cacaosim
 */

#ifndef CACAOSIM_TYPES_H
#define CACAOSIM_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_ALIGN_BYTES 64u
#define CACAOSIM_PI_F        3.14159265358979323846f
#define CACAOSIM_TWO_PI_F    6.28318530717958647692f

/**
 * @brief Input wavefront physical unit convention.
 */
typedef enum
{
    CACAOSIM_WFUNIT_OPD_UM    = 0, /**< Optical path difference in microns */
    CACAOSIM_WFUNIT_PHASE_RAD = 1  /**< Phase in radians at reference wavelength */
} cacaosim_wfunit_t;

/**
 * @brief Execution backend selection.
 */
typedef enum
{
    CACAOSIM_BACKEND_AUTO = 0, /**< Prefer CUDA GPU when available, else CPU */
    CACAOSIM_BACKEND_CPU  = 1, /**< Force CPU OpenMP + SIMD + FFTW3f/BLAS */
    CACAOSIM_BACKEND_CUDA = 2  /**< Force CUDA GPU (falls back if unavailable) */
} cacaosim_backend_t;

/**
 * @brief Detector output pixel datatype.
 */
typedef enum
{
    CACAOSIM_DTYPE_U16 = 0, /**< 16-bit unsigned integer ADU */
    CACAOSIM_DTYPE_F32 = 1  /**< 32-bit IEEE floating-point ADU / intensity */
} cacaosim_dtype_t;

/**
 * @brief Shared detector readout model parameters.
 */
typedef struct
{
    float    flux_ph;      /**< Total optical flux per frame [photons] (<0: raw intensity) */
    float    dark_e;       /**< Per-pixel dark + background level [e-/frame] */
    float    ron_e;        /**< Gaussian readout noise [e- RMS] */
    float    gain_eadu;    /**< Detector conversion gain [e-/ADU] */
    float    bias_adu;     /**< Detector zero-flux bias offset [ADU] */
    float    sat_adu;      /**< Saturation ceiling [ADU] */
    int32_t  shutter;      /**< Optical shutter (1 = open, 0 = closed / dark frame) */
    int32_t  photon_noise; /**< Enable Poisson shot noise (1 = on, 0 = off) */
    int32_t  out_dtype;    /**< Output format: 0 = uint16, 1 = float32 */
} cacaosim_det_cfg_t;

/**
 * cacaosim_aligned_calloc() - Allocate zero-initialized 64-byte aligned memory
 * @count: Element count
 * @size:  Bytes per element
 *
 * Return: Aligned zeroed buffer pointer, or NULL on allocation failure.
 */
static inline void *cacaosim_aligned_calloc(
    size_t count,
    size_t size)
{
    size_t total = count * size;
    if (total == 0)
    {
        total = CACAOSIM_ALIGN_BYTES;
    }
    size_t rem = total % CACAOSIM_ALIGN_BYTES;
    if (rem != 0)
    {
        total += (CACAOSIM_ALIGN_BYTES - rem);
    }
    void *ptr = NULL;
    if (posix_memalign(&ptr, CACAOSIM_ALIGN_BYTES, total) != 0 || ptr == NULL)
    {
        return NULL;
    }
    __builtin_memset(ptr, 0, total);
    return ptr;
}

/**
 * cacaosim_wf_to_opd_scale() - Multiplicative factor converting input WF array to microns OPD
 * @wf_unit:    0 for OPD [um], 1 for phase [rad] at ref_lambda
 * @ref_lambda: Reference wavelength [um] used when wf_unit == 1
 * @turb_gain:  Dimensionless turbulence amplitude multiplier
 *
 * Return: Scale factor in [um / input_unit].
 */
static inline float cacaosim_wf_to_opd_scale(
    int32_t wf_unit,
    float   ref_lambda,
    float   turb_gain)
{
    if (wf_unit == CACAOSIM_WFUNIT_PHASE_RAD)
    {
        float lam = (ref_lambda > 1e-4f) ? ref_lambda : 0.50f;
        return turb_gain * (lam / CACAOSIM_TWO_PI_F);
    }
    return turb_gain;
}

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_TYPES_H */
