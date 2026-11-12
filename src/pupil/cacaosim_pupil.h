// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pupil.h
 * @brief   Anti-aliased telescope pupil mask and static NCPA Zernike synthesis
 */

#ifndef CACAOSIM_PUPIL_H
#define CACAOSIM_PUPIL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Geometric parameters for anti-aliased telescope pupil synthesis.
 */
typedef struct
{
    uint32_t nwf;        /**< Square pupil grid dimension [pixels] */
    float    pup_diam;   /**< Outer diameter as fraction of nwf (0 < pup_diam <= 1) */
    float    cobs;       /**< Central obstruction ratio relative to pup_diam [0..0.95) */
    int32_t  nspider;    /**< Number of radial spider support vanes (0 = none) */
    float    spider_w;   /**< Spider vane thickness as fraction of pup_diam */
    float    spider_deg; /**< Azimuth offset of first spider vane [degrees] */
} cacaosim_pupil_cfg_t;

/**
 * cacaosim_make_pupil() - Generate 4x4 subpixel anti-aliased 2D pupil amplitude array
 * @cfg:     Pupil geometry configuration
 * @pup_out: Destination float buffer [cfg->nwf * cfg->nwf]
 *
 * Return: Sum of pupil amplitude values (effective clear area in pixels), or <= 0 on error.
 */
double cacaosim_make_pupil(
    const cacaosim_pupil_cfg_t *cfg,
    float                      *pup_out);

/**
 * cacaosim_make_ncpa() - Generate deterministic low-order Zernike static NCPA OPD map
 * @nwf:         Grid dimension [pixels]
 * @pup_diam:    Outer pupil diameter fraction of nwf
 * @pup_amp:     Pupil amplitude weights [nwf * nwf] for zero-mean / RMS normalization
 * @ncpa_rms_um: Target pupil-weighted RMS OPD [microns] (0 -> zero map)
 * @seed:        Deterministic mode seed
 * @ncpa_out:    Destination float OPD buffer [nwf * nwf] in microns
 */
void cacaosim_make_ncpa(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        ncpa_rms_um,
    uint64_t     seed,
    float       *ncpa_out);

/**
 * cacaosim_make_wfe_m() - Generate static optical wavefront error map in OPD [meter]
 * @nwf:       Grid dimension [pixels]
 * @pup_diam:  Outer pupil diameter fraction of nwf
 * @pup_amp:   Optional pupil amplitude weights [nwf * nwf] (NULL uses circular pup_diam)
 * @rms_m:     Target pupil-weighted RMS OPD [meter] (e.g. 5.0e-8f for 50 nm; <=0 -> zero map)
 * @psd_alpha: Mid-spatial-frequency amplitude slope k^-alpha (<=0 -> 1.6)
 * @seed:      Deterministic PRNG seed
 * @wfe_out_m: Destination float OPD buffer [nwf * nwf] in meters
 */
void cacaosim_make_wfe_m(
    uint32_t     nwf,
    float        pup_diam,
    const float *pup_amp,
    float        rms_m,
    float        psd_alpha,
    uint64_t     seed,
    float       *wfe_out_m);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_PUPIL_H */
