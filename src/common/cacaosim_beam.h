// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_beam.h
 * @brief   Optical beam [OPD, amplitude], spectral bandpass photometry, 2D registration, & SHM IO
 */

#ifndef CACAOSIM_BEAM_H
#define CACAOSIM_BEAM_H

#include <stdint.h>

#include "ImageStruct.h"
#include "cacaosim_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_NSPEC      32
#define CACAOSIM_SPEC_L0_UM 0.40f
#define CACAOSIM_SPEC_L1_UM 2.40f
#define CACAOSIM_SPEC_DL_UM ((CACAOSIM_SPEC_L1_UM - CACAOSIM_SPEC_L0_UM) / (float) CACAOSIM_NSPEC)

/**
 * @brief Beamsplitter / dichroic spectral splitting mode.
 */
typedef enum
{
    CACAOSIM_SPLIT_GRAY        = 0, /**< Achromatic gray splitter (R_gray, T_gray) */
    CACAOSIM_SPLIT_DICHROIC_LP = 1, /**< Long-pass: refl < lambda_cut, trans > lambda_cut */
    CACAOSIM_SPLIT_DICHROIC_SP = 2, /**< Short-pass: refl > lambda_cut, trans < lambda_cut */
    CACAOSIM_SPLIT_DICHROIC_BP = 3  /**< Band-reflect: refl [cut - w/2, cut + w/2], passes rest */
} cacaosim_split_mode_t;

/**
 * @brief Multi-wavelength photon-rate spectrum [photons / second per bin].
 */
typedef struct
{
    float ph_s[CACAOSIM_NSPEC]; /**< Photon arrival rate [photons/s] in each spectral bin */
} cacaosim_spec_t;

/**
 * @brief Beamsplitter / dichroic spectral configuration.
 */
typedef struct
{
    int32_t mode;       /**< Split mode (cacaosim_split_mode_t) */
    float   r_gray;     /**< Gray reflectance [0..1] when mode == CACAOSIM_SPLIT_GRAY */
    float   t_gray;     /**< Gray transmittance [0..1] when mode == CACAOSIM_SPLIT_GRAY */
    float   lambda_cut; /**< Dichroic cutoff or band center wavelength [um] */
    float   band_width; /**< Band-reflect full width [um] when mode == CACAOSIM_SPLIT_DICHROIC_BP */
    float   roll_um;    /**< Dichroic transition edge width [um] (default: 0.03) */
    float   eff_refl;   /**< Peak reflectance efficiency in reflection band [0..1] */
    float   eff_trans;  /**< Peak transmittance efficiency in pass band [0..1] */
} cacaosim_split_cfg_t;

/**
 * @brief 2D pupil-plane affine registration parameters.
 */
typedef struct
{
    float shift_x; /**< Translation along X [pupil pixels] */
    float shift_y; /**< Translation along Y [pupil pixels] */
    float rot_deg; /**< Counter-clockwise pupil rotation [degrees] */
    float scale;   /**< Isotropic pupil magnification (0 or 1 -> 1.0) */
} cacaosim_reg_t;

/**
 * cacaosim_spec_bin_lambda() - Center wavelength [um] of spectral bin @b (0 .. CACAOSIM_NSPEC-1)
 */
static inline float cacaosim_spec_bin_lambda(int b)
{
    return CACAOSIM_SPEC_L0_UM + ((float) b + 0.5f) * CACAOSIM_SPEC_DL_UM;
}

/**
 * cacaosim_reg_is_identity() - Return 1 if registration is a no-op identity transform
 */
int cacaosim_reg_is_identity(const cacaosim_reg_t *reg);

/**
 * cacaosim_spec_star() - Populate stellar blackbody photon spectrum scaled to V-band magnitude
 * @out:        Output 32-bin photon rate spectrum [photons/s]
 * @tel_diam_m: Primary mirror diameter [m]
 * @obs_frac:   Linear central obscuration fraction [0..1)
 * @star_mag:   Apparent V-band Vega magnitude
 * @teff_k:     Stellar effective blackbody temperature [K] (e.g. 5778 for G2V, 3200 for M4V)
 * @tput:       Overall atmosphere + telescope throughput [0..1]
 */
void cacaosim_spec_star(
    cacaosim_spec_t *out,
    float            tel_diam_m,
    float            obs_frac,
    float            star_mag,
    float            teff_k,
    float            tput);

/**
 * cacaosim_spec_scale() - Multiply all spectral bins by a scalar throughput @tput
 */
void cacaosim_spec_scale(
    const cacaosim_spec_t *in,
    float                  tput,
    cacaosim_spec_t       *out);

/**
 * cacaosim_spec_total() - Total integrated photon rate [photons/s] across 0.40..2.40 um
 */
float cacaosim_spec_total(const cacaosim_spec_t *spec);

/**
 * cacaosim_spec_split() - Split input photon spectrum into reflected and transmitted arms
 * @in:        Incident photon spectrum
 * @cfg:       Splitter / dichroic configuration
 * @out_refl:  Output reflected arm spectrum (may be NULL)
 * @out_trans: Output transmitted arm spectrum (may be NULL)
 *
 * Guarantees per-bin photon conservation: out_refl[b] + out_trans[b] <= in[b].
 */
void cacaosim_spec_split(
    const cacaosim_spec_t      *in,
    const cacaosim_split_cfg_t *cfg,
    cacaosim_spec_t            *out_refl,
    cacaosim_spec_t            *out_trans);

/**
 * cacaosim_spec_band_flux() - Integrate photons collected within a sensor filter bandpass
 * @spec:         Incident beam photon-rate spectrum [photons/s]
 * @lambda0_um:   Sensor central filter wavelength [um]
 * @bw_frac:      Fractional filter bandwidth dlambda / lambda0 (0 -> 0.15 default)
 * @dt_exp_s:     Integration / exposure time [s]
 * @qe:           Detector quantum efficiency [0..1]
 * @eff_lambda:   Optional output flux-weighted effective wavelength [um]
 *
 * Return: Integrated detected photo-events / incident photons over the exposure.
 */
float cacaosim_spec_band_flux(
    const cacaosim_spec_t *spec,
    float                  lambda0_um,
    float                  bw_frac,
    float                  dt_exp_s,
    float                  qe,
    float                 *eff_lambda);

/**
 * cacaosim_beam_init_cpu() - Synthesize initial [OPD, amplitude] beam from atmosphere & pupil
 */
void cacaosim_beam_init_cpu(
    const float *pup_amp,
    const float *atm_amp,
    const float *wf_in,
    const float *ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *out_opd,
    float       *out_amp);

/**
 * cacaosim_beam_apply_dm_cpu() - Subtract reflective DM OPD from beam OPD on CPU
 */
void cacaosim_beam_apply_dm_cpu(
    const float *in_opd,
    const float *in_amp,
    const float *dm_opd,
    uint32_t     nwf,
    float       *out_opd,
    float       *out_amp);

/**
 * cacaosim_beam_relay_cpu() - Apply 2D registration, arm NCPA, and arm pupil mask on CPU
 */
void cacaosim_beam_relay_cpu(
    const float          *in_opd,
    const float          *in_amp,
    const float          *ncpa_opd,
    const float          *pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *out_opd,
    float                *out_amp);

/**
 * cacaosim_beam_shm_init_kw() - Initialize fixed-slot CSIM_BM/S00..S31 keywords on a 3D SHM image
 */
void cacaosim_beam_shm_init_kw(IMAGE *im);

/**
 * cacaosim_beam_shm_write_spec() - Fast O(1) per-frame spectral keyword write into SHM image
 */
void cacaosim_beam_shm_write_spec(
    IMAGE                 *im,
    const cacaosim_spec_t *spec);

/**
 * cacaosim_beam_shm_read_spec() - Read spectral vector from SHM image keywords if CSIM_BM == 1
 *
 * Return: 1 if valid beam spectrum was read, 0 otherwise.
 */
int cacaosim_beam_shm_read_spec(
    const IMAGE     *im,
    cacaosim_spec_t *spec);

/**
 * cacaosim_beam_shm_is_valid() - Return 1 if @im is a valid [nwf x nwf x 2] FLOAT32 beam stream
 */
int cacaosim_beam_shm_is_valid(
    const IMAGE *im,
    uint32_t     nwf);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BEAM_H */
