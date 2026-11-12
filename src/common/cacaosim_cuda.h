// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_cuda.h
 * @brief   Host-callable wrappers for common CUDA kernels (PRNG, detector, pupil field, binning)
 */

#ifndef CACAOSIM_CUDA_H
#define CACAOSIM_CUDA_H

#include <stdint.h>

#include "cacaosim_types.h"
#include "common/cacaosim_beam.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * cacaosim_cuda_available() - Check if CUDA support is compiled in and a GPU is accessible
 *
 * Return: 1 if CUDA GPU is available, 0 otherwise.
 */
int cacaosim_cuda_available(void);

/**
 * cacaosim_cuda_stream_create() - Create a non-blocking CUDA stream (NULL if unavailable)
 */
void *cacaosim_cuda_stream_create(void);

/**
 * cacaosim_cuda_stream_destroy() - Destroy a CUDA stream created by cacaosim_cuda_stream_create()
 */
void cacaosim_cuda_stream_destroy(void *cuda_stream);

/**
 * cacaosim_cuda_stream_sync() - Synchronize a CUDA stream
 */
void cacaosim_cuda_stream_sync(void *cuda_stream);

/**
 * cacaosim_cuda_alloc_f32() - Allocate zero-initialized float32 device buffer [nelem]
 */
float *cacaosim_cuda_alloc_f32(uint64_t nelem);

/**
 * cacaosim_cuda_free_f32() - Free float32 device buffer
 */
void cacaosim_cuda_free_f32(float *d_ptr);

/**
 * cacaosim_cuda_h2d_f32() - Asynchronous Host-to-Device float32 copy
 */
void cacaosim_cuda_h2d_f32(
    float       *d_dst,
    const float *h_src,
    uint64_t     nelem,
    void        *cuda_stream);

/**
 * cacaosim_cuda_d2h_f32() - Asynchronous Device-to-Host float32 copy
 */
void cacaosim_cuda_d2h_f32(
    float       *h_dst,
    const float *d_src,
    uint64_t     nelem,
    void        *cuda_stream);

/**
 * cacaosim_cuda_build_pupil_padded() - Synthesize zero-padded complex pupil field on GPU
 * @d_pup_amp:   Device pupil amplitude [nwf * nwf]
 * @d_atm_amp:   Optional device atmospheric amplitude [nwf * nwf] (NULL -> 1.0)
 * @d_wf_in:     Optional device input wavefront [nwf * nwf] (NULL -> 0.0)
 * @d_dm_opd:    Optional device DM OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @d_ncpa_opd:  Optional device static NCPA OPD [nwf * nwf] in microns (NULL -> 0.0)
 * @wf_to_opd:   Scale converting d_wf_in to microns OPD
 * @opd_to_rad:  2 * pi / lambda_um [rad / um]
 * @nwf:         Input pupil grid dimension
 * @nfft:        Output complex grid dimension (>= nwf)
 * @d_out_cplx:  Destination float2 / cuFloatComplex buffer [nfft * nfft]
 * @cuda_stream: Optional cudaStream_t handle (NULL for default stream)
 */
void cacaosim_cuda_build_pupil_padded(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_dm_opd,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    float        opd_to_rad,
    uint32_t     nwf,
    uint32_t     nfft,
    void        *d_out_cplx,
    void        *cuda_stream);

/**
 * cacaosim_cuda_detector_readout() - Apply flux scaling, Poisson/RON noise, and ADU quantization
 * @cfg:         Detector readout configuration
 * @d_opt_int:   Device input optical intensity [npix]
 * @npix:        Number of detector pixels
 * @norm_total:  Normalization divisor when > 0 (flux_scale = cfg->flux_ph / norm_total)
 * @seed:        Base 64-bit random seed
 * @frame_seq:   Monotonic frame counter for statistically independent noise realizations
 * @d_out_u16:   Device uint16 output buffer when cfg->out_dtype == 0 (else NULL)
 * @d_out_f32:   Device float32 output buffer when cfg->out_dtype != 0 (else NULL)
 * @cuda_stream: Optional cudaStream_t handle
 */
void cacaosim_cuda_detector_readout(
    const cacaosim_det_cfg_t *cfg,
    const float              *d_opt_int,
    uint64_t                  npix,
    float                     norm_total,
    uint64_t                  seed,
    uint64_t                  frame_seq,
    uint16_t                 *d_out_u16,
    float                    *d_out_f32,
    void                     *cuda_stream);

/**
 * cacaosim_cuda_beam_init() - Synthesize [OPD, amplitude] beam on GPU from atmosphere & pupil
 */
void cacaosim_cuda_beam_init(
    const float *d_pup_amp,
    const float *d_atm_amp,
    const float *d_wf_in,
    const float *d_ncpa_opd,
    float        wf_to_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream);

/**
 * cacaosim_cuda_beam_apply_dm() - Subtract reflective DM OPD from beam OPD on GPU
 */
void cacaosim_cuda_beam_apply_dm(
    const float *d_in_opd,
    const float *d_in_amp,
    const float *d_dm_opd,
    uint32_t     nwf,
    float       *d_out_opd,
    float       *d_out_amp,
    void        *cuda_stream);

/**
 * cacaosim_cuda_beam_relay() - Apply 2D registration, arm NCPA, and arm pupil mask on GPU
 */
void cacaosim_cuda_beam_relay(
    const float          *d_in_opd,
    const float          *d_in_amp,
    const float          *d_ncpa_opd,
    const float          *d_pup_mask,
    const cacaosim_reg_t *reg,
    uint32_t              nwf,
    float                *d_out_opd,
    float                *d_out_amp,
    void                 *cuda_stream);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_CUDA_H */
