// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfstat_core.h
 * @brief   Shared pupil wavefront statistics engine (CPU + CUDA) for wfstat and fused htsim
 *
 * Architectural Role:
 *   Implements the piston-removed OPD accumulators, 5-instant sample capture, synchronized
 *   WFS/DM/PSF diagnostic snapshots, and `wf_stats.json` + FITS writer shared by:
 *     - `cacaosim.wfstat` (`csim_wfstat`): standalone SHM consumer of exported beams (CPU path);
 *     - `cacaosim.htsim` (`csim_htsim`): in-DAG accumulation directly on device beam buffers
 *       (CUDA path), which removes per-tick beam D2H copies and SHM export entirely.
 *
 * Design Rationale:
 *   A single engine guarantees identical statistics, file layout, and JSON schema in modular
 *   and fused modes, so `cacaosim-ao-report` consumes either without modification. The CUDA path
 *   accumulates per-pixel sums in FP64 on device and only transfers data on the 5 sample ticks
 *   and at finalization.
 */

#ifndef CACAOSIM_WFSTAT_CORE_H
#define CACAOSIM_WFSTAT_CORE_H

#include <stdint.h>

#include "milk_runnable.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_WFSTAT_NB_LOC    3
#define CACAOSIM_WFSTAT_NB_SAMP   5
#define CACAOSIM_WFSTAT_NB_EXTRA  14
#define CACAOSIM_WFSTAT_NB_STREHL 7

/**
 * struct cacaosim_wfstat_strehl_t - Per-camera Strehl time-series capture track
 * @img:       Connected `out_long` SHM stream (`ID == -1` if absent).
 * @last_cnt0: Last observed `img.md->cnt0` counter to detect new camera exposures.
 * @cap:       Allocated capacity of time-series arrays.
 * @len:       Number of captured exposures.
 * @frame:     Master tick index (`ws->n_frames`) of each captured exposure `[cap]`.
 * @sr_inst:   Short-exposure peak Strehl ratio `[0..1]` `[cap]`.
 * @sr_long:   Cumulative long-exposure peak Strehl ratio `[0..1]` `[cap]`.
 * @sr_axis:   Short-exposure on-axis Strehl ratio `[0..1]` `[cap]`.
 * @active:    1 when the stream is connected and arrays are allocated.
 */
typedef struct
{
    IMGID     img;
    uint64_t  last_cnt0;
    uint64_t  cap;
    uint64_t  len;
    uint64_t *frame;
    float    *sr_inst;
    float    *sr_long;
    float    *sr_axis;
    int       active;
} cacaosim_wfstat_strehl_t;

/**
 * struct cacaosim_wfstat_extra_t - One optional 2D diagnostic stream snapshot track
 * @img:    Connected SHM stream (`ID == -1` if absent).
 * @nx:     Stream X size [pixels].
 * @ny:     Stream Y size [pixels].
 * @cube:   Host sample cube `[NB_SAMP * nx * ny]`.
 * @m1:     First per-sample scalar metric (kind-dependent, see `.c`).
 * @m2:     Second per-sample scalar metric.
 * @active: 1 when the stream was connected and the cube allocated.
 */
typedef struct
{
    IMGID    img;
    uint32_t nx;
    uint32_t ny;
    float   *cube;
    double   m1[CACAOSIM_WFSTAT_NB_SAMP];
    double   m2[CACAOSIM_WFSTAT_NB_SAMP];
    int      active;
} cacaosim_wfstat_extra_t;

/**
 * struct cacaosim_wfstat_t - Accumulation state for 3 optical planes (raw, stg1, final)
 *
 * Host arrays always hold finalized sums after cacaosim_wfstat_write(); when @use_cuda == 1
 * the running per-pixel sums live on device (@gpu) until then.
 */
typedef struct
{
    uint32_t nwf;
    uint64_t npix;
    uint64_t act_pix;
    uint8_t *mask;
    double  *weights;
    float   *pup_amp;
    float   *wf_tmp;
    double  *sortbuf;

    double *sum_wf[CACAOSIM_WFSTAT_NB_LOC];
    double *sum2_wf[CACAOSIM_WFSTAT_NB_LOC];
    float  *samp_wf[CACAOSIM_WFSTAT_NB_LOC];
    double  sum_srms[CACAOSIM_WFSTAT_NB_LOC];
    double  sum_svar[CACAOSIM_WFSTAT_NB_LOC];

    double   samp_srms[CACAOSIM_WFSTAT_NB_SAMP][CACAOSIM_WFSTAT_NB_LOC];
    double   samp_sptv[CACAOSIM_WFSTAT_NB_SAMP][CACAOSIM_WFSTAT_NB_LOC];
    uint64_t samp_ticks[CACAOSIM_WFSTAT_NB_SAMP];
    uint64_t samp_frame[CACAOSIM_WFSTAT_NB_SAMP];
    uint64_t n_frames;
    int      nsamp;
    int      cur_is_samp;

    int                      snap_extra;
    cacaosim_wfstat_extra_t  extras[CACAOSIM_WFSTAT_NB_EXTRA];
    uint64_t                 ts_cap;
    uint64_t                 ts_len;
    float                   *ts_rms[CACAOSIM_WFSTAT_NB_LOC];
    cacaosim_wfstat_strehl_t strehl[CACAOSIM_WFSTAT_NB_STREHL];

    int   use_cuda;
    void *gpu;
} cacaosim_wfstat_t;

/**
 * cacaosim_wfstat_init() - Allocate accumulators and derive the weighted pupil mask
 * @ws:           Destination state (zeroed by this call).
 * @nwf:          Pupil grid size [pixels].
 * @amp:          Host pupil amplitude `[nwf * nwf]` (NULL = unity); mask is `amp > 0.5`.
 * @target_ticks: Planned frame count used to place the 5 sample instants.
 * @snap_extra:   Non-zero to connect and snapshot the 14 diagnostic SHM streams.
 * @use_cuda:     Non-zero to accumulate on device via cacaosim_wfstat_accum_dev().
 * @cuda_stream:  CUDA stream shared with the producer of the device beams (or NULL).
 *
 * Return: 0 on success, -1 on allocation or GPU setup failure.
 */
int cacaosim_wfstat_init(
    cacaosim_wfstat_t *ws,
    uint32_t           nwf,
    const float       *amp,
    uint64_t           target_ticks,
    int                snap_extra,
    int                use_cuda,
    void              *cuda_stream);

/**
 * cacaosim_wfstat_frame() - Accumulate one frame for all planes
 * @ws:     Initialized state.
 * @opd_um: Per-plane OPD pointers `[NB_LOC]` in microns (host if !use_cuda, device otherwise).
 *
 * Captures sample maps and extras snapshots on the 5 planned sample ticks.
 */
void cacaosim_wfstat_frame(cacaosim_wfstat_t *ws, const float *const opd_um[]);

/**
 * cacaosim_wfstat_write() - Fetch device sums (if any) and write FITS products + wf_stats.json
 * @ws:      Initialized state.
 * @out_dir: Output directory (created if missing).
 * @dt_phys: Physical seconds per frame.
 *
 * Return: 0 on success, -1 if wf_stats.json could not be written.
 */
int cacaosim_wfstat_write(cacaosim_wfstat_t *ws, const char *out_dir, double dt_phys);

/** cacaosim_wfstat_free() - Release host, device, and SHM resources held by @ws. */
void cacaosim_wfstat_free(cacaosim_wfstat_t *ws);

/* ---- CUDA backend (cacaosim_wfstat_cuda.cu; no-op stubs in cacaosim_cuda_stub.c) ---------- */

void *cacaosim_wfstat_gpu_create(
    const double *h_weights,
    uint64_t      npix,
    uint32_t      nloc,
    uint64_t      ts_cap,
    void         *cuda_stream);
void cacaosim_wfstat_gpu_free(void *gpu);
void cacaosim_wfstat_gpu_accum(void *gpu, const float *const d_opd[]);
int  cacaosim_wfstat_gpu_fetch_sample(void *gpu, float *const h_maps[], double *h_frame);
int  cacaosim_wfstat_gpu_fetch_sums(
    void         *gpu,
    double *const h_sum[],
    double *const h_sum2[],
    double       *h_acc,
    float *const  h_ts_rms[],
    uint64_t     *out_ts_len);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_WFSTAT_CORE_H */
