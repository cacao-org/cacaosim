// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench.h
 * @brief   Declarative multi-stage optical bench (.bench) parser and multi-rate DAG engine
 */

#ifndef CACAOSIM_BENCH_H
#define CACAOSIM_BENCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "milk_runnable.h"
#include "cacaosim_types.h"
#include "common/cacaosim_beam.h"
#include "coro/cacaosim_coro.h"
#include "dm/cacaosim_dm.h"
#include "fpcam/cacaosim_fpcam.h"
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "shwfs/cacaosim_shwfs.h"
#include "zwfs/cacaosim_zwfs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_BENCH_MAX_BEAMS   32
#define CACAOSIM_BENCH_MAX_NODES   32
#define CACAOSIM_BENCH_MAX_DMS     16
#define CACAOSIM_BENCH_MAX_SENSORS 16
#define CACAOSIM_BENCH_NAME_LEN    64
#define CACAOSIM_BENCH_PATH_LEN    256

/**
 * @brief Optical node type in a declarative .bench DAG.
 */
typedef enum
{
    CACAOSIM_NODE_DM       = 0, /**< Zonal DM or 2-axis Tip-Tilt mirror ([dm.<id>], [ttm.<id>]) */
    CACAOSIM_NODE_SPLITTER = 1, /**< Gray or dichroic beamsplitter ([splitter.<id>]) */
    CACAOSIM_NODE_CORO     = 2, /**< Coronagraph FPM + Lyot + LOWFS pickoff ([coro.<id>]) */
    CACAOSIM_NODE_PYRWFS   = 3, /**< 4-sided Pyramid WFS + camera ([pyrwfs.<id>]) */
    CACAOSIM_NODE_SHWFS    = 4, /**< Diffractive Shack-Hartmann WFS + camera ([shwfs.<id>]) */
    CACAOSIM_NODE_ZWFS     = 5, /**< Babinet MFT Zernike WFS + camera ([zwfs.<id>]) */
    CACAOSIM_NODE_FPCAM    = 6, /**< Focal-plane science PSF camera ([fpcam.<id>]) */
    CACAOSIM_NODE_WFE      = 7  /**< Static wavefront error map in OPD [m] ([wfe.<id>]) */
} cacaosim_node_kind_t;

/**
 * @brief Parsed configuration for a single optical node in a .bench file.
 */
typedef struct
{
    cacaosim_node_kind_t kind;
    char                 id[CACAOSIM_BENCH_NAME_LEN];

    /* Beam connectivity */
    char                 in_beam[CACAOSIM_BENCH_NAME_LEN];
    char                 out_beam[CACAOSIM_BENCH_NAME_LEN];
    char                 out_refl[CACAOSIM_BENCH_NAME_LEN];
    char                 out_trans[CACAOSIM_BENCH_NAME_LEN];
    char                 out_lowfs[CACAOSIM_BENCH_NAME_LEN];

    /* Optional SHM stream names and custom FITS/SHM mask specifiers */
    char                 in_cmd[CACAOSIM_BENCH_NAME_LEN];
    char                 out_opd[CACAOSIM_BENCH_NAME_LEN];
    char                 in_flat[CACAOSIM_BENCH_NAME_LEN];
    char                 in_wfe[CACAOSIM_BENCH_NAME_LEN];
    char                 out_cam[CACAOSIM_BENCH_NAME_LEN];
    char                 out_long[CACAOSIM_BENCH_NAME_LEN];
    char                 in_ncpa[CACAOSIM_BENCH_NAME_LEN];
    char                 in_pup_amp[CACAOSIM_BENCH_PATH_LEN];
    char                 in_pup_pha[CACAOSIM_BENCH_PATH_LEN];
    char                 in_fpm_amp[CACAOSIM_BENCH_PATH_LEN];
    char                 in_fpm_pha[CACAOSIM_BENCH_PATH_LEN];
    char                 in_lyot[CACAOSIM_BENCH_PATH_LEN];
    char                 in_lyot_pha[CACAOSIM_BENCH_PATH_LEN];
    char                 ncpa_refl[CACAOSIM_BENCH_NAME_LEN];
    char                 ncpa_trans[CACAOSIM_BENCH_NAME_LEN];
    char                 pup_refl[CACAOSIM_BENCH_NAME_LEN];
    char                 pup_trans[CACAOSIM_BENCH_NAME_LEN];

    /* Throughput, cadence, photometry, static WFE synthesis, and 2D registration */
    float                tput;
    uint32_t             navg;
    float                rate_hz;
    float                bw_frac;
    float                qe;
    int32_t              use_beam_flux;
    float                wfe_rms_nm;
    float                wfe_alpha;
    float                wfe_gain;
    uint64_t             wfe_seed;
    cacaosim_reg_t       reg;
    cacaosim_reg_t       reg_refl;
    cacaosim_reg_t       reg_trans;

    /* Per-kind engine configurations */
    cacaosim_dm_cfg_t     dm;
    cacaosim_split_cfg_t  split;
    cacaosim_coro_cfg_t   coro;
    cacaosim_pyrwfs_cfg_t pyrwfs;
    cacaosim_shwfs_cfg_t  shwfs;
    cacaosim_zwfs_cfg_t   zwfs;
    cacaosim_fpcam_cfg_t  fpcam;
} cacaosim_node_cfg_t;

/**
 * @brief Top-level parsed configuration of a multi-stage .bench file.
 */
typedef struct
{
    char                name[CACAOSIM_BENCH_NAME_LEN];
    uint32_t            nwf;
    float               pup_diam;
    float               cobs;
    int32_t             nspider;
    float               spider_w;
    float               spider_deg;
    float               tel_diam_m;
    float               star_mag;
    float               star_teff;
    float               tput_tel;
    float               master_hz;
    char                in_wf[CACAOSIM_BENCH_NAME_LEN];
    char                in_amp[CACAOSIM_BENCH_NAME_LEN];
    char                in_pup[CACAOSIM_BENCH_NAME_LEN];
    char                in_ncpa[CACAOSIM_BENCH_NAME_LEN];
    char                out_beam[CACAOSIM_BENCH_NAME_LEN];
    int32_t             wf_unit;
    float               ref_lambda;
    float               turb_gain;
    int32_t             backend;

    uint32_t            n_nodes;
    cacaosim_node_cfg_t nodes[CACAOSIM_BENCH_MAX_NODES];
} cacaosim_bench_cfg_t;

/**
 * @brief Runtime optical beam slot (CPU + optional GPU device buffers + 32-bin spectrum).
 */
typedef struct
{
    char            name[CACAOSIM_BENCH_NAME_LEN];
    cacaosim_spec_t spec;
    uint32_t        nlam;
    float           lam_um[CACAOSIM_CORO_MAX_LAMBDA];
    float          *opd;
    float          *amp;
    float          *d_opd;
    float          *d_amp;
} cacaosim_beam_slot_t;

/**
 * @brief Runtime node instance inside an initialized multi-stage bench DAG.
 */
typedef struct
{
    cacaosim_node_cfg_t cfg;
    int32_t             in_idx;
    int32_t             out_idx;
    int32_t             refl_idx;
    int32_t             trans_idx;
    int32_t             lowfs_idx;
    int32_t             dm_slot;
    int32_t             sensor_slot;

    /* Effective cadence and integrated photon flux per readout */
    uint32_t            eff_navg;
    float               eff_flux_ph;
    float               eff_lambda_um;
    uint32_t            ncam;
    int32_t             out_dtype;
    int32_t             ready_this_tick;

    /* Optional registration relay scratch buffers for sensors */
    int32_t             has_reg;
    float              *reg_opd;
    float              *reg_amp;
    float              *d_reg_opd;
    float              *d_reg_amp;

    /* Optional static map pointers (SHM or NULL) + device copies */
    const float        *h_ncpa_refl;
    const float        *h_ncpa_trans;
    const float        *h_pup_refl;
    const float        *h_pup_trans;
    float              *d_ncpa_refl;
    float              *d_ncpa_trans;
    float              *d_pup_refl;
    float              *d_pup_trans;

    /* Host buffers when running in pure C mode (connect_shm == 0) + device WFE map */
    float              *cmd_buf;
    float              *dm_opd_buf;
    float              *wfe_m_buf;
    float              *d_wfe_m;
    uint64_t            wfe_cnt0;
    uint16_t           *cam_u16;
    float              *cam_f32;
    float              *long_f32;

    /* Connected SHM stream handles when connect_shm == 1 */
    IMGID               img_cmd;
    IMGID               img_opd;
    IMGID               img_flat;
    IMGID               img_wfe;
    IMGID               img_cam;
    IMGID               img_long;
    IMGID               img_ncpa;
    IMGID               img_lyot;
    IMGID               img_ncpa_refl;
    IMGID               img_ncpa_trans;
    IMGID               img_pup_refl;
    IMGID               img_pup_trans;

    /* Active compute unit context */
    cacaosim_dm_ctx_t     dm_ctx;
    cacaosim_coro_ctx_t   coro_ctx;
    cacaosim_pyrwfs_ctx_t pyrwfs_ctx;
    cacaosim_shwfs_ctx_t  shwfs_ctx;
    cacaosim_zwfs_ctx_t   zwfs_ctx;
    cacaosim_fpcam_ctx_t  fpcam_ctx;
} cacaosim_bench_node_t;

/**
 * @brief Persistent runtime context for a multi-stage .bench DAG.
 */
typedef struct
{
    cacaosim_bench_cfg_t  cfg;
    int32_t               connect_shm;
    int32_t               use_cuda;
    uint32_t              nwf;
    uint64_t              npix_wf;
    float                 dt_master_s;
    uint64_t              tick_cnt;
    void                 *cuda_stream;

    /* Entrance pupil, static NCPA, and atmosphere buffers */
    float                *fallback_pup;
    const float          *pup_ptr;
    const float          *ncpa_ptr;
    float                *d_pup_amp;
    float                *d_ncpa_opd;
    float                *d_wf_in;
    float                *d_atm_amp;

    /* Entrance SHM streams (when connect_shm == 1) */
    IMGID                 img_pup;
    IMGID                 img_ncpa;
    IMGID                 img_wf;
    IMGID                 img_amp;

    /* Named optical beams and topologically sorted nodes */
    uint32_t              n_beams;
    cacaosim_beam_slot_t  beams[CACAOSIM_BENCH_MAX_BEAMS];

    uint32_t              n_nodes;
    cacaosim_bench_node_t nodes[CACAOSIM_BENCH_MAX_NODES];

    /* Fast index tables mapping DM / sensor slots -> sorted node index */
    uint32_t              n_dms;
    int32_t               dm_node_idx[CACAOSIM_BENCH_MAX_DMS];
    uint32_t              n_sensors;
    int32_t               sensor_node_idx[CACAOSIM_BENCH_MAX_SENSORS];

    /* Live science Strehl telemetry (populated from primary fpcam node) */
    float                 last_strehl_inst;
    float                 last_strehl_ave;
} cacaosim_bench_ctx_t;

/**
 * @brief Built-in optical bench hardware templates for rapid setup.
 */
typedef enum
{
    CACAOSIM_PRESET_SINGLE_PYRWFS  = 0, /**< Single-stage DM + Dichroic + PyWFS + Sci Cam */
    CACAOSIM_PRESET_SINGLE_SHWFS   = 1, /**< Single-stage TTM + DM + Dichroic + SHWFS + Sci Cam */
    CACAOSIM_PRESET_TWO_STAGE      = 2, /**< Cascaded Woofer + J-PyWFS + Tweeter + Vis-PyWFS */
    CACAOSIM_PRESET_EXAO_CORO      = 3, /**< TTM + DM + SHWFS + Lyot Coro + ZWFS LOWFS + Cam */
    CACAOSIM_PRESET_TRUTH_DUAL_WFS = 4, /**< DM + Dichroic + Gray BS -> PyWFS + Truth SHWFS */
    CACAOSIM_PRESET_MINIMAL        = 5, /**< Minimal Telescope -> Focal-Plane Science Camera */
    CACAOSIM_PRESET_COUNT          = 6
} cacaosim_preset_t;

/**
 * cacaosim_bench_cfg_defaults() - Initialize a .bench configuration struct with defaults
 */
void cacaosim_bench_cfg_defaults(cacaosim_bench_cfg_t *cfg);

/**
 * cacaosim_bench_node_defaults() - Initialize a single optical node configuration with defaults
 * @nd:     Destination node configuration
 * @kind:   Optical node type
 * @id:     Unique node section identifier
 * @is_ttm: Non-zero when initializing a 2-axis Tip-Tilt mirror (CACAOSIM_NODE_DM, dm_mode=1)
 * @idx:    Node slot index (used to derive deterministic distinct PRNG seeds)
 */
void cacaosim_bench_node_defaults(
    cacaosim_node_cfg_t *nd,
    cacaosim_node_kind_t kind,
    const char          *id,
    int                  is_ttm,
    uint32_t             idx);

/**
 * cacaosim_bench_parse_str() - Parse an INI/TOML-style .bench configuration from a string
 *
 * Return: 0 on success, -1 on syntax/validation error (writes message to @err_buf).
 */
int cacaosim_bench_parse_str(
    const char           *text,
    cacaosim_bench_cfg_t *cfg,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_load_file() - Read and parse a .bench configuration file from disk
 *
 * Return: 0 on success, -1 on I/O or syntax error (writes message to @err_buf).
 */
int cacaosim_bench_load_file(
    const char           *path,
    cacaosim_bench_cfg_t *cfg,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_format_str() - Serialize a .bench configuration into INI/TOML text
 * @cfg: Source bench configuration
 * @buf: Destination text buffer (may be NULL when @cap == 0 to query required length)
 * @cap: Capacity of @buf in bytes
 *
 * Return: Number of characters formatted (excluding NUL), or -1 on invalid input.
 */
int cacaosim_bench_format_str(
    const cacaosim_bench_cfg_t *cfg,
    char                       *buf,
    size_t                      cap);

/**
 * cacaosim_bench_save_file() - Write a .bench configuration cleanly to disk
 * @path:    Destination file path
 * @cfg:     Source bench configuration
 * @err_buf: Optional error diagnostic buffer
 * @err_cap: Capacity of @err_buf in bytes
 *
 * Return: 0 on success, -1 on I/O or validation error.
 */
int cacaosim_bench_save_file(
    const char                 *path,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_preset_name() - Short title for built-in hardware template @preset_id
 */
const char *cacaosim_bench_preset_name(int preset_id);

/**
 * cacaosim_bench_preset_desc() - One-line summary for built-in hardware template @preset_id
 */
const char *cacaosim_bench_preset_desc(int preset_id);

/**
 * cacaosim_bench_preset() - Populate @cfg with a complete built-in optical bench template
 *
 * Return: 0 on success, -1 if @preset_id is out of range.
 */
int cacaosim_bench_preset(
    cacaosim_bench_cfg_t *cfg,
    int                   preset_id);

/**
 * cacaosim_bench_list_beams() - Enumerate all optical beam names produced in @cfg
 * @cfg:       Source bench configuration
 * @beams:     Output array of beam name strings
 * @max_beams: Capacity of @beams array
 *
 * Return: Total number of produced beams written to @beams.
 */
uint32_t cacaosim_bench_list_beams(
    const cacaosim_bench_cfg_t *cfg,
    char                        beams[][CACAOSIM_BENCH_NAME_LEN],
    uint32_t                    max_beams);

/**
 * cacaosim_bench_add_node() - Insert a new optical element and auto-wire its optical beams
 * @cfg:            Active bench configuration to modify
 * @kind:           Optical element kind to insert
 * @is_ttm:         1 for 2-axis Tip-Tilt mirror when @kind == CACAOSIM_NODE_DM, else 0
 * @after_node_idx: Node index [0..n_nodes-1] to insert after, or -1 for entrance telescope beam
 * @target_beam:    Explicit input beam name, or NULL/"" to auto-select & splice downstream
 *
 * Return: Index [0..n_nodes-1] of inserted node, or -1 if node capacity is full.
 */
int cacaosim_bench_add_node(
    cacaosim_bench_cfg_t *cfg,
    cacaosim_node_kind_t  kind,
    int                   is_ttm,
    int                   after_node_idx,
    const char           *target_beam);

/**
 * cacaosim_bench_clone_node() - Duplicate node @src_idx with unique ID/streams and auto-wiring
 *
 * Return: Index of cloned node, or -1 on error.
 */
int cacaosim_bench_clone_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              src_idx);

/**
 * cacaosim_bench_remove_node() - Remove node @node_idx and bridge its upstream/downstream beams
 *
 * Return: 0 on success, -1 if @node_idx is out of range.
 */
int cacaosim_bench_remove_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx);

/**
 * cacaosim_bench_move_node() - Reorder node @node_idx by @dir (-1 upstream/up, +1 downstream/down)
 *
 * When moving an inline optical element across an adjacent inline element on the same beam chain,
 * splices their beam connections so physical upstream/downstream optical causality swaps cleanly.
 *
 * Return: New index of the moved node, or -1 if move is not possible.
 */
int cacaosim_bench_move_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx,
    int                   dir);

/**
 * cacaosim_bench_swap_branches() - Swap downstream consumers on splitter refl/trans or coro arms
 *
 * Return: 0 on success, -1 if node has no dual output arms.
 */
int cacaosim_bench_swap_branches(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx);

/**
 * cacaosim_bench_rename_node() - Rename node @node_idx to a validated unique identifier @new_id
 *
 * Return: 0 on success, -1 if invalid or duplicate (writes reason to @err_buf).
 */
int cacaosim_bench_rename_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx,
    const char           *new_id,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_rewire_beam_out() - Rename a produced optical beam and update all its consumers
 */
int cacaosim_bench_rewire_beam_out(
    cacaosim_bench_cfg_t *cfg,
    const char           *old_beam,
    const char           *new_beam);

/**
 * cacaosim_bench_init() - Topologically sort, validate, and allocate a multi-stage bench DAG
 * @ctx:         Output runtime context
 * @cfg:         Parsed bench configuration
 * @ext_pup:     Optional host pupil amplitude [nwf * nwf] (overrides SHM/auto pupil when != NULL)
 * @connect_shm: 1 to bind/create ImageStreamIO SHM streams, 0 for pure in-memory C execution
 * @err_buf:     Optional error diagnostic buffer
 * @err_cap:     Capacity of @err_buf in bytes
 *
 * Return: 0 on success, -1 on failure.
 */
int cacaosim_bench_init(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    const float                *ext_pup,
    int32_t                     connect_shm,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_analyze() - Zero-allocation topological sort & spectral photometry dry-run
 * @ctx:     Output context populated with sorted DAG nodes, beams, cadences, and 32-bin spectra
 * @cfg:     Parsed bench configuration
 * @err_buf: Optional error diagnostic buffer
 * @err_cap: Capacity of @err_buf in bytes
 *
 * Validates topology and propagates photon-rate spectra across all beams and sensor passbands
 * without allocating FFTW/cuFFT plans, host/GPU pixel buffers, or SHM streams (< 0.1 ms).
 *
 * Return: 0 on success, -1 on topology/validation error.
 */
int cacaosim_bench_analyze(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_free() - Release all host, CUDA, and SHM resources owned by @ctx
 */
void cacaosim_bench_free(cacaosim_bench_ctx_t *ctx);

/**
 * cacaosim_bench_print_summary() - Print optical topology and photon budget summary table
 */
void cacaosim_bench_print_summary(
    const cacaosim_bench_ctx_t *ctx,
    FILE                       *fp);

/**
 * cacaosim_bench_step() - Advance one master-clock tick across the entire optical bench DAG
 * @ctx:       Active bench context
 * @ext_wf:    Optional host input wavefront [nwf * nwf] (used when connect_shm == 0 or override)
 * @ext_amp:   Optional host input amplitude [nwf * nwf]
 * @turb_gain: Live atmospheric turbulence multiplier
 * @procinfo:  Optional PROCESSINFO* handle for SHM stream lineage updates
 */
void cacaosim_bench_step(
    cacaosim_bench_ctx_t *ctx,
    const float          *ext_wf,
    const float          *ext_amp,
    float                 turb_gain,
    void                 *procinfo);

/**
 * cacaosim_bench_find_dm() - Lookup DM slot index [0..n_dms-1] by section id (or -1)
 */
int cacaosim_bench_find_dm(
    const cacaosim_bench_ctx_t *ctx,
    const char                 *id);

/**
 * cacaosim_bench_find_sensor() - Lookup sensor slot index [0..n_sensors-1] by section id (or -1)
 */
int cacaosim_bench_find_sensor(
    const cacaosim_bench_ctx_t *ctx,
    const char                 *id);

/**
 * cacaosim_bench_dm_cmd_ptr() - Writable actuator command array [nact_x * nact_y] for DM @dm_slot
 */
float *cacaosim_bench_dm_cmd_ptr(
    cacaosim_bench_ctx_t *ctx,
    int                   dm_slot);

/**
 * cacaosim_bench_sensor_f32() - Latest float32 camera frame [ncam * ncam] for sensor @sensor_slot
 */
const float *cacaosim_bench_sensor_f32(
    const cacaosim_bench_ctx_t *ctx,
    int                         sensor_slot);

/**
 * cacaosim_bench_sensor_u16() - Latest uint16 camera frame [ncam * ncam] for sensor @sensor_slot
 */
const uint16_t *cacaosim_bench_sensor_u16(
    const cacaosim_bench_ctx_t *ctx,
    int                         sensor_slot);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BENCH_H */
