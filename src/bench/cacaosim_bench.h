// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench.h
 * @brief   Declarative multi-stage optical bench (.bench) model, editor, analyzer, and DAG runtime
 *
 * Architectural Role:
 *   Serves as the central public contract uniting the four subsystems of cacaosim's declarative
 *   multi-stage Adaptive Optics bench architecture:
 *     1. Configuration Model, Parser & Serializer (cacaosim_bench_cfg.c): Defines the value-type
 *        AST structs (cacaosim_bench_cfg_t, cacaosim_node_cfg_t), zero-dependency INI/TOML-subset
 *        parser, canonical default-eliding serializer, and built-in AO hardware presets.
 *     2. Topological Graph Editor & Beam Splicer (cacaosim_bench_edit.c): Provides high-level
 *        mutation primitives (insert, clone, delete, reorder, branch-swap, rename, rewire) that
 *        automatically mint collision-free beam/SHM names and heal upstream/downstream optical
 *        continuity.
 *     3. Zero-FFT Static Topology & Spectral Photometry Analyzer (cacaosim_bench_analyze()):
 *        Validates single-producer DAG acyclicity, resolves multi-rate sensor decimation factors,
 *        and propagates 32-bin (0.40..2.40 um) Planck blackbody photon budgets across dichroics,
 *        coronagraphs, and detector passbands without allocating FFT plans, GPU buffers, or SHM.
 *     4. Multi-Rate Zero-Hop CPU/CUDA DAG Engine (cacaosim_bench_dag.c): Compiles named optical
 *        beams into pre-allocated host/GPU [OPD_um, Amplitude] slots and steps arbitrary multi-DM,
 *        multi-WFS, coronagraphic, and polychromatic benches on a single shared CUDA stream or
 *        aligned CPU arena.
 *
 * Design Rationale & Motivation:
 *   - Why fused zero-hop beam passing? Decomposing a multi-kHz AO bench into separate OS
 *     processes connected via ImageStreamIO shared-memory beam cubes incurs semaphore wakeup
 *     jitter, scheduler context switches, LLC cache thrashing, and repeated PCIe host-to-device /
 *     device-to-host copies at every optical surface. Executing the compiled topological DAG
 *     in-process keeps intermediate [OPD_um, Amplitude] wavefronts resident in GPU VRAM (on one
 *     cudaStream_t) or 64-byte aligned CPU buffers while preserving identical optical physics and
 *     full ImageStreamIO compatibility on external actuator command and camera readout streams.
 *   - Why decouple static analysis from runtime initialization? Interactive TUI configurators
 *     (cacaosim-CTRL) and CLI schematic viewers (cacaosim-bench-view) evaluate topology and
 *     photon budgets on every edit. Separating cacaosim_bench_analyze() from FFTW3f/cuFFT plan
 *     creation, CUDA context initialization, and MILK_SHM_DIR stream creation lets previews run
 *     in microseconds (< 0.1 ms) even on headless laptops without a GPU or active SHM directory.
 *   - Why fixed-capacity inline arrays? Bounding beams and nodes (CACAOSIM_BENCH_MAX_*) makes
 *     cacaosim_bench_cfg_t a self-contained Plain-Old-Data value that configurators can snapshot
 *     or copy with a single struct assignment and lets the real-time loop iterate contiguously
 *     without pointer chasing or dynamic allocation.
 */

#ifndef CACAOSIM_BENCH_H
#define CACAOSIM_BENCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "milk_runnable.h"
#include "cacaosim_types.h"
#include "adc/cacaosim_adc.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_disp.h"
#include "common/cacaosim_perturb.h"
#include "coro/cacaosim_coro.h"
#include "dm/cacaosim_dm.h"
#include "fpcam/cacaosim_fpcam.h"
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "shwfs/cacaosim_shwfs.h"
#include "zwfs/cacaosim_zwfs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @name Bench DAG Capacity & Identifier Bounds
 *
 * Compile-time upper bounds sizing inline arrays in cacaosim_bench_cfg_t and cacaosim_bench_ctx_t
 * so configurations and compiled DAGs require zero pointer indirection for metadata tables.
 * @{
 */
/** Maximum distinct named optical beams (entrance source + node outputs) in one .bench DAG. */
#define CACAOSIM_BENCH_MAX_BEAMS   32
/** Maximum optical hardware/sensor node sections ([<type>.<id>]) in one .bench DAG. */
#define CACAOSIM_BENCH_MAX_NODES   32
/** Maximum deformable mirror and Tip-Tilt mirror nodes addressable via fast DM slot indices. */
#define CACAOSIM_BENCH_MAX_DMS     16
/** Maximum terminal wavefront sensor and focal-plane camera nodes addressable via sensor slots. */
#define CACAOSIM_BENCH_MAX_SENSORS 16
/** Maximum NUL-terminated length [bytes] for node IDs, beam names, and SHM stream names. */
#define CACAOSIM_BENCH_NAME_LEN    64
/** Maximum NUL-terminated length [bytes] for custom FITS mask file paths or SHM specifiers. */
#define CACAOSIM_BENCH_PATH_LEN    256
/** @} */

/**
 * enum cacaosim_node_kind_t - Optical element classification in a declarative .bench DAG
 * @CACAOSIM_NODE_DM:       Reflective zonal deformable mirror ([dm.<id>]) or 2-axis Tip-Tilt
 *                          mirror ([ttm.<id>], dm_mode == 1); inline single-input/single-output
 *                          modifier subtracting actuator surface OPD from the beam.
 * @CACAOSIM_NODE_SPLITTER: Achromatic gray or spectral dichroic (LP/SP/BP) beamsplitter
 *                          ([splitter.<id>]); branches one input beam into reflected (out_refl)
 *                          and/or transmitted (out_trans) arms with independent 2D registration,
 *                          static NCPA, and pupil masks while conserving per-bin photon flux.
 * @CACAOSIM_NODE_CORO:     Coronagraphic focal-plane & Lyot-stop relay ([coro.<id>]); propagates
 *                          an input pupil beam to a post-Lyot science pupil beam (out_beam) and
 *                          an optional reflected LOWFS pupil pickoff beam (out_lowfs), optionally
 *                          expanding monochromatic input into a polychromatic [nlam] beam cube.
 * @CACAOSIM_NODE_PYRWFS:   Terminal 4-sided modulated/unmodulated Pyramid WFS + detector
 *                          ([pyrwfs.<id>]); accumulates subframes at master clock and reads out
 *                          every eff_navg ticks.
 * @CACAOSIM_NODE_SHWFS:    Terminal diffractive Shack-Hartmann lenslet array WFS + detector
 *                          ([shwfs.<id>]); accumulates lenslet PSFs and reads out every eff_navg
 *                          ticks.
 * @CACAOSIM_NODE_ZWFS:     Terminal Babinet Matrix-DFT Zernike phase-contrast WFS + detector
 *                          ([zwfs.<id>]); suited for fine cophasing and coronagraphic LOWFS.
 * @CACAOSIM_NODE_FPCAM:    Terminal focal-plane science PSF camera ([fpcam.<id>]); supports MFT
 *                          or padded FFT across monochromatic or polychromatic beams, multi-frame
 *                          integration, cumulative long-exposure PSF, and peak Strehl telemetry.
 * @CACAOSIM_NODE_WFE:      Inline static wavefront error OPD map ([wfe.<id>] / [static_wfe.<id>])
 *                          stored in physical meters [m] in SHM or synthesized from a power-law
 *                          spatial PSD and scaled into microns OPD along the beam.
 *
 * Role:
 *   Discriminates how the parser, topological compiler, spectral analyzer, bench editor, and
 *   per-tick DAG stepper interpret a cacaosim_node_cfg_t / cacaosim_bench_node_t record.
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
    CACAOSIM_NODE_WFE      = 7, /**< Static wavefront error map in OPD [m] ([wfe.<id>]) */
    CACAOSIM_NODE_ADC      = 8  /**< Atmospheric Dispersion Compensator ([adc.<id>]) */
} cacaosim_node_kind_t;

/**
 * struct cacaosim_node_cfg_t - Declarative configuration record for an optical node ([<type>.<id>])
 * @kind:          Optical element category (cacaosim_node_kind_t).
 * @id:            Unique section identifier within the bench (e.g. "dm0", "dichroic", "wfs_vis").
 * @in_beam:       Upstream optical beam name consumed by this node (mandatory for all kinds).
 * @out_beam:      Primary downstream optical beam produced by inline optics (DM, WFE, CORO).
 * @out_refl:      Reflected arm beam name produced by CACAOSIM_NODE_SPLITTER (optional if trans).
 * @out_trans:     Transmitted arm beam name produced by CACAOSIM_NODE_SPLITTER (optional if refl).
 * @out_lowfs:     Optional reflected LOWFS pupil pickoff beam name produced by CACAOSIM_NODE_CORO.
 * @in_cmd:        Input 2D float32 DM/TTM actuator command SHM stream name (auto-created in SHM).
 * @out_opd:       Optional output 2D float32 DM surface OPD [um] telemetry SHM stream name.
 * @in_flat:       Optional input 2D float32 static DM flat offset SHM stream name.
 * @in_wfe:        Input/auto-created 2D float32 static WFE OPD [m] SHM stream name for WFE nodes.
 * @out_cam:       Output 2D camera frame SHM stream name (uint16 or float32) for sensor nodes.
 * @out_long:      Optional output 2D float32 cumulative long-exposure PSF SHM stream (FPCAM only).
 * @in_ncpa:       Optional input 2D float32 per-sensor static NCPA OPD [um] SHM stream name.
 * @in_pup_amp:    Optional FITS path or SHM name for coronagraph pre-FPM pupil apodizer amplitude.
 * @in_pup_pha:    Optional FITS path or SHM name for coronagraph pre-FPM pupil phase/OPD mask.
 * @in_fpm_amp:    Optional FITS path or SHM name for custom coronagraph focal-plane amplitude.
 * @in_fpm_pha:    Optional FITS path or SHM name for custom coronagraph focal-plane phase/OPD.
 * @in_lyot:       Optional FITS path or SHM name for custom coronagraph Lyot-stop amplitude mask.
 * @in_lyot_pha:   Optional FITS path or SHM name for custom coronagraph Lyot-stop phase/OPD mask.
 * @ncpa_refl:     Optional 2D float32 SHM stream adding static NCPA OPD [um] on splitter refl arm.
 * @ncpa_trans:    Optional 2D float32 SHM stream adding static NCPA OPD [um] on splitter trans arm.
 * @pup_refl:      Optional 2D float32 SHM stream multiplying pupil amplitude on splitter refl arm.
 * @pup_trans:     Optional 2D float32 SHM stream multiplying pupil amplitude on splitter trans arm.
 * @tput:          Scalar optical power throughput factor [0..1] applied across all spectral bins.
 * @navg:          Integer master-clock subframes integrated per sensor readout when rate_hz <= 0.
 * @rate_hz:       Target sensor readout frame rate [Hz]; if > 0.1 Hz, overrides navg via
 *                 eff_navg = max(1, round(master_hz / rate_hz)).
 * @bw_frac:       Fractional spectral filter passband dlambda / lambda0 (e.g. 0.20 = 20% BW).
 * @qe:            Detector in-band quantum efficiency [0..1] used in photon budget integration.
 * @use_beam_flux: 1 to derive detector flux_ph from upstream 32-bin stellar spectrum; 0 for manual.
 * @wfe_rms_nm:    Synthesized pupil-plane static WFE RMS amplitude [nm OPD] when creating in_wfe.
 * @wfe_alpha:     Power-law spatial PSD amplitude exponent (PSD_amp ~ f^-alpha; default 1.6).
 * @wfe_gain:      Live dimensionless multiplier applied to the static WFE map (0 = bypass).
 * @wfe_seed:      Deterministic PRNG seed used when synthesizing the default static WFE map.
 * @reg:           2D affine pupil registration (shift_x, shift_y, rot_deg, scale) for DM/WFE/sens.
 * @reg_refl:      2D affine pupil registration applied to the splitter's reflected output arm.
 * @reg_trans:     2D affine pupil registration applied to the splitter's transmitted output arm.
 * @dm:            Hardware/influence parameters when kind == CACAOSIM_NODE_DM.
 * @split:         Spectral cutoff and efficiency parameters when kind == CACAOSIM_NODE_SPLITTER.
 * @coro:          Coronagraph mask, Lyot, and polychromatic parameters for CACAOSIM_NODE_CORO.
 * @pyrwfs:        4-sided Pyramid WFS optical & detector parameters for CACAOSIM_NODE_PYRWFS.
 * @shwfs:         Shack-Hartmann lenslet optical & detector parameters for CACAOSIM_NODE_SHWFS.
 * @zwfs:          Zernike phase-contrast WFS optical & detector parameters for CACAOSIM_NODE_ZWFS.
 * @fpcam:         Focal-plane science camera optical & detector parameters for CACAOSIM_NODE_FPCAM.
 *
 * Rationale:
 *   Keeping per-kind sub-configurations side-by-side in a flat Plain-Old-Data struct (rather than
 *   a C union or heap-allocated subclass hierarchy) lets interactive editors inspect or preserve
 *   parameters safely without union-aliasing pitfalls while keeping the entire bench copyable via
 *   trivial assignment.
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
    char                 out_state[CACAOSIM_BENCH_NAME_LEN];
    char                 in_flat[CACAOSIM_BENCH_NAME_LEN];
    char                 in_wfe[CACAOSIM_BENCH_NAME_LEN];
    char                 out_cam[CACAOSIM_BENCH_NAME_LEN];
    char                 out_long[CACAOSIM_BENCH_NAME_LEN];
    char                 out_pre_lyot[CACAOSIM_BENCH_NAME_LEN];
    char                 out_pre_lyot_long[CACAOSIM_BENCH_NAME_LEN];
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

    /* Throughput, cadence, clock skew, photometry, static WFE synthesis, and 2D registration */
    float                tput;
    uint32_t             navg;
    float                rate_hz;
    float                clock_offset_us;
    float                clock_drift_ppm;
    uint32_t             subframe_stride;
    float                sim_hz;
    float                bw_frac;
    float                qe;
    int32_t              use_beam_flux;
    int32_t              is_wfs;
    float                wfe_rms_nm;
    float                wfe_defocus_nm;
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
    cacaosim_adc_cfg_t    adc;
} cacaosim_node_cfg_t;

/**
 * struct cacaosim_bench_cfg_t - Complete declarative representation of a multi-stage .bench file
 * @name:       Human-readable bench identifier from [bench] (e.g. "two_stage_cascaded_ao").
 * @nwf:        Square pupil-plane wavefront grid dimension [pixels] shared across all beams (>= 8).
 * @pup_diam:   Clear telescope pupil outer diameter as a fraction of @nwf (e.g. 0.9375 = 120/128).
 * @cobs:       Linear central obscuration diameter ratio D_sec / D_tel [0..1).
 * @nspider:    Number of secondary-mirror support spider vanes synthesized in fallback pupil.
 * @spider_w:   Spider vane width as a fraction of clear pupil diameter.
 * @spider_deg: Spider orientation angle [degrees] counter-clockwise from +X.
 * @tel_diam_m: Physical primary mirror diameter [m] determining collecting area for photometry.
 * @star_mag:   Guide/science star apparent V-band Vega magnitude calibrating photon arrival rate.
 * @star_teff:  Stellar effective blackbody temperature [K] shaping the 0.40..2.40 um SED.
 * @tput_tel:   Combined sky + telescope + common-fore-optics scalar transmission [0..1].
 * @master_hz:  Bench master hardware simulation tick frequency [Hz] (dt_master_s = 1 / master_hz).
 * @in_wf:      Optional input 2D float32 atmospheric turbulence stream name (e.g. "outarraypha").
 * @in_amp:     Optional input 2D float32 atmospheric scintillation amplitude stream name.
 * @in_pup:     Optional input 2D float32 entrance pupil amplitude stream (fallback synthesized).
 * @in_ncpa:    Optional input 2D float32 common-path static NCPA OPD [um] stream name.
 * @out_beam:   Name of the root optical beam emitted by the telescope aperture (default "b_pup").
 * @wf_unit:    Physical unit of @in_wf: 0 = OPD [um], 1 = phase [rad] at @ref_lambda.
 * @ref_lambda: Reference wavelength [um] converting @in_wf radians to OPD microns when wf_unit==1.
 * @turb_gain:  Default dimensionless multiplier scaling incoming atmospheric turbulence.
 * @backend:    Requested compute backend (CACAOSIM_BACKEND_AUTO / _CPU / _CUDA).
 * @n_nodes:    Number of active node configurations stored in @nodes [0..CACAOSIM_BENCH_MAX_NODES].
 * @nodes:      Declaration-ordered array of optical hardware and sensor node configurations.
 *
 * Role:
 *   Acts as the pure data AST produced by cacaosim_bench_parse_str() / cacaosim_bench_preset(),
 *   mutated in-memory by cacaosim_bench_edit.c, serialized by cacaosim_bench_format_str(), and
 *   compiled into an executable DAG by cacaosim_bench_analyze() or cacaosim_bench_init().
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
    float               spider_offset;
    float               tel_diam_m;
    float               star_mag;
    float               star_teff;
    float               tput_tel;
    float               master_hz;
    uint32_t            oversample;
    int32_t             atm_interp;
    uint32_t            telemetry_stride;
    char                in_wf[CACAOSIM_BENCH_NAME_LEN];
    char                in_amp[CACAOSIM_BENCH_NAME_LEN];
    char                in_pup[CACAOSIM_BENCH_NAME_LEN];
    char                in_ncpa[CACAOSIM_BENCH_NAME_LEN];
    char                out_beam[CACAOSIM_BENCH_NAME_LEN];
    int32_t             wf_unit;
    float                   ref_lambda;
    float                   turb_gain;
    cacaosim_perturb_cfg_t  perturb;
    cacaosim_atm_disp_cfg_t atm_disp;
    int32_t                 imr_mode;
    float                   parallactic_rate_dps;
    float                   site_alt_m;
    int32_t                 chrom_turb;
    char                    chrom_lambdas[CACAOSIM_BENCH_NAME_LEN];
    int32_t                 backend;

    uint32_t            n_nodes;
    cacaosim_node_cfg_t nodes[CACAOSIM_BENCH_MAX_NODES];
} cacaosim_bench_cfg_t;

/**
 * struct cacaosim_beam_slot_t - Runtime named optical beam slot with spectrum & CPU/GPU wavefronts
 * @name:   Unique beam identifier (e.g. "b_pup", "b_dm0", "b_coro_sci").
 * @spec:   Steady-state 32-bin (0.40..2.40 um) photon arrival rate spectrum [photons/s per bin].
 * @nlam:   Number of coherent spatial wavelength planes carried by this beam (1 = achromatic pupil
 *          [OPD_um, Amp]; >1 = dispersive polychromatic cube emitted by a broadband coronagraph).
 * @lam_um: Center wavelength [um] for each plane c in [0..nlam-1] when @nlam > 1.
 * @opd:    64-byte aligned host OPD cube [nlam * nwf * nwf] in microns [um] (NULL in analyze mode).
 * @amp:    64-byte aligned host pupil amplitude cube [nlam * nwf * nwf] (NULL in analyze mode).
 * @d_opd:  CUDA device OPD cube [nlam * nwf * nwf] in microns [um] when use_cuda == 1, else NULL.
 * @d_amp:  CUDA device amplitude cube [nlam * nwf * nwf] when use_cuda == 1, else NULL.
 *
 * @img_shm: Optional connected 3D float32 SHM export handle (`ID == -1` when disabled).
 * @export_shm: 1 if this beam slot is exported to SHM when `export_beams_active` is 1.
 *
 * Rationale:
 *   Representing beams in pupil space as [OPD_um, Amplitude] makes achromatic reflective DMs,
 *   static WFEs, and dichroic relays wavelength-independent single-plane operations (nlam == 1)
 *   upstream of focal-plane masks, while allowing seamless promotion to [nlam, nwf, nwf] cubes
 *   downstream of chromatic coronagraphs without changing DAG connectivity.
 */
typedef struct
{
    char            name[CACAOSIM_BENCH_NAME_LEN];
    cacaosim_spec_t spec;
    cacaosim_disp_t disp;
    int32_t         poly_mode;
    uint32_t        nlam;
    float           lam_um[CACAOSIM_CORO_MAX_LAMBDA];
    float          *opd;
    float          *amp;
    float          *d_opd;
    float          *d_amp;
    IMGID           img_shm;
    int32_t         export_shm;
    int32_t         owns_buf;
} cacaosim_beam_slot_t;

/**
 * struct cacaosim_bench_node_t - Compiled runtime state for one topologically sorted DAG node
 * @cfg:             Resolved node configuration copy (with inherited nwf, pup_diam, eff_flux_ph).
 * @in_idx:          Index into ctx->beams[] for the consumed upstream beam (>= 0).
 * @out_idx:         Index into ctx->beams[] for primary inline output beam, or -1 if unused.
 * @refl_idx:        Index into ctx->beams[] for splitter reflected output beam, or -1 if unused.
 * @trans_idx:       Index into ctx->beams[] for splitter transmitted output beam, or -1 if unused.
 * @lowfs_idx:       Index into ctx->beams[] for coronagraph LOWFS pickoff beam, or -1 if unused.
 * @dm_slot:         Sequential DM slot index [0..ctx->n_dms-1] for DM/TTM nodes, else -1.
 * @sensor_slot:     Sequential sensor slot index [0..ctx->n_sensors-1] for WFS/FPCAM, else -1.
 * @eff_navg:        Quantized master-clock ticks integrated per camera readout (>= 1).
 * @eff_flux_ph:     Effective incident/detected photons per camera exposure [photons/readout].
 * @eff_lambda_um:   Flux-weighted effective central wavelength [um] across the sensor passband.
 * @eff_nlam:        Resolved number of polychromatic optical slices integrated per subframe.
 * @chan_lam_um:     Flux-weighted sub-band centroid wavelengths [um] for slices [0..eff_nlam-1].
 * @chan_weight:     Normalized sub-band photoevent weights summing to 1.0 across [0..eff_nlam-1].
 * @ncam:            Square output detector dimension [pixels] (n_sub * pix_per_sub for SHWFS).
 * @out_dtype:       Detector pixel datatype (CACAOSIM_DTYPE_U16 or CACAOSIM_DTYPE_F32).
 * @ready_this_tick: 1 on master ticks when this sensor completes an eff_navg readout, else 0.
 * @has_reg:         1 if @cfg.reg is non-identity on a sensor node and requires relay resampling.
 * @reg_opd:         Host scratch OPD cube [nlam * npix_wf] for pre-sensor 2D registration.
 * @reg_amp:         Host scratch amplitude cube [nlam * npix_wf] for pre-sensor 2D registration.
 * @d_reg_opd:       Device scratch OPD cube [nlam * npix_wf] for pre-sensor 2D registration.
 * @d_reg_amp:       Device scratch amplitude cube [nlam * npix_wf] for pre-sensor 2D registration.
 * @h_ncpa_refl:     Optional host pointer to splitter reflected-arm NCPA map [npix_wf], or NULL.
 * @h_ncpa_trans:    Optional host pointer to splitter transmitted-arm NCPA map [npix_wf], or NULL.
 * @h_pup_refl:      Optional host pointer to splitter reflected-arm pupil mask [npix_wf], or NULL.
 * @h_pup_trans:     Optional host pointer to splitter transmitted-arm pupil mask [npix_wf], NULL.
 * @d_ncpa_refl:     Device copy of @h_ncpa_refl when use_cuda == 1, else NULL.
 * @d_ncpa_trans:    Device copy of @h_ncpa_trans when use_cuda == 1, else NULL.
 * @d_pup_refl:      Device copy of @h_pup_refl when use_cuda == 1, else NULL.
 * @d_pup_trans:     Device copy of @h_pup_trans when use_cuda == 1, else NULL.
 * @cmd_buf:         Owned host actuator command array [nact_x * nact_y] when connect_shm == 0.
 * @dm_opd_buf:      Owned host DM surface OPD buffer [npix_wf] when use_cuda == 0 and no out_opd.
 * @wfe_m_buf:       Owned host static WFE map [npix_wf] in meters [m] when connect_shm == 0.
 * @d_wfe_m:         Device static WFE map [npix_wf] in meters [m] when use_cuda == 1.
 * @wfe_cnt0:        Last observed SHM metadata counter md->cnt0 on @img_wfe; triggers automatic
 *                   host-to-device re-upload of @d_wfe_m whenever an external user updates SHM.
 * @cam_u16:         Owned host uint16 camera frame buffer [ncam * ncam] when connect_shm == 0.
 * @cam_f32:         Owned host float32 camera frame buffer [ncam * ncam] when connect_shm == 0.
 * @long_f32:        Owned host float32 long-exposure PSF buffer [ncam * ncam] when connect_shm==0.
 * @img_cmd:         Bound SHM handle for DM actuator commands when connect_shm == 1 (ID == -1 off).
 * @img_opd:         Bound SHM handle for optional DM OPD output when connect_shm == 1.
 * @img_flat:        Bound SHM handle for optional DM flat map input when connect_shm == 1.
 * @img_wfe:         Bound SHM handle for static WFE [m] map when connect_shm == 1.
 * @img_cam:         Bound SHM handle for sensor detector output stream when connect_shm == 1.
 * @img_long:        Bound SHM handle for FPCAM cumulative long-exposure PSF when connect_shm == 1.
 * @img_ncpa:        Bound SHM handle for optional per-sensor static NCPA map when connect_shm == 1.
 * @img_lyot:        Reserved SHM handle slot for coronagraph Lyot stream lifecycle tracking.
 * @img_ncpa_refl:   Bound SHM handle for splitter reflected-arm NCPA stream.
 * @img_ncpa_trans:  Bound SHM handle for splitter transmitted-arm NCPA stream.
 * @img_pup_refl:    Bound SHM handle for splitter reflected-arm pupil mask stream.
 * @img_pup_trans:   Bound SHM handle for splitter transmitted-arm pupil mask stream.
 * @dm_ctx:          Active DM / TTM hardware state & ZOH filter buffers (CACAOSIM_NODE_DM).
 * @coro_ctx:        Active coronagraph MFT/FFT plans & mask buffers (CACAOSIM_NODE_CORO).
 * @pyrwfs_ctx:      Active Pyramid WFS FFT plans, CUDA Graph, & accumulator (CACAOSIM_NODE_PYRWFS).
 * @shwfs_ctx:       Active Shack-Hartmann batched FFT plans & accumulator (CACAOSIM_NODE_SHWFS).
 * @zwfs_ctx:        Active Zernike WFS Babinet MFT matrices & accumulator (CACAOSIM_NODE_ZWFS).
 * @fpcam_ctx:       Active science camera MFT/FFT plans, Strehl & long-PSF (CACAOSIM_NODE_FPCAM).
 *
 * Rationale:
 *   Resolving all string beam names into integer slot indices (@in_idx, @out_idx, etc.) and
 *   pre-allocating every registration scratch buffer and SHM handle during cacaosim_bench_init()
 *   ensures cacaosim_bench_step() performs zero string lookups, zero heap allocations, and zero
 *   branching on resource initialization inside the multi-kHz hot loop.
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

    /* Effective cadence, DM delay, subframe stride, clock phase, and photon flux per readout */
    uint32_t            eff_navg;
    float               eff_delay_ticks;
    uint32_t            eff_stride;
    uint32_t            stagger_tick;
    double              clock_phase;
    double              clock_phase_init;
    uint32_t            prev_subframe_cnt;
    int32_t             alias_refl;
    int32_t             alias_trans;
    float               eff_flux_ph;
    float               eff_lambda_um;
    int32_t             eff_nlam;
    float               chan_lam_um[CACAOSIM_MAX_LAMBDA];
    float               chan_weight[CACAOSIM_MAX_LAMBDA];
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
    IMGID               img_state;
    IMGID               img_flat;
    IMGID               img_wfe;
    IMGID               img_cam;
    IMGID               img_long;
    IMGID               img_ncpa;
    IMGID               img_lyot;
    IMGID               img_pre_lyot;
    IMGID               img_pre_lyot_long;
    IMGID               img_ncpa_refl;
    IMGID               img_ncpa_trans;
    IMGID               img_pup_refl;
    IMGID               img_pup_trans;

    /* Active compute unit context */
    cacaosim_dm_ctx_t     dm_ctx;
    cacaosim_adc_ctx_t    adc_ctx;
    cacaosim_coro_ctx_t   coro_ctx;
    cacaosim_pyrwfs_ctx_t pyrwfs_ctx;
    cacaosim_shwfs_ctx_t  shwfs_ctx;
    cacaosim_zwfs_ctx_t   zwfs_ctx;
    cacaosim_fpcam_ctx_t  fpcam_ctx;
} cacaosim_bench_node_t;

/**
 * struct cacaosim_bench_ctx_t - Compiled runtime execution & analysis context for a .bench DAG
 * @cfg:              Snapshot of the top-level bench configuration compiled into this context.
 * @connect_shm:      1 when bound to live ImageStreamIO SHM streams; 0 for standalone pure-C mode.
 * @use_cuda:         1 when executing optical kernels on CUDA GPU via @cuda_stream; 0 for CPU.
 * @nwf:              Pupil wavefront grid linear size [pixels].
 * @npix_wf:          Total pixels per 2D pupil plane (@nwf * @nwf).
 * @dt_master_s:      Master hardware clock period [seconds] (1.0f / cfg.master_hz).
 * @tick_cnt:         Monotonically increasing count of completed cacaosim_bench_step() ticks.
 * @cuda_stream:      Opaque cudaStream_t handle shared sequentially by all GPU nodes (or NULL).
 * @fallback_pup:     Owned host anti-aliased pupil buffer [npix_wf] when no external/SHM pupil.
 * @pup_ptr:          Active read-only host entrance pupil amplitude pointer [npix_wf].
 * @ncpa_ptr:         Optional host common-path static NCPA OPD [um] pointer [npix_wf], or NULL.
 * @d_pup_amp:        Device entrance pupil amplitude [npix_wf] uploaded at init when use_cuda==1.
 * @d_ncpa_opd:       Device entrance common-path NCPA OPD [npix_wf] when use_cuda==1, else NULL.
 * @d_wf_in:          Device staging buffer [npix_wf] for per-tick incoming atmospheric wavefront.
 * @d_atm_amp:        Device staging buffer [npix_wf] for per-tick atmospheric scintillation amp.
 * @img_pup:          Entrance pupil SHM handle when connect_shm == 1 (ID == -1 if unused).
 * @img_ncpa:         Entrance common-path NCPA SHM handle when connect_shm == 1.
 * @img_wf:           Entrance atmospheric phase/OPD SHM handle (lazily reconnected each tick).
 * @img_amp:          Entrance atmospheric amplitude SHM handle (lazily reconnected each tick).
 * @n_beams:          Number of registered named optical beams in @beams [1..MAX_BEAMS].
 * @beams:            Beam table; beams[0] is always the root telescope entrance beam.
 * @n_nodes:          Number of topologically sorted optical nodes in @nodes [1..MAX_NODES].
 * @nodes:            Nodes ordered so every producer executes strictly before its consumers.
 * @n_dms:            Count of DM and TTM nodes indexed in @dm_node_idx.
 * @dm_node_idx:      Maps sequential DM slot [0..n_dms-1] -> sorted index in @nodes.
 * @n_sensors:        Count of terminal WFS and FPCAM nodes indexed in @sensor_node_idx.
 * @sensor_node_idx:  Maps sequential sensor slot [0..n_sensors-1] -> sorted index in @nodes.
 * @last_strehl_inst: Latest short-exposure peak Strehl ratio [0..1] from FPCAM readout.
 * @last_strehl_ave:  Latest cumulative long-exposure peak Strehl ratio [0..1] from FPCAM readout.
 *
 * Role:
 *   Holds either lightweight static topology + 32-bin spectral analysis results (when populated
 *   by cacaosim_bench_analyze()) or the full GPU/CPU memory arena, FFT/cuFFT/cuBLAS handles, and
 *   ImageStreamIO bindings (when initialized by cacaosim_bench_init()). Always release initialized
 *   contexts via cacaosim_bench_free().
 */
typedef struct
{
    cacaosim_bench_cfg_t  cfg;
    int32_t               connect_shm;
    int32_t               use_cuda;
    uint32_t              nwf;
    uint64_t              npix_wf;
    uint32_t              eff_oversample;
    float                 eff_master_hz;
    float                 dt_master_s;
    uint64_t              tick_cnt;
    void                 *cuda_stream;

    /* Entrance pupil, static NCPA, and dual-screen atmosphere interpolation buffers */
    float                *fallback_pup;
    const float          *pup_ptr;
    const float          *ncpa_ptr;
    float                *h_wf_prev;
    float                *h_wf_curr;
    float                *d_pup_amp;
    float                *d_ncpa_opd;
    float                *d_wf_in;
    float                *d_wf_prev;
    float                *d_wf_curr;
    float                *d_atm_amp;
    uint64_t              wf_prev_cnt0;
    uint32_t              wf_substep;
    int32_t               wf_interp_primed;
    uint32_t              wf_chrom_nwl;
    int32_t               wf_chrom_unit;
    float                 wf_chrom_wls[8];
    float                *d_wf_chrom_in;

    /* Atmospheric dispersion state & IMR parallactic tracking */
    double                  atm_disp_pa_curr_deg;
    cacaosim_atm_disp_cfg_t cached_atm_cfg;
    int32_t                 cached_atm_valid;
    cacaosim_disp_t         cached_atm_disp;

    /* Entrance-pupil vibration & thermal petaling engine */
    cacaosim_perturb_ctx_t perturb;

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
    int32_t               export_beams_active;

    /* Lockstep step-and-shoot / temporal calibration & in-DAG wfstat liveness controls */
    int32_t               calib_fast;
    int32_t               calib_temporal;
    int32_t               calib_dm_node;
    const float          *calib_cmd_override;
    char                  calib_wfs_out[CACAOSIM_BENCH_NAME_LEN];
    int32_t               skip_fpcam;
    int32_t               wfstat_step_active;
    int32_t               wfstat_snap_active;
    int32_t               wfstat_beam_idx[3];
} cacaosim_bench_ctx_t;

/**
 * enum cacaosim_preset_t - Built-in reference AO optical bench templates
 * @CACAOSIM_PRESET_SINGLE_PYRWFS:  Classic single-conjugate AO: 20x20 DM -> 0.95 um LP dichroic ->
 *                                  2 kHz visible Pyramid WFS + 200 Hz H-band science camera.
 * @CACAOSIM_PRESET_SINGLE_SHWFS:   Two-corrector SCAO: fast TTM + 16x16 DM -> 1.00 um LP dichroic
 *                                  -> 16x16 visible Shack-Hartmann WFS + H-band science camera.
 * @CACAOSIM_PRESET_TWO_STAGE:      Cascaded multi-rate woofer/tweeter ExAO: 12x12 woofer -> J-band
 *                                  band-reflect dichroic (1 kHz J-PyWFS) -> 24x24 tweeter -> LP
 *                                  dichroic -> 2 kHz visible PyWFS + 100 Hz H-band science camera.
 * @CACAOSIM_PRESET_EXAO_CORO:      Coronagraphic ExAO: TTM + 24x24 DM -> visible HOWFS + H-band
 *                                  Lyot spot coronagraph with FPM-reflected 500 Hz Zernike LOWFS
 *                                  and post-Lyot 100 Hz coronagraphic science camera.
 * @CACAOSIM_PRESET_TRUTH_DUAL_WFS: Dual-WFS truth sensing: 20x20 DM -> LP dichroic -> 80/20 gray
 *                                  splitter feeding a 2 kHz fast PyWFS and a 100 Hz truth SHWFS.
 * @CACAOSIM_PRESET_MINIMAL:        Open-loop baseline: 8.2 m entrance pupil feeding a direct
 *                                  H-band focal-plane science camera with no active optics.
 * @CACAOSIM_PRESET_COUNT:          Total number of built-in presets for menu enumeration.
 *
 * Role:
 *   Provides ready-to-run, physically realistic reference topologies for interactive bootstrapping
 *   in cacaosim-CTRL, CLI scaffolding, and end-to-end multi-stage integration tests.
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
 * cacaosim_bench_cfg_defaults() - Populate a top-level .bench configuration with VLT/Subaru-class
 *                                 AO defaults
 * @cfg: Caller-owned destination configuration struct (must not be NULL; zeroed and initialized).
 *
 * Role:
 *   Establishes a canonical baseline [bench] state (128x128 pupil, 8.2 m primary, V=8.0 G2V star
 *   at 5778 K, 2 kHz master clock, phase radians at 0.50 um on "outarraypha", auto GPU/CPU
 *   backend, n_nodes = 0) prior to parsing a .bench file or programmatically adding nodes.
 *
 * Rationale:
 *   Centralizing physical defaults ensures omitted keys in user .bench files always resolve to
 *   consistent, physically meaningful telescope and stellar parameters across CLI, TUI, and C API
 *   callers.
 */
void cacaosim_bench_cfg_defaults(cacaosim_bench_cfg_t *cfg);

/**
 * cacaosim_bench_node_defaults() - Initialize a single optical node configuration with physically
 *                                  realistic defaults
 * @nd:     Caller-owned destination node configuration (ignored safely if NULL).
 * @kind:   Optical node category (cacaosim_node_kind_t) to configure.
 * @id:     Section identifier string copied into @nd->id (falls back to "node" if NULL).
 * @is_ttm: Non-zero to configure a 2-axis Tip-Tilt mirror (dm_mode = 1, 2x1 actuators) when
 *          @kind == CACAOSIM_NODE_DM; 0 for a zonal actuator grid (16x16).
 * @idx:    0-based node slot index used to salt deterministic per-node PRNG seeds so sibling
 *          detectors and synthesized WFE maps are statistically independent.
 *
 * Role:
 *   Zero-initializes @nd and populates kind-specific optical, mechanical, photometric, and
 *   detector parameters along with default SHM stream names derived from @id.
 *
 * Rationale:
 *   Sharing a single default factory between the .bench parser (when opening a new [<type>.<id>]
 *   section) and the interactive editor (when inserting a node) guarantees identical out-of-the-box
 *   behavior and deterministic seed decorrelation across both code paths.
 */
void cacaosim_bench_node_defaults(
    cacaosim_node_cfg_t *nd,
    cacaosim_node_kind_t kind,
    const char          *id,
    int                  is_ttm,
    uint32_t             idx);

/**
 * cacaosim_bench_parse_str() - Parse an INI/TOML-subset .bench text buffer into @cfg
 * @text:    NUL-terminated configuration text (must not be NULL; caller retains ownership).
 * @cfg:     Caller-owned destination configuration struct (must not be NULL; reset to defaults).
 * @err_buf: Optional caller-owned diagnostic buffer receiving a human-readable line-numbered
 *           error message on failure (may be NULL).
 * @err_cap: Capacity of @err_buf in bytes (0 if @err_buf is NULL).
 *
 * Role:
 *   Translates declarative [bench] and [<type>.<id>] sections (supporting '#' and ';' comments,
 *   quoted or unquoted values, symbolic enum names, and physical-unit aliases) into an in-memory
 *   cacaosim_bench_cfg_t AST and validates basic invariants (unique node IDs, nwf >= 8, n_nodes>0).
 *
 * Rationale:
 *   Operating purely on an in-memory const char* buffer allows presets, unit tests, and file
 *   loaders to share the exact same zero-dependency parser with thread-safe, caller-scoped error
 *   reporting and zero global state.
 *
 * Return: 0 on success, -1 on syntax, bounds, or validation error.
 */
int cacaosim_bench_parse_str(
    const char           *text,
    cacaosim_bench_cfg_t *cfg,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_load_file() - Read a .bench file from disk and parse it into @cfg
 * @path:    Filesystem path to the .bench file (must be non-NULL and non-empty; <= 1 MiB).
 * @cfg:     Caller-owned destination configuration struct (must not be NULL).
 * @err_buf: Optional caller-owned error message buffer (may be NULL).
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Slurps a bounded .bench file from disk into a temporary heap buffer and delegates parsing and
 *   validation to cacaosim_bench_parse_str().
 *
 * Rationale:
 *   Enforcing a strict 1 MiB file size cap guards against accidental binary/FITS paths passed on
 *   the command line while keeping I/O atomic and single-pass.
 *
 * Return: 0 on success, -1 on I/O, size, or parse error (populates @err_buf).
 */
int cacaosim_bench_load_file(
    const char           *path,
    cacaosim_bench_cfg_t *cfg,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_format_str() - Serialize a .bench AST into canonical INI/TOML-subset text
 * @cfg: Read-only source bench configuration (must not be NULL).
 * @buf: Optional caller-owned destination character buffer (may be NULL when @cap == 0 to compute
 *       exact required byte length via dry-run sizing).
 * @cap: Capacity of @buf in bytes including trailing NUL.
 *
 * Role:
 *   Emits a clean, human-readable .bench document containing a usage header, the [bench] section,
 *   and every [<type>.<id>] node in declaration order while eliding redundant default-valued
 *   optional keys (e.g. identity 2D registration or unused NCPA stream names).
 *
 * Rationale:
 *   Following snprintf() sizing semantics allows cacaosim_bench_save_file() to measure exact
 *   output length in pass 1 and format without truncation in pass 2, ensuring lossless round-trip
 *   persistence from cacaosim-CTRL and programmatic editors.
 *
 * Return: Total characters that would be written (excluding NUL), or -1 if @cfg is NULL.
 */
int cacaosim_bench_format_str(
    const cacaosim_bench_cfg_t *cfg,
    char                       *buf,
    size_t                      cap);

/**
 * cacaosim_bench_save_file() - Format and write a .bench configuration file to disk
 * @path:    Destination filesystem path (must be non-NULL and non-empty).
 * @cfg:     Read-only source bench configuration (must not be NULL).
 * @err_buf: Optional caller-owned diagnostic buffer populated on failure (may be NULL).
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Serializes @cfg into a dynamically sized buffer via cacaosim_bench_format_str() and writes
 *   the complete text to @path, verifying full byte count and stream flush/close status.
 *
 * Rationale:
 *   Formatting into memory first avoids leaving half-written .bench files if configuration
 *   validation fails and guarantees exact byte-count verification on disk write.
 *
 * Return: 0 on success, -1 on invalid arguments, allocation failure, or disk I/O error.
 */
int cacaosim_bench_save_file(
    const char                 *path,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_preset_name() - Return short human-readable title for built-in preset @preset_id
 * @preset_id: Enumerator in [0 .. CACAOSIM_PRESET_COUNT - 1] (cacaosim_preset_t).
 *
 * Role:
 *   Supplies concise menu labels for preset pickers in cacaosim-CTRL and CLI help listings.
 *
 * Rationale:
 *   Returning static string literals avoids allocation and keeps UI labels synchronized with the
 *   preset table; out-of-range IDs safely return "Unknown Preset".
 *
 * Return: Static NUL-terminated title string (never NULL).
 */
const char *cacaosim_bench_preset_name(int preset_id);

/**
 * cacaosim_bench_preset_desc() - Return one-line optical chain summary for preset @preset_id
 * @preset_id: Enumerator in [0 .. CACAOSIM_PRESET_COUNT - 1] (cacaosim_preset_t).
 *
 * Role:
 *   Describes the optical path, wavelengths, and cadences of a preset for TUI/CLI tooltips.
 *
 * Rationale:
 *   Gives users immediate visibility into corrector counts, dichroic splits, and WFS types before
 *   overwriting their active bench configuration; out-of-range IDs safely return "".
 *
 * Return: Static NUL-terminated summary string (never NULL).
 */
const char *cacaosim_bench_preset_desc(int preset_id);

/**
 * cacaosim_bench_preset() - Populate @cfg from a built-in reference AO bench template
 * @cfg:       Caller-owned destination bench configuration (must not be NULL).
 * @preset_id: Preset selector in [0 .. CACAOSIM_PRESET_COUNT - 1] (cacaosim_preset_t).
 *
 * Role:
 *   Loads a complete, ready-to-run single-stage, two-stage, coronagraphic ExAO, dual-WFS, or
 *   minimal bench architecture into @cfg.
 *
 * Rationale:
 *   Implemented by parsing embedded canonical .bench INI strings through cacaosim_bench_parse_str()
 *   so built-in presets exercise the exact same parser defaults and syntax rules as user files.
 *
 * Return: 0 on success, -1 if @cfg is NULL or @preset_id is out of range.
 */
int cacaosim_bench_preset(
    cacaosim_bench_cfg_t *cfg,
    int                   preset_id);

/**
 * cacaosim_bench_list_beams() - Enumerate all distinct optical beam names produced in @cfg
 * @cfg:       Read-only bench configuration to inspect (safe if NULL -> returns 0).
 * @beams:     Caller-allocated 2D character array [max_beams][CACAOSIM_BENCH_NAME_LEN] (or NULL).
 * @max_beams: Maximum entries to write into @beams.
 *
 * Role:
 *   Collects the root entrance beam (@cfg->out_beam) followed by every unique beam produced by
 *   inline, splitter, and coronagraph nodes in declaration order.
 *
 * Rationale:
 *   Powers interactive beam-selector cycling in cacaosim-CTRL so users can re-route sensors or
 *   optics across valid produced beams without typing raw identifiers.
 *
 * Return: Number of unique beam names written to @beams [0..max_beams].
 */
uint32_t cacaosim_bench_list_beams(
    const cacaosim_bench_cfg_t *cfg,
    char                        beams[][CACAOSIM_BENCH_NAME_LEN],
    uint32_t                    max_beams);

/**
 * cacaosim_bench_add_node() - Insert an optical element into @cfg and splice its beam connections
 * @cfg:            Bench configuration to mutate in-place (must not be NULL).
 * @kind:           Optical node type to create (cacaosim_node_kind_t).
 * @is_ttm:         Non-zero to create a 2-axis Tip-Tilt mirror when @kind == CACAOSIM_NODE_DM.
 * @after_node_idx: Anchor node index [0..n_nodes-1] after which to insert, or -1 to splice at the
 *                  telescope entrance beam.
 * @target_beam:    Explicit upstream beam name to attach to, or NULL/"" to infer automatically
 *                  from @after_node_idx and @kind.
 *
 * Role:
 *   Mints a unique node ID, unique output beam names, and unique SHM stream names; splices inline
 *   optics (DM, WFE, CORO) or splitters into the target beam by rewiring existing downstream
 *   consumers onto the newly created output/transmitted arm; auto-tunes new sensor wavelengths to
 *   match upstream dichroic passbands; and inserts the node at a causally valid array position.
 *
 * Rationale:
 *   Encapsulates optical graph surgery so interactive "Add DM / Splitter / WFS" actions never
 *   create duplicate beam producers, SHM stream collisions, or orphaned downstream branches.
 *
 * Return: Array index [0..n_nodes-1] of the newly inserted node, or -1 if @cfg is NULL or full.
 */
int cacaosim_bench_add_node(
    cacaosim_bench_cfg_t *cfg,
    cacaosim_node_kind_t  kind,
    int                   is_ttm,
    int                   after_node_idx,
    const char           *target_beam);

/**
 * cacaosim_bench_clone_node() - Duplicate node @src_idx with collision-free IDs, beams, and SHMs
 * @cfg:     Bench configuration to mutate in-place (must not be NULL).
 * @src_idx: Source node index [0..n_nodes-1] to duplicate.
 *
 * Role:
 *   Copies all optical, dynamical, and detector settings of `cfg->nodes[src_idx]` into a newly
 *   spliced sibling/downstream node while preserving freshly minted unique node IDs, beam names,
 *   SHM stream names, and an offset WFE PRNG seed.
 *
 * Rationale:
 *   Allows rapid construction of multi-DM or multi-WFS benches from an already tuned component
 *   without inadvertently sharing actuator command streams, camera output streams, or static WFE
 *   realizations.
 *
 * Return: Index [0..n_nodes-1] of the cloned node, or -1 on invalid index or full node table.
 */
int cacaosim_bench_clone_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              src_idx);

/**
 * cacaosim_bench_remove_node() - Delete node @node_idx and heal upstream-to-downstream continuity
 * @cfg:      Bench configuration to mutate in-place (must not be NULL).
 * @node_idx: Target node index [0..n_nodes-1] to remove.
 *
 * Role:
 *   Redirects every downstream node that consumed any output beam of `nodes[node_idx]`
 *   (out_beam, out_trans, out_refl, out_lowfs) back onto its upstream input beam (`in_beam`),
 *   then compacts `cfg->nodes[]`.
 *
 * Rationale:
 *   Healing beam edges upon removal ensures deleting an inline DM, static WFE, coronagraph, or
 *   beamsplitter preserves a connected, compile-ready optical bench rather than stranding all
 *   downstream sensors on a non-existent beam name.
 *
 * Return: 0 on success, -1 if @cfg is NULL or @node_idx >= cfg->n_nodes.
 */
int cacaosim_bench_remove_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx);

/**
 * cacaosim_bench_move_node() - Reorder node @node_idx by @dir (-1 up/upstream, +1 down/downstream)
 * @cfg:      Bench configuration to mutate in-place (must not be NULL).
 * @node_idx: Target node index [0..n_nodes-1] to move.
 * @dir:      Step direction: -1 to swap with predecessor, +1 to swap with successor.
 *
 * Role:
 *   Swaps `nodes[node_idx]` with its adjacent neighbor and, when the two nodes form a connected
 *   inline chain (Inline->Inline, Inline->Splitter, or Splitter->Inline), rewires their input and
 *   downstream consumer beams so physical optical causality swaps cleanly with declaration order.
 *
 * Rationale:
 *   In a beam-named DAG, merely swapping array slots without rewiring beams would leave physical
 *   optical order unchanged; splicing beam edges during adjacent moves makes "move up/down" in
 *   cacaosim-CTRL genuinely commute optical elements (e.g. moving a DM before vs after a splitter
 *   or WFE).
 *
 * Return: New index [0..n_nodes-1] of the moved node, or -1 if out of bounds.
 */
int cacaosim_bench_move_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx,
    int                   dir);

/**
 * cacaosim_bench_swap_branches() - Exchange downstream consumers between dual output arms
 * @cfg:      Bench configuration to mutate in-place (must not be NULL).
 * @node_idx: Index [0..n_nodes-1] of a CACAOSIM_NODE_SPLITTER or dual-arm CACAOSIM_NODE_CORO node.
 *
 * Role:
 *   Swaps all downstream references between `out_refl` and `out_trans` (for a splitter) or
 *   between `out_beam` and `out_lowfs` (for a coronagraph).
 *
 * Rationale:
 *   Lets users flip which optical train sits on the reflected vs transmitted arm of a dichroic
 *   in one atomic operation without manually editing every downstream node's `in_beam`.
 *
 * Return: 0 on success, -1 if @cfg is NULL, @node_idx is invalid, or node lacks two active arms.
 */
int cacaosim_bench_swap_branches(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx);

/**
 * cacaosim_bench_rename_node() - Rename node @node_idx to a validated unique identifier @new_id
 * @cfg:      Bench configuration to mutate in-place (must not be NULL).
 * @node_idx: Index [0..n_nodes-1] of the node to rename.
 * @new_id:   Proposed identifier string (must match `[A-Za-z0-9_-]{1,63}` and be unique).
 * @err_buf:  Optional caller-owned diagnostic buffer populated on rejection (may be NULL).
 * @err_cap:  Capacity of @err_buf in bytes.
 *
 * Role:
 *   Validates character set, length, and bench-wide uniqueness of @new_id before updating
 *   `cfg->nodes[node_idx].id`.
 *
 * Rationale:
 *   Restricting identifiers to `[A-Za-z0-9_-]` guarantees clean `[<type>.<id>]` INI section
 *   headers without dots, brackets, or whitespace that would break round-trip serialization.
 *
 * Return: 0 on success, -1 on validation or uniqueness failure.
 */
int cacaosim_bench_rename_node(
    cacaosim_bench_cfg_t *cfg,
    uint32_t              node_idx,
    const char           *new_id,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_rewire_beam_out() - Redirect all consumers of @old_beam onto @new_beam
 * @cfg:      Bench configuration to mutate in-place (must not be NULL).
 * @old_beam: Existing produced beam name whose consumers should be updated (must not be NULL).
 * @new_beam: Replacement non-empty beam name.
 *
 * Role:
 *   Updates `in_beam` across every node in @cfg that previously referenced @old_beam so renaming
 *   an upstream producer's output beam preserves graph connectivity.
 *
 * Rationale:
 *   Keeps producer output-beam renames atomic in interactive editors so downstream nodes never
 *   break due to stale beam names.
 *
 * Return: 0 on success, -1 if arguments are NULL or @new_beam is empty.
 */
int cacaosim_bench_rewire_beam_out(
    cacaosim_bench_cfg_t *cfg,
    const char           *old_beam,
    const char           *new_beam);

/**
 * cacaosim_bench_init() - Compile, validate, and allocate a runnable CPU/CUDA multi-stage DAG
 * @ctx:         Caller-owned runtime context to initialize (must not be NULL; zeroed on entry).
 * @cfg:         Read-only parsed bench configuration to compile (must not be NULL).
 * @ext_pup:     Optional caller-owned host pupil amplitude [nwf * nwf] overriding SHM/fallback
 *               pupil generation when non-NULL (caller must keep alive if CPU mode reads it).
 * @connect_shm: Non-zero to connect/create live ImageStreamIO SHM streams; 0 for standalone
 *               in-memory execution using internal host buffers (`cmd_buf`, `cam_u16`, `cam_f32`).
 * @err_buf:     Optional caller-owned diagnostic buffer populated on failure (may be NULL).
 * @err_cap:     Capacity of @err_buf in bytes.
 *
 * Role:
 *   1. Registers all produced beams, verifies single-producer uniqueness and valid inputs, and
 *      topologically sorts nodes via Kahn's algorithm;
 *   2. Creates the shared CUDA stream (when CUDA is requested and available) and initializes
 *      entrance pupil, NCPA, and named beam CPU/GPU buffers;
 *   3. Computes multi-rate `eff_navg` factors and propagates the 32-bin stellar photon budget;
 *   4. Initializes every DM, splitter, coronagraph (loading optional FITS/SHM masks and promoting
 *      downstream polychromatic beam cubes), static WFE, WFS, and science camera context, rolling
 *      back all partial allocations via cacaosim_bench_free() on any error.
 *
 * Rationale:
 *   Front-loading all topological validation, FFTW/cuFFT plan creation, CUDA Graph capture, mask
 *   loading, and SHM binding into a single transactional setup routine guarantees that
 *   cacaosim_bench_step() executes with zero dynamic memory allocation or setup jitter.
 *
 * Return: 0 on success, -1 on topology, allocation, mask I/O, or SHM error.
 */
int cacaosim_bench_init(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    const float                *ext_pup,
    int32_t                     connect_shm,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_analyze() - Perform zero-allocation topological validation & spectral dry-run
 * @ctx:     Caller-owned context receiving sorted nodes, beam table, cadences, and 32-bin spectra
 *           (must not be NULL; safe to pass to cacaosim_bench_free(), though owns zero heap).
 * @cfg:     Read-only parsed bench configuration to analyze (must not be NULL).
 * @err_buf: Optional caller-owned diagnostic buffer populated on topology error (may be NULL).
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Runs the exact same single-producer beam registration, Kahn topological sort, DM/sensor slot
 *   indexing, `eff_navg` multi-rate quantization, and 32-bin Planck blackbody + dichroic + LOWFS
 *   spectral photometry integration as cacaosim_bench_init(), without allocating pixel buffers,
 *   FFT plans, CUDA streams, or ImageStreamIO handles.
 *
 * Rationale:
 *   Enables sub-100-microsecond live validation, schematic layout, and per-sensor photon/frame
 *   readout previews inside `cacaosim-CTRL` and `cacaosim-bench-view` without touching GPU drivers
 *   or `/milk/shm`.
 *
 * Return: 0 on success, -1 on NULL input, cycle detection, duplicate beam, or missing beam error.
 */
int cacaosim_bench_analyze(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_free() - Release all compute-unit, host, CUDA, and SHM resources owned by @ctx
 * @ctx: Bench context previously populated by cacaosim_bench_init() or cacaosim_bench_analyze()
 *       (safe no-op if NULL or already zeroed; zeroed before return).
 *
 * Role:
 *   Destroys per-node DM/coro/WFS/FPCAM contexts, frees aligned host buffers, device allocations,
 *   and the shared CUDA stream, disconnects all valid ImageStreamIO `IMGID` handles, and zeroes
 *   @ctx so repeated calls are idempotent.
 *
 * Rationale:
 *   Provides a single deterministic teardown path used both for normal shutdown and for clean
 *   mid-initialization rollback inside cacaosim_bench_init().
 */
void cacaosim_bench_free(cacaosim_bench_ctx_t *ctx);

/**
 * cacaosim_bench_print_summary() - Format a concise topology and photon-budget table to @fp
 * @ctx: Analyzed or initialized bench context (ignored safely if NULL).
 * @fp:  Destination stdio stream such as `stdout` or `stderr` (ignored safely if NULL).
 *
 * Role:
 *   Prints bench geometry, stellar magnitude, master clock, active compute backend, total
 *   0.40..2.40 um entrance photon rate, and one line per topologically sorted node showing beam
 *   routing, SHM streams, effective readout rate [Hz], `eff_navg`, and integrated photons/frame.
 *
 * Rationale:
 *   Gives non-TTY daemon logs (`cacaosim.htsim`), CLI runs, and benchmark harnesses a compact,
 *   greppable record of the compiled optical graph and photometric operating point.
 */
void cacaosim_bench_print_summary(
    const cacaosim_bench_ctx_t *ctx,
    FILE                       *fp);

/**
 * cacaosim_bench_step() - Advance the entire compiled optical bench DAG by one master-clock tick
 * @ctx:       Initialized runtime bench context from cacaosim_bench_init() (no-op if NULL).
 * @ext_wf:    Optional caller-supplied host input wavefront [nwf * nwf] overriding `img_wf` SHM
 *             when non-NULL (unit governed by `ctx->cfg.wf_unit`).
 * @ext_amp:   Optional caller-supplied host input scintillation amplitude [nwf * nwf] overriding
 *             `img_amp` SHM when non-NULL.
 * @turb_gain: Live dimensionless multiplier applied to the incoming atmospheric wavefront (set to
 *             0.0f to freeze/disable turbulence during interaction-matrix calibration).
 * @procinfo:  Optional `PROCESSINFO *` handle passed to `processinfo_update_output_stream()` when
 *             publishing updated SHM streams (may be NULL).
 *
 * Role:
 *   Synthesizes the entrance pupil beam `[OPD_um, Amplitude]` for tick `ctx->tick_cnt`, traverses
 *   all topologically sorted nodes without intermediate host/GPU synchronization (stepping DM ZOH
 *   dynamics, applying static WFE maps with automatic SHM `cnt0` hot-reload, relaying dichroic
 *   arms, propagating coronagraphs, and accumulating sensor subframes), and triggers noisy
 *   detector readout + SHM semaphore publication on sensors reaching `subframe_cnt >= eff_navg`.
 *
 * Rationale:
 *   Executing every optical element and multi-rate subframe accumulator sequentially on a single
 *   thread / single `cudaStream_t` eliminates inter-thread locking, semaphore latency, and PCIe
 *   round-trips while faithfully modeling fractional-frame DM delays and finite-exposure WFS /
 *   science PSF integration.
 */
void cacaosim_bench_step(
    cacaosim_bench_ctx_t *ctx,
    const float          *ext_wf,
    const float          *ext_amp,
    float                 turb_gain,
    void                 *procinfo);

/**
 * cacaosim_bench_find_dm() - Look up sequential DM slot index [0..n_dms-1] by node section @id
 * @ctx: Analyzed or initialized bench context (safe if NULL -> returns -1).
 * @id:  Node section identifier string, e.g. "dm0" or "woofer" (safe if NULL -> returns -1).
 *
 * Role:
 *   Maps a human-readable `.bench` DM/TTM identifier to the dense `dm_slot` index accepted by
 *   cacaosim_bench_dm_cmd_ptr().
 *
 * Rationale:
 *   Allows C/C++ test harnesses and closed-loop benchmarks to resolve named DMs once at setup and
 *   address command buffers in O(1) inside the real-time loop regardless of topological sort order.
 *
 * Return: DM slot index in `[0 .. ctx->n_dms - 1]`, or -1 if not found.
 */
int cacaosim_bench_find_dm(
    const cacaosim_bench_ctx_t *ctx,
    const char                 *id);

/**
 * cacaosim_bench_find_sensor() - Look up sequential sensor slot [0..n_sensors-1] by section @id
 * @ctx: Analyzed or initialized bench context (safe if NULL -> returns -1).
 * @id:  Node section identifier string, e.g. "wfs0" or "sci" (safe if NULL -> returns -1).
 *
 * Role:
 *   Maps a human-readable `.bench` WFS or FPCAM identifier to the dense `sensor_slot` index
 *   accepted by cacaosim_bench_sensor_f32() and cacaosim_bench_sensor_u16().
 *
 * Rationale:
 *   Decouples caller code from topological node reordering when inspecting camera readouts in
 *   standalone C simulations and unit tests.
 *
 * Return: Sensor slot index in `[0 .. ctx->n_sensors - 1]`, or -1 if not found.
 */
int cacaosim_bench_find_sensor(
    const cacaosim_bench_ctx_t *ctx,
    const char                 *id);

/**
 * cacaosim_bench_dm_cmd_ptr() - Obtain writable actuator command array for DM slot @dm_slot
 * @ctx:     Initialized bench context (safe if NULL -> returns NULL).
 * @dm_slot: DM slot index in `[0 .. ctx->n_dms - 1]` from cacaosim_bench_find_dm().
 *
 * Role:
 *   Returns a direct pointer to the `[nact_x * nact_y]` float32 actuator command buffer—either
 *   the live SHM `img_cmd.im->array.F` when `connect_shm == 1` or the internal host `cmd_buf`
 *   when `connect_shm == 0`.
 *
 * Rationale:
 *   Unifies actuator command injection across SHM-backed and pure-C standalone execution modes
 *   with zero extra buffer copies.
 *
 * Return: Writable float32 actuator command pointer, or NULL if @ctx or @dm_slot is invalid.
 */
float *cacaosim_bench_dm_cmd_ptr(
    cacaosim_bench_ctx_t *ctx,
    int                   dm_slot);

/**
 * cacaosim_bench_sensor_f32() - Obtain read-only pointer to latest float32 frame for @sensor_slot
 * @ctx:         Initialized bench context (safe if NULL -> returns NULL).
 * @sensor_slot: Sensor slot index in `[0 .. ctx->n_sensors - 1]` from cacaosim_bench_find_sensor().
 *
 * Role:
 *   Returns the `[ncam * ncam]` float32 detector buffer for a sensor configured with
 *   `out_dtype == CACAOSIM_DTYPE_F32` (backed by SHM when `connect_shm == 1` or `cam_f32` when 0).
 *
 * Rationale:
 *   Provides zero-copy inspection of floating-point WFS / PSF frames in unit tests and embedded
 *   controllers without requiring ImageStreamIO when `connect_shm == 0`.
 *
 * Return: Read-only float32 frame pointer, or NULL if invalid slot or unallocated in U16 SHM mode.
 */
const float *cacaosim_bench_sensor_f32(
    const cacaosim_bench_ctx_t *ctx,
    int                         sensor_slot);

/**
 * cacaosim_bench_sensor_u16() - Obtain read-only pointer to latest uint16 frame for @sensor_slot
 * @ctx:         Initialized bench context (safe if NULL -> returns NULL).
 * @sensor_slot: Sensor slot index in `[0 .. ctx->n_sensors - 1]` from cacaosim_bench_find_sensor().
 *
 * Role:
 *   Returns the `[ncam * ncam]` uint16 ADU detector buffer for a sensor configured with
 *   `out_dtype == CACAOSIM_DTYPE_U16` (backed by SHM when `connect_shm == 1` or `cam_u16` when 0).
 *
 * Rationale:
 *   Provides direct zero-copy access to realistic quantized camera frames in standalone C loops
 *   and test suites.
 *
 * Return: Read-only uint16 frame pointer, or NULL if invalid slot or unallocated in F32 SHM mode.
 */
const uint16_t *cacaosim_bench_sensor_u16(
    const cacaosim_bench_ctx_t *ctx,
    int                         sensor_slot);

/**
 * struct cacaosim_plan_opts_t - Deployment-plan compiler options for `fpspipe` orchestration
 * @modular_mode:     `0` for fused zero-hop `cacaosim.htsim`, `1` for modular multi-process FPS.
 * @flat_mode:        `1` for zero-config single-stage flat mode, `0` for declarative `.bench` DAG.
 * @with_atmturb:     Non-zero to include `csim_atmturb` (`atmturb-mkwfs`) when available in `PATH`.
 * @seeing_arcsec:    Zenith seeing FWHM `[arcsec]` at `ref_lambda` when `@with_atmturb != 0`.
 * @atm_master_size:  Master turbulence screen size `[pixels]` (`0` queries `ATMTURB_MASTER_SIZE`).
 * @rate_override:    Master hardware clock override `[Hz]` when `> 0.1f`, else `[bench]` rate.
 * @backend:          Backend override (`-1` inherits `[bench]` backend, `0..2` overrides).
 * @flat_nwf:         Flat-mode pupil grid size `[pixels]` (default: `128`).
 * @flat_nact:        Flat-mode linear DM actuator count (default: `20`).
 * @flat_wfs_ncam:    Flat-mode WFS camera dimension `[pixels]` (default: `64`).
 * @flat_fp_ncam:     Flat-mode science camera dimension `[pixels]` (default: `128`).
 * @flat_mod_rad:     Flat-mode Pyramid modulation radius `[lambda/D]` (default: `3.0`).
 * @staged_bench:     Optional output path for the SHM-rewritten `.bench` snapshot.
 * @active_bench_arg: Value assigned to `csim_htsim.bench_file` in fused `.bench` mode.
 *
 * Role:
 *   Parameterizes `cacaosim_bench_emit_plan()` so both `.bench` multi-stage DAGs and zero-config
 *   flat single-stage benches compile through a single validated C implementation.
 */
typedef struct
{
    int32_t  modular_mode;
    int32_t  flat_mode;
    int32_t  with_atmturb;
    float    seeing_arcsec;
    uint32_t atm_master_size;
    float    rate_override;
    uint32_t oversample_override;
    int32_t  backend;
    uint32_t flat_nwf;
    uint32_t flat_nact;
    uint32_t flat_wfs_ncam;
    uint32_t flat_fp_ncam;
    float    flat_mod_rad;
    char     staged_bench[1024];
    char     active_bench_arg[CACAOSIM_BENCH_NAME_LEN];
} cacaosim_plan_opts_t;

/**
 * cacaosim_plan_opts_defaults() - Initialize `cacaosim_plan_opts_t` with standard CLI defaults
 * @opts: Caller-owned plan options struct (safe no-op if `NULL`).
 *
 * Role:
 *   Sets fused `.bench` mode (`modular_mode = 0`, `flat_mode = 0`), `backend = -1` (inherit),
 *   `seeing_arcsec = 0.7f`, `atm_master_size = 0` (env/1024), standard flat single-stage sizes,
 *   and `active_bench_arg = "csim_active.bench"`.
 */
void cacaosim_plan_opts_defaults(cacaosim_plan_opts_t *opts);

/**
 * cacaosim_bench_emit_plan() - Validate a bench and emit declarative `fpspipe` deployment files
 * @bench_file: Path to input `.bench` file (ignored when `opts->flat_mode != 0`).
 * @plan_dir:   Destination directory created/populated with `meta.tsv`, `loadfits.tsv`,
 *              `units.manifest`, `params.batch`, `waves.tsv`, `units.list`, and `streams.list`.
 * @opts:       Optional plan compiler options (`NULL` uses `cacaosim_plan_opts_defaults()`).
 * @err_buf:    Optional caller-owned diagnostic buffer populated on failure.
 * @err_cap:    Capacity of `@err_buf` in bytes.
 *
 * Role:
 *   Compiles a `.bench` DAG (via `cacaosim_bench_load_file()` + `cacaosim_bench_analyze()`) or
 *   flat single-stage setup into batched `milk-fps-init`, `milk-fps-set -b`, topological-wave
 *   `milk-fps-runstart`, and `milk-stream-wait` specifications consumed by `cacaosim.bash`.
 *   Resolves any FITS / `.fits.gz` / `.fits.fz` mask paths into `loadfits.tsv` and rewrites
 *   long filesystem paths into compact SHM stream stems inside `opts->staged_bench`.
 *
 * Return: `0` on success, `-1` on syntax, DAG topology, missing mask file, or I/O error.
 */
int cacaosim_bench_emit_plan(
    const char                 *bench_file,
    const char                 *plan_dir,
    const cacaosim_plan_opts_t *opts,
    char                       *err_buf,
    size_t                      err_cap);

/**
 * cacaosim_bench_bind_export_beams() - Bind/create SHM streams for selective intermediate export
 * @ctx:       Initialized bench context (`connect_shm` must be 1 for SHM creation).
 * @csv_beams: Comma-separated beam names (or `"*"` / `"all"`) to export to SHM each tick.
 * @err_buf:   Optional caller-owned diagnostic buffer populated on unknown beam / SHM error.
 * @err_cap:   Capacity of `@err_buf` in bytes.
 *
 * Return: `0` on success, `-1` if `@ctx` is NULL, an unknown beam is named, or SHM creation fails.
 */
int cacaosim_bench_bind_export_beams(
    cacaosim_bench_ctx_t *ctx,
    const char           *csv_beams,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_bench_find_beam() - Resolve a named optical beam to its slot index in ctx->beams[]
 * @ctx:  Analyzed or initialized bench context (may be NULL).
 * @name: Beam identifier (e.g. "b_pup", "b_post_stg2").
 *
 * Return: Beam slot index in `[0, ctx->n_beams - 1]`, or `-1` if not found / NULL args.
 */
int cacaosim_bench_find_beam(const cacaosim_bench_ctx_t *ctx, const char *name);

/**
 * cacaosim_bench_find_dm_stream() - Resolve a DM stream name or node id to its bound SHM IMAGE
 * @ctx:  Initialized bench context with `connect_shm == 1` (may be NULL).
 * @spec: DM node id or `in_cmd` stream name (e.g. `"dm01disp"`; `"none"` / empty returns NULL).
 *
 * Return: Pointer to the bound `IMAGE` handle, or `NULL` if unavailable / not found.
 */
IMAGE *cacaosim_bench_find_dm_stream(const cacaosim_bench_ctx_t *ctx, const char *spec);

/**
 * cacaosim_bench_reset_wfs_accum() - Reset subframe accumulators across all sensor nodes
 * @ctx: Initialized bench context (no-op if NULL).
 */
void cacaosim_bench_reset_wfs_accum(cacaosim_bench_ctx_t *ctx);

/**
 * cacaosim_bench_reset_fpcam_accum() - Reset subframe accumulators across all FPCAM nodes
 * @ctx: Initialized bench context (no-op if NULL).
 */
void cacaosim_bench_reset_fpcam_accum(cacaosim_bench_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BENCH_H */
