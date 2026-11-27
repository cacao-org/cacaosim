// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_htsim_FPS.c
 * @brief   Dual-mode fused zero-hop High-Throughput AO bench FPS (`cacaosim.htsim`)
 *
 * Architectural Role in `cacaosim`:
 *   Implements the `csim_htsim` (`cacaosim.htsim` / `milk-fpsexec-cacaosim-htsim`) real-time
 *   Function Parameter Structure (FPS) runnable. It operates in two complementary modes selected
 *   by the `.bench_file` parameter:
 *     1. **Declarative Multi-Stage `.bench` DAG Mode (`.bench_file` non-empty)**:
 *        Loads an arbitrary multi-stage optical bench description (woofer/tweeter DMs, Tip-Tilt
 *        mirrors, static/SHM WFE screens, gray/dichroic beamsplitters, coronagraphs with LOWFS
 *        pickoffs, Pyramid/SH/Zernike WFSs, and polychromatic focal-plane cameras), compiles and
 *        validates its multi-rate DAG via `cacaosim_bench_init()`, prints an ANSI schematic (on
 *        interactive terminals) or compact topology summary (in log files), and steps the entire
 *        bench in-process via `cacaosim_bench_step()`.
 *     2. **Flat Single-Stage SCAO Fast Path (`.bench_file` empty)**:
 *        Directly couples a single Deformable Mirror (`cacaosim_dm_ctx_t`), 4-sided Pyramid WFS
 *        (`cacaosim_pyrwfs_ctx_t`), and optional Soummer MFT focal-plane science camera
 *        (`cacaosim_fpcam_ctx_t`) configured purely from flat FPS parameters—requiring zero
 *        external configuration files for standard `cacao` SCAO closed-loop setups.
 *
 * Design Rationale & Motivation:
 *   - **Zero-Hop In-Memory & On-GPU Wavefront Passing**: Running modular `cacaosim` FPS processes
 *     (`mkbeam`, `dmsim`, `pyrwfs`, `fpcam`) separately requires writing and reading intermediate
 *     `[OPD, Amplitude]` streams through POSIX SHM, waking OS threads via semaphores, and copying
 *     wavefront grids across PCIe on every multi-kHz tick. Fusing the optical bench inside one
 *     process on a single shared `cudaStream_t` keeps DM surfaces (`dm_ctx.d_opd_out`) and complex
 *     beams resident in GPU L2/VRAM, transferring only small `[nact x nact]` DM commands in and
 *     readout-rate `[ncam x ncam]` detector pixels out—while executing the exact same physical
 *     kernels as the standalone modular FPS units.
 *   - **Drift-Free Monotonic Hardware Clocking**: Real-time AO controllers (`cacao`) rely on
 *     deterministic camera frame arrival times. `htsim` paces its master tick with absolute
 *     `CLOCK_MONOTONIC` deadlines (`cacaosim_timespec_add_ns` + `cacaosim_sleep_until` using
 *     `TIMER_ABSTIME`) so variable per-frame compute time never accumulates into clock drift,
 *     while `.frame_rate = 0` falls back cleanly to `PROCESSINFO` stream-triggered stepping for
 *     deterministic offline replay and latency benchmarking.
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "bench/cacaosim_bench.h"
#include "bench/cacaosim_bench_view.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_time.h"
#include "common/cacaosim_wfstat_core.h"
#include "dm/cacaosim_dm.h"
#include "fpcam/cacaosim_fpcam.h"
#include "fused/cacaosim_htsim_FPS.h"
#include "pupil/cacaosim_pupil.h"
#include "pyrwfs/cacaosim_pyrwfs.h"

/* ---------------------------------------------------------------------------------------------- */
/* Static FPS Parameter Backing Store                                                             */
/* ---------------------------------------------------------------------------------------------- */

/* Mode selector & SHM stream names (used in flat SCAO mode; `.bench_file` selects DAG mode) */
static char     param_bench_file[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char     param_export_beams[FUNCTION_PARAMETER_STRMAXLEN] = "";
static int64_t  param_export_enable                              = 1;
static int64_t  param_wfstat_enable                              = 0;
static char     param_wfstat_beams[FUNCTION_PARAMETER_STRMAXLEN] = "b_pup,b_post_stg1,b_post_stg2";
static char     param_wfstat_outdir[FUNCTION_PARAMETER_STRMAXLEN] = ".";
static uint64_t param_wfstat_ticks                                = 20000;
static uint32_t param_wfstat_stride                               = 0;
static int64_t  param_wfstat_extra                                = 1;
static char     param_wfstat_cm1[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static char     param_wfstat_cm2[FUNCTION_PARAMETER_STRMAXLEN]    = "";
static uint64_t param_wfstat_clip_start                           = 0;
static uint32_t param_wfstat_clip_len                             = 0;
static float    param_wfstat_full_fps                             = 0.0f;
static uint64_t param_wfstat_frames                               = 0;
static int64_t  param_calib_fast                                  = 0;
static int64_t  param_calib_temporal                              = 0;
static int64_t  param_calib_ring                                  = 0;
static float    param_calib_poke_offset                           = 0.0f;
static char     param_calib_trig[FUNCTION_PARAMETER_STRMAXLEN]    = "none";
static char     param_calib_wfs[FUNCTION_PARAMETER_STRMAXLEN]     = "none";
static char     param_in_dmcmd[FUNCTION_PARAMETER_STRMAXLEN]      = "aol0_dmC";
static char     param_out_wfscam[FUNCTION_PARAMETER_STRMAXLEN] = "aol0_wfsim";
static char     param_out_dmopd[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static int64_t  param_enable_fpcam                             = 1;
static char     param_out_fpcam[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_fpcam";
static char     param_out_fplong[FUNCTION_PARAMETER_STRMAXLEN] = "csim_fplong";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]      = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]     = "";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]     = "csim_pup";

/* Flat-mode pupil geometry & DM hardware actuator/dynamics configuration */
static uint32_t param_nwf                                      = 128;
static float    param_pup_diam                                 = 0.9375f;
static uint32_t param_nact_x                                   = 20;
static uint32_t param_nact_y                                   = 20;
static float    param_if_coupling                              = 0.20f;
static float    param_delay_frames                             = 1.0f;
static int32_t  param_dyn_mode                                 = 1;
static float    param_tau_us                                   = 120.0f;
static float    param_f0_hz                                    = 2500.0f;
static float    param_zeta                                     = 0.70f;
static float    param_max_stroke                               = 5.0f;
static int32_t  param_dac_bits                                 = 16;

/* Flat-mode 4-sided Pyramid WFS optical & detector readout configuration */
static uint32_t param_wfs_ncam                                 = 64;
static float    param_wfs_lambda                               = 0.75f;
static float    param_mod_rad                                  = 3.0f;
static int32_t  param_mod_npts                                 = 16;
static float    param_wfs_flux                                 = 100000.0f;
static float    param_wfs_ron                                  = 0.5f;
static int64_t  param_wfs_shutter                              = 1;

/* Flat-mode Soummer MFT focal-plane science PSF camera & subframe integration controls */
static uint32_t param_fp_ncam                                  = 128;
static float    param_fp_lambda                                = 1.65f;
static float    param_fp_pixscale                              = 3.0f;
static uint32_t param_fp_navg                                  = 10;
static float    param_fp_flux                                  = 1000000.0f;
static float    param_fp_ron                                   = 1.0f;
static int64_t  param_reset_long                               = 0;

/* Shared noise, atmospheric scaling, monotonic master clock, PRNG, backend, & Strehl telemetry */
static int64_t  param_photon_noise                             = 1;
static int32_t  param_out_dtype                                = 0;
static int32_t  param_wf_unit                                  = 1;
static float    param_ref_lambda                               = 0.50f;
static float    param_turb_gain                                = 1.0f;
static int64_t  param_vib_enable                               = 0;
static float    param_vib_rms_nm                               = 45.0f;
static int64_t  param_petal_enable                             = 0;
static float    param_petal_rms_nm                             = 1000.0f;
static float    param_petal_tau_s                              = 3.0f;
static float    param_petal_off0_nm                            = 0.0f;
static float    param_petal_off1_nm                            = 0.0f;
static float    param_petal_off2_nm                            = 0.0f;
static float    param_petal_off3_nm                            = 0.0f;
static int64_t  param_atm_disp_enable                          = 1;
static float    param_zenith_deg                               = 0.0f;
static float    param_disp_pa_deg                              = 0.0f;
static int32_t  param_adc_mode                                 = -1;
static int32_t  param_adc_opt_band                             = -1;
static float    param_adc_lut_scale_err                        = 0.0f;
static float    param_adc_lut_pa_err_deg                       = 0.0f;
static float    param_frame_rate                               = 2000.0f;
static uint64_t param_rng_seed                                 = 98765;
static int32_t  param_backend                                  = 0;
static float    param_strehl_inst                              = 0.0f;
static float    param_strehl_ave                               = 0.0f;

/**
 * CACAOSIM_HTSIM_PARAMS - X-macro FPS parameter schema for `cacaosim.htsim`
 * @X: Visitor macro invoked as `X(key, ptr, type, required, flags, description)`
 *
 * Exposes both the `.bench_file` multi-stage DAG selector and the complete set of flat SCAO
 * parameters. Parameters marked with `FPFLAG_WRITERUN` remain live-editable via `milk-fpsCTRL` or
 * `milk-fps-set` while the real-time compute loop is running (e.g. setting `.turb_gain = 0` and
 * toggling `.wfs_shutter` during `cacao` dark/response-matrix acquisition without restarting the
 * hardware simulator).
 */
#define CACAOSIM_HTSIM_PARAMS(X)                                                        \
    X(".bench_file", param_bench_file, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional declarative multi-stage .bench DAG file (empty = flat single-stage)")   \
    X(".export_beams", param_export_beams, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,      \
      "CSV optical beams to export to SHM (empty=none, *=all)")                         \
    X(".export_enable", &param_export_enable, FPTYPE_ONOFF, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live gate for SHM export of .export_beams (ON/OFF)")                             \
    X(".wfstat_enable", &param_wfstat_enable, FPTYPE_ONOFF, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "In-DAG wavefront stats: ON starts, OFF (or .wfstat_ticks) writes outputs")       \
    X(".wfstat_beams", param_wfstat_beams, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,      \
      "CSV raw,stg1,final beam names for in-DAG wavefront stats")                       \
    X(".wfstat_outdir", param_wfstat_outdir, FPTYPE_STRING, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Output directory for wf_stats.json and FITS products (<64 chars)")               \
    X(".wfstat_ticks", &param_wfstat_ticks, FPTYPE_UINT64, 0,                           \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Frames to accumulate before auto-writing (0 = until disabled)")                  \
    X(".wfstat_stride", &param_wfstat_stride, FPTYPE_UINT32, 0,                         \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Master-tick stride between wfstat samples (0 = auto oversample)")                \
    X(".wfstat_extra", &param_wfstat_extra, FPTYPE_ONOFF, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Snapshot WFS/DM/PSF diagnostic streams at the 5 sample instants")                \
    X(".wfstat_cm1", param_wfstat_cm1, FPTYPE_STRING, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Optional Stage-1 CMmodesDM SHM stream for within/outside control WFE split")     \
    X(".wfstat_cm2", param_wfstat_cm2, FPTYPE_STRING, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Optional Stage-2 CMmodesDM SHM stream for within/outside control WFE split")     \
    X(".wfstat_clip_start", &param_wfstat_clip_start, FPTYPE_UINT64, 0,                 \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "1-based outer frame start for short-timescale WF/DM video clip (0 = auto mid)")  \
    X(".wfstat_clip_len", &param_wfstat_clip_len, FPTYPE_UINT32, 0,                     \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Number of contiguous outer frames to capture for WF/DM video clip (0 = off)")    \
    X(".wfstat_full_fps", &param_wfstat_full_fps, FPTYPE_FLOAT32, 0,                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Strided full-duration WF/DM capture rate [Hz] for 1x real-time video (0 = off)") \
    X(".wfstat_frames", &param_wfstat_frames, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_OUTPUT,  \
      "Frames accumulated by the current/last in-DAG wavefront stats run")              \
    X(".calib_fast", &param_calib_fast, FPTYPE_ONOFF, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Lockstep calibration: skip fpcam/DM dynamics & force WFS navg=1")                \
    X(".calib_temporal", &param_calib_temporal, FPTYPE_ONOFF, 0,                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Step real DM dynamics & oversampled WFS subframes during .calib_fast")           \
    X(".calib_ring", &param_calib_ring, FPTYPE_ONOFF, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Auto-kick closed WFS->MVM->mfilt->DM ring on 5ms timeout in .calib_fast")        \
    X(".calib_poke_offset", &param_calib_poke_offset, FPTYPE_FLOAT32, 0,                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "DM poke arrival offset [us] relative to WFS frame start in .calib_temporal")     \
    X(".calib_trig", param_calib_trig, FPTYPE_STREAMNAME, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "DM stream triggering step in .calib_fast mode (\"none\" = disabled)")           \
    X(".calib_wfs", param_calib_wfs, FPTYPE_STREAMNAME, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Target WFS cam/id during .calib_fast (\"none\" = all WFS)")                      \
    X(".in_dmcmd", param_in_dmcmd, FPTYPE_STREAMNAME, 1,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Input 2D DM actuator command stream (auto-created if missing)")                  \
    X(".out_wfscam", param_out_wfscam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,      \
      "Output 2D Pyramid WFS camera stream (e.g. aol0_wfsim)")                          \
    X(".out_dmopd", param_out_dmopd, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Optional output 2D DM OPD stream (empty = zero-copy GPU internal)")              \
    X(".enable_fpcam", &param_enable_fpcam, FPTYPE_ONOFF, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Enable in-process focal plane science PSF camera")                               \
    X(".out_fpcam", param_out_fpcam, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D focal plane science camera stream")                                    \
    X(".out_fplong", param_out_fplong, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,          \
      "Optional output float32 cumulative long-exposure PSF stream")                    \
    X(".in_wf", param_in_wf, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Optional atmospheric WF stream (e.g. outarraypha from milkatmturb)")             \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional atmospheric amplitude stream (e.g. outarrayamp)")                       \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Clear pupil diameter fraction of nwf (default: 0.9375)")                         \
    X(".nact_x", &param_nact_x, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "DM actuator grid X size (default: 20)")                                          \
    X(".nact_y", &param_nact_y, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                 \
      "DM actuator grid Y size (default: 20)")                                          \
    X(".if_coupling", &param_if_coupling, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "DM inter-actuator mechanical coupling (default: 0.20)")                          \
    X(".delay_frames", &param_delay_frames, FPTYPE_FLOAT32, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Pure DM hardware latency [frames] (default: 1.0)")                               \
    X(".dyn_mode", &param_dyn_mode, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM dynamics: 0=instant, 1=1st-order RC, 2=2nd-order resonance")                  \
    X(".tau_us", &param_tau_us, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "DM 1st-order RC time constant [us] (default: 120.0)")                            \
    X(".f0_hz", &param_f0_hz, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                  \
      "DM 2nd-order natural resonance [Hz] (default: 2500.0)")                          \
    X(".zeta", &param_zeta, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "DM 2nd-order damping ratio (default: 0.70)")                                     \
    X(".max_stroke", &param_max_stroke, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "DM mechanical stroke clip [um] (default: 5.0)")                                  \
    X(".dac_bits", &param_dac_bits, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "DM DAC bit depth (0 = floating point, default: 16)")                             \
    X(".wfs_ncam", &param_wfs_ncam, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,             \
      "WFS camera size [pixels] (default: 64)")                                         \
    X(".wfs_lambda", &param_wfs_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "WFS wavelength [um] (default: 0.75)")                                            \
    X(".mod_rad", &param_mod_rad, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "PyWFS modulation radius [lambda/D] (default: 3.0)")                              \
    X(".mod_npts", &param_mod_npts, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,              \
      "PyWFS modulation points (default: 16)")                                          \
    X(".wfs_flux", &param_wfs_flux, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Guide-star flux [photons / WFS frame] (default: 100000)")                        \
    X(".wfs_ron", &param_wfs_ron, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "WFS camera readout noise [e- RMS] (default: 0.5)")                               \
    X(".wfs_shutter", &param_wfs_shutter, FPTYPE_ONOFF, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "WFS optical shutter (ON = open, OFF = dark frame)")                              \
    X(".fp_ncam", &param_fp_ncam, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,               \
      "Focal plane camera size [pixels] (default: 128)")                                \
    X(".fp_lambda", &param_fp_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,          \
      "Focal plane wavelength [um] (default: 1.65)")                                    \
    X(".fp_pixscale", &param_fp_pixscale, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,      \
      "Focal plane pixel scale [pixels per lambda/D] (default: 3.0)")                   \
    X(".fp_navg", &param_fp_navg, FPTYPE_UINT32, 0,                                     \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "WFS frames integrated per focal plane readout (default: 10)")                    \
    X(".fp_flux", &param_fp_flux, FPTYPE_FLOAT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Science starlight flux [photons / FP exposure] (default: 1e6)")                  \
    X(".fp_ron", &param_fp_ron, FPTYPE_FLOAT32, 0,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Focal plane readout noise [e- RMS] (default: 1.0)")                              \
    X(".reset_long", &param_reset_long, FPTYPE_ONOFF, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Toggle ON to clear cumulative long-exposure PSF")                                \
    X(".photon_noise", &param_photon_noise, FPTYPE_ONOFF, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Enable Poisson photon shot noise on detectors")                                  \
    X(".out_dtype", &param_out_dtype, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Detector output datatype: 0 = uint16, 1 = float32")                              \
    X(".wf_unit", &param_wf_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1)")                   \
    X(".ref_lambda", &param_ref_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reference wavelength [um] for phase input (default: 0.50)")                      \
    X(".turb_gain", &param_turb_gain, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live turbulence multiplier (0 = off for calibration, 1 = nominal)")              \
    X(".vib_enable", &param_vib_enable, FPTYPE_ONOFF, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Enable stochastic telescope/bench vibrations (14.5..138 Hz)")                    \
    X(".vib_rms_nm", &param_vib_rms_nm, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Total telescope/bench vibration OPD RMS [nm] (default: 45.0)")                   \
    X(".petal_enable", &param_petal_enable, FPTYPE_ONOFF, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Enable thermally induced slow-varying pupil petaling (LWE)")                     \
    X(".petal_rms_nm", &param_petal_rms_nm, FPTYPE_FLOAT32, 0,                          \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Total thermal petaling (LWE) OPD RMS [nm] (default: 1000.0)")                    \
    X(".petal_tau_s", &param_petal_tau_s, FPTYPE_FLOAT32, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Thermal petaling correlation timescale [s] (default: 3.0)")                      \
    X(".petal_off0_nm", &param_petal_off0_nm, FPTYPE_FLOAT32, 0,                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live additive LWE piston step offset on island 0 / East [nm OPD]")               \
    X(".petal_off1_nm", &param_petal_off1_nm, FPTYPE_FLOAT32, 0,                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live additive LWE piston step offset on island 1 / North [nm OPD]")              \
    X(".petal_off2_nm", &param_petal_off2_nm, FPTYPE_FLOAT32, 0,                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live additive LWE piston step offset on island 2 / West [nm OPD]")               \
    X(".petal_off3_nm", &param_petal_off3_nm, FPTYPE_FLOAT32, 0,                        \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live additive LWE piston step offset on island 3 / South [nm OPD]")              \
    X(".atm_disp_enable", &param_atm_disp_enable, FPTYPE_ONOFF, 0,                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live gate for atmospheric dispersion (ON/OFF)")                                  \
    X(".zenith_deg", &param_zenith_deg, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live telescope zenith angle [deg] for atmospheric dispersion & ADC LUT")         \
    X(".disp_pa_deg", &param_disp_pa_deg, FPTYPE_FLOAT32, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live zenith dispersion azimuth on pupil [deg]")                                  \
    X(".adc_mode", &param_adc_mode, FPTYPE_INT32, 0,                                    \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live ADC mode override (-1=bench default, 0=retract, 1=man, 2=lut, 3=shm, 4=l+s)") \
    X(".adc_opt_band", &param_adc_opt_band, FPTYPE_INT32, 0,                            \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live ADC LUT opt_band override (-1=bench default, 0=whole, 1=H, 2=J, ..., 7=cust)") \
    X(".adc_lut_scale_err", &param_adc_lut_scale_err, FPTYPE_FLOAT32, 0,                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live additive LUT dispersion gain error applied to ADC nodes")                   \
    X(".adc_lut_pa_err_deg", &param_adc_lut_pa_err_deg, FPTYPE_FLOAT32, 0,              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live additive LUT PA error [deg] applied to ADC nodes")                          \
    X(".frame_rate", &param_frame_rate, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Wall-clock pacing [Hz] (0 = procinfo trigger; .bench master_hz sets physics)")    \
    X(".rng_seed", &param_rng_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,             \
      "Detector PRNG seed")                                                             \
    X(".backend", &param_backend, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Compute backend: 0=auto, 1=CPU, 2=CUDA")                                         \
    X(".strehl_inst", &param_strehl_inst, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,     \
      "Instantaneous focal plane peak Strehl ratio [0..1]")                             \
    X(".strehl_ave", &param_strehl_ave, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_OUTPUT,       \
      "Cumulative long-exposure peak Strehl ratio [0..1]")

/* Generate `milk_runnable.h` FPS parameter binding descriptors for `cacaosim_htsim` */
MILK_PARAMS_DEFINE(cacaosim_htsim, CACAOSIM_HTSIM_PARAMS);

/**
 * htsim_confcheck() - FPS configuration validation & live-writable flag enforcement callback
 *
 * Role:
 *   Invoked during the `milk` FPS `CONF` stage (`confstart` / `confcheck`). Ensures all 14
 *   real-time operational knobs carry `FPFLAG_WRITERUN` inside the shared-memory `dcfpsptr`
 *   table, even when reattaching to a pre-existing FPS SHM segment created by an older session.
 *
 * Rationale:
 *   `cacao` calibration and loop-closing workflows routinely toggle WFS shutters, zero atmospheric
 *   turbulence (`turb_gain = 0`), adjust PyWFS modulation (`mod_rad`), or reset cumulative Strehl
 *   integrations (`reset_long = 1`) while `RUN` is active. Asserting `FPFLAG_WRITERUN` in
 *   `confcheck` guarantees live mutability across FPS restarts without deleting `.fps.shm` files.
 *
 * Return: Always `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t htsim_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".export_enable", ".delay_frames", ".tau_us",      ".max_stroke", ".mod_rad",
            ".wfs_flux",      ".wfs_ron",      ".wfs_shutter", ".fp_navg",    ".fp_flux",
            ".fp_ron",        ".reset_long",   ".photon_noise", ".turb_gain", ".frame_rate",
            ".wfstat_enable", ".wfstat_outdir", ".wfstat_ticks", ".wfstat_stride",
            ".wfstat_extra",  ".wfstat_cm1",    ".wfstat_cm2",   ".wfstat_clip_start",
            ".wfstat_clip_len", ".wfstat_full_fps", ".calib_fast",
            ".calib_temporal", ".calib_poke_offset",
            ".calib_trig",    ".calib_wfs",     ".vib_enable",     ".vib_rms_nm",
            ".petal_enable",  ".petal_rms_nm",  ".petal_tau_s",
            ".petal_off0_nm", ".petal_off1_nm", ".petal_off2_nm", ".petal_off3_nm",
            ".atm_disp_enable", ".zenith_deg",  ".disp_pa_deg",    ".adc_mode",
            ".adc_opt_band",  ".adc_lut_scale_err", ".adc_lut_pa_err_deg"
        };
        for (size_t i = 0; i < sizeof(live_keys) / sizeof(live_keys[0]); i++)
        {
            long idx = functionparameter_GetParamIndex(dcfpsptr, live_keys[i]);
            if (idx >= 0)
            {
                dcfpsptr->parray[idx].fpflag |= FPFLAG_WRITERUN;
            }
        }
    }
    return RETURN_SUCCESS;
}

/**
 * resolve_bench_path() - Resolve a `.bench_file` specification into a readable filesystem path
 * @spec: User-supplied `.bench_file` string (e.g. `"@env"`, `"scao_pyrwfs.bench"`, or full path)
 * @out:  Caller-allocated output buffer receiving the resolved path
 * @cap:  Byte capacity of `@out` (including terminating NUL)
 *
 * Role:
 *   Implements deterministic 4-tier search precedence for `.bench` configuration files:
 *     1. `"@env"` sentinel -> `$CACAOSIM_BENCH_FILE` environment variable (if set and non-empty).
 *     2. Direct readable path (`access(spec, R_OK) == 0`) relative to CWD or absolute.
 *     3. Session shared-memory directory (`$MILK_SHM_DIR/<spec>`) for transient benches saved by
 *        `cacaosim-CTRL` or `cacaosim-bench`.
 *     4. Installed examples directory (`$MILK_ROOT/share/cacaosim/examples/<spec>`).
 *     5. Verbatim fallback (`spec`) so `cacaosim_bench_load_file()` reports the exact missing name.
 *
 * Rationale:
 *   FPS string parameters have bounded length (`FUNCTION_PARAMETER_STRMAXLEN`, typically 64 bytes),
 *   which can be exceeded by deep workspace paths. Supporting `"@env"` plus `$MILK_SHM_DIR` and
 *   `$MILK_ROOT/share/cacaosim/examples` short names lets `cacao` launch scripts and TUI tools
 *   reference benches concisely without path truncation.
 */
static void resolve_bench_path(const char *spec, char *out, size_t cap)
{
    if (strcmp(spec, "@env") == 0)
    {
        const char *ev = getenv("CACAOSIM_BENCH_FILE");
        if (ev != NULL && ev[0] != '\0')
        {
            snprintf(out, cap, "%s", ev);
            return;
        }
    }
    if (access(spec, R_OK) == 0)
    {
        snprintf(out, cap, "%s", spec);
        return;
    }
    if (spec[0] != '/')
    {
        const char *shm_dir = getenv("MILK_SHM_DIR");
        const char *sdir    = (shm_dir != NULL && shm_dir[0] != '\0') ? shm_dir : "/milk/shm";
        snprintf(out, cap, "%s/%s", sdir, spec);
        if (access(out, R_OK) == 0)
        {
            return;
        }
        const char *milk_root = getenv("MILK_ROOT");
        if (milk_root != NULL && milk_root[0] != '\0')
        {
            snprintf(out, cap, "%s/share/cacaosim/examples/%s", milk_root, spec);
            if (access(out, R_OK) == 0)
            {
                return;
            }
        }
    }
    snprintf(out, cap, "%s", spec);
}

/**
 * htsim_wfstat_start() - Resolve .wfstat_beams and start in-DAG wavefront statistics
 * @ws:   Destination statistics state.
 * @bctx: Initialized bench context (beams accumulated on its CUDA stream or host buffers).
 * @bidx: Output beam slot indices `[CACAOSIM_WFSTAT_NB_LOC]` (raw, stg1, final).
 *
 * Return: 0 on success, -1 on unknown beam name or allocation failure.
 */
static int htsim_wfstat_start(cacaosim_wfstat_t *ws, const cacaosim_bench_ctx_t *bctx,
                              int *bidx, double dt_phys)
{
    char  copy[FUNCTION_PARAMETER_STRMAXLEN];
    char *save = NULL;
    snprintf(copy, sizeof(copy), "%s", param_wfstat_beams);
    char *tok = strtok_r(copy, ", ", &save);
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        bidx[k] = (tok != NULL) ? cacaosim_bench_find_beam(bctx, tok) : -1;
        if (bidx[k] < 0)
        {
            fprintf(stderr, "[cacaosim.htsim] .wfstat_beams: beam %d ('%s') not found\n", k,
                    tok ? tok : "");
            return -1;
        }
        tok = strtok_r(NULL, ", ", &save);
    }
    if (cacaosim_wfstat_init(ws, bctx->nwf, bctx->pup_ptr, param_wfstat_ticks,
                             (int) param_wfstat_extra, bctx->use_cuda,
                             bctx->cuda_stream) != 0)
    {
        fprintf(stderr, "[cacaosim.htsim] in-DAG wavefront stats init failed\n");
        return -1;
    }
    if (param_wfstat_cm1[0] != '\0' || param_wfstat_cm2[0] != '\0')
    {
        const cacaosim_dm_ctx_t *dm1 =
            (bctx->n_dms >= 1) ? &bctx->nodes[bctx->dm_node_idx[0]].dm_ctx : NULL;
        const cacaosim_dm_ctx_t *dm2 =
            (bctx->n_dms >= 2) ? &bctx->nodes[bctx->dm_node_idx[1]].dm_ctx : NULL;
        if (cacaosim_wfstat_load_ctrl_modes(
                ws, param_wfstat_cm1, dm1 ? dm1->stencil_idx : NULL, dm1 ? dm1->stencil_w : NULL,
                dm1 ? dm1->nact_total : 0u, param_wfstat_cm2, dm2 ? dm2->stencil_idx : NULL,
                dm2 ? dm2->stencil_w : NULL, dm2 ? dm2->nact_total : 0u) != 0)
        {
            fprintf(stderr, "[cacaosim.htsim] warning: wfstat control-mode basis setup failed\n");
        }
    }
    if (param_wfstat_clip_len > 0u)
    {
        uint64_t cstart = param_wfstat_clip_start;
        if (cstart == 0ULL)
        {
            uint64_t tgt = (param_wfstat_ticks > 10ULL) ? param_wfstat_ticks : 20000ULL;
            uint64_t mid = tgt / 2ULL;
            uint64_t pre = 20ULL;
            if (mid + (uint64_t) param_wfstat_clip_len > tgt)
            {
                cstart = (tgt > (uint64_t) param_wfstat_clip_len)
                             ? ((tgt - (uint64_t) param_wfstat_clip_len) / 2ULL + 1ULL)
                             : 1ULL;
            }
            else
            {
                cstart = (mid > pre) ? (mid - pre) : 1ULL;
            }
        }
        if (cacaosim_wfstat_enable_clip(ws, cstart, param_wfstat_clip_len) != 0)
        {
            fprintf(stderr, "[cacaosim.htsim] warning: wfstat clip capture alloc failed\n");
        }
    }
    if (param_wfstat_full_fps > 0.0f && dt_phys > 0.0)
    {
        double step_fr = 1.0 / ((double) param_wfstat_full_fps * dt_phys);
        if (step_fr < 1.0)
        {
            step_fr = 1.0;
        }
        uint64_t tgt = (param_wfstat_ticks > 10ULL) ? param_wfstat_ticks : 20000ULL;
        uint32_t cap = (uint32_t) ((double) tgt / step_fr) + 8u;
        if (cap > 3600u)
        {
            cap = 3600u;
        }
        if (cacaosim_wfstat_enable_full(ws, step_fr, cap) != 0)
        {
            fprintf(stderr, "[cacaosim.htsim] warning: wfstat full-video capture alloc failed\n");
        }
    }
    return 0;
}

/* Read float/double SHM keyword from an optional IMAGE, returning def_val if absent */
static float htsim_read_img_kw(const IMAGE *im, const char *name, float def_val)
{
    if (im == NULL || im->kw == NULL || im->md == NULL)
    {
        return def_val;
    }
    uint32_t nb = im->md->NBkw;
    for (uint32_t i = 0; i < nb; i++)
    {
        if (strcmp(im->kw[i].name, name) == 0 && im->kw[i].type == 'D')
        {
            return (float) im->kw[i].value.numf;
        }
    }
    return def_val;
}

/* Accumulate plane 0 of the three selected beams (device or host buffers) */
static void htsim_wfstat_frame(cacaosim_wfstat_t *ws, const cacaosim_bench_ctx_t *bctx,
                               const int *bidx)
{
    const float *opd[CACAOSIM_WFSTAT_NB_LOC];
    for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
    {
        const cacaosim_beam_slot_t *b = &bctx->beams[bidx[k]];
        opd[k]                        = bctx->use_cuda ? b->d_opd : b->opd;
    }
    cacaosim_wfstat_frame(ws, opd);

    cacaosim_wfstat_perturb_sample_t ps;
    memset(&ps, 0, sizeof(ps));
    const IMAGE *wf_im = (bctx->connect_shm && bctx->img_wf.ID != -1) ? bctx->img_wf.im : NULL;
    ps.wind_scale      = htsim_read_img_kw(wf_im, "WND_SCL", 1.0f);
    ps.wind_deg        = htsim_read_img_kw(wf_im, "WND_DEG", 0.0f);
    ps.burst_env       = htsim_read_img_kw(wf_im, "WND_BST", 0.0f);
    ps.cn2_scale       = htsim_read_img_kw(wf_im, "WND_CN2", 1.0f);
    ps.vib_rms_nm      = bctx->perturb.tlm.vib_rms_nm;
    ps.vib_tip_nm      = bctx->perturb.tlm.vib_tip_nm;
    ps.vib_tilt_nm     = bctx->perturb.tlm.vib_tilt_nm;
    ps.vib_foc_nm      = bctx->perturb.tlm.vib_foc_nm;
    ps.petal_rms_nm    = bctx->perturb.tlm.petal_rms_nm;
    ps.petal_ptv_nm    = bctx->perturb.tlm.petal_ptv_nm;
    for (int p = 0; p < 4; p++)
    {
        ps.petal_p_nm[p] = bctx->perturb.tlm.petal_p_nm[p];
    }
    cacaosim_wfstat_record_perturb(ws, &ps);
}

/**
 * htsim_run_bench() - Execute `cacaosim.htsim` in declarative multi-stage `.bench` DAG mode
 * @self: Pointer to the owning `MILK_RUNNABLE` descriptor for `PROCESSINFO` lifecycle macros
 *
 * Role:
 *   Loads and compiles the `.bench` file named by `param_bench_file`, reconciles top-level FPS
 *   overrides (`.backend`, `.frame_rate`, `.turb_gain`) with `[bench]` file settings, prints a
 *   human-readable optical topology diagram/summary, runs the monotonic real-time stepping loop
 *   via `cacaosim_bench_step()`, and tears down all GPU/CPU/SHM resources cleanly on loop exit.
 *
 * Rationale:
 *   Isolating `.bench` execution into a dedicated routine keeps the multi-stage DAG lifecycle
 *   (`cacaosim_bench_init` -> `MILK_PROCINFO_LOOPSTART` -> `cacaosim_bench_free`) self-contained
 *   while still honoring standard `milk` `PROCESSINFO` triggers, loop-stop signals, live FPS
 *   `turb_gain` / `frame_rate` / `reset_long` updates, and output Strehl telemetry.
 *
 * Return: `RETURN_SUCCESS` on clean loop termination, or `RETURN_FAILURE` on parse/compile error.
 */
/* Find sorted node index of DM matching spec (id or in_cmd), else -1 */
static int htsim_find_dm_node_idx(const cacaosim_bench_ctx_t *bctx, const char *spec)
{
    if (spec == NULL || spec[0] == '\0' || strcmp(spec, "none") == 0)
    {
        return -1;
    }
    for (uint32_t i = 0; i < bctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd = &bctx->nodes[i];
        if (nd->cfg.kind != CACAOSIM_NODE_DM)
        {
            continue;
        }
        if (strcmp(nd->cfg.in_cmd, spec) == 0 || strcmp(nd->cfg.id, spec) == 0)
        {
            return (int) i;
        }
    }
    return -1;
}

/* Resolve target WFS eff_navg for temporal calibration stepping */
static uint32_t htsim_calib_wfs_navg(const cacaosim_bench_ctx_t *bctx, const char *spec)
{
    uint32_t best     = 1u;
    int      has_spec = (spec != NULL && spec[0] != '\0' && strcmp(spec, "none") != 0);
    for (uint32_t i = 0; i < bctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd = &bctx->nodes[i];
        if (nd->cfg.kind < CACAOSIM_NODE_PYRWFS || nd->cfg.kind > CACAOSIM_NODE_FPCAM)
        {
            continue;
        }
        if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && !has_spec && !nd->cfg.is_wfs)
        {
            continue;
        }
        if (has_spec)
        {
            if (strcmp(nd->cfg.out_cam, spec) == 0 || strcmp(nd->cfg.id, spec) == 0)
            {
                return (nd->eff_navg >= 1u) ? nd->eff_navg : 1u;
            }
        }
        else if (nd->eff_navg > best)
        {
            best = nd->eff_navg;
        }
    }
    return best;
}

typedef struct
{
    cacaosim_wfstat_t ws;
    char              outdir[FUNCTION_PARAMETER_STRMAXLEN];
    double            dt_phys;
} htsim_wfstat_job_t;

static void *htsim_wfstat_write_thread(void *arg)
{
    htsim_wfstat_job_t *job = (htsim_wfstat_job_t *) arg;
    if (job != NULL)
    {
        cacaosim_wfstat_write(&job->ws, job->outdir, job->dt_phys);
        cacaosim_wfstat_free(&job->ws);
        free(job);
    }
    return NULL;
}

static MILK_HOT errno_t htsim_run_bench(const MILK_RUNNABLE *self)
{
    cacaosim_bench_cfg_t bcfg;
    cacaosim_bench_ctx_t bctx;
    char                 resolved[1024];
    char                 err_buf[256] = { 0 };

    resolve_bench_path(param_bench_file, resolved, sizeof(resolved));
    if (cacaosim_bench_load_file(resolved, &bcfg, err_buf, sizeof(err_buf)) != 0)
    {
        fprintf(stderr, "[cacaosim.htsim] .bench load error: %s\n", err_buf);
        return RETURN_FAILURE;
    }

    if (param_backend != 0)
    {
        bcfg.backend = param_backend;
    }
    if (dcfpsptr != NULL)
    {
        bcfg.turb_gain            = param_turb_gain;
        bcfg.perturb.vib_enable   = param_vib_enable ? 1 : 0;
        bcfg.perturb.vib_rms_nm   = param_vib_rms_nm;
        bcfg.perturb.petal_enable = param_petal_enable ? 1 : 0;
        bcfg.perturb.petal_rms_nm = param_petal_rms_nm;
        bcfg.perturb.petal_tau_s  = param_petal_tau_s;
        bcfg.perturb.petal_offset_nm[0] = param_petal_off0_nm;
        bcfg.perturb.petal_offset_nm[1] = param_petal_off1_nm;
        bcfg.perturb.petal_offset_nm[2] = param_petal_off2_nm;
        bcfg.perturb.petal_offset_nm[3] = param_petal_off3_nm;
        bcfg.atm_disp.enable      = param_atm_disp_enable ? 1 : 0;
        if (param_zenith_deg >= 0.0f)
        {
            bcfg.atm_disp.zenith_deg = param_zenith_deg;
        }
        bcfg.atm_disp.disp_pa_deg = param_disp_pa_deg;
    }
    else
    {
        if (param_frame_rate != 2000.0f)
        {
            bcfg.master_hz = param_frame_rate;
        }
        if (param_turb_gain != 1.0f)
        {
            bcfg.turb_gain = param_turb_gain;
        }
        else
        {
            param_turb_gain = bcfg.turb_gain;
        }
        param_atm_disp_enable = bcfg.atm_disp.enable ? 1 : 0;
        param_zenith_deg      = bcfg.atm_disp.zenith_deg;
        param_disp_pa_deg     = bcfg.atm_disp.disp_pa_deg;
    }

    /* Compile topological DAG, allocate CPU/CUDA beam slots, and bind SHM (`connect_shm = 1`) */
    if (cacaosim_bench_init(&bctx, &bcfg, NULL, 1, err_buf, sizeof(err_buf)) != 0)
    {
        fprintf(stderr, "[cacaosim.htsim] .bench init error: %s\n", err_buf);
        return RETURN_FAILURE;
    }

    /* Record base ADC settings per topologically-sorted node so live overrides remain lossless */
    int32_t base_adc_mode[CACAOSIM_BENCH_MAX_NODES];
    int32_t base_adc_band[CACAOSIM_BENCH_MAX_NODES];
    float   base_adc_sc_err[CACAOSIM_BENCH_MAX_NODES];
    float   base_adc_pa_err[CACAOSIM_BENCH_MAX_NODES];
    for (uint32_t i = 0; i < bctx.n_nodes; i++)
    {
        base_adc_mode[i]   = bctx.nodes[i].cfg.adc.mode;
        base_adc_band[i]   = bctx.nodes[i].cfg.adc.opt_band;
        base_adc_sc_err[i] = bctx.nodes[i].cfg.adc.lut_scale_err;
        base_adc_pa_err[i] = bctx.nodes[i].cfg.adc.lut_pa_err_deg;
    }
    if (dcfpsptr == NULL && param_frame_rate == 2000.0f)
    {
        param_frame_rate = bctx.eff_master_hz;
    }

    if (param_export_beams[0] != '\0')
    {
        if (cacaosim_bench_bind_export_beams(&bctx, param_export_beams, err_buf,
                                             sizeof(err_buf)) != 0)
        {
            fprintf(stderr, "[cacaosim.htsim] .export_beams error: %s\n", err_buf);
            cacaosim_bench_free(&bctx);
            return RETURN_FAILURE;
        }
    }

    /* Pre-allocate scratch buffers for sub-tick poke interpolation in .calib_temporal mode */
    uint32_t max_nact = 1u;
    for (uint32_t d = 0; d < bctx.n_dms; d++)
    {
        const cacaosim_bench_node_t *nd = &bctx.nodes[bctx.dm_node_idx[d]];
        uint32_t                     na = nd->cfg.dm.nact_x * nd->cfg.dm.nact_y;
        if (na > max_nact)
        {
            max_nact = na;
        }
    }
    float *cal_prev_cmd  = (float *) cacaosim_aligned_calloc(max_nact, sizeof(float));
    float *cal_blend_cmd = (float *) cacaosim_aligned_calloc(max_nact, sizeof(float));

    /* Emit full ANSI box-drawing schematic on interactive TTYs; plain summary when piped to logs */
    if (isatty(STDOUT_FILENO))
    {
        cacaosim_view_opts_t vopts;
        cacaosim_view_opts_defaults(&vopts);
        vopts.show_footer = 0;
        cacaosim_bench_render(&bctx, resolved, &vopts, stdout);
    }
    else
    {
        cacaosim_bench_print_summary(&bctx, stdout);
    }

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);

    cacaosim_wfstat_t wfs;
    int               wfs_active                         = 0;
    int               wfs_bidx[CACAOSIM_WFSTAT_NB_LOC]   = { 0 };
    pthread_t         wfstat_bg_thr                      = 0;
    int               wfstat_bg_running                  = 0;
    memset(&wfs, 0, sizeof(wfs));

    IMAGE   *cal_im                                 = NULL;
    int      cal_sem                                = -1;
    int      cal_dm_node                            = -1;
    uint32_t cal_dm_nact                            = 0u;
    char     cal_bound[FUNCTION_PARAMETER_STRMAXLEN] = "";
    int      prev_cal_fast                          = 0;
    int      prev_fpcam_en                          = param_enable_fpcam ? 1 : 0;
    bctx.skip_fpcam                                 = prev_fpcam_en ? 0 : 1;

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        int      do_step          = 1;
        int      cal_fast         = param_calib_fast ? 1 : 0;
        int      cal_temp         = (cal_fast && param_calib_temporal != 0) ? 1 : 0;
        int      fpcam_en         = param_enable_fpcam ? 1 : 0;
        int      trig_stream_mode = (processinfo != NULL && processinfo->triggermode == 3);
        uint32_t n_substeps       = 1u;

        bctx.skip_fpcam = fpcam_en ? 0 : 1;
        if (fpcam_en && !prev_fpcam_en)
        {
            cacaosim_bench_reset_fpcam_accum(&bctx);
        }
        prev_fpcam_en = fpcam_en;

        /* Manage lockstep calibration trigger binding transitions safely */
        if (!cal_fast && prev_cal_fast)
        {
            if (cal_im != NULL && cal_sem >= 0)
            {
                cal_im->semReadPID[cal_sem] = 0;
            }
            cal_im       = NULL;
            cal_sem      = -1;
            cal_dm_node  = -1;
            cal_dm_nact  = 0u;
            cal_bound[0] = '\0';
            cacaosim_bench_reset_wfs_accum(&bctx);
            clock_gettime(CLOCK_MONOTONIC, &deadline);
        }
        else if (cal_fast && (!prev_cal_fast || strcmp(cal_bound, param_calib_trig) != 0))
        {
            if (cal_im != NULL && cal_sem >= 0)
            {
                cal_im->semReadPID[cal_sem] = 0;
            }
            cal_im      = cacaosim_bench_find_dm_stream(&bctx, param_calib_trig);
            cal_dm_node = htsim_find_dm_node_idx(&bctx, param_calib_trig);
            cal_dm_nact = (cal_dm_node >= 0)
                              ? (bctx.nodes[cal_dm_node].cfg.dm.nact_x *
                                 bctx.nodes[cal_dm_node].cfg.dm.nact_y)
                              : 0u;
            if (cal_im != NULL && cal_prev_cmd != NULL && cal_dm_nact > 0u)
            {
                memcpy(cal_prev_cmd, cal_im->array.F, cal_dm_nact * sizeof(float));
            }
            cal_sem = (cal_im != NULL) ? ImageStreamIO_getsemwaitindex(cal_im, -1) : -1;
            if (cal_im != NULL && cal_sem >= 0)
            {
                ImageStreamIO_semflush(cal_im, cal_sem);
            }
            snprintf(cal_bound, sizeof(cal_bound), "%s", param_calib_trig);
            cacaosim_bench_reset_wfs_accum(&bctx);
        }
        prev_cal_fast = cal_fast;

        if (cal_fast && cal_im != NULL && cal_sem >= 0)
        {
            /* Wait synchronously on DM poke instead of wall-clock deadline sleep */
            struct timespec ts_wait;
            clock_gettime(CLOCK_REALTIME, &ts_wait);
            cacaosim_timespec_add_ns(&ts_wait,
                                     (param_calib_ring != 0) ? 5000000ULL : 50000000ULL);
            int wrc = ImageStreamIO_semtimedwait(cal_im, cal_sem, &ts_wait);
            if (wrc == 0 || param_calib_ring != 0)
            {
                if (wrc == 0)
                {
                    ImageStreamIO_semflush(cal_im, cal_sem);
                }
                if (processinfo != NULL)
                {
                    processinfo_exec_start(processinfo);
                }
                n_substeps = cal_temp ? htsim_calib_wfs_navg(&bctx, param_calib_wfs) : 1u;
            }
            else
            {
                do_step = 0;
            }
        }
        else if (trig_stream_mode && bctx.cfg.atm_interp && bctx.eff_oversample > 1u)
        {
            /*
             * 1:K lockstep mode triggered by base-rate `outarraypha` (`triggermode == 3`):
             * advance all `K = eff_oversample` sub-ticks per turbulence screen, spacing sub-ticks
             * by `1 / param_frame_rate` when `.frame_rate > 0.1 Hz` so closed-loop DM commands
             * arriving mid-exposure are sampled at exact sub-frame instants.
             */
            n_substeps = bctx.eff_oversample;
            clock_gettime(CLOCK_MONOTONIC, &deadline);
        }
        else if (param_frame_rate > 0.1f)
        {
            /* Drift-free absolute monotonic pacing when `.frame_rate > 0.1 Hz` */
            uint64_t period_ns = (uint64_t) (1.0e9f / param_frame_rate);
            cacaosim_timespec_add_ns(&deadline, period_ns);
            cacaosim_sleep_until(&deadline);
        }

        if (do_step)
        {
            uint32_t eff_ws_st = (param_wfstat_stride >= 1u) ? param_wfstat_stride
                                                             : bctx.eff_oversample;
            if (eff_ws_st < 1u)
            {
                eff_ws_st = 1u;
            }
            float off_ticks = (param_calib_poke_offset * 1.0e-6f) / bctx.dt_master_s;

            for (uint32_t sub = 0u; sub < n_substeps; sub++)
            {
                if (sub > 0u && !cal_fast && trig_stream_mode && param_frame_rate > 0.1f)
                {
                    if (processinfo != NULL && processinfo->loopstat == 4)
                    {
                        break;
                    }
                    uint64_t sub_ns = (uint64_t) (1.0e9f / param_frame_rate);
                    cacaosim_timespec_add_ns(&deadline, sub_ns);
                    cacaosim_sleep_until(&deadline);
                }

                bctx.calib_fast     = cal_fast;
                bctx.calib_temporal = cal_temp;
                snprintf(bctx.calib_wfs_out, sizeof(bctx.calib_wfs_out), "%s", param_calib_wfs);

                if (cal_temp && cal_dm_node >= 0 && cal_im != NULL && cal_prev_cmd != NULL &&
                    cal_blend_cmd != NULL)
                {
                    const float *cmd_now = cal_im->array.F;
                    bctx.calib_dm_node   = cal_dm_node;
                    if ((float) (sub + 1u) <= off_ticks)
                    {
                        bctx.calib_cmd_override = cal_prev_cmd;
                    }
                    else if ((float) sub >= off_ticks)
                    {
                        bctx.calib_cmd_override = cmd_now;
                    }
                    else
                    {
                        float frac = off_ticks - (float) sub;
                        float inv  = 1.0f - frac;
                        for (uint32_t a = 0u; a < cal_dm_nact; a++)
                        {
                            cal_blend_cmd[a] = frac * cal_prev_cmd[a] + inv * cmd_now[a];
                        }
                        bctx.calib_cmd_override = cal_blend_cmd;
                    }
                }
                else
                {
                    bctx.calib_dm_node      = -1;
                    bctx.calib_cmd_override = NULL;
                }

                /* One-shot reset of cumulative long-exposure PSFs & coro pre-Lyot maps in DAG */
                if (param_reset_long)
                {
                    for (uint32_t i = 0; i < bctx.n_nodes; i++)
                    {
                        if (bctx.nodes[i].cfg.kind == CACAOSIM_NODE_FPCAM)
                        {
                            cacaosim_fpcam_reset_long(&bctx.nodes[i].fpcam_ctx);
                            if (bctx.connect_shm && bctx.nodes[i].img_long.ID != -1)
                            {
                                cacaosim_fpcam_shm_write_kw(bctx.nodes[i].img_long.im, 0.0f, 0.0f,
                                                            0.0f, 0ULL);
                            }
                        }
                        else if (bctx.nodes[i].cfg.kind == CACAOSIM_NODE_CORO)
                        {
                            cacaosim_coro_reset_long(&bctx.nodes[i].coro_ctx);
                            if (bctx.connect_shm && bctx.nodes[i].img_pre_lyot_long.ID != -1)
                            {
                                cacaosim_coro_shm_write_kw(
                                    bctx.nodes[i].img_pre_lyot_long.im, 0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0ULL);
                            }
                        }
                    }
                    param_reset_long = 0;
                }

                /* Start in-DAG wavefront/Strehl telemetry before stepping so tick #1 is captured */
                if (param_wfstat_enable && !wfs_active)
                {
                    if (wfstat_bg_running)
                    {
                        pthread_join(wfstat_bg_thr, NULL);
                        wfstat_bg_running = 0;
                    }
                    param_wfstat_frames = 0;
                    double dt_ws        = (double) bctx.dt_master_s * (double) eff_ws_st;
                    if (htsim_wfstat_start(&wfs, &bctx, wfs_bidx, dt_ws) == 0)
                    {
                        wfs_active = 1;
                    }
                    else
                    {
                        param_wfstat_enable = 0;
                    }
                }

                int sample_ws = (!cal_fast && wfs_active && param_wfstat_enable &&
                                 ((bctx.tick_cnt % eff_ws_st) == 0ULL));
                int snap_ws   = (sample_ws && wfs.nsamp < CACAOSIM_WFSTAT_NB_SAMP &&
                                 (wfs.n_frames + 1ULL) >= wfs.samp_ticks[wfs.nsamp])
                                    ? 1
                                    : 0;
                int clip_ws   = (sample_ws && wfs.clip_len > 0u &&
                                 (wfs.n_frames + 1ULL) >= wfs.clip_start &&
                                 wfs.clip_n < wfs.clip_len)
                                    ? 1
                                    : 0;
                int full_ws   = (sample_ws && wfs.full_cap > 0u && wfs.full_n < wfs.full_cap &&
                                 (wfs.n_frames + 1ULL) >= wfs.full_next_frame)
                                    ? 1
                                    : 0;
                bctx.wfstat_step_active = sample_ws;
                bctx.wfstat_snap_active = (snap_ws || clip_ws || full_ws) ? 1 : 0;
                for (int k = 0; k < CACAOSIM_WFSTAT_NB_LOC; k++)
                {
                    bctx.wfstat_beam_idx[k] = wfs_bidx[k];
                }

                /* Advance entire multi-rate bench by one master tick and publish Strehl */
                if (dcfpsptr != NULL)
                {
                    bctx.cfg.perturb.vib_enable   = param_vib_enable ? 1 : 0;
                    bctx.cfg.perturb.vib_rms_nm   = param_vib_rms_nm;
                    bctx.cfg.perturb.petal_enable = param_petal_enable ? 1 : 0;
                    bctx.cfg.perturb.petal_rms_nm = param_petal_rms_nm;
                    bctx.cfg.perturb.petal_tau_s  = param_petal_tau_s;
                    bctx.cfg.perturb.petal_offset_nm[0] = param_petal_off0_nm;
                    bctx.cfg.perturb.petal_offset_nm[1] = param_petal_off1_nm;
                    bctx.cfg.perturb.petal_offset_nm[2] = param_petal_off2_nm;
                    bctx.cfg.perturb.petal_offset_nm[3] = param_petal_off3_nm;
                    bctx.cfg.atm_disp.enable      = param_atm_disp_enable ? 1 : 0;
                    if (param_zenith_deg >= 0.0f)
                    {
                        bctx.cfg.atm_disp.zenith_deg = param_zenith_deg;
                    }
                    bctx.cfg.atm_disp.disp_pa_deg = param_disp_pa_deg;
                    for (uint32_t i = 0; i < bctx.n_nodes; i++)
                    {
                        cacaosim_bench_node_t *nd = &bctx.nodes[i];
                        if (nd->cfg.kind != CACAOSIM_NODE_ADC)
                        {
                            continue;
                        }
                        int32_t emode = (param_adc_mode >= 0 && param_adc_mode <= 4)
                                            ? param_adc_mode
                                            : base_adc_mode[i];
                        int32_t eband = (param_adc_opt_band >= 0 && param_adc_opt_band <= 7)
                                            ? param_adc_opt_band
                                            : base_adc_band[i];
                        float esc = base_adc_sc_err[i] + param_adc_lut_scale_err;
                        float epa = base_adc_pa_err[i] + param_adc_lut_pa_err_deg;
                        if (nd->adc_ctx.cfg.mode != emode || nd->adc_ctx.cfg.opt_band != eband ||
                            fabsf(nd->adc_ctx.cfg.lut_scale_err - esc) > 1e-7f ||
                            fabsf(nd->adc_ctx.cfg.lut_pa_err_deg - epa) > 1e-7f)
                        {
                            nd->adc_ctx.cfg.mode           = emode;
                            nd->adc_ctx.cfg.opt_band       = eband;
                            nd->adc_ctx.cfg.lut_scale_err  = esc;
                            nd->adc_ctx.cfg.lut_pa_err_deg = epa;
                            nd->adc_ctx.cache_valid        = 0;
                        }
                    }
                }
                bctx.export_beams_active =
                    (param_export_beams[0] != '\0' && param_export_enable != 0) ? 1 : 0;
                cacaosim_bench_step(&bctx, NULL, NULL, param_turb_gain, processinfo);
                param_strehl_inst       = bctx.last_strehl_inst;
                param_strehl_ave        = bctx.last_strehl_ave;
                bctx.wfstat_step_active = 0;
                bctx.wfstat_snap_active = 0;

                /* In-DAG wavefront statistics on device beams (decimated by eff_ws_st) */
                if (wfs_active)
                {
                    if (sample_ws)
                    {
                        htsim_wfstat_frame(&wfs, &bctx, wfs_bidx);
                        param_wfstat_frames = wfs.n_frames;
                        if (param_wfstat_ticks > 0 && wfs.n_frames >= param_wfstat_ticks)
                        {
                            param_wfstat_enable = 0;
                        }
                    }
                    if (!param_wfstat_enable)
                    {
                        double dt_phys = (double) bctx.dt_master_s * (double) eff_ws_st;
                        if (wfstat_bg_running)
                        {
                            pthread_join(wfstat_bg_thr, NULL);
                            wfstat_bg_running = 0;
                        }
                        cacaosim_wfstat_sync_gpu(&wfs);
                        htsim_wfstat_job_t *job =
                            (htsim_wfstat_job_t *) malloc(sizeof(htsim_wfstat_job_t));
                        if (job != NULL)
                        {
                            job->ws = wfs;
                            snprintf(job->outdir, sizeof(job->outdir), "%s", param_wfstat_outdir);
                            job->dt_phys = dt_phys;
                            memset(&wfs, 0, sizeof(wfs));
                            wfs_active = 0;
                            if (pthread_create(&wfstat_bg_thr, NULL, htsim_wfstat_write_thread,
                                               job) == 0)
                            {
                                wfstat_bg_running = 1;
                            }
                            else
                            {
                                htsim_wfstat_write_thread(job);
                            }
                        }
                        else
                        {
                            cacaosim_wfstat_write(&wfs, param_wfstat_outdir, dt_phys);
                            cacaosim_wfstat_free(&wfs);
                            wfs_active = 0;
                        }
                    }
                }
            }

            if (cal_temp && cal_dm_node >= 0 && cal_im != NULL && cal_prev_cmd != NULL &&
                cal_dm_nact > 0u)
            {
                memcpy(cal_prev_cmd, cal_im->array.F, cal_dm_nact * sizeof(float));
            }
            bctx.calib_dm_node      = -1;
            bctx.calib_cmd_override = NULL;
        }
    }
    MILK_PROCINFO_END(self)

    if (cal_im != NULL && cal_sem >= 0)
    {
        cal_im->semReadPID[cal_sem] = 0;
    }

    if (wfstat_bg_running)
    {
        pthread_join(wfstat_bg_thr, NULL);
        wfstat_bg_running = 0;
    }
    if (wfs_active)
    {
        uint32_t eff_ws_st = (param_wfstat_stride >= 1u) ? param_wfstat_stride
                                                         : bctx.eff_oversample;
        if (eff_ws_st < 1u)
        {
            eff_ws_st = 1u;
        }
        cacaosim_wfstat_write(&wfs, param_wfstat_outdir,
                              (double) bctx.dt_master_s * (double) eff_ws_st);
        cacaosim_wfstat_free(&wfs);
    }
    free(cal_blend_cmd);
    free(cal_prev_cmd);
    cacaosim_bench_free(&bctx);
    return RETURN_SUCCESS;
}

/**
 * htsim_compute() - Primary FPS real-time execution entry point (`RUN` stage)
 * @self: Pointer to the owning `MILK_RUNNABLE` descriptor for `PROCESSINFO` lifecycle macros
 *
 * Role:
 *   Dispatches to `htsim_run_bench()` when `.bench_file` is non-empty; otherwise executes the
 *   zero-hop flat single-stage SCAO fast path coupling one DM (`cacaosim_dm_ctx_t`), one 4-sided
 *   Pyramid WFS (`cacaosim_pyrwfs_ctx_t`), and an optional focal-plane science PSF camera
 *   (`cacaosim_fpcam_ctx_t`).
 *
 * Rationale:
 *   In the flat fast path:
 *     - `wfs_ctx` and `fp_ctx` explicitly share `dm_ctx.cuda_stream` so DM surface synthesis,
 *       modulated Pyramid FFT propagation, and Soummer MFT PSF accumulation execute pipelined on a
 *       single CUDA stream without inter-stream event synchronization.
 *     - When `.out_dmopd` is left empty in CUDA mode, `host_dm_dst` is `NULL` (`cacaosim_dm_step`
 *       skips the D2H OPD transfer entirely) and `dm_ctx.d_opd_out` is passed directly as
 *       `d_dm_ext` to both `cacaosim_pyrwfs_step()` and `cacaosim_fpcam_accum_subframe()`,
 *       achieving true zero-copy GPU-resident optical coupling.
 *     - Detector structs (`wfs_det`, `fp_det`) and DM dynamic scalars (`param_delay_frames`,
 *       `param_tau_us`, `param_max_stroke`) are re-sampled from `FPFLAG_WRITERUN` parameters on
 *       every tick so live calibration/shutter/flux adjustments take effect immediately without
 *       allocating memory or rebuilding FFT plans.
 *
 * Return: `RETURN_SUCCESS` on clean loop exit, or `RETURN_FAILURE` on invalid stream/init setup.
 */
static MILK_HOT errno_t htsim_compute(const MILK_RUNNABLE *self)
{
    /* Delegate to full multi-stage `.bench` DAG host whenever `.bench_file` is configured */
    if (param_bench_file[0] != '\0')
    {
        return htsim_run_bench(self);
    }

    if (param_in_dmcmd[0] == '\0' || param_out_wfscam[0] == '\0' || param_nwf < 8)
    {
        return RETURN_FAILURE;
    }

    /*
     * Attach optional external pupil stream (`in_pup`); if absent, synthesize an anti-aliased
     * circular telescope pupil with 30% central obscuration and 45-degree 4-vane spiders.
     */
    IMGID img_pup = { 0 };
    img_pup.ID    = -1;
    if (param_in_pup[0] != '\0')
    {
        img_pup = stream_connect(param_in_pup);
    }

    uint32_t     nwf          = (img_pup.ID != -1) ? img_pup.md->size[0] : param_nwf;
    float       *fallback_pup = NULL;
    const float *pup_ptr      = NULL;

    if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT)
    {
        pup_ptr = img_pup.im->array.F;
    }
    else
    {
        fallback_pup = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
        if (fallback_pup == NULL)
        {
            return RETURN_FAILURE;
        }
        cacaosim_pupil_cfg_t pcfg = {
            .nwf        = nwf,
            .pup_diam   = param_pup_diam,
            .cobs       = 0.30f,
            .nspider    = 4,
            .spider_w   = 0.01f,
            .spider_deg = 45.0f,
        };
        cacaosim_make_pupil(&pcfg, fallback_pup);
        pup_ptr = fallback_pup;
    }

    /* Connect or auto-create required DM command input and Pyramid WFS camera output SHM streams */
    IMGID img_cmd = stream_connect_create_2Df32(param_in_dmcmd, param_nact_x, param_nact_y);
    if (img_cmd.ID == -1)
    {
        free(fallback_pup);
        imgid_free(&img_cmd);
        return RETURN_FAILURE;
    }

    uint8_t out_dt  = (param_out_dtype == 0) ? _DATATYPE_UINT16 : _DATATYPE_FLOAT;
    IMGID   img_wfs = stream_connect_create_2D(param_out_wfscam, param_wfs_ncam, param_wfs_ncam,
                                               out_dt);
    if (img_wfs.ID == -1)
    {
        free(fallback_pup);
        imgid_free(&img_cmd);
        imgid_free(&img_wfs);
        return RETURN_FAILURE;
    }

    /* Bind optional DM diagnostic OPD, focal-plane short/long PSF outputs, and atm inputs */
    IMGID img_dmopd  = { 0 };
    IMGID img_fpcam  = { 0 };
    IMGID img_fplong = { 0 };
    IMGID img_wf     = { 0 };
    IMGID img_amp    = { 0 };
    img_dmopd.ID     = -1;
    img_fpcam.ID     = -1;
    img_fplong.ID    = -1;
    img_wf.ID        = -1;
    img_amp.ID       = -1;

    if (param_out_dmopd[0] != '\0')
    {
        img_dmopd = stream_connect_create_2Df32(param_out_dmopd, nwf, nwf);
    }
    if (param_enable_fpcam && param_out_fpcam[0] != '\0')
    {
        img_fpcam = stream_connect_create_2D(param_out_fpcam, param_fp_ncam, param_fp_ncam, out_dt);
        if (param_out_fplong[0] != '\0')
        {
            img_fplong = stream_connect_create_2Df32(param_out_fplong, param_fp_ncam,
                                                     param_fp_ncam);
            if (img_fplong.ID != -1)
            {
                cacaosim_fpcam_shm_write_kw(img_fplong.im, 0.0f, 0.0f, 0.0f, 0ULL);
            }
        }
    }
    if (param_in_wf[0] != '\0')
    {
        img_wf = stream_connect(param_in_wf);
    }
    if (param_in_amp[0] != '\0')
    {
        img_amp = stream_connect(param_in_amp);
    }

    float dt_sec = (param_frame_rate > 1.0f) ? (1.0f / param_frame_rate) : 0.0005f;

    /* Initialize DM hardware context (`refl_scale = 2.0` converts mechanical stroke to OPD) */
    cacaosim_dm_cfg_t dm_cfg = {
        .nact_x       = img_cmd.md->size[0],
        .nact_y       = img_cmd.md->size[1],
        .nwf          = nwf,
        .dm_diam      = param_pup_diam,
        .if_coupling  = param_if_coupling,
        .refl_scale   = 2.0f,
        .max_stroke   = param_max_stroke,
        .dac_bits     = param_dac_bits,
        .nonlin_quad  = 0.0f,
        .dyn_mode     = param_dyn_mode,
        .dt_sec       = dt_sec,
        .tau_sec      = param_tau_us * 1.0e-6f,
        .f0_hz        = param_f0_hz,
        .zeta         = param_zeta,
        .delay_frames = param_delay_frames,
        .backend      = param_backend,
    };

    cacaosim_dm_ctx_t     dm_ctx;
    cacaosim_pyrwfs_ctx_t wfs_ctx;
    cacaosim_fpcam_ctx_t  fp_ctx;
    int                   has_fp = 0;

    if (cacaosim_dm_init(&dm_ctx, &dm_cfg, NULL) != 0)
    {
        free(fallback_pup);
        imgid_free(&img_cmd);
        imgid_free(&img_wfs);
        return RETURN_FAILURE;
    }

    /* Initialize 4-sided Pyramid WFS on `dm_ctx.cuda_stream` for zero-hop GPU pipelining */
    cacaosim_pyrwfs_cfg_t wfs_cfg = {
        .nwf        = nwf,
        .pad_factor = 2,
        .ncam       = param_wfs_ncam,
        .pup_diam   = param_pup_diam,
        .lambda_um  = param_wfs_lambda,
        .mod_rad_ld = param_mod_rad,
        .mod_npts   = param_mod_npts,
        .pup_sep    = 0.50f,
        .roof_ld    = 0.0f,
        .fstop_frac = 0.45f,
        .wf_unit    = param_wf_unit,
        .ref_lambda = param_ref_lambda,
        .rng_seed   = param_rng_seed,
        .backend    = param_backend,
        .det        = {
            .flux_ph      = param_wfs_flux,
            .dark_e       = 0.0f,
            .ron_e        = param_wfs_ron,
            .gain_eadu    = 1.0f,
            .bias_adu     = 100.0f,
            .sat_adu      = 65535.0f,
            .shutter      = param_wfs_shutter,
            .photon_noise = param_photon_noise,
            .out_dtype    = param_out_dtype,
        },
    };

    if (cacaosim_pyrwfs_init(&wfs_ctx, &wfs_cfg, pup_ptr, NULL, dm_ctx.cuda_stream) != 0)
    {
        cacaosim_dm_free(&dm_ctx);
        free(fallback_pup);
        imgid_free(&img_cmd);
        imgid_free(&img_wfs);
        return RETURN_FAILURE;
    }

    /* Optionally initialize Soummer MFT focal-plane camera on the same `dm_ctx.cuda_stream` */
    if (param_enable_fpcam && img_fpcam.ID != -1)
    {
        cacaosim_fpcam_cfg_t fp_cfg = {
            .nwf         = nwf,
            .ncam        = param_fp_ncam,
            .prop_mode   = CACAOSIM_FPCAM_PROP_MFT,
            .pad_factor  = 2,
            .pup_diam    = param_pup_diam,
            .lambda_um   = param_fp_lambda,
            .bw_frac     = 0.0f,
            .nlambda     = 1,
            .pixscale_ld = param_fp_pixscale,
            .wf_unit     = param_wf_unit,
            .ref_lambda  = param_ref_lambda,
            .rng_seed    = param_rng_seed + 7777ULL,
            .backend     = param_backend,
            .det         = {
                .flux_ph      = param_fp_flux,
                .dark_e       = 0.0f,
                .ron_e        = param_fp_ron,
                .gain_eadu    = 1.0f,
                .bias_adu     = 100.0f,
                .sat_adu      = 65535.0f,
                .shutter      = 1,
                .photon_noise = param_photon_noise,
                .out_dtype    = param_out_dtype,
            },
        };
        if (cacaosim_fpcam_init(&fp_ctx, &fp_cfg, pup_ptr, NULL, dm_ctx.cuda_stream) == 0)
        {
            has_fp = 1;
        }
    }

    /*
     * Allocate internal host DM OPD buffer only on the CPU backend when `.out_dmopd` is unused;
     * on CUDA without `.out_dmopd`, `host_dm_dst` stays NULL so zero PCIe D2H copy occurs.
     */
    float *cpu_dm_buf = NULL;
    if (!dm_ctx.use_cuda && img_dmopd.ID == -1)
    {
        cpu_dm_buf = (float *) cacaosim_aligned_calloc((size_t) nwf * nwf, sizeof(float));
    }

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        /* 1. Pace master loop via monotonic absolute deadline when `.frame_rate > 0.1 Hz` */
        if (param_frame_rate > 0.1f)
        {
            uint64_t period_ns = (uint64_t) (1.0e9f / param_frame_rate);
            cacaosim_timespec_add_ns(&deadline, period_ns);
            cacaosim_sleep_until(&deadline);
        }

        /* 2. Lazily attach atmospheric phase/amplitude SHM streams if started after `htsim` */
        if (img_wf.ID == -1 && param_in_wf[0] != '\0')
        {
            img_wf = stream_connect(param_in_wf);
        }
        if (img_amp.ID == -1 && param_in_amp[0] != '\0')
        {
            img_amp = stream_connect(param_in_amp);
        }

        const float *wf_ptr = (img_wf.ID != -1 && img_wf.md->datatype == _DATATYPE_FLOAT &&
                               img_wf.md->size[0] == nwf)
                                  ? img_wf.im->array.F
                                  : NULL;
        const float *amp_ptr = (img_amp.ID != -1 && img_amp.md->datatype == _DATATYPE_FLOAT &&
                                img_amp.md->size[0] == nwf)
                                   ? img_amp.im->array.F
                                   : NULL;

        /* 3. Advance DM actuator delay FIFO, DAC quantization, and discrete ZOH dynamics */
        float *host_dm_dst = (img_dmopd.ID != -1) ? img_dmopd.im->array.F : cpu_dm_buf;
        if (img_dmopd.ID != -1)
        {
            img_dmopd.md->write = 1;
        }

        cacaosim_dm_step(&dm_ctx, img_cmd.im->array.F, NULL, param_delay_frames,
                         param_tau_us * 1.0e-6f, param_max_stroke, host_dm_dst);

        if (img_dmopd.ID != -1)
        {
            processinfo_update_output_stream(NULL, img_dmopd.im, img_cmd.im);
        }

        /* Route GPU-resident `d_opd_out` directly in CUDA mode; host OPD buffer in CPU mode */
        const float *d_dm_ext    = dm_ctx.use_cuda ? dm_ctx.d_opd_out : NULL;
        const float *host_dm_src = dm_ctx.use_cuda ? NULL : host_dm_dst;

        /* 4. Step modulated 4-sided Pyramid WFS and publish detector readout frame */
        cacaosim_det_cfg_t wfs_det = {
            .flux_ph      = param_wfs_flux,
            .dark_e       = 0.0f,
            .ron_e        = param_wfs_ron,
            .gain_eadu    = 1.0f,
            .bias_adu     = 100.0f,
            .sat_adu      = 65535.0f,
            .shutter      = param_wfs_shutter,
            .photon_noise = param_photon_noise,
            .out_dtype    = param_out_dtype,
        };

        img_wfs.md->write = 1;
        cacaosim_pyrwfs_step(&wfs_ctx, pup_ptr, amp_ptr, wf_ptr, host_dm_src, NULL, d_dm_ext,
                             param_turb_gain, param_mod_rad, &wfs_det,
                             (param_out_dtype == 0) ? img_wfs.im->array.UI16 : NULL,
                             (param_out_dtype != 0) ? img_wfs.im->array.F : NULL);
        processinfo_update_output_stream(processinfo, img_wfs.im, img_cmd.im);

        /* 5. Accumulate science PSF subframe and read out every `fp_navg` WFS ticks */
        if (has_fp)
        {
            if (param_reset_long)
            {
                cacaosim_fpcam_reset_long(&fp_ctx);
                if (img_fplong.ID != -1)
                {
                    cacaosim_fpcam_shm_write_kw(img_fplong.im, 0.0f, 0.0f, 0.0f, 0ULL);
                }
                param_reset_long = 0;
            }
            cacaosim_fpcam_accum_subframe(&fp_ctx, pup_ptr, amp_ptr, wf_ptr, host_dm_src, NULL,
                                          d_dm_ext, param_turb_gain);

            uint32_t navg = (param_fp_navg >= 1u) ? param_fp_navg : 1u;
            if (fp_ctx.subframe_cnt >= navg)
            {
                cacaosim_det_cfg_t fp_det = {
                    .flux_ph      = param_fp_flux,
                    .dark_e       = 0.0f,
                    .ron_e        = param_fp_ron,
                    .gain_eadu    = 1.0f,
                    .bias_adu     = 100.0f,
                    .sat_adu      = 65535.0f,
                    .shutter      = 1,
                    .photon_noise = param_photon_noise,
                    .out_dtype    = param_out_dtype,
                };
                if (img_fplong.ID != -1)
                {
                    img_fplong.md->write = 1;
                }
                img_fpcam.md->write = 1;
                cacaosim_fpcam_readout(&fp_ctx, &fp_det,
                                       (param_out_dtype == 0) ? img_fpcam.im->array.UI16 : NULL,
                                       (param_out_dtype != 0) ? img_fpcam.im->array.F : NULL,
                                       (img_fplong.ID != -1) ? img_fplong.im->array.F : NULL);

                param_strehl_inst = fp_ctx.last_strehl_peak;
                param_strehl_ave  = fp_ctx.long_strehl_peak;

                processinfo_update_output_stream(NULL, img_fpcam.im, img_cmd.im);
                if (img_fplong.ID != -1)
                {
                    cacaosim_fpcam_shm_write_kw(img_fplong.im, fp_ctx.long_strehl_peak,
                                                fp_ctx.last_strehl_peak, fp_ctx.last_strehl_axis,
                                                fp_ctx.long_cnt);
                    processinfo_update_output_stream(NULL, img_fplong.im, img_cmd.im);
                }
            }
        }
    }
    MILK_PROCINFO_END(self)

    /* Tear down optical contexts in reverse dependency order (`dm_ctx` owns `cuda_stream`) */
    if (has_fp)
    {
        cacaosim_fpcam_free(&fp_ctx);
    }
    cacaosim_pyrwfs_free(&wfs_ctx);
    cacaosim_dm_free(&dm_ctx);
    free(cpu_dm_buf);
    free(fallback_pup);

    if (img_amp.ID != -1)
    {
        imgid_free(&img_amp);
    }
    if (img_wf.ID != -1)
    {
        imgid_free(&img_wf);
    }
    if (img_fplong.ID != -1)
    {
        imgid_free(&img_fplong);
    }
    if (img_fpcam.ID != -1)
    {
        imgid_free(&img_fpcam);
    }
    if (img_dmopd.ID != -1)
    {
        imgid_free(&img_dmopd);
    }
    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
    imgid_free(&img_wfs);
    imgid_free(&img_cmd);
    return RETURN_SUCCESS;
}

/* Public `milk` FPS runnable descriptor (`cacaosim.htsim` CLI & `milk-fpsexec-cacaosim-htsim`) */
const MILK_RUNNABLE runnable_cacaosim_htsim = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_htsim",
              .cmdkey      = "htsim",
              .description = "Fused zero-hop HT hardware simulator (DM + PyWFS + FPcam)" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_htsim),
    .compute   = htsim_compute,
    .confcheck = htsim_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-htsim", runnable_cacaosim_htsim);
