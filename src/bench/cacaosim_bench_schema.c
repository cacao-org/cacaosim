// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_schema.c
 * @brief   Authoritative `.bench` option registry, MILK_RUNNABLE FPS introspection & exporters
 *
 * Architectural Role:
 *   Maintains the compiled single-source-of-truth schema for all `cacaosim` configuration surfaces:
 *     1. Every declarative `.bench` section key and parser alias (`cacaosim_bench_cfg.c`),
 *        annotated with type, default, physical unit, range, execution-mode support
 *        (`fused+modular` vs `fused-only`), FPS parameter mapping, and `cacaosim-CTRL` subtab/tier.
 *     2. Direct C runtime introspection of all 12 `MILK_RUNNABLE` FPS compute units via
 *        `cacaosim_list_runnables()` and `runnable->bindings[]` — capturing live default values,
 *        `FPTYPE_*` codes, and `FPFLAG_WRITERUN` / `FPFLAG_WRITE` mutability with zero regex
 *        scraping.
 *     3. CLI flag catalogs for `cacaosim-bench`, `cacaosim-bench-view`, and `cacaosim-CTRL`.
 */

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bench/cacaosim_bench_schema.h"

/* Short shorthand macros keeping static table rows concise and <= 100 chars per line */
#define M_BOTH CACAOSIM_OPT_MODE_BOTH
#define M_FUSD CACAOSIM_OPT_MODE_FUSED
#define T_BAS  CACAOSIM_OPT_TIER_BASIC
#define T_ADV  CACAOSIM_OPT_TIER_ADVANCED

/* ── Authoritative .bench Option Table ────────────────────────────────────── */

static const cacaosim_bench_opt_t g_bench_opts[] = {
    /* [bench] — Global telescope, pupil, stellar photometry, master clock & runtime */
    { "bench", "bench", "name", "", "string", "", "ao_bench", "", "", "optics", M_BOTH, "", 0,
      T_BAS, 1, "Bench configuration identifier shown in headers and reports" },
    { "bench", "bench", "nwf", "", "uint", "", "128", "px", ">= 8 (even)", "optics", M_BOTH,
      "mkbeam.nwf", 0, T_BAS, 1, "Square pupil-plane wavefront grid size shared across all beams" },
    { "bench", "bench", "pup_diam", "", "float", "", "0.9375", "nwf", "(0..1]", "optics", M_BOTH,
      "mkbeam.pup_diam", 0, T_BAS, 1, "Clear telescope pupil outer diameter as fraction of nwf" },
    { "bench", "bench", "cobs", "obs_frac", "float", "", "0.30", "D", "[0..1)", "optics", M_BOTH,
      "mkbeam.obs_frac", 0, T_BAS, 1, "Central secondary obstruction diameter ratio D_sec/D_tel" },
    { "bench", "bench", "nspider", "n_spiders", "int", "", "4", "", "[0..12]", "optics", M_BOTH,
      "mkbeam.n_spiders", 0, T_BAS, 1, "Number of secondary-mirror support spider vanes" },
    { "bench", "bench", "spider_w", "", "float", "", "0.012", "D", "[0..0.15]", "optics", M_BOTH,
      "mkbeam.spider_w", 0, T_BAS, 1, "Spider vane thickness as fraction of clear pupil diameter" },
    { "bench", "bench", "spider_deg", "", "float", "", "45.0", "deg", "[0..180]", "optics", M_BOTH,
      "mkbeam.spider_deg", 0, T_BAS, 1, "Azimuth angle of first spider vane CCW from +X" },
    { "bench", "bench", "spider_offset", "", "float", "", "0.0", "D", "[0..0.45]", "optics",
      M_BOTH, "mkbeam.spider_offset", 0, T_BAS, 1,
      "V-spider X-vertex offset fraction of pupil diameter (0 = radial; Subaru = 0.08289)" },
    { "bench", "bench", "pupil_geom", "", "preset", "subaru|radial|default", "default", "", "",
      "optics", M_BOTH, "", -1, T_ADV, 0,
      "Preset macro setting tel_diam_m, cobs, spiders & spider_offset (e.g. 'subaru')" },
    { "bench", "bench", "tel_diam_m", "", "float", "", "8.2", "m", "> 0", "optics", M_BOTH,
      "mkbeam.tel_diam_m", 0, T_BAS, 1, "Primary mirror outer diameter used for photon collecting" },
    { "bench", "bench", "star_mag", "", "float", "", "8.0", "mV", "[-2..22]", "photometry", M_BOTH,
      "mkbeam.star_mag", 1, T_BAS, 1, "Guide/science star apparent Vega V-band magnitude" },
    { "bench", "bench", "star_teff", "", "float", "", "5778.0", "K", "[2000..35000]", "photometry",
      M_BOTH, "mkbeam.star_teff", 1, T_BAS, 1, "Stellar blackbody temperature shaping 0.4-2.4 um" },
    { "bench", "bench", "tput_tel", "tel_tput", "float", "", "0.75", "", "(0..1]", "photometry",
      M_BOTH, "mkbeam.tel_tput", 1, T_BAS, 1, "Combined sky + telescope + fore-optics transmission" },
    { "bench", "bench", "master_hz", "frame_rate", "float", "", "2000.0", "Hz", "> 0", "timing",
      M_BOTH, "htsim.frame_rate", 1, T_BAS, 1, "Physical bench master hardware clock rate" },
    { "bench", "bench", "oversample", "", "uint", "", "1", "x", ">= 1", "timing", M_BOTH,
      "htsim.frame_rate", 1, T_ADV, 1,
      "Internal temporal oversampling factor K (dt_internal = 1 / (K * master_hz))" },
    { "bench", "bench", "atm_interp", "", "bool", "0|1", "1", "", "0|1", "timing", M_FUSD, "", 1,
      T_ADV, 1, "Linearly interpolate dual atmospheric screens across oversampled substeps" },
    { "bench", "bench", "telemetry_stride", "", "uint", "", "0", "ticks", ">= 0", "runtime", M_FUSD,
      "", 1, T_ADV, 1, "Decimation stride for optional diagnostic beam exports (0 = every tick)" },
    { "bench", "bench", "in_wf", "", "string", "", "outarraypha", "shm", "", "wiring", M_BOTH,
      "mkbeam.in_wf", 2, T_BAS, 1, "Input 2D atmospheric turbulence phase/OPD SHM stream" },
    { "bench", "bench", "in_amp", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "mkbeam.in_amp", 2, T_BAS, 1, "Optional 2D atmospheric scintillation amplitude SHM stream" },
    { "bench", "bench", "in_pup", "", "string", "", "csim_pup", "shm", "", "wiring", M_BOTH,
      "mkbeam.in_pup", 2, T_BAS, 1, "Input entrance pupil amplitude SHM stream (or FITS path)" },
    { "bench", "bench", "in_ncpa", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "mkbeam.in_ncpa", 2, T_BAS, 1, "Optional common-path static NCPA OPD [um] SHM stream" },
    { "bench", "bench", "out_beam", "source_beam", "string", "", "b_pup", "beam", "", "wiring",
      M_BOTH, "mkbeam.out_beam", 2, T_BAS, 1, "Root optical beam name emitted from entrance pupil" },
    { "bench", "bench", "wf_unit", "", "enum", "um|rad", "rad", "", "opd|um|rad|phase|0|1",
      "photometry", M_BOTH, "mkbeam.wf_unit", 1, T_BAS, 1,
      "Physical unit of in_wf stream: 'um' (OPD microns) or 'rad' (phase at ref_lambda)" },
    { "bench", "bench", "ref_lambda", "", "float", "", "0.50", "um", "[0.3..3.0]", "photometry",
      M_BOTH, "mkbeam.ref_lambda", 1, T_BAS, 1, "Reference wavelength when wf_unit == 'rad'" },
    { "bench", "bench", "turb_gain", "", "float", "", "1.0", "x", ">= 0", "photometry", M_BOTH,
      "htsim.turb_gain", 1, T_BAS, 1, "Dimensionless multiplier scaling input atmospheric OPD" },
    { "bench", "bench", "backend", "", "enum", "auto|cpu|cuda", "auto", "", "auto|cpu|cuda|gpu",
      "runtime", M_BOTH, "htsim.backend", 1, T_BAS, 1, "Compute execution engine selection" },

    /* [common] — Universal wiring & 2D affine pupil registration shared by optical nodes */
    { "common", "dm,ttm,splitter,coro,wfe,static_wfe,pyrwfs,shwfs,zwfs,fpcam,fpwfs", "in",
      "in_beam", "string", "", "(required)", "beam", "", "wiring", M_BOTH, "<unit>.in_beam", 2,
      T_BAS, 1, "Upstream optical beam name consumed by this optical element" },
    { "common", "dm,ttm,coro,wfe,static_wfe", "out", "out_beam", "string", "", "(required)", "beam",
      "", "wiring", M_BOTH, "<unit>.out_beam", 2, T_BAS, 1,
      "Primary downstream optical beam produced by inline elements (DM, TTM, WFE, Coro)" },
    { "common", "dm,ttm,coro,wfe,static_wfe,pyrwfs,shwfs,zwfs,fpcam,fpwfs", "shift_x", "", "float",
      "", "0.0", "px", "[-32..32]", "registration", M_BOTH, "<unit>.shift_x", 2, T_BAS, 1,
      "Lateral X pupil misregistration in wavefront pixels (coro: fused-only)" },
    { "common", "dm,ttm,coro,wfe,static_wfe,pyrwfs,shwfs,zwfs,fpcam,fpwfs", "shift_y", "", "float",
      "", "0.0", "px", "[-32..32]", "registration", M_BOTH, "<unit>.shift_y", 2, T_BAS, 1,
      "Lateral Y pupil misregistration in wavefront pixels (coro: fused-only)" },
    { "common", "dm,ttm,coro,wfe,static_wfe,pyrwfs,shwfs,zwfs,fpcam,fpwfs", "rot_deg", "", "float",
      "", "0.0", "deg", "[-180..180]", "registration", M_BOTH, "<unit>.rot_deg", 2, T_BAS, 1,
      "In-plane pupil clocking rotation angle (coro: fused-only)" },
    { "common", "dm,ttm,coro,wfe,static_wfe,pyrwfs,shwfs,zwfs,fpcam,fpwfs", "mag_scale", "",
      "float", "", "1.0", "x", "[0.5..1.5]", "registration", M_BOTH, "<unit>.mag_scale", 2, T_BAS,
      1, "Radial pupil magnification scaling factor (coro: fused-only)" },

    /* [dm.<id>] & [ttm.<id>] — Deformable Mirror and 2-Axis Tip-Tilt Mirror */
    { "dm", "dm,ttm", "in_cmd", "", "string", "", "<id>_cmd", "shm", "", "wiring", M_BOTH,
      "dmsim.in_cmd", 2, T_BAS, 1, "Input 2D float32 actuator command SHM stream [um]" },
    { "dm", "dm,ttm", "out_opd", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "dmsim.out_opd", 2, T_BAS, 1, "Optional output 2D DM surface OPD telemetry SHM stream [um]" },
    { "dm", "dm,ttm", "in_flat", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "dmsim.in_flat", 2, T_BAS, 1, "Optional static flat-map actuator offset SHM stream" },
    { "dm", "dm,ttm", "dm_mode", "", "enum", "0|1", "0 (ttm: 1)", "", "0=zonal|1=ttm", "optics",
      M_BOTH, "dmsim.dm_mode", 0, T_BAS, 0,
      "Mirror geometry mode: 0 = zonal actuator grid ([dm.*]), 1 = 2-axis tip-tilt ([ttm.*])" },
    { "dm", "dm,ttm", "nact_x", "nact", "uint", "", "16 (ttm: 2)", "act", "[2..128]", "optics",
      M_BOTH, "dmsim.nact_x", 0, T_BAS, 1,
      "Actuator grid count along X ('nact' alias sets both nact_x and nact_y)" },
    { "dm", "dm,ttm", "nact_y", "", "uint", "", "16 (ttm: 1)", "act", "[1..128]", "optics", M_BOTH,
      "dmsim.nact_y", 0, T_BAS, 1, "Actuator grid count along Y" },
    { "dm", "dm,ttm", "dm_diam", "", "float", "", "0.0", "act", ">= 0", "optics", M_BOTH,
      "dmsim.dm_diam", 0, T_ADV, 1,
      "Illuminated pupil diameter in actuator pitches (0 = auto nact_x - 2)" },
    { "dm", "dm,ttm", "if_coupling", "", "float", "", "0.20", "", "[0.02..0.60]", "optics", M_BOTH,
      "dmsim.if_coupling", 0, T_BAS, 1, "Nearest-neighbor Gaussian influence function coupling" },
    { "dm", "dm,ttm", "tput", "", "float", "", "0.98", "", "(0..1]", "photometry", M_BOTH,
      "dmsim.tput", 0, T_BAS, 1, "Mirror optical intensity reflectivity" },
    { "dm", "dm,ttm", "refl_scale", "", "float", "", "2.0", "x", "[1..4]", "optics", M_BOTH,
      "dmsim.refl_scale", 0, T_BAS, 1, "OPD per surface displacement (2.0 = normal incidence)" },
    { "dm", "dm,ttm", "max_stroke", "", "float", "", "5.0", "um", "> 0", "optics", M_BOTH,
      "dmsim.max_stroke", 0, T_BAS, 1, "Symmetric mechanical actuator stroke clip [+/- um]" },
    { "dm", "dm,ttm", "dac_bits", "", "int", "", "16", "bits", "[0..24]", "dynamics", M_BOTH,
      "dmsim.dac_bits", 1, T_BAS, 1, "Drive electronics DAC quantization bits (0 = ideal float)" },
    { "dm", "dm,ttm", "nonlin_quad", "", "float", "", "0.0", "", "[-0.5..0.5]", "dynamics", M_BOTH,
      "dmsim.nonlin_quad", 1, T_BAS, 1, "Quadratic electrostrictive/piezo actuator nonlinearity" },
    { "dm", "dm,ttm", "dyn_mode", "", "enum", "instant|rc|osc|plate", "rc", "",
      "instant|rc|osc|plate|0..3", "dynamics", M_BOTH, "dmsim.dyn_mode", 1, T_BAS, 1,
      "DM temporal model: instant(0), RC(1), 2nd-order osc(2), or coupled 2D plate PDE(3)" },
    { "dm", "dm,ttm", "tau_us", "tau_sec", "float", "", "120.0", "us", ">= 0", "dynamics", M_BOTH,
      "dmsim.tau_sec", 1, T_BAS, 1,
      "1st-order RC settling time constant [us] (alias 'tau_sec' takes SI seconds)" },
    { "dm", "dm,ttm", "f0_hz", "", "float", "", "2500.0", "Hz", "> 0", "dynamics", M_BOTH,
      "dmsim.f0_hz", 1, T_BAS, 1, "Natural resonance frequency [Hz] (mode 2 osc & mode 3 plate)" },
    { "dm", "dm,ttm", "zeta", "", "float", "", "0.70", "", "[0.05..2.0]", "dynamics", M_BOTH,
      "dmsim.zeta", 1, T_BAS, 1, "2nd-order oscillator damping ratio (0.7 = critically damped)" },
    { "dm", "dm", "act_pitch_mm", "", "float", "", "1.5", "mm", "> 0", "dynamics", M_BOTH,
      "dmsim.act_pitch_mm", 1, T_ADV, 1,
      "Physical actuator spacing [mm] for coupled 2D plate PDE (dyn_mode=plate)" },
    { "dm", "dm", "wave_c", "", "float", "", "14.0", "m/s", ">= 0", "dynamics", M_BOTH,
      "dmsim.wave_c", 1, T_ADV, 1,
      "Facesheet membrane wave speed c_m [m/s] (dyn_mode=plate)" },
    { "dm", "dm", "bend_beta", "", "float", "", "0.0085", "m2/s", ">= 0", "dynamics", M_BOTH,
      "dmsim.bend_beta", 1, T_ADV, 1,
      "Flexural plate dispersion coeff beta=sqrt(D/rho*h) [m^2/s] (dyn_mode=plate)" },
    { "dm", "dm", "damp_g0", "", "float", "", "195.0", "1/s", "> 0", "dynamics", M_BOTH,
      "dmsim.damp_g0", 1, T_ADV, 1,
      "Viscous modal decay rate gamma_0 [1/s] (dyn_mode=plate)" },
    { "dm", "dm", "damp_g2", "", "float", "", "1.5", "", ">= 0", "dynamics", M_BOTH,
      "dmsim.damp_g2", 1, T_ADV, 1,
      "Kelvin-Voigt Laplacian high-k damping [pitch^2/s] (dyn_mode=plate)" },
    { "dm", "dm", "plate_drive", "", "enum", "disp|force", "force", "", "disp|force|0|1",
      "dynamics", M_BOTH, "dmsim.plate_drive", 1, T_ADV, 1,
      "Plate drive coupling: voice-coil force (force/1) or displacement-calibrated (disp/0)" },
    { "dm", "dm", "rim_bc", "", "enum", "clamped|free", "free", "", "clamped|free|0|1",
      "dynamics", M_BOTH, "dmsim.rim_bc", 1, T_ADV, 1,
      "Circular rim boundary condition: free Neumann (1) or clamped Dirichlet (0)" },
    { "dm", "dm,ttm", "delay_frames", "", "float", "", "1.0", "fr", "[0..7]", "dynamics", M_BOTH,
      "dmsim.delay_frames", 1, T_BAS, 1,
      "Fractional master-clock latency [frames] (active when delay_us == 0)" },
    { "dm", "dm,ttm", "delay_us", "delay_sec", "float", "", "0.0", "us", ">= 0", "dynamics", M_BOTH,
      "dmsim.delay_frames", 1, T_ADV, 1,
      "Physical command transport delay [us] overriding delay_frames ('delay_sec' takes seconds)" },
    { "dm", "dm,ttm", "zoh_mean", "", "bool", "0|1", "1", "", "0|1", "dynamics", M_FUSD, "", 1,
      T_ADV, 1, "Integrate sub-tick ZOH exposure average across actuator transition (1 = ON)" },

    /* [splitter.<id>] — Gray & Dichroic Beamsplitter */
    { "splitter", "splitter", "out_refl", "", "string", "", "", "beam", "", "wiring", M_BOTH,
      "beamsplit.out_refl", 2, T_BAS, 1, "Reflected output arm optical beam name" },
    { "splitter", "splitter", "out_trans", "", "string", "", "", "beam", "", "wiring", M_BOTH,
      "beamsplit.out_trans", 2, T_BAS, 1, "Transmitted output arm optical beam name" },
    { "splitter", "splitter", "mode", "split_mode", "enum",
      "gray|dichroic_lp|dichroic_sp|dichroic_bp", "dichroic_lp", "", "gray|lp|sp|bp|0..3", "optics",
      M_BOTH, "beamsplit.split_mode", 0, T_BAS, 1,
      "Spectral split law: achromatic gray, long-pass, short-pass, or band-reflect notch" },
    { "splitter", "splitter", "r_gray", "", "float", "", "0.50", "", "[0..1]", "optics", M_BOTH,
      "beamsplit.r_gray", 0, T_BAS, 1, "Achromatic reflected power fraction (mode == 'gray')" },
    { "splitter", "splitter", "t_gray", "", "float", "", "0.50", "", "[0..1]", "optics", M_BOTH,
      "beamsplit.t_gray", 0, T_BAS, 1, "Achromatic transmitted power fraction (mode == 'gray')" },
    { "splitter", "splitter", "lambda_cut", "", "float", "", "0.95", "um", "[0.4..2.4]", "optics",
      M_BOTH, "beamsplit.lambda_cut", 0, T_BAS, 1, "Dichroic 50% transition cutoff or notch center" },
    { "splitter", "splitter", "band_width", "", "float", "", "0.25", "um", "[0.05..1.2]", "optics",
      M_BOTH, "beamsplit.band_width", 0, T_BAS, 1, "Reflected notch full width (dichroic_bp mode)" },
    { "splitter", "splitter", "roll_um", "", "float", "", "0.03", "um", "> 0", "optics", M_BOTH,
      "beamsplit.roll_um", 0, T_BAS, 1, "Dichroic logistic transition edge roll-off width" },
    { "splitter", "splitter", "eff_refl", "", "float", "", "0.95", "", "[0..1]", "photometry",
      M_BOTH, "beamsplit.eff_refl", 1, T_BAS, 1, "Peak coating efficiency in reflected passband" },
    { "splitter", "splitter", "eff_trans", "", "float", "", "0.95", "", "[0..1]", "photometry",
      M_BOTH, "beamsplit.eff_trans", 1, T_BAS, 1, "Peak coating efficiency in transmitted passband" },
    { "splitter", "splitter", "ncpa_refl", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "beamsplit.ncpa_refl", 2, T_BAS, 1, "Optional static NCPA OPD [um] SHM map on reflected arm" },
    { "splitter", "splitter", "ncpa_trans", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "beamsplit.ncpa_trans", 2, T_BAS, 1, "Optional static NCPA OPD [um] SHM map on trans arm" },
    { "splitter", "splitter", "pup_refl", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "beamsplit.pup_refl", 2, T_ADV, 1, "Optional pupil amplitude mask SHM on reflected arm" },
    { "splitter", "splitter", "pup_trans", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "beamsplit.pup_trans", 2, T_ADV, 1, "Optional pupil amplitude mask SHM on transmitted arm" },
    { "splitter", "splitter", "refl_dx", "", "float", "", "0.0", "px", "[-32..32]", "registration",
      M_BOTH, "beamsplit.refl_dx", 2, T_ADV, 1, "Reflected-arm lateral X pupil shift [px]" },
    { "splitter", "splitter", "refl_dy", "", "float", "", "0.0", "px", "[-32..32]", "registration",
      M_BOTH, "beamsplit.refl_dy", 2, T_ADV, 1, "Reflected-arm lateral Y pupil shift [px]" },
    { "splitter", "splitter", "refl_rot", "", "float", "", "0.0", "deg", "[-180..180]",
      "registration", M_BOTH, "beamsplit.refl_rot", 2, T_ADV, 1, "Reflected-arm pupil clocking" },
    { "splitter", "splitter", "refl_scale", "", "float", "", "1.0", "x", "[0.5..1.5]",
      "registration", M_BOTH, "beamsplit.refl_scale", 2, T_ADV, 1,
      "Reflected-arm radial pupil magnification factor" },
    { "splitter", "splitter", "trans_dx", "", "float", "", "0.0", "px", "[-32..32]", "registration",
      M_BOTH, "beamsplit.trans_dx", 2, T_ADV, 1, "Transmitted-arm lateral X pupil shift [px]" },
    { "splitter", "splitter", "trans_dy", "", "float", "", "0.0", "px", "[-32..32]", "registration",
      M_BOTH, "beamsplit.trans_dy", 2, T_ADV, 1, "Transmitted-arm lateral Y pupil shift [px]" },
    { "splitter", "splitter", "trans_rot", "", "float", "", "0.0", "deg", "[-180..180]",
      "registration", M_BOTH, "beamsplit.trans_rot", 2, T_ADV, 1, "Transmitted-arm pupil clocking" },
    { "splitter", "splitter", "trans_scale", "", "float", "", "1.0", "x", "[0.5..1.5]",
      "registration", M_BOTH, "beamsplit.trans_scale", 2, T_ADV, 1,
      "Transmitted-arm radial pupil magnification factor" },

    /* [coro.<id>] — Coronagraphic FPM, Lyot Stop, Custom 2D/3D Masks & LOWFS Pickoff */
    { "coro", "coro", "out_lowfs", "", "string", "", "", "beam", "", "wiring", M_BOTH,
      "coro.out_lowfs", 2, T_BAS, 1, "Optional reflected LOWFS pupil pickoff optical beam name" },
    { "coro", "coro", "out_pre_lyot", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "coro.out_pre_lyot", 2, T_ADV, 1, "Optional pre-Lyot pupil intensity telemetry SHM stream" },
    { "coro", "coro", "out_pre_lyot_long", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "coro.out_pre_lyot_long", 2, T_ADV, 1, "Optional cumulative pre-Lyot pupil average SHM" },
    { "coro", "coro", "in_pup_amp", "", "string", "", "", "mask", "", "wiring", M_BOTH,
      "coro.in_pup_amp", 2, T_BAS, 1, "Custom 2D/3D pre-FPM pupil apodizer amplitude (FITS/SHM)" },
    { "coro", "coro", "in_pup_pha", "", "string", "", "", "mask", "", "wiring", M_BOTH,
      "coro.in_pup_pha", 2, T_BAS, 1, "Custom 2D/3D pre-FPM pupil phase/OPD mask (FITS/SHM)" },
    { "coro", "coro", "in_fpm_amp", "", "string", "", "", "mask", "", "wiring", M_BOTH,
      "coro.in_fpm_amp", 2, T_BAS, 1, "Custom 2D/3D focal-plane mask amplitude (FITS/SHM)" },
    { "coro", "coro", "in_fpm_pha", "", "string", "", "", "mask", "", "wiring", M_BOTH,
      "coro.in_fpm_pha", 2, T_BAS, 1, "Custom 2D/3D focal-plane phase/OPD mask (FITS/SHM)" },
    { "coro", "coro", "in_lyot", "in_lyot_amp", "string", "", "", "mask", "", "wiring", M_BOTH,
      "coro.in_lyot", 2, T_BAS, 1, "Custom 2D/3D Lyot-stop amplitude mask (FITS/SHM)" },
    { "coro", "coro", "in_lyot_pha", "", "string", "", "", "mask", "", "wiring", M_BOTH,
      "coro.in_lyot_pha", 2, T_BAS, 1, "Custom 2D/3D Lyot-stop phase/OPD mask (FITS/SHM)" },
    { "coro", "coro", "mode", "coro_mode", "enum",
      "lyot_spot|phase_dimple|vortex|4qpm|custom_mft|custom_fft", "lyot_spot", "", "0..5", "optics",
      M_BOTH, "coro.coro_mode", 0, T_BAS, 1, "Focal-plane mask architecture and propagator" },
    { "coro", "coro", "lambda_um", "", "float", "", "1.65", "um", "[0.4..2.4]", "optics", M_BOTH,
      "coro.lambda_um", 0, T_BAS, 1, "Coronagraph design reference wavelength" },
    { "coro", "coro", "bw_frac", "dlambda_um", "float", "", "0.0", "", "[0..0.6]", "optics", M_BOTH,
      "coro.bw_frac", 0, T_BAS, 1,
      "Polychromatic fractional bandwidth dLambda/Lambda ('dlambda_um' takes microns)" },
    { "coro", "coro", "nlambda", "", "int", "", "0", "ch", "[0..32]", "optics", M_BOTH,
      "coro.nlambda", 0, T_BAS, 1, "Polychromatic cube slices (0 = auto from bw_frac / 3D FITS)" },
    { "coro", "coro", "pha_unit", "mask_pha_unit", "enum", "rad|um", "rad", "", "rad|phase|opd|um",
      "optics", M_BOTH, "coro.pha_unit", 0, T_BAS, 1,
      "Physical unit of custom phase mask maps: 'rad' (radians) or 'um' (OPD microns)" },
    { "coro", "coro", "fpm_rad_ld", "", "float", "", "3.0", "l/D", "[0.5..12]", "optics", M_BOTH,
      "coro.fpm_rad_ld", 0, T_BAS, 1, "Analytic Lyot spot or phase dimple FPM radius [lambda/D]" },
    { "coro", "coro", "fpm_amp", "", "float", "", "0.0", "", "[0..1]", "optics", M_BOTH,
      "coro.fpm_amp", 0, T_BAS, 1, "Analytic FPM inner core field amplitude transmission" },
    { "coro", "coro", "fpm_phase", "", "float", "", "3.14159", "rad", "[0..2pi]", "optics", M_BOTH,
      "coro.fpm_phase", 0, T_BAS, 1, "Phase shift inside phase_dimple (Roddier) FPM [rad]" },
    { "coro", "coro", "n_mft", "", "uint", "", "48", "px", "[16..256]", "optics", M_BOTH,
      "coro.n_mft", 0, T_ADV, 1, "Babinet Matrix-DFT sampling grid across analytic FPM spot" },
    { "coro", "coro", "fpm_pixscale", "fpm_pixscale_ld", "float", "", "8.0", "px/(l/D)",
      "[0.5..64]", "optics", M_BOTH, "coro.fpm_pixscale", 0, T_BAS, 1,
      "Custom MFT focal mask plate scale [pixels per lambda_0/D]" },
    { "coro", "coro", "vortex_charge", "", "int", "", "2", "", "2|4|6|8", "optics", M_BOTH,
      "coro.vortex_charge", 0, T_BAS, 1, "Vortex coronagraph topological charge l (even int)" },
    { "coro", "coro", "pad_factor", "", "uint", "", "4", "x", "[2..8]", "optics", M_BOTH,
      "coro.pad_factor", 0, T_BAS, 1, "Zero-padded 2D FFT oversampling for vortex/4qpm/custom_fft" },
    { "coro", "coro", "lowfs_mode", "", "enum", "fpm_refl|lyot_refl", "fpm_refl", "",
      "fpm_refl|lyot_refl|0|1", "optics", M_BOTH, "coro.lowfs_mode", 1, T_BAS, 1,
      "LOWFS pickoff plane: starlight rejected at FPM ('fpm_refl') or reflective Lyot stop" },
    { "coro", "coro", "lowfs_pup_diam", "lowfs_pup", "float", "", "1.0", "D_pup", "[0.05..2.0]",
      "optics", M_BOTH, "coro.lowfs_pup_diam", 1, T_BAS, 1,
      "Pupil stop diameter ratio on FPM-reflected LOWFS path" },
    { "coro", "coro", "lowfs_refl", "", "float", "", "0.95", "", "[0..1]", "photometry", M_BOTH,
      "coro.lowfs_refl", 1, T_BAS, 1, "Amplitude reflectivity into LOWFS pickoff arm" },
    { "coro", "coro", "tput", "", "float", "", "0.92", "", "(0..1]", "photometry", M_BOTH,
      "coro.tput", 1, T_BAS, 1, "Coronagraph relay optical intensity throughput" },
    { "coro", "coro", "lyot_outer", "", "float", "", "0.85", "D", "[0.5..1.0]", "optics", M_BOTH,
      "coro.lyot_outer", 1, T_BAS, 1, "Lyot stop outer clear diameter ratio vs entrance pupil" },
    { "coro", "coro", "lyot_obs", "", "float", "", "0.36", "D", "[0..0.8]", "optics", M_BOTH,
      "coro.lyot_obs", 1, T_BAS, 1, "Lyot stop central obstruction diameter ratio" },
    { "coro", "coro", "lyot_nspider", "", "int", "", "4", "", "[0..12]", "optics", M_BOTH,
      "coro.lyot_nspider", 1, T_BAS, 1, "Number of spider vanes masked by Lyot stop" },
    { "coro", "coro", "lyot_spider_w", "", "float", "", "0.02", "D", "[0..0.15]", "optics", M_BOTH,
      "coro.lyot_spider_w", 1, T_BAS, 1, "Lyot stop spider mask width fraction" },
    { "coro", "coro", "lyot_spider_deg", "", "float", "", "45.0", "deg", "[0..180]", "optics",
      M_BOTH, "coro.lyot_spider_deg", 1, T_BAS, 1, "Lyot stop spider orientation angle [deg]" },
    { "coro", "coro", "lyot_spider_offset", "", "float", "", "0.0", "D", "[0..0.45]", "optics",
      M_BOTH, "coro.lyot_spider_offset", 1, T_BAS, 1,
      "Lyot stop V-spider X-vertex offset fraction of pup_diam" },

    /* [wfe.<id>] / [static_wfe.<id>] — Inline Static Wavefront Error OPD [meter] */
    { "wfe", "wfe,static_wfe", "in_wfe", "stream", "string", "", "<id>_wfe", "shm", "", "wiring",
      M_BOTH, "wfe.in_wfe", 2, T_BAS, 1, "Input/auto-created 2D static WFE OPD [meter] SHM stream" },
    { "wfe", "wfe,static_wfe", "rms_nm", "rms_m", "float", "", "50.0", "nm", ">= 0", "optics",
      M_BOTH, "wfe.rms_nm", 0, T_BAS, 1,
      "Target pupil-weighted RMS OPD when synthesizing map ('rms_m' takes meters)" },
    { "wfe", "wfe,static_wfe", "defocus_nm", "defocus_m", "float", "", "0.0", "nm", "[-5000..5000]",
      "optics", M_BOTH, "wfe.defocus_nm", 0, T_BAS, 1,
      "Additive smooth Zernike Z4 defocus RMS across pupil ('defocus_m' takes meters)" },
    { "wfe", "wfe,static_wfe", "psd_alpha", "alpha", "float", "", "1.6", "", "[0.5..4.0]", "optics",
      M_BOTH, "wfe.psd_alpha", 0, T_BAS, 1, "Power-law spatial PSD amplitude exponent f^-alpha" },
    { "wfe", "wfe,static_wfe", "wfe_gain", "gain", "float", "", "1.0", "x", "[-10..10]", "optics",
      M_BOTH, "wfe.wfe_gain", 0, T_BAS, 1, "Dimensionless multiplier on static OPD map (0=bypass)" },
    { "wfe", "wfe,static_wfe", "wfe_seed", "seed", "uint", "", "(salted)", "", ">= 1", "optics",
      M_BOTH, "wfe.wfe_seed", 0, T_BAS, 1, "Deterministic PRNG seed for WFE map synthesis" },
    { "wfe", "wfe,static_wfe", "tput", "", "float", "", "0.98", "", "(0..1]", "photometry", M_BOTH,
      "wfe.tput", 0, T_BAS, 1, "Optical intensity transmission factor" },

    /* [sensor] — Shared Cadence, Timing, Photometry & Detector Readout (WFS + FPCAM) */
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "out_cam", "", "string", "", "<id>_cam", "shm", "",
      "wiring", M_BOTH, "<wfs>.out_cam", 2, T_BAS, 1, "Output 2D camera frame SHM stream name" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "in_ncpa", "", "string", "", "", "shm", "",
      "wiring", M_BOTH, "<wfs>.in_ncpa", 2, T_BAS, 1,
      "Optional per-sensor static NCPA OPD [um] SHM map" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "bw_frac", "dlambda_um", "float", "",
      "0.20 (fpcam: 0.18)", "", "[0..0.6]", "optics", M_BOTH, "<wfs>.bw_frac", 0, T_BAS, 1,
      "Fractional spectral passband dLambda/Lambda ('dlambda_um' takes microns)" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "nlambda", "", "int", "", "0", "ch", "[0..32]",
      "optics", M_BOTH, "<wfs>.nlambda", 0, T_BAS, 1,
      "Polychromatic optical slices integrated per subframe (0 = auto)" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "navg", "", "uint", "", "1", "ticks", ">= 1",
      "timing", M_BOTH, "<wfs>.navg", 1, T_BAS, 1,
      "Master-clock subframes integrated per camera readout when rate_hz == 0" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "rate_hz", "", "float", "", "0.0", "Hz", ">= 0",
      "timing", M_BOTH, "<wfs>.navg", 1, T_BAS, 1,
      "Target physical camera readout rate [Hz]; if > 0.1 overrides navg" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "clock_offset_us", "", "float", "", "0.0", "us",
      "[-1e5..1e5]", "timing", M_FUSD, "", 1, T_ADV, 1,
      "Static camera trigger phase offset relative to master clock [us]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "clock_drift_ppm", "", "float", "", "0.0", "ppm",
      "[-5000..5000]", "timing", M_FUSD, "", 1, T_ADV, 1,
      "Relative camera oscillator clock drift [parts per million]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "subframe_stride", "", "uint", "", "0", "ticks",
      ">= 0", "timing", M_FUSD, "", 1, T_ADV, 1,
      "Optical propagation decimation stride across oversampled subframes (0 = auto)" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "sim_hz", "", "float", "", "0.0", "Hz", ">= 0",
      "timing", M_FUSD, "", 1, T_ADV, 1,
      "Target optical propagation sampling rate [Hz] deriving subframe_stride (0 = off)" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "qe", "", "float", "", "0.80 (fpcam: 0.85)", "",
      "(0..1]", "photometry", M_BOTH, "<wfs>.qe", 1, T_BAS, 1,
      "Detector in-band quantum efficiency [e-/photon]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "tput", "", "float", "", "1.0", "", "(0..1]",
      "photometry", M_FUSD, "", 1, T_ADV, 1, "Additional sensor-arm optical intensity throughput" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "use_beam_flux", "", "bool", "0|1", "1", "", "0|1",
      "photometry", M_BOTH, "<wfs>.use_beam_flux", 1, T_BAS, 1,
      "Derive exposure photon budget from upstream 32-bin stellar spectrum (0 = manual flux_ph)" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "flux_ph", "", "float", "", "1e5 (fpcam: 1e6)",
      "ph/fr", ">= 0", "detector", M_BOTH, "<wfs>.flux_ph", 1, T_BAS, 1,
      "Manual photon count per exposure when use_beam_flux == 0" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "ron_e", "ron", "float", "", "0.5 (fpcam: 1.0)",
      "e-", ">= 0", "detector", M_BOTH, "<wfs>.ron_e", 1, T_BAS, 1,
      "Gaussian detector readout noise [e- RMS/pixel]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "dark_e", "", "float", "", "0.0", "e-/fr", ">= 0",
      "detector", M_BOTH, "<wfs>.dark_e", 1, T_ADV, 1,
      "Dark current + sky background [electrons/pixel/frame]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "gain_eadu", "", "float", "", "1.0", "e-/ADU",
      "> 0", "detector", M_BOTH, "<wfs>.gain_eadu", 1, T_ADV, 1,
      "Detector conversion gain [electrons per ADU]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "bias_adu", "", "float", "", "100.0", "ADU",
      ">= 0", "detector", M_BOTH, "<wfs>.bias_adu", 1, T_ADV, 1,
      "Electronic bias pedestal added before quantization [ADU]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "sat_adu", "", "float", "", "65535.0", "ADU",
      "> 0", "detector", M_BOTH, "<wfs>.sat_adu", 1, T_ADV, 1,
      "Detector saturation ceiling [ADU]" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "shutter", "", "bool", "0|1", "1", "", "0|1",
      "detector", M_BOTH, "<wfs>.shutter", 1, T_ADV, 1,
      "Optical shutter state (1 = open sky light, 0 = dark/bias frame only)" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "photon_noise", "", "bool", "0|1", "1", "", "0|1",
      "detector", M_BOTH, "<wfs>.photon_noise", 1, T_BAS, 1,
      "Enable Poisson photon shot noise generation" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "out_dtype", "", "enum", "uint16|float32",
      "uint16", "", "uint16|u16|float32|f32", "detector", M_BOTH, "<wfs>.out_dtype", 1, T_BAS, 1,
      "Output camera SHM pixel datatype" },
    { "sensor", "pyrwfs,shwfs,zwfs,fpcam,fpwfs", "rng_seed", "", "uint", "", "(salted)", "", ">= 1",
      "detector", M_BOTH, "<wfs>.rng_seed", 1, T_ADV, 0,
      "Per-detector deterministic PRNG seed (default prime-salted by node index)" },

    /* [pyrwfs.<id>] — 4-Sided Pyramid Wavefront Sensor */
    { "pyrwfs", "pyrwfs", "ncam", "", "uint", "", "64", "px", "[16..512]", "optics", M_BOTH,
      "pyrwfs.ncam", 0, T_BAS, 1, "Square Pyramid WFS detector frame dimension [pixels]" },
    { "pyrwfs", "pyrwfs", "pad_factor", "", "uint", "", "2", "x", "[2..8]", "optics", M_BOTH,
      "pyrwfs.pad_factor", 0, T_ADV, 1, "Pupil-to-focal FFT zero-padding oversampling factor" },
    { "pyrwfs", "pyrwfs", "lambda_um", "", "float", "", "0.75", "um", "[0.4..2.4]", "optics",
      M_BOTH, "pyrwfs.lambda_um", 0, T_BAS, 1, "Pyramid WFS central sensing wavelength" },
    { "pyrwfs", "pyrwfs", "mod_rad", "mod_rad_ld", "float", "", "3.0", "l/D", "[0..16]", "optics",
      M_BOTH, "pyrwfs.mod_rad", 0, T_BAS, 1, "Circular tip-tilt modulation radius [lambda/D]" },
    { "pyrwfs", "pyrwfs", "mod_npts", "", "int", "", "16", "pts", "[1..64]", "optics", M_BOTH,
      "pyrwfs.mod_npts", 0, T_BAS, 1, "Discrete sample points around circular modulation ring" },
    { "pyrwfs", "pyrwfs", "mod_mode", "mod_subframe_mode", "enum", "full|sync_arc|interleaved",
      "sync_arc", "", "full|sync_arc|interleaved|0..2", "optics", M_BOTH, "pyrwfs.mod_mode", 0,
      T_ADV, 1, "Subframe modulation partitioning across oversampled/navg integration ticks" },
    { "pyrwfs", "pyrwfs", "mod_rev", "", "int", "", "1", "rev", "[1..16]", "optics", M_BOTH,
      "pyrwfs.mod_rev", 0, T_ADV, 1, "Modulation revolutions completed per camera exposure" },
    { "pyrwfs", "pyrwfs", "mod_phase_deg", "", "float", "", "0.0", "deg", "[0..360]", "optics",
      M_BOTH, "pyrwfs.mod_phase_deg", 0, T_ADV, 1, "Initial circular modulation azimuth angle" },
    { "pyrwfs", "pyrwfs", "pup_sep", "", "float", "", "0.50", "ncam", "[0.25..0.75]", "optics",
      M_BOTH, "pyrwfs.pup_sep", 0, T_BAS, 1, "Sub-pupil center separation fraction across ncam" },
    { "pyrwfs", "pyrwfs", "roof_ld", "", "float", "", "0.0", "l/D", "[0..2.0]", "optics", M_BOTH,
      "pyrwfs.roof_ld", 0, T_BAS, 1, "Pyramid glass apex roof defect half-width [lambda/D]" },
    { "pyrwfs", "pyrwfs", "fstop_frac", "", "float", "", "0.45", "", "[0..1.0]", "optics", M_FUSD,
      "", 0, T_ADV, 1, "Focal-plane field stop radius fraction preventing sub-pupil aliasing" },
    { "pyrwfs", "pyrwfs", "chrom_sep", "", "float", "", "0.0", "", "[-0.5..0.5]", "optics", M_FUSD,
      "", 0, T_ADV, 1, "Glass dispersion chromatic pupil-separation coefficient d(sep)/d(ln lam)" },

    /* [shwfs.<id>] — Diffractive Shack-Hartmann Lenslet WFS */
    { "shwfs", "shwfs", "n_sub", "", "uint", "", "16", "sub", "[2..64]", "optics", M_BOTH,
      "shwfs.n_sub", 0, T_BAS, 1, "Linear lenslet subaperture count across pupil" },
    { "shwfs", "shwfs", "pix_per_sub", "", "uint", "", "8", "px", "[4..32]", "optics", M_BOTH,
      "shwfs.pix_per_sub", 0, T_BAS, 1, "Detector pixels per lenslet (ncam = n_sub * pix_per_sub)" },
    { "shwfs", "shwfs", "pad_factor", "", "uint", "", "2", "x", "[1..4]", "optics", M_BOTH,
      "shwfs.pad_factor", 0, T_BAS, 1, "Subaperture Nyquist FFT zero-padding factor" },
    { "shwfs", "shwfs", "lambda_um", "", "float", "", "0.75", "um", "[0.4..2.4]", "optics", M_BOTH,
      "shwfs.lambda_um", 0, T_BAS, 1, "Shack-Hartmann WFS central sensing wavelength" },

    /* [zwfs.<id>] — Babinet Matrix-DFT Zernike Phase-Contrast WFS */
    { "zwfs", "zwfs", "ncam", "", "uint", "", "64", "px", "[16..512]", "optics", M_BOTH,
      "zwfs.ncam", 0, T_BAS, 1, "Square Zernike WFS detector frame size [pixels]" },
    { "zwfs", "zwfs", "n_mft", "", "uint", "", "32", "px", "[16..96]", "optics", M_BOTH,
      "zwfs.n_mft", 0, T_BAS, 1, "Babinet Matrix-DFT focal grid sampling across phase dimple" },
    { "zwfs", "zwfs", "lambda_um", "", "float", "", "0.75", "um", "[0.4..2.4]", "optics", M_BOTH,
      "zwfs.lambda_um", 0, T_BAS, 1, "Zernike WFS central sensing wavelength" },
    { "zwfs", "zwfs", "dimple_ld", "", "float", "", "1.06", "l/D", "[0.4..4.0]", "optics", M_BOTH,
      "zwfs.dimple_ld", 0, T_BAS, 1, "Focal phase-contrast dimple diameter [lambda/D]" },
    { "zwfs", "zwfs", "dimple_rad", "", "float", "", "1.5708", "rad", "[0.1..pi]", "optics",
      M_BOTH, "zwfs.dimple_rad", 0, T_BAS, 1, "Focal dimple phase shift [rad] (pi/2 = 1.5708)" },
    { "zwfs", "zwfs", "dimple_amp", "", "float", "", "1.0", "", "[0..1]", "optics", M_BOTH,
      "zwfs.dimple_amp", 0, T_ADV, 1, "Focal dimple field amplitude transmission (1.0 = pure phase)" },
    { "zwfs", "zwfs", "achrom_dimple", "", "bool", "0|1", "0", "", "0|1", "optics", M_FUSD, "", 0,
      T_ADV, 1, "Keep dimple phase shift wavelength-independent across broadband slices" },

    /* [fpcam.<id>] / [fpwfs.<id>] — Polychromatic Focal-Plane Science / WFS Camera */
    { "fpcam", "fpcam,fpwfs", "out_long", "", "string", "", "", "shm", "", "wiring", M_BOTH,
      "fpcam.out_long", 2, T_BAS, 1, "Optional cumulative long-exposure PSF float32 SHM stream" },
    { "fpcam", "fpcam,fpwfs", "ncam", "", "uint", "", "128", "px", "[16..1024]", "optics", M_BOTH,
      "fpcam.ncam", 0, T_BAS, 1, "Square focal-plane camera detector size [pixels]" },
    { "fpcam", "fpcam,fpwfs", "prop_mode", "", "enum", "mft|fft", "mft", "", "mft|fft|0|1",
      "optics", M_BOTH, "fpcam.prop_mode", 0, T_BAS, 1,
      "Propagator: Soummer Matrix-DFT ('mft', arbitrary pixscale) or zero-padded FFT ('fft')" },
    { "fpcam", "fpcam,fpwfs", "pad_factor", "", "uint", "", "2", "x", "[2..8]", "optics", M_BOTH,
      "fpcam.pad_factor", 0, T_ADV, 1, "Zero-padding factor when prop_mode == 'fft'" },
    { "fpcam", "fpcam,fpwfs", "lambda_um", "", "float", "", "1.65", "um", "[0.4..2.4]", "optics",
      M_BOTH, "fpcam.lambda_um", 0, T_BAS, 1, "Focal-plane filter central wavelength" },
    { "fpcam", "fpcam,fpwfs", "pixscale_ld", "pixscale", "float", "", "3.0", "px/(l/D)",
      "[0.5..16]", "optics", M_BOTH, "fpcam.pixscale_ld", 0, T_BAS, 1,
      "Detector sampling [pixels per lambda/D] in MFT mode" },
    { "fpcam", "fpcam,fpwfs", "is_wfs", "", "bool", "0|1", "0 (fpwfs: 1)", "", "0|1", "runtime",
      M_FUSD, "", 0, T_ADV, 1,
      "Treat focal-plane camera as an active WFS (kept active during calib_fast & skip_fpcam)" },
};

/* ── Authoritative cacaosim CLI Option Table ──────────────────────────────── */

static const cacaosim_cli_opt_t g_cli_opts[] = {
    /* cacaosim-bench */
    { "cacaosim-bench", "-c, --config", "<file.bench>", "", "workflow",
      "Declarative multi-stage optical bench (.bench) configuration file" },
    { "cacaosim-bench", "--mode", "<fused|modular>", "fused", "deploy",
      "Execution topology: fused zero-hop htsim or modular multi-process FPS units" },
    { "cacaosim-bench", "--nwf", "<int>", "128", "plan",
      "Flat-mode pupil wavefront grid size [pixels]" },
    { "cacaosim-bench", "--nact", "<int>", "20", "plan",
      "Flat-mode linear DM actuator count" },
    { "cacaosim-bench", "--wfs-ncam", "<int>", "64", "plan",
      "Flat-mode WFS camera dimension [pixels]" },
    { "cacaosim-bench", "--fp-ncam", "<int>", "128", "plan",
      "Flat-mode focal-plane science camera dimension [pixels]" },
    { "cacaosim-bench", "--rate", "<float>", "2000", "deploy",
      "Override hardware master frame rate [Hz]" },
    { "cacaosim-bench", "--oversample", "<int>", "0", "deploy",
      "Override temporal oversampling factor K (0 = bench default)" },
    { "cacaosim-bench", "--mod-rad", "<float>", "3.0", "plan",
      "Flat-mode Pyramid modulation radius [lambda/D]" },
    { "cacaosim-bench", "--backend", "<0|1|2>", "0", "deploy",
      "Compute backend override: 0 = auto, 1 = CPU, 2 = CUDA" },
    { "cacaosim-bench", "--atmturb", "", "OFF", "deploy",
      "Co-launch milk-fpsexec-atmturb-mkwfs live phase-screen stream" },
    { "cacaosim-bench", "--seeing", "<float>", "0.7", "deploy",
      "Atmospheric seeing [arcsec] when --atmturb is active" },
    { "cacaosim-bench", "--restart", "", "OFF", "deploy",
      "Force unit restart and reset live FPFLAG_WRITERUN parameters" },
    { "cacaosim-bench", "--workdir", "<dir>", "", "deploy",
      "Override fpspipe workspace root directory" },
    { "cacaosim-bench", "--no-view", "", "OFF", "display",
      "Suppress automatic ANSI schematic rendering on start/status" },

    /* cacaosim-bench-view */
    { "cacaosim-bench-view", "-c, --config", "<file.bench>", "", "workflow",
      "Explicit .bench configuration path (also accepted as positional arg)" },
    { "cacaosim-bench-view", "-i, --interactive", "", "OFF", "workflow",
      "Launch interactive milk-CTRL-style configurator (cacaosim-CTRL)" },
    { "cacaosim-bench-view", "-g, --gui", "", "OFF", "workflow",
      "Alias for -i / --interactive TUI configurator" },
    { "cacaosim-bench-view", "-w, --watch", "", "OFF", "workflow",
      "Auto-reload diagram live whenever <config.bench> or MILK_SHM_DIR changes" },
    { "cacaosim-bench-view", "--init", "<file.bench>", "", "workflow",
      "Write a commented starter 2-stage .bench template to disk" },
    { "cacaosim-bench-view", "-d, --deploy, --run", "", "OFF", "workflow",
      "Validate, deploy & start via cacaosim-bench, then visualize" },
    { "cacaosim-bench-view", "--list-options", "", "OFF", "options",
      "Print complete .bench, FPS compute unit, and CLI option reference and exit" },
    { "cacaosim-bench-view", "--section", "<name>", "", "options",
      "Filter --list-options to a specific .bench section (e.g. bench, dm, pyrwfs)" },
    { "cacaosim-bench-view", "--search", "<query>", "", "options",
      "Filter --list-options by case-insensitive substring across keys and descriptions" },
    { "cacaosim-bench-view", "--fps", "<unit|all>", "", "options",
      "Filter --list-options to one or all introspected MILK_RUNNABLE FPS units" },
    { "cacaosim-bench-view", "--live", "", "OFF", "options",
      "With --list-options, list only FPFLAG_WRITERUN live-tunable FPS parameters" },
    { "cacaosim-bench-view", "--format", "<table|markdown|json>", "table", "options",
      "Output format for --list-options (aligned table, GFM Markdown, or JSON)" },
    { "cacaosim-bench-view", "--output", "<file>", "", "options",
      "Write --list-options output to <file> instead of stdout" },
    { "cacaosim-bench-view", "-p, --pretty, --color", "", "OFF", "display",
      "Force 24-bit TrueColor ANSI output even when piped" },
    { "cacaosim-bench-view", "-t, --text, --no-color", "", "OFF", "display",
      "Plain monochrome text output (no ANSI color escapes)" },
    { "cacaosim-bench-view", "--ascii", "", "OFF", "display",
      "Pure 7-bit ASCII box-drawing and spectrum sparklines" },
    { "cacaosim-bench-view", "--theme", "<name>", "dark", "display",
      "Palette: dark, night, accessible, light, or nordic" },
    { "cacaosim-bench-view", "--compact", "", "OFF", "display",
      "Show overview + optical schematic only (omit hardware tables)" },
    { "cacaosim-bench-view", "--no-shm", "", "OFF", "display",
      "Do not probe MILK_SHM_DIR for live .im.shm status" },
    { "cacaosim-bench-view", "--no-footer", "", "OFF", "display",
      "Omit CLI workflow footer line" },
    { "cacaosim-bench-view", "-W, --width", "<cols>", "auto", "display",
      "Override schematic panel width in columns [76..124]" },
    { "cacaosim-bench-view", "--mode", "<fused|modular>", "fused", "deploy",
      "Deployment engine mode (also filters --list-options when passed with it)" },
    { "cacaosim-bench-view", "--atmturb", "", "OFF", "deploy",
      "Co-launch milk-fpsexec-atmturb-mkwfs turbulence stream" },
    { "cacaosim-bench-view", "--seeing", "<arcsec>", "0.7", "deploy",
      "Atmospheric seeing [arcsec] when --atmturb is enabled" },
    { "cacaosim-bench-view", "--emit-plan", "<dir>", "", "plan",
      "Compile .bench (or --flat) into fpspipe plan files in <dir> and exit" },
    { "cacaosim-bench-view", "--flat", "", "OFF", "plan",
      "Emit single-stage flat plan without a .bench file (with --emit-plan)" },
    { "cacaosim-bench-view", "--staged-bench", "<path>", "", "plan",
      "Write resolved .bench file for htsim staging (with --emit-plan)" },
    { "cacaosim-bench-view", "--active-bench-arg", "<str>", "csim_active.bench", "plan",
      "Override .bench_file argument recorded in params.batch (with --emit-plan)" },
    { "cacaosim-bench-view", "--rate", "<hz>", "0", "plan",
      "Override master rate [Hz] in emitted plan" },
    { "cacaosim-bench-view", "--oversample", "<int>", "0", "plan",
      "Override temporal oversampling factor K in emitted plan" },
    { "cacaosim-bench-view", "--backend", "<0|1|2>", "-1", "plan",
      "Override backend (0=auto, 1=CPU, 2=CUDA) in emitted plan" },
    { "cacaosim-bench-view", "--nwf", "<int>", "128", "plan",
      "Flat-mode pupil grid size [px] (with --emit-plan --flat)" },
    { "cacaosim-bench-view", "--nact", "<int>", "20", "plan",
      "Flat-mode DM actuator count (with --emit-plan --flat)" },
    { "cacaosim-bench-view", "--wfs-ncam", "<int>", "64", "plan",
      "Flat-mode WFS camera dimension [px] (with --emit-plan --flat)" },
    { "cacaosim-bench-view", "--fp-ncam", "<int>", "128", "plan",
      "Flat-mode science camera dimension [px] (with --emit-plan --flat)" },
    { "cacaosim-bench-view", "--mod-rad", "<float>", "3.0", "plan",
      "Flat-mode PyWFS modulation radius [l/D] (with --emit-plan --flat)" },

    /* cacaosim-CTRL */
    { "cacaosim-CTRL", "-c, --config", "<file.bench>", "", "workflow",
      "Load or create .bench file (defaults to active SHM bench or preset)" },
    { "cacaosim-CTRL", "-s, --status", "", "OFF", "workflow",
      "Start directly on F5:STATUS live MILK_SHM_DIR monitor tab" },
    { "cacaosim-CTRL", "--schematic", "", "OFF", "workflow",
      "Start directly on F3:SCHEMATIC full-screen optical diagram tab" },
    { "cacaosim-CTRL", "--mode", "<fused|modular>", "fused", "deploy",
      "Default deployment engine mode inside TUI" },
    { "cacaosim-CTRL", "--atmturb", "", "OFF", "deploy",
      "Enable milk-fpsexec-atmturb-mkwfs co-launch on deploy" },
    { "cacaosim-CTRL", "--seeing", "<arcsec>", "0.7", "deploy",
      "Atmospheric seeing [arcsec] when --atmturb is active" },
    { "cacaosim-CTRL", "--theme", "<name>", "dark", "display",
      "milk palette: dark, night, accessible, light, or nordic" },
    { "cacaosim-CTRL", "--ascii", "", "OFF", "display",
      "Pure 7-bit ASCII borders, arrows, and spectrum bars" },
    { "cacaosim-CTRL", "-p, --pretty", "", "OFF", "display",
      "Force 24-bit TrueColor ANSI output" },
    { "cacaosim-CTRL", "-t, --text, --no-color", "", "OFF", "display",
      "Monochrome terminal output (no color escapes)" },
};

/* ── Internal Helpers ─────────────────────────────────────────────────────── */

static int str_contains_ci(const char *hay, const char *needle)
{
    if (needle == NULL || needle[0] == '\0')
    {
        return 1;
    }
    if (hay == NULL || hay[0] == '\0')
    {
        return 0;
    }
    size_t nlen = strlen(needle);
    for (const char *p = hay; *p != '\0'; p++)
    {
        size_t i = 0u;
        while (i < nlen && p[i] != '\0' &&
               tolower((unsigned char) p[i]) == tolower((unsigned char) needle[i]))
        {
            i++;
        }
        if (i == nlen)
        {
            return 1;
        }
    }
    return 0;
}

static int csv_has_token_ci(const char *csv, const char *tok)
{
    if (csv == NULL || tok == NULL || tok[0] == '\0')
    {
        return 0;
    }
    size_t      tlen = strlen(tok);
    const char *p    = csv;
    while (*p != '\0')
    {
        const char *comma = strchr(p, ',');
        size_t      len   = (comma != NULL) ? (size_t) (comma - p) : strlen(p);
        if (len == tlen && strncasecmp(p, tok, len) == 0)
        {
            return 1;
        }
        p = (comma != NULL) ? (comma + 1) : (p + len);
    }
    return 0;
}

static const char *mode_label(cacaosim_opt_mode_t m)
{
    if (m == CACAOSIM_OPT_MODE_FUSED)
    {
        return "fused-only";
    }
    if (m == CACAOSIM_OPT_MODE_MODULAR)
    {
        return "modular-only";
    }
    return "fused+modular";
}

static const char *fptype_label(uint64_t type)
{
    switch (type)
    {
    case FPTYPE_INT32:
        return "int32";
    case FPTYPE_UINT32:
        return "uint32";
    case FPTYPE_INT64:
        return "int64";
    case FPTYPE_UINT64:
        return "uint64";
    case FPTYPE_FLOAT32:
        return "float32";
    case FPTYPE_FLOAT64:
        return "float64";
    case FPTYPE_ONOFF:
        return "onoff";
    case FPTYPE_STREAMNAME:
        return "stream";
    case FPTYPE_FILENAME:
        return "filename";
    case FPTYPE_FITSFILENAME:
        return "fits";
    case FPTYPE_DIRNAME:
        return "dirname";
    case FPTYPE_FPSNAME:
        return "fps";
    case FPTYPE_STRING:
    case FPTYPE_STRING_NOT_STREAM:
        return "string";
    default:
        return "other";
    }
}

static const char *fps_mutability(uint64_t fpflag)
{
    if ((fpflag & FPFLAG_WRITE) == 0u)
    {
        return "output";
    }
    if ((fpflag & FPFLAG_WRITERUN) != 0u)
    {
        return "live";
    }
    return "init";
}

static void format_binding_default(const FPS_CLI_BINDING *b, char *buf, size_t cap)
{
    if (buf == NULL || cap == 0u)
    {
        return;
    }
    buf[0] = '\0';
    if (b == NULL || b->ptr == NULL)
    {
        return;
    }
    switch (b->type)
    {
    case FPTYPE_INT32:
        snprintf(buf, cap, "%" PRId32, *(const int32_t *) b->ptr);
        break;
    case FPTYPE_UINT32:
        snprintf(buf, cap, "%" PRIu32, *(const uint32_t *) b->ptr);
        break;
    case FPTYPE_INT64:
        snprintf(buf, cap, "%" PRId64, *(const int64_t *) b->ptr);
        break;
    case FPTYPE_UINT64:
        snprintf(buf, cap, "%" PRIu64, *(const uint64_t *) b->ptr);
        break;
    case FPTYPE_FLOAT32:
        snprintf(buf, cap, "%.6g", (double) (*(const float *) b->ptr));
        break;
    case FPTYPE_FLOAT64:
        snprintf(buf, cap, "%.6g", *(const double *) b->ptr);
        break;
    case FPTYPE_ONOFF:
        snprintf(buf, cap, "%s", (*(const int64_t *) b->ptr != 0) ? "ON" : "OFF");
        break;
    default:
        if (FPTYPE_IS_STRING(b->type))
        {
            snprintf(buf, cap, "%s", (const char *) b->ptr);
        }
        break;
    }
}

static const char *unit_short_name(const MILK_RUNNABLE *r)
{
    if (r == NULL || r->app.cmdkey == NULL)
    {
        return "unknown";
    }
    const char *dot = strchr(r->app.cmdkey, '.');
    return (dot != NULL && dot[1] != '\0') ? (dot + 1) : r->app.cmdkey;
}

static int bench_opt_matches(
    const cacaosim_bench_opt_t     *o,
    const cacaosim_schema_filter_t *flt)
{
    if (flt == NULL)
    {
        return 1;
    }
    if (flt->section != NULL && flt->section[0] != '\0')
    {
        if (strcasecmp(o->section, flt->section) != 0 &&
            !csv_has_token_ci(o->applies_to, flt->section))
        {
            return 0;
        }
    }
    if (flt->mode != NULL && flt->mode[0] != '\0')
    {
        if (strcasecmp(flt->mode, "fused-only") == 0)
        {
            if (o->mode != CACAOSIM_OPT_MODE_FUSED)
            {
                return 0;
            }
        }
        else if (strcasecmp(flt->mode, "modular") == 0)
        {
            if (o->mode == CACAOSIM_OPT_MODE_FUSED)
            {
                return 0;
            }
        }
        else if (strcasecmp(flt->mode, "fused") == 0)
        {
            if (o->mode == CACAOSIM_OPT_MODE_MODULAR)
            {
                return 0;
            }
        }
    }
    if (flt->search != NULL && flt->search[0] != '\0')
    {
        if (!str_contains_ci(o->key, flt->search) && !str_contains_ci(o->aliases, flt->search) &&
            !str_contains_ci(o->section, flt->search) &&
            !str_contains_ci(o->applies_to, flt->search) &&
            !str_contains_ci(o->category, flt->search) &&
            !str_contains_ci(o->fps_param, flt->search) &&
            !str_contains_ci(o->description, flt->search))
        {
            return 0;
        }
    }
    return 1;
}

static int fps_unit_matches(const MILK_RUNNABLE *r, const cacaosim_schema_filter_t *flt)
{
    if (flt == NULL || flt->fps_unit == NULL || flt->fps_unit[0] == '\0' ||
        strcasecmp(flt->fps_unit, "all") == 0)
    {
        return 1;
    }
    const char *sname = unit_short_name(r);
    return (strcasecmp(sname, flt->fps_unit) == 0 ||
            (r->app.cmdkey != NULL && strcasecmp(r->app.cmdkey, flt->fps_unit) == 0) ||
            (r->app.fps_name != NULL && strcasecmp(r->app.fps_name, flt->fps_unit) == 0));
}

static int fps_binding_matches(
    const MILK_RUNNABLE            *r,
    const FPS_CLI_BINDING          *b,
    const cacaosim_schema_filter_t *flt)
{
    if (flt == NULL)
    {
        return 1;
    }
    if (flt->live_only && (b->fpflag & FPFLAG_WRITERUN) == 0u)
    {
        return 0;
    }
    if (flt->search != NULL && flt->search[0] != '\0')
    {
        const char *sname = unit_short_name(r);
        if (!str_contains_ci(b->fpskeyword, flt->search) &&
            !str_contains_ci(b->descr, flt->search) && !str_contains_ci(sname, flt->search) &&
            !str_contains_ci(r->app.cmdkey, flt->search))
        {
            return 0;
        }
    }
    return 1;
}

static int cli_opt_matches(const cacaosim_cli_opt_t *c, const cacaosim_schema_filter_t *flt)
{
    if (flt == NULL || flt->search == NULL || flt->search[0] == '\0')
    {
        return 1;
    }
    return str_contains_ci(c->tool, flt->search) || str_contains_ci(c->flag, flt->search) ||
           str_contains_ci(c->group, flt->search) || str_contains_ci(c->description, flt->search);
}

static void fprint_json_str(FILE *fp, const char *s)
{
    fputc('"', fp);
    if (s != NULL)
    {
        for (const unsigned char *p = (const unsigned char *) s; *p != '\0'; p++)
        {
            if (*p == '"')
            {
                fputs("\\\"", fp);
            }
            else if (*p == '\\')
            {
                fputs("\\\\", fp);
            }
            else if (*p == '\n')
            {
                fputs("\\n", fp);
            }
            else if (*p == '\r')
            {
                fputs("\\r", fp);
            }
            else if (*p == '\t')
            {
                fputs("\\t", fp);
            }
            else if (*p < 0x20u)
            {
                fprintf(fp, "\\u%04x", (unsigned int) *p);
            }
            else
            {
                fputc((int) *p, fp);
            }
        }
    }
    fputc('"', fp);
}

static void fprint_md_cell(FILE *fp, const char *s)
{
    if (s == NULL)
    {
        return;
    }
    for (const char *p = s; *p != '\0'; p++)
    {
        if (*p == '|')
        {
            fputs("\\|", fp);
        }
        else if (*p == '\n' || *p == '\r')
        {
            fputc(' ', fp);
        }
        else
        {
            fputc(*p, fp);
        }
    }
}

/* ── Public Query API ─────────────────────────────────────────────────────── */

size_t cacaosim_schema_bench_count(void)
{
    return sizeof(g_bench_opts) / sizeof(g_bench_opts[0]);
}

const cacaosim_bench_opt_t *cacaosim_schema_bench_get(size_t idx)
{
    if (idx >= cacaosim_schema_bench_count())
    {
        return NULL;
    }
    return &g_bench_opts[idx];
}

const cacaosim_bench_opt_t *cacaosim_schema_find_bench_opt(
    const char *section,
    const char *key_or_alias)
{
    if (key_or_alias == NULL || key_or_alias[0] == '\0')
    {
        return NULL;
    }
    size_t n = cacaosim_schema_bench_count();
    /* Prefer exact section match before falling back to shared 'sensor'/'common' groups */
    for (int pass = 0; pass < 2; pass++)
    {
        for (size_t i = 0u; i < n; i++)
        {
            const cacaosim_bench_opt_t *o = &g_bench_opts[i];
            if (section != NULL && section[0] != '\0')
            {
                if (pass == 0 && strcasecmp(o->section, section) != 0)
                {
                    continue;
                }
                if (pass == 1 && !csv_has_token_ci(o->applies_to, section))
                {
                    continue;
                }
            }
            else if (pass == 1)
            {
                break;
            }
            if (strcasecmp(o->key, key_or_alias) == 0 ||
                csv_has_token_ci(o->aliases, key_or_alias))
            {
                return o;
            }
        }
    }
    return NULL;
}

size_t cacaosim_schema_cli_count(void)
{
    return sizeof(g_cli_opts) / sizeof(g_cli_opts[0]);
}

const cacaosim_cli_opt_t *cacaosim_schema_cli_get(size_t idx)
{
    if (idx >= cacaosim_schema_cli_count())
    {
        return NULL;
    }
    return &g_cli_opts[idx];
}

void cacaosim_schema_filter_defaults(cacaosim_schema_filter_t *flt)
{
    if (flt == NULL)
    {
        return;
    }
    memset(flt, 0, sizeof(*flt));
    flt->include_bench = 1;
    flt->include_fps   = 1;
    flt->include_cli   = 1;
}

/* ── Aligned Terminal Table Exporter ──────────────────────────────────────── */

int cacaosim_schema_render_table(
    const cacaosim_schema_filter_t *flt,
    int                             color,
    FILE                           *fp)
{
    cacaosim_schema_filter_t def;
    if (flt == NULL)
    {
        cacaosim_schema_filter_defaults(&def);
        flt = &def;
    }
    if (fp == NULL)
    {
        fp = stdout;
    }

    const char *c_hdr = color ? "\033[1;36m" : "";
    const char *c_key = color ? "\033[1;33m" : "";
    const char *c_dim = color ? "\033[2m" : "";
    const char *c_ok  = color ? "\033[32m" : "";
    const char *c_wrn = color ? "\033[35m" : "";
    const char *c_rst = color ? "\033[0m" : "";

    size_t total_shown = 0u;

    if (flt->include_bench)
    {
        size_t n_bench = cacaosim_schema_bench_count();
        size_t matched = 0u;
        for (size_t i = 0u; i < n_bench; i++)
        {
            if (bench_opt_matches(&g_bench_opts[i], flt))
            {
                matched++;
            }
        }
        if (matched > 0u)
        {
            fprintf(fp, "%s=== Declarative .bench Options (%zu) ===%s\n", c_hdr, matched, c_rst);
            fprintf(fp, "%s%-9s %-18s %-7s %-16s %-8s %-13s %s%s\n", c_dim, "SECTION", "KEY",
                    "TYPE", "DEFAULT", "UNIT", "MODE", "DESCRIPTION", c_rst);
            for (size_t i = 0u; i < n_bench; i++)
            {
                const cacaosim_bench_opt_t *o = &g_bench_opts[i];
                if (!bench_opt_matches(o, flt))
                {
                    continue;
                }
                const char *mc = (o->mode == M_FUSD) ? c_wrn : c_ok;
                fprintf(fp, "%-9.9s %s%-18.18s%s %-7.7s %-16.16s %-8.8s %s%-13.13s%s %s",
                        o->section, c_key, o->key, c_rst, o->type,
                        o->default_str[0] ? o->default_str : "\"\"", o->unit, mc,
                        mode_label(o->mode), c_rst, o->description);
                if (o->aliases[0] != '\0')
                {
                    fprintf(fp, " %s[aliases: %s]%s", c_dim, o->aliases, c_rst);
                }
                fputc('\n', fp);
            }
            fputc('\n', fp);
            total_shown += matched;
        }
    }

    if (flt->include_fps)
    {
        size_t                      n_units = 0u;
        const MILK_RUNNABLE *const *units   = cacaosim_list_runnables(&n_units);
        for (size_t u = 0u; u < n_units; u++)
        {
            const MILK_RUNNABLE *r = units[u];
            if (!fps_unit_matches(r, flt))
            {
                continue;
            }
            size_t umatch = 0u;
            for (int bi = 0; bi < r->nb_bindings; bi++)
            {
                if (fps_binding_matches(r, &r->bindings[bi], flt))
                {
                    umatch++;
                }
            }
            if (umatch == 0u)
            {
                continue;
            }
            fprintf(fp, "%s=== FPS Unit: %s (%s, %zu params) ===%s\n", c_hdr,
                    r->app.cmdkey ? r->app.cmdkey : unit_short_name(r),
                    r->app.fps_name ? r->app.fps_name : "", umatch, c_rst);
            fprintf(fp, "%s%-20s %-8s %-16s %-7s %s%s\n", c_dim, "KEYWORD", "TYPE", "DEFAULT",
                    "ACCESS", "DESCRIPTION", c_rst);
            for (int bi = 0; bi < r->nb_bindings; bi++)
            {
                const FPS_CLI_BINDING *b = &r->bindings[bi];
                if (!fps_binding_matches(r, b, flt))
                {
                    continue;
                }
                char dval[64];
                format_binding_default(b, dval, sizeof(dval));
                const char *mut = fps_mutability(b->fpflag);
                const char *mc  = (strcmp(mut, "live") == 0) ? c_ok : c_dim;
                fprintf(fp, "%s%-20.20s%s %-8.8s %-16.16s %s%-7.7s%s %s\n", c_key,
                        b->fpskeyword ? b->fpskeyword : "", c_rst, fptype_label(b->type),
                        dval[0] ? dval : "\"\"", mc, mut, c_rst, b->descr ? b->descr : "");
            }
            fputc('\n', fp);
            total_shown += umatch;
        }
    }

    if (flt->include_cli)
    {
        size_t n_cli   = cacaosim_schema_cli_count();
        size_t matched = 0u;
        for (size_t i = 0u; i < n_cli; i++)
        {
            if (cli_opt_matches(&g_cli_opts[i], flt))
            {
                matched++;
            }
        }
        if (matched > 0u)
        {
            fprintf(fp, "%s=== cacaosim CLI Options (%zu) ===%s\n", c_hdr, matched, c_rst);
            fprintf(fp, "%s%-20s %-22s %-18s %-10s %s%s\n", c_dim, "TOOL", "FLAG", "ARG",
                    "DEFAULT", "DESCRIPTION", c_rst);
            for (size_t i = 0u; i < n_cli; i++)
            {
                const cacaosim_cli_opt_t *c = &g_cli_opts[i];
                if (!cli_opt_matches(c, flt))
                {
                    continue;
                }
                fprintf(fp, "%-20.20s %s%-22.22s%s %-18.18s %-10.10s %s\n", c->tool, c_key,
                        c->flag, c_rst, c->arg, c->default_str, c->description);
            }
            fputc('\n', fp);
            total_shown += matched;
        }
    }

    if (total_shown == 0u)
    {
        fprintf(fp, "No cacaosim options matched the specified filter.\n");
    }
    return ferror(fp) ? -1 : 0;
}

/* ── GitHub-Flavored Markdown Exporter (docs/OPTIONS.md) ──────────────────── */

static const char *const g_bench_sec_order[] = {
    "bench", "common", "dm", "splitter", "coro", "wfe", "sensor", "pyrwfs", "shwfs", "zwfs",
    "fpcam"
};

static const char *const g_bench_sec_titles[] = {
    "`[bench]` — Global Telescope, Pupil, Star & Master Clock",
    "Common Node Keys — Shared Optical Wiring & 2D Registration",
    "`[dm.<id>]` / `[ttm.<id>]` — Deformable Mirror & Tip-Tilt Mirror",
    "`[splitter.<id>]` — Gray & Dichroic Beamsplitter",
    "`[coro.<id>]` — Coronagraphic Masks, Lyot Stop & LOWFS Pickoff",
    "`[wfe.<id>]` / `[static_wfe.<id>]` — Inline Static Wavefront Error (`OPD [m]`)",
    "Shared Sensor Keys — Cadence, Timing & Detector Readout (`pyrwfs`, `shwfs`, `zwfs`, `fpcam`)",
    "`[pyrwfs.<id>]` — 4-Sided Pyramid Wavefront Sensor",
    "`[shwfs.<id>]` — Diffractive Shack-Hartmann Lenslet WFS",
    "`[zwfs.<id>]` — Babinet Matrix-DFT Zernike Phase-Contrast WFS",
    "`[fpcam.<id>]` / `[fpwfs.<id>]` — Focal-Plane Science & WFS Camera"
};

int cacaosim_schema_render_markdown(const cacaosim_schema_filter_t *flt, FILE *fp)
{
    cacaosim_schema_filter_t def;
    if (flt == NULL)
    {
        cacaosim_schema_filter_defaults(&def);
        flt = &def;
    }
    if (fp == NULL)
    {
        fp = stdout;
    }

    size_t                      n_units    = 0u;
    const MILK_RUNNABLE *const *units      = cacaosim_list_runnables(&n_units);
    size_t                      n_fps_tot  = 0u;
    for (size_t u = 0u; u < n_units; u++)
    {
        n_fps_tot += (size_t) units[u]->nb_bindings;
    }

    fprintf(fp,
            "# cacaosim Complete Options & Parameters Reference\n\n"
            "> **Auto-generated from `cacaosim_bench_schema.c` and runtime `MILK_RUNNABLE` "
            "introspection.**\n"
            "> Regenerate at any time with: "
            "`cacaosim-bench options --format markdown --output docs/OPTIONS.md`\n\n"
            "Summary: **%zu** `.bench` option definitions · **%zu** FPS compute units "
            "(**%zu** FPS parameters) · **%zu** CLI flags.\n\n",
            cacaosim_schema_bench_count(), n_units, n_fps_tot, cacaosim_schema_cli_count());

    if (flt->include_bench)
    {
        fprintf(fp, "## 1. Declarative `.bench` Configuration Options\n\n");
        size_t n_sec = sizeof(g_bench_sec_order) / sizeof(g_bench_sec_order[0]);
        for (size_t s = 0u; s < n_sec; s++)
        {
            const char *sec = g_bench_sec_order[s];
            size_t      cnt = 0u;
            for (size_t i = 0u; i < cacaosim_schema_bench_count(); i++)
            {
                if (strcmp(g_bench_opts[i].section, sec) == 0 &&
                    bench_opt_matches(&g_bench_opts[i], flt))
                {
                    cnt++;
                }
            }
            if (cnt == 0u)
            {
                continue;
            }
            fprintf(fp, "### %s\n\n", g_bench_sec_titles[s]);
            fprintf(fp,
                    "| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |\n"
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n");
            for (size_t i = 0u; i < cacaosim_schema_bench_count(); i++)
            {
                const cacaosim_bench_opt_t *o = &g_bench_opts[i];
                if (strcmp(o->section, sec) != 0 || !bench_opt_matches(o, flt))
                {
                    continue;
                }
                fprintf(fp, "| `%s` | ", o->key);
                if (o->aliases[0] != '\0')
                {
                    fprintf(fp, "`%s`", o->aliases);
                }
                else
                {
                    fputs("—", fp);
                }
                fprintf(fp, " | `%s` | ", o->type);
                if (o->default_str[0] != '\0')
                {
                    fputc('`', fp);
                    fprint_md_cell(fp, o->default_str);
                    fputc('`', fp);
                }
                else
                {
                    fputs("`\"\"`", fp);
                }
                fprintf(fp, " | %s | `%s` | %s | ", o->unit[0] ? o->unit : "—",
                        mode_label(o->mode), (o->tier == T_ADV) ? "Advanced" : "Basic");
                fprint_md_cell(fp, o->description);
                if (o->enum_values[0] != '\0')
                {
                    fputs(" Values: `", fp);
                    fprint_md_cell(fp, o->enum_values);
                    fputc('`', fp);
                }
                fputs(" |\n", fp);
            }
            fputc('\n', fp);
        }
    }

    if (flt->include_fps)
    {
        fprintf(fp, "## 2. Standalone FPS Compute Unit Parameters (`libmilkcacaosim.so`)\n\n");
        for (size_t u = 0u; u < n_units; u++)
        {
            const MILK_RUNNABLE *r = units[u];
            if (!fps_unit_matches(r, flt))
            {
                continue;
            }
            size_t umatch = 0u;
            for (int bi = 0; bi < r->nb_bindings; bi++)
            {
                if (fps_binding_matches(r, &r->bindings[bi], flt))
                {
                    umatch++;
                }
            }
            if (umatch == 0u)
            {
                continue;
            }
            const char *sname = unit_short_name(r);
            fprintf(fp, "### `%s` (`milk-fpsexec-cacaosim-%s` · FPS `%s`)\n\n",
                    r->app.cmdkey ? r->app.cmdkey : sname, sname,
                    r->app.fps_name ? r->app.fps_name : "");
            if (r->app.description != NULL && r->app.description[0] != '\0')
            {
                fprintf(fp, "%s\n\n", r->app.description);
            }
            fprintf(fp,
                    "| Keyword | Type | Default | Access | Description |\n"
                    "| :--- | :--- | :--- | :--- | :--- |\n");
            for (int bi = 0; bi < r->nb_bindings; bi++)
            {
                const FPS_CLI_BINDING *b = &r->bindings[bi];
                if (!fps_binding_matches(r, b, flt))
                {
                    continue;
                }
                char dval[64];
                format_binding_default(b, dval, sizeof(dval));
                const char *mut = fps_mutability(b->fpflag);
                fprintf(fp, "| `%s` | `%s` | `%s` | `%s` | ",
                        b->fpskeyword ? b->fpskeyword : "", fptype_label(b->type),
                        dval[0] ? dval : "\"\"", mut);
                fprint_md_cell(fp, b->descr ? b->descr : "");
                fputs(" |\n", fp);
            }
            fputc('\n', fp);
        }
    }

    if (flt->include_cli)
    {
        size_t matched = 0u;
        for (size_t i = 0u; i < cacaosim_schema_cli_count(); i++)
        {
            if (cli_opt_matches(&g_cli_opts[i], flt))
            {
                matched++;
            }
        }
        if (matched > 0u)
        {
            fprintf(fp, "## 3. Command-Line Tools (`cacaosim-bench`, `cacaosim-bench-view`, "
                        "`cacaosim-CTRL`)\n\n");
            fprintf(fp,
                    "| Tool | Flag | Argument | Default | Group | Description |\n"
                    "| :--- | :--- | :--- | :--- | :--- | :--- |\n");
            for (size_t i = 0u; i < cacaosim_schema_cli_count(); i++)
            {
                const cacaosim_cli_opt_t *c = &g_cli_opts[i];
                if (!cli_opt_matches(c, flt))
                {
                    continue;
                }
                fprintf(fp, "| `%s` | `%s` | ", c->tool, c->flag);
                if (c->arg[0] != '\0')
                {
                    fprintf(fp, "`%s`", c->arg);
                }
                else
                {
                    fputs("—", fp);
                }
                fprintf(fp, " | %s | `%s` | ", c->default_str[0] ? c->default_str : "—", c->group);
                fprint_md_cell(fp, c->description);
                fputs(" |\n", fp);
            }
            fputc('\n', fp);
        }
    }

    return ferror(fp) ? -1 : 0;
}

/* ── Structured JSON Exporter (share/cacaosim/options_schema.json) ────────── */

int cacaosim_schema_render_json(const cacaosim_schema_filter_t *flt, FILE *fp)
{
    cacaosim_schema_filter_t def;
    if (flt == NULL)
    {
        cacaosim_schema_filter_defaults(&def);
        flt = &def;
    }
    if (fp == NULL)
    {
        fp = stdout;
    }

    fputs("{\n  \"schema_version\": 1,\n  \"source\": \"cacaosim\",\n", fp);

    /* 1. .bench options */
    fputs("  \"bench_options\": [", fp);
    int first = 1;
    if (flt->include_bench)
    {
        for (size_t i = 0u; i < cacaosim_schema_bench_count(); i++)
        {
            const cacaosim_bench_opt_t *o = &g_bench_opts[i];
            if (!bench_opt_matches(o, flt))
            {
                continue;
            }
            fputs(first ? "\n" : ",\n", fp);
            first = 0;
            fputs("    {", fp);
            fputs("\"section\":", fp);
            fprint_json_str(fp, o->section);
            fputs(",\"applies_to\":", fp);
            fprint_json_str(fp, o->applies_to);
            fputs(",\"key\":", fp);
            fprint_json_str(fp, o->key);
            fputs(",\"aliases\":", fp);
            fprint_json_str(fp, o->aliases);
            fputs(",\"type\":", fp);
            fprint_json_str(fp, o->type);
            fputs(",\"enum_values\":", fp);
            fprint_json_str(fp, o->enum_values);
            fputs(",\"default\":", fp);
            fprint_json_str(fp, o->default_str);
            fputs(",\"unit\":", fp);
            fprint_json_str(fp, o->unit);
            fputs(",\"range\":", fp);
            fprint_json_str(fp, o->range_str);
            fputs(",\"category\":", fp);
            fprint_json_str(fp, o->category);
            fputs(",\"mode\":", fp);
            fprint_json_str(fp, mode_label(o->mode));
            fputs(",\"fps_param\":", fp);
            fprint_json_str(fp, o->fps_param);
            fprintf(fp, ",\"tui_subtab\":%d,\"tier\":", o->tui_subtab);
            fprint_json_str(fp, (o->tier == T_ADV) ? "advanced" : "basic");
            fprintf(fp, ",\"serialized\":%s,\"description\":", o->is_serialized ? "true" : "false");
            fprint_json_str(fp, o->description);
            fputc('}', fp);
        }
    }
    fputs(first ? "],\n" : "\n  ],\n", fp);

    /* 2. Runtime-introspected MILK_RUNNABLE FPS units */
    fputs("  \"fps_units\": [", fp);
    first = 1;
    if (flt->include_fps)
    {
        size_t                      n_units = 0u;
        const MILK_RUNNABLE *const *units   = cacaosim_list_runnables(&n_units);
        for (size_t u = 0u; u < n_units; u++)
        {
            const MILK_RUNNABLE *r = units[u];
            if (!fps_unit_matches(r, flt))
            {
                continue;
            }
            size_t umatch = 0u;
            for (int bi = 0; bi < r->nb_bindings; bi++)
            {
                if (fps_binding_matches(r, &r->bindings[bi], flt))
                {
                    umatch++;
                }
            }
            if (umatch == 0u)
            {
                continue;
            }
            const char *sname = unit_short_name(r);
            char        bname[96];
            snprintf(bname, sizeof(bname), "milk-fpsexec-cacaosim-%s", sname);
            fputs(first ? "\n" : ",\n", fp);
            first = 0;
            fputs("    {\n      \"package\":\"cacaosim\",\"unit\":", fp);
            fprint_json_str(fp, sname);
            fputs(",\"cmdkey\":", fp);
            fprint_json_str(fp, r->app.cmdkey ? r->app.cmdkey : sname);
            fputs(",\"fps_name\":", fp);
            fprint_json_str(fp, r->app.fps_name ? r->app.fps_name : "");
            fputs(",\"binary\":", fp);
            fprint_json_str(fp, bname);
            fputs(",\"description\":", fp);
            fprint_json_str(fp, r->app.description ? r->app.description : "");
            fputs(",\n      \"params\": [", fp);
            int pfirst = 1;
            for (int bi = 0; bi < r->nb_bindings; bi++)
            {
                const FPS_CLI_BINDING *b = &r->bindings[bi];
                if (!fps_binding_matches(r, b, flt))
                {
                    continue;
                }
                char dval[64];
                format_binding_default(b, dval, sizeof(dval));
                const char *mut = fps_mutability(b->fpflag);
                fputs(pfirst ? "\n" : ",\n", fp);
                pfirst = 0;
                fputs("        {\"keyword\":", fp);
                fprint_json_str(fp, b->fpskeyword ? b->fpskeyword : "");
                fputs(",\"type\":", fp);
                fprint_json_str(fp, fptype_label(b->type));
                fputs(",\"default\":", fp);
                fprint_json_str(fp, dval);
                fputs(",\"access\":", fp);
                fprint_json_str(fp, mut);
                fprintf(fp, ",\"writerun\":%s,\"primary\":%s,\"trigger\":%s,\"description\":",
                        ((b->fpflag & FPFLAG_WRITERUN) != 0u) ? "true" : "false",
                        b->is_primary ? "true" : "false",
                        ((b->fpflag & FPFLAG_TRIGGER_STREAM) != 0u) ? "true" : "false");
                fprint_json_str(fp, b->descr ? b->descr : "");
                fputc('}', fp);
            }
            fputs("\n      ]\n    }", fp);
        }
    }
    fputs(first ? "],\n" : "\n  ],\n", fp);

    /* 3. CLI options */
    fputs("  \"cli_options\": [", fp);
    first = 1;
    if (flt->include_cli)
    {
        for (size_t i = 0u; i < cacaosim_schema_cli_count(); i++)
        {
            const cacaosim_cli_opt_t *c = &g_cli_opts[i];
            if (!cli_opt_matches(c, flt))
            {
                continue;
            }
            fputs(first ? "\n" : ",\n", fp);
            first = 0;
            fputs("    {\"tool\":", fp);
            fprint_json_str(fp, c->tool);
            fputs(",\"flag\":", fp);
            fprint_json_str(fp, c->flag);
            fputs(",\"arg\":", fp);
            fprint_json_str(fp, c->arg);
            fputs(",\"default\":", fp);
            fprint_json_str(fp, c->default_str);
            fputs(",\"group\":", fp);
            fprint_json_str(fp, c->group);
            fputs(",\"description\":", fp);
            fprint_json_str(fp, c->description);
            fputc('}', fp);
        }
    }
    fputs(first ? "]\n}\n" : "\n  ]\n}\n", fp);

    return ferror(fp) ? -1 : 0;
}
