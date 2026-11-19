// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mkbeam_FPS.c
 * @brief   MILK_RUNNABLE compute unit synthesizing an [OPD, Amp] optical beam + stellar spectrum
 *
 * Architectural Role:
 *   Implements `cacaosim.mkbeam` (`milk-fpsexec-cacaosim-mkbeam`), the optical
 *   entrance-pupil source node of modular `cacaosim` benches. Combines a 2D
 *   anti-aliased telescope pupil mask (`.in_pup` or an internally generated
 *   fallback), live atmospheric turbulence wavefront phase/OPD (`.in_wf`) and
 *   optional scintillation amplitude (`.in_amp`), optional static telescope NCPA
 *   (`.in_ncpa`), and a 32-bin Vega-calibrated Planck blackbody photon spectrum
 *   into a single 3D `[nwf x nwf x 2]` `FLOAT32` SHM optical beam stream
 *   (`.out_beam`).
 *
 * Design Rationale & Motivation:
 *   - Wavelength-agnostic `[OPD_um, Amplitude]` packaging: Storing achromatic
 *     optical path difference in microns alongside pupil amplitude defers
 *     `exp(i * 2*pi * OPD / lambda)` evaluation to terminal WFS, coronagraph, and
 *     camera units, enabling polychromatic broadband integration and multi-arm
 *     dichroic splitting without duplicating upstream propagation cubes.
 *   - Dual clocking topology: When `.frame_rate <= 0.1`, `mkbeam` runs slaved to
 *     the `.in_wf` `FPFLAG_TRIGGER_STREAM` semaphore (e.g., CACAO's `outarraypha`
 *     turbulence stream). When `.frame_rate > 0.1`, an absolute `CLOCK_MONOTONIC`
 *     deadline timer (`cacaosim_sleep_until`) paces free-running kHz execution
 *     without cumulative OS sleep drift, even when `.in_wf` is static or absent.
 *   - Self-healing startup: Auto-synthesizes an aligned fallback pupil if `.in_pup`
 *     is missing and lazily reconnects `.in_wf` / `.in_amp` on each frame so
 *     multi-process tmux benches start cleanly regardless of process launch order.
 */

#include <stdint.h>
#include <stdlib.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "beam/cacaosim_mkbeam_FPS.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_time.h"
#include "pupil/cacaosim_pupil.h"

/*
 * Module-static FPS parameter backing storage for cacaosim.mkbeam.
 * Defaults model an 8.2m Subaru/VLT-class telescope observing an m_V = 8.0 G2V (5778 K) guide
 * star at 75% atmosphere+telescope throughput, reading phase [rad] at 0.50 um from outarraypha.
 */
static char     param_out_beam[FUNCTION_PARAMETER_STRMAXLEN] = "csim_beam0";
static char     param_in_wf[FUNCTION_PARAMETER_STRMAXLEN]    = "outarraypha";
static char     param_in_amp[FUNCTION_PARAMETER_STRMAXLEN]   = "";
static char     param_in_pup[FUNCTION_PARAMETER_STRMAXLEN]   = "csim_pup";
static char     param_in_ncpa[FUNCTION_PARAMETER_STRMAXLEN]  = "";
static uint32_t param_nwf                                    = 128;
static float    param_pup_diam                               = 0.9375f;
static float    param_obs_frac                               = 0.30f;
static int32_t  param_n_spiders                              = 4;
static float    param_spider_w                               = 0.012f;
static float    param_spider_deg                             = 45.0f;
static float    param_spider_offset                          = 0.0f;
static float    param_tel_diam_m                             = 8.2f;
static float    param_star_mag                               = 8.0f;
static float    param_star_teff                              = 5778.0f;
static float    param_tel_tput                               = 0.75f;
static int32_t  param_wf_unit                                = 1;
static float    param_ref_lambda                             = 0.50f;
static float    param_turb_gain                              = 1.0f;
static float    param_frame_rate                             = 0.0f;

/**
 * CACAOSIM_MKBEAM_PARAMS(X) - X-macro FPS parameter table for `cacaosim.mkbeam`
 *
 * Parameters:
 *   - `.out_beam`   (STREAMNAME, required): Output `[nwf x nwf x 2]` `FLOAT32` optical beam stream
 *                   (`slice 0 = OPD [um]`, `slice 1 = Amplitude`, plus `CSIM_S00..S31` keywords).
 *   - `.in_wf`      (STREAMNAME, trigger): Optional upstream `[nwf x nwf]` `FLOAT32` atmospheric
 *                   turbulence stream (e.g. CACAO `outarraypha`).
 *   - `.in_amp`     (STRING): Optional upstream `[nwf x nwf]` `FLOAT32` atmospheric scintillation
 *                   amplitude stream (e.g. `outarrayamp`), multiplied with pupil amplitude.
 *   - `.in_pup`     (STRING): Input `[nwf x nwf]` `FLOAT32` pupil mask stream; auto-synthesized
 *                   internally from `.pup_diam`/`.obs_frac`/`.n_spiders`/`.spider_w` if absent.
 *   - `.in_ncpa`    (STRING): Optional `[nwf x nwf]` `FLOAT32` static telescope NCPA OPD [um].
 *   - `.nwf`        (UINT32): Square pupil grid size [pixels] (`>= 8`, default `128`).
 *   - `.pup_diam`   (FLOAT32): Clear pupil diameter fraction of `nwf` for fallback pupil.
 *   - `.obs_frac`   (FLOAT32): Linear central obscuration ratio `D_sec / D_tel` `[0..1)` used both
 *                   for fallback pupil geometry and collecting-area photon budget.
 *   - `.n_spiders`  (INT32): Radial spider vane count for fallback pupil synthesis.
 *   - `.spider_w`   (FLOAT32): Spider vane width fraction of `pup_diam` for fallback pupil.
 *   - `.tel_diam_m` (FLOAT32): Primary mirror diameter [m] determining geometric collecting area.
 *   - `.star_mag`   (FLOAT32, WRITERUN): Apparent V-band Vega magnitude (`0.55 um` zero-point).
 *   - `.star_teff`  (FLOAT32, WRITERUN): Stellar effective blackbody temperature [K] (`5778` G2V).
 *   - `.tel_tput`   (FLOAT32, WRITERUN): End-to-end atmosphere + telescope throughput `[0..1]`.
 *   - `.wf_unit`    (INT32): Physical unit of `.in_wf`: `0` = OPD [um], `1` = phase [rad].
 *   - `.ref_lambda` (FLOAT32): Reference wavelength [um] converting phase [rad] -> OPD [um] when
 *                   `.wf_unit == 1` (`OPD = turb_gain * phase * ref_lambda / (2*pi)`).
 *   - `.turb_gain`  (FLOAT32, WRITERUN): Live dimensionless multiplier on `.in_wf` (`0` = flat).
 *   - `.frame_rate` (FLOAT32, WRITERUN): Internal monotonic pacing rate [Hz] (`<= 0.1` uses
 *                   trigger stream pacing).
 */
#define CACAOSIM_MKBEAM_PARAMS(X)                                                       \
    X(".out_beam", param_out_beam, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,          \
      "Output 3D [nwf x nwf x 2] optical beam stream (slice0=OPD_um, slice1=Amp)")      \
    X(".in_wf", param_in_wf, FPTYPE_STREAMNAME, 1,                                      \
      FPFLAG_DEFAULT_INPUT | FPFLAG_TRIGGER_STREAM,                                     \
      "Optional input atmospheric WF stream (e.g. outarraypha)")                        \
    X(".in_amp", param_in_amp, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Optional input atmospheric amplitude stream (e.g. outarrayamp)")                 \
    X(".in_pup", param_in_pup, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                  \
      "Input 2D pupil amplitude stream (auto-generated if absent)")                     \
    X(".in_ncpa", param_in_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,                \
      "Optional static telescope NCPA OPD stream [um]")                                 \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT,                       \
      "Pupil grid size [pixels] (default: 128)")                                        \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Clear pupil diameter fraction of nwf (default: 0.9375)")                         \
    X(".obs_frac", &param_obs_frac, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Central obscuration linear fraction (default: 0.30)")                            \
    X(".n_spiders", &param_n_spiders, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Spider count when auto-generating pupil (default: 4)")                           \
    X(".spider_w", &param_spider_w, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Spider width fraction of pupil diameter (default: 0.012)")                       \
    X(".spider_deg", &param_spider_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Spider orientation angle [deg] when auto-generating pupil (default: 45)")        \
    X(".spider_offset", &param_spider_offset, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,  \
      "V-spider X-vertex offset fraction of pup_diam (default: 0.0)")                   \
    X(".tel_diam_m", &param_tel_diam_m, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Primary mirror diameter [m] for photon budget (default: 8.2)")                   \
    X(".star_mag", &param_star_mag, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Guide/target star V-band Vega magnitude (default: 8.0)")                         \
    X(".star_teff", &param_star_teff, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Stellar blackbody temperature [K] (default: 5778)")                              \
    X(".tel_tput", &param_tel_tput, FPTYPE_FLOAT32, 0,                                  \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Atmosphere + telescope throughput [0..1] (default: 0.75)")                       \
    X(".wf_unit", &param_wf_unit, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1)")                   \
    X(".ref_lambda", &param_ref_lambda, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Reference wavelength [um] for phase input (default: 0.50)")                      \
    X(".turb_gain", &param_turb_gain, FPTYPE_FLOAT32, 0,                                \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Live atmospheric turbulence multiplier (0 = flat WF)")                           \
    X(".frame_rate", &param_frame_rate, FPTYPE_FLOAT32, 0,                              \
      FPFLAG_DEFAULT_INPUT | FPFLAG_WRITERUN,                                           \
      "Internal TIMER_ABSTIME pacing rate [Hz] (0 = use trigger)")

MILK_PARAMS_DEFINE(cacaosim_mkbeam, CACAOSIM_MKBEAM_PARAMS);

/**
 * mkbeam_confcheck() - Validate FPS configuration and enforce live run-time writability flags
 *
 * Role:
 *   FPS `confcheck` callback executed during `CONF` state updates prior to `RUN`.
 *   Ensures live photometric, turbulence-gain, and pacing parameters retain
 *   `FPFLAG_WRITERUN` and clamps `.nwf` to a valid minimum grid dimension (`>= 8`).
 *
 * Rationale:
 *   When reattaching to a pre-existing shared-memory FPS file created by an older
 *   session or external script, `FPFLAG_WRITERUN` bits might be absent on disk.
 *   Explicitly OR-ing `FPFLAG_WRITERUN` onto `.star_mag`, `.star_teff`,
 *   `.tel_tput`, `.turb_gain`, and `.frame_rate` guarantees interactive GUI/CLI
 *   tuning (`cacaosim-CTRL`, `milk-fpsCTRL`) works seamlessly while the loop runs.
 *
 * Return: Always `RETURN_SUCCESS`.
 */
static MILK_COLD errno_t mkbeam_confcheck(void)
{
    if (dcfpsptr != NULL)
    {
        const char *live_keys[] = {
            ".star_mag", ".star_teff", ".tel_tput", ".turb_gain", ".frame_rate"
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
    if (param_nwf < 8)
    {
        param_nwf = 8;
    }
    return RETURN_SUCCESS;
}

/**
 * mkbeam_compute() - Main real-time loop packaging pupil, atmosphere, and star into `.out_beam`
 * @self: Pointer to `runnable_cacaosim_mkbeam` descriptor
 *
 * Role:
 *   Allocates `.out_beam` (`[nwf x nwf x 2]`, `FLOAT32`) with fixed `CSIM_BM`
 *   spectral keyword slots, resolves `.in_pup` (or synthesizes an aligned 4x4
 *   anti-aliased fallback pupil), and continuously updates `[OPD_um, Amplitude]`
 *   plus the 32-bin photon-rate spectrum on each frame.
 *
 * Rationale:
 *   - Re-evaluates `cacaosim_spec_star()` and `cacaosim_wf_to_opd_scale()` inside
 *     the frame loop (costing only 32 scalar bin evaluations) so live edits to
 *     `.star_mag`, `.star_teff`, `.tel_tput`, or `.turb_gain` take effect on the
 *     very next frame with zero heap allocation or stream reconnection.
 *   - Uses absolute `CLOCK_MONOTONIC` deadlines (`cacaosim_timespec_add_ns` +
 *     `cacaosim_sleep_until`) when `.frame_rate > 0.1` so compute jitter never
 *     accumulates into long-term frame-rate drift.
 *
 * Return: `RETURN_SUCCESS` on loop exit, or `RETURN_FAILURE` if `.out_beam` or
 *         fallback pupil allocation fails.
 */
static MILK_HOT errno_t mkbeam_compute(const MILK_RUNNABLE *self)
{
    uint32_t nwf  = (param_nwf >= 8) ? param_nwf : 128;
    uint64_t npix = (uint64_t) nwf * nwf;

    /* Create 3D [nwf x nwf x 2] beam cube (slice 0 = OPD [um], slice 1 = Amp) & init keywords */
    IMGID img_beam = stream_connect_create_3Df32(param_out_beam, nwf, nwf, 2);
    if (img_beam.ID == -1)
    {
        return RETURN_FAILURE;
    }
    cacaosim_beam_shm_init_kw(img_beam.im);

    /* Attach to external 2D pupil SHM stream if compatible; otherwise synthesize a local mask */
    IMGID  img_pup      = { 0 };
    float *fallback_pup = NULL;
    img_pup.ID          = -1;
    if (param_in_pup[0] != '\0')
    {
        img_pup = stream_connect(param_in_pup);
    }
    const float *pup_ptr = NULL;
    if (img_pup.ID != -1 && img_pup.md->datatype == _DATATYPE_FLOAT &&
        img_pup.md->size[0] == nwf && img_pup.md->size[1] == nwf)
    {
        pup_ptr = img_pup.im->array.F;
    }
    else
    {
        fallback_pup = (float *) cacaosim_aligned_calloc(npix, sizeof(float));
        if (fallback_pup == NULL)
        {
            imgid_free(&img_beam);
            return RETURN_FAILURE;
        }
        cacaosim_pupil_cfg_t pcfg = {
            .nwf           = nwf,
            .pup_diam      = param_pup_diam,
            .cobs          = param_obs_frac,
            .nspider       = param_n_spiders,
            .spider_w      = param_spider_w,
            .spider_deg    = param_spider_deg,
            .spider_offset = param_spider_offset,
        };
        cacaosim_make_pupil(&pcfg, fallback_pup);
        pup_ptr = fallback_pup;
    }

    /* Probe optional upstream turbulence WF, scintillation amplitude, and static NCPA streams */
    IMGID img_wf   = { 0 };
    IMGID img_amp  = { 0 };
    IMGID img_ncpa = { 0 };
    img_wf.ID      = -1;
    img_amp.ID     = -1;
    img_ncpa.ID    = -1;

    if (param_in_wf[0] != '\0')
    {
        img_wf = stream_connect(param_in_wf);
    }
    if (param_in_amp[0] != '\0')
    {
        img_amp = stream_connect(param_in_amp);
    }
    if (param_in_ncpa[0] != '\0')
    {
        img_ncpa = stream_connect(param_in_ncpa);
    }

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        /* Optional internal drift-free hardware pacing via absolute CLOCK_MONOTONIC deadline */
        if (param_frame_rate > 0.1f)
        {
            uint64_t period_ns = (uint64_t) (1.0e9f / param_frame_rate);
            cacaosim_timespec_add_ns(&deadline, period_ns);
            cacaosim_sleep_until(&deadline);
        }

        /* Lazily attach to turbulence streams if the atmospheric generator started after mkbeam */
        if (img_wf.ID == -1 && param_in_wf[0] != '\0')
        {
            img_wf = stream_connect(param_in_wf);
        }
        if (img_amp.ID == -1 && param_in_amp[0] != '\0')
        {
            img_amp = stream_connect(param_in_amp);
        }

        /* Validate stream datatypes and dimensions before dereferencing SHM pixel arrays */
        const float *wf_ptr = (img_wf.ID != -1 && img_wf.md->datatype == _DATATYPE_FLOAT &&
                               img_wf.md->size[0] == nwf && img_wf.md->size[1] == nwf)
                                  ? img_wf.im->array.F
                                  : NULL;
        const float *amp_ptr = (img_amp.ID != -1 && img_amp.md->datatype == _DATATYPE_FLOAT &&
                                img_amp.md->size[0] == nwf && img_amp.md->size[1] == nwf)
                                   ? img_amp.im->array.F
                                   : NULL;
        const float *ncpa_ptr = (img_ncpa.ID != -1 && img_ncpa.md->datatype == _DATATYPE_FLOAT &&
                                 img_ncpa.md->size[0] == nwf && img_ncpa.md->size[1] == nwf)
                                    ? img_ncpa.im->array.F
                                    : NULL;

        /* Synthesize 32-bin [0.40..2.40 um] stellar photon-rate spectrum from live parameters */
        cacaosim_spec_t spec;
        cacaosim_spec_star(&spec, param_tel_diam_m, param_obs_frac, param_star_mag, param_star_teff,
                           param_tel_tput);

        float wf_to_opd =
            cacaosim_wf_to_opd_scale(param_wf_unit, param_ref_lambda, param_turb_gain);

        /* Write slice 0 (OPD_um), slice 1 (Amp), and O(1) spectral keywords into SHM beam */
        img_beam.md->write = 1;
        cacaosim_beam_init_cpu(pup_ptr, amp_ptr, wf_ptr, ncpa_ptr, wf_to_opd, nwf,
                               img_beam.im->array.F, img_beam.im->array.F + npix);
        cacaosim_beam_shm_write_spec(img_beam.im, &spec);

        IMAGE *trig_src = (img_wf.ID != -1) ? img_wf.im : NULL;
        processinfo_update_output_stream(processinfo, img_beam.im, trig_src);
    }
    MILK_PROCINFO_END(self)

    free(fallback_pup);
    if (img_ncpa.ID != -1)
    {
        imgid_free(&img_ncpa);
    }
    if (img_amp.ID != -1)
    {
        imgid_free(&img_amp);
    }
    if (img_wf.ID != -1)
    {
        imgid_free(&img_wf);
    }
    if (img_pup.ID != -1)
    {
        imgid_free(&img_pup);
    }
    imgid_free(&img_beam);
    return RETURN_SUCCESS;
}

/* Public MILK_RUNNABLE registration and standalone milk-fpsexec-cacaosim-mkbeam export */
const MILK_RUNNABLE runnable_cacaosim_mkbeam = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_mkbeam",
              .cmdkey      = "mkbeam",
              .description = "Combine telescope pupil, atmosphere, and star spectrum into beam" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_mkbeam),
    .compute   = mkbeam_compute,
    .confcheck = mkbeam_confcheck,
};
EXPORT_FPSEXEC(milk, "cacaosim-mkbeam", runnable_cacaosim_mkbeam);
