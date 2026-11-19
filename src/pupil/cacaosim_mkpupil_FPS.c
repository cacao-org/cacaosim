// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mkpupil_FPS.c
 * @brief   MILK_RUNNABLE compute unit generating telescope pupil mask and static NCPA streams
 *
 * Architectural Role:
 *   Implements the `cacaosim.mkpupil` (`milk-fpsexec-cacaosim-mkpupil`) FPS
 *   compute unit. Wraps the cold-path `cacaosim_make_pupil()` and
 *   `cacaosim_make_ncpa()` library routines inside a `milk` `PROCESSINFO`-managed
 *   execution loop that creates and populates persistent 2D `FLOAT32`
 *   `ImageStreamIO` shared-memory streams (`.out_pup` and optional `.out_ncpa`).
 *
 * Design Rationale & Motivation:
 *   - Decoupled static geometry setup: In modular multi-process AO benches,
 *     generating the 4x4 sub-pixel anti-aliased telescope pupil and static NCPA
 *     once into named SHM streams lets `cacaosim.mkbeam`, `cacaosim.pyrwfs`,
 *     `cacaosim.shwfs`, `cacaosim.zwfs`, `cacaosim.coro`, and external CACAO
 *     calibration scripts read or inspect the exact pupil mask directly in SHM.
 *   - `PROCESSINFO` lifecycle compliance: Even though pupil synthesis is usually
 *     executed as a one-shot initialization (`loopcntMax = 1`) or low-rate task,
 *     wrapping it in `MILK_PROCINFO_LOOPSTART` with proper `md->write = 1` and
 *     `processinfo_update_output_stream()` handshakes ensures downstream stream
 *     semaphores, frame counters, and `milk-procCTRL` telemetry behave identically
 *     to real-time compute units.
 */

#include <stdint.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "pupil/cacaosim_mkpupil_FPS.h"
#include "pupil/cacaosim_pupil.h"

/*
 * Module-static backing storage bound to FPS parameter entries via CACAOSIM_MKPUPIL_PARAMS.
 * Defaults model an 8m-class centrally obscured 4-spider telescope on a 128x128 grid with
 * pup_diam = 120/128 = 0.9375 (leaving a 4-pixel zero guard border around the clear aperture).
 */
static char     param_out_pup[FUNCTION_PARAMETER_STRMAXLEN]  = "csim_pup";
static uint32_t param_nwf                                    = 128;
static float    param_pup_diam                               = 0.9375f;
static float    param_cobs                                   = 0.30f;
static int32_t  param_nspider                                = 4;
static float    param_spider_w                               = 0.01f;
static float    param_spider_deg                             = 45.0f;
static float    param_spider_offset                          = 0.0f;
static char     param_out_ncpa[FUNCTION_PARAMETER_STRMAXLEN] = "";
static float    param_ncpa_rms                               = 0.0f;
static uint64_t param_ncpa_seed                              = 42;

/**
 * CACAOSIM_MKPUPIL_PARAMS(X) - X-macro FPS parameter table for `cacaosim.mkpupil`
 *
 * Parameters:
 *   - `.out_pup`    (STREAMNAME, required): Output `[nwf x nwf]` `FLOAT32` anti-aliased pupil
 *                   amplitude SHM stream in `[0, 1]`.
 *   - `.nwf`        (UINT32, required): Square wavefront grid dimension [pixels] (`>= 8`).
 *   - `.pup_diam`   (FLOAT32): Outer clear aperture diameter as fraction of `nwf` `(0, 1]`.
 *   - `.cobs`       (FLOAT32): Secondary central obstruction linear ratio `D_sec / D_tel`.
 *   - `.nspider`    (INT32): Count of radial spider support vanes (`0..16`, `0` = unobstructed).
 *   - `.spider_w`   (FLOAT32): Full spider vane width expressed as fraction of `pup_diam`.
 *   - `.spider_deg` (FLOAT32): Counter-clockwise azimuth of first spider vane from `+X` [deg].
 *   - `.out_ncpa`   (STRING): Optional output `[nwf x nwf]` `FLOAT32` static NCPA OPD stream [um];
 *                   skipped when empty `""`.
 *   - `.ncpa_rms`   (FLOAT32): Target pupil-weighted RMS amplitude of `.out_ncpa` [microns].
 *   - `.ncpa_seed`  (UINT64): Deterministic `xoshiro128++` seed for Zernike NCPA synthesis.
 */
#define CACAOSIM_MKPUPIL_PARAMS(X)                                                      \
    X(".out_pup", param_out_pup, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,            \
      "Output 2D pupil amplitude stream name")                                          \
    X(".nwf", &param_nwf, FPTYPE_UINT32, 1, FPFLAG_DEFAULT_INPUT,                       \
      "Square pupil grid size [pixels] (default: 128)")                                 \
    X(".pup_diam", &param_pup_diam, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Outer pupil diameter as fraction of nwf (default: 0.9375)")                      \
    X(".cobs", &param_cobs, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,                    \
      "Central obstruction fraction of pup_diam (default: 0.30)")                       \
    X(".nspider", &param_nspider, FPTYPE_INT32, 0, FPFLAG_DEFAULT_INPUT,                \
      "Number of radial spider vanes (default: 4)")                                     \
    X(".spider_w", &param_spider_w, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Spider vane width as fraction of pup_diam (default: 0.01)")                      \
    X(".spider_deg", &param_spider_deg, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,        \
      "Spider vane azimuth offset [deg] (default: 45)")                                 \
    X(".spider_offset", &param_spider_offset, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,  \
      "V-spider X-vertex offset fraction of pup_diam (default: 0.0 = radial)")          \
    X(".out_ncpa", param_out_ncpa, FPTYPE_STRING, 0, FPFLAG_DEFAULT_INPUT,              \
      "Optional output static NCPA OPD stream [um] (empty = skip)")                     \
    X(".ncpa_rms", &param_ncpa_rms, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT,            \
      "Static NCPA RMS OPD [um] (default: 0.0)")                                        \
    X(".ncpa_seed", &param_ncpa_seed, FPTYPE_UINT64, 0, FPFLAG_DEFAULT_INPUT,           \
      "Random seed for NCPA Zernike coefficients")

MILK_PARAMS_DEFINE(cacaosim_mkpupil, CACAOSIM_MKPUPIL_PARAMS);

/**
 * mkpupil_compute() - FPS execution loop synthesizing pupil amplitude and optional NCPA streams
 * @self: Pointer to `runnable_cacaosim_mkpupil` descriptor supplying FPS & processinfo metadata
 *
 * Role:
 *   Entry point invoked when `cacaosim.mkpupil` transitions to `RUN`. Connects or
 *   creates the 2D `FLOAT32` `.out_pup` (and optional `.out_ncpa`) SHM streams,
 *   synthesizes the 4x4 anti-aliased pupil mask and pupil-weighted Zernike NCPA
 *   map directly into SHM pixel buffers on each `PROCESSINFO` iteration, and
 *   notifies downstream consumers via `processinfo_update_output_stream()`.
 *
 * Rationale:
 *   Synthesizing `.out_pup` prior to `.out_ncpa` allows `img_pup.im->array.F` to
 *   be passed directly into `cacaosim_make_ncpa()` as the exact discrete support
 *   and weight map without allocating temporary intermediate buffers.
 *
 * Return: `RETURN_SUCCESS` on clean loop completion, or `RETURN_FAILURE` if
 *         grid/stream parameters are invalid or SHM allocation fails.
 */
static MILK_HOT errno_t mkpupil_compute(const MILK_RUNNABLE *self)
{
    if (param_nwf < 8 || param_out_pup[0] == '\0')
    {
        return RETURN_FAILURE;
    }

    /* Connect or allocate the persistent 2D [nwf x nwf] FLOAT32 pupil amplitude SHM stream */
    IMGID img_pup = stream_connect_create_2Df32(param_out_pup, param_nwf, param_nwf);
    if (img_pup.ID == -1)
    {
        imgid_free(&img_pup);
        return RETURN_FAILURE;
    }

    /* Optionally allocate the companion 2D [nwf x nwf] FLOAT32 static NCPA OPD [um] stream */
    int   has_ncpa = (param_out_ncpa[0] != '\0');
    IMGID img_ncpa = { 0 };
    img_ncpa.ID    = -1;
    if (has_ncpa)
    {
        img_ncpa = stream_connect_create_2Df32(param_out_ncpa, param_nwf, param_nwf);
        if (img_ncpa.ID == -1)
        {
            imgid_free(&img_ncpa);
            imgid_free(&img_pup);
            return RETURN_FAILURE;
        }
    }

    MILK_PROCINFO_INIT(self)
    MILK_PROCINFO_LOOPSTART(self)
    {
        cacaosim_pupil_cfg_t cfg = {
            .nwf           = param_nwf,
            .pup_diam      = param_pup_diam,
            .cobs          = param_cobs,
            .nspider       = param_nspider,
            .spider_w      = param_spider_w,
            .spider_deg    = param_spider_deg,
            .spider_offset = param_spider_offset,
        };

        /* Flag SHM write-in-progress, render 4x4 anti-aliased pupil, and post semaphores */
        img_pup.md->write = 1;
        cacaosim_make_pupil(&cfg, img_pup.im->array.F);
        processinfo_update_output_stream(processinfo, img_pup.im, NULL);

        if (has_ncpa)
        {
            /* Reuse the freshly rendered pupil amplitude array for exact weighted RMS scaling */
            img_ncpa.md->write = 1;
            cacaosim_make_ncpa(param_nwf, param_pup_diam, img_pup.im->array.F, param_ncpa_rms,
                               param_ncpa_seed, img_ncpa.im->array.F);
            processinfo_update_output_stream(processinfo, img_ncpa.im, NULL);
        }
    }
    MILK_PROCINFO_END(self)

    if (has_ncpa)
    {
        imgid_free(&img_ncpa);
    }
    imgid_free(&img_pup);
    return RETURN_SUCCESS;
}

/* Public MILK_RUNNABLE registration and standalone milk-fpsexec-cacaosim-mkpupil export */
const MILK_RUNNABLE runnable_cacaosim_mkpupil = {
    .kind = MILK_RUNNABLE_FPS,
    .app  = { .fps_name    = "csim_mkpupil",
              .cmdkey      = "mkpupil",
              .description = "Generate anti-aliased telescope pupil and static NCPA streams" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(cacaosim_mkpupil),
    .compute = mkpupil_compute,
};
EXPORT_FPSEXEC(milk, "cacaosim-mkpupil", runnable_cacaosim_mkpupil);
