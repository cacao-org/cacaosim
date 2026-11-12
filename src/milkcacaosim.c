// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkcacaosim.c
 * @brief   MILK_MODULE_RUNNABLES registry for the milkcacaosim plugin
 *
 * Exposes all cacaosim compute units under the "cacaosim" namespace:
 *   - cacaosim.mkpupil   : Anti-aliased telescope pupil & static NCPA generator
 *   - cacaosim.mkbeam    : Optical [OPD, Amp] beam + 32-bin stellar spectrum synthesizer
 *   - cacaosim.dmsim     : Zonal DM / 2-axis TTM hardware simulator & in-line beam corrector
 *   - cacaosim.beamsplit : Gray & dichroic beamsplitter with per-arm NCPA, masks, & registration
 *   - cacaosim.coro      : Coronagraphic FPM, Lyot stop, and LOWFS core pickoff simulator
 *   - cacaosim.pyrwfs    : Modulated/unmodulated 4-sided Pyramid WFS + detector simulator
 *   - cacaosim.shwfs     : Diffractive Shack-Hartmann WFS + detector simulator
 *   - cacaosim.zwfs      : Babinet Matrix-DFT Zernike phase-contrast WFS + detector simulator
 *   - cacaosim.fpcam     : Polychromatic MFT/FFT focal plane science PSF + detector simulator
 *   - cacaosim.htsim     : Fused zero-hop single- or multi-stage (.bench) HT AO bench simulator
 */

#define MODULE_SHORTNAME_DEFAULT "cacaosim"
#define MODULE_DESCRIPTION       "High-Throughput Adaptive Optics Hardware and Optics Simulator"

#include "milk_runnable.h"
#include "milkcacaosim.h"

MODULE_DEPS("milkCOREMODmemory", "milkCOREMODarith", "milkCOREMODiofits", "milkCOREMODtools");

static const MILK_RUNNABLE *const runnables[] = {
    &runnable_cacaosim_mkpupil,
    &runnable_cacaosim_mkbeam,
    &runnable_cacaosim_dmsim,
    &runnable_cacaosim_beamsplit,
    &runnable_cacaosim_coro,
    &runnable_cacaosim_pyrwfs,
    &runnable_cacaosim_shwfs,
    &runnable_cacaosim_zwfs,
    &runnable_cacaosim_fpcam,
    &runnable_cacaosim_htsim,
};

MILK_MODULE_RUNNABLES(milkcacaosim, _module_deps, runnables);
