// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkcacaosim.c
 * @brief   MILK_MODULE_RUNNABLES plugin registry for `libmilkcacaosim.so`
 *
 * Architectural Role:
 *   Defines the dynamic module metadata, inter-plugin symbol dependencies, and exported FPS
 *   runnable dispatch table for `libmilkcacaosim.so` when loaded into the `milk` framework
 *   (`feat/bash-engine` architecture) and `cacao` real-time control environments.
 *
 *   Exposes all 12 `cacaosim` compute units under the `"cacaosim"` namespace:
 *     - `cacaosim.mkpupil`   : Anti-aliased telescope pupil & static NCPA generator
 *     - `cacaosim.mkbeam`    : Optical `[OPD, Amp]` beam + 32-bin stellar spectrum synthesizer
 *     - `cacaosim.wfe`       : Inline static wavefront error optical element in OPD `[meter]`
 *     - `cacaosim.dmsim`     : Zonal DM / 2-axis TTM hardware simulator & in-line beam corrector
 *     - `cacaosim.beamsplit` : Gray & dichroic beamsplitter with per-arm NCPA, masks, & reg
 *     - `cacaosim.coro`      : Coronagraphic FPM, Lyot stop, and LOWFS core pickoff simulator
 *     - `cacaosim.pyrwfs`    : Modulated/unmodulated 4-sided Pyramid WFS + detector simulator
 *     - `cacaosim.shwfs`     : Diffractive Shack-Hartmann WFS + detector simulator
 *     - `cacaosim.zwfs`      : Babinet Matrix-DFT Zernike phase-contrast WFS + detector simulator
 *     - `cacaosim.fpcam`     : Polychromatic MFT/FFT focal plane science PSF + detector simulator
 *     - `cacaosim.htsim`     : Fused zero-hop single- or multi-stage (`.bench`) HT AO simulator
 *     - `cacaosim.wfstat`    : Synchronized optical wavefront & PSF telemetry statistics collector
 *
 * Design Rationale & Motivation:
 *   Following the declarative `milk_module_example` (`milk_runnable.h`) plugin pattern instead of
 *   legacy `CLIcore.h` command registration cleanly separates pure simulation compute units from
 *   the interactive CLI parser:
 *     1. Dual deployment without code duplication: Each `MILK_RUNNABLE` descriptor registered here
 *        is also compiled via `EXPORT_FPSEXEC` into a standalone `milk-fpsexec-cacaosim-*` daemon
 *        binary that links zero `CLIcore` symbols, while `milkcacaosim.c` bundles all 11 units
 *        into a single shared object (`libmilkcacaosim.so`) for unified `milk` discovery.
 *     2. Topological ordering: The `runnables[]` array is ordered along the physical optical train
 *        (aperture/source generation -> inline wavefront perturbation/correction -> spectral/coro
 *        relays -> WFS & focal-plane cameras -> full-bench fused engine) so `milk` help listings
 *        and FPS introspection present components in natural bench-assembly sequence.
 */

/* Default short namespace prefix under which `milk` exposes `cacaosim.<runnable>` entries */
#define MODULE_SHORTNAME_DEFAULT "cacaosim"
#define MODULE_DESCRIPTION       "High-Throughput Adaptive Optics Hardware and Optics Simulator"

#include "milk_runnable.h"
#include "milkcacaosim.h"

/*
 * Declare core `milk` shared libraries required at runtime before `libmilkcacaosim.so` initializes:
 *   - milkCOREMODmemory : ImageStreamIO SHM stream creation, mapping, and ID table management
 *   - milkCOREMODarith  : Image buffer arithmetic utilities used by `milk` infrastructure
 *   - milkCOREMODiofits : Standard FITS file I/O support within hosted `milk` sessions
 *   - milkCOREMODtools  : Timing, sorting, and core `milk` helper routines
 */
MODULE_DEPS("milkCOREMODmemory", "milkCOREMODarith", "milkCOREMODiofits", "milkCOREMODtools");

/*
 * Master table of immutable `MILK_RUNNABLE` descriptors exported by this plugin. Every entry is
 * defined in its corresponding `src/<unit>/cacaosim_<unit>_FPS.c` translation unit.
 */
static const MILK_RUNNABLE *const runnables[] = {
    &runnable_cacaosim_mkpupil,
    &runnable_cacaosim_mkbeam,
    &runnable_cacaosim_adc,
    &runnable_cacaosim_wfe,
    &runnable_cacaosim_dmsim,
    &runnable_cacaosim_beamsplit,
    &runnable_cacaosim_coro,
    &runnable_cacaosim_pyrwfs,
    &runnable_cacaosim_shwfs,
    &runnable_cacaosim_zwfs,
    &runnable_cacaosim_fpcam,
    &runnable_cacaosim_htsim,
    &runnable_cacaosim_wfstat,
};

const MILK_RUNNABLE *const *cacaosim_list_runnables(size_t *out_count)
{
    if (out_count != NULL)
    {
        *out_count = sizeof(runnables) / sizeof(runnables[0]);
    }
    return runnables;
}

/* Emit standard `milk` dynamic module init/registration hooks for `libmilkcacaosim.so` */
MILK_MODULE_RUNNABLES(milkcacaosim, _module_deps, runnables);
