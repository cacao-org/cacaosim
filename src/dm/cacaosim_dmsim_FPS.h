// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_dmsim_FPS.h
 * @brief   MILK_RUNNABLE declaration for cacaosim.dmsim
 *
 * Architectural Role:
 *   Declares the `runnable_cacaosim_dmsim` Function Parameter Structure (FPS)
 *   compute-unit descriptor (`csim_dmsim` / CLI `cacaosim.dmsim` / standalone
 *   executable `milk-fpsexec-cacaosim-dmsim`), wrapping the `cacaosim_dm_ctx_t`
 *   hardware physics engine as an `ImageStreamIO` shared-memory AO process.
 *
 * Design Rationale & Motivation:
 *   Exposes dual-mode SHM connectivity so a single runnable serves both as a
 *   standalone DM OPD surface generator (`.in_cmd -> .out_opd`, directly attaching
 *   to CACAO `dmXXdisp` total displacement streams) and as an inline reflective
 *   optical bench relay (`.in_beam -> .out_beam = .in_beam - .out_opd` with
 *   spectral reflectivity `.tput`), allowing Woofer, Tweeter, and Tip-Tilt
 *   stages to be chained modularly in shared memory.
 */

#ifndef CACAOSIM_DMSIM_FPS_H
#define CACAOSIM_DMSIM_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for `cacaosim.dmsim` (`milk-fpsexec-cacaosim-dmsim`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_dmsim;

#endif /* CACAOSIM_DMSIM_FPS_H */
