// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_wfe_FPS.h
 * @brief   FPS runnable declaration for cacaosim.wfe (static wavefront error element in OPD [m])
 *
 * Architectural Role:
 *   Declares the `runnable_cacaosim_wfe` Function Parameter Structure (FPS)
 *   compute-unit descriptor (`csim_wfe` / CLI `cacaosim.wfe` / standalone
 *   executable `milk-fpsexec-cacaosim-wfe` / `.bench` element `[wfe.<id>]`), an
 *   inline optical aberration relay element (`in_beam -> out_beam`).
 *
 * Design Rationale & Motivation:
 *   CACAO calibration loops, phase-diversity estimators, and FITS optical
 *   surface error maps conventionally represent wavefront/OPD errors in SI
 *   meters `[m]`, whereas `cacaosim` optical beams carry OPD in microns `[um]`.
 *   `cacaosim.wfe` provides a dedicated bridge: its `.in_wfe` SHM map is stored
 *   and tagged in `[meter]` (`BUNIT = "m"`), scaled on-the-fly by `1e6 * wfe_gain`
 *   into beam microns, and applied across both monochromatic (`nz = 2`) and
 *   polychromatic (`nz = 2 * N_lambda`) optical beams with optional 2D pupil
 *   registration and throughput attenuation.
 */

#ifndef CACAOSIM_WFE_FPS_H
#define CACAOSIM_WFE_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for `cacaosim.wfe` (`milk-fpsexec-cacaosim-wfe`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_wfe;

#endif /* CACAOSIM_WFE_FPS_H */
