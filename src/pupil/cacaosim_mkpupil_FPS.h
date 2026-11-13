// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mkpupil_FPS.h
 * @brief   MILK_RUNNABLE declaration for cacaosim.mkpupil
 *
 * Architectural Role:
 *   Exposes the `runnable_cacaosim_mkpupil` Function Parameter Structure (FPS)
 *   compute-unit descriptor (`csim_mkpupil` / CLI `cacaosim.mkpupil` / standalone
 *   executable `milk-fpsexec-cacaosim-mkpupil`) to the `libmilkcacaosim.so`
 *   plugin module table.
 *
 * Design Rationale & Motivation:
 *   Factoring telescope pupil mask and static NCPA synthesis into a dedicated
 *   `MILK_RUNNABLE` process allows modular multi-process benches to publish one
 *   canonical 2D anti-aliased pupil mask (`csim_pup`) and optional static NCPA
 *   stream into `ImageStreamIO` shared memory. Downstream optical source, WFS,
 *   coronagraph, camera, and CACAO calibration processes then attach to the
 *   exact same SHM geometry without duplicating aperture configuration parameters
 *   or recomputing sub-pixel anti-aliasing on every frame.
 */

#ifndef CACAOSIM_MKPUPIL_FPS_H
#define CACAOSIM_MKPUPIL_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for `cacaosim.mkpupil` (`milk-fpsexec-cacaosim-mkpupil`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_mkpupil;

#endif /* CACAOSIM_MKPUPIL_FPS_H */
