// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_beamsplit_FPS.h
 * @brief   FPS runnable declaration for cacaosim.beamsplit
 *
 * Architectural Role:
 *   Declares the `runnable_cacaosim_beamsplit` Function Parameter Structure (FPS)
 *   compute-unit descriptor (`csim_beamsplit` / CLI `cacaosim.beamsplit` /
 *   standalone executable `milk-fpsexec-cacaosim-beamsplit`), forming the
 *   optical branching node that divides a single upstream SHM beam into reflected
 *   and transmitted downstream arms (e.g., visible WFS arm vs NIR science arm, or
 *   HOWFS vs LOWFS split).
 *
 * Design Rationale & Motivation:
 *   Because `cacaosim` beams carry an achromatic `[OPD_um, Amplitude]` cube paired
 *   with a 32-bin photon-rate spectrum (`0.40..2.40 um`), a beamsplitter or
 *   interference dichroic never needs expensive focal-plane FFTs. Instead, it
 *   partitions photon arrival rates per spectral bin while applying independent
 *   per-arm differential NCPA, pupil/Lyot stop masks, and 2D affine registration
 *   (`shift_x`, `shift_y`, `rot_deg`, `scale`) directly in the pupil plane.
 */

#ifndef CACAOSIM_BEAMSPLIT_FPS_H
#define CACAOSIM_BEAMSPLIT_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for `cacaosim.beamsplit` (`milk-fpsexec-cacaosim-beamsplit`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_beamsplit;

#endif /* CACAOSIM_BEAMSPLIT_FPS_H */
