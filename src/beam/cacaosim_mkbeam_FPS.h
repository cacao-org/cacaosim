// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mkbeam_FPS.h
 * @brief   FPS runnable declaration for cacaosim.mkbeam
 *
 * Architectural Role:
 *   Declares the `runnable_cacaosim_mkbeam` Function Parameter Structure (FPS)
 *   compute-unit descriptor (`csim_mkbeam` / CLI `cacaosim.mkbeam` / standalone
 *   executable `milk-fpsexec-cacaosim-mkbeam`), serving as the upstream optical
 *   beam source node at the head of modular `cacaosim` SHM pipelines.
 *
 * Design Rationale & Motivation:
 *   Instead of propagating wavelength-locked complex `[Re, Im]` electric fields
 *   between processes, `cacaosim.mkbeam` packages the incident telescope light as
 *   an achromatic 3D `[nwf x nwf x 2]` `FLOAT32` shared-memory cube (`slice 0` =
 *   `OPD_um` in microns, `slice 1` = pupil/scintillation `Amplitude`) paired with
 *   a 32-bin (`0.40..2.40 um`) photon-rate spectrum serialized in `CSIM_S00..S31`
 *   SHM keywords. Decoupling geometric optical path difference from wavelength
 *   eliminates `2*pi` phase wrapping, makes downstream DM/WFE relays purely
 *   additive in microns, and allows a single optical source beam to feed visible
 *   WFSs and near-IR coronagraphs/cameras simultaneously through dichroics.
 */

#ifndef CACAOSIM_MKBEAM_FPS_H
#define CACAOSIM_MKBEAM_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for `cacaosim.mkbeam` (`milk-fpsexec-cacaosim-mkbeam`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_mkbeam;

#endif /* CACAOSIM_MKBEAM_FPS_H */
