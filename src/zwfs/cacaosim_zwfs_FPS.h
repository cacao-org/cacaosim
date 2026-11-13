// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_zwfs_FPS.h
 * @brief   MILK_RUNNABLE declaration for Zernike phase-contrast WFS compute unit
 *
 * Architectural Role in cacaosim:
 *   Exposes the `runnable_cacaosim_zwfs` descriptor (`cacaosim.zwfs` in `libmilkcacaosim.so` and
 *   standalone `milk-fpsexec-cacaosim-zwfs`) so both the plugin module registry and the standalone
 *   FPS executable wrapper bind the Babinet Matrix-DFT Zernike WFS / LOWFS unit without linking
 *   `CLIcore` into `fpsexec` binaries.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   Wraps the microsecond-latency Babinet Soummer Matrix-DFT Zernike phase-contrast engine
 *   (`Psi_out = Psi_pup - (1 - t_d * exp(i * depth)) * MFT^-1[D * MFT[Psi_pup]]`), resolving the
 *   `~1.06 lambda/D` phase-shifting focal dimple with subpixel fidelity via compact `cblas_cgemm`
 *   / `cublasCgemm` matrix multiplies instead of massive zero-padded 2D FFTs, and exposing live
 *   FPS shared-memory controls for CACAO closed-loop AO and coronagraphic LOWFS benches.
 */

#ifndef CACAOSIM_ZWFS_FPS_H
#define CACAOSIM_ZWFS_FPS_H

#include "milk_runnable.h"

/**
 * runnable_cacaosim_zwfs - FPS runnable descriptor for the Zernike phase-contrast WFS unit
 *
 * Role:
 *   Registers the `zwfs` (`csim_zwfs`) Function Parameter Structure schema, configuration
 *   validator, and real-time shared-memory compute loop with the `milk` plugin loader and
 *   `milk-fpsexec-cacaosim-zwfs`.
 */
extern const MILK_RUNNABLE runnable_cacaosim_zwfs;

#endif /* CACAOSIM_ZWFS_FPS_H */
