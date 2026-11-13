// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_shwfs_FPS.h
 * @brief   MILK_RUNNABLE declaration for Diffractive Shack-Hartmann WFS compute unit
 *
 * Architectural Role in cacaosim:
 *   Declares `runnable_cacaosim_shwfs` for registration inside `libmilkcacaosim.so`
 *   (`cacaosim.shwfs`) and standalone `milk-fpsexec-cacaosim-shwfs` binary generation without
 *   `CLIcore` coupling.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   Separates MILK Function Parameter Structure (FPS) shared-memory orchestration from the pure
 *   diffractive lenslet engine (`cacaosim_shwfs.h`), allowing both standalone SHM processes and
 *   fused `cacaosim.htsim` benches to share the same L1-cached OpenMP FFTW3f and batched
 *   `cufftPlanMany` subaperture spot simulator.
 */

#ifndef CACAOSIM_SHWFS_FPS_H
#define CACAOSIM_SHWFS_FPS_H

#include "milk_runnable.h"

/**
 * @brief MILK FPS runnable descriptor for the Diffractive Shack-Hartmann WFS (`csim_shwfs`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_shwfs;

#endif /* CACAOSIM_SHWFS_FPS_H */
