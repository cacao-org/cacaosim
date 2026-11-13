// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_pyrwfs_FPS.h
 * @brief   MILK_RUNNABLE declaration for the 4-sided Pyramid WFS (`cacaosim.pyrwfs`)
 *
 * Architectural Role in cacaosim:
 *   Exposes the `runnable_cacaosim_pyrwfs` descriptor so `libmilkcacaosim.so` can register
 *   `cacaosim.pyrwfs` in its `MILK_MODULE_RUNNABLES` table and build the standalone
 *   `milk-fpsexec-cacaosim-pyrwfs` executable without `CLIcore` linkage.
 *
 * Design Rationale & Physics / Algorithmic Motivation:
 *   Decouples the MILK Function Parameter Structure (FPS) shared-memory transport wrapper from the
 *   pure C/CUDA diffractive Pyramid WFS engine (`cacaosim_pyrwfs.h`), allowing the exact same
 *   Fourier-shifted `(1 + N_mod)` FFT optical engine and CUDA Graph accelerator to run either as an
 *   independent CACAO shared-memory process (`aol0_wfsim`) or in-memory inside `cacaosim.htsim`.
 */

#ifndef CACAOSIM_PYRWFS_FPS_H
#define CACAOSIM_PYRWFS_FPS_H

#include "milk_runnable.h"

/**
 * @brief MILK FPS runnable descriptor for the 4-sided Pyramid WFS simulator (`csim_pyrwfs`).
 */
extern const MILK_RUNNABLE runnable_cacaosim_pyrwfs;

#endif /* CACAOSIM_PYRWFS_FPS_H */
