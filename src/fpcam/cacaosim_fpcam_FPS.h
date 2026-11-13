// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_fpcam_FPS.h
 * @brief   Declaration for cacaosim.fpcam MILK_RUNNABLE compute unit
 *
 * Architectural Role in cacaosim:
 *   Declares the `runnable_cacaosim_fpcam` descriptor (`cacaosim.fpcam`, default FPS instance
 *   `csim_fpcam`, standalone binary `milk-fpsexec-cacaosim-fpcam`) registered in
 *   `libmilkcacaosim.so` so `milk` and `cacao` pipelines can launch polychromatic focal-plane
 *   science PSF and detector simulation stages.
 *
 * Design Rationale:
 *   Follows the `milk_module_example` convention of exposing only a `const MILK_RUNNABLE` symbol
 *   via `milk_runnable.h` without including `CLIcore.h` or private OpenBLAS/FFTW3f/CUDA headers.
 */

#ifndef CACAOSIM_FPCAM_FPS_H
#define CACAOSIM_FPCAM_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for the `cacaosim.fpcam` focal-plane science camera unit.
 */
extern const MILK_RUNNABLE runnable_cacaosim_fpcam;

#endif /* CACAOSIM_FPCAM_FPS_H */
