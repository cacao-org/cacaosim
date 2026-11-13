// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_coro_FPS.h
 * @brief   MILK_RUNNABLE declaration for coronagraphic focal-plane mask & Lyot stop compute unit
 *
 * Architectural Role in cacaosim:
 *   Exposes the `runnable_cacaosim_coro` descriptor (`cacaosim.coro`, default FPS instance name
 *   `csim_coro`, standalone executable `milk-fpsexec-cacaosim-coro`) to the `libmilkcacaosim.so`
 *   module table so `milk` and `cacao` pipelines can instantiate polychromatic coronagraphic
 *   propagation and LOWFS pickoff stages as shared-memory FPS processes.
 *
 * Design Rationale:
 *   Depends solely on `milk_runnable.h` (never `CLIcore.h`) in accordance with the
 *   `milk_module_example` plugin architecture, keeping module registration decoupled from the
 *   OpenBLAS/FFTW3f/CUDA implementation details in `cacaosim_coro.h`.
 */

#ifndef CACAOSIM_CORO_FPS_H
#define CACAOSIM_CORO_FPS_H

#include "milk_runnable.h"

/**
 * @brief FPS runnable descriptor for the `cacaosim.coro` coronagraph & LOWFS compute unit.
 */
extern const MILK_RUNNABLE runnable_cacaosim_coro;

#endif /* CACAOSIM_CORO_FPS_H */
