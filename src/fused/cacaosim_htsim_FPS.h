// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_htsim_FPS.h
 * @brief   Public MILK_RUNNABLE descriptor declaration for the fused cacaosim.htsim FPS unit
 *
 * Architectural Role:
 *   Exposes runnable_cacaosim_htsim, the plugin registration entry point for the dual-mode
 *   high-throughput AO hardware simulator (cacaosim.htsim / milk-fpsexec-cacaosim-htsim). Both
 *   the unified libmilkcacaosim.so module table (MILK_MODULE_RUNNABLES) and the standalone
 *   fpsexec wrapper include this header to register the runnable with the milk framework.
 *
 * Design Rationale & Motivation:
 *   Keeping this interface header minimal—depending solely on milk_runnable.h without leaking
 *   internal DAG, FFTW3f, cuFFT, or ImageStreamIO implementation headers—enforces strict
 *   milk_module_example encapsulation, avoids direct CLIcore.h coupling, and minimizes rebuild
 *   fan-out across the plugin.
 */

#ifndef CACAOSIM_HTSIM_FPS_H
#define CACAOSIM_HTSIM_FPS_H

#include "milk_runnable.h"

/**
 * @brief Fused high-throughput AO bench simulator runnable descriptor (FPS cmdkey: "htsim").
 *
 * Operates either as a zero-config parameter-driven single-stage SCAO fast path (DM + PyWFS +
 * optional multi-rate focal-plane camera) when .bench_file is empty, or as a full multi-stage
 * declarative .bench DAG host with monotonic TIMER_ABSTIME pacing and live Strehl telemetry.
 */
extern const MILK_RUNNABLE runnable_cacaosim_htsim;

#endif /* CACAOSIM_HTSIM_FPS_H */
