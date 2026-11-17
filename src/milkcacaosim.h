// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkcacaosim.h
 * @brief   Top-level umbrella public header for the milkcacaosim AO hardware and optics plugin
 *
 * Architectural Role:
 *   Serves as the single-include public C/C++ API aggregator for `libmilkcacaosim.so`. Re-exports
 *   foundational types (`cacaosim_types.h`), common cross-module services (`src/common/`),
 *   declarative `.bench` topology/TUI/schematic tools (`src/bench/`), low-level CPU/CUDA physics
 *   contexts (`cacaosim_pupil.h`, `cacaosim_dm.h`, `cacaosim_coro.h`, `cacaosim_pyrwfs.h`,
 *   `cacaosim_shwfs.h`, `cacaosim_zwfs.h`, `cacaosim_fpcam.h`), and `milk` Function Parameter
 *   Structure (`_FPS.h`) runnable descriptors for all 11 compute units including the fused
 *   multi-rate `cacaosim.htsim` engine.
 *
 * Design Rationale & Motivation:
 *   `cacaosim` deliberately splits its headers into fine-grained per-subsystem interfaces alongside
 *   this top-level umbrella header to satisfy two distinct compilation regimes:
 *     1. Internal `.c`/`.cu` physics engines and standalone `milk-fpsexec-cacaosim-*` binaries
 *        include only the minimal leaf headers they require, maximizing incremental build speed,
 *        keeping CUDA `nvcc` translation units free of `milk` CLI headers, and enforcing strict
 *        zero-`CLIcore` symbol isolation.
 *     2. The shared plugin registry (`milkcacaosim.c`), integration test suites, throughput
 *        benchmarks (`cacaosim-benchmark`), and external C/C++ applications linking against
 *        `libmilkcacaosim.so` include `milkcacaosim.h` to obtain the complete simulation API from
 *        a single stable entry point.
 */

#ifndef MILKCACAOSIM_H
#define MILKCACAOSIM_H

/* Core leaf constants, enums, detector config, and aligned memory helpers */
#include "cacaosim_types.h"

/* Optical beam source, inline static WFE, and dichroic beamsplitter FPS runnables */
#include "beam/cacaosim_beamsplit_FPS.h"
#include "beam/cacaosim_mkbeam_FPS.h"
#include "beam/cacaosim_wfe_FPS.h"
#include "beam/cacaosim_wfstat_FPS.h"

/* Declarative .bench parser/serializer, schematic viewer, and interactive TUI configurator */
#include "bench/cacaosim_bench.h"
#include "bench/cacaosim_bench_view.h"
#include "bench/cacaosim_tui.h"

/* Shared infrastructure: [OPD, Amp] beams, CUDA wrappers, detector, masks, PRNG, & pacing */
#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "common/cacaosim_detector.h"
#include "common/cacaosim_mask.h"
#include "common/cacaosim_rng.h"
#include "common/cacaosim_time.h"

/* Coronagraphic focal/pupil mask & LOWFS engine and FPS runnable */
#include "coro/cacaosim_coro.h"
#include "coro/cacaosim_coro_FPS.h"

/* Deformable mirror / tip-tilt hardware dynamics engine and FPS runnable */
#include "dm/cacaosim_dm.h"
#include "dm/cacaosim_dmsim_FPS.h"

/* Polychromatic MFT/FFT focal-plane science camera engine and FPS runnable */
#include "fpcam/cacaosim_fpcam.h"
#include "fpcam/cacaosim_fpcam_FPS.h"

/* Fused zero-hop single- and multi-stage (.bench) HT AO bench simulator runnable */
#include "fused/cacaosim_htsim_FPS.h"

/* Anti-aliased telescope pupil & static NCPA generator engine and FPS runnable */
#include "pupil/cacaosim_mkpupil_FPS.h"
#include "pupil/cacaosim_pupil.h"

/* Wavefront sensor engines and FPS runnables: Pyramid, Shack-Hartmann, and Zernike WFS */
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "pyrwfs/cacaosim_pyrwfs_FPS.h"
#include "shwfs/cacaosim_shwfs.h"
#include "shwfs/cacaosim_shwfs_FPS.h"
#include "zwfs/cacaosim_zwfs.h"
#include "zwfs/cacaosim_zwfs_FPS.h"

#endif /* MILKCACAOSIM_H */
