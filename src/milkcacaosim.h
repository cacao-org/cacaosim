// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milkcacaosim.h
 * @brief   Top-level public header for the milkcacaosim AO hardware/optics simulation plugin
 */

#ifndef MILKCACAOSIM_H
#define MILKCACAOSIM_H

#include "cacaosim_types.h"
#include "beam/cacaosim_beamsplit_FPS.h"
#include "beam/cacaosim_mkbeam_FPS.h"
#include "beam/cacaosim_wfe_FPS.h"
#include "bench/cacaosim_bench.h"
#include "bench/cacaosim_bench_view.h"
#include "bench/cacaosim_tui.h"
#include "common/cacaosim_beam.h"
#include "common/cacaosim_cuda.h"
#include "common/cacaosim_detector.h"
#include "common/cacaosim_mask.h"
#include "common/cacaosim_rng.h"
#include "common/cacaosim_time.h"
#include "coro/cacaosim_coro.h"
#include "coro/cacaosim_coro_FPS.h"
#include "dm/cacaosim_dm.h"
#include "dm/cacaosim_dmsim_FPS.h"
#include "fpcam/cacaosim_fpcam.h"
#include "fpcam/cacaosim_fpcam_FPS.h"
#include "fused/cacaosim_htsim_FPS.h"
#include "pupil/cacaosim_mkpupil_FPS.h"
#include "pupil/cacaosim_pupil.h"
#include "pyrwfs/cacaosim_pyrwfs.h"
#include "pyrwfs/cacaosim_pyrwfs_FPS.h"
#include "shwfs/cacaosim_shwfs.h"
#include "shwfs/cacaosim_shwfs_FPS.h"
#include "zwfs/cacaosim_zwfs.h"
#include "zwfs/cacaosim_zwfs_FPS.h"

#endif /* MILKCACAOSIM_H */
