// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_dag.c
 * @brief   Topological compiler, 32-bin photon budget, and zero-hop CPU/CUDA multi-rate DAG engine
 *
 * Architectural Role:
 *   Implements the graph compilation, physical photometry analysis, resource binding, and
 *   real-time multi-rate execution engine for declarative `.bench` optical benches:
 *     1. Topological Compiler (`topo_sort_and_wire_beams()`): Enforces a strict single-producer
 *        invariant on every named optical beam, verifies that all downstream `in_beam` references
 *        resolve to known producers, and orders nodes via a declaration-order-preserving Kahn
 *        topological sort that rejects optical feedback cycles.
 *     2. 32-Bin Physical Spectral Integrator (`propagate_spectra()`): Synthesizes a calibrated
 *        `0.40..2.40 um` Planck blackbody stellar spectrum scaled to `tel_diam_m`, `cobs`,
 *        `star_mag`, `star_teff`, and `tput_tel`, propagates per-bin `[photons/s]` rates across
 *        reflective coatings, gray/dichroic splitters, and coronagraphic LOWFS pickoffs, and
 *        integrates per-exposure detected photo-events (`eff_flux_ph`) and flux-weighted central
 *        wavelengths (`eff_lambda_um`) for every multi-rate sensor.
 *     3. Zero-FFT Static Analyzer (`cacaosim_bench_analyze()`): Runs steps (1) and (2) plus
 *        multi-rate decimation quantization (`eff_navg`) with zero heap, FFT, GPU, or SHM side
 *        effects so `cacaosim-CTRL` and `cacaosim-bench-view` render instant previews.
 *     4. Runtime Arena, SHM Binder & Stepper (`cacaosim_bench_init()`, `cacaosim_bench_step()`,
 *        `cacaosim_bench_free()`): Binds `ImageStreamIO` `IMGID` streams (or standalone C buffers),
 *        promotes polychromatic `[nlam, nwf, nwf]` beam cubes downstream of broadband coronagraphs,
 *        and advances the entire bench on a single shared `cudaStream_t` or aligned CPU arena.
 *
 * Design Rationale & Motivation:
 *   - Zero-hop on-GPU / in-cache execution: Passing intermediate beams via device pointers
 *     (`d_opd`, `d_amp`) on one `cudaStream_t` eliminates inter-process SHM serialization,
 *     semaphore wakeups, OS context-switch jitter, and PCIe ping-pong across multi-kHz AO stages
 *     while maintaining bit-identical physics to modular `cacaosim.*` FPS units.
 *   - Decoupled microsecond static analysis: Separating `cacaosim_bench_analyze()` from FFTW3f /
 *     cuFFT planning and CUDA initialization allows TUI/CLI tools to re-evaluate photon budgets
 *     interactively in microseconds even on headless systems without a GPU or `MILK_SHM_DIR`.
 *   - Deterministic subframe decimation (`eff_navg`): Accumulating coherent intensity on every
 *     master tick (`*_accum_beam`) and gating noisy detector readout on `subframe_cnt >= eff_navg`
 *     models true finite-exposure turbulence/DM smearing across heterogeneous sensor rates
 *     (e.g. 2 kHz Vis-PyWFS + 500 Hz ZWFS LOWFS + 100 Hz H-band FPCam) without multi-thread locks.
 */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "bench/cacaosim_bench.h"
#include "common/cacaosim_cuda.h"
#include "common/cacaosim_disp.h"
#include "common/cacaosim_mask.h"
#include "pupil/cacaosim_pupil.h"

/**
 * dag_err() - Format a diagnostic message into an optional caller-supplied error buffer
 * @buf: Optional output buffer (safely ignored if NULL or @cap == 0).
 * @cap: Capacity of @buf in bytes including trailing NUL.
 * @fmt: printf-compatible format string.
 *
 * Role:
 *   Captures topology, allocation, mask-loading, or SHM-binding failure reasons for CLI/TUI logs.
 *
 * Rationale:
 *   Centralizes NULL-safe `vsnprintf` formatting without touching global error state.
 */
static void dag_err(char *buf, size_t cap, const char *fmt, ...)
{
    if (buf == NULL || cap == 0)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
}

/**
 * find_beam_name() - Look up a registered optical beam slot index by name
 * @ctx:  Bench context whose `beams[0..n_beams-1]` table is searched (must not be NULL).
 * @name: Beam identifier string to locate (returns -1 if NULL or empty).
 *
 * Role:
 *   Resolves string `in_beam` references on consumer nodes into `O(1)` integer `beams[]` indices
 *   during DAG compilation and checks for duplicate beam producers.
 *
 * Rationale:
 *   Performed strictly once at compile/analyze time (`<= 32` beams) so `cacaosim_bench_step()`
 *   addresses beams purely by integer index.
 *
 * Return: Beam slot index in `[0 .. ctx->n_beams - 1]`, or -1 if not registered.
 */
static int find_beam_name(const cacaosim_bench_ctx_t *ctx, const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    for (uint32_t i = 0; i < ctx->n_beams; i++)
    {
        if (strcmp(ctx->beams[i].name, name) == 0)
        {
            return (int) i;
        }
    }
    return -1;
}

/**
 * register_beam_name() - Register a newly produced optical beam and enforce single-producer safety
 * @ctx:     Bench context being compiled (must not be NULL).
 * @name:    Non-empty beam identifier produced by the entrance aperture or an optical node.
 * @err_buf: Optional diagnostic buffer populated on duplicate producer or capacity overflow.
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Allocates the next slot in `ctx->beams[]`, rejecting any duplicate producer (`-2`) or beam
 *   table overflow (`-2`).
 *
 * Rationale:
 *   Guaranteeing at most one producer per named optical beam ensures unambiguous physical
 *   causality—every beam buffer has a single writer and arbitrarily many downstream readers.
 *
 * Return: New beam index `[0 .. CACAOSIM_BENCH_MAX_BEAMS - 1]`, -1 if empty, or -2 on error.
 */
static int register_beam_name(
    cacaosim_bench_ctx_t *ctx,
    const char           *name,
    char                 *err_buf,
    size_t                err_cap)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    if (find_beam_name(ctx, name) >= 0)
    {
        dag_err(err_buf, err_cap, "Duplicate producer for optical beam '%s'", name);
        return -2;
    }
    if (ctx->n_beams >= CACAOSIM_BENCH_MAX_BEAMS)
    {
        dag_err(err_buf, err_cap, "Exceeded max optical beams (%d)", CACAOSIM_BENCH_MAX_BEAMS);
        return -2;
    }
    int idx = (int) ctx->n_beams++;
    snprintf(ctx->beams[idx].name, sizeof(ctx->beams[idx].name), "%s", name);
    ctx->beams[idx].img_shm.ID = -1;
    ctx->beams[idx].export_shm = 0;
    return idx;
}

/**
 * init_node_imgids() - Mark all ImageStreamIO `IMGID` handles on a runtime node as disconnected
 * @nd: Runtime bench node struct whose `img_*` fields have just been zeroed (must not be NULL).
 *
 * Role:
 *   Sets every `IMGID.ID` sentinel to `-1` immediately after `memset(nd, 0, ...)`.
 *
 * Rationale:
 *   In `ImageStreamIO`, `ID == 0` is a valid slot index in the process image table (`-1` denotes
 *   unbound); initializing all `ID` fields to `-1` prevents `cacaosim_bench_free()` from
 *   accidentally closing `image[0]` during partial-init rollback or pure-C teardown.
 */
static void init_node_imgids(cacaosim_bench_node_t *nd)
{
    nd->img_cmd.ID           = -1;
    nd->img_opd.ID           = -1;
    nd->img_state.ID         = -1;
    nd->img_flat.ID          = -1;
    nd->img_wfe.ID           = -1;
    nd->img_cam.ID           = -1;
    nd->img_long.ID          = -1;
    nd->img_ncpa.ID          = -1;
    nd->img_lyot.ID          = -1;
    nd->img_pre_lyot.ID      = -1;
    nd->img_pre_lyot_long.ID = -1;
    nd->img_ncpa_refl.ID     = -1;
    nd->img_ncpa_trans.ID    = -1;
    nd->img_pup_refl.ID      = -1;
    nd->img_pup_trans.ID     = -1;
}

/* Parse comma/space-separated chromatic anchor wavelengths [um] (returns 0..max_n) */
static uint32_t parse_chrom_lambdas_csv(const char *csv, float *out_um, uint32_t max_n)
{
    if (csv == NULL || out_um == NULL || max_n == 0u)
    {
        return 0u;
    }
    uint32_t    cnt = 0u;
    const char *p   = csv;
    while (*p != '\0' && cnt < max_n)
    {
        while (*p == ' ' || *p == '\t' || *p == ',' || *p == ';')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }
        char  *end = NULL;
        double v   = strtod(p, &end);
        if (end == p)
        {
            break;
        }
        p = end;
        if (v > 20.0)
        {
            v *= 1.0e-3;
        }
        if (v >= 0.2 && v <= 20.0)
        {
            out_um[cnt++] = (float) v;
        }
    }
    return cnt;
}

/* Propagate dispersion descriptor through a static 2D registration preserving gen when steady */
static void bench_prop_disp(
    const cacaosim_disp_t *in,
    const cacaosim_reg_t  *reg,
    cacaosim_disp_t       *out)
{
    if (out == NULL)
    {
        return;
    }
    if (in == NULL || !in->active)
    {
        uint64_t g = (in != NULL) ? in->gen : 0ULL;
        if (out->active || out->gen != g)
        {
            memset(out, 0, sizeof(*out));
            out->active = 0;
            out->gen    = g;
        }
        return;
    }
    if (cacaosim_reg_is_identity(reg))
    {
        *out = *in;
        return;
    }
    if (!out->active || out->gen != in->gen)
    {
        cacaosim_disp_apply_reg(in, reg, out);
        out->gen = in->gen;
    }
}

/* Update cached atmospheric dispersion Chebyshev fit only when parameters actually change */
static void bench_update_atm_disp(cacaosim_bench_ctx_t *ctx)
{
    cacaosim_atm_disp_cfg_t cur = ctx->cfg.atm_disp;
    cur.disp_pa_deg             = (float) ctx->atm_disp_pa_curr_deg;
    if (ctx->cached_atm_valid && cur.enable == ctx->cached_atm_cfg.enable &&
        fabsf(cur.zenith_deg - ctx->cached_atm_cfg.zenith_deg) <= 1e-6f &&
        fabsf(cur.disp_pa_deg - ctx->cached_atm_cfg.disp_pa_deg) <= 1e-6f &&
        fabsf(cur.pressure_hpa - ctx->cached_atm_cfg.pressure_hpa) <= 1e-4f &&
        fabsf(cur.temp_c - ctx->cached_atm_cfg.temp_c) <= 1e-4f &&
        fabsf(cur.rh_pct - ctx->cached_atm_cfg.rh_pct) <= 1e-3f &&
        fabsf(cur.ref_lam_um - ctx->cached_atm_cfg.ref_lam_um) <= 1e-5f)
    {
        return;
    }
    cacaosim_disp_set_atm(&ctx->cached_atm_disp, &cur);
    ctx->cached_atm_cfg   = cur;
    ctx->cached_atm_valid = 1;
}

/* Warn once at init if any broadband sensor has per-channel dispersion step > 0.5 lambda/D */
static void bench_warn_chromatic_sampling(const cacaosim_bench_ctx_t *ctx)
{
    if (ctx == NULL || !ctx->cfg.atm_disp.enable || ctx->cfg.atm_disp.zenith_deg <= 1.0f)
    {
        return;
    }
    const double d_tel = (ctx->cfg.tel_diam_m > 0.01f) ? (double) ctx->cfg.tel_diam_m : 8.0;
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd  = &ctx->nodes[i];
        const cacaosim_beam_slot_t  *bin = &ctx->beams[nd->in_idx];
        if (!bin->disp.active)
        {
            continue;
        }
        double lam = 0.0, bw = 0.0;
        int    nch = 1;
        if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            lam = nd->cfg.coro.lambda_um;
            bw  = nd->cfg.coro.bw_frac;
            nch = (nd->coro_ctx.eff_nlam >= 1) ? nd->coro_ctx.eff_nlam : 1;
        }
        else if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS && nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
        {
            lam = (nd->eff_lambda_um > 0.1f) ? (double) nd->eff_lambda_um : 1.0;
            bw  = (double) nd->cfg.bw_frac;
            nch = (nd->eff_nlam >= 1) ? nd->eff_nlam : 1;
        }
        else
        {
            continue;
        }
        if (bw <= 1e-4)
        {
            continue;
        }
        double l0       = lam * (1.0 - 0.5 * bw);
        double l1       = lam * (1.0 + 0.5 * bw);
        double span_mas = cacaosim_disp_band_span_mas(&bin->disp, l0, l1);
        double ld_mas   = (lam * 1.0e-6 / d_tel) * 206264806.24709636;
        double step_mas = (nch > 1) ? (span_mas / (double) (nch - 1)) : span_mas;
        if (step_mas > 0.5 * ld_mas)
        {
            int rec = (int) ceil(span_mas / (0.5 * ld_mas)) + 1;
            fprintf(stderr,
                    "[cacaosim] WARNING: node '%s' (%.2fum, BW=%.0f%%, nlambda=%d) has "
                    "dispersion span %.1f mas (%.2f l/D per step > 0.5 l/D); recommend "
                    "nlambda >= %d\n",
                    nd->cfg.id, lam, bw * 100.0, nch, span_mas, step_mas / ld_mas, rec);
        }
    }
}

/**
 * free_imgid_if_valid() - Disconnect an ImageStreamIO `IMGID` handle if bound and reset to `-1`
 * @img: Pointer to an `IMGID` handle (must not be NULL).
 *
 * Role:
 *   Idempotently releases a connected SHM stream mapping and restores the `-1` sentinel.
 *
 * Rationale:
 *   Allows `cacaosim_bench_free()` to safely clean up partially initialized nodes without
 *   double-freeing unbound handles.
 */
static void free_imgid_if_valid(IMGID *img)
{
    if (img->ID != -1)
    {
        imgid_free(img);
        img->ID = -1;
    }
}

/**
 * connect_opt_2df32() - Opportunistically attach to an existing `[nwf x nwf]` float32 SHM stream
 * @name:    Optional SHM stream name (returns NULL immediately if NULL or empty).
 * @nwf:     Required 2D grid dimension `[nwf, nwf]`.
 * @out_img: Output `IMGID` handle initialized to `-1` and populated if connected.
 *
 * Role:
 *   Connects optional static calibration maps (`in_pup`, `in_ncpa`, `in_flat`, `ncpa_refl/trans`,
 *   `pup_refl/trans`) and validates that their datatype (`_DATATYPE_FLOAT`) and `[nwf x nwf]`
 *   dimensions match the bench grid before returning the raw pixel pointer.
 *
 * Rationale:
 *   Missing or mismatched optional calibration streams gracefully fall back to `NULL` (identity /
 *   zero map) rather than aborting bench startup, while still tracking `*out_img` for clean
 *   unmapping on teardown.
 *
 * Return: Read-only host `float *` array pointer if valid `[nwf x nwf]` float32, else NULL.
 */
static const float *connect_opt_2df32(const char *name, uint32_t nwf, IMGID *out_img)
{
    out_img->ID = -1;
    if (name == NULL || name[0] == '\0')
    {
        return NULL;
    }
    *out_img = stream_connect(name);
    if (out_img->ID != -1 && out_img->md->datatype == _DATATYPE_FLOAT &&
        out_img->md->size[0] == nwf && out_img->md->size[1] == nwf)
    {
        return out_img->im->array.F;
    }
    return NULL;
}

/**
 * load_opt_coro_mask() - Load an optional 2D/3D coronagraph mask from a FITS file or SHM stream
 * @spec:        Mask specifier string (FITS path, SHM stream name, or empty `""`).
 * @connect_shm: Non-zero when SHM mask attachment is enabled; 0 in pure-C file-only mode.
 * @out:         Destination `cacaosim_mask_cube_t` zeroed on entry and populated if loaded.
 * @err_buf:     Optional diagnostic buffer populated on missing/corrupt FITS file.
 * @err_cap:     Capacity of @err_buf in bytes.
 *
 * Role:
 *   Loads custom pupil, focal-plane, or Lyot amplitude/phase masks (`[Nx, Ny]` or polychromatic
 *   `[Nx, Ny, Nlambda]`) during coronagraph node initialization.
 *
 * Rationale:
 *   Explicit FITS file paths fail hard on I/O errors (surfacing typos immediately), whereas named
 *   SHM streams are loaded best-effort when `connect_shm == 1` so analytic fallback masks remain
 *   usable before external mask generators publish streams.
 *
 * Return: 0 on success (or unused/optional skip), -1 on FITS load error.
 */
static int load_opt_coro_mask(
    const char           *spec,
    int                   connect_shm,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap)
{
    memset(out, 0, sizeof(*out));
    if (spec == NULL || spec[0] == '\0')
    {
        return 0;
    }
    if (cacaosim_mask_is_fits_path(spec))
    {
        return cacaosim_mask_load(spec, 0, out, err_buf, err_cap);
    }
    if (connect_shm)
    {
        (void) cacaosim_mask_load(spec, 1, out, NULL, 0);
    }
    return 0;
}

/**
 * upgrade_beam_poly() - Expand a beam slot from monochromatic `[1, nwf, nwf]` to `[nlam, nwf, nwf]`
 * @ctx:     Active bench context owning CPU/CUDA flags and `npix_wf` (must not be NULL).
 * @b:       Target beam slot to widen in place (must not be NULL).
 * @nlam:    Desired number of coherent wavelength planes (clamped to `[1..CORO_MAX_LAMBDA]`).
 * @lam_um:  Optional array `[nlam]` of channel center wavelengths `[um]` to record on @b.
 * @err_buf: Optional diagnostic buffer populated on host/CUDA allocation failure.
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Reallocates `b->opd` / `b->amp` (and `b->d_opd` / `b->d_amp` when `ctx->use_cuda == 1`) to
 *   hold `nlam * npix_wf` floats whenever an upstream broadband coronagraph (or a relay optic
 *   downstream of one) emits a multi-wavelength pupil field.
 *
 * Rationale:
 *   All beams start at `nlam = 1` to minimize memory footprint and kernel bandwidth on common
 *   achromatic pupil paths, and widen lazily along topological order only on branches downstream
 *   of polychromatic coronagraphs.
 *
 * Return: 0 on success, -1 on host or GPU out-of-memory.
 */
static int upgrade_beam_poly(
    cacaosim_bench_ctx_t *ctx,
    cacaosim_beam_slot_t *b,
    uint32_t              nlam,
    const float          *lam_um,
    char                 *err_buf,
    size_t                err_cap)
{
    if (nlam < 1u)
    {
        nlam = 1u;
    }
    if (nlam > CACAOSIM_CORO_MAX_LAMBDA)
    {
        nlam = CACAOSIM_CORO_MAX_LAMBDA;
    }
    if (lam_um != NULL)
    {
        for (uint32_t c = 0; c < nlam; c++)
        {
            b->lam_um[c] = lam_um[c];
        }
    }
    if (nlam <= b->nlam)
    {
        return 0;
    }

    size_t tot = (size_t) nlam * ctx->npix_wf;
    if (b->owns_buf)
    {
        free(b->opd);
        free(b->amp);
    }
    b->opd      = (float *) cacaosim_aligned_calloc(tot, sizeof(float));
    b->amp      = (float *) cacaosim_aligned_calloc(tot, sizeof(float));
    b->owns_buf = 1;
    if (!b->opd || !b->amp)
    {
        dag_err(err_buf, err_cap, "Out of memory allocating %u-lambda beam '%s'", nlam, b->name);
        return -1;
    }
    if (ctx->use_cuda)
    {
        cacaosim_cuda_free_f32(b->d_opd);
        cacaosim_cuda_free_f32(b->d_amp);
        b->d_opd = cacaosim_cuda_alloc_f32(tot);
        b->d_amp = cacaosim_cuda_alloc_f32(tot);
        if (!b->d_opd || !b->d_amp)
        {
            dag_err(err_buf, err_cap, "CUDA alloc failed for %u-lambda beam '%s'", nlam, b->name);
            return -1;
        }
    }
    b->nlam = nlam;
    return 0;
}

/**
 * topo_sort_and_wire_beams() - Validate beam connectivity and topologically sort optical nodes
 * @ctx:     Destination bench context receiving `beams[]` and sorted `nodes[]` (must not be NULL).
 * @cfg:     Source parsed bench configuration (must not be NULL).
 * @err_buf: Optional diagnostic buffer populated on missing/duplicate beam or optical cycle.
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Compiles string-named `.bench` sections into an executable integer-indexed DAG in three passes:
 *     - Pass 1: Registers the root telescope beam at `beams[0]` (`beam_prod[0] = -1`) and registers
 *       every node-produced beam (`out_beam`, `out_refl`, `out_trans`, `out_lowfs`), recording its
 *       unique producer node index in `beam_prod[b]`.
 *     - Pass 2: Resolves each node's `in_beam` to `raw_in[i]` (failing on unknown/dangling names)
 *       and sets its initial dependency in-degree `in_deg[i] = 1` if produced by another node or
 *       `0` if fed directly by `beams[0]`.
 *     - Pass 3: Executes Kahn's topological sort, scanning `0 .. n_nodes-1` each step to pick the
 *       lowest-index ready node (`in_deg == 0`), copying it into `ctx->nodes[sorted_cnt]`, and
 *       decrementing `in_deg` on all direct consumers of its output beam(s); if any unvisited
 *       nodes remain with `in_deg > 0`, reports an optical cycle error.
 *
 * Rationale:
 *   Always picking the lowest-index ready node makes Kahn's sort *stable* with respect to `.bench`
 *   declaration order whenever parallel optical arms are independent, while strictly guaranteeing
 *   producer-before-consumer execution even if sections were written out-of-order in the file.
 *
 * Return: 0 on valid acyclic DAG, -1 on missing input, duplicate producer, or cycle error.
 */
static int topo_sort_and_wire_beams(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap)
{
    /* Pass 1: Register root entrance beam at slot 0 and all node-produced output beams */
    const char *src_name = (cfg->out_beam[0] != '\0') ? cfg->out_beam : "b_pup";
    if (register_beam_name(ctx, src_name, err_buf, err_cap) < 0)
    {
        return -1;
    }

    int32_t beam_prod[CACAOSIM_BENCH_MAX_BEAMS];
    for (int i = 0; i < CACAOSIM_BENCH_MAX_BEAMS; i++)
    {
        beam_prod[i] = -1;
    }

    int32_t raw_out[CACAOSIM_BENCH_MAX_NODES];
    int32_t raw_refl[CACAOSIM_BENCH_MAX_NODES];
    int32_t raw_trans[CACAOSIM_BENCH_MAX_NODES];
    int32_t raw_lowfs[CACAOSIM_BENCH_MAX_NODES];

    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        const cacaosim_node_cfg_t *nc = &cfg->nodes[i];
        raw_out[i]                    = -1;
        raw_refl[i]                   = -1;
        raw_trans[i]                  = -1;
        raw_lowfs[i]                  = -1;

        if (nc->in_beam[0] == '\0')
        {
            dag_err(err_buf, err_cap, "Node '%s' missing input beam ('in = ...')", nc->id);
            return -1;
        }

        if (nc->kind == CACAOSIM_NODE_DM || nc->kind == CACAOSIM_NODE_CORO ||
            nc->kind == CACAOSIM_NODE_WFE || nc->kind == CACAOSIM_NODE_ADC)
        {
            if (nc->out_beam[0] == '\0')
            {
                dag_err(err_buf, err_cap, "Node '%s' missing output beam ('out = ...')", nc->id);
                return -1;
            }
            int b = register_beam_name(ctx, nc->out_beam, err_buf, err_cap);
            if (b < 0)
            {
                return -1;
            }
            raw_out[i]   = b;
            beam_prod[b] = (int32_t) i;

            if (nc->kind == CACAOSIM_NODE_CORO && nc->out_lowfs[0] != '\0')
            {
                int bl = register_beam_name(ctx, nc->out_lowfs, err_buf, err_cap);
                if (bl < 0)
                {
                    return -1;
                }
                raw_lowfs[i]  = bl;
                beam_prod[bl] = (int32_t) i;
            }
        }
        else if (nc->kind == CACAOSIM_NODE_SPLITTER)
        {
            if (nc->out_refl[0] == '\0' && nc->out_trans[0] == '\0')
            {
                dag_err(err_buf, err_cap, "Splitter '%s' needs out_refl and/or out_trans", nc->id);
                return -1;
            }
            if (nc->out_refl[0] != '\0')
            {
                int br = register_beam_name(ctx, nc->out_refl, err_buf, err_cap);
                if (br < 0)
                {
                    return -1;
                }
                raw_refl[i]   = br;
                beam_prod[br] = (int32_t) i;
            }
            if (nc->out_trans[0] != '\0')
            {
                int bt = register_beam_name(ctx, nc->out_trans, err_buf, err_cap);
                if (bt < 0)
                {
                    return -1;
                }
                raw_trans[i]  = bt;
                beam_prod[bt] = (int32_t) i;
            }
        }
    }

    /* Pass 2: Resolve consumer input beams and compute upstream producer in-degrees (0 or 1) */
    int32_t raw_in[CACAOSIM_BENCH_MAX_NODES];
    int32_t in_deg[CACAOSIM_BENCH_MAX_NODES];
    for (uint32_t i = 0; i < cfg->n_nodes; i++)
    {
        int bi = find_beam_name(ctx, cfg->nodes[i].in_beam);
        if (bi < 0)
        {
            dag_err(err_buf, err_cap, "Node '%s' references unknown beam '%s'", cfg->nodes[i].id,
                    cfg->nodes[i].in_beam);
            return -1;
        }
        raw_in[i] = bi;
        in_deg[i] = (beam_prod[bi] >= 0) ? 1 : 0;
    }

    /* Pass 3: Kahn's topological sort preserving declaration order among ready nodes */
    uint8_t visited[CACAOSIM_BENCH_MAX_NODES] = { 0 };
    uint32_t sorted_cnt                       = 0;

    while (sorted_cnt < cfg->n_nodes)
    {
        int found = -1;
        for (uint32_t i = 0; i < cfg->n_nodes; i++)
        {
            if (!visited[i] && in_deg[i] == 0)
            {
                found = (int) i;
                break;
            }
        }
        if (found < 0)
        {
            dag_err(err_buf, err_cap, "Optical cycle detected in .bench beam graph");
            return -1;
        }

        visited[found]                 = 1;
        cacaosim_bench_node_t *dst     = &ctx->nodes[sorted_cnt];
        memset(dst, 0, sizeof(*dst));
        init_node_imgids(dst);
        dst->cfg         = cfg->nodes[found];
        dst->in_idx      = raw_in[found];
        dst->out_idx     = raw_out[found];
        dst->refl_idx    = raw_refl[found];
        dst->trans_idx   = raw_trans[found];
        dst->lowfs_idx   = raw_lowfs[found];
        dst->dm_slot     = -1;
        dst->sensor_slot = -1;
        sorted_cnt++;

        /* Unlock downstream nodes whose upstream beam producer just completed */
        for (uint32_t j = 0; j < cfg->n_nodes; j++)
        {
            if (!visited[j] && beam_prod[raw_in[j]] == found)
            {
                in_deg[j]--;
            }
        }
    }

    ctx->n_nodes = sorted_cnt;
    return 0;
}

/**
 * propagate_spectra() - Propagate 32-bin stellar photon rates and integrate per-sensor exposure
 *                       flux
 * @ctx: Topologically sorted bench context with `eff_navg` already resolved (must not be NULL).
 *
 * Role:
 *   1. Populates `ctx->beams[0].spec` (`0.40..2.40 um` in 32 bins) from `[bench]` aperture area,
 *      central obscuration, V-band Vega magnitude, blackbody `star_teff`, and `tput_tel`;
 *   2. Walks `ctx->nodes[]` in topological order—multiplying by `tput` across DM/WFE surfaces,
 *      splitting bins conservatively across gray/LP/SP/BP dichroics (`cacaosim_spec_split()`),
 *      scaling coronagraph science (`tput`) and LOWFS pickoff (`tput * lowfs_refl^2`) arms, and
 *      integrating sensor passband flux over `dt_exp = eff_navg * dt_master_s` with detector `qe`.
 *
 * Rationale:
 *   Evaluating steady-state spectral photon budgets analytically over 32 wavelength bins requires
 *   zero 2D FFTs yet captures blackbody color terms, dichroic roll-off losses, multi-stage beam
 *   splitting, and sensor exposure cadences accurately for both runtime noise scaling and TUI
 *   photon-budget displays.
 */
static void propagate_spectra(cacaosim_bench_ctx_t *ctx)
{
    cacaosim_spec_star(&ctx->beams[0].spec, ctx->cfg.tel_diam_m, ctx->cfg.cobs, ctx->cfg.star_mag,
                       ctx->cfg.star_teff, ctx->cfg.tput_tel);
    cacaosim_disp_set_atm(&ctx->beams[0].disp, &ctx->cfg.atm_disp);

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd      = &ctx->nodes[i];
        const cacaosim_spec_t *in_spec = &ctx->beams[nd->in_idx].spec;
        const cacaosim_disp_t *in_disp = &ctx->beams[nd->in_idx].disp;

        if (nd->cfg.kind == CACAOSIM_NODE_DM || nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            cacaosim_spec_scale(in_spec, nd->cfg.tput, &ctx->beams[nd->out_idx].spec);
            bench_prop_disp(in_disp, &nd->cfg.reg, &ctx->beams[nd->out_idx].disp);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_ADC)
        {
            cacaosim_adc_ctx_t tmp_adc;
            cacaosim_adc_init(&tmp_adc, &nd->cfg.adc, &ctx->cfg.atm_disp);
            cacaosim_adc_step(&tmp_adc, 0.0, &ctx->cfg.atm_disp, NULL, in_disp, in_spec,
                              &ctx->beams[nd->out_idx].disp, &ctx->beams[nd->out_idx].spec);
            nd->adc_ctx = tmp_adc;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            cacaosim_spec_t *r_sp = (nd->refl_idx >= 0) ? &ctx->beams[nd->refl_idx].spec : NULL;
            cacaosim_spec_t *t_sp = (nd->trans_idx >= 0) ? &ctx->beams[nd->trans_idx].spec : NULL;
            cacaosim_spec_split(in_spec, &nd->cfg.split, r_sp, t_sp);
            if (nd->refl_idx >= 0)
            {
                bench_prop_disp(in_disp, &nd->cfg.reg_refl, &ctx->beams[nd->refl_idx].disp);
            }
            if (nd->trans_idx >= 0)
            {
                bench_prop_disp(in_disp, &nd->cfg.reg_trans, &ctx->beams[nd->trans_idx].disp);
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            cacaosim_spec_scale(in_spec, nd->cfg.tput, &ctx->beams[nd->out_idx].spec);
            cacaosim_disp_zero(&ctx->beams[nd->out_idx].disp);
            if (nd->lowfs_idx >= 0)
            {
                /* lowfs_refl amplitude weight is applied directly inside coro lo_amp */
                cacaosim_spec_scale(in_spec, nd->cfg.tput, &ctx->beams[nd->lowfs_idx].spec);
                cacaosim_disp_zero(&ctx->beams[nd->lowfs_idx].disp);
            }
        }
        else
        {
            float   lam      = 0.75f;
            float   flx      = 100000.0f;
            int32_t cfg_nlam = 0;
            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                lam      = nd->cfg.pyrwfs.lambda_um;
                flx      = nd->cfg.pyrwfs.det.flux_ph;
                cfg_nlam = nd->cfg.pyrwfs.nlambda;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                lam      = nd->cfg.shwfs.lambda_um;
                flx      = nd->cfg.shwfs.det.flux_ph;
                cfg_nlam = nd->cfg.shwfs.nlambda;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                lam      = nd->cfg.zwfs.lambda_um;
                flx      = nd->cfg.zwfs.det.flux_ph;
                cfg_nlam = nd->cfg.zwfs.nlambda;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                lam      = nd->cfg.fpcam.lambda_um;
                flx      = nd->cfg.fpcam.det.flux_ph;
                cfg_nlam = nd->cfg.fpcam.nlambda;
            }

            /* Exposure duration spans eff_navg master ticks; bypass when use_beam_flux == 0 */
            float dt_exp      = (float) nd->eff_navg * ctx->dt_master_s;
            nd->eff_nlam      = cacaosim_resolve_nlambda(cfg_nlam, 1u, nd->cfg.bw_frac);
            nd->eff_lambda_um = lam;
            if (nd->cfg.use_beam_flux && flx >= 0.0f)
            {
                nd->eff_flux_ph = cacaosim_spec_sample_band(
                    in_spec, lam, nd->cfg.bw_frac, nd->eff_nlam, dt_exp, nd->cfg.qe,
                    nd->chan_lam_um, nd->chan_weight, &nd->eff_lambda_um);
            }
            else
            {
                (void) cacaosim_spec_sample_band(
                    NULL, lam, nd->cfg.bw_frac, nd->eff_nlam, dt_exp, nd->cfg.qe, nd->chan_lam_um,
                    nd->chan_weight, &nd->eff_lambda_um);
                nd->eff_flux_ph = flx;
            }
        }
    }
}

/**
 * cacaosim_bench_analyze() - Compile DAG topology, multi-rate cadences, and 32-bin spectra without
 *                            allocating pixel buffers, FFT plans, or SHM streams
 * @ctx:     Destination bench context populated with static analysis metadata (must not be NULL).
 * @cfg:     Read-only parsed bench configuration to analyze (must not be NULL).
 * @err_buf: Optional diagnostic buffer populated on topology or capacity errors (may be NULL).
 * @err_cap: Capacity of @err_buf in bytes.
 *
 * Role:
 *   Validates graph connectivity and acyclicity, resolves `eff_navg` from `rate_hz` / `navg`,
 *   populates `dm_node_idx[]` and `sensor_node_idx[]`, derives detector pixel dimensions (`ncam`),
 *   and computes every beam's 32-bin photon spectrum and every sensor's `eff_flux_ph`.
 *
 * Rationale:
 *   Completes in `< 0.1 ms` with zero dynamic memory allocation or GPU/SHM driver calls, enabling
 *   flicker-free live topology and photometry updates inside `cacaosim-CTRL` and
 *   `cacaosim-bench-view`.
 *
 * Return: 0 on success, -1 on validation or topology failure.
 */
/**
 * resolve_bench_timing() - Compute effective oversampled master clock and per-node cadences
 * @ctx: Topologically sorted bench context (must not be NULL).
 *
 * Role:
 *   Derives `eff_oversample`, `eff_master_hz`, and `dt_master_s` from `[bench]`, resolves DM
 *   delays (`eff_delay_ticks` from `delay_sec` or `delay_frames * eff_oversample`), quantizes
 *   sensor exposure ticks (`eff_navg`), computes subframe accumulation strides (`eff_stride` +
 *   staggered offsets), and initializes fractional exposure clock phases (`clock_phase_init`) from
 *   `clock_offset_us`.
 */
static void resolve_bench_timing(cacaosim_bench_ctx_t *ctx)
{
    ctx->eff_oversample = (ctx->cfg.oversample >= 1u) ? ctx->cfg.oversample : 1u;
    float base_hz       = (ctx->cfg.master_hz > 1.0f) ? ctx->cfg.master_hz : 2000.0f;
    ctx->eff_master_hz  = base_hz * (float) ctx->eff_oversample;
    ctx->dt_master_s    = 1.0f / ctx->eff_master_hz;
    ctx->calib_dm_node  = -1;

    uint32_t strided_idx = 0u;
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            nd->eff_delay_ticks =
                (nd->cfg.dm.delay_sec > 0.0f)
                    ? (nd->cfg.dm.delay_sec / ctx->dt_master_s)
                    : (nd->cfg.dm.delay_frames * (float) ctx->eff_oversample);
            continue;
        }
        if (nd->cfg.kind < CACAOSIM_NODE_PYRWFS || nd->cfg.kind > CACAOSIM_NODE_FPCAM)
        {
            continue;
        }

        uint32_t base_navg = (nd->cfg.navg >= 1u) ? nd->cfg.navg : 1u;
        uint32_t navg      = base_navg * ctx->eff_oversample;
        if (nd->cfg.rate_hz > 0.1f && ctx->eff_master_hz > 0.1f)
        {
            long r = lroundf(ctx->eff_master_hz / nd->cfg.rate_hz);
            navg   = (r >= 1L) ? (uint32_t) r : 1u;
        }
        nd->eff_navg = navg;

        uint32_t stride = 1u;
        if (nd->cfg.sim_hz > 0.1f && ctx->eff_master_hz > 0.1f)
        {
            long st = lroundf(ctx->eff_master_hz / nd->cfg.sim_hz);
            stride  = (st >= 1L) ? (uint32_t) st : 1u;
        }
        else if (nd->cfg.subframe_stride >= 1u)
        {
            stride = nd->cfg.subframe_stride;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && ctx->eff_oversample > 1u)
        {
            stride = ctx->eff_oversample;
        }
        if (stride > navg)
        {
            stride = navg;
        }
        if (nd->cfg.subframe_stride == 0u && nd->cfg.sim_hz <= 0.1f)
        {
            while (stride > 1u && (navg % stride) != 0u)
            {
                stride--;
            }
        }
        nd->eff_stride   = stride;
        nd->stagger_tick = (stride > 1u) ? (strided_idx++ % stride) : 0u;

        double t_exp   = (double) navg * (double) ctx->dt_master_s;
        double off_cyc = (t_exp > 1.0e-12)
                             ? -((double) nd->cfg.clock_offset_us * 1.0e-6) / t_exp
                             : 0.0;
        off_cyc        = off_cyc - floor(off_cyc);
        if (off_cyc < 1.0e-12 || off_cyc > 1.0 - 1.0e-12)
        {
            off_cyc = 0.0;
        }
        nd->clock_phase_init  = off_cyc;
        nd->clock_phase       = off_cyc;
        nd->prev_subframe_cnt = 0u;
    }
}

int cacaosim_bench_analyze(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    char                       *err_buf,
    size_t                      err_cap)
{
    if (ctx == NULL || cfg == NULL)
    {
        dag_err(err_buf, err_cap, "NULL bench context or config");
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->img_pup.ID  = -1;
    ctx->img_ncpa.ID = -1;
    ctx->img_wf.ID   = -1;
    ctx->img_amp.ID  = -1;
    ctx->cfg         = *cfg;
    ctx->nwf         = cfg->nwf;
    ctx->npix_wf     = (uint64_t) cfg->nwf * cfg->nwf;

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    ctx->use_cuda = (want_cuda && cacaosim_cuda_available()) ? 1 : 0;

    if (topo_sort_and_wire_beams(ctx, cfg, err_buf, err_cap) != 0)
    {
        return -1;
    }

    resolve_bench_timing(ctx);

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];

        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            if (ctx->n_dms >= CACAOSIM_BENCH_MAX_DMS)
            {
                dag_err(err_buf, err_cap, "Exceeded max DMs (%d)", CACAOSIM_BENCH_MAX_DMS);
                return -1;
            }
            nd->dm_slot                    = (int32_t) ctx->n_dms;
            ctx->dm_node_idx[ctx->n_dms++] = (int32_t) i;
            if (nd->cfg.dm.dm_diam <= 0.0f)
            {
                nd->cfg.dm.dm_diam = cfg->pup_diam;
            }
            if (nd->cfg.dm.dm_mode == 1)
            {
                nd->cfg.dm.nact_x = 2u;
                nd->cfg.dm.nact_y = 1u;
            }
        }
        else if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS && nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
        {
            if (ctx->n_sensors >= CACAOSIM_BENCH_MAX_SENSORS)
            {
                dag_err(err_buf, err_cap, "Exceeded max sensors (%d)", CACAOSIM_BENCH_MAX_SENSORS);
                return -1;
            }
            nd->sensor_slot                        = (int32_t) ctx->n_sensors;
            ctx->sensor_node_idx[ctx->n_sensors++] = (int32_t) i;

            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                nd->ncam      = nd->cfg.pyrwfs.ncam;
                nd->out_dtype = nd->cfg.pyrwfs.det.out_dtype;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                nd->ncam      = (uint32_t) (nd->cfg.shwfs.n_sub * nd->cfg.shwfs.pix_per_sub);
                nd->out_dtype = nd->cfg.shwfs.det.out_dtype;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                nd->ncam      = nd->cfg.zwfs.ncam;
                nd->out_dtype = nd->cfg.zwfs.det.out_dtype;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                nd->ncam      = nd->cfg.fpcam.ncam;
                nd->out_dtype = nd->cfg.fpcam.det.out_dtype;
            }
        }
    }

    propagate_spectra(ctx);
    return 0;
}

/**
 * cacaosim_bench_init() - Compile, allocate, and bind a runnable multi-stage CPU/CUDA bench DAG
 * @ctx:         Destination runtime bench context zeroed and initialized (must not be NULL).
 * @cfg:         Read-only parsed bench configuration to compile (must not be NULL).
 * @ext_pup:     Optional caller-owned `[nwf * nwf]` pupil amplitude array (overrides SHM/fallback
 *               when non-NULL).
 * @connect_shm: Non-zero to bind/create `ImageStreamIO` SHM streams; 0 for standalone pure-C mode.
 * @err_buf:     Optional diagnostic buffer populated on failure (may be NULL).
 * @err_cap:     Capacity of @err_buf in bytes.
 *
 * Role:
 *   Performs end-to-end transactional setup of the optical bench:
 *     1. Compiles DAG topology via `topo_sort_and_wire_beams()`;
 *     2. Creates a dedicated non-blocking `cudaStream_t` when CUDA is enabled and available;
 *     3. Resolves the entrance pupil (`ext_pup` > SHM `in_pup` > anti-aliased `fallback_pup`) and
 *        uploads static entrance maps to GPU VRAM;
 *     4. Allocates `[1, nwf, nwf]` host/device `[OPD_um, Amplitude]` buffers for all named beams;
 *     5. Resolves multi-rate `eff_navg` cadences and propagates 32-bin stellar spectra;
 *     6. Initializes every node in topological order, binding SHM streams (or host buffers),
 *        loading custom coronagraph FITS/SHM masks, synthesizing missing `[m]` static WFE streams,
 *        propagating polychromatic `[nlam, nwf, nwf]` beam upgrades downstream of broadband
 *        coronagraphs, and synchronizing the CUDA stream before returning.
 *
 * Rationale:
 *   Initializing nodes strictly in topological order allows polychromatic channel counts (`nlam`)
 *   and wavelength grids (`lam_um[]`) emitted by an upstream coronagraph to cascade naturally
 *   through intermediate DMs, splitters, and WFEs into downstream `fpcam` broadband MFT/FFT plans,
 *   while calling `cacaosim_bench_free(ctx)` on any failure prevents partial resource leaks.
 *
 * Return: 0 on success, -1 on topology, memory, mask I/O, or SHM error.
 */
int cacaosim_bench_init(
    cacaosim_bench_ctx_t       *ctx,
    const cacaosim_bench_cfg_t *cfg,
    const float                *ext_pup,
    int32_t                     connect_shm,
    char                       *err_buf,
    size_t                      err_cap)
{
    if (ctx == NULL || cfg == NULL)
    {
        dag_err(err_buf, err_cap, "NULL bench context or config");
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->img_pup.ID  = -1;
    ctx->img_ncpa.ID = -1;
    ctx->img_wf.ID   = -1;
    ctx->img_amp.ID  = -1;
    ctx->cfg         = *cfg;
    ctx->connect_shm = connect_shm;
    ctx->nwf         = cfg->nwf;
    ctx->npix_wf     = (uint64_t) cfg->nwf * cfg->nwf;

    if (topo_sort_and_wire_beams(ctx, cfg, err_buf, err_cap) != 0)
    {
        return -1;
    }

    resolve_bench_timing(ctx);

    int want_cuda = (cfg->backend == CACAOSIM_BACKEND_AUTO ||
                     cfg->backend == CACAOSIM_BACKEND_CUDA);
    if (want_cuda && cacaosim_cuda_available())
    {
        ctx->cuda_stream = cacaosim_cuda_stream_create();
        if (ctx->cuda_stream != NULL)
        {
            ctx->use_cuda = 1;
        }
    }
    int32_t eff_backend = ctx->use_cuda ? CACAOSIM_BACKEND_CUDA : CACAOSIM_BACKEND_CPU;

    /* Entrance pupil and optional entrance NCPA / atmosphere streams */
    if (ext_pup != NULL)
    {
        ctx->pup_ptr = ext_pup;
    }
    else if (connect_shm && cfg->in_pup[0] != '\0')
    {
        ctx->pup_ptr = connect_opt_2df32(cfg->in_pup, ctx->nwf, &ctx->img_pup);
    }

    if (ctx->pup_ptr == NULL)
    {
        ctx->fallback_pup = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
        if (ctx->fallback_pup == NULL)
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "Out of memory allocating pupil");
            return -1;
        }
        cacaosim_pupil_cfg_t pcfg = {
            .nwf           = ctx->nwf,
            .pup_diam      = cfg->pup_diam,
            .cobs          = cfg->cobs,
            .nspider       = cfg->nspider,
            .spider_w      = cfg->spider_w,
            .spider_deg    = cfg->spider_deg,
            .spider_offset = cfg->spider_offset,
        };
        cacaosim_make_pupil(&pcfg, ctx->fallback_pup);
        ctx->pup_ptr = ctx->fallback_pup;
    }

    if (connect_shm)
    {
        ctx->ncpa_ptr = connect_opt_2df32(cfg->in_ncpa, ctx->nwf, &ctx->img_ncpa);
        if (cfg->in_wf[0] != '\0')
        {
            ctx->img_wf = stream_connect(cfg->in_wf);
        }
        if (cfg->in_amp[0] != '\0')
        {
            ctx->img_amp = stream_connect(cfg->in_amp);
        }
    }

    /* Detect multi-wavelength chromatic turbulence cube (3D SHM or [bench] chrom_turb=1) */
    ctx->wf_chrom_nwl  = 1u;
    ctx->wf_chrom_unit = cfg->wf_unit;
    if (connect_shm && ctx->img_wf.ID != -1)
    {
        uint32_t cnwl  = 1u;
        int32_t  cunit = 1;
        float    cwls[8];
        if (cacaosim_wf_shm_read_chrom(ctx->img_wf.im, &cnwl, &cunit, cwls))
        {
            ctx->wf_chrom_nwl  = cnwl;
            ctx->wf_chrom_unit = cunit;
            memcpy(ctx->wf_chrom_wls, cwls, sizeof(cwls));
        }
    }
    if (ctx->wf_chrom_nwl < 2u && cfg->chrom_turb)
    {
        float    cwls[8];
        uint32_t cnwl = parse_chrom_lambdas_csv(cfg->chrom_lambdas, cwls, 8u);
        if (cnwl >= 2u)
        {
            ctx->wf_chrom_nwl  = cnwl;
            ctx->wf_chrom_unit = 0; /* Default milkatmturb chrom_opd_um = 1 -> OPD [um] */
            memcpy(ctx->wf_chrom_wls, cwls, sizeof(cwls));
        }
    }

    size_t wf_tot_pix = (size_t) ctx->wf_chrom_nwl * ctx->npix_wf;
    if (ctx->use_cuda)
    {
        ctx->d_pup_amp = cacaosim_cuda_alloc_f32(ctx->npix_wf);
        ctx->d_wf_in   = cacaosim_cuda_alloc_f32(wf_tot_pix);
        ctx->d_atm_amp = cacaosim_cuda_alloc_f32(wf_tot_pix);
        if (cfg->atm_interp && ctx->eff_oversample > 1u)
        {
            ctx->d_wf_prev = cacaosim_cuda_alloc_f32(wf_tot_pix);
            ctx->d_wf_curr = cacaosim_cuda_alloc_f32(wf_tot_pix);
        }
        if (!ctx->d_pup_amp || !ctx->d_wf_in || !ctx->d_atm_amp ||
            (cfg->atm_interp && ctx->eff_oversample > 1u &&
             (!ctx->d_wf_prev || !ctx->d_wf_curr)))
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "CUDA allocation failed for entrance buffers");
            return -1;
        }
        cacaosim_cuda_h2d_f32(ctx->d_pup_amp, ctx->pup_ptr, ctx->npix_wf, ctx->cuda_stream);
        if (ctx->ncpa_ptr != NULL)
        {
            ctx->d_ncpa_opd = cacaosim_cuda_alloc_f32(ctx->npix_wf);
            cacaosim_cuda_h2d_f32(ctx->d_ncpa_opd, ctx->ncpa_ptr, ctx->npix_wf, ctx->cuda_stream);
        }
    }
    else if (cfg->atm_interp && ctx->eff_oversample > 1u)
    {
        ctx->h_wf_prev = (float *) cacaosim_aligned_calloc(wf_tot_pix, sizeof(float));
        ctx->h_wf_curr = (float *) cacaosim_aligned_calloc(wf_tot_pix, sizeof(float));
        if (!ctx->h_wf_prev || !ctx->h_wf_curr)
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "Host allocation failed for WF lerp buffers");
            return -1;
        }
    }

    /* Allocate named optical beam buffers (initially monochromatic nlam = 1) */
    for (uint32_t b = 0; b < ctx->n_beams; b++)
    {
        ctx->beams[b].nlam      = 1u;
        ctx->beams[b].poly_mode = (int32_t) CACAOSIM_POLY_NONE;
        ctx->beams[b].owns_buf  = 1;
        ctx->beams[b].opd       = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
        ctx->beams[b].amp       = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
        if (!ctx->beams[b].opd || !ctx->beams[b].amp)
        {
            cacaosim_bench_free(ctx);
            dag_err(err_buf, err_cap, "Out of memory allocating beam '%s'", ctx->beams[b].name);
            return -1;
        }
        if (ctx->use_cuda)
        {
            ctx->beams[b].d_opd = cacaosim_cuda_alloc_f32(ctx->npix_wf);
            ctx->beams[b].d_amp = cacaosim_cuda_alloc_f32(ctx->npix_wf);
            if (!ctx->beams[b].d_opd || !ctx->beams[b].d_amp)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "CUDA alloc failed for beam '%s'", ctx->beams[b].name);
                return -1;
            }
        }
    }

    /* Promote root entrance beam to multi-wavelength unwrapped OPD cube when chrom_turb active */
    if (ctx->wf_chrom_nwl >= 2u)
    {
        if (upgrade_beam_poly(ctx, &ctx->beams[0], ctx->wf_chrom_nwl, ctx->wf_chrom_wls, err_buf,
                              err_cap) != 0)
        {
            cacaosim_bench_free(ctx);
            return -1;
        }
        ctx->beams[0].poly_mode = (int32_t) CACAOSIM_POLY_UNWRAPPED_OPD;
    }

    /* Initialize cached atmospheric dispersion state & IMR parallactic tracking */
    ctx->atm_disp_pa_curr_deg = (double) cfg->atm_disp.disp_pa_deg;
    bench_update_atm_disp(ctx);

    /* Precompute steady-state spectral photometry & dispersion across resolved sensor cadences */
    propagate_spectra(ctx);
    ctx->beams[0].disp = ctx->cached_atm_disp;

    if (cacaosim_perturb_init(&ctx->perturb, ctx->nwf, cfg->pup_diam, ctx->pup_ptr, &cfg->perturb,
                              ctx->use_cuda, ctx->cuda_stream) != 0)
    {
        cacaosim_bench_free(ctx);
        dag_err(err_buf, err_cap, "Failed to initialize entrance-pupil perturbation engine");
        return -1;
    }

    /* Initialize each compute unit in topological order */
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];

        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            if (ctx->n_dms >= CACAOSIM_BENCH_MAX_DMS)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Exceeded max DMs (%d)", CACAOSIM_BENCH_MAX_DMS);
                return -1;
            }
            nd->dm_slot                    = (int32_t) ctx->n_dms;
            ctx->dm_node_idx[ctx->n_dms++] = (int32_t) i;

            nd->cfg.dm.nwf     = ctx->nwf;
            nd->cfg.dm.dt_sec  = ctx->dt_master_s;
            nd->cfg.dm.backend = eff_backend;
            if (nd->cfg.dm.dm_diam <= 0.0f)
            {
                nd->cfg.dm.dm_diam = cfg->pup_diam;
            }
            if (nd->cfg.dm.dm_mode == 1)
            {
                nd->cfg.dm.nact_x = 2u;
                nd->cfg.dm.nact_y = 1u;
            }

            if (connect_shm)
            {
                nd->img_cmd = stream_connect_create_2Df32(nd->cfg.in_cmd, nd->cfg.dm.nact_x,
                                                          nd->cfg.dm.nact_y);
                if (nd->img_cmd.ID == -1)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "Failed creating DM stream '%s'", nd->cfg.in_cmd);
                    return -1;
                }
                /* Adopt existing SHM actuator dimensions if stream was pre-created by cacao */
                nd->cfg.dm.nact_x = nd->img_cmd.md->size[0];
                nd->cfg.dm.nact_y = nd->img_cmd.md->size[1];

                if (nd->cfg.out_opd[0] != '\0')
                {
                    nd->img_opd = stream_connect_create_2Df32(nd->cfg.out_opd, ctx->nwf, ctx->nwf);
                }
                connect_opt_2df32(nd->cfg.in_flat, ctx->nwf, &nd->img_flat);
            }
            else
            {
                size_t nact = (size_t) nd->cfg.dm.nact_x * nd->cfg.dm.nact_y;
                nd->cmd_buf = (float *) cacaosim_aligned_calloc(nact, sizeof(float));
            }

            if (!ctx->use_cuda || nd->img_flat.ID != -1)
            {
                nd->dm_opd_buf = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
            }

            if (cacaosim_dm_init(&nd->dm_ctx, &nd->cfg.dm, ctx->cuda_stream) != 0)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Failed initializing DM '%s'", nd->cfg.id);
                return -1;
            }
            /* Propagate polychromatic cube width if DM sits downstream of a poly coronagraph */
            const cacaosim_beam_slot_t *bin_dm = &ctx->beams[nd->in_idx];
            if (bin_dm->nlam > 1u &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], bin_dm->nlam, bin_dm->lam_um,
                                  err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
            ctx->beams[nd->out_idx].poly_mode = bin_dm->poly_mode;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_ADC)
        {
            cacaosim_adc_init(&nd->adc_ctx, &nd->cfg.adc, &cfg->atm_disp);
            if (connect_shm)
            {
                if (nd->cfg.in_cmd[0] != '\0')
                {
                    nd->img_cmd = stream_connect_create_2Df32(nd->cfg.in_cmd, 2u, 1u);
                }
                if (nd->cfg.out_state[0] != '\0')
                {
                    nd->img_state =
                        stream_connect_create_2Df32(nd->cfg.out_state, CACAOSIM_ADC_STATE_LEN, 1u);
                }
            }
            const cacaosim_beam_slot_t *bin_adc = &ctx->beams[nd->in_idx];
            if (bin_adc->nlam > 1u &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], bin_adc->nlam, bin_adc->lam_um,
                                  err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
            ctx->beams[nd->out_idx].poly_mode = bin_adc->poly_mode;
            cacaosim_adc_step(&nd->adc_ctx, 0.0, &cfg->atm_disp, NULL, &bin_adc->disp,
                              &bin_adc->spec, &ctx->beams[nd->out_idx].disp,
                              &ctx->beams[nd->out_idx].spec);
            if (connect_shm && nd->img_state.ID != -1)
            {
                nd->img_state.md->write = 1;
                memcpy(nd->img_state.im->array.F, nd->adc_ctx.state_vec,
                       sizeof(nd->adc_ctx.state_vec));
                ImageStreamIO_UpdateIm(nd->img_state.im);
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            if (connect_shm)
            {
                nd->h_ncpa_refl =
                    connect_opt_2df32(nd->cfg.ncpa_refl, ctx->nwf, &nd->img_ncpa_refl);
                nd->h_ncpa_trans =
                    connect_opt_2df32(nd->cfg.ncpa_trans, ctx->nwf, &nd->img_ncpa_trans);
                nd->h_pup_refl  = connect_opt_2df32(nd->cfg.pup_refl, ctx->nwf, &nd->img_pup_refl);
                nd->h_pup_trans =
                    connect_opt_2df32(nd->cfg.pup_trans, ctx->nwf, &nd->img_pup_trans);
            }
            if (ctx->use_cuda)
            {
                if (nd->h_ncpa_refl)
                {
                    nd->d_ncpa_refl = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_ncpa_refl, nd->h_ncpa_refl, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
                if (nd->h_ncpa_trans)
                {
                    nd->d_ncpa_trans = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_ncpa_trans, nd->h_ncpa_trans, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
                if (nd->h_pup_refl)
                {
                    nd->d_pup_refl = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_pup_refl, nd->h_pup_refl, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
                if (nd->h_pup_trans)
                {
                    nd->d_pup_trans = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                    cacaosim_cuda_h2d_f32(nd->d_pup_trans, nd->h_pup_trans, ctx->npix_wf,
                                          ctx->cuda_stream);
                }
            }
            const cacaosim_beam_slot_t *bin_sp = &ctx->beams[nd->in_idx];
            if (bin_sp->nlam > 1u)
            {
                if (nd->refl_idx >= 0 &&
                    upgrade_beam_poly(ctx, &ctx->beams[nd->refl_idx], bin_sp->nlam, bin_sp->lam_um,
                                      err_buf, err_cap) != 0)
                {
                    cacaosim_bench_free(ctx);
                    return -1;
                }
                if (nd->trans_idx >= 0 &&
                    upgrade_beam_poly(ctx, &ctx->beams[nd->trans_idx], bin_sp->nlam, bin_sp->lam_um,
                                      err_buf, err_cap) != 0)
                {
                    cacaosim_bench_free(ctx);
                    return -1;
                }
            }
            if (nd->refl_idx >= 0)
            {
                ctx->beams[nd->refl_idx].poly_mode = bin_sp->poly_mode;
            }
            if (nd->trans_idx >= 0)
            {
                ctx->beams[nd->trans_idx].poly_mode = bin_sp->poly_mode;
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            const cacaosim_beam_slot_t *bin_co = &ctx->beams[nd->in_idx];
            nd->cfg.coro.nwf                   = ctx->nwf;
            nd->cfg.coro.pup_diam              = cfg->pup_diam;
            nd->cfg.coro.backend               = eff_backend;
            if (bin_co->nlam > 1u && bin_co->poly_mode != CACAOSIM_POLY_UNWRAPPED_OPD &&
                nd->cfg.coro.nlambda <= 1)
            {
                nd->cfg.coro.nlambda = (int32_t) bin_co->nlam;
            }

            cacaosim_coro_masks_t masks = { 0 };
            int                   m_rc  = 0;
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_pup_amp, connect_shm, &masks.pup_amp, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_pup_pha, connect_shm, &masks.pup_pha, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_fpm_amp, connect_shm, &masks.fpm_amp, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_fpm_pha, connect_shm, &masks.fpm_pha, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_lyot, connect_shm, &masks.lyot_amp, err_buf,
                                          err_cap);
            }
            if (m_rc == 0)
            {
                m_rc = load_opt_coro_mask(nd->cfg.in_lyot_pha, connect_shm, &masks.lyot_pha,
                                          err_buf, err_cap);
            }

            int init_rc = (m_rc == 0) ? cacaosim_coro_init_masks(&nd->coro_ctx, &nd->cfg.coro,
                                                                 &masks, ctx->cuda_stream)
                                      : -1;
            cacaosim_mask_free(&masks.pup_amp);
            cacaosim_mask_free(&masks.pup_pha);
            cacaosim_mask_free(&masks.fpm_amp);
            cacaosim_mask_free(&masks.fpm_pha);
            cacaosim_mask_free(&masks.lyot_amp);
            cacaosim_mask_free(&masks.lyot_pha);

            if (init_rc != 0)
            {
                cacaosim_bench_free(ctx);
                if (m_rc == 0)
                {
                    dag_err(err_buf, err_cap, "Failed initializing coronagraph '%s'", nd->cfg.id);
                }
                return -1;
            }
            /* Bind input beam wavelengths so unwrapped chromatic cubes interpolate cleanly */
            if (bin_co->nlam > 1u)
            {
                cacaosim_coro_set_in_lambdas(&nd->coro_ctx, bin_co->lam_um, bin_co->nlam);
            }
            /* Widen output science and LOWFS beams to [eff_nlam, nwf, nwf] when broadband */
            if (nd->coro_ctx.eff_nlam > 1 &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], (uint32_t) nd->coro_ctx.eff_nlam,
                                  nd->coro_ctx.lam_um, err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
            ctx->beams[nd->out_idx].poly_mode =
                (nd->coro_ctx.eff_nlam > 1) ? (int32_t) CACAOSIM_POLY_COMPLEX_FIELD
                                            : (int32_t) CACAOSIM_POLY_NONE;
            cacaosim_disp_zero(&ctx->beams[nd->out_idx].disp);
            if (nd->lowfs_idx >= 0)
            {
                if (nd->coro_ctx.eff_nlam > 1 &&
                    upgrade_beam_poly(ctx, &ctx->beams[nd->lowfs_idx],
                                      (uint32_t) nd->coro_ctx.eff_nlam, nd->coro_ctx.lam_um,
                                      err_buf, err_cap) != 0)
                {
                    cacaosim_bench_free(ctx);
                    return -1;
                }
                ctx->beams[nd->lowfs_idx].poly_mode =
                    (nd->coro_ctx.eff_nlam > 1) ? (int32_t) CACAOSIM_POLY_COMPLEX_FIELD
                                                : (int32_t) CACAOSIM_POLY_NONE;
                cacaosim_disp_zero(&ctx->beams[nd->lowfs_idx].disp);
            }
            if (connect_shm)
            {
                if (nd->cfg.out_pre_lyot[0] != '\0')
                {
                    nd->img_pre_lyot =
                        stream_connect_create_2Df32(nd->cfg.out_pre_lyot, ctx->nwf, ctx->nwf);
                    if (nd->img_pre_lyot.ID != -1)
                    {
                        cacaosim_coro_shm_write_kw(nd->img_pre_lyot.im, 0.0f, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0ULL);
                    }
                }
                if (nd->cfg.out_pre_lyot_long[0] != '\0')
                {
                    nd->img_pre_lyot_long =
                        stream_connect_create_2Df32(nd->cfg.out_pre_lyot_long, ctx->nwf, ctx->nwf);
                    if (nd->img_pre_lyot_long.ID != -1)
                    {
                        cacaosim_coro_shm_write_kw(nd->img_pre_lyot_long.im, 0.0f, 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 0ULL);
                    }
                }
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            /* Static WFE maps are stored in SI meters [m] in SHM; auto-synthesize if absent */
            float rms_m   = (nd->cfg.wfe_rms_nm > 0.0f) ? (nd->cfg.wfe_rms_nm * 1.0e-9f) : 0.0f;
            float defoc_m = nd->cfg.wfe_defocus_nm * 1.0e-9f;
            if (connect_shm)
            {
                nd->img_wfe = stream_connect(nd->cfg.in_wfe);
                if (nd->img_wfe.ID == -1 || nd->img_wfe.md->datatype != _DATATYPE_FLOAT ||
                    nd->img_wfe.md->size[0] != ctx->nwf || nd->img_wfe.md->size[1] != ctx->nwf)
                {
                    free_imgid_if_valid(&nd->img_wfe);
                    nd->img_wfe =
                        stream_connect_create_2Df32(nd->cfg.in_wfe, ctx->nwf, ctx->nwf);
                    if (nd->img_wfe.ID == -1)
                    {
                        cacaosim_bench_free(ctx);
                        dag_err(err_buf, err_cap, "Failed creating WFE stream '%s'",
                                nd->cfg.in_wfe);
                        return -1;
                    }
                    nd->img_wfe.md->write = 1;
                    cacaosim_make_wfe_m(ctx->nwf, cfg->pup_diam, ctx->pup_ptr, rms_m,
                                        nd->cfg.wfe_alpha, nd->cfg.wfe_seed,
                                        nd->img_wfe.im->array.F);
                    if (fabsf(defoc_m) > 1e-15f)
                    {
                        cacaosim_add_defocus_m(ctx->nwf, cfg->pup_diam, defoc_m,
                                               nd->img_wfe.im->array.F);
                    }
                    if (nd->img_wfe.im->kw != NULL && nd->img_wfe.md->NBkw >= 1)
                    {
                        snprintf(nd->img_wfe.im->kw[0].name, KEYWORD_MAX_STRING, "BUNIT");
                        nd->img_wfe.im->kw[0].type = 'S';
                        snprintf(nd->img_wfe.im->kw[0].value.valstr, KEYWORD_MAX_STRING, "m");
                        snprintf(nd->img_wfe.im->kw[0].comment, KEYWORD_MAX_COMMENT,
                                 "Static wavefront error OPD [meter]");
                    }
                    ImageStreamIO_UpdateIm(nd->img_wfe.im);
                }
                nd->wfe_cnt0 = nd->img_wfe.md->cnt0;
            }
            else
            {
                nd->wfe_m_buf = (float *) cacaosim_aligned_calloc(ctx->npix_wf, sizeof(float));
                if (nd->wfe_m_buf == NULL)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "Out of memory allocating WFE map '%s'", nd->cfg.id);
                    return -1;
                }
                cacaosim_make_wfe_m(ctx->nwf, cfg->pup_diam, ctx->pup_ptr, rms_m,
                                    nd->cfg.wfe_alpha, nd->cfg.wfe_seed, nd->wfe_m_buf);
                if (fabsf(defoc_m) > 1e-15f)
                {
                    cacaosim_add_defocus_m(ctx->nwf, cfg->pup_diam, defoc_m, nd->wfe_m_buf);
                }
            }

            if (ctx->use_cuda)
            {
                const float *src_wfe = (connect_shm && nd->img_wfe.ID != -1)
                                           ? nd->img_wfe.im->array.F
                                           : nd->wfe_m_buf;
                nd->d_wfe_m          = cacaosim_cuda_alloc_f32(ctx->npix_wf);
                if (nd->d_wfe_m == NULL)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "CUDA alloc failed for WFE map '%s'", nd->cfg.id);
                    return -1;
                }
                cacaosim_cuda_h2d_f32(nd->d_wfe_m, src_wfe, ctx->npix_wf, ctx->cuda_stream);
            }

            const cacaosim_beam_slot_t *bin_wfe = &ctx->beams[nd->in_idx];
            if (bin_wfe->nlam > 1u &&
                upgrade_beam_poly(ctx, &ctx->beams[nd->out_idx], bin_wfe->nlam, bin_wfe->lam_um,
                                  err_buf, err_cap) != 0)
            {
                cacaosim_bench_free(ctx);
                return -1;
            }
        }
        else
        {
            if (ctx->n_sensors >= CACAOSIM_BENCH_MAX_SENSORS)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Exceeded max sensors (%d)", CACAOSIM_BENCH_MAX_SENSORS);
                return -1;
            }
            nd->sensor_slot                        = (int32_t) ctx->n_sensors;
            ctx->sensor_node_idx[ctx->n_sensors++] = (int32_t) i;

            const float *s_ncpa = NULL;
            if (connect_shm)
            {
                s_ncpa = connect_opt_2df32(nd->cfg.in_ncpa, ctx->nwf, &nd->img_ncpa);
            }

            const cacaosim_beam_slot_t *bin_sens = &ctx->beams[nd->in_idx];
            uint32_t in_nlam                     = (bin_sens->nlam >= 1u) ? bin_sens->nlam : 1u;

            nd->has_reg = !cacaosim_reg_is_identity(&nd->cfg.reg);
            if (nd->has_reg)
            {
                size_t tot_reg = (size_t) in_nlam * ctx->npix_wf;
                nd->reg_opd    = (float *) cacaosim_aligned_calloc(tot_reg, sizeof(float));
                nd->reg_amp    = (float *) cacaosim_aligned_calloc(tot_reg, sizeof(float));
                if (ctx->use_cuda)
                {
                    nd->d_reg_opd = cacaosim_cuda_alloc_f32(tot_reg);
                    nd->d_reg_amp = cacaosim_cuda_alloc_f32(tot_reg);
                }
            }

            /* Re-resolve spectral slices & sub-band weights now that upstream in_nlam is known */
            float   sens_lam = 0.75f;
            int32_t cfg_nlam = 0;
            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                sens_lam = nd->cfg.pyrwfs.lambda_um;
                cfg_nlam = nd->cfg.pyrwfs.nlambda;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                sens_lam = nd->cfg.shwfs.lambda_um;
                cfg_nlam = nd->cfg.shwfs.nlambda;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                sens_lam = nd->cfg.zwfs.lambda_um;
                cfg_nlam = nd->cfg.zwfs.nlambda;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                sens_lam = nd->cfg.fpcam.lambda_um;
                cfg_nlam = nd->cfg.fpcam.nlambda;
            }

            float dt_exp_s = (float) nd->eff_navg * ctx->dt_master_s;
            nd->eff_nlam   = cacaosim_resolve_nlambda(cfg_nlam, in_nlam, nd->cfg.bw_frac);
            (void) cacaosim_spec_sample_band(
                nd->cfg.use_beam_flux ? &bin_sens->spec : NULL, sens_lam, nd->cfg.bw_frac,
                nd->eff_nlam, dt_exp_s, nd->cfg.qe, nd->chan_lam_um, nd->chan_weight, NULL);

            if (in_nlam > 1u && cfg_nlam <= 0 && (int32_t) in_nlam == nd->eff_nlam)
            {
                float mid_in = 0.5f * (bin_sens->lam_um[0] + bin_sens->lam_um[in_nlam - 1u]);
                if (fabsf(sens_lam - mid_in) <= 0.05f * sens_lam)
                {
                    for (int32_t c = 0; c < nd->eff_nlam; c++)
                    {
                        nd->chan_lam_um[c] = bin_sens->lam_um[c];
                    }
                }
            }

            /* Named beams always carry OPD in microns (wf_unit = 0) and integrated eff_flux_ph */
            int rc = -1;
            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                uint32_t eff_st = (nd->eff_stride >= 1u) ? nd->eff_stride : 1u;
                uint32_t m_subf = nd->eff_navg / eff_st;
                nd->cfg.pyrwfs.nwf           = ctx->nwf;
                nd->cfg.pyrwfs.pup_diam      = cfg->pup_diam;
                nd->cfg.pyrwfs.wf_unit       = 0;
                nd->cfg.pyrwfs.backend       = eff_backend;
                nd->cfg.pyrwfs.bw_frac       = nd->cfg.bw_frac;
                nd->cfg.pyrwfs.nlambda       = nd->eff_nlam;
                nd->cfg.pyrwfs.has_ext_chan  = 1;
                for (int32_t c = 0; c < nd->eff_nlam; c++)
                {
                    nd->cfg.pyrwfs.ext_lam_um[c] = nd->chan_lam_um[c];
                    nd->cfg.pyrwfs.ext_weight[c] = nd->chan_weight[c];
                }
                nd->cfg.pyrwfs.mod_subframes = (m_subf >= 1u) ? m_subf : 1u;
                nd->cfg.pyrwfs.det.flux_ph   = nd->eff_flux_ph;
                nd->ncam                     = nd->cfg.pyrwfs.ncam;
                nd->out_dtype                = nd->cfg.pyrwfs.det.out_dtype;
                rc = cacaosim_pyrwfs_init(&nd->pyrwfs_ctx, &nd->cfg.pyrwfs, ctx->pup_ptr, s_ncpa,
                                          ctx->cuda_stream);
                if (rc == 0 && nd->pyrwfs_ctx.n_slices > 1 && nd->clock_phase_init > 0.0)
                {
                    nd->pyrwfs_ctx.slice_cursor =
                        ((int32_t) floor(nd->clock_phase_init *
                                         (double) nd->pyrwfs_ctx.n_slices)) %
                        nd->pyrwfs_ctx.n_slices;
                }
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                nd->cfg.shwfs.nwf          = ctx->nwf;
                nd->cfg.shwfs.pup_diam     = cfg->pup_diam;
                nd->cfg.shwfs.wf_unit      = 0;
                nd->cfg.shwfs.backend      = eff_backend;
                nd->cfg.shwfs.bw_frac      = nd->cfg.bw_frac;
                nd->cfg.shwfs.nlambda      = nd->eff_nlam;
                nd->cfg.shwfs.has_ext_chan = 1;
                for (int32_t c = 0; c < nd->eff_nlam; c++)
                {
                    nd->cfg.shwfs.ext_lam_um[c] = nd->chan_lam_um[c];
                    nd->cfg.shwfs.ext_weight[c] = nd->chan_weight[c];
                }
                nd->cfg.shwfs.det.flux_ph = nd->eff_flux_ph;
                nd->ncam                  = nd->cfg.shwfs.n_sub * nd->cfg.shwfs.pix_per_sub;
                nd->out_dtype             = nd->cfg.shwfs.det.out_dtype;
                rc = cacaosim_shwfs_init(&nd->shwfs_ctx, &nd->cfg.shwfs, ctx->pup_ptr, s_ncpa,
                                         ctx->cuda_stream);
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                nd->cfg.zwfs.nwf          = ctx->nwf;
                nd->cfg.zwfs.pup_diam     = cfg->pup_diam;
                nd->cfg.zwfs.wf_unit      = 0;
                nd->cfg.zwfs.backend      = eff_backend;
                nd->cfg.zwfs.bw_frac      = nd->cfg.bw_frac;
                nd->cfg.zwfs.nlambda      = nd->eff_nlam;
                nd->cfg.zwfs.has_ext_chan = 1;
                for (int32_t c = 0; c < nd->eff_nlam; c++)
                {
                    nd->cfg.zwfs.ext_lam_um[c] = nd->chan_lam_um[c];
                    nd->cfg.zwfs.ext_weight[c] = nd->chan_weight[c];
                }
                nd->cfg.zwfs.det.flux_ph = nd->eff_flux_ph;
                nd->ncam                 = nd->cfg.zwfs.ncam;
                nd->out_dtype            = nd->cfg.zwfs.det.out_dtype;
                rc = cacaosim_zwfs_init(&nd->zwfs_ctx, &nd->cfg.zwfs, ctx->pup_ptr, s_ncpa,
                                        ctx->cuda_stream);
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                nd->cfg.fpcam.nwf            = ctx->nwf;
                nd->cfg.fpcam.pup_diam       = cfg->pup_diam;
                nd->cfg.fpcam.wf_unit        = 0;
                nd->cfg.fpcam.backend        = eff_backend;
                nd->cfg.fpcam.bw_frac        = nd->cfg.bw_frac;
                nd->cfg.fpcam.nlambda        = nd->eff_nlam;
                nd->cfg.fpcam.det.flux_ph    = nd->eff_flux_ph;
                nd->ncam                     = nd->cfg.fpcam.ncam;
                nd->out_dtype                = nd->cfg.fpcam.det.out_dtype;
                nd->cfg.fpcam.has_ext_lam    = 1;
                nd->cfg.fpcam.has_ext_weight = 1;
                for (int32_t c = 0; c < nd->eff_nlam; c++)
                {
                    nd->cfg.fpcam.ext_lam_um[c] = nd->chan_lam_um[c];
                    nd->cfg.fpcam.ext_weight[c] = nd->chan_weight[c];
                }
                nd->cfg.fpcam.has_in_lam = (in_nlam > 1u) ? 1 : 0;
                if (in_nlam > 1u)
                {
                    uint32_t cap_in = (in_nlam <= (uint32_t) CACAOSIM_FPCAM_MAX_LAMBDA)
                                          ? in_nlam
                                          : (uint32_t) CACAOSIM_FPCAM_MAX_LAMBDA;
                    for (uint32_t c = 0; c < cap_in; c++)
                    {
                        nd->cfg.fpcam.in_lam_um[c] = bin_sens->lam_um[c];
                    }
                }
                rc = cacaosim_fpcam_init(&nd->fpcam_ctx, &nd->cfg.fpcam, ctx->pup_ptr, s_ncpa,
                                         ctx->cuda_stream);
            }

            if (rc != 0)
            {
                cacaosim_bench_free(ctx);
                dag_err(err_buf, err_cap, "Failed initializing sensor '%s'", nd->cfg.id);
                return -1;
            }

            size_t npix_cam = (size_t) nd->ncam * nd->ncam;
            if (connect_shm)
            {
                uint8_t dt  = (nd->out_dtype == 0) ? _DATATYPE_UINT16 : _DATATYPE_FLOAT;
                nd->img_cam = stream_connect_create_2D(nd->cfg.out_cam, nd->ncam, nd->ncam, dt);
                if (nd->img_cam.ID == -1)
                {
                    cacaosim_bench_free(ctx);
                    dag_err(err_buf, err_cap, "Failed creating camera stream '%s'",
                            nd->cfg.out_cam);
                    return -1;
                }
                if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && nd->cfg.out_long[0] != '\0')
                {
                    nd->img_long =
                        stream_connect_create_2Df32(nd->cfg.out_long, nd->ncam, nd->ncam);
                    if (nd->img_long.ID != -1)
                    {
                        cacaosim_fpcam_shm_write_kw(nd->img_long.im, 0.0f, 0.0f, 0.0f, 0ULL);
                    }
                }
            }
            else
            {
                nd->cam_u16 = (uint16_t *) cacaosim_aligned_calloc(npix_cam, sizeof(uint16_t));
                nd->cam_f32 = (float *) cacaosim_aligned_calloc(npix_cam, sizeof(float));
                if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
                {
                    nd->long_f32 = (float *) cacaosim_aligned_calloc(npix_cam, sizeof(float));
                }
            }
        }
    }

    /*
     * Zero-copy pointer aliasing for identity splitter arms (no NCPA, no stop mask, identity 2D
     * registration). Performed after all polychromatic `upgrade_beam_poly()` expansions complete
     * so downstream consumers read upstream `[OPD_um, Amplitude]` buffers directly.
     */
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind != CACAOSIM_NODE_SPLITTER)
        {
            continue;
        }
        const cacaosim_beam_slot_t *bin_sp = &ctx->beams[nd->in_idx];
        if (nd->refl_idx >= 0 && nd->h_ncpa_refl == NULL && nd->h_pup_refl == NULL &&
            cacaosim_reg_is_identity(&nd->cfg.reg_refl))
        {
            cacaosim_beam_slot_t *br = &ctx->beams[nd->refl_idx];
            if (br->owns_buf)
            {
                free(br->opd);
                free(br->amp);
                cacaosim_cuda_free_f32(br->d_opd);
                cacaosim_cuda_free_f32(br->d_amp);
                br->owns_buf = 0;
            }
            br->opd        = bin_sp->opd;
            br->amp        = bin_sp->amp;
            br->d_opd      = bin_sp->d_opd;
            br->d_amp      = bin_sp->d_amp;
            nd->alias_refl = 1;
        }
        if (nd->trans_idx >= 0 && nd->h_ncpa_trans == NULL && nd->h_pup_trans == NULL &&
            cacaosim_reg_is_identity(&nd->cfg.reg_trans))
        {
            cacaosim_beam_slot_t *bt = &ctx->beams[nd->trans_idx];
            if (bt->owns_buf)
            {
                free(bt->opd);
                free(bt->amp);
                cacaosim_cuda_free_f32(bt->d_opd);
                cacaosim_cuda_free_f32(bt->d_amp);
                bt->owns_buf = 0;
            }
            bt->opd         = bin_sp->opd;
            bt->amp         = bin_sp->amp;
            bt->d_opd       = bin_sp->d_opd;
            bt->d_amp       = bin_sp->d_amp;
            nd->alias_trans = 1;
        }
    }

    if (ctx->use_cuda)
    {
        cacaosim_cuda_stream_sync(ctx->cuda_stream);
    }
    return 0;
}

/**
 * cacaosim_bench_free() - Release all compute contexts, host/CUDA memory, and SHM handles in @ctx
 * @ctx: Runtime or analyzed bench context to tear down (safe no-op if NULL).
 *
 * Role:
 *   Frees per-node optical engines (`dm`, `coro`, `pyrwfs`, `shwfs`, `zwfs`, `fpcam`), host and
 *   device scratch/beam buffers, the shared CUDA stream, and all connected `IMGID` handles, then
 *   zeroes `*ctx`.
 *
 * Rationale:
 *   Safe to call on both fully initialized and partially initialized contexts (provided `IMGID.ID`
 *   sentinels were set to `-1`), serving as a unified error-rollback and shutdown destructor.
 */
void cacaosim_bench_free(cacaosim_bench_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            cacaosim_dm_free(&nd->dm_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            cacaosim_coro_free(&nd->coro_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
        {
            cacaosim_pyrwfs_free(&nd->pyrwfs_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
        {
            cacaosim_shwfs_free(&nd->shwfs_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
        {
            cacaosim_zwfs_free(&nd->zwfs_ctx);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
        {
            cacaosim_fpcam_free(&nd->fpcam_ctx);
        }

        free(nd->reg_opd);
        free(nd->reg_amp);
        free(nd->cmd_buf);
        free(nd->dm_opd_buf);
        free(nd->wfe_m_buf);
        free(nd->cam_u16);
        free(nd->cam_f32);
        free(nd->long_f32);

        cacaosim_cuda_free_f32(nd->d_wfe_m);
        cacaosim_cuda_free_f32(nd->d_reg_opd);
        cacaosim_cuda_free_f32(nd->d_reg_amp);
        cacaosim_cuda_free_f32(nd->d_ncpa_refl);
        cacaosim_cuda_free_f32(nd->d_ncpa_trans);
        cacaosim_cuda_free_f32(nd->d_pup_refl);
        cacaosim_cuda_free_f32(nd->d_pup_trans);

        free_imgid_if_valid(&nd->img_cmd);
        free_imgid_if_valid(&nd->img_opd);
        free_imgid_if_valid(&nd->img_flat);
        free_imgid_if_valid(&nd->img_wfe);
        free_imgid_if_valid(&nd->img_cam);
        free_imgid_if_valid(&nd->img_long);
        free_imgid_if_valid(&nd->img_ncpa);
        free_imgid_if_valid(&nd->img_lyot);
        free_imgid_if_valid(&nd->img_pre_lyot);
        free_imgid_if_valid(&nd->img_pre_lyot_long);
        free_imgid_if_valid(&nd->img_ncpa_refl);
        free_imgid_if_valid(&nd->img_ncpa_trans);
        free_imgid_if_valid(&nd->img_pup_refl);
        free_imgid_if_valid(&nd->img_pup_trans);
    }

    for (uint32_t b = 0; b < ctx->n_beams; b++)
    {
        if (ctx->beams[b].owns_buf)
        {
            free(ctx->beams[b].opd);
            free(ctx->beams[b].amp);
            cacaosim_cuda_free_f32(ctx->beams[b].d_opd);
            cacaosim_cuda_free_f32(ctx->beams[b].d_amp);
        }
        free_imgid_if_valid(&ctx->beams[b].img_shm);
    }

    cacaosim_perturb_free(&ctx->perturb);
    cacaosim_cuda_free_f32(ctx->d_pup_amp);
    cacaosim_cuda_free_f32(ctx->d_ncpa_opd);
    cacaosim_cuda_free_f32(ctx->d_wf_in);
    cacaosim_cuda_free_f32(ctx->d_wf_prev);
    cacaosim_cuda_free_f32(ctx->d_wf_curr);
    cacaosim_cuda_free_f32(ctx->d_atm_amp);
    cacaosim_cuda_stream_destroy(ctx->cuda_stream);

    free(ctx->h_wf_prev);
    free(ctx->h_wf_curr);
    free(ctx->fallback_pup);
    free_imgid_if_valid(&ctx->img_pup);
    free_imgid_if_valid(&ctx->img_ncpa);
    free_imgid_if_valid(&ctx->img_wf);
    free_imgid_if_valid(&ctx->img_amp);
    memset(ctx, 0, sizeof(*ctx));
}

/**
 * node_kind_str() - Return uppercase mnemonic tag for a DAG node kind in summary tables
 * @k:       Optical node kind (`cacaosim_node_kind_t`).
 * @dm_mode: DM sub-mode (`1` for 2-axis `"TTM"`, `0` for zonal `"DM"`).
 *
 * Role:
 *   Formats fixed-width type columns in `cacaosim_bench_print_summary()`.
 *
 * Rationale:
 *   Distinguishing `"TTM"` from `"DM"` makes two-corrector benches immediately clear in text logs.
 *
 * Return: Static uppercase label string.
 */
static const char *node_kind_str(cacaosim_node_kind_t k, int dm_mode)
{
    switch (k)
    {
    case CACAOSIM_NODE_DM:
        return (dm_mode == 1) ? "TTM" : "DM";
    case CACAOSIM_NODE_SPLITTER:
        return "SPLITTER";
    case CACAOSIM_NODE_CORO:
        return "CORO";
    case CACAOSIM_NODE_PYRWFS:
        return "PYRWFS";
    case CACAOSIM_NODE_SHWFS:
        return "SHWFS";
    case CACAOSIM_NODE_ZWFS:
        return "ZWFS";
    case CACAOSIM_NODE_FPCAM:
        return "FPCAM";
    case CACAOSIM_NODE_WFE:
        return "WFE";
    default:
        return "UNKNOWN";
    }
}

/**
 * cacaosim_bench_print_summary() - Write a human-readable topology & photometry table to @fp
 * @ctx: Analyzed or initialized bench context (ignored if NULL).
 * @fp:  Output stdio stream (ignored if NULL).
 *
 * Role:
 *   Emits the bench header (grid size, primary diameter, stellar V magnitude, master clock,
 *   CPU/CUDA backend), integrated entrance photon rate across `0.40..2.40 um`, and one summary
 *   line per topologically sorted node.
 *
 * Rationale:
 *   Used when `stdout` is redirected to log files or tmux logs where ANSI box-drawing schematics
 *   from `cacaosim_bench_render()` are undesirable.
 */
void cacaosim_bench_print_summary(const cacaosim_bench_ctx_t *ctx, FILE *fp)
{
    if (ctx == NULL || fp == NULL)
    {
        return;
    }
    if (ctx->eff_oversample > 1u)
    {
        fprintf(fp,
                "=== cacaosim Bench '%s' (%ux%u, D=%.2fm, mV=%.1f, %.0f Hz x%u [%.1f kHz], %s)"
                " ===\n",
                ctx->cfg.name, ctx->nwf, ctx->nwf, ctx->cfg.tel_diam_m, ctx->cfg.star_mag,
                ctx->cfg.master_hz, ctx->eff_oversample, ctx->eff_master_hz * 1.0e-3f,
                ctx->use_cuda ? "CUDA" : "CPU");
    }
    else
    {
        fprintf(fp, "=== cacaosim Bench '%s' (%ux%u, D=%.2fm, mV=%.1f, %.0f Hz, %s) ===\n",
                ctx->cfg.name, ctx->nwf, ctx->nwf, ctx->cfg.tel_diam_m, ctx->cfg.star_mag,
                ctx->cfg.master_hz, ctx->use_cuda ? "CUDA" : "CPU");
    }
    fprintf(fp, "  Entrance beam '%s': %.3e ph/s (0.40..2.40 um)\n", ctx->beams[0].name,
            cacaosim_spec_total(&ctx->beams[0].spec));

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[i];
        const char                  *ks = node_kind_str(nd->cfg.kind, nd->cfg.dm.dm_mode);
        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> %s (%ux%u, cmd=%s)\n", i, ks, nd->cfg.id,
                    nd->cfg.in_beam, nd->cfg.out_beam, nd->cfg.dm.nact_x, nd->cfg.dm.nact_y,
                    nd->cfg.in_cmd);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> %s (wfe=%s [m], rms=%.1fnm, gain=%.2f)\n",
                    i, ks, nd->cfg.id, nd->cfg.in_beam, nd->cfg.out_beam, nd->cfg.in_wfe,
                    nd->cfg.wfe_rms_nm, nd->cfg.wfe_gain);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> refl=%s, trans=%s (cut=%.2fum)\n", i, ks,
                    nd->cfg.id, nd->cfg.in_beam,
                    nd->cfg.out_refl[0] ? nd->cfg.out_refl : "-",
                    nd->cfg.out_trans[0] ? nd->cfg.out_trans : "-", nd->cfg.split.lambda_cut);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            fprintf(fp, "  [%u] %-8s %-12s : %s -> %s (lowfs=%s [%s], lam=%.2fum, nlam=%d)\n", i,
                    ks, nd->cfg.id, nd->cfg.in_beam, nd->cfg.out_beam,
                    nd->cfg.out_lowfs[0] ? nd->cfg.out_lowfs : "-",
                    (nd->cfg.coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL) ? "lyot_refl"
                                                                          : "fpm_refl",
                    nd->cfg.coro.lambda_um,
                    (nd->coro_ctx.eff_nlam > 1) ? nd->coro_ctx.eff_nlam : 1);
        }
        else
        {
            float   m_hz = (ctx->eff_master_hz > 0.1f) ? ctx->eff_master_hz : ctx->cfg.master_hz;
            float   hz   = m_hz / (float) nd->eff_navg;
            int32_t nch  = (nd->eff_nlam >= 1) ? nd->eff_nlam : 1;
            fprintf(fp,
                    "  [%u] %-8s %-12s : in=%s -> %s (%ux%u, %.2fum %dch, %.0fHz navg=%u,"
                    " %.2e ph/fr)\n",
                    i, ks, nd->cfg.id, nd->cfg.in_beam, nd->cfg.out_cam, nd->ncam, nd->ncam,
                    nd->eff_lambda_um, nch, hz, nd->eff_navg, nd->eff_flux_ph);
        }
    }
}

/**
 * cacaosim_bench_step() - Advance the compiled multi-rate optical bench DAG by one master tick
 * @ctx:       Initialized bench context (safe no-op if NULL)
 * @ext_wf:    Optional host 2D wavefront override [nwf x nwf] (NULL = use SHM in_wf or zero WF)
 * @ext_amp:   Optional host 2D amplitude override [nwf x nwf] (NULL = use SHM in_amp or unity)
 * @turb_gain: Live atmospheric turbulence multiplier (0.0 = flat calibration, 1.0 = nominal)
 * @procinfo:  Optional MILK PROCESSINFO pointer for output stream signaling (NULL allowed)
 *
 * Role:
 *   Executes one deterministic master clock tick (`1 / master_hz` seconds) of the entire optical
 *   bench: synthesizes the entrance `[OPD_um, Amplitude]` beam, propagates it in topological order
 *   through all active DMs, static WFE screens, dichroic splitters, and coronagraphs, accumulates
 *   diffractive subframes on every WFS and science camera, and triggers detector readout + SHM
 *   publication on each sensor whose `subframe_cnt` reaches its integer decimation `eff_navg`.
 *
 * Rationale:
 *   Designed strictly around real-time hot-loop invariants: zero `malloc`/`free`, zero FFT plan
 *   creation, and zero host/device round-trips between intermediate optical stages. When running
 *   on CUDA, every beam slot (`d_opd`, `d_amp`) and DM surface (`dm_ctx.d_opd_out`) stays on the
 *   shared `ctx->cuda_stream`, transferring only external turbulence inputs (H2D) and completed
 *   camera frames (D2H at each sensor's decimated readout rate).
 */
static inline cacaosim_det_cfg_t bench_sensor_det(const cacaosim_bench_ctx_t *ctx,
                                                  const cacaosim_det_cfg_t   *det)
{
    cacaosim_det_cfg_t out = *det;
    if (ctx->calib_fast)
    {
        /* Noise-free internal calibration source for single-frame step-and-shoot RMs */
        out.photon_noise = 0;
        out.ron_e        = 0.0f;
        out.dark_e       = 0.0f;
    }
    return out;
}

static inline uint32_t bench_sensor_subf_cnt(const cacaosim_bench_node_t *nd)
{
    if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
    {
        return nd->pyrwfs_ctx.subframe_cnt;
    }
    if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
    {
        return nd->shwfs_ctx.subframe_cnt;
    }
    if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
    {
        return nd->zwfs_ctx.subframe_cnt;
    }
    if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
    {
        return nd->fpcam_ctx.subframe_cnt;
    }
    return 0u;
}

static inline int bench_shm_has_sem_readers(const IMAGE *im)
{
    if (im == NULL || im->md == NULL || im->semReadPID == NULL)
    {
        return 0;
    }
    for (uint16_t s = 0; s < im->md->sem; s++)
    {
        if (im->semReadPID[s] > 0)
        {
            return 1;
        }
    }
    return 0;
}

MILK_HOT void cacaosim_bench_step(
    cacaosim_bench_ctx_t *ctx,
    const float          *ext_wf,
    const float          *ext_amp,
    float                 turb_gain,
    void                 *procinfo)
{
    if (ctx == NULL)
    {
        return;
    }

    const uint32_t eff_ov        = (ctx->eff_oversample >= 1u) ? ctx->eff_oversample : 1u;
    const uint32_t tel_st        = (ctx->cfg.telemetry_stride >= 1u) ? ctx->cfg.telemetry_stride
                                                                     : eff_ov;
    const int      is_static_cal = (ctx->calib_fast && !ctx->calib_temporal);
    const int      do_export     = (!ctx->calib_fast && ctx->connect_shm &&
                                    ctx->export_beams_active &&
                                    ((ctx->tick_cnt % tel_st) == 0ULL));

    /*
     * 0. Reverse DAG liveness & per-sensor fractional clock-phase evaluation:
     *    Determine which sensors accumulate and/or read out on this master tick, then walk
     *    nodes backward so upstream optical stages whose output beams are unused on this sub-tick
     *    (e.g. a coronagraph branch between strided FPCAM/LOWFS samples) are pruned cleanly.
     */
    uint8_t beam_needed[CACAOSIM_BENCH_MAX_BEAMS]    = { 0 };
    uint8_t node_live[CACAOSIM_BENCH_MAX_NODES]      = { 0 };
    uint8_t sens_accum[CACAOSIM_BENCH_MAX_NODES]     = { 0 };
    uint8_t sens_read[CACAOSIM_BENCH_MAX_NODES]      = { 0 };
    uint8_t dm_pend_d2h[CACAOSIM_BENCH_MAX_NODES]      = { 0 };
    uint8_t fp_pend_finish[CACAOSIM_BENCH_MAX_NODES]   = { 0 };
    uint8_t coro_pend_finish[CACAOSIM_BENCH_MAX_NODES] = { 0 };
    int     any_dm_d2h                                 = 0;
    int     any_async_readout                          = 0;

    if (do_export)
    {
        for (uint32_t b = 0; b < ctx->n_beams; b++)
        {
            if (ctx->beams[b].export_shm)
            {
                beam_needed[b] = 1u;
            }
        }
    }
    if (!ctx->calib_fast && ctx->wfstat_step_active)
    {
        for (int k = 0; k < 3; k++)
        {
            int32_t bi = ctx->wfstat_beam_idx[k];
            if (bi >= 0 && (uint32_t) bi < ctx->n_beams)
            {
                beam_needed[bi] = 1u;
            }
        }
    }

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        nd->ready_this_tick       = 0;

        if (nd->cfg.kind < CACAOSIM_NODE_PYRWFS || nd->cfg.kind > CACAOSIM_NODE_FPCAM)
        {
            continue;
        }
        int fpcam_is_wfs = (nd->cfg.kind == CACAOSIM_NODE_FPCAM) &&
                           (nd->cfg.is_wfs ||
                            (ctx->calib_wfs_out[0] != '\0' &&
                             strcmp(ctx->calib_wfs_out, "none") != 0 &&
                             (strcmp(nd->cfg.out_cam, ctx->calib_wfs_out) == 0 ||
                              strcmp(nd->cfg.id, ctx->calib_wfs_out) == 0)));
        if (ctx->skip_fpcam && nd->cfg.kind == CACAOSIM_NODE_FPCAM && !fpcam_is_wfs)
        {
            continue;
        }

        if (ctx->calib_fast)
        {
            if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && !fpcam_is_wfs)
            {
                continue;
            }
            if (ctx->calib_wfs_out[0] != '\0' && strcmp(ctx->calib_wfs_out, "none") != 0 &&
                strcmp(nd->cfg.out_cam, ctx->calib_wfs_out) != 0 &&
                strcmp(nd->cfg.id, ctx->calib_wfs_out) != 0)
            {
                continue;
            }
            if (is_static_cal)
            {
                sens_accum[i]            = 1u;
                sens_read[i]             = 1u;
                beam_needed[nd->in_idx]  = 1u;
                continue;
            }
        }

        uint32_t cur_subf = bench_sensor_subf_cnt(nd);
        if (cur_subf == 0u && nd->prev_subframe_cnt != 0u)
        {
            /* Detect external subframe_cnt reset (e.g. live eff_navg override in tests) */
            nd->clock_phase       = nd->clock_phase_init;
            nd->prev_subframe_cnt = 0u;
        }

        uint32_t nav       = (nd->eff_navg >= 1u) ? nd->eff_navg : 1u;
        double   pstep     = (1.0 / (double) nav) *
                             (1.0 + 1.0e-6 * (double) nd->cfg.clock_drift_ppm);
        double   next_p    = nd->clock_phase + pstep;
        int      will_read = (next_p >= 1.0 - 1.0e-9);
        uint32_t st        = (nd->eff_stride >= 1u) ? nd->eff_stride : 1u;
        int      do_acc    = (st <= 1u) ||
                             (((ctx->tick_cnt + nd->stagger_tick) % st) == 0ULL) ||
                             (will_read && cur_subf == 0u);

        nd->clock_phase = next_p;
        if (will_read)
        {
            nd->clock_phase -= 1.0;
            if (nd->clock_phase < 1.0e-12 || nd->clock_phase >= 1.0)
            {
                nd->clock_phase = (nd->clock_phase > 0.0) ? fmod(nd->clock_phase, 1.0) : 0.0;
            }
        }

        sens_accum[i] = (uint8_t) do_acc;
        sens_read[i]  = (uint8_t) will_read;
        if (do_acc)
        {
            beam_needed[nd->in_idx] = 1u;
        }
    }

    for (int idx = (int) ctx->n_nodes - 1; idx >= 0; idx--)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[idx];
        uint8_t live                    = 0u;
        if (nd->cfg.kind == CACAOSIM_NODE_DM || nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            live = beam_needed[nd->out_idx];
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            live = ((nd->refl_idx >= 0 && beam_needed[nd->refl_idx]) ||
                    (nd->trans_idx >= 0 && beam_needed[nd->trans_idx]))
                       ? 1u
                       : 0u;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            live = (beam_needed[nd->out_idx] ||
                    (nd->lowfs_idx >= 0 && beam_needed[nd->lowfs_idx]))
                       ? 1u
                       : 0u;
        }
        node_live[idx] = live;
        if (live)
        {
            beam_needed[nd->in_idx] = 1u;
        }
    }

    const float *wf_ptr  = ext_wf;
    const float *amp_ptr = ext_amp;

    /*
     * Lazily attach optional SHM atmospheric phase (`in_wf`) and scintillation (`in_amp`) streams
     * so external turbulence generators started after the bench come online automatically without
     * restarting `cacaosim.htsim`.
     */
    if (ctx->connect_shm)
    {
        if (ctx->img_wf.ID == -1 && ctx->cfg.in_wf[0] != '\0')
        {
            ctx->img_wf = stream_connect(ctx->cfg.in_wf);
        }
        if (ctx->img_amp.ID == -1 && ctx->cfg.in_amp[0] != '\0')
        {
            ctx->img_amp = stream_connect(ctx->cfg.in_amp);
        }
        if (wf_ptr == NULL && ctx->img_wf.ID != -1 && ctx->img_wf.md->datatype == _DATATYPE_FLOAT &&
            ctx->img_wf.md->size[0] == ctx->nwf)
        {
            wf_ptr = ctx->img_wf.im->array.F;
        }
        if (amp_ptr == NULL && ctx->img_amp.ID != -1 &&
            ctx->img_amp.md->datatype == _DATATYPE_FLOAT && ctx->img_amp.md->size[0] == ctx->nwf)
        {
            amp_ptr = ctx->img_amp.im->array.F;
        }
    }

    /* Convert phase [rad at ref_lambda] to OPD [um] when wf_unit == 1; else scale OPD [um] */
    float wf_scale = (ctx->cfg.wf_unit == 1)
                         ? (turb_gain * ctx->cfg.ref_lambda / (2.0f * CACAOSIM_PI_F))
                         : turb_gain;

    /*
     * 1. Synthesize entrance beam [OPD_um, Amplitude] in slot 0.
     *    When `atm_interp && eff_oversample > 1` on a live SHM `in_wf` stream, upload each new
     *    base-rate `outarraypha` screen once upon `md->cnt0` increment and linearly blend
     *    `(1 - alpha) * wf_prev + alpha * wf_curr` across the `K` sub-ticks.
     */
    cacaosim_beam_slot_t *b0      = &ctx->beams[0];
    int                   do_lerp = (ext_wf == NULL && wf_ptr != NULL && ctx->cfg.atm_interp &&
                                     eff_ov > 1u && fabsf(wf_scale) > 0.0f &&
                                     ((ctx->use_cuda && ctx->d_wf_curr != NULL) ||
                                      (!ctx->use_cuda && ctx->h_wf_curr != NULL)));

    if (do_lerp)
    {
        uint64_t cur_cnt0 = ctx->img_wf.md->cnt0;
        if (ctx->wf_interp_primed == 0)
        {
            if (ctx->use_cuda)
            {
                cacaosim_cuda_h2d_f32(ctx->d_wf_curr, wf_ptr, ctx->npix_wf, ctx->cuda_stream);
            }
            else
            {
                memcpy(ctx->h_wf_curr, wf_ptr, ctx->npix_wf * sizeof(float));
            }
            ctx->wf_prev_cnt0     = cur_cnt0;
            ctx->wf_substep       = 0u;
            ctx->wf_interp_primed = 1;
        }
        else if (cur_cnt0 != ctx->wf_prev_cnt0)
        {
            if (ctx->use_cuda)
            {
                float *tmp     = ctx->d_wf_prev;
                ctx->d_wf_prev = ctx->d_wf_curr;
                ctx->d_wf_curr = tmp;
                cacaosim_cuda_h2d_f32(ctx->d_wf_curr, wf_ptr, ctx->npix_wf, ctx->cuda_stream);
            }
            else
            {
                float *tmp     = ctx->h_wf_prev;
                ctx->h_wf_prev = ctx->h_wf_curr;
                ctx->h_wf_curr = tmp;
                memcpy(ctx->h_wf_curr, wf_ptr, ctx->npix_wf * sizeof(float));
            }
            ctx->wf_prev_cnt0     = cur_cnt0;
            ctx->wf_substep       = 0u;
            ctx->wf_interp_primed = 2;
        }

        float alpha = (ctx->wf_interp_primed >= 2)
                          ? ((float) (ctx->wf_substep + 1u) / (float) eff_ov)
                          : 1.0f;
        if (alpha > 1.0f)
        {
            alpha = 1.0f;
        }
        if (ctx->wf_substep + 1u < eff_ov)
        {
            ctx->wf_substep++;
        }

        if (beam_needed[0])
        {
            if (ctx->use_cuda)
            {
                const float *d_amp = NULL;
                if (amp_ptr != NULL)
                {
                    cacaosim_cuda_h2d_f32(ctx->d_atm_amp, amp_ptr, ctx->npix_wf, ctx->cuda_stream);
                    d_amp = ctx->d_atm_amp;
                }
                cacaosim_cuda_beam_init_lerp(
                    ctx->d_pup_amp, d_amp, (ctx->wf_interp_primed >= 2) ? ctx->d_wf_prev : NULL,
                    ctx->d_wf_curr, alpha, ctx->d_ncpa_opd, wf_scale, ctx->nwf, b0->d_opd,
                    b0->d_amp, ctx->cuda_stream);
            }
            else
            {
                cacaosim_beam_init_lerp_cpu(
                    ctx->pup_ptr, amp_ptr, (ctx->wf_interp_primed >= 2) ? ctx->h_wf_prev : NULL,
                    ctx->h_wf_curr, alpha, ctx->ncpa_ptr, wf_scale, ctx->nwf, b0->opd, b0->amp);
            }
        }
    }
    else if (beam_needed[0])
    {
        if (ctx->use_cuda)
        {
            const float *d_wf  = NULL;
            const float *d_amp = NULL;
            if (wf_ptr != NULL && fabsf(wf_scale) > 0.0f)
            {
                cacaosim_cuda_h2d_f32(ctx->d_wf_in, wf_ptr, ctx->npix_wf, ctx->cuda_stream);
                d_wf = ctx->d_wf_in;
            }
            if (amp_ptr != NULL)
            {
                cacaosim_cuda_h2d_f32(ctx->d_atm_amp, amp_ptr, ctx->npix_wf, ctx->cuda_stream);
                d_amp = ctx->d_atm_amp;
            }
            cacaosim_cuda_beam_init(ctx->d_pup_amp, d_amp, d_wf, ctx->d_ncpa_opd, wf_scale,
                                    ctx->nwf, b0->d_opd, b0->d_amp, ctx->cuda_stream);
        }
        else
        {
            cacaosim_beam_init_cpu(ctx->pup_ptr, amp_ptr, wf_ptr, ctx->ncpa_ptr, wf_scale,
                                   ctx->nwf, b0->opd, b0->amp);
        }
    }

    if (beam_needed[0] && !ctx->calib_fast &&
        (ctx->cfg.perturb.vib_enable || ctx->cfg.perturb.petal_enable))
    {
        cacaosim_perturb_step_apply(
            &ctx->perturb, &ctx->cfg.perturb, (double) ctx->dt_master_s, turb_gain, b0->opd,
            ctx->use_cuda ? b0->d_opd : NULL, ctx->cuda_stream);
    }

    /* Use primary DM command stream as lineage trigger reference for sensor SHM updates */
    IMAGE       *trig_im = (ctx->connect_shm && ctx->n_dms > 0 &&
                            ctx->nodes[ctx->dm_node_idx[0]].img_cmd.ID != -1)
                               ? ctx->nodes[ctx->dm_node_idx[0]].img_cmd.im
                               : NULL;
    PROCESSINFO *pinfo   = (PROCESSINFO *) procinfo;

    /*
     * 2. Traverse optical DAG in compiled topological order (`ctx->nodes[0..n_nodes-1]`).
     *    Topological compilation guarantees every upstream producer writes `bin` before any
     *    downstream consumer reads it within the same master clock tick.
     */
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t      *nd  = &ctx->nodes[i];
        const cacaosim_beam_slot_t *bin = &ctx->beams[nd->in_idx];

        if (nd->cfg.kind == CACAOSIM_NODE_DM)
        {
            /*
             * Active DM / Tip-Tilt mirror: always advance the delay FIFO and ZOH actuator state
             * on every master tick so hardware temporal memory stays exact, and render/subtract
             * the 2D OPD surface whenever downstream beams or decimated SHM telemetry require it.
             */
            cacaosim_beam_slot_t *bout = &ctx->beams[nd->out_idx];
            const float *cmd_in =
                ((int32_t) i == ctx->calib_dm_node && ctx->calib_cmd_override != NULL)
                    ? ctx->calib_cmd_override
                    : ((ctx->connect_shm && nd->img_cmd.ID != -1) ? nd->img_cmd.im->array.F
                                                                  : nd->cmd_buf);
            const float *flat_in =
                (ctx->connect_shm && nd->img_flat.ID != -1) ? nd->img_flat.im->array.F : NULL;

            int want_shm_opd = 0;
            if (!ctx->calib_fast && ctx->connect_shm && nd->img_opd.ID != -1 &&
                ((ctx->tick_cnt % tel_st) == 0ULL))
            {
                if (!ctx->use_cuda || ctx->export_beams_active || ctx->wfstat_snap_active ||
                    bench_shm_has_sem_readers(nd->img_opd.im) ||
                    (((ctx->tick_cnt / tel_st) & 63ULL) == 0ULL))
                {
                    want_shm_opd = 1;
                }
            }
            float eff_delay  = (nd->cfg.dm.delay_sec > 0.0f)
                                   ? (nd->cfg.dm.delay_sec / ctx->dt_master_s)
                                   : (nd->cfg.dm.delay_frames * (float) eff_ov);

            if (!node_live[i] && !want_shm_opd)
            {
                if (!is_static_cal)
                {
                    cacaosim_dm_step_actuators(&nd->dm_ctx, cmd_in, eff_delay, nd->cfg.dm.tau_sec,
                                               nd->cfg.dm.max_stroke);
                }
                continue;
            }

            float *host_opd = NULL;
            if (!ctx->use_cuda)
            {
                if (want_shm_opd)
                {
                    host_opd              = nd->img_opd.im->array.F;
                    nd->img_opd.md->write = 1;
                }
                else
                {
                    host_opd = nd->dm_opd_buf;
                }
            }
            else if (flat_in != NULL)
            {
                host_opd = want_shm_opd ? nd->img_opd.im->array.F : nd->dm_opd_buf;
                if (want_shm_opd)
                {
                    nd->img_opd.md->write = 1;
                }
            }

            if (is_static_cal)
            {
                if (nd->dm_ctx.cfg.dyn_mode == 3 && nd->dm_ctx.cfg.plate_drive == 1 &&
                    nd->dm_ctx.plate_mask != NULL)
                {
                    cacaosim_dm_step_static_plate(&nd->dm_ctx, cmd_in, flat_in,
                                                  nd->cfg.dm.max_stroke, host_opd);
                }
                else
                {
                    int32_t orig_dyn            = nd->dm_ctx.cfg.dyn_mode;
                    int32_t orig_nozoh          = nd->dm_ctx.cfg.no_zoh_mean;
                    nd->dm_ctx.cfg.dyn_mode     = 0;
                    nd->dm_ctx.cfg.no_zoh_mean  = 1;
                    cacaosim_dm_step(&nd->dm_ctx, cmd_in, flat_in, 0.0f, 0.0f,
                                     nd->cfg.dm.max_stroke, host_opd);
                    nd->dm_ctx.cfg.dyn_mode    = orig_dyn;
                    nd->dm_ctx.cfg.no_zoh_mean = orig_nozoh;
                }
            }
            else
            {
                cacaosim_dm_step(&nd->dm_ctx, cmd_in, flat_in, eff_delay, nd->cfg.dm.tau_sec,
                                 nd->cfg.dm.max_stroke, host_opd);
            }

            if (ctx->use_cuda && flat_in != NULL && host_opd != NULL)
            {
                cacaosim_cuda_h2d_f32(nd->dm_ctx.d_opd_out, host_opd, ctx->npix_wf,
                                      ctx->cuda_stream);
            }

            if (want_shm_opd)
            {
                if (ctx->use_cuda && flat_in == NULL)
                {
                    dm_pend_d2h[i] = 1u;
                    any_dm_d2h     = 1;
                }
                else
                {
                    processinfo_update_output_stream(pinfo, nd->img_opd.im, nd->img_cmd.im);
                }
            }

            if (node_live[i])
            {
                uint32_t nl = (bin->nlam >= 1u) ? bin->nlam : 1u;
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_apply_dm(bin->d_opd + off, bin->d_amp + off,
                                                    nd->dm_ctx.d_opd_out, ctx->nwf,
                                                    bout->d_opd + off, bout->d_amp + off,
                                                    ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_apply_dm_cpu(bin->opd + off, bin->amp + off, host_opd,
                                                   ctx->nwf, bout->opd + off, bout->amp + off);
                    }
                }
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
        {
            if (!node_live[i])
            {
                continue;
            }
            uint32_t nl = (bin->nlam >= 1u) ? bin->nlam : 1u;
            if (nd->refl_idx >= 0 && !nd->alias_refl && beam_needed[nd->refl_idx])
            {
                cacaosim_beam_slot_t *br = &ctx->beams[nd->refl_idx];
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_relay(bin->d_opd + off, bin->d_amp + off,
                                                 nd->d_ncpa_refl, nd->d_pup_refl,
                                                 &nd->cfg.reg_refl, ctx->nwf, br->d_opd + off,
                                                 br->d_amp + off, ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_relay_cpu(bin->opd + off, bin->amp + off, nd->h_ncpa_refl,
                                                nd->h_pup_refl, &nd->cfg.reg_refl, ctx->nwf,
                                                br->opd + off, br->amp + off);
                    }
                }
            }
            if (nd->trans_idx >= 0 && !nd->alias_trans && beam_needed[nd->trans_idx])
            {
                cacaosim_beam_slot_t *bt = &ctx->beams[nd->trans_idx];
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_relay(bin->d_opd + off, bin->d_amp + off,
                                                 nd->d_ncpa_trans, nd->d_pup_trans,
                                                 &nd->cfg.reg_trans, ctx->nwf, bt->d_opd + off,
                                                 bt->d_amp + off, ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_relay_cpu(bin->opd + off, bin->amp + off, nd->h_ncpa_trans,
                                                nd->h_pup_trans, &nd->cfg.reg_trans, ctx->nwf,
                                                bt->opd + off, bt->amp + off);
                    }
                }
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            if (!node_live[i])
            {
                continue;
            }
            cacaosim_beam_slot_t *bout = &ctx->beams[nd->out_idx];
            cacaosim_beam_slot_t *blow = (nd->lowfs_idx >= 0) ? &ctx->beams[nd->lowfs_idx] : NULL;
            uint32_t              nl   = (bin->nlam >= 1u) ? bin->nlam : 1u;

            cacaosim_coro_step_poly(&nd->coro_ctx, bin->opd, bin->amp, nl, ctx->npix_wf, bout->opd,
                                    bout->amp, ctx->npix_wf, blow ? blow->opd : NULL,
                                    blow ? blow->amp : NULL, bin->d_opd, bin->d_amp, bout->d_opd,
                                    bout->d_amp, blow ? blow->d_opd : NULL,
                                    blow ? blow->d_amp : NULL);
            if (!ctx->connect_shm)
            {
                if (ctx->use_cuda)
                {
                    cacaosim_coro_readout_telemetry_async(
                        &nd->coro_ctx, nd->coro_ctx.cpu_pre_lyot_int, NULL, 0);
                    coro_pend_finish[i] = 1u;
                    any_async_readout   = 1;
                }
            }
            else if (!ctx->calib_fast &&
                     (nd->img_pre_lyot.ID != -1 || nd->img_pre_lyot_long.ID != -1) &&
                     (((nd->coro_ctx.long_cnt % 8ULL) == 0ULL) || ctx->wfstat_snap_active ||
                      nd->coro_ctx.long_cnt <= 2ULL))
            {
                if (nd->img_pre_lyot.ID != -1)
                {
                    nd->img_pre_lyot.md->write = 1;
                }
                if (nd->img_pre_lyot_long.ID != -1)
                {
                    nd->img_pre_lyot_long.md->write = 1;
                }
                cacaosim_coro_readout_telemetry_async(
                    &nd->coro_ctx,
                    (nd->img_pre_lyot.ID != -1) ? nd->img_pre_lyot.im->array.F : NULL,
                    (nd->img_pre_lyot_long.ID != -1) ? nd->img_pre_lyot_long.im->array.F : NULL, 0);
                coro_pend_finish[i] = 1u;
                any_async_readout   = 1;
            }
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
        {
            if (!node_live[i])
            {
                continue;
            }
            cacaosim_beam_slot_t *bout  = &ctx->beams[nd->out_idx];
            const float          *h_wfe = (ctx->connect_shm && nd->img_wfe.ID != -1)
                                              ? nd->img_wfe.im->array.F
                                              : nd->wfe_m_buf;
            if (ctx->use_cuda && ctx->connect_shm && nd->img_wfe.ID != -1 &&
                nd->img_wfe.md->cnt0 != nd->wfe_cnt0)
            {
                cacaosim_cuda_h2d_f32(nd->d_wfe_m, h_wfe, ctx->npix_wf, ctx->cuda_stream);
                nd->wfe_cnt0 = nd->img_wfe.md->cnt0;
            }

            float    wfe_to_um = 1.0e6f * nd->cfg.wfe_gain;
            uint32_t nl        = (bin->nlam >= 1u) ? bin->nlam : 1u;
            for (uint32_t c = 0; c < nl; c++)
            {
                size_t off = (size_t) c * ctx->npix_wf;
                if (ctx->use_cuda)
                {
                    cacaosim_cuda_beam_apply_wfe(bin->d_opd + off, bin->d_amp + off, nd->d_wfe_m,
                                                 wfe_to_um, &nd->cfg.reg, ctx->nwf,
                                                 bout->d_opd + off, bout->d_amp + off,
                                                 ctx->cuda_stream);
                }
                else
                {
                    cacaosim_beam_apply_wfe_cpu(bin->opd + off, bin->amp + off, h_wfe, wfe_to_um,
                                                &nd->cfg.reg, ctx->nwf, bout->opd + off,
                                                bout->amp + off);
                }
            }
        }
        else
        {
            if (!sens_accum[i] && !sens_read[i])
            {
                continue;
            }

            const float *h_opd = bin->opd;
            const float *h_amp = bin->amp;
            const float *d_opd = bin->d_opd;
            const float *d_amp = bin->d_amp;
            uint32_t     nl    = (bin->nlam >= 1u) ? bin->nlam : 1u;

            if (sens_accum[i] && nd->has_reg)
            {
                for (uint32_t c = 0; c < nl; c++)
                {
                    size_t off = (size_t) c * ctx->npix_wf;
                    if (ctx->use_cuda)
                    {
                        cacaosim_cuda_beam_relay(bin->d_opd + off, bin->d_amp + off, NULL, NULL,
                                                 &nd->cfg.reg, ctx->nwf, nd->d_reg_opd + off,
                                                 nd->d_reg_amp + off, ctx->cuda_stream);
                    }
                    else
                    {
                        cacaosim_beam_relay_cpu(bin->opd + off, bin->amp + off, NULL, NULL,
                                                &nd->cfg.reg, ctx->nwf, nd->reg_opd + off,
                                                nd->reg_amp + off);
                    }
                }
                if (ctx->use_cuda)
                {
                    d_opd = nd->d_reg_opd;
                    d_amp = nd->d_reg_amp;
                }
                else
                {
                    h_opd = nd->reg_opd;
                    h_amp = nd->reg_amp;
                }
            }

            uint16_t *dst_u16 = (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype == 0)
                                    ? nd->img_cam.im->array.UI16
                                    : nd->cam_u16;
            float    *dst_f32 = (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype != 0)
                                    ? nd->img_cam.im->array.F
                                    : nd->cam_f32;
            const float *s_ncpa =
                (ctx->connect_shm && nd->img_ncpa.ID != -1) ? nd->img_ncpa.im->array.F : NULL;

            if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
            {
                nd->pyrwfs_ctx.force_full_mod = is_static_cal ? 1 : 0;
                if (sens_accum[i])
                {
                    cacaosim_pyrwfs_accum_beam_poly(&nd->pyrwfs_ctx, h_opd, h_amp,
                                                    (nl > 1u) ? bin->lam_um : NULL, nl,
                                                    ctx->npix_wf, s_ncpa, d_opd, d_amp,
                                                    nd->cfg.pyrwfs.mod_rad_ld);
                }
                if (sens_read[i] && nd->pyrwfs_ctx.subframe_cnt >= 1u)
                {
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    cacaosim_det_cfg_t det_cfg = bench_sensor_det(ctx, &nd->cfg.pyrwfs.det);
                    cacaosim_pyrwfs_readout_async(&nd->pyrwfs_ctx, &det_cfg, dst_u16, dst_f32);
                    nd->ready_this_tick = 1;
                    any_async_readout   = 1;
                }
                nd->prev_subframe_cnt = nd->pyrwfs_ctx.subframe_cnt;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
            {
                if (sens_accum[i])
                {
                    cacaosim_shwfs_accum_beam_poly(&nd->shwfs_ctx, h_opd, h_amp,
                                                   (nl > 1u) ? bin->lam_um : NULL, nl,
                                                   ctx->npix_wf, s_ncpa, d_opd, d_amp);
                }
                if (sens_read[i] && nd->shwfs_ctx.subframe_cnt >= 1u)
                {
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    cacaosim_det_cfg_t det_cfg = bench_sensor_det(ctx, &nd->cfg.shwfs.det);
                    cacaosim_shwfs_readout(&nd->shwfs_ctx, &det_cfg, dst_u16, dst_f32);
                    nd->ready_this_tick = 1;
                }
                nd->prev_subframe_cnt = nd->shwfs_ctx.subframe_cnt;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
            {
                if (sens_accum[i])
                {
                    cacaosim_zwfs_accum_beam_poly(&nd->zwfs_ctx, h_opd, h_amp,
                                                  (nl > 1u) ? bin->lam_um : NULL, nl,
                                                  ctx->npix_wf, s_ncpa, d_opd, d_amp);
                }
                if (sens_read[i] && nd->zwfs_ctx.subframe_cnt >= 1u)
                {
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    cacaosim_det_cfg_t det_cfg = bench_sensor_det(ctx, &nd->cfg.zwfs.det);
                    cacaosim_zwfs_readout(&nd->zwfs_ctx, &det_cfg, dst_u16, dst_f32);
                    nd->ready_this_tick = 1;
                }
                nd->prev_subframe_cnt = nd->zwfs_ctx.subframe_cnt;
            }
            else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
            {
                if (sens_accum[i])
                {
                    cacaosim_fpcam_accum_beam_poly(&nd->fpcam_ctx, h_opd, h_amp, nl, ctx->npix_wf,
                                                   s_ncpa, d_opd, d_amp);
                }
                if (sens_read[i] && nd->fpcam_ctx.subframe_cnt >= 1u)
                {
                    uint64_t next_long = nd->fpcam_ctx.long_cnt + 1ULL;
                    int      copy_long = (!ctx->use_cuda || nd->eff_navg >= 4u ||
                                          next_long <= 32ULL || ((next_long & 3ULL) == 0ULL));
                    float   *dst_long  = copy_long
                                             ? ((ctx->connect_shm && nd->img_long.ID != -1)
                                                    ? nd->img_long.im->array.F
                                                    : nd->long_f32)
                                             : NULL;
                    if (ctx->connect_shm && nd->img_cam.ID != -1)
                    {
                        nd->img_cam.md->write = 1;
                    }
                    if (copy_long && ctx->connect_shm && nd->img_long.ID != -1)
                    {
                        nd->img_long.md->write = 1;
                    }
                    cacaosim_det_cfg_t det_cfg = bench_sensor_det(ctx, &nd->cfg.fpcam.det);
                    cacaosim_fpcam_readout_async(&nd->fpcam_ctx, &det_cfg, dst_u16, dst_f32,
                                                 dst_long);
                    nd->ready_this_tick = 1;
                    fp_pend_finish[i]   = 1u;
                    any_async_readout   = 1;
                }
                nd->prev_subframe_cnt = nd->fpcam_ctx.subframe_cnt;
            }
        }
    }

    /* 3. Coalesced single-sync D2H completion & SHM semaphore publication across all nodes */
    int any_d2h = 0;
    if (any_dm_d2h)
    {
        for (uint32_t i = 0; i < ctx->n_nodes; i++)
        {
            if (!dm_pend_d2h[i])
            {
                continue;
            }
            cacaosim_bench_node_t *nd = &ctx->nodes[i];
            nd->img_opd.md->write     = 1;
            cacaosim_cuda_d2h_f32(nd->img_opd.im->array.F, nd->dm_ctx.d_opd_out, ctx->npix_wf,
                                  ctx->cuda_stream);
            any_d2h = 1;
        }
    }
    if (do_export)
    {
        for (uint32_t b = 0; b < ctx->n_beams; b++)
        {
            cacaosim_beam_slot_t *bs = &ctx->beams[b];
            if (!bs->export_shm || bs->img_shm.ID == -1 || bs->img_shm.im == NULL)
            {
                continue;
            }
            bs->img_shm.md->write = 1;
            uint32_t nl           = (bs->nlam >= 1u) ? bs->nlam : 1u;
            uint32_t max_nl       = bs->img_shm.md->size[2] / 2u;
            if (nl > max_nl)
            {
                nl = max_nl;
            }
            float *dst = bs->img_shm.im->array.F;
            for (uint32_t c = 0; c < nl; c++)
            {
                size_t src_off = (size_t) c * ctx->npix_wf;
                size_t dst_opd = (size_t) (2u * c) * ctx->npix_wf;
                size_t dst_amp = (size_t) (2u * c + 1u) * ctx->npix_wf;
                if (ctx->use_cuda)
                {
                    cacaosim_cuda_d2h_f32(dst + dst_opd, bs->d_opd + src_off, ctx->npix_wf,
                                          ctx->cuda_stream);
                    cacaosim_cuda_d2h_f32(dst + dst_amp, bs->d_amp + src_off, ctx->npix_wf,
                                          ctx->cuda_stream);
                    any_d2h = 1;
                }
                else
                {
                    memcpy(dst + dst_opd, bs->opd + src_off, ctx->npix_wf * sizeof(float));
                    memcpy(dst + dst_amp, bs->amp + src_off, ctx->npix_wf * sizeof(float));
                }
            }
        }
    }

    if (ctx->use_cuda && (any_async_readout || any_d2h))
    {
        cacaosim_cuda_stream_sync(ctx->cuda_stream);
    }

    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (coro_pend_finish[i])
        {
            cacaosim_coro_readout_telemetry_finish(&nd->coro_ctx);
            if (ctx->connect_shm)
            {
                if (nd->img_pre_lyot.ID != -1)
                {
                    cacaosim_coro_shm_write_kw(
                        nd->img_pre_lyot.im, nd->coro_ctx.long_frac_lowfs,
                        nd->coro_ctx.last_frac_lowfs, nd->coro_ctx.long_frac_pre_lyot,
                        nd->coro_ctx.last_frac_pre_lyot, nd->coro_ctx.long_frac_lyot_trans,
                        nd->coro_ctx.last_frac_lyot_trans, nd->coro_ctx.long_cnt);
                    processinfo_update_output_stream(pinfo, nd->img_pre_lyot.im, trig_im);
                }
                if (nd->img_pre_lyot_long.ID != -1)
                {
                    cacaosim_coro_shm_write_kw(
                        nd->img_pre_lyot_long.im, nd->coro_ctx.long_frac_lowfs,
                        nd->coro_ctx.last_frac_lowfs, nd->coro_ctx.long_frac_pre_lyot,
                        nd->coro_ctx.last_frac_pre_lyot, nd->coro_ctx.long_frac_lyot_trans,
                        nd->coro_ctx.last_frac_lyot_trans, nd->coro_ctx.long_cnt);
                    processinfo_update_output_stream(pinfo, nd->img_pre_lyot_long.im, trig_im);
                }
            }
        }
        if (!nd->ready_this_tick)
        {
            continue;
        }
        if (fp_pend_finish[i])
        {
            cacaosim_fpcam_readout_finish(&nd->fpcam_ctx);
            if (!nd->cfg.is_wfs)
            {
                ctx->last_strehl_inst = nd->fpcam_ctx.last_strehl_peak;
                ctx->last_strehl_ave  = nd->fpcam_ctx.long_strehl_peak;
            }
            if (ctx->connect_shm && nd->img_long.ID != -1)
            {
                cacaosim_fpcam_shm_write_kw(
                    nd->img_long.im, nd->fpcam_ctx.long_strehl_peak,
                    nd->fpcam_ctx.last_strehl_peak, nd->fpcam_ctx.last_strehl_axis,
                    nd->fpcam_ctx.long_cnt);
                processinfo_update_output_stream(pinfo, nd->img_long.im, trig_im);
            }
        }
        if (ctx->connect_shm && nd->img_cam.ID != -1)
        {
            processinfo_update_output_stream(pinfo, nd->img_cam.im, trig_im);
        }
    }

    if (any_dm_d2h)
    {
        for (uint32_t i = 0; i < ctx->n_nodes; i++)
        {
            if (dm_pend_d2h[i])
            {
                cacaosim_bench_node_t *nd = &ctx->nodes[i];
                processinfo_update_output_stream(pinfo, nd->img_opd.im, nd->img_cmd.im);
            }
        }
    }
    if (do_export)
    {
        for (uint32_t b = 0; b < ctx->n_beams; b++)
        {
            cacaosim_beam_slot_t *bs = &ctx->beams[b];
            if (!bs->export_shm || bs->img_shm.ID == -1 || bs->img_shm.im == NULL)
            {
                continue;
            }
            cacaosim_beam_shm_write_spec(bs->img_shm.im, &bs->spec);
            processinfo_update_output_stream(pinfo, bs->img_shm.im, trig_im);
        }
    }

    ctx->tick_cnt++;
}

/**
 * cacaosim_bench_find_dm() - Resolve a DM node identifier to its compact DM slot index
 * @ctx: Initialized bench context (may be NULL)
 * @id:  NUL-terminated DM node `id` (e.g. `"woofer"` or `"tweeter"`; may be NULL)
 *
 * Role:
 *   Looks up `@id` across `ctx->dm_node_idx[0..n_dms-1]` and returns the compact DM slot index
 *   accepted by `cacaosim_bench_dm_cmd_ptr()`.
 *
 * Rationale:
 *   Separates string-based lookup (performed once during test/controller setup) from O(1) slot
 *   indexing inside high-rate in-process control loops.
 *
 * Return: Compact DM slot index in `[0, n_dms - 1]` on match, or `-1` if not found / NULL args.
 */
int cacaosim_bench_find_dm(const cacaosim_bench_ctx_t *ctx, const char *id)
{
    if (ctx == NULL || id == NULL)
    {
        return -1;
    }
    for (uint32_t d = 0; d < ctx->n_dms; d++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->dm_node_idx[d]];
        if (strcmp(nd->cfg.id, id) == 0)
        {
            return (int) d;
        }
    }
    return -1;
}

/**
 * cacaosim_bench_find_sensor() - Resolve a sensor node identifier to its compact sensor slot index
 * @ctx: Initialized bench context (may be NULL)
 * @id:  NUL-terminated sensor node `id` (e.g. `"pyr"`, `"lowfs"`, `"sci"`; may be NULL)
 *
 * Role:
 *   Looks up `@id` across `ctx->sensor_node_idx[0..n_sensors-1]` and returns the compact sensor
 *   slot index accepted by `cacaosim_bench_sensor_f32()` and `cacaosim_bench_sensor_u16()`.
 *
 * Rationale:
 *   Allows headless CTest verification suites and embedded benchmarks to bind named WFS/camera
 *   outputs once at startup without hardcoding topological node indices.
 *
 * Return: Compact sensor slot index in `[0, n_sensors - 1]`, or `-1` if not found / NULL args.
 */
int cacaosim_bench_find_sensor(const cacaosim_bench_ctx_t *ctx, const char *id)
{
    if (ctx == NULL || id == NULL)
    {
        return -1;
    }
    for (uint32_t s = 0; s < ctx->n_sensors; s++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->sensor_node_idx[s]];
        if (strcmp(nd->cfg.id, id) == 0)
        {
            return (int) s;
        }
    }
    return -1;
}

/**
 * cacaosim_bench_dm_cmd_ptr() - Obtain a writable pointer to a DM node's actuator command buffer
 * @ctx:     Initialized bench context (may be NULL)
 * @dm_slot: Compact DM slot index returned by `cacaosim_bench_find_dm()`
 *
 * Role:
 *   Returns the `[nact_x x nact_y]` `float32` command array read by `cacaosim_bench_step()` for
 *   the specified DM, seamlessly selecting either the attached SHM `img_cmd` array (when
 *   `connect_shm != 0`) or the internal aligned host buffer `nd->cmd_buf` (headless mode).
 *
 * Rationale:
 *   Provides a single uniform zero-copy command injection API regardless of whether the bench is
 *   backed by `milk` shared memory or running standalone in unit tests and benchmarks.
 *
 * Return: Writable `float *` command array of length `nact_x * nact_y`, or `NULL` if invalid.
 */
float *cacaosim_bench_dm_cmd_ptr(cacaosim_bench_ctx_t *ctx, int dm_slot)
{
    if (ctx == NULL || dm_slot < 0 || (uint32_t) dm_slot >= ctx->n_dms)
    {
        return NULL;
    }
    cacaosim_bench_node_t *nd = &ctx->nodes[ctx->dm_node_idx[dm_slot]];
    if (ctx->connect_shm && nd->img_cmd.ID != -1)
    {
        return nd->img_cmd.im->array.F;
    }
    return nd->cmd_buf;
}

/**
 * cacaosim_bench_sensor_f32() - Obtain a read-only pointer to a sensor's float32 detector frame
 * @ctx:         Initialized bench context (may be NULL)
 * @sensor_slot: Compact sensor slot index returned by `cacaosim_bench_find_sensor()`
 *
 * Role:
 *   Returns the `[ncam x ncam]` `float32` output image buffer populated on readout ticks when the
 *   sensor's `out_dtype != 0`, resolving transparently to SHM (`img_cam.im->array.F`) or internal
 *   host memory (`nd->cam_f32`).
 *
 * Rationale:
 *   Lets headless tests, calibration loops, and benchmarks inspect calibrated `float32` detector
 *   frames directly without touching SHM files or duplicating output buffers.
 *
 * Return: Read-only `const float *` frame pointer, or `NULL` if `@ctx` or `@sensor_slot` invalid.
 */
const float *cacaosim_bench_sensor_f32(const cacaosim_bench_ctx_t *ctx, int sensor_slot)
{
    if (ctx == NULL || sensor_slot < 0 || (uint32_t) sensor_slot >= ctx->n_sensors)
    {
        return NULL;
    }
    const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->sensor_node_idx[sensor_slot]];
    if (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype != 0)
    {
        return nd->img_cam.im->array.F;
    }
    return nd->cam_f32;
}

/**
 * cacaosim_bench_sensor_u16() - Obtain a read-only pointer to a sensor's uint16 detector frame
 * @ctx:         Initialized bench context (may be NULL)
 * @sensor_slot: Compact sensor slot index returned by `cacaosim_bench_find_sensor()`
 *
 * Role:
 *   Returns the `[ncam x ncam]` `uint16_t` camera ADU buffer populated on readout ticks when the
 *   sensor's `out_dtype == 0`, resolving transparently to SHM (`img_cam.im->array.UI16`) or
 *   internal host memory (`nd->cam_u16`).
 *
 * Rationale:
 *   Mirrors hardware camera link / CoaXPress raw `uint16` ADU readout streams in both SHM-attached
 *   and standalone headless test environments.
 *
 * Return: Read-only `const uint16_t *` frame pointer, or `NULL` if `@ctx` / `@sensor_slot` invalid.
 */
const uint16_t *cacaosim_bench_sensor_u16(const cacaosim_bench_ctx_t *ctx, int sensor_slot)
{
    if (ctx == NULL || sensor_slot < 0 || (uint32_t) sensor_slot >= ctx->n_sensors)
    {
        return NULL;
    }
    const cacaosim_bench_node_t *nd = &ctx->nodes[ctx->sensor_node_idx[sensor_slot]];
    if (ctx->connect_shm && nd->img_cam.ID != -1 && nd->out_dtype == 0)
    {
        return nd->img_cam.im->array.UI16;
    }
    return nd->cam_u16;
}

int cacaosim_bench_find_beam(const cacaosim_bench_ctx_t *ctx, const char *name)
{
    if (ctx == NULL)
    {
        return -1;
    }
    return find_beam_name(ctx, name);
}

IMAGE *cacaosim_bench_find_dm_stream(const cacaosim_bench_ctx_t *ctx, const char *spec)
{
    if (ctx == NULL || !ctx->connect_shm || spec == NULL || spec[0] == '\0' ||
        strcmp(spec, "none") == 0)
    {
        return NULL;
    }
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind != CACAOSIM_NODE_DM || nd->img_cmd.ID == -1)
        {
            continue;
        }
        if (strcmp(nd->cfg.in_cmd, spec) == 0 || strcmp(nd->cfg.id, spec) == 0)
        {
            return nd->img_cmd.im;
        }
    }
    return NULL;
}

void cacaosim_bench_reset_wfs_accum(cacaosim_bench_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    ctx->wf_interp_primed = 0;
    ctx->wf_substep       = 0u;
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS && nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
        {
            nd->clock_phase       = nd->clock_phase_init;
            nd->prev_subframe_cnt = 0u;
        }
        if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
        {
            nd->pyrwfs_ctx.subframe_cnt   = 0u;
            nd->pyrwfs_ctx.force_full_mod = 0;
            nd->pyrwfs_ctx.slice_cursor   =
                (nd->pyrwfs_ctx.n_slices > 1 && nd->clock_phase_init > 0.0)
                    ? (((int32_t) floor(nd->clock_phase_init *
                                        (double) nd->pyrwfs_ctx.n_slices)) %
                       nd->pyrwfs_ctx.n_slices)
                    : 0;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
        {
            nd->shwfs_ctx.subframe_cnt = 0u;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
        {
            nd->zwfs_ctx.subframe_cnt = 0u;
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
        {
            nd->fpcam_ctx.subframe_cnt = 0u;
        }
    }
}

void cacaosim_bench_reset_fpcam_accum(cacaosim_bench_ctx_t *ctx)
{
    if (ctx == NULL)
    {
        return;
    }
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
        {
            nd->clock_phase            = nd->clock_phase_init;
            nd->prev_subframe_cnt      = 0u;
            nd->fpcam_ctx.subframe_cnt = 0u;
        }
    }
}

static int bind_single_export_beam(
    cacaosim_bench_ctx_t *ctx,
    uint32_t              bidx,
    char                 *err_buf,
    size_t                err_cap)
{
    cacaosim_beam_slot_t *bs = &ctx->beams[bidx];
    if (bs->export_shm && bs->img_shm.ID != -1)
    {
        return 0;
    }
    uint32_t nl = (bs->nlam >= 1u) ? bs->nlam : 1u;
    bs->img_shm = stream_connect_create_3Df32(bs->name, ctx->nwf, ctx->nwf, 2u * nl);
    if (bs->img_shm.ID == -1)
    {
        dag_err(err_buf, err_cap, "Failed creating export SHM stream for beam '%s'", bs->name);
        return -1;
    }
    if (bs->img_shm.im != NULL && bs->img_shm.im->array.F != NULL)
    {
        float *dst = bs->img_shm.im->array.F;
        for (uint32_t c = 0; c < nl; c++)
        {
            float *dst_opd = dst + (size_t) (2u * c) * ctx->npix_wf;
            float *dst_amp = dst + (size_t) (2u * c + 1u) * ctx->npix_wf;
            memset(dst_opd, 0, ctx->npix_wf * sizeof(float));
            if (ctx->pup_ptr != NULL)
            {
                memcpy(dst_amp, ctx->pup_ptr, ctx->npix_wf * sizeof(float));
            }
        }
    }
    cacaosim_beam_shm_init_kw(bs->img_shm.im);
    cacaosim_beam_shm_write_spec(bs->img_shm.im, &bs->spec);
    if (nl > 1u)
    {
        cacaosim_beam_shm_write_poly_kw(bs->img_shm.im, nl, bs->lam_um[0], bs->lam_um[nl - 1u]);
    }
    bs->export_shm = 1;
    return 0;
}

int cacaosim_bench_bind_export_beams(
    cacaosim_bench_ctx_t *ctx,
    const char           *csv_beams,
    char                 *err_buf,
    size_t                err_cap)
{
    if (ctx == NULL)
    {
        dag_err(err_buf, err_cap, "NULL bench context");
        return -1;
    }
    if (!ctx->connect_shm || csv_beams == NULL || csv_beams[0] == '\0')
    {
        return 0;
    }

    char copy[512];
    snprintf(copy, sizeof(copy), "%s", csv_beams);
    char *p        = copy;
    int   any_bind = 0;

    while (*p != '\0')
    {
        while (*p == ' ' || *p == '\t' || *p == ',')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }
        char *tok = p;
        while (*p != '\0' && *p != ',')
        {
            p++;
        }
        char *end = p;
        if (*p == ',')
        {
            *p++ = '\0';
        }
        while (end > tok && (end[-1] == ' ' || end[-1] == '\t'))
        {
            *--end = '\0';
        }
        if (tok[0] == '\0')
        {
            continue;
        }
        if (strcmp(tok, "*") == 0 || strcmp(tok, "all") == 0)
        {
            for (uint32_t b = 0; b < ctx->n_beams; b++)
            {
                if (bind_single_export_beam(ctx, b, err_buf, err_cap) != 0)
                {
                    return -1;
                }
                any_bind = 1;
            }
        }
        else
        {
            int bidx = find_beam_name(ctx, tok);
            if (bidx < 0)
            {
                dag_err(err_buf, err_cap, "Unknown beam '%s' in .export_beams", tok);
                return -1;
            }
            if (bind_single_export_beam(ctx, (uint32_t) bidx, err_buf, err_cap) != 0)
            {
                return -1;
            }
            any_bind = 1;
        }
    }

    ctx->export_beams_active = any_bind ? 1 : 0;
    return 0;
}
