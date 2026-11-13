// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mask.h
 * @brief   2D achromatic and 3D polychromatic optical mask loader/writer (FITS HDU + MILK SHM)
 *
 * Architectural Role:
 *   Provides a unified cold-path optical mask cube abstraction (`cacaosim_mask_cube_t`) and I/O
 *   layer shared across `cacaosim` optical modules—most notably custom pupil apodizers, focal-plane
 *   masks (`custom_mft`, `custom_fft`), and Lyot stops in `cacaosim.coro`, as well as external
 *   pupil/WFE/DM static map loaders and test/export utilities.
 *
 * Design Rationale & Motivation:
 *   1. Unified 2D/3D representation (`nz >= 1`): Modeling a 2D achromatic mask (`[nx, ny]`) as a
 *      single-slice (`nz == 1`) instance of a 3D spectral cube (`[nx, ny, nz]`) lets downstream
 *      optical engines handle both achromatic and wavelength-dependent masks (e.g., dispersive
 *      phase plates, multi-layer metamaterial vortex masks, or precomputed chromatic FPMs) through
 *      a single code path.
 *   2. Zero-copy views alongside heap-owned cubes (`owns_data`): Static inline constructors
 *      `cacaosim_mask_view_2d()` and `cacaosim_mask_view_3d()` wrap caller-allocated buffers with
 *      `owns_data = 0`, allowing programmatic C tests and in-memory generators to feed
 *      `cacaosim_coro_init_masks()` without temporary FITS files or heap copies while
 *      `cacaosim_mask_free()` safely cleans up both owned and borrowed descriptors.
 *   3. Spectral WCS preservation (`has_wcs_lam`, `lam_um[]`): Capturing per-slice wavelengths from
 *      FITS WCS (`CRVAL3`/`CDELT3`/`CRPIX3`/`CUNIT3`) or explicit `LAM0`/`LAM1`/`CSIM_WL0`/`WL1`
 *      keywords allows downstream polychromatic propagators to linearly interpolate mask slices
 *      onto the simulation's active wavelength channels when grids differ.
 *   4. Transparent FITS-or-SHM specifier resolution (`cacaosim_mask_load`): Accepting file paths
 *      (`.fits`, `.fit`, `.fts`, paths with `/`, or CFITSIO-style leading `!`) and `ImageStreamIO`
 *      shared-memory stream names interchangeably lets `.bench` configs switch between static disk
 *      designs and live adaptive apodizers/masks in `MILK_SHM_DIR` without changing parameters.
 */

#ifndef CACAOSIM_MASK_H
#define CACAOSIM_MASK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Maximum number of wavelength slices for which explicit WCS wavelength coordinates `lam_um[]`
 * are cached inside `cacaosim_mask_cube_t`, matching `CACAOSIM_NSPEC` (`32`).
 */
#define CACAOSIM_MASK_MAX_LAMBDA 32

/**
 * struct cacaosim_mask_cube_t - 2D achromatic or 3D polychromatic `float32` optical mask cube
 * @data:        Contiguous slice-major `[nz][ny][nx]` (`x` fastest, `z` slowest) `float32` array;
 *               64-byte aligned when heap-allocated by `cacaosim_mask_load*()`.
 * @nx:          Spatial width along FITS axis 1 `[pixels]`.
 * @ny:          Spatial height along FITS axis 2 `[pixels]`.
 * @nz:          Number of spectral slices along FITS axis 3 (`1` = achromatic 2D, `> 1` = 3D
 *               polychromatic cube with slice stride `nx * ny`, `0` = empty/unpopulated).
 * @has_wcs_lam: `1` if explicit per-slice physical wavelengths `[um]` were populated in `@lam_um`;
 *               `0` if slices span the consumer's bandpass implicitly.
 * @lam_um:      Explicit slice wavelengths `[microns]` for `z = 0 .. min(nz, 32) - 1` when
 *               `@has_wcs_lam == 1`.
 * @owns_data:   Ownership flag (`1` = `@data` was heap-allocated and will be freed by
 *               `cacaosim_mask_free()`; `0` = non-owning view over caller memory).
 */
typedef struct
{
    float   *data;                             /**< Row-major [nz][ny][nx] float32 buffer */
    uint32_t nx;                               /**< Axis 0 width [pixels] */
    uint32_t ny;                               /**< Axis 1 height [pixels] */
    uint32_t nz;                               /**< Axis 2 spectral slices (>= 1) */
    int32_t  has_wcs_lam;                      /**< 1 if explicit wavelengths were read into lam */
    float    lam_um[CACAOSIM_MASK_MAX_LAMBDA]; /**< Optional per-slice wavelengths [microns] */
    int32_t  owns_data;                        /**< 1 if @data is heap-owned by this struct */
} cacaosim_mask_cube_t;

/**
 * cacaosim_mask_view_2d() - Create a non-owning 2D (`nz = 1`) mask view around a caller buffer
 * @data: Caller-owned `[nx * ny]` float32 array (`NULL` produces an empty `nz = 0` descriptor)
 * @nx:   Mask width `[pixels]`
 * @ny:   Mask height `[pixels]`
 *
 * Role:
 *   Lightweight zero-copy adapter wrapping in-memory 2D float buffers (such as external Lyot stops
 *   or analytical test masks) into a `cacaosim_mask_cube_t` value.
 *
 * Rationale:
 *   Sets `owns_data = 0` so downstream `cacaosim_mask_free()` calls leave the caller's buffer
 *   untouched while allowing uniform consumption alongside FITS/SHM-loaded masks.
 *
 * Return: Stack-allocated `cacaosim_mask_cube_t` view with `owns_data == 0`.
 */
static inline cacaosim_mask_cube_t cacaosim_mask_view_2d(
    const float *data,
    uint32_t     nx,
    uint32_t     ny)
{
    cacaosim_mask_cube_t m = { 0 };
    m.data                 = (float *) data;
    m.nx                   = nx;
    m.ny                   = ny;
    m.nz                   = (data != NULL) ? 1u : 0u;
    m.owns_data            = 0;
    return m;
}

/**
 * cacaosim_mask_view_3d() - Create a non-owning 3D (`[nx * ny * nz]`) polychromatic mask view
 * @data:   Caller-owned `[nx * ny * nz]` float32 cube (`NULL` produces an empty `nz = 0` view)
 * @nx:     Spatial width `[pixels]`
 * @ny:     Spatial height `[pixels]`
 * @nz:     Number of wavelength slices (`< 1` clamped to `1` when `@data != NULL`)
 * @lam_um: Optional array of `@nz` slice wavelengths `[um]` (`NULL` sets `has_wcs_lam = 0`)
 *
 * Role:
 *   Zero-copy adapter wrapping caller-synthesized 3D chromatic mask cubes and optional wavelength
 *   vectors into a `cacaosim_mask_cube_t` descriptor.
 *
 * Rationale:
 *   Copies up to `CACAOSIM_MASK_MAX_LAMBDA` (`32`) wavelength entries by value into `m.lam_um[]`
 *   so the caller's temporary wavelength array does not need to outlive the view, while keeping
 *   the large pixel payload `@data` zero-copy (`owns_data = 0`).
 *
 * Return: Stack-allocated `cacaosim_mask_cube_t` view with `owns_data == 0`.
 */
static inline cacaosim_mask_cube_t cacaosim_mask_view_3d(
    const float *data,
    uint32_t     nx,
    uint32_t     ny,
    uint32_t     nz,
    const float *lam_um)
{
    cacaosim_mask_cube_t m = { 0 };
    m.data                 = (float *) data;
    m.nx                   = nx;
    m.ny                   = ny;
    m.nz                   = (data != NULL) ? ((nz >= 1u) ? nz : 1u) : 0u;
    m.owns_data            = 0;
    if (lam_um != NULL && m.nz >= 1u)
    {
        uint32_t n = (m.nz <= CACAOSIM_MASK_MAX_LAMBDA) ? m.nz : CACAOSIM_MASK_MAX_LAMBDA;
        m.has_wcs_lam = 1;
        for (uint32_t i = 0; i < n; i++)
        {
            m.lam_um[i] = lam_um[i];
        }
    }
    return m;
}

/**
 * cacaosim_mask_free() - Release heap pixel buffer if owned and zero-reset mask cube descriptor
 * @mask: Mask descriptor to clean up (safe no-op if `NULL`, empty, or a non-owning view)
 *
 * Role:
 *   Cold-path cleanup helper called after mask resampling or during context teardown.
 *
 * Rationale:
 *   Honors `mask->owns_data` before calling `free(mask->data)` and clears all struct fields to
 *   zero so repeated `cacaosim_mask_free()` invocations are strictly idempotent.
 */
void cacaosim_mask_free(cacaosim_mask_cube_t *mask);

/**
 * cacaosim_mask_load_fits() - Load a 2D or 3D primary-HDU FITS image into 64-byte aligned float32
 * @path:    Filesystem path to FITS file (leading whitespace and optional `'!'` are stripped)
 * @out:     Output mask cube descriptor (`owns_data = 1` on success; non-NULL)
 * @err_buf: Optional caller buffer receiving human-readable failure diagnostics (may be `NULL`)
 * @err_cap: Byte capacity of `@err_buf` (`0` disables error string formatting)
 *
 * Role:
 *   Cold-path disk reader parsing standard 2D/3D FITS optical masks (`BITPIX` = `-32`, `-64`,
 *   `32`, `16`, `8`, `BSCALE`/`BZERO`) and optional spectral WCS (`CRVAL3`/`CDELT3`/`CRPIX3`/
 *   `CUNIT3` or `LAM0`/`LAM1`/`DLAM`/`CSIM_WL0`/`CSIM_WL1`).
 *
 * Rationale:
 *   Self-contained RFC 4047 primary-HDU reader avoiding global `CLIcore` image tables and external
 *   `cfitsio` state so standalone `milk-fpsexec-*` processes and multi-threaded initialization
 *   load masks deterministically into SIMD-aligned `float32` buffers.
 *
 * Return: `0` on success, `-1` on I/O, header, dimension, datatype, or memory error.
 */
int cacaosim_mask_load_fits(
    const char           *path,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_mask_save_fits() - Write a 2D (`nz == 1`) or 3D (`nz > 1`) `float32` mask cube to FITS
 * @path:   Destination FITS file path (optional leading `'!'` is stripped and file overwritten)
 * @data:   Source `[nx * ny * nz]` float32 array in slice-major order (non-NULL)
 * @nx:     Spatial width along axis 1 `[pixels]` (`> 0`)
 * @ny:     Spatial height along axis 2 `[pixels]` (`> 0`)
 * @nz:     Spectral slice count along axis 3 (`>= 1`; writes 2D FITS when `1`, 3D when `> 1`)
 * @lam_um: Optional array of `@nz` wavelengths `[um]` (writes `CRPIX3`/`CRVAL3`/`CDELT3`/`CUNIT3`
 *          and `LAM0`/`LAM1` when `nz > 1` and `lam_um[0] > 0`)
 *
 * Role:
 *   Cold-path FITS exporter used by `cacaosim.mkpupil`, calibration scripts, and test suites to
 *   persist generated 2D/3D optical masks in standard big-endian `BITPIX = -32` FITS format.
 *
 * Rationale:
 *   Emits both standard FITS WCS (`CRVAL3`/`CDELT3`/`CUNIT3 = 'um'`) and concise `LAM0`/`LAM1`
 *   keywords with exact 2880-byte zero-padded FITS block alignment so saved cubes round-trip
 *   losslessly through `cacaosim_mask_load_fits()`, `milk`, DS9, and `astropy.io.fits`.
 *
 * Return: `0` on success, `-1` on invalid arguments, allocation failure, or disk write error.
 */
int cacaosim_mask_save_fits(
    const char  *path,
    const float *data,
    uint32_t     nx,
    uint32_t     ny,
    uint32_t     nz,
    const float *lam_um);

/**
 * cacaosim_mask_load() - Load a 2D or 3D optical mask from either a FITS file or MILK SHM stream
 * @spec:      Mask specifier: FITS path (e.g., `"mask.fits"`, `"!mask.fits"`, `"/dir/mask"`) or
 *             `ImageStreamIO` shared-memory stream name in `MILK_SHM_DIR`
 * @allow_shm: Non-zero to query `ImageStreamIO` shared memory when `@spec` is not a FITS path
 * @out:       Output mask cube descriptor (`owns_data = 1` on success; non-NULL)
 * @err_buf:   Optional diagnostic message buffer (may be `NULL`)
 * @err_cap:   Capacity of `@err_buf` in bytes
 *
 * Role:
 *   Primary high-level mask loader called by `cacaosim.coro` (`pup_amp`, `pup_pha`, `fpm_amp`,
 *   `fpm_pha`, `lyot_amp`, `lyot_pha`), `beamsplit`, `wfe`, `dmsim`, and `htsim`.
 *
 * Rationale:
 *   Resolves FITS paths first (explicit `'!'`, `/` path separator, `.fits`/`.fit`/`.fts` suffix,
 *   or existing readable local file via `access(R_OK)`), and otherwise attaches to live shared
 *   memory (`FLOAT` or `DOUBLE` 2D/3D stream), copying into a private aligned buffer so subsequent
 *   upstream SHM writes cannot race with precomputed optical propagation matrices.
 *
 * Return: `0` on success, `-1` if neither FITS nor SHM loading succeeded.
 */
int cacaosim_mask_load(
    const char           *spec,
    int                   allow_shm,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_mask_is_fits_path() - Heuristic check whether `@spec` names a FITS file vs. SHM stream
 * @spec: Candidate mask string (`NULL` or whitespace-only returns `0`)
 *
 * Role:
 *   Used by `.bench` serializers, TUI validators (`cacaosim-CTRL`), and `cacaosim_mask_load()` to
 *   distinguish filesystem FITS references from bare `ImageStreamIO` stream identifiers.
 *
 * Rationale:
 *   SHM stream names in `milk` are flat identifiers without directory slashes, `'!'` prefixes, or
 *   `.fits`/`.fit`/`.fts` extensions, making syntactic classification unambiguous even before a
 *   file exists on disk.
 *
 * Return: `1` if `@spec` syntactically denotes a FITS file path, `0` otherwise.
 */
int cacaosim_mask_is_fits_path(const char *spec);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_MASK_H */
