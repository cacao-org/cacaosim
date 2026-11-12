// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mask.h
 * @brief   2D and 3D polychromatic optical mask loader/writer (FITS primary HDU + MILK SHM)
 */

#ifndef CACAOSIM_MASK_H
#define CACAOSIM_MASK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CACAOSIM_MASK_MAX_LAMBDA 32

/**
 * @brief 2D achromatic [nx, ny] or 3D polychromatic [nx, ny, nz] float32 optical mask cube.
 *
 * When @nz > 1, the 3rd axis (slice index z = 0 .. nz - 1, stride nx * ny) encodes wavelength
 * across the optical bandpass.
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
 * cacaosim_mask_view_2d() - Wrap a caller-owned 2D [nx * ny] float32 array without copying
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
 * cacaosim_mask_view_3d() - Wrap a caller-owned 3D [nx * ny * nz] float32 wavelength cube
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
 * cacaosim_mask_free() - Release heap buffer if owned and reset descriptor
 */
void cacaosim_mask_free(cacaosim_mask_cube_t *mask);

/**
 * cacaosim_mask_load_fits() - Load a 2D or 3D primary-HDU FITS file into float32 memory
 * @path:    Path to FITS file (optional leading '!' is stripped)
 * @out:     Output mask cube descriptor (owns_data = 1 on success)
 * @err_buf: Optional diagnostic message buffer
 * @err_cap: Capacity of @err_buf in bytes
 *
 * Supports BITPIX = -32, -64, 8, 16, 32, BSCALE/BZERO, and optional spectral axis keywords
 * (CRVAL3/CDELT3/CRPIX3/CUNIT3 or LAM0/LAM1/DLAM).
 *
 * Return: 0 on success, -1 on error.
 */
int cacaosim_mask_load_fits(
    const char           *path,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_mask_save_fits() - Save a 2D (nz == 1) or 3D (nz > 1) float32 mask cube to FITS
 * @path:   Destination FITS file path (optional leading '!' is stripped)
 * @data:   Source [nx * ny * nz] float32 array
 * @nx:     Axis 0 width [pixels]
 * @ny:     Axis 1 height [pixels]
 * @nz:     Axis 2 wavelength slice count (>= 1)
 * @lam_um: Optional wavelength array [nz] in microns (writes CRVAL3/CDELT3/CUNIT3 when nz > 1)
 *
 * Return: 0 on success, -1 on error.
 */
int cacaosim_mask_save_fits(
    const char  *path,
    const float *data,
    uint32_t     nx,
    uint32_t     ny,
    uint32_t     nz,
    const float *lam_um);

/**
 * cacaosim_mask_load() - Load a 2D or 3D mask from either a FITS path or MILK_SHM_DIR stream
 * @spec:      FITS file path (e.g. "mask.fits", "!mask.fits", "/path/mask.fits") or SHM name
 * @allow_shm: Non-zero to query ImageStreamIO shared memory when @spec is not a file path
 * @out:       Output mask cube descriptor (owns_data = 1 on success)
 * @err_buf:   Optional diagnostic message buffer
 * @err_cap:   Capacity of @err_buf in bytes
 *
 * Return: 0 on success, -1 on error.
 */
int cacaosim_mask_load(
    const char           *spec,
    int                   allow_shm,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap);

/**
 * cacaosim_mask_is_fits_path() - Return 1 if @spec refers to a FITS file path rather than a SHM ID
 */
int cacaosim_mask_is_fits_path(const char *spec);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_MASK_H */
