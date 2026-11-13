// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_mask.c
 * @brief   Self-contained 2D/3D FITS primary-HDU reader/writer and MILK SHM mask cube importer
 *
 * Architectural Role:
 *   Implements all mask management and I/O routines declared in `cacaosim_mask.h`:
 *     - Direct RFC 4047 FITS primary-HDU header card parsing, big-endian numeric conversion,
 *       `BSCALE`/`BZERO` calibration, and 3rd-axis WCS decoding (`cacaosim_mask_load_fits()`)
 *     - Standard big-endian `BITPIX = -32` 2D/3D FITS serialization (`cacaosim_mask_save_fits()`)
 *     - Dual-path `ImageStreamIO` + `stream_connect()` SHM cube snapshot (`cacaosim_mask_load()`)
 *
 * Design Rationale & Motivation:
 *   1. Zero dependency on `CLIcore` or stateful `cfitsio` globals: Standalone `milk-fpsexec-*`
 *      processes intentionally omit `CLIcore`, and multi-instance `.bench` initialization benefits
 *      from a re-entrant, thread-safe reader that decodes all five standard numeric FITS `BITPIX`
 *      formats (`-32`, `-64`, `32`, `16`, `8`) directly into 64-byte aligned `float32` memory.
 *   2. Broad compatibility with third-party optical design FITS cubes: Coronagraph masks exported
 *      from Python (`POPPY`, `HCIPy`, `PROPER`, `astropy`), MATLAB, or `milk` specify 3rd-axis
 *      wavelengths using varying conventions (`LAM0`/`LAM1`, `CSIM_WL0`/`CSIM_WL1`, `LAM0`+`DLAM`,
 *      or standard FITS WCS `CRVAL3`/`CDELT3`/`CRPIX3`/`CUNIT3` in `um`, `m`, `nm`, or `Angstrom`).
 *      Supporting all of them—including unitless order-of-magnitude inference (`< 1e-4 -> m`,
 *      `> 100 -> nm`) guarded by a `[0.05, 50.0] um` plausibility window—eliminates manual header
 *      patching by users.
 *   3. Two-tier SHM attachment: Querying `ImageStreamIO_read_sharedmem_image_toIMAGE()` first
 *      succeeds in lightweight standalone `fpsexec` daemons (`dcimg == NULL`), while falling back
 *      to `stream_connect()` when hosted inside an interactive `milk` session (`dcimg != NULL`)
 *      seamlessy resolves in-process image table entries.
 */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "milk_runnable.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "cacaosim_types.h"
#include "common/cacaosim_mask.h"

static void mask_err(char *buf, size_t cap, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

/**
 * mask_err() - Safely format a printf-style diagnostic string into an optional caller buffer
 * @buf: Destination character buffer (safe no-op when `NULL`)
 * @cap: Buffer capacity in bytes including terminating NUL (safe no-op when `0`)
 * @fmt: `printf`-compatible format string checked at compile time via `__attribute__((format))`
 *
 * Role:
 *   Internal error-reporting helper shared by FITS and SHM mask loaders so callers (FPS logs, TUI
 *   status bars, or unit tests) receive actionable failure context without `stderr` spam.
 *
 * Rationale:
 *   Centralizes `NULL`/`0` capacity guards around `vsnprintf()` to keep error paths concise.
 */
static void mask_err(char *buf, size_t cap, const char *fmt, ...)
{
    if (buf == NULL || cap == 0u)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
}

/**
 * cacaosim_mask_free() - Free owned mask pixel data and zero-initialize the descriptor
 * @mask: Target mask cube descriptor (`NULL` is safely ignored)
 *
 * Role:
 *   Deallocates heap memory allocated by `cacaosim_mask_load*()` while leaving borrowed memory
 *   from `cacaosim_mask_view_2d()` / `3d()` (`owns_data == 0`) intact.
 *
 * Rationale:
 *   Zeroing the entire struct via `memset` resets `data = NULL`, `nx = ny = nz = 0`, and
 *   `owns_data = 0`, preventing double-free bugs if called both at end-of-init and teardown.
 */
void cacaosim_mask_free(cacaosim_mask_cube_t *mask)
{
    if (mask == NULL)
    {
        return;
    }
    if (mask->owns_data && mask->data != NULL)
    {
        free(mask->data);
    }
    memset(mask, 0, sizeof(*mask));
}

/**
 * strip_bang() - Trim leading whitespace and an optional CFITSIO-style `'!'` prefix from a string
 * @s: Input path or specifier string (`NULL` returns `""`)
 *
 * Role:
 *   Normalizes user FITS paths such as `" !/tmp/lyot.fits"` into `"/tmp/lyot.fits"` before passing
 *   them to POSIX `fopen()` or `access()`.
 *
 * Rationale:
 *   `milk` and `cacao` scripts routinely prefix FITS filenames with `'!'` (the CFITSIO clobber
 *   convention); stripping it transparently prevents `"No such file"` errors on standard `fopen`.
 *
 * Return: Pointer into `@s` (or static `""`) past leading whitespace and at most one `'!'`.
 */
static const char *strip_bang(const char *s)
{
    if (s == NULL)
    {
        return "";
    }
    while (*s != '\0' && isspace((unsigned char) *s))
    {
        s++;
    }
    if (*s == '!')
    {
        s++;
    }
    return s;
}

/**
 * has_fits_suffix_or_slash() - Check for directory separator `'/'` or FITS filename extension
 * @s: Trimmed specifier string (non-NULL)
 *
 * Role:
 *   Syntactic helper distinguishing filesystem paths from flat `ImageStreamIO` SHM stream names.
 *
 * Rationale:
 *   Matches case-insensitive `.fits`, `.fit`, and `.fts` extensions (`n > ext_len` ensures at
 *   least a 1-character stem) as well as any relative/absolute directory slash `'/'`.
 *
 * Return: `1` if `@s` contains `'/'` or ends with a standard FITS suffix, `0` otherwise.
 */
static int has_fits_suffix_or_slash(const char *s)
{
    if (strchr(s, '/') != NULL)
    {
        return 1;
    }
    size_t n = strlen(s);
    if ((n > 5u && strcasecmp(s + n - 5u, ".fits") == 0) ||
        (n > 4u && strcasecmp(s + n - 4u, ".fit") == 0) ||
        (n > 4u && strcasecmp(s + n - 4u, ".fts") == 0))
    {
        return 1;
    }
    return 0;
}

/**
 * cacaosim_mask_is_fits_path() - Classify whether `@spec` syntactically denotes a FITS file path
 * @spec: Candidate mask specifier string (`NULL` or empty returns `0`)
 *
 * Role:
 *   Public predicate used by `.bench` inspection tools and `cacaosim_mask_load()`.
 *
 * Rationale:
 *   Recognizes an explicit `'!'` prefix, any `'/'` directory component, or `.fits`/`.fit`/`.fts`
 *   suffixes without touching the filesystem, so non-existent output paths still classify cleanly.
 *
 * Return: `1` if `@spec` looks like a FITS path, `0` if empty or a bare SHM stream name.
 */
int cacaosim_mask_is_fits_path(const char *spec)
{
    const char *s = strip_bang(spec);
    if (s[0] == '\0')
    {
        return 0;
    }
    if (spec != NULL && strchr(spec, '!') != NULL)
    {
        return 1;
    }
    return has_fits_suffix_or_slash(s);
}

/**
 * parse_fits_card_val() - Extract unquoted, trimmed value token from an 80-byte FITS header card
 * @card:    80-character raw FITS header record (not NUL-terminated)
 * @val_out: Destination NUL-terminated value string buffer (empty if card has no `'= '` value)
 * @cap:     Byte capacity of `@val_out`
 *
 * Role:
 *   Parses numeric and string FITS keyword values (`BITPIX`, `NAXIS*`, `BSCALE`, `CRVAL3`,
 *   `CUNIT3`, etc.) inside `cacaosim_mask_load_fits()`.
 *
 * Rationale:
 *   Per RFC 4047, value-bearing FITS cards carry `'= '` in bytes `8..9` and value + optional `'/'`
 *   comment in bytes `10..79`. Tracking single-quote state (`in_sq`) prevents stripping `'/'`
 *   characters that appear inside string literals (e.g., units or paths), and trimming both outer
 *   and inner FITS fixed-width string padding (`'um      '` -> `"um"`) simplifies downstream unit
 *   matching.
 */
static void parse_fits_card_val(const char card[80], char *val_out, size_t cap)
{
    val_out[0] = '\0';
    /* Non-value cards (COMMENT, HISTORY, blank, END) lack '= ' in columns 9-10 */
    if (card[8] != '=' || card[9] != ' ')
    {
        return;
    }
    char tmp[72];
    memcpy(tmp, card + 10, 70u);
    tmp[70] = '\0';

    /* Strip inline FITS '/' comment outside single quotes */
    int in_sq = 0;
    for (int i = 0; tmp[i] != '\0'; i++)
    {
        if (tmp[i] == '\'')
        {
            in_sq = !in_sq;
        }
        else if (!in_sq && tmp[i] == '/')
        {
            tmp[i] = '\0';
            break;
        }
    }

    /* Trim surrounding whitespace, then unquote and trim FITS fixed-width character strings */
    char *p = tmp;
    while (*p != '\0' && isspace((unsigned char) *p))
    {
        p++;
    }
    size_t len = strlen(p);
    while (len > 0u && isspace((unsigned char) p[len - 1u]))
    {
        p[--len] = '\0';
    }
    if (len >= 2u && p[0] == '\'' && p[len - 1u] == '\'')
    {
        p[--len] = '\0';
        p++;
        while (*p != '\0' && isspace((unsigned char) *p))
        {
            p++;
        }
        len = strlen(p);
        while (len > 0u && isspace((unsigned char) p[len - 1u]))
        {
            p[--len] = '\0';
        }
    }
    snprintf(val_out, cap, "%s", p);
}

/**
 * bswap_f32() - Convert a big-endian IEEE-754 32-bit word from FITS into native host `float`
 * @be_u32: Raw big-endian 32-bit bit pattern read from FITS payload
 *
 * Role:
 *   Decodes `BITPIX = -32` FITS pixels on little-endian x86-64 / ARM64 hosts.
 *
 * Rationale:
 *   Uses `__builtin_bswap32` + `memcpy` type-punning to strictly obey C11 strict-aliasing rules
 *   while compiling down to a single `bswap` + register move instruction.
 *
 * Return: Host-endian single-precision `float`.
 */
static float bswap_f32(uint32_t be_u32)
{
    uint32_t host = __builtin_bswap32(be_u32);
    float    f    = 0.0f;
    memcpy(&f, &host, sizeof(float));
    return f;
}

/**
 * bswap_f64() - Convert a big-endian IEEE-754 64-bit word from FITS into native host `double`
 * @be_u64: Raw big-endian 64-bit bit pattern read from FITS payload
 *
 * Role:
 *   Decodes `BITPIX = -64` double-precision FITS optical masks prior to `BSCALE`/`BZERO` scaling
 *   and narrowing to `float32`.
 *
 * Rationale:
 *   Strict-aliasing-safe `memcpy` + `__builtin_bswap64` preserves full 53-bit mantissa precision
 *   through the `bzero + bscale * val` linear transform before rounding once to `float`.
 *
 * Return: Host-endian double-precision `double`.
 */
static double bswap_f64(uint64_t be_u64)
{
    uint64_t host = __builtin_bswap64(be_u64);
    double   d    = 0.0;
    memcpy(&d, &host, sizeof(double));
    return d;
}

/**
 * to_be_f32() - Convert a native host `float` into a big-endian IEEE-754 32-bit FITS word
 * @f: Host single-precision `float` pixel value
 *
 * Role:
 *   Encodes outgoing `float32` mask pixels into standard FITS network/big-endian byte order inside
 *   `cacaosim_mask_save_fits()`.
 *
 * Rationale:
 *   Type-puns via `memcpy` and swaps bytes with `__builtin_bswap32` without violating C strict
 *   aliasing.
 *
 * Return: Big-endian 32-bit unsigned integer ready for `fwrite()`.
 */
static uint32_t to_be_f32(float f)
{
    uint32_t u = 0u;
    memcpy(&u, &f, sizeof(uint32_t));
    return __builtin_bswap32(u);
}

/**
 * cacaosim_mask_load_fits() - Read a 2D or 3D primary-HDU FITS mask into aligned `float32` memory
 * @path:    Filesystem path to FITS file (leading `'!'` stripped if present)
 * @out:     Output mask cube descriptor populated on success (`owns_data = 1`)
 * @err_buf: Optional error message buffer
 * @err_cap: Capacity of `@err_buf` in bytes
 *
 * Role:
 *   Cold-path (`MILK_COLD`) FITS mask decoder supporting 2D (`NAXIS = 2`) and 3D (`NAXIS = 3`)
 *   floating-point (`BITPIX = -32, -64`) and integer (`BITPIX = 8, 16, 32` with `BSCALE`/`BZERO`,
 *   including unsigned `BITPIX = 16` with `BZERO = 32768`) primary arrays.
 *
 * Rationale:
 *   Scans up to 64 FITS 2880-byte header blocks (`2304` cards) until `END`, reads the contiguous
 *   big-endian binary payload in a single `fread()`, converts to calibrated `float32` inside a
 *   64-byte aligned buffer, and resolves optional 3rd-axis spectral metadata via a three-tier
 *   priority order:
 *     1. Explicit endpoint keywords `LAM0`/`LAM1` (or `CSIM_WL0`/`CSIM_WL1`) `[um]`
 *     2. Start + step keywords `LAM0` + `DLAM` `[um]`
 *     3. Standard FITS WCS `CRVAL3` + `(z + 1 - CRPIX3) * CDELT3` scaled by `CUNIT3` (`um`, `m`,
 *        `nm`, `Angstrom`, or magnitude heuristic when `CUNIT3` is omitted).
 *
 * Return: `0` on success, `-1` on failure (with `@err_buf` populated).
 */
int MILK_COLD cacaosim_mask_load_fits(
    const char           *path,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap)
{
    if (out == NULL)
    {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    const char *fpath = strip_bang(path);
    if (fpath[0] == '\0')
    {
        mask_err(err_buf, err_cap, "Empty FITS mask path");
        return -1;
    }

    FILE *fp = fopen(fpath, "rb");
    if (fp == NULL)
    {
        mask_err(err_buf, err_cap, "Cannot open FITS mask '%s': %s", fpath, strerror(errno));
        return -1;
    }

    int    bitpix     = 0;
    int    naxis      = 0;
    long   naxes[4]   = { 1, 1, 1, 1 };
    double bscale     = 1.0;
    double bzero      = 0.0;
    int    has_crval3 = 0;
    int    has_cdelt3 = 0;
    double crval3     = 0.0;
    double cdelt3     = 0.0;
    double crpix3     = 1.0;
    char   cunit3[32] = "";
    double lam0       = 0.0;
    double lam1       = 0.0;
    double dlam       = 0.0;
    int    found_end  = 0;

    /* Read 2880-byte FITS header blocks (36 x 80-byte cards each) until the END card */
    char block[2880];
    for (int blk = 0; blk < 64 && !found_end; blk++)
    {
        if (fread(block, 1u, 2880u, fp) != 2880u)
        {
            fclose(fp);
            mask_err(err_buf, err_cap, "Truncated FITS header in '%s'", fpath);
            return -1;
        }
        for (int c = 0; c < 36; c++)
        {
            const char *card = block + c * 80;
            if (strncmp(card, "END     ", 8) == 0)
            {
                found_end = 1;
                break;
            }
            /* Extract 8-character keyword name and strip trailing space padding */
            char kw[9];
            memcpy(kw, card, 8u);
            kw[8] = '\0';
            for (int k = 7; k >= 0 && kw[k] == ' '; k--)
            {
                kw[k] = '\0';
            }
            char val[72];
            parse_fits_card_val(card, val, sizeof(val));
            if (val[0] == '\0')
            {
                continue;
            }

            if (strcmp(kw, "BITPIX") == 0)
            {
                bitpix = atoi(val);
            }
            else if (strcmp(kw, "NAXIS") == 0)
            {
                naxis = atoi(val);
            }
            else if (strcmp(kw, "NAXIS1") == 0)
            {
                naxes[0] = atol(val);
            }
            else if (strcmp(kw, "NAXIS2") == 0)
            {
                naxes[1] = atol(val);
            }
            else if (strcmp(kw, "NAXIS3") == 0)
            {
                naxes[2] = atol(val);
            }
            else if (strcmp(kw, "BSCALE") == 0)
            {
                bscale = strtod(val, NULL);
            }
            else if (strcmp(kw, "BZERO") == 0)
            {
                bzero = strtod(val, NULL);
            }
            else if (strcmp(kw, "CRVAL3") == 0)
            {
                crval3     = strtod(val, NULL);
                has_crval3 = 1;
            }
            else if (strcmp(kw, "CDELT3") == 0)
            {
                cdelt3     = strtod(val, NULL);
                has_cdelt3 = 1;
            }
            else if (strcmp(kw, "CRPIX3") == 0)
            {
                crpix3 = strtod(val, NULL);
            }
            else if (strcmp(kw, "CUNIT3") == 0)
            {
                snprintf(cunit3, sizeof(cunit3), "%s", val);
            }
            else if (strcmp(kw, "LAM0") == 0 || strcmp(kw, "CSIM_WL0") == 0)
            {
                lam0 = strtod(val, NULL);
            }
            else if (strcmp(kw, "LAM1") == 0 || strcmp(kw, "CSIM_WL1") == 0)
            {
                lam1 = strtod(val, NULL);
            }
            else if (strcmp(kw, "DLAM") == 0)
            {
                dlam = strtod(val, NULL);
            }
        }
    }

    if (!found_end || (naxis != 2 && naxis != 3) || naxes[0] < 1 || naxes[1] < 1 ||
        (naxis == 3 && naxes[2] < 1))
    {
        fclose(fp);
        mask_err(err_buf, err_cap, "Invalid FITS dimensions (NAXIS=%d) in '%s'", naxis, fpath);
        return -1;
    }

    uint32_t nx   = (uint32_t) naxes[0];
    uint32_t ny   = (uint32_t) naxes[1];
    uint32_t nz   = (naxis == 3) ? (uint32_t) naxes[2] : 1u;
    size_t   nele = (size_t) nx * ny * nz;

    size_t elem_bytes = 0u;
    if (bitpix == -32 || bitpix == 32)
    {
        elem_bytes = 4u;
    }
    else if (bitpix == -64)
    {
        elem_bytes = 8u;
    }
    else if (bitpix == 16)
    {
        elem_bytes = 2u;
    }
    else if (bitpix == 8)
    {
        elem_bytes = 1u;
    }
    else
    {
        fclose(fp);
        mask_err(err_buf, err_cap, "Unsupported FITS BITPIX=%d in '%s'", bitpix, fpath);
        return -1;
    }

    float *dst = (float *) cacaosim_aligned_calloc(nele, sizeof(float));
    void  *raw = malloc(nele * elem_bytes);
    if (dst == NULL || raw == NULL)
    {
        free(raw);
        free(dst);
        fclose(fp);
        mask_err(err_buf, err_cap, "Out of memory reading FITS '%s'", fpath);
        return -1;
    }

    /* Payload begins at the first 2880-byte boundary immediately following the END block */
    size_t nr = fread(raw, elem_bytes, nele, fp);
    fclose(fp);
    if (nr != nele)
    {
        free(raw);
        free(dst);
        mask_err(err_buf, err_cap, "Short FITS pixel payload in '%s'", fpath);
        return -1;
    }

    /* Byte-swap big-endian FITS data and apply physical = BZERO + BSCALE * raw_value */
    if (bitpix == -32)
    {
        const uint32_t *src = (const uint32_t *) raw;
        for (size_t i = 0; i < nele; i++)
        {
            dst[i] = (float) (bzero + bscale * (double) bswap_f32(src[i]));
        }
    }
    else if (bitpix == -64)
    {
        const uint64_t *src = (const uint64_t *) raw;
        for (size_t i = 0; i < nele; i++)
        {
            dst[i] = (float) (bzero + bscale * bswap_f64(src[i]));
        }
    }
    else if (bitpix == 32)
    {
        const uint32_t *src = (const uint32_t *) raw;
        for (size_t i = 0; i < nele; i++)
        {
            int32_t v = (int32_t) __builtin_bswap32(src[i]);
            dst[i]    = (float) (bzero + bscale * (double) v);
        }
    }
    else if (bitpix == 16)
    {
        const uint16_t *src = (const uint16_t *) raw;
        for (size_t i = 0; i < nele; i++)
        {
            int16_t v = (int16_t) __builtin_bswap16(src[i]);
            dst[i]    = (float) (bzero + bscale * (double) v);
        }
    }
    else
    {
        /* FITS BITPIX = 8 is unsigned uint8_t by standard (unlike signed BITPIX 16/32) */
        const uint8_t *src = (const uint8_t *) raw;
        for (size_t i = 0; i < nele; i++)
        {
            dst[i] = (float) (bzero + bscale * (double) src[i]);
        }
    }
    free(raw);

    out->data      = dst;
    out->nx        = nx;
    out->ny        = ny;
    out->nz        = nz;
    out->owns_data = 1;

    /* Decode optional 3rd-axis wavelength coordinates into microns [um] */
    uint32_t n_store = (nz <= CACAOSIM_MASK_MAX_LAMBDA) ? nz : CACAOSIM_MASK_MAX_LAMBDA;
    if (nz > 1u && lam0 > 0.0 && lam1 > 0.0)
    {
        out->has_wcs_lam = 1;
        for (uint32_t z = 0; z < n_store; z++)
        {
            double u      = (double) z / (double) (nz - 1u);
            out->lam_um[z] = (float) (lam0 + (lam1 - lam0) * u);
        }
    }
    else if (nz >= 1u && lam0 > 0.0 && (nz == 1u || fabs(dlam) > 0.0))
    {
        out->has_wcs_lam = 1;
        for (uint32_t z = 0; z < n_store; z++)
        {
            out->lam_um[z] = (float) (lam0 + dlam * (double) z);
        }
    }
    else if (has_crval3 && (nz == 1u || (has_cdelt3 && fabs(cdelt3) > 0.0)))
    {
        /* Normalize standard FITS WCS CUNIT3 ('m', 'nm', 'Angstrom', or implicit) to microns */
        double unit_scale = 1.0;
        if (strcasecmp(cunit3, "m") == 0 || strcasecmp(cunit3, "meter") == 0)
        {
            unit_scale = 1.0e6;
        }
        else if (strcasecmp(cunit3, "nm") == 0)
        {
            unit_scale = 1.0e-3;
        }
        else if (strcasecmp(cunit3, "a") == 0 || strcasecmp(cunit3, "angstrom") == 0)
        {
            unit_scale = 1.0e-4;
        }
        else if (cunit3[0] == '\0')
        {
            /* Unitless heuristic: < 100 nm in meters -> SI [m]; > 100 -> [nm] */
            if (crval3 > 0.0 && crval3 < 1.0e-4)
            {
                unit_scale = 1.0e6;
            }
            else if (crval3 > 100.0)
            {
                unit_scale = 1.0e-3;
            }
        }
        /* Sanity-gate converted slice-1 wavelength to optical/IR domain (0.05..50 um) */
        double l_test = (crval3 + (1.0 - crpix3) * cdelt3) * unit_scale;
        if (l_test > 0.05 && l_test < 50.0)
        {
            out->has_wcs_lam = 1;
            for (uint32_t z = 0; z < n_store; z++)
            {
                /* FITS pixel coordinates (CRPIX3) are 1-indexed: slice z corresponds to z + 1 */
                double w       = (crval3 + ((double) (z + 1u) - crpix3) * cdelt3) * unit_scale;
                out->lam_um[z] = (float) w;
            }
        }
    }

    return 0;
}

static void put_fits_card(char *hdr, int *idx, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

/**
 * put_fits_card() - Format and append one space-padded 80-byte FITS header card into a block
 * @hdr: 2880-byte FITS header block buffer (pre-filled with ASCII `' '`)
 * @idx: In/out card index (`0 .. 35`), incremented by 1 after writing
 * @fmt: `printf`-style card format string
 *
 * Role:
 *   Internal FITS header builder used by `cacaosim_mask_save_fits()`.
 *
 * Rationale:
 *   FITS headers forbid NUL bytes (`'\0'`) and require exact 80-byte space-padded ASCII records;
 *   formatting into a temporary buffer and `memcpy`-ing without the trailing NUL guarantees strict
 *   RFC 4047 compliance.
 */
static void put_fits_card(char *hdr, int *idx, const char *fmt, ...)
{
    char line[96];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n < 0)
    {
        n = 0;
    }
    char *dst = hdr + (*idx) * 80;
    memset(dst, ' ', 80u);
    memcpy(dst, line, ((size_t) n <= 80u) ? (size_t) n : 80u);
    (*idx)++;
}

/**
 * cacaosim_mask_save_fits() - Write a 2D or 3D `float32` mask cube to a standard FITS file
 * @path:   Destination FITS path (leading `'!'` stripped if present)
 * @data:   Source `[nx * ny * nz]` float32 buffer (non-NULL)
 * @nx:     Spatial width `[pixels]` (`> 0`)
 * @ny:     Spatial height `[pixels]` (`> 0`)
 * @nz:     Spectral slice count (`1` writes 2D `NAXIS = 2`; `> 1` writes 3D `NAXIS = 3`)
 * @lam_um: Optional wavelength array `[nz]` in `[um]` written as WCS + `LAM0`/`LAM1` when `nz > 1`
 *
 * Role:
 *   Cold-path (`MILK_COLD`) FITS writer persisting generated pupils, apodizers, FPMs, and test
 *   cubes to disk.
 *
 * Rationale:
 *   Produces a single 2880-byte ASCII header block, byte-swaps all floats to IEEE-754 big-endian
 *   `BITPIX = -32`, and zero-pads the binary payload up to an exact 2880-byte FITS record multiple
 *   as mandated by the FITS standard (whereas header blocks are space-padded, binary data blocks
 *   must be NUL-padded).
 *
 * Return: `0` on success, `-1` on invalid arguments, allocation failure, or I/O error.
 */
int MILK_COLD cacaosim_mask_save_fits(
    const char  *path,
    const float *data,
    uint32_t     nx,
    uint32_t     ny,
    uint32_t     nz,
    const float *lam_um)
{
    const char *fpath = strip_bang(path);
    if (fpath[0] == '\0' || data == NULL || nx == 0u || ny == 0u || nz == 0u)
    {
        return -1;
    }

    char hdr[2880];
    memset(hdr, ' ', sizeof(hdr));
    int c = 0;

    put_fits_card(hdr, &c, "SIMPLE  =                    T / Standard FITS format");
    put_fits_card(hdr, &c, "BITPIX  =                  -32 / IEEE single-precision float");
    put_fits_card(hdr, &c, "NAXIS   = %20d / Number of axes", (nz > 1u) ? 3 : 2);
    put_fits_card(hdr, &c, "NAXIS1  = %20u / Axis 1 width [pixels]", nx);
    put_fits_card(hdr, &c, "NAXIS2  = %20u / Axis 2 height [pixels]", ny);
    if (nz > 1u)
    {
        put_fits_card(hdr, &c, "NAXIS3  = %20u / Axis 3 spectral channels", nz);
        if (lam_um != NULL && lam_um[0] > 0.0f)
        {
            double l0 = (double) lam_um[0];
            double dl = (double) (lam_um[nz - 1u] - lam_um[0]) / (double) (nz - 1u);
            put_fits_card(hdr, &c, "CRPIX3  =                  1.0 / Reference channel (1-based)");
            put_fits_card(hdr, &c, "CRVAL3  = %20.8f / Channel 1 wavelength [um]", l0);
            put_fits_card(hdr, &c, "CDELT3  = %20.8f / Wavelength step [um]", dl);
            put_fits_card(hdr, &c, "CUNIT3  = 'um      '           / Wavelength unit");
            put_fits_card(hdr, &c, "LAM0    = %20.8f / First channel wavelength [um]", l0);
            put_fits_card(hdr, &c, "LAM1    = %20.8f / Last channel wavelength [um]",
                          (double) lam_um[nz - 1u]);
        }
    }
    put_fits_card(hdr, &c, "END");

    size_t    nele = (size_t) nx * ny * nz;
    uint32_t *be   = (uint32_t *) malloc(nele * sizeof(uint32_t));
    if (be == NULL)
    {
        return -1;
    }
    for (size_t i = 0; i < nele; i++)
    {
        be[i] = to_be_f32(data[i]);
    }

    FILE *fp = fopen(fpath, "wb");
    if (fp == NULL)
    {
        free(be);
        return -1;
    }

    int ok = 1;
    if (fwrite(hdr, 1u, 2880u, fp) != 2880u || fwrite(be, sizeof(uint32_t), nele, fp) != nele)
    {
        ok = 0;
    }

    /* Pad binary data payload with NUL bytes to complete the final 2880-byte FITS record */
    size_t pay_bytes = nele * sizeof(uint32_t);
    size_t rem       = pay_bytes % 2880u;
    if (ok && rem != 0u)
    {
        size_t pad = 2880u - rem;
        char   zeros[2880];
        memset(zeros, 0, pad);
        if (fwrite(zeros, 1u, pad, fp) != pad)
        {
            ok = 0;
        }
    }

    free(be);
    if (fclose(fp) != 0)
    {
        ok = 0;
    }
    return ok ? 0 : -1;
}

/**
 * load_from_image() - Snapshot a 2D or 3D `FLOAT`/`DOUBLE` MILK `IMAGE` into an owned mask cube
 * @im:      Source mapped `ImageStreamIO` `IMAGE`
 * @name:    Stream identifier for diagnostic error messages
 * @out:     Output mask cube descriptor (`owns_data = 1` on success)
 * @err_buf: Optional error message buffer
 * @err_cap: Capacity of `@err_buf` in bytes
 *
 * Role:
 *   Internal helper shared by both direct `ImageStreamIO` and `stream_connect()` SHM lookup paths
 *   in `cacaosim_mask_load()`.
 *
 * Rationale:
 *   Copies SHM pixels (converting `_DATATYPE_DOUBLE` to `float` if needed) into a private 64-byte
 *   aligned heap buffer so the caller can immediately unmap/close the SHM handle without holding
 *   shared-memory file descriptors open or risking mid-propagation races, and extracts optional
 *   `LAM0`/`LAM1` or `CSIM_WL0`/`CSIM_WL1` keywords when `nz > 1`.
 *
 * Return: `0` on success, `-1` on invalid axis/datatype or allocation failure.
 */
static int load_from_image(
    const IMAGE          *im,
    const char           *name,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap)
{
    if (im == NULL || im->md == NULL)
    {
        mask_err(err_buf, err_cap, "Invalid SHM image '%s'", name);
        return -1;
    }
    uint8_t nax = im->md->naxis;
    uint8_t dt  = im->md->datatype;
    if ((nax != 2u && nax != 3u) || (dt != _DATATYPE_FLOAT && dt != _DATATYPE_DOUBLE))
    {
        mask_err(err_buf, err_cap, "SHM mask '%s' must be 2D/3D FLOAT or DOUBLE", name);
        return -1;
    }

    uint32_t nx   = im->md->size[0];
    uint32_t ny   = im->md->size[1];
    uint32_t nz   = (nax == 3u && im->md->size[2] >= 1u) ? im->md->size[2] : 1u;
    size_t   nele = (size_t) nx * ny * nz;

    float *dst = (float *) cacaosim_aligned_calloc(nele, sizeof(float));
    if (dst == NULL)
    {
        mask_err(err_buf, err_cap, "Out of memory copying SHM mask '%s'", name);
        return -1;
    }

    if (dt == _DATATYPE_FLOAT)
    {
        memcpy(dst, im->array.F, nele * sizeof(float));
    }
    else
    {
        for (size_t i = 0; i < nele; i++)
        {
            dst[i] = (float) im->array.D[i];
        }
    }

    memset(out, 0, sizeof(*out));
    out->data      = dst;
    out->nx        = nx;
    out->ny        = ny;
    out->nz        = nz;
    out->owns_data = 1;

    /* Inspect SHM keyword table for explicit wavelength endpoints on 3D polychromatic masks */
    if (nz > 1u && im->kw != NULL && im->md->NBkw > 0)
    {
        double l0 = 0.0;
        double l1 = 0.0;
        for (uint16_t k = 0; k < im->md->NBkw; k++)
        {
            if (im->kw[k].type == 'D')
            {
                if (strcmp(im->kw[k].name, "LAM0") == 0 || strcmp(im->kw[k].name, "CSIM_WL0") == 0)
                {
                    l0 = im->kw[k].value.numf;
                }
                else if (strcmp(im->kw[k].name, "LAM1") == 0 ||
                         strcmp(im->kw[k].name, "CSIM_WL1") == 0)
                {
                    l1 = im->kw[k].value.numf;
                }
            }
        }
        if (l0 > 0.0 && l1 > 0.0)
        {
            uint32_t n_store = (nz <= CACAOSIM_MASK_MAX_LAMBDA) ? nz : CACAOSIM_MASK_MAX_LAMBDA;
            out->has_wcs_lam = 1;
            for (uint32_t z = 0; z < n_store; z++)
            {
                double u       = (double) z / (double) (nz - 1u);
                out->lam_um[z] = (float) (l0 + (l1 - l0) * u);
            }
        }
    }

    return 0;
}

/**
 * cacaosim_mask_load() - Resolve and load a 2D/3D optical mask from FITS path or MILK SHM stream
 * @spec:      FITS filename/path or `ImageStreamIO` shared-memory stream name
 * @allow_shm: Non-zero to permit SHM lookup when `@spec` is not a FITS path or existing file
 * @out:       Output mask cube descriptor (`owns_data = 1` on success)
 * @err_buf:   Optional diagnostic message buffer
 * @err_cap:   Capacity of `@err_buf` in bytes
 *
 * Role:
 *   Cold-path (`MILK_COLD`) unified mask loader invoked across `cacaosim` optical modules.
 *
 * Rationale:
 *   Follows a deterministic four-stage resolution cascade:
 *     1. If `@spec` carries `'!'`, `'/'`, a `.fits`/`.fit`/`.fts` suffix, or exists as a readable
 *        file in the working directory (`access(s, R_OK) == 0`), load directly as FITS.
 *     2. If `@allow_shm == 0`, still attempt `cacaosim_mask_load_fits()` to surface an OS `errno`
 *        file error.
 *     3. Map directly from `MILK_SHM_DIR` via `ImageStreamIO_read_sharedmem_image_toIMAGE()` (works
 *        in standalone `milk-fpsexec-*` processes where `dcimg == NULL`) and close immediately.
 *     4. If hosted inside a `milk` CLI session (`dcimg != NULL`), fall back to `stream_connect()`
 *        to resolve process-local images before reporting failure.
 *
 * Return: `0` on success, `-1` on failure.
 */
int MILK_COLD cacaosim_mask_load(
    const char           *spec,
    int                   allow_shm,
    cacaosim_mask_cube_t *out,
    char                 *err_buf,
    size_t                err_cap)
{
    if (out == NULL)
    {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    const char *s = strip_bang(spec);
    if (s[0] == '\0')
    {
        mask_err(err_buf, err_cap, "Empty mask specifier");
        return -1;
    }

    int had_bang = (spec != NULL && strchr(spec, '!') != NULL);
    if (had_bang || has_fits_suffix_or_slash(s) || access(s, R_OK) == 0)
    {
        return cacaosim_mask_load_fits(s, out, err_buf, err_cap);
    }

    if (!allow_shm)
    {
        return cacaosim_mask_load_fits(s, out, err_buf, err_cap);
    }

    /* Standalone-safe SHM attachment directly via ImageStreamIO without CLIcore globals */
    IMAGE im;
    memset(&im, 0, sizeof(im));
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(s, &im) == IMAGESTREAMIO_SUCCESS)
    {
        int rc = load_from_image(&im, s, out, err_buf, err_cap);
        ImageStreamIO_closeIm(&im);
        return rc;
    }

    /* Hosted `milk` session fallback when global image table `dcimg` is active */
    if (dcimg != NULL)
    {
        IMGID img = stream_connect(s);
        if (img.ID != -1 && img.im != NULL)
        {
            int rc = load_from_image(img.im, s, out, err_buf, err_cap);
            imgid_free(&img);
            return rc;
        }
    }

    mask_err(err_buf, err_cap, "Cannot open mask '%s' as FITS file or MILK SHM stream", s);
    return -1;
}
