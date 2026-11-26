// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_plan.c
 * @brief   Declarative `.bench` -> `fpspipe` deployment-plan compiler for `cacaosim.bash`
 *
 * Architectural Role:
 *   Provides a single validated C source of truth (`cacaosim_bench_emit_plan()`) translating
 *   either a multi-stage `.bench` DAG (`cacaosim_bench_load_file()` + `cacaosim_bench_analyze()`)
 *   or a zero-config flat single-stage setup into declarative batch files consumed by
 *   `share/milk/commands.d/cacaosim.bash` (`fpspipe.loadfits`, `fpspipe.init`,
 *   `fpspipe.setparams`, `milk-fps-runstart`, and `milk-stream-wait`):
 *     - `meta.tsv`:       Bench metadata, pupil synthesis parameters, and 64-bit FNV-1a signature.
 *     - `loadfits.tsv`:   `<shm_stem>\t<abs_fits_path>` pairs for 2D/3D `.fits`/`.fits.gz`/`.fz`
 *                         masks, rewriting `.bench` mask specifiers to `< 48`-char SHM stems.
 *     - `units.manifest`: Parallel `milk-fps-init -b` unit declarations clearing optional strings.
 *     - `params.batch`:   Atomic two-pass `milk-fps-set -b` parameter assignments.
 *     - `waves.tsv`:      Forward topological `RUNSTART` and event-driven `WAIT` (`+1`) steps.
 *     - `units.list`:     Tracked FPS unit names for warm pruning and `cacaosim.teardown`.
 *     - `streams.list`:   Tracked SHM stream stems created or staged by this bench.
 *
 * Design Rationale & Motivation:
 *   1. Eliminating Duplicate Shell/AWK Parsers: Driving deployment from `cacaosim_bench_cfg.c` and
 *      `cacaosim_bench_analyze()` guarantees identical enum/alias parsing, cycle/orphan detection,
 *      `eff_navg` decimation, and sensor `bw_frac` / `dt_exp` derivation across `htsim`, modular
 *      FPS units, `cacaosim-bench-view`, and `cacaosim-CTRL`.
 *   2. Arbitrary Path Length & Compressed FITS Support: `FUNCTION_PARAMETER_STRMAXLEN` (`64`) caps
 *      FPS string parameters, whereas real laboratory/test FITS mask paths often exceed 64 bytes
 *      or use `.fits.gz` / `.fits.fz` compression. Staging masks into SHM via `fpspipe.loadfits`
 *      and rewriting `csim_active.bench` to compact `csim_m_*` SHM stems removes path-length limits
 *      and preserves 3D spectral WCS keywords (`CRVAL3`/`CDELT3`/`CUNIT3`) in SHM.
 *   3. Zero-Sleep Topological Readiness: Ordering modular startup in forward DAG waves (upstream
 *      producers before `triggermode = 3` consumers) and emitting exact `milk-stream-wait` specs
 *      (`:WxH`, `:WxHx2+1`, or `+1` for polychromatic/fast streams) eliminates fixed `sleep` polls.
 */

#include <ctype.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "bench/cacaosim_bench.h"
#include "common/cacaosim_mask.h"

/** Maximum FITS mask files staged per bench deployment plan. */
#define PLAN_MAX_FITS    128
/** Maximum tracked FPS units per deployment plan. */
#define PLAN_MAX_UNITS   64
/** Maximum tracked SHM streams per deployment plan. */
#define PLAN_MAX_STREAMS 256

/**
 * struct fits_stage_t - Record mapping a resolved disk FITS file to its staged SHM stream stem
 * @shm_stem: Compact `< 48`-char SHM stream identifier written to `loadfits.tsv`
 * @abs_path: Canonical filesystem path to the `.fits` / `.fits.gz` / `.fits.fz` file
 */
typedef struct
{
    char shm_stem[CACAOSIM_BENCH_NAME_LEN];
    char abs_path[PATH_MAX];
} fits_stage_t;

/**
 * plan_err() - Format a diagnostic error string into caller-supplied `@err_buf`
 * @err_buf: Optional destination character buffer (ignored if `NULL` or `@err_cap == 0`)
 * @err_cap: Capacity of `@err_buf` in bytes
 * @fmt:     `printf`-style format string
 */
static void plan_err(char *err_buf, size_t err_cap, const char *fmt, ...)
{
    if (err_buf == NULL || err_cap == 0)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err_buf, err_cap, fmt, ap);
    va_end(ap);
}

/**
 * fnv1a_update() - Mix `@len` raw bytes into a 64-bit FNV-1a rolling hash
 * @h:    Current 64-bit hash accumulator (`14695981039346656037ULL` seed)
 * @data: Byte buffer to mix
 * @len:  Byte length of `@data`
 *
 * Return: Updated 64-bit FNV-1a digest.
 */
static uint64_t fnv1a_update(uint64_t h, const void *data, size_t len)
{
    const unsigned char *p = (const unsigned char *) data;
    for (size_t i = 0; i < len; i++)
    {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/**
 * fnv1a_str() - Mix a NUL-terminated string (including trailing NUL separator) into FNV-1a
 * @h: Current 64-bit hash accumulator
 * @s: String to mix (`NULL` treated as `""`)
 *
 * Return: Updated 64-bit FNV-1a digest.
 */
static uint64_t fnv1a_str(uint64_t h, const char *s)
{
    const char *v = (s != NULL) ? s : "";
    return fnv1a_update(h, v, strlen(v) + 1u);
}

/**
 * has_exe_in_path() - Check whether an executable exists on `PATH` with `X_OK` permission
 * @exe_name: Executable basename (e.g. `"milk-fpsexec-atmturb-mkwfs"`)
 *
 * Return: `1` if found and executable, `0` otherwise.
 */
static int has_exe_in_path(const char *exe_name)
{
    if (exe_name == NULL || exe_name[0] == '\0')
    {
        return 0;
    }
    if (strchr(exe_name, '/') != NULL)
    {
        return (access(exe_name, X_OK) == 0) ? 1 : 0;
    }
    const char *path_env = getenv("PATH");
    if (path_env == NULL || path_env[0] == '\0')
    {
        return 0;
    }
    const char *cur = path_env;
    while (*cur != '\0')
    {
        const char *sep = strchr(cur, ':');
        size_t      dlen = (sep != NULL) ? (size_t) (sep - cur) : strlen(cur);
        char        cand[PATH_MAX];
        if (dlen == 0)
        {
            snprintf(cand, sizeof(cand), "./%s", exe_name);
        }
        else if (dlen < sizeof(cand) - 64u)
        {
            snprintf(cand, sizeof(cand), "%.*s/%s", (int) dlen, cur, exe_name);
        }
        else
        {
            cand[0] = '\0';
        }
        if (cand[0] != '\0' && access(cand, X_OK) == 0)
        {
            return 1;
        }
        if (sep == NULL)
        {
            break;
        }
        cur = sep + 1;
    }
    return 0;
}

/**
 * sanitize_stem() - Sanitize an identifier into `[A-Za-z0-9_]` capped at `@max_chars`
 * @raw:       Input string
 * @out:       Destination buffer
 * @out_cap:   Capacity of `@out` in bytes
 * @max_chars: Maximum output length before NUL
 */
static void sanitize_stem(const char *raw, char *out, size_t out_cap, size_t max_chars)
{
    if (out_cap == 0)
    {
        return;
    }
    size_t lim = (max_chars < out_cap - 1u) ? max_chars : (out_cap - 1u);
    size_t j   = 0;
    if (raw != NULL)
    {
        for (size_t i = 0; raw[i] != '\0' && j < lim; i++)
        {
            unsigned char c = (unsigned char) raw[i];
            out[j++]        = (isalnum(c) || c == '_') ? (char) c : '_';
        }
    }
    if (j == 0 && lim > 0)
    {
        out[j++] = 'x';
    }
    out[j] = '\0';
}

/**
 * add_unique_item() - Append `@name` to a fixed-capacity string set if non-empty and new
 * @arr:     Target 2D character array
 * @cnt:     In/out item count
 * @max_cnt: Maximum capacity of `@arr`
 * @name:    Candidate string (`NULL` or `""` ignored)
 */
static void add_unique_item(
    char        arr[][CACAOSIM_BENCH_NAME_LEN],
    int        *cnt,
    int         max_cnt,
    const char *name)
{
    if (name == NULL || name[0] == '\0' || *cnt >= max_cnt)
    {
        return;
    }
    for (int i = 0; i < *cnt; i++)
    {
        if (strcmp(arr[i], name) == 0)
        {
            return;
        }
    }
    snprintf(arr[*cnt], CACAOSIM_BENCH_NAME_LEN, "%s", name);
    (*cnt)++;
}

/**
 * stage_fits_field() - Resolve a FITS mask field, register it in `loadfits`, and rewrite to SHM
 * @field:     In/out specifier buffer (`cfg.*` or `nd->*`)
 * @field_cap: Capacity of `@field` in bytes
 * @bench_dir: Directory of the source `.bench` file for relative path lookup
 * @node_tag:  Short sanitized owner tag (e.g. `"bench"` or node `id`)
 * @role_tag:  Short mask role suffix (e.g. `"pamp"`, `"fpha"`, `"lamp"`)
 * @stages:    Staging table populated with unique `<shm_stem, abs_path>` pairs
 * @n_stages:  In/out count of entries in `@stages`
 * @streams:   Tracked stream table updated with `@shm_stem`
 * @n_streams: In/out count of entries in `@streams`
 * @sig_hash:  Rolling FNV-1a signature updated with file path, byte size, and nanosecond mtime
 * @err_buf:   Diagnostic error buffer populated if a referenced FITS file does not exist
 * @err_cap:   Capacity of `@err_buf`
 *
 * Return: `1` if `@field` was a FITS path and rewritten to an SHM stem, `0` if unchanged SHM/empty,
 *         or `-1` if the FITS file could not be found on disk.
 */
static int stage_fits_field(
    char         *field,
    size_t        field_cap,
    const char   *bench_dir,
    const char   *node_tag,
    const char   *role_tag,
    fits_stage_t *stages,
    int          *n_stages,
    char          streams[][CACAOSIM_BENCH_NAME_LEN],
    int          *n_streams,
    uint64_t     *sig_hash,
    char         *err_buf,
    size_t        err_cap)
{
    if (field == NULL || field[0] == '\0' || !cacaosim_mask_is_fits_path(field))
    {
        return 0;
    }

    const char *raw = (field[0] == '!') ? (field + 1) : field;
    char        cand[PATH_MAX];
    struct stat st;
    int         found = 0;

    if (stat(raw, &st) == 0 && S_ISREG(st.st_mode))
    {
        snprintf(cand, sizeof(cand), "%s", raw);
        found = 1;
    }
    else if (raw[0] != '/' && bench_dir != NULL && bench_dir[0] != '\0')
    {
        snprintf(cand, sizeof(cand), "%.2048s/%.1024s", bench_dir, raw);
        if (stat(cand, &st) == 0 && S_ISREG(st.st_mode))
        {
            found = 1;
        }
    }

    if (!found)
    {
        plan_err(err_buf, err_cap, "FITS mask file not found: '%s'", raw);
        return -1;
    }

    char abs_path[PATH_MAX];
    if (realpath(cand, abs_path) == NULL)
    {
        snprintf(abs_path, sizeof(abs_path), "%s", cand);
    }

    int64_t fsize  = (int64_t) st.st_size;
    int64_t mt_sec = (int64_t) st.st_mtim.tv_sec;
    int64_t mt_ns  = (int64_t) st.st_mtim.tv_nsec;
    *sig_hash      = fnv1a_str(*sig_hash, abs_path);
    *sig_hash      = fnv1a_update(*sig_hash, &fsize, sizeof(fsize));
    *sig_hash      = fnv1a_update(*sig_hash, &mt_sec, sizeof(mt_sec));
    *sig_hash      = fnv1a_update(*sig_hash, &mt_ns, sizeof(mt_ns));

    for (int i = 0; i < *n_stages; i++)
    {
        if (strcmp(stages[i].abs_path, abs_path) == 0)
        {
            snprintf(field, field_cap, "%s", stages[i].shm_stem);
            return 1;
        }
    }

    if (*n_stages >= PLAN_MAX_FITS)
    {
        plan_err(err_buf, err_cap, "Exceeded max staged FITS files (%d)", PLAN_MAX_FITS);
        return -1;
    }

    char clean_tag[32];
    sanitize_stem(node_tag, clean_tag, sizeof(clean_tag), 22u);

    char stem[CACAOSIM_BENCH_NAME_LEN];
    snprintf(stem, sizeof(stem), "csim_m_%s_%s", clean_tag, role_tag);

    /* Disambiguate in the unlikely event two truncated node IDs collide on the same role */
    int suffix = 2;
    for (int i = 0; i < *n_stages; i++)
    {
        if (strcmp(stages[i].shm_stem, stem) == 0)
        {
            snprintf(stem, sizeof(stem), "csim_m_%s_%s%d", clean_tag, role_tag, suffix++);
            i = -1;
        }
    }

    snprintf(stages[*n_stages].shm_stem, sizeof(stages[*n_stages].shm_stem), "%s", stem);
    snprintf(stages[*n_stages].abs_path, sizeof(stages[*n_stages].abs_path), "%s", abs_path);
    (*n_stages)++;

    add_unique_item(streams, n_streams, PLAN_MAX_STREAMS, stem);
    snprintf(field, field_cap, "%s", stem);
    return 1;
}

/**
 * open_plan_file() - Open `<plan_dir>/<rel_name>` for writing with diagnostic error reporting
 * @plan_dir: Target plan directory
 * @rel_name: Relative filename inside `@plan_dir`
 * @err_buf:  Optional error buffer
 * @err_cap:  Capacity of `@err_buf`
 *
 * Return: Open `FILE *` handle, or `NULL` on error.
 */
static FILE *open_plan_file(
    const char *plan_dir,
    const char *rel_name,
    char       *err_buf,
    size_t      err_cap)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%.3000s/%.256s", plan_dir, rel_name);
    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        plan_err(err_buf, err_cap, "Cannot write '%s': %s", path, strerror(errno));
    }
    return fp;
}

/**
 * emit_str_if_set() - Write `<fps>.<key> = <val>` to `params.batch` only when `@val` is non-empty
 * @fp:  Open `params.batch` stream
 * @fps: Target FPS unit name
 * @key: Parameter suffix starting with `.` (e.g. `".in_beam"`)
 * @val: String value (`NULL` or `""` skipped because `milk-fps-set -b` forbids empty RHS)
 */
static void emit_str_if_set(FILE *fp, const char *fps, const char *key, const char *val)
{
    if (val != NULL && val[0] != '\0')
    {
        fprintf(fp, "%s%s = %s\n", fps, key, val);
    }
}

/**
 * emit_det_params() - Emit common `cacaosim_det_cfg_t` electronics parameters to `params.batch`
 * @fp:  Open `params.batch` stream
 * @fps: Target sensor FPS unit name
 * @det: Detector configuration struct
 */
static void emit_det_params(FILE *fp, const char *fps, const cacaosim_det_cfg_t *det)
{
    fprintf(fp, "%s.flux_ph = %.8g\n", fps, det->flux_ph);
    fprintf(fp, "%s.dark_e = %.8g\n", fps, det->dark_e);
    fprintf(fp, "%s.ron_e = %.8g\n", fps, det->ron_e);
    fprintf(fp, "%s.gain_eadu = %.8g\n", fps, det->gain_eadu);
    fprintf(fp, "%s.bias_adu = %.8g\n", fps, det->bias_adu);
    fprintf(fp, "%s.sat_adu = %.8g\n", fps, det->sat_adu);
    fprintf(fp, "%s.shutter = %s\n", fps, det->shutter ? "ON" : "OFF");
    fprintf(fp, "%s.photon_noise = %s\n", fps, det->photon_noise ? "ON" : "OFF");
    fprintf(fp, "%s.out_dtype = %d\n", fps, det->out_dtype);
}

/**
 * emit_reg_params() - Emit 2D pupil affine registration parameters to `params.batch`
 * @fp:  Open `params.batch` stream
 * @fps: Target FPS unit name
 * @reg: Registration struct (`shift_x`, `shift_y`, `rot_deg`, `scale`)
 */
static void emit_reg_params(FILE *fp, const char *fps, const cacaosim_reg_t *reg)
{
    float sc = (reg->scale > 0.0f) ? reg->scale : 1.0f;
    fprintf(fp, "%s.shift_x = %.8g\n", fps, reg->shift_x);
    fprintf(fp, "%s.shift_y = %.8g\n", fps, reg->shift_y);
    fprintf(fp, "%s.rot_deg = %.8g\n", fps, reg->rot_deg);
    fprintf(fp, "%s.mag_scale = %.8g\n", fps, sc);
}

/**
 * emit_atmturb_plan() - Append `csim_atmturb` manifest, batch params, and wave-0 readiness step
 * @f_man:     Open `units.manifest` stream
 * @f_bat:     Open `params.batch` stream
 * @f_wav:     Open `waves.tsv` stream
 * @nwf:       Wavefront grid size `[pixels]`
 * @eff_rate:  Master clock frequency `[Hz]`
 * @pup_diam:  Pupil diameter fraction of `@nwf`
 * @tel_diam:  Primary mirror diameter `[m]`
 * @ref_lam:   Turbulence reference wavelength `[um]`
 * @seeing:    Seeing FWHM `[arcsec]`
 * @master_sz: Master turbulence screen size `[pixels]`
 * @out_pha:   Output phase stream name (e.g. `"outarraypha"`)
 * @units:     Tracked unit set
 * @n_units:   In/out unit count
 * @streams:   Tracked stream set
 * @n_streams: In/out stream count
 */
static void emit_atmturb_plan(
    FILE       *f_man,
    FILE       *f_bat,
    FILE       *f_wav,
    uint32_t    nwf,
    float       eff_rate,
    float       pup_diam,
    float       tel_diam,
    float       ref_lam,
    float       seeing,
    uint32_t    master_sz,
    const char *out_pha,
    float       zenith_deg,
    float       disp_pa_deg,
    float       site_alt_m,
    int         chrom_turb,
    const char *chrom_lambdas,
    char        units[][CACAOSIM_BENCH_NAME_LEN],
    int        *n_units,
    char        streams[][CACAOSIM_BENCH_NAME_LEN],
    int        *n_streams)
{
    float dt_s      = (eff_rate > 0.1f) ? (1.0f / eff_rate) : 0.0005f;
    float denom     = (float) nwf * pup_diam;
    float pup_scale = (denom > 0.0f) ? (tel_diam / denom) : 0.04f;
    const char *pha = (out_pha != NULL && out_pha[0] != '\0') ? out_pha : "outarraypha";
    float z_rad     = zenith_deg * (3.14159265358979323846f / 180.0f);
    float pa_rad    = disp_pa_deg * (3.14159265358979323846f / 180.0f);

    fprintf(f_man, "csim_atmturb milk-fpsexec-atmturb-mkwfs\n");

    fprintf(f_bat, "csim_atmturb.slambda = %.6f\n", ref_lam);
    fprintf(f_bat, "csim_atmturb.ref_lambda = %.6f\n", ref_lam);
    fprintf(f_bat, "csim_atmturb.wfsize = %u\n", nwf);
    fprintf(f_bat, "csim_atmturb.pupil_scale = %.6f\n", pup_scale);
    fprintf(f_bat, "csim_atmturb.seeing = %.6g\n", seeing);
    fprintf(f_bat, "csim_atmturb.zenith_angle = %.6f\n", z_rad);
    fprintf(f_bat, "csim_atmturb.parallactic_angle = %.6f\n", pa_rad);
    fprintf(f_bat, "csim_atmturb.site_alt = %.6g\n", site_alt_m);
    if (chrom_turb && chrom_lambdas != NULL && chrom_lambdas[0] != '\0')
    {
        fprintf(f_bat, "csim_atmturb.chrom_lambdas = %s\n", chrom_lambdas);
        fprintf(f_bat, "csim_atmturb.chrom_opd_um = 1\n");
    }
    fprintf(f_bat, "csim_atmturb.time_step = %.7f\n", dt_s);
    fprintf(f_bat, "csim_atmturb.master_size = %u\n", master_sz);
    fprintf(f_bat, "csim_atmturb.stream_mode = 1\n");
    fprintf(f_bat, "csim_atmturb.save_fits = 0\n");
    fprintf(f_bat, "csim_atmturb.out_phase = %s\n", pha);
    fprintf(f_bat, "csim_atmturb.procinfo.triggermode = 4\n");
    fprintf(f_bat, "csim_atmturb.procinfo.triggerdelay = %.7f\n", dt_s);
    fprintf(f_bat, "csim_atmturb.procinfo.loopcntMax = -1\n");

    fprintf(f_wav, "RUNSTART\tcsim_atmturb\n");
    if (chrom_turb && chrom_lambdas != NULL && chrom_lambdas[0] != '\0')
    {
        fprintf(f_wav, "WAIT\t%s+1\n", pha);
    }
    else
    {
        fprintf(f_wav, "WAIT\t%s:%ux%u+1\n", pha, nwf, nwf);
    }

    add_unique_item(units, n_units, PLAN_MAX_UNITS, "csim_atmturb");
    add_unique_item(streams, n_streams, PLAN_MAX_STREAMS, pha);
}

void cacaosim_plan_opts_defaults(cacaosim_plan_opts_t *opts)
{
    if (opts == NULL)
    {
        return;
    }
    memset(opts, 0, sizeof(*opts));
    opts->modular_mode    = 0;
    opts->flat_mode       = 0;
    opts->with_atmturb    = 0;
    opts->seeing_arcsec   = 0.8f;
    opts->atm_master_size = 0u;
    opts->rate_override   = 0.0f;
    opts->backend         = -1;
    opts->flat_nwf        = 128u;
    opts->flat_nact       = 20u;
    opts->flat_wfs_ncam   = 64u;
    opts->flat_fp_ncam    = 128u;
    opts->flat_mod_rad    = 3.0f;
    snprintf(opts->active_bench_arg, sizeof(opts->active_bench_arg), "csim_active.bench");
}

/**
 * write_list_file() - Emit a newline-delimited list file inside `@plan_dir`
 * @plan_dir: Destination plan directory
 * @rel_name: Relative filename (`units.list` or `streams.list`)
 * @items:    Array of strings
 * @n_items:  Count of active strings
 * @err_buf:  Optional error buffer
 * @err_cap:  Capacity of `@err_buf`
 *
 * Return: `0` on success, `-1` on write failure.
 */
static int write_list_file(
    const char *plan_dir,
    const char *rel_name,
    char        items[][CACAOSIM_BENCH_NAME_LEN],
    int         n_items,
    char       *err_buf,
    size_t      err_cap)
{
    FILE *fp = open_plan_file(plan_dir, rel_name, err_buf, err_cap);
    if (fp == NULL)
    {
        return -1;
    }
    for (int i = 0; i < n_items; i++)
    {
        fprintf(fp, "%s\n", items[i]);
    }
    if (fclose(fp) != 0)
    {
        plan_err(err_buf, err_cap, "Failed flushing '%s/%s'", plan_dir, rel_name);
        return -1;
    }
    return 0;
}

int cacaosim_bench_emit_plan(
    const char                 *bench_file,
    const char                 *plan_dir,
    const cacaosim_plan_opts_t *opts_in,
    char                       *err_buf,
    size_t                      err_cap)
{
    if (plan_dir == NULL || plan_dir[0] == '\0')
    {
        plan_err(err_buf, err_cap, "Empty plan_dir passed to cacaosim_bench_emit_plan()");
        return -1;
    }

    cacaosim_plan_opts_t opts;
    if (opts_in != NULL)
    {
        opts = *opts_in;
    }
    else
    {
        cacaosim_plan_opts_defaults(&opts);
    }

    if (mkdir(plan_dir, 0775) != 0 && errno != EEXIST)
    {
        plan_err(err_buf, err_cap, "Cannot create plan_dir '%s': %s", plan_dir, strerror(errno));
        return -1;
    }

    uint32_t atm_master = opts.atm_master_size;
    if (atm_master == 0u)
    {
        const char *env_ms = getenv("ATMTURB_MASTER_SIZE");
        long        parsed = (env_ms != NULL && env_ms[0] != '\0') ? strtol(env_ms, NULL, 10) : 0L;
        atm_master         = (parsed >= 64L) ? (uint32_t) parsed : 1024u;
    }

    int inc_atmturb = 0;
    if (opts.with_atmturb)
    {
        if (has_exe_in_path("milk-fpsexec-atmturb-mkwfs"))
        {
            inc_atmturb = 1;
        }
        else
        {
            fprintf(stderr, "[cacaosim] WARNING: milk-fpsexec-atmturb-mkwfs not found in PATH\n");
        }
    }

    fits_stage_t stages[PLAN_MAX_FITS];
    int          n_stages = 0;
    char         units[PLAN_MAX_UNITS][CACAOSIM_BENCH_NAME_LEN];
    int          n_units = 0;
    char         streams[PLAN_MAX_STREAMS][CACAOSIM_BENCH_NAME_LEN];
    int          n_streams = 0;

    uint64_t sig = 14695981039346656037ULL;
    sig          = fnv1a_update(sig, &opts.modular_mode, sizeof(opts.modular_mode));
    sig          = fnv1a_update(sig, &opts.flat_mode, sizeof(opts.flat_mode));
    sig          = fnv1a_update(sig, &inc_atmturb, sizeof(inc_atmturb));
    if (inc_atmturb)
    {
        sig = fnv1a_update(sig, &opts.seeing_arcsec, sizeof(opts.seeing_arcsec));
        sig = fnv1a_update(sig, &atm_master, sizeof(atm_master));
    }

    FILE *f_meta = NULL;
    FILE *f_fits = NULL;
    FILE *f_man  = NULL;
    FILE *f_bat  = NULL;
    FILE *f_wav  = NULL;

    /* ── Zero-Config Flat Single-Stage Mode ────────────────────────────────────────────────── */
    if (opts.flat_mode)
    {
        uint32_t nwf      = (opts.flat_nwf >= 8u) ? opts.flat_nwf : 128u;
        uint32_t nact     = (opts.flat_nact >= 2u) ? opts.flat_nact : 20u;
        uint32_t wfs_ncam = (opts.flat_wfs_ncam >= 4u) ? opts.flat_wfs_ncam : 64u;
        uint32_t fp_ncam  = (opts.flat_fp_ncam >= 4u) ? opts.flat_fp_ncam : 128u;
        float    eff_rate = (opts.rate_override > 0.1f) ? opts.rate_override : 2000.0f;
        float    mod_rad  = (opts.flat_mod_rad >= 0.0f) ? opts.flat_mod_rad : 3.0f;
        int32_t  backend  = (opts.backend >= 0) ? opts.backend : 0;

        sig = fnv1a_update(sig, &nwf, sizeof(nwf));
        sig = fnv1a_update(sig, &nact, sizeof(nact));
        sig = fnv1a_update(sig, &wfs_ncam, sizeof(wfs_ncam));
        sig = fnv1a_update(sig, &fp_ncam, sizeof(fp_ncam));
        sig = fnv1a_update(sig, &eff_rate, sizeof(eff_rate));
        sig = fnv1a_update(sig, &mod_rad, sizeof(mod_rad));
        sig = fnv1a_update(sig, &backend, sizeof(backend));

        f_meta = open_plan_file(plan_dir, "meta.tsv", err_buf, err_cap);
        f_fits = open_plan_file(plan_dir, "loadfits.tsv", err_buf, err_cap);
        f_man  = open_plan_file(plan_dir, "units.manifest", err_buf, err_cap);
        f_bat  = open_plan_file(plan_dir, "params.batch", err_buf, err_cap);
        f_wav  = open_plan_file(plan_dir, "waves.tsv", err_buf, err_cap);
        if (!f_meta || !f_fits || !f_man || !f_bat || !f_wav)
        {
            goto fail_close;
        }

        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "csim_pup");

        if (inc_atmturb)
        {
            emit_atmturb_plan(f_man, f_bat, f_wav, nwf, eff_rate, 0.9375f, 8.2f, 0.50f,
                              opts.seeing_arcsec, atm_master, "outarraypha", 0.0f, 0.0f, 4205.0f,
                              0, "", units, &n_units, streams, &n_streams);
        }

        if (!opts.modular_mode)
        {
            fprintf(f_man, "csim_htsim milk-fpsexec-cacaosim-htsim .bench_file= .out_dmopd=\n");

            fprintf(f_bat, "csim_htsim.in_dmcmd = aol0_dmC\n");
            fprintf(f_bat, "csim_htsim.out_wfscam = aol0_wfsim\n");
            fprintf(f_bat, "csim_htsim.enable_fpcam = ON\n");
            fprintf(f_bat, "csim_htsim.out_fpcam = csim_fpcam\n");
            fprintf(f_bat, "csim_htsim.out_fplong = csim_fplong\n");
            fprintf(f_bat, "csim_htsim.in_wf = outarraypha\n");
            fprintf(f_bat, "csim_htsim.in_pup = csim_pup\n");
            fprintf(f_bat, "csim_htsim.nwf = %u\n", nwf);
            fprintf(f_bat, "csim_htsim.nact_x = %u\n", nact);
            fprintf(f_bat, "csim_htsim.nact_y = %u\n", nact);
            fprintf(f_bat, "csim_htsim.wfs_ncam = %u\n", wfs_ncam);
            fprintf(f_bat, "csim_htsim.fp_ncam = %u\n", fp_ncam);
            fprintf(f_bat, "csim_htsim.mod_rad = %.6g\n", mod_rad);
            fprintf(f_bat, "csim_htsim.frame_rate = %.6g\n", eff_rate);
            fprintf(f_bat, "csim_htsim.backend = %d\n", backend);
            fprintf(f_bat, "csim_htsim.procinfo.triggermode = 0\n");
            fprintf(f_bat, "csim_htsim.procinfo.loopcntMax = -1\n");

            const char *fp_plus = (10.0f / eff_rate <= 0.5f) ? "+1" : "";
            fprintf(f_wav, "RUNSTART\tcsim_htsim\n");
            fprintf(f_wav, "WAIT\taol0_dmC:%ux%u aol0_wfsim:%ux%u+1 csim_fpcam:%ux%u%s"
                           " csim_fplong:%ux%u%s\n",
                    nact, nact, wfs_ncam, wfs_ncam, fp_ncam, fp_ncam, fp_plus, fp_ncam, fp_ncam,
                    fp_plus);

            add_unique_item(units, &n_units, PLAN_MAX_UNITS, "csim_htsim");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "aol0_dmC");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "aol0_wfsim");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "csim_fpcam");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "csim_fplong");
        }
        else
        {
            fprintf(f_man,
                    "csim_dmsim milk-fpsexec-cacaosim-dmsim -loops .in_beam= .out_beam="
                    " .in_flat=\n");
            fprintf(f_man, "csim_pyrwfs milk-fpsexec-cacaosim-pyrwfs .in_beam= .in_amp="
                           " .in_ncpa=\n");
            fprintf(f_man, "csim_fpcam milk-fpsexec-cacaosim-fpcam .in_beam= .in_amp="
                           " .in_ncpa=\n");

            fprintf(f_bat, "csim_dmsim.in_cmd = dm00disp\n");
            fprintf(f_bat, "csim_dmsim.out_opd = csim_dmopd\n");
            fprintf(f_bat, "csim_dmsim.nwf = %u\n", nwf);
            fprintf(f_bat, "csim_dmsim.nact_x = %u\n", nact);
            fprintf(f_bat, "csim_dmsim.nact_y = %u\n", nact);
            fprintf(f_bat, "csim_dmsim.backend = %d\n", backend);
            fprintf(f_bat, "csim_dmsim.procinfo.triggermode = 3\n");
            fprintf(f_bat, "csim_dmsim.procinfo.triggersname = dm00disp\n");
            fprintf(f_bat, "csim_dmsim.procinfo.loopcntMax = -1\n");

            fprintf(f_bat, "csim_pyrwfs.out_cam = aol0_wfsim\n");
            fprintf(f_bat, "csim_pyrwfs.in_dm = csim_dmopd\n");
            fprintf(f_bat, "csim_pyrwfs.in_wf = outarraypha\n");
            fprintf(f_bat, "csim_pyrwfs.in_pup = csim_pup\n");
            fprintf(f_bat, "csim_pyrwfs.nwf = %u\n", nwf);
            fprintf(f_bat, "csim_pyrwfs.ncam = %u\n", wfs_ncam);
            fprintf(f_bat, "csim_pyrwfs.mod_rad = %.6g\n", mod_rad);
            fprintf(f_bat, "csim_pyrwfs.frame_rate = %.6g\n", eff_rate);
            fprintf(f_bat, "csim_pyrwfs.backend = %d\n", backend);
            fprintf(f_bat, "csim_pyrwfs.procinfo.triggermode = 0\n");
            fprintf(f_bat, "csim_pyrwfs.procinfo.loopcntMax = -1\n");

            fprintf(f_bat, "csim_fpcam.out_cam = csim_fpcam\n");
            fprintf(f_bat, "csim_fpcam.out_long = csim_fplong\n");
            fprintf(f_bat, "csim_fpcam.in_dm = csim_dmopd\n");
            fprintf(f_bat, "csim_fpcam.in_wf = outarraypha\n");
            fprintf(f_bat, "csim_fpcam.in_pup = csim_pup\n");
            fprintf(f_bat, "csim_fpcam.nwf = %u\n", nwf);
            fprintf(f_bat, "csim_fpcam.ncam = %u\n", fp_ncam);
            fprintf(f_bat, "csim_fpcam.frame_rate = 200\n");
            fprintf(f_bat, "csim_fpcam.backend = %d\n", backend);
            fprintf(f_bat, "csim_fpcam.procinfo.triggermode = 0\n");
            fprintf(f_bat, "csim_fpcam.procinfo.loopcntMax = -1\n");

            fprintf(f_wav, "RUNSTART\tcsim_dmsim\n");
            fprintf(f_wav, "WAIT\tdm00disp:%ux%u csim_dmopd:%ux%u\n", nact, nact, nwf, nwf);
            fprintf(f_wav, "RUNSTART\tcsim_pyrwfs csim_fpcam\n");
            fprintf(f_wav, "WAIT\taol0_wfsim:%ux%u+1 csim_fpcam:%ux%u+1 csim_fplong:%ux%u+1\n",
                    wfs_ncam, wfs_ncam, fp_ncam, fp_ncam, fp_ncam, fp_ncam);

            add_unique_item(units, &n_units, PLAN_MAX_UNITS, "csim_dmsim");
            add_unique_item(units, &n_units, PLAN_MAX_UNITS, "csim_pyrwfs");
            add_unique_item(units, &n_units, PLAN_MAX_UNITS, "csim_fpcam");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "dm00disp");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "csim_dmopd");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "aol0_wfsim");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "csim_fpcam");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, "csim_fplong");
        }

        fprintf(f_meta, "BENCH_NAME\tflat_scao\n");
        fprintf(f_meta, "MODE\t%s\n", opts.modular_mode ? "modular" : "fused");
        fprintf(f_meta, "FLAT\t1\n");
        fprintf(f_meta, "NWF\t%u\n", nwf);
        fprintf(f_meta, "EFF_RATE\t%.6g\n", eff_rate);
        fprintf(f_meta, "PUP_DIAM\t0.9375\n");
        fprintf(f_meta, "COBS\t0.30\n");
        fprintf(f_meta, "NSPIDER\t4\n");
        fprintf(f_meta, "SPIDER_W\t0.01\n");
        fprintf(f_meta, "SPIDER_DEG\t45.0\n");
        fprintf(f_meta, "SPIDER_OFFSET\t0.0\n");
        fprintf(f_meta, "MKPUPIL_OUT\tcsim_pup\n");
        fprintf(f_meta, "IN_PUP\tcsim_pup\n");
        fprintf(f_meta, "IN_WF\toutarraypha\n");
        fprintf(f_meta, "SIGNATURE\t%016llx\n", (unsigned long long) sig);
    }
    /* ── Declarative Multi-Stage `.bench` DAG Mode ─────────────────────────────────────────── */
    else
    {
        if (bench_file == NULL || bench_file[0] == '\0')
        {
            plan_err(err_buf, err_cap, "Missing .bench file path");
            return -1;
        }

        cacaosim_bench_cfg_t cfg;
        if (cacaosim_bench_load_file(bench_file, &cfg, err_buf, err_cap) != 0)
        {
            return -1;
        }

        if (opts.rate_override > 0.1f)
        {
            cfg.master_hz = opts.rate_override;
        }
        if (opts.oversample_override >= 1u)
        {
            cfg.oversample = opts.oversample_override;
        }
        if (opts.backend >= 0)
        {
            cfg.backend = opts.backend;
        }

        char bench_dir_buf[PATH_MAX];
        snprintf(bench_dir_buf, sizeof(bench_dir_buf), "%s", bench_file);
        const char *bench_dir = dirname(bench_dir_buf);

        int pup_is_fits = stage_fits_field(cfg.in_pup, sizeof(cfg.in_pup), bench_dir, "bench",
                                           "pup", stages, &n_stages, streams, &n_streams, &sig,
                                           err_buf, err_cap);
        if (pup_is_fits < 0)
        {
            return -1;
        }
        if (stage_fits_field(cfg.in_ncpa, sizeof(cfg.in_ncpa), bench_dir, "bench", "ncpa", stages,
                             &n_stages, streams, &n_streams, &sig, err_buf, err_cap) < 0)
        {
            return -1;
        }

        for (uint32_t i = 0; i < cfg.n_nodes; i++)
        {
            cacaosim_node_cfg_t *nd = &cfg.nodes[i];
            if (stage_fits_field(nd->in_pup_amp, sizeof(nd->in_pup_amp), bench_dir, nd->id, "pamp",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_pup_pha, sizeof(nd->in_pup_pha), bench_dir, nd->id, "ppha",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_fpm_amp, sizeof(nd->in_fpm_amp), bench_dir, nd->id, "famp",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_fpm_pha, sizeof(nd->in_fpm_pha), bench_dir, nd->id, "fpha",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_lyot, sizeof(nd->in_lyot), bench_dir, nd->id, "lamp",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_lyot_pha, sizeof(nd->in_lyot_pha), bench_dir, nd->id,
                                 "lpha", stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_flat, sizeof(nd->in_flat), bench_dir, nd->id, "flat",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_ncpa, sizeof(nd->in_ncpa), bench_dir, nd->id, "ncpa",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->ncpa_refl, sizeof(nd->ncpa_refl), bench_dir, nd->id, "rncp",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->ncpa_trans, sizeof(nd->ncpa_trans), bench_dir, nd->id, "tncp",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->pup_refl, sizeof(nd->pup_refl), bench_dir, nd->id, "rpup",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->pup_trans, sizeof(nd->pup_trans), bench_dir, nd->id, "tpup",
                                 stages, &n_stages, streams, &n_streams, &sig, err_buf,
                                 err_cap) < 0 ||
                stage_fits_field(nd->in_wfe, sizeof(nd->in_wfe), bench_dir, nd->id, "wfe", stages,
                                 &n_stages, streams, &n_streams, &sig, err_buf, err_cap) < 0)
            {
                return -1;
            }
        }

        cacaosim_bench_ctx_t ctx;
        if (cacaosim_bench_analyze(&ctx, &cfg, err_buf, err_cap) != 0)
        {
            return -1;
        }

        int fmt_len = cacaosim_bench_format_str(&cfg, NULL, 0);
        if (fmt_len > 0)
        {
            char *fmt_buf = (char *) malloc((size_t) fmt_len + 1u);
            if (fmt_buf != NULL)
            {
                cacaosim_bench_format_str(&cfg, fmt_buf, (size_t) fmt_len + 1u);
                sig = fnv1a_update(sig, fmt_buf, (size_t) fmt_len);
                free(fmt_buf);
            }
        }

        if (opts.staged_bench[0] != '\0')
        {
            if (cacaosim_bench_save_file(opts.staged_bench, &cfg, err_buf, err_cap) != 0)
            {
                return -1;
            }
        }

        f_meta = open_plan_file(plan_dir, "meta.tsv", err_buf, err_cap);
        f_fits = open_plan_file(plan_dir, "loadfits.tsv", err_buf, err_cap);
        f_man  = open_plan_file(plan_dir, "units.manifest", err_buf, err_cap);
        f_bat  = open_plan_file(plan_dir, "params.batch", err_buf, err_cap);
        f_wav  = open_plan_file(plan_dir, "waves.tsv", err_buf, err_cap);
        if (!f_meta || !f_fits || !f_man || !f_bat || !f_wav)
        {
            goto fail_close;
        }

        for (int i = 0; i < n_stages; i++)
        {
            fprintf(f_fits, "%s\t%s\n", stages[i].shm_stem, stages[i].abs_path);
        }

        const char *mkpupil_out = (!pup_is_fits && cfg.in_pup[0] != '\0') ? cfg.in_pup : "";
        if (cfg.in_pup[0] != '\0')
        {
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, cfg.in_pup);
        }

        float base_rate = (cfg.master_hz > 0.1f) ? cfg.master_hz : 2000.0f;
        float eff_rate  = (ctx.eff_master_hz > 0.1f) ? ctx.eff_master_hz : base_rate;
        float dt_s      = 1.0f / eff_rate;
        float atm_rate  = (!opts.modular_mode && cfg.atm_interp) ? base_rate : eff_rate;

        if (inc_atmturb)
        {
            emit_atmturb_plan(f_man, f_bat, f_wav, cfg.nwf, atm_rate, cfg.pup_diam, cfg.tel_diam_m,
                              cfg.ref_lambda, opts.seeing_arcsec, atm_master, cfg.in_wf,
                              cfg.atm_disp.zenith_deg, cfg.atm_disp.disp_pa_deg, cfg.site_alt_m,
                              cfg.chrom_turb, cfg.chrom_lambdas, units, &n_units, streams,
                              &n_streams);
        }

        if (!opts.modular_mode)
        {
            const char *b_arg = opts.active_bench_arg[0] ? opts.active_bench_arg
                                                         : "csim_active.bench";
            fprintf(f_man,
                    "csim_htsim milk-fpsexec-cacaosim-htsim .bench_file= .out_dmopd="
                    " .out_fpcam= .out_fplong= .in_wf= .in_amp= .in_pup=\n");

            fprintf(f_bat, "csim_htsim.bench_file = %s\n", b_arg);
            fprintf(f_bat, "csim_htsim.frame_rate = %.6g\n", eff_rate);
            fprintf(f_bat, "csim_htsim.turb_gain = %.6g\n", cfg.turb_gain);
            fprintf(f_bat, "csim_htsim.vib_enable = %s\n", cfg.perturb.vib_enable ? "ON" : "OFF");
            fprintf(f_bat, "csim_htsim.vib_rms_nm = %.6g\n", cfg.perturb.vib_rms_nm);
            fprintf(f_bat, "csim_htsim.petal_enable = %s\n",
                    cfg.perturb.petal_enable ? "ON" : "OFF");
            fprintf(f_bat, "csim_htsim.petal_rms_nm = %.6g\n", cfg.perturb.petal_rms_nm);
            fprintf(f_bat, "csim_htsim.petal_tau_s = %.6g\n", cfg.perturb.petal_tau_s);
            fprintf(f_bat, "csim_htsim.atm_disp_enable = %s\n",
                    cfg.atm_disp.enable ? "ON" : "OFF");
            fprintf(f_bat, "csim_htsim.zenith_deg = %.6g\n", cfg.atm_disp.zenith_deg);
            fprintf(f_bat, "csim_htsim.disp_pa_deg = %.6g\n", cfg.atm_disp.disp_pa_deg);
            fprintf(f_bat, "csim_htsim.adc_mode = -1\n");
            fprintf(f_bat, "csim_htsim.adc_opt_band = -1\n");
            fprintf(f_bat, "csim_htsim.backend = %d\n", cfg.backend);
            fprintf(f_bat, "csim_htsim.procinfo.triggermode = 0\n");
            fprintf(f_bat, "csim_htsim.procinfo.loopcntMax = -1\n");

            fprintf(f_wav, "RUNSTART\tcsim_htsim\n");
            fprintf(f_wav, "WAIT\t");
            int first_w = 1;
            for (uint32_t i = 0; i < ctx.n_nodes; i++)
            {
                const cacaosim_bench_node_t *nd = &ctx.nodes[i];
                if (nd->cfg.kind == CACAOSIM_NODE_DM)
                {
                    if (nd->cfg.in_cmd[0] != '\0')
                    {
                        uint32_t nx = (nd->cfg.dm.dm_mode == 1) ? 2u : nd->cfg.dm.nact_x;
                        uint32_t ny = (nd->cfg.dm.dm_mode == 1) ? 1u : nd->cfg.dm.nact_y;
                        fprintf(f_wav, "%s%s:%ux%u", first_w ? "" : " ", nd->cfg.in_cmd, nx, ny);
                        first_w = 0;
                        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.in_cmd);
                    }
                    if (nd->cfg.out_opd[0] != '\0')
                    {
                        fprintf(f_wav, "%s%s:%ux%u+1", first_w ? "" : " ", nd->cfg.out_opd,
                                cfg.nwf, cfg.nwf);
                        first_w = 0;
                        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_opd);
                    }
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_WFE && nd->cfg.in_wfe[0] != '\0')
                {
                    fprintf(f_wav, "%s%s:%ux%u", first_w ? "" : " ", nd->cfg.in_wfe, cfg.nwf,
                            cfg.nwf);
                    first_w = 0;
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.in_wfe);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_ADC)
                {
                    if (nd->cfg.in_cmd[0] != '\0')
                    {
                        fprintf(f_wav, "%s%s:2x1", first_w ? "" : " ", nd->cfg.in_cmd);
                        first_w = 0;
                        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.in_cmd);
                    }
                    if (nd->cfg.out_state[0] != '\0')
                    {
                        fprintf(f_wav, "%s%s:%ux1+1", first_w ? "" : " ", nd->cfg.out_state,
                                (unsigned) CACAOSIM_ADC_STATE_LEN);
                        first_w = 0;
                        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_state);
                    }
                }
                else if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS &&
                         nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
                {
                    const char *plus = ((float) nd->eff_navg * dt_s <= 0.5f) ? "+1" : "";
                    if (nd->cfg.out_cam[0] != '\0')
                    {
                        fprintf(f_wav, "%s%s:%ux%u%s", first_w ? "" : " ", nd->cfg.out_cam,
                                nd->ncam, nd->ncam, plus);
                        first_w = 0;
                        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_cam);
                    }
                    if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && nd->cfg.out_long[0] != '\0')
                    {
                        fprintf(f_wav, "%s%s:%ux%u%s", first_w ? "" : " ", nd->cfg.out_long,
                                nd->ncam, nd->ncam, plus);
                        first_w = 0;
                        add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_long);
                    }
                }
            }
            fprintf(f_wav, "\n");
            add_unique_item(units, &n_units, PLAN_MAX_UNITS, "csim_htsim");
        }
        else
        {
            /* Modular multi-process DAG: compute beam depths & potential polychromatic promotion */
            int beam_depth[CACAOSIM_BENCH_MAX_BEAMS] = { 0 };
            int beam_poly[CACAOSIM_BENCH_MAX_BEAMS]  = { 0 };
            int node_depth[CACAOSIM_BENCH_MAX_NODES] = { 0 };
            int max_depth                            = 0;
            if (cfg.chrom_turb && cfg.chrom_lambdas[0] != '\0')
            {
                beam_poly[0] = 1;
            }

            for (uint32_t i = 0; i < ctx.n_nodes; i++)
            {
                const cacaosim_bench_node_t *nd = &ctx.nodes[i];
                int                          d  = beam_depth[nd->in_idx] + 1;
                int                          in_p = beam_poly[nd->in_idx];
                node_depth[i]                     = d;
                if (d > max_depth)
                {
                    max_depth = d;
                }
                if (nd->cfg.kind == CACAOSIM_NODE_CORO)
                {
                    int c_poly = in_p || (nd->cfg.coro.nlambda > 1) ||
                                 (nd->cfg.coro.nlambda <= 0 && nd->cfg.coro.bw_frac > 1e-6f) ||
                                 (nd->cfg.in_pup_amp[0] != '\0') ||
                                 (nd->cfg.in_pup_pha[0] != '\0') ||
                                 (nd->cfg.in_fpm_amp[0] != '\0') ||
                                 (nd->cfg.in_fpm_pha[0] != '\0') ||
                                 (nd->cfg.in_lyot[0] != '\0') ||
                                 (nd->cfg.in_lyot_pha[0] != '\0');
                    if (nd->out_idx >= 0)
                    {
                        beam_depth[nd->out_idx] = d;
                        beam_poly[nd->out_idx]  = c_poly;
                    }
                    if (nd->lowfs_idx >= 0)
                    {
                        beam_depth[nd->lowfs_idx] = d;
                        beam_poly[nd->lowfs_idx]  = c_poly;
                    }
                }
                else
                {
                    if (nd->out_idx >= 0)
                    {
                        beam_depth[nd->out_idx] = d;
                        beam_poly[nd->out_idx]  = in_p;
                    }
                    if (nd->refl_idx >= 0)
                    {
                        beam_depth[nd->refl_idx] = d;
                        beam_poly[nd->refl_idx]  = in_p;
                    }
                    if (nd->trans_idx >= 0)
                    {
                        beam_depth[nd->trans_idx] = d;
                        beam_poly[nd->trans_idx]  = in_p;
                    }
                }
            }

            /* Root entrance beam unit: csim_mkbeam */
            const char *root_beam = cfg.out_beam[0] ? cfg.out_beam : "b_pup";
            fprintf(f_man,
                    "csim_mkbeam milk-fpsexec-cacaosim-mkbeam .in_wf= .in_amp= .in_pup="
                    " .in_ncpa=\n");
            fprintf(f_bat, "csim_mkbeam.out_beam = %s\n", root_beam);
            emit_str_if_set(f_bat, "csim_mkbeam", ".in_wf", cfg.in_wf);
            emit_str_if_set(f_bat, "csim_mkbeam", ".in_amp", cfg.in_amp);
            emit_str_if_set(f_bat, "csim_mkbeam", ".in_pup", cfg.in_pup);
            emit_str_if_set(f_bat, "csim_mkbeam", ".in_ncpa", cfg.in_ncpa);
            fprintf(f_bat, "csim_mkbeam.nwf = %u\n", cfg.nwf);
            fprintf(f_bat, "csim_mkbeam.pup_diam = %.8g\n", cfg.pup_diam);
            fprintf(f_bat, "csim_mkbeam.obs_frac = %.8g\n", cfg.cobs);
            fprintf(f_bat, "csim_mkbeam.n_spiders = %d\n", cfg.nspider);
            fprintf(f_bat, "csim_mkbeam.spider_w = %.8g\n", cfg.spider_w);
            fprintf(f_bat, "csim_mkbeam.spider_deg = %.8g\n", cfg.spider_deg);
            fprintf(f_bat, "csim_mkbeam.spider_offset = %.8g\n", cfg.spider_offset);
            fprintf(f_bat, "csim_mkbeam.tel_diam_m = %.8g\n", cfg.tel_diam_m);
            fprintf(f_bat, "csim_mkbeam.star_mag = %.8g\n", cfg.star_mag);
            fprintf(f_bat, "csim_mkbeam.star_teff = %.8g\n", cfg.star_teff);
            fprintf(f_bat, "csim_mkbeam.tel_tput = %.8g\n", cfg.tput_tel);
            fprintf(f_bat, "csim_mkbeam.wf_unit = %d\n", cfg.wf_unit);
            fprintf(f_bat, "csim_mkbeam.ref_lambda = %.8g\n", cfg.ref_lambda);
            fprintf(f_bat, "csim_mkbeam.turb_gain = %.8g\n", cfg.turb_gain);
            fprintf(f_bat, "csim_mkbeam.vib_enable = %s\n", cfg.perturb.vib_enable ? "ON" : "OFF");
            fprintf(f_bat, "csim_mkbeam.vib_rms_nm = %.6g\n", cfg.perturb.vib_rms_nm);
            fprintf(f_bat, "csim_mkbeam.petal_enable = %s\n",
                    cfg.perturb.petal_enable ? "ON" : "OFF");
            fprintf(f_bat, "csim_mkbeam.petal_rms_nm = %.6g\n", cfg.perturb.petal_rms_nm);
            fprintf(f_bat, "csim_mkbeam.petal_tau_s = %.6g\n", cfg.perturb.petal_tau_s);
            fprintf(f_bat, "csim_mkbeam.atm_disp_enable = %s\n",
                    cfg.atm_disp.enable ? "ON" : "OFF");
            fprintf(f_bat, "csim_mkbeam.zenith_deg = %.8g\n", cfg.atm_disp.zenith_deg);
            fprintf(f_bat, "csim_mkbeam.disp_pa_deg = %.8g\n", cfg.atm_disp.disp_pa_deg);
            fprintf(f_bat, "csim_mkbeam.atm_pressure_hpa = %.8g\n", cfg.atm_disp.pressure_hpa);
            fprintf(f_bat, "csim_mkbeam.atm_temp_c = %.8g\n", cfg.atm_disp.temp_c);
            fprintf(f_bat, "csim_mkbeam.atm_rh_pct = %.8g\n", cfg.atm_disp.rh_pct);
            fprintf(f_bat, "csim_mkbeam.atm_disp_ref_um = %.8g\n", cfg.atm_disp.ref_lam_um);
            fprintf(f_bat, "csim_mkbeam.frame_rate = %.8g\n", eff_rate);
            fprintf(f_bat, "csim_mkbeam.procinfo.triggermode = 0\n");
            fprintf(f_bat, "csim_mkbeam.procinfo.loopcntMax = -1\n");

            fprintf(f_wav, "RUNSTART\tcsim_mkbeam\n");
            if (beam_poly[0])
            {
                fprintf(f_wav, "WAIT\t%s+1\n", root_beam);
            }
            else
            {
                fprintf(f_wav, "WAIT\t%s:%ux%ux2+1\n", root_beam, cfg.nwf, cfg.nwf);
            }
            add_unique_item(units, &n_units, PLAN_MAX_UNITS, "csim_mkbeam");
            add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, root_beam);

            /* Emit manifest + batch parameters for every topologically sorted optical node */
            for (uint32_t i = 0; i < ctx.n_nodes; i++)
            {
                const cacaosim_bench_node_t *nd = &ctx.nodes[i];
                char                         fps[CACAOSIM_BENCH_NAME_LEN];
                snprintf(fps, sizeof(fps), "csim_%.56s", nd->cfg.id);
                add_unique_item(units, &n_units, PLAN_MAX_UNITS, fps);

                const char *inb = ctx.beams[nd->in_idx].name;

                if (nd->cfg.kind == CACAOSIM_NODE_DM)
                {
                    char cmd[CACAOSIM_BENCH_NAME_LEN];
                    char opd[CACAOSIM_BENCH_NAME_LEN];
                    snprintf(cmd, sizeof(cmd), "%s",
                             nd->cfg.in_cmd[0] ? nd->cfg.in_cmd : nd->cfg.id);
                    if (!nd->cfg.in_cmd[0])
                    {
                        snprintf(cmd, sizeof(cmd), "%.56s_cmd", nd->cfg.id);
                    }
                    if (nd->cfg.out_opd[0] != '\0')
                    {
                        snprintf(opd, sizeof(opd), "%s", nd->cfg.out_opd);
                    }
                    else
                    {
                        snprintf(opd, sizeof(opd), "csim_opd_%.52s", nd->cfg.id);
                    }

                    uint32_t nx = (nd->cfg.dm.dm_mode == 1) ? 2u : nd->cfg.dm.nact_x;
                    uint32_t ny = (nd->cfg.dm.dm_mode == 1) ? 1u : nd->cfg.dm.nact_y;

                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-dmsim .in_beam= .out_beam= .in_flat=\n",
                            fps);
                    fprintf(f_bat, "%s.in_cmd = %s\n", fps, cmd);
                    fprintf(f_bat, "%s.out_opd = %s\n", fps, opd);
                    emit_str_if_set(f_bat, fps, ".in_beam", inb);
                    emit_str_if_set(f_bat, fps, ".out_beam", nd->cfg.out_beam);
                    emit_str_if_set(f_bat, fps, ".in_flat", nd->cfg.in_flat);
                    fprintf(f_bat, "%s.dm_mode = %d\n", fps, nd->cfg.dm.dm_mode);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.nact_x = %u\n", fps, nx);
                    fprintf(f_bat, "%s.nact_y = %u\n", fps, ny);
                    fprintf(f_bat, "%s.dm_diam = %.8g\n", fps, nd->cfg.dm.dm_diam);
                    fprintf(f_bat, "%s.if_coupling = %.8g\n", fps, nd->cfg.dm.if_coupling);
                    fprintf(f_bat, "%s.refl_scale = %.8g\n", fps, nd->cfg.dm.refl_scale);
                    fprintf(f_bat, "%s.tput = %.8g\n", fps, nd->cfg.tput);
                    fprintf(f_bat, "%s.max_stroke = %.8g\n", fps, nd->cfg.dm.max_stroke);
                    fprintf(f_bat, "%s.dac_bits = %d\n", fps, nd->cfg.dm.dac_bits);
                    fprintf(f_bat, "%s.nonlin_quad = %.8g\n", fps, nd->cfg.dm.nonlin_quad);
                    fprintf(f_bat, "%s.dyn_mode = %d\n", fps, nd->cfg.dm.dyn_mode);
                    fprintf(f_bat, "%s.dt_sec = %.7f\n", fps, dt_s);
                    fprintf(f_bat, "%s.tau_sec = %.7f\n", fps, nd->cfg.dm.tau_sec);
                    fprintf(f_bat, "%s.f0_hz = %.8g\n", fps, nd->cfg.dm.f0_hz);
                    fprintf(f_bat, "%s.zeta = %.8g\n", fps, nd->cfg.dm.zeta);
                    if (nd->cfg.dm.dyn_mode == 3)
                    {
                        fprintf(f_bat, "%s.act_pitch_mm = %.8g\n", fps, nd->cfg.dm.act_pitch_mm);
                        fprintf(f_bat, "%s.wave_c = %.8g\n", fps, nd->cfg.dm.wave_c_ms);
                        fprintf(f_bat, "%s.bend_beta = %.8g\n", fps, nd->cfg.dm.bend_beta_m2s);
                        fprintf(f_bat, "%s.damp_g0 = %.8g\n", fps, nd->cfg.dm.damp_g0);
                        fprintf(f_bat, "%s.damp_g2 = %.8g\n", fps, nd->cfg.dm.damp_g2);
                        fprintf(f_bat, "%s.plate_drive = %d\n", fps, nd->cfg.dm.plate_drive);
                        fprintf(f_bat, "%s.rim_bc = %d\n", fps, nd->cfg.dm.rim_bc);
                    }
                    fprintf(f_bat, "%s.delay_frames = %.8g\n", fps, nd->eff_delay_ticks);
                    emit_reg_params(f_bat, fps, &nd->cfg.reg);
                    fprintf(f_bat, "%s.backend = %d\n", fps, cfg.backend);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, cmd);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, opd);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_beam);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
                {
                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-beamsplit .out_refl= .out_trans="
                            " .ncpa_refl= .ncpa_trans= .pup_refl= .pup_trans=\n",
                            fps);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    emit_str_if_set(f_bat, fps, ".out_refl", nd->cfg.out_refl);
                    emit_str_if_set(f_bat, fps, ".out_trans", nd->cfg.out_trans);
                    emit_str_if_set(f_bat, fps, ".ncpa_refl", nd->cfg.ncpa_refl);
                    emit_str_if_set(f_bat, fps, ".ncpa_trans", nd->cfg.ncpa_trans);
                    emit_str_if_set(f_bat, fps, ".pup_refl", nd->cfg.pup_refl);
                    emit_str_if_set(f_bat, fps, ".pup_trans", nd->cfg.pup_trans);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.split_mode = %d\n", fps, nd->cfg.split.mode);
                    fprintf(f_bat, "%s.r_gray = %.8g\n", fps, nd->cfg.split.r_gray);
                    fprintf(f_bat, "%s.t_gray = %.8g\n", fps, nd->cfg.split.t_gray);
                    fprintf(f_bat, "%s.lambda_cut = %.8g\n", fps, nd->cfg.split.lambda_cut);
                    fprintf(f_bat, "%s.band_width = %.8g\n", fps, nd->cfg.split.band_width);
                    fprintf(f_bat, "%s.roll_um = %.8g\n", fps, nd->cfg.split.roll_um);
                    fprintf(f_bat, "%s.eff_refl = %.8g\n", fps, nd->cfg.split.eff_refl);
                    fprintf(f_bat, "%s.eff_trans = %.8g\n", fps, nd->cfg.split.eff_trans);
                    fprintf(f_bat, "%s.refl_dx = %.8g\n", fps, nd->cfg.reg_refl.shift_x);
                    fprintf(f_bat, "%s.refl_dy = %.8g\n", fps, nd->cfg.reg_refl.shift_y);
                    fprintf(f_bat, "%s.refl_rot = %.8g\n", fps, nd->cfg.reg_refl.rot_deg);
                    fprintf(f_bat, "%s.refl_scale = %.8g\n", fps,
                            (nd->cfg.reg_refl.scale > 0.0f) ? nd->cfg.reg_refl.scale : 1.0f);
                    fprintf(f_bat, "%s.trans_dx = %.8g\n", fps, nd->cfg.reg_trans.shift_x);
                    fprintf(f_bat, "%s.trans_dy = %.8g\n", fps, nd->cfg.reg_trans.shift_y);
                    fprintf(f_bat, "%s.trans_rot = %.8g\n", fps, nd->cfg.reg_trans.rot_deg);
                    fprintf(f_bat, "%s.trans_scale = %.8g\n", fps,
                            (nd->cfg.reg_trans.scale > 0.0f) ? nd->cfg.reg_trans.scale : 1.0f);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_refl);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_trans);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
                {
                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-coro .out_lowfs= .out_pre_lyot="
                            " .out_pre_lyot_long= .in_pup_amp= .in_pup_pha= .in_fpm_amp="
                            " .in_fpm_pha= .in_lyot= .in_lyot_pha=\n",
                            fps);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    fprintf(f_bat, "%s.out_beam = %s\n", fps, nd->cfg.out_beam);
                    emit_str_if_set(f_bat, fps, ".out_lowfs", nd->cfg.out_lowfs);
                    emit_str_if_set(f_bat, fps, ".out_pre_lyot", nd->cfg.out_pre_lyot);
                    emit_str_if_set(f_bat, fps, ".out_pre_lyot_long", nd->cfg.out_pre_lyot_long);
                    emit_str_if_set(f_bat, fps, ".in_pup_amp", nd->cfg.in_pup_amp);
                    emit_str_if_set(f_bat, fps, ".in_pup_pha", nd->cfg.in_pup_pha);
                    emit_str_if_set(f_bat, fps, ".in_fpm_amp", nd->cfg.in_fpm_amp);
                    emit_str_if_set(f_bat, fps, ".in_fpm_pha", nd->cfg.in_fpm_pha);
                    emit_str_if_set(f_bat, fps, ".in_lyot", nd->cfg.in_lyot);
                    emit_str_if_set(f_bat, fps, ".in_lyot_pha", nd->cfg.in_lyot_pha);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.coro_mode = %d\n", fps, (int) nd->cfg.coro.mode);
                    fprintf(f_bat, "%s.n_mft = %u\n", fps, nd->cfg.coro.n_mft);
                    fprintf(f_bat, "%s.pad_factor = %u\n", fps, nd->cfg.coro.pad_factor);
                    fprintf(f_bat, "%s.pup_diam = %.8g\n", fps, cfg.pup_diam);
                    fprintf(f_bat, "%s.lambda_um = %.8g\n", fps, nd->cfg.coro.lambda_um);
                    fprintf(f_bat, "%s.bw_frac = %.8g\n", fps, nd->cfg.coro.bw_frac);
                    fprintf(f_bat, "%s.nlambda = %d\n", fps, nd->cfg.coro.nlambda);
                    fprintf(f_bat, "%s.fpm_rad_ld = %.8g\n", fps, nd->cfg.coro.fpm_rad_ld);
                    fprintf(f_bat, "%s.fpm_amp = %.8g\n", fps, nd->cfg.coro.fpm_amp);
                    fprintf(f_bat, "%s.fpm_phase = %.8g\n", fps, nd->cfg.coro.fpm_phase_rad);
                    fprintf(f_bat, "%s.fpm_pixscale = %.8g\n", fps, nd->cfg.coro.fpm_pixscale_ld);
                    fprintf(f_bat, "%s.pha_unit = %d\n", fps, nd->cfg.coro.mask_pha_unit);
                    fprintf(f_bat, "%s.vortex_charge = %d\n", fps, nd->cfg.coro.vortex_charge);
                    fprintf(f_bat, "%s.lowfs_mode = %d\n", fps, nd->cfg.coro.lowfs_mode);
                    fprintf(f_bat, "%s.lowfs_pup_diam = %.8g\n", fps, nd->cfg.coro.lowfs_pup_diam);
                    fprintf(f_bat, "%s.lowfs_refl = %.8g\n", fps, nd->cfg.coro.lowfs_refl);
                    fprintf(f_bat, "%s.tput = %.8g\n", fps, nd->cfg.tput);
                    fprintf(f_bat, "%s.lyot_outer = %.8g\n", fps, nd->cfg.coro.lyot_outer);
                    fprintf(f_bat, "%s.lyot_obs = %.8g\n", fps, nd->cfg.coro.lyot_obs);
                    fprintf(f_bat, "%s.lyot_nspider = %d\n", fps, nd->cfg.coro.lyot_nspider);
                    fprintf(f_bat, "%s.lyot_spider_w = %.8g\n", fps, nd->cfg.coro.lyot_spider_w);
                    fprintf(f_bat, "%s.lyot_spider_deg = %.8g\n", fps,
                            nd->cfg.coro.lyot_spider_deg);
                    fprintf(f_bat, "%s.lyot_spider_offset = %.8g\n", fps,
                            nd->cfg.coro.lyot_spider_offset);
                    fprintf(f_bat, "%s.backend = %d\n", fps, cfg.backend);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_beam);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_lowfs);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_pre_lyot);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS,
                                    nd->cfg.out_pre_lyot_long);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
                {
                    char wfe_s[CACAOSIM_BENCH_NAME_LEN];
                    if (nd->cfg.in_wfe[0] != '\0')
                    {
                        snprintf(wfe_s, sizeof(wfe_s), "%s", nd->cfg.in_wfe);
                    }
                    else
                    {
                        snprintf(wfe_s, sizeof(wfe_s), "%.56s_wfe", nd->cfg.id);
                    }

                    fprintf(f_man, "%s milk-fpsexec-cacaosim-wfe .in_pup=\n", fps);
                    fprintf(f_bat, "%s.init_only = OFF\n", fps);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    fprintf(f_bat, "%s.out_beam = %s\n", fps, nd->cfg.out_beam);
                    fprintf(f_bat, "%s.in_wfe = %s\n", fps, wfe_s);
                    emit_str_if_set(f_bat, fps, ".in_pup", cfg.in_pup);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.pup_diam = %.8g\n", fps, cfg.pup_diam);
                    fprintf(f_bat, "%s.rms_nm = %.8g\n", fps, nd->cfg.wfe_rms_nm);
                    fprintf(f_bat, "%s.defocus_nm = %.8g\n", fps, nd->cfg.wfe_defocus_nm);
                    fprintf(f_bat, "%s.psd_alpha = %.8g\n", fps, nd->cfg.wfe_alpha);
                    fprintf(f_bat, "%s.wfe_seed = %llu\n", fps,
                            (unsigned long long) nd->cfg.wfe_seed);
                    fprintf(f_bat, "%s.wfe_gain = %.8g\n", fps, nd->cfg.wfe_gain);
                    fprintf(f_bat, "%s.tput = %.8g\n", fps, nd->cfg.tput);
                    emit_reg_params(f_bat, fps, &nd->cfg.reg);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, wfe_s);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_beam);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_ADC)
                {
                    fprintf(f_man, "%s milk-fpsexec-cacaosim-adc .in_cmd= .out_state=\n", fps);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    fprintf(f_bat, "%s.out_beam = %s\n", fps, nd->cfg.out_beam);
                    emit_str_if_set(f_bat, fps, ".in_cmd", nd->cfg.in_cmd);
                    emit_str_if_set(f_bat, fps, ".out_state", nd->cfg.out_state);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.preset = %s\n", fps,
                            cacaosim_adc_preset_name(nd->cfg.adc.preset));
                    fprintf(f_bat, "%s.mode = %s\n", fps,
                            cacaosim_adc_mode_name(nd->cfg.adc.mode));
                    fprintf(f_bat, "%s.opt_band = %s\n", fps,
                            cacaosim_adc_opt_band_name(nd->cfg.adc.opt_band));
                    fprintf(f_bat, "%s.opt_lam1_um = %.8g\n", fps, nd->cfg.adc.opt_lam1_um);
                    fprintf(f_bat, "%s.opt_lam2_um = %.8g\n", fps, nd->cfg.adc.opt_lam2_um);
                    fprintf(f_bat, "%s.angle1_deg = %.8g\n", fps, nd->cfg.adc.angle1_deg);
                    fprintf(f_bat, "%s.angle2_deg = %.8g\n", fps, nd->cfg.adc.angle2_deg);
                    fprintf(f_bat, "%s.manual_use_psi = %d\n", fps, nd->cfg.adc.manual_use_psi);
                    fprintf(f_bat, "%s.manual_psi_deg = %.8g\n", fps, nd->cfg.adc.manual_psi_deg);
                    emit_str_if_set(f_bat, fps, ".glass1", nd->cfg.adc.glass1);
                    emit_str_if_set(f_bat, fps, ".glass2", nd->cfg.adc.glass2);
                    fprintf(f_bat, "%s.alpha1_deg = %.8g\n", fps, nd->cfg.adc.alpha1_deg);
                    fprintf(f_bat, "%s.alpha2_deg = %.8g\n", fps, nd->cfg.adc.alpha2_deg);
                    fprintf(f_bat, "%s.beam_mag = %.8g\n", fps, nd->cfg.adc.beam_mag);
                    fprintf(f_bat, "%s.zenith_deg = %.8g\n", fps, cfg.atm_disp.zenith_deg);
                    fprintf(f_bat, "%s.disp_pa_deg = %.8g\n", fps, cfg.atm_disp.disp_pa_deg);
                    fprintf(f_bat, "%s.atm_pressure_hpa = %.8g\n", fps, cfg.atm_disp.pressure_hpa);
                    fprintf(f_bat, "%s.atm_temp_c = %.8g\n", fps, cfg.atm_disp.temp_c);
                    fprintf(f_bat, "%s.atm_rh_pct = %.8g\n", fps, cfg.atm_disp.rh_pct);
                    fprintf(f_bat, "%s.atm_disp_ref_um = %.8g\n", fps, cfg.atm_disp.ref_lam_um);
                    fprintf(f_bat, "%s.lut_scale_err = %.8g\n", fps, nd->cfg.adc.lut_scale_err);
                    fprintf(f_bat, "%s.lut_pa_err_deg = %.8g\n", fps, nd->cfg.adc.lut_pa_err_deg);
                    fprintf(f_bat, "%s.lut_p_err_hpa = %.8g\n", fps, nd->cfg.adc.lut_p_err_hpa);
                    fprintf(f_bat, "%s.lut_t_err_c = %.8g\n", fps, nd->cfg.adc.lut_t_err_c);
                    fprintf(f_bat, "%s.lut_use_nominal = %d\n", fps, nd->cfg.adc.lut_use_nominal);
                    fprintf(f_bat, "%s.ideal_mech = %d\n", fps, nd->cfg.adc.ideal_mech);
                    fprintf(f_bat, "%s.enc_step_deg = %.8g\n", fps, nd->cfg.adc.enc_step_deg);
                    fprintf(f_bat, "%s.max_slew_dps = %.8g\n", fps, nd->cfg.adc.max_slew_dps);
                    fprintf(f_bat, "%s.backlash_deg = %.8g\n", fps, nd->cfg.adc.backlash_deg);
                    fprintf(f_bat, "%s.prism_offset_deg = %.8g\n", fps,
                            nd->cfg.adc.prism_offset_deg);
                    fprintf(f_bat, "%s.unit_rot_err_deg = %.8g\n", fps,
                            nd->cfg.adc.unit_rot_err_deg);
                    fprintf(f_bat, "%s.tput_enable = %d\n", fps, nd->cfg.adc.tput_enable);
                    fprintf(f_bat, "%s.tput_base = %.8g\n", fps, nd->cfg.adc.tput_base);
                    fprintf(f_bat, "%s.dt_sec = %.7f\n", fps, dt_s);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_beam);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.in_cmd);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_state);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
                {
                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-pyrwfs .in_dm= .in_wf= .in_amp= .in_pup="
                            " .in_ncpa=\n",
                            fps);
                    fprintf(f_bat, "%s.out_cam = %s\n", fps, nd->cfg.out_cam);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    emit_str_if_set(f_bat, fps, ".in_pup", cfg.in_pup);
                    emit_str_if_set(f_bat, fps, ".in_ncpa", nd->cfg.in_ncpa);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.pad_factor = %u\n", fps, nd->cfg.pyrwfs.pad_factor);
                    fprintf(f_bat, "%s.ncam = %u\n", fps, nd->cfg.pyrwfs.ncam);
                    fprintf(f_bat, "%s.pup_diam = %.8g\n", fps, cfg.pup_diam);
                    fprintf(f_bat, "%s.lambda_um = %.8g\n", fps, nd->cfg.pyrwfs.lambda_um);
                    fprintf(f_bat, "%s.bw_frac = %.8g\n", fps, nd->cfg.bw_frac);
                    fprintf(f_bat, "%s.nlambda = %d\n", fps, nd->cfg.pyrwfs.nlambda);
                    fprintf(f_bat, "%s.qe = %.8g\n", fps, nd->cfg.qe);
                    fprintf(f_bat, "%s.dt_exp = %.7f\n", fps, dt_s);
                    fprintf(f_bat, "%s.use_beam_flux = %s\n", fps,
                            nd->cfg.use_beam_flux ? "ON" : "OFF");
                    fprintf(f_bat, "%s.navg = %u\n", fps, nd->eff_navg);
                    fprintf(f_bat, "%s.mod_rad = %.8g\n", fps, nd->cfg.pyrwfs.mod_rad_ld);
                    fprintf(f_bat, "%s.mod_npts = %d\n", fps, nd->cfg.pyrwfs.mod_npts);
                    fprintf(f_bat, "%s.mod_mode = %d\n", fps, nd->cfg.pyrwfs.mod_subframe_mode);
                    fprintf(f_bat, "%s.mod_rev = %d\n", fps, nd->cfg.pyrwfs.mod_rev);
                    fprintf(f_bat, "%s.mod_phase_deg = %.8g\n", fps,
                            nd->cfg.pyrwfs.mod_phase_deg);
                    fprintf(f_bat, "%s.pup_sep = %.8g\n", fps, nd->cfg.pyrwfs.pup_sep);
                    fprintf(f_bat, "%s.roof_ld = %.8g\n", fps, nd->cfg.pyrwfs.roof_ld);
                    emit_reg_params(f_bat, fps, &nd->cfg.reg);
                    emit_det_params(f_bat, fps, &nd->cfg.pyrwfs.det);
                    fprintf(f_bat, "%s.frame_rate = 0\n", fps);
                    fprintf(f_bat, "%s.rng_seed = %llu\n", fps,
                            (unsigned long long) nd->cfg.pyrwfs.rng_seed);
                    fprintf(f_bat, "%s.backend = %d\n", fps, cfg.backend);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_cam);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
                {
                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-shwfs .in_dm= .in_wf= .in_amp= .in_pup="
                            " .in_ncpa=\n",
                            fps);
                    fprintf(f_bat, "%s.out_cam = %s\n", fps, nd->cfg.out_cam);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    emit_str_if_set(f_bat, fps, ".in_pup", cfg.in_pup);
                    emit_str_if_set(f_bat, fps, ".in_ncpa", nd->cfg.in_ncpa);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.n_sub = %u\n", fps, nd->cfg.shwfs.n_sub);
                    fprintf(f_bat, "%s.pix_per_sub = %u\n", fps, nd->cfg.shwfs.pix_per_sub);
                    fprintf(f_bat, "%s.pad_factor = %u\n", fps, nd->cfg.shwfs.pad_factor);
                    fprintf(f_bat, "%s.pup_diam = %.8g\n", fps, cfg.pup_diam);
                    fprintf(f_bat, "%s.lambda_um = %.8g\n", fps, nd->cfg.shwfs.lambda_um);
                    fprintf(f_bat, "%s.bw_frac = %.8g\n", fps, nd->cfg.bw_frac);
                    fprintf(f_bat, "%s.nlambda = %d\n", fps, nd->cfg.shwfs.nlambda);
                    fprintf(f_bat, "%s.qe = %.8g\n", fps, nd->cfg.qe);
                    fprintf(f_bat, "%s.dt_exp = %.7f\n", fps, dt_s);
                    fprintf(f_bat, "%s.use_beam_flux = %s\n", fps,
                            nd->cfg.use_beam_flux ? "ON" : "OFF");
                    fprintf(f_bat, "%s.navg = %u\n", fps, nd->eff_navg);
                    emit_reg_params(f_bat, fps, &nd->cfg.reg);
                    emit_det_params(f_bat, fps, &nd->cfg.shwfs.det);
                    fprintf(f_bat, "%s.frame_rate = 0\n", fps);
                    fprintf(f_bat, "%s.rng_seed = %llu\n", fps,
                            (unsigned long long) nd->cfg.shwfs.rng_seed);
                    fprintf(f_bat, "%s.backend = %d\n", fps, cfg.backend);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_cam);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
                {
                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-zwfs .in_dm= .in_wf= .in_amp= .in_pup="
                            " .in_ncpa=\n",
                            fps);
                    fprintf(f_bat, "%s.out_cam = %s\n", fps, nd->cfg.out_cam);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    emit_str_if_set(f_bat, fps, ".in_pup", cfg.in_pup);
                    emit_str_if_set(f_bat, fps, ".in_ncpa", nd->cfg.in_ncpa);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.ncam = %u\n", fps, nd->cfg.zwfs.ncam);
                    fprintf(f_bat, "%s.n_mft = %u\n", fps, nd->cfg.zwfs.n_mft);
                    fprintf(f_bat, "%s.pup_diam = %.8g\n", fps, cfg.pup_diam);
                    fprintf(f_bat, "%s.lambda_um = %.8g\n", fps, nd->cfg.zwfs.lambda_um);
                    fprintf(f_bat, "%s.dimple_ld = %.8g\n", fps, nd->cfg.zwfs.dimple_ld);
                    fprintf(f_bat, "%s.dimple_rad = %.8g\n", fps, nd->cfg.zwfs.dimple_rad);
                    fprintf(f_bat, "%s.dimple_amp = %.8g\n", fps, nd->cfg.zwfs.dimple_amp);
                    fprintf(f_bat, "%s.bw_frac = %.8g\n", fps, nd->cfg.bw_frac);
                    fprintf(f_bat, "%s.nlambda = %d\n", fps, nd->cfg.zwfs.nlambda);
                    fprintf(f_bat, "%s.qe = %.8g\n", fps, nd->cfg.qe);
                    fprintf(f_bat, "%s.dt_exp = %.7f\n", fps, dt_s);
                    fprintf(f_bat, "%s.use_beam_flux = %s\n", fps,
                            nd->cfg.use_beam_flux ? "ON" : "OFF");
                    fprintf(f_bat, "%s.navg = %u\n", fps, nd->eff_navg);
                    emit_reg_params(f_bat, fps, &nd->cfg.reg);
                    emit_det_params(f_bat, fps, &nd->cfg.zwfs.det);
                    fprintf(f_bat, "%s.frame_rate = 0\n", fps);
                    fprintf(f_bat, "%s.rng_seed = %llu\n", fps,
                            (unsigned long long) nd->cfg.zwfs.rng_seed);
                    fprintf(f_bat, "%s.backend = %d\n", fps, cfg.backend);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_cam);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
                {
                    fprintf(f_man,
                            "%s milk-fpsexec-cacaosim-fpcam .out_long= .in_dm= .in_wf= .in_amp="
                            " .in_pup= .in_ncpa=\n",
                            fps);
                    fprintf(f_bat, "%s.out_cam = %s\n", fps, nd->cfg.out_cam);
                    emit_str_if_set(f_bat, fps, ".out_long", nd->cfg.out_long);
                    fprintf(f_bat, "%s.in_beam = %s\n", fps, inb);
                    emit_str_if_set(f_bat, fps, ".in_pup", cfg.in_pup);
                    emit_str_if_set(f_bat, fps, ".in_ncpa", nd->cfg.in_ncpa);
                    fprintf(f_bat, "%s.nwf = %u\n", fps, cfg.nwf);
                    fprintf(f_bat, "%s.ncam = %u\n", fps, nd->cfg.fpcam.ncam);
                    fprintf(f_bat, "%s.prop_mode = %d\n", fps, nd->cfg.fpcam.prop_mode);
                    fprintf(f_bat, "%s.pad_factor = %u\n", fps, nd->cfg.fpcam.pad_factor);
                    fprintf(f_bat, "%s.pup_diam = %.8g\n", fps, cfg.pup_diam);
                    fprintf(f_bat, "%s.lambda_um = %.8g\n", fps, nd->cfg.fpcam.lambda_um);
                    fprintf(f_bat, "%s.bw_frac = %.8g\n", fps, nd->cfg.bw_frac);
                    fprintf(f_bat, "%s.nlambda = %d\n", fps, nd->cfg.fpcam.nlambda);
                    fprintf(f_bat, "%s.pixscale_ld = %.8g\n", fps, nd->cfg.fpcam.pixscale_ld);
                    fprintf(f_bat, "%s.qe = %.8g\n", fps, nd->cfg.qe);
                    fprintf(f_bat, "%s.dt_exp = %.7f\n", fps, dt_s);
                    fprintf(f_bat, "%s.use_beam_flux = %s\n", fps,
                            nd->cfg.use_beam_flux ? "ON" : "OFF");
                    fprintf(f_bat, "%s.navg = %u\n", fps, nd->eff_navg);
                    emit_reg_params(f_bat, fps, &nd->cfg.reg);
                    emit_det_params(f_bat, fps, &nd->cfg.fpcam.det);
                    fprintf(f_bat, "%s.frame_rate = 0\n", fps);
                    fprintf(f_bat, "%s.rng_seed = %llu\n", fps,
                            (unsigned long long) nd->cfg.fpcam.rng_seed);
                    fprintf(f_bat, "%s.backend = %d\n", fps, cfg.backend);

                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_cam);
                    add_unique_item(streams, &n_streams, PLAN_MAX_STREAMS, nd->cfg.out_long);
                }

                fprintf(f_bat, "%s.procinfo.triggermode = 3\n", fps);
                fprintf(f_bat, "%s.procinfo.triggersname = %s\n", fps, inb);
                fprintf(f_bat, "%s.procinfo.loopcntMax = -1\n", fps);
            }

            /* Emit forward topological waves d = 1..max_depth */
            for (int d = 1; d <= max_depth; d++)
            {
                int first_u = 1;
                for (uint32_t i = 0; i < ctx.n_nodes; i++)
                {
                    if (node_depth[i] == d)
                    {
                        if (first_u)
                        {
                            fprintf(f_wav, "RUNSTART\t");
                            first_u = 0;
                        }
                        else
                        {
                            fprintf(f_wav, " ");
                        }
                        fprintf(f_wav, "csim_%s", ctx.nodes[i].cfg.id);
                    }
                }
                if (!first_u)
                {
                    fprintf(f_wav, "\n");
                }

                int first_s = 1;
                for (uint32_t i = 0; i < ctx.n_nodes; i++)
                {
                    if (node_depth[i] != d)
                    {
                        continue;
                    }
                    const cacaosim_bench_node_t *nd = &ctx.nodes[i];

#define EMIT_W_SEP()                        \
    do                                      \
    {                                       \
        if (first_s)                        \
        {                                   \
            fprintf(f_wav, "WAIT\t");       \
            first_s = 0;                    \
        }                                   \
        else                                \
        {                                   \
            fprintf(f_wav, " ");            \
        }                                   \
    } while (0)

#define EMIT_BEAM_WAIT(bidx)                                                                 \
    do                                                                                       \
    {                                                                                        \
        if ((bidx) >= 0)                                                                     \
        {                                                                                    \
            EMIT_W_SEP();                                                                    \
            if (beam_poly[(bidx)])                                                           \
            {                                                                                \
                fprintf(f_wav, "%s+1", ctx.beams[(bidx)].name);                              \
            }                                                                                \
            else                                                                             \
            {                                                                                \
                fprintf(f_wav, "%s:%ux%ux2+1", ctx.beams[(bidx)].name, cfg.nwf, cfg.nwf);    \
            }                                                                                \
        }                                                                                    \
    } while (0)

                    if (nd->cfg.kind == CACAOSIM_NODE_DM)
                    {
                        char cmd[CACAOSIM_BENCH_NAME_LEN];
                        char opd[CACAOSIM_BENCH_NAME_LEN];
                        if (nd->cfg.in_cmd[0] != '\0')
                        {
                            snprintf(cmd, sizeof(cmd), "%s", nd->cfg.in_cmd);
                        }
                        else
                        {
                            snprintf(cmd, sizeof(cmd), "%.56s_cmd", nd->cfg.id);
                        }
                        if (nd->cfg.out_opd[0] != '\0')
                        {
                            snprintf(opd, sizeof(opd), "%s", nd->cfg.out_opd);
                        }
                        else
                        {
                            snprintf(opd, sizeof(opd), "csim_opd_%.52s", nd->cfg.id);
                        }
                        uint32_t nx = (nd->cfg.dm.dm_mode == 1) ? 2u : nd->cfg.dm.nact_x;
                        uint32_t ny = (nd->cfg.dm.dm_mode == 1) ? 1u : nd->cfg.dm.nact_y;
                        EMIT_W_SEP();
                        fprintf(f_wav, "%s:%ux%u", cmd, nx, ny);
                        EMIT_W_SEP();
                        fprintf(f_wav, "%s:%ux%u+1", opd, cfg.nwf, cfg.nwf);
                        EMIT_BEAM_WAIT(nd->out_idx);
                    }
                    else if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
                    {
                        EMIT_BEAM_WAIT(nd->refl_idx);
                        EMIT_BEAM_WAIT(nd->trans_idx);
                    }
                    else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
                    {
                        EMIT_BEAM_WAIT(nd->out_idx);
                        EMIT_BEAM_WAIT(nd->lowfs_idx);
                        if (nd->cfg.out_pre_lyot[0] != '\0')
                        {
                            EMIT_W_SEP();
                            fprintf(f_wav, "%s:%ux%u+1", nd->cfg.out_pre_lyot, cfg.nwf, cfg.nwf);
                        }
                        if (nd->cfg.out_pre_lyot_long[0] != '\0')
                        {
                            EMIT_W_SEP();
                            fprintf(f_wav, "%s:%ux%u+1", nd->cfg.out_pre_lyot_long, cfg.nwf,
                                    cfg.nwf);
                        }
                    }
                    else if (nd->cfg.kind == CACAOSIM_NODE_WFE)
                    {
                        char wfe_s[CACAOSIM_BENCH_NAME_LEN];
                        if (nd->cfg.in_wfe[0] != '\0')
                        {
                            snprintf(wfe_s, sizeof(wfe_s), "%s", nd->cfg.in_wfe);
                        }
                        else
                        {
                            snprintf(wfe_s, sizeof(wfe_s), "%.56s_wfe", nd->cfg.id);
                        }
                        EMIT_W_SEP();
                        fprintf(f_wav, "%s:%ux%u", wfe_s, cfg.nwf, cfg.nwf);
                        EMIT_BEAM_WAIT(nd->out_idx);
                    }
                    else if (nd->cfg.kind == CACAOSIM_NODE_ADC)
                    {
                        if (nd->cfg.out_state[0] != '\0')
                        {
                            EMIT_W_SEP();
                            fprintf(f_wav, "%s:%ux1+1", nd->cfg.out_state,
                                    (unsigned) CACAOSIM_ADC_STATE_LEN);
                        }
                        EMIT_BEAM_WAIT(nd->out_idx);
                    }
                    else if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS &&
                             nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
                    {
                        const char *plus = ((float) nd->eff_navg * dt_s <= 0.5f) ? "+1" : "";
                        if (nd->cfg.out_cam[0] != '\0')
                        {
                            EMIT_W_SEP();
                            fprintf(f_wav, "%s:%ux%u%s", nd->cfg.out_cam, nd->ncam, nd->ncam,
                                    plus);
                        }
                        if (nd->cfg.kind == CACAOSIM_NODE_FPCAM && nd->cfg.out_long[0] != '\0')
                        {
                            EMIT_W_SEP();
                            fprintf(f_wav, "%s:%ux%u%s", nd->cfg.out_long, nd->ncam, nd->ncam,
                                    plus);
                        }
                    }

#undef EMIT_BEAM_WAIT
#undef EMIT_W_SEP
                }
                if (!first_s)
                {
                    fprintf(f_wav, "\n");
                }
            }
        }

        fprintf(f_meta, "BENCH_NAME\t%s\n", cfg.name[0] ? cfg.name : "cacaosim_bench");
        fprintf(f_meta, "MODE\t%s\n", opts.modular_mode ? "modular" : "fused");
        fprintf(f_meta, "FLAT\t0\n");
        fprintf(f_meta, "NWF\t%u\n", cfg.nwf);
        fprintf(f_meta, "BASE_RATE\t%.6g\n", base_rate);
        fprintf(f_meta, "OVERSAMPLE\t%u\n", ctx.eff_oversample);
        fprintf(f_meta, "EFF_RATE\t%.6g\n", eff_rate);
        fprintf(f_meta, "PUP_DIAM\t%.8g\n", cfg.pup_diam);
        fprintf(f_meta, "COBS\t%.8g\n", cfg.cobs);
        fprintf(f_meta, "NSPIDER\t%d\n", cfg.nspider);
        fprintf(f_meta, "SPIDER_W\t%.8g\n", cfg.spider_w);
        fprintf(f_meta, "SPIDER_DEG\t%.8g\n", cfg.spider_deg);
        fprintf(f_meta, "SPIDER_OFFSET\t%.8g\n", cfg.spider_offset);
        fprintf(f_meta, "MKPUPIL_OUT\t%s\n", mkpupil_out);
        fprintf(f_meta, "IN_PUP\t%s\n", cfg.in_pup);
        fprintf(f_meta, "IN_WF\t%s\n", cfg.in_wf);
        fprintf(f_meta, "SIGNATURE\t%016llx\n", (unsigned long long) sig);
    }

    if (fclose(f_meta) != 0)
    {
        f_meta = NULL;
        goto fail_io;
    }
    f_meta = NULL;
    if (fclose(f_fits) != 0)
    {
        f_fits = NULL;
        goto fail_io;
    }
    f_fits = NULL;
    if (fclose(f_man) != 0)
    {
        f_man = NULL;
        goto fail_io;
    }
    f_man = NULL;
    if (fclose(f_bat) != 0)
    {
        f_bat = NULL;
        goto fail_io;
    }
    f_bat = NULL;
    if (fclose(f_wav) != 0)
    {
        f_wav = NULL;
        goto fail_io;
    }
    f_wav = NULL;

    if (write_list_file(plan_dir, "units.list", units, n_units, err_buf, err_cap) != 0 ||
        write_list_file(plan_dir, "streams.list", streams, n_streams, err_buf, err_cap) != 0)
    {
        return -1;
    }

    return 0;

fail_io:
    plan_err(err_buf, err_cap, "I/O error flushing deployment plan in '%s'", plan_dir);
fail_close:
    if (f_meta != NULL)
    {
        fclose(f_meta);
    }
    if (f_fits != NULL)
    {
        fclose(f_fits);
    }
    if (f_man != NULL)
    {
        fclose(f_man);
    }
    if (f_bat != NULL)
    {
        fclose(f_bat);
    }
    if (f_wav != NULL)
    {
        fclose(f_wav);
    }
    return -1;
}
