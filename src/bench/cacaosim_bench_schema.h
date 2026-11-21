// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_schema.h
 * @brief   Single-source-of-truth option registry, FPS runtime introspection, and schema exporters
 *
 * Architectural Role:
 *   Provides a compiled, zero-dependency catalog of every declarative `.bench` key and alias,
 *   runtime introspection of all 12 `MILK_RUNNABLE` FPS compute units (`libmilkcacaosim.so`), and
 *   structured CLI flag metadata for `cacaosim-bench`, `cacaosim-bench-view`, and `cacaosim-CTRL`.
 *   Powers:
 *     1. `cacaosim-bench options` / `cacaosim-bench-view --list-options` (terminal tables, GFM
 *        Markdown for `docs/OPTIONS.md`, and JSON for `share/cacaosim/options_schema.json`).
 *     2. The searchable `[?:Opts]` Option Reference overlay inside `cacaosim-CTRL`.
 *     3. Automated CTest drift verification (`cacaosim_validate.c`) proving parser, serializer,
 *        TUI inspector, and documentation remain 100% synchronized.
 */

#ifndef CACAOSIM_BENCH_SCHEMA_H
#define CACAOSIM_BENCH_SCHEMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "milk_runnable.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * enum cacaosim_opt_mode_t - Execution topology support for a `.bench` option
 * @CACAOSIM_OPT_MODE_BOTH:    Honored in both fused (`htsim`) and modular (`fpspipe`) modes.
 * @CACAOSIM_OPT_MODE_FUSED:   Honored in fused (`htsim`) DAG mode only.
 * @CACAOSIM_OPT_MODE_MODULAR: Honored in modular multi-process FPS mode only.
 */
typedef enum
{
    CACAOSIM_OPT_MODE_BOTH    = 0,
    CACAOSIM_OPT_MODE_FUSED   = 1,
    CACAOSIM_OPT_MODE_MODULAR = 2
} cacaosim_opt_mode_t;

/**
 * enum cacaosim_opt_tier_t - Visibility tier inside interactive configurators
 * @CACAOSIM_OPT_TIER_BASIC:    Shown by default in `cacaosim-CTRL` Inspector.
 * @CACAOSIM_OPT_TIER_ADVANCED: Revealed when `[v:Adv]` Advanced mode is toggled ON.
 */
typedef enum
{
    CACAOSIM_OPT_TIER_BASIC    = 0,
    CACAOSIM_OPT_TIER_ADVANCED = 1
} cacaosim_opt_tier_t;

/**
 * struct cacaosim_bench_opt_t - Authoritative metadata record for one `.bench` option
 * @section:       Primary section group (`bench`, `common`, `dm`, `splitter`, `coro`, `wfe`,
 *                 `sensor`, `pyrwfs`, `shwfs`, `zwfs`, `fpcam`).
 * @applies_to:    Comma-separated `.bench` section prefixes accepting this key.
 * @key:           Canonical key name emitted by `cacaosim_bench_format_str()`.
 * @aliases:       Comma-separated parser aliases accepted by `cacaosim_bench_parse_str()`.
 * @type:          Logical value type (`string`, `uint`, `int`, `float`, `bool`, `enum`, `preset`).
 * @enum_values:   Pipe-separated symbolic choices for `enum`/`preset` keys, or `""`.
 * @default_str:   Canonical default value string.
 * @unit:          Physical or logical unit (`um`, `nm`, `Hz`, `us`, `l/D`, `shm`, `beam`, etc.).
 * @range_str:     Valid domain summary (e.g. `"[0..1]"`, `">= 8"`, `""`).
 * @category:      Functional grouping (`optics`, `dynamics`, `detector`, `timing`, `wiring`,
 *                 `registration`, `photometry`, `runtime`).
 * @mode:          Execution-mode support (`BOTH`, `FUSED`, `MODULAR`).
 * @fps_param:     Corresponding standalone FPS unit parameter (e.g. `"dmsim.tau_sec"`), or `""`.
 * @tui_subtab:    `cacaosim-CTRL` Inspector subtab (`0`=OPTICS, `1`=DYN/DET, `2`=SHM/WIRING, -1).
 * @tier:          Basic vs Advanced visibility tier in interactive tools.
 * @is_serialized: 1 if `cacaosim_bench_format_str()` can emit this key; 0 for parser-only keys.
 * @description:   Concise human-readable description.
 */
typedef struct
{
    const char         *section;
    const char         *applies_to;
    const char         *key;
    const char         *aliases;
    const char         *type;
    const char         *enum_values;
    const char         *default_str;
    const char         *unit;
    const char         *range_str;
    const char         *category;
    cacaosim_opt_mode_t mode;
    const char         *fps_param;
    int                 tui_subtab;
    cacaosim_opt_tier_t tier;
    int                 is_serialized;
    const char         *description;
} cacaosim_bench_opt_t;

/**
 * struct cacaosim_cli_opt_t - Structured metadata record for one CLI flag
 * @tool:        Executable or script name (`cacaosim-bench`, `cacaosim-bench-view`,
 *               `cacaosim-CTRL`).
 * @flag:        Short and/or long flag synopsis (e.g. `"-c, --config"`).
 * @arg:         Argument metavariable (e.g. `"<file.bench>"`), or `""` for boolean flags.
 * @default_str: Default value summary, or `""`.
 * @group:       Functional group (`workflow`, `options`, `display`, `deploy`, `plan`).
 * @description: Concise human-readable description.
 */
typedef struct
{
    const char *tool;
    const char *flag;
    const char *arg;
    const char *default_str;
    const char *group;
    const char *description;
} cacaosim_cli_opt_t;

/**
 * struct cacaosim_schema_filter_t - Query filter applied by schema table/Markdown/JSON exporters
 * @section:       Optional `.bench` section name filter (exact or `applies_to` match).
 * @search:        Optional case-insensitive substring match across key, aliases, and description.
 * @mode:          Optional mode filter (`"fused"`, `"modular"`, or `"fused-only"`).
 * @fps_unit:      Optional FPS unit filter (e.g. `"pyrwfs"`, `"htsim"`, or `"all"`).
 * @live_only:     1 to include only `FPFLAG_WRITERUN` live-tunable FPS parameters.
 * @include_bench: 1 to emit `.bench` options.
 * @include_fps:   1 to emit introspected `MILK_RUNNABLE` FPS unit parameters.
 * @include_cli:   1 to emit CLI tool flag tables.
 */
typedef struct
{
    const char *section;
    const char *search;
    const char *mode;
    const char *fps_unit;
    int         live_only;
    int         include_bench;
    int         include_fps;
    int         include_cli;
} cacaosim_schema_filter_t;

/**
 * cacaosim_list_runnables() - Return the immutable plugin table of all `cacaosim` FPS runnables
 * @out_count: Optional destination receiving the number of registered runnables (12).
 *
 * Return: Pointer to the static array of `MILK_RUNNABLE` descriptors in `milkcacaosim.c`.
 */
const MILK_RUNNABLE *const *cacaosim_list_runnables(size_t *out_count);

/**
 * cacaosim_schema_bench_count() - Return total `.bench` option descriptors in the registry
 *
 * Return: Number of entries accessible via `cacaosim_schema_bench_get()`.
 */
size_t cacaosim_schema_bench_count(void);

/**
 * cacaosim_schema_bench_get() - Fetch one `.bench` option descriptor by 0-based index
 * @idx: Zero-based index `[0 .. cacaosim_schema_bench_count() - 1]`.
 *
 * Return: Pointer to immutable descriptor, or NULL if @idx is out of bounds.
 */
const cacaosim_bench_opt_t *cacaosim_schema_bench_get(size_t idx);

/**
 * cacaosim_schema_find_bench_opt() - Look up a `.bench` option by section and canonical key/alias
 * @section:      Section prefix (`bench`, `dm`, `ttm`, `splitter`, `coro`, `wfe`, `static_wfe`,
 *                `pyrwfs`, `shwfs`, `zwfs`, `fpcam`, `fpwfs`), or NULL to search any section.
 * @key_or_alias: Canonical key name or parser alias (case-insensitive).
 *
 * Return: Matching descriptor pointer, or NULL if not found.
 */
const cacaosim_bench_opt_t *cacaosim_schema_find_bench_opt(
    const char *section,
    const char *key_or_alias);

/**
 * cacaosim_schema_cli_count() - Return total CLI option descriptors in the registry
 *
 * Return: Number of entries accessible via `cacaosim_schema_cli_get()`.
 */
size_t cacaosim_schema_cli_count(void);

/**
 * cacaosim_schema_cli_get() - Fetch one CLI option descriptor by 0-based index
 * @idx: Zero-based index `[0 .. cacaosim_schema_cli_count() - 1]`.
 *
 * Return: Pointer to immutable descriptor, or NULL if @idx is out of bounds.
 */
const cacaosim_cli_opt_t *cacaosim_schema_cli_get(size_t idx);

/**
 * cacaosim_schema_filter_defaults() - Initialize a query filter to include all option layers
 * @flt: Destination filter struct (safe no-op if NULL).
 */
void cacaosim_schema_filter_defaults(cacaosim_schema_filter_t *flt);

/**
 * cacaosim_schema_render_table() - Print filtered options as aligned terminal tables
 * @flt:   Optional filter criteria (NULL uses defaults showing all layers).
 * @color: Non-zero to emit ANSI SGR color highlights; 0 for plain text.
 * @fp:    Destination output stream (defaults to `stdout` if NULL).
 *
 * Return: 0 on success, -1 on write error.
 */
int cacaosim_schema_render_table(
    const cacaosim_schema_filter_t *flt,
    int                             color,
    FILE                           *fp);

/**
 * cacaosim_schema_render_markdown() - Export filtered options as GitHub-Flavored Markdown
 * @flt: Optional filter criteria (NULL exports the complete `docs/OPTIONS.md` reference).
 * @fp:  Destination output stream (defaults to `stdout` if NULL).
 *
 * Return: 0 on success, -1 on write error.
 */
int cacaosim_schema_render_markdown(
    const cacaosim_schema_filter_t *flt,
    FILE                           *fp);

/**
 * cacaosim_schema_render_json() - Export filtered options as structured JSON
 * @flt: Optional filter criteria (NULL exports complete `options_schema.json`).
 * @fp:  Destination output stream (defaults to `stdout` if NULL).
 *
 * Return: 0 on success, -1 on write error.
 */
int cacaosim_schema_render_json(
    const cacaosim_schema_filter_t *flt,
    FILE                           *fp);

#ifdef __cplusplus
}
#endif

#endif /* CACAOSIM_BENCH_SCHEMA_H */
