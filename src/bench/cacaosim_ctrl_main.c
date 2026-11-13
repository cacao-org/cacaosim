// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_ctrl_main.c
 * @brief   Standalone CLI front-end (`cacaosim-CTRL`) for the interactive AO bench TUI
 *
 * Architectural Role in `cacaosim-CTRL`:
 *   Provides the standalone executable entry point (`cacaosim-CTRL`, also invoked via
 *   `cacaosim-bench gui`) for configuring, wiring, inspecting, deploying, and monitoring
 *   multi-stage Adaptive Optics `.bench` simulations. Specifically, this front-end:
 *     1. Initializes the C locale (`LC_CTYPE`) for UTF-8 box-drawing and spectral block glyphs.
 *     2. Integrates with `milk_help.h` so `milk` tooling can query `-h`, `-h1`, `-h2`, and `-hm`
 *        standardized help banners consistently across all `milk` executables.
 *     3. Parses workspace, deployment, atmospheric turbulence, palette theme, color depth, and
 *        7-bit ASCII terminal override flags into `cacaosim_tui_opts_t`.
 *     4. Discovers the executable directory (`opts.exe_dir`) and resolves shorthand `.bench`
 *        arguments against active `MILK_SHM_DIR` state and example search paths before handing
 *        control to `cacaosim_tui_run()`.
 *
 * Design Rationale & Motivation:
 *   - Decoupling CLI argument parsing from the core TUI state machine (`cacaosim_tui.c`) keeps
 *     the configurator embeddable inside `libmilkcacaosim.so`, enabling headless plain-text
 *     frame snapshots (`cacaosim_tui_render_headless()`) in `ctest` suites without spawning a
 *     pseudo-terminal or linking `CLIcore`.
 *   - Resolving `opts.exe_dir` via `/proc/self/exe` at startup guarantees that interactive
 *     deployment commands (`[f:Deploy Fused]`, `[M:Deploy Modular]`, `[g:InitSHM]`) find their
 *     matching `milk-fpsexec-cacaosim-...` binaries in both local `_build/` trees and installed
 *     prefixes even when the build directory is not on `$PATH`.
 *   - Multi-tier `.bench` path resolution allows zero-argument `cacaosim-CTRL` invocations on an
 *     active RTC host to automatically attach to `$MILK_SHM_DIR/csim_active.bench`, while also
 *     allowing bare template filenames (e.g., `two_stage_coro.bench`) to resolve seamlessly from
 *     any current working directory.
 */

#include <libgen.h>
#include <limits.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "milk_help.h"
#include "bench/cacaosim_tui.h"

/* Standardized one-line and multi-line descriptions queried by milk_help (-h1 / -h2) */
static const char g_desc_short[] =
    "Interactive milk-CTRL-style AO hardware bench configurator and status GUI";

static const char g_desc_long[] =
    "Interactive flicker-free terminal GUI (modeled after milk-CTRL) for selecting,\n"
    "arranging, wiring, inspecting, saving, deploying, and monitoring cacaosim\n"
    "multi-stage Adaptive Optics hardware benches (.bench). Supports DMs, TTMs,\n"
    "dichroic/gray beamsplitters, coronagraphs with LOWFS pickoffs, Pyramid/SH/Zernike\n"
    "wavefront sensors, focal-plane science cameras, SGR mouse, and live MILK_SHM_DIR\n"
    "telemetry.";

/**
 * print_opt_line() - Format and print a single CLI flag entry in the help screen
 * @c:    Non-zero to emit ANSI color escapes (`MH_OPT`, `MH_ARG`, `MH_RST`); 0 for monochrome
 * @opt:  Short/long flag string (non-NULL, e.g., `"-c, --config"`)
 * @arg:  Optional argument placeholder (e.g., `"<file.bench>"`), or NULL / empty for boolean flags
 * @desc: Human-readable summary printed inline or indented on the following line (non-NULL)
 *
 * Role:
 *   Renders one aligned CLI option row inside `print_help_full()`, switching between single-line
 *   aligned columns for parameterless flags and two-line indented descriptions for flags taking
 *   value arguments.
 *
 * Rationale:
 *   Factoring ANSI color vs. monochrome (`-hm` / `NO_COLOR`) formatting and two-line wrapping for
 *   parameterized flags into one helper keeps help output under 80 columns on narrow SSH terminals
 *   while matching `milk-CTRL` visual conventions.
 */
static void print_opt_line(int c, const char *opt, const char *arg, const char *desc)
{
    if (c)
    {
        if (arg != NULL && arg[0] != '\0')
        {
            printf("  " MH_OPT "%s" MH_RST " " MH_ARG "%s" MH_RST "\n      %s\n", opt, arg, desc);
        }
        else
        {
            printf("  " MH_OPT "%-22s" MH_RST " %s\n", opt, desc);
        }
    }
    else
    {
        if (arg != NULL && arg[0] != '\0')
        {
            printf("  %s %s\n      %s\n", opt, arg, desc);
        }
        else
        {
            printf("  %-22s %s\n", opt, desc);
        }
    }
}

/**
 * print_help_full() - Render the complete categorized CLI reference banner to stdout
 * @prog:  Invoking program path or name (`argv[0]`, non-NULL)
 * @color: Non-zero for ANSI-colored output (`-h` / `--help`); 0 for plain text (`-hm`)
 *
 * Role:
 *   Emits the `milk_help` banner, synopsis, categorized option groups (Workspace & File, Display
 *   & Theme, Help), and cross-references (`cacaosim-bench-view`, `cacaosim-bench`, `milk-CTRL`).
 *
 * Rationale:
 *   Grouping options by workflow task (file/deployment vs. terminal rendering vs. help queries)
 *   and surfacing companion CLI tools in `See Also` lets users discover both one-shot schematic
 *   rendering (`cacaosim-bench-view`) and scripted bench orchestration (`cacaosim-bench`) directly
 *   from `-h`.
 */
static void print_help_full(const char *prog, int color)
{
    milk_help_banner(prog, g_desc_short, color);

    milk_help_section("Usage", color);
    if (color)
    {
        printf("  " MH_CMD "%s" MH_RST " [" MH_OPT "options" MH_RST "] ["
               MH_ARG "config.bench" MH_RST "]\n\n",
               prog);
    }
    else
    {
        printf("  %s [options] [config.bench]\n\n", prog);
    }

    milk_help_section("Workspace & File Options", color);
    print_opt_line(color, "-c, --config", "<file.bench>",
                   "Load or create .bench file (defaults to active SHM bench or template)");
    print_opt_line(color, "-s, --status", NULL,
                   "Start directly on F5:STATUS live MILK_SHM_DIR monitor tab");
    print_opt_line(color, "--schematic", NULL,
                   "Start directly on F3:SCHEMATIC full-screen optical diagram tab");
    print_opt_line(color, "--mode", "<fused|modular>",
                   "Default deployment engine mode (default: fused)");
    print_opt_line(color, "--atmturb", NULL,
                   "Enable milk-fpsexec-atmturb-mkwfs launch on deploy");
    print_opt_line(color, "--seeing", "<arcsec>",
                   "Atmospheric seeing [arcsec] when --atmturb is active (default: 0.7)");
    printf("\n");

    milk_help_section("Display & Theme Options", color);
    print_opt_line(color, "--theme", "<name>",
                   "milk palette: dark (default), night, accessible, light, nordic");
    print_opt_line(color, "--ascii", NULL, "Pure 7-bit ASCII borders, arrows, and spectrum bars");
    print_opt_line(color, "-p, --pretty", NULL, "Force 24-bit TrueColor ANSI output");
    print_opt_line(color, "-t, --text", NULL, "Monochrome terminal output (no color escapes)");
    printf("\n");

    milk_help_section("Help Options", color);
    print_opt_line(color, "-h, --help", NULL, "Show full help message");
    print_opt_line(color, "-h1", NULL, "One-line summary");
    print_opt_line(color, "-h2", NULL, "Verbose description");
    print_opt_line(color, "-hm", NULL, "Monochrome help (no ANSI color)");
    printf("\n");

    const char *see_also[] = {
        "cacaosim-bench-view:One-shot CLI & watch-mode optical bench visualizer",
        "cacaosim-bench:Turnkey AO bench launcher (gui, view, start, status, stop)",
        "milk-CTRL:Live milk shared-memory & FPS overview monitor",
    };
    milk_help_see_also(see_also, 3, color);
}

/**
 * get_exe_dir() - Determine the directory containing the running `cacaosim-CTRL` executable
 * @argv0:   Fallback `argv[0]` string if `/proc/self/exe` is unavailable (nullable)
 * @out_dir: Destination buffer receiving the parent directory path (non-NULL)
 * @cap:     Capacity of @out_dir in bytes
 *
 * Role:
 *   Populates `cacaosim_tui_opts_t.exe_dir` so the TUI and CLI path resolver can locate sibling
 *   helper executables (`milk-fpsexec-cacaosim-htsim`, `milk-fpsexec-cacaosim-mkpupil`, etc.) and
 *   relative `examples/` directories.
 *
 * Rationale:
 *   Developers frequently run `./_build/cacaosim-CTRL` directly from a CMake build directory
 *   without installing or adding `_build` to `$PATH`. Querying Linux `/proc/self/exe` resolves
 *   symlinks and `$PATH` lookups to the true binary directory, falling back cleanly to
 *   `dirname(argv0)` on non-procfs environments.
 */
static void get_exe_dir(const char *argv0, char *out_dir, size_t cap)
{
    /* Prefer Linux /proc/self/exe so symlinks and bare PATH invocations resolve accurately */
    char    link_buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", link_buf, sizeof(link_buf) - 1);
    if (n > 0)
    {
        link_buf[n] = '\0';
        snprintf(out_dir, cap, "%s", dirname(link_buf));
        return;
    }
    /* Fallback to dirname(argv[0]) when /proc is not mounted */
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", (argv0 != NULL) ? argv0 : ".");
    snprintf(out_dir, cap, "%s", dirname(tmp));
}

/**
 * resolve_bench_arg() - Resolve an optional CLI `.bench` specifier into a concrete file path
 * @spec:    User-supplied `.bench` path or bare filename, or NULL / empty when omitted
 * @exe_dir: Directory containing `cacaosim-CTRL` (from `get_exe_dir()`, non-NULL)
 * @out:     Destination buffer receiving resolved path, or `""` when starting from a built-in
 *           in-memory preset (non-NULL)
 * @cap:     Capacity of @out in bytes
 *
 * Role:
 *   Maps optional CLI arguments to the exact `.bench` file that `cacaosim_tui_run()` should load
 *   or create on save.
 *
 * Rationale:
 *   Supports three distinct operator workflows without extra flags:
 *     1. Live RTC inspection (`spec == NULL`): checks `$MILK_SHM_DIR/csim_active.bench` (written
 *        whenever a bench is deployed) so running `cacaosim-CTRL` attaches automatically to the
 *        currently active hardware simulation, or returns `""` to open a clean in-memory preset.
 *     2. Explicit file editing: returns `spec` verbatim if it is already readable in CWD or starts
 *        with `/` (allowing creation of new absolute paths).
 *     3. Quick example browsing: searches `$CACAOSIM_SRC_DIR/examples/`, installed
 *        `<prefix>/share/cacaosim/examples/`, and build-tree `<build>/../examples/` before
 *        treating a non-existent relative name as a new `.bench` file to create in CWD.
 */
static void resolve_bench_arg(const char *spec, const char *exe_dir, char *out, size_t cap)
{
    /* Tier 1: No file specified -> attach to active deployed SHM bench if present */
    if (spec == NULL || spec[0] == '\0')
    {
        const char *sd = getenv("MILK_SHM_DIR");
        char        cand[PATH_MAX];
        snprintf(cand, sizeof(cand), "%s/csim_active.bench",
                 (sd != NULL && sd[0] != '\0') ? sd : "/milk/shm");
        if (access(cand, R_OK) == 0)
        {
            snprintf(out, cap, "%s", cand);
        }
        else
        {
            out[0] = '\0';
        }
        return;
    }

    /* Tier 2: Existing local path or explicit absolute target path */
    if (access(spec, R_OK) == 0 || spec[0] == '/')
    {
        snprintf(out, cap, "%s", spec);
        return;
    }

    /* Tier 3: Search source, installed share/, and build-tree example directories */
    char cand[PATH_MAX];
    const char *src_env = getenv("CACAOSIM_SRC_DIR");
    if (src_env != NULL && src_env[0] != '\0')
    {
        snprintf(cand, sizeof(cand), "%.1500s/examples/%.512s", src_env, spec);
        if (access(cand, R_OK) == 0)
        {
            snprintf(out, cap, "%s", cand);
            return;
        }
    }
    snprintf(cand, sizeof(cand), "%.1500s/../share/cacaosim/examples/%.512s", exe_dir, spec);
    if (access(cand, R_OK) == 0)
    {
        snprintf(out, cap, "%s", cand);
        return;
    }
    snprintf(cand, sizeof(cand), "%.1500s/../examples/%.512s", exe_dir, spec);
    if (access(cand, R_OK) == 0)
    {
        snprintf(out, cap, "%s", cand);
        return;
    }

    /* Tier 4: Treat unmatched relative name as a new .bench file to create on save */
    snprintf(out, cap, "%s", spec);
}

/**
 * main() - CLI entry point for the `cacaosim-CTRL` interactive AO hardware configurator
 * @argc: Argument count
 * @argv: Argument vector
 *
 * Role:
 *   Configures UTF-8 locale handling, intercepts standardized `milk_help` flags, parses CLI
 *   options into `cacaosim_tui_opts_t`, resolves any `.bench` target path, and runs the
 *   interactive terminal session via `cacaosim_tui_run()`.
 *
 * Rationale:
 *   Calling `setlocale(LC_CTYPE, "")` first ensures standard C character classification and
 *   terminal locale detection reflect the user's UTF-8 environment before `milk_help` or ANSI
 *   capability detection runs. Accepting the `.bench` path either via `-c/--config` or as a
 *   positional argument matches both `cacaosim-bench` and `milk-CTRL` CLI ergonomics.
 *
 * Return: 0 on clean exit or help display, 1 on invalid CLI flag/theme, or non-zero if
 *         `cacaosim_tui_run()` encounters a fatal terminal initialization error.
 */
int main(int argc, char **argv)
{
    /* Activate environment UTF-8 locale before help and terminal capability checks */
    setlocale(LC_CTYPE, "");
    int hact = milk_help_init(argc, argv, g_desc_short, g_desc_long);
    if (hact == MH_ACTION_H1 || hact == MH_ACTION_H2)
    {
        return 0;
    }
    if (hact == MH_ACTION_HELP || hact == MH_ACTION_MONO)
    {
        print_help_full(argv[0], hact == MH_ACTION_HELP);
        return 0;
    }

    cacaosim_tui_opts_t opts;
    cacaosim_tui_opts_defaults(&opts);
    get_exe_dir(argv[0], opts.exe_dir, sizeof(opts.exe_dir));

    const char *bench_arg = NULL;

    /* Parse workspace, deployment, theme, and color-mode CLI overrides */
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        if (strcmp(a, "-s") == 0 || strcmp(a, "--status") == 0)
        {
            opts.initial_view = CACAOSIM_TUI_VIEW_STATUS;
        }
        else if (strcmp(a, "--schematic") == 0)
        {
            opts.initial_view = CACAOSIM_TUI_VIEW_SCHEMATIC;
        }
        else if (strcmp(a, "--ascii") == 0)
        {
            opts.ascii_only = 1;
        }
        else if (strcmp(a, "-p") == 0 || strcmp(a, "--pretty") == 0)
        {
            opts.color_level = 3;
        }
        else if (strcmp(a, "-t") == 0 || strcmp(a, "--text") == 0 || strcmp(a, "--no-color") == 0)
        {
            opts.color_level = 0;
        }
        else if (strcmp(a, "--atmturb") == 0)
        {
            opts.with_atmturb = 1;
        }
        else if (strcmp(a, "--theme") == 0 && i + 1 < argc)
        {
            int th = cacaosim_view_theme_from_name(argv[++i]);
            if (th < 0)
            {
                fprintf(stderr, MH_ERR "Unknown theme '%s'" MH_RST
                        " (choose: dark, night, accessible, light, nordic)\n",
                        argv[i]);
                return 1;
            }
            opts.theme = th;
        }
        else if ((strcmp(a, "-c") == 0 || strcmp(a, "--config") == 0) && i + 1 < argc)
        {
            bench_arg = argv[++i];
        }
        else if (strcmp(a, "--mode") == 0 && i + 1 < argc)
        {
            opts.deploy_mode = (strcasecmp(argv[++i], "modular") == 0) ? 1 : 0;
        }
        else if (strcmp(a, "--seeing") == 0 && i + 1 < argc)
        {
            opts.seeing_arcsec = strtof(argv[++i], NULL);
        }
        else if (a[0] == '-')
        {
            fprintf(stderr, MH_ERR "Unknown option '%s'" MH_RST " (try %s -h)\n", a, argv[0]);
            return 1;
        }
        else if (bench_arg == NULL)
        {
            bench_arg = a;
        }
    }

    char resolved[PATH_MAX];
    resolve_bench_arg(bench_arg, opts.exe_dir, resolved, sizeof(resolved));
    return cacaosim_tui_run(resolved[0] ? resolved : NULL, &opts);
}
