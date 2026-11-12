// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_ctrl_main.c
 * @brief   cacaosim-CTRL standalone milk-CTRL-style AO hardware configurator and status TUI
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

static const char g_desc_short[] =
    "Interactive milk-CTRL-style AO hardware bench configurator and status GUI";

static const char g_desc_long[] =
    "Interactive flicker-free terminal GUI (modeled after milk-CTRL) for selecting,\n"
    "arranging, wiring, inspecting, saving, deploying, and monitoring cacaosim\n"
    "multi-stage Adaptive Optics hardware benches (.bench). Supports DMs, TTMs,\n"
    "dichroic/gray beamsplitters, coronagraphs with LOWFS pickoffs, Pyramid/SH/Zernike\n"
    "wavefront sensors, focal-plane science cameras, SGR mouse, and live MILK_SHM_DIR\n"
    "telemetry.";

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

static void get_exe_dir(const char *argv0, char *out_dir, size_t cap)
{
    char    link_buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", link_buf, sizeof(link_buf) - 1);
    if (n > 0)
    {
        link_buf[n] = '\0';
        snprintf(out_dir, cap, "%s", dirname(link_buf));
        return;
    }
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", (argv0 != NULL) ? argv0 : ".");
    snprintf(out_dir, cap, "%s", dirname(tmp));
}

static void resolve_bench_arg(const char *spec, const char *exe_dir, char *out, size_t cap)
{
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

    if (access(spec, R_OK) == 0 || spec[0] == '/')
    {
        snprintf(out, cap, "%s", spec);
        return;
    }

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
    snprintf(out, cap, "%s", spec);
}

int main(int argc, char **argv)
{
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
