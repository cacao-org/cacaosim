// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_view_main.c
 * @brief   Standalone CLI (cacaosim-bench-view) for .bench schematic visualization, starter
 *          template generation, live file/SHM watch, deployment, and cacaosim-CTRL TUI launch
 *
 * Architectural Role in cacaosim:
 *   Implements the standalone cacaosim-bench-view executable (linked against libmilkcacaosim.so
 *   without CLIcore dependencies). Acts as the primary user-facing terminal entry point for
 *   inspecting, scaffolding, monitoring, and launching optical bench configurations across five
 *   complementary modes:
 *     1. One-Shot Stream Schematic (default): Resolves <config.bench> across the working directory,
 *        CACAOSIM_SRC_DIR/examples, install-tree share/cacaosim/examples, MILK_ROOT, and
 *        MILK_SHM_DIR, then calls cacaosim_bench_render_file() to print the schematic, 32-bin
 *        spectra, and live SHM readiness badges directly into standard scrollback or pipes.
 *     2. Starter Template Scaffolding (--init <file.bench>): Generates a commented two-stage
 *        cascaded AO .bench configuration (8.2 m telescope -> woofer -> J-band PyWFS -> tweeter ->
 *        visible PyWFS + H-band science camera) and renders it immediately if no other .bench path
 *        was supplied.
 *     3. Continuous File & SHM Monitoring (-w / --watch): Enters the terminal alternate screen
 *        buffer without seizing raw keyboard input, polling nanosecond .bench file mtime and
 *        MILK_SHM_DIR directory mtime every 150 ms to redraw live whenever the user saves edits
 *        on disk or starts/stops SHM streams.
 *     4. Pre-Validated Turnkey Deployment (-d / --deploy / --run): Validates .bench syntax and
 *        multi-stage DAG photometry in-memory before invoking cacaosim-bench start (fused or
 *        modular mode, optional --atmturb), then renders the live post-deployment schematic.
 *     5. Interactive Builder & Live SHM TUI Handoff (-i / --interactive / -g / --gui): Delegates
 *        to cacaosim_tui_run() (cacaosim-CTRL). When invoked with no .bench argument, passes NULL
 *        so cacaosim-CTRL can synthesize a virtual cacaosim_bench_cfg_t from live MILK_SHM_DIR
 *        streams/FPS entries or start from built-in hardware templates.
 *
 * Design Rationale & Motivation:
 *   - Separating non-interactive stream rendering (default mode) from alternate-screen watch/TUI
 *     modes ensures cacaosim-bench-view works cleanly inside automated scripts (cacaosim-bench
 *     start/status), SSH commands, CI logs, and pagers (NO_COLOR, TERM=dumb, TIOCGWINSZ auto
 *     width) while still offering a zero-configuration --watch companion for external text editors
 *     ($VISUAL / $EDITOR) and direct -i escalation into the full cacaosim-CTRL GUI.
 *   - Async-signal-safe terminal cleanup (handle_sig() using raw write(2) + tcsetattr(3)) ensures
 *     SIGINT / SIGTERM never leaves the operator's terminal trapped on the alternate screen or
 *     with cursor/echo disabled.
 */

#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <locale.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "milk_help.h"
#include "bench/cacaosim_bench_view.h"
#include "bench/cacaosim_tui.h"

/** Saved original STDIN termios attributes restored upon exiting raw watch/TUI mode or signal. */
static struct termios        g_orig_termios;

/** Async-signal-safe flag: 1 while alternate screen (and optional raw termios) is active. */
static volatile sig_atomic_t g_raw_active = 0;

/** Async-signal-safe flag: set to 1 by SIGINT/SIGTERM handler to break out of the watch loop. */
static volatile sig_atomic_t g_stop_req   = 0;

/** DECSET 1049 (enter alternate screen buffer) + DECTCEM reset (hide text cursor). */
static const char g_tui_enter_seq[] = "\033[?1049h\033[?25l";

/** DECTCEM set (show cursor) + SGR 0 reset + DECRST 1049 (restore primary screen buffer). */
static const char g_tui_exit_seq[]  = "\033[?25h\033[0m\033[?1049l";

/**
 * handle_sig() - Async-signal-safe SIGINT/SIGTERM handler restoring terminal state
 * @sig: Delivered POSIX signal number (unused)
 *
 * Role: Flags clean termination of run_watch_or_tui() (@g_stop_req = 1) and immediately restores
 * the primary screen buffer, cursor visibility, and original STDIN termios flags if active.
 *
 * Rationale: Uses only async-signal-safe POSIX primitives (write(2) and tcsetattr(3), avoiding
 * stdio printf/fflush locks) so Ctrl-C or kill(1) during --watch never corrupts the calling shell.
 */
static void handle_sig(int sig)
{
    (void) sig;
    g_stop_req = 1;
    if (g_raw_active)
    {
        if (write(STDOUT_FILENO, g_tui_exit_seq, sizeof(g_tui_exit_seq) - 1) < 0)
        {
        }
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_raw_active = 0;
    }
}

/**
 * tui_enter() - Switch STDOUT to the alternate screen buffer and optionally put STDIN in raw mode
 * @raw_keys: Non-zero to disable canonical line buffering, echo, XON/XOFF, and CR->NL translation
 *            for single-keypress interactive dispatch; 0 in passive --watch mode so Ctrl-C still
 *            generates kernel SIGINT normally
 *
 * Role: Enters flicker-free full-frame redraw mode prior to run_watch_or_tui() or re-enters it
 * after returning from an external $VISUAL / $EDITOR subshell.
 *
 * Rationale: Guarding with @g_raw_active makes repeated calls idempotent, while keeping ISIG
 * enabled (and skipping termios mutation entirely when @raw_keys == 0) lets passive --watch mode
 * respond instantaneously to standard terminal signals.
 */
static void tui_enter(int raw_keys)
{
    if (g_raw_active)
    {
        return;
    }
    if (raw_keys && isatty(STDIN_FILENO))
    {
        if (tcgetattr(STDIN_FILENO, &g_orig_termios) == 0)
        {
            struct termios raw = g_orig_termios;
            raw.c_iflag &= ~(unsigned int) (IXON | ICRNL);
            raw.c_lflag &= ~(unsigned int) (ECHO | ICANON);
            raw.c_cc[VMIN]  = 0;
            raw.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        }
    }
    if (write(STDOUT_FILENO, g_tui_enter_seq, sizeof(g_tui_enter_seq) - 1) < 0)
    {
    }
    g_raw_active = 1;
}

/**
 * tui_exit() - Restore primary screen buffer, cursor visibility, and original STDIN termios state
 * @raw_keys: Non-zero if STDIN termios was modified on entry and must be restored via TCSAFLUSH
 *
 * Role: Cleanly tears down alternate-screen state when exiting run_watch_or_tui() or suspending
 * the overlay to launch an external text editor ($VISUAL / $EDITOR).
 *
 * Rationale: Emitting g_tui_exit_seq before restoring termios ensures the external editor or
 * parent shell inherits a visible cursor, reset SGR colors, and an unpolluted scrollback buffer.
 */
static void tui_exit(int raw_keys)
{
    if (!g_raw_active)
    {
        return;
    }
    if (write(STDOUT_FILENO, g_tui_exit_seq, sizeof(g_tui_exit_seq) - 1) < 0)
    {
    }
    if (raw_keys && isatty(STDIN_FILENO))
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    }
    g_raw_active = 0;
}

/** One-line CLI summary passed to milk_help_init() (-h1) and help banners. */
static const char g_desc_short[] =
    "Graphical ANSI/UTF-8 optical bench (.bench) visualizer and launcher";

/** Multi-line architectural summary passed to milk_help_init() (-h2). */
static const char g_desc_long[] =
    "Parses a declarative cacaosim .bench file, validates the multi-stage optical DAG,\n"
    "propagates 32-bin (0.40..2.40 um) stellar photometry across all dichroic/coronagraph\n"
    "arms, and renders a milk-styled TrueColor/256/mono optical schematic with 2D pupil\n"
    "preview, spectral sparklines, live MILK_SHM_DIR stream status, and interactive\n"
    "edit/deploy/stop TUI controls.";

/**
 * print_opt_line() - Format a single CLI option flag, optional metavariable, and description
 * @c:    Non-zero to emit milk_help ANSI highlight macros (MH_OPT, MH_ARG), 0 for plain monochrome
 * @opt:  Option flag synopsis (e.g. "-w, --watch" or "--theme")
 * @arg:  Optional argument placeholder (e.g. "<name>" or NULL when flag takes no value)
 * @desc: Concise human-readable description of the option's effect
 *
 * Role: Helper used by print_help_full() to align flag tables consistently across colored (-h)
 * and monochrome (-hm / non-TTY) help outputs.
 *
 * Rationale: Options carrying argument metavariables wrap their description onto an indented
 * continuation line so long "<fused|modular>" or "<file.bench>" placeholders never overflow 80-col
 * terminals, whereas boolean flags stay compact on a single 22-column-aligned line.
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
 * print_help_full() - Render the complete categorized milk-styled CLI reference to stdout
 * @prog:  Invoked program name (argv[0]) displayed in the banner and usage synopsis
 * @color: Non-zero for ANSI-colored output (-h on TTY), 0 for monochrome (-hm or piped help)
 *
 * Role: Displays Usage, Workflow Modes, Display & Theme Options, Deploy Options, Help Options,
 * and cross-references ("See Also") conforming to milk's CLI help standard.
 *
 * Rationale: Grouping flags by operator intent (Workflow vs Display vs Deploy) makes it obvious
 * that cacaosim-bench-view supports one-shot rendering, --watch live reload, --init template
 * creation, --deploy launch, and -i / --gui interactive TUI escalation from one unified tool.
 */
static void print_help_full(const char *prog, int color)
{
    milk_help_banner(prog, g_desc_short, color);

    milk_help_section("Usage", color);
    if (color)
    {
        printf("  " MH_CMD "%s" MH_RST " [" MH_OPT "options" MH_RST "] "
               MH_ARG "<config.bench>" MH_RST "\n\n",
               prog);
    }
    else
    {
        printf("  %s [options] <config.bench>\n\n", prog);
    }

    milk_help_section("Workflow Modes", color);
    print_opt_line(color, "(default)", NULL,
                   "Render optical schematic, spectra, and hardware tables once");
    print_opt_line(color, "-i, --interactive", NULL,
                   "Launch milk-CTRL-style builder & status TUI (cacaosim-CTRL)");
    print_opt_line(color, "-g, --gui", NULL,
                   "Alias for -i / --interactive hardware configuration GUI");
    print_opt_line(color, "-w, --watch", NULL,
                   "Auto-reload diagram live whenever <config.bench> is saved on disk");
    print_opt_line(color, "--init", "<file.bench>",
                   "Write a commented starter 2-stage .bench template to edit");
    print_opt_line(color, "-d, --deploy", NULL,
                   "Validate, deploy & start via cacaosim-bench, then visualize");
    printf("\n");

    milk_help_section("Display & Theme Options", color);
    print_opt_line(color, "-p, --pretty", NULL,
                   "Force TrueColor ANSI output even when piped (e.g. | less -R)");
    print_opt_line(color, "-t, --text", NULL,
                   "Plain monochrome text output (no ANSI color escapes)");
    print_opt_line(color, "--ascii", NULL, "Pure 7-bit ASCII box-drawing and spectrum bars");
    print_opt_line(color, "--theme", "<name>",
                   "milk palette: dark (default), night, accessible, light, nordic");
    print_opt_line(color, "--compact", NULL,
                   "Show overview + optical schematic only (omit hardware tables)");
    print_opt_line(color, "--no-shm", NULL, "Do not probe MILK_SHM_DIR for live .im.shm status");
    print_opt_line(color, "--no-footer", NULL, "Omit CLI workflow footer line");
    print_opt_line(color, "-W, --width", "<cols>", "Override panel width in columns [76..124]");
    printf("\n");

    milk_help_section("Deploy Options", color);
    print_opt_line(color, "--mode", "<fused|modular>",
                   "Execution mode passed to cacaosim-bench (default: fused)");
    print_opt_line(color, "--atmturb", NULL,
                   "Also launch milk-fpsexec-atmturb-mkwfs turbulence stream");
    print_opt_line(color, "--seeing", "<arcsec>",
                   "Seeing [arcsec] when --atmturb is enabled (default: 0.7)");
    printf("\n");

    milk_help_section("Help Options", color);
    print_opt_line(color, "-h, --help", NULL, "Show full help message");
    print_opt_line(color, "-h1", NULL, "One-line summary");
    print_opt_line(color, "-h2", NULL, "Verbose description");
    print_opt_line(color, "-hm", NULL, "Monochrome help (no ANSI color)");
    printf("\n");

    const char *see_also[] = {
        "cacaosim-bench:Turnkey AO bench launcher (view, edit, tui, start, status, stop)",
        "milk-fpsexec-cacaosim-htsim:Fused multi-stage DAG hardware simulator",
        "milk-stream-graph:Inspect live SHM stream lineage in MILK_SHM_DIR",
    };
    milk_help_see_also(see_also, 3, color);
}

/**
 * get_exe_dir() - Resolve the directory containing the running cacaosim-bench-view binary
 * @argv0:   Fallback program path from main()'s argv[0] (may be NULL)
 * @out_dir: Destination buffer receiving NUL-terminated directory path (non-NULL)
 * @cap:     Capacity of @out_dir in bytes
 *
 * Role: Locates the binary's build or install prefix so sibling scripts (cacaosim-bench) and
 * bundled example benches (../share/cacaosim/examples, ../examples) can be discovered without
 * requiring manual environment variable setup.
 *
 * Rationale: Resolves Linux /proc/self/exe first because PATH-dispatched invocations set argv[0]
 * to a bare command name ("cacaosim-bench-view") whose dirname() would Collapse to ".", breaking
 * relative asset lookup when run outside the build/install directory.
 */
static void get_exe_dir(const char *argv0, char *out_dir, size_t cap)
{
    char link_buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", link_buf, sizeof(link_buf) - 1);
    if (n > 0)
    {
        link_buf[n] = '\0';
        char *d     = dirname(link_buf);
        snprintf(out_dir, cap, "%s", d);
        return;
    }
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", (argv0 != NULL) ? argv0 : ".");
    snprintf(out_dir, cap, "%s", dirname(tmp));
}

/**
 * resolve_bench_file() - Locate a .bench file across local, source, install, and SHM search roots
 * @spec:    User-supplied file path or bare example filename (e.g. "two_stage_ao.bench")
 * @exe_dir: Directory of the running executable from get_exe_dir() (non-NULL)
 * @out:     Destination buffer receiving resolved readable path (or @spec fallback on miss)
 * @cap:     Capacity of @out in bytes
 *
 * Role: Resolves shorthand .bench names across six ordered candidate locations:
 *   1. Direct relative/absolute path @spec in the current working directory.
 *   2. $CACAOSIM_SRC_DIR/examples/<spec> (developer source checkout).
 *   3. <exe_dir>/../share/cacaosim/examples/<spec> (installed FHS prefix).
 *   4. <exe_dir>/../examples/<spec> (uninstalled CMake _build/ directory).
 *   5. $MILK_ROOT/share/cacaosim/examples/<spec> (active milk prefix).
 *   6. $MILK_SHM_DIR/<spec> (runtime-generated .bench files stored alongside active SHM loops).
 *
 * Rationale: Letting operators type "cacaosim-bench-view two_stage_ao.bench" from any working
 * directory eliminates tedious path lookups while still copying @spec verbatim into @out on miss
 * so downstream cacaosim_bench_render_file() or cacaosim-CTRL can report the exact missing path or
 * create a new file there.
 *
 * Return: 0 if an existing readable file was resolved into @out, -1 if @spec is empty or not found.
 */
static int resolve_bench_file(const char *spec, const char *exe_dir, char *out, size_t cap)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return -1;
    }
    if (access(spec, R_OK) == 0)
    {
        snprintf(out, cap, "%s", spec);
        return 0;
    }
    if (spec[0] != '/')
    {
        const char *src_env = getenv("CACAOSIM_SRC_DIR");
        if (src_env != NULL && src_env[0] != '\0')
        {
            snprintf(out, cap, "%.1500s/examples/%.512s", src_env, spec);
            if (access(out, R_OK) == 0)
            {
                return 0;
            }
        }
        snprintf(out, cap, "%.1500s/../share/cacaosim/examples/%.512s", exe_dir, spec);
        if (access(out, R_OK) == 0)
        {
            return 0;
        }
        snprintf(out, cap, "%.1500s/../examples/%.512s", exe_dir, spec);
        if (access(out, R_OK) == 0)
        {
            return 0;
        }
        const char *milk_root = getenv("MILK_ROOT");
        if (milk_root != NULL && milk_root[0] != '\0')
        {
            snprintf(out, cap, "%.1500s/share/cacaosim/examples/%.512s", milk_root, spec);
            if (access(out, R_OK) == 0)
            {
                return 0;
            }
        }
        const char *shm_dir = getenv("MILK_SHM_DIR");
        if (shm_dir != NULL && shm_dir[0] != '\0')
        {
            snprintf(out, cap, "%.1500s/%.512s", shm_dir, spec);
            if (access(out, R_OK) == 0)
            {
                return 0;
            }
        }
    }
    snprintf(out, cap, "%.2048s", spec);
    return -1;
}

/**
 * write_starter_template() - Write a self-documenting two-stage AO .bench template to @dest_path
 * @dest_path: Target filesystem path to create (refuses to overwrite existing files; non-NULL)
 *
 * Role: Implements "--init <file.bench>", writing a complete, physically realistic cascaded
 * two-stage AO configuration (8.2 m telescope, 2 kHz master clock, 12x12 woofer DM, J-band
 * band-reflect dichroic + 1 kHz PyWFS, 24x24 tweeter DM, 1.0 um long-pass dichroic + 2 kHz visible
 * PyWFS, and 100 Hz H-band focal-plane science camera).
 *
 * Rationale: Giving users a working multi-rate, multi-dichroic template with CACAO-compatible SHM
 * stream names (dm01disp, aol1_wfsim, dm02disp, aol2_wfsim) is faster and less error-prone than
 * writing section headers from scratch, while failing safely via access(dest_path, F_OK) prevents
 * accidental clobbering of existing bench designs.
 *
 * Return: 0 on success, -1 if @dest_path already exists or cannot be opened for writing.
 */
static int write_starter_template(const char *dest_path)
{
    if (access(dest_path, F_OK) == 0)
    {
        fprintf(stderr, "[cacaosim-bench-view] File '%s' already exists.\n", dest_path);
        return -1;
    }
    FILE *fp = fopen(dest_path, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "[cacaosim-bench-view] Cannot write '%s': %s\n", dest_path,
                strerror(errno));
        return -1;
    }
    fprintf(fp,
            "# Declarative cacaosim Optical Bench Configuration (.bench)\n"
            "# Edit parameters below, visualize with: cacaosim-bench-view %s\n"
            "# Deploy and run with:                   cacaosim-bench start --config %s\n\n"
            "[bench]\n"
            "name       = \"custom_two_stage_ao\"\n"
            "nwf        = 128\n"
            "pup_diam   = 0.9375\n"
            "cobs       = 0.30\n"
            "nspider    = 4\n"
            "spider_w   = 0.012\n"
            "spider_deg = 45.0\n"
            "tel_diam_m = 8.2\n"
            "star_mag   = 8.0\n"
            "star_teff  = 5778.0\n"
            "tput_tel   = 0.75\n"
            "master_hz  = 2000.0\n"
            "in_wf      = \"outarraypha\"\n"
            "in_pup     = \"csim_pup\"\n"
            "out_beam   = \"b_pup\"\n\n"
            "[dm.woofer]\n"
            "in           = \"b_pup\"\n"
            "out          = \"b_post_dm1\"\n"
            "in_cmd       = \"dm01disp\"\n"
            "nact_x       = 12\n"
            "nact_y       = 12\n"
            "max_stroke   = 8.0\n"
            "tau_us       = 250.0\n"
            "delay_frames = 1.0\n\n"
            "[splitter.dichroic_stg1]\n"
            "in         = \"b_post_dm1\"\n"
            "out_refl   = \"b_nir_wfs\"\n"
            "out_trans  = \"b_stg2_in\"\n"
            "mode       = \"dichroic_bp\"\n"
            "lambda_cut = 1.22\n"
            "band_width = 0.30\n\n"
            "[pyrwfs.wfs_nir]\n"
            "in        = \"b_nir_wfs\"\n"
            "out_cam   = \"aol1_wfsim\"\n"
            "ncam      = 64\n"
            "lambda_um = 1.22\n"
            "navg      = 2\n"
            "mod_rad   = 3.0\n\n"
            "[dm.tweeter]\n"
            "in           = \"b_stg2_in\"\n"
            "out          = \"b_post_dm2\"\n"
            "in_cmd       = \"dm02disp\"\n"
            "nact_x       = 24\n"
            "nact_y       = 24\n"
            "max_stroke   = 3.0\n"
            "tau_us       = 80.0\n"
            "delay_frames = 1.0\n\n"
            "[splitter.dichroic_stg2]\n"
            "in         = \"b_post_dm2\"\n"
            "out_refl   = \"b_vis_wfs\"\n"
            "out_trans  = \"b_sci\"\n"
            "mode       = \"dichroic_lp\"\n"
            "lambda_cut = 1.00\n\n"
            "[pyrwfs.wfs_vis]\n"
            "in        = \"b_vis_wfs\"\n"
            "out_cam   = \"aol2_wfsim\"\n"
            "ncam      = 64\n"
            "lambda_um = 0.75\n"
            "navg      = 1\n"
            "mod_rad   = 2.0\n\n"
            "[fpcam.sci_h]\n"
            "in        = \"b_sci\"\n"
            "out_cam   = \"csim_fpcam\"\n"
            "out_long  = \"csim_fplong\"\n"
            "ncam      = 128\n"
            "lambda_um = 1.65\n"
            "navg      = 20\n",
            dest_path, dest_path);
    fclose(fp);
    return 0;
}

/**
 * find_bench_script() - Locate the executable cacaosim-bench orchestration script
 * @exe_dir: Directory containing the running binary (non-NULL)
 * @out:     Destination buffer receiving resolved script path or PATH command name (non-NULL)
 * @cap:     Capacity of @out in bytes
 *
 * Role: Finds the companion cacaosim-bench launcher across installed bin/, $CACAOSIM_SRC_DIR,
 * uninstalled ../scripts/, or $PATH fallback so --deploy and interactive start/stop actions work
 * identically in build trees and installed prefixes.
 *
 * Rationale: Checking sibling <exe_dir>/cacaosim-bench before falling back to $PATH guarantees
 * that testing a local build invokes the matching local launcher rather than a stale system-wide
 * installation.
 *
 * Return: Always 0 (@out is populated with either a verified executable path or "cacaosim-bench").
 */
static int find_bench_script(const char *exe_dir, char *out, size_t cap)
{
    snprintf(out, cap, "%.1500s/cacaosim-bench", exe_dir);
    if (access(out, X_OK) == 0)
    {
        return 0;
    }
    const char *src_env = getenv("CACAOSIM_SRC_DIR");
    if (src_env != NULL && src_env[0] != '\0')
    {
        snprintf(out, cap, "%.1500s/scripts/cacaosim-bench", src_env);
        if (access(out, X_OK) == 0)
        {
            return 0;
        }
    }
    snprintf(out, cap, "%.1500s/../scripts/cacaosim-bench", exe_dir);
    if (access(out, X_OK) == 0)
    {
        return 0;
    }
    snprintf(out, cap, "cacaosim-bench");
    return 0;
}

/**
 * run_bench_action() - Invoke cacaosim-bench start or stop and capture a concise status summary
 * @exe_dir:      Directory of the running executable used to locate cacaosim-bench (non-NULL)
 * @subcmd:       Subcommand string ("start" or "stop")
 * @bench_path:   Path to .bench configuration file (used when @subcmd != "stop")
 * @mode:         Deployment engine mode ("fused" or "modular")
 * @with_atmturb: Non-zero to append "--atmturb" so atmospheric turbulence starts alongside bench
 * @seeing:       Seeing FWHM in arcseconds passed to "--seeing"
 * @status_out:   Output buffer receiving single-line human-readable outcome message (non-NULL)
 * @status_cap:   Capacity of @status_out in bytes
 *
 * Role: Executes process orchestration via cacaosim-bench (--no-view) over a pipe and distills
 * its stdout/stderr stream into its final non-empty diagnostic line for UI/CLI display.
 *
 * Rationale: Passing "--no-view" prevents recursive invocation of cacaosim-bench-view by the
 * shell script, while capturing stdout+stderr via popen() keeps subprocess chatter from corrupting
 * the alternate screen buffer during interactive deployments.
 *
 * Return: 0 if cacaosim-bench exited with status 0, -1 on popen() failure or non-zero exit code.
 */
static int run_bench_action(
    const char *exe_dir,
    const char *subcmd,
    const char *bench_path,
    const char *mode,
    int         with_atmturb,
    float       seeing,
    char       *status_out,
    size_t      status_cap)
{
    char script[PATH_MAX];
    find_bench_script(exe_dir, script, sizeof(script));

    char cmd[4608];
    if (strcmp(subcmd, "stop") == 0)
    {
        snprintf(cmd, sizeof(cmd), "\"%.1800s\" stop --no-view 2>&1", script);
    }
    else
    {
        snprintf(cmd, sizeof(cmd),
                 "\"%.1800s\" start --config \"%.1800s\" --mode %.32s %s --seeing %.2f"
                 " --no-view 2>&1",
                 script, bench_path, mode, with_atmturb ? "--atmturb" : "", seeing);
    }

    FILE *pp = popen(cmd, "r");
    if (pp == NULL)
    {
        snprintf(status_out, status_cap, "Failed to invoke cacaosim-bench");
        return -1;
    }

    /* Drain subprocess output and retain the last non-blank status/error line */
    char last_line[256] = { 0 };
    char line[256];
    while (fgets(line, sizeof(line), pp) != NULL)
    {
        size_t l = strlen(line);
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r'))
        {
            line[--l] = '\0';
        }
        if (l > 0)
        {
            snprintf(last_line, sizeof(last_line), "%s", line);
        }
    }
    int rc = pclose(pp);
    if (rc == 0)
    {
        if (strcmp(subcmd, "stop") == 0)
        {
            snprintf(status_out, status_cap, "Stopped running bench units in MILK_SHM_DIR.");
        }
        else
        {
            snprintf(status_out, status_cap, "Deployed & running (%.16s%s): %.200s", mode,
                     with_atmturb ? " + atmturb" : "",
                     last_line[0] ? last_line : bench_path);
        }
        return 0;
    }
    snprintf(status_out, status_cap, "Action failed (rc=%d): %.220s", WEXITSTATUS(rc),
             last_line[0] ? last_line : cmd);
    return -1;
}

/**
 * launch_editor() - Open @bench_path synchronously in the user's preferred terminal text editor
 * @bench_path: Path to the .bench file to edit (non-NULL)
 *
 * Role: Spawns $VISUAL, $EDITOR, or POSIX fallback "vi" when the user presses 'e' inside
 * run_watch_or_tui().
 *
 * Rationale: Respecting the standard Unix VISUAL -> EDITOR -> vi precedence lets operators edit
 * raw INI/TOML comments and parameters in their own configured editor and immediately see the
 * updated schematic upon quitting.
 */
static void launch_editor(const char *bench_path)
{
    const char *ed = getenv("VISUAL");
    if (ed == NULL || ed[0] == '\0')
    {
        ed = getenv("EDITOR");
    }
    if (ed == NULL || ed[0] == '\0')
    {
        ed = "vi";
    }
    char cmd[4608];
    snprintf(cmd, sizeof(cmd), "%.256s \"%.2048s\"", ed, bench_path);
    int rc = system(cmd);
    (void) rc;
}

/**
 * file_mtime_token() - Return a 64-bit nanosecond modification timestamp token for @path
 * @path: Filesystem path to stat (non-NULL)
 *
 * Role: Detects on-disk saves of <config.bench> inside the 150 ms --watch polling loop.
 *
 * Rationale: Combining tv_sec * 1e9 + tv_nsec catches rapid sub-second saves from modern editors
 * and automated formatters that coarse 1-second time_t comparisons would miss.
 *
 * Return: Monotonic-per-save nanosecond mtime token, or 0 if stat() fails.
 */
static uint64_t file_mtime_token(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        return 0;
    }
    return ((uint64_t) st.st_mtim.tv_sec * 1000000000ULL) + (uint64_t) st.st_mtim.tv_nsec;
}

/**
 * count_shm_files() - Return a lightweight change-detection stamp for MILK_SHM_DIR
 *
 * Role: Detects creation or removal of .im.shm files in MILK_SHM_DIR during --watch monitoring
 * so LIVE / STANDBY stream badges update automatically when loops start or stop externally.
 *
 * Rationale: Stat'ing the MILK_SHM_DIR directory inode mtime (rather than readdir()-scanning
 * hundreds of SHM entries every 150 ms) costs a single VFS lookup on tmpfs/devshm and changes
 * whenever any .im.shm entry is linked or unlinked in that directory.
 *
 * Return: Non-negative directory modification stamp, or 0 if MILK_SHM_DIR does not exist.
 */
static int count_shm_files(void)
{
    const char *sd = getenv("MILK_SHM_DIR");
    char        p[512];
    snprintf(p, sizeof(p), "%s", (sd && sd[0]) ? sd : "/milk/shm");
    struct stat st;
    if (stat(p, &st) != 0)
    {
        return 0;
    }
    return (int) (st.st_mtim.tv_sec & 0x7FFFFFFF);
}

/**
 * run_watch_or_tui() - Run continuous file/SHM watch mode or lightweight single-key TUI loop
 * @bench_path:   Resolved filesystem path to the target .bench configuration file (non-NULL).
 * @exe_dir:      Directory containing the running executable for locating cacaosim-bench.
 * @opts:         In/out rendering options updated on theme/table toggles (non-NULL).
 * @interactive:  0 for passive --watch polling mode; non-zero for raw-terminal hotkey mode.
 * @with_atmturb: Initial state of the --atmturb turbulence generator flag (0 or 1).
 * @seeing:       Atmospheric seeing FWHM at 500 nm [arcsec] forwarded to start actions.
 *
 * Role: Drives the alternate-screen event loop used by -w / --watch continuous monitoring mode
 * (and serves as a self-contained single-key interactive loop if invoked with @interactive != 0).
 * Every 150 ms it checks both the .bench file modification token (file_mtime_token()) and the
 * MILK_SHM_DIR directory modification stamp (count_shm_files()), re-rendering the schematic only
 * when an edit, SHM stream creation/removal, or user hotkey requests a redraw.
 *
 * Rationale: When tuning optical bench parameters in an external editor (vim, VS Code, emacs) in
 * one terminal pane while running simulations in another, operators want an auto-updating side
 * pane that instantly validates edits and reflects live SHM stream state without stealing focus or
 * polluting scrollback. Combining O(1) stat() change tokens with a 150 ms poll/nanosleep cadence
 * keeps idle CPU usage near zero while delivering sub-200 ms visual feedback on file save or
 * bench start/stop events. Suppressing opts->show_footer avoids redundant CLI hint lines beneath
 * the live status bar, and pre-validating the .bench file before invoking cacaosim-bench start
 * prevents tearing down a working simulation due to a transient syntax error.
 */
static void run_watch_or_tui(
    const char           *bench_path,
    const char           *exe_dir,
    cacaosim_view_opts_t *opts,
    int                   interactive,
    int                   with_atmturb,
    float                 seeing)
{
    /* Install async-signal-safe SIGINT/SIGTERM handler so tui_exit() always runs cleanly */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sig;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    /* Replace the static CLI footer with the live watch/hotkey status line */
    opts->show_footer = 0;
    if (opts->color_level < 0 && isatty(STDOUT_FILENO))
    {
        const char *ct = getenv("COLORTERM");
        opts->color_level = (ct && (strstr(ct, "truecolor") || strstr(ct, "24bit"))) ? 3 : 2;
    }

    char     status_msg[320] = { 0 };
    uint64_t last_mtime      = 0;
    int      last_shm_stamp  = -1;
    int      need_redraw     = 1;

    snprintf(status_msg, sizeof(status_msg),
             interactive ? "Ready — press 'e' to edit %.180s, 'f' to deploy & run, 'q' to quit"
                         : "Watching %.180s for edits (save file to update live, Ctrl-C to quit)",
             bench_path);

    tui_enter(interactive);

    while (!g_stop_req)
    {
        /* Poll O(1) stat() tokens on both the .bench file and MILK_SHM_DIR */
        uint64_t mt        = file_mtime_token(bench_path);
        int      shm_stamp = count_shm_files();
        if (mt != last_mtime || shm_stamp != last_shm_stamp)
        {
            if (last_mtime != 0 && mt != last_mtime)
            {
                time_t     now = time(NULL);
                struct tm *tmv = localtime(&now);
                if (tmv != NULL)
                {
                    snprintf(status_msg, sizeof(status_msg),
                             "Reloaded '%.200s' at %02d:%02d:%02d", bench_path, tmv->tm_hour,
                             tmv->tm_min, tmv->tm_sec);
                }
            }
            last_mtime     = mt;
            last_shm_stamp = shm_stamp;
            need_redraw    = 1;
        }

        if (need_redraw)
        {
            /* Home cursor and clear alternate screen before emitting fresh schematic frame */
            fputs("\033[H\033[2J", stdout);
            cacaosim_bench_render_file(bench_path, opts, stdout);

            if (opts->color_level > 0)
            {
                if (interactive)
                {
                    printf("  \033[1;38;2;130;170;255m[e]\033[0m Edit  "
                           "\033[1;38;2;80;220;80m[f]\033[0m Deploy Fused  "
                           "\033[1;38;2;80;200;220m[m]\033[0m Deploy Modular  "
                           "\033[1;38;2;255;215;80m[a]\033[0m AtmTurb:%s  "
                           "\033[1;38;2;240;60;60m[s]\033[0m Stop  "
                           "\033[1;38;2;180;140;255m[t]\033[0m Theme:%s  "
                           "\033[1;38;2;130;170;255m[b]\033[0m Tables  "
                           "\033[1;38;2;200;205;215m[q]\033[0m Quit\n",
                           with_atmturb ? "\033[1;32mON\033[0m" : "OFF",
                           cacaosim_view_theme_name(opts->theme));
                }
                printf("  \033[38;2;135;142;160mStatus:\033[0m %s\n", status_msg);
            }
            else
            {
                if (interactive)
                {
                    printf("  [e] Edit  [f] Deploy Fused  [m] Deploy Modular  [a] AtmTurb:%s  "
                           "[s] Stop  [t] Theme:%s  [b] Tables  [q] Quit\n",
                           with_atmturb ? "ON" : "OFF",
                           cacaosim_view_theme_name(opts->theme));
                }
                printf("  Status: %s\n", status_msg);
            }
            fflush(stdout);
            need_redraw = 0;
        }

        /* Wait up to 150 ms for keystrokes (or sleep 150 ms in passive --watch mode) */
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
        int           pr  = poll(&pfd, interactive ? 1 : 0, 150);
        if (pr <= 0 && !interactive)
        {
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 150000000L };
            nanosleep(&ts, NULL);
            continue;
        }
        if (interactive && pr > 0 && (pfd.revents & POLLIN))
        {
            unsigned char ch = 0;
            if (read(STDIN_FILENO, &ch, 1) == 1)
            {
                /* 'q', 'Q', or Ctrl-C (ETX = 3) exits cleanly */
                if (ch == 'q' || ch == 'Q' || ch == 3)
                {
                    break;
                }
                if (ch == 27)
                {
                    /* Drain any multi-byte CSI escape sequence; bare ESC quits */
                    unsigned char seq[4];
                    if (read(STDIN_FILENO, seq, sizeof(seq)) <= 0)
                    {
                        break;
                    }
                    continue;
                }
                if (ch == 't' || ch == 'T')
                {
                    opts->theme = (opts->theme + 1) % CACAOSIM_THEME_COUNT;
                    snprintf(status_msg, sizeof(status_msg), "Switched theme to '%s'",
                             cacaosim_view_theme_name(opts->theme));
                    need_redraw = 1;
                }
                else if (ch == 'b' || ch == 'B')
                {
                    opts->show_tables = !opts->show_tables;
                    need_redraw       = 1;
                }
                else if (ch == 'a' || ch == 'A')
                {
                    with_atmturb = !with_atmturb;
                    snprintf(status_msg, sizeof(status_msg),
                             "Atmospheric turbulence (--atmturb) %s for next deploy",
                             with_atmturb ? "ENABLED" : "DISABLED");
                    need_redraw = 1;
                }
                else if (ch == 'r' || ch == 'R')
                {
                    snprintf(status_msg, sizeof(status_msg), "Refreshed view and SHM status");
                    need_redraw = 1;
                }
                else if (ch == 'e' || ch == 'E')
                {
                    /* Temporarily restore cooked TTY so $EDITOR owns the terminal cleanly */
                    tui_exit(1);
                    launch_editor(bench_path);
                    tui_enter(1);
                    last_mtime  = file_mtime_token(bench_path);
                    need_redraw = 1;
                    snprintf(status_msg, sizeof(status_msg),
                             "Returned from editor — updated '%.200s'", bench_path);
                }
                else if (ch == 'f' || ch == 'F' || ch == 'm' || ch == 'M')
                {
                    /* Validate syntax & optical graph before invoking cacaosim-bench start */
                    const char          *dmode = (ch == 'm' || ch == 'M') ? "modular" : "fused";
                    cacaosim_bench_cfg_t tcfg;
                    cacaosim_bench_ctx_t tctx;
                    char                 ebuf[256] = { 0 };
                    if (cacaosim_bench_load_file(bench_path, &tcfg, ebuf, sizeof(ebuf)) != 0 ||
                        cacaosim_bench_analyze(&tctx, &tcfg, ebuf, sizeof(ebuf)) != 0)
                    {
                        snprintf(status_msg, sizeof(status_msg),
                                 "Cannot deploy invalid config: %.220s", ebuf);
                    }
                    else
                    {
                        snprintf(status_msg, sizeof(status_msg), "Deploying '%.180s' (%s)...",
                                 bench_path, dmode);
                        fputs("\033[H\033[2J", stdout);
                        cacaosim_bench_render(&tctx, bench_path, opts, stdout);
                        printf("\n  Deploying %s mode...\n", dmode);
                        fflush(stdout);
                        run_bench_action(exe_dir, "start", bench_path, dmode, with_atmturb, seeing,
                                         status_msg, sizeof(status_msg));
                        /* Brief settle window so newly spawned SHM streams appear LIVE on redraw */
                        struct timespec ts = { .tv_sec = 0, .tv_nsec = 150000000L };
                        nanosleep(&ts, NULL);
                    }
                    last_shm_stamp = count_shm_files();
                    need_redraw    = 1;
                }
                else if (ch == 's' || ch == 'S')
                {
                    run_bench_action(exe_dir, "stop", bench_path, "fused", 0, seeing, status_msg,
                                     sizeof(status_msg));
                    last_shm_stamp = count_shm_files();
                    need_redraw    = 1;
                }
            }
        }
    }

    tui_exit(interactive);
}

/**
 * main() - CLI entry point for the cacaosim-bench-view optical bench schematic visualizer
 * @argc: Command-line argument count.
 * @argv: Command-line argument vector.
 *
 * Role: Initializes UTF-8 locale handling and standard milk CLI help hooks (-h1, -h2, -h, --help,
 * --help-mono), parses rendering and workflow flags, optionally writes a starter .bench template
 * (--init), resolves preset shorthand names via resolve_bench_file(), and dispatches to one of
 * four execution paths:
 *   1. Full-screen interactive configurator (-i / --interactive / -g / --gui) via
 *      cacaosim_tui_run(), supporting live MILK_SHM_DIR auto-discovery when no .bench argument is
 *      supplied.
 *   2. Continuous file & SHM watch loop (-w / --watch) via run_watch_or_tui().
 *   3. Pre-validated simulation launch (-d / --deploy / --run) followed by a 150 ms SHM settle
 *      pause and a live-annotated schematic snapshot.
 *   4. Default one-shot stream-oriented schematic rendering to stdout via
 *      cacaosim_bench_render_file().
 *
 * Rationale: Consolidating quick scrollback inspection, starter template scaffolding, continuous
 * watch reloads, one-command deploy-and-verify, and seamless escalation into cacaosim-CTRL behind
 * a single fast startup binary gives AO engineers a frictionless workflow across interactive
 * workstations, remote SSH sessions, cacaosim-bench shell wrappers, and automated CI logs. Calling
 * setlocale(LC_CTYPE, "") upfront ensures standard C/POSIX character-classification routines align
 * with the terminal's UTF-8 environment, while milk_help_init() guarantees uniform flag discovery
 * across the entire milk ecosystem.
 *
 * Return: 0 on successful help display, render, watch session, TUI exit, or deployment; 1 on
 * invalid CLI flags, missing target file, parse/topology validation failure, or deploy error.
 */
int main(int argc, char **argv)
{
    /* Enable environment UTF-8 locale and handle standard milk help queries (-h1/-h2/-h) */
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

    cacaosim_view_opts_t opts;
    cacaosim_view_opts_defaults(&opts);

    const char *bench_arg    = NULL;
    const char *init_path    = NULL;
    const char *deploy_mode  = "fused";
    int         watch_mode   = 0;
    int         inter_mode   = 0;
    int         do_deploy    = 0;
    int         with_atmturb = 0;
    float       seeing       = 0.7f;

    /* Parse CLI options; first non-option positional token is treated as the .bench target */
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        if (strcmp(a, "-p") == 0 || strcmp(a, "--pretty") == 0 || strcmp(a, "--color") == 0)
        {
            opts.color_level = 3;
        }
        else if (strcmp(a, "-t") == 0 || strcmp(a, "--text") == 0 || strcmp(a, "--no-color") == 0)
        {
            opts.color_level = 0;
        }
        else if (strcmp(a, "--ascii") == 0)
        {
            opts.ascii_only = 1;
        }
        else if (strcmp(a, "--compact") == 0)
        {
            opts.show_tables = 0;
        }
        else if (strcmp(a, "--no-shm") == 0)
        {
            opts.check_shm = 0;
        }
        else if (strcmp(a, "--no-footer") == 0)
        {
            opts.show_footer = 0;
        }
        else if (strcmp(a, "-w") == 0 || strcmp(a, "--watch") == 0)
        {
            watch_mode = 1;
        }
        else if (strcmp(a, "-i") == 0 || strcmp(a, "--interactive") == 0 ||
                 strcmp(a, "-g") == 0 || strcmp(a, "--gui") == 0)
        {
            inter_mode = 1;
        }
        else if (strcmp(a, "-d") == 0 || strcmp(a, "--deploy") == 0 || strcmp(a, "--run") == 0)
        {
            do_deploy = 1;
        }
        else if (strcmp(a, "--atmturb") == 0)
        {
            with_atmturb = 1;
        }
        else if ((strcmp(a, "--theme") == 0) && i + 1 < argc)
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
        else if ((strcmp(a, "-W") == 0 || strcmp(a, "--width") == 0) && i + 1 < argc)
        {
            opts.width = atoi(argv[++i]);
        }
        else if ((strcmp(a, "--config") == 0 || strcmp(a, "-c") == 0) && i + 1 < argc)
        {
            bench_arg = argv[++i];
        }
        else if (strcmp(a, "--init") == 0 && i + 1 < argc)
        {
            init_path = argv[++i];
        }
        else if (strcmp(a, "--mode") == 0 && i + 1 < argc)
        {
            deploy_mode = argv[++i];
        }
        else if (strcmp(a, "--seeing") == 0 && i + 1 < argc)
        {
            seeing = strtof(argv[++i], NULL);
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

    /* Write starter .bench file if requested, and immediately render/open it unless overridden */
    if (init_path != NULL)
    {
        if (write_starter_template(init_path) != 0)
        {
            return 1;
        }
        printf("[cacaosim-bench-view] Created starter configuration '%s'\n", init_path);
        if (bench_arg == NULL)
        {
            bench_arg = init_path;
        }
    }

    char exe_dir[PATH_MAX];
    char resolved[PATH_MAX];
    get_exe_dir(argv[0], exe_dir, sizeof(exe_dir));

    /* Hand off -i / --gui to cacaosim-CTRL; NULL bench_arg triggers live MILK_SHM_DIR discovery */
    if (inter_mode)
    {
        cacaosim_tui_opts_t topts;
        cacaosim_tui_opts_defaults(&topts);
        topts.color_level   = opts.color_level;
        topts.theme         = opts.theme;
        topts.ascii_only    = opts.ascii_only;
        topts.with_atmturb  = with_atmturb;
        topts.seeing_arcsec = seeing;
        topts.deploy_mode   = (strcmp(deploy_mode, "modular") == 0) ? 1 : 0;
        snprintf(topts.exe_dir, sizeof(topts.exe_dir), "%.511s", exe_dir);
        if (bench_arg != NULL)
        {
            resolve_bench_file(bench_arg, exe_dir, resolved, sizeof(resolved));
            return cacaosim_tui_run(resolved, &topts);
        }
        return cacaosim_tui_run(NULL, &topts);
    }

    /* Non-interactive modes require a target .bench path or preset name */
    if (bench_arg == NULL)
    {
        print_help_full(argv[0], isatty(STDOUT_FILENO));
        return 1;
    }

    resolve_bench_file(bench_arg, exe_dir, resolved, sizeof(resolved));

    if (watch_mode)
    {
        run_watch_or_tui(resolved, exe_dir, &opts, 0, with_atmturb, seeing);
        return 0;
    }

    /* Pre-validate syntax & topology, deploy via cacaosim-bench start, wait 150 ms, then render */
    if (do_deploy)
    {
        cacaosim_bench_cfg_t cfg;
        cacaosim_bench_ctx_t ctx;
        char                 err_buf[256] = { 0 };
        if (cacaosim_bench_load_file(resolved, &cfg, err_buf, sizeof(err_buf)) != 0)
        {
            cacaosim_bench_render_error(resolved, NULL, err_buf, &opts, stdout);
            return 1;
        }
        if (cacaosim_bench_analyze(&ctx, &cfg, err_buf, sizeof(err_buf)) != 0)
        {
            cacaosim_bench_render_error(resolved, &cfg, err_buf, &opts, stdout);
            return 1;
        }
        char status_msg[320] = { 0 };
        int  rc = run_bench_action(exe_dir, "start", resolved, deploy_mode, with_atmturb, seeing,
                                   status_msg, sizeof(status_msg));
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 150000000L };
        nanosleep(&ts, NULL);
        cacaosim_bench_render_file(resolved, &opts, stdout);
        printf("  %s\n", status_msg);
        return (rc == 0) ? 0 : 1;
    }

    /* Default path: one-shot stream-oriented schematic render to stdout */
    return (cacaosim_bench_render_file(resolved, &opts, stdout) == 0) ? 0 : 1;
}
