// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_view_main.c
 * @brief   Standalone CLI & interactive TUI visualizer/launcher for .bench AO configurations
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

static struct termios        g_orig_termios;
static volatile sig_atomic_t g_raw_active = 0;
static volatile sig_atomic_t g_stop_req   = 0;

static const char g_tui_enter_seq[] = "\033[?1049h\033[?25l";
static const char g_tui_exit_seq[]  = "\033[?25h\033[0m\033[?1049l";

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

static const char g_desc_short[] =
    "Graphical ANSI/UTF-8 optical bench (.bench) visualizer and launcher";

static const char g_desc_long[] =
    "Parses a declarative cacaosim .bench file, validates the multi-stage optical DAG,\n"
    "propagates 32-bin (0.40..2.40 um) stellar photometry across all dichroic/coronagraph\n"
    "arms, and renders a milk-styled TrueColor/256/mono optical schematic with 2D pupil\n"
    "preview, spectral sparklines, live MILK_SHM_DIR stream status, and interactive\n"
    "edit/deploy/stop TUI controls.";

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
                   "Live TUI: [e] edit in $EDITOR, [f/m] deploy & run, [s] stop, [t] theme");
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

static uint64_t file_mtime_token(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
    {
        return 0;
    }
    return ((uint64_t) st.st_mtim.tv_sec * 1000000000ULL) + (uint64_t) st.st_mtim.tv_nsec;
}

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

static void run_watch_or_tui(
    const char           *bench_path,
    const char           *exe_dir,
    cacaosim_view_opts_t *opts,
    int                   interactive,
    int                   with_atmturb,
    float                 seeing)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sig;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

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
                if (ch == 'q' || ch == 'Q' || ch == 3)
                {
                    break;
                }
                if (ch == 27)
                {
                    /* Drain any escape sequence bytes; bare ESC quits */
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
        else if (strcmp(a, "-i") == 0 || strcmp(a, "--interactive") == 0)
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

    if (bench_arg == NULL)
    {
        print_help_full(argv[0], isatty(STDOUT_FILENO));
        return 1;
    }

    char exe_dir[PATH_MAX];
    char resolved[PATH_MAX];
    get_exe_dir(argv[0], exe_dir, sizeof(exe_dir));
    resolve_bench_file(bench_arg, exe_dir, resolved, sizeof(resolved));

    if (inter_mode || watch_mode)
    {
        run_watch_or_tui(resolved, exe_dir, &opts, inter_mode, with_atmturb, seeing);
        return 0;
    }

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

    return (cacaosim_bench_render_file(resolved, &opts, stdout) == 0) ? 0 : 1;
}
