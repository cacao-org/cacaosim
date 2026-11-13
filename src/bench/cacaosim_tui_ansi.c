// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_ansi.c
 * @brief   milk-CTRL-grade raw terminal, SGR mouse/key parser, and delta-flushed cell buffer
 *
 * Architectural Role in cacaosim-CTRL:
 *   Implements the complete zero-`ncurses` terminal I/O and 2D cell-canvas foundation for
 *   `cacaosim-CTRL`:
 *   1. Semantic theme palettes (`g_tui_palettes`) and physical-wavelength colormaps
 *      (`cs_tui_wavelength_rgb()`) across `0.40..2.40 um`.
 *   2. Raw POSIX `termios` alternate-screen entry/exit, non-blocking `stdout`/`stderr` capture
 *      pipe redirection, and async-signal-safe crash/interrupt terminal restoration.
 *   3. Non-blocking VT100/xterm CSI, SS3, bracketed-paste (`DECSET 2004`), and SGR 1006 extended
 *      mouse (`DECSET 1006`) escape-sequence decoding (`cs_tui_poll_input()`).
 *   4. Double-buffered 2D UTF-8 / half-block / sparkline cell canvas (`cs_screen_t`), perceptual
 *      color quantization (24-bit -> 256-color -> 16-color -> monochrome), and synchronized-output
 *      (`DECSET 2026`) cell-diffing presentation (`cs_scr_flush()`).
 *
 * Design Rationale & Motivation:
 *   Building a self-contained double-buffered ANSI cell presenter instead of linking `ncurses`:
 *   - Unlocks full 24-bit TrueColor SGR (`\033[38;2;R;G;Bm` / `48;2;R;G;Bm`) for continuous optical
 *     passband visualization and scotopic Observatory Red themes without `terminfo` pair limits.
 *   - Stores arbitrary single-width UTF-8 codepoints (`char ch[5]`) directly per cell — supporting
 *     rounded/double Unicode borders, `▀`/`▄` 2x1 vertical subpixel pupil previews, `▁`..`█`
 *     spectral sparklines, and Unicode Braille (`U+2800..U+28FF`) bitmaps without `ncursesw` locale
 *     fragility.
 *   - Eliminates external UI library dependencies (`libncurses`, `libtinfo`) and enables pure
 *     in-memory plain-text snapshot rendering (`cs_scr_dump_plain()`) for deterministic CI tests.
 *   - Prevents tearing, cursor flicker, and stray library `printf` corruption during continuous
 *     10-30 Hz live `ImageStreamIO` SHM telemetry refreshes via `DECSET 2026` atomic frames and
 *     dedicated TTY fd duplication.
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "bench/cacaosim_tui_internal.h"

/* ── Exact milk-CTRL Theme Palettes ───────────────────────────────────────── */

/**
 * @brief Built-in semantic RGB theme palettes indexed by `cacaosim_view_theme_t`.
 *
 * Mirrors the exact color tokens of `milk-CTRL` across five operational environments:
 *   - `CACAOSIM_THEME_DARK`: Default high-contrast dark slate / cyan / blue / gold control room.
 *   - `CACAOSIM_THEME_NIGHT`: Long-wavelength red/amber palette preserving scotopic night vision.
 *   - `CACAOSIM_THEME_ACCESSIBLE`: Okabe-Ito deuteranopia/protanopia-safe high-contrast palette.
 *   - `CACAOSIM_THEME_LIGHT`: Daylight / paper-white palette for bright terminal backgrounds.
 *   - `CACAOSIM_THEME_NORDIC`: Low-glare Nord frost and aurora palette for extended sessions.
 */
static const cs_tui_pal_t g_tui_palettes[CACAOSIM_THEME_COUNT] = {
    /* CACAOSIM_THEME_DARK (default milk-CTRL dark slate) */
    {
        .name            = "dark",
        .label           = "Dark Slate (milk default)",
        .bg_bar          = { 22, 24, 32 },
        .bg_panel        = { 30, 32, 40 },
        .bg_panel_alt    = { 35, 38, 48 },
        .bg_selected     = { 48, 64, 96 },
        .bg_hover        = { 40, 46, 62 },
        .bg_popup        = { 34, 38, 50 },
        .bg_input        = { 18, 22, 30 },
        .fg_border       = { 72, 80, 102 },
        .fg_border_focus = { 130, 170, 255 },
        .fg_title        = { 130, 170, 255 },
        .fg_text         = { 200, 205, 215 },
        .fg_bright       = { 240, 245, 255 },
        .fg_dim          = { 100, 105, 120 },
        .fg_muted        = { 135, 142, 160 },
        .fg_stream       = { 80, 200, 220 },
        .fg_fps          = { 130, 170, 255 },
        .fg_proc         = { 180, 140, 255 },
        .fg_active       = { 80, 220, 80 },
        .fg_warn         = { 255, 180, 0 },
        .fg_error        = { 240, 60, 60 },
        .fg_accent       = { 255, 215, 80 },
        .fg_conn         = { 100, 130, 180 },
    },
    /* CACAOSIM_THEME_NIGHT (Observatory Red — night-vision safe) */
    {
        .name            = "night",
        .label           = "Observatory Red (night-vision)",
        .bg_bar          = { 18, 5, 5 },
        .bg_panel        = { 24, 8, 8 },
        .bg_panel_alt    = { 32, 10, 10 },
        .bg_selected     = { 68, 18, 16 },
        .bg_hover        = { 46, 14, 14 },
        .bg_popup        = { 34, 10, 10 },
        .bg_input        = { 16, 4, 4 },
        .fg_border       = { 95, 30, 24 },
        .fg_border_focus = { 255, 95, 60 },
        .fg_title        = { 255, 90, 60 },
        .fg_text         = { 210, 65, 45 },
        .fg_bright       = { 255, 125, 90 },
        .fg_dim          = { 115, 35, 28 },
        .fg_muted        = { 155, 48, 36 },
        .fg_stream       = { 245, 95, 65 },
        .fg_fps          = { 255, 80, 50 },
        .fg_proc         = { 225, 65, 70 },
        .fg_active       = { 255, 115, 60 },
        .fg_warn         = { 255, 95, 40 },
        .fg_error        = { 255, 45, 45 },
        .fg_accent       = { 255, 130, 70 },
        .fg_conn         = { 145, 42, 32 },
    },
    /* CACAOSIM_THEME_ACCESSIBLE (Okabe-Ito colorblind-safe) */
    {
        .name            = "accessible",
        .label           = "Okabe-Ito (colorblind-safe)",
        .bg_bar          = { 18, 20, 26 },
        .bg_panel        = { 26, 28, 34 },
        .bg_panel_alt    = { 32, 35, 44 },
        .bg_selected     = { 16, 68, 104 },
        .bg_hover        = { 38, 44, 56 },
        .bg_popup        = { 32, 36, 46 },
        .bg_input        = { 16, 18, 24 },
        .fg_border       = { 82, 90, 106 },
        .fg_border_focus = { 86, 180, 233 },
        .fg_title        = { 86, 180, 233 },
        .fg_text         = { 215, 220, 228 },
        .fg_bright       = { 250, 252, 255 },
        .fg_dim          = { 110, 116, 128 },
        .fg_muted        = { 150, 158, 172 },
        .fg_stream       = { 86, 180, 233 },
        .fg_fps          = { 0, 114, 178 },
        .fg_proc         = { 204, 121, 167 },
        .fg_active       = { 0, 158, 115 },
        .fg_warn         = { 230, 159, 0 },
        .fg_error        = { 213, 94, 0 },
        .fg_accent       = { 240, 228, 66 },
        .fg_conn         = { 115, 145, 175 },
    },
    /* CACAOSIM_THEME_LIGHT (Paper Light) */
    {
        .name            = "light",
        .label           = "Paper Light (daylight)",
        .bg_bar          = { 210, 218, 230 },
        .bg_panel        = { 236, 239, 244 },
        .bg_panel_alt    = { 226, 231, 240 },
        .bg_selected     = { 188, 210, 244 },
        .bg_hover        = { 216, 224, 238 },
        .bg_popup        = { 244, 246, 250 },
        .bg_input        = { 252, 253, 255 },
        .fg_border       = { 150, 160, 178 },
        .fg_border_focus = { 25, 75, 175 },
        .fg_title        = { 25, 75, 175 },
        .fg_text         = { 35, 40, 50 },
        .fg_bright       = { 10, 15, 25 },
        .fg_dim          = { 130, 136, 150 },
        .fg_muted        = { 95, 102, 118 },
        .fg_stream       = { 0, 118, 142 },
        .fg_fps          = { 30, 85, 185 },
        .fg_proc         = { 125, 55, 175 },
        .fg_active       = { 20, 135, 55 },
        .fg_warn         = { 185, 105, 0 },
        .fg_error        = { 195, 30, 35 },
        .fg_accent       = { 165, 95, 0 },
        .fg_conn         = { 105, 125, 160 },
    },
    /* CACAOSIM_THEME_NORDIC (Nord frost & aurora) */
    {
        .name            = "nordic",
        .label           = "Nordic Frost & Aurora",
        .bg_bar          = { 36, 41, 51 },
        .bg_panel        = { 46, 52, 64 },
        .bg_panel_alt    = { 53, 60, 74 },
        .bg_selected     = { 67, 76, 94 },
        .bg_hover        = { 59, 66, 82 },
        .bg_popup        = { 52, 59, 73 },
        .bg_input        = { 34, 39, 49 },
        .fg_border       = { 76, 86, 106 },
        .fg_border_focus = { 136, 192, 208 },
        .fg_title        = { 136, 192, 208 },
        .fg_text         = { 216, 222, 233 },
        .fg_bright       = { 236, 239, 244 },
        .fg_dim          = { 108, 118, 138 },
        .fg_muted        = { 144, 156, 178 },
        .fg_stream       = { 143, 188, 187 },
        .fg_fps          = { 129, 161, 193 },
        .fg_proc         = { 180, 142, 173 },
        .fg_active       = { 163, 190, 140 },
        .fg_warn         = { 235, 203, 139 },
        .fg_error        = { 191, 97, 106 },
        .fg_accent       = { 235, 203, 139 },
        .fg_conn         = { 94, 129, 172 },
    },
};

/**
 * cs_tui_now_sec() - Read monotonic system clock in floating-point seconds
 *
 * Role:
 *   Provides a jump-free monotonic timebase for SHM polling intervals, `cnt0`-delta Hz telemetry,
 *   toast banner timeouts, and 1 s terminal output bandwidth (`tx_kbps`) windows.
 *
 * Rationale:
 *   `CLOCK_MONOTONIC` is immune to wall-clock NTP adjustments, preventing stalled UI refreshes or
 *   negative `dt` spikes during observatory time synchronization.
 *
 * Return: Monotonic elapsed time in seconds (`double`).
 */
double cs_tui_now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

/**
 * cs_tui_get_palette() - Look up the static semantic RGB palette for @theme
 * @theme: Requested `cacaosim_view_theme_t` index (out-of-bounds defaults to Dark Slate)
 *
 * Role:
 *   Maps integer theme selections to their corresponding `cs_tui_pal_t` color table.
 *
 * Rationale:
 *   Clamping invalid indices to `CACAOSIM_THEME_DARK` guarantees callers always receive a valid
 *   non-NULL palette pointer without repetitive bounds checks at every widget call site.
 *
 * Return: Pointer to immutable `cs_tui_pal_t` entry in `g_tui_palettes`.
 */
const cs_tui_pal_t *cs_tui_get_palette(int theme)
{
    if (theme < 0 || theme >= CACAOSIM_THEME_COUNT)
    {
        theme = CACAOSIM_THEME_DARK;
    }
    return &g_tui_palettes[theme];
}

/**
 * cs_tui_detect_color_level() - Determine SGR color capability of the current terminal
 * @requested: Explicit override (`0..3`), or `< 0` for automatic environment inspection
 *
 * Role:
 *   Chooses between monochrome (`0`), 16-color ANSI (`1`), xterm 256-color (`2`), and 24-bit
 *   TrueColor (`3`) SGR emission modes.
 *
 * Rationale:
 *   Inspects standard environment variables (`NO_COLOR`, `TERM`, `COLORTERM`) in priority order so
 *   `cacaosim-CTRL` automatically exploits 24-bit TrueColor on modern terminals while degrading
 *   cleanly over legacy SSH `TERM=xterm-256color`, Linux console, or `NO_COLOR=1` sessions without
 *   linking `libtinfo`.
 *
 * Return: Active color level in `[0..3]`.
 */
int cs_tui_detect_color_level(int requested)
{
    if (requested >= 0)
    {
        return (requested > 3) ? 3 : requested;
    }
    /* Respect https://no-color.org user accessibility preference unconditionally */
    const char *nc = getenv("NO_COLOR");
    if (nc != NULL && nc[0] != '\0')
    {
        return 0;
    }
    const char *term = getenv("TERM");
    if (term == NULL || term[0] == '\0' || strcmp(term, "dumb") == 0)
    {
        return 0;
    }
    const char *ct = getenv("COLORTERM");
    if (ct != NULL && (strstr(ct, "truecolor") != NULL || strstr(ct, "24bit") != NULL))
    {
        return 3;
    }
    if (strstr(term, "256color") != NULL)
    {
        return 2;
    }
    return 1;
}

/**
 * cs_tui_detect_ascii() - Decide whether to restrict rendering to pure 7-bit ASCII glyphs
 * @requested: Non-zero if `--ascii` was explicitly requested by the caller, else 0
 *
 * Role:
 *   Selects between UTF-8 box-drawing/block symbols (`0`) and 7-bit ASCII fallbacks (`1`).
 *
 * Rationale:
 *   Checks POSIX locale precedence (`LC_ALL` -> `LC_CTYPE` -> `LANG`) and enables 7-bit ASCII
 *   fallback automatically under `"C"` or `"POSIX"` locales where multi-byte UTF-8 sequences would
 *   render as replacement characters or misalign column widths.
 *
 * Return: 1 if pure 7-bit ASCII mode is active, 0 if UTF-8 symbols are enabled.
 */
int cs_tui_detect_ascii(int requested)
{
    if (requested)
    {
        return 1;
    }
    const char *lc = getenv("LC_ALL");
    if (lc == NULL || lc[0] == '\0')
    {
        lc = getenv("LC_CTYPE");
    }
    if (lc == NULL || lc[0] == '\0')
    {
        lc = getenv("LANG");
    }
    if (lc != NULL && (strcmp(lc, "C") == 0 || strcmp(lc, "POSIX") == 0))
    {
        return 1;
    }
    return 0;
}

/**
 * cs_tui_wavelength_rgb() - Convert an optical wavelength `[um]` into an RGB display color
 * @lam_um: Wavelength in micrometers across the `cacaosim` `0.40..2.40 um` spectral grid
 * @theme:  Active `cacaosim_view_theme_t` index
 *
 * Role:
 *   Assigns intuitive spectral colors to 32-bin beam sparklines and passband indicators across UV/B
 *   (`< 0.50 um`), V (`< 0.62 um`), R/I (`< 0.78 um`), z/Y (`< 1.05 um`), J (`< 1.45 um`),
 *   H (`< 1.90 um`), and K (`>= 1.90 um`) astronomical bands.
 *
 * Rationale:
 *   Astronomical AO benches span visible (`0.4..0.9 um`) and near-IR (`1.0..2.4 um`) bands where
 *   true human cone response is black (`> 0.75 um`); mapping NIR bands to warm-orange -> crimson
 *   -> violet gives distinct band identification while `CACAOSIM_THEME_NIGHT` overrides all bands
 *   with a monotonic dim-red ramp to protect dark-adapted vision.
 *
 * Return: 24-bit `cs_tui_rgb_t` color corresponding to @lam_um under @theme.
 */
cs_tui_rgb_t cs_tui_wavelength_rgb(float lam_um, int theme)
{
    if (theme == CACAOSIM_THEME_NIGHT)
    {
        /* Scotopic-safe red-amber gradient keeping blue channel minimal across 0.40..2.40 um */
        float    u = (lam_um - 0.40f) / 2.00f;
        uint8_t  r = (uint8_t) (145.0f + 110.0f * (1.0f - 0.35f * u));
        uint8_t  g = (uint8_t) (32.0f + 55.0f * (1.0f - u));
        cs_tui_rgb_t c = { r, g, 26 };
        return c;
    }
    if (lam_um < 0.50f)
    {
        cs_tui_rgb_t c = { 95, 150, 255 };
        return c;
    }
    if (lam_um < 0.62f)
    {
        cs_tui_rgb_t c = { 70, 220, 175 };
        return c;
    }
    if (lam_um < 0.78f)
    {
        cs_tui_rgb_t c = { 245, 205, 75 };
        return c;
    }
    if (lam_um < 1.05f)
    {
        cs_tui_rgb_t c = { 255, 135, 65 };
        return c;
    }
    if (lam_um < 1.45f)
    {
        cs_tui_rgb_t c = { 240, 80, 95 };
        return c;
    }
    if (lam_um < 1.90f)
    {
        cs_tui_rgb_t c = { 205, 95, 225 };
        return c;
    }
    {
        cs_tui_rgb_t c = { 155, 110, 245 };
        return c;
    }
}

/* ── Raw Terminal, Stdio Capture Pipe & Async-Signal-Safe Cleanup ─────────── */

/** Saved canonical `termios` snapshot restored on exit or fatal signal. */
static struct termios        g_saved_tio;
/** Non-zero once `g_saved_tio` holds a valid `tcgetattr(STDIN_FILENO)` snapshot. */
static int                   g_has_saved_tio = 0;
/** Duplicated controlling TTY output descriptor bypassing redirected `STDOUT_FILENO`. */
static int                   g_tty_out_fd    = -1;
/** Duplicated original `STDOUT_FILENO` saved while stdio capture pipe is active. */
static int                   g_orig_stdout   = -1;
/** Duplicated original `STDERR_FILENO` saved while stdio capture pipe is active. */
static int                   g_orig_stderr   = -1;
/** Non-blocking `[read_fd, write_fd]` pipe capturing stray library `stdout`/`stderr` writes. */
static int                   g_cap_pipe[2]   = { -1, -1 };
/** Async-signal-safe flag (`1` while raw alternate-screen terminal mode is active). */
static volatile sig_atomic_t g_term_active   = 0;
/** Async-signal-safe termination request flag set on `SIGINT`, `SIGTERM`, or `SIGHUP`. */
static volatile sig_atomic_t g_sig_quit      = 0;
/** Async-signal-safe window-resize flag set on `SIGWINCH`. */
static volatile sig_atomic_t g_sig_winch     = 0;

/**
 * DEC private mode activation sequence:
 *   `?1049h` = save cursor + switch to alternate screen buffer
 *   `?25l`   = hide hardware text cursor
 *   `?1000h` = enable mouse button press/release reporting
 *   `?1002h` = enable button-event drag motion reporting
 *   `?1003h` = enable any-event passive hover motion reporting (for hover tooltips)
 *   `?1006h` = enable SGR extended decimal mouse coordinates (`> 223` columns safe)
 *   `?2004h` = enable bracketed paste mode (`\033[200~` ... `\033[201~`)
 */
static const char g_seq_enter[] =
    "\033[?1049h\033[?25l\033[?1000h\033[?1002h\033[?1003h\033[?1006h\033[?2004h";

/** Reverse DEC mode teardown sequence restoring normal cursor, SGR `0m`, and main screen. */
static const char g_seq_exit[] =
    "\033[?2004l\033[?1006l\033[?1003l\033[?1002l\033[?1000l\033[?25h\033[0m\033[?1049l";

/**
 * write_all_fd() - Async-signal-safe loop writing @len bytes to @fd across `EINTR`/short writes
 * @fd:  Open output file descriptor
 * @buf: Source byte buffer
 * @len: Number of bytes to transmit
 *
 * Role:
 *   Delivers complete escape sequences and synchronized frame buffers to the TTY descriptor.
 *
 * Rationale:
 *   Uses only raw POSIX `write()` (async-signal-safe) and retries on `EINTR` (`SIGWINCH` arriving
 *   mid-flush) so neither signal cleanup nor large frame flushes emit truncated CSI sequences.
 */
static void write_all_fd(int fd, const char *buf, size_t len)
{
    size_t off = 0u;
    while (off < len)
    {
        ssize_t nw = write(fd, buf + off, len - off);
        if (nw > 0)
        {
            off += (size_t) nw;
        }
        else if (nw < 0 && errno == EINTR)
        {
            continue;
        }
        else
        {
            break;
        }
    }
}

/**
 * tui_sig_handler() - Async-signal-safe handler for resize, termination, and crash signals
 * @sig: Delivered POSIX signal number
 *
 * Role:
 *   Records `SIGWINCH` for deferred main-loop resize, sets `g_sig_quit` on `SIGINT`/`SIGTERM`/
 *   `SIGHUP`, and immediately restores the user's terminal (`g_seq_exit` + `tcsetattr`) before
 *   re-raising fatal `SIGSEGV`/`SIGABRT` crashes.
 *
 * Rationale:
 *   Invokes exclusively async-signal-safe POSIX primitives (`write`, `tcsetattr`, `signal`,
 *   `raise`) so even an unexpected segfault or `abort()` cleanly exits alternate-screen and mouse
 *   tracking before the kernel generates a core dump.
 */
static void tui_sig_handler(int sig)
{
    if (sig == SIGWINCH)
    {
        g_sig_winch = 1;
        return;
    }
    g_sig_quit = 1;
    if (g_term_active)
    {
        int out_fd = (g_tty_out_fd >= 0) ? g_tty_out_fd : STDOUT_FILENO;
        write_all_fd(out_fd, g_seq_exit, sizeof(g_seq_exit) - 1u);
        if (g_has_saved_tio)
        {
            tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_saved_tio);
        }
        g_term_active = 0;
    }
    if (sig == SIGSEGV || sig == SIGABRT)
    {
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

/**
 * cs_tui_term_query_size() - Query active terminal dimensions with env and minimum fallbacks
 * @out_cols: Destination column count (`>= 40`, default `120`)
 * @out_rows: Destination row count (`>= 12`, default `38`)
 *
 * Role:
 *   Probes the real TTY (`g_tty_out_fd`) via `TIOCGWINSZ` on startup and after `SIGWINCH`.
 *
 * Rationale:
 *   Queries `g_tty_out_fd` rather than `STDOUT_FILENO` because `STDOUT_FILENO` is redirected to an
 *   anonymous pipe while the TUI is active, where `TIOCGWINSZ` would otherwise fail with `ENOTTY`.
 */
void cs_tui_term_query_size(int *out_cols, int *out_rows)
{
    int            fd = (g_tty_out_fd >= 0) ? g_tty_out_fd : STDOUT_FILENO;
    struct winsize ws;
    if (ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col >= 40 && ws.ws_row >= 12)
    {
        *out_cols = (int) ws.ws_col;
        *out_rows = (int) ws.ws_row;
        return;
    }
    const char *ec = getenv("COLUMNS");
    const char *er = getenv("LINES");
    *out_cols      = (ec != NULL && atoi(ec) >= 40) ? atoi(ec) : 120;
    *out_rows      = (er != NULL && atoi(er) >= 12) ? atoi(er) : 38;
}

/**
 * cs_tui_term_enter() - Initialize raw `termios`, stdio capture pipe, signals, and DEC TUI modes
 * @out_cols: Output initial terminal width in columns
 * @out_rows: Output initial terminal height in rows
 *
 * Role:
 *   Transitions the process from line-buffered CLI mode into full-screen interactive TUI mode and
 *   returns the dedicated TTY output file descriptor used by `cs_scr_flush()`.
 *
 * Rationale:
 *   Disables `ICANON`, `ECHO`, `ISIG`, `IXON`, `ICRNL`, and `OPOST` (`VMIN=0, VTIME=0`) so Ctrl
 *   shortcuts (`Ctrl+S` save, `Ctrl+Z` undo, `Ctrl+Q`/`Ctrl+C`) and raw CSI escape bytes arrive
 *   untranslated, while `dup2()` onto a non-blocking pipe shields the TUI canvas from stray writes
 *   to `stdout` or `stderr`.
 *
 * Return: Dedicated TTY file descriptor (`g_tty_out_fd >= 0`).
 */
int cs_tui_term_enter(int *out_cols, int *out_rows)
{
    if (g_term_active)
    {
        cs_tui_term_query_size(out_cols, out_rows);
        return g_tty_out_fd;
    }

    g_sig_quit  = 0;
    g_sig_winch = 0;

    if (isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO, &g_saved_tio) == 0)
    {
        g_has_saved_tio    = 1;
        struct termios raw = g_saved_tio;
        /* Disable break/CR-to-NL/parity/strip/XON-XOFF flow control so Ctrl+S/Ctrl+Q reach TUI */
        raw.c_iflag &= ~(unsigned int) (BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        /* Disable output post-processing; cursor positioning is managed explicitly via CSI */
        raw.c_oflag &= ~(unsigned int) (OPOST);
        raw.c_cflag |= (unsigned int) (CS8);
        /* Disable echo, canonical buffering, IEXTEN, and terminal Ctrl+C/Z signals */
        raw.c_lflag &= ~(unsigned int) (ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    }

    g_tty_out_fd = dup(STDOUT_FILENO);
    if (g_tty_out_fd < 0)
    {
        g_tty_out_fd = STDOUT_FILENO;
    }

    /* Redirect stdout/stderr into non-blocking capture pipe so library prints never tear TUI */
    if (pipe(g_cap_pipe) == 0)
    {
        int fl0 = fcntl(g_cap_pipe[0], F_GETFL, 0);
        int fl1 = fcntl(g_cap_pipe[1], F_GETFL, 0);
        if (fl0 >= 0)
        {
            fcntl(g_cap_pipe[0], F_SETFL, fl0 | O_NONBLOCK);
        }
        if (fl1 >= 0)
        {
            fcntl(g_cap_pipe[1], F_SETFL, fl1 | O_NONBLOCK);
        }
        fflush(stdout);
        fflush(stderr);
        g_orig_stdout = dup(STDOUT_FILENO);
        g_orig_stderr = dup(STDERR_FILENO);
        dup2(g_cap_pipe[1], STDOUT_FILENO);
        dup2(g_cap_pipe[1], STDERR_FILENO);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = tui_sig_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGWINCH, &sa, NULL);

    write_all_fd(g_tty_out_fd, g_seq_enter, sizeof(g_seq_enter) - 1u);
    g_term_active = 1;
    cs_tui_term_query_size(out_cols, out_rows);
    return g_tty_out_fd;
}

/**
 * cs_tui_term_exit() - Leave raw alternate-screen mode and restore original stdio and `termios`
 *
 * Role:
 *   Cleanly tears down all DEC private terminal modes, restores `STDOUT_FILENO`/`STDERR_FILENO`
 *   and `STDIN_FILENO` `termios`, and closes internal pipe/TTY file descriptors.
 *
 * Rationale:
 *   Guarded by `g_term_active` so repeated calls (e.g., temporary suspend around `$EDITOR` launch
 *   followed by final shutdown) remain safe and leak-free.
 */
void cs_tui_term_exit(void)
{
    if (!g_term_active)
    {
        return;
    }
    int out_fd = (g_tty_out_fd >= 0) ? g_tty_out_fd : STDOUT_FILENO;
    write_all_fd(out_fd, g_seq_exit, sizeof(g_seq_exit) - 1u);

    if (g_orig_stdout >= 0)
    {
        dup2(g_orig_stdout, STDOUT_FILENO);
        close(g_orig_stdout);
        g_orig_stdout = -1;
    }
    if (g_orig_stderr >= 0)
    {
        dup2(g_orig_stderr, STDERR_FILENO);
        close(g_orig_stderr);
        g_orig_stderr = -1;
    }
    if (g_cap_pipe[0] >= 0)
    {
        close(g_cap_pipe[0]);
        g_cap_pipe[0] = -1;
    }
    if (g_cap_pipe[1] >= 0)
    {
        close(g_cap_pipe[1]);
        g_cap_pipe[1] = -1;
    }
    if (g_tty_out_fd >= 0 && g_tty_out_fd != STDOUT_FILENO)
    {
        close(g_tty_out_fd);
        g_tty_out_fd = -1;
    }
    if (g_has_saved_tio)
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_saved_tio);
        g_has_saved_tio = 0;
    }
    g_term_active = 0;
}

/**
 * cs_tui_drain_captured_stdio() - Read pending bytes from the stdio capture pipe into `app->logs`
 * @app: Active TUI state to receive stripped log lines (may be NULL to discard output)
 *
 * Role:
 *   Drains buffered `stdout`/`stderr` text generated by `ImageStreamIO` or helper libraries during
 *   each main-loop iteration and routes non-blank lines into `cs_tui_log()`.
 *
 * Rationale:
 *   Keeps the `O_NONBLOCK` capture pipe empty while preserving valuable runtime warnings inside the
 *   `F5:STATUS` activity log instead of silently dropping `/dev/null` output.
 *
 * Return: 1 if data was read from the capture pipe, 0 otherwise.
 */
int cs_tui_drain_captured_stdio(cs_tui_app_t *app)
{
    if (g_cap_pipe[0] < 0)
    {
        return 0;
    }
    fflush(stdout);
    fflush(stderr);
    char    buf[512];
    ssize_t nr = read(g_cap_pipe[0], buf, sizeof(buf) - 1u);
    if (nr <= 0)
    {
        return 0;
    }
    buf[nr]   = '\0';
    char *cur = buf;
    while (*cur != '\0')
    {
        char *nl = strchr(cur, '\n');
        if (nl != NULL)
        {
            *nl = '\0';
        }
        size_t l = strlen(cur);
        while (l > 0u && (cur[l - 1u] == '\r' || cur[l - 1u] == ' '))
        {
            cur[--l] = '\0';
        }
        if (l > 0u && app != NULL)
        {
            cs_tui_log(app, 0, "%s", cur);
        }
        if (nl == NULL)
        {
            break;
        }
        cur = nl + 1;
    }
    return 1;
}

/* ── Non-Blocking Keyboard, SGR 1006 Mouse & Bracketed Paste Parser ───────── */

/** Lookahead byte accumulator for assembling multi-byte VT/CSI/SGR sequences across `read()`s. */
static unsigned char g_in_buf[512];
/** Number of unconsumed raw input bytes currently residing at the front of `g_in_buf`. */
static size_t        g_in_len = 0u;

/**
 * consume_in() - Discard @n decoded leading bytes from the `g_in_buf` lookahead queue
 * @n: Number of bytes to pop from the head of `g_in_buf`
 *
 * Role:
 *   Advances the input parser past a single decoded key, CSI escape sequence, mouse packet, or
 *   bracketed-paste block while preserving trailing coalesced events for the next call.
 *
 * Rationale:
 *   Rapid mouse drags or scroll-wheel bursts often deliver multiple `\033[<...M` packets in a
 *   single `read()` syscall; shifting remaining bytes in-place ensures zero dropped mouse ticks.
 */
static void consume_in(size_t n)
{
    if (n >= g_in_len)
    {
        g_in_len = 0u;
        return;
    }
    memmove(g_in_buf, g_in_buf + n, g_in_len - n);
    g_in_len -= n;
}

/**
 * fill_in_buf() - Wait up to @wait_ms for `STDIN_FILENO` readability and append to `g_in_buf`
 * @wait_ms: Maximum `poll()` timeout in milliseconds (`0` = non-blocking check)
 *
 * Role:
 *   Replenishes the raw input lookahead buffer without blocking indefinitely so live SHM telemetry
 *   refresh timers continue ticking when the user is idle.
 *
 * Rationale:
 *   Using `poll()` before `read()` provides millisecond-resolution timeouts both for main-loop
 *   frame pacing and for short (`10..16 ms`) escape-sequence completion windows over SSH links.
 */
static void fill_in_buf(int wait_ms)
{
    struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN, .revents = 0 };
    int           rc  = poll(&pfd, 1, wait_ms);
    if (rc > 0 && (pfd.revents & POLLIN) && g_in_len < sizeof(g_in_buf))
    {
        ssize_t nr = read(STDIN_FILENO, g_in_buf + g_in_len, sizeof(g_in_buf) - g_in_len);
        if (nr > 0)
        {
            g_in_len += (size_t) nr;
        }
    }
}

/**
 * cs_tui_poll_input() - Poll and decode the next normalized keyboard, mouse, paste, or resize event
 * @ev:         Output event structure zeroed and filled on return (must not be NULL)
 * @timeout_ms: Maximum wait in milliseconds when `g_in_buf` is empty
 *
 * Role:
 *   Translates raw terminal byte streams and async signal flags (`g_sig_quit`, `g_sig_winch`) into
 *   structured `cs_input_ev_t` records consumed by the TUI controller.
 *
 * Rationale:
 *   Implementing a dedicated state-free lookahead parser avoids `ncurses` `ESCDELAY` 1-second
 *   stalls while supporting xterm SGR 1006 decimal mouse coordinates (`> 223` columns, explicit
 *   `'M'` press vs `'m'` release), `DECSET 2004` bracketed paste, SS3 (`\033O`), and modified CSI
 *   sequences (`\033[1;2A` Shift+Up).
 *
 * Return: 1 when an event has been written to @ev, 0 when @timeout_ms elapsed with no input.
 */
int cs_tui_poll_input(cs_input_ev_t *ev, int timeout_ms)
{
    memset(ev, 0, sizeof(*ev));
    if (g_sig_quit)
    {
        ev->key = 3; /* Synthesize Ctrl+C so main loop runs clean unsaved-check / teardown */
        return 1;
    }
    if (g_sig_winch)
    {
        g_sig_winch = 0;
        ev->key     = CS_KEY_RESIZE;
        return 1;
    }

    if (g_in_len == 0u)
    {
        fill_in_buf(timeout_ms);
    }
    if (g_sig_quit)
    {
        ev->key = 3;
        return 1;
    }
    if (g_sig_winch)
    {
        g_sig_winch = 0;
        ev->key     = CS_KEY_RESIZE;
        return 1;
    }
    if (g_in_len == 0u)
    {
        return 0;
    }

    /* Fast path for ordinary ASCII printable bytes and C0 control characters */
    unsigned char c0 = g_in_buf[0];
    if (c0 != 27u)
    {
        consume_in(1u);
        if (c0 == '\r' || c0 == '\n')
        {
            ev->key = CS_KEY_ENTER;
        }
        else if (c0 == 8u || c0 == 127u)
        {
            ev->key = CS_KEY_BACKSPACE;
        }
        else if (c0 == '\t')
        {
            ev->key = CS_KEY_TAB;
        }
        else
        {
            ev->key = (int) c0;
        }
        return 1;
    }

    /* ESC sequence: wait briefly if only '\033' arrived so split packets complete */
    if (g_in_len == 1u)
    {
        fill_in_buf(16);
        if (g_in_len == 1u)
        {
            consume_in(1u);
            ev->key = CS_KEY_ESC;
            return 1;
        }
    }

    /* VT100 / xterm SS3 application-keypad & F1..F4 sequences: \033O{P,Q,R,S,A,B,C,D,H,F} */
    if (g_in_buf[1] == 'O' && g_in_len >= 3u)
    {
        unsigned char f = g_in_buf[2];
        consume_in(3u);
        if (f == 'P')
        {
            ev->key = CS_KEY_F1;
        }
        else if (f == 'Q')
        {
            ev->key = CS_KEY_F2;
        }
        else if (f == 'R')
        {
            ev->key = CS_KEY_F3;
        }
        else if (f == 'S')
        {
            ev->key = CS_KEY_F4;
        }
        else if (f == 'A')
        {
            ev->key = CS_KEY_UP;
        }
        else if (f == 'B')
        {
            ev->key = CS_KEY_DOWN;
        }
        else if (f == 'C')
        {
            ev->key = CS_KEY_RIGHT;
        }
        else if (f == 'D')
        {
            ev->key = CS_KEY_LEFT;
        }
        else if (f == 'H')
        {
            ev->key = CS_KEY_HOME;
        }
        else if (f == 'F')
        {
            ev->key = CS_KEY_END;
        }
        return 1;
    }

    if (g_in_buf[1] == '[')
    {
        if (g_in_len < 3u)
        {
            fill_in_buf(12);
        }
        /*
         * SGR 1006 extended mouse report: \033[<btn;col;rowM (press/drag/wheel) or ...m (release).
         * Wire coordinates (col, row) are 1-based; convert to 0-based (mx, my). `btn` encodes:
         *   bits 0..1: 0=left, 1=middle, 2=right, 3=no-button hover
         *   bit 5 (+32): motion/drag event (e.g., 32=left-drag, 35=hover move)
         *   bit 6 (+64): scroll wheel (64=wheel up, 65=wheel down)
         */
        if (g_in_len >= 3u && g_in_buf[2] == '<')
        {
            size_t end = 3u;
            while (end < g_in_len && g_in_buf[end] != 'M' && g_in_buf[end] != 'm')
            {
                end++;
            }
            if (end >= g_in_len)
            {
                fill_in_buf(12);
                while (end < g_in_len && g_in_buf[end] != 'M' && g_in_buf[end] != 'm')
                {
                    end++;
                }
            }
            if (end < g_in_len)
            {
                char tmp[64];
                size_t slen = end - 3u;
                if (slen >= sizeof(tmp))
                {
                    slen = sizeof(tmp) - 1u;
                }
                memcpy(tmp, g_in_buf + 3u, slen);
                tmp[slen] = '\0';
                int btn = 0, mx = 1, my = 1;
                if (sscanf(tmp, "%d;%d;%d", &btn, &mx, &my) == 3)
                {
                    ev->key      = CS_KEY_MOUSE;
                    ev->mbtn     = btn;
                    ev->mx       = mx - 1;
                    ev->my       = my - 1;
                    ev->mrelease = (g_in_buf[end] == 'm') ? 1 : 0;
                }
                consume_in(end + 1u);
                return 1;
            }
        }

        /* Bracketed paste (\033[200~ ... \033[201~): collect printable ASCII into ev->paste */
        if (g_in_len >= 6u && memcmp(g_in_buf, "\033[200~", 6u) == 0)
        {
            consume_in(6u);
            size_t plen = 0u;
            for (int tries = 0; tries < 20; tries++)
            {
                for (size_t i = 0u; i < g_in_len; i++)
                {
                    if (i + 6u <= g_in_len && memcmp(g_in_buf + i, "\033[201~", 6u) == 0)
                    {
                        consume_in(i + 6u);
                        ev->paste[plen] = '\0';
                        ev->key         = CS_KEY_PASTE;
                        return 1;
                    }
                    if (plen + 1u < sizeof(ev->paste) && g_in_buf[i] >= 32u && g_in_buf[i] < 127u)
                    {
                        ev->paste[plen++] = (char) g_in_buf[i];
                    }
                }
                g_in_len = 0u;
                fill_in_buf(10);
                if (g_in_len == 0u)
                {
                    break;
                }
            }
            ev->paste[plen] = '\0';
            ev->key         = CS_KEY_PASTE;
            return 1;
        }

        /* Standard ECMA-48 CSI sequences: \033[ param_bytes final_byte [0x40..0x7E] */
        size_t end = 2u;
        while (end < g_in_len && (g_in_buf[end] < 0x40u || g_in_buf[end] > 0x7Eu))
        {
            end++;
        }
        if (end < g_in_len)
        {
            unsigned char fin = g_in_buf[end];
            char          mid[32];
            size_t        mlen = end - 2u;
            if (mlen >= sizeof(mid))
            {
                mlen = sizeof(mid) - 1u;
            }
            memcpy(mid, g_in_buf + 2u, mlen);
            mid[mlen] = '\0';
            consume_in(end + 1u);

            /* Modifier parameter p2 == 2 denotes Shift in xterm CSI 1;2A / 1;2B sequences */
            int p1 = 0, p2 = 0;
            int np = sscanf(mid, "%d;%d", &p1, &p2);
            if (fin == 'A')
            {
                ev->key = (np == 2 && p2 == 2) ? CS_KEY_SHIFT_UP : CS_KEY_UP;
            }
            else if (fin == 'B')
            {
                ev->key = (np == 2 && p2 == 2) ? CS_KEY_SHIFT_DOWN : CS_KEY_DOWN;
            }
            else if (fin == 'C')
            {
                ev->key = CS_KEY_RIGHT;
            }
            else if (fin == 'D')
            {
                ev->key = CS_KEY_LEFT;
            }
            else if (fin == 'H')
            {
                ev->key = CS_KEY_HOME;
            }
            else if (fin == 'F')
            {
                ev->key = CS_KEY_END;
            }
            else if (fin == 'Z')
            {
                ev->key = CS_KEY_BTAB;
            }
            else if (fin == '~')
            {
                /* VT220/xterm tilde keycodes (note historic gap at 16 between F5=15 and F6=17) */
                switch (p1)
                {
                case 1:
                case 7:
                    ev->key = CS_KEY_HOME;
                    break;
                case 3:
                    ev->key = CS_KEY_DEL;
                    break;
                case 4:
                case 8:
                    ev->key = CS_KEY_END;
                    break;
                case 5:
                    ev->key = CS_KEY_PGUP;
                    break;
                case 6:
                    ev->key = CS_KEY_PGDN;
                    break;
                case 11:
                    ev->key = CS_KEY_F1;
                    break;
                case 12:
                    ev->key = CS_KEY_F2;
                    break;
                case 13:
                    ev->key = CS_KEY_F3;
                    break;
                case 14:
                    ev->key = CS_KEY_F4;
                    break;
                case 15:
                    ev->key = CS_KEY_F5;
                    break;
                case 17:
                    ev->key = CS_KEY_F6;
                    break;
                case 18:
                    ev->key = CS_KEY_F7;
                    break;
                case 19:
                    ev->key = CS_KEY_F8;
                    break;
                case 20:
                    ev->key = CS_KEY_F9;
                    break;
                case 21:
                    ev->key = CS_KEY_F10;
                    break;
                default:
                    break;
                }
            }
            return 1;
        }
    }

    consume_in(1u);
    ev->key = CS_KEY_ESC;
    return 1;
}

/* ── Shadow/Front Cell Buffer & Synchronized Delta Flush ──────────────────── */

/**
 * rgb_eq() - Test exact 24-bit channel equality between two RGB colors
 * @a: First RGB color triplet
 * @b: Second RGB color triplet
 *
 * Role:
 *   Fast equality predicate for cell-level delta skipping (`cell_eq`) and SGR state caching.
 *
 * Rationale:
 *   Comparing all three 8-bit channels directly avoids emitting redundant SGR escape sequences
 *   across contiguous runs of identically colored cells.
 *
 * Return: Non-zero if `(a.r, a.g, a.b) == (b.r, b.g, b.b)`, else 0.
 */
static int rgb_eq(cs_tui_rgb_t a, cs_tui_rgb_t b)
{
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

/**
 * cell_eq() - Determine whether two terminal cells have identical visual appearance
 * @a: Candidate back-buffer cell
 * @b: Currently displayed front-buffer cell
 *
 * Role:
 *   Core delta-comparison predicate inside `cs_scr_flush()` deciding whether cell `(x, y)` can be
 *   skipped completely on the wire.
 *
 * Rationale:
 *   Short-circuits on cheap integer fields (`attrs`, `has_bg`, `fg`, conditional `bg`) before
 *   comparing the 5-byte `ch` glyph string, ignoring `bg` differences when `has_bg == 0`.
 *
 * Return: Non-zero if @a and @b produce identical terminal output, else 0.
 */
static int cell_eq(const cs_cell_t *a, const cs_cell_t *b)
{
    return a->attrs == b->attrs && a->has_bg == b->has_bg && rgb_eq(a->fg, b->fg) &&
           (!a->has_bg || rgb_eq(a->bg, b->bg)) && strcmp(a->ch, b->ch) == 0;
}

/**
 * tui_rgb_to_256() - Quantize a 24-bit RGB color to the nearest xterm 256-color palette index
 * @c: 24-bit sRGB color triplet
 *
 * Role:
 *   Maps semantic 24-bit palette colors and wavelength hues onto indices `[16..255]` when
 *   `color_level == 2` (`TERM=xterm-256color` without `COLORTERM=truecolor`).
 *
 * Rationale:
 *   Neutral slate/gray UI backgrounds (`|r-g| < 10 && |g-b| < 10`) are routed to the fine 24-step
 *   xterm grayscale ramp (`232..255`, plus `16` black and `231` white) to prevent ugly green/purple
 *   tint casts on dark panels, while chromatic colors map onto the `6x6x6` cube (`16..231`).
 *
 * Return: xterm 256-color index in `[16..255]`.
 */
static int tui_rgb_to_256(cs_tui_rgb_t c)
{
    int dr = abs((int) c.r - (int) c.g);
    int dg = abs((int) c.g - (int) c.b);
    if (dr < 10 && dg < 10)
    {
        if (c.r < 8)
        {
            return 16;
        }
        if (c.r > 248)
        {
            return 231;
        }
        return 232 + (((int) c.r - 8) * 24) / 247;
    }
    int r6 = ((int) c.r * 5 + 127) / 255;
    int g6 = ((int) c.g * 5 + 127) / 255;
    int b6 = ((int) c.b * 5 + 127) / 255;
    return 16 + 36 * r6 + 6 * g6 + b6;
}

/**
 * tui_rgb_to_16() - Quantize a 24-bit RGB color to a 4-bit ANSI 16-color index `[0..15]`
 * @c: 24-bit sRGB color triplet
 *
 * Role:
 *   Provides readable foreground color differentiation on basic 16-color terminals (`color_level`
 *   `== 1`, e.g., Linux virtual console or `TERM=xterm`).
 *
 * Rationale:
 *   Thresholds R/G/B channels into ANSI bit positions (`1`=Red, `2`=Green, `4`=Blue) and combines
 *   ITU-R BT.601 perceived luminance (`0.299 R + 0.587 G + 0.114 B`) to set the bright bit (`+8`)
 *   or rescue dim-gray borders onto index `8` (bright black / dark gray) instead of invisible `0`.
 *
 * Return: ANSI 16-color index `[0..15]`.
 */
static int tui_rgb_to_16(cs_tui_rgb_t c)
{
    int lum    = (299 * (int) c.r + 587 * (int) c.g + 114 * (int) c.b) / 1000;
    int bright = (lum > 150) ? 8 : 0;
    int base   = ((c.r >= 110) ? 1 : 0) | ((c.g >= 110) ? 2 : 0) | ((c.b >= 110) ? 4 : 0);
    if (base == 0 && lum > 55)
    {
        return 8;
    }
    return base | bright;
}

/**
 * cs_scr_init() - Initialize a double-buffered `cs_screen_t` canvas for `(cols, rows)`
 * @s:           Screen state to initialize (must not be NULL)
 * @cols:        Requested width in columns (clamped to `>= 40`)
 * @rows:        Requested height in rows (clamped to `>= 12`)
 * @color_level: Active SGR color level `[0..3]`
 * @ascii_only:  1 to force 7-bit ASCII symbols, 0 to permit UTF-8 glyphs
 *
 * Role:
 *   Allocates zero-initialized `back` and `front` cell grids plus a contiguous ANSI escape output
 *   buffer `out_buf` large enough to serialize a full-screen TrueColor repaint in a single pass.
 *
 * Rationale:
 *   Allocating `ncells * 40 + 4096` bytes up front guarantees zero heap allocations inside
 *   `cs_scr_flush()` and ensures the entire frame leaves user space in a single synchronized batch.
 *
 * Return: 0 on success, -1 if any heap allocation fails.
 */
int cs_scr_init(cs_screen_t *s, int cols, int rows, int color_level, int ascii_only)
{
    memset(s, 0, sizeof(*s));
    if (cols < 40)
    {
        cols = 40;
    }
    if (rows < 12)
    {
        rows = 12;
    }
    s->cols        = cols;
    s->rows        = rows;
    s->color_level = color_level;
    s->ascii_only  = ascii_only;
    s->force_full  = 1;
    s->win_t0_sec  = cs_tui_now_sec();

    size_t ncells = (size_t) cols * (size_t) rows;
    s->back       = (cs_cell_t *) calloc(ncells, sizeof(cs_cell_t));
    s->front      = (cs_cell_t *) calloc(ncells, sizeof(cs_cell_t));
    s->out_cap    = ncells * 40u + 4096u;
    s->out_buf    = (char *) malloc(s->out_cap);
    if (s->back == NULL || s->front == NULL || s->out_buf == NULL)
    {
        cs_scr_free(s);
        return -1;
    }
    return 0;
}

/**
 * cs_scr_resize() - Reallocate `cs_screen_t` cell grids on terminal `SIGWINCH` resize
 * @s:    Screen instance to resize
 * @cols: New column count (clamped to `>= 40`)
 * @rows: New row count (clamped to `>= 12`)
 *
 * Role:
 *   Adapts the double buffer to updated terminal geometry and schedules an unconditional full-frame
 *   repaint (`force_full = 1`).
 *
 * Rationale:
 *   Short-circuits without reallocation when clamped `(cols, rows)` match existing dimensions while
 *   still setting `force_full = 1` so terminal emulator reflow artifacts are wiped cleanly.
 */
void cs_scr_resize(cs_screen_t *s, int cols, int rows)
{
    if (cols < 40)
    {
        cols = 40;
    }
    if (rows < 12)
    {
        rows = 12;
    }
    if (cols == s->cols && rows == s->rows)
    {
        s->force_full = 1;
        return;
    }
    int cl = s->color_level;
    int ao = s->ascii_only;
    double kb = s->tx_kbps;
    cs_scr_free(s);
    cs_scr_init(s, cols, rows, cl, ao);
    s->tx_kbps = kb;
}

/**
 * cs_scr_free() - Free heap buffers owned by a `cs_screen_t` instance
 * @s: Screen instance whose `back`, `front`, and `out_buf` arrays are released
 *
 * Role:
 *   Releases virtual canvas memory on shutdown, resize, or allocation rollback.
 *
 * Rationale:
 *   Nulling pointers after `free()` makes teardown idempotent across error and normal exit paths.
 */
void cs_scr_free(cs_screen_t *s)
{
    free(s->back);
    free(s->front);
    free(s->out_buf);
    s->back    = NULL;
    s->front   = NULL;
    s->out_buf = NULL;
    s->out_cap = 0u;
}

/**
 * cs_scr_clear() - Reset every cell in `s->back` to a space with default @fg/@bg colors
 * @s:  Target screen canvas
 * @fg: Default foreground RGB color
 * @bg: Default panel background RGB color
 *
 * Role:
 *   Prepares a clean back-buffer slate before each immediate-mode `cs_tui_render_frame()` pass.
 *
 * Rationale:
 *   Suppresses `has_bg` when `color_level < 2` so 16-color and `NO_COLOR` terminals keep their
 *   native terminal background instead of painting coarse ANSI 8-color blocks.
 */
void cs_scr_clear(cs_screen_t *s, cs_tui_rgb_t fg, cs_tui_rgb_t bg)
{
    size_t n = (size_t) s->cols * (size_t) s->rows;
    for (size_t i = 0u; i < n; i++)
    {
        s->back[i].ch[0]  = ' ';
        s->back[i].ch[1]  = '\0';
        s->back[i].fg     = fg;
        s->back[i].bg     = bg;
        s->back[i].attrs  = 0u;
        s->back[i].has_bg = (s->color_level >= 2) ? 1u : 0u;
    }
}

/**
 * emit_sgr() - Format a complete ECMA-48 SGR escape sequence for cell @c at @color_level
 * @dst:         Destination byte buffer inside `s->out_buf`
 * @c:           Cell supplying `attrs`, `fg`, `bg`, and `has_bg`
 * @color_level: Active color mode (`0`=mono attrs only, `1`=16c, `2`=256c, `3`=24-bit TrueColor)
 *
 * Role:
 *   Emits `\033[0;...m` resetting prior attributes and applying bold/dim/underline/reverse plus
 *   appropriate foreground and optional background color codes.
 *
 * Rationale:
 *   Starting each SGR transition from `0` (reset) eliminates state-tracking bugs when turning off
 *   individual `BOLD`/`DIM`/`REV` flags or transitioning between `has_bg == 1` and `has_bg == 0`.
 *
 * Return: Number of ASCII bytes appended to @dst.
 */
static size_t emit_sgr(char *dst, const cs_cell_t *c, int color_level)
{
    size_t n = 0u;
    n += (size_t) sprintf(dst + n, "\033[0");
    if (c->attrs & CS_ATTR_BOLD)
    {
        n += (size_t) sprintf(dst + n, ";1");
    }
    if (c->attrs & CS_ATTR_DIM)
    {
        n += (size_t) sprintf(dst + n, ";2");
    }
    if (c->attrs & CS_ATTR_UNDERLINE)
    {
        n += (size_t) sprintf(dst + n, ";4");
    }
    if (c->attrs & CS_ATTR_REV)
    {
        n += (size_t) sprintf(dst + n, ";7");
    }
    if (color_level >= 3)
    {
        n += (size_t) sprintf(dst + n, ";38;2;%u;%u;%u", c->fg.r, c->fg.g, c->fg.b);
        if (c->has_bg)
        {
            n += (size_t) sprintf(dst + n, ";48;2;%u;%u;%u", c->bg.r, c->bg.g, c->bg.b);
        }
    }
    else if (color_level == 2)
    {
        n += (size_t) sprintf(dst + n, ";38;5;%d", tui_rgb_to_256(c->fg));
        if (c->has_bg)
        {
            n += (size_t) sprintf(dst + n, ";48;5;%d", tui_rgb_to_256(c->bg));
        }
    }
    else if (color_level == 1)
    {
        int c16 = tui_rgb_to_16(c->fg);
        int fg  = (c16 >= 8) ? (90 + (c16 - 8)) : (30 + c16);
        n += (size_t) sprintf(dst + n, ";%d", fg);
    }
    dst[n++] = 'm';
    return n;
}

/**
 * cs_scr_flush() - Diff `s->back` against `s->front` and flush synchronized ANSI deltas to @out_fd
 * @s:      Screen canvas to present
 * @out_fd: Controlling TTY output file descriptor (`>= 0`)
 *
 * Role:
 *   Presents the newly rendered `s->back` frame to the user's terminal with zero flicker and
 *   minimal byte count, updating `s->front` and rolling 1 s `s->tx_kbps` bandwidth telemetry.
 *
 * Rationale:
 *   1. Wraps every frame in DEC synchronized output (`\033[?2026h` ... `\033[?2026l`) so modern
 *      terminals (Kitty, WezTerm, Alacritty, Foot, iTerm2, Ghostty, tmux) composite the entire
 *      update atomically on VSync; legacy terminals silently ignore unknown DEC private mode 2026.
 *   2. Skips unchanged cells (`cell_eq`), suppresses `CUP` (`\033[r;cH`) on contiguous horizontal
 *      cells, suppresses redundant SGR sequences when `attrs`/`fg`/`bg` match `cur_sgr`, and skips
 *      the `write()` syscall entirely (`pos <= 20`) when zero cells changed since the last frame.
 */
void cs_scr_flush(cs_screen_t *s, int out_fd)
{
    if (s->out_buf == NULL || out_fd < 0)
    {
        return;
    }

    size_t    pos      = 0u;
    int       cur_x    = -1;
    int       cur_y    = -1;
    cs_cell_t cur_sgr;
    int       has_sgr  = 0;
    memset(&cur_sgr, 0, sizeof(cur_sgr));

    /* Begin synchronized terminal update (\033[?2026h) */
    memcpy(s->out_buf + pos, "\033[?2026h", 8u);
    pos += 8u;
    if (s->force_full)
    {
        memcpy(s->out_buf + pos, "\033[0m\033[H", 7u);
        pos += 7u;
        cur_x = 0;
        cur_y = 0;
    }

    for (int y = 0; y < s->rows; y++)
    {
        for (int x = 0; x < s->cols; x++)
        {
            size_t           idx = (size_t) y * (size_t) s->cols + (size_t) x;
            const cs_cell_t *bc  = &s->back[idx];
            cs_cell_t       *fc  = &s->front[idx];

            if (!s->force_full && cell_eq(bc, fc))
            {
                continue;
            }
            if (pos + 128u >= s->out_cap)
            {
                break;
            }
            /* Emit 1-based CUP (\033[row;colH) only when cursor is not already at (x, y) */
            if (x != cur_x || y != cur_y)
            {
                pos += (size_t) sprintf(s->out_buf + pos, "\033[%d;%dH", y + 1, x + 1);
                cur_x = x;
                cur_y = y;
            }
            if (!has_sgr || bc->attrs != cur_sgr.attrs || bc->has_bg != cur_sgr.has_bg ||
                (s->color_level > 0 &&
                 (!rgb_eq(bc->fg, cur_sgr.fg) || (bc->has_bg && !rgb_eq(bc->bg, cur_sgr.bg)))))
            {
                pos += emit_sgr(s->out_buf + pos, bc, s->color_level);
                cur_sgr = *bc;
                has_sgr = 1;
            }
            size_t clen = strlen(bc->ch);
            if (clen == 0u)
            {
                s->out_buf[pos++] = ' ';
            }
            else
            {
                memcpy(s->out_buf + pos, bc->ch, clen);
                pos += clen;
            }
            *fc = *bc;
            cur_x++;
        }
    }

    /* Reset SGR attributes and commit synchronized terminal update (\033[?2026l) */
    memcpy(s->out_buf + pos, "\033[0m\033[?2026l", 12u);
    pos += 12u;
    s->force_full = 0;

    /* 20 bytes is the empty envelope ("\033[?2026h" + "\033[0m\033[?2026l"); write only if dirty */
    if (pos > 20u)
    {
        write_all_fd(out_fd, s->out_buf, pos);
        s->bytes_window += pos;
    }

    double now = cs_tui_now_sec();
    double dt  = now - s->win_t0_sec;
    if (dt >= 1.0)
    {
        s->tx_kbps      = ((double) s->bytes_window / 1024.0) / dt;
        s->bytes_window = 0u;
        s->win_t0_sec   = now;
    }
}

/**
 * cs_scr_dump_plain() - Dump `s->back` character glyphs into a plain-text newline-joined buffer
 * @s:   Rendered screen state to read (must not be NULL)
 * @out: Destination text buffer (must not be NULL)
 * @cap: Capacity of @out in bytes (`> 0`)
 *
 * Role:
 *   Serializes the rendered 2D character grid without SGR/CSI escape codes for `ctest` assertions
 *   and headless snapshot verification (`cacaosim_tui_render_headless()`).
 *
 * Rationale:
 *   Reading directly from `s->back` verifies the exact production layout and UTF-8/ASCII glyph
 *   output in microseconds with zero PTY or terminal-emulator dependencies.
 *
 * Return: Number of bytes written to @out (excluding NUL), or -1 on NULL/zero-capacity args.
 */
int cs_scr_dump_plain(const cs_screen_t *s, char *out, size_t cap)
{
    if (s == NULL || out == NULL || cap == 0u)
    {
        return -1;
    }
    size_t pos = 0u;
    for (int y = 0; y < s->rows; y++)
    {
        for (int x = 0; x < s->cols; x++)
        {
            const char *g    = s->back[(size_t) y * (size_t) s->cols + (size_t) x].ch;
            size_t      glen = strlen(g);
            if (glen == 0u)
            {
                g    = " ";
                glen = 1u;
            }
            if (pos + glen + 2u >= cap)
            {
                out[pos] = '\0';
                return (int) pos;
            }
            memcpy(out + pos, g, glen);
            pos += glen;
        }
        if (y + 1 < s->rows && pos + 2u < cap)
        {
            out[pos++] = '\n';
        }
    }
    out[pos] = '\0';
    return (int) pos;
}

/* ── Cell & Widget Drawing Primitives ─────────────────────────────────────── */

/**
 * cs_put_cell() - Store one clipped UTF-8/ASCII glyph, color pair, and SGR mask in `s->back`
 * @s:      Target screen canvas
 * @x:      0-based column coordinate (clipped to `[0..s->cols-1]`)
 * @y:      0-based row coordinate (clipped to `[0..s->rows-1]`)
 * @glyph:  Single-column UTF-8 or ASCII character string (`<= 4` bytes; NULL/empty -> `' '`)
 * @fg:     Foreground RGB color
 * @bg:     Background RGB color
 * @attrs:  Bitwise OR of `CS_ATTR_*` flags
 * @has_bg: Non-zero to enable explicit background color when `s->color_level >= 2`
 *
 * Role:
 *   Foundational bounds-checked cell mutator invoked by every text, border, shadow, pupil, and
 *   sparkline primitive.
 *
 * Rationale:
 *   Centralizing viewport clipping and 4-byte UTF-8 truncation makes out-of-bounds writes
 *   impossible even when scrollable DAG canvases or drop shadows extend past screen edges.
 */
void cs_put_cell(
    cs_screen_t *s,
    int          x,
    int          y,
    const char  *glyph,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs,
    int          has_bg)
{
    if (x < 0 || x >= s->cols || y < 0 || y >= s->rows)
    {
        return;
    }
    cs_cell_t *c = &s->back[(size_t) y * (size_t) s->cols + (size_t) x];
    if (glyph == NULL || glyph[0] == '\0')
    {
        c->ch[0] = ' ';
        c->ch[1] = '\0';
    }
    else
    {
        size_t l = strlen(glyph);
        if (l > 4u)
        {
            l = 4u;
        }
        memcpy(c->ch, glyph, l);
        c->ch[l] = '\0';
    }
    c->fg     = fg;
    c->bg     = bg;
    c->attrs  = attrs;
    c->has_bg = (has_bg && s->color_level >= 2) ? 1u : 0u;
}

/**
 * cs_fill_rect() - Paint a clipped solid-background rectangle of spaces into `s->back`
 * @s:  Target screen canvas
 * @x:  Leftmost column
 * @y:  Topmost row
 * @w:  Width in columns
 * @h:  Height in rows
 * @fg: Foreground RGB color
 * @bg: Background RGB fill color
 *
 * Role:
 *   Clears rectangular pane interiors, highlighted selection rows, and popup dialog backdrops.
 *
 * Rationale:
 *   Establishes uniform `bg` across every cell in `[x..x+w-1, y..y+h-1]` so subsequent
 *   `cs_put_str(..., has_bg = 0)` calls inherit the exact panel or highlight background cleanly.
 */
void cs_fill_rect(
    cs_screen_t *s,
    int          x,
    int          y,
    int          w,
    int          h,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg)
{
    for (int ry = y; ry < y + h; ry++)
    {
        if (ry < 0 || ry >= s->rows)
        {
            continue;
        }
        for (int rx = x; rx < x + w; rx++)
        {
            if (rx < 0 || rx >= s->cols)
            {
                continue;
            }
            cs_put_cell(s, rx, ry, " ", fg, bg, 0u, 1);
        }
    }
}

/**
 * utf8_char_len() - Determine byte length `[1..4]` of a UTF-8 sequence from its leading byte @b
 * @b: First unsigned byte of a UTF-8 codepoint
 *
 * Role:
 *   Advances `cs_put_str()` one complete Unicode codepoint per terminal column.
 *
 * Rationale:
 *   Inspects standard RFC 3629 UTF-8 prefix bitmasks (`0xxxxxxx` -> 1, `110xxxxx` -> 2,
 *   `1110xxxx` -> 3, `11110xxx` -> 4) and safely advances 1 byte on stray continuation bytes so
 *   malformed strings can never cause infinite loops or buffer overruns.
 *
 * Return: UTF-8 sequence length in bytes (`1..4`).
 */
static size_t utf8_char_len(unsigned char b)
{
    if ((b & 0x80u) == 0u)
    {
        return 1u;
    }
    if ((b & 0xE0u) == 0xC0u)
    {
        return 2u;
    }
    if ((b & 0xF0u) == 0xE0u)
    {
        return 3u;
    }
    if ((b & 0xF8u) == 0xF0u)
    {
        return 4u;
    }
    return 1u;
}

/**
 * cs_put_str() - Write at most @max_cols UTF-8 glyphs from @str at `(x, y)` in `s->back`
 * @s:        Target screen canvas
 * @x:        Starting 0-based column
 * @y:        Target 0-based row
 * @max_cols: Maximum character columns to emit (`<= 0` is a no-op)
 * @str:      NUL-terminated UTF-8 string (safe if NULL)
 * @fg:       Foreground RGB color
 * @bg:       Background RGB color when @has_bg != 0
 * @attrs:    Bitwise OR of `CS_ATTR_*` flags
 * @has_bg:   1 to overwrite cell background with @bg, 0 to preserve underlying cell background
 *
 * Role:
 *   Primary single-line UTF-8 text renderer with column-accurate truncation and C0 control-char
 *   sanitization.
 *
 * Rationale:
 *   Sanitizing bytes `< 32` to `' '` guarantees embedded tabs/CR/ESC in external log lines or SHM
 *   descriptions can never inject raw terminal control sequences or shift column alignment.
 *
 * Return: Number of terminal columns written `[0..max_cols]`.
 */
int cs_put_str(
    cs_screen_t *s,
    int          x,
    int          y,
    int          max_cols,
    const char  *str,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs,
    int          has_bg)
{
    if (str == NULL || y < 0 || y >= s->rows || max_cols <= 0)
    {
        return 0;
    }
    int         col = 0;
    const char *p   = str;
    while (*p != '\0' && col < max_cols)
    {
        size_t clen = utf8_char_len((unsigned char) *p);
        char   g[5] = { 0 };
        for (size_t i = 0u; i < clen && p[i] != '\0'; i++)
        {
            g[i] = p[i];
        }
        /* Replace C0 control bytes (< 0x20) with space so raw tabs/newlines cannot skew columns */
        if ((unsigned char) g[0] < 32u)
        {
            g[0] = ' ';
            g[1] = '\0';
            clen = 1u;
        }
        if (x + col >= 0 && x + col < s->cols)
        {
            cs_cell_t *existing = &s->back[(size_t) y * (size_t) s->cols + (size_t) (x + col)];
            cs_tui_rgb_t use_bg = has_bg ? bg : existing->bg;
            int          use_hb = has_bg ? 1 : existing->has_bg;
            cs_put_cell(s, x + col, y, g, fg, use_bg, attrs, use_hb);
        }
        p += clen;
        col++;
    }
    return col;
}

/**
 * cs_printf_at() - Format a `printf` string and draw up to @max_cols columns at `(x, y)`
 * @s:        Target screen canvas
 * @x:        Starting 0-based column
 * @y:        Target 0-based row
 * @max_cols: Maximum character columns to emit
 * @fg:       Foreground RGB color
 * @bg:       Background RGB color when @has_bg != 0
 * @attrs:    Bitwise OR of `CS_ATTR_*` flags
 * @has_bg:   1 to paint @bg, 0 to retain existing cell background
 * @fmt:      `printf` format string
 *
 * Role:
 *   Convenience formatted text writer used across status bars, tables, and inspector rows.
 *
 * Rationale:
 *   Uses stack-allocated `vsnprintf` (`512` bytes) and forwards to `cs_put_str()` to avoid heap
 *   allocations while retaining full UTF-8 truncation and control-byte filtering.
 *
 * Return: Number of terminal columns written `[0..max_cols]`.
 */
int cs_printf_at(
    cs_screen_t *s,
    int          x,
    int          y,
    int          max_cols,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs,
    int          has_bg,
    const char  *fmt,
    ...)
{
    char    buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return cs_put_str(s, x, y, max_cols, buf, fg, bg, attrs, has_bg);
}

/**
 * cs_draw_box() - Draw a filled rectangular frame with rounded/double borders and header labels
 * @s:             Target screen canvas
 * @x:             Leftmost box column
 * @y:             Topmost box row
 * @w:             Box width in columns (`< 2` returns immediately)
 * @h:             Box height in rows (`< 2` returns immediately)
 * @double_border: Non-zero for bold double-line focus border, 0 for rounded single-line border
 * @title:         Optional left header title rendered at `(x + 2, y)` (may be NULL)
 * @right_badge:   Optional right header badge rendered near `(x + w - 2, y)` (may be NULL)
 * @border_fg:     Border glyph RGB color
 * @title_fg:      Left title RGB color
 * @badge_fg:      Right badge RGB color
 * @bg:            Panel interior and border cell background RGB color
 *
 * Role:
 *   Renders consistent `milk-CTRL`-styled panes, optical node boxes, and modal frames.
 *
 * Rationale:
 *   Pairs Unicode rounded corners (`╭╮╰╯`, `U+256D..U+2570`) for passive panes with bold double
 *   lines (`╔╗╚╝═║`, `U+2550..U+255D`) for focused panes/modals — degrading cleanly to `+ - |` vs
 *   `+ = |` when `s->ascii_only != 0`.
 */
void cs_draw_box(
    cs_screen_t *s,
    int          x,
    int          y,
    int          w,
    int          h,
    int          double_border,
    const char  *title,
    const char  *right_badge,
    cs_tui_rgb_t border_fg,
    cs_tui_rgb_t title_fg,
    cs_tui_rgb_t badge_fg,
    cs_tui_rgb_t bg)
{
    if (w < 2 || h < 2)
    {
        return;
    }
    cs_fill_rect(s, x, y, w, h, border_fg, bg);

    const char *tl, *tr, *bl, *br, *hz, *vt;
    if (s->ascii_only)
    {
        tl = "+";
        tr = "+";
        bl = "+";
        br = "+";
        hz = double_border ? "=" : "-";
        vt = "|";
    }
    else if (double_border)
    {
        tl = "\xe2\x95\x94"; /* U+2554 BOX DRAWINGS DOUBLE DOWN AND RIGHT */
        tr = "\xe2\x95\x97"; /* U+2557 BOX DRAWINGS DOUBLE DOWN AND LEFT */
        bl = "\xe2\x95\x9a"; /* U+255A BOX DRAWINGS DOUBLE UP AND RIGHT */
        br = "\xe2\x95\x9d"; /* U+255D BOX DRAWINGS DOUBLE UP AND LEFT */
        hz = "\xe2\x95\x90"; /* U+2550 BOX DRAWINGS DOUBLE HORIZONTAL */
        vt = "\xe2\x95\x91"; /* U+2551 BOX DRAWINGS DOUBLE VERTICAL */
    }
    else
    {
        tl = "\xe2\x95\xad"; /* U+256D BOX DRAWINGS LIGHT ARC DOWN AND RIGHT */
        tr = "\xe2\x95\xae"; /* U+256E BOX DRAWINGS LIGHT ARC DOWN AND LEFT */
        bl = "\xe2\x95\xb0"; /* U+2570 BOX DRAWINGS LIGHT ARC UP AND RIGHT */
        br = "\xe2\x95\xaf"; /* U+256F BOX DRAWINGS LIGHT ARC UP AND LEFT */
        hz = "\xe2\x94\x80"; /* U+2500 BOX DRAWINGS LIGHT HORIZONTAL */
        vt = "\xe2\x94\x82"; /* U+2502 BOX DRAWINGS LIGHT VERTICAL */
    }

    uint8_t battr = double_border ? CS_ATTR_BOLD : 0u;
    cs_put_cell(s, x, y, tl, border_fg, bg, battr, 1);
    cs_put_cell(s, x + w - 1, y, tr, border_fg, bg, battr, 1);
    cs_put_cell(s, x, y + h - 1, bl, border_fg, bg, battr, 1);
    cs_put_cell(s, x + w - 1, y + h - 1, br, border_fg, bg, battr, 1);

    for (int cx = x + 1; cx < x + w - 1; cx++)
    {
        cs_put_cell(s, cx, y, hz, border_fg, bg, battr, 1);
        cs_put_cell(s, cx, y + h - 1, hz, border_fg, bg, battr, 1);
    }
    for (int cy = y + 1; cy < y + h - 1; cy++)
    {
        cs_put_cell(s, x, cy, vt, border_fg, bg, battr, 1);
        cs_put_cell(s, x + w - 1, cy, vt, border_fg, bg, battr, 1);
    }

    if (title != NULL && title[0] != '\0' && w > 6)
    {
        char tbuf[128];
        snprintf(tbuf, sizeof(tbuf), " %s ", title);
        cs_put_str(s, x + 2, y, w - 4, tbuf, title_fg, bg, CS_ATTR_BOLD, 1);
    }
    if (right_badge != NULL && right_badge[0] != '\0' && w > 12)
    {
        char bbuf[96];
        snprintf(bbuf, sizeof(bbuf), " %s ", right_badge);
        int blen = (int) strlen(bbuf);
        int bx   = x + w - 2 - blen;
        if (bx > x + 4)
        {
            cs_put_str(s, bx, y, blen, bbuf, badge_fg, bg, CS_ATTR_BOLD, 1);
        }
    }
}

/**
 * cs_draw_shadow() - Paint a 2-col right / 1-row bottom drop shadow around a modal popup box
 * @s:   Target screen canvas
 * @x:   Popup leftmost column
 * @y:   Popup topmost row
 * @w:   Popup width in columns
 * @h:   Popup height in rows
 * @pal: Active palette supplying dim foreground and half-intensity bar background
 *
 * Role:
 *   Gives floating modals and tooltips crisp visual elevation above dense background text.
 *
 * Rationale:
 *   Halving `pal->bg_bar` RGB components and rendering `U+2592` (`▒`, or `:` in ASCII mode)
 *   creates an unmistakable elevation cue across both dark and light themes.
 */
void cs_draw_shadow(cs_screen_t *s, int x, int y, int w, int h, const cs_tui_pal_t *pal)
{
    const char  *sh = s->ascii_only ? ":" : "\xe2\x96\x92"; /* U+2592 MEDIUM SHADE */
    cs_tui_rgb_t bg = { (uint8_t) (pal->bg_bar.r / 2u), (uint8_t) (pal->bg_bar.g / 2u),
                        (uint8_t) (pal->bg_bar.b / 2u) };
    for (int ry = y + 1; ry <= y + h; ry++)
    {
        cs_put_cell(s, x + w, ry, sh, pal->fg_dim, bg, CS_ATTR_DIM, 1);
        cs_put_cell(s, x + w + 1, ry, sh, pal->fg_dim, bg, CS_ATTR_DIM, 1);
    }
    for (int rx = x + 2; rx < x + w; rx++)
    {
        cs_put_cell(s, rx, y + h, sh, pal->fg_dim, bg, CS_ATTR_DIM, 1);
    }
}

/**
 * cs_draw_pupil16() - Render a `16x16` pupil amplitude thumbnail into `16` cols by `8` rows
 * @s:     Target screen canvas
 * @x:     Leftmost widget column
 * @y:     Topmost widget row
 * @pup16: Row-major `16x16` normalized pupil transmission array `[0..1]`
 * @pal:   Active theme palette
 *
 * Role:
 *   Renders an aspect-corrected live thumbnail of the telescope entrance aperture (outer circle,
 *   central obstruction `cobs`, and spider vanes) inside the `[bench]` Inspector pane.
 *
 * Rationale:
 *   Monospace character cells are roughly twice as tall as they are wide. Folding pairs of vertical
 *   pupil rows `(2*r, 2*r+1)` into single cells using `█` (`U+2588` both on), `▀` (`U+2580` top
 *   only), `▄` (`U+2584` bottom only), and `·` (`U+00B7` both off) doubles vertical resolution and
 *   renders circular pupils round rather than vertically stretched.
 */
void cs_draw_pupil16(
    cs_screen_t        *s,
    int                 x,
    int                 y,
    const float        *pup16,
    const cs_tui_pal_t *pal)
{
    for (int r = 0; r < 8; r++)
    {
        for (int c = 0; c < 16; c++)
        {
            float v_top = pup16[(2 * r) * 16 + c];
            float v_bot = pup16[(2 * r + 1) * 16 + c];
            if (s->ascii_only)
            {
                float       v  = 0.5f * (v_top + v_bot);
                const char *ch = (v > 0.70f) ? "#" : (v > 0.25f ? "+" : ".");
                cs_put_cell(s, x + c, y + r, ch, v > 0.25f ? pal->fg_stream : pal->fg_dim,
                            pal->bg_panel, 0u, 1);
                continue;
            }
            /* Pack 2 vertical subpixels into 1 cell via Unicode block elements */
            int on_top = (v_top >= 0.40f);
            int on_bot = (v_bot >= 0.40f);
            if (on_top && on_bot)
            {
                /* U+2588 FULL BLOCK */
                cs_put_cell(s, x + c, y + r, "\xe2\x96\x88", pal->fg_stream, pal->bg_panel, 0u, 1);
            }
            else if (on_top)
            {
                /* U+2580 UPPER HALF BLOCK */
                cs_put_cell(s, x + c, y + r, "\xe2\x96\x80", pal->fg_stream, pal->bg_panel, 0u, 1);
            }
            else if (on_bot)
            {
                /* U+2584 LOWER HALF BLOCK */
                cs_put_cell(s, x + c, y + r, "\xe2\x96\x84", pal->fg_stream, pal->bg_panel, 0u, 1);
            }
            else
            {
                /* U+00B7 MIDDLE DOT for obstructed / exterior pupil background */
                cs_put_cell(s, x + c, y + r, "\xc2\xb7", pal->fg_dim, pal->bg_panel, CS_ATTR_DIM,
                            1);
            }
        }
    }
}

/**
 * cs_tui_spec_mean_lambda() - Compute photon-flux-weighted centroid wavelength `[um]` of @spec
 * @spec: 32-bin optical beam spectrum (`0.40..2.40 um`; NULL returns `0.0f`)
 *
 * Role:
 *   Computes the effective central wavelength displayed next to optical beams in schematic wires
 *   and `F4:TABLES` beam summaries.
 *
 * Rationale:
 *   Using `\sum(ph_s[b] * lambda_b) / \sum(ph_s[b])` in `double` precision captures exact passband
 *   shifts after cascaded LP/SP/BP dichroic beamsplitters.
 *
 * Return: Photon-weighted mean wavelength in micrometers, or `0.0f` if @spec carries zero flux.
 */
float cs_tui_spec_mean_lambda(const cacaosim_spec_t *spec)
{
    if (spec == NULL)
    {
        return 0.0f;
    }
    double wsum = 0.0;
    double fsum = 0.0;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        double f = (double) spec->ph_s[b];
        if (f > 0.0)
        {
            fsum += f;
            wsum += f * (double) cacaosim_spec_bin_lambda(b);
        }
    }
    return (fsum > 0.0) ? (float) (wsum / fsum) : 0.0f;
}

/**
 * cs_draw_spectrum32() - Render a 1-row wavelength-colored spectral sparkline of width @width
 * @s:     Target screen canvas
 * @x:     Leftmost sparkline column
 * @y:     Sparkline row
 * @width: Number of columns to span (clamped to `[4..CACAOSIM_NSPEC]`)
 * @spec:  32-bin photon-rate spectrum (`ph_s[0..31]` over `0.40..2.40 um`)
 * @theme: Active UI theme index controlling `cs_tui_wavelength_rgb()` mapping
 * @bg:    Background RGB color behind sparkline bars
 *
 * Role:
 *   Renders inline polychromatic passband histograms inside DAG node cards and optical beam tables.
 *
 * Rationale:
 *   Max-pools bins `[b0..b1)` when `@width < 32` so narrow notch/bandpass features never vanish due
 *   to boxcar averaging, normalizes against peak bin `vmax`, and maps nonzero bins to 8 Unicode
 *   lower-block heights (`▁▂▃▄▅▆▇█`, `U+2581..U+2588`) colored by physical wavelength.
 */
void cs_draw_spectrum32(
    cs_screen_t           *s,
    int                    x,
    int                    y,
    int                    width,
    const cacaosim_spec_t *spec,
    int                    theme,
    cs_tui_rgb_t           bg)
{
    /* Unicode 1/8 .. 8/8 vertical block elements (U+2581 .. U+2588) */
    static const char *const u8_bars[8] = {
        "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
        "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88",
    };
    static const char *const asc_bars[8] = { ".", "_", "-", "=", "+", "*", "#", "#" };

    if (spec == NULL || width < 4)
    {
        return;
    }
    if (width > CACAOSIM_NSPEC)
    {
        width = CACAOSIM_NSPEC;
    }

    float vmax = 0.0f;
    for (int i = 0; i < CACAOSIM_NSPEC; i++)
    {
        if (spec->ph_s[i] > vmax)
        {
            vmax = spec->ph_s[i];
        }
    }

    const cs_tui_pal_t *pal = cs_tui_get_palette(theme);
    for (int c = 0; c < width; c++)
    {
        /* Max-pool spectral bins falling into column c so narrow passbands remain visible */
        int   b0 = (c * CACAOSIM_NSPEC) / width;
        int   b1 = ((c + 1) * CACAOSIM_NSPEC) / width;
        float fv = 0.0f;
        for (int b = b0; b < b1 || b == b0; b++)
        {
            if (b < CACAOSIM_NSPEC && spec->ph_s[b] > fv)
            {
                fv = spec->ph_s[b];
            }
            if (b1 <= b0)
            {
                break;
            }
        }
        float lam = cacaosim_spec_bin_lambda(b0);
        if (vmax <= 0.0f || fv <= 1.0e-4f * vmax)
        {
            cs_put_cell(s, x + c, y, s->ascii_only ? "." : "\xc2\xb7", pal->fg_dim, bg,
                        CS_ATTR_DIM, 1);
            continue;
        }
        float rel = fv / vmax;
        int   lvl = (int) (rel * 7.99f);
        if (lvl < 0)
        {
            lvl = 0;
        }
        if (lvl > 7)
        {
            lvl = 7;
        }
        cs_tui_rgb_t col = cs_tui_wavelength_rgb(lam, theme);
        cs_put_cell(s, x + c, y, s->ascii_only ? asc_bars[lvl] : u8_bars[lvl], col, bg, 0u, 1);
    }
}
