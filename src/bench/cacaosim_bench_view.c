// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_view.c
 * @brief   Stream-oriented ANSI/UTF-8/ASCII graphical optical bench schematic renderer
 *
 * Architectural Role in cacaosim:
 *   Implements the one-shot, non-interactive optical bench schematic renderer declared in
 *   cacaosim_bench_view.h as part of libmilkcacaosim.so. Consuming a validated and photometrically
 *   propagated cacaosim_bench_ctx_t (from cacaosim_bench_analyze() or cacaosim_bench_init()), this
 *   translation unit sequentially formats and writes a complete graphical hardware summary to any
 *   standard C FILE* stream:
 *     - Panel 1 (Bench Overview, Scales, Pupil Thumbnail & Stellar Spectrum): Synthesizes a 16x16
 *       anti-aliased telescope entrance pupil via cacaosim_make_pupil() and packs two vertical
 *       pupil pixels per terminal row using Unicode upper-half blocks ("▀") with independent
 *       foreground/background RGB alpha-blending. Alongside the 8-row pupil thumbnail, displays
 *       DAG validity and live MILK_SHM_DIR readiness pills, compute backend (CUDA GPU vs CPU
 *       SIMD), physical pupil/focal plate scales (m/pix, lambda/D per pix, sky arcsec/pix), master
 *       clock cadence, stellar magnitude and spectral type, total photon flux, and a 32-bin
 *       (0.40..2.40 um) wavelength-colored blackbody spectrum bar chart.
 *     - Panel 2 (2D Optical Path & Light-Splitting Schematic): Performs a depth-first recursive
 *       traversal of the optical DAG starting from beams[0]. Inline wavefront elements (DMs, TTMs,
 *       static WFE maps) remain aligned on the vertical optical axis, whereas gray/dichroic
 *       beamsplitters and coronagraphic focal/Lyot pickoffs branch laterally with tree connectors,
 *       per-arm photon-share percentages, 24-bin spectral sparklines, custom FITS/SHM mask tags,
 *       2D affine registration badges, and terminal multi-rate WFS / science camera readouts.
 *     - Panel 3 (Hardware & Photon-Budget Tables): Summarizes Active Mirrors (actuator grid,
 *       stroke, DAC bit depth, discrete ZOH/2nd-order dynamics, fractional frame delay), WFS &
 *       Science Cameras (passband, decimated frame rate master_hz/navg, integrated photons/frame,
 *       shot-noise vs RON regime classification), and Beamsplitters / Coronagraphs / Static WFE.
 *     - Error Diagnostic Panel (cacaosim_bench_render_error()): Formats syntax or DAG wiring
 *       errors with a section-by-section dump of partially parsed beam connections.
 *
 * Design Rationale & Motivation:
 *   1. Stream-oriented line emission vs full-screen cell framebuffer:
 *      While cacaosim-CTRL (cacaosim_tui.c) drives an interactive alternate-screen TUI (and can
 *      synthesize a live view from active MILK_SHM_DIR streams/FPS entries when no .bench file is
 *      given), operators and automation scripts (cacaosim-bench start/status, SSH sessions, CI
 *      logs, git/PR attachments, less -R) need schematics that persist in normal scrollback
 *      buffers without seizing a TTY. Emitting width-bounded, self-contained lines to a FILE*
 *      stream satisfies both interactive previews and headless memory-stream capture.
 *   2. Exact display-column accounting across CSI escapes and multi-byte UTF-8:
 *      A single 32-bin TrueColor spectrum bar contains > 600 bytes of ANSI CSI sequences for 34
 *      visible columns. Composing lines in a fixed stack buffer (sbuf_t) and measuring visible
 *      codepoint columns while skipping CSI sequences (vis_cols(), emit_truncated()) guarantees
 *      razor-sharp right-hand panel borders across TrueColor, 256-color, 16-color, monochrome,
 *      UTF-8, and 7-bit ASCII modes with zero heap allocations in the renderer.
 *   3. Non-intrusive live SHM binding:
 *      Checking <MILK_SHM_DIR>/<stream>.im.shm via access(..., F_OK) overlays real-time deployment
 *      status onto declarative .bench schematics without mapping shared memory, acquiring locks,
 *      or perturbing running CACAO real-time control processes.
 */

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "bench/cacaosim_bench_view.h"
#include "common/cacaosim_mask.h"
#include "pupil/cacaosim_pupil.h"

/**
 * struct cs_rgb_t - 24-bit sRGB color triplet used throughout palette and spectral rendering
 * @r: Red channel intensity in [0..255]
 * @g: Green channel intensity in [0..255]
 * @b: Blue channel intensity in [0..255]
 *
 * Role: Canonical internal color representation prior to terminal capability quantization
 * (TrueColor 24-bit CSI, xterm-256 index, ANSI-16 SGR code, or monochrome suppression).
 */
typedef struct
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} cs_rgb_t;

/**
 * struct cs_palette_t - Semantic 15-color theme palette synchronized with milk's overview TUI
 * @name:      Canonical lowercase CLI identifier ("dark", "night", "accessible", "light",
 *             "nordic")
 * @bg_panel:  Nominal panel interior background used as the zero-amplitude blend endpoint for
 *             anti-aliased 2D pupil half-block rendering and soft pill tinting
 * @bg_header: Elevated header/banner background color matching milk TUI title bars
 * @fg_title:  Primary panel border and banner badge accent color
 * @fg_text:   Standard body text color for hardware parameters and numeric scales
 * @fg_bright: High-emphasis foreground for node IDs, file paths, and peak photon rates
 * @fg_dim:    Low-contrast structural foreground for separators, empty spectrum bins, and axes
 * @fg_muted:  Secondary label foreground for parameter keys, units, and auxiliary notes
 * @fg_stream: Highlight color for ImageStreamIO SHM stream names and illuminated pupil amplitude
 * @fg_fps:    Accent color for deformable mirrors, FPS compute badges, and schematic headers
 * @fg_proc:   Accent color for Tip-Tilt mirrors, coronagraphs, and Zernike WFS phase-contrast
 *             nodes
 * @fg_active: Positive status color for valid DAG pills, live SHM indicators, WFS pills, and
 *             shot-noise-limited sensor regimes
 * @fg_warn:   Advisory color for beamsplitters, static WFE nodes, non-identity 2D registrations,
 *             and RON/mixed noise regimes
 * @fg_error:  High-salience alert color for invalid .bench configuration diagnostic panels
 * @fg_accent: Warm highlight color for named optical beams, focal-plane science cameras, and
 *             master clock rates
 * @fg_conn:   Optical axis, tree branch connector, and secondary panel border color
 *
 * Role: Decouples optical schematic semantics (beams, streams, DMs, sensors, warnings, errors)
 * from raw RGB values so switching themes re-colors every panel, badge, and pupil preview
 * coherently.
 */
typedef struct
{
    const char *name;
    cs_rgb_t    bg_panel;
    cs_rgb_t    bg_header;
    cs_rgb_t    fg_title;
    cs_rgb_t    fg_text;
    cs_rgb_t    fg_bright;
    cs_rgb_t    fg_dim;
    cs_rgb_t    fg_muted;
    cs_rgb_t    fg_stream;
    cs_rgb_t    fg_fps;
    cs_rgb_t    fg_proc;
    cs_rgb_t    fg_active;
    cs_rgb_t    fg_warn;
    cs_rgb_t    fg_error;
    cs_rgb_t    fg_accent;
    cs_rgb_t    fg_conn;
} cs_palette_t;

/**
 * g_palettes - Static lookup table of built-in semantic color palettes
 *
 * Exact RGB triplets match milk's src/cli/overview/render/overview_theme_palettes.c so cacaosim
 * schematics blend seamlessly alongside milk-stream-graph and milk overview dashboards:
 *   - dark:       High-contrast cool slate with cyan streams, periwinkle DMs, and gold beams.
 *   - night:      Long-wavelength red/amber ramp minimizing blue/green scotopic rhodopsin
 *                 bleaching in darkened observatory control rooms.
 *   - accessible: Wong / Okabe-Ito colorblind-safe hues with distinct luminance steps.
 *   - light:      Saturated dark ink tones legible against white/paper terminal backgrounds.
 *   - nordic:     Polar-night slate with muted frost cyan and aurora accents for reduced eye
 *                 strain during extended loop monitoring.
 */
static const cs_palette_t g_palettes[CACAOSIM_THEME_COUNT] = {
    /* CACAOSIM_THEME_DARK (default milk dark slate) */
    {
        .name      = "dark",
        .bg_panel  = { 30, 32, 40 },
        .bg_header = { 40, 44, 58 },
        .fg_title  = { 130, 170, 255 },
        .fg_text   = { 200, 205, 215 },
        .fg_bright = { 240, 245, 255 },
        .fg_dim    = { 100, 105, 120 },
        .fg_muted  = { 135, 142, 160 },
        .fg_stream = { 80, 200, 220 },
        .fg_fps    = { 130, 170, 255 },
        .fg_proc   = { 180, 140, 255 },
        .fg_active = { 80, 220, 80 },
        .fg_warn   = { 255, 180, 0 },
        .fg_error  = { 240, 60, 60 },
        .fg_accent = { 255, 215, 80 },
        .fg_conn   = { 100, 130, 180 },
    },
    /* CACAOSIM_THEME_NIGHT (Observatory Red — night-vision safe) */
    {
        .name      = "night",
        .bg_panel  = { 24, 8, 8 },
        .bg_header = { 42, 12, 12 },
        .fg_title  = { 255, 90, 60 },
        .fg_text   = { 210, 65, 45 },
        .fg_bright = { 255, 120, 85 },
        .fg_dim    = { 115, 35, 28 },
        .fg_muted  = { 155, 48, 36 },
        .fg_stream = { 245, 95, 65 },
        .fg_fps    = { 255, 80, 50 },
        .fg_proc   = { 225, 65, 70 },
        .fg_active = { 255, 115, 60 },
        .fg_warn   = { 255, 95, 40 },
        .fg_error  = { 255, 45, 45 },
        .fg_accent = { 255, 130, 70 },
        .fg_conn   = { 145, 42, 32 },
    },
    /* CACAOSIM_THEME_ACCESSIBLE (Okabe-Ito colorblind-safe) */
    {
        .name      = "accessible",
        .bg_panel  = { 26, 28, 34 },
        .bg_header = { 38, 42, 52 },
        .fg_title  = { 86, 180, 233 },
        .fg_text   = { 215, 220, 228 },
        .fg_bright = { 250, 252, 255 },
        .fg_dim    = { 110, 116, 128 },
        .fg_muted  = { 150, 158, 172 },
        .fg_stream = { 86, 180, 233 },
        .fg_fps    = { 0, 114, 178 },
        .fg_proc   = { 204, 121, 167 },
        .fg_active = { 0, 158, 115 },
        .fg_warn   = { 230, 159, 0 },
        .fg_error  = { 213, 94, 0 },
        .fg_accent = { 240, 228, 66 },
        .fg_conn   = { 115, 145, 175 },
    },
    /* CACAOSIM_THEME_LIGHT (Paper Light) */
    {
        .name      = "light",
        .bg_panel  = { 236, 239, 244 },
        .bg_header = { 220, 226, 236 },
        .fg_title  = { 25, 75, 175 },
        .fg_text   = { 35, 40, 50 },
        .fg_bright = { 10, 15, 25 },
        .fg_dim    = { 130, 136, 150 },
        .fg_muted  = { 95, 102, 118 },
        .fg_stream = { 0, 118, 142 },
        .fg_fps    = { 30, 85, 185 },
        .fg_proc   = { 125, 55, 175 },
        .fg_active = { 20, 135, 55 },
        .fg_warn   = { 185, 105, 0 },
        .fg_error  = { 195, 30, 35 },
        .fg_accent = { 165, 95, 0 },
        .fg_conn   = { 105, 125, 160 },
    },
    /* CACAOSIM_THEME_NORDIC (Nord frost & aurora) */
    {
        .name      = "nordic",
        .bg_panel  = { 46, 52, 64 },
        .bg_header = { 59, 66, 82 },
        .fg_title  = { 136, 192, 208 },
        .fg_text   = { 216, 222, 233 },
        .fg_bright = { 236, 239, 244 },
        .fg_dim    = { 108, 118, 138 },
        .fg_muted  = { 144, 156, 178 },
        .fg_stream = { 143, 188, 187 },
        .fg_fps    = { 129, 161, 193 },
        .fg_proc   = { 180, 142, 173 },
        .fg_active = { 163, 190, 140 },
        .fg_warn   = { 235, 203, 139 },
        .fg_error  = { 191, 97, 106 },
        .fg_accent = { 235, 203, 139 },
        .fg_conn   = { 94, 129, 172 },
    },
};

/**
 * cacaosim_view_opts_defaults() - Populate default milk-styled visualization options
 * @opts: Destination rendering options structure (safe no-op if NULL)
 *
 * Role: Initializes @opts with standard auto-detection sentinels (color_level = -1, width = 0)
 * and enables full rich output (dark theme, UTF-8 box-drawing, live SHM checks, hardware tables,
 * and workflow hints footer).
 *
 * Rationale: Providing a single canonical defaults initializer lets callers override one or two
 * fields (e.g. disabling footer in watch mode) without enumerating every rendering knob.
 */
void cacaosim_view_opts_defaults(cacaosim_view_opts_t *opts)
{
    if (opts == NULL)
    {
        return;
    }
    opts->color_level = -1;
    opts->theme       = CACAOSIM_THEME_DARK;
    opts->ascii_only  = 0;
    opts->check_shm   = 1;
    opts->show_tables = 1;
    opts->show_footer = 1;
    opts->width       = 0;
}

/**
 * cacaosim_view_theme_name() - Return human-readable canonical name for a theme index
 * @theme: Palette index from enum cacaosim_view_theme_t
 *
 * Role: Resolves a theme enum value to its lowercase string label in g_palettes[].
 *
 * Rationale: Safely clamps out-of-range indices to "dark" so UI status bars never dereference
 * invalid memory even if an unvalidated integer is passed.
 *
 * Return: Static NUL-terminated theme name ("dark" if @theme < 0 or >= CACAOSIM_THEME_COUNT).
 */
const char *cacaosim_view_theme_name(int theme)
{
    if (theme < 0 || theme >= CACAOSIM_THEME_COUNT)
    {
        return "dark";
    }
    return g_palettes[theme].name;
}

/**
 * cacaosim_view_theme_from_name() - Parse a case-insensitive theme name or alias
 * @name: Candidate theme string (e.g. "dark", "night", "red", "accessible", "light", "nordic",
 *        "nord"); may be NULL
 *
 * Role: Converts CLI --theme strings into validated cacaosim_view_theme_t indices.
 *
 * Rationale: Accepts ergonomic observatory/community aliases ("red" for night-vision mode and
 * "nord" for nordic) while returning -1 on unknown strings so CLI argument parsers can report
 * explicit valid choices.
 *
 * Return: Theme index in [0..CACAOSIM_THEME_COUNT-1], or -1 if @name is NULL or unrecognized.
 */
int cacaosim_view_theme_from_name(const char *name)
{
    if (name == NULL)
    {
        return -1;
    }
    for (int i = 0; i < CACAOSIM_THEME_COUNT; i++)
    {
        if (strcasecmp(name, g_palettes[i].name) == 0)
        {
            return i;
        }
    }
    if (strcasecmp(name, "red") == 0)
    {
        return CACAOSIM_THEME_NIGHT;
    }
    if (strcasecmp(name, "nord") == 0)
    {
        return CACAOSIM_THEME_NORDIC;
    }
    return -1;
}

/**
 * resolve_color_level() - Determine effective ANSI color capability for destination stream @fp
 * @opts: Caller rendering options (NULL or opts->color_level < 0 triggers auto-detection)
 * @fp:   Target output FILE stream (inspected via fileno() + isatty(); may be NULL or memstream)
 *
 * Role: Resolves the effective SGR color emission tier:
 *   0 = monochrome (no CSI color/bold escapes),
 *   1 = 16-color standard ANSI SGR (30..37, 90..97, 40..47),
 *   2 = xterm-256 indexed palette (38;5;N / 48;5;N),
 *   3 = 24-bit TrueColor RGB (38;2;R;G;B / 48;2;R;G;B).
 *
 * Rationale: Mirrors milk's ov_detect_color_level() rules verbatim: explicit caller overrides
 * (--pretty / --text) win first; otherwise any non-empty NO_COLOR environment variable
 * (https://no-color.org), non-TTY output stream (pipes, files, open_memstream with fileno == -1),
 * or TERM=dumb drops cleanly to monochrome so redirected logs never contain raw escape garbage.
 *
 * Return: Effective color level integer in [0..3].
 */
static int resolve_color_level(const cacaosim_view_opts_t *opts, FILE *fp)
{
    if (opts != NULL && opts->color_level >= 0)
    {
        return (opts->color_level > 3) ? 3 : opts->color_level;
    }
    const char *nocolor = getenv("NO_COLOR");
    if (nocolor != NULL && nocolor[0] != '\0')
    {
        return 0;
    }
    int fd = (fp != NULL) ? fileno(fp) : -1;
    if (fd < 0 || !isatty(fd))
    {
        return 0;
    }
    const char *term = getenv("TERM");
    if (term == NULL || term[0] == '\0' || strcmp(term, "dumb") == 0)
    {
        return 0;
    }
    const char *colorterm = getenv("COLORTERM");
    if (colorterm != NULL &&
        (strstr(colorterm, "truecolor") != NULL || strstr(colorterm, "24bit") != NULL))
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
 * resolve_ascii_mode() - Decide whether to restrict glyphs to pure 7-bit ASCII
 * @opts: Caller rendering options (NULL-safe)
 *
 * Role: Selects between rich UTF-8 rounded box-drawing / block sparklines (0) and 7-bit ASCII
 * fallback glyphs (1).
 *
 * Rationale: Minimal container environments, CI runners, and legacy serial/SSH sessions often run
 * under LC_ALL=C or POSIX locales where multi-byte UTF-8 box characters render as mojibake ("???").
 * Checking LC_ALL -> LC_CTYPE -> LANG hierarchy automatically degrades to crisp 7-bit ASCII
 * whenever UTF-8 locale support is absent.
 *
 * Return: 1 for pure 7-bit ASCII output, 0 for UTF-8 box-drawing and block glyphs.
 */
static int resolve_ascii_mode(const cacaosim_view_opts_t *opts)
{
    if (opts != NULL && opts->ascii_only)
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
 * resolve_panel_width() - Compute bounded outer panel width in terminal columns
 * @opts: Caller rendering options (explicit opts->width >= 76 overrides auto-sizing)
 * @fp:   Destination FILE stream queried via TIOCGWINSZ when backed by a TTY
 *
 * Role: Determines the exact outer frame width (including left/right border glyphs) used by
 * panel_top(), panel_sep(), panel_bot(), and panel_row().
 *
 * Rationale: Unbounded wide terminals (e.g. 240+ col ultrawide monitors) stretch schematic tables
 * into hard-to-scan whitespace, while narrow windows (< 98 cols) clip sensor columns prematurely.
 * Auto-detection subtracts a 2-column right margin from TIOCGWINSZ and clamps within [98..114]
 * columns (defaulting to 104 columns on non-TTYs), while honoring explicit -W overrides in
 * [76..128] columns.
 *
 * Return: Target outer panel width in display columns.
 */
static int resolve_panel_width(const cacaosim_view_opts_t *opts, FILE *fp)
{
    if (opts != NULL && opts->width >= 76)
    {
        return (opts->width > 128) ? 128 : opts->width;
    }
    int cols = 104;
    int fd   = (fp != NULL) ? fileno(fp) : -1;
    if (fd >= 0 && isatty(fd))
    {
        struct winsize ws;
        if (ioctl(fd, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 80)
        {
            cols = (int) ws.ws_col - 2;
        }
    }
    if (cols < 98)
    {
        cols = 98;
    }
    if (cols > 114)
    {
        cols = 114;
    }
    return cols;
}

/**
 * rgb_to_256() - Quantize a 24-bit sRGB color into the closest xterm-256 palette index
 * @c: Input 24-bit RGB color triplet
 *
 * Role: Maps arbitrary theme and spectral colors onto xterm's 256-color table (indices 16..255)
 * when rendering on color_level == 2 terminals (e.g. TERM=xterm-256color without COLORTERM).
 *
 * Rationale: Matches milk's overview_theme_draw.h quantizer. Near-neutral grays (|R-G| < 10 and
 * |G-B| < 10) are routed to the 24-step grayscale ramp (indices 232..255, plus pure black 16 and
 * white 231) so slate borders and muted text never acquire unwanted color casts from the coarse
 * 6x6x6 color cube (indices 16..231).
 *
 * Return: xterm-256 palette index in [16..255].
 */
static int rgb_to_256(cs_rgb_t c)
{
    int dr = (int) c.r - (int) c.g;
    int dg = (int) c.g - (int) c.b;
    if (dr < 0)
    {
        dr = -dr;
    }
    if (dg < 0)
    {
        dg = -dg;
    }
    /* Route achromatic / near-neutral slate tones onto the 24-step grayscale ramp */
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
    /* Quantize chromatic colors onto the 6x6x6 xterm RGB cube */
    int r6 = ((int) c.r * 5 + 127) / 255;
    int g6 = ((int) c.g * 5 + 127) / 255;
    int b6 = ((int) c.b * 5 + 127) / 255;
    return 16 + 36 * r6 + 6 * g6 + b6;
}

/**
 * rgb_to_16() - Quantize a 24-bit sRGB color into a 4-bit ANSI-16 color index
 * @c: Input 24-bit RGB color triplet
 *
 * Role: Maps RGB colors onto standard 16-color ANSI indices [0..15] (0..7 normal, 8..15 bright)
 * for legacy color_level == 1 terminals.
 *
 * Rationale: Uses ITU-R BT.601 perceived luminance (0.299*R + 0.587*G + 0.114*B) to select the
 * high-intensity bit (+8) and lifts dim grays (base == 0, lum > 55) to bright-black / dark-gray
 * (index 8) so dimmed borders and muted labels remain legible instead of collapsing to black (0).
 *
 * Return: ANSI-16 color index in [0..15].
 */
static int rgb_to_16(cs_rgb_t c)
{
    int lum    = (299 * (int) c.r + 587 * (int) c.g + 114 * (int) c.b) / 1000;
    int bright = (lum > 150) ? 8 : 0;
    int base   = ((c.r >= 110) ? 1 : 0) | ((c.g >= 110) ? 2 : 0) | ((c.b >= 110) ? 4 : 0);
    if (base == 0 && lum > 55)
    {
        return 8;
    }
    return base + bright;
}

/**
 * rgb_on() - Choose a high-contrast dark or light foreground ink for solid pill background @bg
 * @bg: Solid badge background color
 *
 * Role: Computes readable text foreground color for inverted solid pills (sb_pill()).
 *
 * Rationale: Comparing BT.601 weighted luma (299*R + 587*G + 114*B) against a mid-perceptual
 * threshold (140000 out of 255000) automatically switches between near-black ink on bright/warm
 * pills (e.g. gold/cyan/green) and crisp off-white ink on deep red/blue pills across all 5 themes.
 *
 * Return: High-contrast foreground cs_rgb_t triplet.
 */
static cs_rgb_t rgb_on(cs_rgb_t bg)
{
    int lum = 299 * (int) bg.r + 587 * (int) bg.g + 114 * (int) bg.b;
    if (lum > 140000)
    {
        cs_rgb_t dark = { 18, 20, 26 };
        return dark;
    }
    cs_rgb_t light = { 245, 248, 255 };
    return light;
}

/**
 * rgb_lerp() - Linearly interpolate between two 24-bit sRGB colors with clamping
 * @a: Start color (returned when @t <= 0.0f)
 * @b: End color (returned when @t >= 1.0f)
 * @t: Blend fraction clamped to [0.0f..1.0f]
 *
 * Role: Core color-blending primitive used for anti-aliased pupil amplitude shading, soft-tinted
 * badge backgrounds, and continuous piecewise-linear spectral wavelength color ramps.
 *
 * Rationale: Clamping @t in-place guards against out-of-band wavelengths (< 0.40 um or > 2.40 um)
 * or numerical overshoot in pupil anti-aliasing without requiring callers to pre-clamp.
 *
 * Return: Interpolated and rounded cs_rgb_t color triplet.
 */
static cs_rgb_t rgb_lerp(cs_rgb_t a, cs_rgb_t b, float t)
{
    if (t < 0.0f)
    {
        t = 0.0f;
    }
    if (t > 1.0f)
    {
        t = 1.0f;
    }
    cs_rgb_t out = {
        .r = (uint8_t) ((float) a.r + t * ((float) b.r - (float) a.r) + 0.5f),
        .g = (uint8_t) ((float) a.g + t * ((float) b.g - (float) a.g) + 0.5f),
        .b = (uint8_t) ((float) a.b + t * ((float) b.b - (float) a.b) + 0.5f),
    };
    return out;
}

/**
 * struct sbuf_t - Fixed-capacity stack-allocated append-only line composition buffer
 * @data: NUL-terminated UTF-8 + ANSI CSI byte buffer (4096 bytes)
 * @len:  Current byte length in @data (excluding trailing NUL; always < sizeof(data))
 *
 * Role: Accumulates styled text, multi-byte UTF-8 glyphs, and SGR escape sequences for a single
 * panel row before panel_row() measures, truncates, pads, and flushes it to the output stream.
 *
 * Rationale: A single ~110-column TrueColor schematic row (containing pills, stream tags, and a
 * 24- or 32-bin wavelength-colored spectrum bar) easily expands to 800..1500 raw bytes due to
 * "\033[38;2;R;G;Bm" sequences. Using a 4 KiB stack buffer eliminates all heap allocations while
 * guaranteeing truncation-safe formatting.
 */
typedef struct
{
    char   data[4096];
    size_t len;
} sbuf_t;

/**
 * sb_init() - Reset an sbuf_t line accumulator to the empty NUL-terminated state
 * @sb: Target stack string buffer (non-NULL)
 *
 * Role: Prepares @sb at the start of composing each schematic or table row.
 *
 * Rationale: O(1) initialization writing only the leading NUL byte rather than zeroing all 4096
 * bytes on every row.
 */
static void sb_init(sbuf_t *sb)
{
    sb->data[0] = '\0';
    sb->len     = 0;
}

/**
 * sb_puts() - Append a literal NUL-terminated string to @sb with bounds clamping
 * @sb: Target stack string buffer (non-NULL)
 * @s:  Source C string to append (safe no-op if NULL)
 *
 * Role: Fast literal string and UTF-8 glyph appender for sbuf_t.
 *
 * Rationale: Avoids printf format-string parsing overhead for fixed box-drawing glyphs, spaces,
 * and stream identifiers while strictly preserving NUL termination if capacity is reached.
 */
static void sb_puts(sbuf_t *sb, const char *s)
{
    if (s == NULL)
    {
        return;
    }
    size_t n = strlen(s);
    if (sb->len + n >= sizeof(sb->data))
    {
        n = sizeof(sb->data) - 1 - sb->len;
    }
    if (n > 0)
    {
        memcpy(sb->data + sb->len, s, n);
        sb->len += n;
        sb->data[sb->len] = '\0';
    }
}

/**
 * sb_printf() - Append printf-formatted text to @sb with overflow clamping
 * @sb:  Target stack string buffer (non-NULL)
 * @fmt: Standard printf format string followed by variadic arguments
 *
 * Role: Appends formatted numbers, optical scales, and ANSI CSI sequences into @sb.
 *
 * Rationale: vsnprintf() returns the would-be length on truncation rather than the bytes written;
 * clamping @sb->len against remaining capacity prevents buffer over-advance on long rows.
 */
static void sb_printf(sbuf_t *sb, const char *fmt, ...)
{
    if (sb->len + 1 >= sizeof(sb->data))
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    int w = vsnprintf(sb->data + sb->len, sizeof(sb->data) - sb->len, fmt, ap);
    va_end(ap);
    if (w > 0)
    {
        size_t rem = sizeof(sb->data) - 1 - sb->len;
        sb->len += ((size_t) w > rem) ? rem : (size_t) w;
    }
}

/**
 * sb_fg() - Append an SGR foreground color sequence to @sb matching terminal color tier @lvl
 * @sb:  Target stack string buffer (non-NULL)
 * @lvl: Effective color capability level (0=mono no-op, 1=16c, 2=256c, >=3=24-bit TrueColor)
 * @c:   Desired 24-bit sRGB foreground color
 *
 * Role: Emits the appropriate ANSI foreground escape sequence (or nothing in monochrome mode).
 *
 * Rationale: Funneling all foreground color changes through one tier-aware helper keeps high-level
 * schematic layout code completely agnostic to whether output targets TrueColor, xterm-256,
 * ANSI-16, or plain monochrome text.
 */
static void sb_fg(sbuf_t *sb, int lvl, cs_rgb_t c)
{
    if (lvl <= 0)
    {
        return;
    }
    if (lvl >= 3)
    {
        sb_printf(sb, "\033[38;2;%u;%u;%um", c.r, c.g, c.b);
    }
    else if (lvl == 2)
    {
        sb_printf(sb, "\033[38;5;%dm", rgb_to_256(c));
    }
    else
    {
        int c16 = rgb_to_16(c);
        sb_printf(sb, "\033[%dm", (c16 < 8) ? (30 + c16) : (90 + c16 - 8));
    }
}

/**
 * sb_bg() - Append an SGR background color sequence to @sb matching terminal color tier @lvl
 * @sb:  Target stack string buffer (non-NULL)
 * @lvl: Effective color capability level (0=mono no-op, 1=16c, 2=256c, >=3=24-bit TrueColor)
 * @c:   Desired 24-bit sRGB background color
 *
 * Role: Emits background fill escapes for solid/soft pills and lower-half-pixel blending in the
 * 16x16 anti-aliased pupil thumbnail.
 *
 * Rationale: On 16-color terminals (lvl == 1), masks out the bright bit (& 7 -> SGR 40..47) to
 * avoid non-portable blink/bright-background SGR 100..107 behavior on older emulators.
 */
static void sb_bg(sbuf_t *sb, int lvl, cs_rgb_t c)
{
    if (lvl <= 0)
    {
        return;
    }
    if (lvl >= 3)
    {
        sb_printf(sb, "\033[48;2;%u;%u;%um", c.r, c.g, c.b);
    }
    else if (lvl == 2)
    {
        sb_printf(sb, "\033[48;5;%dm", rgb_to_256(c));
    }
    else
    {
        int c16 = rgb_to_16(c) & 7;
        sb_printf(sb, "\033[%dm", 40 + c16);
    }
}

/**
 * sb_bold() - Append SGR bold attribute ("\033[1m") when ANSI styling is active
 * @sb:  Target stack string buffer (non-NULL)
 * @lvl: Effective color capability level (> 0 emits bold, <= 0 is a no-op)
 *
 * Role: Emphasizes node identifiers, SHM stream names, pills, and key optical plate scales.
 *
 * Rationale: Suppressing SGR bold when @lvl <= 0 ensures --text and NO_COLOR outputs remain
 * 100% free of any control characters.
 */
static void sb_bold(sbuf_t *sb, int lvl)
{
    if (lvl > 0)
    {
        sb_puts(sb, "\033[1m");
    }
}

/**
 * sb_rst() - Append SGR full attribute reset ("\033[0m") when ANSI styling is active
 * @sb:  Target stack string buffer (non-NULL)
 * @lvl: Effective color capability level (> 0 emits reset, <= 0 is a no-op)
 *
 * Role: Terminates active foreground, background, and bold attributes in @sb.
 *
 * Rationale: Explicit per-token resets prevent background pill fills or pupil half-block colors
 * from bleeding into trailing row padding.
 */
static void sb_rst(sbuf_t *sb, int lvl)
{
    if (lvl > 0)
    {
        sb_puts(sb, "\033[0m");
    }
}

/**
 * sb_pill() - Append a high-contrast solid-background badge (" LABEL " or "[LABEL]")
 * @sb:    Target stack string buffer (non-NULL)
 * @lvl:   Effective color capability level
 * @bg:    Solid pill fill color (foreground contrast ink is chosen automatically via rgb_on())
 * @label: Badge text (non-NULL)
 *
 * Role: Renders top-level panel title badges and primary status pills ("VALID DAG", "SHM LIVE",
 * "INVALID CONFIG").
 *
 * Rationale: In monochrome mode (@lvl <= 0), background color disappears; substituting bracket
 * delimiters "[LABEL]" preserves visual badge boundaries without relying on color.
 */
static void sb_pill(sbuf_t *sb, int lvl, cs_rgb_t bg, const char *label)
{
    if (lvl <= 0)
    {
        sb_printf(sb, "[%s]", label);
        return;
    }
    sb_bold(sb, lvl);
    sb_bg(sb, lvl, bg);
    sb_fg(sb, lvl, rgb_on(bg));
    sb_printf(sb, " %s ", label);
    sb_rst(sb, lvl);
}

/**
 * sb_soft_pill() - Append a subtle tinted-background optical element badge
 * @sb:       Target stack string buffer (non-NULL)
 * @lvl:      Effective color capability level
 * @bg_panel: Theme panel background color blended at 76% weight
 * @fg:       Semantic element accent color used for text and 24% background tint
 * @label:    Element kind tag (e.g. "DM", "TTM", "SPLITTER", "CORO", "PYRWFS", "FPCAM")
 *
 * Role: Renders inline hardware node pills throughout the 2D optical path schematic without
 * overwhelming primary alert/status pills.
 *
 * Rationale: Solid inverted pills on every optical node create excessive visual glare in dense
 * multi-stage benches; tinting @bg_panel toward @fg by 24% produces a modern, milk-CTRL-aligned
 * translucent chip effect on TrueColor/256c terminals while falling back to "[LABEL]" in mono.
 */
static void sb_soft_pill(sbuf_t *sb, int lvl, cs_rgb_t bg_panel, cs_rgb_t fg, const char *label)
{
    if (lvl <= 0)
    {
        sb_printf(sb, "[%s]", label);
        return;
    }
    cs_rgb_t tinted = rgb_lerp(bg_panel, fg, 0.24f);
    sb_bold(sb, lvl);
    sb_bg(sb, lvl, tinted);
    sb_fg(sb, lvl, fg);
    sb_printf(sb, " %s ", label);
    sb_rst(sb, lvl);
}

/**
 * vis_cols() - Count visible terminal columns in a mixed ANSI CSI + UTF-8 string
 * @s: NUL-terminated UTF-8 string potentially containing SGR escape sequences (non-NULL)
 *
 * Role: Computes the exact display-column width occupied on screen by @s so table columns and
 * right-hand panel borders align vertically regardless of embedded escape sequences or multi-byte
 * box-drawing glyphs.
 *
 * Rationale: strlen() overcounts both zero-width CSI sequences ("\033[...m") and 2/3-byte UTF-8
 * box/block/math glyphs (e.g. U+2502, U+2588, U+03BB). All non-ASCII glyphs emitted by this module
 * are narrow single-cell Unicode symbols (East Asian Width N/Na/A = 1 column), so skipping ECMA-48
 * CSI sequences (ESC '[' ... 0x40..0x7E) and counting UTF-8 codepoints yields exact column widths
 * without pulling in locale-dependent wcwidth().
 *
 * Return: Number of visible terminal columns >= 0.
 */
static int vis_cols(const char *s)
{
    int                  cols = 0;
    const unsigned char *p    = (const unsigned char *) s;
    while (*p != '\0')
    {
        /* Skip zero-width ECMA-48 CSI escape sequences: ESC '[' <params> <final 0x40..0x7E> */
        if (*p == 0x1B && p[1] == '[')
        {
            p += 2;
            while (*p != '\0' && (*p < 0x40 || *p > 0x7E))
            {
                p++;
            }
            if (*p != '\0')
            {
                p++;
            }
            continue;
        }
        /* Advance across 1-, 2-, 3-, or 4-byte UTF-8 codepoint sequences as 1 display column */
        if ((*p & 0x80u) == 0)
        {
            p += 1;
        }
        else if ((*p & 0xE0u) == 0xC0u && p[1] != '\0')
        {
            p += 2;
        }
        else if ((*p & 0xF0u) == 0xE0u && p[1] != '\0' && p[2] != '\0')
        {
            p += 3;
        }
        else if ((*p & 0xF8u) == 0xF0u && p[1] != '\0' && p[2] != '\0' && p[3] != '\0')
        {
            p += 4;
        }
        else
        {
            p += 1;
        }
        cols++;
    }
    return cols;
}

/**
 * emit_fg_fp() - Write an SGR foreground color escape sequence directly to @fp
 * @fp:  Destination output stream (non-NULL)
 * @lvl: Effective color capability level [0..3]
 * @c:   Target 24-bit sRGB border or frame color
 *
 * Role: Stream-level wrapper around sb_fg() used when drawing outer panel borders in panel_top(),
 * panel_sep(), panel_bot(), and panel_row().
 *
 * Rationale: Reuses sb_fg()'s exact tier quantization logic via a tiny stack buffer rather than
 * duplicating TrueColor / 256c / 16c SGR formatting for direct FILE* writes.
 */
static void emit_fg_fp(FILE *fp, int lvl, cs_rgb_t c)
{
    sbuf_t tmp;
    sb_init(&tmp);
    sb_fg(&tmp, lvl, c);
    fputs(tmp.data, fp);
}

/**
 * emit_rst_fp() - Write an SGR attribute reset ("\033[0m") directly to @fp when color is active
 * @fp:  Destination output stream (non-NULL)
 * @lvl: Effective color capability level (> 0 emits reset, <= 0 is a no-op)
 *
 * Role: Resets terminal attributes around panel border glyphs and after truncated row bodies.
 *
 * Rationale: Prevents unclosed foreground/background SGR states inside a clipped row from
 * discoloring the right-hand vertical border or subsequent terminal prompt.
 */
static void emit_rst_fp(FILE *fp, int lvl)
{
    if (lvl > 0)
    {
        fputs("\033[0m", fp);
    }
}

/**
 * sb_pad_col() - Append string @s followed by trailing spaces to reach at least @col_w columns
 * @sb:    Target stack string buffer (non-NULL)
 * @s:     Cell text (may contain UTF-8 or CSI sequences; non-NULL)
 * @col_w: Target column stride including at least 1 trailing space separator
 *
 * Role: Left-aligns fixed-width columns inside Panel 3 hardware summary tables.
 *
 * Rationale: printf("%-Ns", s) pads by byte count rather than visible glyph count, misaligning
 * columns whenever @s contains multi-byte UTF-8 symbols or ANSI color codes. Measuring @s with
 * vis_cols() guarantees exact visual column alignment and always leaves >= 1 space between
 * adjacent columns even when an identifier overflows @col_w.
 */
static void sb_pad_col(sbuf_t *sb, const char *s, int col_w)
{
    sb_puts(sb, s);
    int vc  = vis_cols(s);
    int pad = (col_w > vc) ? (col_w - vc) : 1;
    for (int i = 0; i < pad; i++)
    {
        sb_puts(sb, " ");
    }
}

/**
 * emit_truncated() - Write at most @max_cols visible columns of @s to @fp with ellipsis clipping
 * @fp:       Destination output stream (non-NULL)
 * @s:        Styled UTF-8 + ANSI CSI line body (non-NULL)
 * @max_cols: Maximum visible interior columns available inside the panel frame
 * @ascii:    1 to use "~" as the truncation marker, 0 to use UTF-8 horizontal ellipsis "…"
 *
 * Role: Safely clips overlong schematic rows (e.g. deeply nested splitter trees or long file
 * paths on narrow 76-col terminals) at clean UTF-8 codepoint boundaries while passing through
 * preceding CSI color escapes intact.
 *
 * Rationale: Byte-slicing a styled UTF-8 string can split a 3-byte UTF-8 character in half
 * (producing invalid UTF-8 replacement diamonds) or slice mid-CSI escape (printing raw ";255m"
 * fragments and breaking the right-hand border). Stepping codepoint-by-codepoint and reserving 1
 * column for "…"/"~" when total > max_cols guarantees intact escapes, valid UTF-8, and exact
 * column accounting.
 *
 * Return: Exact number of visible columns written to @fp (<= @max_cols).
 */
static int emit_truncated(FILE *fp, const char *s, int max_cols, int ascii)
{
    int                  total = vis_cols(s);
    int                  limit = (total > max_cols) ? (max_cols - 1) : max_cols;
    int                  shown = 0;
    const unsigned char *p     = (const unsigned char *) s;

    while (*p != '\0' && shown < limit)
    {
        /* Pass zero-width CSI color sequences straight through without incrementing @shown */
        if (*p == 0x1B && p[1] == '[')
        {
            const unsigned char *start = p;
            p += 2;
            while (*p != '\0' && (*p < 0x40 || *p > 0x7E))
            {
                p++;
            }
            if (*p != '\0')
            {
                p++;
            }
            fwrite(start, 1, (size_t) (p - start), fp);
            continue;
        }
        /* Emit one complete UTF-8 codepoint as a single visible column */
        size_t clen = 1;
        if ((*p & 0x80u) == 0)
        {
            clen = 1;
        }
        else if ((*p & 0xE0u) == 0xC0u && p[1] != '\0')
        {
            clen = 2;
        }
        else if ((*p & 0xF0u) == 0xE0u && p[1] != '\0' && p[2] != '\0')
        {
            clen = 3;
        }
        else if ((*p & 0xF8u) == 0xF0u && p[1] != '\0' && p[2] != '\0' && p[3] != '\0')
        {
            clen = 4;
        }
        fwrite(p, 1, clen, fp);
        p += clen;
        shown++;
    }
    if (total > max_cols && max_cols >= 1)
    {
        fputs(ascii ? "~" : "…", fp);
        shown++;
    }
    return shown;
}

/**
 * struct view_env_t - Resolved per-render terminal, theme, geometry, and SHM environment state
 * @fp:        Destination output FILE stream
 * @pal:       Active semantic color palette pointer into g_palettes[]
 * @theme_idx: Validated palette index in [0..CACAOSIM_THEME_COUNT-1]
 * @lvl:       Resolved SGR color tier in [0..3]
 * @ascii:     1 for pure 7-bit ASCII glyphs, 0 for UTF-8 box-drawing & blocks
 * @check_shm: 1 to probe @shm_dir for live <stream>.im.shm files, 0 to skip filesystem probes
 * @width:     Resolved outer panel frame width in terminal columns
 * @shm_dir:   Resolved ImageStreamIO shared-memory directory (MILK_SHM_DIR or "/milk/shm")
 *
 * Role: Immutable rendering context initialized once per cacaosim_bench_render*() invocation by
 * init_view_env() and threaded by const pointer through all panel, tree, and badge helpers.
 *
 * Rationale: Resolving getenv(), isatty(), and ioctl(TIOCGWINSZ) once upfront avoids repeated
 * syscalls across dozens of rows and guarantees that every panel in a snapshot shares identical
 * width and color/glyph rules even if the terminal resizes mid-frame.
 */
typedef struct
{
    FILE               *fp;
    const cs_palette_t *pal;
    int                 theme_idx;
    int                 lvl;
    int                 ascii;
    int                 check_shm;
    int                 width;
    char                shm_dir[512];
} view_env_t;

/**
 * panel_top() - Emit the top rounded/ASCII border of a framed schematic panel with optional badge
 * @ev:       Active rendering environment (non-NULL)
 * @title:    Optional panel header title rendered inside an inverted pill (NULL or "" omits badge)
 * @border_c: RGB color for the horizontal rule and corner box-drawing glyphs
 * @badge_c:  Solid fill RGB color for the @title pill
 *
 * Role: Opens a visual section card (rounded UTF-8 corners or "+- [TITLE] ----+" in ASCII)
 * spanning exactly @ev->width display columns.
 *
 * Rationale: Measuring the formatted pill with vis_cols() ensures the trailing horizontal border
 * fills the exact remaining width in both colored (" TITLE ") and monochrome ("[TITLE]") modes.
 */
static void panel_top(const view_env_t *ev, const char *title, cs_rgb_t border_c, cs_rgb_t badge_c)
{
    const char *tl = ev->ascii ? "+" : "╭";
    const char *tr = ev->ascii ? "+" : "╮";
    const char *hz = ev->ascii ? "-" : "─";

    emit_fg_fp(ev->fp, ev->lvl, border_c);
    fputs(tl, ev->fp);
    fputs(hz, ev->fp);
    emit_rst_fp(ev->fp, ev->lvl);

    int used = 2;
    if (title != NULL && title[0] != '\0')
    {
        sbuf_t tb;
        sb_init(&tb);
        sb_pill(&tb, ev->lvl, badge_c, title);
        int tcols = vis_cols(tb.data);
        fputs(tb.data, ev->fp);
        used += tcols;
    }

    emit_fg_fp(ev->fp, ev->lvl, border_c);
    int rem = ev->width - used - 1;
    for (int i = 0; i < rem; i++)
    {
        fputs(hz, ev->fp);
    }
    fputs(tr, ev->fp);
    emit_rst_fp(ev->fp, ev->lvl);
    fputc('\n', ev->fp);
}

/**
 * panel_sep() - Emit an interior horizontal divider rule spanning @ev->width columns
 * @ev:       Active rendering environment (non-NULL)
 * @border_c: RGB color for T-junctions ("├"/"┤" or "+") and horizontal rule
 *
 * Role: Separates column headers from data rows in Panel 3 hardware tables and divides summary
 * banners from diagnostic details in error panels.
 *
 * Rationale: Dedicated T-junction glyphs visually bind multi-column hardware tables into the outer
 * card frame without breaking vertical border continuity.
 */
static void panel_sep(const view_env_t *ev, cs_rgb_t border_c)
{
    const char *lt = ev->ascii ? "+" : "├";
    const char *rt = ev->ascii ? "+" : "┤";
    const char *hz = ev->ascii ? "-" : "─";

    emit_fg_fp(ev->fp, ev->lvl, border_c);
    fputs(lt, ev->fp);
    for (int i = 0; i < ev->width - 2; i++)
    {
        fputs(hz, ev->fp);
    }
    fputs(rt, ev->fp);
    emit_rst_fp(ev->fp, ev->lvl);
    fputc('\n', ev->fp);
}

/**
 * panel_bot() - Emit the bottom rounded/ASCII border closing a framed schematic panel
 * @ev:       Active rendering environment (non-NULL)
 * @border_c: RGB color for bottom corners ("╰"/"╯" or "+") and horizontal rule
 *
 * Role: Closes a framed visual card spanning @ev->width columns.
 *
 * Rationale: Pairing rounded UTF-8 corners (U+256D..U+2570) with milk's slate palette matches
 * milk-CTRL card aesthetics while degrading cleanly to "+---+" in 7-bit ASCII mode.
 */
static void panel_bot(const view_env_t *ev, cs_rgb_t border_c)
{
    const char *bl = ev->ascii ? "+" : "╰";
    const char *br = ev->ascii ? "+" : "╯";
    const char *hz = ev->ascii ? "-" : "─";

    emit_fg_fp(ev->fp, ev->lvl, border_c);
    fputs(bl, ev->fp);
    for (int i = 0; i < ev->width - 2; i++)
    {
        fputs(hz, ev->fp);
    }
    fputs(br, ev->fp);
    emit_rst_fp(ev->fp, ev->lvl);
    fputc('\n', ev->fp);
}

/**
 * panel_row() - Emit one left/right-bordered content row clipped and padded to @ev->width columns
 * @ev:       Active rendering environment (non-NULL)
 * @border_c: RGB color for the left and right vertical border glyphs ("│" or "|")
 * @line:     Styled row content string (NULL is treated as an empty row)
 *
 * Role: Wraps any composed sbuf_t line inside "| <inner_w cols> |", truncating with an ellipsis
 * if wider than (ev->width - 4) or right-padding with spaces so the right border always lands on
 * column ev->width.
 *
 * Rationale: Centralizing left/right framing, SGR state reset, UTF-8-safe clipping, and trailing
 * whitespace padding inside a single bottleneck makes vertical border misalignment impossible
 * regardless of what upstream node/table formatters place in @line.
 */
static void panel_row(const view_env_t *ev, cs_rgb_t border_c, const char *line)
{
    const char *vt      = ev->ascii ? "|" : "│";
    int         inner_w = ev->width - 4;

    emit_fg_fp(ev->fp, ev->lvl, border_c);
    fputs(vt, ev->fp);
    emit_rst_fp(ev->fp, ev->lvl);
    fputc(' ', ev->fp);

    int shown = emit_truncated(ev->fp, (line != NULL) ? line : "", inner_w, ev->ascii);
    emit_rst_fp(ev->fp, ev->lvl);
    for (int i = shown; i < inner_w; i++)
    {
        fputc(' ', ev->fp);
    }

    fputc(' ', ev->fp);
    emit_fg_fp(ev->fp, ev->lvl, border_c);
    fputs(vt, ev->fp);
    emit_rst_fp(ev->fp, ev->lvl);
    fputc('\n', ev->fp);
}

/**
 * shm_stream_exists() - Non-intrusively probe whether an ImageStreamIO SHM file exists on disk
 * @ev:          Active rendering environment supplying @check_shm and @shm_dir (non-NULL)
 * @stream_name: Logical milk stream name without directory or ".im.shm" suffix (NULL-safe)
 *
 * Role: Checks whether "<ev->shm_dir>/<stream_name>.im.shm" is currently materialized in shared
 * memory so schematic nodes and tables can light up live deployment badges.
 *
 * Rationale: Calling ImageStreamIO_read_sharedmem_image_toIMAGE() during a passive CLI preview
 * would map POSIX SHM pages, allocate file descriptors, and risk permission failures on another
 * user's loop. A pure access(..., F_OK) metadata probe completes in microseconds with zero side
 * effects on running real-time CACAO processes.
 *
 * Return: 1 if @ev->check_shm is enabled and the .im.shm file exists, 0 otherwise.
 */
static int shm_stream_exists(const view_env_t *ev, const char *stream_name)
{
    if (!ev->check_shm || stream_name == NULL || stream_name[0] == '\0')
    {
        return 0;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.im.shm", ev->shm_dir, stream_name);
    return (access(path, F_OK) == 0) ? 1 : 0;
}

/**
 * sb_stream_tag() - Append a formatted SHM stream identifier with optional live/standby glyph
 * @sb:   Target stack string buffer (non-NULL)
 * @ev:   Active rendering environment (non-NULL)
 * @name: Stream name string (NULL or "" emits a dimmed "-" placeholder)
 *
 * Role: Renders every SHM endpoint (entrance wf/pup/ncpa, DM in_cmd, WFE map, coronagraph SHM
 * masks, and WFS/camera out_cam) consistently across Overview, Schematic, and Hardware Tables.
 *
 * Rationale: Pairing a green filled dot ("●" / "*") for active streams vs a dimmed hollow ring
 * ("○" / "o") for uncreated streams lets operators verify at a glance which hardware endpoints
 * are already live in MILK_SHM_DIR and which remain offline.
 */
static void sb_stream_tag(sbuf_t *sb, const view_env_t *ev, const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        sb_fg(sb, ev->lvl, ev->pal->fg_dim);
        sb_puts(sb, "-");
        sb_rst(sb, ev->lvl);
        return;
    }
    int live = shm_stream_exists(ev, name);
    if (ev->check_shm)
    {
        sb_fg(sb, ev->lvl, live ? ev->pal->fg_active : ev->pal->fg_dim);
        sb_puts(sb, ev->ascii ? (live ? "*" : "o") : (live ? "●" : "○"));
        sb_puts(sb, " ");
    }
    sb_bold(sb, ev->lvl);
    sb_fg(sb, ev->lvl, ev->pal->fg_stream);
    sb_puts(sb, name);
    sb_rst(sb, ev->lvl);
}

/**
 * lambda_color() - Map an optical/NIR wavelength [0.40..2.40 um] to a theme-aware RGB color
 * @ev:     Active rendering environment (controls Night and Accessible overrides; non-NULL)
 * @lam_um: Wavelength in micrometers [um]
 *
 * Role: Assigns physically intuitive false-color hues to spectral sparkline bins, sensor central
 * wavelengths, and passband columns across the visible-to-K-band AO operating range.
 *
 * Rationale: Standard themes interpolate smoothly across 7 chromaticity knots spanning visible
 * blue/cyan/green/orange-red (0.40..0.70 um) into distinct far-red/rose/violet-magenta NIR bands
 * (Y, J/H, K up to 2.40 um) so dichroic splits are immediately legible. Under Night theme, blue
 * and green wavelengths are suppressed in favor of a bright-to-deep red ramp to protect dark
 * adaptation; under Accessible theme, 5 discrete Okabe-Ito bands guarantee colorblind separation.
 *
 * Return: Theme-appropriate 24-bit cs_rgb_t color for @lam_um.
 */
static cs_rgb_t lambda_color(const view_env_t *ev, float lam_um)
{
    /* Night-vision override: monotonic amber-red to deep-crimson ramp across 0.40..2.40 um */
    if (ev->theme_idx == CACAOSIM_THEME_NIGHT)
    {
        float    t  = (lam_um - 0.40f) / 2.00f;
        cs_rgb_t c0 = { 255, 115, 70 };
        cs_rgb_t c1 = { 165, 35, 30 };
        return rgb_lerp(c0, c1, t);
    }
    /* Accessible override: quantized Okabe-Ito astronomical passband colors (Vis, I/Y, J, H, K) */
    if (ev->theme_idx == CACAOSIM_THEME_ACCESSIBLE)
    {
        if (lam_um < 0.70f)
        {
            cs_rgb_t c = { 86, 180, 233 };
            return c;
        }
        if (lam_um < 1.05f)
        {
            cs_rgb_t c = { 0, 158, 115 };
            return c;
        }
        if (lam_um < 1.45f)
        {
            cs_rgb_t c = { 240, 228, 66 };
            return c;
        }
        if (lam_um < 1.90f)
        {
            cs_rgb_t c = { 230, 159, 0 };
            return c;
        }
        cs_rgb_t c = { 213, 94, 0 };
        return c;
    }

    /* Continuous 7-knot visible + false-color NIR chromaticity ramp */
    cs_rgb_t stops[7] = {
        { 115, 115, 255 }, /* 0.40 um Violet-Blue */
        { 65, 205, 235 },  /* 0.50 um Cyan-Green  */
        { 130, 230, 95 },  /* 0.58 um Yellow-Green */
        { 255, 180, 65 },  /* 0.70 um Orange-Red  */
        { 245, 95, 85 },   /* 0.95 um Far-Red / Y */
        { 235, 130, 155 }, /* 1.45 um J/H-band    */
        { 185, 95, 215 },  /* 2.40 um K-band      */
    };
    float knots[7] = { 0.40f, 0.50f, 0.58f, 0.70f, 0.95f, 1.45f, 2.40f };

    if (lam_um <= knots[0])
    {
        return stops[0];
    }
    for (int i = 0; i < 6; i++)
    {
        if (lam_um <= knots[i + 1])
        {
            float t = (lam_um - knots[i]) / (knots[i + 1] - knots[i]);
            return rgb_lerp(stops[i], stops[i + 1], t);
        }
    }
    return stops[6];
}

/**
 * sb_spectrum_bar() - Append a wavelength-colored spectral photon-rate sparkline across 0.40..2.40u
 * @sb:       Target stack string buffer (non-NULL)
 * @ev:       Active rendering environment (non-NULL)
 * @spec:     32-bin photon spectral density [ph/s/bin] for the optical beam (non-NULL)
 * @ref_peak: Reference peak bin flux [ph/s/bin] (typically entrance beams[0] peak) used for
 *            consistent vertical normalization across downstream split arms
 * @nbins:    Number of display columns to resample into (e.g. 32 in Overview, 24 on splitter arms)
 *
 * Role: Visualizes spectral energy distribution and dichroic cutoffs inline on every optical arm
 * using 8-step Unicode vertical blocks ("▁".."█") or 7-bit ASCII density chars ('.'..'@').
 *
 * Rationale: Normalizing every downstream arm against the entrance @ref_peak (rather than each
 * arm's local peak) makes gray attenuation, coronagraphic Lyot losses, and complementary LP/SP/BP
 * dichroic notches directly comparable down the entire DAG. Applying a square-root gamma curve
 * (g = sqrtf(rel)) compresses dynamic range so Rayleigh-Jeans K-band tails and ~5% out-of-band
 * dichroic leakage remain visible alongside peak stellar emission, while bins < 0.2% drop to a
 * baseline dot ("·" / ' ') to clearly show rejected stopbands.
 */
static void sb_spectrum_bar(
    sbuf_t                *sb,
    const view_env_t      *ev,
    const cacaosim_spec_t *spec,
    float                  ref_peak,
    int                    nbins)
{
    static const char *const u8_bars[8] = { "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
    static const char        as_bars[8] = { '.', '_', '-', '=', '+', '*', '#', '@' };

    if (ref_peak <= 0.0f)
    {
        ref_peak = 1.0f;
    }

    sb_fg(sb, ev->lvl, ev->pal->fg_dim);
    sb_puts(sb, ev->ascii ? "[" : "▐");

    for (int i = 0; i < nbins; i++)
    {
        /* Box-average the underlying 32-bin spectrum onto @nbins display columns */
        int   b0  = (i * CACAOSIM_NSPEC) / nbins;
        int   b1  = ((i + 1) * CACAOSIM_NSPEC) / nbins;
        float acc = 0.0f;
        int   cnt = 0;
        for (int b = b0; b < b1; b++)
        {
            acc += spec->ph_s[b];
            cnt++;
        }
        float val = (cnt > 0) ? (acc / (float) cnt) : spec->ph_s[b0];
        float rel = val / ref_peak;
        float lam = CACAOSIM_SPEC_L0_UM + ((float) i + 0.5f) * (2.00f / (float) nbins);

        /* Render rejected dichroic stopbands (< 0.2% of entrance peak) as a dim baseline marker */
        if (rel < 0.002f)
        {
            sb_fg(sb, ev->lvl, ev->pal->fg_dim);
            sb_puts(sb, ev->ascii ? " " : "·");
        }
        else
        {
            /* Apply sqrt gamma compression to map [0.002..1.0] onto 8 vertical bar heights */
            float g   = sqrtf(rel > 1.0f ? 1.0f : rel);
            int   idx = (int) (g * 7.99f);
            if (idx < 0)
            {
                idx = 0;
            }
            if (idx > 7)
            {
                idx = 7;
            }
            sb_fg(sb, ev->lvl, lambda_color(ev, lam));
            if (ev->ascii)
            {
                char ch[2] = { as_bars[idx], '\0' };
                sb_puts(sb, ch);
            }
            else
            {
                sb_puts(sb, u8_bars[idx]);
            }
        }
    }

    sb_fg(sb, ev->lvl, ev->pal->fg_dim);
    sb_puts(sb, ev->ascii ? "]" : "▌");
    sb_rst(sb, ev->lvl);
}

/**
 * sb_pupil_mini_row() - Render one terminal row of a 16x16 anti-aliased telescope pupil thumbnail
 * @sb:    Target stack string buffer receiving 16 display columns (non-NULL)
 * @ev:    Active rendering environment (non-NULL)
 * @pup16: Row-major 16x16 float pupil amplitude array in [0.0..1.0] from cacaosim_make_pupil()
 * @r:     Terminal row index in [0..7] (0 = top edge of pupil, 7 = bottom edge)
 *
 * Role: Packs two vertical rows of the 16x16 telescope pupil (outer diameter, central obscuration,
 * and secondary support spiders) into a single 16-column terminal row so the 8x16 thumbnail
 * displays with a ~1:1 circular aspect ratio on standard ~1:2 aspect terminal fonts.
 *
 * Rationale: Terminal character cells are roughly twice as tall as they are wide. By pairing
 * y_top = 15 - 2*r (foreground of Unicode upper-half block U+2580) with y_bot = 14 - 2*r
 * (background of U+2580) in +Y-up astronomical coordinates and blending both colors between
 * bg_panel and fg_stream on 256c/TrueColor terminals, central obstructions and angled spiders
 * appear crisp and anti-aliased. On 16c/mono/ASCII terminals, dual-threshold glyph quantization
 * preserves sub-row geometric fidelity without requiring RGB backgrounds.
 */
static void sb_pupil_mini_row(
    sbuf_t           *sb,
    const view_env_t *ev,
    const float      *pup16,
    int               r)
{
    /* Map terminal row r [0..7] top-down onto +Y-up astronomical pupil rows [15..0] */
    int y_top = 15 - 2 * r;
    int y_bot = 14 - 2 * r;

    for (int x = 0; x < 16; x++)
    {
        float at = pup16[y_top * 16 + x];
        float ab = pup16[y_bot * 16 + x];

        if (ev->ascii)
        {
            /* 7-bit ASCII sub-row fallback distinguishing upper/lower/full/edge illumination */
            char ch = ' ';
            if (at >= 0.45f && ab >= 0.45f)
            {
                ch = '#';
            }
            else if (at >= 0.45f)
            {
                ch = '^';
            }
            else if (ab >= 0.45f)
            {
                ch = '_';
            }
            else if (at >= 0.12f || ab >= 0.12f)
            {
                ch = '.';
            }
            sb_fg(sb, ev->lvl, (ch == '#') ? ev->pal->fg_stream : ev->pal->fg_dim);
            char s[2] = { ch, '\0' };
            sb_puts(sb, s);
        }
        else if (ev->lvl >= 2)
        {
            /* TrueColor / 256c: continuous alpha blend on upper (FG) & lower (BG) half-pixels */
            cs_rgb_t ct = rgb_lerp(ev->pal->bg_panel, ev->pal->fg_stream, at);
            cs_rgb_t cb = rgb_lerp(ev->pal->bg_panel, ev->pal->fg_stream, ab);
            sb_fg(sb, ev->lvl, ct);
            sb_bg(sb, ev->lvl, cb);
            sb_puts(sb, "▀");
        }
        else
        {
            /* Monochrome / 16c UTF-8 fallback using thresholded block-element glyphs */
            const char *gl = " ";
            if (at >= 0.45f && ab >= 0.45f)
            {
                gl = "█";
            }
            else if (at >= 0.45f)
            {
                gl = "▀";
            }
            else if (ab >= 0.45f)
            {
                gl = "▄";
            }
            else if (at >= 0.12f || ab >= 0.12f)
            {
                gl = "·";
            }
            sb_fg(sb, ev->lvl, ev->pal->fg_stream);
            sb_puts(sb, gl);
        }
    }
    sb_rst(sb, ev->lvl);
}

/**
 * spectral_type_hint() - Return approximate Morgan-Keenan stellar spectral class for @teff_k
 * @teff_k: Stellar effective blackbody temperature in Kelvin [K]
 *
 * Role: Annotates the Panel 1 stellar parameter line (e.g. "Teff=5778 K (G-type)") so operators
 * immediately associate numeric blackbody temperature with familiar AO guide-star color classes.
 *
 * Rationale: Extreme-AO and multi-stage Visible/NIR WFS photon splits depend strongly on whether
 * the target is a red M-dwarf (< 3700 K, NIR-rich / Vis-starved) vs a solar G-star or hot A-star.
 *
 * Return: Static short spectral class string ("A/B", "F", "G", "K", or "M").
 */
static const char *spectral_type_hint(float teff_k)
{
    if (teff_k >= 9500.0f)
    {
        return "A/B";
    }
    if (teff_k >= 6000.0f)
    {
        return "F";
    }
    if (teff_k >= 5200.0f)
    {
        return "G";
    }
    if (teff_k >= 3700.0f)
    {
        return "K";
    }
    return "M";
}

/**
 * sb_reg_badge() - Append a warning-colored 2D pupil misregistration badge when non-identity
 * @sb:  Target stack string buffer (non-NULL)
 * @ev:  Active rendering environment (non-NULL)
 * @reg: 2D affine pupil registration parameters (shift_x, shift_y [pix], rot_deg, scale)
 *
 * Role: Highlights intentional or calibrated DM, WFE, splitter-arm, and WFS pupil shifts,
 * rotations, and magnifications directly on the 2D schematic row.
 *
 * Rationale: Suppresses clutter when registration is exact identity (the common nominal case),
 * while surfacing non-zero shifts/rotations/magnifications in warning amber because uncalibrated
 * pupil misregistration is a primary driver of AO loop instability.
 */
static void sb_reg_badge(sbuf_t *sb, const view_env_t *ev, const cacaosim_reg_t *reg)
{
    if (cacaosim_reg_is_identity(reg))
    {
        return;
    }
    sb_puts(sb, " ");
    sb_fg(sb, ev->lvl, ev->pal->fg_warn);
    float sc = (reg->scale > 0.0f) ? reg->scale : 1.0f;
    if (fabsf(sc - 1.0f) > 1e-4f)
    {
        sb_printf(sb, "[reg dx=%+.2f dy=%+.2f rot=%+.1f%s mag=%.3f]", reg->shift_x, reg->shift_y,
                  reg->rot_deg, ev->ascii ? "deg" : "°", sc);
    }
    else
    {
        sb_printf(sb, "[reg dx=%+.2f dy=%+.2f rot=%+.1f%s]", reg->shift_x, reg->shift_y,
                  reg->rot_deg, ev->ascii ? "deg" : "°");
    }
    sb_rst(sb, ev->lvl);
}

/**
 * format_splitter_desc() - Format a concise optical specification string for a beamsplitter node
 * @sp:    Beamsplitter configuration (gray, dichroic_lp, dichroic_sp, or dichroic_bp; non-NULL)
 * @ascii: 1 to emit "+/-" for bandpass tolerances, 0 to emit UTF-8 "±"
 * @out:   Destination character buffer (non-NULL)
 * @cap:   Capacity of @out in bytes
 *
 * Role: Summarizes reflectance/transmittance or cutoff wavelength semantics on SPLITTER headers
 * in Panel 2.
 *
 * Rationale: Explicitly spelling out which wavelengths reflect vs transmit (e.g. "refl < 1.00,
 * trans > 1.00" for a long-pass dichroic) prevents sign-convention confusion between long-pass,
 * short-pass, and notch band-reflect splitters.
 */
static void format_splitter_desc(const cacaosim_split_cfg_t *sp, int ascii, char *out, size_t cap)
{
    const char *pm = ascii ? "+/-" : "±";
    if (sp->mode == CACAOSIM_SPLIT_GRAY)
    {
        snprintf(out, cap, "GRAY  R=%.0f%%  T=%.0f%%", 100.0f * sp->r_gray, 100.0f * sp->t_gray);
    }
    else if (sp->mode == CACAOSIM_SPLIT_DICHROIC_LP)
    {
        snprintf(out, cap, "LONG-PASS  cut=%.2f um  (refl < %.2f, trans > %.2f)", sp->lambda_cut,
                 sp->lambda_cut, sp->lambda_cut);
    }
    else if (sp->mode == CACAOSIM_SPLIT_DICHROIC_SP)
    {
        snprintf(out, cap, "SHORT-PASS  cut=%.2f um  (refl > %.2f, trans < %.2f)", sp->lambda_cut,
                 sp->lambda_cut, sp->lambda_cut);
    }
    else
    {
        snprintf(out, cap, "BAND-REFLECT  %.2f %s %.2f um  (R=%.0f%% T=%.0f%%)", sp->lambda_cut,
                 pm, 0.5f * sp->band_width, 100.0f * sp->eff_refl, 100.0f * sp->eff_trans);
    }
}

/**
 * sb_mask_spec_tag() - Append a coronagraph custom mask key=value tag for FITS or SHM specifiers
 * @sb:      Target stack string buffer (non-NULL)
 * @ev:      Active rendering environment (non-NULL)
 * @key:     Plane/component label (e.g. "pup_amp", "fpm_pha", "lyot_amp")
 * @spec:    Configured FITS path or SHM stream name (empty/NULL returns 0 without emitting text)
 * @need_sp: Non-zero to prepend a two-space separator when prior mask tags were already emitted
 *
 * Role: Renders custom coronagraphic pupil, focal-plane, and Lyot amplitude/phase mask bindings
 * beneath CORO headers in Panel 2.
 *
 * Rationale: Coronagraph masks in cacaosim can be either static disk FITS cubes (.fits / .fit /
 * path containing '/') or live ImageStreamIO SHM streams. Stripping directory prefixes on FITS
 * files keeps the schematic row compact, whereas delegating SHM specifiers to sb_stream_tag()
 * provides live MILK_SHM_DIR existence badges on dynamic masks.
 *
 * Return: 1 if @spec was non-empty and a tag was appended, 0 otherwise.
 */
static int sb_mask_spec_tag(
    sbuf_t           *sb,
    const view_env_t *ev,
    const char       *key,
    const char       *spec,
    int               need_sp)
{
    if (spec == NULL || spec[0] == '\0')
    {
        return 0;
    }
    if (need_sp)
    {
        sb_puts(sb, "  ");
    }
    sb_fg(sb, ev->lvl, ev->pal->fg_dim);
    sb_printf(sb, "%s=", key);
    if (cacaosim_mask_is_fits_path(spec))
    {
        const char *bn = strrchr(spec, '/');
        sb_bold(sb, ev->lvl);
        sb_fg(sb, ev->lvl, ev->pal->fg_bright);
        sb_puts(sb, (bn != NULL) ? (bn + 1) : spec);
        sb_rst(sb, ev->lvl);
    }
    else
    {
        sb_stream_tag(sb, ev, spec);
    }
    return 1;
}

/**
 * coro_mode_label() - Return uppercase display label for coronagraph optical architecture @mode
 * @mode: Coronagraph architecture enum value (cacaosim_coro_mode_t: 0..5)
 *
 * Role: Converts numeric coronagraph mode codes into concise schematic and table labels.
 *
 * Rationale: Distinguishes analytic focal masks (classical Lyot spot, phase dimple, optical
 * vortex, 4QPM) from user-supplied focal masks propagated via Matrix-DFT ("CUSTOM MFT") or padded
 * FFT ("CUSTOM FFT").
 *
 * Return: Static uppercase C string describing the coronagraph mode.
 */
static const char *coro_mode_label(int mode)
{
    switch (mode)
    {
    case 0:
        return "LYOT SPOT";
    case 1:
        return "PHASE DIMPLE";
    case 2:
        return "VORTEX";
    case 3:
        return "4QPM";
    case 4:
        return "CUSTOM MFT";
    case 5:
        return "CUSTOM FFT";
    default:
        return "CORO";
    }
}

/**
 * render_sensor_inline() - Format a terminal WFS or science camera leaf node onto @sb
 * @sb:  Target stack string buffer (non-NULL)
 * @ev:  Active rendering environment (non-NULL)
 * @ctx: Analyzed optical bench context providing master_hz (non-NULL)
 * @nd:  Sensor node instance (PYRWFS, SHWFS, ZWFS, or FPCAM; non-NULL)
 *
 * Role: Composes the compact single-line summary for leaf detector nodes at the end of optical
 * branches in Panel 2: icon + soft pill + node ID + geometry (NxNsub or NxN pix) + wavelength +
 * decimated frame rate ("Hz(/navg)") + integrated photons/frame + output SHM stream + reg badge.
 *
 * Rationale: Placing geometry, wavelength color, frame-rate decimation factor (/navg), integrated
 * photon count per exposure, and live output stream name inline at each branch tip lets the reader
 * trace an entire multi-rate photon budget directly on the 2D schematic without cross-referencing
 * Panel 3 tables.
 */
static void render_sensor_inline(
    sbuf_t                      *sb,
    const view_env_t            *ev,
    const cacaosim_bench_ctx_t  *ctx,
    const cacaosim_bench_node_t *nd)
{
    const char *sep  = ev->ascii ? " | " : " │ ";
    const char *arr  = ev->ascii ? " --> " : " ──▶ ";
    float       m_hz = (ctx->eff_master_hz > 0.1f) ? ctx->eff_master_hz : ctx->cfg.master_hz;
    float       hz   = (nd->eff_navg > 0) ? (m_hz / (float) nd->eff_navg) : 0.0f;

    /* Select sensor-class glyph and semantic accent color (green=zonal/slope WFS, purple=ZWFS,
     * gold=focal-plane science camera) */
    if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
    {
        sb_fg(sb, ev->lvl, ev->pal->fg_active);
        sb_puts(sb, ev->ascii ? "# " : "▣ ");
        sb_soft_pill(sb, ev->lvl, ev->pal->bg_panel, ev->pal->fg_active, "PYRWFS");
    }
    else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
    {
        sb_fg(sb, ev->lvl, ev->pal->fg_active);
        sb_puts(sb, ev->ascii ? "+ " : "⊞ ");
        sb_soft_pill(sb, ev->lvl, ev->pal->bg_panel, ev->pal->fg_active, "SHWFS");
    }
    else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
    {
        sb_fg(sb, ev->lvl, ev->pal->fg_proc);
        sb_puts(sb, ev->ascii ? "o " : "◉ ");
        sb_soft_pill(sb, ev->lvl, ev->pal->bg_panel, ev->pal->fg_proc, "ZWFS");
    }
    else
    {
        sb_fg(sb, ev->lvl, ev->pal->fg_accent);
        sb_puts(sb, ev->ascii ? "* " : "◆ ");
        sb_soft_pill(sb, ev->lvl, ev->pal->bg_panel, ev->pal->fg_accent, "FPCAM");
    }

    sb_puts(sb, " ");
    sb_bold(sb, ev->lvl);
    sb_fg(sb, ev->lvl, ev->pal->fg_bright);
    sb_printf(sb, "%-8s", nd->cfg.id);
    sb_rst(sb, ev->lvl);

    sb_fg(sb, ev->lvl, ev->pal->fg_text);
    if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
    {
        sb_printf(sb, " %dx%dsub", nd->cfg.shwfs.n_sub, nd->cfg.shwfs.n_sub);
    }
    else
    {
        sb_printf(sb, " %ux%u", nd->ncam, nd->ncam);
    }
    sb_fg(sb, ev->lvl, ev->pal->fg_dim);
    sb_puts(sb, sep);
    sb_fg(sb, ev->lvl, lambda_color(ev, nd->eff_lambda_um));
    sb_printf(sb, "%.2fum", nd->eff_lambda_um);
    sb_fg(sb, ev->lvl, ev->pal->fg_dim);
    sb_puts(sb, sep);
    sb_fg(sb, ev->lvl, ev->pal->fg_text);
    if (nd->eff_navg > 1u)
    {
        sb_printf(sb, "%.0fHz(/%u)", hz, nd->eff_navg);
    }
    else
    {
        sb_printf(sb, "%.0fHz", hz);
    }
    sb_fg(sb, ev->lvl, ev->pal->fg_dim);
    sb_puts(sb, sep);
    sb_fg(sb, ev->lvl, ev->pal->fg_muted);
    sb_printf(sb, "%.2e ph", nd->eff_flux_ph);

    sb_fg(sb, ev->lvl, ev->pal->fg_conn);
    sb_puts(sb, arr);
    sb_stream_tag(sb, ev, nd->cfg.out_cam);
    sb_reg_badge(sb, ev, &nd->cfg.reg);
}

/**
 * render_beam_subtree() - Recursively render the 2D optical path & light-splitting tree from @b_idx
 * @ev:           Active rendering environment (non-NULL)
 * @ctx:          Analyzed optical bench context containing beams[] and sorted nodes[] (non-NULL)
 * @b_idx:        Current optical beam index in ctx->beams[] ([0..ctx->n_beams-1])
 * @prefix:       Accumulated left-margin indentation and vertical tree guide bars ("|     ")
 * @src_tot:      Entrance beam integrated photon flux [ph/s] for throughput percentage scaling
 * @src_peak:     Entrance beam peak spectral bin flux [ph/s/bin] for sparkline vertical scaling
 * @visited_beam: Per-beam recursion guard bitmap [CACAOSIM_BENCH_MAX_BEAMS] preventing cycles
 * @depth:        Current recursion depth (hard-capped at 24 levels as a defensive bound)
 *
 * Role: Core layout algorithm for Panel 2 ("OPTICAL PATH & LIGHT-SPLITTING SCHEMATIC"). Discovers
 * all downstream nodes consuming @b_idx and dispatches across four optical topology idioms:
 *   - Unterminated beam (0 consumers): emits a dimmed leaf warning so dangling beams stand out.
 *   - Case 1 / 1b (Single inline DM, TTM, or static WFE): renders the element directly along the
 *     current vertical optical axis WITHOUT adding horizontal indentation, then recurses onto
 *     nd->out_idx using the unchanged @prefix.
 *   - Case 2 (Single Beamsplitter): renders the SPLITTER header, then branches laterally into
 *     reflected ("refl") and transmitted ("trans") arms with photon-share %, 24-bin spectral
 *     sparklines, and child continuation prefixes (vertical bar for non-last arms, spaces for
 *     last).
 *   - Case 3 (Single Coronagraph): renders the CORO header and optional custom FITS/SHM mask row,
 *     then branches into the optional reflective LOWFS pickoff arm ("lowfs(FPM)" or "lowfs(Lyot)")
 *     and the post-Lyot science arm ("lyot").
 *   - Case 4 (Leaf WFS/FPCAM sensors or multi-consumer fanout): emits terminal sensor summaries
 *     via render_sensor_inline() or general fanout branches.
 *
 * Rationale: Naive generic tree printers indent on every hop, causing a 4-element serial optical
 * train (TTM -> Woofer -> WFE -> Tweeter) to Stair-Step diagonally across the screen and exhaust
 * terminal width before reaching the first dichroic. Keeping 1-in/1-out pupil-plane relays
 * collinear on the vertical optical axis while indenting strictly at physical beam-splitting
 * junctions mirrors how optical engineers draw bench schematics and preserves horizontal room for
 * multi-stage cascaded AO branches.
 */
static void render_beam_subtree(
    const view_env_t           *ev,
    const cacaosim_bench_ctx_t *ctx,
    int                         b_idx,
    const char                 *prefix,
    float                       src_tot,
    float                       src_peak,
    uint8_t                    *visited_beam,
    int                         depth)
{
    /* Defensive bounds & cycle guard (cacaosim_bench_analyze() enforces DAG acyclicity, but
     * guarding visited_beam[] and depth <= 24 makes the renderer safe on synthetic contexts) */
    if (b_idx < 0 || (uint32_t) b_idx >= ctx->n_beams || depth > 24 || visited_beam[b_idx])
    {
        return;
    }
    visited_beam[b_idx] = 1;

    /* Collect all optical nodes whose primary input is connected to beam @b_idx */
    int consumers[CACAOSIM_BENCH_MAX_NODES];
    int n_cons = 0;
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        if (ctx->nodes[i].in_idx == b_idx)
        {
            consumers[n_cons++] = (int) i;
        }
    }

    const char *vbar = ev->ascii ? "|" : "│";
    const char *darr = ev->ascii ? "v" : "▼";
    const char *larr = ev->ascii ? "<-- " : "◀── ";

    /* Dangling optical arm with no downstream WFS, camera, or relay element */
    if (n_cons == 0)
    {
        sbuf_t row;
        sb_init(&row);
        sb_fg(&row, ev->lvl, ev->pal->fg_conn);
        sb_puts(&row, prefix);
        sb_puts(&row, ev->ascii ? "`--> " : "└──▶ ");
        sb_fg(&row, ev->lvl, ev->pal->fg_dim);
        sb_puts(&row, "(unterminated beam)");
        panel_row(ev, ev->pal->fg_conn, row.data);
        return;
    }

    /* Case 1: Single inline DM or TTM -> keep vertical optical axis unindented so serial
     * TTM + Woofer + Tweeter chains flow straight down the same column */
    if (n_cons == 1 && ctx->nodes[consumers[0]].cfg.kind == CACAOSIM_NODE_DM)
    {
        const cacaosim_bench_node_t *nd    = &ctx->nodes[consumers[0]];
        int                          is_tt = (nd->cfg.dm.dm_mode == 1);
        const char                  *sep   = ev->ascii ? " | " : " │ ";
        const char                  *pm    = ev->ascii ? "+/-" : "±";

        sbuf_t c1;
        sb_init(&c1);
        sb_fg(&c1, ev->lvl, ev->pal->fg_conn);
        sb_puts(&c1, prefix);
        sb_puts(&c1, vbar);
        panel_row(ev, ev->pal->fg_conn, c1.data);

        sbuf_t line;
        sb_init(&line);
        sb_fg(&line, ev->lvl, ev->pal->fg_conn);
        sb_puts(&line, prefix);
        sb_puts(&line, darr);
        sb_puts(&line, " ");
        sb_soft_pill(&line, ev->lvl, ev->pal->bg_panel,
                     is_tt ? ev->pal->fg_proc : ev->pal->fg_fps, is_tt ? "TTM" : "DM");
        sb_puts(&line, " ");
        sb_bold(&line, ev->lvl);
        sb_fg(&line, ev->lvl, ev->pal->fg_bright);
        sb_printf(&line, "%-11s", nd->cfg.id);
        sb_rst(&line, ev->lvl);

        sb_fg(&line, ev->lvl, ev->pal->fg_text);
        if (is_tt)
        {
            sb_puts(&line, " 2-axis tip-tilt");
        }
        else
        {
            sb_printf(&line, " %ux%u act", nd->cfg.dm.nact_x, nd->cfg.dm.nact_y);
        }
        sb_fg(&line, ev->lvl, ev->pal->fg_dim);
        sb_puts(&line, sep);
        sb_fg(&line, ev->lvl, ev->pal->fg_text);
        sb_printf(&line, "%s%.1fum", pm, nd->cfg.dm.max_stroke);
        sb_fg(&line, ev->lvl, ev->pal->fg_dim);
        sb_puts(&line, sep);
        sb_fg(&line, ev->lvl, ev->pal->fg_muted);
        if (nd->cfg.dm.dyn_mode == 1)
        {
            sb_printf(&line, "tau=%.0fus d=%.1ffr", nd->cfg.dm.tau_sec * 1.0e6f,
                      nd->cfg.dm.delay_frames);
        }
        else if (nd->cfg.dm.dyn_mode == 2)
        {
            sb_printf(&line, "f0=%.0fHz d=%.1ffr", nd->cfg.dm.f0_hz, nd->cfg.dm.delay_frames);
        }
        else if (nd->cfg.dm.dyn_mode == 3)
        {
            sb_printf(&line, "plate %.0fHz c=%.0fm/s", nd->cfg.dm.f0_hz, nd->cfg.dm.wave_c_ms);
        }
        else
        {
            sb_printf(&line, "instant d=%.1ffr", nd->cfg.dm.delay_frames);
        }

        /* Show closed-loop actuator command stream injection from the right ("<-- dmXXdisp") */
        sb_puts(&line, "  ");
        sb_fg(&line, ev->lvl, ev->pal->fg_conn);
        sb_puts(&line, larr);
        sb_stream_tag(&line, ev, nd->cfg.in_cmd);
        sb_reg_badge(&line, ev, &nd->cfg.reg);
        panel_row(ev, ev->pal->fg_conn, line.data);

        if (nd->out_idx >= 0)
        {
            float bt = cacaosim_spec_total(&ctx->beams[nd->out_idx].spec);
            float pc = (src_tot > 0.0f) ? (100.0f * bt / src_tot) : 0.0f;

            sbuf_t bl;
            sb_init(&bl);
            sb_fg(&bl, ev->lvl, ev->pal->fg_conn);
            sb_puts(&bl, prefix);
            sb_puts(&bl, vbar);
            sb_fg(&bl, ev->lvl, ev->pal->fg_dim);
            sb_printf(&bl, "   beam ");
            sb_fg(&bl, ev->lvl, ev->pal->fg_accent);
            sb_puts(&bl, ctx->beams[nd->out_idx].name);
            sb_fg(&bl, ev->lvl, ev->pal->fg_dim);
            sb_printf(&bl, " (%.1f%%, %.2e ph/s)", pc, bt);
            panel_row(ev, ev->pal->fg_conn, bl.data);

            render_beam_subtree(ev, ctx, nd->out_idx, prefix, src_tot, src_peak, visited_beam,
                                depth + 1);
        }
        return;
    }

    /* Case 1b: Single inline Static Wavefront Error [OPD m] -> keep vertical axis unindented */
    if (n_cons == 1 && ctx->nodes[consumers[0]].cfg.kind == CACAOSIM_NODE_WFE)
    {
        const cacaosim_bench_node_t *nd  = &ctx->nodes[consumers[0]];
        const char                  *sep = ev->ascii ? " | " : " │ ";

        sbuf_t c1;
        sb_init(&c1);
        sb_fg(&c1, ev->lvl, ev->pal->fg_conn);
        sb_puts(&c1, prefix);
        sb_puts(&c1, vbar);
        panel_row(ev, ev->pal->fg_conn, c1.data);

        sbuf_t line;
        sb_init(&line);
        sb_fg(&line, ev->lvl, ev->pal->fg_conn);
        sb_puts(&line, prefix);
        sb_puts(&line, darr);
        sb_puts(&line, " ");
        sb_soft_pill(&line, ev->lvl, ev->pal->bg_panel, ev->pal->fg_warn, "WFE");
        sb_puts(&line, " ");
        sb_bold(&line, ev->lvl);
        sb_fg(&line, ev->lvl, ev->pal->fg_bright);
        sb_printf(&line, "%-11s", nd->cfg.id);
        sb_rst(&line, ev->lvl);

        sb_fg(&line, ev->lvl, ev->pal->fg_text);
        sb_printf(&line, " static OPD [m]");
        sb_fg(&line, ev->lvl, ev->pal->fg_dim);
        sb_puts(&line, sep);
        sb_fg(&line, ev->lvl, ev->pal->fg_text);
        if (fabsf(nd->cfg.wfe_defocus_nm) > 1.0e-3f && nd->cfg.wfe_rms_nm <= 1.0e-3f)
        {
            sb_printf(&line, "Z4=%+.0fnm", nd->cfg.wfe_defocus_nm);
        }
        else if (fabsf(nd->cfg.wfe_defocus_nm) > 1.0e-3f)
        {
            sb_printf(&line, "rms=%.0fnm Z4=%+.0fnm", nd->cfg.wfe_rms_nm, nd->cfg.wfe_defocus_nm);
        }
        else
        {
            sb_printf(&line, "rms=%.0fnm", nd->cfg.wfe_rms_nm);
        }
        sb_fg(&line, ev->lvl, ev->pal->fg_dim);
        sb_puts(&line, sep);
        sb_fg(&line, ev->lvl, ev->pal->fg_muted);
        sb_printf(&line, "gain=%.2f T=%.0f%%", nd->cfg.wfe_gain, 100.0f * nd->cfg.tput);

        sb_puts(&line, "  ");
        sb_fg(&line, ev->lvl, ev->pal->fg_conn);
        sb_puts(&line, larr);
        sb_stream_tag(&line, ev, nd->cfg.in_wfe);
        sb_reg_badge(&line, ev, &nd->cfg.reg);
        panel_row(ev, ev->pal->fg_conn, line.data);

        if (nd->out_idx >= 0)
        {
            float bt = cacaosim_spec_total(&ctx->beams[nd->out_idx].spec);
            float pc = (src_tot > 0.0f) ? (100.0f * bt / src_tot) : 0.0f;

            sbuf_t bl;
            sb_init(&bl);
            sb_fg(&bl, ev->lvl, ev->pal->fg_conn);
            sb_puts(&bl, prefix);
            sb_puts(&bl, vbar);
            sb_fg(&bl, ev->lvl, ev->pal->fg_dim);
            sb_printf(&bl, "   beam ");
            sb_fg(&bl, ev->lvl, ev->pal->fg_accent);
            sb_puts(&bl, ctx->beams[nd->out_idx].name);
            sb_fg(&bl, ev->lvl, ev->pal->fg_dim);
            sb_printf(&bl, " (%.1f%%, %.2e ph/s)", pc, bt);
            panel_row(ev, ev->pal->fg_conn, bl.data);

            render_beam_subtree(ev, ctx, nd->out_idx, prefix, src_tot, src_peak, visited_beam,
                                depth + 1);
        }
        return;
    }

    /* Case 2: Single Dichroic or Gray Beamsplitter -> branch into refl and trans arms */
    if (n_cons == 1 && ctx->nodes[consumers[0]].cfg.kind == CACAOSIM_NODE_SPLITTER)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[consumers[0]];
        char                         sdesc[128];
        format_splitter_desc(&nd->cfg.split, ev->ascii, sdesc, sizeof(sdesc));

        sbuf_t c1;
        sb_init(&c1);
        sb_fg(&c1, ev->lvl, ev->pal->fg_conn);
        sb_puts(&c1, prefix);
        sb_puts(&c1, vbar);
        panel_row(ev, ev->pal->fg_conn, c1.data);

        sbuf_t hdr;
        sb_init(&hdr);
        sb_fg(&hdr, ev->lvl, ev->pal->fg_warn);
        sb_puts(&hdr, prefix);
        sb_puts(&hdr, ev->ascii ? "<>== " : "◈══ ");
        sb_soft_pill(&hdr, ev->lvl, ev->pal->bg_panel, ev->pal->fg_warn, "SPLITTER");
        sb_puts(&hdr, " ");
        sb_bold(&hdr, ev->lvl);
        sb_fg(&hdr, ev->lvl, ev->pal->fg_bright);
        sb_puts(&hdr, nd->cfg.id);
        sb_rst(&hdr, ev->lvl);
        sb_puts(&hdr, "  ");
        sb_fg(&hdr, ev->lvl, ev->pal->fg_muted);
        sb_puts(&hdr, sdesc);
        panel_row(ev, ev->pal->fg_conn, hdr.data);

        int         arms[2]   = { -1, -1 };
        const char *alabel[2] = { NULL, NULL };
        const cacaosim_reg_t *areg[2] = { NULL, NULL };
        int         n_arms    = 0;

        /* Render reflected arm first (typically WFS pickoff) and transmitted arm second */
        if (nd->refl_idx >= 0)
        {
            arms[n_arms]   = nd->refl_idx;
            alabel[n_arms] = "refl ";
            areg[n_arms]   = &nd->cfg.reg_refl;
            n_arms++;
        }
        if (nd->trans_idx >= 0)
        {
            arms[n_arms]   = nd->trans_idx;
            alabel[n_arms] = "trans";
            areg[n_arms]   = &nd->cfg.reg_trans;
            n_arms++;
        }

        for (int a = 0; a < n_arms; a++)
        {
            int   is_last = (a == n_arms - 1);
            int   bi      = arms[a];
            float bt      = cacaosim_spec_total(&ctx->beams[bi].spec);
            float pc      = (src_tot > 0.0f) ? (100.0f * bt / src_tot) : 0.0f;

            sbuf_t arm;
            sb_init(&arm);
            sb_fg(&arm, ev->lvl, ev->pal->fg_conn);
            sb_puts(&arm, prefix);
            sb_puts(&arm, is_last ? (ev->ascii ? "`-- " : "└── ")
                                  : (ev->ascii ? "|-- " : "├── "));
            sb_fg(&arm, ev->lvl, ev->pal->fg_warn);
            sb_puts(&arm, alabel[a]);
            sb_fg(&arm, ev->lvl, ev->pal->fg_conn);
            sb_puts(&arm, ev->ascii ? " --> " : " ──▶ ");
            sb_bold(&arm, ev->lvl);
            sb_fg(&arm, ev->lvl, ev->pal->fg_accent);
            sb_printf(&arm, "%-11s", ctx->beams[bi].name);
            sb_rst(&arm, ev->lvl);
            sb_fg(&arm, ev->lvl, ev->pal->fg_muted);
            sb_printf(&arm, " (%5.1f%%) ", pc);
            sb_spectrum_bar(&arm, ev, &ctx->beams[bi].spec, src_peak, 24);
            sb_reg_badge(&arm, ev, areg[a]);
            panel_row(ev, ev->pal->fg_conn, arm.data);

            /* Propagate vertical continuation bar "│     " while sibling arm(s) remain below */
            char child_pfx[256];
            snprintf(child_pfx, sizeof(child_pfx), "%s%s", prefix,
                     is_last ? "      " : (ev->ascii ? "|     " : "│     "));
            render_beam_subtree(ev, ctx, bi, child_pfx, src_tot, src_peak, visited_beam, depth + 1);

            if (!is_last)
            {
                sbuf_t sp;
                sb_init(&sp);
                sb_fg(&sp, ev->lvl, ev->pal->fg_conn);
                sb_puts(&sp, prefix);
                sb_puts(&sp, vbar);
                panel_row(ev, ev->pal->fg_conn, sp.data);
            }
        }
        return;
    }

    /* Case 3: Single Coronagraph -> optional FPM/Lyot LOWFS pickoff + post-Lyot science beam */
    if (n_cons == 1 && ctx->nodes[consumers[0]].cfg.kind == CACAOSIM_NODE_CORO)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[consumers[0]];

        sbuf_t c1;
        sb_init(&c1);
        sb_fg(&c1, ev->lvl, ev->pal->fg_conn);
        sb_puts(&c1, prefix);
        sb_puts(&c1, vbar);
        panel_row(ev, ev->pal->fg_conn, c1.data);

        sbuf_t hdr;
        sb_init(&hdr);
        sb_fg(&hdr, ev->lvl, ev->pal->fg_proc);
        sb_puts(&hdr, prefix);
        sb_puts(&hdr, ev->ascii ? "(o)== " : "◎══ ");
        sb_soft_pill(&hdr, ev->lvl, ev->pal->bg_panel, ev->pal->fg_proc, "CORO");
        sb_puts(&hdr, " ");
        sb_bold(&hdr, ev->lvl);
        sb_fg(&hdr, ev->lvl, ev->pal->fg_bright);
        sb_puts(&hdr, nd->cfg.id);
        sb_rst(&hdr, ev->lvl);
        sb_fg(&hdr, ev->lvl, ev->pal->fg_muted);
        if (nd->cfg.coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
        {
            sb_printf(&hdr, "  %s (px=%.1f px/[l/D] @ %.2fum, Lyot %.0f%%/%.0f%%)",
                      coro_mode_label(nd->cfg.coro.mode), nd->cfg.coro.fpm_pixscale_ld,
                      nd->cfg.coro.lambda_um, 100.0f * nd->cfg.coro.lyot_outer,
                      100.0f * nd->cfg.coro.lyot_obs);
        }
        else if (nd->cfg.coro.mode >= CACAOSIM_CORO_VORTEX)
        {
            sb_printf(&hdr, "  %s (pad=%ux @ %.2fum, Lyot %.0f%%/%.0f%%)",
                      coro_mode_label(nd->cfg.coro.mode), nd->cfg.coro.pad_factor,
                      nd->cfg.coro.lambda_um, 100.0f * nd->cfg.coro.lyot_outer,
                      100.0f * nd->cfg.coro.lyot_obs);
        }
        else
        {
            sb_printf(&hdr, "  %s (r=%.2f l/D @ %.2fum, Lyot %.0f%%/%.0f%%)",
                      coro_mode_label(nd->cfg.coro.mode), nd->cfg.coro.fpm_rad_ld,
                      nd->cfg.coro.lambda_um, 100.0f * nd->cfg.coro.lyot_outer,
                      100.0f * nd->cfg.coro.lyot_obs);
        }
        if (nd->cfg.coro.nlambda > 1)
        {
            sb_fg(&hdr, ev->lvl, ev->pal->fg_accent);
            sb_printf(&hdr, " [%dch, dL/L=%.0f%%]", nd->cfg.coro.nlambda,
                      100.0f * nd->cfg.coro.bw_frac);
        }
        panel_row(ev, ev->pal->fg_conn, hdr.data);

        /* Emit optional auxiliary mask line when custom FITS or SHM masks are attached */
        if (nd->cfg.in_pup_amp[0] != '\0' || nd->cfg.in_pup_pha[0] != '\0' ||
            nd->cfg.in_fpm_amp[0] != '\0' || nd->cfg.in_fpm_pha[0] != '\0' ||
            nd->cfg.in_lyot[0] != '\0' || nd->cfg.in_lyot_pha[0] != '\0')
        {
            sbuf_t mk;
            sb_init(&mk);
            sb_fg(&mk, ev->lvl, ev->pal->fg_conn);
            sb_puts(&mk, prefix);
            sb_puts(&mk, vbar);
            sb_fg(&mk, ev->lvl, ev->pal->fg_proc);
            sb_puts(&mk, "   masks: ");
            int n_m = 0;
            n_m += sb_mask_spec_tag(&mk, ev, "pup_amp", nd->cfg.in_pup_amp, n_m > 0);
            n_m += sb_mask_spec_tag(&mk, ev, "pup_pha", nd->cfg.in_pup_pha, n_m > 0);
            n_m += sb_mask_spec_tag(&mk, ev, "fpm_amp", nd->cfg.in_fpm_amp, n_m > 0);
            n_m += sb_mask_spec_tag(&mk, ev, "fpm_pha", nd->cfg.in_fpm_pha, n_m > 0);
            n_m += sb_mask_spec_tag(&mk, ev, "lyot_amp", nd->cfg.in_lyot, n_m > 0);
            n_m += sb_mask_spec_tag(&mk, ev, "lyot_pha", nd->cfg.in_lyot_pha, n_m > 0);
            if (nd->cfg.in_pup_pha[0] != '\0' || nd->cfg.in_fpm_pha[0] != '\0' ||
                nd->cfg.in_lyot_pha[0] != '\0')
            {
                sb_fg(&mk, ev->lvl, ev->pal->fg_muted);
                sb_printf(&mk, " [%s]",
                          (nd->cfg.coro.mask_pha_unit == 1) ? "OPD um" : "rad");
            }
            panel_row(ev, ev->pal->fg_conn, mk.data);
        }

        int         arms[2]     = { -1, -1 };
        const char *alabel[2]   = { NULL, NULL };
        int         is_lowfs[2] = { 0, 0 };
        int         n_arms      = 0;

        /* Order rejected starlight LOWFS pickoff arm ahead of post-Lyot science arm */
        if (nd->lowfs_idx >= 0)
        {
            arms[n_arms]     = nd->lowfs_idx;
            alabel[n_arms]   = (nd->cfg.coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL)
                                   ? "lowfs(Lyot)"
                                   : "lowfs(FPM) ";
            is_lowfs[n_arms] = 1;
            n_arms++;
        }
        if (nd->out_idx >= 0)
        {
            arms[n_arms]   = nd->out_idx;
            alabel[n_arms] = (nd->lowfs_idx >= 0) ? "lyot       " : "lyot ";
            n_arms++;
        }

        for (int a = 0; a < n_arms; a++)
        {
            int   is_last = (a == n_arms - 1);
            int   bi      = arms[a];
            float bt      = cacaosim_spec_total(&ctx->beams[bi].spec);
            float pc      = (src_tot > 0.0f) ? (100.0f * bt / src_tot) : 0.0f;

            sbuf_t arm;
            sb_init(&arm);
            sb_fg(&arm, ev->lvl, ev->pal->fg_conn);
            sb_puts(&arm, prefix);
            sb_puts(&arm, is_last ? (ev->ascii ? "`-- " : "└── ")
                                  : (ev->ascii ? "|-- " : "├── "));
            sb_fg(&arm, ev->lvl, ev->pal->fg_proc);
            sb_puts(&arm, alabel[a]);
            sb_fg(&arm, ev->lvl, ev->pal->fg_conn);
            sb_puts(&arm, ev->ascii ? " --> " : " ──▶ ");
            sb_bold(&arm, ev->lvl);
            sb_fg(&arm, ev->lvl, ev->pal->fg_accent);
            sb_printf(&arm, "%-11s", ctx->beams[bi].name);
            sb_rst(&arm, ev->lvl);
            sb_fg(&arm, ev->lvl, ev->pal->fg_muted);
            sb_printf(&arm, " (%5.1f%%) ", pc);
            sb_spectrum_bar(&arm, ev, &ctx->beams[bi].spec, src_peak, 24);
            if (is_lowfs[a] && nd->cfg.coro.lowfs_mode == CACAOSIM_LOWFS_FPM_REFL)
            {
                float ap = (nd->cfg.coro.lowfs_pup_diam > 0.0f) ? nd->cfg.coro.lowfs_pup_diam
                                                                : 1.0f;
                sb_fg(&arm, ev->lvl, ev->pal->fg_warn);
                sb_printf(&arm, " [ap=%.2fD]", ap);
            }
            panel_row(ev, ev->pal->fg_conn, arm.data);

            char child_pfx[256];
            snprintf(child_pfx, sizeof(child_pfx), "%s%s", prefix,
                     is_last ? "      " : (ev->ascii ? "|     " : "│     "));
            render_beam_subtree(ev, ctx, bi, child_pfx, src_tot, src_peak, visited_beam, depth + 1);

            if (!is_last)
            {
                sbuf_t sp;
                sb_init(&sp);
                sb_fg(&sp, ev->lvl, ev->pal->fg_conn);
                sb_puts(&sp, prefix);
                sb_puts(&sp, vbar);
                panel_row(ev, ev->pal->fg_conn, sp.data);
            }
        }
        return;
    }

    /* Case 4: Terminal sensor(s) or general multi-consumer fanout sharing beam @b_idx */
    for (int c = 0; c < n_cons; c++)
    {
        int                          is_last = (c == n_cons - 1);
        const cacaosim_bench_node_t *nd      = &ctx->nodes[consumers[c]];

        sbuf_t line;
        sb_init(&line);
        sb_fg(&line, ev->lvl, ev->pal->fg_conn);
        sb_puts(&line, prefix);
        sb_puts(&line, is_last ? (ev->ascii ? "`--> " : "└──▶ ")
                               : (ev->ascii ? "|--> " : "├──▶ "));

        if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS && nd->cfg.kind <= CACAOSIM_NODE_FPCAM)
        {
            render_sensor_inline(&line, ev, ctx, nd);
            panel_row(ev, ev->pal->fg_conn, line.data);
        }
        else
        {
            sb_soft_pill(&line, ev->lvl, ev->pal->bg_panel, ev->pal->fg_fps, nd->cfg.id);
            panel_row(ev, ev->pal->fg_conn, line.data);
            if (nd->out_idx >= 0)
            {
                char child_pfx[256];
                snprintf(child_pfx, sizeof(child_pfx), "%s%s", prefix,
                         is_last ? "     " : (ev->ascii ? "|    " : "│    "));
                render_beam_subtree(ev, ctx, nd->out_idx, child_pfx, src_tot, src_peak,
                                    visited_beam, depth + 1);
            }
        }
    }
}

/**
 * count_shm_streams() - Tally expected vs currently materialized SHM streams for a bench DAG
 * @ev:        Active rendering environment supplying @check_shm and @shm_dir (non-NULL)
 * @ctx:       Analyzed optical bench context (non-NULL)
 * @out_live:  Output count of configured SHM endpoints currently present in MILK_SHM_DIR (non-NULL)
 * @out_total: Output count of all primary SHM endpoints referenced by @ctx (non-NULL)
 *
 * Role: Computes the aggregate "N/M SHM LIVE" vs "STANDBY" summary pill rendered at the top of
 * Panel 1.
 *
 * Rationale: Counts primary hardware-facing SHM interfaces (entrance pupil stream, DM actuator
 * command streams, static WFE SHM maps, non-FITS coronagraph SHM masks, and sensor camera output
 * streams) while excluding disk FITS mask paths (cacaosim_mask_is_fits_path()) and optional
 * external turbulence streams (in_wf, which may legitimately be absent in static calibration runs).
 */
static void count_shm_streams(
    const view_env_t           *ev,
    const cacaosim_bench_ctx_t *ctx,
    int                        *out_live,
    int                        *out_total)
{
    int live  = 0;
    int total = 0;

    if (ctx->cfg.in_pup[0] != '\0')
    {
        total++;
        live += shm_stream_exists(ev, ctx->cfg.in_pup);
    }
    for (uint32_t i = 0; i < ctx->n_nodes; i++)
    {
        const cacaosim_bench_node_t *nd = &ctx->nodes[i];
        if (nd->cfg.kind == CACAOSIM_NODE_DM && nd->cfg.in_cmd[0] != '\0')
        {
            total++;
            live += shm_stream_exists(ev, nd->cfg.in_cmd);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_WFE && nd->cfg.in_wfe[0] != '\0')
        {
            total++;
            live += shm_stream_exists(ev, nd->cfg.in_wfe);
        }
        else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
        {
            /* Only count coronagraph mask specifiers backed by SHM streams, skipping FITS files */
            const char *ms[6] = {
                nd->cfg.in_pup_amp, nd->cfg.in_pup_pha, nd->cfg.in_fpm_amp,
                nd->cfg.in_fpm_pha, nd->cfg.in_lyot,    nd->cfg.in_lyot_pha
            };
            for (int m = 0; m < 6; m++)
            {
                if (ms[m][0] != '\0' && !cacaosim_mask_is_fits_path(ms[m]))
                {
                    total++;
                    live += shm_stream_exists(ev, ms[m]);
                }
            }
        }
        else if (nd->cfg.kind >= CACAOSIM_NODE_PYRWFS &&
                 nd->cfg.kind <= CACAOSIM_NODE_FPCAM && nd->cfg.out_cam[0] != '\0')
        {
            total++;
            live += shm_stream_exists(ev, nd->cfg.out_cam);
        }
    }
    *out_live  = live;
    *out_total = total;
}

/**
 * init_view_env() - Resolve all terminal capabilities, theme pointers, width, and MILK_SHM_DIR
 * @ev:   Output per-render environment struct to populate (non-NULL)
 * @opts: Optional caller rendering options (NULL falls back to cacaosim_view_opts_defaults())
 * @fp:   Destination output stream (non-NULL)
 *
 * Role: Populates @ev in a single pass before any panel formatting begins, shared between normal
 * schematic rendering (cacaosim_bench_render()) and diagnostic error panels
 * (cacaosim_bench_render_error()).
 *
 * Rationale: Centralizing option normalization, theme bounds clamping, TTY/locale probing, and
 * MILK_SHM_DIR fallback ("/milk/shm") guarantees identical visual behavior across success and
 * failure code paths.
 */
static void init_view_env(
    view_env_t                 *ev,
    const cacaosim_view_opts_t *opts,
    FILE                       *fp)
{
    cacaosim_view_opts_t def;
    if (opts == NULL)
    {
        cacaosim_view_opts_defaults(&def);
        opts = &def;
    }
    int tidx = opts->theme;
    if (tidx < 0 || tidx >= CACAOSIM_THEME_COUNT)
    {
        tidx = CACAOSIM_THEME_DARK;
    }

    ev->fp        = fp;
    ev->theme_idx = tidx;
    ev->pal       = &g_palettes[tidx];
    ev->lvl       = resolve_color_level(opts, fp);
    ev->ascii     = resolve_ascii_mode(opts);
    ev->check_shm = opts->check_shm;
    ev->width     = resolve_panel_width(opts, fp);

    const char *sd = getenv("MILK_SHM_DIR");
    snprintf(ev->shm_dir, sizeof(ev->shm_dir), "%s",
             (sd != NULL && sd[0] != '\0') ? sd : "/milk/shm");
}

/**
 * compute_bench_scales() - Derive pupil-plane and representative focal-plane physical plate scales
 * @ctx:                Analyzed optical bench context (non-NULL)
 * @out_beam_px:        Output clear pupil diameter across the nwf grid [pixels] (non-NULL)
 * @out_pup_m_pix:      Output pupil spatial sampling [meters / pixel] (non-NULL)
 * @out_foc_ld_pix:     Output focal-plane angular sampling [lambda/D per pixel] (non-NULL)
 * @out_foc_arcsec_pix: Output on-sky focal plate scale [arcseconds / pixel] (non-NULL)
 * @out_ref_lam_um:     Output reference wavelength used for focal plate scale [um] (non-NULL)
 * @out_src:            Output short label naming the optical node that defined focal scale
 * @src_cap:            Capacity of @out_src in bytes
 *
 * Role: Computes the physical pupil and focal sampling metrics displayed in Panel 1 rows 3-4
 * ("Pupil Sc" and "Focal Sc").
 *
 * Rationale: Different optical propagators in cacaosim parameterize focal sampling differently:
 *   - Padded FFT modes (FPCAM FFT, Vortex/4QPM/Custom-FFT CORO, PYRWFS): 1 / (pup_diam * pad)
 *     lambda/D per pixel.
 *   - Soummer Matrix-DFT modes (FPCAM MFT, Custom-MFT CORO): 1 / pixscale_ld lambda/D per pixel.
 *   - Analytic Lyot/dimple MFT modes (CORO spot/dimple, ZWFS): focal mask diameter in lambda/D
 *     divided by n_mft grid samples.
 *   - Diffractive lenslet SHWFS: n_sub / pad_factor full-aperture lambda/D per pixel (= 1 / pad
 *     subaperture lambda/d_sub per pixel).
 * Scanning nodes in priority order (FPCAM -> CORO -> SHWFS -> PYRWFS -> ZWFS, with a 2x-padded FFT
 * fallback for mirror-only benches) surfaces the primary science or WFS plate scale and converts
 * lambda/D to sky arcseconds via (lambda / D_tel) * (180 * 3600 / pi = 206264.8062 arcsec/rad).
 */
static void compute_bench_scales(
    const cacaosim_bench_ctx_t *ctx,
    float                      *out_beam_px,
    float                      *out_pup_m_pix,
    float                      *out_foc_ld_pix,
    float                      *out_foc_arcsec_pix,
    float                      *out_ref_lam_um,
    char                       *out_src,
    size_t                      src_cap)
{
    float nwf_f     = (ctx->nwf > 0u) ? (float) ctx->nwf : 128.0f;
    float pdia      = (ctx->cfg.pup_diam > 0.05f) ? ctx->cfg.pup_diam : 1.0f;
    float tel_d     = (ctx->cfg.tel_diam_m > 0.05f) ? ctx->cfg.tel_diam_m : 8.2f;
    *out_beam_px    = pdia * nwf_f;
    *out_pup_m_pix  = tel_d / *out_beam_px;
    *out_foc_ld_pix = 1.0f / (pdia * 2.0f);
    *out_ref_lam_um = (ctx->cfg.ref_lambda > 0.1f) ? ctx->cfg.ref_lambda : 1.65f;
    snprintf(out_src, src_cap, "2xFFT");

    /* Select highest-priority focal-plane instrument present on the bench */
    static const cacaosim_node_kind_t prio[5] = {
        CACAOSIM_NODE_FPCAM, CACAOSIM_NODE_CORO,   CACAOSIM_NODE_SHWFS,
        CACAOSIM_NODE_PYRWFS, CACAOSIM_NODE_ZWFS
    };
    int found = 0;
    for (int p = 0; p < 5 && !found; p++)
    {
        for (uint32_t i = 0; i < ctx->n_nodes; i++)
        {
            const cacaosim_node_cfg_t *nd = &ctx->nodes[i].cfg;
            if (nd->kind != prio[p])
            {
                continue;
            }
            if (nd->kind == CACAOSIM_NODE_FPCAM)
            {
                *out_ref_lam_um = (nd->fpcam.lambda_um > 0.1f) ? nd->fpcam.lambda_um : 1.65f;
                if (nd->fpcam.prop_mode == CACAOSIM_FPCAM_PROP_FFT)
                {
                    float pad = (nd->fpcam.pad_factor >= 2u) ? (float) nd->fpcam.pad_factor : 2.0f;
                    *out_foc_ld_pix = 1.0f / (pdia * pad);
                }
                else
                {
                    float px = (nd->fpcam.pixscale_ld > 0.05f) ? nd->fpcam.pixscale_ld : 3.0f;
                    *out_foc_ld_pix = 1.0f / px;
                }
                snprintf(out_src, src_cap, "%s", nd->id);
                found = 1;
                break;
            }
            if (nd->kind == CACAOSIM_NODE_CORO)
            {
                *out_ref_lam_um = (nd->coro.lambda_um > 0.1f) ? nd->coro.lambda_um : 1.65f;
                if (nd->coro.mode == CACAOSIM_CORO_CUSTOM_MFT)
                {
                    float px = (nd->coro.fpm_pixscale_ld > 0.05f) ? nd->coro.fpm_pixscale_ld : 8.0f;
                    *out_foc_ld_pix = 1.0f / px;
                }
                else if (nd->coro.mode >= CACAOSIM_CORO_VORTEX)
                {
                    float pad = (nd->coro.pad_factor >= 2u) ? (float) nd->coro.pad_factor : 4.0f;
                    *out_foc_ld_pix = 1.0f / (pdia * pad);
                }
                else
                {
                    float r_ld = (nd->coro.fpm_rad_ld > 0.1f) ? nd->coro.fpm_rad_ld : 2.5f;
                    float nmft = (nd->coro.n_mft >= 8u) ? (float) nd->coro.n_mft : 32.0f;
                    *out_foc_ld_pix = (2.4f * r_ld) / nmft;
                }
                snprintf(out_src, src_cap, "%s", nd->id);
                found = 1;
                break;
            }
            if (nd->kind == CACAOSIM_NODE_SHWFS)
            {
                float nsub      = (nd->shwfs.n_sub >= 1u) ? (float) nd->shwfs.n_sub : 16.0f;
                float pad       = (nd->shwfs.pad_factor >= 1u) ? (float) nd->shwfs.pad_factor
                                                               : 2.0f;
                *out_ref_lam_um = (nd->shwfs.lambda_um > 0.1f) ? nd->shwfs.lambda_um : 0.75f;
                *out_foc_ld_pix = nsub / pad;
                snprintf(out_src, src_cap, "%s", nd->id);
                found = 1;
                break;
            }
            if (nd->kind == CACAOSIM_NODE_PYRWFS)
            {
                float pad       = (nd->pyrwfs.pad_factor >= 2u) ? (float) nd->pyrwfs.pad_factor
                                                                : 2.0f;
                *out_ref_lam_um = (nd->pyrwfs.lambda_um > 0.1f) ? nd->pyrwfs.lambda_um : 0.75f;
                *out_foc_ld_pix = 1.0f / (pdia * pad);
                snprintf(out_src, src_cap, "%s", nd->id);
                found = 1;
                break;
            }
            if (nd->kind == CACAOSIM_NODE_ZWFS)
            {
                float d_ld      = (nd->zwfs.dimple_ld > 0.1f) ? nd->zwfs.dimple_ld : 1.06f;
                float nmft      = (nd->zwfs.n_mft >= 8u) ? (float) nd->zwfs.n_mft : 32.0f;
                *out_ref_lam_um = (nd->zwfs.lambda_um > 0.1f) ? nd->zwfs.lambda_um : 0.85f;
                *out_foc_ld_pix = (1.2f * d_ld) / nmft;
                snprintf(out_src, src_cap, "%s", nd->id);
                found = 1;
                break;
            }
        }
    }
    /* Convert diffraction units [lambda/D] to sky angle [arcsec] via 206264.8062 arcsec/radian */
    *out_foc_arcsec_pix = *out_foc_ld_pix * (*out_ref_lam_um * 1.0e-6f / tel_d) * 206264.8062f;
}

/**
 * cacaosim_bench_render() - Render an analyzed or initialized optical bench DAG to a FILE stream
 * @ctx:         Analyzed or initialized optical bench context (safe no-op if NULL)
 * @source_path: Optional .bench configuration file path displayed in Panel 1 and workflow footer
 * @opts:        Optional rendering options (NULL uses cacaosim_view_opts_defaults())
 * @fp:          Destination output stream (safe no-op if NULL)
 *
 * Role: Top-level layout orchestrator emitting up to four sequential visual blocks to @fp:
 *   1. Panel 1 (Bench Overview, 16x16 Pupil Thumbnail, Physical Scales & 32-bin Star Spectrum).
 *   2. Panel 2 (2D Branching Optical Path & Light-Splitting Schematic via render_beam_subtree()).
 *   3. Panel 3 (Optional Hardware & Photon-Budget Tables for DMs/TTMs, Sensors, and Splitters/
 *      Coronagraphs/Static WFE when opts->show_tables != 0).
 *   4. Workflow Footer (Optional one-line CLI edit/deploy hint when opts->show_footer != 0).
 *
 * Rationale: Structuring the output into vertically stacked, self-contained cards -- overview
 * physics at the top, spatial/spectral signal flow in the middle, and dense tabular parameters at
 * the bottom -- lets operators grasp topology in 2 seconds (--compact) or audit full DAC/ZOH/RON
 * hardware budgets without horizontal scrolling.
 */
void cacaosim_bench_render(
    const cacaosim_bench_ctx_t *ctx,
    const char                 *source_path,
    const cacaosim_view_opts_t *opts,
    FILE                       *fp)
{
    if (ctx == NULL || fp == NULL)
    {
        return;
    }

    cacaosim_view_opts_t def;
    if (opts == NULL)
    {
        cacaosim_view_opts_defaults(&def);
        opts = &def;
    }

    view_env_t ev;
    init_view_env(&ev, opts, fp);

    /* Generate a 16x16 anti-aliased pupil miniature for Panel 1; clamp thin spider widths
     * (< 4% of pupil diameter) up to 0.04 so ~1%-wide spiders remain visible at 16x16 sampling */
    float                pup16[256] = { 0 };
    cacaosim_pupil_cfg_t pcfg       = {
        .nwf           = 16,
        .pup_diam      = ctx->cfg.pup_diam,
        .cobs          = ctx->cfg.cobs,
        .nspider       = ctx->cfg.nspider,
        .spider_w      = (ctx->cfg.spider_w > 0.0f && ctx->cfg.spider_w < 0.04f)
                             ? 0.04f
                             : ctx->cfg.spider_w,
        .spider_deg    = ctx->cfg.spider_deg,
        .spider_offset = ctx->cfg.spider_offset,
    };
    cacaosim_make_pupil(&pcfg, pup16);

    /* Compute entrance total flux [ph/s] and peak bin density for downstream normalization */
    float src_tot  = cacaosim_spec_total(&ctx->beams[0].spec);
    float src_peak = 0.0f;
    for (int b = 0; b < CACAOSIM_NSPEC; b++)
    {
        if (ctx->beams[0].spec.ph_s[b] > src_peak)
        {
            src_peak = ctx->beams[0].spec.ph_s[b];
        }
    }

    int shm_live  = 0;
    int shm_total = 0;
    count_shm_streams(&ev, ctx, &shm_live, &shm_total);

    float beam_px = 0.0f, pup_m_pix = 0.0f, foc_ld_pix = 0.0f;
    float foc_arcsec_pix = 0.0f, foc_lam_um = 1.65f;
    char  foc_src[32] = "";
    compute_bench_scales(ctx, &beam_px, &pup_m_pix, &foc_ld_pix, &foc_arcsec_pix, &foc_lam_um,
                         foc_src, sizeof(foc_src));

    /* -- Panel 1: Optical Bench Overview + 2D Pupil Preview + Star Spectrum -- */
    char hdr_title[128];
    snprintf(hdr_title, sizeof(hdr_title), "CACAOSIM OPTICAL BENCH : %s", ctx->cfg.name);
    panel_top(&ev, hdr_title, ev.pal->fg_title, ev.pal->fg_title);

    const char *sep = ev.ascii ? " | " : " │ ";
    for (int r = 0; r < 10; r++)
    {
        sbuf_t ln;
        sb_init(&ln);
        sb_puts(&ln, " ");
        /* Left 16-col pane: rows 0..7 show 16x16 pupil half-blocks; rows 8..9 show grid/beam px */
        if (r < 8)
        {
            sb_pupil_mini_row(&ln, &ev, pup16, r);
        }
        else if (r == 8)
        {
            char ptag[20];
            snprintf(ptag, sizeof(ptag), "%ux%u pix", ctx->nwf, ctx->nwf);
            sb_fg(&ln, ev.lvl, ev.pal->fg_stream);
            sb_printf(&ln, "%-16.16s", ptag);
        }
        else
        {
            char btag[20];
            snprintf(btag, sizeof(btag), "Beam %.1f pix", beam_px);
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_printf(&ln, "%-16.16s", btag);
        }
        sb_fg(&ln, ev.lvl, ev.pal->fg_dim);
        sb_puts(&ln, ev.ascii ? " | " : " │ ");
        sb_rst(&ln, ev.lvl);

        /* Right metadata pane: status pills, scales, clock, photometry & 32-bin spectrum */
        if (r == 0)
        {
            sb_pill(&ln, ev.lvl, ev.pal->fg_active,
                    ev.ascii ? "* VALID DAG" : "● VALID DAG");
            sb_puts(&ln, " ");
            if (ev.check_shm)
            {
                char sbadge[560];
                if (shm_live > 0)
                {
                    snprintf(sbadge, sizeof(sbadge), "%s %d/%d SHM LIVE",
                             ev.ascii ? "*" : "●", shm_live, shm_total);
                    sb_pill(&ln, ev.lvl, ev.pal->fg_stream, sbadge);
                }
                else
                {
                    snprintf(sbadge, sizeof(sbadge), "%s STANDBY (%s)",
                             ev.ascii ? "o" : "○", ev.shm_dir);
                    sb_soft_pill(&ln, ev.lvl, ev.pal->bg_panel, ev.pal->fg_muted, sbadge);
                }
                sb_puts(&ln, " ");
            }
            sb_soft_pill(&ln, ev.lvl, ev.pal->bg_panel, ev.pal->fg_fps,
                         ctx->use_cuda ? "CUDA GPU" : "CPU SIMD");
        }
        else if (r == 1)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Config   : ");
            sb_fg(&ln, ev.lvl, ev.pal->fg_bright);
            sb_puts(&ln, (source_path && source_path[0]) ? source_path : "(in-memory .bench)");
        }
        else if (r == 2)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Telescope: ");
            sb_fg(&ln, ev.lvl, ev.pal->fg_text);
            sb_printf(&ln, "D=%.2f m%scobs=%.0f%%%s%d spiders (%.0f%s)%sgrid %ux%u pix",
                      ctx->cfg.tel_diam_m, sep, 100.0f * ctx->cfg.cobs, sep, ctx->cfg.nspider,
                      ctx->cfg.spider_deg, ev.ascii ? "deg" : "°", sep, ctx->nwf, ctx->nwf);
        }
        else if (r == 3)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Pupil Sc : ");
            sb_fg(&ln, ev.lvl, ev.pal->fg_text);
            sb_printf(&ln, "Array %ux%u pix%sBeam %.1f pix (%.1f%%)%s", ctx->nwf, ctx->nwf, sep,
                      beam_px, 100.0f * ctx->cfg.pup_diam, sep);
            sb_bold(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_stream);
            sb_printf(&ln, "%.4f m/pix", pup_m_pix);
            sb_rst(&ln, ev.lvl);
        }
        else if (r == 4)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Focal Sc : ");
            sb_bold(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_accent);
            sb_printf(&ln, "%.3f l/D per pix", foc_ld_pix);
            sb_rst(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_printf(&ln, " (%s)%sSky ", foc_src, sep);
            sb_fg(&ln, ev.lvl, ev.pal->fg_text);
            sb_printf(&ln, "%.4f arcsec/pix", foc_arcsec_pix);
            sb_fg(&ln, ev.lvl, ev.pal->fg_dim);
            sb_printf(&ln, " @ %.2f um", foc_lam_um);
        }
        else if (r == 5)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Clock/In : ");
            sb_bold(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_accent);
            if (ctx->eff_oversample > 1u)
            {
                sb_printf(&ln, "%.1f Hz x%u", ctx->cfg.master_hz, ctx->eff_oversample);
            }
            else
            {
                sb_printf(&ln, "%.1f Hz", ctx->cfg.master_hz);
            }
            sb_rst(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            if (ctx->eff_oversample > 1u)
            {
                sb_printf(&ln, " (%.1fkHz, dt=%.0f us)%swf=", ctx->eff_master_hz * 1.0e-3f,
                          ctx->dt_master_s * 1.0e6f, sep);
            }
            else
            {
                sb_printf(&ln, " (dt=%.0f us)%swf=", ctx->dt_master_s * 1.0e6f, sep);
            }
            sb_stream_tag(&ln, &ev, ctx->cfg.in_wf);
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "  pup=");
            sb_stream_tag(&ln, &ev, ctx->cfg.in_pup);
        }
        else if (r == 6)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Star     : ");
            sb_fg(&ln, ev.lvl, ev.pal->fg_text);
            sb_printf(&ln, "mV=%.2f mag%sTeff=%.0f K (%s-type)%sT_tel=%.0f%%",
                      ctx->cfg.star_mag, sep, ctx->cfg.star_teff,
                      spectral_type_hint(ctx->cfg.star_teff), sep, 100.0f * ctx->cfg.tput_tel);
        }
        else if (r == 7)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Photons  : ");
            sb_bold(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_bright);
            sb_printf(&ln, "%.3e ph/s", src_tot);
            sb_rst(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_printf(&ln, "  (%u DMs/TTMs, %u sensors, %u beams)", ctx->n_dms, ctx->n_sensors,
                      ctx->n_beams);
        }
        else if (r == 8)
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_puts(&ln, "Spectrum : ");
            sb_spectrum_bar(&ln, &ev, &ctx->beams[0].spec, src_peak, 32);
            sb_fg(&ln, ev.lvl, ev.pal->fg_dim);
            sb_puts(&ln, " 0.4..2.4um");
        }
        else
        {
            sb_fg(&ln, ev.lvl, ev.pal->fg_dim);
            sb_printf(&ln, "        %s:  0.4    0.8     1.2     1.6     2.0    2.4 um",
                      ev.ascii ? "L" : "λ");
        }
        panel_row(&ev, ev.pal->fg_title, ln.data);
    }
    panel_bot(&ev, ev.pal->fg_title);

    /* -- Panel 2: 2D Branching Optical Path & Light-Splitting Schematic -- */
    panel_top(&ev, "OPTICAL PATH & LIGHT-SPLITTING SCHEMATIC", ev.pal->fg_conn, ev.pal->fg_fps);

    {
        /* Entrance atmosphere + telescope aperture root node feeding beams[0] */
        sbuf_t ent;
        sb_init(&ent);
        sb_fg(&ent, ev.lvl, ev.pal->fg_conn);
        sb_puts(&ent, ev.ascii ? "  +-- " : "  ╭── ");
        sb_soft_pill(&ent, ev.lvl, ev.pal->bg_panel, ev.pal->fg_title, "ATM + TELESCOPE");
        sb_puts(&ent, "  wf: ");
        sb_stream_tag(&ent, &ev, ctx->cfg.in_wf);
        sb_fg(&ent, ev.lvl, ev.pal->fg_dim);
        sb_printf(&ent, " (%s)  pup: ", (ctx->cfg.wf_unit == 1) ? "phase rad" : "OPD um");
        sb_stream_tag(&ent, &ev, ctx->cfg.in_pup);
        if (ctx->cfg.in_ncpa[0] != '\0')
        {
            sb_fg(&ent, ev.lvl, ev.pal->fg_dim);
            sb_puts(&ent, "  ncpa: ");
            sb_stream_tag(&ent, &ev, ctx->cfg.in_ncpa);
        }
        panel_row(&ev, ev.pal->fg_conn, ent.data);

        sbuf_t b0;
        sb_init(&b0);
        sb_fg(&b0, ev.lvl, ev.pal->fg_conn);
        sb_puts(&b0, ev.ascii ? "  |   beam " : "  │   beam ");
        sb_bold(&b0, ev.lvl);
        sb_fg(&b0, ev.lvl, ev.pal->fg_accent);
        sb_puts(&b0, ctx->beams[0].name);
        sb_rst(&b0, ev.lvl);
        sb_fg(&b0, ev.lvl, ev.pal->fg_dim);
        sb_printf(&b0, " (100.0%%, %.2e ph/s)  ", src_tot);
        sb_spectrum_bar(&b0, &ev, &ctx->beams[0].spec, src_peak, 24);
        panel_row(&ev, ev.pal->fg_conn, b0.data);

        uint8_t visited_beam[CACAOSIM_BENCH_MAX_BEAMS] = { 0 };
        render_beam_subtree(&ev, ctx, 0, "  ", src_tot, src_peak, visited_beam, 0);
    }
    panel_bot(&ev, ev.pal->fg_conn);

    /* -- Panel 3: Active Mirrors & Wavefront Sensors / Science Cameras -- */
    if (opts->show_tables)
    {
        if (ctx->n_dms > 0)
        {
            panel_top(&ev, "ACTIVE MIRRORS (DM / TTM)", ev.pal->fg_conn, ev.pal->fg_fps);

            sbuf_t th;
            sb_init(&th);
            sb_bold(&th, ev.lvl);
            sb_fg(&th, ev.lvl, ev.pal->fg_muted);
            sb_printf(&th, " %-2s %-10s %-5s %-21s %-14s %-7s %-8s %-5s %s", "#", "ID", "Kind",
                      "Optical Beam Path", "Command SHM", "Grid", "Stroke", "DAC", "ZOH / Delay");
            sb_rst(&th, ev.lvl);
            panel_row(&ev, ev.pal->fg_conn, th.data);
            panel_sep(&ev, ev.pal->fg_dim);

            for (uint32_t d = 0; d < ctx->n_dms; d++)
            {
                const cacaosim_bench_node_t *nd    = &ctx->nodes[ctx->dm_node_idx[d]];
                int                          is_tt = (nd->cfg.dm.dm_mode == 1);
                char                         bpath[64];
                char                         grid[24];
                char                         stroke[24];
                char                         dac[16];
                char                         dyn[48];

                snprintf(bpath, sizeof(bpath), "%s%s%s", nd->cfg.in_beam,
                         ev.ascii ? "->" : "→", nd->cfg.out_beam);
                snprintf(grid, sizeof(grid), "%ux%u", nd->cfg.dm.nact_x, nd->cfg.dm.nact_y);
                snprintf(stroke, sizeof(stroke), "%s%.1fum", ev.ascii ? "+-" : "±",
                         nd->cfg.dm.max_stroke);
                if (nd->cfg.dm.dac_bits > 0)
                {
                    snprintf(dac, sizeof(dac), "%db", nd->cfg.dm.dac_bits);
                }
                else
                {
                    snprintf(dac, sizeof(dac), "f32");
                }
                char dly_s[24];
                if (nd->cfg.dm.delay_sec > 0.0f)
                {
                    snprintf(dly_s, sizeof(dly_s), "%.0fus", nd->cfg.dm.delay_sec * 1.0e6f);
                }
                else
                {
                    snprintf(dly_s, sizeof(dly_s), "%.1ffr", nd->cfg.dm.delay_frames);
                }
                if (nd->cfg.dm.dyn_mode == 1)
                {
                    snprintf(dyn, sizeof(dyn), "RC %.0fus, %s", nd->cfg.dm.tau_sec * 1.0e6f,
                             dly_s);
                }
                else if (nd->cfg.dm.dyn_mode == 2)
                {
                    snprintf(dyn, sizeof(dyn), "2nd %.0fHz, %s", nd->cfg.dm.f0_hz, dly_s);
                }
                else if (nd->cfg.dm.dyn_mode == 3)
                {
                    snprintf(dyn, sizeof(dyn), "plate %.0fHz, %s", nd->cfg.dm.f0_hz, dly_s);
                }
                else
                {
                    snprintf(dyn, sizeof(dyn), "inst, %s", dly_s);
                }

                sbuf_t rline;
                sb_init(&rline);
                sb_fg(&rline, ev.lvl, ev.pal->fg_dim);
                sb_printf(&rline, " %-2u ", d);
                sb_bold(&rline, ev.lvl);
                sb_fg(&rline, ev.lvl, ev.pal->fg_bright);
                sb_pad_col(&rline, nd->cfg.id, 11);
                sb_rst(&rline, ev.lvl);
                sb_fg(&rline, ev.lvl, is_tt ? ev.pal->fg_proc : ev.pal->fg_fps);
                sb_pad_col(&rline, is_tt ? "TTM" : "DM", 6);
                sb_fg(&rline, ev.lvl, ev.pal->fg_accent);
                sb_pad_col(&rline, bpath, 22);

                /* Measure visible columns before/after sb_stream_tag() to pad exactly to col 15 */
                int c0 = vis_cols(rline.data);
                sb_stream_tag(&rline, &ev, nd->cfg.in_cmd);
                int c1 = vis_cols(rline.data);
                for (int sp = c1 - c0; sp < 15; sp++)
                {
                    sb_puts(&rline, " ");
                }

                sb_fg(&rline, ev.lvl, ev.pal->fg_text);
                sb_pad_col(&rline, grid, 8);
                sb_pad_col(&rline, stroke, 9);
                sb_pad_col(&rline, dac, 6);
                sb_fg(&rline, ev.lvl, ev.pal->fg_muted);
                sb_puts(&rline, dyn);
                panel_row(&ev, ev.pal->fg_conn, rline.data);
            }
            panel_bot(&ev, ev.pal->fg_conn);
        }

        if (ctx->n_sensors > 0)
        {
            panel_top(&ev, "WAVEFRONT SENSORS & SCIENCE CAMERAS", ev.pal->fg_conn,
                      ev.pal->fg_active);

            sbuf_t th;
            sb_init(&th);
            sb_bold(&th, ev.lvl);
            sb_fg(&th, ev.lvl, ev.pal->fg_muted);
            sb_printf(&th, " %-2s %-9s %-6s %-12s %-14s %-7s %-9s %-10s %-8s %s", "#", "ID",
                      "Type", "Beam", "Output SHM", "Size", "Band [um]", "Rate", "Ph/Fr",
                      "RON / Mode");
            sb_rst(&th, ev.lvl);
            panel_row(&ev, ev.pal->fg_conn, th.data);
            panel_sep(&ev, ev.pal->fg_dim);

            for (uint32_t s = 0; s < ctx->n_sensors; s++)
            {
                const cacaosim_bench_node_t *nd   = &ctx->nodes[ctx->sensor_node_idx[s]];
                const char                  *ks   = "CAM";
                float                        ron  = 0.5f;
                cs_rgb_t                     kcol = ev.pal->fg_active;

                if (nd->cfg.kind == CACAOSIM_NODE_PYRWFS)
                {
                    ks  = "PYRWFS";
                    ron = nd->cfg.pyrwfs.det.ron_e;
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_SHWFS)
                {
                    ks  = "SHWFS";
                    ron = nd->cfg.shwfs.det.ron_e;
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_ZWFS)
                {
                    ks   = "ZWFS";
                    ron  = nd->cfg.zwfs.det.ron_e;
                    kcol = ev.pal->fg_proc;
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_FPCAM)
                {
                    ks   = "FPCAM";
                    ron  = nd->cfg.fpcam.det.ron_e;
                    kcol = ev.pal->fg_accent;
                }

                /* Classify detector SNR regime assuming ~25% effective illuminated pixel fill:
                 * Shot-limited if Poisson noise > 2*RON, Mixed if N_ph > RON^2, else RON-limited */
                float m_hz   = (ctx->eff_master_hz > 0.1f) ? ctx->eff_master_hz
                                                           : ctx->cfg.master_hz;
                float hz     = (nd->eff_navg > 0) ? (m_hz / (float) nd->eff_navg) : 0.0f;
                float bw     = (nd->cfg.bw_frac > 0.0f) ? nd->cfg.bw_frac : 0.0f;
                int   nch    = (nd->eff_nlam >= 1) ? nd->eff_nlam : 1;
                float ph_pix = nd->eff_flux_ph /
                               (float) ((nd->ncam > 0) ? (nd->ncam * nd->ncam / 4u + 1u) : 256u);
                const char *reg_lbl = (sqrtf(ph_pix) > 2.0f * ron) ? "Shot"
                                                                   : ((ph_pix > ron * ron)
                                                                          ? "Mixed"
                                                                          : "RON");

                char sz_s[20];
                char bd_s[24];
                char rt_s[24];
                char rn_s[32];
                snprintf(sz_s, sizeof(sz_s), "%ux%u", nd->ncam, nd->ncam);
                snprintf(bd_s, sizeof(bd_s), "%.2f%s%.2f", nd->eff_lambda_um,
                         ev.ascii ? "+-" : "±", 0.5f * bw * nd->eff_lambda_um);
                if (nd->eff_navg > 1u)
                {
                    snprintf(rt_s, sizeof(rt_s), "%.0fHz(/%u)", hz, nd->eff_navg);
                }
                else
                {
                    snprintf(rt_s, sizeof(rt_s), "%.0fHz", hz);
                }
                snprintf(rn_s, sizeof(rn_s), "%.1fe %s %dch", ron, reg_lbl, nch);

                sbuf_t rline;
                sb_init(&rline);
                sb_fg(&rline, ev.lvl, ev.pal->fg_dim);
                sb_printf(&rline, " %-2u ", s);
                sb_bold(&rline, ev.lvl);
                sb_fg(&rline, ev.lvl, ev.pal->fg_bright);
                sb_pad_col(&rline, nd->cfg.id, 10);
                sb_rst(&rline, ev.lvl);
                sb_fg(&rline, ev.lvl, kcol);
                sb_pad_col(&rline, ks, 7);
                sb_fg(&rline, ev.lvl, ev.pal->fg_accent);
                sb_pad_col(&rline, nd->cfg.in_beam, 13);

                int c0 = vis_cols(rline.data);
                sb_stream_tag(&rline, &ev, nd->cfg.out_cam);
                int c1 = vis_cols(rline.data);
                for (int sp = c1 - c0; sp < 15; sp++)
                {
                    sb_puts(&rline, " ");
                }

                sb_fg(&rline, ev.lvl, ev.pal->fg_text);
                sb_pad_col(&rline, sz_s, 8);
                sb_fg(&rline, ev.lvl, lambda_color(&ev, nd->eff_lambda_um));
                sb_pad_col(&rline, bd_s, 10);
                sb_fg(&rline, ev.lvl, ev.pal->fg_text);
                sb_pad_col(&rline, rt_s, 11);
                sb_fg(&rline, ev.lvl, ev.pal->fg_bright);
                sb_printf(&rline, "%-8.2e ", nd->eff_flux_ph);
                sb_fg(&rline, ev.lvl,
                      (strcmp(reg_lbl, "Shot") == 0) ? ev.pal->fg_active : ev.pal->fg_warn);
                sb_puts(&rline, rn_s);
                panel_row(&ev, ev.pal->fg_conn, rline.data);
            }
            panel_bot(&ev, ev.pal->fg_conn);
        }

        uint32_t n_aux = 0;
        for (uint32_t i = 0; i < ctx->n_nodes; i++)
        {
            cacaosim_node_kind_t k = ctx->nodes[i].cfg.kind;
            if (k == CACAOSIM_NODE_SPLITTER || k == CACAOSIM_NODE_CORO || k == CACAOSIM_NODE_WFE)
            {
                n_aux++;
            }
        }
        if (n_aux > 0)
        {
            panel_top(&ev, "BEAMSPLITTERS, CORONAGRAPHS & STATIC WFE", ev.pal->fg_conn,
                      ev.pal->fg_proc);

            sbuf_t th;
            sb_init(&th);
            sb_bold(&th, ev.lvl);
            sb_fg(&th, ev.lvl, ev.pal->fg_muted);
            sb_printf(&th, " %-2s %-11s %-8s %-25s %-12s %s", "#", "ID", "Kind",
                      "Mode / Optical Spec", "In Beam", "Output Arms / SHM / Masks");
            sb_rst(&th, ev.lvl);
            panel_row(&ev, ev.pal->fg_conn, th.data);
            panel_sep(&ev, ev.pal->fg_dim);

            uint32_t aidx = 0;
            for (uint32_t i = 0; i < ctx->n_nodes; i++)
            {
                const cacaosim_bench_node_t *nd = &ctx->nodes[i];
                if (nd->cfg.kind != CACAOSIM_NODE_SPLITTER &&
                    nd->cfg.kind != CACAOSIM_NODE_CORO && nd->cfg.kind != CACAOSIM_NODE_WFE)
                {
                    continue;
                }

                sbuf_t rline;
                sb_init(&rline);
                sb_fg(&rline, ev.lvl, ev.pal->fg_dim);
                sb_printf(&rline, " %-2u ", aidx++);
                sb_bold(&rline, ev.lvl);
                sb_fg(&rline, ev.lvl, ev.pal->fg_bright);
                sb_pad_col(&rline, nd->cfg.id, 12);
                sb_rst(&rline, ev.lvl);

                char mspec[32] = "";
                if (nd->cfg.kind == CACAOSIM_NODE_SPLITTER)
                {
                    sb_fg(&rline, ev.lvl, ev.pal->fg_warn);
                    sb_pad_col(&rline, "SPLITTER", 9);
                    if (nd->cfg.split.mode == CACAOSIM_SPLIT_GRAY)
                    {
                        snprintf(mspec, sizeof(mspec), "gray R=%.0f%% T=%.0f%%",
                                 100.0f * nd->cfg.split.r_gray, 100.0f * nd->cfg.split.t_gray);
                    }
                    else
                    {
                        const char *sm = (nd->cfg.split.mode == CACAOSIM_SPLIT_DICHROIC_BP)
                                             ? "BP"
                                             : ((nd->cfg.split.mode == CACAOSIM_SPLIT_DICHROIC_SP)
                                                    ? "SP"
                                                    : "LP");
                        snprintf(mspec, sizeof(mspec), "dichroic %s %.2fum", sm,
                                 nd->cfg.split.lambda_cut);
                    }
                    sb_fg(&rline, ev.lvl, ev.pal->fg_text);
                    sb_pad_col(&rline, mspec, 26);
                    sb_fg(&rline, ev.lvl, ev.pal->fg_accent);
                    sb_pad_col(&rline, nd->cfg.in_beam, 13);
                    sb_fg(&rline, ev.lvl, ev.pal->fg_muted);
                    sb_printf(&rline, "R:%s  T:%s", nd->cfg.out_refl, nd->cfg.out_trans);
                }
                else if (nd->cfg.kind == CACAOSIM_NODE_CORO)
                {
                    sb_fg(&rline, ev.lvl, ev.pal->fg_proc);
                    sb_pad_col(&rline, "CORO", 9);
                    if (nd->cfg.coro.nlambda > 1)
                    {
                        snprintf(mspec, sizeof(mspec), "%s %dch %.2fu",
                                 coro_mode_label(nd->cfg.coro.mode), nd->cfg.coro.nlambda,
                                 nd->cfg.coro.lambda_um);
                    }
                    else
                    {
                        snprintf(mspec, sizeof(mspec), "%s %.2fum",
                                 coro_mode_label(nd->cfg.coro.mode), nd->cfg.coro.lambda_um);
                    }
                    sb_fg(&rline, ev.lvl, ev.pal->fg_text);
                    sb_pad_col(&rline, mspec, 26);
                    sb_fg(&rline, ev.lvl, ev.pal->fg_accent);
                    sb_pad_col(&rline, nd->cfg.in_beam, 13);
                    sb_fg(&rline, ev.lvl, ev.pal->fg_muted);
                    int n_m = (nd->cfg.in_pup_amp[0] != '\0') + (nd->cfg.in_pup_pha[0] != '\0') +
                              (nd->cfg.in_fpm_amp[0] != '\0') + (nd->cfg.in_fpm_pha[0] != '\0') +
                              (nd->cfg.in_lyot[0] != '\0') + (nd->cfg.in_lyot_pha[0] != '\0');
                    if (nd->cfg.out_lowfs[0] != '\0')
                    {
                        if (nd->cfg.coro.lowfs_mode == CACAOSIM_LOWFS_LYOT_REFL)
                        {
                            sb_printf(&rline, "Sci:%s  LOWFS:%s (lyot_refl)", nd->cfg.out_beam,
                                      nd->cfg.out_lowfs);
                        }
                        else
                        {
                            float ap = (nd->cfg.coro.lowfs_pup_diam > 0.0f)
                                           ? nd->cfg.coro.lowfs_pup_diam
                                           : 1.0f;
                            sb_printf(&rline, "Sci:%s  LOWFS:%s (fpm_refl, %.2fD)",
                                      nd->cfg.out_beam, nd->cfg.out_lowfs, ap);
                        }
                    }
                    else
                    {
                        sb_printf(&rline, "Sci:%s", nd->cfg.out_beam);
                    }
                    if (n_m > 0)
                    {
                        sb_fg(&rline, ev.lvl, ev.pal->fg_stream);
                        sb_printf(&rline, " +%dM", n_m);
                    }
                }
                else
                {
                    sb_fg(&rline, ev.lvl, ev.pal->fg_warn);
                    sb_pad_col(&rline, "WFE", 9);
                    if (fabsf(nd->cfg.wfe_defocus_nm) > 1.0e-3f && nd->cfg.wfe_rms_nm <= 1.0e-3f)
                    {
                        snprintf(mspec, sizeof(mspec), "OPD [m] Z4=%+.0fnm x%.2f",
                                 nd->cfg.wfe_defocus_nm, nd->cfg.wfe_gain);
                    }
                    else
                    {
                        snprintf(mspec, sizeof(mspec), "OPD [m] %.0fnm x%.2f", nd->cfg.wfe_rms_nm,
                                 nd->cfg.wfe_gain);
                    }
                    sb_fg(&rline, ev.lvl, ev.pal->fg_text);
                    sb_pad_col(&rline, mspec, 26);
                    sb_fg(&rline, ev.lvl, ev.pal->fg_accent);
                    sb_pad_col(&rline, nd->cfg.in_beam, 13);
                    sb_fg(&rline, ev.lvl, ev.pal->fg_muted);
                    sb_printf(&rline, "Out:%s  SHM:", nd->cfg.out_beam);
                    sb_stream_tag(&rline, &ev, nd->cfg.in_wfe);
                }
                panel_row(&ev, ev.pal->fg_conn, rline.data);
            }
            panel_bot(&ev, ev.pal->fg_conn);
        }
    }

    /* -- Workflow Footer -- */
    if (opts->show_footer)
    {
        const char *p = (source_path && source_path[0]) ? source_path : "<file.bench>";
        sbuf_t      f1;
        sb_init(&f1);
        sb_fg(&f1, ev.lvl, ev.pal->fg_dim);
        sb_puts(&f1, "  Workflow: ");
        sb_fg(&f1, ev.lvl, ev.pal->fg_fps);
        sb_printf(&f1, "cacaosim-bench-view -i %s", p);
        sb_fg(&f1, ev.lvl, ev.pal->fg_dim);
        sb_puts(&f1, " (edit+TUI)  |  ");
        sb_fg(&f1, ev.lvl, ev.pal->fg_active);
        sb_printf(&f1, "cacaosim-bench start --config %s --atmturb", p);
        sb_rst(&f1, ev.lvl);
        fprintf(fp, "%s\n", f1.data);
    }
}

/**
 * cacaosim_bench_render_error() - Render a milk-styled .bench configuration error diagnostic panel
 * @source_path: Optional path to the .bench file being parsed or validated (NULL/"" safe)
 * @partial_cfg: Optional partially parsed bench configuration (NULL when file I/O or INI syntax
 *               parsing failed before DAG analysis)
 * @err_msg:     Syntax or topological DAG validation diagnostic message (NULL/"" safe)
 * @opts:        Optional rendering options (NULL uses cacaosim_view_opts_defaults())
 * @fp:          Destination output stream (safe no-op if NULL)
 *
 * Role: Emits a bordered error card showing the failed file path, specific parser/DAG error text,
 * and (when INI parsing succeeded but topological analysis failed, @partial_cfg != NULL) a compact
 * table of every parsed node's input/output beam identifiers.
 *
 * Rationale: In --watch live-reload workflows, saving a mid-edit .bench file temporarily breaks
 * beam names. Displaying the partial beam wiring table inside a fixed-width panel keeps the
 * screen visually stable and immediately shows which upstream/downstream beam identifier failed to
 * match.
 */
void cacaosim_bench_render_error(
    const char                 *source_path,
    const cacaosim_bench_cfg_t *partial_cfg,
    const char                 *err_msg,
    const cacaosim_view_opts_t *opts,
    FILE                       *fp)
{
    if (fp == NULL)
    {
        return;
    }

    view_env_t ev;
    init_view_env(&ev, opts, fp);

    panel_top(&ev, "CACAOSIM .BENCH CONFIGURATION ERROR", ev.pal->fg_error, ev.pal->fg_error);

    sbuf_t r0;
    sb_init(&r0);
    sb_pill(&r0, ev.lvl, ev.pal->fg_error, ev.ascii ? "X INVALID CONFIG" : "✖ INVALID CONFIG");
    sb_puts(&r0, "  ");
    sb_fg(&r0, ev.lvl, ev.pal->fg_bright);
    sb_puts(&r0, (source_path && source_path[0]) ? source_path : "(in-memory .bench)");
    panel_row(&ev, ev.pal->fg_error, r0.data);

    panel_sep(&ev, ev.pal->fg_error);

    sbuf_t r1;
    sb_init(&r1);
    sb_bold(&r1, ev.lvl);
    sb_fg(&r1, ev.lvl, ev.pal->fg_warn);
    sb_puts(&r1, "Diagnostic : ");
    sb_fg(&r1, ev.lvl, ev.pal->fg_bright);
    sb_puts(&r1, (err_msg && err_msg[0]) ? err_msg : "Unknown configuration error");
    sb_rst(&r1, ev.lvl);
    panel_row(&ev, ev.pal->fg_error, r1.data);

    /* If syntax parsing succeeded, list each section's beam endpoints to aid topology debugging */
    if (partial_cfg != NULL && partial_cfg->n_nodes > 0)
    {
        sbuf_t r2;
        sb_init(&r2);
        sb_fg(&r2, ev.lvl, ev.pal->fg_muted);
        sb_printf(&r2, "Parsed     : bench '%s' (%u sections)  entrance out_beam='%s'",
                  partial_cfg->name, partial_cfg->n_nodes, partial_cfg->out_beam);
        panel_row(&ev, ev.pal->fg_error, r2.data);

        for (uint32_t i = 0; i < partial_cfg->n_nodes; i++)
        {
            const cacaosim_node_cfg_t *nc = &partial_cfg->nodes[i];
            sbuf_t                     rn;
            sb_init(&rn);
            sb_fg(&rn, ev.lvl, ev.pal->fg_dim);
            sb_printf(&rn, "  [%2u] ", i);
            sb_fg(&rn, ev.lvl, ev.pal->fg_fps);
            sb_printf(&rn, "%-14s", nc->id);
            sb_fg(&rn, ev.lvl, ev.pal->fg_muted);
            sb_printf(&rn, " in='%s'", nc->in_beam);
            if (nc->out_beam[0] != '\0')
            {
                sb_printf(&rn, " -> out='%s'", nc->out_beam);
            }
            if (nc->out_refl[0] != '\0' || nc->out_trans[0] != '\0')
            {
                sb_printf(&rn, " -> refl='%s', trans='%s'", nc->out_refl, nc->out_trans);
            }
            if (nc->out_lowfs[0] != '\0')
            {
                sb_printf(&rn, " (lowfs='%s')", nc->out_lowfs);
            }
            panel_row(&ev, ev.pal->fg_error, rn.data);
        }
    }

    panel_bot(&ev, ev.pal->fg_error);
}

/**
 * cacaosim_bench_render_file() - Load, analyze, and render a .bench file in one zero-FFT call
 * @path: Filesystem path to the declarative .bench configuration file (non-NULL)
 * @opts: Optional rendering options (NULL uses cacaosim_view_opts_defaults())
 * @fp:   Destination output stream (non-NULL)
 *
 * Role: Executes the end-to-end file-to-schematic pipeline: parses @path into a stack
 * cacaosim_bench_cfg_t, runs zero-allocation topological & spectral analysis into a stack
 * cacaosim_bench_ctx_t, and renders either the full schematic or a structured error panel.
 *
 * Rationale: Keeping both cfg and ctx on the stack with cacaosim_bench_analyze() avoids any
 * dynamic memory, FFTW/cuFFT planning, or SHM creation overhead, enabling sub-millisecond CLI
 * invocations and flicker-free 150 ms --watch polling loops.
 *
 * Return: 0 if a valid .bench schematic was rendered, -1 on NULL arguments, file/parse error, or
 *         DAG validation failure.
 */
int cacaosim_bench_render_file(
    const char                 *path,
    const cacaosim_view_opts_t *opts,
    FILE                       *fp)
{
    if (path == NULL || fp == NULL)
    {
        return -1;
    }

    cacaosim_bench_cfg_t cfg;
    char                 err_buf[256] = { 0 };
    if (cacaosim_bench_load_file(path, &cfg, err_buf, sizeof(err_buf)) != 0)
    {
        cacaosim_bench_render_error(path, NULL, err_buf, opts, fp);
        return -1;
    }

    cacaosim_bench_ctx_t ctx;
    if (cacaosim_bench_analyze(&ctx, &cfg, err_buf, sizeof(err_buf)) != 0)
    {
        cacaosim_bench_render_error(path, &cfg, err_buf, opts, fp);
        return -1;
    }

    cacaosim_bench_render(&ctx, path, opts, fp);
    return 0;
}
