// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_bench_view.c
 * @brief   milk-styled ANSI / UTF-8 / ASCII graphical visualizer for .bench AO configurations
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

typedef struct
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} cs_rgb_t;

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

/* Exact RGB palettes matching milk's src/cli/overview/render/overview_theme_palettes.c */
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

const char *cacaosim_view_theme_name(int theme)
{
    if (theme < 0 || theme >= CACAOSIM_THEME_COUNT)
    {
        return "dark";
    }
    return g_palettes[theme].name;
}

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

/* Detect ANSI color level identically to milk's ov_detect_color_level() */
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

/* Exact xterm-256 and ANSI-16 color quantizers matching milk overview_theme_draw.h */
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

/* Append-only line buffer for composing styled rows safely */
typedef struct
{
    char   data[4096];
    size_t len;
} sbuf_t;

static void sb_init(sbuf_t *sb)
{
    sb->data[0] = '\0';
    sb->len     = 0;
}

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

static void sb_bold(sbuf_t *sb, int lvl)
{
    if (lvl > 0)
    {
        sb_puts(sb, "\033[1m");
    }
}

static void sb_rst(sbuf_t *sb, int lvl)
{
    if (lvl > 0)
    {
        sb_puts(sb, "\033[0m");
    }
}

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

/* Count visible display columns skipping CSI escape codes; 1 col per UTF-8 codepoint */
static int vis_cols(const char *s)
{
    int                  cols = 0;
    const unsigned char *p    = (const unsigned char *) s;
    while (*p != '\0')
    {
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

static void emit_fg_fp(FILE *fp, int lvl, cs_rgb_t c)
{
    sbuf_t tmp;
    sb_init(&tmp);
    sb_fg(&tmp, lvl, c);
    fputs(tmp.data, fp);
}

static void emit_rst_fp(FILE *fp, int lvl)
{
    if (lvl > 0)
    {
        fputs("\033[0m", fp);
    }
}

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

/* Emit at most max_cols visible columns from an ANSI+UTF-8 string */
static int emit_truncated(FILE *fp, const char *s, int max_cols, int ascii)
{
    int                  total = vis_cols(s);
    int                  limit = (total > max_cols) ? (max_cols - 1) : max_cols;
    int                  shown = 0;
    const unsigned char *p     = (const unsigned char *) s;

    while (*p != '\0' && shown < limit)
    {
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

/* Non-intrusive check whether <stream>.im.shm exists in MILK_SHM_DIR */
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

/* Physical / theme-aware RGB color for optical/NIR wavelength lambda_um in [0.40..2.40] */
static cs_rgb_t lambda_color(const view_env_t *ev, float lam_um)
{
    if (ev->theme_idx == CACAOSIM_THEME_NIGHT)
    {
        float    t  = (lam_um - 0.40f) / 2.00f;
        cs_rgb_t c0 = { 255, 115, 70 };
        cs_rgb_t c1 = { 165, 35, 30 };
        return rgb_lerp(c0, c1, t);
    }
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

/* Render a nbins (16, 24, or 32) spectral bar across 0.40..2.40 um with sqrt gamma scaling */
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

        if (rel < 0.002f)
        {
            sb_fg(sb, ev->lvl, ev->pal->fg_dim);
            sb_puts(sb, ev->ascii ? " " : "·");
        }
        else
        {
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

/* Render row r [0..7] of a 16x16 anti-aliased telescope pupil using Unicode half-blocks */
static void sb_pupil_mini_row(
    sbuf_t           *sb,
    const view_env_t *ev,
    const float      *pup16,
    int               r)
{
    int y_top = 15 - 2 * r;
    int y_bot = 14 - 2 * r;

    for (int x = 0; x < 16; x++)
    {
        float at = pup16[y_top * 16 + x];
        float ab = pup16[y_bot * 16 + x];

        if (ev->ascii)
        {
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
            cs_rgb_t ct = rgb_lerp(ev->pal->bg_panel, ev->pal->fg_stream, at);
            cs_rgb_t cb = rgb_lerp(ev->pal->bg_panel, ev->pal->fg_stream, ab);
            sb_fg(sb, ev->lvl, ct);
            sb_bg(sb, ev->lvl, cb);
            sb_puts(sb, "▀");
        }
        else
        {
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

static void render_sensor_inline(
    sbuf_t                      *sb,
    const view_env_t            *ev,
    const cacaosim_bench_ctx_t  *ctx,
    const cacaosim_bench_node_t *nd)
{
    const char *sep = ev->ascii ? " | " : " │ ";
    const char *arr = ev->ascii ? " --> " : " ──▶ ";
    float       hz  = (nd->eff_navg > 0) ? (ctx->cfg.master_hz / (float) nd->eff_navg) : 0.0f;

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

/* Recursively render the optical beam & light-splitting tree starting from beam b_idx */
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
    if (b_idx < 0 || (uint32_t) b_idx >= ctx->n_beams || depth > 24 || visited_beam[b_idx])
    {
        return;
    }
    visited_beam[b_idx] = 1;

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

    /* Case 1: Single inline DM or TTM -> keep vertical optical axis unindented */
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
        else
        {
            sb_printf(&line, "instant d=%.1ffr", nd->cfg.dm.delay_frames);
        }

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
        sb_printf(&line, "rms=%.0fnm", nd->cfg.wfe_rms_nm);
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

    /* Case 3: Single Coronagraph -> optional FPM LOWFS pickoff + post-Lyot science beam */
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

    /* Case 4: Sensor(s) or general multi-consumer fanout on b_idx */
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

/* Count expected vs currently live SHM streams for the bench */
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
    *out_foc_arcsec_pix = *out_foc_ld_pix * (*out_ref_lam_um * 1.0e-6f / tel_d) * 206264.8062f;
}

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

    /* Generate 16x16 anti-aliased pupil miniature for Panel 1 */
    float                pup16[256] = { 0 };
    cacaosim_pupil_cfg_t pcfg       = {
        .nwf        = 16,
        .pup_diam   = ctx->cfg.pup_diam,
        .cobs       = ctx->cfg.cobs,
        .nspider    = ctx->cfg.nspider,
        .spider_w   = (ctx->cfg.spider_w > 0.0f && ctx->cfg.spider_w < 0.04f)
                          ? 0.04f
                          : ctx->cfg.spider_w,
        .spider_deg = ctx->cfg.spider_deg,
    };
    cacaosim_make_pupil(&pcfg, pup16);

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

    /* ── Panel 1: Optical Bench Overview + 2D Pupil Preview + Star Spectrum ── */
    char hdr_title[128];
    snprintf(hdr_title, sizeof(hdr_title), "CACAOSIM OPTICAL BENCH : %s", ctx->cfg.name);
    panel_top(&ev, hdr_title, ev.pal->fg_title, ev.pal->fg_title);

    const char *sep = ev.ascii ? " | " : " │ ";
    for (int r = 0; r < 10; r++)
    {
        sbuf_t ln;
        sb_init(&ln);
        sb_puts(&ln, " ");
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
            sb_printf(&ln, "%.1f Hz", ctx->cfg.master_hz);
            sb_rst(&ln, ev.lvl);
            sb_fg(&ln, ev.lvl, ev.pal->fg_muted);
            sb_printf(&ln, " (dt=%.0f us)%swf=", ctx->dt_master_s * 1.0e6f, sep);
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

    /* ── Panel 2: 2D Branching Optical Path & Light-Splitting Schematic ────── */
    panel_top(&ev, "OPTICAL PATH & LIGHT-SPLITTING SCHEMATIC", ev.pal->fg_conn, ev.pal->fg_fps);

    {
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

    /* ── Panel 3: Active Mirrors & Wavefront Sensors / Science Cameras ─────── */
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
                if (nd->cfg.dm.dyn_mode == 1)
                {
                    snprintf(dyn, sizeof(dyn), "RC %.0fus, %.1ffr", nd->cfg.dm.tau_sec * 1.0e6f,
                             nd->cfg.dm.delay_frames);
                }
                else if (nd->cfg.dm.dyn_mode == 2)
                {
                    snprintf(dyn, sizeof(dyn), "2nd %.0fHz, %.1ffr", nd->cfg.dm.f0_hz,
                             nd->cfg.dm.delay_frames);
                }
                else
                {
                    snprintf(dyn, sizeof(dyn), "inst, %.1ffr", nd->cfg.dm.delay_frames);
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

                float hz     = (nd->eff_navg > 0)
                                   ? (ctx->cfg.master_hz / (float) nd->eff_navg)
                                   : 0.0f;
                float bw     = (nd->cfg.bw_frac > 0.0f) ? nd->cfg.bw_frac : 0.15f;
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
                snprintf(rn_s, sizeof(rn_s), "%.2fe %s", ron, reg_lbl);

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
                    snprintf(mspec, sizeof(mspec), "OPD [m] %.0fnm x%.2f", nd->cfg.wfe_rms_nm,
                             nd->cfg.wfe_gain);
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

    /* ── Workflow Footer ───────────────────────────────────────────────────── */
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
