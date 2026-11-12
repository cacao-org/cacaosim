// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_internal.h
 * @brief   Internal shared structures and prototypes for the cacaosim-CTRL TUI engine
 */

#ifndef CACAOSIM_TUI_INTERNAL_H
#define CACAOSIM_TUI_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "bench/cacaosim_tui.h"

/* ── Cell Attributes & RGB Palettes ───────────────────────────────────────── */

#define CS_ATTR_BOLD      0x01u
#define CS_ATTR_DIM       0x02u
#define CS_ATTR_UNDERLINE 0x04u
#define CS_ATTR_REV       0x08u

typedef struct
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} cs_tui_rgb_t;

typedef struct
{
    char         ch[5];  /**< Single ASCII byte or UTF-8 symbol + NUL */
    cs_tui_rgb_t fg;
    cs_tui_rgb_t bg;
    uint8_t      attrs;
    uint8_t      has_bg;
} cs_cell_t;

typedef struct
{
    const char  *name;
    const char  *label;
    cs_tui_rgb_t bg_bar;
    cs_tui_rgb_t bg_panel;
    cs_tui_rgb_t bg_panel_alt;
    cs_tui_rgb_t bg_selected;
    cs_tui_rgb_t bg_hover;
    cs_tui_rgb_t bg_popup;
    cs_tui_rgb_t bg_input;
    cs_tui_rgb_t fg_border;
    cs_tui_rgb_t fg_border_focus;
    cs_tui_rgb_t fg_title;
    cs_tui_rgb_t fg_text;
    cs_tui_rgb_t fg_bright;
    cs_tui_rgb_t fg_dim;
    cs_tui_rgb_t fg_muted;
    cs_tui_rgb_t fg_stream;
    cs_tui_rgb_t fg_fps;
    cs_tui_rgb_t fg_proc;
    cs_tui_rgb_t fg_active;
    cs_tui_rgb_t fg_warn;
    cs_tui_rgb_t fg_error;
    cs_tui_rgb_t fg_accent;
    cs_tui_rgb_t fg_conn;
} cs_tui_pal_t;

typedef struct
{
    int        cols;
    int        rows;
    int        color_level;
    int        ascii_only;
    int        force_full;
    cs_cell_t *back;
    cs_cell_t *front;
    char      *out_buf;
    size_t     out_cap;
    uint64_t   bytes_window;
    double     win_t0_sec;
    double     tx_kbps;
} cs_screen_t;

/* ── Keyboard & Mouse Input Codes ─────────────────────────────────────────── */

enum
{
    CS_KEY_NONE       = 0,
    CS_KEY_ESC        = 27,
    CS_KEY_ENTER      = 13,
    CS_KEY_TAB        = 9,
    CS_KEY_BACKSPACE  = 127,
    CS_KEY_UP         = 1000,
    CS_KEY_DOWN       = 1001,
    CS_KEY_LEFT       = 1002,
    CS_KEY_RIGHT      = 1003,
    CS_KEY_HOME       = 1004,
    CS_KEY_END        = 1005,
    CS_KEY_PGUP       = 1006,
    CS_KEY_PGDN       = 1007,
    CS_KEY_DEL        = 1008,
    CS_KEY_BTAB       = 1009,
    CS_KEY_SHIFT_UP   = 1010,
    CS_KEY_SHIFT_DOWN = 1011,
    CS_KEY_F1         = 1021,
    CS_KEY_F2         = 1022,
    CS_KEY_F3         = 1023,
    CS_KEY_F4         = 1024,
    CS_KEY_F5         = 1025,
    CS_KEY_F6         = 1026,
    CS_KEY_F7         = 1027,
    CS_KEY_F8         = 1028,
    CS_KEY_F9         = 1029,
    CS_KEY_F10        = 1030,
    CS_KEY_MOUSE      = 1100,
    CS_KEY_PASTE      = 1101,
    CS_KEY_RESIZE     = 1102
};

enum
{
    CS_MOUSE_LEFT        = 0,
    CS_MOUSE_MIDDLE      = 1,
    CS_MOUSE_RIGHT       = 2,
    CS_MOUSE_RELEASE     = 3,
    CS_MOUSE_DRAG_LEFT   = 32,
    CS_MOUSE_MOVE        = 35,
    CS_MOUSE_WHEEL_UP    = 64,
    CS_MOUSE_WHEEL_DOWN  = 65
};

typedef struct
{
    int  key;
    int  mx;
    int  my;
    int  mbtn;
    int  mrelease;
    char paste[256];
} cs_input_ev_t;

/* ── Hit-Testing & Interactive Regions ────────────────────────────────────── */

#define CS_MAX_HITS 256

typedef enum
{
    CS_HIT_NONE         = 0,
    CS_HIT_VIEW_TAB     = 1,  /**< arg = view index [0..4] */
    CS_HIT_HELP_BTN     = 2,
    CS_HIT_ELEM_ROW     = 3,  /**< arg = item index (0=[bench], 1..N=nodes[arg-1]) */
    CS_HIT_SCHEM_NODE   = 4,  /**< arg = item index (0=[bench], 1..N=nodes[arg-1]) */
    CS_HIT_PARAM_ROW    = 5,  /**< arg = parameter row index */
    CS_HIT_SUBTAB       = 6,  /**< arg = inspector subtab [0..2] */
    CS_HIT_ACT_ADD      = 7,
    CS_HIT_ACT_CLONE    = 8,
    CS_HIT_ACT_UP       = 9,
    CS_HIT_ACT_DOWN     = 10,
    CS_HIT_ACT_SWAP     = 11,
    CS_HIT_ACT_WIRE     = 12,
    CS_HIT_ACT_DEL      = 13,
    CS_HIT_ACT_UNDO     = 14,
    CS_HIT_ACT_PRESET   = 15,
    CS_HIT_ACT_SAVE     = 16,
    CS_HIT_ACT_EDITOR   = 17,
    CS_HIT_ACT_DEPLOY_F = 18,
    CS_HIT_ACT_DEPLOY_M = 19,
    CS_HIT_ACT_STOP     = 20,
    CS_HIT_ACT_MKPUP    = 21,
    CS_HIT_ACT_ATMTURB  = 22,
    CS_HIT_THEME_BTN    = 23,
    CS_HIT_TOOLTIP_BTN  = 24,
    CS_HIT_EXIT_BTN     = 25,
    CS_HIT_SPLIT_LEFT   = 26, /**< Drag handle for left Elements pane width */
    CS_HIT_SPLIT_SCHEM  = 27, /**< Drag handle for top/bottom Schematic/Inspector split */
    CS_HIT_MODAL_ROW    = 28, /**< arg = modal row index */
    CS_HIT_MODAL_OK     = 29,
    CS_HIT_MODAL_CANCEL = 30,
    CS_HIT_MODAL_EXTRA  = 31
} cs_hit_kind_t;

typedef struct
{
    cs_hit_kind_t kind;
    int           arg;
    int           x;
    int           y;
    int           w;
    int           h;
    char          tip[96];
} cs_hit_t;

/* ── Live MILK_SHM_DIR Stream & FPS Telemetry ─────────────────────────────── */

#define CS_MAX_MON_STREAMS 48
#define CS_MAX_MON_FPS     24

typedef struct
{
    char     name[64];
    char     role[24];
    char     owner[32];
    int      expected;
    int      exists;
    int      alive;
    uint8_t  naxis;
    uint32_t size[3];
    uint8_t  datatype;
    uint64_t cnt0;
    double   hz;
    pid_t    writer_pid;
    uint64_t prev_cnt0;
    double   prev_t_sec;
    int      has_prev;
} cs_mon_stream_t;

typedef struct
{
    char  name[64];
    char  descr[80];
    pid_t conf_pid;
    pid_t run_pid;
    int   conf_alive;
    int   run_alive;
    float strehl_inst;
    float strehl_ave;
    int   has_strehl;
} cs_mon_fps_t;

typedef struct
{
    char            shm_dir[256];
    double          last_scan_sec;
    int             n_streams;
    int             n_live_streams;
    cs_mon_stream_t streams[CS_MAX_MON_STREAMS];
    int             n_fps;
    int             n_running_fps;
    cs_mon_fps_t    fps[CS_MAX_MON_FPS];
    float           strehl_inst;
    float           strehl_ave;
    int             has_strehl;
} cs_shm_mon_t;

/* ── Declarative Inspector Parameter Descriptors ──────────────────────────── */

#define CS_MAX_PARAMS 24

typedef enum
{
    CS_PTYPE_FLOAT  = 0,
    CS_PTYPE_INT    = 1,
    CS_PTYPE_UINT   = 2,
    CS_PTYPE_BOOL   = 3,
    CS_PTYPE_ENUM   = 4,
    CS_PTYPE_STR    = 5,
    CS_PTYPE_BEAM   = 6, /**< Upstream input beam (supports [w] Wire picker) */
    CS_PTYPE_ID     = 7  /**< Node section ID (validated & unique) */
} cs_ptype_t;

typedef struct
{
    int         id;
    const char *key;
    const char *label;
    const char *unit;
    const char *tip;
    cs_ptype_t  type;
    double      min_v;
    double      max_v;
    double      step_v;
    int         n_enum;
    const char *enum_names[6];
} cs_param_desc_t;

/* ── Undo/Redo Ring & Modals ──────────────────────────────────────────────── */

#define CS_UNDO_CAP  24
#define CS_LOG_LINES 32

typedef struct
{
    cacaosim_bench_cfg_t snaps[CS_UNDO_CAP];
    int                  sel_item[CS_UNDO_CAP];
    int                  count;
    int                  pos;
} cs_undo_ring_t;

typedef enum
{
    CS_MODAL_NONE    = 0,
    CS_MODAL_ADD     = 1,
    CS_MODAL_WIRE    = 2,
    CS_MODAL_PRESET  = 3,
    CS_MODAL_SAVE    = 4,
    CS_MODAL_OPEN    = 5,
    CS_MODAL_THEME   = 6,
    CS_MODAL_CONFIRM = 7,
    CS_MODAL_HELP    = 8
} cs_modal_t;

typedef enum
{
    CS_FOCUS_ELEMS  = 0,
    CS_FOCUS_SCHEM  = 1,
    CS_FOCUS_PARAMS = 2
} cs_focus_pane_t;

/* ── Master TUI Application State ─────────────────────────────────────────── */

typedef struct
{
    cacaosim_tui_opts_t  opts;
    char                 file_path[512];
    uint64_t             disk_mtime_ns;
    int                  dirty;
    int                  running;
    uint64_t             frame_tick;

    /* Configuration, cached zero-FFT analysis, and 16x16 pupil preview */
    cacaosim_bench_cfg_t cfg;
    cacaosim_bench_ctx_t ctx;
    int                  dag_valid;
    char                 dag_err[256];
    float                pup16[256];
    double               pup16_sum;
    int                  ana_dirty;

    /* Active view, pane focus, selections, scroll offsets, and split ratios */
    int                  view;          /**< cacaosim_tui_view_t [0..4] */
    cs_focus_pane_t      focus;         /**< Active pane in F2:BUILDER */
    int                  sel_item;      /**< 0 = [bench], 1..n_nodes = cfg.nodes[sel_item-1] */
    int                  elem_scroll;
    int                  schem_scroll_y;
    int                  schem_scroll_x;
    int                  subtab;        /**< Inspector subtab: 0=OPTICS, 1=DYN/DET, 2=SHM/WIRING */
    int                  param_sel;
    int                  param_scroll;
    int                  table_scroll;
    int                  status_scroll;
    int                  source_scroll;

    int                  left_w;        /**< Width of Elements pane [26..52] */
    int                  schem_h_pct;   /**< Height % of Schematic vs Inspector [30..70] */
    int                  drag_split;    /**< 0=none, 1=left_w, 2=schem_h */

    /* Inline parameter value editor */
    int                  editing;
    int                  edit_cursor;
    char                 edit_buf[128];
    cs_param_desc_t      params[CS_MAX_PARAMS];
    int                  n_params;

    /* Active floating modal state */
    cs_modal_t           modal;
    int                  modal_sel;
    int                  modal_beam_idx; /**< Target beam override in Add modal (-1=auto) */
    int                  confirm_action; /**< 1=quit unsaved, 2=load preset */
    int                  pending_preset;
    char                 modal_input[512];
    int                  modal_cursor;

    /* Mouse tracking & hover tooltips */
    int                  mouse_x;
    int                  mouse_y;
    int                  show_tooltips;
    cs_hit_t             hits[CS_MAX_HITS];
    int                  n_hits;

    /* Undo/Redo history, SHM telemetry, status toast, and activity log */
    cs_undo_ring_t       undo;
    cs_shm_mon_t         shm;
    char                 toast[256];
    int                  toast_level;   /**< 0=info, 1=ok, 2=warn, 3=err */
    double               toast_until_s;
    char                 logs[CS_LOG_LINES][160];
    int                  n_logs;

    cs_screen_t          scr;
} cs_tui_app_t;

/* ── Terminal & Cell Buffer API (cacaosim_tui_ansi.c) ─────────────────────── */

double              cs_tui_now_sec(void);
int                 cs_tui_detect_color_level(int requested);
int                 cs_tui_detect_ascii(int requested);
const cs_tui_pal_t *cs_tui_get_palette(int theme);
cs_tui_rgb_t        cs_tui_wavelength_rgb(float lam_um, int theme);

int  cs_tui_term_enter(int *out_cols, int *out_rows);
void cs_tui_term_exit(void);
void cs_tui_term_query_size(int *out_cols, int *out_rows);
int  cs_tui_drain_captured_stdio(cs_tui_app_t *app);
int  cs_tui_poll_input(cs_input_ev_t *ev, int timeout_ms);

int  cs_scr_init(cs_screen_t *s, int cols, int rows, int color_level, int ascii_only);
void cs_scr_resize(cs_screen_t *s, int cols, int rows);
void cs_scr_free(cs_screen_t *s);
void cs_scr_clear(cs_screen_t *s, cs_tui_rgb_t fg, cs_tui_rgb_t bg);
void cs_scr_flush(cs_screen_t *s, int out_fd);
int  cs_scr_dump_plain(const cs_screen_t *s, char *out, size_t cap);

void cs_put_cell(
    cs_screen_t *s,
    int          x,
    int          y,
    const char  *glyph,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs,
    int          has_bg);

void cs_fill_rect(
    cs_screen_t *s,
    int          x,
    int          y,
    int          w,
    int          h,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg);

int cs_put_str(
    cs_screen_t *s,
    int          x,
    int          y,
    int          max_cols,
    const char  *str,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs,
    int          has_bg);

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
    ...) __attribute__((format(printf, 9, 10)));

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
    cs_tui_rgb_t bg);

void cs_draw_shadow(cs_screen_t *s, int x, int y, int w, int h, const cs_tui_pal_t *pal);

void cs_draw_pupil16(
    cs_screen_t        *s,
    int                 x,
    int                 y,
    const float        *pup16,
    const cs_tui_pal_t *pal);

float cs_tui_spec_mean_lambda(const cacaosim_spec_t *spec);

void cs_draw_spectrum32(
    cs_screen_t           *s,
    int                    x,
    int                    y,
    int                    width,
    const cacaosim_spec_t *spec,
    int                    theme,
    cs_tui_rgb_t           bg);

/* ── Live SHM & Deploy API (cacaosim_tui_shm.c) ───────────────────────────── */

void cs_shm_scan(cs_shm_mon_t *mon, const cacaosim_bench_cfg_t *cfg, int force);
int  cs_shm_find_stream(const cs_shm_mon_t *mon, const char *name);
int  cs_tui_exec_deploy(cs_tui_app_t *app, const char *mode);
int  cs_tui_exec_stop(cs_tui_app_t *app);
int  cs_tui_exec_mkpupil(cs_tui_app_t *app);

/* ── Parameter Inspector API (cacaosim_tui_params.c) ──────────────────────── */

int  cs_params_build(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    int                         subtab,
    cs_param_desc_t            *out,
    int                         max_out);

void cs_param_format_val(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    const cs_param_desc_t      *pd,
    char                       *buf,
    size_t                      cap);

int  cs_param_step(
    cacaosim_bench_cfg_t  *cfg,
    int                    sel_item,
    const cs_param_desc_t *pd,
    int                    dir,
    int                    coarse,
    char                  *err_buf,
    size_t                 err_cap);

int  cs_param_set_str(
    cacaosim_bench_cfg_t  *cfg,
    int                    sel_item,
    const cs_param_desc_t *pd,
    const char            *val_str,
    char                  *err_buf,
    size_t                 err_cap);

/* ── Frame Rendering & Helpers (cacaosim_tui_render.c & cacaosim_tui.c) ───── */

void cs_tui_add_hit(
    cs_tui_app_t *app,
    cs_hit_kind_t kind,
    int           arg,
    int           x,
    int           y,
    int           w,
    int           h,
    const char   *tip);

void cs_tui_log(cs_tui_app_t *app, int level, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

void cs_tui_refresh_analysis(cs_tui_app_t *app);
void cs_tui_render_frame(cs_tui_app_t *app);
void cs_render_dag_canvas(
    cs_tui_app_t *app,
    int           x,
    int           y,
    int           w,
    int           h,
    int           wide_spectra);
void cs_render_view_schematic(cs_tui_app_t *app, int x, int y, int w, int h);
void cs_render_view_tables(cs_tui_app_t *app, int x, int y, int w, int h);
void cs_render_view_status(cs_tui_app_t *app, int x, int y, int w, int h);
void cs_render_view_source(cs_tui_app_t *app, int x, int y, int w, int h);

#endif /* CACAOSIM_TUI_INTERNAL_H */
