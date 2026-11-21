// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_internal.h
 * @brief   Internal shared structures and prototypes for the cacaosim-CTRL TUI engine
 *
 * Architectural Role in cacaosim-CTRL:
 *   Private internal header shared across the five `cacaosim-CTRL` translation units
 *   (`cacaosim_tui.c`, `cacaosim_tui_ansi.c`, `cacaosim_tui_shm.c`, `cacaosim_tui_params.c`, and
 *   `cacaosim_tui_render.c`). Defines the unified data model uniting all subsystems:
 *   - 2D cell-grid double buffering (`cs_cell_t`, `cs_screen_t`) and semantic 24-bit / 256-color /
 *     16-color palettes (`cs_tui_rgb_t`, `cs_tui_pal_t`).
 *   - Decoded VT/CSI keyboard, SGR 1006 mouse, bracketed-paste, and resize events
 *     (`cs_input_ev_t`).
 *   - Immediate-mode rectangular mouse hit-test regions (`cs_hit_kind_t`, `cs_hit_t`).
 *   - Non-blocking `MILK_SHM_DIR` image-stream and FPS process telemetry (`cs_mon_stream_t`,
 *     `cs_mon_fps_t`, `cs_shm_mon_t`).
 *   - Schema-driven parameter inspector descriptors (`cs_ptype_t`, `cs_param_desc_t`).
 *   - Circular full-snapshot undo/redo history (`cs_undo_ring_t`), modal overlays (`cs_modal_t`),
 *     cached zero-FFT optical DAG / pupil state, and master session context (`cs_tui_app_t`).
 *
 * Design Rationale & Motivation:
 *   1. Custom Cell Canvas vs `ncurses`: Storing explicit 24-bit RGB triplets and inline 4-byte
 *      UTF-8 glyphs (`char ch[5]`) per cell enables TrueColor wavelength colormaps, Unicode
 *      Braille (`U+2800..U+28FF`), half-blocks (`U+2580`/`U+2584`), and vertical bar sparklines
 *      without `ncursesw` wide-character or `terminfo` constraints, while allowing synchronized
 *      delta flushing (`DECSET 2026`) and TTY-free plain-text dumping for CI tests.
 *   2. Zero Per-Frame Heap Allocation: All hit-test tables, parameter schemas, SHM/FPS monitors,
 *      undo snapshots, and log lines reside in bounded fixed-capacity arrays inside `cs_tui_app_t`,
 *      eliminating allocator churn during continuous 10-30 Hz telemetry refreshes.
 *   3. Immediate-Mode Hit Registry: Rebuilding `app->hits[]` alongside widget drawing guarantees
 *      mouse click, drag-splitter, and hover-tooltip coordinates never drift from rendered cells.
 */

#ifndef CACAOSIM_TUI_INTERNAL_H
#define CACAOSIM_TUI_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "bench/cacaosim_tui.h"

/* ── Cell Attributes & RGB Palettes ───────────────────────────────────────── */

/** @name Cell SGR Text Attribute Bitflags (`cs_cell_t.attrs`)
 *  Bitmask flags mapped to ECMA-48 Select Graphic Rendition codes (`1`, `2`, `4`, `7`).
 *  @{ */
#define CS_ATTR_BOLD      0x01u /**< Bold / increased intensity (`\033[1m`) */
#define CS_ATTR_DIM       0x02u /**< Faint / decreased intensity (`\033[2m`) */
#define CS_ATTR_UNDERLINE 0x04u /**< Single underline (`\033[4m`) */
#define CS_ATTR_REV       0x08u /**< Reverse video / swap fg & bg (`\033[7m`) */
/** @} */

/**
 * @brief 24-bit sRGB color triplet `[0..255]` per channel.
 *
 * Stored at full 8-bit-per-channel precision in palettes and screen cells so TrueColor terminals
 * render exact colors while 256-color and 16-color terminals quantize only at flush time.
 */
typedef struct
{
    uint8_t r; /**< Red channel `[0..255]` */
    uint8_t g; /**< Green channel `[0..255]` */
    uint8_t b; /**< Blue channel `[0..255]` */
} cs_tui_rgb_t;

/**
 * @brief Single character cell in the `cs_screen_t` front and back 2D grids.
 *
 * Holds one single-column ASCII character or single-width UTF-8 codepoint (up to 4 UTF-8 bytes
 * plus NUL terminator) together with 24-bit foreground/background colors and SGR attribute bits.
 * Keeping `ch[5]` inline avoids per-cell heap pointers and lets cell equality reduce to a fast
 * fixed-size comparison during delta flushing.
 */
typedef struct
{
    char         ch[5];  /**< Single ASCII byte or up-to-4-byte UTF-8 glyph + NUL terminator */
    cs_tui_rgb_t fg;     /**< Foreground glyph color (24-bit sRGB) */
    cs_tui_rgb_t bg;     /**< Background cell fill color (24-bit sRGB, active when `has_bg != 0`) */
    uint8_t      attrs;  /**< Bitwise OR of `CS_ATTR_*` SGR style flags */
    uint8_t      has_bg; /**< 1 to emit explicit background fill (`color_level >= 2`), 0 default */
} cs_cell_t;

/**
 * @brief Semantic color palette defining a `milk-CTRL`-compatible visual theme.
 *
 * Decouples widget and schematic rendering logic from hardcoded RGB values so all five built-in
 * themes (Dark Slate, Observatory Red night-vision, Okabe-Ito colorblind-safe, Paper Light, and
 * Nordic) share identical layout and contrast hierarchy.
 */
typedef struct
{
    const char  *name;            /**< Short CLI theme identifier (e.g., `"dark"`, `"night"`) */
    const char  *label;           /**< Human-readable title shown in the Theme Picker modal */
    cs_tui_rgb_t bg_bar;          /**< Top tab bar and bottom status/action bar background */
    cs_tui_rgb_t bg_panel;        /**< Primary pane interior background */
    cs_tui_rgb_t bg_panel_alt;    /**< Alternating table row / secondary card background */
    cs_tui_rgb_t bg_selected;     /**< Highlighted / active selection row background */
    cs_tui_rgb_t bg_hover;        /**< Mouse-hover highlight background */
    cs_tui_rgb_t bg_popup;        /**< Floating modal dialog and tooltip background */
    cs_tui_rgb_t bg_input;        /**< Inline text-edit field background */
    cs_tui_rgb_t fg_border;       /**< Unfocused pane border and separator line color */
    cs_tui_rgb_t fg_border_focus; /**< Focused pane double-border highlight color */
    cs_tui_rgb_t fg_title;        /**< Pane header title foreground color */
    cs_tui_rgb_t fg_text;         /**< Standard body and parameter label text color */
    cs_tui_rgb_t fg_bright;       /**< High-emphasis value and active tab foreground color */
    cs_tui_rgb_t fg_dim;          /**< De-emphasized hints, inactive glyphs, and grid dots */
    cs_tui_rgb_t fg_muted;        /**< Secondary metadata, units, and unfocused tab labels */
    cs_tui_rgb_t fg_stream;       /**< SHM image stream names and active pupil glyph color */
    cs_tui_rgb_t fg_fps;          /**< `milk` FPS process names and control badges */
    cs_tui_rgb_t fg_proc;         /**< Compute unit / optical processor category accent */
    cs_tui_rgb_t fg_active;       /**< Healthy / live / running status indicator (green/teal) */
    cs_tui_rgb_t fg_warn;         /**< Unsaved-dirty, degraded, or caution indicator (amber) */
    cs_tui_rgb_t fg_error;        /**< Invalid DAG, missing stream, or error indicator (red) */
    cs_tui_rgb_t fg_accent;       /**< Hotkey mnemonics, cursor badges, and Strehl highlights */
    cs_tui_rgb_t fg_conn;         /**< Optical beam routing wires and connector arrows */
} cs_tui_pal_t;

/**
 * @brief Double-buffered 2D virtual terminal canvas and delta-flush presenter state.
 *
 * Rendering routines mutate only `back[]`. `cs_scr_flush()` diffs `back[]` against `front[]`,
 * serializes minimal cursor jumps + SGR transitions + UTF-8 bytes into `out_buf` inside a single
 * DECSET 2026 synchronized update packet, updates `front[]`, and tracks outbound bandwidth
 * (`tx_kbps`) for status-bar diagnostics.
 */
typedef struct
{
    int        cols;         /**< Canvas width in character columns (`>= 40`) */
    int        rows;         /**< Canvas height in character rows (`>= 12`) */
    int        color_level;  /**< Active SGR mode: 0=mono, 1=16c, 2=256c, 3=24-bit TrueColor */
    int        ascii_only;   /**< 1 for pure 7-bit ASCII borders/blocks, 0 for UTF-8 glyphs */
    int        force_full;   /**< 1 to bypass `front[]` diff and repaint every cell on next flush */
    cs_cell_t *back;         /**< Writable next-frame cell grid `[rows * cols]` */
    cs_cell_t *front;        /**< Last-flushed terminal cell grid `[rows * cols]` */
    char      *out_buf;      /**< Preallocated ANSI escape serialization buffer */
    size_t     out_cap;      /**< Capacity of `out_buf` in bytes */
    uint64_t   bytes_window; /**< Cumulative ANSI bytes written within current 1 s meter window */
    double     win_t0_sec;   /**< Monotonic start timestamp [s] of current bandwidth window */
    double     tx_kbps;      /**< Measured terminal output throughput [KiB/s] */
} cs_screen_t;

/* ── Keyboard & Mouse Input Codes ─────────────────────────────────────────── */

/**
 * @brief Normalized virtual key and terminal event codes stored in `cs_input_ev_t.key`.
 *
 * Printable ASCII characters (`32..126`) and Ctrl chords (`1..26`) pass through as their raw byte
 * values, while multi-byte CSI/SS3 escape sequences, mouse reports, bracketed pastes, and terminal
 * resizes map onto synthetic constants `>= 1000` so switch dispatch cannot collide with byte input.
 */
enum
{
    CS_KEY_NONE       = 0,    /**< No event available within poll timeout */
    CS_KEY_ESC        = 27,   /**< Standalone Escape key (`\033`) */
    CS_KEY_ENTER      = 13,   /**< Carriage return / Enter (`\r` or `\n`) */
    CS_KEY_TAB        = 9,    /**< Forward Tab (`\t`) */
    CS_KEY_BACKSPACE  = 127,  /**< Backspace (`0x7F` or `0x08`) */
    CS_KEY_UP         = 1000, /**< Up arrow (`CSI A` / `SS3 A`) */
    CS_KEY_DOWN       = 1001, /**< Down arrow (`CSI B` / `SS3 B`) */
    CS_KEY_LEFT       = 1002, /**< Left arrow (`CSI D` / `SS3 D`) */
    CS_KEY_RIGHT      = 1003, /**< Right arrow (`CSI C` / `SS3 C`) */
    CS_KEY_HOME       = 1004, /**< Home key (`CSI H` / `SS3 H` / `CSI 1~` / `CSI 7~`) */
    CS_KEY_END        = 1005, /**< End key (`CSI F` / `SS3 F` / `CSI 4~` / `CSI 8~`) */
    CS_KEY_PGUP       = 1006, /**< Page Up (`CSI 5~`) */
    CS_KEY_PGDN       = 1007, /**< Page Down (`CSI 6~`) */
    CS_KEY_DEL        = 1008, /**< Forward Delete (`CSI 3~`) */
    CS_KEY_BTAB       = 1009, /**< Shift+Tab / Back-Tab (`CSI Z`) */
    CS_KEY_SHIFT_UP   = 1010, /**< Shift+Up arrow (`CSI 1;2A`) for node reordering */
    CS_KEY_SHIFT_DOWN = 1011, /**< Shift+Down arrow (`CSI 1;2B`) for node reordering */
    CS_KEY_F1         = 1021, /**< F1 (`SS3 P` / `CSI 11~`): Help modal */
    CS_KEY_F2         = 1022, /**< F2 (`SS3 Q` / `CSI 12~`): Builder workspace */
    CS_KEY_F3         = 1023, /**< F3 (`SS3 R` / `CSI 13~`): Full Schematic workspace */
    CS_KEY_F4         = 1024, /**< F4 (`SS3 S` / `CSI 14~`): Hardware & Photometry Tables */
    CS_KEY_F5         = 1025, /**< F5 (`CSI 15~`): Live SHM & FPS Status workspace */
    CS_KEY_F6         = 1026, /**< F6 (`CSI 17~`): `.bench` Source preview workspace */
    CS_KEY_F7         = 1027, /**< F7 (`CSI 18~`) */
    CS_KEY_F8         = 1028, /**< F8 (`CSI 19~`) */
    CS_KEY_F9         = 1029, /**< F9 (`CSI 20~`) */
    CS_KEY_F10        = 1030, /**< F10 (`CSI 21~`): Quit shortcut */
    CS_KEY_MOUSE      = 1100, /**< SGR 1006 mouse press, release, motion, or wheel event */
    CS_KEY_PASTE      = 1101, /**< Bracketed paste payload (`\033[200~...\033[201~`) */
    CS_KEY_RESIZE     = 1102  /**< Asynchronous `SIGWINCH` terminal resize notification */
};

/**
 * @brief SGR 1006 mouse button, motion, and wheel codes stored in `cs_input_ev_t.mbtn`.
 *
 * Follows xterm SGR 1006 bit encoding (`\033[<btn;col;rowM/m`): low 2 bits encode button index
 * (`0`=left, `1`=middle, `2`=right, `3`=no-button/release), bit 5 (`+32`) flags pointer motion,
 * and bit 6 (`+64`) flags vertical scroll-wheel ticks.
 */
enum
{
    CS_MOUSE_LEFT        = 0,  /**< Left button press/release */
    CS_MOUSE_MIDDLE      = 1,  /**< Middle button press/release */
    CS_MOUSE_RIGHT       = 2,  /**< Right button press/release */
    CS_MOUSE_RELEASE     = 3,  /**< Legacy button-up code (SGR 1006 uses `mrelease == 1` instead) */
    CS_MOUSE_DRAG_LEFT   = 32, /**< Pointer motion while holding left button (`0 + 32`) */
    CS_MOUSE_MOVE        = 35, /**< Passive hover motion with no buttons pressed (`3 + 32`) */
    CS_MOUSE_WHEEL_UP    = 64, /**< Scroll wheel up tick (`0 + 64`) */
    CS_MOUSE_WHEEL_DOWN  = 65  /**< Scroll wheel down tick (`1 + 64`) */
};

/**
 * @brief Decoded terminal input event returned by `cs_tui_poll_input()`.
 */
typedef struct
{
    int  key;        /**< ASCII byte `[1..126]` or `CS_KEY_*` synthetic code */
    int  mx;         /**< 0-based terminal column of mouse event (`CS_KEY_MOUSE`) */
    int  my;         /**< 0-based terminal row of mouse event (`CS_KEY_MOUSE`) */
    int  mbtn;       /**< Raw SGR 1006 button/motion/wheel code (`CS_MOUSE_*` + modifier bits) */
    int  mrelease;   /**< 1 on SGR button release (`'m'` suffix), 0 on press/motion/wheel (`'M'`) */
    char paste[256]; /**< Sanitized printable text captured from bracketed paste (`CS_KEY_PASTE`) */
} cs_input_ev_t;

/* ── Hit-Testing & Interactive Regions ────────────────────────────────────── */

/** Maximum number of interactive mouse hit-test rectangles registered in a single frame. */
#define CS_MAX_HITS 256

/**
 * @brief Semantic target classification for immediate-mode mouse hit-testing (`cs_hit_t.kind`).
 *
 * During each `cs_tui_render_frame()`, widgets register their screen bounding boxes with one of
 * these tags plus an integer `arg` payload so mouse clicks, splitter drags, and hover tooltips map
 * directly back to domain actions without parallel widget trees.
 */
typedef enum
{
    CS_HIT_NONE         = 0,  /**< Empty / unused hit slot */
    CS_HIT_VIEW_TAB     = 1,  /**< Top workspace tab button; `arg` = view index `[0..4]` */
    CS_HIT_HELP_BTN     = 2,  /**< Top-bar `[F1:?]` help modal button */
    CS_HIT_ELEM_ROW     = 3,  /**< Left Elements list row; `arg` = `0` (`[bench]`) or `1..N` */
    CS_HIT_SCHEM_NODE   = 4,  /**< 2D Schematic node box; `arg` = `0` (`[bench]`) or `1..N` */
    CS_HIT_PARAM_ROW    = 5,  /**< Inspector parameter row; `arg` = row index in `app->params[]` */
    CS_HIT_SUBTAB       = 6,  /**< Inspector category subtab; `arg` = subtab index `[0..2]` */
    CS_HIT_ACT_ADD      = 7,  /**< Toolbar `[+Add]` optical element button */
    CS_HIT_ACT_CLONE    = 8,  /**< Toolbar `[Clone]` selected node button */
    CS_HIT_ACT_UP       = 9,  /**< Toolbar `[Up]` move node upstream button */
    CS_HIT_ACT_DOWN     = 10, /**< Toolbar `[Dn]` move node downstream button */
    CS_HIT_ACT_SWAP     = 11, /**< Toolbar `[Swap]` splitter/coro output branches button */
    CS_HIT_ACT_WIRE     = 12, /**< Toolbar `[Wire]` upstream input beam picker button */
    CS_HIT_ACT_DEL      = 13, /**< Toolbar `[Del]` remove node & bridge beam button */
    CS_HIT_ACT_UNDO     = 14, /**< Toolbar `[Undo]` configuration snapshot button */
    CS_HIT_ACT_PRESET   = 15, /**< Toolbar `[Preset]` hardware template loader button */
    CS_HIT_ACT_SAVE     = 16, /**< Toolbar `[Save]` write `.bench` file button */
    CS_HIT_ACT_EDITOR   = 17, /**< Launch external `$EDITOR` on `.bench` file button */
    CS_HIT_ACT_DEPLOY_F = 18, /**< Deploy bench in fused single-process `htsim` mode */
    CS_HIT_ACT_DEPLOY_M = 19, /**< Deploy bench as modular multi-process `milk` FPS units */
    CS_HIT_ACT_STOP     = 20, /**< Stop running `cacaosim` / `atmturb` FPS processes */
    CS_HIT_ACT_MKPUP    = 21, /**< Synthesize SHM pupil & static `[wfe]` maps (`mkpupil` + `wfe`) */
    CS_HIT_ACT_ATMTURB  = 22, /**< Toggle `with_atmturb` co-launch flag for deployment */
    CS_HIT_THEME_BTN    = 23, /**< Cycle or open color-palette theme selector */
    CS_HIT_TOOLTIP_BTN  = 24, /**< Toggle floating hover tooltips on/off */
    CS_HIT_EXIT_BTN     = 25, /**< Top/bottom bar `[F10:Quit]` exit button */
    CS_HIT_SPLIT_LEFT   = 26, /**< Vertical splitter bar dragging `left_w` Elements pane width */
    CS_HIT_SPLIT_SCHEM  = 27, /**< Horizontal splitter bar dragging `schem_h_pct` split ratio */
    CS_HIT_MODAL_ROW    = 28, /**< Selectable list item inside active modal; `arg` = row index */
    CS_HIT_MODAL_OK     = 29, /**< Modal primary confirm / `[OK]` button */
    CS_HIT_MODAL_CANCEL = 30, /**< Modal dismiss / `[Cancel]` button */
    CS_HIT_MODAL_EXTRA  = 31, /**< Modal auxiliary control (e.g., target-beam cycle in Add modal) */
    CS_HIT_ACT_ADVANCED = 32, /**< Inspector `[v:Adv]` toggle for advanced `.bench` options */
    CS_HIT_ACT_OPTREF   = 33  /**< Top/Inspector `[?:Opts]` searchable option reference button */
} cs_hit_kind_t;

/**
 * @brief Rectangular screen hit-box registered during frame rendering for mouse interaction.
 */
typedef struct
{
    cs_hit_kind_t kind;    /**< Semantic action/widget kind */
    int           arg;     /**< Kind-specific index payload (tab, node, parameter, or modal row) */
    int           x;       /**< Leftmost 0-based terminal column (inclusive) */
    int           y;       /**< Topmost 0-based terminal row (inclusive) */
    int           w;       /**< Width in character columns (`>= 1`) */
    int           h;       /**< Height in character rows (`>= 1`) */
    char          tip[96]; /**< Optional hover tooltip string displayed near cursor */
} cs_hit_t;

/* ── Live MILK_SHM_DIR Stream & FPS Telemetry ─────────────────────────────── */

/** Maximum `.im.shm` streams tracked simultaneously in the Status & Schematic views. */
#define CS_MAX_MON_STREAMS 48
/** Maximum `.fps.shm` Function Parameter Structures tracked simultaneously. */
#define CS_MAX_MON_FPS     24

/**
 * @brief Non-intrusive metadata and frame-rate telemetry snapshot for a single `.im.shm` stream.
 *
 * Populated by `cs_shm_scan()` via read-only header `mmap` (`sizeof(IMAGE_METADATA)`) without
 * locking semaphores or spinning on `md->write`. Tracks both streams referenced by the active
 * `.bench` (`expected == 1`) and extra AO/simulation streams discovered in `MILK_SHM_DIR`.
 */
typedef struct
{
    char     name[64];   /**< Stream stem name without `.im.shm` suffix (e.g., `"csim_pup"`) */
    char     role[24];   /**< Short functional role label (`"pupil"`, `"dm_cmd"`, `"wfs_cam"`) */
    char     owner[32];  /**< Owning `.bench` section ID (`"[bench]"`, node `id`, or `"shm"`) */
    int      expected;   /**< 1 if referenced by current `.bench` config, 0 if extra SHM file */
    int      exists;     /**< 1 if `<MILK_SHM_DIR>/<name>.im.shm` exists with valid header */
    int      alive;      /**< 1 if owner/creator PID is alive or `cnt0` incremented recently */
    uint8_t  naxis;      /**< Number of array dimensions (`1..3`) from `IMAGE_METADATA.naxis` */
    uint32_t size[3];    /**< Axis dimensions `[nx, ny, nz]` from `IMAGE_METADATA.size` */
    uint8_t  datatype;   /**< `ImageStreamIO` `_DATATYPE_*` pixel type code */
    uint64_t cnt0;       /**< Monotonic write counter `IMAGE_METADATA.cnt0` */
    double   hz;         /**< Exponentially smoothed frame update rate [Hz] from `cnt0` deltas */
    pid_t    writer_pid; /**< Active `ownerPID` (or fallback `creatorPID`) recorded in SHM header */
    uint64_t prev_cnt0;  /**< Previous scan's `cnt0` value for delta-rate estimation */
    double   prev_t_sec; /**< Monotonic timestamp [s] of `prev_cnt0` sample */
    int      has_prev;   /**< 1 once baseline `(prev_cnt0, prev_t_sec)` has been recorded */
} cs_mon_stream_t;

/**
 * @brief Non-intrusive process and Strehl telemetry snapshot for a single `.fps.shm` module.
 *
 * Extracted by `cs_shm_scan()` via read-only `mmap` of `<MILK_SHM_DIR>/<name>.fps.shm`, probing
 * `confpid`/`runpid` liveness via `kill(pid, 0)` and scanning active float32 FPS parameters for
 * `strehl_inst` / `strehl_ave` published by `cacaosim.fpcam` or `cacaosim.htsim`.
 */
typedef struct
{
    char  name[64];    /**< FPS instance stem without `.fps.shm` suffix */
    char  descr[80];   /**< Human-readable module description from `FUNCTION_PARAMETER_STRUCT_MD` */
    pid_t conf_pid;    /**< Configuration (`conf`) process PID recorded in FPS header */
    pid_t run_pid;     /**< Real-time execution (`run`) process PID recorded in FPS header */
    int   conf_alive;  /**< 1 if `conf_pid` is currently alive in the OS process table */
    int   run_alive;   /**< 1 if `run_pid` is currently alive in the OS process table */
    float strehl_inst; /**< Instantaneous science Strehl ratio `[0..1]` if exported by this FPS */
    float strehl_ave;  /**< Long-exposure / moving-average Strehl ratio `[0..1]` if exported */
    int   has_strehl;  /**< 1 if `strehl_inst` or `strehl_ave` parameter was found on this FPS */
} cs_mon_fps_t;

/**
 * @brief Aggregated `MILK_SHM_DIR` telemetry cache refreshed periodically by `cs_shm_scan()`.
 */
typedef struct
{
    char            shm_dir[256];                 /**< Resolved `MILK_SHM_DIR` filesystem path */
    double          last_scan_sec;                /**< Monotonic timestamp [s] of last scan */
    int             n_streams;                    /**< Number of populated entries in `streams[]` */
    int             n_live_streams;               /**< Count of `streams[]` with `exists != 0` */
    cs_mon_stream_t streams[CS_MAX_MON_STREAMS];  /**< Tracked `.im.shm` stream snapshots */
    int             n_fps;                        /**< Number of populated entries in `fps[]` */
    int             n_running_fps;                /**< Count of `fps[]` with `run_alive != 0` */
    cs_mon_fps_t    fps[CS_MAX_MON_FPS];          /**< Tracked `.fps.shm` module snapshots */
    float           strehl_inst;                  /**< Primary live instantaneous Strehl `[0..1]` */
    float           strehl_ave;                   /**< Primary live average Strehl `[0..1]` */
    int             has_strehl;                   /**< 1 if any scanned FPS exported Strehl */
} cs_shm_mon_t;

/* ── Declarative Inspector Parameter Descriptors ──────────────────────────── */

/** Maximum parameter rows exposed simultaneously in any single Inspector subtab. */
#define CS_MAX_PARAMS 40

/**
 * @brief Value and UI-widget classification for a schema-driven Inspector row (`cs_param_desc_t`).
 */
typedef enum
{
    CS_PTYPE_FLOAT  = 0, /**< Clamped floating-point scalar (`min_v..max_v`, stepped by `step_v`) */
    CS_PTYPE_INT    = 1, /**< Clamped signed integer (`min_v..max_v`, stepped by `step_v`) */
    CS_PTYPE_UINT   = 2, /**< Clamped unsigned integer (`min_v..max_v`, stepped by `step_v`) */
    CS_PTYPE_BOOL   = 3, /**< Two-state `ON`/`OFF` toggle (stepped or toggled on Enter/Space) */
    CS_PTYPE_ENUM   = 4, /**< Discrete named option index `[0..n_enum-1]` in `enum_names[]` */
    CS_PTYPE_STR    = 5, /**< Free-form string (SHM stream name, FITS/SHM mask path, beam name) */
    CS_PTYPE_BEAM   = 6, /**< Upstream optical beam input (opens `[w]` Wire picker modal) */
    CS_PTYPE_ID     = 7  /**< Node section identifier (uniqueness-validated via rename helper) */
} cs_ptype_t;

/**
 * @brief Self-describing schema row built dynamically by `cs_params_build()` for the Inspector.
 *
 * Carries UI presentation metadata (`key`, `label`, `unit`, `tip`), numeric clamping and stepping
 * rules (`min_v`, `max_v`, `step_v`), and enum option strings (`enum_names`) alongside an opaque
 * `PID_*` dispatch `id` interpreted by `cs_param_format_val()` and `cs_param_set_str()`.
 */
typedef struct
{
    int         id;            /**< Internal `PID_*` field identifier for getter/setter dispatch */
    const char *key;           /**< `.bench` INI key name shown in muted monospace column */
    const char *label;         /**< Concise human-readable parameter label */
    const char *unit;          /**< Physical or logical unit suffix (`"um"`, `"Hz"`, `"l/D"`) */
    const char *tip;           /**< One-line contextual help shown in footer/hover tooltip */
    cs_ptype_t  type;          /**< Data and interactive widget behavior type */
    double      min_v;         /**< Inclusive minimum bound for numeric/enum types */
    double      max_v;         /**< Inclusive maximum bound for numeric/enum types */
    double      step_v;        /**< Fine increment step (`Left`/`Right`; `5x` when `coarse != 0`) */
    int         n_enum;        /**< Number of valid entries in `enum_names[]` (`CS_PTYPE_ENUM`) */
    const char *enum_names[6]; /**< Canonical option strings for `CS_PTYPE_ENUM` (`<= 6` choices) */
} cs_param_desc_t;

/* ── Undo/Redo Ring & Modals ──────────────────────────────────────────────── */

/** Number of full `cacaosim_bench_cfg_t` states retained in the undo/redo history buffer. */
#define CS_UNDO_CAP  24
/** Number of recent status/command output lines retained in the rolling activity log. */
#define CS_LOG_LINES 32

/**
 * @brief Full-snapshot undo/redo history buffer for `.bench` topology and parameter edits.
 *
 * Because `cacaosim_bench_cfg_t` is a compact value struct (~few KiB) whereas node insertions,
 * deletions, branch swaps, and beam rewirings mutate multiple downstream nodes simultaneously,
 * storing complete value snapshots guarantees exact, bug-free undo/redo without inverse-patch code.
 */
typedef struct
{
    cacaosim_bench_cfg_t snaps[CS_UNDO_CAP];    /**< Saved bench configuration value snapshots */
    int                  sel_item[CS_UNDO_CAP]; /**< Selected element index paired with snapshot */
    int                  count;                 /**< Valid snapshots stored `[0..CS_UNDO_CAP]` */
    int                  pos;                   /**< Active cursor index inside `snaps[]` */
} cs_undo_ring_t;

/**
 * @brief Active modal dialog overlay state in `cs_tui_app_t.modal`.
 *
 * When `modal != CS_MODAL_NONE`, keyboard and mouse events route exclusively to the modal handler
 * while the underlying workspace remains visible dimmed beneath a drop-shadowed popup box.
 */
typedef enum
{
    CS_MODAL_NONE    = 0, /**< No modal open; normal workspace interaction active */
    CS_MODAL_ADD     = 1, /**< `[+Add]` optical element kind + target beam selector */
    CS_MODAL_WIRE    = 2, /**< `[w]` upstream optical beam rewiring picker */
    CS_MODAL_PRESET  = 3, /**< `[Preset]` built-in AO bench template chooser */
    CS_MODAL_SAVE    = 4, /**< `[Save]` destination `.bench` file path prompt */
    CS_MODAL_OPEN    = 5, /**< Open / load `.bench` file path prompt */
    CS_MODAL_THEME   = 6, /**< Color palette theme selector popup */
    CS_MODAL_CONFIRM = 7, /**< Destructive action confirmation (unsaved quit / preset overwrite) */
    CS_MODAL_HELP    = 8, /**< `[F1]` keyboard, mouse, and workflow quick-reference overlay */
    CS_MODAL_OPTIONS = 9  /**< `[?]` searchable `.bench` & schema option reference overlay */
} cs_modal_t;

/**
 * @brief Active keyboard-focus sub-pane within the 3-pane `F2:BUILDER` workspace.
 */
typedef enum
{
    CS_FOCUS_ELEMS  = 0, /**< Left pane: `[bench]` + optical node list & topology toolbar */
    CS_FOCUS_SCHEM  = 1, /**< Top-right pane: 2D optical DAG schematic & spectral sparklines */
    CS_FOCUS_PARAMS = 2  /**< Bottom-right pane: categorized parameter Inspector & inline editor */
} cs_focus_pane_t;

/* ── Master TUI Application State ─────────────────────────────────────────── */

/**
 * @brief Complete single-instance runtime state for an interactive or headless `cacaosim-CTRL` run.
 *
 * Aggregates document state, cached zero-FFT optical/photometric DAG analysis (`ctx`), 16x16
 * anti-aliased pupil preview (`pup16`), viewport navigation/splitter geometry, inline parameter
 * editing buffers, active modal state, immediate-mode mouse hit boxes, undo/redo snapshots, live
 * `MILK_SHM_DIR` telemetry (`shm`), toast/log feedback, and the `cs_screen_t` cell double buffer.
 */
typedef struct
{
    cacaosim_tui_opts_t  opts;          /**< Active startup & deployment options */
    char                 file_path[512];/**< Current `.bench` file path (`""` if untitled) */
    uint64_t             disk_mtime_ns; /**< Last observed `st_mtim` [ns] for external sync */
    int                  dirty;         /**< 1 if in-memory `cfg` has unsaved modifications */
    int                  running;       /**< 1 while main interactive event loop remains active */
    uint64_t             frame_tick;    /**< Monotonic rendered-frame counter */

    /* Configuration, cached zero-FFT analysis, and 16x16 pupil preview */
    cacaosim_bench_cfg_t cfg;           /**< Authoritative editable `.bench` configuration */
    cacaosim_bench_ctx_t ctx;           /**< Cached zero-FFT `cacaosim_bench_analyze()` output */
    int                  dag_valid;     /**< 1 if `ctx` passed topological & beam validation */
    char                 dag_err[256];  /**< Diagnostic reason when `dag_valid == 0` */
    float                pup16[256];    /**< Cached `16x16` entrance pupil amplitude preview */
    double               pup16_sum;     /**< Integrated aperture fill factor of `pup16[]` */
    int                  ana_dirty;     /**< 1 when `cfg` changed and `ctx`/`pup16` need refresh */

    /* Active view, pane focus, selections, scroll offsets, and split ratios */
    int                  view;          /**< Active workspace tab (`cacaosim_tui_view_t`) */
    cs_focus_pane_t      focus;         /**< Focused pane in `F2:BUILDER` (`CS_FOCUS_*`) */
    int                  sel_item;      /**< `0` = `[bench]`, `1..N` = `cfg.nodes[sel_item-1]` */
    int                  elem_scroll;   /**< Vertical scroll offset of left Elements list */
    int                  schem_scroll_y;/**< Vertical scroll offset of 2D Schematic canvas */
    int                  schem_scroll_x;/**< Horizontal scroll offset of 2D Schematic canvas */
    int                  subtab;        /**< Inspector tab: `0`=OPTICS, `1`=DYN/DET, `2`=WIRING */
    int                  show_advanced; /**< 1 to expose all advanced `.bench` keys in Inspector */
    int                  param_sel;     /**< Highlighted row index in `params[0..n_params-1]` */
    int                  param_scroll;  /**< Vertical scroll offset of Inspector parameter rows */
    int                  table_scroll;  /**< Vertical scroll offset in `F4:TABLES` view */
    int                  status_scroll; /**< Vertical scroll offset in `F5:STATUS` view */
    int                  source_scroll; /**< Vertical line scroll offset in `F6:SOURCE` view */

    int                  left_w;        /**< Width in columns of left Elements pane `[26..52]` */
    int                  schem_h_pct;   /**< Schematic height ratio vs Inspector `[30..70]` */
    int                  drag_split;    /**< Active mouse drag: `0`=none, `1`=left, `2`=schem */

    /* Inline parameter value editor */
    int                  editing;       /**< 1 while inline text editor is active on `param_sel` */
    int                  edit_cursor;   /**< Byte/char cursor offset within `edit_buf` */
    char                 edit_buf[256]; /**< Scratch text buffer for inline parameter value entry */
    cs_param_desc_t      params[CS_MAX_PARAMS]; /**< Schema rows for `(sel_item, subtab)` */
    int                  n_params;      /**< Number of active descriptors in `params[]` */

    /* Active floating modal state */
    cs_modal_t           modal;         /**< Currently open modal dialog (`CS_MODAL_NONE` = 0) */
    int                  modal_sel;     /**< Highlighted list index inside active modal */
    int                  modal_beam_idx;/**< Input beam override in Add modal (`-1` = auto) */
    int                  confirm_action;/**< Pending confirm op: `1` = quit unsaved, `2` = preset */
    int                  pending_preset;/**< Preset index awaiting confirmation when `dirty != 0` */
    char                 modal_input[512]; /**< Text input buffer for Save/Open path modals */
    int                  modal_cursor;  /**< Cursor offset inside `modal_input` */
    char                 opt_search[64];/**< Live filter query inside `CS_MODAL_OPTIONS` */
    int                  opt_scroll;    /**< Top visible filtered row in `CS_MODAL_OPTIONS` */
    int                  opt_sel;       /**< Highlighted filtered row in `CS_MODAL_OPTIONS` */

    /* Mouse tracking & hover tooltips */
    int                  mouse_x;       /**< Latest observed 0-based mouse column (`-1` if none) */
    int                  mouse_y;       /**< Latest observed 0-based mouse row (`-1` if none) */
    int                  show_tooltips; /**< 1 to render hover tooltips for `hits[]` under cursor */
    cs_hit_t             hits[CS_MAX_HITS]; /**< Per-frame interactive rectangular hit regions */
    int                  n_hits;        /**< Valid entries in `hits[]` for current frame */

    /* Undo/Redo history, SHM telemetry, status toast, and activity log */
    cs_undo_ring_t       undo;          /**< Snapshot ring for `u` / `Ctrl+Z` / `Ctrl+Y` undo */
    cs_shm_mon_t         shm;           /**< Live `MILK_SHM_DIR` streams & FPS status cache */
    char                 toast[256];    /**< Ephemeral status banner message in bottom bar */
    int                  toast_level;   /**< Banner severity: `0`=info, `1`=ok, `2`=warn, `3`=err */
    double               toast_until_s; /**< Monotonic expiration time [s] for `toast` */
    char                 logs[CS_LOG_LINES][160]; /**< Rolling activity & captured-stdio ring */
    int                  n_logs;        /**< Number of populated lines in `logs[]` */

    cs_screen_t          scr;           /**< Double-buffered 2D terminal cell canvas */
} cs_tui_app_t;

/* ── Terminal & Cell Buffer API (cacaosim_tui_ansi.c) ─────────────────────── */

/**
 * cs_tui_now_sec() - Query monotonic wall-clock time in seconds
 *
 * Role:
 *   Timebase for `MILK_SHM_DIR` scan throttling, `cnt0`-delta frame-rate calculation, status-toast
 *   expiration, and 1 s terminal bandwidth (`tx_kbps`) metering.
 *
 * Rationale:
 *   Uses `CLOCK_MONOTONIC` so NTP slews or leap-second jumps never stall periodic UI scans or
 *   corrupt measured stream frame rates.
 *
 * Return: Monotonic timestamp in seconds (`double`, nanosecond resolution).
 */
double cs_tui_now_sec(void);

/**
 * cs_tui_detect_color_level() - Resolve active SGR color depth from option override or environment
 * @requested: Explicit level (`0..3`), or `< 0` to auto-detect from `NO_COLOR`, `TERM`, `COLORTERM`
 *
 * Role:
 *   Selects monochrome (`0`), 16-color (`1`), 256-color (`2`), or 24-bit TrueColor (`3`) output.
 *
 * Rationale:
 *   Honors the universal `NO_COLOR` convention and `dumb` terminal checks first, then inspects
 *   `COLORTERM=truecolor|24bit` and `TERM=*256color*` without needing a `terminfo` database.
 *
 * Return: Clamped SGR color level in `[0..3]`.
 */
int cs_tui_detect_color_level(int requested);

/**
 * cs_tui_detect_ascii() - Determine whether pure 7-bit ASCII glyphs should replace UTF-8 symbols
 * @requested: Non-zero to force 7-bit ASCII regardless of locale, 0 to auto-detect from locale env
 *
 * Role:
 *   Controls box-drawing, half-block pupil, shadow, and spectral sparkline glyph sets.
 *
 * Rationale:
 *   Automatically falls back to 7-bit ASCII when `LC_ALL` / `LC_CTYPE` / `LANG` is `"C"` or
 *   `"POSIX"` so minimal container or serial-console environments never display mojibake.
 *
 * Return: 1 for pure 7-bit ASCII mode, 0 for UTF-8 mode.
 */
int cs_tui_detect_ascii(int requested);

/**
 * cs_tui_get_palette() - Retrieve the read-only semantic RGB palette for a theme index
 * @theme: Theme enumerator (`cacaosim_view_theme_t`; out-of-range indices clamp to Dark Slate)
 *
 * Role:
 *   Supplies semantic foreground/background RGB colors to all workspace and modal renderers.
 *
 * Rationale:
 *   Centralizes theme lookup with defensive clamping so callers never dereference NULL or invalid
 *   palette indices.
 *
 * Return: Non-NULL pointer to a static `cs_tui_pal_t` descriptor.
 */
const cs_tui_pal_t *cs_tui_get_palette(int theme);

/**
 * cs_tui_wavelength_rgb() - Map an optical wavelength `[um]` to a perceptual or night-safe RGB hue
 * @lam_um: Optical wavelength in micrometers (nominally `0.40..2.40 um`)
 * @theme:  Active theme index (`CACAOSIM_THEME_NIGHT` forces scotopic red-amber tones)
 *
 * Role:
 *   Colors 32-bin optical spectrum bars and passband badges by physical wavelength band (blue ->
 *   cyan-green -> gold -> orange -> NIR crimson -> H/K-band violet).
 *
 * Rationale:
 *   Gives immediate visual distinction between visible WFS arms and near-IR science/PyWFS arms
 *   while preserving dark-adapted night vision when the Observatory Red theme is active.
 *
 * Return: 24-bit `cs_tui_rgb_t` color for @lam_um.
 */
cs_tui_rgb_t cs_tui_wavelength_rgb(float lam_um, int theme);

/**
 * cs_tui_term_enter() - Switch controlling TTY into raw alternate-screen mouse-enabled TUI mode
 * @out_cols: Output initial terminal width in columns (must not be NULL)
 * @out_rows: Output initial terminal height in rows (must not be NULL)
 *
 * Role:
 *   Saves `STDIN_FILENO` `termios`, enables non-canonical raw input, duplicates the real TTY fd,
 *   redirects `stdout`/`stderr` into a non-blocking capture pipe, installs async-signal-safe
 *   restorers (`SIGINT`, `SIGTERM`, `SIGHUP`, `SIGSEGV`, `SIGABRT`, `SIGWINCH`), and activates
 *   alternate screen (`?1049h`), hidden cursor (`?25l`), SGR 1006 mouse (`?1000h..1006h`), and
 *   bracketed paste (`?2004h`).
 *
 * Rationale:
 *   Duplicating the TTY output descriptor before redirecting `STDOUT_FILENO`/`STDERR_FILENO`
 *   ensures third-party library `printf`/`fprintf(stderr, ...)` calls cannot corrupt TUI cells and
 *   are instead captured cleanly into the bottom activity log.
 *
 * Return: Dedicated TTY output file descriptor (`>= 0`) for `cs_scr_flush()`.
 */
int cs_tui_term_enter(int *out_cols, int *out_rows);

/**
 * cs_tui_term_exit() - Restore terminal modes, stdio descriptors, and normal screen buffer
 *
 * Role:
 *   Idempotently emits exit escape sequences (disabling bracketed paste, SGR mouse, hidden cursor,
 *   and alternate screen), restores original `stdout`/`stderr` and saved `termios`, and closes the
 *   stdio capture pipe.
 *
 * Rationale:
 *   Must be safe to call multiple times on normal exit or prior to spawning an interactive
 *   external `$EDITOR` subprocess so the user's shell is never left in raw/mouse-reporting mode.
 */
void cs_tui_term_exit(void);

/**
 * cs_tui_term_query_size() - Query current terminal viewport dimensions in columns and rows
 * @out_cols: Output terminal width in columns (`>= 40`, fallback `120`)
 * @out_rows: Output terminal height in rows (`>= 12`, fallback `38`)
 *
 * Role:
 *   Reads window geometry via `ioctl(TIOCGWINSZ)` with fallback to `COLUMNS`/`LINES` env vars.
 *
 * Rationale:
 *   Enforces sensible minimum dimensions (`40x12`) so degenerate PTYs or detached sessions cannot
 *   trigger zero-sized allocations or negative layout arithmetic.
 */
void cs_tui_term_query_size(int *out_cols, int *out_rows);

/**
 * cs_tui_drain_captured_stdio() - Drain redirected `stdout`/`stderr` pipe into the TUI log ring
 * @app: Active TUI application state receiving trimmed log lines (may be NULL to discard)
 *
 * Role:
 *   Flushes C stdio streams, performs a non-blocking `read()` on the capture pipe, splits lines,
 *   and appends non-empty messages via `cs_tui_log()`.
 *
 * Rationale:
 *   Prevents the non-blocking pipe buffer from filling while surfacing diagnostic messages from
 *   `milk` / `cacaosim` helpers inside the TUI rather than losing them or tearing the screen.
 *
 * Return: 1 if one or more bytes were drained, 0 if pipe was empty or inactive.
 */
int cs_tui_drain_captured_stdio(cs_tui_app_t *app);

/**
 * cs_tui_poll_input() - Poll and decode one keyboard, SGR 1006 mouse, paste, or resize event
 * @ev:         Destination event structure zeroed and populated on return (must not be NULL)
 * @timeout_ms: Maximum `poll()` wait in milliseconds when input buffer is empty (`0` = instant)
 *
 * Role:
 *   Multiplexes pending `SIGINT`/`SIGTERM` quit flags, `SIGWINCH` resize flags, single-byte keys,
 *   SS3/CSI cursor & function keys, SGR 1006 mouse packets, and bracketed pastes into a single
 *   normalized `cs_input_ev_t`.
 *
 * Rationale:
 *   Uses a persistent lookahead byte buffer with brief `10..16 ms` disambiguation polls after a
 *   lone `ESC` byte so split SSH TCP packets reassemble cleanly without noticeable Escape lag.
 *
 * Return: 1 if an event was decoded into @ev, 0 on timeout with no input.
 */
int cs_tui_poll_input(cs_input_ev_t *ev, int timeout_ms);

/**
 * cs_scr_init() - Allocate front/back 2D cell grids and ANSI output buffer for a virtual screen
 * @s:           Screen state to initialize (must not be NULL)
 * @cols:        Initial width in columns (clamped to `>= 40`)
 * @rows:        Initial height in rows (clamped to `>= 12`)
 * @color_level: Active SGR color level `[0..3]`
 * @ascii_only:  1 for 7-bit ASCII glyphs, 0 for UTF-8 glyphs
 *
 * Role:
 *   Allocates `back[]`, `front[]`, and worst-case `out_buf` (`40 * cols * rows + 4096` bytes) and
 *   marks `force_full = 1` for the initial repaint.
 *
 * Rationale:
 *   Preallocating `out_buf` for worst-case TrueColor SGR per cell guarantees `cs_scr_flush()`
 *   never calls `malloc`/`realloc` inside the interactive render loop.
 *
 * Return: 0 on success, -1 on memory allocation failure.
 */
int cs_scr_init(cs_screen_t *s, int cols, int rows, int color_level, int ascii_only);

/**
 * cs_scr_resize() - Resize screen cell buffers to match new terminal dimensions
 * @s:    Active screen state (must not be NULL)
 * @cols: New width in columns (clamped to `>= 40`)
 * @rows: New height in rows (clamped to `>= 12`)
 *
 * Role:
 *   Reallocates `back[]`, `front[]`, and `out_buf` when dimensions change (or simply sets
 *   `force_full = 1` if unchanged), preserving color/ASCII settings and measured `tx_kbps`.
 *
 * Rationale:
 *   Triggered on `CS_KEY_RESIZE` (`SIGWINCH`) so responsive split panes adapt seamlessly to live
 *   window resizes without stale front-buffer artifacts.
 */
void cs_scr_resize(cs_screen_t *s, int cols, int rows);

/**
 * cs_scr_free() - Release heap buffers owned by a `cs_screen_t` instance
 * @s: Screen instance to clean up (safe if already freed)
 *
 * Role:
 *   Frees `back`, `front`, and `out_buf` and zeroes pointers/capacity.
 *
 * Rationale:
 *   Shared teardown path for `cs_scr_resize()`, `cacaosim_tui_run()`, and headless test snapshots.
 */
void cs_scr_free(cs_screen_t *s);

/**
 * cs_scr_clear() - Fill the entire `back` cell grid with blank spaces and baseline colors
 * @s:  Target screen canvas
 * @fg: Default foreground RGB color
 * @bg: Default background RGB color (active when `s->color_level >= 2`)
 *
 * Role:
 *   Resets `s->back[]` at the start of every rendered frame while leaving `s->front[]` untouched.
 *
 * Rationale:
 *   Clearing only `back[]` allows widgets to draw in pure immediate mode every tick while
 *   `cs_scr_flush()` suppresses all terminal output for cells whose final state matches `front[]`.
 */
void cs_scr_clear(cs_screen_t *s, cs_tui_rgb_t fg, cs_tui_rgb_t bg);

/**
 * cs_scr_flush() - Diff `back` against `front` and atomically flush changed cells to @out_fd
 * @s:      Screen canvas to present
 * @out_fd: Destination TTY file descriptor (`>= 0`)
 *
 * Role:
 *   Serializes minimal cursor positioning (`\033[r;cH`), redundant-suppressed SGR color/attribute
 *   sequences, and UTF-8 glyphs wrapped in DEC synchronized output (`\033[?2026h`..`\033[?2026l`),
 *   commits `back` into `front`, and updates rolling `tx_kbps` telemetry.
 *
 * Rationale:
 *   Combining cell-level delta encoding with `DECSET 2026` eliminates mid-frame cursor flicker and
 *   reduces SSH bandwidth by >90% when only live SHM counters or sparklines change.
 */
void cs_scr_flush(cs_screen_t *s, int out_fd);

/**
 * cs_scr_dump_plain() - Serialize `s->back` glyphs into a newline-delimited plain-text string
 * @s:   Rendered screen canvas to inspect (must not be NULL)
 * @out: Destination character buffer (must not be NULL)
 * @cap: Capacity of @out in bytes (`> 0`)
 *
 * Role:
 *   Extracts the visible UTF-8 / ASCII text grid row-by-row without ANSI escape codes.
 *
 * Rationale:
 *   Powers `cacaosim_tui_render_headless()` so automated tests can assert exact rendered text and
 *   box geometry without parsing VT/SGR control sequences.
 *
 * Return: Number of bytes written (excluding NUL), or -1 on invalid arguments.
 */
int cs_scr_dump_plain(const cs_screen_t *s, char *out, size_t cap);

/**
 * cs_put_cell() - Write a single clipped glyph, color pair, and attribute mask into `s->back`
 * @s:      Target screen canvas
 * @x:      0-based column coordinate (silently clipped if outside `[0..cols-1]`)
 * @y:      0-based row coordinate (silently clipped if outside `[0..rows-1]`)
 * @glyph:  Single-column ASCII or UTF-8 glyph string (`<= 4` bytes; NULL/empty writes `' '`)
 * @fg:     Foreground RGB color
 * @bg:     Background RGB color
 * @attrs:  Bitmask of `CS_ATTR_*` flags
 * @has_bg: Non-zero to paint explicit background color when `s->color_level >= 2`
 *
 * Role:
 *   Lowest-level clipped pixel/character write primitive for all TUI drawing operations.
 *
 * Rationale:
 *   Enforcing bounds clipping and UTF-8 byte-length capping in one primitive lets higher-level box,
 *   scrollable schematic, and shadow painters draw partially off-screen widgets safely.
 */
void cs_put_cell(
    cs_screen_t *s,
    int          x,
    int          y,
    const char  *glyph,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg,
    uint8_t      attrs,
    int          has_bg);

/**
 * cs_fill_rect() - Fill a rectangular region in `s->back` with solid background spaces
 * @s:  Target screen canvas
 * @x:  Leftmost column (clipped to screen bounds)
 * @y:  Topmost row (clipped to screen bounds)
 * @w:  Rectangle width in columns
 * @h:  Rectangle height in rows
 * @fg: Foreground RGB color assigned to cleared cells
 * @bg: Solid background RGB fill color
 *
 * Role:
 *   Clears panel interiors, selection bars, input boxes, and modal backdrops prior to text drawing.
 *
 * Rationale:
 *   Guarantees opaque widget backgrounds when overlaying popups or highlighted rows over existing
 *   canvas content.
 */
void cs_fill_rect(
    cs_screen_t *s,
    int          x,
    int          y,
    int          w,
    int          h,
    cs_tui_rgb_t fg,
    cs_tui_rgb_t bg);

/**
 * cs_put_str() - Render a UTF-8 string horizontally into `s->back` with column-count truncation
 * @s:        Target screen canvas
 * @x:        Starting 0-based column
 * @y:        Target 0-based row
 * @max_cols: Maximum number of character cells to advance (`<= 0` writes nothing)
 * @str:      NUL-terminated UTF-8 text string (safe no-op if NULL)
 * @fg:       Foreground RGB color
 * @bg:       Background RGB color (used when @has_bg != 0)
 * @attrs:    Bitmask of `CS_ATTR_*` flags
 * @has_bg:   1 to overwrite cell background with @bg, 0 to preserve existing cell background
 *
 * Role:
 *   UTF-8-aware single-line text writer supporting transparent-background (`has_bg == 0`) overlays.
 *
 * Rationale:
 *   Counting decoded UTF-8 codepoints rather than raw `strlen()` bytes ensures multi-byte glyphs
 *   (`µ`, arrows, badges) align to exact column widths and truncate cleanly at pane borders, while
 *   `has_bg == 0` preserves underlying alternating-row or selection fills.
 *
 * Return: Number of character columns written `[0..max_cols]`.
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
    int          has_bg);

/**
 * cs_printf_at() - Format and render a `printf`-style UTF-8 string into `s->back`
 * @s:        Target screen canvas
 * @x:        Starting 0-based column
 * @y:        Target 0-based row
 * @max_cols: Maximum number of character columns to emit
 * @fg:       Foreground RGB color
 * @bg:       Background RGB color (when @has_bg != 0)
 * @attrs:    Bitmask of `CS_ATTR_*` flags
 * @has_bg:   1 to apply @bg, 0 to preserve underlying cell background
 * @fmt:      `printf`-compatible format string checked at compile time
 *
 * Role:
 *   Ergonomic formatted text helper used throughout tables, headers, badges, and status bars.
 *
 * Rationale:
 *   Formats into a stack buffer and delegates to `cs_put_str()` so formatted numeric readouts
 *   inherit UTF-8 column counting, control-char sanitization, and transparent-background support.
 *
 * Return: Number of character columns written `[0..max_cols]`.
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
    ...) __attribute__((format(printf, 9, 10)));

/**
 * cs_draw_box() - Draw a filled bordered panel with optional left title and right header badge
 * @s:             Target screen canvas
 * @x:             Leftmost column
 * @y:             Topmost row
 * @w:             Box width in columns (`>= 2`)
 * @h:             Box height in rows (`>= 2`)
 * @double_border: 1 for focused double border (`╔═╗` / `=`), 0 for rounded single border (`╭─╮`)
 * @title:         Optional left-aligned top-border title string (may be NULL)
 * @right_badge:   Optional right-aligned top-border badge string (may be NULL)
 * @border_fg:     Border line RGB color
 * @title_fg:      Title text RGB color
 * @badge_fg:      Right-badge text RGB color
 * @bg:            Interior panel background RGB fill color
 *
 * Role:
 *   Standard container widget for workspace panes, schematic node cards, and modal dialogs.
 *
 * Rationale:
 *   Switching automatically between rounded single borders (unfocused) and bold double borders
 *   (focused) gives instant visual keyboard-focus feedback even on monochrome terminals.
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
    cs_tui_rgb_t bg);

/**
 * cs_draw_shadow() - Render a dimmed drop shadow along the right and bottom edges of a popup box
 * @s:   Target screen canvas
 * @x:   Popup box leftmost column
 * @y:   Popup box topmost row
 * @w:   Popup box width in columns
 * @h:   Popup box height in rows
 * @pal: Active theme palette supplying dim foreground and darkened bar background
 *
 * Role:
 *   Visually separates floating modals and hover tooltips from dense underlying schematic/tables.
 *
 * Rationale:
 *   Uses a 2-column right / 1-row bottom medium-shade (`▒` or `:`) shadow matching classic
 *   `milk-CTRL` / Turbo Vision depth cues so modal overlays remain unmistakable without blurring.
 */
void cs_draw_shadow(cs_screen_t *s, int x, int y, int w, int h, const cs_tui_pal_t *pal);

/**
 * cs_draw_pupil16() - Render a `16x16` telescope pupil amplitude mask inside a `16x8` cell widget
 * @s:     Target screen canvas
 * @x:     Leftmost widget column
 * @y:     Topmost widget row
 * @pup16: Row-major `16x16` normalized pupil amplitude array `[0..1]` (256 floats)
 * @pal:   Active theme palette
 *
 * Role:
 *   Provides real-time visual feedback in the `[bench]` Inspector as the user adjusts `pup_diam`,
 *   central obstruction `cobs`, `nspider`, `spider_w`, and `spider_deg`.
 *
 * Rationale:
 *   Terminal cells have a ~1:2 width-to-height aspect ratio; packing two vertical pupil rows per
 *   cell with Unicode upper/lower/full blocks (`▀`, `▄`, `█`) yields a square-aspect `16x16`
 *   aperture bitmap in just `16x8` terminal cells.
 */
void cs_draw_pupil16(
    cs_screen_t        *s,
    int                 x,
    int                 y,
    const float        *pup16,
    const cs_tui_pal_t *pal);

/**
 * cs_tui_spec_mean_lambda() - Compute photon-flux-weighted mean wavelength of a 32-bin spectrum
 * @spec: 32-bin (`0.40..2.40 um`) photon-rate spectrum (safe if NULL)
 *
 * Role:
 *   Summarizes polychromatic optical beams with a single effective central wavelength `[um]` for
 *   schematic wire labels and beam tables.
 *
 * Rationale:
 *   Flux-weighting `ph_s[b]` across `cacaosim_spec_bin_lambda(b)` reflects both stellar blackbody
 *   tilt (`star_teff`) and upstream dichroic passband cuts accurately without any FFT.
 *
 * Return: Flux-weighted mean wavelength `[um]`, or `0.0f` if total photon flux is zero or NULL.
 */
float cs_tui_spec_mean_lambda(const cacaosim_spec_t *spec);

/**
 * cs_draw_spectrum32() - Draw a wavelength-colored 1-row spectral sparkline bar for @spec
 * @s:     Target screen canvas
 * @x:     Starting column
 * @y:     Target row
 * @width: Sparkline width in columns (`[4..CACAOSIM_NSPEC]`)
 * @spec:  32-bin optical photon-rate spectrum (`0.40..2.40 um`)
 * @theme: Active theme index (controls physical vs night-vision wavelength RGB mapping)
 * @bg:    Cell background RGB color
 *
 * Role:
 *   Visualizes beam spectral energy distribution and dichroic splitting directly inside DAG cards
 *   and `F4:TABLES` beam rows.
 *
 * Rationale:
 *   Max-pooling the 32 spectral bins into @width columns and encoding relative photon flux with
 *   8-level Unicode vertical block elements (`▁`..`█`) tinted by `cs_tui_wavelength_rgb()` lets
 *   users verify LP/SP/BP dichroic cutoffs and starvation bugs at a glance in a single text row.
 */
void cs_draw_spectrum32(
    cs_screen_t           *s,
    int                    x,
    int                    y,
    int                    width,
    const cacaosim_spec_t *spec,
    int                    theme,
    cs_tui_rgb_t           bg);

/* ── Live SHM & Deploy API (cacaosim_tui_shm.c) ───────────────────────────── */

/**
 * cs_shm_scan() - Non-intrusively scan `MILK_SHM_DIR` for `.im.shm` streams and `.fps.shm` units
 * @mon:   Telemetry monitor cache to update (safe no-op if NULL)
 * @cfg:   Optional active `.bench` configuration whose referenced streams are marked `expected`
 * @force: 1 to scan immediately, 0 to throttle scans to at most once every `0.25 s`
 *
 * Role:
 *   Populates `mon->streams[]`, `mon->fps[]`, `cnt0`-delta frame rates `hz`, PID liveness, and
 *   live science Strehl ratios for the `F5:STATUS` view, header badges, and schematic indicators.
 *
 * Rationale:
 *   Uses read-only `mmap` of metadata headers (`IMAGE_METADATA` / `FUNCTION_PARAMETER_STRUCT_MD`)
 *   and rate-limits directory walks to `4 Hz` so monitoring a kHz real-time AO loop adds zero lock
 *   contention or CPU overhead.
 */
void cs_shm_scan(cs_shm_mon_t *mon, const cacaosim_bench_cfg_t *cfg, int force);

/**
 * cs_shm_find_stream() - Look up a tracked stream index in @mon by stem name
 * @mon:  Populated SHM monitor table (may be NULL)
 * @name: Stream stem name without `.im.shm` suffix (may be NULL/empty)
 *
 * Role:
 *   Resolves per-node SHM stream status (`exists`, `alive`, `hz`, dimensions) when rendering node
 *   badges and wiring tables.
 *
 * Rationale:
 *   Linear scan over `<= 48` cached entries fits in a few cache lines without hash-table overhead.
 *
 * Return: Index `[0..mon->n_streams-1]` on match, or `-1` if not found.
 */
int cs_shm_find_stream(const cs_shm_mon_t *mon, const char *name);

/**
 * cs_tui_exec_deploy() - Validate, stage, and launch the active `.bench` via `cacaosim-bench start`
 * @app:  Active TUI application state (must not be NULL)
 * @mode: Deployment backend string (`"fused"` for `cacaosim.htsim` or `"modular"` for FPS units)
 *
 * Role:
 *   Verifies DAG validity, stages unsaved configs to `<MILK_SHM_DIR>/csim_ctrl_staged.bench`,
 *   invokes `cacaosim-bench start` (optionally with `--atmturb`), logs output, and refreshes SHM.
 *
 * Rationale:
 *   Staging dirty configurations into `MILK_SHM_DIR` lets users test parameter tweaks live on
 *   hardware streams immediately without overwriting their saved `.bench` file until `[Save]`.
 *
 * Return: 0 on successful deployment, -1 on DAG error or child command failure.
 */
int cs_tui_exec_deploy(cs_tui_app_t *app, const char *mode);

/**
 * cs_tui_exec_stop() - Terminate active `cacaosim` and `atmturb` FPS simulation processes
 * @app: Active TUI application state (must not be NULL)
 *
 * Role:
 *   Invokes `cacaosim-bench stop --no-view`, streams output into the activity log, and forces an
 *   immediate `cs_shm_scan()` update.
 *
 * Rationale:
 *   Delegates process teardown to `cacaosim-bench stop` so TUI `[Stop]` and CLI `cacaosim-bench
 *   stop` share identical graceful shutdown semantics for both fused and modular deployments.
 *
 * Return: 0 on clean stop, -1 if `cacaosim-bench stop` exited non-zero.
 */
int cs_tui_exec_stop(cs_tui_app_t *app);

/**
 * cs_tui_exec_mkpupil() - Synthesize entrance pupil SHM and any `[wfe.<id>]` static OPD streams
 * @app: Active TUI application state (must not be NULL)
 *
 * Role:
 *   Runs `milk-fpsexec-cacaosim-mkpupil` using `[bench]` pupil geometry, then initializes every
 *   configured `CACAOSIM_NODE_WFE` stream via `milk-fpsexec-cacaosim-wfe .init_only=ON`, followed
 *   by a forced SHM scan.
 *
 * Rationale:
 *   Provides one-keystroke static calibration stream generation (`in_pup` + `[meter]` `in_wfe`
 *   maps) directly from the TUI before or between simulation runs.
 *
 * Return: 0 on success, -1 if `mkpupil` execution failed.
 */
int cs_tui_exec_mkpupil(cs_tui_app_t *app);

/* ── Parameter Inspector API (cacaosim_tui_params.c) ──────────────────────── */

/**
 * cs_params_build() - Populate context-sensitive Inspector parameter rows for `(sel_item, subtab)`
 * @cfg:      Current `.bench` configuration (must not be NULL)
 * @sel_item: Selected element (`0` = `[bench]`, `1..n_nodes` = `cfg->nodes[sel_item - 1]`)
 * @subtab:   Inspector category tab (`0` = OPTICS, `1` = DYN/DET, `2` = SHM/WIRING)
 * @out:      Destination descriptor array (must not be NULL)
 * @max_out:  Capacity of @out (`<= CS_MAX_PARAMS`)
 *
 * Role:
 *   Reflects the active element's editable fields into typed, unit-annotated, range-bounded UI
 *   descriptors, hiding inactive mode-specific rows automatically.
 *
 * Rationale:
 *   Rebuilding the descriptor list dynamically from @cfg ensures mode switches (e.g., Zonal DM vs
 *   2-Axis TTM, Gray vs Dichroic splitter, Vortex vs Custom MFT coronagraph, Auto vs Manual flux)
 *   immediately surface only physically relevant parameters.
 *
 * Return: Number of descriptors written to @out `[0..max_out]`.
 */
int cs_params_build(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    int                         subtab,
    int                         show_advanced,
    cs_param_desc_t            *out,
    int                         max_out);

/**
 * cs_param_format_val() - Format the current value of parameter @pd into a display/edit string
 * @cfg:      Source `.bench` configuration
 * @sel_item: Selected element index (`0` = `[bench]`, `1..n_nodes` = node)
 * @pd:       Parameter descriptor from `cs_params_build()`
 * @buf:      Destination string buffer
 * @cap:      Capacity of @buf in bytes
 *
 * Role:
 *   Produces human-readable text representations (fixed-precision floats, `"ON"`/`"OFF"` booleans,
 *   symbolic enum names, microsecond conversions) for Inspector value columns and inline editing.
 *
 * Rationale:
 *   Centralizing value formatting guarantees that inline edit buffers, incremental `cs_param_step`
 *   reads, and rendered Inspector rows always agree on units and precision.
 */
void cs_param_format_val(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    const cs_param_desc_t      *pd,
    char                       *buf,
    size_t                      cap);

/**
 * cs_param_step() - Increment, decrement, cycle, or toggle parameter @pd by one UI step
 * @cfg:      Mutable `.bench` configuration to update
 * @sel_item: Selected element index (`0` = `[bench]`, `1..n_nodes` = node)
 * @pd:       Parameter descriptor supplying type, bounds, and `step_v`
 * @dir:      Step direction (`< 0` decrement/previous, `> 0` increment/next)
 * @coarse:   Non-zero to multiply numeric `step_v` by `5.0x` (e.g., Shift/Page keys)
 * @err_buf:  Optional error message buffer on validation failure
 * @err_cap:  Capacity of @err_buf in bytes
 *
 * Role:
 *   Handles `Left`/`Right` arrow keys, `+`/`-`, and mouse-wheel adjustments on Inspector rows.
 *
 * Rationale:
 *   Routes stepped values back through `cs_param_set_str()` so clamping, parity rounding, and
 *   coupled field synchronization are enforced identically for keyboard stepping and text entry.
 *
 * Return: 0 if @cfg was updated, -1 if @pd is non-steppable (e.g., string/ID) or invalid.
 */
int cs_param_step(
    cacaosim_bench_cfg_t  *cfg,
    int                    sel_item,
    const cs_param_desc_t *pd,
    int                    dir,
    int                    coarse,
    char                  *err_buf,
    size_t                 err_cap);

/**
 * cs_param_set_str() - Parse, validate, clamp, and apply a string value to parameter @pd in @cfg
 * @cfg:      Mutable `.bench` configuration to update
 * @sel_item: Selected element index (`0` = `[bench]`, `1..n_nodes` = node)
 * @pd:       Target parameter descriptor
 * @val_str:  User-supplied text value (numeric literal, enum label/index, bool, ID, or stream/beam)
 * @err_buf:  Optional human-readable validation error buffer
 * @err_cap:  Capacity of @err_buf in bytes
 *
 * Role:
 *   Single authoritative mutation path for all Inspector edits, enforcing range bounds, even-grid
 *   parity (`nwf`, `vortex_charge`), unique node renames, and cascading output-beam rewires.
 *
 * Rationale:
 *   Funneling every Inspector edit through one validated setter prevents out-of-range physics
 *   parameters, duplicate section IDs, or broken downstream beam links when renaming outputs.
 *
 * Return: 0 on success, -1 on parse or validation error (populating @err_buf).
 */
int cs_param_set_str(
    cacaosim_bench_cfg_t  *cfg,
    int                    sel_item,
    const cs_param_desc_t *pd,
    const char            *val_str,
    char                  *err_buf,
    size_t                 err_cap);

/* ── Frame Rendering & Helpers (cacaosim_tui_render.c & cacaosim_tui.c) ───── */

/**
 * cs_tui_add_hit() - Register an interactive rectangular mouse hit region for the current frame
 * @app:  Active TUI state (`app->hits[]` appended if below `CS_MAX_HITS`)
 * @kind: Semantic hit target kind (`cs_hit_kind_t`)
 * @arg:  Kind-specific integer index payload
 * @x:    Leftmost 0-based column
 * @y:    Topmost 0-based row
 * @w:    Width in columns
 * @h:    Height in rows
 * @tip:  Optional hover tooltip description (may be NULL)
 *
 * Role:
 *   Records clickable, scrollable, and draggable widget bounds during immediate-mode rendering.
 *
 * Rationale:
 *   Later registrations (e.g., floating modals) naturally sit at higher indices so reverse-order
 *   hit scans automatically respect z-order occlusion.
 */
void cs_tui_add_hit(
    cs_tui_app_t *app,
    cs_hit_kind_t kind,
    int           arg,
    int           x,
    int           y,
    int           w,
    int           h,
    const char   *tip);

/**
 * cs_tui_log() - Append a formatted message to the activity log ring and status-bar toast banner
 * @app:   Active TUI state
 * @level: Severity level (`0` = info, `1` = ok/success, `2` = warning, `3` = error)
 * @fmt:   `printf`-style format string
 *
 * Role:
 *   Unified user-feedback channel for edits, file saves, child deployment output, and errors.
 *
 * Rationale:
 *   Updates both the persistent `F5:STATUS` scrollable log ring and a timed bottom-bar toast so
 *   feedback is visible from every workspace tab without disruptive modal popups.
 */
void cs_tui_log(cs_tui_app_t *app, int level, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

/**
 * @brief Derived physical pupil-plane and focal-plane spatial sampling scales for the Inspector.
 *
 * Computed on the fly by `cs_tui_compute_scales()` so users editing `nwf`, `pup_diam`,
 * `tel_diam_m`, `pad_factor`, `pixscale_ld`, or `n_sub` immediately see physical `[m/px]`,
 * `[lambda/D / px]`, and `[mas/px]` or `[arcsec/px]` resolutions.
 */
typedef struct
{
    float d_beam_px;      /**< Illuminated pupil diameter across grid `[px]` (`nwf * pup_diam`) */
    float pup_m_pix;      /**< Physical primary-mirror sampling in pupil plane `[meters / pixel]` */
    float foc_ld_pix;     /**< Angular focal-plane pixel scale `[lambda_0 / D per pixel]` */
    float foc_arcsec_pix; /**< Sky angular pixel scale `[arcsec / pixel]` at `ref_lam_um` */
    float ref_lam_um;     /**< Reference wavelength `[um]` used to evaluate diffraction scale */
    char  foc_src[24];    /**< Short label naming which focal plane `foc_*` describes */
} cs_optical_scales_t;

/**
 * cs_tui_compute_scales() - Derive pupil `[m/px]` and focal `[l/D/px, arcsec/px]` sampling scales
 * @cfg:      Source `.bench` configuration
 * @sel_item: Selected element (`0` = `[bench]`, `1..n_nodes` = node)
 * @out:      Destination optical scale summary (must not be NULL)
 *
 * Role:
 *   Computes live physical plate scales displayed in the Inspector summary header.
 *
 * Rationale:
 *   Prevents Nyquist undersampling mistakes when configuring FFT pad factors, custom FPM pixel
 *   scales, SHWFS subapertures, or science camera FOVs.
 */
void cs_tui_compute_scales(
    const cacaosim_bench_cfg_t *cfg,
    int                         sel_item,
    cs_optical_scales_t        *out);

/**
 * cs_tui_refresh_analysis() - Re-run zero-FFT DAG spectral analysis, pupil preview, and param list
 * @app: Active TUI state (`app->ana_dirty` cleared on completion)
 *
 * Role:
 *   Synchronizes `app->ctx` (`cacaosim_bench_analyze()`), `app->pup16[]` (`16x16` entrance pupil),
 *   and `app->params[]` (`cs_params_build()`) whenever `app->cfg`, `sel_item`, or `subtab` changes.
 *
 * Rationale:
 *   Because `cacaosim_bench_analyze()` executes in `< 0.1 ms` with zero heap/FFT allocations, the
 *   TUI can recompute full multi-arm photometry, effective cadences, and DAG errors interactively
 *   on every keystroke.
 */
void cs_tui_refresh_analysis(cs_tui_app_t *app);

/**
 * cs_tui_render_frame() - Render the complete active workspace tab, bars, modals, and tooltips
 * @app: Active TUI state (`app->scr.back` populated and `app->hits` rebuilt)
 *
 * Role:
 *   Top-level immediate-mode frame compositor shared by `cacaosim_tui_run()` and
 *   `cacaosim_tui_render_headless()`.
 *
 * Rationale:
 *   Keeping all layout and widget composition side-effect-free on `app->scr.back` decouples visual
 *   composition from terminal I/O flushing.
 */
void cs_tui_render_frame(cs_tui_app_t *app);

/**
 * cs_render_dag_canvas() - Render the interactive 2D optical bench DAG schematic and beam spectra
 * @app:          Active TUI state
 * @x:            Canvas leftmost column
 * @y:            Canvas topmost row
 * @w:            Canvas width in columns
 * @h:            Canvas height in rows
 * @wide_spectra: Non-zero to draw wider 32-bin spectral sparklines (used in full-screen `F3`)
 *
 * Role:
 *   Draws `[bench]` telescope entrance, inline optical elements, branching beamsplitters/coros,
 *   routing wires, wavelength sparklines, and clickable `CS_HIT_SCHEM_NODE` boxes.
 *
 * Rationale:
 *   Shared between the compact top-right pane of `F2:BUILDER` and full-screen `F3:SCHEMATIC` so
 *   both views present identical topological layout and selection behavior.
 */
void cs_render_dag_canvas(
    cs_tui_app_t *app,
    int           x,
    int           y,
    int           w,
    int           h,
    int           wide_spectra);

/**
 * cs_render_view_schematic() - Render the `F3:SCHEMATIC` full-screen optical diagram workspace
 * @app: Active TUI state
 * @x:   Workspace interior left column
 * @y:   Workspace interior top row
 * @w:   Workspace interior width in columns
 * @h:   Workspace interior height in rows
 *
 * Role:
 *   Dedicates the entire viewport to `cs_render_dag_canvas()` with wide 32-bin beam spectra and
 *   optical summary badges.
 *
 * Rationale:
 *   Gives complex multi-stage ExAO / dual-WFS / coronagraphic benches ample 2D space without side
 *   panes.
 */
void cs_render_view_schematic(cs_tui_app_t *app, int x, int y, int w, int h);

/**
 * cs_render_view_tables() - Render the `F4:TABLES` hardware mirrors, sensors, and beams audit view
 * @app: Active TUI state
 * @x:   Workspace interior left column
 * @y:   Workspace interior top row
 * @w:   Workspace interior width in columns
 * @h:   Workspace interior height in rows
 *
 * Role:
 *   Displays structured summary tables of active DMs/TTMs, WFS/science cameras (cadences, photon
 *   budgets, readout noise), and named optical beams with wavelength sparklines.
 *
 * Rationale:
 *   Allows rapid cross-node comparison of photon flux, frame rates, and SHM wiring across the
 *   entire bench without clicking through nodes one by one.
 */
void cs_render_view_tables(cs_tui_app_t *app, int x, int y, int w, int h);

/**
 * cs_render_view_status() - Render the `F5:STATUS` live `MILK_SHM_DIR` telemetry & deploy dashboard
 * @app: Active TUI state
 * @x:   Workspace interior left column
 * @y:   Workspace interior top row
 * @w:   Workspace interior width in columns
 * @h:   Workspace interior height in rows
 *
 * Role:
 *   Presents deployment action buttons (`Deploy Fused`, `Deploy Modular`, `Stop`, `mkpupil`,
 *   `atmturb`), live `.im.shm` stream dimensions/rates, `.fps.shm` process health, Strehl
 *   telemetry, and the rolling activity log.
 *
 * Rationale:
 *   Brings `milk-CTRL`-style operational supervision into the same window as optical bench editing.
 */
void cs_render_view_status(cs_tui_app_t *app, int x, int y, int w, int h);

/**
 * cs_render_view_source() - Render the `F6:SOURCE` live serialized `.bench` INI/TOML preview
 * @app: Active TUI state
 * @x:   Workspace interior left column
 * @y:   Workspace interior top row
 * @w:   Workspace interior width in columns
 * @h:   Workspace interior height in rows
 *
 * Role:
 *   Formats `app->cfg` via `cacaosim_bench_format_str()` with line numbers and syntax highlighting.
 *
 * Rationale:
 *   Lets users inspect the exact human-readable `.bench` text produced by their GUI edits before
 *   saving or committing to version control.
 */
void cs_render_view_source(cs_tui_app_t *app, int x, int y, int w, int h);

#endif /* CACAOSIM_TUI_INTERNAL_H */
