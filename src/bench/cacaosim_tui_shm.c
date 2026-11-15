// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_shm.c
 * @brief   Non-intrusive MILK_SHM_DIR stream/FPS scanner and bench deployment actions
 *
 * Architectural Role in cacaosim-CTRL:
 *   Implements the live runtime telemetry inspector and hardware lifecycle bridge between the
 *   declarative `.bench` configurator and running `milk` / `cacao` real-time processes:
 *   1. Non-blocking `ImageStreamIO` Shared Memory (`.im.shm`) & Function Parameter Structure
 *      (`.fps.shm`) directory scanner (`cs_shm_scan()`), reconciling streams expected by the
 *      current `.bench` topology with extra AO/RTC streams discovered in `MILK_SHM_DIR`.
 *   2. Monotonic `cnt0`-delta frame-rate estimator (`probe_one_im_shm()`) and process-table PID
 *      liveness checker (`pid_alive()`), plus live science Strehl ratio (`strehl_inst`,
 *      `strehl_ave`) extractor (`probe_fps_file()`) feeding header badges and `F5:STATUS`.
 *   3. One-keystroke bench deployment (`cs_tui_exec_deploy()`), graceful shutdown
 *      (`cs_tui_exec_stop()`), and static calibration stream synthesis (`cs_tui_exec_mkpupil()`
 *      for anti-aliased pupils and `[wfe.<id>]` OPD maps).
 *
 * Design Rationale & Motivation:
 *   Observatory real-time control loops (`cacao`, `cacaosim.htsim`, WFS cameras, DM servos) update
 *   shared-memory streams at `1..10 kHz`. Attaching via full `ImageStreamIO` client routines or
 *   spinning on `md->write` from a `10..30 Hz` monitoring GUI risks semaphore side effects, page
 *   table churn on multi-megabyte 3D cubes, or stalls if a writer crashed mid-frame. Instead,
 *   `cacaosim_tui_shm.c`:
 *   - Opens `.im.shm` files `O_RDONLY` and `mmap`s strictly `sizeof(IMAGE_METADATA)` bytes
 *     (`PROT_READ`), reading `naxis`, `size[0..2]`, `datatype`, `cnt0`, and `ownerPID`/`creatorPID`
 *     with zero `md->write` spin loops, zero semaphore mutations, and immediate `munmap()`.
 *   - Bounds-checks `.fps.shm` `NBparamMAX` against actual `fstat()` byte size before scanning
 *     active `FPS_PARAM` slots, preventing out-of-bounds reads on partially initialized FPS files.
 *   - Stages dirty in-memory `.bench` edits to `<MILK_SHM_DIR>/csim_ctrl_staged.bench` on deploy
 *     so operators can audition optical or servo tweaks live without overwriting their saved file.
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "bench/cacaosim_tui_internal.h"
#include "common/cacaosim_mask.h"

/**
 * shm_dir_path() - Resolve the active `milk` shared-memory directory path
 * @out: Destination path buffer (must not be NULL)
 * @cap: Capacity of @out in bytes
 *
 * Role:
 *   Reads `MILK_SHM_DIR` from the process environment, falling back to the standard `/milk/shm`
 *   root when unset or empty.
 *
 * Rationale:
 *   Matches `ImageStreamIO`'s exact directory resolution rule so `cacaosim-CTRL` seamlessly tracks
 *   isolated per-user / per-test `MILK_SHM_DIR` sandboxes as well as system-wide `/milk/shm`.
 */
static void shm_dir_path(char *out, size_t cap)
{
    const char *sd = getenv("MILK_SHM_DIR");
    snprintf(out, cap, "%s", (sd != NULL && sd[0] != '\0') ? sd : "/milk/shm");
}

/**
 * pid_alive() - Check whether OS process @p currently exists without sending a signal
 * @p: Process ID read from `IMAGE_METADATA` or `FUNCTION_PARAMETER_STRUCT_MD` (`<= 0` -> dead)
 *
 * Role:
 *   Distinguishes actively owned SHM streams and running FPS processes from stale leftover files in
 *   `/milk/shm` after a process exit or kill.
 *
 * Rationale:
 *   POSIX `kill(p, 0)` performs kernel PID existence validation with zero signal delivery; treating
 *   `errno == EPERM` as alive handles multi-user observatory setups where RTC processes run under a
 *   dedicated real-time UID distinct from the TUI operator.
 *
 * Return: 1 if process @p exists in the kernel process table, 0 if `p <= 0` or `ESRCH`.
 */
static int pid_alive(pid_t p)
{
    if (p <= 0)
    {
        return 0;
    }
    return (kill(p, 0) == 0 || errno == EPERM);
}

/**
 * cs_shm_find_stream() - Locate a monitored stream entry in `mon->streams[]` by stem name
 * @mon:  Populated SHM monitor snapshot (safe if NULL)
 * @name: Stream stem name without `.im.shm` extension (safe if NULL or empty)
 *
 * Role:
 *   Provides fast lookup for deduplicating stream registrations during `cs_shm_scan()` and for
 *   coloring live/missing SHM badges in `F2:BUILDER`, `F3:SCHEMATIC`, and `F4:TABLES`.
 *
 * Rationale:
 *   Bounded linear search across at most `CS_MAX_MON_STREAMS` (`48`) contiguous structs requires
 *   zero dynamic memory or auxiliary index maintenance.
 *
 * Return: Matching index `[0..mon->n_streams-1]`, or `-1` if absent.
 */
int cs_shm_find_stream(const cs_shm_mon_t *mon, const char *name)
{
    if (mon == NULL || name == NULL || name[0] == '\0')
    {
        return -1;
    }
    for (int i = 0; i < mon->n_streams; i++)
    {
        if (strcmp(mon->streams[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * register_stream() - Insert or promote a stream stem in `mon->streams[]` prior to SHM probing
 * @mon:      Target SHM monitor state
 * @name:     Stream stem name (ignored if NULL/empty)
 * @role:     Human-readable functional tag (e.g., `"pupil"`, `"dm_cmd"`, `"sci_psf"`)
 * @owner:    Owning `.bench` section identifier (e.g., `"[bench]"`, `"dm0"`, or `"shm"`)
 * @expected: 1 if explicitly wired in the active `.bench` config, 0 if discovered on disk
 *
 * Role:
 *   Builds the deduplicated candidate list of streams to inspect on each `cs_shm_scan()` cycle.
 *
 * Rationale:
 *   Registering `.bench`-wired streams first (`expected = 1`) ensures missing required inputs
 *   (such as an ungenerated `in_pup` pupil) appear prominently in `F5:STATUS` even before their
 *   `.im.shm` file exists on disk, while deduplication prevents double-listing when the file is
 *   subsequently seen by `readdir()`.
 */
static void register_stream(
    cs_shm_mon_t *mon,
    const char   *name,
    const char   *role,
    const char   *owner,
    int           expected)
{
    if (name == NULL || name[0] == '\0' || mon->n_streams >= CS_MAX_MON_STREAMS)
    {
        return;
    }
    int idx = cs_shm_find_stream(mon, name);
    if (idx >= 0)
    {
        if (expected)
        {
            mon->streams[idx].expected = 1;
        }
        return;
    }
    cs_mon_stream_t *s = &mon->streams[mon->n_streams++];
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", name);
    snprintf(s->role, sizeof(s->role), "%s", (role != NULL) ? role : "stream");
    snprintf(s->owner, sizeof(s->owner), "%s", (owner != NULL) ? owner : "-");
    s->expected = expected;
}

/**
 * probe_one_im_shm() - Non-blockingly inspect one `.im.shm` header and estimate its frame rate
 * @shm_dir:  Directory path containing `.im.shm` files
 * @s:        Stream descriptor updated in-place with dimensions, `cnt0`, liveness, and `hz`
 * @prev_arr: Snapshot of `mon->streams[]` from the preceding scan for `cnt0` delta calculation
 * @n_prev:   Number of valid entries in @prev_arr
 * @now_s:    Current monotonic timestamp `[s]`
 *
 * Role:
 *   Reads `IMAGE_METADATA` fields (`naxis`, `size[0..2]`, `datatype`, `cnt0`, `ownerPID`,
 *   `creatorPID`) from `<shm_dir>/<s->name>.im.shm` and computes an exponentially smoothed update
 *   frequency `s->hz`.
 *
 * Rationale:
 *   1. Maps ONLY `sizeof(IMAGE_METADATA)` bytes with `O_RDONLY` / `PROT_READ` and never spins on
 *      `md->write` or touches `ImageStreamIO` semaphores: scalar header reads are benign even
 *      during concurrent kHz real-time writes and cannot block the TUI if a writer froze with
 *      `md->write == 1`.
 *   2. Blends `0.45 * prev_hz + 0.55 * inst_hz` over windows `dt >= 0.08 s` to suppress polling
 *      jitter while snapping to `0.0 Hz` after `0.6 s` of zero `cnt0` increments so stopped loops
 *      report `0 Hz` promptly.
 */
static void probe_one_im_shm(
    const char            *shm_dir,
    cs_mon_stream_t       *s,
    const cs_mon_stream_t *prev_arr,
    int                    n_prev,
    double                 now_s)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%.250s/%.60s.im.shm", shm_dir, s->name);

    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        s->exists = 0;
        return;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t) st.st_size < sizeof(IMAGE_METADATA))
    {
        close(fd);
        s->exists = 0;
        return;
    }

    /* Map strictly the IMAGE_METADATA header read-only; skip pixel data & md->write spin-locks */
    void *addr = mmap(NULL, sizeof(IMAGE_METADATA), PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED)
    {
        s->exists = 0;
        return;
    }

    const IMAGE_METADATA *md = (const IMAGE_METADATA *) addr;
    s->exists                = 1;
    s->naxis                 = md->naxis;
    s->size[0]               = md->size[0];
    s->size[1]               = md->size[1];
    s->size[2]               = md->size[2];
    s->datatype              = md->datatype;
    s->cnt0                  = md->cnt0;
    s->writer_pid            = (md->ownerPID > 0) ? md->ownerPID : md->creatorPID;
    s->alive                 = pid_alive(md->ownerPID) || pid_alive(md->creatorPID);
    munmap(addr, sizeof(IMAGE_METADATA));

    /* Compute exponentially smoothed frame rate from previous scan's (prev_cnt0, prev_t_sec) */
    for (int i = 0; i < n_prev; i++)
    {
        if (strcmp(prev_arr[i].name, s->name) == 0 && prev_arr[i].exists && prev_arr[i].has_prev)
        {
            double dt = now_s - prev_arr[i].prev_t_sec;
            if (dt >= 0.08)
            {
                /* Guard against cnt0 reset if stream was recreated between scans */
                uint64_t dc = (s->cnt0 >= prev_arr[i].prev_cnt0)
                                  ? (s->cnt0 - prev_arr[i].prev_cnt0)
                                  : 0u;
                double   inst_hz = (double) dc / dt;
                s->hz            = (prev_arr[i].hz > 0.0)
                                       ? (0.45 * prev_arr[i].hz + 0.55 * inst_hz)
                                       : inst_hz;
                if (dc == 0u && dt > 0.6)
                {
                    s->hz = 0.0;
                }
                s->prev_cnt0  = s->cnt0;
                s->prev_t_sec = now_s;
                s->has_prev   = 1;
                if (dc > 0u)
                {
                    s->alive = 1;
                }
                return;
            }
            s->hz         = prev_arr[i].hz;
            s->prev_cnt0  = prev_arr[i].prev_cnt0;
            s->prev_t_sec = prev_arr[i].prev_t_sec;
            s->has_prev   = 1;
            return;
        }
    }

    s->hz         = 0.0;
    s->prev_cnt0  = s->cnt0;
    s->prev_t_sec = now_s;
    s->has_prev   = 1;
}

/**
 * probe_fps_file() - Read-only inspect a `.fps.shm` file for `conf`/`run` PIDs and Strehl ratios
 * @mon:      Monitor context receiving the new `cs_mon_fps_t` record and aggregate Strehl telemetry
 * @shm_dir:  Directory path containing `.fps.shm` files
 * @fps_name: Stem name of the FPS file without `.fps.shm` suffix
 *
 * Role:
 *   Populates `mon->fps[]` with FPS description, `confpid`/`runpid` process health, and optional
 *   `strehl_inst` / `strehl_ave` float32 telemetry exported by `fpcam` or `cacaosim.htsim`.
 *
 * Rationale:
 *   Reads the raw `.fps.shm` memory map directly (`O_RDONLY`, `PROT_READ`) instead of calling
 *   `function_parameter_struct_connect()` so monitoring never registers client locks or alters
 *   FPS state, while validating `NBparamMAX` against `st.st_size` protects against truncated files.
 */
static void probe_fps_file(cs_shm_mon_t *mon, const char *shm_dir, const char *fps_name)
{
    if (mon->n_fps >= CS_MAX_MON_FPS)
    {
        return;
    }
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%.250s/%.60s.fps.shm", shm_dir, fps_name);

    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        return;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t) st.st_size < sizeof(FUNCTION_PARAMETER_STRUCT_MD))
    {
        close(fd);
        return;
    }
    size_t map_len = (size_t) st.st_size;
    void  *addr    = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED)
    {
        return;
    }

    /* Sanity-check parameter array count against actual mapped file size before dereferencing */
    const FUNCTION_PARAMETER_STRUCT_MD *md = (const FUNCTION_PARAMETER_STRUCT_MD *) addr;
    long max_fit = (long) ((map_len - sizeof(*md)) / sizeof(FPS_PARAM));
    if (md->NBparamMAX < 0 || md->NBparamMAX > max_fit + 64)
    {
        munmap(addr, map_len);
        return;
    }

    cs_mon_fps_t *fe = &mon->fps[mon->n_fps++];
    memset(fe, 0, sizeof(*fe));
    snprintf(fe->name, sizeof(fe->name), "%.63s", fps_name);
    snprintf(fe->descr, sizeof(fe->descr), "%.79s", md->description);
    fe->conf_pid   = md->confpid;
    fe->run_pid    = md->runpid;
    fe->conf_alive = pid_alive(md->confpid);
    fe->run_alive  = pid_alive(md->runpid);
    if (fe->run_alive)
    {
        mon->n_running_fps++;
    }

    /* Scan active FPS_PARAM entries immediately following header for live Strehl telemetry */
    long             np   = (md->NBparamMAX < max_fit) ? md->NBparamMAX : max_fit;
    const FPS_PARAM *parr = (const FPS_PARAM *) (md + 1);
    if (np > 512)
    {
        np = 512;
    }
    for (long p = 0; p < np; p++)
    {
        if (!(parr[p].fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }
        const char *k1 = parr[p].keyword[1];
        if (parr[p].type == FPTYPE_FLOAT32)
        {
            if (strcmp(k1, "strehl_inst") == 0)
            {
                fe->strehl_inst = parr[p].val.f32[0];
                fe->has_strehl  = 1;
            }
            else if (strcmp(k1, "strehl_ave") == 0)
            {
                fe->strehl_ave = parr[p].val.f32[0];
                fe->has_strehl = 1;
            }
        }
    }

    /* Prefer Strehl telemetry from an actively running FPS over stale stopped FPS records */
    if (fe->has_strehl && (fe->run_alive || !mon->has_strehl))
    {
        mon->strehl_inst = fe->strehl_inst;
        mon->strehl_ave  = fe->strehl_ave;
        mon->has_strehl  = 1;
    }

    munmap(addr, map_len);
}

/**
 * register_mask_shm() - Conditionally register a coronagraph mask specifier if backed by SHM
 * @mon:   Target SHM monitor state
 * @spec:  Coronagraph mask string (either a `.fits` file path or an SHM stream stem)
 * @role:  Short mask role label (e.g., `"pup_amp"`, `"fpm_pha"`, `"lyot_amp"`)
 * @owner: Owning coronagraph node identifier
 *
 * Role:
 *   Ensures SHM-backed coronagraph masks appear in the monitored stream table while ignoring disk
 *   `.fits` / `.fit` / `.fts` file paths.
 *
 * Rationale:
 *   Coronagraph mask fields (`in_pup_amp`, `in_fpm_pha`, etc.) accept either FITS files on disk or
 *   live SHM stream names via `cacaosim_mask`; filtering through `!cacaosim_mask_is_fits_path()`
 *   prevents false "missing SHM stream" warnings for valid FITS file paths.
 */
static void register_mask_shm(
    cs_shm_mon_t *mon,
    const char   *spec,
    const char   *role,
    const char   *owner)
{
    if (spec != NULL && spec[0] != '\0' && !cacaosim_mask_is_fits_path(spec))
    {
        register_stream(mon, spec, role, owner, 1);
    }
}

/**
 * is_relevant_extra_shm() - Filter unreferenced `.im.shm` stems for AO/simulation relevance
 * @stem: Filename stem stripped of `.im.shm`
 *
 * Role:
 *   Selects auxiliary `cacaosim`, `cacao` (`aol*`, `dm*`, `ttm*`, `lowfs*`), beam (`b_*`), mask,
 *   and `atmturb` (`outarraypha`) streams for display in `F5:STATUS`.
 *
 * Rationale:
 *   Shared `/milk/shm` directories on full RTC machines can hold hundreds of housekeeping streams;
 *   prefix filtering surfaces AO-relevant streams without overflowing `CS_MAX_MON_STREAMS`.
 *
 * Return: Non-zero if @stem matches AO / `cacaosim` naming conventions, else 0.
 */
static int is_relevant_extra_shm(const char *stem)
{
    return (strncmp(stem, "csim_", 5) == 0 || strncmp(stem, "aol", 3) == 0 ||
            strncmp(stem, "dm", 2) == 0 || strncmp(stem, "ttm", 3) == 0 ||
            strncmp(stem, "lowfs", 5) == 0 || strncmp(stem, "b_", 2) == 0 ||
            strncmp(stem, "wfe", 3) == 0 || strstr(stem, "_wfe") != NULL ||
            strncmp(stem, "apod", 4) == 0 || strncmp(stem, "fpm", 3) == 0 ||
            strncmp(stem, "lyot", 4) == 0 || strcmp(stem, "outarraypha") == 0);
}

/**
 * is_relevant_fps() - Filter `.fps.shm` stems belonging to `cacaosim`, `atmturb`, or `cacao` loops
 * @stem: Filename stem stripped of `.fps.shm`
 *
 * Role:
 *   Selects relevant FPS modules during `readdir()` directory enumeration in `cs_shm_scan()`.
 *
 * Rationale:
 *   Keeps `mon->fps[]` focused on optical simulation, turbulence generation, and AO loop control
 *   units so `F5:STATUS` remains concise and bounded.
 *
 * Return: Non-zero if @stem should be probed via `probe_fps_file()`, else 0.
 */
static int is_relevant_fps(const char *stem)
{
    return (strncmp(stem, "csim_", 5) == 0 || strncmp(stem, "cacaosim", 8) == 0 ||
            strncmp(stem, "atmturb", 7) == 0 || strncmp(stem, "mkwfs", 5) == 0 ||
            strncmp(stem, "aol", 3) == 0);
}

/**
 * cs_shm_scan() - Refresh live `MILK_SHM_DIR` image-stream and FPS process telemetry in @mon
 * @mon:   Destination telemetry state (safe no-op if NULL)
 * @cfg:   Optional active `.bench` configuration supplying expected stream wiring
 * @force: Non-zero to bypass the `0.25 s` (`4 Hz`) minimum scan interval
 *
 * Role:
 *   Snapshots previous stream counters, registers all streams wired in @cfg, walks `MILK_SHM_DIR`
 *   for extra `.im.shm` and `.fps.shm` files, probes every registered stream header, and tallies
 *   live stream and running FPS counts.
 *
 * Rationale:
 *   Throttling background scans to `4 Hz` (`0.25 s`) keeps syscall overhead negligible during fast
 *   keyboard/mouse interaction while `force = 1` gives instant visual confirmation right after
 *   `Deploy`, `Stop`, or `mkpupil` actions.
 */
void cs_shm_scan(cs_shm_mon_t *mon, const cacaosim_bench_cfg_t *cfg, int force)
{
    if (mon == NULL)
    {
        return;
    }
    double now_s = cs_tui_now_sec();
    if (!force && mon->last_scan_sec > 0.0 && (now_s - mon->last_scan_sec) < 0.25)
    {
        return;
    }

    cs_mon_stream_t prev_streams[CS_MAX_MON_STREAMS];
    int             n_prev = mon->n_streams;
    if (n_prev > 0)
    {
        memcpy(prev_streams, mon->streams, (size_t) n_prev * sizeof(cs_mon_stream_t));
    }

    shm_dir_path(mon->shm_dir, sizeof(mon->shm_dir));
    mon->last_scan_sec  = now_s;
    mon->n_streams      = 0;
    mon->n_live_streams = 0;
    mon->n_fps          = 0;
    mon->n_running_fps  = 0;
    mon->has_strehl     = 0;

    /* 1. Register streams explicitly declared by [bench] and optical nodes first (expected = 1) */
    if (cfg != NULL)
    {
        register_stream(mon, cfg->in_pup, "pupil", "[bench]", 1);
        register_stream(mon, cfg->in_wf, "atm_wf", "[bench]", 1);
        register_stream(mon, cfg->in_amp, "atm_amp", "[bench]", 1);
        register_stream(mon, cfg->in_ncpa, "ncpa", "[bench]", 1);

        for (uint32_t i = 0; i < cfg->n_nodes; i++)
        {
            const cacaosim_node_cfg_t *nd = &cfg->nodes[i];
            if (nd->kind == CACAOSIM_NODE_DM)
            {
                register_stream(mon, nd->in_cmd, (nd->dm.dm_mode == 1) ? "ttm_cmd" : "dm_cmd",
                                nd->id, 1);
                register_stream(mon, nd->out_opd, "dm_opd", nd->id, 1);
                register_stream(mon, nd->in_flat, "dm_flat", nd->id, 1);
            }
            else if (nd->kind == CACAOSIM_NODE_FPCAM)
            {
                register_stream(mon, nd->out_cam, "sci_psf", nd->id, 1);
                register_stream(mon, nd->out_long, "sci_long", nd->id, 1);
            }
            else if (nd->kind == CACAOSIM_NODE_PYRWFS || nd->kind == CACAOSIM_NODE_SHWFS ||
                     nd->kind == CACAOSIM_NODE_ZWFS)
            {
                register_stream(mon, nd->out_cam, "wfs_cam", nd->id, 1);
            }
            else if (nd->kind == CACAOSIM_NODE_CORO)
            {
                register_mask_shm(mon, nd->in_pup_amp, "pup_amp", nd->id);
                register_mask_shm(mon, nd->in_pup_pha, "pup_pha", nd->id);
                register_mask_shm(mon, nd->in_fpm_amp, "fpm_amp", nd->id);
                register_mask_shm(mon, nd->in_fpm_pha, "fpm_pha", nd->id);
                register_mask_shm(mon, nd->in_lyot, "lyot_amp", nd->id);
                register_mask_shm(mon, nd->in_lyot_pha, "lyot_pha", nd->id);
            }
            else if (nd->kind == CACAOSIM_NODE_WFE)
            {
                register_stream(mon, nd->in_wfe, "wfe_opd_m", nd->id, 1);
            }
        }
    }

    /* 2. Walk MILK_SHM_DIR to discover extra AO streams and active .fps.shm modules */
    DIR *dp = opendir(mon->shm_dir);
    if (dp != NULL)
    {
        struct dirent *de;
        while ((de = readdir(dp)) != NULL)
        {
            size_t len = strlen(de->d_name);
            if (len > 7u && strcmp(de->d_name + len - 7u, ".im.shm") == 0)
            {
                char stem[64];
                size_t slen = len - 7u;
                if (slen < sizeof(stem))
                {
                    memcpy(stem, de->d_name, slen);
                    stem[slen] = '\0';
                    if (is_relevant_extra_shm(stem))
                    {
                        register_stream(mon, stem, "shm_extra", "shm", 0);
                    }
                }
            }
            else if (len > 8u && strcmp(de->d_name + len - 8u, ".fps.shm") == 0)
            {
                char stem[64];
                size_t slen = len - 8u;
                if (slen < sizeof(stem))
                {
                    memcpy(stem, de->d_name, slen);
                    stem[slen] = '\0';
                    if (is_relevant_fps(stem))
                    {
                        probe_fps_file(mon, mon->shm_dir, stem);
                    }
                }
            }
        }
        closedir(dp);
    }

    /* 3. Probe IMAGE_METADATA headers and compute smoothed cnt0 frame rates */
    for (int i = 0; i < mon->n_streams; i++)
    {
        probe_one_im_shm(mon->shm_dir, &mon->streams[i], prev_streams, n_prev, now_s);
        if (mon->streams[i].exists)
        {
            mon->n_live_streams++;
        }
    }
}

/* ── Helper Binary Lookup & Bench Action Execution ────────────────────────── */

/**
 * locate_tool() - Resolve path to a sibling `cacaosim` executable or helper script
 * @exe_dir: Directory containing the running `cacaosim-CTRL` binary (`app->opts.exe_dir`)
 * @tool:    Target binary or script basename (e.g., `"cacaosim-bench"`, `"milk-fpsexec-..."`)
 * @out:     Destination resolved path or bare command name
 * @cap:     Capacity of @out in bytes
 *
 * Role:
 *   Finds companion `cacaosim` binaries/scripts whether running from an uninstalled `_build/` tree,
 *   an installed `_install/bin/` prefix, a `CACAOSIM_SRC_DIR` test environment, or system `PATH`.
 *
 * Rationale:
 *   Checking `<exe_dir>/<tool>`, `<exe_dir>/../scripts/<tool>`, and `$CACAOSIM_SRC_DIR/scripts/`
 *   with `access(..., X_OK)` before falling back to `PATH` ensures developer `_build/cacaosim-CTRL`
 *   invocations always launch matching local binaries rather than stale system-installed copies.
 */
static void locate_tool(const char *exe_dir, const char *tool, char *out, size_t cap)
{
    if (exe_dir != NULL && exe_dir[0] != '\0')
    {
        snprintf(out, cap, "%.420s/%.64s", exe_dir, tool);
        if (access(out, X_OK) == 0)
        {
            return;
        }
        snprintf(out, cap, "%.420s/../scripts/%.64s", exe_dir, tool);
        if (access(out, X_OK) == 0)
        {
            return;
        }
    }
    const char *src_env = getenv("CACAOSIM_SRC_DIR");
    if (src_env != NULL && src_env[0] != '\0')
    {
        snprintf(out, cap, "%.420s/scripts/%.64s", src_env, tool);
        if (access(out, X_OK) == 0)
        {
            return;
        }
    }
    snprintf(out, cap, "%s", tool);
}

/**
 * run_logged_cmd() - Execute a shell command via `popen()` and stream output into `cs_tui_log()`
 * @app:       Active TUI state receiving each trimmed output line
 * @cmd:       Shell command string (typically suffixed with `2>&1`)
 * @last_line: Optional buffer capturing the final non-empty output line for error summaries
 * @last_cap:  Capacity of @last_line in bytes
 *
 * Role:
 *   Runs deployment, stop, and pupil-generation helper commands without leaving alternate-screen
 *   TUI mode, capturing all diagnostic lines cleanly into the `F5:STATUS` log.
 *
 * Rationale:
 *   Using `popen(cmd, "r")` keeps the TUI terminal state intact and captures child `stdout` and
 *   `stderr` line-by-line so failures surface as concise toast banners (`last_line`) with full
 *   history in `app->logs[]`.
 *
 * Return: 0 if the child command exited with status 0, -1 on `popen()` or non-zero exit failure.
 */
static int run_logged_cmd(cs_tui_app_t *app, const char *cmd, char *last_line, size_t last_cap)
{
    if (last_line != NULL && last_cap > 0u)
    {
        last_line[0] = '\0';
    }
    FILE *pp = popen(cmd, "r");
    if (pp == NULL)
    {
        cs_tui_log(app, 3, "popen failed: %s", strerror(errno));
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), pp) != NULL)
    {
        size_t l = strlen(line);
        while (l > 0u && (line[l - 1u] == '\n' || line[l - 1u] == '\r'))
        {
            line[--l] = '\0';
        }
        if (l > 0u)
        {
            cs_tui_log(app, 0, "%s", line);
            if (last_line != NULL && last_cap > 0u)
            {
                snprintf(last_line, last_cap, "%s", line);
            }
        }
    }
    int st = pclose(pp);
    return (st == 0) ? 0 : -1;
}

/**
 * cs_tui_exec_deploy() - Validate, stage, and deploy the active optical bench configuration
 * @app:  Active TUI state (must not be NULL)
 * @mode: Execution mode passed to `cacaosim-bench start` (`"fused"` or `"modular"`)
 *
 * Role:
 *   Launches or restarts the configured optical bench hardware simulation (plus optional `atmturb`
 *   turbulence screen generator when `app->opts.with_atmturb != 0`) and immediately refreshes live
 *   `MILK_SHM_DIR` status.
 *
 * Rationale:
 *   1. Blocks deployment upfront if `app->dag_valid == 0` so broken topologies fail fast with a
 *      clear toast message instead of partial process spawns.
 *   2. Writes unsaved or untitled configurations to `<MILK_SHM_DIR>/csim_ctrl_staged.bench` rather
 *      than mutating `app->file_path`, preserving user save semantics while supporting instant
 *      what-if hardware testing.
 *
 * Return: 0 on successful deployment, -1 on validation, staging, or launch failure.
 */
int cs_tui_exec_deploy(cs_tui_app_t *app, const char *mode)
{
    if (app == NULL)
    {
        return -1;
    }
    cs_tui_refresh_analysis(app);
    if (!app->dag_valid)
    {
        cs_tui_log(app, 3, "Deploy blocked -- DAG error: %s", app->dag_err);
        return -1;
    }

    char shm_dir[256];
    shm_dir_path(shm_dir, sizeof(shm_dir));
    mkdir(shm_dir, 0775);

    /* Use saved disk file directly when clean; otherwise stage into MILK_SHM_DIR */
    char stage_path[512];
    if (app->file_path[0] != '\0' && !app->dirty && access(app->file_path, R_OK) == 0)
    {
        snprintf(stage_path, sizeof(stage_path), "%s", app->file_path);
    }
    else
    {
        snprintf(stage_path, sizeof(stage_path), "%.240s/csim_ctrl_staged.bench", shm_dir);
        char ebuf[128] = { 0 };
        if (cacaosim_bench_save_file(stage_path, &app->cfg, ebuf, sizeof(ebuf)) != 0)
        {
            cs_tui_log(app, 3, "Failed staging .bench: %s", ebuf);
            return -1;
        }
    }

    char bench_bin[512];
    locate_tool(app->opts.exe_dir, "cacaosim-bench", bench_bin, sizeof(bench_bin));

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "\"%.480s\" start --restart --config \"%.480s\" --mode %.16s --backend %d %s"
             " --seeing %.2f --no-view 2>&1",
             bench_bin, stage_path, mode, app->cfg.backend,
             app->opts.with_atmturb ? "--atmturb" : "", app->opts.seeing_arcsec);

    cs_tui_log(app, 0, "Deploying %s mode (%s%s)...", mode, app->cfg.name,
               app->opts.with_atmturb ? " + atmturb" : "");
    char last[200];
    int  rc = run_logged_cmd(app, cmd, last, sizeof(last));
    cs_shm_scan(&app->shm, &app->cfg, 1);

    if (rc == 0)
    {
        cs_tui_log(app, 1, "Bench active (%s): %d live streams, %d FPS running", mode,
                   app->shm.n_live_streams, app->shm.n_running_fps);
    }
    else
    {
        cs_tui_log(app, 3, "Deploy failed (%s): %s", mode, last[0] ? last : "check logs");
    }
    return rc;
}

/**
 * cs_tui_exec_stop() - Stop all running `cacaosim` and `atmturb` FPS processes
 * @app: Active TUI state (must not be NULL)
 *
 * Role:
 *   Executes `cacaosim-bench stop --no-view` and triggers an immediate `cs_shm_scan()` refresh so
 *   `F5:STATUS` reflects stopped FPS PIDs and decaying stream rates right away.
 *
 * Rationale:
 *   Reuses the canonical `cacaosim-bench stop` orchestration script so TUI and CLI process
 *   lifecycle management stay strictly consistent.
 *
 * Return: 0 on clean shutdown, -1 if `cacaosim-bench stop` reported a non-zero exit code.
 */
int cs_tui_exec_stop(cs_tui_app_t *app)
{
    if (app == NULL)
    {
        return -1;
    }
    char bench_bin[512];
    locate_tool(app->opts.exe_dir, "cacaosim-bench", bench_bin, sizeof(bench_bin));

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "\"%.480s\" stop --no-view 2>&1", bench_bin);
    char last[200];
    int  rc = run_logged_cmd(app, cmd, last, sizeof(last));
    cs_shm_scan(&app->shm, &app->cfg, 1);
    if (rc == 0)
    {
        cs_tui_log(app, 1, "Stopped active cacaosim / atmturb FPS units");
    }
    else
    {
        cs_tui_log(app, 2, "Stop finished with warning: %s", last);
    }
    return rc;
}

/**
 * cs_tui_exec_mkpupil() - Generate entrance pupil SHM and static `[wfe.<id>]` OPD maps in SHM
 * @app: Active TUI state (must not be NULL)
 *
 * Role:
 *   Synthesizes the anti-aliased `[bench]` telescope pupil stream (`in_pup`) via
 *   `milk-fpsexec-cacaosim-mkpupil` and initializes every `[wfe.<id>]` static `[meter]` OPD map
 *   via `milk-fpsexec-cacaosim-wfe .init_only=ON`.
 *
 * Rationale:
 *   Running `mkpupil` first guarantees the newly created `pup_name` SHM stream is present when
 *   each `cacaosim.wfe` call computes pupil-weighted RMS normalization (`rms_nm`), allowing full
 *   static optical calibration setup in a single button click or shortcut.
 *
 * Return: 0 on success, -1 if `milk-fpsexec-cacaosim-mkpupil` failed.
 */
int cs_tui_exec_mkpupil(cs_tui_app_t *app)
{
    if (app == NULL)
    {
        return -1;
    }
    char pup_bin[512];
    locate_tool(app->opts.exe_dir, "milk-fpsexec-cacaosim-mkpupil", pup_bin, sizeof(pup_bin));

    const char *pup_name = app->cfg.in_pup[0] ? app->cfg.in_pup : "csim_pup";
    char        cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "\"%.450s\" --no-procinfo exec .nwf=%u .pup_diam=%.6g .cobs=%.4g .nspider=%d"
             " .spider_w=%.4g .spider_deg=%.4g .out_pup=\"%.60s\" 2>&1",
             pup_bin, app->cfg.nwf, app->cfg.pup_diam, app->cfg.cobs, app->cfg.nspider,
             app->cfg.spider_w, app->cfg.spider_deg, pup_name);

    char last[200];
    int  rc    = run_logged_cmd(app, cmd, last, sizeof(last));
    int  n_wfe = 0;

    /* Once pupil SHM exists, synthesize pupil-normalized [meter] OPD maps for all [wfe] nodes */
    if (rc == 0)
    {
        char wfe_bin[512];
        locate_tool(app->opts.exe_dir, "milk-fpsexec-cacaosim-wfe", wfe_bin, sizeof(wfe_bin));
        for (uint32_t i = 0; i < app->cfg.n_nodes; i++)
        {
            const cacaosim_node_cfg_t *nd = &app->cfg.nodes[i];
            if (nd->kind == CACAOSIM_NODE_WFE && nd->in_wfe[0] != '\0')
            {
                char wcmd[1024];
                snprintf(wcmd, sizeof(wcmd),
                         "\"%.420s\" --no-procinfo exec .init_only=ON .nwf=%u .pup_diam=%.6g"
                         " .in_pup=\"%.60s\" .in_wfe=\"%.60s\" .rms_nm=%.4g .psd_alpha=%.4g"
                         " .wfe_seed=%llu 2>&1",
                         wfe_bin, app->cfg.nwf, app->cfg.pup_diam, pup_name, nd->in_wfe,
                         nd->wfe_rms_nm, nd->wfe_alpha, (unsigned long long) nd->wfe_seed);
                if (run_logged_cmd(app, wcmd, last, sizeof(last)) == 0)
                {
                    n_wfe++;
                }
            }
        }
    }

    cs_shm_scan(&app->shm, &app->cfg, 1);
    if (rc == 0)
    {
        if (n_wfe > 0)
        {
            cs_tui_log(app, 1, "Generated %ux%u pupil '%s' + %d static WFE [m] stream(s)",
                       app->cfg.nwf, app->cfg.nwf, pup_name, n_wfe);
        }
        else
        {
            cs_tui_log(app, 1, "Generated %ux%u anti-aliased pupil stream '%s'", app->cfg.nwf,
                       app->cfg.nwf, pup_name);
        }
    }
    else
    {
        cs_tui_log(app, 3, "mkpupil failed: %s", last[0] ? last : cmd);
    }
    return rc;
}
