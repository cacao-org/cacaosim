// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cacaosim_tui_shm.c
 * @brief   Non-intrusive MILK_SHM_DIR stream/FPS scanner and bench deployment actions
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

static void shm_dir_path(char *out, size_t cap)
{
    const char *sd = getenv("MILK_SHM_DIR");
    snprintf(out, cap, "%s", (sd != NULL && sd[0] != '\0') ? sd : "/milk/shm");
}

static int pid_alive(pid_t p)
{
    if (p <= 0)
    {
        return 0;
    }
    return (kill(p, 0) == 0 || errno == EPERM);
}

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

    /* Compute frame rate from previous scan snapshot */
    for (int i = 0; i < n_prev; i++)
    {
        if (strcmp(prev_arr[i].name, s->name) == 0 && prev_arr[i].exists && prev_arr[i].has_prev)
        {
            double dt = now_s - prev_arr[i].prev_t_sec;
            if (dt >= 0.08)
            {
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

    if (fe->has_strehl && (fe->run_alive || !mon->has_strehl))
    {
        mon->strehl_inst = fe->strehl_inst;
        mon->strehl_ave  = fe->strehl_ave;
        mon->has_strehl  = 1;
    }

    munmap(addr, map_len);
}

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

static int is_relevant_extra_shm(const char *stem)
{
    return (strncmp(stem, "csim_", 5) == 0 || strncmp(stem, "aol", 3) == 0 ||
            strncmp(stem, "dm", 2) == 0 || strncmp(stem, "ttm", 3) == 0 ||
            strncmp(stem, "lowfs", 5) == 0 || strncmp(stem, "b_", 2) == 0 ||
            strncmp(stem, "wfe", 3) == 0 || strstr(stem, "_wfe") != NULL ||
            strncmp(stem, "apod", 4) == 0 || strncmp(stem, "fpm", 3) == 0 ||
            strncmp(stem, "lyot", 4) == 0 || strcmp(stem, "outarraypha") == 0);
}

static int is_relevant_fps(const char *stem)
{
    return (strncmp(stem, "csim_", 5) == 0 || strncmp(stem, "cacaosim", 8) == 0 ||
            strncmp(stem, "atmturb", 7) == 0 || strncmp(stem, "mkwfs", 5) == 0 ||
            strncmp(stem, "aol", 3) == 0);
}

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
             "\"%.480s\" start --config \"%.480s\" --mode %.16s --backend %d %s --seeing %.2f"
             " --no-view 2>&1",
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
