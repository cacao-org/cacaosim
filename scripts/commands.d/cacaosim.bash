# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# cacaosim.bash - Declarative optical bench deployment commands built on fpspipe
# Loaded automatically from $MILK_INSTALLDIR/share/milk/commands.d/cacaosim.bash

__cacaosim_default_workdir() {
    printf '%s\n' "${CACAOSIM_WORKDIR:-${MILK_SHM_DIR:-/milk/shm}/cacaosim-bench}"
}

__cacaosim_unit_running() {
    local unit="$1" rpid=0
    @exists "f.${unit}" || return 1
    @get rpid="f.${unit}.runpid" 2>/dev/null || return 1
    (( rpid > 0 )) && kill -0 "${rpid}" 2>/dev/null
}

__cacaosim_load_persisted() {
    local item
    if [[ -f "${FPSPIPE_RUNDIR:-}/cacaosim.units" ]]; then
        while IFS= read -r item || [[ -n "${item}" ]]; do
            [[ -n "${item}" ]] && __fpspipe_track_unit "${item}"
        done < "${FPSPIPE_RUNDIR}/cacaosim.units"
    fi
    if [[ -f "${FPSPIPE_RUNDIR:-}/cacaosim.streams" ]]; then
        while IFS= read -r item || [[ -n "${item}" ]]; do
            [[ -n "${item}" ]] && __fpspipe_track_stream "${item}"
        done < "${FPSPIPE_RUNDIR}/cacaosim.streams"
    fi
}

__cacaosim_in_array() {
    local needle="$1" elem
    shift
    for elem in "$@"; do
        [[ "${elem}" == "${needle}" ]] && return 0
    done
    return 1
}

# Initialize cacaosim fpspipe workspace and restore persisted unit/stream lists
cacaosim.workdir() {
    @args 'workdir:str=:pipeline root directory' \
          'restart:bool=0:restart active units during deployment' -- "$@"

    local wdir="${workdir:-$(__cacaosim_default_workdir)}"
    fpspipe.workdir "cacaosim" "${wdir}" "" "cacaosim" "${restart}" || return 1
    __cacaosim_load_persisted
    return 0
}

# Compile and deploy a cacaosim .bench DAG (or flat SCAO bench) via fpspipe
cacaosim.deploy() {
    @args 'bench_file:str=:path or short name of .bench file (empty for flat mode)' \
          'mode:str=fused:execution mode (fused or modular)' \
          'atmturb:bool=0:launch live csim_atmturb turbulence generator' \
          'seeing:float=0.7:atmospheric seeing FWHM [arcsec]' \
          'rate:float=0:master clock override [Hz] (0 = bench default)' \
          'backend:int=-1:compute backend (-1=bench, 0=auto, 1=CPU, 2=CUDA)' \
          'restart:bool=0:force unit restart and reset live WRITERUN params' \
          'workdir:str=:pipeline workspace directory' \
          'nwf:int=128:flat-mode pupil grid size [pixels]' \
          'nact:int=20:flat-mode DM actuators across pupil' \
          'wfs_ncam:int=64:flat-mode PyWFS camera size [pixels]' \
          'fp_ncam:int=128:flat-mode science camera size [pixels]' \
          'mod_rad:float=3.0:flat-mode PyWFS modulation radius [lambda/D]' -- "$@"

    if [[ "${mode}" != "fused" && "${mode}" != "modular" ]]; then
        echo "cacaosim.deploy: invalid mode '${mode}' (expected 'fused' or 'modular')" >&2
        return 1
    fi

    local orig_pwd="${PWD}"
    if [[ -n "${bench_file}" && "${bench_file}" != /* ]]; then
        if [[ -f "${orig_pwd}/${bench_file}" ]]; then
            bench_file="$(realpath "${orig_pwd}/${bench_file}")"
        elif [[ -n "${MILK_ROOT:-}" &&
                -f "${MILK_ROOT}/share/cacaosim/examples/${bench_file}" ]]; then
            bench_file="${MILK_ROOT}/share/cacaosim/examples/${bench_file}"
        fi
    fi

    local shm_dir="${MILK_SHM_DIR:-/milk/shm}"
    mkdir -p "${shm_dir}" || return 1
    export MILK_SHM_DIR="${shm_dir}"

    local wdir="${workdir:-$(__cacaosim_default_workdir)}"
    fpspipe.workdir "cacaosim" "${wdir}" "" "cacaosim" 0 || return 1

    local -a old_units=()
    local item
    if [[ -f "${FPSPIPE_RUNDIR}/cacaosim.units" ]]; then
        while IFS= read -r item || [[ -n "${item}" ]]; do
            [[ -n "${item}" ]] && old_units+=("${item}")
        done < "${FPSPIPE_RUNDIR}/cacaosim.units"
    fi

    local plan_dir="${FPSPIPE_RUNDIR}/plan.$$"
    rm -rf "${plan_dir}"

    local -a plan_cmd=(
        cacaosim-bench-view
        --emit-plan "${plan_dir}"
        --mode "${mode}"
        --seeing "${seeing}"
    )
    (( atmturb )) && plan_cmd+=(--atmturb)
    if [[ -n "${rate}" && ! "${rate}" =~ ^0+(\.0+)?$ ]]; then
        plan_cmd+=(--rate "${rate}")
    fi
    if (( backend >= 0 )); then
        plan_cmd+=(--backend "${backend}")
    fi

    if [[ -z "${bench_file}" ]]; then
        plan_cmd+=(
            --flat
            --nwf "${nwf}"
            --nact "${nact}"
            --wfs-ncam "${wfs_ncam}"
            --fp-ncam "${fp_ncam}"
            --mod-rad "${mod_rad}"
        )
    else
        plan_cmd+=(
            --staged-bench "${plan_dir}/csim_active.bench"
            --active-bench-arg "csim_active.bench"
            "${bench_file}"
        )
    fi

    if ! "${plan_cmd[@]}"; then
        rm -rf "${plan_dir}"
        return 1
    fi

    local meta_FLAT=0 meta_NWF=128 meta_PUP_DIAM=0.9375 meta_COBS=0.30
    local meta_NSPIDER=4 meta_SPIDER_W=0.01 meta_SPIDER_DEG=45.0
    local meta_MKPUPIL_OUT="" meta_SIGNATURE=""
    local mkey mval
    while IFS=$'\t' read -r mkey mval || [[ -n "${mkey}" ]]; do
        case "${mkey}" in
            FLAT)        meta_FLAT="${mval}" ;;
            NWF)         meta_NWF="${mval}" ;;
            PUP_DIAM)    meta_PUP_DIAM="${mval}" ;;
            COBS)        meta_COBS="${mval}" ;;
            NSPIDER)     meta_NSPIDER="${mval}" ;;
            SPIDER_W)    meta_SPIDER_W="${mval}" ;;
            SPIDER_DEG)  meta_SPIDER_DEG="${mval}" ;;
            MKPUPIL_OUT) meta_MKPUPIL_OUT="${mval}" ;;
            SIGNATURE)   meta_SIGNATURE="${mval}" ;;
        esac
    done < "${plan_dir}/meta.tsv"

    local -a new_units=() new_streams=()
    while IFS= read -r item || [[ -n "${item}" ]]; do
        [[ -n "${item}" ]] && new_units+=("${item}")
    done < "${plan_dir}/units.list"
    while IFS= read -r item || [[ -n "${item}" ]]; do
        [[ -n "${item}" ]] && new_streams+=("${item}")
    done < "${plan_dir}/streams.list"

    # Prune FPS units from previous topology (and cross-mode singletons) not in new plan
    local -a prune_units=()
    local u
    for u in "${old_units[@]}" csim_htsim csim_mkbeam csim_atmturb; do
        [[ -z "${u}" ]] && continue
        if ! __cacaosim_in_array "${u}" "${new_units[@]}" && @exists "f.${u}"; then
            if ! __cacaosim_in_array "${u}" "${prune_units[@]}"; then
                prune_units+=("${u}")
            fi
        fi
    done
    if (( ${#prune_units[@]} )); then
        milk-fps-runstop -q "${prune_units[@]}" >/dev/null 2>&1 || true
        milk-fps-confstop -q "${prune_units[@]}" >/dev/null 2>&1 || true
        milk-fps-rm -f -q --ignore-missing "${prune_units[@]}" >/dev/null 2>&1 || true
    fi

    local prev_sig=""
    if [[ -f "${FPSPIPE_RUNDIR}/cacaosim.signature" ]]; then
        prev_sig="$(<"${FPSPIPE_RUNDIR}/cacaosim.signature")"
    fi

    local need_restart=0
    if (( restart )) || [[ "${prev_sig}" != "${meta_SIGNATURE}" ]]; then
        need_restart=1
    else
        for u in "${new_units[@]}"; do
            if ! __cacaosim_unit_running "${u}"; then
                need_restart=1
                break
            fi
        done
    fi

    local sname fpath
    if (( ! need_restart )); then
        while IFS=$'\t' read -r sname fpath || [[ -n "${sname}" ]]; do
            [[ -z "${sname}" ]] && continue
            if ! @exists "s.${sname}"; then
                need_restart=1
                break
            fi
        done < "${plan_dir}/loadfits.tsv"
        if [[ -n "${meta_MKPUPIL_OUT}" ]] && ! @exists "s.${meta_MKPUPIL_OUT}"; then
            need_restart=1
        fi
    fi

    if (( meta_FLAT )); then
        rm -f "${shm_dir}/csim_active.bench"
        unset CACAOSIM_BENCH_FILE
    else
        cp -f "${plan_dir}/csim_active.bench" "${shm_dir}/csim_active.bench" || {
            rm -rf "${plan_dir}"
            return 1
        }
        export CACAOSIM_BENCH_FILE="${shm_dir}/csim_active.bench"
    fi

    if (( need_restart )); then
        local -a stop_existing=()
        for u in "${new_units[@]}"; do
            @exists "f.${u}" && stop_existing+=("${u}")
        done
        if (( ${#stop_existing[@]} )); then
            milk-fps-runstop -q "${stop_existing[@]}" >/dev/null 2>&1 || true
            milk-fps-confstop -q "${stop_existing[@]}" >/dev/null 2>&1 || true
        fi

        # Stage 2D/3D FITS (.fits, .fits.gz, .fits.fz) masks preserving WCS keywords
        while IFS=$'\t' read -r sname fpath || [[ -n "${sname}" ]]; do
            [[ -z "${sname}" ]] && continue
            milk-stream-rm -q --ignore-missing "${sname}" || true
            if ! fpspipe.loadfits "${sname}" "${fpath}" 1; then
                rm -rf "${plan_dir}"
                return 1
            fi
        done < "${plan_dir}/loadfits.tsv"

        if [[ -n "${meta_MKPUPIL_OUT}" ]]; then
            local pup_existed=0
            @exists "s.${meta_MKPUPIL_OUT}" && pup_existed=1
            if ! milk-fpsexec-cacaosim-mkpupil --no-procinfo exec \
                ".nwf=${meta_NWF}" \
                ".pup_diam=${meta_PUP_DIAM}" \
                ".cobs=${meta_COBS}" \
                ".nspider=${meta_NSPIDER}" \
                ".spider_w=${meta_SPIDER_W}" \
                ".spider_deg=${meta_SPIDER_DEG}" \
                ".out_pup=${meta_MKPUPIL_OUT}" >/dev/null 2>&1; then
                rm -rf "${plan_dir}"
                return 1
            fi
            milk-fps-rm -f -q --ignore-missing csim_mkpupil >/dev/null 2>&1 || true
            __fpspipe_track_stream "${meta_MKPUPIL_OUT}"
            (( ! pup_existed )) && FPSPIPE_CREATED_STREAMS+=("${meta_MKPUPIL_OUT}")
        fi

        fpspipe.init "${plan_dir}/units.manifest" 1 || { rm -rf "${plan_dir}"; return 1; }
        fpspipe.setparams "${plan_dir}/params.batch" 0 1 || { rm -rf "${plan_dir}"; return 1; }

        local w_op w_rest
        local -a w_arr=()
        while IFS=$'\t' read -r w_op w_rest || [[ -n "${w_op}" ]]; do
            [[ -z "${w_op}" || -z "${w_rest}" ]] && continue
            read -r -a w_arr <<< "${w_rest}"
            case "${w_op}" in
                RUNSTART)
                    if ! milk-fps-runstart -q "${w_arr[@]}"; then
                        rm -rf "${plan_dir}"
                        return 1
                    fi
                    ;;
                WAIT)
                    if ! milk-stream-wait -t 15 "${w_arr[@]}"; then
                        rm -rf "${plan_dir}"
                        return 1
                    fi
                    ;;
            esac
        done < "${plan_dir}/waves.tsv"
    else
        # Warm idempotent fast-path: preserve running processes & live WRITERUN edits
        fpspipe.init "${plan_dir}/units.manifest" 0 || { rm -rf "${plan_dir}"; return 1; }
        fpspipe.setparams "${plan_dir}/params.batch" 0 0 || { rm -rf "${plan_dir}"; return 1; }

        local w_op w_rest
        local -a w_arr=()
        while IFS=$'\t' read -r w_op w_rest || [[ -n "${w_op}" ]]; do
            [[ "${w_op}" != "WAIT" || -z "${w_rest}" ]] && continue
            read -r -a w_arr <<< "${w_rest}"
            if ! milk-stream-wait -t 15 "${w_arr[@]}"; then
                rm -rf "${plan_dir}"
                return 1
            fi
        done < "${plan_dir}/waves.tsv"
    fi

    for sname in "${new_streams[@]}"; do
        __fpspipe_track_stream "${sname}"
    done
    cp -f "${plan_dir}/units.list" "${FPSPIPE_RUNDIR}/cacaosim.units"
    cp -f "${plan_dir}/streams.list" "${FPSPIPE_RUNDIR}/cacaosim.streams"
    printf '%s\n' "${meta_SIGNATURE}" > "${FPSPIPE_RUNDIR}/cacaosim.signature"

    rm -rf "${plan_dir}"
    fpspipe.commit
    return 0
}

# Update live .turb_gain across active cacaosim beam/fused FPS units
cacaosim.set_turb() {
    @args 'gain:float:turbulence OPD gain multiplier' -- "$@"

    local shm_dir="${MILK_SHM_DIR:-/milk/shm}" fp u _v=""
    shopt -s nullglob
    for fp in "${shm_dir}"/csim_*.fps.shm; do
        u="${fp##*/}"
        u="${u%.fps.shm}"
        if @get _v="f.${u}.turb_gain" 2>/dev/null; then
            @set "f.${u}.turb_gain=${gain}"
        fi
    done
    return 0
}

# Reset long-exposure PSF accumulators across active cacaosim camera/fused FPS units
cacaosim.reset_long() {
    @args -- "$@"

    local shm_dir="${MILK_SHM_DIR:-/milk/shm}" fp u _v=""
    shopt -s nullglob
    for fp in "${shm_dir}"/csim_*.fps.shm; do
        u="${fp##*/}"
        u="${u%.fps.shm}"
        if @get _v="f.${u}.reset_long" 2>/dev/null; then
            @set "f.${u}.reset_long=ON"
        fi
    done
    return 0
}

# Stop running cacaosim FPS compute units while preserving SHM streams
cacaosim.stop() {
    @args 'workdir:str=:pipeline workspace directory' -- "$@"

    cacaosim.workdir "${workdir}" 0 || return 1
    FPSPIPE_COMMITTED=1

    local -a fallback_units=(
        csim_atmturb csim_htsim csim_mkpupil csim_mkbeam csim_fpcam
        csim_pyrwfs csim_shwfs csim_zwfs csim_coro
        csim_beamsplit csim_wfe csim_dmsim csim_wfstat
    )
    local -a to_stop=()
    local u
    for u in "${FPSPIPE_UNITS[@]}" "${fallback_units[@]}"; do
        [[ -z "${u}" ]] && continue
        if @exists "f.${u}" && ! __cacaosim_in_array "${u}" "${to_stop[@]}"; then
            to_stop+=("${u}")
        fi
    done

    if (( ${#to_stop[@]} )); then
        milk-fps-runstop -q "${to_stop[@]}" >/dev/null 2>&1 || true
        milk-fps-confstop -q "${to_stop[@]}" >/dev/null 2>&1 || true
    fi
    return 0
}

# Display cacaosim optical schematic, FPS unit states, and active SHM streams
cacaosim.status() {
    @args 'workdir:str=:pipeline workspace directory' \
          'view:bool=1:render active optical bench schematic' -- "$@"

    cacaosim.workdir "${workdir}" 0 || return 1
    FPSPIPE_COMMITTED=1

    local shm_dir="${MILK_SHM_DIR:-/milk/shm}"
    echo "=== cacaosim Status (SHM: ${shm_dir}, RunDir: ${FPSPIPE_RUNDIR}) ==="
    if (( view )) && [[ -f "${shm_dir}/csim_active.bench" ]] &&
       command -v cacaosim-bench-view >/dev/null 2>&1; then
        cacaosim-bench-view "${shm_dir}/csim_active.bench" --no-footer || true
        echo ""
    fi

    fpspipe.status || true

    echo ""
    echo "Shared-Memory Streams:"
    local -a shms=("${shm_dir}"/*.im.shm)
    if [[ -e "${shms[0]}" ]]; then
        ls -lh "${shms[@]}" 2>/dev/null |
            awk '{printf "  %-44s %8s\n", $9, $5}' || true
    else
        echo "  (none)"
    fi
    return 0
}

# Stop and remove all tracked cacaosim FPS units, SHM streams, and staged artifacts
cacaosim.teardown() {
    @args 'workdir:str=:pipeline workspace directory' \
          'clean_rundir:bool=1:remove fps.*.{confdir,datadir} in rundir' -- "$@"

    cacaosim.stop "${workdir}" || true
    cacaosim.workdir "${workdir}" 0 || return 1
    FPSPIPE_COMMITTED=1

    local -a fallback_units=(
        csim_atmturb csim_htsim csim_mkpupil csim_mkbeam csim_fpcam
        csim_pyrwfs csim_shwfs csim_zwfs csim_coro
        csim_beamsplit csim_wfe csim_dmsim csim_wfstat
    )
    local u
    for u in "${fallback_units[@]}"; do
        @exists "f.${u}" && __fpspipe_track_unit "${u}"
    done

    fpspipe.teardown "${clean_rundir}" || return 1

    local shm_dir="${MILK_SHM_DIR:-/milk/shm}"
    rm -f "${FPSPIPE_RUNDIR}/cacaosim.units" \
          "${FPSPIPE_RUNDIR}/cacaosim.streams" \
          "${FPSPIPE_RUNDIR}/cacaosim.signature" \
          "${shm_dir}/csim_active.bench"
    return 0
}
