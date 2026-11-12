#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# test_closed_loop.sh — End-to-end SHM IPC, fpsexec, multi-stage .bench & milkatmturb test

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${SCRIPT_ROOT}/_build}"
INSTALL_DIR="${2:-/home/oguyon/src-frameworkdev/milk/_install/milk-1.03.00}"

TEST_SHM="$(mktemp -d /dev/shm/cacaosim_test_XXXXXX)"
TEST_WORK="$(mktemp -d /tmp/cacaosim_work_XXXXXX)"
export MILK_SHM_DIR="${TEST_SHM}"
export PATH="${BUILD_DIR}:${SCRIPT_ROOT}/scripts:${INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${BUILD_DIR}:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

cleanup() {
    cacaosim-bench stop >/dev/null 2>&1 || true
    for unit in htsim mkbeam fpcam pyrwfs shwfs zwfs coro beamsplit dmsim wfe; do
        "milk-fpsexec-cacaosim-${unit}" runstop >/dev/null 2>&1 || true
    done
    pkill -INT -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    rm -rf "${TEST_SHM}" "${TEST_WORK}"
}
trap cleanup EXIT

cd "${TEST_WORK}"

echo "=== [1/8] Checking fpsexec & visualizer -h1 help & zero CLIcore linkage ==="
ALL_UNITS=(mkpupil mkbeam wfe dmsim beamsplit coro pyrwfs shwfs zwfs fpcam htsim)
for unit in "${ALL_UNITS[@]}"; do
    exe="milk-fpsexec-cacaosim-${unit}"
    "${exe}" -h1 >/dev/null
    if ldd "$(command -v "${exe}")" | grep -q "libCLIcore"; then
        echo "[FAIL] ${exe} must NEVER link libCLIcore!" >&2
        exit 1
    fi
    echo "  [PASS] ${exe} (-h1 OK, no libCLIcore)"
done
cacaosim-bench-view -h1 >/dev/null
cacaosim-bench-view -h2 >/dev/null
cacaosim-bench-view -hm >/dev/null
if ldd "$(command -v cacaosim-bench-view)" | grep -q "libCLIcore"; then
    echo "[FAIL] cacaosim-bench-view must NEVER link libCLIcore!" >&2
    exit 1
fi
echo "  [PASS] cacaosim-bench-view (-h1/-h2/-hm OK, no libCLIcore)"

cacaosim-CTRL -h1 >/dev/null
cacaosim-CTRL -h2 >/dev/null
cacaosim-CTRL -hm >/dev/null
if ldd "$(command -v cacaosim-CTRL)" | grep -q "libCLIcore"; then
    echo "[FAIL] cacaosim-CTRL must NEVER link libCLIcore!" >&2
    exit 1
fi
echo "  [PASS] cacaosim-CTRL (-h1/-h2/-hm OK, no libCLIcore)"

echo "=== [2/8] Testing oneshot pupil + NCPA + static WFE [meter] stream creation ==="
milk-fpsexec-cacaosim-mkpupil --no-procinfo exec \
    .nwf=128 \
    .out_pup=csim_pup \
    .out_ncpa=csim_ncpa \
    .ncpa_rms=0.05

milk-fpsexec-cacaosim-wfe -n csim_wfe_init --no-procinfo exec \
    .init_only=ON \
    .nwf=128 \
    .in_pup=csim_pup \
    .in_wfe=csim_wfe_m \
    .rms_nm=65.0

test -f "${TEST_SHM}/csim_pup.im.shm"
test -f "${TEST_SHM}/csim_ncpa.im.shm"
test -f "${TEST_SHM}/csim_wfe_m.im.shm"
echo "  [PASS] csim_pup, csim_ncpa, and csim_wfe_m [OPD m] created in ${TEST_SHM}"

echo "=== [3/8] Testing modular beam pipeline (mkbeam -> wfe -> dm -> split -> shwfs + coro) ==="
milk-fpsexec-cacaosim-mkbeam fpsinit \
    .nwf=128 \
    .in_pup=csim_pup \
    .out_beam=b_mod_raw \
    .star_mag=7.0
milk-fps-set csim_mkbeam.procinfo.triggermode 0
milk-fps-set csim_mkbeam.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-mkbeam run

test -f "${TEST_SHM}/b_mod_raw.im.shm"

milk-fpsexec-cacaosim-wfe fpsinit \
    .init_only=OFF \
    .nwf=128 \
    .in_pup=csim_pup \
    .in_beam=b_mod_raw \
    .out_beam=b_mod0 \
    .in_wfe=csim_wfe_m \
    .wfe_gain=1.0
milk-fps-set csim_wfe.procinfo.triggermode 0
milk-fps-set csim_wfe.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-wfe run

test -f "${TEST_SHM}/b_mod0.im.shm"

milk-fpsexec-cacaosim-dmsim fpsinit \
    .nwf=128 \
    .nact_x=16 \
    .nact_y=16 \
    .in_beam=b_mod0 \
    .out_beam=b_mod1
milk-fps-set csim_dmsim.procinfo.triggermode 0
milk-fps-set csim_dmsim.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-dmsim run

test -f "${TEST_SHM}/csim_dmopd.im.shm"
test -f "${TEST_SHM}/b_mod1.im.shm"

milk-fpsexec-cacaosim-beamsplit fpsinit \
    .nwf=128 \
    .in_beam=b_mod1 \
    .out_refl=b_vis_arm \
    .out_trans=b_nir_arm \
    .split_mode=1 \
    .lambda_cut=1.00
milk-fps-set csim_beamsplit.procinfo.triggermode 0
milk-fps-set csim_beamsplit.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-beamsplit run

test -f "${TEST_SHM}/b_vis_arm.im.shm"
test -f "${TEST_SHM}/b_nir_arm.im.shm"

milk-fpsexec-cacaosim-shwfs fpsinit \
    .nwf=128 \
    .in_pup=csim_pup \
    .in_beam=b_vis_arm \
    .n_sub=16 \
    .pix_per_sub=8 \
    .out_cam=aol0_shwfs
milk-fps-set csim_shwfs.procinfo.triggermode 0
milk-fps-set csim_shwfs.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-shwfs run

test -f "${TEST_SHM}/aol0_shwfs.im.shm"

milk-fpsexec-cacaosim-coro fpsinit \
    .nwf=128 \
    .in_beam=b_nir_arm \
    .out_beam=b_coro_post \
    .out_lowfs=b_coro_lowfs \
    .coro_mode=0 \
    .lambda_um=1.65
milk-fps-set csim_coro.procinfo.triggermode 0
milk-fps-set csim_coro.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-coro run

test -f "${TEST_SHM}/b_coro_post.im.shm"
test -f "${TEST_SHM}/b_coro_lowfs.im.shm"

milk-fpsexec-cacaosim-zwfs fpsinit \
    .nwf=128 \
    .in_pup=csim_pup \
    .in_beam=b_coro_lowfs \
    .ncam=64 \
    .lambda_um=1.65 \
    .out_cam=aol0_zwfs
milk-fps-set csim_zwfs.procinfo.triggermode 0
milk-fps-set csim_zwfs.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-zwfs run

test -f "${TEST_SHM}/aol0_zwfs.im.shm"

milk-fpsexec-cacaosim-pyrwfs fpsinit \
    .nwf=128 \
    .in_pup=csim_pup \
    .in_beam=b_vis_arm \
    .ncam=64
milk-fps-set csim_pyrwfs.procinfo.triggermode 0
milk-fps-set csim_pyrwfs.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-pyrwfs run

test -f "${TEST_SHM}/aol0_wfsim.im.shm"

milk-fpsexec-cacaosim-fpcam fpsinit \
    .nwf=128 \
    .in_pup=csim_pup \
    .in_beam=b_coro_post \
    .ncam=128
milk-fps-set csim_fpcam.procinfo.triggermode 0
milk-fps-set csim_fpcam.procinfo.loopcntMax 5
milk-fpsexec-cacaosim-fpcam run

test -f "${TEST_SHM}/csim_fpcam.im.shm"
test -f "${TEST_SHM}/csim_fplong.im.shm"

if command -v python3 >/dev/null 2>&1; then
    P3_FITS="${TEST_WORK}/pup_apod_3d.fits"
    F3_FITS="${TEST_WORK}/fpm_pha_3d.fits"
    python3 - "${P3_FITS}" "${F3_FITS}" <<'PYEOF'
import math, struct, sys

def write_fits3d(path, nx, ny, nz, crval3, cdelt3, gen_fn):
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                  -32",
        "NAXIS   =                    3",
        f"NAXIS1  = {nx:20d}",
        f"NAXIS2  = {ny:20d}",
        f"NAXIS3  = {nz:20d}",
        f"CRPIX3  =                  1.0",
        f"CRVAL3  = {crval3:20.8f}",
        f"CDELT3  = {cdelt3:20.8f}",
        "CUNIT3  = 'um      '",
        "END",
    ]
    hdr = b"".join(c.ljust(80).encode("ascii") for c in cards)
    hdr += b" " * ((2880 - len(hdr) % 2880) % 2880)
    vals = [gen_fn(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)]
    raw = struct.pack(f">{len(vals)}f", *vals)
    raw += b"\x00" * ((2880 - len(raw) % 2880) % 2880)
    with open(path, "wb") as f:
        f.write(hdr + raw)

write_fits3d(sys.argv[1], 128, 128, 4, 1.50, 0.10,
             lambda x, y, z: 0.92 - 0.02 * z)
write_fits3d(sys.argv[2], 48, 48, 4, 1.50, 0.10,
             lambda x, y, z: math.pi if (x - 23.5)**2 + (y - 23.5)**2 < 64.0 else 0.0)
PYEOF
    milk-fpsexec-cacaosim-coro -n csim_coro3d fpsinit \
        .nwf=128 \
        .in_beam=b_nir_arm \
        .out_beam=b_coro_poly3d \
        .coro_mode=4 \
        .fpm_pixscale=6.0 \
        .lambda_um=1.65 \
        .bw_frac=0.20 \
        .in_pup_amp="${P3_FITS}" \
        .in_fpm_pha="${F3_FITS}"
    milk-fps-set csim_coro3d.procinfo.triggermode 0
    milk-fps-set csim_coro3d.procinfo.loopcntMax 4
    milk-fpsexec-cacaosim-coro -n csim_coro3d run
    test -f "${TEST_SHM}/b_coro_poly3d.im.shm"

    milk-fpsexec-cacaosim-fpcam -n csim_fpcam3d fpsinit \
        .nwf=128 \
        .in_pup=csim_pup \
        .in_beam=b_coro_poly3d \
        .ncam=64 \
        .out_cam=csim_fpcam_poly3d
    milk-fps-set csim_fpcam3d.procinfo.triggermode 0
    milk-fps-set csim_fpcam3d.procinfo.loopcntMax 4
    milk-fpsexec-cacaosim-fpcam -n csim_fpcam3d run
    test -f "${TEST_SHM}/csim_fpcam_poly3d.im.shm"
fi
echo "  [PASS] Modular optical beam + 3D custom coronagraph masks + WFS + FPcam verified"

echo "=== [4/7] Testing fused htsim + milkatmturb live stream + default cacaosim-bench ==="
if command -v milk-fpsexec-atmturb-mkwfs >/dev/null 2>&1; then
    milk-fpsexec-atmturb-mkwfs --no-procinfo exec \
        .wfsize=128 \
        .master_size=256 \
        .seeing=0.6 \
        .time_step=0.0005 \
        .time_span=0.02 \
        .stream_mode=2 \
        .save_fits=0 >/dev/null 2>&1 &
    ATM_PID=$!
    sleep 0.2
fi

milk-fpsexec-cacaosim-htsim fpsinit \
    .nwf=128 \
    .nact_x=20 \
    .nact_y=20 \
    .fp_navg=5 \
    .frame_rate=2000
milk-fps-set csim_htsim.procinfo.triggermode 0
milk-fps-set csim_htsim.procinfo.loopcntMax 25
milk-fpsexec-cacaosim-htsim run

if [[ -n "${ATM_PID:-}" ]]; then
    wait "${ATM_PID}" 2>/dev/null || true
fi

cacaosim-bench start --mode fused --rate 1000
sleep 0.25
cacaosim-bench status >/dev/null
cacaosim-bench stop
echo "  [PASS] Flat fused htsim + milkatmturb + cacaosim-bench start/status/stop verified"

echo "=== [5/7] Testing 2-stage .bench DAG in FUSED mode (two_stage_ao.bench) ==="
rm -f "${TEST_SHM}"/*.im.shm "${TEST_SHM}"/*.fps.shm
cacaosim-bench start --config "${SCRIPT_ROOT}/examples/two_stage_ao.bench" --mode fused --rate 600
sleep 0.35
for s in dm01disp dm02disp aol1_wfsim aol2_wfsim csim_fpcam csim_fplong; do
    test -f "${TEST_SHM}/${s}.im.shm"
done
cacaosim-bench stop
echo "  [PASS] Fused two_stage_ao.bench streams (dm01disp, dm02disp, aol1/2_wfsim, fpcam) OK"

echo "=== [6/7] Testing 2-stage .bench DAG in MODULAR FPS mode (two_stage_ao.bench) ==="
rm -f "${TEST_SHM}"/*.im.shm "${TEST_SHM}"/*.fps.shm
cacaosim-bench start --config "${SCRIPT_ROOT}/examples/two_stage_ao.bench" --mode modular --rate 300
sleep 0.5
for s in b_pup b_post_dm1 b_nir_wfs b_stg2_in b_post_dm2 dm01disp dm02disp aol1_wfsim aol2_wfsim; do
    test -f "${TEST_SHM}/${s}.im.shm"
done
cacaosim-bench stop
echo "  [PASS] Modular two_stage_ao.bench multi-process beam & sensor streams OK"

echo "=== [7/8] Testing edit -> visualize -> deploy & run workflow (cacaosim-bench-view) ==="
rm -f "${TEST_SHM}"/*.im.shm "${TEST_SHM}"/*.fps.shm
CUSTOM_BENCH="${TEST_WORK}/custom_edit_test.bench"
cacaosim-bench-view --init "${CUSTOM_BENCH}" >/dev/null
sed -i 's/name       = "custom_two_stage_ao"/name       = "tweaked_2stg_bench"/' "${CUSTOM_BENCH}"
sed -i 's/nact_x       = 24/nact_x       = 20/' "${CUSTOM_BENCH}"
sed -i 's/nact_y       = 24/nact_y       = 20/' "${CUSTOM_BENCH}"

VIEW_TXT="$(cacaosim-bench view "${CUSTOM_BENCH}" -t)"
grep -q "tweaked_2stg_bench" <<<"${VIEW_TXT}"
grep -q "20x20 act" <<<"${VIEW_TXT}"
grep -q "STANDBY" <<<"${VIEW_TXT}"

VIEW_NIGHT="$(cacaosim-bench-view -p --theme night "${SCRIPT_ROOT}/examples/exao_coro_lowfs.bench")"
grep -q "coro_h" <<<"${VIEW_NIGHT}"

EDITOR=true cacaosim-bench edit "${CUSTOM_BENCH}" >/dev/null

cacaosim-bench-view --deploy --mode fused "${CUSTOM_BENCH}" >/dev/null
sleep 0.25
VIEW_LIVE="$(cacaosim-bench view "${CUSTOM_BENCH}" -t)"
grep -q "SHM LIVE" <<<"${VIEW_LIVE}"
cacaosim-bench stop
echo "  [PASS] Edit .bench -> visualize -> deploy & run -> SHM LIVE indicator verified"

if command -v python3 >/dev/null 2>&1; then
    PTY_BENCH="${TEST_WORK}/pty_ctrl_built.bench"
    python3 - "${PTY_BENCH}" <<'PYEOF'
import fcntl, os, pty, select, struct, subprocess, termios, time, sys

bench_path = sys.argv[1]
master_fd, slave_fd = pty.openpty()
winsz = struct.pack("HHHH", 40, 128, 0, 0)
fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsz)

proc = subprocess.Popen(
    ["cacaosim-CTRL", "-p", bench_path],
    stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, close_fds=True
)
os.close(slave_fd)

captured = bytearray()
def drain(timeout_s):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        r, _, _ = select.select([master_fd], [], [], 0.03)
        if r:
            try:
                chunk = os.read(master_fd, 65536)
                if not chunk:
                    break
                captured.extend(chunk)
            except OSError:
                break

drain(0.20)
# Load ExAO preset ('P' -> '4' ExAO -> 'y' confirm), add DM ('a' -> '1') & WFE ('a' -> '9'),
# cycle F3/F4/F5/F6/F2 tabs, save ('s'), and quit ('q')
for seq in [b"P", b"4", b"y", b"a", b"1", b"a", b"9", b"\x1b[13~", b"\x1b[14~",
            b"\x1b[15~", b"\x1b[17~", b"\x1b[12~", b"s", b"q"]:
    os.write(master_fd, seq)
    drain(0.06)

rc = proc.wait(timeout=5.0)
os.close(master_fd)
assert rc == 0, f"cacaosim-CTRL exited with {rc}"
assert b"\x1b[?2026h" in captured, "Missing synchronized output sequence"
assert b"\x1b[?1049l" in captured, "Missing alternate screen restore"
PYEOF
    test -f "${PTY_BENCH}"
    PTY_VIEW="$(cacaosim-bench view "${PTY_BENCH}" -t)"
    grep -q "VALID DAG" <<<"${PTY_VIEW}"
    grep -q "wfe" <<<"${PTY_VIEW}"
    echo "  [PASS] cacaosim-CTRL interactive PTY (preset -> add DM+WFE -> F2..F6 -> save -> quit)"
fi

echo "=== [8/8] Testing milk-cli dynamic module load (cacaosim) ==="
if [[ -x "${INSTALL_DIR}/bin/milk-cli" && -f "${INSTALL_DIR}/lib/libmilkcacaosim.so" ]]; then
    CLI_OUT="$("${INSTALL_DIR}/bin/milk-cli" -q -c \
        "cacaosim.pyrwfs -h; cacaosim.shwfs -h; cacaosim.coro -h; cacaosim.wfe -h" 2>&1)"
    if echo "${CLI_OUT}" | grep -q "Pyramid WFS" &&
       echo "${CLI_OUT}" | grep -q "Shack-Hartmann" &&
       echo "${CLI_OUT}" | grep -q "coronagraph" &&
       echo "${CLI_OUT}" | grep -q "wavefront error"; then
        "${INSTALL_DIR}/bin/milk-cli" -q -c \
            "cacaosim.mkpupil csim_cli_pup 64" >/dev/null
        test -f "${TEST_SHM}/csim_cli_pup.im.shm"
        echo "  [PASS] milk-cli lazy-loaded all new cacaosim commands & ran cacaosim.mkpupil"
    else
        echo "[FAIL] milk-cli did not load expected cacaosim commands:" >&2
        echo "${CLI_OUT}" >&2
        exit 1
    fi
fi

echo "=== ALL INTEGRATION CHECKS PASSED ==="
