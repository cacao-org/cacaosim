# cacaosim — Modular Multi-Stage Adaptive Optics Hardware & Optics Simulator for milk / cacao

`cacaosim` (`libmilkcacaosim.so`) is a high-performance C11 / OpenMP / CUDA plugin for `milk`
(`feat/bash-engine`) designed as a real-time Hardware-in-the-Loop (HIL) adaptive optics bench
simulator for `cacao`. It natively consumes live 2D phase and amplitude streams (`outarraypha`,
`outarrayamp`) produced by `milkatmturb` (`AtmosphericTurbulenceSimul`) and supports arbitrary
**multi-DM, multi-WFS, multi-rate, and coronagraphic optical bench topologies** via three
interoperable layers:

1. **Pure C/CUDA Physics Engines (`libmilkcacaosim.so`)**: Zero-SHM library APIs callable directly
   from C/C++ calibration harnesses, benchmarks, and unit tests (`cacaosim_bench_*`,
   `cacaosim_dm_*`, `cacaosim_coro_*`, `cacaosim_pyrwfs_*`, `cacaosim_shwfs_*`, `cacaosim_zwfs_*`,
   `cacaosim_fpcam_*`).
2. **Stream-Composable `MILK_RUNNABLE` FPS Processes**: Standalone `milk-fpsexec-cacaosim-*` daemons
   and `milk` CLI commands connected by `[nwf, nwf, 2]` `FLOAT32` optical beam streams
   (`[OPD_um, Amplitude]` + 32-bin photon spectrum keywords).
3. **Declarative `.bench` DAG Engine, `fpspipe` Orchestration & Interactive TUI**: INI/TOML-style
   `.bench` topology files inspected/edited in `cacaosim-CTRL` or `cacaosim-bench-view`, compiled by
   `cacaosim_bench_plan.c` (`--emit-plan`), and deployed via `share/milk/commands.d/cacaosim.bash`
   (`cacaosim.deploy` / `fpspipe.*` + event-driven `milk-stream-wait`) either **fused inside a
   single zero-hop GPU/CPU process** (`cacaosim.htsim`) or **modularly across wave-ordered
   semaphore-chained FPS processes** (`cacaosim-bench deploy --config ... --mode modular`).

---

## Compute Units & Source Architecture

All 11 compute units follow `milk_module_example` (`MILK_RUNNABLE`, `MILK_PARAMS_DEFINE`,
`MILK_MODULE_RUNNABLES`, `EXPORT_FPSEXEC`) with zero `libCLIcore` dependency in standalone
`milk-fpsexec-*` binaries:

| Command | `fpsexec` Binary | Dir | Role & Optical Ports |
| :--- | :--- | :--- | :--- |
| `cacaosim.mkpupil` | `milk-fpsexec-cacaosim-mkpupil` | `pupil/` | Pupil + NCPA (`-> csim_pup`) |
| `cacaosim.mkbeam` | `milk-fpsexec-cacaosim-mkbeam` | `beam/` | `[OPD, Amp]` + 32-bin spectrum |
| `cacaosim.wfe` | `milk-fpsexec-cacaosim-wfe` | `beam/` | Static wavefront error (`OPD [m]`) |
| `cacaosim.dmsim` | `milk-fpsexec-cacaosim-dmsim` | `dm/` | Zonal DM / TTM (`in->out`, ZOH) |
| `cacaosim.beamsplit` | `milk-fpsexec-cacaosim-beamsplit` | `beam/` | Gray/LP/SP/BP splitter |
| `cacaosim.coro` | `milk-fpsexec-cacaosim-coro` | `coro/` | 2D/3D coro + FPM/Lyot LOWFS |
| `cacaosim.pyrwfs` | `milk-fpsexec-cacaosim-pyrwfs` | `pyrwfs/` | 4-sided PyWFS (CUDA Graph) |
| `cacaosim.shwfs` | `milk-fpsexec-cacaosim-shwfs` | `shwfs/` | Diffractive lenslet SHWFS |
| `cacaosim.zwfs` | `milk-fpsexec-cacaosim-zwfs` | `zwfs/` | Babinet MFT Zernike WFS |
| `cacaosim.fpcam` | `milk-fpsexec-cacaosim-fpcam` | `fpcam/` | Polychromatic MFT/FFT camera |
| `cacaosim.htsim` | `milk-fpsexec-cacaosim-htsim` | `fused/` | Fused flat or `.bench` runner |

Supporting infrastructure lives in `src/common/` (64-byte aligned memory, `CLOCK_MONOTONIC`
`TIMER_ABSTIME` pacing, thread-local `xoshiro128++` & CUDA `Philox-2x32` PRNGs, 32-bin spectral
radiometry, 2D/3D FITS/SHM mask I/O, detector readout) and `src/bench/` (`.bench` parser/serializer,
topological beam-splicing editor, `cacaosim-bench-view`, and `cacaosim-CTRL`).

### Key Physical & Algorithmic Choices

- **Wavelength-Agnostic `[OPD_um, Amplitude]` Beams + 32-Bin Spectral Keywords**: Carrying geometric
  optical path difference in microns alongside `CSIM_S00..CSIM_S31` photon arrival rates
  (`[photons/s]` across `0.40..2.40 um`) avoids complex phase wrapping across dichroic relays while
  deferring wavelength-specific `2*pi/lambda` evaluation to diffractive elements (`coro`, WFSs,
  `fpcam`). Post-coronagraph polychromatic fields automatically promote to `[nwf, nwf, 2*N_lambda]`.
- **Pixel-Centric 16-Tap DM Gather + Exact Analytical ZOH Dynamics**: Precomputing a `[npix * 16]`
  gather stencil normalized by the unclipped 4x4 Gaussian lattice sum eliminates piston
  print-through and GPU/OpenMP scatter atomics; closed-form 1st/2nd-order Zero-Order Hold matrices
  remain unconditionally stable at arbitrary `dt_sec / tau_sec` or resonance ratios.
- **Propagator Selection by Mask Support**: Compact focal masks (`lyot_spot`, `phase_dimple`,
  `custom_mft`, `zwfs`) and fine-sampled PSF cores (`fpcam` MFT) use Soummer's Babinet Matrix-DFT
  (`cblas_cgemm` / `cublasCgemm`) for exact sub-`lambda/D` sampling without huge FFT zero-padding,
  while extended phase masks (`vortex`, `fqpm`, `custom_fft`) use padded 2D FFTs, Pyramid modulation
  uses the Fourier-shift `1 + N_mod` FFT reduction captured in a single CUDA Graph, and SHWFS uses
  L1-resident thread-local FFTW3f / batched `cufftPlanMany` diffractive lenslet transforms.

---

## Multi-Stage Optical Bench Architecture (`.bench`)

### 1. Ordered DMs, Dichroics, Registration & Multi-Rate Photometry

- **Sequential DM / TTM Deployment**: Each `[dm.<id>]` or `[ttm.<id>]` subtracts its reflective
  surface OPD (`dm_opd = refl_scale * surface`) from `in` to produce `out`, enforcing exact upstream
  vs downstream optical causality (`Stage 1 Woofer -> Dichroic 1 -> Stage 2 Tweeter -> Dichroic 2`).
- **Photon-Conserving Splitters**: `[splitter.<id>]` supports `"gray"`, `"dichroic_lp"`,
  `"dichroic_sp"`, and `"dichroic_bp"` (band-reflect/notch, e.g. reflecting J-band `1.22 um` to a
  Stage-1 WFS while transmitting both Visible `< 1.0 um` and H-band `1.65 um` downstream), plus
  per-arm static NCPA, pupil/Lyot masks, and 2D subpixel `(shift_x, shift_y, rot_deg, scale)`
  registration.
- **Multi-Rate Subframe Integration**: The bench ticks at `master_hz` (e.g. `2000..3000 Hz`) while
  each WFS or camera integrates `navg` (or `rate_hz`) subframes before detector readout +
  Poisson/RON noise quantization.
- **Bundled Reference Topologies**:
  - `examples/two_stage_ao.bench`: `12x12` Woofer DM (`dm01disp`) + J-band (`1.22 um`) notch
    dichroic + `1 kHz` NIR PyWFS (`aol1_wfsim`) feeding a `24x24` Tweeter DM (`dm02disp`) +
    `1.00 um` LP dichroic + `3 kHz` Visible PyWFS (`aol2_wfsim`) + `100 Hz` H-band `fpcam`.
  - `examples/exao_coro_lowfs.bench`: 2-axis TTM + `16x16` DM + `2 kHz` Visible SHWFS + H-band Lyot
    coronagraph (`3 lambda/D`) with FPM core reflection feeding a `500 Hz` Zernike LOWFS
    (`aol0_lowfs`) and post-Lyot `100 Hz` coronagraphic science camera (`csim_corocam`).

### 2. Hardware Configurator, Status GUI & CLI Visualizer (`cacaosim-CTRL`, `cacaosim-bench-view`)

`cacaosim-CTRL` (`cacaosim-bench gui` / `cacaosim-bench-view -i`) provides an interactive,
flicker-free terminal hardware builder and live `MILK_SHM_DIR` status monitor modeled after
`milk-CTRL` (shadow/front cell-grid delta buffer with `\033[?2026h` synchronized updates, SGR 1006
mouse + draggable splitters + hover tooltips, 5 `milk` themes, and `--ascii` 7-bit mode):

- **`F2:BUILDER`**: 3-pane workspace with **Elements & Arranger** (`[a]` add DM/TTM/Splitter/Coro/
  PyWFS/SHWFS/ZWFS/FPCAM/WFE, `[g]` InitSHM pupil+WFE maps, `[c]` clone, `[K]`/`[J]` optical
  splicing reorder, `[x]` branch swap, `[w]` upstream beam picker, `[d]` beam-bridging delete,
  `[u]`/`Ctrl+R` 24-step undo/redo, live `16x16` pupil dock with array/beam pix & `[m/pix]`,
  `[l/D/pix]`, `[arcsec/pix]` scales), **clickable 2D Optical Schematic**, and **Parameter
  Inspector** (`1:OPTICS`, `2:DYN/DET`, `3:SHM/WIRING`).
- **`F3:SCHEMATIC` / `F4:TABLES` / `F5:STATUS` / `F6:SOURCE`**: Full-screen optical diagram &
  32-bin (`0.40..2.40 um`) spectral deck, hardware summary tables, live `.im.shm` / `.fps.shm`
  rates + Strehl + one-key `fused`/`modular` deployment, and `.bench` INI/TOML preview.

---

## Build & Installation

```bash
export MILK_INSTALL_ROOT=/home/oguyon/src-frameworkdev/milk/_install/milk-1.03.00
cmake -S /home/oguyon/src-frameworkdev/cacaosim -B _build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${MILK_INSTALL_ROOT}" \
    -DCMAKE_INSTALL_PREFIX="${MILK_INSTALL_ROOT}"
cmake --build _build -j$(nproc)
cmake --install _build
ctest --test-dir _build --output-on-failure
```

Pass `-DCACAOSIM_ENABLE_CUDA=OFF` to force a CPU-only build (`FFTW3f` + `OpenBLAS` + `OpenMP`).

---

## Worked Examples

### Example 1: Declarative `.bench` SCAO + Science Camera Walkthrough

Create an isolated shared-memory workspace and write a minimal single-stage AO bench (`my_ao.bench`)
combining an 8.2 m telescope, `60 nm` static WFE, `20x20` DM, `1.00 um` long-pass dichroic, `2 kHz`
Visible Pyramid WFS, and `100 Hz` H-band science camera:

```bash
export PATH="/home/oguyon/src-frameworkdev/milk/_install/milk-1.03.00/bin:${PATH}"
export MILK_SHM_DIR="/tmp/csim_demo_${USER}"
mkdir -p "${MILK_SHM_DIR}"

cat <<'EOF' > my_ao.bench
[bench]
name       = "scao_demo"
nwf        = 128
pup_diam   = 0.9375
cobs       = 0.30
nspider    = 4
spider_w   = 0.012
tel_diam_m = 8.2
star_mag   = 7.0
star_teff  = 5778.0
tput_tel   = 0.78
master_hz  = 2000.0
in_wf      = "outarraypha"
in_pup     = "csim_pup"
out_beam   = "b_pup"

[wfe.ncpa0]
in        = "b_pup"
out       = "b_wfe"
in_wfe    = "csim_wfe_m"
rms_nm    = 60.0
psd_alpha = 3.0

[dm.dm0]
in           = "b_wfe"
out          = "b_corr"
in_cmd       = "dm00disp"
nact_x       = 20
nact_y       = 20
if_coupling  = 0.18
max_stroke   = 5.0
dyn_mode     = 1
tau_us       = 120.0
delay_frames = 1.0

[splitter.bs0]
in         = "b_corr"
out_refl   = "b_vis"
out_trans  = "b_nir"
mode       = "dichroic_lp"
lambda_cut = 1.00

[pyrwfs.wfs0]
in            = "b_vis"
out_cam       = "aol0_wfsim"
ncam          = 64
lambda_um     = 0.75
bw_frac       = 0.22
use_beam_flux = 1
navg          = 1
mod_rad       = 3.0
mod_npts      = 16
ron_e         = 0.5
out_dtype     = "uint16"

[fpcam.sci]
in            = "b_nir"
out_cam       = "csim_fpcam"
out_long      = "csim_fplong"
ncam          = 128
prop_mode     = "mft"
lambda_um     = 1.65
bw_frac       = 0.18
pixscale_ld   = 3.0
use_beam_flux = 1
navg          = 20
ron_e         = 1.0
out_dtype     = "uint16"
EOF
```

Inspect the schematic and photon budget, open the interactive configurator, and run the bench:

```bash
# 1. Render static 2D schematic, 32-bin spectra, plate scales, and detector SNR regime:
cacaosim-bench-view my_ao.bench

# 2. (Optional) Edit topology, splice coronagraphs/DMs, or inspect scales interactively:
cacaosim-CTRL my_ao.bench

# 3. Deploy the simulator via fpspipe in zero-hop fused GPU/CPU mode (add --atmturb for live
#    turbulence, --mode modular for wave-ordered multi-process FPS daemons, or --restart to reset
#    live FPFLAG_WRITERUN edits; 2D/3D .fits/.fits.gz/.fits.fz masks auto-stage into SHM):
cacaosim-bench deploy --config my_ao.bench --mode fused

# 4. Check live SHM streams (dm00disp, aol0_wfsim, csim_fpcam) and FPS status:
cacaosim-bench status
cacaosim-CTRL --status my_ao.bench   # interactive F5:STATUS live SHM & Strehl monitor

# 5. Stop running loops (preserving SHM) or tear down all tracked FPS & SHM artifacts:
cacaosim-bench stop
cacaosim-bench teardown

# 6. Or orchestrate directly from milk-script via share/milk/commands.d/cacaosim.bash:
milk-script -e <<< 'cacaosim.deploy "my_ao.bench" fused; cacaosim.status "" 0; cacaosim.teardown'
```

---

### Example 2: Composing Standalone `milk-fpsexec-cacaosim-*` Units Directly

Every optical element can also be initialized and orchestrated individually via standard `milk` FPS
commands (`fpsinit`, `milk-fps-set`, `confstart`, `runstart`, `milk-fpsCTRL`) without a `.bench`
file:

```bash
export PATH="/home/oguyon/src-frameworkdev/milk/_install/milk-1.03.00/bin:${PATH}"
export MILK_SHM_DIR="/tmp/csim_modular_${USER}"
mkdir -p "${MILK_SHM_DIR}"

# 1. One-shot pupil + static NCPA generation (creates csim_pup.im.shm & csim_ncpa.im.shm):
milk-fpsexec-cacaosim-mkpupil --no-procinfo exec \
    .nwf=128 .out_pup=csim_pup .out_ncpa=csim_ncpa .ncpa_rms=0.05

# 2. Initialize FPS units for source beam -> 16x16 DM -> Pyramid WFS + H-band science camera:
milk-fpsexec-cacaosim-mkbeam fpsinit \
    .nwf=128 .in_pup=csim_pup .in_ncpa=csim_ncpa .out_beam=b_raw .star_mag=7.0 .frame_rate=1000
milk-fpsexec-cacaosim-dmsim fpsinit \
    .nwf=128 .nact_x=16 .nact_y=16 .in_cmd=dm00disp .in_beam=b_raw .out_beam=b_dm
milk-fpsexec-cacaosim-pyrwfs fpsinit \
    .nwf=128 .ncam=64 .in_beam=b_dm .out_cam=aol0_wfsim .mod_rad=3.0 .mod_npts=16
milk-fpsexec-cacaosim-fpcam fpsinit \
    .nwf=128 .ncam=128 .in_beam=b_dm .out_cam=csim_fpcam .lambda_um=1.65 .navg=10

# 3. Chain downstream units via ImageStreamIO semaphores (triggermode = 3); mkbeam paces at 1 kHz:
milk-fps-set csim_mkbeam.procinfo.triggermode 0
for unit in "csim_dmsim:b_raw" "csim_pyrwfs:b_dm" "csim_fpcam:b_dm"; do
    fps="${unit%%:*}"; trig="${unit##*:}"
    milk-fps-set "${fps}.procinfo.triggermode" 3
    milk-fps-set "${fps}.procinfo.triggerstreamname" "${trig}"
done

# 4. Start background compute loops (mkbeam first so upstream beams exist on startup):
for exe in mkbeam dmsim pyrwfs fpcam; do
    "milk-fpsexec-cacaosim-${exe}" confstart
    "milk-fpsexec-cacaosim-${exe}" runstart
done

# 5. Tune live parameters on the fly (e.g. modulation radius or shutter) or open milk-fpsCTRL:
milk-fps-set csim_pyrwfs.mod_rad 2.0
cacaosim-bench-view --live           # auto-discovers live SHM streams even without a .bench file

# 6. Clean shutdown:
for exe in fpcam pyrwfs dmsim mkbeam; do
    "milk-fpsexec-cacaosim-${exe}" runstop
    "milk-fpsexec-cacaosim-${exe}" confstop
done
```

---

### Example 3: Embedded In-Process C API (`libmilkcacaosim.so`)

For high-speed calibration matrices, Monte-Carlo error budgets, or custom C/C++ RTC loops without
shared-memory IPC (`connect_shm = 0`), link against `libmilkcacaosim.so` and step any `.bench` DAG
directly in memory:

```c
#include <stdio.h>
#include "milkcacaosim.h"

int main(void)
{
    cacaosim_bench_cfg_t cfg;
    cacaosim_bench_ctx_t ctx;
    char                 err[256];

    if (cacaosim_bench_load_file("examples/exao_coro_lowfs.bench", &cfg, err, sizeof(err)) != 0 ||
        cacaosim_bench_init(&ctx, &cfg, NULL, 0 /* connect_shm = 0 */, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "cacaosim error: %s\n", err);
        return 1;
    }
    cacaosim_bench_print_summary(&ctx, stdout);

    int    dm_slot   = cacaosim_bench_find_dm(&ctx, "dm0");
    int    wfs_slot  = cacaosim_bench_find_sensor(&ctx, "wfs_ho");
    int    coro_slot = cacaosim_bench_find_sensor(&ctx, "coro_cam");
    float *dm_cmd    = cacaosim_bench_dm_cmd_ptr(&ctx, dm_slot);

    /* Poke central actuator by +0.15 um and advance 20 master ticks (1 full coro_cam exposure) */
    dm_cmd[8 * 16 + 8] = 0.15f;
    for (int tick = 0; tick < 20; tick++)
    {
        cacaosim_bench_step(&ctx, NULL, NULL, 0.0f /* turb_gain */, NULL);
    }

    const uint16_t *wfs_img  = cacaosim_bench_sensor_u16(&ctx, wfs_slot);
    const uint16_t *coro_img = cacaosim_bench_sensor_u16(&ctx, coro_slot);
    printf("SHWFS[0]=%u ADU, CoroCam[center]=%u ADU\n", wfs_img[0], coro_img[64 * 128 + 64]);

    cacaosim_bench_free(&ctx);
    return 0;
}
```

---

## Validation & Measured Runtime Performance

`cacaosim-validate` executes **56** quantitative optical physics, spectral photon-conservation,
CPU-vs-CUDA numerical parity (`rel_L2 < 1e-5`), analytic & custom 2D/3D polychromatic coronagraphic
mask suppression, dual-mode LOWFS (`fpm_refl` + `lowfs_pup_diam` & `lyot_refl`), static WFE `[m]`
synthesis, topological hardware splicing, `cacaosim-CTRL` headless TUI snapshots, and multi-stage
closed-loop AO tests (`H-band Strehl: open=0.153 -> Stage-1=0.815 -> 2-Stage=0.998`).
`tests/test_closed_loop.sh` verifies zero `libCLIcore` linkage across all 11 `fpsexec` binaries,
`cacaosim-CTRL`, and `cacaosim-bench-view`, modular 3D beam SHM pipelines, zero-sleep `fpspipe`
`fused`/`modular` `.bench` deployment (including warm `FPFLAG_WRITERUN` preservation, `--restart`,
`>80`-char `.fits.gz` 3D WCS masks, and `teardown`), interactive `cacaosim-CTRL` PTY workflows, and
`milk-cli` / `milk-script` integration.

Measured with `cacaosim-benchmark` on Intel Core i9-12950HX + NVIDIA RTX A5500 Laptop GPU:

| Compute Pipeline | CPU p50 (us) | CUDA p50 (us) | CUDA p99 (us) | CUDA Rate |
| :--- | ---: | ---: | ---: | ---: |
| `dmsim` $20\times 20 \to 128\times 128$ | 40.6 | **27.7** | 29.7 | **36.29 kHz** |
| `dmsim` $50\times 50 \to 256\times 256$ | 90.5 | **65.6** | 70.2 | **15.21 kHz** |
| `pyrwfs` $128\to 64$ (`nmod=1`, unmod) | 1479.5 | **45.1** | 50.0 | **21.95 kHz** |
| `pyrwfs` $128\to 64$ (`nmod=16`, Graph) | 3377.5 | **186.1** | 191.2 | **5.37 kHz** |
| `shwfs` $128\to 128$ ($16\times 16$ lenslets) | 113.6 | **35.9** | 42.4 | **27.48 kHz** |
| `zwfs` $128\to 64$ ($32\times 32$ Babinet MFT) | 350.0 | **66.7** | 71.6 | **14.88 kHz** |
| `coro` $128\to 128$ (Lyot MFT + LOWFS) | 499.9 | **45.4** | 46.1 | **29.90 kHz** |
| `coro` $128\to 128$ (Vortex $4\times$ + LOWFS) | 8882.7 | **38.4** | 39.4 | **35.10 kHz** |
| `fpcam` $128\to 128$ (MFT, 5-$\lambda$) | 2037.8 | **118.5** | 135.7 | **8.22 kHz** |
| **Fused Single-Stage** ($20^2$ DM + PyWFS + FP) | 6476.4 | **238.9** | 326.8 | **4.06 kHz** |
| **2-Stage DAG** (`two_stage_ao.bench`) | 15572.1 | **429.2** | 497.9 | **2.30 kHz** |
| **ExAO Coro+LOWFS DAG** (`exao_coro_lowfs.bench`) | 1986.2 | **182.2** | 246.9 | **5.32 kHz** |
