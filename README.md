# cacaosim — Modular Multi-Stage Adaptive Optics Hardware & Optics Simulator for milk / cacao

`cacaosim` (`libmilkcacaosim.so`) is a high-performance C17 / OpenMP / CUDA plugin for `milk`
(`feat/bash-engine`) designed as a real-time Hardware-in-the-Loop (HIL) adaptive optics bench
simulator for `cacao`. It natively consumes live 2D phase and amplitude streams (`outarraypha`,
`outarrayamp`) produced by `milkatmturb` (`AtmosphericTurbulenceSimul`) and supports arbitrary
**multi-DM, multi-WFS, multi-rate, and coronagraphic optical bench topologies** via both:

1. **Stream-composable `MILK_RUNNABLE` FPS processes** connected by `[nwf, nwf, 2]` `FLOAT32`
   optical beam streams (`[OPD_um, Amplitude]` + 32-bin photon spectrum keywords), and
2. **Declarative `.bench` DAG files** executable either **fused inside a single GPU/CPU process**
   (`cacaosim.htsim .bench_file=...`) or **modularly across separate FPS processes**
   (`cacaosim-bench start --config ... --mode fused|modular`).

---

## Compute Units

All 10 compute units use the `MILK_RUNNABLE` / `EXPORT_FPSEXEC` architecture on `milk`'s
`feat/bash-engine` branch with zero `libCLIcore` dependency in standalone `milk-fpsexec-*` binaries:

| Command | `fpsexec` Binary | Role & Optical Ports |
| :--- | :--- | :--- |
| `cacaosim.mkpupil` | `milk-fpsexec-cacaosim-mkpupil` | Pupil + NCPA (`-> csim_pup, csim_ncpa`) |
| `cacaosim.mkbeam` | `milk-fpsexec-cacaosim-mkbeam` | `[OPD, Amp]` beam + 32-bin Vega spectrum |
| `cacaosim.dmsim` | `milk-fpsexec-cacaosim-dmsim` | Zonal DM / 2-axis TTM (`in_beam->out_beam`) |
| `cacaosim.beamsplit` | `milk-fpsexec-cacaosim-beamsplit` | Gray / LP / SP / BP dichroic splitter |
| `cacaosim.coro` | `milk-fpsexec-cacaosim-coro` | Lyot, phase, Vortex, 4QPM + `out_lowfs` |
| `cacaosim.pyrwfs` | `milk-fpsexec-cacaosim-pyrwfs` | 4-sided Pyramid WFS (`navg`, CUDA Graph) |
| `cacaosim.shwfs` | `milk-fpsexec-cacaosim-shwfs` | Diffractive lenslet Shack-Hartmann WFS |
| `cacaosim.zwfs` | `milk-fpsexec-cacaosim-zwfs` | Babinet MFT Zernike phase-contrast WFS |
| `cacaosim.fpcam` | `milk-fpsexec-cacaosim-fpcam` | Polychromatic MFT/FFT PSF + Strehl camera |
| `cacaosim.htsim` | `milk-fpsexec-cacaosim-htsim` | Fused zero-hop flat or `.bench` DAG runner |

---

## Multi-Stage Optical Bench Architecture (`.bench`)

### 1. Ordered DMs, Dichroics, Registration & Spectral Photometry

- **Optical Beams (`[nwf, nwf, 2]` `FLOAT32`)**: Slice `0` stores wavefront `OPD [um]`, slice `1`
  stores pupil `Amplitude`, and `CSIM_S00..CSIM_S31` track photon arrival rates `[photons/s]` across
  32 wavelength bins spanning `0.40..2.40 um`.
- **Sequential DM / TTM Deployment**: Each `[dm.<id>]` or `[ttm.<id>]` subtracts its reflective
  surface OPD (`dm_opd = refl_scale * surface`) from `in` to produce `out`, enforcing exact upstream
  vs downstream optical causality (`Stage 1 Woofer -> Dichroic 1 -> Stage 2 Tweeter -> Dichroic 2`).
- **Photon-Conserving Splitters**: `[splitter.<id>]` supports `"gray"`, `"dichroic_lp"`,
  `"dichroic_sp"`, and `"dichroic_bp"` (band-reflect/notch, e.g. reflecting J-band `1.22 um` to a
  Stage-1 WFS while transmitting both Visible `< 1.0 um` and H-band `1.65 um` downstream), plus
  per-arm static NCPA, pupil/Lyot masks, and 2D subpixel `(shift_x, shift_y, rot_deg, scale)`
  registration.
- **Multi-Rate Subframe Integration**: The bench ticks at `master_hz` (e.g. `3000 Hz`) while each
  WFS or camera integrates `navg` (or `rate_hz`) subframes before detector readout + Poisson/RON
  noise quantization.

### 2. Example: 2-Stage Cascaded AO (`examples/two_stage_ao.bench`)

- **Stage 1**: `12x12` Woofer DM (`dm01disp`) -> J-band (`1.22 um`) notch dichroic -> Stage-1 NIR
  Pyramid WFS (`aol1_wfsim`) at `1 kHz` (`navg = 3`).
- **Stage 2**: Transmitted Visible + H-band -> `24x24` Tweeter DM (`dm02disp`) -> `1.00 um`
  long-pass dichroic -> Stage-2 Visible (`0.75 um`) Pyramid WFS (`aol2_wfsim`) at `3 kHz`
  (`navg = 1`) + H-band (`1.65 um`) Science Camera (`csim_fpcam`) at `100 Hz` (`navg = 30`).

### 3. Edit, Visualize & Deploy Workflow (`cacaosim-bench-view`)

`cacaosim-bench-view` renders any `.bench` text file in `< 1 ms` using `milk`'s native TrueColor /
256-color / 16-color / monochrome ANSI theme engine (`dark`, `night` Observatory Red, `accessible`
Okabe-Ito, `light`, `nordic`, plus `--ascii`), displaying an anti-aliased 2D pupil miniature,
wavelength-colored 32-bin (`0.40..2.40 um`) spectral bars on every split arm, live `MILK_SHM_DIR`
stream indicators, and interactive edit/deploy/stop TUI controls:

```bash
export MILK_SHM_DIR=/milk/shm

# 1. Bootstrap or edit a .bench configuration file and visualize immediately:
cacaosim-bench edit my_two_stage.bench
cacaosim-bench-view my_two_stage.bench                 # one-shot milk TrueColor schematic
cacaosim-bench-view --theme night exao_coro_lowfs.bench # Observatory Red palette
cacaosim-bench-view -w my_two_stage.bench              # live-reload on every file save

# 2. Or launch the interactive TUI ([e] edit, [f] deploy fused, [m] modular, [s] stop, [t] theme):
cacaosim-bench-view -i my_two_stage.bench

# 3. Deploy and run in zero-hop fused GPU/CPU mode or multi-process modular FPS mode:
cacaosim-bench start --config my_two_stage.bench --mode fused --atmturb --seeing 0.7
cacaosim-bench start --config two_stage_ao.bench --mode modular
cacaosim-bench status
cacaosim-bench stop
```

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

## Validation & Measured Runtime Performance

`cacaosim-validate` executes **40** quantitative optical physics, spectral photon-conservation,
CPU-vs-CUDA numerical parity (`rel_L2 < 1e-5`), coronagraphic suppression, optical causality,
visualizer rendering, and multi-stage closed-loop AO tests
(`H-band Strehl: open=0.153 -> Stage-1=0.815 -> 2-Stage=0.998`).
`tests/test_closed_loop.sh` verifies zero `libCLIcore` linkage across all 10 `fpsexec` binaries and
`cacaosim-bench-view`, modular 3D beam SHM pipelines, fused/modular `.bench` runs, edit-visualize-
deploy workflows, and `milk-cli` loading.

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
