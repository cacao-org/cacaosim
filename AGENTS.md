# cacaosim Developer & Agent Guidelines (AGENTS.md)

`cacaosim` (`libmilkcacaosim.so`) is a high-throughput (HT) modular multi-stage Adaptive Optics
hardware and optical simulation plugin for the `milk` framework (`feat/bash-engine` branch) and
`cacao`.

---

## 1. Architecture & Compute Units

- **`src/common/`**: Aligned memory, monotonic `TIMER_ABSTIME` pacing, thread-local `xoshiro128++`
  CPU PRNG, Philox-style CUDA PRNG, 32-bin (`0.40..2.40 um`) spectral photometry & dichroic
  beamsplitters, 2D affine pupil registration, SHM beam keyword serialization, 2D/3D polychromatic
  `.fits`/`.fits.gz`/`.fits.fz`/SHM WCS mask cube I/O (`cacaosim_mask`), and detector readout.
- **`src/pupil/`**: Anti-aliased telescope pupil and static NCPA generator (`cacaosim.mkpupil`,
  `milk-fpsexec-cacaosim-mkpupil`).
- **`src/beam/`**: Optical beam `[OPD_um, Amplitude]` source (`cacaosim.mkbeam`,
  `milk-fpsexec-cacaosim-mkbeam`), inline static wavefront error `[meter]` OPD element
  (`cacaosim.wfe`, `milk-fpsexec-cacaosim-wfe`, `[wfe.<id>]`), and gray / LP / SP / band-reflect
  dichroic beamsplitter with per-arm NCPA, pupil/Lyot masks, and 2D registration
  (`cacaosim.beamsplit`, `milk-fpsexec-cacaosim-beamsplit`).
- **`src/dm/`**: DM & 2-axis Tip-Tilt mirror (`dm_mode = 1`) hardware simulator (`cacaosim.dmsim`,
  `milk-fpsexec-cacaosim-dmsim`) with fractional-frame delay, 1st/2nd-order exact discrete ZOH
  dynamics, DAC quantization, stroke saturation, 2D registration, and in-line `in_beam -> out_beam`.
- **`src/coro/`**: Coronagraphic focal-plane & pupil-plane unit (`cacaosim.coro`,
  `milk-fpsexec-cacaosim-coro`) supporting analytic (Lyot spot, phase dimple, Optical Vortex, 4QPM)
  and custom FITS/SHM amplitude and/or phase masks in pupil, focal (`custom_mft`, `custom_fft`),
  and Lyot planes, including 3D `[Nx, Ny, Nlambda]` polychromatic cubes and dual `out_lowfs`
  pickoff modes (`fpm_refl` with `lowfs_pup_diam` aperture stop, or `lyot_refl` reflective Lyot).
- **`src/pyrwfs/`**: 4-sided Pyramid WFS (`cacaosim.pyrwfs`, `milk-fpsexec-cacaosim-pyrwfs`) using
  precomputed shifted focal masks, `N_mod + 1` FFT reduction, multi-subframe `navg`, and CUDA
  Graphs.
- **`src/shwfs/`**: Diffractive Shack-Hartmann lenslet WFS (`cacaosim.shwfs`,
  `milk-fpsexec-cacaosim-shwfs`) using OpenMP L1 FFTW3f and batched `cufftPlanMany`.
- **`src/zwfs/`**: Babinet Matrix-DFT Zernike phase-contrast WFS (`cacaosim.zwfs`,
  `milk-fpsexec-cacaosim-zwfs`) using `cblas_cgemm` / `cublasCgemm`.
- **`src/fpcam/`**: Focal-plane science camera (`cacaosim.fpcam`, `milk-fpsexec-cacaosim-fpcam`)
  supporting Soummer Matrix-DFT and padded FFT across polychromatic channels, dynamic `in_beam`
  amplitude/OPD, subframe integration, and Strehl telemetry.
- **`src/bench/`, `src/fused/` & `scripts/commands.d/cacaosim.bash`**: Declarative `.bench`
  INI/TOML parser/serializer, presets & topological optical-splicing editor, zero-FFT spectral
  analyzer, C `fpspipe` deployment-plan compiler (`cacaosim_bench_plan.c`, `cacaosim-bench-view
  --emit-plan`), `milk-CTRL`-style flicker-free hardware configurator & live SHM status GUI
  (`cacaosim-CTRL`, `cacaosim-bench gui`), one-shot CLI schematic visualizer
  (`cacaosim-bench-view`), multi-rate optical bench engine (`cacaosim.htsim`,
  `milk-fpsexec-cacaosim-htsim`), and `milk-script` / `fpspipe` lifecycle orchestration
  (`cacaosim.deploy`, `cacaosim.stop`, `cacaosim.status`, `cacaosim.teardown`,
  `scripts/cacaosim-bench`).

---

## 2. Coding & Performance Rules

1. **Template Style**: Follow `milk_module_example` (`milk_runnable.h`, `MILK_PARAMS_DEFINE`,
   `MILK_MODULE_RUNNABLES`, `EXPORT_FPSEXEC(milk, ...)`). Never include `CLIcore.h` directly and
   never link `CLIcore` into `fpsexec` executables.
2. **Formatting**: Allman braces, `<= 100` characters per line across all `.c`, `.h`, `.cu`, `.md`,
   `.bench`, and script files, explicit header includes, and Kernel-Doc comments.
3. **Hot-Loop Invariants**:
   - Zero `malloc()` / `calloc()` / `free()` or FFT plan creation inside per-frame loops.
   - Single-precision float math (`sqrtf`, `sinf`, `cosf`, `expf`, `0.5f`) in float kernels.
   - Use `MILK_HOT`, `MILK_COLD`, `restrict`, `MILK_ASSUME_ALIGNED`, and `if (parallel: ...)` on
     OpenMP `parallel for simd` directives.
   - Set `outimg.md->write = 1` before modifying stream pixels and finalize with
     `processinfo_update_output_stream()`.

---

## 3. Build & Verification

```bash
cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release -DCACAOSIM_ENABLE_CUDA=ON
cmake --build _build -j$(nproc)
cmake --install _build
ctest --test-dir _build --output-on-failure
CACAOSIM_SRC_DIR=$PWD ./_build/cacaosim-benchmark
```
