# cacaosim Complete Options & Parameters Reference

> **Auto-generated from `cacaosim_bench_schema.c` and runtime `MILK_RUNNABLE` introspection.**
> Regenerate at any time with: `cacaosim-bench options --format markdown --output docs/OPTIONS.md`

Summary: **165** `.bench` option definitions · **12** FPS compute units (**371** FPS parameters) · **61** CLI flags.

## 1. Declarative `.bench` Configuration Options

### `[bench]` — Global Telescope, Pupil, Star & Master Clock

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `name` | — | `string` | `ao_bench` | — | `fused+modular` | Basic | Bench configuration identifier shown in headers and reports |
| `nwf` | — | `uint` | `128` | px | `fused+modular` | Basic | Square pupil-plane wavefront grid size shared across all beams |
| `pup_diam` | — | `float` | `0.9375` | nwf | `fused+modular` | Basic | Clear telescope pupil outer diameter as fraction of nwf |
| `cobs` | `obs_frac` | `float` | `0.30` | D | `fused+modular` | Basic | Central secondary obstruction diameter ratio D_sec/D_tel |
| `nspider` | `n_spiders` | `int` | `4` | — | `fused+modular` | Basic | Number of secondary-mirror support spider vanes |
| `spider_w` | — | `float` | `0.012` | D | `fused+modular` | Basic | Spider vane thickness as fraction of clear pupil diameter |
| `spider_deg` | — | `float` | `45.0` | deg | `fused+modular` | Basic | Azimuth angle of first spider vane CCW from +X |
| `spider_offset` | — | `float` | `0.0` | D | `fused+modular` | Basic | V-spider X-vertex offset fraction of pupil diameter (0 = radial; Subaru = 0.08289) |
| `pupil_geom` | — | `preset` | `default` | — | `fused+modular` | Advanced | Preset macro setting tel_diam_m, cobs, spiders & spider_offset (e.g. 'subaru') Values: `subaru\|radial\|default` |
| `tel_diam_m` | — | `float` | `8.2` | m | `fused+modular` | Basic | Primary mirror outer diameter used for photon collecting |
| `star_mag` | — | `float` | `8.0` | mV | `fused+modular` | Basic | Guide/science star apparent Vega V-band magnitude |
| `star_teff` | — | `float` | `5778.0` | K | `fused+modular` | Basic | Stellar blackbody temperature shaping 0.4-2.4 um |
| `tput_tel` | `tel_tput` | `float` | `0.75` | — | `fused+modular` | Basic | Combined sky + telescope + fore-optics transmission |
| `master_hz` | `frame_rate` | `float` | `2000.0` | Hz | `fused+modular` | Basic | Physical bench master hardware clock rate |
| `oversample` | — | `uint` | `1` | x | `fused+modular` | Advanced | Internal temporal oversampling factor K (dt_internal = 1 / (K * master_hz)) |
| `atm_interp` | — | `bool` | `1` | — | `fused-only` | Advanced | Linearly interpolate dual atmospheric screens across oversampled substeps Values: `0\|1` |
| `telemetry_stride` | — | `uint` | `0` | ticks | `fused-only` | Advanced | Decimation stride for optional diagnostic beam exports (0 = every tick) |
| `in_wf` | — | `string` | `outarraypha` | shm | `fused+modular` | Basic | Input 2D atmospheric turbulence phase/OPD SHM stream |
| `in_amp` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional 2D atmospheric scintillation amplitude SHM stream |
| `in_pup` | — | `string` | `csim_pup` | shm | `fused+modular` | Basic | Input entrance pupil amplitude SHM stream (or FITS path) |
| `in_ncpa` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional common-path static NCPA OPD [um] SHM stream |
| `out_beam` | `source_beam` | `string` | `b_pup` | beam | `fused+modular` | Basic | Root optical beam name emitted from entrance pupil |
| `wf_unit` | — | `enum` | `rad` | — | `fused+modular` | Basic | Physical unit of in_wf stream: 'um' (OPD microns) or 'rad' (phase at ref_lambda) Values: `um\|rad` |
| `ref_lambda` | — | `float` | `0.50` | um | `fused+modular` | Basic | Reference wavelength when wf_unit == 'rad' |
| `turb_gain` | — | `float` | `1.0` | x | `fused+modular` | Basic | Dimensionless multiplier scaling input atmospheric OPD |
| `backend` | — | `enum` | `auto` | — | `fused+modular` | Basic | Compute execution engine selection Values: `auto\|cpu\|cuda` |

### Common Node Keys — Shared Optical Wiring & 2D Registration

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `in` | `in_beam` | `string` | `(required)` | beam | `fused+modular` | Basic | Upstream optical beam name consumed by this optical element |
| `out` | `out_beam` | `string` | `(required)` | beam | `fused+modular` | Basic | Primary downstream optical beam produced by inline elements (DM, TTM, WFE, Coro) |
| `shift_x` | — | `float` | `0.0` | px | `fused+modular` | Basic | Lateral X pupil misregistration in wavefront pixels (coro: fused-only) |
| `shift_y` | — | `float` | `0.0` | px | `fused+modular` | Basic | Lateral Y pupil misregistration in wavefront pixels (coro: fused-only) |
| `rot_deg` | — | `float` | `0.0` | deg | `fused+modular` | Basic | In-plane pupil clocking rotation angle (coro: fused-only) |
| `mag_scale` | — | `float` | `1.0` | x | `fused+modular` | Basic | Radial pupil magnification scaling factor (coro: fused-only) |

### `[dm.<id>]` / `[ttm.<id>]` — Deformable Mirror & Tip-Tilt Mirror

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `in_cmd` | — | `string` | `<id>_cmd` | shm | `fused+modular` | Basic | Input 2D float32 actuator command SHM stream [um] |
| `out_opd` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional output 2D DM surface OPD telemetry SHM stream [um] |
| `in_flat` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional static flat-map actuator offset SHM stream |
| `dm_mode` | — | `enum` | `0 (ttm: 1)` | — | `fused+modular` | Basic | Mirror geometry mode: 0 = zonal actuator grid ([dm.*]), 1 = 2-axis tip-tilt ([ttm.*]) Values: `0\|1` |
| `nact_x` | `nact` | `uint` | `16 (ttm: 2)` | act | `fused+modular` | Basic | Actuator grid count along X ('nact' alias sets both nact_x and nact_y) |
| `nact_y` | — | `uint` | `16 (ttm: 1)` | act | `fused+modular` | Basic | Actuator grid count along Y |
| `dm_diam` | — | `float` | `0.0` | act | `fused+modular` | Advanced | Illuminated pupil diameter in actuator pitches (0 = auto nact_x - 2) |
| `if_coupling` | — | `float` | `0.20` | — | `fused+modular` | Basic | Nearest-neighbor Gaussian influence function coupling |
| `tput` | — | `float` | `0.98` | — | `fused+modular` | Basic | Mirror optical intensity reflectivity |
| `refl_scale` | — | `float` | `2.0` | x | `fused+modular` | Basic | OPD per surface displacement (2.0 = normal incidence) |
| `max_stroke` | — | `float` | `5.0` | um | `fused+modular` | Basic | Symmetric mechanical actuator stroke clip [+/- um] |
| `dac_bits` | — | `int` | `16` | bits | `fused+modular` | Basic | Drive electronics DAC quantization bits (0 = ideal float) |
| `nonlin_quad` | — | `float` | `0.0` | — | `fused+modular` | Basic | Quadratic electrostrictive/piezo actuator nonlinearity |
| `dyn_mode` | — | `enum` | `rc` | — | `fused+modular` | Basic | Discrete ZOH servo model: instant (0), 1st-order RC lag (1), or 2nd-order oscillator (2) Values: `instant\|rc\|osc` |
| `tau_us` | `tau_sec` | `float` | `120.0` | us | `fused+modular` | Basic | 1st-order RC settling time constant [us] (alias 'tau_sec' takes SI seconds) |
| `f0_hz` | — | `float` | `2500.0` | Hz | `fused+modular` | Basic | 2nd-order damped oscillator natural resonance frequency |
| `zeta` | — | `float` | `0.70` | — | `fused+modular` | Basic | 2nd-order oscillator damping ratio (0.7 = critically damped) |
| `delay_frames` | — | `float` | `1.0` | fr | `fused+modular` | Basic | Fractional master-clock latency [frames] (active when delay_us == 0) |
| `delay_us` | `delay_sec` | `float` | `0.0` | us | `fused+modular` | Advanced | Physical command transport delay [us] overriding delay_frames ('delay_sec' takes seconds) |
| `zoh_mean` | — | `bool` | `1` | — | `fused-only` | Advanced | Integrate sub-tick ZOH exposure average across actuator transition (1 = ON) Values: `0\|1` |

### `[splitter.<id>]` — Gray & Dichroic Beamsplitter

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `out_refl` | — | `string` | `""` | beam | `fused+modular` | Basic | Reflected output arm optical beam name |
| `out_trans` | — | `string` | `""` | beam | `fused+modular` | Basic | Transmitted output arm optical beam name |
| `mode` | `split_mode` | `enum` | `dichroic_lp` | — | `fused+modular` | Basic | Spectral split law: achromatic gray, long-pass, short-pass, or band-reflect notch Values: `gray\|dichroic_lp\|dichroic_sp\|dichroic_bp` |
| `r_gray` | — | `float` | `0.50` | — | `fused+modular` | Basic | Achromatic reflected power fraction (mode == 'gray') |
| `t_gray` | — | `float` | `0.50` | — | `fused+modular` | Basic | Achromatic transmitted power fraction (mode == 'gray') |
| `lambda_cut` | — | `float` | `0.95` | um | `fused+modular` | Basic | Dichroic 50% transition cutoff or notch center |
| `band_width` | — | `float` | `0.25` | um | `fused+modular` | Basic | Reflected notch full width (dichroic_bp mode) |
| `roll_um` | — | `float` | `0.03` | um | `fused+modular` | Basic | Dichroic logistic transition edge roll-off width |
| `eff_refl` | — | `float` | `0.95` | — | `fused+modular` | Basic | Peak coating efficiency in reflected passband |
| `eff_trans` | — | `float` | `0.95` | — | `fused+modular` | Basic | Peak coating efficiency in transmitted passband |
| `ncpa_refl` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional static NCPA OPD [um] SHM map on reflected arm |
| `ncpa_trans` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional static NCPA OPD [um] SHM map on trans arm |
| `pup_refl` | — | `string` | `""` | shm | `fused+modular` | Advanced | Optional pupil amplitude mask SHM on reflected arm |
| `pup_trans` | — | `string` | `""` | shm | `fused+modular` | Advanced | Optional pupil amplitude mask SHM on transmitted arm |
| `refl_dx` | — | `float` | `0.0` | px | `fused+modular` | Advanced | Reflected-arm lateral X pupil shift [px] |
| `refl_dy` | — | `float` | `0.0` | px | `fused+modular` | Advanced | Reflected-arm lateral Y pupil shift [px] |
| `refl_rot` | — | `float` | `0.0` | deg | `fused+modular` | Advanced | Reflected-arm pupil clocking |
| `refl_scale` | — | `float` | `1.0` | x | `fused+modular` | Advanced | Reflected-arm radial pupil magnification factor |
| `trans_dx` | — | `float` | `0.0` | px | `fused+modular` | Advanced | Transmitted-arm lateral X pupil shift [px] |
| `trans_dy` | — | `float` | `0.0` | px | `fused+modular` | Advanced | Transmitted-arm lateral Y pupil shift [px] |
| `trans_rot` | — | `float` | `0.0` | deg | `fused+modular` | Advanced | Transmitted-arm pupil clocking |
| `trans_scale` | — | `float` | `1.0` | x | `fused+modular` | Advanced | Transmitted-arm radial pupil magnification factor |

### `[coro.<id>]` — Coronagraphic Masks, Lyot Stop & LOWFS Pickoff

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `out_lowfs` | — | `string` | `""` | beam | `fused+modular` | Basic | Optional reflected LOWFS pupil pickoff optical beam name |
| `out_pre_lyot` | — | `string` | `""` | shm | `fused+modular` | Advanced | Optional pre-Lyot pupil intensity telemetry SHM stream |
| `out_pre_lyot_long` | — | `string` | `""` | shm | `fused+modular` | Advanced | Optional cumulative pre-Lyot pupil average SHM |
| `in_pup_amp` | — | `string` | `""` | mask | `fused+modular` | Basic | Custom 2D/3D pre-FPM pupil apodizer amplitude (FITS/SHM) |
| `in_pup_pha` | — | `string` | `""` | mask | `fused+modular` | Basic | Custom 2D/3D pre-FPM pupil phase/OPD mask (FITS/SHM) |
| `in_fpm_amp` | — | `string` | `""` | mask | `fused+modular` | Basic | Custom 2D/3D focal-plane mask amplitude (FITS/SHM) |
| `in_fpm_pha` | — | `string` | `""` | mask | `fused+modular` | Basic | Custom 2D/3D focal-plane phase/OPD mask (FITS/SHM) |
| `in_lyot` | `in_lyot_amp` | `string` | `""` | mask | `fused+modular` | Basic | Custom 2D/3D Lyot-stop amplitude mask (FITS/SHM) |
| `in_lyot_pha` | — | `string` | `""` | mask | `fused+modular` | Basic | Custom 2D/3D Lyot-stop phase/OPD mask (FITS/SHM) |
| `mode` | `coro_mode` | `enum` | `lyot_spot` | — | `fused+modular` | Basic | Focal-plane mask architecture and propagator Values: `lyot_spot\|phase_dimple\|vortex\|4qpm\|custom_mft\|custom_fft` |
| `lambda_um` | — | `float` | `1.65` | um | `fused+modular` | Basic | Coronagraph design reference wavelength |
| `bw_frac` | `dlambda_um` | `float` | `0.0` | — | `fused+modular` | Basic | Polychromatic fractional bandwidth dLambda/Lambda ('dlambda_um' takes microns) |
| `nlambda` | — | `int` | `0` | ch | `fused+modular` | Basic | Polychromatic cube slices (0 = auto from bw_frac / 3D FITS) |
| `pha_unit` | `mask_pha_unit` | `enum` | `rad` | — | `fused+modular` | Basic | Physical unit of custom phase mask maps: 'rad' (radians) or 'um' (OPD microns) Values: `rad\|um` |
| `fpm_rad_ld` | — | `float` | `3.0` | l/D | `fused+modular` | Basic | Analytic Lyot spot or phase dimple FPM radius [lambda/D] |
| `fpm_amp` | — | `float` | `0.0` | — | `fused+modular` | Basic | Analytic FPM inner core field amplitude transmission |
| `fpm_phase` | — | `float` | `3.14159` | rad | `fused+modular` | Basic | Phase shift inside phase_dimple (Roddier) FPM [rad] |
| `n_mft` | — | `uint` | `48` | px | `fused+modular` | Advanced | Babinet Matrix-DFT sampling grid across analytic FPM spot |
| `fpm_pixscale` | `fpm_pixscale_ld` | `float` | `8.0` | px/(l/D) | `fused+modular` | Basic | Custom MFT focal mask plate scale [pixels per lambda_0/D] |
| `vortex_charge` | — | `int` | `2` | — | `fused+modular` | Basic | Vortex coronagraph topological charge l (even int) |
| `pad_factor` | — | `uint` | `4` | x | `fused+modular` | Basic | Zero-padded 2D FFT oversampling for vortex/4qpm/custom_fft |
| `lowfs_mode` | — | `enum` | `fpm_refl` | — | `fused+modular` | Basic | LOWFS pickoff plane: starlight rejected at FPM ('fpm_refl') or reflective Lyot stop Values: `fpm_refl\|lyot_refl` |
| `lowfs_pup_diam` | `lowfs_pup` | `float` | `1.0` | D_pup | `fused+modular` | Basic | Pupil stop diameter ratio on FPM-reflected LOWFS path |
| `lowfs_refl` | — | `float` | `0.95` | — | `fused+modular` | Basic | Amplitude reflectivity into LOWFS pickoff arm |
| `tput` | — | `float` | `0.92` | — | `fused+modular` | Basic | Coronagraph relay optical intensity throughput |
| `lyot_outer` | — | `float` | `0.85` | D | `fused+modular` | Basic | Lyot stop outer clear diameter ratio vs entrance pupil |
| `lyot_obs` | — | `float` | `0.36` | D | `fused+modular` | Basic | Lyot stop central obstruction diameter ratio |
| `lyot_nspider` | — | `int` | `4` | — | `fused+modular` | Basic | Number of spider vanes masked by Lyot stop |
| `lyot_spider_w` | — | `float` | `0.02` | D | `fused+modular` | Basic | Lyot stop spider mask width fraction |
| `lyot_spider_deg` | — | `float` | `45.0` | deg | `fused+modular` | Basic | Lyot stop spider orientation angle [deg] |
| `lyot_spider_offset` | — | `float` | `0.0` | D | `fused+modular` | Basic | Lyot stop V-spider X-vertex offset fraction of pup_diam |

### `[wfe.<id>]` / `[static_wfe.<id>]` — Inline Static Wavefront Error (`OPD [m]`)

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `in_wfe` | `stream` | `string` | `<id>_wfe` | shm | `fused+modular` | Basic | Input/auto-created 2D static WFE OPD [meter] SHM stream |
| `rms_nm` | `rms_m` | `float` | `50.0` | nm | `fused+modular` | Basic | Target pupil-weighted RMS OPD when synthesizing map ('rms_m' takes meters) |
| `defocus_nm` | `defocus_m` | `float` | `0.0` | nm | `fused+modular` | Basic | Additive smooth Zernike Z4 defocus RMS across pupil ('defocus_m' takes meters) |
| `psd_alpha` | `alpha` | `float` | `1.6` | — | `fused+modular` | Basic | Power-law spatial PSD amplitude exponent f^-alpha |
| `wfe_gain` | `gain` | `float` | `1.0` | x | `fused+modular` | Basic | Dimensionless multiplier on static OPD map (0=bypass) |
| `wfe_seed` | `seed` | `uint` | `(salted)` | — | `fused+modular` | Basic | Deterministic PRNG seed for WFE map synthesis |
| `tput` | — | `float` | `0.98` | — | `fused+modular` | Basic | Optical intensity transmission factor |

### Shared Sensor Keys — Cadence, Timing & Detector Readout (`pyrwfs`, `shwfs`, `zwfs`, `fpcam`)

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `out_cam` | — | `string` | `<id>_cam` | shm | `fused+modular` | Basic | Output 2D camera frame SHM stream name |
| `in_ncpa` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional per-sensor static NCPA OPD [um] SHM map |
| `bw_frac` | `dlambda_um` | `float` | `0.20 (fpcam: 0.18)` | — | `fused+modular` | Basic | Fractional spectral passband dLambda/Lambda ('dlambda_um' takes microns) |
| `nlambda` | — | `int` | `0` | ch | `fused+modular` | Basic | Polychromatic optical slices integrated per subframe (0 = auto) |
| `navg` | — | `uint` | `1` | ticks | `fused+modular` | Basic | Master-clock subframes integrated per camera readout when rate_hz == 0 |
| `rate_hz` | — | `float` | `0.0` | Hz | `fused+modular` | Basic | Target physical camera readout rate [Hz]; if > 0.1 overrides navg |
| `clock_offset_us` | — | `float` | `0.0` | us | `fused-only` | Advanced | Static camera trigger phase offset relative to master clock [us] |
| `clock_drift_ppm` | — | `float` | `0.0` | ppm | `fused-only` | Advanced | Relative camera oscillator clock drift [parts per million] |
| `subframe_stride` | — | `uint` | `0` | ticks | `fused-only` | Advanced | Optical propagation decimation stride across oversampled subframes (0 = auto) |
| `sim_hz` | — | `float` | `0.0` | Hz | `fused-only` | Advanced | Target optical propagation sampling rate [Hz] deriving subframe_stride (0 = off) |
| `qe` | — | `float` | `0.80 (fpcam: 0.85)` | — | `fused+modular` | Basic | Detector in-band quantum efficiency [e-/photon] |
| `tput` | — | `float` | `1.0` | — | `fused-only` | Advanced | Additional sensor-arm optical intensity throughput |
| `use_beam_flux` | — | `bool` | `1` | — | `fused+modular` | Basic | Derive exposure photon budget from upstream 32-bin stellar spectrum (0 = manual flux_ph) Values: `0\|1` |
| `flux_ph` | — | `float` | `1e5 (fpcam: 1e6)` | ph/fr | `fused+modular` | Basic | Manual photon count per exposure when use_beam_flux == 0 |
| `ron_e` | `ron` | `float` | `0.5 (fpcam: 1.0)` | e- | `fused+modular` | Basic | Gaussian detector readout noise [e- RMS/pixel] |
| `dark_e` | — | `float` | `0.0` | e-/fr | `fused+modular` | Advanced | Dark current + sky background [electrons/pixel/frame] |
| `gain_eadu` | — | `float` | `1.0` | e-/ADU | `fused+modular` | Advanced | Detector conversion gain [electrons per ADU] |
| `bias_adu` | — | `float` | `100.0` | ADU | `fused+modular` | Advanced | Electronic bias pedestal added before quantization [ADU] |
| `sat_adu` | — | `float` | `65535.0` | ADU | `fused+modular` | Advanced | Detector saturation ceiling [ADU] |
| `shutter` | — | `bool` | `1` | — | `fused+modular` | Advanced | Optical shutter state (1 = open sky light, 0 = dark/bias frame only) Values: `0\|1` |
| `photon_noise` | — | `bool` | `1` | — | `fused+modular` | Basic | Enable Poisson photon shot noise generation Values: `0\|1` |
| `out_dtype` | — | `enum` | `uint16` | — | `fused+modular` | Basic | Output camera SHM pixel datatype Values: `uint16\|float32` |
| `rng_seed` | — | `uint` | `(salted)` | — | `fused+modular` | Advanced | Per-detector deterministic PRNG seed (default prime-salted by node index) |

### `[pyrwfs.<id>]` — 4-Sided Pyramid Wavefront Sensor

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ncam` | — | `uint` | `64` | px | `fused+modular` | Basic | Square Pyramid WFS detector frame dimension [pixels] |
| `pad_factor` | — | `uint` | `2` | x | `fused+modular` | Advanced | Pupil-to-focal FFT zero-padding oversampling factor |
| `lambda_um` | — | `float` | `0.75` | um | `fused+modular` | Basic | Pyramid WFS central sensing wavelength |
| `mod_rad` | `mod_rad_ld` | `float` | `3.0` | l/D | `fused+modular` | Basic | Circular tip-tilt modulation radius [lambda/D] |
| `mod_npts` | — | `int` | `16` | pts | `fused+modular` | Basic | Discrete sample points around circular modulation ring |
| `mod_mode` | `mod_subframe_mode` | `enum` | `sync_arc` | — | `fused+modular` | Advanced | Subframe modulation partitioning across oversampled/navg integration ticks Values: `full\|sync_arc\|interleaved` |
| `mod_rev` | — | `int` | `1` | rev | `fused+modular` | Advanced | Modulation revolutions completed per camera exposure |
| `mod_phase_deg` | — | `float` | `0.0` | deg | `fused+modular` | Advanced | Initial circular modulation azimuth angle |
| `pup_sep` | — | `float` | `0.50` | ncam | `fused+modular` | Basic | Sub-pupil center separation fraction across ncam |
| `roof_ld` | — | `float` | `0.0` | l/D | `fused+modular` | Basic | Pyramid glass apex roof defect half-width [lambda/D] |
| `fstop_frac` | — | `float` | `0.45` | — | `fused-only` | Advanced | Focal-plane field stop radius fraction preventing sub-pupil aliasing |
| `chrom_sep` | — | `float` | `0.0` | — | `fused-only` | Advanced | Glass dispersion chromatic pupil-separation coefficient d(sep)/d(ln lam) |

### `[shwfs.<id>]` — Diffractive Shack-Hartmann Lenslet WFS

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `n_sub` | — | `uint` | `16` | sub | `fused+modular` | Basic | Linear lenslet subaperture count across pupil |
| `pix_per_sub` | — | `uint` | `8` | px | `fused+modular` | Basic | Detector pixels per lenslet (ncam = n_sub * pix_per_sub) |
| `pad_factor` | — | `uint` | `2` | x | `fused+modular` | Basic | Subaperture Nyquist FFT zero-padding factor |
| `lambda_um` | — | `float` | `0.75` | um | `fused+modular` | Basic | Shack-Hartmann WFS central sensing wavelength |

### `[zwfs.<id>]` — Babinet Matrix-DFT Zernike Phase-Contrast WFS

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ncam` | — | `uint` | `64` | px | `fused+modular` | Basic | Square Zernike WFS detector frame size [pixels] |
| `n_mft` | — | `uint` | `32` | px | `fused+modular` | Basic | Babinet Matrix-DFT focal grid sampling across phase dimple |
| `lambda_um` | — | `float` | `0.75` | um | `fused+modular` | Basic | Zernike WFS central sensing wavelength |
| `dimple_ld` | — | `float` | `1.06` | l/D | `fused+modular` | Basic | Focal phase-contrast dimple diameter [lambda/D] |
| `dimple_rad` | — | `float` | `1.5708` | rad | `fused+modular` | Basic | Focal dimple phase shift [rad] (pi/2 = 1.5708) |
| `dimple_amp` | — | `float` | `1.0` | — | `fused+modular` | Advanced | Focal dimple field amplitude transmission (1.0 = pure phase) |
| `achrom_dimple` | — | `bool` | `0` | — | `fused-only` | Advanced | Keep dimple phase shift wavelength-independent across broadband slices Values: `0\|1` |

### `[fpcam.<id>]` / `[fpwfs.<id>]` — Focal-Plane Science & WFS Camera

| Key | Aliases | Type | Default | Unit | Mode | Tier | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `out_long` | — | `string` | `""` | shm | `fused+modular` | Basic | Optional cumulative long-exposure PSF float32 SHM stream |
| `ncam` | — | `uint` | `128` | px | `fused+modular` | Basic | Square focal-plane camera detector size [pixels] |
| `prop_mode` | — | `enum` | `mft` | — | `fused+modular` | Basic | Propagator: Soummer Matrix-DFT ('mft', arbitrary pixscale) or zero-padded FFT ('fft') Values: `mft\|fft` |
| `pad_factor` | — | `uint` | `2` | x | `fused+modular` | Advanced | Zero-padding factor when prop_mode == 'fft' |
| `lambda_um` | — | `float` | `1.65` | um | `fused+modular` | Basic | Focal-plane filter central wavelength |
| `pixscale_ld` | `pixscale` | `float` | `3.0` | px/(l/D) | `fused+modular` | Basic | Detector sampling [pixels per lambda/D] in MFT mode |
| `is_wfs` | — | `bool` | `0 (fpwfs: 1)` | — | `fused-only` | Advanced | Treat focal-plane camera as an active WFS (kept active during calib_fast & skip_fpcam) Values: `0\|1` |

## 2. Standalone FPS Compute Unit Parameters (`libmilkcacaosim.so`)

### `mkpupil` (`milk-fpsexec-cacaosim-mkpupil` · FPS `csim_mkpupil`)

Generate anti-aliased telescope pupil and static NCPA streams

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.out_pup` | `stream` | `csim_pup` | `init` | Output 2D pupil amplitude stream name |
| `.nwf` | `uint32` | `128` | `init` | Square pupil grid size [pixels] (default: 128) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Outer pupil diameter as fraction of nwf (default: 0.9375) |
| `.cobs` | `float32` | `0.3` | `init` | Central obstruction fraction of pup_diam (default: 0.30) |
| `.nspider` | `int32` | `4` | `init` | Number of radial spider vanes (default: 4) |
| `.spider_w` | `float32` | `0.01` | `init` | Spider vane width as fraction of pup_diam (default: 0.01) |
| `.spider_deg` | `float32` | `45` | `init` | Spider vane azimuth offset [deg] (default: 45) |
| `.spider_offset` | `float32` | `0` | `init` | V-spider X-vertex offset fraction of pup_diam (default: 0.0 = radial) |
| `.out_ncpa` | `string` | `""` | `init` | Optional output static NCPA OPD stream [um] (empty = skip) |
| `.ncpa_rms` | `float32` | `0` | `init` | Static NCPA RMS OPD [um] (default: 0.0) |
| `.ncpa_seed` | `uint64` | `42` | `init` | Random seed for NCPA Zernike coefficients |

### `mkbeam` (`milk-fpsexec-cacaosim-mkbeam` · FPS `csim_mkbeam`)

Combine telescope pupil, atmosphere, and star spectrum into beam

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.out_beam` | `stream` | `csim_beam0` | `init` | Output 3D [nwf x nwf x 2] optical beam stream (slice0=OPD_um, slice1=Amp) |
| `.in_wf` | `stream` | `outarraypha` | `init` | Optional input atmospheric WF stream (e.g. outarraypha) |
| `.in_amp` | `string` | `""` | `init` | Optional input atmospheric amplitude stream (e.g. outarrayamp) |
| `.in_pup` | `string` | `csim_pup` | `init` | Input 2D pupil amplitude stream (auto-generated if absent) |
| `.in_ncpa` | `string` | `""` | `init` | Optional static telescope NCPA OPD stream [um] |
| `.nwf` | `uint32` | `128` | `init` | Pupil grid size [pixels] (default: 128) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Clear pupil diameter fraction of nwf (default: 0.9375) |
| `.obs_frac` | `float32` | `0.3` | `init` | Central obscuration linear fraction (default: 0.30) |
| `.n_spiders` | `int32` | `4` | `init` | Spider count when auto-generating pupil (default: 4) |
| `.spider_w` | `float32` | `0.012` | `init` | Spider width fraction of pupil diameter (default: 0.012) |
| `.spider_deg` | `float32` | `45` | `init` | Spider orientation angle [deg] when auto-generating pupil (default: 45) |
| `.spider_offset` | `float32` | `0` | `init` | V-spider X-vertex offset fraction of pup_diam (default: 0.0) |
| `.tel_diam_m` | `float32` | `8.2` | `init` | Primary mirror diameter [m] for photon budget (default: 8.2) |
| `.star_mag` | `float32` | `8` | `live` | Guide/target star V-band Vega magnitude (default: 8.0) |
| `.star_teff` | `float32` | `5778` | `live` | Stellar blackbody temperature [K] (default: 5778) |
| `.tel_tput` | `float32` | `0.75` | `live` | Atmosphere + telescope throughput [0..1] (default: 0.75) |
| `.wf_unit` | `int32` | `1` | `init` | Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1) |
| `.ref_lambda` | `float32` | `0.5` | `init` | Reference wavelength [um] for phase input (default: 0.50) |
| `.turb_gain` | `float32` | `1` | `live` | Live atmospheric turbulence multiplier (0 = flat WF) |
| `.frame_rate` | `float32` | `0` | `live` | Internal TIMER_ABSTIME pacing rate [Hz] (0 = use trigger) |

### `wfe` (`milk-fpsexec-cacaosim-wfe` · FPS `csim_wfe`)

Inline static wavefront error optical element specified in OPD [m]

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.in_beam` | `stream` | `csim_beam0` | `init` | Input 3D [nwf x nwf x 2*N] optical beam stream |
| `.out_beam` | `stream` | `csim_beam_wfe` | `init` | Output 3D [nwf x nwf x 2*N] optical beam stream |
| `.in_wfe` | `stream` | `csim_wfe` | `init` | 2D static wavefront error OPD stream in unit of [meter] |
| `.in_pup` | `string` | `""` | `init` | Optional 2D pupil stream for initial WFE RMS normalization |
| `.nwf` | `uint32` | `128` | `init` | Grid size [pixels] when creating .in_wfe / fallback beam (default: 128) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Pupil diameter fraction of nwf for initial WFE map (default: 0.9375) |
| `.rms_nm` | `float32` | `50` | `init` | Initial WFE pupil RMS [nm] (50 nm = 5e-8 m) when creating .in_wfe |
| `.defocus_nm` | `float32` | `0` | `init` | Additive smooth Zernike Z4 defocus RMS [nm] across full grid (default: 0.0) |
| `.psd_alpha` | `float32` | `1.6` | `init` | Spatial PSD amplitude slope k^-alpha for initial WFE map (default: 1.6) |
| `.wfe_seed` | `uint64` | `101` | `init` | Deterministic PRNG seed for initial WFE map synthesis (default: 101) |
| `.wfe_gain` | `float32` | `1` | `live` | Dimensionless multiplier applied to .in_wfe [meter] (default: 1.0) |
| `.tput` | `float32` | `0.98` | `live` | Optical power throughput [0..1] (default: 0.98) |
| `.shift_x` | `float32` | `0` | `live` | Pupil lateral X shift [pixels] (default: 0.0) |
| `.shift_y` | `float32` | `0` | `live` | Pupil lateral Y shift [pixels] (default: 0.0) |
| `.rot_deg` | `float32` | `0` | `live` | Pupil rotation [deg] (default: 0.0) |
| `.mag_scale` | `float32` | `1` | `live` | Pupil magnification factor (default: 1.0) |
| `.init_only` | `onoff` | `OFF` | `init` | ON: create & initialize .in_wfe [meter] stream map and exit immediately |

### `dmsim` (`milk-fpsexec-cacaosim-dmsim` · FPS `csim_dmsim`)

Simulate DM / Tip-Tilt electronics, dynamics, OPD, & in-line beam

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.in_cmd` | `stream` | `dm00disp` | `init` | Input actuator command stream (2D zonal grid or 2x1 Tip-Tilt) |
| `.out_opd` | `stream` | `csim_dmopd` | `init` | Output 2D reflective DM OPD stream [um] |
| `.in_beam` | `string` | `""` | `init` | Optional upstream 3D [nwf x nwf x 2] optical beam stream |
| `.out_beam` | `string` | `""` | `init` | Optional downstream 3D [nwf x nwf x 2] optical beam stream (in_beam - dm_opd) |
| `.dm_mode` | `int32` | `0` | `init` | Mirror type: 0 = 2D zonal DM, 1 = 2-axis Tip-Tilt mirror (2x1 cmd) |
| `.nwf` | `uint32` | `128` | `init` | Output pupil OPD grid dimension [pixels] (default: 128) |
| `.nact_x` | `uint32` | `16` | `init` | Fallback actuator X dimension if in_cmd is created |
| `.nact_y` | `uint32` | `16` | `init` | Fallback actuator Y dimension if in_cmd is created |
| `.dm_diam` | `float32` | `0.9375` | `init` | DM active diameter fraction of nwf (default: 0.9375) |
| `.if_coupling` | `float32` | `0.15` | `init` | Inter-actuator coupling at 1 pitch (default: 0.15) |
| `.refl_scale` | `float32` | `2` | `init` | OPD per mechanical stroke unit (default: 2.0) |
| `.tput` | `float32` | `0.98` | `live` | Mirror optical reflectivity [0..1] applied to out_beam spectrum (default: 0.98) |
| `.max_stroke` | `float32` | `5` | `live` | Mechanical stroke saturation limit [um] (<=0 disables) |
| `.dac_bits` | `int32` | `16` | `init` | DAC quantization bit depth (0 = floating-point) |
| `.nonlin_quad` | `float32` | `0` | `init` | Quadratic voltage/stroke non-linearity coefficient |
| `.dyn_mode` | `int32` | `1` | `init` | Temporal mode: 0=instant, 1=1st-order RC, 2=2nd-order oscillator |
| `.dt_sec` | `float32` | `0.0005` | `init` | Physics time step [s] (default: 0.0005 = 2 kHz) |
| `.tau_sec` | `float32` | `0.00015` | `live` | 1st-order 1/e time constant [s] (default: 0.00015) |
| `.f0_hz` | `float32` | `3000` | `init` | 2nd-order natural resonance frequency [Hz] (default: 3000) |
| `.zeta` | `float32` | `0.7` | `init` | 2nd-order damping ratio (default: 0.7) |
| `.delay_frames` | `float32` | `0` | `live` | Transport latency [frames, fractional supported] |
| `.delay_us` | `float32` | `0` | `live` | Physical transport latency [us] (>0 overrides delay_frames via delay_us/dt) |
| `.shift_x` | `float32` | `0` | `init` | DM-to-pupil X shift [pixels] (default: 0) |
| `.shift_y` | `float32` | `0` | `init` | DM-to-pupil Y shift [pixels] (default: 0) |
| `.rot_deg` | `float32` | `0` | `init` | DM-to-pupil CCW rotation [deg] (default: 0) |
| `.mag_scale` | `float32` | `1` | `init` | DM-to-pupil magnification (default: 1.0) |
| `.in_flat` | `string` | `""` | `init` | Optional static DM flat/shape error OPD stream [um] |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |

### `beamsplit` (`milk-fpsexec-cacaosim-beamsplit` · FPS `csim_beamsplit`)

Split optical beam via dichroic or gray beamsplitter into two arms

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.in_beam` | `stream` | `csim_beam0` | `init` | Input 3D [nwf x nwf x 2] optical beam stream |
| `.out_refl` | `stream` | `csim_beam_refl` | `init` | Output reflected arm [nwf x nwf x 2] beam stream |
| `.out_trans` | `stream` | `csim_beam_trans` | `init` | Output transmitted arm [nwf x nwf x 2] beam stream |
| `.nwf` | `uint32` | `128` | `init` | Fallback pupil grid size [pixels] if in_beam is auto-created (default: 128) |
| `.split_mode` | `int32` | `1` | `live` | 0=gray, 1=dichroic_LP (refl<cut), 2=dichroic_SP (refl>cut), 3=band_refl |
| `.r_gray` | `float32` | `0.5` | `live` | Gray reflectance [0..1] when split_mode == 0 (default: 0.50) |
| `.t_gray` | `float32` | `0.5` | `live` | Gray transmittance [0..1] when split_mode == 0 (default: 0.50) |
| `.lambda_cut` | `float32` | `0.95` | `live` | Dichroic cutoff or band center wavelength [um] (default: 0.95) |
| `.band_width` | `float32` | `0.25` | `live` | Band-reflect full width [um] when split_mode == 3 (default: 0.25) |
| `.roll_um` | `float32` | `0.03` | `init` | Dichroic transition edge width [um] (default: 0.03) |
| `.eff_refl` | `float32` | `0.95` | `live` | Peak reflectance efficiency [0..1] (default: 0.95) |
| `.eff_trans` | `float32` | `0.95` | `live` | Peak transmittance efficiency [0..1] (default: 0.95) |
| `.ncpa_refl` | `string` | `""` | `init` | Optional static NCPA OPD stream [um] on reflected arm |
| `.ncpa_trans` | `string` | `""` | `init` | Optional static NCPA OPD stream [um] on transmitted arm |
| `.pup_refl` | `string` | `""` | `init` | Optional pupil/Lyot mask stream on reflected arm |
| `.pup_trans` | `string` | `""` | `init` | Optional pupil/Lyot mask stream on transmitted arm |
| `.refl_dx` | `float32` | `0` | `init` | Reflected arm X shift [pixels] (default: 0) |
| `.refl_dy` | `float32` | `0` | `init` | Reflected arm Y shift [pixels] (default: 0) |
| `.refl_rot` | `float32` | `0` | `init` | Reflected arm rotation [deg] (default: 0) |
| `.refl_scale` | `float32` | `1` | `init` | Reflected arm magnification (default: 1.0) |
| `.trans_dx` | `float32` | `0` | `init` | Transmitted arm X shift [pixels] (default: 0) |
| `.trans_dy` | `float32` | `0` | `init` | Transmitted arm Y shift [pixels] (default: 0) |
| `.trans_rot` | `float32` | `0` | `init` | Transmitted arm rotation [deg] (default: 0) |
| `.trans_scale` | `float32` | `1` | `init` | Transmitted arm magnification (default: 1.0) |

### `coro` (`milk-fpsexec-cacaosim-coro` · FPS `csim_coro`)

Simulate coronagraphic focal/pupil masks, Lyot, and LOWFS pickoff

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.in_beam` | `stream` | `csim_beam0` | `init` | Input 3D [nwf x nwf x 2*N] optical beam stream |
| `.out_beam` | `stream` | `csim_beam_coro` | `init` | Output post-Lyot [nwf x nwf x 2*Nlam] coronagraphic beam stream |
| `.out_lowfs` | `string` | `""` | `init` | Optional FPM/Lyot-reflected [nwf x nwf x 2*Nlam] LOWFS pupil beam stream |
| `.out_pre_lyot` | `string` | `""` | `init` | Optional 2D [nwf x nwf] instantaneous pupil intensity incident on Lyot stop |
| `.out_pre_lyot_long` | `string` | `""` | `init` | Optional 2D [nwf x nwf] cumulative long-exposure pre-Lyot intensity stream |
| `.in_pup_amp` | `string` | `""` | `init` | Optional custom pre-FPM pupil amplitude mask (FITS or SHM, 2D/3D) |
| `.in_pup_pha` | `string` | `""` | `init` | Optional custom pre-FPM pupil phase/OPD mask (FITS or SHM, 2D/3D) |
| `.in_fpm_amp` | `string` | `""` | `init` | Optional custom focal-plane amplitude mask (FITS or SHM, 2D/3D) |
| `.in_fpm_pha` | `string` | `""` | `init` | Optional custom focal-plane phase/OPD mask (FITS or SHM, 2D/3D) |
| `.in_lyot` | `string` | `""` | `init` | Optional custom Lyot stop amplitude mask (FITS or SHM, 2D/3D) |
| `.in_lyot_pha` | `string` | `""` | `init` | Optional custom Lyot stop phase/OPD mask (FITS or SHM, 2D/3D) |
| `.nwf` | `uint32` | `128` | `init` | Fallback pupil grid size [pixels] (default: 128) |
| `.coro_mode` | `int32` | `0` | `init` | 0=Lyot, 1=dimple, 2=vortex, 3=4QPM, 4=custom MFT, 5=custom FFT |
| `.n_mft` | `uint32` | `48` | `init` | Babinet MFT core resolution for modes 0..1 (default: 48) |
| `.pad_factor` | `uint32` | `4` | `init` | 2D FFT padding factor for modes 2, 3, 5 (>=2, default: 4) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Entrance pupil diameter fraction of nwf (default: 0.9375) |
| `.lambda_um` | `float32` | `1.65` | `init` | Coronagraph central wavelength [um] (default: 1.65) |
| `.bw_frac` | `float32` | `0` | `init` | Fractional bandpass dlambda/lambda0 (default: 0.0, 0.20 for 3D masks) |
| `.nlambda` | `int32` | `1` | `init` | Polychromatic channels (0/1=auto from 3D masks, >1=explicit) |
| `.fpm_rad_ld` | `float32` | `3` | `init` | Focal mask radius [lambda0/D] for modes 0..1 (default: 3.0) |
| `.fpm_amp` | `float32` | `0` | `init` | Core amplitude transmission [0..1] (0 = opaque spot) |
| `.fpm_phase` | `float32` | `3.14159` | `init` | Core phase shift [rad] for mode 1 (default: pi) |
| `.fpm_pixscale` | `float32` | `8` | `init` | Custom MFT focal sampling [pixels per lambda0/D] (default: 8.0) |
| `.pha_unit` | `int32` | `0` | `init` | Custom *_pha mask unit: 0 = radians, 1 = OPD [um] (default: 0) |
| `.vortex_charge` | `int32` | `2` | `init` | Optical vortex topological charge (2 or 4, default: 2) |
| `.lowfs_mode` | `int32` | `0` | `live` | LOWFS pickoff: 0=fpm_refl (FPM absorbed), 1=lyot_refl (Lyot stop reflected) |
| `.lowfs_pup_diam` | `float32` | `1` | `init` | Pupil aperture size on FPM->LOWFS path [D_pup] (default: 1.0) |
| `.lowfs_refl` | `float32` | `0.95` | `live` | Amplitude reflectivity into out_lowfs port (default: 0.95) |
| `.tput` | `float32` | `0.92` | `live` | Coronagraph relay optical throughput [0..1] (default: 0.92) |
| `.lyot_outer` | `float32` | `0.85` | `init` | Lyot stop outer diameter fraction of pup_diam (default: 0.85) |
| `.lyot_obs` | `float32` | `0.36` | `init` | Lyot stop central obscuration fraction of pup_diam (default: 0.36) |
| `.lyot_nspider` | `int32` | `4` | `init` | Number of Lyot stop spider vanes (default: 4) |
| `.lyot_spider_w` | `float32` | `0.02` | `init` | Lyot spider width fraction of pup_diam (default: 0.02) |
| `.lyot_spider_deg` | `float32` | `45` | `init` | Lyot spider orientation [deg] (default: 45.0) |
| `.lyot_spider_offset` | `float32` | `0` | `init` | Lyot V-spider X-vertex offset fraction of pup_diam (default: 0.0) |
| `.reset_long` | `onoff` | `OFF` | `live` | Toggle ON to reset cumulative pre-Lyot intensity and LOWFS power counters |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |

### `pyrwfs` (`milk-fpsexec-cacaosim-pyrwfs` · FPS `csim_pyrwfs`)

Simulate modulated/unmodulated 4-sided Pyramid WFS and detector

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.out_cam` | `stream` | `aol0_wfsim` | `init` | Output 2D WFS camera stream (e.g. aol0_wfsim) |
| `.in_dm` | `stream` | `csim_dmopd` | `init` | Input 2D DM OPD stream [um] (optional trigger) |
| `.in_beam` | `string` | `""` | `init` | Optional 3D [nwf x nwf x 2] input optical beam stream |
| `.in_wf` | `string` | `outarraypha` | `init` | Optional atmospheric WF stream (e.g. outarraypha from milkatmturb) |
| `.in_amp` | `string` | `""` | `init` | Optional atmospheric amplitude stream (e.g. outarrayamp) |
| `.in_pup` | `string` | `csim_pup` | `init` | Input 2D pupil amplitude stream (auto-generated if absent) |
| `.in_ncpa` | `string` | `""` | `init` | Optional static WFS-path NCPA OPD stream [um] |
| `.nwf` | `uint32` | `128` | `init` | Pupil grid size [pixels] (default: 128) |
| `.pad_factor` | `uint32` | `2` | `init` | Optical FFT oversampling factor (>=2, default: 2) |
| `.ncam` | `uint32` | `64` | `init` | WFS detector grid size [pixels] (default: 64) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Clear pupil diameter fraction of nwf (default: 0.9375) |
| `.lambda_um` | `float32` | `0.75` | `init` | WFS sensing wavelength [um] (default: 0.75) |
| `.bw_frac` | `float32` | `0.2` | `init` | Filter fractional bandwidth dlambda/lambda0 (default: 0.20) |
| `.nlambda` | `int32` | `0` | `init` | Polychromatic spectral slices (0 = auto -> 2, 1..32, default: 0) |
| `.qe` | `float32` | `0.8` | `live` | Detector quantum efficiency [0..1] for beam photometry (default: 0.80) |
| `.dt_exp` | `float32` | `0.001` | `live` | Subframe step [s] for beam photometry when frame_rate == 0 (default: 0.001) |
| `.use_beam_flux` | `onoff` | `ON` | `live` | Compute flux_ph from in_beam photon spectrum when available |
| `.navg` | `uint32` | `1` | `live` | Optical subframes integrated per WFS readout (>=1, default: 1) |
| `.mod_rad` | `float32` | `3` | `live` | Modulation radius [lambda/D] (0 = unmodulated, default: 3.0) |
| `.mod_npts` | `int32` | `16` | `init` | Number of modulation points around circle (default: 16) |
| `.mod_mode` | `int32` | `1` | `init` | Subframe modulation mode: 0=full, 1=sync_arc, 2=interleaved (default: 1) |
| `.mod_rev` | `int32` | `1` | `init` | Modulator revolutions per WFS exposure (default: 1) |
| `.mod_phase_deg` | `float32` | `0` | `init` | Initial modulator azimuth at exposure start [deg] (default: 0.0) |
| `.pup_sep` | `float32` | `0.5` | `init` | Sub-pupil center separation fraction of FFT grid (default: 0.50) |
| `.roof_ld` | `float32` | `0` | `init` | Pyramid vertex roof width [lambda/D] (default: 0.0) |
| `.shift_x` | `float32` | `0` | `init` | WFS pupil X registration shift [pixels] (default: 0) |
| `.shift_y` | `float32` | `0` | `init` | WFS pupil Y registration shift [pixels] (default: 0) |
| `.rot_deg` | `float32` | `0` | `init` | WFS pupil CCW rotation [deg] (default: 0) |
| `.mag_scale` | `float32` | `1` | `init` | WFS pupil magnification (default: 1.0) |
| `.wf_unit` | `int32` | `1` | `init` | Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1) |
| `.ref_lambda` | `float32` | `0.5` | `init` | Reference wavelength [um] for phase input (default: 0.50) |
| `.turb_gain` | `float32` | `1` | `live` | Live atmospheric turbulence multiplier (0 = off for calibration) |
| `.flux_ph` | `float32` | `100000` | `live` | Total guide-star photons per WFS frame (<0 raw intensity) |
| `.dark_e` | `float32` | `0` | `live` | Dark + sky background [e-/pix/frame] |
| `.ron_e` | `float32` | `0.5` | `live` | Detector readout noise [e- RMS] (default: 0.5) |
| `.gain_eadu` | `float32` | `1` | `live` | Camera gain [e-/ADU] (default: 1.0) |
| `.bias_adu` | `float32` | `100` | `live` | Camera bias offset [ADU] (default: 100.0) |
| `.sat_adu` | `float32` | `65535` | `init` | Detector saturation ceiling [ADU] (default: 65535) |
| `.shutter` | `onoff` | `ON` | `live` | Optical shutter (ON = open, OFF = dark frame) |
| `.photon_noise` | `onoff` | `ON` | `live` | Enable Poisson photon shot noise |
| `.out_dtype` | `int32` | `0` | `init` | Output stream datatype: 0 = uint16, 1 = float32 |
| `.frame_rate` | `float32` | `0` | `live` | Internal TIMER_ABSTIME pacing rate [Hz] (0 = use procinfo trigger) |
| `.rng_seed` | `uint64` | `12345` | `init` | Detector noise PRNG seed |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |

### `shwfs` (`milk-fpsexec-cacaosim-shwfs` · FPS `csim_shwfs`)

Simulate diffractive Shack-Hartmann lenslet WFS and detector

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.out_cam` | `stream` | `csim_shwfs` | `init` | Output 2D Shack-Hartmann WFS camera stream [n_sub*pix_per_sub] |
| `.in_dm` | `stream` | `csim_dmopd` | `init` | Input 2D DM OPD stream [um] (optional trigger) |
| `.in_beam` | `string` | `""` | `init` | Optional 3D [nwf x nwf x 2] input optical beam stream |
| `.in_wf` | `string` | `outarraypha` | `init` | Optional atmospheric WF stream (e.g. outarraypha) |
| `.in_amp` | `string` | `""` | `init` | Optional atmospheric amplitude stream |
| `.in_pup` | `string` | `csim_pup` | `init` | Input 2D pupil amplitude stream (auto-generated if absent) |
| `.in_ncpa` | `string` | `""` | `init` | Optional static WFS-path NCPA OPD stream [um] |
| `.nwf` | `uint32` | `128` | `init` | Pupil grid size [pixels] (default: 128) |
| `.n_sub` | `uint32` | `16` | `init` | Number of lenslets across pupil (default: 16) |
| `.pix_per_sub` | `uint32` | `8` | `init` | Detector pixels per lenslet side (default: 8 -> 128x128 cam) |
| `.pad_factor` | `uint32` | `2` | `init` | Subaperture FFT oversampling factor (>=2, default: 2) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Clear pupil diameter fraction of nwf (default: 0.9375) |
| `.lambda_um` | `float32` | `0.75` | `init` | WFS sensing wavelength [um] (default: 0.75) |
| `.bw_frac` | `float32` | `0.2` | `init` | Filter fractional bandwidth dlambda/lambda0 (default: 0.20) |
| `.nlambda` | `int32` | `0` | `init` | Polychromatic spectral slices (0 = auto -> 2, 1..32, default: 0) |
| `.qe` | `float32` | `0.8` | `live` | Detector quantum efficiency [0..1] for beam photometry (default: 0.80) |
| `.dt_exp` | `float32` | `0.001` | `live` | Subframe step [s] for beam photometry when frame_rate == 0 (default: 0.001) |
| `.use_beam_flux` | `onoff` | `ON` | `live` | Compute flux_ph from in_beam photon spectrum when available |
| `.navg` | `uint32` | `1` | `live` | Optical subframes integrated per WFS readout (>=1, default: 1) |
| `.shift_x` | `float32` | `0` | `init` | WFS pupil X registration shift [pixels] (default: 0) |
| `.shift_y` | `float32` | `0` | `init` | WFS pupil Y registration shift [pixels] (default: 0) |
| `.rot_deg` | `float32` | `0` | `init` | WFS pupil CCW rotation [deg] (default: 0) |
| `.mag_scale` | `float32` | `1` | `init` | WFS pupil magnification (default: 1.0) |
| `.wf_unit` | `int32` | `1` | `init` | Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1) |
| `.ref_lambda` | `float32` | `0.5` | `init` | Reference wavelength [um] for phase input (default: 0.50) |
| `.turb_gain` | `float32` | `1` | `live` | Live atmospheric turbulence multiplier (0 = off) |
| `.flux_ph` | `float32` | `100000` | `live` | Total guide-star photons per WFS frame (<0 raw intensity) |
| `.dark_e` | `float32` | `0` | `live` | Dark + sky background [e-/pix/frame] |
| `.ron_e` | `float32` | `0.5` | `live` | Detector readout noise [e- RMS] (default: 0.5) |
| `.gain_eadu` | `float32` | `1` | `live` | Camera gain [e-/ADU] (default: 1.0) |
| `.bias_adu` | `float32` | `100` | `live` | Camera bias offset [ADU] (default: 100.0) |
| `.sat_adu` | `float32` | `65535` | `init` | Detector saturation ceiling [ADU] (default: 65535) |
| `.shutter` | `onoff` | `ON` | `live` | Optical shutter (ON = open, OFF = dark frame) |
| `.photon_noise` | `onoff` | `ON` | `live` | Enable Poisson photon shot noise |
| `.out_dtype` | `int32` | `0` | `init` | Output stream datatype: 0 = uint16, 1 = float32 |
| `.frame_rate` | `float32` | `0` | `live` | Internal TIMER_ABSTIME pacing rate [Hz] (0 = use procinfo trigger) |
| `.rng_seed` | `uint64` | `23456` | `init` | Detector noise PRNG seed |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |

### `zwfs` (`milk-fpsexec-cacaosim-zwfs` · FPS `csim_zwfs`)

Simulate Babinet MFT Zernike phase-contrast WFS and detector

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.out_cam` | `stream` | `csim_zwfs` | `init` | Output 2D Zernike WFS camera stream |
| `.in_dm` | `stream` | `csim_dmopd` | `init` | Input 2D DM OPD stream [um] (optional trigger) |
| `.in_beam` | `string` | `""` | `init` | Optional 3D [nwf x nwf x 2] input optical beam stream (e.g. LOWFS core beam) |
| `.in_wf` | `string` | `outarraypha` | `init` | Optional atmospheric WF stream (e.g. outarraypha) |
| `.in_amp` | `string` | `""` | `init` | Optional atmospheric amplitude stream |
| `.in_pup` | `string` | `csim_pup` | `init` | Input 2D pupil amplitude stream (auto-generated if absent) |
| `.in_ncpa` | `string` | `""` | `init` | Optional static WFS-path NCPA OPD stream [um] |
| `.nwf` | `uint32` | `128` | `init` | Pupil grid size [pixels] (default: 128) |
| `.ncam` | `uint32` | `64` | `init` | WFS detector grid size [pixels] (default: 64) |
| `.n_mft` | `uint32` | `32` | `init` | Focal-core Matrix-DFT resolution (>=16, default: 32) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Clear pupil diameter fraction of nwf (default: 0.9375) |
| `.lambda_um` | `float32` | `0.75` | `init` | WFS sensing wavelength [um] (default: 0.75) |
| `.dimple_ld` | `float32` | `1.06` | `init` | Phase dimple diameter [lambda/D] (default: 1.06) |
| `.dimple_rad` | `float32` | `1.5708` | `init` | Dimple phase shift [rad] (default: pi/2 = 1.570796) |
| `.dimple_amp` | `float32` | `1` | `init` | Dimple amplitude transmission [0..1] (default: 1.0) |
| `.bw_frac` | `float32` | `0.2` | `init` | Filter fractional bandwidth dlambda/lambda0 (default: 0.20) |
| `.nlambda` | `int32` | `0` | `init` | Polychromatic spectral slices (0 = auto -> 2, 1..32, default: 0) |
| `.qe` | `float32` | `0.8` | `live` | Detector quantum efficiency [0..1] for beam photometry (default: 0.80) |
| `.dt_exp` | `float32` | `0.001` | `live` | Subframe step [s] for beam photometry when frame_rate == 0 (default: 0.001) |
| `.use_beam_flux` | `onoff` | `ON` | `live` | Compute flux_ph from in_beam photon spectrum when available |
| `.navg` | `uint32` | `1` | `live` | Optical subframes integrated per WFS readout (>=1, default: 1) |
| `.shift_x` | `float32` | `0` | `init` | WFS pupil X registration shift [pixels] (default: 0) |
| `.shift_y` | `float32` | `0` | `init` | WFS pupil Y registration shift [pixels] (default: 0) |
| `.rot_deg` | `float32` | `0` | `init` | WFS pupil CCW rotation [deg] (default: 0) |
| `.mag_scale` | `float32` | `1` | `init` | WFS pupil magnification (default: 1.0) |
| `.wf_unit` | `int32` | `1` | `init` | Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1) |
| `.ref_lambda` | `float32` | `0.5` | `init` | Reference wavelength [um] for phase input (default: 0.50) |
| `.turb_gain` | `float32` | `1` | `live` | Live atmospheric turbulence multiplier (0 = off) |
| `.flux_ph` | `float32` | `100000` | `live` | Total guide-star photons per WFS frame (<0 raw intensity) |
| `.dark_e` | `float32` | `0` | `live` | Dark + sky background [e-/pix/frame] |
| `.ron_e` | `float32` | `0.5` | `live` | Detector readout noise [e- RMS] (default: 0.5) |
| `.gain_eadu` | `float32` | `1` | `live` | Camera gain [e-/ADU] (default: 1.0) |
| `.bias_adu` | `float32` | `100` | `live` | Camera bias offset [ADU] (default: 100.0) |
| `.sat_adu` | `float32` | `65535` | `init` | Detector saturation ceiling [ADU] (default: 65535) |
| `.shutter` | `onoff` | `ON` | `live` | Optical shutter (ON = open, OFF = dark frame) |
| `.photon_noise` | `onoff` | `ON` | `live` | Enable Poisson photon shot noise |
| `.out_dtype` | `int32` | `0` | `init` | Output stream datatype: 0 = uint16, 1 = float32 |
| `.frame_rate` | `float32` | `0` | `live` | Internal TIMER_ABSTIME pacing rate [Hz] (0 = use procinfo trigger) |
| `.rng_seed` | `uint64` | `34567` | `init` | Detector noise PRNG seed |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |

### `fpcam` (`milk-fpsexec-cacaosim-fpcam` · FPS `csim_fpcam`)

Simulate polychromatic MFT/FFT focal plane PSF and detector

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.out_cam` | `stream` | `csim_fpcam` | `init` | Output 2D focal plane camera stream |
| `.out_long` | `string` | `csim_fplong` | `init` | Optional output float32 cumulative Strehl-normalized long-exposure PSF |
| `.in_dm` | `stream` | `csim_dmopd` | `init` | Input 2D DM OPD stream [um] (optional trigger) |
| `.in_beam` | `string` | `""` | `init` | Optional 3D [nwf x nwf x 2*N] input optical beam stream (e.g. post-coronagraph) |
| `.in_wf` | `string` | `outarraypha` | `init` | Optional atmospheric WF stream (e.g. outarraypha from milkatmturb) |
| `.in_amp` | `string` | `""` | `init` | Optional atmospheric amplitude stream (e.g. outarrayamp) |
| `.in_pup` | `string` | `csim_pup` | `init` | Input 2D pupil amplitude stream (auto-generated if absent) |
| `.in_ncpa` | `string` | `""` | `init` | Optional static science-path NCPA OPD stream [um] |
| `.nwf` | `uint32` | `128` | `init` | Pupil grid size [pixels] (default: 128) |
| `.ncam` | `uint32` | `128` | `init` | Science camera grid size [pixels] (default: 128) |
| `.prop_mode` | `int32` | `0` | `init` | Propagation mode: 0 = Soummer MFT, 1 = padded 2D FFT (default: 0) |
| `.pad_factor` | `uint32` | `2` | `init` | FFT oversampling factor when prop_mode == 1 (>=2, default: 2) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Clear pupil diameter fraction of nwf (default: 0.9375) |
| `.lambda_um` | `float32` | `1.65` | `init` | Central science wavelength [um] (default: 1.65) |
| `.bw_frac` | `float32` | `0.18` | `init` | Fractional bandpass dlambda/lambda0 (default: 0.18) |
| `.nlambda` | `int32` | `0` | `init` | Polychromatic spectral slices (0 = auto -> 2 if bw>0, 1..32, default: 0) |
| `.pixscale_ld` | `float32` | `3` | `init` | Focal sampling [pixels per lambda0/D] in MFT mode (default: 3.0) |
| `.qe` | `float32` | `0.85` | `live` | Detector quantum efficiency [0..1] for beam photometry (default: 0.85) |
| `.dt_exp` | `float32` | `0.001` | `live` | Subframe step [s] for beam photometry when frame_rate == 0 (default: 0.001) |
| `.use_beam_flux` | `onoff` | `ON` | `live` | Compute flux_ph from in_beam photon spectrum when available |
| `.shift_x` | `float32` | `0` | `init` | Camera pupil X registration shift [pixels] (default: 0) |
| `.shift_y` | `float32` | `0` | `init` | Camera pupil Y registration shift [pixels] (default: 0) |
| `.rot_deg` | `float32` | `0` | `init` | Camera pupil CCW rotation [deg] (default: 0) |
| `.mag_scale` | `float32` | `1` | `init` | Camera pupil magnification (default: 1.0) |
| `.wf_unit` | `int32` | `1` | `init` | Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1) |
| `.ref_lambda` | `float32` | `0.5` | `init` | Reference wavelength [um] for phase input (default: 0.50) |
| `.turb_gain` | `float32` | `1` | `live` | Live atmospheric turbulence multiplier (0 = off) |
| `.navg` | `uint32` | `1` | `live` | Optical subframes integrated per camera readout (>=1, default: 1) |
| `.reset_long` | `onoff` | `OFF` | `live` | Toggle ON to clear cumulative long-exposure PSF |
| `.flux_ph` | `float32` | `1e+06` | `live` | Total incident starlight photons per camera exposure (<0 raw Strehl PSF) |
| `.dark_e` | `float32` | `0` | `live` | Dark + sky background [e-/pix/exposure] |
| `.ron_e` | `float32` | `1` | `live` | Detector readout noise [e- RMS] (default: 1.0) |
| `.gain_eadu` | `float32` | `1` | `live` | Camera gain [e-/ADU] (default: 1.0) |
| `.bias_adu` | `float32` | `100` | `live` | Camera bias offset [ADU] (default: 100.0) |
| `.sat_adu` | `float32` | `65535` | `init` | Detector saturation ceiling [ADU] (default: 65535) |
| `.shutter` | `onoff` | `ON` | `live` | Optical shutter (ON = open, OFF = dark frame) |
| `.photon_noise` | `onoff` | `ON` | `live` | Enable Poisson photon shot noise |
| `.out_dtype` | `int32` | `0` | `init` | Output stream datatype: 0 = uint16, 1 = float32 |
| `.frame_rate` | `float32` | `0` | `live` | Internal TIMER_ABSTIME subframe pacing rate [Hz] (0 = use trigger) |
| `.rng_seed` | `uint64` | `54321` | `init` | Detector noise PRNG seed |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |
| `.strehl_inst` | `float32` | `0` | `output` | Instantaneous exposure peak optical Strehl ratio [0..1] |
| `.strehl_ave` | `float32` | `0` | `output` | Cumulative long-exposure peak optical Strehl ratio [0..1] |

### `htsim` (`milk-fpsexec-cacaosim-htsim` · FPS `csim_htsim`)

Fused zero-hop HT hardware simulator (DM + PyWFS + FPcam)

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.bench_file` | `string` | `""` | `init` | Optional declarative multi-stage .bench DAG file (empty = flat single-stage) |
| `.export_beams` | `string` | `""` | `init` | CSV optical beams to export to SHM (empty=none, *=all) |
| `.export_enable` | `onoff` | `ON` | `live` | Live gate for SHM export of .export_beams (ON/OFF) |
| `.wfstat_enable` | `onoff` | `OFF` | `live` | In-DAG wavefront stats: ON starts, OFF (or .wfstat_ticks) writes outputs |
| `.wfstat_beams` | `string` | `b_pup,b_post_stg1,b_post_stg2` | `init` | CSV raw,stg1,final beam names for in-DAG wavefront stats |
| `.wfstat_outdir` | `string` | `.` | `live` | Output directory for wf_stats.json and FITS products (<64 chars) |
| `.wfstat_ticks` | `uint64` | `20000` | `live` | Frames to accumulate before auto-writing (0 = until disabled) |
| `.wfstat_stride` | `uint32` | `0` | `live` | Master-tick stride between wfstat samples (0 = auto oversample) |
| `.wfstat_extra` | `onoff` | `ON` | `live` | Snapshot WFS/DM/PSF diagnostic streams at the 5 sample instants |
| `.wfstat_frames` | `uint64` | `0` | `output` | Frames accumulated by the current/last in-DAG wavefront stats run |
| `.calib_fast` | `onoff` | `OFF` | `live` | Lockstep calibration: skip fpcam/DM dynamics & force WFS navg=1 |
| `.calib_temporal` | `onoff` | `OFF` | `live` | Step real DM dynamics & oversampled WFS subframes during .calib_fast |
| `.calib_ring` | `onoff` | `OFF` | `live` | Auto-kick closed WFS->MVM->mfilt->DM ring on 5ms timeout in .calib_fast |
| `.calib_poke_offset` | `float32` | `0` | `live` | DM poke arrival offset [us] relative to WFS frame start in .calib_temporal |
| `.calib_trig` | `stream` | `none` | `live` | DM stream triggering step in .calib_fast mode ("none" = disabled) |
| `.calib_wfs` | `stream` | `none` | `live` | Target WFS cam/id during .calib_fast ("none" = all WFS) |
| `.in_dmcmd` | `stream` | `aol0_dmC` | `init` | Input 2D DM actuator command stream (auto-created if missing) |
| `.out_wfscam` | `stream` | `aol0_wfsim` | `init` | Output 2D Pyramid WFS camera stream (e.g. aol0_wfsim) |
| `.out_dmopd` | `string` | `""` | `init` | Optional output 2D DM OPD stream (empty = zero-copy GPU internal) |
| `.enable_fpcam` | `onoff` | `ON` | `live` | Enable in-process focal plane science PSF camera |
| `.out_fpcam` | `string` | `csim_fpcam` | `init` | Output 2D focal plane science camera stream |
| `.out_fplong` | `string` | `csim_fplong` | `init` | Optional output float32 cumulative long-exposure PSF stream |
| `.in_wf` | `string` | `outarraypha` | `init` | Optional atmospheric WF stream (e.g. outarraypha from milkatmturb) |
| `.in_amp` | `string` | `""` | `init` | Optional atmospheric amplitude stream (e.g. outarrayamp) |
| `.in_pup` | `string` | `csim_pup` | `init` | Input 2D pupil amplitude stream (auto-generated if absent) |
| `.nwf` | `uint32` | `128` | `init` | Pupil grid size [pixels] (default: 128) |
| `.pup_diam` | `float32` | `0.9375` | `init` | Clear pupil diameter fraction of nwf (default: 0.9375) |
| `.nact_x` | `uint32` | `20` | `init` | DM actuator grid X size (default: 20) |
| `.nact_y` | `uint32` | `20` | `init` | DM actuator grid Y size (default: 20) |
| `.if_coupling` | `float32` | `0.2` | `init` | DM inter-actuator mechanical coupling (default: 0.20) |
| `.delay_frames` | `float32` | `1` | `live` | Pure DM hardware latency [frames] (default: 1.0) |
| `.dyn_mode` | `int32` | `1` | `init` | DM dynamics: 0=instant, 1=1st-order RC, 2=2nd-order resonance |
| `.tau_us` | `float32` | `120` | `live` | DM 1st-order RC time constant [us] (default: 120.0) |
| `.f0_hz` | `float32` | `2500` | `init` | DM 2nd-order natural resonance [Hz] (default: 2500.0) |
| `.zeta` | `float32` | `0.7` | `init` | DM 2nd-order damping ratio (default: 0.70) |
| `.max_stroke` | `float32` | `5` | `live` | DM mechanical stroke clip [um] (default: 5.0) |
| `.dac_bits` | `int32` | `16` | `init` | DM DAC bit depth (0 = floating point, default: 16) |
| `.wfs_ncam` | `uint32` | `64` | `init` | WFS camera size [pixels] (default: 64) |
| `.wfs_lambda` | `float32` | `0.75` | `init` | WFS wavelength [um] (default: 0.75) |
| `.mod_rad` | `float32` | `3` | `live` | PyWFS modulation radius [lambda/D] (default: 3.0) |
| `.mod_npts` | `int32` | `16` | `init` | PyWFS modulation points (default: 16) |
| `.wfs_flux` | `float32` | `100000` | `live` | Guide-star flux [photons / WFS frame] (default: 100000) |
| `.wfs_ron` | `float32` | `0.5` | `live` | WFS camera readout noise [e- RMS] (default: 0.5) |
| `.wfs_shutter` | `onoff` | `ON` | `live` | WFS optical shutter (ON = open, OFF = dark frame) |
| `.fp_ncam` | `uint32` | `128` | `init` | Focal plane camera size [pixels] (default: 128) |
| `.fp_lambda` | `float32` | `1.65` | `init` | Focal plane wavelength [um] (default: 1.65) |
| `.fp_pixscale` | `float32` | `3` | `init` | Focal plane pixel scale [pixels per lambda/D] (default: 3.0) |
| `.fp_navg` | `uint32` | `10` | `live` | WFS frames integrated per focal plane readout (default: 10) |
| `.fp_flux` | `float32` | `1e+06` | `live` | Science starlight flux [photons / FP exposure] (default: 1e6) |
| `.fp_ron` | `float32` | `1` | `live` | Focal plane readout noise [e- RMS] (default: 1.0) |
| `.reset_long` | `onoff` | `OFF` | `live` | Toggle ON to clear cumulative long-exposure PSF |
| `.photon_noise` | `onoff` | `ON` | `live` | Enable Poisson photon shot noise on detectors |
| `.out_dtype` | `int32` | `0` | `init` | Detector output datatype: 0 = uint16, 1 = float32 |
| `.wf_unit` | `int32` | `1` | `init` | Input in_wf unit: 0 = OPD [um], 1 = phase [rad] (default: 1) |
| `.ref_lambda` | `float32` | `0.5` | `init` | Reference wavelength [um] for phase input (default: 0.50) |
| `.turb_gain` | `float32` | `1` | `live` | Live turbulence multiplier (0 = off for calibration, 1 = nominal) |
| `.frame_rate` | `float32` | `2000` | `live` | Wall-clock pacing [Hz] (0 = procinfo trigger; .bench master_hz sets physics) |
| `.rng_seed` | `uint64` | `98765` | `init` | Detector PRNG seed |
| `.backend` | `int32` | `0` | `init` | Compute backend: 0=auto, 1=CPU, 2=CUDA |
| `.strehl_inst` | `float32` | `0` | `output` | Instantaneous focal plane peak Strehl ratio [0..1] |
| `.strehl_ave` | `float32` | `0` | `output` | Cumulative long-exposure peak Strehl ratio [0..1] |

### `wfstat` (`milk-fpsexec-cacaosim-wfstat` · FPS `csim_wfstat`)

Real-time pupil wavefront stats & synchronized FITS cube recorder

| Keyword | Type | Default | Access | Description |
| :--- | :--- | :--- | :--- | :--- |
| `.trig_beam` | `stream` | `b_post_stg2` | `init` | Trigger optical beam stream (usually post-Stage-2) |
| `.beam_raw` | `string` | `b_pup` | `init` | Raw entrance pupil beam stream name |
| `.beam_stg1` | `string` | `b_post_stg1` | `init` | Post-Stage-1 optical beam stream name |
| `.beam_final` | `string` | `b_post_stg2` | `init` | Post-Stage-2 final optical beam stream name |
| `.outdir` | `string` | `.` | `init` | Output directory (<64 chars; or set WFSTAT_OUTDIR env) |
| `.target_ticks` | `uint64` | `20000` | `init` | Planned integration tick count for 5 sample markers |
| `.dt_phys` | `float32` | `0.0005` | `init` | Physical frame interval [s] (default: 0.0005) |
| `.snap_extra` | `onoff` | `ON` | `init` | Capture synchronized WFS/DM/PSF 2D snapshot cubes |

## 3. Command-Line Tools (`cacaosim-bench`, `cacaosim-bench-view`, `cacaosim-CTRL`)

| Tool | Flag | Argument | Default | Group | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cacaosim-bench` | `-c, --config` | `<file.bench>` | — | `workflow` | Declarative multi-stage optical bench (.bench) configuration file |
| `cacaosim-bench` | `--mode` | `<fused|modular>` | fused | `deploy` | Execution topology: fused zero-hop htsim or modular multi-process FPS units |
| `cacaosim-bench` | `--nwf` | `<int>` | 128 | `plan` | Flat-mode pupil wavefront grid size [pixels] |
| `cacaosim-bench` | `--nact` | `<int>` | 20 | `plan` | Flat-mode linear DM actuator count |
| `cacaosim-bench` | `--wfs-ncam` | `<int>` | 64 | `plan` | Flat-mode WFS camera dimension [pixels] |
| `cacaosim-bench` | `--fp-ncam` | `<int>` | 128 | `plan` | Flat-mode focal-plane science camera dimension [pixels] |
| `cacaosim-bench` | `--rate` | `<float>` | 2000 | `deploy` | Override hardware master frame rate [Hz] |
| `cacaosim-bench` | `--oversample` | `<int>` | 0 | `deploy` | Override temporal oversampling factor K (0 = bench default) |
| `cacaosim-bench` | `--mod-rad` | `<float>` | 3.0 | `plan` | Flat-mode Pyramid modulation radius [lambda/D] |
| `cacaosim-bench` | `--backend` | `<0|1|2>` | 0 | `deploy` | Compute backend override: 0 = auto, 1 = CPU, 2 = CUDA |
| `cacaosim-bench` | `--atmturb` | — | OFF | `deploy` | Co-launch milk-fpsexec-atmturb-mkwfs live phase-screen stream |
| `cacaosim-bench` | `--seeing` | `<float>` | 0.7 | `deploy` | Atmospheric seeing [arcsec] when --atmturb is active |
| `cacaosim-bench` | `--restart` | — | OFF | `deploy` | Force unit restart and reset live FPFLAG_WRITERUN parameters |
| `cacaosim-bench` | `--workdir` | `<dir>` | — | `deploy` | Override fpspipe workspace root directory |
| `cacaosim-bench` | `--no-view` | — | OFF | `display` | Suppress automatic ANSI schematic rendering on start/status |
| `cacaosim-bench-view` | `-c, --config` | `<file.bench>` | — | `workflow` | Explicit .bench configuration path (also accepted as positional arg) |
| `cacaosim-bench-view` | `-i, --interactive` | — | OFF | `workflow` | Launch interactive milk-CTRL-style configurator (cacaosim-CTRL) |
| `cacaosim-bench-view` | `-g, --gui` | — | OFF | `workflow` | Alias for -i / --interactive TUI configurator |
| `cacaosim-bench-view` | `-w, --watch` | — | OFF | `workflow` | Auto-reload diagram live whenever <config.bench> or MILK_SHM_DIR changes |
| `cacaosim-bench-view` | `--init` | `<file.bench>` | — | `workflow` | Write a commented starter 2-stage .bench template to disk |
| `cacaosim-bench-view` | `-d, --deploy, --run` | — | OFF | `workflow` | Validate, deploy & start via cacaosim-bench, then visualize |
| `cacaosim-bench-view` | `--list-options` | — | OFF | `options` | Print complete .bench, FPS compute unit, and CLI option reference and exit |
| `cacaosim-bench-view` | `--section` | `<name>` | — | `options` | Filter --list-options to a specific .bench section (e.g. bench, dm, pyrwfs) |
| `cacaosim-bench-view` | `--search` | `<query>` | — | `options` | Filter --list-options by case-insensitive substring across keys and descriptions |
| `cacaosim-bench-view` | `--fps` | `<unit|all>` | — | `options` | Filter --list-options to one or all introspected MILK_RUNNABLE FPS units |
| `cacaosim-bench-view` | `--live` | — | OFF | `options` | With --list-options, list only FPFLAG_WRITERUN live-tunable FPS parameters |
| `cacaosim-bench-view` | `--format` | `<table|markdown|json>` | table | `options` | Output format for --list-options (aligned table, GFM Markdown, or JSON) |
| `cacaosim-bench-view` | `--output` | `<file>` | — | `options` | Write --list-options output to <file> instead of stdout |
| `cacaosim-bench-view` | `-p, --pretty, --color` | — | OFF | `display` | Force 24-bit TrueColor ANSI output even when piped |
| `cacaosim-bench-view` | `-t, --text, --no-color` | — | OFF | `display` | Plain monochrome text output (no ANSI color escapes) |
| `cacaosim-bench-view` | `--ascii` | — | OFF | `display` | Pure 7-bit ASCII box-drawing and spectrum sparklines |
| `cacaosim-bench-view` | `--theme` | `<name>` | dark | `display` | Palette: dark, night, accessible, light, or nordic |
| `cacaosim-bench-view` | `--compact` | — | OFF | `display` | Show overview + optical schematic only (omit hardware tables) |
| `cacaosim-bench-view` | `--no-shm` | — | OFF | `display` | Do not probe MILK_SHM_DIR for live .im.shm status |
| `cacaosim-bench-view` | `--no-footer` | — | OFF | `display` | Omit CLI workflow footer line |
| `cacaosim-bench-view` | `-W, --width` | `<cols>` | auto | `display` | Override schematic panel width in columns [76..124] |
| `cacaosim-bench-view` | `--mode` | `<fused|modular>` | fused | `deploy` | Deployment engine mode (also filters --list-options when passed with it) |
| `cacaosim-bench-view` | `--atmturb` | — | OFF | `deploy` | Co-launch milk-fpsexec-atmturb-mkwfs turbulence stream |
| `cacaosim-bench-view` | `--seeing` | `<arcsec>` | 0.7 | `deploy` | Atmospheric seeing [arcsec] when --atmturb is enabled |
| `cacaosim-bench-view` | `--emit-plan` | `<dir>` | — | `plan` | Compile .bench (or --flat) into fpspipe plan files in <dir> and exit |
| `cacaosim-bench-view` | `--flat` | — | OFF | `plan` | Emit single-stage flat plan without a .bench file (with --emit-plan) |
| `cacaosim-bench-view` | `--staged-bench` | `<path>` | — | `plan` | Write resolved .bench file for htsim staging (with --emit-plan) |
| `cacaosim-bench-view` | `--active-bench-arg` | `<str>` | csim_active.bench | `plan` | Override .bench_file argument recorded in params.batch (with --emit-plan) |
| `cacaosim-bench-view` | `--rate` | `<hz>` | 0 | `plan` | Override master rate [Hz] in emitted plan |
| `cacaosim-bench-view` | `--oversample` | `<int>` | 0 | `plan` | Override temporal oversampling factor K in emitted plan |
| `cacaosim-bench-view` | `--backend` | `<0|1|2>` | -1 | `plan` | Override backend (0=auto, 1=CPU, 2=CUDA) in emitted plan |
| `cacaosim-bench-view` | `--nwf` | `<int>` | 128 | `plan` | Flat-mode pupil grid size [px] (with --emit-plan --flat) |
| `cacaosim-bench-view` | `--nact` | `<int>` | 20 | `plan` | Flat-mode DM actuator count (with --emit-plan --flat) |
| `cacaosim-bench-view` | `--wfs-ncam` | `<int>` | 64 | `plan` | Flat-mode WFS camera dimension [px] (with --emit-plan --flat) |
| `cacaosim-bench-view` | `--fp-ncam` | `<int>` | 128 | `plan` | Flat-mode science camera dimension [px] (with --emit-plan --flat) |
| `cacaosim-bench-view` | `--mod-rad` | `<float>` | 3.0 | `plan` | Flat-mode PyWFS modulation radius [l/D] (with --emit-plan --flat) |
| `cacaosim-CTRL` | `-c, --config` | `<file.bench>` | — | `workflow` | Load or create .bench file (defaults to active SHM bench or preset) |
| `cacaosim-CTRL` | `-s, --status` | — | OFF | `workflow` | Start directly on F5:STATUS live MILK_SHM_DIR monitor tab |
| `cacaosim-CTRL` | `--schematic` | — | OFF | `workflow` | Start directly on F3:SCHEMATIC full-screen optical diagram tab |
| `cacaosim-CTRL` | `--mode` | `<fused|modular>` | fused | `deploy` | Default deployment engine mode inside TUI |
| `cacaosim-CTRL` | `--atmturb` | — | OFF | `deploy` | Enable milk-fpsexec-atmturb-mkwfs co-launch on deploy |
| `cacaosim-CTRL` | `--seeing` | `<arcsec>` | 0.7 | `deploy` | Atmospheric seeing [arcsec] when --atmturb is active |
| `cacaosim-CTRL` | `--theme` | `<name>` | dark | `display` | milk palette: dark, night, accessible, light, or nordic |
| `cacaosim-CTRL` | `--ascii` | — | OFF | `display` | Pure 7-bit ASCII borders, arrows, and spectrum bars |
| `cacaosim-CTRL` | `-p, --pretty` | — | OFF | `display` | Force 24-bit TrueColor ANSI output |
| `cacaosim-CTRL` | `-t, --text, --no-color` | — | OFF | `display` | Monochrome terminal output (no color escapes) |

