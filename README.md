# ImpactX/OSIRIS to GENESIS4 Beam Resampling

Convert a **weighted electron-beam particle distribution** from codes such as OSIRIS or IMPACTX into a **slice-based GENESIS4 beam** for time-dependent FEL/SASE simulations.

The resampling procedure is designed to:

- keep a fixed number of macroparticles in every GENESIS slice;
- preserve the local weighted phase-space properties of the source beam;
- suppress artificial numerical sampling noise;
- restore the physical microscopic shot noise required for SASE startup; and
- generate multiple statistically independent GENESIS input beams from the same macroscopic electron-beam distribution.

Repository: <https://github.com/Xuan0533/ImpactX2Genesis.git>

## Method overview

The code performs four main operations:

1. **Quiet longitudinal slicing and weighted resampling**
2. **Small transverse smoothing**
3. **Physical longitudinal shot-noise loading**
4. **Slice-wise momentum covariance rematching**

The final particle distribution is written in GENESIS4 sliced HDF5 format together with seed-specific `genesis4.in` and `genesis4.lat` files.

---

## 1. Quiet slice construction

The resonant wavelength is calculated as

```text
lambda_r = lambda_u / (2 gamma_0^2) * (1 + a_u^2)
```

and the longitudinal numerical slice spacing is

```text
Delta_s = lambda_r / SlicesMultiplyFactor
```

where `SlicesMultiplyFactor` is the number of numerical slices per resonant wavelength.

Within each numerical slice, `NumOfSliceParticles` particles are placed on a low-discrepancy Halton distribution. The source particle used for each new macroparticle is selected from the **same source slice** with probability proportional to its particle weight,

```text
P_i = w_i / sum(w)
```

The selected particle provides the provisional

```text
x, y, px, py, pz
```

while its longitudinal coordinate is replaced by the quiet coordinate assigned to the output slice.

The output slice charge is obtained from a smoothed interpolation of the original weighted longitudinal charge profile.

---

## 2. Transverse smoothing

Because weighted resampling is performed with replacement, the same source particle can be selected multiple times. This can produce duplicated transverse coordinates.

A small Halton-based displacement is therefore applied to `x` and `y` only:

```text
x_new = x_sampled + dx
y_new = y_sampled + dy
```

The displacement scale is determined from the weighted RMS beam size of the corresponding source slice.

The default value is

```python
XY_SMOOTHING = 0.05
```

so the smoothing amplitude is approximately 5% of the local RMS transverse size.

The momentum coordinates are not randomized during this step.

---

## 3. Longitudinal shot-noise loading

The quiet distribution intentionally suppresses microscopic density fluctuations. Physical SASE startup noise is therefore added explicitly after resampling.

The beam is divided into cells of one resonant wavelength. For each cell,

```text
N_real = sum(macroparticle weights)
```

is the number of real electrons represented by that wavelength interval.

For each harmonic `m`, a complex Gaussian bunching coefficient is generated such that

```text
<|b_m|^2> = 1 / N_real
```

which gives the expected shot-noise scaling for independent electrons.

The current implementation loads harmonics

```text
m = 1, 2, 3, 4
```

using a phase probability density followed by weighted inverse-CDF sampling.

An important implementation choice is that the noise changes the **microscopic longitudinal phase** without changing the particle's original quiet numerical-slice assignment. The quiet longitudinal coordinate is retained for final slice ordering and for momentum rematching.

---

## 4. Momentum covariance rematching

Transverse smoothing and finite resampling can slightly modify the coordinate-momentum correlations of the beam. The code therefore rematches the momenta independently within each numerical slice while keeping the new particle coordinates fixed.

For the production path,

```text
q = (x, y, z)
p = (px, py, pz)
```

and the momentum is decomposed as

```text
p = <p> + B (q - <q>) + r
```

where `B` describes the linear coordinate-momentum correlation and `r` is the residual momentum component.

The target regression matrix is chosen so that the new coordinates reproduce the source coordinate-momentum cross covariance. The existing residuals are then linearly transformed so their covariance reproduces the remaining source momentum covariance.

The final momentum has the form

```text
p_new = <p_src> + B_target (q_new - <q_new>) + A r_old
```

Matrix square roots and inverse square roots are obtained from symmetric eigendecomposition, with small eigenvalues regularized for numerical stability.

The shot-noise displacement is **not** used in this rematching step.

---

## Requirements

### Python packages

```bash
pip install numpy scipy h5py
```

The script also uses Python's standard `concurrent.futures.ProcessPoolExecutor` for parallel slice generation.

### Required repository files

The working directory should contain

```text
JDF_NLIST.py          # main resampling script
PARAMS_JDF.py         # user parameters
GENESIS_write.py      # GENESIS input/lattice parser and writer
genesis4.in           # template GENESIS input
genesis4.lat          # template GENESIS lattice
```

`GENESIS_write.py` must provide

```python
parse_config_file
parse_beamline_file
config_to_string
beamline_to_string
```

---

## Input beam format

The input HDF5 file must contain the following datasets:

| Dataset | Description |
|---|---|
| `x` | horizontal position |
| `y` | vertical position |
| `s` | longitudinal/co-moving coordinate |
| `dxdz` | horizontal trajectory slope |
| `dydz` | vertical trajectory slope |
| `gamma` | Lorentz factor |
| `weight` | number of real electrons represented by each source macroparticle |

The script converts `dxdz`, `dydz`, and `gamma` into momentum components before resampling.

---

## Configuration

Parameters are read from `PARAMS_JDF.py`.

| Parameter | Default | Description |
|---|---:|---|
| `lambda_u` | `0.03` | Undulator period [m] |
| `a_u` | `1.0121809` | Undulator parameter used for resonance and lattice update |
| `n_cpu` | `8` | GENESIS CPU count used for output-slice padding |
| `SlicesMultiplyFactor` | `10` | Numerical slices per resonant wavelength |
| `NumOfSliceParticles` | `800` | Macroparticles per numerical slice |
| `BeamStretchFactor` | `0.0` | Longitudinal interpolation-range extension |
| `OUT_DIR` | `run_test` | Output directory |
| `NumOfSeeds` | `1` | Number of independent shot-noise seeds |
| `StartOfSeeds` | `0` | First seed number |
| `RAWFile` | unset | Input HDF5 file when not supplied on the command line |
| `AvgBeamSize` | `1e-5` | Read by the script but not used in the current resampling path |

Several values are currently set directly in the main script:

```python
XY_SMOOTHING = 0.05
MAX_WORKERS_CAP = 32
gamma_0 = 11130
```

The shot-noise routine is currently called with

```python
n_harmonics = 4
n_grid = 1024
keep_local_mean = True
max_dtheta = None
```

### Example `PARAMS_JDF.py`

```python
lambda_u = 0.03
a_u = 2.6
n_cpu = 120

SlicesMultiplyFactor = 10
NumOfSliceParticles = 16384
BeamStretchFactor = 0.0

NumOfSeeds = 10
StartOfSeeds = 0

OUT_DIR = "run_resampled"
RAWFile = "beam_input.h5"
```

---

## Running the converter

Supply the source beam directly:

```bash
python JDF_NLIST.py beam_input.h5
```

or define

```python
RAWFile = "beam_input.h5"
```

in `PARAMS_JDF.py` and run

```bash
python JDF_NLIST.py
```

For each random seed the code will:

1. construct the longitudinal slice grid;
2. resample the source distribution slice by slice;
3. apply small transverse smoothing;
4. add wavelength-scale shot noise;
5. rematch the momentum covariance;
6. write the GENESIS beam HDF5 file;
7. write the corresponding GENESIS input and lattice files.

---

## Output structure

For example, with `a_u = 2.6`:

```text
OUT_DIR/
└── Planar_AW2.6/
    ├── BeamInputs/
    │   ├── seed_0.h5
    │   ├── seed_1.h5
    │   └── ...
    ├── 0/
    │   ├── genesis4.in
    │   └── genesis4.lat
    ├── 1/
    │   ├── genesis4.in
    │   └── genesis4.lat
    └── ...
```

Each `seed_*.h5` contains GENESIS slice groups such as

```text
slice000001
slice000002
...
```

with datasets

```text
current
x
y
theta
px
py
gamma
```

The file also stores slice metadata such as `slicecount`, `slicelength`, `slicespacing`, and `n_part`.

If the physical number of slices is not divisible by `n_cpu`, copies of the final slice are appended so that the total GENESIS slice count is compatible with the requested process count.

---

## GENESIS template updates

For each generated seed, the script modifies the parsed GENESIS templates, including the undulator `aw`, reference energy, particle count, random seed, imported beam path, and simulation length.

### Current `lambda0` convention

The present implementation sets

```python
main_input['setup']['lambda0'] = StepZ
```

with

```text
StepZ = lambda_r / SlicesMultiplyFactor
```

rather than setting `lambda0` directly to `lambda_r`. This README documents the current code behavior; verify that this convention is appropriate for the GENESIS setup being used.

---

## Shot-noise diagnostic

After writing each seed, the code evaluates the fundamental bunching over resonant-wavelength blocks and, when available, prints

```text
Measured <|b1|^2> = ...
```

The expected physical scaling is approximately

```text
<|b1|^2> ~ 1 / N_real
```

where `N_real` is the number of real electrons in one resonant-wavelength interval.

This provides a useful check that the microscopic noise level is determined by the represented physical charge rather than the number of simulation macroparticles.

---

## Recommended validation

Before production FEL simulations, compare the original and resampled beams slice by slice. Recommended checks include:

- current profile;
- mean energy and slice energy spread;
- RMS `x` and `y` beam sizes;
- transverse emittance and Twiss parameters;
- coordinate-momentum correlations;
- representative `x`, `y`, `px`, `py`, and `gamma` distributions;
- first-harmonic bunching and its `1/N_real` shot-noise scaling.

For a multi-seed ensemble, the macroscopic slice properties should remain essentially unchanged while the microscopic shot-noise realization changes from seed to seed.

---

## Numerical notes

- Invalid source values and non-positive source weights are removed before weighted sampling.
- Empty source slices fall back to global weighted mean coordinates and momenta.
- Full 3D covariance rematching is used in the main production path with `mode="full"` and `reg=1e-14`.
- Slice generation is parallelized with at most `MAX_WORKERS_CAP = 32` worker processes.
- The resampled coordinates remain associated with their quiet slices even after microscopic shot-noise phases are loaded.

---

## Physical interpretation

The method separates the beam into two different scales:

- **macroscopic slice phase space**, inherited from the weighted source beam; and
- **microscopic longitudinal density fluctuations**, imposed according to physical electron shot noise.

The quiet loading prevents numerical macroparticle noise from dominating the SASE startup. The shot-noise step then introduces the desired wavelength-scale density fluctuations, while covariance rematching restores the local phase-space statistics affected by resampling.

This makes the generated distributions suitable for time-dependent SASE simulations and multi-seed statistical studies in GENESIS4.

---

## License

Add the repository license here.
