# MTDF Modifications to CLASS v3.2

This directory contains a modified version of the CLASS Boltzmann solver (Lesgourgues & Tram, http://class-code.net) implementing the Mesche Tensor Dynamics Framework (MTDF).

## Base Version

CLASS v3.2.0 (Cosmic Linear Anisotropy Solving System)
Authors: Julien Lesgourgues, Thomas Tram, Nils Schoeneberg et al.

## Summary of Changes

Two physical modifications are made:

1. **Early Field Energy (EFE) injection** in the background equations: a log-Gaussian energy density component peaking at z ~ 3400 (matter-radiation equality) adds ~0.33% extra energy, reducing the sound horizon by ~0.03%.

2. **mu(a) growth modification** in the perturbation equations: at late times (a > a_t), the effective gravitational coupling for CDM is enhanced by up to ~8%, modifying structure growth.

## New Files

| File | Lines | Description |
|------|-------|-------------|
| `source/mtdf.c` | 274 | Core MTDF physics module |
| `include/mtdf.h` | 179 | MTDF type definitions and function prototypes |
| `cobaya/mtdf_classy.py` | 130 | Cobaya theory wrapper for MCMC sampling |
| `mtdf.ini` | 6 | Full MTDF configuration |
| `mtdf_noefe.ini` | 21 | MTDF with EFE disabled (comparison) |
| `test_mtdf.ini` | 6 | Minimal test configuration |

## Modified Files

### `include/background.h`
- Added `#include "mtdf.h"` (line 7)
- Added background vector indices: `index_bg_rho_mtdf_efe`, `index_bg_w_mtdf_efe` (lines 174-175)
- Added `has_mtdf` flag and `struct mtdf_parameters mtdf` to background structure (lines 297-305)

### `source/background.c`
- **EFE density injection** (lines 575-611): After standard species densities are summed, MTDF EFE density is injected into rho_tot and p_tot before the Friedmann equation is evaluated
- **Index allocation** (lines 1140-1142): Two new background vector indices registered conditional on `has_mtdf`

### `source/input.c`
- **Parameter parsing** (lines 3375-3433): Reads MTDF parameters from .ini files: `mtdf` (master switch), `mtdf_alpha`, `mtdf_beta_eos`, `mtdf_z_t`, `mtdf_k_f`, `mtdf_z_peak`, `mtdf_sigma_z`, `mtdf_efe`, `mtdf_growth`. Calls `mtdf_init()` and `mtdf_print_parameters()`.

### `source/perturbations.c`
- **Einstein equation** (lines 6548-6619): h' equation in synchronous gauge multiplied by mu(a)
- **Perturbation ODE derivatives** (lines 8768-8867): mu(a) computed for perturbation evolution
- **CDM velocity equation** (lines 9243-9244): Gravitational acceleration enhanced by mu(a) in Newtonian gauge

### `Makefile`
- Added `mtdf.o` to the SOURCE list (line 114)

## Unmodified Modules

The following CLASS modules were NOT modified: thermodynamics, lensing, transfer, fourier, harmonic, primordial, distortions. The Python wrapper (`classy.pyx`, `cclassy.pxd`) was also not modified; MTDF parameters pass through the standard CLASS parameter interface.

## MTDF Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mtdf` | no | Master switch (yes/no) |
| `mtdf_alpha` | 1.30 | Stress-matter coupling |
| `mtdf_beta_eos` | 0.573 | EOS transition parameter |
| `mtdf_z_t` | 0.74 | Late-time transition redshift |
| `mtdf_k_f` | 1.0 | EFE amplitude scale (0 = LCDM, 1 = full MTDF) |
| `mtdf_z_peak` | 3400.0 | EFE profile peak redshift |
| `mtdf_sigma_z` | 0.5 | EFE profile width in ln(1+z) |
| `mtdf_efe` | yes | Enable EFE background injection |
| `mtdf_growth` | yes | Enable mu(a) perturbation modification |

## Derived Quantities (computed in mtdf_init)

- `lambda_mtdf = (1 - beta_eos)^2 / (1 + alpha)` ~ 0.079
- `f_kick = lambda_mtdf / 24` ~ 0.0033 (EFE amplitude)
- `a_t = 1 / (1 + z_t)` (transition scale factor)

## Building

```bash
make clean && make -j4
```

Produces the `class` executable and `libclass.a` library. The Python wrapper can be installed via:

```bash
cd python && pip install .
```
