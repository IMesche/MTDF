# MTDF CLASS Implementation v1

## Overview

This is a modified fork of CLASS (Cosmic Linear Anisotropy Solving System) implementing the
**Mesche Tensor Dynamics Framework (MTDF)** cosmological model.

MTDF replaces cold dark matter and dark energy with a unified stress tensor field. The model
has been validated against late-time observables and shows consistency with Planck CMB data.

## Key Features

### 1. Early Field Energy (EFE) Injection
- Log-Gaussian energy density profile peaking near matter-radiation equality (z ~ 3400)
- Reduces sound horizon at non-zero k_f (amount scales with k_f)
- Implemented in `source/mtdf.c` via `mtdf_omega_efe()` and `mtdf_rho_efe()`

### 2. Late-Time Gravitational Modification mu(a)
- Modified gravity coupling: mu(a) = 1 + lambda_MTDF * T(a/a_t)
- lambda_MTDF = (1 - beta_eos)^2 / (1 + alpha) ~ 0.0793
- Affects structure growth and CMB lensing
- Implemented in `source/perturbations.c`

## MTDF Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mtdf_alpha` | 1.30 | Stress-matter coupling |
| `mtdf_beta_eos` | 0.573 | EoS transition parameter |
| `mtdf_z_t` | 0.74 | Late-time transition redshift |
| `mtdf_k_f` | 1.0 | EFE amplitude scaling (MCMC parameter) |
| `mtdf_efe` | yes/no | Enable EFE injection |
| `mtdf_growth` | yes/no | Enable mu(a) modification |

## Files Modified from CLASS v3.2

### Source Files
- `source/mtdf.c` - New file: MTDF physics functions
- `source/background.c` - EFE density added to background evolution
- `source/perturbations.c` - mu(a) modification to growth equations
- `source/input.c` - MTDF parameter parsing

### Header Files
- `include/mtdf.h` - New file: MTDF structures and function declarations
- `include/background.h` - MTDF structure added to background struct

### Build System
- `Makefile` - Added mtdf.c compilation

## Configuration Files

- `mtdf.ini` - Full MTDF configuration with EFE and mu(a)
- `mtdf_noefe.ini` - MTDF without EFE (late-time only)
- `lcdm_baseline.ini` - Standard LCDM for comparison

## Validation Scripts

All scripts are in the `scripts/` directory:

| Script | Purpose | Output |
|--------|---------|--------|
| `task1_lcdm_limit_check.py` | Verify k_f=0 recovers LCDM | Chi-squared comparison |
| `task2_fede_profile.py` | EFE density profile f_EDE(z) | `output/mtdf_fEDE_profile.txt` |
| `task3_fsigma8_check.py` | Growth rate validation | `output/mtdf_fsigma8_comparison.txt` |
| `mtdf_joint_TTTEEE_BAO.py` | **Main MCMC**: Planck TTTEEE + DESI BAO | `output/mtdf_joint_TTTEEE_BAO_mu_results.txt` |
| `lcdm_joint_TTTEEE_BAO.py` | LCDM baseline MCMC | `output/lcdm_joint_TTTEEE_BAO_results.txt` |

## Key Results (Phase 5, V74)

### MTDF vs LCDM Comparison (Full Planck plik TTTEEE + lowl + lensing)

| Model | H0 (km/s/Mpc) | k_f (mean) | Total chi^2 | plik TTTEEE chi^2 |
|-------|---------------|------------|-------------|-------------------|
| LCDM | 67.38 +/- 0.54 | - | 2773.20 | 2345.41 |
| MTDF (with mu) | 67.83 +/- 0.54 | 0.495 +/- 0.360 | 2773.82 | 2341.74 |

Delta-chi2 = +0.63 (statistically indistinguishable). Delta-AIC = +2.63.

### Key Findings
1. **k_f unconstrained by Planck alone**: 95% CI = [0.025, 1.34]; both k_f ~ 0 and k_f = 1 consistent
2. **High-ell TTTEEE marginally improved**: Delta-chi2(plik) = -3.67 for MTDF
3. **S8 tension eased**: sigma8 shifts from 0.810 (LCDM) to 0.790 (MTDF), toward weak-lensing values
4. **mu(a) effect robust**: Growth modification self-consistent with CMB lensing

## Building

```bash
# Standard CLASS build
make clean
make class

# Build Python wrapper
cd python && python setup.py build_ext --inplace && cd ..
pip install -e .
```

## Usage

```bash
# Run with MTDF enabled
./class mtdf.ini

# Run LCDM baseline
./class lcdm_baseline.ini
```

## Python Example

```python
from classy import Class

cosmo = Class()
cosmo.set({
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'mtdf_alpha': 1.3,
    'mtdf_beta_eos': 0.573,
    'mtdf_z_t': 0.74,
    'mtdf_k_f': 1.0,
    'mtdf_efe': 'yes',
    'mtdf_growth': 'yes',
    # ... other cosmological parameters
})
cosmo.compute()

# Get observables
cls = cosmo.lensed_cl(2500)
r_d = cosmo.rs_drag()
```

## Dependencies

- CLASS v3.2.0 base
- Python 3.10+
- NumPy, SciPy, emcee
- clik (Planck likelihood)
- cobaya (optional, for advanced MCMC)

## References

- MTDF Theory: Mesche Tensor Dynamics Framework V74
- CLASS: https://github.com/lesgourg/class_public
- Planck 2018: arXiv:1807.06209
- DESI BAO 2024: arXiv:2404.03002

## Version History

- **v1** (2024-11-28): Initial validated implementation
  - EFE injection in background
  - mu(a) modification in perturbations
  - Full Planck TTTEEE + DESI BAO MCMC validation

## License

CLASS is distributed under the GPL v3 license. MTDF modifications follow the same license.
