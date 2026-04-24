# Phase 6 Test C4: fsigma8 x Environment

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Goal

Compare MTDF and LCDM fsigma8(z) predictions against RSD measurements and void-specific growth estimates. Assess discrimination power of current data.

## Growth Modification

MTDF predicts mu(a) = 1 + 0.0793 * T(a), transitioning at z_t = 0.74.
At z=0: mu = 1.053 (5.3% Geff enhancement). This is **homogeneous** (function of scale factor only, not environment-dependent).

The ~1.5% difference in fsigma8 is the same in all environments in linear theory.

## Chi-squared Comparison

| Statistic | LCDM | MTDF | DOF |
|-----------|------|------|-----|
| RSD chi2 | 7.26 | 7.56 | 8 |
| RSD + voids chi2 | 7.61 | 8.10 | 9 |
| Delta chi2 | -- | +0.49 | -- |

## Per-Point Analysis

| Source | z | Observed | LCDM pred | MTDF pred | Pull (L) | Pull (M) | Prefers |
|--------|---|----------|-----------|-----------|----------|----------|---------|
| 6dFGS | 0.067 | 0.423+/-0.055 | 0.443 | 0.438 | -0.36 | -0.27 | MTDF |
| SDSS MGS | 0.150 | 0.530+/-0.160 | 0.458 | 0.453 | +0.45 | +0.48 | LCDM |
| BOSS z1 | 0.380 | 0.497+/-0.045 | 0.476 | 0.469 | +0.48 | +0.62 | LCDM |
| BOSS z2 | 0.510 | 0.459+/-0.038 | 0.474 | 0.467 | -0.39 | -0.20 | MTDF |
| BOSS z3 | 0.610 | 0.436+/-0.034 | 0.468 | 0.461 | -0.95 | -0.74 | MTDF |
| eBOSS LRG | 0.698 | 0.473+/-0.044 | 0.462 | 0.454 | +0.26 | +0.43 | LCDM |
| eBOSS ELG | 0.850 | 0.315+/-0.095 | 0.447 | 0.439 | -1.39 | -1.31 | MTDF |
| eBOSS QSO | 1.480 | 0.462+/-0.045 | 0.376 | 0.368 | +1.91 | +2.09 | LCDM |
| BOSS voids (Hamaus+20) | 0.570 | 0.501+/-0.051 | 0.471 | 0.464 | +0.59 | +0.73 | LCDM |

## Discrimination Power

| Source | z | Model diff | Data error | Diff/error |
|--------|---|------------|------------|------------|
| 6dFGS | 0.067 | 0.0050 | 0.055 | 0.092 |
| SDSS MGS | 0.150 | 0.0055 | 0.160 | 0.034 |
| BOSS z1 | 0.380 | 0.0066 | 0.045 | 0.146 |
| BOSS z2 | 0.510 | 0.0070 | 0.038 | 0.185 |
| BOSS z3 | 0.610 | 0.0073 | 0.034 | 0.215 |
| eBOSS LRG | 0.698 | 0.0075 | 0.044 | 0.171 |
| eBOSS ELG | 0.850 | 0.0078 | 0.095 | 0.083 |
| eBOSS QSO | 1.480 | 0.0084 | 0.045 | 0.186 |
| BOSS voids (Hamaus+20) | 0.570 | 0.0072 | 0.051 | 0.141 |

Maximum discrimination: 0.22 sigma (well below 1 sigma).
**Current RSD data cannot distinguish MTDF from LCDM in fsigma8.**

## Future Requirements

| z | Diff (%) | Error for 2sigma | Error for 5sigma | Current error | Improvement needed |
|---|----------|------------------|------------------|---------------|-------------------|
| 0.1 | 1.16% | 0.0026 | 0.0010 | 0.055 | 21x |
| 0.3 | 1.32% | 0.0031 | 0.0012 | 0.045 | 14x |
| 0.5 | 1.48% | 0.0035 | 0.0014 | 0.038 | 11x |
| 0.7 | 1.63% | 0.0038 | 0.0015 | 0.044 | 12x |
| 1.0 | 1.87% | 0.0040 | 0.0016 | 0.045 | 11x |
| 1.5 | 2.24% | 0.0042 | 0.0017 | 0.095 | 23x |

## Interpretation

The homogeneous mu(a) modification produces a ~1.5% shift in fsigma8(z), which is 5-10x smaller than current RSD error bars. Discrimination requires either:

1. **DESI/Euclid-era RSD** with ~0.003-0.005 precision per z-bin
2. **Environment-resolved estimators** (density-split fsigma8, void-galaxy cross-correlations) which may reveal nonlinear MTDF effects beyond the linear mu(a) prediction
3. **Combined multi-probe analysis** where the coherent 1.5% shift across many z-bins accumulates statistical weight

## Files

| File | Description |
|------|-------------|
| `testC4_fsigma8_environment.json` | Full analysis data |
| `testC4_fsigma8_environment.png` | Predictions vs data |
| `testC4_future_requirements.png` | Error requirements plot |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testC4_fsigma8_environment.py
```
