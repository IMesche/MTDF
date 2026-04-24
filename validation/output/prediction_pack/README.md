# MTDF Prediction Pack v1

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Date:** 2026-02-17
**Source:** class_mtdf + ODE growth solver with Phase 5 MCMC posterior
**Status:** Reproducible grids and first-pass summary plots. Key claims directionally
consistent, pending definition-level and likelihood-level comparisons.

## Overview

This prediction pack provides the numerical foundation for MTDF_07.
Two layers of prediction are computed:

**Layer 1 — CLASS MTDF (EFE only):** Direct output from class_mtdf with Phase 5
posterior parameters. The Early Field Energy modifies the background expansion
around recombination (sound horizon shift ~0.74%), which changes the transfer
function and shifts the Planck-constrained sigma8 from 0.810 (LCDM) to 0.790 (MTDF).
Note: the late-time growth modification mu(a) is NOT implemented in CLASS
perturbations (mtdf_growth='no' in Phase 5 MCMC).

**Layer 2 — Full MTDF (EFE + growth):** ODE growth solver with
mu(a) = 1 + 0.0793 * T(a), where T(a) transitions at z_t = 0.74.
This predicts enhanced structure growth at z < 1 (mu(z=0) = 1.0533,
i.e. 5.3% enhancement in effective G). Note: mu(a) is homogeneous (function
of scale factor only), not environment-dependent. Void-specific growth
estimators are alternative estimators, not a direct environment-dependence
prediction unless MTDF includes explicit environmental modelling.

## Cross-check: ODE vs CLASS

The LCDM growth ODE was validated against CLASS sigma8(z) at 8 redshifts:

| Quantity | Max deviation | Notes |
|----------|---------------|-------|
| D(z)/D(0) shape | 0.13% | ODE growth factor vs CLASS sigma8 ratio |
| f(z) growth rate | 0.28% | ODE vs CLASS numerical derivative |

Both are well within numerical tolerance. The ODE machinery is trustworthy
for the MTDF mu(a) extension.

The ~1.4% offset in absolute fsigma8 values is from sigma8(0) normalization:
CLASS computes 0.822 from the posterior-mean parameters, while the Phase 5
MCMC posterior mean is 0.810 (Jensen's inequality: mean(sigma8) != sigma8(mean(params))).

## Key Results

### S8

| Model | sigma8(0) | Omega_m | S8 |
|-------|-----------|---------|-----|
| LCDM (Phase 5) | 0.8101 | 0.3149 | 0.8300 |
| MTDF (Phase 5) | 0.7903 | 0.3090 | 0.8020 |
| KiDS-1000 | — | — | 0.759 +/- 0.024 |
| DES Y3 | — | — | 0.776 +/- 0.017 |
| Planck 2018 | — | — | 0.832 +/- 0.013 |

S8 definition used throughout: S8 = sigma8 * sqrt(Omega_m / 0.3).

MTDF S8 = 0.802 moves closer to KiDS and DES central values compared to
LCDM S8 = 0.830. Quantifying the sigma-level tension reduction requires
propagating the published covariances through a consistent likelihood,
which is deferred to MTDF_07.

### f*sigma8(z) predictions

| z | LCDM | MTDF (full) | Difference |
|---|------|-------------|------------|
| z=0.067 | 0.4431 | 0.4380 | -1.14% |
| z=0.150 | 0.4581 | 0.4526 | -1.20% |
| z=0.380 | 0.4755 | 0.4689 | -1.38% |
| z=0.510 | 0.4736 | 0.4666 | -1.48% |
| z=0.570 | 0.4708 | 0.4636 | -1.53% |
| z=0.610 | 0.4683 | 0.4610 | -1.56% |
| z=0.698 | 0.4617 | 0.4542 | -1.63% |
| z=0.850 | 0.4473 | 0.4394 | -1.75% |
| z=1.000 | 0.4309 | 0.4229 | -1.87% |
| z=1.480 | 0.3762 | 0.3678 | -2.22% |

The ~1.5% difference is well inside current RSD error bars (~5-10%).
C4 discriminating power requires either much tighter measurements or
environment-resolved estimators (e.g. density-split fσ8).

### P(k) sanity plot (class_mtdf EFE transfer function effect)

| z | Mean ratio | Max deviation |
|---|------------|---------------|
| z=0.0 | 1.00063 | 1.40% |
| z=0.5 | 1.00495 | 1.40% |
| z=1.0 | 1.00711 | 1.62% |
| z=2.0 | 1.00865 | 1.77% |

Note: pk_ratio_kf_band.png shows P(k) in Fourier space (h/Mpc),
NOT angular lensing band powers. A true lensing band-power comparison
requires projection and window functions (deferred).

## Growth modification mu(a)

mu(a) = 1 + amp * T(a) where:
- amp = (1 - beta_eos)^2 / (1 + alpha) = 0.07927
- T(a) = (a/a_t)^alpha / [1 + (a/a_t)^alpha]
- a_t = 1/(1 + z_t) = 0.5747

At z=0: mu = 1.0533 (5.3% effective G enhancement)
At z=z_t=0.74: mu = 1 + amp/2 = 1.0396
At z>>z_t: mu -> 1 (GR limit)

mu(a) is a homogeneous modification: it does not produce environment-dependent
growth in linear theory. Void-specific growth estimators (Hamaus+2020,
Paillas+2024) are nonetheless interesting comparators because they probe
growth in underdense regions where MTDF effects may differ nonlinearly.

## Phase 5 posterior parameters

### k_f (EFE amplitude)
- Mean: 0.495
- 68% CI: [0.136, 0.864]
- 95% CI: [0.025, 1.342]
- k_f = 0 (LCDM) and k_f = 1 (full MTDF) both within 95% CI

### Cosmological parameters (MTDF posterior mean)
- H0 = 67.83 km/s/Mpc
- omega_b = 0.022364
- omega_cdm = 0.119112
- log(10^10 A_s) = 3.0369
- n_s = 0.9681
- tau_reio = 0.0517

## Files

| File | Description |
|------|-------------|
| mtdf_prediction_pack.json | Full numerical grids and parameters |
| pk_ratio_kf_band.png | P(k) ratio sanity plot (Fourier space, not lensing band powers) |
| fsigma8_comparison.png | f*sigma8(z) vs RSD compilation + voids |
| S8_comparison.png | S8 bar chart comparison |
| manifest.json | SHA256 hashes for all files |
