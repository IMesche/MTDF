# Phase 6 Test C5: Cluster Dynamics vs Lensing Mass

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Goal

Compute the predicted mass discrepancy M_dyn/M_lens under MTDF's mu(a) modification for different coupling scenarios. Assess whether current cluster mass comparison data can detect the predicted offset.

## Theory

In modified gravity, dynamical mass (from velocity dispersions) and lensing mass (from gravitational light deflection) probe different gravitational potentials:

- **Dynamical mass:** M_dyn probes Psi (Newtonian potential, modified by mu)
- **Lensing mass:** M_lens probes (Psi + Phi)/2 (Weyl potential, modified by Sigma)

The relationship depends on the gravitational slip eta = Phi/Psi:
- Sigma = mu * (1 + 1/eta) / 2
- M_dyn/M_lens = mu / Sigma = 2 / (1 + 1/eta)

In MTDF: mu(z=0) = 1.053 (5.3% Geff enhancement).

## Coupling Cases

| Case | Assumption | M_dyn/M_lens at z=0 | Offset |
|------|------------|---------------------|--------|
| A | mu-only dynamics (Sigma=1) | 1.0533 | +5.33% |
| B | Equal coupling (Sigma=mu) | 1.0000 | 0.00% |
| C (eta=0.7) | Parameterized slip | 0.8235 | -17.65% |
| C (eta=0.9) | Parameterized slip | 0.9474 | -5.26% |
| C (eta=1.2) | Parameterized slip | 1.0909 | +9.09% |

## mu(a) Parameters

- Amplitude: (1 - beta_eos)^2 / (1 + alpha) = 0.07927
- Transition: z_t = 0.74, a_t = 0.5747
- mu(z=0) = 1.0533 (+5.33%)
- mu(z=0.25) = 1.0480 (+4.80%)

## Comparison with Published Data

| Survey | z_med | Measured | Error | MTDF (A) | SNR | Detectable? |
|--------|-------|----------|-------|----------|-----|-------------|
| Weighing the Giants (WtG) | 0.25 | 1.00 | 0.08 | 1.048 | 0.60 | No |
| CLASH (WL+dynamics) | 0.35 | 1.05 | 0.10 | 1.046 | 0.46 | No |
| LoCuSS | 0.23 | 0.95 | 0.12 | 1.048 | 0.40 | No |
| HeCS-omnibus | 0.15 | 1.08 | 0.15 | 1.050 | 0.33 | No |
| PSZ2 stacked (Planck SZ) | 0.18 | 0.76 | 0.05 | 1.049 | 0.99 | No | *Hydrostatic mass bias, not directly M_dyn/M_lens*

## Future Requirements

| z | mu(z) | Offset | Error for 2sigma | Error for 5sigma |
|---|-------|--------|------------------|------------------|
| 0.1 | 1.0511 | +5.11% | 2.56% | 0.0102 |
| 0.2 | 1.0490 | +4.90% | 2.45% | 0.0098 |
| 0.3 | 1.0471 | +4.71% | 2.35% | 0.0094 |
| 0.5 | 1.0434 | +4.34% | 2.17% | 0.0087 |
| 0.7 | 1.0402 | +4.02% | 2.01% | 0.0080 |
| 1.0 | 1.0361 | +3.61% | 1.80% | 0.0072 |

## Resolution: Gravitational Slip η = 1 (Confirmed)

**The key unknown from the original C5 analysis has been resolved.**

Test C5b computed η(k,z) = Φ(k,z)/Ψ(k,z) directly from class_mtdf transfer functions across k = 0.005–2 h/Mpc and z = 0–10. Result:

- **η = 1.000000** at all cluster-relevant scales (z ≤ 0.8)
- **Max |Δη| between MTDF and ΛCDM: 0.000000** — literally identical
- Tiny deviations (|η-1| ~ 10⁻⁵) appear only at z ≥ 5, from standard radiation anisotropic stress (present equally in both models)

This is because class_mtdf modifies only the Poisson equation source (δρ → μ·δρ) and the CDM Euler equation, without touching the Φ−Ψ anisotropy equation. Both potentials are enhanced equally → Σ = μ → **Case B applies**.

**Resolved verdict:** M_dyn/M_lens = 1.000 under MTDF. The cluster mass ratio channel is not a discriminator. Case A was a theoretical bracket that is ruled out by the actual implementation.

See: `output/phase6/testC5b_gravitational_slip/` for the numerical verification.

## Interpretation

- **Case B (confirmed):** M_dyn/M_lens = 1 exactly. Cluster data provides no constraint on MTDF through this channel.
- **Case A (ruled out):** Would have required η ≠ 1 (gravitational slip). The class_mtdf implementation has η = 1 by construction.
- **Planck SZ mass bias (1-b = 0.76):** MTDF does not contribute to this through the M_dyn/M_lens channel. The hydrostatic mass bias must have other origins.
- **Growth and abundance:** Clusters can still test MTDF through growth-dependent observables (e.g. cluster counts at high z), because μ > 1 modifies the growth rate. Only the mass ratio channel is null.

**Verdict:** Resolved. Case B confirmed numerically. Cluster M_dyn/M_lens is not a discriminator for MTDF.

## Files

| File | Description |
|------|-------------|
| `testC5_cluster_mass.json` | Full prediction data |
| `testC5_cluster_mass.png` | M_dyn/M_lens vs z |
| `testC5_eta_scan.png` | Gravitational slip scan |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testC5_cluster_mass.py
```
