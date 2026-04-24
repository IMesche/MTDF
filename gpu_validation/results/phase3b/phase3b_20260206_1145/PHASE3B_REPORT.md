# Phase 3b: Asymmetry and Detectability Diagnostics

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


Generated: 2026-02-06T11:45:27.670536
Runtime: 20.0s (0.3 min)

## Test A: IPW Matched Redshift (Stabilised)

Logistic propensity model fitted to predict NGC/SGC from z, z^2, host_mass, and survey_id.
Weight truncation: 2.1% at [0.488, 2.015].
ESS: NGC=272 (94%), SGC=229 (83%).

**voidfinder**: Δγ_weighted = -0.0005 ± 0.0051 (z=-0.11, p=0.915)

**revolver**: Δγ_weighted = +0.0063 ± 0.0068 (z=0.92, p=0.355)

**vide**: Δγ_weighted = +0.0000 ± 0.0086 (z=0.01, p=0.995)

## Test B: Void-SN Joint Density Mapping

See `tables/test_b_detectability.csv` and plots for full details.

- **voidfinder_NGC**: 264 voids, 289 SNe, SN/void=1.09
- **voidfinder_SGC**: 35 voids, 275 SNe, SN/void=7.86
- **revolver_NGC**: 511 voids, 289 SNe, SN/void=0.57
- **revolver_SGC**: 82 voids, 275 SNe, SN/void=3.35
- **vide_NGC**: 383 voids, 289 SNe, SN/void=0.75
- **vide_SGC**: 58 voids, 275 SNe, SN/void=4.74

## Test C: Wald Test + Parametric Bootstrap

**voidfinder**: Wald p=0.716, bootstrap p=0.174 (Δγ=-0.0016±0.0045)

**revolver**: Wald p=0.491, bootstrap p=0.243 (Δγ=+0.0042±0.0061)

**vide**: Wald p=0.959, bootstrap p=0.179 (Δγ=+0.0004±0.0074)

## Test D: Geometry and Mode-Coupling Check

**voidfinder**: p_empirical=0.585 (200 mocks, observed Δγ=-0.0016, mock std=0.0030)

**revolver**: p_empirical=0.280 (200 mocks, observed Δγ=+0.0042, mock std=0.0040)

**vide**: p_empirical=0.900 (200 mocks, observed Δγ=+0.0004, mock std=0.0050)

## Test E: Null Signal Injection (SGC Recovery)

**voidfinder** (γ_NGC=0.0000):

- 0.5x: inject=+0.0000, recover=+0.0001±0.0021, bias=+0.0001, det@2σ=4%
- 1.0x: inject=+0.0000, recover=-0.0000±0.0021, bias=-0.0000, det@2σ=4%
- 1.5x: inject=+0.0000, recover=+0.0000±0.0022, bias=+0.0000, det@2σ=6%

**revolver** (γ_NGC=0.0050):

- 0.5x: inject=+0.0025, recover=+0.0025±0.0027, bias=-0.0000, det@2σ=10%
- 1.0x: inject=+0.0050, recover=+0.0048±0.0029, bias=-0.0002, det@2σ=44%
- 1.5x: inject=+0.0075, recover=+0.0076±0.0028, bias=+0.0002, det@2σ=76%

**vide** (γ_NGC=0.0027):

- 0.5x: inject=+0.0013, recover=+0.0012±0.0031, bias=-0.0002, det@2σ=6%
- 1.0x: inject=+0.0027, recover=+0.0028±0.0032, bias=+0.0001, det@2σ=16%
- 1.5x: inject=+0.0040, recover=+0.0041±0.0030, bias=+0.0001, det@2σ=22%

## Interpretation

1. **SGC sensitivity-limited?** See Test B detectability table and Test E recovery rates.
2. **Geometry explains asymmetry?** See Test D empirical p-values.
3. **Δγ statistically robust?** See Test C Wald and bootstrap p-values, Test A IPW-weighted results.

---
File hashes: {'Pantheon+SH0ES.dat': '1cb0fc379ef066af', 'Pantheon+SH0ES_STAT+SYS.cov': 'abf806d966485e64'}
