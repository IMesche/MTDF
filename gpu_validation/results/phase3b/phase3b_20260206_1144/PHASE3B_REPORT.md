# Phase 3b: Asymmetry and Detectability Diagnostics

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


Generated: 2026-02-06T11:44:43.438522
Runtime: 9.3s (0.2 min)

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

**voidfinder**: Wald p=0.716, bootstrap p=0.250 (Δγ=-0.0016±0.0045)

**revolver**: Wald p=0.491, bootstrap p=0.300 (Δγ=+0.0042±0.0061)

**vide**: Wald p=0.959, bootstrap p=0.250 (Δγ=+0.0004±0.0074)

## Test D: Geometry and Mode-Coupling Check

**voidfinder**: p_empirical=0.800 (20 mocks, observed Δγ=-0.0016, mock std=0.0032)

**revolver**: p_empirical=0.400 (20 mocks, observed Δγ=+0.0042, mock std=0.0047)

**vide**: p_empirical=0.900 (20 mocks, observed Δγ=+0.0004, mock std=0.0037)

## Test E: Null Signal Injection (SGC Recovery)

**voidfinder** (γ_NGC=0.0000):

- 0.5x: inject=+0.0000, recover=-0.0002±0.0022, bias=-0.0002, det@2σ=5%
- 1.0x: inject=+0.0000, recover=+0.0000±0.0018, bias=+0.0000, det@2σ=5%
- 1.5x: inject=+0.0000, recover=-0.0004±0.0022, bias=-0.0004, det@2σ=0%

**revolver** (γ_NGC=0.0050):

- 0.5x: inject=+0.0025, recover=+0.0027±0.0026, bias=+0.0003, det@2σ=15%
- 1.0x: inject=+0.0050, recover=+0.0051±0.0025, bias=+0.0001, det@2σ=40%
- 1.5x: inject=+0.0075, recover=+0.0077±0.0030, bias=+0.0002, det@2σ=70%

**vide** (γ_NGC=0.0027):

- 0.5x: inject=+0.0013, recover=+0.0013±0.0024, bias=-0.0001, det@2σ=0%
- 1.0x: inject=+0.0027, recover=+0.0033±0.0023, bias=+0.0006, det@2σ=15%
- 1.5x: inject=+0.0040, recover=+0.0049±0.0037, bias=+0.0009, det@2σ=40%

## Interpretation

1. **SGC sensitivity-limited?** See Test B detectability table and Test E recovery rates.
2. **Geometry explains asymmetry?** See Test D empirical p-values.
3. **Δγ statistically robust?** See Test C Wald and bootstrap p-values, Test A IPW-weighted results.

---
File hashes: {'Pantheon+SH0ES.dat': '1cb0fc379ef066af', 'Pantheon+SH0ES_STAT+SYS.cov': 'abf806d966485e64'}
