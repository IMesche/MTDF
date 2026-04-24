# Test 3: k_f Identifiability and Degeneracy Map

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Canonical file:** `test3_kf_identifiability.json`

This test checks whether k_f is effectively a reparameterisation of a single
nuisance or amplitude combination. We show (i) pairwise correlations,
(ii) principal 2D degeneracy contours, (iii) linear predictability of k_f
from nuisance parameters, and (iv) the A_s·e^{-2τ} degeneracy check.

## Key Results

| Diagnostic | Value | Verdict |
|---|---|---|
| Max \|r(k_f, nuisance)\| | 0.099 | PASS — negligible nuisance coupling |
| Max \|r(k_f, cosmo)\| | 0.301 (ω_b) | Moderate, not problematic |
| R²(nuisance → k_f) | 0.137 | PASS — not a nuisance reskin |
| R²(early-time → k_f) | 0.278 | Weakly constrained, as expected |
| r(k_f, A_eff) | +0.093 | PASS — not tracking A_s·e^{-2τ} |

## omega_b Partial Correlation

k_f shows a moderate covariance with ω_b (marginal r = −0.30; stable partial
r ≈ −0.34 conditioning on n_s and ω_cdm). Full conditioning yields an inflated
partial correlation (+0.97) because the conditioning matrix is ill-conditioned
(κ = 24,207; multiple predictors with VIF > 100). Both precision-matrix and
regression-residual methods agree at every conditioning level, confirming
this is a multicollinearity artefact rather than a physical one-to-one degeneracy.

| Conditioning set | Partial r (precision) | Partial r (residual) |
|---|---|---|
| n_s only | −0.372 | −0.372 |
| n_s, ω_cdm | −0.342 | −0.342 |
| n_s, ω_cdm, logA | −0.341 | −0.341 |
| All other cosmo (7 params) | +0.970 | +0.970 |

Conditioning matrix condition number: κ = 24,207. VIF > 100 for H0, σ8,
Ω_m, ω_cdm, logA.

## Files

| File | Description |
|---|---|
| `test3_kf_identifiability.json` | Full numerical results |
| `test3_correlation_heatmap.png` | Correlation matrix: k_f + 8 cosmo + 8 top nuisance |
| `test3_kf_2d_contours.png` | 2D posteriors: k_f vs σ8, H0, logA, τ, top nuisance |
| `test3_kf_vs_omega_b.png` | 2D posterior: k_f vs ω_b with partial correlation |
| `test3_kf_vs_Aeff.png` | Scatter: k_f vs A_eff = A_s·e^{-2τ} |
| `test3_kf_posterior_annotated.png` | 1D k_f posterior with prior and CIs |

## Source chain

`results/phase5/mtdf_mcmc.1.txt` (27,160 samples, R-1 = 0.019)

## Script

`mtdf_validation/phase5_plik/test3_kf_identifiability.py`
