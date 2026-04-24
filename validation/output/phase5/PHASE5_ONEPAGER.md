# Phase 5: Full Planck MCMC — One-Page Summary

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Dataset:** Planck 2018 plik TTTEEE + lowl TT + lowl EE + CMB lensing
**Solver:** `class_mtdf` (modified CLASS with EFE injection + mu(a) growth)
**Sampler:** Cobaya 3.6 MCMC (drag=True, 2 chains per model)
**Parameters:** 6 cosmological + 1 MTDF (k_f) + 21 Planck nuisance

---

## Convergence

| Metric | LCDM | MTDF |
|--------|------|------|
| Gelman-Rubin R-1 | 0.019 | 0.019 |
| Convergence threshold | < 0.02 | < 0.02 |
| R-1 (credible intervals) | 0.064 | 0.057 |
| Accepted samples | 26,400 | 27,160 |
| ESS (H0) | ~1,920 | ~1,450 |
| ESS (sigma8) | ~2,020 | ~1,600 |
| ESS (k_f) | -- | ~1,470 |

Both chains converged on 11 Feb 2026 (LCDM) and 10 Feb 2026 (MTDF).

---

## Key Comparison: MTDF vs LCDM

| Metric | LCDM | MTDF | Delta |
|--------|------|------|-------|
| chi2 (likelihood-only, best-fit) | 2773.20 | 2773.82 | **+0.63** |
| AIC (k=27 vs 28) | 2827.20 | 2829.82 | +2.63 |
| BIC (N~1676) | 2973.65 | 2981.70 | +8.05 |
| H0 (km s⁻¹ Mpc⁻¹) | 67.38 +/- 0.54 | 67.83 +/- 0.54 | +0.45 (0.6σ) |
| sigma8 (posterior mean) | 0.810 +/- 0.006 | 0.790 +/- 0.006 | **-0.020 (~2.4σ)** |
| S8 | 0.830 | 0.802 | **-0.028** |
| Omega_m | 0.315 +/- 0.007 | 0.309 +/- 0.007 | -0.006 |

All values are posterior means +/- 1σ. Sigma levels are posterior separation in Gaussian approximation: Δ / √(σ²_LCDM + σ²_MTDF). Δχ² = +0.63 is at the BOBYQA minimizer best-fit.
Note: chi2 values above refer to the likelihood-only chi-squared (as written in the *.minimum.txt outputs), not the posterior including prior penalties.

**Verdict:** MTDF is indistinguishable from LCDM at the CMB. Priority 1 hard falsifier cleared.

---

## k_f Posterior

| Statistic | Value |
|-----------|-------|
| Mean | 0.495 |
| Std dev | 0.360 |
| 68% CI | [0.136, 0.864] |
| 95% CI | [0.025, 1.342] |
| k_f = 0 in 95% CI | No (lower edge = 0.025; prior boundary) |
| **k_f = 1 in 95% CI** | **Yes** |

k_f = 1 (full MTDF) is allowed; k_f = 0 (LCDM) is not strongly excluded and sits near the lower boundary shaped by the k_f >= 0 prior. Planck does not constrain k_f tightly, as predicted for a late-time modification.

---
## Robustness (Phase 5)

Three targeted tests confirm k_f is real, stable, and not an artefact.
Full artefacts: `output/phase5/robustness/{kf_identifiability,test2_leave_one_out,test4_prior_sensitivity}/`.

**Identifiability (Test 3):** k_f is not degenerate with nuisance parameters or the amplitude combination A_s e^{-2τ}. Max |r(k_f, nuisance)| = 0.099; R²(nuisance → k_f) = 0.137 (86% of k_f variance is genuinely k_f, not nuisance reskin). Moderate coupling with ω_b (marginal r = −0.30, partial r = −0.34 under minimal conditioning) is physical, not multicollinearity — the ill-conditioned partial r = +0.97 seen when conditioning on all cosmological parameters simultaneously is a known conditioning artefact (condition number κ = 24,207) and is diagnosed, not hidden.

**Leave-one-out Planck stress (Test 2):** Dropping CMB lensing shifts k_f by +0.09σ and σ8 by +0.6σ — both within uncertainty. TT-only broadens the posterior (k_f shifts +1.2σ) as expected from constraint weakening, not a physical shift. Likelihood-only best-fit Δχ² (BOBYQA):

| Subset | Δχ² (MTDF − LCDM) | ΔAIC |
|--------|--------------------|------|
| Baseline (full plik) | +0.63 | +2.63 |
| No lensing | +0.41 | +2.41 |
| TT only | −0.65 | +1.35 |

**Prior sensitivity (Test 4):** Widening the k_f prior from [0, 5] to [0, 10] shifts the mean by −0.03σ. 95% CI: [0.023, 1.341] vs [0.025, 1.342]. KS test p = 0.12 (no significant difference). k_f is data-driven, not prior-driven.

---
## How to Reproduce

```bash
cobaya-run lcdm_mcmc.input.yaml
cobaya-run mtdf_mcmc.input.yaml
```

Requires: `class_mtdf` (commit 94cd5b6), Planck 2018 likelihoods via `cobaya-install`.
See `reproducibility.md` for full environment details.
