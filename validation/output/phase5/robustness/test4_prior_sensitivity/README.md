# Test 4: Prior Range Sensitivity for k_f

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Canonical file:** `test4_prior_sensitivity.json`

This test checks whether the k_f posterior changes when the uniform prior
is widened from [0, 5] to [0, 10]. If the posterior is data-informed,
doubling the prior range should leave the posterior shape essentially unchanged.

## Key Result

k_f mean shift: -0.03σ (baseline σ units).
Kolmogorov-Smirnov distance: D = 0.0148 (p = 0.1175).

**Decision:** k_f posterior is **prior-insensitive**.

## Files

| File | Description |
|------|-------------|
| `test4_prior_sensitivity.json` | Baseline vs wide prior comparison |
| `test4_kf_prior_overlay.png` | Posterior overlay with prior rectangles and KS test |

## Source chains

- Baseline: `mcmc_results/mtdf_mcmc` (27160 samples, prior [0.0, 5.0])
- Wide prior: `mcmc_results/test4/mtdf_wide_prior` (8400 samples, prior [0.0, 10.0])

## Script

`mtdf_validation/phase5_plik/analyze_test2_test4.py`
