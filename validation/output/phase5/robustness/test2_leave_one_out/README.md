# Test 2: Leave-One-Likelihood-Out Planck Stress Test

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Canonical file:** `test2_comparison_table.json`

This test checks whether the k_f posterior and key cosmological parameters
(σ₈, H₀, Ω_m) are stable when Planck likelihood subsets are removed.
Two configurations are compared against the full baseline:

1. **No lensing** — removes `planck_2018_lensing.native`; same 21 nuisance parameters
2. **TT only** — uses only `planck_2018_lowl.TT_clik` + `planck_2018_highl_plik.TT`;
   15 nuisance parameters (drops 6 `galf_TE_*` dust polarisation params)

Convergence target: R-1 < 0.1, N_accepted ≥ 2000 (relaxed from baseline's 0.02).

## Key Result

- **No lensing**: k_f shift = +0.09σ relative to baseline
- **TT only**: k_f shift = +1.23σ relative to baseline

The TT-only σ₈ shift (+3.1σ) reflects constraint weakening (posterior broadening), not a physical shift: TT alone carries less growth information than TTTEEE, so σ₈ reverts toward a less constrained value.

All k_f shifts are < 2σ → **k_f is not driven by any single Planck likelihood component.**

## Files

| File | Description |
|------|-------------|
| `test2_comparison_table.json` | Full numerical comparison |
| `test2_comparison_table.md` | Markdown-formatted table |
| `test2_kf_posteriors_overlay.png` | k_f posterior overlay (baseline + subsets) |
| `test2_sigma8_H0_summary.png` | σ₈ and H₀ dot plot by subset |
| `test2_delta_chi2_bar.png` | Best-fit chi-squared per configuration |

## Source chains

- Baseline: `mcmc_results/mtdf_mcmc` (27160 samples)
- No lensing: `mcmc_results/test2/mtdf_no_lensing` (4299 samples)
- TT only: `mcmc_results/test2/mtdf_TT_only` (7560 samples)

## Script

`mtdf_validation/phase5_plik/analyze_test2_test4.py`
