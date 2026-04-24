# Phase 5: Full Planck MCMC Results

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


This directory contains the curated output from Phase 5 of the MTDF independent GPU validation: a full Planck 2018 (plik TTTEEE + lowl + lensing) MCMC comparison between LCDM and MTDF using the `class_mtdf` modified Boltzmann solver.

**Canonical results files:** `phase5_mcmc_summary.json` (posteriors), `phase5_minimize_comparison.json` (best-fit chi-squared)

**Referenced in:** MTDF_07 Section 6

**Suggested reading order:** `PHASE5_ONEPAGER.md` → key JSON files → plots. For Phase 5 robustness and identifiability checks, see `robustness/`.

## Files

| File | Description |
|------|-------------|
| `PHASE5_ONEPAGER.md` | 60-second reviewer summary with all key numbers |
| `phase5_key_numbers.csv` | Machine-readable parameter comparison table |
| `phase5_mcmc_summary.json` | Full posterior summary (means, std, 68% CI, 95% CI for k_f) |
| `phase5_minimize_comparison.json` | Best-fit chi2 breakdown, AIC/BIC comparison |
| `phase5_kf_posterior.png` | k_f marginalised posterior plot |
| `phase5_triangle.png` | Corner plot of all cosmological parameters |
| `lcdm_mcmc.input.yaml` | Cobaya input config for LCDM MCMC run |
| `mtdf_mcmc.input.yaml` | Cobaya input config for MTDF MCMC run |
| `reproducibility.md` | Software versions, hardware, exact commands |
| `plots.md` | Plot descriptions and reading guide |
| `manifest.json` | SHA256 hashes for all files in this folder |
| `robustness/` | Robustness and identifiability tests (Tests 2-4): leave-one-out stable; prior-insensitive; k_f not nuisance proxy |

## Key Result

Delta-chi2 = +0.63 (MTDF vs LCDM). MTDF is indistinguishable from LCDM at the CMB. Both chains converged with R-1 < 0.02. k_f = 1 (full MTDF) lies within the 95% credible interval.

## Robustness

Robustness artefacts: `output/phase5/robustness/`

- **Test 3:** `kf_identifiability/` — k_f degeneracy and nuisance independence
- **Test 2:** `test2_leave_one_out/` — Planck likelihood subset stability
- **Test 4:** `test4_prior_sensitivity/` — k_f prior range sensitivity

These tests assess identifiability, likelihood subset stability, and prior sensitivity for Phase 5.

## Not included (available on request)

Raw Cobaya chains (~40 MB), covariance matrices, checkpoints, monitor logs.

## Related Documentation

- **MTDF_07**: Full validation paper (Phases 1-5) in `../../../papers/MTDF_07_Independent_GPU_Validation.html`
