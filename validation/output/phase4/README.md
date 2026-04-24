# Phase 4: Sensitivity Forecasts

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Purpose:** Forecast the survey size needed for definitive (5-sigma) detection of the SN x void environment signal, under baseline, realistic, and adversarial systematic scenarios.

**Dataset:** Monte Carlo resampling from Pantheon+ observed distributions

**Key result:** 5-sigma detection requires approximately 3,400 low-redshift SNe under baseline conditions. LSST/Rubin (10k+ SNe at low-z) reaches 5-sigma territory even under adversarial systematics.

**Canonical results file:** `phase4_summary.json`

**Referenced in:** MTDF_07 Section 5

## Files

| File | Description |
|------|-------------|
| `phase4_summary.json` | Full forecast results: detection power at 7 sample sizes (500-10000), three systematic scenarios (baseline, realistic, adversarial), consistency checks, and Phase 3 cross-validation |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes for all files in this folder |

## Key numbers in phase4_summary.json

- `consistency_checks.<catalogue>` — cross-validation against Phase 3 sigma_gamma values
- Look for `power_5sigma` entries to find the detection probability at each sample size
- Three scenarios: `baseline` (diagonal covariance), `realistic` (+ systematic floor), `adversarial` (+ rank-1 systematic aligned with d_signed)
