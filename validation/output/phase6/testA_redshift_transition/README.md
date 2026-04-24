# Phase 6 Test A: Redshift Transition Scan

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Goal

Test for a sharp late-time onset of the SN x void environment signal
around z ~ 0.04, as predicted by MTDF.  Extends the Phase 3 finding
("signal confined to z < 0.04") with an independent metric and formal
controls.

## Metrics

**M1 -- GLS delta-chi2 z-score (Phase 3 replication):**
Fit `mu_resid = intercept + gamma_env * d_signed + gamma_M * step(M>=10)`
via generalised least squares with full Pantheon+ covariance.
z-score = sign(gamma_env) * sqrt(delta-chi2).
Tests linear coupling of Hubble residuals to signed void proximity.

**M2 -- Spearman rho z-score (alternative):**
Spearman rank correlation between d_signed and Hubble residuals.
Non-parametric; tests monotonic (not necessarily linear) association.
Independent of covariance matrix assumptions.

## Null Hypothesis

H0: SN Ia Hubble residuals are independent of void proximity at all
redshifts.  Under H0, neither metric should show a z-dependent signal,
and the z-cut scan should be flat within the control bands.

## Controls

**C1 -- Shuffled environment labels (200 permutations):**
Randomly permute d_signed among all SNe, preserving z-structure.
Tests whether the specific SN-void pairing carries information.
**Global p over the scan:** for each permutation, the maximum GLS
z-score across *all* z_cut values is recorded. The global p is the
fraction of permutation maxima that equal or exceed the observed scan
maximum (3.62 sigma at z_cut = 0.030).
This corrects for the multiple-testing inherent in scanning 16
z_cut thresholds.

**C2 -- Random z-split (200 iterations, Spearman only):**
For each z_cut, randomly select N_low SNe (matching the real count)
regardless of redshift.  Tests whether the z < z_cut subset is special.

## Data

- Pantheon+ SH0ES (Brout et al. 2022): 564 SNe after cuts
- DESIVAST BGS REVOLVER voids (Douglass et al. 2023)
- z range: [0.02, 0.157]
- Seed: 42 (deterministic)

## Baseline Check (Phase 3 Replication)

Full sample: gamma_env = 0.00470 +/- 0.00228,
delta-chi2 = 4.24, p = 0.0394

## Result

| Metric | Peak z_cut | Peak sigma | Global p (scan-corrected) |
|--------|-----------|-----------|--------------------------|
| GLS (M1) | 0.030 | 3.62 | < 0.0050 |
| Spearman (M2) | 0.025 | 3.10 | -- |

Global p method: 0/200 permutation
scan-maxima exceeded the observed scan-maximum.

Transition confirmed: Yes

## Files

| File | Description |
|------|-------------|
| `phase6_testA_summary.json` | Full results: baseline, scan, controls, conclusion |
| `testA_zscan_plot.png` | Detection significance vs z_cut with control bands |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testA_redshift_transition.py --seed 42
```
