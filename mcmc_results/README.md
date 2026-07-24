# Phase 5 MCMC Results

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V75

This directory contains the configuration files and summary results from Phase 5:
full Planck 2018 plik TTTEEE + low-ell + lensing MCMC analysis comparing MTDF and LCDM.

## Current Results (sealed v2 chains, corrected covariant mu(a), Newtonian gauge)

The authoritative summary is **`phase5_v2_sealed_summary.json`** in this directory.
Headline numbers:

- **sigma8** = 0.8265 +/- 0.0062
- **Native S8** = 0.8473 +/- 0.0138 (a sharp higher-amplitude prediction; posture **S8-WORSENED** versus published weak-lensing compressions, adjudication preregistered)
- **k_f** = 0.685 +/- 0.500 (bounded best fit at k_f = 0.476): Planck alone cannot pin the coupling amplitude (consistency, not support)
- **H0** = 67.44 +/- 0.55, unchanged versus the matched LCDM control
- **Delta-chi2 (bounded best fits, matched LCDM control)** = -1.29; **Delta-AIC** = +0.71. This is a maximum-likelihood comparison, not a Bayes factor or Bayesian evidence calculation.
- **Convergence**: 4 chains, 200,000 accepted samples each, R-1 = 0.0085 at stop

The full v2 and matched LCDM control chain tarballs are attached to the Zenodo
record for this release:

| Archive | SHA256 |
|---------|--------|
| `mtdf_v2_output.tar.gz` | `95ce9385153c7c40cf0f125d5a3cb5964751ac1ba4cb8d4c758da1b036610975` |
| `lcdm_v2_output.tar.gz` | `0d51dbca41f4bcbb8d471174c719dee4c529e03fa6ecf4083103ae024c84ee63` |

## Superseded v1 chains (retained for the audit trail)

> **SUPERSEDED.** The v1 numbers below are **retired**: they were produced by an
> implementation with a gauge artifact in the growth sector. The corrected
> implementation (covariant mu(a), Newtonian gauge) **reverses the direction**
> of the result: the retired v1 sigma8 = 0.790 / Delta-chi2 = +0.63 became the
> sealed v2 sigma8 = 0.8265 / Delta-chi2 = -1.29 with posture S8-WORSENED. The
> v1 JSON files (`phase5_mcmc_summary.json`, `phase5_minimize_comparison.json`,
> `validation_singlepoint.json`, `*_timing.json`) are kept unmodified in this
> directory as audit-trail artifacts; this README carries the supersession
> banner for all of them. Do not quote v1 numbers as current results.

Retired v1 summary (for the audit trail only):

- Delta-chi2 = +0.63 (RETIRED, gauge artifact)
- Delta-AIC = +2.63 (RETIRED)
- k_f 95% CI included both 0 and 1 (superseded by the v2 posterior above)
- sigma8: 0.810 (LCDM) vs 0.790 (MTDF), read at the time as easing the S8 tension (RETIRED; v2 reverses the direction)
- Convergence: Gelman-Rubin R-1 < 0.02, 26,000+ accepted samples per chain

## Files

| File | Status | Description |
|------|--------|-------------|
| `phase5_v2_sealed_summary.json` | **CURRENT** | Sealed v2 results summary (posteriors, model comparison, chain archive checksums) |
| `lcdm_mcmc.input.yaml` | Config | LCDM MCMC configuration for cobaya |
| `mtdf_mcmc.input.yaml` | Config | MTDF MCMC configuration for cobaya |
| `*_mcmc.updated.yaml` | Config | Updated configs with final settings |
| `*_minimize.*.yaml` | Config | Minimiser configurations |
| `phase5_mcmc_summary.json` | Superseded (v1) | Retired v1 comparison summary (gauge artifact; see banner above) |
| `phase5_minimize_comparison.json` | Superseded (v1) | Retired v1 best-fit parameter comparison |
| `validation_singlepoint.json` | Superseded (v1) | Retired v1 single-point validation results |
| `*_timing.json` | Superseded (v1) | v1 run timing information |

## Reproducing the MCMC Runs

### Prerequisites

1. Install cobaya and the Planck 2018 likelihood:

```bash
bash scripts/install_cobaya.sh
```

2. Build class_mtdf:

```bash
cd class_mtdf
make clean && make -j4
```

### Running

```bash
# LCDM baseline
cobaya-run mcmc_results/lcdm_mcmc.input.yaml

# MTDF (with k_f parameter)
cobaya-run mcmc_results/mtdf_mcmc.input.yaml
```

Each chain takes approximately 24-48 hours on a modern CPU with 4+ cores.

### Analysis

`phase5_v2_sealed_summary.json` contains the current convergence diagnostics,
parameter posteriors, and model comparison statistics. See
`gpu_validation/phase5_plik/` for the analysis scripts.
