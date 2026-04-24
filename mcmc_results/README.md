# Phase 5 MCMC Results

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


This directory contains the configuration files and summary results from Phase 5:
full Planck 2018 plik TTTEEE + low-ell + lensing MCMC analysis comparing MTDF and LCDM.

## Results Summary

- **Delta-chi2**: +0.63 (MTDF vs LCDM, indistinguishable)
- **Delta-AIC**: +2.63 (slight LCDM preference due to extra k_f parameter)
- **k_f 95% CI**: includes both 0 (pure LCDM) and 1 (full MTDF)
- **sigma8**: 0.810 (LCDM) vs 0.790 (MTDF), easing the S8 tension
- **Convergence**: Gelman-Rubin R-1 < 0.02, 26,000+ accepted samples per chain

## Files

| File | Description |
|------|-------------|
| `lcdm_mcmc.input.yaml` | LCDM MCMC configuration for cobaya |
| `mtdf_mcmc.input.yaml` | MTDF MCMC configuration for cobaya |
| `*_mcmc.updated.yaml` | Updated configs with final settings |
| `*_minimize.*.yaml` | Minimiser configurations |
| `phase5_mcmc_summary.json` | Final comparison summary (chi2, AIC, parameters) |
| `phase5_minimize_comparison.json` | Best-fit parameter comparison |
| `validation_singlepoint.json` | Single-point validation results |
| `*_timing.json` | Run timing information |

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

The summary JSON files contain the final convergence diagnostics, parameter posteriors, and model comparison statistics. See `gpu_validation/phase5_plik/` for the analysis scripts.
