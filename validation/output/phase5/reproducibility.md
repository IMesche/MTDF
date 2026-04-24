# Phase 5 Reproducibility Notes

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Software Environment

| Component | Version / Identity |
|-----------|--------------------|
| Python | 3.12.3 |
| Cobaya | 3.6 |
| CLASS (MTDF) | `class_mtdf` commit `94cd5b65` |
| GCC | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04) |
| OS | Ubuntu 24.04 (WSL2, kernel 6.6.87.1) |
| MPI | Not used (single-node, 2 chains via Cobaya internal) |

## Hardware

- CPU: Desktop workstation
- GPU: NVIDIA RTX 4080 SUPER (16 GB VRAM, CUDA 12.9)
- Note: CLASS runs on CPU; GPU used for earlier phases only

## Planck Likelihoods

Installed via `cobaya-install cosmo -p cobaya_packages`:
- `planck_2018_lowl.TT_clik`
- `planck_2018_lowl.EE_clik`
- `planck_2018_highl_plik.TTTEEE`
- `planck_2018_lensing.native`

## Command Lines

**Minimization (BOBYQA, best of 4):**
```bash
cobaya-run lcdm_minimize.minimize.input.yaml
cobaya-run mtdf_minimize.minimize.input.yaml
```

**MCMC (2 chains each, drag sampling):**
```bash
cobaya-run lcdm_mcmc.input.yaml
cobaya-run mtdf_mcmc.input.yaml
```

## Convergence Criteria

- `Rminus1_stop: 0.02` (Gelman-Rubin R-1 threshold)
- `Rminus1_cl_stop: 0.2` (credible interval convergence)
- `drag: true` (fast-slow parameter splitting)
- `covmat: auto` (adaptive covariance matrix)

## Run Duration

- LCDM MCMC: ~4.5 days (6 Feb -- 11 Feb 2026)
- MTDF MCMC: ~4.5 days (6 Feb -- 10 Feb 2026)

## YAML Configs

The exact input configurations are provided as `lcdm_mcmc.input.yaml` and `mtdf_mcmc.input.yaml` in this directory.

## Full Chains

Raw chain files (lcdm_mcmc.1.txt ~19 MB, mtdf_mcmc.1.txt ~20 MB) are available on request. They are not included in the submission package due to size.
