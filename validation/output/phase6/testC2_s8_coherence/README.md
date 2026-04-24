# Phase 6 Test C2: S8 Cross-Probe Coherence

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Goal

Compare MTDF and LCDM S8 predictions (from Phase 5 MCMC Planck posteriors) against published weak lensing survey constraints. Quantify whether MTDF's sigma8 suppression reduces the S8 tension.

## Definition

S8 = sigma8 * sqrt(Omega_m / 0.3)

## Phase 5 MCMC S8 Values

| Model | sigma8 | Omega_m | S8 |
|-------|--------|---------|----|
| LCDM | 0.8101 +/- 0.0059 | 0.3149 +/- 0.0074 | 0.8300 +/- 0.0115 |
| MTDF | 0.7903 +/- 0.0058 | 0.3090 +/- 0.0072 | 0.8020 +/- 0.0111 |

## S8 Tension with Weak Lensing Surveys

| Survey | S8 | LCDM tension | MTDF tension | Reduction |
|--------|----|--------------|--------------|-----------|
| KiDS-1000 | 0.759 +/- 0.024 | 2.67 sigma | 1.63 sigma | 39% |
| DES Y3 | 0.776 +/- 0.017 | 2.63 sigma | 1.28 sigma | 51% |
| HSC Y3 | 0.763 +/- 0.040 | 1.61 sigma | 0.94 sigma | 42% |
| **WL combined** | 0.770 +/- 0.013 | 3.47 sigma | 1.89 sigma | -- |

## Interpretation

MTDF's sigma8 suppression (0.790 vs 0.810) moves S8 closer to weak lensing values.

**Caveats:**
- Tension metrics use simple Gaussian error propagation from MCMC posteriors
- A proper comparison requires running WL likelihoods within the MTDF framework
- Published WL S8 values assume LCDM; MTDF would modify the lensing kernel
- This is a first-order consistency check, not a full likelihood analysis

## Files

| File | Description |
|------|-------------|
| `testC2_s8_coherence.json` | Full comparison data |
| `testC2_s8_coherence.png` | Visual comparison |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testC2_s8_coherence.py
```
