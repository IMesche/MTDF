# Test 2: Leave-One-Likelihood-Out Comparison

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


| Config | Likelihoods | k_f | σ₈ | H₀ | Ω_m | χ²_best | R-1 | N_acc |
|--------|------------|------|------|------|------|---------|-----|-------|
| Baseline | TT_clik, EE_clik, TTTEEE, native | 0.495 ± 0.360 | 0.7903 ± 0.0058 | 67.83 ± 0.54 | 0.3090 ± 0.0072 | 2772.6 | 0.019 | 27160 |
| No lensing | TT_clik, EE_clik, TTTEEE | 0.527 ± 0.377 | 0.7940 ± 0.0069 | 67.55 ± 0.64 | 0.3128 ± 0.0087 | 2765.4 | 0.104 | 4299 |
| TT only | TT_clik, TT | 0.938 ± 0.670 | 0.8085 ± 0.0210 | 67.74 ± 1.17 | 0.3093 ± 0.0156 | 782.5 | 0.096 | 7560 |

**Shifts relative to baseline (in baseline σ):**

| Config | Δk_f/σ | Δσ₈/σ | ΔH₀/σ | ΔΩ_m/σ |
|--------|--------|--------|--------|--------|
| No lensing | +0.09 | +0.64 | -0.51 | +0.53 |
| TT only | +1.23 | +3.12 | -0.16 | +0.04 |

**BOBYQA best-fit Δχ² (MTDF − ΛCDM, apples-to-apples per subset):**

| Config | χ²_ΛCDM | χ²_MTDF | Δχ² | k_f best-fit |
|--------|---------|---------|-----|-------------|
| Baseline | 2773.20 | 2773.82 | +0.63 | 0.028 |
| No lensing | 2764.98 | 2765.39 | +0.41 | 0.035 |
| TT only | 782.32 | 781.67 | -0.65 | 0.450 |

**Note on TT-only σ₈:** The +3.1σ shift reflects constraint weakening (posterior broadening), not a physical shift. TT alone carries less growth information than TTTEEE, so σ₈ reverts toward a less constrained value.

**Nuisance freedom:** Baseline and no-lensing: 21 sampled nuisance params. TT-only: 15 (drops 6 `galf_TE_*` dust polarisation parameters absent in TT-only).
