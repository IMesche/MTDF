# Phase 6 Test C: Derived Parameter Consistency Check

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74

## Goal
Verify MTDF does not improve sigma8/S8 by breaking other derived
cosmological quantities. All parameters from Phase 5 MCMC chains.

## Pass Condition
No parameter shifts > 3 sigma. Early-universe parameters
(theta_s, omega_b, n_s, tau_reio) must remain LCDM-compatible.
Improvements must not be paid by a broken derived constraint.

## Parameter Comparison
| Parameter | LCDM | MTDF | Shift (sigma) | Category |
|-----------|------|------|---------------|----------|
| H0 | 67.382 +/- 0.539 | 67.832 +/- 0.542 | +0.59 | late |
| sigma8 | 0.81012 +/- 0.00595 | 0.7903 +/- 0.00585 | -2.38 | late |
| Omega_m | 0.31488 +/- 0.00739 | 0.30897 +/- 0.00723 | -0.57 | late |
| n_s | 0.96458 +/- 0.00413 | 0.96809 +/- 0.00414 | +0.60 | early |
| tau_reio | 0.053964 +/- 0.00747 | 0.051688 +/- 0.00744 | -0.22 | early |
| theta_s_100 | 1.0419 +/- 0.000293 | 1.0419 +/- 0.000295 | +0.23 | early |
| omega_b | 0.022359 +/- 0.000147 | 0.022364 +/- 0.000153 | +0.02 | early |
| omega_cdm | 0.11992 +/- 0.0012 | 0.11911 +/- 0.00119 | -0.48 | late |
| S8 | 0.82996 +/- 0.0115 | 0.80204 +/- 0.0111 | -1.75 | late |
| Omega_m_h2 | 0.14297 +/- 0.00406 | 0.14216 +/- 0.00403 | -0.14 | derived |
| A_s | 2.0979 +/- 0.0302 | 2.0841 +/- 0.0297 | -0.33 | early |
| k_f | -- | 0.49501 +/- 0.36 | -- | mtdf |

## S8 Tension with Weak Lensing

**Tension definition:** T = |S8_model - S8_external| / sqrt(sigma_model^2 + sigma_external^2).
This is the standard Gaussian pull: difference divided by quadrature sum of uncertainties.

**External references:**
- DES Y3: S8 = 0.776 +/- 0.017 (Amon et al. 2022, PRD 105, 023514; Secco et al. 2022, PRD 105, 023515)
- KiDS-1000: S8 = 0.766 +/- 0.020 (Asgari et al. 2021, A&A 645, A104)

| Comparison | Model S8 | External S8 | Tension (sigma) |
|------------|----------|-------------|-----------------|
| LCDM vs DES Y3 | 0.830 +/- 0.012 | 0.776 +/- 0.017 | 2.63 |
| MTDF vs DES Y3 | 0.802 +/- 0.011 | 0.776 +/- 0.017 | 1.28 |
| LCDM vs KiDS-1000 | 0.830 +/- 0.012 | 0.766 +/- 0.020 | 2.77 |
| MTDF vs KiDS-1000 | 0.802 +/- 0.011 | 0.766 +/- 0.020 | 1.58 |

## Result: **PASS**

No parameters exceed 3 sigma shift. Early-universe parameters are LCDM-compatible.
S8 improvement is not paid by breaking any derived constraint.

## Files
| File | Description |
|------|-------------|
| `testC_consistency.json` | Full comparison table |
| `testC_parameter_shifts.png` | Visual shift summary |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testC_derived_consistency.py
```
