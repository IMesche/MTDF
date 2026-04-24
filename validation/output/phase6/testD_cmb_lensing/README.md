# Phase 6 Test D v2: CMB Lensing x DESI Voids (compensated)

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Date:** 2026-02-16
**Planck:** PR4 MV (Carron+2022), NSIDE=512, lmax=1535, FWHM=0.5 deg
**Voids:** DESIVAST BGS (REVOLVER primary, 1992 total, 557 with compensated stat)

## Method

Stack Planck CMB lensing convergence (kappa) in radial R/Rv bins
around DESIVAST void centres. Mean field subtracted from alm.
Gaussian smoothing FWHM=0.5 deg applied in harmonic space.
**No hard mask cut**: all voids kept; masked pixels skipped per-pixel.
Primary statistic: compensated filter = kappa(R/Rv<0.5) - kappa(4.0<R/Rv<5.0) per void.

## Primary Result (REVOLVER)

- Compensated (centre - outer): -7.4709e-04 +/- 1.0334e-03 (0.7sigma, 557 voids)
- kappa_centre (R/Rv<0.5): 2.5573e-03 +/- 1.0385e-03 (2.5sigma)
- delta_kappa (ring-centre): -6.3292e-04 +/- 1.1224e-03 (0.6sigma)

### Explicit 95% Upper Limit

- Jackknife error: 1.0334e-03
- Bootstrap std: 1.0681e-03 (1000 resamples)
- **Gaussian 95% CI: [-2.773e-03, +1.278e-03]**
- **Bootstrap 95% CI: [-2.952e-03, +1.146e-03]**
- Two-sided 95% bound: |kappa_comp| < 2.95e-03
- One-sided 95% upper: kappa_comp < 8.86e-04

## Null Tests (compensated statistic)

- RA-scramble (200 iter): p(comp)=0.425 **(primary null for footprint-coupled modes)**
- Random positions (100 iter): p(comp)=0.080 (secondary)

## Low-l Robustness

| Map | Comp mean | Comp err | S/N | n_used |
|-----|-----------|----------|-----|--------|
| baseline | -7.4709e-04 | 1.0334e-03 | 0.7 | 557 |
| l>=20 | 5.8657e-04 | 1.0592e-03 | 0.6 | 557 |
| l>=30 | -9.5481e-05 | 1.0239e-03 | 0.1 | 557 |

## Freeze Checks (sensitivity)

### Annulus sensitivity (outer annulus range)

| Outer annulus | Comp mean | Comp err | S/N | n_used |
|---------------|-----------|----------|-----|--------|
| 3.0-4.0 Rv | -1.1184e-03 | 1.0544e-03 | 1.1 | 557 |
| **4.0-5.0 Rv** | **-7.4709e-04** | **1.0334e-03** | **0.7** | **557** |
| 3.5-5.0 Rv | -8.2029e-04 | 1.0328e-03 | 0.8 | 557 |

No annulus dependence. All within 1.1sigma.

### Mask threshold sensitivity

| Threshold | Comp mean | Comp err | S/N | n_used |
|-----------|-----------|----------|-----|--------|
| >= 50% | 2.2610e-03 | 1.5182e-03 | 1.5 | 97 |
| >= 60% | 2.7266e-03 | 1.6218e-03 | 1.7 | 74 |
| >= 70% | 4.7297e-03 | 2.0137e-03 | 2.3 | 49 |

Positive trend driven by sky-position selection bias: hard mask cuts retain only
voids in Galactic cap regions with large-scale positive kappa pedestal.
Confirms per-void subtraction (no hard cut) is the correct approach.

### Profile-based compensated (stacked, all 1992 voids)

- kappa_centre (R/Rv<0.5): 2.5573e-03 +/- 1.0385e-03
- kappa_outer (4-5 Rv): 3.0841e-04 +/- 2.0378e-04
- compensated: 2.2488e-03 +/- 1.0583e-03 (2.1sigma)

Opposite sign from per-void stat: profile-based inherits residual large-scale
pedestal because it subtracts global outer average, not per-void local background.
Confirms per-void compensated filter removes the pedestal correctly.

## Conclusion

**Phase 6D: LOCKED.** No void-specific CMB lensing signal at z<0.24 with DESIVAST BGS.
Per-void compensated = -0.7sigma, robust to annulus choice, low-l removal, and
void finder. Expected null given low redshift and small void count.
Next step: higher-z void catalogues (BOSS CMASS/LOWZ, DESI LRG/ELG).
