# Test B: Weak Lensing x Environment

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Tangential shear decomposition
Position angle PA computed with spherical trigonometry (North through East):
```
PA = arctan2(sin(da) cos(dec_s),
             cos(dec_0) sin(dec_s) - sin(dec_0) cos(dec_s) cos(da))
```
Tangential and cross components (standard WL convention):
```
gamma_t =  e1 cos(2 PA) - e2 sin(2 PA)     [> 0 = tangential]
gamma_x =  e1 sin(2 PA) + e2 cos(2 PA)     [= 0 if no systematics]
```
Equivalent to gamma_t = -Re(epsilon exp(-2i phi_WL)) with phi_WL = pi/2 - PA.

## Shear calibration (KiDS lensfit)
Per tomographic bin:
- Additive c-term: weighted mean e1, e2 subtracted per bin.
- Multiplicative m-bias: applied in stacking denominator, not per source.
  gamma_t = Sum(w * e_t) / Sum(w * (1 + m))

## Source-behind-lens cut
z_source > z_void + 0.1  (photometric z_B from KiDS SOM).

## Projected separation
R_proj = chi(z_void) * theta, where chi(z) is the flat-LCDM comoving
distance (Omega_m = 0.3, h = 1 absorbed into Mpc/h units) and theta
is the haversine angular separation.

## Control sample
Random centres placed uniformly in KiDS-North (uniform in RA, uniform
in sin(DEC)), with redshifts drawn from the void z distribution.
N_random = 5 x N_voids.

## Systematics gate
gamma_x chi-squared vs zero with jackknife errors.
Pipeline stops if p < 0.05 (gamma_x inconsistent with zero).

## Headline statistic: Delta_gamma_t
Delta_gamma_t = <gamma_t>_void - <gamma_t>_random in [5, 20] Mpc/h.
Errors from bootstrap resampling of both void and random centres.

## Result

- Delta_gamma_t = -3.65e-05 +/- 4.68e-05 (0.8sigma)
- Bootstrap 95% CI: [-1.20e-04, +5.74e-05]
- gamma_x systematics gate: PASSED (p=0.234)
- N_voids = 626, N_random = 3130, N_boot = 500

### Explicit 95% Upper Limit

- **|Delta_gamma_t| < 1.20e-04** (two-sided 95%, bootstrap)
- Gaussian 95% CI: [-1.28e-04, +5.51e-05]

## Status

**COMPLETE.** Systematics-clean null at KiDS-1000 sensitivity.
No environment-dependent shear detected around DESIVAST BGS voids.
