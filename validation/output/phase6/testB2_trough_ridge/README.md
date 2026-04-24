# Test B2: Trough & Ridge Lensing (KiDS-internal, v3)

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Method
Projected foreground galaxy density map built from KiDS-1000 photometric
galaxies at 0.2 < Z_B <= 0.5 using aperture counting
(theta_A = 10 arcmin, cKDTree).

### Mask-aware normalisation
All-galaxy counts (N_ref, weight>0) serve as mask-aware reference:
```
delta = (N_fg / N_ref) * (N_ref_total / N_fg_total) - 1
```
This corrects for survey tile gaps, masking, and depth variations
without requiring a separate survey mask file.  Apertures where
N_ref < 30% of patch mean are excluded.

### Full footprint
Both KiDS-North and KiDS-South patches are used (~1000 deg^2 total).

Troughs = bottom 20th percentile of normalised delta.
Ridges = top 80th percentile.
Median foreground z = 0.380.

Background sources: Z_B > 0.6 (tomo bins 4-5).
Source-behind-lens cut: z_source > z_centre + 0.1.

## Tangential shear decomposition
```
gamma_t =  e1 cos(2 PA) - e2 sin(2 PA)
gamma_x =  e1 sin(2 PA) + e2 cos(2 PA)
```

## Shear calibration (KiDS lensfit)
gamma_t = Sum(w * e_t) / Sum(w * (1 + m))
c-term subtracted per tomographic bin.

## Headline statistics
Primary range: [3.5, 20] Mpc/h.
Delta_gamma_t_trough = <gamma_t>_trough - <gamma_t>_random  (primary)
Delta_gamma_t_ridge  = <gamma_t>_ridge  - <gamma_t>_random  (secondary)
Delta_gamma_t_split  = <gamma_t>_ridge  - <gamma_t>_trough

Errors from bootstrap resampling over centres.

## Density map statistics
Grid points (good): 33,992 / 47683
Masked: 13691
Mean foreground per aperture: 432.5
Mean reference per aperture: 1896.2
Normalisation: mask-aware (all-galaxy reference)

## Result

- Delta_gamma_t_split (ridge - trough): +2.56e-05 +/- 5.46e-05 (0.5sigma)
- Delta_gamma_t_trough (trough - random): +1.01e-05 +/- 4.10e-05 (0.2sigma)
- Delta_gamma_t_ridge (ridge - random): +3.57e-05 +/- 4.28e-05 (0.8sigma)
- gamma_x gate: PASSED for both trough (p=0.188) and ridge (p=0.087)
- N_trough = 2000, N_ridge = 2000, N_random = 10000, N_boot = 500

### Explicit 95% Upper Limits

- **|Delta_gamma_t_split| < 1.28e-04** (two-sided 95%, bootstrap)
- Bootstrap 95% CI (split): [-9.26e-05, +1.28e-04]
- Bootstrap 95% CI (trough): [-6.95e-05, +9.49e-05]

## Status

**COMPLETE.** Systematics-clean null at KiDS-1000 sensitivity.
No environment-dependent lensing signal between troughs and ridges.
