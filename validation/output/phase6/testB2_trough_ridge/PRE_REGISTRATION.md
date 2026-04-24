# Phase 6 Test B2: Trough & Ridge Lensing (KiDS-internal)

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74

# Pre-Registration Document

**Status:** Pre-registered. Awaiting execution.
**Date:** 2026-02-16
**Test matrix reference:** C1 (environment-dependent shear response)

---

## 1. Scientific Claim

MTDF predicts that gravitational lensing strength depends on the local
stress-tensor environment.  Test B0 (DESIVAST void lensing) returned
a null result because the voids sit at z ~ 0.03-0.24, far below the
KiDS lensing kernel peak.

Test B2 uses the KiDS photometric galaxy density field itself to define
environment at higher effective redshift (z ~ 0.35), where the lensing
kernel provides substantial signal.  Troughs (underdense lines of sight)
and ridges (overdense) are identified from a projected density map and
used as stacking centres for tangential shear measurement.

Under LCDM, ridges produce positive gamma_t (tangential alignment from
overdensity lensing) and troughs produce negative gamma_t (radial
alignment from underdensity).  MTDF predicts a modified split amplitude
if the effective lensing strength is environment-dependent.

## 2. Dataset

**KiDS-1000** (same catalogue as Test B0; no additional download).

- **Foreground:** 0.2 < Z_B <= 0.5 (~3.3M galaxies in KiDS-North,
  tomo bins 2-3).  Used only for positions (density map).
- **Background:** Z_B > 0.6 (~2.2M galaxies, tomo bins 4-5).
  Full shear calibration (c-term, m-bias) applied.
- **Source-behind-lens:** z_source > z_centre + 0.1.

No external void catalogue is used.  Environment is defined entirely
from the KiDS photometric galaxy distribution.

## 3. Density Map Construction

- Aperture radius: theta_A = 10 arcmin (~3 Mpc/h at z = 0.35)
- Grid spacing: 10 arcmin (independent cells)
- Flat-sky approximation: (RA * cos(DEC), DEC) in degrees
  (valid for KiDS equatorial strip, DEC ~ [-5, +5])
- Overdensity: delta = N / N_mean - 1
- Edge buffer: theta_A + 1 deg from KiDS-North boundaries

## 4. Centre Selection

- **Troughs:** grid points with delta <= 20th percentile
- **Ridges:** grid points with delta >= 80th percentile
- Maximum 2,000 centres per category (random subsample if exceeded)
- Centre redshift: z = median(foreground Z_B) ~ 0.35

## 5. Control Sample

Random sky positions in KiDS-North (uniform RA, uniform sin(DEC)).
z = median foreground z (same as trough/ridge centres).
N_random = 5 x max(N_trough, N_ridge).

## 6. Primary Statistics

Three headline measurements in [5, 20] Mpc/h:

1. **Delta_gamma_t_trough** = <gamma_t>_trough - <gamma_t>_random
   Expected < 0 (underdensity lensing).

2. **Delta_gamma_t_ridge** = <gamma_t>_ridge - <gamma_t>_random
   Expected > 0 (overdensity lensing).

3. **Delta_gamma_t_split** = <gamma_t>_ridge - <gamma_t>_trough
   Most powerful statistic; expected significantly > 0 under both
   LCDM and MTDF.  MTDF predicts a modified amplitude.

## 7. Null Expectations

Under LCDM:
- Delta_gamma_t_split > 0 (significant detection expected; KiDS has
  sufficient S/N at z ~ 0.35)
- Magnitude: gamma_t ~ 1e-3 to 3e-3 for ridges (comparable to
  published KiDS galaxy-galaxy lensing results)

Under MTDF:
- Stress depletion in troughs modifies the effective lensing signal
- Direction: enhanced split (troughs weaker, ridges similar)
- Quantitative prediction requires full MTDF lensing calculation

## 8. Error Budget

- **Profile errors:** Jackknife over centres (leave-one-out)
- **Headline statistic uncertainty:** Bootstrap resampling over centres
  (500 resamples of trough, ridge, and random sets simultaneously)
- **Sigma from zero:** |mean| / std of bootstrap distribution

## 9. Systematics Checks

| Check | Method | Pass condition |
|-------|--------|----------------|
| Shear calibration | m-bias per tomo bin in stacking denominator | Applied per KiDS-1000 pipeline |
| Additive bias | c-term subtraction per tomo bin | c1, c2 ~ 0 after subtraction |
| PSF leakage | gamma_x gate (chi2 vs zero) | p > 0.05 for both troughs and ridges |
| Edge effects | 1 deg + theta_A buffer from footprint edges | Applied |
| Source-lens separation | z_source > z_centre + 0.1 | Enforced |
| Density map noise | Mean counts per aperture reported | N_mean >> 1 |

## 10. Blinding Protocol

Pre-registration filed before shear stacking was performed.
Pipeline validated on 1% smoke test (gamma_x gate) before full run.

---

## Revision History

| Date | Change |
|------|--------|
| 2026-02-16 | Initial pre-registration |
