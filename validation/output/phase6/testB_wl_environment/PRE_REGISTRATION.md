# Phase 6 Test B: Weak Lensing x Environment

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74

# Pre-Registration Document

**Status:** Unblinded. Real data run complete (2026-02-16).
**Date:** 2026-02-16
**Test matrix reference:** C1 (environment-dependent shear response)

---

## 1. Scientific Claim

MTDF predicts that the effective gravitational lensing strength depends
on the local stress-tensor environment.  Specifically, void regions
(low stress) should produce weaker shear than wall regions (high stress)
at fixed source-lens geometry.  LCDM predicts no such dependence.

## 2. Dataset Choice

### 2a. Rationale

The void catalogue used throughout this work is the DESIVAST BGS
REVOLVER catalogue (Douglass et al. 2023), whose NGC sub-catalogue
contains 1,692 voids in the equatorial strip RA ~ [96, 295],
DEC ~ [-9, +69].  A footprint overlap analysis showed:

| Lensing survey | Overlapping voids | % of catalogue | Footprint quality |
|---------------|-------------------|----------------|-------------------|
| DES Y3 | 255 (SGC only) | 12.8% | Northern edge of DES (worst calibration region) |
| **KiDS-1000** | **734 (NGC)** | **36.8%** | Equatorial core (KiDS designed for this strip) |

KiDS-1000 provides 3x more overlapping voids drawn from the larger and
better-characterized NGC sub-catalogue, in a footprint region where
KiDS shape calibration is optimal.  DES Y3 overlap is confined to a
thin equatorial strip at the northern edge of the DES footprint
(DEC ~ [-7, +5]), where shear systematics are least well controlled.

KiDS-1000 is therefore adopted as the primary lensing survey.
DES Y3 is retained as an optional secondary cross-check (see Section 9).

### 2b. Primary: KiDS-1000 (Kilo-Degree Survey)

- **Gold sample:** 21,262,011 sources (Giblin et al. 2021)
- **Shape measurement:** lensfit (model-fitting)
- **Effective source density:** 6.17 arcmin^-2
- **Shear calibration:** multiplicative bias m per tomographic bin,
  calibrated via image simulations (Kannawadi et al. 2019)
- **Photo-z:** 9-band BPZ with SOM calibration (Hildebrandt et al. 2021)
- **Survey area:** ~1,000 deg^2 (effective ~777 deg^2 after masking)
- **Public download:** single 16 GB FITS file, no registration required
- **Reference:** Asgari et al. 2021 (A&A 645, A104); Giblin et al. 2021
  (A&A 645, A105); Hildebrandt et al. 2021 (A&A 647, A124)

### 2c. Stacking centres and control sample

**Void centres:** DESIVAST BGS REVOLVER NGC sub-catalogue (same as
Phase 3 / Test A).  After 1-degree edge buffer within KiDS-North,
626 void centres remain (from 734 raw overlap).

**Random controls:** Random sky positions placed uniformly in the
KiDS-North footprint (uniform in RA, uniform in sin(DEC) for correct
area weighting), with redshifts drawn from the void z distribution.
N_random = 5 x N_voids = 3,130 centres.

The control sample provides a null baseline: the mean tangential shear
around random positions should be zero, allowing direct measurement
of the void lensing excess.

## 3. Primary Statistic

**Stacked tangential shear excess around void centres:**
1. Select source galaxies behind each centre (z_source > z_void + 0.1)
2. Stack tangential shear gamma_t(R) in projected radial bins around
   all 626 void centres, calibrated as Sum(w * e_t) / Sum(w * (1+m))
3. Repeat for 3,130 random control centres (null baseline)
4. Primary test statistic:
   **Delta_gamma_t = <gamma_t>_void - <gamma_t>_random**
   averaged in [5, 20] Mpc/h

This measures the void lensing excess directly.  It is a shear
amplitude difference, not an inferred S8 value (which would require
a model-dependent calibration mapping gamma_t to S8).

## 4. Null Expectations

Under LCDM:
- gamma_t around voids is negative (underdensity lensing)
- gamma_t around random positions averages to zero
- Delta_gamma_t consistent with zero after random subtraction
  (any residual signal reflects void underdensity, not environment
  dependence of lensing strength)

Under MTDF:
- Void stress depletion modifies the effective lensing signal,
  producing a Delta_gamma_t that differs from LCDM
- Direction: same sign as the sigma8 suppression seen in Phase 5 CMB
  (sigma8_MTDF = 0.790 vs sigma8_LCDM = 0.810)
- Magnitude: order-of-magnitude estimate from alpha coupling suggests
  ~2-5% effect, but this is speculative until the full MTDF lensing
  equation is derived

## 5. Significance Quantification

- Profile errors: jackknife over void/random centres (leave-one-out)
- Delta_gamma_t uncertainty: bootstrap resampling of both void and
  random centre sets (500 resamples)
- Report: Delta_gamma_t, bootstrap 68% and 95% CI,
  p-value against Delta_gamma_t = 0

## 6. Required Systematics Checks

| Check | Method | Pass condition |
|-------|--------|----------------|
| Shear calibration | Multiplicative bias m correction per tomo bin (Kannawadi+2019) | Applied per KiDS-1000 pipeline |
| Additive bias | Weighted-mean c-term subtraction per tomo bin | c1, c2 ~ 0 after subtraction |
| Photo-z bias | SOM-calibrated gold sample; residual delta_z per bin | delta_z < 0.02 per bin |
| PSF leakage | Cross-component (gamma_x) around voids | gamma_x consistent with zero |
| Boost factor | Random-point subtraction | Corrected per Sheldon et al. |
| Mask / edge effects | Exclude voids within 1 deg of KiDS tile edges | Edge buffer applied |
| Source-lens separation | z_source > z_void + 0.1 (spectroscopic voids) | |
| Orientation bias | Void orientation randomisation test | |
| Footprint overlap | Verify void centres fall within KiDS-North tiles | 626 voids after 1-deg edge buffer |

## 7. Data Acquisition Steps

1. Download KiDS-1000 gold shear catalogue (~16 GB FITS):
   `wget https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits`
2. Download SOM-calibrated n(z) distributions:
   `wget https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS1000_SOM_N_of_Z.tar.gz`
3. DESIVAST NGC void catalogue (already in
   `validation/data/External/desivast_voids/`)
4. Stack tangential shear around void centres and random controls

See `DATA.md` for full file inventory and column reference.

## 8. Blinding Protocol

Pre-registration filed before shear measurements were performed.
Pipeline validated on 1% smoke test (gamma_x gate) before full run.

The released KiDS-1000 gold catalogue contains the unblinded
(Blind C) ellipticities and weights.

## 9. Alternative Dataset: DES Y3

DES Y3 (Gatti et al. 2021; Secco et al. 2022) was initially considered
as the primary lensing survey.  Footprint overlap analysis revealed
that only the SGC void sub-catalogue (255 voids, 12.8% of total)
falls within the DES Y3 footprint, confined to a thin equatorial
strip at DEC ~ [-7, +5] near the northern edge of DES where shape
calibration is least well controlled.

DES Y3 remains available as a secondary cross-check if additional
independent confirmation is desired.

## 10. Outcome

**gamma_x systematics gate:** chi2/dof = 1.31 (p = 0.234) -- PASS.
Cross-component consistent with zero; no evidence of coordinate or
PSF systematics.

**Void tangential shear:** gamma_t is consistent with zero across all
radial bins (|gamma_t| < 2e-4), with jackknife errors of ~1e-4.
This is expected for DESIVAST BGS voids at z ~ 0.03-0.24: these
are low-redshift, low-contrast voids where the KiDS lensing kernel
(peaking at z ~ 0.5-1.0) provides very little lensing weight.

**Delta_gamma_t = -0.000037 +/- 0.000047** (95% CI: [-0.000120, +0.000057]).
Consistent with zero at < 1 sigma.  The test is not constraining at
current depth with these low-z voids.  A detection would require
either higher-z voids (DESI LRG sample, z ~ 0.4-0.8) or deeper
lensing data (Rubin LSST).

---

## Revision History

| Date | Change |
|------|--------|
| 2026-02-16 | Initial pre-registration (DES Y3 primary) |
| 2026-02-16 | Switched to KiDS-1000 primary after footprint overlap analysis; DES Y3 demoted to secondary |
| 2026-02-16 | Final run complete: proper tangential decomposition, random controls, Delta_gamma_t reported |
