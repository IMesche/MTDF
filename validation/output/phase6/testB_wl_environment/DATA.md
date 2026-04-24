# Test B: Required Data Files

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Primary: KiDS-1000 Shear Catalogue

**Source:** Kilo-Degree Survey, Data Release 4
**URL:** https://kids.strw.leidenuniv.nl/DR4/KiDS-1000_shearcatalogue.php
**References:**
- Giblin et al. 2021 (A&A 645, A105) -- shape measurements, shear catalogue
- Hildebrandt et al. 2021 (A&A 647, A124) -- photo-z calibration, SOM gold selection
- Asgari et al. 2021 (A&A 645, A104) -- cosmic shear results, S8 = 0.766 +/- 0.020
- Kannawadi et al. 2019 (A&A 624, A92) -- image simulations for m-bias

### Required Files

| File | Description | Size | URL |
|------|-------------|------|-----|
| `KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits` | Gold shear catalogue | ~16 GB | [download](https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits) |
| `KiDS1000_SOM_N_of_Z.tar.gz` | SOM-calibrated n(z) per tomo bin | ~1 MB | [download](https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS1000_SOM_N_of_Z.tar.gz) |

### Key Columns (shear catalogue)

The gold catalogue contains 21,262,011 sources and 241 columns.
Only the following are needed for this test:

| Column | Type | Description |
|--------|------|-------------|
| `ALPHA_J2000` | float64 | Right ascension (deg, J2000) |
| `DELTA_J2000` | float64 | Declination (deg, J2000) |
| `e1` | float32 | lensfit ellipticity component 1 (raw, no m/c applied) |
| `e2` | float32 | lensfit ellipticity component 2 (raw, no m/c applied) |
| `weight` | float32 | Recalibrated lensfit inverse-variance weight |
| `Z_B` | float64 | 9-band BPZ photo-z (posterior peak) |
| `PSF_e1` | float32 | Mean PSF ellipticity, component 1 (for PSF leakage check) |
| `PSF_e2` | float32 | Mean PSF ellipticity, component 2 (for PSF leakage check) |

**Note:** The released gold catalogue is already the SOM-selected sample.
All 21M objects have passed the gold quality cut. No additional
`Flag_SOM_Fid` filtering is needed.

### Tomographic Bins

Defined by `Z_B` (BPZ photo-z peak):

| Bin | Z_B range | n_eff (arcmin^-2) | sigma_e (per comp.) |
|-----|-----------|-------------------|---------------------|
| 1 | 0.1 < Z_B <= 0.3 | 1.35 | 0.265 |
| 2 | 0.3 < Z_B <= 0.5 | 1.65 | 0.263 |
| 3 | 0.5 < Z_B <= 0.7 | 1.62 | 0.271 |
| 4 | 0.7 < Z_B <= 0.9 | 1.32 | 0.273 |
| 5 | 0.9 < Z_B <= 1.2 | 0.83 | 0.290 |

For this test, bins 2-5 (Z_B > 0.3) are the primary source sample,
since DESIVAST voids sit at z ~ 0.03-0.24 and we require
z_source > z_void + 0.1.

### Shear Calibration

**Multiplicative bias m** (per tomographic bin, from Kannawadi et al. 2019):

| Bin | m | sigma_m |
|-----|---|---------|
| 1 | -0.009 | 0.019 |
| 2 | -0.011 | 0.020 |
| 3 | -0.015 | 0.017 |
| 4 | +0.002 | 0.012 |
| 5 | +0.008 | 0.010 |

Applied in the stacking denominator (not per-galaxy division):
```
gamma_t = Sum(w * e_t) / Sum(w * (1 + m[bin]))
gamma_x = Sum(w * e_x) / Sum(w * (1 + m[bin]))
```

**Additive bias c-term** (subtract weighted mean per tomo bin):
```
c1 = sum(w * e1) / sum(w)
c2 = sum(w * e2) / sum(w)
e1_corrected = e1 - c1
e2_corrected = e2 - c2
```

**Photo-z mean bias delta_z** (per bin):

| Bin | delta_z |
|-----|---------|
| 1 | +0.000 |
| 2 | -0.002 |
| 3 | -0.013 |
| 4 | -0.011 |
| 5 | +0.006 |

### Cuts to Apply

1. `weight > 0` (valid lensfit weight)
2. `Z_B > z_void_max + 0.1` (source-lens separation; for DESIVAST
   voids at z < 0.24, this means Z_B > 0.34, i.e. bins 2-5)
3. KiDS-North footprint: RA in [120, 240], DEC in [-5, +5]
   (overlap with DESIVAST NGC voids)
4. Edge buffer: exclude **void centres** within 1 deg of KiDS-North
   footprint boundaries (mitigates edge-of-footprint effects on
   the stacked tangential shear profiles)

## Void Catalogue (already available)

**Location:** `validation/data/External/desivast_voids/`
**File used:** `DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits` (1,692 voids;
626 overlap KiDS-North footprint after 1-deg edge buffer)

No additional download required.

## Download Instructions

```bash
# 1. Create data directory
mkdir -p validation/data/External/kids_1000/

# 2. Download gold shear catalogue (~16 GB)
wget -P validation/data/External/kids_1000/ \
  https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits

# 3. Download n(z) distributions
wget -P validation/data/External/kids_1000/ \
  https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS1000_SOM_N_of_Z.tar.gz

# 4. Extract n(z)
cd validation/data/External/kids_1000/
tar xzf KiDS1000_SOM_N_of_Z.tar.gz

# 5. Run pipeline
cd /path/to/MTDF
python mtdf_validation/phase6/testB_wl_skeleton.py
```

## Footprint Sanity Check

Before running the full analysis, the pipeline performs an automated
footprint overlap check:
1. Load DESIVAST NGC void centres (RA, DEC)
2. Verify each void centre falls within KiDS-North boundaries
3. Apply 1-deg edge buffer to exclude void centres near footprint edges
4. Report effective void count and sky coverage

Result: 626 void centres after edge buffer (from 734 raw overlap;
the 1-deg buffer is applied to void centre coordinates, not to
individual source galaxies).

---

## Alternative Dataset: DES Y3

DES Y3 was initially considered but has limited overlap with the
DESIVAST void catalogue (255 SGC voids only, at the northern edge
of the DES footprint).  See PRE_REGISTRATION.md Section 9 for details.

If DES Y3 is used as a secondary cross-check, the required files are:

| File | Description | Size |
|------|-------------|------|
| `mcal-y3a2-combined-unblind-v1.0.fits` | Metacalibration shear catalogue | ~50 GB |
| `y3_redshift_distributions_sompz_v0.0.h5` | SOMPZ photo-z distributions | ~500 MB |

**Source:** https://des.ncsa.illinois.edu/releases/y3a2 (registration required)
**Reference:** Gatti et al. 2021 (MNRAS 504, 4312); Secco et al. 2022 (PRD 105, 023515)
