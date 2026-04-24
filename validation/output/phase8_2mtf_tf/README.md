# Phase 8: 2MTF Tully-Fisher Environment Coupling

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Purpose:** Test the MTDF environment coupling prediction using an independent distance indicator: 2MASS Tully-Fisher distances for 2,062 nearby spirals (z < 0.033), measured in three infrared bands (J, H, K). The entire sample falls within the predicted MTDF-active regime (z < 0.04).

**Dataset:** 2MTF (Hong et al. 2019), DESI DESIVAST void catalogues (VoidFinder, REVOLVER, VIDE), MDPL2 N-body simulation (CosmoSim)

**Key results:**
- Baseline: VoidFinder K-band 7.9 sigma, H-band 7.5 sigma, J-band 7.3 sigma
- Achromatic: J/H/K consistent within 10% (rules out dust)
- Cross-catalogue: REVOLVER 6.0 sigma, VIDE 3.5 sigma
- Partial correlation beyond density: 4-15 sigma across void finders
- MDPL2 N-body mock (50 observers): 4.2 sigma tension (observed 0.058 vs mock 0.002 +/- 0.013)
- Note: all 2,062 galaxies classified as "wall" by DESIVAST (n_void = 0); signal is a gradient within wall population via continuous d_signed metric

**Canonical results files:** `task3b_2mtf_results.json`, `task3b_control_results.json`, `task3b_nbody_mock_results.json`

**Referenced in:** MTDF_08 Sections 5.4, 7.2

## Files

### Result JSONs

| File | Description |
|------|-------------|
| `task3b_2mtf_results.json` | Baseline gamma_TF fits per void catalogue and band, permutation tests |
| `task3b_control_results.json` | Controls A-C: density substitution, partial correlation, LCDM chain rule |
| `task3b_nbody_mock_results.json` | MDPL2 N-body mock (50 observers), gamma_TF distribution |

### Plots

| File | Description |
|------|-------------|
| `task3b_comparison.png` | Cross-catalogue comparison of gamma_TF |
| `task3b_tf_voidfinder_k.png` | TF residual vs d_signed (VoidFinder, K-band) |
| `task3b_tf_voidfinder_h.png` | TF residual vs d_signed (VoidFinder, H-band) |
| `task3b_tf_voidfinder_j.png` | TF residual vs d_signed (VoidFinder, J-band) |
| `task3b_tf_revolver_k.png` | TF residual vs d_signed (REVOLVER, K-band) |
| `task3b_tf_revolver_h.png` | TF residual vs d_signed (REVOLVER, H-band) |
| `task3b_tf_revolver_j.png` | TF residual vs d_signed (REVOLVER, J-band) |
| `task3b_tf_vide_k.png` | TF residual vs d_signed (VIDE, K-band) |
| `task3b_tf_vide_h.png` | TF residual vs d_signed (VIDE, H-band) |
| `task3b_tf_vide_j.png` | TF residual vs d_signed (VIDE, J-band) |
| `nbody_2mtf_comparison.png` | MDPL2 N-body mock vs observed gamma_TF |

## Reproducibility

All analysis scripts are in `../../code/paper8/`:
- `task3b_2mtf_tully_fisher.py` - Baseline 2MTF analysis
- `task3b_control_tests.py` - Controls A-C
- `task3b_nbody_mock.py` - MDPL2 N-body mock for 2MTF (50 observers)

### Data dependencies
- 2MTF catalogue: [VizieR J/MNRAS/487/2061](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/MNRAS/487/2061) (Hong et al. 2019, tables 1+2)
- DESIVAST void catalogues: [DESI void catalogue DR1](https://data.desi.lbl.gov/public/papers/c3/void_catalog_dr1/v1/) (VoidFinder, REVOLVER, VIDE FITS files)
- MDPL2 halos: auto-downloaded from [CosmoSim TAP API](https://www.cosmosim.org/), cached as `mdpl2_halos_z0.npz` (shared with Phase 7)
