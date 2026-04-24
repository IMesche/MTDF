# Phase 7: Cosmicflows-4 Peculiar Velocity Environment Coupling

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Purpose:** Test the MTDF prediction that peculiar velocity residuals correlate with signed distance to the nearest void boundary (d_signed), concentrated below z = 0.04, using the Cosmicflows-4 group catalogue (38,053 groups) and three independent DESIVAST void catalogues.

**Dataset:** Cosmicflows-4 (Tully et al. 2023), DESI DESIVAST void catalogues (VoidFinder, REVOLVER, VIDE), Carrick et al. (2015) 2M++ reconstructed velocity field, MDPL2 N-body simulation (CosmoSim)

**Key results:**
- Baseline: VoidFinder 19.6 sigma, REVOLVER 15.5 sigma, VIDE 11.0 sigma
- Piecewise z=0.04 split: 14.7 sigma below, 1.9 sigma above (VoidFinder)
- Partial correlation beyond density: 8-17 sigma across all void finders
- 2M++ velocity-field subtraction: 5-9 sigma retained (VoidFinder 9.2, REVOLVER 5.9, VIDE 5.2)
- LOSO: signal present in all distance-method families (TF 8.2 sigma, FP 9.7 sigma, SNIa 2.7 sigma)
- Malmquist bias: signal in all distance bins from 0-300 Mpc
- LCDM linear mock (100 realizations): observed vs mock tension 19.9 sigma (sign mismatch)
- MDPL2 N-body mock (100 observers): 9.0 sigma tension with sign mismatch
- Noise baseline: 18.4 sigma from noise floor

**Canonical results files:** `task3a_cf4_results.json`, `task3a_control_results.json`, `task3a_lcdm_mock_results.json`, `task3a_nbody_mock_results.json`, `task3a_2mpp_results.json`, `task3a_robustness_results.json`

**Referenced in:** MTDF_08 Sections 5.3, 7.1-7.3

## Files

### Result JSONs

| File | Description |
|------|-------------|
| `task3a_cf4_results.json` | Baseline gamma_v fits per void catalogue, piecewise z-split, permutation tests |
| `task3a_control_results.json` | Controls A-C: density substitution, partial correlation, LCDM chain rule |
| `task3a_2mpp_results.json` | Control D: 2M++ velocity-field subtraction results per void catalogue |
| `task3a_robustness_results.json` | Controls E-F: LOSO stability and Malmquist bias diagnostics |
| `task3a_lcdm_mock_results.json` | LCDM linear-theory mock (100 realizations) and noise baseline |
| `task3a_nbody_mock_results.json` | MDPL2 N-body mock (100 observers), gamma_v distribution |

### Plots: Baseline

| File | Description |
|------|-------------|
| `task3a_scatter_voidfinder.png` | vpec vs d_signed scatter (VoidFinder) |
| `task3a_scatter_revolver.png` | vpec vs d_signed scatter (REVOLVER) |
| `task3a_scatter_vide.png` | vpec vs d_signed scatter (VIDE) |
| `task3a_distributions_voidfinder.png` | Sample distributions (VoidFinder) |
| `task3a_distributions_revolver.png` | Sample distributions (REVOLVER) |
| `task3a_distributions_vide.png` | Sample distributions (VIDE) |
| `task3a_piecewise_voidfinder.png` | Piecewise z=0.04 split (VoidFinder) |
| `task3a_piecewise_revolver.png` | Piecewise z=0.04 split (REVOLVER) |
| `task3a_piecewise_vide.png` | Piecewise z=0.04 split (VIDE) |
| `task3a_zscan_voidfinder.png` | Redshift scan (VoidFinder) |
| `task3a_zscan_revolver.png` | Redshift scan (REVOLVER) |
| `task3a_zscan_vide.png` | Redshift scan (VIDE) |

### Plots: Controls

| File | Description |
|------|-------------|
| `control_summary_voidfinder.png` | Control chain summary (VoidFinder) |
| `control_summary_revolver.png` | Control chain summary (REVOLVER) |
| `control_summary_vide.png` | Control chain summary (VIDE) |
| `control_b_piecewise_voidfinder.png` | Partial correlation piecewise (VoidFinder) |
| `control_b_piecewise_revolver.png` | Partial correlation piecewise (REVOLVER) |
| `control_b_piecewise_vide.png` | Partial correlation piecewise (VIDE) |

### Plots: 2M++ Reconstruction

| File | Description |
|------|-------------|
| `2mpp_vs_shell_median.png` | gamma_v comparison: shell-median vs 2M++ residuals |
| `2mpp_subsample_comparison.png` | Subsample-level 2M++ results |

### Plots: Robustness

| File | Description |
|------|-------------|
| `loso_robustness.png` | Leave-one-method-out stability across TF, FP, SNIa, calibrators |
| `malmquist_diagnostics.png` | Malmquist bias: signal vs distance bin and quality split |

### Plots: Mocks

| File | Description |
|------|-------------|
| `mock_comparison_voidfinder.png` | LCDM linear mock vs observed gamma_v |
| `mock_piecewise_voidfinder.png` | LCDM mock piecewise comparison |
| `nbody_mock_comparison.png` | MDPL2 N-body mock vs observed (100 observers) |
| `nbody_mock_piecewise.png` | MDPL2 piecewise comparison |

## Reproducibility

All analysis scripts are in `../../code/paper8/`:
- `task3a_cosmicflows4_vpec.py` - Baseline CF4 analysis
- `task3a_control_tests.py` - Controls A-C
- `task3a_2mpp_reconstruction.py` - Control D (2M++ subtraction)
- `task3a_robustness.py` - Controls E-F (LOSO + Malmquist)
- `task3a_lcdm_mock.py` - LCDM linear mock + noise baseline
- `task3a_nbody_mock.py` - MDPL2 N-body mock (downloads halos from CosmoSim TAP API)

### Data dependencies
- Cosmicflows-4 group catalogue: [EDD CF4 groups](https://edd.ifa.hawaii.edu/CF4groupsHD/)
- DESIVAST void catalogues: [DESI void catalogue DR1](https://data.desi.lbl.gov/public/papers/c3/void_catalog_dr1/v1/) (VoidFinder, REVOLVER, VIDE FITS files)
- 2M++ velocity/density fields: [cosmicflows.iap.fr/2MPP](https://cosmicflows.iap.fr/2MPP/) (`twompp_velocity.npy`, `twompp_density.npy`)
- MDPL2 halos: auto-downloaded from [CosmoSim TAP API](https://www.cosmosim.org/) (snapshot 125, z=0), cached as `mdpl2_halos_z0.npz`
