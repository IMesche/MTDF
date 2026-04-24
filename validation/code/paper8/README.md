# MTDF_08 Analysis Scripts

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74

Analysis scripts for MTDF Companion Paper VIII: "Multi-Probe Evidence for an Environment-Dependent Cosmological Transition at z < 0.04."

## Scripts

### CF4 Peculiar Velocities (Phase 7)

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `task3a_cosmicflows4_vpec.py` | 5.3.1-5.3.2 | Baseline gamma_v, piecewise z-split, permutations |
| `task3a_control_tests.py` | 5.3.3 (A-C) | Density substitution, partial correlation, LCDM chain rule |
| `task3a_2mpp_reconstruction.py` | 5.3.3 (D) | 2M++ velocity-field subtraction |
| `task3a_robustness.py` | 5.3.3 (E-F) | Leave-one-method-out, Malmquist bias |
| `task3a_lcdm_mock.py` | 7.1, 7.3 | LCDM linear mock (100 realisations), noise baseline |
| `task3a_nbody_mock.py` | 7.2 | MDPL2 N-body mock (100 observers) |

### 2MTF Tully-Fisher (Phase 8)

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `task3b_2mtf_tully_fisher.py` | 5.4.1 | Baseline gamma_TF across J/H/K bands |
| `task3b_control_tests.py` | 5.4.2 | Density substitution, partial correlation, LCDM chain rule |
| `task3b_nbody_mock.py` | 7.2 | MDPL2 N-body mock (50 observers) |

## Output

Results are written to `../../output/phase7_cf4_vpec/` and `../../output/phase8_2mtf_tf/`.

## Dependencies

### Python packages
- numpy, scipy, astropy, matplotlib

### External data (place in `../../data/External/`)
- [Cosmicflows-4 groups](https://edd.ifa.hawaii.edu/CF4groupsHD/) -> `cosmicflows4/cf4_groups.csv`
- [2MTF catalogue](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/MNRAS/487/2061) -> `2mtf/2mtf_table1.csv`, `2mtf/2mtf_table2.csv`
- [DESIVAST void catalogues](https://data.desi.lbl.gov/public/papers/c3/void_catalog_dr1/v1/) -> `desivast_voids/*.fits`
- [2M++ velocity/density fields](https://cosmicflows.iap.fr/2MPP/) -> `2mpp/twompp_velocity.npy`, `2mpp/twompp_density.npy`
- [MDPL2 N-body halos](https://www.cosmosim.org/) -> auto-downloaded by `task3a_nbody_mock.py`

See `../../data/External/README.md` for exact filenames and directory layout.
