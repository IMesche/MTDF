# MTDF: Mesche's Tensor Dynamics Framework

**A reproducible, first-principles approach to the latent sector crisis in cosmology.**

| | |
|---|---|
| **Author** | Ingo Mesche |
| **Affiliation** | Independent Researcher, Malta |
| **Theory Version** | V74 |
| **Workbook Version** | V18 |
| **Release** | v1.1.6 (2026-06-10) |
| **DOI (concept, all versions)** | [10.5281/zenodo.19741058](https://doi.org/10.5281/zenodo.19741058) |
| **DOI (this version, v1.1.6)** | [10.5281/zenodo.20629767](https://doi.org/10.5281/zenodo.20629767) |
| **License** | GPL-3.0 |

---

## What Is MTDF?

MTDF replaces the standard LCDM cosmological model by treating spacetime as an elastic medium characterised by an intrinsic stress-energy tensor. Four fundamental parameters govern the framework: one anchored to an independent observation (the void radius distribution), one a frozen void-motivated coupling, one synchronised to the cosmic age, and one fitted (see paper Section 5.0). No dark energy is invoked, and no particle dark matter is required at late times or on galactic scales.

### The Four Fundamental Parameters

| Symbol | Value | Unit | Meaning | Calibration Source |
|--------|-------|------|---------|--------------------|
| alpha | 1.30 +/- 0.26 | dimensionless | Stress-matter coupling | Frozen, void-motivated value (contested anchor; see Kenworthy et al. 2019); not independently measured |
| beta | 7.00 +/- 0.09 x 10^23 m (22.7 +/- 0.3 Mpc) | m | Coherence length scale | Empirically constrained: MTDF-original ansatz Rn = beta(1+sqrt(n)) matched to Sutter et al. 2012 SDSS DR7 void radii (DOI: 10.1088/0004-637X/761/1/44) |
| tau | 13.0 +/- 0.2 | Gyr | Relaxation timescale | Age synchronisation |
| beta_eos | 0.573 +/- 0.012 | dimensionless | EOS transition parameter | Phenomenological fit (QCD critical amplitudes as physics analogue) |

The elastic modulus E = (2/alpha^2) rho_c c^2 = 9.1 x 10^-10 Pa is derived from alpha and the background critical density.

### Key Results

| Metric | MTDF | LCDM |
|--------|------|------|
| Scalar pillars chi^2/nu | 0.11 (15 tests, benchmark tier) | Not directly comparable: galaxy-scale phenomenology in LCDM requires hydrodynamic simulation; per-pillar literature context in MTDF_06 |
| Strict combined chi^2/nu, scalar + vector (same pipeline, same data, same covariances) | 1.1683 (nu = 1745, strict, excluding CMB distance prior diagnostic) | In progress (v1.1.6 addendum): frozen Planck-2018 LCDM through the identical V18 pipeline |
| S8 from full Planck likelihood | 0.802 +/- 0.013 | 0.830 +/- 0.013 |
| Exotic components | None | Cold dark matter + cosmological constant |
| Local H0 | 73.1 +/- 1.0 (conditional on a deep local underdensity; ~70-71 for shallow reconstructions; MTDF does not predict the void itself) vs SH0ES 73.04 +/- 1.04 | 67.4 (CMB-inferred) vs 73.0 (local): unresolved tension |
| Stress coupling k_f (full Planck likelihood) | 0.495 +/- 0.36, 95% CI [0.025, 1.34]; the interval marginally excludes 0 and includes 1; the posterior neither favours nor excludes the coupling (Delta-chi2 = +0.63, Delta-BIC = +8.05, LCDM mildly preferred by the Occam penalty) | Not in model (k_f = 0 limit) |

---

## Reproduce (One-Command Flow)

```bash
# After obtaining the repository (git clone, zip download, or Zenodo):
cd MTDF
bash setup_environment.sh              # Python venv + dependencies
bash scripts/download_data.sh          # External data (~19 GB, SHA256-verified)
source venv/bin/activate
python validation/code/run_validate.py \
    --workbook validation/data/DB_Workbook_STRICT_V18.xlsx \
    --out validation/output/My_Dashboard.html \
    --diag validation/output/My_Diagnostics.csv
```

Open `validation/output/My_Dashboard.html` in a browser and compare against the pre-computed `Validation_Dashboard_V74.html`. Expected result: all 15 scalar pillars pass, combined chi-squared/nu = 1.17 (DOF = 1745).

### Step-by-step breakdown

**1. Setup**

```bash
bash setup_environment.sh    # Creates venv, installs all Python dependencies
```

**2. Download external data**

External scientific datasets are not redistributed. Download scripts fetch them from their official archives with SHA256 verification. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full provenance.

```bash
bash scripts/download_data.sh          # All datasets (~19 GB)

# Or download individually:
bash scripts/download_pantheonplus.sh  # 33 MB  (Pantheon+ SNe)
bash scripts/download_desi_voids.sh    # 1.2 GB (DESI VAST voids)
bash scripts/download_planck.sh        # 482 MB (Planck lensing)
bash scripts/download_kids.sh          # 17 GB  (KiDS-1000 WL)
bash scripts/download_pittordis.sh     # 164 MB (Wide binaries)
bash scripts/download_bao.sh           # ~60 MB (BAO + CC + growth + BOSS voids)
bash scripts/download_ztf_dr2.sh       # ~5 MB  (ZTF DR2 cosmology sample)
bash scripts/download_foundation_dr1.sh # ~30 MB (Foundation DR1 SNe)

bash scripts/verify_checksums.sh       # Verify all downloads
```

**3. Reproduce the validation dashboard**

```bash
source venv/bin/activate
python validation/code/run_validate.py \
    --workbook validation/data/DB_Workbook_STRICT_V18.xlsx \
    --out validation/output/My_Dashboard.html \
    --diag validation/output/My_Diagnostics.csv
```

**4. Reproduce the gravity sector (24 steps)**

```bash
cd gravity && bash run_all_steps.sh
```

**5. Reproduce GPU validation (Phases 1-6)**

Requires CUDA-capable GPU. See `gpu_validation/` for per-phase scripts.

**6. Reproduce SN x void hardening tests (merged sample)**

Requires external data from Step 2 (Pantheon+, DESI voids, ZTF DR2, Foundation DR1).

```bash
source venv/bin/activate
# Build the merged Pantheon+/ZTF/Foundation sample
python validation/code/analysis/sn_void_hardening/build_merged_sample.py
# Run all 6 hardening tests (uses multiprocessing, ~10 min on 16 cores)
python validation/code/analysis/sn_void_hardening/run_merged_hardening.py
# Or run Pantheon-only tests
python validation/code/analysis/sn_void_hardening/run_all_hardening.py
```

Expected result: z ~ 0.04 piecewise transition yields Dchi2 = 36-39 (p < 0.001; 36.8 nominal, diagonal covariance) across all three void catalogues; constant-gamma model is null; population controls shift gamma by < 0.3 sigma.

**7. Reproduce Phase 5 MCMC (Planck likelihood)**

Requires cobaya + Planck likelihood. See `mcmc_results/README.md`.

```bash
bash scripts/install_cobaya.sh         # Install cobaya + Planck likelihood
```

---

## Repository Structure

```
MTDF/
|
|-- README.md                   # This file
|-- LICENSE                     # GPL-3.0 License
|-- CITATION.cff                # Machine-readable citation metadata
|-- THIRD_PARTY_NOTICES.md      # Upstream citations and data provenance
|-- requirements.txt            # Python dependencies
|-- setup_environment.sh        # One-command setup
|
|-- papers/                     # Published papers
|   |-- HTML/                   # Paper suite (HTML)
|   |   |-- MTDF_01_*.html      # Main theory paper
|   |   |-- MTDF_02_*.html      # Environmental/local phenomenology
|   |   |-- MTDF_03_*.html      # Gravity sector & lensing validation
|   |   |-- MTDF_04_*.html      # Photon coupling & redshift (speculative)
|   |   |-- MTDF_05_*.html      # Cosmological validation & high-energy
|   |   |-- MTDF_06_*.html      # Validation suite appendix
|   |   |-- MTDF_07_*.html      # Independent GPU validation
|   |   |-- MTDF_08_*.html      # Multi-probe low-redshift transition
|   |   |-- MTDF_The_Mesche_Hypothesis_short.html  # Summary paper
|   |   +-- MTDF_Executive_Briefing.html
|   |-- LaTex/                  # LaTeX sources
|   |-- PDF_from_Tex/           # PDF builds from LaTeX
|   |-- arxiv_zips/             # arXiv submission packages
|   +-- figures/                # Shared paper figures
|
|-- validation/                 # Core V74 validation engine
|   |-- code/                   # run_validate.py + UI modules
|   |   +-- analysis/           # SN x void scripts
|   |       |-- sn_void_*.py    # Original Pantheon+ GLS analysis (5 scripts)
|   |       +-- sn_void_hardening/  # Merged-sample hardening suite (6 tests)
|   |           |-- common.py           # Shared infrastructure
|   |           |-- build_merged_sample.py  # Merge Pantheon+/ZTF/Foundation
|   |           |-- test_scrambled_voids.py     # Test 1: null geometry
|   |           |-- test_fake_z_transition.py   # Test 2: z~0.04 transition
|   |           |-- test_population_controls.py # Test 3: x1/c covariates
|   |           |-- test_wrong_sign_metric.py   # Test 4: inverted metric
|   |           |-- test_alt_environment_metrics.py # Test 5: IDW etc.
|   |           |-- test_cross_catalogue_overlap.py # Test 6: LOO overlap
|   |           |-- run_all_hardening.py    # Pantheon-only orchestrator
|   |           +-- run_merged_hardening.py # Merged-sample orchestrator (MP)
|   |-- analysis/               # CMB analysis scripts
|   |-- scripts/                # Audit utilities
|   |-- data/                   # DB_Workbook_STRICT_V18.xlsx + SPARC
|   +-- output/                 # Pre-computed results (dashboard, phases 1-6)
|
|-- gravity/                    # MTDF_03 gravity sector (24 steps)
|   |-- code/                   # step01-step24 Python scripts
|   |-- data/                   # Digitised data, published supplementary
|   |-- output/                 # Pre-computed step results
|   +-- run_all_steps.sh
|
|-- gpu_validation/             # Independent GPU pipeline (Phases 1-6)
|   |-- phase1/ - phase6/      # Per-phase scripts
|   |-- utils/                  # Shared utilities
|   +-- results/                # Pre-computed GPU results
|
|-- class_mtdf/                 # Modified CLASS Boltzmann solver
|   |-- MODIFICATIONS.md        # Exact changes vs vanilla CLASS v3.2
|   +-- [CLASS source tree]
|
|-- mcmc_results/               # Phase 5 MCMC configurations + results
|   |-- *.yaml                  # Cobaya chain configs
|   |-- *.json                  # Convergence + comparison results
|   +-- README.md               # Reproduction instructions
|
+-- scripts/                    # Download & setup utilities
    |-- download_data.sh        # Master download script
    |-- download_*.sh           # Per-dataset scripts
    |-- verify_checksums.sh     # SHA256 verification
    +-- install_cobaya.sh       # Cobaya + Planck likelihood setup
```

---

## Paper Suite

| Paper | Title | Scope |
|-------|-------|-------|
| MTDF_01 | The Mesche Hypothesis: A Dynamic Field Theory of Cosmic Structure | Main theory, 15 scalar pillars, 4 vector likelihoods |
| MTDF_02 | Environmental and Local Phenomenology | SN x void analysis, local H0 |
| MTDF_03 | Gravity Sector and Lensing Validation | 24-step galaxy-galaxy lensing pipeline |
| MTDF_04 | Photon Coupling, Redshift, Early Universe | Speculative extensions |
| MTDF_05 | Cosmological Validation, High-Energy Test Programme | Speculative high-energy predictions |
| MTDF_06 | Validation Suite Appendix | Detailed derivations for all 15 pillars |
| MTDF_07 | Independent GPU Validation | Phases 1-6 GPU pipeline results |
| MTDF_08 | Multi-Probe Evidence for an Environment-Dependent Transition at z < 0.04 | Cross-probe tests of the z ~ 0.04 transition |
| Short Summary | The Mesche Hypothesis (Short) | Concise overview for reviewers |

---

## External Data Sources

All external datasets are downloaded from their official archives. We do not redistribute third-party data.

| Dataset | Size | Archive | Used By |
|---------|------|---------|---------|
| Pantheon+ (Brout+2022) | 33 MB | GitHub | Phase 3 (SN x void) |
| ZTF DR2 cosmology (Rigault+2024) | ~5 MB | ztfcosmo package | SN x void hardening |
| Foundation DR1 (Jones+2019) | ~30 MB | GitHub | SN x void hardening |
| DESI VAST void catalogues | 1.2 GB | DESI data release | Phase 3, Phase 6 |
| Planck 2018 lensing maps | 482 MB | ESA Planck Legacy Archive | Phase 5, Phase 6D |
| KiDS-1000 WL catalogue | 17 GB | ESO archive | Phase 6B (WL x voids) |
| Pittordis+2023 wide binaries | 164 MB | CDS/VizieR | Gravity step 19 |
| SDSS DR16 BAO | ~60 MB | SDSS data release | Validation pillars |
| Cosmic chronometers | ~1 MB | Published compilations | Validation pillars |
| Growth f-sigma8 (eBOSS) | ~20 KB | eBOSS DR16 | Validation pillars |
| BOSS DR12 voids | ~140 KB | BOSS data release | Phase 6E |

---

## Data Included in This Repository

The following data files are our own work product and are included directly:

- `validation/data/DB_Workbook_STRICT_V18.xlsx` (109 KB): Single source of truth for all parameters, targets, and formulas
- `validation/data/sparc_clean.json` (585 KB): Our cleaned SPARC galaxy compilation
- `gravity/data/digitised/*.csv` (3.6 KB): Our extractions from published figures
- All `output/` directories: Pre-computed results (JSON, PNG, HTML dashboards)

Published supplementary data included with attribution:
- `gravity/data/brouwer2021/` (~2 MB): Brouwer+2021 galaxy-galaxy lensing data tables
- `gravity/data/mandelbaum2016/` (~15 KB): Mandelbaum+2016 SDSS ESD profiles

---

## Citation

If you use this work, please cite using the metadata in [CITATION.cff](CITATION.cff):

```
Mesche, I. (2026). The Mesche Hypothesis: A Dynamic Field Theory of Cosmic Structure.
```

See also: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for required upstream citations (CLASS, Planck, etc.).

---

## License

GPL-3.0 License (required for CLASS compatibility). See [LICENSE](LICENSE) for details.

External datasets are subject to their respective archive terms of use.

---

*MTDF V74 / Workbook V18 / June 2026*
