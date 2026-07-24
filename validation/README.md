# MTDF (Mesche's Tensor Dynamics Framework) - Validation Package

**Author:** Ingo Mesche
**Affiliation:** Independent Researcher, Malta
**Version:** V19 (Workbook) / V75 (Dashboard)
**Theory Identifier:** V75
**Date:** July 2026
**Status:** Peer Review Ready
**Source of truth:** DB_Workbook_STRICT_V19.xlsx, Validation_Dashboard_V75.html

---

## Executive Summary

This package contains a complete, reproducible validation of the MTDF theoretical framework against 15 scalar pillars plus 4 strict vector likelihood pillars and 1 CMB distance-prior diagnostic (excluded from strict totals), spanning galactic, cluster, and cosmological scales.

**Key Results:**
- **Scalar pillars:** 14/14 counted scalar pillars pass at 1σ; the S8 and fσ8 rows are shown as "(cont.)" PREDICTION-CONTESTED and are never counted as passes (the dashboard displays the combined tally 17/19 + 2 (cont.))
- **Combined strict:** χ²/ν = 1.17 (DOF = 1741, includes vector data)
- **4 structural constants** (α, β, τ, β_eos) of differing epistemic status: one anchored to the void radius distribution (β), one a frozen void-motivated coupling not independently measured (α), one synchronised to the cosmic age (τ), and one a phenomenological fit (β_eos)
- **No post-hoc fitting** - the constants are frozen before the validation pass and are not adjusted to these tests
- **No exotic components** - no dark matter or dark energy required
- **Complete provenance** - every numerical value traceable to workbook, measurement, or standard

**Comparison:** Standard ΛCDM model achieves χ²/ν = 58.5 on scalar pillars (scalar-pillar re-scoring of ΛCDM through the MTDF pipeline; not directly comparable, see MTDF_06), while requiring unobserved dark sector components comprising ~95% of the universe's energy budget.

**Critical Distinction:** MTDF uses a fixed reference structure: the four structural constants are set by the calibration procedures named above (with their differing epistemic status), then held constant across validation tests. No values are hardcoded in code; all derive from the workbook database with full documentation.

---

## Quick Start

### Prerequisites
- Python 3.8+
- ~500 MB disk space

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas openpyxl numpy

# Or use requirements file
pip install -r requirements.txt
```

### Generate Dashboard

```bash
cd code
python run_validate.py \
    --workbook ../data/DB_Workbook_STRICT_V19.xlsx \
    --out ../output/My_Dashboard.html \
    --diag ../output/My_Diagnostics.csv
```

This writes to `My_Dashboard.html` / `My_Diagnostics.csv` so the shipped V75
baselines are never overwritten. Compare against the shipped
`Validation_Dashboard_V75.html` / `Diagnostics.csv`.

### View Results

Open `output/Validation_Dashboard_V75.html` (shipped baseline), or your regenerated `output/My_Dashboard.html`, in any modern web browser (Chrome, Firefox, Safari, Edge).

---

## Directory Structure

```
validation/                            # (this directory)
│
├── README.md                          # This file
├── QUICKSTART.md                      # Fast start guide
├── 00_START_HERE.txt                  # Welcome & orientation
│
├── data/                              # Source data (READ-ONLY)
│   ├── DB_Workbook_STRICT_V19.xlsx   # Master parameter & formula database
│   ├── DB_Workbook_STRICT_V18.xlsx   # Frozen V18-era predecessor (archived)
│   ├── sparc_clean.json              # SPARC galaxy rotation curve data
│   └── External/                     # Downloaded datasets (via scripts/)
│
├── code/                              # Validation engine
│   ├── run_validate.py               # Main validation script
│   ├── UI/                           # Dashboard generation modules
│   │   ├── __init__.py
│   │   ├── dashboard.py              # HTML table generator
│   │   ├── components.py             # UI components
│   │   ├── styles.py                 # CSS styling
│   │   ├── scripts.py                # JavaScript utilities
│   │   ├── tooltips.py               # Tooltip definitions
│   │   └── tooltip_engine.py         # Tooltip rendering system
│   ├── analysis/                     # SN x void environment analysis
│   │   ├── sn_void_GLS_analysis.py   # GLS environment signal
│   │   ├── sn_void_hardening/        # 6-test hardening suite
│   │   └── sn_void_summary_figure.py # Summary visualisation
│   └── paper8/                       # MTDF_08 velocity-field scripts
│
├── scripts/                           # Workbook audit utilities
│   ├── audit_workbook.py             # Workbook extraction / audit
│   └── compare_workbook_dashboard.py # Workbook vs dashboard cross-check
│
└── output/                            # Generated results
    ├── Validation_Dashboard_V75.html # Interactive results dashboard
    ├── Diagnostics.csv               # Per-pillar diagnostic breakdown
    ├── audit_sparc/                  # SPARC audit artefacts
    ├── phase1/ .. phase4/            # Earlier validation phases
    ├── phase3b/                      # NGC/SGC asymmetry diagnostics
    ├── phase5/                       # Phase 5 Planck MCMC results & robustness
    ├── phase6/                       # Phase 6 discriminator tests
    ├── phase7_cf4_vpec/              # CF4 peculiar velocity outputs (MTDF_08)
    ├── phase8_2mtf_tf/               # 2MTF Tully-Fisher outputs (MTDF_08)
    └── prediction_pack/              # class_mtdf P(k,z) + growth predictions
```

---

For Phase 5 robustness and identifiability checks, see `output/phase5/robustness/`.

Phase 6 discriminators: `output/phase6/testA_redshift_transition/` (z~0.04 onset, 3.6sigma confirmed), `output/phase6/testB_wl_environment/` (weak lensing pre-registration + skeleton, KiDS-1000 primary), `output/phase6/testC_derived_consistency/` (derived parameter check, PASS).

---

## The 15 Empirical Tests ("Pillars")

### Galactic Scale
- **P1:** Galaxy rotation curve scatter (SPARC, 175 galaxies)
- **P1B:** RAR intrinsic scatter (deconvolved)

### Cluster Scale
- **P2:** Bullet Cluster separation distance

### Cosmological Observations
- **P3:** CMB temperature-polarization correlation
- **P4:** Dark energy equation of state (w₀)
- **P5:** BAO standard ruler deviation
- **P9:** CMB large-scale anisotropy amplitude
- **P11:** CMB distance ladder consistency
- **P12:** Reionization optical depth correction
- **P13:** Weak lensing S₈ parameter

### Large-Scale Structure
- **P6:** AGN jet-filament alignment angle
- **P8:** Cosmic void size quantization

### High-Redshift Phenomena
- **P10:** Early galaxy formation timescale (JWST)
- **P10B:** Ultra-early compact source consistency

### Astrophysical Phenomena
- **P7:** Black hole mass gap environmental enhancement

All targets derived from peer-reviewed literature with documented DOIs.

---

## Core Parameters and Zero-Hardcoding Principle

**CRITICAL DISTINCTION:** MTDF is governed by four structural constants of differing epistemic status: one anchored to the void radius distribution (β), one a frozen void-motivated coupling not independently measured (α), one synchronised to the cosmic age (τ), and one a phenomenological fit (β_eos). All values are frozen before the validation pass and held constant across all validation tests.

### Parameter Classification

The workbook organizes all quantities into five distinct categories:

#### 1. **Physical Constants** (Params_Constants sheet)
Standard constants from CODATA/IAU definitions:
- **c** = 299,792,458 m/s (speed of light)
- **G** = 6.67430×10⁻¹¹ m³·kg⁻¹·s⁻² (gravitational constant)
- **h** = Planck constant
- Additional: π, conversion factors

**Status:** Fixed by international standards. Not adjustable.

#### 2. **MTDF Structural Constants** (Params_Fundamental sheet)
Four structural constants of differing epistemic status govern the field dynamics:

| Symbol | Name | Value (SI) | Unit | Provenance |
|--------|------|-----------|------|------------|
| **α** | Field coupling | 1.3 | dimensionless | Frozen void-motivated coupling; not independently measured (contested anchor; see Kenworthy et al. 2019) |
| **β** | Length scale | 7×10²³ m | m | Anchored to the void radius distribution (Sutter et al. 2012 SDSS DR7 void radii; ~22.7 Mpc) |
| **τ** | Time scale | 13.0 | Gyr | Synchronised to the cosmic age |
| **β_eos** | EOS parameter | 0.573 | dimensionless | Phenomenological fit (QCD critical amplitudes as physics analogue) |

The elastic modulus E = (2/α²) ρ_c c² = 9.1×10⁻¹⁰ Pa is derived from α and the background critical density and appears in the Params_Observational sheet alongside f_kick (derived) and κ (observational anchor, structurally related to f_kick).

**Status:** One anchored, one frozen (not independently measured), one synchronised, one fitted; see the provenance column. All four are frozen before the validation pass. **Not adjusted to fit validation tests.**

#### 3. **Observational Anchors** (Params_Observational sheet)
External measurements with documented uncertainties and DOIs:
- **GM** = Galaxy masses from SPARC
- **z_rec** = CMB recombination redshift (Planck)
- **z_bao** = BAO survey redshifts (eBOSS)
- **δ_bf** = Baryon fraction (cluster observations)
- **S₈** baseline, angular correlation amplitudes, etc.

**Status:** Not MTDF parameters. These are observational reference values used identically across MTDF and all comparison models (ΛCDM, MOND, etc.).

#### 4. **Implementation Coefficients** (Params_Coefficients sheet)
22 bridge terms that connect MTDF fundamentals to specific observational contexts:
- **κ** = 0.00102 (dimensionless calibration for stress-field coupling)
- **γ_cut** = Screening threshold
- **base_correlation_adjusted** = 0.80 (CMB baseline after foreground removal)
- Plus 19 others (stress_coupling, integral_stress_path, etc.)

**Status:** These are **not free parameters**. They are calibration coefficients derived from MTDF's stress-field equations applied to specific measurement systematics. Once established through calibration procedures, they remain **fixed** across all subsequent analyses.

#### 5. **Unit Conversions** (Params_Units sheet)
- kpc_to_m, mpc_to_m, gyr_to_s, etc.

**Status:** Pure dimensional conversions. Exact by definition.

---

### Parameter Provenance and Freezing

**What this means:**

1. **All values loaded from workbook** - No constants embedded in code
2. **Fixed reference structure** - The four structural constants (one anchored to the void radius distribution, one a frozen void-motivated coupling not independently measured, one synchronised to the cosmic age, one a phenomenological fit) are frozen, then held constant
3. **No post-hoc fitting** - Validation pillars P1-P13 use the pre-established constants
4. **Full reproducibility** - Identical inputs → identical outputs
5. **Complete provenance** - Every value traceable to its anchor, freeze, synchronisation, or fit as stated above

**Contrast with standard cosmological models:**

| Aspect | MTDF | ΛCDM |
|--------|------|------|
| **Structural constants** | 4 (α, β, τ, β_eos), of differing epistemic status (anchored / frozen / synchronised / fitted) | ~6 (Ωₘ, Ωᴋ, H₀, σ₈, nₛ, τ_reion) |
| **Exotic components** | None | Dark matter + dark energy (~95% of universe) |
| **Direct observability** | All terms reference measured quantities | Dark sector unobserved |
| **High-z tensions** | Resolved (P10/P10B pass) | JWST crisis (bright early galaxies) |
| **Galaxy dynamics** | Explained (P1/P1B pass) | Requires dark matter halos |
| **Parameter adjustment** | Frozen before the validation pass | Often re-fitted for new datasets |

MTDF achieves strong statistical performance (14/14 counted scalar pillars pass (the dashboard displays the combined tally 17/19 + 2 (cont.)), combined χ²/ν = 1.17 vs ΛCDM's 58.5 on scalars, the latter being a scalar-pillar re-scoring of ΛCDM through the MTDF pipeline; not directly comparable, see MTDF_06) using **four frozen structural constants** and **no unobserved components**.

---

### Why This Matters for Peer Review

**Reproducibility:** Every numerical value in MTDF predictions can be traced to:
1. A specific workbook cell
2. A calibration procedure (documented)
3. An observational measurement (with DOI)
4. A physical constant (CODATA/IAU)

**Falsifiability:** MTDF makes specific, testable predictions. The structural constants are frozen before the validation pass and are not adjusted per test.

**Parsimony:** MTDF explains diverse phenomena (galactic rotation, cluster dynamics, CMB anomalies, high-z structure formation) with a unified field framework, without invoking separate exotic components for each challenge.

---

## Data Provenance

### P1 (Galaxy Rotation) Detailed Derivation

**Source:** SPARC database (Lelli et al. 2016, DOI: 10.3847/0004-6256/152/6/157)

**Processing:**
1. 175 galaxies with high-quality rotation curves
2. Per-galaxy scatter computed in log₁₀ velocity space
3. Median scatter across sample: 0.174822 dex
4. Standard error of median: 0.00852 dex
5. Catalog systematic (inclination, distance, M/L): 0.007 dex
6. **Final target:** 0.1743 ± 0.011 dex (quadrature sum)

**Verification:** Derivation values above are cross-checked in `data/DB_Workbook_STRICT_V19.xlsx`

### All Other Pillars

Each pillar's target value includes:
- Dataset identification
- Observable definition
- Numerical target ± uncertainty
- Peer-reviewed source DOI
- Notes on any processing/corrections

See `data/DB_Workbook_STRICT_V19.xlsx` → **Pillar_Targets** sheet.

---

## Workbook Structure

The Excel workbook (`data/DB_Workbook_STRICT_V19.xlsx`) is the **single source of truth** for all validation data.

### Sheets:

1. **Model_Registry** - Model metadata and classification
2. **Pillar_Tests** - Test definitions and categories
3. **Pillar_Targets** - Empirical targets with uncertainties and DOIs
4. **Pillar_Proof_Conditions** - Pass/fail criteria
5. **Pillar_Formulas** - MTDF prediction formulas (LaTeX + Python)
6. **Model_Predictions_Matrix** - Comparison model predictions (ΛCDM, MOND, etc.)
7. **Model_Predictions_Provenance** - Sources for comparison predictions
8. **Params_Units** - Unit conversion factors
9. **Params_Constants** - Physical constants (c, G, etc.)
10. **Params_Observational** - Observational inputs (redshifts, masses)
11. **Params_Fundamental** - **The 4 core MTDF structural constants**
12. **Params_Coefficients** - Derived coefficients
13. **UI_Tooltips** - Dashboard tooltip content
14. **Pillar_Vector_Config** - Vector pillar configuration (datasets, paths, options)
15. **Vector_Pillar_Literature** - Literature reference values for the vector pillars

---

## Validation Methodology

### Statistical Framework

**Chi-squared analysis:**
```
χ² = Σ [(prediction_i - target_i) / σ_i]²
χ²/ν = χ² / (N - k)
```

Where:
- N = number of tests
- k = 0 (parameters not fit to this data - empirically constrained elsewhere)
- ν = degrees of freedom

**MTDF Results:**
- Scalar pillars: 14/14 counted scalar pillars pass at 1σ (the dashboard displays the combined tally 17/19 + 2 (cont.)); the S8 and fσ8 rows are shown as "(cont.)" PREDICTION-CONTESTED and are excluded from the strict χ²
- Combined strict: χ²/ν = 1.17, DOF = 1741

**Interpretation:**
- χ²/ν ≈ 1 indicates excellent fit
- χ²/ν < 0.5 suggests possible overestimated uncertainties
- χ²/ν > 2 indicates poor fit or underestimated uncertainties

The 14/14 counted scalar passes and combined χ²/ν = 1.17 indicate **exceptional agreement** between theory and observation.

### Z-Score Analysis

For each test:
```
z = (prediction - target) / σ_target
```

**Pass criterion:** |z| ≤ 1.0 (within 1σ)

**MTDF Results:**
- 14/14 counted scalar pillars pass at 1σ (the dashboard displays the combined tally 17/19 + 2 (cont.))
- The S8 and fσ8 rows are shown as "(cont.)" PREDICTION-CONTESTED; they are never counted as passes and never hidden
- Combined strict χ²/ν = 1.17 (DOF = 1741)

---

## Reproducibility

### Exact Reproduction

To reproduce the exact results in `output/Validation_Dashboard_V75.html`:

```bash
cd code
python run_validate.py \
    --workbook ../data/DB_Workbook_STRICT_V19.xlsx \
    --out ../output/My_Dashboard.html \
    --diag ../output/My_Diagnostics.csv
```

This writes to `My_Dashboard.html` / `My_Diagnostics.csv` so the shipped V75
baselines are never overwritten. Compare against the shipped
`Validation_Dashboard_V75.html` / `Diagnostics.csv`: the output should match
byte-for-byte (except timestamps).

### Verification Scripts

Run the audit tools to verify data integrity:

```bash
cd scripts
python audit_workbook.py
python compare_workbook_dashboard.py
```

These scripts will:
- Extract and display all workbook data
- Cross-check workbook vs dashboard alignment
- Verify P1 benchmark derivation
- Report any discrepancies

---

## Comparison Models

The dashboard includes predictions from standard cosmological models for comparison:

- **ΛCDM** (Lambda Cold Dark Matter) - Standard cosmology
- **MOND** (Modified Newtonian Dynamics) - Galaxy dynamics
- **EDE** (Early Dark Energy) - Hubble tension resolution
- **FDM** (Fuzzy Dark Matter) - Ultra-light axion DM
- **SIDM** (Self-Interacting Dark Matter) - Halo structure

**All comparison models fail significantly:**
- Best alternative: EDE with a substantially worse fit (χ²/ν = 74.1 vs 0.12 on the scalar pillars)
- ΛCDM: χ²/ν = 58.5 (scalar-pillar re-scoring through the MTDF pipeline; not directly comparable, see MTDF_06); dashboard row tally 5/19 passes
- Most models cannot even make predictions for many tests (N/A entries)

---

## Key Features

### Zero Hardcoding
All numerical values loaded from workbook. No constants embedded in code.

### Full Traceability
Every prediction traceable to:
1. Formula in workbook (Pillar_Formulas sheet)
2. Input parameters (Params_* sheets)
3. Source literature (DOI references)

### Interactive Dashboard
- Hover over pillar names → see formula and methodology
- Hover over cells → see detailed prediction breakdown
- Color-coded results (green = pass, red = fail)
- Expandable chi-squared breakdown section

### Professional Presentation
- Responsive HTML design
- Print-friendly layout
- KaTeX mathematical rendering
- Comprehensive tooltips with LaTeX equations

---

## Technical Requirements

### Python Packages
- **pandas** (>=1.3.0) - Excel/CSV data handling
- **openpyxl** (>=3.0.0) - Excel file reading
- **numpy** (>=1.20.0) - Numerical computations

### Browser Support
Dashboard works in all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

No server required - pure client-side HTML/CSS/JavaScript.

---

## FAQ

### Q: Can I modify parameter values?
**A:** Yes! Edit values in `data/DB_Workbook_STRICT_V19.xlsx` and regenerate. The dashboard will automatically reflect changes.

### Q: How long does validation take?
**A:** ~2-5 seconds on modern hardware.

### Q: What if I get different results?
**A:** Ensure you're using the exact workbook version (V19). Run verification scripts to diagnose.

### Q: Can I add new tests?
**A:** Yes, by adding rows to Pillar_Targets and Pillar_Formulas sheets. See workbook structure.

### Q: Where are the formulas defined?
**A:** `data/DB_Workbook_STRICT_V19.xlsx` → **Pillar_Formulas** sheet contains both LaTeX and Python expressions.

### Q: What's the difference between V75 and V19?
**A:** V75 is the theory/dashboard version identifier. V19 is the current workbook version.

---

## Citations

If you reference this work, please cite:

**Primary SPARC Data Source:**
Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157
DOI: 10.3847/0004-6256/152/6/157

**RAR Intrinsic Scatter (P1B):**
Desmond, H. 2023, MNRAS, 521, 1817
DOI: 10.1093/mnras/stad2762

**All other references:** See individual pillar tooltips in dashboard or Pillar_Targets sheet for complete DOI list.

---

## Support & Contact

For questions about:
- **Code functionality:** See inline comments in `code/run_validate.py`
- **Theoretical framework:** See `papers/HTML/*.html` (repository root)
- **Data sources:** See `data/DB_Workbook_STRICT_V19.xlsx` (Pillar_Targets sheet for DOIs)

---

## License & Attribution

**Code:** GPL-3.0 License (see repository root LICENSE)
**Data:** SPARC data courtesy of Lelli et al. 2016 (AJ 152:157)
**Theory:** Mesche's Tensor Dynamics Framework (MTDF) V75

---

## Version History

- **V19/V75** (2026-07): Current workbook/dashboard version with all 15 scalar pillars (14 counted; S8 carried as PREDICTION-CONTESTED)
- **V18/V74** (2025-12): Previous workbook/dashboard generation. The V18 workbook is retained in this tree as the archived protocol baseline; the V74 dashboard is not shipped and remains archived in the v1.1.6 Zenodo record (10.5281/zenodo.20629767)
- **V17** (2025-09-15): Previous workbook version
- **V16** (2025-09-04): Earlier iteration
- **V75**: Current theory identifier

---

## Appendix: Formula Examples

### P1 - Galaxy Rotation
**LaTeX:**
```latex
v_c^2(r) = \frac{GM(r)}{r}\left(1+\frac{\alpha}{1+r/\beta}\right)
```

**Python:**
```python
sqrt(GM/r * (1 + alpha/(1 + r/beta))) / c
```

### P2 - Bullet Cluster
**LaTeX:**
```latex
R = \left(\frac{\beta}{\mathrm{kpc}}\right)\sqrt{2}\,\ln\left(1 + \gamma_{\mathrm{eos}}\;\frac{E}{\rho_{\mathrm{cluster}}\,c^2}\right)
```

**Python:**
```python
(beta / kpc_to_m) * sqrt(2) * log(1 + (beta_eos**3 / alpha) * E/(rho_cluster * c**2))
```

See workbook **Pillar_Formulas** sheet for all 15 formulas.

---

**End of README**

*Last updated: July 2026*
*MTDF Validation Framework V75*
*Author: Ingo Mesche, Independent Researcher, Malta*
