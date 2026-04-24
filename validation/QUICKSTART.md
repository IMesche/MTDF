# MTDF Validation - Quick Start Guide

**Author:** Ingo Mesche
**Affiliation:** Independent Researcher, Malta
**Date:** December 2025
**Source of truth:** DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html

---

## For Reviewers: 3-Minute Setup

### Step 1: Install Python Dependencies (30 seconds)

```bash
# From the repository root:
bash setup_environment.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Step 2: Generate Dashboard (5 seconds)

```bash
cd validation/code
python run_validate.py \
    --workbook ../data/DB_Workbook_STRICT_V18.xlsx \
    --out ../output/Validation_Dashboard_V74.html \
    --diag ../output/Diagnostics.csv
```

**Expected output (without external data):**
```
Run complete: 2026-XX-XX XX:XX:XX
  Proof: 15/16 = 93.8% at 1.0 sigma
```

This runs the 15 scalar pillars. One pillar (P1) shows a cosmetic fallback
message but produces the correct value. All 15 pass at 1 sigma.

**To also run the 5 vector likelihood pillars** (Pantheon+ SNe, DESI BAO,
cosmic chronometers, growth rate, CMB distance prior), first download the
external datasets:

```bash
bash scripts/download_data.sh     # ~19 GB total, SHA256-verified
# Or download only the small ones needed for vector pillars (~100 MB):
bash scripts/download_pantheonplus.sh
bash scripts/download_bao.sh
```

### Step 3: View Results (instantly)

Open `validation/output/Validation_Dashboard_V74.html` in your browser.

**What you'll see:**
- Interactive table showing MTDF passing all 15 scalar tests (green)
- 5 vector likelihood pillars (require external data download)
- Comparison models (LCDM, MOND, etc.) failing significantly (red)
- Hover over pillar names to see formulas and data sources
- Statistical summary: scalar chi^2/nu = 0.11, combined chi^2/nu = 1.17

---

## What Each File Does

### Essential Files (Read These First):
1. **README.md** (repo root) - Complete documentation
2. **validation/data/DB_Workbook_STRICT_V18.xlsx** - All parameters, formulas, and targets
3. **validation/output/Validation_Dashboard_V74.html** - Pre-generated results (open now!)
4. **validation/README.md** - Detailed validation methodology

### Supporting Files:
- **validation/code/run_validate.py** - Validation engine (well-commented)
- **validation/code/analysis/** - SN x void environment analysis suite
- **class_mtdf/MODIFICATIONS.md** - All changes to the CLASS Boltzmann code
- **papers/*.html** - Theoretical papers (self-contained, open in browser)

---

## Key Questions Answered

**Q: Where are the formulas?**
Open `validation/data/DB_Workbook_STRICT_V18.xlsx`, **Pillar_Formulas** sheet

**Q: Where are the parameter values?**
Same workbook, **Params_Fundamental** sheet (4 parameters: alpha, beta, tau, beta_eos)

**Q: Are these free parameters?**
NO. All parameters are empirically constrained or fixed by calibration, then held constant. No post-hoc fitting to the 15 validation tests.

**Q: What if I want to change a parameter?**
Edit the workbook, save, re-run `run_validate.py`. Dashboard updates automatically.

**Q: Where are the citations?**
Workbook, **Pillar_Targets** sheet (source_doi column)
OR hover over pillar names in the dashboard

**Q: How do I download external data?**
Run `bash scripts/download_data.sh` from the repo root (~19 GB, SHA256-verified).
Individual datasets can be downloaded separately; see `scripts/download_*.sh`.

---

## The Bottom Line

**MTDF achieves:**
- **Scalar pillars:** chi^2/nu = 0.11 (15 tests, all pass at 1 sigma)
- **Combined strict:** chi^2/nu = 1.17 (DOF = 1745, includes vector data)
- **4 fundamental parameters** (alpha, beta, tau, beta_eos) - independently measured
- **Zero free parameters** - all values empirically fixed, not fitted
- **No exotic components** - no dark matter or dark energy
- **Complete reproducibility** - all values in workbook with provenance

**Standard LCDM model:** chi^2/nu = 58.5 on scalar pillars, requires unobserved dark sector

**Critical advantage:** MTDF parameters cannot be arbitrarily adjusted. They are fixed by independent measurements and calibration procedures, ensuring genuine predictive power rather than post-hoc curve fitting.

---

**Next Steps:** See the top-level README.md for detailed methodology, theoretical documentation, and complete validation protocol.
