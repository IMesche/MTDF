# MTDF Validation - Quick Start Guide

**Author:** Ingo Mesche
**Affiliation:** Independent Researcher, Malta
**Date:** July 2026
**Source of truth:** DB_Workbook_STRICT_V19.xlsx, Validation_Dashboard_V75.html

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
    --workbook ../data/DB_Workbook_STRICT_V19.xlsx \
    --out ../output/My_Dashboard.html \
    --diag ../output/My_Diagnostics.csv
```

This writes to `My_Dashboard.html` / `My_Diagnostics.csv` so the shipped V75
baselines are never overwritten. Compare your output against the shipped
`Validation_Dashboard_V75.html` / `Diagnostics.csv`.

**Expected result (without external data):** 14/14 counted scalar pillars
pass at 1 sigma; the S8 row is displayed as "(cont.)" PREDICTION-CONTESTED
and is never counted as a pass (the dashboard displays the combined tally
17/19 + 2 (cont.)).

This runs the 15 scalar pillars (14 counted plus the contested S8 row). One
pillar (P1) shows a cosmetic fallback message but produces the correct value.

**To also run the vector likelihood pillars** (4 strict vector pillars:
Pantheon+ SNe, DESI BAO, cosmic chronometers, growth rate; plus 1 CMB
distance-prior diagnostic that is excluded from the strict totals), first
download the external datasets. Run the download scripts from the repository
root:

```bash
cd <repository root>
bash scripts/download_data.sh     # ~19 GB total; checksums for the core
                                  # datasets via scripts/verify_checksums.sh
# Or download only the minimum set for the scriptable vector pillars:
bash scripts/download_pantheonplus.sh   # Pantheon+ SNe
bash scripts/download_planck.sh         # includes the CMB distance prior
bash scripts/download_bao.sh            # cosmic chronometers + growth rate
                                        # (also prints the DESI BAO manual steps)
```

**Manual placement required for two datasets** (their providers do not allow
scripted download): **DESI BAO** (files go under
`validation/data/External/bao_desi/`; `scripts/download_bao.sh` prints the
exact filenames and instructions) and **DESI VAST voids** (see
`scripts/download_desi_voids.sh`). Without the DESI BAO files the combined
strict chi^2/nu = 1.17 headline cannot be regenerated; the shipped
`Validation_Dashboard_V75.html` contains the full result for inspection.

### Step 3: View Results (instantly)

Open `validation/output/Validation_Dashboard_V75.html` (shipped baseline) in
your browser, or `My_Dashboard.html` if you regenerated it in Step 2.

**What you'll see:**
- Interactive table showing MTDF passing all 14 counted scalar tests (green), with the S8 and fsigma8 rows displayed as "(cont.)" PREDICTION-CONTESTED
- 4 strict vector likelihood pillars plus 1 CMB distance-prior diagnostic, excluded from strict totals (these require the external data download)
- Comparison models (LCDM, MOND, etc.) failing significantly (red)
- Hover over pillar names to see formulas and data sources
- Statistical summary: combined strict chi^2/nu = 1.17 (DOF = 1741)

---

## What Each File Does

### Essential Files (Read These First):
1. **README.md** (repo root) - Complete documentation
2. **validation/data/DB_Workbook_STRICT_V19.xlsx** - All parameters, formulas, and targets
3. **validation/output/Validation_Dashboard_V75.html** - Pre-generated results (open now!)
4. **validation/README.md** - Detailed validation methodology

### Supporting Files:
- **validation/code/run_validate.py** - Validation engine (well-commented)
- **validation/code/analysis/** - SN x void environment analysis suite
- **class_mtdf/MODIFICATIONS.md** - All changes to the CLASS Boltzmann code
- **papers/HTML/*.html** - Theoretical papers (self-contained, open in browser)

---

## Key Questions Answered

**Q: Where are the formulas?**
Open `validation/data/DB_Workbook_STRICT_V19.xlsx`, **Pillar_Formulas** sheet

**Q: Where are the parameter values?**
Same workbook, **Params_Fundamental** sheet (4 parameters: alpha, beta, tau, beta_eos)

**Q: Are these free parameters?**
The four structural constants have differing epistemic status: one is anchored
to the void radius distribution (beta), one is a frozen void-motivated coupling
not independently measured (alpha), one is synchronised to the cosmic age
(tau), and one is a phenomenological fit (beta_eos). All are held constant
across the validation tests; there is no post-hoc fitting to the 15 pillars.

**Q: What if I want to change a parameter?**
Edit the workbook, save, re-run `run_validate.py` (writing to `My_Dashboard.html`). Dashboard updates automatically.

**Q: Where are the citations?**
Workbook, **Pillar_Targets** sheet (source_doi column)
OR hover over pillar names in the dashboard

**Q: How do I download external data?**
Run `bash scripts/download_data.sh` from the repo root (~19 GB; checksums for
the core datasets via `scripts/verify_checksums.sh`, coverage listed in the
script). Individual datasets can be downloaded separately; see
`scripts/download_*.sh`. DESI BAO and the DESI VAST voids require manual
placement (see Step 2 above).

---

## The Bottom Line

**MTDF achieves:**
- **Scalar pillars:** 14/14 counted scalar pillars pass at 1 sigma; the S8 and fsigma8 rows are shown as "(cont.)" PREDICTION-CONTESTED (the dashboard displays the combined tally 17/19 + 2 (cont.))
- **Combined strict:** chi^2/nu = 1.17 (DOF = 1741, includes vector data)
- **4 structural constants** (alpha, beta, tau, beta_eos) of differing epistemic status: one anchored to the void radius distribution, one a frozen void-motivated coupling not independently measured, one synchronised to the cosmic age, one a phenomenological fit
- **No post-hoc fitting** - the constants are held fixed across the validation tests, not adjusted to them
- **No exotic components** - no dark matter or dark energy
- **Complete reproducibility** - all values in workbook with provenance

**Standard LCDM model:** chi^2/nu = 58.5 on scalar pillars (scalar-pillar re-scoring of LCDM through the MTDF pipeline; not directly comparable, see MTDF_06), requires unobserved dark sector

**Critical point:** the MTDF constants are frozen before the validation pass and are not adjusted per test; their individual provenance (anchored, frozen, synchronised, fitted) is stated above and in the root README.

---

**Next Steps:** See the top-level README.md for detailed methodology, theoretical documentation, and complete validation protocol.
