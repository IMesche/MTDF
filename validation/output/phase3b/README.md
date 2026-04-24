# Phase 3b: NGC/SGC Asymmetry and Detectability Diagnostics

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Purpose:** Determine whether the observed NGC/SGC contrast in the SN x void signal is physically significant or attributable to survey properties. Five diagnostic tests were executed.

**Dataset:** Same as Phase 3

**Key result:** NGC/SGC asymmetry is not statistically significant, not caused by survey geometry, and attributable to SGC void-catalogue sparsity. All five tests confirm the signal is sensitivity-limited.

**Canonical results file:** `phase3b_summary.json`

**Referenced in:** MTDF_07 Section 5b

## Canonical run

Run `phase3b_20260206_1143` is the canonical run cited in MTDF_07. Three additional runs (1142, 1144, 1145) produced consistent results and are available on request.

## Files

| File | Description |
|------|-------------|
| `phase3b_summary.json` | Full results for all 5 diagnostic tests |
| `test_a_balance.png` | Test A: IPW demographic balance after reweighting |
| `test_c_bootstrap_lr_revolver.png` | Test C: Bootstrap likelihood ratio (REVOLVER) |
| `test_d_mock_delta_gamma_voidfinder.png` | Test D: Null-injection mock delta-gamma distribution |
| `test_e_recovery_revolver_1.0x.png` | Test E: Signal recovery at true amplitude (REVOLVER) |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes for all files in this folder |

## Not included (available on request)

- Full plot sets for all catalogues and amplitude scales (~20 plots per run)
- Three additional timestamped runs (1142, 1144, 1145)
