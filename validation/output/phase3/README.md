# Phase 3: SN x Void Environment Signal

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Purpose:** Independently confirm the SN x void environment signal reported in MTDF_02, using three void catalogues (VoidFinder, REVOLVER, VIDE) and the Pantheon+ dataset.

**Dataset:** Pantheon+ SNe Ia (564 SNe after z-cuts), DESI void catalogues

**Key result:** REVOLVER signed-distance fit: gamma_env = +0.0047, Delta-chi-squared = 4.25, p = 0.039. All three catalogues show positive gamma_env with 95% bootstrap CI excluding zero. Signal confined to z < 0.04.

**Canonical results file:** `phase3_summary.json`

**Referenced in:** MTDF_07 Section 4

## Files

| File | Description |
|------|-------------|
| `phase3_summary.json` | Complete results: per-catalogue fits (signed + binary), NGC/SGC split, survey fixed effects, LOSO stability, z-modulation, permutation and bootstrap tests, CPU cross-check |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes for all files in this folder |

## Key numbers in phase3_summary.json

- `table1.<catalogue>.signed` — main gamma_env fits and delta-chi-squared per catalogue
- `permutation_bootstrap.<catalogue>` — non-parametric significance tests
- `z_modulation.<catalogue>.z_bins` — z-binned signal (strongest in first bin, z < 0.04)
- `table4_loso.voidfinder.loso` — leave-one-survey-out stability (16/16 positive)
