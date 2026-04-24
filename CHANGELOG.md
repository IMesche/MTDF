# Changelog

Reviewer-facing audit trail for the MTDF submission package. Scope: numerical
and scientific corrections, textual and metadata changes, packaging, manifest
regeneration, and reproducibility verification. Entries are factual and
technical. Personal development history is out of scope.

The submission package corresponds to theory version **V74** and validation
workbook version **V18**.

## [v1.1.1] (2026-04-24)

- Integrated reserved Zenodo DOI 10.5281/zenodo.19741059 (record 19741059)
  into the submission package.
- Patched `scripts/download_cosmopower_models.sh` with the final Zenodo
  record ID, direct file URLs (`?download=1`), and SHA-256 hashes for the
  three emulator cache files `TT_v1.npz`, `TE_v1.npz`, `EE_v1.npz`.
- Added a provenance / licence note in the download script: the `.npz`
  files are generated emulator cache artefacts redistributed as MTDF
  reproducibility artefacts only, and inherit the non-commercial
  research-use conditions of the CosmoPower-derived workflow.
- `CITATION.cff`: added `doi` and `identifiers` fields pointing at
  10.5281/zenodo.19741059. Version bumped from 1.1.0 to 1.1.1.
- `README.md`: added `Release` and `DOI` rows to the top-level summary
  table.
- `THIRD_PARTY_NOTICES.md`: added a CosmoPower section covering the
  upstream library, the Zenodo-deposited emulator cache files, and their
  non-commercial research-use conditions.
- Re-froze the public release package after DOI integration.

## [v1.1.0] (2026-04-24)

Everything below in the [V74-submission-cleanup] entry is the v1.1.0
release candidate content; the v1.1.1 entry above contains only the
Zenodo DOI integration applied on top of it.

## [V74-submission-cleanup] (2026-04-24)

Scientific integrity

- **MTDF_05 Δχ² wording correction.** Abstract and introduction previously
  stated a total Δχ² = +1.25 relative to ΛCDM from the Phase 5 full Planck
  comparison. The full Planck Phase 5 total is Δχ² = +0.63, as already
  reported in the paper's own chi-squared breakdown table and in the model
  selection table (the +1.25 figure is the lensing-likelihood *component* of
  that breakdown, and also coincidentally the Δχ² of the narrower CMB+BAO
  summary run in Section 2.1). The abstract, introduction, Section 2.1
  prose, and Section 2.4 breakdown paragraph have been reworded to
  disambiguate the two occurrences of +1.25 and to lead with the correct
  full-Planck total of +0.63.
- **No numerical tables changed for that correction.** All chi-squared and
  parameter tables in MTDF_05 were already correct; only the prose around
  them was updated.
- **Mirror edits** applied in `MTDF_00_Master_Compilation.tex` and in the
  MTDF_07 summary row that references the MTDF_05 result.
- **MTDF_03 bibliography** gained two missing DOIs: Planck 2018 VI
  (10.1051/0004-6361/201833910) and Wong et al. 2020 H0LiCOW XIII
  (10.1093/mnras/stz3094). Mirrored in the Master Compilation bibliography
  and in the HTML render.

Gravity-sector artefact integrity

- **`gravity/MANIFEST.sha256` regenerated.** All 97 entries now verify OK.
  The previous manifest (generated 2026-02-18) had 40 JSON hashes that no
  longer matched: those JSON files were regenerated on 2026-04-03 when a
  provenance metadata block (`_meta: {author, affiliation, framework}`) was
  added to each. The 57 PNG figure artefacts were unchanged. The numerical
  payload of every JSON was unchanged; only provenance metadata was added.
- **`gravity/MANIFEST_NOTES.md` added** to document the regeneration
  rationale, the freeze policy, and the verification procedure.
- **`gravity/VERIFY_MANIFEST.txt` added** as a frozen witness of the
  `sha256sum -c` pass at release time.

Download-script repair

- **`download_pantheonplus.sh`**: upstream directory rename on
  `PantheonPlusSH0ES/DataRelease` main branch, all five occurrences of
  `Pantheon+_Data/4_DISTANCES_AND_COVARIANCES/` updated to
  `Pantheon+_Data/4_DISTANCES_AND_COVAR/`. Upstream file URLs now return 200.
- **`download_pittordis.sh`**: the Zenodo record 7629240 renamed the file
  from `pittordis2023_wb.csv` to
  `CleanedWB_EDR3_Prlx300pc_Gmag20_20230111_Size73087_ZenodoSample.csv`. The
  script now requests the new filename. Direct verification confirmed the
  file is byte-identical to the previous release (SHA-256
  `826cbb2c73768515b883a4ff56588ab148592d5f5e11fb577045faba1d9bb8d9`,
  170,948,678 bytes, 73,088 rows); the local target filename is preserved
  for downstream-pipeline compatibility.
- **Cosmic Chronometer source migrated** from
  `github.com/music-group/CC_covariance` (404, repository deleted or moved)
  to `gitlab.com/mmoresco/CCcovariance` (`master/data/`), the
  author-maintained public source. Three files now fetch cleanly.
- **SDSS/eBOSS DR16 BAOplus source migrated** from `cmbant/CosmoMC/master/data`
  (path no longer exposed) to `github.com/CobayaSampler/bao_data`, the
  current verified public mirror/source used by this analysis. Four files
  (LRG + QSO, data + covariance) now fetch cleanly.
- **`download_bao.sh` hardened** with a `safe_fetch` helper (rejects empty
  and HTML-error-page responses so partial failures do not leave corrupted
  placeholders) and with per-file existence-and-size gates for the CC trio,
  the DR16 BAOplus quartet, and the BOSS voids fetch. Deleting any single
  file now triggers a refetch of only that file rather than the whole group.
- **`scripts/DOWNLOAD_URL_AUDIT_20260424.md` added** with per-file
  provenance: primary source URL, access date, upstream citation,
  redistribution note, size, and SHA-256. Includes a byte-stability note on
  the CDS/VizieR BOSS voids file whose FITS header contains a serve-time
  timestamp.

Repository hygiene and metadata

- `.gitignore` added at repository root (Python + LaTeX + CLASS build
  outputs + Zenodo-hosted emulator caches + external-data directories + OS
  and editor junk).
- `CITATION.cff` version bumped to 1.1.0 and `date-released` set to
  2026-04-24.
- `scripts/download_cosmopower_models.sh` added as a SHA-256-verified
  Zenodo fetch stub (record ID and hashes to be filled once the MTDF
  Zenodo deposit is minted).
- `gpu_validation/phase2/models/` cleaned: the six `.npz`/`.pkl` emulator
  files (~113 MB total) were moved out of the tree; only the folder
  `README.md` remains, which points at the download script and the Zenodo
  deposit.
- Duplicated corner-plot PNGs removed from `validation/output/phase2/`
  (they were byte-identical mirrors of files in
  `gpu_validation/results/phase2/`, which is the canonical location for
  MTDF_07 GPU-validation outputs).
- Leftover CLASS documentation build artefacts
  (`class_mtdf/doc/input/latex/class.{aux,synctex.gz}`) removed from the
  tree.

Reproducibility verification

- **`class_mtdf` C binary rebuilt cleanly from source** in an isolated
  scratch copy outside the repository. 57 compile units produced a 9.9 MB
  `class` executable. The Python `classy` wrapper installs via the
  Makefile's final `python -m pip install .` step when run inside the venv
  created by `setup_environment.sh`.
- **`run_validate.py` executed in a clean venv built from
  `requirements.txt`.** The workbook-only path reproduced the shipped
  `validation/output/Diagnostics.csv` to machine precision (15 scalar
  pillars, one 1-bit IEEE-754 rounding on P2 in the last digit of a
  double). A targeted external-data run was then performed after URL
  repair and per-file gating, adding the CC H(z) and DR16 fσ₈ vector
  pillars.

## [V74-paper-suite] (2026-04-23)

- Paper suite finalised: master theory paper MTDF_01, companions
  MTDF_02 (environmental), MTDF_03 (gravity sector and lensing validation),
  MTDF_04 (photon coupling and early universe), MTDF_05 (cosmological
  validation), MTDF_06 (validation-suite appendix), MTDF_07 (independent
  GPU validation), MTDF_08 (multi-probe low-redshift transition), plus the
  Executive Briefing and Short Summary.
- Claim-status language harmonised across papers. Each paper separates
  calibration anchors, validation targets, diagnostic consistency checks,
  and speculative extensions. Only validation-target results are used as
  evidence for the core MTDF claim.
- Repository-code URL in `CITATION.cff` set to `github.com/IMesche/MTDF`.
- Combined-document (`MTDF_00_Master_Compilation`) rendering checked against
  its per-paper sources for table and cross-reference integrity; issues
  identified and fixed in generated artefacts where applicable.

## [V18-strict-protocol / V74-dashboard] (February to April 2026)

- Strict separation of calibration anchors, benchmarks, validation targets,
  and diagnostic consistency checks is enforced throughout the
  `validation/` pipeline and in the workbook (`DB_Workbook_STRICT_V18.xlsx`).
- The scalar-pillar set and the vector-likelihood set are treated as
  distinct test families. Scalar pillars (z-score consistency checks)
  summarise to `χ²/ν ≈ 0.11` at DOF 14; the combined strict statistic
  (scalar plus vector likelihoods, including Pantheon+, DESI Y1 BAO,
  cosmic-chronometer H(z), DR16 fσ₈, and CMB distance prior) totals
  `χ²/ν = 1.17` at DOF 1745.
- The CMB distance prior is treated as a diagnostic in the scorecard and
  is not counted in the strict-fit totals for either scalar or vector
  families.
- The Validation_Dashboard_V74.html render and the V18 workbook are
  aligned to this protocol.

## [Phase 5 Planck validation] (2026)

- Full Planck 2018 plik TTTEEE + low-ℓ + lensing hard-falsifier executed
  using the `class_mtdf` modified Boltzmann solver. Minimisation via
  cobaya BOBYQA (best of four), floating six cosmological parameters, one
  MTDF parameter k_f, and 21 Planck nuisance parameters.
- Δχ² = +0.63 relative to ΛCDM at the Phase 5 minimum, distributed across
  likelihood components as: high-ℓ plik TTTEEE −3.67, low-ℓ TT +2.65,
  low-ℓ EE +0.39, lensing +1.25. Total chi-squared 2773.82 (MTDF) vs
  2773.20 (ΛCDM).
- MCMC converged with R−1 = 0.019 < 0.02. k_f posterior is
  0.50 ± 0.36; both k_f = 0 (ΛCDM) and k_f = 1 (full MTDF) lie inside the
  95% credible interval. The posterior is broad and non-pathological.
- ΔBIC = +8.05 reflects the Occam penalty of the additional parameter; it
  is not a statistical exclusion. ΔAIC = +2.63 is in the inconclusive
  regime (|ΔAIC| < 4).
- Sprint 1 robustness (leave-one-likelihood-out, prior sensitivity, k_f
  identifiability) is reported in MTDF_07 §5 with numerical posteriors for
  baseline / no-lensing / TT-only subsets.

---

Historical exploratory versions prior to the frozen V18/V74 submission
package are not treated as submission artefacts and are not exhaustively
listed here.
