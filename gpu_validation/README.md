# GPU Validation Pipeline

**Author:** Ingo Mesche
**Affiliation:** Independent Researcher, Malta
**Framework:** MTDF V74

Independent validation of MTDF against Planck CMB data using GPU-accelerated
MCMC sampling, CosmoPower neural emulators, and the modified CLASS Boltzmann solver.

---

## Pipeline Phases

| Phase | Description | Key Output |
|-------|-------------|------------|
| 1 | Single-point CLASS validation (MTDF vs LCDM spectra) | `results/phase1/phase1_results.json` |
| 2 | CosmoPower emulator MCMC (Planck plik-lite + BAO + SNe) | `results/phase2/mcmc_combined_summary.json` |
| 3 | Parameter recovery and goodness-of-fit | `results/phase3/phase3_summary.json` |
| 3b | Asymmetry and detectability diagnostics | `results/phase3b/PHASE3B_REPORT.md` |
| 4 | Derived parameter consistency | `results/phase4/phase4_summary.json` |
| 5 | Full Planck plik MCMC (production chains, R-1 < 0.02) | `../validation/output/phase5/` |
| 6 | Discriminator tests (environment, lensing, growth) | `../validation/output/phase6/` |

## Pre-computed Results

All phase results are included as JSON summaries, plots, and reports in
`results/` and `../validation/output/phase5/` and `phase6/`. A reviewer can
fully evaluate the claims by inspecting these outputs without re-running.

## Reproducing from Scratch

### Phase 1 (no external dependencies)

```bash
source ../venv/bin/activate
python phase1/run_phase1.py
```

Requires compiled `class_mtdf` (see `../class_mtdf/README.md`; run `make class`).

### Phases 2-4 (require CosmoPower models)

The trained CosmoPower neural network models (TT, TE, EE emulators) are not
included in the repository due to size (111 MB). To reproduce phases 2-4:

**Option A: Download from Zenodo (when available)**
Models will be deposited at the same Zenodo record as the main repository.
Place the `.npz` files in `phase2/models/`:
- `TT_v1.npz`, `TT_v1.pkl`
- `TE_v1.npz`, `TE_v1.pkl`
- `EE_v1.npz`, `EE_v1.pkl`

**Option B: Contact the author**
Email the corresponding author (see CITATION.cff) to request the model files.

Once models are in place:
```bash
python phase2/run_mcmc_combined.py    # Combined MCMC
python phase3/run_phase3.py           # Parameter recovery
python phase4/run_phase4.py           # Derived consistency
```

### Phase 5 (requires Cobaya + Planck likelihood)

Phase 5 uses the full Planck plik likelihood (not the lite version) via Cobaya.

```bash
bash ../scripts/install_cobaya.sh     # Install Cobaya + Planck likelihood
cd phase5_plik
bash launch_mcmc.sh                   # Launch MCMC chains (GPU recommended)
python analyze_phase5.py              # Analyze converged chains
```

Production chains require ~48 hours on a modern GPU. Pre-computed results
are in `../validation/output/phase5/`.

### Phase 6 (requires external data)

```bash
bash ../scripts/download_data.sh      # Download all external datasets
python phase6/testA_redshift_transition.py
# ... (see individual test scripts)
```

## Configuration

Central configuration is in `config.py`. All MTDF parameters are read from
`../validation/data/DB_Workbook_STRICT_V18.xlsx` (no hardcoded physics values).

## Directory Layout

```
gpu_validation/
├── config.py              # Central configuration (paths, parameter loader)
├── phase1/                # Single-point CLASS validation
├── phase2/                # CosmoPower emulator MCMC
│   ├── models/            # Trained NN emulators (not in repo, see above)
│   ├── cosmopower_setup.py
│   └── run_mcmc_combined.py
├── phase3/                # Parameter recovery
├── phase3b/               # Asymmetry diagnostics
├── phase4/                # Derived consistency
├── phase5_plik/           # Full Planck MCMC (production)
├── phase6/                # Discriminator tests
├── prediction_pack/       # Pre-registered predictions
├── results/               # All phase outputs (JSON, plots)
└── utils/                 # Shared utilities
```
