# CosmoPower Trained Models

This directory should contain the trained CosmoPower neural network emulators
for CMB power spectra (TT, TE, EE). These replace direct CLASS Boltzmann
solver calls during MCMC sampling, providing ~1000x speedup.

## Required Files

| File | Description | Size |
|------|-------------|------|
| TT_v1.npz | Temperature auto-spectrum emulator (numpy) | ~22 MB |
| TT_v1.pkl | Temperature auto-spectrum emulator (pickle) | ~22 MB |
| TE_v1.npz | Temperature-E-mode cross-spectrum emulator | ~22 MB |
| TE_v1.pkl | Temperature-E-mode cross-spectrum emulator | ~22 MB |
| EE_v1.npz | E-mode auto-spectrum emulator (numpy) | ~11 MB |
| EE_v1.pkl | E-mode auto-spectrum emulator (pickle) | ~11 MB |

Total: ~111 MB

## How to Obtain

These files are excluded from the repository (see `.gitignore`) due to size.
Only the `.npz` variants are required at runtime; the `.pkl` variants exist
for legacy TensorFlow <2.20 compatibility and are not needed.

**Option A: Automated fetch (preferred, once DOI is minted)**
```bash
bash scripts/download_cosmopower_models.sh
```
Downloads the `.npz` emulators into this directory with SHA-256 verification.
Edit the script once the Zenodo record ID is known.

**Option B: Manual Zenodo download**
Download the emulators from the Zenodo record linked in the top-level README
and extract into this directory.

**Option C: Contact the author**
Email the corresponding author listed in `CITATION.cff`.

## Technical Details

- Models were trained with CosmoPower (Spurio Mancini et al. 2022)
- Training data: 100,000 CLASS spectra spanning the Planck prior volume
- The `.npz` format is used at runtime (see `cosmopower_setup.py` for the
  TF 2.20+ compatibility fix that loads from `.npz` instead of `.pkl`)
- Accuracy: sub-0.1% residuals vs CLASS across the Planck ell range (30-2508)

## Verification

After placing files here, phase 2 scripts will automatically load them via
`cosmopower_setup.py`. A quick sanity check:

```python
from cosmopower_setup import load_models
models = load_models()  # Should load without error
```
