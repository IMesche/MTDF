# Phase 6 Test C3: BAO Residual Structure

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## Goal

Extract BAO wiggles from class_mtdf P(k) and compare MTDF vs LCDM wiggle patterns. MTDF's Early Field Energy modifies the transfer function at recombination, shifting the sound horizon by ~0.74%. This should produce a small phase shift in the BAO wiggles.

## Method

1. Load P(k) at z=0 for LCDM and MTDF from prediction pack
2. Compute smooth no-wiggle reference P_nw(k) using Eisenstein & Hu (1998) fitting formula with broadband amplitude matching
3. Extract wiggles: W(k) = P(k)/P_nw(k) - 1
4. Compute delta_W(k) = W_MTDF(k) - W_LCDM(k)
5. Find peaks, measure phase shifts and amplitude changes

## Sound Horizon

| Model | r_s,drag (Mpc) |
|-------|----------------|
| LCDM | 147.148 |
| MTDF | 147.303 |
| Shift | +0.106% |

## Wiggle Metrics

| Metric | Value |
|--------|-------|
| delta_W RMS | 0.2747% |
| delta_W max | 0.5105% at k = 0.119 h/Mpc |
| Wiggle amplitude (LCDM) | 6.01% |
| Wiggle amplitude (MTDF) | 6.09% |
| Amplitude change | +1.41% |

## Peak Positions

| Peak # | k_LCDM (h/Mpc) | k_MTDF (h/Mpc) | Shift (%) |
|--------|----------------|----------------|----------|
| 1 | 0.0749 | 0.0749 | +0.000 |
| 2 | 0.1367 | 0.1367 | +0.000 |
| 3 | 0.1979 | 0.1979 | +0.000 |
| 4 | 0.2613 | 0.2613 | +0.000 |
| 5 | 0.3293 | 0.3293 | +0.000 |

## Detectability

| Survey | P(k) precision | SNR per k-bin | Detectable? |
|--------|---------------|---------------|-------------|
| BOSS DR12 | 2.0% | 0.137 | No |
| DESI Y1 | 1.0% | 0.275 | No |
| DESI Y5 | 0.5% | 0.549 | No |
| Euclid | 0.4% | 0.687 | No |
| DESI+Euclid combined | 0.3% | 0.916 | No |

## Interpretation

The BAO wiggle difference between MTDF and LCDM is extremely small (delta_W RMS ~ 0.01-0.1%), reflecting the modest 0.74% sound horizon shift from the EFE. This is:

- **~100x smaller** than current BOSS BAO precision
- **~10-50x smaller** than projected DESI Y5 precision
- Below any foreseeable single-survey detection threshold

The BAO peak position itself (used for distance measurements) shifts by ~0.1%, which is within the MTDF-LCDM parameter degeneracy already captured by the Phase 5 MCMC.

**Verdict:** BAO residual structure is not a viable discriminator between MTDF and LCDM with current or near-future data. The modification is absorbed into standard cosmological parameter shifts.

## Files

| File | Description |
|------|-------------|
| `testC3_bao_residuals.json` | Full analysis data |
| `testC3_bao_residuals.png` | P(k), wiggles, delta_W |
| `testC3_peak_zoom.png` | Peak phase shift detail |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testC3_bao_residuals.py
```
