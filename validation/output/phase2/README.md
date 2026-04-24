# Phase 2: Planck CMB Consistency (plik-lite)

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Purpose:** Test MTDF compatibility with Planck 2018 CMB power spectrum using a perturbative correction layer on the CosmoPower LCDM emulator.

**Dataset:** Planck plik-lite TTTEEE (613 bins), optionally combined with DESI Y1 BAO and Pantheon+ SNe

**Key result:** Delta-chi-squared < 1; k_f = 1 (full MTDF) within 95% CI in all MCMC runs. LCDM and MTDF are indistinguishable at Planck plik-lite precision.

**Canonical results file:** `mcmc_combined_summary.json` (combined probes)

**Referenced in:** MTDF_07 Section 3

## Files

Three MCMC configurations were run. The **main** result uses all three probes combined.

**Main (combined: plik-lite + BAO + SNe):**
| File | Description |
|------|-------------|
| `mcmc_combined_summary.json` | Posterior summary (main result cited in MTDF_07) |
| `corner_combined.png` | Corner plot (main result) |

**Narrow prior variant:**
| File | Description |
|------|-------------|
| `mcmc_combined_narrow_summary.json` | Posterior summary with narrow k_f prior |
| `corner_combined_narrow.png` | Corner plot with narrow prior |

**plik-lite only (legacy):**
| File | Description |
|------|-------------|
| `mcmc_summary.json` | Posterior summary: plik-lite only |
| `corner_plot.png` | Corner plot: plik-lite only |

**Supporting:**
| File | Description |
|------|-------------|
| `step_a_sanity_report.json` | Emulator validation: CosmoPower vs CAMB accuracy |
| `manifest.json` | SHA256 hashes for all files in this folder |

## Not included (available on request)

Raw MCMC chain files (~54 MB total as .npz).
