# MTDF Validation Output

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


This directory contains the curated artefacts from the MTDF independent GPU validation (Phases 1-6, MTDF_07) and the multi-probe local universe validation (Phases 7-8, MTDF_08). Each phase folder includes the canonical summary data, key plots, and a README linking to the relevant paper section.

**Suggested reading order:** Start with `Validation_Dashboard_V74.html` for the 15-pillar overview, then read `MTDF_07` for Phases 1-6, then `MTDF_08` for Phases 7-8 (CF4 peculiar velocities and 2MTF Tully-Fisher). Click through to the phase folders to verify individual claims against source artefacts.

## Directory Index (19/19 complete)

| Folder / File | Phase | Type | What it tests | Key result | MTDF_07 Section |
|---------------|-------|------|---------------|------------|-----------------|
| `Validation_Dashboard_V74.html` | -- | -- | 15-pillar strict dashboard | chi-squared/nu = 1.17 (DOF=1745) | MTDF_01 |
| `Diagnostics.csv` | -- | -- | Scalar pillar breakdown | 15 pillars, all within 1-sigma | MTDF_01 |
| `phase1/` | Phase 1 | Discriminator | Independent chi-squared reproduction | delta = 0.000049 | Section 2 |
| `phase2/` | Phase 2 | Discriminator | Planck CMB consistency (plik-lite) | Delta-chi-squared < 1; k_f = 1 in 95% CI | Section 3 |
| `phase3/` | Phase 3 | Discriminator | SN x void environment signal | REVOLVER: p = 0.039, gamma_env > 0 | Section 4 |
| `phase3b/` | Phase 3b | Upper limit | NGC/SGC asymmetry diagnostics | Not significant; sensitivity-limited | Section 5b |
| `phase4/` | Phase 4 | Forecast | Sensitivity forecasts | 5-sigma at ~3,400 SNe (LSST/Rubin) | Section 5 |
| `phase5/` | Phase 5 | Discriminator | Full Planck plik + class_mtdf | Delta-chi-squared = +0.63; sigma8 shift 2.4% | Section 6 |
| `prediction_pack/` | -- | -- | class_mtdf P(k,z) + growth predictions | Foundation for MTDF_07 | -- |
| `phase6/testA_redshift_transition/` | Phase 6A | Discriminator | Redshift transition structure | Transition confirmed, p < 0.005 | Section 7.1 |
| `phase6/testB_wl_environment/` | Phase 6B | Upper limit | Weak lensing x void environment | Null: 0.8sigma, 95% UL locked | Section 7.1 |
| `phase6/testB2_trough_ridge/` | Phase 6B2 | Upper limit | Trough & ridge lensing (KiDS) | Null: 0.5sigma, 95% UL locked | Section 7.1 |
| `phase6/testC_derived_consistency/` | Phase 6C | Coherence | Derived parameter consistency | Pass: no params exceed 2.4-sigma shift | Section 7.1 |
| `phase6/testC2_s8_coherence/` | Phase 6C2 | Coherence | S8 cross-probe coherence | Combined WL tension 3.5sigma -> 1.9sigma | Section 7.1 |
| `phase6/testC3_bao_residuals/` | Phase 6C3 | Forecast | BAO residual structure | Delta-W RMS = 0.27%, undetectable | Section 7.1 |
| `phase6/testC4_fsigma8_environment/` | Phase 6C4 | Forecast | fsigma8 x environment | Max 0.22sigma; needs ~10x improvement | Section 7.1 |
| `phase6/testC5_cluster_mass/` | Phase 6C5 | Resolved | Cluster M_dyn/M_lens | eta = 1 confirmed (C5b); Case B: M_dyn/M_lens = 1 | Section 7.1 |
| `phase6/testC5b_gravitational_slip/` | Phase 6C5b | Resolved | Gravitational slip eta = Phi/Psi | eta = 1.000000 at all cluster scales | Section 7.1 |
| `phase6/testD_cmb_lensing/` | Phase 6D | Upper limit | CMB lensing x DESI voids | LOCKED null: 0.7sigma, compensated | Section 7.1 |
| `phase6/testE_boss_benchmark/` | Phase 6E | Discriminator | CMB lensing x BOSS DR12 voids | CTH 1.5sigma, MF 1.9sigma; pipeline validated | Section 7.1 |
| `phase7_cf4_vpec/` | Phase 7 | Discriminator | CF4 peculiar velocity environment coupling | 19.6sigma baseline; 9.0sigma N-body tension; 5-9sigma after 2M++ | MTDF_08 5.3, 7 |
| `phase8_2mtf_tf/` | Phase 8 | Discriminator | 2MTF Tully-Fisher environment coupling | 7.9sigma K-band; achromatic J/H/K; 4.2sigma N-body tension | MTDF_08 5.4, 7 |

### Type taxonomy

| Type | Meaning |
|------|---------|
| **Discriminator** | Active test of MTDF predictions; a failure here would challenge the framework |
| **Coherence** | Cross-probe consistency check; MTDF should not break what LCDM gets right |
| **Upper limit** | Systematics-clean null with explicit bound; expected given current sensitivity |
| **Forecast** | Prediction computed; quantifies what future surveys need to discriminate |

## Canonical numbers

All quoted chi-squared, Delta-chi-squared, AIC, BIC, and posterior values are taken from the JSON files in these folders. Delta-chi-squared is defined as chi-squared_MTDF minus chi-squared_LCDM throughout.

## Large artefacts not included

The following are available on request:
- Phase 2: MCMC chain files (~54 MB as .npz)
- Phase 3b: Full plot sets for all catalogues and additional timestamped runs
- Phase 5: Raw Cobaya chains (~40 MB), covariance matrices, monitor logs

## Related documentation

- `../../papers/MTDF_07_Independent_GPU_Validation.html` -- GPU validation paper (Phases 1-6)
- `../../papers/MTDF_08_Multi_Probe_Evidence_Low_Redshift_Transition.html` -- multi-probe local universe paper (Phases 7-8)
