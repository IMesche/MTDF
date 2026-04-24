# MTDF Test Matrix (Complete)

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## A. Tests Already Completed

These are **done**, documented, and defensible.

### A1. CMB Compatibility (Planck)

* **Type:** Falsifier

* **Data:** Planck 2018 (plik-lite and full plik)

* **Status:** ✔ Passed

* **Outcome:**\
  MTDF reproduces ΛCDM-level CMB fits (Δχ² = +0.63, full Planck plik TTTEEE + lensing).

* **Meaning:**\
  MTDF is _not excluded_ by this dataset under the stated assumptions (the most stringent early-universe constraint).

***

### A2. σ₈ Suppression (Planck-derived)

* **Type:** Discriminator

* **Data:** Planck MCMC

* **Status:** ✔ Detected

* **Outcome:**\
  σ₈\_MTDF ≈ 0.790 vs σ₈\_LCDM ≈ 0.810\
  68% intervals do not overlap.

* **Meaning:**\
  Structural suppression of late-time growth.

***

### A3. H₀ Directional Shift

* **Type:** Consistency / Directional

* **Data:** Planck MCMC

* **Status:** ✔ Observed

* **Outcome:**\
  ΔH₀ ≈ +0.45 km/s/Mpc

* **Meaning:**\
  Consistent with late-time modification, not early-time tuning.

***

### A4. SN Ia × Void Environment (Phase 3)

* **Type:** Discriminator

* **Data:** Pantheon+ SN, void catalogs

* **Status:** ✔ Detected (~2σ)

* **Outcome:**\
  γ\_env ≈ +0.005 at z < 0.04

* **Meaning:**\
  Environment-dependent luminosity residuals.

***

### A5. Redshift Localization of Effects

* **Type:** Structural

* **Data:** Phase 3

* **Status:** ✔ Confirmed

* **Outcome:**\
  Signal confined to z < 0.04

* **Meaning:**\
  Finite coherence scale, not arbitrary scatter.

***

### A6. NGC/SGC Asymmetry Diagnostics (Phase 3b)

* **Type:** Systematic control

* **Data:** Detectability tables, null injections, mocks

* **Status:** ✔ Resolved

* **Outcome:**\
  SGC null is sensitivity-limited, not physical.

* **Meaning:**\
  No hemispherical new physics required.

***

### A7. Parameter Non-Tuning (Global)

* **Type:** Structural integrity

* **Data:** Full framework

* **Status:** ✔ Verified

* **Outcome:**\
  No parameters tuned to validation data.

* **Meaning:**\
  No hidden degrees of freedom.

***

### A8. Full MCMC Convergence (Phase 5)

* **Type:** Statistical hygiene

* **Data:** Full Planck plik MCMC via Cobaya + class_mtdf

* **Status:** ✔ Complete

* **Outcome:**\
  R-1 < 0.02 (both chains, 26k+ accepted samples).\
  Δχ² = +0.63 (indistinguishable from ΛCDM), ΔAIC = +2.63.\
  σ₈ = 0.790 (MTDF) vs 0.810 (ΛCDM) — eases S₈ tension.\
  k_f 95% CI includes both 0 and 1.

* **Meaning:**\
  Posteriors locked for publication. MTDF passes the most stringent CMB constraint.

***

### A9. Weak Lensing × Environment (Phase 6B)

* **Type:** Discriminator / upper limit

* **Data:** KiDS-1000 shear + DESIVAST BGS voids (Phase 6B) and KiDS density troughs/ridges (Phase 6B2)

* **Status:** ✔ Complete — systematics-clean null

* **Outcome:**\
  Test B (void lensing): Δγ_t = -3.65e-5 ± 4.68e-5 (0.8σ), 95% CI: [-1.20e-4, +5.74e-5]\
  Test B2 (ridge−trough): Δγ_t_split = +2.56e-5 ± 5.46e-5 (0.5σ), 95% CI: [-9.26e-5, +1.28e-4]\
  γ_x systematics gate passed (p > 0.05) for all configurations.

* **Meaning:**\
  No environment-dependent shear detected at current KiDS-1000 sensitivity.\
  Upper limit constrains any anomalous void lensing signal to |Δγ_t| < 1.2×10⁻⁴ (95%).

***

### A10. CMB Lensing × Voids (Phase 6D)

* **Type:** Discriminator / upper limit

* **Data:** Planck PR4 MV κ map + DESIVAST BGS voids (z < 0.24)

* **Status:** ✔ Complete and LOCKED — systematics-clean null

* **Outcome:**\
  Per-void compensated: κ_comp = -7.47e-4 ± 1.03e-3 (0.7σ, 557 voids)\
  RA-scramble null p = 0.425 (compensated), random-sky null p = 0.080\
  Robust to outer annulus choice (all < 1.1σ), low-l removal (l≥20, l≥30), and void finder

* **Diagnosed failure mode (v1):**\
  Hard mask quality cut (70% unmasked in 5 Rv) rejected 73% of voids.\
  Large-scale CMB modes (l < 30) created a positive κ pedestal at all radii.

* **Remedy (v2):**\
  Per-void compensated filter subtracts each void's local background.\
  No hard mask cut; per-pixel mask-weighted stacking retains all voids.

* **Meaning:**\
  No void-specific CMB lensing signal at z < 0.24 — expected given low redshift\
  (CMB lensing kernel peaks at z ~ 1-2) and small void count (557 usable).\
  Pipeline validated; next step is higher-z catalogue (BOSS/DESI LRG).

***

### A11. Redshift Transition (Phase 6A)

* **Type:** Structural validation

* **Data:** Multi-probe (SN, void, environmental) split across z ≈ 0.04

* **Status:** ✔ Passed

* **Outcome:**\
  Transition confirmed (p < 0.005).

* **Meaning:**\
  MTDF's predicted late-time onset is structurally coherent across probes.

***

### A12. Derived Parameter Consistency (Phase 6C)

* **Type:** Internal consistency

* **Data:** Phase 5 MCMC posteriors, derived parameters

* **Status:** ✔ Passed

* **Outcome:**\
  No derived parameters exceed 3σ shift between MTDF and ΛCDM.

* **Meaning:**\
  MTDF does not introduce internal contradictions.

***

### A13. Higher-z CMB Lensing × Voids (Phase 6E)

* **Type:** Pipeline benchmark

* **Data:** Planck PR4 κ + Mao+2017 BOSS DR12 ZOBOV voids (1,228, z = 0.2–0.7)

* **Status:** ✔ Complete — five statistics, clean separation of pedestal vs void signal

* **Outcome:**\
  Per-void compensated: κ_comp = -1.6e-4 ± 7.1e-4 (0.2σ) — systematics-clean\
  AP disc: +1.2e-3 (4.7σ) — **contamination monitor** (p_RA=0.95, collapses under l>=20)\
  CTH (disc-ann): +4.8e-4 (1.5σ) — **Cai-class detection** (p_RA=0.18, p_rand=0.03)\
  Matched filter: +2.6e-4 (1.9σ) — **MF detection** (p_RA=0.19, p_rand=0.02)\
  Low-l removal: AP disc 4.7→0.3σ; CTH 1.5→2.0σ; MF 1.9→2.2σ\
  Cai+2017 reference: 3.2σ with theory-template matched filter on same catalogue.

* **Meaning:**\
  Pipeline validated with clean null/signal separation. AP disc detects the\
  large-scale pedestal (contamination). CTH and MF detect void-specific structure\
  that survives RA-scramble and pedestal removal. The gap to 3.2σ reflects\
  template shape (step vs theory-derived), not pipeline issues.

***

### A14. S8 Cross-Probe Coherence (Phase 6 Test C2)

* **Type:** Cross-probe coherence

* **Data:** Phase 5 MCMC S8 vs published WL surveys (KiDS-1000, DES Y3, HSC Y3)

* **Status:** ✔ Complete

* **Outcome:**\
  MTDF S8 = 0.802 vs LCDM S8 = 0.830.\
  Tension with KiDS-1000: 2.67σ (LCDM) → 1.63σ (MTDF), 39% reduction.\
  Tension with DES Y3: 2.63σ (LCDM) → 1.28σ (MTDF), 51% reduction.\
  Tension with WL combined: 3.47σ (LCDM) → 1.89σ (MTDF), 46% reduction.

* **Caveats:**\
  Published WL S8 values assume LCDM growth/lensing kernel.\
  Proper comparison requires WL likelihood within MTDF framework.\
  First-order consistency check using Gaussian error propagation.

* **Meaning:**\
  MTDF's σ₈ suppression is directionally correct: it reduces the S8 tension\
  with all three WL surveys by 39–51%. The combined tension drops from\
  3.5σ to 1.9σ. A full likelihood analysis (deferred to MTDF_07) would\
  properly account for parameter degeneracies and the modified lensing kernel.

***

### A15. fσ₈ × Environment (Phase 6 Test C4)

* **Type:** Growth-geometry link / sensitivity assessment

* **Data:** RSD compilation (8 surveys) + void-specific fσ₈ (Hamaus+2020)

* **Status:** ✔ Complete — sensitivity-limited null

* **Outcome:**\
  χ² (RSD+voids): LCDM = 7.61, MTDF = 8.10 (Δχ² = +0.49, 9 DOF).\
  Max discrimination: 0.22σ — model difference (0.007) is 5–10× smaller than data errors.\
  Current RSD data cannot distinguish MTDF from LCDM in fσ₈.

* **Future requirements:**\
  2σ discrimination needs σ(fσ₈) ~ 0.003–0.004 per z-bin\
  (current errors: 0.03–0.10, improvement factor ~10–15×).\
  Achievable with DESI/Euclid-era RSD or environment-resolved estimators.

* **Meaning:**\
  The homogeneous μ(a) modification is too small for current data.\
  This is expected: the 1.5% signal is comparable to the LCDM–MTDF σ₈\
  difference that produces the S8 result (A14). Discrimination requires\
  next-generation surveys or combined multi-probe analysis.

***

### A16. BAO Residual Structure (Phase 6 Test C3)

* **Type:** Prediction-level assessment

* **Data:** class_mtdf P(k) at z=0, Eisenstein & Hu no-wiggle reference

* **Status:** ✔ Complete — prediction computed, not detectable

* **Outcome:**\
  Sound horizon shift: +0.106% (147.148 → 147.303 Mpc).\
  BAO wiggle difference: ΔW RMS = 0.27%, max = 0.51% at k = 0.12 h/Mpc.\
  No detectable peak phase shift (Δk/k = 0.0% at grid resolution).\
  Best projected SNR: 0.92 (DESI+Euclid combined) — below 1σ.

* **Meaning:**\
  The 0.1% sound horizon shift is absorbed into standard cosmological parameter\
  degeneracies. BAO residual structure is not a viable MTDF–ΛCDM discriminator\
  with any foreseeable survey. The EFE modification is simply too small at the\
  BAO scale.

***

### A17. Cluster Dynamics vs Lensing Mass (Phase 6 Test C5/C5b)

* **Type:** Prediction-level assessment → **Resolved**

* **Data:** Analytical μ(a) prediction + numerical η(k,z) extraction from class_mtdf

* **Status:** ✔ Complete — **Case B confirmed numerically (η = 1)**

* **Outcome:**\
  C5 computed M_dyn/M_lens predictions under two coupling cases.\
  C5b resolved the key unknown by extracting η(k,z) = Φ/Ψ directly from class_mtdf transfer functions:\
  η = 1.000000 at all cluster-relevant scales (k = 0.005–2 h/Mpc, z = 0–0.8).\
  Max |Δη| between MTDF and ΛCDM: 0.000000.\
  **Case B applies: M_dyn/M_lens = 1.000. Clusters are not a discriminator through the mass ratio channel.**

* **Why η = 1:**\
  class_mtdf modifies only the Poisson equation source (δρ → μ·δρ) and the CDM Euler equation.\
  The Φ−Ψ anisotropy equation is unchanged from GR. Both potentials are enhanced equally → Σ = μ.

* **Meaning:**\
  The cluster mass ratio channel is closed as a discriminator for MTDF.\
  Clusters can still test MTDF through growth-dependent observables (e.g. cluster counts),\
  because μ > 1 modifies the growth rate. Only the M_dyn/M_lens ratio is null.

***

## B. Tests Currently In Progress / Pending Finalization

_All prior B-section items have been completed and moved to Section A._

***

## C. Tests Not Yet Performed

_All identified tests have been completed. Section retained for future additions._

***

## D. Explicitly Out-of-Scope / Not Required

Important to state clearly.

* Particle dark matter detection

* Microphysical DM candidates

* Early-universe inflationary reconstruction

* Exotic dark sector interactions

These are **not required** for MTDF validation.

***

## E. Gravity Bridge Tests (Exploratory)

These tests bridge the validated cosmological framework toward the gravity-mechanism programme documented in MTDF_Gravity.md. They are **not** part of the 17-test submission matrix.

***

### E1. Scale-Dependent Transition Signature (F1)

* **Type:** Exploratory discriminator

* **Data:** Phase 3 SN × void data, REVOLVER voids split by radius

* **Status:** ✔ Complete

* **Outcome:**\
  Void bins: small [10,16) Mpc/h (1,012 voids), medium [16,22) (591), large [22,50) (389).\
  Peak z_cut: small = 0.030 (2.89σ), medium = 0.045 (3.82σ), large = 0.040 (4.47σ).\
  Bootstrap 68% CI: 2/3 bins overlap baseline z_cut = 0.030.\
  Reweighted peaks: shifted for medium/large bins (z-distribution confound not fully controlled).

* **Pass criteria:**\
  Primary: PASS — 2/3 bins overlap baseline.\
  Tertiary: FAIL — reweighting shifts peaks for medium/large bins.

* **Meaning:**\
  The SN × void transition persists under void-size splits but peaks do move.\
  Scale independence is not proven — only that the signal is not a single-scale artefact.

* **Artefacts:** `output/testF1_scale_dependence/`

***

### E2. Rotation Curve + Lensing Consistency (F2)

* **Type:** Pre-registered hard discriminator (forecast computed, execution deferred to MTDF_07)

* **Data:** 175 SPARC galaxies, MTDF rotation curve formula, NFW comparison via Moster+2013 SHMR

* **Status:** ✔ Forecast complete — predictions locked

* **Outcome:**\
  At R > 100 kpc, MTDF's baryonic enhancement (factor ~2.3) produces only 0.8–6.0% of\
  the LCDM ESD (NFW halo dominates by factor 15–120×).\
  The "F2 cliff" is confirmed real (not a calculation artefact) via independent stress-halo\
  analysis: M_stress(<100 kpc) ~ 6×10⁷ M_sun vs NFW ~10¹¹ M_sun (factor ~2000 shortfall).

* **Pre-registered success criteria (for MTDF_07):**\
  MTDF passes if ΔΣ(R) within 2σ of observed GGL signal across all R bins.\
  MTDF fails if ΔΣ(R > 100 kpc) requires additional NFW-scale mass.\
  Inconclusive if S/N < 3 in discriminating R bins.

* **Meaning:**\
  This is the hardest open question for the gravity programme.\
  The framework's coherence length (β = 22.7 Mpc) prevents stress concentration\
  at galactic scales. Four testable resolutions identified: non-linear amplification,\
  scale-dependent coherence, data reinterpretation, or genuine failure.\
  See MTDF_Gravity.md Section 4.3 for full diagnosis.

* **Artefacts:** `output/testF2_rotation_lensing/`

***

## One-Line Summary

* **Falsifiers:** All passed (A1, A8, A12)

* **Late-time signatures:** Detected and coherent (A2, A3, A4, A5, A11)

* **Systematics:** Neutralized (A6, A7)

* **Sensitivity-limited nulls:** Clean upper limits locked (A9, A10, A13, A15)

* **Cross-probe coherence:** S8 tension reduced 3.5σ → 1.9σ (A14)

* **Prediction-level / Resolved:** BAO residuals undetectable (A16); cluster mass ratio null — η = 1 confirmed, Case B (A17)

* **Gravity bridge (exploratory):** Transition persists across void sizes (E1); GGL prediction locked as pre-registered hard discriminator (E2)

17 submission tests completed, 0 failures, 0 remaining.\
2 exploratory gravity-bridge tests completed (documented in MTDF_Gravity.md).\
The framework remains consistent with every test applied under the stated assumptions. The S8 tension reduction (A14)\
is the strongest positive discriminator in this set. The GGL prediction (E2) is the\
strongest open challenge for the gravity programme.

