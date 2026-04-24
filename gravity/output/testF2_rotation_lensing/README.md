# Phase 6 Test F2: Rotation Curve + Lensing Consistency

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


**Status:** Exploratory — pre-registered discriminator (forecast).
Not part of the 17/17 validation matrix. Full execution deferred to
MTDF_07. Documented in MTDF_Gravity.md for the gravity-mechanism
research programme.

## Goal

Pre-register a falsifiable test: does the MTDF mass profile from rotation
curves predict the correct galaxy-galaxy lensing (GGL) signal?  Since
η = 1 (confirmed C5b), lensing and dynamics probe the same effective
potential.

## Physics

MTDF predicts:

    v_c²(r) = G M_bar / r × [1 + α/(1 + r/β)]

where α = 1.30, β = 22,685 kpc.  At GGL-relevant scales (R = 10–2000 kpc),
R << β, so the enhancement is approximately constant:

    [1 + α/(1 + R/β)] ≈ [1 + α] = 2.30

MTDF predicts the ESD should be **2.3× the baryonic-only ESD** at all
radii.  ΛCDM explains the same signal with an NFW dark matter halo.

## Assumptions (Pre-Registered)

**Assumption A (primary prediction):**
ΔΣ_MTDF(R) = ΔΣ_baryon(R) × [1 + α/(1 + R/β)]

The MTDF force-law enhancement applies equally to lensing and dynamics
(since η = 1 at perturbation level, C5b).  This is a direct extension
of the rotation curve formula to the lensing kernel.

**Assumption B (alternative, for completeness):**
ΔΣ_MTDF(R) = ΔΣ_baryon(R)

Force-law modification is purely kinematic (affects orbits, not photons).
This would be inconsistent with η = 1 and is NOT the primary prediction.

**The distinction between A and B is a testable prediction.**

## Method

1. Load 175 SPARC galaxies (Lelli+2016)
2. Compute M_bar(<r) = v_bar²(r) × r / G for each galaxy
3. Bin by total baryonic mass (proxy for stellar mass)
4. Average M_bar(<R) per bin on common radial grid
5. Compute ESD predictions:
   - ΔΣ_baryon(R) = M_bar / (πR²) — point-mass approximation (valid R >> R_d)
   - ΔΣ_MTDF(R) = ΔΣ_baryon(R) × [1 + α/(1+R/β)] — Assumption A
   - ΔΣ_ΛCDM(R) = ΔΣ_baryon(R) + ΔΣ_NFW(R) — using Moster+2013 SHMR + Duffy+2008 c(M)

## SPARC Mass Bins

| Bin | log M range | N_gal | <M_bar> | M_halo (SHMR) | c (Duffy) | MTDF/ΛCDM (R>100 kpc) |
|-----|-------------|-------|---------|---------------|-----------|----------------------|
| dwarf | [7, 9) | 30 | 4.40e+08 | 6.94e+10 | 7.6 | 0.008 |
| intermediate | [9, 10) | 61 | 4.28e+09 | 1.96e+11 | 6.9 | 0.031 |
| L-star | [10, 10) | 27 | 1.79e+10 | 4.81e+11 | 6.4 | 0.060 |
| massive | [10, 12) | 57 | 1.83e+11 | 5.00e+13 | 4.4 | 0.027 |

## Published Data Overlay Caveat

Any published GGL ESD profiles (e.g., Brouwer+2021 KiDS×GAMA) overlaid
on the comparison plot are **illustrative context only**, not a direct
comparison.  A proper comparison requires exactly matching the lens
selection, redshift weights, and radial bin definitions used in the
published analysis.  This is deferred to MTDF_07.

## Success Criteria (Pre-Registered)

- **MTDF passes if:** ΔΣ(R) from enhanced baryonic mass is within 2σ of
  observed GGL signal across all R bins
- **MTDF fails if:** ΔΣ(R > 100 kpc) requires additional NFW-scale mass
  beyond what MTDF's force-law enhancement provides
- **Inconclusive if:** S/N < 3 in discriminating R bins (R > 100 kpc)

## Relation to V74 Dashboard

P1 (rotation curve scatter, z-score = 0.888) and P1B (RAR scatter,
z-score = -0.62) in the V74 dashboard confirm MTDF already fits SPARC
rotation curves at sub-1σ.  F2 builds on this by asking: does the same
mass profile that explains rotation curves also predict the correct
lensing signal?  The dashboard pillars are consistency checks; F2 is a
cross-probe prediction.

## Key Physical Insight

At R > 100 kpc, the MTDF and ΛCDM predictions diverge:
- **ΛCDM:** ΔΣ dominated by NFW halo (ΔΣ_NFW >> ΔΣ_baryon)
- **MTDF:** ΔΣ = 2.3 × ΔΣ_baryon (no additional halo mass)

For massive galaxies (M_bar ~ 10^{10.5}), the NFW halo contributes
~10× more ESD than baryons at R = 500 kpc.  MTDF's 2.3× enhancement
cannot match this.  Either MTDF's non-perturbative lensing enhancement
is stronger than the perturbative η = 1 result, or MTDF under-predicts
GGL at large radii for massive galaxies.

This is a genuine, falsifiable tension point — exactly what a pre-
registered test should identify.

## Files

| File | Description |
|------|-------------|
| `testF2_rotation_lensing.json` | Mass profiles, ESD predictions per bin |
| `testF2_esd_prediction.png` | MTDF vs ΛCDM ESD predictions |
| `testF2_sparc_mass_profiles.png` | Binned SPARC mass profiles |
| `README.md` | This file (formal test definition) |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testF2_rotation_lensing.py
```
