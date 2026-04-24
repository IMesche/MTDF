# MTDF Gravity  - Output Artefacts

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


Exploratory bridge tests connecting the validated MTDF framework to the gravity-mechanism research programme. These tests are **not** part of the 17-test submission matrix (MTDF_07).

## Directory Index

| Folder | Test | Type | What it tests | Key result |
|--------|------|------|---------------|------------|
| `testF1_scale_dependence/` | F1 | Exploratory discriminator | Does the z < 0.04 transition survive across different void sizes? | 2/3 bins overlap baseline z_cut = 0.030; scale independence not proven |
| `testF2_rotation_lensing/` | F2 | Pre-registered discriminator (forecast) | Does the rotation curve mass profile predict the correct lensing signal? | MTDF/LCDM ratio 0.008-0.060 at R > 100 kpc; genuine tension point |
| `step1_ggl_comparison/` | Step 1 | Data comparison | MTDF vs Brouwer+2021 KiDS×GAMA published ESD profiles | Cliff confirmed: 41-94× shortfall at R > 100 kpc against real data |
| `step2_algebraic_target/` | Step 2 | Algebraic analysis | What field contribution is needed to close the gap? | Energy budget problem: max stress mass 8-27× too small; shape right (isothermal) |
| `step3_nonlinear_source/` | Step 3 | Theoretical analysis | Can non-linear MTDF field equations produce the needed source term? | Blocked: c² gap between [E1'] and rotation curve; no Lagrangian; self-sourcing diverges |
| `step4_shear_comparison/` | Step 4 | Direct comparison | MTDF shear (from metric) vs observed shear  - no NFW, no mass budget | Shortfall 39-91× at R > 100 kpc; metric correction < 0.01%; result is final |
| `step5_density_inversion/` | Step 5 | Density inversion | Invert ΔΣ_obs into required ρ_stress(r); compare to MTDF prediction | Required/predicted = 2,300-6,000×; shapes match (r⁻²); amplitude off by 10³·⁴ |
| `step6_nonlinear_equation/` | Step 6 | Non-linear equation | Minimal self-consistent field eq with β_local + saturation | **Structural failure**: MTDF profile transitions r⁻² → r⁻⁴ at r ~ β_local; data requires r⁻² to 2000 kpc; shape and amplitude cannot be simultaneously matched |
| `step7_constitutive_ode/` | Step 7 | Constitutive ODE | GPT's prescription: quartic U(S), derive ODE, solve for ρ_stress | **Decisive**: two approaches tested; energy-based gives r⁻⁴ (wrong slope); [E1']-based gives r⁻² (right slope!) but amplitude gap 10³³× (c⁴ normalization); no S_crit fixes it |
| `step8_compression/` | Step 8 | Compression hypothesis | S₀ from cosmological ρ_crit; compression ansatz S(r) = S₀(1 + A×r_g/r) | **Promising**: S₀ = 1.084; strong compression at 100 kpc (x = 18-31); transition scale ≈ β/10; BTFR scaling ρ_s r_s² ∝ M^{0.5} (10% scatter) |
| `step9_4pi_alpha/` | Step 9 | 4π/α identification | Replace fitted "10" with 4π/α = 9.666; derive L = αβ/(4π); test universal profile | **Breakthrough**: universal ρ_s r_s² = 4.84×10⁸ within 2× of all bins; f ∝ M^{0.258} matches BTFR (0.250); zero new parameters |
| `step10_vref_closure/` | Step 10 | v_ref closure | Derive v_ref from isothermal identity; compute f(M) = v_flat/v_ref from SPARC/BTFR; Laplace + Gauss law backing | **Closure**: v_ref = 161.8 km/s (derived); f matches at RMS 5.7% (McGaugh BTFR + gas); ∇²S = 0 → S = S₀ + C/r is a theorem |
| `step11_jbar/` | Step 11 | J_bar identification | Test 3 source candidates (ρc², ρ|Φ|, ρv²); linear vs screened matching | **Identified**: J_bar = ρ_bar c² + quadratic screening (S−S₀)²; C ∝ M^{0.250} matches BTFR; λ ≈ 5.1×10⁻⁴⁸ universal to 5% |
| `step12_delta_sigma/` | Step 12 | Full ΔΣ(R) comparison | Correct elastic energy Δu=(E/2)(δS)² → ρ∝r⁻² → Σ(R) → ΔΣ(R) analytical projection; f from McGaugh BTFR (zero new params) | **Resolved**: χ²/ν = 2.93 (MTDF) vs 8.30 (NFW) vs 31.42 (baryons); cross-term rejected (Δχ²=+44); 41-94× gap closed |
| `step13_solar_system/` | Step 13 | Solar System sanity check | Weak field limit of Route B; PPN parameters; Mercury precession; MW local DM density | **Safe**: all observables safe by > 10^{20}; f_sun = 5.3×10⁻¹⁶; MW rho_stress(8 kpc) = 0.014 matches local stress 0.013 M_sun/pc³ |
| `step14_constitutive_law/` | Step 14 | Constitutive law derivation | BTFR selects n=2 uniquely; energy self-sourcing predicts lambda; Freeman consistency triple | **Derived**: n=2 from BTFR (algebraic); lambda = 2 alpha/beta^2 within 4%; only n=2 matches Brouwer bins |
| `step15_cross_dataset/` | Step 15 | Cross-dataset lensing | Frozen-parameter MTDF prediction vs Mandelbaum+2016 SDSS (independent of KiDS) | **Confirmed**: chi^2/nu = 11.21 (MTDF) vs 44.11 (LCDM); 5/7 bins won; comparable mass range matches KiDS (2.55 vs 2.93) |
| `step16_robustness/` | Step 16 | Robustness suite | MTDF (frozen) vs 6 LCDM-NFW variants (3 SHMR x 2 c(M)); attack-resistant comparison | **Robust**: all 6 variants worse (3.9x-50x); best LCDM = Moster+Duffy baseline; advantage not SHMR-dependent |
| `step17_baryon_completion/` | Step 17 | Baryon completion | Group/cluster baryons (hot gas, satellites, ICL) for bins 5-7 | **Explained**: LCDM scaling overshoots (DM halos); data need 14-15% of LCDM gas; no gravity change needed |

## Related documentation

- `../papers/` - Theory papers (HTML, self-contained)
- `tests.md`  - Test matrix including E1 (F1) and E2 (F2) entries
