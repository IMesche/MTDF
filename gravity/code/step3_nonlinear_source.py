#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 3: Non-linear lensing source term from MTDF field equations.

Derives the effective lensing source density ρ_eff(r) from the MTDF core
postulates, including the self-sourcing feedback (stress energy → gravity
→ more stress). Determines whether any mechanism within the current
framework can close the F2 gap (8–27× shortfall from Step 2).

Core postulates used:
  - Modified EFE:   G_μν + α F_μν = (8πG/c⁴) T_μν^baryon
  - Strain:         S̃_μν = β ∇_μ ξ_ν   (dimensionless)
  - Constitutive:   F_μν = E S̃_μν        (linear Hooke's law)
  - Evolution [E1']: □F_μν − γ ∂_t F_μν = (8πG/c⁴) T_μν^matter
  - Energy density:  u = (E/2) S̃²
  - Equation of state: p = -(1/3) u    (w = -1/3)

Key result: GPT asked "Can you write down the non-linear spherical
equation unambiguously from MTDF's core postulates?" This script provides
the answer.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad, solve_bvp
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS (SI)
# ═══════════════════════════════════════════════════════════════

G    = 6.674e-11      # m^3 kg^-1 s^-2
c    = 2.998e8         # m/s
c2   = c**2
c4   = c**4
M_sun = 1.989e30      # kg
kpc  = 3.0857e19      # m per kpc
Mpc  = 3.0857e22      # m per Mpc

# MTDF parameters
ALPHA   = 1.30
BETA_M  = 7.0e23       # m  (22.7 Mpc)
BETA_KPC = BETA_M / kpc
TAU     = 13.0e9 * 3.156e7  # 13 Gyr in seconds
E_MOD   = 9.1e-10      # Pa (elastic modulus)

# Representative galaxy masses (from Step 2, bin 4 = most massive)
BINS = [
    {'name': 'Bin 1', 'log_Ms': 10.0, 'M_bar': 1.15e10 * M_sun},
    {'name': 'Bin 2', 'log_Ms': 10.45, 'M_bar': 3.04e10 * M_sun},
    {'name': 'Bin 3', 'log_Ms': 10.70, 'M_bar': 5.26e10 * M_sun},
    {'name': 'Bin 4', 'log_Ms': 10.90, 'M_bar': 8.18e10 * M_sun},
]


# ═══════════════════════════════════════════════════════════════
# PART 1: Strain field from [E1'] in static limit
# ═══════════════════════════════════════════════════════════════

def strain_from_E1prime(r_m, M_kg):
    """
    Compute the dimensionless strain S̃(r) from the [E1'] evolution
    equation in the static spherical limit.

    [E1']: □F_μν − γ ∂_t F_μν = (8πG/c⁴) T_μν^matter

    Static (∂_t = 0), using □ = ∂_t² − c² ∇²:
       −c² ∇² F_00 = (8πG/c⁴) T_00

    For T_00 = ρc² (energy density) and point mass M:
       ∇² F_00 = −(8πG M / c⁴) δ(r)

    Solution: F_00(r) = 2GM / (c⁴ r)
    Strain:   S̃_00 = F_00 / E = 2GM / (E c⁴ r)
    """
    return 2 * G * M_kg / (E_MOD * c4 * r_m)


def strain_from_rotation_curve(r_m, M_kg):
    """
    Compute the REQUIRED strain for the rotation curve enhancement.

    From the modified Poisson equation (weak-field EFE):
       ∇²Φ + α(8πG/c⁴) E S̃ = (8πG/c⁴) ρ c²

    For the enhancement α/(1+r/β) to match:
       S̃_required(r) = ρ(r) c² / E  [inside galaxy]
       S̃_required(r) ~ 2GM / (E c² r)  [outside, for point mass]

    The c² version (not c⁴) is needed for the EFE to reproduce the
    rotation curve formula. This is the crux of the theoretical gap.
    """
    return 2 * G * M_kg / (E_MOD * c2 * r_m)


# ═══════════════════════════════════════════════════════════════
# PART 2: Stress field energy density
# ═══════════════════════════════════════════════════════════════

def stress_energy_density(S_tilde):
    """
    Elastic energy density (J/m³).
    u = (E/2) S̃²
    """
    return 0.5 * E_MOD * S_tilde**2


def stress_mass_density(S_tilde):
    """
    Stress field mass-equivalent density (kg/m³).
    ρ_stress = u / c²
    """
    return stress_energy_density(S_tilde) / c2


def stress_mass_density_Msun_kpc3(S_tilde):
    """Convert stress mass density to M_sun/kpc³."""
    rho_kg_m3 = stress_mass_density(S_tilde)
    return rho_kg_m3 * kpc**3 / M_sun


# ═══════════════════════════════════════════════════════════════
# PART 3: Self-sourcing feedback loop
# ═══════════════════════════════════════════════════════════════

def self_sourcing_iteration(r_kpc, M_bar_kg, n_iter=20):
    """
    Iterative self-sourcing: stress energy feeds back into gravity.

    Iteration 0: M_total = M_bar (baryonic only)
    Iteration n: M_total = M_bar + M_stress_enclosed(< r)
                 where M_stress depends on the strain from M_total.

    For each iteration, compute:
    1. Strain S̃(r) from M_total at all radii
    2. Energy density u(r) = (E/2) S̃²
    3. Mass density ρ_stress = u/c²
    4. Enclosed stress mass M_stress(<r) = 4π ∫ ρ_stress r² dr
    5. Update M_total = M_bar + M_stress
    6. Repeat until convergence

    Uses the [E1'] equation for strain (correct from field equations).
    """
    r_m = r_kpc * kpc
    r_min = 1.0 * kpc  # inner cutoff (galaxy core)

    # Integration radii (log-spaced from 1 kpc to r_kpc)
    radii_m = np.logspace(np.log10(r_min), np.log10(r_m), 200)

    M_total = M_bar_kg
    history = [{'iter': 0, 'M_total': M_bar_kg, 'M_stress_enclosed': 0.0}]

    for i in range(1, n_iter + 1):
        # Strain from [E1']: S̃ = 2GM/(Ec⁴r)
        S_tilde = 2 * G * M_total / (E_MOD * c4 * radii_m)

        # Mass density ρ_stress = (E/2c²) S̃²
        rho_stress = 0.5 * E_MOD * S_tilde**2 / c2

        # Enclosed mass: 4π ∫ ρ r² dr
        # Since ρ ∝ 1/r² (from S̃ ∝ 1/r), 4πr²ρ = const, integral = const × (r - r_min)
        integrand = 4 * np.pi * radii_m**2 * rho_stress
        M_stress = np.trapz(integrand, radii_m)

        M_total_new = M_bar_kg + M_stress
        history.append({
            'iter': i,
            'M_total': M_total_new,
            'M_stress_enclosed': M_stress,
            'strain_at_r': float(S_tilde[-1]),
            'rho_stress_at_r': float(rho_stress[-1]),
        })

        # Check convergence
        if abs(M_total_new - M_total) / M_total < 1e-12:
            break
        M_total = M_total_new

    return history


def self_sourcing_rotation_curve(r_kpc, M_bar_kg, n_iter=50):
    """
    Same self-sourcing but using the rotation-curve-consistent strain
    (S̃ = 2GM/(Ec²r), c² not c⁴). This is the "what if" scenario
    where the strain is much larger.

    WARNING: With α = 1.30 > 1, the geometric series diverges,
    meaning the linear theory is unstable. Non-linear saturation
    must be invoked.
    """
    r_m = r_kpc * kpc
    r_min = 1.0 * kpc
    radii_m = np.logspace(np.log10(r_min), np.log10(r_m), 200)

    M_total = M_bar_kg
    history = [{'iter': 0, 'M_total': M_bar_kg, 'M_stress_enclosed': 0.0}]

    for i in range(1, n_iter + 1):
        # Rotation-curve strain: S̃ = 2GM/(Ec²r)
        S_tilde = 2 * G * M_total / (E_MOD * c2 * radii_m)
        rho_stress = 0.5 * E_MOD * S_tilde**2 / c2
        integrand = 4 * np.pi * radii_m**2 * rho_stress
        M_stress = np.trapz(integrand, radii_m)

        M_total_new = M_bar_kg + M_stress

        # Cap at 1e5 × M_bar to prevent overflow (non-linear saturation)
        if M_total_new > 1e5 * M_bar_kg:
            history.append({
                'iter': i,
                'M_total': M_total_new,
                'M_stress_enclosed': M_stress,
                'diverged': True,
            })
            break

        history.append({
            'iter': i,
            'M_total': M_total_new,
            'M_stress_enclosed': M_stress,
            'strain_at_r': float(S_tilde[-1]),
        })

        if abs(M_total_new - M_total) / max(M_total, 1) < 1e-10:
            break
        M_total = M_total_new

    return history


# ═══════════════════════════════════════════════════════════════
# PART 4: ESD computation from effective density profile
# ═══════════════════════════════════════════════════════════════

def esd_from_3d_density(R_kpc_arr, rho_func, r_max_kpc=5000):
    """
    Compute ΔΣ(R) from a 3D density profile ρ(r).

    ΔΣ(R) = Σ̄(<R) − Σ(R)

    where Σ(R) = ∫_{-∞}^{∞} ρ(√(R² + z²)) dz
    and Σ̄(<R) = (2/R²) ∫_0^R Σ(R') R' dR'
    """
    def sigma_R(R_kpc):
        """Surface density Σ(R) by integrating along LOS."""
        def integrand(z):
            r = np.sqrt(R_kpc**2 + z**2)
            return rho_func(r)
        result, _ = quad(integrand, 0, r_max_kpc, limit=200)
        return 2 * result  # factor 2 for z > 0 and z < 0

    def sigma_mean(R_kpc, n_int=100):
        """Mean surface density within R."""
        R_arr = np.linspace(0.01, R_kpc, n_int)
        sig_arr = np.array([sigma_R(r) for r in R_arr])
        # Σ̄(<R) = (2/R²) ∫ Σ(R') R' dR'
        integrand = sig_arr * R_arr
        return 2.0 / R_kpc**2 * np.trapz(integrand, R_arr)

    esd = np.zeros(len(R_kpc_arr))
    for i, R in enumerate(R_kpc_arr):
        sig = sigma_R(R)
        sig_mean = sigma_mean(R)
        esd[i] = sig_mean - sig

    return esd  # M_sun/kpc²


# ═══════════════════════════════════════════════════════════════
# PART 5: The Theoretical Gap analysis
# ═══════════════════════════════════════════════════════════════

def theoretical_gap_analysis(M_bar_kg, label):
    """
    Compare the strain predicted by [E1'] vs the strain required
    by the rotation curve formula, and the resulting stress energy.
    """
    r_kpc_arr = np.logspace(0, 3, 100)  # 1 to 1000 kpc
    r_m_arr = r_kpc_arr * kpc

    # Strain from [E1'] (static limit): S̃ = 2GM/(Ec⁴r)
    S_E1 = strain_from_E1prime(r_m_arr, M_bar_kg)

    # Strain required by rotation curve: S̃ = 2GM/(Ec²r)
    S_RC = strain_from_rotation_curve(r_m_arr, M_bar_kg)

    # Discrepancy factor
    gap = S_RC / S_E1  # = c²

    # Stress energy density from each
    rho_E1 = stress_mass_density_Msun_kpc3(S_E1)
    rho_RC = stress_mass_density_Msun_kpc3(S_RC)

    # Enclosed stress mass from each (shell integration, ρ ∝ 1/r²)
    # M_stress(<R) = 4π ∫ ρ(r) r² dr = 4π ρ(r₀) r₀² × (R - r_min)
    r100_idx = np.argmin(np.abs(r_kpc_arr - 100))
    M_stress_E1_100 = (4 * np.pi * rho_E1[r100_idx] * r_kpc_arr[r100_idx]**2
                        * (100 - 1))  # M_sun
    M_stress_RC_100 = (4 * np.pi * rho_RC[r100_idx] * r_kpc_arr[r100_idx]**2
                        * (100 - 1))  # M_sun

    # Required enclosed mass from Step 2 (typical: ~ α × M_bar × 100/β)
    M_bar_msun = M_bar_kg / M_sun
    M_stress_required = ALPHA * M_bar_msun * (100 / BETA_KPC)

    # NFW enclosed mass at 100 kpc (roughly 30× M_bar for massive galaxies)
    M_nfw_100 = 30 * M_bar_msun  # approximate

    return {
        'label': label,
        'r_kpc': r_kpc_arr,
        'S_E1': S_E1,
        'S_RC': S_RC,
        'gap_factor': float(gap[0]),  # should be c² ~ 9e16
        'rho_E1_100kpc': float(rho_E1[r100_idx]),
        'rho_RC_100kpc': float(rho_RC[r100_idx]),
        'M_stress_E1_100kpc': float(M_stress_E1_100),
        'M_stress_RC_100kpc': float(M_stress_RC_100),
        'M_stress_required_100kpc': float(M_stress_required),
        'M_nfw_100kpc': float(M_nfw_100),
        'M_bar_Msun': M_bar_msun,
    }


# ═══════════════════════════════════════════════════════════════
# PART 6: Non-linear constitutive relation analysis
# ═══════════════════════════════════════════════════════════════

def nonlinear_requirement(gap_result):
    """
    What non-linear constitutive relation F(S̃) would be needed?

    Linear: F = E × S̃ (Hooke's law)
    Required: F_NL(S̃) such that the effective enhancement grows
              from α ≈ 1.3 at r < 50 kpc to α_eff ≈ 30-50 at r > 100 kpc.

    Problem: S̃ DECREASES with r (S̃ ∝ 1/r). So any monotonic F(S̃)
    gives LESS enhancement at larger r, not more.

    The only possibility is a NON-MONOTONIC constitutive relation:
    - Saturated at high strain (small r): F → constant → α_eff ~ 1.3
    - Amplified at low strain (large r): F > E×S̃ → α_eff >> 1.3

    This is physically unusual — it means the material is STIFFER at
    low strain than at high strain (strain softening). In continuum
    mechanics, this requires a specific micro-structure (phase transitions,
    auxetic materials, etc.)
    """
    M_bar = gap_result['M_bar_Msun']

    # Required enhancement at different radii
    # At r < 50 kpc: α_eff = 1.30 (validated by SPARC)
    # At r = 100 kpc: α_eff = enhancement_factor from Step 2
    # Shortfall is 41-94x means α_eff needed ~ 40-100

    r_targets = [10, 50, 100, 200, 500]
    results = []
    for r in r_targets:
        # SPARC validated at r < 50 kpc: α_eff = 1.30
        if r <= 50:
            alpha_eff_needed = 1.30
        else:
            # From Step 2 shortfall factors (approximate)
            shortfall = 41 * (r / 100)**0.5  # rough scaling
            alpha_eff_needed = shortfall

        # Strain at this radius (E1' convention)
        r_m = r * kpc
        S = strain_from_E1prime(r_m, M_bar * M_sun)

        # Required F(S) for this α_eff:
        # α_eff × G_00 = α × F_00_NL → F_00_NL = α_eff × G_00 / α
        # But in linear theory: α × F_00_linear = α × E × S̃
        # Ratio: F_NL / F_linear = α_eff / α
        amplification = alpha_eff_needed / ALPHA

        results.append({
            'r_kpc': r,
            'S_tilde': float(S),
            'alpha_eff_needed': alpha_eff_needed,
            'amplification_over_linear': amplification,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    out_dir = Path(__file__).parent.parent / "output" / "step3_nonlinear_source"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {'description': 'Step 3: Non-linear lensing source term analysis'}

    # ── Part A: Theoretical gap analysis ──
    print("=" * 70)
    print("PART A: Theoretical Gap — [E1'] strain vs rotation-curve strain")
    print("=" * 70)

    gap_results = []
    for b in BINS:
        gap = theoretical_gap_analysis(b['M_bar'], b['name'])
        gap_results.append(gap)

        print(f"\n{b['name']} (M_bar = {b['M_bar']/M_sun:.2e} M_sun):")
        print(f"  Strain at 100 kpc (from [E1']): S̃ = {gap['S_E1'][np.argmin(np.abs(gap['r_kpc'] - 100))]:.2e}")
        print(f"  Strain at 100 kpc (for rot.curve): S̃ = {gap['S_RC'][np.argmin(np.abs(gap['r_kpc'] - 100))]:.2e}")
        print(f"  Gap factor: {gap['gap_factor']:.2e} (= c² = {c2:.2e})")
        print(f"  Stress mass enclosed (<100 kpc):")
        print(f"    From [E1']:     {gap['M_stress_E1_100kpc']:.2e} M_sun")
        print(f"    From rot.curve: {gap['M_stress_RC_100kpc']:.2e} M_sun")
        print(f"    Required (Step 2): {gap['M_stress_required_100kpc']:.2e} M_sun")
        print(f"    NFW reference:  {gap['M_nfw_100kpc']:.2e} M_sun")

    results['gap_analysis'] = [{
        'label': g['label'],
        'gap_factor': g['gap_factor'],
        'S_E1_at_100kpc': float(g['S_E1'][np.argmin(np.abs(g['r_kpc'] - 100))]),
        'S_RC_at_100kpc': float(g['S_RC'][np.argmin(np.abs(g['r_kpc'] - 100))]),
        'M_stress_E1_100kpc': g['M_stress_E1_100kpc'],
        'M_stress_RC_100kpc': g['M_stress_RC_100kpc'],
        'M_stress_required': g['M_stress_required_100kpc'],
    } for g in gap_results]

    # ── Part B: Self-sourcing feedback ──
    print("\n" + "=" * 70)
    print("PART B: Self-sourcing feedback loop")
    print("=" * 70)

    ss_results = {}
    for ib, b in enumerate(BINS):
        # Using [E1'] strain (correct from field equations)
        hist_E1 = self_sourcing_iteration(100, b['M_bar'], n_iter=20)
        # Using rotation-curve strain (hypothetical)
        hist_RC = self_sourcing_rotation_curve(100, b['M_bar'], n_iter=50)

        print(f"\n{b['name']}:")
        last_E1 = hist_E1[-1]
        print(f"  [E1'] self-sourcing: M_stress = {last_E1['M_stress_enclosed']/M_sun:.2e} M_sun "
              f"({last_E1['M_stress_enclosed']/b['M_bar']*100:.2e}% of M_bar) "
              f"after {last_E1['iter']} iterations")

        last_RC = hist_RC[-1]
        if last_RC.get('diverged'):
            print(f"  Rot.curve self-sourcing: DIVERGES at iteration {last_RC['iter']} "
                  f"(M > 10⁵ × M_bar)")
        else:
            print(f"  Rot.curve self-sourcing: M_stress = {last_RC['M_stress_enclosed']/M_sun:.2e} M_sun "
                  f"after {last_RC['iter']} iterations")

        ss_results[b['name']] = {
            'E1_final_M_stress': float(last_E1['M_stress_enclosed'] / M_sun),
            'E1_iterations': last_E1['iter'],
            'E1_fraction_of_Mbar': float(last_E1['M_stress_enclosed'] / b['M_bar']),
            'RC_diverged': last_RC.get('diverged', False),
            'RC_final_iter': last_RC['iter'],
        }

    results['self_sourcing'] = ss_results

    # ── Part C: Non-linear constitutive relation requirement ──
    print("\n" + "=" * 70)
    print("PART C: Required non-linear constitutive relation")
    print("=" * 70)

    # Use bin 4 (most massive, smallest gap)
    gap4 = gap_results[3]
    nl_req = nonlinear_requirement(gap4)

    print(f"\n{BINS[3]['name']} (M_bar = {BINS[3]['M_bar']/M_sun:.2e} M_sun):")
    print(f"{'r (kpc)':<12} {'S̃ (E1 prime)':<16} {'α_eff needed':<16} {'Amplification':<16}")
    print("-" * 60)
    for r in nl_req:
        print(f"{r['r_kpc']:<12} {r['S_tilde']:<16.2e} {r['alpha_eff_needed']:<16.1f} {r['amplification_over_linear']:<16.1f}")

    results['nonlinear_requirement'] = nl_req

    # ── Part D: The c² gap — the decisive calculation ──
    print("\n" + "=" * 70)
    print("PART D: The decisive c² gap")
    print("=" * 70)

    print(f"""
The strain computed from the [E1'] evolution equation differs from the
strain required by the rotation curve formula by EXACTLY c² = {c2:.3e}.

  [E1'] gives:         S̃ = 2GM / (E c⁴ r)
  Rotation curve needs: S̃ = 2GM / (E c² r)

This is not a numerical coincidence — it reflects how [E1'] normalizes
the source term relative to the EFE.

There are two interpretations:

  (A) [E1'] is correct, and the rotation curve formula has a different
      origin (not the static solution of the field equation). In this
      case, the stress field energy is negligible (S̃ ~ 10⁻¹⁵) and
      cannot source any lensing signal.

  (B) The rotation curve formula IS the correct static limit, and [E1']
      has a different normalization at galactic scales (e.g., the □
      convention or source term differs by c²). In this case, S̃ ~ 100
      and the elastic energy density is significant — but the linear
      theory is completely invalid (S̃ >> 1).

Either way, the non-linear lensing source term cannot be computed from
the existing MTDF postulates without resolving this ambiguity.
""")

    # ── Part E: What WOULD be needed ──
    print("=" * 70)
    print("PART E: What the non-linear theory would need to achieve")
    print("=" * 70)

    # Compute required ESD enhancement at each radius for all bins
    print(f"\nFrom Step 2 algebraic target (shortfall factors):")
    print(f"  8× amplification for massive galaxies (Bin 4)")
    print(f"  27× amplification for intermediate galaxies (Bin 1)")
    print(f"\nConstraints on any non-linear solution:")
    print(f"  1. SPARC safe: v_c unchanged at r < 50 kpc (α_eff = 1.30)")
    print(f"  2. GGL match: ΔΣ boosted by 8-27× at R > 100 kpc")
    print(f"  3. η = 1: lensing = dynamics (both potentials enhanced equally)")
    print(f"  4. Cosmological limit: mu(a=1) = 1.053 (5.3% enhancement at z=0)")
    print(f"\nThe non-linear constitutive relation F(S̃) must be:")
    print(f"  - Strain-SOFTENING: F grows SLOWER than E×S̃ at high strain (r < 50 kpc)")
    print(f"  - But the ENCLOSED MASS must grow FASTER than linear at R > 100 kpc")
    print(f"  - This is a contradiction if F(S̃) is monotonic and ρ ∝ S̃²")
    print(f"\nResolution: The non-linear equation would need to change the PROFILE")
    print(f"shape, not just the amplitude. The stress energy must concentrate")
    print(f"at 50-500 kpc (between SPARC range and cosmological scale), while")
    print(f"remaining subdominant at r < 50 kpc.")

    # ── Compute whether profile reshaping is mathematically possible ──
    # The stress density from the rotation curve formula:
    # ρ_stress(r) = α M_bar / (4π β r² (1+r/β)²)
    # This gives M_stress(<R) = α M_bar R/(R+β)
    # At R = 100 kpc: M_stress ≈ α M_bar × 100/β_kpc

    for b in BINS:
        M_bar_msun = b['M_bar'] / M_sun
        M_stress_linear = ALPHA * M_bar_msun * (100 / BETA_KPC)
        M_needed = 8 * M_bar_msun  # from Step 2: ~8-27× M_bar needed
        ratio = M_needed / M_stress_linear

        print(f"\n  {b['name']}:")
        print(f"    Linear stress halo at 100 kpc: {M_stress_linear:.2e} M_sun")
        print(f"    Needed at 100 kpc:             {M_needed:.2e} M_sun")
        print(f"    Amplification required:        {ratio:.0f}×")

    results['c2_gap'] = float(c2)
    results['interpretation'] = (
        "The [E1'] evolution equation in static limit gives a strain field "
        "that is c² ~ 9e16 times smaller than what the rotation curve formula "
        "requires. This discrepancy means either: (A) the stress field energy "
        "is negligible and cannot source lensing, or (B) the field equation "
        "normalization differs from [E1'] at galactic scales. In neither case "
        "can the non-linear lensing source term be computed unambiguously from "
        "the existing MTDF postulates."
    )

    # ═══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    verdict = """
GPT's pre-registered Step 3 pass/fail criterion:
  "Does the non-linear MTDF solution raise ΔΣ at R > 100 kpc by ~10×
   while not spoiling v_c in SPARC range and not contradicting η = 1?"

ANSWER: The question cannot be answered within the current framework.

Three blockers prevent a definitive calculation:

  1. THE c² AMBIGUITY: The [E1'] evolution equation gives strain
     S̃ ~ 10⁻¹⁵ at 100 kpc. The rotation curve formula requires
     S̃ ~ 10² at the same radius. The factor c² ≈ 9×10¹⁶ between
     them is unresolved in the published theory.

  2. NO LAGRANGIAN: The MTDF action term L_MTDF is stated to exist
     but its functional form is never given. Without L_MTDF, the
     stress-energy tensor of the elastic field cannot be derived
     from first principles, and non-linear corrections cannot be
     systematically computed.

  3. CONFLICTING CONSTRAINTS: Even if a non-linear constitutive
     relation F(S̃) exists, it must simultaneously:
     - Keep α_eff = 1.30 at r < 50 kpc (SPARC)
     - Boost enclosed mass by 8-27× at R > 100 kpc (GGL)
     Since strain DECREASES with radius, this requires the material
     to be STIFFER at low strain than high strain — physically
     possible (auxetic/phase-transition materials) but not derivable
     from the current postulates.

WHAT THIS MEANS:

  The F2 gap is not a computational problem — it is a theoretical
  one. The framework needs:

  (a) An explicit L_MTDF Lagrangian from which the non-linear
      field equations can be derived, OR

  (b) A physically motivated non-linear constitutive relation
      F(S̃) with at least one new parameter (critical strain S_c
      or similar), OR

  (c) Acceptance that MTDF works cosmologically but does not
      extend to galactic halo scales (100 kpc - 1 Mpc).

  Option (b) is the most promising: a strain-stiffening constitutive
  relation where F(S̃) ~ E S̃ / (1 - (S̃/S_c)²) would amplify the
  stress at low strain (outer halo) relative to high strain (inner
  galaxy). But S_c becomes a new parameter, and without L_MTDF to
  constrain it, the theory loses its zero-free-parameter claim at
  galactic scales.
"""
    print(verdict)
    results['verdict'] = verdict.strip()

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════

    # Plot 1: Strain profiles (E1' vs rotation curve)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    for i, g in enumerate(gap_results):
        r = g['r_kpc']
        ax1.loglog(r, g['S_E1'], '--', label=f"{g['label']} [E1′]", alpha=0.7)
        ax1.loglog(r, g['S_RC'], '-', label=f"{g['label']} rot.curve", alpha=0.7)

    ax1.axhline(1, color='red', ls=':', lw=1.5, label='S̃ = 1 (linear limit)')
    ax1.axvline(50, color='gray', ls=':', alpha=0.5)
    ax1.text(55, 1e-20, 'SPARC range', fontsize=8, color='gray', rotation=90)
    ax1.set_xlabel('Radius [kpc]')
    ax1.set_ylabel('Dimensionless strain |S̃|')
    ax1.set_title('Strain field: [E1′] vs rotation curve requirement')
    ax1.set_xlim(1, 1000)
    ax1.set_ylim(1e-20, 1e10)
    ax1.legend(fontsize=7, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.2)

    # Annotate the gap
    ax1.annotate('', xy=(100, 5e-16), xytext=(100, 50),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(120, 1e-7, f'c² ≈ 9×10¹⁶\ngap', fontsize=10, color='red',
             ha='left', va='center')

    # Plot 2: Stress mass enclosed
    ax2 = axes[1]
    labels = [b['name'] for b in BINS]
    M_E1 = [g['M_stress_E1_100kpc'] for g in gap_results]
    M_RC = [g['M_stress_RC_100kpc'] for g in gap_results]
    M_req = [g['M_stress_required_100kpc'] for g in gap_results]
    M_bar = [b['M_bar'] / M_sun for b in BINS]

    x = np.arange(len(BINS))
    width = 0.2
    ax2.bar(x - width, M_E1, width, label='Stress mass ([E1′])', color='steelblue')
    ax2.bar(x, M_RC, width, label='Stress mass (rot.curve)', color='darkorange')
    ax2.bar(x + width, M_req, width, label='Required (Step 2)', color='firebrick')

    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('M_stress enclosed (< 100 kpc) [M_sun]')
    ax2.set_title('Stress mass budget at 100 kpc')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')

    # Add M_bar reference line for each bin
    for i, m in enumerate(M_bar):
        ax2.plot([i - 0.3, i + 0.3], [m, m], 'k--', lw=1)
    ax2.text(3.35, M_bar[3], 'M_bar', fontsize=8, va='bottom')

    fig.suptitle('Step 3: Non-linear lensing source term analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'step3_nonlinear_source.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_dir / 'step3_nonlinear_source.png'}")

    # Plot 3: Self-sourcing convergence (E1' case)
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    for ib, b in enumerate(BINS):
        hist = self_sourcing_iteration(100, b['M_bar'], n_iter=10)
        iters = [h['iter'] for h in hist]
        masses = [h['M_stress_enclosed'] / M_sun for h in hist]
        ax3.plot(iters, masses, 'o-', label=b['name'])

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Enclosed stress mass (< 100 kpc) [M_sun]')
    ax3.set_title('Self-sourcing feedback: converges instantly (stress energy negligible)')
    ax3.legend()
    ax3.grid(True, alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'step3_self_sourcing.png', dpi=150, bbox_inches='tight')
    print(f"Self-sourcing plot saved: {out_dir / 'step3_self_sourcing.png'}")

    # ── Save JSON ──
    # Clean up non-serializable arrays
    clean_results = {
        'description': results['description'],
        'c2_gap': results['c2_gap'],
        'interpretation': results['interpretation'],
        'gap_analysis': results['gap_analysis'],
        'self_sourcing': results['self_sourcing'],
        'nonlinear_requirement': results['nonlinear_requirement'],
        'verdict': results['verdict'],
    }

    with open(out_dir / 'step3_nonlinear_source.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"Results saved: {out_dir / 'step3_nonlinear_source.json'}")

    plt.close('all')


if __name__ == "__main__":
    main()
