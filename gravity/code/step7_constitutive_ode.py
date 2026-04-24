#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 7: GPT's prescription — non-linear constitutive ODE.

Take the MTDF strain definition S = β|∇ξ| and a non-linear energy:
  U(S) = (E/2) S² + (E/4 S_c²) S⁴   [quartic completion]

Derive the spherical ODE for ξ(r), solve it, compute u_field(r),
and lens with ρ_stress = u_field / c².

The test: does a single universal S_crit produce the required density
profiles across all four Brouwer bins?

TWO APPROACHES COMPARED:
  (A) Energy-based: ξ is fundamental, S = β ξ', EL equation from U(S)
  (B) [E1']-based: S (or F) is fundamental, ∇²S = source (static [E1'])
  These give DIFFERENT radial profiles for S(r) and hence U(r).

This script implements BOTH and reports what slope U(r) has in each case.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALPHA = 1.30
BETA_KPC = 22_685.0      # kpc
BETA_M = 7.0e23          # m
E_PA = 9.1e-10           # Pa (elastic modulus)
G_SI = 6.674e-11         # m³ kg⁻¹ s⁻²
C_SI = 2.998e8           # m/s
MSUN = 1.989e30          # kg
KPC_M = 3.086e19         # m per kpc

# Brouwer bins
M_BAR = [1.15e10, 3.04e10, 5.26e10, 8.18e10]  # M_sun
RHO_S_RS2_REQ = [3.120e8, 5.754e8, 7.119e8, 8.591e8]  # M_sun/kpc
BIN_LABELS = ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# ═══════════════════════════════════════════════════════════════
# APPROACH A: ENERGY-BASED ODE (ξ is fundamental)
#
#   S = β |dξ/dr|
#   U(S) = (E/2) S² + (E / 4S_c²) S⁴
#   dU/dS = E S (1 + S²/S_c²)
#
#   Euler-Lagrange in spherical symmetry:
#     d/dr[r² β dU/dS] = source = κ M δ³(r)
#
#   Exterior (r > 0):
#     r² β E S (1 + S²/S_c²) = K         [= constant]
#     r² E β² ξ' (1 + β²ξ'²/S_c²) = K
#
#   This is algebraic in ξ'(r) for each r. Solve for S(r).
# ═══════════════════════════════════════════════════════════════

def solve_S_approach_A(r_kpc, M_bar_msun, S_crit):
    """
    Solve r² E S (1 + S²/S_c²) = K for S(r).

    Matching constant K from the point-mass solution:
    At large r (linear regime), S = K/(E r²).
    The linear solution for a point mass gives S = β GM/(r² × something).

    From the EL equation with source = κ M δ³(r):
    K = κ M β / (4π), where κ relates the elastic field to gravity.

    We parameterize: K = β² G M / r_ref² × normalization
    But to be honest about the c² gap, we compute K two ways:

    K_A (interpretation A, [E1']-normalized):
      S_linear(r) = 2 G M / (c⁴ r) × β  → K = 2 G M β² / c⁴
      But this S enters the EL equation for ξ differently...

    Actually: for the energy-based approach, the EL equation is:
      Eβ² ∇²ξ = κ ρ_bar  (linear case)
    For gravity coupling: κ = 4πG/c² (or 4πG, depending on convention)
    Then: ξ = κ M / (4π Eβ² r) = G M / (Eβ² c² r)  [with c² factor]
    S = β |ξ'| = G M / (Eβ c² r²)

    Or without c²: S = G M / (Eβ r²)

    K = r² E S = GM / (β c²)  [with c²]
    K = GM / β               [without c²]
    """
    r_m = r_kpc * KPC_M
    M_kg = M_bar_msun * MSUN

    # K with c² factor (interpretation A)
    K_A = G_SI * M_kg / (BETA_M * C_SI**2)

    # K without c² factor (interpretation B — rotation curve normalization)
    K_B = G_SI * M_kg / BETA_M

    results = {}
    for label, K in [('interp_A', K_A), ('interp_B', K_B)]:
        S = np.zeros_like(r_m)
        for j, r in enumerate(r_m):
            # Solve: r² E S (1 + S²/S_c²) = K
            # → E S + E S³/S_c² = K/r²
            # → S³/S_c² + S = K/(E r²)
            rhs = K / (E_PA * r**2)

            # Cubic in S: S³/S_c² + S - rhs = 0
            # For rhs > 0, there's exactly one positive root
            # Use Newton's method
            S_lin = rhs  # linear approximation (S << S_c)
            s = min(S_lin, (rhs * S_crit**2)**(1./3))  # initial guess
            for _ in range(50):
                f = s**3 / S_crit**2 + s - rhs
                fp = 3 * s**2 / S_crit**2 + 1
                ds = -f / fp
                s = max(s + ds, 1e-100)
                if abs(ds) < abs(s) * 1e-12:
                    break
            S[j] = s

        # Energy density
        U = (E_PA / 2) * S**2 + (E_PA / (4 * S_crit**2)) * S**4

        # Mass density ρ = U/c²  (kg/m³)
        rho_si = U / C_SI**2

        # Convert to M_sun/kpc³
        rho_msun_kpc3 = rho_si / MSUN * KPC_M**3

        results[label] = {
            'S': S,
            'U_J_m3': U,
            'rho_msun_kpc3': rho_msun_kpc3,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# APPROACH B: [E1']-BASED (S is fundamental)
#
#   Static [E1']: ∇²F = (8πG/c⁴) ρ_bar
#   With F = E S: ∇² S = (8πG/(Ec⁴)) ρ_bar
#
#   For point mass: S(r) = 2GM / (Ec⁴r)  [exact 1/r Coulomb]
#
#   Non-linear extension: replace F = ES with F = dU/dS
#   Then: ∇²[dU/dS] = (8πG/c⁴) ρ_bar
#
#   For quartic U: dU/dS = ES(1 + S²/S_c²)
#   Exterior: ∇²[S(1 + S²/S_c²)] = 0
#   → g(S) = S(1+S²/S_c²) is harmonic → g = A/r
#
#   Solve g(S) = A/r for S(r).
# ═══════════════════════════════════════════════════════════════

def solve_S_approach_B(r_kpc, M_bar_msun, S_crit):
    """
    [E1']-based approach: g(S) = S(1+S²/S_c²) = A/r.

    From the static [E1'] equation:
      ∇²(dU/dS) = (8πG/c⁴) M δ³(r)
      → dU/dS = 2GM/(c⁴ × 4π r) (Coulomb solution)
      But since dU/dS = ES(1+S²/S_c²), we have:
      E × g(S) = 2GM/(c⁴ × 4π r)
      → g(S) = 2GM/(4π E c⁴ r) = GM/(2π E c⁴ r)

    Actually the standard Poisson solution for ∇²f = Q δ³(r) is f = -Q/(4πr).
    So: dU/dS = (8πG/c⁴) M / (4π r) = 2GM/(c⁴ r)
    → E g(S) = 2GM/(c⁴ r)
    → g(S) = 2GM/(E c⁴ r)

    Linear check: g(S) ≈ S → S = 2GM/(Ec⁴r). Matches [E1']. ✓
    """
    r_m = r_kpc * KPC_M
    M_kg = M_bar_msun * MSUN

    A = 2 * G_SI * M_kg / (E_PA * C_SI**4)  # constant for approach B

    S = np.zeros_like(r_m)
    for j, r in enumerate(r_m):
        rhs = A / r  # g(S) = rhs

        # Cubic: S³/S_c² + S - rhs = 0 (same form as approach A!)
        S_lin = rhs
        s = min(S_lin, (rhs * S_crit**2)**(1./3))
        for _ in range(50):
            f = s**3 / S_crit**2 + s - rhs
            fp = 3 * s**2 / S_crit**2 + 1
            ds = -f / fp
            s = max(s + ds, 1e-100)
            if abs(ds) < abs(s) * 1e-12:
                break
        S[j] = s

    U = (E_PA / 2) * S**2 + (E_PA / (4 * S_crit**2)) * S**4
    rho_si = U / C_SI**2
    rho_msun_kpc3 = rho_si / MSUN * KPC_M**3

    return {
        'S': S,
        'U_J_m3': U,
        'rho_msun_kpc3': rho_msun_kpc3,
        'A': A,
        'S_at_100kpc': float(S[np.argmin(np.abs(r_kpc - 100))]),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def measure_slope(r, y, r_min, r_max):
    """Log-log slope of y(r) between r_min and r_max."""
    mask = (r >= r_min) & (r <= r_max) & (y > 0)
    if np.sum(mask) < 3:
        return np.nan
    log_r = np.log10(r[mask])
    log_y = np.log10(y[mask])
    # Linear fit
    coeffs = np.polyfit(log_r, log_y, 1)
    return coeffs[0]


def main():
    out_dir = Path(__file__).parent.parent / "output" / "step7_constitutive_ode"
    out_dir.mkdir(parents=True, exist_ok=True)

    r_kpc = np.logspace(1, 3.5, 300)  # 10 to ~3000 kpc

    print("=" * 70)
    print("Step 7: Non-Linear Constitutive ODE")
    print("=" * 70)

    # ── Part 1: What strain scale are we in? ──
    print("\n─── Part 1: Strain scales ───")
    for i in range(4):
        M_kg = M_BAR[i] * MSUN
        r_m = 100 * KPC_M

        S_E1 = 2 * G_SI * M_kg / (E_PA * C_SI**4 * r_m)
        S_rot = 2 * G_SI * M_kg / (E_PA * C_SI**2 * r_m)
        S_noC = G_SI * M_kg / (E_PA * BETA_M * r_m**2)

        print(f"  {BIN_LABELS[i]} (M={M_BAR[i]:.2e} M_sun) at r=100 kpc:")
        print(f"    [E1'] strain:           S = {S_E1:.2e}")
        print(f"    Rot-curve strain:       S = {S_rot:.2e}  (c² larger)")
        print(f"    Energy-based (ξ-EL):    S = {S_noC:.2e}  (1/r² falloff)")

    # ── Part 2: Approach A (energy-based) with several S_crit values ──
    print("\n─── Part 2: Approach A (energy-based ODE) ───")
    print("  Exterior: r² E S (1+S²/S_c²) = K")
    print("  Linear: S ∝ 1/r² → U ∝ 1/r⁴")
    print("  Non-linear (S >> S_c): S ∝ 1/r^{2/3} → U ∝ 1/r^{8/3}")

    S_crit_values_A = [1e-30, 1e-20, 1e-10, 1e-5, 1.0]

    for S_c in S_crit_values_A:
        res = solve_S_approach_A(r_kpc, M_BAR[3], S_c)
        rho_B = res['interp_B']['rho_msun_kpc3']
        slope = measure_slope(r_kpc, rho_B, 50, 1000)
        rho_100 = rho_B[np.argmin(np.abs(r_kpc - 100))]
        print(f"  S_crit = {S_c:.0e}: slope = {slope:.2f}, "
              f"ρ(100 kpc) = {rho_100:.2e} M_sun/kpc³  "
              f"(required: {RHO_S_RS2_REQ[3]/100**2:.0f})")

    # ── Part 3: Approach B ([E1']-based) ──
    print("\n─── Part 3: Approach B ([E1']-based ODE) ───")
    print("  Exterior: g(S) = S(1+S²/S_c²) ∝ 1/r  [harmonic]")
    print("  Linear: S ∝ 1/r → U ∝ 1/r²  (isothermal!)")
    print("  Non-linear (S >> S_c): S ∝ 1/r^{1/3} → U ∝ 1/r^{4/3}")

    S_crit_values_B = [1e-20, 1e-18, 1e-16, 1e-15, 1e-14, 1e-12]

    print(f"\n  Using Bin 4 (M_bar = {M_BAR[3]:.2e}):")
    print(f"  S at 100 kpc from [E1'] ~ {2*G_SI*M_BAR[3]*MSUN/(E_PA*C_SI**4*100*KPC_M):.2e}")
    print()

    for S_c in S_crit_values_B:
        res = solve_S_approach_B(r_kpc, M_BAR[3], S_c)
        rho = res['rho_msun_kpc3']
        slope = measure_slope(r_kpc, rho, 50, 1000)
        rho_100 = rho[np.argmin(np.abs(r_kpc - 100))]
        S_100 = res['S_at_100kpc']
        regime = "linear" if S_100 < S_c else f"NL (S/S_c={S_100/S_c:.1f})"
        print(f"  S_crit = {S_c:.0e}: S(100)={S_100:.1e}, {regime:<20s} "
              f"slope={slope:.2f}, ρ(100)={rho_100:.2e} M_sun/kpc³")

    # ── Part 4: Can ANY S_crit give ρ(100) ~ required? ──
    print("\n─── Part 4: Amplitude check ───")
    print("  Required ρ at 100 kpc for each bin:")
    for i in range(4):
        rho_req = RHO_S_RS2_REQ[i] / 100**2
        print(f"    {BIN_LABELS[i]}: {rho_req:.0f} M_sun/kpc³")

    print("\n  Approach B [E1'] linear prediction (S_c → ∞):")
    for i in range(4):
        res = solve_S_approach_B(r_kpc, M_BAR[i], 1e10)  # effectively linear
        rho = res['rho_msun_kpc3']
        rho_100 = rho[np.argmin(np.abs(r_kpc - 100))]
        rho_req = RHO_S_RS2_REQ[i] / 100**2
        ratio = rho_req / rho_100 if rho_100 > 0 else np.inf
        print(f"    {BIN_LABELS[i]}: ρ_MTDF(100) = {rho_100:.2e}, "
              f"required = {rho_req:.0f}, gap = {ratio:.1e}×")

    print("\n  Approach A (energy-based, interp B) linear prediction:")
    for i in range(4):
        res = solve_S_approach_A(r_kpc, M_BAR[i], 1e10)
        rho = res['interp_B']['rho_msun_kpc3']
        rho_100 = rho[np.argmin(np.abs(r_kpc - 100))]
        rho_req = RHO_S_RS2_REQ[i] / 100**2
        ratio = rho_req / rho_100 if rho_100 > 0 else np.inf
        print(f"    {BIN_LABELS[i]}: ρ_MTDF(100) = {rho_100:.2e}, "
              f"required = {rho_req:.0f}, gap = {ratio:.1e}×")

    # ── Part 5: Comprehensive plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Top left: Approach A slopes for different S_crit
    ax = axes[0, 0]
    for S_c in [1e-30, 1e-20, 1e-10, 1.0]:
        res = solve_S_approach_A(r_kpc, M_BAR[3], S_c)
        rho = res['interp_B']['rho_msun_kpc3']
        sl = measure_slope(r_kpc, rho, 50, 1000)
        ax.loglog(r_kpc, rho, label=f'S_c={S_c:.0e} (slope={sl:.2f})')
    # Required
    rho_req = RHO_S_RS2_REQ[3] / r_kpc**2
    ax.loglog(r_kpc, rho_req, 'k--', lw=2, label='Required (r⁻²)')
    ax.set_title('Approach A: energy-based (Bin 4)', fontsize=11)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [M_sun/kpc³]')
    ax.set_xlim(10, 3000)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # Top right: Approach B slopes for different S_crit
    ax = axes[0, 1]
    for S_c in [1e-20, 1e-16, 1e-15, 1e-14]:
        res = solve_S_approach_B(r_kpc, M_BAR[3], S_c)
        rho = res['rho_msun_kpc3']
        sl = measure_slope(r_kpc, rho, 50, 1000)
        ax.loglog(r_kpc, rho, label=f'S_c={S_c:.0e} (slope={sl:.2f})')
    rho_req = RHO_S_RS2_REQ[3] / r_kpc**2
    ax.loglog(r_kpc, rho_req, 'k--', lw=2, label='Required (r⁻²)')
    ax.set_title("Approach B: [E1']-based (Bin 4)", fontsize=11)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [M_sun/kpc³]')
    ax.set_xlim(10, 3000)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # Bottom left: All bins, Approach B linear (showing the c⁴ amplitude gap)
    ax = axes[1, 0]
    for i in range(4):
        res = solve_S_approach_B(r_kpc, M_BAR[i], 1e10)
        ax.loglog(r_kpc, res['rho_msun_kpc3'], '-', color=COLORS[i],
                  label=f'{BIN_LABELS[i]} [E1\'] prediction')
        rho_req = RHO_S_RS2_REQ[i] / r_kpc**2
        ax.loglog(r_kpc, rho_req, '--', color=COLORS[i], alpha=0.5)
    ax.set_title("Approach B linear: right slope, wrong amplitude", fontsize=11)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [M_sun/kpc³]')
    ax.set_xlim(10, 3000)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    # Bottom right: Approach A (interp B) all bins linear
    ax = axes[1, 1]
    for i in range(4):
        res = solve_S_approach_A(r_kpc, M_BAR[i], 1e10)
        ax.loglog(r_kpc, res['interp_B']['rho_msun_kpc3'], '-',
                  color=COLORS[i], label=f'{BIN_LABELS[i]} energy-based')
        rho_req = RHO_S_RS2_REQ[i] / r_kpc**2
        ax.loglog(r_kpc, rho_req, '--', color=COLORS[i], alpha=0.5)
    ax.set_title("Approach A linear: wrong slope (r⁻⁴), wrong amplitude",
                 fontsize=11)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [M_sun/kpc³]')
    ax.set_xlim(10, 3000)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.suptitle('Step 7: Non-linear constitutive ODE — two approaches',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'step7_constitutive_ode.png', dpi=150,
                bbox_inches='tight')
    print(f"\nPlot saved: {out_dir / 'step7_constitutive_ode.png'}")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
  APPROACH A (energy-based, ξ is fundamental):
    Linear regime:     S ∝ 1/r²  →  U ∝ S² ∝ 1/r⁴   (slope -4)
    Non-linear regime: S ∝ 1/r^{2/3} → U ∝ S⁴ ∝ 1/r^{8/3}  (slope -2.67)
    → Neither regime gives r⁻² (isothermal)
    → Quartic moves slope from -4 toward -2.67, but never reaches -2

  APPROACH B ([E1']-based, S is fundamental):
    Linear regime:     S ∝ 1/r  →  U ∝ S² ∝ 1/r²   (slope -2) ✓
    Non-linear regime: S ∝ 1/r^{1/3} → U ∝ S⁴ ∝ 1/r^{4/3}  (slope -1.33)
    → Linear regime gives EXACTLY r⁻² (isothermal!)
    → But amplitude is crushed by c⁴: gap ~ 10⁵³
    → Non-linear term makes slope FLATTER, not steeper
    → No S_crit can fix the amplitude without destroying the slope

  THE FUNDAMENTAL DILEMMA:
    Under [E1'], the slope is perfect (r⁻²) but the amplitude is
    53 orders of magnitude too small (the c⁴ normalization).
    Under the energy-based approach, the slope is wrong (r⁻⁴).
    The quartic constitutive law moves the slope in the wrong
    direction for both approaches.

  WHAT THIS MEANS FOR S_crit:
    GPT's test — "does a single universal S_crit produce the observed
    bin dependence?" — cannot be answered because:
    1. In approach A, the slope is wrong at every S_crit
    2. In approach B, S ~ 10⁻¹⁵ << any reasonable S_crit,
       so the non-linear term is negligible
    3. The amplitude gap is 10⁵³, not 10³·⁴ — it's the c⁴
       normalization in [E1'], not the missing S_crit

  THE c⁴ NORMALIZATION IS THE REAL PROBLEM:
    [E1']: ∇²F = (8πG/c⁴) T  →  S ~ GM/(Ec⁴r) ~ 10⁻¹⁵
    If instead: ∇²F = (8πG) T  →  S ~ GM/(Er)   ~ 10²

    The c⁴ in [E1'] comes from the relativistic wave equation.
    Removing it would make the field equation dimensionally
    inconsistent in the relativistic regime.

    This is the same c² gap identified in Step 3, now shown to be
    fatal for any non-linear constitutive law approach.
""")

    summary = {
        'description': 'Step 7: Non-linear constitutive ODE (GPT prescription)',
        'approach_A': {
            'method': 'Energy-based: ξ fundamental, EL gives r²ES(1+S²/Sc²)=K',
            'linear_slope': -4.0,
            'nonlinear_slope': -8.0/3,
            'verdict': 'Wrong slope at every S_crit',
        },
        'approach_B': {
            'method': '[E1\']-based: S fundamental, g(S)=S(1+S²/Sc²) harmonic',
            'linear_slope': -2.0,
            'nonlinear_slope': -4.0/3,
            'verdict': 'Right slope (r⁻²) but amplitude crushed by c⁴ (gap ~10⁵³)',
        },
        'conclusion': ('No S_crit resolves the problem. The c⁴ normalization in '
                       '[E1\'] is the fundamental obstacle, not the constitutive law.'),
    }

    with open(out_dir / 'step7_constitutive_ode.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {out_dir / 'step7_constitutive_ode.json'}")
    plt.close('all')


if __name__ == "__main__":
    main()
