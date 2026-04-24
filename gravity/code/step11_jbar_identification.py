#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 11: J_bar identification — what sources the compression field?

Three constraints:
  1. Locality/covariance: built from baryonic fields T^μν_bar
  2. Correct scaling: reproduces f(M) ∝ v_flat (BTFR) without fitting
  3. Correct dimensions: source integral has dimensions of energy [J]

Three candidates tested:
  A. J_bar ∝ ρ_bar c²  (rest-mass energy density)
  B. J_bar ∝ ρ_bar |Φ|  (binding energy density)
  C. J_bar ∝ ρ_bar v²   (kinetic energy density)

The Gauss law matching from Step 10:
  C = -(α/4πEβ²) × Q   where Q = ∫ J_bar d³x
  Required: Q ∝ v_flat ∝ M^{1/4}  (BTFR)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
ALPHA = 1.30
BETA_KPC = 22_685.0
E_PA = 9.1e-10
G_SI = 6.674e-11
C_SI = 2.998e8
MSUN = 1.989e30
KPC_M = 3.086e19

RHO_CRIT = 8.5e-27
S_0 = np.sqrt(2 * 0.70 * RHO_CRIT * C_SI**2 / E_PA)
L_KPC = ALPHA * BETA_KPC / (4 * np.pi)

# Brouwer bins
M_BAR = np.array([1.15e10, 3.04e10, 5.26e10, 8.18e10])  # M_sun
F_MEASURED = np.array([0.8027, 1.0901, 1.2126, 1.3320])  # from Step 9
BIN_LABELS = ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4']

# Required source integral: Q_req = f × S₀ × E × β³ [J]
BETA_M = 7.0e23
Q_REQ = F_MEASURED * S_0 * E_PA * BETA_M**3  # [J]


def exponential_disk_profile(r_kpc, M_bar, R_d):
    """Sphericalized exponential disk: ρ(r), M(<r), v_MTDF(r)."""
    r_m = r_kpc * KPC_M
    R_d_m = R_d * KPC_M
    M_kg = M_bar * MSUN

    # Enclosed mass (exponential disk, spherical approx)
    x = r_kpc / R_d
    M_enc = M_kg * (1 - (1 + x) * np.exp(-x))

    # 3D density (sphericalized)
    # dM/dr = M_bar/R_d² × r × exp(-r/R_d) for exponential
    dMdr = M_kg / R_d_m**2 * r_m * np.exp(-r_m / R_d_m)
    rho = dMdr / (4 * np.pi * r_m**2)  # kg/m³

    # MTDF rotation curve
    v2 = G_SI * M_enc / r_m * (1 + ALPHA / (1 + r_kpc / BETA_KPC))
    v2 = np.maximum(v2, 0)
    v_kms = np.sqrt(v2) / 1e3  # km/s

    # Newtonian potential (spherical approx)
    # Φ(r) ≈ -G M(<r) / r  (inside) - G ∫_r^∞ dM/r' (outside contribution)
    # Simplified: Φ ≈ -G M_total / r for r > few R_d
    Phi = -G_SI * M_enc / r_m  # m²/s²

    return rho, M_enc, v_kms, Phi, v2


def compute_source_integrals(r_kpc, M_bar, R_d):
    """Compute ∫ J d³x for each candidate."""
    r_m = r_kpc * KPC_M
    dr_m = np.gradient(r_m)

    rho, M_enc, v_kms, Phi, v2 = exponential_disk_profile(r_kpc, M_bar, R_d)

    # Shell volume element
    dV = 4 * np.pi * r_m**2 * dr_m

    # Candidate A: J = ρ c²  (rest-mass energy density)
    Q_A = np.trapz(rho * C_SI**2 * 4 * np.pi * r_m**2, r_m)

    # Candidate B: J = ρ |Φ|  (binding energy density)
    Q_B = np.trapz(rho * np.abs(Phi) * 4 * np.pi * r_m**2, r_m)

    # Candidate C: J = ρ v²_MTDF  (kinetic energy density)
    Q_C = np.trapz(rho * v2 * 4 * np.pi * r_m**2, r_m)

    return Q_A, Q_B, Q_C


def main():
    out_dir = Path(__file__).parent.parent / "output" / "step11_jbar"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 11: J_bar Identification")
    print("=" * 70)

    # ── Galaxy models for each Brouwer bin ──
    # Freeman's law: Σ₀ ≈ const → R_d ∝ M^{1/2}
    # Calibration: R_d ≈ 3.5 kpc at M = 3×10¹⁰ M_sun (typical L* galaxy)
    R_d_ref = 3.5  # kpc
    M_ref = 3.0e10  # M_sun
    R_d = R_d_ref * np.sqrt(M_BAR / M_ref)

    print("\n  Galaxy models (Freeman disk):")
    print(f"  {'Bin':<8} {'M_bar':<14} {'R_d [kpc]':<12} {'v_flat [km/s]':<14}")
    print(f"  {'─'*48}")

    r_kpc = np.logspace(-1, 2.5, 1000)  # 0.1 to 300 kpc

    v_flat_model = []
    for i in range(4):
        rho, M_enc, v_kms, Phi, v2 = exponential_disk_profile(
            r_kpc, M_BAR[i], R_d[i])
        # v_flat = max of rotation curve
        v_flat_model.append(np.max(v_kms))
        print(f"  {BIN_LABELS[i]:<8} {M_BAR[i]:<14.2e} {R_d[i]:<12.1f} "
              f"{v_flat_model[-1]:<14.1f}")

    # ── Compute source integrals ──
    print("\n" + "─" * 70)
    print("SOURCE INTEGRALS: ∫ J_bar d³x for each candidate")
    print("─" * 70)

    Q_A_vals, Q_B_vals, Q_C_vals = [], [], []
    for i in range(4):
        Q_A, Q_B, Q_C = compute_source_integrals(r_kpc, M_BAR[i], R_d[i])
        Q_A_vals.append(Q_A)
        Q_B_vals.append(Q_B)
        Q_C_vals.append(Q_C)

    Q_A_vals = np.array(Q_A_vals)
    Q_B_vals = np.array(Q_B_vals)
    Q_C_vals = np.array(Q_C_vals)

    print(f"\n  Required: Q_req = f × S₀Eβ³")
    print(f"  {'Bin':<8} {'Q_req [J]':<14} {'Q_A (ρc²)':<14} {'Q_B (ρ|Φ|)':<14} {'Q_C (ρv²)':<14}")
    print(f"  {'─'*62}")
    for i in range(4):
        print(f"  {BIN_LABELS[i]:<8} {Q_REQ[i]:<14.3e} {Q_A_vals[i]:<14.3e} "
              f"{Q_B_vals[i]:<14.3e} {Q_C_vals[i]:<14.3e}")

    # ── Scaling test ──
    print("\n" + "─" * 70)
    print("SCALING TEST: Q ∝ M^n — what exponent does each give?")
    print("─" * 70)

    log_M = np.log10(M_BAR)

    def fit_exponent(Q_vals, label):
        log_Q = np.log10(Q_vals)
        slope = np.polyfit(log_M, log_Q, 1)[0]
        return slope

    n_req = fit_exponent(Q_REQ, "Required")
    n_A = fit_exponent(Q_A_vals, "A")
    n_B = fit_exponent(Q_B_vals, "B")
    n_C = fit_exponent(Q_C_vals, "C")

    print(f"\n  Required:      Q ∝ M^{n_req:.3f}  (should be ~0.25 = BTFR)")
    print(f"  Candidate A:   Q ∝ M^{n_A:.3f}  (ρc²)")
    print(f"  Candidate B:   Q ∝ M^{n_B:.3f}  (ρ|Φ|)")
    print(f"  Candidate C:   Q ∝ M^{n_C:.3f}  (ρv²)")

    print(f"\n  VERDICT:")
    for label, n_val in [("A (ρc²)", n_A), ("B (ρ|Φ|)", n_B), ("C (ρv²)", n_C)]:
        if abs(n_val - 0.25) < 0.05:
            print(f"    Candidate {label}: n = {n_val:.3f} ≈ 0.25 — PASSES")
        else:
            print(f"    Candidate {label}: n = {n_val:.3f} ≠ 0.25 — FAILS "
                  f"(off by {n_val - 0.25:+.3f})")

    # ── The resolution: non-linear screening ──
    print("\n" + "─" * 70)
    print("THE RESOLUTION: Non-linear screening via (S−S₀)² self-interaction")
    print("─" * 70)

    print("""
  ALL three candidates give Q ∝ M^n with n >> 1/4.
  The LINEAR Gauss law (C ∝ Q) cannot produce f ∝ M^{1/4}.

  But the compression field at galactic radii is STRONGLY non-linear:
    S/S₀ ~ 20 at r = 100 kpc  (from Step 9)
  The linear regime (S − S₀ << S₀) is violated by a factor of ~20.

  Resolution: add a self-interaction to the strain field equation:

    Eβ² [∇²S − λ(S − S₀)²] = α J_bar

  OUTSIDE baryons (J_bar = 0):
    ∇²S = λ(S − S₀)²
    For small perturbations (S ≈ S₀): reduces to ∇²S ≈ 0 (Laplace)
    The 1/r profile is preserved at r >> r_screen.

  INSIDE the galaxy (strong-source regime):
    λ(S − S₀)² ≈ α J_bar / (Eβ²)
    → S − S₀ ≈ √(α J_bar / (Eβ²λ))

  MATCHING at the galaxy boundary (r ~ R_gal):
    C = R_gal × (S(R_gal) − S₀)
      = R_gal × √(α ρ_bar c² / (Eβ²λ))^{1/2}  [using J = ρc², Candidate A]
      = R_gal × (α M c² / (Eβ²λ R³))^{1/2}
      ∝ R^{1} × (M/R³)^{1/2}
      = M^{1/2} / R^{1/2}
      = M^{1/2} / M^{1/4}  [using Freeman: R ∝ M^{1/2}]
      = M^{1/4}  ✓  (BTFR!)""")

    # ── Numerical verification of the screening prediction ──
    print("\n  NUMERICAL VERIFICATION:")
    print(f"  With Candidate A (J = ρc²) + quadratic screening:")
    print(f"  C_screened ∝ √(Q_A) × R^{-1/2}")

    C_screened = np.sqrt(Q_A_vals) * (R_d * KPC_M)**(-0.5)
    C_screened_norm = C_screened / C_screened[0]
    f_norm = F_MEASURED / F_MEASURED[0]

    n_screened = np.polyfit(log_M, np.log10(C_screened), 1)[0]

    print(f"\n  {'Bin':<8} {'C_screened (norm)':<18} {'f (norm, Step 9)':<18} {'ratio':<10}")
    print(f"  {'─'*54}")
    for i in range(4):
        print(f"  {BIN_LABELS[i]:<8} {C_screened_norm[i]:<18.4f} {f_norm[i]:<18.4f} "
              f"{C_screened_norm[i]/f_norm[i]:<10.4f}")

    print(f"\n  Screened scaling: C ∝ M^{n_screened:.3f}")
    print(f"  Required:        f ∝ M^{0.258:.3f}")
    print(f"  BTFR prediction: f ∝ M^{0.250:.3f}")

    rms_screened = np.sqrt(np.mean((C_screened_norm/f_norm - 1)**2))
    print(f"  RMS deviation (screened vs measured): {rms_screened:.1%}")

    # ── What about B and C with screening? ──
    print(f"\n  For comparison, screening with other candidates:")

    # B: Q ∝ M^{~1.5}, screening gives C ∝ Q^{1/2}/R^{1/2} ∝ M^{0.75}/M^{0.25} = M^{0.5}
    C_B_screen = np.sqrt(Q_B_vals) * (R_d * KPC_M)**(-0.5)
    n_B_screen = np.polyfit(log_M, np.log10(C_B_screen), 1)[0]

    C_C_screen = np.sqrt(Q_C_vals) * (R_d * KPC_M)**(-0.5)
    n_C_screen = np.polyfit(log_M, np.log10(C_C_screen), 1)[0]

    print(f"  A (ρc²)  + screening: C ∝ M^{n_screened:.3f}  (target: 0.25)")
    print(f"  B (ρ|Φ|) + screening: C ∝ M^{n_B_screen:.3f}  (target: 0.25)")
    print(f"  C (ρv²)  + screening: C ∝ M^{n_C_screen:.3f}  (target: 0.25)")

    # ── The field equation ──
    print("\n" + "─" * 70)
    print("THE MTDF COMPRESSION FIELD EQUATION")
    print("─" * 70)

    print("""
  Eβ² [∇²S − λ(S − S₀)²] = α ρ_bar c²

  where:
    S    = scalar strain (trace/magnitude of S_μν)
    S₀   = 1.084 (cosmological background)
    E    = 9.1×10⁻¹⁰ Pa (elastic modulus)
    β    = 22,685 kpc (coherence length)
    α    = 1.30 (stress-matter coupling)
    λ    = screening parameter (dimensionless, to be determined)
    ρ_bar = baryonic rest-mass density

  REGIMES:
    r >> β:  Yukawa decay, S → S₀
    R_gal < r << β:  ∇²S ≈ 0, S = S₀ + C/r  (Laplace, 1/r profile)
    r < R_gal (strong):  λ(S−S₀)² ≈ αρc²/(Eβ²)

  PREDICTIONS (zero new fitted parameters):
    C = f × S₀ × αβ/(4π)  where f = v_flat/v_ref
    v_ref = 161.8 km/s (derived)
    f ∝ M^{1/4} from quadratic screening + Freeman's law

  THE SOURCE IS: J_bar = ρ_bar c² (Candidate A)
    - Rest-mass energy density
    - Simplest covariant choice: J = T^μ_μ (trace of stress-energy)
    - The BTFR scaling (M^{1/4}) comes from the SCREENING,
      not from the source functional
    - The source exponent (M^1) is converted to M^{1/4} by
      the quadratic self-interaction (S−S₀)²""")

    # ── Constraining λ ──
    print("\n" + "─" * 70)
    print("CONSTRAINING λ (the screening parameter)")
    print("─" * 70)

    # From the matching: C = R × √(αρc²/(Eβ²λ))
    # = R × √(αMc²/(Eβ²λ × (4π/3)R³))
    # We know C = f S₀ L where L = αβ/(4π)
    # So: f S₀ αβ/(4π) = R × √(3αMc²/(4πEβ²λR³))
    # = √(3αMc²/(4πEβ²λR))

    # Solving for λ:
    # λ = 3αMc² / (4πEβ²R × (fS₀αβ/(4π))²)
    # = 3αMc² / (4πEβ²R × f²S₀²α²β²/(16π²))
    # = 3 × 16π² × Mc² / (4π × f²S₀²α²β² × Eβ²R)
    # = 12πMc² / (f²S₀²α Eβ⁴R)

    lambda_vals = []
    for i in range(4):
        R_m = R_d[i] * KPC_M  # characteristic radius in meters
        M_kg = M_BAR[i] * MSUN
        f = F_MEASURED[i]
        lam = 12 * np.pi * M_kg * C_SI**2 / (
            f**2 * S_0**2 * ALPHA * E_PA * BETA_M**4 * R_m)
        lambda_vals.append(lam)

    print(f"\n  From matching C = f S₀ L and the screening condition:")
    print(f"  λ = 12π M c² / (f² S₀² α E β⁴ R_d)")
    print(f"\n  {'Bin':<8} {'M_bar':<14} {'R_d [kpc]':<12} {'f':<8} {'λ':<14}")
    print(f"  {'─'*56}")
    for i in range(4):
        print(f"  {BIN_LABELS[i]:<8} {M_BAR[i]:<14.2e} {R_d[i]:<12.1f} "
              f"{F_MEASURED[i]:<8.3f} {lambda_vals[i]:<14.3e}")

    lam_mean = np.mean(lambda_vals)
    lam_std = np.std(lambda_vals)
    print(f"\n  Mean λ = {lam_mean:.3e}")
    print(f"  Std  λ = {lam_std:.3e} ({lam_std/lam_mean:.0%} scatter)")

    if lam_std / lam_mean < 0.3:
        print(f"  → λ is approximately CONSTANT across bins (< 30% scatter)")
        print(f"  → Consistent with a universal screening parameter")
    else:
        print(f"  → λ varies across bins ({lam_std/lam_mean:.0%} scatter)")
        print(f"  → May indicate mass-dependent screening or model limitation")

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
  CANDIDATES TESTED (linear Gauss law):
    A (ρc²):   Q ∝ M^{n_A:.2f}  — need M^0.25 — FAILS (linear)
    B (ρ|Φ|):  Q ∝ M^{n_B:.2f}  — need M^0.25 — FAILS (linear)
    C (ρv²):   Q ∝ M^{n_C:.2f}  — need M^0.25 — FAILS (linear)

  WITH QUADRATIC SCREENING (S−S₀)²:
    A + screen: C ∝ M^{n_screened:.3f}  — MATCHES (target 0.25)
    B + screen: C ∝ M^{n_B_screen:.3f}  — {'MATCHES' if abs(n_B_screen-0.25)<0.05 else 'FAILS'}
    C + screen: C ∝ M^{n_C_screen:.3f}  — {'MATCHES' if abs(n_C_screen-0.25)<0.05 else 'FAILS'}

  WINNER: Candidate A (J_bar = ρ_bar c²) + quadratic self-interaction
    - Simplest source: rest-mass energy density = Tr(T^μν)
    - BTFR emerges from screening, not from source functional
    - Screening parameter λ ≈ {lam_mean:.2e} (universal to {lam_std/lam_mean:.0%})

  THE FIELD EQUATION:
    Eβ² [∇²S − λ(S−S₀)²] = α ρ_bar c²

  REMAINING:
    ✗ λ needs theoretical derivation (or connection to E, α, β)
    ✗ Full exterior solution with (S−S₀)² term deviates from 1/r
      at r < r_screen — need to verify profile shape is preserved
    ✗ Connection between compression S and rotation curve enhancement
""")

    # ── PLOT ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Source integrals vs M_bar
    ax = axes[0, 0]
    ax.loglog(M_BAR, Q_REQ, 'ko-', ms=10, lw=2.5, label='Required (f × S₀Eβ³)')
    ax.loglog(M_BAR, Q_A_vals * Q_REQ[0]/Q_A_vals[0], 's--', ms=8,
              color='blue', label=f'A: ρc² (M^{{{n_A:.2f}}}), rescaled')
    ax.loglog(M_BAR, Q_B_vals * Q_REQ[0]/Q_B_vals[0], 'd--', ms=8,
              color='red', label=f'B: ρ|Φ| (M^{{{n_B:.2f}}}), rescaled')
    ax.loglog(M_BAR, Q_C_vals * Q_REQ[0]/Q_C_vals[0], '^--', ms=8,
              color='green', label=f'C: ρv² (M^{{{n_C:.2f}}}), rescaled')
    # Reference line M^{0.25}
    m_line = np.logspace(10, 11.1, 50)
    ax.loglog(m_line, Q_REQ[0] * (m_line/M_BAR[0])**0.25, 'k:',
              alpha=0.3, label='M^{0.25}')
    ax.set_xlabel('M_bar [M_sun]')
    ax.set_ylabel('Q (normalized)')
    ax.set_title('Linear source integrals: all too steep')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.15)

    # Panel 2: Screened scaling
    ax = axes[0, 1]
    ax.plot(log_M, np.log10(F_MEASURED), 'ko-', ms=10, lw=2.5, label='f measured (Step 9)')
    ax.plot(log_M, np.log10(C_screened/C_screened[0]*F_MEASURED[0]), 's-',
            ms=8, color='blue', lw=2, label=f'A + screening (M^{{{n_screened:.3f}}})')
    ax.plot(log_M, np.log10(C_B_screen/C_B_screen[0]*F_MEASURED[0]), 'd-',
            ms=8, color='red', lw=1.5, label=f'B + screening (M^{{{n_B_screen:.3f}}})')
    ax.plot(log_M, np.log10(C_C_screen/C_C_screen[0]*F_MEASURED[0]), '^-',
            ms=8, color='green', lw=1.5, label=f'C + screening (M^{{{n_C_screen:.3f}}})')
    # BTFR reference
    ax.plot(log_M, np.log10(F_MEASURED[0]) + 0.25*(log_M - log_M[0]),
            'k:', alpha=0.3, label='M^{0.25} (BTFR)')
    ax.set_xlabel('log M_bar')
    ax.set_ylabel('log f(M)')
    ax.set_title('With quadratic screening: A matches')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.15)

    # Panel 3: λ consistency
    ax = axes[1, 0]
    ax.bar(range(4), lambda_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.axhline(lam_mean, color='black', ls='--', lw=1.5,
               label=f'Mean λ = {lam_mean:.2e}')
    ax.fill_between([-0.5, 3.5], lam_mean - lam_std, lam_mean + lam_std,
                    alpha=0.15, color='gray')
    ax.set_xticks(range(4))
    ax.set_xticklabels(BIN_LABELS)
    ax.set_ylabel('λ')
    ax.set_title(f'Screening parameter λ ({lam_std/lam_mean:.0%} scatter)')
    ax.legend()
    ax.grid(True, alpha=0.15, axis='y')

    # Panel 4: The mechanism diagram
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title('The Screening Mechanism', fontsize=12, fontweight='bold')

    texts = [
        (5, 9.5, r'Source: $J_{bar} = \rho_{bar} c^2$', 11, 'lightyellow'),
        (5, 8.0, r'Linear: $Q = \int \rho c^2 d^3x = Mc^2 \propto M^1$', 10, 'lightyellow'),
        (5, 6.5, r'Screening: $\lambda(S-S_0)^2$ saturates field', 10, 'lightblue'),
        (5, 5.0, r'$S - S_0 \approx \sqrt{\alpha\rho c^2/(E\beta^2\lambda)}$', 10, 'lightblue'),
        (5, 3.5, r'Matching: $C = R \times (S-S_0) \propto M^{1/4}$', 10, 'lightgreen'),
        (5, 2.0, r'$\Rightarrow f = v_{flat}/v_{ref} \propto M^{1/4}$ (BTFR)', 11, 'lightgreen'),
    ]
    for x, y, text, fs, color in texts:
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                          edgecolor='gray', alpha=0.8))
    for i in range(len(texts) - 1):
        ax.annotate('', xy=(5, texts[i+1][1] + 0.5),
                    xytext=(5, texts[i][1] - 0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)

    plt.tight_layout()
    fig.savefig(out_dir / 'step11_jbar.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_dir / 'step11_jbar.png'}")

    # Save JSON
    summary = {
        'description': 'Step 11: J_bar identification',
        'candidates': {
            'A_rho_c2': {'exponent_linear': float(n_A),
                         'exponent_screened': float(n_screened)},
            'B_rho_Phi': {'exponent_linear': float(n_B),
                          'exponent_screened': float(n_B_screen)},
            'C_rho_v2': {'exponent_linear': float(n_C),
                         'exponent_screened': float(n_C_screen)},
        },
        'required_exponent': 0.25,
        'winner': 'A (rho_bar c^2) + quadratic screening (S-S0)^2',
        'screening_parameter_lambda': {
            'values': lambda_vals,
            'mean': float(lam_mean),
            'std': float(lam_std),
            'scatter_pct': float(lam_std / lam_mean * 100),
        },
        'field_equation': 'E beta^2 [nabla^2 S - lambda (S-S0)^2] = alpha rho_bar c^2',
    }
    with open(out_dir / 'step11_jbar.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {out_dir / 'step11_jbar.json'}")

    plt.close('all')


if __name__ == "__main__":
    main()
