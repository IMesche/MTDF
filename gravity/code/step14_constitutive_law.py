#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 14: The Constitutive Law - Why the Quadratic Self-Interaction

Three results that remove the "this looks chosen" weakness:

1. BTFR EXPONENT SELECTION: For a generic (S-S_0)^n term in the field
   equation, the screened BTFR exponent is (n-1)/(2n). Setting this
   to 1/4 gives n=2 uniquely. No other value works.

2. ENERGY SELF-SOURCING: The elastic energy (E/2)(delta_S)^2 itself
   gravitates (equivalence principle). Adding this self-sourcing to the
   field equation gives lambda_predicted = alpha/(2*beta^2) = 1.33e-48.
   Measured: lambda = 5.1e-48. Ratio ~4 (geometric matching factor).

3. FREEMAN CONSISTENCY: n=2, BTFR exponent 1/4, and Freeman's law
   R proportional to M^{1/2} form a self-consistent triple.
   Perturbing any one breaks the other two.

After Step 14: the screening is what the medium does, not what was chosen.

The matching argument (generalising Step 11 to arbitrary n):
  Field equation: E beta^2 [nabla^2 S - lambda (S-S_0)^n] = alpha rho c^2
  Inside galaxy (strong-source): lambda(S-S_0)^n ~ alpha rho c^2 / (E beta^2)
    => (S-S_0) ~ (alpha rho c^2 / (E beta^2 lambda))^{1/n}
  At boundary (r ~ R_gal): C = R * (S-S_0) at boundary
    => C ~ R * (alpha M c^2 / (E beta^2 lambda R^3))^{1/n}
    => C ~ M^{1/n} * R^{1-3/n}
  With Freeman's law (R ~ M^{1/2}):
    => C ~ M^{1/n + (1-3/n)/2} = M^{(n-1)/(2n)}
  Since f ~ C: f ~ M^{(n-1)/(2n)}
  BTFR requires this exponent = 1/4, giving n = 2.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import hashlib

# ================================================================
# CONSTANTS - ALL FROM MTDF (Steps 8-13)
# ================================================================

ALPHA = 1.30
BETA_KPC = 22_685.0
BETA_M = 7.0e23           # m
E_PA = 9.1e-10            # Pa
G_SI = 6.674e-11
C_SI = 2.998e8             # m/s
MSUN = 1.989e30            # kg
KPC_M = 3.086e19           # m per kpc

# Derived MTDF parameters
L_KPC = ALPHA * BETA_KPC / (4 * np.pi)
L_M = L_KPC * KPC_M
RHO_CRIT = 8.5e-27         # kg/m^3
S_0 = np.sqrt(2 * 0.70 * RHO_CRIT * C_SI**2 / E_PA)  # 1.084
V_REF = 161.8e3             # m/s
RHO0_SI = E_PA * S_0**2 / (2 * C_SI**2)

# Step 11 results
LAMBDA_MEASURED = 5.1e-48   # m^-2
M_BAR = np.array([1.15e10, 3.04e10, 5.26e10, 8.18e10])  # M_sun
F_MEASURED = np.array([0.8027, 1.0901, 1.2126, 1.3320])
BIN_LABELS = ['Bin 1', 'Bin 2', 'Bin 3', 'Bin 4']

# Freeman's law calibration
R_D_REF = 3.5   # kpc at M_ref = 3e10 M_sun
M_REF = 3.0e10  # M_sun


# ================================================================
# PART A: BTFR EXPONENT SELECTION
# ================================================================

def compute_btfr_exponents():
    """Show that the BTFR exponent (n-1)/(2n) = 1/4 uniquely selects n=2."""

    # Continuous curve
    n_cont = np.linspace(0.5, 6.0, 500)
    exp_cont = (n_cont - 1) / (2 * n_cont)

    # Integer and half-integer values
    n_discrete = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
    exp_discrete = [(n - 1) / (2 * n) for n in n_discrete]

    # Algebraic proof: (n-1)/(2n) = 1/4 => 4(n-1) = 2n => 2n = 4 => n = 2
    # The BTFR observational uncertainty is about +/- 0.01 (McGaugh)
    # For exponent in [0.24, 0.26], n ranges:
    #   0.24 = (n-1)/(2n) => n = 1/(1-2*0.24) = 1/0.52 = 1.923
    #   0.26 = (n-1)/(2n) => n = 1/(1-2*0.26) = 1/0.48 = 2.083
    n_low = 1 / (1 - 2 * 0.24)   # 1.923
    n_high = 1 / (1 - 2 * 0.26)  # 2.083

    return {
        'n_continuous': n_cont.tolist(),
        'exponent_continuous': exp_cont.tolist(),
        'n_discrete': n_discrete,
        'exponent_discrete': exp_discrete,
        'unique_solution': 2,
        'algebraic_proof': '4(n-1) = 2n => n = 2',
        'btfr_uncertainty_n_range': [float(n_low), float(n_high)],
    }


# ================================================================
# PART B: ENERGY SELF-SOURCING
# ================================================================

def compute_lambda_prediction():
    """Derive lambda from the principle: elastic energy gravitates."""

    # The elastic energy density is:
    #   u_elastic = (E/2)(S - S_0)^2
    # Its gravitating mass density:
    #   rho_elastic = u_elastic / c^2 = (E / 2c^2)(S - S_0)^2
    #
    # The total source in the field equation:
    #   source = alpha * (rho_baryon + rho_elastic) * c^2
    #          = alpha * rho_baryon * c^2 + alpha * (E/2)(S - S_0)^2
    #
    # So the field equation becomes:
    #   E beta^2 nabla^2 S = alpha rho c^2 + alpha (E/2)(S-S_0)^2
    #   E beta^2 [nabla^2 S - alpha/(2 beta^2) (S-S_0)^2] = alpha rho c^2
    #
    # Reading off: lambda = alpha / (2 beta^2)

    lambda_base = ALPHA / (2 * BETA_M**2)

    # Geometric variants (boundary matching could introduce O(1) factors)
    variants = {
        'alpha/(2*beta^2)': ALPHA / (2 * BETA_M**2),
        'alpha/beta^2': ALPHA / BETA_M**2,
        'pi*alpha/(2*beta^2)': np.pi * ALPHA / (2 * BETA_M**2),
        '2*alpha/beta^2': 2 * ALPHA / BETA_M**2,
        '3*alpha/(2*beta^2)': 3 * ALPHA / (2 * BETA_M**2),
        '4*pi*alpha/beta^2': 4 * np.pi * ALPHA / BETA_M**2,
    }

    # Find closest variant
    ratios = {k: v / LAMBDA_MEASURED for k, v in variants.items()}
    closest = min(ratios.items(), key=lambda x: abs(np.log(x[1])))

    return {
        'principle': 'Elastic energy gravitates (equivalence principle)',
        'derivation': 'lambda = alpha / (2 beta^2)',
        'lambda_base': float(lambda_base),
        'lambda_measured': LAMBDA_MEASURED,
        'ratio_base': float(lambda_base / LAMBDA_MEASURED),
        'variants': {k: float(v) for k, v in variants.items()},
        'variant_ratios': {k: float(v) for k, v in ratios.items()},
        'closest_variant': closest[0],
        'closest_ratio': float(closest[1]),
    }


# ================================================================
# PART C: FREEMAN CONSISTENCY + STABILITY
# ================================================================

def compute_freeman_consistency():
    """Show n=2, Freeman p=1/2, BTFR q=1/4 are mutually consistent."""

    # General relation: f ~ M^{(n-1)/(2n)} requires
    #   (n-1)/(2n) = q  (where q = BTFR exponent)
    # This used Freeman R ~ M^p. The full relation is:
    #   q = (1 + pn - 3p) / n = 1/n + p - 3p/n = 1/n + p(1 - 3/n)
    # So: q = [1 + p(n - 3)] / n
    # Solving for n: qn = 1 + pn - 3p => n(q - p) = 1 - 3p
    #   n = (1 - 3p) / (q - p)
    # For q = 1/4: n = (1 - 3p) / (1/4 - p) = 4(1 - 3p) / (1 - 4p)

    # Verify bidirectional
    p_freeman = 0.5
    n_from_p = 4 * (1 - 3 * p_freeman) / (1 - 4 * p_freeman)  # should be 2.0

    # Inverse: given n=2, what p?
    # q = [1 + p(n-3)]/n = [1 + p(2-3)]/2 = (1-p)/2
    # 1/4 = (1-p)/2 => 1/2 = 1-p => p = 1/2
    n_given = 2
    p_from_n = 1 - 2 * 0.25  # = 0.5

    # Continuous curves for the consistency plot
    p_range = np.linspace(0.3, 0.7, 200)
    # Avoid singularity at p = 1/4 (where 1 - 4p = 0)
    n_from_p_arr = np.where(
        np.abs(1 - 4 * p_range) > 0.01,
        4 * (1 - 3 * p_range) / (1 - 4 * p_range),
        np.nan
    )

    # What BTFR exponent would different n give?
    n_range = np.linspace(1.0, 5.0, 200)
    q_from_n_arr = (n_range - 1) / (2 * n_range)

    # Stability: (S-S_0)^2 is the leading Landau-Ginzburg term
    # Expand any smooth self-interaction V(S) around S_0:
    #   V(S) = V(S_0) + V'(S_0)(S-S_0) + (1/2)V''(S_0)(S-S_0)^2 + ...
    # V'(S_0) = 0 at the equilibrium (S_0 is a minimum)
    # V''(S_0) > 0 for stability
    # The leading non-trivial self-interaction in the EOM:
    #   V''(S_0)(S-S_0) gives a mass term (not screening)
    #   V'''(S_0)(S-S_0)^2 / 2 gives (S-S_0)^2 screening
    # So (S-S_0)^2 in the field equation corresponds to the cubic
    # coefficient of V(S), which is the LEADING anharmonic term.

    return {
        'p_freeman': p_freeman,
        'n_from_p_0.5': float(n_from_p),
        'p_from_n_2': float(p_from_n),
        'bidirectional': True,
        'consistency_formula': 'n = 4(1-3p)/(1-4p)',
        'stability': 'n=2 is leading Landau-Ginzburg anharmonic term',
        'p_range': p_range.tolist(),
        'n_from_p_curve': [float(x) if not np.isnan(x) else None
                           for x in n_from_p_arr],
        'n_range': n_range.tolist(),
        'q_from_n_curve': q_from_n_arr.tolist(),
    }


# ================================================================
# PART D: TRANSITION MASS
# ================================================================

def compute_transition_mass():
    """Derive the self-sourcing amplification factor and transition scale.

    The self-sourcing amplification is f_screened / f_linear. This ratio
    is always >> 1 for astrophysical objects (because f_linear is tiny).
    The physical transition is: at what mass does the screened compression
    become OBSERVATIONALLY SIGNIFICANT (f_screened > threshold)?

    For the Sun, f_screened ~ 0.002 and f_linear ~ 5e-16. Both are
    negligible. For the MW, f_screened ~ 1.15 and f_linear ~ 3e-5.
    Only f_screened matters.

    The self-sourcing amplification factor decreases with mass (as M^{-3/4}),
    which means it is larger for small masses. But at small masses, even
    the amplified f is tiny. The transition where f_screened becomes
    observationally relevant (> 0.01) occurs at M ~ 10^{2.5} M_sun.
    """

    # BTFR coefficient
    A_BTFR = 50.0 * MSUN / (1e3)**4  # kg / (m/s)^4

    # Compute amplification factors for representative objects
    objects = [
        ('Sun', MSUN),
        ('Globular cluster (10^5)', 1e5 * MSUN),
        ('Dwarf galaxy (10^8)', 1e8 * MSUN),
        ('MW (6e10)', 6e10 * MSUN),
        ('Cluster (10^14)', 1e14 * MSUN),
    ]

    amp_table = []
    for name, M_kg in objects:
        f_lin = C_SI**2 * M_kg / (E_PA * BETA_M**3 * S_0)
        v_flat = (M_kg / A_BTFR)**0.25
        f_scr = v_flat / V_REF
        amplification = f_scr / f_lin if f_lin > 0 else np.inf
        amp_table.append({
            'object': name,
            'M_kg': float(M_kg),
            'M_Msun': float(M_kg / MSUN),
            'f_linear': float(f_lin),
            'f_screened': float(f_scr),
            'amplification': float(amplification),
            'regime': 'negligible' if f_scr < 0.01 else 'significant',
        })

    # Transition mass from Step 13 (where f_screened > 0.01)
    # f_screened = (M/A_BTFR)^{0.25} / V_REF = 0.01
    # M = A_BTFR * (0.01 * V_REF)^4
    M_tr_01 = A_BTFR * (0.01 * V_REF)**4
    M_tr_01_log = np.log10(M_tr_01 / MSUN)

    # Transition where f_screened > 0.1 (clearly observable)
    M_tr_01_obs = A_BTFR * (0.1 * V_REF)**4
    M_tr_01_obs_log = np.log10(M_tr_01_obs / MSUN)

    return {
        'amplification_table': amp_table,
        'transition_f_0.01': {
            'M_Msun': float(M_tr_01 / MSUN),
            'log10_Msun': float(M_tr_01_log),
            'meaning': 'self-sourcing produces measurable compression',
        },
        'transition_f_0.1': {
            'M_Msun': float(M_tr_01_obs / MSUN),
            'log10_Msun': float(M_tr_01_obs_log),
            'meaning': 'compression clearly observable (dwarf galaxies)',
        },
        'step13_threshold_log10_Msun': 2.5,
        'key_insight': 'Self-sourcing amplification is always present but '
                       'only produces observable effects above ~10^2.5 M_sun. '
                       'Solar system safety is structural, not fine-tuned.',
    }


# ================================================================
# PART E: BROUWER BIN VERIFICATION
# ================================================================

def compute_brouwer_verification():
    """Compare predicted f(M) for different n against measured values."""

    log_M = np.log10(M_BAR)
    n_values = [1, 2, 3, 4]

    results = {}
    for n in n_values:
        exponent = (n - 1) / (2 * n)
        # Normalize to Bin 1
        f_predicted = F_MEASURED[0] * (M_BAR / M_BAR[0]) ** exponent
        residuals = (f_predicted - F_MEASURED) / F_MEASURED
        rms = float(np.sqrt(np.mean(residuals**2)))
        # chi^2 with 5% uncertainty (from Step 11 lambda scatter)
        sigma_f = 0.05 * F_MEASURED
        chi2 = float(np.sum(((f_predicted - F_MEASURED) / sigma_f)**2))

        results[f'n={n}'] = {
            'n': n,
            'exponent': float(exponent),
            'f_predicted': f_predicted.tolist(),
            'residuals': residuals.tolist(),
            'rms': rms,
            'chi2': chi2,
        }

    # Best n
    chi2_vals = {n: results[f'n={n}']['chi2'] for n in n_values}
    best_n = min(chi2_vals, key=chi2_vals.get)

    results['f_measured'] = F_MEASURED.tolist()
    results['M_bar'] = M_BAR.tolist()
    results['best_n'] = best_n
    results['chi2_table'] = chi2_vals

    return results


# ================================================================
# PLOTTING
# ================================================================

def plot_btfr_exponent(btfr_results, outdir):
    """BTFR exponent vs constitutive power n."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_cont = np.array(btfr_results['n_continuous'])
    exp_cont = np.array(btfr_results['exponent_continuous'])

    # Continuous curve
    ax.plot(n_cont, exp_cont, 'k-', lw=2, label='$(n-1)/(2n)$')

    # Mark discrete values
    n_disc = btfr_results['n_discrete']
    exp_disc = btfr_results['exponent_discrete']
    colors = ['gray'] * len(n_disc)
    sizes = [8] * len(n_disc)
    # Highlight n=2
    for i, n in enumerate(n_disc):
        if n == 2:
            colors[i] = '#2ca02c'
            sizes[i] = 14
    for i, (n, e) in enumerate(zip(n_disc, exp_disc)):
        ax.plot(n, e, 'o', color=colors[i], ms=sizes[i], zorder=5,
                markeredgecolor='k', markeredgewidth=1)
        if n in [1, 2, 3, 4]:
            offset = 0.015 if n != 2 else 0.020
            ax.annotate(f'$n={n}$: {e:.3f}',
                        xy=(n, e), xytext=(n + 0.2, e + offset),
                        fontsize=9, ha='left')

    # Observed BTFR line
    ax.axhline(0.25, color='red', ls='--', lw=1.5,
               label='Observed BTFR: $q = 0.250$')

    # BTFR uncertainty band
    ax.axhspan(0.24, 0.26, alpha=0.1, color='red',
               label='BTFR uncertainty ($\\pm 0.01$)')

    # Mark n=2 intersection
    ax.plot(2, 0.25, '*', color='red', ms=18, zorder=6,
            markeredgecolor='k', markeredgewidth=1)

    ax.set_xlabel('Constitutive power $n$ in $(S - S_0)^n$', fontsize=12)
    ax.set_ylabel('Predicted BTFR exponent $q = (n-1)/(2n)$', fontsize=12)
    ax.set_title('Step 14: The BTFR uniquely selects $n = 2$', fontsize=13)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.05, 0.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)

    # Asymptote
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.text(5.3, 0.48, '$q \\to 1/2$', fontsize=8, color='gray', va='top')

    fig.tight_layout()
    fig.savefig(outdir / 'step14_btfr_exponent.png', dpi=150)
    plt.close(fig)


def plot_screening_comparison(brouwer_results, outdir):
    """Different n values against Brouwer f(M) data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    log_M = np.log10(np.array(brouwer_results['M_bar']))
    f_meas = np.array(brouwer_results['f_measured'])
    sigma_f = 0.05 * f_meas

    # Extended mass range for smooth curves
    log_M_ext = np.linspace(9.8, 11.3, 200)

    colors = {1: 'gray', 2: '#2ca02c', 3: '#d62728', 4: '#1f77b4'}
    styles = {1: ':', 2: '-', 3: '--', 4: '-.'}

    # Left panel: f(M) comparison
    ax1.errorbar(log_M, f_meas, yerr=sigma_f, fmt='ko', ms=10, capsize=4,
                 lw=2, label='Measured (Step 9)', zorder=5)

    for n in [1, 2, 3, 4]:
        exp = (n - 1) / (2 * n)
        f_pred = f_meas[0] * (10**(log_M_ext - log_M[0])) ** exp
        ax1.plot(log_M_ext, f_pred, color=colors[n], ls=styles[n], lw=2,
                 label=f'$n={n}$: $q = {exp:.3f}$')

    ax1.set_xlabel('$\\log_{10}(M_{bar} / M_\\odot)$', fontsize=12)
    ax1.set_ylabel('Compression parameter $f$', fontsize=12)
    ax1.set_title('Predicted $f(M)$ for different constitutive powers', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right panel: chi^2 per n
    n_vals = [1, 2, 3, 4]
    chi2_vals = [brouwer_results[f'n={n}']['chi2'] for n in n_vals]

    bar_colors = ['gray', '#2ca02c', '#d62728', '#1f77b4']
    bars = ax2.bar(n_vals, chi2_vals, color=bar_colors, edgecolor='k', alpha=0.8)
    ax2.set_xlabel('Constitutive power $n$', fontsize=12)
    ax2.set_ylabel('$\\chi^2$ (4 bins, 5% errors)', fontsize=12)
    ax2.set_title('Only $n=2$ fits the data', fontsize=11)
    ax2.set_xticks(n_vals)

    for i, (n, c) in enumerate(zip(n_vals, chi2_vals)):
        ax2.text(n, c + max(chi2_vals) * 0.02, f'{c:.1f}', ha='center',
                 va='bottom', fontsize=11, fontweight='bold')

    # Dotted line at chi^2 = 4 (1 per bin)
    ax2.axhline(4.0, color='gray', ls=':', lw=1, label='$\\chi^2 = 4$ (1 per bin)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(outdir / 'step14_screening_comparison.png', dpi=150)
    plt.close(fig)


# ================================================================
# OUTPUT
# ================================================================

def make_json_serializable(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    outdir = Path(__file__).parent.parent / 'output' / 'step14_constitutive_law'
    outdir.mkdir(parents=True, exist_ok=True)

    print("Step 14: The Constitutive Law - Why (S-S_0)^2")
    print("=" * 65)

    # ── Part A ──
    btfr = compute_btfr_exponents()
    print("\n  Part A: BTFR Exponent Selection")
    print(f"  For (S-S_0)^n screening, the BTFR exponent is (n-1)/(2n)")
    print(f"  {'n':<6} {'Exponent':<12} {'vs BTFR (0.25)':<16}")
    for n, e in zip(btfr['n_discrete'], btfr['exponent_discrete']):
        if isinstance(n, int) or n == int(n):
            dev = e - 0.25
            marker = ' <-- MATCH' if abs(dev) < 0.001 else ''
            print(f"  {n:<6.0f} {e:<12.4f} {dev:+.4f}{marker}")
    print(f"\n  Algebraic proof: {btfr['algebraic_proof']}")
    print(f"  n = 2 is the UNIQUE solution (integer or otherwise)")
    print(f"  BTFR uncertainty [0.24, 0.26] maps to n in "
          f"[{btfr['btfr_uncertainty_n_range'][0]:.3f}, "
          f"{btfr['btfr_uncertainty_n_range'][1]:.3f}]")

    # ── Part B ──
    lam = compute_lambda_prediction()
    print(f"\n  Part B: Energy Self-Sourcing")
    print(f"  Principle: {lam['principle']}")
    print(f"  Derivation: {lam['derivation']}")
    print(f"  lambda_predicted (base) = {lam['lambda_base']:.3e} m^-2")
    print(f"  lambda_measured          = {lam['lambda_measured']:.3e} m^-2")
    print(f"  Ratio (predicted/measured) = {lam['ratio_base']:.2f}")
    print(f"\n  Geometric variants:")
    for name, val in lam['variants'].items():
        ratio = lam['variant_ratios'][name]
        marker = ' <-- closest' if name == lam['closest_variant'] else ''
        print(f"    {name:<25} = {val:.3e}  (ratio {ratio:.3f}){marker}")
    print(f"\n  Best match: {lam['closest_variant']} "
          f"(ratio {lam['closest_ratio']:.3f})")

    # ── Part C ──
    freeman = compute_freeman_consistency()
    print(f"\n  Part C: Freeman Consistency Triple")
    print(f"  Relation: {freeman['consistency_formula']}")
    print(f"  p = 0.5 (Freeman) => n = {freeman['n_from_p_0.5']:.1f}")
    print(f"  n = 2 (constitutive) => p = {freeman['p_from_n_2']:.1f}")
    print(f"  Bidirectional: {freeman['bidirectional']}")
    print(f"  Stability: {freeman['stability']}")

    # ── Part D ──
    trans = compute_transition_mass()
    print(f"\n  Part D: Self-Sourcing Amplification")
    print(f"  {'Object':<24} {'f_linear':<12} {'f_screened':<12} "
          f"{'Amplification':<16} {'Regime':<12}")
    for row in trans['amplification_table']:
        print(f"  {row['object']:<24} {row['f_linear']:<12.2e} "
              f"{row['f_screened']:<12.4f} {row['amplification']:<16.1e} "
              f"{row['regime']:<12}")
    t01 = trans['transition_f_0.01']
    t10 = trans['transition_f_0.1']
    print(f"\n  Transition (f_screened > 0.01): M > 10^{t01['log10_Msun']:.1f} M_sun")
    print(f"  Transition (f_screened > 0.1):  M > 10^{t10['log10_Msun']:.1f} M_sun")
    print(f"  Step 13 threshold:              10^{trans['step13_threshold_log10_Msun']:.1f} M_sun")
    print(f"  Solar system safety: n-independent (linear regime, f ~ 10^-16)")

    # ── Part E ──
    brouwer = compute_brouwer_verification()
    print(f"\n  Part E: Brouwer Bin Verification")
    print(f"  {'n':<6} {'Exponent':<12} {'chi^2':<10} {'RMS':<10}")
    for n in [1, 2, 3, 4]:
        r = brouwer[f'n={n}']
        marker = ' <-- BEST' if n == brouwer['best_n'] else ''
        print(f"  {n:<6} {r['exponent']:<12.4f} {r['chi2']:<10.2f} "
              f"{r['rms']:<10.4f}{marker}")

    print(f"\n  Per-bin detail for n=2:")
    r2 = brouwer['n=2']
    print(f"  {'Bin':<8} {'f_meas':<10} {'f_pred':<10} {'residual':<10}")
    for i in range(4):
        print(f"  {BIN_LABELS[i]:<8} {F_MEASURED[i]:<10.4f} "
              f"{r2['f_predicted'][i]:<10.4f} {r2['residuals'][i]:+.4f}")

    # ── Part F: Summary ──
    print(f"\n" + "=" * 65)
    print(f"  SUMMARY: What the constitutive law determines")
    print(f"  {'Quantity':<28} {'Value':<24} {'Status':<14}")
    print(f"  {'-'*65}")
    print(f"  {'n (screening power)':<28} {'2 (unique from BTFR)':<24} {'DERIVED':<14}")
    print(f"  {'lambda (self-sourcing)':<28} "
          f"{lam['lambda_base']:.2e} m^-2"
          f"{'':>2} {'PREDICTED (~4x)':<14}")
    print(f"  {'Freeman p = 1/2':<28} {'Bidirectional':<24} {'CONSISTENT':<14}")
    print(f"  {'f ~ M^(1/4)':<28} {'BTFR scaling':<24} {'MATCHED':<14}")
    print(f"  {'Solar system':<28} {'n-independent':<24} {'SAFE (>10^20)':<14}")
    print(f"  {'V(S) form':<28} {'Leading Landau-Ginzburg':<24} {'NATURAL':<14}")
    print(f"  {'-'*65}")
    print(f"  The (S-S_0)^2 screening is what the medium does,")
    print(f"  not what was chosen.")
    print(f"=" * 65)

    # ── Save JSON ──
    all_results = {
        'description': 'Step 14: Constitutive law - why (S-S_0)^2',
        'part_A_btfr_exponent': {
            'exponent_formula': '(n-1)/(2n)',
            'n_values': btfr['n_discrete'],
            'exponents': btfr['exponent_discrete'],
            'unique_solution': btfr['unique_solution'],
            'algebraic_proof': btfr['algebraic_proof'],
            'btfr_uncertainty_n_range': btfr['btfr_uncertainty_n_range'],
        },
        'part_B_lambda_prediction': lam,
        'part_C_freeman_consistency': {
            'p_freeman': freeman['p_freeman'],
            'n_from_p': freeman['n_from_p_0.5'],
            'p_from_n': freeman['p_from_n_2'],
            'bidirectional': freeman['bidirectional'],
            'formula': freeman['consistency_formula'],
            'stability': freeman['stability'],
        },
        'part_D_transition': {
            'transition_f_0.01_log10_Msun': trans['transition_f_0.01']['log10_Msun'],
            'transition_f_0.1_log10_Msun': trans['transition_f_0.1']['log10_Msun'],
            'step13_threshold': trans['step13_threshold_log10_Msun'],
            'key_insight': trans['key_insight'],
        },
        'part_E_brouwer_verification': {
            'chi2_table': brouwer['chi2_table'],
            'best_n': brouwer['best_n'],
            'rms_n2': brouwer['n=2']['rms'],
        },
        'field_equation': 'E beta^2 [nabla^2 S - alpha/(2 beta^2) (S-S0)^2] = alpha rho c^2',
        'status': 'n=2 uniquely selected; lambda approximately predicted',
    }

    json_results = make_json_serializable(all_results)
    with open(outdir / 'step14_constitutive_law.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # ── Plots ──
    plot_btfr_exponent(btfr, outdir)
    plot_screening_comparison(brouwer, outdir)

    # ── Manifest ──
    manifest = {}
    for p in sorted(outdir.iterdir()):
        if p.name != 'manifest.json':
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            manifest[p.name] = h
    with open(outdir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Output saved to {outdir}/")


if __name__ == '__main__':
    main()
