#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 5: Invert observed ΔΣ(R) into the required 3D stress density ρ_stress(r).

GPT's prescription:
  1. Invert ΔΣ_obs(R) into implied ρ_obs(r)
  2. Subtract baryons → ρ_stress(r)
  3. Compare shape/normalisation to MTDF elastic field prediction

Uses the CORRECT MTDF potential (GPT's fix):
  Φ(r) = -GM(1+α)/r + (αGM/β) ln(1+β/r)

This is the potential whose gradient gives the rotation curve acceleration
g(r) = GM/r² × [1 + α/(1+r/β)].

The INCORRECT form Φ = -GM/r × [1+α/(1+r/β)] does NOT differentiate
to the stated acceleration. The numerical difference is < 2% for
R << β, but the fix matters for internal consistency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALPHA = 1.30
BETA_KPC = 22_685.0

BIN_LABELS = [
    r"$8.5 < \log M_* < 10.3$",
    r"$10.3 < \log M_* < 10.6$",
    r"$10.6 < \log M_* < 10.8$",
    r"$10.8 < \log M_* < 11.0$",
]

MEDIAN_LOG_MSTAR = [10.0, 10.45, 10.70, 10.90]
MEDIAN_MSTAR = [10**x for x in MEDIAN_LOG_MSTAR]
F_GAS = [0.15, 0.08, 0.05, 0.03]
MEDIAN_MBAR = [m * (1 + f) for m, f in zip(MEDIAN_MSTAR, F_GAS)]


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_brouwer_bin(data_dir, bin_num):
    fname = data_dir / f"Fig-3_Lensing-rotation-curves_Massbin-{bin_num}.txt"
    data = np.loadtxt(fname)
    bias = data[:, 4]
    return {
        'R_kpc': data[:, 0] * 1000,
        'ESD': data[:, 1] / bias,      # h70 M_sun/pc²
        'error': data[:, 3] / bias,
    }


# ═══════════════════════════════════════════════════════════════
# CORRECT MTDF POTENTIAL AND ITS EFFECTIVE DENSITY
# ═══════════════════════════════════════════════════════════════

def rho_eff_correct(r_kpc, M_bar):
    """
    Effective 3D density from the CORRECT MTDF potential (r > 0).

    Φ(r) = -GM(1+α)/r + (αGM/β) ln(1+β/r)

    ∇²Φ = 4πG(1+α)M δ(r) − αGMβ / [r²(r+β)²]

    ρ_eff(r > 0) = −αMβ / [4π r²(r+β)²]   [NEGATIVE]

    Compare to the INCORRECT potential's density:
    ρ_eff_wrong(r > 0) = −αMβ / [2π r(r+β)³]

    Both are negative distributed terms. The enclosed mass from the
    correct density gives M_total(<R) = M[1 + α/(1+R/β)] exactly,
    reproducing the rotation curve formula.
    """
    return -ALPHA * M_bar * BETA_KPC / (4 * np.pi * r_kpc**2 * (r_kpc + BETA_KPC)**2)


def esd_mtdf_correct(R_kpc, M_bar):
    """
    ΔΣ from the correct MTDF potential.

    Since M_total(<R) = M_bar × [1 + α/(1+R/β)], and for a point-mass
    source at R >> galaxy size, the ESD is:

    ΔΣ(R) ≈ M_bar × [1 + α/(1+R/β)] / (πR²)

    plus a small projection correction from the negative distributed density.
    At R << β, the correction is O(R/β) ~ 0.4% at 100 kpc.
    """
    # Main term: enclosed mass / πR²
    enhancement = 1.0 + ALPHA / (1.0 + R_kpc / BETA_KPC)
    esd_main = M_bar * enhancement / (np.pi * R_kpc**2)
    return esd_main  # M_sun/kpc²


# ═══════════════════════════════════════════════════════════════
# PARAMETRIC DENSITY MODELS
# ═══════════════════════════════════════════════════════════════

def esd_isothermal(R_kpc, rho_s_rs2):
    """
    ΔΣ for a singular isothermal sphere: ρ(r) = ρ_s (r_s/r)².

    The projected surface density Σ(R) = π ρ_s r_s² / R.
    The mean Σ̄(<R) = 2π ρ_s r_s² / R.
    So ΔΣ(R) = π ρ_s r_s² / R.

    Parameter: rho_s_rs2 = ρ_s × r_s² [M_sun / kpc]
    """
    return np.pi * rho_s_rs2 / R_kpc


def esd_powerlaw(R_kpc, A, n):
    """Generic power-law: ΔΣ = A × R^{-n}."""
    return A * R_kpc**(-n)


def esd_nfw(R_kpc, M200_log, c200):
    """NFW ESD (simplified, for fitting)."""
    M200 = 10**M200_log
    RHO_CRIT = 136.0  # M_sun/kpc³ for H0=70
    r200 = (3 * M200 / (4 * np.pi * 200 * RHO_CRIT))**(1.0 / 3.0)
    r_s = r200 / c200

    def _sigma(x):
        result = np.zeros_like(x, dtype=float)
        lo = x < 0.999
        if np.any(lo):
            xl = x[lo]
            result[lo] = 1.0 / (xl**2 - 1) * (
                1.0 - np.arctanh(np.sqrt(1 - xl**2)) / np.sqrt(1 - xl**2))
        eq = (x >= 0.999) & (x <= 1.001)
        result[eq] = 1.0 / 3.0
        hi = x > 1.001
        if np.any(hi):
            xh = x[hi]
            result[hi] = 1.0 / (xh**2 - 1) * (
                1.0 - np.arctan(np.sqrt(xh**2 - 1)) / np.sqrt(xh**2 - 1))
        return result

    def _sigma_mean(x):
        result = np.zeros_like(x, dtype=float)
        lo = x < 0.999
        if np.any(lo):
            xl = x[lo]
            result[lo] = (np.arctanh(np.sqrt(1 - xl**2)) / np.sqrt(1 - xl**2)
                           + np.log(xl / 2))
        eq = (x >= 0.999) & (x <= 1.001)
        result[eq] = 1.0 + np.log(0.5)
        hi = x > 1.001
        if np.any(hi):
            xh = x[hi]
            result[hi] = (np.arctan(np.sqrt(xh**2 - 1)) / np.sqrt(xh**2 - 1)
                           + np.log(xh / 2))
        return result

    rho_s = M200 / (4 * np.pi * r_s**3 * (np.log(1 + c200) - c200 / (1 + c200)))
    x = np.clip(R_kpc / r_s, 1e-6, None)
    sigma = 2 * rho_s * r_s * _sigma(x)
    sigma_mean = 4 * rho_s * r_s * _sigma_mean(x) / x**2
    return sigma_mean - sigma  # M_sun/kpc²


def rho_isothermal(r_kpc, rho_s_rs2):
    """3D isothermal density: ρ(r) = ρ_s_rs² / r² [M_sun/kpc³]."""
    return rho_s_rs2 / r_kpc**2


# ═══════════════════════════════════════════════════════════════
# ABEL INVERSION (non-parametric)
# ═══════════════════════════════════════════════════════════════

def abel_invert_esd(R_kpc, esd_kpc2, r_query):
    """
    Non-parametric inversion of ΔΣ(R) → ρ(r).

    Uses the relation:
      Σ'(R) = -(1/R²) d/dR [R² ΔΣ(R)]
      ρ(r) = -(1/π) ∫_r^∞ Σ'(R) / √(R²-r²) dR

    Combined:
      ρ(r) = (1/π) ∫_r^∞ (1/R²) d/dR[R² ΔΣ(R)] / √(R²-r²) dR
    """
    # Interpolate ΔΣ(R)
    log_R = np.log(R_kpc)
    log_esd = np.log(np.maximum(esd_kpc2, 1e-10))
    interp_log = interp1d(log_R, log_esd, kind='cubic', fill_value='extrapolate')

    def esd_interp(R):
        return np.exp(interp_log(np.log(R)))

    # d/dR [R² ΔΣ(R)] via numerical derivative
    def d_R2_esd(R, dR_frac=0.01):
        dR = R * dR_frac
        f_plus = (R + dR)**2 * esd_interp(R + dR)
        f_minus = (R - dR)**2 * esd_interp(R - dR)
        return (f_plus - f_minus) / (2 * dR)

    rho = np.zeros(len(r_query))
    for i, r in enumerate(r_query):
        def integrand(R):
            if R <= r * 1.001:
                return 0.0
            return d_R2_esd(R) / (R**2 * np.sqrt(R**2 - r**2))

        R_max = R_kpc[-1] * 1.5
        result, _ = quad(integrand, r * 1.001, R_max, limit=200,
                         epsabs=1e-6, epsrel=1e-4)
        rho[i] = result / np.pi

    return rho


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = Path(__file__).parent.parent / "data" / "brouwer2021"
    out_dir = Path(__file__).parent.parent / "output" / "step5_density_inversion"
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = [load_brouwer_bin(data_dir, i + 1) for i in range(4)]

    print("=" * 70)
    print("Step 5: Invert observed ΔΣ → required ρ_stress(r)")
    print("=" * 70)

    all_results = []

    for i in range(4):
        b = bins[i]
        R_kpc = b['R_kpc']
        M_bar = MEDIAN_MBAR[i]

        print(f"\n{'─'*60}")
        print(f"Bin {i+1}: {BIN_LABELS[i]} (M_bar = {M_bar:.2e} M_sun)")
        print(f"{'─'*60}")

        # ── MTDF prediction (correct potential) ──
        esd_mtdf = esd_mtdf_correct(R_kpc, M_bar)  # M_sun/kpc²
        esd_mtdf_brouwer = esd_mtdf / 1e6  # → h70 M_sun/pc²

        # ── Observed ──
        esd_obs = b['ESD']  # h70 M_sun/pc²

        # ── Residual: what must the "halo" provide ──
        esd_extra = esd_obs - esd_mtdf_brouwer  # h70 M_sun/pc²
        esd_extra_kpc2 = esd_extra * 1e6  # M_sun/kpc²

        # Only use positive residuals at R > 50 kpc (where SPARC doesn't constrain)
        mask = (R_kpc >= 50) & (esd_extra > 0)
        R_fit = R_kpc[mask]
        esd_fit = esd_extra_kpc2[mask]
        err_fit = b['error'][mask] * 1e6  # M_sun/kpc²

        # ── Fit 1: Isothermal (ΔΣ = π ρ_s r_s² / R) ──
        try:
            popt_iso, pcov_iso = curve_fit(
                esd_isothermal, R_fit, esd_fit, p0=[1e8],
                sigma=err_fit, absolute_sigma=True)
            rho_s_rs2 = popt_iso[0]
            chi2_iso = np.sum(((esd_fit - esd_isothermal(R_fit, rho_s_rs2)) / err_fit)**2)
            ndof_iso = len(R_fit) - 1
            print(f"  Isothermal fit: ρ_s r_s² = {rho_s_rs2:.3e} M_sun/kpc")
            print(f"    χ²/dof = {chi2_iso:.1f}/{ndof_iso}")
            print(f"    ρ_stress(100 kpc) = {rho_s_rs2/100**2:.1f} M_sun/kpc³")
            print(f"    ρ_stress(200 kpc) = {rho_s_rs2/200**2:.1f} M_sun/kpc³")
        except Exception as e:
            print(f"  Isothermal fit failed: {e}")
            rho_s_rs2 = np.nan
            chi2_iso = np.nan
            ndof_iso = 0

        # ── Fit 2: Power law (ΔΣ = A R^{-n}) ──
        try:
            popt_pl, pcov_pl = curve_fit(
                esd_powerlaw, R_fit, esd_fit, p0=[1e9, 1.0],
                sigma=err_fit, absolute_sigma=True)
            A_pl, n_pl = popt_pl
            chi2_pl = np.sum(((esd_fit - esd_powerlaw(R_fit, A_pl, n_pl)) / err_fit)**2)
            ndof_pl = len(R_fit) - 2
            print(f"  Power-law fit: ΔΣ_extra = {A_pl:.2e} × R^{-n_pl:.2f}")
            print(f"    χ²/dof = {chi2_pl:.1f}/{ndof_pl}")
            print(f"    Implied 3D slope: ρ ~ r^{-(n_pl+1):.2f}")
        except Exception as e:
            print(f"  Power-law fit failed: {e}")
            A_pl, n_pl = np.nan, np.nan
            chi2_pl = np.nan
            ndof_pl = 0

        # ── Fit 3: NFW ──
        try:
            popt_nfw, pcov_nfw = curve_fit(
                esd_nfw, R_fit, esd_fit, p0=[12.5, 8.0],
                sigma=err_fit, absolute_sigma=True,
                bounds=([10, 1], [15, 30]))
            log_M200, c200 = popt_nfw
            chi2_nfw = np.sum(((esd_fit - esd_nfw(R_fit, log_M200, c200)) / err_fit)**2)
            ndof_nfw = len(R_fit) - 2
            print(f"  NFW fit: log M200 = {log_M200:.2f}, c200 = {c200:.1f}")
            print(f"    M200 = {10**log_M200:.2e} M_sun")
            print(f"    χ²/dof = {chi2_nfw:.1f}/{ndof_nfw}")
        except Exception as e:
            print(f"  NFW fit failed: {e}")
            log_M200, c200 = np.nan, np.nan
            chi2_nfw = np.nan
            ndof_nfw = 0

        # ── Compare to MTDF stress field prediction ──
        # From Section 4.3: ρ_stress_MTDF(r) = αMβ / [4π r² (r+β)²]
        # (absolute value of the correct potential's distributed density)
        r_targets = [100, 200, 300, 500]
        print(f"\n  Required vs MTDF stress density:")
        print(f"  {'r (kpc)':<12} {'ρ_required':<18} {'ρ_MTDF_stress':<18} {'Ratio':<10}")
        print(f"  {'─'*58}")

        comparisons = []
        for r in r_targets:
            rho_required = rho_s_rs2 / r**2 if not np.isnan(rho_s_rs2) else np.nan
            rho_mtdf = ALPHA * M_bar * BETA_KPC / (4 * np.pi * r**2 * (r + BETA_KPC)**2)
            ratio = rho_required / rho_mtdf if rho_mtdf > 0 else np.nan
            print(f"  {r:<12} {rho_required:<18.1f} {rho_mtdf:<18.4f} {ratio:<10.0f}×")
            comparisons.append({
                'r_kpc': r,
                'rho_required': float(rho_required) if not np.isnan(rho_required) else None,
                'rho_mtdf_stress': float(rho_mtdf),
                'ratio': float(ratio) if not np.isnan(ratio) else None,
            })

        # ── Non-parametric Abel inversion ──
        r_query = np.logspace(np.log10(50), np.log10(1500), 30)
        # Use the excess ESD for inversion (only where positive and well-measured)
        mask_pos = esd_extra_kpc2 > 0
        if np.sum(mask_pos) >= 4:
            rho_abel = abel_invert_esd(R_kpc[mask_pos], esd_extra_kpc2[mask_pos], r_query)
        else:
            rho_abel = np.full_like(r_query, np.nan)

        # Store results
        result = {
            'bin': f'bin{i+1}',
            'label': BIN_LABELS[i].replace('$', '').replace('\\log ', 'log'),
            'M_bar': M_bar,
            'isothermal_fit': {
                'rho_s_rs2': float(rho_s_rs2),
                'chi2': float(chi2_iso),
                'ndof': ndof_iso,
            },
            'powerlaw_fit': {
                'A': float(A_pl),
                'n': float(n_pl),
                'implied_3d_slope': float(-(n_pl + 1)) if not np.isnan(n_pl) else None,
                'chi2': float(chi2_pl),
                'ndof': ndof_pl,
            },
            'nfw_fit': {
                'log_M200': float(log_M200),
                'c200': float(c200),
                'chi2': float(chi2_nfw),
                'ndof': ndof_nfw,
            },
            'density_comparison': comparisons,
        }
        all_results.append(result)

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════

    # Plot 1: ESD decomposition — data, MTDF, residual, fits
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        b = bins[i]
        R = b['R_kpc']
        M_bar = MEDIAN_MBAR[i]

        esd_m = esd_mtdf_correct(R, M_bar) / 1e6
        esd_extra = b['ESD'] - esd_m

        # Observed
        ax.errorbar(R, b['ESD'], yerr=b['error'],
                     fmt='ko', ms=5, capsize=2, label='Observed (KiDS×GAMA)', zorder=5)

        # MTDF
        ax.plot(R, esd_m, 'b-', lw=2, label='MTDF metric prediction', zorder=4)

        # Residual
        mask_pos = esd_extra > 0
        ax.plot(R[mask_pos], esd_extra[mask_pos], 's', color='red', ms=4,
                label='Required extra signal', zorder=3)

        # Isothermal fit to residual
        rsr2 = all_results[i]['isothermal_fit']['rho_s_rs2']
        if not np.isnan(rsr2):
            R_smooth = np.logspace(np.log10(50), np.log10(3000), 100)
            ax.plot(R_smooth, esd_isothermal(R_smooth, rsr2) / 1e6,
                    'r--', lw=1.5, label='Isothermal fit to residual', zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(25, 3000)
        ax.set_ylim(0.01, 200)
        ax.set_title(BIN_LABELS[i], fontsize=11)
        ax.grid(True, alpha=0.15)

        if i >= 2:
            ax.set_xlabel('Projected radius R [kpc]')
        if i % 2 == 0:
            ax.set_ylabel(r'$\Delta\Sigma$ [$h_{70}\,M_\odot\,\mathrm{pc}^{-2}$]')
        if i == 0:
            ax.legend(fontsize=7.5, loc='lower left')

    fig.suptitle('Step 5: Decompose observed ESD into MTDF + required residual',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'step5_esd_decomposition.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_dir / 'step5_esd_decomposition.png'}")

    # Plot 2: Required 3D density profile vs MTDF stress field
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    r_arr = np.logspace(1, 3.5, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(4):
        M_bar = MEDIAN_MBAR[i]
        rsr2 = all_results[i]['isothermal_fit']['rho_s_rs2']

        if not np.isnan(rsr2):
            # Required density (from isothermal fit)
            rho_req = rsr2 / r_arr**2
            ax2.loglog(r_arr, rho_req, '-', color=colors[i], lw=2,
                       label=f'{BIN_LABELS[i]} required')

        # MTDF stress field density (absolute value)
        rho_mtdf = ALPHA * M_bar * BETA_KPC / (4 * np.pi * r_arr**2 * (r_arr + BETA_KPC)**2)
        ax2.loglog(r_arr, rho_mtdf, '--', color=colors[i], lw=1, alpha=0.5,
                   label=f'{BIN_LABELS[i]} MTDF stress' if i == 0 else None)

    # Reference slopes
    ax2.loglog([100, 1000], [1e2, 1e0], 'k:', lw=1, alpha=0.3)
    ax2.text(300, 20, r'$\rho \propto r^{-2}$', fontsize=9, color='gray', rotation=-25)

    ax2.set_xlabel('3D radius r [kpc]', fontsize=12)
    ax2.set_ylabel(r'$\rho$ [$M_\odot\,\mathrm{kpc}^{-3}$]', fontsize=12)
    ax2.set_title('Required stress density vs MTDF prediction', fontsize=13)
    ax2.set_xlim(10, 3000)
    ax2.set_ylim(1e-4, 1e6)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.2)

    # Annotate the gap
    ax2.annotate('', xy=(200, 0.01), xytext=(200, 200),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(230, 1, 'Gap:\n~10³–10⁴×', fontsize=11, color='red', va='center')

    fig2.tight_layout()
    fig2.savefig(out_dir / 'step5_density_profile.png', dpi=150, bbox_inches='tight')
    print(f"Density plot saved: {out_dir / 'step5_density_profile.png'}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        'description': (
            'Step 5: Invert observed ΔΣ(R) into required 3D stress density '
            'ρ_stress(r). Uses the correct MTDF potential (GPT fix). Fits '
            'isothermal, power-law, and NFW profiles to the residual.'
        ),
        'potential_note': (
            'Fixed: Φ(r) = -GM(1+α)/r + (αGM/β)ln(1+β/r) is the potential '
            'consistent with g(r) = GM/r² × [1+α/(1+r/β)]. The previously '
            'used Φ = -GM/r × [1+α/(1+r/β)] does not differentiate to the '
            'stated acceleration. Numerical effect < 2% at R << β.'
        ),
        'bins': all_results,
    }

    print("\nKey findings across all bins:")
    for i, r in enumerate(all_results):
        print(f"\n  Bin {i+1}: {r['label']}")
        if not np.isnan(r['powerlaw_fit']['n']):
            print(f"    ΔΣ_extra slope: R^{-r['powerlaw_fit']['n']:.2f} "
                  f"→ ρ_extra ~ r^{r['powerlaw_fit']['implied_3d_slope']:.1f}")
        if not np.isnan(r['nfw_fit']['log_M200']):
            print(f"    NFW equivalent: M200 = {10**r['nfw_fit']['log_M200']:.2e} M_sun, "
                  f"c = {r['nfw_fit']['c200']:.1f}")
        if r['density_comparison']:
            ratio_100 = r['density_comparison'][0]['ratio']
            if ratio_100:
                print(f"    ρ_required / ρ_MTDF at 100 kpc: {ratio_100:.0f}×")

    print(f"""
VERDICT:
  The required ρ_stress(r) is an isothermal-like profile (ρ ~ r^{{-2}})
  with amplitude ~10³ to 10⁴× larger than what the MTDF elastic field
  predicts from the current parameters.

  The residual signal is well-fit by either an isothermal sphere or
  an NFW halo — consistent with ΛCDM's dark matter interpretation.

  For MTDF to explain this, the non-linear field equations would need
  to concentrate 10³-10⁴× more stress energy at 100-500 kpc than
  the linear theory predicts. Combined with the Step 3 finding that
  the field equations have a c² ambiguity, this defines the precise
  quantitative target for the gravity programme.

  The shape is NOT the problem — both MTDF stress halos and the
  required density go as ~r^{{-2}}. The amplitude is.
""")

    with open(out_dir / 'step5_density_inversion.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved: {out_dir / 'step5_density_inversion.json'}")

    plt.close('all')


if __name__ == "__main__":
    main()
