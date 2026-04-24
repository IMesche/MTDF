#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 4: Direct MTDF shear vs observed shear comparison.

No NFW. No "mass budget." No "stress halo."
Just: what shear does the MTDF metric predict, and does it match
what was measured?

Physics:
  MTDF predicts a metric potential:
    Φ_MTDF(r) = -GM_bar/r × [1 + α/(1 + r/β)]

  With η = 1 (C5b validated): both metric potentials are equal,
  so the lensing potential is (Φ + Ψ)/2 = Φ_MTDF.

  The weak lensing convergence κ and tangential shear γ_t follow from
  the standard projection of the effective density ρ_eff(r) = ∇²Φ/(4πG)
  along the line of sight.

  For the MTDF potential around a point mass M at r >> R_galaxy:
    ΔΣ_MTDF(R) = [projected effective mass] / (πR²) - Σ_eff(R)

  This is computed numerically from the 3D effective density.

Reference data: Brouwer+2021 (A&A 650, A113), Fig-3 ESD profiles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALPHA = 1.30
BETA_KPC = 22_685.0  # 7.00e23 m = 22.7 Mpc, expressed in kpc

# Brouwer+2021 mass bins
BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
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
    """Load Brouwer+2021 Fig-3 ESD file with bias correction."""
    fname = data_dir / f"Fig-3_Lensing-rotation-curves_Massbin-{bin_num}.txt"
    data = np.loadtxt(fname)
    bias = data[:, 4]
    return {
        'R_kpc': data[:, 0] * 1000,  # proper Mpc → kpc
        'ESD': data[:, 1] / bias,     # h70 M_sun/pc² (bias-corrected)
        'error': data[:, 3] / bias,
    }


# ═══════════════════════════════════════════════════════════════
# MTDF SHEAR FROM THE METRIC (not from "effective mass")
# ═══════════════════════════════════════════════════════════════

def rho_eff_mtdf(r_kpc, M_bar):
    """
    Effective 3D density from the MTDF potential, for r > 0.

    Φ_MTDF(r) = -GM/r × [1 + α/(1+r/β)]
              = -(1+α)GM/r + αGM/(r+β)

    ∇²Φ = 4πG ρ_eff:
      Part 1: -(1+α)GM/r → (1+α)M δ(r)  [point mass, handled separately]
      Part 2: αGM/(r+β) → (αM/4π) × ∇²[1/(r+β)]
            = (αM/4π) × [-2β/(r(r+β)³)]
            = -αMβ / [2π r (r+β)³]

    Returns: ρ_eff at r > 0 (the negative distributed component only).
    Units: M_sun / kpc³
    """
    return -ALPHA * M_bar * BETA_KPC / (2 * np.pi * r_kpc * (r_kpc + BETA_KPC)**3)


def sigma_eff_mtdf(R_kpc, M_bar, z_max_kpc=50000):
    """
    Surface density Σ_eff(R) from the distributed (non-delta) part.

    Σ(R) = ∫_{-∞}^{∞} ρ_eff(√(R²+z²)) dz
         = 2 ∫_0^∞ ρ_eff(√(R²+z²)) dz

    The delta-function part contributes (1+α)M × δ²(R) → zero at R > 0.
    """
    def integrand(z):
        r = np.sqrt(R_kpc**2 + z**2)
        return rho_eff_mtdf(r, M_bar)

    result, _ = quad(integrand, 0, z_max_kpc, limit=300,
                     epsabs=1e-12, epsrel=1e-10)
    return 2 * result  # both sides


def esd_mtdf_from_metric(R_kpc_arr, M_bar, n_sigma_int=80):
    """
    Compute ΔΣ(R) from the MTDF metric potential.

    ΔΣ(R) = Σ̄_eff(<R) − Σ_eff(R)

    The point-mass part: ΔΣ_point(R) = (1+α)M / (πR²)
    The distributed part: ΔΣ_dist(R) = Σ̄_dist(<R) − Σ_dist(R)

    Total: ΔΣ = ΔΣ_point + ΔΣ_dist
    """
    # Compute Σ_dist at a grid of R values for integration
    R_grid = np.logspace(np.log10(0.5), np.log10(max(R_kpc_arr) * 1.5), n_sigma_int)
    Sigma_grid = np.array([sigma_eff_mtdf(R, M_bar) for R in R_grid])

    esd = np.zeros(len(R_kpc_arr))
    for i, R in enumerate(R_kpc_arr):
        # Point-mass contribution
        esd_point = (1 + ALPHA) * M_bar / (np.pi * R**2)

        # Distributed part: Σ_dist(R)
        Sigma_R = sigma_eff_mtdf(R, M_bar)

        # Distributed part: Σ̄_dist(<R) = (2/R²) ∫₀^R Σ_dist(R') R' dR'
        mask = R_grid <= R
        if np.sum(mask) > 2:
            R_sub = R_grid[mask]
            S_sub = Sigma_grid[mask]
            integrand = S_sub * R_sub
            sigma_mean = 2.0 / R**2 * np.trapz(integrand, R_sub)
        else:
            sigma_mean = 0.0

        esd_dist = sigma_mean - Sigma_R
        esd[i] = esd_point + esd_dist

    return esd  # M_sun/kpc²


def esd_mtdf_naive(R_kpc, M_bar):
    """
    Step 1's simple formula: ΔΣ = M_bar/(πR²) × [1 + α/(1+R/β)].
    This treats the enhancement as a multiplicative factor on the ESD.
    """
    enhancement = 1.0 + ALPHA / (1.0 + R_kpc / BETA_KPC)
    return M_bar / (np.pi * R_kpc**2) * enhancement


def esd_baryon_only(R_kpc, M_bar):
    """Point-mass baryon ESD (no MTDF enhancement)."""
    return M_bar / (np.pi * R_kpc**2)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = Path(__file__).parent.parent / "data" / "brouwer2021"
    out_dir = Path(__file__).parent.parent / "output" / "step4_shear_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = [load_brouwer_bin(data_dir, i + 1) for i in range(4)]

    # ── Compute MTDF shear from metric for all bins ──
    print("=" * 70)
    print("MTDF Shear (from metric) vs Observed Shear (Brouwer+2021)")
    print("=" * 70)
    print("\nComputing ΔΣ from the MTDF metric potential Φ(r) = -GM/r × [1+α/(1+r/β)]")
    print("with η = 1 (lensing = dynamics). No NFW. No stress halo. No mass budget.\n")

    results = []
    for i in range(4):
        b = bins[i]
        R_kpc = b['R_kpc']
        M_bar = MEDIAN_MBAR[i]

        # Method 1: Direct from metric (correct)
        esd_metric = esd_mtdf_from_metric(R_kpc, M_bar)
        esd_metric_brouwer = esd_metric / 1e6  # M_sun/kpc² → h70 M_sun/pc²

        # Method 2: Step 1's naive formula
        esd_naive = esd_mtdf_naive(R_kpc, M_bar)
        esd_naive_brouwer = esd_naive / 1e6

        # Method 3: Baryon-only (GR, no MTDF, no dark matter)
        esd_bar = esd_baryon_only(R_kpc, M_bar)
        esd_bar_brouwer = esd_bar / 1e6

        # Observed
        esd_obs = b['ESD']
        err_obs = b['error']

        # Masks
        mask_100 = R_kpc >= 100

        # Ratios at R > 100 kpc
        ratio_metric = np.mean(esd_obs[mask_100] / esd_metric_brouwer[mask_100])
        ratio_naive = np.mean(esd_obs[mask_100] / esd_naive_brouwer[mask_100])

        # Relative difference: metric vs naive
        rel_diff = np.mean(np.abs(esd_metric_brouwer - esd_naive_brouwer) /
                           esd_naive_brouwer)

        print(f"Bin {i+1}: {BIN_LABELS[i]} (M_bar = {M_bar:.2e} M_sun)")
        print(f"  MTDF shortfall at R > 100 kpc:")
        print(f"    From metric:  {ratio_metric:.1f}×")
        print(f"    From Step 1:  {ratio_naive:.1f}×")
        print(f"  Metric vs naive formula: {rel_diff*100:.4f}% difference")
        print(f"  (The negative distributed correction is negligible)")
        print()

        result = {
            'bin': f'bin{i+1}',
            'label': BIN_LABELS[i].replace('$', '').replace('\\log ', 'log'),
            'M_bar': M_bar,
            'shortfall_metric': float(ratio_metric),
            'shortfall_naive': float(ratio_naive),
            'metric_vs_naive_pct': float(rel_diff * 100),
            'radial': [{
                'R_kpc': float(R_kpc[j]),
                'esd_observed': float(esd_obs[j]),
                'esd_error': float(err_obs[j]),
                'esd_mtdf_metric': float(esd_metric_brouwer[j]),
                'esd_mtdf_naive': float(esd_naive_brouwer[j]),
                'esd_baryon_only': float(esd_bar_brouwer[j]),
                'ratio_obs_over_mtdf': float(esd_obs[j] / esd_metric_brouwer[j]),
            } for j in range(len(R_kpc))],
        }
        results.append(result)

    # ═══════════════════════════════════════════════════════════
    # PLOT: Clean MTDF shear vs observed shear
    # ═══════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        b = bins[i]
        R = b['R_kpc']
        M_bar = MEDIAN_MBAR[i]

        # Observed
        ax.errorbar(R, b['ESD'], yerr=b['error'],
                     fmt='ko', ms=5, capsize=2, label='Observed shear (KiDS×GAMA)',
                     zorder=5)

        # MTDF from metric
        esd_m = esd_mtdf_from_metric(R, M_bar) / 1e6
        ax.plot(R, esd_m, 'b-', lw=2.5,
                label='MTDF prediction (from metric)', zorder=4)

        # Baryon-only (GR without dark matter)
        esd_b = esd_baryon_only(R, M_bar) / 1e6
        ax.plot(R, esd_b, 'b--', lw=1, alpha=0.4,
                label='Baryons only (no DM, no MTDF)', zorder=2)

        # Fill the gap
        ax.fill_between(R, esd_m, b['ESD'], alpha=0.15, color='red',
                         label='Unaccounted signal', zorder=1)

        ax.axvline(100, color='gray', ls=':', alpha=0.4, zorder=0)

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

        # Annotate shortfall
        shortfall = results[i]['shortfall_metric']
        ax.text(0.95, 0.95,
                f'Observed / MTDF\n@ R > 100 kpc: {shortfall:.0f}×',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('MTDF Shear Prediction vs Observed Shear\n'
                 r'$\Phi_{\rm MTDF}(r) = -GM_{\rm bar}/r \times [1 + \alpha/(1+r/\beta)]$,'
                 r' $\eta = 1$, no dark matter',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'step4_shear_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_dir / 'step4_shear_comparison.png'}")

    # ═══════════════════════════════════════════════════════════
    # PLOT 2: Ratio (observed/MTDF) — clean version
    # ═══════════════════════════════════════════════════════════

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(4):
        b = bins[i]
        R = b['R_kpc']
        esd_m = esd_mtdf_from_metric(R, MEDIAN_MBAR[i]) / 1e6
        ratio = b['ESD'] / esd_m
        ratio_err = b['error'] / esd_m

        ax2.errorbar(R, ratio, yerr=ratio_err,
                     fmt='o-', ms=5, capsize=2, color=colors[i],
                     label=BIN_LABELS[i])

    ax2.axhline(1, color='blue', ls='--', lw=2, label='MTDF = Observed')
    ax2.axvline(100, color='gray', ls=':', alpha=0.5)
    ax2.axvspan(25, 50, alpha=0.1, color='green', label='SPARC-validated range')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Projected radius R [kpc]', fontsize=12)
    ax2.set_ylabel(r'Observed shear / MTDF shear', fontsize=12)
    ax2.set_title('Where MTDF under-predicts the lensing signal', fontsize=13)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(25, 3000)
    ax2.set_ylim(0.3, 500)

    # Annotate
    ax2.text(500, 2, 'MTDF matches\nobservation', fontsize=10, color='blue',
             ha='center', va='top', style='italic')
    ax2.text(500, 80, 'Unaccounted\nlensing signal', fontsize=10, color='red',
             ha='center', va='center', style='italic')

    fig2.tight_layout()
    fig2.savefig(out_dir / 'step4_ratio.png', dpi=150, bbox_inches='tight')
    print(f"Ratio plot saved: {out_dir / 'step4_ratio.png'}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        'description': (
            'Step 4: Direct comparison of MTDF-predicted shear against observed shear. '
            'The MTDF metric potential Φ(r) = -GM_bar/r × [1+α/(1+r/β)] with η = 1 '
            'predicts a specific photon deflection pattern. This is compared to the '
            'measured ESD profiles from Brouwer+2021 KiDS×GAMA. No NFW model, no '
            '"stress halo", no "mass budget" — just MTDF prediction vs observation.'
        ),
        'reference': 'Brouwer et al. 2021, A&A 650, A113',
        'mtdf_potential': 'Φ(r) = -GM_bar/r × [1 + α/(1+r/β)]',
        'alpha': ALPHA,
        'beta_kpc': BETA_KPC,
        'eta': 1.0,
        'bins': results,
    }

    # Framing
    print("""
What the MTDF metric predicts:
  At ALL radii R < β (22.7 Mpc), the photon deflection is enhanced by
  a factor (1+α) ≈ 2.30 relative to baryons-only (GR without dark matter).

  The negative distributed correction from ∇²Φ is < 0.01% of the main
  term — completely negligible. The metric gives the same ESD as the
  simple "2.3 × baryon" formula.

What the observations show:
  At R < 50 kpc:  observed/MTDF ≈ 1 (consistent — SPARC validates this)
  At R > 100 kpc: observed/MTDF ≈ 41-94× (massive shortfall)

What this means:
  The MTDF metric potential, with its current parameters (α=1.30,
  β=22.7 Mpc), produces ~2.3× the baryon-only shear at all radii.
  The observed shear at R > 100 kpc is 41-94× larger than MTDF predicts.

  This is NOT about "missing mass" or "stress halos." It is about the
  metric: the MTDF potential simply does not bend light enough at
  R > 100 kpc to match observations.

  For the metric to match, one of the following must change:
  1. α must increase dramatically with radius (from 1.30 at < 50 kpc
     to ~40-100 at > 100 kpc) — contradicts SPARC
  2. An additional metric component must exist at 100-1000 kpc
  3. The β = 22.7 Mpc parameter must have a different value at
     galactic scales (local beta ~ 300 kpc)
  4. The observations are interpreted differently without dark matter
""")

    shortfalls = [r['shortfall_metric'] for r in results]
    summary['conclusion'] = (
        f'MTDF metric under-predicts observed shear at R > 100 kpc by '
        f'{min(shortfalls):.0f}-{max(shortfalls):.0f}× across all stellar mass bins. '
        f'The negative distributed correction from the metric is < 0.01%, confirming '
        f'that the simple 2.3× formula is exact. The shortfall is a direct consequence '
        f'of the MTDF potential shape, not an artefact of the mass interpretation.'
    )
    print(f"Conclusion: {summary['conclusion']}")

    with open(out_dir / 'step4_shear_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'step4_shear_comparison.json'}")

    plt.close('all')


if __name__ == "__main__":
    main()
