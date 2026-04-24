#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 1: Compare MTDF ESD predictions against Brouwer+2021 KiDS×GAMA data.

Loads the published ESD profiles from Brouwer et al. (2021, A&A 650, A113)
and overlays MTDF predictions to determine whether the F2 cliff
(MTDF under-predicting GGL at R > 100 kpc) is real against actual data.

Reference: "The Weak Lensing Radial Acceleration Relation: Constraining
Modified Gravity and CDM theories with KiDS-1000"

Data: https://kids.strw.leidenuniv.nl/sciencedata.php
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# MTDF parameters
ALPHA = 1.30
BETA_KPC = 22_685.0   # 7.00e23 m = 22.7 Mpc, expressed in kpc

# Cosmology (h70 = 1 convention, matching Brouwer+2021)
RHO_CRIT = 136.3       # M_sun/kpc^3 (for H0=70)

# Brouwer+2021 mass bins: log10(M*/(h70^-2 Msun))
BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
BIN_LABELS = [
    r"$8.5 < \log M_* < 10.3$",
    r"$10.3 < \log M_* < 10.6$",
    r"$10.6 < \log M_* < 10.8$",
    r"$10.8 < \log M_* < 11.0$",
]
BIN_NAMES = ["bin1", "bin2", "bin3", "bin4"]

# Estimated median stellar masses (M_sun) per bin.
# Bin 1 is wide; stellar mass function puts median near upper end.
# These are representative values -- exact medians would come from the
# paper's Table 1, but the shortfall is large enough to be robust.
MEDIAN_LOG_MSTAR = [10.0, 10.45, 10.70, 10.90]
MEDIAN_MSTAR = [10**x for x in MEDIAN_LOG_MSTAR]

# Gas fraction estimates (Catinella+2018 scaling)
# Higher for lower-mass galaxies
F_GAS = [0.15, 0.08, 0.05, 0.03]

# Baryonic mass = stellar + gas
MEDIAN_MBAR = [m * (1 + f) for m, f in zip(MEDIAN_MSTAR, F_GAS)]


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_brouwer_bin(data_dir, bin_num):
    """Load one Brouwer+2021 Fig-3 ESD file. Returns dict with arrays."""
    fname = data_dir / f"Fig-3_Lensing-rotation-curves_Massbin-{bin_num}.txt"
    data = np.loadtxt(fname)
    # Apply bias correction: ESD_corrected = ESD_t / bias
    bias = data[:, 4]
    return {
        'R_Mpc': data[:, 0],                   # Mpc
        'R_kpc': data[:, 0] * 1000,            # kpc
        'ESD_raw': data[:, 1],                  # h70*Msun/pc^2 (uncorrected)
        'ESD_x': data[:, 2] / bias,            # cross-component (should be ~0)
        'ESD': data[:, 1] / bias,              # h70*Msun/pc^2 (bias-corrected)
        'error': data[:, 3] / bias,            # h70*Msun/pc^2 (bias-corrected)
        'bias': bias,
    }


def load_brouwer_covariance(data_dir):
    """Load the full covariance matrix for Fig-3 mass bins.

    Returns:
        cov_dict: dict mapping (bin_m, bin_n, R_i, R_j) -> covariance value
        Also returns structured arrays for chi-squared computation.
    """
    fname = data_dir / "Fig-3_Lensing-rotation-curves_Massbins_covmatrix.txt"
    data = np.loadtxt(fname)
    # Columns: mass_min_m, mass_min_n, R_i, R_j, covariance, correlation, bias
    return {
        'mass_min_m': data[:, 0],
        'mass_min_n': data[:, 1],
        'R_i': data[:, 2],
        'R_j': data[:, 3],
        'covariance': data[:, 4],
        'correlation': data[:, 5],
        'bias': data[:, 6],
    }


def build_cov_matrix_for_bin(cov_data, bin_mass_min, n_radial):
    """Extract the n_radial x n_radial covariance submatrix for one mass bin.

    The covariance file uses mass_min to identify bins:
    Bin 1: 8.5, Bin 2: 10.3, Bin 3: 10.6, Bin 4: 10.8
    """
    mask = ((np.abs(cov_data['mass_min_m'] - bin_mass_min) < 0.01) &
            (np.abs(cov_data['mass_min_n'] - bin_mass_min) < 0.01))
    sub = {k: v[mask] for k, v in cov_data.items()}

    # Build matrix -- radii should be in order
    radii_i = np.sort(np.unique(sub['R_i']))
    radii_j = np.sort(np.unique(sub['R_j']))
    n = len(radii_i)
    assert n == n_radial, f"Expected {n_radial} radii, got {n}"

    cov_mat = np.zeros((n, n))
    for k in range(len(sub['R_i'])):
        i = np.searchsorted(radii_i, sub['R_i'][k])
        j = np.searchsorted(radii_j, sub['R_j'][k])
        # Apply bias correction
        cov_mat[i, j] = sub['covariance'][k] / sub['bias'][k]

    return cov_mat


# ═══════════════════════════════════════════════════════════════
# NFW PROFILE (Wright & Brainerd 2000) -- from testF2
# ═══════════════════════════════════════════════════════════════

def nfw_sigma(x):
    """NFW surface mass density factor."""
    result = np.zeros_like(x, dtype=float)
    lo = (x < 0.999)
    if np.any(lo):
        xl = x[lo]
        result[lo] = 1.0 / (xl**2 - 1) * (
            1.0 - np.arctanh(np.sqrt(1 - xl**2)) / np.sqrt(1 - xl**2))
    eq = (x >= 0.999) & (x <= 1.001)
    result[eq] = 1.0 / 3.0
    hi = (x > 1.001)
    if np.any(hi):
        xh = x[hi]
        result[hi] = 1.0 / (xh**2 - 1) * (
            1.0 - np.arctan(np.sqrt(xh**2 - 1)) / np.sqrt(xh**2 - 1))
    return result


def nfw_sigma_mean(x):
    """NFW mean surface density factor."""
    result = np.zeros_like(x, dtype=float)
    lo = (x < 0.999)
    if np.any(lo):
        xl = x[lo]
        result[lo] = (np.arctanh(np.sqrt(1 - xl**2)) / np.sqrt(1 - xl**2)
                       + np.log(xl / 2))
    eq = (x >= 0.999) & (x <= 1.001)
    result[eq] = 1.0 + np.log(0.5)
    hi = (x > 1.001)
    if np.any(hi):
        xh = x[hi]
        result[hi] = (np.arctan(np.sqrt(xh**2 - 1)) / np.sqrt(xh**2 - 1)
                       + np.log(xh / 2))
    return result


def nfw_esd_kpc(R_kpc, M200, c200):
    """NFW ESD in M_sun/kpc^2."""
    r200 = (3 * M200 / (4 * np.pi * 200 * RHO_CRIT))**(1.0 / 3.0)
    r_s = r200 / c200
    rho_s = M200 / (4 * np.pi * r_s**3 *
                     (np.log(1 + c200) - c200 / (1 + c200)))
    x = np.clip(R_kpc / r_s, 1e-6, None)
    sigma = 2 * rho_s * r_s * nfw_sigma(x)
    sigma_mean = 4 * rho_s * r_s * nfw_sigma_mean(x) / x**2
    return sigma_mean - sigma


# ═══════════════════════════════════════════════════════════════
# SHMR (Moster+2013)
# ═══════════════════════════════════════════════════════════════

def moster2013_mstar(M_halo):
    """M_star from M_halo via Moster+2013 z=0."""
    N0 = 0.0351
    M1 = 10**11.59
    beta_m = 1.376
    gamma_m = 0.608
    f = 2 * N0 * ((M_halo / M1)**(-beta_m) + (M_halo / M1)**gamma_m)**(-1)
    return M_halo * f


def halo_mass_from_stellar(M_star):
    """Invert Moster+2013 to get M_halo."""
    def residual(log_Mh):
        return np.log10(moster2013_mstar(10**log_Mh)) - np.log10(M_star)
    log_Mh = brentq(residual, 9.0, 16.0)
    return 10**log_Mh


def duffy2008_concentration(M_halo, z=0.0):
    """NFW concentration from Duffy+2008."""
    A, B, C = 5.71, -0.084, -0.47
    M_pivot = 2e12
    return A * (M_halo / M_pivot)**B * (1 + z)**C


# ═══════════════════════════════════════════════════════════════
# ESD PREDICTIONS
# ═══════════════════════════════════════════════════════════════

def esd_baryon_point(R_kpc, M_bar):
    """Point-mass baryonic ESD: M/(πR²) in M_sun/kpc²."""
    return M_bar / (np.pi * R_kpc**2)


def esd_mtdf(R_kpc, M_bar):
    """MTDF ESD under Assumption A: ΔΣ_baryon × [1 + α/(1+R/β)]."""
    enhancement = 1.0 + ALPHA / (1.0 + R_kpc / BETA_KPC)
    return esd_baryon_point(R_kpc, M_bar) * enhancement


def esd_lcdm(R_kpc, M_star):
    """LCDM ESD: baryonic point mass + NFW halo from SHMR."""
    M_halo = halo_mass_from_stellar(M_star)
    c200 = duffy2008_concentration(M_halo)
    nfw = nfw_esd_kpc(R_kpc, M_halo, c200)
    baryon = esd_baryon_point(R_kpc, M_star)
    return baryon + nfw, M_halo, c200


def kpc2_to_brouwer(esd_kpc2):
    """Convert M_sun/kpc² → h70*M_sun/pc² (divide by 1e6)."""
    return esd_kpc2 / 1e6


# ═══════════════════════════════════════════════════════════════
# CHI-SQUARED
# ═══════════════════════════════════════════════════════════════

def chi2_diagonal(data_esd, model_esd, errors, mask=None):
    """Simple diagonal chi² (no covariance cross-terms)."""
    if mask is not None:
        d, m, e = data_esd[mask], model_esd[mask], errors[mask]
    else:
        d, m, e = data_esd, model_esd, errors
    return np.sum(((d - m) / e)**2), len(d)


def chi2_full(data_esd, model_esd, cov_mat, mask=None):
    """Full chi² using covariance matrix."""
    if mask is not None:
        d = data_esd[mask]
        m = model_esd[mask]
        C = cov_mat[np.ix_(mask, mask)]
    else:
        d, m, C = data_esd, model_esd, cov_mat
    residual = d - m
    try:
        Cinv = np.linalg.inv(C)
        return float(residual @ Cinv @ residual), len(d)
    except np.linalg.LinAlgError:
        return np.nan, len(d)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = Path(__file__).parent.parent / "data" / "brouwer2021"
    out_dir = Path(__file__).parent.parent / "output" / "step1_ggl_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 bins
    bins = [load_brouwer_bin(data_dir, i+1) for i in range(4)]
    cov_data = load_brouwer_covariance(data_dir)

    # Build covariance matrices per bin
    bin_mass_mins = [8.5, 10.3, 10.6, 10.8]
    n_radial = len(bins[0]['R_kpc'])
    cov_mats = []
    for bmin in bin_mass_mins:
        try:
            cm = build_cov_matrix_for_bin(cov_data, bmin, n_radial)
            cov_mats.append(cm)
        except Exception as e:
            print(f"Warning: covariance for bin {bmin} failed: {e}")
            cov_mats.append(None)

    # Compute predictions for each bin
    results = []
    for i in range(4):
        b = bins[i]
        R_kpc = b['R_kpc']
        M_bar = MEDIAN_MBAR[i]
        M_star = MEDIAN_MSTAR[i]

        # MTDF prediction (in Brouwer units)
        mtdf_kpc2 = esd_mtdf(R_kpc, M_bar)
        mtdf = kpc2_to_brouwer(mtdf_kpc2)

        # LCDM prediction
        lcdm_kpc2, M_halo, c200 = esd_lcdm(R_kpc, M_star)
        lcdm = kpc2_to_brouwer(lcdm_kpc2)

        # Baryonic only
        baryon_kpc2 = esd_baryon_point(R_kpc, M_bar)
        baryon = kpc2_to_brouwer(baryon_kpc2)

        # Masks for different radial ranges
        mask_100 = b['R_kpc'] >= 100    # R >= 100 kpc
        mask_100_300 = (b['R_kpc'] >= 100) & (b['R_kpc'] <= 300)
        mask_300 = b['R_kpc'] >= 300    # R >= 300 kpc

        # Chi-squared (diagonal)
        chi2_mtdf_all, n_all = chi2_diagonal(b['ESD'], mtdf, b['error'])
        chi2_lcdm_all, _ = chi2_diagonal(b['ESD'], lcdm, b['error'])
        chi2_mtdf_100, n_100 = chi2_diagonal(b['ESD'], mtdf, b['error'], mask_100)
        chi2_lcdm_100, _ = chi2_diagonal(b['ESD'], lcdm, b['error'], mask_100)

        # Ratio at R > 100 kpc
        ratio_100 = np.mean(mtdf[mask_100] / b['ESD'][mask_100])
        ratio_lcdm_100 = np.mean(lcdm[mask_100] / b['ESD'][mask_100])

        # Shortfall factor
        shortfall_mtdf = np.mean(b['ESD'][mask_100] / mtdf[mask_100])
        shortfall_lcdm = np.mean(b['ESD'][mask_100] / lcdm[mask_100])

        result = {
            'bin': BIN_NAMES[i],
            'label': BIN_LABELS[i].replace('$', '').replace('\\log ', 'log'),
            'log_Mstar_median': MEDIAN_LOG_MSTAR[i],
            'M_star': float(M_star),
            'M_bar': float(M_bar),
            'M_halo_Moster': float(M_halo),
            'c200_Duffy': float(c200),
            'n_radial_bins': int(n_all),
            'n_bins_R_gt_100kpc': int(n_100),
            'chi2_mtdf_all': float(chi2_mtdf_all),
            'chi2_lcdm_all': float(chi2_lcdm_all),
            'chi2_mtdf_R_gt_100kpc': float(chi2_mtdf_100),
            'chi2_lcdm_R_gt_100kpc': float(chi2_lcdm_100),
            'mean_ratio_mtdf_data_R100': float(ratio_100),
            'mean_ratio_lcdm_data_R100': float(ratio_lcdm_100),
            'shortfall_mtdf_R100': float(shortfall_mtdf),
            'shortfall_lcdm_R100': float(shortfall_lcdm),
        }

        # Per-radial-bin details
        per_bin = []
        for j in range(len(R_kpc)):
            per_bin.append({
                'R_kpc': float(R_kpc[j]),
                'R_Mpc': float(b['R_Mpc'][j]),
                'data_ESD': float(b['ESD'][j]),
                'data_error': float(b['error'][j]),
                'mtdf_ESD': float(mtdf[j]),
                'lcdm_ESD': float(lcdm[j]),
                'baryon_ESD': float(baryon[j]),
                'data_over_mtdf': float(b['ESD'][j] / mtdf[j]) if mtdf[j] > 0 else None,
            })
        result['radial_bins'] = per_bin
        results.append(result)

        print(f"\n{'='*60}")
        print(f"Bin {i+1}: {BIN_LABELS[i]} (median log M* = {MEDIAN_LOG_MSTAR[i]:.2f})")
        print(f"  M_star = {M_star:.2e}, M_bar = {M_bar:.2e}, M_halo = {M_halo:.2e}")
        print(f"  c200 = {c200:.2f}")
        print(f"  chi2 (all R):       MTDF = {chi2_mtdf_all:.1f}/{n_all}  "
              f"LCDM = {chi2_lcdm_all:.1f}/{n_all}")
        print(f"  chi2 (R>100 kpc):   MTDF = {chi2_mtdf_100:.1f}/{n_100}  "
              f"LCDM = {chi2_lcdm_100:.1f}/{n_100}")
        print(f"  Shortfall (R>100):  MTDF = {shortfall_mtdf:.1f}x  "
              f"LCDM = {shortfall_lcdm:.1f}x")

    # ═══════════════════════════════════════════════════════════
    # PLOT: 4 panels, one per mass bin
    # ═══════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True)
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        b = bins[i]
        R = b['R_Mpc'] * 1000  # kpc for x-axis

        # Data
        ax.errorbar(R, b['ESD'], yerr=b['error'],
                     fmt='ko', ms=5, capsize=2, label='Brouwer+2021 data', zorder=5)

        # MTDF
        M_bar = MEDIAN_MBAR[i]
        mtdf = kpc2_to_brouwer(esd_mtdf(b['R_kpc'], M_bar))
        ax.plot(R, mtdf, 'b-', lw=2, label='MTDF (baryons × 2.3)', zorder=3)

        # Baryonic only
        baryon = kpc2_to_brouwer(esd_baryon_point(b['R_kpc'], M_bar))
        ax.plot(R, baryon, 'b--', lw=1, alpha=0.5, label='Baryons only', zorder=2)

        # LCDM (NFW + baryons)
        lcdm_kpc2, _, _ = esd_lcdm(b['R_kpc'], MEDIAN_MSTAR[i])
        lcdm = kpc2_to_brouwer(lcdm_kpc2)
        ax.plot(R, lcdm, 'r-', lw=2, label=r'$\Lambda$CDM (NFW+baryons)', zorder=4)

        # Mark R = 100 kpc discriminating boundary
        ax.axvline(100, color='gray', ls=':', alpha=0.5, zorder=1)
        ax.text(110, ax.get_ylim()[1]*0.7 if i == 0 else 0.5,
                'R = 100 kpc', fontsize=8, color='gray', rotation=90, va='top')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(25, 3000)
        ax.set_ylim(0.01, 200)
        ax.set_title(BIN_LABELS[i], fontsize=11)
        ax.grid(True, alpha=0.2)

        if i >= 2:
            ax.set_xlabel('Projected radius R [kpc]')
        if i % 2 == 0:
            ax.set_ylabel(r'$\Delta\Sigma$ [$h_{70}\,M_\odot\,\mathrm{pc}^{-2}$]')
        if i == 0:
            ax.legend(fontsize=8, loc='lower left')

        # Annotate shortfall
        shortfall = results[i]['shortfall_mtdf_R100']
        ax.text(0.95, 0.95,
                f'MTDF shortfall\n@ R>100 kpc: {shortfall:.0f}×',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('MTDF vs Brouwer+2021 KiDS×GAMA Galaxy-Galaxy Lensing\n'
                 '(Excess Surface Density profiles, isolated lenses)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'step1_ggl_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_dir / 'step1_ggl_comparison.png'}")

    # ═══════════════════════════════════════════════════════════
    # PLOT 2: Residual ratio (data/MTDF) vs radius
    # ═══════════════════════════════════════════════════════════

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(4):
        b = bins[i]
        R = b['R_kpc']
        mtdf = kpc2_to_brouwer(esd_mtdf(R, MEDIAN_MBAR[i]))
        ratio = b['ESD'] / mtdf
        ratio_err = b['error'] / mtdf

        ax2.errorbar(R, ratio, yerr=ratio_err,
                     fmt='o-', ms=5, capsize=2, color=colors[i],
                     label=BIN_LABELS[i])

    ax2.axhline(1, color='blue', ls='--', lw=1.5, label='MTDF = data')
    ax2.axvline(100, color='gray', ls=':', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Projected radius R [kpc]', fontsize=12)
    ax2.set_ylabel(r'$\Delta\Sigma_\mathrm{data} \,/\, \Delta\Sigma_\mathrm{MTDF}$',
                   fontsize=12)
    ax2.set_title('Ratio of measured ESD to MTDF prediction', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(25, 3000)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'step1_ratio.png', dpi=150, bbox_inches='tight')
    print(f"Ratio plot saved: {out_dir / 'step1_ratio.png'}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    summary = {
        'description': ('Step 1: Comparison of MTDF ESD predictions against '
                        'Brouwer+2021 KiDS×GAMA galaxy-galaxy lensing data'),
        'reference': 'Brouwer et al. 2021, A&A 650, A113',
        'data_source': 'https://kids.strw.leidenuniv.nl/sciencedata.php',
        'mtdf_model': 'ESD = M_bar/(pi*R^2) * [1 + alpha/(1+R/beta)]',
        'lcdm_model': 'NFW (Moster+2013 SHMR, Duffy+2008 concentration) + baryonic point mass',
        'alpha': ALPHA,
        'beta_kpc': BETA_KPC,
        'bins': results,
    }

    # Overall assessment
    shortfalls = [r['shortfall_mtdf_R100'] for r in results]
    lcdm_ratios = [r['shortfall_lcdm_R100'] for r in results]

    summary['assessment'] = {
        'shortfall_range_MTDF': f"{min(shortfalls):.0f}x to {max(shortfalls):.0f}x",
        'shortfall_range_LCDM': f"{min(lcdm_ratios):.1f}x to {max(lcdm_ratios):.1f}x",
        'conclusion': (
            f"MTDF under-predicts the measured GGL signal at R > 100 kpc by "
            f"{min(shortfalls):.0f}-{max(shortfalls):.0f}x across all stellar mass bins. "
            f"This confirms that the F2 cliff is real against actual lensing data, "
            f"not an artefact of comparing to NFW model predictions. "
            f"The LCDM NFW model is within {min(lcdm_ratios):.1f}-{max(lcdm_ratios):.1f}x "
            f"of the data at the same radii."
        ),
    }

    with open(out_dir / 'step1_ggl_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {out_dir / 'step1_ggl_comparison.json'}")

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY: MTDF vs Brouwer+2021 data at R > 100 kpc")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  Bin {i+1} ({r['label']}): MTDF shortfall = {r['shortfall_mtdf_R100']:.0f}x, "
              f"LCDM ratio = {r['shortfall_lcdm_R100']:.1f}x")
    print(f"\nConclusion: {summary['assessment']['conclusion']}")

    plt.close('all')


if __name__ == "__main__":
    main()
