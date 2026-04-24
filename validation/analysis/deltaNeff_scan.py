#!/usr/bin/env python3
"""
ΔN_eff Scan for MTDF Proxy in CLASS

Scans ΔN_eff at fixed H₀ = 70 km/s/Mpc to find the value that minimizes
χ²_TT relative to Planck baseline, identifying the best proxy for MTDF
early field energy.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from datetime import datetime

try:
    from classy import Class
except ImportError:
    raise ImportError("classy not installed. Run: pip install classy")

# Constants
PLANCK_LMAX = 2500
TCMB = 2.7255  # K
Z_STAR = 1089.80

# Chi^2 parameters
CHI2_LMIN = 30
CHI2_LMAX = 2000
CHI2_NBINS = 50

# Fixed parameters (Planck 2018 physical densities)
FIXED_PARAMS = {
    'omega_b': 0.02237,      # ω_b h² fixed
    'omega_cdm': 0.1200,     # ω_cdm h² fixed (ω_m h² = ω_b + ω_cdm + ω_ν)
    'tau_reio': 0.0544,
    'n_s': 0.9649,
    'ln10^{10}A_s': 3.044,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
}

# Standard N_ur (without extra radiation)
N_UR_STANDARD = 2.0328

# ΔN_eff values to scan
DELTA_NEFF_VALUES = [0.00, 0.10, 0.15, 0.20, 0.22, 0.25, 0.30]


def run_class_for_scan(h, delta_neff):
    """Run CLASS for a given H0 and ΔN_eff."""

    params = FIXED_PARAMS.copy()
    params['h'] = h
    params['N_ur'] = N_UR_STANDARD + delta_neff
    params['output'] = 'tCl,pCl,lCl'
    params['lensing'] = 'yes'
    params['l_max_scalars'] = PLANCK_LMAX
    params['accurate_lensing'] = 1

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    # Extract quantities
    bg = cosmo.get_background()
    thermo = cosmo.get_thermodynamics()

    r_d = cosmo.rs_drag()
    theta_s_100 = 100 * cosmo.theta_s_100()
    H0 = cosmo.h() * 100

    # r_s at recombination
    z_rec = thermo['z_rec'][-1] if 'z_rec' in thermo else Z_STAR
    if 'r_s' in thermo and 'z' in thermo:
        z_thermo = thermo['z']
        r_s_thermo = thermo['r_s']
        idx = np.argmin(np.abs(z_thermo - z_rec))
        r_s_rec = r_s_thermo[idx]
    else:
        r_s_rec = r_d * 0.9819

    # D_M(z*)
    z_bg = bg['z']
    if 'comov. dist.' in bg:
        D_M_bg = bg['comov. dist.']
    elif 'ang.diam.dist.' in bg:
        D_A_bg = bg['ang.diam.dist.']
        D_M_bg = D_A_bg * (1 + z_bg)
    else:
        D_M_bg = None

    if D_M_bg is not None:
        D_M_zstar = np.interp(Z_STAR, z_bg[::-1], D_M_bg[::-1])
    else:
        D_M_zstar = 0.0

    # C_l spectra
    cls = cosmo.lensed_cl(PLANCK_LMAX)
    ell = cls['ell']
    factor = (TCMB * 1e6)**2
    cls_tt = cls['tt'] * factor

    # First peak
    dl_tt = ell * (ell + 1) * cls_tt / (2 * np.pi)
    mask = (ell >= 150) & (ell <= 350)
    first_peak_ell = int(ell[mask][np.argmax(dl_tt[mask])])

    cosmo.struct_cleanup()
    cosmo.empty()

    return {
        'H0': H0,
        'delta_neff': delta_neff,
        'r_d': r_d,
        'r_s_rec': r_s_rec,
        'D_M_zstar': D_M_zstar,
        'theta_s_100': theta_s_100,
        'first_peak_ell': first_peak_ell,
        'ell': ell,
        'cls_tt': cls_tt,
    }


def compute_chi2(model, baseline):
    """Compute χ² between model and baseline."""

    bin_edges = np.linspace(CHI2_LMIN, CHI2_LMAX, CHI2_NBINS + 1)

    def bin_spectrum(ell, cls_tt):
        binned = np.zeros(CHI2_NBINS)
        for i in range(CHI2_NBINS):
            mask = (ell >= bin_edges[i]) & (ell < bin_edges[i + 1])
            if np.any(mask):
                ell_bin = ell[mask]
                cls_bin = cls_tt[mask]
                dl_bin = ell_bin * (ell_bin + 1) * cls_bin / (2 * np.pi)
                binned[i] = np.mean(dl_bin)
        return binned

    model_binned = bin_spectrum(model['ell'], model['cls_tt'])
    baseline_binned = bin_spectrum(baseline['ell'], baseline['cls_tt'])

    sigma = 0.02 * baseline_binned
    sigma = np.maximum(sigma, 1.0)

    residuals = model_binned - baseline_binned
    residuals_percent = 100 * residuals / baseline_binned

    chi2 = np.sum((residuals / sigma)**2)
    chi2_red = chi2 / CHI2_NBINS
    rms_percent = np.sqrt(np.mean(residuals_percent**2))

    return chi2, chi2_red, rms_percent


def main():
    print("=" * 70)
    print("ΔN_eff Scan for MTDF Proxy")
    print("=" * 70)
    print(f"Fixed: H₀ = 70.0 km/s/Mpc")
    print(f"Fixed: ω_b h² = {FIXED_PARAMS['omega_b']}")
    print(f"Fixed: ω_cdm h² = {FIXED_PARAMS['omega_cdm']}")
    print(f"Scanning: ΔN_eff ∈ {DELTA_NEFF_VALUES}")
    print("=" * 70)

    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # First, run Planck baseline for comparison
    print("\nRunning Planck baseline (H₀ = 67.36)...")
    baseline = run_class_for_scan(h=0.6736, delta_neff=0.0)
    baseline_theta_s = baseline['theta_s_100']
    print(f"  Planck baseline: 100×θ_s = {baseline_theta_s:.6f}")

    # Scan ΔN_eff
    print("\nScanning ΔN_eff values...")
    results = []

    for delta_neff in DELTA_NEFF_VALUES:
        print(f"\n  ΔN_eff = {delta_neff:.2f}...")

        model = run_class_for_scan(h=0.70, delta_neff=delta_neff)
        chi2, chi2_red, rms_percent = compute_chi2(model, baseline)

        result = {
            'delta_neff': delta_neff,
            'r_d': model['r_d'],
            'r_s_rec': model['r_s_rec'],
            'D_M_zstar': model['D_M_zstar'],
            'theta_s_100': model['theta_s_100'],
            'l_peak': model['first_peak_ell'],
            'chi2_TT': chi2,
            'chi2_red': chi2_red,
            'rms_percent': rms_percent,
        }
        results.append(result)

        print(f"    r_d = {model['r_d']:.4f} Mpc")
        print(f"    100×θ_s = {model['theta_s_100']:.6f}")
        print(f"    χ²_TT = {chi2:.2f}")

    # Find best-fit ΔN_eff
    chi2_values = [r['chi2_TT'] for r in results]
    best_idx = np.argmin(chi2_values)
    best_delta_neff = results[best_idx]['delta_neff']
    best_chi2 = results[best_idx]['chi2_TT']

    print("\n" + "=" * 70)
    print("SCAN RESULTS")
    print("=" * 70)
    print(f"\nBest-fit ΔN_eff proxy for MTDF: {best_delta_neff:.2f}")
    print(f"Minimum χ²_TT: {best_chi2:.2f}")
    print(f"At ΔN_eff = {best_delta_neff:.2f}:")
    print(f"  r_d = {results[best_idx]['r_d']:.4f} Mpc")
    print(f"  r_s_rec = {results[best_idx]['r_s_rec']:.4f} Mpc")
    print(f"  100×θ_s = {results[best_idx]['theta_s_100']:.6f}")
    print(f"  l_peak = {results[best_idx]['l_peak']}")

    # Print table
    print("\n" + "-" * 90)
    print(f"{'ΔN_eff':>8} {'r_d':>10} {'r_s_rec':>10} {'D_M(z*)':>12} {'100×θ_s':>12} {'l_peak':>7} {'χ²_TT':>10} {'χ²/N':>8} {'RMS%':>8}")
    print("-" * 90)
    for r in results:
        marker = " *" if r['delta_neff'] == best_delta_neff else ""
        print(f"{r['delta_neff']:>8.2f} {r['r_d']:>10.4f} {r['r_s_rec']:>10.4f} "
              f"{r['D_M_zstar']:>12.2f} {r['theta_s_100']:>12.6f} {r['l_peak']:>7d} "
              f"{r['chi2_TT']:>10.2f} {r['chi2_red']:>8.4f} {r['rms_percent']:>8.4f}{marker}")
    print("-" * 90)
    print("* = minimum χ²_TT")

    # Save CSV
    csv_file = output_dir / 'class_results_scan.csv'
    with open(csv_file, 'w', newline='') as f:
        # Header comment
        f.write(f"# MTDF ΔN_eff Scan Results\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Best-fit ΔN_eff proxy for MTDF: {best_delta_neff:.2f}\n")
        f.write(f"# Minimum χ²_TT: {best_chi2:.2f}\n")
        f.write(f"# Fixed: H0 = 70.0 km/s/Mpc\n")
        f.write(f"# Fixed: omega_b_h2 = {FIXED_PARAMS['omega_b']}\n")
        f.write(f"# Fixed: omega_cdm_h2 = {FIXED_PARAMS['omega_cdm']}\n")
        f.write(f"# χ² computed over {CHI2_LMIN} <= l <= {CHI2_LMAX}, {CHI2_NBINS} bins\n")
        f.write(f"# Planck baseline 100×θ_s = {baseline_theta_s:.6f}\n")
        f.write("#\n")

        writer = csv.writer(f)
        writer.writerow([
            'delta_N_eff', 'r_d_Mpc', 'r_s_rec_Mpc', 'D_M_zstar_Mpc',
            'theta_s_100', 'l_peak_1', 'chi2_TT', 'chi2_TT_per_bin', 'rms_percent'
        ])
        for r in results:
            writer.writerow([
                f"{r['delta_neff']:.2f}",
                f"{r['r_d']:.4f}",
                f"{r['r_s_rec']:.4f}",
                f"{r['D_M_zstar']:.4f}",
                f"{r['theta_s_100']:.6f}",
                r['l_peak'],
                f"{r['chi2_TT']:.2f}",
                f"{r['chi2_red']:.4f}",
                f"{r['rms_percent']:.4f}"
            ])
    print(f"\nSaved: {csv_file}")

    # Generate diagnostic figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    delta_neff_arr = np.array([r['delta_neff'] for r in results])
    theta_s_arr = np.array([r['theta_s_100'] for r in results])
    chi2_arr = np.array([r['chi2_TT'] for r in results])

    # Top panel: 100×θ_s vs ΔN_eff
    ax1 = axes[0]
    ax1.plot(delta_neff_arr, theta_s_arr, 'bo-', linewidth=2, markersize=8, label='H₀ = 70')
    ax1.axhline(baseline_theta_s, color='black', linestyle='--', linewidth=1.5,
                label=f'Planck baseline ({baseline_theta_s:.4f})')
    ax1.axvline(best_delta_neff, color='red', linestyle=':', alpha=0.7)

    # Mark best-fit point
    ax1.plot(best_delta_neff, results[best_idx]['theta_s_100'], 'r*', markersize=15,
             label=f'Best fit (ΔN_eff = {best_delta_neff:.2f})')

    ax1.set_ylabel(r'$100 \times \theta_s$', fontsize=12)
    ax1.set_title('Angular Scale vs Extra Radiation', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(104.0, 105.2)

    # Bottom panel: χ²_TT vs ΔN_eff
    ax2 = axes[1]
    ax2.plot(delta_neff_arr, chi2_arr, 'bo-', linewidth=2, markersize=8)
    ax2.axvline(best_delta_neff, color='red', linestyle=':', alpha=0.7)

    # Mark minimum
    ax2.plot(best_delta_neff, best_chi2, 'r*', markersize=15,
             label=f'Minimum: χ² = {best_chi2:.1f} at ΔN_eff = {best_delta_neff:.2f}')

    ax2.set_xlabel(r'$\Delta N_{\rm eff}$', fontsize=12)
    ax2.set_ylabel(r'$\chi^2_{\rm TT}$', fontsize=12)
    ax2.set_title('CMB χ² vs Extra Radiation (relative to Planck baseline)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(chi2_arr) * 1.1)

    plt.tight_layout()

    fig_file = output_dir / 'class_deltaNeff_scan.png'
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'class_deltaNeff_scan.pdf', bbox_inches='tight')
    print(f"Saved: {fig_file}")
    plt.close()

    print("\n" + "=" * 70)
    print("Scan complete!")
    print("=" * 70)

    return results, best_delta_neff


if __name__ == "__main__":
    results, best_delta_neff = main()
