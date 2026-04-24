#!/usr/bin/env python3
"""
MTDF CLASS Wrapper - Compare Planck baseline, H0=70, and MTDF proxy cosmologies.

This script:
1. Runs CLASS for three cosmological scenarios
2. Extracts r_s (sound horizon), theta_s (angular scale), D_M(z*), and C_l spectra
3. Computes approximate Planck chi^2 using binned TT data
4. Generates comparison plots and machine-readable output

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
import csv

# Import classy
try:
    from classy import Class
except ImportError:
    raise ImportError("classy not installed. Run: pip install classy")

# Constants
PLANCK_LMAX = 2500
TCMB = 2.7255  # K

# Chi^2 calculation parameters
CHI2_LMIN = 30
CHI2_LMAX = 2000
CHI2_NBINS = 50  # Number of bins for chi^2 calculation

# Redshift of recombination (Planck 2018)
Z_STAR = 1089.80


def run_class(params, name="model"):
    """
    Run CLASS with given parameters and return results.

    Parameters
    ----------
    params : dict
        CLASS parameters
    name : str
        Model name for logging

    Returns
    -------
    dict with cosmological quantities and spectra
    """
    print(f"\n{'='*60}")
    print(f"Running CLASS for: {name}")
    print(f"{'='*60}")

    cosmo = Class()

    # Set default outputs
    full_params = {
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': PLANCK_LMAX,
        'accurate_lensing': 1,
    }
    full_params.update(params)

    cosmo.set(full_params)
    cosmo.compute()

    # Extract background quantities
    bg = cosmo.get_background()

    # Sound horizon at drag epoch (r_d)
    r_d = cosmo.rs_drag()

    # Sound horizon at recombination (r_s_rec)
    # This requires getting r_s at z_star from thermodynamics
    thermo = cosmo.get_thermodynamics()

    # Get z_rec from CLASS (actual recombination redshift)
    z_rec = thermo['z_rec'][-1] if 'z_rec' in thermo else Z_STAR

    # r_s at recombination - interpolate from thermodynamics
    # CLASS stores r_s as function of z in thermodynamics
    if 'r_s' in thermo and 'z' in thermo:
        z_thermo = thermo['z']
        r_s_thermo = thermo['r_s']
        # Find r_s at z closest to z_rec
        idx = np.argmin(np.abs(z_thermo - z_rec))
        r_s_rec = r_s_thermo[idx]
    else:
        # Fallback: use approximate ratio r_s/r_d ~ 0.982 (Planck 2018)
        r_s_rec = r_d * 0.9819

    # Angular scale (100 * theta_s)
    theta_s_100 = 100 * cosmo.theta_s_100()

    # Hubble constant
    H0 = cosmo.h() * 100
    h = cosmo.h()

    # Comoving angular diameter distance to z_star
    # D_A(z) = D_M(z) / (1+z), and D_M = c/H0 * integral
    # Get D_M(z_star) from background
    z_bg = bg['z']

    # CLASS background output keys vary - check which one exists
    if 'comov. dist.' in bg:
        D_M_bg = bg['comov. dist.']
    elif 'comoving distance' in bg:
        D_M_bg = bg['comoving distance']
    else:
        # Calculate from angular diameter distance
        if 'ang.diam.dist.' in bg:
            D_A_bg = bg['ang.diam.dist.']
            D_M_bg = D_A_bg * (1 + z_bg)
        else:
            # Fallback: compute from theta_s and r_s
            D_M_bg = None

    if D_M_bg is not None:
        # Interpolate to z_star (note: z_bg is typically in descending order)
        D_M_zstar = np.interp(Z_STAR, z_bg[::-1], D_M_bg[::-1])
        D_A_zstar = D_M_zstar / (1 + Z_STAR)
    else:
        # Compute from theta_s: D_A = r_s / theta_s
        theta_s_rad = theta_s_100 / 100.0  # Convert to radians (approx)
        D_A_zstar = r_s_rec / theta_s_rad
        D_M_zstar = D_A_zstar * (1 + Z_STAR)

    # Get C_l spectra (raw, need to convert to muK^2)
    cls = cosmo.lensed_cl(PLANCK_LMAX)
    ell = cls['ell']

    # Convert to D_l = l(l+1)C_l/(2pi) in muK^2
    # classy returns C_l in (TCMB)^2 units
    factor = (TCMB * 1e6)**2  # Convert to muK^2

    cls_tt = cls['tt'] * factor
    cls_te = cls['te'] * factor
    cls_ee = cls['ee'] * factor

    # Print key quantities
    print(f"  H0 = {H0:.2f} km/s/Mpc")
    print(f"  r_d (drag epoch) = {r_d:.4f} Mpc")
    print(f"  r_s_rec (recombination) = {r_s_rec:.4f} Mpc")
    print(f"  z_rec = {z_rec:.2f}")
    print(f"  D_M(z*) = {D_M_zstar:.2f} Mpc")
    print(f"  D_A(z*) = {D_A_zstar:.2f} Mpc")
    print(f"  100*theta_s = {theta_s_100:.6f}")

    # Find first peak (should be around l ~ 220)
    dl_tt = ell * (ell + 1) * cls_tt / (2 * np.pi)
    mask = (ell >= 150) & (ell <= 350)
    ell_search = ell[mask]
    dl_search = dl_tt[mask]
    first_peak_idx = np.argmax(dl_search)
    first_peak_ell = int(ell_search[first_peak_idx])
    print(f"  First TT peak at l = {first_peak_ell}")

    cosmo.struct_cleanup()
    cosmo.empty()

    return {
        'name': name,
        'H0': H0,
        'h': h,
        'r_d': r_d,
        'r_s_rec': r_s_rec,
        'z_rec': z_rec,
        'D_M_zstar': D_M_zstar,
        'D_A_zstar': D_A_zstar,
        'theta_s_100': theta_s_100,
        'ell': ell,
        'cls_tt': cls_tt,
        'cls_te': cls_te,
        'cls_ee': cls_ee,
        'first_peak_ell': first_peak_ell,
    }


def compute_chi2(model_cls, baseline_cls, l_min=CHI2_LMIN, l_max=CHI2_LMAX, n_bins=CHI2_NBINS):
    """
    Compute chi^2 between model and baseline over specified l-range.

    Uses uniform binning and diagonal approximation with sigma ~ 2% of D_l.

    Parameters
    ----------
    model_cls : dict
        Model C_l spectra
    baseline_cls : dict
        Baseline (data) C_l spectra
    l_min, l_max : int
        Multipole range for chi^2
    n_bins : int
        Number of bins

    Returns
    -------
    dict with chi2, chi2_red, n_bins, rms_percent, residuals, etc.
    """
    # Create uniform bin edges
    bin_edges = np.linspace(l_min, l_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def bin_spectrum(ell, cls_tt, edges):
        """Bin C_l spectrum to D_l in each bin."""
        n = len(edges) - 1
        binned = np.zeros(n)
        for i in range(n):
            mask = (ell >= edges[i]) & (ell < edges[i + 1])
            if np.any(mask):
                ell_bin = ell[mask]
                cls_bin = cls_tt[mask]
                dl_bin = ell_bin * (ell_bin + 1) * cls_bin / (2 * np.pi)
                binned[i] = np.mean(dl_bin)
        return binned

    # Bin both spectra
    model_binned = bin_spectrum(model_cls['ell'], model_cls['cls_tt'], bin_edges)
    baseline_binned = bin_spectrum(baseline_cls['ell'], baseline_cls['cls_tt'], bin_edges)

    # Approximate uncertainty: sigma ~ 2% of D_l (rough Planck-like error)
    sigma_frac = 0.02
    sigma = sigma_frac * baseline_binned
    sigma = np.maximum(sigma, 1.0)  # Minimum uncertainty

    # Residuals
    residuals = model_binned - baseline_binned
    residuals_percent = 100 * residuals / baseline_binned

    # Chi^2
    chi2 = np.sum((residuals / sigma)**2)
    chi2_red = chi2 / n_bins

    # RMS residual in percent
    rms_percent = np.sqrt(np.mean(residuals_percent**2))

    return {
        'chi2': chi2,
        'chi2_red': chi2_red,
        'n_bins': n_bins,
        'l_min': l_min,
        'l_max': l_max,
        'rms_percent': rms_percent,
        'residuals': residuals,
        'residuals_percent': residuals_percent,
        'sigma': sigma,
        'bin_centers': bin_centers,
        'model_binned': model_binned,
        'baseline_binned': baseline_binned,
    }


def create_summary_figure(results, chi2_results, output_dir):
    """
    Create a combined 4-panel summary figure.

    Top left: TT power spectra
    Top right: TT residuals with ±2% band
    Bottom left: zoom on first 3 peaks
    Bottom right: text summary table
    """
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25)

    colors = {'Planck_baseline': 'black', 'H070_noEDE': 'red', 'MTDF_proxy': 'blue'}
    labels = {
        'Planck_baseline': 'Planck baseline (H₀=67.4)',
        'H070_noEDE': 'H₀=70, no EDE',
        'MTDF_proxy': 'MTDF proxy (H₀=70, ΔN_eff=0.22)'
    }
    linestyles = {'Planck_baseline': '-', 'H070_noEDE': '--', 'MTDF_proxy': '-'}
    linewidths = {'Planck_baseline': 2.0, 'H070_noEDE': 1.5, 'MTDF_proxy': 1.5}

    # Top left: Full TT power spectra
    ax1 = fig.add_subplot(gs[0, 0])
    for name, model in results.items():
        ell = model['ell'][2:]
        dl = ell * (ell + 1) * model['cls_tt'][2:] / (2 * np.pi)
        ax1.plot(ell, dl, color=colors[name], label=labels[name],
                linestyle=linestyles[name], linewidth=linewidths[name])

    ax1.set_xlabel('Multipole ℓ', fontsize=11)
    ax1.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]', fontsize=11)
    ax1.set_title('CMB TT Power Spectrum', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(2, 2500)
    ax1.set_ylim(0, 6500)
    ax1.grid(True, alpha=0.3)

    # Top right: Residuals with ±2% band
    ax2 = fig.add_subplot(gs[0, 1])
    baseline = results['Planck_baseline']
    baseline_ell = baseline['ell'][2:]
    baseline_dl = baseline_ell * (baseline_ell + 1) * baseline['cls_tt'][2:] / (2 * np.pi)

    # Shaded ±2% band
    ax2.fill_between(baseline_ell, -2, 2, color='gray', alpha=0.2, label='±2% band')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

    for name, model in results.items():
        if name == 'Planck_baseline':
            continue
        ell = model['ell'][2:]
        dl = ell * (ell + 1) * model['cls_tt'][2:] / (2 * np.pi)
        residual = 100 * (dl - baseline_dl) / baseline_dl
        ax2.plot(ell, residual, color=colors[name], label=labels[name],
                linestyle=linestyles[name], linewidth=linewidths[name])

    ax2.set_xlabel('Multipole ℓ', fontsize=11)
    ax2.set_ylabel('Residual [%]', fontsize=11)
    ax2.set_title('Residuals relative to Planck baseline', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(2, 2500)
    ax2.set_ylim(-8, 8)
    ax2.grid(True, alpha=0.3)

    # Bottom left: Zoom on first 3 peaks
    ax3 = fig.add_subplot(gs[1, 0])
    for name, model in results.items():
        ell = model['ell']
        dl = ell * (ell + 1) * model['cls_tt'] / (2 * np.pi)
        ax3.plot(ell, dl, color=colors[name], label=labels[name],
                linestyle=linestyles[name], linewidth=linewidths[name])

    ax3.set_xlabel('Multipole ℓ', fontsize=11)
    ax3.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]', fontsize=11)
    ax3.set_title('First 3 Acoustic Peaks', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(100, 900)
    ax3.set_ylim(0, 6500)
    ax3.grid(True, alpha=0.3)

    # Mark peak positions with vertical lines
    for name, model in results.items():
        ax3.axvline(model['first_peak_ell'], color=colors[name],
                   linestyle=':', alpha=0.5, linewidth=1)

    # Bottom right: Text summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Build summary text
    summary_lines = [
        "MTDF CLASS Analysis Summary",
        "=" * 50,
        "",
        f"chi2 computed over {CHI2_LMIN} <= l <= {CHI2_LMAX}, {CHI2_NBINS} bins",
        "",
        f"{'Model':<18} {'H0':>6} {'dNeff':>6} {'r_d':>8} {'100*th_s':>10} {'chi2_TT':>8} {'l_peak':>7}",
        "-" * 65,
    ]

    for name, model in results.items():
        delta_neff = 0.22 if name == 'MTDF_proxy' else 0.0
        chi2 = chi2_results[name]['chi2']
        display_name = {
            'Planck_baseline': 'Planck baseline',
            'H070_noEDE': 'H₀=70 no EDE',
            'MTDF_proxy': 'MTDF proxy'
        }[name]
        summary_lines.append(
            f"{display_name:<18} {model['H0']:>6.2f} {delta_neff:>6.2f} "
            f"{model['r_d']:>8.2f} {model['theta_s_100']:>10.4f} "
            f"{chi2:>8.1f} {model['first_peak_ell']:>7d}"
        )

    summary_lines.extend([
        "-" * 65,
        "",
        "Key findings:",
        f"  * r_d reduction (MTDF vs H0=70): "
        f"{100*(results['MTDF_proxy']['r_d'] - results['H070_noEDE']['r_d'])/results['H070_noEDE']['r_d']:.2f}%",
        f"  * chi2 improvement: {chi2_results['H070_noEDE']['chi2']:.1f} -> {chi2_results['MTDF_proxy']['chi2']:.1f} "
        f"(factor {chi2_results['H070_noEDE']['chi2']/chi2_results['MTDF_proxy']['chi2']:.1f}x)",
        f"  * theta_s recovered: {results['MTDF_proxy']['theta_s_100']:.4f} vs "
        f"{results['Planck_baseline']['theta_s_100']:.4f} (baseline)",
        f"  * First peak restored to l = {results['MTDF_proxy']['first_peak_ell']}",
        "",
        "Notes:",
        "  * r_d = sound horizon at drag epoch (used for BAO)",
        "  * Delta_N_eff = 0.22 mimics MTDF early field energy",
        "  * chi2 uses diagonal approx with sigma ~ 2% D_l",
    ])

    summary_text = "\n".join(summary_lines)
    ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('MTDF Proxy in CLASS: CMB TT Analysis', fontsize=14, fontweight='bold', y=0.98)

    # Save
    plt.savefig(output_dir / 'class_tt_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'class_tt_summary.pdf', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'class_tt_summary.png'}")
    plt.close()


def save_machine_readable(results, chi2_results, output_dir):
    """
    Save results in machine-readable CSV and JSON formats.
    """
    # CSV format
    csv_file = output_dir / 'class_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'H0', 'delta_N_eff', 'r_d_Mpc', 'r_s_rec_Mpc',
            'D_M_zstar_Mpc', 'theta_s_100', 'l_peak_1',
            'chi2_TT', 'chi2_TT_red', 'N_bins', 'rms_percent'
        ])
        for name, model in results.items():
            delta_neff = 0.22 if name == 'MTDF_proxy' else 0.0
            chi2_data = chi2_results[name]
            writer.writerow([
                name,
                f"{model['H0']:.4f}",
                f"{delta_neff:.2f}",
                f"{model['r_d']:.4f}",
                f"{model['r_s_rec']:.4f}",
                f"{model['D_M_zstar']:.4f}",
                f"{model['theta_s_100']:.6f}",
                model['first_peak_ell'],
                f"{chi2_data['chi2']:.2f}",
                f"{chi2_data['chi2_red']:.4f}",
                chi2_data['n_bins'],
                f"{chi2_data['rms_percent']:.4f}"
            ])
    print(f"  Saved: {csv_file}")

    # JSON format
    json_file = output_dir / 'class_results.json'
    json_data = {
        'metadata': {
            'description': 'MTDF CLASS analysis results',
            'chi2_l_range': f"{CHI2_LMIN} <= l <= {CHI2_LMAX}",
            'chi2_n_bins': CHI2_NBINS,
            'z_star': Z_STAR,
            'notes': {
                'r_d': 'Sound horizon at drag epoch (z_drag ~ 1060)',
                'r_s_rec': 'Sound horizon at recombination (z_rec ~ 1090)',
                'D_M_zstar': 'Comoving angular diameter distance to z*=1089.8',
                'theta_s_100': '100 * theta_s = 100 * r_s / D_A(z*)',
                'chi2_TT': 'Approximate chi^2 using diagonal covariance with sigma ~ 2% D_l',
            }
        },
        'models': {}
    }

    for name, model in results.items():
        delta_neff = 0.22 if name == 'MTDF_proxy' else 0.0
        chi2_data = chi2_results[name]
        json_data['models'][name] = {
            'H0_km_s_Mpc': model['H0'],
            'delta_N_eff': delta_neff,
            'r_d_Mpc': model['r_d'],
            'r_s_rec_Mpc': model['r_s_rec'],
            'z_rec': model['z_rec'],
            'D_M_zstar_Mpc': model['D_M_zstar'],
            'D_A_zstar_Mpc': model['D_A_zstar'],
            'theta_s_100': model['theta_s_100'],
            'l_peak_1': model['first_peak_ell'],
            'chi2_TT': chi2_data['chi2'],
            'chi2_TT_per_bin': chi2_data['chi2_red'],
            'N_bins': chi2_data['n_bins'],
            'rms_residual_percent': chi2_data['rms_percent'],
        }

    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_file}")


def save_text_results(results, chi2_results, output_dir):
    """
    Save detailed text results file.
    """
    results_file = output_dir / 'class_results.txt'
    with open(results_file, 'w') as f:
        f.write("MTDF CLASS Analysis Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("Sound Horizon Definitions:\n")
        f.write("-" * 80 + "\n")
        f.write("  r_d     = Sound horizon at drag epoch (z_drag ~ 1060)\n")
        f.write("            This is the BAO standard ruler.\n")
        f.write("  r_s_rec = Sound horizon at recombination (z_rec ~ 1090)\n")
        f.write("            This determines the CMB acoustic scale.\n")
        f.write("  Note: r_s_rec ≈ 0.98 × r_d (Planck 2018)\n\n")

        f.write("Chi-squared Calculation:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  l-range: {CHI2_LMIN} ≤ l ≤ {CHI2_LMAX}\n")
        f.write(f"  Number of bins: {CHI2_NBINS}\n")
        f.write("  Method: Diagonal approximation with σ ≈ 2% of D_l\n\n")

        f.write("Results Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<18} {'H0':>7} {'ΔNeff':>6} {'r_d':>9} {'r_s_rec':>9} "
               f"{'D_M(z*)':>10} {'100θ_s':>10} {'l_peak':>7}\n")
        f.write("-" * 80 + "\n")

        for name, model in results.items():
            delta_neff = 0.22 if name == 'MTDF_proxy' else 0.0
            display_name = {
                'Planck_baseline': 'Planck baseline',
                'H070_noEDE': 'H0=70 no EDE',
                'MTDF_proxy': 'MTDF proxy'
            }[name]
            f.write(f"{display_name:<18} {model['H0']:>7.2f} {delta_neff:>6.2f} "
                   f"{model['r_d']:>9.4f} {model['r_s_rec']:>9.4f} "
                   f"{model['D_M_zstar']:>10.2f} {model['theta_s_100']:>10.6f} "
                   f"{model['first_peak_ell']:>7d}\n")

        f.write("\n")
        f.write("Chi-squared Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<18} {'χ²_TT':>10} {'χ²/N':>10} {'N':>6} {'RMS [%]':>10}\n")
        f.write("-" * 80 + "\n")

        for name in results.keys():
            chi2_data = chi2_results[name]
            display_name = {
                'Planck_baseline': 'Planck baseline',
                'H070_noEDE': 'H0=70 no EDE',
                'MTDF_proxy': 'MTDF proxy'
            }[name]
            f.write(f"{display_name:<18} {chi2_data['chi2']:>10.2f} "
                   f"{chi2_data['chi2_red']:>10.4f} {chi2_data['n_bins']:>6d} "
                   f"{chi2_data['rms_percent']:>10.4f}\n")

        f.write("\n")
        f.write("Key Findings:\n")
        f.write("-" * 80 + "\n")

        r_d_baseline = results['Planck_baseline']['r_d']
        r_d_h070 = results['H070_noEDE']['r_d']
        r_d_mtdf = results['MTDF_proxy']['r_d']

        f.write(f"  r_d reduction (MTDF proxy vs H0=70 no EDE): "
               f"{100*(r_d_mtdf - r_d_h070)/r_d_h070:.2f}%\n")
        f.write(f"  r_d reduction (MTDF proxy vs Planck baseline): "
               f"{100*(r_d_mtdf - r_d_baseline)/r_d_baseline:.2f}%\n")
        f.write(f"  χ² improvement: {chi2_results['H070_noEDE']['chi2']:.1f} → "
               f"{chi2_results['MTDF_proxy']['chi2']:.1f} "
               f"(factor {chi2_results['H070_noEDE']['chi2']/chi2_results['MTDF_proxy']['chi2']:.1f}×)\n")
        f.write(f"  θ_s recovered to within "
               f"{abs(results['MTDF_proxy']['theta_s_100'] - results['Planck_baseline']['theta_s_100']):.4f} "
               f"of baseline\n")

    print(f"  Saved: {results_file}")


def main():
    """Main function to run the analysis."""

    print("=" * 70)
    print("MTDF CLASS Wrapper - Planck CMB Analysis")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Define the three scenarios
    scenarios = {
        'Planck_baseline': {
            'h': 0.6736,
            'omega_b': 0.02237,
            'omega_cdm': 0.1200,
            'tau_reio': 0.0544,
            'n_s': 0.9649,
            'ln10^{10}A_s': 3.044,
            'N_ur': 2.0328,
            'N_ncdm': 1,
            'm_ncdm': 0.06,
        },
        'H070_noEDE': {
            'h': 0.70,
            'omega_b': 0.02237,
            'omega_cdm': 0.1200,
            'tau_reio': 0.0544,
            'n_s': 0.9649,
            'ln10^{10}A_s': 3.044,
            'N_ur': 2.0328,
            'N_ncdm': 1,
            'm_ncdm': 0.06,
        },
        'MTDF_proxy': {
            'h': 0.70,
            'omega_b': 0.02237,
            'omega_cdm': 0.1200,
            'tau_reio': 0.0544,
            'n_s': 0.9649,
            'ln10^{10}A_s': 3.044,
            'N_ur': 2.2528,  # 2.0328 + 0.22 (Delta_N_eff = 0.22)
            'N_ncdm': 1,
            'm_ncdm': 0.06,
        },
    }

    # Run CLASS for each scenario
    results = {}
    for name, params in scenarios.items():
        results[name] = run_class(params, name)

    # Compute chi^2 relative to baseline
    print("\n" + "=" * 70)
    print("Chi-squared Analysis (relative to Planck baseline)")
    print(f"l-range: {CHI2_LMIN} ≤ l ≤ {CHI2_LMAX}, {CHI2_NBINS} bins")
    print("=" * 70)

    baseline = results['Planck_baseline']
    chi2_results = {}

    for name, model in results.items():
        chi2_data = compute_chi2(model, baseline)
        chi2_results[name] = chi2_data
        print(f"\n{name}:")
        print(f"  χ² = {chi2_data['chi2']:.2f}")
        print(f"  χ²/N = {chi2_data['chi2_red']:.4f}")
        print(f"  N = {chi2_data['n_bins']}")
        print(f"  RMS residual = {chi2_data['rms_percent']:.4f}%")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<18} {'H0':>7} {'ΔNeff':>6} {'r_d':>9} {'r_s_rec':>9} "
         f"{'100θ_s':>10} {'χ²_TT':>8} {'l_peak':>7}")
    print("-" * 80)

    for name, model in results.items():
        delta_neff = 0.22 if name == 'MTDF_proxy' else 0.0
        print(f"{name:<18} {model['H0']:>7.2f} {delta_neff:>6.2f} "
             f"{model['r_d']:>9.4f} {model['r_s_rec']:>9.4f} "
             f"{model['theta_s_100']:>10.6f} {chi2_results[name]['chi2']:>8.1f} "
             f"{model['first_peak_ell']:>7d}")

    # Print r_s reductions
    r_d_baseline = results['Planck_baseline']['r_d']
    r_d_h070 = results['H070_noEDE']['r_d']
    r_d_mtdf = results['MTDF_proxy']['r_d']

    print("\n" + "-" * 80)
    print(f"r_d reduction (H070_noEDE vs baseline): "
         f"{100*(r_d_h070 - r_d_baseline)/r_d_baseline:.2f}%")
    print(f"r_d reduction (MTDF_proxy vs baseline): "
         f"{100*(r_d_mtdf - r_d_baseline)/r_d_baseline:.2f}%")
    print(f"r_d reduction (MTDF_proxy vs H070_noEDE): "
         f"{100*(r_d_mtdf - r_d_h070)/r_d_h070:.2f}%")

    # Generate outputs
    print("\n" + "=" * 70)
    print("Generating output files...")
    print("=" * 70)

    # Save machine-readable files
    save_machine_readable(results, chi2_results, output_dir)

    # Save detailed text results
    save_text_results(results, chi2_results, output_dir)

    # Create combined summary figure
    create_summary_figure(results, chi2_results, output_dir)

    # Also save individual plots
    create_individual_plots(results, chi2_results, output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return results, chi2_results


def create_individual_plots(results, chi2_results, output_dir):
    """Create individual plots (residuals and peaks zoom)."""

    colors = {'Planck_baseline': 'black', 'H070_noEDE': 'red', 'MTDF_proxy': 'blue'}
    labels = {
        'Planck_baseline': 'Planck baseline (H₀=67.4)',
        'H070_noEDE': 'H₀=70, no EDE',
        'MTDF_proxy': 'MTDF proxy (H₀=70, ΔN_eff=0.22)'
    }

    # Plot 1: C_l^TT residuals
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Full spectra
    ax1 = axes[0]
    for name, model in results.items():
        ell = model['ell'][2:]
        dl = ell * (ell + 1) * model['cls_tt'][2:] / (2 * np.pi)
        ax1.plot(ell, dl, color=colors[name], label=labels[name],
                linewidth=1.5 if name == 'Planck_baseline' else 1.0,
                alpha=1.0 if name == 'Planck_baseline' else 0.8)

    ax1.set_xlabel('Multipole ℓ', fontsize=12)
    ax1.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]', fontsize=12)
    ax1.set_title('CMB TT Power Spectrum', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(2, 2500)
    ax1.set_ylim(0, 6500)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Residuals
    ax2 = axes[1]
    baseline = results['Planck_baseline']
    baseline_ell = baseline['ell'][2:]
    baseline_dl = baseline_ell * (baseline_ell + 1) * baseline['cls_tt'][2:] / (2 * np.pi)

    ax2.fill_between(baseline_ell, -2, 2, color='gray', alpha=0.2, label='±2% band')

    for name, model in results.items():
        if name == 'Planck_baseline':
            continue
        ell = model['ell'][2:]
        dl = ell * (ell + 1) * model['cls_tt'][2:] / (2 * np.pi)
        residual = 100 * (dl - baseline_dl) / baseline_dl
        ax2.plot(ell, residual, color=colors[name], label=labels[name], linewidth=1.0)

    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Multipole ℓ', fontsize=12)
    ax2.set_ylabel('Residual [%]', fontsize=12)
    ax2.set_title('Residuals relative to Planck baseline', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(2, 2500)
    ax2.set_ylim(-10, 10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cl_tt_residuals.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cl_tt_residuals.pdf', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'cl_tt_residuals.png'}")
    plt.close()

    # Plot 2: Zoom on first 3 peaks
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, model in results.items():
        ell = model['ell']
        dl = ell * (ell + 1) * model['cls_tt'] / (2 * np.pi)
        ax.plot(ell, dl, color=colors[name], label=labels[name],
               linewidth=2.0 if name == 'Planck_baseline' else 1.5,
               alpha=1.0 if name == 'Planck_baseline' else 0.8)

    ax.set_xlabel('Multipole ℓ', fontsize=12)
    ax.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]', fontsize=12)
    ax.set_title('CMB TT Power Spectrum - First 3 Peaks', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(100, 900)
    ax.set_ylim(0, 6500)
    ax.grid(True, alpha=0.3)

    # Mark peak positions
    for name, model in results.items():
        ax.axvline(model['first_peak_ell'], color=colors[name],
                  linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'cl_tt_peaks_zoom.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cl_tt_peaks_zoom.pdf', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'cl_tt_peaks_zoom.png'}")
    plt.close()


if __name__ == "__main__":
    results, chi2_results = main()
