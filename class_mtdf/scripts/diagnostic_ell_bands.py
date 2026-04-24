#!/usr/bin/env python3
"""
MTDF Diagnostic: ℓ-Band Analysis
================================
Identifies which multipole ranges and spectra drive the rejection of higher k_f.

Computes Planck χ² contributions in ℓ-bands for k_f = 0, 0.1, 1.0
"""

import numpy as np
import sys
import os

# Use the installed classy module (with MTDF support)
from classy import Class

# Planck likelihood
try:
    import clik
    HAS_CLIK = True
except:
    HAS_CLIK = False
    print("Warning: clik not available, using approximate χ² calculation")

# Output directory
output_dir = str(Path(__file__).parent.parent / 'output')

# Planck 2018 best-fit cosmology (baseline)
PLANCK_COSMO = {
    'h': 0.6774,
    'omega_b': 0.02242,
    'omega_cdm': 0.11933,
    'n_s': 0.9665,
    'ln10^{10}A_s': 3.047,
    'tau_reio': 0.0561,
}

# MTDF fixed parameters
MTDF_FIXED = {
    'mtdf_alpha': 1.3,
    'mtdf_beta_eos': 0.573,
    'mtdf_z_t': 0.74,
}

# ℓ-bands to analyze
ELL_BANDS_TT = [(2, 30), (30, 250), (250, 800), (800, 2508)]
ELL_BANDS_TE = [(30, 250), (250, 800), (800, 2508)]
ELL_BANDS_EE = [(30, 250), (250, 800), (800, 2508)]

# k_f values to test
KF_VALUES = [0.0, 0.1, 1.0]


def compute_cls(k_f, cosmo_params=PLANCK_COSMO):
    """Compute C_ℓ for given k_f value"""
    cosmo = Class()

    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2600,
        'P_k_max_1/Mpc': 1.0,
        **cosmo_params,
    }

    # Add MTDF parameters only if k_f > 0
    if k_f > 0:
        params.update({
            'mtdf': 'yes',
            **MTDF_FIXED,
            'mtdf_k_f': k_f,
            'mtdf_efe': 'yes',
            'mtdf_growth': 'yes',
        })

    cosmo.set(params)
    cosmo.compute()

    cls = cosmo.lensed_cl(2600)

    # Extract and convert to μK²
    T_cmb = 2.7255e6  # μK
    ell = cls['ell']

    result = {
        'ell': ell,
        'TT': cls['tt'] * T_cmb**2,
        'TE': cls['te'] * T_cmb**2,
        'EE': cls['ee'] * T_cmb**2,
    }

    # Get derived parameters
    result['r_d'] = cosmo.rs_drag()
    result['sigma8'] = cosmo.sigma8()
    result['H0'] = cosmo.Hubble(0) * 299792.458

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


def planck_noise_variance(ell, spectrum='TT'):
    """
    Approximate Planck noise + cosmic variance per ℓ
    Returns variance on D_ℓ = ℓ(ℓ+1)C_ℓ/(2π)

    Based on Planck 2018 noise levels
    """
    ell = np.atleast_1d(ell).astype(float)

    # Planck beam FWHM ~ 5 arcmin for 143 GHz
    theta_fwhm = 5.0 * np.pi / (180 * 60)  # radians
    sigma_beam = theta_fwhm / np.sqrt(8 * np.log(2))

    # Beam suppression
    beam = np.exp(-ell * (ell + 1) * sigma_beam**2)

    # Instrument noise (approximate)
    if spectrum == 'TT':
        noise_level = 40.0  # μK²
        f_sky = 0.57
    elif spectrum == 'TE':
        noise_level = 60.0
        f_sky = 0.50
    elif spectrum == 'EE':
        noise_level = 80.0
        f_sky = 0.50
    else:
        noise_level = 40.0
        f_sky = 0.57

    # Noise per ℓ
    N_ell = noise_level / beam**2

    # Cosmic variance factor
    cv_factor = np.sqrt(2.0 / ((2 * ell + 1) * f_sky))

    return N_ell, cv_factor


def compute_chi2_band(cls_model, cls_ref, ell_min, ell_max, spectrum='TT'):
    """
    Compute approximate χ² contribution for a given ℓ-band

    χ² = Σ_ℓ [(D_ℓ^model - D_ℓ^ref)² / σ²_ℓ]

    where σ²_ℓ includes cosmic variance and noise
    """
    ell = cls_ref['ell']
    mask = (ell >= ell_min) & (ell <= ell_max)
    ell_band = ell[mask]

    if len(ell_band) == 0:
        return 0.0

    # Get C_ℓ values
    cl_model = cls_model[spectrum][mask]
    cl_ref = cls_ref[spectrum][mask]

    # Convert to D_ℓ
    prefactor = ell_band * (ell_band + 1) / (2 * np.pi)
    dl_model = prefactor * cl_model
    dl_ref = prefactor * cl_ref

    # Get noise and cosmic variance
    N_ell, cv_factor = planck_noise_variance(ell_band, spectrum)

    # Total variance: cosmic variance on signal + noise
    # σ² ~ (2/(2ℓ+1)/f_sky) * (C_ℓ + N_ℓ)²
    sigma2 = (cv_factor**2) * (np.abs(dl_ref) + N_ell)**2

    # Avoid division by zero
    sigma2 = np.maximum(sigma2, 1e-10)

    # Chi-squared
    chi2 = np.sum((dl_model - dl_ref)**2 / sigma2)

    return chi2


def load_planck_likelihood():
    """Load Planck plik_lite TTTEEE likelihood"""
    clik_path = "cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik"

    if HAS_CLIK and os.path.exists(clik_path):
        try:
            lkl = clik.clik(clik_path)
            return lkl
        except:
            return None
    return None


def compute_full_planck_chi2(cls_dict, lkl=None):
    """
    Compute full Planck high-ℓ TTTEEE chi-squared
    """
    if lkl is None:
        # Approximate calculation using our band method
        chi2 = 0
        for band in ELL_BANDS_TT:
            chi2 += compute_chi2_band(cls_dict, cls_dict, band[0], band[1], 'TT')
        return chi2  # This won't work well without reference - placeholder

    # Use actual clik likelihood
    ell = cls_dict['ell']

    # Prepare C_ℓ array for clik (TT, EE, BB, TE from ℓ=0)
    lmax = 2508
    cl_tt = np.zeros(lmax + 1)
    cl_ee = np.zeros(lmax + 1)
    cl_bb = np.zeros(lmax + 1)
    cl_te = np.zeros(lmax + 1)

    # Fill in values (convert from μK² to dimensionless)
    T_cmb = 2.7255e6
    for i, l in enumerate(ell):
        if l <= lmax:
            cl_tt[int(l)] = cls_dict['TT'][i] / T_cmb**2
            cl_ee[int(l)] = cls_dict['EE'][i] / T_cmb**2
            cl_te[int(l)] = cls_dict['TE'][i] / T_cmb**2

    # Build input vector for plik_lite
    # Order: TT[30:2509], EE[30:2509], TE[30:2509], A_planck
    vec = np.concatenate([
        cl_tt[30:2509],
        cl_ee[30:2509],
        cl_te[30:2509],
        [1.0]  # A_planck
    ])

    try:
        loglike = lkl(vec)[0]
        chi2 = -2 * loglike
        return chi2
    except:
        return np.nan


def main():
    print("=" * 70)
    print("MTDF Diagnostic: ℓ-Band Analysis")
    print("=" * 70)
    print()

    # Load Planck likelihood if available
    lkl = load_planck_likelihood()
    if lkl:
        print("Loaded Planck plik_lite TTTEEE likelihood")
    else:
        print("Using approximate χ² calculation")
    print()

    # Compute C_ℓ for each k_f value
    print("Computing C_ℓ spectra...")
    cls_data = {}

    for k_f in KF_VALUES:
        print(f"  k_f = {k_f}...")
        cls_data[k_f] = compute_cls(k_f)
        print(f"    r_d = {cls_data[k_f]['r_d']:.2f} Mpc, "
              f"H0 = {cls_data[k_f]['H0']:.2f} km/s/Mpc, "
              f"σ8 = {cls_data[k_f]['sigma8']:.4f}")

    print()

    # Reference is k_f = 0 (ΛCDM)
    cls_ref = cls_data[0.0]

    # Compute χ² for each k_f, spectrum, and ℓ-band
    print("Computing χ² contributions by ℓ-band...")
    print()

    results = []

    # Table header
    header = "| k_f  | Spectrum | ℓ-range    | χ²_approx | Δχ² vs k_f=0 |"
    separator = "|------|----------|------------|-----------|--------------|"

    print(header)
    print(separator)

    for k_f in KF_VALUES:
        cls_model = cls_data[k_f]

        # TT bands
        for ell_min, ell_max in ELL_BANDS_TT:
            chi2 = compute_chi2_band(cls_model, cls_ref, ell_min, ell_max, 'TT')
            chi2_ref = compute_chi2_band(cls_ref, cls_ref, ell_min, ell_max, 'TT')
            delta_chi2 = chi2 - chi2_ref

            results.append({
                'k_f': k_f,
                'spectrum': 'TT',
                'ell_min': ell_min,
                'ell_max': ell_max,
                'chi2': chi2,
                'delta_chi2': delta_chi2
            })

            print(f"| {k_f:.1f}  | TT       | {ell_min:4d}-{ell_max:<4d} | {chi2:9.2f} | {delta_chi2:+12.2f} |")

        # TE bands
        for ell_min, ell_max in ELL_BANDS_TE:
            chi2 = compute_chi2_band(cls_model, cls_ref, ell_min, ell_max, 'TE')
            chi2_ref = compute_chi2_band(cls_ref, cls_ref, ell_min, ell_max, 'TE')
            delta_chi2 = chi2 - chi2_ref

            results.append({
                'k_f': k_f,
                'spectrum': 'TE',
                'ell_min': ell_min,
                'ell_max': ell_max,
                'chi2': chi2,
                'delta_chi2': delta_chi2
            })

            print(f"| {k_f:.1f}  | TE       | {ell_min:4d}-{ell_max:<4d} | {chi2:9.2f} | {delta_chi2:+12.2f} |")

        # EE bands
        for ell_min, ell_max in ELL_BANDS_EE:
            chi2 = compute_chi2_band(cls_model, cls_ref, ell_min, ell_max, 'EE')
            chi2_ref = compute_chi2_band(cls_ref, cls_ref, ell_min, ell_max, 'EE')
            delta_chi2 = chi2 - chi2_ref

            results.append({
                'k_f': k_f,
                'spectrum': 'EE',
                'ell_min': ell_min,
                'ell_max': ell_max,
                'chi2': chi2,
                'delta_chi2': delta_chi2
            })

            print(f"| {k_f:.1f}  | EE       | {ell_min:4d}-{ell_max:<4d} | {chi2:9.2f} | {delta_chi2:+12.2f} |")

        print(separator)

    print()

    # Find the largest Δχ² for k_f = 1.0
    results_kf1 = [r for r in results if r['k_f'] == 1.0]
    worst = max(results_kf1, key=lambda x: x['delta_chi2'])

    print("=" * 70)
    print("LARGEST χ² DEGRADATION FOR k_f = 1.0:")
    print(f"  Spectrum: {worst['spectrum']}")
    print(f"  ℓ-range:  {worst['ell_min']}-{worst['ell_max']}")
    print(f"  Δχ²:      {worst['delta_chi2']:.2f}")
    print("=" * 70)
    print()

    # Compute full Planck χ² if likelihood available
    if lkl:
        print("Full Planck TTTEEE χ² comparison:")
        for k_f in KF_VALUES:
            chi2 = compute_full_planck_chi2(cls_data[k_f], lkl)
            print(f"  k_f = {k_f}: χ² = {chi2:.2f}")

    # Save results
    print()
    print("Saving results...")

    # Save table
    with open(f"{output_dir}/diagnostic_ell_bands.txt", 'w') as f:
        f.write("MTDF Diagnostic: ℓ-Band Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write("Reference cosmology: Planck 2018 best-fit\n")
        f.write(f"MTDF parameters: α={MTDF_FIXED['mtdf_alpha']}, "
                f"β_eos={MTDF_FIXED['mtdf_beta_eos']}, z_t={MTDF_FIXED['mtdf_z_t']}\n\n")

        f.write(header + "\n")
        f.write(separator + "\n")

        for r in results:
            f.write(f"| {r['k_f']:.1f}  | {r['spectrum']:8s} | {r['ell_min']:4d}-{r['ell_max']:<4d} | "
                   f"{r['chi2']:9.2f} | {r['delta_chi2']:+12.2f} |\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("LARGEST χ² DEGRADATION FOR k_f = 1.0:\n")
        f.write(f"  Spectrum: {worst['spectrum']}\n")
        f.write(f"  ℓ-range:  {worst['ell_min']}-{worst['ell_max']}\n")
        f.write(f"  Δχ²:      {worst['delta_chi2']:.2f}\n")

    print(f"  Saved: {output_dir}/diagnostic_ell_bands.txt")

    # Create plot
    create_diagnostic_plot(cls_data, results)
    print(f"  Saved: {output_dir}/diagnostic_ell_bands.png")

    print()
    print("Done!")


def create_diagnostic_plot(cls_data, results):
    """Create Δχ² vs ℓ plot"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get ℓ values
    ell = cls_data[0.0]['ell']

    # Plot 1: C_ℓ comparison (TT)
    ax = axes[0, 0]
    for k_f in [0.0, 0.1, 1.0]:
        prefactor = ell * (ell + 1) / (2 * np.pi)
        dl = prefactor * cls_data[k_f]['TT']
        label = f'k_f = {k_f}'
        ax.plot(ell, dl, label=label, alpha=0.8)
    ax.set_xlim(2, 2500)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell^{TT}$ [$\mu K^2$]')
    ax.set_title('Temperature Power Spectrum')
    ax.legend()
    ax.set_xscale('log')

    # Plot 2: Relative difference (TT)
    ax = axes[0, 1]
    for k_f in [0.1, 1.0]:
        dl_ref = ell * (ell + 1) / (2 * np.pi) * cls_data[0.0]['TT']
        dl_model = ell * (ell + 1) / (2 * np.pi) * cls_data[k_f]['TT']
        rel_diff = (dl_model - dl_ref) / dl_ref * 100
        ax.plot(ell, rel_diff, label=f'k_f = {k_f}')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(2, 2500)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\Delta D_\ell^{TT} / D_\ell^{TT}$ [%]')
    ax.set_title('TT Relative Difference vs ΛCDM')
    ax.legend()
    ax.set_xscale('log')

    # Plot 3: Δχ² by ℓ-band (bar chart)
    ax = axes[1, 0]

    # Prepare data for k_f = 1.0
    results_kf1 = [r for r in results if r['k_f'] == 1.0]

    x_labels = []
    delta_chi2_tt = []
    delta_chi2_te = []
    delta_chi2_ee = []

    # Get unique ℓ-bands
    bands_all = set()
    for r in results_kf1:
        bands_all.add((r['ell_min'], r['ell_max']))
    bands_sorted = sorted(bands_all)

    for band in bands_sorted:
        x_labels.append(f"{band[0]}-{band[1]}")

        tt_result = [r for r in results_kf1 if r['spectrum'] == 'TT'
                     and r['ell_min'] == band[0] and r['ell_max'] == band[1]]
        te_result = [r for r in results_kf1 if r['spectrum'] == 'TE'
                     and r['ell_min'] == band[0] and r['ell_max'] == band[1]]
        ee_result = [r for r in results_kf1 if r['spectrum'] == 'EE'
                     and r['ell_min'] == band[0] and r['ell_max'] == band[1]]

        delta_chi2_tt.append(tt_result[0]['delta_chi2'] if tt_result else 0)
        delta_chi2_te.append(te_result[0]['delta_chi2'] if te_result else 0)
        delta_chi2_ee.append(ee_result[0]['delta_chi2'] if ee_result else 0)

    x = np.arange(len(x_labels))
    width = 0.25

    ax.bar(x - width, delta_chi2_tt, width, label='TT', color='blue', alpha=0.7)
    ax.bar(x, delta_chi2_te, width, label='TE', color='green', alpha=0.7)
    ax.bar(x + width, delta_chi2_ee, width, label='EE', color='red', alpha=0.7)

    ax.set_xlabel('ℓ-band')
    ax.set_ylabel(r'$\Delta\chi^2$ (k_f=1.0 vs k_f=0)')
    ax.set_title(r'$\chi^2$ Degradation by ℓ-band for k_f = 1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Total Δχ² by spectrum
    ax = axes[1, 1]

    total_tt = sum(r['delta_chi2'] for r in results_kf1 if r['spectrum'] == 'TT')
    total_te = sum(r['delta_chi2'] for r in results_kf1 if r['spectrum'] == 'TE')
    total_ee = sum(r['delta_chi2'] for r in results_kf1 if r['spectrum'] == 'EE')

    spectra = ['TT', 'TE', 'EE', 'Total']
    totals = [total_tt, total_te, total_ee, total_tt + total_te + total_ee]
    colors = ['blue', 'green', 'red', 'purple']

    bars = ax.bar(spectra, totals, color=colors, alpha=0.7)
    ax.set_ylabel(r'Total $\Delta\chi^2$ (k_f=1.0 vs k_f=0)')
    ax.set_title(r'Total $\chi^2$ Degradation by Spectrum')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/diagnostic_ell_bands.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
