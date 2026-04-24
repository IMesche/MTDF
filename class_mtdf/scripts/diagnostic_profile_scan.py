#!/usr/bin/env python3
"""
MTDF Diagnostic: Profile Shape Scan
====================================
Tests whether ANY profile shape can make k_f = 1.0 acceptable to the CMB.

Fix k_f = 1.0 and scan over σ_z and z_peak to find minimum χ².
"""

import numpy as np
import sys
import os
from itertools import product

# Use the installed classy module (with MTDF support)
from classy import Class

# Output directory
output_dir = str(Path(__file__).parent.parent / 'output')

# Planck 2018 best-fit cosmology
PLANCK_COSMO = {
    'h': 0.6774,
    'omega_b': 0.02242,
    'omega_cdm': 0.11933,
    'n_s': 0.9665,
    'ln10^{10}A_s': 3.047,
    'tau_reio': 0.0561,
}

# MTDF fixed parameters (except profile shape)
MTDF_FIXED = {
    'mtdf_alpha': 1.3,
    'mtdf_beta_eos': 0.573,
    'mtdf_z_t': 0.74,
    'mtdf_k_f': 1.0,  # Fix at full theory amplitude
}

# Profile parameters to scan
SIGMA_Z_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
Z_PEAK_VALUES = [2500, 3000, 3400, 4000, 5000]

# BAO data
BAO_DATA = [
    # (z_eff, observable, value, error, type)
    (0.295, 'DV_over_rd', 7.93, 0.15, 'DV'),
    (0.51, 'DM_over_rd', 13.62, 0.25, 'DM'),
    (0.51, 'DH_over_rd', 22.31, 0.63, 'DH'),
    (0.706, 'DM_over_rd', 17.86, 0.33, 'DM'),
    (0.706, 'DH_over_rd', 20.08, 0.61, 'DH'),
    (0.93, 'DM_over_rd', 21.71, 0.28, 'DM'),
    (0.93, 'DH_over_rd', 17.88, 0.35, 'DH'),
    (1.317, 'DM_over_rd', 27.79, 0.69, 'DM'),
    (1.317, 'DH_over_rd', 13.82, 0.42, 'DH'),
    (2.33, 'DM_over_rd', 39.71, 0.94, 'DM'),
    (2.33, 'DH_over_rd', 8.52, 0.17, 'DH'),
    (1.49, 'DV_over_rd', 26.07, 0.67, 'DV'),
]


def compute_cls_and_bao(sigma_z, z_peak, cosmo_params=PLANCK_COSMO):
    """Compute C_ℓ and BAO observables for given profile parameters"""
    cosmo = Class()

    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2600,
        'P_k_max_1/Mpc': 1.0,
        **cosmo_params,
        'mtdf': 'yes',
        **MTDF_FIXED,
        'mtdf_efe': 'yes',
        'mtdf_growth': 'yes',
        'mtdf_z_peak': z_peak,
        'mtdf_sigma_z': sigma_z,
    }

    try:
        cosmo.set(params)
        cosmo.compute()
    except Exception as e:
        print(f"  CLASS failed for σ_z={sigma_z}, z_peak={z_peak}: {e}")
        return None

    # Get C_ℓ
    cls = cosmo.lensed_cl(2600)
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

    # Compute BAO observables
    result['bao'] = {}
    c = 299792.458  # km/s

    for z_eff, obs_type, _, _, _ in BAO_DATA:
        # Angular diameter distance
        D_A = cosmo.angular_distance(z_eff)  # Mpc
        # Hubble parameter
        H_z = cosmo.Hubble(z_eff) * c  # km/s/Mpc
        # Comoving distance
        D_M = D_A * (1 + z_eff)

        # Store
        key = (z_eff, obs_type)
        if obs_type == 'DM_over_rd':
            result['bao'][key] = D_M / result['r_d']
        elif obs_type == 'DH_over_rd':
            result['bao'][key] = c / H_z / result['r_d']
        elif obs_type == 'DV_over_rd':
            D_V = (z_eff * D_M**2 * c / H_z) ** (1./3.)
            result['bao'][key] = D_V / result['r_d']

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


def compute_bao_chi2(result):
    """Compute BAO χ²"""
    chi2 = 0

    for z_eff, obs_type, value, error, _ in BAO_DATA:
        key = (z_eff, obs_type)
        if key in result['bao']:
            model = result['bao'][key]
            chi2 += ((model - value) / error) ** 2

    return chi2


def planck_noise_variance(ell, spectrum='TT'):
    """Approximate Planck noise + cosmic variance"""
    ell = np.atleast_1d(ell).astype(float)

    theta_fwhm = 5.0 * np.pi / (180 * 60)
    sigma_beam = theta_fwhm / np.sqrt(8 * np.log(2))
    beam = np.exp(-ell * (ell + 1) * sigma_beam**2)

    if spectrum == 'TT':
        noise_level = 40.0
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

    N_ell = noise_level / beam**2
    cv_factor = np.sqrt(2.0 / ((2 * ell + 1) * f_sky))

    return N_ell, cv_factor


def compute_planck_chi2_approx(result, ref_result):
    """Compute approximate Planck χ² relative to ΛCDM"""
    chi2_total = 0

    for spectrum in ['TT', 'TE', 'EE']:
        ell = result['ell']
        mask = (ell >= 30) & (ell <= 2508)
        ell_band = ell[mask]

        prefactor = ell_band * (ell_band + 1) / (2 * np.pi)
        dl_model = prefactor * result[spectrum][mask]
        dl_ref = prefactor * ref_result[spectrum][mask]

        N_ell, cv_factor = planck_noise_variance(ell_band, spectrum)
        sigma2 = (cv_factor**2) * (np.abs(dl_ref) + N_ell)**2
        sigma2 = np.maximum(sigma2, 1e-10)

        chi2_total += np.sum((dl_model - dl_ref)**2 / sigma2)

    return chi2_total


def main():
    print("=" * 70)
    print("MTDF Diagnostic: Profile Shape Scan")
    print("=" * 70)
    print()
    print(f"Fixed: k_f = {MTDF_FIXED['mtdf_k_f']}")
    print(f"Scanning: σ_z = {SIGMA_Z_VALUES}")
    print(f"          z_peak = {Z_PEAK_VALUES}")
    print()

    # First compute ΛCDM reference (no MTDF parameters needed)
    print("Computing ΛCDM reference...")
    cosmo_ref = Class()
    params_ref = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2600,
        'P_k_max_1/Mpc': 1.0,
        **PLANCK_COSMO,
    }
    cosmo_ref.set(params_ref)
    cosmo_ref.compute()

    cls_ref = cosmo_ref.lensed_cl(2600)
    T_cmb = 2.7255e6

    ref_result = {
        'ell': cls_ref['ell'],
        'TT': cls_ref['tt'] * T_cmb**2,
        'TE': cls_ref['te'] * T_cmb**2,
        'EE': cls_ref['ee'] * T_cmb**2,
        'r_d': cosmo_ref.rs_drag(),
        'sigma8': cosmo_ref.sigma8(),
        'H0': cosmo_ref.Hubble(0) * 299792.458,
    }

    # Compute BAO for reference
    ref_result['bao'] = {}
    c = 299792.458
    for z_eff, obs_type, _, _, _ in BAO_DATA:
        D_A = cosmo_ref.angular_distance(z_eff)
        H_z = cosmo_ref.Hubble(z_eff) * c
        D_M = D_A * (1 + z_eff)

        key = (z_eff, obs_type)
        if obs_type == 'DM_over_rd':
            ref_result['bao'][key] = D_M / ref_result['r_d']
        elif obs_type == 'DH_over_rd':
            ref_result['bao'][key] = c / H_z / ref_result['r_d']
        elif obs_type == 'DV_over_rd':
            D_V = (z_eff * D_M**2 * c / H_z) ** (1./3.)
            ref_result['bao'][key] = D_V / ref_result['r_d']

    ref_chi2_bao = compute_bao_chi2(ref_result)

    cosmo_ref.struct_cleanup()
    cosmo_ref.empty()

    print(f"  ΛCDM: r_d = {ref_result['r_d']:.2f} Mpc, H0 = {ref_result['H0']:.2f} km/s/Mpc")
    print(f"  ΛCDM BAO χ² = {ref_chi2_bao:.2f}")
    print()

    # Scan over profile parameters
    print("Scanning profile parameters...")
    print()

    results = []
    best_result = None
    best_chi2 = np.inf

    for sigma_z, z_peak in product(SIGMA_Z_VALUES, Z_PEAK_VALUES):
        print(f"  σ_z = {sigma_z}, z_peak = {z_peak}...", end=" ")

        result = compute_cls_and_bao(sigma_z, z_peak)

        if result is None:
            print("FAILED")
            continue

        # Compute χ²
        chi2_planck = compute_planck_chi2_approx(result, ref_result)
        chi2_bao = compute_bao_chi2(result)
        chi2_total = chi2_planck + chi2_bao

        # Δχ² vs ΛCDM (Planck χ² baseline is 0 for matching ref)
        delta_chi2 = chi2_planck + (chi2_bao - ref_chi2_bao)

        results.append({
            'sigma_z': sigma_z,
            'z_peak': z_peak,
            'chi2_planck': chi2_planck,
            'chi2_bao': chi2_bao,
            'chi2_total': chi2_total,
            'delta_chi2_vs_lcdm': delta_chi2,
            'r_d': result['r_d'],
            'H0': result['H0'],
            'sigma8': result['sigma8'],
        })

        print(f"χ²_Planck={chi2_planck:.1f}, χ²_BAO={chi2_bao:.1f}, Δχ²={delta_chi2:.1f}")

        if delta_chi2 < best_chi2:
            best_chi2 = delta_chi2
            best_result = results[-1]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Print table
    header = "| σ_z  | z_peak | χ²_Planck | χ²_BAO | χ²_total | Δχ² vs ΛCDM | r_d (Mpc) |"
    separator = "|------|--------|-----------|--------|----------|-------------|-----------|"

    print(header)
    print(separator)

    for r in results:
        print(f"| {r['sigma_z']:.1f}  | {r['z_peak']:5d}  | {r['chi2_planck']:9.1f} | "
              f"{r['chi2_bao']:6.1f} | {r['chi2_total']:8.1f} | {r['delta_chi2_vs_lcdm']:+11.1f} | "
              f"{r['r_d']:9.2f} |")

    print(separator)
    print()

    # Best result
    print("BEST PROFILE SHAPE (minimum Δχ²):")
    print(f"  σ_z    = {best_result['sigma_z']}")
    print(f"  z_peak = {best_result['z_peak']}")
    print(f"  Δχ²    = {best_result['delta_chi2_vs_lcdm']:.1f}")
    print(f"  r_d    = {best_result['r_d']:.2f} Mpc")
    print(f"  H0     = {best_result['H0']:.2f} km/s/Mpc")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if best_result['delta_chi2_vs_lcdm'] < 5:
        print("✓ Profile shape CAN accommodate k_f = 1.0!")
        print(f"  Best shape: σ_z = {best_result['sigma_z']}, z_peak = {best_result['z_peak']}")
        print("  → The problem is the SHAPE, not the amplitude")
        print("  → Re-derive profile shape from field equation")
    elif best_result['delta_chi2_vs_lcdm'] < 20:
        print("~ Profile shape provides PARTIAL accommodation")
        print(f"  Best Δχ² = {best_result['delta_chi2_vs_lcdm']:.1f}")
        print("  → Both shape AND amplitude may need adjustment")
    else:
        print("✗ NO profile shape can accommodate k_f = 1.0")
        print(f"  Minimum Δχ² = {best_result['delta_chi2_vs_lcdm']:.1f}")
        print("  → The problem is the AMPLITUDE (C_MTDF)")
        print("  → Re-derive coupling constant from theory")

    print()

    # Save results
    with open(f"{output_dir}/profile_scan_grid.txt", 'w') as f:
        f.write("MTDF Diagnostic: Profile Shape Scan\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fixed: k_f = {MTDF_FIXED['mtdf_k_f']}\n\n")

        f.write(header + "\n")
        f.write(separator + "\n")

        for r in results:
            f.write(f"| {r['sigma_z']:.1f}  | {r['z_peak']:5d}  | {r['chi2_planck']:9.1f} | "
                    f"{r['chi2_bao']:6.1f} | {r['chi2_total']:8.1f} | {r['delta_chi2_vs_lcdm']:+11.1f} | "
                    f"{r['r_d']:9.2f} |\n")

        f.write("\n")
        f.write("ΛCDM Reference:\n")
        f.write(f"  r_d = {ref_result['r_d']:.2f} Mpc\n")
        f.write(f"  BAO χ² = {ref_chi2_bao:.2f}\n")

    with open(f"{output_dir}/profile_scan_best.txt", 'w') as f:
        f.write("MTDF Profile Scan: Best Result\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"σ_z         = {best_result['sigma_z']}\n")
        f.write(f"z_peak      = {best_result['z_peak']}\n")
        f.write(f"χ²_Planck   = {best_result['chi2_planck']:.1f}\n")
        f.write(f"χ²_BAO      = {best_result['chi2_bao']:.1f}\n")
        f.write(f"χ²_total    = {best_result['chi2_total']:.1f}\n")
        f.write(f"Δχ² vs ΛCDM = {best_result['delta_chi2_vs_lcdm']:.1f}\n")
        f.write(f"r_d         = {best_result['r_d']:.2f} Mpc\n")
        f.write(f"H0          = {best_result['H0']:.2f} km/s/Mpc\n")
        f.write(f"σ8          = {best_result['sigma8']:.4f}\n")

    print(f"Saved: {output_dir}/profile_scan_grid.txt")
    print(f"Saved: {output_dir}/profile_scan_best.txt")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
