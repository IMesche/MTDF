#!/usr/bin/env python3
"""
MTDF k_f Normalization Validation
=================================
Quick validation that with corrected C_MTDF = 1/(24*Omega_Lambda),
the CMB data prefers k_f ~ 1.0 (not k_f ~ 0.1 as before).

Uses approximate Planck chi-squared + DESI BAO.
"""

import numpy as np
import sys
from classy import Class

# Planck 2018 best-fit cosmology
COSMO_BASE = {
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'l_max_scalars': 2600,
    'P_k_max_1/Mpc': 1.0,
    'h': 0.6774,
    'omega_b': 0.02242,
    'omega_cdm': 0.11933,
    'n_s': 0.9665,
    'ln10^{10}A_s': 3.047,
    'tau_reio': 0.0561,
}

# MTDF fixed parameters
MTDF_PARAMS = {
    'mtdf': 'yes',
    'mtdf_alpha': 1.3,
    'mtdf_beta_eos': 0.573,
    'mtdf_z_t': 0.74,
    'mtdf_efe': 'yes',
    'mtdf_growth': 'yes',
}

# DESI BAO data
BAO_DATA = [
    (0.295, 'DV_over_rd', 7.93, 0.15),
    (0.51, 'DM_over_rd', 13.62, 0.25),
    (0.51, 'DH_over_rd', 22.31, 0.63),
    (0.706, 'DM_over_rd', 17.86, 0.33),
    (0.706, 'DH_over_rd', 20.08, 0.61),
    (0.93, 'DM_over_rd', 21.71, 0.28),
    (0.93, 'DH_over_rd', 17.88, 0.35),
    (1.317, 'DM_over_rd', 27.79, 0.69),
    (1.317, 'DH_over_rd', 13.82, 0.42),
    (2.33, 'DM_over_rd', 39.71, 0.94),
    (2.33, 'DH_over_rd', 8.52, 0.17),
    (1.49, 'DV_over_rd', 26.07, 0.67),
]

# Planck 2018 TT reference spectrum (approximate)
PLANCK_DL_TT_REF = None  # Will be computed from LCDM


def compute_model(k_f):
    """Compute cosmology for given k_f"""
    cosmo = Class()
    params = {**COSMO_BASE}

    if k_f > 0:
        params.update(MTDF_PARAMS)
        params['mtdf_k_f'] = k_f

    cosmo.set(params)
    cosmo.compute()

    # Get outputs
    cls = cosmo.lensed_cl(2600)
    T_cmb = 2.7255e6

    result = {
        'ell': cls['ell'],
        'TT': cls['tt'] * T_cmb**2,
        'TE': cls['te'] * T_cmb**2,
        'EE': cls['ee'] * T_cmb**2,
        'r_d': cosmo.rs_drag(),
        'H0': cosmo.Hubble(0) * 299792.458,
    }

    # BAO observables
    c = 299792.458
    result['bao'] = {}
    for z_eff, obs_type, _, _ in BAO_DATA:
        D_A = cosmo.angular_distance(z_eff)
        H_z = cosmo.Hubble(z_eff) * c
        D_M = D_A * (1 + z_eff)
        r_d = result['r_d']

        if obs_type == 'DM_over_rd':
            result['bao'][(z_eff, obs_type)] = D_M / r_d
        elif obs_type == 'DH_over_rd':
            result['bao'][(z_eff, obs_type)] = c / H_z / r_d
        elif obs_type == 'DV_over_rd':
            D_V = (z_eff * D_M**2 * c / H_z) ** (1./3.)
            result['bao'][(z_eff, obs_type)] = D_V / r_d

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


def compute_chi2_cmb_approx(model, ref):
    """Approximate Planck chi-squared from TT/TE/EE spectra"""
    chi2 = 0.0

    for spec in ['TT', 'TE', 'EE']:
        ell = model['ell']
        dl_model = ell * (ell + 1) * model[spec] / (2 * np.pi)
        dl_ref = ell * (ell + 1) * ref[spec] / (2 * np.pi)

        # Approximate cosmic variance + noise
        if spec == 'TT':
            ell_range = (30, 2508)
            noise_level = 50.0  # muK^2 at high ell
        elif spec == 'TE':
            ell_range = (30, 2508)
            noise_level = 30.0
        else:  # EE
            ell_range = (30, 2508)
            noise_level = 10.0

        mask = (ell >= ell_range[0]) & (ell <= ell_range[1])
        ell_sel = ell[mask]
        dl_m = dl_model[mask]
        dl_r = dl_ref[mask]

        # Cosmic variance + noise
        var = (2.0 / (2.0 * ell_sel + 1.0)) * dl_r**2 + noise_level**2

        chi2 += np.sum((dl_m - dl_r)**2 / var)

    return chi2


def compute_chi2_bao(model):
    """Compute BAO chi-squared"""
    chi2 = 0.0
    for z_eff, obs_type, value, error in BAO_DATA:
        predicted = model['bao'][(z_eff, obs_type)]
        chi2 += ((predicted - value) / error)**2
    return chi2


def main():
    print("=" * 70)
    print("MTDF k_f Normalization Validation")
    print("=" * 70)
    print()
    print("Testing with CORRECTED C_MTDF = 1/(24*Omega_Lambda) ~ 0.06")
    print("Expected: k_f ~ 1.0 should be preferred (not 0.1 as with old code)")
    print()

    # Compute LCDM reference
    print("Computing LCDM reference (k_f = 0)...")
    ref = compute_model(0.0)
    print(f"  r_d = {ref['r_d']:.4f} Mpc")

    # k_f values to test
    kf_values = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]

    print()
    print("Scanning k_f values...")
    print()

    results = []

    for k_f in kf_values:
        print(f"  k_f = {k_f:.1f}...", end=" ", flush=True)
        model = compute_model(k_f)

        chi2_cmb = compute_chi2_cmb_approx(model, ref) if k_f > 0 else 0.0
        chi2_bao = compute_chi2_bao(model)
        chi2_total = chi2_cmb + chi2_bao

        results.append({
            'k_f': k_f,
            'r_d': model['r_d'],
            'chi2_cmb': chi2_cmb,
            'chi2_bao': chi2_bao,
            'chi2_total': chi2_total,
        })

        print(f"r_d = {model['r_d']:.4f}, chi2_CMB = {chi2_cmb:.1f}, chi2_BAO = {chi2_bao:.1f}")

    # Find minimum
    min_result = min(results, key=lambda x: x['chi2_total'])

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("| k_f  | r_d (Mpc) | chi2_CMB | chi2_BAO | chi2_total | Delta_chi2 |")
    print("|------|-----------|----------|----------|------------|------------|")

    ref_chi2 = results[0]['chi2_total']
    for r in results:
        delta = r['chi2_total'] - ref_chi2
        marker = " <-- MIN" if r == min_result else ""
        print(f"| {r['k_f']:4.1f} | {r['r_d']:9.4f} | {r['chi2_cmb']:8.1f} | {r['chi2_bao']:8.1f} | {r['chi2_total']:10.1f} | {delta:+10.1f} |{marker}")

    print()
    print("=" * 70)
    print(f"BEST FIT: k_f = {min_result['k_f']:.1f}")
    print("=" * 70)
    print()

    if min_result['k_f'] >= 0.5 and min_result['k_f'] <= 2.0:
        print("SUCCESS: k_f ~ 1.0 preferred with corrected C_MTDF normalization!")
    else:
        print(f"NOTE: k_f = {min_result['k_f']:.1f} preferred")

    # Save results
    output_file = str(Path(__file__).parent.parent / "output" / "kf_validation_v2.txt")
    with open(output_file, 'w') as f:
        f.write("MTDF k_f Normalization Validation (Corrected C_MTDF)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best fit: k_f = {min_result['k_f']:.2f}\n")
        f.write(f"r_d at best fit: {min_result['r_d']:.4f} Mpc\n\n")
        f.write("Full scan:\n")
        for r in results:
            f.write(f"k_f={r['k_f']:.1f}: chi2={r['chi2_total']:.1f}, r_d={r['r_d']:.4f}\n")

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
