#!/usr/bin/env python3
"""
Task 5: 1D χ² scan in k_f

Fix cosmology at Planck 2018 best-fit values and scan k_f from 0 to 2.
Compute χ² for each likelihood:
- Planck TT
- DESI BAO
- SH0ES H0 prior

Find where the compromise exists.
"""

import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'cobaya_packages/code/planck/clipy')
import classy

# =============================================================================
# Constants
# =============================================================================
C_LIGHT = 299792.458

# Fixed cosmology (Planck 2018 best-fit)
COSMO_PARAMS = {
    'H0': 67.27,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'n_s': 0.9649,
    'logA': 3.044,
    'tau_reio': 0.0544,
}

# MTDF parameters
MTDF_ALPHA = 1.30
MTDF_BETA_EOS = 0.573
MTDF_Z_T = 0.74

# SH0ES
H0_SHOES = 73.04
H0_SHOES_ERR = 1.04

# =============================================================================
# Data loading
# =============================================================================

def load_desi_bao():
    """Load DESI Y1 BAO data."""
    data_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'bao_desi')
    mean_path = f'{data_dir}/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
    cov_path = f'{data_dir}/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

    z_eff, obs_values, obs_types = [], [], []
    with open(mean_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                z_eff.append(float(parts[0]))
                obs_values.append(float(parts[1]))
                obs_types.append(parts[2])

    return np.array(z_eff), np.array(obs_values), obs_types, np.loadtxt(cov_path)

# =============================================================================
# Likelihood computation
# =============================================================================

_planck_lkl = None

def init_planck_likelihood():
    global _planck_lkl
    if _planck_lkl is None:
        import clipy
        clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik'
        _planck_lkl = clipy.clik(clik_path)
    return _planck_lkl


def compute_model(k_f, H0_override=None):
    """Compute Cls and distances for given k_f."""
    cosmo = classy.Class()

    H0 = H0_override if H0_override else COSMO_PARAMS['H0']
    A_s = np.exp(COSMO_PARAMS['logA']) * 1e-10

    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2600,
        'P_k_max_h/Mpc': 1.0,
        'omega_b': COSMO_PARAMS['omega_b'],
        'omega_cdm': COSMO_PARAMS['omega_cdm'],
        'H0': H0,
        'tau_reio': COSMO_PARAMS['tau_reio'],
        'A_s': A_s,
        'n_s': COSMO_PARAMS['n_s'],
        'mtdf': 'yes',
        'mtdf_efe': 'yes',
        'mtdf_growth': 'no',
        'mtdf_k_f': k_f,
        'mtdf_alpha': MTDF_ALPHA,
        'mtdf_beta_eos': MTDF_BETA_EOS,
        'mtdf_z_t': MTDF_Z_T,
    }

    cosmo.set(params)
    cosmo.compute()

    cls = cosmo.lensed_cl(2600)
    T_cmb = 2.7255e6
    factor = T_cmb**2

    cl_tt = cls['tt'][:2601] * factor
    cl_ee = cls['ee'][:2601] * factor
    cl_te = cls['te'][:2601] * factor

    bg = cosmo.get_background()
    idx_drag = np.argmin(np.abs(bg['z'] - 1060))
    r_d = bg['comov.snd.hrz.'][idx_drag]

    # Also get r_s at recombination
    idx_rec = np.argmin(np.abs(bg['z'] - 1089))
    r_s = bg['comov.snd.hrz.'][idx_rec]

    cosmo.struct_cleanup()
    cosmo.empty()

    return cl_tt, cl_ee, cl_te, bg, r_d, r_s


def chi2_planck(cl_tt, cl_ee, cl_te):
    """Compute Planck TT chi-squared."""
    lkl = init_planck_likelihood()
    lmax = lkl.get_lmax()

    if lmax[1] < 0:  # TT-only
        input_vec = np.concatenate([cl_tt[:lmax[0]+1], [1.0]])
    else:
        cl_bb = np.zeros(lmax[2]+1) if lmax[2] >= 0 else np.array([])
        input_vec = np.concatenate([
            cl_tt[:lmax[0]+1],
            cl_ee[:lmax[1]+1],
            cl_bb,
            cl_te[:lmax[3]+1],
            [1.0]
        ])

    loglike = float(lkl(input_vec))
    return -2 * loglike


def chi2_bao(z_arr, obs_values, obs_types, cov_matrix, bg, r_d):
    """Compute DESI BAO chi-squared."""
    predictions = []
    for z, obs_type in zip(z_arr, obs_types):
        idx = np.argmin(np.abs(bg['z'] - z))
        D_M = bg['comov. dist.'][idx]
        H_z = bg['H [1/Mpc]'][idx] * C_LIGHT
        D_H = C_LIGHT / H_z
        D_V = (D_M**2 * C_LIGHT * z / H_z)**(1/3)

        if 'DV' in obs_type:
            predictions.append(D_V / r_d)
        elif 'DM' in obs_type:
            predictions.append(D_M / r_d)
        elif 'DH' in obs_type:
            predictions.append(D_H / r_d)

    residual = obs_values - np.array(predictions)
    cov_inv = np.linalg.inv(cov_matrix)
    return float(residual @ cov_inv @ residual)


def chi2_h0(H0):
    """Compute SH0ES H0 chi-squared."""
    return ((H0 - H0_SHOES) / H0_SHOES_ERR)**2


# =============================================================================
# Main scan
# =============================================================================

def main():
    print("=" * 70)
    print("Task 5: 1D χ² scan in k_f")
    print("=" * 70)

    # Initialize
    init_planck_likelihood()
    z_bao, obs_bao, types_bao, cov_bao = load_desi_bao()

    print(f"\nFixed cosmology (Planck 2018 best-fit):")
    for name, val in COSMO_PARAMS.items():
        print(f"  {name:12s} = {val}")

    print(f"\nMTDF parameters:")
    print(f"  alpha    = {MTDF_ALPHA}")
    print(f"  beta_eos = {MTDF_BETA_EOS}")
    print(f"  z_t      = {MTDF_Z_T}")

    # Scan k_f
    kf_values = np.linspace(0, 2.0, 21)

    results = []

    print("\n" + "=" * 70)
    print("SCANNING k_f...")
    print("=" * 70)
    print(f"\n{'k_f':>6} | {'χ²_TT':>10} | {'χ²_BAO':>10} | {'χ²_H0':>8} | {'χ²_tot':>10} | {'r_s':>8}")
    print("-" * 70)

    for k_f in kf_values:
        try:
            cl_tt, cl_ee, cl_te, bg, r_d, r_s = compute_model(k_f)

            c2_tt = chi2_planck(cl_tt, cl_ee, cl_te)
            c2_bao = chi2_bao(z_bao, obs_bao, types_bao, cov_bao, bg, r_d)
            c2_h0 = chi2_h0(COSMO_PARAMS['H0'])
            c2_tot = c2_tt + c2_bao + c2_h0

            print(f"{k_f:6.2f} | {c2_tt:10.2f} | {c2_bao:10.2f} | {c2_h0:8.2f} | {c2_tot:10.2f} | {r_s:8.2f}")

            results.append({
                'k_f': k_f,
                'chi2_tt': c2_tt,
                'chi2_bao': c2_bao,
                'chi2_h0': c2_h0,
                'chi2_tot': c2_tot,
                'r_s': r_s,
                'r_d': r_d,
            })

        except Exception as e:
            print(f"{k_f:6.2f} | ERROR: {e}")

    # Find minima
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    results = np.array([(r['k_f'], r['chi2_tt'], r['chi2_bao'], r['chi2_h0'], r['chi2_tot'], r['r_s']) for r in results])
    kf_arr = results[:, 0]
    chi2_tt_arr = results[:, 1]
    chi2_bao_arr = results[:, 2]
    chi2_h0_arr = results[:, 3]
    chi2_tot_arr = results[:, 4]
    rs_arr = results[:, 5]

    # Best k_f for each likelihood
    best_tt = kf_arr[np.argmin(chi2_tt_arr)]
    best_bao = kf_arr[np.argmin(chi2_bao_arr)]
    best_tot = kf_arr[np.argmin(chi2_tot_arr)]

    print(f"\nBest k_f by likelihood:")
    print(f"  Planck TT:   k_f = {best_tt:.2f} (χ² = {np.min(chi2_tt_arr):.2f})")
    print(f"  DESI BAO:    k_f = {best_bao:.2f} (χ² = {np.min(chi2_bao_arr):.2f})")
    print(f"  Combined:    k_f = {best_tot:.2f} (χ² = {np.min(chi2_tot_arr):.2f})")

    # H0 chi2 is constant (fixed cosmology), but we can show its value
    print(f"  H0 tension:  χ² = {chi2_h0_arr[0]:.2f} (H0 = {COSMO_PARAMS['H0']:.2f} vs SH0ES {H0_SHOES:.2f})")

    # r_s reduction
    rs_0 = rs_arr[0]  # k_f = 0
    print(f"\nSound horizon reduction:")
    print(f"  r_s(k_f=0)   = {rs_0:.4f} Mpc")
    print(f"  r_s(k_f=1)   = {rs_arr[10]:.4f} Mpc")
    print(f"  r_s(k_f=2)   = {rs_arr[-1]:.4f} Mpc")
    print(f"  Δr_s/r_s(k_f=1) = {100*(rs_arr[10] - rs_0)/rs_0:.2f}%")

    # Save results
    output_dir = str(Path(__file__).parent.parent / 'output')
    np.savetxt(f"{output_dir}/task5_kf_scan.txt",
               np.column_stack([kf_arr, chi2_tt_arr, chi2_bao_arr, chi2_h0_arr, chi2_tot_arr, rs_arr]),
               header="k_f chi2_TT chi2_BAO chi2_H0 chi2_total r_s", fmt="%.4f")

    print(f"\nResults saved to {output_dir}/task5_kf_scan.txt")

    # Now scan with H0 = 70 to see the effect
    print("\n" + "=" * 70)
    print("SCAN WITH H0 = 70 (MTDF TARGET)")
    print("=" * 70)

    print(f"\n{'k_f':>6} | {'χ²_TT':>10} | {'χ²_BAO':>10} | {'χ²_H0':>8} | {'χ²_tot':>10} | {'r_s':>8}")
    print("-" * 70)

    results_h70 = []
    for k_f in kf_values:
        try:
            cl_tt, cl_ee, cl_te, bg, r_d, r_s = compute_model(k_f, H0_override=70.0)

            c2_tt = chi2_planck(cl_tt, cl_ee, cl_te)
            c2_bao = chi2_bao(z_bao, obs_bao, types_bao, cov_bao, bg, r_d)
            c2_h0 = chi2_h0(70.0)
            c2_tot = c2_tt + c2_bao + c2_h0

            print(f"{k_f:6.2f} | {c2_tt:10.2f} | {c2_bao:10.2f} | {c2_h0:8.2f} | {c2_tot:10.2f} | {r_s:8.2f}")

            results_h70.append({
                'k_f': k_f,
                'chi2_tt': c2_tt,
                'chi2_bao': c2_bao,
                'chi2_h0': c2_h0,
                'chi2_tot': c2_tot,
                'r_s': r_s,
            })

        except Exception as e:
            print(f"{k_f:6.2f} | ERROR: {e}")

    if results_h70:
        results_h70_arr = np.array([(r['k_f'], r['chi2_tt'], r['chi2_bao'], r['chi2_h0'], r['chi2_tot'], r['r_s']) for r in results_h70])
        np.savetxt(f"{output_dir}/task5_kf_scan_H70.txt",
                   results_h70_arr,
                   header="k_f chi2_TT chi2_BAO chi2_H0 chi2_total r_s", fmt="%.4f")

        best_h70 = results_h70_arr[np.argmin(results_h70_arr[:, 4]), 0]
        print(f"\nWith H0=70: Best combined k_f = {best_h70:.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAt fixed Planck cosmology (H0={COSMO_PARAMS['H0']}):")
    print(f"  - Planck TT prefers k_f ~ {best_tt:.1f} (minimal EFE)")
    print(f"  - DESI BAO prefers k_f ~ {best_bao:.1f}")
    print(f"  - SH0ES tension: {np.sqrt(chi2_h0_arr[0]):.1f}σ")

    print(f"\nPhysical interpretation:")
    print(f"  - k_f=1 reduces r_s by ~{100*(rs_arr[10] - rs_0)/rs_0:.1f}%")
    print(f"  - This is the expected MTDF early field energy effect")

    return True


if __name__ == '__main__':
    success = main()
