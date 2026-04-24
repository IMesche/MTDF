#!/usr/bin/env python3
"""
Single-point Planck likelihood evaluation for MTDF cosmologies.
Compares three cases:
  1. ΛCDM Planck 2018 best fit
  2. ΛCDM with H0=70 (no MTDF)
  3. MTDF + EFE with H0=70

Uses Planck 2018 high-ℓ TT,TE,EE lite likelihood via clipy.
"""

import numpy as np
from datetime import datetime
import sys
import os

# Add clipy to path
sys.path.insert(0, 'cobaya_packages/code/planck/clipy')

import classy

# =============================================================================
# Define cosmologies
# =============================================================================

# Planck 2018 best-fit ΛCDM (TT,TE,EE+lowE+lensing, Table 2 of arXiv:1807.06209)
planck_bestfit = {
    'name': 'LCDM Planck best-fit',
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'H0': 67.36,
    'tau_reio': 0.0544,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'mtdf': False,
}

# ΛCDM with H0=70 (keeping other Planck parameters)
lcdm_h70 = {
    'name': 'LCDM H0=70',
    'omega_b': 0.02226,
    'omega_cdm': 0.1186,
    'H0': 70.0,
    'tau_reio': 0.066,
    'A_s': 2.1e-9,
    'n_s': 0.9665,
    'mtdf': False,
}

# MTDF + EFE with H0=70
mtdf_efe = {
    'name': 'MTDF + EFE',
    'omega_b': 0.02226,
    'omega_cdm': 0.1186,
    'H0': 70.0,
    'tau_reio': 0.066,
    'A_s': 2.1e-9,
    'n_s': 0.9665,
    'mtdf': True,
    'mtdf_efe': True,
    'mtdf_alpha': 1.30,
    'mtdf_beta_eos': 0.573,
    'mtdf_z_t': 0.74,
}

cosmologies = [planck_bestfit, lcdm_h70, mtdf_efe]

# =============================================================================
# CLASS wrapper
# =============================================================================

def compute_cls(cosmo_params, ell_max=2600):
    """Compute Cls using our custom classy."""
    cosmo = classy.Class()

    # Basic CLASS params
    params = {
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': ell_max,
        'omega_b': cosmo_params['omega_b'],
        'omega_cdm': cosmo_params['omega_cdm'],
        'H0': cosmo_params['H0'],
        'tau_reio': cosmo_params['tau_reio'],
        'A_s': cosmo_params['A_s'],
        'n_s': cosmo_params['n_s'],
    }

    # Add MTDF if enabled
    if cosmo_params.get('mtdf', False):
        params['mtdf'] = 'yes'
        params['mtdf_efe'] = 'yes' if cosmo_params.get('mtdf_efe', False) else 'no'
        params['mtdf_alpha'] = cosmo_params.get('mtdf_alpha', 1.30)
        params['mtdf_beta_eos'] = cosmo_params.get('mtdf_beta_eos', 0.573)
        params['mtdf_z_t'] = cosmo_params.get('mtdf_z_t', 0.74)

    cosmo.set(params)
    cosmo.compute()

    # Get lensed Cls
    cls_dict = cosmo.lensed_cl(ell_max)

    # Convert to μK² (CLASS outputs in dimensionless units)
    T_cmb = 2.7255e6  # μK
    factor = T_cmb**2

    # Extract arrays
    ells = np.arange(ell_max + 1)
    cl_tt = cls_dict['tt'][:ell_max+1] * factor
    cl_ee = cls_dict['ee'][:ell_max+1] * factor
    cl_te = cls_dict['te'][:ell_max+1] * factor

    # Also get r_s
    bg = cosmo.get_background()
    idx = np.argmin(np.abs(bg['z'] - 1089))
    r_s = bg['comov.snd.hrz.'][idx]

    cosmo.struct_cleanup()
    cosmo.empty()

    return {
        'ells': ells,
        'cl_tt': cl_tt,
        'cl_ee': cl_ee,
        'cl_te': cl_te,
        'r_s': r_s,
    }

# =============================================================================
# Planck likelihood via clipy
# =============================================================================

def evaluate_planck_likelihood(cosmo_params):
    """Evaluate Planck high-ℓ TTTEEE lite likelihood."""
    try:
        import clipy
    except ImportError:
        return None, "clipy not installed"

    # Path to likelihood
    clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik'

    if not os.path.exists(clik_path):
        return None, f"likelihood file not found at {clik_path}"

    # Compute Cls
    cls_data = compute_cls(cosmo_params, ell_max=2600)

    # Initialize likelihood
    lkl = clipy.clik(clik_path)

    # Get ell ranges for TT, EE, TE
    lmax = lkl.get_lmax()
    # lmax is a tuple: (lmax_TT, lmax_EE, lmax_BB, lmax_TE, ...)
    lmax_tt = lmax[0]
    lmax_ee = lmax[1]
    lmax_te = lmax[3]

    # Prepare input vector
    # Format: [TT(0..lmax_tt), EE(0..lmax_ee), BB(0..lmax_bb), TE(0..lmax_te), nuisance_params...]
    # For plik_lite, we need TT, EE, TE and one nuisance parameter (A_planck)

    # Build the input array
    cl_tt = cls_data['cl_tt'][:lmax_tt+1]
    cl_ee = cls_data['cl_ee'][:lmax_ee+1]
    cl_te = cls_data['cl_te'][:lmax_te+1]

    # Pad with zeros if needed
    cl_bb = np.zeros(lmax[2]+1) if lmax[2] >= 0 else np.array([])

    # A_planck = 1.0 (no calibration correction)
    A_planck = 1.0

    # Concatenate in the order expected by clik
    input_vec = np.concatenate([cl_tt, cl_ee, cl_bb, cl_te, [A_planck]])

    # Evaluate likelihood
    loglike = lkl(input_vec)

    return loglike, cls_data['r_s']

# =============================================================================
# Main evaluation
# =============================================================================

def evaluate_cosmology(cosmo):
    """Evaluate a single cosmology."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {cosmo['name']}")
    print(f"{'='*60}")
    print(f"  H0 = {cosmo['H0']:.2f} km/s/Mpc")
    print(f"  omega_b = {cosmo['omega_b']:.5f}")
    print(f"  omega_cdm = {cosmo['omega_cdm']:.5f}")
    if cosmo.get('mtdf', False):
        print(f"  MTDF = enabled, EFE = {cosmo.get('mtdf_efe', False)}")

    try:
        loglike, r_s = evaluate_planck_likelihood(cosmo)

        if loglike is None:
            print(f"  ERROR: {r_s}")
            # Fall back to computing diagnostics
            cls_data = compute_cls(cosmo)
            return {
                'name': cosmo['name'],
                'H0': cosmo['H0'],
                'r_s': cls_data['r_s'],
                'loglike': None,
                'chi2': None,
            }

        chi2 = -2 * loglike

        print(f"\n  Results:")
        print(f"    r_s(z*) = {r_s:.2f} Mpc")
        print(f"    log(L) = {loglike:.2f}")
        print(f"    χ² = {chi2:.2f}")

        return {
            'name': cosmo['name'],
            'H0': cosmo['H0'],
            'r_s': r_s,
            'loglike': loglike,
            'chi2': chi2,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

        # Fall back to computing just r_s
        try:
            cls_data = compute_cls(cosmo)
            return {
                'name': cosmo['name'],
                'H0': cosmo['H0'],
                'r_s': cls_data['r_s'],
                'loglike': None,
                'chi2': None,
                'error': str(e),
            }
        except:
            return {
                'name': cosmo['name'],
                'H0': cosmo['H0'],
                'loglike': None,
                'chi2': None,
                'error': str(e),
            }


def main():
    """Run evaluations and produce report."""
    print("="*70)
    print("MTDF Planck Likelihood Evaluation")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Likelihood: Planck 2018 high-ℓ TT,TE,EE lite")

    results = []
    for cosmo in cosmologies:
        result = evaluate_cosmology(cosmo)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    has_chi2 = any(r.get('chi2') is not None for r in results)

    if has_chi2:
        print(f"{'Cosmology':<25} {'H0':>8} {'r_s':>10} {'χ²':>12} {'Δχ²':>10}")
        print("-"*70)
        ref_chi2 = results[0]['chi2'] if results[0].get('chi2') is not None else 0
        for r in results:
            r_s_str = f"{r['r_s']:.2f}" if r.get('r_s') else "N/A"
            if r.get('chi2') is not None:
                delta = r['chi2'] - ref_chi2
                print(f"{r['name']:<25} {r['H0']:>8.2f} {r_s_str:>10} {r['chi2']:>12.2f} {delta:>+10.2f}")
            else:
                print(f"{r['name']:<25} {r['H0']:>8.2f} {r_s_str:>10} {'N/A':>12} {'---':>10}")
    else:
        print(f"{'Cosmology':<25} {'H0':>8} {'r_s [Mpc]':>12}")
        print("-"*70)
        for r in results:
            r_s_str = f"{r['r_s']:.2f}" if r.get('r_s') else "N/A"
            print(f"{r['name']:<25} {r['H0']:>8.2f} {r_s_str:>12}")

    print("-"*70)

    # Write results to file
    output_dir = str(Path(__file__).parent.parent / 'output')
    with open(f'{output_dir}/planck_chi2_comparison.txt', 'w') as f:
        f.write("# MTDF Planck Likelihood Comparison\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Likelihood: Planck 2018 high-ℓ TT,TE,EE lite\n")
        f.write("# ----------------------------------------\n")
        if has_chi2:
            f.write(f"{'# Cosmology':<25} {'H0':>8} {'r_s':>10} {'chi2':>12} {'delta_chi2':>12}\n")
            ref_chi2 = results[0]['chi2'] if results[0].get('chi2') is not None else 0
            for r in results:
                r_s_str = f"{r['r_s']:.2f}" if r.get('r_s') else "N/A"
                if r.get('chi2') is not None:
                    delta = r['chi2'] - ref_chi2
                    f.write(f"{r['name']:<25} {r['H0']:>8.2f} {r_s_str:>10} {r['chi2']:>12.2f} {delta:>+12.2f}\n")
        else:
            for r in results:
                f.write(f"# {r['name']}: H0={r['H0']:.2f}")
                if r.get('r_s') is not None:
                    f.write(f", r_s={r['r_s']:.2f}")
                f.write("\n")

    print(f"\nResults written to: {output_dir}/planck_chi2_comparison.txt")

    return results


if __name__ == '__main__':
    main()
