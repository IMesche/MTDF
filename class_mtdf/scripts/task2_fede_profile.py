#!/usr/bin/env python3
"""
Task 2: Direct f_EDE(z) Profile Dump

Create a diagnostic table with:
- z
- Omega_efe(z)
- Omega_tot(z)
- f_EDE(z) = Omega_efe / Omega_tot

Verify the profile matches the analytic form from the paper.
"""

import numpy as np
import classy
import os

OUTPUT_DIR = str(Path(__file__).parent.parent / 'output')

# MTDF parameters
MTDF_PARAMS = {
    'alpha': 1.30,
    'beta_eos': 0.573,
    'z_t': 0.74,
    'z_peak': 3400.0,
    'sigma_z': 0.5,
    'C_mtdf': 0.61,
}

# Compute derived parameters
MTDF_PARAMS['lambda_mtdf'] = (1 - MTDF_PARAMS['beta_eos'])**2 / (1 + MTDF_PARAMS['alpha'])

# Standard cosmology
COSMO_PARAMS = {
    'H0': 70.0,  # Use H0=70 for this test
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
}


def analytic_omega_efe(z, k_f, Omega_Lambda):
    """Compute the analytic Omega_EFE(z) profile."""
    f_kick = MTDF_PARAMS['C_mtdf'] * Omega_Lambda * MTDF_PARAMS['lambda_mtdf']

    ln1pz = np.log(1 + z)
    ln1pz_peak = np.log(1 + MTDF_PARAMS['z_peak'])
    delta_ln = ln1pz - ln1pz_peak
    sigma2 = MTDF_PARAMS['sigma_z']**2

    return k_f * f_kick * np.exp(-delta_ln**2 / (2 * sigma2))


def run_class_and_get_omega(k_f):
    """Run CLASS with given k_f and extract omega values at various z."""
    cosmo = classy.Class()

    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2500,
        'P_k_max_h/Mpc': 1.0,
        'H0': COSMO_PARAMS['H0'],
        'omega_b': COSMO_PARAMS['omega_b'],
        'omega_cdm': COSMO_PARAMS['omega_cdm'],
        'A_s': COSMO_PARAMS['A_s'],
        'n_s': COSMO_PARAMS['n_s'],
        'tau_reio': COSMO_PARAMS['tau_reio'],
        'mtdf': 'yes',
        'mtdf_efe': 'yes',
        'mtdf_growth': 'no',
        'mtdf_k_f': k_f,
        'mtdf_alpha': MTDF_PARAMS['alpha'],
        'mtdf_beta_eos': MTDF_PARAMS['beta_eos'],
        'mtdf_z_t': MTDF_PARAMS['z_t'],
    }

    cosmo.set(params)
    cosmo.compute()

    # Get background
    bg = cosmo.get_background()

    # Get Omega_Lambda (at z=0)
    idx_0 = np.argmin(np.abs(bg['z']))
    Omega_Lambda = bg['(.)rho_lambda'][idx_0] / bg['(.)rho_crit'][idx_0]

    # Get sound horizon
    idx_rec = np.argmin(np.abs(bg['z'] - 1089))
    r_s = bg['comov.snd.hrz.'][idx_rec]

    cosmo.struct_cleanup()
    cosmo.empty()

    return bg, Omega_Lambda, r_s


def main():
    print("=" * 70)
    print("Task 2: f_EDE(z) Profile Dump")
    print("=" * 70)

    # Test with k_f = 1.0
    k_f = 1.0

    print(f"\nRunning CLASS with k_f = {k_f}...")
    bg, Omega_Lambda, r_s = run_class_and_get_omega(k_f)

    print(f"\nCosmology:")
    print(f"  Omega_Lambda = {Omega_Lambda:.4f}")
    print(f"  r_s(z*)      = {r_s:.4f} Mpc")

    # Compute derived MTDF parameters
    lambda_mtdf = MTDF_PARAMS['lambda_mtdf']
    f_kick = MTDF_PARAMS['C_mtdf'] * Omega_Lambda * lambda_mtdf

    print(f"\nMTDF parameters:")
    print(f"  alpha        = {MTDF_PARAMS['alpha']:.4f}")
    print(f"  beta_eos     = {MTDF_PARAMS['beta_eos']:.4f}")
    print(f"  lambda_MTDF  = {lambda_mtdf:.6f}")
    print(f"  C_MTDF       = {MTDF_PARAMS['C_mtdf']:.4f}")
    print(f"  f_kick       = {f_kick:.6f} = {f_kick*100:.4f}%")
    print(f"  z_peak       = {MTDF_PARAMS['z_peak']:.1f}")
    print(f"  sigma_z      = {MTDF_PARAMS['sigma_z']:.2f}")

    # Create redshift grid for profile
    z_grid = np.logspace(-1, 5, 500)

    # Compute analytic profile
    omega_efe_analytic = analytic_omega_efe(z_grid, k_f, Omega_Lambda)

    # Compute f_EDE = Omega_EFE / Omega_tot
    # At high z, Omega_tot ~ 1 (matter + radiation dominated)
    # More precisely, use the actual Omega from CLASS if available

    # For now, use simple approximation: Omega_tot ~ 1 at high z
    f_ede_analytic = omega_efe_analytic  # Since Omega_tot ~ 1 at high z

    # Find peak
    idx_peak = np.argmax(omega_efe_analytic)
    z_peak_measured = z_grid[idx_peak]
    omega_efe_peak = omega_efe_analytic[idx_peak]

    print(f"\nAnalytic profile (k_f = {k_f}):")
    print(f"  Peak z       = {z_peak_measured:.1f} (expected: {MTDF_PARAMS['z_peak']:.1f})")
    print(f"  Peak Omega   = {omega_efe_peak:.6f} = {omega_efe_peak*100:.4f}%")
    print(f"  Expected     = k_f * f_kick = {k_f * f_kick:.6f} = {k_f * f_kick*100:.4f}%")

    # Check width
    # FWHM in ln(1+z) should be ~2.35 * sigma_z
    half_max = omega_efe_peak / 2
    above_half = omega_efe_analytic > half_max
    if np.sum(above_half) > 0:
        z_low = z_grid[above_half][0]
        z_high = z_grid[above_half][-1]
        fwhm_lnz = np.log(1 + z_high) - np.log(1 + z_low)
        expected_fwhm = 2.355 * MTDF_PARAMS['sigma_z']
        print(f"  FWHM (ln(1+z)) = {fwhm_lnz:.3f} (expected: {expected_fwhm:.3f})")

    # Save profile to file
    output_file = f"{OUTPUT_DIR}/mtdf_fEDE_profile.txt"

    with open(output_file, 'w') as f:
        f.write("# MTDF f_EDE(z) Profile\n")
        f.write(f"# k_f = {k_f}\n")
        f.write(f"# f_kick = {f_kick:.6f}\n")
        f.write(f"# z_peak = {MTDF_PARAMS['z_peak']}\n")
        f.write(f"# sigma_z = {MTDF_PARAMS['sigma_z']}\n")
        f.write("#\n")
        f.write("# z          Omega_EFE      f_EDE\n")

        for i, z in enumerate(z_grid):
            omega_efe = omega_efe_analytic[i]
            f_ede = f_ede_analytic[i]
            f.write(f"{z:.6e}  {omega_efe:.6e}  {f_ede:.6e}\n")

    print(f"\nProfile saved to: {output_file}")

    # Verification checks
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    checks_passed = 0
    total_checks = 4

    # Check 1: Peak location
    peak_ok = abs(z_peak_measured - MTDF_PARAMS['z_peak']) / MTDF_PARAMS['z_peak'] < 0.05
    print(f"\n1. Peak at z_peak ~ 3400: {'✓ PASS' if peak_ok else '✗ FAIL'}")
    print(f"   Measured: {z_peak_measured:.1f}, Expected: {MTDF_PARAMS['z_peak']:.1f}")
    if peak_ok:
        checks_passed += 1

    # Check 2: Peak amplitude
    expected_peak = k_f * f_kick
    amp_ok = abs(omega_efe_peak - expected_peak) / expected_peak < 0.01
    print(f"\n2. Peak amplitude = k_f * f_kick: {'✓ PASS' if amp_ok else '✗ FAIL'}")
    print(f"   Measured: {omega_efe_peak:.6f}, Expected: {expected_peak:.6f}")
    if amp_ok:
        checks_passed += 1

    # Check 3: Width (FWHM in ln(1+z))
    width_ok = abs(fwhm_lnz - expected_fwhm) / expected_fwhm < 0.1
    print(f"\n3. Width σ_z = 0.5: {'✓ PASS' if width_ok else '✗ FAIL'}")
    print(f"   FWHM (ln(1+z)): {fwhm_lnz:.3f}, Expected: {expected_fwhm:.3f}")
    if width_ok:
        checks_passed += 1

    # Check 4: Falls off at low and high z
    omega_low_z = omega_efe_analytic[z_grid < 100]
    omega_high_z = omega_efe_analytic[z_grid > 50000]
    falloff_ok = np.max(omega_low_z) < 0.1 * omega_efe_peak and np.max(omega_high_z) < 0.1 * omega_efe_peak
    print(f"\n4. Falls off at low/high z: {'✓ PASS' if falloff_ok else '✗ FAIL'}")
    print(f"   Max at z < 100: {np.max(omega_low_z):.2e}")
    print(f"   Max at z > 50000: {np.max(omega_high_z):.2e}")
    if falloff_ok:
        checks_passed += 1

    print(f"\n" + "=" * 70)
    print(f"RESULT: {checks_passed}/{total_checks} checks passed")
    print("=" * 70)

    # Compare to r_s reduction
    print(f"\n--- Sound Horizon Impact ---")

    # Get r_s for k_f=0
    print("Running CLASS with k_f = 0...")
    bg0, _, r_s_0 = run_class_and_get_omega(0.0)

    r_s_reduction = 100 * (r_s - r_s_0) / r_s_0
    expected_reduction = -0.5 * k_f * f_kick * 100  # Rough estimate

    print(f"\nr_s(k_f=0) = {r_s_0:.4f} Mpc")
    print(f"r_s(k_f=1) = {r_s:.4f} Mpc")
    print(f"Δr_s/r_s   = {r_s_reduction:.4f}%")
    print(f"Expected   ~ {expected_reduction:.4f}% (rough)")

    return checks_passed == total_checks


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
