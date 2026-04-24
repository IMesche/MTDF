#!/usr/bin/env python3
"""
Task 1: ΛCDM Limit Check

Compare three CLASS configurations:
1. Vanilla CLASS (no MTDF module)
2. MTDF CLASS with k_f=0 and MTDF disabled
3. MTDF CLASS with k_f=0

Verify that all three give identical results (within 0.1%).
"""

import numpy as np
import subprocess
import os
import sys

# Paths
VANILLA_CLASS = str(Path(__file__).parent.parent.parent / 'class_vanilla' / 'class')
MTDF_CLASS = str(Path(__file__).parent.parent / 'class')
OUTPUT_DIR = str(Path(__file__).parent.parent / 'output')

# Standard cosmological parameters (Planck 2018 best-fit)
COSMO_PARAMS = {
    'H0': 67.27,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
    'tau_reio': 0.0544,
}

def create_ini_file(filepath, params, mtdf_enabled=False, mtdf_kf=0.0, mtdf_efe='no', mtdf_growth='no'):
    """Create a CLASS .ini file."""
    with open(filepath, 'w') as f:
        f.write("# CLASS configuration for LCDM limit check\n\n")

        # Output settings
        f.write("output = tCl,pCl,lCl,mPk\n")
        f.write("lensing = yes\n")
        f.write("l_max_scalars = 2500\n")
        f.write("P_k_max_h/Mpc = 1.0\n")
        f.write("z_pk = 0.0\n")

        # Cosmological parameters
        f.write(f"\nH0 = {params['H0']}\n")
        f.write(f"omega_b = {params['omega_b']}\n")
        f.write(f"omega_cdm = {params['omega_cdm']}\n")
        f.write(f"A_s = {params['A_s']}\n")
        f.write(f"n_s = {params['n_s']}\n")
        f.write(f"tau_reio = {params['tau_reio']}\n")

        # MTDF settings (only for MTDF CLASS)
        if mtdf_enabled:
            f.write(f"\nmtdf = yes\n")
            f.write(f"mtdf_efe = {mtdf_efe}\n")
            f.write(f"mtdf_growth = {mtdf_growth}\n")
            f.write(f"mtdf_k_f = {mtdf_kf}\n")
            f.write(f"mtdf_alpha = 1.30\n")
            f.write(f"mtdf_beta_eos = 0.573\n")
            f.write(f"mtdf_z_t = 0.74\n")

        # Output file settings
        basename = os.path.splitext(os.path.basename(filepath))[0]
        f.write(f"\nroot = {OUTPUT_DIR}/{basename}_\n")
        f.write("write background = yes\n")
        f.write("write thermodynamics = yes\n")

    return filepath


def run_class(executable, ini_file, label):
    """Run CLASS and capture output."""
    print(f"\nRunning {label}...")
    result = subprocess.run(
        [executable, ini_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR in {label}:")
        print(result.stderr)
        return False

    # Parse key outputs from stdout
    output = result.stdout

    # Look for sound horizon, D_M, theta_s
    results = {}

    for line in output.split('\n'):
        if 'r_s(z_rec)' in line or 'sound horizon' in line.lower():
            try:
                results['r_s'] = float(line.split()[-1])
            except:
                pass
        if 'D_A(z_rec)' in line or 'angular diameter' in line.lower():
            try:
                results['D_A'] = float(line.split()[-1])
            except:
                pass
        if '100*theta_s' in line:
            try:
                results['100theta_s'] = float(line.split()[-1])
            except:
                pass

    return results


def read_cl_file(filepath):
    """Read Cl from CLASS output file."""
    if not os.path.exists(filepath):
        return None

    data = np.loadtxt(filepath)
    ell = data[:, 0]

    # Column order depends on output type
    # For lensed Cls: ell, TT, EE, TE, phiphi, Tphi
    if data.shape[1] >= 4:
        cl_tt = data[:, 1]
        cl_ee = data[:, 2]
        cl_te = data[:, 3]
        return {'ell': ell, 'tt': cl_tt, 'ee': cl_ee, 'te': cl_te}

    return None


def read_background_file(filepath):
    """Read background quantities from CLASS output."""
    if not os.path.exists(filepath):
        return None

    # Read header to get column names
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    # This is a column definition line
                    continue
                else:
                    header_line = line
            else:
                break

    data = np.loadtxt(filepath)

    return data


def compare_cls(cl1, cl2, label1, label2, max_ell=2000):
    """Compare two Cl arrays and report differences."""
    if cl1 is None or cl2 is None:
        print(f"Cannot compare - missing data")
        return False

    # Trim to max_ell
    mask1 = cl1['ell'] <= max_ell
    mask2 = cl2['ell'] <= max_ell

    results = {}
    all_good = True

    for key in ['tt', 'ee', 'te']:
        if key not in cl1 or key not in cl2:
            continue

        c1 = cl1[key][mask1]
        c2 = cl2[key][mask2]

        # Ensure same length
        min_len = min(len(c1), len(c2))
        c1 = c1[:min_len]
        c2 = c2[:min_len]

        # Compute relative difference where Cl is significant
        # Avoid division by very small numbers
        significant = np.abs(c1) > 1e-20 * np.max(np.abs(c1))
        if np.sum(significant) == 0:
            continue

        rel_diff = np.abs((c1 - c2) / c1)[significant]
        max_diff = np.max(rel_diff) * 100  # percent
        mean_diff = np.mean(rel_diff) * 100

        results[key] = {
            'max_diff_percent': max_diff,
            'mean_diff_percent': mean_diff,
        }

        status = "OK" if max_diff < 0.1 else "FAIL"
        if max_diff >= 0.1:
            all_good = False

        print(f"  C_ell^{key.upper()}: max diff = {max_diff:.4f}%, mean diff = {mean_diff:.4f}% [{status}]")

    return all_good


def main():
    print("=" * 70)
    print("Task 1: ΛCDM Limit Check")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create ini files
    ini_vanilla = f"{OUTPUT_DIR}/test_vanilla.ini"
    ini_mtdf_off = f"{OUTPUT_DIR}/test_mtdf_off.ini"
    ini_mtdf_kf0 = f"{OUTPUT_DIR}/test_mtdf_kf0.ini"

    create_ini_file(ini_vanilla, COSMO_PARAMS, mtdf_enabled=False)
    create_ini_file(ini_mtdf_off, COSMO_PARAMS, mtdf_enabled=True, mtdf_kf=0.0,
                    mtdf_efe='no', mtdf_growth='no')
    create_ini_file(ini_mtdf_kf0, COSMO_PARAMS, mtdf_enabled=True, mtdf_kf=0.0,
                    mtdf_efe='yes', mtdf_growth='no')

    # Run each configuration
    print("\n" + "-" * 50)
    print("Running CLASS configurations...")
    print("-" * 50)

    # 1. Vanilla CLASS
    result1 = run_class(VANILLA_CLASS, ini_vanilla, "Case 1: Vanilla CLASS")

    # 2. MTDF CLASS with MTDF disabled
    result2 = run_class(MTDF_CLASS, ini_mtdf_off, "Case 2: MTDF CLASS (mtdf=no)")

    # 3. MTDF CLASS with k_f=0
    result3 = run_class(MTDF_CLASS, ini_mtdf_kf0, "Case 3: MTDF CLASS (k_f=0, efe=yes)")

    # Read Cl files
    cl_vanilla = read_cl_file(f"{OUTPUT_DIR}/test_vanilla_cl_lensed.dat")
    cl_mtdf_off = read_cl_file(f"{OUTPUT_DIR}/test_mtdf_off_cl_lensed.dat")
    cl_mtdf_kf0 = read_cl_file(f"{OUTPUT_DIR}/test_mtdf_kf0_cl_lensed.dat")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n--- Case 2 (MTDF disabled) vs Case 1 (Vanilla) ---")
    test1_ok = compare_cls(cl_mtdf_off, cl_vanilla, "MTDF off", "Vanilla")

    print("\n--- Case 3 (MTDF k_f=0) vs Case 1 (Vanilla) ---")
    test2_ok = compare_cls(cl_mtdf_kf0, cl_vanilla, "MTDF k_f=0", "Vanilla")

    print("\n--- Case 3 (MTDF k_f=0) vs Case 2 (MTDF disabled) ---")
    test3_ok = compare_cls(cl_mtdf_kf0, cl_mtdf_off, "MTDF k_f=0", "MTDF off")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if test1_ok and test2_ok and test3_ok:
        print("\n✓ All tests PASSED - MTDF implementation has clean ΛCDM limit")
    else:
        print("\n✗ Some tests FAILED - MTDF is leaking into ΛCDM case")
        print("  Need to investigate residual differences")

    # Also compare using Python classy for more detailed diagnostics
    print("\n" + "-" * 50)
    print("Detailed comparison using classy Python interface...")
    print("-" * 50)

    try:
        import classy

        # Run MTDF CLASS with k_f=0
        cosmo = classy.Class()
        params = {
            'output': 'tCl,pCl,lCl',
            'lensing': 'yes',
            'l_max_scalars': 2500,
            'H0': COSMO_PARAMS['H0'],
            'omega_b': COSMO_PARAMS['omega_b'],
            'omega_cdm': COSMO_PARAMS['omega_cdm'],
            'A_s': COSMO_PARAMS['A_s'],
            'n_s': COSMO_PARAMS['n_s'],
            'tau_reio': COSMO_PARAMS['tau_reio'],
            'mtdf': 'yes',
            'mtdf_efe': 'yes',
            'mtdf_growth': 'no',
            'mtdf_k_f': 0.0,
            'mtdf_alpha': 1.30,
            'mtdf_beta_eos': 0.573,
            'mtdf_z_t': 0.74,
        }
        cosmo.set(params)
        cosmo.compute()

        # Get thermodynamics
        thermo = cosmo.get_thermodynamics()

        # Get background
        bg = cosmo.get_background()

        # Find z_star (recombination)
        idx_rec = np.argmin(np.abs(bg['z'] - 1089))

        r_s_mtdf_kf0 = bg['comov.snd.hrz.'][idx_rec]
        D_M_mtdf_kf0 = bg['comov. dist.'][idx_rec]

        print(f"\nMTDF k_f=0 results:")
        print(f"  r_s(z*)    = {r_s_mtdf_kf0:.4f} Mpc")
        print(f"  D_M(z*)    = {D_M_mtdf_kf0:.4f} Mpc")
        print(f"  100*θ_s    = {100 * r_s_mtdf_kf0 / D_M_mtdf_kf0:.6f}")

        cosmo.struct_cleanup()
        cosmo.empty()

        # Run with MTDF efe=no for comparison
        cosmo2 = classy.Class()
        params2 = params.copy()
        params2['mtdf_efe'] = 'no'
        cosmo2.set(params2)
        cosmo2.compute()

        bg2 = cosmo2.get_background()
        idx_rec2 = np.argmin(np.abs(bg2['z'] - 1089))

        r_s_mtdf_off = bg2['comov.snd.hrz.'][idx_rec2]
        D_M_mtdf_off = bg2['comov. dist.'][idx_rec2]

        print(f"\nMTDF efe=no results:")
        print(f"  r_s(z*)    = {r_s_mtdf_off:.4f} Mpc")
        print(f"  D_M(z*)    = {D_M_mtdf_off:.4f} Mpc")
        print(f"  100*θ_s    = {100 * r_s_mtdf_off / D_M_mtdf_off:.6f}")

        cosmo2.struct_cleanup()
        cosmo2.empty()

        # Compare
        r_s_diff = 100 * (r_s_mtdf_kf0 - r_s_mtdf_off) / r_s_mtdf_off
        D_M_diff = 100 * (D_M_mtdf_kf0 - D_M_mtdf_off) / D_M_mtdf_off

        print(f"\nDifference (k_f=0 vs efe=no):")
        print(f"  Δr_s/r_s   = {r_s_diff:.4f}%")
        print(f"  ΔD_M/D_M   = {D_M_diff:.4f}%")

        if abs(r_s_diff) < 0.01 and abs(D_M_diff) < 0.01:
            print("\n✓ k_f=0 gives identical r_s and D_M to MTDF disabled")
        else:
            print("\n⚠ Small residual difference detected")

    except Exception as e:
        print(f"Error in classy comparison: {e}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
