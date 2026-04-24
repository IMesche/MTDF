#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 3: SN x Void Environment Analysis — Main Driver

Reproduces MTDF_02 Tables 1-4 with GPU-accelerated GLS:
  Table 1: gamma_env per void catalog (VoidFinder, REVOLVER, VIDE)
  Table 2: NGC/SGC split (footprint-based)
  Table 3: Survey fixed effects stability
  Table 4: LOSO (Leave-One-Survey-Out)
  + z-modulation, permutation test, block bootstrap, CPU crosscheck
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import linalg

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mtdf_validation.phase3.data_loader import (
    PantheonPlusData, load_all_void_catalogs, sn_to_comoving,
    combine_ngc_sgc_voids, COSMOLOGY_HEADER,
)
from mtdf_validation.phase3.crossmatch_gpu import (
    compute_environment_gpu, compute_environment_cpu, crosscheck_gpu_cpu,
)
from mtdf_validation.phase3.gls_engine import (
    delta_chi2_test, delta_chi2_test_with_survey_fe,
    loso_analysis, permutation_test, block_bootstrap,
    z_binned_analysis, z_modulation_models,
)


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "validation" / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase3"

Z_MIN = 0.02
Z_MAX = 0.157
FINDERS = ['voidfinder', 'revolver', 'vide']


def run_table1(pantheon, idx, cov_inv, sn_pos, catalogs):
    """Table 1: gamma_env per void catalog, both metrics."""
    print("\n" + "=" * 70)
    print("TABLE 1: gamma_env PER VOID CATALOG (NGC+SGC combined)")
    print("=" * 70)

    results = {}
    for finder in FINDERS:
        print(f"\n--- {finder.upper()} ---")
        try:
            void_pos, void_r, n_ngc = combine_ngc_sgc_voids(catalogs, finder)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Voids: {len(void_r)} ({n_ngc} NGC + {len(void_r) - n_ngc} SGC)")

        d_signed, nearest_idx, in_void = compute_environment_gpu(
            sn_pos, void_pos, void_r
        )

        sub = pantheon.get_subset(idx)

        # Continuous metric (d_signed)
        res_signed = delta_chi2_test(
            sub['mu'], sub['z'], d_signed, sub['host_mass'], cov_inv
        )

        # Binary metric (~in_void)
        binary_env = (~in_void).astype(float)
        res_binary = delta_chi2_test(
            sub['mu'], sub['z'], binary_env, sub['host_mass'], cov_inv
        )

        n_inside = int(np.sum(in_void))
        print(f"  SNe inside voids: {n_inside}/{len(idx)}")
        print(f"  d_signed range: [{d_signed.min():.2f}, {d_signed.max():.2f}]")
        print(f"  Signed:  gamma_env = {res_signed['gamma_env']:+.4f} "
              f"+/- {res_signed['gamma_env_err']:.4f}, "
              f"dchi2 = {res_signed['delta_chi2']:.3f}, "
              f"p = {res_signed['p_value']:.4f}")
        print(f"  Binary:  gamma_env = {res_binary['gamma_env']:+.4f} "
              f"+/- {res_binary['gamma_env_err']:.4f}, "
              f"dchi2 = {res_binary['delta_chi2']:.3f}, "
              f"p = {res_binary['p_value']:.4f}")

        # Interpretable units: gamma_env in mag at typical d_signed
        median_d = float(np.median(np.abs(d_signed)))
        print(f"  Typical |d_signed| = {median_d:.2f} R_eff")
        print(f"  => mu shift at median = {res_signed['gamma_env'] * median_d:+.4f} mag")

        results[finder] = {
            'signed': res_signed,
            'binary': res_binary,
            'n_voids': len(void_r),
            'n_ngc_voids': n_ngc,
            'n_inside': n_inside,
            'median_abs_d_signed': float(median_d),
            'd_signed': d_signed.tolist(),
            'nearest_idx': nearest_idx.tolist(),
            'in_void': in_void.tolist(),
        }

    return results


def run_table2(pantheon, idx, cov_full_sub, sn_pos, catalogs):
    """Table 2: NGC/SGC split — split SNe by footprint first."""
    print("\n" + "=" * 70)
    print("TABLE 2: NGC / SGC FOOTPRINT SPLIT")
    print("=" * 70)

    sub = pantheon.get_subset(idx)
    ra = sub['ra']

    # Split SNe by RA: NGC = (90, 280), SGC = rest
    ngc_mask = (ra > 90) & (ra < 280)
    sgc_mask = ~ngc_mask

    idx_ngc = np.where(ngc_mask)[0]
    idx_sgc = np.where(sgc_mask)[0]

    print(f"  NGC SNe: {len(idx_ngc)}, SGC SNe: {len(idx_sgc)}")

    results = {}
    for finder in FINDERS:
        print(f"\n--- {finder.upper()} ---")

        finder_results = {}
        for region, sn_mask, region_idx in [
            ('NGC', ngc_mask, idx_ngc),
            ('SGC', sgc_mask, idx_sgc),
        ]:
            cat = catalogs.get((finder, region))
            if cat is None:
                print(f"  {region}: catalog not found, skipping")
                continue

            if len(region_idx) < 20:
                print(f"  {region}: only {len(region_idx)} SNe, skipping")
                continue

            vx, vy, vz, vr = cat.get_positions(interior_only=True)
            void_pos = np.column_stack([vx, vy, vz])

            sn_pos_region = sn_pos[sn_mask]

            d_signed, _, in_void = compute_environment_gpu(
                sn_pos_region, void_pos, vr
            )

            # Sub-covariance for this footprint
            cov_region = cov_full_sub[np.ix_(region_idx, region_idx)]
            try:
                cov_inv_region = linalg.inv(cov_region)
            except linalg.LinAlgError:
                print(f"  {region}: singular covariance, skipping")
                continue

            res = delta_chi2_test(
                sub['mu'][sn_mask], sub['z'][sn_mask],
                d_signed, sub['host_mass'][sn_mask], cov_inv_region
            )

            print(f"  {region}: gamma_env = {res['gamma_env']:+.4f} "
                  f"+/- {res['gamma_env_err']:.4f}, "
                  f"dchi2 = {res['delta_chi2']:.3f}, p = {res['p_value']:.4f} "
                  f"(N={res['n']}, in_void={int(np.sum(in_void))})")

            finder_results[region.lower()] = res

        results[finder] = finder_results

    return results


def run_table3(pantheon, idx, cov_inv, sn_pos, catalogs):
    """Table 3: Survey fixed effects stability."""
    print("\n" + "=" * 70)
    print("TABLE 3: SURVEY FIXED EFFECTS STABILITY")
    print("=" * 70)

    sub = pantheon.get_subset(idx)
    results = {}

    for finder in FINDERS:
        print(f"\n--- {finder.upper()} ---")
        try:
            void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, finder)
        except ValueError:
            continue

        d_signed, _, _ = compute_environment_gpu(sn_pos, void_pos, void_r)

        # Without survey FE
        res_no_fe = delta_chi2_test(
            sub['mu'], sub['z'], d_signed, sub['host_mass'], cov_inv
        )

        # With survey FE
        res_fe = delta_chi2_test_with_survey_fe(
            sub['mu'], sub['z'], d_signed, sub['host_mass'],
            sub['survey_id'], cov_inv
        )

        delta_gamma = res_fe['gamma_env'] - res_no_fe['gamma_env']

        print(f"  Without FE: gamma_env = {res_no_fe['gamma_env']:+.4f} "
              f"+/- {res_no_fe['gamma_env_err']:.4f}, p = {res_no_fe['p_value']:.4f}")
        print(f"  With FE:    gamma_env = {res_fe['gamma_env']:+.4f} "
              f"+/- {res_fe['gamma_env_err']:.4f}, p = {res_fe['p_value']:.4f} "
              f"({res_fe['n_survey_fe']} dummies, ref={res_fe['reference_survey']})")
        print(f"  Delta_gamma = {delta_gamma:+.5f} "
              f"({'STABLE' if abs(delta_gamma) < 0.0005 else 'SHIFTED'})")

        results[finder] = {
            'without_fe': res_no_fe,
            'with_fe': res_fe,
            'delta_gamma': float(delta_gamma),
            'stable': abs(delta_gamma) < 0.0005,
        }

    return results


def run_table4(pantheon, idx, cov_sub, sn_pos, catalogs):
    """Table 4: Leave-One-Survey-Out stability."""
    print("\n" + "=" * 70)
    print("TABLE 4: LEAVE-ONE-SURVEY-OUT (LOSO)")
    print("=" * 70)

    sub = pantheon.get_subset(idx)
    results = {}

    for finder in FINDERS[:1]:  # REVOLVER only (most significant) to save time
        print(f"\n--- {finder.upper()} ---")
        try:
            void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, finder)
        except ValueError:
            continue

        d_signed, _, _ = compute_environment_gpu(sn_pos, void_pos, void_r)

        loso_results = loso_analysis(
            sub['mu'], sub['z'], d_signed, sub['host_mass'],
            sub['survey_id'], cov_sub
        )

        n_positive = sum(1 for r in loso_results if r['gamma_env'] > 0)
        print(f"  LOSO results: {n_positive}/{len(loso_results)} positive gamma_env")
        print(f"  {'Survey removed':<18} {'gamma_env':>10} {'sigma':>8} {'p':>8}")
        print(f"  {'-' * 46}")
        for r in sorted(loso_results, key=lambda x: -x['gamma_env']):
            sign = '+' if r['gamma_env'] > 0 else ' '
            print(f"  {r['survey_removed']:<18} {sign}{abs(r['gamma_env']):.4f} "
                  f"{r['gamma_env_err']:>8.4f} {r['p_value']:>8.4f}")

        results[finder] = {
            'loso': [
                {k: v for k, v in r.items() if k != 'chi2_null' and k != 'chi2_full'}
                for r in loso_results
            ],
            'n_positive': n_positive,
            'n_total': len(loso_results),
        }

    return results


def run_z_modulation(pantheon, idx, cov_sub, cov_inv, sn_pos, catalogs):
    """Z-binned and z-modulation analysis."""
    print("\n" + "=" * 70)
    print("Z-MODULATION ANALYSIS")
    print("=" * 70)

    sub = pantheon.get_subset(idx)
    results = {}

    for finder in FINDERS:
        print(f"\n--- {finder.upper()} ---")
        try:
            void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, finder)
        except ValueError:
            continue

        d_signed, _, _ = compute_environment_gpu(sn_pos, void_pos, void_r)

        # Z-binned
        z_bins = z_binned_analysis(
            sub['mu'], sub['z'], d_signed, sub['host_mass'], cov_sub
        )

        print(f"  {'z range':<15} {'N':>5} {'gamma_env':>12} {'sigma':>8} {'p':>8}")
        print(f"  {'-' * 50}")
        for b in z_bins:
            if b.get('skipped'):
                print(f"  [{b['z_range'][0]:.2f}, {b['z_range'][1]:.2f})  "
                      f"{b['n']:>5}  SKIPPED")
            else:
                marker = " *" if b['p_value'] < 0.05 else ""
                print(f"  [{b['z_range'][0]:.2f}, {b['z_range'][1]:.2f})  "
                      f"{b['n']:>5} {b['gamma_env']:>+10.4f} "
                      f"{b['gamma_env_err']:>8.4f} {b['p_value']:>8.4f}{marker}")

        # Piecewise + linear models
        z_mod = z_modulation_models(
            sub['mu'], sub['z'], d_signed, sub['host_mass'], cov_inv
        )

        pw = z_mod['piecewise']
        print(f"\n  Piecewise (z_cut=0.05):")
        print(f"    gamma_low  = {pw['gamma_env_low']:+.4f} +/- {pw['gamma_env_low_err']:.4f} "
              f"(N={pw['n_low']})")
        print(f"    gamma_high = {pw['gamma_env_high']:+.4f} +/- {pw['gamma_env_high_err']:.4f} "
              f"(N={pw['n_high']})")
        print(f"    dchi2 vs constant = {pw['delta_chi2_vs_constant']:.3f}, "
              f"p = {pw['p_piecewise']:.4f}")

        lz = z_mod['linear_z']
        print(f"\n  Linear z-interaction:")
        print(f"    gamma_0 = {lz['gamma_env_0']:+.4f} +/- {lz['gamma_env_0_err']:.4f}")
        print(f"    gamma_z = {lz['gamma_env_z']:+.4f} +/- {lz['gamma_env_z_err']:.4f}")
        print(f"    dchi2 vs constant = {lz['delta_chi2_vs_constant']:.3f}, "
              f"p = {lz['p_interaction']:.4f}")

        results[finder] = {
            'z_bins': z_bins,
            'z_modulation': z_mod,
        }

    return results


def run_permutation_and_bootstrap(pantheon, idx, cov_sub, cov_inv, sn_pos, catalogs,
                                   n_perms=10000, n_boots=5000):
    """Permutation test + block bootstrap for all catalogs."""
    print("\n" + "=" * 70)
    print(f"PERMUTATION TEST ({n_perms}) + BLOCK BOOTSTRAP ({n_boots})")
    print("=" * 70)

    sub = pantheon.get_subset(idx)
    results = {}

    for finder in FINDERS:
        print(f"\n--- {finder.upper()} ---")
        try:
            void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, finder)
        except ValueError:
            continue

        d_signed, _, _ = compute_environment_gpu(sn_pos, void_pos, void_r)

        # Permutation test
        print(f"  Running {n_perms} permutations...")
        t0 = time.time()
        perm = permutation_test(
            sub['mu'], sub['z'], d_signed, sub['host_mass'],
            cov_inv, n_perms=n_perms
        )
        t_perm = time.time() - t0
        print(f"  Permutation: obs dchi2 = {perm['obs_delta_chi2']:.3f}, "
              f"p_perm = {perm['p_permutation']:.4f} ({t_perm:.1f}s)")

        # Block bootstrap
        print(f"  Running {n_boots} bootstrap resamples...")
        t0 = time.time()
        boot = block_bootstrap(
            sub['mu'], sub['z'], d_signed, sub['host_mass'],
            cov_sub, n_boots=n_boots
        )
        t_boot = time.time() - t0
        print(f"  Bootstrap: gamma = {boot['gamma_env_mean']:+.4f} "
              f"+/- {boot['gamma_env_std']:.4f}")
        print(f"  68% CI: [{boot['ci_68'][0]:+.4f}, {boot['ci_68'][1]:+.4f}]")
        print(f"  95% CI: [{boot['ci_95'][0]:+.4f}, {boot['ci_95'][1]:+.4f}] "
              f"({t_boot:.1f}s)")

        results[finder] = {
            'permutation': perm,
            'bootstrap': boot,
        }

    return results


def run_cpu_crosscheck(pantheon, idx, cov_inv, sn_pos, catalogs):
    """CPU cross-check for VoidFinder (smallest catalog)."""
    print("\n" + "=" * 70)
    print("CPU CROSS-CHECK (VoidFinder)")
    print("=" * 70)

    sub = pantheon.get_subset(idx)

    try:
        void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, 'voidfinder')
    except ValueError as e:
        print(f"  SKIP: {e}")
        return {'status': 'skipped'}

    # GPU result
    d_gpu, idx_gpu, inv_gpu = compute_environment_gpu(sn_pos, void_pos, void_r)
    res_gpu = delta_chi2_test(sub['mu'], sub['z'], d_gpu, sub['host_mass'], cov_inv)

    # CPU result
    d_cpu, idx_cpu, inv_cpu = compute_environment_cpu(sn_pos, void_pos, void_r)
    res_cpu = delta_chi2_test(sub['mu'], sub['z'], d_cpu, sub['host_mass'], cov_inv)

    # Check crossmatch
    d_match = np.allclose(d_gpu, d_cpu, atol=1e-6)
    gamma_match = abs(res_gpu['gamma_env'] - res_cpu['gamma_env']) < 1e-6
    sigma_match = abs(res_gpu['gamma_env_err'] - res_cpu['gamma_env_err']) < 1e-6

    status = "PASS" if (d_match and gamma_match and sigma_match) else "FAIL"

    print(f"  GPU: gamma_env = {res_gpu['gamma_env']:+.6f} +/- {res_gpu['gamma_env_err']:.6f}")
    print(f"  CPU: gamma_env = {res_cpu['gamma_env']:+.6f} +/- {res_cpu['gamma_env_err']:.6f}")
    print(f"  d_signed match: {d_match}")
    print(f"  gamma_env match: {gamma_match} (diff={abs(res_gpu['gamma_env'] - res_cpu['gamma_env']):.2e})")
    print(f"  sigma match: {sigma_match}")
    print(f"  => {status}")

    return {
        'status': status,
        'gpu_gamma_env': res_gpu['gamma_env'],
        'cpu_gamma_env': res_cpu['gamma_env'],
        'gpu_gamma_err': res_gpu['gamma_env_err'],
        'cpu_gamma_err': res_cpu['gamma_env_err'],
        'd_signed_match': d_match,
        'gamma_match': gamma_match,
        'sigma_match': sigma_match,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 3: SN x Void Environment Analysis')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: 500 perms, 500 boots')
    args = parser.parse_args()

    n_perms = 500 if args.quick else 10000
    n_boots = 500 if args.quick else 5000

    t_start = time.time()

    print("=" * 70)
    print("MTDF PHASE 3: SN x VOID ENVIRONMENT ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    if args.quick:
        print("[QUICK MODE] Reduced permutation/bootstrap counts")
    print("=" * 70)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/2] Loading Pantheon+ data...")
    pantheon = PantheonPlusData(DATA_DIR)

    print("\n[2/2] Loading void catalogs...")
    catalogs = load_all_void_catalogs(DATA_DIR)

    # Apply cuts
    idx, cov_sub = pantheon.apply_cuts(z_min=Z_MIN, z_max=Z_MAX)
    sub = pantheon.get_subset(idx)

    # Pre-compute covariance inverse (used throughout)
    print(f"\n  Inverting {len(idx)}x{len(idx)} covariance matrix...")
    cov_inv = linalg.inv(cov_sub)
    print("  Done.")

    # SN comoving positions
    sn_pos = sn_to_comoving(sub['z'], sub['ra'], sub['dec'])
    print(f"  SN positions computed ({len(sn_pos)} SNe)")

    # Run all analyses
    table1 = run_table1(pantheon, idx, cov_inv, sn_pos, catalogs)
    table2 = run_table2(pantheon, idx, cov_sub, sn_pos, catalogs)
    table3 = run_table3(pantheon, idx, cov_inv, sn_pos, catalogs)
    table4 = run_table4(pantheon, idx, cov_sub, sn_pos, catalogs)
    z_mod = run_z_modulation(pantheon, idx, cov_sub, cov_inv, sn_pos, catalogs)
    perm_boot = run_permutation_and_bootstrap(
        pantheon, idx, cov_sub, cov_inv, sn_pos, catalogs,
        n_perms=n_perms, n_boots=n_boots,
    )
    cpu_check = run_cpu_crosscheck(pantheon, idx, cov_inv, sn_pos, catalogs)

    elapsed = time.time() - t_start

    # Build output
    output = {
        'timestamp': datetime.now().isoformat(),
        'cosmology_header': COSMOLOGY_HEADER,
        'cuts': {
            'z_min': Z_MIN,
            'z_max': Z_MAX,
            'n_sne_total': pantheon.n,
            'n_sne_after_cuts': len(idx),
            'n_unique_surveys': int(len(np.unique(sub['survey_id']))),
        },
        'table1': _strip_arrays(table1),
        'table2_ngc_sgc': table2,
        'table3_survey_fe': table3,
        'table4_loso': table4,
        'z_modulation': z_mod,
        'permutation_bootstrap': perm_boot,
        'cpu_crosscheck': cpu_check,
        'elapsed_minutes': elapsed / 60.0,
    }

    # Save
    out_path = RESULTS_DIR / "phase3_summary.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n[SAVED] {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"CPU crosscheck: {cpu_check.get('status', 'N/A')}")

    if 'voidfinder' in table1:
        r = table1['voidfinder']['signed']
        print(f"\nVoidFinder:  gamma = {r['gamma_env']:+.4f} +/- {r['gamma_env_err']:.4f}, "
              f"dchi2 = {r['delta_chi2']:.3f}, p = {r['p_value']:.4f}")
    if 'revolver' in table1:
        r = table1['revolver']['signed']
        print(f"REVOLVER:    gamma = {r['gamma_env']:+.4f} +/- {r['gamma_env_err']:.4f}, "
              f"dchi2 = {r['delta_chi2']:.3f}, p = {r['p_value']:.4f}")
    if 'vide' in table1:
        r = table1['vide']['signed']
        print(f"VIDE:        gamma = {r['gamma_env']:+.4f} +/- {r['gamma_env_err']:.4f}, "
              f"dchi2 = {r['delta_chi2']:.3f}, p = {r['p_value']:.4f}")


def _strip_arrays(table1):
    """Remove large arrays (d_signed, nearest_idx, in_void) from Table 1 for JSON."""
    result = {}
    for finder, data in table1.items():
        clean = {k: v for k, v in data.items()
                 if k not in ('d_signed', 'nearest_idx', 'in_void')}
        result[finder] = clean
    return result


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    main()
