#!/usr/bin/env python3
"""
Hardening Test 4: Wrong-Sign Metric (Cluster/Overdensity Proximity)

MTDF predicts that the SN environment signal arises from void stress
depletion. If we replace void-boundary distance with distance to the
nearest overdensity (cluster/filament proxy), the signal should vanish.

Method:
  A. Invert the void metric: use -d_signed (flip sign)
     This tests whether "closer to void centre" vs "farther from void centre"
     is the active direction.
  B. Use distance to void centre (ignoring radius) as a non-physical metric
  C. Random environment metric (Gaussian noise matched to d_signed distribution)
     as a pure noise baseline

If gamma_env is significant for the real signed distance but not for
inverted, centroid-only, or random metrics, the signal is specifically
tied to void boundary proximity.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from scipy import stats
from common import (
    PantheonData, standard_low_z_setup, compute_environment,
    delta_chi2_test, save_results, CATALOGUE_GROUPS, sn_to_comoving
)


def compute_centroid_distance(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r):
    """Distance to nearest void centre in Mpc/h (ignoring void radius)."""
    n_sn = len(sn_x)
    d_centroid = np.full(n_sn, np.inf)

    for i in range(n_sn):
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        d_centroid[i] = np.min(dist)

    # Normalize to zero mean, unit variance for comparability
    d_centroid = (d_centroid - np.mean(d_centroid)) / np.std(d_centroid)
    return d_centroid


def run_wrong_sign(pantheon, catalogue_name, n_random=200, seed=42):
    """Run all wrong-sign metric tests for one catalogue."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(pantheon, catalogue_name)

    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    mu = pantheon.mu[idx]
    z = pantheon.z[idx]
    hm = pantheon.host_mass[idx]

    # Real signal (baseline)
    baseline = delta_chi2_test(mu, z, d_signed, hm, cov_sub)
    print(f"\n  {catalogue_name}: real Dchi2 = {baseline['delta_chi2']:.3f}, "
          f"p = {baseline['p_value']:.4f}")

    # A. Inverted metric
    inverted = delta_chi2_test(mu, z, -d_signed, hm, cov_sub)
    print(f"    Inverted: Dchi2 = {inverted['delta_chi2']:.3f}, "
          f"gamma = {inverted['gamma_env']:+.4f}, p = {inverted['p_value']:.4f}")

    # B. Centroid-only distance (no boundary information)
    d_centroid = compute_centroid_distance(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    centroid = delta_chi2_test(mu, z, d_centroid, hm, cov_sub)
    print(f"    Centroid-only: Dchi2 = {centroid['delta_chi2']:.3f}, "
          f"gamma = {centroid['gamma_env']:+.4f}, p = {centroid['p_value']:.4f}")

    # C. Random Gaussian metric (matched mean/std to d_signed)
    rng = np.random.default_rng(seed)
    random_dchi2 = []
    for _ in range(n_random):
        d_rand = rng.normal(np.mean(d_signed), np.std(d_signed), size=len(d_signed))
        r = delta_chi2_test(mu, z, d_rand, hm, cov_sub)
        random_dchi2.append(r['delta_chi2'])
    random_dchi2 = np.array(random_dchi2)
    p_random = np.mean(random_dchi2 >= baseline['delta_chi2'])
    print(f"    Random noise: mean Dchi2 = {random_dchi2.mean():.3f}, "
          f"obs exceeds {(1-p_random)*100:.1f}% of randoms")

    return {
        'catalogue': catalogue_name,
        'n_sn': len(idx),
        'real_signed_distance': baseline,
        'inverted_metric': inverted,
        'centroid_only': centroid,
        'random_noise': {
            'n_realisations': n_random,
            'null_mean': float(random_dchi2.mean()),
            'null_std': float(random_dchi2.std()),
            'p_random': float(p_random),
        },
    }


def run(output_dir):
    print("=" * 70)
    print("HARDENING TEST 4: Wrong-Sign Metric")
    print("=" * 70)

    pantheon = PantheonData()
    all_results = {}

    for cat_name in CATALOGUE_GROUPS:
        all_results[cat_name] = run_wrong_sign(pantheon, cat_name)

    save_results(all_results, 'test4_wrong_sign_metric.json', output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Wrong-Sign Metric Test")
    print("-" * 70)
    print(f"{'Catalogue':<14} {'Real Dchi2':>10} {'Inverted':>10} "
          f"{'Centroid':>10} {'Rand mean':>10}")
    print("-" * 70)
    for cat_name, res in all_results.items():
        print(f"{cat_name:<14} "
              f"{res['real_signed_distance']['delta_chi2']:10.3f} "
              f"{res['inverted_metric']['delta_chi2']:10.3f} "
              f"{res['centroid_only']['delta_chi2']:10.3f} "
              f"{res['random_noise']['null_mean']:10.3f}")
    print("=" * 70)
    print("\nExpected: Real >> Inverted, Centroid, Random")

    return all_results


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')
    run(out)
