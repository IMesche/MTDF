#!/usr/bin/env python3
"""
Hardening Test 1: Scrambled Void Geometry Null Test

Keep SNe fixed. Randomise the void map (rotate on sky + jitter radii).
If the signal survives scrambled voids, it is not tied to real void geometry.
If it vanishes, the signal requires the actual cosmic void distribution.

Three scrambling strategies:
  A. Random rotation of the full void catalogue (preserves internal structure)
  B. Random reassignment of void radii (preserves positions)
  C. Fully synthetic voids (uniform random in survey volume)

For each: 500 realisations, compute Dchi2, compare to observed.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from scipy import stats
from common import (
    PantheonData, standard_low_z_setup, compute_environment,
    delta_chi2_test, save_results, CATALOGUE_GROUPS, sn_to_comoving,
    load_void_pair
)


def random_rotation_matrix(rng):
    """Generate a random 3D rotation matrix (uniform on SO(3))."""
    # QR decomposition of random Gaussian matrix
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def scramble_rotation(vx, vy, vz, vr, rng):
    """Rotate all void positions by a random rotation. Radii unchanged."""
    R = random_rotation_matrix(rng)
    coords = np.column_stack([vx, vy, vz])
    rotated = (R @ coords.T).T
    return rotated[:, 0], rotated[:, 1], rotated[:, 2], vr.copy()


def scramble_radii(vx, vy, vz, vr, rng):
    """Keep void positions fixed, randomly permute radii."""
    return vx.copy(), vy.copy(), vz.copy(), rng.permutation(vr)


def scramble_synthetic(vx, vy, vz, vr, rng):
    """Replace voids with uniform random positions in the same bounding volume."""
    n = len(vx)
    lo = np.array([vx.min(), vy.min(), vz.min()])
    hi = np.array([vx.max(), vy.max(), vz.max()])
    new_pos = rng.uniform(lo, hi, size=(n, 3))
    # Keep real radius distribution
    new_r = rng.permutation(vr)
    return new_pos[:, 0], new_pos[:, 1], new_pos[:, 2], new_r


STRATEGIES = {
    'rotation': scramble_rotation,
    'radii_shuffle': scramble_radii,
    'synthetic': scramble_synthetic,
}


def run_scrambled_null(pantheon, catalogue_name, n_realisations=500, seed=42):
    """Run all three scrambling strategies for one void catalogue."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(pantheon, catalogue_name)

    # Observed signal
    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    obs = delta_chi2_test(
        pantheon.mu[idx], pantheon.z[idx], d_signed,
        pantheon.host_mass[idx], cov_sub)
    obs_dchi2 = obs['delta_chi2']

    print(f"\n  {catalogue_name}: observed Dchi2 = {obs_dchi2:.3f}, "
          f"p = {obs['p_value']:.4f}")

    results = {
        'catalogue': catalogue_name,
        'n_sn': len(idx),
        'n_voids': len(vx),
        'observed': obs,
    }

    rng = np.random.default_rng(seed)

    for strategy_name, scramble_fn in STRATEGIES.items():
        print(f"    Strategy: {strategy_name} ({n_realisations} realisations)...",
              end='', flush=True)
        null_dchi2 = []
        for i in range(n_realisations):
            svx, svy, svz, svr = scramble_fn(vx, vy, vz, vr, rng)
            d_scr, _, _, _ = compute_environment(sn_x, sn_y, sn_z,
                                                  svx, svy, svz, svr)
            r = delta_chi2_test(
                pantheon.mu[idx], pantheon.z[idx], d_scr,
                pantheon.host_mass[idx], cov_sub)
            null_dchi2.append(r['delta_chi2'])

        null_dchi2 = np.array(null_dchi2)
        p_scrambled = np.mean(null_dchi2 >= obs_dchi2)

        results[strategy_name] = {
            'p_scrambled': float(p_scrambled),
            'null_mean': float(null_dchi2.mean()),
            'null_std': float(null_dchi2.std()),
            'null_max': float(null_dchi2.max()),
            'null_95': float(np.percentile(null_dchi2, 95)),
            'observed_exceeds_pct': float((1 - p_scrambled) * 100),
        }
        print(f" p = {p_scrambled:.4f}")

    return results


def run(output_dir):
    print("=" * 70)
    print("HARDENING TEST 1: Scrambled Void Geometry Null Test")
    print("=" * 70)

    pantheon = PantheonData()
    all_results = {}

    for cat_name in CATALOGUE_GROUPS:
        all_results[cat_name] = run_scrambled_null(pantheon, cat_name)

    save_results(all_results, 'test1_scrambled_voids.json', output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Scrambled Void Null Test")
    print("-" * 70)
    print(f"{'Catalogue':<14} {'Strategy':<16} {'Obs Dchi2':>10} "
          f"{'Null mean':>10} {'p_scram':>8}")
    print("-" * 70)
    for cat_name, res in all_results.items():
        obs_d = res['observed']['delta_chi2']
        for strat in STRATEGIES:
            s = res[strat]
            print(f"{cat_name:<14} {strat:<16} {obs_d:10.3f} "
                  f"{s['null_mean']:10.3f} {s['p_scrambled']:8.4f}")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')
    run(out)
