#!/usr/bin/env python3
"""
Hardening Test 5: Additional Environment Metrics

If the signal depends on one special metric construction, a referee can
dismiss it as a fitting artefact. This test shows the signal is robust
across multiple independent environment parameterisations:

  A. Signed distance (existing, normalised by void radius): (r - R)/R
  B. Physical distance to boundary in Mpc/h: r - R
  C. Normalised distance to centre: r/R (unsigned)
  D. Rank-based proximity (nonparametric)
  E. Binary in/out membership
  F. Inverse-distance weighting (sum of 1/d_norm over all nearby voids)

If they all point the same direction, the result does not depend on
one special construction.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from scipy import stats
from common import (
    PantheonData, standard_low_z_setup, compute_environment,
    compute_rank_metric, delta_chi2_test, gls_fit,
    save_results, CATALOGUE_GROUPS, COSMO_SN, sn_to_comoving
)


def compute_all_metrics(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r):
    """Compute all six environment metrics."""
    n_sn = len(sn_x)

    d_signed = np.full(n_sn, np.inf)
    d_phys = np.full(n_sn, np.inf)
    d_norm_centre = np.full(n_sn, np.inf)
    in_void = np.zeros(n_sn, dtype=bool)
    idw = np.zeros(n_sn)

    for i in range(n_sn):
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        d_normalised = dist / void_r
        d_sign = (dist - void_r) / void_r

        idx = np.argmin(d_normalised)
        d_signed[i] = d_sign[idx]
        d_phys[i] = dist[idx] - void_r[idx]
        d_norm_centre[i] = d_normalised[idx]
        in_void[i] = np.any(dist < void_r)

        # IDW: sum of 1/d_norm for all voids within 3 R_void
        nearby = d_normalised < 3.0
        if np.any(nearby):
            idw[i] = np.sum(1.0 / d_normalised[nearby])

    # Rank metric
    d_rank = compute_rank_metric(d_signed)

    # Binary (0 = inside, 1 = outside)
    binary = (~in_void).astype(float)

    return {
        'signed_distance': d_signed,
        'physical_mpc': d_phys,
        'normalised_centre': d_norm_centre,
        'rank': d_rank,
        'binary': binary,
        'inverse_distance_weight': -idw,  # Negative so "more void" = more negative
    }


def run_alt_metrics(pantheon, catalogue_name):
    """Test all metrics for one void catalogue."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(pantheon, catalogue_name)

    metrics = compute_all_metrics(sn_x, sn_y, sn_z, vx, vy, vz, vr)

    mu = pantheon.mu[idx]
    z = pantheon.z[idx]
    hm = pantheon.host_mass[idx]

    print(f"\n  {catalogue_name} (N={len(idx)} SNe)")

    results = {'catalogue': catalogue_name, 'n_sn': len(idx), 'metrics': {}}

    for metric_name, metric_values in metrics.items():
        r = delta_chi2_test(mu, z, metric_values, hm, cov_sub)
        results['metrics'][metric_name] = r
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        print(f"    {metric_name:<28} Dchi2={r['delta_chi2']:6.3f}  "
              f"gamma={r['gamma_env']:+.5f}  p={r['p_value']:.4f} {sig}")

    # Consistency check: do all metrics agree on sign of gamma?
    signs = {k: np.sign(v['gamma_env']) for k, v in results['metrics'].items()}
    # For signed_distance, physical_mpc, rank (d_signed < 0 inside void):
    # positive gamma means "inside/nearer voids = brighter" (consistent with stress depletion)
    # For binary (0 inside, 1 outside): positive gamma means "outside = dimmer, inside = brighter" (same direction)
    # For IDW: already flipped (-idw) so more voids nearby = more negative; positive gamma same direction
    results['sign_consistency'] = len(set(signs.values())) == 1
    results['signs'] = {k: int(v) for k, v in signs.items()}

    return results


def run(output_dir):
    print("=" * 70)
    print("HARDENING TEST 5: Alternative Environment Metrics")
    print("=" * 70)

    pantheon = PantheonData()
    all_results = {}

    for cat_name in CATALOGUE_GROUPS:
        all_results[cat_name] = run_alt_metrics(pantheon, cat_name)

    save_results(all_results, 'test5_alt_environment_metrics.json', output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Alternative Metrics Consistency")
    print("-" * 70)
    for cat_name, res in all_results.items():
        consistent = "YES" if res['sign_consistency'] else "NO"
        print(f"  {cat_name}: all metrics same sign = {consistent}")
        print(f"    Signs: {res['signs']}")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')
    run(out)
