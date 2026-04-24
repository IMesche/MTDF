#!/usr/bin/env python3
"""
Hardening Test 2: Fake Transition Redshift Scan

MTDF predicts a sharp signal cutoff near z ~ 0.04, set by the coherence
length beta = 22.7 Mpc. This test checks whether equally sharp "transitions"
appear at arbitrary nearby z values.

Method:
  For each candidate z_cut in [0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
  0.055, 0.060, 0.070, 0.080]:
    - Split sample at z_cut
    - Fit piecewise model: gamma_env_low (z < z_cut), gamma_env_high (z >= z_cut)
    - Compare to constant-gamma model via Dchi2

If z ~ 0.04 is genuinely special, it should produce the largest Dchi2
for the piecewise split. If random z values produce comparable splits,
the "transition" is not physically meaningful.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from scipy import stats
from common import (
    PantheonData, standard_low_z_setup, compute_environment,
    gls_fit, delta_chi2_test, save_results, CATALOGUE_GROUPS, COSMO_SN
)


Z_CUTS = [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070, 0.080]


def piecewise_dchi2(mu, z, env_metric, host_mass, cov, z_cut):
    """
    Compare constant-gamma vs piecewise-gamma at z_cut.

    Constant model: intercept + gamma_env * d + gamma_M * step(M*)
    Piecewise model: intercept + gamma_low * d * I(z<z_cut)
                               + gamma_high * d * I(z>=z_cut) + gamma_M * step(M*)

    Returns Dchi2 (1 extra dof), p-value, and fitted coefficients.
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    # Constant model (3 params: intercept, gamma_env, gamma_M)
    X_const = np.column_stack([np.ones(n), env_metric, mass_step])
    _, _, chi2_const, _ = gls_fit(residual, X_const, cov)

    # Piecewise model (4 params: intercept, gamma_low, gamma_high, gamma_M)
    low_mask = (z < z_cut).astype(float)
    high_mask = (z >= z_cut).astype(float)
    env_low = env_metric * low_mask
    env_high = env_metric * high_mask
    X_piece = np.column_stack([np.ones(n), env_low, env_high, mass_step])
    beta_p, beta_cov_p, chi2_piece, _ = gls_fit(residual, X_piece, cov)

    dchi2 = chi2_const - chi2_piece
    p = 1 - stats.chi2.cdf(dchi2, 1)  # 1 extra dof

    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())

    return {
        'z_cut': z_cut,
        'n_low': n_low,
        'n_high': n_high,
        'delta_chi2': float(dchi2),
        'p_value': float(p),
        'gamma_low': float(beta_p[1]),
        'gamma_low_err': float(np.sqrt(beta_cov_p[1, 1])),
        'gamma_high': float(beta_p[2]),
        'gamma_high_err': float(np.sqrt(beta_cov_p[2, 2])),
        'gamma_ratio': float(beta_p[1] / beta_p[2]) if abs(beta_p[2]) > 1e-10 else None,
    }


def run_fake_z_scan(pantheon, catalogue_name):
    """Scan all candidate z_cuts for one void catalogue."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(pantheon, catalogue_name)

    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)

    # Also get the baseline (constant model) signal
    baseline = delta_chi2_test(
        pantheon.mu[idx], pantheon.z[idx], d_signed,
        pantheon.host_mass[idx], cov_sub)

    print(f"\n  {catalogue_name}: baseline Dchi2 = {baseline['delta_chi2']:.3f}")

    scan_results = []
    for z_cut in Z_CUTS:
        r = piecewise_dchi2(
            pantheon.mu[idx], pantheon.z[idx], d_signed,
            pantheon.host_mass[idx], cov_sub, z_cut)
        scan_results.append(r)
        marker = " <<<" if z_cut == 0.040 else ""
        print(f"    z_cut={z_cut:.3f}: Dchi2={r['delta_chi2']:6.3f}, "
              f"gamma_low={r['gamma_low']:+.4f}, "
              f"gamma_high={r['gamma_high']:+.4f}, "
              f"p={r['p_value']:.4f}{marker}")

    # Find the z_cut with maximum Dchi2
    best = max(scan_results, key=lambda x: x['delta_chi2'])

    return {
        'catalogue': catalogue_name,
        'n_sn': len(idx),
        'baseline': baseline,
        'scan': scan_results,
        'best_z_cut': best['z_cut'],
        'best_delta_chi2': best['delta_chi2'],
        'mtdf_predicted_z_cut': 0.040,
        'mtdf_z_cut_rank': sorted(
            [r['delta_chi2'] for r in scan_results], reverse=True
        ).index(
            next(r['delta_chi2'] for r in scan_results if r['z_cut'] == 0.040)
        ) + 1,
    }


def run(output_dir):
    print("=" * 70)
    print("HARDENING TEST 2: Fake Transition Redshift Scan")
    print("=" * 70)

    pantheon = PantheonData()
    all_results = {}

    for cat_name in CATALOGUE_GROUPS:
        all_results[cat_name] = run_fake_z_scan(pantheon, cat_name)

    save_results(all_results, 'test2_fake_z_transition.json', output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Fake Transition Redshift Scan")
    print("-" * 70)
    for cat_name, res in all_results.items():
        print(f"  {cat_name}: best z_cut = {res['best_z_cut']:.3f} "
              f"(Dchi2 = {res['best_delta_chi2']:.3f}), "
              f"MTDF-predicted z=0.04 rank = #{res['mtdf_z_cut_rank']}/{len(Z_CUTS)}")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')
    run(out)
