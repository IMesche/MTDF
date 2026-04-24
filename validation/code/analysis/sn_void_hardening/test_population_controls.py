#!/usr/bin/env python3
"""
Hardening Test 3: Stretch (x1) and Colour (c) Population Controls

The existing pipeline controls for host-mass but not for SALT2 stretch (x1)
or colour (c). If voids preferentially host SNe with unusual stretch/colour,
the environment signal could be a population artefact.

Tests:
  A. Add x1 and c as covariates in the full GLS model
     (if gamma_env survives, it is not absorbed by population differences)
  B. Matched-subsample analysis: restrict to SNe in overlapping x1/c ranges
     between void-interior and void-exterior populations
  C. z < 0.04 cutoff survival: repeat A and B in the low-z bin only

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from scipy import stats
from common import (
    PantheonData, standard_low_z_setup, compute_environment,
    delta_chi2_test, gls_fit, save_results, CATALOGUE_GROUPS, COSMO_SN
)


def test_with_x1c_covariates(pantheon, idx, cov_sub, d_signed):
    """Test A: Add x1, c as covariates. Compare gamma_env with and without."""
    mu = pantheon.mu[idx]
    z = pantheon.z[idx]
    hm = pantheon.host_mass[idx]
    x1 = pantheon.x1[idx]
    c = pantheon.c[idx]

    # Baseline: no x1/c control (matches existing pipeline)
    baseline = delta_chi2_test(mu, z, d_signed, hm, cov_sub)

    # With x1 only
    with_x1 = delta_chi2_test(mu, z, d_signed, hm, cov_sub,
                               extra_covariates=[x1],
                               extra_names=['x1'])

    # With c only
    with_c = delta_chi2_test(mu, z, d_signed, hm, cov_sub,
                              extra_covariates=[c],
                              extra_names=['c'])

    # With both x1 and c
    with_x1c = delta_chi2_test(mu, z, d_signed, hm, cov_sub,
                                extra_covariates=[x1, c],
                                extra_names=['x1', 'c'])

    return {
        'baseline': baseline,
        'with_x1': with_x1,
        'with_c': with_c,
        'with_x1_and_c': with_x1c,
        'gamma_env_shift_x1c': float(
            abs(with_x1c['gamma_env'] - baseline['gamma_env'])
            / baseline['gamma_env_err']
        ) if baseline['gamma_env_err'] > 0 else None,
    }


def test_matched_subsamples(pantheon, idx, cov_sub, d_signed):
    """
    Test B: Restrict to overlapping x1/c ranges between inside-void
    and outside-void populations.
    """
    in_void = d_signed < 0
    out_void = d_signed >= 0

    if in_void.sum() < 5 or out_void.sum() < 5:
        return {'status': 'insufficient_void_interior_SNe'}

    x1 = pantheon.x1[idx]
    c = pantheon.c[idx]

    # Find overlap range for x1
    x1_lo = max(np.percentile(x1[in_void], 10), np.percentile(x1[out_void], 10))
    x1_hi = min(np.percentile(x1[in_void], 90), np.percentile(x1[out_void], 90))

    # Find overlap range for c
    c_lo = max(np.percentile(c[in_void], 10), np.percentile(c[out_void], 10))
    c_hi = min(np.percentile(c[in_void], 90), np.percentile(c[out_void], 90))

    # Apply matching cuts
    match_mask = (x1 >= x1_lo) & (x1 <= x1_hi) & (c >= c_lo) & (c <= c_hi)
    matched_idx = np.where(match_mask)[0]

    if len(matched_idx) < 30:
        return {
            'status': 'too_few_matched_SNe',
            'n_matched': len(matched_idx),
        }

    # Extract matched sub-covariance
    cov_matched = cov_sub[np.ix_(matched_idx, matched_idx)]

    result = delta_chi2_test(
        pantheon.mu[idx[matched_idx]],
        pantheon.z[idx[matched_idx]],
        d_signed[matched_idx],
        pantheon.host_mass[idx[matched_idx]],
        cov_matched)

    return {
        'status': 'ok',
        'n_total': len(idx),
        'n_matched': len(matched_idx),
        'x1_range': [float(x1_lo), float(x1_hi)],
        'c_range': [float(c_lo), float(c_hi)],
        'n_in_void_matched': int((d_signed[matched_idx] < 0).sum()),
        'n_out_void_matched': int((d_signed[matched_idx] >= 0).sum()),
        'result': result,
    }


def test_low_z_cutoff_survival(pantheon, catalogue_name, z_cut=0.04):
    """
    Test C: Does the z < 0.04 signal survive x1/c controls?
    """
    idx_full, _, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(pantheon, catalogue_name)

    # Further restrict to z < z_cut
    z_mask = pantheon.z[idx_full] < z_cut
    idx_low = idx_full[z_mask]

    if len(idx_low) < 20:
        return {'status': f'too_few_SNe_below_z={z_cut}', 'n': len(idx_low)}

    cov_low = pantheon.cov_full[np.ix_(idx_low, idx_low)]
    sn_xl, sn_yl, sn_zl = sn_x[z_mask], sn_y[z_mask], sn_z[z_mask]
    d_signed, _, _, _ = compute_environment(sn_xl, sn_yl, sn_zl, vx, vy, vz, vr)

    mu = pantheon.mu[idx_low]
    z = pantheon.z[idx_low]
    hm = pantheon.host_mass[idx_low]
    x1 = pantheon.x1[idx_low]
    c = pantheon.c[idx_low]

    baseline = delta_chi2_test(mu, z, d_signed, hm, cov_low)
    with_x1c = delta_chi2_test(mu, z, d_signed, hm, cov_low,
                                extra_covariates=[x1, c],
                                extra_names=['x1', 'c'])

    return {
        'status': 'ok',
        'z_cut': z_cut,
        'n_sn': len(idx_low),
        'baseline': baseline,
        'with_x1_and_c': with_x1c,
        'gamma_env_shift': float(
            abs(with_x1c['gamma_env'] - baseline['gamma_env'])
            / baseline['gamma_env_err']
        ) if baseline['gamma_env_err'] > 0 else None,
    }


def population_summary(pantheon, idx, d_signed):
    """Report x1, c, host_mass distributions inside vs outside voids."""
    in_void = d_signed < 0
    out_void = ~in_void

    def describe(arr, mask):
        a = arr[mask]
        return {
            'n': int(mask.sum()),
            'mean': float(np.mean(a)),
            'std': float(np.std(a)),
            'median': float(np.median(a)),
        }

    x1, c, hm = pantheon.x1[idx], pantheon.c[idx], pantheon.host_mass[idx]

    result = {}
    for name, arr in [('x1', x1), ('c', c), ('host_mass', hm)]:
        d_in = describe(arr, in_void)
        d_out = describe(arr, out_void)
        # Welch t-test for difference
        if d_in['n'] > 2 and d_out['n'] > 2:
            t, p = stats.ttest_ind(arr[in_void], arr[out_void], equal_var=False)
        else:
            t, p = np.nan, np.nan
        result[name] = {
            'inside_void': d_in,
            'outside_void': d_out,
            'welch_t': float(t),
            'welch_p': float(p),
        }

    return result


def run_catalogue(pantheon, catalogue_name):
    """Run all population control tests for one catalogue."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(pantheon, catalogue_name)
    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)

    print(f"\n  {catalogue_name} (N={len(idx)} SNe)")

    # Population demographics
    pop = population_summary(pantheon, idx, d_signed)
    for var in ('x1', 'c', 'host_mass'):
        p = pop[var]
        print(f"    {var}: in-void mean={p['inside_void']['mean']:.3f}, "
              f"out-void mean={p['outside_void']['mean']:.3f}, "
              f"Welch p={p['welch_p']:.4f}")

    # Test A: x1/c covariates
    cov_test = test_with_x1c_covariates(pantheon, idx, cov_sub, d_signed)
    print(f"    Covariate control: baseline gamma={cov_test['baseline']['gamma_env']:.4f}, "
          f"with x1+c gamma={cov_test['with_x1_and_c']['gamma_env']:.4f}, "
          f"shift={cov_test['gamma_env_shift_x1c']:.2f} sigma")

    # Test B: matched subsamples
    match_test = test_matched_subsamples(pantheon, idx, cov_sub, d_signed)
    if match_test['status'] == 'ok':
        print(f"    Matched subsample: N={match_test['n_matched']}, "
              f"gamma={match_test['result']['gamma_env']:.4f}, "
              f"p={match_test['result']['p_value']:.4f}")
    else:
        print(f"    Matched subsample: {match_test['status']}")

    # Test C: low-z survival
    low_z = test_low_z_cutoff_survival(pantheon, catalogue_name)
    if low_z['status'] == 'ok':
        print(f"    Low-z (z<0.04): baseline gamma={low_z['baseline']['gamma_env']:.4f}, "
              f"with x1+c gamma={low_z['with_x1_and_c']['gamma_env']:.4f}")

    return {
        'catalogue': catalogue_name,
        'population_demographics': pop,
        'covariate_control': cov_test,
        'matched_subsample': match_test,
        'low_z_survival': low_z,
    }


def run(output_dir):
    print("=" * 70)
    print("HARDENING TEST 3: Stretch (x1) and Colour (c) Population Controls")
    print("=" * 70)

    pantheon = PantheonData()
    all_results = {}

    for cat_name in CATALOGUE_GROUPS:
        all_results[cat_name] = run_catalogue(pantheon, cat_name)

    save_results(all_results, 'test3_population_controls.json', output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Population Controls")
    print("-" * 70)
    print(f"{'Catalogue':<14} {'Baseline g':>11} {'w/ x1+c g':>11} "
          f"{'Shift (s)':>10} {'Matched p':>10}")
    print("-" * 70)
    for cat_name, res in all_results.items():
        cov = res['covariate_control']
        g_base = cov['baseline']['gamma_env']
        g_ctrl = cov['with_x1_and_c']['gamma_env']
        shift = cov['gamma_env_shift_x1c']
        m = res['matched_subsample']
        mp = m['result']['p_value'] if m['status'] == 'ok' else None
        mp_str = f"{mp:.4f}" if mp is not None else "N/A"
        print(f"{cat_name:<14} {g_base:11.4f} {g_ctrl:11.4f} "
              f"{shift:10.2f} {mp_str:>10}")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')
    run(out)
