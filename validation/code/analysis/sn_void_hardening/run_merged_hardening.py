#!/usr/bin/env python3
"""
MTDF SN x Void Hardening Suite  - Merged Sample Edition
=======================================================

Runs all 6 hardening tests on the merged Pantheon+ / ZTF DR2 / Foundation
sample (~4200 SNe, ~3300 at z < 0.157).

Uses multiprocessing for Monte Carlo and leave-one-out loops.

Key difference from Pantheon-only run:
  - Diagonal covariance (no off-diagonal systematics for ZTF/Foundation)
  - ~3x more SNe in the critical z < 0.04 range

Usage:
  python run_merged_hardening.py              # All tests
  python run_merged_hardening.py --test 1 2   # Specific tests
  python run_merged_hardening.py --ncpu 8     # Set CPU count

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import sys
import os
import time
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))

from common import (
    MergedData, standard_low_z_setup, compute_environment,
    compute_rank_metric, delta_chi2_test, gls_fit,
    save_results, CATALOGUE_GROUPS, COSMO_SN, sn_to_comoving,
    load_void_pair
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening_merged')
NCPU = max(1, cpu_count() - 1)


# ================================================================
#  TEST 1: Scrambled Void Geometry (parallelised)
# ================================================================

def _scramble_one(args):
    """Worker for parallel scramble test."""
    seed, strategy, mu, z, hm, cov, sn_x, sn_y, sn_z, vx, vy, vz, vr = args
    rng = np.random.default_rng(seed)

    if strategy == 'rotation':
        A = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        coords = np.column_stack([vx, vy, vz])
        rot = (Q @ coords.T).T
        svx, svy, svz, svr = rot[:, 0], rot[:, 1], rot[:, 2], vr.copy()
    elif strategy == 'radii_shuffle':
        svx, svy, svz = vx.copy(), vy.copy(), vz.copy()
        svr = rng.permutation(vr)
    else:  # synthetic
        n = len(vx)
        lo = np.array([vx.min(), vy.min(), vz.min()])
        hi = np.array([vx.max(), vy.max(), vz.max()])
        pos = rng.uniform(lo, hi, size=(n, 3))
        svx, svy, svz = pos[:, 0], pos[:, 1], pos[:, 2]
        svr = rng.permutation(vr)

    d_scr, _, _, _ = compute_environment(sn_x, sn_y, sn_z, svx, svy, svz, svr)
    r = delta_chi2_test(mu, z, d_scr, hm, cov)
    return r['delta_chi2']


def run_test1_scrambled(data, catalogue_name, n_real=500, ncpu=NCPU):
    """Scrambled void null test with multiprocessing."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(data, catalogue_name)
    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    mu, z, hm = data.mu[idx], data.z[idx], data.host_mass[idx]

    obs = delta_chi2_test(mu, z, d_signed, hm, cov_sub)
    obs_dchi2 = obs['delta_chi2']
    print(f"\n  {catalogue_name} (N={len(idx)}): observed Dchi2 = {obs_dchi2:.3f}")

    results = {'catalogue': catalogue_name, 'n_sn': len(idx), 'observed': obs}

    for strategy in ['rotation', 'radii_shuffle', 'synthetic']:
        print(f"    {strategy} ({n_real} realisations, {ncpu} CPUs)...", end='', flush=True)
        args = [(42 + i, strategy, mu, z, hm, cov_sub,
                 sn_x, sn_y, sn_z, vx, vy, vz, vr) for i in range(n_real)]
        with Pool(ncpu) as pool:
            null_dchi2 = np.array(pool.map(_scramble_one, args))
        p = float(np.mean(null_dchi2 >= obs_dchi2))
        results[strategy] = {
            'p_scrambled': p,
            'null_mean': float(null_dchi2.mean()),
            'null_std': float(null_dchi2.std()),
            'observed_exceeds_pct': float((1 - p) * 100),
        }
        print(f" p = {p:.4f}")

    return results


# ================================================================
#  TEST 2: Fake z-transition scan (already fast, no MP needed)
# ================================================================

Z_CUTS = [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070, 0.080]


def run_test2_fake_z(data, catalogue_name):
    """Scan candidate z_cuts for piecewise model improvement."""
    from scipy import stats as sp_stats
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(data, catalogue_name)
    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    mu, z, hm = data.mu[idx], data.z[idx], data.host_mass[idx]

    baseline = delta_chi2_test(mu, z, d_signed, hm, cov_sub)
    print(f"\n  {catalogue_name} (N={len(idx)}): baseline Dchi2 = {baseline['delta_chi2']:.3f}")

    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (hm >= 10).astype(float)
    n = len(mu)

    # Constant model chi2
    X_const = np.column_stack([np.ones(n), d_signed, mass_step])
    _, _, chi2_const, _ = gls_fit(residual, X_const, cov_sub)

    scan = []
    for z_cut in Z_CUTS:
        low = (z < z_cut).astype(float)
        high = (z >= z_cut).astype(float)
        X_piece = np.column_stack([np.ones(n), d_signed * low, d_signed * high, mass_step])
        beta_p, beta_cov_p, chi2_p, _ = gls_fit(residual, X_piece, cov_sub)
        dchi2 = chi2_const - chi2_p
        p = 1 - sp_stats.chi2.cdf(dchi2, 1)
        r = {
            'z_cut': z_cut, 'delta_chi2': float(dchi2), 'p_value': float(p),
            'gamma_low': float(beta_p[1]), 'gamma_low_err': float(np.sqrt(beta_cov_p[1, 1])),
            'gamma_high': float(beta_p[2]), 'gamma_high_err': float(np.sqrt(beta_cov_p[2, 2])),
            'n_low': int(low.sum()), 'n_high': int(high.sum()),
        }
        scan.append(r)
        marker = " <<<" if z_cut == 0.040 else ""
        print(f"    z={z_cut:.3f}: Dchi2={dchi2:6.3f}, g_low={r['gamma_low']:+.4f}, "
              f"g_high={r['gamma_high']:+.4f}, p={p:.4f}{marker}")

    best = max(scan, key=lambda x: x['delta_chi2'])
    rank_04 = sorted([r['delta_chi2'] for r in scan], reverse=True).index(
        next(r['delta_chi2'] for r in scan if r['z_cut'] == 0.040)) + 1

    return {
        'catalogue': catalogue_name, 'n_sn': len(idx), 'baseline': baseline,
        'scan': scan, 'best_z_cut': best['z_cut'], 'best_delta_chi2': best['delta_chi2'],
        'mtdf_z_cut_rank': rank_04,
    }


# ================================================================
#  TEST 3: Population controls (x1, c)
# ================================================================

def run_test3_population(data, catalogue_name):
    """Test gamma_env stability under x1/c controls."""
    from scipy import stats as sp_stats
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(data, catalogue_name)
    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    mu, z, hm = data.mu[idx], data.z[idx], data.host_mass[idx]
    x1, c = data.x1[idx], data.c[idx]

    baseline = delta_chi2_test(mu, z, d_signed, hm, cov_sub)
    with_x1c = delta_chi2_test(mu, z, d_signed, hm, cov_sub,
                                extra_covariates=[x1, c], extra_names=['x1', 'c'])

    shift = abs(with_x1c['gamma_env'] - baseline['gamma_env']) / baseline['gamma_env_err'] \
        if baseline['gamma_env_err'] > 0 else None

    print(f"\n  {catalogue_name} (N={len(idx)}): "
          f"baseline g={baseline['gamma_env']:.4f}, "
          f"w/ x1+c g={with_x1c['gamma_env']:.4f}, shift={shift:.2f}s")

    # Population demographics
    in_void = d_signed < 0
    pop = {}
    for name, arr in [('x1', x1), ('c', c), ('host_mass', hm)]:
        if in_void.sum() > 2 and (~in_void).sum() > 2:
            t, p = sp_stats.ttest_ind(arr[in_void], arr[~in_void], equal_var=False)
        else:
            t, p = np.nan, np.nan
        pop[name] = {
            'in_void_mean': float(np.nanmean(arr[in_void])),
            'out_void_mean': float(np.nanmean(arr[~in_void])),
            'welch_p': float(p),
        }

    # Low-z survival
    z_mask = z < 0.04
    if z_mask.sum() > 20:
        idx_low = np.where(z_mask)[0]
        cov_low = cov_sub[np.ix_(idx_low, idx_low)]
        bl_low = delta_chi2_test(mu[idx_low], z[idx_low], d_signed[idx_low],
                                  hm[idx_low], cov_low)
        ctrl_low = delta_chi2_test(mu[idx_low], z[idx_low], d_signed[idx_low],
                                    hm[idx_low], cov_low,
                                    extra_covariates=[x1[idx_low], c[idx_low]])
        low_z = {'n': len(idx_low),
                 'baseline_gamma': bl_low['gamma_env'],
                 'controlled_gamma': ctrl_low['gamma_env']}
    else:
        low_z = {'n': int(z_mask.sum()), 'status': 'too_few'}

    return {
        'catalogue': catalogue_name, 'n_sn': len(idx),
        'baseline': baseline, 'with_x1_and_c': with_x1c,
        'gamma_shift_sigma': float(shift) if shift else None,
        'demographics': pop, 'low_z_survival': low_z,
    }


# ================================================================
#  TEST 4: Wrong-sign metric (with parallel random noise)
# ================================================================

def _random_noise_one(args):
    """Worker for parallel random metric test."""
    seed, mu, z, hm, cov, d_mean, d_std = args
    rng = np.random.default_rng(seed)
    d_rand = rng.normal(d_mean, d_std, size=len(mu))
    r = delta_chi2_test(mu, z, d_rand, hm, cov)
    return r['delta_chi2']


def run_test4_wrong_sign(data, catalogue_name, n_random=200, ncpu=NCPU):
    """Wrong-sign and centroid-only metric tests."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(data, catalogue_name)
    d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    mu, z, hm = data.mu[idx], data.z[idx], data.host_mass[idx]

    baseline = delta_chi2_test(mu, z, d_signed, hm, cov_sub)
    inverted = delta_chi2_test(mu, z, -d_signed, hm, cov_sub)

    # Centroid-only
    n_sn = len(sn_x)
    d_centroid = np.full(n_sn, np.inf)
    for i in range(n_sn):
        dist = np.sqrt((sn_x[i] - vx)**2 + (sn_y[i] - vy)**2 + (sn_z[i] - vz)**2)
        d_centroid[i] = np.min(dist)
    d_centroid = (d_centroid - d_centroid.mean()) / d_centroid.std()
    centroid = delta_chi2_test(mu, z, d_centroid, hm, cov_sub)

    # Parallel random noise
    d_mean, d_std = float(np.mean(d_signed)), float(np.std(d_signed))
    args = [(42 + i, mu, z, hm, cov_sub, d_mean, d_std) for i in range(n_random)]
    with Pool(ncpu) as pool:
        rand_dchi2 = np.array(pool.map(_random_noise_one, args))
    p_rand = float(np.mean(rand_dchi2 >= baseline['delta_chi2']))

    print(f"\n  {catalogue_name} (N={len(idx)}): "
          f"real={baseline['delta_chi2']:.3f}, inv={inverted['delta_chi2']:.3f}, "
          f"centroid={centroid['delta_chi2']:.3f}, rand_mean={rand_dchi2.mean():.3f}")

    return {
        'catalogue': catalogue_name, 'n_sn': len(idx),
        'real_signed_distance': baseline,
        'inverted_metric': inverted,
        'centroid_only': centroid,
        'random_noise': {'null_mean': float(rand_dchi2.mean()), 'p_random': p_rand},
    }


# ================================================================
#  TEST 5: Alternative metrics (already fast)
# ================================================================

def run_test5_alt_metrics(data, catalogue_name):
    """Test all environment metric variants."""
    idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr = \
        standard_low_z_setup(data, catalogue_name)
    mu, z, hm = data.mu[idx], data.z[idx], data.host_mass[idx]

    d_signed, d_phys, _, in_void = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
    d_rank = compute_rank_metric(d_signed)
    binary = (~in_void).astype(float)

    # IDW
    n_sn = len(sn_x)
    idw = np.zeros(n_sn)
    for i in range(n_sn):
        dist = np.sqrt((sn_x[i] - vx)**2 + (sn_y[i] - vy)**2 + (sn_z[i] - vz)**2)
        d_norm = dist / vr
        nearby = d_norm < 3.0
        if np.any(nearby):
            idw[i] = np.sum(1.0 / d_norm[nearby])

    metrics = {
        'signed_distance': d_signed,
        'physical_mpc': d_phys,
        'rank': d_rank,
        'binary': binary,
        'inverse_distance_weight': -idw,
    }

    print(f"\n  {catalogue_name} (N={len(idx)})")
    results = {'catalogue': catalogue_name, 'n_sn': len(idx), 'metrics': {}}
    for name, vals in metrics.items():
        r = delta_chi2_test(mu, z, vals, hm, cov_sub)
        results['metrics'][name] = r
        sig = "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        print(f"    {name:<28} Dchi2={r['delta_chi2']:6.3f} p={r['p_value']:.4f} {sig}")

    signs = {k: int(np.sign(v['gamma_env'])) for k, v in results['metrics'].items()}
    results['sign_consistency'] = len(set(signs.values())) == 1
    results['signs'] = signs
    return results


# ================================================================
#  TEST 6: Cross-catalogue overlap (parallelised LOO)
# ================================================================

def _loo_one(args):
    """Worker for parallel leave-one-out."""
    i, mu, z, env, hm, cov = args
    n = len(mu)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    idx = np.where(mask)[0]
    r = delta_chi2_test(mu[idx], z[idx], env[idx], hm[idx], cov[np.ix_(idx, idx)])
    return r['delta_chi2']


def run_test6_cross_cat(data, ncpu=NCPU):
    """Cross-catalogue SN overlap diagnostic with parallel LOO."""
    idx_base, cov_base = data.apply_cuts(z_pv_cut=0.02, z_max=0.157)
    sn_x, sn_y, sn_z = sn_to_comoving(data.z[idx_base], data.ra[idx_base], data.dec[idx_base])
    mu = data.mu[idx_base]
    z = data.z[idx_base]
    hm = data.host_mass[idx_base]

    env_metrics = {}
    contributions = {}
    rank_indices = {}

    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            continue
        d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
        env_metrics[cat_name] = d_signed

        full = delta_chi2_test(mu, z, d_signed, hm, cov_base)
        full_dchi2 = full['delta_chi2']

        print(f"\n  {cat_name}: LOO on {len(idx_base)} SNe ({ncpu} CPUs)...", end='', flush=True)
        args = [(i, mu, z, d_signed, hm, cov_base) for i in range(len(idx_base))]
        with Pool(ncpu) as pool:
            loo_dchi2 = np.array(pool.map(_loo_one, args))

        contribs = full_dchi2 - loo_dchi2
        contributions[cat_name] = contribs
        sorted_idx = np.argsort(-contribs)
        rank_indices[cat_name] = sorted_idx

        top5 = contribs[sorted_idx[:5]].sum()
        print(f" Dchi2={full_dchi2:.3f}, top-5 contribute {top5:.3f} ({top5/full_dchi2*100:.1f}%)")

    # Overlap
    from itertools import combinations
    overlap_results = {}
    for top_n in [10, 20, 30]:
        top_sets = {k: set(v[:top_n]) for k, v in rank_indices.items()}
        overlap = {}
        cat_names = list(rank_indices.keys())
        for a, b in combinations(cat_names, 2):
            o = top_sets[a] & top_sets[b]
            overlap[f"{a}_vs_{b}"] = {'count': len(o), 'fraction': len(o) / top_n}
        if len(cat_names) == 3:
            three = top_sets[cat_names[0]] & top_sets[cat_names[1]] & top_sets[cat_names[2]]
            overlap['three_way'] = {'count': len(three), 'fraction': len(three) / top_n}
        overlap_results[f'top_{top_n}'] = overlap
        print(f"  Top-{top_n}: " + ", ".join(f"{k}={v['count']}/{top_n}" for k, v in overlap.items()))

    return {
        'n_sn': len(idx_base),
        'overlap': overlap_results,
        'top5_indices': {k: v[:5].tolist() for k, v in rank_indices.items()},
    }


# ================================================================
#  MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='MTDF Hardening Suite (Merged Sample)')
    parser.add_argument('--test', nargs='+', type=int, default=None)
    parser.add_argument('--ncpu', type=int, default=NCPU)
    args = parser.parse_args()

    tests = args.test or [1, 2, 3, 4, 5, 6]
    ncpu = args.ncpu

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("MTDF SN x Void Hardening Suite  - MERGED SAMPLE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Tests: {tests}, CPUs: {ncpu}")
    print("=" * 70)

    data = MergedData()
    all_results = {'sample': 'merged', 'n_total': len(data.z)}
    total_start = time.time()

    for t in tests:
        t_start = time.time()
        try:
            if t == 1:
                r = {}
                for cat in CATALOGUE_GROUPS:
                    r[cat] = run_test1_scrambled(data, cat, ncpu=ncpu)
                all_results[f'test_{t}'] = r
            elif t == 2:
                r = {}
                for cat in CATALOGUE_GROUPS:
                    r[cat] = run_test2_fake_z(data, cat)
                all_results[f'test_{t}'] = r
            elif t == 3:
                r = {}
                for cat in CATALOGUE_GROUPS:
                    r[cat] = run_test3_population(data, cat)
                all_results[f'test_{t}'] = r
            elif t == 4:
                r = {}
                for cat in CATALOGUE_GROUPS:
                    r[cat] = run_test4_wrong_sign(data, cat, ncpu=ncpu)
                all_results[f'test_{t}'] = r
            elif t == 5:
                r = {}
                for cat in CATALOGUE_GROUPS:
                    r[cat] = run_test5_alt_metrics(data, cat)
                all_results[f'test_{t}'] = r
            elif t == 6:
                all_results[f'test_{t}'] = run_test6_cross_cat(data, ncpu=ncpu)
        except Exception as e:
            print(f"\n  *** Test {t} FAILED: {e} ***")
            import traceback
            traceback.print_exc()
            all_results[f'test_{t}'] = {'error': str(e)}

        elapsed = time.time() - t_start
        print(f"\n  [Test {t} done in {elapsed:.1f}s]")

    save_results(all_results, 'merged_hardening_results.json', OUTPUT_DIR)

    total = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE. Total: {total:.0f}s. Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
