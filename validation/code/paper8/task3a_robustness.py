#!/usr/bin/env python3
"""
Task 3A Robustness Tests: LOSO and Malmquist bias checks for CF4 gamma_v.

Two tests to bring CF4 channel maturity closer to the SN pipeline:

  Test 1: Leave-One-Survey-Out (LOSO)
    CF4 groups are measured via TF, FP, SNIa, or calibrators.
    Remove each method in turn and check if gamma_v is stable.
    If the signal is driven by one method's systematics, it will
    weaken when that method is removed.

  Test 2: Malmquist Bias Diagnostics
    Inhomogeneous Malmquist bias creates distance-dependent systematics.
    Tests:
    (a) gamma_v in distance bins (does it concentrate at specific distances?)
    (b) gamma_v vs distance error (does it correlate with measurement quality?)
    (c) gamma_v with inverse-variance weighting (upweighting precise measurements)

Author: Ingo Mesche
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import load_void_pair, save_results, CATALOGUE_GROUPS

sys.path.insert(0, os.path.dirname(__file__))
from task3a_cosmicflows4_vpec import (
    load_cf4, groups_to_comoving, compute_vpec_residuals,
    compute_environment_fast
)
from task3a_lcdm_mock import fast_wls_gamma

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)
H0_FID = 75.0
Z_CUT = 0.04
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3a_robustness')


# ============================================================
# Test 1: Leave-One-Survey-Out (LOSO)
# ============================================================

def test_loso(groups, gx, gy, gz, d_signed, vpec_err, z_arr):
    """
    Remove groups measured by each distance method and re-measure gamma_v.
    """
    print("\n" + "=" * 60, flush=True)
    print("TEST 1: Leave-One-Survey-Out (LOSO)", flush=True)
    print("=" * 60, flush=True)

    methods = {
        'TF': 'o_tf',
        'FP': 'o_fp',
        'SNIa': 'o_snia',
        'Calibrators': 'o_cal',
    }

    # Compute observed vpec residuals
    vpec = compute_vpec_residuals(groups)
    w = 1.0 / vpec_err**2

    # Baseline: full sample
    gamma_full, gamma_err_full, dchi2_full = fast_wls_gamma(vpec, d_signed, w)
    sig_full = abs(gamma_full) / gamma_err_full
    print(f"\n  Baseline (all methods): gamma_v = {gamma_full:.2f} +/- {gamma_err_full:.2f} "
          f"({sig_full:.1f} sigma, n={len(vpec)})", flush=True)

    results = {
        'baseline': {
            'gamma_v': float(gamma_full),
            'gamma_v_err': float(gamma_err_full),
            'significance': float(sig_full),
            'dchi2': float(dchi2_full),
            'n': len(vpec),
        },
        'leave_out': {},
        'only_use': {},
    }

    # Leave-one-out
    print("\n  --- Leave-one-out ---", flush=True)
    for method_name, flag_key in methods.items():
        mask = np.array([not g[flag_key] for g in groups])
        n_removed = (~mask).sum()
        n_kept = mask.sum()

        if n_kept < 100:
            print(f"  Leave out {method_name}: too few remaining ({n_kept})", flush=True)
            continue

        gamma, gamma_err, dchi2 = fast_wls_gamma(
            vpec[mask], d_signed[mask], w[mask])
        sig = abs(gamma) / gamma_err if gamma_err > 0 else 0

        change = (gamma - gamma_full) / gamma_err_full * 100
        print(f"  Leave out {method_name} ({n_removed} removed, {n_kept} kept): "
              f"gamma_v = {gamma:.2f} +/- {gamma_err:.2f} ({sig:.1f} sigma) "
              f"[{change:+.1f}% shift]", flush=True)

        results['leave_out'][method_name] = {
            'gamma_v': float(gamma),
            'gamma_v_err': float(gamma_err),
            'significance': float(sig),
            'dchi2': float(dchi2),
            'n_removed': int(n_removed),
            'n_kept': int(n_kept),
            'shift_pct': float(change),
        }

    # Only-one (use ONLY that method)
    print("\n  --- Single-method ---", flush=True)
    for method_name, flag_key in methods.items():
        mask = np.array([bool(g[flag_key]) for g in groups])
        n_sel = mask.sum()

        if n_sel < 50:
            print(f"  Only {method_name}: too few ({n_sel})", flush=True)
            results['only_use'][method_name] = {'n': int(n_sel), 'too_few': True}
            continue

        gamma, gamma_err, dchi2 = fast_wls_gamma(
            vpec[mask], d_signed[mask], w[mask])
        sig = abs(gamma) / gamma_err if gamma_err > 0 else 0

        print(f"  Only {method_name} (n={n_sel}): "
              f"gamma_v = {gamma:.2f} +/- {gamma_err:.2f} ({sig:.1f} sigma)",
              flush=True)

        results['only_use'][method_name] = {
            'gamma_v': float(gamma),
            'gamma_v_err': float(gamma_err),
            'significance': float(sig),
            'dchi2': float(dchi2),
            'n': int(n_sel),
        }

    return results


# ============================================================
# Test 2: Malmquist Bias Diagnostics
# ============================================================

def test_malmquist(groups, gx, gy, gz, d_signed, vpec_err, z_arr):
    """
    Test whether gamma_v is driven by Malmquist bias.
    """
    print("\n" + "=" * 60, flush=True)
    print("TEST 2: Malmquist Bias Diagnostics", flush=True)
    print("=" * 60, flush=True)

    vpec = compute_vpec_residuals(groups)
    w = 1.0 / vpec_err**2
    dists = np.array([g['dist'] for g in groups])
    e_dmav = np.array([g['e_dmav'] for g in groups])

    results = {}

    # (a) gamma_v in distance bins
    print("\n  --- (a) Distance bins ---", flush=True)
    dist_edges = [0, 50, 100, 150, 200, 300, 500, 800]
    dist_bin_results = []

    for j in range(len(dist_edges) - 1):
        d_lo, d_hi = dist_edges[j], dist_edges[j + 1]
        mask = (dists >= d_lo) & (dists < d_hi)
        n = mask.sum()

        if n < 50:
            print(f"  d=[{d_lo},{d_hi}) Mpc: n={n} (too few)", flush=True)
            dist_bin_results.append({
                'd_lo': d_lo, 'd_hi': d_hi, 'n': int(n), 'too_few': True})
            continue

        gamma, gamma_err, dchi2 = fast_wls_gamma(
            vpec[mask], d_signed[mask], w[mask])
        sig = abs(gamma) / gamma_err if gamma_err > 0 else 0
        z_med = np.median(z_arr[mask])

        print(f"  d=[{d_lo},{d_hi}) Mpc (n={n}, z_med={z_med:.3f}): "
              f"gamma_v = {gamma:.2f} +/- {gamma_err:.2f} ({sig:.1f} sigma)",
              flush=True)

        dist_bin_results.append({
            'd_lo': d_lo, 'd_hi': d_hi, 'n': int(n),
            'z_median': float(z_med),
            'gamma_v': float(gamma),
            'gamma_v_err': float(gamma_err),
            'significance': float(sig),
            'dchi2': float(dchi2),
        })

    results['distance_bins'] = dist_bin_results

    # (b) gamma_v in error bins (e_dmav quartiles)
    print("\n  --- (b) Distance error (e_dmav) quartiles ---", flush=True)
    quartiles = np.percentile(e_dmav, [0, 25, 50, 75, 100])
    err_bin_results = []

    for j in range(len(quartiles) - 1):
        e_lo, e_hi = quartiles[j], quartiles[j + 1]
        mask = (e_dmav >= e_lo) & (e_dmav < e_hi + 1e-10)
        n = mask.sum()

        if n < 50:
            continue

        gamma, gamma_err, dchi2 = fast_wls_gamma(
            vpec[mask], d_signed[mask], w[mask])
        sig = abs(gamma) / gamma_err if gamma_err > 0 else 0

        print(f"  e_dmav=[{e_lo:.3f},{e_hi:.3f}) (n={n}): "
              f"gamma_v = {gamma:.2f} +/- {gamma_err:.2f} ({sig:.1f} sigma)",
              flush=True)

        err_bin_results.append({
            'e_lo': float(e_lo), 'e_hi': float(e_hi), 'n': int(n),
            'gamma_v': float(gamma),
            'gamma_v_err': float(gamma_err),
            'significance': float(sig),
        })

    results['error_quartiles'] = err_bin_results

    # (c) Correlation between d_signed and distance (Malmquist proxy)
    print("\n  --- (c) d_signed vs distance correlation ---", flush=True)
    corr_d_dist = np.corrcoef(d_signed, dists)[0, 1]
    corr_d_edm = np.corrcoef(d_signed, e_dmav)[0, 1]
    corr_d_z = np.corrcoef(d_signed, z_arr)[0, 1]

    print(f"  corr(d_signed, distance) = {corr_d_dist:.4f}", flush=True)
    print(f"  corr(d_signed, e_dmav)   = {corr_d_edm:.4f}", flush=True)
    print(f"  corr(d_signed, z)        = {corr_d_z:.4f}", flush=True)

    results['correlations'] = {
        'dsigned_distance': float(corr_d_dist),
        'dsigned_e_dmav': float(corr_d_edm),
        'dsigned_z': float(corr_d_z),
    }

    # (d) High-quality subsample (lowest 50% errors)
    print("\n  --- (d) High-quality subsample (lowest 50% e_dmav) ---", flush=True)
    e_median = np.median(e_dmav)
    hq_mask = e_dmav <= e_median
    n_hq = hq_mask.sum()

    gamma_hq, gamma_err_hq, dchi2_hq = fast_wls_gamma(
        vpec[hq_mask], d_signed[hq_mask], w[hq_mask])
    sig_hq = abs(gamma_hq) / gamma_err_hq if gamma_err_hq > 0 else 0

    gamma_lq, gamma_err_lq, dchi2_lq = fast_wls_gamma(
        vpec[~hq_mask], d_signed[~hq_mask], w[~hq_mask])
    sig_lq = abs(gamma_lq) / gamma_err_lq if gamma_err_lq > 0 else 0

    print(f"  High-quality (e_dmav <= {e_median:.3f}, n={n_hq}): "
          f"gamma_v = {gamma_hq:.2f} +/- {gamma_err_hq:.2f} ({sig_hq:.1f} sigma)",
          flush=True)
    print(f"  Low-quality  (e_dmav > {e_median:.3f}, n={len(vpec)-n_hq}): "
          f"gamma_v = {gamma_lq:.2f} +/- {gamma_err_lq:.2f} ({sig_lq:.1f} sigma)",
          flush=True)

    results['quality_split'] = {
        'e_dmav_threshold': float(e_median),
        'high_quality': {
            'gamma_v': float(gamma_hq),
            'gamma_v_err': float(gamma_err_hq),
            'significance': float(sig_hq),
            'n': int(n_hq),
        },
        'low_quality': {
            'gamma_v': float(gamma_lq),
            'gamma_v_err': float(gamma_err_lq),
            'significance': float(sig_lq),
            'n': int(len(vpec) - n_hq),
        },
    }

    return results


# ============================================================
# Plotting
# ============================================================

def plot_loso(loso_results, output_dir):
    """Plot LOSO results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Leave-one-out
    ax = axes[0]
    baseline = loso_results['baseline']
    methods = list(loso_results['leave_out'].keys())
    gammas = [loso_results['leave_out'][m]['gamma_v'] for m in methods]
    errs = [loso_results['leave_out'][m]['gamma_v_err'] for m in methods]
    x = range(len(methods))

    ax.errorbar(x, gammas, yerr=errs, fmt='o', color='#2196F3', capsize=5,
                markersize=8, label='Leave-one-out')
    ax.axhline(baseline['gamma_v'], color='red', ls='--', lw=2,
               label=f"Baseline: {baseline['gamma_v']:.1f}")
    ax.fill_between([-0.5, len(methods) - 0.5],
                     baseline['gamma_v'] - baseline['gamma_v_err'],
                     baseline['gamma_v'] + baseline['gamma_v_err'],
                     alpha=0.2, color='red')
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
    ax.set_title('Leave-One-Survey-Out')
    ax.legend(fontsize=9)

    # Single-method
    ax = axes[1]
    methods_only = [m for m in loso_results['only_use']
                    if not loso_results['only_use'][m].get('too_few')]
    if methods_only:
        gammas_only = [loso_results['only_use'][m]['gamma_v'] for m in methods_only]
        errs_only = [loso_results['only_use'][m]['gamma_v_err'] for m in methods_only]
        ns = [loso_results['only_use'][m]['n'] for m in methods_only]
        x_only = range(len(methods_only))

        ax.errorbar(x_only, gammas_only, yerr=errs_only, fmt='s', color='#4CAF50',
                    capsize=5, markersize=8, label='Single method')
        ax.axhline(baseline['gamma_v'], color='red', ls='--', lw=2,
                   label=f"Baseline: {baseline['gamma_v']:.1f}")
        ax.fill_between([-0.5, len(methods_only) - 0.5],
                         baseline['gamma_v'] - baseline['gamma_v_err'],
                         baseline['gamma_v'] + baseline['gamma_v_err'],
                         alpha=0.2, color='red')
        labels_only = [f'{m}\n(n={n})' for m, n in zip(methods_only, ns)]
        ax.set_xticks(list(x_only))
        ax.set_xticklabels(labels_only)
        ax.set_ylabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
        ax.set_title('Single Method Only')
        ax.legend(fontsize=9)

    fig.suptitle('CF4 gamma_v: Leave-One-Survey-Out Robustness', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'loso_robustness.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_malmquist(malmquist_results, output_dir):
    """Plot Malmquist bias diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distance bins
    ax = axes[0]
    bins = [b for b in malmquist_results['distance_bins'] if not b.get('too_few')]
    if bins:
        x_centers = [(b['d_lo'] + b['d_hi']) / 2 for b in bins]
        gammas = [b['gamma_v'] for b in bins]
        errs = [b['gamma_v_err'] for b in bins]

        ax.errorbar(x_centers, gammas, yerr=errs, fmt='o-', color='#2196F3',
                    capsize=5, markersize=8)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Distance (Mpc)')
        ax.set_ylabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
        ax.set_title('gamma_v in Distance Bins')

        # Add sample sizes
        for xc, g, b in zip(x_centers, gammas, bins):
            ax.annotate(f"n={b['n']}", (xc, g), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=7)

    # Error quartiles
    ax = axes[1]
    eq = malmquist_results['error_quartiles']
    if eq:
        x_centers = [(b['e_lo'] + b['e_hi']) / 2 for b in eq]
        gammas = [b['gamma_v'] for b in eq]
        errs = [b['gamma_v_err'] for b in eq]

        ax.errorbar(x_centers, gammas, yerr=errs, fmt='s-', color='#4CAF50',
                    capsize=5, markersize=8)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('e_dmav (distance modulus error)')
        ax.set_ylabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
        ax.set_title('gamma_v vs Measurement Quality')

        for xc, g, b in zip(x_centers, gammas, eq):
            ax.annotate(f"n={b['n']}", (xc, g), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=7)

    fig.suptitle('CF4 gamma_v: Malmquist Bias Diagnostics', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'malmquist_diagnostics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print("Task 3A ROBUSTNESS TESTS: LOSO + Malmquist Bias", flush=True)
    print("=" * 70, flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(flush=True)

    # Load CF4
    print("--- Loading Cosmicflows-4 ---", flush=True)
    groups = load_cf4()
    z_arr = np.array([g['z'] for g in groups])
    z_max = 0.15
    valid = z_arr < z_max
    groups = [g for g, m in zip(groups, valid) if m]
    z_arr = np.array([g['z'] for g in groups])
    print(f"  {len(groups)} groups at z < {z_max}", flush=True)

    # Method counts
    for method, flag in [('TF', 'o_tf'), ('FP', 'o_fp'),
                          ('SNIa', 'o_snia'), ('Cal', 'o_cal')]:
        n = sum(1 for g in groups if g[flag])
        print(f"  {method}: {n} ({n/len(groups)*100:.1f}%)", flush=True)

    # Comoving coordinates
    gx, gy, gz = groups_to_comoving(groups)

    # Vpec uncertainties
    e_dmav = np.array([g['e_dmav'] for g in groups])
    dist = np.array([g['dist'] for g in groups])
    vpec_err = H0_FID * dist * np.log(10) / 5.0 * e_dmav
    vpec_err = np.maximum(vpec_err, 100.0)

    # Run for primary void catalogue (VoidFinder)
    cat_name = 'VoidFinder'
    ngc_key, sgc_key, cat_type = CATALOGUE_GROUPS[cat_name]
    vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
    d_signed, in_void = compute_environment_fast(gx, gy, gz, vx, vy, vz, vr)

    print(f"\n  Using void catalogue: {cat_name}", flush=True)

    # Test 1: LOSO
    loso_results = test_loso(groups, gx, gy, gz, d_signed, vpec_err, z_arr)

    # Test 2: Malmquist
    malmquist_results = test_malmquist(groups, gx, gy, gz, d_signed, vpec_err, z_arr)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("ROBUSTNESS SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print("\n  LOSO Stability:", flush=True)
    baseline_gamma = loso_results['baseline']['gamma_v']
    all_consistent = True
    for method, r in loso_results['leave_out'].items():
        shift = abs(r['gamma_v'] - baseline_gamma)
        consistent = shift < 2 * loso_results['baseline']['gamma_v_err']
        status = "CONSISTENT" if consistent else "SHIFTED"
        if not consistent:
            all_consistent = False
        print(f"    Leave out {method}: shift = {shift:.2f} km/s ({status})",
              flush=True)

    if all_consistent:
        print("    VERDICT: gamma_v is stable across all distance methods", flush=True)
    else:
        print("    VERDICT: some methods show significant shifts", flush=True)

    print("\n  Malmquist Diagnostics:", flush=True)
    corr = malmquist_results['correlations']
    print(f"    d_signed-distance correlation: {corr['dsigned_distance']:.4f}",
          flush=True)
    if abs(corr['dsigned_distance']) < 0.1:
        print("    VERDICT: d_signed is NOT strongly correlated with distance",
              flush=True)
        print("    -> Malmquist bias unlikely to drive the signal", flush=True)
    else:
        print("    WARNING: moderate d_signed-distance correlation", flush=True)

    qs = malmquist_results['quality_split']
    hq_sig = qs['high_quality']['significance']
    lq_sig = qs['low_quality']['significance']
    print(f"\n    High-quality subsample: {hq_sig:.1f} sigma", flush=True)
    print(f"    Low-quality subsample:  {lq_sig:.1f} sigma", flush=True)
    if hq_sig > 3 and lq_sig > 3:
        print("    VERDICT: Signal present in BOTH quality subsamples", flush=True)

    # Plots
    print("\n--- Generating plots ---", flush=True)
    plot_loso(loso_results, OUTPUT_DIR)
    plot_malmquist(malmquist_results, OUTPUT_DIR)

    # Save
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'CF4 robustness tests: LOSO + Malmquist bias diagnostics',
        'void_catalogue': cat_name,
        'n_groups': len(groups),
        'loso': loso_results,
        'malmquist': malmquist_results,
    }

    save_results(all_results, 'task3a_robustness_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
