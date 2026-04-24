#!/usr/bin/env python3
"""
Task 3A 2M++ Reconstruction: CF4 gamma_v after proper velocity-field subtraction.

Addresses the caveat: "CF4 residuals use shell-median subtraction, not a
reconstructed velocity field."

Uses the Carrick et al. (2015) 2M++ reconstructed velocity field to predict
the LCDM peculiar velocity at each CF4 galaxy position. Subtracts the
predicted velocity and tests whether gamma_v survives on the TRUE residuals.

2M++ specifications:
  - 257^3 grid, 400 Mpc/h box (-200 to +200), cell spacing 1.5625 Mpc/h
  - Galactic Cartesian coordinates, CMB frame
  - beta* = 0.43 baked in
  - Based on 2MASS photometric survey (independent of CF4 velocities)

If gamma_v survives 2M++ subtraction, the signal CANNOT be explained by
standard LCDM density-velocity coupling.

Author: Ingo Mesche
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u
import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import load_void_pair, gls_fit, save_results, CATALOGUE_GROUPS

sys.path.insert(0, os.path.dirname(__file__))
from task3a_cosmicflows4_vpec import (
    load_cf4, groups_to_comoving, compute_environment_fast
)
from task3a_lcdm_mock import fast_wls_gamma

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)
C_KMS = 299792.458
H0_FID = 75.0
Z_CUT = 0.04

# 2M++ grid parameters
TWOMPP_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                           'External', '2mpp')
TWOMPP_NGRID = 257
TWOMPP_BOX = 400.0   # Mpc/h
TWOMPP_DMAX = 200.0  # Mpc/h, half-box
TWOMPP_CELL = TWOMPP_BOX / (TWOMPP_NGRID - 1)  # 1.5625 Mpc/h

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3a_2mpp')


def load_2mpp_velocity():
    """Load the 2M++ velocity field and build interpolators."""
    vel_path = os.path.join(TWOMPP_DIR, 'twompp_velocity.npy')
    print(f"  Loading 2M++ velocity field from {vel_path}", flush=True)
    vel = np.load(vel_path)  # shape (3, 257, 257, 257)
    print(f"  Shape: {vel.shape}, range: [{vel.min():.0f}, {vel.max():.0f}] km/s",
          flush=True)

    # Coordinate axis: cell centers in Mpc/h (Galactic Cartesian)
    ax = np.array([(i - 128) * TWOMPP_BOX / (TWOMPP_NGRID - 1)
                    for i in range(TWOMPP_NGRID)])

    # Build interpolators for each velocity component
    interp_vx = RegularGridInterpolator(
        (ax, ax, ax), vel[0], method='linear',
        bounds_error=False, fill_value=np.nan)
    interp_vy = RegularGridInterpolator(
        (ax, ax, ax), vel[1], method='linear',
        bounds_error=False, fill_value=np.nan)
    interp_vz = RegularGridInterpolator(
        (ax, ax, ax), vel[2], method='linear',
        bounds_error=False, fill_value=np.nan)

    print(f"  Grid axis: [{ax[0]:.1f}, {ax[-1]:.1f}] Mpc/h, "
          f"cell = {TWOMPP_CELL:.4f} Mpc/h", flush=True)

    return interp_vx, interp_vy, interp_vz


def cf4_to_galactic_cartesian(groups):
    """
    Convert CF4 galaxy positions from (RA, Dec, z) to Galactic Cartesian (X, Y, Z).

    The 2M++ velocity field uses Galactic Cartesian coordinates:
      X = d * cos(b) * cos(l)
      Y = d * cos(b) * sin(l)
      Z = d * sin(b)
    where (l, b) are Galactic longitude/latitude and d is comoving distance in Mpc/h.
    """
    ra = np.array([g['ra'] for g in groups])
    dec = np.array([g['dec'] for g in groups])
    z = np.array([g['z'] for g in groups])

    # Convert equatorial to galactic coordinates
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    l_gal = coords.galactic.l.rad  # radians
    b_gal = coords.galactic.b.rad  # radians

    # Comoving distance in Mpc/h (H0=100)
    d_com = COSMO.comoving_distance(z).value

    # Galactic Cartesian
    gx = d_com * np.cos(b_gal) * np.cos(l_gal)
    gy = d_com * np.cos(b_gal) * np.sin(l_gal)
    gz = d_com * np.sin(b_gal)

    # LOS unit vectors (Galactic Cartesian)
    rhat_x = np.cos(b_gal) * np.cos(l_gal)
    rhat_y = np.cos(b_gal) * np.sin(l_gal)
    rhat_z = np.sin(b_gal)

    return gx, gy, gz, rhat_x, rhat_y, rhat_z, d_com, l_gal, b_gal


def predict_2mpp_velocities(interp_vx, interp_vy, interp_vz,
                              gx, gy, gz, rhat_x, rhat_y, rhat_z, d_com):
    """
    Predict LOS peculiar velocities from the 2M++ field at CF4 positions.

    Returns v_pred_los (km/s) and a mask for valid positions (within grid).
    """
    n = len(gx)
    points = np.column_stack([gx, gy, gz])

    vx_pred = interp_vx(points)
    vy_pred = interp_vy(points)
    vz_pred = interp_vz(points)

    # LOS component
    v_pred_los = vx_pred * rhat_x + vy_pred * rhat_y + vz_pred * rhat_z

    # Valid mask: within grid and not NaN
    valid = np.isfinite(v_pred_los) & (d_com < TWOMPP_DMAX * 0.95)
    n_valid = valid.sum()
    n_total = len(valid)

    print(f"  2M++ predictions: {n_valid}/{n_total} valid "
          f"({n_valid/n_total*100:.1f}%)", flush=True)
    print(f"  Predicted v_LOS: mean={np.nanmean(v_pred_los[valid]):.0f}, "
          f"std={np.nanstd(v_pred_los[valid]):.0f} km/s", flush=True)

    return v_pred_los, valid


def compute_residuals(groups, v_pred_los, valid_mask):
    """
    Compute true residuals: v_residual = v_observed - v_2mpp_predicted.

    For galaxies outside the 2M++ grid, fall back to the observed vpec
    (no correction applied, equivalent to assuming v_pred = 0).
    """
    z_arr = np.array([g['z'] for g in groups])
    dist = np.array([g['dist'] for g in groups])
    e_dmav = np.array([g['e_dmav'] for g in groups])

    # Observed peculiar velocity from CF4 (already computed in the catalogue)
    v_obs = np.array([g['vpec'] for g in groups], dtype=float)

    # Subtract 2M++ prediction where valid
    v_residual = v_obs.copy()
    v_residual[valid_mask] -= v_pred_los[valid_mask]

    # Uncertainties (same as main analysis)
    vpec_err = H0_FID * dist * np.log(10) / 5.0 * e_dmav
    vpec_err = np.maximum(vpec_err, 100.0)

    n_val = valid_mask.sum()
    print(f"  Observed vpec: mean={np.mean(v_obs):.0f}, std={np.std(v_obs):.0f} km/s",
          flush=True)
    print(f"  Residual (2M++ subtracted, n={n_val}): "
          f"mean={np.mean(v_residual[valid_mask]):.0f}, "
          f"std={np.std(v_residual[valid_mask]):.0f} km/s", flush=True)
    print(f"  Variance reduction: {1 - np.var(v_residual[valid_mask])/np.var(v_obs[valid_mask]):.3f}",
          flush=True)

    return v_obs, v_residual, vpec_err


def analyze_gamma_v(label, vpec, vpec_err, d_signed, z_arr, method_label):
    """Run gamma_v analysis on a velocity set."""
    w = 1.0 / vpec_err**2

    # Full sample
    gamma_full, gamma_err_full, dchi2_full = fast_wls_gamma(vpec, d_signed, w)
    sig_full = abs(gamma_full) / gamma_err_full if gamma_err_full > 0 else 0

    # Piecewise
    mask_low = z_arr < Z_CUT
    mask_high = z_arr >= Z_CUT

    gamma_low, gamma_err_low, dchi2_low = 0, 0, 0
    gamma_high, gamma_err_high, dchi2_high = 0, 0, 0

    if mask_low.sum() > 50:
        gamma_low, gamma_err_low, dchi2_low = fast_wls_gamma(
            vpec[mask_low], d_signed[mask_low], w[mask_low])
    if mask_high.sum() > 50:
        gamma_high, gamma_err_high, dchi2_high = fast_wls_gamma(
            vpec[mask_high], d_signed[mask_high], w[mask_high])

    sig_low = abs(gamma_low) / gamma_err_low if gamma_err_low > 0 else 0
    sig_high = abs(gamma_high) / gamma_err_high if gamma_err_high > 0 else 0

    print(f"\n  {label} ({method_label}):", flush=True)
    print(f"    Full:      gamma_v = {gamma_full:.2f} +/- {gamma_err_full:.2f} "
          f"({sig_full:.1f} sigma, dchi2={dchi2_full:.1f})", flush=True)
    print(f"    z < {Z_CUT}: gamma_v = {gamma_low:.2f} +/- {gamma_err_low:.2f} "
          f"({sig_low:.1f} sigma, dchi2={dchi2_low:.1f})", flush=True)
    print(f"    z >= {Z_CUT}: gamma_v = {gamma_high:.2f} +/- {gamma_err_high:.2f} "
          f"({sig_high:.1f} sigma, dchi2={dchi2_high:.1f})", flush=True)

    return {
        'gamma_v_full': float(gamma_full),
        'gamma_v_full_err': float(gamma_err_full),
        'significance_full': float(sig_full),
        'dchi2_full': float(dchi2_full),
        'gamma_v_lowz': float(gamma_low),
        'gamma_v_lowz_err': float(gamma_err_low),
        'significance_lowz': float(sig_low),
        'dchi2_lowz': float(dchi2_low),
        'gamma_v_highz': float(gamma_high),
        'gamma_v_highz_err': float(gamma_err_high),
        'significance_highz': float(sig_high),
        'dchi2_highz': float(dchi2_high),
        'n': len(vpec),
        'n_lowz': int(mask_low.sum()),
        'n_highz': int(mask_high.sum()),
    }


def plot_comparison(results_by_cat, output_dir):
    """Plot gamma_v before and after 2M++ subtraction for each void catalogue."""
    cats = list(results_by_cat.keys())
    n_cats = len(cats)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6))
    if n_cats == 1:
        axes = [axes]

    for ax, cat_name in zip(axes, cats):
        r = results_by_cat[cat_name]

        # Data for plotting
        labels = ['Full', f'z<{Z_CUT}', f'z>={Z_CUT}']
        x = np.arange(len(labels))
        width = 0.35

        # Shell-median values
        sm = r['shell_median']
        sm_vals = [sm['gamma_v_full'], sm['gamma_v_lowz'], sm['gamma_v_highz']]
        sm_errs = [sm['gamma_v_full_err'], sm['gamma_v_lowz_err'], sm['gamma_v_highz_err']]

        # 2M++ values
        tp = r['2mpp_residual']
        tp_vals = [tp['gamma_v_full'], tp['gamma_v_lowz'], tp['gamma_v_highz']]
        tp_errs = [tp['gamma_v_full_err'], tp['gamma_v_lowz_err'], tp['gamma_v_highz_err']]

        bars1 = ax.bar(x - width/2, sm_vals, width, yerr=sm_errs,
                       label='Shell-median', color='#FF7043', alpha=0.8, capsize=4)
        bars2 = ax.bar(x + width/2, tp_vals, width, yerr=tp_errs,
                       label='2M++ subtracted', color='#42A5F5', alpha=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
        ax.set_title(cat_name)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.legend(fontsize=9)

        # Annotate significance
        for i, (sv, tv) in enumerate(zip(sm_vals, tp_vals)):
            sm_sig = [sm['significance_full'], sm['significance_lowz'],
                      sm['significance_highz']][i]
            tp_sig = [tp['significance_full'], tp['significance_lowz'],
                      tp['significance_highz']][i]
            y_off = max(abs(sv), abs(tv)) * 0.15
            ax.text(i - width/2, sv + np.sign(sv) * y_off, f'{sm_sig:.1f}s',
                    ha='center', fontsize=8, color='#BF360C')
            ax.text(i + width/2, tv + np.sign(tv) * y_off, f'{tp_sig:.1f}s',
                    ha='center', fontsize=8, color='#1565C0')

    fig.suptitle('CF4 gamma_v: Shell-Median vs 2M++ Velocity Field Subtraction',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, '2mpp_vs_shell_median.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_2mpp_only_subsample(results_by_cat, output_dir):
    """Plot results using only galaxies within 2M++ grid coverage."""
    cats = list(results_by_cat.keys())
    n_cats = len(cats)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 6))
    if n_cats == 1:
        axes = [axes]

    for ax, cat_name in zip(axes, cats):
        r = results_by_cat[cat_name]

        if '2mpp_only' not in r:
            continue

        labels = ['Full (2M++ vol)', f'z<{Z_CUT}', f'z>={Z_CUT}']
        x = np.arange(len(labels))
        width = 0.35

        # Original on 2M++ subsample
        orig = r['shell_median_2mpp_subsample']
        orig_vals = [orig['gamma_v_full'], orig['gamma_v_lowz'], orig['gamma_v_highz']]
        orig_errs = [orig['gamma_v_full_err'], orig['gamma_v_lowz_err'],
                     orig['gamma_v_highz_err']]

        # 2M++ subtracted on same subsample
        sub = r['2mpp_only']
        sub_vals = [sub['gamma_v_full'], sub['gamma_v_lowz'], sub['gamma_v_highz']]
        sub_errs = [sub['gamma_v_full_err'], sub['gamma_v_lowz_err'],
                    sub['gamma_v_highz_err']]

        bars1 = ax.bar(x - width/2, orig_vals, width, yerr=orig_errs,
                       label='Original (2M++ vol)', color='#FF7043', alpha=0.8,
                       capsize=4)
        bars2 = ax.bar(x + width/2, sub_vals, width, yerr=sub_errs,
                       label='2M++ subtracted', color='#42A5F5', alpha=0.8,
                       capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
        ax.set_title(f'{cat_name} (2M++ volume only)')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.legend(fontsize=9)

    fig.suptitle('CF4 gamma_v within 2M++ volume: Before vs After Subtraction',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, '2mpp_subsample_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print("Task 3A: 2M++ Velocity Field Reconstruction Test", flush=True)
    print("Does gamma_v survive proper LCDM velocity-field subtraction?", flush=True)
    print("=" * 70, flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"2M++ grid: {TWOMPP_NGRID}^3, {TWOMPP_BOX:.0f} Mpc/h box, "
          f"{TWOMPP_CELL:.4f} Mpc/h cells", flush=True)
    print(flush=True)

    # Step 1: Load 2M++ velocity field
    print("--- Step 1: Loading 2M++ velocity field ---", flush=True)
    interp_vx, interp_vy, interp_vz = load_2mpp_velocity()

    # Step 2: Load CF4 data
    print("\n--- Step 2: Loading Cosmicflows-4 ---", flush=True)
    groups = load_cf4()
    z_arr = np.array([g['z'] for g in groups])
    z_max = 0.15
    valid = z_arr < z_max
    groups = [g for g, m in zip(groups, valid) if m]
    z_arr = np.array([g['z'] for g in groups])
    print(f"  {len(groups)} groups at z < {z_max}", flush=True)

    # Step 3: Convert to Galactic Cartesian for 2M++
    print("\n--- Step 3: Converting to Galactic Cartesian ---", flush=True)
    gx_gal, gy_gal, gz_gal, rhat_x, rhat_y, rhat_z, d_com, l_gal, b_gal = \
        cf4_to_galactic_cartesian(groups)
    print(f"  Distance range: [{d_com.min():.0f}, {d_com.max():.0f}] Mpc/h",
          flush=True)
    print(f"  Within 2M++ grid (d < {TWOMPP_DMAX:.0f}): "
          f"{(d_com < TWOMPP_DMAX).sum()}/{len(d_com)}", flush=True)

    # Step 4: Predict 2M++ velocities
    print("\n--- Step 4: Predicting 2M++ velocities ---", flush=True)
    v_pred_los, valid_2mpp = predict_2mpp_velocities(
        interp_vx, interp_vy, interp_vz,
        gx_gal, gy_gal, gz_gal, rhat_x, rhat_y, rhat_z, d_com
    )

    # Step 5: Compute residuals
    print("\n--- Step 5: Computing residuals ---", flush=True)
    v_obs, v_residual, vpec_err = compute_residuals(groups, v_pred_los, valid_2mpp)

    # Also compute shell-median subtracted residuals for comparison
    z_edges = np.arange(0, z_arr.max() + 0.005, 0.005)
    v_shell_median = v_obs.copy()
    for j in range(len(z_edges) - 1):
        zmask = (z_arr >= z_edges[j]) & (z_arr < z_edges[j + 1])
        if zmask.sum() > 10:
            v_shell_median[zmask] -= np.median(v_obs[zmask])

    # Step 6: Equatorial comoving coordinates (for d_signed computation)
    print("\n--- Step 6: Computing d_signed ---", flush=True)
    # Need equatorial comoving coordinates for void matching
    ra_arr = np.array([g['ra'] for g in groups])
    dec_arr = np.array([g['dec'] for g in groups])
    ra_rad = np.radians(ra_arr)
    dec_rad = np.radians(dec_arr)
    gx_eq = d_com * np.cos(dec_rad) * np.cos(ra_rad)
    gy_eq = d_com * np.cos(dec_rad) * np.sin(ra_rad)
    gz_eq = d_com * np.sin(dec_rad)

    # Step 7: Analyze for each void catalogue
    results_by_cat = {}

    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"  Void catalogue: {cat_name}", flush=True)
        print(f"{'='*60}", flush=True)

        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            continue

        d_signed, in_void = compute_environment_fast(gx_eq, gy_eq, gz_eq,
                                                      vx, vy, vz, vr)

        # A: Shell-median subtracted (full sample, original method)
        sm_full = analyze_gamma_v(cat_name, v_shell_median, vpec_err,
                                   d_signed, z_arr, "shell-median, full sample")

        # B: 2M++ subtracted (full sample, hybrid: 2M++ where valid, raw elsewhere)
        tp_full = analyze_gamma_v(cat_name, v_residual, vpec_err,
                                   d_signed, z_arr, "2M++ subtracted, full sample")

        # C: Shell-median on 2M++ subsample only (for fair comparison)
        sm_sub = analyze_gamma_v(
            cat_name, v_shell_median[valid_2mpp], vpec_err[valid_2mpp],
            d_signed[valid_2mpp], z_arr[valid_2mpp],
            f"shell-median, 2M++ volume only (n={valid_2mpp.sum()})")

        # D: 2M++ subtracted on 2M++ subsample only (cleanest test)
        tp_sub = analyze_gamma_v(
            cat_name, v_residual[valid_2mpp], vpec_err[valid_2mpp],
            d_signed[valid_2mpp], z_arr[valid_2mpp],
            f"2M++ subtracted, 2M++ volume only (n={valid_2mpp.sum()})")

        results_by_cat[cat_name] = {
            'shell_median': sm_full,
            '2mpp_residual': tp_full,
            'shell_median_2mpp_subsample': sm_sub,
            '2mpp_only': tp_sub,
        }

    # Step 8: Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Does gamma_v survive 2M++ velocity-field subtraction?", flush=True)
    print("=" * 70, flush=True)

    for cat_name, r in results_by_cat.items():
        sm = r['shell_median']
        tp = r['2mpp_residual']
        tp_sub = r['2mpp_only']

        print(f"\n  {cat_name}:", flush=True)
        print(f"    Shell-median (full):   gamma_v = {sm['gamma_v_full']:.2f} "
              f"({sm['significance_full']:.1f} sigma)", flush=True)
        print(f"    2M++ subtracted (full): gamma_v = {tp['gamma_v_full']:.2f} "
              f"({tp['significance_full']:.1f} sigma)", flush=True)
        print(f"    2M++ subtracted (2M++ vol): gamma_v = {tp_sub['gamma_v_full']:.2f} "
              f"({tp_sub['significance_full']:.1f} sigma)", flush=True)

        retention = tp['significance_full'] / sm['significance_full'] \
            if sm['significance_full'] > 0 else 0
        print(f"    Signal retention: {retention:.1%}", flush=True)

        if tp_sub['significance_full'] > 3:
            print(f"    VERDICT: gamma_v SURVIVES 2M++ subtraction", flush=True)
        elif tp_sub['significance_full'] > 2:
            print(f"    VERDICT: gamma_v marginally survives", flush=True)
        else:
            print(f"    VERDICT: gamma_v absorbed by 2M++ field", flush=True)

    # Step 9: Plots
    print("\n--- Generating plots ---", flush=True)
    plot_comparison(results_by_cat, OUTPUT_DIR)
    plot_2mpp_only_subsample(results_by_cat, OUTPUT_DIR)

    # Step 10: Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': '2M++ velocity field reconstruction test for CF4 gamma_v',
        'twompp': {
            'grid': f'{TWOMPP_NGRID}^3',
            'box_size_mpc_h': TWOMPP_BOX,
            'cell_size_mpc_h': TWOMPP_CELL,
            'max_distance_mpc_h': TWOMPP_DMAX,
            'beta_star': 0.43,
            'reference': 'Carrick et al. 2015',
        },
        'n_groups_total': len(groups),
        'n_within_2mpp': int(valid_2mpp.sum()),
        'variance_reduction': float(
            1 - np.var(v_residual[valid_2mpp]) / np.var(v_obs[valid_2mpp])),
        'catalogues': results_by_cat,
    }

    save_results(all_results, 'task3a_2mpp_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
