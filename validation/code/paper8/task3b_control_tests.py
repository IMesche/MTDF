#!/usr/bin/env python3
"""
Task 3B Control Tests: Is the 2MTF gamma_TF signal MTDF or environmental TF systematics?

Same logic as Task 3A controls but applied to Tully-Fisher residuals:

  Control A: Density substitution
    Replace d_signed with local galaxy number density.

  Control B: Partial correlation
    Regress out local density, then test d_signed on residuals.
    The key question: does void geometry add information about TF residuals
    beyond what raw local density already explains?

  Control C: LCDM expectation
    TF residuals should NOT correlate with density in LCDM if the TF relation
    is properly calibrated. Any correlation is either:
    (a) known environmental TF systematics (morphology, gas, tidal), or
    (b) MTDF.
    We check whether d_signed adds anything beyond density.

Author: Ingo Mesche
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import (load_void_pair, gls_fit,
                    save_results, CATALOGUE_GROUPS, COSMO_VOIDS)

# Reuse 2MTF and environment from task3b/3a
sys.path.insert(0, os.path.dirname(__file__))
from task3b_2mtf_tully_fisher import load_2mtf, fit_tully_fisher
from task3a_cosmicflows4_vpec import compute_environment_fast

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)
C_KMS = 299792.458

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3b_controls')


def galaxies_to_comoving(galaxies):
    """Convert galaxy positions to comoving coordinates (Mpc/h)."""
    z = np.array([g['z'] for g in galaxies])
    ra = np.radians(np.array([g['ra'] for g in galaxies]))
    dec = np.radians(np.array([g['dec'] for g in galaxies]))
    d_c = COSMO.comoving_distance(z).value
    x = d_c * np.cos(dec) * np.cos(ra)
    y = d_c * np.cos(dec) * np.sin(ra)
    z_cart = d_c * np.sin(dec)
    return x, y, z_cart


def compute_local_density(gx, gy, gz, radii_mpc_h=(5.0, 10.0, 20.0)):
    """Local galaxy number density from 2MTF positions."""
    coords = np.column_stack([gx, gy, gz])
    tree = cKDTree(coords)
    densities = {}
    for r in radii_mpc_h:
        counts = tree.query_ball_point(coords, r, workers=-1, return_length=True)
        counts = np.array(counts, dtype=float) - 1
        vol = (4.0 / 3.0) * np.pi * r**3
        densities[r] = counts / vol
        print(f"    R = {r:.0f} Mpc/h: median neighbors = {np.median(counts):.0f}")
    return densities


def gls_env(delta_mu, metric, mu_err):
    """GLS regression: delta_mu = alpha + gamma * metric."""
    n = len(delta_mu)
    cov = np.diag(mu_err**2)
    X_null = np.ones((n, 1))
    X_full = np.column_stack([np.ones(n), metric])
    _, _, chi2_null, _ = gls_fit(delta_mu, X_null, cov)
    beta, beta_cov, chi2_full, dof = gls_fit(delta_mu, X_full, cov)
    dchi2 = chi2_null - chi2_full
    p = 1 - stats.chi2.cdf(dchi2, 1)
    gamma = beta[1]
    gamma_err = np.sqrt(beta_cov[1, 1])
    return {
        'gamma': float(gamma),
        'gamma_err': float(gamma_err),
        'significance': float(abs(gamma) / gamma_err) if gamma_err > 0 else 0,
        'delta_chi2': float(dchi2),
        'p_value': float(p),
        'n': int(n),
    }


def control_a(delta_mu, mu_err, densities):
    """Control A: Replace d_signed with local density."""
    print("\n" + "=" * 60)
    print("CONTROL A: Density Substitution (2MTF)")
    print("=" * 60)
    results = {}
    for radius, rho in densities.items():
        label = f"R{radius:.0f}"
        rho_std = (rho - np.mean(rho)) / np.std(rho)
        r = gls_env(delta_mu, rho_std, mu_err)
        print(f"  {label}: gamma_rho = {r['gamma']:.5f} +/- {r['gamma_err']:.5f} "
              f"({r['significance']:.2f} sigma, dchi2={r['delta_chi2']:.2f})")
        results[label] = {'radius': float(radius), **r}
    return results


def control_b(delta_mu, mu_err, d_signed, densities, cat_name):
    """Control B: Partial correlation -- d_signed beyond density."""
    print(f"\n  --- Control B: Partial Correlation ({cat_name}) ---")
    n = len(delta_mu)
    cov = np.diag(mu_err**2)
    results = {}

    for radius, rho in densities.items():
        label = f"R{radius:.0f}"
        rho_std = (rho - np.mean(rho)) / np.std(rho)
        d_std = (d_signed - np.mean(d_signed)) / np.std(d_signed)

        corr = np.corrcoef(d_signed, rho)[0, 1]

        # M0: null
        X0 = np.ones((n, 1))
        _, _, chi2_0, _ = gls_fit(delta_mu, X0, cov)

        # M1: density only
        X1 = np.column_stack([np.ones(n), rho_std])
        _, _, chi2_1, _ = gls_fit(delta_mu, X1, cov)

        # M2: density + d_signed
        X2 = np.column_stack([np.ones(n), rho_std, d_std])
        beta2, beta2_cov, chi2_2, _ = gls_fit(delta_mu, X2, cov)

        dchi2_density = chi2_0 - chi2_1
        p_density = 1 - stats.chi2.cdf(dchi2_density, 1)

        dchi2_void = chi2_1 - chi2_2
        p_void = 1 - stats.chi2.cdf(dchi2_void, 1)

        gamma_d = beta2[2]
        gamma_d_err = np.sqrt(beta2_cov[2, 2])
        sig_d = abs(gamma_d) / gamma_d_err if gamma_d_err > 0 else 0

        print(f"    {label}: corr(d,rho)={corr:.3f}")
        print(f"      Density alone: dchi2={dchi2_density:.2f} (p={p_density:.4e})")
        print(f"      d_signed beyond density: dchi2={dchi2_void:.2f} "
              f"(p={p_void:.4e}, gamma_d={gamma_d:.5f} [{sig_d:.1f}sig])")

        results[label] = {
            'radius': float(radius),
            'corr_d_rho': float(corr),
            'dchi2_density_vs_null': float(dchi2_density),
            'p_density': float(p_density),
            'dchi2_void_beyond_density': float(dchi2_void),
            'p_void_beyond_density': float(p_void),
            'gamma_d_partial': float(gamma_d),
            'gamma_d_err': float(gamma_d_err),
            'gamma_d_significance': float(sig_d),
            'n': n,
        }

    return results


def control_c(delta_mu, mu_err, d_signed, densities, cat_name):
    """Control C: Is TF-density correlation expected in LCDM?"""
    print(f"\n  --- Control C: LCDM Expectation ({cat_name}) ---")
    results = {}

    for radius, rho in densities.items():
        label = f"R{radius:.0f}"
        rho_mean = np.mean(rho)
        delta = (rho - rho_mean) / rho_mean if rho_mean > 0 else rho - rho_mean

        # TF residuals vs density
        sl_mu_rho, _, r_mu_rho, p_mu_rho, _ = stats.linregress(delta, delta_mu)
        # Density vs d_signed
        sl_rho_d, _, r_rho_d, p_rho_d, _ = stats.linregress(d_signed, delta)
        # Expected gamma from chain
        gamma_exp = sl_mu_rho * sl_rho_d
        # Observed gamma
        r_obs = gls_env(delta_mu, d_signed, mu_err)
        gamma_obs = r_obs['gamma']
        ratio = gamma_obs / gamma_exp if abs(gamma_exp) > 1e-15 else np.inf

        print(f"    {label}: TF vs density slope={sl_mu_rho:.5f} (r={r_mu_rho:.3f})")
        print(f"      observed gamma={gamma_obs:.5f}, expected={gamma_exp:.5f}, ratio={ratio:.3f}")

        if abs(ratio - 1.0) < 0.3:
            verdict = "CONSISTENT with density-only"
        elif abs(ratio) > 1.3:
            verdict = "EXCESS beyond density-only"
        else:
            verdict = "DEFICIT relative to density-only"
        print(f"      Verdict: {verdict}")

        results[label] = {
            'radius': float(radius),
            'gamma_observed': float(gamma_obs),
            'gamma_expected': float(gamma_exp),
            'ratio': float(ratio),
            'slope_tf_vs_delta': float(sl_mu_rho),
            'r_tf_delta': float(r_mu_rho),
            'slope_delta_vs_dsigned': float(sl_rho_d),
            'r_delta_dsigned': float(r_rho_d),
            'verdict': verdict,
        }

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Task 3B CONTROL TESTS: 2MTF Tully-Fisher")
    print("Is the 7.9 sigma gamma_TF signal MTDF, or TF environmental systematics?")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load 2MTF
    print("--- Loading 2MTF ---")
    galaxies = load_2mtf()

    # Comoving coordinates
    gx, gy, gz = galaxies_to_comoving(galaxies)

    # Fit TF and get residuals (K-band primary)
    print("\n--- Fitting Tully-Fisher (K-band) ---")
    a, b, delta_mu, sigma_tf = fit_tully_fisher(galaxies, band='k')

    # TF residual uncertainties (approximate from magnitude errors + TF scatter)
    e_k = np.array([g['e_k'] for g in galaxies])
    mu_err = np.sqrt(e_k**2 + 0.05**2)  # magnitude err + floor
    mu_err = np.maximum(mu_err, sigma_tf * 0.5)  # at least half the TF scatter

    # Local densities
    print("\n--- Computing local density ---")
    densities = compute_local_density(gx, gy, gz, radii_mpc_h=(5.0, 10.0, 20.0))

    # Control A: density substitution (void-independent)
    ctrl_a = control_a(delta_mu, mu_err, densities)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Control tests for Task 3B 2MTF gamma_TF signal',
        'n_galaxies': len(galaxies),
        'tf_band': 'K',
        'tf_scatter': float(sigma_tf),
        'control_a': ctrl_a,
        'control_b': {},
        'control_c': {},
    }

    # Controls B and C per void catalogue
    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"  Void catalogue: {cat_name}")
        print(f"{'='*60}")

        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            print(f"  WARNING: Could not load {cat_name}")
            continue

        d_signed, in_void = compute_environment_fast(gx, gy, gz, vx, vy, vz, vr)

        ctrl_b_results = control_b(delta_mu, mu_err, d_signed, densities, cat_name)
        all_results['control_b'][cat_name] = ctrl_b_results

        ctrl_c_results = control_c(delta_mu, mu_err, d_signed, densities, cat_name)
        all_results['control_c'][cat_name] = ctrl_c_results

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY (2MTF)")
    print("=" * 70)

    print("\nControl A (Density substitution):")
    for key, val in ctrl_a.items():
        print(f"  {key}: {val['significance']:.1f} sigma (dchi2={val['delta_chi2']:.2f})")

    print("\nControl B (d_signed beyond density):")
    for cat_name, cr in all_results['control_b'].items():
        for key, val in cr.items():
            print(f"  {cat_name}/{key}: dchi2_beyond = {val['dchi2_void_beyond_density']:.2f} "
                  f"(p={val['p_void_beyond_density']:.4e}, {val['gamma_d_significance']:.1f}sig)")

    print("\nControl C (Observed/Expected ratio):")
    for cat_name, cr in all_results['control_c'].items():
        for key, val in cr.items():
            print(f"  {cat_name}/{key}: ratio={val['ratio']:.3f} ({val['verdict']})")

    save_results(all_results, 'task3b_control_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
