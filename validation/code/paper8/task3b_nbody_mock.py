#!/usr/bin/env python3
"""
Task 3B N-body Mock: MDPL2 halo velocities vs observed 2MTF gamma_TF signal.

Same logic as task3a_nbody_mock.py but for Tully-Fisher residuals.

In LCDM, peculiar velocities bias redshift-based distances. For a galaxy
with v_pec, the redshift z_obs = z_cosmo + v_LOS/c differs from z_cosmo,
creating a distance modulus residual:
    delta_mu_vpec = 5*log10(d_L(z_obs)/d_L(z_cosmo))
              ~ (5/ln10) * v_LOS / (c * z_cosmo)   for v << cz

This effect could, in principle, create a gamma_TF signal if v_pec
correlates with void geometry. The question: is the LCDM effect large
enough to explain the observed gamma_TF = 0.058 (7.9 sigma)?

Uses the same cached MDPL2 halos and voids from task3a_nbody_mock.

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
from common import save_results

sys.path.insert(0, os.path.dirname(__file__))
from task3a_lcdm_mock import fast_wls_gamma
from task3a_nbody_mock import (
    download_mdpl2_halos, find_voids_in_halos, compute_d_signed,
    BOX_SIZE, OMEGA_M, H0, C_KMS, COSMO
)

# 2MTF parameters
Z_MAX_2MTF = 0.033   # 2MTF redshift limit
D_MAX_2MTF = COSMO.comoving_distance(Z_MAX_2MTF).value  # ~99 Mpc/h
SIGMA_TF = 0.55      # TF intrinsic scatter (mag), from observed K-band fit
N_OBSERVERS = 50     # more observers since small volume = more variance
H0_FID = 75.0        # for uncertainty conversion

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3b_nbody_mock')


def run_2mtf_observer(hx, hy, hz, hvx, hvy, hvz,
                       void_x, void_y, void_z, void_r,
                       obs_pos, rng):
    """
    Run 2MTF-like mock for a single observer position.

    1. Select halos within z < 0.033
    2. Compute LOS peculiar velocity
    3. Convert vpec to TF residual (distance modulus bias)
    4. Add TF scatter
    5. Compute d_signed from MDPL2 voids
    6. Measure gamma_TF
    """
    ox, oy, oz = obs_pos

    # Relative positions (periodic)
    dx = hx - ox
    dy = hy - oy
    dz = hz - oz
    dx = np.where(dx > BOX_SIZE / 2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE / 2, dx + BOX_SIZE, dx)
    dy = np.where(dy > BOX_SIZE / 2, dy - BOX_SIZE, dy)
    dy = np.where(dy < -BOX_SIZE / 2, dy + BOX_SIZE, dy)
    dz = np.where(dz > BOX_SIZE / 2, dz - BOX_SIZE, dz)
    dz = np.where(dz < -BOX_SIZE / 2, dz + BOX_SIZE, dz)

    r_com = np.sqrt(dx**2 + dy**2 + dz**2)

    # Select halos within 2MTF volume
    sel = (r_com > 5.0) & (r_com < D_MAX_2MTF)
    n_sel = sel.sum()
    if n_sel < 100:
        return None

    dx_s, dy_s, dz_s = dx[sel], dy[sel], dz[sel]
    r_s = r_com[sel]
    vx_s, vy_s, vz_s = hvx[sel], hvy[sel], hvz[sel]

    # LOS unit vectors
    rhat_x = dx_s / r_s
    rhat_y = dy_s / r_s
    rhat_z = dz_s / r_s

    # LOS peculiar velocity (N-body, fully non-linear)
    v_los = vx_s * rhat_x + vy_s * rhat_y + vz_s * rhat_z

    # Cosmological redshift from comoving distance
    z_cosmo = H0 * r_s / C_KMS  # low-z approximation

    # Distance modulus bias from peculiar velocity
    # delta_mu = 5/ln(10) * v_LOS / (c * z_cosmo)
    # At z ~ 0.01-0.03, this is significant
    delta_mu_vpec = (5.0 / np.log(10)) * v_los / (C_KMS * z_cosmo)

    # Add TF intrinsic scatter (Gaussian)
    tf_noise = rng.normal(0, SIGMA_TF, size=n_sel)
    delta_mu_mock = delta_mu_vpec + tf_noise

    # Uncertainties (approximate: TF scatter + magnitude error floor)
    # Real 2MTF has e_k ~ 0.03-0.1 mag, but TF scatter dominates
    mu_err = np.full(n_sel, SIGMA_TF)
    mu_err = np.maximum(mu_err, 0.1)

    # Void positions relative to observer (periodic wrapping)
    vdx = void_x - ox
    vdy = void_y - oy
    vdz = void_z - oz
    vdx = np.where(vdx > BOX_SIZE / 2, vdx - BOX_SIZE, vdx)
    vdx = np.where(vdx < -BOX_SIZE / 2, vdx + BOX_SIZE, vdx)
    vdy = np.where(vdy > BOX_SIZE / 2, vdy - BOX_SIZE, vdy)
    vdy = np.where(vdy < -BOX_SIZE / 2, vdy + BOX_SIZE, vdy)
    vdz = np.where(vdz > BOX_SIZE / 2, vdz - BOX_SIZE, vdz)
    vdz = np.where(vdz < -BOX_SIZE / 2, vdz + BOX_SIZE, vdz)

    # Use voids out to 200 Mpc/h buffer (void boundaries extend beyond D_MAX)
    vr_com = np.sqrt(vdx**2 + vdy**2 + vdz**2)
    v_sel = vr_com < D_MAX_2MTF + 100

    if v_sel.sum() < 3:
        return None

    # Compute d_signed
    d_signed, in_void = compute_d_signed(
        dx_s, dy_s, dz_s,
        vdx[v_sel], vdy[v_sel], vdz[v_sel], void_r[v_sel]
    )

    # Weights
    w = 1.0 / mu_err**2

    # Measure gamma_TF
    gamma_tf, gamma_tf_err, dchi2 = fast_wls_gamma(delta_mu_mock, d_signed, w)

    # Also measure gamma from vpec bias alone (no TF scatter)
    gamma_vpec_only, _, dchi2_vpec = fast_wls_gamma(delta_mu_vpec, d_signed, w)

    return {
        'gamma_tf': float(gamma_tf),
        'gamma_tf_err': float(gamma_tf_err),
        'dchi2': float(dchi2),
        'gamma_vpec_only': float(gamma_vpec_only),
        'dchi2_vpec_only': float(dchi2_vpec),
        'n_halos': int(n_sel),
        'n_voids_used': int(v_sel.sum()),
        'n_in_void': int(in_void.sum()),
        'frac_in_void': float(in_void.mean()),
        'v_los_std': float(np.std(v_los)),
        'delta_mu_vpec_std': float(np.std(delta_mu_vpec)),
        'z_median': float(np.median(z_cosmo)),
    }


def plot_2mtf_nbody(gamma_tf_dist, observed_gamma_tf, output_dir):
    """Plot N-body gamma_TF distribution vs observed."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gamma_arr = np.array(gamma_tf_dist)
    ax.hist(gamma_arr, bins=20, alpha=0.7, color='#2196F3', density=True,
            label=f'MDPL2 N-body (n={len(gamma_arr)})')

    ax.axvline(observed_gamma_tf, color='red', lw=2.5, ls='--',
               label=f'2MTF Observed: {observed_gamma_tf:.4f}')
    ax.axvline(np.mean(gamma_arr), color='#2196F3', lw=2,
               label=f'N-body mean: {np.mean(gamma_arr):.4f} +/- {np.std(gamma_arr):.4f}')
    ax.axvline(0, color='gray', lw=1, ls='-', alpha=0.5)

    if np.std(gamma_arr) > 0:
        n_sigma = (observed_gamma_tf - np.mean(gamma_arr)) / np.std(gamma_arr)
        ax.set_title(f'MDPL2 N-body Mock vs Observed 2MTF Signal\n'
                     f'Observed is {abs(n_sigma):.1f} sigma from N-body prediction',
                     fontsize=13)

    ax.set_xlabel(r'$\gamma_{\rm TF}$ (mag per $d_{\rm signed}$)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, 'nbody_2mtf_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print("Task 3B N-BODY MOCK (MDPL2) -- 2MTF Tully-Fisher", flush=True)
    print("Full N-body velocities vs observed 2MTF gamma_TF signal", flush=True)
    print("=" * 70, flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"2MTF volume: z < {Z_MAX_2MTF} (D < {D_MAX_2MTF:.0f} Mpc/h)", flush=True)
    print(f"TF scatter: {SIGMA_TF:.3f} mag", flush=True)
    print(f"N_observers: {N_OBSERVERS}", flush=True)
    print(flush=True)

    # Step 1: Load cached MDPL2 halos
    print("--- Step 1: Loading MDPL2 halos ---", flush=True)
    hx, hy, hz, hvx, hvy, hvz, hmvir = download_mdpl2_halos()
    n_halos = len(hx)
    print(f"  {n_halos} halos loaded", flush=True)

    # Estimate how many halos per observer volume
    vol_obs = (4.0 / 3.0) * np.pi * D_MAX_2MTF**3
    vol_box = BOX_SIZE**3
    n_expected = int(n_halos * vol_obs / vol_box)
    print(f"  Expected halos per observer: ~{n_expected} "
          f"(volume fraction: {vol_obs/vol_box:.5f})", flush=True)

    # Step 2: Load/find voids
    print("\n--- Step 2: Loading MDPL2 voids ---", flush=True)
    # Try to reuse cached void data from CF4 mock
    cache_file = os.path.join(os.path.dirname(__file__),
                               'output', 'task3a_nbody_mock', 'mdpl2_halos_z0.npz')
    # Voids are computed fresh (not cached separately), so recompute
    from task3a_nbody_mock import find_voids_in_halos as find_voids
    void_x, void_y, void_z, void_r = find_voids(hx, hy, hz, BOX_SIZE)

    if len(void_r) == 0:
        for delta_try in [-0.6, -0.5, -0.4]:
            print(f"  Trying delta_void = {delta_try}...", flush=True)
            from task3a_nbody_mock import find_voids_in_halos_adaptive
            void_x, void_y, void_z, void_r = find_voids_in_halos_adaptive(
                hx, hy, hz, BOX_SIZE, delta_try)
            if len(void_r) > 0:
                break

    if len(void_r) == 0:
        print("  FATAL: No voids found. Exiting.", flush=True)
        return

    print(f"  {len(void_r)} voids available", flush=True)

    # Step 3: Load observed 2MTF results
    print("\n--- Step 3: Loading observed 2MTF results ---", flush=True)
    task3b_path = os.path.join(os.path.dirname(__file__),
                                'output', 'task3b_2mtf', 'task3b_2mtf_results.json')
    with open(task3b_path) as f:
        task3b = json.load(f)

    # VoidFinder K-band results
    obs_vf = task3b['catalogues']['VoidFinder']['bands']['k']['gls']
    obs_gamma_tf = obs_vf['gamma_tf']
    obs_gamma_tf_err = obs_vf['gamma_tf_err']
    obs_significance = obs_vf['significance']

    print(f"  Observed gamma_TF (VoidFinder, K): {obs_gamma_tf:.5f} "
          f"+/- {obs_gamma_tf_err:.5f} ({obs_significance:.1f} sigma)", flush=True)

    # Step 4: Run mock for multiple observer positions
    print(f"\n--- Step 4: Running {N_OBSERVERS} observer placements ---", flush=True)
    rng = np.random.RandomState(123)
    obs_positions = rng.uniform(0, BOX_SIZE, size=(N_OBSERVERS, 3))

    results_list = []
    gamma_tf_dist = []
    gamma_vpec_dist = []

    for i, obs_pos in enumerate(obs_positions):
        result = run_2mtf_observer(
            hx, hy, hz, hvx, hvy, hvz,
            void_x, void_y, void_z, void_r,
            obs_pos, rng
        )

        if result is None:
            if (i + 1) % 10 == 0:
                print(f"  Observer {i+1}/{N_OBSERVERS}: SKIPPED", flush=True)
            continue

        results_list.append(result)
        gamma_tf_dist.append(result['gamma_tf'])
        gamma_vpec_dist.append(result['gamma_vpec_only'])

        if (i + 1) % 10 == 0:
            print(f"  Observer {i+1}/{N_OBSERVERS}: n={result['n_halos']}, "
                  f"gamma_tf={result['gamma_tf']:.5f} "
                  f"(vpec_only={result['gamma_vpec_only']:.5f}), "
                  f"dchi2={result['dchi2']:.2f}", flush=True)

    # Step 5: Summary
    print("\n" + "=" * 70, flush=True)
    print("2MTF N-BODY MOCK RESULTS", flush=True)
    print("=" * 70, flush=True)

    if len(gamma_tf_dist) == 0:
        print("  No valid observer placements!", flush=True)
        return

    gamma_arr = np.array(gamma_tf_dist)
    gamma_vpec_arr = np.array(gamma_vpec_dist)

    nbody_mean = np.mean(gamma_arr)
    nbody_std = np.std(gamma_arr)
    nbody_median = np.median(gamma_arr)

    vpec_mean = np.mean(gamma_vpec_arr)
    vpec_std = np.std(gamma_vpec_arr)

    tension = abs(obs_gamma_tf - nbody_mean) / nbody_std if nbody_std > 0 else float('inf')

    print(f"\n  N-body gamma_TF distribution ({len(gamma_arr)} observers):", flush=True)
    print(f"    With TF scatter:", flush=True)
    print(f"      Mean:   {nbody_mean:.5f} +/- {nbody_std:.5f}", flush=True)
    print(f"      Median: {nbody_median:.5f}", flush=True)
    print(f"      Range:  [{gamma_arr.min():.5f}, {gamma_arr.max():.5f}]", flush=True)
    print(f"    Vpec contribution only (no TF scatter):", flush=True)
    print(f"      Mean:   {vpec_mean:.5f} +/- {vpec_std:.5f}", flush=True)
    print(f"\n  Observed 2MTF: {obs_gamma_tf:.5f} +/- {obs_gamma_tf_err:.5f} "
          f"({obs_significance:.1f} sigma)", flush=True)
    print(f"\n  *** TENSION: {tension:.1f} sigma ***", flush=True)

    sign_mismatch = (nbody_mean * obs_gamma_tf < 0)
    print(f"\n  Sign check:", flush=True)
    print(f"    N-body mean sign: {'positive' if nbody_mean > 0 else 'negative'}",
          flush=True)
    print(f"    Observed sign: {'positive' if obs_gamma_tf > 0 else 'negative'}",
          flush=True)
    print(f"    Sign mismatch: {sign_mismatch}", flush=True)

    # Verdict
    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    if tension > 5:
        print(f"  N-body LCDM mock: {tension:.1f} sigma tension with observed 2MTF.",
              flush=True)
        print(f"  Full non-linear velocities CANNOT produce the observed gamma_TF.",
              flush=True)
    elif tension > 3:
        print(f"  N-body mock: {tension:.1f} sigma tension (marginal).", flush=True)
    else:
        print(f"  N-body mock: {tension:.1f} sigma (consistent with LCDM).", flush=True)

    print("=" * 70, flush=True)

    # Step 6: Plots
    print("\n--- Generating plots ---", flush=True)
    plot_2mtf_nbody(gamma_tf_dist, obs_gamma_tf, OUTPUT_DIR)

    # Step 7: Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'MDPL2 N-body mock test for 2MTF gamma_TF signal',
        'simulation': {
            'name': 'MDPL2',
            'box_size_mpc_h': BOX_SIZE,
            'omega_m': OMEGA_M,
            'source': 'CosmoSim TAP API',
        },
        'selection': {
            'z_max': Z_MAX_2MTF,
            'd_max_mpc_h': float(D_MAX_2MTF),
            'tf_scatter_mag': SIGMA_TF,
            'n_halos_total': n_halos,
            'n_voids_total': len(void_r),
            'expected_halos_per_observer': n_expected,
        },
        'n_observers': N_OBSERVERS,
        'n_valid_observers': len(gamma_tf_dist),
        'observed': {
            'gamma_tf': obs_gamma_tf,
            'gamma_tf_err': obs_gamma_tf_err,
            'significance': obs_significance,
            'catalogue': 'VoidFinder',
            'band': 'K',
        },
        'nbody_mock': {
            'gamma_tf_mean': float(nbody_mean),
            'gamma_tf_std': float(nbody_std),
            'gamma_tf_median': float(nbody_median),
            'gamma_tf_min': float(gamma_arr.min()),
            'gamma_tf_max': float(gamma_arr.max()),
            'gamma_tf_ci95': [float(np.percentile(gamma_arr, 2.5)),
                               float(np.percentile(gamma_arr, 97.5))],
            'tension_sigma': float(tension),
            'sign_mismatch': bool(sign_mismatch),
            'vpec_only_mean': float(vpec_mean),
            'vpec_only_std': float(vpec_std),
            'distribution': [float(g) for g in gamma_tf_dist],
            'distribution_vpec_only': [float(g) for g in gamma_vpec_dist],
        },
        'per_observer': results_list,
    }

    save_results(all_results, 'task3b_nbody_mock_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
