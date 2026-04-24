#!/usr/bin/env python3
"""
Task 3A Control Tests: Is the CF4 gamma_v signal MTDF or just gravity?

Three controls to distinguish MTDF-specific environment dependence from
the trivial fact that peculiar velocities correlate with density:

  Control A: Density substitution
    Replace d_signed with local galaxy number density estimated from CF4
    positions. If gamma_v is equally strong, d_signed is just a density proxy.

  Control B: Partial correlation
    Regress out local density first, then test d_signed on the residuals.
    If the signal survives after controlling for density, void geometry
    matters beyond raw density.

  Control C: LCDM linear-theory expectation
    Estimate the expected gamma_v from the density-velocity relation in
    linear perturbation theory. Compare observed vs expected. If they
    match, the signal is standard gravity, not MTDF.

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
from common import load_void_pair, gls_fit, save_results, CATALOGUE_GROUPS, COSMO_VOIDS

# Reuse CF4 loading and environment from task3a
from task3a_cosmicflows4_vpec import (
    load_cf4, groups_to_comoving, compute_vpec_residuals,
    compute_environment_fast, gls_vpec_env, permutation_test,
    bootstrap_test, Z_CUT
)

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)
C_KMS = 299792.458
H0_FID = 75.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3a_controls')


# ============================================================
# Control A: Local galaxy number density from CF4 itself
# ============================================================

def compute_local_density(gx, gy, gz, radii_mpc_h=(5.0, 10.0, 20.0)):
    """
    Estimate local galaxy number density using CF4 group positions.

    For each group, count neighbors within spheres of various radii.
    Returns dict of {radius: density_array}.

    Uses a KD-tree for efficiency on 38k points.
    """
    coords = np.column_stack([gx, gy, gz])
    tree = cKDTree(coords)

    densities = {}
    for r in radii_mpc_h:
        # Count neighbors within radius r (excluding self)
        counts = tree.query_ball_point(coords, r, workers=-1, return_length=True)
        counts = np.array(counts, dtype=float) - 1  # subtract self
        vol = (4.0 / 3.0) * np.pi * r**3
        densities[r] = counts / vol

        med = np.median(counts)
        print(f"    R = {r:.0f} Mpc/h: median neighbors = {med:.0f}, "
              f"density range = [{densities[r].min():.4f}, {densities[r].max():.4f}]")

    return densities


def control_a_density_substitution(vpec_residual, vpec_err, densities, groups):
    """
    Control A: Replace d_signed with local density.

    If gamma is equally significant, d_signed was just a density proxy.
    """
    print("\n" + "=" * 60)
    print("CONTROL A: Density Substitution")
    print("=" * 60)

    z_arr = np.array([g['z'] for g in groups])
    results = {}

    for radius, rho in densities.items():
        label = f"R{radius:.0f}"
        print(f"\n  --- Local density at R = {radius:.0f} Mpc/h ---")

        # Standardize density for numerical stability
        rho_std = (rho - np.mean(rho)) / np.std(rho)

        # Full sample
        r_full = gls_vpec_env(vpec_residual, rho_std, vpec_err)
        print(f"    Full: gamma_rho = {r_full['gamma_v']:.2f} +/- "
              f"{r_full['gamma_v_err']:.2f} ({r_full['significance']:.2f} sigma)")

        # Piecewise z-split
        mask_low = z_arr < Z_CUT
        mask_high = z_arr >= Z_CUT

        piecewise = {}
        for zlab, mask in [('low_z', mask_low), ('high_z', mask_high)]:
            if mask.sum() < 30:
                continue
            r = gls_vpec_env(vpec_residual[mask], rho_std[mask], vpec_err[mask])
            piecewise[zlab] = r
            print(f"    {zlab}: gamma_rho = {r['gamma_v']:.2f} +/- "
                  f"{r['gamma_v_err']:.2f} ({r['significance']:.2f} sigma, n={r['n']})")

        # Permutation test
        p_perm, _ = permutation_test(vpec_residual, rho_std, vpec_err, n_perm=5000)
        print(f"    Permutation p = {p_perm:.4f}")

        results[label] = {
            'radius_mpc_h': float(radius),
            'full_sample': r_full,
            'piecewise': piecewise,
            'p_perm': float(p_perm),
            'density_stats': {
                'mean': float(np.mean(rho)),
                'std': float(np.std(rho)),
                'median': float(np.median(rho)),
            }
        }

    return results


# ============================================================
# Control B: Partial correlation (regress out density)
# ============================================================

def control_b_partial_correlation(vpec_residual, vpec_err, d_signed, densities,
                                  groups, cat_name):
    """
    Control B: Regress out local density, then test d_signed.

    Model hierarchy:
      M0: vpec_resid = alpha                          (null)
      M1: vpec_resid = alpha + gamma_rho * rho        (density only)
      M2: vpec_resid = alpha + gamma_rho * rho + gamma_d * d_signed  (density + void)

    The key test: is M2 significantly better than M1?
    If yes, void geometry adds information beyond raw density.
    If no, d_signed is redundant once you know density.
    """
    print("\n" + "=" * 60)
    print(f"CONTROL B: Partial Correlation ({cat_name})")
    print("=" * 60)

    z_arr = np.array([g['z'] for g in groups])
    n = len(vpec_residual)
    cov = np.diag(vpec_err**2)

    results = {}

    for radius, rho in densities.items():
        label = f"R{radius:.0f}"
        rho_std = (rho - np.mean(rho)) / np.std(rho)
        d_std = (d_signed - np.mean(d_signed)) / np.std(d_signed)

        print(f"\n  --- Density radius = {radius:.0f} Mpc/h ---")

        # Correlation between d_signed and density
        corr_d_rho = np.corrcoef(d_signed, rho)[0, 1]
        print(f"    Pearson(d_signed, density) = {corr_d_rho:.4f}")

        for zlab, mask in [('full', np.ones(n, dtype=bool)),
                           ('low_z', z_arr < Z_CUT),
                           ('high_z', z_arr >= Z_CUT)]:
            if mask.sum() < 30:
                continue

            nm = mask.sum()
            v = vpec_residual[mask]
            w = vpec_err[mask]
            cov_m = np.diag(w**2)

            # M0: null (intercept only)
            X0 = np.ones((nm, 1))
            _, _, chi2_0, _ = gls_fit(v, X0, cov_m)

            # M1: density only
            X1 = np.column_stack([np.ones(nm), rho_std[mask]])
            beta1, _, chi2_1, _ = gls_fit(v, X1, cov_m)

            # M2: density + d_signed
            X2 = np.column_stack([np.ones(nm), rho_std[mask], d_std[mask]])
            beta2, beta2_cov, chi2_2, dof2 = gls_fit(v, X2, cov_m)

            # Test: M1 vs M0 (does density explain vpec?)
            dchi2_density = chi2_0 - chi2_1
            p_density = 1 - stats.chi2.cdf(dchi2_density, 1)

            # Test: M2 vs M1 (does d_signed add anything beyond density?)
            dchi2_void = chi2_1 - chi2_2
            p_void = 1 - stats.chi2.cdf(dchi2_void, 1)

            # Test: M2 vs M0 (does d_signed alone add anything?)
            dchi2_d_alone = chi2_0 - chi2_2

            gamma_rho = beta2[1]
            gamma_rho_err = np.sqrt(beta2_cov[1, 1])
            gamma_d = beta2[2]
            gamma_d_err = np.sqrt(beta2_cov[2, 2])

            sig_d = abs(gamma_d) / gamma_d_err if gamma_d_err > 0 else 0
            sig_rho = abs(gamma_rho) / gamma_rho_err if gamma_rho_err > 0 else 0

            print(f"    {zlab} (n={nm}):")
            print(f"      M0->M1 (density): dchi2 = {dchi2_density:.2f}, "
                  f"p = {p_density:.4e}, gamma_rho = {gamma_rho:.2f} ({sig_rho:.1f}sig)")
            print(f"      M1->M2 (+ void):  dchi2 = {dchi2_void:.2f}, "
                  f"p = {p_void:.4e}, gamma_d = {gamma_d:.2f} ({sig_d:.1f}sig)")
            print(f"      *** d_signed adds {dchi2_void:.2f} chi2 beyond density ***")

            results[f"{label}_{zlab}"] = {
                'radius': float(radius),
                'z_range': zlab,
                'n': nm,
                'corr_d_rho': float(corr_d_rho),
                'chi2_null': float(chi2_0),
                'chi2_density_only': float(chi2_1),
                'chi2_density_plus_void': float(chi2_2),
                'dchi2_density_vs_null': float(dchi2_density),
                'p_density_vs_null': float(p_density),
                'dchi2_void_beyond_density': float(dchi2_void),
                'p_void_beyond_density': float(p_void),
                'gamma_rho_partial': float(gamma_rho),
                'gamma_rho_err': float(gamma_rho_err),
                'gamma_d_partial': float(gamma_d),
                'gamma_d_err': float(gamma_d_err),
                'gamma_d_significance': float(sig_d),
            }

    return results


# ============================================================
# Control C: LCDM linear-theory expectation
# ============================================================

def control_c_lcdm_expectation(vpec_residual, vpec_err, d_signed, densities,
                                groups, cat_name):
    """
    Control C: Compare observed gamma_v to the LCDM expectation.

    In linear perturbation theory:
      v_pec = H * f * delta * r_hat  (roughly)

    where f = Omega_m^0.55 is the growth rate.

    If d_signed is a density proxy, we can estimate the expected
    slope from the density-velocity relation and compare to observed.

    Method: compute the density-velocity slope (vpec vs density),
    then the density-d_signed slope, and multiply to get the
    expected vpec-d_signed slope. Compare to observed.
    """
    print("\n" + "=" * 60)
    print(f"CONTROL C: LCDM Linear-Theory Comparison ({cat_name})")
    print("=" * 60)

    # LCDM parameters
    Omega_m = 0.315
    f_growth = Omega_m**0.55  # growth rate
    H0 = 100.0  # km/s/Mpc/h

    print(f"  f(growth) = {f_growth:.4f}")

    z_arr = np.array([g['z'] for g in groups])
    results = {}

    for radius, rho in densities.items():
        label = f"R{radius:.0f}"

        # Compute overdensity: delta = (rho - <rho>) / <rho>
        rho_mean = np.mean(rho)
        if rho_mean > 0:
            delta = (rho - rho_mean) / rho_mean
        else:
            delta = rho - rho_mean

        print(f"\n  --- R = {radius:.0f} Mpc/h ---")

        # Observed slopes
        # 1. vpec vs density (the expected LCDM relation)
        slope_v_rho, _, r_v_rho, p_v_rho, _ = stats.linregress(delta, vpec_residual)
        print(f"    vpec vs delta: slope = {slope_v_rho:.2f} km/s, "
              f"r = {r_v_rho:.4f}, p = {p_v_rho:.4e}")

        # 2. density vs d_signed (how much does d_signed proxy density?)
        slope_rho_d, _, r_rho_d, p_rho_d, _ = stats.linregress(d_signed, delta)
        print(f"    delta vs d_signed: slope = {slope_rho_d:.4f}, "
              f"r = {r_rho_d:.4f}, p = {p_rho_d:.4e}")

        # 3. Expected gamma_v from chain rule: gamma_v_expected = slope_v_rho * slope_rho_d
        gamma_v_expected = slope_v_rho * slope_rho_d

        # 4. Observed gamma_v
        r_obs = gls_vpec_env(vpec_residual, d_signed, vpec_err)
        gamma_v_observed = r_obs['gamma_v']

        ratio = gamma_v_observed / gamma_v_expected if abs(gamma_v_expected) > 1e-10 else np.inf

        print(f"    gamma_v (observed):  {gamma_v_observed:.2f} +/- {r_obs['gamma_v_err']:.2f}")
        print(f"    gamma_v (expected from density chain): {gamma_v_expected:.2f}")
        print(f"    Ratio (observed/expected): {ratio:.3f}")

        if abs(ratio - 1.0) < 0.3:
            verdict = "CONSISTENT with density-only explanation"
        elif abs(ratio) > 1.3:
            verdict = "EXCESS beyond density-only (potential MTDF signal)"
        else:
            verdict = "DEFICIT relative to density-only"
        print(f"    Verdict: {verdict}")

        # Also do piecewise
        for zlab, mask in [('low_z', z_arr < Z_CUT), ('high_z', z_arr >= Z_CUT)]:
            if mask.sum() < 30:
                continue
            sl_v_rho, _, _, _, _ = stats.linregress(delta[mask], vpec_residual[mask])
            sl_rho_d, _, _, _, _ = stats.linregress(d_signed[mask], delta[mask])
            gv_exp = sl_v_rho * sl_rho_d
            r_z = gls_vpec_env(vpec_residual[mask], d_signed[mask], vpec_err[mask])
            gv_obs = r_z['gamma_v']
            rat = gv_obs / gv_exp if abs(gv_exp) > 1e-10 else np.inf
            print(f"    {zlab}: observed={gv_obs:.2f}, expected={gv_exp:.2f}, ratio={rat:.3f}")

            results[f"{label}_{zlab}"] = {
                'radius': float(radius),
                'z_range': zlab,
                'n': int(mask.sum()),
                'gamma_v_observed': float(gv_obs),
                'gamma_v_observed_err': float(r_z['gamma_v_err']),
                'gamma_v_expected_chain': float(gv_exp),
                'ratio': float(rat),
                'slope_vpec_vs_delta': float(sl_v_rho),
                'slope_delta_vs_dsigned': float(sl_rho_d),
            }

        results[f"{label}_full"] = {
            'radius': float(radius),
            'z_range': 'full',
            'n': len(vpec_residual),
            'gamma_v_observed': float(gamma_v_observed),
            'gamma_v_observed_err': float(r_obs['gamma_v_err']),
            'gamma_v_expected_chain': float(gamma_v_expected),
            'ratio': float(ratio),
            'slope_vpec_vs_delta': float(slope_v_rho),
            'slope_delta_vs_dsigned': float(slope_rho_d),
            'r_vpec_delta': float(r_v_rho),
            'r_delta_dsigned': float(r_rho_d),
            'verdict': verdict,
        }

    return results


# ============================================================
# Plotting
# ============================================================

def plot_control_summary(ctrl_a, ctrl_b, ctrl_c, cat_name, output_dir):
    """Summary plot comparing d_signed signal with density controls."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Control A - density vs d_signed significance
    ax = axes[0]
    radii = []
    sig_density = []
    sig_dsigned_ref = None

    for key, val in ctrl_a.items():
        r = val['radius_mpc_h']
        s = val['full_sample']['significance']
        radii.append(r)
        sig_density.append(s)

    ax.bar(range(len(radii)), sig_density, color='#2196F3', alpha=0.7,
           tick_label=[f'R={r:.0f}' for r in radii])
    ax.set_ylabel('Significance (sigma)')
    ax.set_title(f'Control A: Density Substitution\n({cat_name})')
    ax.axhline(5, color='red', ls='--', alpha=0.5, label='5 sigma')
    ax.legend()

    # Panel 2: Control B - d_signed beyond density
    ax = axes[1]
    labels_b = []
    dchi2_beyond = []
    for key, val in ctrl_b.items():
        if 'full' in key:
            labels_b.append(key.replace('_full', ''))
            dchi2_beyond.append(val['dchi2_void_beyond_density'])

    colors_b = ['#4CAF50' if d > 3.84 else '#FF9800' for d in dchi2_beyond]
    ax.bar(range(len(labels_b)), dchi2_beyond, color=colors_b, alpha=0.7,
           tick_label=labels_b)
    ax.axhline(3.84, color='red', ls='--', label='95% CL')
    ax.axhline(6.63, color='darkred', ls=':', label='99% CL')
    ax.set_ylabel('Delta-chi2 (d_signed beyond density)')
    ax.set_title(f'Control B: Partial Correlation\n({cat_name})')
    ax.legend()

    # Panel 3: Control C - observed/expected ratio
    ax = axes[2]
    labels_c = []
    ratios = []
    for key, val in ctrl_c.items():
        if 'full' in key:
            labels_c.append(key.replace('_full', ''))
            ratios.append(val['ratio'])

    colors_c = ['#F44336' if abs(r - 1) < 0.3 else '#4CAF50' for r in ratios]
    ax.bar(range(len(labels_c)), ratios, color=colors_c, alpha=0.7,
           tick_label=labels_c)
    ax.axhline(1.0, color='black', ls='-', lw=2)
    ax.axhspan(0.7, 1.3, color='gray', alpha=0.15, label='Consistent with LCDM')
    ax.set_ylabel('Ratio (observed / expected from density)')
    ax.set_title(f'Control C: LCDM Expectation\n({cat_name})')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'control_summary_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_partial_piecewise(ctrl_b, cat_name, output_dir):
    """Piecewise view of Control B: does d_signed survive at low-z vs high-z?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by radius
    radii_seen = {}
    for key, val in ctrl_b.items():
        r = val['radius']
        zr = val['z_range']
        if r not in radii_seen:
            radii_seen[r] = {}
        radii_seen[r][zr] = val

    x_pos = 0
    labels = []
    for r in sorted(radii_seen.keys()):
        for zr in ['low_z', 'high_z']:
            if zr not in radii_seen[r]:
                continue
            v = radii_seen[r][zr]
            color = '#2196F3' if zr == 'low_z' else '#FF9800'
            sig = v['gamma_d_significance']
            ax.bar(x_pos, v['dchi2_void_beyond_density'], color=color, alpha=0.7)
            ax.text(x_pos, v['dchi2_void_beyond_density'] + 0.3,
                    f'{sig:.1f}sig', ha='center', fontsize=8)
            labels.append(f'R{r:.0f}\n{zr}')
            x_pos += 1
        x_pos += 0.5  # gap between radii

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(3.84, color='red', ls='--', label='95% CL')
    ax.set_ylabel('Delta-chi2 (d_signed beyond density)')
    ax.set_title(f'Control B Piecewise: Does void geometry add info beyond density?\n({cat_name})')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'control_b_piecewise_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Task 3A CONTROL TESTS: Density vs Void Geometry")
    print("Is the CF4 gamma_v signal MTDF, or just gravity?")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load CF4 (same as task3a)
    print("--- Loading Cosmicflows-4 ---")
    groups = load_cf4()
    z_arr = np.array([g['z'] for g in groups])
    z_max = 0.15
    valid_mask = z_arr < z_max
    groups_valid = [g for g, m in zip(groups, valid_mask) if m]
    z_valid = np.array([g['z'] for g in groups_valid])
    print(f"  {len(groups_valid)} groups at z < {z_max}")

    # Comoving coordinates
    gx, gy, gz = groups_to_comoving(groups_valid)

    # Peculiar velocity residuals (same as task3a)
    vpec_residual = compute_vpec_residuals(groups_valid)

    # Vpec uncertainties (same as task3a)
    e_dmav = np.array([g['e_dmav'] for g in groups_valid])
    dist_mpc = np.array([g['dist'] for g in groups_valid])
    vpec_err = H0_FID * dist_mpc * np.log(10) / 5.0 * e_dmav
    vpec_err = np.maximum(vpec_err, 100.0)

    # Compute local densities at multiple scales
    print("\n--- Computing local galaxy number density ---")
    densities = compute_local_density(gx, gy, gz, radii_mpc_h=(5.0, 10.0, 20.0))

    # Control A: density substitution (void-catalogue independent)
    ctrl_a = control_a_density_substitution(vpec_residual, vpec_err, densities, groups_valid)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Control tests for Task 3A CF4 gamma_v signal',
        'n_groups': len(groups_valid),
        'control_a': ctrl_a,
        'control_b': {},
        'control_c': {},
    }

    # Controls B and C need d_signed, so run per void catalogue
    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"  Void catalogue: {cat_name}")
        print(f"{'='*60}")

        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            print(f"  WARNING: Could not load {cat_name}")
            continue

        d_signed, in_void = compute_environment_fast(gx, gy, gz, vx, vy, vz, vr)

        # Control B: partial correlation
        ctrl_b = control_b_partial_correlation(
            vpec_residual, vpec_err, d_signed, densities, groups_valid, cat_name)
        all_results['control_b'][cat_name] = ctrl_b

        # Control C: LCDM expectation
        ctrl_c = control_c_lcdm_expectation(
            vpec_residual, vpec_err, d_signed, densities, groups_valid, cat_name)
        all_results['control_c'][cat_name] = ctrl_c

        # Plots
        plot_control_summary(ctrl_a, ctrl_b, ctrl_c, cat_name, OUTPUT_DIR)
        plot_partial_piecewise(ctrl_b, cat_name, OUTPUT_DIR)

    # ============================================================
    # Verdict
    # ============================================================
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)

    # Check Control A: is density substitution as strong as d_signed?
    print("\nControl A (Density substitution):")
    for key, val in ctrl_a.items():
        print(f"  {key}: {val['full_sample']['significance']:.1f} sigma "
              f"(p_perm = {val['p_perm']:.4f})")

    # Check Control B: does d_signed survive partial correlation?
    print("\nControl B (d_signed beyond density):")
    for cat_name, cat_results in all_results['control_b'].items():
        for key, val in cat_results.items():
            if 'full' in key:
                print(f"  {cat_name} {key}: dchi2_void_beyond_density = "
                      f"{val['dchi2_void_beyond_density']:.2f} "
                      f"(p = {val['p_void_beyond_density']:.4e}, "
                      f"gamma_d = {val['gamma_d_partial']:.2f} +/- "
                      f"{val['gamma_d_err']:.2f})")

    # Check Control C: observed/expected ratio
    print("\nControl C (Observed/Expected ratio):")
    for cat_name, cat_results in all_results['control_c'].items():
        for key, val in cat_results.items():
            if 'full' in key:
                print(f"  {cat_name} {key}: ratio = {val['ratio']:.3f} "
                      f"({val.get('verdict', 'N/A')})")

    # Overall assessment
    print("\n" + "-" * 70)

    # Count how many controls the signal passes
    passes = 0
    total = 0

    # A: if density substitution is WEAKER than d_signed, point for MTDF
    # B: if d_signed survives beyond density (dchi2 > 3.84), point for MTDF
    # C: if ratio deviates from 1.0 by > 30%, point for MTDF

    for cat_name in all_results['control_b']:
        for key, val in all_results['control_b'][cat_name].items():
            if 'full' in key:
                total += 1
                if val['dchi2_void_beyond_density'] > 3.84:
                    passes += 1

    if passes == 0:
        print("CONCLUSION: d_signed adds NO information beyond local density.")
        print("The 19.6 sigma CF4 signal is standard gravity, not MTDF.")
    elif passes == total:
        print("CONCLUSION: d_signed adds SIGNIFICANT information beyond density")
        print("at all scales. The void geometry matters. Further investigation warranted.")
    else:
        print(f"CONCLUSION: Mixed results ({passes}/{total} scales show void-beyond-density).")
        print("Signal may be partially void-geometric, partially density.")

    print("-" * 70)

    save_results(all_results, 'task3a_control_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
