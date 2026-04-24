#!/usr/bin/env python3
"""
Task 3A: Cosmicflows-4 Peculiar Velocity Environment Correlation

Tests whether peculiar velocity residuals at z < 0.04 correlate with
void/wall environment, using the DESIVAST void catalogue and CF4
group-averaged peculiar velocities.

MTDF prediction: gamma_v should be non-zero below z = 0.04 (the coherence
scale cutoff, beta = 22.7 Mpc) and vanish above.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import os, sys, json
from datetime import datetime

# Reuse existing infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import (load_void_pair, gls_fit,
                    save_results, CATALOGUE_GROUPS, COSMO_VOIDS)

def compute_environment_fast(obj_x, obj_y, obj_z, void_x, void_y, void_z, void_r):
    """
    Vectorized signed distance to nearest void boundary.
    Processes in batches to avoid memory explosion.
    """
    n_obj = len(obj_x)
    n_void = len(void_x)
    d_signed = np.full(n_obj, np.inf)
    in_void = np.zeros(n_obj, dtype=bool)

    batch_size = 500
    vx = void_x.astype(np.float32)
    vy = void_y.astype(np.float32)
    vz = void_z.astype(np.float32)
    vr = void_r.astype(np.float32)

    for start in range(0, n_obj, batch_size):
        end = min(start + batch_size, n_obj)
        bs = end - start
        # Shape: (batch, 1) - (1, n_void) -> (batch, n_void)
        dx = obj_x[start:end, None].astype(np.float32) - vx[None, :]
        dy = obj_y[start:end, None].astype(np.float32) - vy[None, :]
        dz = obj_z[start:end, None].astype(np.float32) - vz[None, :]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        del dx, dy, dz

        # Normalized distance to each void centre
        d_norm = dist / vr[None, :]

        # Nearest void by normalized distance
        idx_nearest = np.argmin(d_norm, axis=1)
        del d_norm
        batch_idx = np.arange(bs)
        nearest_dist = dist[batch_idx, idx_nearest]
        nearest_r = vr[idx_nearest]
        d_signed[start:end] = (nearest_dist - nearest_r) / nearest_r

        # in_void: check if min(dist/R) < 1
        min_d_over_r = np.min(dist / vr[None, :], axis=1)
        in_void[start:end] = min_d_over_r < 1.0
        del dist

        if start % 5000 == 0:
            print(f"    Environment: {start}/{n_obj} done", flush=True)

    return d_signed, in_void


# ---------- Paths ----------
BASE_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'External')
CF4_PATH = os.path.join(BASE_DATA, 'cosmicflows4', 'cf4_groups.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3a_cf4')

# ---------- Cosmology ----------
COSMO = FlatLambdaCDM(H0=100, Om0=0.315)  # Mpc/h for DESIVAST matching
C_KMS = 299792.458

# ---------- z = 0.04 cutoff ----------
Z_CUT = 0.04


def load_cf4():
    """Load Cosmicflows-4 group-averaged catalogue."""
    import csv
    groups = []
    with open(CF4_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vcmb = float(row['Vcmb'])
                ra = float(row['RAJ2000'])
                dec = float(row['DEJ2000'])
                vpec = row['Vpec'].strip()
                dist = row['Dist'].strip()
                dmav = row['DMav'].strip()
                e_dmav = row['e_DMav'].strip()

                # Skip rows with missing essential data
                if not vpec or not dist or not dmav or not e_dmav:
                    continue

                vpec = float(vpec)
                dist = float(dist)
                dmav = float(dmav)
                e_dmav = float(e_dmav)

                z = vcmb / C_KMS

                # Count distance method contributions
                o_tf = int(row['o_DMtf']) if row['o_DMtf'].strip() else 0
                o_fp = int(row['o_DMfp']) if row['o_DMfp'].strip() else 0
                o_snia = int(row['o_DMsnIa']) if row['o_DMsnIa'].strip() else 0
                o_cal = int(row['o_DMcal']) if row['o_DMcal'].strip() else 0
                ngal = int(row['Ngal']) if row['Ngal'].strip() else 1

                groups.append({
                    'ra': ra, 'dec': dec, 'z': z, 'vcmb': vcmb,
                    'dist': dist, 'dmav': dmav, 'e_dmav': e_dmav,
                    'vpec': vpec, 'ngal': ngal,
                    'o_tf': o_tf, 'o_fp': o_fp, 'o_snia': o_snia, 'o_cal': o_cal,
                })
            except (ValueError, KeyError):
                continue
    print(f"  Loaded {len(groups)} CF4 groups with valid data")
    return groups


def groups_to_comoving(groups):
    """Convert group positions to comoving coordinates (Mpc/h) for DESIVAST matching."""
    z = np.array([g['z'] for g in groups])
    ra = np.array([g['ra'] for g in groups])
    dec = np.array([g['dec'] for g in groups])

    # Use the group distance in Mpc, convert to Mpc/h
    # DESIVAST uses H0=100, so Mpc/h = Mpc * h = Mpc * 1.0
    # CF4 distances are luminosity distances in Mpc (h-free)
    # Convert to comoving: d_comoving = d_luminosity / (1+z)
    # Then to Mpc/h: d_comoving_h = d_comoving * h
    # For H0=100 cosmology, h=1.0
    # But CF4 Dist column already uses H0=75 convention...
    # Actually, CF4 distances are h-free (physical Mpc).
    # DESIVAST void positions are in Mpc/h with h=1 (H0=100).
    # So we compute comoving distance from z using the same cosmology as DESIVAST.
    d_c = COSMO.comoving_distance(z).value  # Mpc/h

    ra_r = np.radians(ra)
    dec_r = np.radians(dec)
    x = d_c * np.cos(dec_r) * np.cos(ra_r)
    y = d_c * np.cos(dec_r) * np.sin(ra_r)
    z_cart = d_c * np.sin(dec_r)
    return x, y, z_cart


def compute_vpec_residuals(groups):
    """
    Compute peculiar velocity residuals.

    The CF4 Vpec column already contains peculiar velocities corrected for
    bulk flow. The "residual" here is Vpec itself, as we're testing whether
    environment correlates with peculiar velocity amplitude/sign.

    For a more refined test, we compute:
    v_residual = v_pec - <v_pec>(z-shell)
    to remove any residual z-dependent bulk flow not captured by CF4's correction.
    """
    z_arr = np.array([g['z'] for g in groups])
    vpec_arr = np.array([g['vpec'] for g in groups])

    # Shell-averaged subtraction: remove mean vpec in redshift shells
    # to isolate environment-dependent component
    z_edges = np.arange(0, z_arr.max() + 0.005, 0.005)
    vpec_residual = vpec_arr.copy()

    for i in range(len(z_edges) - 1):
        mask = (z_arr >= z_edges[i]) & (z_arr < z_edges[i + 1])
        if mask.sum() > 10:
            vpec_residual[mask] -= np.median(vpec_arr[mask])

    return vpec_residual


def gls_vpec_env(vpec_residual, d_signed, vpec_err):
    """
    GLS regression: v_residual = alpha + gamma_v * d_signed + epsilon

    Uses diagonal covariance from peculiar velocity uncertainties.
    """
    n = len(vpec_residual)
    # Typical vpec uncertainty: ~300 km/s per group, scaling with 1/sqrt(Ngal)
    cov = np.diag(vpec_err**2)

    # Null model: intercept only
    X_null = np.ones((n, 1))
    # Full model: intercept + d_signed
    X_full = np.column_stack([np.ones(n), d_signed])

    _, _, chi2_null, _ = gls_fit(vpec_residual, X_null, cov)
    beta, beta_cov, chi2_full, dof = gls_fit(vpec_residual, X_full, cov)

    dchi2 = chi2_null - chi2_full
    p = 1 - stats.chi2.cdf(dchi2, 1)

    gamma_v = beta[1]
    gamma_v_err = np.sqrt(beta_cov[1, 1])

    return {
        'gamma_v': float(gamma_v),
        'gamma_v_err': float(gamma_v_err),
        'significance': float(abs(gamma_v) / gamma_v_err),
        'delta_chi2': float(dchi2),
        'p_value': float(p),
        'chi2_null': float(chi2_null),
        'chi2_full': float(chi2_full),
        'dof': int(dof),
        'n': int(n),
    }


def permutation_test(vpec_residual, d_signed, vpec_err, n_perm=10000):
    """Permutation test: shuffle d_signed, measure gamma_v distribution."""
    # Observed gamma_v
    obs = gls_vpec_env(vpec_residual, d_signed, vpec_err)
    obs_gamma = obs['gamma_v']

    gamma_perm = np.zeros(n_perm)
    for i in range(n_perm):
        d_shuffled = np.random.permutation(d_signed)
        n = len(vpec_residual)
        # Fast: use weighted LS directly instead of gls_fit for speed
        w = 1.0 / vpec_err**2
        X = np.column_stack([np.ones(n), d_shuffled])
        WX = X * w[:, np.newaxis]
        XtWX = X.T @ WX
        beta = np.linalg.solve(XtWX, WX.T @ vpec_residual)
        gamma_perm[i] = beta[1]

    p_perm = np.mean(np.abs(gamma_perm) >= np.abs(obs_gamma))
    return p_perm, gamma_perm


def bootstrap_test(vpec_residual, d_signed, vpec_err, n_boot=5000):
    """Bootstrap: resample with replacement, measure gamma_v distribution."""
    n = len(vpec_residual)
    gamma_boot = np.zeros(n_boot)
    w = 1.0 / vpec_err**2

    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        vr = vpec_residual[idx]
        ds = d_signed[idx]
        wi = w[idx]
        X = np.column_stack([np.ones(n), ds])
        WX = X * wi[:, np.newaxis]
        XtWX = X.T @ WX
        beta = np.linalg.solve(XtWX, WX.T @ vr)
        gamma_boot[i] = beta[1]

    return gamma_boot


def piecewise_split(groups, d_signed, vpec_residual, vpec_err, z_cut=Z_CUT):
    """Split at z_cut and measure gamma_v in each bin."""
    z_arr = np.array([g['z'] for g in groups])

    mask_low = z_arr < z_cut
    mask_high = z_arr >= z_cut

    results = {}
    for label, mask in [('low_z', mask_low), ('high_z', mask_high)]:
        if mask.sum() < 20:
            results[label] = {'n': int(mask.sum()), 'gamma_v': None}
            continue
        r = gls_vpec_env(vpec_residual[mask], d_signed[mask], vpec_err[mask])
        results[label] = r
        results[label]['z_range'] = [float(z_arr[mask].min()), float(z_arr[mask].max())]

    return results


def sliding_z_scan(groups, d_signed, vpec_residual, vpec_err,
                   z_min=0.005, z_max=0.05, z_step=0.002):
    """Scan z_cut and measure delta-chi2 at each split point."""
    z_arr = np.array([g['z'] for g in groups])
    z_cuts = np.arange(z_min, z_max + z_step, z_step)
    results = []

    for zc in z_cuts:
        mask_low = z_arr < zc
        if mask_low.sum() < 30 or (~mask_low).sum() < 30:
            continue
        r_low = gls_vpec_env(vpec_residual[mask_low], d_signed[mask_low], vpec_err[mask_low])
        r_high = gls_vpec_env(vpec_residual[~mask_low], d_signed[~mask_low], vpec_err[~mask_low])
        results.append({
            'z_cut': float(zc),
            'gamma_low': r_low['gamma_v'],
            'gamma_low_err': r_low['gamma_v_err'],
            'gamma_low_sig': r_low['significance'],
            'dchi2_low': r_low['delta_chi2'],
            'n_low': r_low['n'],
            'gamma_high': r_high['gamma_v'],
            'gamma_high_err': r_high['gamma_v_err'],
            'gamma_high_sig': r_high['significance'],
            'dchi2_high': r_high['delta_chi2'],
            'n_high': r_high['n'],
        })

    return results


def method_split(groups, d_signed, vpec_residual, vpec_err):
    """Test TF-only vs FP-only subsamples."""
    o_tf = np.array([g['o_tf'] for g in groups])
    o_fp = np.array([g['o_fp'] for g in groups])

    results = {}
    # TF-only: groups where TF is the primary method
    tf_mask = (o_tf > 0) & (o_fp == 0)
    if tf_mask.sum() > 50:
        r = gls_vpec_env(vpec_residual[tf_mask], d_signed[tf_mask], vpec_err[tf_mask])
        r['n_groups'] = int(tf_mask.sum())
        results['TF_only'] = r

    # FP-only: groups where FP is the primary method
    fp_mask = (o_fp > 0) & (o_tf == 0)
    if fp_mask.sum() > 50:
        r = gls_vpec_env(vpec_residual[fp_mask], d_signed[fp_mask], vpec_err[fp_mask])
        r['n_groups'] = int(fp_mask.sum())
        results['FP_only'] = r

    return results


def ngc_sgc_split(groups, d_signed, vpec_residual, vpec_err):
    """NGC (dec > 0 proxy) vs SGC split for systematics."""
    dec = np.array([g['dec'] for g in groups])
    results = {}
    for label, mask in [('NGC', dec > 0), ('SGC', dec <= 0)]:
        if mask.sum() < 50:
            continue
        r = gls_vpec_env(vpec_residual[mask], d_signed[mask], vpec_err[mask])
        r['n_groups'] = int(mask.sum())
        results[label] = r
    return results


# ---------- Plotting ----------

def plot_scatter(groups, d_signed, vpec_residual, gamma_result, cat_name, output_dir):
    """Scatter: vpec residual vs d_signed, coloured by z."""
    z_arr = np.array([g['z'] for g in groups])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, mask) in zip(axes, [('z < 0.04', z_arr < Z_CUT),
                                         ('z >= 0.04', z_arr >= Z_CUT)]):
        if mask.sum() == 0:
            continue
        sc = ax.scatter(d_signed[mask], vpec_residual[mask],
                       c=z_arr[mask], cmap='viridis', s=1, alpha=0.3, rasterized=True)
        plt.colorbar(sc, ax=ax, label='z')

        # Binned medians
        d_bins = np.linspace(d_signed[mask].min(), d_signed[mask].max(), 20)
        d_centres = 0.5 * (d_bins[:-1] + d_bins[1:])
        vmed = np.zeros(len(d_centres))
        verr = np.zeros(len(d_centres))
        for i in range(len(d_centres)):
            bin_mask = mask & (d_signed >= d_bins[i]) & (d_signed < d_bins[i+1])
            if bin_mask.sum() > 5:
                vmed[i] = np.median(vpec_residual[bin_mask])
                verr[i] = np.std(vpec_residual[bin_mask]) / np.sqrt(bin_mask.sum())
        ax.errorbar(d_centres, vmed, verr, fmt='ro-', ms=4, lw=1.5, zorder=5)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.axvline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('d_signed (void boundary units)')
        ax.set_ylabel('v_pec residual (km/s)')
        ax.set_title(f'{label} ({mask.sum()} groups)')

    fig.suptitle(f'Task 3A: CF4 Peculiar Velocity vs Environment ({cat_name})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'task3a_scatter_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_piecewise(piecewise_results, cat_name, output_dir):
    """Bar chart: gamma_v in low-z vs high-z bins."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    gammas = []
    errs = []
    colors = []

    for label, r in piecewise_results.items():
        if r.get('gamma_v') is None:
            continue
        labels.append(f"{label}\n(n={r['n']})")
        gammas.append(r['gamma_v'])
        errs.append(r['gamma_v_err'])
        colors.append('#2196F3' if 'low' in label else '#FF9800')

    x = np.arange(len(labels))
    ax.bar(x, gammas, yerr=errs, color=colors, alpha=0.7, capsize=5)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('gamma_v (km/s per d_signed)')
    ax.set_title(f'Task 3A: Piecewise z-split ({cat_name})')

    for i, (g, e) in enumerate(zip(gammas, errs)):
        sig = abs(g) / e if e > 0 else 0
        ax.text(i, g + (1 if g >= 0 else -1) * e * 1.2,
                f'{g:.1f} +/- {e:.1f}\n({sig:.1f}sigma)',
                ha='center', va='bottom' if g >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f'task3a_piecewise_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_sliding_scan(scan_results, cat_name, output_dir):
    """Sliding z_cut scan: delta-chi2 and gamma_v vs z_cut."""
    if not scan_results:
        return

    z_cuts = [r['z_cut'] for r in scan_results]
    dchi2_low = [r['dchi2_low'] for r in scan_results]
    gamma_low = [r['gamma_low'] for r in scan_results]
    gamma_low_err = [r['gamma_low_err'] for r in scan_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(z_cuts, dchi2_low, 'b-o', ms=4)
    ax1.axvline(Z_CUT, color='red', ls='--', label=f'z = {Z_CUT}')
    ax1.axhline(3.84, color='gray', ls=':', alpha=0.7, label='95% CL (dchi2=3.84)')
    ax1.set_ylabel('Delta-chi2 (low-z bin)')
    ax1.set_title(f'Task 3A: Sliding z-cut scan ({cat_name})')
    ax1.legend()

    ax2.errorbar(z_cuts, gamma_low, gamma_low_err, fmt='b-o', ms=4, capsize=3)
    ax2.axhline(0, color='gray', ls='--')
    ax2.axvline(Z_CUT, color='red', ls='--')
    ax2.set_xlabel('z_cut')
    ax2.set_ylabel('gamma_v (low-z bin)')

    plt.tight_layout()
    path = os.path.join(output_dir, f'task3a_zscan_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_void_wall_distributions(groups, d_signed, vpec_residual, cat_name, output_dir):
    """Void vs wall peculiar velocity residual distributions."""
    z_arr = np.array([g['z'] for g in groups])
    mask_lowz = z_arr < Z_CUT

    in_void = d_signed < 0
    in_wall = d_signed >= 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, zmask) in zip(axes, [('z < 0.04', mask_lowz),
                                          ('z >= 0.04', ~mask_lowz)]):
        void_v = vpec_residual[zmask & in_void]
        wall_v = vpec_residual[zmask & in_wall]

        if len(void_v) > 0:
            ax.hist(void_v, bins=50, alpha=0.6, color='blue', density=True,
                    label=f'Void (n={len(void_v)}, med={np.median(void_v):.0f})')
        if len(wall_v) > 0:
            ax.hist(wall_v, bins=50, alpha=0.6, color='red', density=True,
                    label=f'Wall (n={len(wall_v)}, med={np.median(wall_v):.0f})')

        ax.axvline(0, color='gray', ls='--')
        ax.set_xlabel('v_pec residual (km/s)')
        ax.set_ylabel('Density')
        ax.set_title(f'{label}')
        ax.legend()

    fig.suptitle(f'Task 3A: Void/Wall Vpec Distributions ({cat_name})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'task3a_distributions_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------- Main ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Task 3A: Cosmicflows-4 Peculiar Velocity Environment Correlation")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Load CF4
    print("--- Step 1: Load Cosmicflows-4 ---")
    groups = load_cf4()
    z_arr = np.array([g['z'] for g in groups])
    print(f"  z range: [{z_arr.min():.4f}, {z_arr.max():.4f}]")
    print(f"  Groups at z < {Z_CUT}: {(z_arr < Z_CUT).sum()}")
    print(f"  Groups at z >= {Z_CUT}: {(z_arr >= Z_CUT).sum()}")

    # DESIVAST voids cover z < 0.15, but CF4 mostly z < 0.05
    # We restrict to z < 0.15 for DESIVAST overlap
    z_max_desivast = 0.15
    valid_mask = z_arr < z_max_desivast
    print(f"  Groups at z < {z_max_desivast} (DESIVAST overlap): {valid_mask.sum()}")

    # Filter groups
    groups_valid = [g for g, m in zip(groups, valid_mask) if m]
    z_valid = np.array([g['z'] for g in groups_valid])

    # Step 2: Convert to comoving coordinates
    print("\n--- Step 2: Cross-match with DESIVAST voids ---")
    gx, gy, gz = groups_to_comoving(groups_valid)

    # Step 3: Compute vpec residuals
    print("\n--- Step 3: Compute peculiar velocity residuals ---")
    vpec_residual = compute_vpec_residuals(groups_valid)
    vpec_raw = np.array([g['vpec'] for g in groups_valid])
    print(f"  Raw vpec: mean={vpec_raw.mean():.1f}, std={vpec_raw.std():.1f} km/s")
    print(f"  Residual: mean={vpec_residual.mean():.1f}, std={vpec_residual.std():.1f} km/s")

    # Estimate vpec uncertainties
    # CF4 distance uncertainties -> vpec uncertainties via v_pec ~ H0*d - cz
    # sigma_vpec ~ H0 * sigma_d ~ H0 * d * ln(10)/5 * sigma_DM
    e_dmav = np.array([g['e_dmav'] for g in groups_valid])
    dist_mpc = np.array([g['dist'] for g in groups_valid])
    ngal = np.array([g['ngal'] for g in groups_valid])
    # sigma_vpec from distance modulus uncertainty
    # v = H0*d, sigma_v = H0 * sigma_d = H0 * d * ln(10)/5 * sigma_DM
    H0_fid = 75.0  # CF4 fiducial
    vpec_err = H0_fid * dist_mpc * np.log(10) / 5.0 * e_dmav
    vpec_err = np.maximum(vpec_err, 100.0)  # Floor at 100 km/s
    print(f"  Vpec uncertainty: median={np.median(vpec_err):.0f} km/s")

    # Run analysis for each void catalogue
    all_results = {'timestamp': datetime.now().isoformat(), 'catalogues': {}}

    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        print(f"\n{'='*50}")
        print(f"  Catalogue: {cat_name}")
        print(f"{'='*50}")

        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            print(f"  WARNING: Could not load {cat_name} voids")
            continue

        # Compute environment (vectorized)
        d_signed, in_void = compute_environment_fast(
            gx, gy, gz, vx, vy, vz, vr)

        n_void = in_void.sum()
        n_wall = (~in_void).sum()
        print(f"  Void: {n_void}, Wall: {n_wall}")

        # Step 4: GLS regression
        print(f"\n  --- GLS regression (full sample) ---")
        full_result = gls_vpec_env(vpec_residual, d_signed, vpec_err)
        print(f"    gamma_v = {full_result['gamma_v']:.2f} +/- {full_result['gamma_v_err']:.2f}")
        print(f"    Significance: {full_result['significance']:.2f} sigma")
        print(f"    Delta-chi2 = {full_result['delta_chi2']:.3f}, p = {full_result['p_value']:.4f}")

        # Step 4b: Permutation test
        print(f"  --- Permutation test (10,000 shuffles) ---")
        p_perm, gamma_perm = permutation_test(vpec_residual, d_signed, vpec_err, n_perm=10000)
        print(f"    p_perm = {p_perm:.4f}")

        # Step 4c: Bootstrap
        print(f"  --- Bootstrap (5,000 resamples) ---")
        gamma_boot = bootstrap_test(vpec_residual, d_signed, vpec_err, n_boot=5000)
        boot_ci = np.percentile(gamma_boot, [2.5, 97.5])
        print(f"    gamma_v bootstrap: {np.mean(gamma_boot):.2f} +/- {np.std(gamma_boot):.2f}")
        print(f"    95% CI: [{boot_ci[0]:.2f}, {boot_ci[1]:.2f}]")

        # Step 5: Piecewise z-split
        print(f"\n  --- Piecewise z-split at z = {Z_CUT} ---")
        pw = piecewise_split(groups_valid, d_signed, vpec_residual, vpec_err)
        for label, r in pw.items():
            if r.get('gamma_v') is not None:
                print(f"    {label}: gamma_v = {r['gamma_v']:.2f} +/- {r['gamma_v_err']:.2f} "
                      f"({r['significance']:.2f} sigma, n={r['n']})")

        # Step 5b: Sliding z-cut scan
        print(f"  --- Sliding z-cut scan ---")
        scan = sliding_z_scan(groups_valid, d_signed, vpec_residual, vpec_err)
        if scan:
            best = max(scan, key=lambda r: r['dchi2_low'])
            print(f"    Best z_cut = {best['z_cut']:.3f} (dchi2_low = {best['dchi2_low']:.3f})")

        # Step 6: Systematics
        print(f"\n  --- Systematics: NGC/SGC split ---")
        ngc_sgc = ngc_sgc_split(groups_valid, d_signed, vpec_residual, vpec_err)
        for label, r in ngc_sgc.items():
            print(f"    {label}: gamma_v = {r['gamma_v']:.2f} +/- {r['gamma_v_err']:.2f} "
                  f"({r['significance']:.2f} sigma, n={r['n_groups']})")

        print(f"\n  --- Systematics: Distance method split ---")
        method = method_split(groups_valid, d_signed, vpec_residual, vpec_err)
        for label, r in method.items():
            print(f"    {label}: gamma_v = {r['gamma_v']:.2f} +/- {r['gamma_v_err']:.2f} "
                  f"({r['significance']:.2f} sigma, n={r['n_groups']})")

        # Plots
        plot_scatter(groups_valid, d_signed, vpec_residual, full_result, cat_name, OUTPUT_DIR)
        plot_piecewise(pw, cat_name, OUTPUT_DIR)
        plot_sliding_scan(scan, cat_name, OUTPUT_DIR)
        plot_void_wall_distributions(groups_valid, d_signed, vpec_residual, cat_name, OUTPUT_DIR)

        # Store results
        all_results['catalogues'][cat_name] = {
            'full_sample': full_result,
            'p_perm': float(p_perm),
            'bootstrap_mean': float(np.mean(gamma_boot)),
            'bootstrap_std': float(np.std(gamma_boot)),
            'bootstrap_ci95': [float(boot_ci[0]), float(boot_ci[1])],
            'piecewise': pw,
            'sliding_scan_best': best if scan else None,
            'ngc_sgc': ngc_sgc,
            'method_split': method,
            'n_void': int(n_void),
            'n_wall': int(n_wall),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Task 3A Cosmicflows-4 Peculiar Velocity Environment Correlation")
    print("=" * 70)
    all_results['summary'] = {}
    for cat_name, cr in all_results['catalogues'].items():
        fs = cr['full_sample']
        pw = cr['piecewise']
        print(f"  {cat_name}:")
        print(f"    Full: gamma_v = {fs['gamma_v']:.2f} +/- {fs['gamma_v_err']:.2f} "
              f"({fs['significance']:.2f} sigma)")
        if pw.get('low_z', {}).get('gamma_v') is not None:
            lo = pw['low_z']
            print(f"    z < {Z_CUT}: gamma_v = {lo['gamma_v']:.2f} +/- {lo['gamma_v_err']:.2f} "
                  f"({lo['significance']:.2f} sigma, n={lo['n']})")
        if pw.get('high_z', {}).get('gamma_v') is not None:
            hi = pw['high_z']
            print(f"    z >= {Z_CUT}: gamma_v = {hi['gamma_v']:.2f} +/- {hi['gamma_v_err']:.2f} "
                  f"({hi['significance']:.2f} sigma, n={hi['n']})")
        print(f"    Permutation p = {cr['p_perm']:.4f}")
        all_results['summary'][cat_name] = {
            'gamma_v': fs['gamma_v'],
            'gamma_v_err': fs['gamma_v_err'],
            'significance': fs['significance'],
        }

    save_results(all_results, 'task3a_cf4_results.json', OUTPUT_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
