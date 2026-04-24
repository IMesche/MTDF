#!/usr/bin/env python3
"""
Task 3B: 2MTF Tully-Fisher Environment Residuals

Tests whether Tully-Fisher distance residuals at z < 0.04 show an
environment-dependent offset between void and wall galaxies.

Key advantage: 2MTF covers z < 0.033, so 100% of the sample is in the
MTDF-predicted active regime (below z = 0.04 coherence scale).

TF is a completely independent distance indicator from SNe (HI linewidth
vs infrared luminosity, not lightcurve standardization).

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import os, sys, csv, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import (load_void_pair, gls_fit,
                    save_results, CATALOGUE_GROUPS, COSMO_VOIDS)

# Import fast environment computation from Task 3A
sys.path.insert(0, os.path.dirname(__file__))
from task3a_cosmicflows4_vpec import compute_environment_fast

# ---------- Paths ----------
BASE_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'External')
TABLE1_PATH = os.path.join(BASE_DATA, '2mtf', '2mtf_table1.csv')
TABLE2_PATH = os.path.join(BASE_DATA, '2mtf', '2mtf_table2.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3b_2mtf')

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)
C_KMS = 299792.458


def load_2mtf():
    """Load and merge 2MTF tables 1 and 2."""
    # Table 1: photometry + HI widths
    t1 = {}
    with open(TABLE1_PATH) as f:
        for row in csv.DictReader(f):
            t1[row['2MASX'].strip()] = row

    # Table 2: distances + peculiar velocities
    galaxies = []
    with open(TABLE2_PATH) as f:
        for row in csv.DictReader(f):
            name = row['2MASX'].strip()
            phot = t1.get(name)
            if not phot:
                continue

            try:
                cz_cmb = float(row['czCMB'])
                z = cz_cmb / C_KMS
                ra = float(phot['RAJ2000'])
                dec = float(phot['DEJ2000'])

                # HI linewidth
                whic = float(phot['WHIc'])
                e_whic = float(phot['e_WHIc'])

                # Magnitudes
                k_mag = float(phot['Kmag'])
                h_mag = float(phot['Hmag'])
                j_mag = float(phot['Jmag'])
                e_k = float(phot['e_Kmag'])
                e_h = float(phot['e_Hmag'])
                e_j = float(phot['e_Jmag'])

                # Log distances and peculiar velocities per band
                logd_k = row['logdK'].strip()
                logd_h = row['logdH'].strip()
                logd_j = row['logdJ'].strip()
                vpec_k = row['VpecK'].strip()
                e_vpec_k = row['e_VpecK'].strip()
                flag = row['Flag'].strip()

                if not logd_k or not vpec_k or not e_vpec_k:
                    continue

                logd_k = float(logd_k)
                logd_h = float(logd_h) if logd_h else np.nan
                logd_j = float(logd_j) if logd_j else np.nan
                vpec_k = float(vpec_k)
                e_vpec_k = float(e_vpec_k)

                galaxies.append({
                    'name': name, 'ra': ra, 'dec': dec, 'z': z,
                    'cz_cmb': cz_cmb,
                    'whic': whic, 'e_whic': e_whic,
                    'k_mag': k_mag, 'h_mag': h_mag, 'j_mag': j_mag,
                    'e_k': e_k, 'e_h': e_h, 'e_j': e_j,
                    'logd_k': logd_k, 'logd_h': logd_h, 'logd_j': logd_j,
                    'vpec_k': vpec_k, 'e_vpec_k': e_vpec_k,
                    'flag': flag,
                })
            except (ValueError, KeyError):
                continue

    print(f"  Loaded {len(galaxies)} 2MTF galaxies with complete data")
    return galaxies


def fit_tully_fisher(galaxies, band='k'):
    """
    Fit the standard TF relation: M = a + b * (log W - 2.5)
    where W = HI linewidth, M = absolute magnitude.

    Returns: a, b, residuals (delta_mu), sigma_TF
    """
    log_w = np.log10(np.array([g['whic'] for g in galaxies]))
    mag = np.array([g[f'{band}_mag'] for g in galaxies])
    z = np.array([g['z'] for g in galaxies])

    # Distance modulus from redshift (Hubble law)
    # mu = 5*log10(cz/H0) + 25 for nearby universe
    # Use cosmological distance modulus
    mu_cosmo = COSMO_VOIDS.distmod(z).value  # H0=100 convention

    # Absolute magnitude
    abs_mag = mag - mu_cosmo

    # Fit TF: M = a + b * (log W - 2.5)
    eta = log_w - 2.5
    # Linear regression
    X = np.column_stack([np.ones(len(eta)), eta])
    beta = np.linalg.lstsq(X, abs_mag, rcond=None)[0]
    a, b = beta

    # TF-predicted absolute magnitude
    m_predicted = a + b * eta

    # Distance modulus residual
    # delta_mu = (mag - m_predicted) - mu_cosmo
    # = abs_mag - m_predicted
    delta_mu = abs_mag - m_predicted

    sigma_tf = np.std(delta_mu)

    print(f"  TF fit ({band}-band): a={a:.3f}, b={b:.3f}, sigma={sigma_tf:.3f} mag")
    print(f"    log W range: [{log_w.min():.3f}, {log_w.max():.3f}]")
    print(f"    Abs mag range: [{abs_mag.min():.2f}, {abs_mag.max():.2f}]")

    return a, b, delta_mu, sigma_tf


def gls_tf_env(delta_mu, d_signed, mu_err):
    """GLS regression: delta_mu = alpha + gamma_TF * d_signed."""
    n = len(delta_mu)
    cov = np.diag(mu_err**2)

    X_null = np.ones((n, 1))
    X_full = np.column_stack([np.ones(n), d_signed])

    _, _, chi2_null, _ = gls_fit(delta_mu, X_null, cov)
    beta, beta_cov, chi2_full, dof = gls_fit(delta_mu, X_full, cov)

    dchi2 = chi2_null - chi2_full
    p = 1 - stats.chi2.cdf(dchi2, 1)

    gamma_tf = beta[1]
    gamma_tf_err = np.sqrt(beta_cov[1, 1])

    return {
        'gamma_tf': float(gamma_tf),
        'gamma_tf_err': float(gamma_tf_err),
        'significance': float(abs(gamma_tf) / gamma_tf_err),
        'delta_chi2': float(dchi2),
        'p_value': float(p),
        'n': int(n),
    }


def permutation_test(delta_mu, d_signed, mu_err, n_perm=10000):
    """Permutation test for gamma_TF."""
    obs = gls_tf_env(delta_mu, d_signed, mu_err)
    obs_gamma = obs['gamma_tf']

    w = 1.0 / mu_err**2
    n = len(delta_mu)
    gamma_perm = np.zeros(n_perm)

    for i in range(n_perm):
        d_shuffled = np.random.permutation(d_signed)
        X = np.column_stack([np.ones(n), d_shuffled])
        WX = X * w[:, np.newaxis]
        XtWX = X.T @ WX
        beta = np.linalg.solve(XtWX, WX.T @ delta_mu)
        gamma_perm[i] = beta[1]

    p_perm = np.mean(np.abs(gamma_perm) >= np.abs(obs_gamma))
    return p_perm, gamma_perm


def bootstrap_test(delta_mu, d_signed, mu_err, n_boot=5000):
    """Bootstrap for gamma_TF confidence interval."""
    n = len(delta_mu)
    w = 1.0 / mu_err**2
    gamma_boot = np.zeros(n_boot)

    for i in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        X = np.column_stack([np.ones(n), d_signed[idx]])
        wi = w[idx]
        WX = X * wi[:, np.newaxis]
        XtWX = X.T @ WX
        beta = np.linalg.solve(XtWX, WX.T @ delta_mu[idx])
        gamma_boot[i] = beta[1]

    return gamma_boot


def void_wall_tf_comparison(galaxies, in_void, band='k'):
    """Fit separate TF relations for void and wall subsamples."""
    log_w = np.log10(np.array([g['whic'] for g in galaxies]))
    mag = np.array([g[f'{band}_mag'] for g in galaxies])
    z = np.array([g['z'] for g in galaxies])
    mu_cosmo = COSMO_VOIDS.distmod(z).value
    abs_mag = mag - mu_cosmo
    eta = log_w - 2.5

    results = {}
    for label, mask in [('void', in_void), ('wall', ~in_void)]:
        if mask.sum() < 20:
            results[label] = {'n': int(mask.sum())}
            continue
        X = np.column_stack([np.ones(mask.sum()), eta[mask]])
        beta = np.linalg.lstsq(X, abs_mag[mask], rcond=None)[0]
        resid = abs_mag[mask] - X @ beta
        sigma = np.std(resid)

        # Standard errors via (X'X)^-1 * sigma^2
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma**2)

        results[label] = {
            'n': int(mask.sum()),
            'zero_point': float(beta[0]),
            'zero_point_err': float(se[0]),
            'slope': float(beta[1]),
            'slope_err': float(se[1]),
            'scatter': float(sigma),
        }

    # Zero-point offset test
    if 'zero_point' in results.get('void', {}) and 'zero_point' in results.get('wall', {}):
        dz = results['void']['zero_point'] - results['wall']['zero_point']
        dz_err = np.sqrt(results['void']['zero_point_err']**2 +
                         results['wall']['zero_point_err']**2)
        results['zero_point_offset'] = float(dz)
        results['zero_point_offset_err'] = float(dz_err)
        results['zero_point_offset_sig'] = float(abs(dz) / dz_err) if dz_err > 0 else 0

    return results


# ---------- Plotting ----------

def plot_tf_relation(galaxies, in_void, delta_mu, d_signed, band, cat_name, output_dir):
    """TF relation and residuals."""
    log_w = np.log10(np.array([g['whic'] for g in galaxies]))
    mag = np.array([g[f'{band}_mag'] for g in galaxies])
    z = np.array([g['z'] for g in galaxies])
    mu_cosmo = COSMO_VOIDS.distmod(z).value
    abs_mag = mag - mu_cosmo

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: TF relation colour-coded by environment
    ax = axes[0]
    ax.scatter(log_w[in_void], abs_mag[in_void], s=5, alpha=0.5,
               c='blue', label=f'Void ({in_void.sum()})')
    ax.scatter(log_w[~in_void], abs_mag[~in_void], s=5, alpha=0.5,
               c='red', label=f'Wall ({(~in_void).sum()})')
    ax.set_xlabel('log W_HI')
    ax.set_ylabel(f'M_{band.upper()} (abs mag)')
    ax.invert_yaxis()
    ax.legend()
    ax.set_title('TF Relation')

    # Panel 2: Residual vs d_signed
    ax = axes[1]
    sc = ax.scatter(d_signed, delta_mu, c=z, cmap='viridis', s=5, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label='z')

    # Binned medians
    d_bins = np.linspace(np.percentile(d_signed, 1), np.percentile(d_signed, 99), 15)
    d_centres = 0.5 * (d_bins[:-1] + d_bins[1:])
    for i in range(len(d_centres)):
        bm = (d_signed >= d_bins[i]) & (d_signed < d_bins[i+1])
        if bm.sum() > 5:
            med = np.median(delta_mu[bm])
            err = np.std(delta_mu[bm]) / np.sqrt(bm.sum())
            ax.errorbar(d_centres[i], med, err, fmt='ro', ms=5, zorder=5)

    ax.axhline(0, color='gray', ls='--')
    ax.axvline(0, color='gray', ls=':')
    ax.set_xlabel('d_signed')
    ax.set_ylabel('TF residual (mag)')
    ax.set_title('Residual vs Environment')

    # Panel 3: Void vs wall residual histograms
    ax = axes[2]
    void_r = delta_mu[in_void]
    wall_r = delta_mu[~in_void]
    if len(void_r) > 0:
        ax.hist(void_r, bins=30, alpha=0.6, color='blue', density=True,
                label=f'Void (med={np.median(void_r):.3f})')
    if len(wall_r) > 0:
        ax.hist(wall_r, bins=30, alpha=0.6, color='red', density=True,
                label=f'Wall (med={np.median(wall_r):.3f})')
    ax.axvline(0, color='gray', ls='--')
    ax.set_xlabel('TF residual (mag)')
    ax.set_ylabel('Density')
    ax.set_title('Void/Wall Distributions')
    ax.legend()

    fig.suptitle(f'Task 3B: 2MTF Tully-Fisher Environment ({cat_name}, {band.upper()}-band)',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'task3b_tf_{cat_name.lower()}_{band}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison(all_results, output_dir):
    """Compare gamma_TF across catalogues and bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = 0
    labels = []
    for cat_name, cat_r in all_results['catalogues'].items():
        for band, br in cat_r.get('bands', {}).items():
            labels.append(f'{cat_name}\n{band.upper()}-band')
            g = br['gls']['gamma_tf']
            e = br['gls']['gamma_tf_err']
            sig = br['gls']['significance']
            color = '#2196F3' if sig > 2 else '#FF9800' if sig > 1 else '#999'
            ax.errorbar(x_pos, g, e, fmt='o', color=color, ms=8, capsize=5)
            ax.text(x_pos, g + e * 1.3, f'{sig:.1f}sigma', ha='center', fontsize=8)
            x_pos += 1

    ax.axhline(0, color='gray', ls='--')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('gamma_TF (mag per d_signed)')
    ax.set_title('Task 3B: TF Environment Coefficient Comparison')
    plt.tight_layout()
    path = os.path.join(output_dir, 'task3b_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------- Main ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Task 3B: 2MTF Tully-Fisher Environment Residuals")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Load 2MTF
    print("--- Step 1: Load 2MTF ---")
    galaxies = load_2mtf()
    z_arr = np.array([g['z'] for g in galaxies])
    print(f"  z range: [{z_arr.min():.4f}, {z_arr.max():.4f}]")
    print(f"  All at z < 0.04: {(z_arr < 0.04).sum()} / {len(z_arr)}")
    print(f"  (100% in MTDF-active regime)")

    # Step 2: Convert to comoving
    print("\n--- Step 2: Cross-match with DESIVAST voids ---")
    ra = np.array([g['ra'] for g in galaxies])
    dec = np.array([g['dec'] for g in galaxies])
    d_c = COSMO.comoving_distance(z_arr).value
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    gx = d_c * np.cos(dec_r) * np.cos(ra_r)
    gy = d_c * np.cos(dec_r) * np.sin(ra_r)
    gz = d_c * np.sin(dec_r)

    all_results = {'timestamp': datetime.now().isoformat(), 'catalogues': {}}

    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        print(f"\n{'='*50}")
        print(f"  Catalogue: {cat_name}")
        print(f"{'='*50}")

        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            print(f"  WARNING: Could not load {cat_name} voids")
            continue

        d_signed, in_void = compute_environment_fast(gx, gy, gz, vx, vy, vz, vr)
        n_void = in_void.sum()
        n_wall = (~in_void).sum()
        print(f"  Void: {n_void}, Wall: {n_wall}")

        cat_results = {'n_void': int(n_void), 'n_wall': int(n_wall), 'bands': {}}

        # Step 3: Fit TF and compute residuals for each band
        for band in ['k', 'h', 'j']:
            print(f"\n  --- {band.upper()}-band TF fit ---")

            a, b, delta_mu, sigma_tf = fit_tully_fisher(galaxies, band)

            # TF residual uncertainty: combine distance modulus error and TF scatter
            # Using published logd errors -> sigma_mu = 5 * sigma_logd
            logd_col = f'logd_{band}'
            # For simplicity, use TF scatter as per-galaxy uncertainty
            mu_err = np.full(len(delta_mu), sigma_tf)

            # Step 4: GLS regression
            print(f"\n  --- GLS regression ---")
            gls_result = gls_tf_env(delta_mu, d_signed, mu_err)
            print(f"    gamma_TF = {gls_result['gamma_tf']:.4f} +/- {gls_result['gamma_tf_err']:.4f}")
            print(f"    Significance: {gls_result['significance']:.2f} sigma")
            print(f"    Delta-chi2 = {gls_result['delta_chi2']:.3f}, p = {gls_result['p_value']:.4f}")

            # Step 4b: Permutation test
            print(f"  --- Permutation test (10,000 shuffles) ---", flush=True)
            p_perm, gamma_perm = permutation_test(delta_mu, d_signed, mu_err, n_perm=10000)
            print(f"    p_perm = {p_perm:.4f}")

            # Step 4c: Bootstrap
            print(f"  --- Bootstrap (5,000 resamples) ---", flush=True)
            gamma_boot = bootstrap_test(delta_mu, d_signed, mu_err, n_boot=5000)
            boot_ci = np.percentile(gamma_boot, [2.5, 97.5])
            print(f"    gamma_TF bootstrap: {np.mean(gamma_boot):.4f} +/- {np.std(gamma_boot):.4f}")
            print(f"    95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")

            # Step 5: Void vs wall TF comparison
            print(f"\n  --- Void vs Wall TF relation ---")
            vw_comp = void_wall_tf_comparison(galaxies, in_void, band)
            for env in ['void', 'wall']:
                r = vw_comp.get(env, {})
                if 'zero_point' in r:
                    print(f"    {env}: a={r['zero_point']:.3f}+/-{r['zero_point_err']:.3f}, "
                          f"b={r['slope']:.3f}+/-{r['slope_err']:.3f}, "
                          f"sigma={r['scatter']:.3f} (n={r['n']})")
            if 'zero_point_offset' in vw_comp:
                print(f"    Zero-point offset (void-wall): {vw_comp['zero_point_offset']:.4f} "
                      f"+/- {vw_comp['zero_point_offset_err']:.4f} "
                      f"({vw_comp['zero_point_offset_sig']:.2f} sigma)")

            # Plot
            plot_tf_relation(galaxies, in_void, delta_mu, d_signed, band, cat_name, OUTPUT_DIR)

            cat_results['bands'][band] = {
                'tf_fit': {'a': float(a), 'b': float(b), 'sigma': float(sigma_tf)},
                'gls': gls_result,
                'p_perm': float(p_perm),
                'bootstrap_mean': float(np.mean(gamma_boot)),
                'bootstrap_std': float(np.std(gamma_boot)),
                'bootstrap_ci95': [float(boot_ci[0]), float(boot_ci[1])],
                'void_wall_comparison': vw_comp,
            }

        # Step 6: NGC/SGC systematics
        print(f"\n  --- Systematics: NGC/SGC split (K-band) ---")
        _, _, delta_mu_k, sigma_k = fit_tully_fisher(galaxies, 'k')
        mu_err_k = np.full(len(delta_mu_k), sigma_k)
        dec_arr = np.array([g['dec'] for g in galaxies])
        for label, mask in [('NGC', dec_arr > 0), ('SGC', dec_arr <= 0)]:
            if mask.sum() < 30:
                continue
            r = gls_tf_env(delta_mu_k[mask], d_signed[mask], mu_err_k[mask])
            print(f"    {label}: gamma_TF = {r['gamma_tf']:.4f} +/- {r['gamma_tf_err']:.4f} "
                  f"({r['significance']:.2f} sigma, n={r['n']})")
            cat_results[f'ngc_sgc_{label}'] = r

        all_results['catalogues'][cat_name] = cat_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Task 3B 2MTF Tully-Fisher Environment Residuals")
    print("=" * 70)
    for cat_name, cr in all_results['catalogues'].items():
        print(f"  {cat_name}:")
        for band, br in cr.get('bands', {}).items():
            g = br['gls']
            print(f"    {band.upper()}-band: gamma_TF = {g['gamma_tf']:.4f} +/- {g['gamma_tf_err']:.4f} "
                  f"({g['significance']:.2f} sigma, dchi2={g['delta_chi2']:.3f})")
            vw = br['void_wall_comparison']
            if 'zero_point_offset' in vw:
                print(f"      Zero-point offset: {vw['zero_point_offset']:.4f} "
                      f"({vw['zero_point_offset_sig']:.2f} sigma)")

    # Comparison plot
    plot_comparison(all_results, OUTPUT_DIR)
    save_results(all_results, 'task3b_2mtf_results.json', OUTPUT_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
