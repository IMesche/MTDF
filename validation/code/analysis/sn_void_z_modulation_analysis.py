#!/usr/bin/env python3
"""
MTDF Part I: Z-Modulation Analysis
"Make Them Sweat" Robustness Checks

1. Z-matched comparison: reweight NGC/SGC to same z distribution
2. Continuous z-modulation: fit γ_env(z) to show weakening with z

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
Date: 2025-12-16
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy import linalg, stats
from scipy.spatial import cKDTree
import warnings
import os

COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)


class PantheonData:
    """Load and manage Pantheon+ data with full covariance."""

    def __init__(self, data_path, cov_path):
        self.data = np.genfromtxt(data_path, names=True, dtype=None, encoding='utf-8')
        n = len(self.data)
        print(f"  Loading {n}×{n} covariance matrix...")
        cov_flat = np.loadtxt(cov_path, skiprows=1)
        self.cov_full = cov_flat.reshape(n, n)

        self.z = self.data['zCMB']
        self.mu_obs = self.data['MU_SH0ES']
        self.m_b_corr = self.data['m_b_corr']
        self.host_mass = self.data['HOST_LOGMASS']
        self.ra = self.data['RA']
        self.dec = self.data['DEC']
        self.mu = np.where(self.mu_obs > 0, self.mu_obs, self.m_b_corr + 19.25)
        print(f"Loaded {n} SNe")


class VoidCatalog:
    """Load void catalogs."""

    def __init__(self, fits_path, catalog_type='voidfinder'):
        self.path = fits_path
        self.catalog_type = catalog_type

        with fits.open(fits_path) as hdu:
            if catalog_type == 'voidfinder':
                self.data = hdu['MAXIMALS'].data
                self.x = self.data['X']
                self.y = self.data['Y']
                self.z = self.data['Z']
                self.r = self.data['R_EFF']
                self.edge = self.data['EDGE']
            else:
                self.data = hdu['VOIDS'].data
                self.x = self.data['X']
                self.y = self.data['Y']
                self.z = self.data['Z']
                self.r = self.data['RADIUS']
                self.edge = np.zeros(len(self.x))

    def filter_interior(self):
        if self.catalog_type == 'voidfinder':
            mask = self.edge == 0
            return self.x[mask], self.y[mask], self.z[mask], self.r[mask]
        return self.x, self.y, self.z, self.r


def sn_to_comoving(z, ra, dec, cosmo=COSMO_VOIDS):
    """Convert SN positions to comoving Cartesian coordinates."""
    d_c = cosmo.comoving_distance(z).value
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d_c * np.sin(dec_rad)
    return x, y, z_cart


def compute_environment(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r):
    """Compute signed distance to nearest void boundary."""
    n_sn = len(sn_x)
    d_signed = np.full(n_sn, np.inf)
    nearest_void_idx = np.zeros(n_sn, dtype=int)

    for i in range(n_sn):
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        d_normalized = dist / void_r
        d_sign = (dist - void_r) / void_r
        idx_nearest = np.argmin(d_normalized)
        d_signed[i] = d_sign[idx_nearest]
        nearest_void_idx[i] = idx_nearest

    return d_signed, nearest_void_idx


def gls_fit(y, X, cov, weights=None):
    """
    Generalized Least Squares fit with optional weights.

    If weights are provided, they are applied as sqrt(w) scaling to y and X,
    effectively doing weighted least squares within the GLS framework.
    """
    if weights is not None:
        # Apply weights via scaling
        w_sqrt = np.sqrt(weights)
        y_w = y * w_sqrt
        X_w = X * w_sqrt[:, np.newaxis]
        cov_w = cov * np.outer(w_sqrt, w_sqrt)
    else:
        y_w, X_w, cov_w = y, X, cov

    cov_inv = linalg.inv(cov_w)
    XtCinv = X_w.T @ cov_inv
    XtCinvX = XtCinv @ X_w
    beta_cov = linalg.inv(XtCinvX)
    beta = beta_cov @ (XtCinv @ y_w)
    residual = y_w - X_w @ beta
    chi2 = residual @ cov_inv @ residual
    dof = len(y) - X.shape[1]

    return beta, beta_cov, chi2, dof


def compute_ipw_weights(z_target, z_source, n_bins=10):
    """
    Compute Inverse Probability Weights to match z distributions.

    Weights source distribution to match target distribution.
    """
    # Create histogram bins from combined range
    z_min = min(z_target.min(), z_source.min())
    z_max = max(z_target.max(), z_source.max())
    bins = np.linspace(z_min, z_max, n_bins + 1)

    # Compute densities
    hist_target, _ = np.histogram(z_target, bins=bins, density=True)
    hist_source, _ = np.histogram(z_source, bins=bins, density=True)

    # Compute weights for each source point
    weights = np.ones(len(z_source))
    bin_indices = np.digitize(z_source, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0 and hist_source[i] > 0:
            weights[mask] = hist_target[i] / hist_source[i]

    # Normalize weights to sum to sample size
    weights = weights * len(weights) / np.sum(weights)

    return weights


def delta_chi2_test(mu, z, env_metric, host_mass, cov, weights=None):
    """Test significance of environment term."""
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory

    mass_step = (host_mass >= 10).astype(float)
    X_null = np.column_stack([np.ones(n), mass_step])
    beta_null, _, chi2_null, _ = gls_fit(residual, X_null, cov, weights)

    X_full = np.column_stack([np.ones(n), env_metric, mass_step])
    beta_full, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov, weights)

    delta_chi2 = chi2_null - chi2_full
    p_value = 1 - stats.chi2.cdf(delta_chi2, 1)

    return {
        'gamma_env': beta_full[1],
        'gamma_env_err': np.sqrt(beta_cov[1, 1]),
        'delta_chi2': delta_chi2,
        'p_value': p_value
    }


def fit_z_modulated_model(mu, z, env_metric, host_mass, cov):
    """
    Fit model with z-dependent environment coefficient:

    μ = μ_theory + α + γ_env × d_signed × f(z) + γ_M × step(M*)

    where f(z) = (1 - z/z_scale) or similar to capture weakening with z.

    We test two forms:
    1. Linear: γ_env × d_signed × (1 + β_z × z)
    2. Piecewise: γ_env_low (z < 0.05) vs γ_env_high (z >= 0.05)
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    results = {}

    # =========================================================================
    # Test 1: Constant γ_env (baseline)
    # =========================================================================
    X_const = np.column_stack([np.ones(n), env_metric, mass_step])
    beta_const, beta_cov_const, chi2_const, dof_const = gls_fit(residual, X_const, cov)

    results['constant'] = {
        'gamma_env': beta_const[1],
        'gamma_env_err': np.sqrt(beta_cov_const[1, 1]),
        'chi2': chi2_const,
        'dof': dof_const
    }

    # =========================================================================
    # Test 2: Linear z-modulation: γ_env × d × (1 + β_z × (z - z_mean))
    # Equivalently: α + γ_0 × d + γ_1 × d × z + γ_M × step
    # =========================================================================
    z_centered = z - np.mean(z)
    env_z_interaction = env_metric * z_centered

    X_linear = np.column_stack([np.ones(n), env_metric, env_z_interaction, mass_step])
    beta_lin, beta_cov_lin, chi2_lin, dof_lin = gls_fit(residual, X_linear, cov)

    # Δχ² for the z-interaction term (1 additional parameter)
    delta_chi2_interaction = chi2_const - chi2_lin
    p_interaction = 1 - stats.chi2.cdf(delta_chi2_interaction, 1)

    results['linear_z'] = {
        'gamma_env_0': beta_lin[1],  # Base environment coefficient
        'gamma_env_0_err': np.sqrt(beta_cov_lin[1, 1]),
        'gamma_env_z': beta_lin[2],  # z-interaction coefficient
        'gamma_env_z_err': np.sqrt(beta_cov_lin[2, 2]),
        'chi2': chi2_lin,
        'dof': dof_lin,
        'delta_chi2_vs_constant': delta_chi2_interaction,
        'p_interaction': p_interaction
    }

    # =========================================================================
    # Test 3: Piecewise: separate γ_env for z < 0.05 and z >= 0.05
    # =========================================================================
    z_cut = 0.05
    low_z = (z < z_cut).astype(float)
    high_z = (z >= z_cut).astype(float)

    env_low = env_metric * low_z
    env_high = env_metric * high_z

    X_piece = np.column_stack([np.ones(n), env_low, env_high, mass_step])
    beta_piece, beta_cov_piece, chi2_piece, dof_piece = gls_fit(residual, X_piece, cov)

    # Δχ² vs constant model (1 additional parameter)
    delta_chi2_piece = chi2_const - chi2_piece
    p_piece = 1 - stats.chi2.cdf(delta_chi2_piece, 1)

    results['piecewise'] = {
        'gamma_env_low': beta_piece[1],
        'gamma_env_low_err': np.sqrt(beta_cov_piece[1, 1]),
        'gamma_env_high': beta_piece[2],
        'gamma_env_high_err': np.sqrt(beta_cov_piece[2, 2]),
        'z_cut': z_cut,
        'n_low': np.sum(z < z_cut),
        'n_high': np.sum(z >= z_cut),
        'chi2': chi2_piece,
        'dof': dof_piece,
        'delta_chi2_vs_constant': delta_chi2_piece,
        'p_piecewise': p_piece
    }

    # =========================================================================
    # Test 4: Finer z-bins for direct visualization
    # =========================================================================
    z_bins = [0.02, 0.04, 0.06, 0.10, 0.157]
    bin_results = []

    for i in range(len(z_bins) - 1):
        z_lo, z_hi = z_bins[i], z_bins[i+1]
        mask = (z >= z_lo) & (z < z_hi)
        n_bin = np.sum(mask)

        if n_bin > 30:
            idx = np.where(mask)[0]
            cov_bin = cov[np.ix_(idx, idx)]
            result_bin = delta_chi2_test(
                mu[mask], z[mask], env_metric[mask],
                host_mass[mask], cov_bin
            )
            bin_results.append({
                'z_range': (z_lo, z_hi),
                'z_mean': np.mean(z[mask]),
                'n': n_bin,
                'gamma_env': result_bin['gamma_env'],
                'gamma_env_err': result_bin['gamma_env_err'],
                'p_value': result_bin['p_value']
            })

    results['z_bins'] = bin_results

    return results


def run_z_modulation_analysis():
    """Run both z-matched and z-modulation analyses."""
    base = str(Path(__file__).parent.parent.parent / 'data' / 'External')

    print("="*70)
    print("MTDF PART I: Z-MODULATION ANALYSIS")
    print("'Make Them Sweat' Robustness Checks")
    print("="*70)

    # Load data
    print("\nLoading Pantheon+ with full covariance...")
    pantheon = PantheonData(
        os.path.join(base, 'pantheonplus/Pantheon+SH0ES.dat'),
        os.path.join(base, 'pantheonplus/Pantheon+SH0ES_STAT+SYS.cov')
    )

    # Load void catalogs
    print("\nLoading void catalogs...")
    void_files = {
        'revolver_ngc': ('DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits', 'v2'),
        'revolver_sgc': ('DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits', 'v2'),
        'vide_ngc': ('DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits', 'v2'),
        'vide_sgc': ('DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits', 'v2'),
    }

    voids = {}
    for key, (fname, cat_type) in void_files.items():
        path = os.path.join(base, 'desivast_voids', fname)
        if os.path.exists(path):
            voids[key] = VoidCatalog(path, cat_type)

    # Get sample
    mask = (pantheon.z >= 0.02) & (pantheon.z <= 0.157)
    idx = np.where(mask)[0]

    sn_x, sn_y, sn_z = sn_to_comoving(
        pantheon.z[idx], pantheon.ra[idx], pantheon.dec[idx]
    )

    for catalog_name, ngc_key, sgc_key in [
        ('REVOLVER', 'revolver_ngc', 'revolver_sgc'),
        ('VIDE', 'vide_ngc', 'vide_sgc')
    ]:
        if ngc_key not in voids or sgc_key not in voids:
            continue

        print(f"\n{'#'*70}")
        print(f"# {catalog_name}")
        print(f"{'#'*70}")

        # Get void positions
        vx_ngc, vy_ngc, vz_ngc, vr_ngc = voids[ngc_key].filter_interior()
        vx_sgc, vy_sgc, vz_sgc, vr_sgc = voids[sgc_key].filter_interior()
        vx_all = np.concatenate([vx_ngc, vx_sgc])
        vy_all = np.concatenate([vy_ngc, vy_sgc])
        vz_all = np.concatenate([vz_ngc, vz_sgc])
        vr_all = np.concatenate([vr_ngc, vr_sgc])

        # Compute environment
        d_signed, nearest_idx = compute_environment(
            sn_x, sn_y, sn_z, vx_all, vy_all, vz_all, vr_all
        )

        # Footprint-based split
        n_ngc_voids = len(vx_ngc)
        is_ngc = nearest_idx < n_ngc_voids

        # Get subsets
        idx_ngc_mask = is_ngc
        idx_sgc_mask = ~is_ngc

        z_full = pantheon.z[idx]
        mu_full = pantheon.mu[idx]
        mass_full = pantheon.host_mass[idx]
        cov_full = pantheon.cov_full[np.ix_(idx, idx)]

        # =====================================================================
        # CHECK 1: Z-MATCHED COMPARISON
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"CHECK 1: Z-MATCHED NGC vs SGC COMPARISON")
        print("="*70)

        z_ngc = z_full[idx_ngc_mask]
        z_sgc = z_full[idx_sgc_mask]

        print(f"\nOriginal z distributions:")
        print(f"  NGC: mean z = {np.mean(z_ngc):.4f} ± {np.std(z_ngc):.4f} (N={len(z_ngc)})")
        print(f"  SGC: mean z = {np.mean(z_sgc):.4f} ± {np.std(z_sgc):.4f} (N={len(z_sgc)})")

        # Compute IPW weights to match SGC z-distribution to NGC
        weights_sgc = compute_ipw_weights(z_ngc, z_sgc, n_bins=8)

        print(f"\nAfter IPW reweighting:")
        print(f"  NGC: effective N = {len(z_ngc):.0f}")
        print(f"  SGC: effective N = {np.sum(weights_sgc):.0f} (weights range: {weights_sgc.min():.2f} - {weights_sgc.max():.2f})")

        # Run analysis on NGC (unweighted)
        idx_ngc_full = np.where(idx_ngc_mask)[0]
        cov_ngc = cov_full[np.ix_(idx_ngc_full, idx_ngc_full)]

        d_signed_ngc, _ = compute_environment(
            sn_x[idx_ngc_mask], sn_y[idx_ngc_mask], sn_z[idx_ngc_mask],
            vx_ngc, vy_ngc, vz_ngc, vr_ngc
        )

        result_ngc = delta_chi2_test(
            mu_full[idx_ngc_mask], z_full[idx_ngc_mask], d_signed_ngc,
            mass_full[idx_ngc_mask], cov_ngc
        )

        print(f"\nNGC (unweighted):")
        print(f"  γ_env = {result_ngc['gamma_env']:+.4f} ± {result_ngc['gamma_env_err']:.4f}")
        print(f"  p = {result_ngc['p_value']:.4f}")

        # Run analysis on SGC (IPW-weighted to match NGC z-distribution)
        idx_sgc_full = np.where(idx_sgc_mask)[0]
        cov_sgc = cov_full[np.ix_(idx_sgc_full, idx_sgc_full)]

        d_signed_sgc, _ = compute_environment(
            sn_x[idx_sgc_mask], sn_y[idx_sgc_mask], sn_z[idx_sgc_mask],
            vx_sgc, vy_sgc, vz_sgc, vr_sgc
        )

        result_sgc_weighted = delta_chi2_test(
            mu_full[idx_sgc_mask], z_full[idx_sgc_mask], d_signed_sgc,
            mass_full[idx_sgc_mask], cov_sgc, weights=weights_sgc
        )

        print(f"\nSGC (IPW-weighted to match NGC z-distribution):")
        print(f"  γ_env = {result_sgc_weighted['gamma_env']:+.4f} ± {result_sgc_weighted['gamma_env_err']:.4f}")
        print(f"  p = {result_sgc_weighted['p_value']:.4f}")

        # Interpretation
        print("\n>>> INTERPRETATION:")
        if result_ngc['p_value'] < 0.05 and result_sgc_weighted['p_value'] > 0.1:
            print("    Signal is in NGC but NOT in SGC even after z-matching.")
            print("    → This suggests the effect is FOOTPRINT-SPECIFIC, not just low-z.")
        elif result_sgc_weighted['p_value'] < 0.05:
            print("    Signal APPEARS in z-matched SGC.")
            print("    → NGC/SGC difference was driven by z-distribution difference.")
        else:
            print("    SGC shows weak/no signal even after z-matching.")
            print("    → Signal may be footprint-specific or SGC has lower sensitivity.")

        # =====================================================================
        # CHECK 2: CONTINUOUS Z-MODULATION
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"CHECK 2: CONTINUOUS Z-MODULATION γ_env(z)")
        print("="*70)

        # Use combined sample with combined void catalog
        d_signed_all, _ = compute_environment(
            sn_x, sn_y, sn_z, vx_all, vy_all, vz_all, vr_all
        )

        z_mod = fit_z_modulated_model(
            mu_full, z_full, d_signed_all, mass_full, cov_full
        )

        print("\n--- Model 1: Constant γ_env ---")
        print(f"  γ_env = {z_mod['constant']['gamma_env']:+.4f} ± {z_mod['constant']['gamma_env_err']:.4f}")

        print("\n--- Model 2: Linear z-modulation ---")
        print(f"  γ_env = γ₀ + γ₁ × (z - z_mean)")
        print(f"  γ₀ = {z_mod['linear_z']['gamma_env_0']:+.4f} ± {z_mod['linear_z']['gamma_env_0_err']:.4f}")
        print(f"  γ₁ = {z_mod['linear_z']['gamma_env_z']:+.4f} ± {z_mod['linear_z']['gamma_env_z_err']:.4f}")
        print(f"  Δχ² for z-interaction = {z_mod['linear_z']['delta_chi2_vs_constant']:.3f}")
        print(f"  p-value for z-interaction = {z_mod['linear_z']['p_interaction']:.4f}")

        if z_mod['linear_z']['gamma_env_z'] < 0:
            print("  → NEGATIVE γ₁ means γ_env WEAKENS with increasing z (as predicted)")
        else:
            print("  → Positive γ₁ means γ_env strengthens with z (unexpected)")

        print("\n--- Model 3: Piecewise (z < 0.05 vs z ≥ 0.05) ---")
        pw = z_mod['piecewise']
        print(f"  γ_env(z < 0.05)  = {pw['gamma_env_low']:+.4f} ± {pw['gamma_env_low_err']:.4f} (N={pw['n_low']})")
        print(f"  γ_env(z ≥ 0.05) = {pw['gamma_env_high']:+.4f} ± {pw['gamma_env_high_err']:.4f} (N={pw['n_high']})")
        print(f"  Δχ² vs constant = {pw['delta_chi2_vs_constant']:.3f}, p = {pw['p_piecewise']:.4f}")

        # Calculate significance of difference
        gamma_diff = pw['gamma_env_low'] - pw['gamma_env_high']
        sigma_diff = np.sqrt(pw['gamma_env_low_err']**2 + pw['gamma_env_high_err']**2)
        print(f"  Δγ = γ_low - γ_high = {gamma_diff:+.4f} ± {sigma_diff:.4f} ({abs(gamma_diff/sigma_diff):.1f}σ)")

        print("\n--- Model 4: Z-Binned Analysis ---")
        print(f"  {'z range':<15} {'N':>5} {'γ_env':>12} {'± σ':>8} {'p-val':>8}")
        print(f"  {'-'*50}")
        for bin_result in z_mod['z_bins']:
            z_lo, z_hi = bin_result['z_range']
            marker = "★" if bin_result['p_value'] < 0.05 else " "
            print(f"  [{z_lo:.2f}, {z_hi:.2f})  {bin_result['n']:>5} "
                  f"{bin_result['gamma_env']:>+10.4f} ± {bin_result['gamma_env_err']:.4f} "
                  f"{bin_result['p_value']:>8.4f} {marker}")

        # Summary
        print("\n>>> SUMMARY:")
        low_z_significant = any(b['p_value'] < 0.05 for b in z_mod['z_bins'] if b['z_range'][0] < 0.05)
        high_z_null = all(b['p_value'] > 0.1 for b in z_mod['z_bins'] if b['z_range'][0] >= 0.05)

        if low_z_significant and high_z_null:
            print("    ✓ Signal is CONCENTRATED at low z (z < 0.05)")
            print("    ✓ Signal WEAKENS or disappears at higher z")
            print("    → Consistent with MTDF stress coherence scale prediction")
        elif low_z_significant:
            print("    ✓ Signal is present at low z")
            print("    ? Signal behavior at high z is unclear")
        else:
            print("    ? No clear z-dependent pattern detected")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    run_z_modulation_analysis()
