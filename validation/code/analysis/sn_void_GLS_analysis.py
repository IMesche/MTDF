#!/usr/bin/env python3
"""
MTDF Part I: Rigorous SN × Void Environment Analysis with GLS

Following quality-first framework:
1. Full Pantheon+ STAT+SYS covariance (GLS, not OLS)
2. Environment model: μ_i = μ_base(z_i;θ) + γ_env × E_i + γ_M × step(log M*)
3. Two environment definitions: binary and signed distance
4. Controls: redshift structure, host mass step, PV cut
5. Robustness: NGC/SGC split, multiple z cuts, permutation test, multiple void finders

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy import linalg, optimize, stats
import warnings
import os
from pathlib import Path

# Cosmology matching DESIVAST (Ω_m = 0.315, h = 1 for Mpc/h coordinates)
COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)  # For void coords in Mpc/h
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)  # For SN distance modulus


class PantheonData:
    """Load and manage Pantheon+ data with full covariance."""

    def __init__(self, data_path, cov_path):
        self.data = np.genfromtxt(data_path, names=True, dtype=None, encoding='utf-8')

        # Load full STAT+SYS covariance matrix
        # Format: first line is dimension, then n^2 values
        n = len(self.data)
        print(f"  Loading {n}×{n} covariance matrix...")
        cov_flat = np.loadtxt(cov_path, skiprows=1)
        self.cov_full = cov_flat.reshape(n, n)
        print(f"  Covariance loaded.")

        # Extract key columns
        self.z = self.data['zCMB']
        self.mu_obs = self.data['MU_SH0ES']
        self.m_b_corr = self.data['m_b_corr']
        self.host_mass = self.data['HOST_LOGMASS']
        self.ra = self.data['RA']
        self.dec = self.data['DEC']

        # For non-calibrators (MU_SH0ES = -9), compute from m_b_corr
        self.mu = np.where(self.mu_obs > 0, self.mu_obs,
                          self.m_b_corr + 19.25)  # Fiducial M_B

        print(f"Loaded {n} SNe with {n}×{n} covariance matrix")

    def apply_cuts(self, z_min=None, z_max=None, z_pv_cut=0.02,
                   ra_range=None, dec_range=None, indices=None):
        """Return subset of data with specified cuts."""
        mask = np.ones(len(self.z), dtype=bool)

        if z_min is not None:
            mask &= (self.z >= z_min)
        if z_max is not None:
            mask &= (self.z <= z_max)
        if z_pv_cut is not None:
            mask &= (self.z >= z_pv_cut)  # Cut peculiar velocity dominated region
        if ra_range is not None:
            mask &= (self.ra >= ra_range[0]) & (self.ra <= ra_range[1])
        if dec_range is not None:
            mask &= (self.dec >= dec_range[0]) & (self.dec <= dec_range[1])
        if indices is not None:
            idx_mask = np.zeros(len(self.z), dtype=bool)
            idx_mask[indices] = True
            mask &= idx_mask

        idx = np.where(mask)[0]
        return idx, self.cov_full[np.ix_(idx, idx)]


class VoidCatalog:
    """Load and manage void catalogs."""

    def __init__(self, fits_path, catalog_type='voidfinder'):
        self.path = fits_path
        self.catalog_type = catalog_type

        with fits.open(fits_path) as hdu:
            # Different extensions for different catalogs
            if catalog_type == 'voidfinder':
                self.data = hdu['MAXIMALS'].data
                self.x = self.data['X']
                self.y = self.data['Y']
                self.z = self.data['Z']
                self.r = self.data['R_EFF']
                self.edge = self.data['EDGE']
            else:  # V2/REVOLVER or V2/VIDE
                self.data = hdu['VOIDS'].data
                self.x = self.data['X']
                self.y = self.data['Y']
                self.z = self.data['Z']
                self.r = self.data['RADIUS']
                self.edge = np.zeros(len(self.x))  # V2 doesn't have EDGE

            # Get cosmology from header
            self.omega_m = hdu[0].header.get('OMEGAM', 0.315)

        print(f"Loaded {len(self.x)} voids from {os.path.basename(fits_path)}")

    def filter_interior(self):
        """Return only interior voids (EDGE=0)."""
        if self.catalog_type == 'voidfinder':
            mask = self.edge == 0
            return self.x[mask], self.y[mask], self.z[mask], self.r[mask]
        return self.x, self.y, self.z, self.r


def sn_to_comoving(z, ra, dec, cosmo=COSMO_VOIDS):
    """Convert SN positions to comoving Cartesian coordinates in Mpc/h."""
    # Comoving distance in Mpc/h (h=1 for DESIVAST convention)
    d_c = cosmo.comoving_distance(z).value

    # Convert to Cartesian
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d_c * np.sin(dec_rad)

    return x, y, z_cart


def compute_environment(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r):
    """
    Compute environment metrics for each SN.

    Returns:
        in_void: Boolean array (True if SN is inside any void)
        d_signed: Signed distance to nearest void boundary: (r - R_void) / R_void
                  Negative = inside void, Positive = outside void
        d_norm: Normalized distance to void center: r / R_void
    """
    n_sn = len(sn_x)

    in_void = np.zeros(n_sn, dtype=bool)
    d_signed = np.full(n_sn, np.inf)
    d_norm = np.full(n_sn, np.inf)

    for i in range(n_sn):
        # Distance to all void centers
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Normalized distance and signed distance
        d_normalized = dist / void_r
        d_sign = (dist - void_r) / void_r  # Negative inside, positive outside

        # Find nearest void (by normalized distance)
        idx_nearest = np.argmin(d_normalized)
        d_norm[i] = d_normalized[idx_nearest]
        d_signed[i] = d_sign[idx_nearest]

        # Inside any void?
        in_void[i] = np.any(dist < void_r)

    return in_void, d_signed, d_norm


def gls_fit(y, X, cov):
    """
    Generalized Least Squares fit.

    Args:
        y: Observations (n,)
        X: Design matrix (n, p)
        cov: Covariance matrix (n, n)

    Returns:
        beta: GLS parameter estimates
        beta_cov: Parameter covariance
        chi2: Chi-squared statistic
        dof: Degrees of freedom
    """
    # Cholesky decomposition of precision matrix
    cov_inv = linalg.inv(cov)

    # GLS: β = (X'C⁻¹X)⁻¹ X'C⁻¹y
    XtCinv = X.T @ cov_inv
    XtCinvX = XtCinv @ X
    beta_cov = linalg.inv(XtCinvX)
    beta = beta_cov @ (XtCinv @ y)

    # Chi-squared
    residual = y - X @ beta
    chi2 = residual @ cov_inv @ residual
    dof = len(y) - X.shape[1]

    return beta, beta_cov, chi2, dof


def fit_environment_model(mu, z, env_metric, host_mass, cov,
                          include_env=True, include_mass_step=True,
                          include_z_baseline=False, z_poly_order=2):
    """
    Fit the environment model:
    μ_i = μ_base(z_i) + γ_env × E_i + γ_M × step(log M*) [+ z-polynomial]

    Using a simple cosmographic expansion for μ_base:
    μ_base ≈ 5 log10(d_L/10pc) = 5 log10(c*z/H0) + 25 + higher order

    Args:
        include_z_baseline: If True, add polynomial in log(z) to allow
                           flexible baseline shape (addresses spurious trends from fixed cosmology)
        z_poly_order: Order of polynomial in log(z) for baseline flexibility
    """
    n = len(mu)

    # Theoretical distance modulus (baseline)
    mu_theory = COSMO_SN.distmod(z).value

    # Residuals from baseline cosmology
    residual = mu - mu_theory

    # Build design matrix
    # Column 0: intercept (absorbs any overall offset)
    cols = [np.ones(n)]
    param_names = ['intercept']

    # Optional: smooth z-baseline terms (polynomial in log z)
    # This addresses the "model dependence" vulnerability
    if include_z_baseline:
        log_z = np.log10(z)
        log_z_centered = log_z - np.mean(log_z)  # Center for numerical stability
        for order in range(1, z_poly_order + 1):
            cols.append(log_z_centered ** order)
            param_names.append(f'z_poly_{order}')

    if include_env:
        cols.append(env_metric)
        param_names.append('gamma_env')

    if include_mass_step:
        # Mass step at 10^10 M_sun
        mass_step = (host_mass >= 10).astype(float)
        cols.append(mass_step)
        param_names.append('gamma_M')

    X = np.column_stack(cols)

    # GLS fit
    beta, beta_cov, chi2, dof = gls_fit(residual, X, cov)

    # Find gamma_env index for error extraction
    gamma_env_idx = param_names.index('gamma_env') if 'gamma_env' in param_names else None

    return {
        'params': dict(zip(param_names, beta)),
        'param_cov': beta_cov,
        'param_names': param_names,
        'chi2': chi2,
        'dof': dof,
        'chi2_red': chi2 / dof if dof > 0 else np.nan,
        'beta': beta,
        'beta_err': np.sqrt(np.diag(beta_cov)),
        'gamma_env_idx': gamma_env_idx
    }


def delta_chi2_test(mu, z, env_metric, host_mass, cov, include_z_baseline=False):
    """
    Test significance of γ_env via Δχ² between nested models.

    H0: γ_env = 0 (no environment effect)
    H1: γ_env ≠ 0 (environment effect exists)

    Args:
        include_z_baseline: If True, include smooth z-polynomial in both models
                           to test stability against baseline shape changes
    """
    # Fit without environment term
    fit_null = fit_environment_model(mu, z, env_metric, host_mass, cov,
                                     include_env=False, include_mass_step=True,
                                     include_z_baseline=include_z_baseline)

    # Fit with environment term
    fit_full = fit_environment_model(mu, z, env_metric, host_mass, cov,
                                     include_env=True, include_mass_step=True,
                                     include_z_baseline=include_z_baseline)

    # Δχ² and p-value (1 dof for single parameter)
    delta_chi2 = fit_null['chi2'] - fit_full['chi2']
    p_value = 1 - stats.chi2.cdf(delta_chi2, 1)

    # Get gamma_env and its error from the correct index
    gamma_env_idx = fit_full['gamma_env_idx']
    gamma_env = fit_full['params'].get('gamma_env', np.nan)
    gamma_env_err = fit_full['beta_err'][gamma_env_idx] if gamma_env_idx is not None else np.nan

    return {
        'delta_chi2': delta_chi2,
        'p_value': p_value,
        'gamma_env': gamma_env,
        'gamma_env_err': gamma_env_err,
        'fit_null': fit_null,
        'fit_full': fit_full
    }


def permutation_test(mu, z, env_metric, host_mass, cov, n_perms=1000, seed=42):
    """
    Non-parametric permutation test for environment correlation.
    Shuffle environment labels within redshift bins to preserve z structure.
    """
    np.random.seed(seed)

    # Observed Δχ²
    obs_result = delta_chi2_test(mu, z, env_metric, host_mass, cov)
    obs_delta_chi2 = obs_result['delta_chi2']

    # Define redshift bins for stratified shuffling
    z_bins = np.digitize(z, [0.02, 0.05, 0.08, 0.12, 0.157])

    null_delta_chi2 = []
    for _ in range(n_perms):
        # Shuffle environment within each z bin
        env_shuffled = env_metric.copy()
        for b in np.unique(z_bins):
            idx = np.where(z_bins == b)[0]
            env_shuffled[idx] = np.random.permutation(env_shuffled[idx])

        result = delta_chi2_test(mu, z, env_shuffled, host_mass, cov)
        null_delta_chi2.append(result['delta_chi2'])

    null_delta_chi2 = np.array(null_delta_chi2)
    p_perm = np.mean(null_delta_chi2 >= obs_delta_chi2)

    return {
        'obs_delta_chi2': obs_delta_chi2,
        'null_delta_chi2': null_delta_chi2,
        'p_permutation': p_perm,
        'significance': f"{(1-p_perm)*100:.1f}%"
    }


def run_full_analysis(pantheon_path, cov_path, void_paths, output_dir=None):
    """
    Run complete analysis with all robustness checks.

    Args:
        pantheon_path: Path to Pantheon+SH0ES.dat
        cov_path: Path to Pantheon+SH0ES_STAT+SYS.cov
        void_paths: Dict with keys 'voidfinder_ngc', 'voidfinder_sgc', etc.
        output_dir: Directory for output files
    """
    print("="*70)
    print("MTDF Part I: Rigorous SN × Void Environment Analysis")
    print("="*70)
    print()

    # Load Pantheon+ data
    print("Loading Pantheon+ with full covariance...")
    pantheon = PantheonData(pantheon_path, cov_path)

    # Load void catalogs
    print("\nLoading void catalogs...")
    voids = {}
    for key, path in void_paths.items():
        if os.path.exists(path):
            cat_type = 'voidfinder' if 'VoidFinder' in path else 'v2'
            voids[key] = VoidCatalog(path, cat_type)

    # Analysis configurations
    configs = [
        {'name': 'VoidFinder (z<0.157, z>0.02)', 'z_max': 0.157, 'z_pv': 0.02,
         'void_key': 'voidfinder'},
        {'name': 'VoidFinder NGC only', 'z_max': 0.157, 'z_pv': 0.02,
         'void_key': 'voidfinder_ngc', 'ra_range': (90, 280)},
        {'name': 'VoidFinder SGC only', 'z_max': 0.157, 'z_pv': 0.02,
         'void_key': 'voidfinder_sgc', 'ra_range': (280, 360)},  # Wrap around
    ]

    results = {}

    # Combine NGC + SGC VoidFinder
    if 'voidfinder_ngc' in voids and 'voidfinder_sgc' in voids:
        vf_ngc = voids['voidfinder_ngc']
        vf_sgc = voids['voidfinder_sgc']

        ngc_x, ngc_y, ngc_z, ngc_r = vf_ngc.filter_interior()
        sgc_x, sgc_y, sgc_z, sgc_r = vf_sgc.filter_interior()

        combined_x = np.concatenate([ngc_x, sgc_x])
        combined_y = np.concatenate([ngc_y, sgc_y])
        combined_z = np.concatenate([ngc_z, sgc_z])
        combined_r = np.concatenate([ngc_r, sgc_r])

        print(f"\nCombined VoidFinder: {len(combined_x)} interior voids")

        # Main analysis with full sample
        print("\n" + "="*70)
        print("MAIN ANALYSIS: VoidFinder NGC+SGC, z ∈ [0.02, 0.157]")
        print("="*70)

        # Apply cuts
        idx, cov_sub = pantheon.apply_cuts(z_max=0.157, z_pv_cut=0.02)
        print(f"SNe after cuts: {len(idx)}")

        # Compute comoving positions for SNe
        sn_x, sn_y, sn_z = sn_to_comoving(
            pantheon.z[idx], pantheon.ra[idx], pantheon.dec[idx]
        )

        # Compute environment metrics
        in_void, d_signed, d_norm = compute_environment(
            sn_x, sn_y, sn_z, combined_x, combined_y, combined_z, combined_r
        )

        print(f"SNe inside voids: {np.sum(in_void)}")
        print(f"d_signed range: [{d_signed.min():.2f}, {d_signed.max():.2f}]")

        # Test with signed distance (continuous)
        print("\n--- Test 1: Signed distance metric ---")
        result_signed = delta_chi2_test(
            pantheon.mu[idx], pantheon.z[idx], d_signed,
            pantheon.host_mass[idx], cov_sub
        )
        print(f"γ_env (signed dist) = {result_signed['gamma_env']:.4f} ± {result_signed['gamma_env_err']:.4f}")
        print(f"Δχ² = {result_signed['delta_chi2']:.3f}")
        print(f"p-value = {result_signed['p_value']:.4f}")

        # Test with binary in/out
        print("\n--- Test 2: Binary void membership ---")
        binary_env = (~in_void).astype(float)  # 0=in void, 1=out of void
        result_binary = delta_chi2_test(
            pantheon.mu[idx], pantheon.z[idx], binary_env,
            pantheon.host_mass[idx], cov_sub
        )
        print(f"γ_env (binary) = {result_binary['gamma_env']:.4f} ± {result_binary['gamma_env_err']:.4f}")
        print(f"Δχ² = {result_binary['delta_chi2']:.3f}")
        print(f"p-value = {result_binary['p_value']:.4f}")

        # Permutation test
        print("\n--- Permutation test (n=500) ---")
        perm_result = permutation_test(
            pantheon.mu[idx], pantheon.z[idx], d_signed,
            pantheon.host_mass[idx], cov_sub, n_perms=500
        )
        print(f"Observed Δχ² = {perm_result['obs_delta_chi2']:.3f}")
        print(f"Permutation p-value = {perm_result['p_permutation']:.4f}")
        print(f"Significance: {perm_result['significance']}")

        results['main'] = {
            'signed_distance': result_signed,
            'binary': result_binary,
            'permutation': perm_result
        }

        # Robustness: NGC only
        print("\n" + "-"*50)
        print("ROBUSTNESS: NGC only")
        print("-"*50)

        # Filter SNe to NGC footprint (roughly RA 90-280)
        ngc_mask = (pantheon.ra[idx] > 90) & (pantheon.ra[idx] < 280)
        idx_ngc = idx[ngc_mask]
        cov_ngc = cov_sub[np.ix_(ngc_mask, ngc_mask)]

        sn_x_ngc = sn_x[ngc_mask]
        sn_y_ngc = sn_y[ngc_mask]
        sn_z_ngc = sn_z[ngc_mask]

        in_void_ngc, d_signed_ngc, _ = compute_environment(
            sn_x_ngc, sn_y_ngc, sn_z_ngc, ngc_x, ngc_y, ngc_z, ngc_r
        )

        print(f"NGC SNe: {len(idx_ngc)}, in voids: {np.sum(in_void_ngc)}")

        result_ngc = delta_chi2_test(
            pantheon.mu[idx_ngc], pantheon.z[idx_ngc], d_signed_ngc,
            pantheon.host_mass[idx_ngc], cov_ngc
        )
        print(f"γ_env = {result_ngc['gamma_env']:.4f} ± {result_ngc['gamma_env_err']:.4f}")
        print(f"Δχ² = {result_ngc['delta_chi2']:.3f}, p = {result_ngc['p_value']:.4f}")

        results['ngc'] = result_ngc

        # Robustness: SGC only
        print("\n" + "-"*50)
        print("ROBUSTNESS: SGC only")
        print("-"*50)

        sgc_mask = ~ngc_mask
        idx_sgc = idx[sgc_mask]
        cov_sgc = cov_sub[np.ix_(sgc_mask, sgc_mask)]

        sn_x_sgc = sn_x[sgc_mask]
        sn_y_sgc = sn_y[sgc_mask]
        sn_z_sgc = sn_z[sgc_mask]

        in_void_sgc, d_signed_sgc, _ = compute_environment(
            sn_x_sgc, sn_y_sgc, sn_z_sgc, sgc_x, sgc_y, sgc_z, sgc_r
        )

        print(f"SGC SNe: {len(idx_sgc)}, in voids: {np.sum(in_void_sgc)}")

        if len(idx_sgc) > 10:
            result_sgc = delta_chi2_test(
                pantheon.mu[idx_sgc], pantheon.z[idx_sgc], d_signed_sgc,
                pantheon.host_mass[idx_sgc], cov_sgc
            )
            print(f"γ_env = {result_sgc['gamma_env']:.4f} ± {result_sgc['gamma_env_err']:.4f}")
            print(f"Δχ² = {result_sgc['delta_chi2']:.3f}, p = {result_sgc['p_value']:.4f}")
            results['sgc'] = result_sgc
        else:
            print("Insufficient SNe for robust analysis")

        # Robustness: Without host mass step
        print("\n" + "-"*50)
        print("ROBUSTNESS: Without host-mass step")
        print("-"*50)

        fit_no_mass = fit_environment_model(
            pantheon.mu[idx], pantheon.z[idx], d_signed,
            pantheon.host_mass[idx], cov_sub,
            include_env=True, include_mass_step=False
        )
        print(f"γ_env (no mass step) = {fit_no_mass['params']['gamma_env']:.4f}")
        print(f"χ²_red = {fit_no_mass['chi2_red']:.4f}")
        results['no_mass_step'] = fit_no_mass

        # Robustness: Extended z range (z < 0.24)
        print("\n" + "-"*50)
        print("ROBUSTNESS: Extended z < 0.24")
        print("-"*50)

        idx_ext, cov_ext = pantheon.apply_cuts(z_max=0.24, z_pv_cut=0.02)
        print(f"SNe with z < 0.24: {len(idx_ext)}")

        sn_x_ext, sn_y_ext, sn_z_ext = sn_to_comoving(
            pantheon.z[idx_ext], pantheon.ra[idx_ext], pantheon.dec[idx_ext]
        )

        _, d_signed_ext, _ = compute_environment(
            sn_x_ext, sn_y_ext, sn_z_ext, combined_x, combined_y, combined_z, combined_r
        )

        result_ext = delta_chi2_test(
            pantheon.mu[idx_ext], pantheon.z[idx_ext], d_signed_ext,
            pantheon.host_mass[idx_ext], cov_ext
        )
        print(f"γ_env = {result_ext['gamma_env']:.4f} ± {result_ext['gamma_env_err']:.4f}")
        print(f"Δχ² = {result_ext['delta_chi2']:.3f}, p = {result_ext['p_value']:.4f}")
        results['extended_z'] = result_ext

        # CRITICAL ROBUSTNESS: z-baseline stability test
        print("\n" + "-"*50)
        print("ROBUSTNESS: With flexible z-baseline (polynomial in log z)")
        print("-"*50)
        print("(Tests if γ_env is stable when baseline shape is allowed to vary)")

        result_zbase = delta_chi2_test(
            pantheon.mu[idx], pantheon.z[idx], d_signed,
            pantheon.host_mass[idx], cov_sub, include_z_baseline=True
        )
        print(f"γ_env (with z-baseline) = {result_zbase['gamma_env']:.4f} ± {result_zbase['gamma_env_err']:.4f}")
        print(f"Δχ² = {result_zbase['delta_chi2']:.3f}, p = {result_zbase['p_value']:.4f}")

        # Compare to original
        gamma_shift = result_zbase['gamma_env'] - results['main']['signed_distance']['gamma_env']
        print(f"Shift from fixed baseline: Δγ_env = {gamma_shift:+.4f}")
        if abs(gamma_shift) < 0.5 * results['main']['signed_distance']['gamma_env_err']:
            print("→ γ_env is STABLE under baseline shape change")
        else:
            print("→ WARNING: γ_env shows sensitivity to baseline shape")
        results['z_baseline'] = result_zbase

    # Test with V2 void catalogs for method independence
    for void_type, ngc_key, sgc_key in [('REVOLVER', 'revolver_ngc', 'revolver_sgc'),
                                         ('VIDE', 'vide_ngc', 'vide_sgc')]:
        if ngc_key in voids and sgc_key in voids:
            print(f"\n{'='*70}")
            print(f"ROBUSTNESS: {void_type} void catalog")
            print("="*70)

            v2_ngc = voids[ngc_key]
            v2_sgc = voids[sgc_key]

            # V2 catalogs use RADIUS instead of R_EFF
            v2_x = np.concatenate([v2_ngc.x, v2_sgc.x])
            v2_y = np.concatenate([v2_ngc.y, v2_sgc.y])
            v2_z = np.concatenate([v2_ngc.z, v2_sgc.z])
            v2_r = np.concatenate([v2_ngc.r, v2_sgc.r])

            print(f"Combined {void_type}: {len(v2_x)} voids")

            # Use same SN sample as main analysis
            _, d_signed_v2, _ = compute_environment(sn_x, sn_y, sn_z, v2_x, v2_y, v2_z, v2_r)

            result_v2 = delta_chi2_test(
                pantheon.mu[idx], pantheon.z[idx], d_signed_v2,
                pantheon.host_mass[idx], cov_sub
            )
            print(f"γ_env = {result_v2['gamma_env']:.4f} ± {result_v2['gamma_env_err']:.4f}")
            print(f"Δχ² = {result_v2['delta_chi2']:.3f}, p = {result_v2['p_value']:.4f}")
            results[void_type.lower()] = result_v2

            # V2 NGC/SGC split
            print(f"\n  --- {void_type} NGC/SGC split ---")

            # NGC split for this V2 catalog
            _, d_signed_v2_ngc, _ = compute_environment(
                sn_x_ngc, sn_y_ngc, sn_z_ngc, v2_ngc.x, v2_ngc.y, v2_ngc.z, v2_ngc.r
            )
            result_v2_ngc = delta_chi2_test(
                pantheon.mu[idx_ngc], pantheon.z[idx_ngc], d_signed_v2_ngc,
                pantheon.host_mass[idx_ngc], cov_ngc
            )
            print(f"  NGC: γ_env = {result_v2_ngc['gamma_env']:.4f} ± {result_v2_ngc['gamma_env_err']:.4f}, p = {result_v2_ngc['p_value']:.4f}")
            results[f'{void_type.lower()}_ngc'] = result_v2_ngc

            # SGC split for this V2 catalog
            _, d_signed_v2_sgc, _ = compute_environment(
                sn_x_sgc, sn_y_sgc, sn_z_sgc, v2_sgc.x, v2_sgc.y, v2_sgc.z, v2_sgc.r
            )
            if len(idx_sgc) > 10:
                result_v2_sgc = delta_chi2_test(
                    pantheon.mu[idx_sgc], pantheon.z[idx_sgc], d_signed_v2_sgc,
                    pantheon.host_mass[idx_sgc], cov_sgc
                )
                print(f"  SGC: γ_env = {result_v2_sgc['gamma_env']:.4f} ± {result_v2_sgc['gamma_env_err']:.4f}, p = {result_v2_sgc['p_value']:.4f}")
                results[f'{void_type.lower()}_sgc'] = result_v2_sgc

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
The environment model is:
  μ_i = μ_ΛCDM(z_i) + intercept + γ_env × d_signed + γ_M × step(M*)

where d_signed = (r_SN - R_void) / R_void is the signed distance to the
nearest void boundary (negative inside, positive outside).

Key findings:
""")

    if 'main' in results:
        r = results['main']['signed_distance']
        print(f"  γ_env = {r['gamma_env']:.4f} ± {r['gamma_env_err']:.4f} mag")
        print(f"  Δχ² = {r['delta_chi2']:.2f} (p = {r['p_value']:.4f})")

        if r['gamma_env'] > 0:
            print("  → SNe OUTSIDE voids appear FAINTER (positive residual)")
            print("  → Equivalent to: SNe in voids appear BRIGHTER than expected")
        else:
            print("  → SNe OUTSIDE voids appear BRIGHTER (negative residual)")

        perm = results['main']['permutation']
        print(f"\n  Permutation test: p = {perm['p_permutation']:.4f}")
        print(f"  → Signal is {'SIGNIFICANT' if perm['p_permutation'] < 0.05 else 'NOT significant'} at 95% CL")

    return results


if __name__ == '__main__':
    base = str(Path(__file__).parent.parent.parent / 'data' / 'External')

    void_paths = {
        'voidfinder_ngc': os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits'),
        'voidfinder_sgc': os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits'),
        'revolver_ngc': os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits'),
        'revolver_sgc': os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits'),
        'vide_ngc': os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits'),
        'vide_sgc': os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits'),
    }

    results = run_full_analysis(
        pantheon_path=os.path.join(base, 'pantheonplus/Pantheon+SH0ES.dat'),
        cov_path=os.path.join(base, 'pantheonplus/Pantheon+SH0ES_STAT+SYS.cov'),
        void_paths=void_paths
    )
