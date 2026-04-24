#!/usr/bin/env python3
"""
MTDF Part I: Pre-Registered NGC/SGC Analysis with Selection Function Checks

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

# Cosmology matching DESIVAST (Ω_m = 0.315, h = 1 for Mpc/h coordinates)
COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)  # For void coords in Mpc/h
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)  # For SN distance modulus


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

        print(f"Loaded {n} SNe with {n}×{n} covariance matrix")

    def get_subset(self, idx):
        """Return data and covariance for subset of SNe."""
        return {
            'z': self.z[idx],
            'mu': self.mu[idx],
            'host_mass': self.host_mass[idx],
            'ra': self.ra[idx],
            'dec': self.dec[idx],
            'cov': self.cov_full[np.ix_(idx, idx)]
        }


class VoidCatalog:
    """Load and manage void catalogs."""

    def __init__(self, fits_path, catalog_type='voidfinder'):
        self.path = fits_path
        self.catalog_type = catalog_type
        self.region = 'NGC' if 'NGC' in fits_path else 'SGC'

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

        print(f"  Loaded {len(self.x)} voids from {os.path.basename(fits_path)}")

    def filter_interior(self):
        """Return only interior voids (EDGE=0) for VoidFinder."""
        if self.catalog_type == 'voidfinder':
            mask = self.edge == 0
            return self.x[mask], self.y[mask], self.z[mask], self.r[mask]
        return self.x, self.y, self.z, self.r


def sn_to_comoving(z, ra, dec, cosmo=COSMO_VOIDS):
    """Convert SN positions to comoving Cartesian coordinates in Mpc/h."""
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


def compute_local_void_density(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r,
                                radius_mpc=50.0):
    """
    Compute local void density as alternate environment metric.

    This metric is the number of void centers within a fixed radius,
    weighted by inverse void radius (larger voids = more weight).

    This is INDEPENDENT of signed distance and tests whether the
    environment correlation is specific to void boundaries.
    """
    n_sn = len(sn_x)
    void_density = np.zeros(n_sn)

    # Build KD-tree for void centers
    void_coords = np.column_stack([void_x, void_y, void_z])
    tree = cKDTree(void_coords)

    for i in range(n_sn):
        sn_pos = np.array([sn_x[i], sn_y[i], sn_z[i]])
        # Find voids within radius
        idx_nearby = tree.query_ball_point(sn_pos, radius_mpc)
        if len(idx_nearby) > 0:
            # Weight by void size (larger voids = more underdense)
            weights = void_r[idx_nearby]
            void_density[i] = np.sum(weights) / radius_mpc**3

    # Normalize to zero mean for regression
    void_density = (void_density - np.mean(void_density)) / np.std(void_density)
    return void_density


def gls_fit(y, X, cov):
    """Generalized Least Squares fit."""
    cov_inv = linalg.inv(cov)
    XtCinv = X.T @ cov_inv
    XtCinvX = XtCinv @ X
    beta_cov = linalg.inv(XtCinvX)
    beta = beta_cov @ (XtCinv @ y)
    residual = y - X @ beta
    chi2 = residual @ cov_inv @ residual
    dof = len(y) - X.shape[1]
    return beta, beta_cov, chi2, dof


def delta_chi2_test(mu, z, env_metric, host_mass, cov):
    """Test significance of environment term via Δχ²."""
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory

    # Null model: intercept + mass step only
    mass_step = (host_mass >= 10).astype(float)
    X_null = np.column_stack([np.ones(n), mass_step])
    beta_null, _, chi2_null, _ = gls_fit(residual, X_null, cov)

    # Full model: intercept + environment + mass step
    X_full = np.column_stack([np.ones(n), env_metric, mass_step])
    beta_full, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov)

    delta_chi2 = chi2_null - chi2_full
    p_value = 1 - stats.chi2.cdf(delta_chi2, 1)

    gamma_env = beta_full[1]
    gamma_env_err = np.sqrt(beta_cov[1, 1])

    return {
        'gamma_env': gamma_env,
        'gamma_env_err': gamma_env_err,
        'delta_chi2': delta_chi2,
        'p_value': p_value,
        'chi2_full': chi2_full,
        'dof': dof
    }


def selection_function_check(data_ngc, data_sgc, label=''):
    """
    Compare selection function between NGC and SGC.

    Tests:
    1. Sample size comparison
    2. Redshift distribution (KS test)
    3. Host mass distribution (KS test)
    4. Sky coverage (RA/Dec ranges)
    """
    print(f"\n{'='*70}")
    print(f"SELECTION FUNCTION CHECK: {label}")
    print("="*70)

    z_ngc, z_sgc = data_ngc['z'], data_sgc['z']
    m_ngc, m_sgc = data_ngc['host_mass'], data_sgc['host_mass']

    print(f"\n1. SAMPLE SIZES:")
    print(f"   NGC: {len(z_ngc)} SNe")
    print(f"   SGC: {len(z_sgc)} SNe")
    print(f"   Ratio NGC/SGC: {len(z_ngc)/len(z_sgc):.2f}")

    print(f"\n2. REDSHIFT DISTRIBUTION:")
    print(f"   NGC: z = {np.mean(z_ngc):.4f} ± {np.std(z_ngc):.4f} (range: {z_ngc.min():.4f} - {z_ngc.max():.4f})")
    print(f"   SGC: z = {np.mean(z_sgc):.4f} ± {np.std(z_sgc):.4f} (range: {z_sgc.min():.4f} - {z_sgc.max():.4f})")
    ks_z, p_z = stats.ks_2samp(z_ngc, z_sgc)
    print(f"   KS test: D = {ks_z:.4f}, p = {p_z:.4f}")
    if p_z < 0.05:
        print(f"   ⚠️  WARNING: z distributions differ significantly (p < 0.05)")
    else:
        print(f"   ✓ z distributions are consistent (p > 0.05)")

    print(f"\n3. HOST MASS DISTRIBUTION:")
    # Filter valid host masses
    valid_ngc = m_ngc > 0
    valid_sgc = m_sgc > 0
    m_ngc_valid = m_ngc[valid_ngc]
    m_sgc_valid = m_sgc[valid_sgc]
    print(f"   NGC: log M* = {np.mean(m_ngc_valid):.2f} ± {np.std(m_ngc_valid):.2f} (N={len(m_ngc_valid)})")
    print(f"   SGC: log M* = {np.mean(m_sgc_valid):.2f} ± {np.std(m_sgc_valid):.2f} (N={len(m_sgc_valid)})")

    if len(m_ngc_valid) > 10 and len(m_sgc_valid) > 10:
        ks_m, p_m = stats.ks_2samp(m_ngc_valid, m_sgc_valid)
        print(f"   KS test: D = {ks_m:.4f}, p = {p_m:.4f}")
        if p_m < 0.05:
            print(f"   ⚠️  WARNING: Host mass distributions differ significantly")
        else:
            print(f"   ✓ Host mass distributions are consistent")

    # Fraction above mass step
    frac_high_ngc = np.mean(m_ngc_valid >= 10)
    frac_high_sgc = np.mean(m_sgc_valid >= 10)
    print(f"   Fraction log M* ≥ 10: NGC = {frac_high_ngc:.1%}, SGC = {frac_high_sgc:.1%}")

    print(f"\n4. SKY COVERAGE:")
    print(f"   NGC: RA = {data_ngc['ra'].min():.1f}° - {data_ngc['ra'].max():.1f}°")
    print(f"   SGC: RA = {data_sgc['ra'].min():.1f}° - {data_sgc['ra'].max():.1f}°")
    print(f"   NGC: Dec = {data_ngc['dec'].min():.1f}° - {data_ngc['dec'].max():.1f}°")
    print(f"   SGC: Dec = {data_sgc['dec'].min():.1f}° - {data_sgc['dec'].max():.1f}°")

    return {
        'n_ngc': len(z_ngc),
        'n_sgc': len(z_sgc),
        'ks_z': ks_z,
        'p_z': p_z,
        'ks_m': ks_m if len(m_ngc_valid) > 10 and len(m_sgc_valid) > 10 else np.nan,
        'p_m': p_m if len(m_ngc_valid) > 10 and len(m_sgc_valid) > 10 else np.nan
    }


def run_preregistered_analysis(pantheon, void_catalog_ngc, void_catalog_sgc,
                                catalog_name, z_min=0.02, z_max=0.157):
    """
    Run pre-registered analysis for a single void catalog pair.

    Treats NGC and SGC as TWO SEPARATE analyses with identical pipeline.
    """
    print(f"\n{'#'*70}")
    print(f"# PRE-REGISTERED ANALYSIS: {catalog_name}")
    print(f"# NGC and SGC treated as SEPARATE analyses with IDENTICAL pipeline")
    print(f"{'#'*70}")

    # Get full sample with cuts
    mask = (pantheon.z >= z_min) & (pantheon.z <= z_max)
    idx_all = np.where(mask)[0]

    # Compute comoving positions for all SNe
    sn_x_all, sn_y_all, sn_z_all = sn_to_comoving(
        pantheon.z[idx_all], pantheon.ra[idx_all], pantheon.dec[idx_all]
    )

    # Get void positions
    vx_ngc, vy_ngc, vz_ngc, vr_ngc = void_catalog_ngc.filter_interior()
    vx_sgc, vy_sgc, vz_sgc, vr_sgc = void_catalog_sgc.filter_interior()

    # Combine voids for footprint-based assignment
    vx_all = np.concatenate([vx_ngc, vx_sgc])
    vy_all = np.concatenate([vy_ngc, vy_sgc])
    vz_all = np.concatenate([vz_ngc, vz_sgc])
    vr_all = np.concatenate([vr_ngc, vr_sgc])

    # Footprint-based NGC/SGC assignment
    n_ngc_voids = len(vx_ngc)
    _, nearest_void_idx = compute_environment(
        sn_x_all, sn_y_all, sn_z_all, vx_all, vy_all, vz_all, vr_all
    )
    is_ngc_footprint = nearest_void_idx < n_ngc_voids

    # Split into NGC and SGC samples
    idx_ngc = idx_all[is_ngc_footprint]
    idx_sgc = idx_all[~is_ngc_footprint]

    # Get data subsets
    data_ngc = pantheon.get_subset(idx_ngc)
    data_sgc = pantheon.get_subset(idx_sgc)

    # Selection function check
    sel_check = selection_function_check(data_ngc, data_sgc, label=catalog_name)

    results = {'catalog': catalog_name, 'selection': sel_check}

    # =========================================================================
    # ANALYSIS 1: NGC (Pre-Registered)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: {catalog_name} NGC (Pre-Registered)")
    print("="*70)

    sn_x_ngc = sn_x_all[is_ngc_footprint]
    sn_y_ngc = sn_y_all[is_ngc_footprint]
    sn_z_ngc = sn_z_all[is_ngc_footprint]

    # Primary metric: signed distance to void boundary
    d_signed_ngc, _ = compute_environment(
        sn_x_ngc, sn_y_ngc, sn_z_ngc, vx_ngc, vy_ngc, vz_ngc, vr_ngc
    )

    print(f"\nSample: {len(idx_ngc)} SNe in NGC footprint")
    print(f"d_signed range: [{d_signed_ngc.min():.2f}, {d_signed_ngc.max():.2f}]")

    # Primary analysis
    result_ngc = delta_chi2_test(
        data_ngc['mu'], data_ngc['z'], d_signed_ngc,
        data_ngc['host_mass'], data_ngc['cov']
    )

    print(f"\n  PRIMARY METRIC (signed distance to void boundary):")
    print(f"  γ_env = {result_ngc['gamma_env']:+.4f} ± {result_ngc['gamma_env_err']:.4f} mag")
    print(f"  Δχ² = {result_ngc['delta_chi2']:.3f}")
    print(f"  p-value = {result_ngc['p_value']:.4f}")

    if result_ngc['p_value'] < 0.01:
        print(f"  ★★ HIGHLY SIGNIFICANT (p < 0.01)")
    elif result_ngc['p_value'] < 0.05:
        print(f"  ★ SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Not significant at p < 0.05")

    # Alternate metric: local void density
    void_density_ngc = compute_local_void_density(
        sn_x_ngc, sn_y_ngc, sn_z_ngc, vx_ngc, vy_ngc, vz_ngc, vr_ngc
    )

    result_ngc_alt = delta_chi2_test(
        data_ngc['mu'], data_ngc['z'], void_density_ngc,
        data_ngc['host_mass'], data_ngc['cov']
    )

    print(f"\n  ALTERNATE METRIC (local void density, 50 Mpc sphere):")
    print(f"  γ_env = {result_ngc_alt['gamma_env']:+.4f} ± {result_ngc_alt['gamma_env_err']:.4f}")
    print(f"  Δχ² = {result_ngc_alt['delta_chi2']:.3f}, p = {result_ngc_alt['p_value']:.4f}")

    results['ngc'] = {
        'n_sn': len(idx_ngc),
        'primary': result_ngc,
        'alternate': result_ngc_alt
    }

    # =========================================================================
    # ANALYSIS 2: SGC (Pre-Registered)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: {catalog_name} SGC (Pre-Registered)")
    print("="*70)

    sn_x_sgc = sn_x_all[~is_ngc_footprint]
    sn_y_sgc = sn_y_all[~is_ngc_footprint]
    sn_z_sgc = sn_z_all[~is_ngc_footprint]

    # Primary metric: signed distance to void boundary
    d_signed_sgc, _ = compute_environment(
        sn_x_sgc, sn_y_sgc, sn_z_sgc, vx_sgc, vy_sgc, vz_sgc, vr_sgc
    )

    print(f"\nSample: {len(idx_sgc)} SNe in SGC footprint")
    print(f"d_signed range: [{d_signed_sgc.min():.2f}, {d_signed_sgc.max():.2f}]")

    if len(idx_sgc) > 20:
        result_sgc = delta_chi2_test(
            data_sgc['mu'], data_sgc['z'], d_signed_sgc,
            data_sgc['host_mass'], data_sgc['cov']
        )

        print(f"\n  PRIMARY METRIC (signed distance to void boundary):")
        print(f"  γ_env = {result_sgc['gamma_env']:+.4f} ± {result_sgc['gamma_env_err']:.4f} mag")
        print(f"  Δχ² = {result_sgc['delta_chi2']:.3f}")
        print(f"  p-value = {result_sgc['p_value']:.4f}")

        if result_sgc['p_value'] < 0.01:
            print(f"  ★★ HIGHLY SIGNIFICANT (p < 0.01)")
        elif result_sgc['p_value'] < 0.05:
            print(f"  ★ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  Not significant at p < 0.05")

        # Alternate metric
        void_density_sgc = compute_local_void_density(
            sn_x_sgc, sn_y_sgc, sn_z_sgc, vx_sgc, vy_sgc, vz_sgc, vr_sgc
        )

        result_sgc_alt = delta_chi2_test(
            data_sgc['mu'], data_sgc['z'], void_density_sgc,
            data_sgc['host_mass'], data_sgc['cov']
        )

        print(f"\n  ALTERNATE METRIC (local void density, 50 Mpc sphere):")
        print(f"  γ_env = {result_sgc_alt['gamma_env']:+.4f} ± {result_sgc_alt['gamma_env_err']:.4f}")
        print(f"  Δχ² = {result_sgc_alt['delta_chi2']:.3f}, p = {result_sgc_alt['p_value']:.4f}")

        results['sgc'] = {
            'n_sn': len(idx_sgc),
            'primary': result_sgc,
            'alternate': result_sgc_alt
        }
    else:
        print("  Insufficient SNe for robust analysis")
        results['sgc'] = None

    return results


def main():
    base = str(Path(__file__).parent.parent.parent / 'data' / 'External')

    print("="*70)
    print("MTDF PART I: PRE-REGISTERED NGC/SGC ANALYSIS")
    print("With Selection Function Checks and Alternate Metrics")
    print("="*70)

    # Load Pantheon+ data
    print("\nLoading Pantheon+ with full covariance...")
    pantheon = PantheonData(
        os.path.join(base, 'pantheonplus/Pantheon+SH0ES.dat'),
        os.path.join(base, 'pantheonplus/Pantheon+SH0ES_STAT+SYS.cov')
    )

    # Load void catalogs
    print("\nLoading void catalogs...")
    voids = {}

    void_files = {
        'voidfinder_ngc': 'DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits',
        'voidfinder_sgc': 'DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits',
        'revolver_ngc': 'DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits',
        'revolver_sgc': 'DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits',
        'vide_ngc': 'DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits',
        'vide_sgc': 'DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits',
    }

    for key, fname in void_files.items():
        path = os.path.join(base, 'desivast_voids', fname)
        if os.path.exists(path):
            cat_type = 'voidfinder' if 'VoidFinder' in fname else 'v2'
            voids[key] = VoidCatalog(path, cat_type)

    # Run pre-registered analyses
    all_results = {}

    for catalog_name, ngc_key, sgc_key in [
        ('VoidFinder', 'voidfinder_ngc', 'voidfinder_sgc'),
        ('REVOLVER', 'revolver_ngc', 'revolver_sgc'),
        ('VIDE', 'vide_ngc', 'vide_sgc')
    ]:
        if ngc_key in voids and sgc_key in voids:
            results = run_preregistered_analysis(
                pantheon, voids[ngc_key], voids[sgc_key], catalog_name
            )
            all_results[catalog_name] = results

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: PRE-REGISTERED NGC/SGC RESULTS")
    print("="*70)

    print("""
┌────────────────────────────────────────────────────────────────────────┐
│                    PRIMARY METRIC (Signed Distance)                    │
├───────────────┬─────────────────────────┬──────────────────────────────┤
│ Catalog       │ NGC γ_env ± σ (p-val)   │ SGC γ_env ± σ (p-val)        │
├───────────────┼─────────────────────────┼──────────────────────────────┤""")

    for cat_name, res in all_results.items():
        ngc = res.get('ngc', {})
        sgc = res.get('sgc', {})

        if ngc and 'primary' in ngc:
            ngc_p = ngc['primary']
            ngc_str = f"{ngc_p['gamma_env']:+.4f} ± {ngc_p['gamma_env_err']:.4f} ({ngc_p['p_value']:.4f})"
            if ngc_p['p_value'] < 0.01:
                ngc_str += " ★★"
            elif ngc_p['p_value'] < 0.05:
                ngc_str += " ★"
        else:
            ngc_str = "N/A"

        if sgc and 'primary' in sgc:
            sgc_p = sgc['primary']
            sgc_str = f"{sgc_p['gamma_env']:+.4f} ± {sgc_p['gamma_env_err']:.4f} ({sgc_p['p_value']:.4f})"
            if sgc_p['p_value'] < 0.01:
                sgc_str += " ★★"
            elif sgc_p['p_value'] < 0.05:
                sgc_str += " ★"
        else:
            sgc_str = "N/A"

        print(f"│ {cat_name:<13} │ {ngc_str:<23} │ {sgc_str:<28} │")

    print("└───────────────┴─────────────────────────┴──────────────────────────────┘")
    print("★★ = p < 0.01, ★ = p < 0.05")

    print("""
┌────────────────────────────────────────────────────────────────────────┐
│              ALTERNATE METRIC (Local Void Density, 50 Mpc)             │
├───────────────┬─────────────────────────┬──────────────────────────────┤
│ Catalog       │ NGC γ_env ± σ (p-val)   │ SGC γ_env ± σ (p-val)        │
├───────────────┼─────────────────────────┼──────────────────────────────┤""")

    for cat_name, res in all_results.items():
        ngc = res.get('ngc', {})
        sgc = res.get('sgc', {})

        if ngc and 'alternate' in ngc:
            ngc_a = ngc['alternate']
            ngc_str = f"{ngc_a['gamma_env']:+.4f} ± {ngc_a['gamma_env_err']:.4f} ({ngc_a['p_value']:.4f})"
        else:
            ngc_str = "N/A"

        if sgc and 'alternate' in sgc:
            sgc_a = sgc['alternate']
            sgc_str = f"{sgc_a['gamma_env']:+.4f} ± {sgc_a['gamma_env_err']:.4f} ({sgc_a['p_value']:.4f})"
        else:
            sgc_str = "N/A"

        print(f"│ {cat_name:<13} │ {ngc_str:<23} │ {sgc_str:<28} │")

    print("└───────────────┴─────────────────────────┴──────────────────────────────┘")

    # Selection function summary
    print("\n" + "="*70)
    print("SELECTION FUNCTION SUMMARY")
    print("="*70)
    print("""
Tests whether NGC and SGC samples have similar selection properties.
If distributions differ significantly, the NGC/SGC comparison may be confounded.
""")

    for cat_name, res in all_results.items():
        sel = res.get('selection', {})
        print(f"\n{cat_name}:")
        print(f"  Sample sizes: NGC = {sel.get('n_ngc', 'N/A')}, SGC = {sel.get('n_sgc', 'N/A')}")
        print(f"  z distributions: KS p = {sel.get('p_z', np.nan):.4f} {'(differ!)' if sel.get('p_z', 1) < 0.05 else '(consistent)'}")
        print(f"  Mass distributions: KS p = {sel.get('p_m', np.nan):.4f} {'(differ!)' if sel.get('p_m', 1) < 0.05 else '(consistent)'}")

    return all_results


if __name__ == '__main__':
    results = main()
