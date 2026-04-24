#!/usr/bin/env python3
"""
SN × Void Cross-Match Pipeline for MTDF Part I Validation

Cross-matches Pantheon+ Type Ia supernovae with DESIVAST void catalog
to test for environment-dependent distance residuals.

MTDF Hypothesis (from Companion Part I, Section 2.3):
  If the stress field carries a finite coherence scale and relaxes on
  cosmological timescales, then large underdensities can sustain a
  different stress state than high density regions.

Test: Do SN residuals correlate with void environment after controlling
for host properties?

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
import warnings

# Cosmology for distance calculations
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def load_pantheon(filepath):
    """Load Pantheon+ SN catalog."""
    data = np.genfromtxt(filepath, names=True, dtype=None, encoding='utf-8')
    return data


def load_voidfinder(ngc_path, sgc_path=None):
    """Load VoidFinder catalogs (NGC + optional SGC)."""
    voids = []

    with fits.open(ngc_path) as hdu:
        ngc = hdu['MAXIMALS'].data
        voids.append(ngc)

    if sgc_path:
        with fits.open(sgc_path) as hdu:
            sgc = hdu['MAXIMALS'].data
            voids.append(sgc)

    # Stack NGC + SGC
    if len(voids) > 1:
        return np.concatenate(voids)
    return voids[0]


def sn_to_comoving(sn_data):
    """Convert SN positions to comoving Cartesian coordinates."""
    z = sn_data['zCMB']
    ra = sn_data['RA']
    dec = sn_data['DEC']

    # Comoving distance in Mpc/h (h=0.7)
    d_c = COSMO.comoving_distance(z).value * 0.7  # Mpc → Mpc/h

    # Convert to Cartesian
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    x = d_c * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = d_c * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z_cart = d_c * np.sin(np.radians(dec))

    return x, y, z_cart, d_c


def void_membership(sn_x, sn_y, sn_z, sn_d, voids, margin=1.0):
    """
    Compute void membership metrics for each SN.

    Returns:
        in_void: Boolean array (True if SN center is inside any void)
        d_norm: Normalized distance to nearest void center (d/R_eff)
        nearest_void_idx: Index of nearest void
        void_r_eff: Effective radius of nearest void
    """
    n_sn = len(sn_x)
    n_void = len(voids)

    # Extract void positions and radii
    void_x = voids['X']
    void_y = voids['Y']
    void_z = voids['Z']
    void_r = voids['R_EFF']
    void_edge = voids['EDGE']

    # Initialize output arrays
    in_void = np.zeros(n_sn, dtype=bool)
    d_norm = np.full(n_sn, np.inf)
    nearest_void_idx = np.zeros(n_sn, dtype=int)
    void_r_eff = np.zeros(n_sn)

    # For each SN, find nearest void and check membership
    for i in range(n_sn):
        # Distance to all void centers
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Normalized distance (d / R_eff)
        d_normalized = dist / void_r

        # Find nearest void
        idx_nearest = np.argmin(d_normalized)
        d_norm[i] = d_normalized[idx_nearest]
        nearest_void_idx[i] = idx_nearest
        void_r_eff[i] = void_r[idx_nearest]

        # Check if inside void (d < R_eff * margin)
        in_void[i] = d_norm[i] < margin

    return in_void, d_norm, nearest_void_idx, void_r_eff


def compute_residuals(sn_data, model='LCDM'):
    """
    Compute Hubble residuals from Pantheon+ data.

    The m_b_corr column already has corrections applied.
    Residual = m_b_corr - mu_model(z)

    For simplicity, use MU_SH0ES where available, else compute from z.
    """
    z = sn_data['zCMB']
    m_b = sn_data['m_b_corr']

    # Theoretical distance modulus for flat ΛCDM
    mu_theory = COSMO.distmod(z).value

    # Use MU_SH0ES where available (calibrated distances)
    mu_obs = sn_data['MU_SH0ES']

    # Some entries have -9 for non-calibrators - use m_b_corr + M_B instead
    # For now, compute residual from corrected magnitude
    # Residual relative to a fiducial cosmology
    residual = mu_obs - mu_theory

    # For non-calibrators (MU_SH0ES = -9), use m_b_corr with fiducial M_B
    mask_nocalib = (mu_obs < 0)
    M_B = -19.25  # Approximate fiducial absolute magnitude
    residual[mask_nocalib] = m_b[mask_nocalib] - mu_theory[mask_nocalib] - M_B

    return residual


def run_crossmatch(pantheon_path, void_ngc_path, void_sgc_path=None, z_max=0.157):
    """
    Main cross-match routine.

    Args:
        pantheon_path: Path to Pantheon+SH0ES.dat
        void_ngc_path: Path to VoidFinder NGC FITS
        void_sgc_path: Optional path to VoidFinder SGC FITS
        z_max: Maximum redshift for cross-match (DESIVAST limit)

    Returns:
        Dictionary with cross-match results
    """
    print("Loading Pantheon+ catalog...")
    sn_data = load_pantheon(pantheon_path)
    print(f"  Loaded {len(sn_data)} SNe")

    print("Loading VoidFinder catalog...")
    voids = load_voidfinder(void_ngc_path, void_sgc_path)
    print(f"  Loaded {len(voids)} void centers")

    # Filter SNe by redshift
    z = sn_data['zCMB']
    mask_z = z <= z_max
    sn_cut = sn_data[mask_z]
    print(f"  SNe with z ≤ {z_max}: {len(sn_cut)}")

    # Filter voids by EDGE flag (exclude boundary voids for cleaner signal)
    edge_mask = voids['EDGE'] == 0  # Interior voids only
    voids_interior = voids[edge_mask]
    print(f"  Interior voids (EDGE=0): {len(voids_interior)}")

    # Convert SN positions to comoving
    print("Computing comoving positions...")
    sn_x, sn_y, sn_z, sn_d = sn_to_comoving(sn_cut)

    # Compute void membership
    print("Computing void membership...")
    in_void, d_norm, void_idx, void_reff = void_membership(
        sn_x, sn_y, sn_z, sn_d, voids_interior
    )

    # Compute residuals
    print("Computing Hubble residuals...")
    residuals = compute_residuals(sn_cut)

    # Basic statistics
    n_in_void = np.sum(in_void)
    n_out_void = len(in_void) - n_in_void
    print(f"\nVoid membership:")
    print(f"  Inside voids: {n_in_void}")
    print(f"  Outside voids: {n_out_void}")

    # Residual statistics
    res_in = residuals[in_void]
    res_out = residuals[~in_void]

    print(f"\nHubble residuals:")
    print(f"  Inside voids:  mean = {np.nanmean(res_in):.4f} ± {np.nanstd(res_in)/np.sqrt(n_in_void):.4f}")
    print(f"  Outside voids: mean = {np.nanmean(res_out):.4f} ± {np.nanstd(res_out)/np.sqrt(n_out_void):.4f}")

    # Welch's t-test for difference
    t_stat, p_value = stats.ttest_ind(res_in, res_out, equal_var=False, nan_policy='omit')
    print(f"  Welch t-test: t = {t_stat:.3f}, p = {p_value:.4f}")

    # Correlation with normalized distance
    valid = np.isfinite(residuals) & np.isfinite(d_norm)
    r_corr, p_corr = stats.pearsonr(d_norm[valid], residuals[valid])
    print(f"\nCorrelation (d/R_eff vs residual):")
    print(f"  Pearson r = {r_corr:.4f}, p = {p_corr:.4f}")

    # Package results
    results = {
        'sn_data': sn_cut,
        'residuals': residuals,
        'in_void': in_void,
        'd_norm': d_norm,
        'void_idx': void_idx,
        'void_reff': void_reff,
        'n_in': n_in_void,
        'n_out': n_out_void,
        'mean_res_in': np.nanmean(res_in),
        'mean_res_out': np.nanmean(res_out),
        't_stat': t_stat,
        'p_value': p_value,
        'r_corr': r_corr,
        'p_corr': p_corr,
    }

    return results


if __name__ == '__main__':
    import os

    base = str(Path(__file__).parent.parent.parent / 'data' / 'External')

    pantheon = os.path.join(base, 'pantheonplus/Pantheon+SH0ES.dat')
    void_ngc = os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits')
    void_sgc = os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits')

    print("=" * 60)
    print("MTDF Part I: SN × Void Environment Test")
    print("=" * 60)
    print()

    results = run_crossmatch(pantheon, void_ngc, void_sgc, z_max=0.157)

    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if results['p_value'] < 0.05:
        print("⚠️  Significant difference in residuals between void/non-void SNe")
        print("    This warrants further investigation with host-mass controls")
    else:
        print("✓  No significant difference detected at p < 0.05")
        print("    Consistent with null hypothesis (no environment effect)")
