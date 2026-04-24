#!/usr/bin/env python3
"""
Shared infrastructure for SN x Void hardening tests.

Reuses the same data loading, coordinate conversion, GLS fitting,
and environment computation as the main SN x Void analysis pipeline.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy import linalg, stats
import os
import json
from datetime import datetime

# ---------- Cosmologies (matching main pipeline) ----------
COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)  # Mpc/h for DESIVAST
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)         # For SN distance modulus

# ---------- Default paths ----------
BASE_DATA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'External')
PANTHEON_PATH = os.path.join(BASE_DATA, 'pantheonplus', 'Pantheon+SH0ES.dat')
COV_PATH = os.path.join(BASE_DATA, 'pantheonplus', 'Pantheon+SH0ES_STAT+SYS.cov')
VOID_DIR = os.path.join(BASE_DATA, 'desivast_voids')

VOID_PATHS = {
    'voidfinder_ngc': os.path.join(VOID_DIR, 'DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits'),
    'voidfinder_sgc': os.path.join(VOID_DIR, 'DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits'),
    'revolver_ngc':   os.path.join(VOID_DIR, 'DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits'),
    'revolver_sgc':   os.path.join(VOID_DIR, 'DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits'),
    'vide_ngc':       os.path.join(VOID_DIR, 'DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits'),
    'vide_sgc':       os.path.join(VOID_DIR, 'DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits'),
}

# Catalogue groupings for combined NGC+SGC runs
CATALOGUE_GROUPS = {
    'VoidFinder': ('voidfinder_ngc', 'voidfinder_sgc', 'voidfinder'),
    'REVOLVER':   ('revolver_ngc',   'revolver_sgc',   'v2'),
    'VIDE':       ('vide_ngc',       'vide_sgc',       'v2'),
}


# ---------- Data loaders ----------

class PantheonData:
    """Load Pantheon+ data with full covariance, including SALT2 parameters."""

    def __init__(self, data_path=PANTHEON_PATH, cov_path=COV_PATH):
        self.data = np.genfromtxt(data_path, names=True, dtype=None, encoding='utf-8')
        n = len(self.data)

        cov_flat = np.loadtxt(cov_path, skiprows=1)
        self.cov_full = cov_flat.reshape(n, n)

        self.z = self.data['zCMB']
        self.mu_obs = self.data['MU_SH0ES']
        self.m_b_corr = self.data['m_b_corr']
        self.host_mass = self.data['HOST_LOGMASS']
        self.ra = self.data['RA']
        self.dec = self.data['DEC']
        self.x1 = self.data['x1']        # SALT2 stretch
        self.c = self.data['c']           # SALT2 colour
        self.idsurvey = self.data['IDSURVEY']

        self.mu = np.where(self.mu_obs > 0, self.mu_obs,
                           self.m_b_corr + 19.25)

        print(f"Loaded {n} SNe (with x1, c, host_mass)")

    def apply_cuts(self, z_min=None, z_max=None, z_pv_cut=0.02):
        """Return index array and sub-covariance after z cuts."""
        mask = np.ones(len(self.z), dtype=bool)
        if z_min is not None:
            mask &= (self.z >= z_min)
        if z_max is not None:
            mask &= (self.z <= z_max)
        if z_pv_cut is not None:
            mask &= (self.z >= z_pv_cut)
        idx = np.where(mask)[0]
        return idx, self.cov_full[np.ix_(idx, idx)]


class VoidCatalog:
    """Load a DESIVAST void catalogue."""

    def __init__(self, fits_path, catalog_type='voidfinder'):
        self.catalog_type = catalog_type
        with fits.open(fits_path) as hdu:
            if catalog_type == 'voidfinder':
                d = hdu['MAXIMALS'].data
                self.x, self.y, self.z = d['X'], d['Y'], d['Z']
                self.r = d['R_EFF']
                self.edge = d['EDGE']
            else:
                d = hdu['VOIDS'].data
                self.x, self.y, self.z = d['X'], d['Y'], d['Z']
                self.r = d['RADIUS']
                self.edge = np.zeros(len(self.x))
        print(f"  Loaded {len(self.x)} voids from {os.path.basename(fits_path)}")

    def filter_interior(self):
        if self.catalog_type == 'voidfinder':
            m = self.edge == 0
            return self.x[m], self.y[m], self.z[m], self.r[m]
        return self.x, self.y, self.z, self.r


def load_void_pair(ngc_key, sgc_key, cat_type):
    """Load NGC + SGC and concatenate interior voids."""
    voids = []
    for key in (ngc_key, sgc_key):
        path = VOID_PATHS[key]
        if os.path.exists(path):
            vc = VoidCatalog(path, cat_type)
            voids.append(vc.filter_interior())
    if not voids:
        return None, None, None, None
    arrays = [np.concatenate([v[i] for v in voids]) for i in range(4)]
    return arrays


# ---------- Coordinate conversion ----------

def sn_to_comoving(z, ra, dec, cosmo=COSMO_VOIDS):
    d_c = cosmo.comoving_distance(z).value
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    x = d_c * np.cos(dec_r) * np.cos(ra_r)
    y = d_c * np.cos(dec_r) * np.sin(ra_r)
    z_cart = d_c * np.sin(dec_r)
    return x, y, z_cart


# ---------- Environment metrics ----------

def compute_environment(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r):
    """
    Signed distance to nearest void boundary: (r - R_void) / R_void
    Also returns physical distance in Mpc/h and nearest void index.
    """
    n_sn = len(sn_x)
    d_signed = np.full(n_sn, np.inf)
    d_phys_mpc = np.full(n_sn, np.inf)
    nearest_idx = np.zeros(n_sn, dtype=int)
    in_void = np.zeros(n_sn, dtype=bool)

    for i in range(n_sn):
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        d_norm = dist / void_r
        d_sign = (dist - void_r) / void_r

        idx = np.argmin(d_norm)
        d_signed[i] = d_sign[idx]
        d_phys_mpc[i] = dist[idx] - void_r[idx]  # Physical Mpc/h to boundary
        nearest_idx[i] = idx
        in_void[i] = np.any(dist < void_r)

    return d_signed, d_phys_mpc, nearest_idx, in_void


def compute_rank_metric(d_signed):
    """Convert signed distance to rank-based metric (0 = deepest in void, 1 = farthest)."""
    ranks = stats.rankdata(d_signed)
    return ranks / len(ranks)


# ---------- GLS fitting ----------

def gls_fit(y, X, cov):
    """GLS fit. If cov is diagonal, uses fast weighted LS instead of full inversion."""
    diag = np.diag(cov)
    is_diagonal = np.allclose(cov, np.diag(diag), atol=1e-30)

    if is_diagonal:
        # Weighted least squares (much faster, numerically stable)
        w = 1.0 / diag
        WX = X * w[:, np.newaxis]
        XtWX = X.T @ WX
        beta_cov = linalg.inv(XtWX)
        beta = beta_cov @ (WX.T @ y)
        residual = y - X @ beta
        chi2 = float(np.sum(w * residual**2))
    else:
        cov_inv = linalg.inv(cov)
        XtCinv = X.T @ cov_inv
        beta_cov = linalg.inv(XtCinv @ X)
        beta = beta_cov @ (XtCinv @ y)
        residual = y - X @ beta
        chi2 = float(residual @ cov_inv @ residual)

    dof = len(y) - X.shape[1]
    return beta, beta_cov, chi2, dof


def delta_chi2_test(mu, z, env_metric, host_mass, cov,
                    extra_covariates=None, extra_names=None):
    """
    Nested-model Dchi2 test for gamma_env.

    Args:
        extra_covariates: list of (n,) arrays to include in both models
                          (e.g. x1, c for population control)
        extra_names: list of names for the extra covariates
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    # Build null design matrix: intercept + mass_step [+ extras]
    null_cols = [np.ones(n), mass_step]
    if extra_covariates:
        null_cols.extend(extra_covariates)
    X_null = np.column_stack(null_cols)

    # Full: null + env_metric
    full_cols = [np.ones(n), env_metric, mass_step]
    if extra_covariates:
        full_cols.extend(extra_covariates)
    X_full = np.column_stack(full_cols)

    beta_null, _, chi2_null, _ = gls_fit(residual, X_null, cov)
    beta_full, beta_cov_full, chi2_full, dof = gls_fit(residual, X_full, cov)

    dchi2 = chi2_null - chi2_full
    p = 1 - stats.chi2.cdf(dchi2, 1)

    gamma_env = beta_full[1]
    gamma_env_err = np.sqrt(beta_cov_full[1, 1])

    return {
        'delta_chi2': dchi2,
        'p_value': p,
        'gamma_env': gamma_env,
        'gamma_env_err': gamma_env_err,
    }


# ---------- Merged sample loader ----------

MERGED_PATH = os.path.join(BASE_DATA, 'merged_low_z', 'merged_sn_ia.csv')
MERGED_COV_PATH = os.path.join(BASE_DATA, 'merged_low_z', 'merged_cov_diag.npy')


class MergedData:
    """Load merged Pantheon+ / ZTF DR2 / Foundation sample with diagonal covariance."""

    def __init__(self, data_path=MERGED_PATH, cov_path=MERGED_COV_PATH):
        import pandas as pd
        df = pd.read_csv(data_path)
        self.df = df
        self.z = df['z'].values
        self.ra = df['ra'].values
        self.dec = df['dec'].values
        self.host_mass = df['host_mass'].values
        self.x1 = df['x1'].values
        self.c = df['c'].values
        self.source = df['source'].values

        # Distance modulus: mB + 19.25 (fiducial M_B)
        self.mu = df['mB'].values + 19.25
        self.mB_err = df['mB_err'].values

        # Diagonal covariance
        self.cov_full = np.load(cov_path)

        # Fill NaN host masses with 10.0 (neutral for mass step)
        nan_mask = np.isnan(self.host_mass)
        if nan_mask.any():
            self.host_mass[nan_mask] = 10.0

        n = len(df)
        print(f"Loaded merged sample: {n} SNe "
              f"(P+: {(self.source == 'Pantheon+').sum()}, "
              f"ZTF: {(self.source == 'ZTF_DR2').sum()}, "
              f"Fdn: {(self.source == 'Foundation').sum()})")

    def apply_cuts(self, z_min=None, z_max=None, z_pv_cut=0.02):
        mask = np.ones(len(self.z), dtype=bool)
        if z_min is not None:
            mask &= (self.z >= z_min)
        if z_max is not None:
            mask &= (self.z <= z_max)
        if z_pv_cut is not None:
            mask &= (self.z >= z_pv_cut)
        idx = np.where(mask)[0]
        return idx, self.cov_full[np.ix_(idx, idx)]


# ---------- Standard analysis config ----------

def standard_low_z_setup(pantheon, catalogue_name):
    """
    Return (idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr)
    for the standard z in [0.02, 0.157] low-z analysis.
    Works with both PantheonData and MergedData objects.
    """
    idx, cov_sub = pantheon.apply_cuts(z_pv_cut=0.02, z_max=0.157)
    sn_x, sn_y, sn_z = sn_to_comoving(
        pantheon.z[idx], pantheon.ra[idx], pantheon.dec[idx])

    ngc_key, sgc_key, cat_type = CATALOGUE_GROUPS[catalogue_name]
    vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)

    return idx, cov_sub, sn_x, sn_y, sn_z, vx, vy, vz, vr


# ---------- Output helper ----------

def save_results(results, filename, output_dir):
    """Save results dict as JSON with numpy conversion."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"  Saved: {path}")
    return path
