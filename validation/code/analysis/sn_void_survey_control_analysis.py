#!/usr/bin/env python3
"""
MTDF Part I: Survey Fixed-Effects Analysis
Addresses "footprint-specific = survey-specific?" concern

1. Show which surveys dominate NGC vs SGC
2. Add per-survey fixed effects (intercepts)
3. Test if γ_env survives with survey fixed effects
4. Within-survey subset analysis for largest surveys

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
import os

COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)

# Pantheon+ survey IDs (from Scolnic et al. 2022)
SURVEY_NAMES = {
    1: 'CfA1',
    4: 'CfA2',
    5: 'CfA3S',
    10: 'CfA3K',
    15: 'CfA4',
    18: 'CSP',
    50: 'PS1MD',
    51: 'SNLS',
    56: 'SDSS',
    57: 'Foundation',
    61: 'CNIa0.02',
    62: 'LOSS',
    63: 'SOUSA',
    64: 'Misc_lowz',
    65: 'Misc_highz',
    66: 'DES',
    100: 'HST',
    101: 'CFA4_p2',
    106: 'PS1_LOWZ',
    150: 'CfA3'
}


class PantheonData:
    """Load Pantheon+ data with survey information."""

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
        self.survey_id = self.data['IDSURVEY']
        self.mu = np.where(self.mu_obs > 0, self.mu_obs, self.m_b_corr + 19.25)
        print(f"Loaded {n} SNe from {len(np.unique(self.survey_id))} surveys")


class VoidCatalog:
    """Load void catalogs."""

    def __init__(self, fits_path, catalog_type='voidfinder'):
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
        if hasattr(self, 'edge') and self.edge is not None:
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
    nearest_idx = np.zeros(n_sn, dtype=int)

    for i in range(n_sn):
        dx = sn_x[i] - void_x
        dy = sn_y[i] - void_y
        dz = sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        d_normalized = dist / void_r
        d_sign = (dist - void_r) / void_r
        idx = np.argmin(d_normalized)
        d_signed[i] = d_sign[idx]
        nearest_idx[i] = idx

    return d_signed, nearest_idx


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


def create_survey_dummies(survey_ids):
    """Create one-hot encoded survey dummies (drop one for identifiability)."""
    unique_surveys = np.unique(survey_ids)
    # Drop the most common survey as reference
    counts = {s: np.sum(survey_ids == s) for s in unique_surveys}
    reference_survey = max(counts, key=counts.get)

    dummies = []
    survey_cols = []
    for s in unique_surveys:
        if s != reference_survey:
            dummies.append((survey_ids == s).astype(float))
            survey_cols.append(SURVEY_NAMES.get(s, f'Survey_{s}'))

    return np.column_stack(dummies) if dummies else None, survey_cols, reference_survey


def delta_chi2_test_with_survey_fe(mu, z, env_metric, host_mass, survey_ids, cov):
    """
    Test significance of γ_env with per-survey fixed effects.

    Model: μ = μ_theory + Σ_s α_s × I(survey=s) + γ_env × d_signed + γ_M × step(M*)
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory

    mass_step = (host_mass >= 10).astype(float)

    # Create survey dummies
    survey_dummies, survey_cols, ref_survey = create_survey_dummies(survey_ids)
    n_surveys = len(survey_cols)

    # Null model: survey FE + mass step (no environment)
    if survey_dummies is not None:
        X_null = np.column_stack([np.ones(n), survey_dummies, mass_step])
    else:
        X_null = np.column_stack([np.ones(n), mass_step])

    beta_null, _, chi2_null, _ = gls_fit(residual, X_null, cov)

    # Full model: survey FE + environment + mass step
    if survey_dummies is not None:
        X_full = np.column_stack([np.ones(n), survey_dummies, env_metric, mass_step])
        gamma_env_idx = 1 + n_surveys  # Position of γ_env in beta
    else:
        X_full = np.column_stack([np.ones(n), env_metric, mass_step])
        gamma_env_idx = 1

    beta_full, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov)

    delta_chi2 = chi2_null - chi2_full
    p_value = 1 - stats.chi2.cdf(delta_chi2, 1)

    return {
        'gamma_env': beta_full[gamma_env_idx],
        'gamma_env_err': np.sqrt(beta_cov[gamma_env_idx, gamma_env_idx]),
        'delta_chi2': delta_chi2,
        'p_value': p_value,
        'n_survey_fe': n_surveys,
        'reference_survey': SURVEY_NAMES.get(ref_survey, f'Survey_{ref_survey}')
    }


def within_survey_analysis(mu, z, env_metric, host_mass, survey_ids, cov, min_n=30):
    """Run analysis within each survey separately."""
    results = {}
    unique_surveys = np.unique(survey_ids)

    for survey_id in unique_surveys:
        mask = survey_ids == survey_id
        n_survey = np.sum(mask)

        if n_survey >= min_n:
            idx = np.where(mask)[0]
            cov_sub = cov[np.ix_(idx, idx)]

            mu_sub = mu[mask]
            z_sub = z[mask]
            env_sub = env_metric[mask]
            mass_sub = host_mass[mask]

            # Simple model: intercept + environment + mass step
            mu_theory = COSMO_SN.distmod(z_sub).value
            residual = mu_sub - mu_theory
            mass_step = (mass_sub >= 10).astype(float)

            X_null = np.column_stack([np.ones(n_survey), mass_step])
            X_full = np.column_stack([np.ones(n_survey), env_sub, mass_step])

            try:
                beta_null, _, chi2_null, _ = gls_fit(residual, X_null, cov_sub)
                beta_full, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov_sub)

                delta_chi2 = chi2_null - chi2_full
                p_value = 1 - stats.chi2.cdf(delta_chi2, 1)

                survey_name = SURVEY_NAMES.get(survey_id, f'Survey_{survey_id}')
                results[survey_name] = {
                    'n': n_survey,
                    'gamma_env': beta_full[1],
                    'gamma_env_err': np.sqrt(beta_cov[1, 1]),
                    'delta_chi2': delta_chi2,
                    'p_value': p_value,
                    'z_mean': np.mean(z_sub)
                }
            except Exception as e:
                pass  # Skip surveys with singular covariance

    return results


def run_survey_control_analysis():
    """Run survey-controlled analysis."""
    base = str(Path(__file__).parent.parent.parent / 'data' / 'External')

    print("="*70)
    print("MTDF PART I: SURVEY FIXED-EFFECTS ANALYSIS")
    print("Addresses 'footprint-specific = survey-specific?' concern")
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
    }

    voids = {}
    for key, (fname, cat_type) in void_files.items():
        path = os.path.join(base, 'desivast_voids', fname)
        if os.path.exists(path):
            voids[key] = VoidCatalog(path, cat_type)

    # Get sample
    mask = (pantheon.z >= 0.02) & (pantheon.z <= 0.157)
    idx = np.where(mask)[0]

    z = pantheon.z[idx]
    mu = pantheon.mu[idx]
    host_mass = pantheon.host_mass[idx]
    survey_ids = pantheon.survey_id[idx]
    ra = pantheon.ra[idx]
    cov = pantheon.cov_full[np.ix_(idx, idx)]

    sn_x, sn_y, sn_z = sn_to_comoving(z, ra, pantheon.dec[idx])

    # Get void positions
    vx_ngc, vy_ngc, vz_ngc, vr_ngc = voids['revolver_ngc'].filter_interior()
    vx_sgc, vy_sgc, vz_sgc, vr_sgc = voids['revolver_sgc'].filter_interior()
    vx_all = np.concatenate([vx_ngc, vx_sgc])
    vy_all = np.concatenate([vy_ngc, vy_sgc])
    vz_all = np.concatenate([vz_ngc, vz_sgc])
    vr_all = np.concatenate([vr_ngc, vr_sgc])

    # Compute environment
    d_signed, nearest_idx = compute_environment(sn_x, sn_y, sn_z, vx_all, vy_all, vz_all, vr_all)

    # Footprint-based split
    n_ngc_voids = len(vx_ngc)
    is_ngc = nearest_idx < n_ngc_voids

    # =========================================================================
    # PART 1: Survey composition in NGC vs SGC
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: SURVEY COMPOSITION IN NGC vs SGC")
    print("="*70)

    print("\n  Survey distribution (low-z sample only):")
    print(f"  {'Survey':<15} {'Total':>6} {'NGC':>6} {'SGC':>6} {'% NGC':>8}")
    print(f"  {'-'*45}")

    unique_surveys = np.unique(survey_ids)
    survey_data = []

    for s in unique_surveys:
        mask_s = survey_ids == s
        n_total = np.sum(mask_s)
        n_ngc = np.sum(mask_s & is_ngc)
        n_sgc = np.sum(mask_s & ~is_ngc)
        pct_ngc = 100 * n_ngc / n_total if n_total > 0 else 0

        survey_name = SURVEY_NAMES.get(s, f'Survey_{s}')
        survey_data.append((survey_name, n_total, n_ngc, n_sgc, pct_ngc))

    # Sort by total count
    survey_data.sort(key=lambda x: -x[1])

    for name, total, ngc, sgc, pct in survey_data:
        if total >= 10:
            print(f"  {name:<15} {total:>6} {ngc:>6} {sgc:>6} {pct:>7.1f}%")

    print(f"\n  Total: {len(idx)} SNe, NGC: {np.sum(is_ngc)}, SGC: {np.sum(~is_ngc)}")

    # =========================================================================
    # PART 2: With and without survey fixed effects
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: EFFECT OF SURVEY FIXED EFFECTS")
    print("="*70)

    # Without survey FE (baseline)
    print("\n--- Without survey fixed effects (baseline) ---")
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    X_null = np.column_stack([np.ones(len(z)), mass_step])
    X_full = np.column_stack([np.ones(len(z)), d_signed, mass_step])

    beta_null, _, chi2_null, _ = gls_fit(residual, X_null, cov)
    beta_full, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov)

    delta_chi2_base = chi2_null - chi2_full
    p_base = 1 - stats.chi2.cdf(delta_chi2_base, 1)

    print(f"  γ_env = {beta_full[1]:+.4f} ± {np.sqrt(beta_cov[1,1]):.4f}")
    print(f"  Δχ² = {delta_chi2_base:.3f}, p = {p_base:.4f}")

    # With survey FE
    print("\n--- With survey fixed effects ---")
    result_fe = delta_chi2_test_with_survey_fe(mu, z, d_signed, host_mass, survey_ids, cov)

    print(f"  γ_env = {result_fe['gamma_env']:+.4f} ± {result_fe['gamma_env_err']:.4f}")
    print(f"  Δχ² = {result_fe['delta_chi2']:.3f}, p = {result_fe['p_value']:.4f}")
    print(f"  Number of survey FE: {result_fe['n_survey_fe']}")
    print(f"  Reference survey: {result_fe['reference_survey']}")

    # Compare
    print("\n>>> COMPARISON:")
    gamma_shift = result_fe['gamma_env'] - beta_full[1]
    print(f"  Δγ_env (with FE - without FE) = {gamma_shift:+.4f}")

    if abs(gamma_shift) < 0.5 * np.sqrt(beta_cov[1,1]):
        print("  ✓ γ_env is STABLE under survey fixed effects")
        print("  → Signal is NOT driven by survey-specific systematics")
    else:
        print("  ⚠️ γ_env changes significantly with survey FE")
        print("  → Survey composition may be confounding the signal")

    if result_fe['p_value'] < 0.05:
        print(f"  ✓ γ_env remains significant (p = {result_fe['p_value']:.4f}) with survey FE")
    else:
        print(f"  ⚠️ γ_env becomes non-significant (p = {result_fe['p_value']:.4f}) with survey FE")

    # =========================================================================
    # PART 3: NGC only with survey fixed effects
    # =========================================================================
    print("\n" + "="*70)
    print("PART 3: NGC-ONLY WITH SURVEY FIXED EFFECTS")
    print("="*70)

    idx_ngc = np.where(is_ngc)[0]
    z_ngc = z[is_ngc]
    mu_ngc = mu[is_ngc]
    host_mass_ngc = host_mass[is_ngc]
    survey_ids_ngc = survey_ids[is_ngc]
    cov_ngc = cov[np.ix_(idx_ngc, idx_ngc)]

    # Use NGC-only voids
    sn_x_ngc = sn_x[is_ngc]
    sn_y_ngc = sn_y[is_ngc]
    sn_z_ngc = sn_z[is_ngc]
    d_signed_ngc, _ = compute_environment(sn_x_ngc, sn_y_ngc, sn_z_ngc, vx_ngc, vy_ngc, vz_ngc, vr_ngc)

    # Without survey FE
    print("\n--- NGC without survey FE ---")
    mu_theory_ngc = COSMO_SN.distmod(z_ngc).value
    residual_ngc = mu_ngc - mu_theory_ngc
    mass_step_ngc = (host_mass_ngc >= 10).astype(float)

    X_null_ngc = np.column_stack([np.ones(len(z_ngc)), mass_step_ngc])
    X_full_ngc = np.column_stack([np.ones(len(z_ngc)), d_signed_ngc, mass_step_ngc])

    beta_null_ngc, _, chi2_null_ngc, _ = gls_fit(residual_ngc, X_null_ngc, cov_ngc)
    beta_full_ngc, beta_cov_ngc, chi2_full_ngc, _ = gls_fit(residual_ngc, X_full_ngc, cov_ngc)

    delta_chi2_ngc = chi2_null_ngc - chi2_full_ngc
    p_ngc = 1 - stats.chi2.cdf(delta_chi2_ngc, 1)

    print(f"  γ_env = {beta_full_ngc[1]:+.4f} ± {np.sqrt(beta_cov_ngc[1,1]):.4f}")
    print(f"  Δχ² = {delta_chi2_ngc:.3f}, p = {p_ngc:.4f}")

    # With survey FE
    print("\n--- NGC with survey FE ---")
    result_ngc_fe = delta_chi2_test_with_survey_fe(
        mu_ngc, z_ngc, d_signed_ngc, host_mass_ngc, survey_ids_ngc, cov_ngc
    )

    print(f"  γ_env = {result_ngc_fe['gamma_env']:+.4f} ± {result_ngc_fe['gamma_env_err']:.4f}")
    print(f"  Δχ² = {result_ngc_fe['delta_chi2']:.3f}, p = {result_ngc_fe['p_value']:.4f}")
    print(f"  Number of survey FE: {result_ngc_fe['n_survey_fe']}")

    # =========================================================================
    # PART 4: Within-survey analysis
    # =========================================================================
    print("\n" + "="*70)
    print("PART 4: WITHIN-SURVEY ANALYSIS")
    print("="*70)
    print("\nAnalyzing γ_env within each survey separately (min N=30):")

    within_results = within_survey_analysis(mu, z, d_signed, host_mass, survey_ids, cov, min_n=30)

    print(f"\n  {'Survey':<15} {'N':>5} {'z_mean':>7} {'γ_env':>10} {'± σ':>8} {'p-val':>8}")
    print(f"  {'-'*55}")

    # Sort by sample size
    sorted_surveys = sorted(within_results.items(), key=lambda x: -x[1]['n'])

    for survey_name, res in sorted_surveys:
        marker = "★" if res['p_value'] < 0.05 else " "
        print(f"  {survey_name:<15} {res['n']:>5} {res['z_mean']:>7.4f} "
              f"{res['gamma_env']:>+10.4f} ± {res['gamma_env_err']:.4f} "
              f"{res['p_value']:>8.4f} {marker}")

    # Check sign consistency
    positive_signs = sum(1 for r in within_results.values() if r['gamma_env'] > 0)
    total_surveys = len(within_results)

    print(f"\n>>> SIGN CONSISTENCY:")
    print(f"  {positive_signs}/{total_surveys} surveys show positive γ_env")

    if positive_signs >= total_surveys * 0.7:
        print("  ✓ Majority of surveys show consistent positive sign")
    else:
        print("  ? Mixed signs across surveys - interpret with caution")

    # =========================================================================
    # PART 5: Low-z only (z < 0.05) with survey fixed effects
    # =========================================================================
    print("\n" + "="*70)
    print("PART 5: LOW-Z (z < 0.05) WITH SURVEY FIXED EFFECTS")
    print("="*70)

    low_z_mask = z < 0.05
    idx_lowz = np.where(low_z_mask)[0]

    z_lowz = z[low_z_mask]
    mu_lowz = mu[low_z_mask]
    host_mass_lowz = host_mass[low_z_mask]
    survey_ids_lowz = survey_ids[low_z_mask]
    d_signed_lowz = d_signed[low_z_mask]
    cov_lowz = cov[np.ix_(idx_lowz, idx_lowz)]

    print(f"\nLow-z sample: {np.sum(low_z_mask)} SNe")

    # Without survey FE
    print("\n--- Low-z without survey FE ---")
    mu_theory_lowz = COSMO_SN.distmod(z_lowz).value
    residual_lowz = mu_lowz - mu_theory_lowz
    mass_step_lowz = (host_mass_lowz >= 10).astype(float)

    X_null_lowz = np.column_stack([np.ones(len(z_lowz)), mass_step_lowz])
    X_full_lowz = np.column_stack([np.ones(len(z_lowz)), d_signed_lowz, mass_step_lowz])

    beta_null_lowz, _, chi2_null_lowz, _ = gls_fit(residual_lowz, X_null_lowz, cov_lowz)
    beta_full_lowz, beta_cov_lowz, chi2_full_lowz, _ = gls_fit(residual_lowz, X_full_lowz, cov_lowz)

    delta_chi2_lowz = chi2_null_lowz - chi2_full_lowz
    p_lowz = 1 - stats.chi2.cdf(delta_chi2_lowz, 1)

    print(f"  γ_env = {beta_full_lowz[1]:+.4f} ± {np.sqrt(beta_cov_lowz[1,1]):.4f}")
    print(f"  Δχ² = {delta_chi2_lowz:.3f}, p = {p_lowz:.4f}")

    # With survey FE
    print("\n--- Low-z with survey FE ---")
    result_lowz_fe = delta_chi2_test_with_survey_fe(
        mu_lowz, z_lowz, d_signed_lowz, host_mass_lowz, survey_ids_lowz, cov_lowz
    )

    print(f"  γ_env = {result_lowz_fe['gamma_env']:+.4f} ± {result_lowz_fe['gamma_env_err']:.4f}")
    print(f"  Δχ² = {result_lowz_fe['delta_chi2']:.3f}, p = {result_lowz_fe['p_value']:.4f}")
    print(f"  Number of survey FE: {result_lowz_fe['n_survey_fe']}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: SURVEY CONTROL RESULTS")
    print("="*70)

    print("""
┌──────────────────────────────────────────────────────────────────────┐
│                      γ_env WITH/WITHOUT SURVEY FE                    │
├────────────────────┬─────────────────────┬───────────────────────────┤
│ Sample             │ Without Survey FE   │ With Survey FE            │
├────────────────────┼─────────────────────┼───────────────────────────┤""")

    print(f"│ {'Full sample':<18} │ {beta_full[1]:+.4f} (p={p_base:.4f})  │ {result_fe['gamma_env']:+.4f} (p={result_fe['p_value']:.4f})        │")
    print(f"│ {'NGC only':<18} │ {beta_full_ngc[1]:+.4f} (p={p_ngc:.4f})  │ {result_ngc_fe['gamma_env']:+.4f} (p={result_ngc_fe['p_value']:.4f})        │")
    print(f"│ {'Low-z (z<0.05)':<18} │ {beta_full_lowz[1]:+.4f} (p={p_lowz:.4f})  │ {result_lowz_fe['gamma_env']:+.4f} (p={result_lowz_fe['p_value']:.4f})        │")
    print("└────────────────────┴─────────────────────┴───────────────────────────┘")

    print("\n>>> CONCLUSION:")
    if result_fe['p_value'] < 0.1 and result_lowz_fe['p_value'] < 0.05:
        print("  ✓ Signal SURVIVES survey fixed effects")
        print("  ✓ NOT driven by survey-specific systematics")
        print("  → 'Footprint-specific' is NOT 'survey-specific'")
    else:
        print("  ? Results are mixed - further investigation needed")


if __name__ == '__main__':
    run_survey_control_analysis()
