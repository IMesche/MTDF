# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
GLS fitting engine for SN x void environment analysis.

Implements:
- Core GLS fit: beta = (X'C_inv X)^-1 X'C_inv y
- Delta-chi2 test for environment term
- Survey fixed effects (k-1 dummies)
- Permutation test (stratified by z-bin)
- Block bootstrap (resample within z-bins)
"""

import numpy as np
from scipy import linalg, stats
from .data_loader import COSMO_SN

# Pantheon+ survey name mapping
SURVEY_NAMES = {
    1: 'CfA1', 4: 'CfA2', 5: 'CfA3S', 10: 'CfA3K', 15: 'CfA4',
    18: 'CSP', 50: 'PS1MD', 51: 'SNLS', 56: 'SDSS', 57: 'Foundation',
    61: 'CNIa0.02', 62: 'LOSS', 63: 'SOUSA', 64: 'Misc_lowz',
    65: 'Misc_highz', 66: 'DES', 100: 'HST', 101: 'CFA4_p2',
    106: 'PS1_LOWZ', 150: 'CfA3',
}


def gls_fit(y, X, cov_inv):
    """Generalized Least Squares fit.

    Args:
        y: (n,) observations
        X: (n, p) design matrix
        cov_inv: (n, n) pre-computed precision matrix

    Returns:
        beta: (p,) parameter estimates
        beta_cov: (p, p) parameter covariance
        chi2: chi-squared statistic
        dof: degrees of freedom (n - p)
    """
    XtCinv = X.T @ cov_inv
    XtCinvX = XtCinv @ X
    beta_cov = linalg.inv(XtCinvX)
    beta = beta_cov @ (XtCinv @ y)

    residual = y - X @ beta
    chi2 = float(residual @ cov_inv @ residual)
    dof = len(y) - X.shape[1]

    return beta, beta_cov, chi2, dof


def delta_chi2_test(mu, z, env_metric, host_mass, cov_inv):
    """Test significance of environment term via nested model comparison.

    Null:  mu_resid = intercept + gamma_M * step(M >= 10)
    Full:  mu_resid = intercept + gamma_env * env + gamma_M * step(M >= 10)

    Returns dict with gamma_env, sigma, delta_chi2, p_value.
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory

    mass_step = (host_mass >= 10).astype(float)

    # Null model (no environment)
    X_null = np.column_stack([np.ones(n), mass_step])
    _, _, chi2_null, _ = gls_fit(residual, X_null, cov_inv)

    # Full model (with environment)
    X_full = np.column_stack([np.ones(n), env_metric, mass_step])
    beta_full, beta_cov_full, chi2_full, dof = gls_fit(residual, X_full, cov_inv)

    dchi2 = chi2_null - chi2_full
    p_value = float(1.0 - stats.chi2.cdf(dchi2, 1))

    gamma_env = float(beta_full[1])
    gamma_env_err = float(np.sqrt(beta_cov_full[1, 1]))

    return {
        'gamma_env': gamma_env,
        'gamma_env_err': gamma_env_err,
        'delta_chi2': float(dchi2),
        'p_value': p_value,
        'chi2_null': float(chi2_null),
        'chi2_full': float(chi2_full),
        'dof': dof,
        'n': n,
    }


def delta_chi2_test_with_survey_fe(mu, z, env_metric, host_mass, survey_ids, cov_inv):
    """Test gamma_env significance with per-survey fixed effects.

    Uses k-1 survey dummies (drops most common survey as reference)
    to avoid multicollinearity with the intercept.
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory

    mass_step = (host_mass >= 10).astype(float)

    # Create k-1 survey dummies
    unique_surveys = np.unique(survey_ids)
    counts = {s: int(np.sum(survey_ids == s)) for s in unique_surveys}
    ref_survey = max(counts, key=counts.get)

    survey_dummies = []
    survey_cols = []
    for s in unique_surveys:
        if s != ref_survey:
            survey_dummies.append((survey_ids == s).astype(float))
            survey_cols.append(SURVEY_NAMES.get(int(s), f'Survey_{int(s)}'))

    n_fe = len(survey_cols)

    if n_fe > 0:
        S = np.column_stack(survey_dummies)
        X_null = np.column_stack([np.ones(n), S, mass_step])
        X_full = np.column_stack([np.ones(n), S, env_metric, mass_step])
        gamma_env_idx = 1 + n_fe
    else:
        X_null = np.column_stack([np.ones(n), mass_step])
        X_full = np.column_stack([np.ones(n), env_metric, mass_step])
        gamma_env_idx = 1

    _, _, chi2_null, _ = gls_fit(residual, X_null, cov_inv)
    beta_full, beta_cov_full, chi2_full, dof = gls_fit(residual, X_full, cov_inv)

    dchi2 = chi2_null - chi2_full
    p_value = float(1.0 - stats.chi2.cdf(dchi2, 1))

    return {
        'gamma_env': float(beta_full[gamma_env_idx]),
        'gamma_env_err': float(np.sqrt(beta_cov_full[gamma_env_idx, gamma_env_idx])),
        'delta_chi2': float(dchi2),
        'p_value': p_value,
        'n_survey_fe': n_fe,
        'reference_survey': SURVEY_NAMES.get(int(ref_survey), f'Survey_{int(ref_survey)}'),
        'n': n,
    }


def loso_analysis(mu, z, env_metric, host_mass, survey_ids, cov, min_n=20):
    """Leave-One-Survey-Out stability analysis.

    For each survey, remove it and refit gamma_env.
    Returns list of per-survey results.
    """
    unique_surveys = np.unique(survey_ids)
    results = []

    for s in unique_surveys:
        mask = survey_ids != s
        n_removed = int(np.sum(~mask))
        if n_removed < 5:  # Skip trivial surveys
            continue

        idx = np.where(mask)[0]
        n_remain = len(idx)
        if n_remain < min_n:
            continue

        cov_sub = cov[np.ix_(idx, idx)]
        try:
            cov_inv_sub = linalg.inv(cov_sub)
        except linalg.LinAlgError:
            continue

        result = delta_chi2_test(
            mu[idx], z[idx], env_metric[idx], host_mass[idx], cov_inv_sub
        )
        result['survey_removed'] = SURVEY_NAMES.get(int(s), f'Survey_{int(s)}')
        result['n_removed'] = n_removed
        results.append(result)

    return results


def permutation_test(mu, z, env_metric, host_mass, cov_inv, n_perms=10000, seed=42):
    """Stratified permutation test for environment correlation.

    Shuffles env_metric within z-bins to preserve redshift structure.
    """
    rng = np.random.RandomState(seed)

    # Observed delta-chi2
    obs = delta_chi2_test(mu, z, env_metric, host_mass, cov_inv)
    obs_dchi2 = obs['delta_chi2']

    # Z-bin assignment for stratified shuffling
    z_bin_edges = [0.02, 0.05, 0.08, 0.12, 0.157, np.inf]
    z_bins = np.digitize(z, z_bin_edges)

    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    # Pre-compute null chi2 (doesn't change across permutations)
    X_null = np.column_stack([np.ones(n), mass_step])
    _, _, chi2_null, _ = gls_fit(residual, X_null, cov_inv)

    null_dchi2 = np.zeros(n_perms)

    for ip in range(n_perms):
        env_shuffled = env_metric.copy()
        for b in np.unique(z_bins):
            idx_bin = np.where(z_bins == b)[0]
            env_shuffled[idx_bin] = rng.permutation(env_shuffled[idx_bin])

        X_full = np.column_stack([np.ones(n), env_shuffled, mass_step])
        _, _, chi2_full, _ = gls_fit(residual, X_full, cov_inv)
        null_dchi2[ip] = chi2_null - chi2_full

    p_perm = float(np.mean(null_dchi2 >= obs_dchi2))

    return {
        'obs_delta_chi2': float(obs_dchi2),
        'p_permutation': p_perm,
        'null_mean': float(np.mean(null_dchi2)),
        'null_std': float(np.std(null_dchi2)),
        'null_95pct': float(np.percentile(null_dchi2, 95)),
        'n_perms': n_perms,
    }


def block_bootstrap(mu, z, env_metric, host_mass, cov, n_boots=5000, seed=123):
    """Block bootstrap resampling within z-bins.

    Resamples SNe with replacement within z-bins, refits gamma_env.
    Uses Tikhonov regularization on the bootstrap covariance since
    resampling with replacement creates duplicate rows -> singular matrix.
    """
    rng = np.random.RandomState(seed)

    z_bin_edges = [0.02, 0.05, 0.08, 0.12, 0.157, np.inf]
    z_bins = np.digitize(z, z_bin_edges)
    unique_bins = np.unique(z_bins)

    # Regularization scale: small fraction of diagonal
    reg_scale = 1e-8 * np.mean(np.diag(cov))

    gamma_boots = np.zeros(n_boots)
    n_failed = 0

    for ib in range(n_boots):
        # Resample indices within each z-bin
        boot_idx = []
        for b in unique_bins:
            idx_bin = np.where(z_bins == b)[0]
            resampled = rng.choice(idx_bin, size=len(idx_bin), replace=True)
            boot_idx.extend(resampled)

        boot_idx = np.array(boot_idx)
        cov_boot = cov[np.ix_(boot_idx, boot_idx)]
        # Regularize for duplicate rows
        cov_boot += reg_scale * np.eye(len(boot_idx))

        try:
            cov_inv_boot = linalg.inv(cov_boot)
        except linalg.LinAlgError:
            gamma_boots[ib] = np.nan
            n_failed += 1
            continue

        result = delta_chi2_test(
            mu[boot_idx], z[boot_idx], env_metric[boot_idx],
            host_mass[boot_idx], cov_inv_boot
        )
        gamma_boots[ib] = result['gamma_env']

    valid = gamma_boots[~np.isnan(gamma_boots)]

    if len(valid) == 0:
        return {
            'gamma_env_mean': float('nan'),
            'gamma_env_std': float('nan'),
            'gamma_env_median': float('nan'),
            'ci_68': [float('nan'), float('nan')],
            'ci_95': [float('nan'), float('nan')],
            'n_boots': n_boots,
            'n_valid': 0,
            'n_failed': n_failed,
        }

    return {
        'gamma_env_mean': float(np.mean(valid)),
        'gamma_env_std': float(np.std(valid)),
        'gamma_env_median': float(np.median(valid)),
        'ci_68': [float(np.percentile(valid, 16)), float(np.percentile(valid, 84))],
        'ci_95': [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))],
        'n_boots': n_boots,
        'n_valid': len(valid),
        'n_failed': n_failed,
    }


def z_binned_analysis(mu, z, env_metric, host_mass, cov):
    """Fit gamma_env in redshift bins to show z-dependent weakening."""
    z_bins = [(0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.157)]
    results = []

    for z_lo, z_hi in z_bins:
        mask = (z >= z_lo) & (z < z_hi)
        n_bin = int(np.sum(mask))
        if n_bin < 20:
            results.append({
                'z_range': [z_lo, z_hi],
                'z_mean': float(np.mean(z[mask])) if n_bin > 0 else None,
                'n': n_bin,
                'skipped': True,
            })
            continue

        idx = np.where(mask)[0]
        cov_sub = cov[np.ix_(idx, idx)]

        try:
            cov_inv_sub = linalg.inv(cov_sub)
        except linalg.LinAlgError:
            results.append({'z_range': [z_lo, z_hi], 'n': n_bin, 'skipped': True})
            continue

        result = delta_chi2_test(
            mu[idx], z[idx], env_metric[idx], host_mass[idx], cov_inv_sub
        )
        result['z_range'] = [z_lo, z_hi]
        result['z_mean'] = float(np.mean(z[idx]))
        result['skipped'] = False
        results.append(result)

    return results


def z_modulation_models(mu, z, env_metric, host_mass, cov_inv):
    """Fit piecewise and linear z-modulated environment models.

    Returns results for constant, piecewise, and linear interaction models.
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    results = {}

    # Constant gamma_env
    X_const = np.column_stack([np.ones(n), env_metric, mass_step])
    beta_c, cov_c, chi2_c, dof_c = gls_fit(residual, X_const, cov_inv)
    results['constant'] = {
        'gamma_env': float(beta_c[1]),
        'gamma_env_err': float(np.sqrt(cov_c[1, 1])),
        'chi2': float(chi2_c),
    }

    # Piecewise: separate gamma for z < 0.05 and z >= 0.05
    z_cut = 0.05
    low_z = (z < z_cut).astype(float)
    high_z = (z >= z_cut).astype(float)
    X_piece = np.column_stack([np.ones(n), env_metric * low_z, env_metric * high_z, mass_step])
    beta_p, cov_p, chi2_p, dof_p = gls_fit(residual, X_piece, cov_inv)

    dchi2_piece = chi2_c - chi2_p
    p_piece = float(1.0 - stats.chi2.cdf(dchi2_piece, 1))

    results['piecewise'] = {
        'gamma_env_low': float(beta_p[1]),
        'gamma_env_low_err': float(np.sqrt(cov_p[1, 1])),
        'gamma_env_high': float(beta_p[2]),
        'gamma_env_high_err': float(np.sqrt(cov_p[2, 2])),
        'z_cut': z_cut,
        'n_low': int(np.sum(z < z_cut)),
        'n_high': int(np.sum(z >= z_cut)),
        'delta_chi2_vs_constant': float(dchi2_piece),
        'p_piecewise': p_piece,
    }

    # Linear z-interaction: gamma_env * d * (1 + beta_z * (z - z_mean))
    z_centered = z - np.mean(z)
    env_z_interaction = env_metric * z_centered
    X_lin = np.column_stack([np.ones(n), env_metric, env_z_interaction, mass_step])
    beta_l, cov_l, chi2_l, dof_l = gls_fit(residual, X_lin, cov_inv)

    dchi2_lin = chi2_c - chi2_l
    p_lin = float(1.0 - stats.chi2.cdf(dchi2_lin, 1))

    results['linear_z'] = {
        'gamma_env_0': float(beta_l[1]),
        'gamma_env_0_err': float(np.sqrt(cov_l[1, 1])),
        'gamma_env_z': float(beta_l[2]),
        'gamma_env_z_err': float(np.sqrt(cov_l[2, 2])),
        'delta_chi2_vs_constant': float(dchi2_lin),
        'p_interaction': p_lin,
    }

    return results
