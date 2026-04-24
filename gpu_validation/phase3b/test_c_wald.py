# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Test C: Wald Test for NGC/SGC asymmetry + Parametric Bootstrap LR calibration.

Tests H0: gamma_NGC = gamma_SGC using:
1. Wald statistic (analytic)
2. Parametric bootstrap under H0 (shared gamma) for LR calibration
"""

import numpy as np
from scipy import linalg, stats

from mtdf_validation.phase3.gls_engine import gls_fit, delta_chi2_test, COSMO_SN
from .common import FINDERS, delta_gamma_with_se, region_gls


def wald_test_delta_gamma(data, finder):
    """Wald test: W = (γ_NGC - γ_SGC)^2 / (SE_NGC^2 + SE_SGC^2) ~ chi2(1)."""
    res_ngc = region_gls(data, 'ngc', finder)
    res_sgc = region_gls(data, 'sgc', finder)
    dg = delta_gamma_with_se(res_ngc, res_sgc)

    wald_stat = dg['z_score'] ** 2  # W = z^2 ~ chi2(1)
    p_wald = float(1.0 - stats.chi2.cdf(wald_stat, 1))

    return {
        'gamma_ngc': res_ngc['gamma_env'],
        'gamma_ngc_err': res_ngc['gamma_env_err'],
        'gamma_sgc': res_sgc['gamma_env'],
        'gamma_sgc_err': res_sgc['gamma_env_err'],
        'delta_gamma': dg['delta_gamma'],
        'delta_se': dg['delta_se'],
        'wald_stat': float(wald_stat),
        'p_wald': float(p_wald),
        'z_score': dg['z_score'],
        'n_ngc': res_ngc['n'],
        'n_sgc': res_sgc['n'],
    }


def _fit_null_combined(data, finder):
    """Fit pooled model (shared gamma) on full combined sample.

    Returns fitted mu values and null model parameters.
    """
    n = len(data.sub['z'])
    mu = data.sub['mu']
    z = data.sub['z']
    hm = data.sub['host_mass']
    d = data.d_signed_combined[finder]

    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (hm >= 10).astype(float)

    # Full model with shared gamma (not split by region)
    X_full = np.column_stack([np.ones(n), d, mass_step])
    beta, beta_cov, chi2, dof = gls_fit(residual, X_full, data.cov_inv)

    # Fitted values
    mu_fitted = mu_theory + X_full @ beta

    return {
        'beta': beta,
        'mu_fitted': mu_fitted,
        'mu_theory': mu_theory,
        'chi2_combined': chi2,
        'X_full': X_full,
    }


def _compute_obs_lr(data, finder):
    """Observed LR = chi2_combined(shared gamma) - chi2_ngc - chi2_sgc."""
    # Combined model (shared gamma)
    null_fit = _fit_null_combined(data, finder)

    # Separate models
    res_ngc = region_gls(data, 'ngc', finder)
    res_sgc = region_gls(data, 'sgc', finder)

    lr = null_fit['chi2_combined'] - res_ngc['chi2_full'] - res_sgc['chi2_full']
    return float(lr), null_fit, res_ngc, res_sgc


def parametric_bootstrap_lr(data, finder, n_bootstrap=1000, seed=42):
    """Parametric bootstrap calibration of the LR statistic under H0.

    Under H0 (shared gamma), simulate data from the combined model
    and measure how often the separate-fit LR exceeds the observed.
    """
    rng = np.random.RandomState(seed)

    obs_lr, null_fit, _, _ = _compute_obs_lr(data, finder)

    n = len(data.sub['z'])
    mu_theory = null_fit['mu_theory']
    X_full = null_fit['X_full']
    beta_null = null_fit['beta']
    L = data.L_sub

    # Pre-extract regional info
    ngc = data.ngc_mask
    sgc = data.sgc_mask
    ngc_idx = data.ngc_idx
    sgc_idx = data.sgc_idx

    z_all = data.sub['z']
    hm_all = data.sub['host_mass']
    d_combined = data.d_signed_combined[finder]

    bootstrap_lr = np.zeros(n_bootstrap)

    for ib in range(n_bootstrap):
        # Simulate under H0: mu = X @ beta_null + L @ epsilon
        eps = rng.randn(n)
        mu_sim = mu_theory + X_full @ beta_null + L @ eps

        # Fit combined (shared gamma)
        residual_sim = mu_sim - mu_theory
        mass_step_all = (hm_all >= 10).astype(float)
        X_comb = np.column_stack([np.ones(n), d_combined, mass_step_all])
        _, _, chi2_comb, _ = gls_fit(residual_sim, X_comb, data.cov_inv)

        # Fit NGC separately
        res_ngc_sim = residual_sim[ngc]
        d_ngc = d_combined[ngc]
        ms_ngc = mass_step_all[ngc]
        X_ngc = np.column_stack([np.ones(len(ngc_idx)), d_ngc, ms_ngc])
        _, _, chi2_ngc, _ = gls_fit(res_ngc_sim, X_ngc, data.cov_inv_ngc)

        # Fit SGC separately
        res_sgc_sim = residual_sim[sgc]
        d_sgc = d_combined[sgc]
        ms_sgc = mass_step_all[sgc]
        X_sgc = np.column_stack([np.ones(len(sgc_idx)), d_sgc, ms_sgc])
        _, _, chi2_sgc, _ = gls_fit(res_sgc_sim, X_sgc, data.cov_inv_sgc)

        bootstrap_lr[ib] = chi2_comb - chi2_ngc - chi2_sgc

    p_bootstrap = float(np.mean(bootstrap_lr >= obs_lr))

    return {
        'obs_lr': obs_lr,
        'bootstrap_lr': bootstrap_lr,
        'p_bootstrap': p_bootstrap,
        'mean_lr': float(np.mean(bootstrap_lr)),
        'std_lr': float(np.std(bootstrap_lr)),
        'pct_95': float(np.percentile(bootstrap_lr, 95)),
        'n_bootstrap': n_bootstrap,
    }


def run_test_c(data):
    """Execute Test C for all finders."""
    config_c = data.config.get('test_c', {})
    n_boot = config_c.get('n_bootstrap', 1000)
    seed = config_c.get('seed', 42)
    finders = data.config.get('data', {}).get('finders', FINDERS)

    results = {}
    for finder in finders:
        print(f"\n  --- {finder.upper()} ---")

        # Wald test
        wald = wald_test_delta_gamma(data, finder)
        print(f"    Wald: Δγ = {wald['delta_gamma']:+.4f} ± {wald['delta_se']:.4f}, "
              f"W = {wald['wald_stat']:.3f}, p = {wald['p_wald']:.3f}")

        # Parametric bootstrap
        print(f"    Running parametric bootstrap ({n_boot} reps)...")
        boot = parametric_bootstrap_lr(data, finder, n_boot, seed)
        print(f"    Bootstrap: obs LR = {boot['obs_lr']:.3f}, "
              f"p_boot = {boot['p_bootstrap']:.3f}")

        results[finder] = {
            'wald': wald,
            'bootstrap': {k: v for k, v in boot.items() if k != 'bootstrap_lr'},
            'bootstrap_lr_array': boot['bootstrap_lr'],
        }

    return results
