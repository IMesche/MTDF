# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Test E: Null Signal Injection (SGC Recovery).

Injects a known environment signal into SGC data using the
precomputed Phase 3 d_signed values and tests whether the pipeline
can recover it. This diagnoses whether the SGC is sensitivity-limited.
"""

import numpy as np
from scipy import linalg

from mtdf_validation.phase3.gls_engine import gls_fit, delta_chi2_test, COSMO_SN
from .common import FINDERS


def _fit_null_sgc(data, finder):
    """Fit null model (no env) on SGC to get baseline parameters."""
    sgc = data.sgc_mask
    mu = data.sub['mu'][sgc]
    z = data.sub['z'][sgc]
    hm = data.sub['host_mass'][sgc]
    n = len(z)

    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (hm >= 10).astype(float)

    X_null = np.column_stack([np.ones(n), mass_step])
    beta_null, _, chi2_null, _ = gls_fit(residual, X_null, data.cov_inv_sgc)

    # Fitted null values
    mu_null_fitted = mu_theory + X_null @ beta_null

    return beta_null, mu_null_fitted, mu_theory, mass_step


def run_single_injection(data, finder, gamma_inject, mu_null_fitted,
                          mu_theory, mass_step, L_sgc, rng):
    """Run one injection realization on SGC."""
    d_sgc = data.d_signed_region[(finder, 'sgc')]
    n = len(d_sgc)

    # Simulate: mu_sim = null_fitted + gamma_inject * d_signed + noise
    eps = rng.randn(n)
    noise = L_sgc @ eps
    mu_sim = mu_null_fitted + gamma_inject * d_sgc + noise

    # Recover gamma via standard pipeline
    residual_sim = mu_sim - mu_theory
    X_null = np.column_stack([np.ones(n), mass_step])
    _, _, chi2_null, _ = gls_fit(residual_sim, X_null, data.cov_inv_sgc)

    X_full = np.column_stack([np.ones(n), d_sgc, mass_step])
    beta, beta_cov, chi2_full, dof = gls_fit(residual_sim, X_full, data.cov_inv_sgc)

    from scipy.stats import chi2 as chi2_dist
    dchi2 = chi2_null - chi2_full
    p_value = float(1.0 - chi2_dist.cdf(max(dchi2, 0), 1))

    gamma_recovered = float(beta[1])
    gamma_err = float(np.sqrt(beta_cov[1, 1]))

    return {
        'gamma_recovered': gamma_recovered,
        'gamma_err': gamma_err,
        'delta_chi2': float(dchi2),
        'p_value': p_value,
        'detected_2sigma': dchi2 > 4.0,  # chi2(1) > 4 ~ 2σ
        'detected_3sigma': dchi2 > 9.0,  # chi2(1) > 9 ~ 3σ
    }


def run_test_e(data):
    """Execute Test E for all finders and injection strengths."""
    config_e = data.config.get('test_e', {})
    n_real = config_e.get('n_realizations', 200)
    strengths = config_e.get('injection_strengths', [0.5, 1.0, 1.5])
    seed = config_e.get('seed', 456)
    finders = data.config.get('data', {}).get('finders', FINDERS)

    rng = np.random.RandomState(seed)
    results = {}

    # Get NGC gamma values from Phase 3 results
    p3 = data.phase3_results.get('table2_ngc_sgc', {})

    for finder in finders:
        print(f"\n  --- {finder.upper()} ---")

        # NGC gamma to use as injection signal
        gamma_ngc = p3.get(finder, {}).get('ngc', {}).get('gamma_env', 0.005)
        print(f"    γ_NGC (Phase 3) = {gamma_ngc:+.6f}")

        # Fit SGC null model once
        beta_null, mu_null_fitted, mu_theory, mass_step = _fit_null_sgc(data, finder)
        L_sgc = data.L_sgc

        finder_results = {}

        for strength in strengths:
            gamma_inject = strength * gamma_ngc
            print(f"    Injection {strength:.1f}x: γ_inject = {gamma_inject:+.6f} "
                  f"({n_real} reps)...")

            gammas = np.zeros(n_real)
            detected_2s = np.zeros(n_real, dtype=bool)
            detected_3s = np.zeros(n_real, dtype=bool)

            for ir in range(n_real):
                res = run_single_injection(
                    data, finder, gamma_inject,
                    mu_null_fitted, mu_theory, mass_step, L_sgc, rng
                )
                gammas[ir] = res['gamma_recovered']
                detected_2s[ir] = res['detected_2sigma']
                detected_3s[ir] = res['detected_3sigma']

            mean_g = float(np.mean(gammas))
            bias = mean_g - gamma_inject
            bias_rel = bias / gamma_inject if gamma_inject != 0 else 0.0

            summary = {
                'gamma_inject': float(gamma_inject),
                'strength': float(strength),
                'gamma_recovered_mean': mean_g,
                'gamma_recovered_std': float(np.std(gammas)),
                'gamma_recovered_median': float(np.median(gammas)),
                'bias': float(bias),
                'bias_relative': float(bias_rel),
                'ci_68': [float(np.percentile(gammas, 16)),
                          float(np.percentile(gammas, 84))],
                'ci_95': [float(np.percentile(gammas, 2.5)),
                          float(np.percentile(gammas, 97.5))],
                'detection_rate_2sigma': float(np.mean(detected_2s)),
                'detection_rate_3sigma': float(np.mean(detected_3s)),
                'n_realizations': n_real,
                'gammas_array': gammas,
            }

            print(f"      Recovered: {mean_g:+.4f} ± {np.std(gammas):.4f}, "
                  f"bias = {bias:+.4f} ({bias_rel:+.1%}), "
                  f"det@2σ = {np.mean(detected_2s):.1%}")

            finder_results[f'{strength:.1f}x'] = summary

        finder_results['gamma_ngc'] = float(gamma_ngc)
        results[finder] = finder_results

    return results
