# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Test D: Geometry and Mode-Coupling Check.

Generates isotropic mock catalogues by globally shuffling d_signed
(destroying any real environment signal) and running the full
NGC/SGC split pipeline. Tests whether survey geometry alone can
produce the observed Δγ.
"""

import numpy as np
from mtdf_validation.phase3.gls_engine import gls_fit, COSMO_SN
from .common import FINDERS, delta_gamma_with_se


def _fit_region_from_dsigned(mu, z, host_mass, d_signed, cov_inv):
    """Quick GLS fit returning gamma_env and error."""
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    X_null = np.column_stack([np.ones(n), mass_step])
    _, _, chi2_null, _ = gls_fit(residual, X_null, cov_inv)

    X_full = np.column_stack([np.ones(n), d_signed, mass_step])
    beta, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov_inv)

    return {
        'gamma_env': float(beta[1]),
        'gamma_env_err': float(np.sqrt(beta_cov[1, 1])),
        'chi2_full': float(chi2_full),
    }


def run_single_mock(data, finder, d_shuffled):
    """Run NGC/SGC split on a single mock d_signed."""
    ngc = data.ngc_mask
    sgc = data.sgc_mask

    # NGC
    res_ngc = _fit_region_from_dsigned(
        data.sub['mu'][ngc], data.sub['z'][ngc],
        data.sub['host_mass'][ngc],
        d_shuffled[ngc], data.cov_inv_ngc,
    )

    # SGC
    res_sgc = _fit_region_from_dsigned(
        data.sub['mu'][sgc], data.sub['z'][sgc],
        data.sub['host_mass'][sgc],
        d_shuffled[sgc], data.cov_inv_sgc,
    )

    dg = res_ngc['gamma_env'] - res_sgc['gamma_env']
    return dg, res_ngc['gamma_env'], res_sgc['gamma_env']


def run_test_d(data):
    """Execute Test D: generate isotropic mocks, measure Δγ distribution."""
    config_d = data.config.get('test_d', {})
    n_mocks = config_d.get('n_mocks', 200)
    seed = config_d.get('seed', 123)
    finders = data.config.get('data', {}).get('finders', FINDERS)

    rng = np.random.RandomState(seed)
    results = {}

    for finder in finders:
        print(f"\n  --- {finder.upper()} ({n_mocks} mocks) ---")

        d_orig = data.d_signed_combined[finder]

        # Observed Δγ from Phase 3
        from .common import region_gls
        res_ngc_obs = region_gls(data, 'ngc', finder)
        res_sgc_obs = region_gls(data, 'sgc', finder)
        obs_dg = res_ngc_obs['gamma_env'] - res_sgc_obs['gamma_env']

        mock_dg = np.zeros(n_mocks)
        mock_g_ngc = np.zeros(n_mocks)
        mock_g_sgc = np.zeros(n_mocks)

        for im in range(n_mocks):
            d_shuffled = rng.permutation(d_orig)
            mock_dg[im], mock_g_ngc[im], mock_g_sgc[im] = \
                run_single_mock(data, finder, d_shuffled)

        p_empirical = float(np.mean(np.abs(mock_dg) >= np.abs(obs_dg)))

        print(f"    Observed Δγ = {obs_dg:+.4f}")
        print(f"    Mock Δγ: mean={np.mean(mock_dg):+.4f}, std={np.std(mock_dg):.4f}")
        print(f"    p_empirical = {p_empirical:.3f} "
              f"(fraction of |mock| >= |obs|)")

        results[finder] = {
            'obs_delta_gamma': float(obs_dg),
            'obs_gamma_ngc': float(res_ngc_obs['gamma_env']),
            'obs_gamma_sgc': float(res_sgc_obs['gamma_env']),
            'mock_delta_gammas': mock_dg,
            'mock_mean': float(np.mean(mock_dg)),
            'mock_std': float(np.std(mock_dg)),
            'p_empirical': p_empirical,
            'n_mocks': n_mocks,
        }

    return results
