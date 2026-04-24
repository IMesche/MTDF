# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Mock SN sample generator for sensitivity forecasts.

Generates mock Type Ia SN samples with injected environment signal,
drawing from the observed distributions of:
  - redshift z (from Pantheon+ low-z sample)
  - environment metric d_signed (from Phase 3 crossmatch)
  - host mass (from Pantheon+)
  - measurement uncertainties (from Pantheon+ diagonal errors)
"""

import numpy as np


class MockGenerator:
    """Generate mock SN datasets based on observed distributions."""

    def __init__(self, z_obs, d_signed_obs, host_mass_obs, mu_err_obs):
        """Initialize from observed data.

        Args:
            z_obs: observed redshifts (N,)
            d_signed_obs: observed signed distances to nearest void (N,)
            host_mass_obs: observed host log masses (N,)
            mu_err_obs: observed distance modulus errors (N,) — sqrt(diag(cov))
        """
        self.z_obs = z_obs
        self.d_signed_obs = d_signed_obs
        self.host_mass_obs = host_mass_obs
        self.mu_err_obs = mu_err_obs
        self.n_obs = len(z_obs)

    def generate(self, n_sne, gamma_env_true, rng=None):
        """Generate one mock SN sample.

        Args:
            n_sne: number of SNe in the mock sample
            gamma_env_true: true injected environment signal (mag)
            rng: numpy RandomState

        Returns:
            dict with z, d_signed, host_mass, mu_residual, mu_err
        """
        if rng is None:
            rng = np.random.RandomState()

        # Draw from observed distributions (with replacement)
        idx = rng.choice(self.n_obs, size=n_sne, replace=True)

        z = self.z_obs[idx]
        d_signed = self.d_signed_obs[idx]
        host_mass = self.host_mass_obs[idx]
        mu_err = self.mu_err_obs[idx]

        # Generate mock residuals:
        # mu_residual = gamma_env_true * d_signed + noise
        # (intercept and mass step are nuisance parameters absorbed by the fit)
        noise = rng.normal(0, mu_err)
        mu_residual = gamma_env_true * d_signed + noise

        return {
            'z': z,
            'd_signed': d_signed,
            'host_mass': host_mass,
            'mu_residual': mu_residual,
            'mu_err': mu_err,
        }

    def generate_batch(self, n_sne, gamma_env_true, n_mocks, seed=42):
        """Generate a batch of mock samples.

        Args:
            n_sne: SNe per mock
            gamma_env_true: true injected signal
            n_mocks: number of mock realizations
            seed: random seed

        Returns:
            list of mock dicts
        """
        rng = np.random.RandomState(seed)
        return [self.generate(n_sne, gamma_env_true, rng) for _ in range(n_mocks)]
