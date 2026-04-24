# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Late-universe likelihoods for combined MCMC: BAO + SNe Ia.

MTDF's Hubble tension mechanism operates through the stress correction:
  H(z) = H0 * E(z) * (1 + k_f * kappa * alpha * z/(1+z))

At k_f=0: pure LCDM. At k_f=1: full MTDF theory prediction.

BAO and SNe probe distances at low-to-intermediate redshifts where this
correction has its primary effect.
"""

import math
import numpy as np
from pathlib import Path

# MTDF fixed parameters
KAPPA = 0.00102  # Anchor: approx f_kick/3 = (1-beta_eos)^2/(72*(1+alpha))
ALPHA = 1.30
K_RS_EFE = -0.215
LAMBDA_MTDF = (1.0 - 0.573)**2 / (1.0 + 1.30)  # = 0.07927
F_KICK = LAMBDA_MTDF / 24.0                       # = 0.003303

C_LIGHT = 299792.458  # km/s


def H_mtdf(z, H0, Omega_m, k_f):
    """MTDF Hubble rate with stress correction scaled by k_f.

    H(z) = H0 * E(z) * (1 + k_f * kappa * alpha * z/(1+z))
    """
    Omega_L = 1.0 - Omega_m
    E = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    stress = k_f * KAPPA * ALPHA * z / (1 + z)
    return H0 * E * (1 + stress)


def comoving_distances(z_query, H0, Omega_m, k_f, n_grid=2000):
    """Vectorized comoving distance via grid interpolation.

    Returns D_C in Mpc for an array of redshifts.
    """
    z_max = float(np.max(z_query)) * 1.05 + 0.05
    z_grid = np.linspace(0, z_max, n_grid)
    H_grid = H_mtdf(z_grid, H0, Omega_m, k_f)
    integrand = C_LIGHT / H_grid
    dz = np.diff(z_grid)
    D_C_grid = np.zeros(n_grid)
    D_C_grid[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    return np.interp(z_query, z_grid, D_C_grid)


def sound_horizon_rd(omega_b, omega_m, omega_nu=0.0):
    """Drag-epoch sound horizon r_d (Aubourg et al. 2015, Eq. 16)."""
    return 55.154 * math.exp(-72.3 * (omega_nu + 0.0006)**2) / \
           (omega_m**0.25351 * omega_b**0.12807)


def sound_horizon_mtdf(omega_b, omega_m, k_f):
    """MTDF-corrected sound horizon.

    r_d_MTDF = r_d_LCDM * (1 + k_f * K_RS_EFE * F_KICK)
    """
    r_d_lcdm = sound_horizon_rd(omega_b, omega_m)
    return r_d_lcdm * (1.0 + k_f * K_RS_EFE * F_KICK)


class BAOLikelihood:
    """DESI Y1 BAO likelihood (12 measurements)."""

    def __init__(self, data_dir):
        from phase1.data_loaders import load_desi_bao
        self.z_eff, self.obs, self.types, cov = load_desi_bao(data_dir)
        self.n = len(self.obs)
        self.invcov = np.linalg.inv(cov)

    def chi2(self, H0, Omega_m, omega_b, omega_m_h2, k_f):
        """Compute BAO chi2.

        Parameters: H0, Omega_m, omega_b (=omega_b*h^2), omega_m_h2, k_f.
        """
        r_d = sound_horizon_mtdf(omega_b, omega_m_h2, k_f)
        D_M = comoving_distances(self.z_eff, H0, Omega_m, k_f)

        model = np.empty(self.n)
        for i in range(self.n):
            z = self.z_eff[i]
            dm = D_M[i]
            H_z = float(H_mtdf(z, H0, Omega_m, k_f))
            D_H = C_LIGHT / H_z
            otype = self.types[i]

            if 'DV' in otype:
                D_V = (dm**2 * C_LIGHT * z / H_z)**(1.0/3.0)
                model[i] = D_V / r_d
            elif 'DM' in otype:
                model[i] = dm / r_d
            elif 'DH' in otype:
                model[i] = D_H / r_d
            else:
                model[i] = 0.0

        residual = self.obs - model
        return float(residual @ self.invcov @ residual)


class SNeLikelihood:
    """Pantheon+ SH0ES SNe Ia likelihood (1701 distance moduli).

    Analytically marginalizes over the absolute magnitude M.
    """

    def __init__(self, data_dir):
        from phase1.data_loaders import load_pantheonplus
        self.z, self.mu_obs, cov = load_pantheonplus(data_dir)
        self.n = len(self.z)
        # Pre-invert covariance (regularized)
        try:
            self.invcov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            reg = 1e-10 * np.trace(cov) / self.n
            self.invcov = np.linalg.inv(cov + reg * np.eye(self.n))
        # Pre-compute constants for M marginalization
        self.ones = np.ones(self.n)
        self.Cinv_ones = self.invcov @ self.ones
        self.A = float(self.ones @ self.Cinv_ones)

    def chi2(self, H0, Omega_m, k_f):
        """SNe chi2 with analytic M marginalization.

        chi2 = C - B^2/A where C = delta^T C_inv delta,
        B = 1^T C_inv delta, A = 1^T C_inv 1.
        """
        D_C = comoving_distances(self.z, H0, Omega_m, k_f)
        D_L = (1 + self.z) * D_C
        # Guard against D_L <= 0
        if np.any(D_L <= 0):
            return 1e10
        mu_model = 5.0 * np.log10(D_L) + 25.0

        delta = self.mu_obs - mu_model
        Cinv_delta = self.invcov @ delta

        C_val = float(delta @ Cinv_delta)
        B = float(self.ones @ Cinv_delta)

        return C_val - B**2 / self.A
