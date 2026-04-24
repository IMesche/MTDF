# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Chi-squared computation engine for Phase 1.
Implements all chi2 calculations independently.
"""

import numpy as np
from . import standalone_mtdf as mtdf


def chi2_sne_marginalized(z, mu_obs, cov, params):
    """
    SNe chi2 with analytic marginalization over absolute magnitude M.
    chi2 = C - B^2/A where:
      A = 1^T C_inv 1
      B = 1^T C_inv delta_mu
      C = delta_mu^T C_inv delta_mu
    Returns: (chi2, dof)
    """
    mu_model = mtdf.distance_modulus(z, params)
    delta_mu = mu_obs - mu_model

    try:
        C_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov) / len(z)
        C_inv = np.linalg.inv(cov + reg * np.eye(len(z)))

    ones = np.ones(len(z))
    A = ones @ C_inv @ ones
    B = ones @ C_inv @ delta_mu
    C_val = delta_mu @ C_inv @ delta_mu

    chi2 = C_val - B**2 / A
    dof = len(z) - 1  # marginalized M

    return float(chi2), dof


def chi2_bao(z_eff, obs, obs_types, cov, params):
    """
    BAO chi2: (obs - model)^T C_inv (obs - model)
    Returns: (chi2, dof)
    """
    model = mtdf.bao_predictions(z_eff, obs_types, params)
    residual = obs - model

    try:
        C_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov) / len(obs)
        C_inv = np.linalg.inv(cov + reg * np.eye(len(obs)))

    chi2 = float(residual @ C_inv @ residual)
    dof = len(obs)

    return chi2, dof


def chi2_hz(z, H_obs, cov, params):
    """
    Cosmic chronometer H(z) chi2.
    Returns: (chi2, dof)
    """
    H_model = np.array([mtdf.H_mtdf(zi, params) for zi in z])
    residual = H_obs - H_model

    try:
        C_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov) / len(z)
        C_inv = np.linalg.inv(cov + reg * np.eye(len(z)))

    chi2 = float(residual @ C_inv @ residual)
    dof = len(z)

    return chi2, dof


def chi2_fsigma8(z, fsig8_obs, cov, params):
    """
    fsigma8 chi2 with analytic sigma8_0 fit.
    Returns: (chi2, dof, sigma8_bf, sigma8_err)
    """
    sigma8_bf, sigma8_err, chi2, dof = mtdf.fit_sigma8_analytic(
        z, fsig8_obs, cov, params
    )
    return chi2, dof, sigma8_bf, sigma8_err


def chi2_cmb_distance(obs_means, obs_cov, params):
    """
    CMB distance prior chi2.
    Returns: (chi2, dof)
    """
    pred, diag = mtdf.cmb_distance_predictions(params)
    residual = obs_means - pred

    try:
        C_inv = np.linalg.inv(obs_cov)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(obs_cov) / len(obs_means)
        C_inv = np.linalg.inv(obs_cov + reg * np.eye(len(obs_means)))

    chi2 = float(residual @ C_inv @ residual)
    dof = len(obs_means)

    return chi2, dof, diag
