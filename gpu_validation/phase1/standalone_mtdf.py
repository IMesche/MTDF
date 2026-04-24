# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Standalone MTDF physics engine — independent reimplementation.
Does NOT import from vector_pillars.py.

Implements:
  - H(z) with MTDF stress correction
  - Comoving, luminosity distances
  - Sound horizon (Aubourg et al. 2015)
  - Growth ODE with mu(a) modification
  - CMB distance prior predictions
"""

import numpy as np
from scipy.integrate import quad
import math

# Physical constants
C_LIGHT = 299792.458   # km/s
T_CMB = 2.7255         # K
N_EFF = 3.046


# ============================================================================
# EXPANSION HISTORY
# ============================================================================

def H_mtdf(z, params):
    """
    MTDF Hubble parameter.
    H(z) = H0 * E_LCDM(z) * (1 + kappa * alpha * z / (1+z))
    """
    H0 = params['H0']
    Omega_m = params['Omega_m']
    kappa = params['kappa']
    alpha = params['alpha']

    Omega_L = 1.0 - Omega_m
    E = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    stress_corr = kappa * alpha * z / (1 + z)

    return H0 * E * (1 + stress_corr)


def comoving_distance(z, params):
    """
    Comoving distance D_C(z) = c * integral(0, z, dz'/H(z')).
    Uses scipy.integrate.quad for accuracy.
    """
    if np.isscalar(z):
        if z <= 0:
            return 0.0
        result, _ = quad(lambda zp: C_LIGHT / H_mtdf(zp, params), 0, z)
        return result
    return np.array([comoving_distance(zi, params) for zi in z])


def luminosity_distance(z, params):
    """D_L(z) = (1+z) * D_C(z)"""
    return (1 + np.asarray(z)) * comoving_distance(z, params)


def distance_modulus(z, params):
    """mu(z) = 5 * log10(D_L [Mpc]) + 25"""
    D_L = luminosity_distance(z, params)
    return 5.0 * np.log10(D_L) + 25.0


# ============================================================================
# SOUND HORIZON
# ============================================================================

def sound_horizon_aubourg(params):
    """
    Aubourg et al. 2015 (arXiv:1411.1074) Eq. 16.
    Returns (r_d, r_s) in Mpc.
      r_d = drag epoch sound horizon (for BAO)
      r_s = recombination sound horizon = RS_ZSTAR_RATIO * r_d (for CMB)
    """
    w_b = params['omegab_h2']
    w_m = params['omegam_h2']
    w_nu = params.get('omeganuh2', 0.0)
    ratio = params.get('RS_ZSTAR_RATIO', 0.9819)

    r_d = 55.154 * math.exp(-72.3 * (w_nu + 0.0006)**2) \
          / (w_m**0.25351 * w_b**0.12807)
    r_s = ratio * r_d

    return r_d, r_s


# ============================================================================
# BAO PREDICTIONS
# ============================================================================

def bao_predictions(z_eff, obs_types, params):
    """
    BAO observables: D_V/r_d, D_M/r_d, D_H/r_d.
    """
    r_d, _ = sound_horizon_aubourg(params)
    preds = []

    for z, otype in zip(z_eff, obs_types):
        H_z = H_mtdf(z, params)
        D_M = comoving_distance(z, params)
        D_H = C_LIGHT / H_z
        D_V = (D_M**2 * C_LIGHT * z / H_z)**(1.0/3.0)

        if 'DV' in otype:
            preds.append(D_V / r_d)
        elif 'DM' in otype:
            preds.append(D_M / r_d)
        elif 'DH' in otype:
            preds.append(D_H / r_d)
        else:
            preds.append(np.nan)

    return np.array(preds)


# ============================================================================
# GROWTH RATE (ODE SOLVER)
# ============================================================================

def mu_mtdf(a, params):
    """
    MTDF effective gravitational coupling.
    mu(a) = 1 + amp * T(a)
    where T(a) = (a/a_t)^alpha / (1 + (a/a_t)^alpha)
    and   amp  = (1 - beta_eos)^2 / (1 + alpha)
    """
    alpha = params['alpha']
    beta_eos = params['beta_eos']
    z_t = params['z_t']

    a_t = 1.0 / (1.0 + z_t)
    x = a / a_t

    if x <= 0:
        return 1.0

    x_pow = x ** alpha
    T = x_pow / (1.0 + x_pow)
    amp = (1.0 - beta_eos)**2 / (1.0 + alpha)

    return 1.0 + amp * T


def solve_growth_ode(params, a_init=1e-3, n_steps=500):
    """
    Solve linear growth ODE with MTDF mu(a):
      D''(a) + (3/a + H'(a)/H(a)) D'(a) - (3/2) mu(a) Omega_m(a) D(a) / a^2 = 0

    Returns: a_grid, D_grid (normalized D(1)=1), f_grid = d ln D / d ln a
    """
    H0 = params['H0']
    Omega_m = params['Omega_m']
    Omega_L = 1.0 - Omega_m

    a_grid = np.logspace(np.log10(a_init), 0, n_steps)

    def H_of_a(a):
        return H0 * np.sqrt(Omega_m / a**3 + Omega_L)

    def dH_da(a):
        H = H_of_a(a)
        return H0**2 * (-1.5 * Omega_m / a**4) / (2 * H)

    def Om_of_a(a):
        return Omega_m / a**3 / (Omega_m / a**3 + Omega_L)

    def ode_rhs(a, D, Dp):
        H = H_of_a(a)
        dHda = dH_da(a)
        Om_a = Om_of_a(a)
        mu = mu_mtdf(a, params)

        coeff1 = 3.0 / a + dHda / H
        coeff2 = 1.5 * mu * Om_a / (a * a)

        return Dp, -coeff1 * Dp + coeff2 * D

    # RK4 integration
    D_vals = [a_init]
    Dp_vals = [1.0]

    for i in range(len(a_grid) - 1):
        a = a_grid[i]
        da = a_grid[i + 1] - a
        D, Dp = D_vals[-1], Dp_vals[-1]

        k1_D, k1_Dp = ode_rhs(a, D, Dp)
        k2_D, k2_Dp = ode_rhs(a + da/2, D + da/2*k1_D, Dp + da/2*k1_Dp)
        k3_D, k3_Dp = ode_rhs(a + da/2, D + da/2*k2_D, Dp + da/2*k2_Dp)
        k4_D, k4_Dp = ode_rhs(a + da, D + da*k3_D, Dp + da*k3_Dp)

        D_vals.append(D + da/6 * (k1_D + 2*k2_D + 2*k3_D + k4_D))
        Dp_vals.append(Dp + da/6 * (k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp))

    D_grid = np.array(D_vals)
    Dp_grid = np.array(Dp_vals)

    # Normalize D(a=1) = 1
    D_at_1 = D_grid[-1]
    D_grid /= D_at_1
    Dp_grid /= D_at_1

    # f = (a/D) * dD/da
    f_grid = (a_grid / D_grid) * Dp_grid

    return a_grid, D_grid, f_grid


def fsigma8_shape(z, params):
    """
    Shape factor g(z) = f(a) * D(a) for fsigma8.
    fsigma8(z) = sigma8_0 * g(z)
    """
    a_grid, D_grid, f_grid = solve_growth_ode(params)
    a_obs = 1.0 / (1.0 + np.asarray(z))

    log_a_grid = np.log(a_grid)
    log_a_obs = np.log(a_obs)

    D_obs = np.interp(log_a_obs, log_a_grid, D_grid)
    f_obs = np.interp(log_a_obs, log_a_grid, f_grid)

    return f_obs * D_obs


def fit_sigma8_analytic(z, fsig8_obs, cov, params):
    """
    Analytic best-fit sigma8_0 from fsigma8 data.
    Returns: sigma8_bf, sigma8_err, chi2, dof
    """
    g = fsigma8_shape(z, params)

    try:
        C_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov) / len(z)
        C_inv = np.linalg.inv(cov + reg * np.eye(len(z)))

    gCd = g @ C_inv @ fsig8_obs
    gCg = g @ C_inv @ g

    sigma8_bf = gCd / gCg
    sigma8_err = 1.0 / np.sqrt(gCg)

    residual = fsig8_obs - sigma8_bf * g
    chi2 = float(residual @ C_inv @ residual)
    dof = len(z) - 1

    return sigma8_bf, sigma8_err, chi2, dof


# ============================================================================
# CMB DISTANCE PRIOR
# ============================================================================

def compute_z_star(omegab_h2, omegam_h2):
    """Hu & Sugiyama (1996) recombination redshift."""
    g1 = 0.0783 * omegab_h2**(-0.238) / (1 + 39.5 * omegab_h2**(0.763))
    g2 = 0.560 / (1 + 21.1 * omegab_h2**(1.81))
    return 1048 * (1 + 0.00124 * omegab_h2**(-0.738)) * (1 + g1 * omegam_h2**g2)


def cmb_distance_predictions(params):
    """
    Compute CMB distance prior: [R, lA, omegab_h2].
    Clean baseline using standard Friedmann (no MTDF corrections at high z).
    """
    H0 = params['H0']
    h = params['h']
    omegab_h2 = params['omegab_h2']
    omegam_h2 = params['omegam_h2']
    Omega_b = params['Omega_b']
    Omega_m = params['Omega_m']

    # Radiation density (needed at z ~ 1090)
    Omega_gamma = 2.469e-5 / h**2 * (T_CMB / 2.7)**4
    Omega_r = Omega_gamma * (1 + 0.2271 * N_EFF)
    Omega_L = 1.0 - Omega_m - Omega_r

    # Recombination redshift
    z_star = compute_z_star(omegab_h2, omegam_h2)

    # Hubble distance
    D_H = C_LIGHT / H0  # Mpc

    # Comoving distance to z* (standard Friedmann, including radiation)
    n_steps = 5000
    z_arr = np.linspace(0, z_star, n_steps)
    E_z = np.sqrt(Omega_r * (1 + z_arr)**4 + Omega_m * (1 + z_arr)**3 + Omega_L)
    D_M_star = D_H * np.trapz(1.0 / E_z, z_arr)

    # Sound horizon at recombination
    _, r_s = sound_horizon_aubourg(params)

    # CMB observables
    R = np.sqrt(Omega_m) * D_M_star / D_H
    lA = np.pi * D_M_star / r_s

    return np.array([R, lA, omegab_h2]), {
        'z_star': z_star, 'D_M_star': D_M_star, 'r_s': r_s,
        'R': R, 'lA': lA, 'D_H': D_H,
    }
