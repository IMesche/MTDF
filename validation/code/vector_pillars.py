# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
vector_pillars.py
Vector pillar infrastructure for MTDF validation with full covariance matrices.

Implements:
- Data loaders for Pantheon+, DESI BAO, CC H(z), DR16 fσ₈
- MTDF model prediction functions
- Generic chi² computation with covariance matrices
- Analytic marginalization over SN absolute magnitude M
"""

import numpy as np
from pathlib import Path
from functools import lru_cache
import re
import math

# Physical constants
C_LIGHT = 299792.458  # km/s
H0_FIDUCIAL = 70.0  # km/s/Mpc for rd scaling

# Sound horizon ratio: r_s(z*) / r_d
# Planck 2018: r_s(z*) ≈ 144.39 Mpc, r_d ≈ 147.09 Mpc
# This ratio is approximately constant for small variations around fiducial values
RS_ZSTAR_RATIO = 0.9819


# =============================================================================
# SOUND HORIZON CALCULATOR (Aubourg et al. 2015)
# =============================================================================

def compute_sound_horizon_from_densities(params, f_ede_correction=None):
    """
    Compute drag-epoch sound horizon r_d and recombination sound horizon r_s
    from physical densities, using the Aubourg et al. 2015 fitting formula.

    This provides a calibrated analytic sound horizon that:
    - Uses physical densities directly (ω_x h²)
    - Matches Planck 2018 to ~0.1% for fiducial parameters
    - Is consistent between BAO (r_d) and CMB (r_s) analyses
    - Has a hook for future EDE/MTDF early-universe corrections

    Reference: Aubourg et al. 2015 (arXiv:1411.1074), Eq. 16

    Args:
        params: dict with keys:
            - omegab_h2: ω_b h² (physical baryon density, BBN + CMB calibrated)
            - omegam_h2: ω_m h² (physical total matter density)
            - omeganuh2: optional ω_ν h² (neutrino density, default 0)
        f_ede_correction: optional callable f(r_d, r_s, params) -> (r_d_corr, r_s_corr)
            for future EDE/MTDF early-universe corrections (not implemented yet)

    Returns:
        tuple: (r_d, r_s) where
            r_d: sound horizon at drag epoch [Mpc] - for BAO
            r_s: sound horizon at photon decoupling z* [Mpc] - for CMB

    Validation against Planck 2018:
        Input: ω_b h² = 0.02237, ω_m h² = 0.1430, ω_ν h² ≈ 0
        Expected: r_d ≈ 147.09 Mpc, r_s ≈ 144.39 Mpc
        (within ~0.1% of Planck 2018 values)
    """
    # Extract physical densities
    omegab_h2 = params.get('omegab_h2', 0.02236)
    omegam_h2 = params.get('omegam_h2', 0.1430)
    omeganuh2 = params.get('omeganuh2', 0.0)

    # Aubourg et al. 2015 (arXiv:1411.1074), Eq. 16:
    # r_d [Mpc] = 55.154 * exp[-72.3 (ω_ν + 0.0006)²] / (ω_m^0.25351 * ω_b^0.12807)
    #
    # This formula is calibrated to reproduce CAMB results for standard ΛCDM
    # with Planck 2015/2018 compatible parameters.

    w_m = omegam_h2
    w_b = omegab_h2
    w_nu = omeganuh2

    # Drag epoch sound horizon (for BAO)
    r_d = 55.154 * math.exp(-72.3 * (w_nu + 0.0006)**2) \
          / (w_m**0.25351 * w_b**0.12807)

    # Convert to r_s at recombination z* using the fixed Planck ratio
    # This ensures BAO (r_d) and CMB (r_s) are built on the same sound horizon physics
    r_s = RS_ZSTAR_RATIO * r_d

    # Apply optional EDE/MTDF correction (hook for future extension)
    if f_ede_correction is not None:
        r_d, r_s = f_ede_correction(r_d, r_s, params)

    return r_d, r_s


# =============================================================================
# DATA LOADERS (cached)
# =============================================================================

@lru_cache(maxsize=1)
def load_pantheonplus(data_dir):
    """
    Load Pantheon+ SNe data.
    Returns: z_cmb, mu_obs, cov_matrix
    """
    data_path = Path(data_dir) / "External" / "pantheonplus" / "Pantheon+SH0ES.dat"
    cov_path = Path(data_dir) / "External" / "pantheonplus" / "Pantheon+SH0ES_STAT+SYS.cov"

    if not data_path.exists() or not cov_path.exists():
        raise FileNotFoundError(f"Pantheon+ data not found at {data_path}")

    # Parse data file
    z_cmb_list = []
    mu_obs_list = []

    with open(data_path, 'r') as f:
        header = f.readline().strip().split()
        # Find column indices
        zcmb_idx = header.index('zCMB') if 'zCMB' in header else header.index('zHD')
        mu_idx = None
        for candidate in ['MU_SH0ES', 'm_b_corr', 'mu']:
            if candidate in header:
                mu_idx = header.index(candidate)
                break

        if mu_idx is None:
            raise ValueError("Cannot find distance modulus column in Pantheon+ data")

        for line in f:
            parts = line.strip().split()
            if len(parts) > max(zcmb_idx, mu_idx):
                try:
                    z = float(parts[zcmb_idx])
                    mu = float(parts[mu_idx])
                    if z > 0 and not np.isnan(mu):
                        z_cmb_list.append(z)
                        mu_obs_list.append(mu)
                except (ValueError, IndexError):
                    continue

    z_cmb = np.array(z_cmb_list)
    mu_obs = np.array(mu_obs_list)
    n = len(z_cmb)

    # Parse covariance matrix
    cov_lines = []
    with open(cov_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                cov_lines.append(line)

    # First line might be dimension
    try:
        dim = int(cov_lines[0])
        cov_lines = cov_lines[1:]
    except ValueError:
        dim = n

    # Parse covariance values
    cov_values = []
    for line in cov_lines:
        vals = [float(x) for x in line.split()]
        cov_values.extend(vals)

    # Reshape to matrix - could be upper/lower triangular or full
    total_vals = len(cov_values)
    if total_vals == n * n:
        cov_matrix = np.array(cov_values).reshape(n, n)
    elif total_vals == n * (n + 1) // 2:
        # Upper triangular
        cov_matrix = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                cov_matrix[i, j] = cov_values[idx]
                cov_matrix[j, i] = cov_values[idx]
                idx += 1
    else:
        # Try to match available dimensions
        actual_n = int(np.sqrt(total_vals))
        if actual_n * actual_n == total_vals:
            cov_matrix = np.array(cov_values).reshape(actual_n, actual_n)
            # Truncate data to match covariance
            z_cmb = z_cmb[:actual_n]
            mu_obs = mu_obs[:actual_n]
        else:
            raise ValueError(f"Cannot parse Pantheon+ covariance: {total_vals} values for {n} data points")

    return z_cmb, mu_obs, cov_matrix


@lru_cache(maxsize=1)
def load_desi_bao(data_dir):
    """
    Load DESI Y1 BAO data.
    Returns: z_eff, obs_vec, obs_types, cov_matrix

    obs_types[i] = 'DV_over_rs', 'DM_over_rs', or 'DH_over_rs'
    """
    mean_path = Path(data_dir) / "External" / "bao_desi" / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_path = Path(data_dir) / "External" / "bao_desi" / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"

    if not mean_path.exists() or not cov_path.exists():
        raise FileNotFoundError(f"DESI BAO data not found at {mean_path}")

    z_eff = []
    obs_vec = []
    obs_types = []

    with open(mean_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                z_eff.append(float(parts[0]))
                obs_vec.append(float(parts[1]))
                obs_types.append(parts[2])

    z_eff = np.array(z_eff)
    obs_vec = np.array(obs_vec)
    n = len(obs_vec)

    # Load covariance
    cov_lines = []
    with open(cov_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                cov_lines.append([float(x) for x in line.split()])

    cov_matrix = np.array(cov_lines)

    return z_eff, obs_vec, obs_types, cov_matrix


@lru_cache(maxsize=1)
def load_cc_hz(data_dir):
    """
    Load Cosmic Chronometer H(z) data.
    Returns: z, H_obs, H_err (diagonal errors for now)
    """
    data_path = Path(data_dir) / "External" / "hz_cc" / "HzTable_MM_BC03.dat"

    if not data_path.exists():
        raise FileNotFoundError(f"CC H(z) data not found at {data_path}")

    z_list = []
    H_list = []
    err_list = []

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            # Handle comma-separated values
            parts = line.replace(',', ' ').split()
            if len(parts) >= 3:
                try:
                    z_list.append(float(parts[0]))
                    H_list.append(float(parts[1]))
                    err_list.append(float(parts[2]))
                except ValueError:
                    continue

    z = np.array(z_list)
    H_obs = np.array(H_list)
    H_err = np.array(err_list)

    # Build diagonal covariance
    cov_matrix = np.diag(H_err**2)

    return z, H_obs, cov_matrix


@lru_cache(maxsize=1)
def load_dr16_fsigma8(data_dir):
    """
    Load DR16 fσ₈ measurements (LRG + QSO combined).
    Returns: z_eff, fsig8_obs, cov_matrix (for fσ₈ components only)

    The full covariance is 9×9 for LRG (3 z-bins × [DM, DH, fσ₈])
    and 3×3 for QSO. We extract the fσ₈ submatrix.
    """
    lrg_path = Path(data_dir) / "External" / "growth_fsig8" / "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat"
    qso_path = Path(data_dir) / "External" / "growth_fsig8" / "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat"
    lrg_cov_path = Path(data_dir) / "External" / "growth_fsig8" / "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt"
    qso_cov_path = Path(data_dir) / "External" / "growth_fsig8" / "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt"

    z_eff = []
    fsig8_obs = []
    fsig8_indices = []  # Track which rows in full cov are fσ₈

    def parse_dat(path):
        """Parse data file, return (z, values, types) for all rows."""
        if not path.exists():
            return [], [], []
        z_list, val_list, type_list = [], [], []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    z_list.append(float(parts[0]))
                    val_list.append(float(parts[1]))
                    type_list.append(parts[2])
        return z_list, val_list, type_list

    def load_cov(path):
        """Load covariance matrix from file."""
        if not path.exists():
            return None
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    rows.append([float(x) for x in line.split()])
        return np.array(rows) if rows else None

    # Parse LRG data
    z_lrg, val_lrg, types_lrg = parse_dat(lrg_path)
    lrg_fsig_idx = [i for i, t in enumerate(types_lrg) if 'f_sigma8' in t]
    for i in lrg_fsig_idx:
        z_eff.append(z_lrg[i])
        fsig8_obs.append(val_lrg[i])
        fsig8_indices.append(i)

    # Parse QSO data
    z_qso, val_qso, types_qso = parse_dat(qso_path)
    qso_fsig_idx = [i for i, t in enumerate(types_qso) if 'f_sigma8' in t]
    for i in qso_fsig_idx:
        z_eff.append(z_qso[i])
        fsig8_obs.append(val_qso[i])

    z_eff = np.array(z_eff)
    fsig8_obs = np.array(fsig8_obs)
    n_lrg_fsig = len(lrg_fsig_idx)
    n_qso_fsig = len(qso_fsig_idx)
    n_total = n_lrg_fsig + n_qso_fsig

    # Build covariance by extracting fσ₈ subblocks
    cov_matrix = np.zeros((n_total, n_total))

    # LRG fσ₈ submatrix (indices 2, 5, 8 in the 9×9 matrix)
    lrg_cov = load_cov(lrg_cov_path)
    if lrg_cov is not None and len(lrg_fsig_idx) > 0:
        for i, idx_i in enumerate(lrg_fsig_idx):
            for j, idx_j in enumerate(lrg_fsig_idx):
                if idx_i < lrg_cov.shape[0] and idx_j < lrg_cov.shape[1]:
                    cov_matrix[i, j] = lrg_cov[idx_i, idx_j]

    # QSO fσ₈ submatrix (index 2 in the 3×3 matrix)
    qso_cov = load_cov(qso_cov_path)
    if qso_cov is not None and len(qso_fsig_idx) > 0:
        for i, idx_i in enumerate(qso_fsig_idx):
            for j, idx_j in enumerate(qso_fsig_idx):
                if idx_i < qso_cov.shape[0] and idx_j < qso_cov.shape[1]:
                    cov_matrix[n_lrg_fsig + i, n_lrg_fsig + j] = qso_cov[idx_i, idx_j]

    return z_eff, fsig8_obs, cov_matrix


# =============================================================================
# MTDF MODEL FUNCTIONS
# =============================================================================

def mtdf_H_of_z(z, params):
    """
    MTDF Hubble parameter H(z).

    In MTDF, the expansion rate includes stress-tensor corrections:
    H(z) = H0 * sqrt(Ω_m(1+z)³ + Ω_Λ) * (1 + κ * stress_correction(z))

    For now, using simplified ΛCDM backbone with MTDF modification.
    """
    H0 = params.get('H0', 70.0)
    Omega_m = params.get('Omega_m', 0.3)
    kappa = params.get('kappa', 1.02e-3)  # Anchor: approx f_kick/3 = (1-beta_eos)^2/(72*(1+alpha))
    alpha = params.get('alpha', 1.30)

    # ΛCDM backbone
    Omega_L = 1.0 - Omega_m
    Ez = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

    # MTDF stress correction (small at low z)
    stress_corr = kappa * alpha * z / (1 + z)

    return H0 * Ez * (1 + stress_corr)


def mtdf_comoving_distance(z, params):
    """
    Comoving distance D_C(z) via numerical integration (trapezoidal rule).
    D_C = c * ∫₀ᶻ dz'/H(z')
    """
    def integrate_to_z(z_max, n_steps=500):
        if z_max <= 0:
            return 0.0
        z_arr = np.linspace(0, z_max, n_steps)
        integrand = C_LIGHT / np.array([mtdf_H_of_z(zp, params) for zp in z_arr])
        return np.trapz(integrand, z_arr)

    if np.isscalar(z):
        return integrate_to_z(z)
    else:
        return np.array([integrate_to_z(zi) for zi in z])


def mtdf_luminosity_distance(z, params):
    """
    Luminosity distance D_L(z) = (1+z) * D_C(z)
    """
    D_C = mtdf_comoving_distance(z, params)
    return (1 + z) * D_C


def mtdf_mu_vector(z, params):
    """
    Distance modulus μ(z) = 5 * log10(D_L/10pc)
    Returns μ for array of redshifts.

    For SNe Ia, the absolute magnitude M is marginalized analytically.
    """
    D_L = mtdf_luminosity_distance(z, params)  # in Mpc
    # μ = 5*log10(D_L) + 25 (D_L in Mpc → 10pc)
    mu = 5 * np.log10(D_L) + 25
    return mu


def mtdf_bao_vector(z_eff, obs_types, params):
    """
    BAO observables: D_V/r_d, D_M/r_d, D_H/r_d

    D_V = [D_M² * c*z/H(z)]^(1/3)  (volume-averaged distance)
    D_M = D_C (comoving distance, flat universe)
    D_H = c/H(z)
    r_d = sound horizon at drag epoch from Aubourg et al. 2015 formula

    Uses compute_sound_horizon_from_densities() to ensure BAO and CMB
    share a consistent microphysical sound horizon calculation.
    """
    # Compute r_d from physical densities using calibrated Aubourg formula
    # This replaces the previous hardcoded r_d = 147.09 Mpc
    r_d, _ = compute_sound_horizon_from_densities(params)

    predictions = []
    for z, obs_type in zip(z_eff, obs_types):
        H_z = mtdf_H_of_z(z, params)
        D_M = mtdf_comoving_distance(z, params)
        D_H = C_LIGHT / H_z
        D_V = (D_M**2 * C_LIGHT * z / H_z)**(1/3)

        if 'DV' in obs_type:
            predictions.append(D_V / r_d)
        elif 'DM' in obs_type:
            predictions.append(D_M / r_d)
        elif 'DH' in obs_type:
            predictions.append(D_H / r_d)
        else:
            predictions.append(np.nan)

    return np.array(predictions)


def mtdf_Hz_vector(z, params):
    """
    H(z) predictions for cosmic chronometer data.
    """
    return np.array([mtdf_H_of_z(zi, params) for zi in z])


def mu_mtdf(a, params):
    """
    MTDF effective gravitational coupling μ(a).

    This modifies the Poisson equation for structure growth:
    ∇²Φ = 4πG μ(a) ρ_m δ

    The functional form uses a smooth transition:
        μ(a) = 1 + amp × T(a)
    where:
        T(a) = x^α / (1 + x^α)  with x = a/a_t
        amp = (1 - β_eos)² / (1 + α)

    Args:
        a: scale factor
        params: dict with alpha, beta_eos, z_t

    Returns:
        μ(a) - dimensionless effective coupling
    """
    alpha = params.get("alpha", 1.3)
    beta_eos = params.get("beta_eos", 0.573)
    z_t = params.get("z_t", 0.74)

    a_t = 1.0 / (1.0 + z_t)
    x = a / a_t

    # Safe power for early times
    if x <= 0:
        return 1.0
    x_pow = x ** alpha
    T = x_pow / (1.0 + x_pow)

    amp = (1.0 - beta_eos) ** 2 / (1.0 + alpha)
    mu = 1.0 + amp * T
    return mu


def mu_lcdm(a, params):
    """
    ΛCDM effective gravitational coupling μ(a) = 1 (no modification).
    """
    return 1.0


def solve_growth_ode(params, mu_func, a_grid=None, a_init=1e-3):
    """
    Solve the linear growth ODE with arbitrary μ(a) modification:

    D''(a) + (3/a + H'(a)/H(a)) D'(a) - (3/2) μ(a) Ω_m(a) D(a) / a² = 0

    Initial conditions (matter era):
        D(a_init) = a_init
        D'(a_init) = 1

    Returns D(a) normalized so D(a=1) = 1.

    Args:
        params: dict with H0, Omega_m (and alpha, beta_eos, z_t for MTDF)
        mu_func: function μ(a, params) - gravitational coupling
        a_grid: array of scale factors (default: logspace from a_init to 1)
        a_init: initial scale factor for integration

    Returns:
        a_grid, D_grid, f_grid (growth rate f = d ln D / d ln a)
    """
    H0 = params.get('H0', 70.0)
    Omega_m = params.get('Omega_m', 0.3)
    Omega_L = 1.0 - Omega_m

    if a_grid is None:
        a_grid = np.logspace(np.log10(a_init), 0, 500)  # a_init to 1

    def H_of_a(a):
        """Hubble parameter H(a) in km/s/Mpc"""
        return H0 * np.sqrt(Omega_m / a**3 + Omega_L)

    def dH_da(a):
        """dH/da"""
        H = H_of_a(a)
        return H0**2 * (-1.5 * Omega_m / a**4) / (2 * H)

    def Omega_m_of_a(a):
        """Matter density parameter Ω_m(a)"""
        return Omega_m / a**3 / (Omega_m / a**3 + Omega_L)

    def growth_ode(a, y):
        """ODE system for [D, D']."""
        D, Dp = y

        H = H_of_a(a)
        dHda = dH_da(a)
        Om_a = Omega_m_of_a(a)
        mu = mu_func(a, params)

        coeff1 = 3.0 / a + dHda / H
        coeff2 = 1.5 * mu * Om_a / (a * a)

        dD_da = Dp
        dDp_da = -coeff1 * Dp + coeff2 * D

        return [dD_da, dDp_da]

    # Initial conditions in matter era
    D_init = a_init
    Dp_init = 1.0

    # Integrate using simple RK4
    D_vals = [D_init]
    Dp_vals = [Dp_init]

    for i in range(len(a_grid) - 1):
        a = a_grid[i]
        da = a_grid[i + 1] - a
        y = [D_vals[-1], Dp_vals[-1]]

        # RK4 step
        k1 = growth_ode(a, y)
        k2 = growth_ode(a + da/2, [y[0] + da/2*k1[0], y[1] + da/2*k1[1]])
        k3 = growth_ode(a + da/2, [y[0] + da/2*k2[0], y[1] + da/2*k2[1]])
        k4 = growth_ode(a + da, [y[0] + da*k3[0], y[1] + da*k3[1]])

        D_new = y[0] + da/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        Dp_new = y[1] + da/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

        D_vals.append(D_new)
        Dp_vals.append(Dp_new)

    D_grid = np.array(D_vals)
    Dp_grid = np.array(Dp_vals)

    # Normalize so D(a=1) = 1
    D_at_1 = D_grid[-1]
    D_grid = D_grid / D_at_1
    Dp_grid = Dp_grid / D_at_1

    # Compute growth rate f = d ln D / d ln a = (a/D) * dD/da
    f_grid = (a_grid / D_grid) * Dp_grid

    return a_grid, D_grid, f_grid


def solve_growth_ode_mtdf(params, a_grid=None, a_init=1e-3):
    """
    Solve the linear growth ODE with MTDF μ(a) modification.
    Wrapper around solve_growth_ode() using mu_mtdf.
    """
    return solve_growth_ode(params, mu_mtdf, a_grid, a_init)


def solve_growth_ode_lcdm(params, a_grid=None, a_init=1e-3):
    """
    Solve the linear growth ODE for flat ΛCDM (μ = 1).
    Wrapper around solve_growth_ode() using mu_lcdm.
    """
    return solve_growth_ode(params, mu_lcdm, a_grid, a_init)


def mtdf_fsigma8_shape(z, params, return_diagnostics=False):
    """
    Compute the fσ₈ "shape" factor g(z) = f(a) × D(a).

    This is the part of fσ₈ that depends on MTDF growth physics,
    independent of σ₈,₀ normalization:
        fσ₈(z) = σ₈,₀ × g(z)

    Args:
        z: array of redshifts
        params: dict with H0, Omega_m, alpha, beta_eos, z_t
        return_diagnostics: if True, return additional diagnostic info

    Returns:
        g(z) = f(a) × D(a) shape factors (and optionally diagnostics dict)
    """
    Omega_m = params.get('Omega_m', 0.3)

    # Solve the growth ODE
    a_init = 1e-3
    a_grid, D_grid, f_grid = solve_growth_ode_mtdf(params, a_init=a_init)

    # Compute mu over the grid for diagnostics
    mu_grid = np.array([mu_mtdf(a, params) for a in a_grid])

    # Interpolate to requested redshifts
    a_obs = 1.0 / (1.0 + np.array(z))

    # Linear interpolation in log(a)
    log_a_grid = np.log(a_grid)
    log_a_obs = np.log(a_obs)

    D_obs = np.interp(log_a_obs, log_a_grid, D_grid)
    f_obs = np.interp(log_a_obs, log_a_grid, f_grid)

    # Compute Ω_m(a) at observation points
    Omega_L = 1.0 - Omega_m
    Omega_m_obs = Omega_m / a_obs**3 / (Omega_m / a_obs**3 + Omega_L)

    # Compute μ at observation points
    mu_obs = np.array([mu_mtdf(a, params) for a in a_obs])

    # Shape factor g(z) = f(a) × D(a)
    # fσ₈(z) = σ₈,₀ × f(a) × D(a) = σ₈,₀ × g(z)
    g_shape = f_obs * D_obs

    if return_diagnostics:
        diagnostics = {
            'z': np.array(z),
            'a': a_obs,
            'mu_mtdf': mu_obs,
            'Omega_m_a': Omega_m_obs,
            'D': D_obs,
            'f': f_obs,
            'g_shape': g_shape,
            'mu_min': mu_grid.min(),
            'mu_max': mu_grid.max(),
            'a_grid': a_grid,
            'D_grid': D_grid,
            'f_grid': f_grid,
            'mu_grid': mu_grid,
        }
        return g_shape, diagnostics

    return g_shape


def fit_sigma8_analytic(z, fsig8_obs, cov_matrix, params):
    """
    Analytically fit σ₈,₀ to fσ₈ data using full covariance.

    For the linear model fσ₈(z) = σ₈,₀ × g(z):
        σ₈,₀ = (gᵀ C⁻¹ d) / (gᵀ C⁻¹ g)

    Also returns the χ² at the best-fit value.

    Args:
        z: array of redshifts
        fsig8_obs: observed fσ₈ values
        cov_matrix: covariance matrix
        params: MTDF parameters (excluding sigma8)

    Returns:
        sigma8_bf: best-fit σ₈,₀
        sigma8_err: uncertainty on σ₈,₀ (from Fisher matrix)
        chi2: χ² at best-fit
        dof: degrees of freedom (n - 1 for fitted σ₈,₀)
    """
    # Get shape factor g(z) = f(a) × D(a)
    g_shape = mtdf_fsigma8_shape(z, params)

    # Invert covariance
    try:
        C_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov_matrix) / len(z)
        C_inv = np.linalg.inv(cov_matrix + reg * np.eye(len(z)))

    # Analytic solution: σ₈,₀ = (gᵀ C⁻¹ d) / (gᵀ C⁻¹ g)
    gCd = g_shape @ C_inv @ fsig8_obs
    gCg = g_shape @ C_inv @ g_shape

    sigma8_bf = gCd / gCg

    # Uncertainty: σ(σ₈,₀) = 1 / √(gᵀ C⁻¹ g)
    sigma8_err = 1.0 / np.sqrt(gCg)

    # χ² at best-fit: (d - σ₈,₀ × g)ᵀ C⁻¹ (d - σ₈,₀ × g)
    residual = fsig8_obs - sigma8_bf * g_shape
    chi2 = float(residual @ C_inv @ residual)

    # DOF: n_data - 1 (for fitted σ₈,₀)
    dof = len(z) - 1

    return sigma8_bf, sigma8_err, chi2, dof


def mtdf_fsigma8_vector(z, params, return_diagnostics=False, fit_sigma8=False,
                         fsig8_obs=None, cov_matrix=None):
    """
    fσ₈(z) using proper MTDF growth ODE solution.

    Solves the linear growth equation with μ_MTDF(a) modification,
    then computes:
        f(a) = d ln D / d ln a
        σ₈(z) = σ₈_0 * D(a)
        fσ₈(z) = f(a) * σ₈(z)

    Args:
        z: array of redshifts
        params: dict with H0, Omega_m, alpha, beta_eos, z_t, sigma8
        return_diagnostics: if True, return additional diagnostic info
        fit_sigma8: if True, fit σ₈,₀ to data (requires fsig8_obs and cov_matrix)
        fsig8_obs: observed fσ₈ values (required if fit_sigma8=True)
        cov_matrix: covariance matrix (required if fit_sigma8=True)

    Returns:
        fσ₈ predictions (and optionally diagnostics dict)
    """
    Omega_m = params.get('Omega_m', 0.3)

    # Handle σ₈,₀ fitting
    if fit_sigma8 and fsig8_obs is not None and cov_matrix is not None:
        sigma8_bf, sigma8_err, chi2_bf, dof = fit_sigma8_analytic(
            z, fsig8_obs, cov_matrix, params
        )
        sigma8_0 = sigma8_bf
        sigma8_fitted = True
    else:
        sigma8_0 = params.get('sigma8', 0.811)
        sigma8_bf = sigma8_0
        sigma8_err = None
        chi2_bf = None
        sigma8_fitted = False

    # Solve the growth ODE
    a_init = 1e-3
    a_grid, D_grid, f_grid = solve_growth_ode_mtdf(params, a_init=a_init)

    # Compute mu over the grid for diagnostics
    mu_grid = np.array([mu_mtdf(a, params) for a in a_grid])

    # Interpolate to requested redshifts
    a_obs = 1.0 / (1.0 + np.array(z))

    # Linear interpolation in log(a)
    log_a_grid = np.log(a_grid)
    log_a_obs = np.log(a_obs)

    D_obs = np.interp(log_a_obs, log_a_grid, D_grid)
    f_obs = np.interp(log_a_obs, log_a_grid, f_grid)

    # Compute Ω_m(a) at observation points
    Omega_L = 1.0 - Omega_m
    Omega_m_obs = Omega_m / a_obs**3 / (Omega_m / a_obs**3 + Omega_L)

    # Compute μ at observation points
    mu_obs = np.array([mu_mtdf(a, params) for a in a_obs])

    # σ₈(z) = σ₈_0 * D(a)
    sigma8_z = sigma8_0 * D_obs

    # fσ₈(z) = f(a) * σ₈(z)
    fsigma8 = f_obs * sigma8_z

    if return_diagnostics:
        diagnostics = {
            'z': np.array(z),
            'a': a_obs,
            'mu_mtdf': mu_obs,
            'Omega_m_a': Omega_m_obs,
            'D': D_obs,
            'f': f_obs,
            'sigma8_z': sigma8_z,
            'fsigma8': fsigma8,
            'mu_min': mu_grid.min(),
            'mu_max': mu_grid.max(),
            'a_grid': a_grid,
            'D_grid': D_grid,
            'f_grid': f_grid,
            'mu_grid': mu_grid,
            # σ₈,₀ fitting results
            'sigma8_0': sigma8_0,
            'sigma8_fitted': sigma8_fitted,
            'sigma8_bf': sigma8_bf,
            'sigma8_err': sigma8_err,
            'chi2_fitted': chi2_bf,
        }
        return fsigma8, diagnostics

    return fsigma8


def chi2_fsigma8_fitted(z, fsig8_obs, cov_matrix, params):
    """
    Compute χ² for fσ₈ with σ₈,₀ fitted analytically.

    Returns chi2, dof, and fitting diagnostics.
    """
    sigma8_bf, sigma8_err, chi2, dof = fit_sigma8_analytic(
        z, fsig8_obs, cov_matrix, params
    )

    return chi2, dof, {
        'sigma8_bf': sigma8_bf,
        'sigma8_err': sigma8_err,
    }


# =============================================================================
# ΛCDM GROWTH MODEL
# =============================================================================

def lcdm_fsigma8_shape(z, params, return_diagnostics=False):
    """
    Compute the fσ₈ "shape" factor g(z) = f(a) × D(a) for ΛCDM.
    """
    Omega_m = params.get('Omega_m', 0.3)

    # Solve the growth ODE with μ = 1
    a_init = 1e-3
    a_grid, D_grid, f_grid = solve_growth_ode_lcdm(params, a_init=a_init)

    # Interpolate to requested redshifts
    a_obs = 1.0 / (1.0 + np.array(z))

    log_a_grid = np.log(a_grid)
    log_a_obs = np.log(a_obs)

    D_obs = np.interp(log_a_obs, log_a_grid, D_grid)
    f_obs = np.interp(log_a_obs, log_a_grid, f_grid)

    # Compute Ω_m(a) at observation points
    Omega_L = 1.0 - Omega_m
    Omega_m_obs = Omega_m / a_obs**3 / (Omega_m / a_obs**3 + Omega_L)

    # Shape factor g(z) = f(a) × D(a)
    g_shape = f_obs * D_obs

    if return_diagnostics:
        diagnostics = {
            'z': np.array(z),
            'a': a_obs,
            'mu_lcdm': np.ones_like(a_obs),  # μ = 1 for ΛCDM
            'Omega_m_a': Omega_m_obs,
            'D': D_obs,
            'f': f_obs,
            'g_shape': g_shape,
            'a_grid': a_grid,
            'D_grid': D_grid,
            'f_grid': f_grid,
        }
        return g_shape, diagnostics

    return g_shape


def fit_sigma8_lcdm(z, fsig8_obs, cov_matrix, params):
    """
    Analytically fit σ₈,₀ to fσ₈ data for ΛCDM using full covariance.
    """
    # Get shape factor g(z) = f(a) × D(a)
    g_shape = lcdm_fsigma8_shape(z, params)

    # Invert covariance
    try:
        C_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov_matrix) / len(z)
        C_inv = np.linalg.inv(cov_matrix + reg * np.eye(len(z)))

    # Analytic solution: σ₈,₀ = (gᵀ C⁻¹ d) / (gᵀ C⁻¹ g)
    gCd = g_shape @ C_inv @ fsig8_obs
    gCg = g_shape @ C_inv @ g_shape

    sigma8_bf = gCd / gCg
    sigma8_err = 1.0 / np.sqrt(gCg)

    # χ² at best-fit
    residual = fsig8_obs - sigma8_bf * g_shape
    chi2 = float(residual @ C_inv @ residual)

    # DOF: n_data - 1
    dof = len(z) - 1

    return sigma8_bf, sigma8_err, chi2, dof


def lcdm_fsigma8_vector(z, params, return_diagnostics=False, fit_sigma8=False,
                         fsig8_obs=None, cov_matrix=None):
    """
    fσ₈(z) using flat ΛCDM growth ODE solution (μ = 1).
    """
    Omega_m = params.get('Omega_m', 0.3)

    # Handle σ₈,₀ fitting
    if fit_sigma8 and fsig8_obs is not None and cov_matrix is not None:
        sigma8_bf, sigma8_err, chi2_bf, dof = fit_sigma8_lcdm(
            z, fsig8_obs, cov_matrix, params
        )
        sigma8_0 = sigma8_bf
        sigma8_fitted = True
    else:
        sigma8_0 = params.get('sigma8', 0.811)
        sigma8_bf = sigma8_0
        sigma8_err = None
        chi2_bf = None
        sigma8_fitted = False

    # Solve the growth ODE
    a_init = 1e-3
    a_grid, D_grid, f_grid = solve_growth_ode_lcdm(params, a_init=a_init)

    # Interpolate to requested redshifts
    a_obs = 1.0 / (1.0 + np.array(z))

    log_a_grid = np.log(a_grid)
    log_a_obs = np.log(a_obs)

    D_obs = np.interp(log_a_obs, log_a_grid, D_grid)
    f_obs = np.interp(log_a_obs, log_a_grid, f_grid)

    # Compute Ω_m(a) at observation points
    Omega_L = 1.0 - Omega_m
    Omega_m_obs = Omega_m / a_obs**3 / (Omega_m / a_obs**3 + Omega_L)

    # σ₈(z) = σ₈_0 * D(a)
    sigma8_z = sigma8_0 * D_obs

    # fσ₈(z) = f(a) * σ₈(z)
    fsigma8 = f_obs * sigma8_z

    if return_diagnostics:
        diagnostics = {
            'z': np.array(z),
            'a': a_obs,
            'mu_lcdm': np.ones_like(a_obs),
            'Omega_m_a': Omega_m_obs,
            'D': D_obs,
            'f': f_obs,
            'sigma8_z': sigma8_z,
            'fsigma8': fsigma8,
            'a_grid': a_grid,
            'D_grid': D_grid,
            'f_grid': f_grid,
            # σ₈,₀ fitting results
            'sigma8_0': sigma8_0,
            'sigma8_fitted': sigma8_fitted,
            'sigma8_bf': sigma8_bf,
            'sigma8_err': sigma8_err,
            'chi2_fitted': chi2_bf,
        }
        return fsigma8, diagnostics

    return fsigma8


# =============================================================================
# CMB DISTANCE PRIOR
# =============================================================================

def E_high_z(z, params):
    """
    High-redshift E(z) = H(z)/H0 using physical densities only.

    At z >> 1, dark energy and late-time stress effects are negligible,
    so we use a pure matter + radiation model. This ensures the sound
    horizon integral depends only on early-universe physics.

    Args:
        z: redshift (scalar or array)
        params: dict with H0, omegab_h2, omegam_h2

    Returns:
        E(z) = H(z)/H0
    """
    H0 = params.get('H0', 70.0)
    h = params.get('h', H0 / 100.0)

    # Primary physical densities
    omegam_h2 = params.get('omegam_h2', 0.1430)

    # Photon density from T_CMB
    T_cmb = params.get('T_cmb', 2.7255)
    omegag_h2 = 2.469e-5 * (T_cmb / 2.7)**4

    # Include neutrinos in radiation: omega_r = omega_gamma * (1 + 0.2271 * N_eff)
    N_eff = params.get('N_eff', 3.046)
    omegar_h2 = omegag_h2 * (1.0 + 0.2271 * N_eff)

    # E^2(z) = (omega_m (1+z)^3 + omega_r (1+z)^4) / h^2
    # Note: No dark energy or MTDF corrections at high z
    z = np.asarray(z)
    one_plus_z = 1.0 + z
    E_sq = (omegam_h2 * one_plus_z**3 + omegar_h2 * one_plus_z**4) / h**2

    return np.sqrt(E_sq)


def compute_z_star(omegab_h2, omegam_h2):
    """
    Compute recombination redshift z* using Hu & Sugiyama (1996) fitting formula.

    This depends only on physical baryon and matter densities,
    not on MTDF-specific tuning.

    Args:
        omegab_h2: Physical baryon density (Omega_b * h^2)
        omegam_h2: Physical matter density (Omega_m * h^2)

    Returns:
        z_star: Recombination redshift

    Reference: Hu & Sugiyama (1996), ApJ 471, 542
               Planck 2018 VI uses z* = 1089.92 for their fiducial cosmology
    """
    # Fitting formula coefficients
    g1 = 0.0783 * omegab_h2**(-0.238) / (1 + 39.5 * omegab_h2**(0.763))
    g2 = 0.560 / (1 + 21.1 * omegab_h2**(1.81))

    z_star = 1048 * (1 + 0.00124 * omegab_h2**(-0.738)) * (1 + g1 * omegam_h2**g2)

    return z_star


def compute_sound_horizon(params, z_drag=1059.94, z_max=1e5, n_steps=2000):
    """
    Compute the comoving sound horizon at drag epoch r_s(z_drag).

    Uses numerical integration with EARLY-UNIVERSE physics only.
    This is consistent with compute_sound_horizon_at_zstar() to ensure
    CMB and BAO use the same sound horizon physics.

    IMPORTANT: Uses E_high_z() which includes only matter + radiation,
    ignoring dark energy and MTDF late-time corrections.

    Args:
        params: dict with H0, omegab_h2, omegam_h2
        z_drag: drag epoch redshift (default 1059.94)
        z_max: upper integration limit
        n_steps: number of integration steps

    Returns:
        r_s in Mpc
    """
    H0 = params.get('H0', 70.0)
    h = params.get('h', H0 / 100.0)

    # Primary physical densities (external constraints)
    omegab_h2 = params.get('omegab_h2', 0.02236)

    # Derive Omega_b from physical density
    Omega_b = params.get('Omega_b', omegab_h2 / h**2)

    # Photon density
    T_cmb = params.get('T_cmb', 2.7255)
    Omega_gamma = 2.469e-5 / h**2 * (T_cmb / 2.7)**4

    # Baryon-to-photon ratio prefactor: R_b(z) = R_b_prefactor / (1+z)
    R_b_prefactor = 3.0 * Omega_b / (4.0 * Omega_gamma)

    # Set up z grid from z_drag to z_max
    z_grid = np.linspace(z_drag, z_max, n_steps)

    # Use E_high_z for pure matter + radiation (no DE, no MTDF corrections)
    E_grid = E_high_z(z_grid, params)

    # Sound speed: c_s = c / sqrt(3(1 + R_b))
    # R_b(z) = R_b_prefactor / (1+z)
    R_b_grid = R_b_prefactor / (1.0 + z_grid)
    c_s_over_c = 1.0 / np.sqrt(3.0 * (1.0 + R_b_grid))

    # Integrand: c_s / H(z) = (c_s/c) * c / (H0 * E(z))
    integrand = c_s_over_c / E_grid

    # Integrate
    r_s_over_DH = np.trapz(integrand, z_grid)

    # Convert to Mpc: r_s = D_H * integral, where D_H = c/H0
    D_H = C_LIGHT / H0
    r_s = D_H * r_s_over_DH

    return r_s


def compute_angular_diameter_distance(z, params):
    """
    Angular diameter distance D_A(z) = D_C(z) / (1+z) for flat universe.
    Returns D_A in Mpc.
    """
    D_C = mtdf_comoving_distance(z, params)
    return D_C / (1.0 + z)


def compute_sound_horizon_at_zstar(params, z_star, z_max=1e5, n_steps=2000):
    """
    Compute the comoving sound horizon at recombination r_s(z*).

    Uses numerical integration with EARLY-UNIVERSE physics only:
    r_s(z*) = integral_{z*}^{inf} c_s(z) / H(z) dz

    where c_s(z) = c / sqrt(3(1 + R_b(z))) is the sound speed
    and R_b(z) = 3*Omega_b / (4*Omega_gamma) / (1+z) is the baryon-to-photon ratio.

    IMPORTANT: Uses E_high_z() which includes only matter + radiation,
    ignoring dark energy and MTDF late-time corrections. This ensures
    r_s depends only on {omegab_h2, omegam_h2, T_CMB, N_eff, H0}.

    Args:
        params: dict with H0, omegab_h2, omegam_h2
        z_star: recombination redshift
        z_max: upper integration limit (effectively infinity)
        n_steps: number of integration steps

    Returns:
        r_s: sound horizon at z* in Mpc
    """
    H0 = params.get('H0', 70.0)
    h = params.get('h', H0 / 100.0)

    # Primary physical densities (external constraints)
    omegab_h2 = params.get('omegab_h2', 0.02236)

    # Derive Omega_b from physical density
    Omega_b = params.get('Omega_b', omegab_h2 / h**2)

    # Photon density
    T_cmb = params.get('T_cmb', 2.7255)
    Omega_gamma = 2.469e-5 / h**2 * (T_cmb / 2.7)**4

    # Baryon-to-photon ratio prefactor: R_b(z) = R_b_prefactor / (1+z)
    R_b_prefactor = 3.0 * Omega_b / (4.0 * Omega_gamma)

    # Set up z grid from z_star to z_max
    z_grid = np.linspace(z_star, z_max, n_steps)

    # Use E_high_z for pure matter + radiation (no DE, no MTDF corrections)
    E_grid = E_high_z(z_grid, params)

    # Sound speed: c_s = c / sqrt(3(1 + R_b))
    # R_b(z) = R_b_prefactor / (1+z)
    R_b_grid = R_b_prefactor / (1.0 + z_grid)
    c_s_over_c = 1.0 / np.sqrt(3.0 * (1.0 + R_b_grid))

    # Integrand: c_s / H(z) = (c_s/c) * c / (H0 * E(z))
    # r_s = (c/H0) * integral(c_s_over_c / E dz)
    integrand = c_s_over_c / E_grid

    # Integrate
    r_s_over_DH = np.trapz(integrand, z_grid)

    # Convert to Mpc: r_s = D_H * integral, where D_H = c/H0
    D_H = C_LIGHT / H0  # Hubble distance in Mpc
    r_s = D_H * r_s_over_DH

    return r_s


def mtdf_cmb_distance_vector(params, return_diagnostics=False):
    """
    Compute predictions for Planck 2018 CMB distance prior.

    This is a CLEAN BASELINE implementation using:
    - Standard Friedmann H(z) with Omega_m, Omega_r, Omega_L (NO MTDF corrections)
    - Aubourg et al. 2015 sound horizon formula
    - Fixed r_s/r_d ratio for recombination vs drag epoch

    Returns predictions for [R, lA, omegab_h2] in the same order as
    planck2018_distance_means.txt in cmb_planck2018/.

    The Planck 2018 distance prior parameters are:
    - R: Shift parameter = sqrt(Omega_m) * D_M(z*) / D_H
         where D_M is comoving distance, D_H = c/H0
    - lA: Acoustic scale = pi * D_M(z*) / r_s(z*)
         where r_s = RS_ZSTAR_RATIO * r_d from Aubourg formula
    - omegab_h2: Physical baryon density (passed through, NOT recomputed)

    PARAMETER HIERARCHY:
        Primary inputs: omegab_h2, omegam_h2 (from workbook)
        Chosen value: H0
        Derived: h = H0/100, Omega_b = omegab_h2/h², Omega_m = omegam_h2/h²

    SOUND HORIZON:
        BAO and CMB use the SAME Aubourg et al. 2015 r_d formula.
        CMB uses r_s = 0.9819 * r_d (fixed Planck 2018 ratio).

    NOTE: Future experimental CMB pillar will test MTDF early field energy.
    This baseline shows the genuine H0 tension without any modifications.

    Args:
        params: dict with H0, omegab_h2, omegam_h2
        return_diagnostics: if True, return additional diagnostic info

    Returns:
        Array [R, lA, omegab_h2] and optionally diagnostics
    """
    H0 = params.get('H0', 70.0)
    h = params.get('h', H0 / 100.0)

    # PRIMARY: Physical densities (external microphysics constraints)
    # These are fixed by BBN + CMB, NOT derived from density parameters
    omegab_h2 = params.get('omegab_h2', 0.02236)  # Planck 2018 BBN/CMB value
    omegam_h2 = params.get('omegam_h2', 0.1430)   # Physical matter density

    # DERIVED: Density parameters from physical densities and H0
    Omega_b = params.get('Omega_b', omegab_h2 / h**2)
    Omega_m = params.get('Omega_m', omegam_h2 / h**2)

    # Compute recombination redshift using Hu & Sugiyama fitting formula
    # Note: This uses the PRIMARY omegab_h2 and omegam_h2, not derived values
    z_star = compute_z_star(omegab_h2, omegam_h2)

    # Radiation density (needed for high-z distance calculation)
    T_cmb = 2.7255  # K
    Omega_gamma = 2.469e-5 / h**2 * (T_cmb / 2.7)**4
    Omega_r = Omega_gamma * (1 + 0.2271 * 3.046)  # Include neutrinos
    Omega_L = 1.0 - Omega_m - Omega_r  # Consistent closure

    # Hubble distance D_H = c/H0 in Mpc
    D_H = C_LIGHT / H0

    # Compute comoving distance to z* INCLUDING RADIATION
    # This is critical for CMB distances at z~1000
    #
    # NOTE: This is a CLEAN BASELINE using standard Friedmann H(z) only.
    # No MTDF late-time corrections are applied here.
    # Future experimental CMB pillar will test MTDF early field energy.
    n_steps = 5000
    z_arr = np.linspace(0, z_star, n_steps)

    # Standard Friedmann E(z) = H(z)/H0 with matter, radiation, and cosmological constant
    # E(z) = sqrt(Omega_r*(1+z)^4 + Omega_m*(1+z)^3 + Omega_L)
    E_z = np.sqrt(Omega_r * (1+z_arr)**4 + Omega_m * (1+z_arr)**3 + Omega_L)

    # Integrate: D_M = D_H * integral(1/E(z) dz)
    integrand = 1.0 / E_z
    D_M_star = D_H * np.trapz(integrand, z_arr)

    # Compute sound horizon at recombination using calibrated Aubourg formula
    # This ensures consistency with BAO analyses that use r_d from the same formula
    r_d, r_s_star = compute_sound_horizon_from_densities(params)

    # R: Shift parameter
    R = np.sqrt(Omega_m) * D_M_star / D_H

    # lA: Acoustic scale (uses r_s at z*, not r_d at drag epoch)
    lA = np.pi * D_M_star / r_s_star

    prediction = np.array([R, lA, omegab_h2])

    if return_diagnostics:
        diagnostics = {
            'z_star': z_star,
            'r_s_star': r_s_star,
            'r_d': r_d,  # Drag epoch sound horizon (for BAO)
            'D_M_star': D_M_star,
            'D_H': D_H,
            'R': R,
            'lA': lA,
            # Primary parameters (external microphysics)
            'omegab_h2': omegab_h2,  # PRIMARY: from workbook, NOT computed
            'omegam_h2': omegam_h2,  # PRIMARY: from workbook
            # Derived parameters
            'Omega_b': Omega_b,      # DERIVED: omegab_h2 / h^2
            'Omega_m': Omega_m,      # DERIVED: omegam_h2 / h^2
            'Omega_r': Omega_r,
            # MTDF choice
            'H0': H0,
            'h': h,
        }
        return prediction, diagnostics

    return prediction


def debug_cmb_geometry(params, obs_means=None):
    """
    Print detailed CMB distance prior diagnostics.

    Computes and displays all intermediate quantities for debugging
    the CMB pillar calculation.

    Args:
        params: dict with H0, omegab_h2, omegam_h2, and MTDF parameters
        obs_means: optional array [R_obs, lA_obs, omegab_h2_obs] for comparison

    Reference values (Planck 2018 best-fit):
        z_star ≈ 1089.92
        r_s(z*) ≈ 144.4 Mpc
        D_M(z*) ≈ 13.9 Gpc
        lA ≈ 301.47
        R ≈ 1.7502
    """
    H0 = params.get('H0', 70.0)
    h = params.get('h', H0 / 100.0)
    omegab_h2 = params.get('omegab_h2', 0.02236)
    omegam_h2 = params.get('omegam_h2', 0.1430)

    # Derived
    Omega_b = params.get('Omega_b', omegab_h2 / h**2)
    Omega_m = params.get('Omega_m', omegam_h2 / h**2)

    # Compute z_star
    z_star = compute_z_star(omegab_h2, omegam_h2)

    # Compute D_M(z*) using the CMB distance calculation
    pred, diag = mtdf_cmb_distance_vector(params, return_diagnostics=True)
    D_M = diag['D_M_star']
    r_s = diag['r_s_star']
    r_d = diag.get('r_d', r_s / 0.9819)  # Drag epoch sound horizon for BAO
    R_pred = diag['R']
    lA_pred = diag['lA']

    # Photon and radiation densities
    T_cmb = params.get('T_cmb', 2.7255)
    omegag_h2 = 2.469e-5 * (T_cmb / 2.7)**4
    N_eff = params.get('N_eff', 3.046)
    omegar_h2 = omegag_h2 * (1.0 + 0.2271 * N_eff)

    print("=" * 60)
    print("CMB DISTANCE PRIOR DIAGNOSTICS (MTDF)")
    print("=" * 60)
    print()
    print("PRIMARY INPUTS (external microphysics):")
    print(f"  omegab_h2   = {omegab_h2:.5f}")
    print(f"  omegam_h2   = {omegam_h2:.5f}")
    print(f"  omegar_h2   = {omegar_h2:.6f}  (from T_CMB={T_cmb}, N_eff={N_eff})")
    print()
    print("MTDF CHOICE:")
    print(f"  H0          = {H0:.3f} km/s/Mpc")
    print(f"  h           = {h:.4f}")
    print()
    print("DERIVED PARAMETERS:")
    print(f"  Omega_b     = {Omega_b:.5f}  (= omegab_h2 / h²)")
    print(f"  Omega_m     = {Omega_m:.5f}  (= omegam_h2 / h²)")
    print()
    print("COMPUTED QUANTITIES:")
    print(f"  z_star      = {z_star:.2f}")
    print(f"  D_M(z*)     = {D_M:.2f} Mpc  = {D_M/1000:.3f} Gpc")
    print(f"  r_s(z*)     = {r_s:.3f} Mpc  (sound horizon at recombination)")
    print(f"  r_d         = {r_d:.3f} Mpc  (sound horizon at drag epoch, for BAO)")
    print()
    print("CMB OBSERVABLES:")
    print(f"  R           = {R_pred:.5f}  (= sqrt(Omega_m) * D_M / D_H)")
    print(f"  lA          = {lA_pred:.3f}  (= pi * D_M / r_s)")
    print(f"  omegab_h2   = {omegab_h2:.5f}  (by construction)")

    if obs_means is not None:
        R_obs, lA_obs, omegab_obs = obs_means
        # Compute sigmas from standard Planck values
        sigma_R = 0.0046
        sigma_lA = 0.090
        sigma_omegab = 0.00015

        print()
        print("COMPARISON WITH PLANCK 2018:")
        print(f"  R:       {R_pred:.5f} vs {R_obs:.5f}  (Δ = {R_pred - R_obs:+.5f}, z = {(R_pred - R_obs)/sigma_R:+.2f}σ)")
        print(f"  lA:      {lA_pred:.3f} vs {lA_obs:.3f}  (Δ = {lA_pred - lA_obs:+.3f}, z = {(lA_pred - lA_obs)/sigma_lA:+.2f}σ)")
        print(f"  omegab:  {omegab_h2:.5f} vs {omegab_obs:.5f}  (Δ = {omegab_h2 - omegab_obs:+.5f}, z = {(omegab_h2 - omegab_obs)/sigma_omegab:+.2f}σ)")
        print()
        print("FRACTIONAL DIFFERENCES:")
        print(f"  ΔR / R_obs     = {(R_pred - R_obs)/R_obs:+.4f}  ({(R_pred - R_obs)/R_obs * 100:+.2f}%)")
        print(f"  ΔlA / lA_obs   = {(lA_pred - lA_obs)/lA_obs:+.4f}  ({(lA_pred - lA_obs)/lA_obs * 100:+.2f}%)")

    print()
    print("REFERENCE (Planck 2018 best-fit cosmology):")
    print("  z_star ≈ 1089.92")
    print("  r_s(z*) ≈ 144.39 Mpc  (recombination)")
    print("  r_d     ≈ 147.09 Mpc  (drag epoch)")
    print("  D_M(z*) ≈ 13.9 Gpc")
    print("  lA ≈ 301.47")
    print("  R ≈ 1.7502")
    print("=" * 60)

    return {
        'z_star': z_star,
        'D_M': D_M,
        'r_s': r_s,
        'r_d': r_d,
        'R': R_pred,
        'lA': lA_pred,
        'omegab_h2': omegab_h2,
    }


def load_cmb_distance_prior(data_dir):
    """
    Load Planck 2018 CMB distance prior data.

    Looks for files in data/External/cmb_planck2018/ first (new format),
    falls back to data/External/cmb_distance/ (old format) if not found.

    Returns:
        means: array of [R, lA, omegab_h2]
        cov: 3x3 covariance matrix
    """
    # Try new location first
    new_means_path = Path(data_dir) / "External" / "cmb_planck2018" / "planck2018_distance_means.txt"
    new_cov_path = Path(data_dir) / "External" / "cmb_planck2018" / "planck2018_distance_cov.txt"

    # Fall back to old location
    old_means_path = Path(data_dir) / "External" / "cmb_distance" / "planck2018_distance_means.txt"
    old_cov_path = Path(data_dir) / "External" / "cmb_distance" / "planck2018_distance_cov.txt"

    if new_means_path.exists():
        means_path = new_means_path
        cov_path = new_cov_path
    elif old_means_path.exists():
        means_path = old_means_path
        cov_path = old_cov_path
    else:
        raise FileNotFoundError(f"CMB distance prior data not found at {new_means_path} or {old_means_path}")

    # Parse means file
    means = []
    with open(means_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                means.append(float(parts[1]))

    means = np.array(means)

    # Parse covariance file
    cov_rows = []
    with open(cov_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            cov_rows.append([float(x) for x in line.split()])

    cov = np.array(cov_rows)

    return means, cov


# =============================================================================
# CHI-SQUARED COMPUTATION
# =============================================================================

def chi2_vector_pillar(data, model_pred, cov_matrix, nuisance_count=0):
    """
    Compute χ² for a vector pillar: χ² = (d-m)ᵀ C⁻¹ (d-m)

    Args:
        data: observed data vector
        model_pred: model prediction vector
        cov_matrix: covariance matrix
        nuisance_count: number of marginalized parameters (for DOF)

    Returns:
        chi2: chi-squared value
        dof: degrees of freedom (n_data - nuisance_count)
    """
    residual = data - model_pred

    # Regularize covariance if needed
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add small regularization
        reg = 1e-10 * np.trace(cov_matrix) / len(data)
        cov_inv = np.linalg.inv(cov_matrix + reg * np.eye(len(data)))

    chi2 = float(residual @ cov_inv @ residual)
    dof = len(data) - nuisance_count

    return chi2, dof


def chi2_sne_marginalized(z, mu_obs, cov_matrix, params):
    """
    χ² for SNe with analytic marginalization over absolute magnitude M.

    The distance modulus observed is: μ_obs = m_B - M
    Model prediction: μ_model(z) = 5*log10(D_L(z)) + 25

    Marginalizing over M analytically:
    χ² = (Δμ - ΔM·1)ᵀ C⁻¹ (Δμ - ΔM·1)

    where ΔM = (1ᵀ C⁻¹ Δμ) / (1ᵀ C⁻¹ 1) is the ML estimate of M offset.
    """
    mu_model = mtdf_mu_vector(z, params)
    delta_mu = mu_obs - mu_model

    # Invert covariance
    try:
        C_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        reg = 1e-10 * np.trace(cov_matrix) / len(z)
        C_inv = np.linalg.inv(cov_matrix + reg * np.eye(len(z)))

    ones = np.ones(len(z))

    # Analytic marginalization
    A = ones @ C_inv @ ones
    B = ones @ C_inv @ delta_mu
    C = delta_mu @ C_inv @ delta_mu

    # χ² with M marginalized out
    chi2 = C - B**2 / A

    # DOF: n_data - 1 (for marginalized M)
    dof = len(z) - 1

    return float(chi2), dof


# =============================================================================
# VECTOR PILLAR EVALUATION
# =============================================================================

def evaluate_vector_pillar(pillar_config, params, data_dir):
    """
    Evaluate a vector pillar and return chi², DOF, z-score equivalent.

    Args:
        pillar_config: dict with keys:
            - pillar_id: e.g., 'P12_SNe_Vector'
            - data_file: path relative to data_dir
            - model_function: 'mtdf_mu_vector', 'mtdf_bao_vector', etc.
            - nuisance_count: number of nuisance parameters
        params: MTDF parameters dict
        data_dir: base data directory

    Returns:
        dict with chi2, dof, chi2_red, z_equiv, n_data
    """
    pillar_id = pillar_config.get('pillar_id', 'Unknown')
    model_func = pillar_config.get('model_function', '')

    result = {
        'pillar_id': pillar_id,
        'chi2': np.nan,
        'dof': 0,
        'chi2_red': np.nan,
        'z_equiv': np.nan,
        'n_data': 0,
    }

    try:
        if 'sne' in model_func.lower() or 'mu' in model_func.lower():
            z, mu_obs, cov = load_pantheonplus(data_dir)
            chi2, dof = chi2_sne_marginalized(z, mu_obs, cov, params)
            result['n_data'] = len(z)

        elif 'bao' in model_func.lower():
            z_eff, obs_vec, obs_types, cov = load_desi_bao(data_dir)
            model_pred = mtdf_bao_vector(z_eff, obs_types, params)
            chi2, dof = chi2_vector_pillar(obs_vec, model_pred, cov)
            result['n_data'] = len(obs_vec)

        elif 'hz' in model_func.lower():
            z, H_obs, cov = load_cc_hz(data_dir)
            model_pred = mtdf_Hz_vector(z, params)
            chi2, dof = chi2_vector_pillar(H_obs, model_pred, cov)
            result['n_data'] = len(z)

        elif 'fsigma8' in model_func.lower() or 'growth' in model_func.lower():
            z, fsig8_obs, cov = load_dr16_fsigma8(data_dir)
            model_pred = mtdf_fsigma8_vector(z, params)
            chi2, dof = chi2_vector_pillar(fsig8_obs, model_pred, cov)
            result['n_data'] = len(z)

        else:
            raise ValueError(f"Unknown model function: {model_func}")

        result['chi2'] = chi2
        result['dof'] = dof
        result['chi2_red'] = chi2 / dof if dof > 0 else np.nan

        # z-equivalent: (χ² - dof) / sqrt(2*dof) for large dof
        if dof > 0:
            result['z_equiv'] = (chi2 - dof) / np.sqrt(2 * dof)

    except Exception as e:
        print(f"[VECTOR_PILLAR] Error evaluating {pillar_id}: {e}")

    return result


def get_vector_pillar_configs():
    """
    Return default configurations for vector pillars.
    These can be overridden by workbook entries.
    """
    return [
        {
            'pillar_id': 'P12_SNe_Vector',
            'pillar_name': 'Type Ia Supernovae (Pantheon+)',
            'model_function': 'mtdf_mu_vector',
            'nuisance_count': 1,  # M marginalized
        },
        {
            'pillar_id': 'P5_BAO_Vector',
            'pillar_name': 'Baryon Acoustic Oscillations (DESI)',
            'model_function': 'mtdf_bao_vector',
            'nuisance_count': 0,
        },
        {
            'pillar_id': 'P_Hz_Vector',
            'pillar_name': 'Hubble Parameter H(z) (CC)',
            'model_function': 'mtdf_hz_vector',
            'nuisance_count': 0,
        },
        {
            'pillar_id': 'P13_Growth_Vector',
            'pillar_name': 'Structure Growth fσ₈ (DR16)',
            'model_function': 'mtdf_fsigma8_vector',
            'nuisance_count': 0,
        },
    ]
