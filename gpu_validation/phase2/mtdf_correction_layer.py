# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
MTDF correction layer applied on top of LCDM C_l emulator output.

The correction modifies the LCDM power spectra by:
1. Shifting the sound horizon: r_s_MTDF = r_s_LCDM * (1 + K_RS_EFE * f_kick)
2. Modifying the angular diameter distance via MTDF's H(z) stress correction
3. Rescaling multipoles by the ratio theta_s_MTDF / theta_s_LCDM

These are small perturbative corrections (<0.1% on r_s) that keep us
well within the CosmoPower training domain.
"""

import numpy as np
from scipy.integrate import quad

# MTDF EFE parameters (from workbook)
K_RS_EFE = -0.215
ALPHA = 1.30
KAPPA = 0.00102  # Anchor: approx f_kick/3 = (1-beta_eos)^2/(72*(1+alpha))
BETA_EOS = 0.573
Z_T = 0.74

# f_kick = lambda_MTDF / 24, where lambda_MTDF is the dimensionless coupling
# lambda_MTDF = (1 - beta_eos)^2 / (1 + alpha) — first-principles derivation
# The factor 24 comes from gamma_int = 1/12:
#   1/3: tensor-to-scalar projection  |  1/2: incomplete transition  |  1/2: time-averaging
# Validated against class_mtdf/output/CMTDF_normalization_validation.txt
LAMBDA_MTDF = (1.0 - BETA_EOS)**2 / (1.0 + ALPHA)  # = 0.427^2 / 2.3 = 0.07927
F_KICK = LAMBDA_MTDF / 24.0  # = 0.003303


def sound_horizon_ratio():
    """Compute the ratio r_s_MTDF / r_s_LCDM.

    Returns
    -------
    ratio : float
        r_s_MTDF / r_s_LCDM = 1 + K_RS_EFE * f_kick
    """
    return 1.0 + K_RS_EFE * F_KICK


def H_mtdf(z, H0, Omega_m):
    """MTDF Hubble rate with stress correction."""
    Omega_L = 1.0 - Omega_m
    E = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    stress = KAPPA * ALPHA * z / (1 + z)
    return H0 * E * (1 + stress)


def H_lcdm(z, H0, Omega_m):
    """Standard LCDM Hubble rate."""
    Omega_L = 1.0 - Omega_m
    E = np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    return H0 * E


def angular_diameter_distance(z, H_func, H0, Omega_m):
    """Comoving angular diameter distance D_A(z) = D_C(z)/(1+z)."""
    c_km_s = 299792.458  # km/s

    def integrand(zp):
        return 1.0 / H_func(zp, H0, Omega_m)

    D_C, _ = quad(integrand, 0, z, limit=200)
    D_C *= c_km_s
    return D_C / (1 + z)


def theta_s_ratio(H0=67.36, Omega_m=0.3153, z_star=1089.92, r_s_lcdm=144.43):
    """Compute the angular sound horizon ratio theta_s_MTDF / theta_s_LCDM.

    theta_s = r_s / D_A(z_star)

    Returns
    -------
    ratio : float
        theta_s_MTDF / theta_s_LCDM
    """
    # Sound horizon ratio
    rs_ratio = sound_horizon_ratio()

    # Angular diameter distance ratio
    D_A_lcdm = angular_diameter_distance(z_star, H_lcdm, H0, Omega_m)
    D_A_mtdf = angular_diameter_distance(z_star, H_mtdf, H0, Omega_m)
    da_ratio = D_A_mtdf / D_A_lcdm

    # theta_s ratio = (r_s_MTDF / r_s_LCDM) / (D_A_MTDF / D_A_LCDM)
    return rs_ratio / da_ratio


def apply_mtdf_correction(ells, cl_lcdm, H0=67.36, Omega_m=0.3153):
    """Apply MTDF corrections to an LCDM C_l spectrum.

    The correction rescales the ell axis by theta_s_MTDF/theta_s_LCDM
    and interpolates back to the original ell grid.

    Parameters
    ----------
    ells : array
        Original multipole values.
    cl_lcdm : array
        LCDM power spectrum at those multipoles.
    H0, Omega_m : float
        Cosmological parameters.

    Returns
    -------
    cl_mtdf : array
        MTDF-corrected power spectrum on the original ell grid.
    """
    ratio = theta_s_ratio(H0, Omega_m)

    # MTDF peaks are at ell_MTDF = ell_LCDM / ratio
    # So we need Cl_MTDF(ell) = Cl_LCDM(ell * ratio)
    ells_shifted = ells * ratio

    # Interpolate back to original grid
    cl_mtdf = np.interp(ells_shifted, ells.astype(float), cl_lcdm,
                        left=cl_lcdm[0], right=cl_lcdm[-1])

    # Also apply amplitude correction from modified growth
    # The ISW effect is enhanced at low ell due to MTDF's modified w(z)
    # This is a small effect: delta_Cl/Cl ~ 2*kappa*alpha at low ell
    isw_boost = 1.0 + 2.0 * KAPPA * ALPHA * np.exp(-ells / 30.0)
    cl_mtdf *= isw_boost

    return cl_mtdf


def correction_summary():
    """Print a summary of MTDF correction magnitudes."""
    rs_ratio = sound_horizon_ratio()
    delta_rs = (rs_ratio - 1) * 100

    print(f"MTDF Correction Layer Summary")
    print(f"  K_RS_EFE     = {K_RS_EFE}")
    print(f"  f_kick       = {F_KICK:.6f}")
    print(f"  delta_r_s/r_s = {delta_rs:+.4f}%")
    print(f"  r_s ratio    = {rs_ratio:.6f}")

    ratio = theta_s_ratio()
    delta_theta = (ratio - 1) * 100
    print(f"  theta_s ratio = {ratio:.6f}")
    print(f"  delta_theta_s = {delta_theta:+.4f}%")
    print(f"  Peak shift   ~ {delta_theta:+.3f}% in ell")
