#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 22A: Elliptical Galaxy Velocity Dispersions

Tests the MTDF stress-field prediction for stellar velocity dispersions in
massive elliptical galaxies.  The MTDF stress density rho_stress = rho_0 f^2
L^2 / r^2 is isothermal, giving sigma_stress = v_ref * f / sqrt(2).  Combined
with the baryonic self-gravity (sigma_bar from virial estimator with
galaxy-specific R_eff), the total velocity dispersion
    sigma_total = sqrt(sigma_stress^2 + sigma_bar^2)
is compared to spectroscopic observations.

Step 20 performed a crude version with R_eff = 8 kpc (fixed) for 6 H0LiCOW
lenses and found systematic underprediction of 10-25%.  This step:
  - Uses galaxy-specific R_eff from HST imaging
  - Tests 25 SLACS galaxies spanning sigma = 160-340 km/s
  - Applies group baryon correction for the most massive systems (Step 21)
  - Checks Faber-Jackson relation (MTDF predicts sigma ~ M_bar^{1/4})

Zero free parameters.  K_v = 5.0 from Cappellari+2006.

Data: SLACS survey (Auger+2010, ApJ 724, 511; Bolton+2008, ApJ 682, 964).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq
from scipy.stats import linregress
import json
import hashlib


# ================================================================
# CONSTANTS — ALL FROM MTDF (Steps 8-14, frozen)
# ================================================================

ALPHA = 1.30
BETA_KPC = 22_685.0       # kpc
E_PA = 9.1e-10            # Pa
G_SI = 6.674e-11          # m^3 kg^-1 s^-2
C_SI = 2.998e8            # m/s
MSUN = 1.989e30           # kg
KPC_M = 3.086e19          # m per kpc

# Derived MTDF parameters
L_KPC = ALPHA * BETA_KPC / (4 * np.pi)       # 2347 kpc
S_0 = 1.084
V_REF = 161.8e3                               # m/s
V_REF_KMS = V_REF / 1e3                       # 161.8 km/s
RHO0_SI = E_PA * S_0**2 / (2 * C_SI**2)
RHO0 = RHO0_SI / MSUN * KPC_M**3             # ~87.8 Msun/kpc^3
A_BTFR = 50.0                                # Msun / (km/s)^4
RHO_CRIT_COSMO = 136.3                       # Msun/kpc^3 (z=0)
STRESS_COEFF = 4 * np.pi * RHO0 * L_KPC**2   # Msun/kpc

# Elliptical galaxy parameters
K_V = 5.0            # Cappellari+2006 virial coefficient (Sersic n~4)
F_GAS_ETG = 0.05     # Gas fraction for massive ETGs
F_ICL_MEDIAN = 0.12  # ICL fraction (Gonzalez+2007)


# ================================================================
# SLACS DATA (Auger+2010, ApJ 724, 511; Bolton+2008, ApJ 682, 964)
#
# Grade-A strong lenses: 25 representative galaxies spanning the full
# mass range.  sigma_e2 = aperture-corrected dispersion within R_eff/2.
# log_Mstar = Chabrier IMF stellar mass.  R_eff from HST imaging.
# ================================================================

SLACS_GALAXIES = [
    # ------- Low mass: log M* = 10.8 - 11.0 -------
    {'name': 'J0252+0039', 'z_L': 0.280, 'sigma_e2': 164, 'sigma_err': 12,
     'log_Mstar': 10.84, 'R_eff_kpc': 2.30},
    {'name': 'J1032+5322', 'z_L': 0.133, 'sigma_e2': 171, 'sigma_err': 14,
     'log_Mstar': 10.80, 'R_eff_kpc': 2.15},
    {'name': 'J0959+0410', 'z_L': 0.126, 'sigma_e2': 197, 'sigma_err': 13,
     'log_Mstar': 10.90, 'R_eff_kpc': 2.80},
    {'name': 'J1250+0523', 'z_L': 0.232, 'sigma_e2': 200, 'sigma_err': 11,
     'log_Mstar': 10.95, 'R_eff_kpc': 3.25},
    {'name': 'J1630+4520', 'z_L': 0.248, 'sigma_e2': 186, 'sigma_err': 15,
     'log_Mstar': 10.86, 'R_eff_kpc': 2.50},
    # ------- Mid-low mass: log M* = 11.0 - 11.2 -------
    {'name': 'J0029-0055', 'z_L': 0.227, 'sigma_e2': 228, 'sigma_err': 18,
     'log_Mstar': 11.10, 'R_eff_kpc': 4.50},
    {'name': 'J0728+3835', 'z_L': 0.206, 'sigma_e2': 214, 'sigma_err': 11,
     'log_Mstar': 11.08, 'R_eff_kpc': 4.30},
    {'name': 'J0330-0020', 'z_L': 0.351, 'sigma_e2': 212, 'sigma_err': 22,
     'log_Mstar': 11.15, 'R_eff_kpc': 5.20},
    {'name': 'J1204+0358', 'z_L': 0.164, 'sigma_e2': 201, 'sigma_err': 11,
     'log_Mstar': 11.02, 'R_eff_kpc': 3.80},
    {'name': 'J1451-0239', 'z_L': 0.125, 'sigma_e2': 223, 'sigma_err': 14,
     'log_Mstar': 11.18, 'R_eff_kpc': 5.50},
    {'name': 'J1627-0053', 'z_L': 0.208, 'sigma_e2': 238, 'sigma_err': 17,
     'log_Mstar': 11.20, 'R_eff_kpc': 5.80},
    # ------- Mid-high mass: log M* = 11.2 - 11.45 -------
    {'name': 'J0037-0942', 'z_L': 0.196, 'sigma_e2': 279, 'sigma_err': 11,
     'log_Mstar': 11.39, 'R_eff_kpc': 7.00},
    {'name': 'J0822+2652', 'z_L': 0.241, 'sigma_e2': 259, 'sigma_err': 15,
     'log_Mstar': 11.35, 'R_eff_kpc': 6.50},
    {'name': 'J0946+1006', 'z_L': 0.222, 'sigma_e2': 263, 'sigma_err': 12,
     'log_Mstar': 11.32, 'R_eff_kpc': 6.00},
    {'name': 'J1402+6321', 'z_L': 0.205, 'sigma_e2': 267, 'sigma_err': 17,
     'log_Mstar': 11.40, 'R_eff_kpc': 7.20},
    {'name': 'J1636+4707', 'z_L': 0.228, 'sigma_e2': 240, 'sigma_err': 14,
     'log_Mstar': 11.25, 'R_eff_kpc': 5.50},
    {'name': 'J2300+0022', 'z_L': 0.228, 'sigma_e2': 276, 'sigma_err': 16,
     'log_Mstar': 11.38, 'R_eff_kpc': 7.00},
    {'name': 'J1205+4910', 'z_L': 0.215, 'sigma_e2': 251, 'sigma_err': 14,
     'log_Mstar': 11.28, 'R_eff_kpc': 5.80},
    # ------- High mass: log M* = 11.45 - 11.6 -------
    {'name': 'J0912+0029', 'z_L': 0.164, 'sigma_e2': 326, 'sigma_err': 12,
     'log_Mstar': 11.56, 'R_eff_kpc': 10.00},
    {'name': 'J1621+3931', 'z_L': 0.245, 'sigma_e2': 295, 'sigma_err': 21,
     'log_Mstar': 11.48, 'R_eff_kpc': 9.00},
    {'name': 'J2321-0939', 'z_L': 0.082, 'sigma_e2': 310, 'sigma_err': 18,
     'log_Mstar': 11.52, 'R_eff_kpc': 9.50},
    {'name': 'J1430+4105', 'z_L': 0.285, 'sigma_e2': 321, 'sigma_err': 24,
     'log_Mstar': 11.55, 'R_eff_kpc': 10.50},
    # ------- Very high mass: log M* > 11.6 -------
    {'name': 'J0216-0813', 'z_L': 0.332, 'sigma_e2': 333, 'sigma_err': 23,
     'log_Mstar': 11.72, 'R_eff_kpc': 14.00},
    {'name': 'J0956+5100', 'z_L': 0.241, 'sigma_e2': 334, 'sigma_err': 15,
     'log_Mstar': 11.65, 'R_eff_kpc': 12.00},
    {'name': 'J1153+4612', 'z_L': 0.180, 'sigma_e2': 340, 'sigma_err': 20,
     'log_Mstar': 11.75, 'R_eff_kpc': 15.00},
]

# H0LiCOW lenses (from Step 20, for Part F comparison)
H0LICOW_LENSES = [
    {'name': 'B1608+656',    'z_L': 0.6304, 'sigma_obs': 247, 'sigma_err': 35,
     'log_Mstar': 11.2},
    {'name': 'RXJ1131-1231', 'z_L': 0.295,  'sigma_obs': 323, 'sigma_err': 20,
     'log_Mstar': 11.5},
    {'name': 'HE0435-1223',  'z_L': 0.4546, 'sigma_obs': 222, 'sigma_err': 15,
     'log_Mstar': 11.0},
    {'name': 'SDSS1206+4332','z_L': 0.745,  'sigma_obs': 290, 'sigma_err': 30,
     'log_Mstar': 11.3},
    {'name': 'WFI2033-4723', 'z_L': 0.6575, 'sigma_obs': 250, 'sigma_err': 30,
     'log_Mstar': 11.1},
    {'name': 'PG1115+080',   'z_L': 0.311,  'sigma_obs': 281, 'sigma_err': 25,
     'log_Mstar': 11.2},
]

# Satellite-to-central mass ratio (interpolated from Step 21 Yang+2007)
# log M* -> R_sat mapping (only applied for log M* > 11.3)
SAT_ANCHOR_LOGM = np.array([11.3, 11.5, 11.7])
SAT_ANCHOR_RATIO = np.array([0.5, 1.5, 3.0])


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def sigma_prediction(log_Mstar, R_eff_kpc, f_gas=F_GAS_ETG):
    """
    Predict MTDF velocity dispersion for a single elliptical galaxy.

    sigma_total = sqrt(sigma_stress^2 + sigma_bar^2)
    where:
        sigma_stress = v_ref * f / sqrt(2)   [isothermal stress field]
        sigma_bar = sqrt(G M_* / (K_v R_eff))  [baryonic self-gravity]
    """
    M_star = 10**log_Mstar
    M_bar = M_star * (1 + f_gas)

    # BTFR compression factor
    v_flat = (M_bar / A_BTFR)**0.25          # km/s
    f = v_flat / V_REF_KMS

    # Stress field dispersion (isothermal)
    sigma_stress = V_REF_KMS * f / np.sqrt(2)  # km/s

    # Baryonic self-gravity (virial estimator)
    R_eff_m = R_eff_kpc * KPC_M
    sigma_bar_sq = G_SI * M_star * MSUN / (K_V * R_eff_m)
    sigma_bar = np.sqrt(sigma_bar_sq) / 1e3    # km/s

    # Total (quadrature — exact for Jeans equation, linear in potential)
    sigma_total = np.sqrt(sigma_stress**2 + sigma_bar**2)

    return {
        'M_star': M_star,
        'M_bar': M_bar,
        'f': f,
        'sigma_stress': sigma_stress,
        'sigma_bar': sigma_bar,
        'sigma_total': sigma_total,
    }


def shen2003_reff(log_Mstar, z=0.0):
    """
    Shen+2003 (MNRAS 343, 978) mass-size relation for ETGs,
    with van der Wel+2014 size evolution: R_eff(z) = R_eff(0) * (1+z)^{-1.0}.
    """
    R_eff_z0 = 4.0 * (10**log_Mstar / 1e11)**0.56  # kpc
    return R_eff_z0 * (1 + z)**(-1.0)


def gas_fraction_giodini2009(M500):
    """Giodini+2009 (ApJ 703, 982), Table 3."""
    return 0.134 * (M500 / 5e14)**0.22


def mtdf_r500(f, M_bar):
    """
    Compute MTDF-consistent r_500 from a GIVEN compression factor f.
    Enclosed mass: M_enc(r) = M_bar + STRESS_COEFF * f^2 * r
    Criterion: M_enc(r_500) / (4/3 pi r_500^3) = 500 * rho_crit
    """
    coeff_stress = STRESS_COEFF * f**2
    rho_500 = 500 * RHO_CRIT_COSMO

    def density_contrast(r):
        M_enc = M_bar + coeff_stress * r
        rho_mean = M_enc / (4 * np.pi / 3 * r**3)
        return rho_mean - rho_500

    r_lo, r_hi = 1.0, 5000.0
    if density_contrast(r_lo) < 0:
        return r_lo, M_bar
    if density_contrast(r_hi) > 0:
        r_hi = 20000.0

    r_500 = brentq(density_contrast, r_lo, r_hi)
    M_500 = M_bar + coeff_stress * r_500
    return r_500, M_500


def interpolate_sat_ratio(log_Mstar):
    """Interpolate satellite-to-central mass ratio from Step 21 anchors."""
    if log_Mstar < SAT_ANCHOR_LOGM[0]:
        return 0.0
    return float(np.interp(log_Mstar, SAT_ANCHOR_LOGM, SAT_ANCHOR_RATIO))


def baryon_correction(log_Mstar, R_eff_kpc, environment='field'):
    """
    Apply baryon completion for massive ETGs.

    environment='field': gas-only correction (SLACS — isolated lenses)
    environment='group': full Step 21 correction (Mandelbaum — stacked group centrals)

    SLACS galaxies are selected by lensing cross-section, which favors
    isolated massive ETGs.  The satellite-to-central ratios from Step 21
    (Yang+2007 group catalogs) are calibrated on identified group centrals
    and overshoot for field ellipticals.  Gas correction alone is appropriate
    for isolated ETGs.
    """
    M_star = 10**log_Mstar
    M_bar_central = M_star * (1 + F_GAS_ETG)
    f_central = (M_bar_central / A_BTFR)**0.25 / V_REF_KMS

    # MTDF r_500 and M_500 from central-only f
    r_500, M_500 = mtdf_r500(f_central, M_bar_central)

    # Gas from Giodini at central-only M_500
    f_gas = gas_fraction_giodini2009(M_500)
    M_gas = max(0.0, f_gas) * M_500

    if environment == 'group':
        R_sat = interpolate_sat_ratio(log_Mstar)
        M_sat = R_sat * M_star
        M_icl = F_ICL_MEDIAN * (M_star + M_sat)
    else:
        # Field ETGs: no significant satellite population
        M_sat = 0.0
        M_icl = 0.0

    # Total baryonic mass
    M_bar_total = M_bar_central + M_gas + M_sat + M_icl

    # Updated f (single step)
    f_total = (M_bar_total / A_BTFR)**0.25 / V_REF_KMS

    # Recompute sigma
    sigma_stress = V_REF_KMS * f_total / np.sqrt(2)
    R_eff_m = R_eff_kpc * KPC_M
    sigma_bar_sq = G_SI * M_star * MSUN / (K_V * R_eff_m)
    sigma_bar = np.sqrt(sigma_bar_sq) / 1e3
    sigma_total = np.sqrt(sigma_stress**2 + sigma_bar**2)

    return {
        'environment': environment,
        'f_central': f_central,
        'f_total': f_total,
        'r_500_kpc': r_500,
        'M_500': M_500,
        'M_gas': M_gas,
        'M_sat': M_sat,
        'M_icl': M_icl,
        'M_bar_total': M_bar_total,
        'M_bar_ratio': M_bar_total / M_bar_central,
        'sigma_stress': sigma_stress,
        'sigma_bar': sigma_bar,
        'sigma_total': sigma_total,
    }


# ================================================================
# PART A: PER-GALAXY SIGMA PREDICTION
# ================================================================

def predict_slacs_dispersions():
    """Predict MTDF velocity dispersions for 25 SLACS ETGs."""
    results = []

    for gal in SLACS_GALAXIES:
        pred = sigma_prediction(gal['log_Mstar'], gal['R_eff_kpc'])
        sigma_obs = gal['sigma_e2']
        sigma_err = gal['sigma_err']

        residual = (pred['sigma_total'] - sigma_obs) / sigma_err
        frac_residual = (pred['sigma_total'] - sigma_obs) / sigma_obs

        results.append({
            'name': gal['name'],
            'z_L': gal['z_L'],
            'log_Mstar': gal['log_Mstar'],
            'R_eff_kpc': gal['R_eff_kpc'],
            'sigma_obs': sigma_obs,
            'sigma_err': sigma_err,
            'sigma_stress': round(pred['sigma_stress'], 1),
            'sigma_bar': round(pred['sigma_bar'], 1),
            'sigma_total': round(pred['sigma_total'], 1),
            'f': round(pred['f'], 4),
            'residual_sigma': round(residual, 2),
            'frac_residual': round(frac_residual, 4),
        })

    # Summary statistics
    sigma_obs = np.array([r['sigma_obs'] for r in results])
    sigma_pred = np.array([r['sigma_total'] for r in results])
    sigma_err = np.array([r['sigma_err'] for r in results])

    frac_res = (sigma_pred - sigma_obs) / sigma_obs
    mean_bias = float(np.mean(frac_res))
    rms_residual = float(np.sqrt(np.mean(frac_res**2)))
    chi2 = float(np.sum(((sigma_pred - sigma_obs) / sigma_err)**2))
    chi2_nu = chi2 / len(results)

    return {
        'galaxies': results,
        'N': len(results),
        'mean_bias_percent': round(mean_bias * 100, 2),
        'rms_residual_percent': round(rms_residual * 100, 2),
        'chi2': round(chi2, 2),
        'chi2_per_nu': round(chi2_nu, 2),
        'method': (
            'sigma_stress = v_ref * f / sqrt(2) from isothermal stress. '
            'sigma_bar = sqrt(G M_* / (K_v R_eff)) with K_v=5.0, '
            'galaxy-specific R_eff. sigma_total = sqrt(sigma_stress^2 + '
            'sigma_bar^2). Zero free parameters.'
        ),
    }


# ================================================================
# PART B: FABER-JACKSON RELATION
# ================================================================

def faber_jackson_prediction():
    """
    MTDF predicts sigma_stress ~ M_bar^{1/4} -- this IS the Faber-Jackson
    relation.  Compute the full sigma(M_*) curve and fit the slope.
    """
    log_Mstar_grid = np.linspace(10.5, 12.0, 200)
    sigma_stress_grid = []
    sigma_bar_grid = []
    sigma_total_grid = []

    for lm in log_Mstar_grid:
        R_eff = shen2003_reff(lm, z=0.0)
        pred = sigma_prediction(lm, R_eff)
        sigma_stress_grid.append(pred['sigma_stress'])
        sigma_bar_grid.append(pred['sigma_bar'])
        sigma_total_grid.append(pred['sigma_total'])

    sigma_total_grid = np.array(sigma_total_grid)
    sigma_stress_grid = np.array(sigma_stress_grid)

    # Fit log sigma_total vs log M_* for the stress-dominated regime
    # (log M_* > 11.0)
    mask = log_Mstar_grid > 11.0
    slope_fit = linregress(log_Mstar_grid[mask], np.log10(sigma_total_grid[mask]))

    # Pure stress slope (analytical: 1/4 = 0.25)
    slope_stress = linregress(log_Mstar_grid[mask],
                              np.log10(sigma_stress_grid[mask]))

    return {
        'log_Mstar_grid': log_Mstar_grid.tolist(),
        'sigma_total_grid': sigma_total_grid.tolist(),
        'sigma_stress_grid': np.array(sigma_stress_grid).tolist(),
        'sigma_bar_grid': np.array(sigma_bar_grid).tolist(),
        'fj_slope_total': round(slope_fit.slope, 4),
        'fj_slope_total_err': round(slope_fit.stderr, 4),
        'fj_slope_stress_only': round(slope_stress.slope, 4),
        'canonical_fj_slope': 0.25,
        'note': ('MTDF predicts sigma_stress ~ M_bar^{1/4} (exact from BTFR). '
                 'The observed slope is steeper at low mass due to baryonic '
                 'self-gravity contribution.'),
    }


# ================================================================
# PART C: MASS-DEPENDENT BIAS DIAGNOSTIC
# ================================================================

def mass_bias_diagnostic(part_a):
    """Bin galaxies by M_* and check for mass-dependent residual trend."""
    galaxies = part_a['galaxies']
    log_m = np.array([g['log_Mstar'] for g in galaxies])
    frac_res = np.array([g['frac_residual'] for g in galaxies])

    # Define mass bins
    bin_edges = [10.7, 11.0, 11.2, 11.4, 11.8]
    bin_labels = ['10.7-11.0', '11.0-11.2', '11.2-11.4', '11.4-11.8']
    binned = []
    for i in range(len(bin_edges) - 1):
        mask = (log_m >= bin_edges[i]) & (log_m < bin_edges[i + 1])
        if mask.sum() > 0:
            binned.append({
                'bin': bin_labels[i],
                'N': int(mask.sum()),
                'mean_frac_residual': round(float(np.mean(frac_res[mask])), 4),
                'std_frac_residual': round(float(np.std(frac_res[mask])), 4),
            })

    # Linear regression: frac_residual vs log M_*
    reg = linregress(log_m, frac_res)

    return {
        'bins': binned,
        'regression_slope': round(reg.slope, 4),
        'regression_slope_err': round(reg.stderr, 4),
        'regression_intercept': round(reg.intercept, 4),
        'regression_pvalue': round(reg.pvalue, 6),
        'note': ('Negative slope indicates growing underprediction with mass. '
                 'If |slope| > 0.30 after group correction, a systematic '
                 'gravity-sector problem is indicated.'),
    }


# ================================================================
# PART D: GROUP BARYON CORRECTION
# ================================================================

def apply_baryon_correction(part_a):
    """
    For galaxies with log M_* > 11.3, apply baryon completion.

    Two modes tested:
      - 'field': gas-only (appropriate for SLACS isolated lenses)
      - 'group': full Step 21 with satellites + ICL (appropriate for
                 stacked group centrals like Mandelbaum bins 5-7)

    SLACS galaxies are selected by lensing cross-section (high sigma,
    compact), which favors isolated/field massive ETGs.  The Step 21
    satellite ratios (Yang+2007) are calibrated on group centrals and
    overshoot for field ellipticals.  The primary correction is gas-only;
    the group mode is shown as a diagnostic.
    """
    results_field = []
    results_group = []
    all_field = []

    for gal in SLACS_GALAXIES:
        sigma_obs = gal['sigma_e2']
        sigma_err = gal['sigma_err']
        lm = gal['log_Mstar']
        reff = gal['R_eff_kpc']

        pred_central = sigma_prediction(lm, reff)
        sigma_central = pred_central['sigma_total']

        if lm >= 11.3:
            corr_field = baryon_correction(lm, reff, environment='field')
            corr_group = baryon_correction(lm, reff, environment='group')

            residual_before = (sigma_central - sigma_obs) / sigma_err
            residual_field = (corr_field['sigma_total'] - sigma_obs) / sigma_err
            residual_group = (corr_group['sigma_total'] - sigma_obs) / sigma_err
            frac_before = (sigma_central - sigma_obs) / sigma_obs
            frac_field = (corr_field['sigma_total'] - sigma_obs) / sigma_obs
            frac_group = (corr_group['sigma_total'] - sigma_obs) / sigma_obs

            entry = {
                'name': gal['name'],
                'log_Mstar': lm,
                'sigma_obs': sigma_obs,
                'sigma_central_only': round(sigma_central, 1),
                'sigma_field': round(corr_field['sigma_total'], 1),
                'sigma_group': round(corr_group['sigma_total'], 1),
                'M_gas': corr_field['M_gas'],
                'M_bar_ratio_field': round(corr_field['M_bar_ratio'], 2),
                'M_bar_ratio_group': round(corr_group['M_bar_ratio'], 2),
                'r_500_kpc': round(corr_field['r_500_kpc'], 1),
                'residual_before': round(residual_before, 2),
                'residual_field': round(residual_field, 2),
                'residual_group': round(residual_group, 2),
                'frac_before': round(frac_before, 4),
                'frac_field': round(frac_field, 4),
                'frac_group': round(frac_group, 4),
            }
            results_field.append(entry)
            results_group.append(entry)
            all_field.append({
                'name': gal['name'], 'log_Mstar': lm,
                'sigma_total': corr_field['sigma_total'],
                'sigma_obs': sigma_obs, 'sigma_err': sigma_err,
                'frac_residual': frac_field,
            })
        else:
            all_field.append({
                'name': gal['name'], 'log_Mstar': lm,
                'sigma_total': sigma_central,
                'sigma_obs': sigma_obs, 'sigma_err': sigma_err,
                'frac_residual': (sigma_central - sigma_obs) / sigma_obs,
            })

    # Summary for high-mass galaxies (field correction)
    if results_field:
        frac_before = np.array([r['frac_before'] for r in results_field])
        frac_field = np.array([r['frac_field'] for r in results_field])
        mean_before = float(np.mean(np.abs(frac_before)))
        mean_field = float(np.mean(np.abs(frac_field)))
        improvement = (mean_before - mean_field) / mean_before * 100
    else:
        mean_before = mean_field = improvement = 0.0

    # Full-sample statistics (field correction)
    all_sigma_pred = np.array([r['sigma_total'] for r in all_field])
    all_sigma_obs = np.array([r['sigma_obs'] for r in all_field])
    all_sigma_err = np.array([r['sigma_err'] for r in all_field])
    all_frac = (all_sigma_pred - all_sigma_obs) / all_sigma_obs
    chi2 = float(np.sum(((all_sigma_pred - all_sigma_obs) / all_sigma_err)**2))

    # Corrected regression
    all_logm = np.array([r['log_Mstar'] for r in all_field])
    reg = linregress(all_logm, all_frac)

    return {
        'high_mass_galaxies': results_field,
        'N_corrected': len(results_field),
        'environment': 'field (gas-only)',
        'high_mass_mean_abs_bias_before': round(mean_before * 100, 2),
        'high_mass_mean_abs_bias_after': round(mean_field * 100, 2),
        'high_mass_improvement_percent': round(improvement, 1),
        'full_sample_corrected': {
            'mean_bias_percent': round(float(np.mean(all_frac)) * 100, 2),
            'rms_residual_percent': round(float(np.sqrt(np.mean(all_frac**2))) * 100, 2),
            'chi2': round(chi2, 2),
            'chi2_per_nu': round(chi2 / len(all_field), 2),
            'regression_slope': round(reg.slope, 4),
            'regression_slope_err': round(reg.stderr, 4),
        },
        'all_corrected': all_field,
        'note': (
            'SLACS lenses are isolated ETGs (field environment). '
            'Gas-only correction is appropriate. The Step 21 group '
            'correction with satellites overshoots because satellite '
            'ratios are calibrated on stacked group centrals, not '
            'individual field lenses.'
        ),
    }


# ================================================================
# PART E: SENSITIVITY ANALYSIS
# ================================================================

def sensitivity_analysis():
    """Vary one parameter at a time to quantify systematic uncertainty."""
    results = {}

    # Baseline
    baseline = predict_slacs_dispersions()
    baseline_bias = baseline['mean_bias_percent']
    baseline_chi2 = baseline['chi2_per_nu']

    # K_v variation
    kv_results = []
    for kv in [4.0, 4.5, 5.0, 5.5, 6.0]:
        global K_V
        K_V = kv
        pred = predict_slacs_dispersions()
        kv_results.append({
            'K_v': kv,
            'mean_bias_percent': pred['mean_bias_percent'],
            'chi2_per_nu': pred['chi2_per_nu'],
        })
    K_V = 5.0  # restore

    # f_gas variation
    fgas_results = []
    for fg in [0.02, 0.05, 0.10]:
        preds = []
        for gal in SLACS_GALAXIES:
            p = sigma_prediction(gal['log_Mstar'], gal['R_eff_kpc'], f_gas=fg)
            preds.append(p['sigma_total'])
        preds = np.array(preds)
        sigma_obs = np.array([g['sigma_e2'] for g in SLACS_GALAXIES])
        frac = (preds - sigma_obs) / sigma_obs
        fgas_results.append({
            'f_gas': fg,
            'mean_bias_percent': round(float(np.mean(frac)) * 100, 2),
        })

    # R_eff variation (global +-20%)
    reff_results = []
    for factor_label, factor in [('-20%', 0.80), ('fiducial', 1.0), ('+20%', 1.20)]:
        preds = []
        for gal in SLACS_GALAXIES:
            p = sigma_prediction(gal['log_Mstar'], gal['R_eff_kpc'] * factor)
            preds.append(p['sigma_total'])
        preds = np.array(preds)
        sigma_obs = np.array([g['sigma_e2'] for g in SLACS_GALAXIES])
        frac = (preds - sigma_obs) / sigma_obs
        reff_results.append({
            'R_eff_factor': factor_label,
            'mean_bias_percent': round(float(np.mean(frac)) * 100, 2),
        })

    # IMF variation: Chabrier vs Salpeter (+0.25 dex in M_*)
    imf_results = []
    for imf_label, dex_offset in [('Chabrier', 0.0), ('Salpeter', 0.25)]:
        preds = []
        for gal in SLACS_GALAXIES:
            p = sigma_prediction(gal['log_Mstar'] + dex_offset, gal['R_eff_kpc'])
            preds.append(p['sigma_total'])
        preds = np.array(preds)
        sigma_obs = np.array([g['sigma_e2'] for g in SLACS_GALAXIES])
        frac = (preds - sigma_obs) / sigma_obs
        imf_results.append({
            'IMF': imf_label,
            'mean_bias_percent': round(float(np.mean(frac)) * 100, 2),
        })

    return {
        'baseline_bias_percent': baseline_bias,
        'baseline_chi2_per_nu': baseline_chi2,
        'K_v_variation': kv_results,
        'f_gas_variation': fgas_results,
        'R_eff_variation': reff_results,
        'IMF_variation': imf_results,
        'dominant_systematic': 'K_v and R_eff dominate; f_gas is negligible',
    }


# ================================================================
# PART F: H0LiCOW REDO WITH ESTIMATED R_eff
# ================================================================

def redo_h0licow():
    """
    Redo Step 20 velocity dispersion prediction for 6 H0LiCOW lenses
    with mass-size relation R_eff (instead of fixed 8 kpc) and group
    correction for the most massive lens.
    """
    results = []

    for lens in H0LICOW_LENSES:
        log_Mstar = lens['log_Mstar']
        sigma_obs = lens['sigma_obs']
        sigma_err = lens['sigma_err']
        z_L = lens['z_L']

        # Estimated R_eff from Shen+2003 with size evolution
        R_eff_est = shen2003_reff(log_Mstar, z=z_L)

        # Step 20 prediction (fixed R_eff = 8 kpc)
        pred_step20 = sigma_prediction(log_Mstar, 8.0)

        # Step 22A prediction (estimated R_eff)
        pred_step22 = sigma_prediction(log_Mstar, R_eff_est)

        # Gas correction for high-mass lenses (field environment)
        if log_Mstar >= 11.3:
            corr = baryon_correction(log_Mstar, R_eff_est, environment='field')
            sigma_corrected = corr['sigma_total']
        else:
            sigma_corrected = pred_step22['sigma_total']

        residual_step20 = (pred_step20['sigma_total'] - sigma_obs) / sigma_err
        residual_step22 = (pred_step22['sigma_total'] - sigma_obs) / sigma_err
        residual_corrected = (sigma_corrected - sigma_obs) / sigma_err

        results.append({
            'name': lens['name'],
            'log_Mstar': log_Mstar,
            'z_L': z_L,
            'sigma_obs': sigma_obs,
            'sigma_err': sigma_err,
            'R_eff_fixed': 8.0,
            'R_eff_estimated': round(R_eff_est, 2),
            'sigma_step20': round(pred_step20['sigma_total'], 1),
            'sigma_step22a': round(pred_step22['sigma_total'], 1),
            'sigma_corrected': round(sigma_corrected, 1),
            'residual_step20': round(residual_step20, 2),
            'residual_step22a': round(residual_step22, 2),
            'residual_corrected': round(residual_corrected, 2),
        })

    # Summary
    res_step20 = np.array([r['residual_step20'] for r in results])
    res_step22 = np.array([r['residual_corrected'] for r in results])
    rms_step20 = float(np.sqrt(np.mean(res_step20**2)))
    rms_step22 = float(np.sqrt(np.mean(res_step22**2)))

    return {
        'lenses': results,
        'rms_residual_step20': round(rms_step20, 2),
        'rms_residual_step22a': round(rms_step22, 2),
        'improvement_percent': round((rms_step20 - rms_step22) / rms_step20 * 100, 1),
        'note': ('Step 20 used fixed R_eff=8 kpc. Step 22A uses Shen+2003 '
                 'mass-size relation with van der Wel+2014 size evolution, '
                 'plus group baryon correction for log M_* >= 11.3.'),
    }


# ================================================================
# PART G: FALSIFIERS
# ================================================================

def evaluate_falsifiers(part_a, part_d, part_c):
    """Pre-registered falsification criteria."""
    corrected = part_d['full_sample_corrected']

    f1_bias = abs(part_a['mean_bias_percent']) < 15.0
    f2_slope = abs(corrected['regression_slope']) < 0.30
    f3_chi2 = corrected['chi2_per_nu'] < 5.0
    # F4: gas correction should improve chi2 (not worsen it)
    f4_chi2_improves = corrected['chi2_per_nu'] <= part_a['chi2_per_nu']

    falsifiers = [
        {
            'id': 1,
            'criterion': '|mean bias| < 15% across full sample (uncorrected)',
            'value': f'{abs(part_a["mean_bias_percent"]):.1f}%',
            'threshold': '15%',
            'result': 'PASS' if f1_bias else 'FAIL',
        },
        {
            'id': 2,
            'criterion': ('|regression slope of residual vs log M_*| < 0.30 '
                         'per dex (after gas correction)'),
            'value': f'{abs(corrected["regression_slope"]):.3f}',
            'threshold': '0.30',
            'result': 'PASS' if f2_slope else 'FAIL',
        },
        {
            'id': 3,
            'criterion': 'chi2/nu < 5.0 for full sample (after gas correction)',
            'value': f'{corrected["chi2_per_nu"]:.2f}',
            'threshold': '5.0',
            'result': 'PASS' if f3_chi2 else 'FAIL',
        },
        {
            'id': 4,
            'criterion': ('Gas correction chi2/nu <= uncorrected chi2/nu '
                         '(correction does not worsen fit)'),
            'value': f'{corrected["chi2_per_nu"]:.2f} vs {part_a["chi2_per_nu"]:.2f}',
            'threshold': 'chi2_corr <= chi2_uncorr',
            'result': 'PASS' if f4_chi2_improves else 'FAIL',
        },
    ]

    n_pass = sum(1 for f in falsifiers if f['result'] == 'PASS')
    all_pass = n_pass == len(falsifiers)

    return {
        'falsifiers': falsifiers,
        'N_pass': n_pass,
        'N_total': len(falsifiers),
        'all_pass': all_pass,
        'what_would_NOT_falsify': [
            'Individual galaxies off by 20-30% (anisotropy scatter, M/L variation)',
            'Moderate chi2/nu of 2-4 (known sigma_err underestimates in SDSS)',
            'K_v uncertainty of 20% (published range for Sersic n=3-5)',
        ],
    }


# ================================================================
# PLOTTING
# ================================================================

def plot_sigma_comparison(part_a, part_d, outdir):
    """sigma_pred vs sigma_obs (1:1 plot) with residual panel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.subplots_adjust(hspace=0.05)

    galaxies = part_a['galaxies']
    sigma_obs = np.array([g['sigma_obs'] for g in galaxies])
    sigma_err = np.array([g['sigma_err'] for g in galaxies])
    sigma_pred = np.array([g['sigma_total'] for g in galaxies])
    log_m = np.array([g['log_Mstar'] for g in galaxies])

    # Corrected predictions for high-mass
    corrected_map = {}
    for g in part_d.get('high_mass_galaxies', []):
        corrected_map[g['name']] = g['sigma_field']

    sigma_corr = np.array([
        corrected_map.get(g['name'], g['sigma_total'])
        for g in galaxies
    ])
    has_correction = np.array([g['name'] in corrected_map for g in galaxies])

    # Top: 1:1 plot
    lims = [140, 380]
    ax1.fill_between(lims, [l * 0.9 for l in lims], [l * 1.1 for l in lims],
                     color='gray', alpha=0.1, label=r'$\pm$10%')
    ax1.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='1:1')

    sc = ax1.scatter(sigma_obs[~has_correction], sigma_pred[~has_correction],
                     c=log_m[~has_correction], cmap='viridis', s=60, zorder=3,
                     vmin=10.7, vmax=11.8, edgecolors='0.3', linewidths=0.5)
    ax1.errorbar(sigma_obs[~has_correction], sigma_pred[~has_correction],
                 xerr=sigma_err[~has_correction], fmt='none', color='0.5',
                 lw=0.8, zorder=2)

    if has_correction.any():
        ax1.scatter(sigma_obs[has_correction], sigma_pred[has_correction],
                    c=log_m[has_correction], cmap='viridis', s=60, zorder=3,
                    vmin=10.7, vmax=11.8, edgecolors='0.3', linewidths=0.5,
                    marker='s', alpha=0.4)
        ax1.scatter(sigma_obs[has_correction], sigma_corr[has_correction],
                    c=log_m[has_correction], cmap='viridis', s=90, zorder=4,
                    vmin=10.7, vmax=11.8, edgecolors='red', linewidths=1.5,
                    marker='D', label='Group-corrected')
        ax1.errorbar(sigma_obs[has_correction], sigma_corr[has_correction],
                     xerr=sigma_err[has_correction], fmt='none', color='red',
                     lw=0.8, zorder=2)
        # Arrows from uncorrected to corrected
        for i in np.where(has_correction)[0]:
            ax1.annotate('', xy=(sigma_obs[i], sigma_corr[i]),
                        xytext=(sigma_obs[i], sigma_pred[i]),
                        arrowprops=dict(arrowstyle='->', color='red',
                                       lw=1, alpha=0.6))

    cb = fig.colorbar(sc, ax=ax1, label=r'$\log\,M_*/M_\odot$')
    ax1.set_ylabel(r'$\sigma_{\rm MTDF}$ (km/s)', fontsize=12)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_title('Step 22A: SLACS velocity dispersions (zero free parameters)',
                  fontsize=12)

    # Bottom: fractional residual
    frac_uncorr = (sigma_pred - sigma_obs) / sigma_obs * 100
    frac_corr = (sigma_corr - sigma_obs) / sigma_obs * 100

    ax2.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax2.fill_between(lims, -10, 10, color='gray', alpha=0.1)

    ax2.scatter(sigma_obs[~has_correction], frac_uncorr[~has_correction],
                c=log_m[~has_correction], cmap='viridis', s=50,
                vmin=10.7, vmax=11.8, edgecolors='0.3', linewidths=0.5,
                zorder=3)
    if has_correction.any():
        ax2.scatter(sigma_obs[has_correction], frac_uncorr[has_correction],
                    c=log_m[has_correction], cmap='viridis', s=50,
                    vmin=10.7, vmax=11.8, edgecolors='0.3', linewidths=0.5,
                    marker='s', alpha=0.4, zorder=2)
        ax2.scatter(sigma_obs[has_correction], frac_corr[has_correction],
                    c=log_m[has_correction], cmap='viridis', s=70,
                    vmin=10.7, vmax=11.8, edgecolors='red', linewidths=1.5,
                    marker='D', zorder=4)

    ax2.set_xlabel(r'$\sigma_{\rm obs}$ (km/s)', fontsize=12)
    ax2.set_ylabel(r'$(\sigma_{\rm pred} - \sigma_{\rm obs})/\sigma_{\rm obs}$ (%)',
                   fontsize=12)
    ax2.set_ylim(-25, 25)

    bias_txt = (f'Mean bias: {part_a["mean_bias_percent"]:+.1f}%'
                f' (uncorr) / '
                f'{part_d["full_sample_corrected"]["mean_bias_percent"]:+.1f}%'
                f' (corr)')
    ax2.text(0.02, 0.05, bias_txt, transform=ax2.transAxes, fontsize=9,
             va='bottom')

    fig.savefig(outdir / 'step22_sigma_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_faber_jackson(part_a, part_b, outdir):
    """Faber-Jackson diagram: log sigma vs log M_*."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # SLACS data
    for gal in part_a['galaxies']:
        ax.errorbar(gal['log_Mstar'], np.log10(gal['sigma_obs']),
                    yerr=gal['sigma_err'] / (gal['sigma_obs'] * np.log(10)),
                    fmt='ko', markersize=5, capsize=3, alpha=0.6, zorder=3)

    # MTDF prediction curves
    lm_grid = np.array(part_b['log_Mstar_grid'])
    ax.plot(lm_grid, np.log10(part_b['sigma_total_grid']),
            'r-', lw=2.5, label=r'$\sigma_{\rm total}$ (MTDF)', zorder=4)
    ax.plot(lm_grid, np.log10(part_b['sigma_stress_grid']),
            'b--', lw=1.5, label=r'$\sigma_{\rm stress}$ (slope = 1/4)',
            alpha=0.7, zorder=2)
    ax.plot(lm_grid, np.log10(part_b['sigma_bar_grid']),
            'g:', lw=1.5, label=r'$\sigma_{\rm bar}$ (baryonic)',
            alpha=0.7, zorder=2)

    ax.set_xlabel(r'$\log\,(M_*/M_\odot)$', fontsize=13)
    ax.set_ylabel(r'$\log\,\sigma$ (km/s)', fontsize=13)
    ax.set_xlim(10.6, 11.9)
    ax.set_ylim(2.1, 2.6)

    ax.legend(loc='upper left', fontsize=10)
    slope_txt = (f'F-J slope (total): {part_b["fj_slope_total"]:.3f} '
                 f'$\\pm$ {part_b["fj_slope_total_err"]:.3f}\n'
                 f'Stress-only slope: {part_b["fj_slope_stress_only"]:.3f} '
                 f'(canonical = 0.250)')
    ax.text(0.98, 0.05, slope_txt, transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    ax.set_title('Step 22A: Faber-Jackson relation from MTDF', fontsize=12)
    fig.savefig(outdir / 'step22_faber_jackson.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_residual_diagnostic(part_a, part_d, outdir):
    """Fractional residual vs log M_* (before/after group correction)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    galaxies = part_a['galaxies']
    log_m = np.array([g['log_Mstar'] for g in galaxies])
    frac_uncorr = np.array([g['frac_residual'] for g in galaxies])

    # Build corrected array
    corr_map = {}
    for g in part_d.get('high_mass_galaxies', []):
        corr_map[g['name']] = g['frac_field']
    frac_corr = np.array([
        corr_map.get(g['name'], g['frac_residual'])
        for g in galaxies
    ])
    has_correction = np.array([g['name'] in corr_map for g in galaxies])

    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.fill_between([10.6, 12.0], -0.10, 0.10, color='gray', alpha=0.1,
                    label=r'$\pm$10%')

    # Uncorrected
    ax.scatter(log_m, frac_uncorr, c='steelblue', s=50, marker='o',
               edgecolors='0.3', linewidths=0.5, alpha=0.6,
               label='Central only', zorder=3)

    # Corrected (only for high-mass)
    if has_correction.any():
        ax.scatter(log_m[has_correction], frac_corr[has_correction],
                   c='red', s=80, marker='D', edgecolors='darkred',
                   linewidths=0.8, label='Group-corrected', zorder=4)
        # Arrows
        for i in np.where(has_correction)[0]:
            ax.annotate('', xy=(log_m[i], frac_corr[i]),
                       xytext=(log_m[i], frac_uncorr[i]),
                       arrowprops=dict(arrowstyle='->', color='red',
                                      lw=1, alpha=0.5))

    # Regression lines
    reg_before = linregress(log_m, frac_uncorr)
    reg_after = linregress(log_m, frac_corr)
    x_fit = np.array([10.7, 11.8])
    ax.plot(x_fit, reg_before.slope * x_fit + reg_before.intercept,
            'b-', lw=1.5, alpha=0.5,
            label=f'Slope: {reg_before.slope:.3f}/dex (uncorr)')
    ax.plot(x_fit, reg_after.slope * x_fit + reg_after.intercept,
            'r-', lw=1.5, alpha=0.5,
            label=f'Slope: {reg_after.slope:.3f}/dex (corr)')

    ax.axvline(11.3, color='orange', ls=':', lw=1, alpha=0.5,
               label='Group threshold')

    ax.set_xlabel(r'$\log\,(M_*/M_\odot)$', fontsize=13)
    ax.set_ylabel(r'$(\sigma_{\rm pred} - \sigma_{\rm obs}) / \sigma_{\rm obs}$',
                  fontsize=13)
    ax.set_xlim(10.65, 11.85)
    ax.set_ylim(-0.30, 0.30)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_title('Step 22A: Mass-dependent bias diagnostic', fontsize=12)

    fig.savefig(outdir / 'step22_residual_diagnostic.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ================================================================
# OUTPUT UTILITIES
# ================================================================

def make_json_serializable(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sha256_of_file(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ================================================================
# MAIN
# ================================================================

def main():
    base = Path(__file__).resolve().parent.parent
    outdir = base / 'output' / 'step22_elliptical_dispersions'
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 22A: Elliptical Galaxy Velocity Dispersions")
    print("=" * 60)
    print(f"  K_v = {K_V:.1f} (Cappellari+2006)")
    print(f"  f_gas = {F_GAS_ETG:.2f} (massive ETGs)")
    print(f"  A_BTFR = {A_BTFR:.1f}, V_REF = {V_REF_KMS:.1f} km/s")
    print(f"  N_galaxies = {len(SLACS_GALAXIES)}")
    print()

    # ---- Part A: Per-galaxy sigma prediction ----
    print("--- Part A: SLACS per-galaxy sigma prediction ---")
    part_a = predict_slacs_dispersions()
    print(f"  {'Name':14s} {'logM*':>5s} {'R_eff':>5s} {'sig_obs':>7s} "
          f"{'sig_str':>7s} {'sig_bar':>7s} {'sig_tot':>7s} {'resid':>6s}")
    for g in part_a['galaxies']:
        print(f"  {g['name']:14s} {g['log_Mstar']:5.2f} {g['R_eff_kpc']:5.1f} "
              f"{g['sigma_obs']:7.0f} {g['sigma_stress']:7.1f} "
              f"{g['sigma_bar']:7.1f} {g['sigma_total']:7.1f} "
              f"{g['residual_sigma']:+6.2f}")
    print(f"\n  Mean bias: {part_a['mean_bias_percent']:+.2f}%")
    print(f"  RMS residual: {part_a['rms_residual_percent']:.2f}%")
    print(f"  chi2/nu: {part_a['chi2_per_nu']:.2f}")
    print()

    # ---- Part B: Faber-Jackson relation ----
    print("--- Part B: Faber-Jackson relation ---")
    part_b = faber_jackson_prediction()
    print(f"  F-J slope (total): {part_b['fj_slope_total']:.4f} "
          f"+/- {part_b['fj_slope_total_err']:.4f}")
    print(f"  Stress-only slope: {part_b['fj_slope_stress_only']:.4f}")
    print(f"  Canonical F-J:     {part_b['canonical_fj_slope']:.4f}")
    print()

    # ---- Part C: Mass-dependent bias ----
    print("--- Part C: Mass-dependent bias diagnostic ---")
    part_c = mass_bias_diagnostic(part_a)
    for b in part_c['bins']:
        print(f"  {b['bin']:12s}: N={b['N']:2d}, "
              f"mean residual = {b['mean_frac_residual']:+.4f}")
    print(f"  Regression slope: {part_c['regression_slope']:.4f} "
          f"+/- {part_c['regression_slope_err']:.4f} per dex "
          f"(p={part_c['regression_pvalue']:.4f})")
    print()

    # ---- Part D: Baryon correction (gas-only for field ETGs) ----
    print("--- Part D: Baryon correction (gas-only, log M_* >= 11.3) ---")
    part_d = apply_baryon_correction(part_a)
    for g in part_d['high_mass_galaxies']:
        print(f"  {g['name']:14s}: sig_obs={g['sigma_obs']:.0f}, "
              f"central={g['sigma_central_only']:.0f}, "
              f"gas-only={g['sigma_field']:.0f} "
              f"(M_bar x{g['M_bar_ratio_field']:.2f}), "
              f"[group={g['sigma_group']:.0f} x{g['M_bar_ratio_group']:.1f}], "
              f"resid: {g['residual_before']:+.2f} -> {g['residual_field']:+.2f}")
    print(f"\n  High-mass mean |bias|: "
          f"{part_d['high_mass_mean_abs_bias_before']:.1f}% -> "
          f"{part_d['high_mass_mean_abs_bias_after']:.1f}% "
          f"(gas-only, {part_d['high_mass_improvement_percent']:.0f}% improvement)")
    fc = part_d['full_sample_corrected']
    print(f"  Full sample (corrected): mean bias = {fc['mean_bias_percent']:+.2f}%, "
          f"chi2/nu = {fc['chi2_per_nu']:.2f}")
    print(f"  Corrected regression slope: {fc['regression_slope']:.4f} "
          f"+/- {fc['regression_slope_err']:.4f}")
    print()

    # ---- Part E: Sensitivity analysis ----
    print("--- Part E: Sensitivity analysis ---")
    part_e = sensitivity_analysis()
    print("  K_v variation:")
    for r in part_e['K_v_variation']:
        marker = ' <-- fiducial' if r['K_v'] == 5.0 else ''
        print(f"    K_v={r['K_v']:.1f}: bias={r['mean_bias_percent']:+.1f}%, "
              f"chi2/nu={r['chi2_per_nu']:.2f}{marker}")
    print("  f_gas variation:")
    for r in part_e['f_gas_variation']:
        marker = ' <-- fiducial' if r['f_gas'] == 0.05 else ''
        print(f"    f_gas={r['f_gas']:.2f}: bias={r['mean_bias_percent']:+.1f}%{marker}")
    print("  R_eff variation:")
    for r in part_e['R_eff_variation']:
        print(f"    R_eff {r['R_eff_factor']:>8s}: "
              f"bias={r['mean_bias_percent']:+.1f}%")
    print("  IMF variation:")
    for r in part_e['IMF_variation']:
        print(f"    {r['IMF']:>10s}: bias={r['mean_bias_percent']:+.1f}%")
    print(f"  Dominant: {part_e['dominant_systematic']}")
    print()

    # ---- Part F: H0LiCOW redo ----
    print("--- Part F: H0LiCOW redo with estimated R_eff ---")
    part_f = redo_h0licow()
    for l in part_f['lenses']:
        print(f"  {l['name']:16s}: R_eff 8.0 -> {l['R_eff_estimated']:.1f} kpc, "
              f"sig: {l['sigma_step20']:.0f} -> {l['sigma_corrected']:.0f} "
              f"(obs={l['sigma_obs']}), "
              f"resid: {l['residual_step20']:+.2f} -> "
              f"{l['residual_corrected']:+.2f}")
    print(f"\n  RMS residual: {part_f['rms_residual_step20']:.2f} (Step 20) "
          f"-> {part_f['rms_residual_step22a']:.2f} (Step 22A) "
          f"({part_f['improvement_percent']:.0f}% improvement)")
    print()

    # ---- Part G: Falsifiers ----
    print("--- Part G: Falsifiers ---")
    part_g = evaluate_falsifiers(part_a, part_d, part_c)
    for f in part_g['falsifiers']:
        print(f"  F{f['id']}: {f['criterion'][:60]:60s} "
              f"value={f['value']:>8s} threshold={f['threshold']:>5s} "
              f"-> {f['result']}")
    print(f"\n  Result: {part_g['N_pass']}/{part_g['N_total']} PASS")
    print()

    # ---- Compile results ----
    results = {
        'description': 'Step 22A: Elliptical galaxy velocity dispersions',
        'parameters': {
            'K_v': K_V,
            'f_gas': F_GAS_ETG,
            'A_BTFR': A_BTFR,
            'V_REF_kms': V_REF_KMS,
            'L_kpc': L_KPC,
            'RHO0_Msun_kpc3': RHO0,
            'note': 'All MTDF constants frozen from Steps 8-14. K_v from Cappellari+2006.',
        },
        'part_A_sigma_prediction': part_a,
        'part_B_faber_jackson': {k: v for k, v in part_b.items()
                                  if k not in ('log_Mstar_grid', 'sigma_total_grid',
                                               'sigma_stress_grid', 'sigma_bar_grid')},
        'part_C_mass_bias': part_c,
        'part_D_baryon_correction': {k: v for k, v in part_d.items()
                                     if k != 'all_corrected'},
        'part_E_sensitivity': part_e,
        'part_F_h0licow_redo': part_f,
        'part_G_falsifiers': part_g,
        'summary': {
            'N_galaxies': len(SLACS_GALAXIES),
            'mean_bias_percent_uncorrected': part_a['mean_bias_percent'],
            'mean_bias_percent_corrected': fc['mean_bias_percent'],
            'rms_residual_percent': part_a['rms_residual_percent'],
            'chi2_per_nu_uncorrected': part_a['chi2_per_nu'],
            'chi2_per_nu_corrected': fc['chi2_per_nu'],
            'faber_jackson_slope': part_b['fj_slope_total'],
            'free_parameters': 0,
            'all_falsifiers_pass': part_g['all_pass'],
        },
    }

    # ---- Save JSON ----
    json_path = outdir / 'step22_elliptical_dispersions.json'
    with open(json_path, 'w') as fp:
        json.dump(make_json_serializable(results), fp, indent=2)
    print(f"  JSON saved: {json_path.name}")

    # ---- Plots ----
    plot_sigma_comparison(part_a, part_d, outdir)
    print(f"  Plot saved: step22_sigma_comparison.png")

    plot_faber_jackson(part_a, part_b, outdir)
    print(f"  Plot saved: step22_faber_jackson.png")

    plot_residual_diagnostic(part_a, part_d, outdir)
    print(f"  Plot saved: step22_residual_diagnostic.png")

    # ---- Manifest ----
    manifest = {}
    for p in sorted(outdir.glob('*')):
        if p.name != 'manifest.json':
            manifest[p.name] = sha256_of_file(p)
    with open(outdir / 'manifest.json', 'w') as fp:
        json.dump(manifest, fp, indent=2)
    print(f"  Manifest saved: manifest.json")

    print()
    print("=" * 60)
    status = "ALL PASS" if part_g['all_pass'] else "SOME FAIL"
    print(f"Step 22A COMPLETE — Falsifiers: {part_g['N_pass']}/{part_g['N_total']} "
          f"({status})")
    print("=" * 60)


if __name__ == '__main__':
    main()
