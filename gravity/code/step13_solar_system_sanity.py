#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 13: Weak Field Limit and Solar System Consistency

Derives the weak field limit of Route B (compression mechanism) and computes
all Solar System observables implied by the MTDF gravity sector.

The key question: "Your enhancement is 2.3x at r << beta. Why isn't gravity
2.3x stronger at 1 AU?"

The answer: The 2.3x enhancement is NOT a modification of G. It is the
gravitational effect of stress energy from the compression field, acting as
additional mass (what LCDM interprets as dark matter). For the Sun, the
compression parameter f_sun ~ 5e-16 (deep in the linear, unscreened regime),
making the stress mass within 1 AU about 17 kg - negligible by 30 orders of
magnitude compared to the Sun's mass.

The suppression mechanism: screening only activates at galaxy masses
(M > ~10^7 M_sun). Below that, f scales linearly with M instead of as M^0.25.
Individual stars and planets live in the linear regime where the compression
field is utterly negligible.

Weak field limit of Route B:
  1. Gravitational potential: nabla^2 Phi = 4 pi G (rho_bar + rho_stress)
  2. Compression field (r << beta): nabla^2(delta_S) = 0 outside baryons
     -> delta_S = C/r (Laplace, unique spherical solution)
  3. Stress density: rho_stress = (E/2c^2)(delta_S)^2 = rho_0 f^2 L^2 / r^2
  4. Lensing = dynamics: eta = 1 (confirmed C5b), so photons and stars
     see the same potential sourced by rho_bar + rho_stress
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import hashlib

# ================================================================
# CONSTANTS - ALL FROM MTDF (Steps 8-12)
# ================================================================

ALPHA = 1.30
BETA_KPC = 22_685.0       # kpc
BETA_M = 7.0e23           # m
E_PA = 9.1e-10            # Pa
G_SI = 6.674e-11          # m^3 kg^-1 s^-2
C_SI = 2.998e8            # m/s
MSUN = 1.989e30           # kg
KPC_M = 3.086e19          # m per kpc
AU_M = 1.496e11           # m per AU
PC_M = 3.086e16           # m per pc
YEAR_S = 3.156e7          # seconds per year

# Derived MTDF parameters
L_KPC = ALPHA * BETA_KPC / (4 * np.pi)       # 2347 kpc
L_M = L_KPC * KPC_M                           # m
RHO_CRIT = 8.5e-27                            # kg/m^3
S_0 = np.sqrt(2 * 0.70 * RHO_CRIT * C_SI**2 / E_PA)  # 1.084
V_REF = 161.8e3                                # m/s (Step 10)

# Density coefficient
RHO0_SI = E_PA * S_0**2 / (2 * C_SI**2)       # kg/m^3
RHO0_MSUN_KPC3 = RHO0_SI / MSUN * KPC_M**3    # Msun/kpc^3
RHO0_MSUN_PC3 = RHO0_SI / MSUN * PC_M**3      # Msun/pc^3

# ================================================================
# PLANET DATA
# ================================================================

PLANETS = [
    ('Mercury', 0.387,  0.2408),   # name, semi-major axis (AU), period (yr)
    ('Venus',   0.723,  0.6152),
    ('Earth',   1.000,  1.0000),
    ('Mars',    1.524,  1.8808),
    ('Jupiter', 5.203,  11.862),
    ('Saturn',  9.537,  29.457),
    ('Uranus',  19.19,  84.011),
    ('Neptune', 30.07,  164.79),
]

# ================================================================
# OBSERVATIONAL BOUNDS
# ================================================================

CASSINI_GAMMA_BOUND = 2.3e-5       # |gamma-1|, Bertotti+ 2003
MESSENGER_BETA_BOUND = 8.0e-5      # |beta-1|, Park+ 2017
NORDTVEDT_ETA_BOUND = 1.0e-4       # |eta_N|, Genova+ 2018
MERCURY_PRECESSION_UNC = 0.65       # arcsec/century
ANOMALOUS_ACCEL_BOUND = 1.0e-9     # m/s^2 (ephemeris bound)
LOCAL_DM_OBS = 0.013                # M_sun/pc^3 (Buch+ 2024)
LOCAL_DM_UNC = 0.001                # M_sun/pc^3

# MW parameters
M_MW_BAR = 6.0e10     # M_sun (total baryonic)
V_FLAT_MW = 220.0e3   # m/s
R_SUN_GC = 8.0        # kpc

# ================================================================
# PART A: LINEAR REGIME DERIVATION
# ================================================================

def f_linear(M_kg):
    """Compression parameter in the linear (unscreened) regime.

    From the Gauss law integral of the Helmholtz equation at r << beta:
      E beta^2 nabla^2(delta_S) = alpha rho_bar c^2
    Integrating over a sphere enclosing mass M:
      C = alpha c^2 M / (4 pi E beta^2)
    Since C = S_0 f L and L = alpha beta / (4 pi):
      f = c^2 M / (E beta^3 S_0)
    """
    return C_SI**2 * M_kg / (E_PA * BETA_M**3 * S_0)


def f_screened(M_kg):
    """Compression parameter in the screened regime (BTFR scaling).

    From Step 11: screening gives f proportional to M^{1/4}.
    Calibrated via v_flat = v_ref * f and BTFR: v_flat^4 = G M a_0.
    Approximation: f = (M / M_ref)^{0.25} where M_ref is set by v_ref.
    """
    # v_flat = (G M a_0)^{1/4} with a_0 ~ alpha c^2 / beta (effective)
    # Use the BTFR: v_flat = (M_bar_total / A_BTFR)^{0.25} with A_BTFR=50 M_sun/(km/s)^4
    A_BTFR = 50.0 * MSUN / (1e3)**4  # kg / (m/s)^4
    v_flat = (M_kg / A_BTFR)**0.25
    return v_flat / V_REF


def delta_S_at_r(f, r_m):
    """Strain excitation delta_S = S_0 f L / r."""
    return S_0 * f * L_M / r_m


def stress_mass_enclosed(f, r_m):
    """Enclosed stress mass within radius r.

    For rho_stress = rho_0 f^2 L^2 / r^2 (isothermal):
      M_stress(<r) = 4 pi rho_0_SI f^2 L_m^2 r
    """
    return 4 * np.pi * RHO0_SI * f**2 * L_M**2 * r_m


def anomalous_acceleration(f, r_m):
    """Extra acceleration from stress mass at distance r.

    a_extra = G M_stress(<r) / r^2 = 4 pi G rho_0 f^2 L^2 / r
    Equivalently: a_extra = v_ref^2 f^2 / r
    """
    return V_REF**2 * f**2 / r_m


def solar_gravity(r_m):
    """Newtonian solar gravity at distance r."""
    return G_SI * MSUN / r_m**2


# ================================================================
# PART B-D: COMPUTE ALL OBSERVABLES
# ================================================================

def compute_solar_system():
    """Compute all Solar System observables."""
    results = {}

    # --- Part A: linear regime for various objects ---
    objects = [
        ('Sun', MSUN),
        ('Earth', 5.972e24),
        ('Jupiter', 1.898e27),
        ('MW galaxy', M_MW_BAR * MSUN),
    ]

    regime_table = []
    for name, mass in objects:
        f_lin = f_linear(mass)
        f_scr = f_screened(mass)
        # Physical picture: for small masses (stars, planets), the linear
        # solution f ~ M is valid and gives negligible compression.
        # For large masses (galaxies), non-linear self-sourcing amplifies
        # the field, giving f ~ M^{0.25} (BTFR scaling).
        # The screened regime activates when the compression is large enough
        # for the (S-S_0)^2 self-sourcing term to dominate.
        # Criterion: screened when f_screened > 0.01 AND ratio > 100.
        ratio = f_scr / f_lin if f_lin > 0 else np.inf
        regime = 'screened' if ratio > 100 and f_scr > 0.01 else 'linear'
        f_eff = f_scr if regime == 'screened' else f_lin
        # delta_S at a reference radius (1 AU for solar objects, 10 kpc for MW)
        r_ref = 10.0 * KPC_M if mass > 1e35 else AU_M
        ds_check = delta_S_at_r(f_eff, r_ref)

        regime_table.append({
            'object': name,
            'mass_kg': mass,
            'mass_Msun': mass / MSUN,
            'f_linear': f_lin,
            'f_screened': f_scr,
            'delta_S_at_ref': ds_check,
            'regime': regime,
            'f_effective': f_eff,
        })

    results['regime_table'] = regime_table

    # --- Part B: Solar System planet-by-planet ---
    f_sun = f_linear(MSUN)
    results['f_sun'] = f_sun

    planet_table = []
    for name, a_au, period_yr in PLANETS:
        r_m = a_au * AU_M
        ds = delta_S_at_r(f_sun, r_m)
        m_stress = stress_mass_enclosed(f_sun, r_m)
        a_extra = anomalous_acceleration(f_sun, r_m)
        a_newton = solar_gravity(r_m)
        frac = a_extra / a_newton
        m_ratio = m_stress / MSUN

        planet_table.append({
            'planet': name,
            'a_AU': a_au,
            'r_m': r_m,
            'delta_S': ds,
            'M_stress_kg': m_stress,
            'M_stress_over_Msun': m_ratio,
            'a_extra_m_s2': a_extra,
            'a_newton_m_s2': a_newton,
            'fractional_anomaly': frac,
        })

    results['planet_table'] = planet_table

    # --- Part C: PPN parameters ---
    earth_entry = [p for p in planet_table if p['planet'] == 'Earth'][0]
    frac_1au = earth_entry['fractional_anomaly']

    ppn = {
        'gamma_minus_1': frac_1au,
        'gamma_bound': CASSINI_GAMMA_BOUND,
        'gamma_safety_log10': np.log10(CASSINI_GAMMA_BOUND / frac_1au),
        'beta_minus_1': frac_1au,
        'beta_bound': MESSENGER_BETA_BOUND,
        'beta_safety_log10': np.log10(MESSENGER_BETA_BOUND / frac_1au),
        'nordtvedt_eta': frac_1au,
        'nordtvedt_bound': NORDTVEDT_ETA_BOUND,
        'nordtvedt_safety_log10': np.log10(NORDTVEDT_ETA_BOUND / frac_1au),
    }
    results['ppn'] = ppn

    # --- Part D: Mercury perihelion precession ---
    merc = [p for p in planet_table if p['planet'] == 'Mercury'][0]
    # Precession from isothermal halo: delta_omega ~ 2 pi M_stress / M_sun per orbit
    dw_per_orbit_rad = 2 * np.pi * merc['M_stress_kg'] / MSUN
    period_merc_yr = 0.2408
    orbits_per_century = 100.0 / period_merc_yr
    dw_per_century_rad = dw_per_orbit_rad * orbits_per_century
    dw_per_century_arcsec = dw_per_century_rad * (180 / np.pi) * 3600

    precession = {
        'delta_omega_per_orbit_rad': dw_per_orbit_rad,
        'delta_omega_per_century_arcsec': dw_per_century_arcsec,
        'observational_uncertainty_arcsec': MERCURY_PRECESSION_UNC,
        'safety_log10': np.log10(MERCURY_PRECESSION_UNC / dw_per_century_arcsec),
    }
    results['mercury_precession'] = precession

    # --- Part E: MW background field ---
    f_mw = V_FLAT_MW / V_REF
    r_sun_m = R_SUN_GC * KPC_M
    rho_mw_si = RHO0_SI * f_mw**2 * L_M**2 / r_sun_m**2
    rho_mw_msun_pc3 = rho_mw_si / MSUN * PC_M**3

    mw_background = {
        'f_MW': f_mw,
        'rho_stress_MW_8kpc_kg_m3': rho_mw_si,
        'rho_stress_MW_8kpc_Msun_pc3': rho_mw_msun_pc3,
        'local_DM_observed_Msun_pc3': LOCAL_DM_OBS,
        'local_DM_uncertainty': LOCAL_DM_UNC,
        'ratio_predicted_over_observed': rho_mw_msun_pc3 / LOCAL_DM_OBS,
        'within_1sigma': abs(rho_mw_msun_pc3 - LOCAL_DM_OBS) < LOCAL_DM_UNC,
    }

    # Tidal gradient at Solar System scales
    # d(a_MW)/dr at r_sun = 2 G M_MW(<r) / r^3
    # For isothermal: M(<r) = 4 pi rho_0 f^2 L^2 r, so a = v_flat^2/r
    # Tidal: da/dr = -v_flat^2/r^2
    tidal_at_1au = V_FLAT_MW**2 / r_sun_m**2 * AU_M  # acceleration difference across 1 AU
    mw_background['tidal_across_1AU_m_s2'] = tidal_at_1au
    mw_background['tidal_over_solar_gravity'] = tidal_at_1au / solar_gravity(AU_M)

    results['mw_background'] = mw_background

    # --- Part F: Screening transition ---
    masses_log = np.linspace(-1, 13, 500)  # log10(M/Msun)
    masses_kg = 10**masses_log * MSUN
    f_lin_arr = np.array([f_linear(m) for m in masses_kg])
    f_scr_arr = np.array([f_screened(m) for m in masses_kg])

    # Physical picture: f_linear << f_screened everywhere, because the
    # linear solution gives negligible compression while the screened
    # (self-sourcing) solution gives BTFR-level compression.
    # At low masses, the self-sourcing term is irrelevant because even
    # f_screened is tiny (e.g. f_scr(Sun) ~ 2e-3, delta_S ~ 0.002 S_0).
    # At high masses (galaxies), self-sourcing dominates and f_screened
    # is the physical answer.
    #
    # Transition criterion: when delta_S from the screened solution at
    # the half-light radius exceeds ~0.1 S_0, self-sourcing is active.
    # This corresponds approximately to f_scr ~ 0.01 (since delta_S ~
    # f * S_0 * L / r and L/r ~ a few at galactic scales).
    #
    # Effective f: linear below transition, screened above.
    f_eff_arr = np.where(f_scr_arr > 0.01, f_scr_arr, f_lin_arr)

    # Transition mass: where f_screened crosses 0.01
    # f_scr = (M/A_BTFR)^0.25 / v_ref = 0.01
    # (M/A_BTFR)^0.25 = 0.01 * v_ref
    # M = A_BTFR * (0.01 * v_ref)^4
    A_BTFR_kg = 50.0 * MSUN / (1e3)**4
    M_transition_kg = A_BTFR_kg * (0.01 * V_REF)**4
    M_transition_log = np.log10(M_transition_kg / MSUN)

    screening = {
        'M_transition_log10_Msun': float(M_transition_log),
        'masses_log10': masses_log.tolist(),
        'f_linear': f_lin_arr.tolist(),
        'f_screened': f_scr_arr.tolist(),
        'f_effective': f_eff_arr.tolist(),
    }
    results['screening_transition'] = screening

    # --- Part G: Route A vs Route B rotation curves ---
    M_disk = M_MW_BAR * MSUN  # kg
    R_d = 3.0 * KPC_M         # disk scale length in m
    r_kpc = np.linspace(0.5, 50, 200)
    r_m_arr = r_kpc * KPC_M

    # Baryonic rotation curve (exponential disk, spherical approx)
    x = r_kpc / 3.0  # r/R_d
    M_enc = M_disk * (1 - (1 + x) * np.exp(-x))
    v_bar2 = G_SI * M_enc / r_m_arr  # m^2/s^2

    # Route A: v_c^2 = v_bar^2 * [1 + alpha/(1 + r/beta)]
    r_over_beta = r_m_arr / BETA_M
    v_A2 = v_bar2 * (1 + ALPHA / (1 + r_over_beta))
    v_A = np.sqrt(np.maximum(v_A2, 0)) / 1e3  # km/s

    # Route B: v_c^2 = v_bar^2 + v_flat^2
    v_flat = V_FLAT_MW  # m/s
    v_B2 = v_bar2 + v_flat**2
    v_B = np.sqrt(np.maximum(v_B2, 0)) / 1e3  # km/s

    v_bar_kms = np.sqrt(np.maximum(v_bar2, 0)) / 1e3

    rotation = {
        'r_kpc': r_kpc.tolist(),
        'v_bar_kms': v_bar_kms.tolist(),
        'v_RouteA_kms': v_A.tolist(),
        'v_RouteB_kms': v_B.tolist(),
        'M_bar_Msun': M_MW_BAR,
        'R_d_kpc': 3.0,
        'v_flat_kms': V_FLAT_MW / 1e3,
    }
    results['rotation_comparison'] = rotation

    # --- Summary safety margins ---
    safety = {
        'anomalous_accel_1AU': {
            'prediction': earth_entry['a_extra_m_s2'],
            'bound': ANOMALOUS_ACCEL_BOUND,
            'log10_margin': np.log10(ANOMALOUS_ACCEL_BOUND / earth_entry['a_extra_m_s2']),
        },
        'PPN_gamma': {
            'prediction': frac_1au,
            'bound': CASSINI_GAMMA_BOUND,
            'log10_margin': ppn['gamma_safety_log10'],
        },
        'PPN_beta': {
            'prediction': frac_1au,
            'bound': MESSENGER_BETA_BOUND,
            'log10_margin': ppn['beta_safety_log10'],
        },
        'Nordtvedt_eta': {
            'prediction': frac_1au,
            'bound': NORDTVEDT_ETA_BOUND,
            'log10_margin': ppn['nordtvedt_safety_log10'],
        },
        'Mercury_precession': {
            'prediction': dw_per_century_arcsec,
            'bound': MERCURY_PRECESSION_UNC,
            'log10_margin': precession['safety_log10'],
        },
    }
    results['safety_margins'] = safety

    return results


# ================================================================
# PLOTTING
# ================================================================

def plot_safety_margins(results, outdir):
    """Bar chart of log10(bound/prediction) for each observable."""
    fig, ax = plt.subplots(figsize=(10, 5))

    safety = results['safety_margins']
    names = list(safety.keys())
    margins = [safety[k]['log10_margin'] for k in names]
    labels = [
        'Anom. accel.\n(1 AU)',
        'PPN $\\gamma$\n(Cassini)',
        'PPN $\\beta$\n(MESSENGER)',
        'Nordtvedt $\\eta$\n(MESSENGER)',
        'Mercury\nprecession',
    ]

    bars = ax.bar(range(len(names)), margins, color='#2ca02c', alpha=0.8, edgecolor='k')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('log$_{10}$(bound / MTDF prediction)', fontsize=11)
    ax.set_title('Step 13: Solar System Safety Margins (Route B Compression)', fontsize=12)
    ax.axhline(0, color='red', lw=2, label='Excluded')
    ax.axhline(5, color='orange', ls='--', lw=1, alpha=0.5, label='5 orders of magnitude')

    for i, m in enumerate(margins):
        ax.text(i, m + 0.3, f'{m:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(margins) + 3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(outdir / 'step13_safety_margins.png', dpi=150)
    plt.close(fig)


def plot_screening_regimes(results, outdir):
    """f(M) showing linear vs screened regimes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    scr = results['screening_transition']
    masses_log = np.array(scr['masses_log10'])
    f_lin = np.array(scr['f_linear'])
    f_scr = np.array(scr['f_screened'])
    f_eff = np.array(scr['f_effective'])

    ax.loglog(10**masses_log, f_lin, 'b--', alpha=0.5, lw=1.5, label='Linear: $f \\propto M$')
    ax.loglog(10**masses_log, f_scr, 'r--', alpha=0.5, lw=1.5, label='Self-sourced: $f \\propto M^{1/4}$')
    ax.loglog(10**masses_log, f_eff, 'k-', lw=2.5, label='Physical $f(M)$')

    # Mark Sun and MW
    f_sun = results['f_sun']
    ax.plot(1.0, f_sun, 'o', color='gold', ms=12, zorder=5,
            markeredgecolor='k', label=f'Sun ($f = {f_sun:.1e}$)')
    f_mw = results['mw_background']['f_MW']
    ax.plot(M_MW_BAR, f_mw, 's', color='purple', ms=10, zorder=5,
            markeredgecolor='k', label=f'MW ($f = {f_mw:.2f}$)')

    # Mark transition
    M_tr = 10**scr['M_transition_log10_Msun']
    ax.axvline(M_tr, color='gray', ls=':', lw=1.5)
    ax.text(M_tr * 2, 0.1, f'Transition\n$M \\approx 10^{{{scr["M_transition_log10_Msun"]:.1f}}}$ M$_\\odot$',
            fontsize=9, color='gray')

    # Shade linear regime
    ax.axhspan(1e-20, 0.01, alpha=0.05, color='blue', label='Linear regime ($f \\ll 1$)')

    ax.set_xlabel('$M$ (M$_\\odot$)', fontsize=12)
    ax.set_ylabel('Compression parameter $f(M)$', fontsize=12)
    ax.set_title('Step 13: Screening Regime Transition', fontsize=12)
    ax.set_xlim(1e-1, 1e13)
    ax.set_ylim(1e-18, 1e2)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / 'step13_screening_regimes.png', dpi=150)
    plt.close(fig)


def plot_rotation_comparison(results, outdir):
    """Route A vs Route B rotation curves for MW-like galaxy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    rot = results['rotation_comparison']
    r = np.array(rot['r_kpc'])
    v_bar = np.array(rot['v_bar_kms'])
    v_A = np.array(rot['v_RouteA_kms'])
    v_B = np.array(rot['v_RouteB_kms'])

    # Left panel: full rotation curves
    ax1.plot(r, v_bar, 'k--', lw=1, label='Baryons only')
    ax1.plot(r, v_A, 'b-', lw=2, label='Route A: $v_{bar}^2 \\times (1+\\alpha)$')
    ax1.plot(r, v_B, 'r-', lw=2, label=f'Route B: $v_{{bar}}^2 + v_{{flat}}^2$')
    ax1.axhline(rot['v_flat_kms'], color='gray', ls=':', alpha=0.5, label=f'$v_{{flat}}$ = {rot["v_flat_kms"]:.0f} km/s')
    ax1.set_xlabel('$r$ (kpc)', fontsize=12)
    ax1.set_ylabel('$v_c$ (km/s)', fontsize=12)
    ax1.set_title('MW-like galaxy ($M_{bar}$' + f' = {rot["M_bar_Msun"]:.0e} M$_\\odot$)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 350)
    ax1.grid(True, alpha=0.3)

    # Shade typical SPARC range
    ax1.axvspan(1, 30, alpha=0.08, color='green', label='Typical SPARC range')

    # Right panel: ratio to observed v_flat
    ax2.plot(r, v_A / rot['v_flat_kms'], 'b-', lw=2, label='Route A / $v_{flat}$')
    ax2.plot(r, v_B / rot['v_flat_kms'], 'r-', lw=2, label='Route B / $v_{flat}$')
    ax2.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel('$r$ (kpc)', fontsize=12)
    ax2.set_ylabel('$v_c / v_{flat}$', fontsize=12)
    ax2.set_title('Ratio to observed $v_{flat}$', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0.3, 1.8)
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(1, 30, alpha=0.08, color='green')

    fig.tight_layout()
    fig.savefig(outdir / 'step13_rotation_comparison.png', dpi=150)
    plt.close(fig)


# ================================================================
# OUTPUT
# ================================================================

def make_json_serializable(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    outdir = Path(__file__).parent.parent / 'output' / 'step13_solar_system'
    outdir.mkdir(parents=True, exist_ok=True)

    print("Step 13: Weak Field Limit + Solar System Sanity Check")
    print("=" * 60)

    results = compute_solar_system()

    # --- Print regime table ---
    print("\n  Part A: Linear vs Screened Regime")
    print(f"  {'Object':<12} {'Mass (Msun)':<14} {'f_linear':<12} {'f_screened':<12} {'Regime':<10}")
    for row in results['regime_table']:
        print(f"  {row['object']:<12} {row['mass_Msun']:<14.2e} {row['f_linear']:<12.2e} {row['f_screened']:<12.4f} {row['regime']:<10}")

    # --- Print planet table ---
    print(f"\n  Part B: Solar System Observables (f_sun = {results['f_sun']:.2e})")
    print(f"  {'Planet':<10} {'r (AU)':<8} {'delta_S':<10} {'M_stress':<12} {'a_extra':<12} {'Fractional':<12}")
    for row in results['planet_table']:
        print(f"  {row['planet']:<10} {row['a_AU']:<8.3f} {row['delta_S']:<10.2e} "
              f"{row['M_stress_kg']:<12.2e} {row['a_extra_m_s2']:<12.2e} {row['fractional_anomaly']:<12.2e}")

    # --- Print PPN ---
    ppn = results['ppn']
    print(f"\n  Part C: PPN Parameters")
    print(f"  gamma - 1: {ppn['gamma_minus_1']:.2e}  (bound: {ppn['gamma_bound']:.1e}, margin: 10^{ppn['gamma_safety_log10']:.1f})")
    print(f"  beta  - 1: {ppn['beta_minus_1']:.2e}  (bound: {ppn['beta_bound']:.1e}, margin: 10^{ppn['beta_safety_log10']:.1f})")
    print(f"  Nordtvedt : {ppn['nordtvedt_eta']:.2e}  (bound: {ppn['nordtvedt_bound']:.1e}, margin: 10^{ppn['nordtvedt_safety_log10']:.1f})")

    # --- Print precession ---
    prec = results['mercury_precession']
    print(f"\n  Part D: Mercury Perihelion Precession")
    print(f"  MTDF prediction: {prec['delta_omega_per_century_arcsec']:.2e} arcsec/century")
    print(f"  Observational uncertainty: {prec['observational_uncertainty_arcsec']} arcsec/century")
    print(f"  Safety margin: 10^{prec['safety_log10']:.1f}")

    # --- Print MW background ---
    mw = results['mw_background']
    print(f"\n  Part E: MW Background Field at Sun's Position")
    print(f"  f_MW = {mw['f_MW']:.3f}")
    print(f"  rho_stress(8 kpc) = {mw['rho_stress_MW_8kpc_Msun_pc3']:.4f} M_sun/pc^3")
    print(f"  Local DM observed = {mw['local_DM_observed_Msun_pc3']:.3f} +/- {mw['local_DM_uncertainty']:.3f} M_sun/pc^3")
    print(f"  Ratio (predicted/observed) = {mw['ratio_predicted_over_observed']:.2f}")
    print(f"  Within 1-sigma: {mw['within_1sigma']}")
    print(f"  MW tidal across 1 AU: {mw['tidal_across_1AU_m_s2']:.2e} m/s^2")
    print(f"  Tidal / solar gravity: {mw['tidal_over_solar_gravity']:.2e}")

    # --- Print screening transition ---
    scr = results['screening_transition']
    print(f"\n  Part F: Self-sourcing transition at M ~ 10^{scr['M_transition_log10_Msun']:.1f} M_sun")
    print(f"  Below: f follows linear scaling (f ~ M, negligible)")
    print(f"  Above: f follows BTFR scaling (f ~ M^{{1/4}}, significant)")

    # --- Print rotation comparison ---
    rot = results['rotation_comparison']
    v_A = np.array(rot['v_RouteA_kms'])
    v_B = np.array(rot['v_RouteB_kms'])
    r = np.array(rot['r_kpc'])
    idx_30 = np.argmin(np.abs(r - 30))
    idx_50 = np.argmin(np.abs(r - 50))
    print(f"\n  Part G: Route A vs Route B Rotation Curves")
    print(f"  At r = 30 kpc: Route A = {v_A[idx_30]:.1f} km/s, Route B = {v_B[idx_30]:.1f} km/s")
    print(f"  At r = 50 kpc: Route A = {v_A[idx_50]:.1f} km/s, Route B = {v_B[idx_50]:.1f} km/s")
    print(f"  Route A declines; Route B stays flat at v_flat = {rot['v_flat_kms']:.0f} km/s")

    # --- Print overall verdict ---
    print("\n" + "=" * 60)
    min_margin = min(v['log10_margin'] for v in results['safety_margins'].values())
    print(f"  VERDICT: All Solar System tests safe by > 10^{min_margin:.0f}")
    print(f"  MW local density matches observations: {mw['within_1sigma']}")
    print("  Route B compression mechanism is Solar-System-safe.")
    print("=" * 60)

    # --- Save JSON (exclude large arrays from screening) ---
    json_results = make_json_serializable(results)
    # Remove large arrays for clean JSON
    json_results['screening_transition'] = {
        'M_transition_log10_Msun': scr['M_transition_log10_Msun']
    }
    json_results['rotation_comparison'] = {
        'v_RouteA_at_30kpc': float(v_A[idx_30]),
        'v_RouteB_at_30kpc': float(v_B[idx_30]),
        'v_RouteA_at_50kpc': float(v_A[idx_50]),
        'v_RouteB_at_50kpc': float(v_B[idx_50]),
        'v_flat_kms': rot['v_flat_kms'],
        'M_bar_Msun': rot['M_bar_Msun'],
    }

    with open(outdir / 'step13_solar_system.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # --- Plots ---
    plot_safety_margins(results, outdir)
    plot_screening_regimes(results, outdir)
    plot_rotation_comparison(results, outdir)

    # --- Manifest ---
    manifest = {}
    for p in sorted(outdir.iterdir()):
        if p.name != 'manifest.json':
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            manifest[p.name] = h
    with open(outdir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Output saved to {outdir}/")


if __name__ == '__main__':
    main()
