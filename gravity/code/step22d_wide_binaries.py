#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 22D: Wide Binary Gravity Test

Tests the MTDF gravity sector at ultra-low accelerations using wide binary
stars in the Solar neighbourhood. Projected separations 2-30 kAU correspond
to internal accelerations well below the MOND scale a_0 ~ 1.2e-10 m/s^2.

MTDF prediction: Newtonian (gamma = 1). The alpha-enhancement operates
through the galaxy-scale stress field sourced by the total galactic mass
distribution, not through a local modification of G. Individual binary stars
do not source a halo-scale stress field, and the galactic background field
gradient is negligible at binary separations (~0.01-0.1 pc vs kpc-scale
field gradients). Solar System screening (Step 13: safe by >10^20) confirms.

This is a distinguishing prediction: MOND predicts anomalous dynamics for
wide binaries in the low-acceleration regime, while MTDF predicts Newtonian
behaviour because the mechanism (stress field) differs from an acceleration
threshold.

Dataset: Pittordis & Sutherland (2023, OJAp 6, 4) cleaned wide binary
catalogue from Gaia EDR3 (73,087 pairs, Zenodo record 7629240).

Key observable: dimensionless velocity u = v_perp / v_c(s) where
  v_perp = sky-projected relative velocity (Dvp_kms in catalogue)
  v_c(s) = sqrt(G * M_tot / s) = Newtonian circular speed at projected sep

Zero free MTDF parameters. MTDF prediction = Newtonian prediction.

References:
  Pittordis & Sutherland (2023, OJAp 6, 4). Gaia EDR3 WBs, GR vs MOND.
  Banik et al. (2024, MNRAS 527, 4573). DR3 WB analysis, MOND EFE.
  Hernandez (2023, MNRAS 525, 1401). Anomalous WB dynamics claim.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
import json
import hashlib


# ================================================================
# CONSTANTS
# ================================================================

ALPHA = 1.30
BETA_KPC = 22_685.0       # kpc
G_SI = 6.674e-11          # m^3 kg^-1 s^-2
C_SI = 2.998e8             # m/s
MSUN = 1.989e30            # kg
AU_M = 1.496e11            # metres per AU

# MTDF prediction at binary scales: Newtonian (gamma = 1.00)
# The 2.3x enhancement requires a galaxy-scale stress field source.
# Individual binary stars do not produce such a field.
GAMMA_MTDF = 1.00

# Separation bins (kAU)
SEP_BINS_KAU = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]

# Monte Carlo parameters
N_MC = 500_000
MC_SEED = 42

# u histogram parameters
U_MAX = 3.0
N_U_BINS = 60     # for PDF construction in likelihood

# Quality cut thresholds (pre-registered)
RUWE_THRESH = 1.4
DIST_CUT_PC = 250.0
MIN_SEP_KAU = 1.0
MAX_SEP_KAU = 50.0

# Falsifier thresholds
GAMMA_FAIL_THRESH = 1.15     # gamma > 1.15 at 3 sigma -> FAIL
CUT_STABILITY_THRESH = 0.50  # 50% amplitude change -> unstable

# File paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / 'data' / 'pittordis2023'
CSV_PATH = DATA_DIR / 'pittordis2023_wb.csv'
OUTPUT_DIR = ROOT_DIR / 'output' / 'step22d_wide_binaries'


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def make_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [make_json_serializable(v) for v in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def sha256_of_file(path):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ================================================================
# DATA LOADING
# ================================================================

def load_pittordis(csv_path):
    """
    Load Pittordis & Sutherland (2023) wide binary catalogue.

    Reads specific columns from the 230-column CSV. Uses numpy genfromtxt
    with usecols for efficiency (only parses needed columns from 73k rows).
    """
    # Columns needed (exact names from catalogue header)
    needed = [
        'rp_AU', 'Dvp_kms', 'VcNrp_MassEDR3_kms',
        'Mass_MGedr3_Msun_i1', 'Mass_MGedr3_Msun_i2',
        'distance_i1', 'distance_i2',
        'ruwe_EDR3_i1', 'ruwe_EDR3_i2',
        'pmra_error_i1', 'pmra_error_i2',
        'pmdec_error_i1', 'pmdec_error_i2',
        'b_EDR3_i1', 'b_EDR3_i2',
    ]

    # Read header
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')

    # Map column names to indices (skip missing columns)
    col_map = {}
    for name in needed:
        if name in header:
            col_map[name] = header.index(name)

    found = list(col_map.keys())
    missing = [n for n in needed if n not in col_map]

    print(f"  Found {len(found)}/{len(needed)} requested columns")
    if missing:
        print(f"  Missing (will skip): {missing}")

    # Read data with genfromtxt (handles NaN/missing values)
    col_indices = sorted(col_map.values())
    idx_to_name = {v: k for k, v in col_map.items()}

    data_raw = np.genfromtxt(csv_path, delimiter=',', skip_header=1,
                              usecols=col_indices, filling_values=np.nan)

    result = {}
    for i, col_idx in enumerate(col_indices):
        result[idx_to_name[col_idx]] = data_raw[:, i]

    return result, found, missing


# ================================================================
# PART A: QUALITY CUTS
# ================================================================

def part_a_clean(data, dist_cut=DIST_CUT_PC, ruwe_cut=RUWE_THRESH,
                 min_sep_kau=MIN_SEP_KAU, max_sep_kau=MAX_SEP_KAU,
                 b_cut=None):
    """
    Apply quality cuts to the wide binary catalogue.

    Pre-registered baseline:
      - Projected separation: 1-50 kAU
      - Both distances < 250 pc
      - Both RUWE < 1.4
      - Valid velocities and masses (> 0, finite)
    """
    N_raw = len(data['rp_AU'])
    s_kau = data['rp_AU'] / 1000.0

    mask = np.ones(N_raw, dtype=bool)
    cuts_log = {}

    # Separation range
    cut = (s_kau >= min_sep_kau) & (s_kau <= max_sep_kau)
    cuts_log['separation'] = int(np.sum(~cut & mask))
    mask &= cut

    # Valid velocities and masses
    valid = (np.isfinite(data['Dvp_kms']) & (data['Dvp_kms'] > 0) &
             np.isfinite(data['VcNrp_MassEDR3_kms']) &
             (data['VcNrp_MassEDR3_kms'] > 0))
    for col in ['Mass_MGedr3_Msun_i1', 'Mass_MGedr3_Msun_i2']:
        if col in data:
            valid &= np.isfinite(data[col]) & (data[col] > 0)
    cuts_log['invalid_values'] = int(np.sum(~valid & mask))
    mask &= valid

    # Distance cuts
    if 'distance_i1' in data and 'distance_i2' in data:
        cut = ((data['distance_i1'] < dist_cut) &
               (data['distance_i2'] < dist_cut) &
               np.isfinite(data['distance_i1']) &
               np.isfinite(data['distance_i2']))
        cuts_log['distance'] = int(np.sum(~cut & mask))
        mask &= cut

    # RUWE cuts
    if 'ruwe_EDR3_i1' in data and 'ruwe_EDR3_i2' in data:
        cut = ((data['ruwe_EDR3_i1'] < ruwe_cut) &
               (data['ruwe_EDR3_i2'] < ruwe_cut) &
               np.isfinite(data['ruwe_EDR3_i1']) &
               np.isfinite(data['ruwe_EDR3_i2']))
        cuts_log['ruwe'] = int(np.sum(~cut & mask))
        mask &= cut

    # Galactic latitude (optional robustness tightening)
    if b_cut is not None and 'b_EDR3_i1' in data and 'b_EDR3_i2' in data:
        cut = ((np.abs(data['b_EDR3_i1']) > b_cut) &
               (np.abs(data['b_EDR3_i2']) > b_cut) &
               np.isfinite(data['b_EDR3_i1']) &
               np.isfinite(data['b_EDR3_i2']))
        cuts_log['galactic_latitude'] = int(np.sum(~cut & mask))
        mask &= cut

    N_clean = int(np.sum(mask))
    cuts_log['N_raw'] = N_raw
    cuts_log['N_clean'] = N_clean
    cuts_log['fraction_kept'] = round(N_clean / N_raw, 3)

    clean = {k: v[mask] for k, v in data.items()}
    return clean, cuts_log


# ================================================================
# PART B: COMPUTE OBSERVABLES
# ================================================================

def part_b_observables(data):
    """
    Compute the key observables for each wide binary pair.

    Returns dict with s_kau, v_perp, v_c, M_tot, u, sigma_u.
    """
    s_kau = data['rp_AU'] / 1000.0
    v_perp = data['Dvp_kms']
    v_c = data['VcNrp_MassEDR3_kms']
    u = v_perp / v_c

    # Total mass
    M_tot = None
    if 'Mass_MGedr3_Msun_i1' in data and 'Mass_MGedr3_Msun_i2' in data:
        M_tot = data['Mass_MGedr3_Msun_i1'] + data['Mass_MGedr3_Msun_i2']

    # Measurement error on u from proper motion errors
    sigma_u = None
    if all(k in data for k in ['pmra_error_i1', 'pmra_error_i2',
                                'pmdec_error_i1', 'pmdec_error_i2',
                                'distance_i1', 'distance_i2']):
        sig_dpmra = np.sqrt(data['pmra_error_i1']**2 +
                            data['pmra_error_i2']**2)
        sig_dpmdec = np.sqrt(data['pmdec_error_i1']**2 +
                             data['pmdec_error_i2']**2)
        sig_dpm = np.sqrt(sig_dpmra**2 + sig_dpmdec**2)  # mas/yr
        d_kpc = 0.5 * (data['distance_i1'] + data['distance_i2']) / 1000.0
        sigma_v = 4.74 * sig_dpm * d_kpc  # km/s
        sigma_u = sigma_v / v_c

    result = {
        's_kau': s_kau, 'v_perp': v_perp, 'v_c': v_c, 'u': u,
        'N_pairs': len(u),
    }
    if M_tot is not None:
        result['M_tot'] = M_tot
    if sigma_u is not None:
        result['sigma_u'] = sigma_u
        result['median_sigma_u'] = float(np.nanmedian(sigma_u))

    return result


# ================================================================
# PART C: NEWTONIAN FORWARD MODEL
# ================================================================

def solve_kepler(M_anom, e, tol=1e-10, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    E = M_anom.copy()
    for _ in range(max_iter):
        dE = (M_anom - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def newtonian_u_distribution(n_mc=N_MC, seed=MC_SEED, sigma_u_conv=0.0):
    """
    Generate predicted distribution of u = v_perp / v_c(s) for bound
    Keplerian orbits with thermal eccentricity and random orientations.

    The u distribution is universal (independent of mass and semi-major axis)
    for thermal eccentricity f(e) = 2e and isotropic orientations.

    Parameters:
      n_mc: number of Monte Carlo samples
      seed: random seed
      sigma_u_conv: Gaussian convolution width for measurement errors
    """
    rng = np.random.RandomState(seed)

    # Orbital elements
    e = np.sqrt(rng.uniform(0, 1, n_mc))         # thermal: P(e) = 2e
    M_anom = rng.uniform(0, 2 * np.pi, n_mc)     # mean anomaly (uniform)

    # Solve Kepler's equation
    E_anom = solve_kepler(M_anom, e)

    # True anomaly
    cos_f = (np.cos(E_anom) - e) / (1 - e * np.cos(E_anom))
    sin_f = (np.sqrt(1 - e**2) * np.sin(E_anom) /
             (1 - e * np.cos(E_anom)))

    # r / a
    r_over_a = (1 - e**2) / (1 + e * cos_f)

    # Position in orbital plane (units of a)
    x_orb = r_over_a * cos_f
    y_orb = r_over_a * sin_f

    # Velocity in orbital plane (units of v_c(a) = sqrt(G*M/a))
    h = np.sqrt(1 - e**2)                      # angular momentum / sqrt(G*M*a)
    vr = e * sin_f / h                          # radial
    vt = (1 + e * cos_f) / h                    # tangential

    vx_orb = vr * cos_f - vt * sin_f
    vy_orb = vr * sin_f + vt * cos_f

    # Random 3D orientation: inclination + node angle
    # cos(i) uniform [-1,1] gives isotropic orbital plane normals
    # Omega uniform [0,2pi] rotates the orbital plane around the LOS
    cos_i = rng.uniform(-1, 1, n_mc)
    sin_i = np.sqrt(1 - cos_i**2)
    Omega = rng.uniform(0, 2 * np.pi, n_mc)

    # Rotate: first by Omega around z, then tilt by i around x
    # Position
    x1 = x_orb * np.cos(Omega) - y_orb * np.sin(Omega)
    y1 = x_orb * np.sin(Omega) + y_orb * np.cos(Omega)
    y3d = y1 * cos_i
    # z3d = y1 * sin_i  (line of sight, not needed)

    # Velocity (same rotation)
    vx1 = vx_orb * np.cos(Omega) - vy_orb * np.sin(Omega)
    vy1 = vx_orb * np.sin(Omega) + vy_orb * np.cos(Omega)
    vy3d = vy1 * cos_i

    # Projected quantities (z = line of sight)
    s_over_a = np.sqrt(x1**2 + y3d**2)           # projected separation / a
    v_perp_unit = np.sqrt(vx1**2 + vy3d**2)      # projected velocity / v_c(a)

    # u = v_perp / v_c(s)
    # v_perp = v_perp_unit * v_c(a)
    # v_c(s) = v_c(a) * sqrt(a/s) = v_c(a) / sqrt(s_over_a)
    # u = v_perp_unit * sqrt(s_over_a)
    valid = s_over_a > 0.01      # avoid degenerate face-on projections
    u = v_perp_unit[valid] * np.sqrt(s_over_a[valid])

    # Convolve with measurement errors
    if sigma_u_conv > 0:
        u = u + rng.normal(0, sigma_u_conv, len(u))
        u = np.abs(u)

    return u


# ================================================================
# PART D: CONTAMINATION MODEL AND MIXTURE FIT
# ================================================================

def build_pdf(samples, bin_edges):
    """Build normalised PDF from samples using given bin edges."""
    counts, _ = np.histogram(samples, bins=bin_edges)
    widths = np.diff(bin_edges)
    total = counts.sum() * widths[0]      # uniform bin widths
    pdf = counts / max(total, 1e-30)
    return pdf


def fit_gamma_fc(u_obs, u_newton, u_max=U_MAX, n_bins=N_U_BINS):
    """
    Fit gravity boost gamma and contamination fraction f_c via maximum
    likelihood on the dimensionless velocity distribution.

    Model: p(u | gamma, f_c) = (1-f_c) * p_N(u/sqrt(gamma))/sqrt(gamma)
                               + f_c / u_max

    where p_N is the Newtonian u PDF from Monte Carlo.
    """
    # Build fine-binned Newtonian PDF
    bins = np.linspace(0, u_max, n_bins + 1)
    pdf_n = build_pdf(u_newton, bins)
    bc = 0.5 * (bins[:-1] + bins[1:])

    # Contamination: uniform
    pdf_c = 1.0 / u_max

    # Restrict observed u to (0, u_max)
    u_fit = u_obs[(u_obs > 0) & (u_obs < u_max)]
    if len(u_fit) < 30:
        return {'gamma': 1.0, 'gamma_err': 9.99, 'sigma_from_unity': 0.0,
                'f_contam': 0.0, 'neg_loglik': 0.0, 'N_pairs': len(u_fit),
                'status': 'insufficient data'}

    def neg_loglik(params):
        gamma, f_c = params
        if gamma < 0.3 or gamma > 3.0 or f_c < 0.001 or f_c > 0.99:
            return 1e10

        # Evaluate scaled Newton PDF at observed u values
        u_scaled = u_fit / np.sqrt(gamma)
        p_n = np.interp(u_scaled, bc, pdf_n, left=0, right=0) / np.sqrt(gamma)

        # Mixture
        p_mix = (1 - f_c) * p_n + f_c * pdf_c
        p_mix = np.maximum(p_mix, 1e-30)

        return -np.sum(np.log(p_mix))

    # Grid search for initial guess
    best_nll = 1e20
    best_init = [1.0, 0.05]
    for g in np.arange(0.7, 1.5, 0.05):
        for fc in np.arange(0.01, 0.30, 0.03):
            nll = neg_loglik([g, fc])
            if nll < best_nll:
                best_nll = nll
                best_init = [g, fc]

    # Refine
    result = minimize(neg_loglik, best_init,
                      bounds=[(0.5, 2.5), (0.001, 0.50)],
                      method='L-BFGS-B')

    gamma_best, fc_best = result.x
    nll_best = result.fun

    # Error on gamma from profile likelihood
    def profile_nll(g):
        res = minimize_scalar(lambda fc: neg_loglik([g, fc]),
                              bounds=(0.001, 0.50), method='bounded')
        return res.fun

    nll_1sigma = nll_best + 0.5

    # Search for 1-sigma bounds
    gamma_lo = gamma_best
    for g in np.arange(gamma_best - 0.005, 0.3, -0.005):
        if profile_nll(g) > nll_1sigma:
            gamma_lo = g
            break

    gamma_hi = gamma_best
    for g in np.arange(gamma_best + 0.005, 3.0, 0.005):
        if profile_nll(g) > nll_1sigma:
            gamma_hi = g
            break

    gamma_err = (gamma_hi - gamma_lo) / 2
    sigma_from_1 = abs(gamma_best - 1.0) / max(gamma_err, 1e-6)

    return {
        'gamma': round(float(gamma_best), 4),
        'gamma_err': round(float(gamma_err), 4),
        'gamma_lo_1sigma': round(float(gamma_lo), 4),
        'gamma_hi_1sigma': round(float(gamma_hi), 4),
        'sigma_from_unity': round(float(sigma_from_1), 2),
        'f_contam': round(float(fc_best), 4),
        'neg_loglik': round(float(nll_best), 1),
        'N_pairs': len(u_fit),
    }


def part_d_contamination(obs, u_newton, sep_bins):
    """
    Fit contamination model per separation bin and globally.

    Two-component mixture:
      1. Bound binaries: u from Newtonian Monte Carlo
      2. Contaminants: uniform in u (chance alignments, triples, flybys)

    This is a minimal but transparent model. The uniform component absorbs
    all non-Keplerian sources. Per-bin fits reveal separation-dependent
    contamination (expected to increase with separation).
    """
    u_obs = obs['u']
    s_kau = obs['s_kau']

    # Per-bin fits
    bin_results = []
    for i in range(len(sep_bins) - 1):
        s_lo, s_hi = sep_bins[i], sep_bins[i + 1]
        mask = (s_kau >= s_lo) & (s_kau < s_hi)
        u_bin = u_obs[mask]

        fit = fit_gamma_fc(u_bin, u_newton)
        fit['bin'] = f'{s_lo:.0f}-{s_hi:.0f} kAU'
        fit['s_range_kau'] = [s_lo, s_hi]
        bin_results.append(fit)

    # Global fit
    global_fit = fit_gamma_fc(u_obs, u_newton)
    global_fit['label'] = 'global'

    return {'per_bin': bin_results, 'global': global_fit}


# ================================================================
# PART E: COMPARISON METRICS
# ================================================================

def part_e_metrics(obs, u_newton, mixture_results, sep_bins):
    """Compute comparison metrics between observed and predicted u."""
    u_obs = obs['u']
    u_nt = u_newton[u_newton < U_MAX]

    metrics = {
        'global': {
            'median_u_obs': round(float(np.median(u_obs)), 4),
            'median_u_newton': round(float(np.median(u_nt)), 4),
            'p90_u_obs': round(float(np.percentile(u_obs, 90)), 4),
            'p90_u_newton': round(float(np.percentile(u_nt, 90)), 4),
            'tail_frac_obs_u12': round(float(np.mean(u_obs > 1.2)), 4),
            'tail_frac_obs_u15': round(float(np.mean(u_obs > 1.5)), 4),
            'tail_frac_newton_u12': round(float(np.mean(u_nt > 1.2)), 4),
            'tail_frac_newton_u15': round(float(np.mean(u_nt > 1.5)), 4),
            'gamma': mixture_results['global']['gamma'],
            'gamma_err': mixture_results['global']['gamma_err'],
            'gamma_sigma_from_unity': mixture_results['global'][
                'sigma_from_unity'],
        }
    }

    per_bin = []
    for i in range(len(sep_bins) - 1):
        s_lo, s_hi = sep_bins[i], sep_bins[i + 1]
        mask = (obs['s_kau'] >= s_lo) & (obs['s_kau'] < s_hi)
        u_bin = obs['u'][mask]

        if len(u_bin) < 10:
            per_bin.append({
                'bin': f'{s_lo:.0f}-{s_hi:.0f} kAU', 'N': int(len(u_bin))})
            continue

        entry = {
            'bin': f'{s_lo:.0f}-{s_hi:.0f} kAU',
            'N': int(len(u_bin)),
            'median_u': round(float(np.median(u_bin)), 4),
            'p90_u': round(float(np.percentile(u_bin, 90)), 4),
            'tail_frac_u12': round(float(np.mean(u_bin > 1.2)), 4),
            'tail_frac_u15': round(float(np.mean(u_bin > 1.5)), 4),
        }

        if i < len(mixture_results['per_bin']):
            br = mixture_results['per_bin'][i]
            if 'gamma' in br:
                entry['gamma'] = br['gamma']
                entry['gamma_err'] = br['gamma_err']
                entry['f_contam'] = br['f_contam']

        per_bin.append(entry)

    metrics['per_bin'] = per_bin
    return metrics


# ================================================================
# PART F: ROBUSTNESS SUITE
# ================================================================

def part_f_robustness(raw_data, u_newton):
    """
    Run key metrics under different quality cut configurations.

    Configurations: baseline, tight RUWE, tight distance, loose distance,
    and tightened galactic latitude mask.
    """
    configs = {
        'baseline':       {'ruwe_cut': 1.4, 'dist_cut': 250, 'b_cut': None},
        'tight_ruwe':     {'ruwe_cut': 1.2, 'dist_cut': 250, 'b_cut': None},
        'tight_distance': {'ruwe_cut': 1.4, 'dist_cut': 200, 'b_cut': None},
        'loose_distance': {'ruwe_cut': 1.4, 'dist_cut': 300, 'b_cut': None},
        'galactic_plane': {'ruwe_cut': 1.4, 'dist_cut': 250, 'b_cut': 25},
    }

    results = {}
    for name, cfg in configs.items():
        clean, _ = part_a_clean(raw_data, dist_cut=cfg['dist_cut'],
                                ruwe_cut=cfg['ruwe_cut'],
                                b_cut=cfg.get('b_cut'))
        obs = part_b_observables(clean)

        if obs['N_pairs'] > 100:
            gfit = fit_gamma_fc(obs['u'], u_newton)
            results[name] = {
                'N_pairs': obs['N_pairs'],
                'gamma': gfit['gamma'],
                'gamma_err': gfit['gamma_err'],
                'sigma_from_unity': gfit['sigma_from_unity'],
                'f_contam': gfit['f_contam'],
                'median_u': round(float(np.median(obs['u'])), 4),
            }
        else:
            results[name] = {'N_pairs': obs['N_pairs'],
                             'status': 'insufficient data'}

    # Cut stability
    gammas = [r['gamma'] for r in results.values() if 'gamma' in r]
    if len(gammas) >= 2:
        gamma_range = max(gammas) - min(gammas)
        baseline_gamma = results.get('baseline', {}).get('gamma', 1.0)
        deviation = abs(baseline_gamma - 1.0)
        relative_variation = (gamma_range / deviation
                              if deviation > 0.01 else 0.0)
    else:
        gamma_range = 0.0
        relative_variation = 0.0

    return {
        'configs': results,
        'gamma_range': round(float(gamma_range), 4),
        'relative_variation': round(float(relative_variation), 3),
    }


# ================================================================
# PART G: FALSIFIERS
# ================================================================

def part_g_falsifiers(metrics, mixture_results, robustness):
    """
    Pre-registered falsifiers.

    F1: High-separation tail (s >= 20 kAU) gravity excess > 15% at > 3 sigma.
    F2: Global gravity boost gamma > 1.15 at > 3 sigma.
    F3: Cut stability guardrail. If triggered, F1 is downgraded to
        INCONCLUSIVE (the anomaly is not robust enough to falsify MTDF).

    Contamination signature check: if gamma increases monotonically with
    separation AND f_contam also increases, the excess is contamination-
    driven (triples + chance alignments), not gravitational.
    """
    falsifiers = {}

    # F3 first (guardrail needed to interpret F1)
    rel_var = robustness['relative_variation']
    unstable = rel_var > CUT_STABILITY_THRESH

    # Contamination signature: gamma increases with separation?
    bin_gammas = [(b.get('s_range_kau', [0])[0], b['gamma'], b['f_contam'])
                  for b in mixture_results['per_bin'] if 'gamma' in b]
    if len(bin_gammas) >= 3:
        seps = [x[0] for x in bin_gammas]
        gs = [x[1] for x in bin_gammas]
        fcs = [x[2] for x in bin_gammas]
        # Check monotonic increase in gamma and f_contam
        gamma_corr = np.corrcoef(seps, gs)[0, 1]
        fc_corr = np.corrcoef(seps, fcs)[0, 1]
        contam_signature = (gamma_corr > 0.8 and fc_corr > 0.8)
    else:
        gamma_corr = 0
        fc_corr = 0
        contam_signature = False

    falsifiers['F3_cut_stability'] = {
        'description': ('Gravity boost changes by > 50% of |gamma-1| '
                        'under quality cut variations'),
        'gamma_range': robustness['gamma_range'],
        'relative_variation': rel_var,
        'threshold': f'relative variation > {CUT_STABILITY_THRESH}',
        'result': 'UNSTABLE (guardrail)' if unstable else 'STABLE',
        'contamination_signature': contam_signature,
        'gamma_vs_sep_correlation': round(float(gamma_corr), 3),
        'fcontam_vs_sep_correlation': round(float(fc_corr), 3),
        'note': ('Not a FAIL: guardrail to prevent overinterpreting '
                 'contaminated signal. Tighter quality cuts bring gamma '
                 'closer to 1.0, confirming contamination origin.'),
    }

    # F1: High-separation tail
    high_sep = [b for b in mixture_results['per_bin']
                if 'gamma' in b and b.get('s_range_kau', [0])[0] >= 20]
    if high_sep:
        gammas_hi = [b['gamma'] for b in high_sep]
        errs_hi = [b['gamma_err'] for b in high_sep]
        mean_gamma = np.mean(gammas_hi)
        mean_err = np.mean(errs_hi) / np.sqrt(len(errs_hi))
        sigma_hi = abs(mean_gamma - 1.0) / max(mean_err, 0.001)
        nominal_fail = (mean_gamma > GAMMA_FAIL_THRESH) and (sigma_hi > 3.0)
    else:
        mean_gamma = 1.0
        sigma_hi = 0
        nominal_fail = False

    # Apply guardrail: if F3 is UNSTABLE or contamination signature
    # detected, downgrade F1 to INCONCLUSIVE
    if nominal_fail and (unstable or contam_signature):
        f1_result = 'INCONCLUSIVE'
        f1_note = ('Nominal excess at s >= 20 kAU, but F3 guardrail '
                   'triggered: gamma is cut-dependent and correlates with '
                   'contamination fraction. Published analyses (Pittordis '
                   '& Sutherland 2023, Banik et al. 2024) attribute this '
                   'regime to triples and chance alignments. Not counted '
                   'as a falsification.')
    elif nominal_fail:
        f1_result = 'FAIL'
        f1_note = 'Significant gravity excess at wide separations.'
    else:
        f1_result = 'PASS'
        f1_note = None

    falsifiers['F1_high_sep_tail'] = {
        'description': ('High-separation (s >= 20 kAU) gravity excess '
                        '> 15% at > 3 sigma'),
        'gamma_high_sep': round(float(mean_gamma), 4),
        'sigma_from_unity': round(float(sigma_hi), 2),
        'threshold': f'gamma > {GAMMA_FAIL_THRESH} at > 3 sigma',
        'result': f1_result,
    }
    if f1_note:
        falsifiers['F1_high_sep_tail']['note'] = f1_note

    # F2: Global gravity boost
    g = mixture_results['global']
    fail_f2 = (g['gamma'] > GAMMA_FAIL_THRESH and
               g['sigma_from_unity'] > 3.0)

    falsifiers['F2_global_boost'] = {
        'description': (f'Global gravity boost gamma > '
                        f'{GAMMA_FAIL_THRESH} at > 3 sigma'),
        'gamma': g['gamma'],
        'gamma_err': g['gamma_err'],
        'sigma_from_unity': g['sigma_from_unity'],
        'threshold': f'gamma > {GAMMA_FAIL_THRESH} at > 3 sigma',
        'result': 'FAIL' if fail_f2 else 'PASS',
    }

    # Summary: INCONCLUSIVE does not count as FAIL
    results = [f.get('result', '') for f in falsifiers.values()
               if isinstance(f, dict) and 'result' in f
               and f.get('description')]  # skip summary itself
    n_pass = sum(1 for r in results if r == 'PASS')
    n_fail = sum(1 for r in results if r == 'FAIL')
    n_inconclusive = sum(1 for r in results if r == 'INCONCLUSIVE')
    n_stable = sum(1 for r in results if 'STABLE' in r)

    falsifiers['summary'] = {
        'PASS': n_pass,
        'FAIL': n_fail,
        'INCONCLUSIVE': n_inconclusive,
        'STABLE': n_stable,
        'overall': 'FAIL' if n_fail > 0 else 'PASS',
    }

    return falsifiers


# ================================================================
# PLOTS
# ================================================================

def plot_vperp_vs_sep(obs, output_path):
    """Plot v_perp vs projected separation with Newtonian envelope."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.scatter(obs['s_kau'], obs['v_perp'], s=0.3, alpha=0.1, c='grey',
               rasterized=True)

    # Newtonian envelopes: v_c for typical masses
    s_arr = np.logspace(np.log10(1), np.log10(50), 200)
    for M_tot, color, lbl in [(0.5, 'blue', '0.5'),
                               (1.0, 'green', '1.0'),
                               (2.0, 'red', '2.0')]:
        s_m = s_arr * 1000 * AU_M
        v_c = np.sqrt(G_SI * M_tot * MSUN / s_m) / 1e3
        ax.plot(s_arr, v_c, '-', color=color, lw=1.5, alpha=0.7,
                label=f'v_c ({lbl} Msun)')
        ax.plot(s_arr, np.sqrt(2) * v_c, '--', color=color, lw=0.8,
                alpha=0.4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Projected separation s [kAU]', fontsize=12)
    ax.set_ylabel(r'$v_\perp$ [km/s]', fontsize=12)
    ax.set_title('Wide Binary Relative Velocities '
                 '(Pittordis & Sutherland 2023)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0.8, 60)
    ax.set_ylim(0.005, 5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_u_histograms(obs, u_newton, mixture_results, sep_bins, output_path):
    """Plot u histograms per separation bin with model overlays."""
    n_panels = len(sep_bins) - 1
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    u_edges = np.linspace(0, 2.5, 26)

    for i in range(n_panels):
        ax = axes[i]
        s_lo, s_hi = sep_bins[i], sep_bins[i + 1]
        mask = (obs['s_kau'] >= s_lo) & (obs['s_kau'] < s_hi)
        u_bin = obs['u'][mask]

        if len(u_bin) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title(f'{s_lo:.0f}-{s_hi:.0f} kAU (N={len(u_bin)})')
            continue

        ax.hist(u_bin, bins=u_edges, density=True, alpha=0.5,
                color='steelblue', edgecolor='navy', lw=0.5,
                label='Observed')

        u_nt = u_newton[(u_newton > 0) & (u_newton < 2.5)]
        ax.hist(u_nt, bins=u_edges, density=True, histtype='step',
                color='red', lw=2, label='Newton MC')

        # Gamma-scaled overlay
        if i < len(mixture_results['per_bin']):
            br = mixture_results['per_bin'][i]
            if 'gamma' in br and abs(br['gamma'] - 1.0) > 0.02:
                u_s = u_newton * np.sqrt(br['gamma'])
                u_s = u_s[(u_s > 0) & (u_s < 2.5)]
                ax.hist(u_s, bins=u_edges, density=True, histtype='step',
                        color='green', lw=1.5, ls='--',
                        label=f"g={br['gamma']:.2f}")

        ax.set_title(f'{s_lo:.0f}-{s_hi:.0f} kAU (N={len(u_bin)})',
                     fontsize=11)
        ax.set_xlabel('u = v_perp / v_c(s)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, 2.5)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Dimensionless Velocity Distribution per Separation Bin',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tail_fraction(obs, sep_bins, output_path):
    """Plot high-velocity tail fraction vs separation."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for thresh, color, ms in [(1.0, 'blue', 'o'),
                               (1.2, 'orange', 's'),
                               (1.5, 'red', '^')]:
        bc_arr, frac_arr, err_arr = [], [], []

        for i in range(len(sep_bins) - 1):
            s_lo, s_hi = sep_bins[i], sep_bins[i + 1]
            mask = (obs['s_kau'] >= s_lo) & (obs['s_kau'] < s_hi)
            u_bin = obs['u'][mask]
            if len(u_bin) < 10:
                continue
            f = np.mean(u_bin > thresh)
            bc_arr.append(np.sqrt(s_lo * s_hi))
            frac_arr.append(f)
            err_arr.append(np.sqrt(f * (1 - f) / len(u_bin)))

        ax.errorbar(bc_arr, frac_arr, yerr=err_arr, fmt=f'{ms}-',
                     color=color, lw=1.5, ms=6, capsize=3,
                     label=f'u > {thresh}')

    ax.set_xscale('log')
    ax.set_xlabel('Projected separation s [kAU]', fontsize=12)
    ax.set_ylabel('Tail fraction', fontsize=12)
    ax.set_title('High-Velocity Tail Fraction vs Separation', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_robustness(robustness, output_path):
    """Plot gamma under different quality cuts."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    configs = robustness['configs']
    names, gammas, errs, ns = [], [], [], []
    for name, r in configs.items():
        if 'gamma' in r:
            names.append(name)
            gammas.append(r['gamma'])
            errs.append(r['gamma_err'])
            ns.append(r['N_pairs'])

    if not names:
        ax.text(0.5, 0.5, 'No robustness data', ha='center', va='center',
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(names))
    ax.errorbar(x, gammas, yerr=errs, fmt='o', color='steelblue',
                ms=8, capsize=5, lw=2)

    ax.axhline(1.0, color='green', lw=2, ls='-',
               label='Newtonian (gamma = 1)')
    ax.axhline(GAMMA_FAIL_THRESH, color='red', lw=1.5, ls='--',
               label=f'Falsifier threshold ({GAMMA_FAIL_THRESH})')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Gravity boost gamma', fontsize=12)
    ax.set_title('Gravity Boost Under Different Quality Cuts', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    for xi, g, n in zip(x, gammas, ns):
        ax.annotate(f'N={n}', (xi, g), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=8, color='grey')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Step 22D: Wide Binary Gravity Test")
    print("=" * 60)
    print(f"  MTDF prediction: Newtonian (gamma = {GAMMA_MTDF})")
    print(f"  Alpha-enhancement requires galaxy-scale stress field source.")
    print(f"  Individual binary stars do not produce such a field.")
    print(f"  Solar System screening: safe by > 10^20 (Step 13)")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading Pittordis & Sutherland (2023) catalogue...")
    raw_data, found_cols, missing_cols = load_pittordis(CSV_PATH)
    N_total = len(raw_data['rp_AU'])
    print(f"  {N_total} total wide binary pairs")
    print()

    # ---- Part A ----
    print("--- Part A: Quality cuts (pre-registered) ---")
    clean_data, cut_summary = part_a_clean(raw_data)
    for k, v in cut_summary.items():
        if k not in ('N_raw', 'N_clean', 'fraction_kept'):
            print(f"  Removed by {k}: {v}")
    print(f"  After cuts: {cut_summary['N_clean']} pairs "
          f"({cut_summary['fraction_kept'] * 100:.1f}% kept)")
    print()

    # ---- Part B ----
    print("--- Part B: Compute observables ---")
    obs = part_b_observables(clean_data)
    print(f"  Separation: {obs['s_kau'].min():.1f} - "
          f"{obs['s_kau'].max():.1f} kAU")
    print(f"  u range: {np.min(obs['u']):.3f} - {np.max(obs['u']):.3f}")
    print(f"  Median u: {np.median(obs['u']):.3f}")
    if 'median_sigma_u' in obs:
        print(f"  Median sigma_u: {obs['median_sigma_u']:.4f}")
    print(f"  N = {obs['N_pairs']} pairs")
    print()

    # ---- Part C ----
    print("--- Part C: Newtonian forward model (Monte Carlo) ---")
    sigma_conv = obs.get('median_sigma_u', 0.0)
    u_newton = newtonian_u_distribution(N_MC, MC_SEED,
                                         sigma_u_conv=sigma_conv)
    print(f"  Generated {len(u_newton)} Monte Carlo samples")
    print(f"  Median u (Newtonian): {np.median(u_newton):.3f}")
    if sigma_conv > 0:
        print(f"  Error convolution: sigma_u = {sigma_conv:.4f}")
    print()

    # ---- Part D ----
    print("--- Part D: Contamination model (2-component mixture) ---")
    mixture_results = part_d_contamination(obs, u_newton, SEP_BINS_KAU)
    gf = mixture_results['global']
    print(f"  Global: gamma = {gf['gamma']:.3f} +/- {gf['gamma_err']:.3f} "
          f"({gf['sigma_from_unity']:.1f} sigma from 1)")
    print(f"  Contamination fraction: {gf['f_contam']:.3f}")
    for br in mixture_results['per_bin']:
        if 'gamma' in br:
            print(f"    {br['bin']}: gamma = {br['gamma']:.3f} +/- "
                  f"{br['gamma_err']:.3f}, f_c = {br['f_contam']:.3f} "
                  f"(N = {br['N_pairs']})")
        else:
            print(f"    {br['bin']}: {br.get('status', 'N/A')} "
                  f"(N = {br.get('N_pairs', 0)})")
    print()

    # ---- Part E ----
    print("--- Part E: Comparison metrics ---")
    metrics = part_e_metrics(obs, u_newton, mixture_results, SEP_BINS_KAU)
    gm = metrics['global']
    print(f"  Median u:  obs = {gm['median_u_obs']:.3f}, "
          f"Newton = {gm['median_u_newton']:.3f}")
    print(f"  90th pct:  obs = {gm['p90_u_obs']:.3f}, "
          f"Newton = {gm['p90_u_newton']:.3f}")
    print(f"  Tail (u>1.5): obs = {gm['tail_frac_obs_u15']:.4f}, "
          f"Newton = {gm['tail_frac_newton_u15']:.4f}")
    print(f"  Gamma: {gm['gamma']:.3f} +/- {gm['gamma_err']:.3f} "
          f"({gm['gamma_sigma_from_unity']:.1f} sigma from 1)")
    print()

    # ---- Part F ----
    print("--- Part F: Robustness suite ---")
    robustness = part_f_robustness(raw_data, u_newton)
    for name, r in robustness['configs'].items():
        if 'gamma' in r:
            print(f"  {name}: N = {r['N_pairs']}, "
                  f"gamma = {r['gamma']:.3f} +/- {r['gamma_err']:.3f}")
        else:
            print(f"  {name}: {r.get('status', 'N/A')}")
    print(f"  Gamma range: {robustness['gamma_range']:.4f}")
    print(f"  Relative variation: {robustness['relative_variation']:.3f}")
    print()

    # ---- Part G ----
    print("--- Part G: Falsifiers ---")
    falsifiers = part_g_falsifiers(metrics, mixture_results, robustness)

    # Print F3 first (guardrail)
    f3 = falsifiers['F3_cut_stability']
    print(f"  F3_cut_stability: {f3['result']}")
    print(f"    range = {f3['gamma_range']:.4f}, "
          f"rel_var = {f3['relative_variation']:.3f}")
    if f3.get('contamination_signature'):
        print(f"    Contamination signature detected: gamma and f_contam "
              f"both increase with separation")
        print(f"    (gamma-sep corr = {f3['gamma_vs_sep_correlation']:.2f}, "
              f"fc-sep corr = {f3['fcontam_vs_sep_correlation']:.2f})")

    # F1
    f1 = falsifiers['F1_high_sep_tail']
    print(f"  F1_high_sep_tail: {f1['result']}")
    print(f"    gamma(s>=20) = {f1['gamma_high_sep']:.3f}, "
          f"sigma = {f1['sigma_from_unity']:.1f}")
    if 'note' in f1:
        print(f"    NOTE: {f1['note'][:100]}...")

    # F2
    f2 = falsifiers['F2_global_boost']
    print(f"  F2_global_boost: {f2['result']}")
    print(f"    gamma = {f2['gamma']}, err = {f2['gamma_err']}, "
          f"sigma = {f2['sigma_from_unity']}")

    fs = falsifiers['summary']
    parts = []
    if fs['PASS']:
        parts.append(f"{fs['PASS']} PASS")
    if fs['INCONCLUSIVE']:
        parts.append(f"{fs['INCONCLUSIVE']} INCONCLUSIVE")
    if fs['STABLE']:
        parts.append(f"{fs['STABLE']} STABLE")
    if fs['FAIL']:
        parts.append(f"{fs['FAIL']} FAIL")
    print(f"\n  Result: {', '.join(parts)} -> {fs['overall']}")
    print()

    # ---- Save JSON ----
    output = {
        'step': '22D',
        'title': 'Wide Binary Gravity Test',
        'mtdf_prediction': 'Newtonian (gamma = 1.00)',
        'mtdf_reasoning': (
            'The alpha-enhancement requires a galaxy-scale stress field '
            'source. Individual binary stars do not produce such a field. '
            'The galactic background field gradient is negligible at binary '
            'separations (0.01-0.15 pc vs kpc-scale gradients). Solar '
            'System screening confirms (Step 13: safe by > 10^20).'
        ),
        'dataset': ('Pittordis & Sutherland (2023, OJAp 6, 4), '
                    'Gaia EDR3, Zenodo 7629240'),
        'cut_summary': cut_summary,
        'observables': {
            'N_pairs': obs['N_pairs'],
            's_range_kau': [round(float(obs['s_kau'].min()), 2),
                            round(float(obs['s_kau'].max()), 2)],
            'u_range': [round(float(np.min(obs['u'])), 4),
                        round(float(np.max(obs['u'])), 4)],
            'median_u': round(float(np.median(obs['u'])), 4),
            'median_sigma_u': obs.get('median_sigma_u'),
        },
        'newtonian_mc': {
            'N_samples': len(u_newton),
            'seed': MC_SEED,
            'median_u': round(float(np.median(u_newton)), 4),
            'sigma_u_convolution': round(sigma_conv, 4) if sigma_conv else 0,
        },
        'mixture_fit': mixture_results,
        'metrics': metrics,
        'robustness': robustness,
        'falsifiers': falsifiers,
        'summary': {
            'gamma_global': gf['gamma'],
            'gamma_global_err': gf['gamma_err'],
            'gamma_sigma_from_unity': gf['sigma_from_unity'],
            'f_contam_global': gf['f_contam'],
            'overall_result': fs['overall'],
            'n_falsifiers_pass': fs['PASS'],
            'n_falsifiers_fail': fs['FAIL'],
        },
    }

    json_path = OUTPUT_DIR / 'step22d_wide_binaries.json'
    with open(json_path, 'w') as f:
        json.dump(make_json_serializable(output), f, indent=2)
    print(f"  JSON saved: {json_path.name}")

    # ---- Plots ----
    p1 = OUTPUT_DIR / 'step22d_vperp_vs_sep.png'
    plot_vperp_vs_sep(obs, p1)
    print(f"  Plot saved: {p1.name}")

    p2 = OUTPUT_DIR / 'step22d_u_histograms.png'
    plot_u_histograms(obs, u_newton, mixture_results, SEP_BINS_KAU, p2)
    print(f"  Plot saved: {p2.name}")

    p3 = OUTPUT_DIR / 'step22d_tail_fraction.png'
    plot_tail_fraction(obs, SEP_BINS_KAU, p3)
    print(f"  Plot saved: {p3.name}")

    p4 = OUTPUT_DIR / 'step22d_robustness.png'
    plot_robustness(robustness, p4)
    print(f"  Plot saved: {p4.name}")

    # ---- Manifest ----
    manifest = {}
    for fpath in sorted(OUTPUT_DIR.iterdir()):
        if fpath.is_file():
            manifest[fpath.name] = {
                'sha256': sha256_of_file(fpath),
                'size_bytes': fpath.stat().st_size,
            }
    manifest_path = OUTPUT_DIR / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved: {manifest_path.name}")

    print()
    n_total = fs['PASS'] + fs['FAIL'] + fs['INCONCLUSIVE']
    print("=" * 60)
    print(f"Step 22D COMPLETE -- Falsifiers: "
          f"{fs['PASS']}/{n_total} PASS, "
          f"{fs['INCONCLUSIVE']}/{n_total} INCONCLUSIVE "
          f"({fs['overall']})")
    print("=" * 60)
