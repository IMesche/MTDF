# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Shared weak-lensing utilities for Phase 6 tests.

Contains: cosmology, spherical geometry, tangential shear decomposition,
stacking, profile computation, systematics gates, and I/O helpers.
"""

import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats as spstats
from scipy.integrate import quad

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "validation" / "data"

# ── KiDS-1000 ────────────────────────────────────────────────────
KIDS_CAT_PATH = DATA_DIR / "External" / "kids_1000" / \
    "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"

KIDS_TOMO_EDGES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
KIDS_M_BIAS = np.array([-0.00930364, -0.01121353, -0.01518512,
                         0.00243974, 0.00754766])

KIDS_RA_MIN, KIDS_RA_MAX = 120.0, 240.0
KIDS_DEC_MIN, KIDS_DEC_MAX = -5.0, 5.0
KIDS_EDGE_BUFFER = 1.0  # degrees

# ── Physics ──────────────────────────────────────────────────────
OMEGA_M = 0.3
C_LIGHT = 299792.458       # km/s
DH_H = C_LIGHT / 100.0    # 2997.92 Mpc/h
Z_BUFFER = 0.1             # source-lens redshift gap

# ── Analysis ─────────────────────────────────────────────────────
R_BIN_EDGES = np.array([2.0, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0])
R_DELTA_MIN, R_DELTA_MAX = 5.0, 20.0
P_THRESHOLD = 0.05


# ── Cosmology ────────────────────────────────────────────────────

def comoving_distance(z):
    """Comoving distance in Mpc/h for flat LCDM (Omega_m=0.3)."""
    if np.isscalar(z) or isinstance(z, (float, np.floating)):
        integrand = lambda zp: 1.0 / np.sqrt(OMEGA_M * (1 + zp)**3 + (1 - OMEGA_M))
        val, _ = quad(integrand, 0, float(z))
        return DH_H * val
    return np.array([comoving_distance(float(zi)) for zi in z])


# ── Spherical geometry ───────────────────────────────────────────

def angular_separation(ra0, dec0, ra_s, dec_s):
    """Haversine separation in radians. All inputs in radians."""
    dra = ra_s - ra0
    ddec = dec_s - dec0
    a = np.sin(ddec * 0.5)**2 + np.cos(dec0) * np.cos(dec_s) * np.sin(dra * 0.5)**2
    return 2.0 * np.arcsin(np.minimum(np.sqrt(np.abs(a)), 1.0))


def celestial_pa(ra0, dec0, ra_s, dec_s):
    """Position angle North through East, in radians. All inputs radians."""
    dra = ra_s - ra0
    return np.arctan2(
        np.sin(dra) * np.cos(dec_s),
        np.cos(dec0) * np.sin(dec_s) - np.sin(dec0) * np.cos(dec_s) * np.cos(dra))


def tangential_shear(e1, e2, pa):
    """Decompose into gamma_t, gamma_x using celestial PA.

    Standard WL convention via phi_WL = pi/2 - PA:
      gamma_t = -(e1 cos 2phi + e2 sin 2phi) =  e1 cos 2PA - e2 sin 2PA
      gamma_x =  (e1 sin 2phi - e2 cos 2phi) =  e1 sin 2PA + e2 cos 2PA

    gamma_t > 0 for tangential alignment (overdensity).
    Voids (underdensity) produce gamma_t < 0 in LCDM.
    """
    c2 = np.cos(2.0 * pa)
    s2 = np.sin(2.0 * pa)
    return e1 * c2 - e2 * s2, e1 * s2 + e2 * c2


# ── Core stacking ────────────────────────────────────────────────

def stack_shear(cat, centres, r_edges, label="centres"):
    """Stack tangential/cross shear around centres in radial bins.

    Returns per-centre accumulators (n_centres x n_bins):
      sum_wet: Sum(w * e_t),  sum_wex: Sum(w * e_x),
      sum_w1m: Sum(w * (1+m)),  n_pairs: count
    """
    n_c = len(centres['ra'])
    n_bins = len(r_edges) - 1
    R_max = float(r_edges[-1])
    R_min = float(r_edges[0])

    sum_wet = np.zeros((n_c, n_bins))
    sum_wex = np.zeros((n_c, n_bins))
    sum_w1m = np.zeros((n_c, n_bins))
    n_pairs = np.zeros((n_c, n_bins), dtype=np.int64)

    # Pre-convert sources to radians
    s_ra = np.deg2rad(cat['ra'])
    s_dec = np.deg2rad(cat['dec'])
    s_e1, s_e2 = cat['e1'], cat['e2']
    s_w, s_m, s_z = cat['weight'], cat['m_bias'], cat['z_b']

    # Pre-compute comoving distance per centre
    chi_c = comoving_distance(centres['z'])
    c_ra = np.deg2rad(centres['ra'])
    c_dec = np.deg2rad(centres['dec'])

    t0 = datetime.now()
    for j in range(n_c):
        # Source-behind-lens cut
        behind = s_z > centres['z'][j] + Z_BUFFER
        if not np.any(behind):
            continue

        # Box pre-cut (fast)
        theta_max = R_max / chi_c[j]
        ra_box = theta_max / max(np.cos(c_dec[j]), 0.01)
        in_box = (behind &
                  (np.abs(s_ra - c_ra[j]) < ra_box) &
                  (np.abs(s_dec - c_dec[j]) < theta_max))
        if not np.any(in_box):
            continue

        idx = np.where(in_box)[0]

        # Exact angular separation
        sep = angular_separation(c_ra[j], c_dec[j], s_ra[idx], s_dec[idx])
        R = chi_c[j] * sep

        in_r = (R >= R_min) & (R < R_max)
        if not np.any(in_r):
            continue

        idx2 = idx[in_r]
        R_cut = R[in_r]

        # Position angle + tangential decomposition
        pa = celestial_pa(c_ra[j], c_dec[j], s_ra[idx2], s_dec[idx2])
        et, ex = tangential_shear(s_e1[idx2], s_e2[idx2], pa)

        # Bin accumulation
        bi = np.digitize(R_cut, r_edges) - 1
        valid = (bi >= 0) & (bi < n_bins)
        if not np.any(valid):
            continue

        bv = bi[valid]
        wv = s_w[idx2[valid]]
        mv = s_m[idx2[valid]]
        sum_wet[j] = np.bincount(bv, weights=wv * et[valid], minlength=n_bins)[:n_bins]
        sum_wex[j] = np.bincount(bv, weights=wv * ex[valid], minlength=n_bins)[:n_bins]
        sum_w1m[j] = np.bincount(bv, weights=wv * (1.0 + mv), minlength=n_bins)[:n_bins]
        n_pairs[j] = np.bincount(bv, minlength=n_bins)[:n_bins]

        if (j + 1) % 500 == 0:
            dt = (datetime.now() - t0).total_seconds()
            eta = dt / (j + 1) * (n_c - j - 1)
            print(f"    {j+1}/{n_c} {label} ({dt:.0f}s, ~{eta:.0f}s left)")

    dt = (datetime.now() - t0).total_seconds()
    print(f"    {n_c} {label}: {int(n_pairs.sum()):,} pairs in {dt:.1f}s")

    return dict(sum_wet=sum_wet, sum_wex=sum_wex,
                sum_w1m=sum_w1m, n_pairs=n_pairs)


# ── Profile from accumulators ────────────────────────────────────

def compute_profile(accum, r_edges):
    """Compute gamma_t, gamma_x with jackknife errors over centres."""
    n_c = accum['sum_wet'].shape[0]
    n_bins = accum['sum_wet'].shape[1]

    tot_wet = accum['sum_wet'].sum(0)
    tot_wex = accum['sum_wex'].sum(0)
    tot_w1m = accum['sum_w1m'].sum(0)
    tot_np = accum['n_pairs'].sum(0)

    gt = np.where(tot_w1m > 0, tot_wet / tot_w1m, np.nan)
    gx = np.where(tot_w1m > 0, tot_wex / tot_w1m, np.nan)

    # Jackknife over centres
    jk_gt = np.full((n_c, n_bins), np.nan)
    jk_gx = np.full((n_c, n_bins), np.nan)
    for k in range(n_c):
        w1m_k = tot_w1m - accum['sum_w1m'][k]
        ok = w1m_k > 0
        jk_gt[k, ok] = (tot_wet[ok] - accum['sum_wet'][k, ok]) / w1m_k[ok]
        jk_gx[k, ok] = (tot_wex[ok] - accum['sum_wex'][k, ok]) / w1m_k[ok]

    fac = (n_c - 1.0) / n_c
    gt_err = np.sqrt(fac * np.nansum((jk_gt - gt[None, :])**2, axis=0))
    gx_err = np.sqrt(fac * np.nansum((jk_gx - gx[None, :])**2, axis=0))

    r_centres = np.sqrt(r_edges[:-1] * r_edges[1:])

    return dict(
        R=r_centres.tolist(),
        gamma_t=[float(x) for x in gt],
        gamma_x=[float(x) for x in gx],
        gamma_t_err=[float(x) for x in gt_err],
        gamma_x_err=[float(x) for x in gx_err],
        n_pairs=[int(x) for x in tot_np],
        n_centres=n_c,
    )


# ── Systematics gate ─────────────────────────────────────────────

def gamma_x_test(profile):
    """Chi-squared: gamma_x vs zero. Fails if p < 0.05."""
    gx = np.array(profile['gamma_x'])
    err = np.array(profile['gamma_x_err'])
    ok = np.isfinite(gx) & np.isfinite(err) & (err > 0)
    chi2 = float(np.sum((gx[ok] / err[ok])**2))
    dof = int(ok.sum())
    chi2_dof = chi2 / dof if dof > 0 else np.inf
    p = float(1.0 - spstats.chi2.cdf(chi2, dof)) if dof > 0 else 0.0
    passed = p > P_THRESHOLD
    return dict(chi2=chi2, dof=dof, chi2_dof=chi2_dof,
                p_value=p, threshold=P_THRESHOLD, passed=passed)


# ── I/O ──────────────────────────────────────────────────────────

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
