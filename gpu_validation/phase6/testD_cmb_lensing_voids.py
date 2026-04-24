#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test D v2: CMB Lensing × DESI Voids (compensated filter)
====================================================================
Stacks Planck PR4 CMB lensing convergence (kappa) around DESIVAST
void centres using a per-void compensated filter:
  comp_i = kappa_centre(R/Rv<0.5) - kappa_outer(4<R/Rv<5)
This removes the large-scale positive pedestal seen in v1.

v2 changes from v1:
  - No hard mask cut: all voids kept, per-pixel mask weighting
  - Compensated statistic (centre - outer per void) as primary
  - Low-l removal robustness (l<20, l<30)
  - Null tests report p-values on compensated statistic

Pipeline:
  1. Load Planck PR4 kappa map (MV, NSIDE=512, FWHM=0.5 deg)
  2. Load DESIVAST void centres (REVOLVER primary, + VIDE, VoidFinder)
  3. Keep all voids (no hard mask cut; masked pixels skipped in stacking)
  4. Smoke test: 200 random positions -> compensated stat consistent with zero
  5. Full stacking: radial profile + compensated statistic
  6. Bootstrap errors
  7. Null tests: RA-scramble + random positions (report comp p-values)
  8. Low-l robustness: re-stack with l<20 and l<30 removed
  9. Systematics: NGC vs SGC, redshift bins
  10. Catalogue comparison: REVOLVER vs VIDE vs VoidFinder
  11. Plots, JSON summary, manifest

Usage:
  python mtdf_validation/phase6/testD_cmb_lensing_voids.py [--seed 42]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.integrate import quad
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Paths ---------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "validation" / "data"
OUT_DIR = PROJECT_ROOT / "validation" / "output" / "phase6" / "testD_cmb_lensing"

PLANCK_DIR = DATA_DIR / "External" / "planck_lensing" / "PR4_variations"
VOID_DIR = DATA_DIR / "External" / "desivast_voids"

# -- Physics --------------------------------------------------------------
OMEGA_M = 0.3
C_LIGHT = 299792.458       # km/s
DH_H = C_LIGHT / 100.0     # Mpc/h

# -- Analysis parameters --------------------------------------------------
NSIDE = 512
LMAX = 3 * NSIDE - 1       # 1535
FWHM_DEG = 0.5
N_RADIAL_BINS = 25
R_MAX_RV = 5.0

# Centre and ring ranges for headline stats (in R/Rv)
R_CENTRE_MAX = 0.5
R_RING_LO, R_RING_HI = 1.0, 2.0

# Outer annulus for compensated filter (in R/Rv)
R_OUTER_LO, R_OUTER_HI = 4.0, 5.0

# Minimum unmasked pixels per void for compensated stat
MIN_PIX_CENTRE = 5
MIN_PIX_OUTER = 10

# Low-l cuts for robustness
LMIN_CUTS = [20, 30]

# Null tests
N_RANDOM_ITER = 100
N_BOOTSTRAP = 1000
N_RA_SCRAMBLE = 200

# Smoke test
N_SMOKE = 200

P_THRESHOLD = 0.05


# -- Cosmology -----------------------------------------------------------

def comoving_distance(z):
    """Comoving distance in Mpc/h for flat LCDM (Omega_m=0.3)."""
    integrand = lambda zp: 1.0 / np.sqrt(OMEGA_M * (1 + zp)**3 + (1 - OMEGA_M))
    if np.isscalar(z):
        val, _ = quad(integrand, 0, float(z))
        return DH_H * val
    return np.array([comoving_distance(float(zi)) for zi in z])


def redshift_from_comoving(d_mpc_h, z_lo=0.001, z_hi=1.0):
    """Invert comoving_distance to get z from distance in Mpc/h."""
    f = lambda z: comoving_distance(z) - d_mpc_h
    return brentq(f, z_lo, z_hi)


# -- I/O helpers ----------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def pr(msg, **kw):
    print(msg, flush=True, **kw)


# -- Planck kappa map -----------------------------------------------------

def load_planck_kappa(planck_dir, nside=NSIDE, lmax=LMAX, fwhm_deg=FWHM_DEG,
                      lmin_cut=0):
    """Load PR4 MV kappa map, subtract mean field, smooth, mask.

    Parameters
    ----------
    lmin_cut : int
        If > 0, zero out all alm with l < lmin_cut (robustness test).
    """
    planck_dir = Path(planck_dir)

    # Read kappa alm (data and mean field)
    dat_path = planck_dir / "PR42018like_klm_dat_MV.fits"
    mf_path = planck_dir / "PR42018like_klm_mf_MV.fits"
    mask_path = planck_dir / "mask.fits.gz"

    pr(f"  Loading Planck PR4 MV kappa alm: {dat_path.name}")
    dat_alm = hp.read_alm(str(dat_path))
    mf_alm = hp.read_alm(str(mf_path))
    klm = dat_alm - mf_alm

    lmax_in = hp.Alm.getlmax(len(klm))
    pr(f"  alm lmax: {lmax_in}, target lmax: {lmax}")

    # Low-l removal (robustness)
    if lmin_cut > 0:
        for l in range(lmin_cut):
            for m in range(l + 1):
                idx = hp.Alm.getidx(lmax_in, l, m)
                klm[idx] = 0.0
        pr(f"  Removed l < {lmin_cut} modes")

    # Apply Gaussian smoothing in harmonic space
    bl = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax_in)
    hp.almxfl(klm, bl, inplace=True)

    # Convert at full NSIDE=2048, then downgrade
    kmap_hr = hp.alm2map(klm, 2048, lmax=lmax_in)
    mask_hr = hp.read_map(str(mask_path))
    pr(f"  Mask NSIDE: {hp.get_nside(mask_hr)}")

    # Set masked pixels before downgrade
    kmap_hr[mask_hr < 0.5] = 0.0

    # Downgrade
    kmap = hp.ud_grade(kmap_hr, nside)
    mask = hp.ud_grade(mask_hr, nside)
    mask_bin = mask > 0.5

    # Subtract monopole from unmasked region
    kmap[mask_bin] -= kmap[mask_bin].mean()
    kmap[~mask_bin] = hp.UNSEEN

    fsky = mask_bin.sum() / len(mask_bin)
    pr(f"  Map NSIDE={nside}, FWHM={fwhm_deg} deg, f_sky={fsky:.3f}"
       + (f", lmin_cut={lmin_cut}" if lmin_cut > 0 else ""))
    pr(f"  RMS (unmasked): {kmap[mask_bin].std():.4e}")

    return kmap, mask_bin


# -- Void loaders ---------------------------------------------------------

def load_revolver_vide(void_dir, finder="revolver"):
    """Load REVOLVER or VIDE void centres from V2 format FITS."""
    void_dir = Path(void_dir)
    prefix = "DESIVAST_BGS_VOLLIM_V2_" + finder.upper()
    records = []
    for cap in ("NGC", "SGC"):
        fpath = void_dir / f"{prefix}_{cap}.fits"
        if not fpath.exists():
            pr(f"  WARNING: {fpath.name} not found, skipping")
            continue
        with fits.open(fpath) as hdul:
            d = hdul["VOIDS"].data
            for i in range(len(d)):
                records.append(dict(
                    ra=float(d["RA"][i]),
                    dec=float(d["DEC"][i]),
                    z=float(d["REDSHIFT"][i]),
                    R_mpc_h=float(d["RADIUS"][i]),
                    cap=cap,
                    finder=finder,
                ))
    return records


def load_voidfinder(void_dir):
    """Load VoidFinder void centres (EDGE=0 interior only)."""
    void_dir = Path(void_dir)
    records = []
    for cap in ("NGC", "SGC"):
        fpath = void_dir / f"DESIVAST_BGS_VOLLIM_VoidFinder_{cap}.fits"
        if not fpath.exists():
            pr(f"  WARNING: {fpath.name} not found, skipping")
            continue
        with fits.open(fpath) as hdul:
            d = hdul["MAXIMALS"].data
            interior = d["EDGE"] == 0
            d = d[interior]
            for i in range(len(d)):
                r_comoving = float(d["R"][i])  # Mpc/h
                z = redshift_from_comoving(r_comoving)
                records.append(dict(
                    ra=float(d["RA"][i]),
                    dec=float(d["DEC"][i]),
                    z=z,
                    R_mpc_h=float(d["R_EFF"][i]),
                    cap=cap,
                    finder="voidfinder",
                ))
    return records


def prepare_centres(records, kappa_map, mask_bin, nside=NSIDE):
    """Convert records to structured arrays, compute angular radii.
    No hard mask cut -- all voids are kept. Masked pixels are skipped
    during stacking, and voids with too few unmasked pixels are filtered
    out by the compensated statistic's MIN_PIX thresholds."""
    n = len(records)
    ra = np.array([r["ra"] for r in records])
    dec = np.array([r["dec"] for r in records])
    z = np.array([r["z"] for r in records])
    R = np.array([r["R_mpc_h"] for r in records])
    cap = np.array([r["cap"] for r in records])

    # Angular void radius: theta_v = R / d_c(z) [radians]
    d_c = comoving_distance(z)
    theta_v = R / d_c  # radians

    # Diagnostic: how many centres are in unmasked pixels
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    centre_unmasked = mask_bin[pix]

    pr(f"    Total: {n}, centre unmasked: {centre_unmasked.sum()}, "
       f"centre masked: {(~centre_unmasked).sum()}")

    return dict(
        ra=ra, dec=dec, z=z, R=R,
        theta_v=theta_v, cap=cap,
        centre_unmasked=centre_unmasked,
        n_total=n, n_good=n,
    )


# -- Stacking -------------------------------------------------------------

def stack_kappa_profile(kappa_map, mask_bin, ra, dec, theta_v,
                        nside=NSIDE, n_bins=N_RADIAL_BINS,
                        r_max=R_MAX_RV, label="voids"):
    """Stack kappa in radial R/Rv bins. Returns per-void accumulators."""
    n_voids = len(ra)
    dr = r_max / n_bins
    bin_edges = np.linspace(0, r_max, n_bins + 1)

    sum_kappa = np.zeros((n_voids, n_bins))
    sum_weight = np.zeros((n_voids, n_bins))

    t0 = time.time()
    for i in range(n_voids):
        vec = hp.ang2vec(ra[i], dec[i], lonlat=True)
        disc_pix = hp.query_disc(nside, vec, r_max * theta_v[i])
        if len(disc_pix) == 0:
            continue

        # Pixel angular distances
        pix_vecs = np.array(hp.pix2vec(nside, disc_pix)).T
        cos_sep = np.dot(pix_vecs, vec)
        cos_sep = np.clip(cos_sep, -1, 1)
        sep = np.arccos(cos_sep)

        # Convert to R/Rv
        r_rv = sep / theta_v[i]

        # Bin assignment
        bi = np.searchsorted(bin_edges, r_rv) - 1
        valid = (bi >= 0) & (bi < n_bins) & mask_bin[disc_pix]

        for j in range(n_bins):
            in_bin = valid & (bi == j)
            if in_bin.any():
                sum_kappa[i, j] = kappa_map[disc_pix[in_bin]].sum()
                sum_weight[i, j] = in_bin.sum()

        if (i + 1) % 500 == 0 or i + 1 == n_voids:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remain = (n_voids - i - 1) / rate
            pr(f"    {i+1}/{n_voids} {label} ({elapsed:.0f}s, ~{remain:.0f}s left)")

    total_kappa = sum_kappa.sum(axis=0)
    total_weight = sum_weight.sum(axis=0)
    profile = np.where(total_weight > 0, total_kappa / total_weight, np.nan)

    # Jackknife errors
    jk_profiles = np.full((n_voids, n_bins), np.nan)
    for k in range(n_voids):
        w_k = total_weight - sum_weight[k]
        ok = w_k > 0
        jk_profiles[k, ok] = (total_kappa[ok] - sum_kappa[k, ok]) / w_k[ok]

    fac = (n_voids - 1.0) / n_voids
    profile_err = np.sqrt(fac * np.nansum((jk_profiles - profile) ** 2, axis=0))

    r_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    elapsed = time.time() - t0
    pr(f"    Stacked {n_voids} {label} in {elapsed:.1f}s")

    return dict(
        r_rv=r_centres,
        kappa=profile,
        kappa_err=profile_err,
        n_pixels=total_weight.astype(int),
        n_voids=n_voids,
        sum_kappa=sum_kappa,
        sum_weight=sum_weight,
    )


def compute_headline(profile):
    """Compute kappa_centre, kappa_ring, delta_kappa from a stacked profile."""
    r = profile["r_rv"]
    k = profile["kappa"]
    ke = profile["kappa_err"]
    npx = profile["n_pixels"]

    # Centre: R/Rv < R_CENTRE_MAX
    c_mask = r < R_CENTRE_MAX
    if c_mask.any() and npx[c_mask].sum() > 0:
        w_c = npx[c_mask].astype(float)
        kappa_centre = np.average(k[c_mask], weights=w_c)
        centre_err = np.sqrt(np.average(ke[c_mask]**2, weights=w_c))
    else:
        kappa_centre, centre_err = np.nan, np.nan

    # Ring: R_RING_LO < R/Rv < R_RING_HI
    r_mask = (r >= R_RING_LO) & (r < R_RING_HI)
    if r_mask.any() and npx[r_mask].sum() > 0:
        w_r = npx[r_mask].astype(float)
        kappa_ring = np.average(k[r_mask], weights=w_r)
        ring_err = np.sqrt(np.average(ke[r_mask]**2, weights=w_r))
    else:
        kappa_ring, ring_err = np.nan, np.nan

    delta_kappa = kappa_ring - kappa_centre
    delta_err = np.sqrt(centre_err**2 + ring_err**2)
    snr_centre = abs(kappa_centre) / centre_err if centre_err > 0 else 0.0
    snr_delta = abs(delta_kappa) / delta_err if delta_err > 0 else 0.0

    return dict(
        kappa_centre=float(kappa_centre),
        kappa_centre_err=float(centre_err),
        kappa_ring=float(kappa_ring),
        kappa_ring_err=float(ring_err),
        delta_kappa=float(delta_kappa),
        delta_kappa_err=float(delta_err),
        snr_centre=float(snr_centre),
        snr_delta=float(snr_delta),
    )


# -- Compensated statistic ------------------------------------------------

def compute_compensated(sum_kappa, sum_weight, r_rv):
    """Per-void compensated filter: kappa_centre - kappa_outer per void.

    For each void, computes mean kappa in centre (R/Rv < 0.5) minus mean
    kappa in outer annulus (4 < R/Rv < 5). Voids without enough unmasked
    pixels in either range are dropped. Returns stacked mean + jackknife error.
    """
    n_voids = sum_kappa.shape[0]

    c_bins = r_rv < R_CENTRE_MAX
    o_bins = (r_rv >= R_OUTER_LO) & (r_rv < R_OUTER_HI)

    comp_values = np.full(n_voids, np.nan)
    good_mask = np.zeros(n_voids, dtype=bool)

    for i in range(n_voids):
        wc = sum_weight[i, c_bins].sum()
        wo = sum_weight[i, o_bins].sum()
        if wc < MIN_PIX_CENTRE or wo < MIN_PIX_OUTER:
            continue
        kc_i = sum_kappa[i, c_bins].sum() / wc
        ko_i = sum_kappa[i, o_bins].sum() / wo
        comp_values[i] = kc_i - ko_i
        good_mask[i] = True

    comp = comp_values[good_mask]
    n_used = len(comp)

    if n_used == 0:
        return dict(comp_mean=np.nan, comp_err=np.nan, snr=0.0, n_used=0,
                    comp_values=np.array([]), good_mask=good_mask)

    comp_mean = comp.mean()

    # Jackknife error
    total = comp.sum()
    jk = (total - comp) / (n_used - 1)
    comp_err = np.sqrt((n_used - 1.0) / n_used * np.sum((jk - comp_mean)**2))

    snr = abs(comp_mean) / comp_err if comp_err > 0 else 0.0

    return dict(
        comp_mean=float(comp_mean),
        comp_err=float(comp_err),
        snr=float(snr),
        n_used=int(n_used),
        comp_values=comp,
        good_mask=good_mask,
    )


# -- Bootstrap ------------------------------------------------------------

def bootstrap_headline(sum_kappa, sum_weight, r_rv, n_boot=N_BOOTSTRAP, rng=None):
    """Bootstrap over void centres for delta_kappa + compensated stat."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_voids = sum_kappa.shape[0]

    boot_centre = []
    boot_delta = []
    boot_comp = []

    c_mask = r_rv < R_CENTRE_MAX
    r_mask = (r_rv >= R_RING_LO) & (r_rv < R_RING_HI)
    o_mask = (r_rv >= R_OUTER_LO) & (r_rv < R_OUTER_HI)

    for _ in range(n_boot):
        idx = rng.integers(0, n_voids, size=n_voids)
        sk = sum_kappa[idx]
        sw = sum_weight[idx]

        # Profile-level stats
        sk_sum = sk.sum(axis=0)
        sw_sum = sw.sum(axis=0)
        profile = np.where(sw_sum > 0, sk_sum / sw_sum, np.nan)

        if c_mask.any():
            wc = sw_sum[c_mask]
            kc = np.average(profile[c_mask], weights=wc) if wc.sum() > 0 else np.nan
        else:
            kc = np.nan
        if r_mask.any():
            wr = sw_sum[r_mask]
            kr = np.average(profile[r_mask], weights=wr) if wr.sum() > 0 else np.nan
        else:
            kr = np.nan

        boot_centre.append(kc)
        boot_delta.append(kr - kc)

        # Per-void compensated
        comp = compute_compensated(sk, sw, r_rv)
        boot_comp.append(comp["comp_mean"])

    return dict(
        centre_mean=float(np.nanmean(boot_centre)),
        centre_std=float(np.nanstd(boot_centre)),
        delta_mean=float(np.nanmean(boot_delta)),
        delta_std=float(np.nanstd(boot_delta)),
        comp_mean=float(np.nanmean(boot_comp)),
        comp_std=float(np.nanstd(boot_comp)),
    )


# -- Null tests -----------------------------------------------------------

def null_ra_scramble(kappa_map, mask_bin, centres, n_iter=N_RA_SCRAMBLE,
                     nside=NSIDE, rng=None, label="RA-scramble"):
    """Footprint-respecting null: shuffle RA among void centres."""
    if rng is None:
        rng = np.random.default_rng(99)

    ra = centres["ra"].copy()
    dec = centres["dec"]
    theta_v = centres["theta_v"]

    null_deltas = []
    null_centres_val = []
    null_comp = []

    pr(f"\n  Null test ({label}, {n_iter} iterations)...")
    t0 = time.time()
    for it in range(n_iter):
        ra_shuffled = rng.permutation(ra)
        prof = stack_kappa_profile(
            kappa_map, mask_bin, ra_shuffled, dec, theta_v,
            nside=nside, label=f"null-{it+1}", r_max=R_MAX_RV,
        )
        hl = compute_headline(prof)
        null_deltas.append(hl["delta_kappa"])
        null_centres_val.append(hl["kappa_centre"])

        comp = compute_compensated(prof["sum_kappa"], prof["sum_weight"], prof["r_rv"])
        null_comp.append(comp["comp_mean"])

        if (it + 1) % 20 == 0:
            elapsed = time.time() - t0
            pr(f"    {it+1}/{n_iter} ({elapsed:.0f}s)")

    null_deltas = np.array(null_deltas)
    null_centres_val = np.array(null_centres_val)
    null_comp = np.array(null_comp)
    elapsed = time.time() - t0
    pr(f"    {label} complete in {elapsed:.0f}s")

    return dict(
        delta_kappa_null=null_deltas,
        kappa_centre_null=null_centres_val,
        comp_null=null_comp,
        delta_mean=float(np.mean(null_deltas)),
        delta_std=float(np.std(null_deltas)),
        centre_mean=float(np.mean(null_centres_val)),
        centre_std=float(np.std(null_centres_val)),
        comp_mean=float(np.nanmean(null_comp)),
        comp_std=float(np.nanstd(null_comp)),
    )


def null_random_positions(kappa_map, mask_bin, centres, n_iter=N_RANDOM_ITER,
                          nside=NSIDE, rng=None, label="random-pos"):
    """Random sky position null: draw positions from unmasked Planck pixels."""
    if rng is None:
        rng = np.random.default_rng(77)

    n_voids = len(centres["ra"])
    theta_v = centres["theta_v"]

    unmasked_pix = np.where(mask_bin)[0]

    null_deltas = []
    null_centres_val = []
    null_comp = []

    pr(f"\n  Null test ({label}, {n_iter} iterations)...")
    t0 = time.time()
    for it in range(n_iter):
        pix_choice = rng.choice(unmasked_pix, size=n_voids, replace=True)
        theta_hp, phi_hp = hp.pix2ang(nside, pix_choice)
        ra_rand = np.degrees(phi_hp)
        dec_rand = 90.0 - np.degrees(theta_hp)

        tv_rand = rng.permutation(theta_v)

        prof = stack_kappa_profile(
            kappa_map, mask_bin, ra_rand, dec_rand, tv_rand,
            nside=nside, label=f"rand-{it+1}", r_max=R_MAX_RV,
        )
        hl = compute_headline(prof)
        null_deltas.append(hl["delta_kappa"])
        null_centres_val.append(hl["kappa_centre"])

        comp = compute_compensated(prof["sum_kappa"], prof["sum_weight"], prof["r_rv"])
        null_comp.append(comp["comp_mean"])

        if (it + 1) % 20 == 0:
            elapsed = time.time() - t0
            pr(f"    {it+1}/{n_iter} ({elapsed:.0f}s)")

    null_deltas = np.array(null_deltas)
    null_centres_val = np.array(null_centres_val)
    null_comp = np.array(null_comp)
    elapsed = time.time() - t0
    pr(f"    {label} complete in {elapsed:.0f}s")

    return dict(
        delta_kappa_null=null_deltas,
        kappa_centre_null=null_centres_val,
        comp_null=null_comp,
        delta_mean=float(np.mean(null_deltas)),
        delta_std=float(np.std(null_deltas)),
        centre_mean=float(np.mean(null_centres_val)),
        centre_std=float(np.std(null_centres_val)),
        comp_mean=float(np.nanmean(null_comp)),
        comp_std=float(np.nanstd(null_comp)),
    )


# -- Plots ----------------------------------------------------------------

def plot_profile(profile, headline, comp_result, out_path, title=""):
    """Radial kappa profile with error bars, headline + compensated annotation."""
    fig, ax = plt.subplots(figsize=(9, 6))
    r = profile["r_rv"]
    k = profile["kappa"]
    ke = profile["kappa_err"]

    ax.errorbar(r, k, yerr=ke, fmt="o-", color="steelblue", capsize=3, ms=5,
                label=f"Data ({profile['n_voids']} voids)")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvspan(0, R_CENTRE_MAX, alpha=0.08, color="blue",
               label=f"Centre (R/Rv<{R_CENTRE_MAX})")
    ax.axvspan(R_OUTER_LO, R_OUTER_HI, alpha=0.08, color="green",
               label=f"Outer ({R_OUTER_LO}<R/Rv<{R_OUTER_HI})")

    # Annotate headline + compensated
    txt = (f"$\\kappa_{{centre}}$ = {headline['kappa_centre']:.4e} "
           f"$\\pm$ {headline['kappa_centre_err']:.4e}  ({headline['snr_centre']:.1f}$\\sigma$)\n"
           f"$\\Delta\\kappa$ (ring$-$centre) = {headline['delta_kappa']:.4e} "
           f"$\\pm$ {headline['delta_kappa_err']:.4e}  ({headline['snr_delta']:.1f}$\\sigma$)\n"
           f"Compensated = {comp_result['comp_mean']:.4e} "
           f"$\\pm$ {comp_result['comp_err']:.4e}  ({comp_result['snr']:.1f}$\\sigma$)"
           f"  [{comp_result['n_used']} voids]")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))

    ax.set_xlabel("$R / R_v$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title(title or "CMB lensing convergence around voids")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    pr(f"  Plot: {Path(out_path).name}")


def plot_null(data_value, null_result, null_key, out_path, title="",
              label="RA-scramble", stat_label="compensated"):
    """Null test distribution with data value marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = null_result[null_key]
    vals_clean = vals[np.isfinite(vals)]
    ax.hist(vals_clean, bins=25, color="lightgray", edgecolor="gray",
            label=f"{label} null ({len(vals_clean)} iter)")
    ax.axvline(data_value, color="red", lw=2, label=f"Data: {data_value:.4e}")
    ax.axvline(0, color="gray", ls="--", lw=0.8)

    # p-value: fraction of nulls more extreme than data
    if data_value < 0:
        n_more_extreme = np.sum(vals_clean <= data_value)
    else:
        n_more_extreme = np.sum(vals_clean >= data_value)
    p_val = n_more_extreme / len(vals_clean) if len(vals_clean) > 0 else np.nan
    ax.text(0.98, 0.97, f"p = {p_val:.3f}", transform=ax.transAxes,
            fontsize=12, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    ax.set_xlabel(stat_label)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Null test: {label}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    pr(f"  Plot: {Path(out_path).name}")


def plot_comparison(profiles, out_path, title=""):
    """Overlay profiles from multiple catalogues."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"revolver": "steelblue", "vide": "darkorange", "voidfinder": "forestgreen"}
    for name, prof in profiles.items():
        r = prof["r_rv"]
        k = prof["kappa"]
        ke = prof["kappa_err"]
        c = colors.get(name, "gray")
        ax.errorbar(r + 0.03 * list(profiles.keys()).index(name), k, yerr=ke,
                    fmt="o-", color=c, capsize=3, ms=4, label=f"{name} ({prof['n_voids']})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("$R / R_v$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title(title or "Catalogue comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    pr(f"  Plot: {Path(out_path).name}")


def plot_ngc_sgc(prof_ngc, prof_sgc, out_path, title=""):
    """NGC vs SGC consistency."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(prof_ngc["r_rv"] - 0.03, prof_ngc["kappa"], yerr=prof_ngc["kappa_err"],
                fmt="s-", color="steelblue", capsize=3, ms=4, label=f"NGC ({prof_ngc['n_voids']})")
    ax.errorbar(prof_sgc["r_rv"] + 0.03, prof_sgc["kappa"], yerr=prof_sgc["kappa_err"],
                fmt="^-", color="tomato", capsize=3, ms=4, label=f"SGC ({prof_sgc['n_voids']})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("$R / R_v$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title(title or "NGC vs SGC consistency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    pr(f"  Plot: {Path(out_path).name}")


def plot_zbins(zbin_profiles, out_path, title=""):
    """Redshift-binned profiles."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["steelblue", "darkorange", "forestgreen"]
    for i, (label, prof) in enumerate(zbin_profiles.items()):
        ax.errorbar(prof["r_rv"] + 0.03 * i, prof["kappa"], yerr=prof["kappa_err"],
                    fmt="o-", color=colors[i % 3], capsize=3, ms=4,
                    label=f"{label} ({prof['n_voids']})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("$R / R_v$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title(title or "Redshift-binned profiles")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    pr(f"  Plot: {Path(out_path).name}")


def plot_lowl_robustness(profiles_dict, comp_dict, out_path):
    """Overlay baseline + low-l cut profiles with compensated stats."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"baseline": "steelblue", "l>=20": "darkorange", "l>=30": "forestgreen"}

    for name, prof in profiles_dict.items():
        r = prof["r_rv"]
        k = prof["kappa"]
        ke = prof["kappa_err"]
        c = colors.get(name, "gray")
        ax1.errorbar(r + 0.02 * list(profiles_dict.keys()).index(name), k, yerr=ke,
                     fmt="o-", color=c, capsize=2, ms=3, label=name)
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel("$R / R_v$")
    ax1.set_ylabel("$\\kappa$")
    ax1.set_title("Radial profiles: low-l robustness")
    ax1.legend(fontsize=9)

    # Compensated stats as bar chart
    names = list(comp_dict.keys())
    means = [comp_dict[n]["comp_mean"] for n in names]
    errs = [comp_dict[n]["comp_err"] for n in names]
    cols = [colors.get(n, "gray") for n in names]
    x = np.arange(len(names))
    ax2.bar(x, means, yerr=errs, color=cols, alpha=0.7, capsize=5, edgecolor="black")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("Compensated $\\kappa_{centre} - \\kappa_{outer}$")
    ax2.set_title("Compensated statistic")

    for i, n in enumerate(names):
        snr = comp_dict[n]["snr"]
        nv = comp_dict[n]["n_used"]
        ax2.text(i, means[i] + errs[i] + 0.0002, f"{snr:.1f}$\\sigma$\n({nv}v)",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    pr(f"  Plot: {Path(out_path).name}")


# -- Main pipeline --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6D v2: CMB Lensing x DESI Voids")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pr("=" * 60)
    pr("Phase 6 Test D v2: CMB Lensing x DESI Voids")
    pr("  Planck PR4 MV convergence x DESIVAST void centres")
    pr("  Compensated filter: centre - outer annulus per void")
    pr("=" * 60)

    # -- Step 1: Load Planck kappa (baseline) -------------------------
    kappa_map, mask_bin = load_planck_kappa(PLANCK_DIR)

    # -- Step 2: Load void catalogues (no hard mask cut) --------------
    pr("\n  Loading void catalogues (no hard mask cut):")

    pr("  REVOLVER:")
    rev_records = load_revolver_vide(VOID_DIR, "revolver")
    pr(f"    Raw: {len(rev_records)} (NGC+SGC)")
    rev = prepare_centres(rev_records, kappa_map, mask_bin)

    pr("  VIDE:")
    vide_records = load_revolver_vide(VOID_DIR, "vide")
    pr(f"    Raw: {len(vide_records)} (NGC+SGC)")
    vide = prepare_centres(vide_records, kappa_map, mask_bin)

    pr("  VoidFinder:")
    vf_records = load_voidfinder(VOID_DIR)
    pr(f"    Raw: {len(vf_records)} (NGC+SGC, interior only)")
    vf = prepare_centres(vf_records, kappa_map, mask_bin)

    # Summary
    pr(f"\n  Void counts (all kept):")
    pr(f"    REVOLVER: {rev['n_good']} (centre unmasked: {rev['centre_unmasked'].sum()})")
    pr(f"    VIDE:     {vide['n_good']} (centre unmasked: {vide['centre_unmasked'].sum()})")
    pr(f"    VoidFinder: {vf['n_good']} (centre unmasked: {vf['centre_unmasked'].sum()})")
    pr(f"    REVOLVER z: [{rev['z'].min():.3f}, {rev['z'].max():.3f}], "
       f"median: {np.median(rev['z']):.3f}")
    pr(f"    REVOLVER R: [{rev['R'].min():.1f}, {rev['R'].max():.1f}] Mpc/h, "
       f"median: {np.median(rev['R']):.1f}")
    pr(f"    REVOLVER theta_v: [{np.degrees(rev['theta_v'].min()):.2f}, "
       f"{np.degrees(rev['theta_v'].max()):.2f}] deg, "
       f"median: {np.degrees(np.median(rev['theta_v'])):.2f} deg")

    # -- Step 3: Smoke test -------------------------------------------
    pr("\n" + "=" * 60)
    pr("SMOKE TEST (200 random positions)")
    pr("=" * 60)

    unmasked_pix = np.where(mask_bin)[0]
    smoke_pix = rng.choice(unmasked_pix, size=N_SMOKE, replace=True)
    theta_hp, phi_hp = hp.pix2ang(NSIDE, smoke_pix)
    smoke_ra = np.degrees(phi_hp)
    smoke_dec = 90.0 - np.degrees(theta_hp)
    smoke_tv = rng.permutation(rev["theta_v"][:N_SMOKE]) if rev["n_good"] >= N_SMOKE \
        else np.full(N_SMOKE, np.median(rev["theta_v"]))

    smoke_prof = stack_kappa_profile(kappa_map, mask_bin, smoke_ra, smoke_dec,
                                     smoke_tv, label="smoke")
    smoke_comp = compute_compensated(smoke_prof["sum_kappa"], smoke_prof["sum_weight"],
                                     smoke_prof["r_rv"])
    pr(f"  Smoke compensated = {smoke_comp['comp_mean']:.4e} +/- {smoke_comp['comp_err']:.4e}"
       f" ({smoke_comp['snr']:.1f}sigma, {smoke_comp['n_used']} voids)")

    if smoke_comp["snr"] > 3.0:
        pr(f"  WARNING: smoke test compensated S/N = {smoke_comp['snr']:.1f} > 3")
    else:
        pr("  SMOKE PASSED: random positions consistent with zero.")

    # -- Step 4: Full REVOLVER stack ----------------------------------
    pr("\n" + "=" * 60)
    pr("FULL STACK: REVOLVER (primary)")
    pr("=" * 60)

    rev_prof = stack_kappa_profile(kappa_map, mask_bin, rev["ra"], rev["dec"],
                                   rev["theta_v"], label="REVOLVER")
    rev_hl = compute_headline(rev_prof)
    rev_comp = compute_compensated(rev_prof["sum_kappa"], rev_prof["sum_weight"],
                                   rev_prof["r_rv"])

    pr(f"\n  kappa_centre (R/Rv<{R_CENTRE_MAX}) = "
       f"{rev_hl['kappa_centre']:.4e} +/- {rev_hl['kappa_centre_err']:.4e}"
       f"  ({rev_hl['snr_centre']:.1f}sigma)")
    pr(f"  delta_kappa (ring - centre) = "
       f"{rev_hl['delta_kappa']:.4e} +/- {rev_hl['delta_kappa_err']:.4e}"
       f"  ({rev_hl['snr_delta']:.1f}sigma)")
    pr(f"  COMPENSATED (centre - outer) = "
       f"{rev_comp['comp_mean']:.4e} +/- {rev_comp['comp_err']:.4e}"
       f"  ({rev_comp['snr']:.1f}sigma, {rev_comp['n_used']} voids)")

    plot_profile(rev_prof, rev_hl, rev_comp,
                 OUT_DIR / "testD_kappa_profile.png",
                 f"CMB lensing x REVOLVER voids [Planck PR4, v2]")

    # -- Step 5: Bootstrap -------------------------------------------
    pr(f"\n  Bootstrap ({N_BOOTSTRAP} resamples)...")
    boot = bootstrap_headline(rev_prof["sum_kappa"], rev_prof["sum_weight"],
                              rev_prof["r_rv"], n_boot=N_BOOTSTRAP, rng=rng)
    pr(f"    delta_kappa:  {boot['delta_mean']:.4e} +/- {boot['delta_std']:.4e}")
    pr(f"    compensated:  {boot['comp_mean']:.4e} +/- {boot['comp_std']:.4e}")
    boot_snr = abs(boot["comp_mean"]) / boot["comp_std"] if boot["comp_std"] > 0 else 0
    pr(f"    compensated S/N (bootstrap): {boot_snr:.1f}")

    # -- Step 6: Null tests ------------------------------------------
    pr("\n" + "=" * 60)
    pr("NULL TESTS")
    pr("=" * 60)

    # 6a: RA scramble (footprint-respecting)
    null_ra = null_ra_scramble(kappa_map, mask_bin, rev,
                               n_iter=N_RA_SCRAMBLE, rng=rng)
    # p-values on compensated statistic
    ra_comp_vals = null_ra["comp_null"]
    ra_comp_clean = ra_comp_vals[np.isfinite(ra_comp_vals)]
    if rev_comp["comp_mean"] < 0:
        p_ra_comp = np.mean(ra_comp_clean <= rev_comp["comp_mean"]) if len(ra_comp_clean) > 0 else np.nan
    else:
        p_ra_comp = np.mean(ra_comp_clean >= rev_comp["comp_mean"]) if len(ra_comp_clean) > 0 else np.nan
    # Also keep old delta p-values for reference
    p_ra_centre = np.mean(null_ra["kappa_centre_null"] <= rev_hl["kappa_centre"])
    p_ra_delta = np.mean(null_ra["delta_kappa_null"] >= rev_hl["delta_kappa"])

    pr(f"  RA-scramble: p(comp) = {p_ra_comp:.3f}, p(delta) = {p_ra_delta:.3f}, "
       f"p(centre) = {p_ra_centre:.3f}")

    plot_null(rev_comp["comp_mean"], null_ra, "comp_null",
             OUT_DIR / "testD_null_ra_scramble.png",
             "RA-scramble null test (compensated) [REVOLVER]",
             label="RA-scramble", stat_label="compensated $\\kappa_{centre}-\\kappa_{outer}$")

    # 6b: Random positions
    null_rand = null_random_positions(kappa_map, mask_bin, rev,
                                      n_iter=N_RANDOM_ITER, rng=rng)
    rand_comp_vals = null_rand["comp_null"]
    rand_comp_clean = rand_comp_vals[np.isfinite(rand_comp_vals)]
    if rev_comp["comp_mean"] < 0:
        p_rand_comp = np.mean(rand_comp_clean <= rev_comp["comp_mean"]) if len(rand_comp_clean) > 0 else np.nan
    else:
        p_rand_comp = np.mean(rand_comp_clean >= rev_comp["comp_mean"]) if len(rand_comp_clean) > 0 else np.nan
    p_rand_centre = np.mean(null_rand["kappa_centre_null"] <= rev_hl["kappa_centre"])
    p_rand_delta = np.mean(null_rand["delta_kappa_null"] >= rev_hl["delta_kappa"])

    pr(f"  Random: p(comp) = {p_rand_comp:.3f}, p(delta) = {p_rand_delta:.3f}, "
       f"p(centre) = {p_rand_centre:.3f}")

    plot_null(rev_comp["comp_mean"], null_rand, "comp_null",
             OUT_DIR / "testD_null_random.png",
             "Random positions null test (compensated)",
             label="random", stat_label="compensated $\\kappa_{centre}-\\kappa_{outer}$")

    # -- Step 7: Low-l robustness ------------------------------------
    pr("\n" + "=" * 60)
    pr("LOW-l ROBUSTNESS")
    pr("=" * 60)

    lowl_profiles = {"baseline": rev_prof}
    lowl_comp = {"baseline": rev_comp}

    for lmin in LMIN_CUTS:
        pr(f"\n  --- l >= {lmin} ---")
        kmap_ll, mask_ll = load_planck_kappa(PLANCK_DIR, lmin_cut=lmin)
        prof_ll = stack_kappa_profile(kmap_ll, mask_ll, rev["ra"], rev["dec"],
                                      rev["theta_v"], label=f"l>={lmin}")
        hl_ll = compute_headline(prof_ll)
        comp_ll = compute_compensated(prof_ll["sum_kappa"], prof_ll["sum_weight"],
                                      prof_ll["r_rv"])
        tag = f"l>={lmin}"
        lowl_profiles[tag] = prof_ll
        lowl_comp[tag] = comp_ll
        pr(f"  kappa_centre = {hl_ll['kappa_centre']:.4e}, "
           f"delta = {hl_ll['delta_kappa']:.4e}")
        pr(f"  compensated = {comp_ll['comp_mean']:.4e} +/- {comp_ll['comp_err']:.4e} "
           f"({comp_ll['snr']:.1f}sigma, {comp_ll['n_used']} voids)")

    plot_lowl_robustness(lowl_profiles, lowl_comp,
                         OUT_DIR / "testD_lowl_robustness.png")

    # -- Step 8: NGC vs SGC ------------------------------------------
    pr("\n" + "=" * 60)
    pr("SYSTEMATICS: NGC vs SGC")
    pr("=" * 60)

    ngc_mask = rev["cap"] == "NGC"
    sgc_mask = rev["cap"] == "SGC"

    if ngc_mask.sum() > 10 and sgc_mask.sum() > 10:
        prof_ngc = stack_kappa_profile(kappa_map, mask_bin,
                                       rev["ra"][ngc_mask], rev["dec"][ngc_mask],
                                       rev["theta_v"][ngc_mask], label="NGC")
        prof_sgc = stack_kappa_profile(kappa_map, mask_bin,
                                       rev["ra"][sgc_mask], rev["dec"][sgc_mask],
                                       rev["theta_v"][sgc_mask], label="SGC")
        hl_ngc = compute_headline(prof_ngc)
        hl_sgc = compute_headline(prof_sgc)
        comp_ngc = compute_compensated(prof_ngc["sum_kappa"], prof_ngc["sum_weight"],
                                       prof_ngc["r_rv"])
        comp_sgc = compute_compensated(prof_sgc["sum_kappa"], prof_sgc["sum_weight"],
                                       prof_sgc["r_rv"])
        pr(f"  NGC: comp={comp_ngc['comp_mean']:.4e} ({comp_ngc['snr']:.1f}sigma, {comp_ngc['n_used']}v)")
        pr(f"  SGC: comp={comp_sgc['comp_mean']:.4e} ({comp_sgc['snr']:.1f}sigma, {comp_sgc['n_used']}v)")

        plot_ngc_sgc(prof_ngc, prof_sgc,
                     OUT_DIR / "testD_ngc_vs_sgc.png",
                     "NGC vs SGC consistency [REVOLVER]")
    else:
        pr("  Insufficient voids in one cap, skipping NGC/SGC split.")
        prof_ngc = prof_sgc = hl_ngc = hl_sgc = None
        comp_ngc = comp_sgc = None

    # -- Step 9: Redshift bins ----------------------------------------
    pr("\n" + "=" * 60)
    pr("SYSTEMATICS: Redshift bins")
    pr("=" * 60)

    z_edges = [0.0, 0.10, 0.16, 0.25]
    zbin_profiles = {}
    for j in range(len(z_edges) - 1):
        zlo, zhi = z_edges[j], z_edges[j + 1]
        zmask = (rev["z"] >= zlo) & (rev["z"] < zhi)
        label = f"z=[{zlo:.2f},{zhi:.2f})"
        n_in = zmask.sum()
        if n_in < 20:
            pr(f"  {label}: only {n_in} voids, skipping")
            continue
        prof_z = stack_kappa_profile(kappa_map, mask_bin,
                                     rev["ra"][zmask], rev["dec"][zmask],
                                     rev["theta_v"][zmask], label=label)
        comp_z = compute_compensated(prof_z["sum_kappa"], prof_z["sum_weight"],
                                     prof_z["r_rv"])
        zbin_profiles[label] = prof_z
        pr(f"  {label}: {n_in} voids, comp={comp_z['comp_mean']:.4e} "
           f"({comp_z['snr']:.1f}sigma, {comp_z['n_used']}v)")

    if zbin_profiles:
        plot_zbins(zbin_profiles,
                   OUT_DIR / "testD_zbins.png",
                   "Redshift-binned profiles [REVOLVER]")

    # -- Step 10: Catalogue comparison --------------------------------
    pr("\n" + "=" * 60)
    pr("CATALOGUE COMPARISON")
    pr("=" * 60)

    all_profiles = {"revolver": rev_prof}

    if vide["n_good"] > 10:
        vide_prof = stack_kappa_profile(kappa_map, mask_bin,
                                        vide["ra"], vide["dec"],
                                        vide["theta_v"], label="VIDE")
        vide_hl = compute_headline(vide_prof)
        vide_comp = compute_compensated(vide_prof["sum_kappa"], vide_prof["sum_weight"],
                                        vide_prof["r_rv"])
        all_profiles["vide"] = vide_prof
        pr(f"  VIDE: comp={vide_comp['comp_mean']:.4e} ({vide_comp['snr']:.1f}sigma, "
           f"{vide_comp['n_used']}v)")
    else:
        vide_hl = vide_comp = None

    if vf["n_good"] > 10:
        vf_prof = stack_kappa_profile(kappa_map, mask_bin,
                                      vf["ra"], vf["dec"],
                                      vf["theta_v"], label="VoidFinder")
        vf_hl = compute_headline(vf_prof)
        vf_comp = compute_compensated(vf_prof["sum_kappa"], vf_prof["sum_weight"],
                                      vf_prof["r_rv"])
        all_profiles["voidfinder"] = vf_prof
        pr(f"  VoidFinder: comp={vf_comp['comp_mean']:.4e} ({vf_comp['snr']:.1f}sigma, "
           f"{vf_comp['n_used']}v)")
    else:
        vf_hl = vf_comp = None

    if len(all_profiles) > 1:
        plot_comparison(all_profiles,
                        OUT_DIR / "testD_catalogue_comparison.png",
                        "Catalogue comparison: CMB lensing x voids [Planck PR4]")

    # -- Step 11: JSON summary ----------------------------------------
    pr("\n" + "=" * 60)
    pr("OUTPUT")
    pr("=" * 60)

    def prof_to_json(prof, headline):
        return dict(
            r_rv=prof["r_rv"].tolist(),
            kappa=prof["kappa"].tolist(),
            kappa_err=prof["kappa_err"].tolist(),
            n_pixels=prof["n_pixels"].tolist(),
            n_voids=prof["n_voids"],
            headline=headline,
        )

    def comp_to_json(comp):
        return dict(
            comp_mean=comp["comp_mean"],
            comp_err=comp["comp_err"],
            snr=comp["snr"],
            n_used=comp["n_used"],
        )

    summary = dict(
        test="Phase 6 Test D v2: CMB lensing x DESI voids (compensated)",
        date=datetime.now().strftime("%Y-%m-%d"),
        seed=args.seed,
        version="v2",
        planck=dict(
            product="PR4 (Carron+2022)",
            estimator="MV (minimum variance)",
            nside=NSIDE,
            lmax=LMAX,
            fwhm_deg=FWHM_DEG,
            mask_frac_sky=float(mask_bin.sum() / len(mask_bin)),
            alm_type="kappa (convergence)",
        ),
        mask_strategy="per-pixel: no hard void rejection, masked pixels skipped in stacking",
        compensated=dict(
            centre_range=f"R/Rv < {R_CENTRE_MAX}",
            outer_range=f"{R_OUTER_LO} < R/Rv < {R_OUTER_HI}",
            min_pix_centre=MIN_PIX_CENTRE,
            min_pix_outer=MIN_PIX_OUTER,
            **comp_to_json(rev_comp),
        ),
        void_catalogues=dict(
            revolver=dict(n_total=rev["n_total"],
                         centre_unmasked=int(rev["centre_unmasked"].sum()),
                         z_range=[float(rev["z"].min()), float(rev["z"].max())],
                         z_median=float(np.median(rev["z"])),
                         R_median_mpc_h=float(np.median(rev["R"]))),
            vide=dict(n_total=vide["n_total"],
                     centre_unmasked=int(vide["centre_unmasked"].sum())),
            voidfinder=dict(n_total=vf["n_total"],
                           centre_unmasked=int(vf["centre_unmasked"].sum())),
        ),
        primary_result=prof_to_json(rev_prof, rev_hl),
        bootstrap=dict(
            n_boot=N_BOOTSTRAP,
            delta_kappa=dict(mean=boot["delta_mean"], std=boot["delta_std"]),
            compensated=dict(mean=boot["comp_mean"], std=boot["comp_std"]),
        ),
        null_ra_scramble=dict(
            n_iter=N_RA_SCRAMBLE,
            p_comp=float(p_ra_comp),
            p_delta=float(p_ra_delta),
            p_centre=float(p_ra_centre),
            comp_mean=null_ra["comp_mean"],
            comp_std=null_ra["comp_std"],
            delta_mean=null_ra["delta_mean"],
            delta_std=null_ra["delta_std"],
        ),
        null_random=dict(
            n_iter=N_RANDOM_ITER,
            p_comp=float(p_rand_comp),
            p_delta=float(p_rand_delta),
            p_centre=float(p_rand_centre),
            comp_mean=null_rand["comp_mean"],
            comp_std=null_rand["comp_std"],
            delta_mean=null_rand["delta_mean"],
            delta_std=null_rand["delta_std"],
        ),
        lowl_robustness={},
        headline_ranges=dict(
            centre=f"R/Rv < {R_CENTRE_MAX}",
            ring=f"{R_RING_LO} < R/Rv < {R_RING_HI}",
            outer=f"{R_OUTER_LO} < R/Rv < {R_OUTER_HI}",
        ),
    )

    # Low-l robustness
    for lmin in LMIN_CUTS:
        tag = f"l>={lmin}"
        if tag in lowl_comp:
            summary["lowl_robustness"][f"lmin_{lmin}"] = comp_to_json(lowl_comp[tag])

    if vide_hl:
        summary["vide_result"] = prof_to_json(vide_prof, vide_hl)
        summary["vide_result"]["compensated"] = comp_to_json(vide_comp)
    if vf_hl:
        summary["voidfinder_result"] = prof_to_json(vf_prof, vf_hl)
        summary["voidfinder_result"]["compensated"] = comp_to_json(vf_comp)

    json_path = OUT_DIR / "phase6_testD_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    pr(f"  JSON: {json_path.name}")

    # README
    readme_path = OUT_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Phase 6 Test D v2: CMB Lensing x DESI Voids (compensated)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Planck:** PR4 MV (Carron+2022), NSIDE={NSIDE}, "
                f"lmax={LMAX}, FWHM={FWHM_DEG} deg\n")
        f.write(f"**Voids:** DESIVAST BGS (REVOLVER primary, "
                f"{rev['n_total']} total, {rev_comp['n_used']} with compensated stat)\n\n")
        f.write("## Method\n\n")
        f.write("Stack Planck CMB lensing convergence (kappa) in radial R/Rv bins\n"
                "around DESIVAST void centres. Mean field subtracted from alm.\n"
                "Gaussian smoothing FWHM=0.5 deg applied in harmonic space.\n"
                "**No hard mask cut**: all voids kept; masked pixels skipped per-pixel.\n"
                f"Primary statistic: compensated filter = kappa(R/Rv<{R_CENTRE_MAX})"
                f" - kappa({R_OUTER_LO}<R/Rv<{R_OUTER_HI}) per void.\n\n")
        f.write("## Primary Result (REVOLVER)\n\n")
        f.write(f"- Compensated (centre - outer): "
                f"{rev_comp['comp_mean']:.4e} +/- {rev_comp['comp_err']:.4e} "
                f"({rev_comp['snr']:.1f}sigma, {rev_comp['n_used']} voids)\n")
        f.write(f"- kappa_centre (R/Rv<{R_CENTRE_MAX}): "
                f"{rev_hl['kappa_centre']:.4e} +/- {rev_hl['kappa_centre_err']:.4e} "
                f"({rev_hl['snr_centre']:.1f}sigma)\n")
        f.write(f"- delta_kappa (ring-centre): "
                f"{rev_hl['delta_kappa']:.4e} +/- {rev_hl['delta_kappa_err']:.4e} "
                f"({rev_hl['snr_delta']:.1f}sigma)\n\n")
        f.write("## Null Tests (compensated statistic)\n\n")
        f.write(f"- RA-scramble ({N_RA_SCRAMBLE} iter): p(comp)={p_ra_comp:.3f}\n")
        f.write(f"- Random positions ({N_RANDOM_ITER} iter): p(comp)={p_rand_comp:.3f}\n\n")
        f.write("## Low-l Robustness\n\n")
        f.write("| Map | Comp mean | Comp err | S/N | n_used |\n")
        f.write("|-----|-----------|----------|-----|--------|\n")
        for tag in ["baseline"] + [f"l>={lmin}" for lmin in LMIN_CUTS]:
            c = lowl_comp[tag]
            f.write(f"| {tag} | {c['comp_mean']:.4e} | {c['comp_err']:.4e} | "
                    f"{c['snr']:.1f} | {c['n_used']} |\n")
    pr(f"  README: {readme_path.name}")

    # Manifest
    manifest = {}
    for p in sorted(OUT_DIR.glob("*")):
        if p.name != "manifest.json" and p.is_file():
            manifest[p.name] = sha256_file(p)
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    pr(f"  Manifest: {len(manifest)} files hashed")

    # -- Final summary -----------------------------------------------
    pr("\n" + "=" * 60)
    pr("Pipeline complete (v2).")
    pr(f"  Voids: {rev['n_total']} total, {rev_comp['n_used']} with compensated stat")
    pr(f"  COMPENSATED = {rev_comp['comp_mean']:.4e} +/- {rev_comp['comp_err']:.4e}"
       f"  ({rev_comp['snr']:.1f}sigma)")
    pr(f"  delta_kappa  = {rev_hl['delta_kappa']:.4e} +/- {rev_hl['delta_kappa_err']:.4e}"
       f"  ({rev_hl['snr_delta']:.1f}sigma)")
    pr(f"  RA-scramble null: p(comp) = {p_ra_comp:.3f}")
    pr(f"  Random null:      p(comp) = {p_rand_comp:.3f}")
    pr(f"  Low-l robustness: lmin=20 comp={lowl_comp['l>=20']['comp_mean']:.4e}, "
       f"lmin=30 comp={lowl_comp['l>=30']['comp_mean']:.4e}")
    pr("=" * 60)


if __name__ == "__main__":
    main()
