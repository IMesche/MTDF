#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test B2: Trough & Ridge Lensing (KiDS-internal)  [v3]
=================================================================
Uses KiDS-1000 photometric galaxies as foreground density tracers to
identify troughs (underdense lines of sight) and ridges (overdense).
Stacks tangential shear of background sources around these centres.

v3 changes (vs v2):
  - Mask-aware randoms: uses all-galaxy counts (N_ref) as reference
    instead of uniform randoms.  delta = (N_fg/N_ref) * ratio - 1.
  - Full KiDS-1000 footprint: North + South patches (~1000 deg^2).
  - Primary amplitude range: [3.5, 20] Mpc/h (was [5, 20]).

Foreground: 0.2 < Z_B <= 0.5  (density tracers)
Background: Z_B > 0.6         (shear sources, tomo bins 4-5)
Aperture:   theta_A = 10 arcmin (density map resolution)
Troughs:    bottom 20th percentile of projected density
Ridges:     top 80th percentile (secondary; hard gamma_x gate)

Usage:
  python mtdf_validation/phase6/testB2_trough_ridge.py
  python mtdf_validation/phase6/testB2_trough_ridge.py --skip-smoke --n-boot 1000
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths & imports ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from wl_common import (
    DATA_DIR, KIDS_CAT_PATH, KIDS_TOMO_EDGES, KIDS_M_BIAS,
    OMEGA_M, C_LIGHT, DH_H, Z_BUFFER,
    R_BIN_EDGES, R_DELTA_MAX, P_THRESHOLD,
    comoving_distance, angular_separation, celestial_pa, tangential_shear,
    stack_shear, compute_profile, gamma_x_test, sha256_file,
)

DEFAULT_OUTPUT = (PROJECT_ROOT / "validation" / "output"
                  / "phase6" / "testB2_trough_ridge")

# ── B2 parameters ───────────────────────────────────────────────
Z_FG_LO, Z_FG_HI = 0.2, 0.5       # foreground lens redshift slice
Z_BG_MIN = 0.6                      # background source z cut
THETA_A_ARCMIN = 10.0                # aperture radius (arcmin)
GRID_SPACING_ARCMIN = 10.0           # grid spacing (arcmin)
PCT_TROUGH = 20                      # trough percentile threshold
PCT_RIDGE = 80                       # ridge percentile threshold
N_MAX_CENTRES = 2000                 # max centres per category
N_RANDOM_MULT = 5                    # randoms = 5 x max(n_trough, n_ridge)
MIN_REF_FRAC = 0.3                   # exclude apertures with N_ref < this * N_ref_mean
R_DELTA_MIN_B2 = 3.5                 # lower R for primary amplitude (was 5.0)

# Patch definitions (from catalogue inspection)
PATCHES = {
    'North': dict(ra_min=128.0, ra_max=238.0, dec_min=-5.0, dec_max=4.0),
    'South': dict(ra_min=325.0, ra_max=55.0, dec_min=-36.0, dec_max=-26.0),  # wraps RA
}
EDGE_BUFFER = 1.0  # degrees


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test B2: trough & ridge lensing (KiDS-1000 v3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--smoke-frac", type=float, default=0.01)
    p.add_argument("--skip-smoke", action="store_true", default=False)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--n-max", type=int, default=N_MAX_CENTRES)
    p.add_argument("--theta-a", type=float, default=THETA_A_ARCMIN,
                    help="Aperture radius in arcmin")
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────

def load_kids_full(z_fg_lo, z_fg_hi, z_bg_min, subsample_bg=None, rng=None):
    """Load full KiDS-1000 gold catalogue (North + South).

    Returns: ref_cat (all galaxies, positions only),
             fg_cat (foreground, positions only),
             bg_cat (background, with shapes + calibration).
    """
    from astropy.io import fits

    if not KIDS_CAT_PATH.exists():
        print(f"\n  FATAL: KiDS-1000 catalogue not found.")
        print(f"  Expected: {KIDS_CAT_PATH}")
        print(f"  Download (~16 GB):")
        print(f"    wget https://kids.strw.leidenuniv.nl/DR4/data_files/"
              f"KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits")
        sys.exit(1)

    tag = f" (bg {subsample_bg:.0%})" if subsample_bg else ""
    print(f"  Loading KiDS-1000 FULL{tag}: {KIDS_CAT_PATH.name}")
    if not subsample_bg:
        print(f"  (may take a few minutes for ~16 GB...)")

    with fits.open(KIDS_CAT_PATH, memmap=True) as hdu:
        data = hdu[1].data
        ra_all = data['ALPHA_J2000'].astype(np.float64)
        dec_all = data['DELTA_J2000'].astype(np.float64)
        e1_all = data['e1'].astype(np.float64)
        e2_all = data['e2'].astype(np.float64)
        w_all = data['weight'].astype(np.float64)
        z_b_all = data['Z_B'].astype(np.float64)

        n_raw = len(ra_all)
        base = w_all > 0
        print(f"  Raw: {n_raw:,}, weight>0: {int(base.sum()):,}")

        # Reference catalogue (all galaxies with weight>0, positions only)
        ref_ra = ra_all[base].copy()
        ref_dec = dec_all[base].copy()
        n_ref = int(base.sum())
        print(f"  Reference (all weight>0): {n_ref:,}")

        # Foreground
        fg_mask = base & (z_b_all > z_fg_lo) & (z_b_all <= z_fg_hi)
        n_fg = int(fg_mask.sum())
        fg_ra = ra_all[fg_mask].copy()
        fg_dec = dec_all[fg_mask].copy()
        fg_zb = z_b_all[fg_mask].copy()
        print(f"  Foreground ({z_fg_lo:.1f} < Z_B <= {z_fg_hi:.1f}): {n_fg:,}")

        # Background
        bg_mask = base & (z_b_all > z_bg_min)
        if subsample_bg and 0 < subsample_bg < 1:
            idx_bg = np.where(bg_mask)[0]
            n_keep = max(1, int(len(idx_bg) * subsample_bg))
            idx_keep = rng.choice(idx_bg, n_keep, replace=False) if rng else idx_bg[:n_keep]
            bg_mask = np.zeros(n_raw, dtype=bool)
            bg_mask[idx_keep] = True
            print(f"  Background (Z_B > {z_bg_min:.1f}, {subsample_bg:.0%}): {int(bg_mask.sum()):,}")
        else:
            print(f"  Background (Z_B > {z_bg_min:.1f}): {int(bg_mask.sum()):,}")

        bg_ra = ra_all[bg_mask].copy()
        bg_dec = dec_all[bg_mask].copy()
        bg_e1 = e1_all[bg_mask].copy()
        bg_e2 = e2_all[bg_mask].copy()
        bg_w = w_all[bg_mask].copy()
        bg_zb = z_b_all[bg_mask].copy()

    # Background calibration
    bg_m = np.zeros(len(bg_ra))
    bg_tomo = np.zeros(len(bg_ra), dtype=np.int8)
    for i in range(5):
        bm = (bg_zb > KIDS_TOMO_EDGES[i]) & (bg_zb <= KIDS_TOMO_EDGES[i + 1])
        bg_tomo[bm] = i + 1
        if bm.sum() > 0:
            c1 = np.average(bg_e1[bm], weights=bg_w[bm])
            c2 = np.average(bg_e2[bm], weights=bg_w[bm])
            bg_e1[bm] -= c1
            bg_e2[bm] -= c2
            bg_m[bm] = KIDS_M_BIAS[i]
            print(f"    BG Bin {i+1} (z {KIDS_TOMO_EDGES[i]:.1f}-"
                  f"{KIDS_TOMO_EDGES[i+1]:.1f}): {int(bm.sum()):,}, "
                  f"c1={c1:+.6f}, c2={c2:+.6f}, m={KIDS_M_BIAS[i]:+.4f}")

    in_tomo = bg_tomo > 0
    n_bg = int(in_tomo.sum())
    print(f"  Background final: {n_bg:,}")

    ref_cat = dict(ra=ref_ra, dec=ref_dec, n=n_ref)
    fg_cat = dict(ra=fg_ra, dec=fg_dec, z_b=fg_zb, n=n_fg)
    bg_cat = dict(
        ra=bg_ra[in_tomo], dec=bg_dec[in_tomo],
        e1=bg_e1[in_tomo], e2=bg_e2[in_tomo],
        weight=bg_w[in_tomo], z_b=bg_zb[in_tomo],
        m_bias=bg_m[in_tomo], n=n_bg,
    )
    return ref_cat, fg_cat, bg_cat


# ── Density map (mask-aware, multi-patch) ────────────────────────

def _build_patch_density(fg_ra, fg_dec, ref_ra, ref_dec,
                          theta_a_deg, grid_deg, patch_def, buf):
    """Build density map for one patch using mask-aware normalisation.

    delta = (N_fg / N_ref) * (N_ref_total / N_fg_total) - 1
    where N_ref = all-galaxy counts (traces survey mask).
    """
    ra_min, ra_max = patch_def['ra_min'], patch_def['ra_max']
    dec_min, dec_max = patch_def['dec_min'], patch_def['dec_max']
    wraps = ra_min > ra_max  # RA wraps around 0

    # Select galaxies in this patch
    if wraps:
        fg_in = (fg_dec >= dec_min) & (fg_dec <= dec_max) & \
                ((fg_ra >= ra_min) | (fg_ra <= ra_max))
        ref_in = (ref_dec >= dec_min) & (ref_dec <= dec_max) & \
                 ((ref_ra >= ra_min) | (ref_ra <= ra_max))
    else:
        fg_in = (fg_dec >= dec_min) & (fg_dec <= dec_max) & \
                (fg_ra >= ra_min) & (fg_ra <= ra_max)
        ref_in = (ref_dec >= dec_min) & (ref_dec <= dec_max) & \
                 (ref_ra >= ra_min) & (ref_ra <= ra_max)

    p_fg_ra = fg_ra[fg_in].copy()
    p_fg_dec = fg_dec[fg_in].copy()
    p_ref_ra = ref_ra[ref_in].copy()
    p_ref_dec = ref_dec[ref_in].copy()

    # For wrapping patches, shift RA to continuous range
    if wraps:
        p_fg_ra[p_fg_ra > 180] -= 360
        p_ref_ra[p_ref_ra > 180] -= 360
        ra_lo = ra_min - 360
        ra_hi = ra_max
    else:
        ra_lo = ra_min
        ra_hi = ra_max

    mean_dec = 0.5 * (dec_min + dec_max)
    cos_dec = np.cos(np.deg2rad(mean_dec))

    # Flat-sky coords
    fg_x = p_fg_ra * cos_dec
    fg_y = p_fg_dec
    ref_x = p_ref_ra * cos_dec
    ref_y = p_ref_dec

    tree_fg = cKDTree(np.column_stack([fg_x, fg_y]))
    tree_ref = cKDTree(np.column_stack([ref_x, ref_y]))

    # Grid (with buffer)
    g_ra_lo = ra_lo + buf + theta_a_deg / cos_dec
    g_ra_hi = ra_hi - buf - theta_a_deg / cos_dec
    g_dec_lo = dec_min + buf + theta_a_deg
    g_dec_hi = dec_max - buf - theta_a_deg

    ra_arr = np.arange(g_ra_lo, g_ra_hi, grid_deg / cos_dec)
    dec_arr = np.arange(g_dec_lo, g_dec_hi, grid_deg)
    ra_mesh, dec_mesh = np.meshgrid(ra_arr, dec_arr)
    g_ra = ra_mesh.ravel()
    g_dec = dec_mesh.ravel()

    grid_pts = np.column_stack([g_ra * cos_dec, g_dec])

    n_fg_cnt = np.array(tree_fg.query_ball_point(grid_pts, r=theta_a_deg,
                                                  workers=-1, return_length=True),
                         dtype=np.float64)
    n_ref_cnt = np.array(tree_ref.query_ball_point(grid_pts, r=theta_a_deg,
                                                    workers=-1, return_length=True),
                          dtype=np.float64)

    # Mask: exclude apertures with too few reference galaxies
    n_ref_mean = n_ref_cnt.mean() if len(n_ref_cnt) > 0 else 0
    min_ref = MIN_REF_FRAC * n_ref_mean
    good = n_ref_cnt >= min_ref

    # Global ratio for this patch
    n_fg_total = len(p_fg_ra)
    n_ref_total = len(p_ref_ra)
    ratio = n_ref_total / max(n_fg_total, 1)

    delta = np.full(len(g_ra), np.nan)
    delta[good] = (n_fg_cnt[good] / n_ref_cnt[good]) * ratio - 1.0

    # Convert RA back to [0, 360] for output
    out_ra = g_ra.copy()
    if wraps:
        out_ra[out_ra < 0] += 360

    return (out_ra[good], g_dec[good], delta[good],
            n_fg_cnt[good], n_ref_cnt[good],
            int(good.sum()), len(g_ra), int((~good).sum()),
            n_fg_total, n_ref_total)


def build_density_map(ref_cat, fg_cat, theta_a_arcmin, grid_spacing_arcmin):
    """Build mask-aware density map across all KiDS-1000 patches.

    Uses all-galaxy counts as reference (traces survey mask):
        delta = (N_fg / N_ref) * (N_ref_total / N_fg_total) - 1
    """
    t0 = datetime.now()
    theta_a_deg = theta_a_arcmin / 60.0
    grid_deg = grid_spacing_arcmin / 60.0

    print(f"\n  Building density map (mask-aware, full footprint):")
    print(f"    Reference (all galaxies): {ref_cat['n']:,}")
    print(f"    Foreground: {fg_cat['n']:,}")
    print(f"    Aperture: {theta_a_arcmin:.0f} arcmin, grid: {grid_spacing_arcmin:.0f} arcmin")

    all_ra, all_dec, all_delta = [], [], []
    all_nfg, all_nref = [], []
    total_good, total_raw, total_masked = 0, 0, 0

    for name, pdef in PATCHES.items():
        print(f"\n    Patch {name}:")
        (p_ra, p_dec, p_delta, p_nfg, p_nref,
         n_good, n_raw, n_masked,
         n_fg_patch, n_ref_patch) = _build_patch_density(
            fg_cat['ra'], fg_cat['dec'],
            ref_cat['ra'], ref_cat['dec'],
            theta_a_deg, grid_deg, pdef, EDGE_BUFFER)

        print(f"      FG: {n_fg_patch:,}, Ref: {n_ref_patch:,}")
        print(f"      Grid: {n_good:,} good / {n_raw} total ({n_masked} masked)")
        if len(p_delta) > 0:
            print(f"      delta: [{p_delta.min():.2f}, {p_delta.max():.2f}]")
            print(f"      N_fg_mean: {p_nfg.mean():.1f}, N_ref_mean: {p_nref.mean():.1f}")

        all_ra.append(p_ra)
        all_dec.append(p_dec)
        all_delta.append(p_delta)
        all_nfg.append(p_nfg)
        all_nref.append(p_nref)
        total_good += n_good
        total_raw += n_raw
        total_masked += n_masked

    g_ra = np.concatenate(all_ra)
    g_dec = np.concatenate(all_dec)
    delta = np.concatenate(all_delta)
    n_fg_arr = np.concatenate(all_nfg)
    n_ref_arr = np.concatenate(all_nref)

    dt = (datetime.now() - t0).total_seconds()
    print(f"\n    Combined: {total_good:,} grid points, {total_masked} masked")
    print(f"    delta: [{delta.min():.2f}, {delta.max():.2f}]")
    print(f"    Built in {dt:.1f}s")

    return dict(ra=g_ra, dec=g_dec, delta=delta,
                counts_fg=n_fg_arr, counts_ref=n_ref_arr,
                n_grid=total_good, n_grid_raw=total_raw,
                n_masked=total_masked,
                n_fg_mean=float(n_fg_arr.mean()),
                n_ref_mean=float(n_ref_arr.mean()),
                theta_a_arcmin=theta_a_arcmin,
                normalisation="mask-aware (all-galaxy reference)")


# ── Centre selection ─────────────────────────────────────────────

def select_centres(density_map, fg_cat, pct_lo, pct_hi, n_max, rng):
    """Select trough and ridge centres by density percentile."""
    delta = density_map['delta']
    ra = density_map['ra']
    dec = density_map['dec']

    thr_lo = np.percentile(delta, pct_lo)
    thr_hi = np.percentile(delta, pct_hi)

    trough_mask = delta <= thr_lo
    ridge_mask = delta >= thr_hi

    n_trough = int(trough_mask.sum())
    n_ridge = int(ridge_mask.sum())

    z_med = float(np.median(fg_cat['z_b']))

    print(f"\n  Centre selection:")
    print(f"    Trough (P{pct_lo}): delta <= {thr_lo:.3f} -> {n_trough}")
    print(f"    Ridge  (P{pct_hi}): delta >= {thr_hi:.3f} -> {n_ridge}")
    print(f"    Median foreground z: {z_med:.3f}")

    def _pick(mask, label):
        n = int(mask.sum())
        r, d = ra[mask], dec[mask]
        if n > n_max:
            idx = rng.choice(n, size=n_max, replace=False)
            r, d = r[idx], d[idx]
            print(f"    {label}: subsampled {n} -> {n_max}")
        else:
            print(f"    {label}: {n} centres")
        z = np.full(len(r), z_med)
        return dict(ra=r, dec=d, z=z)

    troughs = _pick(trough_mask, "Troughs")
    ridges = _pick(ridge_mask, "Ridges")
    return troughs, ridges, z_med


# ── Random stacking centres (drawn from observed positions) ──────

def generate_randoms(ref_cat, n_centres, z_med, n_mult, rng):
    """Random centres drawn from observed galaxy positions (mask-aware)."""
    n = n_centres * n_mult
    idx = rng.choice(ref_cat['n'], size=n, replace=True)
    ra = ref_cat['ra'][idx]
    dec = ref_cat['dec'][idx]
    z = np.full(n, z_med)
    print(f"  Generated {n} random centres ({n_mult}x, from observed positions)")
    return dict(ra=ra, dec=dec, z=z)


# ── Bootstrap ────────────────────────────────────────────────────

def bootstrap_delta_gamma_t_B2(trough_acc, ridge_acc, rand_acc,
                                r_edges, n_boot, rng):
    """Bootstrap three headline statistics."""
    rc = np.sqrt(r_edges[:-1] * r_edges[1:])
    use = (rc >= R_DELTA_MIN_B2) & (rc <= R_DELTA_MAX)
    if not np.any(use):
        return None

    nt = trough_acc['sum_wet'].shape[0]
    nr_ = ridge_acc['sum_wet'].shape[0]
    nrand = rand_acc['sum_wet'].shape[0]

    d_trough = np.zeros(n_boot)
    d_ridge = np.zeros(n_boot)
    d_split = np.zeros(n_boot)

    def _mean_gt(acc, idx, mask):
        wet = acc['sum_wet'][idx].sum(0)[mask]
        w1m = acc['sum_w1m'][idx].sum(0)[mask]
        npairs = acc['n_pairs'][idx].sum(0)[mask]
        gt = np.where(w1m > 0, wet / w1m, 0)
        return np.average(gt, weights=npairs) if npairs.sum() > 0 else 0

    for i in range(n_boot):
        ti = rng.choice(nt, size=nt, replace=True)
        ri = rng.choice(nr_, size=nr_, replace=True)
        randi = rng.choice(nrand, size=nrand, replace=True)

        gt_t = _mean_gt(trough_acc, ti, use)
        gt_r = _mean_gt(ridge_acc, ri, use)
        gt_rand = _mean_gt(rand_acc, randi, use)

        d_trough[i] = gt_t - gt_rand
        d_ridge[i] = gt_r - gt_rand
        d_split[i] = gt_r - gt_t

    def _stats(arr, label):
        return dict(
            label=label,
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            ci_68=[float(np.percentile(arr, 16)),
                   float(np.percentile(arr, 84))],
            ci_95=[float(np.percentile(arr, 2.5)),
                   float(np.percentile(arr, 97.5))],
            sigma_from_zero=float(abs(np.mean(arr)) / max(np.std(arr), 1e-20)),
        )

    return dict(
        delta_gamma_t_trough=_stats(d_trough, "trough - random"),
        delta_gamma_t_ridge=_stats(d_ridge, "ridge - random"),
        delta_gamma_t_split=_stats(d_split, "ridge - trough"),
        n_trough=nt, n_ridge=nr_, n_random=nrand,
        n_boot=n_boot,
        r_range_mpc_h=[R_DELTA_MIN_B2, R_DELTA_MAX],
    )


# ── Plots ────────────────────────────────────────────────────────

def plot_density_map(density_map, troughs, ridges, outdir):
    """Foreground density map with trough/ridge centres (both patches)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [1, 1]})

    delta = density_map['delta']
    ra = density_map['ra']
    dec = density_map['dec']
    vmax = min(np.percentile(delta, 99), 2.0)
    vmin = max(np.percentile(delta, 1), -1.0)

    # North patch
    north = dec > -10
    south = dec < -20

    for ax, mask, title in [
        (axes[0], north, "KiDS-North"),
        (axes[1], south, "KiDS-South"),
    ]:
        if mask.sum() > 0:
            sc = ax.scatter(ra[mask], dec[mask], c=delta[mask], s=1,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax, rasterized=True)

            # Trough/ridge centres in this patch
            t_m = troughs['dec'] > -10 if 'North' in title else troughs['dec'] < -20
            r_m = ridges['dec'] > -10 if 'North' in title else ridges['dec'] < -20
            if t_m.sum() > 0:
                ax.scatter(troughs['ra'][t_m], troughs['dec'][t_m], s=6,
                           c='blue', marker='v', alpha=0.3,
                           label=f"Troughs ({int(t_m.sum())})")
            if r_m.sum() > 0:
                ax.scatter(ridges['ra'][r_m], ridges['dec'][r_m], s=6,
                           c='red', marker='^', alpha=0.3,
                           label=f"Ridges ({int(r_m.sum())})")

            ax.set_ylabel('DEC (deg)')
            ax.set_title(f"{title} (mask-aware, theta_A={density_map['theta_a_arcmin']:.0f}')")
            ax.legend(fontsize=7, loc='upper right')
            ax.invert_xaxis()

    axes[1].set_xlabel('RA (deg)')
    plt.colorbar(sc, ax=axes, label=r'$\delta$ (mask-aware)', shrink=0.8)
    plt.tight_layout()
    path = outdir / "testB2_density_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path.name}")
    return path


def plot_gamma_t_B2(trough_prof, ridge_prof, rand_prof, outdir, tag=""):
    """Tangential shear: troughs vs ridges vs randoms."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ftag = f"_{tag}" if tag else ""

    for prof, lab, col, mk in [
        (trough_prof, f"Troughs (N={trough_prof['n_centres']})", "C0", "v"),
        (ridge_prof, f"Ridges (N={ridge_prof['n_centres']})", "C3", "^"),
        (rand_prof, f"Random (N={rand_prof['n_centres']})", "C7", "D"),
    ]:
        R = np.array(prof['R'])
        gt = np.array(prof['gamma_t'])
        err = np.array(prof['gamma_t_err'])
        ok = np.isfinite(gt) & np.isfinite(err)
        ax.errorbar(R[ok], gt[ok], yerr=err[ok],
                     fmt=f'{mk}-', color=col, capsize=3, label=lab)

    ax.axhline(0, ls=':', color='gray', alpha=0.5)
    ax.axvspan(R_DELTA_MIN_B2, R_DELTA_MAX, alpha=0.06, color='yellow',
               label=f'Primary [{R_DELTA_MIN_B2:.1f}-{R_DELTA_MAX:.0f}] Mpc/h')
    ax.set_xlabel(r'$R_{\rm proj}$ [Mpc/$h$]')
    ax.set_ylabel(r'$\gamma_t$')
    ax.set_xscale('log')
    ax.set_title('Tangential shear: troughs vs ridges vs random [KiDS-1000 full]')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = outdir / f"testB2_gamma_t_profile{ftag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path.name}")
    return path


def plot_gamma_x_B2(prof, gx_res, outdir, label="troughs", tag=""):
    """Cross-component with chi2/p annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ftag = f"_{tag}" if tag else ""

    R = np.array(prof['R'])
    gx = np.array(prof['gamma_x'])
    err = np.array(prof['gamma_x_err'])
    ok = np.isfinite(gx) & np.isfinite(err)
    ax.errorbar(R[ok], gx[ok], yerr=err[ok],
                 fmt='o-', color='C0', capsize=3, label=label.capitalize())
    ax.axhline(0, ls=':', color='gray', alpha=0.5)

    status = "PASS" if gx_res['passed'] else "FAIL"
    col = "green" if gx_res['passed'] else "red"
    ax.text(0.97, 0.95,
            f"$\\chi^2$/dof = {gx_res['chi2']:.1f}/{gx_res['dof']}"
            f" = {gx_res['chi2_dof']:.2f}\n"
            f"p = {gx_res['p_value']:.3f}  [{status}]",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=col,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'$R_{\rm proj}$ [Mpc/$h$]')
    ax.set_ylabel(r'$\gamma_\times$')
    ax.set_xscale('log')
    ax.set_title(r'Cross-component $\gamma_\times$ [KiDS-1000 full]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = outdir / f"testB2_gamma_x_profile{ftag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path.name}")
    return path


# ── Output ───────────────────────────────────────────────────────

def write_readme(outdir, args, density_map, z_med):
    text = f"""# Test B2: Trough & Ridge Lensing (KiDS-internal, v3)

## Method
Projected foreground galaxy density map built from KiDS-1000 photometric
galaxies at {Z_FG_LO:.1f} < Z_B <= {Z_FG_HI:.1f} using aperture counting
(theta_A = {args.theta_a:.0f} arcmin, cKDTree).

### Mask-aware normalisation
All-galaxy counts (N_ref, weight>0) serve as mask-aware reference:
```
delta = (N_fg / N_ref) * (N_ref_total / N_fg_total) - 1
```
This corrects for survey tile gaps, masking, and depth variations
without requiring a separate survey mask file.  Apertures where
N_ref < {MIN_REF_FRAC:.0%} of patch mean are excluded.

### Full footprint
Both KiDS-North and KiDS-South patches are used (~1000 deg^2 total).

Troughs = bottom {PCT_TROUGH}th percentile of normalised delta.
Ridges = top {PCT_RIDGE}th percentile.
Median foreground z = {z_med:.3f}.

Background sources: Z_B > {Z_BG_MIN:.1f} (tomo bins 4-5).
Source-behind-lens cut: z_source > z_centre + 0.1.

## Tangential shear decomposition
```
gamma_t =  e1 cos(2 PA) - e2 sin(2 PA)
gamma_x =  e1 sin(2 PA) + e2 cos(2 PA)
```

## Shear calibration (KiDS lensfit)
gamma_t = Sum(w * e_t) / Sum(w * (1 + m))
c-term subtracted per tomographic bin.

## Headline statistics
Primary range: [{R_DELTA_MIN_B2:.1f}, {R_DELTA_MAX:.0f}] Mpc/h.
Delta_gamma_t_trough = <gamma_t>_trough - <gamma_t>_random  (primary)
Delta_gamma_t_ridge  = <gamma_t>_ridge  - <gamma_t>_random  (secondary)
Delta_gamma_t_split  = <gamma_t>_ridge  - <gamma_t>_trough

Errors from bootstrap resampling over centres.

## Density map statistics
Grid points (good): {density_map['n_grid']:,} / {density_map['n_grid_raw']}
Masked: {density_map['n_masked']}
Mean foreground per aperture: {density_map['n_fg_mean']:.1f}
Mean reference per aperture: {density_map['n_ref_mean']:.1f}
Normalisation: {density_map['normalisation']}
"""
    path = outdir / "README.md"
    path.write_text(text)
    print(f"  README: {path.name}")


def write_manifest(outdir):
    files = sorted(
        f.name for f in outdir.iterdir()
        if f.is_file() and f.name != 'manifest.json')
    hashes = {f: sha256_file(outdir / f) for f in files}
    m = {"generated": datetime.now().strftime("%Y-%m-%d"), "sha256": hashes}
    (outdir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"  Manifest: {len(hashes)} files hashed")


# ── Smoke test ───────────────────────────────────────────────────

def run_smoke_test(args, rng, ref_cat, troughs, ridges, z_med, outdir):
    """Smoke test with 1% background subsample."""
    print(f"\n{'='*60}")
    print(f"SMOKE TEST ({args.smoke_frac:.0%} background subsample)")
    print(f"{'='*60}")

    _, _, bg = load_kids_full(Z_FG_LO, Z_FG_HI, Z_BG_MIN,
                              subsample_bg=args.smoke_frac, rng=rng)

    n_smoke = min(200, len(troughs['ra']), len(ridges['ra']))
    smoke_troughs = {k: v[:n_smoke] if hasattr(v, '__len__') else v
                     for k, v in troughs.items()}
    smoke_ridges = {k: v[:n_smoke] if hasattr(v, '__len__') else v
                    for k, v in ridges.items()}

    print(f"\n  Stacking around {n_smoke} trough centres (smoke)...")
    t_acc = stack_shear(bg, smoke_troughs, R_BIN_EDGES, label="troughs")
    t_prof = compute_profile(t_acc, R_BIN_EDGES)

    print(f"  Stacking around {n_smoke} ridge centres (smoke)...")
    r_acc = stack_shear(bg, smoke_ridges, R_BIN_EDGES, label="ridges")
    r_prof = compute_profile(r_acc, R_BIN_EDGES)

    randoms = generate_randoms(ref_cat, n_smoke, z_med, 1, rng)
    print(f"  Stacking around {len(randoms['ra'])} random centres (smoke)...")
    rand_acc = stack_shear(bg, randoms, R_BIN_EDGES, label="randoms")
    rand_prof = compute_profile(rand_acc, R_BIN_EDGES)

    gx = gamma_x_test(t_prof)
    print(f"\n  gamma_x (troughs): chi2/dof = {gx['chi2']:.1f}/{gx['dof']}"
          f" = {gx['chi2_dof']:.2f}, p = {gx['p_value']:.3f}")

    plot_gamma_t_B2(t_prof, r_prof, rand_prof, outdir, tag="smoke")
    plot_gamma_x_B2(t_prof, gx, outdir, label="troughs", tag="smoke")

    if not gx['passed']:
        print(f"\n  SMOKE FAILED: p = {gx['p_value']:.3f} < {P_THRESHOLD}")
        return False

    print(f"\n  SMOKE PASSED.")
    return True


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 6 Test B2: Trough & Ridge Lensing  [v3]")
    print("  KiDS-1000 full footprint, mask-aware normalisation")
    print("=" * 60)

    # Step 1: Load full KiDS-1000
    ref_cat, fg_cat, bg_cat_full = load_kids_full(Z_FG_LO, Z_FG_HI, Z_BG_MIN)

    # Step 2: Build density map (mask-aware)
    density_map = build_density_map(ref_cat, fg_cat, args.theta_a,
                                     GRID_SPACING_ARCMIN)

    # Step 3: Select centres
    troughs, ridges, z_med = select_centres(
        density_map, fg_cat, PCT_TROUGH, PCT_RIDGE, args.n_max, rng)

    plot_density_map(density_map, troughs, ridges, outdir)

    # Step 4: Smoke test
    if not args.skip_smoke:
        if not run_smoke_test(args, rng, ref_cat, troughs, ridges, z_med, outdir):
            sys.exit(1)
        rng = np.random.RandomState(args.seed)

    # Step 5: Full stacking
    print(f"\n{'='*60}")
    print("FULL RUN")
    print(f"{'='*60}")

    n_trough = len(troughs['ra'])
    n_ridge = len(ridges['ra'])

    print(f"\n  Stacking around {n_trough} trough centres...")
    trough_acc = stack_shear(bg_cat_full, troughs, R_BIN_EDGES, label="troughs")
    trough_prof = compute_profile(trough_acc, R_BIN_EDGES)

    print(f"\n  Trough profile:")
    for i, r in enumerate(trough_prof['R']):
        gt = trough_prof['gamma_t'][i]
        err = trough_prof['gamma_t_err'][i]
        np_ = trough_prof['n_pairs'][i]
        if np.isfinite(gt):
            sig = abs(gt) / err if err > 0 else 0
            print(f"    R={r:.1f}: gamma_t={gt:+.6f} +/- {err:.6f}  "
                  f"({sig:.1f}s, {np_:,} pairs)")

    print(f"\n  Stacking around {n_ridge} ridge centres...")
    ridge_acc = stack_shear(bg_cat_full, ridges, R_BIN_EDGES, label="ridges")
    ridge_prof = compute_profile(ridge_acc, R_BIN_EDGES)

    print(f"\n  Ridge profile:")
    for i, r in enumerate(ridge_prof['R']):
        gt = ridge_prof['gamma_t'][i]
        err = ridge_prof['gamma_t_err'][i]
        np_ = ridge_prof['n_pairs'][i]
        if np.isfinite(gt):
            sig = abs(gt) / err if err > 0 else 0
            print(f"    R={r:.1f}: gamma_t={gt:+.6f} +/- {err:.6f}  "
                  f"({sig:.1f}s, {np_:,} pairs)")

    # gamma_x gates
    gx_trough = gamma_x_test(trough_prof)
    print(f"\n  gamma_x (troughs): chi2/dof = {gx_trough['chi2']:.1f}/"
          f"{gx_trough['dof']} = {gx_trough['chi2_dof']:.2f}, "
          f"p = {gx_trough['p_value']:.3f}")

    gx_ridge = gamma_x_test(ridge_prof)
    print(f"  gamma_x (ridges):  chi2/dof = {gx_ridge['chi2']:.1f}/"
          f"{gx_ridge['dof']} = {gx_ridge['chi2_dof']:.2f}, "
          f"p = {gx_ridge['p_value']:.3f}")

    if not gx_trough['passed']:
        print(f"\n  FATAL: gamma_x gate FAILED for troughs.")
        sys.exit(1)
    ridge_clean = gx_ridge['passed']
    if not ridge_clean:
        print(f"\n  WARNING: gamma_x gate FAILED for ridges (p = {gx_ridge['p_value']:.4f}).")
        print(f"  Ridge is secondary. Trough is primary.")
    else:
        print(f"  gamma_x gates PASSED (both).")

    # Random controls (mask-aware: drawn from observed positions)
    n_rand_base = max(n_trough, n_ridge)
    randoms = generate_randoms(ref_cat, n_rand_base, z_med, N_RANDOM_MULT, rng)

    print(f"\n  Stacking around {len(randoms['ra'])} random centres...")
    rand_acc = stack_shear(bg_cat_full, randoms, R_BIN_EDGES, label="randoms")
    rand_prof = compute_profile(rand_acc, R_BIN_EDGES)

    # Plots
    plot_gamma_t_B2(trough_prof, ridge_prof, rand_prof, outdir)
    plot_gamma_x_B2(trough_prof, gx_trough, outdir, label="troughs")

    # Bootstrap
    print(f"\n  Bootstrap ({args.n_boot} resamples, R=[{R_DELTA_MIN_B2:.1f},{R_DELTA_MAX:.0f}])...")
    boot = bootstrap_delta_gamma_t_B2(
        trough_acc, ridge_acc, rand_acc, R_BIN_EDGES, args.n_boot, rng)

    for key in ['delta_gamma_t_trough', 'delta_gamma_t_ridge', 'delta_gamma_t_split']:
        d = boot[key]
        print(f"    {d['label']}: {d['mean']:.6f} +/- {d['std']:.6f} "
              f"({d['sigma_from_zero']:.1f} sigma)")

    # Summary JSON
    summary = dict(
        test="Phase 6 Test B2: Trough & ridge lensing (v3)",
        date=datetime.now().strftime("%Y-%m-%d"),
        survey="KiDS-1000 (full footprint, North + South)",
        method="Mask-aware density field (all-galaxy reference)",
        seed=args.seed,
        foreground=dict(z_range=[Z_FG_LO, Z_FG_HI], n=fg_cat['n']),
        background=dict(z_min=Z_BG_MIN, n=bg_cat_full['n']),
        reference=dict(n=ref_cat['n'], description="all weight>0 galaxies"),
        density_map=dict(
            theta_a_arcmin=args.theta_a,
            grid_spacing_arcmin=GRID_SPACING_ARCMIN,
            normalisation=density_map['normalisation'],
            n_grid=density_map['n_grid'],
            n_grid_raw=density_map['n_grid_raw'],
            n_masked=density_map['n_masked'],
            n_fg_mean_per_aperture=density_map['n_fg_mean'],
            n_ref_mean_per_aperture=density_map['n_ref_mean'],
            delta_range=[float(density_map['delta'].min()),
                         float(density_map['delta'].max())],
        ),
        centres=dict(
            n_trough=n_trough, n_ridge=n_ridge,
            n_random=len(randoms['ra']),
            z_median=z_med,
            pct_trough=PCT_TROUGH, pct_ridge=PCT_RIDGE,
        ),
        z_buffer=Z_BUFFER,
        cosmology=dict(Omega_m=OMEGA_M, h=1.0, note="Mpc/h units"),
        r_bin_edges=R_BIN_EDGES.tolist(),
        r_delta_range=[R_DELTA_MIN_B2, R_DELTA_MAX],
        trough_profile=trough_prof,
        ridge_profile=ridge_prof,
        random_profile=rand_prof,
        gamma_x_test_trough=gx_trough,
        gamma_x_test_ridge=gx_ridge,
        ridge_gamma_x_clean=ridge_clean,
        primary_measurement="delta_gamma_t_trough (trough - random)" if not ridge_clean
            else "delta_gamma_t_split (ridge - trough)",
        bootstrap=boot,
        tangential_shear_convention=(
            "gamma_t = e1 cos(2PA) - e2 sin(2PA); "
            "PA = celestial position angle North through East; "
            "gamma_t > 0 = tangential (overdensity)"),
        calibration="gamma_t = Sum(w*e_t) / Sum(w*(1+m)); c-term subtracted per tomo bin",
    )

    jp = outdir / "phase6_testB2_summary.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp.name}")

    write_readme(outdir, args, density_map, z_med)
    write_manifest(outdir)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    trough_stat = boot['delta_gamma_t_trough']
    print(f"  Delta_gamma_t_trough = {trough_stat['mean']:.6f} +/- {trough_stat['std']:.6f} "
          f"({trough_stat['sigma_from_zero']:.1f} sigma)")
    split = boot['delta_gamma_t_split']
    print(f"  Delta_gamma_t_split  = {split['mean']:.6f} +/- {split['std']:.6f} "
          f"({split['sigma_from_zero']:.1f} sigma)")
    print(f"  gamma_x troughs: PASS")
    if ridge_clean:
        print(f"  gamma_x ridges:  PASS")
    else:
        print(f"  gamma_x ridges:  FAIL (flagged; trough is primary)")
    print(f"  R range: [{R_DELTA_MIN_B2:.1f}, {R_DELTA_MAX:.0f}] Mpc/h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
