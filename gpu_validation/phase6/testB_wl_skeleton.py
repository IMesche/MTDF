#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test B: Weak Lensing x Environment (KiDS-1000)
=========================================================
Stacks tangential shear from KiDS-1000 around DESIVAST NGC void centres.
Random control centres provide the null baseline.

Tangential/cross decomposition (celestial PA, North through East):
  gamma_t =  e1 cos(2 PA) - e2 sin(2 PA)   [>0 = tangential; voids expect <0]
  gamma_x =  e1 sin(2 PA) + e2 cos(2 PA)   [should be zero]

Shear calibration (KiDS lensfit):
  gamma_t = Sum(w e_t) / Sum(w (1+m))       per radial bin, stacked over centres

Delta_gamma_t = <gamma_t>_void - <gamma_t>_random  in [5, 20] Mpc/h.

Pipeline:
  1. Load DESIVAST NGC voids + KiDS-North footprint (1 deg buffer)
  2. Load KiDS-1000 gold catalogue (hard fail if missing)
  3. Smoke test (1%): gamma_x consistent with zero (p > 0.05)
  4. Full run: stack voids + randoms, gamma_x gate, Delta_gamma_t

Usage:
  python mtdf_validation/phase6/testB_wl_skeleton.py
  python mtdf_validation/phase6/testB_wl_skeleton.py --skip-smoke --n-boot 1000
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from wl_common import (
    DATA_DIR, KIDS_CAT_PATH, KIDS_TOMO_EDGES, KIDS_M_BIAS,
    KIDS_RA_MIN, KIDS_RA_MAX, KIDS_DEC_MIN, KIDS_DEC_MAX, KIDS_EDGE_BUFFER,
    OMEGA_M, C_LIGHT, DH_H, Z_BUFFER,
    R_BIN_EDGES, R_DELTA_MIN, R_DELTA_MAX, P_THRESHOLD,
    comoving_distance, angular_separation, celestial_pa, tangential_shear,
    stack_shear, compute_profile, gamma_x_test, sha256_file,
)

DEFAULT_OUTPUT = (PROJECT_ROOT / "validation" / "output"
                  / "phase6" / "testB_wl_environment")

# ── DESIVAST ─────────────────────────────────────────────────────
VOID_NGC = DATA_DIR / "External" / "desivast_voids" / \
    "DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits"

N_RANDOM_MULT = 5


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test B: WL x void environment (KiDS-1000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--smoke-frac", type=float, default=0.01)
    p.add_argument("--skip-smoke", action="store_true", default=False)
    p.add_argument("--n-boot", type=int, default=500)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────

def load_voids():
    """Load DESIVAST REVOLVER NGC voids."""
    from astropy.io import fits
    if not VOID_NGC.exists():
        print(f"  FATAL: Void catalogue not found: {VOID_NGC}")
        sys.exit(1)
    with fits.open(VOID_NGC) as hdu:
        d = hdu[1].data
        voids = dict(
            ra=d['RA'].astype(np.float64),
            dec=d['DEC'].astype(np.float64),
            z=d['REDSHIFT'].astype(np.float64),
            radius=d['RADIUS'].astype(np.float64),
        )
    print(f"  Loaded {len(voids['ra'])} voids from {VOID_NGC.name}")
    return voids


def footprint_overlap(voids):
    """Boolean mask: voids inside KiDS-North with edge buffer."""
    buf = KIDS_EDGE_BUFFER
    mask = ((voids['ra'] >= KIDS_RA_MIN + buf) &
            (voids['ra'] <= KIDS_RA_MAX - buf) &
            (voids['dec'] >= KIDS_DEC_MIN + buf) &
            (voids['dec'] <= KIDS_DEC_MAX - buf))
    n = int(mask.sum())
    print(f"\n  Footprint overlap (KiDS-North, {buf} deg buffer):")
    print(f"    {n}/{len(mask)} voids ({100*n/len(mask):.1f}%)")
    if n > 0:
        print(f"    RA:  [{voids['ra'][mask].min():.1f}, {voids['ra'][mask].max():.1f}]")
        print(f"    DEC: [{voids['dec'][mask].min():.1f}, {voids['dec'][mask].max():.1f}]")
        print(f"    z:   [{voids['z'][mask].min():.3f}, {voids['z'][mask].max():.3f}]")
    return mask


def load_kids(subsample=None, rng=None):
    """Load KiDS-1000 gold catalogue.

    Applies: weight>0, KiDS-North cut, c-term subtraction per tomo bin.
    Does NOT divide by (1+m) -- m goes into the stacking denominator.
    Returns dict with e1, e2, weight, z_b, m_bias, ra, dec.
    """
    from astropy.io import fits
    if not KIDS_CAT_PATH.exists():
        print(f"\n  FATAL: KiDS-1000 catalogue not found.")
        print(f"  Expected: {KIDS_CAT_PATH}")
        print(f"  Download (~16 GB):")
        print(f"    wget https://kids.strw.leidenuniv.nl/DR4/data_files/"
              f"KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits")
        sys.exit(1)

    tag = f" ({subsample:.0%} subsample)" if subsample else ""
    print(f"  Loading KiDS-1000{tag}: {KIDS_CAT_PATH.name}")
    if not subsample:
        print(f"  (may take a few minutes for ~16 GB...)")

    with fits.open(KIDS_CAT_PATH, memmap=True) as hdu:
        data = hdu[1].data
        ra = data['ALPHA_J2000']
        dec = data['DELTA_J2000']
        e1 = data['e1'].astype(np.float64)
        e2 = data['e2'].astype(np.float64)
        w = data['weight'].astype(np.float64)
        z_b = data['Z_B'].astype(np.float64)

        n_raw = len(ra)
        print(f"  Raw: {n_raw:,}")

        mask = ((w > 0) &
                (ra >= KIDS_RA_MIN) & (ra <= KIDS_RA_MAX) &
                (dec >= KIDS_DEC_MIN) & (dec <= KIDS_DEC_MAX))
        print(f"  After weight>0 + KiDS-North: {int(mask.sum()):,}")

        if subsample and 0 < subsample < 1:
            idx_all = np.where(mask)[0]
            n_keep = max(1, int(len(idx_all) * subsample))
            idx_keep = rng.choice(idx_all, n_keep, replace=False) if rng else idx_all[:n_keep]
            mask = np.zeros(n_raw, dtype=bool)
            mask[idx_keep] = True
            print(f"  After {subsample:.0%} subsample: {int(mask.sum()):,}")

        ra = ra[mask].astype(np.float64).copy()
        dec = dec[mask].astype(np.float64).copy()
        e1 = e1[mask].copy()
        e2 = e2[mask].copy()
        w = w[mask].copy()
        z_b = z_b[mask].copy()

    # Tomo bins: c-term subtraction, m-bias per source (NOT divided out)
    m_bias = np.zeros(len(ra))
    tomo = np.zeros(len(ra), dtype=np.int8)

    for i in range(5):
        bm = (z_b > KIDS_TOMO_EDGES[i]) & (z_b <= KIDS_TOMO_EDGES[i + 1])
        tomo[bm] = i + 1
        if bm.sum() > 0:
            c1 = np.average(e1[bm], weights=w[bm])
            c2 = np.average(e2[bm], weights=w[bm])
            e1[bm] -= c1
            e2[bm] -= c2
            m_bias[bm] = KIDS_M_BIAS[i]
            print(f"    Bin {i+1} (z {KIDS_TOMO_EDGES[i]:.1f}-"
                  f"{KIDS_TOMO_EDGES[i+1]:.1f}): {int(bm.sum()):,}, "
                  f"c1={c1:+.6f}, c2={c2:+.6f}, m={KIDS_M_BIAS[i]:+.4f}")

    in_tomo = tomo > 0
    n_final = int(in_tomo.sum())
    print(f"  Final: {n_final:,} sources in tomo bins 1-5")

    return dict(
        ra=ra[in_tomo], dec=dec[in_tomo],
        e1=e1[in_tomo], e2=e2[in_tomo],
        weight=w[in_tomo], z_b=z_b[in_tomo],
        m_bias=m_bias[in_tomo], n=n_final,
    )


# ── Random control centres ──────────────────────────────────────

def generate_randoms(voids_fp, n_mult, rng):
    """Random centres in KiDS-North, z drawn from void distribution."""
    n = len(voids_fp['ra']) * n_mult
    buf = KIDS_EDGE_BUFFER
    ra = rng.uniform(KIDS_RA_MIN + buf, KIDS_RA_MAX - buf, n)
    sd_lo = np.sin(np.deg2rad(KIDS_DEC_MIN + buf))
    sd_hi = np.sin(np.deg2rad(KIDS_DEC_MAX - buf))
    dec = np.rad2deg(np.arcsin(rng.uniform(sd_lo, sd_hi, n)))
    z = rng.choice(voids_fp['z'], size=n, replace=True)
    print(f"  Generated {n} random centres ({n_mult}x voids)")
    return dict(ra=ra, dec=dec, z=z)


# ── Delta gamma_t ────────────────────────────────────────────────

def bootstrap_delta_gamma_t(void_acc, rand_acc, r_edges, n_boot, rng):
    """Bootstrap Delta_gamma_t = <gamma_t>_void - <gamma_t>_random  [5-20 Mpc/h]."""
    rc = np.sqrt(r_edges[:-1] * r_edges[1:])
    use = (rc >= R_DELTA_MIN) & (rc <= R_DELTA_MAX)
    if not np.any(use):
        return None

    nv = void_acc['sum_wet'].shape[0]
    nr = rand_acc['sum_wet'].shape[0]
    deltas = np.zeros(n_boot)

    for i in range(n_boot):
        vi = rng.choice(nv, size=nv, replace=True)
        v_wet = void_acc['sum_wet'][vi].sum(0)[use]
        v_w1m = void_acc['sum_w1m'][vi].sum(0)[use]
        v_np = void_acc['n_pairs'][vi].sum(0)[use]

        ri = rng.choice(nr, size=nr, replace=True)
        r_wet = rand_acc['sum_wet'][ri].sum(0)[use]
        r_w1m = rand_acc['sum_w1m'][ri].sum(0)[use]
        r_np = rand_acc['n_pairs'][ri].sum(0)[use]

        v_gt = np.where(v_w1m > 0, v_wet / v_w1m, 0)
        r_gt = np.where(r_w1m > 0, r_wet / r_w1m, 0)

        v_mean = np.average(v_gt, weights=v_np) if v_np.sum() > 0 else 0
        r_mean = np.average(r_gt, weights=r_np) if r_np.sum() > 0 else 0
        deltas[i] = v_mean - r_mean

    return dict(
        delta_mean=float(np.mean(deltas)),
        delta_std=float(np.std(deltas)),
        ci_68=[float(np.percentile(deltas, 16)),
               float(np.percentile(deltas, 84))],
        ci_95=[float(np.percentile(deltas, 2.5)),
               float(np.percentile(deltas, 97.5))],
        n_void_centres=nv, n_random_centres=nr,
        n_boot=n_boot,
        r_range_mpc_h=[R_DELTA_MIN, R_DELTA_MAX],
    )


# ── Plots ────────────────────────────────────────────────────────

def plot_gamma_t(void_prof, rand_prof, outdir, tag=""):
    """Tangential shear: voids vs random controls."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ftag = f"_{tag}" if tag else ""

    for prof, lab, col, mk in [
        (void_prof, f"Voids (N={void_prof['n_centres']})", "C0", "o"),
        (rand_prof, f"Random (N={rand_prof['n_centres']})", "C7", "D"),
    ]:
        R = np.array(prof['R'])
        gt = np.array(prof['gamma_t'])
        err = np.array(prof['gamma_t_err'])
        ok = np.isfinite(gt) & np.isfinite(err)
        ax.errorbar(R[ok], gt[ok], yerr=err[ok],
                     fmt=f'{mk}-', color=col, capsize=3, label=lab)

    ax.axhline(0, ls=':', color='gray', alpha=0.5)
    ax.set_xlabel(r'$R_{\rm proj}$ [Mpc/$h$]')
    ax.set_ylabel(r'$\gamma_t$')
    ax.set_xscale('log')
    ax.set_title('Tangential shear: voids vs random [KiDS-1000]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = outdir / f"testB_gamma_t_profile{ftag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path.name}")
    return path


def plot_gamma_x(prof, gx_res, outdir, tag=""):
    """Cross-component with chi2/p annotation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ftag = f"_{tag}" if tag else ""

    R = np.array(prof['R'])
    gx = np.array(prof['gamma_x'])
    err = np.array(prof['gamma_x_err'])
    ok = np.isfinite(gx) & np.isfinite(err)
    ax.errorbar(R[ok], gx[ok], yerr=err[ok],
                 fmt='o-', color='C0', capsize=3, label='Voids')
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
    ax.set_title(r'Cross-component $\gamma_\times$ (should be zero) [KiDS-1000]')
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = outdir / f"testB_gamma_x_profile{ftag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path.name}")
    return path


# ── Output ───────────────────────────────────────────────────────

def write_readme(outdir):
    """Write methodology README into output directory."""
    text = """# Test B: Weak Lensing x Environment

## Tangential shear decomposition
Position angle PA computed with spherical trigonometry (North through East):
```
PA = arctan2(sin(da) cos(dec_s),
             cos(dec_0) sin(dec_s) - sin(dec_0) cos(dec_s) cos(da))
```
Tangential and cross components (standard WL convention):
```
gamma_t =  e1 cos(2 PA) - e2 sin(2 PA)     [> 0 = tangential]
gamma_x =  e1 sin(2 PA) + e2 cos(2 PA)     [= 0 if no systematics]
```
Equivalent to gamma_t = -Re(epsilon exp(-2i phi_WL)) with phi_WL = pi/2 - PA.

## Shear calibration (KiDS lensfit)
Per tomographic bin:
- Additive c-term: weighted mean e1, e2 subtracted per bin.
- Multiplicative m-bias: applied in stacking denominator, not per source.
  gamma_t = Sum(w * e_t) / Sum(w * (1 + m))

## Source-behind-lens cut
z_source > z_void + 0.1  (photometric z_B from KiDS SOM).

## Projected separation
R_proj = chi(z_void) * theta, where chi(z) is the flat-LCDM comoving
distance (Omega_m = 0.3, h = 1 absorbed into Mpc/h units) and theta
is the haversine angular separation.

## Control sample
Random centres placed uniformly in KiDS-North (uniform in RA, uniform
in sin(DEC)), with redshifts drawn from the void z distribution.
N_random = 5 x N_voids.

## Systematics gate
gamma_x chi-squared vs zero with jackknife errors.
Pipeline stops if p < 0.05 (gamma_x inconsistent with zero).

## Headline statistic: Delta_gamma_t
Delta_gamma_t = <gamma_t>_void - <gamma_t>_random in [5, 20] Mpc/h.
Errors from bootstrap resampling of both void and random centres.
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

def run_smoke_test(args, rng, voids_fp, outdir):
    """1% subsample smoke test: validates stacking + gamma_x gate."""
    print(f"\n{'='*60}")
    print(f"SMOKE TEST ({args.smoke_frac:.0%} subsample)")
    print(f"{'='*60}")

    cat = load_kids(subsample=args.smoke_frac, rng=rng)

    print(f"\n  Stacking around {len(voids_fp['ra'])} void centres (smoke)...")
    void_acc = stack_shear(cat, voids_fp, R_BIN_EDGES, label="voids")
    void_prof = compute_profile(void_acc, R_BIN_EDGES)

    # Quick random for smoke (1x voids)
    randoms = generate_randoms(voids_fp, 1, rng)
    print(f"  Stacking around {len(randoms['ra'])} random centres (smoke)...")
    rand_acc = stack_shear(cat, randoms, R_BIN_EDGES, label="randoms")
    rand_prof = compute_profile(rand_acc, R_BIN_EDGES)

    gx = gamma_x_test(void_prof)
    print(f"\n  gamma_x: chi2/dof = {gx['chi2']:.1f}/{gx['dof']}"
          f" = {gx['chi2_dof']:.2f}, p = {gx['p_value']:.3f}")

    plot_gamma_t(void_prof, rand_prof, outdir, tag="smoke")
    plot_gamma_x(void_prof, gx, outdir, tag="smoke")

    if not gx['passed']:
        print(f"\n  SMOKE FAILED: p = {gx['p_value']:.3f} < {P_THRESHOLD}")
        print(f"  Investigate sign conventions / coordinate handling.")
        return False

    print(f"\n  SMOKE PASSED: gamma_x consistent with zero (p = {gx['p_value']:.3f}).")
    return True


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 6 Test B: Weak Lensing x Environment")
    print("  KiDS-1000 + DESIVAST NGC voids")
    print("=" * 60)

    # Step 1: Voids + footprint
    voids = load_voids()
    fp_mask = footprint_overlap(voids)
    if fp_mask.sum() == 0:
        print("  FATAL: No voids in KiDS-North footprint.")
        sys.exit(1)
    voids_fp = {k: v[fp_mask] for k, v in voids.items()}
    n_voids = len(voids_fp['ra'])

    # Step 2: Smoke test
    if not args.skip_smoke:
        if not run_smoke_test(args, rng, voids_fp, outdir):
            sys.exit(1)
        rng = np.random.RandomState(args.seed)  # reset for reproducibility

    # Step 3: Full run
    print(f"\n{'='*60}")
    print("FULL RUN")
    print(f"{'='*60}")

    cat = load_kids()

    print(f"\n  Stacking around {n_voids} void centres...")
    void_acc = stack_shear(cat, voids_fp, R_BIN_EDGES, label="voids")
    void_prof = compute_profile(void_acc, R_BIN_EDGES)

    for i, r in enumerate(void_prof['R']):
        gt = void_prof['gamma_t'][i]
        err = void_prof['gamma_t_err'][i]
        np_ = void_prof['n_pairs'][i]
        if np.isfinite(gt):
            print(f"    R={r:.1f}: gamma_t={gt:+.6f} +/- {err:.6f}  ({np_:,} pairs)")

    # Gamma_x gate
    gx = gamma_x_test(void_prof)
    print(f"\n  gamma_x: chi2/dof = {gx['chi2']:.1f}/{gx['dof']}"
          f" = {gx['chi2_dof']:.2f}, p = {gx['p_value']:.3f}")

    if not gx['passed']:
        print(f"\n  FATAL: gamma_x gate FAILED (p = {gx['p_value']:.3f} < {P_THRESHOLD})")
        print(f"  Do NOT trust Delta_gamma_t. Investigate systematics.")
        summary = dict(
            test="Phase 6 Test B", date=datetime.now().strftime("%Y-%m-%d"),
            status="FAILED_GAMMA_X_GATE", gamma_x=gx,
            n_sources=cat['n'], n_voids=n_voids)
        (outdir / "phase6_testB_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n")
        plot_gamma_x(void_prof, gx, outdir)
        write_manifest(outdir)
        sys.exit(1)

    print(f"  gamma_x gate PASSED.")

    # Random controls
    randoms = generate_randoms(voids_fp, N_RANDOM_MULT, rng)
    print(f"\n  Stacking around {len(randoms['ra'])} random centres...")
    rand_acc = stack_shear(cat, randoms, R_BIN_EDGES, label="randoms")
    rand_prof = compute_profile(rand_acc, R_BIN_EDGES)

    # Plots
    plot_gamma_t(void_prof, rand_prof, outdir)
    plot_gamma_x(void_prof, gx, outdir)

    # Delta_gamma_t
    print(f"\n  Bootstrap Delta_gamma_t ({args.n_boot} resamples)...")
    boot = bootstrap_delta_gamma_t(void_acc, rand_acc, R_BIN_EDGES, args.n_boot, rng)
    print(f"    Delta_gamma_t = {boot['delta_mean']:.6f} +/- {boot['delta_std']:.6f}")
    print(f"    68% CI: [{boot['ci_68'][0]:.6f}, {boot['ci_68'][1]:.6f}]")
    print(f"    95% CI: [{boot['ci_95'][0]:.6f}, {boot['ci_95'][1]:.6f}]")

    # Summary JSON
    summary = dict(
        test="Phase 6 Test B: Weak lensing x environment",
        date=datetime.now().strftime("%Y-%m-%d"),
        survey="KiDS-1000",
        void_catalogue="DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC",
        seed=args.seed,
        n_sources=cat['n'],
        n_voids=n_voids,
        n_random_centres=len(randoms['ra']),
        z_buffer=Z_BUFFER,
        cosmology=dict(Omega_m=OMEGA_M, h=1.0, note="Mpc/h units"),
        r_bin_edges=R_BIN_EDGES.tolist(),
        void_profile=void_prof,
        random_profile=rand_prof,
        gamma_x_test=gx,
        delta_gamma_t=boot,
        tangential_shear_convention=(
            "gamma_t = e1 cos(2PA) - e2 sin(2PA); "
            "PA = celestial position angle North through East; "
            "gamma_t > 0 = tangential (overdensity); voids expect gamma_t < 0"),
        calibration="gamma_t = Sum(w*e_t) / Sum(w*(1+m)); c-term subtracted per tomo bin",
    )

    jp = outdir / "phase6_testB_summary.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp.name}")

    write_readme(outdir)
    write_manifest(outdir)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  gamma_x: chi2/dof = {gx['chi2_dof']:.2f}, p = {gx['p_value']:.3f} [PASS]")
    print(f"  Delta_gamma_t = {boot['delta_mean']:.6f} +/- {boot['delta_std']:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
