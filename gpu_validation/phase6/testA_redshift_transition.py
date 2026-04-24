#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test A: Redshift Transition Scan
=========================================
Scans z_cut in [0.025, 0.100] to locate the sharp late-time onset
of the SN x void environment signal predicted by MTDF at z ~ 0.04.

Two independent metrics:
  M1  GLS delta-chi2 z-score  (continuous d_signed coupling; Phase 3 replication)
  M2  Spearman rho z-score    (rank-based monotonic test; independent check)

Two controls:
  C1  Shuffled environment labels  (breaks SN-void pairing)
  C2  Random z-split               (tests whether z-ordering matters)

Entry point:
  python mtdf_validation/phase6/testA_redshift_transition.py [--seed 42]
"""

import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import linalg, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "validation" / "data"
DEFAULT_OUTPUT = (PROJECT_ROOT / "validation" / "output"
                  / "phase6" / "testA_redshift_transition")

from mtdf_validation.phase3.data_loader import (
    PantheonPlusData, load_all_void_catalogs,
    sn_to_comoving, combine_ngc_sgc_voids, COSMO_SN,
)
from mtdf_validation.phase3.crossmatch_gpu import compute_environment_cpu
from mtdf_validation.phase3.gls_engine import delta_chi2_test

# ── Config ───────────────────────────────────────────────────────
FINDER = "revolver"
Z_MIN, Z_MAX = 0.02, 0.157
Z_CUTS = np.round(np.arange(0.025, 0.105, 0.005), 4)
MIN_N = 20


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test A: Redshift transition scan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--n-perm", type=int, default=200,
                   help="C1 permutations for shuffled env")
    p.add_argument("--n-rand", type=int, default=200,
                   help="C2 iterations for random z-split")
    return p.parse_args()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── Data loading ─────────────────────────────────────────────────

def load_data():
    print("=" * 60)
    print("Phase 6 Test A: Redshift Transition Scan")
    print("=" * 60)

    pp = PantheonPlusData(str(DATA_DIR))
    idx, cov = pp.apply_cuts(Z_MIN, Z_MAX)
    sub = pp.get_subset(idx)

    catalogs = load_all_void_catalogs(str(DATA_DIR))
    void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, FINDER)

    sn_xyz = sn_to_comoving(sub['z'], sub['ra'], sub['dec'])
    d_signed, _, in_void = compute_environment_cpu(sn_xyz, void_pos, void_r)

    residual = sub['mu'] - COSMO_SN.distmod(sub['z']).value

    print(f"  N_SN = {len(idx)}, N_void = {len(void_r)}, "
          f"N_in_void = {int(in_void.sum())}")

    return dict(
        z=sub['z'], mu=sub['mu'], residual=residual,
        host_mass=sub['host_mass'], d_signed=d_signed,
        cov=cov, n=len(idx),
    )


def precompute_cov_inv(data):
    """Cache cov_inv for each z_cut low-z subset."""
    cache = {}
    for zc in Z_CUTS:
        mask = (data['z'] >= Z_MIN) & (data['z'] < zc)
        idx = np.where(mask)[0]
        if len(idx) < MIN_N:
            continue
        try:
            cache[round(float(zc), 4)] = (
                idx, linalg.inv(data['cov'][np.ix_(idx, idx)]))
        except linalg.LinAlgError:
            pass
    print(f"  Cached cov_inv for {len(cache)}/{len(Z_CUTS)} z_cuts")
    return cache


# ── Metrics ──────────────────────────────────────────────────────

def gls_zscore(data, cache, zc, d_override=None):
    """M1: signed sqrt(delta-chi2) from GLS gamma_env fit."""
    key = round(float(zc), 4)
    if key not in cache:
        return 0.0, False, {}
    idx, cinv = cache[key]
    d = d_override[idx] if d_override is not None else data['d_signed'][idx]
    r = delta_chi2_test(
        data['mu'][idx], data['z'][idx], d,
        data['host_mass'][idx], cinv)
    zs = float(np.sign(r['gamma_env'])
               * np.sqrt(max(0.0, r['delta_chi2'])))
    return zs, True, r


def spearman_zscore(data, zc, d_override=None, idx_override=None):
    """M2: Spearman rho significance between d_signed and residual."""
    if idx_override is not None:
        idx = idx_override
    else:
        mask = (data['z'] >= Z_MIN) & (data['z'] < zc)
        idx = np.where(mask)[0]
    if len(idx) < MIN_N:
        return 0.0, False, {}
    d = d_override[idx] if d_override is not None else data['d_signed'][idx]
    rho, p = stats.spearmanr(d, data['residual'][idx])
    if np.isnan(rho):
        return 0.0, False, {}
    zs = (float(np.sign(rho) * stats.norm.ppf(1 - p / 2))
          if p < 1.0 else 0.0)
    return zs, True, dict(rho=float(rho), p_value=float(p), n=len(idx))


# ── Analysis steps ───────────────────────────────────────────────

def baseline_check(data):
    """Reproduce Phase 3 full-sample and z-binned results."""
    print("\n--- Baseline check (Phase 3 replication) ---")
    cinv = linalg.inv(data['cov'])
    full = delta_chi2_test(
        data['mu'], data['z'], data['d_signed'],
        data['host_mass'], cinv)
    print(f"  Full: gamma = {full['gamma_env']:.5f} +/- "
          f"{full['gamma_env_err']:.5f}, "
          f"dchi2 = {full['delta_chi2']:.2f}, p = {full['p_value']:.4f}")

    zbins = [(0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.157)]
    binned = []
    for zlo, zhi in zbins:
        mask = (data['z'] >= zlo) & (data['z'] < zhi)
        idx = np.where(mask)[0]
        n = len(idx)
        if n < MIN_N:
            binned.append(dict(z_range=[zlo, zhi], n=n, valid=False))
            continue
        ci = linalg.inv(data['cov'][np.ix_(idx, idx)])
        r = delta_chi2_test(
            data['mu'][idx], data['z'][idx],
            data['d_signed'][idx], data['host_mass'][idx], ci)
        r.update(z_range=[zlo, zhi], valid=True)
        binned.append(r)
        print(f"  [{zlo},{zhi}): gamma={r['gamma_env']:.5f}, "
              f"dchi2={r['delta_chi2']:.2f}, p={r['p_value']:.4f}, "
              f"N={r['n']}")
    return dict(full_sample=full, z_binned=binned)


def zscan(data, cache):
    """Scan z_cut with both metrics on z < z_cut subsample."""
    print("\n--- Z-cut scan ---")
    gls_zs, spm_zs = [], []
    gls_detail, spm_detail = [], []

    for zc in Z_CUTS:
        g, gv, gd = gls_zscore(data, cache, zc)
        s, sv, sd = spearman_zscore(data, zc)
        gls_zs.append(g if gv else np.nan)
        spm_zs.append(s if sv else np.nan)
        gls_detail.append(dict(z_cut=float(zc), valid=gv, z_score=g, **gd))
        spm_detail.append(dict(z_cut=float(zc), valid=sv, z_score=s, **sd))
        n = gd.get('n', 0) if gv else 0
        print(f"  z_cut={zc:.3f}: N={n:3d}, "
              f"GLS={g:+.2f}s, Spearman={s:+.2f}s")

    return (np.array(gls_zs), np.array(spm_zs),
            dict(gls=gls_detail, spearman=spm_detail))


def run_controls(data, cache, rng, n_perm, n_rand):
    """C1: shuffled d_signed.  C2: random z-split (Spearman only)."""
    nz = len(Z_CUTS)

    # ── C1: shuffled environment ─────────────────────────────────
    print(f"\n--- C1: Shuffled environment ({n_perm} perms) ---")
    c1g = np.zeros((n_perm, nz))
    c1s = np.zeros((n_perm, nz))
    for ip in range(n_perm):
        ds = rng.permutation(data['d_signed'])
        for iz, zc in enumerate(Z_CUTS):
            c1g[ip, iz], _, _ = gls_zscore(data, cache, zc, d_override=ds)
            c1s[ip, iz], _, _ = spearman_zscore(data, zc, d_override=ds)
        if (ip + 1) % 50 == 0:
            print(f"  C1: {ip + 1}/{n_perm}")

    # ── C2: random z-split (Spearman only) ───────────────────────
    print(f"\n--- C2: Random z-split ({n_rand} iters, Spearman) ---")
    c2s = np.zeros((n_rand, nz))
    ntot = data['n']
    for ir in range(n_rand):
        for iz, zc in enumerate(Z_CUTS):
            nlow = int(np.sum((data['z'] >= Z_MIN) & (data['z'] < zc)))
            if nlow < MIN_N:
                continue
            ridx = rng.choice(ntot, size=nlow, replace=False)
            c2s[ir, iz], _, _ = spearman_zscore(
                data, zc, idx_override=ridx)
        if (ir + 1) % 50 == 0:
            print(f"  C2: {ir + 1}/{n_rand}")

    return dict(c1_gls=c1g, c1_spearman=c1s, c2_spearman=c2s)


# ── Output ───────────────────────────────────────────────────────

def make_plot(z_cuts, gls_zs, spm_zs, ctrl, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    zc = z_cuts

    # Control bands (95%)
    c1g_lo = np.percentile(ctrl['c1_gls'], 2.5, axis=0)
    c1g_hi = np.percentile(ctrl['c1_gls'], 97.5, axis=0)
    c1s_lo = np.percentile(ctrl['c1_spearman'], 2.5, axis=0)
    c1s_hi = np.percentile(ctrl['c1_spearman'], 97.5, axis=0)

    ax.fill_between(zc, c1g_lo, c1g_hi, alpha=0.15, color='C0',
                    label='C1 shuffled 95% (GLS)')
    ax.fill_between(zc, c1s_lo, c1s_hi, alpha=0.15, color='C3',
                    label='C1 shuffled 95% (Spearman)')

    # Signal lines
    ax.plot(zc, gls_zs, 'o-', color='C0', lw=2, ms=5,
            label='M1: GLS z-score')
    ax.plot(zc, spm_zs, 's-', color='C3', lw=2, ms=5,
            label='M2: Spearman z-score')

    # Reference
    ax.axhline(2, ls=':', color='gray', alpha=0.5)
    ax.axhline(3, ls=':', color='gray', alpha=0.5)
    ax.axhline(0, ls='-', color='gray', alpha=0.3, lw=0.5)
    ax.axvline(0.04, ls='--', color='green', alpha=0.6,
               label='z = 0.04 (MTDF prediction)')

    ax.set_xlabel(r'$z_{\rm cut}$ (include $z < z_{\rm cut}$)',
                  fontsize=12)
    ax.set_ylabel(r'Detection significance ($\sigma$)', fontsize=12)
    ax.set_title('Phase 6 Test A: Redshift Transition Scan', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax.set_xlim(zc[0] - 0.002, zc[-1] + 0.002)

    ax.text(zc[-1] + 0.001, 2.05, r'2$\sigma$', fontsize=8, color='gray')
    ax.text(zc[-1] + 0.001, 3.05, r'3$\sigma$', fontsize=8, color='gray')

    plt.tight_layout()
    path = outdir / "testA_zscan_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot: {path}")


def build_summary(baseline, scan_detail, gls_zs, spm_zs, ctrl, args):
    gls_peak = int(np.nanargmax(gls_zs))
    spm_peak = int(np.nanargmax(spm_zs))

    # Global p-value over the scan: for each permutation, take the
    # maximum GLS z-score across ALL z_cut values, then compare the
    # observed scan-maximum to that distribution.  This corrects for
    # multiple testing across the z_cut grid.
    observed_max_gls = float(np.nanmax(gls_zs))
    perm_max_gls = np.nanmax(ctrl['c1_gls'], axis=1)  # (n_perm,)
    n_perm = len(perm_max_gls)
    n_exceed = int(np.sum(perm_max_gls >= observed_max_gls))
    global_p = float(n_exceed / n_perm) if n_perm > 0 else 1.0
    # Report floor when no permutation exceeds the observed max
    global_p_str = (f"< {1.0 / n_perm:.4f}"
                    if n_exceed == 0 else f"{global_p:.4f}")

    return {
        "test": "Phase 6 Test A: Redshift transition scan",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "seed": args.seed,
        "config": {
            "finder": FINDER,
            "z_range": [Z_MIN, Z_MAX],
            "z_cuts": [float(z) for z in Z_CUTS],
            "min_n": MIN_N,
            "n_perm_c1": args.n_perm,
            "n_rand_c2": args.n_rand,
        },
        "baseline": baseline,
        "z_scan": scan_detail,
        "controls_summary": {
            "c1_gls_95pct": {
                "lo": [float(x) for x in
                       np.percentile(ctrl['c1_gls'], 2.5, axis=0)],
                "hi": [float(x) for x in
                       np.percentile(ctrl['c1_gls'], 97.5, axis=0)],
            },
            "c1_spearman_95pct": {
                "lo": [float(x) for x in
                       np.percentile(ctrl['c1_spearman'], 2.5, axis=0)],
                "hi": [float(x) for x in
                       np.percentile(ctrl['c1_spearman'], 97.5, axis=0)],
            },
            "c2_spearman_95pct": {
                "lo": [float(x) for x in
                       np.percentile(ctrl['c2_spearman'], 2.5, axis=0)],
                "hi": [float(x) for x in
                       np.percentile(ctrl['c2_spearman'], 97.5, axis=0)],
            },
        },
        "conclusion": {
            "peak_z_cut_gls": float(Z_CUTS[gls_peak]),
            "peak_z_score_gls": float(gls_zs[gls_peak]),
            "peak_z_cut_spearman": float(Z_CUTS[spm_peak]),
            "peak_z_score_spearman": float(spm_zs[spm_peak]),
            "global_p_over_scan": global_p,
            "global_p_str": global_p_str,
            "n_perm_exceed": n_exceed,
            "n_perm_total": n_perm,
            "transition_confirmed": bool(
                gls_zs[gls_peak] > 2.0
                and Z_CUTS[gls_peak] <= 0.055),
        },
    }


def write_readme(summary, outdir):
    s = summary
    c = s['conclusion']
    bl = s['baseline']['full_sample']

    readme = f"""# Phase 6 Test A: Redshift Transition Scan

## Goal

Test for a sharp late-time onset of the SN x void environment signal
around z ~ 0.04, as predicted by MTDF.  Extends the Phase 3 finding
("signal confined to z < 0.04") with an independent metric and formal
controls.

## Metrics

**M1 -- GLS delta-chi2 z-score (Phase 3 replication):**
Fit `mu_resid = intercept + gamma_env * d_signed + gamma_M * step(M>=10)`
via generalised least squares with full Pantheon+ covariance.
z-score = sign(gamma_env) * sqrt(delta-chi2).
Tests linear coupling of Hubble residuals to signed void proximity.

**M2 -- Spearman rho z-score (alternative):**
Spearman rank correlation between d_signed and Hubble residuals.
Non-parametric; tests monotonic (not necessarily linear) association.
Independent of covariance matrix assumptions.

## Null Hypothesis

H0: SN Ia Hubble residuals are independent of void proximity at all
redshifts.  Under H0, neither metric should show a z-dependent signal,
and the z-cut scan should be flat within the control bands.

## Controls

**C1 -- Shuffled environment labels ({s['config']['n_perm_c1']} permutations):**
Randomly permute d_signed among all SNe, preserving z-structure.
Tests whether the specific SN-void pairing carries information.
**Global p over the scan:** for each permutation, the maximum GLS
z-score across *all* z_cut values is recorded. The global p is the
fraction of permutation maxima that equal or exceed the observed scan
maximum ({c['peak_z_score_gls']:.2f} sigma at z_cut = {c['peak_z_cut_gls']:.3f}).
This corrects for the multiple-testing inherent in scanning {len(s['config']['z_cuts'])}
z_cut thresholds.

**C2 -- Random z-split ({s['config']['n_rand_c2']} iterations, Spearman only):**
For each z_cut, randomly select N_low SNe (matching the real count)
regardless of redshift.  Tests whether the z < z_cut subset is special.

## Data

- Pantheon+ SH0ES (Brout et al. 2022): {bl['n']} SNe after cuts
- DESIVAST BGS REVOLVER voids (Douglass et al. 2023)
- z range: [{s['config']['z_range'][0]}, {s['config']['z_range'][1]}]
- Seed: {s['seed']} (deterministic)

## Baseline Check (Phase 3 Replication)

Full sample: gamma_env = {bl['gamma_env']:.5f} +/- {bl['gamma_env_err']:.5f},
delta-chi2 = {bl['delta_chi2']:.2f}, p = {bl['p_value']:.4f}

## Result

| Metric | Peak z_cut | Peak sigma | Global p (scan-corrected) |
|--------|-----------|-----------|--------------------------|
| GLS (M1) | {c['peak_z_cut_gls']:.3f} | {c['peak_z_score_gls']:.2f} | {c['global_p_str']} |
| Spearman (M2) | {c['peak_z_cut_spearman']:.3f} | {c['peak_z_score_spearman']:.2f} | -- |

Global p method: {c['n_perm_exceed']}/{c['n_perm_total']} permutation
scan-maxima exceeded the observed scan-maximum.

Transition confirmed: {'Yes' if c['transition_confirmed'] else 'No'}

## Files

| File | Description |
|------|-------------|
| `phase6_testA_summary.json` | Full results: baseline, scan, controls, conclusion |
| `testA_zscan_plot.png` | Detection significance vs z_cut with control bands |
| `README.md` | This file |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testA_redshift_transition.py --seed {s['seed']}
```
"""
    (outdir / "README.md").write_text(readme)
    print(f"  README: {outdir / 'README.md'}")


def write_manifest(outdir):
    files = sorted(
        f.name for f in outdir.iterdir()
        if f.is_file() and f.name != 'manifest.json')
    hashes = {f: sha256_file(outdir / f) for f in files}
    m = {"generated": datetime.now().strftime("%Y-%m-%d"), "sha256": hashes}
    (outdir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"  Manifest: {len(hashes)} files hashed")


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    cache = precompute_cov_inv(data)

    baseline = baseline_check(data)
    gls_zs, spm_zs, scan_detail = zscan(data, cache)
    ctrl = run_controls(data, cache, rng, args.n_perm, args.n_rand)

    summary = build_summary(baseline, scan_detail, gls_zs, spm_zs,
                            ctrl, args)

    # Write outputs
    jp = outdir / "phase6_testA_summary.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp}")

    make_plot(Z_CUTS, gls_zs, spm_zs, ctrl, outdir)
    write_readme(summary, outdir)
    write_manifest(outdir)

    c = summary['conclusion']
    print(f"\n{'=' * 60}")
    print(f"RESULT: Peak at z_cut = {c['peak_z_cut_gls']:.3f} "
          f"({c['peak_z_score_gls']:.2f}s GLS)")
    print(f"Transition confirmed: {c['transition_confirmed']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
