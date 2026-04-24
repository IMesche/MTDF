#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test C4: fsigma8 x Environment
==========================================
Compares MTDF and LCDM fsigma8(z) predictions against RSD measurements
and void-specific growth estimates. Assesses whether the MTDF growth
modification mu(a) is distinguishable with current data.

Key insight: mu(a) is homogeneous (function of scale factor only), so the
~1.5% difference in fsigma8 is the same in all environments. Environment-
resolved estimators (voids, density splits) are interesting comparators
but do not test an environment-dependent MTDF prediction unless MTDF
includes explicit environmental modelling.

Entry point:
  python mtdf_validation/phase6/testC4_fsigma8_environment.py
"""

import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = (PROJECT_ROOT / "validation" / "output"
                  / "phase6" / "testC4_fsigma8_environment")
PRED_PACK = (PROJECT_ROOT / "validation" / "output"
             / "prediction_pack" / "mtdf_prediction_pack.json")


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test C4: fsigma8 x environment")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prediction_pack():
    """Load prediction pack with grids and data compilations."""
    pack = json.loads(PRED_PACK.read_text())
    return pack


def interpolate_prediction(z_grid, fsig8_grid, z_target):
    """Interpolate fsigma8 at target redshift."""
    return float(np.interp(z_target, z_grid, fsig8_grid))


def compute_chi2(data_points, z_grid, fsig8_grid):
    """Compute chi2 of model prediction vs data points."""
    chi2 = 0.0
    for dp in data_points:
        z = dp['z']
        obs = dp['fsig8']
        err = dp['err']
        pred = interpolate_prediction(z_grid, fsig8_grid, z)
        chi2 += ((obs - pred) / err)**2
    return chi2


def analyze_individual_points(data_points, z_grid, fsig8_lcdm, fsig8_mtdf):
    """Per-data-point comparison: residuals, pulls, preference."""
    results = []
    for dp in data_points:
        z = dp['z']
        obs = dp['fsig8']
        err = dp['err']
        source = dp['source']

        pred_l = interpolate_prediction(z_grid, fsig8_lcdm, z)
        pred_m = interpolate_prediction(z_grid, fsig8_mtdf, z)

        resid_l = obs - pred_l
        resid_m = obs - pred_m
        pull_l = resid_l / err
        pull_m = resid_m / err

        # Which model is closer?
        prefers = "MTDF" if abs(resid_m) < abs(resid_l) else "LCDM"

        results.append({
            'z': z,
            'source': source,
            'observed': obs,
            'error': err,
            'pred_lcdm': float(pred_l),
            'pred_mtdf': float(pred_m),
            'resid_lcdm': float(resid_l),
            'resid_mtdf': float(resid_m),
            'pull_lcdm': float(pull_l),
            'pull_mtdf': float(pull_m),
            'prefers': prefers,
        })

    return results


def compute_discrimination_power(data_points, z_grid, fsig8_lcdm, fsig8_mtdf):
    """Estimate the sigma-level at which MTDF differs from LCDM for each data point."""
    results = []
    for dp in data_points:
        z = dp['z']
        err = dp['err']
        pred_l = interpolate_prediction(z_grid, fsig8_lcdm, z)
        pred_m = interpolate_prediction(z_grid, fsig8_mtdf, z)
        model_diff = abs(pred_l - pred_m)
        diff_sigma = model_diff / err if err > 0 else 0.0

        results.append({
            'z': z,
            'source': dp['source'],
            'model_diff': float(model_diff),
            'data_error': float(err),
            'diff_in_sigma': float(diff_sigma),
            'discriminating': diff_sigma > 1.0,
        })

    return results


def compute_future_requirements(z_grid, fsig8_lcdm, fsig8_mtdf):
    """Compute error requirements for 2sigma and 5sigma discrimination."""
    z_targets = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    results = []
    for z in z_targets:
        pred_l = interpolate_prediction(z_grid, fsig8_lcdm, z)
        pred_m = interpolate_prediction(z_grid, fsig8_mtdf, z)
        diff = abs(pred_l - pred_m)
        err_2sig = diff / 2.0
        err_5sig = diff / 5.0

        results.append({
            'z': z,
            'fsig8_lcdm': float(pred_l),
            'fsig8_mtdf': float(pred_m),
            'diff': float(diff),
            'diff_pct': float(diff / pred_l * 100),
            'err_needed_2sigma': float(err_2sig),
            'err_needed_5sigma': float(err_5sig),
        })

    return results


# ── Plotting ─────────────────────────────────────────────────────

def plot_fsigma8(pack, point_results, disc_power, outdir):
    """Main fsigma8 comparison plot."""
    z_fine = np.array(pack['grids']['z_fine'])
    fsig8_l = np.array(pack['grids']['fsigma8_lcdm_z'])
    fsig8_m = np.array(pack['grids']['fsigma8_mtdf_z'])

    rsd_data = pack['data_compilation']['fsigma8_rsd']
    void_data = pack['data_compilation']['fsigma8_voids']

    fig, axes = plt.subplots(3, 1, figsize=(10, 12),
                              gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # Top panel: fsigma8(z) predictions + data
    ax = axes[0]
    ax.plot(z_fine, fsig8_l, 'C0-', lw=2, label='LCDM (Phase 5)')
    ax.plot(z_fine, fsig8_m, 'C3--', lw=2, label='MTDF full (EFE+growth)')

    # RSD data
    zd = [d['z'] for d in rsd_data]
    yd = [d['fsig8'] for d in rsd_data]
    ed = [d['err'] for d in rsd_data]
    ax.errorbar(zd, yd, yerr=ed, fmt='s', color='C2', markersize=7,
                capsize=4, label='RSD measurements', zorder=5)

    # Void data
    for d in void_data:
        ax.errorbar(d['z'], d['fsig8'], yerr=d['err'], fmt='D',
                     color='C1', markersize=9, capsize=5,
                     label=d['source'], zorder=6)

    ax.set_ylabel(r'$f\sigma_8(z)$', fontsize=13)
    ax.set_title('Test C4: $f\\sigma_8(z)$ Predictions vs Data', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0.25, 0.60)

    # Middle panel: residuals (data - LCDM, data - MTDF)
    ax = axes[1]
    all_data = rsd_data + void_data

    for i, dp in enumerate(all_data):
        z = dp['z']
        pred_l = interpolate_prediction(z_fine, fsig8_l, z)
        pred_m = interpolate_prediction(z_fine, fsig8_m, z)
        err = dp['err']

        offset = 0.005
        is_void = dp in void_data
        marker = 'D' if is_void else 's'

        if i == 0:
            ax.errorbar(z - offset, dp['fsig8'] - pred_l, yerr=err, fmt=marker,
                       color='C0', markersize=6, capsize=3, label='data - LCDM')
            ax.errorbar(z + offset, dp['fsig8'] - pred_m, yerr=err, fmt=marker,
                       color='C3', markersize=6, capsize=3, label='data - MTDF')
        else:
            ax.errorbar(z - offset, dp['fsig8'] - pred_l, yerr=err, fmt=marker,
                       color='C0', markersize=6, capsize=3)
            ax.errorbar(z + offset, dp['fsig8'] - pred_m, yerr=err, fmt=marker,
                       color='C3', markersize=6, capsize=3)

    ax.axhline(0, ls='-', color='gray', lw=0.5)
    ax.set_ylabel(r'$f\sigma_8^{obs} - f\sigma_8^{model}$', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.6)

    # Bottom panel: discrimination power (model difference / data error)
    ax = axes[2]
    zp = [d['z'] for d in disc_power]
    dp_vals = [d['diff_in_sigma'] for d in disc_power]
    sources = [d['source'] for d in disc_power]
    colors_bar = ['C1' if 'void' in s.lower() else 'C2' for s in sources]

    bars = ax.bar(range(len(zp)), dp_vals, color=colors_bar, alpha=0.8,
                  edgecolor='k', lw=0.5)
    ax.set_xticks(range(len(zp)))
    ax.set_xticklabels([f'{s}\nz={z:.2f}' for s, z in zip(sources, zp)],
                        fontsize=7, rotation=30, ha='right')
    ax.set_ylabel(r'$|\Delta f\sigma_8| / \sigma_{data}$', fontsize=11)
    ax.set_title('Model Difference as Fraction of Data Error', fontsize=11)
    ax.axhline(1.0, ls=':', color='red', alpha=0.5, label='1$\\sigma$ threshold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = outdir / "testC4_fsigma8_environment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


def plot_future_requirements(future_req, outdir):
    """Plot: error requirements for future discrimination."""
    fig, ax = plt.subplots(figsize=(8, 5))

    z_vals = [r['z'] for r in future_req]
    diff_pct = [r['diff_pct'] for r in future_req]
    err_2 = [r['err_needed_2sigma'] for r in future_req]
    err_5 = [r['err_needed_5sigma'] for r in future_req]

    # Current typical errors
    current_err = {0.1: 0.055, 0.3: 0.045, 0.5: 0.038, 0.7: 0.044, 1.0: 0.045, 1.5: 0.095}

    ax.plot(z_vals, err_2, 'C3o--', lw=2, markersize=8, label='2$\\sigma$ discrimination')
    ax.plot(z_vals, err_5, 'C0s--', lw=2, markersize=8, label='5$\\sigma$ discrimination')

    # Current errors
    z_curr = sorted(current_err.keys())
    e_curr = [current_err[z] for z in z_curr]
    ax.plot(z_curr, e_curr, 'kD-', lw=1.5, markersize=7, label='Current RSD errors')

    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel(r'$\sigma(f\sigma_8)$ required', fontsize=12)
    ax.set_title(r'Error Requirements to Discriminate MTDF from $\Lambda$CDM', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 0.2)
    ax.set_xlim(0, 1.6)

    # Annotate improvement factors
    for r in future_req:
        z = r['z']
        if z in current_err:
            factor = current_err[z] / r['err_needed_2sigma']
            if factor < 50:
                ax.annotate(f'{factor:.0f}x', (z, r['err_needed_2sigma']),
                           textcoords='offset points', xytext=(10, 5),
                           fontsize=8, color='C3')

    plt.tight_layout()
    path = outdir / "testC4_future_requirements.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


# ── Output ───────────────────────────────────────────────────────

def write_readme(chi2_results, point_results, disc_power, future_req, outdir):
    lines = ["# Phase 6 Test C4: fsigma8 x Environment\n"]
    lines.append("\n## Goal\n\n")
    lines.append("Compare MTDF and LCDM fsigma8(z) predictions against RSD measurements "
                 "and void-specific growth estimates. Assess discrimination power of "
                 "current data.\n")

    lines.append("\n## Growth Modification\n\n")
    lines.append("MTDF predicts mu(a) = 1 + 0.0793 * T(a), transitioning at z_t = 0.74.\n")
    lines.append("At z=0: mu = 1.053 (5.3% Geff enhancement). This is **homogeneous** "
                 "(function of scale factor only, not environment-dependent).\n")
    lines.append("\nThe ~1.5% difference in fsigma8 is the same in all environments "
                 "in linear theory.\n")

    lines.append("\n## Chi-squared Comparison\n\n")
    lines.append("| Statistic | LCDM | MTDF | DOF |\n")
    lines.append("|-----------|------|------|-----|\n")
    lines.append(f"| RSD chi2 | {chi2_results['rsd_lcdm']:.2f} "
                 f"| {chi2_results['rsd_mtdf']:.2f} "
                 f"| {chi2_results['rsd_dof']} |\n")
    lines.append(f"| RSD + voids chi2 | {chi2_results['all_lcdm']:.2f} "
                 f"| {chi2_results['all_mtdf']:.2f} "
                 f"| {chi2_results['all_dof']} |\n")
    lines.append(f"| Delta chi2 | -- | {chi2_results['delta_chi2_all']:+.2f} | -- |\n")

    lines.append("\n## Per-Point Analysis\n\n")
    lines.append("| Source | z | Observed | LCDM pred | MTDF pred | Pull (L) | Pull (M) | Prefers |\n")
    lines.append("|--------|---|----------|-----------|-----------|----------|----------|---------|\n")
    for r in point_results:
        lines.append(f"| {r['source']} | {r['z']:.3f} "
                     f"| {r['observed']:.3f}+/-{r['error']:.3f} "
                     f"| {r['pred_lcdm']:.3f} | {r['pred_mtdf']:.3f} "
                     f"| {r['pull_lcdm']:+.2f} | {r['pull_mtdf']:+.2f} "
                     f"| {r['prefers']} |\n")

    lines.append("\n## Discrimination Power\n\n")
    lines.append("| Source | z | Model diff | Data error | Diff/error |\n")
    lines.append("|--------|---|------------|------------|------------|\n")
    for d in disc_power:
        lines.append(f"| {d['source']} | {d['z']:.3f} "
                     f"| {d['model_diff']:.4f} "
                     f"| {d['data_error']:.3f} "
                     f"| {d['diff_in_sigma']:.3f} |\n")

    max_disc = max(d['diff_in_sigma'] for d in disc_power)
    lines.append(f"\nMaximum discrimination: {max_disc:.2f} sigma (well below 1 sigma).\n")
    lines.append("**Current RSD data cannot distinguish MTDF from LCDM in fsigma8.**\n")

    lines.append("\n## Future Requirements\n\n")
    lines.append("| z | Diff (%) | Error for 2sigma | Error for 5sigma | Current error | Improvement needed |\n")
    lines.append("|---|----------|------------------|------------------|---------------|-------------------|\n")
    current_err = {0.1: 0.055, 0.3: 0.045, 0.5: 0.038, 0.7: 0.044, 1.0: 0.045, 1.5: 0.095}
    for r in future_req:
        z = r['z']
        ce = current_err.get(z, None)
        ce_str = f"{ce:.3f}" if ce else "--"
        imp = f"{ce / r['err_needed_2sigma']:.0f}x" if ce else "--"
        lines.append(f"| {z:.1f} | {r['diff_pct']:.2f}% "
                     f"| {r['err_needed_2sigma']:.4f} "
                     f"| {r['err_needed_5sigma']:.4f} "
                     f"| {ce_str} | {imp} |\n")

    lines.append("\n## Interpretation\n\n")
    lines.append("The homogeneous mu(a) modification produces a ~1.5% shift in "
                 "fsigma8(z), which is 5-10x smaller than current RSD error bars. "
                 "Discrimination requires either:\n\n")
    lines.append("1. **DESI/Euclid-era RSD** with ~0.003-0.005 precision per z-bin\n")
    lines.append("2. **Environment-resolved estimators** (density-split fsigma8, "
                 "void-galaxy cross-correlations) which may reveal nonlinear MTDF "
                 "effects beyond the linear mu(a) prediction\n")
    lines.append("3. **Combined multi-probe analysis** where the coherent 1.5% "
                 "shift across many z-bins accumulates statistical weight\n")

    lines.append("\n## Files\n\n")
    lines.append("| File | Description |\n")
    lines.append("|------|-------------|\n")
    lines.append("| `testC4_fsigma8_environment.json` | Full analysis data |\n")
    lines.append("| `testC4_fsigma8_environment.png` | Predictions vs data |\n")
    lines.append("| `testC4_future_requirements.png` | Error requirements plot |\n")
    lines.append("| `README.md` | This file |\n")
    lines.append("| `manifest.json` | SHA256 hashes |\n")

    lines.append("\n## How to Reproduce\n\n")
    lines.append("```bash\n")
    lines.append("python mtdf_validation/phase6/testC4_fsigma8_environment.py\n")
    lines.append("```\n")

    (outdir / "README.md").write_text("".join(lines))
    print(f"  README: {outdir / 'README.md'}")


def write_manifest(outdir):
    files = sorted(
        f.name for f in outdir.iterdir()
        if f.is_file() and f.name != 'manifest.json')
    hashes = {f: sha256_file(outdir / f) for f in files}
    m = {"generated": datetime.now().strftime("%Y-%m-%d"), "sha256": hashes}
    (outdir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"  Manifest: {len(hashes)} files hashed")


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 6 Test C4: fsigma8 x Environment")
    print("=" * 60)

    pack = load_prediction_pack()

    z_fine = np.array(pack['grids']['z_fine'])
    fsig8_l = np.array(pack['grids']['fsigma8_lcdm_z'])
    fsig8_m = np.array(pack['grids']['fsigma8_mtdf_z'])

    rsd_data = pack['data_compilation']['fsigma8_rsd']
    void_data = pack['data_compilation']['fsigma8_voids']
    all_data = rsd_data + void_data

    # Chi-squared
    chi2_rsd_l = compute_chi2(rsd_data, z_fine, fsig8_l)
    chi2_rsd_m = compute_chi2(rsd_data, z_fine, fsig8_m)
    chi2_all_l = compute_chi2(all_data, z_fine, fsig8_l)
    chi2_all_m = compute_chi2(all_data, z_fine, fsig8_m)

    chi2_results = {
        'rsd_lcdm': float(chi2_rsd_l),
        'rsd_mtdf': float(chi2_rsd_m),
        'rsd_dof': len(rsd_data),
        'all_lcdm': float(chi2_all_l),
        'all_mtdf': float(chi2_all_m),
        'all_dof': len(all_data),
        'delta_chi2_rsd': float(chi2_rsd_m - chi2_rsd_l),
        'delta_chi2_all': float(chi2_all_m - chi2_all_l),
    }

    print(f"\n--- Chi-squared ---")
    print(f"  RSD only:    LCDM={chi2_rsd_l:.2f}, MTDF={chi2_rsd_m:.2f} "
          f"(delta={chi2_rsd_m - chi2_rsd_l:+.3f})")
    print(f"  RSD + voids: LCDM={chi2_all_l:.2f}, MTDF={chi2_all_m:.2f} "
          f"(delta={chi2_all_m - chi2_all_l:+.3f})")

    # Per-point analysis
    point_results = analyze_individual_points(all_data, z_fine, fsig8_l, fsig8_m)

    print(f"\n--- Per-point pulls ---")
    for r in point_results:
        print(f"  {r['source']:20s} z={r['z']:.3f}: "
              f"pull_L={r['pull_lcdm']:+.2f}, pull_M={r['pull_mtdf']:+.2f} "
              f"-> {r['prefers']}")

    # Discrimination power
    disc_power = compute_discrimination_power(all_data, z_fine, fsig8_l, fsig8_m)

    print(f"\n--- Discrimination power ---")
    max_disc = 0
    for d in disc_power:
        max_disc = max(max_disc, d['diff_in_sigma'])
        print(f"  {d['source']:20s} z={d['z']:.3f}: "
              f"model diff = {d['model_diff']:.4f}, "
              f"data err = {d['data_error']:.3f}, "
              f"diff/err = {d['diff_in_sigma']:.3f}")
    print(f"  Max discrimination: {max_disc:.3f} sigma")

    # Future requirements
    future_req = compute_future_requirements(z_fine, fsig8_l, fsig8_m)

    print(f"\n--- Future requirements (2 sigma) ---")
    for r in future_req:
        print(f"  z={r['z']:.1f}: diff={r['diff_pct']:.2f}%, "
              f"need err < {r['err_needed_2sigma']:.4f}")

    # mu(a) profile
    mu_profile = pack['grids']['mu_profile']
    mu_amp = pack['growth_predictions']['mu_amplitude']
    mu_z0 = pack['growth_predictions']['mu_z0']

    # Output
    summary = {
        "test": "Phase 6 Test C4: fsigma8 x environment",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "Prediction pack (class_mtdf + ODE growth)",
        "growth_modification": {
            "mu_amplitude": mu_amp,
            "mu_z0": mu_z0,
            "mu_formula": "mu(a) = 1 + 0.0793 * T(a), T(a) = (a/a_t)^alpha / (1 + (a/a_t)^alpha)",
            "z_t": 0.74,
            "note": "Homogeneous (scale-factor only, not environment-dependent)",
        },
        "chi2": chi2_results,
        "per_point": point_results,
        "discrimination_power": disc_power,
        "future_requirements": future_req,
        "conclusion": {
            "max_discrimination_sigma": float(max_disc),
            "distinguishable": max_disc > 1.0,
            "summary": (
                f"Maximum model difference is {max_disc:.2f} sigma of data error. "
                f"Current RSD data cannot distinguish MTDF from LCDM in fsigma8. "
                f"DESI/Euclid-era precision (~0.003-0.005) needed for 2sigma discrimination."
            ),
        },
    }

    jp = outdir / "testC4_fsigma8_environment.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp}")

    plot_fsigma8(pack, point_results, disc_power, outdir)
    plot_future_requirements(future_req, outdir)
    write_readme(chi2_results, point_results, disc_power, future_req, outdir)
    write_manifest(outdir)

    print(f"\n{'=' * 60}")
    print(f"Test C4 complete. Max discrimination: {max_disc:.3f} sigma")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
