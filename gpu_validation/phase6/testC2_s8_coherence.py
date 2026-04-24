#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test C2: S8 Cross-Probe Coherence
=============================================
Compares MTDF and LCDM S8 predictions (from Phase 5 MCMC) against
published weak lensing survey values. Quantifies whether MTDF's sigma8
suppression moves S8 in the right direction and by how much.

Key output: tension metrics (sigma-level) between CMB-derived S8 and
each WL survey, for both LCDM and MTDF.

Entry point:
  python mtdf_validation/phase6/testC2_s8_coherence.py
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
                  / "phase6" / "testC2_s8_coherence")
PHASE5_DIR = PROJECT_ROOT / "validation" / "output" / "phase5"
PRED_PACK = (PROJECT_ROOT / "validation" / "output"
             / "prediction_pack" / "mtdf_prediction_pack.json")


# ── External S8 measurements ─────────────────────────────────────
# (value, stat_error, source_label, source_ref)
WL_SURVEYS = [
    (0.759, 0.024, "KiDS-1000", "Asgari et al. 2021"),
    (0.776, 0.017, "DES Y3", "Amon et al. 2022; Secco et al. 2022"),
    (0.763, 0.040, "HSC Y3", "Li et al. 2023; Dalal et al. 2023"),
]

PLANCK_S8 = (0.832, 0.013, "Planck 2018", "Planck Collaboration 2020")


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test C2: S8 cross-probe coherence")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_phase5():
    """Load Phase 5 MCMC summary for sigma8 and Omega_m posteriors."""
    mcmc = json.loads((PHASE5_DIR / "phase5_mcmc_summary.json").read_text())
    return mcmc


def compute_s8(sigma8, omega_m):
    """S8 = sigma8 * sqrt(Omega_m / 0.3)"""
    return sigma8 * np.sqrt(omega_m / 0.3)


def propagate_s8_error(sigma8, sigma8_err, omega_m, omega_m_err):
    """Gaussian error propagation for S8."""
    ds_dsig = np.sqrt(omega_m / 0.3)
    ds_dom = sigma8 / (2.0 * np.sqrt(0.3 * omega_m))
    return np.sqrt((ds_dsig * sigma8_err)**2 + (ds_dom * omega_m_err)**2)


def tension_sigma(val1, err1, val2, err2):
    """Tension in sigma between two measurements with independent errors."""
    return abs(val1 - val2) / np.sqrt(err1**2 + err2**2)


def build_comparison(mcmc):
    """Build full S8 comparison table."""
    results = {}

    for model_key in ['lcdm', 'mtdf']:
        p = mcmc[model_key]['params']
        sig8 = p['sigma8']['mean']
        sig8_e = p['sigma8']['std']
        om = p['Omega_m']['mean']
        om_e = p['Omega_m']['std']

        s8_val = compute_s8(sig8, om)
        s8_err = propagate_s8_error(sig8, sig8_e, om, om_e)

        model_results = {
            'sigma8': sig8,
            'sigma8_err': sig8_e,
            'Omega_m': om,
            'Omega_m_err': om_e,
            'S8': float(s8_val),
            'S8_err': float(s8_err),
            'tensions': {},
        }

        for wl_s8, wl_err, wl_name, wl_ref in WL_SURVEYS:
            t = tension_sigma(s8_val, s8_err, wl_s8, wl_err)
            model_results['tensions'][wl_name] = {
                'wl_S8': wl_s8,
                'wl_err': wl_err,
                'wl_ref': wl_ref,
                'tension_sigma': float(t),
            }

        # Also tension with Planck itself (should be ~0)
        t_planck = tension_sigma(s8_val, s8_err, PLANCK_S8[0], PLANCK_S8[1])
        model_results['tensions']['Planck 2018'] = {
            'wl_S8': PLANCK_S8[0],
            'wl_err': PLANCK_S8[1],
            'wl_ref': PLANCK_S8[3],
            'tension_sigma': float(t_planck),
        }

        results[model_key] = model_results

    return results


def compute_tension_reduction(results):
    """Compute how much MTDF reduces S8 tension with each WL survey."""
    reductions = {}
    lcdm = results['lcdm']
    mtdf = results['mtdf']

    for survey_name in lcdm['tensions']:
        if survey_name == 'Planck 2018':
            continue
        t_lcdm = lcdm['tensions'][survey_name]['tension_sigma']
        t_mtdf = mtdf['tensions'][survey_name]['tension_sigma']
        delta_t = t_lcdm - t_mtdf
        pct = (delta_t / t_lcdm * 100) if t_lcdm > 0 else 0.0

        reductions[survey_name] = {
            'tension_lcdm': float(t_lcdm),
            'tension_mtdf': float(t_mtdf),
            'delta_tension': float(delta_t),
            'reduction_pct': float(pct),
        }

    return reductions


def compute_combined_tension(results):
    """Compute combined S8 tension across WL surveys (inverse-variance weighted mean)."""
    combined = {}

    # Inverse-variance weighted mean of WL S8 values
    wl_vals = np.array([s[0] for s in WL_SURVEYS])
    wl_errs = np.array([s[1] for s in WL_SURVEYS])
    w = 1.0 / wl_errs**2
    wl_mean = np.sum(w * wl_vals) / np.sum(w)
    wl_mean_err = 1.0 / np.sqrt(np.sum(w))

    combined['wl_weighted_mean'] = {
        'S8': float(wl_mean),
        'S8_err': float(wl_mean_err),
        'surveys': [s[2] for s in WL_SURVEYS],
    }

    for model_key in ['lcdm', 'mtdf']:
        s8 = results[model_key]['S8']
        s8e = results[model_key]['S8_err']
        t = tension_sigma(s8, s8e, wl_mean, wl_mean_err)
        combined[f'{model_key}_vs_wl_combined'] = {
            'model_S8': float(s8),
            'model_S8_err': float(s8e),
            'tension_sigma': float(t),
        }

    return combined


# ── Plotting ─────────────────────────────────────────────────────

def plot_s8_comparison(results, reductions, combined, outdir):
    """S8 comparison: CMB predictions vs WL surveys."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [2, 1]})

    # Left panel: S8 values with error bars
    labels = []
    values = []
    errors = []
    colors = []

    # Planck LCDM baseline
    labels.append("Planck 2018\n(LCDM)")
    values.append(PLANCK_S8[0])
    errors.append(PLANCK_S8[1])
    colors.append('C4')

    # Phase 5 LCDM
    labels.append("Phase 5\nLCDM")
    values.append(results['lcdm']['S8'])
    errors.append(results['lcdm']['S8_err'])
    colors.append('C0')

    # Phase 5 MTDF
    labels.append("Phase 5\nMTDF")
    values.append(results['mtdf']['S8'])
    errors.append(results['mtdf']['S8_err'])
    colors.append('C3')

    # WL surveys
    for wl_s8, wl_err, wl_name, _ in WL_SURVEYS:
        labels.append(wl_name)
        values.append(wl_s8)
        errors.append(wl_err)
        colors.append('C2')

    # Combined WL
    labels.append("WL combined\n(inv-var)")
    values.append(combined['wl_weighted_mean']['S8'])
    errors.append(combined['wl_weighted_mean']['S8_err'])
    colors.append('C1')

    y_pos = np.arange(len(labels))
    ax1.errorbar(values, y_pos, xerr=errors, fmt='o', markersize=8,
                 capsize=5, capthick=1.5, elinewidth=1.5,
                 color='black', zorder=5)

    for i, (v, c) in enumerate(zip(values, colors)):
        ax1.plot(v, i, 'o', color=c, markersize=10, zorder=6)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel(r'$S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$', fontsize=13)
    ax1.set_title('S8 Cross-Probe Coherence (Test C2)', fontsize=13)

    # Shaded bands for LCDM and MTDF
    lcdm_s8 = results['lcdm']['S8']
    lcdm_e = results['lcdm']['S8_err']
    mtdf_s8 = results['mtdf']['S8']
    mtdf_e = results['mtdf']['S8_err']

    ax1.axvspan(lcdm_s8 - lcdm_e, lcdm_s8 + lcdm_e,
                alpha=0.08, color='C0', label='LCDM 1$\\sigma$')
    ax1.axvspan(mtdf_s8 - mtdf_e, mtdf_s8 + mtdf_e,
                alpha=0.08, color='C3', label='MTDF 1$\\sigma$')

    ax1.legend(fontsize=9, loc='lower left')
    ax1.set_xlim(0.70, 0.87)
    ax1.invert_yaxis()

    # Right panel: tension bar chart
    surveys = list(reductions.keys())
    t_lcdm = [reductions[s]['tension_lcdm'] for s in surveys]
    t_mtdf = [reductions[s]['tension_mtdf'] for s in surveys]

    # Add combined
    surveys.append('WL combined')
    t_lcdm.append(combined['lcdm_vs_wl_combined']['tension_sigma'])
    t_mtdf.append(combined['mtdf_vs_wl_combined']['tension_sigma'])

    x = np.arange(len(surveys))
    w = 0.35
    bars1 = ax2.bar(x - w/2, t_lcdm, w, label='LCDM', color='C0', alpha=0.8)
    bars2 = ax2.bar(x + w/2, t_mtdf, w, label='MTDF', color='C3', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(surveys, fontsize=9, rotation=15, ha='right')
    ax2.set_ylabel(r'Tension ($\sigma$)', fontsize=12)
    ax2.set_title('S8 Tension with WL Surveys', fontsize=12)

    # Reference lines
    ax2.axhline(1, ls=':', color='green', alpha=0.5, label='1$\\sigma$')
    ax2.axhline(2, ls=':', color='orange', alpha=0.5, label='2$\\sigma$')
    ax2.axhline(3, ls=':', color='red', alpha=0.5, label='3$\\sigma$')

    ax2.legend(fontsize=8, ncol=2)
    ax2.set_ylim(0, max(max(t_lcdm), max(t_mtdf)) * 1.3)

    # Add tension values on bars
    for bar, val in zip(bars1, t_lcdm):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.1f}$\\sigma$', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, t_mtdf):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.1f}$\\sigma$', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = outdir / "testC2_s8_coherence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


# ── Output ───────────────────────────────────────────────────────

def write_readme(results, reductions, combined, outdir):
    lines = ["# Phase 6 Test C2: S8 Cross-Probe Coherence\n"]
    lines.append("\n## Goal\n\n")
    lines.append("Compare MTDF and LCDM S8 predictions (from Phase 5 MCMC Planck "
                 "posteriors) against published weak lensing survey constraints. "
                 "Quantify whether MTDF's sigma8 suppression reduces the S8 tension.\n")

    lines.append("\n## Definition\n\n")
    lines.append("S8 = sigma8 * sqrt(Omega_m / 0.3)\n")

    lines.append("\n## Phase 5 MCMC S8 Values\n\n")
    lines.append("| Model | sigma8 | Omega_m | S8 |\n")
    lines.append("|-------|--------|---------|----|\n")
    for model in ['lcdm', 'mtdf']:
        r = results[model]
        lines.append(f"| {model.upper()} | {r['sigma8']:.4f} +/- {r['sigma8_err']:.4f} "
                     f"| {r['Omega_m']:.4f} +/- {r['Omega_m_err']:.4f} "
                     f"| {r['S8']:.4f} +/- {r['S8_err']:.4f} |\n")

    lines.append("\n## S8 Tension with Weak Lensing Surveys\n\n")
    lines.append("| Survey | S8 | LCDM tension | MTDF tension | Reduction |\n")
    lines.append("|--------|----|--------------|--------------|-----------|\n")
    for wl_s8, wl_err, wl_name, wl_ref in WL_SURVEYS:
        t_l = results['lcdm']['tensions'][wl_name]['tension_sigma']
        t_m = results['mtdf']['tensions'][wl_name]['tension_sigma']
        red = reductions[wl_name]
        lines.append(f"| {wl_name} | {wl_s8:.3f} +/- {wl_err:.3f} "
                     f"| {t_l:.2f} sigma | {t_m:.2f} sigma "
                     f"| {red['reduction_pct']:.0f}% |\n")

    # Combined
    c_wl = combined['wl_weighted_mean']
    c_l = combined['lcdm_vs_wl_combined']
    c_m = combined['mtdf_vs_wl_combined']
    lines.append(f"| **WL combined** | {c_wl['S8']:.3f} +/- {c_wl['S8_err']:.3f} "
                 f"| {c_l['tension_sigma']:.2f} sigma "
                 f"| {c_m['tension_sigma']:.2f} sigma | -- |\n")

    lines.append("\n## Interpretation\n\n")
    lines.append("MTDF's sigma8 suppression (0.790 vs 0.810) moves S8 closer to "
                 "weak lensing values.\n\n")
    lines.append("**Caveats:**\n")
    lines.append("- Tension metrics use simple Gaussian error propagation from "
                 "MCMC posteriors\n")
    lines.append("- A proper comparison requires running WL likelihoods within the "
                 "MTDF framework\n")
    lines.append("- Published WL S8 values assume LCDM; MTDF would modify the "
                 "lensing kernel\n")
    lines.append("- This is a first-order consistency check, not a full "
                 "likelihood analysis\n")

    lines.append("\n## Files\n\n")
    lines.append("| File | Description |\n")
    lines.append("|------|-------------|\n")
    lines.append("| `testC2_s8_coherence.json` | Full comparison data |\n")
    lines.append("| `testC2_s8_coherence.png` | Visual comparison |\n")
    lines.append("| `README.md` | This file |\n")
    lines.append("| `manifest.json` | SHA256 hashes |\n")

    lines.append("\n## How to Reproduce\n\n")
    lines.append("```bash\n")
    lines.append("python mtdf_validation/phase6/testC2_s8_coherence.py\n")
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
    print("Phase 6 Test C2: S8 Cross-Probe Coherence")
    print("=" * 60)

    # Load Phase 5 posteriors
    mcmc = load_phase5()

    # Build comparison
    results = build_comparison(mcmc)
    reductions = compute_tension_reduction(results)
    combined = compute_combined_tension(results)

    # Print summary
    print("\n--- S8 Values ---")
    for model in ['lcdm', 'mtdf']:
        r = results[model]
        print(f"  {model.upper()}: S8 = {r['S8']:.4f} +/- {r['S8_err']:.4f}")

    print("\n--- Tension with WL surveys ---")
    print(f"  {'Survey':15s} | {'LCDM':8s} | {'MTDF':8s} | Reduction")
    print("  " + "-" * 55)
    for wl_s8, wl_err, wl_name, _ in WL_SURVEYS:
        t_l = results['lcdm']['tensions'][wl_name]['tension_sigma']
        t_m = results['mtdf']['tensions'][wl_name]['tension_sigma']
        red = reductions[wl_name]['reduction_pct']
        print(f"  {wl_name:15s} | {t_l:5.2f} sig | {t_m:5.2f} sig | {red:+.0f}%")

    print(f"\n--- Combined WL ---")
    c_wl = combined['wl_weighted_mean']
    print(f"  WL weighted mean S8 = {c_wl['S8']:.4f} +/- {c_wl['S8_err']:.4f}")
    print(f"  LCDM vs WL: {combined['lcdm_vs_wl_combined']['tension_sigma']:.2f} sigma")
    print(f"  MTDF vs WL: {combined['mtdf_vs_wl_combined']['tension_sigma']:.2f} sigma")

    # Output
    summary = {
        "test": "Phase 6 Test C2: S8 cross-probe coherence",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "definition": "S8 = sigma8 * sqrt(Omega_m / 0.3)",
        "source": "Phase 5 MCMC posteriors (Planck PR4 TTTEEE+lensing)",
        "results": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in results.items()
        },
        "tension_reductions": reductions,
        "combined_wl": combined,
        "external_data": {
            "weak_lensing": [
                {"survey": s[2], "S8": s[0], "err": s[1], "ref": s[3]}
                for s in WL_SURVEYS
            ],
            "planck": {"S8": PLANCK_S8[0], "err": PLANCK_S8[1],
                       "ref": PLANCK_S8[3]},
        },
        "caveats": [
            "Gaussian error propagation from MCMC posteriors",
            "Published WL S8 values assume LCDM growth/lensing kernel",
            "Proper comparison requires WL likelihood within MTDF framework",
            "First-order consistency check, not full likelihood analysis",
        ],
    }

    jp = outdir / "testC2_s8_coherence.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp}")

    plot_s8_comparison(results, reductions, combined, outdir)
    write_readme(results, reductions, combined, outdir)
    write_manifest(outdir)

    print(f"\n{'=' * 60}")
    print("Test C2 complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
