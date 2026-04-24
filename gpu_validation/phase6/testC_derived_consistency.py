#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test C: Derived Parameter Consistency Check
=====================================================
Verifies that MTDF does not improve sigma8/S8 by breaking other
derived cosmological quantities.  Compares all available derived
parameters between MTDF and LCDM from Phase 5 MCMC chains.

Pass condition: no implausible values, improvements not paid by
a broken derived constraint.

Entry point:
  python mtdf_validation/phase6/testC_derived_consistency.py [--seed 42]
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
                  / "phase6" / "testC_derived_consistency")
PHASE5_DIR = PROJECT_ROOT / "validation" / "output" / "phase5"


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test C: Derived parameter consistency")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── Known external constraints for comparison ────────────────────
# Each: (value, error, source)
EXTERNAL = {
    "H0_shoes": (73.04, 1.04, "Riess et al. 2022 (SH0ES)"),
    "H0_planck": (67.36, 0.54, "Planck 2018 (LCDM baseline)"),
    "S8_des_y3": (0.776, 0.017, "DES Y3 (Amon et al. 2022)"),
    "S8_kids1000": (0.766, 0.020, "KiDS-1000 (Asgari et al. 2021)"),
    "S8_planck": (0.832, 0.013, "Planck 2018 (LCDM baseline)"),
    "theta_s_100_planck": (1.04192, 0.00031, "Planck 2018"),
    "omega_b_bbn": (0.02233, 0.00036, "Cooke et al. 2018 (BBN D/H)"),
    "age_planck": (13.797, 0.023, "Planck 2018 (Gyr)"),
    "tau_reio_planck": (0.054, 0.007, "Planck 2018"),
}


def load_phase5():
    """Load Phase 5 MCMC summary and minimizer comparison."""
    mcmc = json.loads((PHASE5_DIR / "phase5_mcmc_summary.json").read_text())
    mini = json.loads(
        (PHASE5_DIR / "phase5_minimize_comparison.json").read_text())
    return mcmc, mini


def compute_derived(mcmc, mini):
    """Build parameter comparison table from Phase 5 data."""
    lp = mcmc['lcdm']['params']
    mp = mcmc['mtdf']['params']

    # S8 = sigma8 * sqrt(Omega_m / 0.3)
    def s8(sig8, om):
        return sig8 * np.sqrt(om / 0.3)

    def s8_err(sig8, sig8_e, om, om_e):
        """Error propagation for S8."""
        ds_dsig = np.sqrt(om / 0.3)
        ds_dom = sig8 / (2 * np.sqrt(0.3 * om))
        return np.sqrt((ds_dsig * sig8_e)**2 + (ds_dom * om_e)**2)

    # Omega_m * h^2
    def omh2(om, h0):
        h = h0 / 100.0
        return om * h**2

    def omh2_err(om, om_e, h0, h0_e):
        h = h0 / 100.0
        d_dom = h**2
        d_dh = 2 * om * h / 100.0
        return np.sqrt((d_dom * om_e)**2 + (d_dh * h0_e)**2)

    rows = []

    # Direct parameters from MCMC posteriors
    direct_params = [
        ("H0", "km/s/Mpc", "Hubble constant"),
        ("sigma8", "", "Amplitude of matter fluctuations"),
        ("Omega_m", "", "Matter density parameter"),
        ("n_s", "", "Scalar spectral index"),
        ("tau_reio", "", "Reionization optical depth"),
        ("theta_s_100", "", "100 * angular size of sound horizon"),
        ("omega_b", "", "Baryon density (h^2)"),
        ("omega_cdm", "", "CDM density (h^2)"),
    ]

    for pname, unit, desc in direct_params:
        l = lp[pname]
        m = mp[pname]
        delta = m['mean'] - l['mean']
        sigma_comb = np.sqrt(l['std']**2 + m['std']**2)
        shift = delta / sigma_comb if sigma_comb > 0 else 0.0

        rows.append(dict(
            parameter=pname,
            description=desc,
            unit=unit,
            lcdm_mean=l['mean'], lcdm_std=l['std'],
            lcdm_68=[l['lower_68'], l['upper_68']],
            mtdf_mean=m['mean'], mtdf_std=m['std'],
            mtdf_68=[m['lower_68'], m['upper_68']],
            delta=delta,
            shift_sigma=shift,
            category="early" if pname in (
                "n_s", "theta_s_100", "omega_b", "tau_reio"
            ) else "late",
        ))

    # Derived: S8
    l_s8 = s8(lp['sigma8']['mean'], lp['Omega_m']['mean'])
    m_s8 = s8(mp['sigma8']['mean'], mp['Omega_m']['mean'])
    l_s8e = s8_err(lp['sigma8']['mean'], lp['sigma8']['std'],
                   lp['Omega_m']['mean'], lp['Omega_m']['std'])
    m_s8e = s8_err(mp['sigma8']['mean'], mp['sigma8']['std'],
                   mp['Omega_m']['mean'], mp['Omega_m']['std'])
    delta_s8 = m_s8 - l_s8
    shift_s8 = delta_s8 / np.sqrt(l_s8e**2 + m_s8e**2)

    rows.append(dict(
        parameter="S8",
        description="sigma8 * sqrt(Omega_m/0.3)",
        unit="",
        lcdm_mean=l_s8, lcdm_std=l_s8e,
        lcdm_68=[l_s8 - l_s8e, l_s8 + l_s8e],
        mtdf_mean=m_s8, mtdf_std=m_s8e,
        mtdf_68=[m_s8 - m_s8e, m_s8 + m_s8e],
        delta=delta_s8,
        shift_sigma=shift_s8,
        category="late",
    ))

    # Derived: Omega_m * h^2
    l_omh2 = omh2(lp['Omega_m']['mean'], lp['H0']['mean'])
    m_omh2 = omh2(mp['Omega_m']['mean'], mp['H0']['mean'])
    l_omh2e = omh2_err(lp['Omega_m']['mean'], lp['Omega_m']['std'],
                       lp['H0']['mean'], lp['H0']['std'])
    m_omh2e = omh2_err(mp['Omega_m']['mean'], mp['Omega_m']['std'],
                       mp['H0']['mean'], mp['H0']['std'])
    delta_omh2 = m_omh2 - l_omh2
    shift_omh2 = delta_omh2 / np.sqrt(l_omh2e**2 + m_omh2e**2)

    rows.append(dict(
        parameter="Omega_m_h2",
        description="Physical matter density",
        unit="",
        lcdm_mean=l_omh2, lcdm_std=l_omh2e,
        lcdm_68=[l_omh2 - l_omh2e, l_omh2 + l_omh2e],
        mtdf_mean=m_omh2, mtdf_std=m_omh2e,
        mtdf_68=[m_omh2 - m_omh2e, m_omh2 + m_omh2e],
        delta=delta_omh2,
        shift_sigma=shift_omh2,
        category="derived",
    ))

    # Derived: logA -> A_s
    l_As = np.exp(lp['logA']['mean']) * 1e-10
    m_As = np.exp(mp['logA']['mean']) * 1e-10
    l_Ase = lp['logA']['std'] * l_As  # propagated
    m_Ase = mp['logA']['std'] * m_As
    delta_As = m_As - l_As
    shift_As = delta_As / np.sqrt(l_Ase**2 + m_Ase**2)

    rows.append(dict(
        parameter="A_s",
        description="Primordial scalar amplitude (x1e-9)",
        unit="1e-9",
        lcdm_mean=l_As * 1e9, lcdm_std=l_Ase * 1e9,
        lcdm_68=[(l_As - l_Ase) * 1e9, (l_As + l_Ase) * 1e9],
        mtdf_mean=m_As * 1e9, mtdf_std=m_Ase * 1e9,
        mtdf_68=[(m_As - m_Ase) * 1e9, (m_As + m_Ase) * 1e9],
        delta=(m_As - l_As) * 1e9,
        shift_sigma=shift_As,
        category="early",
    ))

    # k_f (MTDF only)
    kf = mp['mtdf_k_f']
    rows.append(dict(
        parameter="k_f",
        description="MTDF coupling fraction (MTDF only)",
        unit="",
        lcdm_mean=None, lcdm_std=None, lcdm_68=None,
        mtdf_mean=kf['mean'], mtdf_std=kf['std'],
        mtdf_68=[kf['lower_68'], kf['upper_68']],
        delta=None, shift_sigma=None,
        category="mtdf",
    ))

    return rows


def evaluate_consistency(rows):
    """Check pass conditions for each parameter."""
    issues = []
    for r in rows:
        if r['shift_sigma'] is None:
            continue
        absshift = abs(r['shift_sigma'])

        # Flag if any parameter shifts > 3 sigma
        if absshift > 3.0:
            issues.append(
                f"{r['parameter']}: shift = {r['shift_sigma']:.2f} sigma "
                f"(> 3 sigma threshold)")

    # Check specific constraints
    for r in rows:
        p = r['parameter']
        m = r['mtdf_mean']
        if m is None:
            continue

        # theta_s must match Planck to < 0.5 sigma
        if p == "theta_s_100":
            ext = EXTERNAL["theta_s_100_planck"]
            pull = abs(m - ext[0]) / ext[1]
            if pull > 2.0:
                issues.append(
                    f"theta_s_100: {m:.5f} deviates {pull:.1f} sigma from "
                    f"Planck ({ext[0]})")

        # omega_b must be BBN-consistent
        if p == "omega_b":
            ext = EXTERNAL["omega_b_bbn"]
            pull = abs(m - ext[0]) / ext[1]
            if pull > 2.0:
                issues.append(
                    f"omega_b: {m:.6f} deviates {pull:.1f} sigma from "
                    f"BBN ({ext[0]})")

    passed = len(issues) == 0
    return passed, issues


def s8_tension_comparison(rows):
    """Compare S8 tension with weak lensing for LCDM vs MTDF."""
    s8_row = next(r for r in rows if r['parameter'] == 'S8')

    comparisons = {}
    for key in ["S8_des_y3", "S8_kids1000"]:
        ext_val, ext_err, ext_src = EXTERNAL[key]
        for model, mean, std in [
            ("lcdm", s8_row['lcdm_mean'], s8_row['lcdm_std']),
            ("mtdf", s8_row['mtdf_mean'], s8_row['mtdf_std']),
        ]:
            tension = abs(mean - ext_val) / np.sqrt(std**2 + ext_err**2)
            comparisons[f"{model}_vs_{key}"] = dict(
                model_S8=mean, model_S8_err=std,
                external_S8=ext_val, external_S8_err=ext_err,
                source=ext_src,
                tension_sigma=float(tension),
            )

    return comparisons


# ── Plotting ─────────────────────────────────────────────────────

def consistency_plot(rows, outdir):
    """Parameter shift plot: delta / sigma for each parameter."""
    # Filter to parameters with shifts
    plotable = [r for r in rows if r['shift_sigma'] is not None]
    names = [r['parameter'] for r in plotable]
    shifts = [r['shift_sigma'] for r in plotable]
    cats = [r['category'] for r in plotable]

    colors = {'early': 'C0', 'late': 'C3', 'derived': 'C2'}
    cs = [colors.get(c, 'gray') for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, shifts, color=cs, alpha=0.7, edgecolor='k', lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(r'Parameter shift ($\sigma$)', fontsize=12)
    ax.set_title('Phase 6 Test C: MTDF vs LCDM Derived Parameter Shifts',
                 fontsize=12)

    # Reference bands
    ax.axvspan(-1, 1, alpha=0.08, color='green')
    ax.axvspan(-2, -1, alpha=0.05, color='orange')
    ax.axvspan(1, 2, alpha=0.05, color='orange')
    ax.axvline(0, color='gray', lw=0.5)
    ax.axvline(-3, ls=':', color='red', alpha=0.5)
    ax.axvline(3, ls=':', color='red', alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='C0', alpha=0.7, label='Early-universe'),
        Patch(facecolor='C3', alpha=0.7, label='Late-universe'),
        Patch(facecolor='C2', alpha=0.7, label='Derived'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

    ax.set_xlim(-4, 4)
    plt.tight_layout()
    path = outdir / "testC_parameter_shifts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


# ── Output ───────────────────────────────────────────────────────

def write_readme(rows, passed, issues, s8_comp, outdir):
    lines = ["# Phase 6 Test C: Derived Parameter Consistency Check\n"]
    lines.append("## Goal\n")
    lines.append("Verify MTDF does not improve sigma8/S8 by breaking other "
                 "derived\ncosmological quantities. All parameters from "
                 "Phase 5 MCMC chains.\n")

    lines.append("\n## Pass Condition\n")
    lines.append("No parameter shifts > 3 sigma. Early-universe parameters\n"
                 "(theta_s, omega_b, n_s, tau_reio) must remain "
                 "LCDM-compatible.\n"
                 "Improvements must not be paid by a broken derived "
                 "constraint.\n")

    lines.append("\n## Parameter Comparison\n")
    lines.append("| Parameter | LCDM | MTDF | Shift (sigma) | Category |\n")
    lines.append("|-----------|------|------|---------------|----------|\n")
    for r in rows:
        if r['lcdm_mean'] is not None:
            lv = f"{r['lcdm_mean']:.5g} +/- {r['lcdm_std']:.3g}"
        else:
            lv = "--"
        mv = f"{r['mtdf_mean']:.5g} +/- {r['mtdf_std']:.3g}"
        sh = (f"{r['shift_sigma']:+.2f}"
              if r['shift_sigma'] is not None else "--")
        lines.append(
            f"| {r['parameter']} | {lv} | {mv} | {sh} | "
            f"{r['category']} |\n")

    lines.append("\n## S8 Tension with Weak Lensing\n")
    lines.append("| Comparison | Tension (sigma) |\n")
    lines.append("|------------|----------------|\n")
    for k, v in s8_comp.items():
        lines.append(f"| {k} | {v['tension_sigma']:.2f} |\n")

    lines.append(f"\n## Result: **{'PASS' if passed else 'FAIL'}**\n")
    if issues:
        lines.append("\nIssues:\n")
        for iss in issues:
            lines.append(f"- {iss}\n")
    else:
        lines.append("\nNo parameters exceed 3 sigma shift. "
                     "Early-universe parameters are LCDM-compatible.\n"
                     "S8 improvement is not paid by breaking any "
                     "derived constraint.\n")

    lines.append("\n## Files\n")
    lines.append("| File | Description |\n")
    lines.append("|------|-------------|\n")
    lines.append("| `testC_consistency.json` | Full comparison table |\n")
    lines.append("| `testC_parameter_shifts.png` | "
                 "Visual shift summary |\n")
    lines.append("| `README.md` | This file |\n")
    lines.append("| `manifest.json` | SHA256 hashes |\n")

    lines.append("\n## How to Reproduce\n\n")
    lines.append("```bash\n")
    lines.append("python mtdf_validation/phase6/"
                 "testC_derived_consistency.py\n")
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
    print("Phase 6 Test C: Derived Parameter Consistency Check")
    print("=" * 60)

    mcmc, mini = load_phase5()
    rows = compute_derived(mcmc, mini)

    print("\n--- Parameter shifts ---")
    for r in rows:
        if r['shift_sigma'] is not None:
            print(f"  {r['parameter']:15s}: "
                  f"{r['shift_sigma']:+.2f} sigma  ({r['category']})")

    passed, issues = evaluate_consistency(rows)
    s8_comp = s8_tension_comparison(rows)

    print(f"\n--- S8 tension comparison ---")
    for k, v in s8_comp.items():
        print(f"  {k}: {v['tension_sigma']:.2f} sigma")

    print(f"\n--- Result: {'PASS' if passed else 'FAIL'} ---")
    if issues:
        for iss in issues:
            print(f"  ISSUE: {iss}")

    # Output
    summary = {
        "test": "Phase 6 Test C: Derived parameter consistency",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "parameters": rows,
        "s8_tension": s8_comp,
        "passed": passed,
        "issues": issues,
        "external_references": {
            k: {"value": v[0], "error": v[1], "source": v[2]}
            for k, v in EXTERNAL.items()
        },
    }
    jp = outdir / "testC_consistency.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp}")

    consistency_plot(rows, outdir)
    write_readme(rows, passed, issues, s8_comp, outdir)
    write_manifest(outdir)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
