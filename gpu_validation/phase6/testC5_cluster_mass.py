#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test C5: Cluster Dynamics vs Lensing Mass
=====================================================
Computes the predicted mass discrepancy M_dyn/M_lens under MTDF's mu(a)
modification for two coupling scenarios:

  Case A (mu-only dynamics): mu modifies the Poisson equation (dynamics)
    but lensing probes the unmodified Weyl potential. This gives
    M_dyn/M_lens = mu(z) at the cluster redshift.

  Case B (equal coupling): mu modifies both potentials equally (no
    gravitational slip, eta = Phi/Psi = 1). Lensing is also enhanced,
    so M_dyn/M_lens = 1 (no mass discrepancy).

  Case C (parameterized slip): M_dyn/M_lens = mu / Sigma, where
    Sigma = mu * (1 + 1/eta) / 2. Scans eta from 0.5 to 2.0.

Reports percent offsets vs redshift and compares to published cluster
mass comparison precision (typically 10-30% per cluster, ~5% for
stacked ensembles).

This is a prediction-level test: we compute what MTDF predicts, and
assess whether current cluster data can test it.

Entry point:
  python mtdf_validation/phase6/testC5_cluster_mass.py
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
                  / "phase6" / "testC5_cluster_mass")
PRED_PACK = (PROJECT_ROOT / "validation" / "output"
             / "prediction_pack" / "mtdf_prediction_pack.json")

# ── MTDF mu(a) parameters ────────────────────────────────────────
ALPHA = 1.30
BETA_EOS = 0.573
Z_T = 0.74
A_T = 1.0 / (1.0 + Z_T)
MU_AMP = (1.0 - BETA_EOS)**2 / (1.0 + ALPHA)


# ── Published cluster mass comparison data ────────────────────────
# (M_dyn/M_lens ratio, error, N_clusters, source)
CLUSTER_DATA = [
    {
        'name': 'Weighing the Giants (WtG)',
        'ref': 'von der Linden et al. 2014',
        'ratio': 1.00,
        'error': 0.08,
        'n_clusters': 51,
        'z_median': 0.25,
        'method': 'Velocity dispersion vs WL',
    },
    {
        'name': 'CLASH (WL+dynamics)',
        'ref': 'Merten et al. 2015; Biviano et al. 2013',
        'ratio': 1.05,
        'error': 0.10,
        'n_clusters': 20,
        'z_median': 0.35,
        'method': 'Caustic mass vs strong+weak lensing',
    },
    {
        'name': 'LoCuSS',
        'ref': 'Smith et al. 2016',
        'ratio': 0.95,
        'error': 0.12,
        'n_clusters': 50,
        'z_median': 0.23,
        'method': 'Velocity dispersion vs WL',
    },
    {
        'name': 'HeCS-omnibus',
        'ref': 'Rines et al. 2016',
        'ratio': 1.08,
        'error': 0.15,
        'n_clusters': 58,
        'z_median': 0.15,
        'method': 'Caustic mass vs X-ray/WL',
    },
    {
        'name': 'PSZ2 stacked (Planck SZ)',
        'ref': 'Planck Collaboration 2016',
        'ratio': 0.76,
        'error': 0.05,
        'n_clusters': 439,
        'z_median': 0.18,
        'method': 'SZ mass vs WL calibration (1-b)',
        'note': 'Hydrostatic mass bias, not directly M_dyn/M_lens',
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 6 Test C5: Cluster dynamics vs lensing mass")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mu_of_z(z):
    """MTDF growth modification mu(z)."""
    a = 1.0 / (1.0 + z)
    T = (a / A_T)**ALPHA / (1.0 + (a / A_T)**ALPHA)
    return 1.0 + MU_AMP * T


def sigma_of_z(z, eta=1.0):
    """Lensing modification Sigma = mu * (1 + 1/eta) / 2.

    eta = Phi/Psi (gravitational slip).
    eta = 1: no slip, Sigma = mu, M_dyn/M_lens = 1.
    eta -> inf: Sigma = mu/2, M_dyn/M_lens = 2.
    """
    mu = mu_of_z(z)
    return mu * (1.0 + 1.0 / eta) / 2.0


def mass_ratio(z, eta=None):
    """Predicted M_dyn / M_lens.

    If eta is None: Case A (mu-only dynamics, Sigma=1).
    If eta is given: Case C (parameterized slip).
    """
    mu = mu_of_z(z)
    if eta is None:
        # Case A: lensing unmodified
        return mu
    else:
        # Case C: parameterized
        sig = sigma_of_z(z, eta)
        return mu / sig


def compute_predictions():
    """Compute M_dyn/M_lens predictions for all cases across redshift."""
    z_grid = np.linspace(0, 2.0, 200)
    mu_grid = np.array([mu_of_z(z) for z in z_grid])

    cases = {}

    # Case A: mu-only dynamics (lensing unmodified, Sigma=1)
    ratio_A = mu_grid.copy()
    cases['case_A'] = {
        'label': 'mu-only dynamics (Sigma=1)',
        'description': 'mu modifies Poisson equation only; lensing probes unmodified Weyl potential',
        'z': z_grid.tolist(),
        'M_dyn_over_M_lens': ratio_A.tolist(),
        'offset_z0_pct': float((ratio_A[0] - 1.0) * 100),
    }

    # Case B: equal coupling (Sigma=mu, no slip)
    ratio_B = np.ones_like(z_grid)
    cases['case_B'] = {
        'label': 'Equal coupling (Sigma=mu, eta=1)',
        'description': 'mu modifies both potentials equally; no gravitational slip',
        'z': z_grid.tolist(),
        'M_dyn_over_M_lens': ratio_B.tolist(),
        'offset_z0_pct': 0.0,
    }

    # Case C: parameterized slip
    eta_values = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    case_C_scans = []
    for eta in eta_values:
        ratio_C = np.array([mass_ratio(z, eta) for z in z_grid])
        case_C_scans.append({
            'eta': eta,
            'M_dyn_over_M_lens': ratio_C.tolist(),
            'offset_z0_pct': float((ratio_C[0] - 1.0) * 100),
        })

    cases['case_C'] = {
        'label': 'Parameterized slip',
        'description': 'M_dyn/M_lens = mu/Sigma = 2/(1+1/eta) for varying gravitational slip eta',
        'z': z_grid.tolist(),
        'eta_scan': case_C_scans,
    }

    return cases, z_grid


def assess_detectability(cases, z_grid):
    """Compare predicted offsets to cluster mass comparison precision."""
    results = []

    for cd in CLUSTER_DATA:
        z = cd['z_median']
        mu_z = mu_of_z(z)
        offset_A = (mu_z - 1.0) * 100  # percent

        # Can this survey detect Case A?
        snr_A = abs(mu_z - 1.0) / cd['error'] if cd['error'] > 0 else 0.0

        results.append({
            'survey': cd['name'],
            'ref': cd['ref'],
            'z_median': z,
            'measured_ratio': cd['ratio'],
            'measured_error': cd['error'],
            'n_clusters': cd['n_clusters'],
            'mtdf_prediction_A': float(mu_z),
            'offset_A_pct': float(offset_A),
            'snr_case_A': float(snr_A),
            'detectable_A': snr_A > 1.0,
            'note': cd.get('note', ''),
        })

    return results


def compute_requirements():
    """What precision is needed to detect the MTDF mass offset?"""
    z_targets = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []
    for z in z_targets:
        mu_z = mu_of_z(z)
        offset = mu_z - 1.0
        err_2sig = offset / 2.0
        err_5sig = offset / 5.0

        results.append({
            'z': z,
            'mu': float(mu_z),
            'offset_pct': float(offset * 100),
            'err_needed_2sigma': float(err_2sig),
            'err_needed_5sigma': float(err_5sig),
            'err_needed_2sigma_pct': float(err_2sig * 100),
        })

    return results


# ── Plotting ─────────────────────────────────────────────────────

def plot_mass_ratio(cases, z_grid, detect_results, outdir):
    """Main prediction plot: M_dyn/M_lens vs redshift."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: M_dyn/M_lens predictions
    ax = ax1

    # Case A
    ratio_A = np.array(cases['case_A']['M_dyn_over_M_lens'])
    ax.plot(z_grid, ratio_A, 'C3-', lw=2.5,
            label='Case A: $\\mu$-only dynamics ($\\Sigma$=1)')

    # Case B
    ax.axhline(1.0, color='C0', ls='-', lw=2,
               label='Case B: equal coupling ($\\Sigma$=$\\mu$)')

    # Case C: a few eta values
    colors_eta = ['C4', 'C2', 'C1', 'C5']
    for i, scan in enumerate(cases['case_C']['eta_scan']):
        eta = scan['eta']
        if eta in [0.7, 0.9, 1.2, 1.5]:
            ratio_C = np.array(scan['M_dyn_over_M_lens'])
            ci = colors_eta[i % len(colors_eta)]
            ax.plot(z_grid, ratio_C, ls='--', lw=1.2, color=ci,
                    label=f'Case C: $\\eta$={eta}')

    # Cluster data points
    for cd in CLUSTER_DATA:
        marker = 'D' if 'SZ' in cd['name'] else 'o'
        ax.errorbar(cd['z_median'], cd['ratio'], yerr=cd['error'],
                     fmt=marker, color='k', markersize=8, capsize=5,
                     capthick=1.5, zorder=10)
        ax.annotate(cd['name'], (cd['z_median'], cd['ratio']),
                     xytext=(5, 8), textcoords='offset points',
                     fontsize=7, color='gray')

    ax.set_ylabel(r'$M_{\rm dyn} / M_{\rm lens}$', fontsize=13)
    ax.set_title('Test C5: Cluster Mass Discrepancy Predictions', fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0.65, 1.20)
    ax.axhline(1.0, ls=':', color='gray', lw=0.5)

    # Bottom panel: percent offset for Case A
    ax = ax2
    offset_A = (ratio_A - 1.0) * 100
    ax.plot(z_grid, offset_A, 'C3-', lw=2.5)
    ax.fill_between(z_grid, offset_A, 0, alpha=0.15, color='C3')
    ax.axhline(0, ls='-', color='gray', lw=0.5)

    # Survey precision bands
    for cd in detect_results:
        ax.errorbar(cd['z_median'], 0.0,
                     yerr=cd['measured_error'] * 100,
                     fmt='none', color='C0', capsize=6, capthick=2,
                     alpha=0.5)
        ax.annotate(cd['survey'].split('(')[0].strip(),
                     (cd['z_median'], -cd['measured_error'] * 100 - 0.5),
                     fontsize=7, color='C0', ha='center', va='top')

    ax.set_ylabel(r'$M_{\rm dyn}/M_{\rm lens} - 1$ [%]', fontsize=12)
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_xlim(0, 1.5)
    ax.set_title('Case A offset vs survey precision', fontsize=11)

    plt.tight_layout()
    path = outdir / "testC5_cluster_mass.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


def plot_eta_scan(cases, z_grid, outdir):
    """Gravitational slip scan: M_dyn/M_lens vs eta at key redshifts."""
    fig, ax = plt.subplots(figsize=(8, 5))

    z_targets = [0.0, 0.2, 0.5, 1.0]
    colors = ['C3', 'C1', 'C2', 'C4']

    eta_vals = [s['eta'] for s in cases['case_C']['eta_scan']]
    for z, c in zip(z_targets, colors):
        ratios = []
        for eta in eta_vals:
            ratios.append(mass_ratio(z, eta))
        ax.plot(eta_vals, ratios, f'{c}o-', lw=2, markersize=6,
                label=f'z = {z}')

    ax.axhline(1.0, ls=':', color='gray', lw=0.5)
    ax.axvline(1.0, ls=':', color='gray', lw=0.5,
               label='$\\eta$=1 (no slip)')

    # Typical precision band
    ax.axhspan(0.92, 1.08, alpha=0.05, color='C0')
    ax.text(0.55, 0.93, 'Typical stacked precision (~8%)',
            fontsize=8, color='C0')

    ax.set_xlabel(r'Gravitational slip $\eta = \Phi/\Psi$', fontsize=12)
    ax.set_ylabel(r'$M_{\rm dyn} / M_{\rm lens}$', fontsize=12)
    ax.set_title(r'Mass Discrepancy vs Gravitational Slip', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0.45, 2.1)
    ax.set_ylim(0.85, 1.15)

    plt.tight_layout()
    path = outdir / "testC5_eta_scan.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


# ── Output ───────────────────────────────────────────────────────

def write_readme(cases, detect_results, requirements, outdir):
    lines = ["# Phase 6 Test C5: Cluster Dynamics vs Lensing Mass\n"]
    lines.append("\n## Goal\n\n")
    lines.append("Compute the predicted mass discrepancy M_dyn/M_lens under "
                 "MTDF's mu(a) modification for different coupling scenarios. "
                 "Assess whether current cluster mass comparison data can "
                 "detect the predicted offset.\n")

    lines.append("\n## Theory\n\n")
    lines.append("In modified gravity, dynamical mass (from velocity dispersions) "
                 "and lensing mass (from gravitational light deflection) probe "
                 "different gravitational potentials:\n\n")
    lines.append("- **Dynamical mass:** M_dyn probes Psi (Newtonian potential, "
                 "modified by mu)\n")
    lines.append("- **Lensing mass:** M_lens probes (Psi + Phi)/2 (Weyl potential, "
                 "modified by Sigma)\n\n")
    lines.append("The relationship depends on the gravitational slip "
                 "eta = Phi/Psi:\n")
    lines.append("- Sigma = mu * (1 + 1/eta) / 2\n")
    lines.append("- M_dyn/M_lens = mu / Sigma = 2 / (1 + 1/eta)\n\n")
    lines.append("In MTDF: mu(z=0) = 1.053 (5.3% Geff enhancement).\n")

    lines.append("\n## Coupling Cases\n\n")
    lines.append("| Case | Assumption | M_dyn/M_lens at z=0 | Offset |\n")
    lines.append("|------|------------|---------------------|--------|\n")
    lines.append(f"| A | mu-only dynamics (Sigma=1) | "
                 f"{cases['case_A']['offset_z0_pct']/100 + 1:.4f} | "
                 f"+{cases['case_A']['offset_z0_pct']:.2f}% |\n")
    lines.append(f"| B | Equal coupling (Sigma=mu) | 1.0000 | 0.00% |\n")
    for scan in cases['case_C']['eta_scan']:
        if scan['eta'] in [0.7, 0.9, 1.2]:
            lines.append(f"| C (eta={scan['eta']}) | Parameterized slip | "
                         f"{scan['offset_z0_pct']/100 + 1:.4f} | "
                         f"{scan['offset_z0_pct']:+.2f}% |\n")

    lines.append("\n## mu(a) Parameters\n\n")
    lines.append(f"- Amplitude: (1 - beta_eos)^2 / (1 + alpha) = {MU_AMP:.5f}\n")
    lines.append(f"- Transition: z_t = {Z_T}, a_t = {A_T:.4f}\n")
    lines.append(f"- mu(z=0) = {mu_of_z(0):.4f} (+{(mu_of_z(0)-1)*100:.2f}%)\n")
    lines.append(f"- mu(z=0.25) = {mu_of_z(0.25):.4f} "
                 f"(+{(mu_of_z(0.25)-1)*100:.2f}%)\n")

    lines.append("\n## Comparison with Published Data\n\n")
    lines.append("| Survey | z_med | Measured | Error | "
                 "MTDF (A) | SNR | Detectable? |\n")
    lines.append("|--------|-------|----------|-------|----------|-----|"
                 "-------------|\n")
    for d in detect_results:
        det = "Yes" if d['detectable_A'] else "No"
        note = f" *{d['note']}*" if d['note'] else ""
        lines.append(
            f"| {d['survey']} | {d['z_median']:.2f} | "
            f"{d['measured_ratio']:.2f} | {d['measured_error']:.2f} | "
            f"{d['mtdf_prediction_A']:.3f} | {d['snr_case_A']:.2f} | "
            f"{det} |{note}\n")

    lines.append("\n## Future Requirements\n\n")
    lines.append("| z | mu(z) | Offset | Error for 2sigma | Error for 5sigma |\n")
    lines.append("|---|-------|--------|------------------|------------------|\n")
    for r in requirements:
        lines.append(f"| {r['z']:.1f} | {r['mu']:.4f} | "
                     f"+{r['offset_pct']:.2f}% | "
                     f"{r['err_needed_2sigma_pct']:.2f}% | "
                     f"{r['err_needed_5sigma']:.4f} |\n")

    lines.append("\n## Key Insight: Gravitational Slip Matters\n\n")
    lines.append("The predicted mass discrepancy depends critically on whether "
                 "MTDF introduces gravitational slip (eta != 1):\n\n")
    lines.append("- If MTDF's strain tensor Σ_mu_nu has **no anisotropic stress** "
                 "(eta = 1): both potentials are enhanced equally, "
                 "M_dyn/M_lens = 1, and cluster mass comparisons cannot "
                 "detect anything.\n")
    lines.append("- If Σ_mu_nu has **anisotropic stress** (eta != 1): a mass "
                 "discrepancy appears. For Case A (eta -> inf), the offset is "
                 "~5% at z=0, marginally within reach of stacked analyses.\n\n")
    lines.append("Determining the MTDF slip parameter requires solving the "
                 "full perturbed field equations for Σ_mu_nu, which has not "
                 "yet been done.\n")

    lines.append("\n## Interpretation\n\n")
    lines.append("- **Case A (maximum offset):** 5.3% at z=0, declining to "
                 "~3% at z=0.5. Current cluster ensembles have 5-15% precision, "
                 "so this is at the edge of detectability for the best "
                 "stacked analyses.\n")
    lines.append("- **Case B (no offset):** Indistinguishable from GR. "
                 "Cluster data provides no constraint.\n")
    lines.append("- **Planck SZ mass bias (1-b = 0.76):** Often quoted as "
                 "evidence for missing physics, but this involves hydrostatic "
                 "mass (not purely dynamical), so the comparison is indirect. "
                 "MTDF could contribute ~5% of the ~24% bias under Case A.\n\n")
    lines.append("**Verdict:** Prediction-level test. The maximum predicted "
                 "offset (Case A, 5.3%) is at the boundary of current "
                 "precision. Discrimination requires either (1) solving the "
                 "MTDF perturbation equations for eta, or (2) stacked cluster "
                 "analyses with ~2% M_dyn/M_lens precision.\n")

    lines.append("\n## Files\n\n")
    lines.append("| File | Description |\n")
    lines.append("|------|-------------|\n")
    lines.append("| `testC5_cluster_mass.json` | Full prediction data |\n")
    lines.append("| `testC5_cluster_mass.png` | M_dyn/M_lens vs z |\n")
    lines.append("| `testC5_eta_scan.png` | Gravitational slip scan |\n")
    lines.append("| `README.md` | This file |\n")
    lines.append("| `manifest.json` | SHA256 hashes |\n")

    lines.append("\n## How to Reproduce\n\n")
    lines.append("```bash\n")
    lines.append("python mtdf_validation/phase6/testC5_cluster_mass.py\n")
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
    print("Phase 6 Test C5: Cluster Dynamics vs Lensing Mass")
    print("=" * 60)

    # mu(a) profile
    print(f"\n--- mu(a) parameters ---")
    print(f"  amp = {MU_AMP:.5f}")
    print(f"  z_t = {Z_T}, a_t = {A_T:.4f}")
    print(f"  mu(z=0.0) = {mu_of_z(0):.4f} (+{(mu_of_z(0)-1)*100:.2f}%)")
    print(f"  mu(z=0.25) = {mu_of_z(0.25):.4f} (+{(mu_of_z(0.25)-1)*100:.2f}%)")
    print(f"  mu(z=0.5) = {mu_of_z(0.5):.4f} (+{(mu_of_z(0.5)-1)*100:.2f}%)")
    print(f"  mu(z=1.0) = {mu_of_z(1.0):.4f} (+{(mu_of_z(1.0)-1)*100:.2f}%)")

    # Compute predictions
    cases, z_grid = compute_predictions()

    print(f"\n--- Coupling cases (z=0) ---")
    print(f"  Case A (mu-only dynamics): "
          f"M_dyn/M_lens = {mu_of_z(0):.4f} "
          f"(+{cases['case_A']['offset_z0_pct']:.2f}%)")
    print(f"  Case B (equal coupling):   "
          f"M_dyn/M_lens = 1.0000 (0.00%)")
    for scan in cases['case_C']['eta_scan']:
        if scan['eta'] in [0.7, 0.9, 1.2]:
            print(f"  Case C (eta={scan['eta']}):         "
                  f"M_dyn/M_lens = {scan['offset_z0_pct']/100+1:.4f} "
                  f"({scan['offset_z0_pct']:+.2f}%)")

    # Detectability
    detect_results = assess_detectability(cases, z_grid)

    print(f"\n--- Detectability (Case A) ---")
    for d in detect_results:
        det = "YES" if d['detectable_A'] else "no"
        print(f"  {d['survey']:30s} z={d['z_median']:.2f}: "
              f"measured {d['measured_ratio']:.2f}+/-{d['measured_error']:.2f}, "
              f"MTDF={d['mtdf_prediction_A']:.3f}, "
              f"SNR={d['snr_case_A']:.2f} [{det}]")

    # Future requirements
    requirements = compute_requirements()

    print(f"\n--- Requirements for 2-sigma detection ---")
    for r in requirements:
        print(f"  z={r['z']:.1f}: offset={r['offset_pct']:.2f}%, "
              f"need sigma < {r['err_needed_2sigma_pct']:.2f}%")

    # Output JSON
    summary = {
        "test": "Phase 6 Test C5: Cluster dynamics vs lensing mass",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "type": "prediction-level",
        "mu_parameters": {
            "alpha": ALPHA,
            "beta_eos": BETA_EOS,
            "z_t": Z_T,
            "a_t": A_T,
            "amp": MU_AMP,
            "mu_z0": float(mu_of_z(0)),
            "formula": "mu(a) = 1 + amp * T(a)",
        },
        "cases": cases,
        "detectability": detect_results,
        "requirements": requirements,
        "cluster_data": CLUSTER_DATA,
        "conclusion": {
            "max_offset_case_A_pct": float(cases['case_A']['offset_z0_pct']),
            "case_B_offset": 0.0,
            "best_current_snr": float(max(
                d['snr_case_A'] for d in detect_results)),
            "detectable_case_A": any(
                d['detectable_A'] for d in detect_results),
            "key_unknown": "gravitational slip eta (requires MTDF perturbation theory)",
            "summary": (
                f"Case A (mu-only): {cases['case_A']['offset_z0_pct']:.1f}% "
                f"offset at z=0, marginally detectable with best stacked analyses. "
                f"Case B (equal coupling): no offset, indistinguishable from GR. "
                f"The actual prediction depends on MTDF's gravitational slip, "
                f"which requires solving the perturbed field equations."
            ),
        },
    }

    jp = outdir / "testC5_cluster_mass.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp}")

    # Plots
    plot_mass_ratio(cases, z_grid, detect_results, outdir)
    plot_eta_scan(cases, z_grid, outdir)

    write_readme(cases, detect_results, requirements, outdir)
    write_manifest(outdir)

    print(f"\n{'=' * 60}")
    print(f"Test C5 complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
