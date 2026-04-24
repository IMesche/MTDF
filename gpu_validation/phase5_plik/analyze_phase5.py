# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 5: Post-processing and analysis of full Planck plik MCMC results.

Compares LCDM vs MTDF: chi2 breakdown, parameter values, nuisance pulls,
AIC/BIC deltas.

Usage:
  source venv/bin/activate  # from repo root
  python -m mtdf_validation.phase5_plik.analyze_phase5 --minimize-only
  python -m mtdf_validation.phase5_plik.analyze_phase5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("../../mcmc_results")

# Key cosmological parameters to report
COSMO_PARAMS = ["logA", "n_s", "theta_s_100", "omega_b", "omega_cdm",
                "tau_reio", "H0", "sigma8", "Omega_m"]
MTDF_PARAMS = ["mtdf_k_f"]

# Nuisance parameters (sampled, not fixed)
NUISANCE_SAMPLED = [
    "A_planck", "calib_100T", "calib_217T",
    "A_cib_217", "xi_sz_cib", "A_sz", "ksz_norm",
    "gal545_A_100", "gal545_A_143", "gal545_A_143_217", "gal545_A_217",
    "ps_A_100_100", "ps_A_143_143", "ps_A_143_217", "ps_A_217_217",
    "galf_TE_A_100", "galf_TE_A_100_143", "galf_TE_A_100_217",
    "galf_TE_A_143", "galf_TE_A_143_217", "galf_TE_A_217",
]

# Chi2 component names from cobaya output
CHI2_COMPONENTS = [
    "chi2__planck_2018_lowl.TT_clik",
    "chi2__planck_2018_lowl.EE_clik",
    "chi2__planck_2018_highl_plik.TTTEEE",
    "chi2__planck_2018_lensing.native",
]
CHI2_SHORT = {
    "chi2__planck_2018_lowl.TT_clik": "lowl TT",
    "chi2__planck_2018_lowl.EE_clik": "lowl EE",
    "chi2__planck_2018_highl_plik.TTTEEE": "plik TTTEEE",
    "chi2__planck_2018_lensing.native": "lensing",
}


def parse_minimum_txt(filepath):
    """Parse cobaya .minimum.txt file into a parameter dict."""
    with open(filepath) as f:
        lines = f.readlines()

    # First line is header (starts with #)
    header = None
    data = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line.lstrip("#").split()
        else:
            data = line.split()

    if header is None or data is None:
        raise ValueError(f"Could not parse {filepath}")

    result = {}
    for name, val in zip(header, data):
        try:
            result[name] = float(val)
        except ValueError:
            result[name] = val
    return result


def analyze_minimize():
    """Compare best-fit chi2 from minimization runs."""
    print("=" * 80)
    print("PHASE 5: LCDM vs MTDF  —  Full Planck plik TTTEEE Best-Fit Comparison")
    print("=" * 80)

    results = {}

    for model in ["lcdm", "mtdf"]:
        prefix = RESULTS_DIR / f"{model}_minimize"
        minimum_file = Path(f"{prefix}.minimum.txt")

        if not minimum_file.exists():
            print(f"\n  WARNING: No minimize output found for {model}")
            print(f"  Expected: {minimum_file}")
            continue

        results[model] = parse_minimum_txt(minimum_file)

    if len(results) < 2:
        print("\nBoth LCDM and MTDF results needed for comparison.")
        return results

    lcdm = results["lcdm"]
    mtdf = results["mtdf"]

    # ─── Chi-squared breakdown ───────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"{'CHI-SQUARED BREAKDOWN':^80}")
    print(f"{'─' * 80}")
    print(f"  {'Component':<30s} {'LCDM':>12s} {'MTDF':>12s} {'Delta':>12s}")
    print(f"  {'─' * 66}")

    chi2_lcdm_total = 0
    chi2_mtdf_total = 0
    chi2_breakdown = {}
    for comp in CHI2_COMPONENTS:
        short = CHI2_SHORT.get(comp, comp)
        cl = lcdm.get(comp, float('nan'))
        cm = mtdf.get(comp, float('nan'))
        delta = cm - cl
        chi2_lcdm_total += cl
        chi2_mtdf_total += cm
        chi2_breakdown[short] = {"lcdm": cl, "mtdf": cm, "delta": delta}
        print(f"  {short:<30s} {cl:12.2f} {cm:12.2f} {delta:+12.2f}")

    delta_total = chi2_mtdf_total - chi2_lcdm_total
    print(f"  {'─' * 66}")
    print(f"  {'TOTAL chi2':<30s} {chi2_lcdm_total:12.2f} {chi2_mtdf_total:12.2f} {delta_total:+12.2f}")

    # Prior contribution
    prior_l = 2 * lcdm.get("minuslogprior", 0)
    prior_m = 2 * mtdf.get("minuslogprior", 0)
    print(f"  {'-2 log(prior)':<30s} {prior_l:12.2f} {prior_m:12.2f} {prior_m - prior_l:+12.2f}")

    post_l = 2 * lcdm.get("minuslogpost", 0)
    post_m = 2 * mtdf.get("minuslogpost", 0)
    print(f"  {'-2 log(posterior)':<30s} {post_l:12.2f} {post_m:12.2f} {post_m - post_l:+12.2f}")

    # ─── AIC / BIC ──────────────────────────────────────────────────────
    # LCDM: 6 cosmo + ~21 nuisance = 27 sampled params
    # MTDF: 6 cosmo + 1 k_f + ~21 nuisance = 28 sampled params
    # Data points: plik TTTEEE has 613 (TT) + 996 (TE+EE) = 1609 bins
    #              + lowl TT (29 bins) + lowl EE (~29 bins) + lensing (9 bins) ≈ 1676
    n_data = 1676  # approximate
    k_lcdm = 27
    k_mtdf = 28

    aic_lcdm = chi2_lcdm_total + 2 * k_lcdm
    aic_mtdf = chi2_mtdf_total + 2 * k_mtdf
    bic_lcdm = chi2_lcdm_total + k_lcdm * np.log(n_data)
    bic_mtdf = chi2_mtdf_total + k_mtdf * np.log(n_data)

    print(f"\n{'─' * 80}")
    print(f"{'MODEL SELECTION':^80}")
    print(f"{'─' * 80}")
    print(f"  {'Metric':<30s} {'LCDM':>12s} {'MTDF':>12s} {'Delta':>12s} {'Verdict':>14s}")
    print(f"  {'─' * 68}")
    print(f"  {'Δχ²':<30s} {'—':>12s} {'—':>12s} {delta_total:+12.2f}  {'MTDF better' if delta_total < 0 else 'LCDM better':>12s}")
    delta_aic = aic_mtdf - aic_lcdm
    delta_bic = bic_mtdf - bic_lcdm
    print(f"  {'AIC':<30s} {aic_lcdm:12.2f} {aic_mtdf:12.2f} {delta_aic:+12.2f}  {'MTDF better' if delta_aic < 0 else 'inconclusive' if abs(delta_aic) < 2 else 'LCDM better':>12s}")
    print(f"  {'BIC':<30s} {bic_lcdm:12.2f} {bic_mtdf:12.2f} {delta_bic:+12.2f}  {'MTDF better' if delta_bic < 0 else 'inconclusive' if abs(delta_bic) < 2 else 'LCDM better':>12s}")
    print(f"  (k_LCDM={k_lcdm}, k_MTDF={k_mtdf}, N_data≈{n_data})")
    print(f"  AIC: Δ<0 favours MTDF, |Δ|<2 inconclusive")
    print(f"  BIC: Δ<0 favours MTDF, |Δ|<2 inconclusive, Δ>6 strong evidence against")

    # ─── Cosmological parameters ─────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"{'COSMOLOGICAL PARAMETERS (best-fit)':^80}")
    print(f"{'─' * 80}")
    print(f"  {'Parameter':<20s} {'LCDM':>14s} {'MTDF':>14s} {'Delta':>12s}")
    print(f"  {'─' * 60}")

    for p in COSMO_PARAMS:
        vl = lcdm.get(p, float('nan'))
        vm = mtdf.get(p, float('nan'))
        d = vm - vl
        if p in ["A_s"]:
            print(f"  {p:<20s} {vl:14.4e} {vm:14.4e} {d:+12.4e}")
        elif abs(vl) < 0.01:
            print(f"  {p:<20s} {vl:14.6f} {vm:14.6f} {d:+12.6f}")
        elif abs(vl) < 10:
            print(f"  {p:<20s} {vl:14.6f} {vm:14.6f} {d:+12.6f}")
        else:
            print(f"  {p:<20s} {vl:14.4f} {vm:14.4f} {d:+12.4f}")

    # MTDF-specific
    if "mtdf_k_f" in mtdf:
        print(f"  {'mtdf_k_f':<20s} {'—':>14s} {mtdf['mtdf_k_f']:14.6f}")

    # ─── Nuisance parameters ─────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"{'NUISANCE PARAMETERS (best-fit)':^80}")
    print(f"{'─' * 80}")
    print(f"  {'Parameter':<25s} {'LCDM':>12s} {'MTDF':>12s} {'Delta':>10s} {'|Δ/σ_L|':>8s}")
    print(f"  {'─' * 67}")

    # Compute sigma from LCDM spread (rough: use difference from ref values)
    pulls = []
    for p in NUISANCE_SAMPLED:
        vl = lcdm.get(p, float('nan'))
        vm = mtdf.get(p, float('nan'))
        if np.isnan(vl) or np.isnan(vm):
            continue
        d = vm - vl
        # Use absolute value as a rough scale for reporting
        scale = max(abs(vl) * 0.01, 1e-6)  # 1% as rough sigma
        pull = abs(d) / scale if scale > 0 else 0
        pulls.append((p, vl, vm, d, pull))
        if abs(vl) > 10:
            print(f"  {p:<25s} {vl:12.3f} {vm:12.3f} {d:+10.3f}")
        else:
            print(f"  {p:<25s} {vl:12.6f} {vm:12.6f} {d:+10.6f}")

    # Report biggest nuisance shifts
    pulls.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"\n  Top 5 nuisance shifts (MTDF − LCDM):")
    for p, vl, vm, d, _ in pulls[:5]:
        pct = (d / vl * 100) if vl != 0 else 0
        print(f"    {p:<25s}  Δ = {d:+.4f}  ({pct:+.2f}%)")

    # ─── Summary JSON ────────────────────────────────────────────────────
    summary = {
        "lcdm_chi2_total": chi2_lcdm_total,
        "mtdf_chi2_total": chi2_mtdf_total,
        "delta_chi2": delta_total,
        "delta_AIC": delta_aic,
        "delta_BIC": delta_bic,
        "k_lcdm": k_lcdm,
        "k_mtdf": k_mtdf,
        "chi2_breakdown": chi2_breakdown,
        "lcdm_cosmo": {p: lcdm.get(p) for p in COSMO_PARAMS},
        "mtdf_cosmo": {p: mtdf.get(p) for p in COSMO_PARAMS + MTDF_PARAMS},
        "mtdf_k_f_bestfit": mtdf.get("mtdf_k_f"),
    }

    out_path = RESULTS_DIR / "phase5_minimize_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")

    return summary


def analyze_mcmc():
    """Full posterior analysis from MCMC chains."""
    try:
        from getdist import loadMCSamples, plots
    except ImportError:
        print("ERROR: getdist not installed. Install with: pip install getdist")
        return None

    print("\n" + "=" * 80)
    print("PHASE 5 ANALYSIS: MCMC Posterior Results")
    print("=" * 80)

    chains = {}
    summary = {}

    for model in ["lcdm", "mtdf"]:
        prefix = RESULTS_DIR / f"{model}_mcmc"
        chain_file = Path(f"{prefix}.1.txt")

        if not chain_file.exists():
            print(f"\n  WARNING: No MCMC chain found for {model}")
            print(f"  Looked for: {chain_file}")
            continue

        print(f"\n--- {model.upper()} Posterior ---")

        try:
            samples = loadMCSamples(str(prefix), no_cache=True)
            chains[model] = samples
        except Exception as e:
            print(f"  ERROR loading chains: {e}")
            continue

        print(f"  Effective samples: {samples.numrows}")

        model_summary = {"params": {}}
        params_to_show = COSMO_PARAMS + (MTDF_PARAMS if model == "mtdf" else [])

        print(f"\n  {'Parameter':20s} {'Mean':>12s} {'Std':>10s} {'68% CI':>24s}")
        print(f"  {'-' * 70}")

        for p in params_to_show:
            try:
                mean = float(samples.mean(p))
                std = float(samples.std(p))
                lower, upper = samples.twoTailLimits(p, 0.68)
                print(f"  {p:20s} {mean:12.6f} {std:10.6f}   [{lower:.6f}, {upper:.6f}]")
                model_summary["params"][p] = {
                    "mean": mean, "std": std,
                    "lower_68": float(lower), "upper_68": float(upper),
                }
            except Exception:
                pass

        # 95% CI for k_f
        if model == "mtdf":
            try:
                lower95, upper95 = samples.twoTailLimits("mtdf_k_f", 0.95)
                model_summary["k_f_95CI"] = [float(lower95), float(upper95)]
                k_f_1_in_95 = lower95 <= 1.0 <= upper95
                model_summary["k_f_1_in_95CI"] = k_f_1_in_95
                print(f"\n  k_f 95% CI: [{lower95:.4f}, {upper95:.4f}]")
                print(f"  k_f = 1 within 95% CI: {k_f_1_in_95}")
            except Exception as e:
                print(f"  Could not compute k_f 95% CI: {e}")

        summary[model] = model_summary

    # Save summary
    out_path = RESULTS_DIR / "phase5_mcmc_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_path}")

    # Generate plots
    try:
        generate_plots(chains)
    except Exception as e:
        print(f"\nPlot generation failed: {e}")

    return summary


def generate_plots(chains):
    """Generate comparison triangle plots."""
    import matplotlib
    matplotlib.use("Agg")
    from getdist import plots

    if len(chains) < 1:
        return

    g = plots.get_subplot_plotter()
    samples_list = list(chains.values())
    labels = [k.upper() for k in chains.keys()]
    params_plot = ["omega_b", "omega_cdm", "H0", "n_s", "tau_reio", "sigma8"]

    if "mtdf" in chains:
        params_plot.append("mtdf_k_f")

    g.triangle_plot(samples_list, params_plot, filled=True, legend_labels=labels)
    plot_path = RESULTS_DIR / "phase5_triangle.png"
    g.export(str(plot_path))
    print(f"\nTriangle plot saved to {plot_path}")

    if "mtdf" in chains:
        g2 = plots.get_single_plotter()
        g2.plot_1d(chains["mtdf"], "mtdf_k_f")
        kf_path = RESULTS_DIR / "phase5_kf_posterior.png"
        g2.export(str(kf_path))
        print(f"k_f posterior saved to {kf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Post-processing of Planck plik MCMC results"
    )
    parser.add_argument(
        "--minimize-only", action="store_true",
        help="Only analyze minimization results"
    )
    args = parser.parse_args()

    if args.minimize_only:
        analyze_minimize()
    else:
        analyze_minimize()
        analyze_mcmc()


if __name__ == "__main__":
    main()
