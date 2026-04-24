# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 5: Full Planck plik TTTEEE + class_mtdf MCMC.

Two-stage workflow for LCDM and MTDF:
  Stage 1 (minimize): BOBYQA best-fit optimization
  Stage 2 (mcmc): Full posterior sampling with drag sampler

Usage:
  source venv/bin/activate  # from repo root

  # Quick single-point chi2 validation (run this first!)
  python -m mtdf_validation.phase5_plik.run_phase5 --validate

  # Minimization (both models, ~30 min each)
  python -m mtdf_validation.phase5_plik.run_phase5 --stage minimize --model both

  # MCMC (both models, hours to days)
  python -m mtdf_validation.phase5_plik.run_phase5 --stage mcmc --model both

  # Single model
  python -m mtdf_validation.phase5_plik.run_phase5 --stage minimize --model lcdm
  python -m mtdf_validation.phase5_plik.run_phase5 --stage mcmc --model mtdf
"""

import argparse
import faulthandler
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Enable faulthandler: dumps Python traceback on SIGSEGV/SIGABRT/SIGFPE
faulthandler.enable()


# Paths
PHASE5_DIR = Path(__file__).parent
RESULTS_DIR = Path("../../mcmc_results")
PACKAGES_PATH = "cobaya_packages"

# Planck 2018 best-fit parameters (from Planck 2018 results VI, Table 2)
PLANCK_BESTFIT = {
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "theta_s_100": 1.04092,
    "tau_reio": 0.0544,
    "logA": 3.044,
    "n_s": 0.9649,
}

# Nuisance parameter best-fit values (Planck 2018 plik TTTEEE baseline)
PLANCK_NUISANCE_BESTFIT = {
    "A_cib_217": 67.0,
    "xi_sz_cib": 0.0,
    "A_sz": 7.0,
    "ksz_norm": 0.0,
    "gal545_A_100": 7.0,
    "gal545_A_143": 9.0,
    "gal545_A_143_217": 21.0,
    "gal545_A_217": 80.0,
    "ps_A_100_100": 257.0,
    "ps_A_143_143": 47.0,
    "ps_A_143_217": 40.0,
    "ps_A_217_217": 104.0,
    "galf_TE_A_100": 0.130,
    "galf_TE_A_100_143": 0.130,
    "galf_TE_A_100_217": 0.46,
    "galf_TE_A_143": 0.207,
    "galf_TE_A_143_217": 0.69,
    "galf_TE_A_217": 1.938,
    "calib_100T": 1.0002,
    "calib_217T": 0.99805,
    "A_planck": 1.0,
}


def load_yaml_config(name):
    """Load a YAML config file from the phase5_plik directory."""
    path = PHASE5_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def validate_single_point():
    """Run single-point chi2 evaluation for LCDM and MTDF.

    This is a quick sanity check before committing to long MCMC runs.
    Expected: LCDM chi2 ~ 2779 for full plik TTTEEE.
    """
    from cobaya.model import get_model

    print("=" * 70)
    print("PHASE 5: Single-point chi2 validation")
    print("=" * 70)

    results = {}

    for model_name in ["lcdm", "mtdf"]:
        print(f"\n--- {model_name.upper()} ---")

        # Load minimize config (has all the right settings)
        info = load_yaml_config(f"{model_name}_minimize")

        # Override sampler with evaluate (single point)
        del info["sampler"]
        del info["output"]

        # Build model
        t0 = time.time()
        print("Building cobaya model...")
        model = get_model(info)

        # Set cosmological parameters at Planck best-fit
        point = dict(PLANCK_BESTFIT)
        if model_name == "mtdf":
            point["mtdf_k_f"] = 1.0
            print(f"  mtdf_k_f = 1.0 (full MTDF)")

        # Add nuisance best-fit values
        point.update(PLANCK_NUISANCE_BESTFIT)

        print(f"Evaluating at Planck best-fit...")
        logpost = model.logposterior(point)
        t1 = time.time()

        print(f"\nResults for {model_name.upper()}:")
        print(f"  Total -2 loglike  = {-2 * logpost.loglike:.2f}")
        print(f"  Total -2 logpost  = {-2 * logpost.logpost:.2f}")
        print(f"  Prior contribution = {logpost.logprior:.4f}")

        # Per-likelihood breakdown
        print(f"\n  Per-likelihood chi2 (-2 loglike):")
        loglikes = logpost.loglikes
        like_names = list(model.likelihood.keys())
        chi2_total = 0
        chi2_breakdown = {}
        for name, ll in zip(like_names, loglikes):
            chi2 = -2 * ll
            chi2_total += chi2
            chi2_breakdown[name] = float(chi2)
            print(f"    {name:45s} : {chi2:10.2f}")
        print(f"    {'TOTAL':45s} : {chi2_total:10.2f}")

        results[model_name] = {
            "chi2_total": float(chi2_total),
            "chi2_breakdown": chi2_breakdown,
            "logpost": float(logpost.logpost),
            "loglike": float(logpost.loglike),
            "logprior": float(logpost.logprior),
            "eval_time_s": t1 - t0,
            "params": {k: float(v) for k, v in point.items()
                       if isinstance(v, (int, float))},
        }

        print(f"  Evaluation time: {t1 - t0:.1f}s")

    # Compare
    if "lcdm" in results and "mtdf" in results:
        delta = results["mtdf"]["chi2_total"] - results["lcdm"]["chi2_total"]
        print(f"\n{'=' * 70}")
        print(f"COMPARISON:")
        print(f"  LCDM chi2  = {results['lcdm']['chi2_total']:.2f}")
        print(f"  MTDF chi2  = {results['mtdf']['chi2_total']:.2f}")
        print(f"  Delta chi2 = {delta:+.2f} (MTDF - LCDM)")
        print(f"{'=' * 70}")
        results["delta_chi2"] = float(delta)

    # Save results
    out_path = RESULTS_DIR / "validation_singlepoint.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def run_cobaya(config_name, resume=False):
    """Run a cobaya job from a YAML config.

    Args:
        config_name: Name of YAML file (without .yaml extension)
        resume: Whether to resume a previous run
    """
    from cobaya.run import run as cobaya_run

    info = load_yaml_config(config_name)

    print(f"\n{'=' * 70}")
    print(f"PHASE 5: Running {config_name}")
    print(f"  Output: {info['output']}")
    print(f"  Resume: {resume}")
    print(f"{'=' * 70}\n")

    t0 = time.time()

    # Run cobaya
    updated_info, sampler = cobaya_run(info, resume=resume)

    t1 = time.time()
    elapsed = t1 - t0
    hours = elapsed / 3600

    print(f"\n{'=' * 70}")
    print(f"COMPLETED: {config_name}")
    print(f"  Wall time: {elapsed:.0f}s ({hours:.2f} hours)")
    print(f"{'=' * 70}")

    # Save timing info
    timing_path = RESULTS_DIR / f"{config_name}_timing.json"
    timing = {
        "config": config_name,
        "wall_time_s": elapsed,
        "wall_time_hours": hours,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    return updated_info, sampler


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Full Planck plik TTTEEE + class_mtdf"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run single-point chi2 validation only"
    )
    parser.add_argument(
        "--stage", choices=["minimize", "mcmc"],
        help="Which stage to run"
    )
    parser.add_argument(
        "--model", choices=["lcdm", "mtdf", "both"], default="both",
        help="Which model(s) to run"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous run"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate:
        validate_single_point()
        return

    if not args.stage:
        parser.error("--stage is required unless using --validate")

    models = ["lcdm", "mtdf"] if args.model == "both" else [args.model]

    for model in models:
        config_name = f"{model}_{args.stage}"
        run_cobaya(config_name, resume=args.resume)

    print("\nAll runs complete.")
    if args.stage == "minimize":
        print("Next: run with --stage mcmc for posterior sampling")
    elif args.stage == "mcmc":
        print("Next: run analyze_phase5.py for post-processing")


if __name__ == "__main__":
    main()
