# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Runner for Test 2 (leave-one-likelihood-out) and Test 4 (prior sensitivity).

Launches short MCMC chains via cobaya for robustness testing.

Usage:
  source venv/bin/activate  # from repo root

  # Run a single config:
  python -m mtdf_validation.phase5_plik.run_test2_test4 --config test2/mtdf_no_lensing
  python -m mtdf_validation.phase5_plik.run_test2_test4 --config test2/mtdf_TT_only
  python -m mtdf_validation.phase5_plik.run_test2_test4 --config test4/mtdf_wide_prior

  # Run all:
  python -m mtdf_validation.phase5_plik.run_test2_test4 --all
"""

import argparse
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("../../mcmc_results")
VENV_PYTHON = "python"

CONFIGS = {
    "test2/mtdf_no_lensing": "Test 2a: MTDF without lensing",
    "test2/mtdf_TT_only": "Test 2b: MTDF TT-only",
    "test4/mtdf_wide_prior": "Test 4: MTDF with k_f prior [0, 10]",
}


def run_config(config_name):
    yaml_path = RESULTS_DIR / f"{config_name}.yaml"
    if not yaml_path.exists():
        print(f"ERROR: Config not found: {yaml_path}")
        return False

    print(f"\n{'=' * 70}")
    print(f"Running: {CONFIGS.get(config_name, config_name)}")
    print(f"Config:  {yaml_path}")
    print(f"{'=' * 70}\n")

    cmd = [VENV_PYTHON, "-m", "cobaya", "run", str(yaml_path), "-f"]
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(RESULTS_DIR),
            capture_output=False,
            timeout=86400,  # 24h max
        )
        if result.returncode == 0:
            print(f"\n  SUCCESS: {config_name}")
            return True
        else:
            print(f"\n  FAILED: {config_name} (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: {config_name} (24h limit)")
        return False
    except KeyboardInterrupt:
        print(f"\n  INTERRUPTED: {config_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Test 2/4 MCMC configs")
    parser.add_argument("--config", type=str, help="Config name (e.g. test2/mtdf_no_lensing)")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()

    if args.list or (not args.config and not args.all):
        print("Available configs:")
        for name, desc in CONFIGS.items():
            yaml_path = RESULTS_DIR / f"{name}.yaml"
            status = "READY" if yaml_path.exists() else "MISSING"
            # Check if output chain exists
            chain_file = RESULTS_DIR / f"{name}.1.txt"
            if chain_file.exists():
                status = "DONE"
            print(f"  {name:<30s}  [{status}]  {desc}")
        return

    if args.all:
        results = {}
        for name in CONFIGS:
            results[name] = run_config(name)
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print(f"{'=' * 70}")
        for name, ok in results.items():
            print(f"  {name:<30s}  {'OK' if ok else 'FAILED'}")
    else:
        run_config(args.config)


if __name__ == "__main__":
    main()
