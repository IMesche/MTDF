#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Background monitor for Phase 5 robustness MCMC chains.

Polls progress files every POLL_INTERVAL seconds. When a chain meets the
analysis-ready criterion, triggers interim analysis. When all three are
ready, triggers final analysis and exits.

Readiness criterion (all must hold):
  1. R-1 < threshold in TWO consecutive cobaya checks
  2. N_accepted >= MIN_ACCEPTED
  3. k_f mean moves < KF_MEAN_TOL between those two checks
  4. k_f 95% CI width moves < KF_CI_TOL between those two checks

Tracked quantities per poll (not just k_f mean):
  - k_f: mean, median, 68% CI, 95% CI
  - σ₈: mean ± std
  - H₀: mean ± std

R-1 note: All runs use Cobaya's Rminus1_single_split=4.  Cobaya splits
the single chain into 4 segments and computes Gelman-Rubin R-1 across
them.  This is the same method used by the converged baseline.

Usage:
  source venv/bin/activate  # from repo root
  python -m mtdf_validation.phase5_plik.monitor_robustness

Log: results/phase5/monitor_robustness.log
"""

import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

POLL_INTERVAL = 300  # seconds between checks (5 min)
CONVERGENCE_R1 = 0.1
MIN_ACCEPTED = 2000
KF_MEAN_TOL = 0.05   # k_f mean must not move more than this
KF_CI_TOL = 0.15     # k_f 95% CI width must not move more than this

RESULTS_DIR = Path("../../mcmc_results")
LOG_FILE = RESULTS_DIR / "monitor_robustness.log"

CHAINS = {
    "no_lensing": {
        "label": "No lensing",
        "prefix": RESULTS_DIR / "test2" / "mtdf_no_lensing",
        "test": "test2",
    },
    "TT_only": {
        "label": "TT only",
        "prefix": RESULTS_DIR / "test2" / "mtdf_TT_only",
        "test": "test2",
    },
    "wide_prior": {
        "label": "Wide prior",
        "prefix": RESULTS_DIR / "test4" / "mtdf_wide_prior",
        "test": "test4",
    },
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def parse_progress(prefix: Path) -> list[dict]:
    """Parse cobaya progress file. Columns: N, timestamp, acc_rate, R-1, R-1_cl."""
    progress_file = Path(f"{prefix}.progress")
    if not progress_file.exists():
        return []

    records = []
    for line in progress_file.read_text().strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                r1_val = float(parts[3])
                r1 = r1_val if not math.isnan(r1_val) else None
                r1_cl = None
                if len(parts) >= 5:
                    r1_cl_val = float(parts[4])
                    r1_cl = r1_cl_val if not math.isnan(r1_cl_val) else None
                records.append({
                    "N": int(float(parts[0])),
                    "timestamp": parts[1],
                    "acceptance_rate": float(parts[2]),
                    "R1": r1,
                    "R1_cl": r1_cl,
                })
            except (ValueError, IndexError):
                pass
    return records


def get_chain_samples(prefix: Path) -> int:
    """Count accepted samples in chain file."""
    chain_file = Path(f"{prefix}.1.txt")
    if not chain_file.exists():
        return 0
    return sum(1 for _ in open(chain_file)) - 1


def get_full_summary(prefix: Path) -> dict | None:
    """Get full parameter summary from chain: k_f (mean, median, CIs), σ₈, H₀."""
    try:
        from getdist import loadMCSamples
        samples = loadMCSamples(str(prefix), no_cache=True)
    except Exception:
        return None

    result = {}

    # k_f
    try:
        kf_vals = samples.getParams().mtdf_k_f
        weights = samples.weights
        kf_mean = float(samples.mean("mtdf_k_f"))
        kf_std = float(samples.std("mtdf_k_f"))

        # Weighted median
        sorted_idx = np.argsort(kf_vals)
        cum_w = np.cumsum(weights[sorted_idx])
        cum_w /= cum_w[-1]
        kf_median = float(kf_vals[sorted_idx[np.searchsorted(cum_w, 0.5)]])

        lo68, hi68 = samples.twoTailLimits("mtdf_k_f", 0.68)
        lo95, hi95 = samples.twoTailLimits("mtdf_k_f", 0.95)
        result["kf"] = {
            "mean": kf_mean, "std": kf_std, "median": kf_median,
            "68CI": [float(lo68), float(hi68)],
            "95CI": [float(lo95), float(hi95)],
            "95CI_width": float(hi95 - lo95),
        }
    except Exception:
        result["kf"] = None

    # σ₈
    try:
        result["sigma8"] = {
            "mean": float(samples.mean("sigma8")),
            "std": float(samples.std("sigma8")),
        }
    except Exception:
        result["sigma8"] = None

    # H₀
    try:
        result["H0"] = {
            "mean": float(samples.mean("H0")),
            "std": float(samples.std("H0")),
        }
    except Exception:
        result["H0"] = None

    return result


def is_analysis_ready(records: list[dict], n_samples: int,
                      prev_summary: dict | None,
                      curr_summary: dict | None) -> tuple[bool, str]:
    """Check readiness. Returns (ready, reason)."""
    if n_samples < MIN_ACCEPTED:
        return False, f"N={n_samples} < {MIN_ACCEPTED}"

    if len(records) < 2:
        return False, "< 2 convergence checks"

    r1_last = records[-1]["R1"]
    r1_prev = records[-2]["R1"]

    if r1_last is None or r1_prev is None:
        return False, "R-1 not available"

    if r1_last >= CONVERGENCE_R1:
        return False, f"R-1={r1_last:.4f} >= {CONVERGENCE_R1}"

    if r1_prev >= CONVERGENCE_R1:
        return False, f"R-1 not below threshold in 2 consecutive (prev={r1_prev:.4f})"

    # k_f stability: mean
    if (prev_summary and curr_summary
            and prev_summary.get("kf") and curr_summary.get("kf")):
        prev_kf = prev_summary["kf"]
        curr_kf = curr_summary["kf"]
        mean_delta = abs(curr_kf["mean"] - prev_kf["mean"])
        if mean_delta > KF_MEAN_TOL:
            return False, f"k_f mean moving: Δ={mean_delta:.3f} > {KF_MEAN_TOL}"

        # k_f stability: 95% CI width
        ci_delta = abs(curr_kf["95CI_width"] - prev_kf["95CI_width"])
        if ci_delta > KF_CI_TOL:
            return False, f"k_f 95% CI width moving: Δ={ci_delta:.3f} > {KF_CI_TOL}"

    return True, "all criteria met"


def run_interim_analysis(chain_key: str):
    """Launch interim analysis for a chain."""
    log(f"  Launching interim analysis for {chain_key}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mtdf_validation.phase5_plik.analyze_test2_test4",
             "--interim", chain_key],
            capture_output=True, text=True, timeout=300,
            cwd="../../",
        )
        if result.returncode == 0:
            log(f"  Interim analysis for {chain_key} completed successfully")
            for line in result.stdout.strip().split("\n"):
                if any(kw in line.lower() for kw in ["shift", "decision", "pass", "fail", "interim"]):
                    log(f"    {line.strip()}")
        else:
            log(f"  Interim analysis for {chain_key} FAILED (rc={result.returncode})")
            for line in result.stderr.strip().split("\n")[-5:]:
                log(f"    {line.strip()}")
    except subprocess.TimeoutExpired:
        log(f"  Interim analysis for {chain_key} timed out")


def run_final_analysis():
    """Launch final analysis for all tests."""
    log("=" * 70)
    log("ALL CHAINS CONVERGED — running final analysis")
    log("=" * 70)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mtdf_validation.phase5_plik.analyze_test2_test4",
             "--all"],
            capture_output=True, text=True, timeout=600,
            cwd="../../",
        )
        if result.returncode == 0:
            log("Final analysis completed successfully")
            for line in result.stdout.strip().split("\n"):
                log(f"  {line}")
        else:
            log(f"Final analysis FAILED (rc={result.returncode})")
            for line in result.stderr.strip().split("\n")[-10:]:
                log(f"  {line}")
    except subprocess.TimeoutExpired:
        log("Final analysis timed out")


def format_summary_line(label: str, n: int, r1, summary: dict | None, reason: str = "") -> str:
    """Format a single chain's status into a log line."""
    r1_str = f"{r1:.4f}" if r1 is not None else "  pend"
    parts = [f"{label:<15s}: N={n:>5d}, R-1={r1_str:>8s}"]

    if summary and summary.get("kf"):
        kf = summary["kf"]
        parts.append(
            f"k_f: mean={kf['mean']:.3f} med={kf['median']:.3f} "
            f"95%=[{kf['95CI'][0]:.2f},{kf['95CI'][1]:.2f}]"
        )
    if summary and summary.get("sigma8"):
        s8 = summary["sigma8"]
        parts.append(f"σ8={s8['mean']:.4f}±{s8['std']:.4f}")
    if summary and summary.get("H0"):
        h0 = summary["H0"]
        parts.append(f"H0={h0['mean']:.2f}±{h0['std']:.2f}")
    if reason:
        parts.append(f"[{reason}]")

    return "  " + "  |  ".join(parts)


def main():
    log("Monitor started (v2 — expanded tracking)")
    log(f"R-1 method: Rminus1_single_split=4 (chain split into 4 segments)")
    log(f"Convergence: R-1 < {CONVERGENCE_R1} (2 consecutive), N >= {MIN_ACCEPTED}")
    log(f"Stability: k_f mean Δ < {KF_MEAN_TOL}, 95% CI width Δ < {KF_CI_TOL}")
    log(f"Polling every {POLL_INTERVAL}s")
    log("")

    state = {
        key: {
            "interim_done": False,
            "ready": False,
            "prev_summary": None,
            "last_n_records": 0,
        }
        for key in CHAINS
    }

    while True:
        for key, info in CHAINS.items():
            if state[key]["ready"]:
                continue

            n_samples = get_chain_samples(info["prefix"])
            records = parse_progress(info["prefix"])

            # Get full summary if enough samples
            curr_summary = None
            if n_samples >= 500:
                curr_summary = get_full_summary(info["prefix"])

            # Check if new cobaya convergence check appeared
            n_records = len(records)
            new_check = n_records > state[key]["last_n_records"]
            state[key]["last_n_records"] = n_records

            r1_latest = records[-1]["R1"] if records else None

            # Log on new convergence check (not every poll)
            if new_check:
                ready, reason = is_analysis_ready(
                    records, n_samples,
                    state[key]["prev_summary"],
                    curr_summary,
                )
                log(format_summary_line(info["label"], n_samples, r1_latest,
                                        curr_summary, reason))

                if ready and not state[key]["interim_done"]:
                    log(f"  >>> {info['label']} is ANALYSIS-READY <<<")
                    state[key]["ready"] = True
                    state[key]["interim_done"] = True
                    run_interim_analysis(key)

                # Update previous summary for next stability check
                if curr_summary:
                    state[key]["prev_summary"] = curr_summary

        # Check if all done
        all_ready = all(state[k]["ready"] for k in CHAINS)
        if all_ready:
            log("")
            run_final_analysis()
            log("")
            log("Monitor complete. All robustness tests analysed.")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
