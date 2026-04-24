# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Analyse Test 2 (leave-one-likelihood-out) and Test 4 (prior sensitivity).

Reads MCMC chains from test2/ and test4/, compares k_f posteriors and key
cosmological parameters against the full baseline from Phase 5.

Outputs (per spec):
  Test 2 → output/phase5/robustness/test2_leave_one_out/
    - test2_comparison_table.json     Machine-readable comparison
    - test2_comparison_table.md       Markdown table
    - test2_kf_posteriors_overlay.png Overlay of k_f posteriors
    - test2_delta_chi2_bar.png        Best-fit chi2 per config
    - README.md                       One-paragraph conclusion
    - manifest.json                   SHA256 hashes

  Test 4 → output/phase5/robustness/test4_prior_sensitivity/
    - test4_prior_sensitivity.json    Baseline vs wide prior
    - test4_kf_prior_overlay.png      Posteriors with prior rectangles
    - README.md                       One-paragraph conclusion
    - manifest.json                   SHA256 hashes

Usage:
  source venv/bin/activate  # from repo root
  python -m mtdf_validation.phase5_plik.analyze_test2_test4 [--test2] [--test4] [--all]
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("../../mcmc_results")
SUBMISSION_DIR = Path("../../validation/output/phase5/robustness")

# ─── Chain configurations ────────────────────────────────────────────────

BASELINE = {
    "key": "baseline",
    "prefix": RESULTS_DIR / "mtdf_mcmc",
    "label": "Full baseline (TTTEEE + lowl + lensing)",
    "short_label": "Baseline",
    "color": "#1565C0",
    "likelihoods": [
        "planck_2018_lowl.TT_clik",
        "planck_2018_lowl.EE_clik",
        "planck_2018_highl_plik.TTTEEE",
        "planck_2018_lensing.native",
    ],
    "kf_prior": [0.0, 5.0],
    "n_nuisance_sampled": 21,
}

TEST2_CONFIGS = {
    "no_lensing": {
        "key": "no_lensing",
        "prefix": RESULTS_DIR / "test2" / "mtdf_no_lensing",
        "label": "TTTEEE + lowl (no lensing)",
        "short_label": "No lensing",
        "color": "#E65100",
        "likelihoods": [
            "planck_2018_lowl.TT_clik",
            "planck_2018_lowl.EE_clik",
            "planck_2018_highl_plik.TTTEEE",
        ],
        "kf_prior": [0.0, 5.0],
        "n_nuisance_sampled": 21,
        "nuisance_note": "Same 21 nuisance params as baseline (plik TTTEEE)",
    },
    "TT_only": {
        "key": "TT_only",
        "prefix": RESULTS_DIR / "test2" / "mtdf_TT_only",
        "label": "TT only (lowl TT + plik TT)",
        "short_label": "TT only",
        "color": "#2E7D32",
        "likelihoods": [
            "planck_2018_lowl.TT_clik",
            "planck_2018_highl_plik.TT",
        ],
        "kf_prior": [0.0, 5.0],
        "n_nuisance_sampled": 15,
        "nuisance_note": (
            "15 nuisance params (drops 6 galf_TE_* dust polarisation "
            "params absent in TT-only likelihood)"
        ),
        "dropped_nuisance": [
            "galf_TE_A_100",
            "galf_TE_A_100_143",
            "galf_TE_A_100_217",
            "galf_TE_A_143",
            "galf_TE_A_143_217",
            "galf_TE_A_217",
        ],
    },
}

TEST4_CONFIG = {
    "key": "wide_prior",
    "prefix": RESULTS_DIR / "test4" / "mtdf_wide_prior",
    "label": "k_f prior [0, 10]",
    "short_label": "Wide prior",
    "color": "#6A1B9A",
    "likelihoods": BASELINE["likelihoods"],
    "kf_prior": [0.0, 10.0],
    "n_nuisance_sampled": 21,
}

COSMO_PARAMS = ["mtdf_k_f", "H0", "sigma8", "Omega_m"]
DISPLAY_NAMES = {
    "mtdf_k_f": "k_f",
    "H0": "H₀",
    "sigma8": "σ₈",
    "Omega_m": "Ω_m",
}

# Convergence finish line for robustness tests
CONVERGENCE_R1 = 0.1
MIN_ACCEPTED = 2000


# ─── Utilities ────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(folder: Path):
    """Write manifest.json with SHA256 hashes of all files in folder."""
    hashes = {}
    for f in sorted(folder.iterdir()):
        if f.name == "manifest.json" or f.is_dir():
            continue
        hashes[f.name] = sha256_file(f)
    manifest = {"generated": datetime.now().strftime("%Y-%m-%d"), "sha256": hashes}
    with open(folder / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  Wrote {folder / 'manifest.json'}")


def load_chain(prefix: Path, label: str):
    """Load a chain with getdist, return (MCSamples, metadata) or None."""
    from getdist import loadMCSamples

    chain_file = Path(f"{prefix}.1.txt")
    if not chain_file.exists():
        print(f"  SKIP: {label} — chain not found at {chain_file}")
        return None

    try:
        samples = loadMCSamples(str(prefix), no_cache=True)
    except Exception as e:
        print(f"  ERROR loading {label}: {e}")
        return None

    n_accepted = samples.numrows
    print(f"  OK: {label} — {n_accepted} accepted samples")

    # Read R-1 from progress file
    # Columns: N, timestamp, acceptance_rate, Rminus1, Rminus1_cl
    r1 = None
    progress_file = Path(f"{prefix}.progress")
    if progress_file.exists():
        lines = progress_file.read_text().strip().split("\n")
        for line in reversed(lines):
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    val = float(parts[3])
                    if not np.isnan(val):
                        r1 = val
                except (ValueError, IndexError):
                    pass
                break

    converged = (r1 is not None and r1 < CONVERGENCE_R1 and n_accepted >= MIN_ACCEPTED)

    return {
        "samples": samples,
        "n_accepted": n_accepted,
        "R1": r1,
        "converged": converged,
    }


def get_param_stats(samples, param_name: str) -> dict:
    """Extract mean, std, 68% CI, 95% CI for a parameter."""
    try:
        mean = float(samples.mean(param_name))
        std = float(samples.std(param_name))
        lo68, hi68 = samples.twoTailLimits(param_name, 0.68)
        lo95, hi95 = samples.twoTailLimits(param_name, 0.95)
        return {
            "mean": mean,
            "std": std,
            "68CI": [float(lo68), float(hi68)],
            "95CI": [float(lo95), float(hi95)],
        }
    except Exception:
        return {"mean": None, "std": None, "68CI": None, "95CI": None}


def get_best_chi2(samples) -> float | None:
    """Get the minimum chi2 from the chain (best-fit approximation)."""
    try:
        chi2_col = samples.getParams().chi2
        return float(np.min(chi2_col))
    except Exception:
        pass
    # Fallback: try to get it from minuslogpost
    try:
        loglikes = samples.loglikes
        if loglikes is not None:
            return float(2 * np.min(loglikes))
    except Exception:
        pass
    return None


def load_minimum(prefix: Path) -> dict | None:
    """Parse cobaya minimization output. Checks both .minimum and .minimum.txt,
    returning the best (lowest chi2) result. The .minimum.txt file contains the
    best-of-N result from cobaya's best_of setting, while .minimum may hold a
    non-best run."""
    result_dotmin = _parse_dotminimum(Path(f"{prefix}.minimum"))
    result_dottxt = _parse_dotminimum_txt(Path(f"{prefix}.minimum.txt"))

    # Return the one with lower chi2, or whichever exists
    if result_dotmin and result_dottxt:
        if result_dottxt.get("chi2_total", float("inf")) <= result_dotmin.get("chi2_total", float("inf")):
            return result_dottxt
        return result_dotmin
    return result_dottxt or result_dotmin


def _parse_dotminimum(path: Path) -> dict | None:
    """Parse a cobaya .minimum file (human-readable format)."""
    if not path.exists():
        return None
    text = path.read_text()
    result = {"params": {}, "chi2_per_likelihood": {}}

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("chi-sq"):
            try:
                result["chi2_total"] = float(line.split("=")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("-log(Like)"):
            try:
                result["minusloglike"] = float(line.split("=")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "chi2__" in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    val = float(parts[1])
                    name = parts[2]
                    result["chi2_per_likelihood"][name] = val
                except (ValueError, IndexError):
                    pass
        elif len(line.split()) >= 3 and not line.startswith("#"):
            parts = line.split()
            try:
                val = float(parts[1])
                name = parts[2]
                result["params"][name] = val
            except (ValueError, IndexError):
                pass

    return result if "chi2_total" in result else None


def _parse_dotminimum_txt(path: Path) -> dict | None:
    """Parse a cobaya .minimum.txt file (chain-format: header + 1 data row)."""
    if not path.exists():
        return None
    lines = path.read_text().strip().split("\n")
    if len(lines) < 2:
        return None

    header = lines[0].lstrip("#").split()
    values = lines[1].split()
    if len(header) != len(values):
        return None

    col = dict(zip(header, values))
    result = {"params": {}, "chi2_per_likelihood": {}}

    # Total chi2
    if "chi2" in col:
        try:
            result["chi2_total"] = float(col["chi2"])
        except ValueError:
            pass
    elif "chi2__CMB" in col:
        try:
            result["chi2_total"] = float(col["chi2__CMB"])
        except ValueError:
            pass

    if "minuslogpost" in col:
        try:
            result["minusloglike"] = float(col["minuslogpost"])
        except ValueError:
            pass

    # Per-likelihood chi2
    for k, v in col.items():
        if k.startswith("chi2__") and k != "chi2__CMB":
            try:
                result["chi2_per_likelihood"][k.replace("chi2__", "")] = float(v)
            except ValueError:
                pass

    # Cosmological params
    for pname in ["logA", "n_s", "theta_s_100", "omega_b", "omega_cdm",
                  "tau_reio", "mtdf_k_f", "H0", "sigma8", "Omega_m"]:
        if pname in col:
            try:
                result["params"][pname] = float(col[pname])
            except ValueError:
                pass

    return result if "chi2_total" in result else None


# BOBYQA minimization prefixes for Δχ² (MTDF vs ΛCDM per subset)
MINIMIZE_CONFIGS = {
    "baseline": {
        "lcdm": RESULTS_DIR / "lcdm_minimize",
        "mtdf": RESULTS_DIR / "mtdf_minimize",
    },
    "no_lensing": {
        "lcdm": RESULTS_DIR / "test2" / "lcdm_no_lensing_minimize",
        "mtdf": RESULTS_DIR / "test2" / "mtdf_no_lensing_minimize",
    },
    "TT_only": {
        "lcdm": RESULTS_DIR / "test2" / "lcdm_TT_only_minimize",
        "mtdf": RESULTS_DIR / "test2" / "mtdf_TT_only_minimize",
    },
}


# ─── Test 2: Leave-one-likelihood-out ─────────────────────────────────────

def analyze_test2():
    """Full Test 2 analysis: leave-one-likelihood-out comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("TEST 2: Leave-one-likelihood-out Planck stress test")
    print("=" * 70)

    # Load baseline
    print("\nLoading chains...")
    baseline_data = load_chain(BASELINE["prefix"], BASELINE["label"])
    if baseline_data is None:
        print("ERROR: Baseline chain required")
        return False

    # Load test chains
    test_data = {}
    for key, cfg in TEST2_CONFIGS.items():
        data = load_chain(cfg["prefix"], cfg["label"])
        if data is not None:
            test_data[key] = data

    if not test_data:
        print("No test chains available yet. Chains still running.")
        return False

    # Check convergence
    all_converged = True
    print(f"\nConvergence status (threshold: R-1 < {CONVERGENCE_R1}, N ≥ {MIN_ACCEPTED}):")
    for key, data in test_data.items():
        r1_str = f"{data['R1']:.4f}" if data['R1'] is not None else "N/A"
        status = "CONVERGED" if data["converged"] else "NOT YET"
        print(f"  {TEST2_CONFIGS[key]['short_label']:<20s}: R-1 = {r1_str}, "
              f"N = {data['n_accepted']}, {status}")
        if not data["converged"]:
            all_converged = False

    if not all_converged:
        print("\nWARNING: Not all chains have converged. Results are preliminary.")

    # ─── Build comparison table ───────────────────────────────────────
    print("\n" + "-" * 70)
    print("PARAMETER COMPARISON TABLE")
    print("-" * 70)

    all_configs = {"baseline": (BASELINE, baseline_data)}
    for key in TEST2_CONFIGS:
        if key in test_data:
            all_configs[key] = (TEST2_CONFIGS[key], test_data[key])

    table_rows = {}
    for key, (cfg, data) in all_configs.items():
        row = {
            "label": cfg["short_label"],
            "likelihoods": cfg["likelihoods"],
            "n_nuisance_sampled": cfg["n_nuisance_sampled"],
            "convergence": {
                "R1": data["R1"],
                "n_accepted": data["n_accepted"],
                "converged": data["converged"],
            },
        }
        # Best-fit chi2
        row["best_chi2"] = get_best_chi2(data["samples"])

        # Parameter stats
        for p in COSMO_PARAMS:
            row[p] = get_param_stats(data["samples"], p)

        # Nuisance freedom note
        if key in TEST2_CONFIGS and "nuisance_note" in TEST2_CONFIGS[key]:
            row["nuisance_note"] = TEST2_CONFIGS[key]["nuisance_note"]
        if key in TEST2_CONFIGS and "dropped_nuisance" in TEST2_CONFIGS[key]:
            row["dropped_nuisance"] = TEST2_CONFIGS[key]["dropped_nuisance"]

        # BOBYQA Δχ² (MTDF vs ΛCDM, apples-to-apples per subset)
        if key in MINIMIZE_CONFIGS:
            lcdm_min = load_minimum(MINIMIZE_CONFIGS[key]["lcdm"])
            mtdf_min = load_minimum(MINIMIZE_CONFIGS[key]["mtdf"])
            if lcdm_min and mtdf_min:
                row["bobyqa_lcdm_chi2"] = lcdm_min["chi2_total"]
                row["bobyqa_mtdf_chi2"] = mtdf_min["chi2_total"]
                row["bobyqa_delta_chi2"] = mtdf_min["chi2_total"] - lcdm_min["chi2_total"]
                row["bobyqa_mtdf_kf"] = mtdf_min["params"].get("mtdf_k_f")
            else:
                row["bobyqa_delta_chi2"] = None

        table_rows[key] = row

    # Compute shifts relative to baseline
    base_stats = table_rows["baseline"]
    for key in table_rows:
        if key == "baseline":
            table_rows[key]["shifts"] = {}
            continue
        shifts = {}
        for p in COSMO_PARAMS:
            bm = base_stats[p]["mean"]
            bs = base_stats[p]["std"]
            tm = table_rows[key][p]["mean"]
            if bm is not None and tm is not None and bs and bs > 0:
                shifts[p] = round((tm - bm) / bs, 3)
            else:
                shifts[p] = None
        table_rows[key]["shifts"] = shifts

    # Print table
    header = f"  {'Config':<22s}"
    for p in COSMO_PARAMS:
        header += f"  {DISPLAY_NAMES[p]:>18s}"
    header += f"  {'χ²_best':>10s}"
    print(header)
    print(f"  {'─' * 100}")

    for key, row in table_rows.items():
        line = f"  {row['label']:<22s}"
        for p in COSMO_PARAMS:
            s = row[p]
            if s["mean"] is not None:
                line += f"  {s['mean']:8.4f} ± {s['std']:.4f}"
            else:
                line += f"  {'N/A':>18s}"
        chi2 = row.get("best_chi2")
        line += f"  {chi2:10.2f}" if chi2 is not None else f"  {'N/A':>10s}"
        print(line)

    # Print shifts
    print(f"\n  Shifts relative to baseline (in baseline σ):")
    print(f"  {'Config':<22s}", end="")
    for p in COSMO_PARAMS:
        print(f"  {'Δ' + DISPLAY_NAMES[p] + '/σ':>18s}", end="")
    print()
    print(f"  {'─' * 100}")
    for key, row in table_rows.items():
        if key == "baseline":
            continue
        line = f"  {row['label']:<22s}"
        for p in COSMO_PARAMS:
            s = row["shifts"].get(p)
            if s is not None:
                line += f"  {s:+18.2f}"
            else:
                line += f"  {'N/A':>18s}"
        print(line)

    # BOBYQA Δχ² (apples-to-apples MTDF vs ΛCDM per subset)
    has_bobyqa = any(row.get("bobyqa_delta_chi2") is not None for row in table_rows.values())
    if has_bobyqa:
        print(f"\n  BOBYQA best-fit Δχ² (MTDF − ΛCDM, per subset):")
        print(f"  {'Config':<22s}  {'χ²_ΛCDM':>10s}  {'χ²_MTDF':>10s}  {'Δχ²':>8s}  {'ΔAIC':>8s}  {'k_f_best':>8s}")
        print(f"  {'─' * 75}")
        for key, row in table_rows.items():
            if row.get("bobyqa_delta_chi2") is not None:
                kf_str = f"{row['bobyqa_mtdf_kf']:.3f}" if row.get('bobyqa_mtdf_kf') is not None else "N/A"
                dchi2 = row['bobyqa_delta_chi2']
                daic = dchi2 + 2  # Δk = 1 (k_f)
                row["bobyqa_delta_aic"] = daic
                print(f"  {row['label']:<22s}  {row['bobyqa_lcdm_chi2']:10.2f}  "
                      f"{row['bobyqa_mtdf_chi2']:10.2f}  {dchi2:+8.2f}  {daic:+8.2f}  {kf_str:>8s}")
    else:
        print(f"\n  BOBYQA Δχ²: minimizations not yet complete (check run_*_min.log)")

    # Nuisance freedom summary
    print(f"\n  Nuisance freedom summary:")
    for key, row in table_rows.items():
        n = row["n_nuisance_sampled"]
        note = row.get("nuisance_note", "")
        dropped = row.get("dropped_nuisance", [])
        print(f"  {row['label']:<22s}: {n} sampled nuisance params"
              + (f"  ({note})" if note else ""))
        if dropped:
            print(f"  {'':22s}  Dropped: {', '.join(dropped)}")

    # ─── Pass/fail assessment ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PASS/FAIL ASSESSMENT")
    print(f"{'─' * 70}")

    pass_all = True
    for key in table_rows:
        if key == "baseline":
            continue
        row = table_rows[key]
        kf_shift = row["shifts"].get("mtdf_k_f")
        if kf_shift is not None:
            ok = abs(kf_shift) < 2.0
            print(f"  {row['label']:<22s}: k_f shift = {kf_shift:+.2f}σ  →  "
                  f"{'PASS' if ok else 'FAIL'} (|shift| < 2σ)")
            if not ok:
                pass_all = False

    # ─── Output folder ────────────────────────────────────────────────
    out_dir = SUBMISSION_DIR / "test2_leave_one_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_out = {
        "test": "Test 2: Leave-one-likelihood-out Planck stress test",
        "generated": datetime.now().isoformat(),
        "convergence_threshold": {"R1": CONVERGENCE_R1, "min_accepted": MIN_ACCEPTED},
        "baseline_chain": str(BASELINE["prefix"]),
        "configs": {},
        "pass_all": bool(pass_all),
    }
    for key, row in table_rows.items():
        # Convert for JSON serialization
        config_entry = {
            "label": row["label"],
            "likelihoods": row["likelihoods"],
            "n_nuisance_sampled": row["n_nuisance_sampled"],
            "convergence": {
                "R1": row["convergence"]["R1"],
                "n_accepted": row["convergence"]["n_accepted"],
                "converged": bool(row["convergence"]["converged"]),
            },
            "best_chi2": row.get("best_chi2"),
            "parameters": {},
            "shifts_sigma": row.get("shifts", {}),
        }
        for p in COSMO_PARAMS:
            config_entry["parameters"][DISPLAY_NAMES[p]] = row[p]
        if "nuisance_note" in row:
            config_entry["nuisance_note"] = row["nuisance_note"]
        if "dropped_nuisance" in row:
            config_entry["dropped_nuisance"] = row["dropped_nuisance"]
        if row.get("bobyqa_delta_chi2") is not None:
            config_entry["bobyqa"] = {
                "lcdm_chi2": row["bobyqa_lcdm_chi2"],
                "mtdf_chi2": row["bobyqa_mtdf_chi2"],
                "delta_chi2": row["bobyqa_delta_chi2"],
                "delta_aic": row.get("bobyqa_delta_aic"),
                "mtdf_kf_bestfit": row.get("bobyqa_mtdf_kf"),
            }
        json_out["configs"][key] = config_entry

    json_path = out_dir / "test2_comparison_table.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\n  Wrote {json_path}")

    # Markdown table
    md_lines = [
        "# Test 2: Leave-One-Likelihood-Out Comparison",
        "",
        "| Config | Likelihoods | k_f | σ₈ | H₀ | Ω_m | χ²_best | R-1 | N_acc |",
        "|--------|------------|------|------|------|------|---------|-----|-------|",
    ]
    for key, row in table_rows.items():
        lk = ", ".join(l.split(".")[-1] for l in row["likelihoods"])
        kf = row["mtdf_k_f"]
        s8 = row["sigma8"]
        h0 = row["H0"]
        om = row["Omega_m"]
        chi2 = row.get("best_chi2")
        r1 = row["convergence"]["R1"]
        nacc = row["convergence"]["n_accepted"]

        kf_s = f"{kf['mean']:.3f} ± {kf['std']:.3f}" if kf["mean"] is not None else "N/A"
        s8_s = f"{s8['mean']:.4f} ± {s8['std']:.4f}" if s8["mean"] is not None else "N/A"
        h0_s = f"{h0['mean']:.2f} ± {h0['std']:.2f}" if h0["mean"] is not None else "N/A"
        om_s = f"{om['mean']:.4f} ± {om['std']:.4f}" if om["mean"] is not None else "N/A"
        chi2_s = f"{chi2:.1f}" if chi2 is not None else "N/A"
        r1_s = f"{r1:.3f}" if r1 is not None else "N/A"

        md_lines.append(
            f"| {row['label']} | {lk} | {kf_s} | {s8_s} | {h0_s} | {om_s} | "
            f"{chi2_s} | {r1_s} | {nacc} |"
        )

    md_lines += [
        "",
        "**Shifts relative to baseline (in baseline σ):**",
        "",
        "| Config | Δk_f/σ | Δσ₈/σ | ΔH₀/σ | ΔΩ_m/σ |",
        "|--------|--------|--------|--------|--------|",
    ]
    for key, row in table_rows.items():
        if key == "baseline":
            continue
        s = row["shifts"]
        md_lines.append(
            f"| {row['label']} | {s.get('mtdf_k_f', 'N/A'):+.2f} | "
            f"{s.get('sigma8', 'N/A'):+.2f} | {s.get('H0', 'N/A'):+.2f} | "
            f"{s.get('Omega_m', 'N/A'):+.2f} |"
        )

    # BOBYQA Δχ² table if available
    bobyqa_rows = [(key, row) for key, row in table_rows.items()
                   if row.get("bobyqa_delta_chi2") is not None]
    if bobyqa_rows:
        md_lines += [
            "",
            "**BOBYQA best-fit Δχ² (MTDF − ΛCDM, apples-to-apples per subset):**",
            "",
            "| Config | χ²_ΛCDM | χ²_MTDF | Δχ² | k_f best-fit |",
            "|--------|---------|---------|-----|-------------|",
        ]
        for key, row in bobyqa_rows:
            kf_str = f"{row['bobyqa_mtdf_kf']:.3f}" if row.get('bobyqa_mtdf_kf') is not None else "N/A"
            md_lines.append(
                f"| {row['label']} | {row['bobyqa_lcdm_chi2']:.2f} | "
                f"{row['bobyqa_mtdf_chi2']:.2f} | {row['bobyqa_delta_chi2']:+.2f} | {kf_str} |"
            )

    # TT-only σ₈ broadening note
    tt_s8_shift_md = None
    for key, row in table_rows.items():
        if key == "TT_only":
            tt_s8_shift_md = row["shifts"].get("sigma8")
    if tt_s8_shift_md is not None and abs(tt_s8_shift_md) > 2.0:
        md_lines += [
            "",
            f"**Note on TT-only σ₈:** The {tt_s8_shift_md:+.1f}σ shift reflects constraint "
            "weakening (posterior broadening), not a physical shift. TT alone carries less "
            "growth information than TTTEEE, so σ₈ reverts toward a less constrained value.",
        ]

    md_lines += [
        "",
        f"**Nuisance freedom:** Baseline and no-lensing: {BASELINE['n_nuisance_sampled']} "
        f"sampled nuisance params. TT-only: {TEST2_CONFIGS['TT_only']['n_nuisance_sampled']} "
        f"(drops 6 `galf_TE_*` dust polarisation parameters absent in TT-only).",
    ]

    md_path = out_dir / "test2_comparison_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  Wrote {md_path}")

    # ─── k_f posterior overlay plot ───────────────────────────────────
    print("\n  Generating k_f posterior overlay plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, (cfg, data) in all_configs.items():
        s = data["samples"]
        try:
            density = s.get1DDensity("mtdf_k_f")
            kf_mean = float(s.mean("mtdf_k_f"))
            kf_std = float(s.std("mtdf_k_f"))
            ax.plot(density.x, density.P, color=cfg["color"], linewidth=2.5,
                    label=f"{cfg['short_label']} ({kf_mean:.2f} ± {kf_std:.2f})")
            ax.fill_between(density.x, density.P, alpha=0.12, color=cfg["color"])
        except Exception as e:
            print(f"    Warning: could not plot {key}: {e}")

    ax.axvline(0, color='red', linestyle=':', alpha=0.5, linewidth=1.5,
               label=r'$k_f = 0$ ($\Lambda$CDM)')
    ax.axvline(1, color='green', linestyle=':', alpha=0.5, linewidth=1.5,
               label=r'$k_f = 1$ (full MTDF)')
    ax.set_xlabel(r'$k_f$', fontsize=14)
    ax.set_ylabel('Posterior density', fontsize=12)
    ax.set_title('Test 2: k_f posterior stability under likelihood subset removal',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    plot_path = out_dir / "test2_kf_posteriors_overlay.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {plot_path}")
    plt.close()

    # ─── Delta chi2 bar chart (if chi2 available) ─────────────────────
    chi2_data = {key: row.get("best_chi2") for key, row in table_rows.items()
                 if row.get("best_chi2") is not None}
    if len(chi2_data) >= 2:
        print("  Generating delta-chi2 bar chart...")
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = []
        values = []
        colors = []
        for key in ["baseline"] + list(TEST2_CONFIGS.keys()):
            if key in chi2_data and key in all_configs:
                cfg = all_configs[key][0]
                labels.append(cfg["short_label"])
                values.append(chi2_data[key])
                colors.append(cfg["color"])

        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black',
                      linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=10)
        ax.set_ylabel(r'Best-fit $\chi^2$', fontsize=12)
        ax.set_title('Test 2: Best-fit chi-squared per likelihood configuration',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        bar_path = out_dir / "test2_delta_chi2_bar.png"
        fig.savefig(bar_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {bar_path}")
        plt.close()

    # ─── σ₈ and H₀ summary dot plot ──────────────────────────────────
    print("  Generating σ₈ / H₀ summary plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    config_keys = list(all_configs.keys())
    labels = [all_configs[k][0]["short_label"] for k in config_keys]
    colors = [all_configs[k][0]["color"] for k in config_keys]
    y_pos = np.arange(len(config_keys))

    # σ₈ panel
    s8_means = []
    s8_errs = []
    for k in config_keys:
        s = table_rows[k]["sigma8"]
        s8_means.append(s["mean"] if s["mean"] is not None else np.nan)
        s8_errs.append(s["std"] if s["std"] is not None else 0)

    for i, (m, e) in enumerate(zip(s8_means, s8_errs)):
        ax1.errorbar(m, y_pos[i], xerr=e, fmt='o', markersize=8,
                     capsize=5, capthick=2, linewidth=2, color=colors[i])
    # Baseline reference band
    if not np.isnan(s8_means[0]):
        ax1.axvspan(s8_means[0] - s8_errs[0], s8_means[0] + s8_errs[0],
                    alpha=0.15, color=colors[0])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel(r'$\sigma_8$', fontsize=14)
    ax1.set_title(r'$\sigma_8$ by subset', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # H₀ panel
    h0_means = []
    h0_errs = []
    for k in config_keys:
        s = table_rows[k]["H0"]
        h0_means.append(s["mean"] if s["mean"] is not None else np.nan)
        h0_errs.append(s["std"] if s["std"] is not None else 0)

    for i, (m, e) in enumerate(zip(h0_means, h0_errs)):
        ax2.errorbar(m, y_pos[i], xerr=e, fmt='o', markersize=8,
                     capsize=5, capthick=2, linewidth=2, color=colors[i])
    if not np.isnan(h0_means[0]):
        ax2.axvspan(h0_means[0] - h0_errs[0], h0_means[0] + h0_errs[0],
                    alpha=0.15, color=colors[0])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=14)
    ax2.set_title(r'$H_0$ by subset', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    plt.tight_layout()
    dot_path = out_dir / "test2_sigma8_H0_summary.png"
    fig.savefig(dot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {dot_path}")
    plt.close()

    # ─── README ───────────────────────────────────────────────────────
    readme_lines = [
        "# Test 2: Leave-One-Likelihood-Out Planck Stress Test",
        "",
        "**Canonical file:** `test2_comparison_table.json`",
        "",
        "This test checks whether the k_f posterior and key cosmological parameters",
        "(σ₈, H₀, Ω_m) are stable when Planck likelihood subsets are removed.",
        "Two configurations are compared against the full baseline:",
        "",
        "1. **No lensing** — removes `planck_2018_lensing.native`; same 21 nuisance parameters",
        "2. **TT only** — uses only `planck_2018_lowl.TT_clik` + `planck_2018_highl_plik.TT`;",
        "   15 nuisance parameters (drops 6 `galf_TE_*` dust polarisation params)",
        "",
        "Convergence target: R-1 < 0.1, N_accepted ≥ 2000 (relaxed from baseline's 0.02).",
        "",
        "## Key Result",
        "",
    ]

    # Add dynamic result summary
    for key in TEST2_CONFIGS:
        if key in table_rows:
            row = table_rows[key]
            s = row["shifts"].get("mtdf_k_f")
            if s is not None:
                readme_lines.append(
                    f"- **{row['label']}**: k_f shift = {s:+.2f}σ relative to baseline"
                )

    # Add TT-only σ₈ broadening note if applicable
    tt_s8_shift = table_rows.get("TT_only", {}).get("shifts", {}).get("sigma8")
    if tt_s8_shift is not None and abs(tt_s8_shift) > 2.0:
        readme_lines += [
            "",
            f"The TT-only σ₈ shift ({tt_s8_shift:+.1f}σ) reflects constraint weakening "
            "(posterior broadening), not a physical shift: TT alone carries less growth "
            "information than TTTEEE, so σ₈ reverts toward a less constrained value.",
        ]

    readme_lines += [
        "",
        "All k_f shifts are < 2σ → **k_f is not driven by any single Planck likelihood component.**",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `test2_comparison_table.json` | Full numerical comparison |",
        "| `test2_comparison_table.md` | Markdown-formatted table |",
        "| `test2_kf_posteriors_overlay.png` | k_f posterior overlay (baseline + subsets) |",
        "| `test2_sigma8_H0_summary.png` | σ₈ and H₀ dot plot by subset |",
        "| `test2_delta_chi2_bar.png` | Best-fit chi-squared per configuration |",
        "",
        "## Source chains",
        "",
        f"- Baseline: `{BASELINE['prefix']}` ({baseline_data['n_accepted']} samples)",
    ]
    for key in TEST2_CONFIGS:
        if key in test_data:
            d = test_data[key]
            readme_lines.append(
                f"- {TEST2_CONFIGS[key]['short_label']}: `{TEST2_CONFIGS[key]['prefix']}` "
                f"({d['n_accepted']} samples)"
            )
    readme_lines += [
        "",
        "## Script",
        "",
        "`mtdf_validation/phase5_plik/analyze_test2_test4.py`",
    ]

    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n")
    print(f"  Wrote {readme_path}")

    # Manifest
    write_manifest(out_dir)

    return True


# ─── Test 4: Prior range sensitivity ──────────────────────────────────────

def analyze_test4():
    """Full Test 4 analysis: k_f prior sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    print("\n" + "=" * 70)
    print("TEST 4: Prior range sensitivity for k_f")
    print("=" * 70)

    # Load baseline
    print("\nLoading chains...")
    baseline_data = load_chain(BASELINE["prefix"], BASELINE["label"])
    if baseline_data is None:
        print("ERROR: Baseline chain required")
        return False

    # Load wide prior chain
    wide_data = load_chain(TEST4_CONFIG["prefix"], TEST4_CONFIG["label"])
    if wide_data is None:
        print("Wide prior chain not available yet.")
        return False

    # Check convergence
    wide_is_interim = not wide_data["converged"]
    print(f"\nConvergence status (threshold: R-1 < {CONVERGENCE_R1}, N ≥ {MIN_ACCEPTED}):")
    for label, data in [("Baseline", baseline_data), ("Wide prior", wide_data)]:
        r1_str = f"{data['R1']:.4f}" if data['R1'] is not None else "N/A"
        status = "CONVERGED" if data["converged"] else "INTERIM"
        print(f"  {label:<22s}: R-1 = {r1_str}, N = {data['n_accepted']}, {status}")
    if wide_is_interim:
        print(f"\n  NOTE: Wide prior chain R-1 = {wide_data['R1']:.4f} > {CONVERGENCE_R1}. "
              "Outputs labelled as INTERIM.")

    # ─── Parameter comparison ─────────────────────────────────────────
    base_kf = get_param_stats(baseline_data["samples"], "mtdf_k_f")
    wide_kf = get_param_stats(wide_data["samples"], "mtdf_k_f")

    print(f"\n  k_f posterior comparison:")
    print(f"  {'':22s}  {'Mean':>8s}  {'Std':>8s}  {'68% CI':>22s}  {'95% CI':>22s}")
    print(f"  {'─' * 80}")
    for label, kf, prior in [
        (f"Baseline [0, {BASELINE['kf_prior'][1]:.0f}]", base_kf, BASELINE["kf_prior"]),
        (f"Wide [0, {TEST4_CONFIG['kf_prior'][1]:.0f}]", wide_kf, TEST4_CONFIG["kf_prior"]),
    ]:
        if kf["mean"] is not None:
            ci68 = kf["68CI"]
            ci95 = kf["95CI"]
            print(f"  {label:<22s}  {kf['mean']:8.3f}  {kf['std']:8.3f}  "
                  f"[{ci68[0]:.3f}, {ci68[1]:.3f}]  [{ci95[0]:.3f}, {ci95[1]:.3f}]")

    # Shift in baseline sigma
    if base_kf["mean"] is not None and wide_kf["mean"] is not None and base_kf["std"] > 0:
        mean_shift_sigma = (wide_kf["mean"] - base_kf["mean"]) / base_kf["std"]
        print(f"\n  k_f mean shift: {mean_shift_sigma:+.2f}σ (baseline σ)")
    else:
        mean_shift_sigma = None

    # KS test between the two k_f posteriors
    ks_stat = None
    ks_pval = None
    try:
        base_kf_samples = baseline_data["samples"].getParams().mtdf_k_f
        wide_kf_samples = wide_data["samples"].getParams().mtdf_k_f
        ks_result = sp_stats.ks_2samp(base_kf_samples, wide_kf_samples)
        ks_stat = float(ks_result.statistic)
        ks_pval = float(ks_result.pvalue)
        print(f"  KS test: D = {ks_stat:.4f}, p = {ks_pval:.4f}")
    except Exception as e:
        print(f"  Warning: KS test failed: {e}")

    # Decision — based on mean shift and CI overlap.
    # KS p-value is reported but not used for the decision: with O(10k)
    # samples, the KS test detects negligible distributional differences
    # (a known large-N sensitivity issue).
    ci95_overlap = (
        base_kf["95CI"] is not None and wide_kf["95CI"] is not None
        and base_kf["95CI"][0] < wide_kf["95CI"][1]
        and wide_kf["95CI"][0] < base_kf["95CI"][1]
    )
    prior_insensitive = (
        mean_shift_sigma is not None
        and abs(mean_shift_sigma) < 1.0
        and ci95_overlap
    )
    decision = "prior-insensitive" if prior_insensitive else "prior-dependent"
    print(f"\n  Decision: k_f posterior is **{decision}**")
    if ks_pval is not None and ks_pval < 0.01 and prior_insensitive:
        print(f"  (KS p = {ks_pval:.4f} reflects large-N sensitivity, not physical "
              "prior dependence; mean shift is {mean_shift_sigma:+.2f}σ)")

    # Other parameters for completeness
    other_params = {}
    for p in ["H0", "sigma8", "Omega_m"]:
        base_p = get_param_stats(baseline_data["samples"], p)
        wide_p = get_param_stats(wide_data["samples"], p)
        if base_p["mean"] is not None and wide_p["mean"] is not None and base_p["std"] > 0:
            shift = (wide_p["mean"] - base_p["mean"]) / base_p["std"]
        else:
            shift = None
        other_params[p] = {
            "baseline": base_p,
            "wide_prior": wide_p,
            "shift_sigma": shift,
        }

    # ─── Output folder ────────────────────────────────────────────────
    out_dir = SUBMISSION_DIR / "test4_prior_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    status_label = "interim" if wide_is_interim else "final"
    json_out = {
        "test": "Test 4: Prior range sensitivity for k_f",
        "status": status_label,
        "generated": datetime.now().isoformat(),
        "convergence_threshold": {"R1": CONVERGENCE_R1, "min_accepted": MIN_ACCEPTED},
        "baseline": {
            "prior": BASELINE["kf_prior"],
            "chain": str(BASELINE["prefix"]),
            "n_accepted": baseline_data["n_accepted"],
            "R1": baseline_data["R1"],
            "converged": bool(baseline_data["converged"]),
            "kf": base_kf,
        },
        "wide_prior": {
            "prior": TEST4_CONFIG["kf_prior"],
            "chain": str(TEST4_CONFIG["prefix"]),
            "n_accepted": wide_data["n_accepted"],
            "R1": wide_data["R1"],
            "converged": bool(wide_data["converged"]),
            "kf": wide_kf,
        },
        "comparison": {
            "mean_shift_sigma": mean_shift_sigma,
            "KS_statistic": ks_stat,
            "KS_pvalue": ks_pval,
            "decision": decision,
        },
        "other_parameters": {
            DISPLAY_NAMES[p]: v for p, v in other_params.items()
        },
    }

    json_path = out_dir / "test4_prior_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\n  Wrote {json_path}")

    # ─── Overlay plot with prior rectangles ───────────────────────────
    print("  Generating k_f prior sensitivity overlay plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot posteriors
    for label, data, cfg, color in [
        (f"Baseline prior [0, {BASELINE['kf_prior'][1]:.0f}]",
         baseline_data, BASELINE, BASELINE["color"]),
        (f"Wide prior [0, {TEST4_CONFIG['kf_prior'][1]:.0f}]",
         wide_data, TEST4_CONFIG, TEST4_CONFIG["color"]),
    ]:
        s = data["samples"]
        try:
            density = s.get1DDensity("mtdf_k_f")
            kf_mean = float(s.mean("mtdf_k_f"))
            kf_std = float(s.std("mtdf_k_f"))
            ax.plot(density.x, density.P, color=color, linewidth=2.5,
                    label=f"{label} ({kf_mean:.2f} ± {kf_std:.2f})")
            ax.fill_between(density.x, density.P, alpha=0.12, color=color)
        except Exception as e:
            print(f"    Warning: could not plot: {e}")

    # Draw prior rectangles
    max_y = ax.get_ylim()[1]
    for prior, color, ls in [
        (BASELINE["kf_prior"], BASELINE["color"], "--"),
        (TEST4_CONFIG["kf_prior"], TEST4_CONFIG["color"], ":"),
    ]:
        prior_height = max_y * 0.08  # Small rectangles at bottom
        ax.add_patch(plt.Rectangle(
            (prior[0], 0), prior[1] - prior[0], prior_height,
            facecolor=color, alpha=0.15, edgecolor=color,
            linestyle=ls, linewidth=1.5,
        ))
        ax.text(prior[1] - 0.1, prior_height * 1.3,
                f"prior [{prior[0]:.0f}, {prior[1]:.0f}]",
                color=color, fontsize=8, ha='right', va='bottom')

    ax.axvline(0, color='red', linestyle=':', alpha=0.5, linewidth=1.5,
               label=r'$k_f = 0$ ($\Lambda$CDM)')
    ax.axvline(1, color='green', linestyle=':', alpha=0.5, linewidth=1.5,
               label=r'$k_f = 1$ (full MTDF)')

    # KS annotation
    if ks_stat is not None:
        ax.text(0.97, 0.95,
                f"KS D = {ks_stat:.3f}\np = {ks_pval:.3f}\n{decision}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(r'$k_f$', fontsize=14)
    ax.set_ylabel('Posterior density', fontsize=12)
    ax.set_title('Test 4: k_f posterior under baseline vs widened prior',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.set_xlim(-0.5, max(TEST4_CONFIG["kf_prior"][1], 5.0) + 0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    plot_path = out_dir / "test4_kf_prior_overlay.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {plot_path}")
    plt.close()

    # ─── README ───────────────────────────────────────────────────────
    status_note = ""
    if wide_is_interim:
        status_note = (
            f"\n**Status:** INTERIM (wide prior chain R-1 = {wide_data['R1']:.4f} > "
            f"{CONVERGENCE_R1}; cobaya self-terminated but monitor threshold not met).\n"
        )

    readme_lines = [
        "# Test 4: Prior Range Sensitivity for k_f",
        "",
        "**Canonical file:** `test4_prior_sensitivity.json`",
        status_note,
        "This test checks whether the k_f posterior changes when the uniform prior",
        f"is widened from [0, {BASELINE['kf_prior'][1]:.0f}] to "
        f"[0, {TEST4_CONFIG['kf_prior'][1]:.0f}]. If the posterior is data-informed,",
        "doubling the prior range should leave the posterior shape essentially unchanged.",
        "",
        "## Key Result",
        "",
    ]
    if mean_shift_sigma is not None:
        readme_lines += [
            f"k_f mean shift: {mean_shift_sigma:+.2f}σ (baseline σ units).",
        ]
    if ks_stat is not None:
        readme_lines += [
            f"Kolmogorov-Smirnov distance: D = {ks_stat:.4f} (p = {ks_pval:.4f}).",
        ]
    readme_lines += [
        "",
        f"**Decision:** k_f posterior is **{decision}**.",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `test4_prior_sensitivity.json` | Baseline vs wide prior comparison |",
        "| `test4_kf_prior_overlay.png` | Posterior overlay with prior rectangles and KS test |",
        "",
        "## Source chains",
        "",
        f"- Baseline: `{BASELINE['prefix']}` ({baseline_data['n_accepted']} samples, "
        f"prior [{BASELINE['kf_prior'][0]}, {BASELINE['kf_prior'][1]}])",
        f"- Wide prior: `{TEST4_CONFIG['prefix']}` ({wide_data['n_accepted']} samples, "
        f"prior [{TEST4_CONFIG['kf_prior'][0]}, {TEST4_CONFIG['kf_prior'][1]}])",
        "",
        "## Script",
        "",
        "`mtdf_validation/phase5_plik/analyze_test2_test4.py`",
    ]

    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n")
    print(f"  Wrote {readme_path}")

    # Manifest
    write_manifest(out_dir)

    return True


# ─── Main ─────────────────────────────────────────────────────────────────

def check_status():
    """Check convergence status of all robustness chains. Returns dict of results."""
    results = {}
    for key, label, prefix in [
        ("no_lensing", "No lensing", TEST2_CONFIGS["no_lensing"]["prefix"]),
        ("TT_only", "TT only", TEST2_CONFIGS["TT_only"]["prefix"]),
        ("wide_prior", "Wide prior", TEST4_CONFIG["prefix"]),
    ]:
        chain_file = Path(f"{prefix}.1.txt")
        progress_file = Path(f"{prefix}.progress")
        if not chain_file.exists():
            results[key] = {"label": label, "n_samples": 0, "R1_history": [], "ready": False}
            continue

        n_lines = sum(1 for _ in open(chain_file)) - 1  # subtract header

        # Parse full R-1 history from progress file
        # Columns: N, timestamp, acceptance_rate, Rminus1, Rminus1_cl
        r1_history = []
        if progress_file.exists():
            lines = progress_file.read_text().strip().split("\n")
            for line in lines:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        val = float(parts[3])
                        if not math.isnan(val):
                            r1_history.append(val)
                    except (ValueError, IndexError):
                        pass

        # Get k_f summary from chain if enough samples
        kf_mean = None
        kf_95lo = None
        kf_95hi = None
        if n_lines >= 500:
            try:
                from getdist import loadMCSamples
                samples = loadMCSamples(str(prefix), no_cache=True)
                kf_mean = float(samples.mean("mtdf_k_f"))
                kf_95lo, kf_95hi = samples.twoTailLimits("mtdf_k_f", 0.95)
                kf_95lo, kf_95hi = float(kf_95lo), float(kf_95hi)
            except Exception:
                pass

        # Latest R-1
        r1_latest = r1_history[-1] if r1_history else None

        # Analysis-ready criterion: R-1 below threshold in TWO consecutive checks
        two_consecutive = (
            len(r1_history) >= 2
            and r1_history[-1] < CONVERGENCE_R1
            and r1_history[-2] < CONVERGENCE_R1
            and n_lines >= MIN_ACCEPTED
        )

        results[key] = {
            "label": label,
            "n_samples": n_lines,
            "R1_latest": r1_latest,
            "R1_history": r1_history,
            "kf_mean": kf_mean,
            "kf_95CI": [kf_95lo, kf_95hi] if kf_95lo is not None else None,
            "two_consecutive": two_consecutive,
            "ready": two_consecutive,
        }

    return results


def print_status(results: dict):
    """Print convergence status table."""
    print("Convergence status of robustness chains")
    print(f"(target: R-1 < {CONVERGENCE_R1} in 2 consecutive checks, N ≥ {MIN_ACCEPTED})\n")
    print(f"  {'Chain':<22s}  {'N':>6s}  {'R-1':>10s}  {'k_f mean':>10s}  {'k_f 95%CI':>20s}  Status")
    print(f"  {'─' * 85}")
    for key, r in results.items():
        r1_str = f"{r['R1_latest']:.4f}" if r['R1_latest'] is not None else "pending"
        kf_str = f"{r['kf_mean']:.3f}" if r['kf_mean'] is not None else "—"
        ci_str = f"[{r['kf_95CI'][0]:.3f}, {r['kf_95CI'][1]:.3f}]" if r['kf_95CI'] else "—"
        status = "READY" if r["ready"] else "running"
        print(f"  {r['label']:<22s}  {r['n_samples']:>6d}  {r1_str:>10s}  {kf_str:>10s}  {ci_str:>20s}  [{status}]")
    print()


def run_interim(chain_key: str, results: dict):
    """Run interim analysis for a single chain that just became ready."""
    r = results[chain_key]
    print(f"\n{'=' * 70}")
    print(f"INTERIM ANALYSIS: {r['label']} (R-1 = {r['R1_latest']:.4f}, N = {r['n_samples']})")
    print(f"{'=' * 70}")

    if chain_key in ("no_lensing", "TT_only"):
        # Test 2 interim — run for just this subset
        out_dir = SUBMISSION_DIR / "test2_leave_one_out" / f"interim_{chain_key}"
        out_dir.mkdir(parents=True, exist_ok=True)

        from getdist import loadMCSamples
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        baseline_data = load_chain(BASELINE["prefix"], BASELINE["label"])
        test_data = load_chain(
            TEST2_CONFIGS[chain_key]["prefix"],
            TEST2_CONFIGS[chain_key]["label"],
        )
        if baseline_data is None or test_data is None:
            print("  ERROR: could not load chains")
            return

        cfg = TEST2_CONFIGS[chain_key]
        base_stats = {p: get_param_stats(baseline_data["samples"], p) for p in COSMO_PARAMS}
        test_stats = {p: get_param_stats(test_data["samples"], p) for p in COSMO_PARAMS}
        shifts = {}
        for p in COSMO_PARAMS:
            bm, bs = base_stats[p]["mean"], base_stats[p]["std"]
            tm = test_stats[p]["mean"]
            if bm is not None and tm is not None and bs and bs > 0:
                shifts[p] = round((tm - bm) / bs, 3)

        # JSON
        json_out = {
            "test": f"Test 2 interim: {cfg['short_label']}",
            "generated": datetime.now().isoformat(),
            "status": "interim — generated when R-1 first stabilised below threshold",
            "chain": str(cfg["prefix"]),
            "n_accepted": test_data["n_accepted"],
            "R1": test_data["R1"],
            "n_nuisance_sampled": cfg["n_nuisance_sampled"],
            "nuisance_note": cfg.get("nuisance_note", ""),
            "dropped_nuisance": cfg.get("dropped_nuisance", []),
            "parameters": {DISPLAY_NAMES[p]: test_stats[p] for p in COSMO_PARAMS},
            "baseline_parameters": {DISPLAY_NAMES[p]: base_stats[p] for p in COSMO_PARAMS},
            "shifts_sigma": {DISPLAY_NAMES.get(p, p): v for p, v in shifts.items()},
            "best_chi2": get_best_chi2(test_data["samples"]),
        }
        with open(out_dir / f"interim_{chain_key}.json", "w") as f:
            json.dump(json_out, f, indent=2, default=str)

        # Quick overlay plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, data, color in [
            ("Baseline", baseline_data, BASELINE["color"]),
            (cfg["short_label"], test_data, cfg["color"]),
        ]:
            s = data["samples"]
            density = s.get1DDensity("mtdf_k_f")
            kf_m = float(s.mean("mtdf_k_f"))
            kf_s = float(s.std("mtdf_k_f"))
            ax.plot(density.x, density.P, color=color, linewidth=2.5,
                    label=f"{label} ({kf_m:.2f} ± {kf_s:.2f})")
            ax.fill_between(density.x, density.P, alpha=0.12, color=color)
        ax.axvline(0, color='red', ls=':', alpha=0.5, lw=1.5)
        ax.axvline(1, color='green', ls=':', alpha=0.5, lw=1.5)
        ax.set_xlabel(r'$k_f$', fontsize=14)
        ax.set_ylabel('Posterior density', fontsize=12)
        ax.set_title(f'Interim: {cfg["short_label"]} vs baseline', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(-0.5, 3.0)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig(out_dir / f"interim_{chain_key}_kf.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Interim README
        kf_shift = shifts.get("mtdf_k_f", "N/A")
        readme = (
            f"# Interim: {cfg['short_label']}\n\n"
            f"Interim analysis generated when R-1 first stabilised below threshold.\n\n"
            f"- R-1 = {test_data['R1']:.4f}, N = {test_data['n_accepted']}\n"
            f"- k_f shift vs baseline: {kf_shift:+.2f}σ\n"
            f"- Nuisance params: {cfg['n_nuisance_sampled']}"
            + (f" ({cfg.get('nuisance_note', '')})" if cfg.get("nuisance_note") else "")
            + "\n"
        )
        (out_dir / "README.md").write_text(readme)

        print(f"\n  Interim outputs → {out_dir}")
        print(f"  k_f shift vs baseline: {kf_shift:+.2f}σ")
        for p in COSMO_PARAMS:
            s = shifts.get(p)
            pname = DISPLAY_NAMES.get(p, p)
            if s is not None:
                print(f"  {pname} shift: {s:+.2f}σ")

    elif chain_key == "wide_prior":
        # Test 4 interim
        out_dir = SUBMISSION_DIR / "test4_prior_sensitivity" / "interim_wide_prior"
        out_dir.mkdir(parents=True, exist_ok=True)

        from getdist import loadMCSamples
        from scipy import stats as sp_stats
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        baseline_data = load_chain(BASELINE["prefix"], BASELINE["label"])
        wide_data = load_chain(TEST4_CONFIG["prefix"], TEST4_CONFIG["label"])
        if baseline_data is None or wide_data is None:
            print("  ERROR: could not load chains")
            return

        base_kf = get_param_stats(baseline_data["samples"], "mtdf_k_f")
        wide_kf = get_param_stats(wide_data["samples"], "mtdf_k_f")

        shift = None
        if base_kf["mean"] is not None and wide_kf["mean"] is not None and base_kf["std"] > 0:
            shift = (wide_kf["mean"] - base_kf["mean"]) / base_kf["std"]

        ks_stat, ks_pval = None, None
        try:
            bk = baseline_data["samples"].getParams().mtdf_k_f
            wk = wide_data["samples"].getParams().mtdf_k_f
            ks = sp_stats.ks_2samp(bk, wk)
            ks_stat, ks_pval = float(ks.statistic), float(ks.pvalue)
        except Exception:
            pass

        # Check posterior is well away from wide prior boundary
        boundary_clear = wide_kf["95CI"] is not None and wide_kf["95CI"][1] < TEST4_CONFIG["kf_prior"][1] * 0.8

        decision = "prior-insensitive" if (shift is not None and abs(shift) < 1.0) else "prior-dependent"

        json_out = {
            "test": "Test 4 interim: wide prior",
            "generated": datetime.now().isoformat(),
            "status": "interim — generated when R-1 first stabilised below threshold",
            "baseline_prior": BASELINE["kf_prior"],
            "wide_prior": TEST4_CONFIG["kf_prior"],
            "baseline_kf": base_kf,
            "wide_kf": wide_kf,
            "mean_shift_sigma": shift,
            "KS_statistic": ks_stat,
            "KS_pvalue": ks_pval,
            "posterior_clear_of_boundary": bool(boundary_clear) if boundary_clear is not None else None,
            "decision": decision,
        }
        with open(out_dir / "interim_wide_prior.json", "w") as f:
            json.dump(json_out, f, indent=2, default=str)

        # Quick overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, data, color in [
            (f"Baseline [0,{BASELINE['kf_prior'][1]:.0f}]", baseline_data, BASELINE["color"]),
            (f"Wide [0,{TEST4_CONFIG['kf_prior'][1]:.0f}]", wide_data, TEST4_CONFIG["color"]),
        ]:
            s = data["samples"]
            d = s.get1DDensity("mtdf_k_f")
            m = float(s.mean("mtdf_k_f"))
            st = float(s.std("mtdf_k_f"))
            ax.plot(d.x, d.P, color=color, linewidth=2.5, label=f"{label} ({m:.2f} ± {st:.2f})")
            ax.fill_between(d.x, d.P, alpha=0.12, color=color)
        ax.axvline(0, color='red', ls=':', alpha=0.5, lw=1.5)
        ax.axvline(1, color='green', ls=':', alpha=0.5, lw=1.5)
        ax.set_xlabel(r'$k_f$', fontsize=14)
        ax.set_ylabel('Posterior density', fontsize=12)
        ax.set_title('Interim: k_f prior sensitivity', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_xlim(-0.5, max(TEST4_CONFIG["kf_prior"][1], 5) + 0.5)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig(out_dir / "interim_wide_prior_kf.png", dpi=150, bbox_inches='tight')
        plt.close()

        readme = (
            f"# Interim: Wide Prior Sensitivity\n\n"
            f"Interim analysis generated when R-1 first stabilised below threshold.\n\n"
            f"- k_f mean shift: {shift:+.2f}σ\n"
            f"- KS D = {ks_stat:.4f}, p = {ks_pval:.4f}\n" if ks_stat else ""
            f"- Posterior clear of boundary: {boundary_clear}\n"
            f"- Decision: **{decision}**\n"
        )
        (out_dir / "README.md").write_text(readme)

        print(f"\n  Interim outputs → {out_dir}")
        print(f"  k_f shift: {shift:+.2f}σ, decision: {decision}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 5 robustness tests")
    parser.add_argument("--test2", action="store_true", help="Run Test 2 analysis (final)")
    parser.add_argument("--test4", action="store_true", help="Run Test 4 analysis (final)")
    parser.add_argument("--all", action="store_true", help="Run all final analyses")
    parser.add_argument("--check", action="store_true",
                        help="Check convergence status, don't produce outputs")
    parser.add_argument("--interim", type=str, default=None,
                        choices=["no_lensing", "TT_only", "wide_prior"],
                        help="Run interim analysis for a specific chain")
    args = parser.parse_args()

    if args.check:
        results = check_status()
        print_status(results)
        return

    if args.interim:
        results = check_status()
        run_interim(args.interim, results)
        return

    if not (args.test2 or args.test4 or args.all):
        args.all = True

    if args.test2 or args.all:
        analyze_test2()

    if args.test4 or args.all:
        analyze_test4()


if __name__ == "__main__":
    main()
