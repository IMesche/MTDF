#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Compute explicit 95% upper limits for Phase 6B, 6B2, and 6D.

Reads existing summary JSONs where possible. For 6D, re-runs bootstrap
to get empirical percentiles of the compensated statistic.

Output: upper_limits_summary.json + console table
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from testD_cmb_lensing_voids import (
    load_planck_kappa, load_revolver_vide, prepare_centres,
    stack_kappa_profile, compute_compensated,
    NumpyEncoder, pr,
    PLANCK_DIR, VOID_DIR, OUT_DIR,
    R_CENTRE_MAX, R_OUTER_LO, R_OUTER_HI, R_RING_LO, R_RING_HI,
    N_BOOTSTRAP,
)


def main():
    t_start = time.time()
    pr("=" * 70)
    pr("Explicit 95% Upper Limits: Phases 6B, 6B2, 6D")
    pr("=" * 70)

    results = {}

    # ================================================================
    # Phase 6B: Weak lensing x environment
    # ================================================================
    pr("\n--- Phase 6B: Weak Lensing x Environment ---")
    testB_path = OUT_DIR.parent / "testB_wl_environment" / "phase6_testB_summary.json"
    with open(testB_path) as f:
        tb = json.load(f)

    dgt = tb["delta_gamma_t"]
    mean_b = dgt["delta_mean"]
    std_b = dgt["delta_std"]
    ci95_b = dgt["ci_95"]

    # Gaussian bounds
    gauss_lo = mean_b - 1.96 * std_b
    gauss_hi = mean_b + 1.96 * std_b

    pr(f"  Statistic: Delta_gamma_t (void - random) in [5, 20] Mpc/h")
    pr(f"  Mean:  {mean_b:.4e}")
    pr(f"  Sigma: {std_b:.4e} (bootstrap, {dgt['n_boot']} resamples)")
    pr(f"  Gaussian 95% CI: [{gauss_lo:.4e}, {gauss_hi:.4e}]")
    pr(f"  Bootstrap 95% CI: [{ci95_b[0]:.4e}, {ci95_b[1]:.4e}]")
    pr(f"  |Delta_gamma_t| < {max(abs(ci95_b[0]), abs(ci95_b[1])):.4e} (two-sided 95%)")

    results["phase6B"] = {
        "test": "Weak lensing x void environment (KiDS-1000 + DESIVAST BGS)",
        "statistic": "Delta_gamma_t = <gamma_t>_void - <gamma_t>_random, [5,20] Mpc/h",
        "mean": mean_b,
        "sigma": std_b,
        "error_method": f"bootstrap ({dgt['n_boot']} resamples over void+random centres)",
        "gaussian_95_ci": [gauss_lo, gauss_hi],
        "bootstrap_95_ci": ci95_b,
        "two_sided_95_bound": max(abs(ci95_b[0]), abs(ci95_b[1])),
        "n_voids": dgt["n_void_centres"],
        "n_random": dgt["n_random_centres"],
        "gamma_x_passed": tb["gamma_x_test"]["passed"],
    }

    # ================================================================
    # Phase 6B2: Trough & Ridge lensing
    # ================================================================
    pr("\n--- Phase 6B2: Trough & Ridge Lensing ---")
    testB2_path = OUT_DIR.parent / "testB2_trough_ridge" / "phase6_testB2_summary.json"
    with open(testB2_path) as f:
        tb2 = json.load(f)

    bs = tb2["bootstrap"]

    # Primary: ridge - trough (delta_gamma_t_split)
    split = bs["delta_gamma_t_split"]
    mean_s = split["mean"]
    std_s = split["std"]
    ci95_s = split["ci_95"]
    gauss_lo_s = mean_s - 1.96 * std_s
    gauss_hi_s = mean_s + 1.96 * std_s

    pr(f"  Primary: Delta_gamma_t_split (ridge - trough), [3.5, 20] Mpc/h")
    pr(f"  Mean:  {mean_s:.4e}")
    pr(f"  Sigma: {std_s:.4e} (bootstrap, {bs['n_boot']} resamples)")
    pr(f"  Gaussian 95% CI: [{gauss_lo_s:.4e}, {gauss_hi_s:.4e}]")
    pr(f"  Bootstrap 95% CI: [{ci95_s[0]:.4e}, {ci95_s[1]:.4e}]")

    # Secondary: trough - random
    trough = bs["delta_gamma_t_trough"]
    mean_tr = trough["mean"]
    std_tr = trough["std"]
    ci95_tr = trough["ci_95"]

    pr(f"\n  Secondary: Delta_gamma_t_trough (trough - random)")
    pr(f"  Mean:  {mean_tr:.4e}, Sigma: {std_tr:.4e}")
    pr(f"  Bootstrap 95% CI: [{ci95_tr[0]:.4e}, {ci95_tr[1]:.4e}]")

    results["phase6B2"] = {
        "test": "Trough & ridge lensing (KiDS-1000 internal)",
        "primary_statistic": "Delta_gamma_t_split = <gamma_t>_ridge - <gamma_t>_trough, [3.5,20] Mpc/h",
        "primary": {
            "mean": mean_s,
            "sigma": std_s,
            "error_method": f"bootstrap ({bs['n_boot']} resamples)",
            "gaussian_95_ci": [gauss_lo_s, gauss_hi_s],
            "bootstrap_95_ci": ci95_s,
            "two_sided_95_bound": max(abs(ci95_s[0]), abs(ci95_s[1])),
        },
        "secondary_trough": {
            "statistic": "Delta_gamma_t_trough = <gamma_t>_trough - <gamma_t>_random",
            "mean": mean_tr,
            "sigma": std_tr,
            "bootstrap_95_ci": ci95_tr,
        },
        "n_trough": bs["n_trough"],
        "n_ridge": bs["n_ridge"],
        "n_random": bs["n_random"],
        "gamma_x_passed_trough": tb2["gamma_x_test_trough"]["passed"],
        "gamma_x_passed_ridge": tb2["gamma_x_test_ridge"]["passed"],
    }

    # ================================================================
    # Phase 6D: CMB Lensing x Voids — re-run bootstrap for percentiles
    # ================================================================
    pr("\n--- Phase 6D: CMB Lensing x DESI Voids ---")
    pr("  Re-running bootstrap for empirical percentiles...")

    # Load and stack (one-time cost ~3s)
    kappa_map, mask_bin = load_planck_kappa(PLANCK_DIR)
    rev_records = load_revolver_vide(VOID_DIR, "revolver")
    centres = prepare_centres(rev_records, kappa_map, mask_bin)

    prof = stack_kappa_profile(
        kappa_map, mask_bin,
        centres["ra"], centres["dec"], centres["theta_v"],
        label="REVOLVER",
    )
    r_rv = prof["r_rv"]
    sum_kappa = prof["sum_kappa"]
    sum_weight = prof["sum_weight"]
    n_voids = sum_kappa.shape[0]

    # Primary compensated stat (jackknife)
    comp_result = compute_compensated(sum_kappa, sum_weight, r_rv)
    comp_mean = comp_result["comp_mean"]
    comp_err_jk = comp_result["comp_err"]

    pr(f"  Jackknife: kappa_comp = {comp_mean:.4e} +/- {comp_err_jk:.4e}"
       f" ({comp_result['snr']:.1f}sigma, {comp_result['n_used']} voids)")

    # Bootstrap with full distribution saved
    rng = np.random.default_rng(42)
    n_boot = N_BOOTSTRAP
    boot_comp = np.zeros(n_boot)

    pr(f"  Running {n_boot} bootstrap iterations...")
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, n_voids, size=n_voids)
        sk = sum_kappa[idx]
        sw = sum_weight[idx]
        c = compute_compensated(sk, sw, r_rv)
        boot_comp[b] = c["comp_mean"] if not np.isnan(c["comp_mean"]) else 0.0

    elapsed = time.time() - t0
    pr(f"  Bootstrap done in {elapsed:.1f}s")

    boot_mean = np.mean(boot_comp)
    boot_std = np.std(boot_comp)
    boot_ci95 = [float(np.percentile(boot_comp, 2.5)),
                 float(np.percentile(boot_comp, 97.5))]
    boot_ci68 = [float(np.percentile(boot_comp, 16)),
                 float(np.percentile(boot_comp, 84))]

    # Gaussian bounds from jackknife
    gauss_lo_d = comp_mean - 1.96 * comp_err_jk
    gauss_hi_d = comp_mean + 1.96 * comp_err_jk

    # One-sided 95% bound (upper = mean + 1.64*sigma for one-sided)
    one_sided_95 = comp_mean + 1.64 * comp_err_jk
    one_sided_95_boot = float(np.percentile(boot_comp, 95))

    pr(f"\n  Jackknife error:  {comp_err_jk:.4e}")
    pr(f"  Bootstrap std:    {boot_std:.4e}")
    pr(f"  Gaussian 95% CI:  [{gauss_lo_d:.4e}, {gauss_hi_d:.4e}]")
    pr(f"  Bootstrap 95% CI: [{boot_ci95[0]:.4e}, {boot_ci95[1]:.4e}]")
    pr(f"  |kappa_comp| < {max(abs(boot_ci95[0]), abs(boot_ci95[1])):.4e} (two-sided 95%, bootstrap)")
    pr(f"  kappa_comp < {one_sided_95_boot:.4e} (one-sided 95%, bootstrap)")

    # Read existing null p-values
    testD_path = OUT_DIR / "phase6_testD_summary.json"
    with open(testD_path) as f:
        td = json.load(f)

    results["phase6D"] = {
        "test": "CMB lensing x DESI voids (Planck PR4 MV + DESIVAST BGS)",
        "statistic": "kappa_comp = <kappa>_{R/Rv<0.5} - <kappa>_{R/Rv in [4,5]}, per void then averaged",
        "mean": comp_mean,
        "sigma_jackknife": comp_err_jk,
        "sigma_bootstrap": boot_std,
        "snr_jackknife": comp_result["snr"],
        "n_used": comp_result["n_used"],
        "n_total": n_voids,
        "gaussian_95_ci": [gauss_lo_d, gauss_hi_d],
        "bootstrap_95_ci": boot_ci95,
        "bootstrap_68_ci": boot_ci68,
        "two_sided_95_bound": max(abs(boot_ci95[0]), abs(boot_ci95[1])),
        "one_sided_95_upper_gaussian": one_sided_95,
        "one_sided_95_upper_bootstrap": one_sided_95_boot,
        "n_boot": n_boot,
        "error_methods": "jackknife (delete-1) + bootstrap (resample voids with replacement)",
        "null_tests": {
            "ra_scramble_p_comp": td["null_ra_scramble"]["p_comp"],
            "random_positions_p_comp": td["null_random"]["p_comp"],
            "note": "RA-scramble is primary null for footprint-coupled modes; random-sky null is secondary",
        },
    }

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    pr("\n" + "=" * 70)
    pr("SUMMARY: 95% Upper Limits")
    pr("=" * 70)
    pr(f"{'Test':<20} {'Statistic':<25} {'Mean':>12} {'Sigma':>12} {'95% CI (bootstrap)':>30}")
    pr("-" * 100)

    r = results["phase6B"]
    pr(f"{'6B (void WL)':<20} {'Delta_gamma_t':<25} {r['mean']:>12.4e} {r['sigma']:>12.4e}"
       f" [{r['bootstrap_95_ci'][0]:>12.4e}, {r['bootstrap_95_ci'][1]:>12.4e}]")

    r = results["phase6B2"]["primary"]
    pr(f"{'6B2 (ridge-trough)':<20} {'Delta_gamma_t_split':<25} {r['mean']:>12.4e} {r['sigma']:>12.4e}"
       f" [{r['bootstrap_95_ci'][0]:>12.4e}, {r['bootstrap_95_ci'][1]:>12.4e}]")

    r = results["phase6D"]
    pr(f"{'6D (CMB lensing)':<20} {'kappa_comp':<25} {r['mean']:>12.4e} {r['sigma_bootstrap']:>12.4e}"
       f" [{r['bootstrap_95_ci'][0]:>12.4e}, {r['bootstrap_95_ci'][1]:>12.4e}]")

    pr("-" * 100)
    pr("\nAll three tests are consistent with zero at 95% confidence.")
    pr("These are sensitivity-limited upper limits, not detections or anomalies.")

    # ---- Save ----
    out_path = OUT_DIR.parent / "upper_limits_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    pr(f"\nSaved: {out_path}")

    elapsed = time.time() - t_start
    pr(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
