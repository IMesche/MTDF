#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6D v2 Freeze Checks — annulus, mask threshold, profile-based compensated.

Three quick sensitivity checks before locking Test D:
  1. Annulus sensitivity: compensated stat with outer annuli [3,4], [4,5], [3.5,5] Rv
  2. Mask threshold: require unmasked fraction >= 50%, 60%, 70% in centre+outer
  3. Profile-based compensated: use stacked profile (all 1992 voids) directly

Usage:
  python mtdf_validation/phase6/testD_freeze_checks.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add parent so we can import the main module's functions
sys.path.insert(0, str(Path(__file__).resolve().parent))
from testD_cmb_lensing_voids import (
    load_planck_kappa, load_revolver_vide, prepare_centres,
    stack_kappa_profile, compute_compensated, compute_headline,
    NumpyEncoder, pr,
    PLANCK_DIR, VOID_DIR, OUT_DIR,
    NSIDE, LMAX, FWHM_DEG, R_MAX_RV, N_RADIAL_BINS,
    R_CENTRE_MAX, R_OUTER_LO, R_OUTER_HI,
    MIN_PIX_CENTRE, MIN_PIX_OUTER,
)


def compensated_with_annulus(sum_kappa, sum_weight, r_rv,
                             outer_lo, outer_hi, min_pix_c=5, min_pix_o=10):
    """Compensated stat with arbitrary outer annulus."""
    n_voids = sum_kappa.shape[0]
    c_bins = r_rv < R_CENTRE_MAX
    o_bins = (r_rv >= outer_lo) & (r_rv < outer_hi)

    comp_values = []
    for i in range(n_voids):
        wc = sum_weight[i, c_bins].sum()
        wo = sum_weight[i, o_bins].sum()
        if wc < min_pix_c or wo < min_pix_o:
            continue
        kc_i = sum_kappa[i, c_bins].sum() / wc
        ko_i = sum_kappa[i, o_bins].sum() / wo
        comp_values.append(kc_i - ko_i)

    comp = np.array(comp_values)
    n_used = len(comp)
    if n_used == 0:
        return dict(comp_mean=np.nan, comp_err=np.nan, snr=0.0, n_used=0)

    comp_mean = comp.mean()
    total = comp.sum()
    jk = (total - comp) / (n_used - 1)
    comp_err = np.sqrt((n_used - 1.0) / n_used * np.sum((jk - comp_mean)**2))
    snr = abs(comp_mean) / comp_err if comp_err > 0 else 0.0

    return dict(comp_mean=float(comp_mean), comp_err=float(comp_err),
                snr=float(snr), n_used=int(n_used))


def compensated_mask_threshold(sum_kappa, sum_weight, r_rv,
                               frac_threshold, n_bins):
    """Compensated stat requiring unmasked fraction >= threshold in centre+outer."""
    n_voids = sum_kappa.shape[0]
    c_bins = r_rv < R_CENTRE_MAX
    o_bins = (r_rv >= R_OUTER_LO) & (r_rv < R_OUTER_HI)

    # Estimate max possible pixels per bin from median across all voids
    # Use the number of radial bins and approximate pixel count
    dr = R_MAX_RV / n_bins
    bin_edges = np.linspace(0, R_MAX_RV, n_bins + 1)

    comp_values = []
    for i in range(n_voids):
        wc = sum_weight[i, c_bins].sum()
        wo = sum_weight[i, o_bins].sum()
        # Skip if no pixels at all
        if wc == 0 or wo == 0:
            continue
        # For mask threshold: estimate expected pixels from all-void average
        # Use the total weight across ALL voids as reference for "full" coverage
        # Actually, we just need to compare per-void weight to some reference.
        # Simpler: collect per-void centre and outer weights, then filter
        comp_values.append((i, wc, wo))

    if not comp_values:
        return dict(comp_mean=np.nan, comp_err=np.nan, snr=0.0, n_used=0)

    indices = np.array([x[0] for x in comp_values])
    wc_arr = np.array([x[1] for x in comp_values])
    wo_arr = np.array([x[2] for x in comp_values])

    # Reference: max observed per void (proxy for fully unmasked)
    wc_max = np.percentile(wc_arr, 95)
    wo_max = np.percentile(wo_arr, 95)

    # Filter by fraction
    frac_c = wc_arr / wc_max
    frac_o = wo_arr / wo_max
    keep = (frac_c >= frac_threshold) & (frac_o >= frac_threshold)

    kept_idx = indices[keep]
    n_used = len(kept_idx)
    if n_used == 0:
        return dict(comp_mean=np.nan, comp_err=np.nan, snr=0.0, n_used=0,
                    frac_threshold=frac_threshold)

    # Compute per-void compensated for kept voids
    comp = np.zeros(n_used)
    for j, i in enumerate(kept_idx):
        kc_i = sum_kappa[i, c_bins].sum() / sum_weight[i, c_bins].sum()
        ko_i = sum_kappa[i, o_bins].sum() / sum_weight[i, o_bins].sum()
        comp[j] = kc_i - ko_i

    comp_mean = comp.mean()
    total = comp.sum()
    jk = (total - comp) / (n_used - 1)
    comp_err = np.sqrt((n_used - 1.0) / n_used * np.sum((jk - comp_mean)**2))
    snr = abs(comp_mean) / comp_err if comp_err > 0 else 0.0

    return dict(comp_mean=float(comp_mean), comp_err=float(comp_err),
                snr=float(snr), n_used=int(n_used),
                frac_threshold=frac_threshold)


def profile_based_compensated(profile, r_rv):
    """Compensated stat from the stacked profile directly (all voids).

    mean(kappa[R/Rv<0.5]) - mean(kappa[4<R/Rv<5]) using pixel-weighted
    averages from the stacked profile.
    """
    k = profile["kappa"]
    npx = profile["n_pixels"]
    ke = profile["kappa_err"]

    c_mask = r_rv < R_CENTRE_MAX
    o_mask = (r_rv >= R_OUTER_LO) & (r_rv < R_OUTER_HI)

    if not c_mask.any() or not o_mask.any():
        return dict(comp_mean=np.nan, comp_err=np.nan, snr=0.0)

    wc = npx[c_mask].astype(float)
    wo = npx[o_mask].astype(float)

    kc = np.average(k[c_mask], weights=wc)
    ko = np.average(k[o_mask], weights=wo)

    # Error propagation from jackknife errors
    kc_err = np.sqrt(np.average(ke[c_mask]**2, weights=wc))
    ko_err = np.sqrt(np.average(ke[o_mask]**2, weights=wo))

    comp_mean = kc - ko
    comp_err = np.sqrt(kc_err**2 + ko_err**2)
    snr = abs(comp_mean) / comp_err if comp_err > 0 else 0.0

    return dict(
        comp_mean=float(comp_mean), comp_err=float(comp_err),
        snr=float(snr), n_voids=int(profile["n_voids"]),
        kappa_centre=float(kc), kappa_centre_err=float(kc_err),
        kappa_outer=float(ko), kappa_outer_err=float(ko_err),
    )


def main():
    t_start = time.time()

    pr("=" * 70)
    pr("Phase 6D v2 Freeze Checks")
    pr("=" * 70)

    # ---- Load Planck kappa ----
    pr("\n[1] Loading Planck PR4 kappa map...")
    kappa_map, mask_bin = load_planck_kappa(PLANCK_DIR)

    # ---- Load REVOLVER voids ----
    pr("\n[2] Loading REVOLVER voids...")
    rev_records = load_revolver_vide(VOID_DIR, "revolver")
    centres = prepare_centres(rev_records, kappa_map, mask_bin)
    pr(f"    {centres['n_total']} REVOLVER voids loaded")

    # ---- Single full stack (reuse for all checks) ----
    pr("\n[3] Stacking all REVOLVER voids...")
    prof = stack_kappa_profile(
        kappa_map, mask_bin,
        centres["ra"], centres["dec"], centres["theta_v"],
        label="REVOLVER",
    )
    r_rv = prof["r_rv"]

    results = {}

    # ================================================================
    # CHECK 1: Annulus sensitivity
    # ================================================================
    pr("\n" + "=" * 70)
    pr("CHECK 1: Annulus sensitivity")
    pr("=" * 70)

    annulus_configs = [
        ("3.0-4.0 Rv", 3.0, 4.0),
        ("4.0-5.0 Rv", 4.0, 5.0),   # baseline
        ("3.5-5.0 Rv", 3.5, 5.0),
    ]

    annulus_results = {}
    for label, lo, hi in annulus_configs:
        res = compensated_with_annulus(
            prof["sum_kappa"], prof["sum_weight"], r_rv,
            outer_lo=lo, outer_hi=hi,
        )
        annulus_results[label] = res
        tag = " (baseline)" if lo == 4.0 and hi == 5.0 else ""
        pr(f"  {label}{tag}: comp = {res['comp_mean']:.4e} +/- {res['comp_err']:.4e}"
           f"  ({res['snr']:.1f}sigma, {res['n_used']} voids)")

    results["annulus_sensitivity"] = annulus_results

    # ================================================================
    # CHECK 2: Mask threshold sensitivity
    # ================================================================
    pr("\n" + "=" * 70)
    pr("CHECK 2: Mask threshold sensitivity")
    pr("=" * 70)

    thresholds = [0.50, 0.60, 0.70]
    mask_results = {}
    for thr in thresholds:
        res = compensated_mask_threshold(
            prof["sum_kappa"], prof["sum_weight"], r_rv,
            frac_threshold=thr, n_bins=N_RADIAL_BINS,
        )
        mask_results[f"{int(thr*100)}%"] = res
        pr(f"  >= {int(thr*100)}%: comp = {res['comp_mean']:.4e} +/- {res['comp_err']:.4e}"
           f"  ({res['snr']:.1f}sigma, {res['n_used']} voids)")

    results["mask_threshold_sensitivity"] = mask_results

    # ================================================================
    # CHECK 3: Profile-based compensated
    # ================================================================
    pr("\n" + "=" * 70)
    pr("CHECK 3: Profile-based compensated (stacked, all voids)")
    pr("=" * 70)

    prof_comp = profile_based_compensated(prof, r_rv)
    results["profile_based_compensated"] = prof_comp

    pr(f"  kappa_centre (R/Rv<0.5): {prof_comp['kappa_centre']:.4e} +/- {prof_comp['kappa_centre_err']:.4e}")
    pr(f"  kappa_outer  (4-5 Rv):   {prof_comp['kappa_outer']:.4e} +/- {prof_comp['kappa_outer_err']:.4e}")
    pr(f"  compensated:             {prof_comp['comp_mean']:.4e} +/- {prof_comp['comp_err']:.4e}"
       f"  ({prof_comp['snr']:.1f}sigma, {prof_comp['n_voids']} voids)")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    pr("\n" + "=" * 70)
    pr("SUMMARY: All compensated statistics")
    pr("=" * 70)
    pr(f"{'Check':<35} {'comp_mean':>12} {'comp_err':>12} {'S/N':>6} {'N_voids':>8}")
    pr("-" * 75)

    for label, lo, hi in annulus_configs:
        r = annulus_results[label]
        tag = " *" if lo == 4.0 and hi == 5.0 else ""
        pr(f"Annulus {label:<27} {r['comp_mean']:>12.4e} {r['comp_err']:>12.4e} {r['snr']:>6.1f} {r['n_used']:>8}{tag}")

    for thr in thresholds:
        r = mask_results[f"{int(thr*100)}%"]
        pr(f"Mask >= {int(thr*100)}%{'':<25} {r['comp_mean']:>12.4e} {r['comp_err']:>12.4e} {r['snr']:>6.1f} {r['n_used']:>8}")

    r = prof_comp
    pr(f"Profile-based (all voids){'':<11} {r['comp_mean']:>12.4e} {r['comp_err']:>12.4e} {r['snr']:>6.1f} {r['n_voids']:>8}")

    pr("-" * 75)
    pr("* = baseline from v2 run")

    # ---- All consistent with zero? ----
    all_snr = []
    for label in annulus_results:
        all_snr.append(annulus_results[label]["snr"])
    for thr_key in mask_results:
        all_snr.append(mask_results[thr_key]["snr"])
    all_snr.append(prof_comp["snr"])

    max_snr = max(all_snr)
    pr(f"\nMax S/N across all checks: {max_snr:.2f}")
    if max_snr < 2.0:
        pr("ALL CHECKS CONSISTENT WITH ZERO. Phase 6D is locked.")
    else:
        pr(f"WARNING: max S/N = {max_snr:.2f} >= 2.0. Investigate before locking.")

    # ---- Save JSON ----
    out_path = OUT_DIR / "testD_freeze_checks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    pr(f"\nSaved: {out_path}")

    elapsed = time.time() - t_start
    pr(f"\nTotal elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
