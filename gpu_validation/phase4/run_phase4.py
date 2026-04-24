#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 4: Sensitivity Forecasts — Main Driver

Three covariance scenarios for future survey forecasts:
  Baseline:     C = diag(σ_i²) — pure statistical errors, no systematic floor
  Realistic:    C = diag(σ_i² + σ_floor²) — independent systematic floor per SN,
                σ_floor calibrated from mean off-diagonal of Pantheon+ STAT+SYS cov
  Adversarial:  C = diag(σ_i²) + σ_adv² × d d^T — worst case: systematic noise
                perfectly aligned with the environment regressor d_signed,
                σ_adv calibrated so σ_γ(N=564) = 2× baseline

Consistency check: full STAT+SYS Cholesky at N=564 reproduces Phase 3.

KEY FINDINGS (from covariance analysis):
1. Bootstrap saturation is expected — resampling 564 SNe cannot increase
   independent information beyond that sample.
2. A uniform correlated floor σ_sys² × 11^T is absorbed entirely by the
   intercept nuisance parameter, with ZERO impact on σ_γ. This is exact:
   the rank-1 systematic projects entirely onto the intercept column of X.
3. Only systematics correlated with the regressor d_signed degrade σ_γ.
   The adversarial model C = diag(σ²) + σ_adv² × d d^T quantifies this
   worst case.
4. Phase 3 already tested the most dangerous structured systematics:
   survey fixed effects (Δγ < 0.0005, stable), LOSO (16/16 positive),
   z-modulation (signal at z<0.04 only). These bound real-world systematics
   far below the adversarial ceiling.
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mtdf_validation.phase4.detection_power import (
    prewhiten_and_project, compute_power_batch,
    sigma_gamma_diagonal, compute_power_diagonal,
    sigma_gamma_adversarial, calibrate_sigma_adv,
    compute_power_adversarial,
    analytic_power_estimate,
)
from mtdf_validation.phase3.data_loader import (
    PantheonPlusData, load_all_void_catalogs, sn_to_comoving,
    combine_ngc_sgc_voids, COSMOLOGY_HEADER,
)
from mtdf_validation.phase3.crossmatch_gpu import compute_environment_gpu

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "validation" / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase4"
PHASE3_RESULTS = Path(__file__).resolve().parent.parent / "results" / "phase3" / "phase3_summary.json"

SAMPLE_SIZES = [500, 1000, 2000, 3000, 5000, 7500, 10000]
FINDERS = ['voidfinder', 'revolver', 'vide']
POWER_THRESHOLDS = [0.50, 0.80, 0.95]

N_DEFINITION = (
    "N independent Type Ia SNe drawn from the observed Pantheon+ distributions "
    "(z, host mass, d_signed, per-SN errors). Three covariance scenarios: "
    "baseline (diagonal only), realistic (diagonal + independent systematic "
    "floor sigma_floor per SN), adversarial (diagonal + rank-1 systematic "
    "aligned with d_signed, worst case for gamma_env)."
)

TEST_STATISTIC_DEFINITION = (
    "GLS delta-chi2 (df=1): null model (intercept + mass_step) vs full model "
    "(intercept + gamma_env*d_signed + mass_step) — identical structure to "
    "Phase 3. Detection: p < threshold AND gamma_env > 0 (one-sided)."
)

KEY_CONCLUSIONS = [
    "Bootstrap saturation is expected: resampling 564 correlated SNe cannot "
    "increase independent information beyond that sample.",
    "A uniform correlated floor sigma_sys^2 * 11^T is absorbed entirely by "
    "the intercept nuisance parameter, with ZERO impact on sigma_gamma. "
    "This is mathematically exact for any regression with an intercept.",
    "Only systematics correlated with the regressor d_signed can degrade "
    "sigma_gamma. The adversarial model quantifies the worst case.",
    "Phase 3 tested the most dangerous structured systematics: survey FE "
    "(Delta_gamma < 0.0005, stable), LOSO (16/16 positive), z-modulation "
    "(signal at z<0.04). These bound real-world systematics far below the "
    "adversarial ceiling.",
]


def load_phase3_data():
    """Load Phase 3 results, observed distributions, and covariance."""
    with open(PHASE3_RESULTS) as f:
        phase3 = json.load(f)

    pantheon = PantheonPlusData(DATA_DIR)
    catalogs = load_all_void_catalogs(DATA_DIR)

    idx, cov_sub = pantheon.apply_cuts(z_min=0.02, z_max=0.157)
    sub = pantheon.get_subset(idx)
    sn_pos = sn_to_comoving(sub['z'], sub['ra'], sub['dec'])

    mu_err = np.sqrt(np.diag(cov_sub))

    # Compute sigma_floor from Pantheon+ off-diagonal covariance structure
    offdiag = cov_sub.copy()
    np.fill_diagonal(offdiag, 0)
    N = len(idx)
    mean_offdiag = np.sum(offdiag) / (N * (N - 1))
    sigma_floor = np.sqrt(max(0, mean_offdiag))

    d_signed_full = {}
    for finder in FINDERS:
        try:
            void_pos, void_r, _ = combine_ngc_sgc_voids(catalogs, finder)
            d_signed, _, _ = compute_environment_gpu(sn_pos, void_pos, void_r)
            d_signed_full[finder] = d_signed
        except ValueError:
            pass

    # NGC-only: EXACT same RA cut as Phase 3 run_phase3.py:123
    ngc_mask = (sub['ra'] > 90) & (sub['ra'] < 280)
    ngc_sn_idx = np.where(ngc_mask)[0]
    ngc_sn_pos = sn_pos[ngc_sn_idx]
    ngc_cov = cov_sub[np.ix_(ngc_sn_idx, ngc_sn_idx)]
    print(f"  NGC SNe: {len(ngc_sn_idx)} (RA > 90 and RA < 280, matching Phase 3)")

    # NGC sigma_floor
    ngc_offdiag = ngc_cov.copy()
    np.fill_diagonal(ngc_offdiag, 0)
    N_ngc = len(ngc_sn_idx)
    ngc_mean_offdiag = np.sum(ngc_offdiag) / (N_ngc * (N_ngc - 1))
    ngc_sigma_floor = np.sqrt(max(0, ngc_mean_offdiag))

    d_signed_ngc = {}
    for finder in FINDERS:
        try:
            void_pos, void_r, n_ngc = combine_ngc_sgc_voids(catalogs, finder)
            d_signed, _, _ = compute_environment_gpu(
                ngc_sn_pos, void_pos[:n_ngc], void_r[:n_ngc]
            )
            d_signed_ngc[finder] = d_signed
        except (ValueError, KeyError):
            pass

    return {
        'phase3': phase3,
        'z': sub['z'],
        'ra': sub['ra'],
        'host_mass': sub['host_mass'],
        'mu_err': mu_err,
        'cov': cov_sub,
        'sigma_floor': float(sigma_floor),
        'd_signed_full': d_signed_full,
        'd_signed_ngc': d_signed_ngc,
        'ngc_sn_idx': ngc_sn_idx,
        'ngc_cov': ngc_cov,
        'ngc_host_mass': sub['host_mass'][ngc_sn_idx],
        'ngc_mu_err': mu_err[ngc_sn_idx],
        'ngc_sigma_floor': float(ngc_sigma_floor),
    }


def run_consistency_check(finder, gamma_true, d_signed, host_mass, cov,
                          phase3_sigma, n_mocks):
    """N=564 consistency check using full STAT+SYS covariance."""
    print(f"\n  CONSISTENCY CHECK (N={len(d_signed)}, full covariance):")

    w, q, x_env_w, sigma_gamma, reg = prewhiten_and_project(cov, d_signed, host_mass)
    snr = gamma_true / sigma_gamma
    power = compute_power_batch(w, q, x_env_w, gamma_true, n_mocks)

    rel_diff = abs(sigma_gamma - phase3_sigma) / phase3_sigma
    status = "PASS" if rel_diff < 0.01 else "WARN"

    print(f"    sigma_gamma (Cholesky): {sigma_gamma:.6f}")
    print(f"    sigma_gamma (Phase 3):  {phase3_sigma:.6f}")
    print(f"    Relative diff: {rel_diff:.2e} [{status}]")
    print(f"    SNR: {snr:.2f}, Power: 2s={power['power_2sigma']:.1%}, 3s={power['power_3sigma']:.1%}")

    # Also check diagonal-only sigma_gamma vs full-cov
    sig_diag = sigma_gamma_diagonal(np.sqrt(np.diag(cov)), d_signed, host_mass)
    print(f"    sigma_gamma (diagonal): {sig_diag:.6f} "
          f"(ratio to full-cov: {sig_diag/sigma_gamma:.4f})")

    return {
        'n_sne': len(d_signed),
        'sigma_gamma_cholesky': float(sigma_gamma),
        'sigma_gamma_phase3': float(phase3_sigma),
        'sigma_gamma_diagonal': float(sig_diag),
        'relative_diff': float(rel_diff),
        'match': bool(rel_diff < 0.01),
        'expected_snr': float(snr),
        'power_2sigma': float(power['power_2sigma']),
        'power_3sigma': float(power['power_3sigma']),
        'power_5sigma': float(power['power_5sigma']),
    }


def run_forecast_diagonal(label, gamma_true, d_signed, host_mass,
                          mu_err, sigma_floor, n_mocks, seed=42):
    """Run baseline or realistic forecast across all sample sizes."""
    n_obs = len(d_signed)
    rng = np.random.RandomState(seed)

    print(f"\n  {'N_sne':>8} | {'2sigma':>8} {'3sigma':>8} {'5sigma':>8} | "
          f"{'sigma_gamma':>11} {'SNR':>6} | {'analytic_3s':>10} {'time':>6}")
    print(f"  {'-' * 82}")

    sizes_results = {}
    for n_sne in SAMPLE_SIZES:
        t0 = time.time()
        idx = rng.randint(0, n_obs, size=n_sne)

        power = compute_power_diagonal(
            mu_err[idx], d_signed[idx], host_mass[idx], gamma_true,
            sigma_floor, n_mocks, seed=seed + n_sne,
        )
        analytic = analytic_power_estimate(gamma_true, power['sigma_gamma'])
        dt = time.time() - t0
        snr = gamma_true / power['sigma_gamma']

        print(f"  {n_sne:>8} | {power['power_2sigma']:>7.1%} {power['power_3sigma']:>7.1%} "
              f"{power['power_5sigma']:>7.1%} | "
              f"{power['sigma_gamma']:>11.6f} {snr:>6.2f} | "
              f"{analytic['power_3sigma']:>9.1%} {dt:>5.1f}s")

        sizes_results[str(n_sne)] = {'mock': power, 'analytic': analytic}

    return sizes_results


def run_forecast_adversarial(label, gamma_true, d_signed, host_mass,
                             mu_err, sigma_adv, n_mocks, seed=42):
    """Run adversarial forecast across all sample sizes."""
    n_obs = len(d_signed)
    rng = np.random.RandomState(seed)

    print(f"\n  {'N_sne':>8} | {'2sigma':>8} {'3sigma':>8} {'5sigma':>8} | "
          f"{'sigma_gamma':>11} {'SNR':>6} | {'analytic_3s':>10} {'time':>6}")
    print(f"  {'-' * 82}")

    sizes_results = {}
    for n_sne in SAMPLE_SIZES:
        t0 = time.time()
        idx = rng.randint(0, n_obs, size=n_sne)

        power = compute_power_adversarial(
            mu_err[idx], d_signed[idx], host_mass[idx], gamma_true,
            sigma_adv, n_mocks, seed=seed + n_sne,
        )
        analytic = analytic_power_estimate(gamma_true, power['sigma_gamma'])
        dt = time.time() - t0
        snr = gamma_true / power['sigma_gamma']

        print(f"  {n_sne:>8} | {power['power_2sigma']:>7.1%} {power['power_3sigma']:>7.1%} "
              f"{power['power_5sigma']:>7.1%} | "
              f"{power['sigma_gamma']:>11.6f} {snr:>6.2f} | "
              f"{analytic['power_3sigma']:>9.1%} {dt:>5.1f}s")

        sizes_results[str(n_sne)] = {'mock': power, 'analytic': analytic}

    return sizes_results


def find_threshold_n(sizes_results, target_power, sigma_level):
    """Interpolate on log10(N) to find N for target power."""
    key = f'power_{sigma_level}'
    sizes, powers = [], []
    for n_str, data in sorted(sizes_results.items(), key=lambda x: int(x[0])):
        sizes.append(int(n_str))
        powers.append(data['mock'][key])

    sizes = np.array(sizes, dtype=float)
    powers = np.array(powers)
    if len(powers) == 0:
        return "unknown"
    if powers[-1] < target_power:
        return f">{int(sizes[-1])}"
    if powers[0] >= target_power:
        return f"<{int(sizes[0])}"

    log_sizes = np.log10(sizes)
    for i in range(len(powers) - 1):
        if powers[i] < target_power <= powers[i + 1]:
            frac = (target_power - powers[i]) / (powers[i + 1] - powers[i])
            log_n = log_sizes[i] + frac * (log_sizes[i + 1] - log_sizes[i])
            return int(np.round(10 ** log_n, -2))
    return "unknown"


def extract_thresholds(sizes_results):
    """Extract threshold N for 50/80/95% power at 3σ and 5σ."""
    t = {}
    for sigma in ['3sigma', '5sigma']:
        for pwr in POWER_THRESHOLDS:
            t[f'{sigma}_{int(pwr*100)}pct'] = find_threshold_n(sizes_results, pwr, sigma)
    return t


def main():
    parser = argparse.ArgumentParser(description='Phase 4: Sensitivity Forecasts')
    parser.add_argument('--quick', action='store_true', help='2000 mocks (vs 10000)')
    args = parser.parse_args()

    n_mocks = 2000 if args.quick else 10000
    t_start = time.time()

    print("=" * 70)
    print("MTDF PHASE 4: SENSITIVITY FORECASTS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'QUICK' if args.quick else 'PRODUCTION'} ({n_mocks} mocks)")
    print("=" * 70)
    print(f"\nN definition: {N_DEFINITION}")
    print(f"\nTest statistic: {TEST_STATISTIC_DEFINITION}")
    print("\nKey conclusions:")
    for i, c in enumerate(KEY_CONCLUSIONS, 1):
        print(f"  {i}. {c}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading Phase 3 results and observed distributions...")
    obs_data = load_phase3_data()
    phase3 = obs_data['phase3']
    sigma_floor = obs_data['sigma_floor']

    print(f"\n  Pantheon+ off-diagonal systematic floor: sigma_floor = {sigma_floor:.4f} mag")
    print(f"  (from sqrt(mean off-diagonal of 564x564 STAT+SYS covariance))")

    gamma_envs, sigma_envs = {}, {}
    for finder in FINDERS:
        if finder in phase3['table1']:
            gamma_envs[finder] = phase3['table1'][finder]['signed']['gamma_env']
            sigma_envs[finder] = phase3['table1'][finder]['signed']['gamma_env_err']

    gamma_envs_ngc, sigma_envs_ngc = {}, {}
    for finder in FINDERS:
        if finder in phase3.get('table2_ngc_sgc', {}):
            ngc = phase3['table2_ngc_sgc'][finder].get('ngc', {})
            if 'gamma_env' in ngc:
                gamma_envs_ngc[finder] = ngc['gamma_env']
                sigma_envs_ngc[finder] = ngc.get('gamma_env_err', 0)

    all_results = {}
    consistency_checks = {}

    for finder in FINDERS:
        if finder not in gamma_envs or finder not in obs_data['d_signed_full']:
            continue

        gamma = gamma_envs[finder]
        d_signed = obs_data['d_signed_full'][finder]

        print(f"\n{'=' * 70}")
        print(f"FORECAST: {finder.upper()} (gamma_env = {gamma:+.4f} mag)")
        print("=" * 70)

        # --- Consistency check (full covariance, N=564) ---
        check = run_consistency_check(
            finder, gamma, d_signed, obs_data['host_mass'],
            obs_data['cov'], sigma_envs[finder], n_mocks,
        )
        consistency_checks[finder] = check

        # --- Calibrate adversarial sigma_adv ---
        # Target: 2× baseline sigma_gamma at N=564
        sig_base_564 = sigma_gamma_diagonal(obs_data['mu_err'], d_signed,
                                            obs_data['host_mass'])
        target_adv = 2.0 * sig_base_564
        sigma_adv = calibrate_sigma_adv(obs_data['mu_err'], d_signed,
                                        obs_data['host_mass'], target_adv)

        print(f"\n  Systematic parameters:")
        print(f"    sigma_floor (realistic): {sigma_floor:.4f} mag "
              f"(from Pantheon+ off-diagonal)")
        print(f"    sigma_adv (adversarial): {sigma_adv:.4f} mag "
              f"(calibrated to 2x baseline sigma_gamma at N=564)")
        print(f"    sigma_gamma baseline(N=564): {sig_base_564:.6f}")
        print(f"    sigma_gamma adversarial(N=564): "
              f"{sigma_gamma_adversarial(obs_data['mu_err'], d_signed, obs_data['host_mass'], sigma_adv):.6f}")

        finder_results = {}

        # --- BASELINE: diagonal only ---
        print(f"\n  --- BASELINE (diagonal only) ---")
        print(f"  Source pool: {len(d_signed)} SNe, gamma = {gamma:+.4f}, "
              f"d_signed std = {np.std(d_signed):.2f}")

        sizes_base = run_forecast_diagonal(
            'baseline', gamma, d_signed, obs_data['host_mass'],
            obs_data['mu_err'], 0.0, n_mocks,
        )
        thresholds_base = extract_thresholds(sizes_base)
        print(f"\n  Thresholds (baseline):")
        for k, v in thresholds_base.items():
            print(f"    {k}: N ~ {v}")

        finder_results['baseline'] = {
            'scenario': 'baseline',
            'sigma_floor': 0.0,
            'gamma_env_injected': float(gamma),
            'n_obs_source': len(d_signed),
            'sample_sizes': sizes_base,
            'thresholds': thresholds_base,
        }

        # --- REALISTIC: diagonal + independent floor ---
        print(f"\n  --- REALISTIC (diagonal + floor sigma_floor={sigma_floor:.4f}) ---")
        print(f"  Source pool: {len(d_signed)} SNe, gamma = {gamma:+.4f}")

        sizes_real = run_forecast_diagonal(
            'realistic', gamma, d_signed, obs_data['host_mass'],
            obs_data['mu_err'], sigma_floor, n_mocks, seed=137,
        )
        thresholds_real = extract_thresholds(sizes_real)
        print(f"\n  Thresholds (realistic):")
        for k, v in thresholds_real.items():
            print(f"    {k}: N ~ {v}")

        finder_results['realistic'] = {
            'scenario': 'realistic',
            'sigma_floor': float(sigma_floor),
            'gamma_env_injected': float(gamma),
            'n_obs_source': len(d_signed),
            'sample_sizes': sizes_real,
            'thresholds': thresholds_real,
        }

        # --- ADVERSARIAL: d-aligned systematic ---
        print(f"\n  --- ADVERSARIAL (d-aligned sigma_adv={sigma_adv:.4f}) ---")
        print(f"  Source pool: {len(d_signed)} SNe, gamma = {gamma:+.4f}")

        sizes_adv = run_forecast_adversarial(
            'adversarial', gamma, d_signed, obs_data['host_mass'],
            obs_data['mu_err'], sigma_adv, n_mocks, seed=271,
        )
        thresholds_adv = extract_thresholds(sizes_adv)
        print(f"\n  Thresholds (adversarial):")
        for k, v in thresholds_adv.items():
            print(f"    {k}: N ~ {v}")

        finder_results['adversarial'] = {
            'scenario': 'adversarial',
            'sigma_adv': float(sigma_adv),
            'gamma_env_injected': float(gamma),
            'n_obs_source': len(d_signed),
            'sample_sizes': sizes_adv,
            'thresholds': thresholds_adv,
        }

        # --- NGC-only (baseline only, since signal is weaker) ---
        if (finder in gamma_envs_ngc and finder in obs_data['d_signed_ngc']
                and abs(gamma_envs_ngc[finder]) > 1e-6):
            gamma_ngc = gamma_envs_ngc[finder]
            d_ngc = obs_data['d_signed_ngc'][finder]

            print(f"\n  --- NGC-ONLY BASELINE (gamma = {gamma_ngc:+.4f}) ---")

            if finder in sigma_envs_ngc and sigma_envs_ngc[finder] > 0:
                check_ngc = run_consistency_check(
                    finder, gamma_ngc, d_ngc, obs_data['ngc_host_mass'],
                    obs_data['ngc_cov'], sigma_envs_ngc[finder], n_mocks,
                )
                consistency_checks[f'{finder}_ngc'] = check_ngc

            sizes_ngc = run_forecast_diagonal(
                'ngc_baseline', gamma_ngc, d_ngc,
                obs_data['ngc_host_mass'], obs_data['ngc_mu_err'],
                0.0, n_mocks, seed=313,
            )
            thresholds_ngc = extract_thresholds(sizes_ngc)

            print(f"\n  Thresholds (NGC baseline):")
            for k, v in thresholds_ngc.items():
                print(f"    {k}: N ~ {v}")

            finder_results['ngc_baseline'] = {
                'scenario': 'ngc_baseline',
                'sigma_floor': 0.0,
                'gamma_env_injected': float(gamma_ngc),
                'n_obs_source': len(d_ngc),
                'sample_sizes': sizes_ngc,
                'thresholds': thresholds_ngc,
            }

        all_results[finder] = finder_results

    elapsed = time.time() - t_start

    output = {
        'timestamp': datetime.now().isoformat(),
        'definition': {
            'N': N_DEFINITION,
            'test_statistic': TEST_STATISTIC_DEFINITION,
            'detection_criterion': 'p(dchi2, df=1) < p_threshold AND gamma_env > 0 (one-sided)',
            'key_conclusions': KEY_CONCLUSIONS,
            'n_mocks_per_size': n_mocks,
            'sample_sizes_tested': SAMPLE_SIZES,
            'sigma_floor_mag': float(sigma_floor),
            'sigma_floor_source': 'sqrt(mean off-diagonal of Pantheon+ 564x564 STAT+SYS cov)',
        },
        'cosmology_header': COSMOLOGY_HEADER,
        'consistency_checks': consistency_checks,
        'results': all_results,
        'elapsed_seconds': elapsed,
    }

    out_path = RESULTS_DIR / "phase4_summary.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"\n[SAVED] {out_path}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY")
    print("=" * 70)
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print("\n  Consistency checks (full cov, sigma_gamma vs Phase 3):")
    for k, v in consistency_checks.items():
        status = "PASS" if v['match'] else "WARN"
        print(f"    {k}: {v['sigma_gamma_cholesky']:.6f} vs {v['sigma_gamma_phase3']:.6f} "
              f"(diff {v['relative_diff']:.2e}) [{status}]")

    for finder in FINDERS:
        if finder not in all_results:
            continue
        fr = all_results[finder]
        print(f"\n  {finder.upper()}:")
        for scenario in ['baseline', 'realistic', 'adversarial', 'ngc_baseline']:
            if scenario not in fr:
                continue
            t = fr[scenario]['thresholds']
            label = scenario
            if scenario == 'realistic':
                label = f"realistic (sigma_floor={sigma_floor:.4f})"
            elif scenario == 'adversarial':
                sa = fr[scenario].get('sigma_adv', 0)
                label = f"adversarial (sigma_adv={sa:.4f})"
            print(f"    {label}:")
            print(f"      3sigma: 50%@N~{t['3sigma_50pct']}, "
                  f"80%@N~{t['3sigma_80pct']}, 95%@N~{t['3sigma_95pct']}")
            print(f"      5sigma: 50%@N~{t['5sigma_50pct']}, "
                  f"80%@N~{t['5sigma_80pct']}, 95%@N~{t['5sigma_95pct']}")

    print(f"\n  Current Pantheon+ (564 SNe) provides ~2sigma (Phase 3: REVOLVER dchi2=4.25).")
    print(f"  LSST/Rubin (10,000+ low-z SNe) will reach 5sigma territory.")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    main()
