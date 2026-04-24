# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""MCMC chain monitor — reads live chain files and reports diagnostics.

Usage:
    python -m mtdf_validation.phase5_plik.monitor_mcmc [--watch SECONDS]

Reports per-block R-1, acceptance rate, ESS for key parameters.
No restart needed — reads chain .txt files directly.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'phase5'

# Parameter blocks
COSMO_LCDM = ['logA', 'n_s', 'theta_s_100', 'omega_b', 'omega_cdm', 'tau_reio']
COSMO_MTDF = COSMO_LCDM + ['mtdf_k_f']
DERIVED_KEY = ['H0', 'sigma8', 'Omega_m']

NUISANCE = [
    'A_planck', 'calib_100T', 'calib_217T',
    'A_cib_217', 'xi_sz_cib', 'A_sz', 'ksz_norm',
    'gal545_A_100', 'gal545_A_143', 'gal545_A_143_217', 'gal545_A_217',
    'ps_A_100_100', 'ps_A_143_143', 'ps_A_143_217', 'ps_A_217_217',
    'galf_TE_A_100', 'galf_TE_A_100_143', 'galf_TE_A_100_217',
    'galf_TE_A_143', 'galf_TE_A_143_217', 'galf_TE_A_217',
]

ESS_PARAMS_LCDM = ['H0', 'sigma8']
ESS_PARAMS_MTDF = ['H0', 'sigma8', 'mtdf_k_f']


def load_chain(prefix):
    """Load chain from cobaya output files. Returns (columns, data, checkpoint)."""
    chain_file = RESULTS_DIR / f'{prefix}.1.txt'
    if not chain_file.exists():
        return None, None, None

    # Read header for column names
    with open(chain_file) as f:
        header = f.readline().strip().lstrip('#').split()

    # Load data (skip header)
    data = np.loadtxt(chain_file, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Load checkpoint
    ckpt = None
    ckpt_file = RESULTS_DIR / f'{prefix}.checkpoint'
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = yaml.safe_load(f)

    return header, data, ckpt


def gelman_rubin_split(samples, weights=None):
    """Compute R-1 by splitting a single weighted chain in half.

    Returns R-1 (scalar) for a 1D array of samples.
    """
    n = len(samples)
    if n < 20:
        return np.nan

    mid = n // 2
    chains = [samples[:mid], samples[mid:]]
    if weights is not None:
        w_chains = [weights[:mid], weights[mid:]]
    else:
        w_chains = [np.ones(mid), np.ones(n - mid)]

    # Weighted means and variances per half-chain
    chain_means = []
    chain_vars = []
    chain_ns = []
    for c, w in zip(chains, w_chains):
        ws = w.sum()
        if ws == 0:
            return np.nan
        mu = np.average(c, weights=w)
        var = np.average((c - mu) ** 2, weights=w)
        chain_means.append(mu)
        chain_vars.append(var)
        chain_ns.append(ws)

    # Between-chain variance
    grand_mean = np.mean(chain_means)
    B = np.var(chain_means, ddof=0) # variance of 2 means
    W = np.mean(chain_vars)

    if W == 0:
        return np.nan

    # R-1 approximation
    n_eff = np.mean(chain_ns)
    R_minus_1 = B / W
    return float(R_minus_1)


def block_r1(data, weights, columns, param_names):
    """Compute max R-1 across a block of parameters."""
    r1_vals = {}
    for p in param_names:
        if p not in columns:
            continue
        idx = columns.index(p)
        r1 = gelman_rubin_split(data[:, idx], weights)
        r1_vals[p] = r1

    if not r1_vals:
        return np.nan, {}
    max_r1 = max(r1_vals.values())
    return max_r1, r1_vals


def ess_autocorr(samples, weights=None, max_lag=500):
    """Estimate ESS via initial positive sequence estimator.

    Uses weighted samples with autocorrelation-based ESS.
    """
    n = len(samples)
    if n < 50:
        return 0.0

    # For weighted chains, use effective sample count
    if weights is not None:
        w_sum = weights.sum()
        w2_sum = (weights ** 2).sum()
        n_eff_weights = w_sum ** 2 / w2_sum
    else:
        n_eff_weights = float(n)
        weights = np.ones(n)

    # Normalize samples
    mu = np.average(samples, weights=weights)
    x = samples - mu
    var = np.average(x ** 2, weights=weights)
    if var == 0:
        return 0.0

    # Compute autocorrelation via FFT (unweighted approximation)
    max_lag = min(max_lag, n // 3)
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf[0] = 1.0
        else:
            c = np.mean(x[:-lag] * x[lag:])
            acf[lag] = c / var

    # Initial positive sequence: sum consecutive pairs until negative
    tau = 1.0
    for i in range(1, max_lag - 1, 2):
        pair_sum = acf[i] + acf[i + 1]
        if pair_sum <= 0:
            break
        tau += 2 * pair_sum

    ess = n_eff_weights / tau
    return max(1.0, float(ess))


def report_chain(prefix, model):
    """Generate full diagnostic report for one chain."""
    columns, data, ckpt = load_chain(prefix)
    if data is None:
        print(f"\n  [{model.upper()}] No chain file found at {RESULTS_DIR}/{prefix}.1.txt")
        return

    n_rows = data.shape[0]
    weights = data[:, columns.index('weight')]
    total_weight = weights.sum()

    # Acceptance rate: rows with weight > 0 that are new points
    # In cobaya drag sampling, weight = number of drag steps at that point
    n_accepted = n_rows
    n_steps = int(total_weight)

    # Burn-in: discard first 30%
    burn = int(0.3 * n_rows)
    data_post = data[burn:]
    weights_post = weights[burn:]

    cosmo_params = COSMO_MTDF if model == 'mtdf' else COSMO_LCDM
    ess_params = ESS_PARAMS_MTDF if model == 'mtdf' else ESS_PARAMS_LCDM

    # Block R-1
    cosmo_r1, cosmo_detail = block_r1(data_post, weights_post, columns, cosmo_params + DERIVED_KEY)
    nuis_r1, nuis_detail = block_r1(data_post, weights_post, columns, NUISANCE)
    overall_r1, _ = block_r1(data_post, weights_post, columns, cosmo_params + NUISANCE)

    # Checkpoint R-1
    ckpt_r1 = None
    if ckpt:
        ckpt_r1 = ckpt.get('sampler', {}).get('mcmc', {}).get('Rminus1_last')

    # ESS for key params
    ess_results = {}
    for p in ess_params:
        if p in columns:
            idx = columns.index(p)
            ess_results[p] = ess_autocorr(data_post[:, idx], weights_post)

    # Best chi2
    chi2_idx = columns.index('chi2') if 'chi2' in columns else None
    best_chi2 = float(np.min(data[:, chi2_idx])) if chi2_idx is not None else None

    # Last point summary
    last = data[-1]

    # Print report
    print(f"\n{'='*70}")
    print(f"  {model.upper()} MCMC  ({prefix})")
    print(f"{'='*70}")
    print(f"  Samples: {n_rows} rows, {int(total_weight)} total weight (post-burn: {len(data_post)})")
    if ckpt_r1 is not None:
        print(f"  Cobaya R-1 (checkpoint): {ckpt_r1:.6f}")
    print(f"  Acceptance rate: {n_rows / total_weight:.3f}" if total_weight > 0 else "")
    if best_chi2 is not None:
        print(f"  Best chi2: {best_chi2:.2f}")

    print(f"\n  --- R-1 by block (split-chain, post 30% burn-in) ---")
    print(f"  Overall max R-1:    {overall_r1:.4f}")
    print(f"  Cosmological block: {cosmo_r1:.4f}")
    worst_cosmo = max(cosmo_detail, key=cosmo_detail.get) if cosmo_detail else '?'
    print(f"    (worst: {worst_cosmo} = {cosmo_detail.get(worst_cosmo, 0):.4f})")
    print(f"  Nuisance block:     {nuis_r1:.4f}")
    worst_nuis = max(nuis_detail, key=nuis_detail.get) if nuis_detail else '?'
    print(f"    (worst: {worst_nuis} = {nuis_detail.get(worst_nuis, 0):.4f})")

    # Individual cosmo params
    print(f"\n  --- Cosmological R-1 detail ---")
    for p in cosmo_params + DERIVED_KEY:
        if p in cosmo_detail:
            print(f"    {p:20s}  R-1 = {cosmo_detail[p]:.4f}")

    print(f"\n  --- ESS (effective sample size) ---")
    for p, ess in ess_results.items():
        print(f"    {p:20s}  ESS = {ess:.0f}")

    # Current parameter values (last accepted point)
    print(f"\n  --- Current position (last sample) ---")
    for p in cosmo_params:
        if p in columns:
            print(f"    {p:20s} = {last[columns.index(p)]:.6f}")
    for p in DERIVED_KEY:
        if p in columns:
            print(f"    {p:20s} = {last[columns.index(p)]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='MCMC chain monitor')
    parser.add_argument('--watch', type=int, default=0,
                        help='Re-check every N seconds (0 = once)')
    parser.add_argument('--model', type=str, default='both',
                        choices=['lcdm', 'mtdf', 'both'],
                        help='Which model to monitor')
    args = parser.parse_args()

    while True:
        print(f"\n{'#'*70}")
        print(f"  MCMC MONITOR — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

        if args.model in ('lcdm', 'both'):
            report_chain('lcdm_mcmc', 'lcdm')
        if args.model in ('mtdf', 'both'):
            report_chain('mtdf_mcmc', 'mtdf')

        if args.watch <= 0:
            break
        print(f"\n  [next check in {args.watch}s]")
        sys.stdout.flush()
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
