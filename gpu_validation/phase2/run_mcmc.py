#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 2 Step B: MCMC with CosmoPower + MTDF correction layer.

Runs emcee MCMC sampling on Planck plik-lite TTTEEE likelihood with:
  - CosmoPower emulators for fast C_l evaluation (~5ms/eval)
  - MTDF perturbative correction layer on top of LCDM baseline
  - 7 free parameters: {omega_b, omega_cdm, h, tau, n_s, ln10^10*A_s, k_f}

k_f convention (matches corrected CLASS, CMTDF_normalization_validation.txt):
  k_f = 0   → pure LCDM
  k_f = 1.0 → full MTDF theory prediction (f_kick = lambda_MTDF/24 = 0.0033)

Expected result: k_f consistent with 0-1 range from Planck alone.
MTDF's primary action is at late times, not in the early universe.

Strategy:
  Step B1: Short chain (5000 steps) to verify convergence
  Step B2: Production chain (50000 steps) with burn-in analysis
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results" / "phase2"


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------

# Parameter names, Planck 2018 best-fit values, and prior ranges
PARAM_NAMES = ['omega_b', 'omega_cdm', 'H0', 'tau_reio', 'n_s', 'ln10^{10}A_s', 'k_f']

PLANCK_BESTFIT = {
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'H0': 67.36,
    'tau_reio': 0.0544,
    'n_s': 0.9649,
    'ln10^{10}A_s': 3.044,
    'k_f': 0.0,  # LCDM limit
}

# Flat prior bounds (chosen to stay within CosmoPower training range)
PRIOR_BOUNDS = {
    'omega_b':       (0.019, 0.026),
    'omega_cdm':     (0.08, 0.20),
    'H0':            (50, 90),
    'tau_reio':      (0.01, 0.12),
    'n_s':           (0.88, 1.06),
    'ln10^{10}A_s':  (2.5, 3.5),
    'k_f':           (0.0, 5.0),  # MTDF EFE multiplier: 0=LCDM, 1.0=full MTDF theory
}

# Initial scatter for walkers (fraction of prior range)
INIT_SCATTER = 0.01


def log_prior(theta):
    """Flat prior within bounds."""
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR_BOUNDS[name]
        if theta[i] < lo or theta[i] > hi:
            return -np.inf
    # Gaussian prior on tau from Planck lowE
    tau = theta[3]
    tau_mean, tau_sigma = 0.0544, 0.0073
    lp = -0.5 * ((tau - tau_mean) / tau_sigma) ** 2
    return lp


def make_log_likelihood(plik, emulators):
    """Create log-likelihood function using CosmoPower + MTDF correction.

    Parameters
    ----------
    plik : PlanckLiteLikelihood
    emulators : dict with keys 'TT', 'TE', 'EE'
    """
    from phase2.cosmopower_setup import predict_dl, T_CMB_SQ

    def log_likelihood(theta):
        # Unpack parameters
        params = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}

        # CosmoPower input (first 6 standard LCDM params)
        cp_params = {k: params[k] for k in PARAM_NAMES[:6]}

        try:
            # Get LCDM baseline spectra from emulators
            ells_tt, dl_tt = predict_dl(emulators['TT'], cp_params)
            ells_te, dl_te = predict_dl(emulators['TE'], cp_params)
            ells_ee, dl_ee = predict_dl(emulators['EE'], cp_params)
        except Exception:
            return -np.inf

        # Apply MTDF correction if k_f > 0
        k_f = params['k_f']
        if k_f > 1e-6:
            dl_tt = _apply_kf_correction(ells_tt, dl_tt, k_f, params)
            dl_te = _apply_kf_correction(ells_te, dl_te, k_f, params)
            dl_ee = _apply_kf_correction(ells_ee, dl_ee, k_f, params)

        # Pad to start from ell=0 for the likelihood
        lmax = max(int(ells_tt[-1]), 2600) + 1
        dl_tt_full = np.zeros(lmax)
        dl_te_full = np.zeros(lmax)
        dl_ee_full = np.zeros(lmax)

        ell_start = int(ells_tt[0])
        n = len(dl_tt)
        dl_tt_full[ell_start:ell_start + n] = dl_tt
        dl_te_full[ell_start:ell_start + n] = dl_te
        dl_ee_full[ell_start:ell_start + n] = dl_ee

        # Compute chi2
        chi2 = plik.chi2(dl_tt_full, dl_te_full, dl_ee_full, A_planck=1.0)

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2

    return log_likelihood


def _apply_kf_correction(ells, dl, k_f, params):
    """Apply MTDF EFE correction parameterized by k_f.

    k_f is the dimensionless EFE multiplier matching the corrected CLASS
    convention (CMTDF_normalization_validation.txt):
      k_f = 0   → pure LCDM
      k_f = 1.0 → full MTDF (theory prediction)

    Physical coupling: f_kick = k_f * lambda_MTDF / 24
    where lambda_MTDF = (1-beta_eos)^2 / (1+alpha) = 0.07927

    All corrections scale linearly with k_f.
    """
    # Corrected reference values at k_f = 1.0 (full MTDF)
    # theta_s ratio from mtdf_correction_layer.py with corrected f_kick
    THETA_RATIO_REF = 1.000212  # theta_s_MTDF / theta_s_LCDM at k_f=1
    KAPPA = 0.00102  # Anchor: approx f_kick/3 = (1-beta_eos)^2/(72*(1+alpha))
    ALPHA = 1.30

    # Scale theta_s ratio linearly with k_f
    theta_ratio = 1.0 + k_f * (THETA_RATIO_REF - 1.0)

    # Rescale ell axis: Cl_MTDF(ell) = Cl_LCDM(ell * theta_ratio)
    ells_f = ells.astype(float)
    ells_shifted = ells_f * theta_ratio

    # Interpolate back to original grid
    dl_corrected = np.interp(ells_shifted, ells_f, dl,
                             left=dl[0], right=dl[-1])

    # ISW boost at low ell (scales with k_f)
    isw_amplitude = k_f * 2.0 * KAPPA * ALPHA
    isw_boost = 1.0 + isw_amplitude * np.exp(-ells_f / 30.0)
    dl_corrected *= isw_boost

    return dl_corrected


def log_posterior(theta, plik, emulators, log_like_fn):
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_like_fn(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def initialize_walkers(n_walkers, n_dim):
    """Initialize walker positions near Planck best-fit."""
    p0 = np.array([PLANCK_BESTFIT[name] for name in PARAM_NAMES])
    scatter = np.array([
        (PRIOR_BOUNDS[name][1] - PRIOR_BOUNDS[name][0]) * INIT_SCATTER
        for name in PARAM_NAMES
    ])
    # k_f starts at 0, scatter small to let the sampler explore from LCDM
    scatter[-1] = 0.1  # k_f scatter: ~0.1 (prior is [0, 5])

    pos = p0 + scatter * np.random.randn(n_walkers, n_dim)

    # Enforce prior bounds
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PRIOR_BOUNDS[name]
        pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)

    return pos


def run_mcmc(n_walkers=32, n_steps=5000, n_burn=1000, progress=True):
    """Run the MCMC chain.

    Parameters
    ----------
    n_walkers : int
        Number of emcee walkers.
    n_steps : int
        Number of MCMC steps per walker.
    n_burn : int
        Burn-in steps to discard.
    progress : bool
        Show progress bar.
    """
    import emcee

    print("=" * 70)
    print("MTDF PHASE 2 STEP B: MCMC with CosmoPower + MTDF")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    n_dim = len(PARAM_NAMES)

    # Load emulators
    print("\n[1] Loading emulators...")
    from phase2.cosmopower_setup import load_emulator
    t0 = time.time()
    emulators = {
        'TT': load_emulator('TT'),
        'TE': load_emulator('TE'),
        'EE': load_emulator('EE'),
    }
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load likelihood
    print("\n[2] Loading plik-lite likelihood...")
    from phase2.planck_lite_likelihood import PlanckLiteLikelihood
    plik = PlanckLiteLikelihood()
    print(f"  {plik.n_used} bins (TT:{plik.n_tt} TE:{plik.n_te} EE:{plik.n_ee})")

    # Create log-likelihood function
    log_like_fn = make_log_likelihood(plik, emulators)

    # Test single evaluation
    print("\n[3] Testing single likelihood evaluation...")
    theta0 = np.array([PLANCK_BESTFIT[name] for name in PARAM_NAMES])
    t0 = time.time()
    ll0 = log_like_fn(theta0)
    eval_time = time.time() - t0
    print(f"  logL(LCDM best-fit) = {ll0:.2f}  ({eval_time*1000:.0f}ms)")
    print(f"  chi2 = {-2*ll0:.2f}")

    # Test with k_f = 1.0 (full MTDF theory prediction)
    theta_kf = theta0.copy()
    theta_kf[-1] = 1.0
    t0 = time.time()
    ll_kf = log_like_fn(theta_kf)
    eval_time = time.time() - t0
    print(f"  logL(k_f=1.0)       = {ll_kf:.2f}  ({eval_time*1000:.0f}ms)")
    print(f"  Delta-logL          = {ll_kf - ll0:+.2f}")
    print(f"  Delta-chi2          = {-2*(ll_kf - ll0):+.2f}")

    # Initialize walkers
    print(f"\n[4] Initializing {n_walkers} walkers...")
    pos = initialize_walkers(n_walkers, n_dim)
    print(f"  Initial positions: mean={pos.mean(axis=0)}")

    # Run MCMC
    print(f"\n[5] Running MCMC: {n_steps} steps, {n_walkers} walkers...")
    print(f"  Expected time: ~{n_steps * n_walkers * 0.015 / 60:.0f} min")

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior,
        args=(plik, emulators, log_like_fn),
    )

    t0 = time.time()
    if progress:
        try:
            from tqdm import tqdm
            for _ in tqdm(sampler.sample(pos, iterations=n_steps), total=n_steps):
                pass
        except ImportError:
            sampler.run_mcmc(pos, n_steps, progress=True)
    else:
        sampler.run_mcmc(pos, n_steps)

    elapsed = time.time() - t0
    print(f"\n  MCMC completed in {elapsed/60:.1f} min ({elapsed/n_steps/n_walkers*1000:.1f} ms/eval)")

    # Analyze chains
    print("\n[6] Chain analysis...")
    chain = sampler.get_chain(flat=False)  # (n_steps, n_walkers, n_dim)
    log_prob = sampler.get_log_prob(flat=False)  # (n_steps, n_walkers)

    # Acceptance fraction
    acc = sampler.acceptance_fraction
    print(f"  Acceptance fraction: {acc.mean():.3f} (range: {acc.min():.3f}-{acc.max():.3f})")

    # Auto-correlation time (try, may fail for short chains)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  Autocorrelation time: {tau}")
        print(f"  Effective samples: ~{n_steps * n_walkers / tau.max():.0f}")
    except Exception as e:
        print(f"  Autocorrelation: could not compute ({e})")
        tau = None

    # Discard burn-in
    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=n_burn, flat=True)

    print(f"\n  Post burn-in: {flat_chain.shape[0]} samples")

    # Best fit
    best_idx = np.argmax(flat_log_prob)
    best_theta = flat_chain[best_idx]
    best_logL = flat_log_prob[best_idx]

    print(f"\n  Best-fit (max logL = {best_logL:.2f}, chi2 = {-2*best_logL:.2f}):")
    for i, name in enumerate(PARAM_NAMES):
        med = np.median(flat_chain[:, i])
        lo, hi = np.percentile(flat_chain[:, i], [16, 84])
        print(f"    {name:20s} = {best_theta[i]:.6f}  (median: {med:.6f} +{hi-med:.6f} -{med-lo:.6f})")

    # Key result: k_f posterior
    k_f_samples = flat_chain[:, -1]
    print(f"\n  k_f posterior:")
    print(f"    Mean:   {k_f_samples.mean():.6f}")
    print(f"    Median: {np.median(k_f_samples):.6f}")
    print(f"    95% UL: {np.percentile(k_f_samples, 95):.6f}")
    print(f"    Converges to 0: {'YES' if np.median(k_f_samples) < 0.01 else 'CHECK'}")

    # Compare LCDM vs MTDF best-fit
    lcdm_logL = log_like_fn(theta0)
    delta_chi2 = -2 * (best_logL - lcdm_logL)
    print(f"\n  LCDM chi2:      {-2*lcdm_logL:.2f}")
    print(f"  MTDF best chi2: {-2*best_logL:.2f}")
    print(f"  Delta-chi2:     {delta_chi2:+.2f} (negative = MTDF better)")

    # Interpretation
    print(f"\n  Interpretation:")
    if np.median(k_f_samples) < 0.01:
        print(f"  k_f → 0 confirms MTDF ≈ ΛCDM at CMB epoch.")
        print(f"  The early universe is effectively ΛCDM.")
        print(f"  MTDF's action is at late times (voids, environment).")
    else:
        print(f"  k_f > 0 suggests some early-universe EFE contribution.")
        print(f"  Check consistency with framework predictions.")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save full chain
    np.savez(
        RESULTS_DIR / "mcmc_chain.npz",
        chain=chain,
        log_prob=log_prob,
        flat_chain=flat_chain,
        flat_log_prob=flat_log_prob,
        param_names=PARAM_NAMES,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
    )

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burn': n_burn,
            'n_dim': n_dim,
        },
        'acceptance_fraction': float(acc.mean()),
        'best_fit': {name: float(best_theta[i]) for i, name in enumerate(PARAM_NAMES)},
        'best_logL': float(best_logL),
        'best_chi2': float(-2 * best_logL),
        'lcdm_chi2': float(-2 * lcdm_logL),
        'delta_chi2': float(delta_chi2),
        'posteriors': {},
        'elapsed_minutes': elapsed / 60,
    }
    for i, name in enumerate(PARAM_NAMES):
        med = float(np.median(flat_chain[:, i]))
        lo, hi = float(np.percentile(flat_chain[:, i], 16)), float(np.percentile(flat_chain[:, i], 84))
        summary['posteriors'][name] = {
            'median': med, 'lower_1sigma': lo, 'upper_1sigma': hi,
            'mean': float(flat_chain[:, i].mean()),
            'std': float(flat_chain[:, i].std()),
        }
    summary['posteriors']['k_f']['95_upper_limit'] = float(np.percentile(k_f_samples, 95))

    with open(RESULTS_DIR / "mcmc_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Chain saved to {RESULTS_DIR / 'mcmc_chain.npz'}")
    print(f"  Summary saved to {RESULTS_DIR / 'mcmc_summary.json'}")

    # Generate corner plot if corner is available
    try:
        import corner
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        labels = ['$\\omega_b$', '$\\omega_{cdm}$', '$H_0$', '$\\tau$',
                  '$n_s$', '$\\ln(10^{10}A_s)$', '$k_f$']
        fig = corner.corner(
            flat_chain, labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
        )
        fig.savefig(RESULTS_DIR / "corner_plot.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Corner plot saved to {RESULTS_DIR / 'corner_plot.png'}")
    except Exception as e:
        print(f"  Corner plot skipped: {e}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 2 MCMC')
    parser.add_argument('--walkers', type=int, default=32)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--burn', type=int, default=1000)
    args = parser.parse_args()

    run_mcmc(n_walkers=args.walkers, n_steps=args.steps, n_burn=args.burn)
