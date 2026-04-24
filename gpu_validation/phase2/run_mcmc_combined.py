#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 2 Step C: Combined MCMC — Planck + BAO + SNe.

Extends the Planck-only MCMC with late-universe probes to test MTDF's
Hubble tension mechanism:
  - Planck plik-lite TTTEEE (613 bins, ell=30-2508)
  - DESI Y1 BAO (12 measurements)
  - Pantheon+ SNe Ia (1701 distance moduli, M marginalized)

k_f convention:
  k_f = 0   -> pure LCDM
  k_f = 1.0 -> full MTDF theory prediction

Key physics: MTDF's stress correction H(z) = H0*E(z)*(1 + k_f*kappa*alpha*z/(1+z))
modifies late-universe distances, potentially resolving the Hubble tension by
shifting H0 toward ~70 km/s/Mpc when k_f ~ 1.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results" / "phase2"
DATA_DIR = Path(__file__).parent.parent.parent / "validation" / "data"

# Import shared definitions from Planck-only MCMC
from phase2.run_mcmc import (
    PARAM_NAMES, PLANCK_BESTFIT, PRIOR_BOUNDS,
    log_prior, _apply_kf_correction, initialize_walkers,
)


def make_combined_log_likelihood(plik, emulators, bao_like, sne_like):
    """Create combined log-likelihood: Planck + BAO + SNe.

    Returns a function (theta) -> (logL, logL_planck, logL_bao, logL_sne).
    The main return is logL; the components are for diagnostics.
    """
    from phase2.cosmopower_setup import predict_dl

    def log_likelihood(theta):
        params = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}

        omega_b = params['omega_b']
        omega_cdm = params['omega_cdm']
        H0 = params['H0']
        k_f = params['k_f']

        h = H0 / 100.0
        omega_m_h2 = omega_b + omega_cdm
        Omega_m = omega_m_h2 / h**2

        # --- Planck ---
        cp_params = {k: params[k] for k in PARAM_NAMES[:6]}
        try:
            ells_tt, dl_tt = predict_dl(emulators['TT'], cp_params)
            ells_te, dl_te = predict_dl(emulators['TE'], cp_params)
            ells_ee, dl_ee = predict_dl(emulators['EE'], cp_params)
        except Exception:
            return -np.inf

        if k_f > 1e-6:
            dl_tt = _apply_kf_correction(ells_tt, dl_tt, k_f, params)
            dl_te = _apply_kf_correction(ells_te, dl_te, k_f, params)
            dl_ee = _apply_kf_correction(ells_ee, dl_ee, k_f, params)

        lmax = max(int(ells_tt[-1]), 2600) + 1
        dl_tt_full = np.zeros(lmax)
        dl_te_full = np.zeros(lmax)
        dl_ee_full = np.zeros(lmax)
        ell_start = int(ells_tt[0])
        n = len(dl_tt)
        dl_tt_full[ell_start:ell_start + n] = dl_tt
        dl_te_full[ell_start:ell_start + n] = dl_te
        dl_ee_full[ell_start:ell_start + n] = dl_ee

        chi2_planck = plik.chi2(dl_tt_full, dl_te_full, dl_ee_full)
        if not np.isfinite(chi2_planck):
            return -np.inf

        # --- BAO ---
        try:
            chi2_bao = bao_like.chi2(H0, Omega_m, omega_b, omega_m_h2, k_f)
        except Exception:
            return -np.inf
        if not np.isfinite(chi2_bao):
            return -np.inf

        # --- SNe ---
        try:
            chi2_sne = sne_like.chi2(H0, Omega_m, k_f)
        except Exception:
            return -np.inf
        if not np.isfinite(chi2_sne):
            return -np.inf

        return -0.5 * (chi2_planck + chi2_bao + chi2_sne)

    return log_likelihood


def log_posterior(theta, log_like_fn):
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_like_fn(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def compute_per_probe_chi2(theta, plik, emulators, bao_like, sne_like):
    """Compute chi2 breakdown by probe for a given parameter vector."""
    from phase2.cosmopower_setup import predict_dl

    params = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
    omega_b = params['omega_b']
    omega_cdm = params['omega_cdm']
    H0 = params['H0']
    k_f = params['k_f']
    h = H0 / 100.0
    omega_m_h2 = omega_b + omega_cdm
    Omega_m = omega_m_h2 / h**2

    # Planck
    cp_params = {k: params[k] for k in PARAM_NAMES[:6]}
    ells_tt, dl_tt = predict_dl(emulators['TT'], cp_params)
    ells_te, dl_te = predict_dl(emulators['TE'], cp_params)
    ells_ee, dl_ee = predict_dl(emulators['EE'], cp_params)
    if k_f > 1e-6:
        dl_tt = _apply_kf_correction(ells_tt, dl_tt, k_f, params)
        dl_te = _apply_kf_correction(ells_te, dl_te, k_f, params)
        dl_ee = _apply_kf_correction(ells_ee, dl_ee, k_f, params)
    lmax = max(int(ells_tt[-1]), 2600) + 1
    dl_tt_full = np.zeros(lmax)
    dl_te_full = np.zeros(lmax)
    dl_ee_full = np.zeros(lmax)
    ell_start = int(ells_tt[0])
    n = len(dl_tt)
    dl_tt_full[ell_start:ell_start + n] = dl_tt
    dl_te_full[ell_start:ell_start + n] = dl_te
    dl_ee_full[ell_start:ell_start + n] = dl_ee
    chi2_planck = plik.chi2(dl_tt_full, dl_te_full, dl_ee_full)

    # BAO
    chi2_bao = bao_like.chi2(H0, Omega_m, omega_b, omega_m_h2, k_f)

    # SNe
    chi2_sne = sne_like.chi2(H0, Omega_m, k_f)

    return chi2_planck, chi2_bao, chi2_sne


def run_combined_mcmc(n_walkers=32, n_steps=5000, n_burn=1000, progress=True, tag=''):
    """Run combined Planck + BAO + SNe MCMC."""
    suffix = f"_{tag}" if tag else ""
    import emcee

    print("=" * 70)
    print("MTDF PHASE 2 STEP C: Combined MCMC — Planck + BAO + SNe")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    n_dim = len(PARAM_NAMES)

    # --- Load Planck emulators ---
    print("\n[1] Loading CosmoPower emulators...")
    from phase2.cosmopower_setup import load_emulator
    t0 = time.time()
    emulators = {
        'TT': load_emulator('TT'),
        'TE': load_emulator('TE'),
        'EE': load_emulator('EE'),
    }
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # --- Load Planck likelihood ---
    print("\n[2] Loading Planck plik-lite likelihood...")
    from phase2.planck_lite_likelihood import PlanckLiteLikelihood
    plik = PlanckLiteLikelihood()
    print(f"  {plik.n_used} bins (TT:{plik.n_tt} TE:{plik.n_te} EE:{plik.n_ee})")

    # --- Load BAO likelihood ---
    print("\n[3] Loading DESI Y1 BAO likelihood...")
    from phase2.late_universe_likelihood import BAOLikelihood
    t0 = time.time()
    bao_like = BAOLikelihood(DATA_DIR)
    print(f"  {bao_like.n} measurements, loaded in {time.time()-t0:.1f}s")
    print(f"  z_eff: {bao_like.z_eff}")

    # --- Load SNe likelihood ---
    print("\n[4] Loading Pantheon+ SNe likelihood...")
    from phase2.late_universe_likelihood import SNeLikelihood
    t0 = time.time()
    sne_like = SNeLikelihood(DATA_DIR)
    print(f"  {sne_like.n} SNe, loaded in {time.time()-t0:.1f}s")
    print(f"  z range: [{sne_like.z.min():.4f}, {sne_like.z.max():.4f}]")

    # --- Create combined likelihood ---
    log_like_fn = make_combined_log_likelihood(plik, emulators, bao_like, sne_like)

    # --- Test evaluations ---
    print("\n[5] Testing likelihood evaluations...")
    theta0 = np.array([PLANCK_BESTFIT[name] for name in PARAM_NAMES])

    # LCDM baseline (k_f=0)
    t0 = time.time()
    ll_lcdm = log_like_fn(theta0)
    eval_ms = (time.time() - t0) * 1000
    chi2_p, chi2_b, chi2_s = compute_per_probe_chi2(theta0, plik, emulators, bao_like, sne_like)
    print(f"  LCDM (k_f=0):  logL = {ll_lcdm:.2f}  ({eval_ms:.0f}ms)")
    print(f"    Planck chi2 = {chi2_p:.2f}  (613 bins)")
    print(f"    BAO chi2    = {chi2_b:.2f}  ({bao_like.n} points)")
    print(f"    SNe chi2    = {chi2_s:.2f}  ({sne_like.n} SNe)")
    print(f"    Total chi2  = {chi2_p + chi2_b + chi2_s:.2f}")

    # MTDF (k_f=1)
    theta_kf1 = theta0.copy()
    theta_kf1[-1] = 1.0
    t0 = time.time()
    ll_mtdf = log_like_fn(theta_kf1)
    eval_ms = (time.time() - t0) * 1000
    chi2_p1, chi2_b1, chi2_s1 = compute_per_probe_chi2(theta_kf1, plik, emulators, bao_like, sne_like)
    print(f"\n  MTDF (k_f=1):  logL = {ll_mtdf:.2f}  ({eval_ms:.0f}ms)")
    print(f"    Planck chi2 = {chi2_p1:.2f}  (delta = {chi2_p1-chi2_p:+.2f})")
    print(f"    BAO chi2    = {chi2_b1:.2f}  (delta = {chi2_b1-chi2_b:+.2f})")
    print(f"    SNe chi2    = {chi2_s1:.2f}  (delta = {chi2_s1-chi2_s:+.2f})")
    print(f"    Total chi2  = {chi2_p1 + chi2_b1 + chi2_s1:.2f}  (delta = {(chi2_p1+chi2_b1+chi2_s1)-(chi2_p+chi2_b+chi2_s):+.2f})")

    # --- Initialize walkers ---
    print(f"\n[6] Initializing {n_walkers} walkers...")
    pos = initialize_walkers(n_walkers, n_dim)
    print(f"  Initial positions: mean={pos.mean(axis=0)}")

    # --- Run MCMC ---
    print(f"\n[7] Running MCMC: {n_steps} steps, {n_walkers} walkers...")

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior,
        args=(log_like_fn,),
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

    # --- Analyze chains ---
    print("\n[8] Chain analysis...")
    chain = sampler.get_chain(flat=False)
    log_prob = sampler.get_log_prob(flat=False)

    acc = sampler.acceptance_fraction
    print(f"  Acceptance fraction: {acc.mean():.3f} (range: {acc.min():.3f}-{acc.max():.3f})")

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  Autocorrelation time: {tau}")
        print(f"  Effective samples: ~{n_steps * n_walkers / tau.max():.0f}")
    except Exception as e:
        print(f"  Autocorrelation: could not compute ({e})")
        tau = None

    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=n_burn, flat=True)
    print(f"\n  Post burn-in: {flat_chain.shape[0]} samples")

    # Best fit
    best_idx = np.argmax(flat_log_prob)
    best_theta = flat_chain[best_idx]
    best_logL = flat_log_prob[best_idx]

    # Per-probe chi2 at best fit
    chi2_p_bf, chi2_b_bf, chi2_s_bf = compute_per_probe_chi2(
        best_theta, plik, emulators, bao_like, sne_like
    )

    print(f"\n  Best-fit (max logL = {best_logL:.2f}, total chi2 = {-2*best_logL:.2f}):")
    print(f"    Planck: {chi2_p_bf:.2f}  |  BAO: {chi2_b_bf:.2f}  |  SNe: {chi2_s_bf:.2f}")
    for i, name in enumerate(PARAM_NAMES):
        med = np.median(flat_chain[:, i])
        lo, hi = np.percentile(flat_chain[:, i], [16, 84])
        print(f"    {name:20s} = {best_theta[i]:.6f}  (median: {med:.6f} +{hi-med:.6f} -{med-lo:.6f})")

    # k_f posterior
    k_f_samples = flat_chain[:, -1]
    print(f"\n  k_f posterior:")
    print(f"    Mean:   {k_f_samples.mean():.6f}")
    print(f"    Median: {np.median(k_f_samples):.6f}")
    print(f"    95% UL: {np.percentile(k_f_samples, 95):.6f}")

    # H0 posterior
    H0_samples = flat_chain[:, 2]
    print(f"\n  H0 posterior:")
    print(f"    Mean:   {H0_samples.mean():.4f}")
    print(f"    Median: {np.median(H0_samples):.4f}")
    h0_lo, h0_hi = np.percentile(H0_samples, [16, 84])
    print(f"    68% CI: [{h0_lo:.2f}, {h0_hi:.2f}]")

    # LCDM comparison
    lcdm_logL = log_like_fn(theta0)
    delta_chi2 = -2 * (best_logL - lcdm_logL)
    print(f"\n  LCDM total chi2:       {-2*lcdm_logL:.2f}")
    print(f"  MTDF best total chi2:  {-2*best_logL:.2f}")
    print(f"  Delta-chi2:            {delta_chi2:+.2f} (negative = MTDF better)")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        RESULTS_DIR / f"mcmc_combined{suffix}_chain.npz",
        chain=chain,
        log_prob=log_prob,
        flat_chain=flat_chain,
        flat_log_prob=flat_log_prob,
        param_names=PARAM_NAMES,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
    )

    summary = {
        'timestamp': datetime.now().isoformat(),
        'probes': ['planck_plik_lite', 'desi_y1_bao', 'pantheonplus_sne'],
        'config': {
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burn': n_burn,
            'n_dim': n_dim,
        },
        'acceptance_fraction': float(acc.mean()),
        'best_fit': {name: float(best_theta[i]) for i, name in enumerate(PARAM_NAMES)},
        'best_logL': float(best_logL),
        'best_chi2_total': float(-2 * best_logL),
        'best_chi2_planck': float(chi2_p_bf),
        'best_chi2_bao': float(chi2_b_bf),
        'best_chi2_sne': float(chi2_s_bf),
        'lcdm_chi2_total': float(-2 * lcdm_logL),
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

    summary['config']['kf_prior'] = list(PRIOR_BOUNDS['k_f'])
    with open(RESULTS_DIR / f"mcmc_combined{suffix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Chain saved to {RESULTS_DIR / f'mcmc_combined{suffix}_chain.npz'}")
    print(f"  Summary saved to {RESULTS_DIR / f'mcmc_combined{suffix}_summary.json'}")

    # Corner plot
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
        fig.suptitle("Planck + BAO + SNe", fontsize=14, y=1.02)
        fig.savefig(RESULTS_DIR / f"corner_combined{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Corner plot saved to {RESULTS_DIR / f'corner_combined{suffix}.png'}")
    except Exception as e:
        print(f"  Corner plot skipped: {e}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 2 Combined MCMC')
    parser.add_argument('--walkers', type=int, default=32)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--burn', type=int, default=1000)
    parser.add_argument('--kf-max', type=float, default=5.0,
                        help='Upper bound for k_f prior (default: 5.0)')
    parser.add_argument('--tag', type=str, default='',
                        help='Tag for output filenames (e.g. "narrow")')
    args = parser.parse_args()

    # Override k_f prior bounds if specified
    if args.kf_max != 5.0:
        PRIOR_BOUNDS['k_f'] = (0.0, args.kf_max)
        print(f"[Override] k_f prior: [0, {args.kf_max}]")

    run_combined_mcmc(n_walkers=args.walkers, n_steps=args.steps,
                      n_burn=args.burn, tag=args.tag)
