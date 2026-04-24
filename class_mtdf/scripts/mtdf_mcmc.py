#!/usr/bin/env python3
"""
MCMC sampling for MTDF+EFE with Planck TT likelihood.
Uses emcee with our custom CLASS implementation.
"""

import numpy as np
import emcee
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add clipy to path
sys.path.insert(0, 'cobaya_packages/code/planck/clipy')

import classy

# =============================================================================
# Parameter definitions
# =============================================================================

# Parameter names and their priors [min, max]
PARAM_NAMES = ['H0', 'omega_b', 'omega_cdm', 'n_s', 'logA', 'tau_reio', 'mtdf_k_f']
PARAM_PRIORS = {
    'H0':        [60.0, 80.0],
    'omega_b':   [0.020, 0.024],
    'omega_cdm': [0.10, 0.16],
    'n_s':       [0.90, 1.02],
    'logA':      [2.5, 3.6],      # ln(10^10 A_s)
    'tau_reio':  [0.03, 0.10],
    'mtdf_k_f':  [0.0, 2.0],
}

# Fixed MTDF parameters
MTDF_ALPHA = 1.30
MTDF_BETA_EOS = 0.573
MTDF_Z_T = 0.74

# =============================================================================
# Likelihood functions
# =============================================================================

# Global likelihood object (initialized once)
_planck_lkl = None

def init_planck_likelihood():
    """Initialize Planck TT likelihood."""
    global _planck_lkl
    if _planck_lkl is None:
        import clipy
        # TT-only likelihood
        clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik'
        if os.path.exists(clik_path):
            _planck_lkl = clipy.clik(clik_path)
            print(f"Loaded TT-only likelihood: {clik_path}")
        else:
            # Fall back to TTTEEE
            clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik'
            _planck_lkl = clipy.clik(clik_path)
            print(f"Loaded TTTEEE likelihood: {clik_path}")
    return _planck_lkl


def compute_cls(params_dict, ell_max=2600):
    """Compute Cls using MTDF CLASS."""
    cosmo = classy.Class()

    # Convert logA to A_s
    A_s = np.exp(params_dict['logA']) * 1e-10

    class_params = {
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': ell_max,
        'omega_b': params_dict['omega_b'],
        'omega_cdm': params_dict['omega_cdm'],
        'H0': params_dict['H0'],
        'tau_reio': params_dict['tau_reio'],
        'A_s': A_s,
        'n_s': params_dict['n_s'],
        # MTDF parameters
        'mtdf': 'yes',
        'mtdf_efe': 'yes',
        'mtdf_growth': 'no',
        'mtdf_k_f': params_dict['mtdf_k_f'],
        'mtdf_alpha': MTDF_ALPHA,
        'mtdf_beta_eos': MTDF_BETA_EOS,
        'mtdf_z_t': MTDF_Z_T,
    }

    cosmo.set(class_params)
    cosmo.compute()

    # Get lensed Cls
    cls_dict = cosmo.lensed_cl(ell_max)

    # Convert to μK²
    T_cmb = 2.7255e6
    factor = T_cmb**2

    cl_tt = cls_dict['tt'][:ell_max+1] * factor
    cl_ee = cls_dict['ee'][:ell_max+1] * factor
    cl_te = cls_dict['te'][:ell_max+1] * factor

    # Get r_s
    bg = cosmo.get_background()
    idx = np.argmin(np.abs(bg['z'] - 1089))
    r_s = bg['comov.snd.hrz.'][idx]

    cosmo.struct_cleanup()
    cosmo.empty()

    return cl_tt, cl_ee, cl_te, r_s


def log_likelihood(theta):
    """Compute log-likelihood for parameter vector theta."""
    # Unpack parameters
    params_dict = {name: val for name, val in zip(PARAM_NAMES, theta)}

    try:
        # Compute Cls
        cl_tt, cl_ee, cl_te, r_s = compute_cls(params_dict)

        # Get likelihood
        lkl = init_planck_likelihood()
        lmax = lkl.get_lmax()

        # Build input vector (depends on likelihood type)
        # For TT-only lite: [TT(0..lmax_tt), A_planck]
        # For TTTEEE lite: [TT, EE, BB, TE, A_planck]

        if lmax[1] < 0:  # TT-only
            lmax_tt = lmax[0]
            input_vec = np.concatenate([cl_tt[:lmax_tt+1], [1.0]])
        else:  # TTTEEE
            lmax_tt = lmax[0]
            lmax_ee = lmax[1]
            lmax_te = lmax[3]
            cl_bb = np.zeros(lmax[2]+1) if lmax[2] >= 0 else np.array([])
            input_vec = np.concatenate([
                cl_tt[:lmax_tt+1],
                cl_ee[:lmax_ee+1],
                cl_bb,
                cl_te[:lmax_te+1],
                [1.0]  # A_planck
            ])

        loglike = lkl(input_vec)
        return loglike

    except Exception as e:
        # Return very negative log-likelihood on failure
        return -1e30


def log_prior(theta):
    """Compute log-prior for parameter vector theta."""
    for name, val in zip(PARAM_NAMES, theta):
        pmin, pmax = PARAM_PRIORS[name]
        if val < pmin or val > pmax:
            return -np.inf
    return 0.0


def log_probability(theta):
    """Compute log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# =============================================================================
# MCMC sampling
# =============================================================================

def run_mcmc(nwalkers=32, nsteps=500, initial_guess=None, output_dir=None):
    """Run MCMC sampling."""
    ndim = len(PARAM_NAMES)

    # Initial guess (roughly Planck + k_f=1)
    if initial_guess is None:
        initial_guess = {
            'H0': 70.0,
            'omega_b': 0.02237,
            'omega_cdm': 0.12,
            'n_s': 0.9649,
            'logA': 3.044,  # ln(10^10 * 2.1e-9) = 3.044
            'tau_reio': 0.054,
            'mtdf_k_f': 1.0,
        }

    # Convert to array
    p0_center = np.array([initial_guess[name] for name in PARAM_NAMES])

    # Initialize walkers with small scatter around initial guess
    scatter = np.array([0.5, 0.0005, 0.005, 0.01, 0.1, 0.01, 0.2])
    p0 = p0_center + scatter * np.random.randn(nwalkers, ndim)

    # Ensure initial positions are within prior bounds
    for i, name in enumerate(PARAM_NAMES):
        pmin, pmax = PARAM_PRIORS[name]
        p0[:, i] = np.clip(p0[:, i], pmin + 1e-6, pmax - 1e-6)

    print(f"Starting MCMC with {nwalkers} walkers, {nsteps} steps")
    print(f"Parameters: {PARAM_NAMES}")
    print(f"Initial center: {p0_center}")

    # Initialize likelihood
    init_planck_likelihood()

    # Run sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    print("\nRunning burn-in (100 steps)...")
    state = sampler.run_mcmc(p0, 100, progress=True)
    sampler.reset()

    print(f"\nRunning production ({nsteps} steps)...")
    sampler.run_mcmc(state, nsteps, progress=True)

    # Get samples
    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    # Find best fit
    best_idx = np.argmax(log_probs)
    best_params = samples[best_idx]
    best_logprob = log_probs[best_idx]
    best_chi2 = -2 * best_logprob

    print("\n" + "="*60)
    print("MCMC Results")
    print("="*60)
    print(f"\nBest fit (χ² = {best_chi2:.2f}):")
    for name, val in zip(PARAM_NAMES, best_params):
        print(f"  {name:12s} = {val:.6f}")

    # Compute statistics
    print("\nPosterior statistics (mean ± std):")
    for i, name in enumerate(PARAM_NAMES):
        mean = np.mean(samples[:, i])
        std = np.std(samples[:, i])
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name:12s} = {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}")

    # Compute r_s for best fit
    best_dict = {name: val for name, val in zip(PARAM_NAMES, best_params)}
    _, _, _, r_s_best = compute_cls(best_dict)
    print(f"\nDerived (best fit):")
    print(f"  r_s(z*)     = {r_s_best:.4f} Mpc")

    # Save results
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / 'output')

    np.save(f'{output_dir}/mtdf_mcmc_samples.npy', samples)
    np.save(f'{output_dir}/mtdf_mcmc_logprobs.npy', log_probs)

    # Write summary
    with open(f'{output_dir}/mtdf_mcmc_results.txt', 'w') as f:
        f.write("# MTDF+EFE MCMC Results\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Walkers: {nwalkers}, Steps: {nsteps}\n")
        f.write("# ----------------------------------------\n\n")

        f.write("Best fit:\n")
        f.write(f"  chi2 = {best_chi2:.2f}\n")
        for name, val in zip(PARAM_NAMES, best_params):
            f.write(f"  {name:12s} = {val:.6f}\n")
        f.write(f"  r_s(z*)     = {r_s_best:.4f} Mpc\n\n")

        f.write("Posterior statistics (median +upper -lower 68%):\n")
        for i, name in enumerate(PARAM_NAMES):
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            f.write(f"  {name:12s} = {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}\n")

    print(f"\nResults saved to: {output_dir}/mtdf_mcmc_results.txt")

    return sampler, samples, log_probs


if __name__ == '__main__':
    # Short test run
    run_mcmc(nwalkers=16, nsteps=200)
