#!/usr/bin/env python3
"""
Task 4 (LCDM baseline): Joint Planck TT + BAO + H0 prior MCMC

Standard LCDM cosmology (no MTDF) for comparison with MTDF results.
Uses the same likelihood combination and priors.
"""

import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'cobaya_packages/code/planck/clipy')
import classy
import emcee
from tqdm import tqdm

# =============================================================================
# Constants
# =============================================================================
C_LIGHT = 299792.458

# SH0ES H0 prior
H0_SHOES = 73.04
H0_SHOES_ERR = 1.04

# Parameter priors (same as MTDF run, minus k_f)
PRIORS = {
    'H0': (60.0, 80.0),
    'omega_b': (0.019, 0.025),
    'omega_cdm': (0.10, 0.14),
    'n_s': (0.9, 1.05),
    'logA': (2.9, 3.2),
    'tau_reio': (0.04, 0.12),
}

PARAM_NAMES = ['H0', 'omega_b', 'omega_cdm', 'n_s', 'logA', 'tau_reio']

# =============================================================================
# Data loading
# =============================================================================

def load_desi_bao():
    """Load DESI Y1 BAO data."""
    data_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'bao_desi')
    mean_path = f'{data_dir}/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
    cov_path = f'{data_dir}/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

    z_eff, obs_values, obs_types = [], [], []
    with open(mean_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                z_eff.append(float(parts[0]))
                obs_values.append(float(parts[1]))
                obs_types.append(parts[2])

    return np.array(z_eff), np.array(obs_values), obs_types, np.loadtxt(cov_path)

# =============================================================================
# Likelihood computation
# =============================================================================

_planck_lkl = None

def init_planck_likelihood():
    global _planck_lkl
    if _planck_lkl is None:
        import clipy
        clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik'
        _planck_lkl = clipy.clik(clik_path)
        print("Loaded TT-only likelihood")
    return _planck_lkl


def compute_model(params_dict):
    """Compute Cls and distances for given parameters (LCDM, no MTDF)."""
    cosmo = classy.Class()

    A_s = np.exp(params_dict['logA']) * 1e-10

    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': 2600,
        'P_k_max_h/Mpc': 1.0,
        'omega_b': params_dict['omega_b'],
        'omega_cdm': params_dict['omega_cdm'],
        'H0': params_dict['H0'],
        'tau_reio': params_dict['tau_reio'],
        'A_s': A_s,
        'n_s': params_dict['n_s'],
        # NO MTDF parameters - vanilla LCDM
    }

    cosmo.set(params)
    cosmo.compute()

    cls = cosmo.lensed_cl(2600)
    T_cmb = 2.7255e6
    factor = T_cmb**2

    cl_tt = cls['tt'][:2601] * factor
    cl_ee = cls['ee'][:2601] * factor
    cl_te = cls['te'][:2601] * factor

    bg = cosmo.get_background()
    idx_drag = np.argmin(np.abs(bg['z'] - 1060))
    r_d = bg['comov.snd.hrz.'][idx_drag]

    cosmo.struct_cleanup()
    cosmo.empty()

    return cl_tt, cl_ee, cl_te, bg, r_d


def compute_bao_observables(z_arr, obs_types, bg, r_d, H0):
    """Compute BAO observables from background."""
    predictions = []
    for z, obs_type in zip(z_arr, obs_types):
        idx = np.argmin(np.abs(bg['z'] - z))
        D_M = bg['comov. dist.'][idx]
        H_z = bg['H [1/Mpc]'][idx] * C_LIGHT
        D_H = C_LIGHT / H_z
        D_V = (D_M**2 * C_LIGHT * z / H_z)**(1/3)

        if 'DV' in obs_type:
            predictions.append(D_V / r_d)
        elif 'DM' in obs_type:
            predictions.append(D_M / r_d)
        elif 'DH' in obs_type:
            predictions.append(D_H / r_d)

    return np.array(predictions)


def log_likelihood_planck(cl_tt, cl_ee, cl_te):
    """Compute Planck TT log-likelihood."""
    lkl = init_planck_likelihood()
    lmax = lkl.get_lmax()

    if lmax[1] < 0:  # TT-only
        lmax_tt = lmax[0]
        input_vec = np.concatenate([cl_tt[:lmax_tt+1], [1.0]])
    else:  # TTTEEE
        lmax_tt, lmax_ee, lmax_bb, lmax_te = lmax[0], lmax[1], lmax[2], lmax[3]
        cl_bb = np.zeros(lmax_bb+1) if lmax_bb >= 0 else np.array([])
        input_vec = np.concatenate([
            cl_tt[:lmax_tt+1],
            cl_ee[:lmax_ee+1],
            cl_bb,
            cl_te[:lmax_te+1],
            [1.0]
        ])

    logl = float(lkl(input_vec))
    return logl


def log_likelihood_bao(z_arr, obs_values, obs_types, cov_matrix, bg, r_d, H0):
    """Compute BAO chi-squared."""
    predictions = compute_bao_observables(z_arr, obs_types, bg, r_d, H0)

    residual = obs_values - predictions
    cov_inv = np.linalg.inv(cov_matrix)
    chi2 = float(residual @ cov_inv @ residual)

    return -0.5 * chi2


def log_likelihood_h0(H0):
    """Compute SH0ES H0 chi-squared."""
    chi2 = ((H0 - H0_SHOES) / H0_SHOES_ERR)**2
    return -0.5 * chi2


def log_prior(theta):
    """Uniform priors on all parameters."""
    for i, name in enumerate(PARAM_NAMES):
        low, high = PRIORS[name]
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.0


def log_probability(theta, z_bao, obs_bao, types_bao, cov_bao,
                    use_planck=True, use_bao=True, use_h0=True):
    """Total log-probability."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    params_dict = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}

    try:
        cl_tt, cl_ee, cl_te, bg, r_d = compute_model(params_dict)

        ll = 0.0
        if use_planck:
            ll += log_likelihood_planck(cl_tt, cl_ee, cl_te)
        if use_bao:
            ll += log_likelihood_bao(z_bao, obs_bao, types_bao, cov_bao,
                                     bg, r_d, params_dict['H0'])
        if use_h0:
            ll += log_likelihood_h0(params_dict['H0'])

        return lp + ll

    except Exception as e:
        return -np.inf


# =============================================================================
# MCMC
# =============================================================================

def run_mcmc(z_bao, obs_bao, types_bao, cov_bao,
             use_planck=True, use_bao=True, use_h0=True,
             nwalkers=16, nburn=50, nprod=100):
    """Run MCMC."""

    ndim = len(PARAM_NAMES)

    # Initialize walkers near Planck best-fit
    p0_center = np.array([67.5, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])
    p0 = p0_center + 0.01 * p0_center * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(z_bao, obs_bao, types_bao, cov_bao, use_planck, use_bao, use_h0)
    )

    # Burn-in
    print(f"\nRunning burn-in ({nburn} steps)...")
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    # Production
    print(f"Running production ({nprod} steps)...")
    sampler.run_mcmc(state, nprod, progress=True)

    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    # Best-fit
    best_idx = np.argmax(log_probs)
    best_params = samples[best_idx]

    # Compute chi2 breakdown for best-fit
    params_dict = {name: best_params[i] for i, name in enumerate(PARAM_NAMES)}
    cl_tt, cl_ee, cl_te, bg, r_d = compute_model(params_dict)

    chi2_planck = -2 * log_likelihood_planck(cl_tt, cl_ee, cl_te) if use_planck else 0
    chi2_bao = -2 * log_likelihood_bao(z_bao, obs_bao, types_bao, cov_bao,
                                        bg, r_d, params_dict['H0']) if use_bao else 0
    chi2_h0 = -2 * log_likelihood_h0(params_dict['H0']) if use_h0 else 0

    return samples, log_probs, best_params, r_d, chi2_planck, chi2_bao, chi2_h0


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Task 4 (LCDM Baseline): Joint MCMC")
    print("=" * 70)

    use_planck = True
    use_bao = True
    use_h0 = True

    print(f"\nLikelihoods enabled:")
    print(f"  Planck TT:  {use_planck}")
    print(f"  DESI BAO:   {use_bao}")
    print(f"  SH0ES H0:   {use_h0}")

    # Load data
    z_bao, obs_bao, types_bao, cov_bao = load_desi_bao()
    print(f"Loaded {len(z_bao)} BAO data points")

    # Initialize Planck
    init_planck_likelihood()

    # Run MCMC
    samples, log_probs, best, r_d, chi2_planck, chi2_bao, chi2_h0 = run_mcmc(
        z_bao, obs_bao, types_bao, cov_bao,
        use_planck=use_planck, use_bao=use_bao, use_h0=use_h0,
        nwalkers=16, nburn=50, nprod=100
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS (LCDM - No MTDF)")
    print("=" * 70)

    print("\nBest-fit parameters:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:12s} = {best[i]:.6f}")
    print(f"  {'r_d':12s} = {r_d:.4f} Mpc")

    chi2_total = chi2_planck + chi2_bao + chi2_h0
    print(f"\nChi-squared breakdown:")
    print(f"  Planck TT: {chi2_planck:.2f}")
    print(f"  DESI BAO:  {chi2_bao:.2f} ({len(z_bao)} data points)")
    print(f"  SH0ES H0:  {chi2_h0:.2f} (1 data point)")
    print(f"  Total:     {chi2_total:.2f}")

    # Posterior statistics
    print(f"\nPosterior statistics (median +upper -lower 68%):")
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        med = percentiles[1, i]
        lower = med - percentiles[0, i]
        upper = percentiles[2, i] - med
        print(f"  {name:12s} = {med:.4f} +{upper:.4f} -{lower:.4f}")

    # Save results
    output_dir = str(Path(__file__).parent.parent / 'output')
    suffix = "TT_BAO_H0"

    np.save(f"{output_dir}/lcdm_chain_{suffix}.npy", samples)
    np.savez(f"{output_dir}/lcdm_best_{suffix}.npz",
             best=best, param_names=PARAM_NAMES,
             chi2_planck=chi2_planck, chi2_bao=chi2_bao, chi2_h0=chi2_h0,
             r_d=r_d)

    # Save summary text file
    with open(f"{output_dir}/lcdm_joint_TT_BAO_H0_results.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LCDM Joint MCMC Results (Planck TT + DESI BAO + SH0ES H0)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Best-fit parameters:\n")
        for i, name in enumerate(PARAM_NAMES):
            f.write(f"  {name:12s} = {best[i]:.6f}\n")
        f.write(f"  {'r_d':12s} = {r_d:.4f} Mpc\n\n")

        f.write("Chi-squared breakdown:\n")
        f.write(f"  Planck TT: {chi2_planck:.2f}\n")
        f.write(f"  DESI BAO:  {chi2_bao:.2f} ({len(z_bao)} data points)\n")
        f.write(f"  SH0ES H0:  {chi2_h0:.2f} (1 data point)\n")
        f.write(f"  Total:     {chi2_total:.2f}\n\n")

        f.write("Posterior statistics (median +upper -lower 68%):\n")
        for i, name in enumerate(PARAM_NAMES):
            med = percentiles[1, i]
            lower = med - percentiles[0, i]
            upper = percentiles[2, i] - med
            f.write(f"  {name:12s} = {med:.4f} +{upper:.4f} -{lower:.4f}\n")

    print(f"\nResults saved to {output_dir}/lcdm_*_{suffix}.*")

    return True


if __name__ == '__main__':
    success = main()
