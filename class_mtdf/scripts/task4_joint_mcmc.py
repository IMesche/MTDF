#!/usr/bin/env python3
"""
Task 4: Joint MCMC with Planck TT + DESI BAO + SH0ES H0 prior

Combines:
1. Planck 2018 TT-only likelihood
2. DESI Y1 BAO measurements (DV/rd, DM/rd, DH/rd)
3. SH0ES H0 prior: 73.04 ± 1.04 km/s/Mpc
"""

import numpy as np
import emcee
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'cobaya_packages/code/planck/clipy')
import classy

# =============================================================================
# Constants
# =============================================================================
C_LIGHT = 299792.458  # km/s

# =============================================================================
# Parameter definitions
# =============================================================================

PARAM_NAMES = ['H0', 'omega_b', 'omega_cdm', 'n_s', 'logA', 'tau_reio', 'mtdf_k_f']
PARAM_PRIORS = {
    'H0':        [60.0, 80.0],
    'omega_b':   [0.020, 0.024],
    'omega_cdm': [0.10, 0.16],
    'n_s':       [0.90, 1.02],
    'logA':      [2.5, 3.6],
    'tau_reio':  [0.03, 0.10],
    'mtdf_k_f':  [0.0, 2.0],
}

# Fixed MTDF parameters
MTDF_ALPHA = 1.30
MTDF_BETA_EOS = 0.573
MTDF_Z_T = 0.74

# SH0ES H0 prior
H0_SHOES = 73.04
H0_SHOES_ERR = 1.04

# =============================================================================
# Data loading
# =============================================================================

def load_desi_bao():
    """Load DESI Y1 BAO data."""
    data_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'bao_desi')

    mean_path = f'{data_dir}/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt'
    cov_path = f'{data_dir}/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt'

    z_eff = []
    obs_values = []
    obs_types = []

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

    z_eff = np.array(z_eff)
    obs_values = np.array(obs_values)

    # Load covariance
    cov_matrix = np.loadtxt(cov_path)

    return z_eff, obs_values, obs_types, cov_matrix

# =============================================================================
# Likelihood functions
# =============================================================================

_planck_lkl = None

def init_planck_likelihood():
    """Initialize Planck TT likelihood."""
    global _planck_lkl
    if _planck_lkl is None:
        import clipy
        clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik'
        if os.path.exists(clik_path):
            _planck_lkl = clipy.clik(clik_path)
            print(f"Loaded TT-only likelihood")
        else:
            clik_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik'
            _planck_lkl = clipy.clik(clik_path)
            print(f"Loaded TTTEEE likelihood")
    return _planck_lkl


def compute_cls_and_distances(params_dict, ell_max=2600):
    """Compute Cls and distance quantities using MTDF CLASS."""
    cosmo = classy.Class()

    A_s = np.exp(params_dict['logA']) * 1e-10

    class_params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': ell_max,
        'P_k_max_h/Mpc': 1.0,
        'omega_b': params_dict['omega_b'],
        'omega_cdm': params_dict['omega_cdm'],
        'H0': params_dict['H0'],
        'tau_reio': params_dict['tau_reio'],
        'A_s': A_s,
        'n_s': params_dict['n_s'],
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
    T_cmb = 2.7255e6
    factor = T_cmb**2

    cl_tt = cls_dict['tt'][:ell_max+1] * factor
    cl_ee = cls_dict['ee'][:ell_max+1] * factor
    cl_te = cls_dict['te'][:ell_max+1] * factor

    # Get background for distances
    bg = cosmo.get_background()

    # Get r_d (sound horizon at drag epoch)
    # Find z_drag ~ 1060
    idx_drag = np.argmin(np.abs(bg['z'] - 1060))
    r_d = bg['comov.snd.hrz.'][idx_drag]

    # Get H0 in km/s/Mpc
    H0 = params_dict['H0']

    cosmo.struct_cleanup()
    cosmo.empty()

    return cl_tt, cl_ee, cl_te, bg, r_d, H0


def compute_bao_observables(z_arr, obs_types, bg, r_d, H0):
    """Compute BAO observables at given redshifts."""
    predictions = []

    for z, obs_type in zip(z_arr, obs_types):
        # Interpolate background to this z
        idx = np.argmin(np.abs(bg['z'] - z))

        # Comoving distance D_M
        D_M = bg['comov. dist.'][idx]

        # H(z)
        H_z = bg['H [1/Mpc]'][idx] * C_LIGHT  # Convert from 1/Mpc to km/s/Mpc

        # D_H = c/H(z)
        D_H = C_LIGHT / H_z

        # D_V = (D_M^2 * c*z / H(z))^(1/3)
        D_V = (D_M**2 * C_LIGHT * z / H_z)**(1/3)

        if 'DV' in obs_type:
            predictions.append(D_V / r_d)
        elif 'DM' in obs_type:
            predictions.append(D_M / r_d)
        elif 'DH' in obs_type:
            predictions.append(D_H / r_d)
        else:
            predictions.append(np.nan)

    return np.array(predictions)


def log_likelihood_planck(cl_tt, cl_ee, cl_te):
    """Compute Planck TT log-likelihood."""
    lkl = init_planck_likelihood()
    lmax = lkl.get_lmax()

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
            [1.0]
        ])

    logl = float(lkl(input_vec))
    return logl


def log_likelihood_bao(z_arr, obs_values, obs_types, cov_matrix, bg, r_d, H0):
    """Compute BAO chi-squared."""
    predictions = compute_bao_observables(z_arr, obs_types, bg, r_d, H0)

    residual = obs_values - predictions

    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except:
        reg = 1e-10 * np.trace(cov_matrix) / len(obs_values)
        cov_inv = np.linalg.inv(cov_matrix + reg * np.eye(len(obs_values)))

    chi2 = float(residual @ cov_inv @ residual)
    return -0.5 * chi2


def log_likelihood_h0(H0):
    """Compute SH0ES H0 prior log-likelihood."""
    chi2 = ((H0 - H0_SHOES) / H0_SHOES_ERR)**2
    return -0.5 * chi2


def log_prior(theta):
    """Flat priors on all parameters."""
    for name, val in zip(PARAM_NAMES, theta):
        pmin, pmax = PARAM_PRIORS[name]
        if val < pmin or val > pmax:
            return -np.inf
    return 0.0


def log_probability(theta, z_bao, obs_bao, types_bao, cov_bao, use_planck=True, use_bao=True, use_h0=True):
    """Combined log-probability."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    params_dict = {name: val for name, val in zip(PARAM_NAMES, theta)}

    try:
        cl_tt, cl_ee, cl_te, bg, r_d, H0 = compute_cls_and_distances(params_dict)

        logl = 0.0

        if use_planck:
            logl += log_likelihood_planck(cl_tt, cl_ee, cl_te)

        if use_bao:
            logl += log_likelihood_bao(z_bao, obs_bao, types_bao, cov_bao, bg, r_d, H0)

        if use_h0:
            logl += log_likelihood_h0(H0)

        if not np.isfinite(logl):
            return -np.inf

        return lp + logl

    except Exception as e:
        return -np.inf


# =============================================================================
# MCMC Runner
# =============================================================================

def run_mcmc(n_walkers=16, n_burn=50, n_steps=100, use_planck=True, use_bao=True, use_h0=True):
    """Run MCMC with specified likelihoods."""
    print("=" * 70)
    print("Task 4: Joint MCMC")
    print("=" * 70)
    print(f"\nLikelihoods enabled:")
    print(f"  Planck TT:  {use_planck}")
    print(f"  DESI BAO:   {use_bao}")
    print(f"  SH0ES H0:   {use_h0}")

    # Load BAO data
    z_bao, obs_bao, types_bao, cov_bao = load_desi_bao()
    print(f"\nLoaded {len(z_bao)} BAO data points")

    # Initialize Planck
    if use_planck:
        init_planck_likelihood()

    # Initial positions
    ndim = len(PARAM_NAMES)

    # Start near TT-only best-fit with some k_f
    p0_center = np.array([67.68, 0.02208, 0.1198, 0.9642, 3.102, 0.084, 0.5])

    # Add small scatter
    p0 = p0_center + 0.01 * np.abs(p0_center) * np.random.randn(n_walkers, ndim)

    # Ensure within priors
    for i, name in enumerate(PARAM_NAMES):
        pmin, pmax = PARAM_PRIORS[name]
        p0[:, i] = np.clip(p0[:, i], pmin + 0.001*(pmax-pmin), pmax - 0.001*(pmax-pmin))

    # Create sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(z_bao, obs_bao, types_bao, cov_bao, use_planck, use_bao, use_h0)
    )

    # Burn-in
    print(f"\nRunning burn-in ({n_burn} steps)...")
    state = sampler.run_mcmc(p0, n_burn, progress=True)
    sampler.reset()

    # Production
    print(f"Running production ({n_steps} steps)...")
    sampler.run_mcmc(state, n_steps, progress=True)

    # Get samples
    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    # Best fit
    best_idx = np.argmax(log_probs)
    best_params = samples[best_idx]
    best_logl = log_probs[best_idx]

    # Compute chi2 breakdown for best-fit
    params_dict = {name: val for name, val in zip(PARAM_NAMES, best_params)}
    cl_tt, cl_ee, cl_te, bg, r_d, H0 = compute_cls_and_distances(params_dict)

    chi2_planck = -2 * log_likelihood_planck(cl_tt, cl_ee, cl_te) if use_planck else 0
    chi2_bao = -2 * log_likelihood_bao(z_bao, obs_bao, types_bao, cov_bao, bg, r_d, H0) if use_bao else 0
    chi2_h0 = -2 * log_likelihood_h0(H0) if use_h0 else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nBest-fit parameters:")
    for name, val in zip(PARAM_NAMES, best_params):
        print(f"  {name:12s} = {val:.6f}")
    print(f"  r_d          = {r_d:.4f} Mpc")

    print(f"\nChi-squared breakdown:")
    print(f"  Planck TT: {chi2_planck:.2f}")
    print(f"  DESI BAO:  {chi2_bao:.2f} (12 data points)")
    print(f"  SH0ES H0:  {chi2_h0:.2f} (1 data point)")
    print(f"  Total:     {chi2_planck + chi2_bao + chi2_h0:.2f}")

    # Posterior statistics
    print(f"\nPosterior statistics (median +upper -lower 68%):")
    for i, name in enumerate(PARAM_NAMES):
        q = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name:12s} = {q[1]:.4f} +{q[2]-q[1]:.4f} -{q[1]-q[0]:.4f}")

    # Save results
    output_dir = str(Path(__file__).parent.parent / 'output')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine filename suffix
    suffix = ""
    if use_planck:
        suffix += "_TT"
    if use_bao:
        suffix += "_BAO"
    if use_h0:
        suffix += "_H0"

    np.save(f"{output_dir}/task4_samples{suffix}.npy", samples)
    np.save(f"{output_dir}/task4_logprob{suffix}.npy", log_probs)

    # Write summary
    with open(f"{output_dir}/task4_results{suffix}.txt", 'w') as f:
        f.write(f"# Task 4 Joint MCMC Results\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Likelihoods: Planck={use_planck}, BAO={use_bao}, H0={use_h0}\n")
        f.write(f"# Walkers: {n_walkers}, Burn-in: {n_burn}, Production: {n_steps}\n\n")

        f.write(f"Best-fit (log_prob = {best_logl:.2f}):\n")
        for name, val in zip(PARAM_NAMES, best_params):
            f.write(f"  {name} = {val:.6f}\n")
        f.write(f"  r_d = {r_d:.4f} Mpc\n\n")

        f.write(f"Chi-squared:\n")
        f.write(f"  Planck TT: {chi2_planck:.2f}\n")
        f.write(f"  DESI BAO: {chi2_bao:.2f}\n")
        f.write(f"  SH0ES H0: {chi2_h0:.2f}\n")
        f.write(f"  Total: {chi2_planck + chi2_bao + chi2_h0:.2f}\n\n")

        f.write(f"Posterior (median +/- 68%):\n")
        for i, name in enumerate(PARAM_NAMES):
            q = np.percentile(samples[:, i], [16, 50, 84])
            f.write(f"  {name} = {q[1]:.4f} +{q[2]-q[1]:.4f} -{q[1]-q[0]:.4f}\n")

    print(f"\nResults saved to {output_dir}/task4_*{suffix}.*")

    return samples, log_probs, best_params


if __name__ == '__main__':
    # Run joint MCMC with all likelihoods
    samples, log_probs, best = run_mcmc(
        n_walkers=16,
        n_burn=50,
        n_steps=100,
        use_planck=True,
        use_bao=True,
        use_h0=True
    )
