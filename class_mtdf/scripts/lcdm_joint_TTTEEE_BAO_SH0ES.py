#!/usr/bin/env python3
"""
LCDM Joint MCMC: Planck TTTEEE + DESI BAO + SH0ES H0 Prior

Full constraint on standard LCDM including SH0ES local H0 measurement.
Uses plik_lite TTTEEE + low-l TT (commander) + low-l EE (simall).
SH0ES prior: H0 = 73.04 +/- 1.04 km/s/Mpc (Riess et al. 2022)

This run quantifies the H0 tension in LCDM for comparison with MTDF.
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

# SH0ES prior (Riess et al. 2022)
SHOES_H0 = 73.04
SHOES_SIGMA = 1.04

# Parameter priors (no k_f for LCDM)
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

_planck_ttteee = None
_planck_lowl_tt = None
_planck_lowl_ee = None

def init_planck_likelihoods():
    """Initialize all Planck likelihoods."""
    global _planck_ttteee, _planck_lowl_tt, _planck_lowl_ee
    import clipy

    if _planck_ttteee is None:
        ttteee_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TTTEEE.clik'
        _planck_ttteee = clipy.clik(ttteee_path)
        print("Loaded high-l TTTEEE likelihood")

    if _planck_lowl_tt is None:
        lowl_tt_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik'
        _planck_lowl_tt = clipy.clik(lowl_tt_path)
        print("Loaded low-l TT likelihood (commander)")

    if _planck_lowl_ee is None:
        lowl_ee_path = 'cobaya_packages/data/planck_2018/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik'
        _planck_lowl_ee = clipy.clik(lowl_ee_path)
        print("Loaded low-l EE likelihood (simall)")

    return _planck_ttteee, _planck_lowl_tt, _planck_lowl_ee


def compute_model(params_dict):
    """Compute Cls and distances for given parameters (LCDM)."""
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
        # No MTDF parameters - vanilla LCDM
    }

    cosmo.set(params)
    cosmo.compute()

    cls = cosmo.lensed_cl(2600)
    T_cmb = 2.7255e6
    factor = T_cmb**2

    cl_tt = cls['tt'][:2601] * factor
    cl_ee = cls['ee'][:2601] * factor
    cl_te = cls['te'][:2601] * factor
    cl_bb = cls['bb'][:2601] * factor if 'bb' in cls else np.zeros(2601)

    bg = cosmo.get_background()
    idx_drag = np.argmin(np.abs(bg['z'] - 1060))
    r_d = bg['comov.snd.hrz.'][idx_drag]

    idx_rec = np.argmin(np.abs(bg['z'] - 1089))
    r_s = bg['comov.snd.hrz.'][idx_rec]

    cosmo.struct_cleanup()
    cosmo.empty()

    return cl_tt, cl_ee, cl_te, cl_bb, bg, r_d, r_s


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


def log_likelihood_ttteee(cl_tt, cl_ee, cl_te, cl_bb):
    """Compute high-l Planck TTTEEE log-likelihood."""
    lkl = _planck_ttteee
    lmax = lkl.get_lmax()

    lmax_tt = lmax[0]
    lmax_ee = lmax[1]
    lmax_bb = lmax[2]
    lmax_te = lmax[3]

    cl_bb_input = np.zeros(lmax_bb+1) if lmax_bb >= 0 else np.array([])

    input_vec = np.concatenate([
        cl_tt[:lmax_tt+1],
        cl_ee[:lmax_ee+1],
        cl_bb_input,
        cl_te[:lmax_te+1],
        [1.0]
    ])

    return float(lkl(input_vec))


def log_likelihood_lowl_tt(cl_tt):
    """Compute low-l TT (commander) log-likelihood."""
    lkl = _planck_lowl_tt
    lmax = lkl.get_lmax()
    lmax_tt = lmax[0]

    input_vec = np.concatenate([cl_tt[:lmax_tt+1], [1.0]])
    return float(lkl(input_vec))


def log_likelihood_lowl_ee(cl_ee):
    """Compute low-l EE (simall) log-likelihood."""
    lkl = _planck_lowl_ee
    lmax = lkl.get_lmax()

    lmax_ee = lmax[1] if lmax[1] >= 0 else lmax[0]

    input_vec = np.concatenate([cl_ee[:lmax_ee+1], [1.0]])
    return float(lkl(input_vec))


def log_likelihood_bao(z_arr, obs_values, obs_types, cov_matrix, bg, r_d, H0):
    """Compute BAO log-likelihood."""
    predictions = compute_bao_observables(z_arr, obs_types, bg, r_d, H0)

    residual = obs_values - predictions
    cov_inv = np.linalg.inv(cov_matrix)
    chi2 = float(residual @ cov_inv @ residual)

    return -0.5 * chi2


def log_likelihood_shoes(H0):
    """Compute SH0ES H0 prior log-likelihood.

    SH0ES prior: H0 = 73.04 +/- 1.04 km/s/Mpc (Riess et al. 2022)
    """
    chi2 = ((H0 - SHOES_H0) / SHOES_SIGMA) ** 2
    return -0.5 * chi2


def log_prior(theta):
    """Uniform priors on all parameters."""
    for i, name in enumerate(PARAM_NAMES):
        low, high = PRIORS[name]
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.0


def log_probability(theta, z_bao, obs_bao, types_bao, cov_bao):
    """Total log-probability (TTTEEE + low-l + BAO + SH0ES)."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    params_dict = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}

    try:
        cl_tt, cl_ee, cl_te, cl_bb, bg, r_d, r_s = compute_model(params_dict)

        ll = 0.0
        ll += log_likelihood_ttteee(cl_tt, cl_ee, cl_te, cl_bb)
        ll += log_likelihood_lowl_tt(cl_tt)
        ll += log_likelihood_lowl_ee(cl_ee)
        ll += log_likelihood_bao(z_bao, obs_bao, types_bao, cov_bao,
                                  bg, r_d, params_dict['H0'])

        # SH0ES H0 prior
        ll += log_likelihood_shoes(params_dict['H0'])

        return lp + ll

    except Exception as e:
        return -np.inf


# =============================================================================
# MCMC
# =============================================================================

def run_mcmc(z_bao, obs_bao, types_bao, cov_bao,
             nwalkers=16, nburn=50, nprod=100):
    """Run MCMC."""

    ndim = len(PARAM_NAMES)

    # Initialize walkers - start H0 between CMB and SH0ES
    p0_center = np.array([70.0, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])
    p0 = p0_center + 0.01 * p0_center * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(z_bao, obs_bao, types_bao, cov_bao)
    )

    print(f"\nRunning burn-in ({nburn} steps)...")
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()

    print(f"Running production ({nprod} steps)...")
    sampler.run_mcmc(state, nprod, progress=True)

    samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)

    best_idx = np.argmax(log_probs)
    best_params = samples[best_idx]

    params_dict = {name: best_params[i] for i, name in enumerate(PARAM_NAMES)}
    cl_tt, cl_ee, cl_te, cl_bb, bg, r_d, r_s = compute_model(params_dict)

    chi2_ttteee = -2 * log_likelihood_ttteee(cl_tt, cl_ee, cl_te, cl_bb)
    chi2_lowl_tt = -2 * log_likelihood_lowl_tt(cl_tt)
    chi2_lowl_ee = -2 * log_likelihood_lowl_ee(cl_ee)
    chi2_bao = -2 * log_likelihood_bao(z_bao, obs_bao, types_bao, cov_bao,
                                        bg, r_d, params_dict['H0'])
    chi2_shoes = -2 * log_likelihood_shoes(params_dict['H0'])

    return samples, log_probs, best_params, r_d, r_s, chi2_ttteee, chi2_lowl_tt, chi2_lowl_ee, chi2_bao, chi2_shoes


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("LCDM Joint MCMC: Planck TTTEEE + DESI BAO + SH0ES")
    print("=" * 70)

    print(f"\nLikelihoods enabled:")
    print(f"  Planck high-l TTTEEE: True")
    print(f"  Planck low-l TT:      True")
    print(f"  Planck low-l EE:      True")
    print(f"  DESI BAO:             True")
    print(f"  SH0ES H0:             True (H0 = {SHOES_H0} +/- {SHOES_SIGMA} km/s/Mpc)")

    z_bao, obs_bao, types_bao, cov_bao = load_desi_bao()
    print(f"\nLoaded {len(z_bao)} BAO data points")

    init_planck_likelihoods()

    samples, log_probs, best, r_d, r_s, chi2_ttteee, chi2_lowl_tt, chi2_lowl_ee, chi2_bao, chi2_shoes = run_mcmc(
        z_bao, obs_bao, types_bao, cov_bao,
        nwalkers=16, nburn=50, nprod=100
    )

    print("\n" + "=" * 70)
    print("RESULTS (LCDM)")
    print("=" * 70)

    print("\nBest-fit parameters:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:12s} = {best[i]:.6f}")
    print(f"  {'r_d':12s} = {r_d:.4f} Mpc")
    print(f"  {'r_s':12s} = {r_s:.4f} Mpc")

    chi2_planck_total = chi2_ttteee + chi2_lowl_tt + chi2_lowl_ee
    chi2_total = chi2_planck_total + chi2_bao + chi2_shoes

    print(f"\nChi-squared breakdown:")
    print(f"  Planck high-l TTTEEE: {chi2_ttteee:.2f}")
    print(f"  Planck low-l TT:      {chi2_lowl_tt:.2f}")
    print(f"  Planck low-l EE:      {chi2_lowl_ee:.2f}")
    print(f"  Planck Total:         {chi2_planck_total:.2f}")
    print(f"  DESI BAO:             {chi2_bao:.2f} ({len(z_bao)} data points)")
    print(f"  SH0ES H0:             {chi2_shoes:.2f} (1 data point)")
    print(f"  Total:                {chi2_total:.2f}")

    # Compute H0 tension
    H0_best = best[0]
    tension_sigma = abs(H0_best - SHOES_H0) / SHOES_SIGMA
    print(f"\n  H0 = {H0_best:.2f} km/s/Mpc (tension with SH0ES: {tension_sigma:.1f}σ)")

    print(f"\nPosterior statistics (median +upper -lower 68%):")
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        med = percentiles[1, i]
        lower = med - percentiles[0, i]
        upper = percentiles[2, i] - med
        print(f"  {name:12s} = {med:.4f} +{upper:.4f} -{lower:.4f}")

    output_dir = str(Path(__file__).parent.parent / 'output')

    np.save(f"{output_dir}/lcdm_chain_TTTEEE_BAO_SH0ES.npy", samples)
    np.savez(f"{output_dir}/lcdm_best_TTTEEE_BAO_SH0ES.npz",
             best=best, param_names=PARAM_NAMES,
             chi2_ttteee=chi2_ttteee, chi2_lowl_tt=chi2_lowl_tt,
             chi2_lowl_ee=chi2_lowl_ee, chi2_bao=chi2_bao,
             chi2_shoes=chi2_shoes,
             r_d=r_d, r_s=r_s,
             shoes_h0=SHOES_H0, shoes_sigma=SHOES_SIGMA)

    with open(f"{output_dir}/lcdm_joint_TTTEEE_BAO_SH0ES_results.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("LCDM Joint MCMC Results (Planck TTTEEE + DESI BAO + SH0ES)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"SH0ES prior: H0 = {SHOES_H0} +/- {SHOES_SIGMA} km/s/Mpc\n\n")

        f.write("Best-fit parameters:\n")
        for i, name in enumerate(PARAM_NAMES):
            f.write(f"  {name:12s} = {best[i]:.6f}\n")
        f.write(f"  {'r_d':12s} = {r_d:.4f} Mpc\n")
        f.write(f"  {'r_s':12s} = {r_s:.4f} Mpc\n\n")

        f.write("Chi-squared breakdown:\n")
        f.write(f"  Planck high-l TTTEEE: {chi2_ttteee:.2f}\n")
        f.write(f"  Planck low-l TT:      {chi2_lowl_tt:.2f}\n")
        f.write(f"  Planck low-l EE:      {chi2_lowl_ee:.2f}\n")
        f.write(f"  Planck Total:         {chi2_planck_total:.2f}\n")
        f.write(f"  DESI BAO:             {chi2_bao:.2f} ({len(z_bao)} data points)\n")
        f.write(f"  SH0ES H0:             {chi2_shoes:.2f} (1 data point)\n")
        f.write(f"  Total:                {chi2_total:.2f}\n\n")

        f.write(f"H0 tension: {tension_sigma:.1f}σ from SH0ES\n\n")

        f.write("Posterior statistics (median +upper -lower 68%):\n")
        for i, name in enumerate(PARAM_NAMES):
            med = percentiles[1, i]
            lower = med - percentiles[0, i]
            upper = percentiles[2, i] - med
            f.write(f"  {name:12s} = {med:.4f} +{upper:.4f} -{lower:.4f}\n")

    print(f"\nResults saved to {output_dir}/lcdm_*_TTTEEE_BAO_SH0ES.*")

    return True


if __name__ == '__main__':
    success = main()
