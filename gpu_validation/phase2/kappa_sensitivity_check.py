#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74
#
# Kappa sensitivity check: compare chi2 at KAPPA=0.00102 vs KAPPA=0.001101
# This is a read-only investigative script; it modifies no existing files.

"""
Forward likelihood evaluation comparing two kappa values.

Evaluates the MTDF correction magnitudes and (where data is available)
the chi-squared impact of shifting KAPPA from 0.00102 to 0.001101.

Code paths tested:
  1. CMB correction layer (mtdf_correction_layer.py) -- uses KAPPA directly
  2. Late-universe H(z) and distances (late_universe_likelihood.py) -- uses k_f * KAPPA
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure gpu_validation is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Two kappa values to compare
# ---------------------------------------------------------------------------
KAPPA_A = 0.00102    # current best-fit
KAPPA_B = 0.001101   # alternative

# Fixed MTDF parameters (from the modules)
ALPHA = 1.30
K_RS_EFE = -0.215
BETA_EOS = 0.573
LAMBDA_MTDF = (1.0 - BETA_EOS)**2 / (1.0 + ALPHA)
F_KICK = LAMBDA_MTDF / 24.0

# Planck 2018 best-fit cosmology (from run_mcmc.py)
H0 = 67.36
OMEGA_M = 0.3153
OMEGA_B_H2 = 0.02237
OMEGA_CDM_H2 = 0.1200
Z_STAR = 1089.92

C_LIGHT = 299792.458  # km/s


# ---------------------------------------------------------------------------
# 1. CMB correction layer quantities (no external data needed)
# ---------------------------------------------------------------------------
def cmb_diagnostics(kappa):
    """Compute CMB correction layer diagnostics for a given kappa.

    Returns dict with:
      - sound_horizon_ratio
      - theta_s_ratio
      - isw_boost_ell2 (ISW boost at ell=2)
      - isw_boost_ell30 (ISW boost at ell=30)
      - stress_at_zstar (H stress correction at z_star)
    """
    from scipy.integrate import quad

    f_kick = F_KICK  # f_kick is derived from lambda_MTDF, independent of kappa

    # Sound horizon ratio: r_s_MTDF / r_s_LCDM = 1 + K_RS_EFE * f_kick
    rs_ratio = 1.0 + K_RS_EFE * f_kick

    # H(z) functions with this kappa
    def H_lcdm(z):
        Omega_L = 1.0 - OMEGA_M
        E = np.sqrt(OMEGA_M * (1 + z)**3 + Omega_L)
        return H0 * E

    def H_mtdf(z):
        Omega_L = 1.0 - OMEGA_M
        E = np.sqrt(OMEGA_M * (1 + z)**3 + Omega_L)
        stress = kappa * ALPHA * z / (1 + z)
        return H0 * E * (1 + stress)

    # Angular diameter distances to last scattering
    def da(H_func):
        D_C, _ = quad(lambda z: C_LIGHT / H_func(z), 0, Z_STAR, limit=300)
        return D_C / (1 + Z_STAR)

    D_A_lcdm = da(H_lcdm)
    D_A_mtdf = da(H_mtdf)
    da_ratio = D_A_mtdf / D_A_lcdm

    theta_ratio = rs_ratio / da_ratio

    # ISW boost: 1 + 2*kappa*alpha*exp(-ell/30)
    isw_ell2 = 1.0 + 2.0 * kappa * ALPHA * np.exp(-2.0 / 30.0)
    isw_ell30 = 1.0 + 2.0 * kappa * ALPHA * np.exp(-30.0 / 30.0)

    # Stress at z_star
    stress_zstar = kappa * ALPHA * Z_STAR / (1 + Z_STAR)

    return {
        'kappa': kappa,
        'sound_horizon_ratio': rs_ratio,
        'D_A_lcdm_Mpc': D_A_lcdm,
        'D_A_mtdf_Mpc': D_A_mtdf,
        'D_A_ratio': da_ratio,
        'theta_s_ratio': theta_ratio,
        'delta_theta_s_pct': (theta_ratio - 1) * 100,
        'isw_boost_ell2': isw_ell2,
        'isw_boost_ell30': isw_ell30,
        'stress_at_zstar': stress_zstar,
    }


# ---------------------------------------------------------------------------
# 2. Late-universe distance diagnostics (no external data needed)
# ---------------------------------------------------------------------------
def late_universe_diagnostics(kappa, k_f=1.0):
    """Compute late-universe distance diagnostics for a given kappa.

    Uses k_f * kappa in the stress correction, matching late_universe_likelihood.py.
    Default k_f=1.0 (full MTDF).

    Returns dict with H(z), D_M(z), D_V(z)/r_d at several redshifts.
    """
    z_probes = np.array([0.30, 0.51, 0.70, 0.93, 1.32, 2.33])

    def H_mtdf_late(z):
        Omega_L = 1.0 - OMEGA_M
        E = np.sqrt(OMEGA_M * (1 + z)**3 + Omega_L)
        stress = k_f * kappa * ALPHA * z / (1 + z)
        return H0 * E * (1 + stress)

    # Comoving distances via trapezoidal integration
    n_grid = 5000
    z_max = float(np.max(z_probes)) * 1.1 + 0.1
    z_grid = np.linspace(0, z_max, n_grid)
    H_grid = np.array([H_mtdf_late(z) for z in z_grid])
    integrand = C_LIGHT / H_grid
    dz = np.diff(z_grid)
    D_C_grid = np.zeros(n_grid)
    D_C_grid[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
    D_M = np.interp(z_probes, z_grid, D_C_grid)

    # Sound horizon (Aubourg et al. 2015)
    import math
    h = H0 / 100.0
    omega_m_h2 = OMEGA_B_H2 + OMEGA_CDM_H2
    r_d_lcdm = 55.154 * math.exp(-72.3 * (0.0 + 0.0006)**2) / \
               (omega_m_h2**0.25351 * OMEGA_B_H2**0.12807)
    r_d_mtdf = r_d_lcdm * (1.0 + k_f * K_RS_EFE * F_KICK)

    results = {'kappa': kappa, 'k_f': k_f, 'r_d_mtdf_Mpc': r_d_mtdf}
    for i, z in enumerate(z_probes):
        H_z = H_mtdf_late(z)
        D_H = C_LIGHT / H_z
        D_V = (D_M[i]**2 * C_LIGHT * z / H_z)**(1.0/3.0)
        results[f'z={z:.2f}'] = {
            'H_z': H_z,
            'D_M_Mpc': D_M[i],
            'D_H_Mpc': D_H,
            'D_V_over_rd': D_V / r_d_mtdf,
            'D_M_over_rd': D_M[i] / r_d_mtdf,
            'D_H_over_rd': D_H / r_d_mtdf,
        }
    return results


# ---------------------------------------------------------------------------
# 3. Synthetic CMB chi2 estimate (mock spectrum, no CosmoPower needed)
# ---------------------------------------------------------------------------
def cmb_chi2_estimate(kappa):
    """Estimate the CMB chi2 shift from the peak-shift effect.

    Uses a simple model: the correction shifts ell by delta_theta_s/theta_s.
    For a featureless power spectrum this gives zero delta-chi2, but for an
    oscillatory spectrum the fractional shift matters.

    We create a mock C_l with acoustic oscillations and compute the
    residual from shifting it by theta_s_ratio.
    """
    diag = cmb_diagnostics(kappa)
    ratio = diag['theta_s_ratio']

    # Mock D_l with acoustic peaks (Planck-like)
    ells = np.arange(2, 2509)
    # Rough envelope + oscillations matching Planck TT
    envelope = 6000.0 * (ells / 220.0)**(-0.1) * np.exp(-ells / 2000.0)
    peaks = 1.0 + 0.15 * np.cos(np.pi * ells / 302.0)  # ~302 ell spacing
    Dl_fid = envelope * peaks

    # MTDF-corrected: shift ell axis
    ells_shifted = ells * ratio
    Dl_mtdf = np.interp(ells_shifted, ells.astype(float), Dl_fid,
                         left=Dl_fid[0], right=Dl_fid[-1])

    # Apply ISW boost
    isw_boost = 1.0 + 2.0 * kappa * ALPHA * np.exp(-ells / 30.0)
    Dl_mtdf *= isw_boost

    # Simple diagonal chi2 with Planck-like noise
    # sigma ~ Dl / sqrt(2*ell+1) * f_sky^(-0.5), f_sky ~ 0.57
    sigma = Dl_fid / np.sqrt((2.0 * ells + 1.0) * 0.57)
    sigma = np.maximum(sigma, 1.0)  # floor

    residual = Dl_mtdf - Dl_fid
    chi2 = np.sum((residual / sigma)**2)
    return chi2, residual, sigma


# ---------------------------------------------------------------------------
# 4. Synthetic BAO chi2 estimate (mock data at DESI Y1 redshifts)
# ---------------------------------------------------------------------------
def bao_chi2_estimate(kappa, k_f=1.0):
    """Estimate BAO chi2 using LCDM predictions as 'data'.

    'Observed' values are computed at the fiducial kappa=0 (pure LCDM).
    'Model' values are computed at the given kappa. The difference
    measures the kappa-induced shift in BAO distances.
    """
    # DESI Y1 effective redshifts and types (approximate)
    z_eff = np.array([0.30, 0.51, 0.51, 0.70, 0.70, 0.93, 0.93,
                      1.32, 1.32, 2.33, 2.33, 2.33])
    types = ['DV', 'DM', 'DH', 'DM', 'DH', 'DM', 'DH',
             'DM', 'DH', 'DM', 'DH', 'DV']
    # Approximate DESI Y1 fractional errors (percent)
    frac_err = np.array([0.015, 0.013, 0.020, 0.012, 0.018, 0.015, 0.022,
                         0.020, 0.025, 0.030, 0.035, 0.025])

    import math
    h = H0 / 100.0
    omega_m_h2 = OMEGA_B_H2 + OMEGA_CDM_H2

    def compute_bao_model(kap, kf):
        """Compute BAO observables for given kappa and k_f."""
        def H_func(z):
            Omega_L = 1.0 - OMEGA_M
            E = np.sqrt(OMEGA_M * (1 + z)**3 + Omega_L)
            stress = kf * kap * ALPHA * z / (1 + z)
            return H0 * E * (1 + stress)

        r_d_lcdm = 55.154 * math.exp(-72.3 * (0.0 + 0.0006)**2) / \
                   (omega_m_h2**0.25351 * OMEGA_B_H2**0.12807)
        r_d = r_d_lcdm * (1.0 + kf * K_RS_EFE * F_KICK)

        n_grid = 5000
        z_grid = np.linspace(0, 3.0, n_grid)
        H_grid = np.array([H_func(z) for z in z_grid])
        integrand = C_LIGHT / H_grid
        dz = np.diff(z_grid)
        D_C_grid = np.zeros(n_grid)
        D_C_grid[1:] = np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz)
        D_M = np.interp(z_eff, z_grid, D_C_grid)

        model = np.empty(len(z_eff))
        for i in range(len(z_eff)):
            z = z_eff[i]
            dm = D_M[i]
            H_z = H_func(z)
            D_H = C_LIGHT / H_z
            if 'DV' in types[i]:
                D_V = (dm**2 * C_LIGHT * z / H_z)**(1.0/3.0)
                model[i] = D_V / r_d
            elif 'DM' in types[i]:
                model[i] = dm / r_d
            elif 'DH' in types[i]:
                model[i] = D_H / r_d
        return model

    # 'Data' = pure LCDM (kappa=0, k_f=0)
    obs = compute_bao_model(0.0, 0.0)
    # Model at given kappa
    pred = compute_bao_model(kappa, k_f)

    # Diagonal covariance from fractional errors
    sigma = np.abs(obs) * frac_err
    chi2 = np.sum(((obs - pred) / sigma)**2)
    return chi2, obs, pred, sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("MTDF Kappa Sensitivity Check")
    print(f"  KAPPA_A = {KAPPA_A}  (current)")
    print(f"  KAPPA_B = {KAPPA_B}  (alternative)")
    print(f"  delta_kappa = {KAPPA_B - KAPPA_A:.6e}")
    print(f"  delta_kappa/kappa = {(KAPPA_B - KAPPA_A)/KAPPA_A * 100:.2f}%")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'kappa_A': KAPPA_A,
        'kappa_B': KAPPA_B,
        'delta_kappa': KAPPA_B - KAPPA_A,
        'cosmology': {
            'H0': H0, 'Omega_m': OMEGA_M,
            'omega_b_h2': OMEGA_B_H2, 'omega_cdm_h2': OMEGA_CDM_H2,
        },
    }

    # --- CMB correction layer ---
    print("\n--- CMB Correction Layer (uses KAPPA directly, no k_f) ---")
    cmb_A = cmb_diagnostics(KAPPA_A)
    cmb_B = cmb_diagnostics(KAPPA_B)

    print(f"\n  {'Quantity':<25s} {'kappa_A':>14s} {'kappa_B':>14s} {'delta':>14s}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*14}")
    for key in ['theta_s_ratio', 'delta_theta_s_pct', 'D_A_ratio',
                'isw_boost_ell2', 'isw_boost_ell30', 'stress_at_zstar']:
        vA = cmb_A[key]
        vB = cmb_B[key]
        print(f"  {key:<25s} {vA:>14.8f} {vB:>14.8f} {vB-vA:>14.2e}")

    # Synthetic CMB chi2
    chi2_cmb_A, _, _ = cmb_chi2_estimate(KAPPA_A)
    chi2_cmb_B, _, _ = cmb_chi2_estimate(KAPPA_B)
    delta_chi2_cmb = chi2_cmb_B - chi2_cmb_A

    print(f"\n  Synthetic CMB chi2 (mock oscillatory D_l, diagonal cov):")
    print(f"    chi2(kappa_A) = {chi2_cmb_A:.4f}")
    print(f"    chi2(kappa_B) = {chi2_cmb_B:.4f}")
    print(f"    Delta chi2    = {delta_chi2_cmb:+.4f}")

    results['cmb_correction'] = {
        'kappa_A': {k: v for k, v in cmb_A.items()},
        'kappa_B': {k: v for k, v in cmb_B.items()},
        'delta_theta_s_pct': cmb_B['delta_theta_s_pct'] - cmb_A['delta_theta_s_pct'],
        'synthetic_chi2_A': chi2_cmb_A,
        'synthetic_chi2_B': chi2_cmb_B,
        'delta_chi2_cmb': delta_chi2_cmb,
    }

    # --- Late-universe likelihood (k_f * KAPPA) ---
    print("\n--- Late-Universe Likelihood (uses k_f * KAPPA) ---")
    for k_f in [1.0]:
        print(f"\n  k_f = {k_f}")
        late_A = late_universe_diagnostics(KAPPA_A, k_f)
        late_B = late_universe_diagnostics(KAPPA_B, k_f)

        print(f"\n  {'z':<6s} {'quantity':<12s} {'kappa_A':>14s} {'kappa_B':>14s} {'delta':>14s} {'rel%':>10s}")
        print(f"  {'-'*6} {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
        z_keys = [k for k in late_A if k.startswith('z=')]
        for zk in z_keys:
            for qty in ['H_z', 'D_M_over_rd', 'D_H_over_rd']:
                vA = late_A[zk][qty]
                vB = late_B[zk][qty]
                delta = vB - vA
                rel = delta / vA * 100 if vA != 0 else 0
                print(f"  {zk:<6s} {qty:<12s} {vA:>14.4f} {vB:>14.4f} {delta:>14.4e} {rel:>+9.4f}%")

        # Synthetic BAO chi2
        chi2_bao_A, obs, pred_A, sigma = bao_chi2_estimate(KAPPA_A, k_f)
        chi2_bao_B, _, pred_B, _ = bao_chi2_estimate(KAPPA_B, k_f)
        delta_chi2_bao = chi2_bao_B - chi2_bao_A

        print(f"\n  Synthetic BAO chi2 (LCDM as 'data', DESI Y1 errors):")
        print(f"    chi2(kappa_A) = {chi2_bao_A:.6f}")
        print(f"    chi2(kappa_B) = {chi2_bao_B:.6f}")
        print(f"    Delta chi2    = {delta_chi2_bao:+.6f}")

        results[f'late_universe_kf{k_f}'] = {
            'r_d_mtdf_A': late_A['r_d_mtdf_Mpc'],
            'r_d_mtdf_B': late_B['r_d_mtdf_Mpc'],
            'synthetic_bao_chi2_A': chi2_bao_A,
            'synthetic_bao_chi2_B': chi2_bao_B,
            'delta_chi2_bao': delta_chi2_bao,
        }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  kappa shift: {KAPPA_A} -> {KAPPA_B}  ({(KAPPA_B-KAPPA_A)/KAPPA_A*100:+.1f}%)")
    print(f"  Delta chi2 (CMB, synthetic):  {delta_chi2_cmb:+.4f}")
    print(f"  Delta chi2 (BAO, synthetic):  {delta_chi2_bao:+.6f}")
    print(f"  Total Delta chi2 (synthetic): {delta_chi2_cmb + delta_chi2_bao:+.4f}")
    print()
    print("  NOTE: These are synthetic estimates using mock spectra and")
    print("  diagonal covariances. For production-quality results, use the")
    print("  full MCMC pipeline with CosmoPower emulators and real data.")

    results['summary'] = {
        'delta_chi2_cmb_synthetic': delta_chi2_cmb,
        'delta_chi2_bao_synthetic': delta_chi2_bao,
        'delta_chi2_total_synthetic': delta_chi2_cmb + delta_chi2_bao,
    }

    # Save results
    out_path = Path(__file__).parent.parent.parent / "validation" / "kappa_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
