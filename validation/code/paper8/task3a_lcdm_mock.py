#!/usr/bin/env python3
"""
Task 3A LCDM Mock: Does standard cosmology produce the CF4 d_signed signal?

Three independent mock approaches:

  Mock 1: LCDM Reconstruction
    Use observed CF4 positions, predict v_pec from LCDM linear theory
    using the local density field, add realistic noise, run pipeline.
    Tests: can LCDM velocities at CF4 positions produce gamma_v ~ -54?

  Mock 2: Log-Normal Field
    Generate a synthetic LCDM universe (log-normal density + linear velocities),
    place mock galaxies, find voids in the mock, compute d_signed, run pipeline.
    Tests: does a full LCDM realisation produce the signal end-to-end?

  Mock 3: Null (Pure Noise)
    Inject Gaussian noise at CF4 positions (no physical velocity signal).
    Establishes the floor: what does random scatter produce?

Each mock is run N_REALIZATIONS times to build distributions of gamma_v.
The observed gamma_v is compared to these distributions.

Author: Ingo Mesche
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, fft as sp_fft
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import load_void_pair, gls_fit, save_results, CATALOGUE_GROUPS

sys.path.insert(0, os.path.dirname(__file__))
from task3a_cosmicflows4_vpec import (
    load_cf4, groups_to_comoving, compute_vpec_residuals,
    compute_environment_fast, gls_vpec_env, Z_CUT
)

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)
C_KMS = 299792.458
H0_FID = 75.0
N_REALIZATIONS = 100


def fast_wls_gamma(y, d_signed, w):
    """
    Fast weighted least squares for gamma_v.
    y = alpha + gamma * d_signed, weighted by w = 1/sigma^2.
    Returns gamma, gamma_err, chi2_null, chi2_full.
    Avoids constructing NxN covariance matrix.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), d_signed])
    WX = X * w[:, None]
    XtWX = X.T @ WX
    beta = np.linalg.solve(XtWX, WX.T @ y)
    beta_cov = np.linalg.inv(XtWX)

    resid_full = y - X @ beta
    chi2_full = float(np.sum(w * resid_full**2))

    # Null model
    alpha_null = np.sum(w * y) / np.sum(w)
    resid_null = y - alpha_null
    chi2_null = float(np.sum(w * resid_null**2))

    gamma = beta[1]
    gamma_err = np.sqrt(beta_cov[1, 1])
    dchi2 = chi2_null - chi2_full

    return float(gamma), float(gamma_err), float(dchi2)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3a_lcdm_mock')


# ============================================================
# LCDM Power Spectrum (Eisenstein & Hu 1998 transfer function)
# ============================================================

def eisenstein_hu_transfer(k, omega_m=0.315, omega_b=0.0493, h=1.0, ns=0.965):
    """
    Eisenstein & Hu (1998) transfer function (no wiggles version).
    k in h/Mpc, returns T(k).
    """
    theta_cmb = 2.7255 / 2.7  # T_CMB / 2.7K
    omega_mh2 = omega_m * h**2
    omega_bh2 = omega_b * h**2
    f_b = omega_b / omega_m

    # Sound horizon
    z_eq = 2.5e4 * omega_mh2 * theta_cmb**(-4)
    k_eq = 7.46e-2 * omega_mh2 * theta_cmb**(-2)  # h/Mpc

    # Silk damping
    b1 = 0.313 * omega_mh2**(-0.419) * (1 + 0.607 * omega_mh2**0.674)
    b2 = 0.238 * omega_mh2**0.223
    z_d = 1291 * omega_mh2**0.251 / (1 + 0.659 * omega_mh2**0.828) * (1 + b1 * omega_bh2**b2)

    # Sound horizon at drag epoch
    R_eq = 31.5 * omega_bh2 * theta_cmb**(-4) * (1000 / z_eq)
    R_d = 31.5 * omega_bh2 * theta_cmb**(-4) * (1000 / z_d)
    s = 2.0 / (3.0 * k_eq) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq)))

    # Transfer function (zero-baryon approximation for speed)
    q = k / (13.41 * k_eq)
    gamma_eff = omega_m * h * (
        1 - 0.328 * np.log(431 * omega_mh2) * f_b +
        0.380 * np.log(22.3 * omega_mh2) * f_b**2
    )
    q_eff = k * theta_cmb**2 / gamma_eff

    L0 = np.log(2 * np.e + 1.8 * q_eff)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
    T0 = L0 / (L0 + C0 * q_eff**2)

    return T0


def lcdm_power_spectrum(k, sigma8=0.811, omega_m=0.315, ns=0.965):
    """
    LCDM matter power spectrum P(k) at z=0.
    k in h/Mpc, returns P(k) in (Mpc/h)^3.
    Normalized to sigma8.
    """
    T = eisenstein_hu_transfer(k, omega_m=omega_m)
    Pk_unnorm = k**ns * T**2

    # Normalize to sigma8 using top-hat window
    k_norm = np.logspace(-4, 2, 10000)
    T_norm = eisenstein_hu_transfer(k_norm, omega_m=omega_m)
    Pk_norm_unnorm = k_norm**ns * T_norm**2

    R8 = 8.0  # Mpc/h
    x = k_norm * R8
    W = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1.0)
    integrand = k_norm**2 * Pk_norm_unnorm * W**2
    sigma2 = np.trapz(integrand, k_norm) / (2 * np.pi**2)

    A = sigma8**2 / sigma2
    return A * Pk_unnorm


# ============================================================
# Mock 1: LCDM Reconstruction
# ============================================================

def mock1_lcdm_reconstruction(groups, gx, gy, gz, d_signed, vpec_err,
                               n_real=N_REALIZATIONS):
    """
    Predict LCDM peculiar velocities from the local density field,
    add realistic noise, and measure gamma_v.

    In linear theory: v_pec ~ H0 * f * delta * R
    where f = Omega_m^0.55, delta = local overdensity, R = smoothing scale.

    Uses multiple smoothing scales and takes the average prediction.
    """
    print("\n" + "=" * 60)
    print("MOCK 1: LCDM Reconstruction")
    print("=" * 60)

    omega_m = 0.315
    f_growth = omega_m**0.55
    H0 = 100.0  # km/s/Mpc for Mpc/h coordinates

    z_arr = np.array([g['z'] for g in groups])

    # Compute density at R=10 Mpc/h (single scale for speed)
    coords = np.column_stack([gx, gy, gz])
    tree = cKDTree(coords)

    R_smooth = 10.0
    print(f"  Computing density at R = {R_smooth:.0f} Mpc/h...", flush=True)
    counts = tree.query_ball_point(coords, R_smooth, workers=-1, return_length=True)
    counts = np.array(counts, dtype=float) - 1
    vol = (4.0 / 3.0) * np.pi * R_smooth**3
    rho = counts / vol
    rho_mean = np.mean(rho)
    delta = (rho - rho_mean) / rho_mean if rho_mean > 0 else np.zeros_like(rho)

    # LCDM predicted peculiar velocity: v = H0 * f * delta * R / 3 (1D component)
    v_lcdm_avg = H0 * f_growth * delta * R_smooth / 3.0

    print(f"  f(growth) = {f_growth:.4f}", flush=True)
    print(f"  v_LCDM range: [{v_lcdm_avg.min():.0f}, {v_lcdm_avg.max():.0f}] km/s", flush=True)
    print(f"  v_LCDM std: {np.std(v_lcdm_avg):.0f} km/s", flush=True)

    # Precompute weights (avoids NxN matrix)
    w = 1.0 / vpec_err**2
    z_edges = np.arange(0, z_arr.max() + 0.005, 0.005)

    # Run N realizations with noise
    gamma_v_dist = np.zeros(n_real)
    gamma_v_lowz_dist = np.zeros(n_real)
    gamma_v_highz_dist = np.zeros(n_real)

    mask_low = z_arr < Z_CUT
    mask_high = z_arr >= Z_CUT
    w_low = w[mask_low]
    w_high = w[mask_high]
    d_low = d_signed[mask_low]
    d_high = d_signed[mask_high]

    for i in range(n_real):
        # Add Gaussian noise matching CF4 uncertainties
        noise = np.random.normal(0, vpec_err)
        v_mock = v_lcdm_avg + noise

        # Apply same shell-median subtraction as real analysis
        v_resid = v_mock.copy()
        for j in range(len(z_edges) - 1):
            zmask = (z_arr >= z_edges[j]) & (z_arr < z_edges[j + 1])
            if zmask.sum() > 10:
                v_resid[zmask] -= np.median(v_mock[zmask])

        # Measure gamma_v (fast WLS, no NxN matrix)
        gamma_v_dist[i], _, _ = fast_wls_gamma(v_resid, d_signed, w)

        # Piecewise
        if mask_low.sum() > 30:
            gamma_v_lowz_dist[i], _, _ = fast_wls_gamma(v_resid[mask_low], d_low, w_low)
        if mask_high.sum() > 30:
            gamma_v_highz_dist[i], _, _ = fast_wls_gamma(v_resid[mask_high], d_high, w_high)

        if (i + 1) % 25 == 0:
            print(f"    Realization {i+1}/{n_real}: gamma_v = {gamma_v_dist[i]:.2f}", flush=True)

    return {
        'gamma_v_mean': float(np.mean(gamma_v_dist)),
        'gamma_v_std': float(np.std(gamma_v_dist)),
        'gamma_v_median': float(np.median(gamma_v_dist)),
        'gamma_v_ci95': [float(np.percentile(gamma_v_dist, 2.5)),
                          float(np.percentile(gamma_v_dist, 97.5))],
        'gamma_v_lowz_mean': float(np.mean(gamma_v_lowz_dist)),
        'gamma_v_lowz_std': float(np.std(gamma_v_lowz_dist)),
        'gamma_v_highz_mean': float(np.mean(gamma_v_highz_dist)),
        'gamma_v_highz_std': float(np.std(gamma_v_highz_dist)),
        'v_lcdm_std': float(np.std(v_lcdm_avg)),
        'n_realizations': n_real,
        'distribution': gamma_v_dist.tolist(),
        'distribution_lowz': gamma_v_lowz_dist.tolist(),
        'distribution_highz': gamma_v_highz_dist.tolist(),
    }


# ============================================================
# Mock 2: Log-Normal Field
# ============================================================

def generate_lognormal_field(ngrid=128, box_size=400.0, seed=None):
    """
    Generate a log-normal density field with LCDM P(k).

    Returns: delta (overdensity), vx, vy, vz (velocity components)
    all on a (ngrid, ngrid, ngrid) grid.
    """
    rng = np.random.RandomState(seed)

    # k-space grid
    dk = 2 * np.pi / box_size
    kx = np.fft.fftfreq(ngrid, d=box_size/ngrid) * 2 * np.pi
    ky = np.fft.fftfreq(ngrid, d=box_size/ngrid) * 2 * np.pi
    kz = np.fft.rfftfreq(ngrid, d=box_size/ngrid) * 2 * np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K = np.sqrt(K2)
    K[0, 0, 0] = 1e-10  # avoid division by zero

    # Power spectrum
    Pk = lcdm_power_spectrum(K.ravel()).reshape(K.shape)
    Pk[0, 0, 0] = 0  # zero mean

    # Amplitude: sqrt(P(k) * V / (2*pi)^3) where V = box_size^3
    # For discrete FFT: amplitude = sqrt(P(k) * (dk)^3 / 2)
    vol_k = dk**3
    amplitude = np.sqrt(Pk * vol_k / 2.0)

    # Gaussian random field in k-space
    delta_k = amplitude * (rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape))
    delta_k[0, 0, 0] = 0  # zero mean

    # Gaussian field in real space
    delta_g = np.fft.irfftn(delta_k, s=(ngrid, ngrid, ngrid))

    # Log-normal transform: rho = exp(delta_g - sigma^2/2) - 1
    sigma2 = np.var(delta_g)
    delta_ln = np.exp(delta_g - sigma2 / 2) - 1

    # Velocity field from linear theory: v = i * H * f * k * delta_k / k^2
    omega_m = 0.315
    f_growth = omega_m**0.55
    H0 = 100.0  # km/s / (Mpc/h)

    # Use Gaussian delta_k for velocity (linear theory)
    vx_k = 1j * H0 * f_growth * KX * delta_k / K2
    vy_k = 1j * H0 * f_growth * KY * delta_k / K2
    vz_k = 1j * H0 * f_growth * KZ * delta_k / K2

    vx_k[0, 0, 0] = 0
    vy_k[0, 0, 0] = 0
    vz_k[0, 0, 0] = 0

    vx = np.fft.irfftn(vx_k, s=(ngrid, ngrid, ngrid)).real
    vy = np.fft.irfftn(vy_k, s=(ngrid, ngrid, ngrid)).real
    vz = np.fft.irfftn(vz_k, s=(ngrid, ngrid, ngrid)).real

    return delta_ln, vx, vy, vz


def find_voids_simple(delta, box_size, ngrid, r_min=10.0, r_max=40.0,
                       delta_threshold=-0.5, n_max=2000):
    """
    Simple spherical void finder on a density grid.

    Finds the most underdense cells and assigns fixed-radius voids,
    excluding overlapping regions. Fast approximation of VoidFinder.
    """
    cell_size = box_size / ngrid

    # Find cells below threshold
    underdense = np.argwhere(delta < delta_threshold)
    if len(underdense) == 0:
        # Try less aggressive threshold
        delta_threshold = -0.3
        underdense = np.argwhere(delta < delta_threshold)
        if len(underdense) == 0:
            print(f"    WARNING: No cells below delta = {delta_threshold}", flush=True)
            return np.array([]), np.array([]), np.array([]), np.array([])

    # Sort by density (most underdense first)
    densities = delta[underdense[:, 0], underdense[:, 1], underdense[:, 2]]
    order = np.argsort(densities)
    underdense = underdense[order]

    # Place non-overlapping voids with fixed radius based on local density
    voids_x, voids_y, voids_z, voids_r = [], [], [], []
    void_centers = []
    r_default = (r_min + r_max) / 2.0  # 25 Mpc/h typical void radius

    for idx in underdense:
        ix, iy, iz = idx
        cx = (ix + 0.5) * cell_size
        cy = (iy + 0.5) * cell_size
        cz = (iz + 0.5) * cell_size

        # Check overlap with existing voids
        overlaps = False
        for vc in void_centers:
            dx = cx - vc[0]
            dy = cy - vc[1]
            dz = cz - vc[2]
            # Periodic
            if abs(dx) > box_size/2: dx = dx - np.sign(dx) * box_size
            if abs(dy) > box_size/2: dy = dy - np.sign(dy) * box_size
            if abs(dz) > box_size/2: dz = dz - np.sign(dz) * box_size
            if dx**2 + dy**2 + dz**2 < (r_default * 1.5)**2:
                overlaps = True
                break

        if not overlaps:
            # Radius proportional to how underdense
            r_void = r_min + (r_max - r_min) * min(abs(densities[np.where(
                (underdense == idx).all(axis=1))[0][0]]) / 1.0, 1.0)
            voids_x.append(cx)
            voids_y.append(cy)
            voids_z.append(cz)
            voids_r.append(r_void)
            void_centers.append((cx, cy, cz))

        if len(voids_x) >= n_max:
            break

    print(f"    Found {len(voids_x)} voids (R range: "
          f"{min(voids_r) if voids_r else 0:.0f}-{max(voids_r) if voids_r else 0:.0f} Mpc/h)",
          flush=True)

    return (np.array(voids_x), np.array(voids_y),
            np.array(voids_z), np.array(voids_r))


def mock2_lognormal_field(groups, vpec_err, n_real=20):
    """
    Generate LCDM log-normal realizations, place mock galaxies,
    find voids, compute d_signed, and measure gamma_v.

    Fewer realizations than Mock 1 because each is expensive.
    """
    print("\n" + "=" * 60)
    print("MOCK 2: Log-Normal Field (End-to-End LCDM)")
    print("=" * 60)

    z_arr = np.array([g['z'] for g in groups])
    n_groups = len(groups)
    box_size = 400.0  # Mpc/h (covers z < 0.04 ~ 120 Mpc/h plus margin)
    ngrid = 128

    mask_low = z_arr < Z_CUT
    mask_high = z_arr >= Z_CUT

    gamma_v_dist = []
    gamma_v_lowz_dist = []
    gamma_v_highz_dist = []
    n_voids_dist = []

    for i in range(n_real):
        seed = 42 + i
        print(f"\n  --- Realization {i+1}/{n_real} (seed={seed}) ---")

        # Generate field
        delta, vx_field, vy_field, vz_field = generate_lognormal_field(
            ngrid=ngrid, box_size=box_size, seed=seed)

        # Place mock galaxies: Poisson sample proportional to (1 + delta)
        cell_size = box_size / ngrid
        rng = np.random.RandomState(seed + 1000)

        # Target: same number of galaxies as CF4
        mean_density = n_groups / box_size**3
        rho_field = np.maximum(1 + delta, 0)
        total_expected = mean_density * cell_size**3 * rho_field
        n_per_cell = rng.poisson(total_expected)

        # Vectorized galaxy placement
        occupied = np.argwhere(n_per_cell > 0)  # (N_occupied, 3)
        n_counts = n_per_cell[n_per_cell > 0]
        total_gal = int(n_counts.sum())

        # Repeat cell indices by count
        cell_ix = np.repeat(occupied[:, 0], n_counts)
        cell_iy = np.repeat(occupied[:, 1], n_counts)
        cell_iz = np.repeat(occupied[:, 2], n_counts)

        # Random offsets within cells
        mock_x = (cell_ix + rng.random(total_gal)) * cell_size
        mock_y = (cell_iy + rng.random(total_gal)) * cell_size
        mock_z_cart = (cell_iz + rng.random(total_gal)) * cell_size

        # Radial velocity (line of sight from box center)
        dx = mock_x - box_size / 2
        dy = mock_y - box_size / 2
        dz = mock_z_cart - box_size / 2
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        r = np.maximum(r, 1e-10)

        mock_vpec = (vx_field[cell_ix, cell_iy, cell_iz] * dx +
                     vy_field[cell_ix, cell_iy, cell_iz] * dy +
                     vz_field[cell_ix, cell_iy, cell_iz] * dz) / r

        n_mock = len(mock_x)
        print(f"    Generated {total_gal} mock galaxies", flush=True)
        n_mock = total_gal

        if n_mock < 100:
            print(f"    Too few galaxies, skipping")
            continue

        # Subsample to match CF4 size
        if n_mock > n_groups:
            idx = rng.choice(n_mock, n_groups, replace=False)
            mock_x = mock_x[idx]
            mock_y = mock_y[idx]
            mock_z_cart = mock_z_cart[idx]
            mock_vpec = mock_vpec[idx]
            n_mock = n_groups

        # Add noise
        noise_level = np.median(vpec_err)
        noise = rng.normal(0, noise_level, n_mock)
        mock_vpec_noisy = mock_vpec + noise

        # Assign mock redshifts based on distance from center
        dist_from_center = np.sqrt(
            (mock_x - box_size/2)**2 +
            (mock_y - box_size/2)**2 +
            (mock_z_cart - box_size/2)**2)
        mock_z = dist_from_center / (C_KMS / 100.0)  # z ~ d * H0 / c

        # Shell-median subtraction
        z_edges = np.arange(0, mock_z.max() + 0.005, 0.005)
        v_resid = mock_vpec_noisy.copy()
        for j in range(len(z_edges) - 1):
            zmask = (mock_z >= z_edges[j]) & (mock_z < z_edges[j + 1])
            if zmask.sum() > 10:
                v_resid[zmask] -= np.median(mock_vpec_noisy[zmask])

        # Find voids in the mock
        vx, vy, vz, vr = find_voids_simple(delta, box_size, ngrid,
                                             r_min=10, r_max=40)
        n_voids_dist.append(len(vx))

        if len(vx) == 0:
            print(f"    No voids found, skipping")
            continue

        # Compute d_signed for mock galaxies
        d_signed_mock, _ = compute_environment_fast(
            mock_x, mock_y, mock_z_cart, vx, vy, vz, vr)

        # Measure gamma_v (fast WLS)
        mock_w = np.full(n_mock, 1.0 / noise_level**2)
        gv, gv_err, dchi2 = fast_wls_gamma(v_resid, d_signed_mock, mock_w)
        gamma_v_dist.append(gv)
        sig = abs(gv) / gv_err if gv_err > 0 else 0

        # Piecewise
        z_cut_mock = Z_CUT
        mask_l = mock_z < z_cut_mock
        mask_h = mock_z >= z_cut_mock

        if mask_l.sum() > 30:
            gv_l, _, _ = fast_wls_gamma(v_resid[mask_l], d_signed_mock[mask_l], mock_w[mask_l])
            gamma_v_lowz_dist.append(gv_l)
        if mask_h.sum() > 30:
            gv_h, _, _ = fast_wls_gamma(v_resid[mask_h], d_signed_mock[mask_h], mock_w[mask_h])
            gamma_v_highz_dist.append(gv_h)

        print(f"    gamma_v = {gv:.2f} (sig={sig:.1f})", flush=True)

    gamma_v_dist = np.array(gamma_v_dist)
    gamma_v_lowz_dist = np.array(gamma_v_lowz_dist)
    gamma_v_highz_dist = np.array(gamma_v_highz_dist)

    return {
        'gamma_v_mean': float(np.mean(gamma_v_dist)) if len(gamma_v_dist) > 0 else None,
        'gamma_v_std': float(np.std(gamma_v_dist)) if len(gamma_v_dist) > 0 else None,
        'gamma_v_ci95': [float(np.percentile(gamma_v_dist, 2.5)),
                          float(np.percentile(gamma_v_dist, 97.5))] if len(gamma_v_dist) > 2 else None,
        'gamma_v_lowz_mean': float(np.mean(gamma_v_lowz_dist)) if len(gamma_v_lowz_dist) > 0 else None,
        'gamma_v_lowz_std': float(np.std(gamma_v_lowz_dist)) if len(gamma_v_lowz_dist) > 0 else None,
        'gamma_v_highz_mean': float(np.mean(gamma_v_highz_dist)) if len(gamma_v_highz_dist) > 0 else None,
        'gamma_v_highz_std': float(np.std(gamma_v_highz_dist)) if len(gamma_v_highz_dist) > 0 else None,
        'n_realizations_completed': len(gamma_v_dist),
        'n_voids_mean': float(np.mean(n_voids_dist)) if n_voids_dist else None,
        'box_size': box_size,
        'ngrid': ngrid,
        'distribution': gamma_v_dist.tolist(),
    }


# ============================================================
# Mock 3: Pure Noise (Null Baseline)
# ============================================================

def mock3_pure_noise(groups, d_signed, vpec_err, n_real=N_REALIZATIONS):
    """
    Inject pure Gaussian noise (no physical signal) at CF4 positions.
    Establishes the null floor.
    """
    print("\n" + "=" * 60)
    print("MOCK 3: Pure Noise (Null Baseline)")
    print("=" * 60)

    z_arr = np.array([g['z'] for g in groups])
    mask_low = z_arr < Z_CUT
    mask_high = z_arr >= Z_CUT

    gamma_v_dist = np.zeros(n_real)
    gamma_v_lowz_dist = np.zeros(n_real)
    gamma_v_highz_dist = np.zeros(n_real)

    w = 1.0 / vpec_err**2
    z_edges = np.arange(0, z_arr.max() + 0.005, 0.005)
    w_low = w[mask_low]
    w_high = w[mask_high]
    d_low = d_signed[mask_low]
    d_high = d_signed[mask_high]

    for i in range(n_real):
        noise = np.random.normal(0, vpec_err)

        # Shell-median subtraction
        v_resid = noise.copy()
        for j in range(len(z_edges) - 1):
            zmask = (z_arr >= z_edges[j]) & (z_arr < z_edges[j + 1])
            if zmask.sum() > 10:
                v_resid[zmask] -= np.median(noise[zmask])

        gamma_v_dist[i], _, _ = fast_wls_gamma(v_resid, d_signed, w)

        if mask_low.sum() > 30:
            gamma_v_lowz_dist[i], _, _ = fast_wls_gamma(v_resid[mask_low], d_low, w_low)
        if mask_high.sum() > 30:
            gamma_v_highz_dist[i], _, _ = fast_wls_gamma(v_resid[mask_high], d_high, w_high)

        if (i + 1) % 25 == 0:
            print(f"    Realization {i+1}/{n_real}: gamma_v = {gamma_v_dist[i]:.2f}", flush=True)

    return {
        'gamma_v_mean': float(np.mean(gamma_v_dist)),
        'gamma_v_std': float(np.std(gamma_v_dist)),
        'gamma_v_ci95': [float(np.percentile(gamma_v_dist, 2.5)),
                          float(np.percentile(gamma_v_dist, 97.5))],
        'gamma_v_lowz_mean': float(np.mean(gamma_v_lowz_dist)),
        'gamma_v_lowz_std': float(np.std(gamma_v_lowz_dist)),
        'gamma_v_highz_mean': float(np.mean(gamma_v_highz_dist)),
        'gamma_v_highz_std': float(np.std(gamma_v_highz_dist)),
        'n_realizations': n_real,
        'distribution': gamma_v_dist.tolist(),
    }


# ============================================================
# Plotting
# ============================================================

def plot_mock_comparison(mock1, mock2, mock3, observed_gamma_v,
                          observed_gamma_lowz, cat_name, output_dir):
    """Compare mock gamma_v distributions to observed value."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (label, mock_data) in zip(axes, [
        ('Mock 1: LCDM Reconstruction', mock1),
        ('Mock 2: Log-Normal Field', mock2),
        ('Mock 3: Pure Noise', mock3),
    ]):
        dist = mock_data.get('distribution', [])
        if len(dist) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        dist = np.array(dist)
        ax.hist(dist, bins=30, alpha=0.7, color='#2196F3', density=True,
                label=f'Mock (n={len(dist)})')
        ax.axvline(observed_gamma_v, color='red', lw=2, ls='--',
                   label=f'Observed: {observed_gamma_v:.1f}')
        ax.axvline(np.mean(dist), color='blue', lw=1.5,
                   label=f'Mock mean: {np.mean(dist):.1f}')

        # How many sigma is observed from mock distribution?
        if np.std(dist) > 0:
            n_sigma = (observed_gamma_v - np.mean(dist)) / np.std(dist)
            ax.set_title(f'{label}\nObserved is {n_sigma:.1f} sigma from mock')
        else:
            ax.set_title(label)

        ax.set_xlabel('gamma_v (km/s per d_signed)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    fig.suptitle(f'LCDM Mock Test: Observed vs Mock gamma_v ({cat_name})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'mock_comparison_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_piecewise_comparison(mock1, mock3, observed_lowz, observed_highz,
                               cat_name, output_dir):
    """Compare piecewise mock distributions to observed."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, obs_val, mock_key) in zip(axes, [
        ('z < 0.04', observed_lowz, 'gamma_v_lowz'),
        ('z >= 0.04', observed_highz, 'gamma_v_highz'),
    ]):
        # Mock 1
        m1_mean = mock1.get(f'{mock_key}_mean', 0)
        m1_std = mock1.get(f'{mock_key}_std', 1)
        # Mock 3
        m3_mean = mock3.get(f'{mock_key}_mean', 0)
        m3_std = mock3.get(f'{mock_key}_std', 1)

        x = [0, 1, 2]
        vals = [obs_val, m1_mean, m3_mean]
        errs = [0, m1_std, m3_std]
        colors = ['red', '#2196F3', '#4CAF50']
        labels_bar = ['Observed', 'LCDM Recon', 'Pure Noise']

        ax.bar(x, vals, yerr=errs, color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar)
        ax.set_ylabel('gamma_v (km/s per d_signed)')
        ax.set_title(f'{label}')
        ax.axhline(0, color='gray', ls='--')

        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + (1 if v >= 0 else -1) * max(abs(e), 2) * 1.3,
                    f'{v:.1f}', ha='center', fontsize=9)

    fig.suptitle(f'Piecewise z-split: Observed vs LCDM Mocks ({cat_name})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f'mock_piecewise_{cat_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("Task 3A LCDM MOCK TEST")
    print("Does standard cosmology produce the CF4 d_signed signal?")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Realizations: Mock1={N_REALIZATIONS}, Mock2=20, Mock3={N_REALIZATIONS}")
    print()

    # Load CF4
    print("--- Loading Cosmicflows-4 ---")
    groups = load_cf4()
    z_arr = np.array([g['z'] for g in groups])
    z_max = 0.15
    valid_mask = z_arr < z_max
    groups_valid = [g for g, m in zip(groups, valid_mask) if m]
    z_valid = np.array([g['z'] for g in groups_valid])

    gx, gy, gz = groups_to_comoving(groups_valid)

    # Vpec uncertainties
    e_dmav = np.array([g['e_dmav'] for g in groups_valid])
    dist_mpc = np.array([g['dist'] for g in groups_valid])
    vpec_err = H0_FID * dist_mpc * np.log(10) / 5.0 * e_dmav
    vpec_err = np.maximum(vpec_err, 100.0)

    # Observed gamma_v for comparison (load from task3a results)
    task3a_path = os.path.join(os.path.dirname(__file__),
                                'output', 'task3a_cf4', 'task3a_cf4_results.json')
    with open(task3a_path) as f:
        task3a = json.load(f)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'LCDM mock test for CF4 gamma_v signal',
        'n_groups': len(groups_valid),
        'mocks': {},
    }

    # Run for primary void catalogue (VoidFinder) first
    cat_name = 'VoidFinder'
    ngc_key, sgc_key, cat_type = CATALOGUE_GROUPS[cat_name]
    vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
    d_signed, in_void = compute_environment_fast(gx, gy, gz, vx, vy, vz, vr)

    observed = task3a['catalogues'][cat_name]
    obs_gamma_v = observed['full_sample']['gamma_v']
    obs_gamma_lowz = observed['piecewise']['low_z']['gamma_v']
    obs_gamma_highz = observed['piecewise']['high_z']['gamma_v']

    print(f"\nObserved gamma_v ({cat_name}):")
    print(f"  Full: {obs_gamma_v:.2f}")
    print(f"  z < 0.04: {obs_gamma_lowz:.2f}")
    print(f"  z >= 0.04: {obs_gamma_highz:.2f}")

    # Mock 1: LCDM Reconstruction
    mock1 = mock1_lcdm_reconstruction(groups_valid, gx, gy, gz, d_signed, vpec_err)

    # Mock 2: Log-Normal Field (fewer realizations, more expensive)
    mock2 = mock2_lognormal_field(groups_valid, vpec_err, n_real=10)

    # Mock 3: Pure Noise
    mock3 = mock3_pure_noise(groups_valid, d_signed, vpec_err)

    # Comparison statistics
    print("\n" + "=" * 70)
    print("MOCK TEST RESULTS")
    print("=" * 70)

    for label, mock_data in [('Mock 1 (LCDM Recon)', mock1),
                               ('Mock 2 (Log-Normal)', mock2),
                               ('Mock 3 (Noise)', mock3)]:
        if mock_data.get('gamma_v_mean') is None:
            print(f"\n  {label}: No valid realizations")
            continue

        m = mock_data['gamma_v_mean']
        s = mock_data['gamma_v_std']
        n_sigma = (obs_gamma_v - m) / s if s > 0 else float('inf')

        print(f"\n  {label}:")
        print(f"    Mock gamma_v: {m:.2f} +/- {s:.2f}")
        print(f"    Observed: {obs_gamma_v:.2f}")
        print(f"    Tension: {n_sigma:.1f} sigma")

        if mock_data.get('gamma_v_lowz_mean') is not None:
            m_l = mock_data['gamma_v_lowz_mean']
            s_l = mock_data['gamma_v_lowz_std']
            n_sig_l = (obs_gamma_lowz - m_l) / s_l if s_l > 0 else float('inf')
            print(f"    z < 0.04: mock = {m_l:.2f} +/- {s_l:.2f}, "
                  f"obs = {obs_gamma_lowz:.2f}, tension = {n_sig_l:.1f} sigma")

    # Plots
    plot_mock_comparison(mock1, mock2, mock3, obs_gamma_v,
                          obs_gamma_lowz, cat_name, OUTPUT_DIR)
    plot_piecewise_comparison(mock1, mock3, obs_gamma_lowz, obs_gamma_highz,
                              cat_name, OUTPUT_DIR)

    all_results['mocks'][cat_name] = {
        'observed': {
            'gamma_v_full': obs_gamma_v,
            'gamma_v_lowz': obs_gamma_lowz,
            'gamma_v_highz': obs_gamma_highz,
        },
        'mock1_lcdm_reconstruction': mock1,
        'mock2_lognormal_field': mock2,
        'mock3_pure_noise': mock3,
    }

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    mock1_tension = abs(obs_gamma_v - mock1['gamma_v_mean']) / mock1['gamma_v_std'] \
        if mock1['gamma_v_std'] > 0 else float('inf')

    if mock1_tension > 5:
        print(f"  LCDM Reconstruction Mock: {mock1_tension:.1f} sigma tension")
        print(f"  LCDM CANNOT reproduce the observed CF4 signal.")
        print(f"  The void-geometric peculiar velocity correlation is NOT")
        print(f"  a standard gravitational effect.")
    elif mock1_tension > 3:
        print(f"  LCDM Reconstruction Mock: {mock1_tension:.1f} sigma tension")
        print(f"  Marginal: LCDM has difficulty reproducing the signal.")
    else:
        print(f"  LCDM Reconstruction Mock: {mock1_tension:.1f} sigma tension")
        print(f"  LCDM CAN reproduce the observed signal.")
        print(f"  The CF4 result is consistent with standard gravity.")

    print("=" * 70)

    save_results(all_results, 'task3a_lcdm_mock_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
