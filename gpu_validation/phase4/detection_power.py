# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Cholesky-based detection power computation for sensitivity forecasts.

Three covariance models:
  1. Full STAT+SYS Cholesky: for N=564 consistency check against Phase 3
  2. Diagonal (baseline): C = diag(σ_i²), pure statistical errors
     → σ_floor variant: C = diag(σ_i² + σ_floor²) for realistic scenario
  3. Adversarial (d-aligned): C = diag(σ_i²) + σ_adv² × d̃ d̃ᵀ
     where d̃ = d_signed. Worst case: systematic noise aligned with regressor.

Key insight: a uniform correlated floor σ_sys² × 11ᵀ is absorbed entirely by
the intercept nuisance parameter and has ZERO effect on σ_γ. Only systematics
correlated with the regressor d_signed degrade the environment coefficient.

Test statistic: GLS Δχ² (df=1), identical to Phase 3.
"""

import numpy as np
from scipy import linalg, stats, optimize


# p-value thresholds for sigma levels (two-sided → one-sided via sign check)
SIGMA_TO_P = {2.0: 0.0455, 3.0: 0.0027, 5.0: 5.73e-7}


def prewhiten_and_project(cov_sub, d_signed, host_mass):
    """Cholesky-factorize FULL covariance, whiten, compute projectors.

    Used only for N=564 consistency check against Phase 3.
    Adaptive Tikhonov regularization: starts at 1e-12, ×10 until success, cap 1e-6.

    Returns: w, q, x_env_w, sigma_gamma, reg_scale
    """
    N = len(d_signed)
    diag_mean = np.mean(np.diag(cov_sub))

    L = None
    reg_scale = 1e-12
    while reg_scale <= 1e-6:
        reg = reg_scale * diag_mean
        C = cov_sub + reg * np.eye(N)
        try:
            L = linalg.cholesky(C, lower=True)
            break
        except linalg.LinAlgError:
            reg_scale *= 10
    if L is None:
        raise linalg.LinAlgError(
            f"Cholesky failed even at reg_scale=1e-6 (N={N})"
        )

    mass_step = (host_mass >= 10).astype(float)
    X_null = np.column_stack([np.ones(N), mass_step])
    X_full = np.column_stack([np.ones(N), d_signed, mass_step])

    X_null_w = linalg.solve_triangular(L, X_null, lower=True)
    X_full_w = linalg.solve_triangular(L, X_full, lower=True)

    XnTXn_inv = linalg.inv(X_null_w.T @ X_null_w)
    x_env_w = X_full_w[:, 1]
    H_null_x_env = X_null_w @ (XnTXn_inv @ (X_null_w.T @ x_env_w))
    v = x_env_w - H_null_x_env
    v_norm = np.sqrt(v @ v)
    w = v / v_norm

    XfTXf_inv = linalg.inv(X_full_w.T @ X_full_w)
    q = XfTXf_inv[1, :] @ X_full_w.T
    sigma_gamma = np.sqrt(XfTXf_inv[1, 1])

    return w, q, x_env_w, sigma_gamma, reg_scale


def sigma_gamma_diagonal(mu_err, d_signed, host_mass, sigma_floor=0.0):
    """Compute sigma_gamma for C = diag(σ_i² + σ_floor²). O(N)."""
    N = len(d_signed)
    sigma_eff_sq = mu_err ** 2 + sigma_floor ** 2
    w_vec = 1.0 / sigma_eff_sq

    mass_step = (host_mass >= 10).astype(float)
    X = np.column_stack([np.ones(N), d_signed, mass_step])

    XtWX = X.T @ (w_vec[:, None] * X)
    A_inv = linalg.inv(XtWX)
    return np.sqrt(A_inv[1, 1])


def calibrate_sigma_floor(mu_err, d_signed, host_mass, target_sigma_gamma):
    """Find σ_floor such that sigma_gamma(diag + floor) = target at N=564.

    Brent's method on [0, 1].
    """
    sig_diag = sigma_gamma_diagonal(mu_err, d_signed, host_mass, 0.0)

    if sig_diag >= target_sigma_gamma:
        return 0.0

    def objective(log_sf):
        sf = 10 ** log_sf
        sig = sigma_gamma_diagonal(mu_err, d_signed, host_mass, sf)
        return sig - target_sigma_gamma

    try:
        log_sol = optimize.brentq(objective, -6, 0, xtol=1e-10)
        return 10 ** log_sol
    except ValueError:
        return 0.0


def compute_power_diagonal(mu_err, d_signed, host_mass, gamma_true,
                           sigma_floor, n_mocks, seed=42):
    """Batch Monte Carlo power for C = diag(σ_i² + σ_floor²).

    Baseline (σ_floor=0) or realistic (σ_floor > 0).
    Vectorized WLS, O(N×K).
    """
    N = len(d_signed)
    rng = np.random.RandomState(seed)

    sigma_eff = np.sqrt(mu_err ** 2 + sigma_floor ** 2)
    w_vec = 1.0 / sigma_eff ** 2

    mass_step = (host_mass >= 10).astype(float)
    X_null = np.column_stack([np.ones(N), mass_step])
    X_full = np.column_stack([np.ones(N), d_signed, mass_step])

    A_null = X_null.T @ (w_vec[:, None] * X_null)
    A_full = X_full.T @ (w_vec[:, None] * X_full)
    A_null_inv = linalg.inv(A_null)
    A_full_inv = linalg.inv(A_full)

    sg = float(np.sqrt(A_full_inv[1, 1]))

    # Generate mocks: y = γ×d + σ_eff × ε
    E = rng.standard_normal((N, n_mocks))
    Y = gamma_true * d_signed[:, None] + sigma_eff[:, None] * E

    # WLS: C^{-1} Y = W Y
    wY = w_vec[:, None] * Y

    beta_null = A_null_inv @ (X_null.T @ wY)
    beta_full = A_full_inv @ (X_full.T @ wY)
    gammas = beta_full[1, :]

    R_null = Y - X_null @ beta_null
    R_full = Y - X_full @ beta_full

    chi2_null = np.sum(R_null * (w_vec[:, None] * R_null), axis=0)
    chi2_full = np.sum(R_full * (w_vec[:, None] * R_full), axis=0)

    dchi2 = np.maximum(chi2_null - chi2_full, 0)
    pvals = 1.0 - stats.chi2.cdf(dchi2, 1)

    results = {
        'n_mocks': n_mocks,
        'sigma_gamma': sg,
        'sigma_floor': float(sigma_floor),
        'gamma_env_mean': float(np.mean(gammas)),
        'gamma_env_std': float(np.std(gammas)),
        'gamma_env_median': float(np.median(gammas)),
        'dchi2_mean': float(np.mean(dchi2)),
        'dchi2_median': float(np.median(dchi2)),
    }

    for thresh, p_thresh in SIGMA_TO_P.items():
        detected = (pvals < p_thresh) & (gammas > 0)
        results[f'power_{thresh:.0f}sigma'] = float(np.mean(detected))

    return results


# ---------- Adversarial model: systematic aligned with d_signed ----------


def sigma_gamma_adversarial(mu_err, d_signed, host_mass, sigma_adv):
    """Compute sigma_gamma for C = diag(σ²) + σ_adv² × d d^T.

    Uses Woodbury identity for the rank-1 update aligned with d_signed.
    O(N) computation.
    """
    N = len(d_signed)
    w_vec = 1.0 / mu_err ** 2
    d = d_signed

    # Woodbury: C^{-1} = W - α_d (Wd)(Wd)^T / (d^T W d)
    # with α_d = σ_adv² / (1 + σ_adv² × d^T W d)
    dWd = np.sum(d ** 2 * w_vec)
    alpha_d = sigma_adv ** 2 / (1 + sigma_adv ** 2 * dWd) if sigma_adv > 0 else 0.0

    mass_step = (host_mass >= 10).astype(float)
    X = np.column_stack([np.ones(N), d, mass_step])

    XtWX = X.T @ (w_vec[:, None] * X)
    XtWd = X.T @ (w_vec * d)
    A = XtWX - alpha_d * np.outer(XtWd, XtWd)

    A_inv = linalg.inv(A)
    return np.sqrt(A_inv[1, 1])


def calibrate_sigma_adv(mu_err, d_signed, host_mass, target_sigma_gamma):
    """Find σ_adv such that sigma_gamma(adversarial) = target at N=564.

    target_sigma_gamma should be > baseline sigma_gamma.
    """
    sig_base = sigma_gamma_adversarial(mu_err, d_signed, host_mass, 0.0)

    if sig_base >= target_sigma_gamma:
        return 0.0

    def objective(log_sa):
        sa = 10 ** log_sa
        sig = sigma_gamma_adversarial(mu_err, d_signed, host_mass, sa)
        return sig - target_sigma_gamma

    try:
        log_sol = optimize.brentq(objective, -6, 1, xtol=1e-10)
        return 10 ** log_sol
    except ValueError:
        return 0.0


def compute_power_adversarial(mu_err, d_signed, host_mass, gamma_true,
                              sigma_adv, n_mocks, seed=42):
    """Batch Monte Carlo power for C = diag(σ²) + σ_adv² × d d^T.

    Adversarial: systematic noise perfectly correlated with the environment
    regressor. Uses Woodbury identity for C^{-1}, fully O(N×K).

    Noise generation: y_m = γ_true × d + σ_i × ε_stat + σ_adv × ζ × d
    where ε_stat ~ N(0, I_N) and ζ ~ N(0, 1).
    """
    N = len(d_signed)
    rng = np.random.RandomState(seed)
    d = d_signed

    w_vec = 1.0 / mu_err ** 2
    dWd = float(np.sum(d ** 2 * w_vec))
    alpha_d = sigma_adv ** 2 / (1 + sigma_adv ** 2 * dWd) if sigma_adv > 0 else 0.0

    mass_step = (host_mass >= 10).astype(float)
    X_null = np.column_stack([np.ones(N), mass_step])
    X_full = np.column_stack([np.ones(N), d, mass_step])

    # GLS normal equations: X^T C^{-1} X = X^T W X - α_d (X^T Wd)(Wd^T X)
    Wd = w_vec * d

    def _build_gls(X):
        XtWX = X.T @ (w_vec[:, None] * X)
        XtWd_vec = X.T @ Wd
        return XtWX - alpha_d * np.outer(XtWd_vec, XtWd_vec)

    A_null = _build_gls(X_null)
    A_full = _build_gls(X_full)
    A_null_inv = linalg.inv(A_null)
    A_full_inv = linalg.inv(A_full)

    sg = float(np.sqrt(A_full_inv[1, 1]))

    # Generate mocks: y = γ×d + σ_i×ε + σ_adv×ζ×d
    E = rng.standard_normal((N, n_mocks))
    Y = gamma_true * d[:, None] + mu_err[:, None] * E
    if sigma_adv > 0:
        zeta = rng.standard_normal(n_mocks)
        Y += sigma_adv * d[:, None] * zeta[None, :]

    # Apply C^{-1}: C^{-1} Y = W Y - α_d Wd (Wd^T Y) / (note: already factored)
    wY = w_vec[:, None] * Y
    WdTY = Wd @ Y  # (K,)
    CinvY = wY - alpha_d * Wd[:, None] * WdTY[None, :]

    beta_null = A_null_inv @ (X_null.T @ CinvY)
    beta_full = A_full_inv @ (X_full.T @ CinvY)
    gammas = beta_full[1, :]

    R_null = Y - X_null @ beta_null
    R_full = Y - X_full @ beta_full

    def _chi2_batch(R):
        wR = w_vec[:, None] * R
        WdTR = Wd @ R
        return np.sum(R * wR, axis=0) - alpha_d * WdTR ** 2

    chi2_null = _chi2_batch(R_null)
    chi2_full = _chi2_batch(R_full)

    dchi2 = np.maximum(chi2_null - chi2_full, 0)
    pvals = 1.0 - stats.chi2.cdf(dchi2, 1)

    results = {
        'n_mocks': n_mocks,
        'sigma_gamma': sg,
        'sigma_adv': float(sigma_adv),
        'gamma_env_mean': float(np.mean(gammas)),
        'gamma_env_std': float(np.std(gammas)),
        'gamma_env_median': float(np.median(gammas)),
        'dchi2_mean': float(np.mean(dchi2)),
        'dchi2_median': float(np.median(dchi2)),
    }

    for thresh, p_thresh in SIGMA_TO_P.items():
        detected = (pvals < p_thresh) & (gammas > 0)
        results[f'power_{thresh:.0f}sigma'] = float(np.mean(detected))

    return results


# ---------- Pre-whitened Cholesky (consistency check only) ----------


def compute_power_batch(w, q, x_env_w, gamma_true, n_mocks, seed=42):
    """Vectorized power for pre-whitened Cholesky projectors (consistency check)."""
    rng = np.random.RandomState(seed)
    N = len(w)
    E = rng.standard_normal((N, n_mocks))
    Y = gamma_true * x_env_w[:, None] + E

    dchi2 = (w @ Y) ** 2
    gammas = q @ Y
    pvals = 1.0 - stats.chi2.cdf(np.maximum(dchi2, 0), 1)

    results = {
        'n_mocks': n_mocks,
        'gamma_env_mean': float(np.mean(gammas)),
        'gamma_env_std': float(np.std(gammas)),
        'gamma_env_median': float(np.median(gammas)),
        'dchi2_mean': float(np.mean(dchi2)),
        'dchi2_median': float(np.median(dchi2)),
    }
    for thresh, p_thresh in SIGMA_TO_P.items():
        detected = (pvals < p_thresh) & (gammas > 0)
        results[f'power_{thresh:.0f}sigma'] = float(np.mean(detected))
    return results


def analytic_power_estimate(gamma_true, sigma_gamma):
    """Quick analytic cross-check: power ~ Phi(SNR - z_threshold)."""
    snr = gamma_true / sigma_gamma
    results = {'expected_snr': float(snr), 'sigma_gamma': float(sigma_gamma)}
    for thresh in [2.0, 3.0, 5.0]:
        power = float(stats.norm.cdf(snr - thresh))
        results[f'power_{thresh:.0f}sigma'] = max(0.0, min(1.0, power))
    return results
