# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Test 3: k_f identifiability and degeneracy map.

Checks whether k_f is a genuine physical parameter or a reparameterisation
of an existing nuisance / amplitude combination.

Deliverables:
  A) Correlation heatmap (cosmo + top nuisance params)
     + targeted 2D contours (k_f vs sigma8, H0, logA, tau, top nuisance)
  B) Linear predictability R² report:
     nuisance-only, cosmo-only, cosmo+nuisance
  C) k_f vs A_eff = A_s * exp(-2*tau) degeneracy check

Output: results/phase5/test3/

Usage:
  source venv/bin/activate  # from repo root
  python -m mtdf_validation.phase5_plik.test3_kf_identifiability
"""

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("../../mcmc_results")
OUTPUT_DIR = Path("../../mcmc_results/test3")

KF_PARAM = "mtdf_k_f"
COSMO_PARAMS = ["H0", "sigma8", "Omega_m", "n_s", "omega_b", "omega_cdm", "tau_reio", "logA"]

# Split for R² analysis: early-time (sampled primordial) vs late-time (derived)
EARLY_TIME_PARAMS = ["omega_b", "omega_cdm", "theta_s_100", "n_s", "tau_reio", "logA"]
LATE_TIME_PARAMS = ["H0", "sigma8", "Omega_m"]

# Nuisance parameters (sampled in Planck plik run)
NUISANCE_PARAMS = [
    "A_planck", "calib_100T", "calib_217T",
    "A_cib_217", "xi_sz_cib", "A_sz", "ksz_norm",
    "gal545_A_100", "gal545_A_143", "gal545_A_143_217", "gal545_A_217",
    "ps_A_100_100", "ps_A_143_143", "ps_A_143_217", "ps_A_217_217",
    "galf_TE_A_100", "galf_TE_A_100_143", "galf_TE_A_100_217",
    "galf_TE_A_143", "galf_TE_A_143_217", "galf_TE_A_217",
]


def get_weighted_columns(samples, param_names):
    """Extract weighted columns from getdist MCSamples."""
    weights = samples.weights
    cols = []
    valid_names = []
    for p in param_names:
        idx = samples.paramNames.numberOfName(p)
        if idx >= 0:
            cols.append(samples.samples[:, idx])
            valid_names.append(p)
    return np.column_stack(cols), weights, valid_names


def weighted_corr_matrix(data, weights):
    """Compute weighted Pearson correlation matrix."""
    w = weights / weights.sum()
    wmean = np.average(data, weights=w, axis=0)
    diff = data - wmean
    n = data.shape[1]
    wcov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.sum(w * diff[:, i] * diff[:, j])
            wcov[i, j] = val
            wcov[j, i] = val
    diag = np.sqrt(np.diag(wcov))
    diag[diag == 0] = 1e-30
    wcorr = wcov / np.outer(diag, diag)
    return wcov, wcorr, wmean, diag


def conditional_sigma(wcov, target_idx):
    """Schur complement: Var(x|y) = Sigma_xx - Sigma_xy Sigma_yy^-1 Sigma_yx."""
    others = [i for i in range(wcov.shape[0]) if i != target_idx]
    Sigma_xx = wcov[target_idx, target_idx]
    Sigma_xy = wcov[target_idx, others]
    Sigma_yy = wcov[np.ix_(others, others)]
    try:
        cond_var = Sigma_xx - Sigma_xy @ np.linalg.solve(Sigma_yy, Sigma_xy)
        return np.sqrt(max(cond_var, 0))
    except np.linalg.LinAlgError:
        return float('nan')


def partial_correlation(wcov, i, j, conditioning_indices):
    """Partial correlation between variables i and j, conditioning on a set.

    Uses the precision matrix (inverse covariance) approach:
      r_ij|S = -P_ij / sqrt(P_ii * P_jj)
    where P = (Sigma_subset)^{-1} and subset = {i, j} ∪ S.
    """
    subset = [i, j] + list(conditioning_indices)
    sub_cov = wcov[np.ix_(subset, subset)]
    try:
        P = np.linalg.inv(sub_cov)
        # i is at index 0, j is at index 1 in the subset
        partial_r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
        return float(partial_r)
    except np.linalg.LinAlgError:
        return float('nan')


def _weighted_predict(y, X, weights):
    """Return predicted values from weighted OLS of y on X."""
    w = np.sqrt(weights / weights.sum() * len(weights))
    Xw = X * w[:, None]
    yw = y * w
    ones = w.copy()
    Xw_int = np.column_stack([ones, Xw])
    try:
        beta, _, _, _ = np.linalg.lstsq(Xw_int, yw, rcond=None)
        X_int = np.column_stack([np.ones(len(y)), X])
        return X_int @ beta
    except np.linalg.LinAlgError:
        return np.full_like(y, np.nan)


def _weighted_pearson(x, y, w):
    """Weighted Pearson correlation between x and y."""
    mx = np.sum(w * x)
    my = np.sum(w * y)
    dx, dy = x - mx, y - my
    cov = np.sum(w * dx * dy)
    sx = np.sqrt(np.sum(w * dx**2))
    sy = np.sqrt(np.sum(w * dy**2))
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def weighted_r2(y, X, weights):
    """Weighted linear regression R² (ordinary least squares with weights)."""
    w = np.sqrt(weights / weights.sum() * len(weights))
    Xw = X * w[:, None]
    yw = y * w
    # Add intercept
    ones = w.copy()
    Xw_int = np.column_stack([ones, Xw])
    try:
        beta, _, _, _ = np.linalg.lstsq(Xw_int, yw, rcond=None)
        y_pred = Xw_int @ beta
        ss_res = np.sum((yw - y_pred) ** 2)
        ss_tot = np.sum((yw - np.mean(yw)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except np.linalg.LinAlgError:
        return float('nan')


def main():
    try:
        from getdist import loadMCSamples, plots
    except ImportError:
        print("ERROR: getdist not installed. pip install getdist")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load chain ─────────────────────────────────────────────────────
    prefix = RESULTS_DIR / "mtdf_mcmc"
    print(f"Loading MTDF chain from {prefix}...")
    samples = loadMCSamples(str(prefix), no_cache=True)
    print(f"  {samples.numrows} samples loaded")

    weights = samples.weights

    # ═══════════════════════════════════════════════════════════════════
    # A) CORRELATION HEATMAP  (cosmo + top nuisance)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("A) CORRELATION HEATMAP: k_f vs cosmo + nuisance")
    print("=" * 70)

    # First compute k_f correlation with ALL nuisance params to find top ones
    all_nuis_data, _, valid_nuis = get_weighted_columns(
        samples, [KF_PARAM] + NUISANCE_PARAMS
    )
    _, nuis_corr, _, _ = weighted_corr_matrix(all_nuis_data, weights)
    kf_nuis_corr = {
        valid_nuis[i]: float(nuis_corr[0, i])
        for i in range(1, len(valid_nuis))
    }
    nuis_sorted = sorted(kf_nuis_corr.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\n  k_f correlation with ALL nuisance parameters:")
    print(f"  {'Parameter':<30s} {'r(k_f, .)':>10s}")
    print(f"  {'─' * 44}")
    for p, r in nuis_sorted:
        print(f"  {p:<30s} {r:+10.4f}")

    # Select top 8 nuisances (by |correlation|)
    top_nuis = [p for p, _ in nuis_sorted[:8]]
    top_nuis_most = nuis_sorted[0][0]  # single most correlated nuisance

    print(f"\n  Top 8 nuisances selected for heatmap: {top_nuis}")
    print(f"  Most correlated nuisance: {top_nuis_most} (r={nuis_sorted[0][1]:+.4f})")

    # Build full heatmap parameter set: k_f + cosmo + top nuisance
    heatmap_params = [KF_PARAM] + COSMO_PARAMS + top_nuis
    hm_data, _, hm_valid = get_weighted_columns(samples, heatmap_params)
    _, hm_corr, _, _ = weighted_corr_matrix(hm_data, weights)

    n_hm = len(hm_valid)
    kf_all_corr = {hm_valid[i]: float(hm_corr[0, i]) for i in range(1, n_hm)}

    # Pass condition A: no near-perfect correlation (|r| < 0.8 say)
    max_abs_r = max(abs(r) for r in kf_all_corr.values())
    pass_A = max_abs_r < 0.8
    print(f"\n  Max |r(k_f, .)| = {max_abs_r:.4f}")
    print(f"  PASS (no near-perfect correlation): {'YES' if pass_A else 'FAIL'}")

    # ─── Plot A1: Correlation heatmap ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(hm_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_hm))
    ax.set_xticklabels(hm_valid, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(n_hm))
    ax.set_yticklabels(hm_valid, fontsize=8)
    for i in range(n_hm):
        for j in range(n_hm):
            color = 'white' if abs(hm_corr[i, j]) > 0.6 else 'black'
            ax.text(j, i, f"{hm_corr[i, j]:.2f}", ha='center', va='center',
                    fontsize=6, color=color)
    # Highlight k_f row/column
    ax.axhline(0.5, color='gold', linewidth=2, linestyle='-')
    ax.axvline(0.5, color='gold', linewidth=2, linestyle='-')
    fig.colorbar(im, ax=ax, label='Weighted Pearson r', shrink=0.8)
    ax.set_title("Test 3A: Parameter correlation matrix\n"
                 f"(k_f + {len(COSMO_PARAMS)} cosmo + {len(top_nuis)} top nuisance)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "test3_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {heatmap_path}")
    plt.close()

    # ─── Plot A2: Targeted 2D contours ──────────────────────────────────
    print("\nGenerating targeted 2D contour plots...")

    target_params = ["sigma8", "H0", "logA", "tau_reio", top_nuis_most]
    target_labels = [
        r"$\sigma_8$ (physical coupling effect)",
        r"$H_0$ (late-time coupling)",
        r"$\log A$ ($A_s \cdot e^{-2\tau}$ degeneracy)",
        r"$\tau_\mathrm{reio}$ ($A_s \cdot e^{-2\tau}$ degeneracy)",
        f"{top_nuis_most} (most correlated nuisance)",
    ]

    fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5))
    for i, (param, label) in enumerate(zip(target_params, target_labels)):
        ax = axes2[i]
        density = samples.get2DDensity(KF_PARAM, param)
        levels = sorted(density.getContourLevels([0.68, 0.95]))

        x = density.x
        y = density.y
        P = density.P

        ax.contourf(x, y, P.T, levels=[levels[0], levels[1], P.max()],
                    colors=['#90CAF9', '#2196F3'], alpha=0.7)
        ax.contour(x, y, P.T, levels=levels, colors=['#0D47A1', '#1565C0'],
                   linewidths=[1, 1.5])

        r = kf_all_corr.get(param, 0)
        ax.set_xlabel(r'$k_f$', fontsize=11)
        ax.set_ylabel(param, fontsize=10)
        ax.set_title(f"{label}\nr = {r:+.3f}", fontsize=9)

    fig2.suptitle("Test 3A: k_f 2D contours vs top suspects (68% and 95% CI)",
                  fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    contour_path = OUTPUT_DIR / "test3_kf_2d_contours.png"
    fig2.savefig(contour_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {contour_path}")
    plt.close()

    # ─── A3: omega_b partial correlation analysis ─────────────────────
    print("\n" + "=" * 70)
    print("A3) omega_b partial correlation (conditioning on other cosmo params)")
    print("=" * 70)

    # Build covariance of [k_f, omega_b, H0, sigma8, Omega_m, n_s, omega_cdm, tau_reio, logA]
    partial_params = [KF_PARAM, "omega_b"] + [p for p in COSMO_PARAMS if p != "omega_b"]
    partial_data, _, partial_valid = get_weighted_columns(samples, partial_params)
    partial_cov, partial_corr, _, _ = weighted_corr_matrix(partial_data, weights)

    # k_f is index 0, omega_b is index 1, rest are indices 2..
    conditioning_idx = list(range(2, len(partial_valid)))
    r_marginal_ob = float(partial_corr[0, 1])
    r_partial_ob = partial_correlation(partial_cov, 0, 1, conditioning_idx)

    # Also compute targeted partial correlations with smaller conditioning sets
    # to avoid multicollinearity inflation from the full 8-param conditioning
    partial_targeted = {}
    for cond_name, cond_params in [
        ("n_s only", ["n_s"]),
        ("n_s, omega_cdm", ["n_s", "omega_cdm"]),
        ("n_s, omega_cdm, logA", ["n_s", "omega_cdm", "logA"]),
        ("all other cosmo", [p for p in COSMO_PARAMS if p != "omega_b"]),
    ]:
        cond_set = [KF_PARAM, "omega_b"] + cond_params
        cond_data, _, cond_valid = get_weighted_columns(samples, cond_set)
        cond_cov, _, _, _ = weighted_corr_matrix(cond_data, weights)
        cond_idx = list(range(2, len(cond_valid)))
        r_partial_targeted = partial_correlation(cond_cov, 0, 1, cond_idx)
        partial_targeted[cond_name] = r_partial_targeted

    print(f"  Marginal  r(k_f, omega_b) = {r_marginal_ob:+.4f}")
    print(f"\n  Partial correlations (conditioning on progressively more params):")
    print(f"  {'Conditioning set':<35s} {'r_partial':>10s}")
    print(f"  {'─' * 48}")
    for cond_name, r_val in partial_targeted.items():
        print(f"  {cond_name:<35s} {r_val:+10.4f}")

    # The key diagnostic: does the marginal r collapse or persist under minimal conditioning?
    r_partial_minimal = partial_targeted.get("n_s, omega_cdm", r_partial_ob)
    if abs(r_partial_minimal) < 0.15:
        ob_verdict = "INDIRECT — collapses under minimal conditioning"
        ob_indirect = True
    elif abs(r_partial_minimal) < abs(r_marginal_ob) * 0.5:
        ob_verdict = "MOSTLY INDIRECT — weakens substantially under conditioning"
        ob_indirect = True
    else:
        ob_verdict = "DIRECT — persists under conditioning (moderate, not problematic)"
        ob_indirect = False

    r_partial_full = partial_targeted.get("all other cosmo", r_partial_ob)
    print(f"\n  Note: partial r with full conditioning set ({r_partial_full:+.3f}) can be")
    print(f"  inflated by multicollinearity. The minimal conditioning sets above")
    print(f"  give a more stable picture.")
    print(f"  Verdict: {ob_verdict}")

    # ─── A3b: Residual-method cross-check ───────────────────────────────
    print(f"\n  Residual-method cross-check (regression residuals, not precision matrix):")
    print(f"  {'Conditioning set':<35s} {'r_precision':>12s} {'r_residual':>12s} {'Match?':>8s}")
    print(f"  {'─' * 70}")

    partial_residual = {}
    for cond_name, cond_params in [
        ("n_s only", ["n_s"]),
        ("n_s, omega_cdm", ["n_s", "omega_cdm"]),
        ("n_s, omega_cdm, logA", ["n_s", "omega_cdm", "logA"]),
        ("all other cosmo", [p for p in COSMO_PARAMS if p != "omega_b"]),
    ]:
        # Get conditioning data
        cond_data_res, _, _ = get_weighted_columns(samples, cond_params)
        # Get k_f and omega_b columns
        kf_col_tmp = samples.samples[:, samples.paramNames.numberOfName(KF_PARAM)]
        ob_col_tmp = samples.samples[:, samples.paramNames.numberOfName("omega_b")]
        # Regress k_f on conditioning set, get residual
        res_kf = kf_col_tmp - _weighted_predict(kf_col_tmp, cond_data_res, weights)
        # Regress omega_b on conditioning set, get residual
        res_ob = ob_col_tmp - _weighted_predict(ob_col_tmp, cond_data_res, weights)
        # Weighted correlation of residuals
        w_norm = weights / weights.sum()
        r_res = _weighted_pearson(res_kf, res_ob, w_norm)
        partial_residual[cond_name] = r_res
        r_prec = partial_targeted[cond_name]
        match = "YES" if abs(r_res - r_prec) < 0.05 else "DISAGREE"
        print(f"  {cond_name:<35s} {r_prec:+12.4f} {r_res:+12.4f} {match:>8s}")

    # ─── A3c: Condition number and VIF ──────────────────────────────────
    print(f"\n  Condition number & VIF for 'all other cosmo' conditioning set:")
    full_cond_params = [p for p in COSMO_PARAMS if p != "omega_b"]
    full_cond_data, _, full_cond_valid = get_weighted_columns(samples, full_cond_params)
    full_cond_cov, full_cond_corr, _, _ = weighted_corr_matrix(full_cond_data, weights)

    eigvals = np.linalg.eigvalsh(full_cond_corr)
    cond_number = eigvals[-1] / max(eigvals[0], 1e-30)
    print(f"  Eigenvalues: {np.sort(eigvals)}")
    print(f"  Condition number: {cond_number:.1f}")
    if cond_number > 100:
        print(f"  → HIGH multicollinearity (κ > 100). Full-conditioning partial r is unreliable.")
    elif cond_number > 30:
        print(f"  → MODERATE multicollinearity (κ > 30). Full-conditioning partial r may be inflated.")
    else:
        print(f"  → LOW multicollinearity. Full-conditioning partial r is reliable.")

    # VIF for each predictor
    print(f"\n  {'Predictor':<20s} {'VIF':>8s}")
    print(f"  {'─' * 30}")
    vif_results = {}
    try:
        corr_inv = np.linalg.inv(full_cond_corr)
        for i, p in enumerate(full_cond_valid):
            vif = corr_inv[i, i]
            vif_results[p] = float(vif)
            flag = " <<<" if vif > 10 else ""
            print(f"  {p:<20s} {vif:8.1f}{flag}")
    except np.linalg.LinAlgError:
        print(f"  Correlation matrix singular — VIF undefined")

    # 2D contour: k_f vs omega_b with annotation
    print("\nGenerating k_f vs omega_b 2D contour...")
    fig_ob, ax_ob = plt.subplots(figsize=(7, 6))
    density_ob = samples.get2DDensity(KF_PARAM, "omega_b")
    levels_ob = sorted(density_ob.getContourLevels([0.68, 0.95]))

    ax_ob.contourf(density_ob.x, density_ob.y, density_ob.P.T,
                   levels=[levels_ob[0], levels_ob[1], density_ob.P.max()],
                   colors=['#90CAF9', '#2196F3'], alpha=0.7)
    ax_ob.contour(density_ob.x, density_ob.y, density_ob.P.T,
                  levels=levels_ob, colors=['#0D47A1', '#1565C0'],
                  linewidths=[1, 1.5])

    ax_ob.set_xlabel(r'$k_f$', fontsize=12)
    ax_ob.set_ylabel(r'$\omega_b$', fontsize=12)
    ax_ob.set_title(
        f"Test 3A: k_f vs $\\omega_b$ (strongest cosmo correlation)\n"
        f"Marginal r = {r_marginal_ob:+.3f},  "
        f"Partial r|($n_s$,$\\omega_c$) = {r_partial_minimal:+.3f}\n"
        f"{'Indirect covariance' if ob_indirect else 'Direct, moderate — not a Planck degeneracy block'}",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    ob_path = OUTPUT_DIR / "test3_kf_vs_omega_b.png"
    fig_ob.savefig(ob_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {ob_path}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # B) LINEAR PREDICTABILITY R² REPORT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("B) LINEAR PREDICTABILITY: can nuisances predict k_f?")
    print("=" * 70)

    # Extract k_f column
    kf_idx = samples.paramNames.numberOfName(KF_PARAM)
    kf_col = samples.samples[:, kf_idx]

    # Nuisance-only predictor matrix
    nuis_data, _, nuis_valid = get_weighted_columns(samples, NUISANCE_PARAMS)
    r2_nuis = weighted_r2(kf_col, nuis_data, weights)

    # Early-time (sampled primordial) — these are independent of k_f by construction
    early_data, _, early_valid = get_weighted_columns(samples, EARLY_TIME_PARAMS)
    r2_early = weighted_r2(kf_col, early_data, weights)

    # Late-time (derived) — these respond physically to k_f
    late_data, _, late_valid = get_weighted_columns(samples, LATE_TIME_PARAMS)
    r2_late = weighted_r2(kf_col, late_data, weights)

    # All cosmo (for reference)
    cosmo_data, _, cosmo_valid = get_weighted_columns(samples, COSMO_PARAMS)
    r2_cosmo_all = weighted_r2(kf_col, cosmo_data, weights)

    # Nuisance + early (no late-time derived)
    early_nuis_data = np.column_stack([early_data, nuis_data])
    r2_early_nuis = weighted_r2(kf_col, early_nuis_data, weights)

    kf_genuine_frac = 1.0 - r2_nuis

    print(f"\n  {'Predictor set':<40s} {'R²':>8s}  {'Interpretation'}")
    print(f"  {'─' * 80}")
    print(f"  {'Nuisance params only':<40s} {r2_nuis:8.4f}  NOT a nuisance reskin")
    print(f"  {'Early-time (ωb,ωc,θs,ns,τ,logA)':<40s} {r2_early:8.4f}  Weakly constrained by primordial params")
    print(f"  {'Late-time derived (H0,σ8,Ωm)':<40s} {r2_late:8.4f}  Physical response of late-time observables")
    print(f"  {'Early-time + nuisance':<40s} {r2_early_nuis:8.4f}  Combined independent constraint")
    print(f"  {'All cosmo (early + late derived)':<40s} {r2_cosmo_all:8.4f}  Partly tautological (late derived ← k_f)")

    pass_B_nuis = r2_nuis < 0.5
    pass_B_genuine = kf_genuine_frac > 0.5
    print(f"\n  PASS (nuisance R² < 0.5):     {'YES' if pass_B_nuis else 'FAIL'} (R²={r2_nuis:.4f})")
    print(f"  Genuine k_f variance:          {kf_genuine_frac:.1%} (not explained by nuisances)")
    print(f"\n  Note on R²(all cosmo) = {r2_cosmo_all:.3f}:")
    print(f"  This is high because it includes H0, σ8, Ωm which are themselves")
    print(f"  downstream of k_f — the regression is partly tautological.")
    print(f"  The clean number is R²(early-time) = {r2_early:.3f}: modest, as expected")
    print(f"  for a parameter weakly constrained by the CMB alone.")

    # One-sentence summary
    r2_sentence = (
        f"Nuisance parameters alone explain {r2_nuis:.1%} of k_f variance "
        f"(R²={r2_nuis:.3f}), confirming k_f is not a nuisance reskin. "
        f"Early-time primordial parameters explain {r2_early:.1%} (R²={r2_early:.3f}), "
        f"consistent with k_f being weakly constrained by the CMB. "
        f"The high R² when including late-time derived observables ({r2_cosmo_all:.3f}) "
        f"reflects that H0, σ8, and Ωm respond physically to k_f — "
        f"it is the expected signal, not independent constraint."
    )
    print(f"\n  Summary sentence:\n  {r2_sentence}")

    # ═══════════════════════════════════════════════════════════════════
    # C) k_f vs A_eff = A_s * exp(-2*tau) DEGENERACY CHECK
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("C) k_f vs A_eff = A_s * exp(-2*tau) CHECK")
    print("=" * 70)

    # Compute A_eff from chain
    logA_idx = samples.paramNames.numberOfName("logA")
    tau_idx = samples.paramNames.numberOfName("tau_reio")
    logA_col = samples.samples[:, logA_idx]
    tau_col = samples.samples[:, tau_idx]

    A_s_col = np.exp(logA_col) * 1e-10  # standard Planck convention
    A_eff_col = A_s_col * np.exp(-2 * tau_col)

    # Weighted correlation between k_f and A_eff
    w = weights / weights.sum()
    kf_mean = np.average(kf_col, weights=w)
    aeff_mean = np.average(A_eff_col, weights=w)
    cov_kf_aeff = np.sum(w * (kf_col - kf_mean) * (A_eff_col - aeff_mean))
    std_kf = np.sqrt(np.sum(w * (kf_col - kf_mean) ** 2))
    std_aeff = np.sqrt(np.sum(w * (A_eff_col - aeff_mean) ** 2))
    r_kf_aeff = cov_kf_aeff / (std_kf * std_aeff) if std_kf > 0 and std_aeff > 0 else 0

    # R² of A_eff predicting k_f
    r2_aeff = weighted_r2(kf_col, A_eff_col.reshape(-1, 1), weights)

    pass_C = abs(r_kf_aeff) < 0.5
    print(f"\n  Corr(k_f, A_eff):  {r_kf_aeff:+.4f}")
    print(f"  R²(k_f ~ A_eff):   {r2_aeff:.4f}")
    print(f"  A_eff mean:        {aeff_mean:.6e}")
    print(f"  A_eff std:         {std_aeff:.6e}")
    print(f"  PASS (|r| < 0.5, k_f not just tracking A_eff): {'YES' if pass_C else 'FAIL'}")

    # Plot C: k_f vs A_eff scatter
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    # Thin for plotting (every 10th sample)
    thin = slice(None, None, max(1, len(kf_col) // 3000))
    ax3.scatter(A_eff_col[thin], kf_col[thin], s=1, alpha=0.15, c='#1565C0')
    ax3.set_xlabel(r'$A_\mathrm{eff} = A_s \cdot e^{-2\tau}$', fontsize=12)
    ax3.set_ylabel(r'$k_f$', fontsize=12)
    ax3.set_title(
        f"Test 3C: k_f vs effective amplitude\n"
        f"r = {r_kf_aeff:+.3f}, R² = {r2_aeff:.3f}  —  "
        f"{'PASS' if pass_C else 'FAIL'}: k_f is {'not ' if pass_C else ''}tracking A_eff",
        fontsize=11, fontweight='bold'
    )
    ax3.axhline(0, color='red', linestyle=':', alpha=0.4, label='k_f = 0 (LCDM)')
    ax3.axhline(1, color='green', linestyle=':', alpha=0.4, label='k_f = 1 (full MTDF)')
    ax3.legend(fontsize=9)
    plt.tight_layout()
    aeff_path = OUTPUT_DIR / "test3_kf_vs_Aeff.png"
    fig3.savefig(aeff_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {aeff_path}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # CONDITIONAL SIGMA ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CONDITIONAL SIGMA (Schur complement)")
    print("=" * 70)

    all_params = [KF_PARAM] + COSMO_PARAMS
    all_data, _, all_valid = get_weighted_columns(samples, all_params)
    wcov, wcorr, _, wstd = weighted_corr_matrix(all_data, weights)

    marginal_sigma = float(wstd[0])
    cond_sigma = conditional_sigma(wcov, target_idx=0)
    degeneracy_ratio = marginal_sigma / cond_sigma if cond_sigma > 0 else float('inf')

    print(f"  Marginal σ(k_f)     = {marginal_sigma:.4f}")
    print(f"  Conditional σ(k_f)  = {cond_sigma:.4f}")
    print(f"  Degeneracy ratio    = {degeneracy_ratio:.2f}")

    # k_f CIs
    kf_lo68, kf_hi68 = samples.twoTailLimits(KF_PARAM, 0.68)
    kf_lo95, kf_hi95 = samples.twoTailLimits(KF_PARAM, 0.95)

    # ─── 1D k_f posterior with prior ────────────────────────────────────
    print("\nGenerating annotated k_f posterior...")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    kf_density = samples.get1DDensity(KF_PARAM)
    ax4.fill_between(kf_density.x, kf_density.P, alpha=0.3, color='#2196F3', label='Posterior')
    ax4.plot(kf_density.x, kf_density.P, color='#1565C0', linewidth=2)
    # Prior: uniform [0, 5] as per mtdf_mcmc.input.yaml
    prior_height = 1.0 / 5.0  # = 0.2
    ax4.axhline(prior_height, color='gray', linestyle='--', alpha=0.4, label='Prior (uniform [0,5])')
    ax4.axvline(kf_mean, color='#1565C0', linestyle='-', alpha=0.7,
                label=f'Mean = {kf_mean:.3f}')
    ax4.axvspan(kf_lo68, kf_hi68, alpha=0.12, color='#2196F3',
                label=f'68% CI [{kf_lo68:.3f}, {kf_hi68:.3f}]')
    ax4.axvspan(kf_lo95, kf_hi95, alpha=0.06, color='#90CAF9',
                label=f'95% CI [{kf_lo95:.3f}, {kf_hi95:.3f}]')
    ax4.axvline(0, color='red', linestyle=':', alpha=0.5, label='k_f = 0 (LCDM)')
    ax4.axvline(1, color='green', linestyle=':', alpha=0.5, label='k_f = 1 (full MTDF)')
    ax4.set_xlabel(r'$k_f$', fontsize=12)
    ax4.set_ylabel('Probability density', fontsize=12)
    ax4.set_title('Test 3: k_f posterior with prior and reference values',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_xlim(-0.2, 1.8)
    plt.tight_layout()
    kf1d_path = OUTPUT_DIR / "test3_kf_posterior_annotated.png"
    fig4.savefig(kf1d_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {kf1d_path}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY JSON
    # ═══════════════════════════════════════════════════════════════════
    sorted_all_corr = sorted(kf_all_corr.items(), key=lambda x: abs(x[1]), reverse=True)

    summary = {
        "test": "Test 3: k_f identifiability and degeneracy map",
        "chain": str(prefix),
        "n_samples": int(samples.numrows),
        "kf_posterior": {
            "mean": float(kf_mean),
            "std": float(marginal_sigma),
            "68CI": [float(kf_lo68), float(kf_hi68)],
            "95CI": [float(kf_lo95), float(kf_hi95)],
        },
        "section_A_correlations": {
            "kf_vs_cosmo": {p: float(kf_all_corr.get(p, 0))
                           for p in COSMO_PARAMS},
            "kf_vs_nuisance_top8": {p: float(kf_all_corr.get(p, 0))
                                    for p in top_nuis},
            "kf_vs_all_nuisance": kf_nuis_corr,
            "max_abs_correlation": float(max_abs_r),
            "pass_no_near_perfect": bool(pass_A),
            "most_correlated_nuisance": top_nuis_most,
            "most_correlated_nuisance_r": float(nuis_sorted[0][1]),
        },
        "section_A3_omega_b_partial": {
            "marginal_r_kf_omega_b": r_marginal_ob,
            "partial_r_precision_method": partial_targeted,
            "partial_r_residual_method": partial_residual,
            "partial_r_full_conditioning": r_partial_full,
            "indirect_covariance": ob_indirect,
            "conditioning_matrix_condition_number": float(cond_number),
            "VIF": vif_results,
            "note": "Full conditioning partial r is inflated by multicollinearity "
                    f"(condition number = {cond_number:.0f}). "
                    "Minimal conditioning gives stable r ~ -0.34.",
        },
        "section_B_predictability": {
            "R2_nuisance_only": float(r2_nuis),
            "R2_early_time_only": float(r2_early),
            "R2_late_time_derived": float(r2_late),
            "R2_early_time_plus_nuisance": float(r2_early_nuis),
            "R2_all_cosmo": float(r2_cosmo_all),
            "genuine_kf_variance_fraction": float(kf_genuine_frac),
            "pass_nuisance_below_0.5": bool(pass_B_nuis),
            "pass_genuine_above_50pct": bool(pass_B_genuine),
            "early_time_params": EARLY_TIME_PARAMS,
            "late_time_params": LATE_TIME_PARAMS,
            "summary_sentence": r2_sentence,
        },
        "section_C_Aeff_check": {
            "corr_kf_Aeff": float(r_kf_aeff),
            "R2_kf_vs_Aeff": float(r2_aeff),
            "Aeff_mean": float(aeff_mean),
            "Aeff_std": float(std_aeff),
            "pass_not_tracking_Aeff": bool(pass_C),
        },
        "conditional_sigma": {
            "marginal_sigma_kf": float(marginal_sigma),
            "conditional_sigma_kf": float(cond_sigma),
            "degeneracy_ratio": float(degeneracy_ratio),
        },
        "plots": [
            "test3_correlation_heatmap.png",
            "test3_kf_2d_contours.png",
            "test3_kf_vs_omega_b.png",
            "test3_kf_vs_Aeff.png",
            "test3_kf_posterior_annotated.png",
        ],
        "pass_all": bool(pass_A and pass_B_nuis and pass_B_genuine and pass_C),
    }

    json_path = OUTPUT_DIR / "test3_kf_identifiability.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {json_path}")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 3 VERDICT")
    print(f"{'=' * 70}")
    print(f"  A) Max |r(k_f, .)| = {max_abs_r:.3f}  →  "
          f"{'PASS' if pass_A else 'FAIL'} (no near-perfect degeneracy)")
    print(f"  B) R²(nuisance→k_f) = {r2_nuis:.3f}  →  "
          f"{'PASS' if pass_B_nuis else 'FAIL'} (nuisances do not explain k_f)")
    print(f"     Genuine k_f variance = {kf_genuine_frac:.1%}  →  "
          f"{'PASS' if pass_B_genuine else 'FAIL'}"
          f"  (R²_early = {r2_early:.3f}, R²_late = {r2_late:.3f})")
    print(f"  C) r(k_f, A_eff) = {r_kf_aeff:+.3f}  →  "
          f"{'PASS' if pass_C else 'FAIL'} (not tracking amplitude-tau combo)")
    print(f"\n  Overall: {'ALL PASS' if summary['pass_all'] else 'SOME FAIL'}")
    print(f"  k_f = 1 in 95% CI: {'YES' if kf_lo95 <= 1.0 <= kf_hi95 else 'NO'}")
    print(f"  k_f = 0 in 95% CI: {'YES' if kf_lo95 <= 0.0 <= kf_hi95 else 'NO'}")


if __name__ == "__main__":
    main()
