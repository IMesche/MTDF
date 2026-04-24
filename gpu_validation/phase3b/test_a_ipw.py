# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Test A: Stabilised Inverse Probability Weighting for NGC/SGC balance.

Fits a logistic propensity model P(NGC | z, host_mass, survey_id),
computes stabilised IPW weights, and reruns the Phase 3 regression
with weights applied to check if covariate imbalance drives the asymmetry.
"""

import numpy as np
from scipy import linalg, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mtdf_validation.phase3.gls_engine import gls_fit, COSMO_SN
from .common import FINDERS, delta_gamma_with_se


def _prepare_covariates(z, host_mass, survey_id, min_survey_count=10):
    """Build feature matrix for propensity score model.

    Returns (X, feature_names) where X is standardised.
    Groups rare surveys into 'Other'.
    """
    n = len(z)

    # Continuous: z, z^2, host_mass
    features = [z, z ** 2, host_mass]
    names = ['z', 'z_sq', 'host_mass']

    # Survey one-hot (k-1 dummies, group rare surveys)
    unique_surveys, counts = np.unique(survey_id, return_counts=True)
    survey_map = {}
    for s, c in zip(unique_surveys, counts):
        survey_map[s] = int(s) if c >= min_survey_count else -1  # -1 = Other

    mapped = np.array([survey_map[s] for s in survey_id])
    unique_mapped = np.unique(mapped)
    # Drop reference = most common mapped survey
    ref = unique_mapped[np.argmax([np.sum(mapped == u) for u in unique_mapped])]

    for u in sorted(unique_mapped):
        if u != ref:
            features.append((mapped == u).astype(float))
            label = f"survey_{u}" if u != -1 else "survey_other"
            names.append(label)

    X = np.column_stack(features)
    return X, names


def compute_propensity_scores(data, config_a):
    """Fit logistic model P(NGC=1 | covariates) and return propensity scores."""
    z = data.sub['z']
    hm = data.sub['host_mass']
    sid = data.sub['survey_id']
    ngc = data.ngc_mask.astype(int)

    min_sc = config_a.get('min_survey_count', 10)
    X_raw, feat_names = _prepare_covariates(z, hm, sid, min_sc)

    # Standardise for logistic regression stability
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = LogisticRegression(
        l1_ratio=0, C=1.0, solver='lbfgs', max_iter=1000, random_state=42
    )
    model.fit(X, ngc)

    p_hat = model.predict_proba(X)[:, 1]  # P(NGC=1)
    accuracy = model.score(X, ngc)

    # Coefficient info
    coefs = {}
    for name, coef in zip(feat_names, model.coef_[0]):
        coefs[name] = float(coef)

    return p_hat, {
        'accuracy': float(accuracy),
        'coefficients': coefs,
        'intercept': float(model.intercept_[0]),
        'n_features': len(feat_names),
        'feature_names': feat_names,
    }


def compute_ipw_weights(p_hat, ngc_mask, clip_percentiles=(1, 99)):
    """Compute stabilised IPW weights with truncation.

    Stabilised: w_i = P(region) / P(region | X_i)
    """
    n = len(p_hat)
    p_ngc = float(np.mean(ngc_mask))
    p_sgc = 1.0 - p_ngc

    weights = np.zeros(n)
    weights[ngc_mask] = p_ngc / p_hat[ngc_mask]
    weights[~ngc_mask] = p_sgc / (1.0 - p_hat[~ngc_mask])

    # Truncate at percentiles
    lo, hi = np.percentile(weights, clip_percentiles)
    n_truncated = int(np.sum((weights < lo) | (weights > hi)))
    weights = np.clip(weights, lo, hi)
    truncation_rate = n_truncated / n

    # Effective sample sizes
    w_ngc = weights[ngc_mask]
    w_sgc = weights[~ngc_mask]
    ess_ngc = float(np.sum(w_ngc) ** 2 / np.sum(w_ngc ** 2))
    ess_sgc = float(np.sum(w_sgc) ** 2 / np.sum(w_sgc ** 2))

    return weights, {
        'truncation_rate': float(truncation_rate),
        'n_truncated': n_truncated,
        'clip_lo': float(lo),
        'clip_hi': float(hi),
        'ess_ngc': ess_ngc,
        'ess_sgc': ess_sgc,
        'ess_ngc_frac': ess_ngc / float(np.sum(ngc_mask)),
        'ess_sgc_frac': ess_sgc / float(np.sum(~ngc_mask)),
        'weight_mean': float(np.mean(weights)),
        'weight_std': float(np.std(weights)),
    }


def covariate_balance_table(data, weights):
    """Standardised mean differences before/after weighting."""
    ngc = data.ngc_mask
    z = data.sub['z']
    hm = data.sub['host_mass']

    rows = []
    for name, vals in [('z', z), ('host_mass', hm)]:
        # Raw SMD
        mean_ngc = np.mean(vals[ngc])
        mean_sgc = np.mean(vals[~ngc])
        pooled_sd = np.sqrt(0.5 * (np.var(vals[ngc]) + np.var(vals[~ngc])))
        smd_raw = (mean_ngc - mean_sgc) / pooled_sd if pooled_sd > 0 else 0.0

        # Weighted SMD
        w_ngc = weights[ngc]
        w_sgc = weights[~ngc]
        wmean_ngc = np.average(vals[ngc], weights=w_ngc)
        wmean_sgc = np.average(vals[~ngc], weights=w_sgc)
        smd_weighted = (wmean_ngc - wmean_sgc) / pooled_sd if pooled_sd > 0 else 0.0

        rows.append({
            'covariate': name,
            'mean_ngc_raw': float(mean_ngc),
            'mean_sgc_raw': float(mean_sgc),
            'smd_raw': float(smd_raw),
            'mean_ngc_weighted': float(wmean_ngc),
            'mean_sgc_weighted': float(wmean_sgc),
            'smd_weighted': float(smd_weighted),
            'balanced': abs(smd_weighted) < 0.1,
        })

    return rows


def weighted_delta_chi2_test(mu, z, env_metric, host_mass, cov_inv, weights):
    """delta_chi2_test with IPW weights.

    Transforms: y -> W*y, X -> W*X, C_inv -> W*C_inv*W
    where W = diag(sqrt(weights)).
    """
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    W = np.diag(np.sqrt(weights))
    cov_inv_w = W @ cov_inv @ W

    # Null model (no environment)
    X_null = np.column_stack([np.ones(n), mass_step])
    X_null_w = W @ X_null
    res_w = W @ residual
    _, _, chi2_null, _ = gls_fit(res_w, X_null_w, cov_inv_w)

    # Full model (with environment)
    X_full = np.column_stack([np.ones(n), env_metric, mass_step])
    X_full_w = W @ X_full
    beta_full, beta_cov_full, chi2_full, dof = gls_fit(res_w, X_full_w, cov_inv_w)

    dchi2 = chi2_null - chi2_full
    from scipy.stats import chi2 as chi2_dist
    p_value = float(1.0 - chi2_dist.cdf(dchi2, 1))

    gamma_env = float(beta_full[1])
    gamma_env_err = float(np.sqrt(beta_cov_full[1, 1]))

    return {
        'gamma_env': gamma_env,
        'gamma_env_err': gamma_env_err,
        'delta_chi2': float(dchi2),
        'p_value': p_value,
        'chi2_null': float(chi2_null),
        'chi2_full': float(chi2_full),
        'dof': dof,
        'n': n,
    }


def run_test_a(data):
    """Execute Test A for all finders.

    Returns dict with per-finder results including propensity model,
    weights, balance table, and weighted gamma per region.
    """
    config_a = data.config.get('test_a', {})
    clip = tuple(config_a.get('ipw_percentile_clip', [1, 99]))

    print("\n  Fitting propensity score model...")
    p_hat, model_info = compute_propensity_scores(data, config_a)
    print(f"    Accuracy: {model_info['accuracy']:.3f}")

    weights, weight_info = compute_ipw_weights(p_hat, data.ngc_mask, clip)
    print(f"    Truncation rate: {weight_info['truncation_rate']:.3f}")
    print(f"    ESS NGC: {weight_info['ess_ngc']:.0f}/{np.sum(data.ngc_mask)} "
          f"({weight_info['ess_ngc_frac']:.1%})")
    print(f"    ESS SGC: {weight_info['ess_sgc']:.0f}/{np.sum(data.sgc_mask)} "
          f"({weight_info['ess_sgc_frac']:.1%})")

    balance = covariate_balance_table(data, weights)
    print("\n  Covariate balance (SMD):")
    for row in balance:
        print(f"    {row['covariate']:<15s} raw: {row['smd_raw']:+.3f}  "
              f"weighted: {row['smd_weighted']:+.3f}  "
              f"{'OK' if row['balanced'] else 'IMBALANCED'}")

    results = {
        'propensity_model': model_info,
        'weights_info': weight_info,
        'weights_array': weights,
        'p_hat': p_hat,
        'balance': balance,
        'finders': {},
    }

    for finder in data.config.get('data', {}).get('finders', FINDERS):
        print(f"\n  --- {finder.upper()} (weighted) ---")

        finder_res = {}

        # Weighted NGC
        w_ngc = weights[data.ngc_mask]
        d_ngc = data.d_signed_region.get((finder, 'ngc'))
        if d_ngc is not None:
            res_ngc = weighted_delta_chi2_test(
                data.sub['mu'][data.ngc_mask],
                data.sub['z'][data.ngc_mask],
                d_ngc,
                data.sub['host_mass'][data.ngc_mask],
                data.cov_inv_ngc, w_ngc,
            )
            finder_res['ngc_weighted'] = res_ngc
            print(f"    NGC weighted: gamma = {res_ngc['gamma_env']:+.4f} "
                  f"+/- {res_ngc['gamma_env_err']:.4f}")

        # Weighted SGC
        w_sgc = weights[data.sgc_mask]
        d_sgc = data.d_signed_region.get((finder, 'sgc'))
        if d_sgc is not None:
            res_sgc = weighted_delta_chi2_test(
                data.sub['mu'][data.sgc_mask],
                data.sub['z'][data.sgc_mask],
                d_sgc,
                data.sub['host_mass'][data.sgc_mask],
                data.cov_inv_sgc, w_sgc,
            )
            finder_res['sgc_weighted'] = res_sgc
            print(f"    SGC weighted: gamma = {res_sgc['gamma_env']:+.4f} "
                  f"+/- {res_sgc['gamma_env_err']:.4f}")

        # Delta gamma (weighted)
        if 'ngc_weighted' in finder_res and 'sgc_weighted' in finder_res:
            dg = delta_gamma_with_se(finder_res['ngc_weighted'],
                                     finder_res['sgc_weighted'])
            finder_res['delta_gamma_weighted'] = dg
            print(f"    Δγ weighted: {dg['delta_gamma']:+.4f} "
                  f"+/- {dg['delta_se']:.4f}, z={dg['z_score']:.2f}, "
                  f"p={dg['p_value']:.3f}")

        results['finders'][finder] = finder_res

    return results
