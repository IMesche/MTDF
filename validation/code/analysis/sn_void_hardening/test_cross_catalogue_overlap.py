#!/usr/bin/env python3
"""
Hardening Test 6: Cross-Catalogue SN Overlap Diagnostic

Three void catalogues (VoidFinder, REVOLVER, VIDE) all show a positive
environment signal. But do the SAME supernovae drive the signal in all
three, or are different catalogues picking different SNe?

If the same core SNe contribute most of the Dchi2 across all catalogues,
the signal is robust to void-finder methodology. If different SNe drive
each catalogue, the signal may be an artefact of void definitions.

Method:
  1. For each catalogue, compute per-SN contribution to chi2 improvement
     (leave-one-out Dchi2 diagnostic)
  2. Rank SNe by contribution within each catalogue
  3. Compute overlap of top-N contributors across catalogues
  4. Jackknife: remove top-5 contributors from one catalogue,
     check if signal survives in the others

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
from itertools import combinations
from common import (
    PantheonData, standard_low_z_setup, compute_environment,
    delta_chi2_test, gls_fit, save_results, CATALOGUE_GROUPS, COSMO_SN
)


def per_sn_contribution(mu, z, env_metric, host_mass, cov):
    """
    Compute per-SN contribution to Dchi2 via leave-one-out.
    Returns array of Dchi2 decrements when each SN is removed.
    """
    n = len(mu)
    full_result = delta_chi2_test(mu, z, env_metric, host_mass, cov)
    full_dchi2 = full_result['delta_chi2']

    contributions = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        idx = np.where(mask)[0]
        cov_sub = cov[np.ix_(idx, idx)]
        r = delta_chi2_test(mu[idx], z[idx], env_metric[idx],
                            host_mass[idx], cov_sub)
        # Contribution = how much Dchi2 drops when this SN is removed
        contributions[i] = full_dchi2 - r['delta_chi2']

    return contributions, full_dchi2


def compute_overlap(ranks_dict, top_n=20):
    """
    Compute pairwise and three-way overlap of top-N SNe across catalogues.
    ranks_dict: {catalogue_name: array of SN indices sorted by contribution}
    """
    cat_names = list(ranks_dict.keys())
    top_sets = {k: set(v[:top_n]) for k, v in ranks_dict.items()}

    pairwise = {}
    for a, b in combinations(cat_names, 2):
        overlap = top_sets[a] & top_sets[b]
        pairwise[f"{a}_vs_{b}"] = {
            'overlap_count': len(overlap),
            'overlap_fraction': len(overlap) / top_n,
            'jaccard': len(overlap) / len(top_sets[a] | top_sets[b]),
        }

    # Three-way overlap
    if len(cat_names) == 3:
        three_way = top_sets[cat_names[0]] & top_sets[cat_names[1]] & top_sets[cat_names[2]]
        pairwise['three_way'] = {
            'overlap_count': len(three_way),
            'overlap_fraction': len(three_way) / top_n,
        }

    return pairwise


def jackknife_top_contributors(mu, z, env_metrics_dict, host_mass, cov_full,
                                ranks_dict, n_remove=5):
    """
    Remove top-N contributors from catalogue A, rerun on catalogues B and C.
    Tests whether the signal depends on a handful of outlier SNe.
    """
    cat_names = list(ranks_dict.keys())
    results = {}

    for source_cat in cat_names:
        # SNe to remove (top contributors in source catalogue)
        remove_idx = set(ranks_dict[source_cat][:n_remove])
        keep_mask = np.array([i not in remove_idx for i in range(len(mu))])
        keep_idx = np.where(keep_mask)[0]
        cov_sub = cov_full[np.ix_(keep_idx, keep_idx)]

        results[f"remove_top{n_remove}_from_{source_cat}"] = {}
        for target_cat in cat_names:
            env = env_metrics_dict[target_cat][keep_idx]
            r = delta_chi2_test(mu[keep_idx], z[keep_idx], env,
                                host_mass[keep_idx], cov_sub)
            results[f"remove_top{n_remove}_from_{source_cat}"][target_cat] = r

    return results


def run_cross_catalogue(pantheon):
    """Run the full cross-catalogue diagnostic."""
    # Use the same SN sample for all catalogues
    idx_base, cov_base = pantheon.apply_cuts(z_pv_cut=0.02, z_max=0.157)
    sn_x, sn_y, sn_z = sn_to_comoving_cached(pantheon, idx_base)

    mu = pantheon.mu[idx_base]
    z = pantheon.z[idx_base]
    hm = pantheon.host_mass[idx_base]

    env_metrics = {}
    contributions = {}
    rank_indices = {}  # Sorted SN indices by contribution

    for cat_name, (ngc_key, sgc_key, cat_type) in CATALOGUE_GROUPS.items():
        from common import load_void_pair
        vx, vy, vz, vr = load_void_pair(ngc_key, sgc_key, cat_type)
        if vx is None:
            continue

        d_signed, _, _, _ = compute_environment(sn_x, sn_y, sn_z, vx, vy, vz, vr)
        env_metrics[cat_name] = d_signed

        print(f"\n  {cat_name}: computing per-SN contributions (N={len(idx_base)})...")
        contribs, full_dchi2 = per_sn_contribution(mu, z, d_signed, hm, cov_base)
        contributions[cat_name] = contribs

        # Sort by contribution (descending)
        sorted_idx = np.argsort(-contribs)
        rank_indices[cat_name] = sorted_idx

        top5_contrib = contribs[sorted_idx[:5]].sum()
        print(f"    Full Dchi2 = {full_dchi2:.3f}, "
              f"top-5 contribute {top5_contrib:.3f} ({top5_contrib/full_dchi2*100:.1f}%)")

    # Overlap analysis
    print("\n  Computing cross-catalogue overlap...")
    for top_n in [10, 20, 30]:
        overlap = compute_overlap(rank_indices, top_n=top_n)
        print(f"    Top-{top_n} overlap:")
        for pair, info in overlap.items():
            print(f"      {pair}: {info['overlap_count']}/{top_n} "
                  f"({info['overlap_fraction']*100:.0f}%)")

    # Jackknife test
    print("\n  Jackknife: removing top-5 contributors...")
    jack = jackknife_top_contributors(mu, z, env_metrics, hm, cov_base,
                                       rank_indices, n_remove=5)
    for removal, targets in jack.items():
        print(f"    {removal}:")
        for target, r in targets.items():
            sig = "sig" if r['p_value'] < 0.05 else "n.s."
            print(f"      {target}: Dchi2={r['delta_chi2']:.3f}, "
                  f"p={r['p_value']:.4f} ({sig})")

    return {
        'n_sn': len(idx_base),
        'contributions': {k: v.tolist() for k, v in contributions.items()},
        'overlap_top10': compute_overlap(rank_indices, 10),
        'overlap_top20': compute_overlap(rank_indices, 20),
        'overlap_top30': compute_overlap(rank_indices, 30),
        'jackknife': jack,
        'top5_sn_indices': {k: v[:5].tolist() for k, v in rank_indices.items()},
    }


def sn_to_comoving_cached(pantheon, idx):
    from common import sn_to_comoving
    return sn_to_comoving(pantheon.z[idx], pantheon.ra[idx], pantheon.dec[idx])


def run(output_dir):
    print("=" * 70)
    print("HARDENING TEST 6: Cross-Catalogue SN Overlap Diagnostic")
    print("=" * 70)

    pantheon = PantheonData()
    results = run_cross_catalogue(pantheon)

    save_results(results, 'test6_cross_catalogue_overlap.json', output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Catalogue Overlap")
    print("-" * 70)
    o = results['overlap_top20']
    for pair, info in o.items():
        print(f"  Top-20 {pair}: {info['overlap_count']}/20 "
              f"({info['overlap_fraction']*100:.0f}%)")
    print("=" * 70)

    return results


if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'hardening')
    run(out)
