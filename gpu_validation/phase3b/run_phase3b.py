#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 3b: Asymmetry and Detectability Diagnostics — Main Driver.

Usage:
  source venv/bin/activate  # from repo root
  python -m mtdf_validation.phase3b.run_phase3b [--config phase3b_config.yaml] [--quick] [--tests ABCDE]
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import (
    load_config, load_phase3b_data, make_output_dir,
    compute_file_hashes, delta_gamma_with_se, region_gls,
    json_default, FINDERS,
)
from .test_a_ipw import run_test_a
from .test_b_detectability import run_test_b, write_detectability_csv, write_detectability_markdown
from .test_c_wald import run_test_c
from .test_d_geometry import run_test_d
from .test_e_injection import run_test_e
from .plotting import (
    plot_weight_distribution, plot_balance_table,
    plot_void_per_volume, plot_sn_per_void, plot_median_void_radius,
    plot_bootstrap_lr, plot_mock_delta_gamma, plot_recovery,
)


def build_summary_rows(results, data):
    """Build CSV summary rows from all test results."""
    rows = []
    finders = data.config.get('data', {}).get('finders', FINDERS)

    for finder in finders:
        # Baseline NGC/SGC from Phase 3
        res_ngc = region_gls(data, 'ngc', finder)
        res_sgc = region_gls(data, 'sgc', finder)
        dg_base = delta_gamma_with_se(res_ngc, res_sgc)

        # Also with survey FE
        res_ngc_fe = region_gls(data, 'ngc', finder, with_fe=True)
        res_sgc_fe = region_gls(data, 'sgc', finder, with_fe=True)
        dg_base_fe = delta_gamma_with_se(res_ngc_fe, res_sgc_fe)

        base_row = {
            'test': 'baseline',
            'finder': finder,
            'gamma_ngc': res_ngc['gamma_env'],
            'gamma_ngc_err': res_ngc['gamma_env_err'],
            'gamma_sgc': res_sgc['gamma_env'],
            'gamma_sgc_err': res_sgc['gamma_env_err'],
            'delta_gamma': dg_base['delta_gamma'],
            'delta_se': dg_base['delta_se'],
            'z_score': dg_base['z_score'],
            'p_value': dg_base['p_value'],
            'n_ngc': res_ngc['n'],
            'n_sgc': res_sgc['n'],
            'notes': 'Phase 3 baseline',
        }
        rows.append(base_row)

        base_fe_row = dict(base_row)
        base_fe_row['test'] = 'baseline_FE'
        base_fe_row['gamma_ngc'] = res_ngc_fe['gamma_env']
        base_fe_row['gamma_ngc_err'] = res_ngc_fe['gamma_env_err']
        base_fe_row['gamma_sgc'] = res_sgc_fe['gamma_env']
        base_fe_row['gamma_sgc_err'] = res_sgc_fe['gamma_env_err']
        base_fe_row['delta_gamma'] = dg_base_fe['delta_gamma']
        base_fe_row['delta_se'] = dg_base_fe['delta_se']
        base_fe_row['z_score'] = dg_base_fe['z_score']
        base_fe_row['p_value'] = dg_base_fe['p_value']
        base_fe_row['notes'] = 'Phase 3 with survey FE'
        rows.append(base_fe_row)

        # Test A
        if 'test_a' in results:
            ta = results['test_a'].get('finders', {}).get(finder, {})
            if 'delta_gamma_weighted' in ta:
                dg_w = ta['delta_gamma_weighted']
                ngc_w = ta.get('ngc_weighted', {})
                sgc_w = ta.get('sgc_weighted', {})
                rows.append({
                    'test': 'A_IPW',
                    'finder': finder,
                    'gamma_ngc': ngc_w.get('gamma_env', ''),
                    'gamma_ngc_err': ngc_w.get('gamma_env_err', ''),
                    'gamma_sgc': sgc_w.get('gamma_env', ''),
                    'gamma_sgc_err': sgc_w.get('gamma_env_err', ''),
                    'delta_gamma': dg_w['delta_gamma'],
                    'delta_se': dg_w['delta_se'],
                    'z_score': dg_w['z_score'],
                    'p_value': dg_w['p_value'],
                    'n_ngc': ngc_w.get('n', ''),
                    'n_sgc': sgc_w.get('n', ''),
                    'notes': 'IPW weighted',
                })

        # Test C
        if 'test_c' in results:
            tc = results['test_c'].get(finder, {})
            wald = tc.get('wald', {})
            boot = tc.get('bootstrap', {})
            if wald:
                rows.append({
                    'test': 'C_Wald',
                    'finder': finder,
                    'gamma_ngc': wald.get('gamma_ngc', ''),
                    'gamma_ngc_err': wald.get('gamma_ngc_err', ''),
                    'gamma_sgc': wald.get('gamma_sgc', ''),
                    'gamma_sgc_err': wald.get('gamma_sgc_err', ''),
                    'delta_gamma': wald.get('delta_gamma', ''),
                    'delta_se': wald.get('delta_se', ''),
                    'z_score': wald.get('z_score', ''),
                    'p_value': wald.get('p_wald', ''),
                    'n_ngc': wald.get('n_ngc', ''),
                    'n_sgc': wald.get('n_sgc', ''),
                    'notes': f"Wald p={wald.get('p_wald', ''):.3f}, "
                             f"boot p={boot.get('p_bootstrap', ''):.3f}"
                             if boot else f"Wald p={wald.get('p_wald', ''):.3f}",
                })

        # Test D
        if 'test_d' in results:
            td = results['test_d'].get(finder, {})
            if td:
                rows.append({
                    'test': 'D_geometry',
                    'finder': finder,
                    'gamma_ngc': '', 'gamma_ngc_err': '',
                    'gamma_sgc': '', 'gamma_sgc_err': '',
                    'delta_gamma': td.get('obs_delta_gamma', ''),
                    'delta_se': '', 'z_score': '',
                    'p_value': td.get('p_empirical', ''),
                    'n_ngc': '', 'n_sgc': '',
                    'notes': f"mock p={td.get('p_empirical', ''):.3f}, "
                             f"n_mocks={td.get('n_mocks', '')}",
                })

        # Test E
        if 'test_e' in results:
            te = results['test_e'].get(finder, {})
            for strength_key, sr in te.items():
                if not isinstance(sr, dict) or 'gamma_inject' not in sr:
                    continue
                rows.append({
                    'test': f'E_inject_{strength_key}',
                    'finder': finder,
                    'gamma_ngc': '', 'gamma_ngc_err': '',
                    'gamma_sgc': sr.get('gamma_recovered_mean', ''),
                    'gamma_sgc_err': sr.get('gamma_recovered_std', ''),
                    'delta_gamma': '', 'delta_se': '',
                    'z_score': '',
                    'p_value': '',
                    'n_ngc': '', 'n_sgc': '',
                    'notes': f"inject={sr.get('gamma_inject', ''):.4f}, "
                             f"bias={sr.get('bias', ''):.4f}, "
                             f"det@2σ={sr.get('detection_rate_2sigma', ''):.1%}",
                })

    return rows


def generate_report(out_dir, results, data, elapsed, hashes):
    """Generate PHASE3B_REPORT.md."""
    lines = []
    lines.append("# Phase 3b: Asymmetry and Detectability Diagnostics\n\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")

    # Test A
    if 'test_a' in results:
        lines.append("## Test A: IPW Matched Redshift (Stabilised)\n\n")
        wi = results['test_a'].get('weights_info', {})
        lines.append(f"Logistic propensity model fitted to predict NGC/SGC from "
                     f"z, z^2, host_mass, and survey_id.\n")
        lines.append(f"Weight truncation: {wi.get('truncation_rate', 0):.1%} "
                     f"at [{wi.get('clip_lo', 0):.3f}, {wi.get('clip_hi', 0):.3f}].\n")
        lines.append(f"ESS: NGC={wi.get('ess_ngc', 0):.0f} "
                     f"({wi.get('ess_ngc_frac', 0):.0%}), "
                     f"SGC={wi.get('ess_sgc', 0):.0f} "
                     f"({wi.get('ess_sgc_frac', 0):.0%}).\n\n")
        for finder, fr in results['test_a'].get('finders', {}).items():
            dg = fr.get('delta_gamma_weighted', {})
            if dg:
                lines.append(f"**{finder}**: Δγ_weighted = {dg['delta_gamma']:+.4f} "
                             f"± {dg['delta_se']:.4f} (z={dg['z_score']:.2f}, "
                             f"p={dg['p_value']:.3f})\n\n")

    # Test B
    if 'test_b' in results:
        lines.append("## Test B: Void-SN Joint Density Mapping\n\n")
        lines.append("See `tables/test_b_detectability.csv` and plots for full details.\n\n")
        summary = results['test_b'].get('summary', {})
        for key, val in summary.items():
            lines.append(f"- **{key}**: {val['total_voids']} voids, "
                        f"{val['total_sn']} SNe, "
                        f"SN/void={val['overall_sn_per_void']:.2f}\n")
        lines.append("\n")

    # Test C
    if 'test_c' in results:
        lines.append("## Test C: Wald Test + Parametric Bootstrap\n\n")
        for finder, fr in results['test_c'].items():
            w = fr.get('wald', {})
            b = fr.get('bootstrap', {})
            lines.append(f"**{finder}**: Wald p={w.get('p_wald', 0):.3f}, "
                        f"bootstrap p={b.get('p_bootstrap', 0):.3f} "
                        f"(Δγ={w.get('delta_gamma', 0):+.4f}±{w.get('delta_se', 0):.4f})\n\n")

    # Test D
    if 'test_d' in results:
        lines.append("## Test D: Geometry and Mode-Coupling Check\n\n")
        for finder, fr in results['test_d'].items():
            lines.append(f"**{finder}**: p_empirical={fr.get('p_empirical', 0):.3f} "
                        f"({fr.get('n_mocks', 0)} mocks, "
                        f"observed Δγ={fr.get('obs_delta_gamma', 0):+.4f}, "
                        f"mock std={fr.get('mock_std', 0):.4f})\n\n")

    # Test E
    if 'test_e' in results:
        lines.append("## Test E: Null Signal Injection (SGC Recovery)\n\n")
        for finder, fr in results['test_e'].items():
            lines.append(f"**{finder}** (γ_NGC={fr.get('gamma_ngc', 0):.4f}):\n\n")
            for key, sr in fr.items():
                if not isinstance(sr, dict) or 'gamma_inject' not in sr:
                    continue
                lines.append(f"- {key}: inject={sr['gamma_inject']:+.4f}, "
                            f"recover={sr['gamma_recovered_mean']:+.4f}±"
                            f"{sr['gamma_recovered_std']:.4f}, "
                            f"bias={sr['bias']:+.4f}, "
                            f"det@2σ={sr['detection_rate_2sigma']:.0%}\n")
            lines.append("\n")

    # Interpretation
    lines.append("## Interpretation\n\n")
    lines.append("1. **SGC sensitivity-limited?** See Test B detectability table "
                "and Test E recovery rates.\n")
    lines.append("2. **Geometry explains asymmetry?** See Test D empirical p-values.\n")
    lines.append("3. **Δγ statistically robust?** See Test C Wald and bootstrap p-values, "
                "Test A IPW-weighted results.\n\n")

    lines.append(f"---\nFile hashes: {hashes}\n")

    with open(Path(out_dir) / 'PHASE3B_REPORT.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3b: Asymmetry and Detectability Diagnostics')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to phase3b_config.yaml')
    parser.add_argument('--quick', action='store_true',
                        help='Reduce bootstrap/mock counts for fast testing')
    parser.add_argument('--tests', type=str, default='ABCDE',
                        help='Which tests to run (e.g., "ACE")')
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config(None)

    if args.quick:
        config['test_c']['n_bootstrap'] = 100
        config['test_d']['n_mocks'] = 20
        config['test_e']['n_realizations'] = 20

    t_start = time.time()

    # Banner
    print("=" * 70)
    print("MTDF PHASE 3b: ASYMMETRY AND DETECTABILITY DIAGNOSTICS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Tests: {args.tests}")
    if args.quick:
        print("MODE: QUICK (reduced counts)")
    print("=" * 70)

    # Output directory
    base_results = Path(__file__).resolve().parent.parent / "results" / "phase3b"
    out_dir = make_output_dir(base_results)
    print(f"Output: {out_dir}")

    # Save frozen config
    import yaml
    with open(out_dir / 'phase3b_config_used.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load data (once)
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = load_phase3b_data(config)

    # File hashes
    data_dir = config['data'].get('data_dir')
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent.parent
                       / "validation" / "data")
    hashes = compute_file_hashes(data_dir)

    # Get plot formats from config
    plot_fmts = tuple(config.get('output', {}).get('plot_formats', ['png', 'pdf']))

    # Run tests
    results = {}

    if 'A' in args.tests.upper():
        print("\n" + "=" * 70)
        print("TEST A: IPW MATCHED REDSHIFT (STABILISED)")
        print("=" * 70)
        results['test_a'] = run_test_a(data)
        if config.get('output', {}).get('save_plots', True):
            plot_weight_distribution(
                results['test_a']['weights_array'],
                data.ngc_mask, out_dir, plot_fmts)
            plot_balance_table(results['test_a']['balance'], out_dir, plot_fmts)

    if 'B' in args.tests.upper():
        print("\n" + "=" * 70)
        print("TEST B: VOID-SN JOINT DENSITY MAPPING")
        print("=" * 70)
        results['test_b'] = run_test_b(data)
        table = results['test_b']['table']
        write_detectability_csv(table, out_dir / 'tables' / 'test_b_detectability.csv')
        write_detectability_markdown(table, out_dir / 'tables' / 'test_b_detectability.md')
        if config.get('output', {}).get('save_plots', True):
            plot_void_per_volume(table, out_dir, plot_fmts)
            plot_sn_per_void(table, out_dir, plot_fmts)
            plot_median_void_radius(table, out_dir, plot_fmts)

    if 'C' in args.tests.upper():
        print("\n" + "=" * 70)
        print("TEST C: WALD TEST + PARAMETRIC BOOTSTRAP")
        print("=" * 70)
        results['test_c'] = run_test_c(data)
        if config.get('output', {}).get('save_plots', True):
            for finder, fr in results['test_c'].items():
                boot_lr = fr.get('bootstrap_lr_array')
                obs_lr = fr.get('bootstrap', {}).get('obs_lr',
                         fr.get('wald', {}).get('wald_stat', 0))
                if boot_lr is not None:
                    plot_bootstrap_lr(
                        fr['bootstrap']['obs_lr'], boot_lr,
                        finder, out_dir, plot_fmts)

    if 'D' in args.tests.upper():
        print("\n" + "=" * 70)
        print("TEST D: GEOMETRY AND MODE-COUPLING CHECK")
        print("=" * 70)
        results['test_d'] = run_test_d(data)
        if config.get('output', {}).get('save_plots', True):
            for finder, fr in results['test_d'].items():
                plot_mock_delta_gamma(
                    fr['obs_delta_gamma'], fr['mock_delta_gammas'],
                    finder, out_dir, plot_fmts)

    if 'E' in args.tests.upper():
        print("\n" + "=" * 70)
        print("TEST E: NULL SIGNAL INJECTION (SGC RECOVERY)")
        print("=" * 70)
        results['test_e'] = run_test_e(data)
        if config.get('output', {}).get('save_plots', True):
            for finder, fr in results['test_e'].items():
                for key, sr in fr.items():
                    if not isinstance(sr, dict) or 'gammas_array' not in sr:
                        continue
                    plot_recovery(
                        sr['gamma_inject'], sr['gammas_array'],
                        key, finder, out_dir, plot_fmts)

    elapsed = time.time() - t_start

    # Build summary CSV rows
    summary_rows = build_summary_rows(results, data)
    if summary_rows:
        csv_path = out_dir / 'phase3b_summary.csv'
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[SAVED] {csv_path}")

    # Save JSON summary (strip large arrays)
    json_results = {}
    for tk, tv in results.items():
        if tk == 'test_a':
            # Don't serialize weight/phat arrays
            jr = dict(tv)
            jr.pop('weights_array', None)
            jr.pop('p_hat', None)
            json_results[tk] = jr
        elif tk == 'test_c':
            jr = {}
            for f, fv in tv.items():
                jf = dict(fv)
                jf.pop('bootstrap_lr_array', None)
                jr[f] = jf
            json_results[tk] = jr
        elif tk == 'test_d':
            jr = {}
            for f, fv in tv.items():
                jf = dict(fv)
                jf.pop('mock_delta_gammas', None)
                jr[f] = jf
            json_results[tk] = jr
        elif tk == 'test_e':
            jr = {}
            for f, fv in tv.items():
                if isinstance(fv, dict):
                    jf = {}
                    for k, v in fv.items():
                        if isinstance(v, dict):
                            jv = dict(v)
                            jv.pop('gammas_array', None)
                            jf[k] = jv
                        else:
                            jf[k] = v
                    jr[f] = jf
                else:
                    jr[f] = fv
            json_results[tk] = jr
        else:
            json_results[tk] = tv

    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'file_hashes': hashes,
        'config': config,
        'results': json_results,
    }
    json_path = out_dir / 'phase3b_summary.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=json_default)
    print(f"[SAVED] {json_path}")

    # Generate report
    generate_report(out_dir, results, data, elapsed, hashes)
    print(f"[SAVED] {out_dir / 'PHASE3B_REPORT.md'}")

    print(f"\n{'=' * 70}")
    print(f"PHASE 3b COMPLETE: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
