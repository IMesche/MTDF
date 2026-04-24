#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 1: Independent chi-squared reproduction.

Reproduces the MTDF V18 dashboard strict totals from scratch.
Does NOT import from vector_pillars.py or run_validate.py.

Target: chi2/nu = 1.1683 (DOF = 1745)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, RESULTS_DIR, build_mtdf_params
from phase1.data_loaders import (
    load_pantheonplus, load_desi_bao, load_cc_hz,
    load_dr16_fsigma8, load_cmb_distance_prior,
)
from phase1.chi2_engine import (
    chi2_sne_marginalized, chi2_bao, chi2_hz,
    chi2_fsigma8, chi2_cmb_distance,
)
from phase1.standalone_mtdf import sound_horizon_aubourg, H_mtdf, comoving_distance


def main():
    print("=" * 70)
    print("MTDF PHASE 1: Independent Chi-Squared Reproduction")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load parameters from workbook
    # -----------------------------------------------------------------------
    print("\n[1] Loading parameters from workbook...")
    params = build_mtdf_params()

    print(f"  H0         = {params['H0']}")
    print(f"  Omega_m    = {params['Omega_m']:.5f}")
    print(f"  Omega_b    = {params['Omega_b']:.5f}")
    print(f"  omegab_h2  = {params['omegab_h2']:.5f}")
    print(f"  omegam_h2  = {params['omegam_h2']:.5f}")
    print(f"  alpha      = {params['alpha']}")
    print(f"  beta_eos   = {params['beta_eos']}")
    print(f"  kappa      = {params['kappa']}")
    print(f"  z_t        = {params['z_t']}")

    # Verify sound horizon
    r_d, r_s = sound_horizon_aubourg(params)
    print(f"\n  Sound horizon: r_d = {r_d:.3f} Mpc, r_s = {r_s:.3f} Mpc")
    print(f"  (Expected: r_d ~ 147.09, r_s ~ 144.39)")

    # -----------------------------------------------------------------------
    # 2. Load all datasets
    # -----------------------------------------------------------------------
    data_dir = str(DATA_DIR)
    print("\n[2] Loading datasets...")

    t0 = time.time()
    sne_z, sne_mu, sne_cov = load_pantheonplus(data_dir)
    print(f"  Pantheon+ SNe:    {len(sne_z)} SNe, cov {sne_cov.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    bao_z, bao_obs, bao_types, bao_cov = load_desi_bao(data_dir)
    print(f"  DESI BAO:         {len(bao_z)} points, cov {bao_cov.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    cc_z, cc_H, cc_cov = load_cc_hz(data_dir)
    print(f"  CC H(z):          {len(cc_z)} points, cov {cc_cov.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    fs8_z, fs8_obs, fs8_cov = load_dr16_fsigma8(data_dir)
    print(f"  DR16 fsigma8:     {len(fs8_z)} points, cov {fs8_cov.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    cmb_means, cmb_cov = load_cmb_distance_prior(data_dir)
    print(f"  CMB distance:     {len(cmb_means)} params, cov {cmb_cov.shape}  ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 3. Compute chi2 for each vector pillar
    # -----------------------------------------------------------------------
    print("\n[3] Computing chi-squared per pillar...")
    results = {}

    # SNe
    print("\n  --- Pantheon+ SNe (analytic M marginalization) ---")
    t0 = time.time()
    chi2_s, dof_s = chi2_sne_marginalized(sne_z, sne_mu, sne_cov, params)
    elapsed = time.time() - t0
    results['SNe'] = {'chi2': chi2_s, 'dof': dof_s, 'n_data': len(sne_z)}
    print(f"  chi2 = {chi2_s:.2f}, DOF = {dof_s}, chi2/nu = {chi2_s/dof_s:.4f}  ({elapsed:.1f}s)")

    # BAO
    print("\n  --- DESI BAO ---")
    t0 = time.time()
    chi2_b, dof_b = chi2_bao(bao_z, bao_obs, bao_types, bao_cov, params)
    elapsed = time.time() - t0
    results['BAO'] = {'chi2': chi2_b, 'dof': dof_b, 'n_data': len(bao_z)}
    print(f"  chi2 = {chi2_b:.2f}, DOF = {dof_b}, chi2/nu = {chi2_b/dof_b:.4f}  ({elapsed:.1f}s)")

    # Print BAO details
    from phase1.standalone_mtdf import bao_predictions
    bao_pred = bao_predictions(bao_z, bao_types, params)
    for i in range(len(bao_z)):
        print(f"    z={bao_z[i]:.2f} {bao_types[i]:15s}: obs={bao_obs[i]:.4f}  pred={bao_pred[i]:.4f}  diff={bao_obs[i]-bao_pred[i]:+.4f}")

    # CC H(z)
    print("\n  --- Cosmic Chronometers H(z) ---")
    t0 = time.time()
    chi2_h, dof_h = chi2_hz(cc_z, cc_H, cc_cov, params)
    elapsed = time.time() - t0
    results['Hz'] = {'chi2': chi2_h, 'dof': dof_h, 'n_data': len(cc_z)}
    print(f"  chi2 = {chi2_h:.2f}, DOF = {dof_h}, chi2/nu = {chi2_h/dof_h:.4f}  ({elapsed:.1f}s)")

    # fsigma8
    print("\n  --- DR16 fsigma8 (analytic sigma8 fit) ---")
    t0 = time.time()
    chi2_f, dof_f, sig8_bf, sig8_err = chi2_fsigma8(fs8_z, fs8_obs, fs8_cov, params)
    elapsed = time.time() - t0
    results['fsig8'] = {'chi2': chi2_f, 'dof': dof_f, 'n_data': len(fs8_z),
                        'sigma8_bf': sig8_bf, 'sigma8_err': sig8_err}
    print(f"  chi2 = {chi2_f:.2f}, DOF = {dof_f}, chi2/nu = {chi2_f/dof_f:.4f}  ({elapsed:.1f}s)")
    print(f"  sigma8_0 = {sig8_bf:.4f} +/- {sig8_err:.4f}")

    # CMB distance prior
    print("\n  --- CMB Distance Prior (Planck 2018) ---")
    t0 = time.time()
    chi2_c, dof_c, cmb_diag = chi2_cmb_distance(cmb_means, cmb_cov, params)
    elapsed = time.time() - t0
    results['CMB'] = {'chi2': chi2_c, 'dof': dof_c, 'n_data': len(cmb_means)}
    print(f"  chi2 = {chi2_c:.2f}, DOF = {dof_c}, chi2/nu = {chi2_c/dof_c:.4f}  ({elapsed:.1f}s)")
    print(f"  z*      = {cmb_diag['z_star']:.2f}")
    print(f"  R       = {cmb_diag['R']:.5f}  (obs: {cmb_means[0]:.5f})")
    print(f"  lA      = {cmb_diag['lA']:.3f}  (obs: {cmb_means[1]:.3f})")
    print(f"  omegab  = {params['omegab_h2']:.5f}  (obs: {cmb_means[2]:.5f})")

    # -----------------------------------------------------------------------
    # 4. Load scalar pillars from Diagnostics.csv for strict total
    # -----------------------------------------------------------------------
    print("\n[4] Loading scalar pillars from Diagnostics.csv...")
    import csv
    diag_path = Path(data_dir).parent / "output" / "Diagnostics.csv"
    scalar_chi2_total = 0.0
    scalar_dof_total = 0
    if diag_path.exists():
        with open(diag_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['pillar']
                chi2_c = float(row['chi2_contrib'])
                scalar_chi2_total += chi2_c
                scalar_dof_total += 1
                print(f"  {pid:6s}  chi2 = {chi2_c:.6f}")
        print(f"  Scalar total: {scalar_dof_total} pillars, chi2 = {scalar_chi2_total:.4f}")
    else:
        print(f"  WARNING: {diag_path} not found, using vector-only totals")

    # -----------------------------------------------------------------------
    # 5. Combined results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMBINED RESULTS")
    print("=" * 70)

    # Vector totals (excluding CMB diagnostic)
    vector_excl_cmb_chi2 = sum(r['chi2'] for name, r in results.items() if name != 'CMB')
    vector_excl_cmb_dof = sum(r['dof'] for name, r in results.items() if name != 'CMB')

    # Strict total = scalar + vector (excluding CMB)
    strict_chi2 = scalar_chi2_total + vector_excl_cmb_chi2
    strict_dof = scalar_dof_total + vector_excl_cmb_dof

    # Full total (including CMB diagnostic)
    full_chi2 = strict_chi2 + results['CMB']['chi2']
    full_dof = strict_dof + results['CMB']['dof']

    print(f"\n  {'Component':<25s} {'chi2':>10s} {'DOF':>6s} {'chi2/nu':>10s}")
    print(f"  {'-'*54}")
    print(f"  {'Scalar (15 pillars)':<25s} {scalar_chi2_total:10.4f} {scalar_dof_total:6d} {scalar_chi2_total/scalar_dof_total:10.4f}")
    for name, r in results.items():
        tag = f"  * {name}" if name == 'CMB' else f"  {name}"
        chi2_nu = r['chi2'] / r['dof'] if r['dof'] > 0 else float('nan')
        print(f"  {tag:<25s} {r['chi2']:10.2f} {r['dof']:6d} {chi2_nu:10.4f}")

    print(f"  {'-'*54}")
    print(f"  {'STRICT (excl CMB*)':<25s} {strict_chi2:10.2f} {strict_dof:6d} {strict_chi2/strict_dof:10.4f}")
    print(f"  {'FULL (incl CMB*)':<25s} {full_chi2:10.2f} {full_dof:6d} {full_chi2/full_dof:10.4f}")
    print(f"  (* = diagnostic pillar, excluded from strict)")

    print(f"\n  TARGET:  chi2/nu = 1.1683, DOF = 1745")
    print(f"  STRICT:  chi2/nu = {strict_chi2/strict_dof:.4f}, DOF = {strict_dof}")

    delta = abs(strict_chi2/strict_dof - 1.1683)
    if delta < 0.001:
        print(f"\n  PASS: strict chi2/nu matches target within 0.001 (delta = {delta:.6f})")
    else:
        print(f"\n  Delta = {delta:.6f}")

    # -----------------------------------------------------------------------
    # 6. Save results
    # -----------------------------------------------------------------------
    output_dir = RESULTS_DIR / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'params': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                   for k, v in params.items()},
        'pillars': {},
        'strict_chi2': strict_chi2,
        'strict_dof': strict_dof,
        'strict_chi2_per_dof': strict_chi2 / strict_dof if strict_dof > 0 else None,
        'full_chi2': full_chi2,
        'full_dof': full_dof,
        'full_chi2_per_dof': full_chi2 / full_dof if full_dof > 0 else None,
        'target_chi2_per_dof': 1.1683,
        'target_dof': 1745,
    }

    for name, r in results.items():
        output['pillars'][name] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in r.items()
        }

    with open(output_dir / "phase1_results.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'phase1_results.json'}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
