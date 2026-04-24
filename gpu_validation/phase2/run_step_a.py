#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 2 Step A: Sanity checkpoint.

Before running any MCMC chains, verify:
1. CosmoPower emulators load and produce physically reasonable spectra
2. CosmoPower vs CAMB agreement at Planck best-fit (<0.5% in TT)
3. Planck plik-lite likelihood gives correct chi2 for LCDM
4. MTDF correction layer produces expected shifts
5. MTDF vs LCDM delta-chi2 is in the expected ballpark

Target reference values:
  LCDM chi2 ~ 1018.57 (from full Planck plik-lite)
  MTDF chi2 ~ 1019.82 (from MTDF_04)
  Delta-chi2 ~ 1.25
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results" / "phase2"


def step1_load_emulators():
    """Step 1: Load CosmoPower emulators and verify basic output."""
    print("\n" + "=" * 70)
    print("STEP 1: Load CosmoPower emulators")
    print("=" * 70)

    from phase2.cosmopower_setup import load_emulator, PLANCK_BESTFIT, predict_dl

    results = {}
    for spec in ['TT', 'TE', 'EE']:
        t0 = time.time()
        try:
            emu = load_emulator(spec)
            elapsed = time.time() - t0
            print(f"\n  {spec} emulator loaded in {elapsed:.2f}s")
            print(f"    Parameters: {emu.parameters}")
            print(f"    N_modes: {emu.n_modes}")
            print(f"    Architecture: {emu.architecture}")
            print(f"    ell range: {emu.modes[0]} to {emu.modes[-1]}")

            t0 = time.time()
            ells, dl = predict_dl(emu, PLANCK_BESTFIT)
            pred_time = time.time() - t0

            print(f"    Prediction time: {pred_time*1000:.1f}ms")
            print(f"    D_l range: [{dl[ells>=2].min():.1f}, {dl[ells>=2].max():.1f}] uK^2")

            if spec == 'TT':
                mask = (ells >= 150) & (ells <= 300)
                peak_idx = np.argmax(dl[mask])
                peak_ell = ells[mask][peak_idx]
                peak_dl = dl[mask][peak_idx]
                print(f"    First TT peak: ell={peak_ell}, D_l={peak_dl:.1f} uK^2")
                results['tt_peak_ell'] = int(peak_ell)
                results['tt_peak_dl'] = float(peak_dl)

            results[f'{spec}_loaded'] = True
            results[f'{spec}_prediction_ms'] = pred_time * 1000

        except Exception as e:
            print(f"\n  FAILED to load {spec}: {e}")
            import traceback
            traceback.print_exc()
            results[f'{spec}_loaded'] = False
            results[f'{spec}_error'] = str(e)

    return results


def step2_compare_camb():
    """Step 2: Compare CosmoPower vs CAMB at Planck best-fit."""
    print("\n" + "=" * 70)
    print("STEP 2: CosmoPower vs CAMB comparison")
    print("=" * 70)

    from phase2.cosmopower_setup import load_emulator, PLANCK_BESTFIT, predict_dl

    ref_path = Path(__file__).parent / "camb_reference_cls.npz"
    if not ref_path.exists():
        print("  ERROR: CAMB reference file not found.")
        return {'camb_comparison': 'SKIP'}

    ref = np.load(ref_path)
    camb_ells = ref['ells']
    camb_tt = ref['cl_tt']   # D_l in uK^2
    camb_te = ref['cl_te']
    camb_ee = ref['cl_ee']

    results = {}
    for spec, camb_dl in [('TT', camb_tt), ('TE', camb_te), ('EE', camb_ee)]:
        emu = load_emulator(spec)
        ells, cp_dl = predict_dl(emu, PLANCK_BESTFIT)

        # Common ell range for plik-lite (30-2508)
        common_ells = np.arange(30, min(int(ells[-1]), int(camb_ells[-1]), 2509))
        cp_interp = np.interp(common_ells, ells.astype(float), cp_dl)
        camb_interp = np.interp(common_ells, camb_ells.astype(float), camb_dl)

        # Relative difference (avoid div by zero for TE which crosses zero)
        mask = np.abs(camb_interp) > 1.0  # at least 1 uK^2
        rel_diff = np.zeros_like(common_ells, dtype=float)
        rel_diff[mask] = (cp_interp[mask] - camb_interp[mask]) / np.abs(camb_interp[mask])

        if mask.sum() > 0:
            mean_diff = np.mean(np.abs(rel_diff[mask])) * 100
            max_diff = np.max(np.abs(rel_diff[mask])) * 100
            rms_diff = np.sqrt(np.mean(rel_diff[mask]**2)) * 100
        else:
            mean_diff = max_diff = rms_diff = float('nan')

        print(f"\n  {spec} (ell=30-2508):")
        print(f"    Mean |rel. diff|  = {mean_diff:.4f}%")
        print(f"    Max  |rel. diff|  = {max_diff:.4f}%")
        print(f"    RMS  rel. diff    = {rms_diff:.4f}%")

        ok = max_diff < 1.0
        print(f"    Status: {'PASS' if ok else 'WARN'} (threshold: <1%)")

        results[f'{spec}_mean_pct'] = mean_diff
        results[f'{spec}_max_pct'] = max_diff
        results[f'{spec}_rms_pct'] = rms_diff
        results[f'{spec}_pass'] = ok

    return results


def step3_plik_lite_chi2():
    """Step 3: Compute plik-lite chi2 for LCDM using CAMB spectra."""
    print("\n" + "=" * 70)
    print("STEP 3: Planck plik-lite chi2 (CAMB LCDM)")
    print("=" * 70)

    import camb
    from phase2.planck_lite_likelihood import PlanckLiteLikelihood

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200,
                       tau=0.0544, mnu=0.06, omk=0)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    pars.set_for_lmax(2600, lens_potential_accuracy=1)

    results_camb = camb.get_results(pars)
    powers = results_camb.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']

    ells = np.arange(totCL.shape[0])
    dl_tt = totCL[:, 0]
    dl_ee = totCL[:, 1]
    dl_te = totCL[:, 3]

    plik = PlanckLiteLikelihood()

    print(f"\n  Plik-lite data: {plik.n_used} bins used")
    print(f"    TT bins: {plik.n_tt}, TE bins: {plik.n_te}, EE bins: {plik.n_ee}")

    chi2_lcdm = plik.chi2(dl_tt, dl_te, dl_ee, A_planck=1.0)

    print(f"\n  LCDM chi2 (CAMB)  = {chi2_lcdm:.2f}")
    print(f"  Target            ~ 1018.57")
    print(f"  N_bins            = {plik.n_used}")
    print(f"  chi2/bin          = {chi2_lcdm/plik.n_used:.4f}")

    # Quick diagnostic: compare first few bins
    binned = plik.bin_theory(dl_tt, dl_te, dl_ee)
    print(f"\n  First 5 TT bins: theory vs data (C_l units)")
    for i in range(min(5, plik.n_tt)):
        print(f"    bin {i}: theory={binned[i]:.6e}  data={plik.X_data[i]:.6e}  diff={binned[i]-plik.X_data[i]:+.4e}")

    return {
        'chi2_lcdm_camb': float(chi2_lcdm),
        'n_bins': int(plik.n_used),
        'target_chi2': 1018.57,
    }


def step4_mtdf_correction():
    """Step 4: Apply MTDF correction and check shift magnitudes."""
    print("\n" + "=" * 70)
    print("STEP 4: MTDF correction layer")
    print("=" * 70)

    from phase2.mtdf_correction_layer import (
        correction_summary, sound_horizon_ratio, theta_s_ratio,
        apply_mtdf_correction
    )

    correction_summary()

    # Load CAMB reference
    ref = np.load(Path(__file__).parent / "camb_reference_cls.npz")
    ells = ref['ells']
    dl_tt = ref['cl_tt']

    # Apply MTDF correction to D_l directly
    dl_tt_mtdf = apply_mtdf_correction(ells[2:].astype(float), dl_tt[2:])

    # Check peak shifts
    for peak_name, ell_lo, ell_hi in [('1st', 180, 270), ('2nd', 490, 590), ('3rd', 770, 870)]:
        m = (ells[2:] >= ell_lo) & (ells[2:] <= ell_hi)
        if m.sum() > 0:
            lcdm_peak = ells[2:][m][np.argmax(dl_tt[2:][m])]
            mtdf_peak = ells[2:][m][np.argmax(dl_tt_mtdf[m])]
            dl_lcdm_peak = dl_tt[2:][m].max()
            dl_mtdf_peak = dl_tt_mtdf[m].max()
            print(f"\n  {peak_name} TT peak: LCDM ell={lcdm_peak} ({dl_lcdm_peak:.1f} uK^2)")
            print(f"              MTDF ell={mtdf_peak} ({dl_mtdf_peak:.1f} uK^2)")

    rs_ratio = sound_horizon_ratio()
    ts_ratio = theta_s_ratio()

    return {
        'rs_ratio': float(rs_ratio),
        'theta_s_ratio': float(ts_ratio),
        'delta_rs_pct': float((rs_ratio - 1) * 100),
        'delta_theta_pct': float((ts_ratio - 1) * 100),
    }


def step5_delta_chi2():
    """Step 5: Compute MTDF vs LCDM delta-chi2 on plik-lite."""
    print("\n" + "=" * 70)
    print("STEP 5: MTDF vs LCDM delta-chi2")
    print("=" * 70)

    import camb
    from phase2.planck_lite_likelihood import PlanckLiteLikelihood
    from phase2.mtdf_correction_layer import apply_mtdf_correction

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200,
                       tau=0.0544, mnu=0.06, omk=0)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    pars.set_for_lmax(2600, lens_potential_accuracy=1)

    results_camb = camb.get_results(pars)
    powers = results_camb.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']

    ells = np.arange(totCL.shape[0])
    dl_tt = totCL[:, 0]
    dl_ee = totCL[:, 1]
    dl_te = totCL[:, 3]

    # Apply MTDF corrections to D_l
    ells_f = ells[2:].astype(float)
    dl_tt_mtdf_full = np.copy(dl_tt)
    dl_ee_mtdf_full = np.copy(dl_ee)
    dl_te_mtdf_full = np.copy(dl_te)
    dl_tt_mtdf_full[2:] = apply_mtdf_correction(ells_f, dl_tt[2:])
    dl_ee_mtdf_full[2:] = apply_mtdf_correction(ells_f, dl_ee[2:])
    dl_te_mtdf_full[2:] = apply_mtdf_correction(ells_f, dl_te[2:])

    plik = PlanckLiteLikelihood()

    chi2_lcdm = plik.chi2(dl_tt, dl_te, dl_ee)
    chi2_mtdf = plik.chi2(dl_tt_mtdf_full, dl_te_mtdf_full, dl_ee_mtdf_full)
    delta_chi2 = chi2_mtdf - chi2_lcdm

    print(f"\n  LCDM chi2  = {chi2_lcdm:.2f}")
    print(f"  MTDF chi2  = {chi2_mtdf:.2f}")
    print(f"  Delta chi2 = {delta_chi2:+.2f}")
    print(f"\n  Target delta-chi2 ~ +1.25 (from MTDF_04)")

    if abs(delta_chi2) < 50:
        print(f"  Status: REASONABLE (small delta)")
    else:
        print(f"  Status: INVESTIGATE (large delta, expected <50)")

    return {
        'chi2_lcdm': float(chi2_lcdm),
        'chi2_mtdf': float(chi2_mtdf),
        'delta_chi2': float(delta_chi2),
        'target_delta': 1.25,
    }


def main():
    print("=" * 70)
    print("MTDF PHASE 2 STEP A: Sanity Checkpoint")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {'timestamp': datetime.now().isoformat()}

    all_results['step1_emulators'] = step1_load_emulators()
    all_results['step2_camb_comparison'] = step2_compare_camb()
    all_results['step3_plik_chi2'] = step3_plik_lite_chi2()
    all_results['step4_mtdf_correction'] = step4_mtdf_correction()
    all_results['step5_delta_chi2'] = step5_delta_chi2()

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 STEP A — SANITY REPORT SUMMARY")
    print("=" * 70)

    s1 = all_results['step1_emulators']
    s2 = all_results['step2_camb_comparison']
    s3 = all_results['step3_plik_chi2']
    s4 = all_results['step4_mtdf_correction']
    s5 = all_results['step5_delta_chi2']

    checks = []

    emu_ok = all(s1.get(f'{s}_loaded', False) for s in ['TT', 'TE', 'EE'])
    checks.append(('Emulators loaded (TT, TE, EE)', emu_ok))

    if isinstance(s2, dict) and 'TT_pass' in s2:
        cp_ok = s2.get('TT_pass', False)
    else:
        cp_ok = False
    checks.append(('CosmoPower TT vs CAMB <1%', cp_ok))

    if isinstance(s3, dict) and 'chi2_lcdm_camb' in s3:
        chi2_val = s3['chi2_lcdm_camb']
        chi2_ok = 500 < chi2_val < 2000
    else:
        chi2_val = None
        chi2_ok = False
    checks.append(('LCDM plik-lite chi2 reasonable', chi2_ok))

    if isinstance(s4, dict) and 'delta_rs_pct' in s4:
        corr_ok = abs(s4['delta_rs_pct']) < 1.0
    else:
        corr_ok = False
    checks.append(('MTDF r_s correction <1%', corr_ok))

    if isinstance(s5, dict) and 'delta_chi2' in s5:
        dchi2_ok = abs(s5['delta_chi2']) < 50
    else:
        dchi2_ok = False
    checks.append(('Delta-chi2 reasonable (<50)', dchi2_ok))

    print()
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS NEED ATTENTION'}")

    if all_pass:
        print("\n  --> Ready to proceed to Phase 2 Step B (MCMC chains)")
    else:
        print("\n  --> Review flagged items before proceeding")
        if chi2_val is not None:
            print(f"      (LCDM chi2={chi2_val:.2f} — if far from 1018, check binning)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "step_a_sanity_report.json"

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\n  Full report saved to {output_path}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
