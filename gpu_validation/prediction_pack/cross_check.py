#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Cross-check: CLASS sigma8(z) and f(z) vs ODE prediction pack (LCDM)."""

import sys, json, numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'class_mtdf'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'class_mtdf' / 'python'))
import classy

# Phase 5 LCDM posterior
params = {
    'output': 'tCl,pCl,lCl,mPk',
    'lensing': 'yes',
    'l_max_scalars': 2500,
    'P_k_max_h/Mpc': 1.0,
    'H0': 67.382,
    'omega_b': 0.022359,
    'omega_cdm': 0.119919,
    'A_s': np.exp(3.0435) * 1e-10,
    'n_s': 0.9646,
    'tau_reio': 0.053964,
}

# Dense z grid for numerical f(z) — CLASS limits z_pk count
z_dense = np.linspace(0.0, 3.0, 100)  # Stay under CLASS _Z_PK_NUM_MAX_
params['z_pk'] = ','.join([f'{z:.4f}' for z in z_dense])

cosmo = classy.Class()
cosmo.set(params)
cosmo.compute()

h = cosmo.h()
sig8_class = np.array([cosmo.sigma(8.0/h, z) for z in z_dense])

# f(z) from numerical derivative: f = -(1+z) * d ln sigma8 / dz
ln_sig8 = np.log(sig8_class)
f_class = np.zeros_like(z_dense)
for i in range(1, len(z_dense)-1):
    dz = z_dense[i+1] - z_dense[i-1]
    dlns = ln_sig8[i+1] - ln_sig8[i-1]
    f_class[i] = -(1 + z_dense[i]) * dlns / dz
f_class[0] = -(1+z_dense[0]) * (ln_sig8[1]-ln_sig8[0]) / (z_dense[1]-z_dense[0])
f_class[-1] = -(1+z_dense[-1]) * (ln_sig8[-1]-ln_sig8[-2]) / (z_dense[-1]-z_dense[-2])

fsig8_class = f_class * sig8_class

# Load ODE predictions from JSON
with open(str(Path(__file__).parent.parent.parent / 'validation' / 'output' / 'prediction_pack' / 'mtdf_prediction_pack.json')) as fp:
    pack = json.load(fp)

z_ode = np.array(pack['grids']['z_fine'])
sig8_ode = np.array(pack['grids']['sigma8_lcdm_z'])
fsig8_ode = np.array(pack['grids']['fsigma8_lcdm_z'])
# f not stored directly; reconstruct from fsig8 / sig8
f_ode = np.where(sig8_ode > 0, fsig8_ode / sig8_ode, 0.0)

# ── Shape comparison (normalize to z=0) ──
# This is the fair test: CLASS and ODE use different sigma8(0) normalizations
# but the SHAPE (D(z)/D(0) and f(z)) should match.
D_class = sig8_class / sig8_class[0]
D_ode = sig8_ode / sig8_ode[0]

print("=" * 80)
print("CROSS-CHECK: CLASS vs ODE growth factor and growth rate (LCDM)")
print("=" * 80)
print()
print(f"sigma8(0): CLASS = {sig8_class[0]:.5f}, ODE = {sig8_ode[0]:.5f}")
print(f"  (CLASS computes sigma8 from full P(k); ODE uses Phase 5 posterior = 0.810)")
print()

print("SHAPE COMPARISON (D(z) = sigma8(z)/sigma8(0), normalized):")
print(f"{'z':>6} | {'D_CLASS':>10} | {'D_ODE':>10} | {'dD(%)':>8} | {'f_CLASS':>8} | {'f_ODE':>8} | {'df(%)':>8}")
print("-" * 80)

z_check = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
max_dD = 0
max_df = 0

for z in z_check:
    dc = float(np.interp(z, z_dense, D_class))
    do = float(np.interp(z, z_ode, D_ode))
    fc = float(np.interp(z, z_dense, f_class))
    fo = float(np.interp(z, z_ode, f_ode))

    dd = (do - dc) / dc * 100 if dc > 0 else 0
    dfp = (fo - fc) / fc * 100 if abs(fc) > 1e-6 else 0

    max_dD = max(max_dD, abs(dd))
    max_df = max(max_df, abs(dfp))

    print(f"{z:6.1f} | {dc:10.5f} | {do:10.5f} | {dd:+8.3f} | {fc:8.5f} | {fo:8.5f} | {dfp:+8.3f}")

print()
print(f"Max |delta D|: {max_dD:.3f}%")
print(f"Max |delta f|: {max_df:.3f}%")
print()

# fσ8 comparison (using each method's own sigma8(0))
print("fσ8(z) COMPARISON (each using own sigma8(0) normalization):")
print(f"{'z':>6} | {'fσ8_CLASS':>10} | {'fσ8_ODE':>10} | {'Δ(%)':>8}")
print("-" * 50)

for z in z_check:
    fsc = float(np.interp(z, z_dense, fsig8_class))
    fso = float(np.interp(z, z_ode, fsig8_ode))
    dp = (fso - fsc) / fsc * 100 if abs(fsc) > 1e-6 else 0
    print(f"{z:6.1f} | {fsc:10.5f} | {fso:10.5f} | {dp:+8.3f}")

print()

# Verdict
if max_dD < 1.0 and max_df < 2.0:
    print("VERDICT: PASS")
    print(f"  Growth factor D(z) agrees to {max_dD:.2f}% (threshold: 1%)")
    print(f"  Growth rate f(z) agrees to {max_df:.2f}% (threshold: 2%)")
    print("  ODE machinery is trustworthy for MTDF extension.")
else:
    print("VERDICT: NEEDS INVESTIGATION")
    print(f"  D(z) max diff: {max_dD:.2f}%")
    print(f"  f(z) max diff: {max_df:.2f}%")

cosmo.struct_cleanup()
cosmo.empty()
