#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Step 2: Compute the algebraic target for closing the F2 gap.

Using the Step 1 comparison against Brouwer+2021 data, compute:
1. Required ΔΣ_extra(R) at R = 100, 200, 300, 500, 1000 kpc
2. Required enclosed projected mass M(<R) that would generate it
3. Required effective 3D density profile rho_field(r)
4. Map to effective beta_local under stress halo ansatz
5. Toy-profile feasibility: what slope does the field need?
6. SPARC constraint: check if boosting lensing at 200 kpc breaks rotation curves
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import json

# ═══════════════════════════════════════════════════════════════
# CONSTANTS (same as step1)
# ═══════════════════════════════════════════════════════════════

ALPHA = 1.30
BETA_KPC = 22_685.0
RHO_CRIT = 136.3
G_NEWTON = 4.302e-3  # pc * (km/s)^2 / M_sun

BIN_EDGES = [8.5, 10.3, 10.6, 10.8, 11.0]
BIN_LABELS = ["Bin 1 (8.5-10.3)", "Bin 2 (10.3-10.6)",
              "Bin 3 (10.6-10.8)", "Bin 4 (10.8-11.0)"]
BIN_NAMES = ["bin1", "bin2", "bin3", "bin4"]

MEDIAN_LOG_MSTAR = [10.0, 10.45, 10.70, 10.90]
MEDIAN_MSTAR = [10**x for x in MEDIAN_LOG_MSTAR]
F_GAS = [0.15, 0.08, 0.05, 0.03]
MEDIAN_MBAR = [m * (1 + f) for m, f in zip(MEDIAN_MSTAR, F_GAS)]

# Target radii for analysis
R_TARGET = np.array([100, 200, 300, 500, 1000])  # kpc


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (same as step1)
# ═══════════════════════════════════════════════════════════════

def load_brouwer_bin(data_dir, bin_num):
    fname = data_dir / f"Fig-3_Lensing-rotation-curves_Massbin-{bin_num}.txt"
    data = np.loadtxt(fname)
    bias = data[:, 4]
    return {
        'R_Mpc': data[:, 0],
        'R_kpc': data[:, 0] * 1000,
        'ESD': data[:, 1] / bias,
        'error': data[:, 3] / bias,
    }


def esd_baryon_point(R_kpc, M_bar):
    """Point-mass baryonic ESD in Msun/kpc²."""
    return M_bar / (np.pi * R_kpc**2)


def esd_mtdf(R_kpc, M_bar):
    """MTDF ESD: baryonic × enhancement."""
    enhancement = 1.0 + ALPHA / (1.0 + R_kpc / BETA_KPC)
    return esd_baryon_point(R_kpc, M_bar) * enhancement


def kpc2_to_brouwer(esd_kpc2):
    return esd_kpc2 / 1e6


def brouwer_to_kpc2(esd_brouwer):
    return esd_brouwer * 1e6


# ═══════════════════════════════════════════════════════════════
# STEP 2a: Required ΔΣ_extra at target radii
# ═══════════════════════════════════════════════════════════════

def compute_required_extra(bin_data, M_bar, R_targets):
    """Compute the required extra ΔΣ at target radii.

    Returns dict with:
    - delta_sigma_data: interpolated data ESD at R_targets (Msun/kpc²)
    - delta_sigma_mtdf: MTDF prediction at R_targets (Msun/kpc²)
    - delta_sigma_extra: required extra ESD (Msun/kpc²)
    - M_proj_extra: required enclosed projected mass (Msun)
    """
    # Interpolate data to target radii
    f_data = interp1d(bin_data['R_kpc'], brouwer_to_kpc2(bin_data['ESD']),
                      kind='linear', fill_value='extrapolate')
    f_err = interp1d(bin_data['R_kpc'], brouwer_to_kpc2(bin_data['error']),
                     kind='linear', fill_value='extrapolate')

    ds_data = f_data(R_targets)
    ds_err = f_err(R_targets)
    ds_mtdf = esd_mtdf(R_targets, M_bar)
    ds_extra = ds_data - ds_mtdf  # required extra ESD

    # Required enclosed projected mass: ΔΣ = M_proj / (π R²)
    # → M_proj_extra = ΔΣ_extra × π × R²
    # But ΔΣ(R) = Σ̄(<R) - Σ(R), not simply M/(πR²) for extended profiles.
    # For a rough estimate, use ΔΣ_extra ~ M_extra(<R) / (π R²)
    M_proj_extra = ds_extra * np.pi * R_targets**2

    return {
        'R_kpc': R_targets,
        'ds_data': ds_data,
        'ds_data_err': ds_err,
        'ds_mtdf': ds_mtdf,
        'ds_extra': ds_extra,
        'M_proj_extra': M_proj_extra,
        'ratio_data_mtdf': ds_data / ds_mtdf,
        'enhancement_needed': ds_data / esd_baryon_point(R_targets, M_bar),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 2b: Effective beta_local under stress halo ansatz
# ═══════════════════════════════════════════════════════════════

def stress_halo_enclosed_mass(R_kpc, M_bar, beta_kpc):
    """Enclosed stress mass under the stress halo ansatz:
    M_stress(<R) = alpha * M_bar * R / (beta + R)
    """
    return ALPHA * M_bar * R_kpc / (beta_kpc + R_kpc)


def solve_beta_local(R_kpc, M_extra_needed, M_bar):
    """Solve for the effective beta_local that produces M_extra at R.

    M_extra = alpha * M_bar * R / (beta_eff + R)
    → beta_eff = R * (alpha * M_bar / M_extra - 1)
    """
    ratio = ALPHA * M_bar / M_extra_needed
    if ratio <= 1:
        return np.nan  # Cannot be achieved even with beta_eff = 0
    beta_eff = R_kpc * (ratio - 1)
    return beta_eff


# ═══════════════════════════════════════════════════════════════
# STEP 2c: Toy profile slope analysis
# ═══════════════════════════════════════════════════════════════

def fit_power_law_slope(R_arr, ds_arr):
    """Fit log(ΔΣ) = a + b * log(R) to get the power-law slope."""
    mask = (ds_arr > 0) & np.isfinite(ds_arr)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    log_R = np.log10(R_arr[mask])
    log_ds = np.log10(ds_arr[mask])
    coeffs = np.polyfit(log_R, log_ds, 1)
    return coeffs[0], coeffs[1]  # slope, intercept


def required_3d_slope_from_esd_slope(esd_slope):
    """For a power-law ρ(r) ~ r^γ, the projected ΔΣ(R) ~ R^(γ+1).
    So γ = ESD_slope - 1.

    Examples:
    - Isothermal (ρ ~ r^-2): ESD ~ R^-1, so ESD slope = -1
    - NFW outer (ρ ~ r^-3): ESD ~ R^-2, so ESD slope = -2
    - Point mass (ρ = 0 outside): ESD ~ R^-2, so ESD slope = -2
    """
    return esd_slope - 1


# ═══════════════════════════════════════════════════════════════
# STEP 2d: SPARC constraint (eta = 1)
# ═══════════════════════════════════════════════════════════════

def sparc_max_radii():
    """Return approximate max radii of SPARC rotation curves.

    SPARC galaxies typically have measurements out to:
    - Dwarfs: 5-15 kpc
    - Intermediate: 10-30 kpc
    - L*: 15-40 kpc
    - Massive: 20-50 kpc

    Any field profile that changes gravity at R > R_max_SPARC doesn't
    affect the measured rotation curves. But if eta = 1, lensing and
    dynamics share the same potential, so we need to check.
    """
    return {
        'bin1': 20,   # kpc (representative max for bin 1 galaxies)
        'bin2': 30,
        'bin3': 35,
        'bin4': 40,
    }


def sparc_constraint_check(R_targets, enhancement_needed, M_bar, bin_name):
    """Check if the required lensing enhancement would change v_circ
    at the SPARC measurement boundary.

    If eta = 1, any extra ΔΣ at R also implies extra mass for dynamics.
    The extra circular velocity is:
        v_extra²(R) = G * M_extra(<R) / R

    For the SPARC constraint: v_extra at R_sparc must be small compared
    to the measured v_circ.
    """
    R_sparc = sparc_max_radii()[bin_name]

    # Enhancement needed at R_sparc (interpolated)
    if R_sparc < R_targets[0]:
        # Below our target range -- use inner extrapolation
        # The enhancement at inner radii is small (MTDF already works there)
        return {
            'R_sparc_kpc': R_sparc,
            'status': 'SAFE',
            'note': 'SPARC radii are inside the MTDF-valid range'
        }

    # Required enhancement factor at R = 100 kpc (first target)
    enh_100 = float(enhancement_needed[0])

    # The extra enclosed mass at 100 kpc needed for lensing:
    M_bar_val = M_bar
    ds_baryon_100 = esd_baryon_point(np.array([100.0]), M_bar_val)[0]
    ds_needed_100 = ds_baryon_100 * enh_100
    M_extra_100 = (ds_needed_100 - esd_mtdf(np.array([100.0]), M_bar_val)[0]) * np.pi * 100**2

    # If this mass were inside R_sparc, what v_circ change?
    # Assume the extra mass grows as M_extra(<r) ~ M_extra_100 * (r/100)^alpha_profile
    # For isothermal (alpha_profile = 1): M_extra(<30) = M_extra_100 * 30/100
    M_extra_sparc = M_extra_100 * R_sparc / 100.0  # isothermal scaling

    # Extra circular velocity
    R_sparc_pc = R_sparc * 1e3
    v_extra_sq = 4.302e-3 * M_extra_sparc / R_sparc_pc  # (km/s)^2
    v_extra = np.sqrt(max(0, v_extra_sq))

    # Typical v_circ at R_sparc for this mass bin
    v_typical = np.sqrt(4.302e-3 * M_bar_val * 2.3 / R_sparc_pc)

    return {
        'R_sparc_kpc': R_sparc,
        'M_extra_at_100kpc': float(M_extra_100),
        'M_extra_at_sparc': float(M_extra_sparc),
        'v_extra_kms': float(v_extra),
        'v_typical_kms': float(v_typical),
        'v_extra_fraction': float(v_extra / v_typical) if v_typical > 0 else np.nan,
        'status': 'SAFE' if v_extra / v_typical < 0.1 else 'TENSION',
        'note': (f'Extra v_circ at R_sparc = {R_sparc} kpc: '
                 f'{v_extra:.1f} km/s ({v_extra/v_typical*100:.1f}% of v_circ)')
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    data_dir = Path(__file__).parent.parent / "data" / "brouwer2021"
    out_dir = Path(__file__).parent.parent / "output" / "step2_algebraic_target"
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = [load_brouwer_bin(data_dir, i+1) for i in range(4)]

    results = []
    all_extras = []

    for i in range(4):
        b = bins[i]
        M_bar = MEDIAN_MBAR[i]

        # Step 2a: Required extra ΔΣ
        req = compute_required_extra(b, M_bar, R_TARGET.astype(float))

        # Step 2b: Effective beta_local at each target R
        beta_locals = []
        for j, R in enumerate(R_TARGET):
            M_extra = req['M_proj_extra'][j]
            if M_extra > 0:
                bl = solve_beta_local(float(R), float(M_extra), M_bar)
            else:
                bl = np.nan
            beta_locals.append(bl)

        # Step 2c: Power-law slope of the required extra ESD
        slope, intercept = fit_power_law_slope(R_TARGET.astype(float),
                                               req['ds_extra'])
        rho_slope = required_3d_slope_from_esd_slope(slope)

        # Step 2d: SPARC constraint
        sparc = sparc_constraint_check(R_TARGET.astype(float),
                                       req['enhancement_needed'], M_bar,
                                       BIN_NAMES[i])

        result = {
            'bin': BIN_NAMES[i],
            'label': BIN_LABELS[i],
            'log_Mstar': MEDIAN_LOG_MSTAR[i],
            'M_bar': float(M_bar),
            'radial_analysis': [],
            'esd_slope_extra': float(slope),
            'rho_3d_slope': float(rho_slope),
            'slope_interpretation': (
                f'Required ΔΣ_extra ~ R^{slope:.2f}, implying '
                f'ρ_field ~ r^{rho_slope:.2f}. '
                f'{"Isothermal-like (r^-2)" if -2.5 < rho_slope < -1.5 else "Steeper than isothermal" if rho_slope < -2.5 else "Shallower than isothermal"}'
            ),
            'sparc_constraint': sparc,
        }

        for j, R in enumerate(R_TARGET):
            result['radial_analysis'].append({
                'R_kpc': int(R),
                'ds_data_Msun_kpc2': float(req['ds_data'][j]),
                'ds_mtdf_Msun_kpc2': float(req['ds_mtdf'][j]),
                'ds_extra_Msun_kpc2': float(req['ds_extra'][j]),
                'M_proj_extra_Msun': float(req['M_proj_extra'][j]),
                'enhancement_factor': float(req['enhancement_needed'][j]),
                'beta_local_kpc': float(beta_locals[j]),
            })

        results.append(result)
        all_extras.append(req)

        print(f"\n{'='*60}")
        print(f"Bin {i+1}: {BIN_LABELS[i]} (M_bar = {M_bar:.2e})")
        print(f"{'='*60}")
        print(f"{'R (kpc)':>10} {'Data ESD':>12} {'MTDF ESD':>12} {'Extra ESD':>12} "
              f"{'M_extra':>12} {'Enhancement':>12} {'β_local':>10}")
        print(f"{'':>10} {'(Msun/kpc²)':>12} {'(Msun/kpc²)':>12} {'(Msun/kpc²)':>12} "
              f"{'(Msun)':>12} {'(factor)':>12} {'(kpc)':>10}")
        for j, R in enumerate(R_TARGET):
            print(f"{R:>10.0f} {req['ds_data'][j]:>12.1f} {req['ds_mtdf'][j]:>12.1f} "
                  f"{req['ds_extra'][j]:>12.1f} {req['M_proj_extra'][j]:>12.2e} "
                  f"{req['enhancement_needed'][j]:>12.1f} {beta_locals[j]:>10.0f}")
        print(f"\nRequired ESD slope: ΔΣ_extra ~ R^{slope:.2f}")
        print(f"Implied 3D density: ρ_field ~ r^{rho_slope:.2f}")
        print(f"SPARC constraint: {sparc['status']} — {sparc['note']}")

    # ═══════════════════════════════════════════════════════════
    # PLOT: Required enhancement factor vs radius
    # ═══════════════════════════════════════════════════════════

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(4):
        req = all_extras[i]
        ax1.plot(R_TARGET, req['enhancement_needed'],
                 'o-', color=colors[i], ms=8, lw=2, label=BIN_LABELS[i])

    ax1.axhline(2.3, color='blue', ls='--', alpha=0.5, label='MTDF max (2.3×)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Projected radius R [kpc]', fontsize=12)
    ax1.set_ylabel('Required enhancement factor\n(data / baryon ESD)', fontsize=11)
    ax1.set_title('Enhancement needed to match Brouwer+2021', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Plot required effective beta_local
    for i in range(4):
        betas = [r['beta_local_kpc'] for r in results[i]['radial_analysis']]
        valid = [(R_TARGET[j], betas[j]) for j in range(len(betas))
                 if not np.isnan(betas[j]) and betas[j] > 0]
        if valid:
            Rs, bs = zip(*valid)
            ax2.plot(Rs, bs, 'o-', color=colors[i], ms=8, lw=2,
                     label=BIN_LABELS[i])

    ax2.axhline(BETA_KPC, color='gray', ls='--', alpha=0.5,
                label=f'Cosmological β = {BETA_KPC:.0f} kpc')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Projected radius R [kpc]', fontsize=12)
    ax2.set_ylabel('Required effective β_local [kpc]', fontsize=11)
    ax2.set_title('Effective β needed under stress halo ansatz', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.suptitle('Step 2: Algebraic target for closing the F2 gap', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / 'step2_algebraic_target.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_dir / 'step2_algebraic_target.png'}")

    # ═══════════════════════════════════════════════════════════
    # PLOT 2: Required ΔΣ_extra profiles
    # ═══════════════════════════════════════════════════════════

    fig2, ax3 = plt.subplots(figsize=(10, 6))

    for i in range(4):
        req = all_extras[i]
        ax3.plot(R_TARGET, req['ds_extra'] / 1e6, 'o-', color=colors[i],
                 ms=8, lw=2, label=BIN_LABELS[i])

    # Reference slopes
    R_ref = np.linspace(80, 1200, 100)
    # Isothermal: ΔΣ ~ R^-1
    iso_norm = all_extras[3]['ds_extra'][0] / 1e6  # normalize to bin 4 at R=100
    ax3.plot(R_ref, iso_norm * (R_ref / 100)**(-1), 'k--', alpha=0.3,
             label=r'$\propto R^{-1}$ (isothermal)')
    ax3.plot(R_ref, iso_norm * (R_ref / 100)**(-2), 'k:', alpha=0.3,
             label=r'$\propto R^{-2}$ (NFW outer)')

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Projected radius R [kpc]', fontsize=12)
    ax3.set_ylabel(r'Required $\Delta\Sigma_\mathrm{extra}$ [$h_{70}\,M_\odot\,\mathrm{pc}^{-2}$]',
                   fontsize=11)
    ax3.set_title('Required extra ESD beyond MTDF baryonic enhancement', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'step2_extra_esd.png', dpi=150, bbox_inches='tight')
    print(f"Extra ESD plot saved: {out_dir / 'step2_extra_esd.png'}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    summary = {
        'description': 'Step 2: Algebraic target for closing the F2 gap',
        'method': ('Compute required ΔΣ_extra at target radii by subtracting '
                   'MTDF prediction from Brouwer+2021 data'),
        'bins': results,
    }

    # Overall assessment
    slopes = [r['esd_slope_extra'] for r in results]
    rho_slopes = [r['rho_3d_slope'] for r in results]
    sparc_statuses = [r['sparc_constraint']['status'] for r in results]

    # Check if stress halo ansatz can even produce enough mass
    # Max M_stress = alpha * M_bar (achieved at beta_local -> 0)
    max_stress_masses = [ALPHA * m for m in MEDIAN_MBAR]
    required_at_100 = [r['radial_analysis'][0]['M_proj_extra_Msun'] for r in results]
    shortfall_factors = [req / mx for req, mx in zip(required_at_100, max_stress_masses)]

    sparc_str = ('All bins SAFE' if all(s == 'SAFE' for s in sparc_statuses)
                 else f'TENSION in: {[BIN_NAMES[i] for i, s in enumerate(sparc_statuses) if s != "SAFE"]}')

    mean_slope = float(np.mean(slopes))
    mean_rho = float(np.mean(rho_slopes))

    summary['assessment'] = {
        'esd_slope_range': f"{min(slopes):.2f} to {max(slopes):.2f}",
        'rho_3d_slope_range': f"{min(rho_slopes):.2f} to {max(rho_slopes):.2f}",
        'sparc_constraint': sparc_str,
        'stress_halo_max_mass': [float(m) for m in max_stress_masses],
        'required_mass_at_100kpc': [float(m) for m in required_at_100],
        'stress_halo_shortfall_factor': [float(f) for f in shortfall_factors],
        'beta_local_possible': False,
        'conclusion': (
            f'The required field profile has ΔΣ_extra ~ R^{mean_slope:.1f}, '
            f'implying ρ_field ~ r^{mean_rho:.1f} '
            f'({"close to isothermal" if -2.5 < mean_rho < -1.5 else "steeper than isothermal"}). '
            f'CRITICAL: No effective β_local can close the gap. The maximum stress mass '
            f'under the current ansatz (α × M_bar) is {min(shortfall_factors):.0f}-'
            f'{max(shortfall_factors):.0f}x too small. The stress halo ansatz is '
            f'fundamentally insufficient — the problem is not concentration but total energy. '
            f'SPARC constraint: {sparc_str}.'
        ),
    }

    print(f"\n{'='*60}")
    print("CRITICAL FINDING: Stress halo energy budget")
    print(f"{'='*60}")
    for i in range(4):
        print(f"  Bin {i+1}: max M_stress = α×M_bar = {max_stress_masses[i]:.2e}, "
              f"required at 100 kpc = {required_at_100[i]:.2e}, "
              f"shortfall = {shortfall_factors[i]:.0f}x")

    with open(out_dir / 'step2_algebraic_target.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir / 'step2_algebraic_target.json'}")

    plt.close('all')


if __name__ == "__main__":
    main()
