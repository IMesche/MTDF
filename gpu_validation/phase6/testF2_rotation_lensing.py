#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 6 Test F2: Rotation Curve + Lensing Consistency
========================================================
Pre-registered test definition for cross-probe consistency:
does the MTDF mass profile from rotation curves predict the correct
galaxy-galaxy lensing (GGL) signal?

Since η = 1 (confirmed C5b), lensing and dynamics probe the same
effective potential.  MTDF predicts v_c²(r) = GM_bar/r × [1 + α/(1+r/β)]
using only baryonic mass.  This test asks whether the resulting ESD
profile matches observations that ΛCDM explains with dark matter halos.

Deliverables (test definition, not full GGL execution):
  1. SPARC mass profile computation — M_bar(<R) per stellar mass bin
  2. MTDF ΔΣ prediction — ESD under Assumption A
  3. ΛCDM ΔΣ prediction — baryonic + NFW halo (via SHMR)
  4. Comparison plot with pre-registered success criteria

Full GGL execution (stacking KiDS-1000 shear around GAMA lenses)
is deferred to MTDF_07.

Entry point:
  python mtdf_validation/phase6/testF2_rotation_lensing.py
"""

import sys
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "validation" / "data"
DEFAULT_OUTPUT = (PROJECT_ROOT / "validation" / "output"
                  / "phase6" / "testF2_rotation_lensing")

# ── MTDF Parameters ──────────────────────────────────────────────
ALPHA = 1.30                # Stress-matter coupling
BETA_M = 7.0e23             # Coherence length in metres
BETA_KPC = BETA_M / 3.0857e19  # Convert to kpc: ~22,685 kpc

# ── Physical Constants ───────────────────────────────────────────
G_KPC = 4.302e-6            # G in kpc (km/s)² M_sun^{-1}
H0 = 70.0                   # km/s/Mpc
RHO_CRIT = 136.3            # Critical density: M_sun / kpc³ (for H0=70)

# ── Stellar Mass Bins (log10 M_sun) ─────────────────────────────
MASS_BIN_EDGES = [7.0, 9.0, 10.0, 10.5, 12.0]
MASS_BIN_LABELS = ["dwarf", "intermediate", "L-star", "massive"]

# ── Radial grid for ESD predictions (kpc) ────────────────────────
R_ESD = np.logspace(np.log10(10), np.log10(2000), 50)  # 10 kpc – 2 Mpc


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════
# 1. SPARC DATA
# ═══════════════════════════════════════════════════════════════════

def load_sparc():
    """Load SPARC galaxies and compute baryonic mass profiles."""
    path = DATA_DIR / "sparc_clean.json"
    with open(path) as f:
        raw = json.load(f)

    galaxies = []
    for gal_name, gal in raw['galaxies'].items():
        pts = gal['points']
        if len(pts) < 3:
            continue

        r_kpc = np.array([p['r'] for p in pts])
        v_obs = np.array([p['v_obs'] for p in pts])
        v_bar = np.array([p['v_bar'] for p in pts])
        sigma_v = np.array([p.get('sigma_v', 5.0) for p in pts])

        # Enclosed baryonic mass: M(<r) = v_bar²(r) × r / G
        M_bar = v_bar**2 * r_kpc / G_KPC  # M_sun

        # Total baryonic mass from outermost point
        M_bar_total = float(M_bar[-1])
        log_M_bar = np.log10(max(M_bar_total, 1.0))

        galaxies.append({
            'name': gal['name'],
            'morphology': gal.get('morphology', ''),
            'r_kpc': r_kpc,
            'v_obs': v_obs,
            'v_bar': v_bar,
            'sigma_v': sigma_v,
            'M_bar_enclosed': M_bar,
            'M_bar_total': M_bar_total,
            'log_M_bar': log_M_bar,
        })

    print(f"  Loaded {len(galaxies)} SPARC galaxies")
    print(f"  log M_bar range: [{min(g['log_M_bar'] for g in galaxies):.1f}, "
          f"{max(g['log_M_bar'] for g in galaxies):.1f}]")

    return galaxies


def bin_galaxies(galaxies):
    """Bin galaxies by baryonic mass."""
    bins = {}
    for i, label in enumerate(MASS_BIN_LABELS):
        lo, hi = MASS_BIN_EDGES[i], MASS_BIN_EDGES[i + 1]
        members = [g for g in galaxies if lo <= g['log_M_bar'] < hi]
        if not members:
            continue
        bins[label] = {
            'log_M_range': [lo, hi],
            'n_galaxies': len(members),
            'galaxies': members,
            'mean_log_M': float(np.mean([g['log_M_bar'] for g in members])),
            'mean_M_bar': float(np.mean([g['M_bar_total'] for g in members])),
        }
        print(f"  Bin '{label}' [{lo:.1f}, {hi:.1f}): "
              f"{len(members)} galaxies, "
              f"<log M> = {bins[label]['mean_log_M']:.2f}")
    return bins


def average_mass_profile(mass_bin, r_grid):
    """Average M_bar(<R) over galaxies in a bin, interpolated to r_grid."""
    profiles = []
    for g in mass_bin['galaxies']:
        r = g['r_kpc']
        M = g['M_bar_enclosed']
        # Interpolate; extrapolate flat beyond last point
        M_interp = np.interp(r_grid, r, M, right=M[-1])
        # Below first data point: approximate M ∝ r
        below = r_grid < r[0]
        if np.any(below):
            M_interp[below] = M[0] * (r_grid[below] / r[0])
        profiles.append(M_interp)

    profiles = np.array(profiles)
    mean_profile = np.mean(profiles, axis=0)
    std_profile = np.std(profiles, axis=0) / np.sqrt(len(profiles))

    return mean_profile, std_profile


# ═══════════════════════════════════════════════════════════════════
# 2. NFW PROFILE (Wright & Brainerd 2000)
# ═══════════════════════════════════════════════════════════════════

def nfw_sigma(x):
    """NFW surface mass density Σ(x) / (2 ρ_s r_s), x = R/r_s."""
    result = np.zeros_like(x, dtype=float)
    # x < 1
    lo = (x < 0.999)
    if np.any(lo):
        xl = x[lo]
        result[lo] = 1.0 / (xl**2 - 1) * (
            1.0 - np.arctanh(np.sqrt(1 - xl**2)) / np.sqrt(1 - xl**2))
    # x = 1
    eq = (x >= 0.999) & (x <= 1.001)
    result[eq] = 1.0 / 3.0
    # x > 1
    hi = (x > 1.001)
    if np.any(hi):
        xh = x[hi]
        result[hi] = 1.0 / (xh**2 - 1) * (
            1.0 - np.arctan(np.sqrt(xh**2 - 1)) / np.sqrt(xh**2 - 1))
    return result


def nfw_sigma_mean(x):
    """NFW mean surface density Σ̄(<x) / (4 ρ_s r_s / x²), x = R/r_s."""
    result = np.zeros_like(x, dtype=float)
    lo = (x < 0.999)
    if np.any(lo):
        xl = x[lo]
        result[lo] = (np.arctanh(np.sqrt(1 - xl**2)) / np.sqrt(1 - xl**2)
                       + np.log(xl / 2))
    eq = (x >= 0.999) & (x <= 1.001)
    result[eq] = 1.0 + np.log(0.5)
    hi = (x > 1.001)
    if np.any(hi):
        xh = x[hi]
        result[hi] = (np.arctan(np.sqrt(xh**2 - 1)) / np.sqrt(xh**2 - 1)
                       + np.log(xh / 2))
    return result


def nfw_esd(R_kpc, M200, c200):
    """Compute NFW Excess Surface Density ΔΣ(R) in M_sun/kpc².

    Args:
        R_kpc: projected radii in kpc
        M200: halo mass in M_sun
        c200: concentration parameter

    Returns:
        delta_sigma: ΔΣ(R) in M_sun/kpc²
    """
    r200 = (3 * M200 / (4 * np.pi * 200 * RHO_CRIT))**(1.0 / 3.0)
    r_s = r200 / c200
    rho_s = M200 / (4 * np.pi * r_s**3 *
                     (np.log(1 + c200) - c200 / (1 + c200)))

    x = R_kpc / r_s
    x = np.clip(x, 1e-6, None)

    sigma = 2 * rho_s * r_s * nfw_sigma(x)
    sigma_mean = 4 * rho_s * r_s * nfw_sigma_mean(x) / x**2

    return sigma_mean - sigma


# ═══════════════════════════════════════════════════════════════════
# 3. STELLAR-MASS-HALO-MASS RELATION (Moster+2013 at z=0)
# ═══════════════════════════════════════════════════════════════════

def moster2013_mstar(M_halo):
    """M_star given M_halo using Moster+2013 z=0 parameters."""
    N0 = 0.0351
    M1 = 10**11.59  # M_sun
    beta = 1.376
    gamma = 0.608
    f = 2 * N0 * ((M_halo / M1)**(-beta) + (M_halo / M1)**gamma)**(-1)
    return M_halo * f


def halo_mass_from_stellar(M_star):
    """Invert Moster+2013 to get M_halo from M_star."""
    def residual(log_Mh):
        return np.log10(moster2013_mstar(10**log_Mh)) - np.log10(M_star)
    try:
        log_Mh = brentq(residual, 9.0, 15.0)
        return 10**log_Mh
    except ValueError:
        return None


def duffy2008_concentration(M_halo, z=0.0):
    """NFW concentration from Duffy+2008 (full sample)."""
    A, B, C = 5.71, -0.084, -0.47
    M_pivot = 2e12  # M_sun/h (using h=0.7 convention)
    return A * (M_halo / M_pivot)**B * (1 + z)**C


# ═══════════════════════════════════════════════════════════════════
# 4. ESD PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

def esd_baryon_point_mass(R_kpc, M_bar):
    """ESD for a point-mass baryon at the centre: ΔΣ = M/(πR²).

    Valid for R >> stellar half-mass radius (~3-5 kpc for L* galaxies).
    At R > 10 kpc this is an excellent approximation.
    """
    return M_bar / (np.pi * R_kpc**2)


def esd_mtdf(R_kpc, M_bar):
    """MTDF ESD under Assumption A.

    ΔΣ_MTDF(R) = ΔΣ_baryon(R) × [1 + α/(1 + R/β)]

    Since β ~ 22,685 kpc and R_max ~ 2000 kpc, α/(1+R/β) ≈ α across
    the entire GGL range. The MTDF enhancement is effectively a constant
    factor of [1 + α] = 2.30.
    """
    enhancement = 1.0 + ALPHA / (1.0 + R_kpc / BETA_KPC)
    return esd_baryon_point_mass(R_kpc, M_bar) * enhancement


def esd_lcdm(R_kpc, M_bar, M_halo, c):
    """ΛCDM ESD: baryonic point mass + NFW halo."""
    return esd_baryon_point_mass(R_kpc, M_bar) + nfw_esd(R_kpc, M_halo, c)


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_bin(label, bin_data, r_grid):
    """Compute ESD predictions for a stellar mass bin."""
    M_bar = bin_data['mean_M_bar']
    log_M = bin_data['mean_log_M']

    # Halo mass from SHMR
    M_halo = halo_mass_from_stellar(M_bar)
    if M_halo is None:
        M_halo = M_bar * 30  # Fallback
    c = duffy2008_concentration(M_halo)

    # Average mass profile from SPARC
    M_profile, M_profile_err = average_mass_profile(bin_data, r_grid)

    # ESD predictions
    ds_baryon = esd_baryon_point_mass(r_grid, M_bar)
    ds_mtdf = esd_mtdf(r_grid, M_bar)
    ds_nfw = nfw_esd(r_grid, M_halo, c)
    ds_lcdm = ds_baryon + ds_nfw

    # MTDF enhancement factor
    enhancement = 1.0 + ALPHA / (1.0 + r_grid / BETA_KPC)

    # Key discriminant: ratio at R > 100 kpc
    mask_100 = r_grid > 100
    if np.any(mask_100):
        ratio_100 = np.mean(ds_mtdf[mask_100] / ds_lcdm[mask_100])
    else:
        ratio_100 = np.nan

    result = {
        'label': label,
        'log_M_range': bin_data['log_M_range'],
        'n_galaxies': bin_data['n_galaxies'],
        'mean_log_M_bar': log_M,
        'mean_M_bar': float(M_bar),
        'M_halo_shmr': float(M_halo),
        'c_duffy': float(c),
        'r_kpc': r_grid.tolist(),
        'M_bar_profile': M_profile.tolist(),
        'M_bar_profile_err': M_profile_err.tolist(),
        'esd_baryon': ds_baryon.tolist(),
        'esd_mtdf': ds_mtdf.tolist(),
        'esd_nfw': ds_nfw.tolist(),
        'esd_lcdm': ds_lcdm.tolist(),
        'enhancement_factor': enhancement.tolist(),
        'mtdf_lcdm_ratio_R100': float(ratio_100),
    }

    print(f"  {label}: <M_bar>={M_bar:.2e}, M_halo={M_halo:.2e}, "
          f"c={c:.1f}, MTDF/LCDM(R>100)={ratio_100:.3f}")

    return result


# ═══════════════════════════════════════════════════════════════════
# 6. PLOTTING
# ═══════════════════════════════════════════════════════════════════

def make_plots(bin_results, r_grid, outdir):
    """Generate ESD prediction plot and SPARC mass profile plot."""

    # ── Plot 1: ESD predictions ──────────────────────────────────
    n_bins = len(bin_results)
    fig, axes = plt.subplots(1, n_bins, figsize=(4.5 * n_bins, 5),
                              squeeze=False)

    for i, (label, res) in enumerate(bin_results.items()):
        ax = axes[0, i]
        r = np.array(res['r_kpc'])

        ax.loglog(r, res['esd_mtdf'], 'C0-', lw=2.5,
                  label='MTDF (Assumption A)')
        ax.loglog(r, res['esd_lcdm'], 'C3--', lw=2.5,
                  label=r'$\Lambda$CDM (baryon + NFW)')
        ax.loglog(r, res['esd_baryon'], 'k:', lw=1, alpha=0.5,
                  label='Baryon only')
        ax.loglog(r, res['esd_nfw'], 'C3:', lw=1, alpha=0.5,
                  label='NFW only')

        ax.set_xlabel('R  [kpc]', fontsize=11)
        if i == 0:
            ax.set_ylabel(r'$\Delta\Sigma(R)$  [$M_\odot$/kpc$^2$]',
                          fontsize=11)
        ax.set_title(f"{label}\n"
                     f"N={res['n_galaxies']}, "
                     f"log M={res['mean_log_M_bar']:.1f}",
                     fontsize=10)
        ax.legend(fontsize=7, loc='lower left')
        ax.set_xlim(10, 2000)
        ax.set_ylim(1e-2, 1e6)

    fig.suptitle('Test F2: MTDF vs ΛCDM ESD Predictions from SPARC',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = outdir / "testF2_esd_prediction.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot: {path}")

    # ── Plot 2: SPARC mass profiles ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['C0', 'C1', 'C2', 'C3']
    for i, (label, res) in enumerate(bin_results.items()):
        r = np.array(res['r_kpc'])
        M = np.array(res['M_bar_profile'])
        M_err = np.array(res['M_bar_profile_err'])
        ax.loglog(r, M, f'{colors[i]}-', lw=2, label=f"{label} (N={res['n_galaxies']})")
        ax.fill_between(r, M - M_err, M + M_err, alpha=0.2, color=colors[i])

    ax.set_xlabel('r  [kpc]', fontsize=12)
    ax.set_ylabel(r'$M_{\rm bar}(<r)$  [$M_\odot$]', fontsize=12)
    ax.set_title('SPARC Baryonic Mass Profiles by Stellar Mass Bin',
                 fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = outdir / "testF2_sparc_mass_profiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


# ═══════════════════════════════════════════════════════════════════
# 7. OUTPUT
# ═══════════════════════════════════════════════════════════════════

def build_summary(bin_results):
    return {
        "test": "Phase 6 Test F2: Rotation curve + lensing consistency",
        "type": "Test definition (analytical predictions)",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mtdf_parameters": {
            "alpha": ALPHA,
            "beta_kpc": BETA_KPC,
            "beta_m": BETA_M,
        },
        "assumptions": {
            "A": ("ΔΣ_MTDF(R) = ΔΣ_baryon(R) × [1 + α/(1+R/β)]. "
                  "Lensing sees the same enhanced potential as dynamics "
                  "(η = 1 from C5b)."),
            "B": ("ΔΣ_MTDF(R) = ΔΣ_baryon(R). Force-law modification "
                  "is purely kinematic. Inconsistent with η = 1."),
        },
        "shmr": "Moster+2013 (z=0)",
        "concentration": "Duffy+2008 (full sample)",
        "r_grid_kpc": R_ESD.tolist(),
        "bins": bin_results,
        "key_insight": (
            "Since β ~ 22,685 kpc >> R_max ~ 2 Mpc, the MTDF enhancement "
            "is effectively constant: [1 + α] = 2.30 across the entire "
            "GGL range. MTDF predicts the ESD should be 2.3× the baryonic "
            "ESD at all radii."),
        "success_criteria": {
            "pass": ("ΔΣ(R) from enhanced baryonic mass is within 2σ of "
                     "observed GGL signal across all R bins"),
            "fail": ("ΔΣ(R > 100 kpc) requires additional NFW-scale mass "
                     "beyond MTDF's force-law enhancement"),
            "inconclusive": ("S/N < 3 in discriminating R bins (R > 100 kpc)"),
        },
    }


def write_readme(summary, bin_results, outdir):
    rows = []
    for label, res in bin_results.items():
        rows.append(
            f"| {label} | [{res['log_M_range'][0]:.0f}, "
            f"{res['log_M_range'][1]:.0f}) | {res['n_galaxies']} | "
            f"{res['mean_M_bar']:.2e} | {res['M_halo_shmr']:.2e} | "
            f"{res['c_duffy']:.1f} | {res['mtdf_lcdm_ratio_R100']:.3f} |")
    table = "\n".join(rows)

    readme = f"""# Phase 6 Test F2: Rotation Curve + Lensing Consistency

## Goal

Test whether the MTDF mass profile from rotation curves predicts the
correct galaxy-galaxy lensing (GGL) signal.  Since η = 1 (confirmed C5b),
lensing and dynamics probe the same effective potential.

## Physics

MTDF predicts:

    v_c²(r) = G M_bar / r × [1 + α/(1 + r/β)]

where α = 1.30, β = 22,685 kpc.  At GGL-relevant scales (R = 10–2000 kpc),
R << β, so the enhancement is approximately constant:

    [1 + α/(1 + R/β)] ≈ [1 + α] = 2.30

MTDF predicts the ESD should be **2.3× the baryonic-only ESD** at all
radii.  ΛCDM explains the same signal with an NFW dark matter halo.

## Assumptions (Pre-Registered)

**Assumption A (primary prediction):**
ΔΣ_MTDF(R) = ΔΣ_baryon(R) × [1 + α/(1 + R/β)]

The MTDF force-law enhancement applies equally to lensing and dynamics
(since η = 1 at perturbation level, C5b).  This is a direct extension
of the rotation curve formula to the lensing kernel.

**Assumption B (alternative, for completeness):**
ΔΣ_MTDF(R) = ΔΣ_baryon(R)

Force-law modification is purely kinematic (affects orbits, not photons).
This would be inconsistent with η = 1 and is NOT the primary prediction.

**The distinction between A and B is a testable prediction.**

## Method

1. Load 175 SPARC galaxies (Lelli+2016)
2. Compute M_bar(<r) = v_bar²(r) × r / G for each galaxy
3. Bin by total baryonic mass (proxy for stellar mass)
4. Average M_bar(<R) per bin on common radial grid
5. Compute ESD predictions:
   - ΔΣ_baryon(R) = M_bar / (πR²) — point-mass approximation (valid R >> R_d)
   - ΔΣ_MTDF(R) = ΔΣ_baryon(R) × [1 + α/(1+R/β)] — Assumption A
   - ΔΣ_ΛCDM(R) = ΔΣ_baryon(R) + ΔΣ_NFW(R) — using Moster+2013 SHMR + Duffy+2008 c(M)

## SPARC Mass Bins

| Bin | log M range | N_gal | <M_bar> | M_halo (SHMR) | c (Duffy) | MTDF/ΛCDM (R>100 kpc) |
|-----|-------------|-------|---------|---------------|-----------|----------------------|
{table}

## Published Data Overlay Caveat

Any published GGL ESD profiles (e.g., Brouwer+2021 KiDS×GAMA) overlaid
on the comparison plot are **illustrative context only**, not a direct
comparison.  A proper comparison requires exactly matching the lens
selection, redshift weights, and radial bin definitions used in the
published analysis.  This is deferred to MTDF_07.

## Success Criteria (Pre-Registered)

- **MTDF passes if:** ΔΣ(R) from enhanced baryonic mass is within 2σ of
  observed GGL signal across all R bins
- **MTDF fails if:** ΔΣ(R > 100 kpc) requires additional NFW-scale mass
  beyond what MTDF's force-law enhancement provides
- **Inconclusive if:** S/N < 3 in discriminating R bins (R > 100 kpc)

## Relation to V74 Dashboard

P1 (rotation curve scatter, z-score = 0.888) and P1B (RAR scatter,
z-score = -0.62) in the V74 dashboard confirm MTDF already fits SPARC
rotation curves at sub-1σ.  F2 builds on this by asking: does the same
mass profile that explains rotation curves also predict the correct
lensing signal?  The dashboard pillars are consistency checks; F2 is a
cross-probe prediction.

## Key Physical Insight

At R > 100 kpc, the MTDF and ΛCDM predictions diverge:
- **ΛCDM:** ΔΣ dominated by NFW halo (ΔΣ_NFW >> ΔΣ_baryon)
- **MTDF:** ΔΣ = 2.3 × ΔΣ_baryon (no additional halo mass)

For massive galaxies (M_bar ~ 10^{{10.5}}), the NFW halo contributes
~10× more ESD than baryons at R = 500 kpc.  MTDF's 2.3× enhancement
cannot match this.  Either MTDF's non-perturbative lensing enhancement
is stronger than the perturbative η = 1 result, or MTDF under-predicts
GGL at large radii for massive galaxies.

This is a genuine, falsifiable tension point — exactly what a pre-
registered test should identify.

## Files

| File | Description |
|------|-------------|
| `testF2_rotation_lensing.json` | Mass profiles, ESD predictions per bin |
| `testF2_esd_prediction.png` | MTDF vs ΛCDM ESD predictions |
| `testF2_sparc_mass_profiles.png` | Binned SPARC mass profiles |
| `README.md` | This file (formal test definition) |
| `manifest.json` | SHA256 hashes |

## How to Reproduce

```bash
python mtdf_validation/phase6/testF2_rotation_lensing.py
```
"""
    (outdir / "README.md").write_text(readme)
    print(f"  README: {outdir / 'README.md'}")


def write_manifest(outdir):
    files = sorted(
        f.name for f in outdir.iterdir()
        if f.is_file() and f.name != 'manifest.json')
    hashes = {f: sha256_file(outdir / f) for f in files}
    m = {"generated": datetime.now().strftime("%Y-%m-%d"), "sha256": hashes}
    (outdir / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(f"  Manifest: {len(hashes)} files hashed")


# ═══════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    outdir = Path(DEFAULT_OUTPUT)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 6 Test F2: Rotation Curve + Lensing Consistency")
    print("=" * 60)

    galaxies = load_sparc()
    mass_bins = bin_galaxies(galaxies)

    print("\n--- ESD predictions ---")
    bin_results = {}
    for label in MASS_BIN_LABELS:
        if label not in mass_bins:
            continue
        bin_results[label] = analyze_bin(label, mass_bins[label], R_ESD)

    summary = build_summary(bin_results)

    jp = outdir / "testF2_rotation_lensing.json"
    jp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n  JSON: {jp}")

    make_plots(bin_results, R_ESD, outdir)
    write_readme(summary, bin_results, outdir)
    write_manifest(outdir)

    print(f"\n{'=' * 60}")
    print("Test F2: Pre-registered test definition complete")
    print(f"  Bins: {list(bin_results.keys())}")
    for label, res in bin_results.items():
        print(f"  {label}: MTDF/LCDM ratio (R>100 kpc) = "
              f"{res['mtdf_lcdm_ratio_R100']:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
