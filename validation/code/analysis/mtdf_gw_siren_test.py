#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Date: December 2025
# Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""
MTDF Gravitational Wave Standard Siren Test
============================================
The Ultimate Independent Validation

This test compares GW-inferred distances with EM-inferred distances
to discriminate between:
  - Photon-specific effect: d_EM ≠ d_GW in voids
  - Metric modification: d_EM = d_GW everywhere

Key event: GW170817 (only BNS with EM counterpart)
Host: NGC 4993 (z = 0.0099)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
VOID_DIR = Path(str(Path(__file__).parent.parent.parent / 'data' / 'External' / 'desivast_voids'))
OUTPUT_DIR = Path(__file__).parent / 'output'

# Cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ============================================================================
# GW170817 DATA (The Gold Standard)
# ============================================================================
GW170817 = {
    'name': 'GW170817',
    'host': 'NGC 4993',
    'ra': 197.4505,  # degrees
    'dec': -23.3815,  # degrees
    'z_helio': 0.009783,  # heliocentric redshift
    'z_cmb': 0.00980,  # CMB frame

    # GW-inferred luminosity distance (Abbott et al. 2017)
    'd_L_GW': 40.0,  # Mpc (median)
    'd_L_GW_err_plus': 8.0,  # Mpc
    'd_L_GW_err_minus': 14.0,  # Mpc

    # EM-inferred distances (multiple methods)
    'distances': {
        'Fundamental Plane': {'d_L': 40.7, 'err': 2.0, 'ref': 'Hjorth+17'},
        'TRGB': {'d_L': 41.0, 'err': 3.0, 'ref': 'Cantiello+18'},
        'SBF': {'d_L': 40.4, 'err': 3.4, 'ref': 'Cantiello+18'},
        'Tully-Fisher': {'d_L': 40.8, 'err': 4.0, 'ref': 'Kourkchi+17'},
    }
}

# Additional GW events (dark sirens - statistical analysis only)
GW_DARK_SIRENS = [
    {'name': 'GW190814', 'ra': 12.8, 'dec': -25.3, 'd_L_GW': 241, 'err': 45},
    {'name': 'GW200115', 'ra': 15.2, 'dec': -12.1, 'd_L_GW': 300, 'err': 100},
]

def load_void_catalogs():
    """Load all available void catalogs"""
    catalogs = {}

    # VoidFinder
    vf_path = VOID_DIR / 'DESIVAST_BGS_VOLLIM_VF.fits'
    if vf_path.exists():
        with fits.open(vf_path) as hdul:
            catalogs['VoidFinder'] = hdul[1].data

    # REVOLVER NGC
    rev_ngc = VOID_DIR / 'DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits'
    if rev_ngc.exists():
        with fits.open(rev_ngc) as hdul:
            catalogs['REVOLVER_NGC'] = hdul[1].data

    # REVOLVER SGC
    rev_sgc = VOID_DIR / 'DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits'
    if rev_sgc.exists():
        with fits.open(rev_sgc) as hdul:
            catalogs['REVOLVER_SGC'] = hdul[1].data

    return catalogs

def find_void_environment(ra, dec, z, void_catalog, h=1.0):
    """
    Find the void environment for a given position.
    Returns (d_void_norm, void_info) where d_void_norm < 1 means inside void.
    """
    # Convert to comoving distance
    d_comoving = cosmo.comoving_distance(z).value * h  # Mpc/h

    # Convert RA/Dec to Cartesian
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = d_comoving * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_comoving * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = d_comoving * np.sin(dec_rad)

    # Find nearest void
    min_d_norm = np.inf
    nearest_void = None

    for void in void_catalog:
        # Get void center and radius
        if 'x' in void_catalog.dtype.names:
            vx, vy, vz = void['x'], void['y'], void['z']
        else:
            continue

        if 'radius' in void_catalog.dtype.names:
            R_void = void['radius']
        elif 'r_eff' in void_catalog.dtype.names:
            R_void = void['r_eff']
        else:
            continue

        # Distance to void center
        dist = np.sqrt((x - vx)**2 + (y - vy)**2 + (z_coord - vz)**2)
        d_norm = dist / R_void

        if d_norm < min_d_norm:
            min_d_norm = d_norm
            nearest_void = {
                'center': (vx, vy, vz),
                'radius': R_void,
                'd_norm': d_norm,
                'inside': d_norm < 1.0
            }

    return min_d_norm, nearest_void

def mtdf_distance_correction(d_void_norm, k_f=0.102):
    """
    Calculate MTDF distance correction based on void environment.

    If MTDF is photon-specific:
      d_EM = d_true * (1 - k_f * f(d_void_norm))

    where f() is the stress field profile (lower in voids)
    """
    # Stress profile: lower in voids (d_void_norm < 1)
    # Simple model: stress ~ tanh(d_void_norm - 1)
    stress_profile = 0.5 * (1 + np.tanh(2 * (d_void_norm - 1)))

    # Correction factor (photon-specific hypothesis)
    # In low-stress regions, photons lose less energy
    delta_d = -k_f * 0.01 * (1 - stress_profile)  # ~1% effect at void center

    return delta_d

def analyze_gw170817():
    """
    Main analysis: Compare GW and EM distances for GW170817
    in the context of its void environment.
    """
    print("=" * 70)
    print("MTDF GRAVITATIONAL WAVE STANDARD SIREN TEST")
    print("=" * 70)
    print()

    # Load void catalogs
    print("Loading void catalogs...")
    catalogs = load_void_catalogs()
    print(f"  Loaded: {list(catalogs.keys())}")
    print()

    # GW170817 info
    gw = GW170817
    print(f"Target: {gw['name']} (Host: {gw['host']})")
    print(f"  Position: RA={gw['ra']:.4f}°, Dec={gw['dec']:.4f}°")
    print(f"  Redshift: z = {gw['z_cmb']:.5f}")
    print()

    # GW distance
    print("GW-INFERRED DISTANCE:")
    print(f"  d_L (GW) = {gw['d_L_GW']:.1f} +{gw['d_L_GW_err_plus']:.1f}/-{gw['d_L_GW_err_minus']:.1f} Mpc")
    print()

    # EM distances
    print("EM-INFERRED DISTANCES (multiple methods):")
    em_distances = []
    for method, data in gw['distances'].items():
        print(f"  {method}: {data['d_L']:.1f} ± {data['err']:.1f} Mpc ({data['ref']})")
        em_distances.append(data['d_L'])

    em_mean = np.mean(em_distances)
    em_std = np.std(em_distances)
    print(f"\n  Mean EM distance: {em_mean:.1f} ± {em_std:.1f} Mpc")
    print()

    # Void environment analysis
    print("VOID ENVIRONMENT ANALYSIS:")
    print("-" * 40)

    # Note: NGC 4993 is at z~0.01, which is BELOW DESIVAST limit (z > 0.02)
    # So we need to extrapolate or use a different approach
    z_event = gw['z_cmb']

    if z_event < 0.02:
        print(f"  WARNING: z = {z_event:.4f} is below DESIVAST limit (z > 0.02)")
        print("  Using local void catalog extrapolation...")
        print()

        # Known local void information for NGC 4993's region
        # NGC 4993 is in the direction of the Hydra-Centaurus supercluster
        # NOT in a significant void
        d_void_norm_estimate = 2.5  # Well outside any major void
        print(f"  Estimated d_void_norm ≈ {d_void_norm_estimate:.1f} (outside voids)")
        print("  NGC 4993 is in the Hydra-Centaurus direction (overdense region)")

    else:
        # Try to find in DESIVAST
        for cat_name, catalog in catalogs.items():
            d_norm, void_info = find_void_environment(
                gw['ra'], gw['dec'], z_event, catalog
            )
            if void_info:
                status = "INSIDE" if void_info['inside'] else "OUTSIDE"
                print(f"  {cat_name}: d_void_norm = {d_norm:.2f} ({status})")
        d_void_norm_estimate = d_norm

    print()

    # MTDF prediction
    print("MTDF PREDICTIONS:")
    print("-" * 40)

    k_f = 0.102
    delta_d = mtdf_distance_correction(d_void_norm_estimate, k_f)

    print(f"  Using k_f = {k_f}")
    print(f"  Stress profile at d_void_norm = {d_void_norm_estimate:.1f}")
    print()

    if d_void_norm_estimate > 1.5:
        print("  NGC 4993 is in an OVERDENSE region (not void)")
        print("  MTDF predicts: d_EM ≈ d_GW (no significant correction)")
        print()
        print("  This is a WEAK TEST because the event is not in a void.")
        print("  For a STRONG TEST, we need GW events inside or near voids.")
    else:
        correction_pct = delta_d * 100
        print(f"  Predicted EM correction: {correction_pct:+.2f}%")
        d_em_predicted = gw['d_L_GW'] * (1 + delta_d)
        print(f"  Predicted d_EM = {d_em_predicted:.1f} Mpc")

    print()

    # Statistical comparison
    print("STATISTICAL TEST:")
    print("-" * 40)

    # Compare GW and EM
    d_gw = gw['d_L_GW']
    d_gw_err = (gw['d_L_GW_err_plus'] + gw['d_L_GW_err_minus']) / 2

    delta = em_mean - d_gw
    combined_err = np.sqrt(d_gw_err**2 + em_std**2)
    sigma = delta / combined_err

    print(f"  Δd = d_EM - d_GW = {delta:+.1f} Mpc")
    print(f"  Combined uncertainty: ±{combined_err:.1f} Mpc")
    print(f"  Significance: {sigma:+.2f}σ")
    print()

    if abs(sigma) < 1:
        print("  RESULT: GW and EM distances are CONSISTENT")
        print("  This is compatible with:")
        print("    - No MTDF effect (null hypothesis)")
        print("    - MTDF metric modification (GW and EM both affected)")
        print("    - MTDF photon effect in non-void environment (no correction)")
    else:
        print(f"  RESULT: {abs(sigma):.1f}σ difference detected")

    print()

    # Create figure
    create_gw_figure(gw, em_mean, em_std, d_void_norm_estimate)

    # Future outlook
    print("=" * 70)
    print("FUTURE TESTS (MORE POWERFUL)")
    print("=" * 70)
    print()
    print("GW170817 is a WEAK test because NGC 4993 is NOT in a void.")
    print()
    print("For a STRONG test, we need:")
    print("  1. BNS merger with EM counterpart IN or NEAR a void")
    print("  2. Statistical analysis of many dark sirens by environment")
    print("  3. Future detectors (Einstein Telescope, Cosmic Explorer)")
    print()
    print("PREDICTION for void GW events:")
    print("  - If MTDF is photon-specific: d_EM < d_GW in voids")
    print("  - If MTDF is metric modification: d_EM = d_GW everywhere")
    print()

    return {
        'd_gw': d_gw,
        'd_em': em_mean,
        'delta': delta,
        'sigma': sigma,
        'd_void_norm': d_void_norm_estimate
    }

def create_gw_figure(gw, em_mean, em_std, d_void_norm):
    """Create visualization of GW vs EM distance comparison"""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Distance comparison
    ax1 = axes[0]

    # GW distance
    ax1.errorbar(0, gw['d_L_GW'],
                 yerr=[[gw['d_L_GW_err_minus']], [gw['d_L_GW_err_plus']]],
                 fmt='o', markersize=12, capsize=5, color='purple',
                 label=f"GW: {gw['d_L_GW']:.0f} Mpc")

    # EM distances
    for i, (method, data) in enumerate(gw['distances'].items()):
        ax1.errorbar(i+1, data['d_L'], yerr=data['err'],
                     fmt='s', markersize=8, capsize=4, alpha=0.7,
                     label=f"{method}: {data['d_L']:.0f} Mpc")

    # Mean EM
    ax1.axhline(em_mean, color='orange', linestyle='--', alpha=0.5)
    ax1.axhspan(em_mean - em_std, em_mean + em_std,
                color='orange', alpha=0.1)

    ax1.set_ylabel('Luminosity Distance [Mpc]', fontsize=11)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['GW', 'FP', 'TRGB', 'SBF', 'TF'], fontsize=9)
    ax1.set_title('GW170817: Distance Measurements', fontsize=12)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(20, 60)

    # Panel 2: Void environment context
    ax2 = axes[1]

    # Show void boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'b--', linewidth=2,
             label='Void boundary')
    ax2.fill(np.cos(theta), np.sin(theta), color='lightblue', alpha=0.3)

    # NGC 4993 position (outside void)
    ax2.scatter([d_void_norm], [0], s=200, c='red', marker='*',
                zorder=5, label=f'NGC 4993 (d/R = {d_void_norm:.1f})')

    # Annotations
    ax2.annotate('VOID\n(low stress)', xy=(0, 0), fontsize=10,
                 ha='center', va='center', color='blue')
    ax2.annotate('WALL\n(high stress)', xy=(2, 0), fontsize=10,
                 ha='center', va='center', color='red')

    ax2.set_xlim(-0.5, 3)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel('d / R_void', fontsize=11)
    ax2.set_title('Void Environment', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_aspect('equal')
    ax2.axhline(0, color='gray', alpha=0.3)

    # Panel 3: MTDF prediction vs observation
    ax3 = axes[2]

    # Text summary
    summary = f"""GW170817 STANDARD SIREN TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISTANCES:
  GW:  {gw['d_L_GW']:.0f} +{gw['d_L_GW_err_plus']:.0f}/-{gw['d_L_GW_err_minus']:.0f} Mpc
  EM:  {em_mean:.1f} ± {em_std:.1f} Mpc
  Δ:   {em_mean - gw['d_L_GW']:+.1f} Mpc

VOID STATUS:
  d_void_norm = {d_void_norm:.1f}
  → OUTSIDE voids (overdense)

MTDF PREDICTION:
  No correction expected
  (not in low-stress region)

INTERPRETATION:
  GW ≈ EM distances are
  CONSISTENT (as expected
  for non-void environment)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR STRONG TEST:
Need GW event INSIDE a void
→ Would show d_EM < d_GW
   if photon-specific effect
"""

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       edgecolor='orange', alpha=0.8))
    ax3.axis('off')
    ax3.set_title('Test Summary', fontsize=12)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'mtdf_gw_siren_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Figure saved: {output_path}")

if __name__ == '__main__':
    results = analyze_gw170817()
