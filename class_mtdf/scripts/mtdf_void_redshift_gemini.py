#!/usr/bin/env python3
"""
MTDF Void Redshift Test - Gemini's "Court Proof" Version
=========================================================
Adapted from Gemini's script for GAMA G3C + DESIVAST data.

Key improvements over v2:
1. Binary Inside/Outside comparison (cleaner signal)
2. 1000-iteration label shuffle null test
3. Malmquist bias check with magnitude cut
4. Proper Hubble residual definition with z-binning
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
GAMA_CATALOG = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'gama' / 'G3CGalv10.fits')
VOID_DIR = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'desivast_voids')

C_KMS = 299792.458  # Speed of light in km/s

# =============================================================================
# DATA LOADING
# =============================================================================
def load_gama_data():
    """Load GAMA G3C galaxy catalog."""
    print("Loading GAMA G3C catalog...")
    with fits.open(GAMA_CATALOG) as hdul:
        data = hdul[1].data

    # Extract relevant columns
    gals = {
        'ra': data['RA'],
        'dec': data['DEC'],
        'z': data['Z'],
        'mag_r': data['RPETRO'] if 'RPETRO' in data.dtype.names else data['RABSMAG'] if 'RABSMAG' in data.dtype.names else np.zeros(len(data)),
        'group_id': data['GroupID'] if 'GroupID' in data.dtype.names else np.zeros(len(data)),
    }

    # Convert to arrays
    for key in gals:
        gals[key] = np.array(gals[key])

    # Quality cuts
    mask = (gals['z'] > 0.01) & (gals['z'] < 0.5) & np.isfinite(gals['ra']) & np.isfinite(gals['dec'])
    for key in gals:
        gals[key] = gals[key][mask]

    print(f"  Loaded {len(gals['ra'])} galaxies after quality cuts")
    return gals

def load_void_data():
    """Load DESIVAST void catalogs."""
    print("Loading DESIVAST void catalogs...")

    void_files = [
        f'{VOID_DIR}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits',
        f'{VOID_DIR}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits',
    ]

    all_ra, all_dec, all_z, all_radius = [], [], [], []

    for vf in void_files:
        try:
            with fits.open(vf) as hdul:
                data = hdul[1].data
                all_ra.extend(data['RA'])
                all_dec.extend(data['DEC'])
                all_z.extend(data['REDSHIFT'])
                all_radius.extend(data['RADIUS'])
        except Exception as e:
            print(f"  Warning: Could not load {vf}: {e}")

    voids = {
        'ra': np.array(all_ra),
        'dec': np.array(all_dec),
        'z': np.array(all_z),
        'radius_mpc': np.array(all_radius),
    }

    print(f"  Loaded {len(voids['ra'])} voids")
    return voids

# =============================================================================
# PECULIAR VELOCITY CALCULATION
# =============================================================================
def calculate_redshift_residual(gals):
    """
    Calculate line-of-sight redshift residual (NOT true peculiar velocity).

    Method: Bin by redshift, compute residual from bin median.
    cz_resid = c × (z - z_median) / (1 + z_median)

    IMPORTANT: This is a redshift-space proxy, NOT a peculiar velocity.
    True peculiar velocity requires independent distance indicators.
    This measures environment-dependent redshift residuals in z-space.
    """
    print("Computing redshift residuals (NOT peculiar velocity)...")

    # Create redshift bins (20 quantile bins)
    n_bins = 20
    z_sorted_idx = np.argsort(gals['z'])
    bin_size = len(gals['z']) // n_bins

    v_pec_resid = np.zeros(len(gals['z']))
    z_bin = np.zeros(len(gals['z']), dtype=int)

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(gals['z'])
        idx = z_sorted_idx[start:end]

        z_bin[idx] = i
        z_median = np.median(gals['z'][idx])
        v_pec_resid[idx] = C_KMS * (gals['z'][idx] - z_median) / (1 + z_median)

    gals['cz_resid'] = v_pec_resid  # Renamed: NOT peculiar velocity
    gals['z_bin'] = z_bin

    print(f"  cz_resid range: [{v_pec_resid.min():.1f}, {v_pec_resid.max():.1f}] km/s")
    return gals

# =============================================================================
# VOID PROXIMITY
# =============================================================================
def compute_void_distances(gals, voids):
    """
    Compute distance to nearest void center, normalized by void radius.
    d_void_norm = dist_to_center / r_void
    in_void = (d_void_norm < 1.0)
    """
    print("Computing void proximity...")

    # Convert to 3D Cartesian coordinates
    gal_dist = cosmo.comoving_distance(gals['z']).value
    gal_coords = SkyCoord(ra=gals['ra']*u.deg, dec=gals['dec']*u.deg, distance=gal_dist*u.Mpc)

    void_dist = cosmo.comoving_distance(voids['z']).value
    void_coords = SkyCoord(ra=voids['ra']*u.deg, dec=voids['dec']*u.deg, distance=void_dist*u.Mpc)

    gal_xyz = np.array([gal_coords.cartesian.x.value,
                        gal_coords.cartesian.y.value,
                        gal_coords.cartesian.z.value]).T
    void_xyz = np.array([void_coords.cartesian.x.value,
                         void_coords.cartesian.y.value,
                         void_coords.cartesian.z.value]).T

    # KDTree for fast nearest-neighbor search
    tree = cKDTree(void_xyz)
    dists, idxs = tree.query(gal_xyz, k=1)

    # Normalize by void radius
    void_radii = voids['radius_mpc'][idxs]
    gals['d_void_norm'] = dists / void_radii
    gals['in_void'] = gals['d_void_norm'] < 1.0
    gals['nearest_void_idx'] = idxs

    n_in = np.sum(gals['in_void'])
    n_out = len(gals['in_void']) - n_in
    print(f"  Inside voids: {n_in}")
    print(f"  Outside voids: {n_out}")

    return gals

# =============================================================================
# STATISTICAL TESTS
# =============================================================================
def run_tests(gals):
    """
    Run differential redshift tests with null checks.
    """
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    results = {}

    # --- A. Main Signal: Inside vs Outside ---
    print("\n--- MAIN RESULT ---")

    in_mask = gals['in_void']
    out_mask = ~gals['in_void']

    v_in = gals['cz_resid'][in_mask]
    v_out = gals['cz_resid'][out_mask]

    mean_v_in = np.mean(v_in)
    mean_v_out = np.mean(v_out)
    delta_v = mean_v_in - mean_v_out

    n_in = len(v_in)
    n_out = len(v_out)

    # Standard error of difference
    se_diff = np.sqrt(np.var(v_in)/n_in + np.var(v_out)/n_out)
    sigma = abs(delta_v / se_diff)

    print(f"  Mean v_pec (Inside):  {mean_v_in:+.2f} km/s (n={n_in})")
    print(f"  Mean v_pec (Outside): {mean_v_out:+.2f} km/s (n={n_out})")
    print(f"  Delta V: {delta_v:+.2f} km/s")
    print(f"  Standard Error: {se_diff:.2f} km/s")
    print(f"  Significance: {sigma:.2f}σ")

    results['delta_v'] = delta_v
    results['sigma'] = sigma
    results['mean_v_in'] = mean_v_in
    results['mean_v_out'] = mean_v_out

    # --- B. Null Test 1: Shuffled Labels ---
    print("\n--- NULL TEST 1: Label Shuffling (1000 iterations) ---")

    null_deltas = []
    for i in range(1000):
        shuffled = np.random.permutation(gals['in_void'])
        v_in_s = gals['cz_resid'][shuffled]
        v_out_s = gals['cz_resid'][~shuffled]
        null_deltas.append(np.mean(v_in_s) - np.mean(v_out_s))

    null_mean = np.mean(null_deltas)
    null_std = np.std(null_deltas)

    print(f"  Null Delta V: {null_mean:.2f} ± {null_std:.2f} km/s")
    print(f"  Observed Delta V: {delta_v:+.2f} km/s")

    # P-value from permutation test
    p_value = np.mean(np.abs(null_deltas) >= np.abs(delta_v))
    print(f"  Permutation p-value: {p_value:.4f}")

    if abs(delta_v) > 3*null_std:
        print("  RESULT: Signal ROBUST against random association (PASS)")
        results['null_test_pass'] = True
    else:
        print("  RESULT: Signal indistinguishable from noise (FAIL)")
        results['null_test_pass'] = False

    results['null_std'] = null_std
    results['p_value'] = p_value

    # --- C. Systematics: Magnitude Cut ---
    print("\n--- SYSTEMATICS: Bright Galaxy Subsample ---")

    # Check if mag_r exists and is meaningful
    if np.std(gals['mag_r']) > 0.1:  # Has actual magnitude data
        bright_mask = gals['mag_r'] < 19.0
        bright_in = gals['in_void'] & bright_mask
        bright_out = (~gals['in_void']) & bright_mask

        if np.sum(bright_in) > 10 and np.sum(bright_out) > 10:
            v_in_b = gals['cz_resid'][bright_in]
            v_out_b = gals['cz_resid'][bright_out]
            delta_v_bright = np.mean(v_in_b) - np.mean(v_out_b)

            print(f"  Bright galaxies (r < 19): n_in={np.sum(bright_in)}, n_out={np.sum(bright_out)}")
            print(f"  Delta V (Bright): {delta_v_bright:+.2f} km/s")

            if np.sign(delta_v_bright) == np.sign(delta_v):
                print("  RESULT: Same sign as full sample (consistent)")
            else:
                print("  WARNING: Sign flipped - possible Malmquist bias")
        else:
            print("  Not enough bright galaxies for test")
    else:
        print("  Magnitude data not available in catalog")

    return results, null_deltas

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_figure(gals, results, null_deltas):
    """Create summary figure."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) cz_resid distribution Inside vs Outside
    ax = axes[0, 0]
    bins = np.linspace(-2000, 2000, 50)
    ax.hist(gals['cz_resid'][gals['in_void']], bins=bins, alpha=0.7,
            label=f'Inside (n={np.sum(gals["in_void"])})', density=True, color='blue')
    ax.hist(gals['cz_resid'][~gals['in_void']], bins=bins, alpha=0.7,
            label=f'Outside (n={np.sum(~gals["in_void"])})', density=True, color='red')
    ax.axvline(results['mean_v_in'], color='blue', ls='--', lw=2, label=f'Mean In: {results["mean_v_in"]:.0f}')
    ax.axvline(results['mean_v_out'], color='red', ls='--', lw=2, label=f'Mean Out: {results["mean_v_out"]:.0f}')
    ax.set_xlabel('cz_resid [km/s] (NOT v_pec)')
    ax.set_ylabel('Density')
    ax.set_title('(a) Redshift Residual: Inside vs Outside Voids')
    ax.legend(fontsize=8)
    ax.set_xlim(-2000, 2000)

    # (b) cz_resid vs d_void_norm
    ax = axes[0, 1]
    # Bin by d_void_norm
    d_bins = np.linspace(0, 5, 11)
    d_centers = 0.5 * (d_bins[:-1] + d_bins[1:])
    v_means = []
    v_errs = []
    for i in range(len(d_bins)-1):
        mask = (gals['d_void_norm'] >= d_bins[i]) & (gals['d_void_norm'] < d_bins[i+1])
        if np.sum(mask) > 10:
            v_means.append(np.mean(gals['cz_resid'][mask]))
            v_errs.append(np.std(gals['cz_resid'][mask]) / np.sqrt(np.sum(mask)))
        else:
            v_means.append(np.nan)
            v_errs.append(np.nan)

    ax.errorbar(d_centers, v_means, yerr=v_errs, fmt='ko', capsize=3, markersize=8)
    ax.axvline(1.0, color='gray', ls=':', label='Void boundary')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Distance from void center / Void radius')
    ax.set_ylabel('Mean cz_resid [km/s]')
    ax.set_title('(b) Redshift Residual vs Void Distance')
    ax.legend()

    # (c) Null Test Distribution
    ax = axes[1, 0]
    ax.hist(null_deltas, bins=50, alpha=0.7, color='gray', density=True)
    ax.axvline(results['delta_v'], color='red', lw=2, label=f'Observed: {results["delta_v"]:.0f} km/s')
    ax.axvline(0, color='black', ls='--', alpha=0.5)
    ax.axvline(3*results['null_std'], color='green', ls=':', label=f'3σ: ±{3*results["null_std"]:.0f}')
    ax.axvline(-3*results['null_std'], color='green', ls=':')
    ax.set_xlabel('Delta V (In - Out) [km/s]')
    ax.set_ylabel('Density')
    ax.set_title('(c) Null Test: Shuffled Labels (1000 iter)')
    ax.legend()

    # (d) Summary Box
    ax = axes[1, 1]
    ax.axis('off')

    summary = f"""
VOID REDSHIFT-SPACE KINEMATIC TEST
{'='*40}

DEFINITION (NOT peculiar velocity!):
  cz_resid = c × (z - median(z in bin)) / (1+z)
  in_void = d_void_norm < 1.0
  This is a z-space proxy, NOT true v_pec.

MAIN RESULT:
  Mean cz_resid (Inside):  {results['mean_v_in']:+.1f} km/s
  Mean cz_resid (Outside): {results['mean_v_out']:+.1f} km/s
  Delta: {results['delta_v']:+.1f} km/s
  Significance: {results['sigma']:.1f}σ

NULL TEST (Label Shuffle):
  Null σ: {results['null_std']:.1f} km/s
  Permutation p-value: {results['p_value']:.4f}
  Result: {'PASS' if results['null_test_pass'] else 'FAIL'}

INTERPRETATION:
  This detects KINEMATIC void dynamics (outflow/RSD)
  NOT a direct probe of photon coupling.
  Consistent with ΛCDM void outflow expectations.
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent / 'output' / 'mtdf_void_redshift_gemini.png'), dpi=150, bbox_inches='tight')
    plt.savefig(str(Path(__file__).parent.parent / 'output' / 'mtdf_void_redshift_gemini.pdf'), bbox_inches='tight')
    print(f"\nFigure saved to output/mtdf_void_redshift_gemini.png/pdf")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("MTDF VOID REDSHIFT TEST - GEMINI 'COURT PROOF' VERSION")
    print("="*70)

    # Load data
    gals = load_gama_data()
    voids = load_void_data()

    # Process
    gals = calculate_redshift_residual(gals)
    gals = compute_void_distances(gals, voids)

    # Test
    results, null_deltas = run_tests(gals)

    # Visualize
    create_figure(gals, results, null_deltas)

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if results['null_test_pass'] and results['sigma'] > 3:
        print("✓ Signal passes null test")
        print("✓ Significance > 3σ")
        print("\nThis is a ROBUST detection requiring explanation.")
        print("Possible interpretations:")
        print("  1. MTDF stress-photon coupling (if sign matches prediction)")
        print("  2. Void outflow signature (if redshift for void galaxies)")
        print("  3. Unknown systematic (requires further investigation)")
    else:
        print("Signal does not meet 'court proof' threshold.")
        print(f"  Null test: {'PASS' if results['null_test_pass'] else 'FAIL'}")
        print(f"  Significance: {results['sigma']:.1f}σ")
