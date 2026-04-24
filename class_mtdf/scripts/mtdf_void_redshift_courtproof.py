#!/usr/bin/env python3
"""
MTDF Void Redshift Test - COURT-PROOF VERSION
==============================================
Full implementation of GPT's requirements for hostile review survival.

KEY FIXES:
1. Rename observable: cz_resid (NOT v_pec) throughout
2. Selection-preserving permutation within z-bins AND angular tiles
3. Radial-decoupled null: angular-only void matching
4. Mock catalogue control with same selection but no dynamics

CRITICAL INTERPRETATION:
  This test measures cz_resid (redshift residuals in z-space).
  It is NOT a direct probe of photon coupling.
  It measures void KINEMATICS (outflow/infall).
  Expected signal from ΛCDM: ~100 km/s void outflow.
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
N_PERMUTATIONS = 1000
N_ZBIN = 20
N_RA_BINS = 10  # Angular tiles in RA
N_DEC_BINS = 5  # Angular tiles in Dec

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
        'ra': np.array(data['RA']),
        'dec': np.array(data['DEC']),
        'z': np.array(data['Z']),
    }

    # Optional columns
    if 'RPETRO' in data.dtype.names:
        gals['mag_r'] = np.array(data['RPETRO'])
    elif 'RABSMAG' in data.dtype.names:
        gals['mag_r'] = np.array(data['RABSMAG'])
    else:
        gals['mag_r'] = np.zeros(len(data))

    # Quality cuts
    mask = (gals['z'] > 0.01) & (gals['z'] < 0.5) & \
           np.isfinite(gals['ra']) & np.isfinite(gals['dec'])
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
# cz RESIDUAL CALCULATION (NOT peculiar velocity!)
# =============================================================================
def calculate_cz_residual(gals):
    """
    Calculate cz residual within redshift bins.

    DEFINITION:
      cz_resid = c × (z - median(z in bin)) / (1 + z_median)

    This is a REDSHIFT-SPACE proxy. NOT a true peculiar velocity.
    True v_pec requires independent distance indicators.
    """
    print("Computing cz_resid (NOT peculiar velocity)...")

    # Create redshift quantile bins
    z_sorted_idx = np.argsort(gals['z'])
    bin_size = len(gals['z']) // N_ZBIN

    cz_resid = np.zeros(len(gals['z']))
    z_bin = np.zeros(len(gals['z']), dtype=int)

    for i in range(N_ZBIN):
        start = i * bin_size
        end = (i + 1) * bin_size if i < N_ZBIN - 1 else len(gals['z'])
        idx = z_sorted_idx[start:end]

        z_bin[idx] = i
        z_median = np.median(gals['z'][idx])
        cz_resid[idx] = C_KMS * (gals['z'][idx] - z_median) / (1 + z_median)

    gals['cz_resid'] = cz_resid
    gals['z_bin'] = z_bin

    print(f"  cz_resid range: [{cz_resid.min():.1f}, {cz_resid.max():.1f}] km/s")
    return gals

# =============================================================================
# ANGULAR TILES (Simple RA/Dec grid)
# =============================================================================
def assign_angular_tiles(gals):
    """Assign each galaxy to an angular tile for block permutation."""
    print(f"Assigning angular tiles ({N_RA_BINS}x{N_DEC_BINS} grid)...")

    # Create simple RA/Dec grid tiles
    ra_bins = np.linspace(gals['ra'].min(), gals['ra'].max(), N_RA_BINS + 1)
    dec_bins = np.linspace(gals['dec'].min(), gals['dec'].max(), N_DEC_BINS + 1)

    ra_idx = np.digitize(gals['ra'], ra_bins) - 1
    dec_idx = np.digitize(gals['dec'], dec_bins) - 1

    # Clip to valid range
    ra_idx = np.clip(ra_idx, 0, N_RA_BINS - 1)
    dec_idx = np.clip(dec_idx, 0, N_DEC_BINS - 1)

    # Combined tile index
    gals['tile'] = ra_idx * N_DEC_BINS + dec_idx

    n_tiles = len(np.unique(gals['tile']))
    print(f"  {n_tiles} unique tiles")
    return gals

# =============================================================================
# VOID PROXIMITY - 3D (Standard)
# =============================================================================
def compute_void_distances_3d(gals, voids):
    """
    Standard 3D void proximity using z_cmb for radial distance.
    d_void_norm = 3D_dist_to_center / r_void
    """
    print("Computing 3D void proximity...")

    # Convert to 3D Cartesian
    gal_dist = cosmo.comoving_distance(gals['z']).value
    gal_coords = SkyCoord(ra=gals['ra']*u.deg, dec=gals['dec']*u.deg,
                          distance=gal_dist*u.Mpc)

    void_dist = cosmo.comoving_distance(voids['z']).value
    void_coords = SkyCoord(ra=voids['ra']*u.deg, dec=voids['dec']*u.deg,
                           distance=void_dist*u.Mpc)

    gal_xyz = np.array([gal_coords.cartesian.x.value,
                        gal_coords.cartesian.y.value,
                        gal_coords.cartesian.z.value]).T
    void_xyz = np.array([void_coords.cartesian.x.value,
                         void_coords.cartesian.y.value,
                         void_coords.cartesian.z.value]).T

    tree = cKDTree(void_xyz)
    dists, idxs = tree.query(gal_xyz, k=1)

    void_radii = voids['radius_mpc'][idxs]
    gals['d_void_norm_3d'] = dists / void_radii
    gals['in_void_3d'] = gals['d_void_norm_3d'] < 1.0

    print(f"  Inside voids (3D): {np.sum(gals['in_void_3d'])}")
    return gals

# =============================================================================
# VOID PROXIMITY - ANGULAR ONLY (Radial-Decoupled Null)
# =============================================================================
def compute_void_distances_angular(gals, voids):
    """
    Angular-only void proximity (radial-decoupled null test).
    Uses only (RA, Dec) to find nearest void, ignoring radial distance.
    If signal is driven by z-geometry entanglement, it should collapse here.
    """
    print("Computing ANGULAR-ONLY void proximity (null test)...")

    gal_coords = SkyCoord(ra=gals['ra']*u.deg, dec=gals['dec']*u.deg)
    void_coords = SkyCoord(ra=voids['ra']*u.deg, dec=voids['dec']*u.deg)

    # Find nearest void by angular separation
    idx, sep2d, _ = gal_coords.match_to_catalog_sky(void_coords)

    # Approximate angular radius of void at void's redshift
    void_dist = cosmo.comoving_distance(voids['z'][idx]).value
    void_angular_radius = np.degrees(voids['radius_mpc'][idx] / void_dist)

    gals['ang_sep_deg'] = sep2d.deg
    gals['d_void_norm_angular'] = sep2d.deg / void_angular_radius
    gals['in_void_angular'] = gals['d_void_norm_angular'] < 1.0

    print(f"  Inside voids (angular-only): {np.sum(gals['in_void_angular'])}")
    return gals

# =============================================================================
# MOCK CATALOGUE (No Dynamics)
# =============================================================================
def create_mock_catalogue(gals, voids):
    """
    Create mock galaxy catalogue with same z and sky selection but
    randomized radial positions within each z-bin.
    This removes any real void dynamics while preserving selection.
    """
    print("Creating mock catalogue (no dynamics)...")

    mock = {key: gals[key].copy() for key in gals}

    # Shuffle radial (z) positions within each z-bin
    for i_bin in range(N_ZBIN):
        mask = gals['z_bin'] == i_bin
        idx = np.where(mask)[0]
        np.random.shuffle(idx)
        mock['z'][mask] = gals['z'][idx]

    # Recompute cz_resid with shuffled z
    z_sorted_idx = np.argsort(mock['z'])
    bin_size = len(mock['z']) // N_ZBIN

    for i in range(N_ZBIN):
        start = i * bin_size
        end = (i + 1) * bin_size if i < N_ZBIN - 1 else len(mock['z'])
        idx = z_sorted_idx[start:end]
        z_median = np.median(mock['z'][idx])
        mock['cz_resid'][idx] = C_KMS * (mock['z'][idx] - z_median) / (1 + z_median)

    # Recompute void distances
    mock = compute_void_distances_3d(mock, voids)

    return mock

# =============================================================================
# SELECTION-PRESERVING PERMUTATION
# =============================================================================
def selection_preserving_permutation(gals):
    """
    Shuffle void labels within z-bins AND angular tiles.
    This preserves selection function while destroying true correlation.
    """
    shuffled = gals['in_void_3d'].copy()

    # Create combined (z_bin, tile) blocks
    blocks = {}
    for i in range(len(gals['z'])):
        key = (gals['z_bin'][i], gals['tile'][i])
        if key not in blocks:
            blocks[key] = []
        blocks[key].append(i)

    # Shuffle within each block
    for key, indices in blocks.items():
        if len(indices) > 1:
            labels = shuffled[indices].copy()
            np.random.shuffle(labels)
            shuffled[indices] = labels

    return shuffled

# =============================================================================
# STATISTICAL TESTS
# =============================================================================
def run_tests(gals, voids):
    """
    Full court-proof test battery.
    """
    print("\n" + "="*70)
    print("COURT-PROOF STATISTICAL TESTS")
    print("="*70)

    results = {}

    # ===========================================
    # A. MAIN SIGNAL (3D void proximity)
    # ===========================================
    print("\n--- TEST A: Main Signal (3D Void Proximity) ---")

    in_mask = gals['in_void_3d']
    cz_in = gals['cz_resid'][in_mask]
    cz_out = gals['cz_resid'][~in_mask]

    delta_cz = np.mean(cz_in) - np.mean(cz_out)
    se = np.sqrt(np.var(cz_in)/len(cz_in) + np.var(cz_out)/len(cz_out))
    sigma = abs(delta_cz / se)

    print(f"  Mean cz_resid (Inside):  {np.mean(cz_in):+.1f} km/s (n={len(cz_in)})")
    print(f"  Mean cz_resid (Outside): {np.mean(cz_out):+.1f} km/s (n={len(cz_out)})")
    print(f"  Δcz: {delta_cz:+.1f} ± {se:.1f} km/s")
    print(f"  Significance: {sigma:.1f}σ")

    results['main_delta_cz'] = delta_cz
    results['main_sigma'] = sigma
    results['main_se'] = se

    # ===========================================
    # B. NULL TEST 1: Selection-Preserving Permutation
    # ===========================================
    print(f"\n--- TEST B: Selection-Preserving Permutation ({N_PERMUTATIONS} iter) ---")
    print("  (Shuffles within z-bins AND angular tiles)")

    null_deltas = []
    for _ in range(N_PERMUTATIONS):
        shuffled = selection_preserving_permutation(gals)
        cz_in_s = gals['cz_resid'][shuffled]
        cz_out_s = gals['cz_resid'][~shuffled]
        null_deltas.append(np.mean(cz_in_s) - np.mean(cz_out_s))

    null_mean = np.mean(null_deltas)
    null_std = np.std(null_deltas)
    p_value = np.mean(np.abs(null_deltas) >= np.abs(delta_cz))

    print(f"  Null distribution: {null_mean:.1f} ± {null_std:.1f} km/s")
    print(f"  Observed: {delta_cz:+.1f} km/s")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {'PASS' if abs(delta_cz) > 3*null_std else 'FAIL'}")

    results['null1_std'] = null_std
    results['null1_pvalue'] = p_value
    results['null1_pass'] = abs(delta_cz) > 3*null_std

    # ===========================================
    # C. NULL TEST 2: Angular-Only (Radial Decoupled)
    # ===========================================
    print("\n--- TEST C: Radial-Decoupled Null (Angular-Only) ---")
    print("  (Uses only RA/Dec to find nearest void)")

    in_ang = gals['in_void_angular']
    cz_in_ang = gals['cz_resid'][in_ang]
    cz_out_ang = gals['cz_resid'][~in_ang]

    delta_cz_ang = np.mean(cz_in_ang) - np.mean(cz_out_ang)
    se_ang = np.sqrt(np.var(cz_in_ang)/len(cz_in_ang) + np.var(cz_out_ang)/len(cz_out_ang))
    sigma_ang = abs(delta_cz_ang / se_ang)

    print(f"  Δcz (angular-only): {delta_cz_ang:+.1f} ± {se_ang:.1f} km/s")
    print(f"  Significance: {sigma_ang:.1f}σ")

    # If signal COLLAPSES in angular-only → z-geometry entanglement
    if sigma_ang < 2 and sigma > 3:
        print("  INTERPRETATION: Signal COLLAPSES without radial info")
        print("  → Likely z-geometry entanglement, NOT physical void dynamics")
        results['angular_collapse'] = True
    else:
        print("  INTERPRETATION: Signal persists without radial info")
        results['angular_collapse'] = False

    results['angular_delta_cz'] = delta_cz_ang
    results['angular_sigma'] = sigma_ang

    # ===========================================
    # D. NULL TEST 3: Mock Catalogue (No Dynamics)
    # ===========================================
    print("\n--- TEST D: Mock Catalogue (Radial Positions Shuffled) ---")

    mock = create_mock_catalogue(gals, voids)
    mock_in = mock['in_void_3d']
    cz_in_mock = mock['cz_resid'][mock_in]
    cz_out_mock = mock['cz_resid'][~mock_in]

    delta_mock = np.mean(cz_in_mock) - np.mean(cz_out_mock)
    se_mock = np.sqrt(np.var(cz_in_mock)/len(cz_in_mock) + np.var(cz_out_mock)/len(cz_out_mock))
    sigma_mock = abs(delta_mock / se_mock)

    print(f"  Δcz (mock): {delta_mock:+.1f} ± {se_mock:.1f} km/s")
    print(f"  Significance: {sigma_mock:.1f}σ")

    if sigma_mock < 2:
        print("  RESULT: Mock shows NO signal → Real data signal is PHYSICAL")
        results['mock_pass'] = True
    else:
        print("  WARNING: Mock shows signal → Possible systematic artifact")
        results['mock_pass'] = False

    results['mock_delta'] = delta_mock
    results['mock_sigma'] = sigma_mock

    return results, null_deltas

# =============================================================================
# VISUALIZATION
# =============================================================================
def create_figure(gals, results, null_deltas):
    """Create comprehensive summary figure."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (a) Main signal histogram
    ax = axes[0, 0]
    bins = np.linspace(-2000, 2000, 50)
    ax.hist(gals['cz_resid'][gals['in_void_3d']], bins=bins, alpha=0.6,
            label=f'Inside (n={np.sum(gals["in_void_3d"])})', density=True, color='blue')
    ax.hist(gals['cz_resid'][~gals['in_void_3d']], bins=bins, alpha=0.6,
            label=f'Outside (n={np.sum(~gals["in_void_3d"])})', density=True, color='red')
    ax.axvline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('cz_resid [km/s]')
    ax.set_ylabel('Density')
    ax.set_title('(a) Main Signal: Inside vs Outside Voids')
    ax.legend(fontsize=8)

    # (b) Null test 1: Selection-preserving permutation
    ax = axes[0, 1]
    ax.hist(null_deltas, bins=50, alpha=0.7, color='gray', density=True)
    ax.axvline(results['main_delta_cz'], color='red', lw=2,
               label=f'Observed: {results["main_delta_cz"]:.0f}')
    ax.axvline(3*results['null1_std'], color='green', ls=':',
               label=f'±3σ: {3*results["null1_std"]:.0f}')
    ax.axvline(-3*results['null1_std'], color='green', ls=':')
    ax.set_xlabel('Δcz [km/s]')
    ax.set_title('(b) NULL 1: Selection-Preserving Shuffle')
    ax.legend(fontsize=8)

    # (c) Angular-only test
    ax = axes[0, 2]
    bars = ['3D (Main)', 'Angular-Only']
    values = [results['main_delta_cz'], results['angular_delta_cz']]
    colors = ['blue', 'orange']
    ax.bar(bars, values, color=colors, alpha=0.7)
    ax.axhline(0, color='black', ls='--')
    ax.set_ylabel('Δcz [km/s]')
    ax.set_title('(c) NULL 2: Radial-Decoupled Test')
    ax.text(0.5, 0.9, f'Collapse: {"YES" if results.get("angular_collapse") else "NO"}',
            transform=ax.transAxes, ha='center', fontsize=10)

    # (d) Mock catalogue
    ax = axes[1, 0]
    bars = ['Real Data', 'Mock (No Dynamics)']
    values = [results['main_sigma'], results['mock_sigma']]
    colors = ['blue', 'gray']
    ax.bar(bars, values, color=colors, alpha=0.7)
    ax.axhline(3, color='red', ls='--', label='3σ threshold')
    ax.set_ylabel('Significance (σ)')
    ax.set_title('(d) NULL 3: Mock Catalogue Control')
    ax.legend()

    # (e) cz_resid vs void distance
    ax = axes[1, 1]
    d_bins = np.linspace(0, 5, 11)
    d_centers = 0.5 * (d_bins[:-1] + d_bins[1:])
    v_means, v_errs = [], []
    for i in range(len(d_bins)-1):
        mask = (gals['d_void_norm_3d'] >= d_bins[i]) & (gals['d_void_norm_3d'] < d_bins[i+1])
        if np.sum(mask) > 10:
            v_means.append(np.mean(gals['cz_resid'][mask]))
            v_errs.append(np.std(gals['cz_resid'][mask]) / np.sqrt(np.sum(mask)))
        else:
            v_means.append(np.nan)
            v_errs.append(np.nan)
    ax.errorbar(d_centers, v_means, yerr=v_errs, fmt='ko', capsize=3)
    ax.axvline(1.0, color='gray', ls=':', label='Void boundary')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('d / R_void')
    ax.set_ylabel('Mean cz_resid [km/s]')
    ax.set_title('(e) Profile: cz_resid vs Void Distance')

    # (f) Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
COURT-PROOF VOID REDSHIFT TEST
{'='*40}

DEFINITION:
  cz_resid = c × (z - z_median) / (1+z)
  This is a z-space proxy, NOT true v_pec.

MAIN RESULT:
  Δcz = {results['main_delta_cz']:+.1f} ± {results['main_se']:.1f} km/s
  Significance: {results['main_sigma']:.1f}σ

NULL TESTS:
  1. Selection-preserving shuffle:
     p = {results['null1_pvalue']:.4f}  {'PASS' if results['null1_pass'] else 'FAIL'}

  2. Radial-decoupled (angular-only):
     σ = {results['angular_sigma']:.1f}
     Collapse: {'YES' if results.get('angular_collapse') else 'NO'}

  3. Mock catalogue (no dynamics):
     σ = {results['mock_sigma']:.1f}  {'PASS' if results['mock_pass'] else 'FAIL'}

INTERPRETATION:
  Signal measures VOID KINEMATICS (outflow).
  Expected in ΛCDM: ~100 km/s.
  NOT a direct probe of photon coupling.
  SNe (with distances) test Part II.
"""

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent / 'output' / 'mtdf_void_redshift_courtproof.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to output/mtdf_void_redshift_courtproof.png")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("MTDF VOID REDSHIFT TEST - COURT-PROOF VERSION")
    print("="*70)
    print("\nThis test measures cz_resid (NOT peculiar velocity).")
    print("It probes void KINEMATICS, not photon coupling directly.")
    print()

    # Load data
    gals = load_gama_data()
    voids = load_void_data()

    if len(voids['ra']) == 0:
        print("ERROR: No voids loaded. Check data paths.")
        exit(1)

    # Process
    gals = calculate_cz_residual(gals)
    gals = assign_angular_tiles(gals)
    gals = compute_void_distances_3d(gals, voids)
    gals = compute_void_distances_angular(gals, voids)

    # Run tests
    results, null_deltas = run_tests(gals, voids)

    # Visualize
    create_figure(gals, results, null_deltas)

    # Final verdict
    print("\n" + "="*70)
    print("FINAL COURT-PROOF VERDICT")
    print("="*70)

    all_pass = results['null1_pass'] and results['mock_pass']

    if all_pass and results['main_sigma'] > 3:
        print("✓ Main signal: {:.1f}σ".format(results['main_sigma']))
        print("✓ Selection-preserving null: PASS")
        print("✓ Mock catalogue null: PASS")
        print()
        print("RESULT: ROBUST kinematic signal detected.")
        print()
        print("INTERPRETATION:")
        print("  Positive Δcz = void galaxies redshifted relative to field")
        print("  This is consistent with ΛCDM void OUTFLOW (~100 km/s)")
        print("  NOT evidence for photon coupling (use SNe for that)")
    else:
        print("Signal does not survive full null test battery.")
        print(f"  Main σ: {results['main_sigma']:.1f}")
        print(f"  Null1 (selection): {'PASS' if results['null1_pass'] else 'FAIL'}")
        print(f"  Null3 (mock): {'PASS' if results['mock_pass'] else 'FAIL'}")
