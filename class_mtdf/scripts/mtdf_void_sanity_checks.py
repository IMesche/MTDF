#!/usr/bin/env python3
"""
MTDF Void Redshift Test - GPT's 6 Sanity Checks
================================================
Comprehensive validation before claiming any detection.
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

C_KMS = 299792.458

# =============================================================================
# CHECK 1: Column and Unit Audit
# =============================================================================
def check_1_column_audit():
    """Verify columns, units, and coordinate frames."""
    print("\n" + "="*70)
    print("CHECK 1: Column and Unit Audit")
    print("="*70)

    # GAMA catalog
    print("\n--- GAMA G3C Catalog ---")
    with fits.open(GAMA_CATALOG) as hdul:
        cols = hdul[1].columns
        data = hdul[1].data

        print(f"Available columns: {[c.name for c in cols]}")

        # Check RA/DEC ranges
        ra = data['RA']
        dec = data['DEC']
        z = data['Z']

        print(f"\nRA range: [{ra.min():.2f}, {ra.max():.2f}]")
        print(f"  Expected: [0, 360] degrees")
        print(f"  Status: {'OK' if 0 <= ra.min() and ra.max() <= 360 else 'WARNING - check units!'}")

        print(f"\nDEC range: [{dec.min():.2f}, {dec.max():.2f}]")
        print(f"  Expected: [-90, 90] degrees")
        print(f"  Status: {'OK' if -90 <= dec.min() and dec.max() <= 90 else 'WARNING - check units!'}")

        print(f"\nZ range: [{z.min():.4f}, {z.max():.4f}]")
        print(f"  Expected: [0, ~0.5] for GAMA")
        valid_z = (z > 0) & (z < 1)
        print(f"  Valid z fraction: {valid_z.mean():.3f}")

    # Void catalog
    print("\n--- DESIVAST Void Catalog ---")
    void_files = [
        f'{VOID_DIR}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits',
        f'{VOID_DIR}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits',
    ]

    for vf in void_files:
        print(f"\nFile: {vf.split('/')[-1]}")
        with fits.open(vf) as hdul:
            cols = hdul[1].columns
            data = hdul[1].data

            print(f"  Columns: {[c.name for c in cols]}")

            # Check for required columns
            if 'RA' in [c.name for c in cols]:
                print(f"  RA range: [{data['RA'].min():.2f}, {data['RA'].max():.2f}]")
            if 'DEC' in [c.name for c in cols]:
                print(f"  DEC range: [{data['DEC'].min():.2f}, {data['DEC'].max():.2f}]")
            if 'REDSHIFT' in [c.name for c in cols]:
                print(f"  REDSHIFT range: [{data['REDSHIFT'].min():.4f}, {data['REDSHIFT'].max():.4f}]")
            if 'RADIUS' in [c.name for c in cols]:
                print(f"  RADIUS range: [{data['RADIUS'].min():.2f}, {data['RADIUS'].max():.2f}] Mpc")

            # Check if z column exists
            z_cols = [c.name for c in cols if 'z' in c.name.lower() or 'redshift' in c.name.lower()]
            print(f"  Redshift columns found: {z_cols}")

# =============================================================================
# CHECK 2: Inside Void Fraction
# =============================================================================
def check_2_inside_fraction(gals):
    """Check if inside void fraction is reasonable."""
    print("\n" + "="*70)
    print("CHECK 2: Inside Void Fraction")
    print("="*70)

    fraction = gals['in_void'].mean()
    n_in = np.sum(gals['in_void'])
    n_out = len(gals['in_void']) - n_in

    print(f"\nInside void fraction: {fraction:.4f} ({fraction*100:.2f}%)")
    print(f"  Inside: {n_in:,}")
    print(f"  Outside: {n_out:,}")

    # Expectations
    print(f"\nExpected range: ~5-15% for typical void catalogs")

    if fraction < 0.01:
        print("  WARNING: Very few galaxies inside voids - check matching!")
    elif fraction > 0.30:
        print("  WARNING: Too many galaxies inside voids - radius mismatch?")
    else:
        print("  Status: OK - fraction is reasonable")

    return fraction

# =============================================================================
# CHECK 3: d_void_norm Distribution
# =============================================================================
def check_3_dvoid_distribution(gals):
    """Check distribution of normalized void distances."""
    print("\n" + "="*70)
    print("CHECK 3: d_void_norm Distribution")
    print("="*70)

    d = gals['d_void_norm']

    print(f"\nd_void_norm statistics:")
    print(f"  Min: {d.min():.3f}")
    print(f"  Max: {d.max():.3f}")
    print(f"  Mean: {d.mean():.3f}")
    print(f"  Median: {np.median(d):.3f}")

    # Check for sharp peak below 1
    frac_below_1 = np.mean(d < 1)
    frac_below_0_5 = np.mean(d < 0.5)

    print(f"\n  Fraction d < 1.0: {frac_below_1:.4f}")
    print(f"  Fraction d < 0.5: {frac_below_0_5:.4f}")

    if frac_below_0_5 > 0.10:
        print("  WARNING: Too many galaxies very close to void centers")
    else:
        print("  Status: OK - distribution looks reasonable")

    return d

# =============================================================================
# CHECK 4: Proper Shuffle Test (within z-bins)
# =============================================================================
def check_4_shuffle_within_zbins(gals):
    """Shuffle within z-bins, not globally."""
    print("\n" + "="*70)
    print("CHECK 4: Shuffle Within Z-Bins (not global)")
    print("="*70)

    # Original signal
    in_mask = gals['in_void']
    delta_orig = np.mean(gals['cz_resid'][in_mask]) - np.mean(gals['cz_resid'][~in_mask])

    print(f"\nOriginal Delta: {delta_orig:.2f} km/s")

    # Global shuffle (what we did before)
    null_global = []
    for _ in range(500):
        shuffled = np.random.permutation(gals['in_void'])
        d = np.mean(gals['cz_resid'][shuffled]) - np.mean(gals['cz_resid'][~shuffled])
        null_global.append(d)

    print(f"\nGlobal shuffle null: {np.mean(null_global):.2f} +/- {np.std(null_global):.2f} km/s")
    global_sigma = abs(delta_orig) / np.std(null_global)
    print(f"Global shuffle significance: {global_sigma:.1f}σ")

    # Within z-bin shuffle (more conservative)
    null_zbin = []
    z_bins = gals['z_bin']
    unique_bins = np.unique(z_bins)

    for _ in range(500):
        shuffled_in_void = np.zeros(len(gals['in_void']), dtype=bool)
        for zb in unique_bins:
            mask = z_bins == zb
            in_void_this_bin = gals['in_void'][mask]
            shuffled_in_void[mask] = np.random.permutation(in_void_this_bin)

        d = np.mean(gals['cz_resid'][shuffled_in_void]) - np.mean(gals['cz_resid'][~shuffled_in_void])
        null_zbin.append(d)

    print(f"\nZ-bin shuffle null: {np.mean(null_zbin):.2f} +/- {np.std(null_zbin):.2f} km/s")
    zbin_sigma = abs(delta_orig) / np.std(null_zbin)
    print(f"Z-bin shuffle significance: {zbin_sigma:.1f}σ")

    print(f"\nComparison:")
    print(f"  Global: {global_sigma:.1f}σ")
    print(f"  Z-bin:  {zbin_sigma:.1f}σ")

    if zbin_sigma < global_sigma * 0.5:
        print("  WARNING: Significance drops a lot with proper z-bin shuffle!")
    else:
        print("  Status: Signal survives z-bin shuffle")

    return null_global, null_zbin

# =============================================================================
# CHECK 5: Jackknife the Sky
# =============================================================================
def check_5_jackknife_sky(gals):
    """Split sky into tiles and check if signal survives."""
    print("\n" + "="*70)
    print("CHECK 5: Jackknife the Sky")
    print("="*70)

    # Create sky tiles based on RA
    n_tiles = 10
    ra_edges = np.percentile(gals['ra'], np.linspace(0, 100, n_tiles + 1))

    in_mask = gals['in_void']
    delta_full = np.mean(gals['cz_resid'][in_mask]) - np.mean(gals['cz_resid'][~in_mask])

    print(f"\nFull sample Delta: {delta_full:.2f} km/s")
    print(f"\nJackknife (leaving out one RA tile at a time):")

    jackknife_deltas = []
    for i in range(n_tiles):
        # Leave out tile i
        if i < n_tiles - 1:
            mask = ~((gals['ra'] >= ra_edges[i]) & (gals['ra'] < ra_edges[i+1]))
        else:
            mask = ~(gals['ra'] >= ra_edges[i])

        in_jk = gals['in_void'][mask]
        cz_jk = gals['cz_resid'][mask]

        if np.sum(in_jk) > 100 and np.sum(~in_jk) > 100:
            delta_jk = np.mean(cz_jk[in_jk]) - np.mean(cz_jk[~in_jk])
            jackknife_deltas.append(delta_jk)
            print(f"  Tile {i+1}: RA [{ra_edges[i]:.1f}, {ra_edges[i+1]:.1f}] left out -> Delta = {delta_jk:.1f} km/s")

    jk_mean = np.mean(jackknife_deltas)
    jk_std = np.std(jackknife_deltas) * np.sqrt(n_tiles - 1)  # Jackknife error

    print(f"\nJackknife mean: {jk_mean:.2f} +/- {jk_std:.2f} km/s")
    print(f"Jackknife significance: {abs(jk_mean)/jk_std:.1f}σ")

    if np.std(jackknife_deltas) > 0.5 * abs(delta_full):
        print("  WARNING: High variance across sky tiles - check for localized effects")
    else:
        print("  Status: Signal is stable across sky regions")

    return jackknife_deltas

# =============================================================================
# CHECK 6: Random Void Catalog Null
# =============================================================================
def check_6_random_voids(gals, voids):
    """Randomize void positions and check if signal persists."""
    print("\n" + "="*70)
    print("CHECK 6: Random Void Catalog Null")
    print("="*70)

    in_mask = gals['in_void']
    delta_real = np.mean(gals['cz_resid'][in_mask]) - np.mean(gals['cz_resid'][~in_mask])

    print(f"\nReal voids Delta: {delta_real:.2f} km/s")

    # Create random void catalog (shuffle RA)
    random_deltas = []

    for _ in range(20):  # 20 random realizations
        # Shuffle void RA while keeping DEC and z
        random_ra = np.random.permutation(voids['ra'])

        # Recompute void distances with randomized positions
        gal_dist = cosmo.comoving_distance(gals['z']).value
        gal_coords = SkyCoord(ra=gals['ra']*u.deg, dec=gals['dec']*u.deg, distance=gal_dist*u.Mpc)

        void_dist = cosmo.comoving_distance(voids['z']).value
        void_coords = SkyCoord(ra=random_ra*u.deg, dec=voids['dec']*u.deg, distance=void_dist*u.Mpc)

        gal_xyz = np.array([gal_coords.cartesian.x.value,
                            gal_coords.cartesian.y.value,
                            gal_coords.cartesian.z.value]).T
        void_xyz = np.array([void_coords.cartesian.x.value,
                            void_coords.cartesian.y.value,
                            void_coords.cartesian.z.value]).T

        tree = cKDTree(void_xyz)
        dists, idxs = tree.query(gal_xyz, k=1)

        d_void_random = dists / voids['radius_mpc'][idxs]
        in_void_random = d_void_random < 1.0

        delta_random = np.mean(gals['cz_resid'][in_void_random]) - np.mean(gals['cz_resid'][~in_void_random])
        random_deltas.append(delta_random)

    print(f"Random voids Delta: {np.mean(random_deltas):.2f} +/- {np.std(random_deltas):.2f} km/s")

    if abs(np.mean(random_deltas)) > 0.5 * abs(delta_real):
        print("  WARNING: Random voids also show strong signal - possible systematic!")
    else:
        print("  Status: Random voids show much weaker signal - good")

    return random_deltas

# =============================================================================
# DATA LOADING (reused from main script)
# =============================================================================
def load_data():
    """Load GAMA and DESIVAST data."""
    # GAMA
    with fits.open(GAMA_CATALOG) as hdul:
        data = hdul[1].data

    gals = {
        'ra': np.array(data['RA']),
        'dec': np.array(data['DEC']),
        'z': np.array(data['Z']),
    }

    mask = (gals['z'] > 0.01) & (gals['z'] < 0.5) & np.isfinite(gals['ra']) & np.isfinite(gals['dec'])
    for key in gals:
        gals[key] = gals[key][mask]

    # Voids
    void_files = [
        f'{VOID_DIR}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits',
        f'{VOID_DIR}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits',
    ]

    all_ra, all_dec, all_z, all_radius = [], [], [], []
    for vf in void_files:
        with fits.open(vf) as hdul:
            data = hdul[1].data
            all_ra.extend(data['RA'])
            all_dec.extend(data['DEC'])
            all_z.extend(data['REDSHIFT'])
            all_radius.extend(data['RADIUS'])

    voids = {
        'ra': np.array(all_ra),
        'dec': np.array(all_dec),
        'z': np.array(all_z),
        'radius_mpc': np.array(all_radius),
    }

    return gals, voids

def compute_cz_resid(gals):
    """Compute redshift residuals."""
    n_bins = 20
    z_sorted_idx = np.argsort(gals['z'])
    bin_size = len(gals['z']) // n_bins

    cz_resid = np.zeros(len(gals['z']))
    z_bin = np.zeros(len(gals['z']), dtype=int)

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(gals['z'])
        idx = z_sorted_idx[start:end]
        z_bin[idx] = i
        z_median = np.median(gals['z'][idx])
        cz_resid[idx] = C_KMS * (gals['z'][idx] - z_median) / (1 + z_median)

    gals['cz_resid'] = cz_resid
    gals['z_bin'] = z_bin
    return gals

def compute_void_distances(gals, voids):
    """Compute void proximity."""
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

    tree = cKDTree(void_xyz)
    dists, idxs = tree.query(gal_xyz, k=1)

    gals['d_void_norm'] = dists / voids['radius_mpc'][idxs]
    gals['in_void'] = gals['d_void_norm'] < 1.0

    return gals

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("GPT's 6 SANITY CHECKS FOR VOID REDSHIFT TEST")
    print("="*70)

    # Check 1: Column audit (doesn't need loaded data)
    check_1_column_audit()

    # Load data for remaining checks
    print("\n\nLoading data...")
    gals, voids = load_data()
    print(f"  Galaxies: {len(gals['ra']):,}")
    print(f"  Voids: {len(voids['ra']):,}")

    gals = compute_cz_resid(gals)
    gals = compute_void_distances(gals, voids)

    # Remaining checks
    check_2_inside_fraction(gals)
    d_dist = check_3_dvoid_distribution(gals)
    null_global, null_zbin = check_4_shuffle_within_zbins(gals)
    jk_deltas = check_5_jackknife_sky(gals)
    random_deltas = check_6_random_voids(gals, voids)

    # Summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # (a) d_void_norm distribution
    ax = axes[0, 0]
    ax.hist(d_dist, bins=50, range=(0, 5), alpha=0.7, edgecolor='black')
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Void boundary')
    ax.set_xlabel('d_void_norm')
    ax.set_ylabel('Count')
    ax.set_title('Check 3: d_void_norm Distribution')
    ax.legend()

    # (b) Global vs Z-bin shuffle
    ax = axes[0, 1]
    ax.hist(null_global, bins=30, alpha=0.5, label='Global shuffle', density=True)
    ax.hist(null_zbin, bins=30, alpha=0.5, label='Z-bin shuffle', density=True)
    delta_orig = np.mean(gals['cz_resid'][gals['in_void']]) - np.mean(gals['cz_resid'][~gals['in_void']])
    ax.axvline(delta_orig, color='red', lw=2, label=f'Observed: {delta_orig:.0f}')
    ax.set_xlabel('Delta cz_resid [km/s]')
    ax.set_ylabel('Density')
    ax.set_title('Check 4: Shuffle Comparison')
    ax.legend()

    # (c) Jackknife
    ax = axes[0, 2]
    ax.bar(range(len(jk_deltas)), jk_deltas, alpha=0.7)
    ax.axhline(delta_orig, color='red', ls='--', label=f'Full: {delta_orig:.0f}')
    ax.set_xlabel('Tile left out')
    ax.set_ylabel('Delta cz_resid [km/s]')
    ax.set_title('Check 5: Jackknife Sky')
    ax.legend()

    # (d) Random void catalog
    ax = axes[1, 0]
    ax.hist(random_deltas, bins=15, alpha=0.7, edgecolor='black')
    ax.axvline(delta_orig, color='red', lw=2, label=f'Real: {delta_orig:.0f}')
    ax.axvline(np.mean(random_deltas), color='blue', ls='--', label=f'Random: {np.mean(random_deltas):.0f}')
    ax.set_xlabel('Delta cz_resid [km/s]')
    ax.set_ylabel('Count')
    ax.set_title('Check 6: Random Void Catalog')
    ax.legend()

    # (e) Summary table
    ax = axes[1, 1]
    ax.axis('off')

    global_sigma = abs(delta_orig) / np.std(null_global)
    zbin_sigma = abs(delta_orig) / np.std(null_zbin)
    jk_sigma = abs(np.mean(jk_deltas)) / (np.std(jk_deltas) * np.sqrt(len(jk_deltas)-1))

    summary = f"""
SANITY CHECK SUMMARY
====================

1. Column Audit: See terminal output

2. Inside Void Fraction: {gals['in_void'].mean()*100:.1f}%
   (Expected: 5-15%)

3. d_void_norm: median = {np.median(d_dist):.2f}

4. Shuffle Tests:
   Global: {global_sigma:.1f}σ
   Z-bin:  {zbin_sigma:.1f}σ

5. Jackknife: {jk_sigma:.1f}σ
   (std across tiles: {np.std(jk_deltas):.1f} km/s)

6. Random Voids: {np.mean(random_deltas):.1f} km/s
   vs Real: {delta_orig:.1f} km/s

VERDICT:
{'Signal appears ROBUST' if zbin_sigma > 3 and jk_sigma > 2 else 'Signal needs investigation'}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (f) RA distribution check
    ax = axes[1, 2]
    ax.hist(gals['ra'][gals['in_void']], bins=36, alpha=0.5, label='Inside', density=True)
    ax.hist(gals['ra'][~gals['in_void']], bins=36, alpha=0.5, label='Outside', density=True)
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Density')
    ax.set_title('RA Distribution Check')
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent.parent / 'output' / 'mtdf_void_sanity_checks.png'), dpi=150, bbox_inches='tight')
    print(f"\n\nFigure saved to output/mtdf_void_sanity_checks.png")

    print("\n" + "="*70)
    print("ALL SANITY CHECKS COMPLETE")
    print("="*70)
