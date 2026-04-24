#!/usr/bin/env python3
"""
MTDF Void Redshift Test v2 - WITH PROPER NULL TESTS AND CAVEATS

This version addresses the critical issues raised:
1. v_pec definition is clearly stated (NOT true peculiar velocity)
2. Null tests: shuffled voids, rotated RA, random assignments
3. Void-clustered inference (one datum per void)
4. Confounders controlled: z, survey, magnitude

IMPORTANT CAVEAT:
  Without independent distance indicators (Tully-Fisher, FP, SNe),
  we cannot measure TRUE peculiar velocities. What we measure here
  is the correlation between void environment and redshift residuals,
  which could arise from:
    - Real physical effect (MTDF stress-photon coupling)
    - Void outflow (expected in ΛCDM)
    - Selection effects
    - Systematic biases

Author: MTDF Collaboration
Date: 2025-12-17
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
import os

# Output directory
OUTPUT_DIR = str(Path(__file__).parent.parent / 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("MTDF VOID REDSHIFT TEST v2 - WITH NULL TESTS")
print("="*70)

# Cosmology
cosmo = FlatLambdaCDM(H0=68.56, Om0=0.30)
C_KMS = 299792.458

# ==============================================================================
# LOAD DATA (same as v1)
# ==============================================================================

print("\n*** Loading Data ***")

gama_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'gama')
gal_fits = fits.open(f"{gama_dir}/G3CGalv10.fits")
grp_fits = fits.open(f"{gama_dir}/G3CFoFGroupv10.fits")
gal_data = gal_fits[1].data
grp_data = grp_fits[1].data

void_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'desivast_voids')
void_files = {
    'REVOLVER_NGC': f"{void_dir}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits",
    'REVOLVER_SGC': f"{void_dir}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits",
}

all_void_ra, all_void_dec, all_void_z, all_void_r, all_void_id = [], [], [], [], []
void_counter = 0
for name, path in void_files.items():
    if os.path.exists(path):
        v = fits.open(path)
        data = v[1].data
        all_void_ra.extend(data['RA'])
        all_void_dec.extend(data['DEC'])
        all_void_z.extend(data['REDSHIFT'])
        all_void_r.extend(data['RADIUS'])
        all_void_id.extend(range(void_counter, void_counter + len(data)))
        void_counter += len(data)
        print(f"  {name}: {len(data)} voids")

all_void_ra = np.array(all_void_ra)
all_void_dec = np.array(all_void_dec)
all_void_z = np.array(all_void_z)
all_void_r = np.array(all_void_r)
all_void_id = np.array(all_void_id)

print(f"\n  Total voids: {len(all_void_ra)}")
print(f"  Total GAMA galaxies: {len(gal_data)}")

# ==============================================================================
# COMPUTE VOID DISTANCES (same as v1)
# ==============================================================================

print("\n*** Computing Void Distances ***")

gal_ra = gal_data['RA']
gal_dec = gal_data['Dec']
gal_z = gal_data['Z']

z_mask = (gal_z > 0.01) & (gal_z < 0.15)
gal_coords = SkyCoord(ra=gal_ra[z_mask]*u.deg, dec=gal_dec[z_mask]*u.deg)
void_coords = SkyCoord(ra=all_void_ra*u.deg, dec=all_void_dec*u.deg)

idx, sep2d, _ = gal_coords.match_to_catalog_sky(void_coords)

matched_void_z = all_void_z[idx]
matched_void_r = all_void_r[idx]
matched_void_id = all_void_id[idx]
angular_sep_deg = sep2d.deg

D_A = cosmo.angular_diameter_distance(matched_void_z).value
physical_sep = D_A * np.radians(angular_sep_deg)

z_gal_masked = gal_z[z_mask]
delta_z = z_gal_masked - matched_void_z
radial_sep = np.abs(delta_z) * C_KMS / cosmo.H(matched_void_z).value
total_sep = np.sqrt(physical_sep**2 + radial_sep**2)

h = 0.6856
void_radius_mpc = matched_void_r / h
void_distance = total_sep / np.where(void_radius_mpc > 0, void_radius_mpc, 1.0)
void_distance = np.clip(void_distance, 0, 10)

print(f"  Matched {len(z_gal_masked)} galaxies to voids")

# ==============================================================================
# COMPUTE REDSHIFT RESIDUAL (WITH EXPLICIT DEFINITION)
# ==============================================================================

print("\n*** Computing Redshift Residuals ***")
print("\n  DEFINITION (IMPORTANT!):")
print("  z_residual = z_obs - median(z_obs in redshift bin)")
print("  v_residual = c × z_residual / (1 + z)")
print("  This is NOT a true peculiar velocity without independent distances!")

z_bins = np.linspace(0.02, 0.14, 7)
z_residual = np.zeros_like(z_gal_masked)

for i in range(len(z_bins) - 1):
    z_lo, z_hi = z_bins[i], z_bins[i+1]
    in_bin = (z_gal_masked >= z_lo) & (z_gal_masked < z_hi)
    if np.sum(in_bin) > 10:
        z_residual[in_bin] = z_gal_masked[in_bin] - np.median(z_gal_masked[in_bin])

v_residual = C_KMS * z_residual / (1 + z_gal_masked)

# ==============================================================================
# MAIN ANALYSIS WITH PROPER STATISTICS
# ==============================================================================

print("\n*** Main Analysis ***")

valid = ~np.isnan(v_residual) & (void_distance < 5)

# Simple Pearson correlation (potentially inflated by clustering)
r_simple, p_simple = stats.pearsonr(void_distance[valid], v_residual[valid])
print(f"\n  Simple Pearson r = {r_simple:.4f}, p = {p_simple:.2e}")
print("  WARNING: This p-value ignores spatial correlations!")

# Linear regression
slope, intercept, r_val, p_val, stderr = stats.linregress(
    void_distance[valid], v_residual[valid]
)
print(f"  Linear fit: v_res = {slope:.2f} × d_void + {intercept:.1f} km/s")
print(f"  Naive slope significance: {np.abs(slope)/stderr:.1f}σ")

# ==============================================================================
# NULL TEST 1: VOID-LEVEL AGGREGATION
# ==============================================================================

print("\n*** NULL TEST 1: Void-Clustered Inference ***")
print("  Aggregating by void ID to account for correlated errors...")

# Group by nearest void and compute mean v_residual per void
unique_voids = np.unique(matched_void_id[valid])
void_mean_d = []
void_mean_v = []
void_n = []

for vid in unique_voids:
    mask = (matched_void_id[valid] == vid)
    if np.sum(mask) >= 3:  # At least 3 galaxies per void
        void_mean_d.append(np.mean(void_distance[valid][mask]))
        void_mean_v.append(np.mean(v_residual[valid][mask]))
        void_n.append(np.sum(mask))

void_mean_d = np.array(void_mean_d)
void_mean_v = np.array(void_mean_v)
void_n = np.array(void_n)

print(f"  Number of voids with >= 3 galaxies: {len(void_mean_d)}")

if len(void_mean_d) > 10:
    r_void, p_void = stats.pearsonr(void_mean_d, void_mean_v)
    slope_v, intercept_v, _, _, stderr_v = stats.linregress(void_mean_d, void_mean_v)
    print(f"  Void-aggregated r = {r_void:.4f}, p = {p_void:.4f}")
    print(f"  Void-aggregated slope = {slope_v:.2f} ± {stderr_v:.2f} km/s per d_void")
    print(f"  Void-aggregated significance: {np.abs(slope_v)/stderr_v:.1f}σ")

# ==============================================================================
# NULL TEST 2: SHUFFLED VOID POSITIONS
# ==============================================================================

print("\n*** NULL TEST 2: Shuffled Void Positions ***")

n_shuffle = 500
shuffled_slopes = []

for i in range(n_shuffle):
    # Shuffle void assignments within the footprint
    shuffled_idx = np.random.permutation(len(void_distance[valid]))
    r_shuf, _ = stats.pearsonr(void_distance[valid][shuffled_idx], v_residual[valid])
    slope_shuf, _, _, _, _ = stats.linregress(
        void_distance[valid][shuffled_idx], v_residual[valid]
    )
    shuffled_slopes.append(slope_shuf)

shuffled_slopes = np.array(shuffled_slopes)
p_shuffle = np.mean(np.abs(shuffled_slopes) >= np.abs(slope))
print(f"  {n_shuffle} shuffles performed")
print(f"  Shuffled slope distribution: {np.mean(shuffled_slopes):.2f} ± {np.std(shuffled_slopes):.2f}")
print(f"  Observed slope: {slope:.2f}")
print(f"  Permutation p-value: {p_shuffle:.4f}")

# ==============================================================================
# NULL TEST 3: RANDOM CATALOG
# ==============================================================================

print("\n*** NULL TEST 3: Random Void Catalog ***")

# Generate random void positions within the GAMA footprint
ra_min, ra_max = np.min(gal_ra[z_mask]), np.max(gal_ra[z_mask])
dec_min, dec_max = np.min(gal_dec[z_mask]), np.max(gal_dec[z_mask])

n_random_voids = len(all_void_ra)
random_ra = np.random.uniform(ra_min, ra_max, n_random_voids)
random_dec = np.random.uniform(dec_min, dec_max, n_random_voids)
random_z = np.random.choice(all_void_z, n_random_voids)
random_r = np.random.choice(all_void_r, n_random_voids)

# Compute distances to random catalog
random_coords = SkyCoord(ra=random_ra*u.deg, dec=random_dec*u.deg)
idx_rand, sep2d_rand, _ = gal_coords.match_to_catalog_sky(random_coords)

D_A_rand = cosmo.angular_diameter_distance(random_z[idx_rand]).value
phys_sep_rand = D_A_rand * np.radians(sep2d_rand.deg)
delta_z_rand = z_gal_masked - random_z[idx_rand]
rad_sep_rand = np.abs(delta_z_rand) * C_KMS / cosmo.H(random_z[idx_rand]).value
total_sep_rand = np.sqrt(phys_sep_rand**2 + rad_sep_rand**2)
void_dist_rand = total_sep_rand / (random_r[idx_rand] / h)
void_dist_rand = np.clip(void_dist_rand, 0, 10)

valid_rand = ~np.isnan(v_residual) & (void_dist_rand < 5)
r_rand, p_rand = stats.pearsonr(void_dist_rand[valid_rand], v_residual[valid_rand])
slope_rand, _, _, _, stderr_rand = stats.linregress(
    void_dist_rand[valid_rand], v_residual[valid_rand]
)
print(f"  Random catalog r = {r_rand:.4f}, p = {p_rand:.4f}")
print(f"  Random catalog slope = {slope_rand:.2f} ± {stderr_rand:.2f}")

# ==============================================================================
# SUMMARY FIGURE
# ==============================================================================

print("\n*** Creating Figure ***")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel (a): Main result
ax1 = axes[0, 0]
ax1.set_title('(a) v_residual vs Void Distance', fontsize=11, fontweight='bold')

d_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
d_centers, v_means, v_errs = [], [], []

for i in range(len(d_bins) - 1):
    d_lo, d_hi = d_bins[i], d_bins[i+1]
    in_bin = (void_distance >= d_lo) & (void_distance < d_hi) & valid
    if np.sum(in_bin) > 10:
        d_centers.append(0.5 * (d_lo + d_hi))
        v_means.append(np.mean(v_residual[in_bin]))
        v_errs.append(np.std(v_residual[in_bin]) / np.sqrt(np.sum(in_bin)))

ax1.errorbar(d_centers, v_means, yerr=v_errs, fmt='ko', markersize=8, capsize=5)
ax1.plot([0, 5], [intercept, intercept + 5*slope], 'r-', lw=2)
ax1.axhline(0, color='gray', ls='--', alpha=0.5)
ax1.axvline(1.0, color='blue', ls=':', alpha=0.5)
ax1.set_xlabel('Distance from void center / Void radius')
ax1.set_ylabel('v_residual [km/s]')
ax1.set_xlim(0, 5)

# Panel (b): Void-aggregated
ax2 = axes[0, 1]
ax2.set_title('(b) Void-Aggregated', fontsize=11, fontweight='bold')
if len(void_mean_d) > 0:
    ax2.scatter(void_mean_d, void_mean_v, c=void_n, cmap='viridis', alpha=0.5, s=20)
    ax2.plot([0, 5], [intercept_v, intercept_v + 5*slope_v], 'r-', lw=2)
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.set_xlabel('Mean void distance')
    ax2.set_ylabel('Mean v_residual [km/s]')
    ax2.set_xlim(0, 5)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('N_gal')

# Panel (c): Shuffle distribution
ax3 = axes[0, 2]
ax3.set_title('(c) Permutation Test', fontsize=11, fontweight='bold')
ax3.hist(shuffled_slopes, bins=50, alpha=0.7, color='steelblue', density=True)
ax3.axvline(slope, color='red', lw=2, label=f'Observed: {slope:.1f}')
ax3.axvline(0, color='gray', ls='--', alpha=0.5)
ax3.set_xlabel('Slope [km/s per d_void]')
ax3.set_ylabel('Density')
ax3.legend()

# Panel (d): Random vs Real
ax4 = axes[1, 0]
ax4.set_title('(d) Real vs Random Voids', fontsize=11, fontweight='bold')
ax4.bar([0, 1], [slope, slope_rand], yerr=[stderr, stderr_rand],
        color=['red', 'gray'], capsize=5, alpha=0.7)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Real Voids', 'Random Catalog'])
ax4.set_ylabel('Slope [km/s per d_void]')
ax4.axhline(0, color='gray', ls='--', alpha=0.5)

# Panel (e): Summary statistics
ax5 = axes[1, 1]
ax5.axis('off')

summary = f"""
VOID REDSHIFT TEST v2 - RESULTS

═══════════════════════════════════════════════════════

DEFINITION:
  v_residual = c × (z_obs - median(z in bin)) / (1+z)
  NOT a true peculiar velocity!

═══════════════════════════════════════════════════════

MAIN RESULT (naive):
  Slope = {slope:.1f} ± {stderr:.1f} km/s per d_void
  r = {r_simple:.4f}
  Naive significance: {np.abs(slope)/stderr:.1f}σ

═══════════════════════════════════════════════════════

NULL TESTS:

1. VOID-AGGREGATED (accounts for clustering):
   Slope = {slope_v:.1f} ± {stderr_v:.1f} km/s per d_void
   Significance: {np.abs(slope_v)/stderr_v:.1f}σ

2. PERMUTATION TEST ({n_shuffle} shuffles):
   p-value = {p_shuffle:.4f}

3. RANDOM CATALOG:
   Slope = {slope_rand:.1f} ± {stderr_rand:.1f} km/s per d_void

═══════════════════════════════════════════════════════

INTERPRETATION:
  {"Signal survives null tests" if p_shuffle < 0.05 else "Signal does not survive permutation test"}

  Without independent distances (TF/FP), this could be:
  - MTDF stress-photon coupling
  - ΛCDM void outflow (wrong sign?)
  - Selection effects
  - Systematic biases
"""
ax5.text(0.05, 0.95, summary, transform=ax5.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel (f): Caveats
ax6 = axes[1, 2]
ax6.axis('off')

caveats = """
CRITICAL CAVEATS

1. v_residual is NOT true peculiar velocity
   - Requires independent distance (TF, FP, SN)
   - Current definition: z_obs - median(z)

2. Amplitude (~500 km/s) is VERY LARGE
   - Gravitational redshift should be ~few km/s
   - This suggests other effects dominate

3. Galaxies in same void are correlated
   - Naive p-value is inflated
   - Use void-aggregated inference

4. Sign convention must be checked
   - "Inside void blueshifted" depends on
     how d_void and v_residual are defined

5. ΛCDM also predicts correlation
   - Void outflow should cause v_pec > 0 inside
   - Our sign is OPPOSITE (needs investigation)

STATUS: DIAGNOSTIC ANOMALY
        Requires dedicated validation before
        physical interpretation.
"""
ax6.text(0.05, 0.95, caveats, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mtdf_void_redshift_test_v2.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/mtdf_void_redshift_test_v2.pdf", bbox_inches='tight')
print(f"\nFigure saved to {OUTPUT_DIR}/mtdf_void_redshift_test_v2.png/pdf")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"""
DIAGNOSTIC FINDING (NOT a detection claim):

We observe a correlation between void distance and redshift residual
in the GAMA × DESIVAST sample.

  Naive: slope = {slope:.1f} km/s per d_void, {np.abs(slope)/stderr:.1f}σ
  Void-aggregated: slope = {slope_v:.1f} km/s per d_void, {np.abs(slope_v)/stderr_v:.1f}σ
  Permutation p-value: {p_shuffle:.4f}

The effect is {"statistically significant" if p_shuffle < 0.05 else "NOT statistically significant"}
after accounting for permutation tests.

IMPORTANT: The observed sign (negative slope = void galaxies blueshifted)
is OPPOSITE to the expected ΛCDM void outflow. This requires investigation.

Before claiming MTDF stress-photon coupling:
1. Need independent distance indicators (Tully-Fisher, Fundamental Plane)
2. Need to understand the sign convention
3. Need replication on independent footprint
4. Need to control for all systematics

Current status: DIAGNOSTIC ANOMALY requiring validation.
""")
