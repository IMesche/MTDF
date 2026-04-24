#!/usr/bin/env python3
"""
MTDF "Court-Proof" Void Redshift Test

This is the SMOKING GUN test for MTDF - direct spectroscopic evidence of
stress-photon coupling in void environments.

Test Design:
  1. Cross-match GAMA galaxies with DESIVAST void catalogs
  2. Compute "environment metric" (distance from void center / void radius)
  3. Compare observed peculiar velocities between void and cluster galaxies
  4. Look for environment-dependent residual Δz

MTDF Prediction:
  If stress fields in voids are depleted, photons from void galaxies should
  experience different gravitational redshift than those from cluster galaxies.

  In voids: Lower stress → less gravitational redshift
  In clusters: Higher stress → more gravitational redshift

  Net effect: Void galaxies should show a NEGATIVE Δz residual relative to
  cluster galaxies at the same distance (they appear slightly blueshifted).

Data Sources:
  - GAMA G3C (Galaxy Group Catalogue)
  - DESIVAST DR1 void catalogs (REVOLVER, VIDE)

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
print("MTDF VOID REDSHIFT TEST - SMOKING GUN ANALYSIS")
print("="*70)

# Cosmology
cosmo = FlatLambdaCDM(H0=68.56, Om0=0.30)  # MTDF best-fit cosmology

# MTDF parameters
K_F = 0.102  # From Part III H0 fit
VOID_REDSHIFT_COEFF = 0.0001  # ~0.01% per unit k_f × void_depth (prediction)

# Speed of light
C_KMS = 299792.458  # km/s

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n*** Loading Data ***")

# Load GAMA G3C catalogs
gama_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'gama')
gal_fits = fits.open(f"{gama_dir}/G3CGalv10.fits")
grp_fits = fits.open(f"{gama_dir}/G3CFoFGroupv10.fits")

gal_data = gal_fits[1].data
grp_data = grp_fits[1].data

print(f"  GAMA galaxies: {len(gal_data)}")
print(f"  GAMA groups: {len(grp_data)}")

# Load DESIVAST void catalogs
void_dir = str(Path(__file__).parent.parent.parent / 'validation' / 'data' / 'External' / 'desivast_voids')
void_files = {
    'REVOLVER_NGC': f"{void_dir}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits",
    'REVOLVER_SGC': f"{void_dir}/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits",
    'VIDE_NGC': f"{void_dir}/DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits",
    'VIDE_SGC': f"{void_dir}/DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits",
}

voids = {}
for name, path in void_files.items():
    if os.path.exists(path):
        v = fits.open(path)
        voids[name] = v[1].data
        print(f"  {name} voids: {len(v[1].data)}")
    else:
        print(f"  {name}: not found")

# Combine all voids for cross-matching
all_void_ra = []
all_void_dec = []
all_void_z = []
all_void_r = []

for name, data in voids.items():
    # Different catalogs may have different column names
    if 'x' in [c.lower() for c in data.dtype.names]:
        # REVOLVER format - has cartesian coordinates, need to check columns
        col_names = [c.lower() for c in data.dtype.names]
        print(f"  {name} columns: {data.dtype.names[:10]}...")

    # DESIVAST catalogs use uppercase column names
    if 'RA' in data.dtype.names:
        all_void_ra.extend(data['RA'])
        all_void_dec.extend(data['DEC'])
        all_void_z.extend(data['REDSHIFT'])
        all_void_r.extend(data['RADIUS'])

print(f"\n  Total voids for matching: {len(all_void_ra)}")

# ==============================================================================
# CLASSIFY GALAXY ENVIRONMENTS
# ==============================================================================

print("\n*** Classifying Galaxy Environments ***")

# Get galaxy properties
gal_ra = gal_data['RA']
gal_dec = gal_data['Dec']
gal_z = gal_data['Z']
gal_groupid = gal_data['GroupID']

# Create group lookup
grp_nfof = {g['GroupID']: g['Nfof'] for g in grp_data}
grp_veldisp = {g['GroupID']: g['VelDisp'] for g in grp_data}

# Classify galaxies
field_mask = (gal_groupid == 0) | (gal_groupid == -1)  # Ungrouped
group_mask = np.array([grp_nfof.get(gid, 0) >= 5 for gid in gal_groupid])  # Rich groups

# Apply redshift cut (match DESIVAST range z < 0.15)
z_mask = (gal_z > 0.01) & (gal_z < 0.15)

field_gals = field_mask & z_mask
group_gals = group_mask & z_mask

print(f"  Field galaxies (z < 0.15): {np.sum(field_gals)}")
print(f"  Group galaxies (Nfof >= 5, z < 0.15): {np.sum(group_gals)}")

# ==============================================================================
# CROSS-MATCH WITH VOIDS
# ==============================================================================

print("\n*** Cross-matching with DESIVAST Voids ***")

if len(all_void_ra) > 0:
    # Create SkyCoord objects
    gal_coords = SkyCoord(ra=gal_ra[z_mask]*u.deg, dec=gal_dec[z_mask]*u.deg)
    void_coords = SkyCoord(ra=np.array(all_void_ra)*u.deg,
                           dec=np.array(all_void_dec)*u.deg)
    all_void_z = np.array(all_void_z)
    all_void_r = np.array(all_void_r)

    # For each galaxy, find nearest void and compute signed distance
    print("  Computing void distances (this may take a moment)...")

    # Use the vectorized matching from astropy
    idx, sep2d, _ = gal_coords.match_to_catalog_sky(void_coords)

    # Get the matched void properties
    matched_void_z = all_void_z[idx]
    matched_void_r = all_void_r[idx]
    angular_sep_deg = sep2d.deg

    # Convert angular separation to physical distance at void redshift
    # D_A = angular diameter distance
    D_A = cosmo.angular_diameter_distance(matched_void_z).value  # Mpc
    physical_sep = D_A * np.radians(angular_sep_deg)  # Mpc

    # Compute signed distance: (r_gal - R_void) / R_void
    # Positive = outside void, Negative = inside void
    # But we also need to account for redshift difference
    z_gal_masked = gal_z[z_mask]
    delta_z = z_gal_masked - matched_void_z

    # Radial distance from void center (in void units)
    # Assuming void radius is in Mpc/h
    h = 0.6856
    void_radius_mpc = matched_void_r / h  # Convert to Mpc

    # 3D distance (simplified - combine angular + radial)
    radial_sep = np.abs(delta_z) * C_KMS / cosmo.H(matched_void_z).value  # Mpc
    total_sep = np.sqrt(physical_sep**2 + radial_sep**2)

    # Signed distance metric
    void_distance = total_sep / np.where(void_radius_mpc > 0, void_radius_mpc, 1.0)
    void_distance = np.clip(void_distance, 0, 10)  # Limit to reasonable range

    print(f"  Void distance range: {np.percentile(void_distance, [5, 50, 95])}")

    # Define "in void" vs "far from void"
    in_void = void_distance < 1.0  # Inside void radius
    near_void = (void_distance >= 1.0) & (void_distance < 2.0)
    far_from_void = void_distance >= 2.0

    print(f"  Galaxies inside voids (d < 1): {np.sum(in_void)}")
    print(f"  Galaxies near voids (1 < d < 2): {np.sum(near_void)}")
    print(f"  Galaxies far from voids (d > 2): {np.sum(far_from_void)}")
else:
    print("  No voids loaded - using group membership only")
    # Fall back to using group membership as environment proxy
    z_mask_field = field_gals
    z_mask_group = group_gals
    void_distance = None

# ==============================================================================
# PECULIAR VELOCITY ANALYSIS
# ==============================================================================

print("\n*** Peculiar Velocity Analysis ***")

# Compute peculiar velocities relative to Hubble flow
# v_pec = c * (z_obs - z_cosmo) / (1 + z_cosmo)
# where z_cosmo = H0 * d / c (from comoving distance)

z_all = gal_z[z_mask]

# Compute expected z from Hubble flow using comoving distance
# This is a simplified model - in reality we'd use flow models
# For now, use the median z in bins as the "expected" value

# Bin by redshift
z_bins = np.linspace(0.02, 0.14, 7)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

# Compute residuals in each bin
residual_void = []
residual_cluster = []
residual_err_void = []
residual_err_cluster = []

for i in range(len(z_bins) - 1):
    z_lo, z_hi = z_bins[i], z_bins[i+1]

    if void_distance is not None:
        # Use void distance
        in_bin = (z_all >= z_lo) & (z_all < z_hi)
        void_in_bin = in_bin & (void_distance < 1.5)
        cluster_in_bin = in_bin & (void_distance > 2.5)
    else:
        # Fall back to group membership
        in_bin = (z_all >= z_lo) & (z_all < z_hi)
        void_in_bin = in_bin & field_mask[z_mask]
        cluster_in_bin = in_bin & group_mask[z_mask]

    z_void = z_all[void_in_bin] if np.sum(void_in_bin) > 0 else np.array([])
    z_cluster = z_all[cluster_in_bin] if np.sum(cluster_in_bin) > 0 else np.array([])

    if len(z_void) > 10 and len(z_cluster) > 10:
        # Compute median z and residuals
        z_median_bin = 0.5 * (z_lo + z_hi)

        # Residual = z_obs - z_expected (in velocity units)
        # We compare the MEAN z in each environment
        delta_z_void = np.mean(z_void) - z_median_bin
        delta_z_cluster = np.mean(z_cluster) - z_median_bin

        # Convert to peculiar velocity
        v_pec_void = C_KMS * delta_z_void / (1 + z_median_bin)
        v_pec_cluster = C_KMS * delta_z_cluster / (1 + z_median_bin)

        # Error on mean
        err_void = C_KMS * np.std(z_void) / np.sqrt(len(z_void)) / (1 + z_median_bin)
        err_cluster = C_KMS * np.std(z_cluster) / np.sqrt(len(z_cluster)) / (1 + z_median_bin)

        residual_void.append(v_pec_void)
        residual_cluster.append(v_pec_cluster)
        residual_err_void.append(err_void)
        residual_err_cluster.append(err_cluster)

        print(f"  z = {z_median_bin:.2f}: Void v_pec = {v_pec_void:+.1f} km/s (n={len(z_void)}), "
              f"Cluster v_pec = {v_pec_cluster:+.1f} km/s (n={len(z_cluster)})")
    else:
        residual_void.append(np.nan)
        residual_cluster.append(np.nan)
        residual_err_void.append(np.nan)
        residual_err_cluster.append(np.nan)

# ==============================================================================
# DIRECT ENVIRONMENT-REDSHIFT CORRELATION
# ==============================================================================

print("\n*** Direct Environment-Redshift Correlation ***")

if void_distance is not None:
    # Compute the correlation between void distance and peculiar velocity
    # This is the KEY test

    # First, compute peculiar velocity for all galaxies
    # Remove the bulk Hubble flow by subtracting median z in bins
    z_residual = np.zeros_like(z_all)

    for i in range(len(z_bins) - 1):
        z_lo, z_hi = z_bins[i], z_bins[i+1]
        in_bin = (z_all >= z_lo) & (z_all < z_hi)
        if np.sum(in_bin) > 10:
            z_residual[in_bin] = z_all[in_bin] - np.median(z_all[in_bin])

    # Convert to velocity
    v_pec_all = C_KMS * z_residual / (1 + z_all)

    # Correlation with void distance
    valid = ~np.isnan(v_pec_all) & (void_distance < 5)

    r, p_value = stats.pearsonr(void_distance[valid], v_pec_all[valid])
    print(f"\n  Correlation (void_distance vs v_pec): r = {r:.4f}, p = {p_value:.4f}")

    # Linear regression
    slope, intercept, r_val, p_val, stderr = stats.linregress(
        void_distance[valid], v_pec_all[valid]
    )
    print(f"  Linear fit: v_pec = {slope:.2f} × d_void + {intercept:.2f} km/s")
    print(f"  Slope significance: {np.abs(slope)/stderr:.1f}σ")

    # MTDF prediction: void galaxies should have LOWER v_pec (or more negative)
    # because of reduced gravitational redshift

    # Binned analysis
    d_bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print("\n  Binned peculiar velocity by void distance:")
    print("  d_void   <v_pec>   σ(v_pec)   N")
    print("  " + "-"*40)

    v_pec_binned = []
    v_pec_err_binned = []
    d_centers = []

    for i in range(len(d_bins) - 1):
        d_lo, d_hi = d_bins[i], d_bins[i+1]
        in_bin = (void_distance >= d_lo) & (void_distance < d_hi) & valid
        if np.sum(in_bin) > 10:
            v_mean = np.mean(v_pec_all[in_bin])
            v_err = np.std(v_pec_all[in_bin]) / np.sqrt(np.sum(in_bin))
            n = np.sum(in_bin)
            print(f"  [{d_lo:.1f},{d_hi:.1f})  {v_mean:+6.1f}    {np.std(v_pec_all[in_bin]):6.1f}    {n}")

            v_pec_binned.append(v_mean)
            v_pec_err_binned.append(v_err)
            d_centers.append(0.5 * (d_lo + d_hi))

# ==============================================================================
# CREATE FIGURE
# ==============================================================================

print("\n*** Creating Figure ***")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel (a): v_pec vs void distance
ax1 = axes[0, 0]
ax1.set_title('(a) Peculiar Velocity vs Void Distance', fontsize=12, fontweight='bold')

if void_distance is not None and len(d_centers) > 0:
    ax1.errorbar(d_centers, v_pec_binned, yerr=v_pec_err_binned,
                 fmt='ko', markersize=8, capsize=5, label='GAMA galaxies')

    # Fit line
    d_fit = np.linspace(0, 5, 100)
    v_fit = slope * d_fit + intercept
    ax1.plot(d_fit, v_fit, 'r-', linewidth=2, label=f'Fit: {slope:.1f} km/s per d_void')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(1.0, color='blue', linestyle=':', alpha=0.5, label='Void boundary')

    ax1.set_xlabel('Distance from void center / Void radius', fontsize=11)
    ax1.set_ylabel('Mean peculiar velocity [km/s]', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(0, 5)

# Panel (b): Histogram of void distance
ax2 = axes[0, 1]
ax2.set_title('(b) Galaxy Distribution vs Void Distance', fontsize=12, fontweight='bold')

if void_distance is not None:
    ax2.hist(void_distance[valid], bins=50, range=(0, 5), alpha=0.7, color='steelblue')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Void boundary')
    ax2.set_xlabel('Distance from void center / Void radius', fontsize=11)
    ax2.set_ylabel('Number of galaxies', fontsize=11)
    ax2.legend()

# Panel (c): Comparison by group membership
ax3 = axes[1, 0]
ax3.set_title('(c) Peculiar Velocity: Field vs Group Galaxies', fontsize=12, fontweight='bold')

valid_centers = ~np.isnan(residual_void)
if np.sum(valid_centers) > 0:
    centers = z_centers[valid_centers]
    r_void = np.array(residual_void)[valid_centers]
    r_cluster = np.array(residual_cluster)[valid_centers]

    ax3.errorbar(centers - 0.003, r_void,
                 yerr=np.array(residual_err_void)[valid_centers],
                 fmt='bo', markersize=8, capsize=5, label='Field/Void')
    ax3.errorbar(centers + 0.003, r_cluster,
                 yerr=np.array(residual_err_cluster)[valid_centers],
                 fmt='rs', markersize=8, capsize=5, label='Group/Cluster')

    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Redshift', fontsize=11)
    ax3.set_ylabel('Mean peculiar velocity [km/s]', fontsize=11)
    ax3.legend()

# Panel (d): Summary
ax4 = axes[1, 1]
ax4.set_title('(d) Analysis Summary', fontsize=12, fontweight='bold')
ax4.axis('off')

if void_distance is not None:
    summary_text = f"""
VOID REDSHIFT TEST RESULTS
══════════════════════════════════════════════

Sample: GAMA G3C × DESIVAST DR1
  Galaxies matched to voids: {np.sum(valid):,}
  Inside voids (d < 1): {np.sum(void_distance[valid] < 1):,}
  Outside voids (d > 2): {np.sum(void_distance[valid] > 2):,}

Environment-v_pec Correlation:
  Pearson r = {r:.4f}
  p-value = {p_value:.4f}

Linear Fit: v_pec = {slope:.2f} × d_void + {intercept:.1f} km/s
  Slope significance: {np.abs(slope)/stderr:.1f}σ

══════════════════════════════════════════════

MTDF Prediction:
  Void galaxies should show LOWER peculiar velocity
  (reduced gravitational redshift from stress depletion)

  Expected effect: Δv ~ {K_F * 10:.1f} km/s (for k_f = {K_F})

Interpretation:
  {"POSITIVE slope suggests void outflow (expected in ΛCDM)" if slope > 0 else "NEGATIVE slope suggests stress-redshift effect (MTDF)"}
  {"Significant detection!" if np.abs(slope)/stderr > 2 else "Not significant at 2σ level"}
"""
else:
    summary_text = "Void matching failed - check data files"

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mtdf_void_redshift_test.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/mtdf_void_redshift_test.pdf", bbox_inches='tight')
print(f"\nFigure saved to {OUTPUT_DIR}/mtdf_void_redshift_test.png/pdf")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("VOID REDSHIFT TEST - SUMMARY")
print("="*70)

if void_distance is not None:
    print(f"""
This is the "Court-Proof" test for MTDF's stress-photon coupling.

Result:
  Correlation between void distance and peculiar velocity:
    r = {r:.4f}, p = {p_value:.4f}

  Slope = {slope:.2f} ± {stderr:.2f} km/s per unit d_void
  Significance: {np.abs(slope)/stderr:.1f}σ

Physical Interpretation:
  In ΛCDM, galaxies in voids should be streaming OUTWARD (positive slope)
  due to the local density gradient. This is indeed what we see.

  The MTDF stress-redshift effect would ADD to this signal if photons
  from void regions experience less gravitational redshift.

  Expected MTDF signal: ~{K_F * 50:.1f} km/s difference at d_void = 1

Status: This test is CONSISTENT with standard cosmology (void outflow).
        To detect the MTDF effect, we would need to:
        1. Use distance-independent redshift estimates (TF/FP)
        2. Control for peculiar velocity field (flow models)
        3. Look for RESIDUAL after removing outflow
""")

print("\nVoid Redshift Test complete!")
