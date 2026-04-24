#!/usr/bin/env python3
"""
Task 3A N-body Mock: MDPL2 halo velocities vs observed CF4 signal.

Addresses the referee concern: "linear theory mock might miss non-linear effects."
Uses real N-body halo velocities from the MDPL2 simulation (CosmoSim).

MDPL2: 1 Gpc/h box, 3840^3 particles, Planck cosmology
       (Om=0.307, Ob=0.048, h=0.678, sigma8=0.829, ns=0.96)

Approach:
  1. Download MDPL2 z=0 halos (Mvir > 1e13, ~480K group-scale halos)
  2. Place observer at random positions in the box
  3. Select halos within z < 0.15 (comoving distance < 450 Mpc/h)
  4. Compute line-of-sight peculiar velocities (full N-body, not linear theory)
  5. Find voids in the halo distribution (density grid + spherical void finder)
  6. Compute d_signed for each halo
  7. Add CF4-like noise, apply shell-median subtraction
  8. Measure gamma_v using fast WLS
  9. Repeat for N_OBSERVERS random positions
  10. Compare distribution with observed CF4 gamma_v = -53.6 km/s

The critical test: Mock 1 (linear theory) found gamma_v = +8.7 (wrong sign).
If N-body also gives the wrong sign, non-linear corrections cannot save LCDM.

Author: Ingo Mesche
Date: April 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label
from astropy.cosmology import FlatLambdaCDM
import os, sys, json, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sn_void_hardening'))
from common import save_results

sys.path.insert(0, os.path.dirname(__file__))
from task3a_lcdm_mock import fast_wls_gamma

# MDPL2 parameters
BOX_SIZE = 1000.0   # Mpc/h
OMEGA_M = 0.307
H0 = 100.0          # km/s/Mpc (for Mpc/h coordinates)
H0_FID = 75.0       # for CF4 uncertainty conversion
C_KMS = 299792.458
COSMO = FlatLambdaCDM(H0=100, Om0=OMEGA_M)

# Analysis parameters
Z_MAX = 0.15
D_MAX = COSMO.comoving_distance(Z_MAX).value  # ~440 Mpc/h
Z_CUT = 0.04        # piecewise redshift boundary
D_CUT = COSMO.comoving_distance(Z_CUT).value  # ~120 Mpc/h
N_OBSERVERS = 100    # random observer placements
MVIR_MIN = 1e13      # Msun/h, galaxy group scale (matches CF4)

# Void finding parameters
NGRID = 200          # density grid cells per side (5 Mpc/h resolution)
R_SMOOTH = 10.0      # Mpc/h, Gaussian smoothing for density
DELTA_VOID = -0.7    # overdensity threshold for void identification
R_VOID_MIN = 10.0    # Mpc/h, minimum void radius
R_VOID_MAX = 50.0    # Mpc/h, maximum void radius

DATA_DIR = os.path.join(os.path.dirname(__file__), 'output', 'task3a_nbody_mock')
CACHE_FILE = os.path.join(DATA_DIR, 'mdpl2_halos_z0.npz')

OUTPUT_DIR = DATA_DIR


# ============================================================
# CosmoSim TAP Download
# ============================================================

def parse_votable(xml_text):
    """Parse VOTable XML and return data as numpy arrays."""
    root = ET.fromstring(xml_text)
    ns = {'v': 'http://www.ivoa.net/xml/VOTable/v1.3'}
    rows = root.findall('.//v:TABLEDATA/v:TR', ns)
    if not rows:
        return None
    data = []
    for tr in rows:
        vals = [float(td.text) for td in tr.findall('v:TD', ns)]
        data.append(vals)
    return np.array(data)


def download_mdpl2_halos():
    """Download MDPL2 z=0 halos from CosmoSim TAP API."""
    if os.path.exists(CACHE_FILE):
        print(f"  Loading cached halos from {CACHE_FILE}", flush=True)
        d = np.load(CACHE_FILE)
        return d['x'], d['y'], d['z'], d['vx'], d['vy'], d['vz'], d['mvir']

    print("  Downloading MDPL2 halos from CosmoSim TAP API...", flush=True)
    print(f"  Selection: snapnum=125 (z=0), Mvir > {MVIR_MIN:.0e} Msun/h", flush=True)

    all_data = []
    batch_size = 100000

    # Partition by x-coordinate to avoid hitting row limits
    # MDPL2 box is 1000 Mpc/h, split into 10 slices
    n_slices = 10
    slice_width = BOX_SIZE / n_slices

    for i in range(n_slices):
        x_lo = i * slice_width
        x_hi = (i + 1) * slice_width
        query = (
            f"SELECT x,y,z,vx,vy,vz,mvir FROM mdpl2.rockstar "
            f"WHERE snapnum=125 AND mvir>{MVIR_MIN:.1f} "
            f"AND x>={x_lo:.1f} AND x<{x_hi:.1f}"
        )
        url = (
            f"https://www.cosmosim.org/tap/sync?"
            f"REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={urllib.parse.quote(query)}"
        )
        print(f"    Slice {i+1}/{n_slices} (x=[{x_lo:.0f},{x_hi:.0f}])...",
              end='', flush=True)

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                xml_text = resp.read().decode('utf-8')
            batch = parse_votable(xml_text)
            if batch is not None:
                all_data.append(batch)
                print(f" {len(batch)} halos", flush=True)
            else:
                print(" 0 halos (or parse error)", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

    if not all_data:
        raise RuntimeError("Failed to download any halos from CosmoSim")

    data = np.vstack(all_data)
    print(f"  Total: {len(data)} halos downloaded", flush=True)

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]
    mvir = data[:, 6]

    # Cache locally
    np.savez_compressed(CACHE_FILE, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, mvir=mvir)
    print(f"  Cached to {CACHE_FILE}", flush=True)

    return x, y, z, vx, vy, vz, mvir


# ============================================================
# Void Finding in N-body Halo Distribution
# ============================================================

def find_voids_in_halos_adaptive(hx, hy, hz, box_size, delta_threshold):
    """Wrapper with custom threshold."""
    return find_voids_in_halos(hx, hy, hz, box_size, delta_threshold)


def find_voids_in_halos(hx, hy, hz, box_size, delta_threshold=None):
    """
    Find voids in the halo distribution using a density grid approach.

    1. Assign halos to 3D grid using nearest-grid-point
    2. Smooth with Gaussian
    3. Find connected underdense regions
    4. Compute void centers and effective radii
    """
    if delta_threshold is None:
        delta_threshold = DELTA_VOID
    cell_size = box_size / NGRID
    print(f"    Density grid: {NGRID}^3 ({cell_size:.1f} Mpc/h cells)", flush=True)

    # NGP assignment (periodic)
    ix = (np.floor(hx / cell_size) % NGRID).astype(int)
    iy = (np.floor(hy / cell_size) % NGRID).astype(int)
    iz = (np.floor(hz / cell_size) % NGRID).astype(int)

    density = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    np.add.at(density, (ix, iy, iz), 1)

    # Convert to overdensity
    mean_density = np.mean(density)
    if mean_density > 0:
        delta = (density - mean_density) / mean_density
    else:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Smooth
    sigma_cells = R_SMOOTH / cell_size
    delta_smooth = gaussian_filter(delta.astype(np.float64), sigma=sigma_cells,
                                   mode='wrap')

    print(f"    delta range: [{delta_smooth.min():.2f}, {delta_smooth.max():.2f}]",
          flush=True)
    print(f"    Fraction below {delta_threshold}: "
          f"{(delta_smooth < delta_threshold).sum() / delta_smooth.size:.4f}", flush=True)

    # Find connected underdense regions
    void_mask = delta_smooth < delta_threshold
    labeled, n_regions = label(void_mask)
    print(f"    Found {n_regions} underdense regions", flush=True)

    if n_regions == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Compute void properties
    void_centers_x = []
    void_centers_y = []
    void_centers_z = []
    void_radii = []

    for region_id in range(1, min(n_regions + 1, 5000)):  # cap at 5000 voids
        cells = np.argwhere(labeled == region_id)
        if len(cells) < 5:  # skip tiny regions
            continue

        # Volume and effective radius
        vol_cells = len(cells) * cell_size**3
        r_eff = (3 * vol_cells / (4 * np.pi))**(1.0 / 3.0)

        if r_eff < R_VOID_MIN or r_eff > R_VOID_MAX:
            continue

        # Center of volume (handle periodic boundaries via circular mean)
        cx = np.mean(cells[:, 0]) * cell_size + cell_size / 2
        cy = np.mean(cells[:, 1]) * cell_size + cell_size / 2
        cz = np.mean(cells[:, 2]) * cell_size + cell_size / 2

        void_centers_x.append(cx % box_size)
        void_centers_y.append(cy % box_size)
        void_centers_z.append(cz % box_size)
        void_radii.append(r_eff)

    vx_arr = np.array(void_centers_x)
    vy_arr = np.array(void_centers_y)
    vz_arr = np.array(void_centers_z)
    vr_arr = np.array(void_radii)

    print(f"    Valid voids: {len(vr_arr)}", flush=True)
    if len(vr_arr) > 0:
        print(f"    Radius range: [{vr_arr.min():.1f}, {vr_arr.max():.1f}] Mpc/h",
              flush=True)
        print(f"    Median radius: {np.median(vr_arr):.1f} Mpc/h", flush=True)

    return vx_arr, vy_arr, vz_arr, vr_arr


def compute_d_signed(gx, gy, gz, void_x, void_y, void_z, void_r):
    """
    Compute signed distance to nearest void boundary.
    d_signed < 0 inside void, > 0 outside.
    Same convention as DESIVAST analysis.
    """
    if len(void_r) == 0:
        return np.zeros(len(gx)), np.zeros(len(gx), dtype=bool)

    void_coords = np.column_stack([void_x, void_y, void_z])
    tree = cKDTree(void_coords)

    gal_coords = np.column_stack([gx, gy, gz])
    dists, indices = tree.query(gal_coords)

    # d_signed = (distance_to_center - R_void) / R_void
    d_signed = (dists - void_r[indices]) / void_r[indices]
    in_void = d_signed < 0

    return d_signed, in_void


# ============================================================
# Observer Placement and Mock Pipeline
# ============================================================

def run_single_observer(hx, hy, hz, hvx, hvy, hvz, hmvir,
                        void_x, void_y, void_z, void_r,
                        obs_pos, cf4_vpec_err_median, rng):
    """
    Run the full pipeline for a single observer position.

    Returns gamma_v, gamma_v_lowz, gamma_v_highz, n_selected, n_voids_used.
    """
    ox, oy, oz = obs_pos

    # Relative positions (periodic boundary handling)
    dx = hx - ox
    dy = hy - oy
    dz = hz - oz

    # Periodic wrapping
    dx = np.where(dx > BOX_SIZE / 2, dx - BOX_SIZE, dx)
    dx = np.where(dx < -BOX_SIZE / 2, dx + BOX_SIZE, dx)
    dy = np.where(dy > BOX_SIZE / 2, dy - BOX_SIZE, dy)
    dy = np.where(dy < -BOX_SIZE / 2, dy + BOX_SIZE, dy)
    dz = np.where(dz > BOX_SIZE / 2, dz - BOX_SIZE, dz)
    dz = np.where(dz < -BOX_SIZE / 2, dz + BOX_SIZE, dz)

    # Comoving distance from observer
    r_com = np.sqrt(dx**2 + dy**2 + dz**2)

    # Select halos within z < Z_MAX
    sel = (r_com > 10.0) & (r_com < D_MAX)  # avoid r=0 singularity
    if sel.sum() < 1000:
        return None

    dx_s, dy_s, dz_s = dx[sel], dy[sel], dz[sel]
    r_s = r_com[sel]
    vx_s, vy_s, vz_s = hvx[sel], hvy[sel], hvz[sel]

    # Line-of-sight unit vectors
    rhat_x = dx_s / r_s
    rhat_y = dy_s / r_s
    rhat_z = dz_s / r_s

    # Line-of-sight peculiar velocity (N-body, fully non-linear)
    v_los = vx_s * rhat_x + vy_s * rhat_y + vz_s * rhat_z

    # Approximate redshift from comoving distance
    # z = H0 * r / c (low-z approximation, good enough for z < 0.15)
    z_cosmo = H0 * r_s / C_KMS

    # Add CF4-like noise
    # CF4 uncertainties scale with distance: vpec_err ~ H0 * dist * ln(10)/5 * e_dmav
    # Typical e_dmav ~ 0.15 mag for CF4
    e_dmav_typical = 0.15
    dist_mpc = r_s  # already in Mpc/h
    vpec_err = H0_FID * dist_mpc * np.log(10) / 5.0 * e_dmav_typical
    vpec_err = np.maximum(vpec_err, 100.0)

    noise = rng.normal(0, vpec_err)
    v_mock = v_los + noise

    # Shell-median subtraction (same as CF4 analysis)
    z_edges = np.arange(0, z_cosmo.max() + 0.005, 0.005)
    v_resid = v_mock.copy()
    for j in range(len(z_edges) - 1):
        zmask = (z_cosmo >= z_edges[j]) & (z_cosmo < z_edges[j + 1])
        if zmask.sum() > 10:
            v_resid[zmask] -= np.median(v_mock[zmask])

    # Compute d_signed using voids (also need periodic wrapping for void positions)
    # Void positions relative to observer
    vdx = void_x - ox
    vdy = void_y - oy
    vdz = void_z - oz
    vdx = np.where(vdx > BOX_SIZE / 2, vdx - BOX_SIZE, vdx)
    vdx = np.where(vdx < -BOX_SIZE / 2, vdx + BOX_SIZE, vdx)
    vdy = np.where(vdy > BOX_SIZE / 2, vdy - BOX_SIZE, vdy)
    vdy = np.where(vdy < -BOX_SIZE / 2, vdy + BOX_SIZE, vdy)
    vdz = np.where(vdz > BOX_SIZE / 2, vdz - BOX_SIZE, vdz)
    vdz = np.where(vdz < -BOX_SIZE / 2, vdz + BOX_SIZE, vdz)

    # Only use voids within observable volume
    vr_com = np.sqrt(vdx**2 + vdy**2 + vdz**2)
    v_sel = vr_com < D_MAX + 50  # slight buffer
    if v_sel.sum() < 5:
        return None

    # Galaxy positions in observer frame
    gx_obs = dx_s
    gy_obs = dy_s
    gz_obs = dz_s

    # Void positions in observer frame
    vx_obs = vdx[v_sel]
    vy_obs = vdy[v_sel]
    vz_obs = vdz[v_sel]
    vr_obs = void_r[v_sel]

    d_signed, in_void = compute_d_signed(gx_obs, gy_obs, gz_obs,
                                          vx_obs, vy_obs, vz_obs, vr_obs)

    # Weights
    w = 1.0 / vpec_err**2

    # Full sample gamma_v
    gamma_full, gamma_err_full, dchi2_full = fast_wls_gamma(v_resid, d_signed, w)

    # Piecewise
    mask_low = z_cosmo < Z_CUT
    mask_high = z_cosmo >= Z_CUT

    gamma_low, gamma_high = 0.0, 0.0
    if mask_low.sum() > 50:
        gamma_low, _, _ = fast_wls_gamma(v_resid[mask_low], d_signed[mask_low],
                                          w[mask_low])
    if mask_high.sum() > 50:
        gamma_high, _, _ = fast_wls_gamma(v_resid[mask_high], d_signed[mask_high],
                                           w[mask_high])

    return {
        'gamma_v': gamma_full,
        'gamma_v_err': gamma_err_full,
        'gamma_v_lowz': gamma_low,
        'gamma_v_highz': gamma_high,
        'n_halos': int(sel.sum()),
        'n_voids_used': int(v_sel.sum()),
        'n_in_void': int(in_void.sum()),
        'frac_in_void': float(in_void.mean()),
        'v_los_std': float(np.std(v_los)),
        'dchi2': dchi2_full,
    }


# ============================================================
# Plotting
# ============================================================

def plot_nbody_comparison(gamma_v_dist, observed_gamma_v, mock1_mean, mock1_std,
                           output_dir):
    """Compare N-body gamma_v distribution to observed and linear theory."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gamma_arr = np.array(gamma_v_dist)
    ax.hist(gamma_arr, bins=15, alpha=0.7, color='#2196F3', density=True,
            label=f'MDPL2 N-body (n={len(gamma_arr)})')

    ax.axvline(observed_gamma_v, color='red', lw=2.5, ls='--',
               label=f'CF4 Observed: {observed_gamma_v:.1f} km/s')
    ax.axvline(np.mean(gamma_arr), color='#2196F3', lw=2,
               label=f'N-body mean: {np.mean(gamma_arr):.1f} +/- {np.std(gamma_arr):.1f}')
    ax.axvline(mock1_mean, color='#4CAF50', lw=2, ls=':',
               label=f'Linear theory: {mock1_mean:.1f} +/- {mock1_std:.1f}')
    ax.axvline(0, color='gray', lw=1, ls='-', alpha=0.5)

    if np.std(gamma_arr) > 0:
        n_sigma = (observed_gamma_v - np.mean(gamma_arr)) / np.std(gamma_arr)
        ax.set_title(f'MDPL2 N-body Mock vs Observed CF4 Signal\n'
                     f'Observed is {abs(n_sigma):.1f} sigma from N-body prediction',
                     fontsize=13)
    else:
        ax.set_title('MDPL2 N-body Mock vs Observed CF4 Signal', fontsize=13)

    ax.set_xlabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, 'nbody_mock_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


def plot_nbody_piecewise(results_list, observed_lowz, observed_highz, output_dir):
    """Compare piecewise N-body results to observed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    gamma_low = [r['gamma_v_lowz'] for r in results_list if r is not None]
    gamma_high = [r['gamma_v_highz'] for r in results_list if r is not None]

    for ax, (label, obs_val, mock_vals) in zip(axes, [
        (f'z < {Z_CUT}', observed_lowz, gamma_low),
        (f'z >= {Z_CUT}', observed_highz, gamma_high),
    ]):
        mock_arr = np.array(mock_vals)
        if len(mock_arr) > 0:
            ax.hist(mock_arr, bins=12, alpha=0.7, color='#2196F3', density=True,
                    label=f'N-body (n={len(mock_arr)})')
            ax.axvline(np.mean(mock_arr), color='#2196F3', lw=2,
                       label=f'Mean: {np.mean(mock_arr):.1f}')
        ax.axvline(obs_val, color='red', lw=2.5, ls='--',
                   label=f'Observed: {obs_val:.1f}')
        ax.axvline(0, color='gray', lw=1, ls='-', alpha=0.5)
        ax.set_xlabel(r'$\gamma_v$ (km/s per $d_{\rm signed}$)')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle('MDPL2 N-body Piecewise: Observed vs Mock', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'nbody_mock_piecewise.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}", flush=True)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70, flush=True)
    print("Task 3A N-BODY MOCK (MDPL2)", flush=True)
    print("Full N-body velocities vs observed CF4 gamma_v signal", flush=True)
    print("=" * 70, flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"MDPL2: {BOX_SIZE:.0f} Mpc/h box, Planck cosmology", flush=True)
    print(f"Mvir > {MVIR_MIN:.0e}, N_observers = {N_OBSERVERS}", flush=True)
    print(f"Void finding: {NGRID}^3 grid, R_smooth={R_SMOOTH}, "
          f"delta_void={DELTA_VOID}", flush=True)
    print(flush=True)

    # Step 1: Download/load MDPL2 halos
    print("--- Step 1: Loading MDPL2 halos ---", flush=True)
    hx, hy, hz, hvx, hvy, hvz, hmvir = download_mdpl2_halos()
    n_halos = len(hx)
    print(f"  {n_halos} halos loaded", flush=True)
    print(f"  Position range: x=[{hx.min():.1f},{hx.max():.1f}], "
          f"y=[{hy.min():.1f},{hy.max():.1f}], z=[{hz.min():.1f},{hz.max():.1f}]",
          flush=True)
    print(f"  Velocity dispersion: vx={np.std(hvx):.0f}, vy={np.std(hvy):.0f}, "
          f"vz={np.std(hvz):.0f} km/s", flush=True)
    print(f"  Mvir range: [{hmvir.min():.2e}, {hmvir.max():.2e}] Msun/h", flush=True)

    # Step 2: Find voids in the full halo distribution
    print("\n--- Step 2: Finding voids in MDPL2 halo distribution ---", flush=True)
    void_x, void_y, void_z, void_r = find_voids_in_halos(hx, hy, hz, BOX_SIZE)

    if len(void_r) == 0:
        print("  ERROR: No voids found! Adjusting threshold...", flush=True)
        for delta_try in [-0.6, -0.5, -0.4]:
            print(f"  Trying delta_void = {delta_try}...", flush=True)
            void_x, void_y, void_z, void_r = find_voids_in_halos_adaptive(
                hx, hy, hz, BOX_SIZE, delta_try)

    if len(void_r) == 0:
        print("  FATAL: Cannot find voids in MDPL2. Exiting.", flush=True)
        return

    # Step 3: Load observed CF4 results for comparison
    print("\n--- Step 3: Loading observed CF4 results ---", flush=True)
    task3a_path = os.path.join(os.path.dirname(__file__),
                                'output', 'task3a_cf4', 'task3a_cf4_results.json')
    with open(task3a_path) as f:
        task3a = json.load(f)

    observed = task3a['catalogues']['VoidFinder']
    obs_gamma_v = observed['full_sample']['gamma_v']
    obs_gamma_lowz = observed['piecewise']['low_z']['gamma_v']
    obs_gamma_highz = observed['piecewise']['high_z']['gamma_v']

    print(f"  Observed gamma_v (VoidFinder):", flush=True)
    print(f"    Full:      {obs_gamma_v:.2f} km/s", flush=True)
    print(f"    z < {Z_CUT}: {obs_gamma_lowz:.2f} km/s", flush=True)
    print(f"    z >= {Z_CUT}: {obs_gamma_highz:.2f} km/s", flush=True)

    # Load linear theory mock results for comparison
    mock_path = os.path.join(os.path.dirname(__file__),
                              'output', 'task3a_lcdm_mock',
                              'task3a_lcdm_mock_results.json')
    mock1_mean, mock1_std = 0.0, 1.0
    if os.path.exists(mock_path):
        with open(mock_path) as f:
            mock_data = json.load(f)
        m1 = mock_data['mocks']['VoidFinder']['mock1_lcdm_reconstruction']
        mock1_mean = m1['gamma_v_mean']
        mock1_std = m1['gamma_v_std']
        print(f"  Linear theory mock: {mock1_mean:.2f} +/- {mock1_std:.2f}", flush=True)

    # Step 4: Run mock for multiple observer positions
    print(f"\n--- Step 4: Running {N_OBSERVERS} observer placements ---", flush=True)
    rng = np.random.RandomState(42)

    # Generate random observer positions (avoiding box edges)
    margin = D_MAX + 50  # need space for full survey volume
    # For a 1 Gpc/h box with D_MAX ~ 440 Mpc/h, margin is ~490.
    # That leaves only ~10 Mpc/h of freedom. Use box center and nearby positions.
    # With periodic boundaries, we can place anywhere.
    obs_positions = rng.uniform(0, BOX_SIZE, size=(N_OBSERVERS, 3))

    results_list = []
    gamma_v_dist = []
    gamma_v_lowz_dist = []
    gamma_v_highz_dist = []

    for i, obs_pos in enumerate(obs_positions):
        print(f"\n  Observer {i+1}/{N_OBSERVERS} at "
              f"({obs_pos[0]:.0f}, {obs_pos[1]:.0f}, {obs_pos[2]:.0f}) Mpc/h",
              flush=True)

        result = run_single_observer(
            hx, hy, hz, hvx, hvy, hvz, hmvir,
            void_x, void_y, void_z, void_r,
            obs_pos, 100.0, rng
        )

        if result is None:
            print("    SKIPPED (too few halos or voids)", flush=True)
            continue

        results_list.append(result)
        gamma_v_dist.append(result['gamma_v'])
        gamma_v_lowz_dist.append(result['gamma_v_lowz'])
        gamma_v_highz_dist.append(result['gamma_v_highz'])

        print(f"    n_halos={result['n_halos']}, n_voids={result['n_voids_used']}, "
              f"frac_in_void={result['frac_in_void']:.3f}", flush=True)
        print(f"    v_los_std = {result['v_los_std']:.0f} km/s", flush=True)
        print(f"    gamma_v = {result['gamma_v']:.2f} +/- {result['gamma_v_err']:.2f} "
              f"(dchi2={result['dchi2']:.1f})", flush=True)
        print(f"    gamma_v_lowz = {result['gamma_v_lowz']:.2f}, "
              f"gamma_v_highz = {result['gamma_v_highz']:.2f}", flush=True)

    # Step 5: Summary statistics
    print("\n" + "=" * 70, flush=True)
    print("N-BODY MOCK RESULTS", flush=True)
    print("=" * 70, flush=True)

    if len(gamma_v_dist) == 0:
        print("  No valid observer placements! Cannot compute statistics.", flush=True)
        return

    gamma_arr = np.array(gamma_v_dist)
    gamma_low_arr = np.array(gamma_v_lowz_dist)
    gamma_high_arr = np.array(gamma_v_highz_dist)

    nbody_mean = np.mean(gamma_arr)
    nbody_std = np.std(gamma_arr)
    nbody_median = np.median(gamma_arr)

    tension = abs(obs_gamma_v - nbody_mean) / nbody_std if nbody_std > 0 else float('inf')

    print(f"\n  N-body gamma_v distribution ({len(gamma_arr)} observers):", flush=True)
    print(f"    Mean:   {nbody_mean:.2f} +/- {nbody_std:.2f} km/s", flush=True)
    print(f"    Median: {nbody_median:.2f} km/s", flush=True)
    print(f"    Range:  [{gamma_arr.min():.2f}, {gamma_arr.max():.2f}]", flush=True)
    print(f"\n  Observed CF4: {obs_gamma_v:.2f} km/s", flush=True)
    print(f"  Linear theory mock: {mock1_mean:.2f} +/- {mock1_std:.2f} km/s", flush=True)
    print(f"\n  *** TENSION: {tension:.1f} sigma ***", flush=True)

    print(f"\n  Sign check:", flush=True)
    n_negative = (gamma_arr < 0).sum()
    print(f"    N-body observers with gamma_v < 0: {n_negative}/{len(gamma_arr)}",
          flush=True)
    print(f"    Observed gamma_v sign: {'negative' if obs_gamma_v < 0 else 'positive'}",
          flush=True)

    print(f"\n  Piecewise:", flush=True)
    print(f"    z < {Z_CUT}: N-body = {np.mean(gamma_low_arr):.2f} +/- "
          f"{np.std(gamma_low_arr):.2f}, Observed = {obs_gamma_lowz:.2f}", flush=True)
    print(f"    z >= {Z_CUT}: N-body = {np.mean(gamma_high_arr):.2f} +/- "
          f"{np.std(gamma_high_arr):.2f}, Observed = {obs_gamma_highz:.2f}", flush=True)

    # Verdict
    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    if tension > 5:
        print(f"  N-body LCDM mock: {tension:.1f} sigma tension with observed CF4.",
              flush=True)
        print(f"  Full non-linear N-body velocities CANNOT reproduce the signal.",
              flush=True)
        print(f"  This rules out the 'non-linear corrections might fix it' argument.",
              flush=True)
        if nbody_mean * obs_gamma_v < 0:  # opposite signs
            print(f"  SIGN MISMATCH: N-body predicts {'+' if nbody_mean > 0 else '-'}"
                  f", observed is {'+' if obs_gamma_v > 0 else '-'}.", flush=True)
            print(f"  No perturbative correction can flip the sign.", flush=True)
    elif tension > 3:
        print(f"  N-body mock: {tension:.1f} sigma tension (marginal).", flush=True)
    else:
        print(f"  N-body mock: {tension:.1f} sigma (consistent with LCDM).", flush=True)

    print("=" * 70, flush=True)

    # Step 6: Plots
    print("\n--- Generating plots ---", flush=True)
    plot_nbody_comparison(gamma_v_dist, obs_gamma_v, mock1_mean, mock1_std,
                           OUTPUT_DIR)
    plot_nbody_piecewise(results_list, obs_gamma_lowz, obs_gamma_highz, OUTPUT_DIR)

    # Step 7: Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'description': 'MDPL2 N-body mock test for CF4 gamma_v signal',
        'simulation': {
            'name': 'MDPL2',
            'box_size_mpc_h': BOX_SIZE,
            'omega_m': OMEGA_M,
            'h': 0.678,
            'sigma8': 0.829,
            'n_s': 0.96,
            'snapshot': 125,
            'redshift': 0.0,
            'source': 'CosmoSim TAP API (cosmosim.org)',
        },
        'selection': {
            'mvir_min': MVIR_MIN,
            'n_halos_total': n_halos,
            'n_voids_total': len(void_r),
            'void_delta_threshold': DELTA_VOID,
            'void_r_min': R_VOID_MIN,
            'void_r_max': R_VOID_MAX,
        },
        'n_observers': N_OBSERVERS,
        'n_valid_observers': len(gamma_v_dist),
        'observed': {
            'gamma_v_full': obs_gamma_v,
            'gamma_v_lowz': obs_gamma_lowz,
            'gamma_v_highz': obs_gamma_highz,
        },
        'linear_theory_mock': {
            'gamma_v_mean': mock1_mean,
            'gamma_v_std': mock1_std,
        },
        'nbody_mock': {
            'gamma_v_mean': float(nbody_mean),
            'gamma_v_std': float(nbody_std),
            'gamma_v_median': float(nbody_median),
            'gamma_v_min': float(gamma_arr.min()),
            'gamma_v_max': float(gamma_arr.max()),
            'gamma_v_ci95': [float(np.percentile(gamma_arr, 2.5)),
                              float(np.percentile(gamma_arr, 97.5))],
            'tension_sigma': float(tension),
            'sign_mismatch': bool(nbody_mean * obs_gamma_v < 0),
            'n_negative': int(n_negative),
            'gamma_v_lowz_mean': float(np.mean(gamma_low_arr)),
            'gamma_v_lowz_std': float(np.std(gamma_low_arr)),
            'gamma_v_highz_mean': float(np.mean(gamma_high_arr)),
            'gamma_v_highz_std': float(np.std(gamma_high_arr)),
            'distribution': [float(g) for g in gamma_v_dist],
        },
        'per_observer': [
            {k: v for k, v in r.items()} for r in results_list
        ],
    }

    save_results(all_results, 'task3a_nbody_mock_results.json', OUTPUT_DIR)
    print(f"\nAll results saved to {OUTPUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()
