#!/usr/bin/env python3
"""
Generate real data JSON for MTDF Local Universe Field Map.

Loads CF4 groups, VoidFinder voids, and 2M++ velocity field to produce
a JSON blob that replaces the synthetic DATA in the HTML visualization.
"""

import numpy as np
import json
import os
import sys
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

COSMO = FlatLambdaCDM(H0=100, Om0=0.315)  # Mpc/h for DESIVAST

BASE_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'External')
CF4_PATH = os.path.join(BASE_DATA, 'cosmicflows4', 'cf4_groups.csv')
VOID_DIR = os.path.join(BASE_DATA, 'desivast_voids')
TWOMPP_DIR = os.path.join(BASE_DATA, '2mpp')

N_SUBSAMPLE = 0  # 0 = use all groups (GPU can handle it)


def load_cf4():
    """Load CF4 groups, return structured array."""
    import csv
    groups = []
    with open(CF4_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ra = float(row['RAJ2000'])
                dec = float(row['DEJ2000'])
                vcmb = float(row['Vcmb'])
                dist = float(row['Dist'])
                vpec = float(row['Vpec'])
                dm = float(row['DMav'])
                e_dm = float(row['e_DMav'])
            except (ValueError, KeyError):
                continue
            if dist <= 0 or e_dm <= 0:
                continue
            z = vcmb / 299792.458
            if z < 0.001 or z > 0.05:
                continue
            groups.append({
                'ra': ra, 'dec': dec, 'vcmb': vcmb, 'dist': dist,
                'vpec': vpec, 'dm': dm, 'e_dm': e_dm, 'z': z,
            })
    print(f"  Loaded {len(groups)} CF4 groups", flush=True)
    return groups


def groups_to_cartesian(groups):
    """Convert RA/Dec/distance to supergalactic Cartesian Mpc/h."""
    coords = SkyCoord(
        ra=[g['ra'] for g in groups] * u.deg,
        dec=[g['dec'] for g in groups] * u.deg,
        distance=[g['dist'] for g in groups] * u.Mpc,
    )
    # Convert to supergalactic cartesian
    sgc = coords.supergalactic
    x = sgc.cartesian.x.value  # Mpc
    y = sgc.cartesian.y.value
    z = sgc.cartesian.z.value
    return x, y, z


def load_voidfinder():
    """Load VoidFinder NGC+SGC voids."""
    voids_x, voids_y, voids_z, voids_r = [], [], [], []
    for cap in ['NGC', 'SGC']:
        path = os.path.join(VOID_DIR,
                            f'DESIVAST_BGS_VOLLIM_VoidFinder_{cap}.fits')
        with fits.open(path) as h:
            data = h[1].data
            voids_x.extend(data['X'])
            voids_y.extend(data['Y'])
            voids_z.extend(data['Z'])
            voids_r.extend(data['RADIUS'])
    voids_x = np.array(voids_x)
    voids_y = np.array(voids_y)
    voids_z = np.array(voids_z)
    voids_r = np.array(voids_r)
    print(f"  Loaded {len(voids_x)} VoidFinder voids (NGC+SGC)", flush=True)
    return voids_x, voids_y, voids_z, voids_r


def compute_d_signed(gx, gy, gz, vx, vy, vz, vr):
    """Compute signed distance to nearest void boundary for each group."""
    d_signed = np.full(len(gx), np.inf)
    for i in range(len(gx)):
        dx = gx[i] - vx
        dy = gy[i] - vy
        dz = gz[i] - vz
        dist_to_center = np.sqrt(dx**2 + dy**2 + dz**2)
        # Signed distance: (dist_to_center - radius) / radius
        # Negative = inside void, positive = outside
        signed = (dist_to_center - vr) / vr
        nearest_idx = np.argmin(np.abs(signed))
        d_signed[i] = signed[nearest_idx]
    return d_signed


def load_2mpp_and_predict(groups, gx, gy, gz):
    """Load 2M++ velocity field and predict vpec at group positions."""
    vel_path = os.path.join(TWOMPP_DIR, 'twompp_velocity.npy')
    vel = np.load(vel_path)  # shape (3, 257, 257, 257) in Galactic Cartesian
    ngrid = 257
    box = 400.0  # Mpc/h
    dmax = 200.0  # Mpc/h, half-box

    # Build interpolator
    axis = np.linspace(-dmax, dmax, ngrid)
    interp_vx = RegularGridInterpolator((axis, axis, axis), vel[0],
                                        bounds_error=False, fill_value=np.nan)
    interp_vy = RegularGridInterpolator((axis, axis, axis), vel[1],
                                        bounds_error=False, fill_value=np.nan)
    interp_vz = RegularGridInterpolator((axis, axis, axis), vel[2],
                                        bounds_error=False, fill_value=np.nan)

    # Convert group positions to Galactic Cartesian for 2M++ lookup
    coords = SkyCoord(
        ra=[g['ra'] for g in groups] * u.deg,
        dec=[g['dec'] for g in groups] * u.deg,
        distance=[g['dist'] for g in groups] * u.Mpc,
    )
    gc = coords.galactic
    gl = gc.l.rad
    gb = gc.b.rad
    dist_mpc = np.array([g['dist'] for g in groups])

    # Convert to Galactic Cartesian Mpc/h (h=1 for 2M++)
    gal_x = dist_mpc * np.cos(gb) * np.cos(gl)
    gal_y = dist_mpc * np.cos(gb) * np.sin(gl)
    gal_z = dist_mpc * np.sin(gb)

    pts = np.column_stack([gal_x, gal_y, gal_z])
    vx_pred = interp_vx(pts)
    vy_pred = interp_vy(pts)
    vz_pred = interp_vz(pts)

    # Project onto line of sight (radial direction in Galactic)
    r_hat_x = np.cos(gb) * np.cos(gl)
    r_hat_y = np.cos(gb) * np.sin(gl)
    r_hat_z = np.sin(gb)

    v_los_pred = vx_pred * r_hat_x + vy_pred * r_hat_y + vz_pred * r_hat_z
    valid = np.isfinite(v_los_pred)

    print(f"  2M++ predictions: {valid.sum()}/{len(groups)} valid", flush=True)
    return v_los_pred, valid


def main():
    print("=== Generating MTDF Field Map data ===", flush=True)

    # 1. Load CF4 groups
    print("Step 1: Loading CF4 groups...", flush=True)
    groups = load_cf4()

    # 2. Convert to Cartesian
    print("Step 2: Converting to Cartesian...", flush=True)
    gx, gy, gz = groups_to_cartesian(groups)
    dist_mpc = np.array([g['dist'] for g in groups])

    # 3. Load VoidFinder voids
    print("Step 3: Loading VoidFinder voids...", flush=True)
    vx, vy, vz, vr = load_voidfinder()

    # 4. Compute d_signed
    print("Step 4: Computing d_signed (this takes a minute)...", flush=True)
    d_signed = compute_d_signed(gx, gy, gz, vx, vy, vz, vr)

    # 5. Load 2M++ and compute residuals
    print("Step 5: Loading 2M++ velocity field...", flush=True)
    v_los_pred, valid_2mpp = load_2mpp_and_predict(groups, gx, gy, gz)

    vpec_obs = np.array([g['vpec'] for g in groups], dtype=float)
    vpec_residual = np.where(valid_2mpp, vpec_obs - v_los_pred, np.nan)
    z_obs = np.array([g['z'] for g in groups])

    # 6. Subsample (0 = keep all, GPU can handle 18K points easily)
    if N_SUBSAMPLE > 0 and len(groups) > N_SUBSAMPLE:
        print(f"Step 6: Subsampling {len(groups)} -> {N_SUBSAMPLE}...", flush=True)
        np.random.seed(42)
        idx = np.random.choice(len(groups), N_SUBSAMPLE, replace=False)
        idx.sort()
    else:
        print(f"Step 6: Keeping all {len(groups)} groups (GPU mode)...", flush=True)
        idx = np.arange(len(groups))

    gx_s, gy_s, gz_s = gx[idx], gy[idx], gz[idx]
    dist_s = dist_mpc[idx]
    d_signed_s = d_signed[idx]
    vpec_obs_s = vpec_obs[idx]
    vpec_res_s = vpec_residual[idx]
    z_obs_s = z_obs[idx]
    in_void_s = (d_signed_s < 0).astype(int)

    # 7. Filter voids to within the CF4 volume
    max_dist = 250.0
    void_dist = np.sqrt(vx**2 + vy**2 + vz**2)
    v_mask = void_dist < max_dist
    vx_f, vy_f, vz_f, vr_f = vx[v_mask], vy[v_mask], vz[v_mask], vr[v_mask]
    print(f"  Voids within {max_dist} Mpc: {v_mask.sum()}", flush=True)

    # 8. Build JSON
    def to_list(arr):
        """Convert to list, replace NaN with null-safe 0."""
        out = []
        for v in arr:
            if np.isfinite(v):
                out.append(round(float(v), 2))
            else:
                out.append(0.0)
        return out

    data = {
        "metadata": {
            "description": "Real CF4/DESIVAST/2M++ data for MTDF field visualization",
            "note": "Cosmicflows-4 groups (Tully+2023), VoidFinder voids (DESIVAST), 2M++ residuals (Carrick+2015)",
            "gamma_v": -53.6,
            "beta_mpc": 22.7,
            "z_transition": 0.04,
            "transition_mpc": 120,
            "max_distance_mpc": 210,
            "S0": 1.084,
            "H0_local": 73.1,
            "H0_planck": 67.36,
            "n_groups_total": len(groups),
            "n_groups_displayed": len(idx),
            "n_voids_total": len(vx),
            "n_voids_displayed": int(v_mask.sum()),
        },
        "voids": {
            "x": to_list(vx_f),
            "y": to_list(vy_f),
            "z": to_list(vz_f),
            "radius": to_list(vr_f),
            "n": int(v_mask.sum()),
        },
        "groups": {
            "x": to_list(gx_s),
            "y": to_list(gy_s),
            "z": to_list(gz_s),
            "distance_mpc": to_list(dist_s),
            "d_signed": to_list(d_signed_s),
            "vpec_observed": to_list(vpec_obs_s),
            "vpec_residual": to_list(vpec_res_s),
            "z_observed": to_list(z_obs_s),
            "in_void": [int(v) for v in in_void_s],
            "n": len(idx),
        },
    }

    # Write JSON
    out_path = os.path.join(os.path.dirname(__file__), 'output', 'field_map_data.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nDone! Written to {out_path} ({size_kb:.0f} KB)", flush=True)
    print(f"  {data['groups']['n']} groups, {data['voids']['n']} voids", flush=True)

    # Stats
    ds = np.array(data['groups']['d_signed'])
    vp = np.array(data['groups']['vpec_observed'])
    print(f"  d_signed range: [{ds.min():.2f}, {ds.max():.2f}]", flush=True)
    print(f"  vpec range: [{vp.min():.0f}, {vp.max():.0f}] km/s", flush=True)
    print(f"  in_void: {sum(data['groups']['in_void'])}/{data['groups']['n']}", flush=True)


if __name__ == '__main__':
    main()
