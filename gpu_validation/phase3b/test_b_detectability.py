# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Test B: Void-SN Joint Density Mapping (Detectability).

Quantifies whether the SGC null result is driven by reduced
void/SN density, by computing per-redshift-bin density diagnostics.
"""

import numpy as np
from .common import COSMO_VOIDS, FINDERS


def comoving_volume_shell(z_lo, z_hi, area_deg2):
    """Comoving volume of a redshift shell in (Mpc/h)^3."""
    area_sr = area_deg2 * (np.pi / 180.0) ** 2
    d_lo = COSMO_VOIDS.comoving_distance(z_lo).value  # Mpc/h
    d_hi = COSMO_VOIDS.comoving_distance(z_hi).value
    return (area_sr / (4 * np.pi)) * (4.0 / 3.0) * np.pi * (d_hi ** 3 - d_lo ** 3)


def void_positions_to_redshift(pos):
    """Convert void Cartesian positions to redshifts via distance inversion."""
    d_c = np.sqrt(np.sum(pos ** 2, axis=1))

    # Build lookup table
    z_grid = np.linspace(0.001, 0.25, 20000)
    d_grid = COSMO_VOIDS.comoving_distance(z_grid).value
    return np.interp(d_c, d_grid, z_grid)


def build_detectability_table(data, config_b):
    """Build detectability table for all finders, regions, z-bins."""
    z_edges = config_b.get('z_bin_edges', [0.02, 0.04, 0.06, 0.10, 0.157])
    area_ngc = config_b.get('area_ngc_deg2', 7500.0)
    area_sgc = config_b.get('area_sgc_deg2', 2500.0)
    finders = data.config.get('data', {}).get('finders', FINDERS)

    sn_z = data.sub['z']
    rows = []

    for finder in finders:
        for region, sn_mask, area in [
            ('NGC', data.ngc_mask, area_ngc),
            ('SGC', data.sgc_mask, area_sgc),
        ]:
            # Get void data for this region
            vd = data.void_data.get((finder, region.lower()))
            if vd is None:
                continue

            void_z = void_positions_to_redshift(vd['pos'])
            void_r = vd['r']
            region_sn_z = sn_z[sn_mask]

            for i in range(len(z_edges) - 1):
                z_lo, z_hi = z_edges[i], z_edges[i + 1]

                # Volume
                vol = comoving_volume_shell(z_lo, z_hi, area)

                # Voids in this z-bin
                void_mask = (void_z >= z_lo) & (void_z < z_hi)
                n_void = int(np.sum(void_mask))
                r_voids = void_r[void_mask]
                median_r = float(np.median(r_voids)) if n_void > 0 else 0.0
                total_void_vol = float(np.sum((4.0 / 3.0) * np.pi * r_voids ** 3))

                # SNe in this z-bin
                sn_mask_z = (region_sn_z >= z_lo) & (region_sn_z < z_hi)
                n_sn = int(np.sum(sn_mask_z))

                # Densities
                void_density = n_void / vol if vol > 0 else 0.0
                sn_density = n_sn / vol if vol > 0 else 0.0
                sn_per_void = n_sn / n_void if n_void > 0 else 0.0
                filling_fraction = total_void_vol / vol if vol > 0 else 0.0

                rows.append({
                    'finder': finder,
                    'region': region,
                    'z_lo': z_lo,
                    'z_hi': z_hi,
                    'z_mid': (z_lo + z_hi) / 2,
                    'volume_Mpch3': float(vol),
                    'n_void': n_void,
                    'n_sn': n_sn,
                    'void_density': float(void_density),
                    'sn_density': float(sn_density),
                    'sn_per_void': float(sn_per_void),
                    'median_r_void': median_r,
                    'filling_fraction': float(filling_fraction),
                })

    return rows


def write_detectability_csv(table, path):
    """Write detectability table as CSV."""
    import csv
    if not table:
        return
    fieldnames = list(table[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


def write_detectability_markdown(table, path):
    """Write detectability table as markdown."""
    if not table:
        return

    lines = ["# Detectability Table (Test B)\n"]
    lines.append("| Finder | Region | z range | Volume (Mpc/h)^3 | N_void | N_SN | "
                 "SN/void | Med R | Filling |\n")
    lines.append("|--------|--------|---------|------------------|--------|------|"
                 "---------|-------|----------|\n")

    for r in table:
        lines.append(
            f"| {r['finder']} | {r['region']} | "
            f"{r['z_lo']:.2f}-{r['z_hi']:.2f} | "
            f"{r['volume_Mpch3']:.2e} | "
            f"{r['n_void']} | {r['n_sn']} | "
            f"{r['sn_per_void']:.2f} | "
            f"{r['median_r_void']:.1f} | "
            f"{r['filling_fraction']:.4f} |\n"
        )

    with open(path, 'w') as f:
        f.writelines(lines)


def run_test_b(data):
    """Execute Test B: build detectability table and summary."""
    config_b = data.config.get('test_b', {})

    print("\n  Building detectability table...")
    table = build_detectability_table(data, config_b)

    # Summary per region per finder
    summary = {}
    finders = data.config.get('data', {}).get('finders', FINDERS)

    for finder in finders:
        for region in ['NGC', 'SGC']:
            region_rows = [r for r in table
                           if r['finder'] == finder and r['region'] == region]
            total_voids = sum(r['n_void'] for r in region_rows)
            total_sn = sum(r['n_sn'] for r in region_rows)
            total_vol = sum(r['volume_Mpch3'] for r in region_rows)

            key = f"{finder}_{region}"
            summary[key] = {
                'total_voids': total_voids,
                'total_sn': total_sn,
                'total_volume': total_vol,
                'overall_void_density': total_voids / total_vol if total_vol > 0 else 0,
                'overall_sn_per_void': total_sn / total_voids if total_voids > 0 else 0,
            }

            print(f"  {finder} {region}: {total_voids} voids, {total_sn} SNe, "
                  f"SN/void = {total_sn / total_voids:.2f}" if total_voids > 0 else
                  f"  {finder} {region}: {total_voids} voids, {total_sn} SNe")

    return {'table': table, 'summary': summary}
