#!/usr/bin/env python3
"""
Build a merged low-z SN Ia sample from:
  1. Pantheon+ (existing)
  2. ZTF DR2 (new)
  3. Foundation DR1 (new)

Cross-match by sky position (< 5 arcsec) to remove duplicates.
Output: unified CSV with columns matching Pantheon+ conventions.

For the merged sample, we do NOT have a combined covariance matrix.
The hardening tests use a diagonal approximation (m_b_corr_err) for
the non-Pantheon+ SNe. This is acceptable for the hardening tests
(which test robustness, not precision cosmology), but should be
noted as a limitation.

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: April 2026
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""

import numpy as np
import pandas as pd
import os
import glob

BASE_DATA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'External')


def load_pantheon():
    """Load Pantheon+ with all columns we need."""
    path = os.path.join(BASE_DATA, 'pantheonplus', 'Pantheon+SH0ES.dat')
    data = np.genfromtxt(path, names=True, dtype=None, encoding='utf-8')

    df = pd.DataFrame({
        'name': data['CID'],
        'ra': data['RA'],
        'dec': data['DEC'],
        'z': data['zCMB'],
        'mB': data['m_b_corr'],
        'mB_err': data['m_b_corr_err_DIAG'],
        'x1': data['x1'],
        'x1_err': data['x1ERR'],
        'c': data['c'],
        'c_err': data['cERR'],
        'host_mass': data['HOST_LOGMASS'],
        'host_mass_err': data['HOST_LOGMASS_ERR'],
        'source': 'Pantheon+',
    })
    print(f"Pantheon+: {len(df)} SNe, {(df['z'] < 0.06).sum()} at z<0.06")
    return df


def load_ztf_dr2():
    """Load ZTF DR2 cosmology sample with Tripp-standardized magnitudes."""
    path = os.path.join(BASE_DATA, 'ztf_dr2', 'ztf_dr2_cosmology.csv')
    raw = pd.read_csv(path)

    # Raw SALT2 peak magnitude
    mB_raw = raw['mB']

    # Apply Tripp standardization: m_corr = mB + alpha * x1 - beta * c
    # Using Pantheon+ global best-fit values (Brout+2022)
    alpha_tripp = 0.148
    beta_tripp = 3.112
    m_b_corr = mB_raw + alpha_tripp * raw['x1'] - beta_tripp * raw['c']

    # Propagate errors (simplified, ignoring covariance terms)
    mB_err_raw = 2.5 / (np.log(10)) * raw['x0_err'] / np.abs(raw['x0'])
    m_b_corr_err = np.sqrt(
        mB_err_raw**2
        + (alpha_tripp * raw['x1_err'])**2
        + (beta_tripp * raw['c_err'])**2
    )

    df = pd.DataFrame({
        'name': raw['iau_name'].fillna(pd.Series(raw.index.astype(str), index=raw.index)),
        'ra': raw['ra'],
        'dec': raw['dec'],
        'z': raw['redshift'],
        'mB': m_b_corr,
        'mB_err': m_b_corr_err,
        'x1': raw['x1'],
        'x1_err': raw['x1_err'],
        'c': raw['c'],
        'c_err': raw['c_err'],
        'host_mass': raw['globalmass'],
        'host_mass_err': raw['globalmass_err'],
        'source': 'ZTF_DR2',
    })
    print(f"ZTF DR2: {len(df)} SNe, {(df['z'] < 0.06).sum()} at z<0.06 (Tripp-standardized)")
    return df


def load_foundation():
    """Load Foundation DR1 from FITRES + light curve headers for RA/DEC."""
    fitres_path = os.path.join(BASE_DATA, 'foundation_dr1',
                                'Foundation_DR1.FITRES.TEXT')
    lc_dir = os.path.join(BASE_DATA, 'foundation_dr1', 'Foundation_DR1')

    # Parse FITRES
    rows = []
    with open(fitres_path) as f:
        for line in f:
            if line.startswith('SN:'):
                parts = line.split()
                rows.append(parts[1:])  # Skip 'SN:'

    # Get column names from VARNAMES line
    with open(fitres_path) as f:
        for line in f:
            if line.startswith('VARNAMES:'):
                cols = line.split()[1:]  # Skip 'VARNAMES:'
                break

    fitres = pd.DataFrame(rows, columns=cols)
    for col in ['zCMB', 'x1', 'x1ERR', 'c', 'cERR', 'mB', 'mBERR',
                'HOST_LOGMASS', 'HOST_LOGMASS_ERR']:
        fitres[col] = pd.to_numeric(fitres[col], errors='coerce')

    # Get RA/DEC from light curve headers
    ra_dec = {}
    lc_files = glob.glob(os.path.join(lc_dir, 'Foundation_DR1_*.txt'))
    for lc_file in lc_files:
        sn_ra, sn_dec, sn_id = None, None, None
        with open(lc_file) as f:
            for line in f:
                if line.startswith('SNID:'):
                    sn_id = line.split()[1].strip()
                elif line.startswith('RA:'):
                    sn_ra = float(line.split()[1])
                elif line.startswith('DECL:'):
                    sn_dec = float(line.split()[1])
                elif line.startswith('NOBS:'):
                    break
        if sn_id and sn_ra is not None and sn_dec is not None:
            ra_dec[sn_id] = (sn_ra, sn_dec)

    # Merge
    fitres['ra'] = fitres['CID'].map(lambda x: ra_dec.get(x, (np.nan, np.nan))[0])
    fitres['dec'] = fitres['CID'].map(lambda x: ra_dec.get(x, (np.nan, np.nan))[1])

    df = pd.DataFrame({
        'name': fitres['CID'],
        'ra': fitres['ra'],
        'dec': fitres['dec'],
        'z': fitres['zCMB'],
        'mB': fitres['mB'],
        'mB_err': fitres['mBERR'],
        'x1': fitres['x1'],
        'x1_err': fitres['x1ERR'],
        'c': fitres['c'],
        'c_err': fitres['cERR'],
        'host_mass': fitres['HOST_LOGMASS'],
        'host_mass_err': fitres['HOST_LOGMASS_ERR'],
        'source': 'Foundation',
    })
    df = df.dropna(subset=['ra', 'dec', 'z', 'mB'])
    print(f"Foundation: {len(df)} SNe, {(df['z'] < 0.06).sum()} at z<0.06")
    return df


def cross_match_remove_duplicates(df, match_radius_arcsec=5.0):
    """
    Remove duplicates by sky position match.
    Priority: Pantheon+ > ZTF DR2 > Foundation
    (Pantheon+ has the best covariance; keep it when there's overlap)
    """
    match_radius_deg = match_radius_arcsec / 3600.0

    # Sort by priority
    priority = {'Pantheon+': 0, 'ZTF_DR2': 1, 'Foundation': 2}
    df = df.sort_values('source', key=lambda x: x.map(priority)).reset_index(drop=True)

    keep = np.ones(len(df), dtype=bool)
    ra = df['ra'].values
    dec = df['dec'].values

    for i in range(len(df)):
        if not keep[i]:
            continue
        # Check all subsequent SNe
        cos_dec = np.cos(np.radians(dec[i]))
        dra = (ra[i+1:] - ra[i]) * cos_dec
        ddec = dec[i+1:] - dec[i]
        sep = np.sqrt(dra**2 + ddec**2)

        matches = np.where(sep < match_radius_deg)[0] + i + 1
        for j in matches:
            if keep[j]:
                keep[j] = False  # Remove lower-priority duplicate

    n_removed = (~keep).sum()
    df_clean = df[keep].reset_index(drop=True)
    print(f"\nCross-match: removed {n_removed} duplicates (radius={match_radius_arcsec}\")")
    return df_clean


def build_diagonal_covariance(df, sigma_int=0.12):
    """
    Build a diagonal covariance matrix from mB_err + intrinsic scatter.

    ZTF/Foundation mB_err values are SALT2-fit-only (no systematics),
    while Pantheon+ diagonal errors include partial systematics.
    Adding sigma_int ~ 0.12 mag (standard SN Ia intrinsic scatter)
    prevents over-weighting the ZTF sample in the GLS fit.

    Total variance per SN: sigma_tot^2 = mB_err^2 + sigma_int^2
    """
    n = len(df)
    sigma_tot = np.sqrt(df['mB_err'].values**2 + sigma_int**2)
    cov = np.diag(sigma_tot**2)
    return cov


def main():
    print("=" * 70)
    print("Building Merged Low-z SN Ia Sample")
    print("=" * 70)

    # Load all three datasets
    p = load_pantheon()
    z = load_ztf_dr2()
    f = load_foundation()

    # Combine
    merged = pd.concat([p, z, f], ignore_index=True)
    print(f"\nBefore dedup: {len(merged)} total")

    # Drop rows with NaN in critical columns
    before = len(merged)
    merged = merged.dropna(subset=['mB', 'mB_err', 'z', 'ra', 'dec']).reset_index(drop=True)
    print(f"Dropped {before - len(merged)} rows with NaN in critical columns")

    # Remove duplicates
    merged = cross_match_remove_duplicates(merged)

    # Summary
    print(f"\nFinal merged sample: {len(merged)} unique SNe Ia")
    print(f"  Pantheon+:  {(merged['source'] == 'Pantheon+').sum()}")
    print(f"  ZTF DR2:    {(merged['source'] == 'ZTF_DR2').sum()}")
    print(f"  Foundation: {(merged['source'] == 'Foundation').sum()}")
    print(f"\n  z < 0.06: {(merged['z'] < 0.06).sum()}")
    print(f"  z < 0.05: {(merged['z'] < 0.05).sum()}")
    print(f"  z < 0.04: {(merged['z'] < 0.04).sum()}")
    print(f"  z < 0.157 (DESIVAST limit): {(merged['z'] < 0.157).sum()}")
    print(f"\n  Host mass coverage: {merged['host_mass'].notna().sum()}/{len(merged)}")

    # Save
    out_dir = os.path.join(BASE_DATA, 'merged_low_z')
    os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(os.path.join(out_dir, 'merged_sn_ia.csv'), index=False)

    # Build and save diagonal covariance
    cov = build_diagonal_covariance(merged)
    np.save(os.path.join(out_dir, 'merged_cov_diag.npy'), cov)

    print(f"\nSaved to: {out_dir}/")
    print(f"  merged_sn_ia.csv ({len(merged)} rows)")
    print(f"  merged_cov_diag.npy ({cov.shape})")

    return merged


if __name__ == '__main__':
    main()
