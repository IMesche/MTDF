# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Independent dataset loaders for Phase 1 chi-squared reproduction.
Does NOT import from vector_pillars.py — all parsing written from scratch.
"""

import numpy as np
from pathlib import Path


def load_pantheonplus(data_dir):
    """
    Load Pantheon+ SH0ES SNe Ia data.
    Returns: z_cmb (1701,), mu_obs (1701,), cov (1701, 1701)
    """
    data_path = Path(data_dir) / "External" / "pantheonplus" / "Pantheon+SH0ES.dat"
    cov_path = Path(data_dir) / "External" / "pantheonplus" / "Pantheon+SH0ES_STAT+SYS.cov"

    # Parse data
    z_list, mu_list = [], []
    with open(data_path, 'r') as f:
        header = f.readline().strip().split()
        z_col = header.index('zCMB') if 'zCMB' in header else header.index('zHD')
        mu_col = next(header.index(c) for c in ['MU_SH0ES', 'm_b_corr', 'mu'] if c in header)

        for line in f:
            parts = line.strip().split()
            if len(parts) > max(z_col, mu_col):
                try:
                    z = float(parts[z_col])
                    mu = float(parts[mu_col])
                    if z > 0 and not np.isnan(mu):
                        z_list.append(z)
                        mu_list.append(mu)
                except (ValueError, IndexError):
                    continue

    z_cmb = np.array(z_list)
    mu_obs = np.array(mu_list)
    n = len(z_cmb)

    # Parse covariance (one value per line, first line is dimension)
    cov_values = []
    with open(cov_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                for v in line.split():
                    cov_values.append(float(v))

    # First value might be the dimension
    dim = int(cov_values[0])
    cov_values = cov_values[1:]

    total = len(cov_values)
    if total == dim * dim:
        cov = np.array(cov_values).reshape(dim, dim)
    elif total == n * n:
        cov = np.array(cov_values).reshape(n, n)
    else:
        actual_n = int(np.sqrt(total))
        cov = np.array(cov_values).reshape(actual_n, actual_n)
        z_cmb = z_cmb[:actual_n]
        mu_obs = mu_obs[:actual_n]

    return z_cmb, mu_obs, cov


def load_desi_bao(data_dir):
    """
    Load DESI Y1 BAO data.
    Returns: z_eff (12,), obs (12,), types (12,), cov (12, 12)
    """
    mean_path = Path(data_dir) / "External" / "bao_desi" / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_path = Path(data_dir) / "External" / "bao_desi" / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"

    z_eff, obs, types = [], [], []
    with open(mean_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                z_eff.append(float(parts[0]))
                obs.append(float(parts[1]))
                types.append(parts[2])

    z_eff = np.array(z_eff)
    obs = np.array(obs)

    # Load covariance
    cov_rows = []
    with open(cov_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                cov_rows.append([float(x) for x in line.split()])
    cov = np.array(cov_rows)

    return z_eff, obs, types, cov


def load_cc_hz(data_dir):
    """
    Load Cosmic Chronometer H(z) data (BC03 model).
    Returns: z, H_obs, cov (diagonal)
    """
    data_path = Path(data_dir) / "External" / "hz_cc" / "HzTable_MM_BC03.dat"

    z_list, H_list, err_list = [], [], []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 3:
                try:
                    z_list.append(float(parts[0]))
                    H_list.append(float(parts[1]))
                    err_list.append(float(parts[2]))
                except ValueError:
                    continue

    z = np.array(z_list)
    H_obs = np.array(H_list)
    H_err = np.array(err_list)
    cov = np.diag(H_err**2)

    return z, H_obs, cov


def load_dr16_fsigma8(data_dir):
    """
    Load DR16 fsigma8 data (LRG + QSO), extracting only fsigma8 rows.
    Returns: z_eff (4,), fsig8_obs (4,), cov (4, 4) block-diagonal
    """
    base = Path(data_dir) / "External" / "growth_fsig8"
    lrg_path = base / "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat"
    qso_path = base / "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat"
    lrg_cov_path = base / "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt"
    qso_cov_path = base / "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt"

    def parse_dat(path):
        z, val, typ = [], [], []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    z.append(float(parts[0]))
                    val.append(float(parts[1]))
                    typ.append(parts[2])
        return z, val, typ

    def load_cov(path):
        rows = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    rows.append([float(x) for x in line.split()])
        return np.array(rows)

    # LRG: 9 rows (3 z-bins x [DM, DH, fsig8]), fsig8 at indices with 'f_sigma8'
    z_lrg, val_lrg, types_lrg = parse_dat(lrg_path)
    lrg_fsig_idx = [i for i, t in enumerate(types_lrg) if 'f_sigma8' in t]

    # QSO: 3 rows [DM, DH, fsig8], fsig8 at index with 'f_sigma8'
    z_qso, val_qso, types_qso = parse_dat(qso_path)
    qso_fsig_idx = [i for i, t in enumerate(types_qso) if 'f_sigma8' in t]

    z_eff = [z_lrg[i] for i in lrg_fsig_idx] + [z_qso[i] for i in qso_fsig_idx]
    fsig8_obs = [val_lrg[i] for i in lrg_fsig_idx] + [val_qso[i] for i in qso_fsig_idx]

    z_eff = np.array(z_eff)
    fsig8_obs = np.array(fsig8_obs)
    n_lrg = len(lrg_fsig_idx)
    n_qso = len(qso_fsig_idx)
    n_total = n_lrg + n_qso

    # Build block-diagonal covariance from fsig8 sub-blocks
    cov = np.zeros((n_total, n_total))

    lrg_cov_full = load_cov(lrg_cov_path)
    for i, ii in enumerate(lrg_fsig_idx):
        for j, jj in enumerate(lrg_fsig_idx):
            if ii < lrg_cov_full.shape[0] and jj < lrg_cov_full.shape[1]:
                cov[i, j] = lrg_cov_full[ii, jj]

    qso_cov_full = load_cov(qso_cov_path)
    for i, ii in enumerate(qso_fsig_idx):
        for j, jj in enumerate(qso_fsig_idx):
            if ii < qso_cov_full.shape[0] and jj < qso_cov_full.shape[1]:
                cov[n_lrg + i, n_lrg + j] = qso_cov_full[ii, jj]

    return z_eff, fsig8_obs, cov


def load_cmb_distance_prior(data_dir):
    """
    Load Planck 2018 CMB distance prior.
    Returns: means [R, lA, omegab_h2] (3,), cov (3, 3)
    """
    base = Path(data_dir) / "External" / "cmb_planck2018"
    means_path = base / "planck2018_distance_means.txt"
    cov_path = base / "planck2018_distance_cov.txt"

    means = []
    with open(means_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                means.append(float(parts[1]))
    means = np.array(means)

    cov_rows = []
    with open(cov_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            cov_rows.append([float(x) for x in line.split()])
    cov = np.array(cov_rows)

    return means, cov
