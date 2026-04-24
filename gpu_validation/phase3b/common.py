# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 3b shared data loading and NGC/SGC infrastructure.

Loads all Phase 3 data products once, prepares NGC/SGC splits,
and provides utility functions used by all five tests.
"""

import hashlib
import json
import numpy as np
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from scipy import linalg, stats

from mtdf_validation.phase3.data_loader import (
    PantheonPlusData, load_all_void_catalogs, sn_to_comoving,
    combine_ngc_sgc_voids, COSMO_VOIDS, COSMO_SN,
)
from mtdf_validation.phase3.crossmatch_gpu import compute_environment_gpu
from mtdf_validation.phase3.gls_engine import (
    gls_fit, delta_chi2_test, delta_chi2_test_with_survey_fe, SURVEY_NAMES,
)

# Match Phase 3 exactly
Z_MIN = 0.02
Z_MAX = 0.157
FINDERS = ['voidfinder', 'revolver', 'vide']


@dataclass
class Phase3bData:
    """Container for all pre-computed data shared across tests."""
    # Raw Pantheon+ object
    pantheon: object
    catalogs: dict
    idx: np.ndarray
    cov_sub: np.ndarray
    cov_inv: np.ndarray
    sub: dict  # z, mu, ra, dec, host_mass, survey_id

    # Comoving positions
    sn_pos: np.ndarray  # (N, 3)

    # NGC/SGC masks (boolean, length N)
    ngc_mask: np.ndarray
    sgc_mask: np.ndarray
    ngc_idx: np.ndarray  # integer indices into sub arrays
    sgc_idx: np.ndarray

    # Per-finder environment metrics (computed from combined NGC+SGC voids)
    d_signed_combined: dict = field(default_factory=dict)
    # Per-finder per-region environment metrics (region-specific voids)
    d_signed_region: dict = field(default_factory=dict)

    # Per-finder per-region void metadata for Test B
    void_data: dict = field(default_factory=dict)

    # Regional sub-covariances
    cov_ngc: np.ndarray = None
    cov_sgc: np.ndarray = None
    cov_inv_ngc: np.ndarray = None
    cov_inv_sgc: np.ndarray = None

    # Cholesky factors for simulation
    L_sub: np.ndarray = None
    L_ngc: np.ndarray = None
    L_sgc: np.ndarray = None

    # Phase 3 results
    phase3_results: dict = field(default_factory=dict)

    # Config
    config: dict = field(default_factory=dict)


def load_config(config_path):
    """Load YAML configuration, falling back to defaults."""
    if config_path is None:
        return default_config()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Merge with defaults for any missing keys
    defaults = default_config()
    for key in defaults:
        if key not in config:
            config[key] = defaults[key]
    return config


def default_config():
    """Return default configuration."""
    return {
        'data': {
            'data_dir': None,
            'z_min': 0.02,
            'z_max': 0.157,
            'finders': ['voidfinder', 'revolver', 'vide'],
        },
        'ngc_sgc': {'ngc_ra_min': 90.0, 'ngc_ra_max': 280.0},
        'test_a': {
            'enabled': True,
            'ipw_percentile_clip': [1, 99],
            'covariates': ['z', 'host_mass', 'survey_id'],
            'min_survey_count': 10,
        },
        'test_b': {
            'enabled': True,
            'z_bin_edges': [0.02, 0.04, 0.06, 0.10, 0.157],
            'area_ngc_deg2': 7500.0,
            'area_sgc_deg2': 2500.0,
        },
        'test_c': {'enabled': True, 'n_bootstrap': 1000, 'seed': 42},
        'test_d': {'enabled': True, 'n_mocks': 200, 'seed': 123},
        'test_e': {
            'enabled': True,
            'n_realizations': 200,
            'injection_strengths': [0.5, 1.0, 1.5],
            'seed': 456,
        },
        'output': {'save_plots': True, 'plot_formats': ['png', 'pdf'], 'dpi': 150},
    }


def split_ngc_sgc(ra, ngc_ra_min=90.0, ngc_ra_max=280.0):
    """Split SNe by RA into NGC and SGC. Matches Phase 3 run_phase3.py line 123."""
    ngc_mask = (ra > ngc_ra_min) & (ra < ngc_ra_max)
    return ngc_mask, ~ngc_mask


def safe_cholesky(cov, label=""):
    """Cholesky decomposition with Tikhonov regularization fallback."""
    diag_mean = np.mean(np.diag(cov))
    for eps_exp in range(-12, -5):
        eps = 10.0 ** eps_exp * diag_mean
        try:
            L = linalg.cholesky(cov + eps * np.eye(len(cov)), lower=True)
            if eps_exp > -12:
                print(f"  [cholesky {label}] regularized with eps=1e{eps_exp} * diag_mean")
            return L
        except linalg.LinAlgError:
            continue
    raise linalg.LinAlgError(f"Cholesky failed for {label} even with eps=1e-6")


def delta_gamma_with_se(result_ngc, result_sgc):
    """Compute Δγ = γ_NGC − γ_SGC with combined SE (independence assumed)."""
    dg = result_ngc['gamma_env'] - result_sgc['gamma_env']
    se = np.sqrt(result_ngc['gamma_env_err'] ** 2 + result_sgc['gamma_env_err'] ** 2)
    z_score = dg / se if se > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
    return {
        'delta_gamma': float(dg),
        'delta_se': float(se),
        'z_score': float(z_score),
        'p_value': float(p_value),
    }


def make_output_dir(base_results_dir):
    """Create timestamped output directory."""
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = Path(base_results_dir) / f'phase3b_{ts}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plots').mkdir(exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    return out


def compute_file_hashes(data_dir):
    """SHA-256 of key data files for reproducibility."""
    hashes = {}
    data_dir = Path(data_dir)
    files_to_hash = [
        data_dir / "External" / "pantheonplus" / "Pantheon+SH0ES.dat",
        data_dir / "External" / "pantheonplus" / "Pantheon+SH0ES_STAT+SYS.cov",
    ]
    for f in files_to_hash:
        if f.exists():
            h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            hashes[f.name] = h
    return hashes


def load_phase3b_data(config):
    """Load all data needed for Phase 3b tests. Called once at startup."""
    data_cfg = config['data']
    ngc_cfg = config['ngc_sgc']

    # Resolve data directory
    data_dir = data_cfg.get('data_dir')
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent.parent
                       / "validation" / "data")

    z_min = data_cfg.get('z_min', Z_MIN)
    z_max = data_cfg.get('z_max', Z_MAX)
    finders = data_cfg.get('finders', FINDERS)

    # Load Pantheon+
    print("[Phase3b] Loading Pantheon+ data...")
    pantheon = PantheonPlusData(data_dir)
    idx, cov_sub = pantheon.apply_cuts(z_min, z_max)
    sub = pantheon.get_subset(idx)
    n = len(idx)

    print(f"[Phase3b] Inverting {n}x{n} covariance...")
    cov_inv = linalg.inv(cov_sub)

    # Comoving positions
    sn_pos = sn_to_comoving(sub['z'], sub['ra'], sub['dec'])

    # NGC/SGC split
    ngc_mask, sgc_mask = split_ngc_sgc(
        sub['ra'], ngc_cfg['ngc_ra_min'], ngc_cfg['ngc_ra_max']
    )
    ngc_idx = np.where(ngc_mask)[0]
    sgc_idx = np.where(sgc_mask)[0]
    print(f"[Phase3b] NGC: {len(ngc_idx)} SNe, SGC: {len(sgc_idx)} SNe")

    # Regional sub-covariances
    cov_ngc = cov_sub[np.ix_(ngc_idx, ngc_idx)]
    cov_sgc = cov_sub[np.ix_(sgc_idx, sgc_idx)]
    cov_inv_ngc = linalg.inv(cov_ngc)
    cov_inv_sgc = linalg.inv(cov_sgc)

    # Load void catalogs
    print("[Phase3b] Loading void catalogs...")
    catalogs = load_all_void_catalogs(data_dir)

    # Compute d_signed for all finders
    d_signed_combined = {}
    d_signed_region = {}
    void_data = {}

    for finder in finders:
        print(f"[Phase3b] Crossmatching {finder}...")

        # Combined NGC+SGC voids (matches Phase 3 Table 1)
        void_pos, void_r, n_ngc_voids = combine_ngc_sgc_voids(catalogs, finder)
        d_signed, nearest_idx, in_void = compute_environment_gpu(sn_pos, void_pos, void_r)
        d_signed_combined[finder] = d_signed

        # Store void data for Test B
        void_data[(finder, 'combined')] = {
            'pos': void_pos, 'r': void_r, 'n_voids': len(void_r),
            'n_ngc_voids': n_ngc_voids,
        }

        # Per-region d_signed (matches Phase 3 Table 2)
        for region, sn_mask in [('ngc', ngc_mask), ('sgc', sgc_mask)]:
            cat = catalogs.get((finder, region.upper()))
            if cat is None:
                continue
            vx, vy, vz, vr = cat.get_positions(interior_only=True)
            vpos = np.column_stack([vx, vy, vz])
            d_reg, _, _ = compute_environment_gpu(sn_pos[sn_mask], vpos, vr)
            d_signed_region[(finder, region)] = d_reg
            void_data[(finder, region)] = {
                'pos': vpos, 'r': vr, 'n_voids': len(vr),
            }

    # Cholesky factors for simulation (Tests C, E)
    print("[Phase3b] Computing Cholesky factors...")
    L_sub = safe_cholesky(cov_sub, "cov_sub")
    L_ngc = safe_cholesky(cov_ngc, "cov_ngc")
    L_sgc = safe_cholesky(cov_sgc, "cov_sgc")

    # Load Phase 3 results
    phase3_json = (Path(__file__).resolve().parent.parent
                   / "results" / "phase3" / "phase3_summary.json")
    phase3_results = {}
    if phase3_json.exists():
        with open(phase3_json) as f:
            phase3_results = json.load(f)
        print(f"[Phase3b] Loaded Phase 3 results from {phase3_json.name}")

    return Phase3bData(
        pantheon=pantheon,
        catalogs=catalogs,
        idx=idx,
        cov_sub=cov_sub,
        cov_inv=cov_inv,
        sub=sub,
        sn_pos=sn_pos,
        ngc_mask=ngc_mask,
        sgc_mask=sgc_mask,
        ngc_idx=ngc_idx,
        sgc_idx=sgc_idx,
        d_signed_combined=d_signed_combined,
        d_signed_region=d_signed_region,
        void_data=void_data,
        cov_ngc=cov_ngc,
        cov_sgc=cov_sgc,
        cov_inv_ngc=cov_inv_ngc,
        cov_inv_sgc=cov_inv_sgc,
        L_sub=L_sub,
        L_ngc=L_ngc,
        L_sgc=L_sgc,
        phase3_results=phase3_results,
        config=config,
    )


def region_gls(data, region, finder, with_fe=False):
    """Run GLS delta-chi2 test for a specific region and finder.

    Args:
        data: Phase3bData
        region: 'ngc', 'sgc', or 'combined'
        finder: void finder name
        with_fe: if True, include survey fixed effects

    Returns:
        dict from delta_chi2_test or delta_chi2_test_with_survey_fe
    """
    if region == 'combined':
        mu = data.sub['mu']
        z = data.sub['z']
        hm = data.sub['host_mass']
        sid = data.sub['survey_id']
        env = data.d_signed_combined[finder]
        ci = data.cov_inv
    elif region == 'ngc':
        mask = data.ngc_mask
        mu = data.sub['mu'][mask]
        z = data.sub['z'][mask]
        hm = data.sub['host_mass'][mask]
        sid = data.sub['survey_id'][mask]
        env = data.d_signed_region[(finder, 'ngc')]
        ci = data.cov_inv_ngc
    elif region == 'sgc':
        mask = data.sgc_mask
        mu = data.sub['mu'][mask]
        z = data.sub['z'][mask]
        hm = data.sub['host_mass'][mask]
        sid = data.sub['survey_id'][mask]
        env = data.d_signed_region[(finder, 'sgc')]
        ci = data.cov_inv_sgc
    else:
        raise ValueError(f"Unknown region: {region}")

    if with_fe:
        return delta_chi2_test_with_survey_fe(mu, z, env, hm, sid, ci)
    return delta_chi2_test(mu, z, env, hm, ci)


def json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
