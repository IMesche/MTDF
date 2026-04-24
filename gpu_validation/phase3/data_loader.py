# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 3 data loaders: Pantheon+ (extended) and DESIVAST void catalogs.

Coordinate convention: all XYZ in comoving Mpc/h using
FlatLambdaCDM(H0=100, Om0=0.315) — matching DESIVAST void catalogs.
"""

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from pathlib import Path

# Cosmologies — locked for the entire analysis
COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)  # Void coords in Mpc/h
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)         # SN baseline distmod

COSMOLOGY_HEADER = {
    "cosmo_xyz": "FlatLambdaCDM(H0=100, Om0=0.315)",
    "cosmo_sn_baseline": "FlatLambdaCDM(H0=70, Om0=0.3)",
    "unit": "Mpc/h (comoving)",
    "d_c_convention": "astropy comoving_distance().value with H0=100",
}


class PantheonPlusData:
    """Pantheon+ SH0ES data with full covariance and sky positions."""

    def __init__(self, data_dir):
        data_path = Path(data_dir) / "External" / "pantheonplus" / "Pantheon+SH0ES.dat"
        cov_path = Path(data_dir) / "External" / "pantheonplus" / "Pantheon+SH0ES_STAT+SYS.cov"

        print(f"[Phase3] Cosmology: {COSMOLOGY_HEADER['cosmo_xyz']}")
        print(f"[Phase3] SN baseline: {COSMOLOGY_HEADER['cosmo_sn_baseline']}")
        print(f"[Phase3] Units: {COSMOLOGY_HEADER['unit']}")

        # Parse data file
        self.raw = np.genfromtxt(data_path, names=True, dtype=None, encoding='utf-8')
        self.n = len(self.raw)

        self.z = self.raw['zCMB']
        self.mu_shoes = self.raw['MU_SH0ES']
        self.m_b_corr = self.raw['m_b_corr']
        self.host_mass = self.raw['HOST_LOGMASS']
        self.ra = self.raw['RA']
        self.dec = self.raw['DEC']
        self.survey_id = self.raw['IDSURVEY']

        # mu: use SH0ES calibrated where available, else m_b_corr + fiducial M_B
        self.mu = np.where(self.mu_shoes > 0, self.mu_shoes, self.m_b_corr + 19.25)

        # Load covariance
        print(f"  Loading {self.n}x{self.n} STAT+SYS covariance...")
        cov_values = []
        with open(cov_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    for v in line.split():
                        cov_values.append(float(v))

        dim = int(cov_values[0])
        cov_values = cov_values[1:]
        self.cov_full = np.array(cov_values).reshape(dim, dim)

        print(f"  Loaded {self.n} SNe, cov shape {self.cov_full.shape}")

    def apply_cuts(self, z_min=0.02, z_max=0.157):
        """Apply redshift cuts. Returns indices and sub-covariance."""
        mask = (self.z >= z_min) & (self.z <= z_max)
        idx = np.where(mask)[0]
        cov_sub = self.cov_full[np.ix_(idx, idx)]

        print(f"  Cuts z in [{z_min}, {z_max}]: {len(idx)} SNe remain")
        return idx, cov_sub

    def get_subset(self, idx):
        """Return all arrays for a subset of indices."""
        return {
            'z': self.z[idx],
            'mu': self.mu[idx],
            'host_mass': self.host_mass[idx],
            'ra': self.ra[idx],
            'dec': self.dec[idx],
            'survey_id': self.survey_id[idx],
        }


class VoidCatalog:
    """DESIVAST void catalog from FITS file."""

    def __init__(self, fits_path, catalog_type='voidfinder'):
        self.path = fits_path
        self.catalog_type = catalog_type
        self.region = 'NGC' if 'NGC' in str(fits_path) else 'SGC'

        with fits.open(fits_path) as hdu:
            # Verify cosmology from header
            omega_m = hdu[0].header.get('OMEGAM', None)
            if omega_m is not None:
                assert abs(omega_m - 0.315) < 0.01, \
                    f"OMEGAM mismatch: {omega_m} vs expected 0.315"

            if catalog_type == 'voidfinder':
                data = hdu['MAXIMALS'].data
                self.x = data['X'].astype(np.float64)
                self.y = data['Y'].astype(np.float64)
                self.z_cart = data['Z'].astype(np.float64)
                self.r = data['R_EFF'].astype(np.float64)
                self.edge = data['EDGE']
            else:  # V2 (REVOLVER, VIDE)
                data = hdu['VOIDS'].data
                self.x = data['X'].astype(np.float64)
                self.y = data['Y'].astype(np.float64)
                self.z_cart = data['Z'].astype(np.float64)
                self.r = data['RADIUS'].astype(np.float64)
                self.edge = None

        print(f"  Loaded {len(self.x)} voids from {Path(fits_path).name} "
              f"({self.region}, {catalog_type})")

    def get_positions(self, interior_only=True):
        """Return (x, y, z, r) arrays. For VoidFinder, optionally filter EDGE==0."""
        if interior_only and self.catalog_type == 'voidfinder' and self.edge is not None:
            mask = self.edge == 0
            return self.x[mask], self.y[mask], self.z_cart[mask], self.r[mask]
        return self.x, self.y, self.z_cart, self.r


def load_all_void_catalogs(data_dir):
    """Load all 6 DESIVAST void catalogs.

    Returns dict keyed by (finder, region) tuples.
    """
    base = Path(data_dir) / "External" / "desivast_voids"

    catalog_specs = {
        ('voidfinder', 'NGC'): ('DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits', 'voidfinder'),
        ('voidfinder', 'SGC'): ('DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits', 'voidfinder'),
        ('revolver', 'NGC'): ('DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits', 'v2'),
        ('revolver', 'SGC'): ('DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits', 'v2'),
        ('vide', 'NGC'): ('DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits', 'v2'),
        ('vide', 'SGC'): ('DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits', 'v2'),
    }

    catalogs = {}
    for key, (fname, cat_type) in catalog_specs.items():
        path = base / fname
        if path.exists():
            catalogs[key] = VoidCatalog(str(path), cat_type)
        else:
            print(f"  WARNING: {fname} not found")

    return catalogs


def sn_to_comoving(z, ra, dec):
    """Convert SN positions to comoving Cartesian in Mpc/h.

    Uses COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315) to match
    DESIVAST void coordinate system.
    """
    d_c = COSMO_VOIDS.comoving_distance(z).value  # Mpc/h (H0=100)
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d_c * np.sin(dec_rad)
    return np.column_stack([x, y, z_cart])


def combine_ngc_sgc_voids(catalogs, finder_name):
    """Combine NGC + SGC void positions for a given finder.

    For VoidFinder, applies EDGE==0 interior filter.
    Returns (positions (N,3), radii (N,)) and N_ngc count for splitting.
    """
    ngc = catalogs.get((finder_name, 'NGC'))
    sgc = catalogs.get((finder_name, 'SGC'))

    if ngc is None or sgc is None:
        raise ValueError(f"Missing NGC or SGC catalog for {finder_name}")

    ngc_x, ngc_y, ngc_z, ngc_r = ngc.get_positions(interior_only=True)
    sgc_x, sgc_y, sgc_z, sgc_r = sgc.get_positions(interior_only=True)

    n_ngc = len(ngc_x)

    pos = np.column_stack([
        np.concatenate([ngc_x, sgc_x]),
        np.concatenate([ngc_y, sgc_y]),
        np.concatenate([ngc_z, sgc_z]),
    ])
    radii = np.concatenate([ngc_r, sgc_r])

    return pos, radii, n_ngc
