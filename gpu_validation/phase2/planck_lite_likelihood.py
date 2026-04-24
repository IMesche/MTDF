# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Planck plik-lite TTTEEE likelihood (bins ell=30-2508).

Standalone reimplementation based on cobaya's PlanckPlikLite by
Erminia Calabrese and Antony Lewis.

Data vector is in C_l units. Theory input is D_l = l(l+1)/(2pi) C_l.
Weights include the 2pi/l/(l+1) conversion factor.
"""

import numpy as np
from pathlib import Path
from scipy.io import FortranFile

DATA_DIR = Path(__file__).parent / "data" / "planck"

# Standard plik-lite v22 TTTEEE parameters
NBINTT = 215
NBINTE = 199
NBINEE = 199
BIN_LMIN_OFFSET = 30


class PlanckLiteLikelihood:
    """Planck 2018 plik-lite TTTEEE likelihood."""

    def __init__(self, data_dir=None, use_cl=('tt', 'te', 'ee')):
        if data_dir is None:
            data_dir = DATA_DIR
        data_dir = Path(data_dir)

        self.use_cl = [c.lower() for c in use_cl]

        # Load binning scheme
        blmin_raw = np.loadtxt(data_dir / "blmin.dat").astype(int)
        blmax_raw = np.loadtxt(data_dir / "blmax.dat").astype(int)
        self.blmin = blmin_raw + BIN_LMIN_OFFSET
        self.blmax = blmax_raw + BIN_LMIN_OFFSET

        # Weights: convert from C_l weighting to D_l weighting
        weights_raw = np.loadtxt(data_dir / "bweight.dat")
        ls = np.arange(len(weights_raw)) + BIN_LMIN_OFFSET
        weights_raw *= 2 * np.pi / ls / (ls + 1)  # so dot(D_l, weights) gives C_l
        self.weights = np.hstack((np.zeros(BIN_LMIN_OFFSET), weights_raw))

        # Load data vector (3 columns: ell_eff, C_l, sigma)
        data = np.loadtxt(data_dir / "cl_cmb_plik_v22.dat")
        self.nbins = NBINTT + NBINTE + NBINEE
        assert data.shape[0] == self.nbins

        # Load covariance (Fortran binary, lower triangular)
        cov_file = data_dir / "c_matrix_plik_v22.dat"
        f = FortranFile(str(cov_file), 'r')
        cov_flat = f.read_reals(dtype=float)
        f.close()
        cov = cov_flat.reshape((self.nbins, self.nbins))
        cov = np.tril(cov) + np.tril(cov, -1).T  # symmetrize

        # Select used bins per spectrum
        maxbin = max(NBINTT, NBINTE, NBINEE)
        self.lav = (self.blmin[:maxbin] + self.blmax[:maxbin]) // 2
        cl_names = ['tt', 'te', 'ee']
        nbins_per = [NBINTT, NBINTE, NBINEE]

        self.used_bins = []
        used_indices = []
        offset = 0
        for i, (cl, nbin) in enumerate(zip(cl_names, nbins_per)):
            if cl in self.use_cl:
                bins = np.arange(nbin, dtype=int)
                self.used_bins.append(bins)
                used_indices.append(bins + offset)
            else:
                self.used_bins.append(np.arange(0, dtype=int))
            offset += nbin

        self.used_indices = np.hstack(used_indices)
        self.n_used = len(self.used_indices)
        self.X_data = data[self.used_indices, 1]  # C_l observed values
        self.cov = cov[np.ix_(self.used_indices, self.used_indices)]
        self.invcov = np.linalg.inv(self.cov)

        # For diagnostics
        self.n_tt = NBINTT if 'tt' in self.use_cl else 0
        self.n_te = NBINTE if 'te' in self.use_cl else 0
        self.n_ee = NBINEE if 'ee' in self.use_cl else 0

    def bin_theory(self, dl_tt, dl_te=None, dl_ee=None, A_planck=1.0):
        """Bin theory D_l spectra into the plik-lite bin scheme.

        Parameters
        ----------
        dl_tt, dl_te, dl_ee : array
            Theory D_l = l(l+1)/(2pi)C_l in uK^2, starting from ell=0.
        A_planck : float
            Calibration parameter (default 1.0).

        Returns
        -------
        cl_binned : array of shape (n_used,)
            Binned theory vector in C_l units matching the data.
        """
        spectra = [dl_tt, dl_te, dl_ee]
        cl = np.empty(self.n_used)
        ix = 0
        for tp in range(3):
            cell = spectra[tp]
            if cell is None:
                continue
            for i in self.used_bins[tp]:
                lmin = self.blmin[i]
                lmax = self.blmax[i]
                cl[ix] = np.dot(
                    cell[lmin:lmax + 1],
                    self.weights[lmin:lmax + 1],
                )
                ix += 1
        cl /= A_planck ** 2
        return cl

    def chi2(self, dl_tt, dl_te=None, dl_ee=None, A_planck=1.0):
        """Compute chi-squared for theory D_l spectra.

        Parameters
        ----------
        dl_tt, dl_te, dl_ee : array
            Theory D_l starting from ell=0.
        A_planck : float
            Calibration parameter.

        Returns
        -------
        chi2 : float
        """
        cl_theory = self.bin_theory(dl_tt, dl_te, dl_ee, A_planck)
        diff = self.X_data - cl_theory
        return float(diff @ self.invcov @ diff)

    def log_likelihood(self, dl_tt, dl_te=None, dl_ee=None, A_planck=1.0):
        """Compute log-likelihood."""
        return -0.5 * self.chi2(dl_tt, dl_te, dl_ee, A_planck)
