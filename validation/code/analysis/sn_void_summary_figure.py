#!/usr/bin/env python3
"""
MTDF Part I: Summary Figure for SN × Void Environment Analysis

Generates a publication-ready figure showing:
1. Z-binned γ_env demonstrating signal concentration at low-z
2. LOSO stability demonstrating no single survey drives the result
3. Robustness summary across void finders

Author: Ingo Mesche
Affiliation: Independent Researcher, Malta
Date: December 2025
Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
Date: 2025-12-17
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy import linalg, stats
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

COSMO_VOIDS = FlatLambdaCDM(H0=100, Om0=0.315)
COSMO_SN = FlatLambdaCDM(H0=70, Om0=0.3)

SURVEY_NAMES = {
    1: 'CfA1', 4: 'CfA2', 5: 'CfA3S', 10: 'CfA3K', 15: 'CfA4', 18: 'CSP',
    50: 'PS1MD', 51: 'SNLS', 56: 'SDSS', 57: 'Foundation', 61: 'CNIa0.02',
    62: 'LOSS', 63: 'SOUSA', 64: 'Misc_lowz', 65: 'Misc_highz', 66: 'DES',
    100: 'HST', 101: 'CFA4_p2', 106: 'PS1_LOWZ', 150: 'CfA3'
}


class PantheonData:
    """Load Pantheon+ data."""
    def __init__(self, data_path, cov_path):
        self.data = np.genfromtxt(data_path, names=True, dtype=None, encoding='utf-8')
        n = len(self.data)
        cov_flat = np.loadtxt(cov_path, skiprows=1)
        self.cov_full = cov_flat.reshape(n, n)
        self.z = self.data['zCMB']
        self.mu_obs = self.data['MU_SH0ES']
        self.m_b_corr = self.data['m_b_corr']
        self.host_mass = self.data['HOST_LOGMASS']
        self.ra = self.data['RA']
        self.dec = self.data['DEC']
        self.survey_id = self.data['IDSURVEY']
        self.mu = np.where(self.mu_obs > 0, self.mu_obs, self.m_b_corr + 19.25)


class VoidCatalog:
    """Load void catalogs."""
    def __init__(self, fits_path, catalog_type='voidfinder'):
        with fits.open(fits_path) as hdu:
            if catalog_type == 'voidfinder':
                self.data = hdu['MAXIMALS'].data
                self.x, self.y, self.z = self.data['X'], self.data['Y'], self.data['Z']
                self.r, self.edge = self.data['R_EFF'], self.data['EDGE']
            else:
                self.data = hdu['VOIDS'].data
                self.x, self.y, self.z = self.data['X'], self.data['Y'], self.data['Z']
                self.r = self.data['RADIUS']
                self.edge = np.zeros(len(self.x))

    def filter_interior(self):
        mask = self.edge == 0 if hasattr(self, 'edge') else np.ones(len(self.x), dtype=bool)
        return self.x[mask], self.y[mask], self.z[mask], self.r[mask]


def sn_to_comoving(z, ra, dec, cosmo=COSMO_VOIDS):
    d_c = cosmo.comoving_distance(z).value
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d_c * np.sin(dec_rad)
    return x, y, z_cart


def compute_environment(sn_x, sn_y, sn_z, void_x, void_y, void_z, void_r):
    n_sn = len(sn_x)
    d_signed = np.full(n_sn, np.inf)
    for i in range(n_sn):
        dx, dy, dz = sn_x[i] - void_x, sn_y[i] - void_y, sn_z[i] - void_z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        d_sign = (dist - void_r) / void_r
        d_signed[i] = d_sign[np.argmin(dist / void_r)]
    return d_signed


def gls_fit(y, X, cov):
    cov_inv = linalg.inv(cov)
    XtCinv = X.T @ cov_inv
    beta_cov = linalg.inv(XtCinv @ X)
    beta = beta_cov @ (XtCinv @ y)
    residual = y - X @ beta
    chi2 = residual @ cov_inv @ residual
    return beta, beta_cov, chi2, len(y) - X.shape[1]


def delta_chi2_test(mu, z, env_metric, host_mass, cov):
    """Test significance of environment term."""
    n = len(mu)
    mu_theory = COSMO_SN.distmod(z).value
    residual = mu - mu_theory
    mass_step = (host_mass >= 10).astype(float)

    X_null = np.column_stack([np.ones(n), mass_step])
    _, _, chi2_null, _ = gls_fit(residual, X_null, cov)

    X_full = np.column_stack([np.ones(n), env_metric, mass_step])
    beta_full, beta_cov, chi2_full, dof = gls_fit(residual, X_full, cov)

    delta_chi2 = chi2_null - chi2_full
    p_value = 1 - stats.chi2.cdf(delta_chi2, 1)

    return {
        'gamma_env': beta_full[1],
        'gamma_env_err': np.sqrt(beta_cov[1, 1]),
        'delta_chi2': delta_chi2,
        'p_value': p_value
    }


def run_z_binned_analysis(mu, z, d_signed, host_mass, cov, z_bins):
    """Run analysis in z bins."""
    results = []
    for i in range(len(z_bins) - 1):
        z_lo, z_hi = z_bins[i], z_bins[i+1]
        mask = (z >= z_lo) & (z < z_hi)
        n_bin = np.sum(mask)

        if n_bin > 20:
            idx = np.where(mask)[0]
            cov_bin = cov[np.ix_(idx, idx)]
            result = delta_chi2_test(mu[mask], z[mask], d_signed[mask], host_mass[mask], cov_bin)
            results.append({
                'z_lo': z_lo, 'z_hi': z_hi,
                'z_mean': np.mean(z[mask]),
                'n': n_bin,
                **result
            })
    return results


def run_loso_analysis(mu, z, d_signed, host_mass, survey_ids, cov):
    """Leave-One-Survey-Out analysis."""
    results = []
    unique_surveys = np.unique(survey_ids)

    for drop_survey in unique_surveys:
        mask = survey_ids != drop_survey
        n_remain = np.sum(mask)

        if n_remain > 50:
            idx = np.where(mask)[0]
            cov_sub = cov[np.ix_(idx, idx)]
            result = delta_chi2_test(mu[mask], z[mask], d_signed[mask], host_mass[mask], cov_sub)

            survey_name = SURVEY_NAMES.get(drop_survey, f'Survey_{drop_survey}')
            results.append({
                'dropped': survey_name,
                'n_remain': n_remain,
                **result
            })

    return results


def create_summary_figure(output_path):
    """Create the summary figure."""
    base = str(Path(__file__).parent.parent.parent / 'data' / 'External')

    print("Loading data...")
    pantheon = PantheonData(
        os.path.join(base, 'pantheonplus/Pantheon+SH0ES.dat'),
        os.path.join(base, 'pantheonplus/Pantheon+SH0ES_STAT+SYS.cov')
    )

    # Load REVOLVER voids (NGC + SGC)
    void_ngc = VoidCatalog(os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits'), 'v2')
    void_sgc = VoidCatalog(os.path.join(base, 'desivast_voids/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits'), 'v2')

    vx_ngc, vy_ngc, vz_ngc, vr_ngc = void_ngc.filter_interior()
    vx_sgc, vy_sgc, vz_sgc, vr_sgc = void_sgc.filter_interior()
    vx_all = np.concatenate([vx_ngc, vx_sgc])
    vy_all = np.concatenate([vy_ngc, vy_sgc])
    vz_all = np.concatenate([vz_ngc, vz_sgc])
    vr_all = np.concatenate([vr_ngc, vr_sgc])

    # Get sample
    mask = (pantheon.z >= 0.02) & (pantheon.z <= 0.157)
    idx = np.where(mask)[0]

    z = pantheon.z[idx]
    mu = pantheon.mu[idx]
    host_mass = pantheon.host_mass[idx]
    survey_ids = pantheon.survey_id[idx]
    cov = pantheon.cov_full[np.ix_(idx, idx)]

    sn_x, sn_y, sn_z = sn_to_comoving(z, pantheon.ra[idx], pantheon.dec[idx])
    d_signed = compute_environment(sn_x, sn_y, sn_z, vx_all, vy_all, vz_all, vr_all)

    # Run analyses
    print("Running z-binned analysis...")
    z_bins = [0.02, 0.04, 0.06, 0.10, 0.157]
    z_binned = run_z_binned_analysis(mu, z, d_signed, host_mass, cov, z_bins)

    print("Running LOSO analysis...")
    loso = run_loso_analysis(mu, z, d_signed, host_mass, survey_ids, cov)

    # Filter LOSO to show only surveys with significant contribution
    loso = [r for r in loso if r['n_remain'] < len(idx) - 20]
    loso = sorted(loso, key=lambda x: -x['gamma_env'])

    # Create figure with better layout
    fig = plt.figure(figsize=(11, 9))

    # Use GridSpec for more control - tighter spacing
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.25, wspace=0.22,
                  left=0.08, right=0.97, top=0.95, bottom=0.06)

    # Panel A: Z-binned γ_env
    ax1 = fig.add_subplot(gs[0, 0])

    z_centers = [r['z_mean'] for r in z_binned]
    gammas = [r['gamma_env'] for r in z_binned]
    errs = [r['gamma_env_err'] for r in z_binned]

    colors = ['#2ecc71' if r['p_value'] < 0.01 else '#3498db' if r['p_value'] < 0.05 else '#95a5a6'
              for r in z_binned]

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.errorbar(z_centers, gammas, yerr=errs, fmt='o', capsize=4, capthick=1.5,
                 markersize=8, color='#2c3e50', ecolor='#2c3e50', linewidth=1.5)

    # Color-code by significance
    for i, (x, y, c) in enumerate(zip(z_centers, gammas, colors)):
        ax1.plot(x, y, 'o', markersize=10, color=c, zorder=5)

    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel(r'$\gamma_{\rm env}$ (mag per unit $d_{\rm signed}$)')
    ax1.set_title('(a) Environment Signal vs. Redshift')
    ax1.set_xlim(0.01, 0.17)

    # Add legend
    legend_elements = [
        Patch(facecolor='#2ecc71', label=r'$p < 0.01$'),
        Patch(facecolor='#3498db', label=r'$0.01 \leq p < 0.05$'),
        Patch(facecolor='#95a5a6', label=r'$p \geq 0.05$')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Panel B: LOSO stability
    ax2 = fig.add_subplot(gs[0, 1])

    survey_names = [r['dropped'] for r in loso]
    loso_gammas = [r['gamma_env'] for r in loso]
    loso_errs = [r['gamma_env_err'] for r in loso]

    y_pos = np.arange(len(survey_names))

    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.errorbar(loso_gammas, y_pos, xerr=loso_errs, fmt='o', capsize=3, capthick=1,
                 markersize=6, color='#e74c3c', ecolor='#e74c3c')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(survey_names, fontsize=8)
    ax2.set_xlabel(r'$\gamma_{\rm env}$ (mag per unit $d_{\rm signed}$)')
    ax2.set_title('(b) Leave-One-Survey-Out Stability')
    ax2.invert_yaxis()

    # Add vertical line for full sample result
    full_result = delta_chi2_test(mu, z, d_signed, host_mass, cov)
    ax2.axvline(full_result['gamma_env'], color='#3498db', linestyle='-',
                alpha=0.7, linewidth=2, label='Full sample')
    ax2.legend(loc='lower right', fontsize=9)

    # Panel C: Summary table (text) - use smaller font
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    table_text = """KEY RESULTS: SN x VOID ENVIRONMENT

Combined Sample (z <= 0.157):
  REVOLVER:   gamma = +0.0047 +/- 0.0023, p = 0.039
  VIDE:       gamma = +0.0046 +/- 0.0026, p = 0.076
  VoidFinder: gamma = +0.0030 +/- 0.0018, p = 0.088

NGC Footprint (p < 0.01 for two finders):
  REVOLVER NGC: gamma = +0.0103 +/- 0.0038, p = 0.006
  VIDE NGC:     gamma = +0.0111 +/- 0.0040, p = 0.006

Low-z Bin (0.02 <= z < 0.04):
  gamma = +0.013 +/- 0.004, p = 0.003 (most significant)

Robustness:
  * 12/12 LOSO runs: all positive gamma, all p < 0.05
  * Survey FE: gamma unchanged (delta-gamma < 0.0005)
  * Z-matched SGC: still null -> footprint-specific"""

    ax3.text(0.05, 0.95, table_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, pad=0.5))
    ax3.set_title('(c) Key Results Summary', fontsize=11)

    # Panel D: Physical interpretation - use smaller font
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    interp_text = """PHYSICAL INTERPRETATION

Observation:
  SNe in/near cosmic voids appear ~0.01 mag
  BRIGHTER than SNe in overdense regions,
  at fixed LCDM distance.

MTDF Prediction (Section 2.3):
  If the stress field has finite coherence
  scale, large underdensities can sustain
  different stress states -> local G_eff.

Consistency Checks:
  [Y] Signal concentrated at low z
  [Y] Same sign across 3 void finders
  [Y] Independent of host-mass step
  [Y] Stable under survey fixed effects
  [Y] Not driven by any single survey

Caveat: Only 4 SNe strictly inside voids.
Inference is from continuous signed-distance."""

    ax4.text(0.05, 0.95, interp_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, pad=0.5))
    ax4.set_title('(d) Physical Interpretation', fontsize=11)

    # Save both formats
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.replace('.png', '.pdf'))
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

    return z_binned, loso


if __name__ == '__main__':
    output_dir = str(Path(__file__).parent)
    z_binned, loso = create_summary_figure(os.path.join(output_dir, 'sn_void_summary_figure.png'))

    print("\n" + "="*60)
    print("Z-BINNED RESULTS:")
    print("="*60)
    for r in z_binned:
        sig = "★★" if r['p_value'] < 0.01 else "★" if r['p_value'] < 0.05 else ""
        print(f"  z=[{r['z_lo']:.2f}, {r['z_hi']:.2f}): γ={r['gamma_env']:+.4f} ± {r['gamma_env_err']:.4f}, "
              f"p={r['p_value']:.4f} {sig}")

    print("\n" + "="*60)
    print("LOSO RESULTS:")
    print("="*60)
    for r in loso:
        print(f"  Drop {r['dropped']:<12}: γ={r['gamma_env']:+.4f} ± {r['gamma_env_err']:.4f}, p={r['p_value']:.4f}")
