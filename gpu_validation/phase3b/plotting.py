# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""Phase 3b diagnostic plots."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def _save(fig, out_dir, name, formats):
    """Save figure in requested formats."""
    for fmt in formats:
        fig.savefig(Path(out_dir) / 'plots' / f'{name}.{fmt}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_weight_distribution(weights, ngc_mask, out_dir, formats=('png', 'pdf')):
    """Test A: IPW weight distribution per footprint."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, mask, label in [(axes[0], ngc_mask, 'NGC'),
                             (axes[1], ~ngc_mask, 'SGC')]:
        w = weights[mask]
        ax.hist(w, bins=30, color='steelblue' if label == 'NGC' else 'coral',
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='w=1')
        ax.set_xlabel('IPW Weight')
        ax.set_title(f'{label} (N={len(w)}, ESS={np.sum(w)**2/np.sum(w**2):.0f})')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Count')
    fig.suptitle('Test A: Stabilised IPW Weight Distribution', fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'test_a_weight_distribution', formats)


def plot_balance_table(balance_rows, out_dir, formats=('png', 'pdf')):
    """Test A: SMD before/after weighting."""
    if not balance_rows:
        return

    labels = [r['covariate'] for r in balance_rows]
    smd_raw = [r['smd_raw'] for r in balance_rows]
    smd_weighted = [r['smd_weighted'] for r in balance_rows]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.8)))

    ax.barh(y + 0.15, smd_raw, height=0.3, color='salmon', label='Raw')
    ax.barh(y - 0.15, smd_weighted, height=0.3, color='steelblue', label='Weighted')
    ax.axvline(0.1, color='gray', linestyle=':', alpha=0.7, label='|SMD|=0.1')
    ax.axvline(-0.1, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Standardised Mean Difference (NGC - SGC)')
    ax.set_title('Test A: Covariate Balance')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, 'test_a_balance', formats)


def plot_detectability(table, metric, ylabel, title, name,
                        out_dir, formats=('png', 'pdf')):
    """Generic Test B plot: metric vs redshift, NGC vs SGC per finder."""
    finders = sorted(set(r['finder'] for r in table))
    colors = {'voidfinder': 'C0', 'revolver': 'C1', 'vide': 'C2'}
    markers = {'NGC': 'o', 'SGC': 's'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for finder in finders:
        for region in ['NGC', 'SGC']:
            rows = [r for r in table
                    if r['finder'] == finder and r['region'] == region]
            if not rows:
                continue
            z_mid = [r['z_mid'] for r in rows]
            vals = [r[metric] for r in rows]
            ls = '-' if region == 'NGC' else '--'
            ax.plot(z_mid, vals,
                    marker=markers[region], linestyle=ls,
                    color=colors.get(finder, 'gray'),
                    label=f'{finder} {region}', markersize=6)

    ax.set_xlabel('Redshift')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Test B: {title}')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, name, formats)


def plot_void_per_volume(table, out_dir, formats=('png', 'pdf')):
    plot_detectability(table, 'void_density',
                       'Void density [(Mpc/h)$^{-3}$]',
                       'Void Number Density vs Redshift',
                       'test_b_void_per_volume', out_dir, formats)


def plot_sn_per_void(table, out_dir, formats=('png', 'pdf')):
    plot_detectability(table, 'sn_per_void',
                       'SN per void',
                       'SN per Void vs Redshift',
                       'test_b_sn_per_void', out_dir, formats)


def plot_median_void_radius(table, out_dir, formats=('png', 'pdf')):
    plot_detectability(table, 'median_r_void',
                       'Median void radius [Mpc/h]',
                       'Median Void Radius vs Redshift',
                       'test_b_median_void_radius', out_dir, formats)


def plot_bootstrap_lr(obs_lr, bootstrap_lr, finder, out_dir,
                       formats=('png', 'pdf')):
    """Test C: Bootstrap LR distribution with observed statistic."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(bootstrap_lr, bins=40, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.5, density=True,
            label=f'Bootstrap H0 (n={len(bootstrap_lr)})')
    ax.axvline(obs_lr, color='red', linewidth=2, linestyle='--',
               label=f'Observed LR = {obs_lr:.3f}')

    p = float(np.mean(bootstrap_lr >= obs_lr))
    ax.set_xlabel('Likelihood Ratio Statistic')
    ax.set_ylabel('Density')
    ax.set_title(f'Test C: Parametric Bootstrap [{finder.upper()}] (p={p:.3f})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, f'test_c_bootstrap_lr_{finder}', formats)


def plot_mock_delta_gamma(obs_dg, mock_dgs, finder, out_dir,
                           formats=('png', 'pdf')):
    """Test D: Mock Δγ histogram with observed marked."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mock_dgs, bins=40, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.5, density=True,
            label=f'Isotropic mocks (n={len(mock_dgs)})')
    ax.axvline(obs_dg, color='red', linewidth=2, linestyle='--',
               label=f'Observed Δγ = {obs_dg:+.4f}')
    ax.axvline(-obs_dg, color='red', linewidth=1, linestyle=':',
               alpha=0.5)

    p = float(np.mean(np.abs(mock_dgs) >= np.abs(obs_dg)))
    ax.set_xlabel('Δγ (NGC − SGC)')
    ax.set_ylabel('Density')
    ax.set_title(f'Test D: Geometry Mocks [{finder.upper()}] (p={p:.3f})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, f'test_d_mock_delta_gamma_{finder}', formats)


def plot_recovery(gamma_inject, gammas, strength_label, finder,
                   out_dir, formats=('png', 'pdf')):
    """Test E: Recovered γ distribution with true value marked."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(gammas, bins=30, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.5, density=True,
            label=f'Recovered γ (n={len(gammas)})')
    ax.axvline(gamma_inject, color='red', linewidth=2, linestyle='--',
               label=f'Injected γ = {gamma_inject:+.4f}')
    ax.axvline(np.mean(gammas), color='orange', linewidth=1.5,
               label=f'Mean = {np.mean(gammas):+.4f}')

    bias = np.mean(gammas) - gamma_inject
    ax.set_xlabel('γ_env (recovered)')
    ax.set_ylabel('Density')
    ax.set_title(f'Test E: SGC Recovery [{finder.upper()}] {strength_label} '
                 f'(bias={bias:+.4f})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, f'test_e_recovery_{finder}_{strength_label}', formats)
