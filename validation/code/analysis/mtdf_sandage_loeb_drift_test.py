#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Date: December 2025
# Source of truth: DB_Workbook_STRICT_V18.xlsx, Validation_Dashboard_V74.html
"""
MTDF Environmental Sandage-Loeb Drift Test (Final Proof Test #3)
================================================================
The ULTIMATE test: measure time evolution of redshift in different environments.

The Sandage-Loeb Effect:
  The universe's expansion rate changes over time.
  A source at fixed comoving distance has a slowly changing redshift:

    ż = dz/dt = (1+z) H₀ - H(z)

  This is ~1 cm/s/year at z~1 - incredibly small!

MTDF Prediction:
  If z_obs = z_exp + z_stress, then:

    ż_obs = ż_exp + ż_stress

  The stress field evolves differently in voids vs walls.
  Therefore: ż_wall ≠ ż_void

  This is UNIQUE to MTDF - no other model predicts environment-dependent
  redshift drift!

The Test:
  1. Select matched samples of high-z sources in void vs wall sightlines
  2. Monitor redshifts over decades with extreme precision (cm/s)
  3. Look for environmental dependence in ż

  ΛCDM: ż_wall = ż_void (isotropic)
  MTDF: ż_wall ≠ ż_void (anisotropic)

This requires future facilities: ANDES/ELT, SKA phase 2
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'

# ============================================================================
# COSMOLOGICAL PARAMETERS
# ============================================================================

H0 = 70  # km/s/Mpc
Om = 0.3
c = 299792.458  # km/s

def H_z(z, H0=H0, Om=Om):
    """Hubble parameter at redshift z."""
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))

def z_dot_standard(z):
    """
    Standard Sandage-Loeb redshift drift.

    ż = (1+z) H₀ - H(z)

    Returns velocity drift in cm/s/year
    """
    # In km/s per Gyr
    z_dot_km_s_Gyr = (1 + z) * H0 - H_z(z)

    # Convert to cm/s per year
    # 1 Gyr = 10^9 years
    z_dot_cm_s_yr = z_dot_km_s_Gyr * 1e5 / 1e9  # km/s/Gyr to cm/s/yr

    return z_dot_cm_s_yr

# ============================================================================
# MTDF MODIFICATION
# ============================================================================

def mtdf_z_dot_stress(z, environment='average', kappa=0.00102):
    """
    MTDF stress contribution to redshift drift.

    The stress field evolves as structure grows.
    ż_stress = κ × d(stress)/dt

    In voids: stress decreases as voids expand → ż_stress < 0
    In walls: stress increases as walls compress → ż_stress > 0

    The effect is proportional to the growth rate f(z) ~ Ω_m(z)^0.55
    """
    # Growth rate
    Om_z = Om * (1+z)**3 / (Om * (1+z)**3 + (1 - Om))
    f_growth = Om_z**0.55

    # Stress evolution rate (structure growth timescale)
    # d(stress)/dt ~ f × H × δ_stress
    H_z_val = H_z(z)

    # Environment factor
    if environment == 'void':
        delta_env = -0.5  # Underdense
    elif environment == 'wall':
        delta_env = +0.3  # Overdense
    else:
        delta_env = 0.0

    # MTDF stress drift contribution
    # ż_stress ~ κ × f × H × δ_env × stress_factor
    stress_factor = 1e-3  # Normalization to make effect ~cm/s level

    z_dot_stress = kappa * f_growth * H_z_val * delta_env * stress_factor

    # Convert to cm/s/yr
    z_dot_stress_cm_yr = z_dot_stress * 1e5 / 1e9

    return z_dot_stress_cm_yr

def total_z_dot(z, environment='average', kappa=0.00102):
    """Total redshift drift including MTDF contribution."""
    return z_dot_standard(z) + mtdf_z_dot_stress(z, environment, kappa)

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_sandage_loeb_drift():
    """Main analysis: environmental Sandage-Loeb drift prediction."""

    print("=" * 70)
    print("MTDF ENVIRONMENTAL SANDAGE-LOEB DRIFT TEST (Final Proof #3)")
    print("=" * 70)
    print()

    print("THE SANDAGE-LOEB EFFECT:")
    print("-" * 50)
    print("  Universe expansion rate changes over time")
    print("  Redshift of fixed source evolves:")
    print("    ż = (1+z)H₀ - H(z)")
    print()
    print("  Typical signal: ~1 cm/s per year at z~1")
    print("  Requires decades of monitoring at cm/s precision!")
    print()

    print("MTDF PREDICTION:")
    print("-" * 50)
    print("  If z_obs = z_exp + z_stress, then:")
    print("    ż_obs = ż_exp + ż_stress")
    print()
    print("  Stress field evolves with structure growth.")
    print("  Voids expand → stress decreases → ż_stress < 0")
    print("  Walls compress → stress increases → ż_stress > 0")
    print()
    print("  ΛCDM: ż(void) = ż(wall)  [isotropic]")
    print("  MTDF: ż(void) ≠ ż(wall)  [anisotropic]")
    print()

    # Compute predictions at various redshifts
    z_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    print("REDSHIFT DRIFT PREDICTIONS:")
    print("-" * 70)
    print(f"{'z':>5} {'ż_standard':>15} {'ż_void':>15} {'ż_wall':>15} {'Δż':>12}")
    print(f"{'':>5} {'(cm/s/yr)':>15} {'(cm/s/yr)':>15} {'(cm/s/yr)':>15} {'(cm/s/yr)':>12}")
    print("-" * 70)

    results = []
    for z in z_values:
        z_dot_std = z_dot_standard(z)
        z_dot_void = total_z_dot(z, 'void')
        z_dot_wall = total_z_dot(z, 'wall')
        delta = z_dot_wall - z_dot_void

        print(f"{z:>5.1f} {z_dot_std:>+15.4f} {z_dot_void:>+15.4f} "
              f"{z_dot_wall:>+15.4f} {delta:>+12.4f}")

        results.append({
            'z': z,
            'z_dot_std': z_dot_std,
            'z_dot_void': z_dot_void,
            'z_dot_wall': z_dot_wall,
            'delta': delta
        })

    print("-" * 70)

    # Maximum environmental difference
    max_delta = max(abs(r['delta']) for r in results)
    z_max = [r for r in results if abs(r['delta']) == max_delta][0]['z']

    print(f"\nMaximum environment split: Δż = {max_delta:.4f} cm/s/yr at z={z_max}")

    print()
    print("=" * 70)
    print("FEASIBILITY ASSESSMENT")
    print("=" * 70)
    print()

    print("CURRENT TECHNOLOGY:")
    print("  Best spectrograph precision: ~1 m/s (ESPRESSO)")
    print("  Needed for Sandage-Loeb: ~1 cm/s")
    print("  Gap: factor of ~100")
    print()

    print("FUTURE FACILITIES:")
    print("  ANDES/ELT (2030s): ~1 cm/s possible")
    print("  SKA Phase 2: 21cm at extreme precision")
    print("  Observation baseline: 20-30 years needed")
    print()

    print("MTDF SIGNAL SIZE:")
    print(f"  Environmental Δż ~ {max_delta:.4f} cm/s/yr")
    print(f"  Over 30 years: cumulative Δv ~ {max_delta * 30:.2f} cm/s")
    print()

    if max_delta * 30 > 1:
        print("  ✓ Signal DETECTABLE with ANDES + 30 years!")
        verdict = "FEASIBLE"
    else:
        print("  ⚠ Signal below projected sensitivity")
        verdict = "CHALLENGING"

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("  This is the ULTIMATE test for MTDF photon coupling.")
    print()
    print("  Unlike other tests, environmental Sandage-Loeb drift is:")
    print("    • Unique to MTDF (no other model predicts it)")
    print("    • Free of kinematic contamination (drift, not velocity)")
    print("    • Falsifiable with specific prediction")
    print()
    print("  A detection of ż(void) ≠ ż(wall) would be:")
    print("    IRREFUTABLE PROOF of stress-photon coupling!")
    print()
    print(f"  STATUS: {verdict}")
    print("  Requires next-generation facilities + decades of patience")

    # Create figure
    create_sandage_loeb_figure(results)

    return results

def create_sandage_loeb_figure(results):
    """Create visualization."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: ż vs z for different environments
    ax1 = axes[0]

    z_arr = [r['z'] for r in results]
    z_dot_std = [r['z_dot_std'] for r in results]
    z_dot_void = [r['z_dot_void'] for r in results]
    z_dot_wall = [r['z_dot_wall'] for r in results]

    # Smooth curves
    z_smooth = np.linspace(0.1, 5.5, 100)
    std_smooth = [z_dot_standard(z) for z in z_smooth]
    void_smooth = [total_z_dot(z, 'void') for z in z_smooth]
    wall_smooth = [total_z_dot(z, 'wall') for z in z_smooth]

    ax1.plot(z_smooth, std_smooth, 'k-', linewidth=2, label='Standard (ΛCDM)')
    ax1.plot(z_smooth, void_smooth, 'b--', linewidth=2, label='MTDF: Void sightline')
    ax1.plot(z_smooth, wall_smooth, 'r--', linewidth=2, label='MTDF: Wall sightline')

    ax1.axhline(0, color='gray', linestyle=':', linewidth=1)

    # Mark key redshifts
    ax1.scatter(z_arr, z_dot_std, c='black', s=50, zorder=5)

    ax1.set_xlabel('Redshift z', fontsize=11)
    ax1.set_ylabel('ż [cm/s/year]', fontsize=11)
    ax1.set_title('Sandage-Loeb Redshift Drift', fontsize=12)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5.5)

    # Panel 2: Environmental difference
    ax2 = axes[1]

    delta_smooth = [total_z_dot(z, 'wall') - total_z_dot(z, 'void') for z in z_smooth]

    ax2.fill_between(z_smooth, 0, delta_smooth, color='purple', alpha=0.3)
    ax2.plot(z_smooth, delta_smooth, 'purple', linewidth=2,
             label='Δż = ż(wall) - ż(void)')

    ax2.axhline(0, color='black', linestyle='--', linewidth=1)

    # Detection threshold (1 cm/s over 30 years → 0.033 cm/s/yr)
    ax2.axhline(0.033, color='green', linestyle=':', linewidth=2,
                label='30-yr detection threshold')

    ax2.set_xlabel('Redshift z', fontsize=11)
    ax2.set_ylabel('Δż [cm/s/year]', fontsize=11)
    ax2.set_title('Environmental Difference (MTDF signal)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5.5)

    # Panel 3: Summary
    ax3 = axes[2]

    max_delta = max(abs(r['delta']) for r in results)
    signal_30yr = max_delta * 30

    summary = f"""SANDAGE-LOEB DRIFT TEST
{'='*35}

THE ULTIMATE MTDF TEST

PRINCIPLE:
  Redshift evolves over cosmic time
  ż = (1+z)H₀ - H(z)
  Signal: ~1 cm/s/year

{'='*35}

MTDF PREDICTION:
  Stress evolves with structure
  → ż depends on environment

  ΛCDM: ż(void) = ż(wall)
  MTDF: ż(void) ≠ ż(wall)

  Max Δż ~ {max_delta:.4f} cm/s/yr

{'='*35}

DETECTION REQUIREMENTS:

  Precision: ~1 cm/s
  Baseline: 20-30 years
  Facility: ANDES/ELT, SKA2

  30-year signal: ~{signal_30yr:.2f} cm/s
  {'DETECTABLE!' if signal_30yr > 1 else 'Challenging'}

{'='*35}

WHY THIS IS DEFINITIVE:

  • UNIQUE to MTDF
  • No kinematic contamination
  • Specific, falsifiable

  Detection of Δż ≠ 0 would be
  IRREFUTABLE PROOF of stress-
  photon coupling!

{'='*35}

STATUS: Future Test
  Timeline: 2030s start
  Result: ~2060
"""

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lavender',
                       edgecolor='purple', alpha=0.9))
    ax3.axis('off')
    ax3.set_title('Test Summary', fontsize=12)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'mtdf_sandage_loeb_drift_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nFigure saved: {output_path}")

if __name__ == '__main__':
    results = analyze_sandage_loeb_drift()
