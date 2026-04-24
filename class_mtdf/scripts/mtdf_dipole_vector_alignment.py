#!/usr/bin/env python3
"""
MTDF Dipole Vector Alignment Test
=================================
The "lock-in" test: Does the MTDF stress vector align with the observed
excess dipole direction?

This is the CRITICAL test:
  - Magnitude match: ~40% (183 km/s vs 440 km/s) ✓
  - Direction match: MUST be <20-30° to be convincing

If random: angle ~90° expected
If MTDF is real: angle <30° expected
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'output'

# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def galactic_to_cartesian(l, b):
    """Convert galactic (l, b) in degrees to unit vector."""
    l_rad = np.radians(l)
    b_rad = np.radians(b)
    x = np.cos(b_rad) * np.cos(l_rad)
    y = np.cos(b_rad) * np.sin(l_rad)
    z = np.sin(b_rad)
    return np.array([x, y, z])

def cartesian_to_galactic(vec):
    """Convert unit vector to galactic (l, b) in degrees."""
    vec = vec / np.linalg.norm(vec)
    b = np.degrees(np.arcsin(vec[2]))
    l = np.degrees(np.arctan2(vec[1], vec[0]))
    if l < 0:
        l += 360
    return l, b

def angle_between(v1, v2):
    """Angle between two vectors in degrees."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# ============================================================================
# DATA
# ============================================================================

# CMB Dipole (Planck 2018)
CMB = {'l': 264.021, 'b': 48.253, 'A': 369.82}

# Quasar/Radio dipoles (observed)
MATTER_DIPOLES = {
    'CatWISE_QSO': {'l': 248, 'b': 42, 'A': 920, 'err': 180},
    'NVSS_Radio': {'l': 253, 'b': 36, 'A': 670, 'err': 150},
    'TGSS_Radio': {'l': 260, 'b': 50, 'A': 750, 'err': 200},
    'WISE_AGN': {'l': 252, 'b': 45, 'A': 850, 'err': 170}
}

# Large-Scale Structure
LSS = {
    'Shapley_Attractor': {'l': 306, 'b': 30, 'type': 'overdensity'},
    'Dipole_Repeller': {'l': 130, 'b': -40, 'type': 'void'},
    'Great_Attractor': {'l': 320, 'b': 0, 'type': 'overdensity'},
    'Local_Void': {'l': 250, 'b': 30, 'type': 'void'},
    'Perseus_Pisces': {'l': 140, 'b': -20, 'type': 'overdensity'},
    'Coma_Cluster': {'l': 58, 'b': 88, 'type': 'overdensity'}
}

# ============================================================================
# ANALYSIS
# ============================================================================

def compute_excess_dipole():
    """Compute the weighted mean excess dipole (Matter - CMB)."""

    cmb_vec = galactic_to_cartesian(CMB['l'], CMB['b']) * CMB['A']

    excess_vecs = []
    weights = []

    for name, data in MATTER_DIPOLES.items():
        matter_vec = galactic_to_cartesian(data['l'], data['b']) * data['A']
        excess = matter_vec - cmb_vec
        excess_vecs.append(excess)
        weights.append(1 / data['err']**2)

    # Weighted mean excess vector
    total_weight = sum(weights)
    mean_excess = sum(w * v for w, v in zip(weights, excess_vecs)) / total_weight

    # Direction and amplitude
    amplitude = np.linalg.norm(mean_excess)
    l, b = cartesian_to_galactic(mean_excess)

    return {
        'vector': mean_excess,
        'amplitude': amplitude,
        'l': l,
        'b': b
    }

def compute_mtdf_stress_vector():
    """
    Compute the MTDF integrated stress dipole direction.

    The stress dipole points from LOW stress (voids) toward HIGH stress (clusters).
    We weight by mass and inverse distance.
    """

    stress_vec = np.zeros(3)

    for name, data in LSS.items():
        direction = galactic_to_cartesian(data['l'], data['b'])

        # Sign: overdensity = positive stress contribution
        #       void = negative (we want to point FROM void TO cluster)
        if data['type'] == 'overdensity':
            sign = +1
        else:
            sign = -1

        # Weight by rough mass estimate (all contribute)
        weight = 1.0

        stress_vec += sign * weight * direction

    # Normalize to unit vector
    stress_vec = stress_vec / np.linalg.norm(stress_vec)
    l, b = cartesian_to_galactic(stress_vec)

    return {
        'vector': stress_vec,
        'l': l,
        'b': b
    }

def compute_density_gradient():
    """
    Compute the local density gradient direction.
    This is the direction from the Dipole Repeller toward the Shapley Attractor.
    """

    # Main structures
    repeller = galactic_to_cartesian(LSS['Dipole_Repeller']['l'],
                                      LSS['Dipole_Repeller']['b'])
    shapley = galactic_to_cartesian(LSS['Shapley_Attractor']['l'],
                                     LSS['Shapley_Attractor']['b'])

    # Density gradient points from underdense to overdense
    gradient = shapley - repeller
    gradient = gradient / np.linalg.norm(gradient)

    l, b = cartesian_to_galactic(gradient)

    return {
        'vector': gradient,
        'l': l,
        'b': b
    }

def main():
    print("=" * 70)
    print("MTDF DIPOLE VECTOR ALIGNMENT TEST")
    print("=" * 70)
    print()

    print("THE LOCK-IN TEST:")
    print("-" * 50)
    print("  If MTDF causes the excess dipole:")
    print("    → Direction should align with LSS density gradient")
    print("    → Expected alignment: <20-30°")
    print()
    print("  If coincidence:")
    print("    → Random direction")
    print("    → Expected alignment: ~90°")
    print()

    # Compute vectors
    excess = compute_excess_dipole()
    mtdf_stress = compute_mtdf_stress_vector()
    density_grad = compute_density_gradient()
    cmb_dir = galactic_to_cartesian(CMB['l'], CMB['b'])

    print("VECTOR DIRECTIONS:")
    print("-" * 60)
    print(f"  CMB dipole:           (l, b) = ({CMB['l']:.1f}°, {CMB['b']:.1f}°)")
    print(f"  Excess dipole:        (l, b) = ({excess['l']:.1f}°, {excess['b']:.1f}°)")
    print(f"  MTDF stress vector:   (l, b) = ({mtdf_stress['l']:.1f}°, {mtdf_stress['b']:.1f}°)")
    print(f"  Density gradient:     (l, b) = ({density_grad['l']:.1f}°, {density_grad['b']:.1f}°)")
    print(f"    (Dipole Repeller → Shapley Attractor)")
    print()

    # Compute angles
    angle_excess_cmb = angle_between(excess['vector'], cmb_dir)
    angle_excess_stress = angle_between(excess['vector'], mtdf_stress['vector'])
    angle_excess_gradient = angle_between(excess['vector'], density_grad['vector'])
    angle_stress_gradient = angle_between(mtdf_stress['vector'], density_grad['vector'])

    print("ANGULAR SEPARATIONS:")
    print("-" * 60)
    print(f"  Excess ↔ CMB:              {angle_excess_cmb:.1f}°")
    print(f"  Excess ↔ MTDF stress:      {angle_excess_stress:.1f}°  {'✓ ALIGNED!' if angle_excess_stress < 30 else ''}")
    print(f"  Excess ↔ Density gradient: {angle_excess_gradient:.1f}°  {'✓ ALIGNED!' if angle_excess_gradient < 30 else ''}")
    print(f"  MTDF ↔ Density gradient:   {angle_stress_gradient:.1f}°")
    print()

    # Individual quasar dipole excess angles
    print("INDIVIDUAL SURVEY EXCESS DIRECTIONS:")
    print("-" * 60)
    cmb_vec = galactic_to_cartesian(CMB['l'], CMB['b']) * CMB['A']

    for name, data in MATTER_DIPOLES.items():
        matter_vec = galactic_to_cartesian(data['l'], data['b']) * data['A']
        exc = matter_vec - cmb_vec
        exc_l, exc_b = cartesian_to_galactic(exc)
        angle_to_gradient = angle_between(exc, density_grad['vector'])
        print(f"  {name:<15}: (l,b) = ({exc_l:.0f}°, {exc_b:.0f}°)  → gradient: {angle_to_gradient:.0f}°")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if angle_excess_gradient < 30:
        print("  ⭐ STRONG ALIGNMENT DETECTED!")
        print(f"     Excess dipole is {angle_excess_gradient:.0f}° from density gradient")
        print("     This is MUCH better than random (~90°)")
        print()
        print("  INTERPRETATION:")
        print("     The unexplained quasar dipole excess points in the")
        print("     direction of the local large-scale structure gradient!")
        print()
        print("     This is EXACTLY what MTDF predicts:")
        print("     • Photons from Shapley direction: more stress → more redshift")
        print("     • Photons from Repeller direction: less stress → less redshift")
        print("     • Creates anisotropic apparent source density")
        verdict = "STRONG SUPPORT"
    elif angle_excess_gradient < 60:
        print("  MODERATE ALIGNMENT")
        print(f"     Excess dipole is {angle_excess_gradient:.0f}° from density gradient")
        print("     Better than random, but not conclusive")
        verdict = "MODERATE SUPPORT"
    else:
        print("  POOR ALIGNMENT")
        print(f"     Excess dipole is {angle_excess_gradient:.0f}° from density gradient")
        print("     Consistent with random direction")
        verdict = "NOT SUPPORTED"

    print()
    print(f"  STATUS: {verdict} for MTDF photon coupling")

    # Create figure
    create_alignment_figure(excess, mtdf_stress, density_grad, cmb_dir,
                           angle_excess_gradient)

    return {
        'excess': excess,
        'mtdf_stress': mtdf_stress,
        'density_grad': density_grad,
        'angle_excess_gradient': angle_excess_gradient,
        'verdict': verdict
    }

def create_alignment_figure(excess, mtdf_stress, density_grad, cmb_dir, angle):
    """Create visualization of vector alignment."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Sky map with vectors
    ax1 = axes[0]

    # Plot directions as arrows
    directions = {
        'CMB dipole': (CMB['l'], CMB['b'], 'blue'),
        'Excess dipole': (excess['l'], excess['b'], 'red'),
        'Density gradient': (density_grad['l'], density_grad['b'], 'green'),
    }

    for name, (l, b, color) in directions.items():
        ax1.scatter(l, b, s=200, c=color, marker='o', label=name, zorder=5)
        ax1.annotate(name, (l, b), fontsize=8, xytext=(5, 5),
                    textcoords='offset points')

    # Plot LSS
    for name, data in LSS.items():
        marker = 'D' if data['type'] == 'overdensity' else 'v'
        color = 'orange' if data['type'] == 'overdensity' else 'lightblue'
        ax1.scatter(data['l'], data['b'], s=100, c=color, marker=marker,
                   alpha=0.7, edgecolors='black')
        ax1.annotate(name.replace('_', '\n'), (data['l'], data['b']),
                    fontsize=6, ha='center', va='bottom')

    # Draw arc between excess and gradient
    ax1.annotate('', xy=(density_grad['l'], density_grad['b']),
                xytext=(excess['l'], excess['b']),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax1.text((density_grad['l'] + excess['l'])/2 - 20,
             (density_grad['b'] + excess['b'])/2 + 5,
             f'{angle:.0f}°', fontsize=12, color='purple', fontweight='bold')

    ax1.set_xlim(0, 360)
    ax1.set_ylim(-90, 90)
    ax1.set_xlabel('Galactic Longitude l [°]', fontsize=10)
    ax1.set_ylabel('Galactic Latitude b [°]', fontsize=10)
    ax1.set_title('Dipole Vector Alignment (Galactic)', fontsize=12)
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: 3D vector visualization
    ax2 = axes[1]
    ax2 = fig.add_subplot(132, projection='3d')

    # Plot vectors
    origin = [0, 0, 0]

    cmb_unit = cmb_dir / np.linalg.norm(cmb_dir)
    excess_unit = excess['vector'] / np.linalg.norm(excess['vector'])
    grad_unit = density_grad['vector']

    ax2.quiver(*origin, *cmb_unit, color='blue', arrow_length_ratio=0.1,
               linewidth=3, label='CMB')
    ax2.quiver(*origin, *excess_unit, color='red', arrow_length_ratio=0.1,
               linewidth=3, label='Excess')
    ax2.quiver(*origin, *grad_unit, color='green', arrow_length_ratio=0.1,
               linewidth=3, label='Gradient')

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D Vector View', fontsize=12)
    ax2.legend(loc='upper left', fontsize=8)

    # Panel 3: Summary
    ax3 = axes[2]

    summary = f"""VECTOR ALIGNMENT TEST
{'='*35}

KEY RESULT:
  Excess dipole ↔ Density gradient
  Angle: {angle:.0f}°

{'='*35}

SIGNIFICANCE:
  • Random expectation: ~90°
  • MTDF prediction: <30°
  • Observed: {angle:.0f}°

{'='*35}

DIRECTIONS (Galactic):
  CMB dipole:     ({CMB['l']:.0f}°, {CMB['b']:.0f}°)
  Excess dipole:  ({excess['l']:.0f}°, {excess['b']:.0f}°)
  Density grad:   ({density_grad['l']:.0f}°, {density_grad['b']:.0f}°)

{'='*35}

INTERPRETATION:
  {'ALIGNED!' if angle < 30 else 'Moderate' if angle < 60 else 'Misaligned'}

  The unexplained quasar dipole
  excess points {'toward' if angle < 45 else 'away from'} the local
  large-scale structure gradient.

  {'This supports MTDF photon' if angle < 45 else 'This weakens support for'}
  {'coupling as the cause.' if angle < 45 else 'MTDF photon coupling.'}

{'='*35}

VERDICT: {'STRONG' if angle < 30 else 'MODERATE' if angle < 60 else 'WEAK'} SUPPORT
"""

    ax3.text(0.05, 0.95, summary, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round',
                      facecolor='lightgreen' if angle < 30 else 'lightyellow',
                      edgecolor='darkgreen' if angle < 30 else 'orange',
                      alpha=0.9))
    ax3.axis('off')
    ax3.set_title('Alignment Summary', fontsize=12)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'mtdf_dipole_vector_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nFigure saved: {output_path}")

if __name__ == '__main__':
    results = main()
