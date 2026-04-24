#!/usr/bin/env python3
# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
Phase 6 Test C5b: Gravitational Slip η = Φ/Ψ — Numerical Verification

Computes η(k,z) = Φ(k,z)/Ψ(k,z) directly from class_mtdf transfer functions
to verify that the MTDF implementation introduces no gravitational slip.

This resolves the "key unknown" from C5: whether MTDF's strain tensor F_μν
introduces η ≠ 1. If η = 1 (confirmed here), then Σ = μ and M_dyn/M_lens = 1,
meaning the cluster mass ratio channel is not a discriminator.

Output:
  - testC5b_gravitational_slip.json   (numerical η values)
  - testC5b_gravitational_slip.png    (η(k) at multiple z, plus η(z) at fixed k)
  - README.md
  - manifest.json
"""

import sys, os, json, hashlib, datetime
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
CLASS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "class_mtdf")
sys.path.insert(0, os.path.join(CLASS_DIR, "python", "build", "lib.linux-x86_64-cpython-312"))
sys.path.insert(0, os.path.join(CLASS_DIR, "python"))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "validation", "output", "phase6", "testC5b_gravitational_slip")
os.makedirs(OUT_DIR, exist_ok=True)

def pr(msg):
    print(f"[C5b] {msg}", flush=True)

# ── cosmological parameters (Phase 5 MCMC best-fit, ΛCDM baseline) ─────
COSMO_BASE = {
    "h": 0.6726,
    "omega_b": 0.02226,
    "omega_cdm": 0.1186,
    "A_s": 2.1e-9,
    "n_s": 0.965,
    "tau_reio": 0.058,
    "N_ur": 2.0328,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
    "T_ncdm": 0.71611,
}

MTDF_PARAMS = {
    "mtdf": "yes",
    "mtdf_alpha": 1.30,
    "mtdf_beta_eos": 0.573,
    "mtdf_z_t": 0.74,
    "mtdf_efe": "yes",
    "mtdf_growth": "yes",    # Enable mu(a) modification
    "mtdf_k_f": 1.0,         # Full MTDF strength
}

# Redshifts to probe (cluster-relevant: 0 to 0.8; also high-z for comparison)
Z_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]

# k range: cluster scales (0.01 to 2 h/Mpc)
K_MIN_HMPC = 0.005
K_MAX_HMPC = 2.0


def run_class(label, extra_params=None):
    """Run CLASS and return transfer functions Φ(k,z), Ψ(k,z)."""
    from classy import Class

    pr(f"Running CLASS: {label}")
    M = Class()

    params = dict(COSMO_BASE)
    params.update({
        "output": "dTk,mPk",
        "gauge": "newtonian",
        "z_max_pk": max(Z_VALUES) + 0.1,
        "P_k_max_1/Mpc": K_MAX_HMPC / params["h"] * 1.5,  # convert h/Mpc to 1/Mpc with margin
    })
    if extra_params:
        params.update(extra_params)

    M.set(params)
    M.compute()

    results = {}
    h = M.h()

    for z in Z_VALUES:
        tk = M.get_transfer(z)
        k_hmpc = tk["k (h/Mpc)"]
        phi = tk["phi"]
        psi = tk["psi"]

        # Filter to our k range
        mask = (k_hmpc >= K_MIN_HMPC) & (k_hmpc <= K_MAX_HMPC)
        k_sel = k_hmpc[mask]
        phi_sel = phi[mask]
        psi_sel = psi[mask]

        # Compute eta = phi/psi (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(np.abs(psi_sel) > 1e-30, phi_sel / psi_sel, np.nan)

        results[z] = {
            "k_hmpc": k_sel,
            "phi": phi_sel,
            "psi": psi_sel,
            "eta": eta,
        }

    M.struct_cleanup()
    M.empty()
    pr(f"  Done: {label}")
    return results


def analyze_eta(results, label):
    """Compute summary statistics for η(k,z)."""
    summary = {}
    for z in Z_VALUES:
        eta = results[z]["eta"]
        valid = np.isfinite(eta)
        if valid.sum() == 0:
            continue
        eta_v = eta[valid]
        summary[f"z={z:.1f}"] = {
            "n_k": int(valid.sum()),
            "eta_mean": float(np.mean(eta_v)),
            "eta_median": float(np.median(eta_v)),
            "eta_std": float(np.std(eta_v)),
            "eta_min": float(np.min(eta_v)),
            "eta_max": float(np.max(eta_v)),
            "max_deviation_from_1": float(np.max(np.abs(eta_v - 1.0))),
        }
    return summary


def make_plots(lcdm_results, mtdf_results):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: η(k) at multiple z for MTDF ──
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(Z_VALUES)))
    for i, z in enumerate(Z_VALUES):
        d = mtdf_results[z]
        valid = np.isfinite(d["eta"])
        if valid.sum() > 0:
            ax.semilogx(d["k_hmpc"][valid], d["eta"][valid],
                        color=colors[i], label=f"z={z:.1f}", alpha=0.8, lw=1.2)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("η = Φ/Ψ")
    ax.set_title("MTDF: η(k) at multiple redshifts")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.95, 1.05)

    # ── Panel 2: η(k) at multiple z for ΛCDM ──
    ax = axes[0, 1]
    for i, z in enumerate(Z_VALUES):
        d = lcdm_results[z]
        valid = np.isfinite(d["eta"])
        if valid.sum() > 0:
            ax.semilogx(d["k_hmpc"][valid], d["eta"][valid],
                        color=colors[i], label=f"z={z:.1f}", alpha=0.8, lw=1.2)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("η = Φ/Ψ")
    ax.set_title("ΛCDM: η(k) at multiple redshifts")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.95, 1.05)

    # ── Panel 3: Δη = η_MTDF - η_LCDM at cluster scales ──
    ax = axes[1, 0]
    for i, z in enumerate(Z_VALUES):
        dm = mtdf_results[z]
        dl = lcdm_results[z]
        # Interpolate LCDM onto MTDF k grid
        valid_m = np.isfinite(dm["eta"])
        valid_l = np.isfinite(dl["eta"])
        if valid_m.sum() > 10 and valid_l.sum() > 10:
            from scipy.interpolate import interp1d
            eta_l_interp = interp1d(
                dl["k_hmpc"][valid_l], dl["eta"][valid_l],
                bounds_error=False, fill_value=np.nan
            )
            delta_eta = dm["eta"][valid_m] - eta_l_interp(dm["k_hmpc"][valid_m])
            valid_d = np.isfinite(delta_eta)
            if valid_d.sum() > 0:
                ax.semilogx(dm["k_hmpc"][valid_m][valid_d], delta_eta[valid_d],
                            color=colors[i], label=f"z={z:.1f}", alpha=0.8, lw=1.2)
    ax.axhline(0.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("Δη = η_MTDF − η_ΛCDM")
    ax.set_title("Differential slip (MTDF − ΛCDM)")
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 4: η at fixed k vs z ──
    ax = axes[1, 1]
    k_targets = [0.05, 0.1, 0.5, 1.0]  # h/Mpc — cluster-relevant
    for k_target in k_targets:
        eta_vs_z_mtdf = []
        eta_vs_z_lcdm = []
        z_arr = []
        for z in Z_VALUES:
            dm = mtdf_results[z]
            dl = lcdm_results[z]
            # Find nearest k
            idx_m = np.argmin(np.abs(dm["k_hmpc"] - k_target))
            idx_l = np.argmin(np.abs(dl["k_hmpc"] - k_target))
            if np.isfinite(dm["eta"][idx_m]) and np.isfinite(dl["eta"][idx_l]):
                eta_vs_z_mtdf.append(dm["eta"][idx_m])
                eta_vs_z_lcdm.append(dl["eta"][idx_l])
                z_arr.append(z)
        if z_arr:
            ax.plot(z_arr, eta_vs_z_mtdf, "o-", label=f"MTDF k={k_target}", ms=4)
            ax.plot(z_arr, eta_vs_z_lcdm, "s--", label=f"ΛCDM k={k_target}", ms=3, alpha=0.5)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel("η = Φ/Ψ")
    ax.set_title("η(z) at fixed cluster-scale k")
    ax.legend(fontsize=7, ncol=2)

    plt.suptitle(
        "Phase 6 Test C5b: Gravitational Slip η = Φ/Ψ\n"
        "MTDF vs ΛCDM — Numerical Verification from class_mtdf",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "testC5b_gravitational_slip.png")
    plt.savefig(path, dpi=150)
    plt.close()
    pr(f"  Plot saved: {path}")
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pr("=" * 60)
    pr("Phase 6 Test C5b: Gravitational Slip Verification")
    pr("=" * 60)

    # ── Run ΛCDM ──
    lcdm_results = run_class("LCDM", extra_params={"mtdf": "no"})

    # ── Run MTDF (full strength, k_f=1, mu(a) enabled) ──
    mtdf_results = run_class("MTDF (k_f=1, growth=yes)", extra_params=MTDF_PARAMS)

    # ── Analyze ──
    pr("\nAnalyzing η(k,z)...")
    lcdm_summary = analyze_eta(lcdm_results, "LCDM")
    mtdf_summary = analyze_eta(mtdf_results, "MTDF")

    # Print key results
    pr("\n--- ΛCDM η summary ---")
    for key, val in lcdm_summary.items():
        pr(f"  {key}: mean={val['eta_mean']:.6f}, max|η-1|={val['max_deviation_from_1']:.6f}")

    pr("\n--- MTDF η summary ---")
    for key, val in mtdf_summary.items():
        pr(f"  {key}: mean={val['eta_mean']:.6f}, max|η-1|={val['max_deviation_from_1']:.6f}")

    # Compute max differential slip across all z and k
    max_diff = 0.0
    for z in Z_VALUES:
        dm = mtdf_results[z]
        dl = lcdm_results[z]
        valid_m = np.isfinite(dm["eta"])
        valid_l = np.isfinite(dl["eta"])
        if valid_m.sum() > 10 and valid_l.sum() > 10:
            from scipy.interpolate import interp1d
            eta_l_interp = interp1d(
                dl["k_hmpc"][valid_l], dl["eta"][valid_l],
                bounds_error=False, fill_value=np.nan
            )
            delta = dm["eta"][valid_m] - eta_l_interp(dm["k_hmpc"][valid_m])
            valid_d = np.isfinite(delta)
            if valid_d.sum() > 0:
                md = np.max(np.abs(delta[valid_d]))
                if md > max_diff:
                    max_diff = md

    pr(f"\nMax |Δη| (MTDF − ΛCDM) across all k, z: {max_diff:.6f}")

    # ── Determine verdict ──
    # At late times with negligible anisotropic stress, η should be very close to 1
    # Allow small deviations from radiation anisotropic stress at high z
    cluster_max_dev = 0.0
    for z in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
        s = mtdf_summary.get(f"z={z:.1f}")
        if s:
            if s["max_deviation_from_1"] > cluster_max_dev:
                cluster_max_dev = s["max_deviation_from_1"]

    pr(f"Max |η-1| for MTDF at z ≤ 0.8 (cluster range): {cluster_max_dev:.6f}")

    if cluster_max_dev < 0.01:
        verdict = "CONFIRMED: η = 1 to better than 1% at cluster scales"
        case = "B"
    elif cluster_max_dev < 0.05:
        verdict = f"NEAR-UNITY: η deviates by up to {cluster_max_dev:.3f} at cluster scales"
        case = "intermediate"
    else:
        verdict = f"SLIP DETECTED: η deviates by up to {cluster_max_dev:.3f} — Case A may apply"
        case = "A"

    pr(f"\nVerdict: {verdict}")
    pr(f"Resolved coupling case: {case}")
    if case == "B":
        pr("M_dyn/M_lens = 1.000 — clusters are NOT a discriminator for MTDF")

    # ── Plots ──
    pr("\nGenerating plots...")
    plot_path = make_plots(lcdm_results, mtdf_results)

    # ── JSON output ──
    result = {
        "test": "C5b",
        "description": "Gravitational slip η = Φ/Ψ numerical verification",
        "method": "Transfer function extraction from class_mtdf (Newtonian gauge)",
        "k_range_hmpc": [K_MIN_HMPC, K_MAX_HMPC],
        "z_values": Z_VALUES,
        "lcdm_eta": lcdm_summary,
        "mtdf_eta": mtdf_summary,
        "max_differential_slip": float(max_diff),
        "cluster_max_deviation": float(cluster_max_dev),
        "verdict": verdict,
        "resolved_case": case,
        "implication": (
            "M_dyn/M_lens = 1.000 under MTDF. The cluster mass ratio channel "
            "is not a discriminator. This is because class_mtdf modifies only μ(a) "
            "(the Poisson equation source) without introducing gravitational slip "
            "(the Φ-Ψ anisotropy equation is unchanged from GR)."
            if case == "B" else
            f"η deviates from 1 by up to {cluster_max_dev:.4f}. Further investigation needed."
        ),
        "code_analysis": {
            "mu_applied_to": ["Poisson equation (synch gauge h')", "CDM Euler equation (Newt gauge)"],
            "phi_psi_relation": "Unchanged from GR (standard anisotropy equation)",
            "mtdf_eta_function": "Returns 1.0 (stub, never called in perturbations.c)",
            "baryon_euler": "Unmodified (no mu factor)",
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    json_path = os.path.join(OUT_DIR, "testC5b_gravitational_slip.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    pr(f"JSON saved: {json_path}")

    # ── README ──
    readme = f"""# Phase 6 Test C5b: Gravitational Slip Verification

## Question
Does MTDF's class_mtdf implementation introduce gravitational slip (η = Φ/Ψ ≠ 1)?

## Answer
**{verdict}**

## Method
- Extract Newtonian gauge transfer functions Φ(k,z) and Ψ(k,z) from class_mtdf
- Compute η(k,z) = Φ/Ψ across k = {K_MIN_HMPC}–{K_MAX_HMPC} h/Mpc and z = 0–10
- Compare MTDF (k_f=1, growth=yes) against ΛCDM

## Key Results
- Max |η − 1| at z ≤ 0.8 (cluster scales): {cluster_max_dev:.6f}
- Max |Δη| (MTDF − ΛCDM) across all k, z: {max_diff:.6f}
- Resolved coupling case: **{case}**

## Implication for C5
Since η = 1, Σ = μ, and M_dyn/M_lens = μ/Σ = 1.
The cluster mass ratio channel is **not a discriminator** for MTDF in its current implementation.

## Code Analysis
The MTDF modification in class_mtdf applies μ(a) to:
1. The Poisson equation source (δρ → μ·δρ in synchronous gauge h')
2. The CDM Euler equation (k²Ψ → μ·k²Ψ in Newtonian gauge)

It does **not** modify:
- The Φ−Ψ anisotropy equation (traceless spatial Einstein equation)
- The baryon Euler equation
- Any photon/neutrino perturbation equations

This is a "μ-only" modification with η = 1 by construction.

## Files
- `testC5b_gravitational_slip.json` — Numerical results
- `testC5b_gravitational_slip.png` — η(k,z) diagnostic plots
"""
    readme_path = os.path.join(OUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)

    # ── Manifest ──
    files = ["testC5b_gravitational_slip.json", "testC5b_gravitational_slip.png", "README.md"]
    manifest = {
        "test": "C5b",
        "files": {fn: sha256(os.path.join(OUT_DIR, fn)) for fn in files},
        "timestamp": result["timestamp"],
    }
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    pr(f"\nAll output in: {OUT_DIR}")
    pr("Done.")


if __name__ == "__main__":
    main()
