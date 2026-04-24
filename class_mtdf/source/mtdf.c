/**
 * MTDF (Mesche Tensor Dynamics Framework) implementation for CLASS
 *
 * This file implements the MTDF cosmological model functions for integration
 * into the CLASS Boltzmann solver.
 *
 * Physical basis:
 * - The MTDF stress tensor field contributes early field energy (EFE) near
 *   matter-radiation equality, reducing the sound horizon by ~0.03%
 * - At late times, the field modifies gravitational growth via μ(a)
 * - Combined effect maintains CMB consistency
 *
 * Key parameters (all derived from 3 fundamentals: α, β_eos, z_t):
 * - λ_MTDF = (1-β_eos)² / (1+α) ≈ 0.079
 * - f_kick = λ_MTDF / 24 ≈ 0.0033 (0.33% energy injection)
 *
 * First-principles derivation of f_kick:
 *   f_kick = γ_int × λ_MTDF / 2, where γ_int = 1/12
 *   γ_int = 1/12 from three geometric factors:
 *     - 1/3: tensor-to-scalar projection of stress field
 *     - 1/2: incomplete transition response at equality
 *     - 1/2: sound horizon time-averaging
 *   Thus: f_kick = (1/12) × λ_MTDF / 2 = λ_MTDF / 24
 *
 * Equivalently, in the old C_MTDF parameterization:
 *   f_kick = C_MTDF × Ω_Λ × λ_MTDF
 *   with C_MTDF = 1/(24 × Ω_Λ) ≈ 0.06
 *   (replaces the incorrect back-calculated value of 0.61)
 *
 * k_f is an amplitude rescaling factor:
 *   k_f = 1.0 corresponds to the first-principles derived value
 *   CMB data should prefer k_f ≈ 1.0 with correct normalization
 *
 * See: MTDF_C_MTDF_Derivation_v2.md for full derivation
 *
 * Author: MTDF Implementation for CLASS
 */

#include "mtdf.h"
#include <math.h>
#include <stdio.h>

/**
 * Initialize MTDF parameters
 */
int mtdf_init(
    struct mtdf_parameters * pmtdf,
    double Omega_Lambda
) {
    /* Compute derived parameter λ_MTDF */
    pmtdf->lambda_mtdf = pow(1.0 - pmtdf->beta_eos, 2) / (1.0 + pmtdf->alpha);

    /* Early field energy fraction: first-principles derivation
     *
     * f_kick = λ_MTDF / 24
     *
     * Derivation: f_kick = γ_int × λ_MTDF / 2, where γ_int = 1/12
     * γ_int = 1/12 from three geometric factors:
     *   - 1/3: tensor-to-scalar projection of stress field
     *   - 1/2: incomplete transition response at equality
     *   - 1/2: sound horizon time-averaging
     * Thus: f_kick = (1/12) × λ_MTDF / 2 = λ_MTDF / 24
     *
     * For backward compatibility with diagnostics, we also compute:
     * C_MTDF = 1 / (24 × Ω_Λ) ≈ 0.06
     * (This replaces the old incorrect value of 0.61)
     */
    pmtdf->f_kick = pmtdf->lambda_mtdf / 24.0;
    pmtdf->C_mtdf = 1.0 / (24.0 * Omega_Lambda);  /* Derived, not hardcoded */

    /* k_f, z_peak, sigma_z are set in input.c - no need to override here */

    /* Late-time transition scale factor */
    pmtdf->a_t = 1.0 / (1.0 + pmtdf->z_t);

    /* Note: has_mtdf_efe and has_mtdf_growth are set in input.c
     * Do NOT overwrite them here */

    return _SUCCESS_;
}

/**
 * Early field energy density fraction Ω_EFE(z)
 *
 * Log-Gaussian profile:
 *   Ω_EFE(z) = f_kick * exp(-[ln(1+z) - ln(1+z_peak)]² / (2σ_z²))
 *
 * This peaks at z_peak ≈ 3400 (matter-radiation equality) and
 * redshifts away both to higher and lower z.
 */
double mtdf_omega_efe(
    double z,
    struct mtdf_parameters * pmtdf
) {
    if (pmtdf->has_mtdf_efe == _FALSE_) {
        return 0.0;
    }

    double ln1pz = log(1.0 + z);
    double ln1pz_peak = log(1.0 + pmtdf->z_peak);
    double delta_ln = ln1pz - ln1pz_peak;
    double sigma2 = pmtdf->sigma_z * pmtdf->sigma_z;

    double exponent = -delta_ln * delta_ln / (2.0 * sigma2);

    /* Apply k_f scaling factor for MCMC sampling */
    return pmtdf->k_f * pmtdf->f_kick * exp(exponent);
}

/**
 * Early field energy density ρ_EFE(z)
 */
double mtdf_rho_efe(
    double z,
    struct mtdf_parameters * pmtdf,
    double rho_crit
) {
    return mtdf_omega_efe(z, pmtdf) * rho_crit;
}

/**
 * Effective equation of state for EFE component
 *
 * Physical behavior:
 * - At z >> z_eq: w ≈ 1/3 (radiation-like, subdominant)
 * - At z ~ z_eq: w ≈ 0.5-0.6 (stiff fluid during injection)
 * - At z << z_eq: w → 0 or negative (but component is negligible)
 *
 * Interpolation formula:
 *   w_EFE = 1/3 + 0.2 * [1 - exp(-(z/z_eq)²)]
 *
 * This ensures the field redshifts away faster than matter after equality.
 */
double mtdf_w_efe(
    double z,
    struct mtdf_parameters * pmtdf
) {
    double z_eq = 3400.0;

    /* At very high z, return pure radiation EoS */
    if (z > 10.0 * z_eq) {
        return 1.0 / 3.0;
    }

    /* Smooth interpolation */
    double z_ratio = z / z_eq;
    double transition = 1.0 - exp(-z_ratio * z_ratio);

    /* w ranges from 1/3 at high z to ~0.53 at z_eq */
    return 1.0 / 3.0 + 0.2 * transition;
}

/**
 * Sound speed squared for EFE component
 *
 * The stress field has relativistic sound speed c_s² ≈ 1 during the
 * EFE phase to prevent clustering on sub-horizon scales.
 */
double mtdf_cs2_efe(
    double z,
    struct mtdf_parameters * pmtdf
) {
    /* Relativistic sound speed */
    return 1.0;
}

/**
 * Late-time gravitational coupling modification μ(a)
 *
 * μ_MTDF(a) = 1 + λ_MTDF * T(a/a_t)
 *
 * where T(x) = x^α / (1 + x^α) is a smooth transition function.
 *
 * This modifies the effective gravitational coupling for matter
 * perturbations, enhancing growth at late times (a > a_t).
 *
 * Physical interpretation:
 * - For a << a_t: T → 0, so μ → 1 (GR)
 * - For a >> a_t: T → 1, so μ → 1 + λ_MTDF ≈ 1.08
 * - Transition is smooth with characteristic scale a_t
 */
double mtdf_mu(
    double a,
    struct mtdf_parameters * pmtdf
) {
    if (pmtdf->has_mtdf_growth == _FALSE_) {
        return 1.0;
    }

    double x = a / pmtdf->a_t;
    double x_alpha = pow(x, pmtdf->alpha);
    double T = x_alpha / (1.0 + x_alpha);

    return 1.0 + pmtdf->lambda_mtdf * T;
}

/**
 * Derivative of μ(a) with respect to ln(a)
 *
 * d(μ)/d(ln a) = λ_MTDF * dT/d(ln a)
 *
 * where dT/d(ln a) = α * x^α / (1 + x^α)² = α * T * (1 - T)
 */
double mtdf_mu_prime(
    double a,
    struct mtdf_parameters * pmtdf
) {
    if (pmtdf->has_mtdf_growth == _FALSE_) {
        return 0.0;
    }

    double x = a / pmtdf->a_t;
    double x_alpha = pow(x, pmtdf->alpha);
    double T = x_alpha / (1.0 + x_alpha);

    /* dT/d(ln a) = α * T * (1 - T) */
    double dT_dlna = pmtdf->alpha * T * (1.0 - T);

    return pmtdf->lambda_mtdf * dT_dlna;
}

/**
 * Slip parameter η = Φ/Ψ modification
 *
 * In GR, η = 1. MTDF introduces a small slip at late times.
 * For the current implementation, we set η = 1 (no slip) as
 * a first approximation. This can be refined later.
 */
double mtdf_eta(
    double a,
    struct mtdf_parameters * pmtdf
) {
    /* No slip in current implementation */
    return 1.0;
}

/**
 * Print MTDF parameters for debugging
 */
void mtdf_print_parameters(
    struct mtdf_parameters * pmtdf
) {
    printf("\n");
    printf("=========================================================\n");
    printf("MTDF (Mesche Tensor Dynamics Framework) Parameters\n");
    printf("=========================================================\n");
    printf("\n");
    printf("Input parameters:\n");
    printf("  alpha      = %.4f  (stress-matter coupling)\n", pmtdf->alpha);
    printf("  beta_eos   = %.4f  (EoS transition parameter)\n", pmtdf->beta_eos);
    printf("  z_t        = %.4f  (transition redshift)\n", pmtdf->z_t);
    printf("\n");
    printf("Derived parameters:\n");
    printf("  lambda_MTDF = %.6f  = (1 - beta_eos)^2 / (1 + alpha)\n", pmtdf->lambda_mtdf);
    printf("  f_kick      = %.6f  = lambda_MTDF / 24  (first-principles)\n", pmtdf->f_kick);
    printf("  C_MTDF      = %.4f  = 1 / (24 * Omega_Lambda)  (derived)\n", pmtdf->C_mtdf);
    printf("  a_t         = %.6f  = 1 / (1 + z_t)\n", pmtdf->a_t);
    printf("\n");
    printf("EFE profile:\n");
    printf("  z_peak     = %.1f  (peak redshift)\n", pmtdf->z_peak);
    printf("  sigma_z    = %.2f  (width in ln(1+z))\n", pmtdf->sigma_z);
    printf("\n");
    printf("Features enabled:\n");
    printf("  Early Field Energy (EFE): %s\n", pmtdf->has_mtdf_efe ? "YES" : "NO");
    printf("  Late-time growth mu(a):   %s\n", pmtdf->has_mtdf_growth ? "YES" : "NO");
    printf("\n");
    printf("Expected effects:\n");
    printf("  Sound horizon reduction: ~%.2f%%\n", -0.5 * pmtdf->f_kick * 100.0);
    printf("  Equivalent Delta N_eff:  ~%.2f\n", pmtdf->f_kick * (8.0/7.0) * pow(11.0/4.0, 4.0/3.0));
    printf("  Late-time mu(a=1):       ~%.4f\n", mtdf_mu(1.0, pmtdf));
    printf("=========================================================\n");
    printf("\n");
}
