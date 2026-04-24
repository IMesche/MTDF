/**
 * MTDF (Mesche Tensor Dynamics Framework) module for CLASS
 *
 * This module implements the MTDF cosmological model where a stress tensor
 * field replaces dark matter and dark energy. The model has been validated
 * against late-time observables (SNe, BAO, H(z), fσ₈, rotation curves).
 *
 * Key features:
 * - Early Field Energy (EFE) injection near matter-radiation equality
 * - Modified growth via μ(a) at late times
 * - Sound horizon reduction of ~0.03% at k_f=1.0
 *
 * First-principles EFE amplitude derivation:
 *   f_kick = λ_MTDF / 24 ≈ 0.0033 (0.33%)
 *   where λ_MTDF = (1-β_eos)²/(1+α) ≈ 0.079
 *   The factor 24 arises from γ_int=1/12 (three geometric suppressions)
 *   See: MTDF_C_MTDF_Derivation_v2.md
 *
 * Author: MTDF Implementation for CLASS
 * Reference: Mesche Tensor Dynamics Framework V71
 */

#ifndef __MTDF__
#define __MTDF__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MTDF parameter structure
 * Contains both input parameters and derived quantities
 */
struct mtdf_parameters {
    /* Input parameters (from .ini file) */
    double alpha;        /**< Stress-matter coupling, dimensionless (default: 1.30) */
    double beta_eos;     /**< EoS transition parameter, dimensionless (default: 0.573) */
    double z_t;          /**< Transition redshift (default: 0.74) */

    /* Derived parameters (computed from inputs) */
    double lambda_mtdf;  /**< (1 - beta_eos)^2 / (1 + alpha), ~0.079 */
    double f_kick;       /**< λ_MTDF/24, early field energy fraction ~0.0033 */
    double C_mtdf;       /**< 1/(24*Ω_Λ), derived ~0.06 (for diagnostics) */

    /* MCMC sampling parameter */
    double k_f;          /**< EFE amplitude scaling: f_kick_eff = k_f * f_kick (default: 1.0) */

    /* EFE profile parameters */
    double z_peak;       /**< Peak redshift for EFE injection, ~3400 */
    double sigma_z;      /**< Width in ln(1+z) space, ~0.5 */

    /* Late-time transition parameters */
    double a_t;          /**< Scale factor at transition, 1/(1+z_t) */

    /* Flags */
    short has_mtdf_efe;      /**< Include early field energy */
    short has_mtdf_growth;   /**< Include late-time μ(a) modification */
};

/**
 * Initialize MTDF parameters
 * Computes derived quantities from input parameters
 *
 * @param pmtdf Pointer to MTDF parameter structure
 * @param Omega_Lambda Dark energy density fraction (needed for f_kick)
 * @return _SUCCESS_ or _FAILURE_
 */
int mtdf_init(
    struct mtdf_parameters * pmtdf,
    double Omega_Lambda
);

/**
 * Early field energy density fraction Ω_EFE(z)
 * Log-Gaussian profile peaking near matter-radiation equality
 *
 * @param z Redshift
 * @param pmtdf Pointer to MTDF parameters
 * @return Ω_EFE(z) as fraction of critical density
 */
double mtdf_omega_efe(
    double z,
    struct mtdf_parameters * pmtdf
);

/**
 * Early field energy density ρ_EFE(z)
 *
 * @param z Redshift
 * @param pmtdf Pointer to MTDF parameters
 * @param rho_crit Critical density at z=0
 * @return ρ_EFE(z) in same units as rho_crit
 */
double mtdf_rho_efe(
    double z,
    struct mtdf_parameters * pmtdf,
    double rho_crit
);

/**
 * Effective equation of state for EFE component
 * Interpolates from w=1/3 at high z to w~0.5 near equality
 *
 * @param z Redshift
 * @param pmtdf Pointer to MTDF parameters
 * @return w_EFE(z)
 */
double mtdf_w_efe(
    double z,
    struct mtdf_parameters * pmtdf
);

/**
 * Sound speed squared for EFE component
 *
 * @param z Redshift
 * @param pmtdf Pointer to MTDF parameters
 * @return c_s^2
 */
double mtdf_cs2_efe(
    double z,
    struct mtdf_parameters * pmtdf
);

/**
 * Late-time gravitational coupling modification μ(a)
 * μ_MTDF(a) = 1 + λ_MTDF * T(a/a_t)
 * where T(x) = x^α / (1 + x^α)
 *
 * @param a Scale factor
 * @param pmtdf Pointer to MTDF parameters
 * @return μ(a) modification factor
 */
double mtdf_mu(
    double a,
    struct mtdf_parameters * pmtdf
);

/**
 * Derivative of μ(a) with respect to ln(a)
 * Needed for perturbation equations
 *
 * @param a Scale factor
 * @param pmtdf Pointer to MTDF parameters
 * @return d(μ)/d(ln a)
 */
double mtdf_mu_prime(
    double a,
    struct mtdf_parameters * pmtdf
);

/**
 * Slip parameter η = Φ/Ψ modification
 *
 * @param a Scale factor
 * @param pmtdf Pointer to MTDF parameters
 * @return η_MTDF(a)
 */
double mtdf_eta(
    double a,
    struct mtdf_parameters * pmtdf
);

/**
 * Print MTDF parameters for debugging
 *
 * @param pmtdf Pointer to MTDF parameters
 */
void mtdf_print_parameters(
    struct mtdf_parameters * pmtdf
);

#ifdef __cplusplus
}
#endif

#endif /* __MTDF__ */
