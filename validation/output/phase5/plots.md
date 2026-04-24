# Phase 5 Plot Descriptions

> **Author:** Ingo Mesche | **Affiliation:** Independent Researcher, Malta | **Framework:** MTDF V74


## phase5_kf_posterior.png

**Title:** Marginalised 1D posterior of k_f from the MTDF full Planck MCMC

- **x-axis:** k_f (MTDF coupling parameter), range [0, 5]
- **y-axis:** Posterior probability density (normalised)
- **Shaded regions:** 68% CI (dark) and 95% CI (light)
- **Vertical lines:** Posterior mean (solid), k_f = 0 and k_f = 1 reference lines (dashed)
- **Key reading:** The posterior is broad and approximately Gaussian, centred near k_f ~ 0.5. k_f = 1 (full MTDF) lies within the 95% credible interval. k_f = 0 (LCDM) is not strongly excluded and sits near the lower boundary, shaped by the non-negativity prior (k_f >= 0). Planck does not constrain k_f tightly, consistent with MTDF being a late-time modification.

## phase5_triangle.png

**Title:** Corner (triangle) plot of MTDF cosmological parameters from full Planck MCMC

- **Diagonal panels:** Marginalised 1D posteriors for each sampled parameter
- **Off-diagonal panels:** 2D marginalised posteriors showing 68% and 95% contour levels
- **Parameters shown:** logA, n_s, theta_s_100, omega_b, omega_cdm, tau_reio, k_f, H0, sigma8, Omega_m
- **Colour coding:** LCDM (blue) and MTDF (red) posteriors overlaid
- **Key reading:** Standard cosmological parameters (H0, omega_b, omega_cdm, n_s) are nearly identical between LCDM and MTDF. The sigma8 posterior is visibly shifted downward for MTDF. k_f shows no strong degeneracy with any other parameter, confirming that the MTDF coupling is not fine-tuned against nuisance or cosmological parameters.
