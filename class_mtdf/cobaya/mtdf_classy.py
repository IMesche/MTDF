"""
Custom Cobaya theory class for MTDF CLASS.
Wraps our custom classy with MTDF+EFE support.
"""

import numpy as np
from cobaya.theory import Theory
from cobaya.log import LoggedError

import classy


class MTDF_Classy(Theory):
    """
    Cobaya theory wrapper for MTDF CLASS.

    Supports the k_f scaling parameter for EFE amplitude.
    """

    # Parameters that this theory provides
    _provides = ['Cl', 'CAMBdata']  # We'll mimic CAMB interface for compatibility

    # Speed parameters for sampler tuning
    speed = 1

    def initialize(self):
        """Initialize the theory."""
        self.classy = classy.Class()
        self.lmax = self.provider.requirement('lmax') if hasattr(self, 'provider') else 2600

    def must_provide(self, **requirements):
        """Declare what this theory provides."""
        pass

    def get_requirements(self):
        """Return requirements for this theory."""
        return {}

    def calculate(self, state, want_derived=True, **params):
        """
        Calculate CMB power spectra.

        Parameters in state['params']:
          - omega_b, omega_cdm, H0, tau_reio, logA, n_s
          - mtdf_k_f (EFE scaling factor)
        """
        try:
            # Reset CLASS
            self.classy.struct_cleanup()
            self.classy.empty()

            # Get cosmological parameters
            omega_b = state['params'].get('omega_b')
            omega_cdm = state['params'].get('omega_cdm')
            H0 = state['params'].get('H0')
            tau_reio = state['params'].get('tau_reio')
            logA = state['params'].get('logA')
            n_s = state['params'].get('n_s')
            k_f = state['params'].get('mtdf_k_f', 1.0)

            # Convert logA to A_s
            A_s = np.exp(logA) * 1e-10

            # Build CLASS parameters
            params_class = {
                'output': 'tCl,pCl,lCl',
                'lensing': 'yes',
                'l_max_scalars': 2600,
                'omega_b': omega_b,
                'omega_cdm': omega_cdm,
                'H0': H0,
                'tau_reio': tau_reio,
                'A_s': A_s,
                'n_s': n_s,
                # MTDF parameters
                'mtdf': 'yes',
                'mtdf_efe': 'yes',
                'mtdf_growth': 'no',  # Disable late-time growth for CMB-only test
                'mtdf_k_f': k_f,
                # Fixed MTDF parameters
                'mtdf_alpha': 1.30,
                'mtdf_beta_eos': 0.573,
                'mtdf_z_t': 0.74,
            }

            self.classy.set(params_class)
            self.classy.compute()

            # Get lensed Cls
            cls = self.classy.lensed_cl(2600)

            # Convert to μK² (CLASS outputs in dimensionless units)
            T_cmb = 2.7255e6  # μK
            factor = T_cmb**2

            # Store in state for likelihoods
            state['Cl'] = {
                'ell': np.arange(len(cls['tt'])),
                'tt': cls['tt'] * factor,
                'ee': cls['ee'] * factor,
                'te': cls['te'] * factor,
                'bb': cls.get('bb', np.zeros_like(cls['tt'])) * factor,
                'pp': cls.get('pp', np.zeros_like(cls['tt'])) * factor,
            }

            # Also store derived parameters
            if want_derived:
                bg = self.classy.get_background()
                idx = np.argmin(np.abs(bg['z'] - 1089))
                state['derived'] = {
                    'r_s': bg['comov.snd.hrz.'][idx],
                    'H0': H0,
                }

            return True

        except Exception as e:
            self.log.error(f"CLASS computation failed: {e}")
            return False

    def get_Cl(self, ell_factor=False, units='muK2'):
        """Return Cls in format expected by likelihoods."""
        return self._current_state['Cl']

    def close(self):
        """Clean up."""
        if hasattr(self, 'classy'):
            self.classy.struct_cleanup()
            self.classy.empty()
