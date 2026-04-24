# Author: Ingo Mesche
# Affiliation: Independent Researcher, Malta
# Framework: MTDF V74

"""
CosmoPower emulator setup with TF 2.20 compatibility fix.

The .pkl model files are incompatible with TF 2.20+ due to removal of
tensorflow.python.training.tracking. We monkey-patch restore() to load
from .npz files instead, which contain the same data in a compatible format.

Models come in two types:
  - cosmopower_NN: direct NN (TT, EE) — output dim matches n_modes
  - cosmopower_PCAplusNN: PCA+NN (TE) — output dim = n_pcas, expanded via PCA matrix
"""

import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"


def _restore_nn_from_npz(self, filename):
    """Replace cosmopower_NN.restore() with npz-based loading."""
    data = np.load(filename + ".npz", allow_pickle=True)
    obj = data['arr_0'].item()

    self.W_ = [np.array(w, dtype=np.float32) for w in obj['weights_']]
    self.b_ = [np.array(b, dtype=np.float32) for b in obj['biases_']]
    self.alphas_ = [np.array(a, dtype=np.float32) for a in obj['alphas_']]
    self.betas_ = [np.array(b, dtype=np.float32) for b in obj['betas_']]

    self.parameters_mean_ = np.array(obj['param_train_mean'], dtype=np.float32)
    self.parameters_std_ = np.array(obj['param_train_std'], dtype=np.float32)
    self.features_mean_ = np.array(obj['feature_train_mean'], dtype=np.float32)
    self.features_std_ = np.array(obj['feature_train_std'], dtype=np.float32)

    self.n_parameters = int(obj['n_parameters'])
    self.parameters = list(obj['parameters'])
    self.n_modes = int(obj['n_modes'])
    self.modes = np.array(obj['modes'])
    self.n_hidden = list(obj['n_hidden'])
    self.n_layers = int(obj['n_layers'])
    self.architecture = list(obj['architecture'])


def _restore_pca_from_npz(self, filename):
    """Replace cosmopower_PCAplusNN.restore() with npz-based loading."""
    data = np.load(filename + ".npz", allow_pickle=True)
    obj = data['arr_0'].item()

    self.W_ = [np.array(w, dtype=np.float32) for w in obj['weights_']]
    self.b_ = [np.array(b, dtype=np.float32) for b in obj['biases_']]
    self.alphas_ = [np.array(a, dtype=np.float32) for a in obj['alphas_']]
    self.betas_ = [np.array(b, dtype=np.float32) for b in obj['betas_']]

    self.parameters_mean_ = np.array(obj['param_train_mean'], dtype=np.float32)
    self.parameters_std_ = np.array(obj['param_train_std'], dtype=np.float32)
    self.pca_mean_ = np.array(obj['pca_mean'], dtype=np.float32)
    self.pca_std_ = np.array(obj['pca_std'], dtype=np.float32)
    self.features_mean_ = np.array(obj['feature_train_mean'], dtype=np.float32)
    self.features_std_ = np.array(obj['feature_train_std'], dtype=np.float32)

    self.parameters = list(obj['parameters'])
    self.n_parameters = int(obj['n_parameters'])
    self.modes = np.array(obj['modes'])
    self.n_modes = int(obj['n_modes'])
    self.n_pcas = int(obj['n_pcas'])
    self.pca_transform_matrix_ = np.array(obj['pca_transform_matrix'], dtype=np.float32)
    self.n_hidden = list(obj['n_hidden'])
    self.n_layers = int(obj['n_layers'])
    self.architecture = list(obj['architecture'])


_patched = False


def patch_cosmopower():
    """Monkey-patch both NN and PCAplusNN restore methods."""
    global _patched
    if _patched:
        return
    from cosmopower import cosmopower_NN
    from cosmopower import cosmopower_PCAplusNN
    cosmopower_NN.restore = _restore_nn_from_npz
    cosmopower_PCAplusNN.restore = _restore_pca_from_npz
    _patched = True


def _is_pca_model(spectrum):
    """Check if a model file contains PCA keys."""
    npz_path = MODELS_DIR / f"{spectrum}_v1.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    obj = data['arr_0'].item()
    return 'n_pcas' in obj


def load_emulator(spectrum='TT'):
    """Load a CosmoPower emulator for the given spectrum type.

    Automatically detects whether the model is NN or PCAplusNN.

    Parameters
    ----------
    spectrum : str
        One of 'TT', 'TE', 'EE'

    Returns
    -------
    emulator : cosmopower_NN or cosmopower_PCAplusNN
    """
    patch_cosmopower()

    model_path = str(MODELS_DIR / f"{spectrum}_v1")

    if _is_pca_model(spectrum):
        from cosmopower import cosmopower_PCAplusNN
        return cosmopower_PCAplusNN(restore=True, restore_filename=model_path)
    else:
        from cosmopower import cosmopower_NN
        return cosmopower_NN(restore=True, restore_filename=model_path)


T_CMB = 2.7255e6  # CMB temperature in uK
T_CMB_SQ = T_CMB ** 2  # T_CMB^2 in uK^2 = 7.4284e12


def predict_dl(emulator, params_dict):
    """Generate D_l predictions in uK^2 from an emulator.

    For TT/EE (NN models): outputs log10(D_l / T_CMB^2), always positive.
    For TE (PCAplusNN models): outputs D_l / T_CMB^2 directly (can be negative).

    Parameters
    ----------
    emulator : cosmopower_NN or cosmopower_PCAplusNN
    params_dict : dict
        Keys must match emulator.parameters. Values are floats.

    Returns
    -------
    ells : ndarray
        Multipole values.
    dl : ndarray
        D_l = l(l+1)/(2pi) C_l in uK^2.
    """
    input_dict = {p: np.array([params_dict[p]], dtype=np.float32)
                  for p in emulator.parameters}

    raw = emulator.predictions_np(input_dict)

    # Detect model type: PCAplusNN (TE) outputs linear, NN (TT/EE) outputs log10
    is_pca = hasattr(emulator, 'pca_transform_matrix_')

    if is_pca:
        # TE: raw output is D_l / T_CMB^2 (linear, can be negative)
        dl = raw[0] * T_CMB_SQ
    else:
        # TT/EE: raw output is log10(D_l / T_CMB^2)
        dl = 10.0 ** raw[0] * T_CMB_SQ

    return emulator.modes, dl


# Planck 2018 best-fit LCDM parameters in CosmoPower's naming convention
PLANCK_BESTFIT = {
    'ln10^{10}A_s': 3.044,
    'n_s': 0.9649,
    'H0': 67.36,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'tau_reio': 0.0544,
}
