import numpy as np
from numpy.linalg import eigh


def nearest_psd(C):
    C = (C + C.T) / 2
    w, v = eigh(C)
    w[w < 0] = 0
    C_psd = (v * w) @ v.T  # same as v @ np.diag(w) @ v.T but faster
    C_psd = (C_psd + C_psd.T) / 2

    if (eigh(C_psd)[0] < 0).any():
        # If the projection is not enough, add a scale-aware regularization
        eps = 1e-12 * np.trace(C_psd) / C_psd.shape[0]   # scale-aware
        C_psd = C_psd + eps * np.eye(C_psd.shape[0])

    return C_psd