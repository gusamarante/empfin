import numpy as np


def nearest_psd(sigma, strict_pd=True, delta=1e-10):
    # TODO documentation
    B = (sigma + sigma.T) / 2.0  # Symmetrize
    eigvals, eigvecs = np.linalg.eigh(B)  # Eigen-decompose
    eigvals_clipped = np.clip(eigvals, 0, None)# Clip negative eigenvalues to zero (projection onto PSD cone)
    B_psd = (eigvecs * eigvals_clipped) @ eigvecs.T   # Reconstruct (Q diag(lam) Q^T)
    B_psd = (B_psd + B_psd.T) / 2.0  # Re-symmetrize to clean numerical noise
    if strict_pd:
        # Ensure strictly PD by lifting the spectrum if needed
        min_eig = eigvals_clipped.min() if eigvals_clipped.size else 0.0
        eps = max(0.0, delta - min_eig)
        if eps > 0.0:
            B_psd = B_psd + eps * np.eye(B_psd.shape[0])
    return B_psd
