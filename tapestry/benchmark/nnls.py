"""Coverage-weighted Non-Negative Least Squares (NNLS) baseline.

Solves the deconvolution problem per-sample using scipy's NNLS with
coverage-based weighting to downweight markers with low or zero coverage.
"""

import numpy as np
from scipy.optimize import nnls


def run_weighted_nnls(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
) -> np.ndarray:
    """Estimate cell-type proportions via coverage-weighted NNLS.

    For each sample, solves::

        min_x || W @ (A @ x - b) ||^2  subject to x >= 0

    where W = diag(coverage), A = reference profiles, b = observed values.
    Results are normalised to sum to 1.

    Parameters
    ----------
    X : ndarray, shape (N, M)
        Observed methylation fractions per marker.
    coverage : ndarray, shape (N, M)
        Read counts per marker.
    reference_profiles : ndarray, shape (C, M)
        Reference methylation profiles (one row per cell type).

    Returns
    -------
    ndarray, shape (N, C)
        Estimated cell-type proportions (rows sum to 1).
    """
    n_samples = X.shape[0]
    n_cell_types = reference_profiles.shape[0]
    estimated = np.zeros((n_samples, n_cell_types))
    A = reference_profiles.T  # (M, C)

    for i in range(n_samples):
        w = coverage[i]
        b = X[i]
        valid = (w > 0) & np.isfinite(b)
        if not np.any(valid):
            continue
        A_w = A[valid] * w[valid, np.newaxis]
        b_w = b[valid] * w[valid]
        x, _ = nnls(A_w, b_w)
        total = x.sum()
        if total > 0:
            x /= total
        estimated[i] = x

    return estimated
