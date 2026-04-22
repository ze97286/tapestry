"""Scaled Total Least Squares (TLS) deconvolution with observation + atlas noise.

Classical weighted NNLS minimises ``||W (A x − b)||²`` with ``x ≥ 0``, treating
the atlas ``A`` as exact. In cfDNA deconvolution the atlas is noisy — each
reference methylation value is a finite-sample estimate. Standard NNLS
effectively assumes infinite confidence in the atlas and so over-weights
markers for which the atlas entry happens to be imprecise.

This module solves the **errors-in-variables / scaled TLS** formulation:

    min  Σ_m  (b_m − A_m x)² / (σ_b_m² + σ_A_m² · ‖x‖²)
    s.t. x ≥ 0,  Σx = 1

where ``σ_b_m²`` is per-marker observation variance (binomial, from coverage)
and ``σ_A_m²`` is per-marker atlas uncertainty. The denominator is the
marginal variance of the residual ``b_m − A_m x`` when both ``b`` and ``A``
carry independent Gaussian noise; the ratio is a correctly-weighted squared
residual under that noise model.

Solved via iteratively-reweighted NNLS (IRLS): fix weights → NNLS → recompute
weights → iterate. Converges in <20 iterations on this problem shape.

References
----------
- Van Huffel & Vandewalle, 1991. *The Total Least Squares Problem*.
- Paige & Strakoš, 2002. *Scaled total least squares fundamentals*.
"""

import numpy as np
from scipy.optimize import nnls


def _observation_variance(fraction: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Per-marker variance of the empirical U-fraction estimate.

    Uses the Beta(u+1, m+1) posterior mean under a Uniform(0, 1) prior — i.e.
    Laplace-smoothed p̂ = (u + 1)/(c + 2) — so variance remains meaningful even
    when raw ``p̂ ∈ {0, 1}``. Plain plug-in ``p̂(1-p̂)/c`` would return zero at
    the boundaries and give those markers infinite IRLS weight.

    Markers with zero coverage get infinite variance and drop out of the fit.
    """
    cov_safe = np.maximum(coverage, 1)
    u = fraction * coverage
    p_smooth = (u + 1.0) / (coverage + 2.0)
    var = p_smooth * (1.0 - p_smooth) / cov_safe
    var = np.where(coverage > 0, var, np.inf)
    return var


def run_tls_deconvolution(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    atlas_sigma: np.ndarray | float = 0.05,
    max_iter: int = 25,
    tol: float = 1e-4,
) -> np.ndarray:
    """Estimate cell-type proportions via scaled TLS (errors-in-variables NNLS).

    Parameters
    ----------
    X : (N, M) observed U-fractions per marker.
    coverage : (N, M) per-marker read counts.
    reference_profiles : (C, M) atlas (one row per cell type).
    atlas_sigma : float or (M,) per-marker atlas uncertainty (std, in fraction
        units). Defaults to 0.05 — conservative 5-point uncertainty on each
        atlas entry. Pass a per-marker array if you have quality metrics.
    max_iter : IRLS cap.
    tol : convergence threshold on ‖Δx‖.

    Returns
    -------
    (N, C) proportions with rows summing to 1.
    """
    N, M = X.shape
    C = reference_profiles.shape[0]
    A = reference_profiles.T  # (M, C)

    sigma_A_sq = np.asarray(atlas_sigma, dtype=np.float64) ** 2
    if sigma_A_sq.ndim == 0:
        sigma_A_sq = np.full(M, float(sigma_A_sq))

    estimates = np.zeros((N, C))

    for i in range(N):
        b = X[i].astype(np.float64)
        cov = coverage[i].astype(np.float64)
        sigma_b_sq = _observation_variance(b, cov)

        # Initial x from plain coverage-weighted NNLS.
        w = cov
        A_w = A * w[:, np.newaxis]
        b_w = b * w
        zero_cov = cov == 0
        if zero_cov.any():
            A_w[zero_cov] = 0
            b_w[zero_cov] = 0
        x_hat, _ = nnls(A_w, b_w)
        if x_hat.sum() > 0:
            x_hat = x_hat / x_hat.sum()
        else:
            x_hat = np.full(C, 1.0 / C)

        # IRLS loop: reweight by the TLS residual variance.
        for _ in range(max_iter):
            x_prev = x_hat.copy()
            var = sigma_b_sq + sigma_A_sq * np.sum(x_hat ** 2)
            w = 1.0 / np.sqrt(var)
            w = np.where(np.isfinite(w), w, 0.0)

            A_w = A * w[:, np.newaxis]
            b_w = b * w
            x_hat, _ = nnls(A_w, b_w)

            total = x_hat.sum()
            if total > 0:
                x_hat = x_hat / total
            else:
                x_hat = x_prev
                break

            if np.linalg.norm(x_hat - x_prev) < tol:
                break

        estimates[i] = x_hat

    return estimates
