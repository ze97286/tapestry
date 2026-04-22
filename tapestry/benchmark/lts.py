"""Least Trimmed Squares (LTS) deconvolution with concentration-step refinement.

Rousseeuw's LTS (1984) replaces the sum of *all* squared residuals with the
sum of the smallest ``h`` — effectively identifying and ignoring the markers
whose atlas entries most disagree with the rest of the fit. Breakdown point
up to ``(N − h) / N``: tolerates that fraction of arbitrarily-bad atlas
entries without degrading the estimate.

The FAST-LTS algorithm (Rousseeuw & Van Driessen 2006) alternates between:

  1. **C-step (concentration)** — given a candidate proportion vector, select
     the ``h`` markers with smallest weighted squared residual as the new
     "inlier" set.
  2. **Refit** — re-estimate proportions on that inlier set.

The inlier set is the combinatorial variable; the proportion vector is the
continuous variable. Iterating halves the trimmed sum of squares at each
step and converges to a local optimum of the LTS objective. Multiple random
starts protect against local optima in principle, but on this problem the
NNLS initialisation plus 10 C-steps is nearly always sufficient.

References
----------
- Rousseeuw, 1984. *Least median of squares regression*. JASA.
- Rousseeuw & Van Driessen, 2006. *Computing LTS regression for large data
  sets*. Data Mining and Knowledge Discovery.
"""

import numpy as np
from scipy.optimize import nnls


def _weighted_nnls_solve(A: np.ndarray, b: np.ndarray, weights: np.ndarray):
    """Coverage-weighted NNLS returning a simplex-normalised proportion vector."""
    C = A.shape[1]
    w = np.sqrt(np.maximum(weights, 0.0))
    Aw = A * w[:, np.newaxis]
    bw = b * w
    x, _ = nnls(Aw, bw)
    total = x.sum()
    if total > 0:
        return x / total
    return np.full(C, 1.0 / C)


def _lts_single(
    b: np.ndarray,
    cov: np.ndarray,
    A: np.ndarray,
    trim_frac: float,
    max_c_steps: int,
    n_starts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Solve LTS for one sample via the FAST-LTS C-step iteration.

    Returns the proportion vector with the lowest trimmed squared residual
    across ``n_starts`` initialisations.
    """
    M, C = A.shape
    valid_markers = np.where(cov > 0)[0]
    if len(valid_markers) < 2 * C:
        # Not enough data; fall back to plain NNLS on available markers.
        return _weighted_nnls_solve(A[valid_markers], b[valid_markers], cov[valid_markers])

    M_valid = len(valid_markers)
    h = max(int(round(M_valid * (1.0 - trim_frac))), 2 * C)
    h = min(h, M_valid)

    best_x = None
    best_score = np.inf

    for start in range(n_starts):
        if start == 0:
            # Deterministic NNLS start on all valid markers.
            x = _weighted_nnls_solve(A[valid_markers], b[valid_markers], cov[valid_markers])
        else:
            # Random subset start for multi-modal protection.
            subset = rng.choice(valid_markers, size=min(4 * C, M_valid), replace=False)
            x = _weighted_nnls_solve(A[subset], b[subset], cov[subset])

        prev_inliers: set[int] | None = None
        for _ in range(max_c_steps):
            pred = A @ x
            # Coverage-weighted squared residual per marker — zero-cov markers
            # are placed at infinity so they're never selected as inliers.
            resid = (b - pred) ** 2 * cov
            scored = np.where(cov > 0, resid, np.inf)
            inlier_idx = np.argpartition(scored, h - 1)[:h]
            inliers = set(inlier_idx.tolist())
            if prev_inliers is not None and inliers == prev_inliers:
                break
            prev_inliers = inliers

            x = _weighted_nnls_solve(
                A[inlier_idx], b[inlier_idx], cov[inlier_idx],
            )

        # Score this start by its trimmed squared-residual sum.
        pred = A @ x
        scored = np.where(cov > 0, (b - pred) ** 2 * cov, np.inf)
        inlier_idx = np.argpartition(scored, h - 1)[:h]
        score = scored[inlier_idx].sum()

        if score < best_score:
            best_score = score
            best_x = x

    return best_x if best_x is not None else np.full(C, 1.0 / C)


def run_lts_deconvolution(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    trim_frac: float = 0.15,
    max_c_steps: int = 10,
    n_starts: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Per-sample LTS deconvolution.

    Parameters
    ----------
    X : (N, M) observed U-fractions per marker.
    coverage : (N, M) per-marker read counts.
    reference_profiles : (C, M) atlas (one row per cell type).
    trim_frac : float in [0, 0.5] — fraction of markers treated as potential
        atlas outliers. 0.15 ≈ "assume 15% of atlas entries are unreliable".
        Classical LTS convention is 0.25–0.5 for high-breakdown estimation.
    max_c_steps : cap on concentration-step iterations per start.
    n_starts : number of random initialisations per sample. 1 is deterministic
        (NNLS warm-start only); >1 adds random-subset starts for local-optima
        protection.
    seed : RNG seed for the random-subset starts.

    Returns
    -------
    (N, C) proportions with rows summing to 1.
    """
    N = X.shape[0]
    C = reference_profiles.shape[0]
    A = reference_profiles.T  # (M, C)
    rng = np.random.default_rng(seed)

    results = np.zeros((N, C))
    for i in range(N):
        results[i] = _lts_single(
            X[i].astype(np.float64),
            coverage[i].astype(np.float64),
            A,
            trim_frac=trim_frac,
            max_c_steps=max_c_steps,
            n_starts=n_starts,
            rng=rng,
        )
    return results
