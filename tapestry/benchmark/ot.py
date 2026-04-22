"""Sliced Wasserstein / Optimal Transport deconvolution.

Hybrid objective — per-marker squared error anchored by a 1D Wasserstein term
on the coverage-weighted methylation distribution:

    min  (1 − α) · Σ_m w_m · (b_m − A_m x)²
       +   α   · W₁(sort(b, w), sort(A x, w))
    s.t. x ≥ 0,  Σ x = 1

At α = 0 this is coverage-weighted NNLS. At α = 1 it's pure 1D Wasserstein
matching, which is invariant to per-marker identity — robust to locally bad
atlas entries but underdetermined (many proportion vectors give the same
sorted distribution). Intermediate α blends: identity-preserving residual
with distributional robustness.

1D Wasserstein-1 between two equally-weighted empirical distributions of the
same size reduces to the mean absolute difference of their sorted values; with
coverage weights we use the quantile-function representation and integrate
|F_b⁻¹(q) − F_pred⁻¹(q)| over q.

References
----------
- Villani, 2008. *Optimal Transport: Old and New*.
- Bonneel et al., 2015. *Sliced and Radon Wasserstein Barycenters of Measures*.
- Peyré & Cuturi, 2019. *Computational Optimal Transport*.
"""

import numpy as np
from scipy.optimize import minimize


def _weighted_sorted_quantiles(
    values: np.ndarray, weights: np.ndarray, probs: np.ndarray,
) -> np.ndarray:
    """Weighted quantile function F⁻¹(q) evaluated at the given probabilities."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    total = w_sorted.sum()
    if total <= 0:
        return np.zeros_like(probs)
    cdf = np.cumsum(w_sorted) / total
    # np.searchsorted returns the leftmost index where cdf >= probs
    idx = np.searchsorted(cdf, probs, side="left")
    idx = np.clip(idx, 0, len(v_sorted) - 1)
    return v_sorted[idx]


def _w1_1d(
    b: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    n_quantiles: int = 64,
) -> float:
    """Coverage-weighted Wasserstein-1 distance between two 1D empirical distributions."""
    probs = (np.arange(n_quantiles) + 0.5) / n_quantiles
    q_b = _weighted_sorted_quantiles(b, weights, probs)
    q_pred = _weighted_sorted_quantiles(pred, weights, probs)
    return float(np.mean(np.abs(q_b - q_pred)))


def _w1_1d_subgrad(
    b: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    A: np.ndarray,
    n_quantiles: int = 64,
) -> np.ndarray:
    """Finite-difference subgradient of W₁(b, Ax) w.r.t. x.

    The 1D Wasserstein-1 term is piecewise linear in x; its derivative is a
    matching between observed and predicted quantile indices. We approximate
    it with centered finite differences here — simpler and robust for SLSQP.
    """
    C = A.shape[1]
    eps = 1e-4
    grad = np.zeros(C)
    for j in range(C):
        # x appears only through pred = A @ x; perturb pred directly along A[:, j]
        pred_plus = pred + eps * A[:, j]
        pred_minus = pred - eps * A[:, j]
        loss_plus = _w1_1d(b, pred_plus, weights, n_quantiles)
        loss_minus = _w1_1d(b, pred_minus, weights, n_quantiles)
        grad[j] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def _ot_single(
    b: np.ndarray,
    cov: np.ndarray,
    A: np.ndarray,
    alpha: float,
    n_quantiles: int = 64,
    max_iter: int = 100,
) -> np.ndarray:
    """Solve the hybrid SE + W₁ objective for one sample."""
    C = A.shape[1]
    weights = np.where(cov > 0, cov.astype(np.float64), 0.0)
    if weights.sum() <= 0:
        return np.full(C, 1.0 / C)

    def objective(x):
        pred = A @ x
        se = np.sum(weights * (b - pred) ** 2) / max(weights.sum(), 1.0)
        w1 = _w1_1d(b, pred, weights, n_quantiles)
        return (1.0 - alpha) * se + alpha * w1

    def objective_grad(x):
        pred = A @ x
        se_grad = -2.0 * A.T @ (weights * (b - pred)) / max(weights.sum(), 1.0)
        w1_grad = _w1_1d_subgrad(b, pred, weights, A, n_quantiles)
        return (1.0 - alpha) * se_grad + alpha * w1_grad

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                    "jac": lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0)] * C
    x0 = np.full(C, 1.0 / C)

    result = minimize(
        objective, x0, jac=objective_grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-8},
    )
    x = np.clip(result.x, 0.0, 1.0)
    total = x.sum()
    return x / total if total > 0 else np.full(C, 1.0 / C)


def run_ot_deconvolution(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    alpha: float = 0.3,
    n_quantiles: int = 64,
    max_iter: int = 100,
) -> np.ndarray:
    """Per-sample OT deconvolution with hybrid SE + W₁ objective.

    Parameters
    ----------
    X : (N, M) observed U-fractions per marker.
    coverage : (N, M) per-marker read counts.
    reference_profiles : (C, M) atlas (one row per cell type).
    alpha : float in [0, 1] — blend between NNLS (0) and pure W₁ matching (1).
    n_quantiles : resolution of 1D Wasserstein integral.
    max_iter : SLSQP cap.

    Returns
    -------
    (N, C) proportions with rows summing to 1.
    """
    N = X.shape[0]
    C = reference_profiles.shape[0]
    A = reference_profiles.T
    results = np.zeros((N, C))
    for i in range(N):
        results[i] = _ot_single(
            X[i].astype(np.float64),
            coverage[i].astype(np.float64),
            A,
            alpha=alpha,
            n_quantiles=n_quantiles,
            max_iter=max_iter,
        )
    return results
