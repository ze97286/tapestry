"""Maximum Entropy Method (MEM) deconvolution.

Classical MEM formulation, used for decades in radio astronomy image
reconstruction (Gull & Skilling 1984) and NMR spectroscopy (Sibisi 1984).
Solves a constrained optimisation whose objective is the Shannon entropy of
the proportion vector, with the data appearing only as a chi-squared
consistency term:

    max  H(x) = −Σ x_i log(x_i)
    s.t. x ≥ 0,  Σ x = 1
         Σ_m [(b_m − A_m x)² / σ_m²] ≤ χ²_target

Equivalently (Lagrangian form), for a chosen ``lambda_reg``:

    min  λ · Σ_m [(b_m − A_m x)² / σ_m²]  −  H(x)
    s.t. x ≥ 0, Σ x = 1

H(x) is the canonical "non-informative" objective — it makes no assumption
about the proportion distribution and prefers the flattest x consistent with
the data. No training, no learned priors, per-sample inference.

References
----------
- Gull & Skilling, 1984. *Maximum entropy method in image processing*.
- Sibisi, 1984. *NMR spectral reconstruction as a maximum-entropy problem*.
- Skilling & Bryan, 1984. *Maximum entropy image reconstruction: general
  algorithm*.
"""

import numpy as np
from scipy.optimize import minimize


def _observation_variance(fraction: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Laplace-smoothed per-marker binomial variance of the empirical fraction.

    Returns ``inf`` for zero-coverage markers so they drop out of the fit.
    """
    cov_safe = np.maximum(coverage, 1)
    u = fraction * coverage
    p_smooth = (u + 1.0) / (coverage + 2.0)
    var = p_smooth * (1.0 - p_smooth) / cov_safe
    var = np.where(coverage > 0, var, np.inf)
    return var


def _mem_single(
    b: np.ndarray,
    cov: np.ndarray,
    A: np.ndarray,
    lambda_reg: float,
    max_iter: int = 200,
    ftol: float = 1e-9,
    eps: float = 1e-10,
) -> np.ndarray:
    """Solve MEM for a single sample via SLSQP."""
    C = A.shape[1]
    sigma_sq = _observation_variance(b, cov)
    inv_var = np.where(np.isfinite(sigma_sq) & (sigma_sq > 0), 1.0 / sigma_sq, 0.0)

    def objective(x):
        res = b - A @ x
        data = np.sum(res * res * inv_var)
        neg_H = np.sum(x * np.log(np.maximum(x, eps)))
        return lambda_reg * data + neg_H

    def objective_grad(x):
        res = b - A @ x
        data_g = -2.0 * lambda_reg * (A.T @ (res * inv_var))
        # d/dx [x log x] = log x + 1. At boundary x=eps we use log(eps).
        H_g = np.log(np.maximum(x, eps)) + 1.0
        return data_g + H_g

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                    "jac": lambda x: np.ones_like(x)}]
    bounds = [(eps, 1.0)] * C
    x0 = np.full(C, 1.0 / C)

    result = minimize(
        objective, x0, jac=objective_grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": ftol},
    )
    x = np.clip(result.x, 0.0, 1.0)
    total = x.sum()
    return x / total if total > 0 else np.full(C, 1.0 / C)


def run_mem_deconvolution(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    lambda_reg: float = 0.1,
    max_iter: int = 200,
) -> np.ndarray:
    """Estimate cell-type proportions for each sample via MEM.

    Parameters
    ----------
    X : (N, M) observed U-fractions per marker.
    coverage : (N, M) per-marker read counts.
    reference_profiles : (C, M) atlas (one row per cell type).
    lambda_reg : float — data/entropy tradeoff. Higher = fit data harder,
        lower = prefer flatter distribution. Classical MEM picks λ such that
        the chi-squared residual matches the expected value (M − C). A coarse
        starting estimate is ``λ ≈ log(C) / (M − C)``. Treat as a hyperparam.
    max_iter : SLSQP iteration cap per sample.

    Returns
    -------
    (N, C) proportions with rows summing to 1.
    """
    N = X.shape[0]
    C = reference_profiles.shape[0]
    A = reference_profiles.T  # (M, C)
    results = np.zeros((N, C))
    for i in range(N):
        results[i] = _mem_single(
            X[i].astype(np.float64),
            coverage[i].astype(np.float64),
            A,
            lambda_reg,
            max_iter=max_iter,
        )
    return results
