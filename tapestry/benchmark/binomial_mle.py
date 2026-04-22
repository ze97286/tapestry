"""Binomial-likelihood MLE deconvolution.

Maximises the binomial log-likelihood of observed methylation counts directly
— no least-squares approximation, no Gaussian noise assumption. For each
sample and each marker, the methylated count ``u_m`` is modelled as

    u_m | c_m, x ~ Binomial(c_m, p_m(x)),      p_m(x) = Σ_c x_c · A_{c,m}

and we find the proportion vector ``x`` on the simplex that maximises the
summed marker log-likelihoods:

    max_x  Σ_m  [ u_m · log p_m(x) + (c_m − u_m) · log(1 − p_m(x)) ]
    s.t.   x ≥ 0,  Σx = 1

The objective is concave in ``x`` on the open simplex (sum of concave
functions of affine arguments), so the MLE is unique. Solved per-sample via
SLSQP with analytical gradient; warm-started from the NNLS solution so
Newton-style steps converge in a few iterations.

References
----------
- Classical MLE for binomial mixtures; EM treatments in RSEM (Li & Dewey 2011),
  DADA2 (Callahan 2016), pooled-screen analysis (MAGeCK).
"""

import numpy as np
from scipy.optimize import minimize, nnls


def _nnls_init(b: np.ndarray, cov: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Coverage-weighted NNLS warm-start, renormalised to the simplex."""
    C = A.shape[1]
    w = np.sqrt(np.maximum(cov, 0.0))
    Aw = A * w[:, np.newaxis]
    bw = b * w
    x, _ = nnls(Aw, bw)
    total = x.sum()
    if total > 0:
        return x / total
    return np.full(C, 1.0 / C)


def _binomial_mle_single(
    b: np.ndarray,
    cov: np.ndarray,
    A: np.ndarray,
    max_iter: int,
    p_clip: float,
) -> np.ndarray:
    """Solve binomial MLE for one sample via SLSQP."""
    C = A.shape[1]
    u = b * cov                # observed methylated-ish count per marker
    valid = cov > 0
    cov_valid = cov[valid]
    u_valid = u[valid]
    A_valid = A[valid]

    def neg_log_lik(x):
        p = A_valid @ x
        p = np.clip(p, p_clip, 1.0 - p_clip)
        return -float(np.sum(u_valid * np.log(p) + (cov_valid - u_valid) * np.log(1.0 - p)))

    def neg_log_lik_grad(x):
        p = A_valid @ x
        p_clamped = np.clip(p, p_clip, 1.0 - p_clip)
        # Gradient of negative log-likelihood
        # ∂L/∂x_c = Σ_m A_{m,c} · (c_m · p_m − u_m) / (p_m (1 − p_m))
        num = cov_valid * p_clamped - u_valid
        den = p_clamped * (1.0 - p_clamped)
        return A_valid.T @ (num / den)

    x0 = _nnls_init(b, cov, A)
    # Nudge off exact zeros so SLSQP starts interior.
    x0 = np.clip(x0, 1e-6, None)
    x0 = x0 / x0.sum()

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                    "jac": lambda x: np.ones_like(x)}]
    bounds = [(1e-8, 1.0)] * C

    result = minimize(
        neg_log_lik, x0, jac=neg_log_lik_grad,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )
    x = np.clip(result.x, 0.0, 1.0)
    total = x.sum()
    return x / total if total > 0 else x0


def run_binomial_mle(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    max_iter: int = 200,
    p_clip: float = 1e-3,
) -> np.ndarray:
    """Estimate cell-type proportions by maximising the binomial log-likelihood.

    Parameters
    ----------
    X : (N, M) observed methylation fractions per marker.
    coverage : (N, M) per-marker read counts.
    reference_profiles : (C, M) atlas (one row per cell type; values in [0, 1]).
    max_iter : SLSQP iteration cap per sample.
    p_clip : clamp predicted probability to ``[p_clip, 1 − p_clip]`` to keep the
        likelihood finite at boundaries. Only matters at p ≈ {0, 1}.

    Returns
    -------
    (N, C) proportions with rows summing to 1.
    """
    N = X.shape[0]
    C = reference_profiles.shape[0]
    A = reference_profiles.T.astype(np.float64)  # (M, C)
    results = np.zeros((N, C))
    for i in range(N):
        results[i] = _binomial_mle_single(
            X[i].astype(np.float64),
            coverage[i].astype(np.float64),
            A,
            max_iter=max_iter,
            p_clip=p_clip,
        )
    return results
