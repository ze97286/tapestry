"""Beta-Binomial deconvolution with empirical per-marker atlas priors.

Replaces the deterministic atlas of NNLS / binomial-MLE with a Beta(α, β)
prior at every (cell type, marker) position, fitted by method-of-moments
from the per-reference-sample homog data (one observation per reference
sample per cell type per marker).

Per sample i and marker m, the observation is

    u_{i,m} | c_{i,m}, p_{i,m}  ~  Binomial(c_{i,m}, p_{i,m})
    p_{i,m}                     =  Σ_c x_{i,c} · θ_{c,m}
    θ_{c,m}                     ~  Beta(α_{c,m}, β_{c,m})

The cell-type variates θ_{c,m} are independent draws from independent priors,
so the mixture mean and variance are

    μ_m(x)   = Σ_c x_{i,c} · μ_{c,m}
    σ²_m(x)  = Σ_c x_{i,c}² · σ²_{c,m}

with μ_{c,m} = α/(α+β) and σ²_{c,m} = αβ / [(α+β)²(α+β+1)].

We moment-match (μ_m, σ²_m) to a Beta-Binomial likelihood with parameters

    ν'_m = μ_m(1-μ_m) / σ²_m  −  1
    α'_m = μ_m · ν'_m
    β'_m = (1 − μ_m) · ν'_m

and use the closed-form Beta-Binomial log-likelihood (lgamma-stable). This
captures:

  - count uncertainty (binomial component) → graceful low-coverage behaviour
  - atlas uncertainty per marker (beta component) → cell types with few
    reference samples or high cross-reference variance dominate softly,
    not by force-fitting noise

Solved per sample on the simplex via SLSQP with a coverage-weighted NNLS
warm start. No training distribution prior, no synthetic data — only Type-3
empirical priors from the reference cohort.
"""

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import gammaln


def _nnls_init(b: np.ndarray, cov: np.ndarray, mu_atlas: np.ndarray) -> np.ndarray:
    """Coverage-weighted NNLS warm start, simplex-renormalised.

    mu_atlas : (M, C) per-marker mean atlas values.
    """
    C = mu_atlas.shape[1]
    w = np.sqrt(np.maximum(cov, 0.0))
    Aw = mu_atlas * w[:, np.newaxis]
    bw = b * w
    x, _ = nnls(Aw, bw)
    total = x.sum()
    if total > 0:
        return x / total
    return np.full(C, 1.0 / C)


def _beta_binomial_log_lik(
    u: np.ndarray, cov: np.ndarray, alpha_eff: np.ndarray, beta_eff: np.ndarray,
) -> np.ndarray:
    """Per-marker Beta-Binomial log-likelihood (vectorised, lgamma-stable).

    Drops the constant C(cov, u) term — cancels in optimisation.
    """
    return (
        gammaln(alpha_eff + u)
        + gammaln(beta_eff + cov - u)
        - gammaln(alpha_eff + beta_eff + cov)
        - gammaln(alpha_eff)
        - gammaln(beta_eff)
        + gammaln(alpha_eff + beta_eff)
    )


def _beta_binomial_single(
    b: np.ndarray,
    cov: np.ndarray,
    mu_atlas: np.ndarray,    # (M, C)
    var_atlas: np.ndarray,   # (M, C)
    max_iter: int,
    p_clip: float,
    var_floor: float,
    nu_floor: float,
) -> np.ndarray:
    """Solve Beta-Binomial MLE for one sample on the simplex via SLSQP."""
    C = mu_atlas.shape[1]
    u_count = (b * cov).round()
    valid = cov > 0
    cov_v = cov[valid]
    u_v = u_count[valid]
    mu_v = mu_atlas[valid]            # (Mv, C)
    var_v = var_atlas[valid]          # (Mv, C)

    def neg_log_lik(x):
        mu_m = mu_v @ x                      # (Mv,)
        sig2_m = var_v @ (x ** 2)            # (Mv,)
        mu_m = np.clip(mu_m, p_clip, 1.0 - p_clip)
        sig2_m = np.maximum(sig2_m, var_floor)
        # moment-match to Beta-Binomial
        nu_m = mu_m * (1.0 - mu_m) / sig2_m - 1.0
        nu_m = np.maximum(nu_m, nu_floor)
        alpha_eff = mu_m * nu_m
        beta_eff = (1.0 - mu_m) * nu_m
        ll = _beta_binomial_log_lik(u_v, cov_v, alpha_eff, beta_eff)
        return -float(ll.sum())

    x0 = _nnls_init(b, cov, mu_atlas)
    x0 = np.clip(x0, 1e-6, None)
    x0 = x0 / x0.sum()

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                    "jac": lambda x: np.ones_like(x)}]
    bounds = [(1e-8, 1.0)] * C

    result = minimize(
        neg_log_lik, x0,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )
    x = np.clip(result.x, 0.0, 1.0)
    total = x.sum()
    return x / total if total > 0 else x0


def run_beta_binomial(
    X: np.ndarray,
    coverage: np.ndarray,
    mu_profiles: np.ndarray,
    var_profiles: np.ndarray,
    max_iter: int = 200,
    p_clip: float = 1e-3,
    var_floor: float = 1e-6,
    nu_floor: float = 1e-3,
) -> np.ndarray:
    """Beta-Binomial deconvolution with per-marker empirical Beta priors.

    Parameters
    ----------
    X : (N, M) observed U-fractions.
    coverage : (N, M) per-marker read counts.
    mu_profiles : (C, M) per (cell type, marker) Beta mean = α/(α+β); same
        as the conventional atlas U-fraction reference.
    var_profiles : (C, M) per (cell type, marker) Beta variance =
        αβ / [(α+β)²(α+β+1)]. Computed by build_atlas_priors.py.
    max_iter : SLSQP iteration cap per sample.
    p_clip : clamp predicted mean to [p_clip, 1 − p_clip] for log stability.
    var_floor : clamp mixture variance from below; protects against zero
        variance at markers where every reference sample agreed exactly.
    nu_floor : clamp moment-matched concentration from below; protects
        against ν'_m → 0 when σ²_m → μ_m(1-μ_m) (Bernoulli boundary).

    Returns
    -------
    (N, C) proportions on the simplex.
    """
    N = X.shape[0]
    C = mu_profiles.shape[0]
    assert var_profiles.shape == mu_profiles.shape, (
        f"mu_profiles {mu_profiles.shape} vs var_profiles {var_profiles.shape}"
    )
    mu_atlas = mu_profiles.T.astype(np.float64)   # (M, C)
    var_atlas = var_profiles.T.astype(np.float64) # (M, C)

    out = np.zeros((N, C))
    for i in range(N):
        out[i] = _beta_binomial_single(
            X[i].astype(np.float64),
            coverage[i].astype(np.float64),
            mu_atlas, var_atlas,
            max_iter=max_iter,
            p_clip=p_clip,
            var_floor=var_floor,
            nu_floor=nu_floor,
        )
    return out
