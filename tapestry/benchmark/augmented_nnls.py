"""Augmented-basis NNLS with data-driven unknown-tissue components.

Builds a residual basis from healthy control samples in the cohort, then
augments the atlas with that basis for deconvolution. The augmented
coefficients on the learned components can take either sign and absorb
methylation signal that the atlas can't explain — preventing NNLS from
force-fitting non-atlas signal into atlas cell types.

Addresses the completeness gap: real cfDNA contains methylation from tissues
not listed in the atlas. Standard NNLS has no outlet for that signal and
smears it across atlas cell types that happen to have signatures with
non-zero overlap with the unexplained residual direction. The augmented
basis is exactly the direction(s) NNLS needs to vent that signal to.

The lambda=0 endpoint is the original free unknown-channel model. As lambda
increases, the solution is anchored back to the shipped coverage-weighted NNLS
objective: same coverage weights, same post-fit normalisation. This makes the
lambda path interpretable as "how much unknown-channel absorption is useful"
rather than a comparison against a different atlas estimator.
"""

import numpy as np
from scipy.optimize import minimize, nnls, lsq_linear


def _nnls_row(b: np.ndarray, cov: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Coverage-weighted NNLS for one sample. Returns the raw (unnormalised)
    NNLS vector — used for residual computation, so we don't normalise."""
    w = np.maximum(cov, 0.0)
    Aw = A * w[:, np.newaxis]
    bw = b * w
    x, _ = nnls(Aw, bw)
    return x


def build_unknown_basis(
    X_controls: np.ndarray,
    coverage_controls: np.ndarray,
    reference_profiles: np.ndarray,
    n_components: int = 3,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an unknown-tissue basis from healthy-control residuals.

    Parameters
    ----------
    X_controls : (N_ctrl, M) observed U-fractions for healthy control samples.
    coverage_controls : (N_ctrl, M) per-marker read counts for controls.
    reference_profiles : (C, M) atlas.
    n_components : number of top singular-vector directions to keep.
    center : if True, subtract the mean residual before SVD (captures residual
        *variation* across controls rather than the common residual). Default
        False: the mean residual direction is a legitimate "non-atlas healthy
        tissue" signal and should be kept.

    Returns
    -------
    U : (M, K) orthonormal basis vectors, each a direction in marker space
        that appears in healthy-control residuals but isn't spanned by the
        atlas.
    variance_explained : (K,) fraction of residual variance each component
        captures; diagnostic only.
    """
    N_ctrl, M = X_controls.shape
    A = reference_profiles.T.astype(np.float64)  # (M, C)

    residuals = np.zeros((N_ctrl, M), dtype=np.float64)
    for i in range(N_ctrl):
        x_hat = _nnls_row(X_controls[i].astype(np.float64),
                          coverage_controls[i].astype(np.float64), A)
        residuals[i] = X_controls[i] - A @ x_hat

    if center:
        residuals = residuals - residuals.mean(axis=0, keepdims=True)

    # Economy SVD. Vh is (min(N_ctrl, M), M); rows are right singular vectors.
    _, s, Vh = np.linalg.svd(residuals, full_matrices=False)
    K = min(n_components, Vh.shape[0])
    U = Vh[:K].T  # (M, K)
    total_var = (s ** 2).sum() if (s ** 2).sum() > 0 else 1.0
    var_explained = (s[:K] ** 2) / total_var
    return U, var_explained


def target_contrast(reference_profiles: np.ndarray, target_index: int) -> np.ndarray:
    """Return the marker-space contrast for a target cell type.

    The contrast is the target atlas row minus the mean of all other atlas rows.
    This is a pragmatic marker-space approximation of the direction we do not
    want the learned unknown channel to absorb.
    """
    reference_profiles = np.asarray(reference_profiles, dtype=np.float64)
    if reference_profiles.ndim != 2:
        raise ValueError("reference_profiles must have shape (C, M)")
    C = reference_profiles.shape[0]
    if target_index < 0 or target_index >= C:
        raise ValueError(f"target_index {target_index} is out of range for C={C}")
    if C == 1:
        return reference_profiles[target_index].copy()
    other = np.delete(reference_profiles, target_index, axis=0).mean(axis=0)
    return reference_profiles[target_index] - other


def project_basis_orthogonal_to(
    U: np.ndarray,
    direction: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Project a marker-space basis onto the complement of ``direction``.

    Parameters
    ----------
    U : ndarray, shape (M, K)
        Basis columns in marker space.
    direction : ndarray, shape (M,)
        Marker-space vector to remove from the basis.
    tol : float
        Singular-value tolerance used after projection. Components that become
        numerically zero are dropped.

    Returns
    -------
    ndarray, shape (M, K')
        Orthonormal projected basis. K' can be smaller than K.
    """
    U = np.asarray(U, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError("U must have shape (M, K)")
    if direction.ndim != 1 or direction.shape[0] != U.shape[0]:
        raise ValueError("direction must have shape (M,)")
    if U.shape[1] == 0:
        return U.copy()

    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= tol:
        return U.copy()

    d = direction / norm
    U_projected = U - np.outer(d, d @ U)
    q, s, _ = np.linalg.svd(U_projected, full_matrices=False)
    keep = s > tol
    if not np.any(keep):
        return np.zeros((U.shape[0], 0), dtype=np.float64)
    return q[:, keep]


def _solve_augmented_row_regularized(
    b: np.ndarray,
    cov: np.ndarray,
    A: np.ndarray,
    U: np.ndarray,
    lambda_unknown: float,
    simplex_known: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve one augmented NNLS row with an optional unknown ridge penalty."""
    if lambda_unknown < 0:
        raise ValueError("lambda_unknown must be non-negative")

    C = A.shape[1]
    K = U.shape[1]

    w = np.maximum(cov.astype(np.float64), 0.0)
    valid = w > 0
    if not valid.any():
        return np.full(C, 1.0 / C), np.zeros(K), 0.0, 0.0

    U = U.astype(np.float64)
    B_aug = np.concatenate([A, U], axis=1)
    Aw = B_aug[valid] * w[valid, np.newaxis]
    bw = b[valid].astype(np.float64) * w[valid]

    if simplex_known:
        x0 = _nnls_row(b.astype(np.float64), cov.astype(np.float64), A)
        total0 = x0.sum()
        if total0 > 0:
            x0 = x0 / total0
        else:
            x0 = np.full(C, 1.0 / C)
        z0 = np.concatenate([x0, np.zeros(K, dtype=np.float64)])

        bounds = [(0.0, None)] * C + [(None, None)] * K
        constraints = [{"type": "eq", "fun": lambda z: np.sum(z[:C]) - 1.0}]

        def objective(z: np.ndarray) -> float:
            residual = Aw @ z - bw
            value = float(residual @ residual)
            if K > 0 and lambda_unknown > 0:
                y = z[C:]
                value += float(lambda_unknown * (y @ y))
            return value

        def gradient(z: np.ndarray) -> np.ndarray:
            residual = Aw @ z - bw
            grad = 2.0 * (Aw.T @ residual)
            if K > 0 and lambda_unknown > 0:
                grad[C:] += 2.0 * lambda_unknown * z[C:]
            return grad

        result = minimize(
            objective,
            z0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        z = result.x if result.success and np.all(np.isfinite(result.x)) else z0
    else:
        lower = np.concatenate([np.zeros(C), np.full(K, -np.inf)])
        upper = np.full(C + K, np.inf)
        Aw_fit = Aw
        bw_fit = bw
        if K > 0 and lambda_unknown > 0:
            penalty = np.zeros((K, C + K), dtype=np.float64)
            penalty[:, C:] = np.sqrt(lambda_unknown) * np.eye(K)
            Aw_fit = np.vstack([Aw_fit, penalty])
            bw_fit = np.concatenate([bw_fit, np.zeros(K, dtype=np.float64)])
        result = lsq_linear(
            Aw_fit, bw_fit, bounds=(lower, upper), method="trf", max_iter=200
        )
        z = result.x

    x = np.maximum(z[:C], 0.0)
    y = z[C:]

    if simplex_known:
        proportions = x
    else:
        total = x.sum()
        if total > 0:
            proportions = x / total
        else:
            proportions = np.full(C, 1.0 / C)

    fitted = A @ x
    if K > 0:
        fitted = fitted + U @ y
    residual_norm = float(np.linalg.norm((fitted[valid] - b[valid]) * w[valid]))
    unknown_mag = float(np.sum(np.abs(y)))
    return proportions, y, unknown_mag, residual_norm


def run_augmented_nnls_regularized(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    U: np.ndarray,
    lambda_unknown: float = 0.0,
    simplex_known: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Augmented-basis NNLS with a ridge penalty on unknown coefficients.

    With the default ``simplex_known=False``, high lambda values converge to
    the shipped weighted-NNLS estimator because the known coefficients are fit
    with the same objective and normalised after fitting. ``simplex_known=True``
    is available as an experimental alternative, but it is a different atlas
    estimator and should not be compared directly to shipped NNLS.

    Returns
    -------
    proportions : (N, C)
        Rows sum to 1 across known atlas cell types.
    unknown_coef : (N, K)
        Free-sign coefficients on unknown-tissue components.
    unknown_mag : (N,)
        Sum of absolute unknown coefficients per sample.
    residual_norm : (N,)
        Weighted residual norm excluding the ridge-penalty rows.
    """
    N, _ = X.shape
    C = reference_profiles.shape[0]
    K = U.shape[1]
    A = reference_profiles.T.astype(np.float64)  # (M, C)

    proportions = np.zeros((N, C), dtype=np.float64)
    unknown_coef = np.zeros((N, K), dtype=np.float64)
    unknown_mag = np.zeros(N, dtype=np.float64)
    residual_norm = np.zeros(N, dtype=np.float64)

    for i in range(N):
        prop, coef, mag, resid = _solve_augmented_row_regularized(
            X[i].astype(np.float64),
            coverage[i].astype(np.float64),
            A,
            U.astype(np.float64),
            lambda_unknown,
            simplex_known,
        )
        proportions[i] = prop
        unknown_coef[i] = coef
        unknown_mag[i] = mag
        residual_norm[i] = resid

    return proportions, unknown_coef, unknown_mag, residual_norm


def run_augmented_nnls_path(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    U: np.ndarray,
    lambda_values: np.ndarray | list[float] | tuple[float, ...],
    simplex_known: bool = False,
) -> dict[str, np.ndarray]:
    """Run regularised augmented NNLS over a grid of unknown penalties."""
    lambdas = np.asarray(lambda_values, dtype=np.float64)
    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("lambda_values must be a non-empty 1D sequence")
    if np.any(lambdas < 0):
        raise ValueError("lambda_values must be non-negative")

    N = X.shape[0]
    C = reference_profiles.shape[0]
    K = U.shape[1]
    L = lambdas.size
    proportions = np.zeros((L, N, C), dtype=np.float64)
    unknown_coef = np.zeros((L, N, K), dtype=np.float64)
    unknown_mag = np.zeros((L, N), dtype=np.float64)
    residual_norm = np.zeros((L, N), dtype=np.float64)

    for l_idx, lam in enumerate(lambdas):
        prop, coef, mag, resid = run_augmented_nnls_regularized(
            X,
            coverage,
            reference_profiles,
            U,
            lambda_unknown=float(lam),
            simplex_known=simplex_known,
        )
        proportions[l_idx] = prop
        unknown_coef[l_idx] = coef
        unknown_mag[l_idx] = mag
        residual_norm[l_idx] = resid

    return {
        "lambdas": lambdas,
        "proportions": proportions,
        "unknown_coef": unknown_coef,
        "unknown_mag": unknown_mag,
        "residual_norm": residual_norm,
    }


def run_augmented_nnls(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augmented-basis NNLS: min ||[A | U] z - b||² with z[:C] ≥ 0, z[C:] free.

    Cell-type proportions are reported renormalised to sum to 1 among the C
    known types. Unknown-tissue coefficients (free sign) and their magnitude
    are returned separately as diagnostic outputs.

    Returns
    -------
    proportions : (N, C) rows sum to 1.
    unknown_coef : (N, K) free-sign coefficients on unknown-tissue components.
    unknown_mag : (N,) sum of absolute unknown coefficients — a rough measure
        of how much non-atlas signal the augmented basis absorbed per sample.
    """
    proportions, unknown_coef, unknown_mag, _ = run_augmented_nnls_regularized(
        X,
        coverage,
        reference_profiles,
        U,
        lambda_unknown=0.0,
        simplex_known=False,
    )
    return proportions, unknown_coef, unknown_mag
