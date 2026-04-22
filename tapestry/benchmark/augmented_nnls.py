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

No priors, no regularisation, no training-cohort labels — the unknown basis
is extracted from the test cohort's own healthy controls via SVD of their
NNLS residuals. Transductive in that sense.
"""

import numpy as np
from scipy.optimize import nnls, lsq_linear


def _nnls_row(b: np.ndarray, cov: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Coverage-weighted NNLS for one sample. Returns the raw (unnormalised)
    NNLS vector — used for residual computation, so we don't normalise."""
    w = np.sqrt(np.maximum(cov, 0.0))
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
    N, M = X.shape
    C = reference_profiles.shape[0]
    K = U.shape[1]

    A = reference_profiles.T.astype(np.float64)  # (M, C)
    B_aug = np.concatenate([A, U.astype(np.float64)], axis=1)  # (M, C+K)

    lower = np.concatenate([np.zeros(C), np.full(K, -np.inf)])
    upper = np.full(C + K, np.inf)

    proportions = np.zeros((N, C))
    unknown_coef = np.zeros((N, K))
    unknown_mag = np.zeros(N)

    for i in range(N):
        w = np.sqrt(np.maximum(coverage[i].astype(np.float64), 0.0))
        valid = w > 0
        if not valid.any():
            proportions[i] = np.full(C, 1.0 / C)
            continue

        Aw = B_aug[valid] * w[valid, np.newaxis]
        bw = X[i, valid].astype(np.float64) * w[valid]

        result = lsq_linear(Aw, bw, bounds=(lower, upper), method="trf",
                            max_iter=200)
        z = result.x
        x = np.maximum(z[:C], 0.0)
        y = z[C:]

        total = x.sum()
        if total > 0:
            proportions[i] = x / total
        else:
            proportions[i] = np.full(C, 1.0 / C)
        unknown_coef[i] = y
        unknown_mag[i] = np.sum(np.abs(y))

    return proportions, unknown_coef, unknown_mag
