"""Cohort-level Non-negative Matrix Factorisation deconvolution with atlas anchor.

Unlike every other method in this benchmark module, NMF is **not per-sample**.
It jointly estimates all sample proportions ``W ∈ ℝ^{N×C}_+`` and a shared
cell-type signature matrix ``S ∈ ℝ^{C×M}_+`` by factorising the full
observation matrix ``D ∈ ℝ^{N×M}``:

    min   Σ_{i,m}  c_{im} · (D_{im} − (W S)_{im})²   +   λ · ‖S − atlas‖²_F
    s.t.  W ≥ 0,  rows of W on the simplex
          S ≥ 0

``λ`` controls how strongly the learned signatures are pulled toward the
atlas. ``λ → ∞`` pins ``S`` to the atlas (cohort-NNLS); ``λ = 0`` discards
the atlas entirely (pure unsupervised NMF). Intermediate values let the
cohort's internal structure correct consistent atlas errors while retaining
identifiability.

Algorithm: alternating multiplicative updates (Lee & Seung 2001 / Paatero &
Tapper 1994), extended with coverage weighting and the anchor penalty. The
W update is followed by simplex renormalisation (projected gradient step);
convergence guarantees of the unconstrained Lee-Seung algorithm don't
strictly carry over, but empirically the iteration is stable.

Transductive: takes the evaluation cohort itself as input and refines
signatures on that cohort's joint structure. No separate training step, no
labels, no prior over proportions.
"""

import numpy as np
from scipy.optimize import nnls


def _init_W_from_atlas(X: np.ndarray, coverage: np.ndarray, atlas: np.ndarray) -> np.ndarray:
    """Warm-start W via per-sample coverage-weighted NNLS against the atlas."""
    N = X.shape[0]
    C = atlas.shape[0]
    A = atlas.T  # (M, C)
    W = np.zeros((N, C))
    for i in range(N):
        w = np.sqrt(np.maximum(coverage[i], 0.0))
        Aw = A * w[:, np.newaxis]
        bw = X[i] * w
        x, _ = nnls(Aw, bw)
        total = x.sum()
        W[i] = x / total if total > 0 else np.full(C, 1.0 / C)
    return W


def run_nmf_deconvolution(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    anchor_weight: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate proportions jointly across the cohort via atlas-anchored NMF.

    Parameters
    ----------
    X : (N, M) observed U-fractions per marker.
    coverage : (N, M) per-marker read counts (used as element-wise weights).
    reference_profiles : (C, M) atlas (serves as both init and anchor for S).
    anchor_weight : λ penalising ‖S − atlas‖². Sweep candidate: 0 (no atlas)
        through 100+ (atlas essentially fixed).
    max_iter : outer iteration cap.
    tol : convergence threshold on relative change of W.

    Returns
    -------
    W : (N, C) proportions with rows summing to 1.
    S : (C, M) learned signatures (non-negative, clamped to [0, 1]).
    """
    X = X.astype(np.float64)
    coverage = coverage.astype(np.float64)
    atlas = np.clip(reference_profiles.astype(np.float64), eps, 1.0 - eps)

    N, M = X.shape
    C = atlas.shape[0]

    S = atlas.copy()
    W = _init_W_from_atlas(X, coverage, atlas)
    W = np.maximum(W, eps)

    CX = coverage * X  # (N, M), constant

    for it in range(max_iter):
        W_prev = W.copy()

        # --- S update (shape (C, M)) ---
        WS = W @ S
        CWS = coverage * WS
        S_num = W.T @ CX + anchor_weight * atlas
        S_den = W.T @ CWS + anchor_weight * S + eps
        S = S * (S_num / S_den)
        S = np.clip(S, 0.0, 1.0)

        # --- W update (shape (N, C)) ---
        WS = W @ S
        CWS = coverage * WS
        W_num = CX @ S.T
        W_den = CWS @ S.T + eps
        W = W * (W_num / W_den)
        row_sums = W.sum(axis=1, keepdims=True)
        W = W / np.maximum(row_sums, eps)

        # Convergence check
        rel_change = np.linalg.norm(W - W_prev) / (np.linalg.norm(W_prev) + eps)
        if rel_change < tol:
            break

    return W, S
