"""Marker selection utilities for unknown-robust UXM deconvolution.

The routines in this module score marker sets by the geometry of the
downstream deconvolution problem rather than by one-vs-all per-marker SNR
alone.  A marker set is useful when a target atlas column cannot be
reconstructed by the remaining atlas columns plus an optional unknown basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear, minimize, nnls


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of reconstructing one atlas column from a comparison basis."""

    distance: float
    relative_distance: float
    coefficients: np.ndarray
    residual: np.ndarray
    success: bool


@dataclass(frozen=True)
class GreedySelectionResult:
    """Selected marker indices and objective trace from greedy selection."""

    selected_indices: np.ndarray
    objective_trace: np.ndarray
    distance_trace: np.ndarray
    condition_trace: np.ndarray


def direct_weight_vector(weights: np.ndarray | None, n_markers: int) -> np.ndarray:
    """Return non-negative direct marker multipliers for weighted residuals.

    The existing weighted NNLS implementation multiplies both sides of the
    linear system by a marker weight vector.  This helper follows that
    convention: the objective is ``||diag(w) residual||^2``.
    """
    if weights is None:
        return np.ones(n_markers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_markers,):
        raise ValueError(f"weights must have shape ({n_markers},)")
    return np.maximum(weights, 0.0)


def weighted_condition_number(
    reference_profiles: np.ndarray,
    marker_indices: np.ndarray | list[int],
    weights: np.ndarray | None = None,
) -> float:
    """Condition number of the weighted atlas submatrix.

    Parameters
    ----------
    reference_profiles
        Atlas with shape ``(C, M)``.
    marker_indices
        Marker indices to include.
    weights
        Optional direct marker multipliers with shape ``(M,)``.
    """
    reference_profiles = np.asarray(reference_profiles, dtype=np.float64)
    marker_indices = np.asarray(marker_indices, dtype=int)
    if marker_indices.size == 0:
        return np.inf

    selected = reference_profiles[:, marker_indices].T
    w = direct_weight_vector(weights, reference_profiles.shape[1])[marker_indices]
    selected = selected * w[:, None]
    if selected.shape[0] < selected.shape[1]:
        return np.inf

    singular_values = np.linalg.svd(selected, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.size == 0:
        return np.inf
    if singular_values.size < selected.shape[1]:
        return np.inf
    return float(singular_values.max() / singular_values.min())


def single_marker_outside_hull_scores(
    reference_profiles: np.ndarray,
    target_index: int,
    comparison_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Per-marker 1D distance of target value outside comparison range.

    In one marker dimension, the convex hull of comparison cell types is just
    ``[min(comparison), max(comparison)]``.  A marker has positive score when
    the target lies outside that range.
    """
    reference_profiles = np.asarray(reference_profiles, dtype=np.float64)
    if comparison_indices is None:
        comparison_indices = [
            i for i in range(reference_profiles.shape[0]) if i != target_index
        ]
    comparison_indices = np.asarray(comparison_indices, dtype=int)
    if comparison_indices.size == 0:
        return np.zeros(reference_profiles.shape[1], dtype=np.float64)

    target = reference_profiles[target_index]
    comparison = reference_profiles[comparison_indices]
    lower = comparison.min(axis=0)
    upper = comparison.max(axis=0)
    return np.maximum.reduce([lower - target, target - upper, np.zeros_like(target)])


def reconstruct_target_from_basis(
    target_profile: np.ndarray,
    basis_profiles: np.ndarray,
    weights: np.ndarray | None = None,
    n_nonnegative: int | None = None,
    nonnegative_sum_to_one: bool = False,
) -> ReconstructionResult:
    """Reconstruct a target profile from a mixed constrained basis.

    Parameters
    ----------
    target_profile
        Vector with shape ``(M,)``.
    basis_profiles
        Basis rows with shape ``(K, M)``.  The first ``n_nonnegative`` rows are
        constrained to have non-negative coefficients.  Remaining rows are
        signed nuisance/unknown directions.
    weights
        Optional direct marker multipliers.
    n_nonnegative
        Number of leading basis profiles constrained to non-negative
        coefficients.  Defaults to all rows.
    nonnegative_sum_to_one
        If true, the non-negative coefficients are constrained to sum to one.
        This scores convex-hull reconstruction rather than cone
        reconstruction.
    """
    target_profile = np.asarray(target_profile, dtype=np.float64)
    basis_profiles = np.asarray(basis_profiles, dtype=np.float64)
    if target_profile.ndim != 1:
        raise ValueError("target_profile must be one-dimensional")
    if basis_profiles.ndim != 2 or basis_profiles.shape[1] != target_profile.size:
        raise ValueError("basis_profiles must have shape (K, M)")

    n_basis = basis_profiles.shape[0]
    if n_basis == 0:
        weights_full = direct_weight_vector(weights, target_profile.size)
        residual = target_profile.copy()
        distance = float(np.linalg.norm(weights_full * residual))
        denom = float(np.linalg.norm(weights_full * target_profile))
        return ReconstructionResult(
            distance=distance,
            relative_distance=distance / denom if denom > 0 else 0.0,
            coefficients=np.zeros(0, dtype=np.float64),
            residual=residual,
            success=True,
        )

    if n_nonnegative is None:
        n_nonnegative = n_basis
    if n_nonnegative < 0 or n_nonnegative > n_basis:
        raise ValueError("n_nonnegative is out of range")

    w = direct_weight_vector(weights, target_profile.size)
    design = basis_profiles.T
    design_w = design * w[:, None]
    target_w = target_profile * w

    if n_nonnegative == n_basis and not nonnegative_sum_to_one:
        coef, _ = nnls(design_w, target_w)
        success = True
    elif not nonnegative_sum_to_one:
        lower = np.concatenate([
            np.zeros(n_nonnegative, dtype=np.float64),
            np.full(n_basis - n_nonnegative, -np.inf, dtype=np.float64),
        ])
        upper = np.full(n_basis, np.inf, dtype=np.float64)
        result = lsq_linear(design_w, target_w, bounds=(lower, upper))
        coef = result.x
        success = bool(result.success)
    else:
        start = np.zeros(n_basis, dtype=np.float64)
        if n_nonnegative > 0:
            start[:n_nonnegative] = 1.0 / n_nonnegative

        bounds = [(0.0, None)] * n_nonnegative
        bounds += [(None, None)] * (n_basis - n_nonnegative)
        constraints = []
        if n_nonnegative > 0:
            constraints.append({
                "type": "eq",
                "fun": lambda z: np.sum(z[:n_nonnegative]) - 1.0,
            })

        def objective(z: np.ndarray) -> float:
            residual_w = design_w @ z - target_w
            return float(residual_w @ residual_w)

        def gradient(z: np.ndarray) -> np.ndarray:
            residual_w = design_w @ z - target_w
            return 2.0 * (design_w.T @ residual_w)

        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        coef = result.x
        success = bool(result.success and np.all(np.isfinite(result.x)))

    residual = target_profile - design @ coef
    distance = float(np.linalg.norm(w * residual))
    denom = float(np.linalg.norm(w * target_profile))
    return ReconstructionResult(
        distance=distance,
        relative_distance=distance / denom if denom > 0 else 0.0,
        coefficients=coef,
        residual=residual,
        success=success,
    )


def target_separability(
    reference_profiles: np.ndarray,
    target_index: int,
    marker_indices: np.ndarray | list[int],
    weights: np.ndarray | None = None,
    unknown_basis: np.ndarray | None = None,
    nonnegative_sum_to_one: bool = True,
) -> ReconstructionResult:
    """Distance from a target atlas column to non-target columns plus unknown."""
    reference_profiles = np.asarray(reference_profiles, dtype=np.float64)
    marker_indices = np.asarray(marker_indices, dtype=int)
    if marker_indices.size == 0:
        raise ValueError("marker_indices cannot be empty")

    comparison_indices = [
        i for i in range(reference_profiles.shape[0]) if i != target_index
    ]
    target = reference_profiles[target_index, marker_indices]
    basis_parts = [reference_profiles[comparison_indices][:, marker_indices]]
    n_nonnegative = len(comparison_indices)

    if unknown_basis is not None:
        unknown_basis = np.asarray(unknown_basis, dtype=np.float64)
        if unknown_basis.ndim != 2 or unknown_basis.shape[0] != reference_profiles.shape[1]:
            raise ValueError("unknown_basis must have shape (M, K)")
        basis_parts.append(unknown_basis[marker_indices].T)

    basis = np.vstack(basis_parts)
    selected_weights = None
    if weights is not None:
        selected_weights = direct_weight_vector(weights, reference_profiles.shape[1])[
            marker_indices
        ]
    return reconstruct_target_from_basis(
        target,
        basis,
        weights=selected_weights,
        n_nonnegative=n_nonnegative,
        nonnegative_sum_to_one=nonnegative_sum_to_one,
    )


def leave_one_out_separability(
    reference_profiles: np.ndarray,
    marker_indices: np.ndarray | list[int],
    weights: np.ndarray | None = None,
    unknown_basis: np.ndarray | None = None,
    nonnegative_sum_to_one: bool = True,
) -> np.ndarray:
    """Target separability distance for every atlas column."""
    reference_profiles = np.asarray(reference_profiles, dtype=np.float64)
    distances = np.zeros(reference_profiles.shape[0], dtype=np.float64)
    for target_index in range(reference_profiles.shape[0]):
        result = target_separability(
            reference_profiles,
            target_index,
            marker_indices,
            weights=weights,
            unknown_basis=unknown_basis,
            nonnegative_sum_to_one=nonnegative_sum_to_one,
        )
        distances[target_index] = result.distance
    return distances


def greedy_target_marker_selection(
    reference_profiles: np.ndarray,
    target_index: int,
    candidate_indices: np.ndarray | list[int],
    n_select: int,
    weights: np.ndarray | None = None,
    unknown_basis: np.ndarray | None = None,
    condition_penalty: float = 0.0,
    nonnegative_sum_to_one: bool = True,
    initial_indices: np.ndarray | list[int] | None = None,
) -> GreedySelectionResult:
    """Greedily select markers by robust target separability.

    At each step, add the marker that maximizes target separability minus a
    condition-number penalty.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    if n_select < 0:
        raise ValueError("n_select must be non-negative")

    selected: list[int] = []
    if initial_indices is not None:
        selected = [int(i) for i in np.asarray(initial_indices, dtype=int)]
    selected_set = set(selected)
    remaining = [int(i) for i in candidate_indices if int(i) not in selected_set]

    objective_trace: list[float] = []
    distance_trace: list[float] = []
    condition_trace: list[float] = []

    while remaining and len(selected) < n_select:
        best_marker = None
        best_objective = -np.inf
        best_distance = np.nan
        best_condition = np.inf

        for marker in remaining:
            trial = np.array(selected + [marker], dtype=int)
            distance = target_separability(
                reference_profiles,
                target_index,
                trial,
                weights=weights,
                unknown_basis=unknown_basis,
                nonnegative_sum_to_one=nonnegative_sum_to_one,
            ).distance
            condition = weighted_condition_number(
                reference_profiles,
                trial,
                weights=weights,
            )
            condition_term = 0.0
            if condition_penalty > 0 and np.isfinite(condition):
                condition_term = condition_penalty * np.log1p(condition)
            elif condition_penalty > 0 and trial.size >= reference_profiles.shape[0]:
                condition_term = np.inf
            objective = distance - condition_term
            if objective > best_objective:
                best_marker = marker
                best_objective = float(objective)
                best_distance = float(distance)
                best_condition = float(condition)

        if best_marker is None:
            break
        selected.append(best_marker)
        remaining.remove(best_marker)
        objective_trace.append(best_objective)
        distance_trace.append(best_distance)
        condition_trace.append(best_condition)

    return GreedySelectionResult(
        selected_indices=np.asarray(selected, dtype=int),
        objective_trace=np.asarray(objective_trace, dtype=np.float64),
        distance_trace=np.asarray(distance_trace, dtype=np.float64),
        condition_trace=np.asarray(condition_trace, dtype=np.float64),
    )


def rank_backbone_candidates(
    reference_profiles: np.ndarray,
    target_indices: np.ndarray | list[int],
) -> dict[int, np.ndarray]:
    """Rank markers for each backbone cell type by one-marker separability."""
    scores = {}
    for target_index in target_indices:
        marker_scores = single_marker_outside_hull_scores(
            reference_profiles,
            target_index,
            comparison_indices=[i for i in target_indices if i != target_index],
        )
        scores[int(target_index)] = np.argsort(marker_scores)[::-1]
    return scores
