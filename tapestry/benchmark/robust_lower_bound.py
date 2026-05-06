"""Robust target-fraction lower bounds under a calibrated nuisance set.

This module implements the partial-identification formulation

    y = theta * a_T + A_H x_H + eta

with non-negative mixture weights summing to one and ``eta`` constrained to a
healthy-control-calibrated nuisance set. The reported quantity is not a point
estimate; it is the smallest target fraction required after allowing every
plausible healthy-cfDNA residual shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from tapestry.benchmark.nnls import run_weighted_nnls


@dataclass
class FactorNuisanceModel:
    """Low-rank plus diagonal Gaussian nuisance model.

    The nuisance set is the ellipsoid

        (eta - mean)^T Sigma^{-1} (eta - mean) <= tau

    where ``Sigma = loadings @ loadings.T + diag(diag_var)``. The inverse and
    log determinant are evaluated with the Woodbury identity, so callers never
    materialise an M x M inverse.
    """

    mean: np.ndarray
    loadings: np.ndarray
    diag_var: np.ndarray

    def subset(self, mask: np.ndarray | None) -> "FactorNuisanceModel":
        if mask is None:
            return self
        mask = np.asarray(mask, dtype=bool)
        return FactorNuisanceModel(
            mean=self.mean[mask],
            loadings=self.loadings[mask],
            diag_var=self.diag_var[mask],
        )

    @property
    def n_markers(self) -> int:
        return int(self.mean.shape[0])

    @property
    def rank(self) -> int:
        return int(self.loadings.shape[1])

    def mahalanobis(self, eta: np.ndarray, mask: np.ndarray | None = None) -> float:
        model = self.subset(mask)
        centered = np.asarray(eta, dtype=np.float64)
        if mask is not None:
            centered = centered[np.asarray(mask, dtype=bool)]
        centered = centered - model.mean
        inv_diag = 1.0 / model.diag_var
        first = float(np.sum(centered * centered * inv_diag))
        if model.rank == 0:
            return first
        weighted_loadings = model.loadings * inv_diag[:, np.newaxis]
        middle = np.eye(model.rank) + model.loadings.T @ weighted_loadings
        rhs = model.loadings.T @ (centered * inv_diag)
        correction = float(rhs @ np.linalg.solve(middle, rhs))
        return max(0.0, first - correction)

    def normalized_mahalanobis(
        self,
        eta: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        if mask is None:
            return self.mahalanobis(eta)
        n_obs = int(np.asarray(mask, dtype=bool).sum())
        if n_obs == 0:
            return np.inf
        return self.mahalanobis(eta, mask=mask) * (self.n_markers / n_obs)

    def log_likelihood(self, residuals: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=np.float64)
        if residuals.ndim != 2:
            raise ValueError("residuals must have shape (N, M)")
        inv_diag = 1.0 / self.diag_var
        logdet = float(np.sum(np.log(self.diag_var)))
        if self.rank > 0:
            weighted_loadings = self.loadings * inv_diag[:, np.newaxis]
            middle = np.eye(self.rank) + self.loadings.T @ weighted_loadings
            sign, logdet_middle = np.linalg.slogdet(middle)
            if sign <= 0:
                return -np.inf
            logdet += float(logdet_middle)
        q = np.array([self.mahalanobis(row) for row in residuals])
        m = self.n_markers
        return float(-0.5 * np.sum(m * np.log(2.0 * np.pi) + logdet + q))


@dataclass
class RobustLowerBoundCalibration:
    target_cell_type: str
    healthy_cell_types: list[str]
    target_index: int
    healthy_indices: np.ndarray
    marker_mask: np.ndarray
    nuisance_model: FactorNuisanceModel
    tau: float
    alpha: float
    factor_rank: int
    fit_control_indices: np.ndarray
    tau_control_indices: np.ndarray
    validation_control_indices: np.ndarray
    tau_distances: np.ndarray
    validation_distances: np.ndarray
    validation_containment: float


def parse_rank_grid(value: str) -> list[int]:
    ranks = []
    for part in value.split(","):
        part = part.strip()
        if part:
            ranks.append(int(part))
    if not ranks:
        raise ValueError("factor-rank grid must contain at least one rank")
    if any(rank < 0 for rank in ranks):
        raise ValueError("factor ranks must be non-negative")
    return sorted(set(ranks))


def split_control_indices(
    control_indices: np.ndarray,
    random_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split controls into covariance-fit, tau-calibration, validation folds.

    For very small smoke tests we reuse controls across folds with a warning
    left to the caller. Production analyses should have enough controls for
    disjoint folds.
    """
    control_indices = np.asarray(control_indices, dtype=int)
    if control_indices.size < 6:
        return control_indices, control_indices, control_indices
    rng = np.random.default_rng(random_seed)
    shuffled = control_indices.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, 3)
    fit = folds[0]
    tau = folds[1]
    validation = folds[2]
    return fit, tau, validation


def fit_non_target_residuals(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit OAC-excluded NNLS and return residuals on the original marker scale."""
    healthy_indices = np.array(
        [i for i in range(reference_profiles.shape[0]) if i != target_index],
        dtype=int,
    )
    A_H = reference_profiles[healthy_indices]
    props_H = run_weighted_nnls(X, coverage, A_H)
    fitted = props_H @ A_H
    residuals = X - fitted
    return props_H, residuals, healthy_indices


def fit_factor_nuisance(
    residuals: np.ndarray,
    rank: int,
    diag_floor_quantile: float = 0.1,
    min_diag: float = 1e-8,
) -> FactorNuisanceModel:
    """Fit a PCA factor-plus-diagonal covariance model to residuals."""
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape (N, M)")
    n, m = residuals.shape
    rank = min(rank, max(0, n - 1), m)
    mean = residuals.mean(axis=0)
    centered = residuals - mean
    total_var = centered.var(axis=0, ddof=1 if n > 1 else 0)
    if rank == 0 or n <= 1:
        diag_var = np.maximum(total_var, min_diag)
        return FactorNuisanceModel(mean=mean, loadings=np.zeros((m, 0)), diag_var=diag_var)

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigvals = (singular_values[:rank] ** 2) / max(n - 1, 1)
    loadings = vt[:rank].T * np.sqrt(np.maximum(eigvals, 0.0))
    factor_var = np.sum(loadings * loadings, axis=1)
    diag_var = np.maximum(total_var - factor_var, 0.0)
    positive = total_var[total_var > min_diag]
    floor = min_diag
    if positive.size:
        floor = max(float(np.quantile(positive, diag_floor_quantile)), min_diag)
    diag_var = np.maximum(diag_var, floor)
    return FactorNuisanceModel(mean=mean, loadings=loadings, diag_var=diag_var)


def select_factor_rank(
    residuals: np.ndarray,
    rank_grid: list[int],
    random_seed: int = 1,
) -> tuple[int, dict[int, float]]:
    """Choose factor rank by held-out Gaussian log likelihood."""
    residuals = np.asarray(residuals, dtype=np.float64)
    n = residuals.shape[0]
    if n < 6:
        rank = min(max(rank_grid), max(0, n - 1))
        return rank, {rank: np.nan}
    rng = np.random.default_rng(random_seed)
    order = np.arange(n)
    rng.shuffle(order)
    split = max(2, int(round(n * 0.7)))
    train_idx = order[:split]
    val_idx = order[split:]
    if val_idx.size == 0:
        val_idx = order[-1:]
        train_idx = order[:-1]
    scores: dict[int, float] = {}
    best_rank = 0
    best_score = -np.inf
    for rank in rank_grid:
        model = fit_factor_nuisance(residuals[train_idx], rank)
        score = model.log_likelihood(residuals[val_idx])
        scores[rank] = score
        if score > best_score:
            best_score = score
            best_rank = rank
    return best_rank, scores


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    """Conformal-style upper quantile for nuisance containment."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        raise ValueError("cannot calibrate tau from zero distances")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), values.size - 1)
    return float(values[rank])


def calibrate_robust_lower_bound(
    X: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    cell_types: list[str],
    target_cell_type: str = "OAC",
    control_mask: np.ndarray | None = None,
    alpha: float = 0.05,
    factor_rank: int | None = None,
    factor_rank_grid: list[int] | None = None,
    min_control_coverage_fraction: float = 0.8,
    min_sample_coverage: float = 1.0,
    random_seed: int = 1,
) -> RobustLowerBoundCalibration:
    """Fit the healthy residual nuisance set and calibrate its radius."""
    if target_cell_type not in cell_types:
        raise ValueError(f"target cell type {target_cell_type!r} not found")
    target_index = cell_types.index(target_cell_type)
    if control_mask is None:
        raise ValueError("control_mask is required for nuisance calibration")
    control_mask = np.asarray(control_mask, dtype=bool)
    control_indices = np.where(control_mask)[0]
    if control_indices.size < 2:
        raise ValueError("need at least two controls to calibrate nuisance set")

    marker_mask = (
        np.mean(coverage[control_indices] >= min_sample_coverage, axis=0)
        >= min_control_coverage_fraction
    )
    if not np.any(marker_mask):
        raise ValueError("no markers pass the control coverage filter")

    X_m = X[:, marker_mask]
    coverage_m = coverage[:, marker_mask]
    ref_m = reference_profiles[:, marker_mask]
    _, residuals, healthy_indices = fit_non_target_residuals(
        X_m, coverage_m, ref_m, target_index
    )

    fit_idx, tau_idx, validation_idx = split_control_indices(
        control_indices, random_seed=random_seed
    )
    local_fit = np.array([np.where(control_indices == idx)[0][0] for idx in fit_idx])
    local_tau = np.array([np.where(control_indices == idx)[0][0] for idx in tau_idx])
    local_validation = np.array(
        [np.where(control_indices == idx)[0][0] for idx in validation_idx]
    )
    control_residuals = residuals[control_indices]

    if factor_rank_grid is None:
        factor_rank_grid = [0, 3, 5, 10]
    max_rank = max(0, min(control_residuals[local_fit].shape[0] - 1, marker_mask.sum()))
    factor_rank_grid = [rank for rank in factor_rank_grid if rank <= max_rank]
    if not factor_rank_grid:
        factor_rank_grid = [0]
    if factor_rank is None:
        factor_rank, _ = select_factor_rank(
            control_residuals[local_fit],
            factor_rank_grid,
            random_seed=random_seed,
        )
    factor_rank = min(factor_rank, max_rank)
    nuisance_model = fit_factor_nuisance(control_residuals[local_fit], factor_rank)

    control_coverage_m = coverage_m[control_indices]
    tau_distances = []
    for idx in local_tau:
        obs_mask = control_coverage_m[idx] >= min_sample_coverage
        tau_distances.append(
            nuisance_model.normalized_mahalanobis(
                control_residuals[idx], mask=obs_mask
            )
        )
    tau_distances = np.asarray(tau_distances, dtype=np.float64)
    tau = finite_sample_quantile(tau_distances, alpha=alpha)

    validation_distances = []
    for idx in local_validation:
        obs_mask = control_coverage_m[idx] >= min_sample_coverage
        validation_distances.append(
            nuisance_model.normalized_mahalanobis(
                control_residuals[idx], mask=obs_mask
            )
        )
    validation_distances = np.asarray(validation_distances, dtype=np.float64)
    validation_containment = float(np.mean(validation_distances <= tau))

    healthy_cell_types = [cell_types[i] for i in healthy_indices]
    return RobustLowerBoundCalibration(
        target_cell_type=target_cell_type,
        healthy_cell_types=healthy_cell_types,
        target_index=target_index,
        healthy_indices=healthy_indices,
        marker_mask=marker_mask,
        nuisance_model=nuisance_model,
        tau=tau,
        alpha=alpha,
        factor_rank=factor_rank,
        fit_control_indices=fit_idx,
        tau_control_indices=tau_idx,
        validation_control_indices=validation_idx,
        tau_distances=tau_distances,
        validation_distances=validation_distances,
        validation_containment=validation_containment,
    )


def _objective_weights(
    y: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    calibration: RobustLowerBoundCalibration,
    maximize: bool,
    min_sample_coverage: float,
) -> dict[str, object]:
    marker_mask = calibration.marker_mask
    model_mask = coverage[marker_mask] >= min_sample_coverage
    ref_m = reference_profiles[:, marker_mask]
    y_m = y[marker_mask]
    a_t = ref_m[calibration.target_index]
    a_h = ref_m[calibration.healthy_indices]

    def eta_from_weights(weights: np.ndarray) -> np.ndarray:
        theta = weights[0]
        x_h = weights[1:]
        return y_m - theta * a_t - x_h @ a_h

    def distance(weights: np.ndarray) -> float:
        eta = eta_from_weights(weights)
        return calibration.nuisance_model.normalized_mahalanobis(eta, mask=model_mask)

    def objective(weights: np.ndarray) -> float:
        theta = weights[0]
        return -theta if maximize else theta

    return {
        "marker_mask": marker_mask,
        "model_mask": model_mask,
        "objective": objective,
        "distance": distance,
        "n_variables": 1 + len(calibration.healthy_indices),
    }


def solve_theta_bound(
    y: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    calibration: RobustLowerBoundCalibration,
    maximize: bool = False,
    min_sample_coverage: float = 1.0,
) -> dict[str, object]:
    """Solve one theta lower or upper bound for a sample."""
    problem = _objective_weights(
        y,
        coverage,
        reference_profiles,
        calibration,
        maximize=maximize,
        min_sample_coverage=min_sample_coverage,
    )
    n_variables = int(problem["n_variables"])
    model_mask = np.asarray(problem["model_mask"], dtype=bool)
    if model_mask.sum() == 0:
        return {
            "theta": np.nan,
            "distance": np.inf,
            "success": False,
            "message": "no observed markers",
            "n_markers_used": 0,
        }

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: calibration.tau - problem["distance"](w)},
    ]
    bounds = [(0.0, 1.0)] * n_variables
    starts = []
    starts.append(np.r_[0.0, np.full(n_variables - 1, 1.0 / (n_variables - 1))])
    starts.append(np.r_[1.0, np.zeros(n_variables - 1)])
    starts.append(np.full(n_variables, 1.0 / n_variables))

    best = None
    for start in starts:
        result = minimize(
            problem["objective"],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        weights = result.x
        distance = float(problem["distance"](weights))
        feasible = bool(result.success and distance <= calibration.tau * (1.0 + 1e-6))
        score = float(problem["objective"](weights))
        if feasible and (best is None or score < best["score"]):
            best = {
                "theta": float(weights[0]),
                "weights": weights,
                "distance": distance,
                "success": True,
                "message": result.message,
                "score": score,
                "n_markers_used": int(model_mask.sum()),
            }

    if best is None:
        return {
            "theta": np.nan,
            "weights": np.full(n_variables, np.nan),
            "distance": np.inf,
            "success": False,
            "message": "no feasible solution found",
            "n_markers_used": int(model_mask.sum()),
        }
    return best


def solve_best_distance(
    y: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    calibration: RobustLowerBoundCalibration,
    min_sample_coverage: float = 1.0,
) -> dict[str, object]:
    """Find the closest point in the atlas simplex to the nuisance set.

    This diagnostic is meaningful when ``theta_min`` is infeasible: it tells
    whether the sample is merely outside the calibrated nuisance radius and
    which target fraction minimises the Mahalanobis residual distance.
    """
    problem = _objective_weights(
        y,
        coverage,
        reference_profiles,
        calibration,
        maximize=False,
        min_sample_coverage=min_sample_coverage,
    )
    n_variables = int(problem["n_variables"])
    model_mask = np.asarray(problem["model_mask"], dtype=bool)
    if model_mask.sum() == 0:
        return {
            "theta": np.nan,
            "distance": np.inf,
            "success": False,
            "message": "no observed markers",
            "n_markers_used": 0,
        }

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_variables
    starts = [
        np.r_[0.0, np.full(n_variables - 1, 1.0 / (n_variables - 1))],
        np.r_[1.0, np.zeros(n_variables - 1)],
        np.full(n_variables, 1.0 / n_variables),
    ]

    best = None
    for start in starts:
        result = minimize(
            problem["distance"],
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        weights = result.x
        distance = float(problem["distance"](weights))
        if best is None or distance < best["distance"]:
            best = {
                "theta": float(weights[0]),
                "weights": weights,
                "distance": distance,
                "success": bool(result.success),
                "message": result.message,
                "n_markers_used": int(model_mask.sum()),
            }
    return best


def solve_theta_interval(
    y: np.ndarray,
    coverage: np.ndarray,
    reference_profiles: np.ndarray,
    calibration: RobustLowerBoundCalibration,
    min_sample_coverage: float = 1.0,
) -> dict[str, object]:
    lower = solve_theta_bound(
        y,
        coverage,
        reference_profiles,
        calibration,
        maximize=False,
        min_sample_coverage=min_sample_coverage,
    )
    upper = solve_theta_bound(
        y,
        coverage,
        reference_profiles,
        calibration,
        maximize=True,
        min_sample_coverage=min_sample_coverage,
    )
    best_distance = solve_best_distance(
        y,
        coverage,
        reference_profiles,
        calibration,
        min_sample_coverage=min_sample_coverage,
    )
    return {
        "theta_min": lower["theta"],
        "theta_max": upper["theta"],
        "theta_min_distance": lower["distance"],
        "theta_max_distance": upper["distance"],
        "theta_min_success": lower["success"],
        "theta_max_success": upper["success"],
        "n_markers_used": lower["n_markers_used"],
        "best_distance": best_distance["distance"],
        "best_distance_theta": best_distance["theta"],
        "best_distance_success": best_distance["success"],
        "inside_nuisance": bool(best_distance["distance"] <= calibration.tau),
    }
