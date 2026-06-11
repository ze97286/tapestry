"""Tabular foundation-model head: calibrated detection + TF regression.

Consumes the per-sample read-level features and produces a calibrated
cancer-vs-healthy probability and/or a tumour-fraction regression, via an
in-context tabular model — TabICL (open, the linked model) or TabPFN — with a
scikit-learn fallback. In-context ⇒ no gradient training on our data ⇒ it cannot
overfit the few labelled samples the way a bespoke MLP would, and it is
calibrated. Used strictly as a head; sensitivity/batch-robustness come from the
upstream features. Evaluation is leave-one-group-out (group = cohort/batch).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("tabicl", "tabpfn", "sklearn")
VALID_TASKS = ("classify", "regress")


def _make_estimator(task: str, backend: str, device: str, random_state: int):
    if backend == "tabicl":
        try:
            if task == "classify":
                from tabicl import TabICLClassifier

                return TabICLClassifier(random_state=random_state), "tabicl"
            warnings.warn("TabICL has no regressor; using sklearn for regression.")
            backend = "sklearn"
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"TabICL unavailable ({exc!r}); falling back to sklearn.")
            backend = "sklearn"

    if backend == "tabpfn":
        try:
            if task == "classify":
                from tabpfn import TabPFNClassifier

                return TabPFNClassifier(device=device, random_state=random_state), "tabpfn"
            from tabpfn import TabPFNRegressor

            return TabPFNRegressor(device=device, random_state=random_state), "tabpfn"
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"TabPFN unavailable ({exc!r}); falling back to sklearn.")
            backend = "sklearn"

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task == "classify":
        from sklearn.linear_model import LogisticRegression

        est = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)),
        ])
    else:
        from sklearn.linear_model import Ridge

        est = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ])
    return est, "sklearn"


class TabularTumourHead:
    def __init__(self, task: str = "classify", backend: str = "tabicl", device: str = "cpu", random_state: int = 0) -> None:
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}")
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend must be one of {VALID_BACKENDS}")
        self.task = task
        self.requested_backend = backend
        self.device = device
        self.random_state = random_state
        self.estimator, self.backend = _make_estimator(task, backend, device, random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularTumourHead":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        try:
            self.estimator.fit(X, y)
        except Exception as exc:
            if self.backend == "sklearn":
                raise
            warnings.warn(f"{self.backend} backend failed at fit ({exc!r}); falling back to sklearn.")
            self.estimator, self.backend = _make_estimator(self.task, "sklearn", self.device, self.random_state)
            self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classify":
            raise ValueError("predict_proba is only valid for classification.")
        return self.estimator.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(np.asarray(X, dtype=np.float64))


@dataclass
class CVResult:
    task: str
    backend: str
    oof_pred: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metrics: dict = field(default_factory=dict)


def _classification_metrics(y: np.ndarray, p: np.ndarray, specificities=(0.90, 0.95, 0.99)) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out: dict = {}
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "average_precision": float("nan")}
    out["auc"] = float(roc_auc_score(y, p))
    out["average_precision"] = float(average_precision_score(y, p))
    neg, pos = p[y == 0], p[y == 1]
    for spec in specificities:
        thr = float(np.quantile(neg, spec)) if len(neg) else float("nan")
        out[f"sens_at_spec_{spec:g}"] = float(np.mean(pos >= thr)) if len(pos) else float("nan")
        out[f"threshold_at_spec_{spec:g}"] = thr
    return out


def _regression_metrics(y: np.ndarray, yhat: np.ndarray, min_tf: float = 0.0) -> dict:
    from scipy.stats import pearsonr, spearmanr

    mask = np.isfinite(y) & np.isfinite(yhat) & (y >= min_tf)
    out = {"n_eval": int(mask.sum()), "min_tf": min_tf}
    if mask.sum() < 3:
        out.update({"pearson_r": float("nan"), "spearman_r": float("nan"), "mae": float("nan"), "rmse": float("nan")})
        return out
    ym, yh = y[mask], yhat[mask]
    out["pearson_r"] = float(pearsonr(ym, yh)[0])
    out["spearman_r"] = float(spearmanr(ym, yh)[0])
    out["mae"] = float(np.mean(np.abs(ym - yh)))
    out["rmse"] = float(np.sqrt(np.mean((ym - yh) ** 2)))
    return out


def leave_one_group_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    task: str = "classify",
    backend: str = "tabicl",
    device: str = "cpu",
    random_state: int = 0,
    regression_min_tf: float = 0.0,
) -> CVResult:
    """Leave-one-group-out CV (group = cohort/batch ⇒ AUC can't be batch memorisation)."""
    from sklearn.model_selection import LeaveOneGroupOut

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    resolved = backend

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if len(train_idx) < 2:
            continue
        if task == "classify" and len(np.unique(y[train_idx])) < 2:
            logger.warning("Skipping fold (group=%s): one class in train.", np.unique(groups[test_idx]))
            continue
        head = TabularTumourHead(task=task, backend=backend, device=device, random_state=random_state)
        head.fit(X[train_idx], y[train_idx])
        resolved = head.backend
        oof[test_idx] = head.predict_proba(X[test_idx]) if task == "classify" else head.predict(X[test_idx])

    if task == "classify":
        ev = np.isfinite(oof)
        metrics = _classification_metrics(y[ev].astype(int), oof[ev])
    else:
        metrics = _regression_metrics(y, oof, min_tf=regression_min_tf)
    metrics["n_oof"] = int(np.isfinite(oof).sum())
    return CVResult(task=task, backend=resolved, oof_pred=oof, y=y, groups=groups, metrics=metrics)
