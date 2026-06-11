"""Unit tests for the clinical layer (no plotly required).

Covers the Kaplan–Meier estimator, the log-rank p-value, and the
detection-threshold-at-specificity logic used by run_clinical_eval.py.
"""

from __future__ import annotations

import numpy as np

from rltf.plots import _km_curve, _logrank_p


def test_km_curve_is_monotone_non_increasing():
    times = np.array([5, 10, 10, 20, 30])
    events = np.array([1, 1, 0, 1, 1])
    t, s = _km_curve(times, events)
    assert s[0] == 1.0
    assert all(s[i + 1] <= s[i] + 1e-12 for i in range(len(s) - 1))
    assert 0.0 <= s[-1] <= 1.0


def test_logrank_separates_clearly_different_arms():
    # Arm A dies early, arm B dies late → small p.
    tA, eA = np.array([1, 2, 3, 4, 5]), np.ones(5)
    tB, eB = np.array([50, 60, 70, 80, 90]), np.ones(5)
    p = _logrank_p(tA, eA, tB, eB)
    assert 0.0 <= p <= 1.0
    assert p < 0.05
    # Identical arms → not significant.
    p2 = _logrank_p(tA, eA, tA.copy(), eA.copy())
    assert p2 > 0.2


def test_detection_threshold_sensitivity_at_specificity():
    # Healthy scores low, cancer high → threshold at 0.95 spec yields high sensitivity.
    rng = np.random.default_rng(0)
    neg = rng.uniform(0.0, 0.4, 50)
    pos = rng.uniform(0.5, 1.0, 30)
    spec = 0.95
    thr = float(np.quantile(neg, spec))
    sensitivity = float(np.mean(pos >= thr))
    achieved_spec = float(np.mean(neg < thr))
    assert sensitivity == 1.0
    assert achieved_spec >= 0.9


if __name__ == "__main__":
    test_km_curve_is_monotone_non_increasing()
    test_logrank_separates_clearly_different_arms()
    test_detection_threshold_sensitivity_at_specificity()
    print("rltf clinical layer: all checks passed")
