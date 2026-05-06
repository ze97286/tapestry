import numpy as np

from tapestry.benchmark.robust_lower_bound import (
    calibrate_robust_lower_bound,
    fit_factor_nuisance,
    solve_theta_interval,
)


def test_factor_nuisance_mahalanobis_is_finite():
    residuals = np.array([
        [0.01, 0.00, -0.01, 0.02],
        [0.00, 0.01, -0.02, 0.01],
        [0.02, 0.01, -0.01, 0.00],
        [0.01, -0.01, 0.00, 0.02],
    ])

    model = fit_factor_nuisance(residuals, rank=1)
    q = model.mahalanobis(residuals[0])

    assert np.isfinite(q)
    assert q >= 0.0


def test_robust_lower_bound_detects_required_target_signal():
    cell_types = ["HealthyA", "HealthyB", "OAC"]
    reference_profiles = np.array([
        [0.1, 0.1, 0.2, 0.2, 0.1, 0.2],
        [0.2, 0.2, 0.1, 0.1, 0.2, 0.1],
        [0.8, 0.8, 0.7, 0.7, 0.8, 0.7],
    ])
    healthy_props = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.5, 0.5],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.2, 0.8],
        [0.65, 0.35],
        [0.35, 0.65],
    ])
    noise = np.array([0.002, -0.001, 0.001, -0.002, 0.001, -0.001])
    controls = healthy_props @ reference_profiles[:2]
    controls = controls + np.arange(healthy_props.shape[0])[:, None] * 0.0001 * noise
    tumour = 0.2 * reference_profiles[2] + 0.8 * (np.array([0.5, 0.5]) @ reference_profiles[:2])
    X = np.vstack([controls, tumour])
    coverage = np.full_like(X, 100.0)
    control_mask = np.array([True] * len(controls) + [False])

    calibration = calibrate_robust_lower_bound(
        X,
        coverage,
        reference_profiles,
        cell_types,
        target_cell_type="OAC",
        control_mask=control_mask,
        alpha=0.2,
        factor_rank=0,
        min_control_coverage_fraction=1.0,
        random_seed=3,
    )
    healthy_interval = solve_theta_interval(
        X[0], coverage[0], reference_profiles, calibration
    )
    tumour_interval = solve_theta_interval(
        X[-1], coverage[-1], reference_profiles, calibration
    )

    assert healthy_interval["theta_min"] < 1e-4
    assert tumour_interval["theta_min"] > 0.05
    assert tumour_interval["theta_max"] >= tumour_interval["theta_min"]
