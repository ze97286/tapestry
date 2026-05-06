import numpy as np

from tapestry.benchmark.augmented_nnls import (
    project_basis_orthogonal_to,
    run_augmented_nnls,
    run_augmented_nnls_path,
    run_augmented_nnls_regularized,
    target_contrast,
)
from tapestry.benchmark.nnls import run_weighted_nnls


def test_project_basis_orthogonal_to_removes_target_direction():
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    U = np.eye(4)[:, :2]

    projected = project_basis_orthogonal_to(U, direction)

    assert projected.shape == (4, 1)
    assert np.allclose(projected.T @ direction, 0.0, atol=1e-12)
    assert np.allclose(projected.T @ projected, np.eye(1), atol=1e-12)


def test_target_contrast_uses_target_minus_other_mean():
    reference_profiles = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    contrast = target_contrast(reference_profiles, target_index=0)

    assert np.allclose(contrast, np.array([1.0, -0.5, -0.5]))


def test_regularized_path_suppresses_unknown_channel():
    reference_profiles = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.9, 0.9, 0.2, 0.2],
    ])
    U = np.array([[0.0], [0.0], [1.0], [-1.0]]) / np.sqrt(2.0)
    X = np.array([[0.2, 0.2, 0.3, 0.1]])
    coverage = np.full_like(X, 100.0)

    path = run_augmented_nnls_path(
        X, coverage, reference_profiles, U, lambda_values=[0.0, 1_000_000.0]
    )

    assert path["proportions"].shape == (2, 1, 2)
    assert np.allclose(path["proportions"].sum(axis=2), 1.0)
    assert path["unknown_mag"][0, 0] > path["unknown_mag"][1, 0]
    assert path["residual_norm"][0, 0] < path["residual_norm"][1, 0]


def test_unregularized_wrapper_matches_lambda_zero():
    reference_profiles = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.9, 0.9, 0.2, 0.2],
    ])
    U = np.array([[0.0], [0.0], [1.0], [-1.0]]) / np.sqrt(2.0)
    X = np.array([
        [0.2, 0.2, 0.3, 0.1],
        [0.55, 0.55, 0.2, 0.2],
    ])
    coverage = np.full_like(X, 50.0)

    prop_old, coef_old, mag_old = run_augmented_nnls(
        X, coverage, reference_profiles, U
    )
    prop_new, coef_new, mag_new, resid_new = run_augmented_nnls_regularized(
        X, coverage, reference_profiles, U, lambda_unknown=0.0, simplex_known=False
    )

    assert np.allclose(prop_old, prop_new)
    assert np.allclose(coef_old, coef_new)
    assert np.allclose(mag_old, mag_new)
    assert np.all(np.isfinite(resid_new))


def test_simplex_constrained_augmented_fit_returns_simplex_rows():
    reference_profiles = np.array([
        [0.2, 0.2, 0.2, 0.2],
        [0.9, 0.9, 0.2, 0.2],
    ])
    U = np.array([[0.0], [0.0], [1.0], [-1.0]]) / np.sqrt(2.0)
    X = np.array([
        [0.2, 0.2, 0.3, 0.1],
        [0.55, 0.55, 0.2, 0.2],
    ])
    coverage = np.full_like(X, 50.0)

    prop, coef, mag, resid = run_augmented_nnls_regularized(
        X,
        coverage,
        reference_profiles,
        U,
        lambda_unknown=10.0,
        simplex_known=True,
    )

    assert prop.shape == (2, 2)
    assert coef.shape == (2, 1)
    assert np.allclose(prop.sum(axis=1), 1.0)
    assert np.all(prop >= 0.0)
    assert np.all(np.isfinite(mag))
    assert np.all(np.isfinite(resid))


def test_high_lambda_path_converges_to_weighted_nnls():
    reference_profiles = np.array([
        [0.1, 0.2, 0.2, 0.1],
        [0.8, 0.7, 0.2, 0.2],
        [0.2, 0.2, 0.8, 0.9],
    ])
    U = np.array([
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
    ]) / np.sqrt(2.0)
    X = np.array([
        [0.25, 0.25, 0.35, 0.15],
        [0.6, 0.55, 0.25, 0.25],
    ])
    coverage = np.array([
        [20.0, 50.0, 100.0, 10.0],
        [80.0, 20.0, 40.0, 60.0],
    ])

    nnls_props = run_weighted_nnls(X, coverage, reference_profiles)
    aug_props, coef, mag, _ = run_augmented_nnls_regularized(
        X, coverage, reference_profiles, U, lambda_unknown=1e10
    )

    assert np.allclose(aug_props, nnls_props, atol=1e-5)
    assert np.all(np.abs(coef) < 1e-5)
    assert np.all(mag < 1e-5)
