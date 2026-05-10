import numpy as np

from tapestry.markers.robust_selection import (
    greedy_target_marker_selection,
    single_marker_outside_hull_scores,
    target_separability,
    weighted_condition_number,
)


def test_single_marker_outside_hull_scores_identify_target_specific_markers():
    reference_profiles = np.array([
        [0.10, 0.10, 0.80, 0.80],
        [0.20, 0.20, 0.70, 0.70],
        [0.90, 0.85, 0.75, 0.75],
    ])

    scores = single_marker_outside_hull_scores(
        reference_profiles,
        target_index=2,
        comparison_indices=[0, 1],
    )

    assert np.all(scores[:2] > 0.0)
    assert np.allclose(scores[2:], 0.0)


def test_target_separability_is_higher_on_target_specific_markers():
    reference_profiles = np.array([
        [0.10, 0.10, 0.80, 0.80],
        [0.20, 0.20, 0.70, 0.70],
        [0.90, 0.85, 0.75, 0.75],
    ])

    target_markers = np.array([0, 1])
    ambiguous_markers = np.array([2, 3])

    target_result = target_separability(
        reference_profiles,
        target_index=2,
        marker_indices=target_markers,
    )
    ambiguous_result = target_separability(
        reference_profiles,
        target_index=2,
        marker_indices=ambiguous_markers,
    )

    assert target_result.distance > 0.5
    assert ambiguous_result.distance < 1e-8
    assert target_result.distance > ambiguous_result.distance


def test_unknown_basis_reduces_target_separability_when_aligned_with_target():
    reference_profiles = np.array([
        [0.10, 0.10, 0.80, 0.80],
        [0.20, 0.20, 0.70, 0.70],
        [0.90, 0.85, 0.75, 0.75],
    ])
    markers = np.array([0, 1, 2, 3])
    healthy_mean = reference_profiles[:2].mean(axis=0)
    target_residual = reference_profiles[2] - healthy_mean
    unknown_basis = target_residual[:, None]

    without_unknown = target_separability(
        reference_profiles,
        target_index=2,
        marker_indices=markers,
        unknown_basis=None,
    )
    with_unknown = target_separability(
        reference_profiles,
        target_index=2,
        marker_indices=markers,
        unknown_basis=unknown_basis,
    )

    assert with_unknown.distance < without_unknown.distance
    assert with_unknown.distance < 1e-8


def test_greedy_target_marker_selection_prefers_target_specific_markers():
    reference_profiles = np.array([
        [0.10, 0.10, 0.80, 0.80],
        [0.20, 0.20, 0.70, 0.70],
        [0.90, 0.85, 0.75, 0.75],
    ])

    result = greedy_target_marker_selection(
        reference_profiles,
        target_index=2,
        candidate_indices=np.array([0, 1, 2, 3]),
        n_select=2,
    )

    assert set(result.selected_indices.tolist()) == {0, 1}
    assert result.distance_trace[-1] > 0.5


def test_weighted_condition_number_returns_finite_for_full_rank_marker_set():
    reference_profiles = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    condition = weighted_condition_number(
        reference_profiles,
        marker_indices=np.array([0, 1, 2]),
    )

    assert np.isfinite(condition)
    assert np.isclose(condition, 1.0)
