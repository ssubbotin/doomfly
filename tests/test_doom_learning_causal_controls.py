import numpy as np
import pytest


def test_original_boundaries():
    from doom_learning_v6.causal_controls import frame_ticks
    assert frame_ticks(7).tolist() == [286, 285, 286, 286, 286, 285, 286]


def test_duration_rotation_preserves_dose():
    from doom_learning_v6.causal_controls import duration_permutation, dose_proof
    ticks = np.array([286, 285, 286, 286, 286, 285, 286])
    source = np.array([0, .5, 1, 1.5, 3, 2.5, 4])
    shifted, mapping = duration_permutation(source, ticks)
    assert mapping.tolist() == [0, 5, 4, 6, 2, 1, 3]
    assert shifted.tolist() == [0, 2.5, 3, 4, 1, .5, 1.5]
    proof = dose_proof(source, shifted, ticks)
    assert proof['requested_current_ms'] == pytest.approx(357.2)
    assert proof['float32_current_ms'] == pytest.approx(357.2)


def test_plain_rotation_is_rejected():
    from doom_learning_v6.causal_controls import dose_proof
    with pytest.raises(ValueError):
        dose_proof([0, .5, 1, 1.5, 3, 2.5, 4], [0, 3, 2.5, 4, .5, 1, 1.5],
                   [286, 285, 286, 286, 286, 285, 286])


def test_balanced_error_is_equal_class_weighted():
    from doom_learning_v6.causal_controls import diagnostic_score
    score = diagnostic_score([0, 0, 0, 1, 0], [-2, -4, 0, 3, 9])
    assert score['class_mae_degrees'] == {'left': 3., 'idle': 0., 'right': 5.5}
    assert score['balanced_mae_degrees'] == pytest.approx(8.5 / 3)
    assert score['mae_degrees'] == pytest.approx(17 / 5)


def test_normalized_existing_displacement():
    from doom_learning_v6.causal_controls import learned_direction
    np.testing.assert_allclose(learned_direction([10, 20, 5], [9, 24, 5]),
                               [-.5, 1, 0], atol=1e-14)


@pytest.mark.parametrize('count', [0, -1, True, 1.5, np.uint64(2**63)])
def test_frame_count_rejections(count):
    from doom_learning_v6.causal_controls import frame_ticks
    with pytest.raises(ValueError):
        frame_ticks(count)


@pytest.mark.parametrize('ticks', [np.empty((0,), dtype=np.int64), np.array([[1]], dtype=np.int64),
                                   np.array([1.5]), np.array([0], dtype=np.int64),
                                   np.array([2**63], dtype=np.uint64)])
def test_schedule_rejections(ticks):
    from doom_learning_v6.causal_controls import dose_proof
    with pytest.raises(ValueError):
        dose_proof([0], [0], ticks)


def test_initialization_updates_only_plastic_state():
    from doom_learning_v6.causal_controls import initialize_efficacies

    class Brain:
        circuit = {'edges': np.array([1, 3], dtype=np.int64)}
        baseline_plastic = np.array([2., 4.], dtype=np.float32)
        weight = np.array([9., 2., 8., 4., 7.], dtype=np.float32)
        memory_u = np.zeros(2)
        memory_w = np.zeros(2)

    brain = Brain()
    initialize_efficacies(brain, [.1, -.2])
    np.testing.assert_allclose(brain.memory_u, [.1, -.2])
    np.testing.assert_allclose(brain.memory_w, [.1, -.2])
    np.testing.assert_allclose(brain.weight, [9., 2.2, 8., 3.2, 7.])


@pytest.mark.parametrize('currents', [[0, np.nan], [0, np.inf], [0, -1], [0, 5],
                                      [1, 2], []])
def test_invalid_current_schedules_are_rejected(currents):
    from doom_learning_v6.causal_controls import dose_proof
    with pytest.raises(ValueError):
        dose_proof(currents, currents, np.ones(len(currents), dtype=np.int64))


def test_mismatched_schedule_lengths_are_rejected():
    from doom_learning_v6.causal_controls import dose_proof
    with pytest.raises(ValueError):
        dose_proof([0, 1], [0], [1, 1])


@pytest.mark.parametrize('baseline,learned', [([0, 1], [1, 1]), ([1, np.inf], [1, 2]),
                                               ([1], [1, 2]), ([1, 1], [1, 1])])
def test_invalid_direction_vectors_are_rejected(baseline, learned):
    from doom_learning_v6.causal_controls import learned_direction
    with pytest.raises(ValueError):
        learned_direction(baseline, learned)


def test_diagnostic_score_reports_four_contiguous_blocks():
    from doom_learning_v6.causal_controls import diagnostic_score
    result = diagnostic_score(np.zeros(8), np.zeros(8))
    assert [(b['start'], b['end_exclusive']) for b in result['contiguous_blocks']] == [
        (0, 2), (2, 4), (4, 6), (6, 8)]


def test_invalid_initialization_does_not_mutate_state():
    from doom_learning_v6.causal_controls import initialize_efficacies

    class Brain:
        circuit = {'edges': np.array([1, 3], dtype=np.int64)}
        baseline_plastic = np.array([2., 4.], dtype=np.float32)
        weight = np.array([9., 2., 8., 4., 7.], dtype=np.float32)
        memory_u = np.array([.2, .3])
        memory_w = np.array([.4, .5])

    brain = Brain()
    before = (brain.weight.copy(), brain.memory_u.copy(), brain.memory_w.copy())
    with pytest.raises(ValueError):
        initialize_efficacies(brain, [np.nan, 0])
    for actual, expected in zip((brain.weight, brain.memory_u, brain.memory_w), before):
        np.testing.assert_array_equal(actual, expected)
