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


def test_diagnostic_score_preserves_existing_score_contract():
    from doom_learning_v6.causal_controls import diagnostic_score
    from doom_learning_v6.imitation import score_turns
    predictions, targets = [0, 1, -2], [-1, 2, 0]
    expected = score_turns(predictions, targets)
    actual = diagnostic_score(predictions, targets)
    for key in ('mae_degrees', 'direction_support', 'direction_recall',
                'balanced_direction_recall', 'dead_zone_degrees',
                'constant_baselines', 'constant_baseline_definition'):
        assert actual[key] == expected[key]


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
                                   np.array([-1], dtype=np.int64),
                                   np.array([2**63], dtype=np.uint64)])
def test_schedule_rejections(ticks):
    from doom_learning_v6.causal_controls import dose_proof
    with pytest.raises(ValueError):
        dose_proof([0], [0], ticks)


def test_initialization_updates_only_plastic_state(tmp_path):
    from doom_learning_v6.causal_controls import initialize_efficacies
    from test_doom_learning_v6 import brain as make_brain
    brain = make_brain(tmp_path)
    fixed = brain.weight[1:].copy()
    rate_kc, rate_dan = brain.rate_kc.copy(), brain.rate_dan.copy()
    initialize_efficacies(brain, [.1])
    np.testing.assert_allclose(brain.memory_u, [.1])
    np.testing.assert_allclose(brain.memory_w, [.1])
    np.testing.assert_allclose(brain.weight[brain.circuit['edges']], [22.])
    np.testing.assert_array_equal(brain.weight[1:], fixed)
    np.testing.assert_array_equal(brain.rate_kc, rate_kc)
    np.testing.assert_array_equal(brain.rate_dan, rate_dan)


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


@pytest.mark.parametrize('baseline,learned', [([0, 1], [1, 1]), ([-1, 1], [1, 2]),
                                               pytest.param([np.nan, 1], [1, 2], id='existing-guard-nan-baseline'),
                                               pytest.param([np.inf, 1], [1, 2], id='existing-guard-inf-baseline'),
                                               ([1, 1], [np.nan, 2]), ([1, 1], [np.inf, 2]),
                                               ([1], [1, 2]), ([1, 1], [1, 1])])
def test_invalid_direction_vectors_are_rejected(baseline, learned):
    """Characterize the existing finite-baseline and direction guards."""
    from doom_learning_v6.causal_controls import learned_direction
    with pytest.raises(ValueError):
        learned_direction(baseline, learned)


def test_diagnostic_score_reports_four_contiguous_blocks():
    from doom_learning_v6.causal_controls import diagnostic_score
    predictions = np.array([-2, .2, 1, -1, 0, 2, -.2, 3.])
    targets = np.array([-1, 0, 2, -2, .5, -.5, 1, -3.])
    result = diagnostic_score(predictions, targets)
    assert [(b['start'], b['end_exclusive']) for b in result['contiguous_blocks']] == [
        (0, 2), (2, 4), (4, 6), (6, 8)]
    assert [b['mae_degrees'] for b in result['contiguous_blocks']] == pytest.approx([.6, 1., 1.5, 3.6])
    assert [b['direction_support'] for b in result['contiguous_blocks']] == [
        {'left': 1, 'idle': 1, 'right': 0}, {'left': 1, 'idle': 0, 'right': 1},
        {'left': 1, 'idle': 0, 'right': 1}, {'left': 1, 'idle': 0, 'right': 1}]
    assert sum(b['end_exclusive'] - b['start'] for b in result['contiguous_blocks']) == 8


def test_invalid_initialization_does_not_mutate_state(tmp_path):
    from doom_learning_v6.causal_controls import initialize_efficacies
    from test_doom_learning_v6 import brain as make_brain
    brain = make_brain(tmp_path)
    before = {name: value.copy() for name, value in vars(brain).items()
              if isinstance(value, np.ndarray)}
    with pytest.raises(ValueError):
        initialize_efficacies(brain, [np.nan, 0])
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(brain, name), expected)


def test_owned_brain_rejected_before_mutation(tmp_path):
    from doom_learning_v6.causal_controls import initialize_efficacies
    from test_doom_learning_v6 import brain as make_brain
    brain = make_brain(tmp_path)
    brain._metal_batch_owner = object()
    before = {name: value.copy() for name, value in vars(brain).items()
              if isinstance(value, np.ndarray)}
    with pytest.raises(ValueError):
        initialize_efficacies(brain, [0.1])
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(brain, name), expected)


def test_float32_dose_proof_covers_fractional_amplitudes():
    from doom_learning_v6.causal_controls import dose_proof
    import math
    ticks = np.array([286, 285, 286, 286, 286, 285, 286])
    source = np.array([0, .1234567, 1.234567, 1.75, 2.25, 3.125, 4.])
    shifted = np.array([0, 3.125, 2.25, 4., 1.234567, .1234567, 1.75])
    proof = dose_proof(source, shifted, ticks)
    requested = math.fsum(float(v) * int(dt) * .1 for v, dt in zip(source, ticks))
    float32 = math.fsum(float(v) * int(dt) * .1
                        for v, dt in zip(source.astype(np.float32), ticks))
    assert proof['duration_groups'] == {'285': 2, '286': 5}
    assert proof['requested_current_ms'] == requested
    assert proof['float32_current_ms'] == float32
    assert proof['requested_current_ms'] != proof['float32_current_ms']
    assert proof['exact'] is True
