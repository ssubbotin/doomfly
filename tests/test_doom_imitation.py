"""Causal reinforcement tests; numeric fixtures establish no biological claim."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_doom_demonstrations import checked_episode_fixture
from test_doom_learning_v6 import brain as memory_brain
from doom_learning_v6.demonstrations import OfflineEpisode, OfflineFrame
from doom_learning_v6 import imitation


READOUTS = [{'index': 0, 'type': 'DNp20', 'side': 'R'},
            {'index': 1, 'type': 'DNp20', 'side': 'L'},
            {'index': 3, 'type': 'DNpe017', 'side': 'R'}]


def numeric_brain(tmp_path):
    b = memory_brain(tmp_path)
    b.tonic[0] = 20
    b.tonic[2] = 12
    b.rgb_calls = []
    def rgb_step(rgb, ms, *, learning=False, stimulation=None):
        b.rgb_calls.append((b.cursor, round(ms * 10), learning,
                            None if stimulation is None else float(stimulation[1])))
        return b.step([], ms, learning=learning, stimulation=stimulation, lamina_bias=0)
    b.rgb_step = rgb_step
    return b


class Frames:
    def __init__(self, targets, fail_at=None):
        self.targets = np.asarray(targets, dtype=float)
        self.frame_count = len(targets)
        self.width = self.height = 2
        self.actions = np.zeros((len(targets), 9), dtype=np.float32)
        self.identity = {'dataset': 'fixture', 'dataset_revision': 'numeric',
                         'scenario': 'defend_the_center', 'episode_index': 0,
                         'video_sha256': 'a', 'control_sha256': 'b'}
        self.fail_at = fail_at
        self.closed = False
    def turn_targets(self):
        return self.targets.copy()
    def iter_frames(self, limit=None):
        try:
            for i in range(min(limit or self.frame_count, self.frame_count)):
                if i == self.fail_at:
                    raise RuntimeError('decoder failed')
                yield OfflineFrame(i, float(np.float32(i / 35)),
                                   np.zeros((2, 2, 3), dtype=np.uint8), self.actions[i])
        finally:
            self.closed = True


def test_teacher_error_is_bounded_and_symmetric():
    assert imitation.teacher_error(0, -3) == .5
    assert imitation.teacher_error(0, 3) == .5
    assert imitation.teacher_error(-6, 6) == 1
    assert imitation.teacher_error(2, 2) == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_teacher_and_scores_rejected(value):
    with pytest.raises(ValueError): imitation.teacher_error(value, 0)
    with pytest.raises(ValueError): imitation.teacher_error(0, value)
    with pytest.raises(ValueError): imitation.score_turns([0], [value])


@pytest.mark.parametrize('predictions,targets', [([], []), ([0], []), ([[0]], [[0]])])
def test_invalid_score_inputs_rejected(predictions, targets):
    with pytest.raises(ValueError): imitation.score_turns(predictions, targets)


def test_direction_recall_uses_independent_present_class_supports_and_baselines():
    score = imitation.score_turns([-1, 1, 0, .1, -.1], [-2, -2, 2, 0, 0])
    assert score['mae_degrees'] == pytest.approx(1.24)
    assert score['direction_support'] == {'left': 2, 'idle': 2, 'right': 1}
    assert score['direction_recall'] == {'left': .5, 'idle': 1., 'right': 0.}
    assert score['balanced_direction_recall'] == .5
    assert score['constant_baselines']['idle']['mae_degrees'] == 1.2
    one = imitation.score_turns([0, 0], [2, 2])
    assert one['direction_support'] == {'left': 0, 'idle': 0, 'right': 2}
    assert one['direction_recall'] == {'right': 0.}
    assert one['constant_baselines']['right']['mae_degrees'] == 0


def test_actual_memory_runner_delays_teacher_and_retains_exact_neural_time(tmp_path):
    b = numeric_brain(tmp_path); initial = b.weight.copy(); ptr = b.ptr.copy(); post = b.post.copy()
    data = Frames([-3] * 35)
    result = imitation.episode(b, data, READOUTS, training=True, frozen=False, warmup_ms=0)
    assert result['brain_steps'] == 10000 and b.cursor == 10000
    assert result['trace'][0]['teacher_current'] == 0
    assert b.rgb_calls[1][3] == pytest.approx(4 * result['trace'][0]['teacher_error'])
    assert result['trace'][-1]['brain_steps'] == 10000
    assert data.closed
    assert b.weight[0] != initial[0]
    np.testing.assert_array_equal(b.weight[1:], initial[1:])
    np.testing.assert_array_equal(b.ptr, ptr); np.testing.assert_array_equal(b.post, post)


def test_frozen_arm_still_receives_teacher_but_preserves_efficacies(tmp_path):
    b = numeric_brain(tmp_path); initial = b.weight.copy()
    result = imitation.episode(b, Frames([-3] * 8), READOUTS,
                               training=True, frozen=True, warmup_ms=0)
    assert result['teacher_dose_current_ms'] > 0
    np.testing.assert_array_equal(b.weight, initial)
    assert b.rate_kc.any()


def test_teacher_free_labels_never_change_controls_or_impose_current(tmp_path):
    b = numeric_brain(tmp_path)
    a = imitation.episode(b, Frames([-3] * 8), READOUTS, training=False, frozen=True, warmup_ms=0)
    c = imitation.episode(b, Frames([3] * 8), READOUTS, training=False, frozen=True, warmup_ms=0)
    assert [r['action'] for r in a['trace']] == [r['action'] for r in c['trace']]
    assert all(r['teacher_current'] == 0 for r in a['trace'] + c['trace'])


def test_shift_preserves_frames_raw_controls_and_first_action(tmp_path):
    b = numeric_brain(tmp_path); data = Frames([-3, -3, 3, 3])
    a = imitation.episode(b, data, READOUTS, training=True, frozen=True, warmup_ms=0)
    c = imitation.episode(b, data, READOUTS, training=True, frozen=True, target_shift=2, warmup_ms=0)
    assert a['trace'][0]['action'] == c['trace'][0]['action']
    assert [r['teacher_index'] for r in c['trace']] == [2, 3, 0, 1]
    assert [r['raw_action'] for r in a['trace']] == [r['raw_action'] for r in c['trace']]
    assert [r['frame_sha256'] for r in a['trace']] == [r['frame_sha256'] for r in c['trace']]
    assert a['teacher_dose_current_ms'] != c['teacher_dose_current_ms']


def test_episode_failure_preserves_partial_trace_and_closes_frames(tmp_path):
    b = numeric_brain(tmp_path); data = Frames([-3] * 5, fail_at=2)
    with pytest.raises(imitation.EpisodeFailure) as error:
        imitation.episode(b, data, READOUTS, training=True, frozen=False, warmup_ms=0)
    assert len(error.value.summary['trace']) == 2
    assert error.value.summary['complete'] is False and data.closed


@pytest.mark.parametrize('kwargs', [{'limit': 0}, {'warmup_ms': -1}, {'target_shift': 1.2}])
def test_invalid_episode_parameters_fail_before_neural_input(tmp_path, kwargs):
    b = numeric_brain(tmp_path)
    with pytest.raises(ValueError):
        imitation.episode(b, Frames([0] * 4), READOUTS, training=True, frozen=False, **kwargs)
    assert b.rgb_calls == []


def test_reader_integration_preserves_float32_timestamp_and_source_controls(tmp_path, checked_episode_fixture):
    data = OfflineEpisode(checked_episode_fixture(turns=[-1] * 8))
    result = imitation.episode(numeric_brain(tmp_path), data, READOUTS,
                               training=False, frozen=True, limit=3, warmup_ms=0)
    assert result['complete'] is False
    assert result['trace'][2]['timestamp'] == float(np.float32(2 / 35))
    assert result['trace'][0]['raw_action'] == [0, 0, 0, 0, 0, 0, 1, 0, 0]


def test_cli_rejects_leakage_and_invalid_counts_before_model(tmp_path, checked_episode_fixture, monkeypatch):
    directory = checked_episode_fixture()
    monkeypatch.setattr(imitation, 'calibrated_brain', lambda *a, **k: pytest.fail('model constructed'))
    args = imitation.build_parser().parse_args(['--train', str(directory), '--eval', str(directory),
                                               '--out', str(tmp_path / 'run')])
    with pytest.raises(ValueError, match='overlap'): imitation.run(args)
    args.epochs = 0
    with pytest.raises(ValueError): imitation.run(args)
    assert not (tmp_path / 'run').exists()


def test_cohort_restores_learned_memory_independently_and_runs_controls(tmp_path):
    b = numeric_brain(tmp_path); train = Frames([-3] * 4); held = Frames([3] * 4)
    out = tmp_path / 'run'; out.mkdir(); initial = out / 'initial.npz'; b.checkpoint(initial)
    args = SimpleNamespace(epochs=1, max_frames=None, target_shift=2)
    result = imitation._cohort(b, [train], [held, held], READOUTS, args, out, initial)
    assert set(r['arm'] for r in result) == {'plastic', 'frozen', 'shifted'}
    for arm in ['plastic', 'frozen', 'shifted']:
        evaluations = [r for r in result if r['arm'] == arm and r['phase'] == 'held_out']
        assert evaluations[0]['before'] == evaluations[1]['before']
        assert evaluations[0]['score'] == evaluations[1]['score']
        assert evaluations[0]['teacher_dose_current_ms'] == 0
    assert len([r for r in result if r['phase'] == 'retention_5s']) == 2
    assert len([r for r in result if r['phase'] == 'memory_erased']) == 2
    retention = [r for r in result if r['phase'] == 'retention_5s'][0]['retention']
    assert retention['brain_steps'] == 50000
    assert retention['before']['memory_u_sha256'] != retention['after']['memory_u_sha256']
    assert retention['teacher_current'] == 0 and retention['weights_frozen'] is False
    erased = [r for r in result if r['phase'] == 'memory_erased'][0]
    assert erased['before']['changed_edges'] == 0
    assert (out / 'plastic' / 'learned.npz').exists()
    assert not (out / 'results.json').exists()


def test_cohort_failure_writes_trace_and_failure_checkpoint(tmp_path):
    b = numeric_brain(tmp_path); out = tmp_path / 'run'; out.mkdir()
    initial = out / 'initial.npz'; b.checkpoint(initial)
    with pytest.raises(imitation.EpisodeFailure):
        imitation._cohort(b, [Frames([-3] * 4, fail_at=2)], [Frames([0] * 4)], READOUTS,
                          SimpleNamespace(epochs=1, max_frames=None, target_shift=2), out, initial)
    record = json.loads((out / 'plastic' / 'train-0-0' / 'episode.json').read_text())
    assert len(record['trace']) == 2 and record['complete'] is False
    assert (out / 'plastic' / 'train-0-0' / 'failure.npz').exists()


def test_memory_records_include_hidden_memory_state_hashes(tmp_path):
    b = numeric_brain(tmp_path)
    result = imitation.episode(b, Frames([-3] * 4), READOUTS,
                               training=True, frozen=False, warmup_ms=0)
    assert len(result['trace'][0]['memory']['memory_u_sha256']) == 64
    assert result['before']['memory_u_sha256'] != result['after']['memory_u_sha256']


def test_decoder_close_failure_preserves_completed_trace(tmp_path):
    data = Frames([0] * 3)
    generator = data.iter_frames
    class ClosingFrames:
        def __iter__(self): return generator()
        def close(self): raise RuntimeError('decoder reap failed')
    data.iter_frames = lambda limit=None: ClosingFrames()
    with pytest.raises(imitation.EpisodeFailure) as error:
        imitation.episode(numeric_brain(tmp_path), data, READOUTS,
                          training=False, frozen=True, warmup_ms=0)
    assert len(error.value.summary['trace']) == 3
    assert error.value.summary['complete'] is False


def test_partial_warmup_reports_actual_neural_steps(tmp_path):
    b = numeric_brain(tmp_path)
    original = b.rgb_step
    def failed_warmup(rgb, ms, **kwargs):
        original(rgb, 10, **kwargs)
        raise RuntimeError('warmup interrupted')
    b.rgb_step = failed_warmup
    with pytest.raises(imitation.EpisodeFailure) as error:
        imitation.episode(b, Frames([0] * 4), READOUTS, training=True, frozen=False)
    assert error.value.summary['warmup']['brain_steps'] == 100
    assert error.value.summary['brain_steps'] == 0


def test_cohort_rejects_nonplastic_weight_mutation_and_saves_failure_state(tmp_path):
    b = numeric_brain(tmp_path); original = b.rgb_step
    def corrupted(rgb, ms, **kwargs):
        result = original(rgb, ms, **kwargs)
        b.weight[1] += .01
        return result
    b.rgb_step = corrupted
    out = tmp_path / 'run'; out.mkdir(); initial = out / 'initial.npz'; b.checkpoint(initial)
    with pytest.raises(ValueError, match='Immutable'):
        imitation._cohort(b, [Frames([-3] * 4)], [Frames([0] * 4)], READOUTS,
                          SimpleNamespace(epochs=1, max_frames=None, target_shift=2), out, initial)
    assert (out / 'plastic' / 'train-0-0' / 'failure.npz').exists()
    assert (out / 'plastic' / 'train-0-0' / 'episode.json').exists()


def test_invalid_targets_are_rejected_before_reset_or_warmup(tmp_path):
    b = numeric_brain(tmp_path); b.cursor = 123
    for values in [[np.nan], [[0]], [np.inf]]:
        with pytest.raises(ValueError):
            imitation.episode(b, Frames(values), READOUTS, training=True, frozen=False)
    assert b.cursor == 123 and b.rgb_calls == []


def test_cli_validates_incompatible_heldout_before_construction(tmp_path, checked_episode_fixture, monkeypatch):
    train = checked_episode_fixture()
    held = checked_episode_fixture(scenario='basic')
    monkeypatch.setattr(imitation, 'calibrated_brain', lambda *a, **k: pytest.fail('model constructed'))
    args = imitation.build_parser().parse_args(['--train', str(train), '--eval', str(held),
                                               '--out', str(tmp_path / 'run')])
    with pytest.raises(ValueError, match='Unsupported scenario'): imitation.run(args)
    assert not (tmp_path / 'run').exists()


def test_cli_refuses_fixture_as_execution_model(tmp_path, checked_episode_fixture, monkeypatch):
    train = checked_episode_fixture(count=8)
    held = checked_episode_fixture(count=12)
    b = numeric_brain(tmp_path)
    monkeypatch.setattr(imitation, 'calibrated_brain', lambda *a, **k: b)
    monkeypatch.setattr(imitation, 'capture_provenance', lambda *a, **k: {})
    args = imitation.build_parser().parse_args(['--train', str(train), '--eval', str(held),
                                               '--out', str(tmp_path / 'run')])
    with pytest.raises(ValueError, match='Exact full retained'): imitation.run(args)
    assert (tmp_path / 'run' / 'failure.npz').exists()
    public = json.loads((tmp_path / 'run' / 'failure.json').read_text())
    assert str(tmp_path) not in json.dumps(public)
    inputs = (tmp_path / 'run' / 'inputs.jsonl').read_text()
    assert str(train.resolve()) in inputs and str(held.resolve()) in inputs
