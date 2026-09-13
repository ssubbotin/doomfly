"""Resident causal runner contracts, including real native RGB trajectories."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


READOUTS = [
    {'index': 0, 'type': 'DNp20', 'side': 'L'},
    {'index': 2, 'type': 'DNp20', 'side': 'R'},
    {'index': 1, 'type': 'DNpe017', 'side': 'L'},
    {'index': 3, 'type': 'DNpe017', 'side': 'R'},
]


class _ValidationData:
    frame_count = 3
    height = 3
    width = 4

    def __init__(self):
        self.iterated = False

    def turn_targets(self):
        return np.zeros(self.frame_count, dtype=np.float64)

    def iter_frames(self):
        self.iterated = True
        raise AssertionError('validation must precede frame iteration')


class _ResetProbe:
    def __init__(self):
        self.reset_count = 0

    def reset(self, *, keep_memory):
        assert keep_memory is True
        self.reset_count += 1


def _replay_episode():
    from doom_learning_v6.causal_pilot import replay_episode
    return replay_episode


@pytest.mark.parametrize(
    'learning,frozen,currents,reverse_lanes',
    [
        ([True, 1], [False, False], np.zeros((2, 3)), False),
        ([True, False], [False, np.bool_(False)], np.zeros((2, 3)), False),
        ([True, False], [False, False], np.zeros((2, 2)), False),
        ([True, False], [False, False], np.zeros((2, 3)), True),
    ],
)
def test_replay_rejects_bad_schedule_flags_and_lane_order_before_reset_or_iteration(
        tmp_path, learning, frozen, currents, reverse_lanes):
    """Changing preflight validation must never permit an early state mutation."""
    data = _ValidationData()
    brains = [_ResetProbe(), _ResetProbe()]
    executor = SimpleNamespace(brains=tuple(reversed(brains) if reverse_lanes else brains))

    with pytest.raises((TypeError, ValueError)):
        _replay_episode()(brains, executor, data, READOUTS, currents,
                          learning=learning, frozen=frozen, directory=tmp_path / 'phase',
                          warmup_ms=0)

    assert [brain.reset_count for brain in brains] == [0, 0]
    assert data.iterated is False


@pytest.mark.parametrize('warmup,existing_phase,frame_count', [
    (float('nan'), False, 3),
    (-.1, False, 3),
    (0., True, 3),
    (0., False, 0),
])
def test_replay_rejects_warmup_phase_freshness_and_frame_count_before_mutation(
        tmp_path, warmup, existing_phase, frame_count):
    """Changing preflight ordering must not reset a lane for an invalid phase."""
    data = _ValidationData()
    data.frame_count = frame_count
    brains = [_ResetProbe(), _ResetProbe()]
    directory = tmp_path / 'phase'
    if existing_phase:
        directory.mkdir()
    with pytest.raises(ValueError):
        _replay_episode()(brains, SimpleNamespace(brains=tuple(brains)), data, READOUTS,
                          np.zeros((2, 3)), learning=[False, False], frozen=[True, True],
                          directory=directory, warmup_ms=warmup)
    assert [brain.reset_count for brain in brains] == [0, 0]
    assert data.iterated is False


def _cli(arguments):
    environment = {**os.environ, 'OPENBLAS_NUM_THREADS': '1'}
    return subprocess.run([sys.executable, '-m', 'doom_learning_v6.causal_pilot', *arguments],
                          text=True, capture_output=True, env=environment, check=False)


def test_cli_rejects_fresh_output_and_source_commit_before_platform_or_native_import(tmp_path):
    """Freshness and pinned source identity are portable CLI preflight gates."""
    existing = tmp_path / 'existing'; existing.mkdir()
    result = _cli(['--mode', 'sensitivity', '--train', 'missing-train', '--eval', 'missing-eval',
                   '--reference', 'missing-reference', '--out', str(existing),
                   '--source-commit', '0' * 40])
    assert result.returncode != 0
    assert 'Fresh output directory required' in result.stderr + result.stdout

    fresh = tmp_path / 'fresh'
    result = _cli(['--mode', 'causal', '--train', 'missing-train', '--eval', 'missing-eval',
                   '--reference', 'missing-reference', '--out', str(fresh),
                   '--source-commit', '0' * 40])
    assert result.returncode != 0
    assert 'Source commit mismatch' in result.stderr + result.stdout
    assert not fresh.exists()


class _ThreeFrames:
    """Only video acquisition is synthetic. The neural execution remains native."""
    frame_count = 3
    height = 3
    width = 4

    def __init__(self, *, fail_after=None):
        self.fail_after = fail_after
        self.closed = False
        self.actions = np.zeros((self.frame_count, 9), dtype=np.float32)
        self.identity = {'dataset': 'synthetic', 'dataset_revision': 'native-test',
                         'scenario': 'defend_the_center', 'episode_index': 0,
                         'video_sha256': '0' * 64, 'control_sha256': '1' * 64}

    def turn_targets(self):
        return np.array([0., 1.7578125, -1.7578125], dtype=np.float64)

    def iter_frames(self):
        try:
            for index, interval in enumerate((28.6, 28.5, 28.6)):
                if self.fail_after is not None and index > self.fail_after:
                    raise RuntimeError('controlled frame iterator failure')
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                image[0, 0, 2] = (255, 128, 64)[index]
                image[2, 3, 1] = (64, 128, 255)[index]
                image[0, 2] = (100, 120, 140)
                yield SimpleNamespace(index=index, timestamp=float(np.float32(index / 35)),
                                      rgb=image, action=self.actions[index], interval_ms=interval)
        finally:
            self.closed = True


def _native_serial(tmp_path, data, currents, checkpoint, *, frozen):
    """Independent serial Metal calls are the reference, not runner internals."""
    from doom.engine import NeuralControls
    from test_doom_metal_batch_rgb import visual_brain

    brain = visual_brain(tmp_path, backend='metal')
    brain.restore(checkpoint)
    brain.reset(keep_memory=True)
    brain.weights_frozen = frozen
    control = NeuralControls(READOUTS, mode='bci')
    trace = []
    for sample, interval, current in zip(data.iter_frames(), (28.6, 28.5, 28.6), currents):
        counts, _ = brain.rgb_step(sample.rgb, interval, learning=True,
                                   stimulation=(brain.circuit['dan'], float(current)) if current else None)
        trace.append((counts.copy(), control.decode(counts, interval * .001)))
    return brain, trace


def _without_wall(value):
    if isinstance(value, dict):
        return {key: _without_wall(item) for key, item in value.items()
                if 'wall' not in key and 'elapsed' not in key}
    if isinstance(value, list):
        return [_without_wall(item) for item in value]
    return value


@pytest.mark.skipif(sys.platform != 'darwin', reason='Real Metal requires macOS')
def test_metal_replay_matches_independent_serial_trajectory_and_frozen_lane(tmp_path):
    """A resident 18-tick executor must preserve serial RGB, rule and BCI results."""
    from doom_learning.common import digest
    from test_doom_metal_batch import assert_equal
    from test_doom_metal_batch import executor_type
    from test_doom_metal_batch_rgb import visual_brain

    replay = _replay_episode()
    data = _ThreeFrames()
    currents = np.array([[0., 1., 2.], [0., 0., 0.]], dtype=np.float64)
    lanes = [visual_brain(tmp_path), visual_brain(tmp_path)]
    initial = [tmp_path / f'initial-{index}.npz' for index in range(len(lanes))]
    for lane, checkpoint in zip(lanes, initial):
        lane.checkpoint(checkpoint)
    serial_active, serial_active_trace = _native_serial(
        tmp_path, _ThreeFrames(), currents[0], initial[0], frozen=False)
    serial_frozen, serial_frozen_trace = _native_serial(
        tmp_path, _ThreeFrames(), currents[1], initial[1], frozen=True)
    try:
        with executor_type()(lanes, window_ticks=18) as executor:
            for lane, checkpoint in zip(lanes, initial):
                lane.restore(checkpoint)
            first = replay(lanes, executor, data, READOUTS, currents,
                           learning=[True, True], frozen=[False, True],
                           directory=tmp_path / 'first', warmup_ms=0)
            for lane, serial, expected in zip(lanes, [serial_active, serial_frozen],
                                              [serial_active_trace, serial_frozen_trace]):
                assert_equal(lane, serial)
                actual = first['lanes'][lanes.index(lane)]['trace']
                assert [row['spikes_sha256'] for row in actual] == [digest(counts) for counts, _ in expected]
                assert [row['action'] for row in actual] == [action for _, action in expected]
                assert [row['neural_steps'] for row in actual] == [286, 285, 286]
            assert [row['teacher_current'] for row in first['lanes'][1]['trace']] == [0., 0., 0.]
            before = frozen_before[1]
            np.testing.assert_array_equal(lanes[1].memory_u, before[0])
            np.testing.assert_array_equal(lanes[1].memory_w, before[1])
            np.testing.assert_array_equal(lanes[1].weight, before[2])
            assert (tmp_path / 'first' / 'lane-0' / 'episode.json').is_file()
            assert (tmp_path / 'first' / 'lane-1' / 'episode.json').is_file()
            learned = [tmp_path / f'learned-{index}.npz' for index in range(len(lanes))]
            for lane, checkpoint in zip(lanes, learned):
                lane.checkpoint(checkpoint)
            frozen_before = [(lane.memory_u.copy(), lane.memory_w.copy(), lane.weight.copy()) for lane in lanes]
            evaluation = replay(lanes, executor, _ThreeFrames(), READOUTS, np.zeros((2, 3)),
                                learning=[False, False], frozen=[True, True],
                                directory=tmp_path / 'evaluation', warmup_ms=0)
            assert all(row['teacher_current'] == 0. for lane in evaluation['lanes'] for row in lane['trace'])
            for lane, before in zip(lanes, frozen_before):
                np.testing.assert_array_equal(lane.memory_u, before[0])
                np.testing.assert_array_equal(lane.memory_w, before[1])
                np.testing.assert_array_equal(lane.weight, before[2])
            for lane, checkpoint in zip(lanes, initial):
                lane.restore(checkpoint)
            second = replay(lanes, executor, _ThreeFrames(), READOUTS, currents,
                            learning=[True, True], frozen=[False, True],
                            directory=tmp_path / 'second', warmup_ms=0)
            assert _without_wall(first['lanes']) == _without_wall(second['lanes'])
    finally:
        serial_active.backend.close()
        serial_frozen.backend.close()


@pytest.mark.skipif(sys.platform != 'darwin', reason='Real Metal requires macOS')
def test_metal_iterator_failure_keeps_partial_lane_records_closes_iterator_and_checkpoints(tmp_path):
    """A decoder failure after frame zero retains real partial resident evidence."""
    from test_doom_metal_batch import executor_type
    from test_doom_metal_batch_rgb import visual_brain

    data = _ThreeFrames(fail_after=0)
    lanes = [visual_brain(tmp_path), visual_brain(tmp_path)]
    with executor_type()(lanes, window_ticks=18) as executor:
        with pytest.raises(RuntimeError, match='controlled frame iterator failure'):
            _replay_episode()(lanes, executor, data, READOUTS, np.zeros((2, 3)),
                              learning=[False, False], frozen=[True, True],
                              directory=tmp_path / 'failure', warmup_ms=0)
    assert data.closed
    summary = json.loads((tmp_path / 'failure' / 'summary.json').read_text())
    assert summary['failure']['type'] == 'RuntimeError'
    assert [len(lane['trace']) for lane in summary['lanes']] == [1, 1]
    for index, lane in enumerate(lanes):
        episode = tmp_path / 'failure' / f'lane-{index}' / 'episode.json'
        checkpoint = tmp_path / 'failure' / f'lane-{index}' / 'failure.npz'
        assert episode.is_file() and checkpoint.is_file()
        record = json.loads(episode.read_text())
        assert record['failure']['type'] == 'RuntimeError' and len(record['trace']) == 1
        with np.load(checkpoint, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            assert metadata['cursor'] == 286
            for name in ['weight', *lane.fields]:
                np.testing.assert_array_equal(saved[name], getattr(lane, name))
    assert (tmp_path / 'failure' / 'progress.json').is_file()
