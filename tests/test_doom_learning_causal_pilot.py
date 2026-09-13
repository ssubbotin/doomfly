"""Resident causal runner contracts, including real native RGB trajectories."""

import json
import hashlib
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


def test_round2_pressure_persists_rejected_boundary_and_detects_phase_gap(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    readings = iter([{'free_percent': 80, 'swap_used_mib': 0},
                     {'free_percent': 70, 'swap_used_mib': 0},
                     {'free_percent': 60, 'swap_used_mib': 1}])
    monkeypatch.setattr(pilot, '_memory_pressure', lambda: next(readings))
    pressure = pilot._PressureHistory(tmp_path)
    pressure.observe('before_training')
    pressure.observe('after_training')
    with pytest.raises(ValueError, match='Swap grew'):
        pressure.observe('before_evaluation')
    saved = json.loads((tmp_path / 'pressure.json').read_text())
    assert [r['name'] for r in saved['observations']] == [
        'before_training', 'after_training', 'before_evaluation']
    assert saved['observations'][-1]['value']['swap_used_mib'] == 1
    assert saved['observations'][-1]['accepted'] is False


def test_round2_low_pressure_is_recorded_before_rejection(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    monkeypatch.setattr(pilot, '_memory_pressure', lambda: {'free_percent': 24, 'swap_used_mib': 0})
    with pytest.raises(ValueError, match='free memory'):
        pilot._PressureHistory(tmp_path).observe('initial')
    assert json.loads((tmp_path / 'pressure.json').read_text())['observations'][0]['value']['free_percent'] == 24


def test_round2_cleanup_preserves_original_error_and_closes_registered_resources(tmp_path):
    from doom_learning_v6 import causal_pilot as pilot
    closed = []
    class Resource:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail
        def close(self):
            closed.append(self.name)
            if self.fail:
                raise OSError('close failed')
    original = RuntimeError('restore failed')
    with pytest.raises(RuntimeError) as caught:
        with pilot._Resources() as resources:
            resources.add(Resource('first'))
            resources.add(Resource('second', True))
            raise original
    assert caught.value is original
    assert closed == ['second', 'first']
    assert any('OSError' in note for note in original.__notes__)


def test_round2_expected_pins_rejects_matching_but_empty_role_maps(tmp_path):
    from doom_learning_v6.causal_pilot import _validate_expected_pins
    empty = {'source_commit': 'a' * 40, 'sources': {}, 'inputs': {},
             'references': {}, 'cpu': {}, 'native': {}, 'native_abi': 8}
    path = tmp_path / 'expected.json'
    path.write_text(json.dumps(empty))
    with pytest.raises(ValueError):
        _validate_expected_pins(path, empty)


def test_round2_reference_rejects_missing_controls_even_when_trace_complete(tmp_path):
    from doom_learning_v6.causal_pilot import _check_reference
    (tmp_path / 'plastic/train-0-0').mkdir(parents=True)
    (tmp_path / 'results.json').write_text(json.dumps({'complete': True}))
    (tmp_path / 'protocol.json').write_text(json.dumps({'eta': .001, 'epochs': 1, 'diagnostic_limit': None}))
    (tmp_path / 'plastic/train-0-0/episode.json').write_text(json.dumps(
        {'complete': True, 'identity': {'test': True}, 'trace': [{}, {}, {}]}))
    with pytest.raises(ValueError):
        _check_reference(tmp_path, SimpleNamespace(frame_count=3, identity={'test': True}))


def test_round2_residual_correlations_record_undefined_and_nonzero_relations():
    from doom_learning_v6.causal_pilot import _schedule_correlations
    value = _schedule_correlations([0, 1, 2], [0, 2, 1], [0, 1, 2], [1, 1, 1])
    assert value['aligned_shifted'] == pytest.approx(.5)
    assert value['shifted_target'] == pytest.approx(.5)
    assert value['shifted_error'] is None


def test_round2_checkpoint_bitwise_comparison_rejects_signed_zero(tmp_path):
    from doom_learning_v6.causal_pilot import _compare_checkpoint_arrays
    np.savez(tmp_path / 'a.npz', memory_w=np.array([0.], dtype=np.float64))
    np.savez(tmp_path / 'b.npz', memory_w=np.array([-0.], dtype=np.float64))
    with pytest.raises(ValueError, match='memory_w'):
        _compare_checkpoint_arrays(tmp_path / 'a.npz', tmp_path / 'b.npz')


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


@pytest.mark.parametrize('reference,replay,expected', [
    ({'spikes_sha256': 'a'}, {'spikes_sha256': 'b'}, False),
    ({'action': {'turn': 1.}}, {'action': {'turn': 2.}}, False),
    ({'memory_u_sha256': 'a'}, {}, False),
    ({'wall_seconds': 1., 'teacher_float32_current': 1., 'spikes_sha256': 'a'},
     {'wall_seconds': 9., 'teacher_float32_current': 2., 'spikes_sha256': 'a'}, True),
])
def test_original_replay_identity_rejects_every_scientific_difference(reference, replay, expected):
    """Only clock fields and the added float32 diagnostic are excluded."""
    from doom_learning_v6.causal_pilot import _same_without_wall
    assert _same_without_wall(reference, replay) is expected


def test_checkpoint_comparison_rejects_missing_dtype_shape_and_data(tmp_path):
    """A golden replay needs all checkpoint arrays, not producer metadata."""
    from doom_learning_v6.causal_pilot import _compare_checkpoint_arrays
    base = tmp_path / 'base.npz'; candidate = tmp_path / 'candidate.npz'
    np.savez(base, metadata='base', weight=np.array([1, 2], dtype=np.float32), memory_u=np.array([.1]))
    np.savez(candidate, metadata='candidate', weight=np.array([1, 3], dtype=np.float32), memory_u=np.array([.1]))
    with pytest.raises(ValueError, match='weight'):
        _compare_checkpoint_arrays(base, candidate)
    np.savez(candidate, metadata='candidate', weight=np.array([1, 2], dtype=np.float64), memory_u=np.array([.1]))
    with pytest.raises(ValueError, match='dtype'):
        _compare_checkpoint_arrays(base, candidate)
    np.savez(candidate, metadata='candidate', weight=np.array([[1, 2]], dtype=np.float32), memory_u=np.array([.1]))
    with pytest.raises(ValueError, match='shape'):
        _compare_checkpoint_arrays(base, candidate)
    np.savez(candidate, metadata='candidate', weight=np.array([1, 2], dtype=np.float32))
    with pytest.raises(ValueError, match='array keys'):
        _compare_checkpoint_arrays(base, candidate)


def test_directional_sensitivity_controls_are_distinct_from_uniform_controls():
    """Changing a direction to a uniform offset must fail the intervention contract."""
    from doom_learning_v6.causal_pilot import _sensitivity_memory_controls
    direction = np.array([-.5, 1., 0.])
    controls = _sensitivity_memory_controls(direction)
    np.testing.assert_allclose(controls['primary'][2], [-.025, .05, 0])
    np.testing.assert_allclose(controls['primary'][3], [.025, -.05, 0])
    np.testing.assert_allclose(controls['fallback'][:2], [[-.1, .2, 0], [.1, -.2, 0]])
    np.testing.assert_allclose(controls['fallback'][2:], [[-.05, -.05, -.05], [.05, .05, .05]])


def test_expected_pins_require_matching_source_and_required_roles(tmp_path):
    """A self-hash cannot substitute for the controller's trusted expectation."""
    from doom_learning_v6.causal_pilot import _validate_expected_pins, _physical_pins
    root, reference = _physical_tree(tmp_path)
    import doom_learning_v6.causal_pilot as pilot
    old_root = pilot._ROOT
    pilot._ROOT = root
    try:
        observed = _physical_pins(reference)
    finally:
        pilot._ROOT = old_root
    expected = tmp_path / 'expected.json'
    expected.write_text(json.dumps(observed))
    assert _validate_expected_pins(expected, observed)['source_commit'] == observed['source_commit']
    observed['native_abi'] = 7
    with pytest.raises(ValueError, match='expected pins'):
        _validate_expected_pins(expected, observed)


def _physical_tree(tmp_path):
    """Independent physical bytes and complete build manifests; no native load."""
    root, reference = tmp_path / 'repo', tmp_path / 'reference'
    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return hashlib.sha256(value.encode()).hexdigest()
    for directory in ['doom', 'doom_learning', 'doom_learning_v2', 'doom_learning_v4',
                      'doom_learning_v5', 'doom_learning_v6', 'tests']:
        put(directory + '/example.py', 'source')
    for directory, phase, model in [('doom_learning', '', 'gamma1-eligibility-ltd-v1'),
            ('doom_learning_v2', 'physiology-v2/', 'all-modulators-separated-v2'),
            ('doom_learning_v4', 'physiology-v4/', 'kc-adaptive-lif-v4'),
            ('doom_learning_v5', 'physiology-v5/', 'centered-antihebbian-v5'),
            ('doom_learning_v6', 'physiology-v6/', 'adaptive-centered-v6')]:
        source = put(directory + '/kernel.cpp', 'kernel for ' + directory)
        name = 'outputs/doom-learning/' + phase + 'libmemory.dylib'
        binary = put(name, 'physical binary for ' + directory)
        put(name + '.json', json.dumps({'model': model, 'source_sha256': source,
             'binary_sha256': binary, 'flags': ['-O3', '-std=c++17', '-shared', '-fPIC']}))
    sources = {n: put('doom_learning_v6/metal/' + n, 'native ' + n)
               for n in ('api.h', 'backend.mm', 'kernels.metal', 'decay_tables.h')}
    builder = put('doom_learning_v6/metal/build.py', 'builder')
    binaries = {role: put('outputs/doom-learning/metal/' + name, 'binary ' + name)
                for role, name in [('air', 'kernels.air'), ('metallib', 'kernels.metallib'),
                                   ('library', 'libmemory-metal.dylib')]}
    configuration = {'abi_version': 8, 'builder_sha256': builder,
        'numerical_parent': 'b24249ecfee12aa84b55ed803ffe47bd09c487ad',
        'numerical_order': 'epoch-5-zero-seeded-ascending-incoming',
        'metal_compile_flags': ['-std=macos-metal2.4', '-fno-fast-math', '-ffp-contract=off'],
        'library_compile_flags': ['-O3', '-std=c++17', '-dynamiclib', '-fobjc-arc', '-arch', 'arm64',
                                  '-mmacosx-version-min=13.0']}
    put('outputs/doom-learning/metal/build.json', json.dumps({'schema': 2, 'abi_version': 8,
        'architecture': 'arm64', 'metal_language': 'macos-metal2.4', 'sources': sources,
        'binaries': binaries, 'build_configuration': configuration}))
    for name in ['outputs/doom/malecns_v1/graph.npz', 'connectome_data/malecns_v1/annotations.feather',
                 'connectome_data/malecns_v1/normalized/neurons.feather']:
        put(name, 'physical input')
    for name in ['results.json', 'protocol.json', 'provenance.json', 'initial.npz', 'plastic/learned.npz',
                 'frozen/learned.npz', 'shifted/learned.npz',
                 'plastic/train-0-0/episode.json', 'plastic/eval-0/episode.json',
                 'plastic/retention-0/episode.json', 'plastic/erased-0/episode.json',
                 'frozen/train-0-0/episode.json', 'frozen/eval-0/episode.json',
                 'shifted/train-0-0/episode.json', 'shifted/eval-0/episode.json']:
        path = reference / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('reference bytes')
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                    'commit', '--allow-empty', '-qm', 'test'], check=True)
    return root, reference


def test_round2_trusted_reference_coverage_includes_every_consumed_control_and_checkpoint(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    root, reference = _physical_tree(tmp_path)
    monkeypatch.setattr(pilot, '_ROOT', root)
    observed = pilot._physical_pins(reference)
    roles = {'results.json', 'protocol.json', 'provenance.json', 'initial.npz',
             'plastic/learned.npz', 'frozen/learned.npz', 'shifted/learned.npz',
             'plastic/train-0-0/episode.json', 'plastic/eval-0/episode.json',
             'plastic/retention-0/episode.json', 'plastic/erased-0/episode.json',
             'frozen/train-0-0/episode.json', 'frozen/eval-0/episode.json',
             'shifted/train-0-0/episode.json', 'shifted/eval-0/episode.json'}
    assert set(observed['references']) == roles
    expected = tmp_path / 'expected.json'
    expected.write_text(json.dumps(observed))
    pilot._validate_expected_pins(expected, observed)
    observed['references'].pop('shifted/learned.npz')
    expected.write_text(json.dumps(observed))
    with pytest.raises(ValueError):
        pilot._validate_expected_pins(expected, observed)


@pytest.mark.parametrize('mutation', ['cpu_source', 'cpu_binary', 'cpu_flags', 'metal_source',
                                    'metal_binary', 'metal_flags', 'metal_builder', 'missing_file'])
def test_round2_physical_manifest_relationships_reject_tampering(tmp_path, monkeypatch, mutation):
    from doom_learning_v6 import causal_pilot as pilot
    root, reference = _physical_tree(tmp_path)
    monkeypatch.setattr(pilot, '_ROOT', root)
    valid = pilot._physical_pins(reference)
    assert len(valid['cpu']) == 5 and len(valid['native']) == 4
    if mutation == 'cpu_source':
        (root / 'doom_learning_v6/kernel.cpp').write_text('changed source')
    elif mutation == 'cpu_binary':
        (root / 'outputs/doom-learning/physiology-v6/libmemory.dylib').write_text('changed binary')
    elif mutation == 'cpu_flags':
        path = root / 'outputs/doom-learning/physiology-v6/libmemory.dylib.json'
        record = json.loads(path.read_text()); record['flags'].append('-ffast-math'); path.write_text(json.dumps(record))
    elif mutation == 'metal_source':
        (root / 'doom_learning_v6/metal/kernels.metal').write_text('changed source')
    elif mutation == 'metal_binary':
        (root / 'outputs/doom-learning/metal/kernels.air').write_text('changed binary')
    elif mutation == 'metal_builder':
        (root / 'doom_learning_v6/metal/build.py').write_text('changed builder')
    elif mutation == 'metal_flags':
        path = root / 'outputs/doom-learning/metal/build.json'
        record = json.loads(path.read_text()); record['build_configuration']['metal_compile_flags'] = ['-ffast-math']
        path.write_text(json.dumps(record))
    else:
        (root / 'outputs/doom-learning/metal/kernels.metallib').unlink()
    with pytest.raises(ValueError):
        pilot._physical_pins(reference)


def test_round2_registered_brain_cleanup_covers_factory_and_restore_failures(tmp_path):
    from doom_learning_v6 import causal_pilot as pilot
    closed = []
    original = RuntimeError('restore failed')
    class Brain:
        def __init__(self):
            self.backend = SimpleNamespace(close=lambda: closed.append('brain'))
        def restore(self, path):
            raise original
    with pytest.raises(RuntimeError) as caught:
        with pilot._Resources() as resources:
            pilot._restored_brains(resources, Brain, tmp_path / 'initial.npz', 4)
    assert caught.value is original and closed == ['brain']


def test_round2_failure_writer_keeps_error_and_persists_secondary_notes(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    from doom_learning import common
    def fail(*args, **kwargs):
        raise OSError('writer failed')
    monkeypatch.setattr(common, 'save_json', fail)
    original = RuntimeError('simulation failed')
    pilot._save_failure(tmp_path, original, pilot._PressureHistory(tmp_path))
    record = json.loads((tmp_path / 'failure-fallback.json').read_text())
    assert record['failure']['type'] == 'RuntimeError'
    assert any('OSError' in note for note in record['failure']['secondary_notes'])


def test_round2_snapshot_captures_configuration_values_before_mutation(monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    configuration = {'rule': {'minimum_fraction': .1}}
    brain = SimpleNamespace(configuration_signature=lambda: configuration)
    monkeypatch.setattr(pilot, '_model_gate', lambda brains, protocol: {'ids_sha256': 'original'})
    before = pilot._snapshot([brain], {})
    configuration['rule']['minimum_fraction'] = .2
    after = pilot._snapshot([brain], {})
    assert before['configuration'][0]['rule']['minimum_fraction'] == .1
    assert before != after


def test_round2_phase_failure_keeps_post_pressure_and_primary_error(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    readings = iter([{'free_percent': 80, 'swap_used_mib': 0}, {'free_percent': 20, 'swap_used_mib': 0}])
    monkeypatch.setattr(pilot, '_memory_pressure', lambda: next(readings))
    original = RuntimeError('iterator failed')
    pressure = pilot._PressureHistory(tmp_path)
    with pytest.raises(RuntimeError) as caught:
        with pilot._phase_boundary(pressure, 'training'):
            raise original
    assert caught.value is original
    saved = json.loads((tmp_path / 'pressure.json').read_text())['observations']
    assert [r['name'] for r in saved] == ['before_training', 'after_training']
    assert saved[1]['accepted'] is False
    assert any('ValueError' in note for note in original.__notes__)


def test_round2_phase_failure_attempts_all_lane_checkpoints_when_json_writer_fails(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    from doom_learning import common
    checkpointed = []
    brains = [SimpleNamespace(checkpoint=lambda path: checkpointed.append(path)) for _ in range(2)]
    def fail(*args, **kwargs):
        raise OSError('json writer failed')
    monkeypatch.setattr(common, 'save_json', fail)
    original = RuntimeError('frame iterator failed')
    records = [{'complete': False, 'trace': []} for _ in brains]
    pilot._write_phase(tmp_path, brains, records, failure=original)
    assert checkpointed == [tmp_path / 'lane-0/failure.npz', tmp_path / 'lane-1/failure.npz']
    assert any('OSError' in note for note in original.__notes__)


@pytest.mark.parametrize('fail_at', [None, 'construct', 'restore', 'model', 'pressure', 'replay', 'close'])
def test_round2_causal_orchestration_cleans_all_brains_and_snapshots_while_open(tmp_path, monkeypatch, fail_at):
    """Keep orchestration real; replace only full graph/native execution boundaries."""
    from doom_learning_v6 import causal_pilot as pilot, calibration
    from doom_learning_v6.metal import batch
    from doom_learning.common import digest
    created, snapshots = [], []
    original = RuntimeError('controlled ' + str(fail_at))
    class Brain:
        def __init__(self):
            self.closed = False
            self.weight = np.array([1.], dtype=np.float32)
            self.backend = SimpleNamespace(close=self.close)
        def close(self):
            self.closed = True
        def restore(self, path):
            if fail_at == 'restore':
                raise original
        def configuration_signature(self):
            return {'initial_weight': digest(self.weight)}
        def checkpoint(self, path):
            pass
    def factory(*args, **kwargs):
        if fail_at == 'construct' and created:
            raise original
        brain = Brain(); created.append(brain); return brain
    class Executor:
        def __init__(self, brains, *, window_ticks):
            assert window_ticks == 18
            self.brains = brains
        def close(self):
            for brain in self.brains:
                brain.closed = True
            if fail_at == 'close':
                raise original
    def model_gate(brains, protocol):
        assert all(not b.closed for b in brains)
        snapshots.append(len(brains))
        if fail_at == 'model':
            raise original
        return {'ids_sha256': 'original', 'plastic_slots_sha256': 'positive-original'}
    def memory_pressure():
        if fail_at == 'pressure':
            raise original
        return {'free_percent': 80, 'swap_used_mib': 0}
    trace = [{'target_turn': float(i), 'teacher_error': 0.} for i in range(4)]
    def replay(*args, **kwargs):
        if fail_at == 'replay':
            raise original
        return {'lanes': [{'trace': trace} for _ in range(4)]}
    monkeypatch.setattr(calibration, 'calibrated_brain', factory)
    monkeypatch.setattr(batch, 'MetalBatchExecutor', Executor)
    monkeypatch.setattr(pilot, '_check_reference', lambda *a, **k: (tmp_path, {'trace': trace}))
    monkeypatch.setattr(pilot, '_reference_schedule', lambda *a: (np.array([0., 1., 2., 3.]), np.array([286, 285, 286, 286])))
    monkeypatch.setattr(pilot, '_model_gate', model_gate)
    monkeypatch.setattr(pilot, '_memory_pressure', memory_pressure)
    monkeypatch.setattr(pilot, 'replay_episode', replay)
    monkeypatch.setattr(pilot, '_compare_checkpoint_arrays', lambda *a: {'passed': True, 'arrays': list(range(24))})
    (tmp_path / 'protocol.json').write_text('{}')
    data = SimpleNamespace(frame_count=4)
    if fail_at is None:
        result = pilot._run_causal(SimpleNamespace(), data, data, tmp_path, READOUTS, tmp_path)
        assert snapshots == [4, 4, 4]
        assert result['invariants']['before'] == result['invariants']['after']
        assert result['configuration']['before'] == result['configuration']['after']
    else:
        with pytest.raises(RuntimeError) as caught:
            pilot._run_causal(SimpleNamespace(), data, data, tmp_path, READOUTS, tmp_path)
        assert caught.value is original
    assert created and all(b.closed for b in created)


def test_round2_cli_preserves_all_manifest_readouts_for_original_trace(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    from doom_learning import common
    rows = [{'index': i + 10, 'type': 'DNa02', 'side': 'L'} for i in range(10)] + READOUTS
    (tmp_path / 'manifest.json').write_text(json.dumps({'readouts': rows}))
    monkeypatch.setattr(common, 'GRAPH', tmp_path / 'graph.npz')
    assert pilot._readouts() == rows


def test_round2_replay_records_separate_actual_wall_intervals_using_real_cpu_rgb(tmp_path, monkeypatch):
    from doom_learning_v6 import causal_pilot as pilot
    from test_doom_metal_batch_rgb import visual_brain
    brain = visual_brain(tmp_path)
    class SerialCpuExecutor:
        brains = (brain,)
        def metadata(self):
            return {'shared_resident_bytes': 0, 'mutable_resident_bytes': 0}
        def rgb_step(self, images, duration, *, learning, stimulations=None):
            flag = learning[0] if isinstance(learning, list) else learning
            counts, elapsed = brain.rgb_step(images[0], duration, learning=flag,
                stimulation=None if stimulations is None else stimulations[0])
            return [counts], elapsed
    ticks = iter(range(100))
    monkeypatch.setattr(pilot, 'time', SimpleNamespace(perf_counter=lambda: float(next(ticks))))
    try:
        result = pilot.replay_episode([brain], SerialCpuExecutor(), _ThreeFrames(), READOUTS,
            np.zeros((1, 3)), learning=[False], frozen=[True], directory=tmp_path / 'timed', warmup_ms=.1)
    finally:
        brain.backend.close()
    intervals = result['lanes'][0]['phase_timing']
    assert intervals == {'setup_reset_wall_seconds': 1., 'dark_warmup_wall_seconds': 1., 'rgb_loop_wall_seconds': 1.}
    assert result['lanes'][0]['warmup']['brain_steps'] == [1]
    assert result['lanes'][0]['brain_steps'] == 857
    assert (tmp_path / 'timed/lane-0/final.npz').exists()


def test_round2_sensitivity_null_uses_actual_controls_despite_readout_changes():
    from doom_learning_v6 import causal_pilot as pilot
    baseline = {'trace': [{'action': {'turn': 1., 'forward': True, 'attack': False, 'readouts': {'DNp20': 1}}}]}
    changed = {'trace': [{'action': {'turn': 1., 'forward': True, 'attack': False, 'readouts': {'DNp20': 2}}}]}
    assert pilot._decoded_controls(baseline) == pilot._decoded_controls(changed)
    changed['trace'][0]['action']['attack'] = True
    assert pilot._decoded_controls(baseline) != pilot._decoded_controls(changed)


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
    frozen_before = [(lane.memory_u.copy(), lane.memory_w.copy(), lane.weight.copy()) for lane in lanes]
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
            evaluation_before = [(lane.memory_u.copy(), lane.memory_w.copy(), lane.weight.copy()) for lane in lanes]
            evaluation = replay(lanes, executor, _ThreeFrames(), READOUTS, np.zeros((2, 3)),
                                learning=[False, False], frozen=[True, True],
                                directory=tmp_path / 'evaluation', warmup_ms=0)
            assert all(row['teacher_current'] == 0. for lane in evaluation['lanes'] for row in lane['trace'])
            for lane, before in zip(lanes, evaluation_before):
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
