"""Portable evidence contracts for the real bounded driver scheduler."""

import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doom_learning_v6.same_slot_optimizer import load_protocol


PROTOCOL_PATH = Path('docs/experiments/2026-09-13-same-slot-optimizer-protocol.json')
PROTOCOL_SHA = 'fae8309b51cbc24b63b23fcfaf9c4a5e5f5d141a0ce060bee3d2913821fe9c03'
READOUTS = [dict(index=i, type=t, side=s) for i, (t, s) in enumerate([
    ('DNp20', 'L'), ('DNp20', 'R'), ('DNpe017', 'L'), ('DNpe017', 'R'),
    *[('other', 'L')] * 10,
])]


def runner():
    assert importlib.util.find_spec('doom_learning_v6.same_slot_runner') is not None, 'bounded driver is missing'
    return importlib.import_module('doom_learning_v6.same_slot_runner')


class Brain:
    def __init__(self):
        self.circuit = {'edges': np.arange(4184, dtype=np.int64)}
        self.baseline_plastic = np.ones(4184, dtype=np.float32)
        self.weight = np.ones(4186, dtype=np.float32)
        self.memory_u = np.zeros(4184, dtype=np.float64)
        self.memory_w = np.zeros(4184, dtype=np.float64)
        self.ids = np.arange(8, dtype=np.int64)
        self.ptr = np.arange(9, dtype=np.int64)
        self.post = np.arange(8, dtype=np.int32)
        self.backend = SimpleNamespace(_host_weight_epoch=0, _device_weight_epoch=0,
                                       materialize=lambda reason: None)
        self.extra = {f'field{i}': np.array([i], dtype=np.int64) for i in range(21)}

    def configuration_signature(self):
        return {'eta': .001, 'decoder': 'fixed'}

    def checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self.weight, memory_u=self.memory_u, memory_w=self.memory_w,
                 **self.extra)


class Executor:
    def __init__(self, out):
        self.brains = [Brain() for _ in range(4)]
        self.out = out
        self.window_ticks = 18
        self.mutable_bytes = 64

    def metadata(self):
        return dict(shared_resident_bytes=128, mutable_resident_bytes=self.mutable_bytes,
                    window_ticks=self.window_ticks, lane_count=4)

    def install_efficacies(self, lane, values, expected_edges):
        budget = json.loads((self.out / 'attempt-ledger.json').read_text())
        assert budget['used'] == 4 * len(budget['waves'])
        assert budget['waves'][-1]['state'] == 'charged'
        b = self.brains[lane]
        assert np.array_equal(expected_edges, b.circuit['edges'])
        b.memory_u[:] = values - 1
        b.memory_w[:] = values - 1
        b.weight[b.circuit['edges']] = (b.baseline_plastic.astype(np.float64) * values).astype(np.float32)
        b.backend._host_weight_epoch += 1
        b.backend._device_weight_epoch += 1


@pytest.fixture
def study(tmp_path, monkeypatch):
    module = runner()
    from doom_learning_v6 import causal_pilot
    monkeypatch.setattr(causal_pilot, '_memory_pressure', lambda: dict(free_percent=80, swap_used_mib=0))
    monkeypatch.setattr(module, '_model_gate', lambda brains, protocol=None: {'retained': True})
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20 * 1024 ** 3))
    protocol = load_protocol(PROTOCOL_PATH, PROTOCOL_SHA)
    episodes = [SimpleNamespace(frame_count=row['frames'], identity=dict(
        dataset=protocol['dataset'], dataset_revision=protocol['dataset_revision'],
        scenario=protocol['scenario'], episode_index=row['index'], video_sha256=row['video_sha256'],
        control_sha256=row['control_sha256']),
        turn_targets=lambda count=row['frames']: np.zeros(count)) for row in protocol['episodes']]
    out = tmp_path / 'fit'
    return module, Executor(out), episodes[:2], episodes[2:], protocol, out


def replay_double(loss=None, *, fail_wave=None, error=None, corrupt=None):
    def replay(brains, executor, data, readouts, currents, *, learning, frozen, directory, warmup_ms):
        directory = Path(directory)
        ledger = json.loads((executor.out / 'attempt-ledger.json').read_text())
        wave = ledger['waves'][-1]
        assert ledger['used'] == 4 * len(ledger['waves'])
        assert learning == [False] * 4 and frozen == [True] * 4 and warmup_ms == 2000
        assert np.array_equal(currents, np.zeros((4, data.frame_count)))
        if data.identity['episode_index'] >= 4:
            assert (executor.out / 'final-candidates.json').is_file()
        directory.mkdir(parents=True)
        if fail_wave == len(ledger['waves']):
            (directory / 'partial.json').write_text('{"frames":1}')
            raise error
        records = []
        for lane, brain in enumerate(brains):
            for _ in range(2):
                brain.backend._host_weight_epoch += 1
                brain.backend._device_weight_epoch += 1
            value = loss(wave, lane, data) if loss else (3. if data.identity['episode_index'] == 2 else 5.)
            trace = [dict(index=i, action={'turn': value}, target_turn=0., teacher_current=0.)
                     for i in range(data.frame_count)]
            row = dict(identity=data.identity, complete=True, frames=data.frame_count,
                       source_frame_count=data.frame_count, weights_frozen=True, trace=trace,
                       phase_timing=dict(setup_reset_wall_seconds=.1, dark_warmup_wall_seconds=.2,
                                         rgb_loop_wall_seconds=.3))
            if corrupt:
                corrupt(wave, lane, brain, row)
            brain.checkpoint(directory / f'lane-{lane}' / 'final.npz')
            records.append(row)
        return dict(complete=True, lanes=records)
    return replay


def test_driver_module_is_available():
    runner()


def test_real_schedule_zero_gradient_spends_all_waves_and_freezes_controls(study):
    module, ex, train, held, protocol, out = study
    result = module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double())
    ledger = json.loads((out / 'attempt-ledger.json').read_text())
    assert result['complete'] is True
    assert result['learning_demonstrated'] is False and result['announcement_ready'] is False
    assert ledger['used'] == 120 and len(ledger['waves']) == 30
    assert len([w for w in ledger['waves'] if w['kind'] == 'proposal']) == 8
    assert [w['used'] for w in ledger['waves'][:8]] == [4, 8, 12, 16, 20, 24, 28, 32]
    for generation in range(4):
        saved = json.loads((out / 'generations' / f'g-{generation}' / 'decisions.json').read_text())
        assert all(row['zero_update'] and not row['accepted'] for row in saved)
        assert all(row['incumbent_loss'] == 4. and row['proposal_loss'] == 4. for row in saved)
        for row in saved:
            folder = out / 'generations' / f'g-{generation}' / row['run']
            direction = np.load(folder / 'direction.npy')
            plus = np.load(folder / 'plus-theta.npy')
            minus = np.load(folder / 'minus-theta.npy')
            assert np.array_equal(plus, protocol['probe_scales'][generation] * direction)
            assert np.array_equal(minus, -protocol['probe_scales'][generation] * direction)
            assert np.array_equal(np.load(folder / 'plus-fractions.npy'), 1 + .1 * np.tanh(plus))
            method, seed = row['run'].rsplit('-', 1)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0 if method == 'spsa' else 1])))
            for _ in range(generation + 1):
                expected = 2. * rng.integers(0, 2, size=4184) - 1. if method == 'spsa' else rng.standard_normal(4184)
            assert np.array_equal(direction, expected)
        group = [w for w in ledger['waves'] if w['generation'] == generation]
        for name in result['runs']:
            coverage = [(w['kind'], w['episode_index'], w['signs'][lane] if w['signs'] else None)
                        for w in group for lane, role in enumerate(w['roles']) if role == name]
            assert sorted(coverage, key=str) == sorted([
                ('probe', 2, '+'), ('probe', 2, '-'), ('probe', 3, '+'), ('probe', 3, '-'),
                ('proposal', 2, None), ('proposal', 3, None)], key=str)
    for name in result['runs']:
        assert np.array_equal(np.load(out / 'incumbents' / name / 'theta.npy'), np.zeros(4184))
    manifest = json.loads((out / 'final-candidates.json').read_text())
    assert len(manifest['candidates']) == 7
    randoms = [row for row in manifest['candidates'] if row['role'].startswith('random-')]
    assert [row['seed'] for row in randoms] == [200000301, 200000302]
    for row in randoms:
        values = np.load(out / row['fractions'])
        expected = np.random.Generator(np.random.PCG64(row['seed'])).uniform(.9, 1.1, size=4184)
        assert np.array_equal(values, expected)
        assert .9 <= values.min() < values.max() <= 1.1
        assert 'training_loss' not in row
    held_waves = [w for w in ledger['waves'] if w['kind'] == 'held']
    assert [w['episode_index'] for w in held_waves] == [4, 5, 4, 5]
    assert sum(w['roles'].count('baseline-filler') for w in held_waves) == 2
    assert all(w['state'] == 'complete' for w in ledger['waves'])
    assert all(not role.startswith('random-') for w in ledger['waves'] if w['kind'] != 'held' for role in w['roles'])


def test_complete_bank_strict_selection_saves_only_improving_proposal(study):
    module, ex, train, held, protocol, out = study
    def losses(wave, lane, data):
        index = data.identity['episode_index']
        if wave['kind'] == 'probe':
            return 1. if wave['signs'][lane] == '+' else 3.
        if wave['kind'] == 'proposal':
            if wave['generation'] == 0:
                return 2. if index == 2 else 7.  # Mean 4.5 loses against baseline 4.
            if wave['generation'] == 1:
                return 1. if index == 2 else 5.  # Mean 3 strictly wins.
            return 3.  # Ties with cached center preserve it.
        return 3. if index == 2 else 5.
    module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double(losses))
    assert all(not r['accepted'] and r['proposal_loss'] == 4.5 for r in
               json.loads((out / 'generations/g-0/decisions.json').read_text()))
    assert all(r['accepted'] and r['proposal_loss'] == 3. for r in
               json.loads((out / 'generations/g-1/decisions.json').read_text()))
    for name in ['spsa-200000201', 'spsa-200000202', 'gaussian_es-200000201', 'gaussian_es-200000202']:
        saved = np.load(out / 'incumbents' / name / 'theta.npy')
        proposed = np.load(out / 'generations/g-1' / name / 'proposal-theta.npy')
        assert np.array_equal(saved, proposed)
        assert np.max(np.abs(saved)) == pytest.approx(.15)
        direction = np.load(out / 'generations/g-1' / name / 'direction.npy')
        # Algebraic cancellation changes floating-point operation ordering.
        np.testing.assert_allclose(saved, .15 * direction / np.max(np.abs(direction)), rtol=0, atol=1e-16)
        assert np.array_equal(np.load(out / 'incumbents' / name / 'fractions.npy'), 1 + .1 * np.tanh(saved))


@pytest.mark.parametrize('error', [RuntimeError('primary replay failure'), KeyboardInterrupt('stop')])
def test_interrupted_wave_keeps_charge_partial_records_and_last_centers(study, error):
    module, ex, train, held, protocol, out = study
    with pytest.raises(type(error)) as caught:
        module.fit(ex, train, held, READOUTS, protocol, out,
                   replay=replay_double(fail_wave=8, error=error))
    assert caught.value is error
    ledger = json.loads((out / 'attempt-ledger.json').read_text())
    assert ledger['used'] == 32 and ledger['waves'][-1]['state'] == 'charged'
    assert list(out.glob('waves/*/partial.json'))
    for path in (out / 'incumbents').glob('*/theta.npy'):
        assert np.array_equal(np.load(path), np.zeros(4184))
    assert json.loads((out / 'failure.json').read_text())['complete'] is False


@pytest.mark.parametrize('change', ['count', 'identity', 'protocol', 'readouts'])
def test_bad_portable_contract_rejects_before_any_charge_or_install(study, change):
    module, ex, train, held, protocol, out = study
    if change == 'count':
        train[0].frame_count -= 1
    elif change == 'identity':
        held[0].identity['video_sha256'] = train[0].identity['video_sha256']
    elif change == 'protocol':
        protocol['window_ticks'] = 0
    readouts = READOUTS[:-1] if change == 'readouts' else READOUTS
    with pytest.raises(ValueError):
        module.fit(ex, train, held, readouts, protocol, out, replay=replay_double())
    assert not (out / 'attempt-ledger.json').exists()
    assert all(b.backend._host_weight_epoch == 0 for b in ex.brains)


@pytest.mark.parametrize('path,value', [
    (('learning',), 0), (('frozen',), 1), (('generations',), 4.0),
    (('schema',), True), (('pairs_per_update',), True),
    (('episodes', 0, 'frames'), 948.0),
    (('budget', 'baseline'), 8.0), (('historical_indices', 0), False),
    (('historical_indices', 1), True), (('fraction_bounds', 1), 2.0),
    (('episodes',), ()), (('budget',), []),
    (('budget', 'key-type'), None), (('episodes', 0, 'key-type'), None),
])
def test_direct_fit_rejects_recursive_protocol_type_mutations_without_work(study, monkeypatch, path, value):
    module, ex, train, held, protocol, out = study
    target = protocol
    for key in path[:-1]:
        target = target[key]
    if path[-1] == 'key-type':
        class NonBuiltinKey(str):
            pass
        key = next(iter(target))
        item = target.pop(key)
        target[NonBuiltinKey(key)] = item
    else:
        target[path[-1]] = value
    calls = []
    monkeypatch.setattr(ex, 'install_efficacies', lambda *args: calls.append('install'))
    def forbidden_replay(*args, **kwargs):
        calls.append('replay')
        raise RuntimeError('malformed protocol reached replay')
    with pytest.raises(ValueError, match='Exact committed fitting protocol'):
        module.fit(ex, train, held, READOUTS, protocol, out, replay=forbidden_replay)
    assert calls == []
    assert not (out / 'attempt-ledger.json').exists()


def _same_slot_visual_brain(tmp_path):
    """Real visual CPU/checkpoint implementation with a bounded synthetic graph."""
    from doom_learning_v6.brain import MemoryBrain
    from test_doom_metal_batch_rgb import visual_brain
    b = visual_brain(tmp_path)
    path = tmp_path / 'same-slot-synthetic.npz'
    np.savez(path, ptr=np.array([0, *([4186] * 14)], dtype=np.int64),
             post=np.ones(4186, dtype=np.int32), weight=np.ones(4186, dtype=np.float32),
             ids=np.arange(14, dtype=np.int64), retina=np.array([3], dtype=np.int32),
             uv=np.array([[.75, .25]], dtype=np.float32), lamina=np.empty(0, dtype=np.int32),
             sugar=np.empty(0, dtype=np.int32), superclass=np.array(['test'] * 14))
    circuit = dict(b.circuit, edges=np.arange(4184, dtype=np.int64),
                   pre=np.zeros(4184, dtype=np.int32), gain=np.ones((4184, 1), dtype=np.float32),
                   kc_mask=np.array([1, *([0] * 13)], dtype=np.uint8),
                   dan_index=np.array([-1, -1, 0, *([-1] * 11)], dtype=np.int8))
    MemoryBrain.__init__(b, path, circuit=circuit, modulation_mask=np.array([0, 0, 1, *([0] * 11)]))
    b.fields.append('r8_light')
    b.initial['r8_light'] = b.r8_light.copy()
    b.backend._host_weight_epoch = b.backend._device_weight_epoch = 0
    return b


@pytest.mark.parametrize('interrupted', [False, True])
@pytest.mark.parametrize('secondary', [None, 'json', 'read', 'json-read'])
def test_fit_consumes_real_replay_unavailable_lane_without_retry(study, tmp_path, monkeypatch, interrupted, secondary):
    from doom_learning import common
    from doom_learning_v6 import causal_pilot
    from doom_learning_v6.backend import BackendError
    from test_doom_learning_causal_pilot import _SerialCpuExecutor
    module, ex, train, held, protocol, out = study
    ex.brains = [_same_slot_visual_brain(tmp_path) for _ in range(4)]
    serial = _SerialCpuExecutor(ex.brains)
    monkeypatch.setattr(ex, 'rgb_step', serial.rgb_step, raising=False)
    download_error = BackendError('non-poisoning terminal download failure')
    primary = KeyboardInterrupt('original frame interruption') if interrupted else download_error
    train[0].height, train[0].width = 3, 4
    def frames():
        for index in range(train[0].frame_count):
            if interrupted and index == 1:
                raise primary
            yield SimpleNamespace(index=index, timestamp=index / 35,
                                  rgb=np.zeros((3, 4, 3), dtype=np.uint8), action=np.zeros(9))
    train[0].iter_frames = frames
    downloads, checkpoints = [], []
    def materialize(reason):
        if reason == 'checkpoint':
            downloads.append(reason)
            # Deliberately leave the owner usable. A redundant sync would succeed.
            if len(downloads) == 1:
                raise download_error
    monkeypatch.setattr(ex.brains[0].backend, 'materialize', materialize)
    for lane, brain in enumerate(ex.brains):
        checkpoint = brain.checkpoint
        def save(path, lane=lane, checkpoint=checkpoint):
            checkpoints.append(lane)
            return checkpoint(path)
        monkeypatch.setattr(brain, 'checkpoint', save)
    phase = out / 'waves/w-00'
    save_json = common.save_json
    def writer(path, value):
        if secondary in ('json', 'json-read') and Path(path) == phase / 'summary.json':
            raise OSError('secondary replay summary writer')
        return save_json(path, value)
    monkeypatch.setattr(common, 'save_json', writer)
    load = np.load
    def reader(path, *args, **kwargs):
        if secondary in ('read', 'json-read') and Path(path) == phase / 'lane-1/failure.npz':
            raise OSError('secondary checkpoint evidence read')
        return load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', reader)
    try:
        with pytest.raises(type(primary)) as caught:
            module.fit(ex, train, held, READOUTS, protocol, out, replay=causal_pilot.replay_episode)
        assert caught.value is primary
        assert checkpoints == [1, 2, 3]
        assert downloads == ['checkpoint']
        assert not (phase / 'lane-0/failure.npz').exists()
        for lane in range(4):
            record = json.loads((phase / f'lane-{lane}/episode.json').read_text())
            assert record['frames'] == len(record['trace']) == (1 if interrupted else 948)
            assert record['trace'][0]['index'] == 0
        failed = json.loads((phase / 'lane-0/episode.json').read_text())
        assert failed['terminal_state'] is None
        assert failed['terminal_state_available'] is False
        assert failed['checkpoint_available'] is False and failed['checkpoint_attempted'] is False
        availability = json.loads((out / 'failure-checkpoints.json').read_text())['lanes']
        assert availability[0]['all24_available'] is False
        assert 'BackendError' in availability[0]['reason']
        assert [row['all24_available'] for row in availability[1:]] == [
            secondary not in ('read', 'json-read'), True, True]
        ledger = json.loads((out / 'attempt-ledger.json').read_text())
        assert ledger['used'] == 4 and ledger['waves'][0]['state'] == 'charged'
        assert json.loads((out / 'interrupted-progress.json').read_text())['attempts'] == 4
        if secondary is not None:
            assert any('OSError' in note for note in primary.__notes__)
    finally:
        for brain in ex.brains:
            brain.backend.close()


@pytest.mark.parametrize('change', ['trace', 'checkpoint', 'epoch', 'weight', 'teacher', 'target'])
def test_runtime_cross_lane_and_frozen_contracts_stop_before_cached_scores(study, change):
    module, ex, train, held, protocol, out = study
    def corrupt(wave, lane, brain, row):
        if lane != 1 and change != 'target':
            return
        if change == 'trace':
            row['trace'][0]['action']['turn'] = 9.
        elif change == 'checkpoint':
            brain.extra['field7'][0] = 8
        elif change == 'epoch':
            brain.backend._host_weight_epoch += 1
            brain.backend._device_weight_epoch += 1
        elif change == 'weight':
            brain.weight[-1] = 2.
        elif change == 'teacher':
            row['trace'][0]['teacher_current'] = 1.
        else:
            row['trace'][0]['target_turn'] = 1.
    with pytest.raises(ValueError):
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double(corrupt=corrupt))
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4
    assert not (out / 'baseline-scores.json').exists()


def test_failure_writer_cannot_replace_keyboard_interrupt(study, monkeypatch):
    module, ex, train, held, protocol, out = study
    from doom_learning import common
    original_writer = common.save_json
    def writer(path, data):
        if Path(path).name == 'failure.json':
            raise OSError('secondary writer')
        original_writer(path, data)
    monkeypatch.setattr(common, 'save_json', writer)
    error = KeyboardInterrupt('primary')
    with pytest.raises(KeyboardInterrupt) as caught:
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double(fail_wave=8, error=error))
    assert caught.value is error
    assert any('secondary writer' in note for note in error.__notes__)
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 32
    assert json.loads((out / 'failure-fallback.json').read_text())['complete'] is False


def test_swap_growth_across_existing_allocation_history_rejects_first_wave(study, monkeypatch):
    module, ex, train, held, protocol, out = study
    from doom_learning_v6 import causal_pilot
    out.mkdir()
    pressure = causal_pilot._PressureHistory(out)
    pressure.observe('before_native_allocation')
    monkeypatch.setattr(causal_pilot, '_memory_pressure', lambda: dict(free_percent=80, swap_used_mib=1))
    with pytest.raises(ValueError, match='Swap grew'):
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double())
    saved = json.loads((out / 'pressure.json').read_text())
    assert saved['observations'][0]['name'] == 'before_native_allocation'
    assert saved['observations'][-1]['accepted'] is False
    assert all(b.backend._host_weight_epoch == 0 for b in ex.brains)


@pytest.mark.parametrize('change', ['source', 'dirty', 'protocol-sha', 'committed-protocol', 'pins', 'frames', 'count', 'fresh', 'readouts'])
def test_cli_preflight_rejects_altered_contract_before_lease_or_native(study, tmp_path, monkeypatch, change):
    module, ex, train, held, protocol, out = study
    from doom_learning_v6 import causal_pilot, demonstrations
    pin = dict(bytes=1, sha256='a' * 64)
    observed = dict(source_commit='b' * 40, native_abi=8,
                    sources={root + '/file.py': pin for root in causal_pilot._SOURCE_ROOTS},
                    inputs={name: pin for name in causal_pilot._INPUT_ROLES},
                    references={name: pin for name in causal_pilot._REFERENCE_ROLES},
                    native={name: pin for name in causal_pilot._NATIVE_ROLES},
                    cpu={name: dict(metadata=pin, binary=pin) for name in causal_pilot._CPU_ROLES})
    expected = tmp_path / 'pins.json'
    expected.write_text(json.dumps(observed))
    args = SimpleNamespace(out=str(out), source_commit='b' * 40, protocol=str(PROTOCOL_PATH),
                           protocol_sha256=PROTOCOL_SHA, reference=str(tmp_path / 'reference'),
                           expected_pins=str(expected), train=['2', '3'], eval=['4', '5'])
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    def git(*arguments):
        if arguments[0] == 'rev-parse':
            return 'b' * 40
        if arguments[0] == 'status':
            return ' M doom_learning_v6/brain.py' if change == 'dirty' else ''
        return '{}' if change == 'committed-protocol' else PROTOCOL_PATH.read_text().strip()
    monkeypatch.setattr(module, '_git_output', git)
    monkeypatch.setattr(causal_pilot, '_git_output', git)
    monkeypatch.setattr(module, '_readouts', lambda: READOUTS)
    monkeypatch.setattr(demonstrations, 'OfflineEpisode', lambda path: [*train, *held][int(path) - 2])
    monkeypatch.setattr(module, '_physical_pins', lambda reference: observed)
    def forbidden():
        raise AssertionError('rejected preflight acquired GPU lease')
    monkeypatch.setattr(module, '_acquire_gpu_lock', forbidden)
    if change == 'source':
        args.source_commit = 'c' * 40
    elif change == 'protocol-sha':
        args.protocol_sha256 = '0' * 64
    elif change == 'pins':
        altered = json.loads(expected.read_text())
        altered['native']['build.json']['sha256'] = 'c' * 64
        expected.write_text(json.dumps(altered))
    elif change == 'frames':
        train[0].frame_count -= 1
    elif change == 'count':
        args.train = ['2']
    elif change == 'fresh':
        out.mkdir()
    elif change == 'readouts':
        altered = [dict(row, index=row['index'] + 100) for row in READOUTS]
        monkeypatch.setattr(module, '_readouts', lambda: altered)
        monkeypatch.setattr(module, '_initial_reference', lambda reference, pins: {'protocol': {'readouts': READOUTS}})
    with pytest.raises(ValueError):
        module.run(args)
    assert all(b.backend._host_weight_epoch == 0 for b in ex.brains)
    if change != 'fresh':
        assert not out.exists()


def test_runtime_rejects_new_native_buffer_storage(study):
    module, ex, train, held, protocol, out = study
    def grow(wave, lane, brain, record):
        ex.mutable_bytes = 128
    with pytest.raises(ValueError, match='resident storage'):
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double(corrupt=grow))
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4


def test_portable_executor_protocol_window_rejects_before_charge(study):
    module, ex, train, held, protocol, out = study
    ex.window_ticks = 0
    with pytest.raises(ValueError):
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double())
    assert not (out / 'attempt-ledger.json').exists()


def test_secondary_pressure_read_failure_never_replaces_primary(study, monkeypatch):
    module, ex, train, held, protocol, out = study
    from doom_learning_v6 import causal_pilot
    def pressure():
        if list(out.glob('waves/*/partial.json')):
            raise OSError('secondary pressure')
        return dict(free_percent=80, swap_used_mib=0)
    monkeypatch.setattr(causal_pilot, '_memory_pressure', pressure)
    error = RuntimeError('primary')
    with pytest.raises(RuntimeError) as caught:
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double(fail_wave=8, error=error))
    assert caught.value is error
    assert any('secondary pressure' in note for note in error.__notes__)
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 32


def test_late_failure_invalidates_results_without_masking_bad_pressure_file(study):
    module, ex, train, held, protocol, out = study
    module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double())
    (out / 'pressure.json').write_text('{')
    error = RuntimeError('late cleanup')
    module._failed(out, error, module._PressureHistory(out))
    assert json.loads((out / 'results.json').read_text())['complete'] is False
    assert json.loads((out / 'failure.json').read_text())['failure']['message'] == 'late cleanup'
    assert any('pressure evidence reload failure' in note for note in error.__notes__)


def test_in_repository_unignored_output_rejects_before_native(study, monkeypatch):
    module, *_ = study
    monkeypatch.setattr(module, '_git_output', lambda *args: '')
    args = SimpleNamespace(out=str(module._ROOT / 'unignored-new-study'))
    with pytest.raises(ValueError, match='must be ignored'):
        module.run(args)


@pytest.mark.parametrize('interrupted,generated_products', [
    (True, False), (False, False),
    pytest.param(False, True, id='validated-generated-products-dirty'),
])
def test_cli_real_fit_keeps_primary_through_resource_and_lease_cleanup(study, tmp_path, monkeypatch, interrupted, generated_products):
    module, ex, train, held, protocol, out = study
    from doom_learning_v6 import causal_pilot, demonstrations
    from doom_learning_v6.metal import batch
    pin = dict(bytes=1, sha256='a' * 64)
    observed = dict(source_commit='b' * 40, native_abi=8,
                    sources={root + '/file.py': pin for root in causal_pilot._SOURCE_ROOTS},
                    inputs={name: pin for name in causal_pilot._INPUT_ROLES},
                    references={name: pin for name in causal_pilot._REFERENCE_ROLES},
                    native={name: pin for name in causal_pilot._NATIVE_ROLES},
                    cpu={name: dict(metadata=pin, binary=pin) for name in causal_pilot._CPU_ROLES})
    expected = tmp_path / 'pins.json'
    expected.write_text(json.dumps(observed))
    reference = tmp_path / 'reference'
    ex.brains[0].checkpoint(reference / 'initial.npz')
    args = SimpleNamespace(out=str(out), source_commit='b' * 40, protocol=str(PROTOCOL_PATH),
                           protocol_sha256=PROTOCOL_SHA, reference=str(reference),
                           expected_pins=str(expected), train=['2', '3'], eval=['4', '5'])
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    def git(*arguments):
        if arguments[0] == 'rev-parse':
            return 'b' * 40
        if arguments[0] == 'show':
            return PROTOCOL_PATH.read_text().strip()
        if arguments == ('status', '--porcelain') and generated_products:
            return '\n'.join([
                ' M outputs/doom-learning/libmemory.dylib.json',
                *[f' M outputs/doom-learning/physiology-v{version}/libmemory.dylib.json' for version in (2, 4, 5, 6)],
                '?? outputs/doom-learning/metal/',
            ])
        return ''  # The actual scoped model/test source query is clean.
    monkeypatch.setattr(module, '_git_output', git)
    monkeypatch.setattr(causal_pilot, '_git_output', git)
    monkeypatch.setattr(module, '_readouts', lambda: READOUTS)
    monkeypatch.setattr(demonstrations, 'OfflineEpisode', lambda path: [*train, *held][int(path) - 2])
    monkeypatch.setattr(module, '_physical_pins', lambda reference: observed)
    monkeypatch.setattr(module, '_initial_reference', lambda reference, pins: {'protocol': {'readouts': READOUTS}})
    closed = []
    def brains(resources, factory, initial, count):
        for lane, brain in enumerate(ex.brains):
            brain.backend.close = lambda lane=lane: closed.append(f'brain-{lane}')
            resources.add(brain.backend)
        return ex.brains
    monkeypatch.setattr(module, '_restored_brains', brains)
    resident_error = OSError('resident cleanup')
    def close_executor():
        closed.append('executor')
        raise resident_error
    ex.close = close_executor
    monkeypatch.setattr(batch, 'MetalBatchExecutor', lambda brains, window_ticks: ex)
    class Lease:
        def fileno(self):
            return 17
        def close(self):
            closed.append('lease')
            raise OSError('lease cleanup')
    monkeypatch.setattr(module, '_acquire_gpu_lock', Lease)
    monkeypatch.setattr(module.fcntl, 'flock', lambda fd, operation: None)
    original_fit = module.fit
    error = KeyboardInterrupt('primary replay') if interrupted else resident_error
    def fitting(*args):
        return original_fit(*args, replay=replay_double(fail_wave=8 if interrupted else None, error=error))
    monkeypatch.setattr(module, 'fit', fitting)
    with pytest.raises(type(error)) as caught:
        module.run(args)
    assert caught.value is error
    assert closed == ['executor', 'brain-3', 'brain-2', 'brain-1', 'brain-0', 'lease']
    assert any('lease cleanup' in note for note in error.__notes__)
    if interrupted:
        assert any('resident cleanup' in note for note in error.__notes__)
    else:
        assert json.loads((out / 'results.json').read_text())['complete'] is False
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == (32 if interrupted else 120)
    assert json.loads((out / 'failure.json').read_text())['complete'] is False
    assert (out.stat().st_mode & 0o777) == 0o700


def test_second_bank_proposal_failure_preserves_previous_accepted_center_and_contrast(study):
    module, ex, train, held, protocol, out = study
    def loss(wave, lane, data):
        if wave['kind'] == 'probe':
            return 1. if wave['signs'][lane] == '+' else 3.
        return 2. if wave['kind'] == 'proposal' else 4.
    error = RuntimeError('second-bank failure')
    with pytest.raises(RuntimeError) as caught:
        module.fit(ex, train, held, READOUTS, protocol, out,
                   replay=replay_double(loss, fail_wave=14, error=error))
    assert caught.value is error
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 56
    planned = json.loads((out / 'generations/g-1/proposal-plan.json').read_text())
    assert all(row['contrast'] == -2. and row['incumbent_loss'] == 2. for row in planned)
    for row in planned:
        saved = np.load(out / 'incumbents' / row['run'] / 'theta.npy')
        accepted = np.load(out / 'generations/g-0' / row['run'] / 'proposal-theta.npy')
        assert np.array_equal(saved, accepted)
        assert np.max(np.abs(saved)) == pytest.approx(.2)
    assert not (out / 'generations/g-1/decisions.json').exists()
    assert not (out / 'final-candidates.json').exists()


@pytest.mark.parametrize('change', ['ignored-fractions', 'all-nonplastic'])
def test_installer_cannot_silently_ignore_candidate_or_change_fixed_graph(study, change):
    module, ex, train, held, protocol, out = study
    installer = ex.install_efficacies
    def corrupt(lane, values, edges):
        installer(lane, np.ones(4184) if change == 'ignored-fractions' else values, edges)
        if change == 'all-nonplastic':
            ex.brains[lane].weight[-1] = 2.
    ex.install_efficacies = corrupt
    with pytest.raises(ValueError):
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double())
    ledger = json.loads((out / 'attempt-ledger.json').read_text())
    assert ledger['used'] == (12 if change == 'ignored-fractions' else 4)


@pytest.mark.parametrize('poisoned', [False, True])
def test_before_replay_install_failure_records_safe_all24_availability(study, poisoned):
    module, ex, train, held, protocol, out = study
    original_installer = ex.install_efficacies
    error = RuntimeError('half-installed native candidate')
    def bad_checkpoint(path):
        raise OSError('poisoned materialization unavailable')
    def installer(lane, values, edges):
        original_installer(lane, values, edges)
        if lane == 2:
            if poisoned:
                for brain in ex.brains:
                    brain.checkpoint = bad_checkpoint
            raise error
    ex.install_efficacies = installer
    with pytest.raises(RuntimeError) as caught:
        module.fit(ex, train, held, READOUTS, protocol, out, replay=replay_double())
    assert caught.value is error
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4
    availability = json.loads((out / 'failure-checkpoints.json').read_text())
    assert len(availability['lanes']) == 4
    assert all(row['all24_available'] is not poisoned for row in availability['lanes'])
    if poisoned:
        assert all('poisoned materialization unavailable' in row['reason'] for row in availability['lanes'])
        assert all(f'lane-{lane}' in ' '.join(error.__notes__) for lane in range(4))
        assert not list(out.glob('waves/*/lane-*/failure.npz'))
    else:
        for row in availability['lanes']:
            with np.load(out / row['path'], allow_pickle=False) as checkpoint:
                assert len(checkpoint.files) == 24
    assert not (out / 'baseline-scores.json').exists()
