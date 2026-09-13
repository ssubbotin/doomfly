"""Closed-loop numerical fixtures establish no biological learning claim."""
import copy
import importlib
import importlib.util
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from doom_learning.common import digest
from doom_learning_v6 import live_controls
from test_doom_learning_same_slot_runner import _same_slot_visual_brain
from test_doom_learning_causal_pilot import READOUTS

ROOT = Path(__file__).resolve().parents[1]
READOUTS14 = READOUTS + [{'index': i, 'type': 'traced', 'side': ''} for i in range(4, 14)]


def runner():
    assert importlib.util.find_spec('doom_learning_v6.live_transfer') is not None, 'Frozen live runner is missing'
    return importlib.import_module('doom_learning_v6.live_transfer')


class ShortGame:
    """Only the external engine boundary is substituted; neural physics is real."""
    def __init__(self, death=2, *, failure=None, variant=0):
        self.tick = 0
        self.death, self.failure, self.variant = death, failure, variant
        self.act_calls = self.pixels_calls = self.close_calls = 0
        self.game = SimpleNamespace(get_episode_time=lambda: self.tick + 1,
                                    is_player_dead=lambda: self.tick >= self.death,
                                    is_episode_finished=lambda: self.tick >= self.death,
                                    get_episode_timeout=lambda: 99)
        self.assets = {'fixture': True}

    def pixels(self):
        assert self.tick < self.death, 'Pixels requested after termination'
        self.pixels_calls += 1
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 2] = 100 + self.variant
        return image

    def act(self, action):
        assert self.tick < self.death, 'Action applied after termination'
        assert set(action) == {'turn', 'forward', 'attack', 'readouts'}
        self.act_calls += 1
        if self.failure is not None:
            raise self.failure
        self.tick += 1

    def observation(self):
        return {'tick': self.tick, 'health': 100 - self.tick * 10, 'kills': self.tick,
                'finished': self.tick >= self.death}

    def close(self):
        self.close_calls += 1


class SerialVisualExecutor:
    """Real CPU visual propagation, with explicit simulated upload accounting."""
    window_ticks = 18

    def __init__(self, brains):
        self.brains = tuple(brains)
        self.rgb_calls, self.installs, self.last_timing = [], [], {}
        self.before_install = None
        for brain in brains:
            backend = brain.backend
            backend._host_weight_epoch = backend._device_weight_epoch = 0
            def upload(*args, backend=backend, **kwargs):
                backend._host_weight_epoch += 1
                backend._device_weight_epoch += 1
            backend.restore_from_host = backend.update_weights = upload

    def install_efficacies(self, lane, fractions, expected_edges):
        from doom_learning_v6.causal_controls import initialize_efficacies
        if self.before_install:
            self.before_install(lane)
        brain = self.brains[lane]
        np.testing.assert_array_equal(expected_edges, brain.circuit['edges'])
        initialize_efficacies(brain, fractions - 1.)
        brain.backend.update_weights(expected_edges, brain.weight[expected_edges])
        self.installs.append(lane)

    def metadata(self):
        return {'window_ticks': 18, 'lane_count': 4, 'shared_resident_bytes': 11,
                'mutable_resident_bytes': 22}

    def rgb_step(self, images, duration, *, learning, stimulations):
        assert learning == [False] * 4 and stimulations == [None] * 4
        assert all(brain.weights_frozen is True for brain in self.brains)
        self.rgb_calls.append({'duration': duration, 'learning': learning, 'stimulations': stimulations})
        counts, elapsed = [], 0.
        for brain, image in zip(self.brains, images):
            spikes, clock = brain.rgb_step(image, duration, learning=False, stimulation=None)
            counts.append(spikes)
            elapsed += clock
        self.last_timing = {'sparse_weight_update_bytes': 0, 'materialize_bytes': 0, 'backend_seconds': elapsed}
        return counts, elapsed


@pytest.fixture
def case(tmp_path):
    brains = [_same_slot_visual_brain(tmp_path) for _ in range(4)]
    return SimpleNamespace(brains=brains, executor=SerialVisualExecutor(brains),
                           games=[ShortGame(death=d) for d in (2, 3, 5, 99)],
                           out=tmp_path / 'phase', readouts=READOUTS14)


def wave(case, **kwargs):
    return runner().live_wave(case.brains, case.executor, case.games, case.readouts,
                              horizon_tics=kwargs.pop('horizon_tics', 7),
                              warmup_ms=kwargs.pop('warmup_ms', 2000), directory=case.out, **kwargs)


def test_dead_game_pads_without_steering_or_masking(case):
    phase = wave(case)
    assert phase['complete'] is True
    assert [len(row['trace']) for row in phase['lanes']] == [7] * 4
    assert [(g.act_calls, g.pixels_calls) for g in case.games] == [(2, 2), (3, 3), (5, 5), (7, 7)]
    assert phase['lanes'][0]['trace'][2]['phase'] == 'padding'
    assert phase['lanes'][0]['trace'][2]['applied_action'] is None
    assert phase['lanes'][0]['trace'][2]['requested_action']['readouts']
    assert [row['neural_steps'] for row in phase['lanes'][0]['trace']] == [286, 285, 286, 286, 286, 285, 286]
    assert all(b.cursor == 22000 for b in case.brains)
    for lane, record in enumerate(phase['lanes']):
        assert record['checkpoint_scope'] == 'wave-final after complete horizon padding'
        with np.load(case.out / f'lane-{lane}/final.npz', allow_pickle=False) as cp:
            assert len(set(cp.files) - {'metadata'}) == 24
            assert json.loads(str(cp['metadata']))['cursor'] == 22000
        assert record['memory_diagnostic_scope'] == 'host mirrors; terminal materialization is authoritative'
    assert phase['timing']['rgb_calls'] == 8
    assert phase['timing']['batch_native_seconds'] == pytest.approx(sum(row['native_elapsed_seconds'] for row in phase['batch_calls']))


@pytest.mark.parametrize('break_contract', ['duplicate-brain', 'duplicate-game', 'duplicate-engine', 'readouts', 'horizon', 'fresh'])
def test_live_preflight_has_no_reset_or_rgb_side_effects(case, break_contract):
    kwargs = {}
    if break_contract == 'duplicate-brain': case.brains[1] = case.brains[0]
    if break_contract == 'duplicate-game': case.games[1] = case.games[0]
    if break_contract == 'duplicate-engine': case.games[1].game = case.games[0].game
    if break_contract == 'readouts': case.readouts = READOUTS
    if break_contract == 'horizon': kwargs['horizon_tics'] = True
    if break_contract == 'fresh': case.out.mkdir()
    with pytest.raises(ValueError): wave(case, **kwargs)
    assert not case.executor.rgb_calls
    assert all(b.backend._host_weight_epoch == 0 for b in case.brains)


@pytest.mark.parametrize('secondary', [None, 'json', 'checkpoint', 'sync'])
def test_interruption_preserves_exact_primary_and_independent_checkpoints(case, monkeypatch, secondary):
    module = runner()
    primary = KeyboardInterrupt('engine action interruption')
    case.games[0].failure = primary
    syncs, cps = [], []
    for lane, brain in enumerate(case.brains):
        sync, checkpoint = brain.backend.sync_for_checkpoint, brain.checkpoint
        def synchronize(lane=lane, sync=sync):
            syncs.append(lane)
            if secondary == 'sync' and lane == 1: raise RuntimeError('usable-owner download failed')
            return sync()
        def write(path, lane=lane, checkpoint=checkpoint):
            cps.append(lane)
            if secondary == 'checkpoint' and lane == 1: raise RuntimeError('checkpoint failed')
            return checkpoint(path)
        monkeypatch.setattr(brain.backend, 'sync_for_checkpoint', synchronize)
        monkeypatch.setattr(brain, 'checkpoint', write)
    writer = module.save_json
    def save(path, value):
        if secondary == 'json' and Path(path).name == 'summary.json': raise OSError('evidence writer failed')
        return writer(path, value)
    monkeypatch.setattr(module, 'save_json', save)
    with pytest.raises(KeyboardInterrupt) as caught: wave(case, warmup_ms=0)
    assert caught.value is primary
    availability = primary._doomfly_live_availability
    assert availability['directory'] == str(case.out.resolve())
    assert len(availability['lanes']) == 4
    assert (case.out / 'lane-3/failure.npz').exists()
    if secondary == 'sync':
        assert syncs.count(1) == 1 and 1 not in cps
        assert availability['lanes'][1]['checkpoint_available'] is False


@pytest.fixture
def study(case, monkeypatch, tmp_path):
    module = runner()
    protocol = live_controls.load_live_protocol(live_controls._PROTOCOL_PATH, live_controls.LIVE_PROTOCOL_SHA256)
    protocol['horizon_tics'], protocol['warmup_ms'] = 2, 0
    monkeypatch.setattr(module, '_disk', lambda *args: None)
    monkeypatch.setattr(importlib.import_module('doom_learning_v6.causal_pilot'), '_memory_pressure',
                        lambda: {'free_percent': 80, 'swap_used_mib': 0})
    candidates = {}
    for i, row in enumerate(protocol['candidates']):
        values = np.full(4184, 1. + i / 1000, dtype=np.float64)
        values.setflags(write=False)
        candidates[row['role']] = values
        raw = BytesIO()
        np.save(raw, values, allow_pickle=False)
        data = raw.getvalue()
        row['pin'] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    candidates['baseline-filler'] = candidates['baseline']
    monkeypatch.setattr(live_controls, '_PINNED_PROTOCOL', copy.deepcopy(protocol))
    return module, protocol, candidates, case.executor, tmp_path / 'study'


def test_evaluate_runs_all_roles_and_charges_before_every_install(study):
    module, protocol, candidates, executor, out = study
    games = []
    def before_install(lane):
        ledger = json.loads((out / 'attempt-ledger.json').read_text())
        assert ledger['used'] == len(executor.installs) // 4 * 4 + 4
        assert ledger['waves'][-1]['state'] == 'charged'
    executor.before_install = before_install
    def factory(wave, lane, directory):
        game = ShortGame(death=99)
        games.append(game)
        return game
    result = module.evaluate(executor, candidates, READOUTS14, protocol, out, game_factory=factory)
    assert result['complete'] is True
    assert result['attempts'] == 72 and result['remaining'] == 8
    assert len(result['waves']) == 18 and len(result['role_results']) == 64
    assert sum(row['role'] == 'baseline-filler' for row in result['role_results']) == 8
    assert [row['kind'] for row in result['waves'][:2]] == ['calibration'] * 2
    assert all(game.close_calls == 1 for game in games)
    assert all(not value.flags.writeable for value in candidates.values())
    assert result['learning_demonstrated'] is False
    assert result['waves'][4]['roles'] == ['spsa-200000201', 'spsa-200000202', 'gaussian_es-200000201', 'gaussian_es-200000202']
    assert result['waves'][5]['roles'] == ['random-200000301', 'random-200000302', 'baseline', 'baseline-filler']
    assert all(all(snapshot for snapshot in wave['runtime_frozen']) for wave in result['waves'])


@pytest.mark.parametrize('mutation', [lambda p: p.__setitem__('learning', 0), lambda p: p.__setitem__('horizon_tics', 2)])
def test_exact_protocol_rejection_precedes_charge_and_native(case, mutation, tmp_path):
    protocol = live_controls.load_live_protocol(live_controls._PROTOCOL_PATH, live_controls.LIVE_PROTOCOL_SHA256)
    mutation(protocol)
    with pytest.raises(ValueError):
        runner().evaluate(case.executor, {}, READOUTS14, protocol, tmp_path / 'bad', game_factory=lambda *args: pytest.fail('game allocated'))
    assert not case.executor.installs and not case.executor.rgb_calls
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('malformed', [None, {}, {'doom/game.py': '0' * 64},
                                    {'doom/game.py': 'X' * 64, 'doom_learning/survival_arena.py': '0' * 64}])
def test_declared_boundary_pins_are_checked_before_reference_access(tmp_path, malformed):
    from doom_learning_v6.same_slot_runner import _initial_reference
    observed = {'sources': {name: {'sha256': 'a' * 64} for name in ('doom/game.py', 'doom_learning/survival_arena.py')}}
    declaration = malformed if malformed is not None else {name: 'b' * 64 for name in observed['sources']}
    with pytest.raises(ValueError):
        _initial_reference(tmp_path / 'missing-reference', observed, boundary_sources=declaration)


def test_reference_transitions_preserve_default_and_unrelated_source_guards():
    from doom_learning_v6 import same_slot_runner as same
    previous = {'doom/game.py': '0' * 64, 'doom_learning/survival_arena.py': '1' * 64,
                'doom_learning_v6/rule.py': '2' * 64}
    observed = {'sources': {name: {'sha256': value} for name, value in previous.items()}}
    observed['sources']['doom/game.py']['sha256'] = 'a' * 64
    observed['sources']['doom_learning/survival_arena.py']['sha256'] = 'b' * 64
    assert hasattr(same, '_reference_source_transitions'), 'Source-transition validator missing'
    with pytest.raises(ValueError): same._reference_source_transitions(previous, observed)
    declaration = {name: observed['sources'][name]['sha256'] for name in ('doom/game.py', 'doom_learning/survival_arena.py')}
    assert same._reference_source_transitions(previous, observed, boundary_sources=declaration) == {
        'doom/game.py': {'original': '0' * 64, 'current': 'a' * 64},
        'doom_learning/survival_arena.py': {'original': '1' * 64, 'current': 'b' * 64}}
    observed['sources']['doom_learning_v6/rule.py']['sha256'] = 'c' * 64
    with pytest.raises(ValueError): same._reference_source_transitions(previous, observed, boundary_sources=declaration)


def test_cli_requires_every_declared_input_and_rejects_platform_before_native(case):
    module = runner()
    flags = {name: 'unused' for name in ('protocol', 'protocol-sha256', 'reference', 'expected-pins', 'source-commit', 'candidates-root', 'out')}
    with pytest.raises(SystemExit): module.build_parser().parse_args([])
    args = module.build_parser().parse_args([word for key, value in flags.items() for word in ('--' + key, value)])
    with pytest.raises(ValueError): module.run(args)
    assert not case.executor.rgb_calls


def _native_live_visual_brain(tmp_path):
    """Keep the numerical graph while giving gain the native DAN-by-slot schema."""
    brain = _same_slot_visual_brain(tmp_path)
    brain.circuit['gain'] = np.ascontiguousarray(brain.circuit['gain'].reshape(1, 4184), dtype=np.float32)
    return brain


@pytest.mark.skipif(sys.platform != 'darwin', reason='Real Metal requires macOS')
def test_native_four_lane_live_fixture(tmp_path):
    from doom_learning_v6.metal.batch import MetalBatchExecutor
    brains = [_native_live_visual_brain(tmp_path) for _ in range(4)]
    with MetalBatchExecutor(brains, window_ticks=18) as executor:
        result = runner().live_wave(brains, executor, [ShortGame(death=d) for d in (1, 2, 3, 99)],
                                    READOUTS14, horizon_tics=4, warmup_ms=10, directory=tmp_path / 'native')
        assert result['complete'] is True
        assert [b.cursor for b in brains] == [1243] * 4


def test_native_live_synthetic_graph_passes_real_portable_schema_validator(tmp_path):
    from doom_learning_v6.metal.batch import _validate_graph
    original = _same_slot_visual_brain(tmp_path)
    brain = _native_live_visual_brain(tmp_path)
    assert _validate_graph(brain) is None
    assert brain.circuit['gain'].shape == (1, 4184)
    assert brain.circuit['gain'].dtype == np.float32 and brain.circuit['gain'].flags.c_contiguous
    assert brain.circuit['gain'].tobytes() == original.circuit['gain'].tobytes()
    for name in ('ptr', 'post', 'ids', 'weight', 'baseline_plastic'):
        np.testing.assert_array_equal(getattr(brain, name), getattr(original, name))
    for name in ('edges', 'pre', 'dan', 'kc_mask', 'dan_index'):
        np.testing.assert_array_equal(brain.circuit[name], original.circuit[name])


def test_real_doom_rgb_act_observer_boundary(tmp_path):
    from doom.game import Game
    module = runner()
    game = Game(seed=202609131, episode_timeout_tics=2, episode_start_tics=0)
    try:
        initial = module._observe(game)
        assert initial['engine_tic'] == 1
        assert game.pixels().shape == (480, 640, 3)
        game.act({'turn': 0., 'forward': 0., 'attack': False})
        after = module._observe(game)
        assert after['engine_tic'] == initial['engine_tic'] + 1
        assert after['timeout'] is False
    finally:
        game.close()


@pytest.mark.parametrize('change', ['weight', 'u', 'w', 'configuration', 'epoch', 'storage', 'frozen-flag'])
def test_runtime_mutations_stop_frozen_success(case, monkeypatch, change):
    module = runner()
    original = case.executor.rgb_step
    def advance(*args, **kwargs):
        counts, elapsed = original(*args, **kwargs)
        brain = case.brains[0]
        if change == 'weight': brain.weight[-1] += .01
        if change == 'u': brain.memory_u[0] += .01
        if change == 'w': brain.memory_w[0] += .01
        if change == 'configuration': brain.tonic[-1] += .01
        if change == 'epoch': brain.backend._host_weight_epoch += 1; brain.backend._device_weight_epoch += 1
        if change == 'storage':
            monkeypatch.setattr(case.executor, 'metadata', lambda: {'window_ticks': 18, 'lane_count': 4,
                                                                  'shared_resident_bytes': 11, 'mutable_resident_bytes': 23})
        if change == 'frozen-flag': brain.weights_frozen = False
        return counts, elapsed
    monkeypatch.setattr(case.executor, 'rgb_step', advance)
    with pytest.raises(ValueError): wave(case, warmup_ms=0, horizon_tics=1)
    assert json.loads((case.out / 'summary.json').read_text())['complete'] is False


def test_non_poisoning_terminal_failure_is_primary_and_never_retried(case, monkeypatch):
    module = runner()
    primary = RuntimeError('single terminal download failure')
    downloads, checkpoints = [], []
    def materialize(reason):
        downloads.append(reason)
        if reason == 'checkpoint': raise primary
    monkeypatch.setattr(case.brains[1].backend, 'materialize', materialize)
    cp = case.brains[1].checkpoint
    def checkpoint(path):
        checkpoints.append(path)
        return cp(path)
    monkeypatch.setattr(case.brains[1], 'checkpoint', checkpoint)
    with pytest.raises(RuntimeError) as caught: wave(case, warmup_ms=0)
    assert caught.value is primary
    assert downloads.count('checkpoint') == 1 and not checkpoints
    assert primary._doomfly_live_availability['lanes'][1]['checkpoint_attempted'] is False
    assert all((case.out / f'lane-{lane}/failure.npz').exists() for lane in (2, 3))
    assert json.loads((case.out / 'lane-1/episode.json').read_text())['complete'] is False


def test_all_neural_lanes_keep_trace_when_first_action_interrupts(case):
    primary = KeyboardInterrupt('first action')
    case.games[0].failure = primary
    with pytest.raises(KeyboardInterrupt) as caught: wave(case, warmup_ms=0)
    assert caught.value is primary
    assert [len(record['trace']) for record in primary._doomfly_live_availability['lanes']] == [1] * 4
    assert all(brain.cursor == 286 for brain in case.brains)
    assert [game.act_calls for game in case.games] == [1, 0, 0, 0]


@pytest.mark.parametrize('where', ['constructor', 'install', 'render', 'act', 'game-close', 'pressure'])
def test_evaluation_failure_keeps_charge_games_and_healthy_evidence(study, monkeypatch, where):
    module, protocol, candidates, executor, out = study
    primary = KeyboardInterrupt(where)
    games = []
    if where == 'install':
        install = executor.install_efficacies
        def fail(lane, *args):
            if lane == 1: raise primary
            return install(lane, *args)
        monkeypatch.setattr(executor, 'install_efficacies', fail)
    if where == 'pressure':
        pilot = importlib.import_module('doom_learning_v6.causal_pilot')
        readings = iter([{'free_percent': 80, 'swap_used_mib': 0}, {'free_percent': 80, 'swap_used_mib': 1}])
        monkeypatch.setattr(pilot, '_memory_pressure', lambda: next(readings))
    def factory(wave, lane, directory):
        if where == 'constructor' and lane == 1: raise primary
        game = ShortGame(death=99)
        if where == 'act' and lane == 0: game.failure = primary
        if where == 'render' and lane == 0:
            def pixels(): raise primary
            game.pixels = pixels
        if where == 'game-close' and lane == 0:
            def close():
                game.close_calls += 1
                raise primary
            game.close = close
        games.append(game)
        return game
    with pytest.raises(BaseException) as caught:
        module.evaluate(executor, candidates, READOUTS14, protocol, out, game_factory=factory)
    if where != 'pressure': assert caught.value is primary
    else: assert isinstance(caught.value, ValueError)
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4
    assert all(game.close_calls == 1 for game in games)
    assert any((out / 'waves/w-00/lane-3' / name).exists() for name in ('failure.npz', 'final.npz'))
    assert json.loads((out / 'failure.json').read_text())['complete'] is False


def test_baseline_calibration_mismatch_precedes_metrics(study, monkeypatch):
    module, protocol, candidates, executor, out = study
    metrics_calls = []
    def metrics(*args):
        metrics_calls.append(args)
        pytest.fail('Metrics precede calibration validation')
    monkeypatch.setattr(module, 'gameplay_metrics', metrics)
    with pytest.raises(ValueError, match='Baseline canonical'):
        module.evaluate(executor, candidates, READOUTS14, protocol, out,
                        game_factory=lambda wave, lane, directory: ShortGame(death=99, variant=lane))
    assert not metrics_calls
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4


@pytest.mark.parametrize('malform', ['writable', 'pin', 'missing', 'filler'])
def test_candidate_preflight_rejects_before_charge(study, malform):
    module, protocol, candidates, executor, out = study
    if malform == 'writable': candidates['baseline'] = candidates['baseline'].copy()
    if malform == 'pin':
        value = candidates[protocol['candidates'][1]['role']].copy()
        value[0] += .001
        value.setflags(write=False)
        candidates[protocol['candidates'][1]['role']] = value
    if malform == 'missing': del candidates[protocol['candidates'][1]['role']]
    if malform == 'filler': candidates['baseline-filler'] = candidates['baseline'].copy()
    with pytest.raises(ValueError):
        module.evaluate(executor, candidates, READOUTS14, protocol, out, game_factory=lambda *args: pytest.fail('allocated'))
    assert not executor.installs and not out.exists()


@pytest.mark.parametrize('alias', ['dictionary', 'key', 'sha', 'missing', 'extra', 'uppercase'])
def test_boundary_declaration_rejects_builtin_type_aliases_before_io(tmp_path, alias):
    from doom_learning_v6.same_slot_runner import _initial_reference
    class String(str): pass
    class Dictionary(dict): pass
    declaration = {name: 'a' * 64 for name in ('doom/game.py', 'doom_learning/survival_arena.py')}
    observed = {'sources': {name: {'sha256': value} for name, value in declaration.items()}}
    if alias == 'dictionary': declaration = Dictionary(declaration)
    if alias == 'key': declaration[String('doom/game.py')] = declaration.pop('doom/game.py')
    if alias == 'sha': declaration['doom/game.py'] = String('a' * 64)
    if alias == 'missing': declaration.pop('doom/game.py')
    if alias == 'extra': declaration['doom_learning_v6/rule.py'] = 'a' * 64
    if alias == 'uppercase': declaration['doom/game.py'] = 'A' * 64
    with pytest.raises(ValueError): _initial_reference(tmp_path / 'absent', observed, boundary_sources=declaration)


@pytest.fixture
def cli(study, tmp_path, monkeypatch):
    module, protocol, candidates, executor, out = study
    from doom_learning_v6 import causal_pilot, calibration
    from doom_learning_v6.metal import batch
    pin = {'bytes': 1, 'sha256': 'a' * 64}
    observed = {'source_commit': 'b' * 40, 'native_abi': 8,
                'sources': {root + '/file.py': pin for root in causal_pilot._SOURCE_ROOTS},
                'inputs': {name: pin for name in causal_pilot._INPUT_ROLES},
                'references': {name: pin for name in causal_pilot._REFERENCE_ROLES},
                'native': {name: pin for name in causal_pilot._NATIVE_ROLES},
                'cpu': {name: {'metadata': pin, 'binary': pin} for name in causal_pilot._CPU_ROLES}}
    observed['sources'].update({name: pin for name in ('doom/game.py', 'doom_learning/survival_arena.py')})
    expected = tmp_path / 'pins.json'
    expected.write_text(json.dumps(observed))
    reference = tmp_path / 'reference'
    executor.brains[0].checkpoint(reference / 'initial.npz')
    source = tmp_path / 'portable-protocol.json'
    source.write_text(json.dumps(protocol) + '\n')
    args = SimpleNamespace(out=str(out), source_commit='b' * 40, protocol=str(source),
                           protocol_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           reference=str(reference), expected_pins=str(expected), candidates_root=str(tmp_path))
    for row in protocol['candidates']:
        path = tmp_path / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, candidates[row['role']], allow_pickle=False)
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.platform, 'system', lambda: 'Darwin')
    def git(*arguments):
        if arguments[0] == 'rev-parse': return 'b' * 40
        if arguments[0] == 'show': return source.read_text().strip()
        return ''
    monkeypatch.setattr(module, '_git_output', git)
    monkeypatch.setattr(causal_pilot, '_git_output', git)
    # Full graph and Darwin engine availability are the substituted external gates.
    monkeypatch.setattr(module, 'load_live_protocol', lambda path, sha: copy.deepcopy(protocol))
    monkeypatch.setattr(module, '_engine_pins', lambda p: p['game_assets'])
    monkeypatch.setattr(module, '_physical_pins', lambda reference: copy.deepcopy(observed))
    declarations = []
    def relationships(reference, pins, *, boundary_sources):
        declarations.append(boundary_sources)
        assert boundary_sources == {name: 'a' * 64 for name in ('doom/game.py', 'doom_learning/survival_arena.py')}
        return {'protocol': {'readouts': READOUTS14}}
    monkeypatch.setattr(module, '_initial_reference', relationships)
    monkeypatch.setattr(module, '_readouts', lambda: READOUTS14)
    monkeypatch.setattr(module, '_model_gate', lambda brains, original: None)
    iterator = iter(executor.brains)
    monkeypatch.setattr(calibration, 'calibrated_brain', lambda *args, **kwargs: next(iterator))
    monkeypatch.setattr(batch, 'MetalBatchExecutor', lambda brains, window_ticks: executor)
    monkeypatch.setattr(module, '_game_factory', lambda p: lambda *args: ShortGame(death=99))
    closed = []
    for lane, brain in enumerate(executor.brains):
        monkeypatch.setattr(brain.backend, 'close', lambda lane=lane: closed.append(f'brain-{lane}'))
    executor.close = lambda: closed.append('executor')
    class Lease:
        def fileno(self): return 17
        def close(self): closed.append('lease')
    lease = Lease()
    monkeypatch.setattr(module, '_acquire_gpu_lock', lambda: lease)
    monkeypatch.setattr(module.fcntl, 'flock', lambda *args: closed.append('unlock'))
    return SimpleNamespace(module=module, args=args, executor=executor, out=out,
                           closed=closed, declarations=declarations, observed=observed, lease=lease)


@pytest.mark.parametrize('failure_at', ['resource', 'after-pins', 'lease', 'interrupted-with-cleanup'])
def test_cli_late_failure_invalidates_results_and_preserves_primary_without_second_cp(cli, monkeypatch, failure_at):
    primary = KeyboardInterrupt(failure_at)
    downloads = []
    for lane, brain in enumerate(cli.executor.brains):
        sync = brain.backend.sync_for_checkpoint
        def synchronize(lane=lane, sync=sync):
            downloads.append(lane)
            return sync()
        monkeypatch.setattr(brain.backend, 'sync_for_checkpoint', synchronize)
    if failure_at in ('resource', 'interrupted-with-cleanup'):
        def release():
            cli.closed.append('executor')
            raise primary if failure_at == 'resource' else RuntimeError('secondary resident cleanup')
        cli.executor.close = release
    if failure_at == 'after-pins':
        calls = []
        def pins(reference):
            calls.append(reference)
            if len(calls) == 2: raise primary
            return copy.deepcopy(cli.observed)
        monkeypatch.setattr(cli.module, '_physical_pins', pins)
    if failure_at in ('lease', 'interrupted-with-cleanup'):
        def close():
            cli.closed.append('lease')
            raise primary if failure_at == 'lease' else OSError('secondary lease cleanup')
        monkeypatch.setattr(cli.lease, 'close', close)
    if failure_at == 'interrupted-with-cleanup':
        monkeypatch.setattr(cli.module, '_game_factory', lambda p: lambda *args: ShortGame(failure=primary))
    with pytest.raises(KeyboardInterrupt) as caught: cli.module.run(cli.args)
    assert caught.value is primary
    assert cli.closed == ['executor', 'brain-3', 'brain-2', 'brain-1', 'brain-0', 'unlock', 'lease']
    ledger = json.loads((cli.out / 'attempt-ledger.json').read_text())
    assert ledger['used'] == (4 if failure_at == 'interrupted-with-cleanup' else 72)
    # Four initial comparisons plus one terminal sync and one CP per lane/wave.
    assert len(downloads) == 4 + (8 if failure_at == 'interrupted-with-cleanup' else 18 * 8)
    if (cli.out / 'results.json').exists():
        assert json.loads((cli.out / 'results.json').read_text())['complete'] is False
    assert json.loads((cli.out / 'failure.json').read_text())['complete'] is False
    assert cli.declarations == [{name: 'a' * 64 for name in ('doom/game.py', 'doom_learning/survival_arena.py')}]


def test_checkpoint_bytes_are_validated_against_materialized_terminal(case, monkeypatch):
    cp = case.brains[0].checkpoint
    def corrupt(path):
        cp(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays['weight'][0] += .1
        np.savez(path, **arrays)
    monkeypatch.setattr(case.brains[0], 'checkpoint', corrupt)
    with pytest.raises(ValueError, match='checkpoint'): wave(case, warmup_ms=0)


def test_shared_native_timing_aggregates_once_with_warmup_separate(case, monkeypatch):
    original = case.executor.rgb_step
    def advance(*args, **kwargs):
        counts, elapsed = original(*args, **kwargs)
        case.executor.last_timing.update(input_preparation_seconds=.125, input_preflight_seconds=.25,
                                        host_rule_seconds=.5, encoder_count=3)
        return counts, elapsed
    monkeypatch.setattr(case.executor, 'rgb_step', advance)
    phase = wave(case)
    assert phase['timing']['shared_native_timing']['input_preparation_seconds'] == 1.
    assert phase['timing']['shared_native_timing']['host_rule_seconds'] == 4.
    assert phase['timing']['shared_native_timing']['encoder_count'] == 24
    assert phase['warmup_timing']['shared_native_timing']['input_preparation_seconds'] == .125
    assert phase['rollout_timing']['shared_native_timing']['input_preparation_seconds'] == .875
    persisted = json.loads((case.out / 'summary.json').read_text())
    assert persisted['warmup_timing'] == phase['warmup_timing']
    assert persisted['rollout_timing'] == phase['rollout_timing']


def test_duplicate_factory_game_is_owned_and_closed_once_with_pre_wave_evidence(study):
    module, protocol, candidates, executor, out = study
    game = ShortGame(death=99)
    with pytest.raises(ValueError):
        module.evaluate(executor, candidates, READOUTS14, protocol, out, game_factory=lambda *args: game)
    assert game.close_calls == 1
    assert not executor.rgb_calls
    assert (out / 'waves/w-00/lane-3/failure.npz').exists()


def test_terminal_materialization_detects_device_only_frozen_change(case, monkeypatch):
    original = case.brains[0].backend.materialize
    calls = []
    def materialize(reason):
        calls.append(reason)
        if reason == 'checkpoint': case.brains[0].memory_w[0] += .125
        return original(reason)
    monkeypatch.setattr(case.brains[0].backend, 'materialize', materialize)
    with pytest.raises(ValueError): wave(case, warmup_ms=0)
    assert calls == ['same-slot-frozen-full-weight', 'checkpoint', 'checkpoint']
    assert case.brains[0].memory_w[0] == .25


@pytest.mark.parametrize('gate', ['source', 'protocol-bytes', 'pins', 'readouts', 'initial-arrays'])
def test_cli_preflight_stops_before_resident_constructor_and_charge(cli, monkeypatch, gate):
    from doom_learning_v6.metal import batch
    def construct(*args, **kwargs): pytest.fail('Native owner allocated before preflight')
    monkeypatch.setattr(batch, 'MetalBatchExecutor', construct)
    if gate == 'source': cli.args.source_commit = 'c' * 40
    if gate == 'protocol-bytes': cli.args.protocol_sha256 = 'c' * 64
    if gate == 'pins':
        expected = json.loads(Path(cli.args.expected_pins).read_text())
        expected['sources']['doom/game.py']['sha256'] = 'c' * 64
        Path(cli.args.expected_pins).write_text(json.dumps(expected))
    if gate == 'readouts': monkeypatch.setattr(cli.module, '_readouts', lambda: READOUTS14[::-1])
    if gate == 'initial-arrays':
        cp = cli.executor.brains[0].checkpoint
        def checkpoint(path):
            cp(path)
            if Path(path).parent.name == 'initial':
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {name: archive[name].copy() for name in archive.files}
                arrays['g'][0] += 1.
                np.savez(path, **arrays)
        monkeypatch.setattr(cli.executor.brains[0], 'checkpoint', checkpoint)
    with pytest.raises(ValueError): cli.module.run(cli.args)
    assert not cli.executor.installs and not cli.executor.rgb_calls
    assert not (cli.out / 'attempt-ledger.json').exists()


@pytest.mark.parametrize('timing_key', ['full_upload_bytes', 'sparse_weight_update_bytes', 'materialize_bytes'])
def test_native_diagnostics_reject_any_rgb_state_transfer(case, monkeypatch, timing_key):
    original = case.executor.rgb_step
    def advance(*args, **kwargs):
        counts, elapsed = original(*args, **kwargs)
        case.executor.last_timing[timing_key] = 4
        return counts, elapsed
    monkeypatch.setattr(case.executor, 'rgb_step', advance)
    with pytest.raises(ValueError): wave(case, warmup_ms=0)
    assert not any(game.act_calls for game in case.games)


def test_failed_wave_directory_creation_preserves_charge_and_guarded_evidence(study, monkeypatch):
    module, protocol, candidates, executor, out = study
    def factory(wave, lane, directory):
        if lane == 0: directory.mkdir(parents=True)
        return ShortGame(death=99)
    with pytest.raises(ValueError) as caught:
        module.evaluate(executor, candidates, READOUTS14, protocol, out, game_factory=factory)
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4
    assert not executor.rgb_calls
    assert hasattr(caught.value, '_doomfly_live_availability')
    assert (out / 'waves/w-00/lane-3/failure.npz').exists()


def test_render_interrupt_and_secondary_json_pressure_cleanup_keep_exact_primary(study, monkeypatch):
    module, protocol, candidates, executor, out = study
    primary = KeyboardInterrupt('render')
    game_list = []
    def factory(wave, lane, directory):
        game = ShortGame(death=99)
        if lane == 0:
            def pixels(): raise primary
            game.pixels = pixels
        def close():
            game.close_calls += 1
            raise OSError('secondary game close')
        game.close = close
        game_list.append(game)
        return game
    writer = module.save_json
    def save(path, value):
        if Path(path).name in ('summary.json', 'failure.json', 'interrupted-progress.json'):
            raise OSError('secondary JSON writer')
        return writer(path, value)
    monkeypatch.setattr(module, 'save_json', save)
    pilot = importlib.import_module('doom_learning_v6.causal_pilot')
    reads = []
    def pressure():
        reads.append(1)
        if len(reads) > 1: raise OSError('secondary pressure observation')
        return {'free_percent': 80, 'swap_used_mib': 0}
    monkeypatch.setattr(pilot, '_memory_pressure', pressure)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.evaluate(executor, candidates, READOUTS14, protocol, out, game_factory=factory)
    assert caught.value is primary
    assert len(primary._doomfly_live_availability['lanes']) == 4
    assert all(game.close_calls == 1 for game in game_list)
    assert (out / 'waves/w-00/lane-3/failure.npz').exists()
    assert json.loads((out / 'attempt-ledger.json').read_text())['used'] == 4


def test_game_metrics_use_live_engine_time_and_exclude_padding(case):
    phase = wave(case, warmup_ms=0)
    dead = live_controls.gameplay_metrics(phase['lanes'][0]['trace'], 7)
    survivor = live_controls.gameplay_metrics(phase['lanes'][3]['trace'], 7)
    assert dead == {'live_game_tics': 2, 'restricted_survival_seconds': 2 / 35, 'died': True,
                    'right_censored': False, 'censor_reason': None, 'kills_at_fixed_horizon': 2,
                    'damage_total': 20., 'final_health': 80.}
    assert survivor['right_censored'] is True and survivor['restricted_survival_seconds'] == 7 / 35
    assert survivor['kills_at_fixed_horizon'] == 7


def test_all_finished_brains_continue_on_labeled_dark_padding(case):
    case.games = [ShortGame(death=1) for _ in range(4)]
    phase = wave(case, warmup_ms=0)
    black = digest(np.zeros((480, 640, 3), dtype=np.uint8))
    assert [game.act_calls for game in case.games] == [1] * 4
    assert phase['timing']['rgb_calls'] == 7
    assert [call['phase'] for call in phase['batch_calls']] == ['live'] + ['padding'] * 6
    for lane in phase['lanes']:
        assert all(row['frame_sha256'] == black and row['applied_action'] is None for row in lane['trace'][1:])


def test_terminal_io_and_complete_envelope_clocks_are_durable_and_disjoint(case):
    phase = wave(case, warmup_ms=10)
    timing = json.loads((case.out / 'timing.json').read_text())
    assert timing == phase['timing']
    disjoint = ['game_render_wall_seconds', 'game_act_wall_seconds', 'game_observer_wall_seconds',
                'batch_rgb_wall_seconds', 'trace_record_wall_seconds', 'trace_io_wall_seconds',
                'terminal_materialization_wall_seconds', 'checkpoint_wall_seconds', 'terminal_json_wall_seconds']
    assert timing['terminal_json_wall_seconds'] > 0
    assert sum(timing[name] for name in disjoint) <= timing['wave_envelope_wall_seconds']
    assert timing['envelope_scope'] == 'entry through phase JSON; timing.json write excluded'


@pytest.mark.parametrize('source', ['doom_learning_v6/rule.py', 'doom_learning_v6/calibration.py', 'doom_learning_v6/r8_map.py'])
def test_declared_game_boundary_never_relaxes_rule_calibration_or_sensory(source):
    from doom_learning_v6.same_slot_runner import _reference_source_transitions
    declaration = {name: 'a' * 64 for name in ('doom/game.py', 'doom_learning/survival_arena.py')}
    previous = {**declaration, source: 'b' * 64}
    observed = {'sources': {name: {'sha256': value} for name, value in declaration.items()}}
    observed['sources'][source] = {'sha256': 'c' * 64}
    with pytest.raises(ValueError): _reference_source_transitions(previous, observed, boundary_sources=declaration)


@pytest.mark.parametrize('late_file', ['summary.json', 'progress.json', 'timing.json'])
@pytest.mark.parametrize('persistent', [False, True])
def test_late_phase_json_invalidates_prior_success_without_neural_retries(case, monkeypatch, late_file, persistent):
    module = runner()
    primary = KeyboardInterrupt('late phase writer')
    writer, failures, syncs, checkpoints, subsequent_availability = module.save_json, [], [], [], []
    for lane, brain in enumerate(case.brains):
        sync, cp = brain.backend.sync_for_checkpoint, brain.checkpoint
        def synchronize(lane=lane, sync=sync):
            syncs.append(lane)
            return sync()
        def checkpoint(path, lane=lane, cp=cp):
            checkpoints.append(lane)
            return cp(path)
        monkeypatch.setattr(brain.backend, 'sync_for_checkpoint', synchronize)
        monkeypatch.setattr(brain, 'checkpoint', checkpoint)
    def save(path, value):
        if failures and hasattr(primary, '_doomfly_live_availability'):
            subsequent_availability.append([row['complete'] for row in primary._doomfly_live_availability['lanes']])
        target = Path(path).name == late_file
        final = late_file != 'progress.json' or value.get('complete') is True
        if target and final and not failures:
            failures.append(path)
            raise primary
        if persistent and target and failures: raise OSError('persistent secondary writer')
        return writer(path, value)
    monkeypatch.setattr(module, 'save_json', save)
    with pytest.raises(KeyboardInterrupt) as caught: wave(case, warmup_ms=0)
    assert caught.value is primary
    records = primary._doomfly_live_availability['lanes']
    assert all(row['complete'] is False for row in records)
    assert all(flags == [False] * 4 for flags in subsequent_availability)
    assert all(row['checkpoint_available'] is True and row['frozen_verified'] is True for row in records)
    assert all(row['frozen_proof_scope'] == 'materialized terminal only; overall phase incomplete' for row in records)
    assert checkpoints == [0, 1, 2, 3] and sorted(syncs) == [0, 0, 1, 1, 2, 2, 3, 3]
    for lane in range(4):
        assert json.loads((case.out / f'lane-{lane}/episode.json').read_text())['complete'] is False
    if late_file != 'summary.json' or not persistent:
        assert json.loads((case.out / 'summary.json').read_text())['complete'] is False
    if late_file != 'progress.json' or not persistent:
        assert json.loads((case.out / 'progress.json').read_text())['complete'] is False
    if persistent and late_file != 'timing.json':
        assert any('persistent secondary writer' in note for note in primary.__notes__)


def test_phase_invalidation_writer_failure_preserves_primary_and_in_memory_state(case, monkeypatch):
    module = runner()
    primary = KeyboardInterrupt('timing write')
    writer = module.save_json
    def save(path, value):
        if Path(path).name == 'timing.json': raise primary
        if Path(path) == case.out / 'lane-1/episode.json' and value['complete'] is False:
            raise OSError('secondary lane invalidation')
        return writer(path, value)
    monkeypatch.setattr(module, 'save_json', save)
    with pytest.raises(KeyboardInterrupt) as caught: wave(case, warmup_ms=0)
    assert caught.value is primary
    assert all(row['complete'] is False for row in primary._doomfly_live_availability['lanes'])
    assert json.loads((case.out / 'summary.json').read_text())['complete'] is False
    assert json.loads((case.out / 'lane-0/episode.json').read_text())['complete'] is False
    assert any('secondary lane invalidation' in note for note in primary.__notes__)


@pytest.mark.parametrize('secondary', [False, True])
def test_initial_charged_ledger_failure_has_one_safe_pre_wave_evidence_pass(study, monkeypatch, secondary):
    module, protocol, candidates, executor, out = study
    primary = KeyboardInterrupt('initial charged ledger')
    writer, writes, syncs, checkpoints = module.save_json, [], [], []
    for lane, brain in enumerate(executor.brains):
        sync, cp = brain.backend.sync_for_checkpoint, brain.checkpoint
        def synchronize(lane=lane, sync=sync):
            syncs.append(lane)
            return sync()
        def checkpoint(path, lane=lane, cp=cp):
            checkpoints.append(lane)
            return cp(path)
        monkeypatch.setattr(brain.backend, 'sync_for_checkpoint', synchronize)
        monkeypatch.setattr(brain, 'checkpoint', checkpoint)
    def save(path, value):
        if Path(path).name == 'attempt-ledger.json':
            writes.append(value['used'])
            if len(writes) == 1: raise primary
        if secondary and Path(path).name == 'summary.json': raise OSError('secondary evidence write')
        return writer(path, value)
    monkeypatch.setattr(module, 'save_json', save)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.evaluate(executor, candidates, READOUTS14, protocol, out,
                        game_factory=lambda *args: pytest.fail('game constructed before durable charge'))
    assert caught.value is primary
    ledger = json.loads((out / 'attempt-ledger.json').read_text())
    assert ledger['used'] == 4 and ledger['waves'][0]['state'] == 'failed'
    assert all(row['checkpoint_available'] for row in primary._doomfly_live_availability['lanes'])
    assert checkpoints == [0, 1, 2, 3] and sorted(syncs) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert not executor.installs and not executor.rgb_calls
    assert all(brain.cursor == 0 and brain.backend._host_weight_epoch == 0 for brain in executor.brains)


@pytest.mark.parametrize('main_failure', ['evaluation', 'resource'])
@pytest.mark.parametrize('secondary', [None, 'pressure', 'pins', 'validation', 'writer'])
def test_failed_cli_attempts_after_release_pressure_and_pins_without_new_neural_evidence(cli, monkeypatch, main_failure, secondary):
    module = cli.module
    primary = KeyboardInterrupt(main_failure)
    if main_failure == 'evaluation':
        monkeypatch.setattr(module, '_game_factory', lambda p: lambda *args: ShortGame(failure=primary))
    else:
        def close():
            cli.closed.append('executor')
            raise primary
        cli.executor.close = close
    syncs, cp_calls, pin_calls, pressure_names, events = [], [], [], [], []
    for lane, brain in enumerate(cli.executor.brains):
        sync, cp = brain.backend.sync_for_checkpoint, brain.checkpoint
        def synchronize(lane=lane, sync=sync):
            syncs.append(lane)
            return sync()
        def checkpoint(path, lane=lane, cp=cp):
            cp_calls.append(lane)
            return cp(path)
        monkeypatch.setattr(brain.backend, 'sync_for_checkpoint', synchronize)
        monkeypatch.setattr(brain, 'checkpoint', checkpoint)
    def pins(reference):
        pin_calls.append(reference)
        if len(pin_calls) == 2:
            events.append(list(cli.closed))
            if secondary == 'pins': raise OSError('secondary after physical pins')
            if secondary == 'validation':
                changed = copy.deepcopy(cli.observed)
                changed['sources']['doom/game.py']['sha256'] = 'c' * 64
                return changed
        return copy.deepcopy(cli.observed)
    monkeypatch.setattr(module, '_physical_pins', pins)
    pilot = importlib.import_module('doom_learning_v6.causal_pilot')
    observe = pilot._PressureHistory.observe
    def pressure(self, name):
        pressure_names.append(name)
        if name == 'after_native_release':
            events.append(list(cli.closed))
            if secondary == 'pressure': raise OSError('secondary release pressure')
        return observe(self, name)
    monkeypatch.setattr(pilot._PressureHistory, 'observe', pressure)
    writer = module.save_json
    def save(path, value):
        if secondary == 'writer' and Path(path).name == 'observed-pins-after.json':
            raise OSError('secondary after pins writer')
        return writer(path, value)
    monkeypatch.setattr(module, 'save_json', save)
    with pytest.raises(KeyboardInterrupt) as caught: module.run(cli.args)
    assert caught.value is primary
    assert pressure_names.count('after_native_release') == 1 and len(pin_calls) == 2
    assert events == [['executor', 'brain-3', 'brain-2', 'brain-1', 'brain-0']] * 2
    assert cli.closed == ['executor', 'brain-3', 'brain-2', 'brain-1', 'brain-0', 'unlock', 'lease']
    waves = 1 if main_failure == 'evaluation' else 18
    assert len(syncs) == 4 + waves * 8 and len(cp_calls) == 4 + waves * 4
    release = json.loads((cli.out / 'release-evidence.json').read_text())
    assert release['scope'] == 'after registered resource release attempts'
    if secondary is not None:
        assert release['errors'] and any('Secondary' in note for note in primary.__notes__)
    if secondary in ('pins', 'validation', 'writer'):
        assert (cli.out / 'observed-pins-after-error.json').exists()


def test_unavailable_terminal_never_gets_materialized_frozen_proof_label(case, monkeypatch):
    primary = KeyboardInterrupt('terminal unavailable')
    def synchronize(): raise primary
    monkeypatch.setattr(case.brains[0].backend, 'sync_for_checkpoint', synchronize)
    with pytest.raises(KeyboardInterrupt) as caught: wave(case, warmup_ms=0)
    assert caught.value is primary
    record = primary._doomfly_live_availability['lanes'][0]
    assert record['checkpoint_available'] is False and record['complete'] is False
    assert record['frozen_proof_scope'] == 'frozen proof unavailable; overall phase incomplete'
