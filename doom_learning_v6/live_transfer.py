"""Private, frozen RGB-only transfer evaluation with complete resident evidence."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

from doom.engine import NeuralControls
from doom_learning.common import digest, save_json
from .causal_controls import frame_ticks
from .causal_pilot import (
    _Resources, _phase_boundary, _state_hashes, _physical_pins,
    _validate_expected_pins, _source_clean, _git_output, _acquire_gpu_lock,
    _file_pin, _restored_brains, _model_gate, _readouts,
    _compare_checkpoint_arrays, _same_without_wall,
)
from .imitation import _memory, _invariants
from .live_controls import (
    validate_live_protocol, load_live_protocol, load_candidates, plan_waves,
    gameplay_metrics,
)
from .same_slot_optimizer import AttemptBudget
from .same_slot_runner import _initial_reference, _frozen, _storage, _pressure, _disk, _failed

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_NAME = 'docs/experiments/2026-09-13-metal-live-transfer-protocol.json'
_CHECKPOINT_SCOPE = 'wave-final after complete horizon padding'
_MEMORY_SCOPE = 'host mirrors; terminal materialization is authoritative'


def _readout_contract(readouts, brains):
    if (type(readouts) is not list or len(readouts) != 14 or
            any(type(row) is not dict or type(row.get('index')) is not int for row in readouts) or
            len({row['index'] for row in readouts}) != 14 or
            any(not 0 <= row['index'] < brain.n for row in readouts for brain in brains) or
            any(sum(row.get('type') == kind and row.get('side') == side for row in readouts) != 1
                for kind in ('DNp20', 'DNpe017') for side in ('L', 'R'))):
        raise ValueError('Exact fourteen readouts and four fixed BCI neurons required')


def _lanes(executor, brains=None):
    resident = tuple(executor.brains)
    lanes = resident if brains is None else tuple(brains)
    if (len(lanes) != 4 or len(resident) != 4 or len({id(brain) for brain in lanes}) != 4 or
            any(a is not b for a, b in zip(lanes, resident))):
        raise ValueError('Exactly four distinct ordered resident brains required')
    return lanes


def _epochs(brains):
    result = []
    for brain in brains:
        backend = brain.backend
        pair = [backend._host_weight_epoch, backend._device_weight_epoch]
        if any(type(value) is not int or value < 0 for value in pair) or pair[0] != pair[1]:
            raise ValueError('Resident weight upload epochs disagree')
        result.append(pair)
    return result


def _host_frozen(brain):
    """Call only after successful checked materialization."""
    return {'weight': digest(brain.weight), 'u': digest(brain.memory_u),
            'w': digest(brain.memory_w), 'fixed': _invariants(brain),
            'configuration': brain.configuration_signature()}


def _observe(game):
    """Observer state controls termination only, downstream of neural decoding."""
    raw = game.observation()
    engine = game.game
    finished, dead = bool(engine.is_episode_finished()), bool(engine.is_player_dead())
    engine_tic = int(engine.get_episode_time())
    timeout_tic = int(engine.get_episode_timeout())
    return {'tick': int(raw['tick']), 'engine_tic': engine_tic, 'finished': finished,
            'dead': dead, 'timeout': finished and not dead and timeout_tic > 0 and engine_tic >= timeout_tic,
            'health': float(raw['health']), 'kills': int(raw['kills'])}


def _secondary(primary, error, scope):
    if primary is None:
        return error
    if primary is not error:
        primary.add_note(f'Secondary {scope}: {type(error).__name__}: {error}')
    return primary


def _retain(failure, directory, brains, records):
    failure._doomfly_live_availability = {'directory': str(Path(directory).resolve()),
        'lane_ids': tuple(map(id, brains)), 'lanes': records}


def _terminal_evidence(brains, directory, records, failure, expected, timing):
    """One terminal pass per lane. A failed sync permanently suppresses its CP."""
    directory = Path(directory)
    for lane, (brain, record) in enumerate(zip(brains, records)):
        record.update(terminal_state_available=False, checkpoint_available=False,
                      checkpoint_attempted=False,
                      checkpoint_scope=_CHECKPOINT_SCOPE if record['complete'] else 'interrupted neural prefix; complete padding unavailable')
        materialize = time.perf_counter()
        try:
            record['terminal_state'] = _state_hashes(brain)
            record['terminal_state_available'] = True
            record['frozen_after'] = _host_frozen(brain)
            record['frozen_verified'] = expected is not None and record['frozen_after'] == expected[lane]
            if expected is not None and not record['frozen_verified']:
                raise ValueError('Frozen full weights/u/w or fixed configuration changed')
        except BaseException as error:
            record['terminal_state_error'] = {'type': type(error).__name__}
            failure = _secondary(failure, error, f'lane-{lane} terminal evidence')
            _retain(failure, directory, brains, records)
        finally:
            timing['terminal_materialization_wall_seconds'] += time.perf_counter() - materialize
        # A numerical mismatch still has materialized state worth preserving.
        if record['terminal_state_available']:
            checkpoint_started = time.perf_counter()
            record['checkpoint_attempted'] = True
            path = directory / f'lane-{lane}' / ('failure.npz' if failure else 'final.npz')
            record['checkpoint_path'] = str(path.relative_to(directory))
            try:
                brain.checkpoint(path)
                with np.load(path, allow_pickle=False) as checkpoint:
                    names = {'weight', *brain.fields}
                    if len(names) != 24 or set(checkpoint.files) != names | {'metadata'}:
                        raise ValueError('Wave-final checkpoint must contain all24 arrays')
                    for name in names:
                        array = checkpoint[name]
                        host = getattr(brain, name)
                        if (array.dtype != host.dtype or array.shape != host.shape or
                                digest(array) != record['terminal_state'][name + '_sha256']):
                            raise ValueError(f'Wave-final checkpoint differs from materialized terminal: {name}')
                    if json.loads(str(checkpoint['metadata']))['cursor'] != record['terminal_state']['cursor']:
                        raise ValueError('Wave-final checkpoint cursor differs from materialized terminal')
                record['checkpoint_available'] = True
                record['checkpoint_pin'] = _file_pin(path)
            except BaseException as error:
                record['checkpoint_error'] = {'type': type(error).__name__}
                failure = _secondary(failure, error, f'lane-{lane} checkpoint')
                _retain(failure, directory, brains, records)
            finally:
                timing['checkpoint_wall_seconds'] += time.perf_counter() - checkpoint_started
    if failure is not None:
        _retain(failure, directory, brains, records)
    return failure


def _invalidate_phase(directory, brains, summary, failure):
    """Invalidate metadata only; retain independent terminal/CP availability."""
    records = summary['lanes']
    for record in records:
        scope = ('materialized terminal only; overall phase incomplete'
                 if record.get('terminal_state_available') and record.get('frozen_verified')
                 else 'frozen proof unavailable; overall phase incomplete')
        record.update(complete=False, frozen_proof_scope=scope)
    summary.update(complete=False, frozen_success=False, failure={'type': type(failure).__name__})
    _retain(failure, directory, brains, records)
    writes = [(Path(directory) / f'lane-{lane}/episode.json', record)
              for lane, record in enumerate(records)]
    writes += [(Path(directory) / 'summary.json', summary),
               (Path(directory) / 'progress.json', {'complete': False,
                   'frames': [len(row['trace']) for row in records], 'failure': summary['failure']})]
    for path, value in writes:
        try:
            save_json(path, value)
        except BaseException as error:
            _secondary(failure, error, f'phase metadata invalidation {path.name}')


def _persist_phase(directory, brains, records, failure, timing, batch_calls):
    """Persist metadata only; checkpoint ownership stays in terminal evidence."""
    if failure is not None:
        for record in records:
            record['complete'] = False
    summary = {'lanes': records, 'complete': failure is None and all(row['complete'] for row in records),
               'timing': timing, 'batch_calls': batch_calls}
    for name, calls in [('warmup_timing', [call for call in batch_calls if call['phase'] == 'warmup']),
                        ('rollout_timing', [call for call in batch_calls if call['phase'] != 'warmup'])]:
        shared = {}
        for call in calls:
            for key, value in call['last_timing'].items():
                if type(value) in (int, float) and key not in ('shared_resident_bytes', 'mutable_resident_bytes', 'edge_bitmap_words'):
                    shared[key] = shared.get(key, 0) + value
        summary[name] = {'rgb_calls': len(calls), 'shared_native_timing': shared,
                         'batch_native_seconds': sum(call['native_elapsed_seconds'] for call in calls)}
    io_started = time.perf_counter()
    for lane, record in enumerate(records):
        try:
            save_json(Path(directory) / f'lane-{lane}/episode.json', record)
        except BaseException as error:
            failure = _secondary(failure, error, f'lane-{lane} JSON')
            for item in records:
                item['complete'] = False
            _retain(failure, directory, brains, records)
    if failure is not None:
        for record in records:
            record['complete'] = False
        summary.update(complete=False, failure={'type': type(failure).__name__})
        _retain(failure, directory, brains, records)
    for filename, value in [('summary.json', summary), ('progress.json', {
            'complete': summary['complete'], 'frames': [len(row['trace']) for row in records]})]:
        try:
            save_json(Path(directory) / filename, value)
        except BaseException as error:
            failure = _secondary(failure, error, 'phase JSON')
            for record in records:
                record['complete'] = False
            summary['complete'] = False
            _retain(failure, directory, brains, records)
    if failure is not None:
        _invalidate_phase(directory, brains, summary, failure)
    timing['terminal_json_wall_seconds'] = time.perf_counter() - io_started
    return summary, failure


def live_wave(brains, executor, games, readouts, *, horizon_tics, warmup_ms, directory):
    """Advance all four neural lanes, padding each terminated game independently."""
    started = time.perf_counter()
    lanes = _lanes(executor, brains)
    games = tuple(games)
    if (len(games) != 4 or len({id(game) for game in games}) != 4 or
            any(getattr(game, 'game', None) is None for game in games) or
            len({id(game.game) for game in games}) != 4):
        raise ValueError('Exactly four distinct games and independent engines required')
    if type(horizon_tics) is not int or horizon_tics <= 0:
        raise ValueError('Positive built-in horizon tics required')
    if type(warmup_ms) not in (int, float) or not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError('Finite nonnegative dark warmup required')
    phase = Path(directory)
    if phase.exists():
        raise ValueError('Fresh live phase directory required')
    _readout_contract(readouts, lanes)
    storage = _storage(executor)
    ticks = frame_ticks(horizon_tics)
    controls = [NeuralControls(readouts, mode='bci') for _ in lanes]
    records = [{'trace': [], 'complete': False, 'weights_frozen': True,
                'learning': False, 'reinforcement_current': 0,
                'memory_diagnostic_scope': _MEMORY_SCOPE} for _ in lanes]
    timing = {name: 0. for name in ('setup_reset_wall_seconds', 'dark_warmup_wall_seconds',
        'rgb_loop_wall_seconds', 'game_render_wall_seconds', 'game_act_wall_seconds',
        'game_observer_wall_seconds', 'batch_rgb_wall_seconds', 'batch_native_seconds',
        'trace_record_wall_seconds', 'trace_io_wall_seconds', 'terminal_json_wall_seconds',
        'terminal_materialization_wall_seconds',
        'checkpoint_wall_seconds', 'live_batch_wall_seconds', 'padding_only_batch_wall_seconds')}
    timing['rgb_calls'] = 0
    timing['shared_native_timing'] = {}
    batch_calls, failure, expected = [], None, None
    phase.mkdir(parents=True, mode=0o700)
    setup_epochs = None

    def advance(images, duration, label, index):
        began = time.perf_counter()
        counts, elapsed = executor.rgb_step(images, duration, learning=[False] * 4, stimulations=[None] * 4)
        wall = time.perf_counter() - began
        native = dict(getattr(executor, 'last_timing', {}))
        # Batch timing is shared: count it once, never once per resident lane.
        call = {'phase': label, 'index': index, 'wall_seconds': wall,
                'native_elapsed_seconds': float(elapsed), 'last_timing': native}
        batch_calls.append(call)
        timing['rgb_calls'] += 1
        timing['batch_native_seconds'] += float(elapsed)
        timing['batch_rgb_wall_seconds'] += wall
        for name, value in native.items():
            if type(value) in (int, float) and name not in ('shared_resident_bytes', 'mutable_resident_bytes', 'edge_bitmap_words'):
                timing['shared_native_timing'][name] = timing['shared_native_timing'].get(name, 0) + value
        if label == 'live': timing['live_batch_wall_seconds'] += wall
        if label == 'padding': timing['padding_only_batch_wall_seconds'] += wall
        if native.get('sparse_weight_update_bytes', 0) != 0:
            raise ValueError('Frozen RGB advance uploaded sparse weights')
        if native.get('full_upload_bytes', 0) != 0:
            raise ValueError('Frozen RGB advance uploaded full state')
        if native.get('materialize_bytes', 0) != 0:
            raise ValueError('Frozen RGB advance unexpectedly materialized full state')
        if _epochs(lanes) != setup_epochs or _storage(executor) != storage:
            raise ValueError('Rollout upload epochs or resident storage changed')
        if any(brain.weights_frozen is not True for brain in lanes):
            raise ValueError('Rollout frozen flag changed')
        return counts

    try:
        expected = [_frozen(brain) for brain in lanes]
        installed_epochs = _epochs(lanes)
        retained = [brain.weight[brain.circuit['edges']].copy() for brain in lanes]
        for brain in lanes:
            brain.reset(keep_memory=True)
        for brain, values in zip(lanes, retained):
            brain.weight[brain.circuit['edges']] = values
            brain.backend.update_weights(brain.circuit['edges'], values)
            brain.weights_frozen = True
        setup_epochs = _epochs(lanes)
        if setup_epochs != [[a + 2, b + 2] for a, b in installed_epochs]:
            raise ValueError('Exactly reset plus retained-slot upload required before rollout')
        if [_host_frozen(brain) for brain in lanes] != expected:
            raise ValueError('Setup changed retained frozen candidate')
        timing['setup_reset_wall_seconds'] = time.perf_counter() - started
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        warm = time.perf_counter()
        if warmup_ms:
            advance([black] * 4, warmup_ms, 'warmup', None)
        timing['dark_warmup_wall_seconds'] = time.perf_counter() - warm
        origins = [int(brain.cursor) for brain in lanes]
        if origins != [round(warmup_ms / .1)] * 4:
            raise ValueError('Dark warmup neural boundary differs')
        observer_started = time.perf_counter()
        observations = [_observe(game) for game in games]
        timing['game_observer_wall_seconds'] += time.perf_counter() - observer_started
        for record, observation in zip(records, observations):
            record['initial_observer'] = observation
        for lane, record in enumerate(records):
            record.update(frozen_before=expected[lane], installed_epochs=installed_epochs[lane],
                          setup_epochs=setup_epochs[lane], warmup_steps=origins[lane])
        loop = time.perf_counter()
        for index, steps in enumerate(ticks):
            alive = [not row['finished'] for row in observations]
            render = time.perf_counter()
            images = [game.pixels() if active else black for game, active in zip(games, alive)]
            if any(image.dtype != np.uint8 or image.shape != black.shape for image in images):
                raise ValueError('Live RGB24 640x480 pixels required')
            timing['game_render_wall_seconds'] += time.perf_counter() - render
            counts = advance(images, int(steps) * .1, 'live' if any(alive) else 'padding', index)
            record_started = time.perf_counter()
            requested = [control.decode(count, int(steps) * .0001) for control, count in zip(controls, counts)]
            for lane, (brain, game, active) in enumerate(zip(lanes, games, alive)):
                before = observations[lane]
                row = {'index': index, 'phase': 'live' if active else 'padding',
                    'neural_steps': int(steps), 'cursor': int(brain.cursor),
                    'brain_steps': int(brain.cursor - origins[lane]),
                    'brain_seconds': (brain.cursor - origins[lane]) * .0001,
                    'requested_action': requested[lane], 'applied_action': None,
                    'game_before': before, 'game_after': before,
                    'frame_sha256': digest(images[lane]), 'spikes_sha256': digest(counts[lane]),
                    'readouts': requested[lane]['readouts'], 'memory': _memory(brain),
                    'memory_diagnostic_scope': _MEMORY_SCOPE,
                    'KC_spikes': int(counts[lane][brain.circuit['kc']].sum()),
                    'DAN_spikes': counts[lane][brain.circuit['dan']].tolist(),
                    'MBON_spikes': counts[lane][brain.circuit['mb']].tolist(),
                    'learning': False, 'weights_frozen': True, 'stimulation': None}
                records[lane]['trace'].append(row)
                if brain.cursor - origins[lane] != round((index + 1) * 10000 / 35):
                    raise ValueError('Exact original 285/286 neural boundary differs')
            timing['trace_record_wall_seconds'] += time.perf_counter() - record_started
            for lane, (game, active) in enumerate(zip(games, alive)):
                row = records[lane]['trace'][-1]
                if active:
                    row['applied_action'] = requested[lane]
                    act = time.perf_counter()
                    try:
                        game.act(requested[lane])
                    finally:
                        timing['game_act_wall_seconds'] += time.perf_counter() - act
                    observer = time.perf_counter()
                    try:
                        after = _observe(game)
                    finally:
                        timing['game_observer_wall_seconds'] += time.perf_counter() - observer
                    observations[lane] = after
                    row['game_after'] = after
            io = time.perf_counter()
            save_json(phase / 'progress.json', {'complete': False, 'frames': [len(row['trace']) for row in records],
                                              'private_experiment': True})
            timing['trace_io_wall_seconds'] += time.perf_counter() - io
        timing['rgb_loop_wall_seconds'] = time.perf_counter() - loop
        for record in records:
            record['complete'] = len(record['trace']) == horizon_tics
    except BaseException as error:
        failure = error
        _retain(failure, phase, lanes, records)
    failure = _terminal_evidence(lanes, phase, records, failure, expected, timing)
    timing['wave_envelope_wall_seconds'] = time.perf_counter() - started
    for record in records:
        record['resident_bytes'] = storage
    summary, failure = _persist_phase(phase, lanes, records, failure, timing, batch_calls)
    timing.update(wave_envelope_wall_seconds=time.perf_counter() - started,
                  envelope_scope='entry through phase JSON; timing.json write excluded')
    try:
        save_json(phase / 'timing.json', timing)
    except BaseException as error:
        failure = _secondary(failure, error, 'terminal timing JSON')
        _invalidate_phase(phase, lanes, summary, failure)
    if failure is not None:
        raise failure
    return summary


def _candidate_contract(candidates, protocol):
    roles = {row['role'] for row in protocol['candidates']}
    if type(candidates) is not dict or set(candidates) not in (roles, roles | {'baseline-filler'}):
        raise ValueError('All seven immutable candidates required')
    for row in protocol['candidates']:
        value = candidates[row['role']]
        if (type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != (4184,) or
                value.flags.writeable or not value.flags.c_contiguous or not np.isfinite(value).all() or
                np.any(value < .9) or np.any(value > 1.1)):
            raise ValueError('Frozen contiguous float64 4184-slot candidate required')
        raw = BytesIO()
        np.save(raw, value, allow_pickle=False)
        data = raw.getvalue()
        if {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} != row['pin']:
            raise ValueError('Candidate vector differs from immutable file pin')
    if not np.array_equal(candidates['baseline'], np.ones(4184)):
        raise ValueError('Original baseline fractions must equal one')
    if 'baseline-filler' in candidates and candidates['baseline-filler'] is not candidates['baseline']:
        raise ValueError('Filler must retain the original baseline vector')


def evaluate(executor, candidates, readouts, protocol, out, *, game_factory):
    """Run eighteen fixed waves, retaining every role without fitting or selection."""
    validate_live_protocol(protocol)
    schedule = plan_waves(protocol)
    brains = _lanes(executor)
    _readout_contract(readouts, brains)
    _candidate_contract(candidates, protocol)
    _storage(executor)
    if type(executor.window_ticks) is not int or executor.window_ticks != 18 or not callable(game_factory):
        raise ValueError('Four-lane W18 executor and callable game factory required')
    out = Path(out)
    if any((out / name).exists() for name in ('attempt-ledger.json', 'results.json', 'waves')):
        raise ValueError('Fresh live evaluation output required')
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    budget = AttemptBudget(protocol['budget']['maximum'])
    waves, results, fixed = [], [], [_frozen(brain) for brain in brains]
    pressure = _pressure(out)
    def ledger():
        save_json(out / 'attempt-ledger.json', {'maximum': budget.maximum, 'used': budget.used,
                                              'remaining': budget.remaining, 'waves': waves})
    failure = None
    try:
        for planned in schedule:
            _disk(out, protocol['minimum_disk_gib'])
            directory = out / 'waves' / f"w-{planned['ordinal']:02d}"
            entry = {**planned, 'state': 'charged', 'directory': str(directory.relative_to(out)),
                     'used': budget.charge(4, reserve_after=protocol['budget']['reserved'])}
            waves.append(entry)
            phase = None
            try:
                ledger()  # Durable four-lane charge precedes installation/reset/rollout.
                with _phase_boundary(pressure, directory.name), _Resources() as resources:
                    storage = _storage(executor)
                    for lane, role in enumerate(planned['roles']):
                        vector = candidates['baseline' if role == 'baseline-filler' else role]
                        executor.install_efficacies(lane, vector, brains[lane].circuit['edges'])
                    installed = [_frozen(brain) for brain in brains]
                    for lane, (brain, role, snapshot) in enumerate(zip(brains, planned['roles'], installed)):
                        vector = candidates['baseline' if role == 'baseline-filler' else role]
                        if (snapshot['u'] != digest(vector - 1.) or snapshot['w'] != digest(vector - 1.) or
                                digest(brain.weight[brain.circuit['edges']]) != digest(
                                    (brain.baseline_plastic.astype(np.float64) * vector).astype(np.float32)) or
                                any(snapshot[key] != fixed[lane][key] for key in ('fixed', 'configuration'))):
                            raise ValueError('Candidate installation changed fixed model or fractions')
                    games = []
                    for lane in range(4):
                        game = game_factory(planned, lane, directory)
                        if any(game is previous for previous in games):
                            raise ValueError('Game factory returned duplicate lane ownership')
                        games.append(resources.add(game))
                    entry['assets'] = [getattr(game, 'assets', None) for game in games]
                    phase = live_wave(brains, executor, games, readouts, horizon_tics=protocol['horizon_tics'],
                                      warmup_ms=protocol['warmup_ms'], directory=directory)
                    if not phase['complete'] or _storage(executor) != storage:
                        raise ValueError('Incomplete wave or changed resident storage')
                    entry.update(state='complete', complete=True, timing=phase['timing'],
                                 runtime_frozen=[record['frozen_verified'] for record in phase['lanes']],
                                 setup_epochs=[record['setup_epochs'] for record in phase['lanes']], resident_bytes=storage)
                    if planned['kind'] == 'calibration':
                        canonical = phase['lanes'][0]['trace']
                        for lane in range(1, 4):
                            if not _same_without_wall(canonical, phase['lanes'][lane]['trace']):
                                raise ValueError('Baseline canonical live trace differs')
                            _compare_checkpoint_arrays(directory / 'lane-0/final.npz', directory / f'lane-{lane}/final.npz')
                        entry['calibration_verified'] = True
                    else:
                        for lane, role in enumerate(planned['roles']):
                            metrics = gameplay_metrics(phase['lanes'][lane]['trace'], protocol['horizon_tics'])
                            results.append({'environment': planned['environment'], 'seed': planned['seed'],
                                            'role': role, 'lane': lane, 'wave': planned['ordinal'], 'metrics': metrics})
                    ledger()
            except BaseException as error:
                entry.update(state='failed', complete=False, failure_type=type(error).__name__)
                availability = getattr(error, '_doomfly_live_availability', None)
                owns_terminal = (availability is not None and availability['directory'] == str(directory.resolve()) and
                                 availability['lane_ids'] == tuple(map(id, brains)))
                if phase is None and not owns_terminal:
                    records = [{'trace': [], 'complete': False, 'frozen_verified': False} for _ in brains]
                    timing = {'terminal_materialization_wall_seconds': 0., 'checkpoint_wall_seconds': 0.}
                    error = _terminal_evidence(brains, directory, records, error, None, timing)
                    _persist_phase(directory, brains, records, error, timing, [])
                raise error
        result = {'complete': True, 'attempts': budget.used, 'remaining': budget.remaining,
                  'maximum': budget.maximum, 'waves': waves, 'role_results': results,
                  **protocol['claim_flags'], 'private_experiment': True}
        save_json(out / 'results.json', result)
        return result
    except BaseException as error:
        failure = error
        for function in (ledger, lambda: save_json(out / 'interrupted-progress.json', {
                'complete': False, 'attempts': budget.used, 'role_results': results}),
                lambda: _failed(out, error, pressure)):
            try:
                function()
            except BaseException as secondary:
                _secondary(error, secondary, 'evaluation failure metadata')
        raise failure


def _engine_pins(protocol):
    import vizdoom as vzd
    root = Path(vzd.__file__).parent
    paths = {'vizdoom': root / 'vizdoom', 'freedoom2.wad': root / 'freedoom2.wad',
             'defend_the_center.cfg': Path(vzd.scenarios_path) / 'defend_the_center.cfg',
             'defend_the_center.wad': Path(vzd.scenarios_path) / 'defend_the_center.wad',
             'vizdoom.cpython-311-darwin.so': Path(vzd.vizdoom.__file__)}
    observed = {name: _file_pin(path) for name, path in paths.items()}
    if vzd.__version__ != '1.3.0' or observed != protocol['game_assets']:
        raise ValueError('Exact ViZDoom 1.3.0 engine/API/IWAD/scenario assets required')
    return observed


def _game_factory(protocol):
    def create(wave, lane, directory):
        if wave['environment'] == 'blue-floor-survival-v1':
            from doom_learning.survival_arena import SurvivalArena
            # Keep maps separate from the fresh live phase directory.
            path = directory.parent / 'maps' / directory.name / f'lane-{lane}.wad'
            return SurvivalArena(path, seed=wave['seed'], seconds=protocol['horizon_tics'] / 35,
                                 hazard_left=wave['hazard_left'])
        from doom.game import Game
        return Game(seed=wave['seed'], scenario='defend_the_center',
                    episode_timeout_tics=protocol['horizon_tics'], episode_start_tics=protocol['episode_start_tics'])
    return create


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'protocol-sha256', 'reference', 'expected-pins', 'source-commit', 'candidates-root', 'out'):
        parser.add_argument('--' + name, required=True)
    return parser


def _after_release(args, out, pressure, failure):
    """Collect release-boundary evidence independently after all release attempts."""
    errors = []
    def attempt(stage, function):
        nonlocal failure
        try:
            return function(), True
        except BaseException as error:
            errors.append({'stage': stage, 'type': type(error).__name__, 'message': str(error)})
            failure = _secondary(failure, error, 'release evidence ' + stage)
            return None, False
    restored_pressure, available = attempt('pressure_reload', lambda: _pressure(out))
    if available:
        pressure = restored_pressure
    attempt('pressure_observe', lambda: pressure.observe('after_native_release'))
    after, physical_available = attempt('physical_pins', lambda: _physical_pins(args.reference))
    if physical_available:
        attempt('observed_pins_write', lambda: save_json(Path(out) / 'observed-pins-after.json', after))
        attempt('expected_pins_validation', lambda: _validate_expected_pins(args.expected_pins, after))
    pin_errors = [row for row in errors if row['stage'] in
                  ('physical_pins', 'observed_pins_write', 'expected_pins_validation')]
    if pin_errors:
        attempt('pin_error_write', lambda: save_json(Path(out) / 'observed-pins-after-error.json', {
            'physical_available': physical_available, 'errors': pin_errors}))
    attempt('release_summary_write', lambda: save_json(Path(out) / 'release-evidence.json', {
        'scope': 'after registered resource release attempts', 'errors': errors,
        'physical_available': physical_available}))
    return pressure, failure


def run(args):
    """Validate independent physical pins before ownership, then release one lease."""
    out = Path(args.out)
    if out.exists():
        raise ValueError('Fresh private output required')
    if sys.platform != 'darwin' or platform.system() != 'Darwin':
        raise ValueError('Frozen live Metal evaluation requires macOS')
    if type(args.source_commit) is not str or not re.fullmatch(r'[0-9a-f]{40}', args.source_commit) or _git_output('rev-parse', 'HEAD') != args.source_commit:
        raise ValueError('Exact source HEAD required')
    try:
        out.resolve().relative_to(_ROOT.resolve())
    except ValueError:
        pass
    else:
        try:
            ignored = _git_output('check-ignore', '--', str(out.resolve()))
        except Exception as error:
            raise ValueError('Private in-repository output must be ignored') from error
        if not ignored:
            raise ValueError('Private in-repository output must be ignored')
    _source_clean()
    protocol = load_live_protocol(args.protocol, args.protocol_sha256)
    committed = _git_output('show', args.source_commit + ':' + _PROTOCOL_NAME)
    if hashlib.sha256((committed + '\n').encode()).hexdigest() != args.protocol_sha256:
        raise ValueError('Protocol must match exact committed source bytes')
    assets = _engine_pins(protocol)
    candidates = load_candidates(protocol, args.candidates_root)
    observed = _physical_pins(args.reference)
    expected = _validate_expected_pins(args.expected_pins, observed)
    declaration = {name: expected['sources'][name]['sha256'] for name in ('doom/game.py', 'doom_learning/survival_arena.py')}
    relationships = _initial_reference(args.reference, observed, boundary_sources=declaration)
    readouts = _readouts()
    if readouts != relationships['protocol']['readouts']:
        raise ValueError('Exact ordered original fourteen readouts required')
    _disk(out.parent, protocol['minimum_disk_gib'])
    pressure = _pressure(out)
    lock = _acquire_gpu_lock()
    failure, result = None, None
    try:
        out.mkdir(parents=True, mode=0o700)
        for filename, value in [('expected-pins.json', expected), ('observed-pins.json', observed),
                ('reference-validation.json', relationships), ('engine-pins.json', assets),
                ('provenance.json', {'arguments': vars(args), 'source_commit': args.source_commit,
                                     'protocol_pin': _file_pin(args.protocol), 'pid': os.getpid()})]:
            save_json(out / filename, value)
        pressure.observe('before_native_allocation')
        from .calibration import calibrated_brain
        from .metal.batch import MetalBatchExecutor
        try:
            with _Resources() as resources:
                brains = _restored_brains(resources, lambda: calibrated_brain(.001, backend='cpu'),
                                          Path(args.reference) / 'initial.npz', 4)
                _model_gate(brains, relationships['protocol'])
                _readout_contract(readouts, brains)
                for lane, brain in enumerate(brains):
                    checkpoint = out / 'initial' / f'lane-{lane}.npz'
                    brain.checkpoint(checkpoint)
                    _compare_checkpoint_arrays(Path(args.reference) / 'initial.npz', checkpoint)
                executor = resources.add(MetalBatchExecutor(brains, window_ticks=18))
                result = evaluate(executor, candidates, readouts, protocol, out, game_factory=_game_factory(protocol))
        except BaseException as error:
            failure = error
        finally:
            pressure, failure = _after_release(args, out, pressure, failure)
        if failure is None:
            result.update(source_commit=args.source_commit, physical_identity_records=[
                'expected-pins.json', 'observed-pins.json', 'observed-pins-after.json', 'reference-validation.json'])
            save_json(out / 'results.json', result)
    except BaseException as error:
        failure = _secondary(failure, error, 'CLI orchestration')
    finally:
        for cleanup in (lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN), lock.close):
            try:
                cleanup()
            except BaseException as error:
                failure = _secondary(failure, error, 'lease cleanup')
        if failure is not None:
            if result is not None:
                result['complete'] = False
            if out.exists():
                try:
                    _failed(out, failure, pressure)  # Metadata only, no repeated CP/materialization.
                except BaseException as error:
                    _secondary(failure, error, 'CLI failure metadata')
            raise failure
    return result


def main():
    from doom_learning.common import require_single_blas_thread
    require_single_blas_thread()
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
