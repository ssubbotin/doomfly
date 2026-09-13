"""Resident Metal replay for the controlled causal learning pilot.

Pixels, neural propagation, the centered plasticity rule, and fixed BCI
decoding retain their existing implementations.  This runner records their
separate inputs and outputs without using game telemetry as an action policy.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


_ROOT = Path(__file__).resolve().parents[1]
_LOCK_ROOT = _ROOT / 'outputs/doom-learning/causal-pilot-20260913'
_SOURCE_ROOTS = ('doom', 'doom_learning', 'doom_learning_v2', 'doom_learning_v4',
                 'doom_learning_v5', 'doom_learning_v6', 'tests')
_INPUT_ROLES = ('outputs/doom/malecns_v1/graph.npz',
                'connectome_data/malecns_v1/annotations.feather',
                'connectome_data/malecns_v1/normalized/neurons.feather')
_REFERENCE_ROLES = ('results.json', 'protocol.json', 'provenance.json', 'initial.npz',
                    'plastic/learned.npz', 'frozen/learned.npz', 'shifted/learned.npz',
                    'plastic/train-0-0/episode.json', 'plastic/eval-0/episode.json',
                    'plastic/retention-0/episode.json', 'plastic/erased-0/episode.json',
                    'frozen/train-0-0/episode.json', 'frozen/eval-0/episode.json',
                    'shifted/train-0-0/episode.json', 'shifted/eval-0/episode.json')
_CPU_ROLES = {
    'outputs/doom-learning/libmemory.dylib.json': ('doom_learning/kernel.cpp', 'gamma1-eligibility-ltd-v1'),
    **{f'outputs/doom-learning/physiology-v{version}/libmemory.dylib.json':
       (f'doom_learning_v{version}/kernel.cpp', model) for version, model in
       [(2, 'all-modulators-separated-v2'), (4, 'kc-adaptive-lif-v4'),
        (5, 'centered-antihebbian-v5'), (6, 'adaptive-centered-v6')]}}
_NATIVE_ROLES = ('build.json', 'kernels.air', 'kernels.metallib', 'libmemory-metal.dylib')


class _Resources:
    """Register ownership immediately; preserve the primary failure on cleanup."""
    def __init__(self):
        self.resources = []

    def add(self, resource):
        self.resources.append(resource)
        return resource

    def brain(self, factory):
        brain = factory()
        # Keep the original CPU backend too: attachment changes brain.backend.
        self.add(brain.backend)
        return brain

    def __enter__(self):
        return self

    def __exit__(self, kind, failure, traceback):
        first = failure
        for resource in reversed(self.resources):
            try:
                resource.close()
            except BaseException as secondary:
                if first is None:
                    first = secondary
                else:
                    first.add_note(f'Secondary cleanup failure: {type(secondary).__name__}: {secondary}')
        if failure is None and first is not None:
            raise first
        return False


class _PressureHistory:
    """Persist every observation, including rejected readings and phase gaps."""
    def __init__(self, out):
        self.out, self.observations, self.previous = Path(out), [], None

    def observe(self, name):
        from doom_learning.common import save_json
        value = _memory_pressure()
        reason = None
        if (not all(isinstance(value.get(k), (int, float)) and math.isfinite(value[k])
                    for k in ('free_percent', 'swap_used_mib')) or
                not 0 <= value['free_percent'] <= 100 or value['swap_used_mib'] < 0):
            reason = 'Invalid memory pressure reading'
        elif value['free_percent'] < 25:
            reason = 'Insufficient free memory for causal pilot'
        elif self.previous is not None and value['swap_used_mib'] > self.previous['swap_used_mib']:
            reason = f'Swap grew during {name}'
        self.observations.append({'name': name, 'value': value, 'accepted': reason is None,
                                  'rejection': reason, 'wall_timestamp': time.time()})
        rejection = ValueError(reason) if reason else None
        try:
            save_json(self.out / 'pressure.json', {'observations': self.observations})
        except Exception as secondary:
            if rejection is None:
                raise
            rejection.add_note(f'Secondary pressure writer failure: {type(secondary).__name__}: {secondary}')
        if rejection is not None:
            raise rejection
        self.previous = value
        return value


@contextmanager
def _phase_boundary(pressure, name):
    pressure.observe('before_' + name)
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            pressure.observe('after_' + name)
        except BaseException as secondary:
            if failure is None:
                raise
            failure.add_note(f'Secondary phase boundary failure: {type(secondary).__name__}: {secondary}')


def _snapshot(brains, protocol):
    return {'graph': _model_gate(brains, protocol),
            'configuration': json.loads(json.dumps([brain.configuration_signature() for brain in brains]))}


def _capture_snapshot(brains, protocol, out, name):
    from doom_learning.common import save_json
    value = _snapshot(brains, protocol)
    save_json(Path(out) / 'invariants' / (name + '.json'), value)
    return value


def _restored_brains(resources, factory, initial, count, memories=None):
    from doom_learning.common import digest
    from .causal_controls import initialize_efficacies
    brains = []
    for index in range(count):
        brain = resources.brain(factory)
        brains.append(brain)
        brain.restore(initial)
        if digest(brain.weight) != brain.configuration_signature()['initial_weight']:
            raise ValueError('Restored initial weights differ from original configuration')
        if memories is not None:
            initialize_efficacies(brain, memories[index])
    return brains


def _flags(value, count, name):
    """Require one built-in boolean flag for every resident lane."""
    if not isinstance(value, (list, tuple)) or len(value) != count or any(type(item) is not bool for item in value):
        raise ValueError(f'{name} must contain one strict boolean per lane')
    return list(value)


def _phase_inputs(brains, executor, data, readouts, currents, learning, frozen, directory, warmup_ms):
    """Perform every check that can precede a reset or iterator acquisition."""
    import numpy as np
    from doom.engine import NeuralControls
    from .causal_controls import frame_ticks

    lanes = tuple(brains)
    if not lanes or tuple(getattr(executor, 'brains', ())) != lanes:
        raise ValueError('Executor brains must have identical lane identity and order')
    if any(left is not right for left, right in zip(tuple(executor.brains), lanes)):
        raise ValueError('Executor brains must have identical lane identity and order')
    try:
        count = data.frame_count
        ticks = frame_ticks(count)
        height, width = data.height, data.width
    except (AttributeError, ValueError, TypeError) as error:
        raise ValueError('Original positive frame count and RGB dimensions required') from error
    if (type(height) is not int or type(width) is not int or height <= 0 or width <= 0):
        raise ValueError('Original positive frame count and RGB dimensions required')
    if not isinstance(warmup_ms, (int, float)) or isinstance(warmup_ms, bool) or not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError('Finite nonnegative warmup required')
    learn = _flags(learning, len(lanes), 'learning')
    freeze = _flags(frozen, len(lanes), 'frozen')
    try:
        schedule = np.asarray(currents, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Finite lane-by-original-frame current schedule required') from error
    if schedule.shape != (len(lanes), int(count)) or not np.isfinite(schedule).all() or np.any(schedule < 0) or np.any(schedule > 4):
        raise ValueError('Finite lane-by-original-frame current schedule required')
    try:
        targets = np.asarray(data.turn_targets(), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Original target vector required') from error
    if targets.shape != (int(count),) or not np.isfinite(targets).all():
        raise ValueError('Original target vector required')
    phase = Path(directory)
    if phase.exists():
        raise ValueError('Fresh phase directory required')
    # Decoder construction validates the fixed interface before state changes.
    controls = [NeuralControls(readouts, mode='bci') for _ in lanes]
    return lanes, ticks, int(height), int(width), learn, freeze, schedule, targets, phase, controls


def _state_hashes(brain):
    from doom_learning.common import digest
    return {'cursor': int(brain.cursor), 'total_spikes': int(brain.total_spikes),
            **{name + '_sha256': digest(getattr(brain, name)) for name in ['weight', *brain.fields]}}


def _lane_record(brain, data, frozen, before, trace, origin, started, warmup, failure=None, close_failure=None,
                 timing=None):
    from .causal_controls import diagnostic_score
    from .imitation import _memory
    result = {
        'identity': dict(data.identity), 'weights_frozen': bool(frozen),
        'source_frame_count': int(data.frame_count), 'frames': len(trace),
        'complete': failure is None and len(trace) == data.frame_count,
        'brain_steps': int(brain.cursor - origin), 'brain_seconds': (brain.cursor - origin) * .0001,
        'recorded_seconds': len(trace) / 35, 'wall_seconds': time.perf_counter() - started,
        'warmup': warmup, 'before': before, 'after': _memory(brain),
        'phase_timing': timing,
        'terminal_state': _state_hashes(brain), 'trace': trace,
        'diagnostic_score': diagnostic_score([row['action']['turn'] for row in trace],
                                             [row['target_turn'] for row in trace]) if trace else None,
    }
    if failure is not None:
        result['failure'] = {'type': type(failure).__name__}
        if close_failure is not None:
            result['failure']['close_type'] = type(close_failure).__name__
    return result


def _write_phase(phase, lanes, records, *, failure=None, resident_bytes=None):
    from doom_learning.common import save_json
    original = failure
    def attempt(function, *args):
        nonlocal failure
        try:
            function(*args)
        except BaseException as secondary:
            if failure is None:
                failure = secondary
            else:
                failure.add_note(f'Secondary phase evidence failure: {type(secondary).__name__}: {secondary}')
    for index, (brain, record) in enumerate(zip(lanes, records)):
        lane = phase / f'lane-{index}'
        attempt(save_json, lane / 'episode.json', record)
        attempt(brain.checkpoint, lane / ('failure.npz' if failure is not None else 'final.npz'))
    summary = {'lanes': records, 'complete': failure is None and all(row['complete'] for row in records),
               'resident_bytes': resident_bytes}
    if failure is not None:
        summary['failure'] = {'type': type(failure).__name__, 'secondary_notes': list(getattr(failure, '__notes__', []))}
    attempt(save_json, phase / 'summary.json', summary)
    if failure is not None:
        summary['complete'] = False
        summary['failure'] = {'type': type(failure).__name__, 'secondary_notes': list(getattr(failure, '__notes__', []))}
    attempt(save_json, phase / 'progress.json', {'complete': summary['complete'],
                                        'frames': [len(row['trace']) for row in records],
                                        **({'failure': summary['failure']} if failure is not None else {})})
    if failure is not None and original is None:
        raise failure
    return summary


def replay_episode(brains, executor, data, readouts, currents, *, learning, frozen, directory, warmup_ms=2000):
    """Replay every original RGB frame through one already-resident executor.

    Validation completes before lane reset.  Any failure writes only evidence
    already produced, then raises that original exception unchanged. Recoverable
    process interruptions use this path; forced termination cannot recover state.
    """
    import numpy as np
    from doom_learning.common import digest
    from .imitation import _memory, teacher_error

    started = time.perf_counter()
    lanes, ticks, height, width, learn, freeze, schedule, targets, phase, controls = _phase_inputs(
        brains, executor, data, readouts, currents, learning, frozen, directory, warmup_ms)
    phase.mkdir(parents=True)
    try:
        metadata = executor.metadata()
        resident_bytes = {name: metadata[name] for name in ('shared_resident_bytes', 'mutable_resident_bytes')}
    except Exception:
        resident_bytes = None
    timing = {'setup_reset_wall_seconds': 0., 'dark_warmup_wall_seconds': 0., 'rgb_loop_wall_seconds': 0.}
    warmup_started = rgb_started = rgb_finished = None
    traces = [[] for _ in lanes]
    frames = None
    failure = close_failure = None
    origins = [0 for _ in lanes]
    before = [{} for _ in lanes]
    warmup = {'requested_ms': float(warmup_ms), 'brain_steps': 0, 'brain_seconds': 0.,
              'wall_seconds': 0., 'actual_wall_seconds': 0.}
    frozen_state = None
    try:
        # reset(keep_memory=True) retains u/w but restores the initial weight
        # array.  Re-upload the already retained plastic slots so a frozen
        # learned evaluation is a real evaluation of its saved state.
        retained_plastic = [brain.weight[brain.circuit['edges']].copy() for brain in lanes]
        for brain, value in zip(lanes, freeze):
            brain.reset(keep_memory=True)
        for brain, values in zip(lanes, retained_plastic):
            brain.weight[brain.circuit['edges']] = values
            brain.backend.update_weights(brain.circuit['edges'], values)
        for brain, value in zip(lanes, freeze):
            brain.weights_frozen = value
        black = np.zeros((height, width, 3), dtype=np.uint8)
        warmup_started = time.perf_counter()
        timing['setup_reset_wall_seconds'] = warmup_started - started
        warmup_cursor = [brain.cursor for brain in lanes]
        try:
            if warmup_ms:
                executor.rgb_step([black] * len(lanes), warmup_ms, learning=False)
        finally:
            timing['dark_warmup_wall_seconds'] = time.perf_counter() - warmup_started
        warmup['wall_seconds'] = timing['dark_warmup_wall_seconds']
        warmup['actual_wall_seconds'] = warmup['wall_seconds']
        warmup['brain_steps'] = [int(brain.cursor - cursor) for brain, cursor in zip(lanes, warmup_cursor)]
        warmup['brain_seconds'] = [steps * .0001 for steps in warmup['brain_steps']]
        origins = [int(brain.cursor) for brain in lanes]
        before = [_memory(brain) for brain in lanes]
        frozen_state = [(brain.memory_u.copy(), brain.memory_w.copy(),
                         brain.weight[brain.circuit['edges']].copy()) if value else None
                        for brain, value in zip(lanes, freeze)]
        rgb_started = time.perf_counter()
        frames = data.iter_frames()
        for sample in frames:
            index = sample.index
            if index != len(traces[0]) or index >= data.frame_count:
                raise ValueError('Original contiguous frames required')
            image = np.asarray(sample.rgb)
            if image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise ValueError('Original uint8 RGB dimensions required')
            count = int(ticks[index])
            pulses = [(brain.circuit['dan'], float(schedule[lane, index]))
                      if schedule[lane, index] else None for lane, brain in enumerate(lanes)]
            counts, elapsed = executor.rgb_step([image] * len(lanes), count * .1,
                                                learning=learn, stimulations=pulses)
            if any(brain.cursor - origin != int(ticks[:index + 1].sum())
                   for brain, origin in zip(lanes, origins)):
                raise ValueError('Original neural boundary mismatch')
            for lane, brain in enumerate(lanes):
                action = controls[lane].decode(counts[lane], count * .0001)
                imposed = float(schedule[lane, index])
                target = float(targets[index])
                traces[lane].append({
                    'index': int(index), 'timestamp': float(sample.timestamp),
                    'raw_action': np.asarray(sample.action).tolist(), 'teacher_index': int(index),
                    'target_turn': target, 'teacher_error': teacher_error(action['turn'], target),
                    'teacher_current': imposed, 'teacher_float32_current': float(np.float32(imposed)),
                    'teacher_dose_current_ms': imposed * count * .1, 'neural_steps': count,
                    'brain_steps': brain.cursor - origins[lane],
                    'brain_seconds': (brain.cursor - origins[lane]) * .0001,
                    'action': action, 'KC_spikes': int(counts[lane][brain.circuit['kc']].sum()),
                    'DAN_spikes': counts[lane][brain.circuit['dan']].tolist(),
                    'MBON_spikes': counts[lane][brain.circuit['mb']].tolist(),
                    'frame_sha256': digest(image), 'spikes_sha256': digest(counts[lane]),
                    'memory': _memory(brain), 'native_elapsed_seconds': float(elapsed),
                })
            from doom_learning.common import save_json
            save_json(phase / 'progress.json', {'complete': False,
                                                'decoded_frames': [len(trace) for trace in traces],
                                                'resident_bytes': resident_bytes})
        rgb_finished = time.perf_counter()
        if any(len(trace) != data.frame_count for trace in traces):
            raise ValueError('Incomplete original episode')
        for brain, value, expected in zip(lanes, freeze, frozen_state):
            if value and any(not np.array_equal(actual, wanted) for actual, wanted in zip(
                    (brain.memory_u, brain.memory_w, brain.weight[brain.circuit['edges']]), expected)):
                raise ValueError('Frozen plastic state changed')
    except BaseException as error:
        failure = error
    finally:
        if warmup_started is None:
            timing['setup_reset_wall_seconds'] = time.perf_counter() - started
        if rgb_started is not None:
            timing['rgb_loop_wall_seconds'] = (rgb_finished if rgb_finished is not None else time.perf_counter()) - rgb_started
        if frames is not None and hasattr(frames, 'close'):
            try:
                frames.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    close_failure = error
                    failure.add_note(f'Secondary iterator cleanup failure: {type(error).__name__}: {error}')
    try:
        records = [_lane_record(brain, data, value, item_before, trace, origin, started, warmup,
                                failure, close_failure, timing)
                   for brain, value, item_before, trace, origin in zip(lanes, freeze, before, traces, origins)]
    except BaseException as evidence_failure:
        if failure is None:
            raise
        failure.add_note(f'Secondary record failure: {type(evidence_failure).__name__}')
        raise failure
    try:
        summary = _write_phase(phase, lanes, records, failure=failure, resident_bytes=resident_bytes)
    except BaseException as evidence_failure:
        if failure is None:
            raise
        failure.add_note(f'Secondary evidence failure: {type(evidence_failure).__name__}')
        raise failure
    if failure is not None:
        raise failure
    return summary


def _git_output(*arguments):
    return subprocess.run(['git', *arguments], cwd=_ROOT, text=True, capture_output=True, check=True).stdout.strip()


def _source_clean():
    dirty = _git_output('status', '--porcelain', '--', *_SOURCE_ROOTS)
    if dirty:
        raise ValueError('Tracked or untracked model/test sources must be clean')


def _acquire_gpu_lock():
    _LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    stream = (_LOCK_ROOT / 'causal-pilot-gpu.lock').open('a+', encoding='utf-8')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise ValueError('Causal pilot GPU lock is already held') from None
    stream.seek(0); stream.truncate(); stream.write(json.dumps({'pid': os.getpid()}) + '\n'); stream.flush()
    return stream


def _memory_pressure():
    from .metal.benchmark import _memory_status
    return _memory_status()


def _model_gate(brains, protocol=None):
    """Validate the full retained graph and materialize immutable evidence."""
    from .imitation import _invariants
    records = []
    for brain in brains:
        brain.backend.materialize('causal-pilot-invariants')
        edges = brain.circuit['edges']
        if (brain.n, len(brain.weight), len(edges), len(brain.circuit['dan'])) != (166700, 25582938, 4184, 2):
            raise ValueError('Exact retained graph, plastic slots and two DAN neurons required')
        import numpy as np
        if (len(np.unique(edges)) != 4184 or not np.isfinite(brain.baseline_plastic).all()
                or not (brain.baseline_plastic > 0).all()):
            raise ValueError('Positive identified plastic baseline required')
        values = brain.weight[edges] / brain.baseline_plastic
        if not np.isfinite(values).all() or not (values >= .1).all() or not (values <= 2).all():
            raise ValueError('Plastic efficacies exceed declared baseline-relative bounds')
        record = _invariants(brain)
        if protocol is not None:
            identity = protocol['model_identity']
            for actual, expected in [('ids_sha256', 'ids_sha256'), ('ptr_sha256', 'out_ptr_sha256'),
                                     ('post_sha256', 'out_post_sha256'),
                                     ('plastic_slots_sha256', 'plastic_edges_sha256')]:
                if record[actual] != identity[expected]:
                    raise ValueError('Original graph/plastic identity differs')
            if brain.configuration_signature() != protocol['configuration']:
                raise ValueError('Original model configuration differs')
            if brain.calibration != protocol['calibration']:
                raise ValueError('Original calibration differs')
        records.append(record)
    if any(record != records[0] for record in records[1:]):
        raise ValueError('Fixed invariants differ between lanes')
    return records[0]


def _file_pin(path):
    from .metal.benchmark import _file_digest
    value = Path(path)
    if not value.is_file():
        raise ValueError(f'Required physical input missing: {value.name}')
    return {'bytes': value.stat().st_size, 'sha256': _file_digest(value)}


def _physical_pins(reference):
    """Validate physical build relationships without building or probing a GPU."""
    from .metal import build
    reference = Path(reference)
    sources = {str(path.relative_to(_ROOT)): _file_pin(path)
               for directory in _SOURCE_ROOTS for path in sorted((_ROOT / directory).rglob('*'))
               if path.is_file() and path.suffix in ('.py', '.cpp', '.h', '.mm', '.metal')
               and '__pycache__' not in path.parts}
    cpu = {}
    for name, (source, model) in _CPU_ROLES.items():
        meta = _ROOT / name
        record = json.loads(meta.read_text())
        binary = _file_pin(meta.with_suffix(''))
        if record != {'model': model, 'source_sha256': sources[source]['sha256'],
                      'binary_sha256': binary['sha256'],
                      'flags': ['-O3', '-std=c++17', '-shared', '-fPIC']}:
            raise ValueError(f'Physical CPU build relationships differ: {name}')
        cpu[name] = {'metadata': _file_pin(meta), 'binary': binary, 'record': record}
    native_root = _ROOT / 'outputs/doom-learning/metal'
    native = {name: _file_pin(native_root / name) for name in _NATIVE_ROLES}
    record = json.loads((native_root / 'build.json').read_text())
    configuration = {'abi_version': build.ABI_VERSION,
                     'builder_sha256': sources['doom_learning_v6/metal/build.py']['sha256'],
                     'numerical_parent': build.NUMERICAL_PARENT, 'numerical_order': build.NUMERICAL_ORDER,
                     'metal_compile_flags': list(build.METAL_COMPILE_FLAGS),
                     'library_compile_flags': list(build.LIBRARY_COMPILE_FLAGS)}
    if (record.get('schema') != 2 or record.get('abi_version') != build.ABI_VERSION or
            record.get('architecture') != 'arm64' or record.get('metal_language') != 'macos-metal2.4' or
            record.get('build_configuration') != configuration or
            record.get('sources') != {name: sources['doom_learning_v6/metal/' + name]['sha256']
                                      for name in ('api.h', 'backend.mm', 'kernels.metal', 'decay_tables.h')} or
            record.get('binaries') != {role: native[name]['sha256'] for role, name in
                                      [('air', 'kernels.air'), ('metallib', 'kernels.metallib'),
                                       ('library', 'libmemory-metal.dylib')]}):
        raise ValueError('Physical Metal build relationships differ')
    return {'source_commit': _git_output('rev-parse', 'HEAD'), 'sources': sources,
            'inputs': {name: _file_pin(_ROOT / name) for name in _INPUT_ROLES},
            'references': {name: _file_pin(reference / name) for name in _REFERENCE_ROLES},
            'cpu': cpu, 'native': native, 'native_abi': record['abi_version']}


def _validate_expected_pins(path, observed):
    """Compare controller-provided trusted identities before native execution."""
    try:
        expected = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('Trusted expected pins are required') from error
    required = {'source_commit', 'sources', 'inputs', 'references', 'cpu', 'native', 'native_abi'}
    if not isinstance(expected, dict) or not required.issubset(expected):
        raise ValueError('Trusted expected pins lack required roles')
    for value in (expected, observed):
        if (not re.fullmatch(r'[0-9a-f]{40}', str(value.get('source_commit', ''))) or
                not value.get('sources') or value.get('native_abi') != 8 or
                any(set(value.get(key, {})) != set(roles) for key, roles in
                    [('inputs', _INPUT_ROLES), ('references', _REFERENCE_ROLES),
                     ('cpu', _CPU_ROLES), ('native', _NATIVE_ROLES)]) or
                not all(any(name.startswith(root + '/') for name in value['sources']) for root in _SOURCE_ROOTS)):
            raise ValueError('Trusted expected pins lack required roles')
        pins = [pin for key in ('sources', 'inputs', 'references', 'native') for pin in value[key].values()]
        pins.extend(item[role] for item in value['cpu'].values() for role in ('metadata', 'binary'))
        if any(set(pin) != {'bytes', 'sha256'} or type(pin['bytes']) is not int or pin['bytes'] <= 0
               or not re.fullmatch(r'[0-9a-f]{64}', str(pin['sha256'])) for pin in pins):
            raise ValueError('Invalid physical byte/SHA pin')
    if any(expected[key] != observed.get(key) for key in required):
        raise ValueError('Trusted expected pins differ from observed identities')
    return expected


def _readouts():
    from doom_learning.common import GRAPH
    rows = json.loads((GRAPH.parent / 'manifest.json').read_text())['readouts']
    result = [row for row in rows if row['type'] in ('DNp20', 'DNpe017')]
    if len([row for row in result if row['type'] == 'DNp20']) != 2 or len([row for row in result if row['type'] == 'DNpe017']) != 2:
        raise ValueError('Require exactly two DNp20 and two DNpe017 BCI readouts')
    return rows


def _check_reference(reference, train, held=None, readouts=None):
    root = Path(reference)
    try:
        result = json.loads((root / 'results.json').read_text())
        protocol = json.loads((root / 'protocol.json').read_text())
        provenance = json.loads((root / 'provenance.json').read_text())
        trace = json.loads((root / 'plastic' / 'train-0-0' / 'episode.json').read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('Complete original reference results and protocol required') from error
    required = {'schema': 1, 'model': 'adaptive-centered-v6', 'connectome': 'MaleCNS v1.0',
                'neurons': 166700, 'retained_edges': 25582938, 'plastic_slots': 4184,
                'eta': .001, 'epochs': 1, 'diagnostic_limit': None,
                'arms': ['plastic', 'frozen', 'shifted'], 'shift_offsets': [491],
                'training': [train.identity],
                'timing': {'source_hz': 35, 'neural_dt_ms': .1, 'warmup_ms': 2000,
                           'retention_ms': 5000, 'frame_end_steps': 'round((index+1)*10000/35)'},
                'teacher': 'After decoding, min(abs(student-target)/6,1); next original frame gets 4*error at PPL101. First frame and all evaluations receive zero teacher current.',
                'shift': 'Circular positive np.roll turn targets only. Original RGB/raw controls unchanged; feedback dose can differ and is measured.',
                'evaluation_mode': 'Teacher-free frozen efficacies, learned full checkpoint restored independently per episode; decoder and fast state reset.',
                'retention_mode': 'Unfrozen efficacies; no teacher or learning during five seconds dark, then frozen teacher-free evaluation.'}
    if (result.get('complete') is not True or result.get('protocol') != protocol or
            result.get('learning_demonstrated') is not False or result.get('announcement_ready') is not False or
            any(protocol.get(key) != value for key, value in required.items()) or
            not all(protocol.get(key) for key in ('configuration', 'model_identity', 'execution',
                                                'calibration', 'readouts', 'target_calibration', 'claim_gate')) or
            len(protocol.get('evaluation', [])) != 1 or train.frame_count != 982):
        raise ValueError('Complete original reference results and protocol required')
    if held is not None and (held.frame_count != 988 or protocol['evaluation'] != [held.identity]):
        raise ValueError('Original held-out identity differs')
    if readouts is not None and protocol['readouts'] != readouts:
        raise ValueError('Original fixed readout identities differ')
    if (provenance.get('schema') != 2 or provenance.get('captured_before_first_episode') is not True or
            provenance.get('OPENBLAS_NUM_THREADS') != '1' or not provenance.get('source_sha256')):
        raise ValueError('Original provenance relationships are incomplete')
    execution = protocol['execution']
    native = execution.get('backend_metadata', {})
    sources = provenance['source_sha256']
    if (execution.get('backend') != 'metal' or native.get('abi_version') != 5 or
            native.get('build_configuration', {}).get('abi_version') != 5 or
            native.get('build_configuration', {}).get('builder_sha256') != sources.get('doom_learning_v6/metal/build.py') or
            set(native.get('sources', {})) != {'api.h', 'backend.mm', 'kernels.metal', 'decay_tables.h'} or
            any(sources.get('doom_learning_v6/metal/' + name) != sha for name, sha in native['sources'].items()) or
            protocol['configuration'].get('rule_sha256') != sources.get('doom_learning_v6/rule.py')):
        raise ValueError('Original native/provenance relationships differ')
    if len(trace.get('trace', [])) != train.frame_count or trace.get('complete') is not True:
        raise ValueError('Complete original plastic reference trace required')
    if trace.get('identity') != train.identity:
        raise ValueError('Reference trace does not byte-match the training artifact')
    expected_controls = [('plastic', 'training', 'train-0-0'), ('plastic', 'held_out', 'eval-0'),
                         ('plastic', 'retention_5s', 'retention-0'), ('plastic', 'memory_erased', 'erased-0'),
                         ('frozen', 'training', 'train-0-0'), ('frozen', 'held_out', 'eval-0'),
                         ('shifted', 'training', 'train-0-0'), ('shifted', 'held_out', 'eval-0')]
    episodes = result.get('episodes', [])
    if len(episodes) != len(expected_controls):
        raise ValueError('Complete original controls required')
    from .causal_controls import frame_ticks
    memory_keys = ('sha256', 'memory_u_sha256', 'memory_w_sha256')
    baseline = trace['before']
    learned = trace['after']
    for summary, (arm, phase, directory) in zip(episodes, expected_controls):
        try:
            episode = json.loads((root / arm / directory / 'episode.json').read_text())
        except (OSError, ValueError) as error:
            raise ValueError('Complete original control artifact required') from error
        if {k: v for k, v in episode.items() if k != 'trace'} != summary:
            raise ValueError('Original control summary/artifact relationships differ')
        training = phase == 'training'
        count = 982 if training else 988
        flags = {'arm': arm, 'phase': phase, 'complete': True, 'invariants_unchanged': True,
                 'training': training, 'weights_frozen': not training or arm == 'frozen',
                 'frames': count, 'source_frame_count': count, 'diagnostic_limit': None,
                 'target_shift': 491 if training and arm == 'shifted' else 0,
                 'identity': protocol['training' if training else 'evaluation'][0],
                 'brain_steps': int(frame_ticks(count).sum())}
        if (any(key not in episode or episode[key] != value for key, value in flags.items()) or
                len(episode.get('trace', [])) != count or
                episode.get('warmup', {}).get('requested_ms') != 2000 or
                episode['warmup'].get('brain_steps') != 20000):
            raise ValueError('Original control protocol is incomplete')
        rows = episode['trace']
        if training:
            _reference_schedule(episode, count, shift=flags['target_shift'])
            if any(episode['before'][key] != baseline[key] for key in memory_keys):
                raise ValueError('Original controls did not share initial plastic state')
        else:
            if (episode.get('teacher_dose_current_ms') != 0 or episode.get('pending_teacher_current') != 0 or
                    any(row.get('teacher_current') != 0 for row in rows) or
                    any(episode['before'][key] != episode['after'][key] for key in memory_keys)):
                raise ValueError('Original evaluation teacher/freeze relationships differ')
        if arm == 'frozen' or phase == 'memory_erased':
            if any(episode['after'][key] != baseline[key] for key in memory_keys):
                raise ValueError('Original frozen/erasure control changed plastic state')
        if arm == 'plastic' and phase == 'held_out':
            if any(episode['before'][key] != learned[key] for key in memory_keys):
                raise ValueError('Original learned evaluation restore differs')
        if phase == 'retention_5s':
            retention = episode.get('retention', {})
            if (any(retention.get(k) != v for k, v in {'brain_steps': 50000, 'brain_seconds': 5.,
                    'weights_frozen': False, 'learning': False, 'teacher_current': 0.}.items()) or
                    any(retention.get('before', {}).get(k) != learned[k] or
                        retention.get('after', {}).get(k) != episode['before'][k] for k in memory_keys)):
                raise ValueError('Original retention control differs')
        ticks = frame_ticks(count)
        for index, row in enumerate(rows):
            if (row.get('index') != index or row.get('teacher_index') != (index - flags['target_shift']) % count or
                    row.get('neural_steps') != int(ticks[index]) or
                    row.get('brain_steps') != int(ticks[:index + 1].sum())):
                raise ValueError('Original control frame boundaries differ')
    return root, trace


def _reference_relationships(reference, observed):
    """Tie the trusted corpus to graph, checkpoint, rule and backend identities."""
    import numpy as np
    from doom_learning.common import digest
    root = Path(reference)
    protocol = json.loads((root / 'protocol.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance['graph_sha256'] != observed['inputs'][_INPUT_ROLES[0]]['sha256']:
        raise ValueError('Original graph physical identity differs')
    with np.load(_ROOT / _INPUT_ROLES[0], allow_pickle=False) as graph:
        if digest(graph['weight']) != protocol['model_identity']['original_weight_sha256']:
            raise ValueError('Original released graph weights differ')
    implementation_changes = {'doom_learning_v6/brain.py', 'doom_learning_v6/visual.py',
                              'doom_learning_v6/metal/api.h', 'doom_learning_v6/metal/backend.mm',
                              'doom_learning_v6/metal/build.py', 'doom_learning_v6/metal/kernels.metal'}
    changes = {}
    for name, original in provenance['source_sha256'].items():
        current = observed['sources'].get(name, {}).get('sha256')
        if original != current:
            if name not in implementation_changes or current is None:
                raise ValueError(f'Original measured model/rule/source differs: {name}')
            changes[name] = {'original': original, 'current': current}
    checkpoints = {}
    for name in ('initial.npz', 'plastic/learned.npz', 'frozen/learned.npz', 'shifted/learned.npz'):
        with np.load(root / name, allow_pickle=False) as checkpoint:
            metadata = json.loads(str(checkpoint['metadata']))
            if len(set(checkpoint.files) - {'metadata'}) != 24:
                raise ValueError('Original checkpoint must contain all 24 arrays')
            for key, expected in [('model', protocol['model']), ('eta', .001),
                                  ('configuration_sha256', protocol['configuration']),
                                  ('graph_ids_sha256', protocol['model_identity']['ids_sha256']),
                                  ('graph_ptr_sha256', protocol['model_identity']['out_ptr_sha256']),
                                  ('graph_post_sha256', protocol['model_identity']['out_post_sha256']),
                                  ('plastic_edges_sha256', protocol['model_identity']['plastic_edges_sha256'])]:
                if metadata.get(key) != expected:
                    raise ValueError(f'Original checkpoint relationship differs: {name}/{key}')
            if (metadata['build']['source_sha256'] != provenance['source_sha256']['doom_learning_v6/kernel.cpp'] or
                    metadata['build']['flags'] != ['-O3', '-std=c++17', '-shared', '-fPIC'] or
                    metadata.get('producer_backend') != 'metal' or metadata['backend']['abi_version'] != 5 or
                    any(metadata['backend'][k] != protocol['execution']['backend_metadata'][k]
                        for k in ('sources', 'binaries', 'build_configuration'))):
                raise ValueError('Original checkpoint native/CPU relationship differs')
            if name == 'initial.npz' and (metadata['cursor'] != 0 or
                    digest(checkpoint['weight']) != protocol['configuration']['initial_weight'] or
                    np.any(checkpoint['memory_u']) or np.any(checkpoint['memory_w'])):
                raise ValueError('Original initial checkpoint is not baseline state')
            checkpoints[name] = {'pin': _file_pin(root / name), 'metadata': metadata}
    configuration_hash = hashlib.sha256(json.dumps(protocol['configuration'], sort_keys=True,
                                                    separators=(',', ':')).encode()).hexdigest()
    if configuration_hash != protocol['model_identity']['configuration_sha256']:
        raise ValueError('Original configuration hash relationship differs')
    return {'protocol': protocol, 'provenance': provenance, 'checkpoints': checkpoints,
            'implementation_changes': changes,
            'abi_transition': {'original': 5, 'current': observed['native_abi'],
                               'condition': 'Exact A canonical trace and all 24 checkpoint arrays required'},
            'control_pins': {name: _file_pin(root / name)
                             for name in _REFERENCE_ROLES if name.endswith('/episode.json')}}


def _schedule_correlations(aligned, shifted, targets, errors):
    import numpy as np
    def correlation(left, right):
        left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
            return None
        return float(np.corrcoef(left, right)[0, 1])
    return {'aligned_shifted': correlation(aligned, shifted),
            'aligned_target': correlation(aligned, targets), 'shifted_target': correlation(shifted, targets),
            'aligned_error': correlation(aligned, errors), 'shifted_error': correlation(shifted, errors),
            'definition': 'Pearson correlation at identical original frame indices; null means undefined.'}


def _reference_schedule(trace, count, *, shift=0):
    """Validate the preserved one-frame delayed reinforcement records."""
    import numpy as np
    from .causal_controls import frame_ticks
    from .imitation import teacher_error

    rows = trace.get('trace')
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('Complete original plastic reference trace required')
    ticks = frame_ticks(count)
    currents = np.empty(count, dtype=np.float64)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get('index') != index or row.get('teacher_index') != (index - shift) % count:
            raise ValueError('Reference records must preserve contiguous original indices')
        if row.get('neural_steps') != int(ticks[index]):
            raise ValueError('Reference current/tick trace is not the canonical original schedule')
        try:
            current = float(row['teacher_current'])
            error = teacher_error(float(row['action']['turn']), float(row['target_turn']))
        except (KeyError, TypeError, ValueError) as failure:
            raise ValueError('Reference delayed-feedback record is invalid') from failure
        if not math.isfinite(current) or not 0 <= current <= 4 or row.get('teacher_error') != error:
            raise ValueError('Reference delayed-feedback record is invalid')
        if index == 0:
            if current != 0:
                raise ValueError('First reference teacher current must be zero')
        elif current != 4 * float(rows[index - 1]['teacher_error']):
            raise ValueError('Reference delayed-feedback recurrence differs')
        currents[index] = current
    return currents, ticks


def _same_without_wall(left, right):
    """Compare canonical science records after removing only physical clocks."""
    excluded = {'wall_seconds', 'actual_wall_seconds', 'native_elapsed_seconds',
                'teacher_float32_current'}
    if isinstance(left, dict):
        left = {key: value for key, value in left.items() if key not in excluded}
        right = {key: value for key, value in right.items() if key not in excluded}
        return set(left) == set(right) and all(_same_without_wall(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_without_wall(x, y) for x, y in zip(left, right))
    return left == right


def _compare_checkpoint_arrays(reference, candidate):
    """Require exact dtype, shape and data equality for every checkpoint array."""
    import numpy as np
    try:
        with np.load(reference, allow_pickle=False) as expected, np.load(candidate, allow_pickle=False) as actual:
            expected_keys = set(expected.files) - {'metadata'}
            actual_keys = set(actual.files) - {'metadata'}
            if expected_keys != actual_keys:
                raise ValueError('Checkpoint array keys differ')
            for name in sorted(expected_keys):
                if expected[name].dtype != actual[name].dtype:
                    raise ValueError(f'Checkpoint dtype differs: {name}')
                if expected[name].shape != actual[name].shape:
                    raise ValueError(f'Checkpoint shape differs: {name}')
                if expected[name].tobytes() != actual[name].tobytes():
                    raise ValueError(f'Checkpoint data differs: {name}')
    except OSError as error:
        raise ValueError('Checkpoint arrays cannot be compared') from error
    return {'passed': True, 'arrays': sorted(expected_keys)}


def _sensitivity_memory_controls(direction):
    """Return the prescribed directional and uniform memory interventions."""
    import numpy as np
    value = np.asarray(direction, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Finite learned direction required')
    return {'primary': np.stack([np.zeros_like(value), np.zeros_like(value), .05 * value, -.05 * value]),
            'fallback': np.stack([.20 * value, -.20 * value,
                                  np.full_like(value, -.05), np.full_like(value, .05)])}


def _run_causal(args, train, held, reference, readouts, out, pressure=None):
    import numpy as np
    from .causal_controls import dose_proof, duration_permutation, frame_ticks
    from .calibration import calibrated_brain
    from .metal.batch import MetalBatchExecutor

    _, reference_trace = _check_reference(reference, train, held, readouts)
    protocol = json.loads((Path(reference) / 'protocol.json').read_text())
    aligned, ticks = _reference_schedule(reference_trace, train.frame_count)
    shifted, mapping = duration_permutation(aligned, ticks)
    proof = dose_proof(aligned, shifted, ticks)
    correlations = _schedule_correlations(aligned, shifted,
                                         [row['target_turn'] for row in reference_trace['trace']],
                                         [row['teacher_error'] for row in reference_trace['trace']])
    initial = Path(reference) / 'initial.npz'
    pressure = pressure or _PressureHistory(out)
    with _Resources() as resources:
        brains = _restored_brains(resources, lambda: calibrated_brain(.001, backend='cpu'), initial, 4)
        before = _capture_snapshot(brains, protocol, out, 'before-training')
        executor = resources.add(MetalBatchExecutor(brains, window_ticks=18))
        with _phase_boundary(pressure, 'training'):
            train_result = replay_episode(brains, executor, train, readouts,
                                          np.stack([aligned, np.zeros_like(aligned), shifted, np.zeros_like(aligned)]),
                                          learning=[True, True, True, True], frozen=[False, False, False, True],
                                          directory=out / 'train')
        if not _same_without_wall(train_result['lanes'][0]['trace'], reference_trace['trace']):
            raise ValueError('A replay differs from the preserved original plastic trace')
        after_training = _capture_snapshot(brains, protocol, out, 'after-training')
        if after_training != before:
            raise ValueError('Fixed invariants changed during training')
        learned = []
        for index, brain in enumerate(brains):
            path = out / f'lane-{index}' / 'learned.npz'; brain.checkpoint(path); learned.append(path)
        golden = _compare_checkpoint_arrays(Path(reference) / 'plastic' / 'learned.npz', learned[0])
        if len(golden['arrays']) != 24:
            raise ValueError('Original A comparison must cover all 24 checkpoint arrays')
        for brain, checkpoint in zip(brains, learned):
            brain.restore(checkpoint)
        with _phase_boundary(pressure, 'evaluation'):
            evaluation = replay_episode(brains, executor, held, readouts,
                                        np.zeros((4, held.frame_count), dtype=np.float64),
                                        learning=[False] * 4, frozen=[True] * 4,
                                        directory=out / 'evaluation')
        after = _capture_snapshot(brains, protocol, out, 'after-evaluation')
        if after != before:
            raise ValueError('Fixed invariants changed')
    return {'mode': 'causal', 'schedule_mapping': mapping.tolist(), 'dose_proof': proof,
            'residual_schedule_correlations': correlations,
            'train': train_result, 'evaluation': evaluation,
            'learned_checkpoints': [str(path.relative_to(out)) for path in learned],
            'golden_checkpoint_comparison': golden,
            'invariants': {'before': before['graph'], 'after_training': after_training['graph'], 'after': after['graph']},
            'configuration': {'before': before['configuration'], 'after_training': after_training['configuration'],
                              'after': after['configuration']},
            'phase_pressure': pressure.observations}


def _run_sensitivity(train, reference, readouts, out, pressure=None):
    """Measure declared initial-efficacy perturbations with frozen weights."""
    import numpy as np
    from .causal_controls import initialize_efficacies, learned_direction
    from .calibration import calibrated_brain
    from .metal.batch import MetalBatchExecutor

    _check_reference(reference, train, readouts=readouts)
    protocol = json.loads((Path(reference) / 'protocol.json').read_text())
    pressure = pressure or _PressureHistory(out)
    initial = Path(reference) / 'initial.npz'
    with _Resources() as resources:
        probe = _restored_brains(resources, lambda: calibrated_brain(.001, backend='cpu'), initial, 1)[0]
        _model_gate([probe], protocol)
        baseline = probe.weight[probe.circuit['edges']].copy()
        with np.load(Path(reference) / 'plastic' / 'learned.npz', allow_pickle=False) as saved:
            learned = saved['weight'][probe.circuit['edges']].copy()
        direction = learned_direction(baseline, learned)
    controls = _sensitivity_memory_controls(direction)
    with _Resources() as resources:
        brains = _restored_brains(resources, lambda: calibrated_brain(.001, backend='cpu'),
                                  initial, 4, controls['primary'])
        before = _capture_snapshot(brains, protocol, out, 'before-sensitivity')
        executor = resources.add(MetalBatchExecutor(brains, window_ticks=18))
        with _phase_boundary(pressure, 'sensitivity'):
            first = replay_episode(brains, executor, train, readouts,
                                   np.zeros((len(brains), train.frame_count), dtype=np.float64),
                                   learning=[True] * len(brains), frozen=[True] * len(brains),
                                   directory=out / 'sensitivity')
        replica = [_without_wall_lane(first['lanes'][index]) for index in (0, 1)]
        if replica[0] != replica[1]:
            raise ValueError('Zero-memory sensitivity replicas diverged')
        actions = [_decoded_controls(lane) for lane in first['lanes']]
        null = actions[2] == actions[0] and actions[3] == actions[0]
        after = _capture_snapshot(brains, protocol, out, 'after-sensitivity')
        if after != before:
            raise ValueError('Fixed invariants changed')
    additional = None
    extra_before = extra_after = None
    if null:
        with _Resources() as resources:
            extra = _restored_brains(resources, lambda: calibrated_brain(.001, backend='cpu'),
                                    initial, 4, controls['fallback'])
            extra_before = _capture_snapshot(extra, protocol, out, 'before-additional')
            if extra_before != before:
                raise ValueError('Fallback original fixed invariants differ')
            executor = resources.add(MetalBatchExecutor(extra, window_ticks=18))
            with _phase_boundary(pressure, 'additional'):
                additional = replay_episode(extra, executor, train, readouts,
                                            np.zeros((len(extra), train.frame_count), dtype=np.float64),
                                            learning=[True] * len(extra), frozen=[True] * len(extra),
                                            directory=out / 'sensitivity-additional')
            extra_after = _capture_snapshot(extra, protocol, out, 'after-additional')
            if extra_after != extra_before:
                raise ValueError('Fixed invariants changed')
    baseline_actions = _decoded_controls(first['lanes'][0])
    fallback_comparisons = (None if additional is None else
                            [_decoded_controls(lane) == baseline_actions
                             for lane in additional['lanes']])
    return {'mode': 'sensitivity', 'direction': direction.tolist(),
            'perturbations': controls['primary'].tolist(), 'training': first,
            'replica_exact': True, 'manual_perturbation_actions_unchanged': null,
            'additional_perturbations': (controls['fallback'].tolist() if null else None),
            'additional_training': additional, 'fallback_baseline_action_comparisons': fallback_comparisons,
            'invariants': {'before': before['graph'], 'after': after['graph']},
            'configuration': {'before': before['configuration'], 'after': after['configuration']},
            'additional_invariants': {'before': extra_before, 'after': extra_after},
            'phase_pressure': pressure.observations,
            'limitation': 'Frozen perturbations probe readout sensitivity, not learned behavior.'}


def _decoded_controls(lane):
    """Actual buttons/turns determine readout sensitivity; diagnostics remain in traces."""
    return [{key: row['action'][key] for key in ('turn', 'forward', 'attack')} for row in lane['trace']]


def _without_wall_lane(value):
    if isinstance(value, dict):
        return {key: _without_wall_lane(item) for key, item in value.items()
                if 'wall' not in key and 'elapsed' not in key}
    if isinstance(value, list):
        return [_without_wall_lane(item) for item in value]
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['sensitivity', 'causal'], required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--expected-pins')
    return parser


def run(args):
    """Run one locked native pilot only after portable identity gates pass."""
    out = Path(args.out)
    if out.exists():
        raise ValueError('Fresh output directory required')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit) or _git_output('rev-parse', 'HEAD') != args.source_commit:
        raise ValueError('Source commit mismatch')
    if not getattr(args, 'expected_pins', None):
        raise ValueError('Trusted expected pins are required for native causal pilot')
    if sys.platform != 'darwin' or platform.system() != 'Darwin':
        raise ValueError('Resident causal pilot requires macOS Metal')
    _source_clean()
    import numpy as np
    from doom_learning.common import save_json
    from .demonstrations import OfflineEpisode
    from .metal.benchmark import _file_digest

    train, held = OfflineEpisode(args.train), OfflineEpisode(args.eval)
    if train.frame_count != 982 or held.frame_count != 988:
        raise ValueError('Require original 982-frame train and 988-frame evaluation episodes')
    if train.identity['video_sha256'] == held.identity['video_sha256'] or train.identity['control_sha256'] == held.identity['control_sha256']:
        raise ValueError('Training and evaluation artifacts must be distinct')
    readouts = _readouts()
    _check_reference(args.reference, train, held, readouts)
    observed = _physical_pins(args.reference)
    expected = _validate_expected_pins(args.expected_pins, observed)
    relationships = _reference_relationships(args.reference, observed)
    lock = _acquire_gpu_lock()
    pressure = _PressureHistory(out)
    failure = None
    try:
        out.mkdir(parents=True)
        save_json(out / 'expected-pins.json', expected)
        save_json(out / 'observed-pins.json', observed)
        save_json(out / 'reference-validation.json', relationships)
        save_json(out / 'inputs.json', {'arguments': vars(args), 'train': train.identity, 'evaluation': held.identity,
                                        'reference': _file_digest(Path(args.reference) / 'results.json'),
                                        'source_commit': args.source_commit, 'lock_pid': os.getpid(),
                                        'expected_pins_file': _file_pin(args.expected_pins),
                                        'physical_pins': observed})
        before = pressure.observe('initial')
        result = (_run_causal(args, train, held, args.reference, readouts, out, pressure)
                  if args.mode == 'causal' else _run_sensitivity(train, args.reference, readouts, out, pressure))
        after = pressure.observe('final')
        after_pins = _physical_pins(args.reference)
        _validate_expected_pins(args.expected_pins, after_pins)
        save_json(out / 'observed-pins-after.json', after_pins)
        result.update({'complete': True, 'learning_demonstrated': False, 'announcement_ready': False,
                       'source_commit': args.source_commit,
                       'inputs': {'train': train.identity, 'evaluation': held.identity},
                       'physical_identity_records': ['expected-pins.json', 'observed-pins.json', 'observed-pins-after.json',
                                                     'reference-validation.json'],
                       'memory_pressure': {'before': before, 'after': after},
                       'mechanism_diagnostic_scope': 'Controlled schedule comparison only; it does not establish fly learning.'})
    except BaseException as error:
        failure = error
        if out.exists():
            _save_failure(out, failure, pressure)
        raise
    finally:
        for cleanup in (lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN), lock.close):
            try:
                cleanup()
            except BaseException as secondary:
                if failure is None:
                    failure = secondary
                else:
                    failure.add_note(f'Secondary lock cleanup failure: {type(secondary).__name__}: {secondary}')
        if failure is not None:
            if out.exists():
                _save_failure(out, failure, pressure)
            raise failure
    try:
        save_json(out / 'results.json', result)
    except BaseException as failure:
        _save_failure(out, failure, pressure)
        raise
    return result


def _save_failure(out, failure, pressure):
    """Evidence writers must never replace the cause of a failed experiment."""
    from doom_learning.common import save_json
    def record():
        return {'failure': {'type': type(failure).__name__, 'message': str(failure),
                            'secondary_notes': list(getattr(failure, '__notes__', []))},
                'progress': 'incomplete', 'phase_pressure': pressure.observations,
                'complete': False, 'learning_demonstrated': False, 'announcement_ready': False}
    try:
        save_json(Path(out) / 'failure.json', record())
    except BaseException as secondary:
        failure.add_note(f'Secondary failure writer failure: {type(secondary).__name__}: {secondary}')
        try:
            (Path(out) / 'failure-fallback.json').write_text(json.dumps(record(), indent=2, allow_nan=False) + '\n')
        except BaseException as fallback:
            failure.add_note(f'Secondary fallback writer failure: {type(fallback).__name__}: {fallback}')


def main():
    from doom_learning.common import require_single_blas_thread
    require_single_blas_thread()
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
