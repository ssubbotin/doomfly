"""Resident Metal replay for the controlled causal learning pilot.

Pixels, neural propagation, the centered plasticity rule, and fixed BCI
decoding retain their existing implementations.  This runner records their
separate inputs and outputs without using game telemetry as an action policy.
"""

import argparse
import fcntl
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


def _lane_record(brain, data, frozen, before, trace, origin, started, warmup, failure=None, close_failure=None):
    from .causal_controls import diagnostic_score
    from .imitation import _memory
    result = {
        'identity': dict(data.identity), 'weights_frozen': bool(frozen),
        'source_frame_count': int(data.frame_count), 'frames': len(trace),
        'complete': failure is None and len(trace) == data.frame_count,
        'brain_steps': int(brain.cursor - origin), 'brain_seconds': (brain.cursor - origin) * .0001,
        'recorded_seconds': len(trace) / 35, 'wall_seconds': time.perf_counter() - started,
        'warmup': warmup, 'before': before, 'after': _memory(brain),
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
    for index, (brain, record) in enumerate(zip(lanes, records)):
        lane = phase / f'lane-{index}'
        save_json(lane / 'episode.json', record)
        brain.checkpoint(lane / ('failure.npz' if failure is not None else 'final.npz'))
    summary = {'lanes': records, 'complete': failure is None and all(row['complete'] for row in records),
               'resident_bytes': resident_bytes}
    if failure is not None:
        summary['failure'] = {'type': type(failure).__name__}
    save_json(phase / 'summary.json', summary)
    save_json(phase / 'progress.json', {'complete': summary['complete'],
                                        'frames': [len(row['trace']) for row in records],
                                        **({'failure': summary['failure']} if failure is not None else {})})
    return summary


def replay_episode(brains, executor, data, readouts, currents, *, learning, frozen, directory, warmup_ms=2000):
    """Replay every original RGB frame through one already-resident executor.

    Validation completes before lane reset.  Any failure writes only evidence
    already produced, then raises that original exception unchanged.
    """
    import numpy as np
    from doom_learning.common import digest
    from .imitation import _memory, teacher_error

    lanes, ticks, height, width, learn, freeze, schedule, targets, phase, controls = _phase_inputs(
        brains, executor, data, readouts, currents, learning, frozen, directory, warmup_ms)
    phase.mkdir(parents=True)
    try:
        metadata = executor.metadata()
        resident_bytes = {name: metadata[name] for name in ('shared_resident_bytes', 'mutable_resident_bytes')}
    except Exception:
        resident_bytes = None
    started = time.perf_counter()
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
        warmup_cursor = [brain.cursor for brain in lanes]
        if warmup_ms:
            executor.rgb_step([black] * len(lanes), warmup_ms, learning=False)
        warmup['wall_seconds'] = time.perf_counter() - warmup_started
        warmup['actual_wall_seconds'] = warmup['wall_seconds']
        warmup['brain_steps'] = [int(brain.cursor - cursor) for brain, cursor in zip(lanes, warmup_cursor)]
        warmup['brain_seconds'] = [steps * .0001 for steps in warmup['brain_steps']]
        origins = [int(brain.cursor) for brain in lanes]
        before = [_memory(brain) for brain in lanes]
        frozen_state = [(brain.memory_u.copy(), brain.memory_w.copy(),
                         brain.weight[brain.circuit['edges']].copy()) if value else None
                        for brain, value in zip(lanes, freeze)]
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
        if any(len(trace) != data.frame_count for trace in traces):
            raise ValueError('Incomplete original episode')
        for brain, value, expected in zip(lanes, freeze, frozen_state):
            if value and any(not np.array_equal(actual, wanted) for actual, wanted in zip(
                    (brain.memory_u, brain.memory_w, brain.weight[brain.circuit['edges']]), expected)):
                raise ValueError('Frozen plastic state changed')
    except Exception as error:
        failure = error
    finally:
        if frames is not None and hasattr(frames, 'close'):
            try:
                frames.close()
            except Exception as error:
                if failure is None:
                    failure = error
                else:
                    close_failure = error
    try:
        records = [_lane_record(brain, data, value, item_before, trace, origin, started, warmup,
                                failure, close_failure)
                   for brain, value, item_before, trace, origin in zip(lanes, freeze, before, traces, origins)]
    except Exception as evidence_failure:
        if failure is None:
            raise
        failure.add_note(f'Secondary record failure: {type(evidence_failure).__name__}')
        raise failure
    try:
        summary = _write_phase(phase, lanes, records, failure=failure, resident_bytes=resident_bytes)
    except Exception as evidence_failure:
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
    value = _memory_status()
    if value['free_percent'] < 25:
        raise ValueError('Insufficient free memory for causal pilot')
    return value


def _phase_pressure(before, name):
    """Sample a phase boundary and reject either unsafe free RAM or swap growth."""
    after = _memory_pressure()
    if after['swap_used_mib'] > before['swap_used_mib']:
        raise ValueError(f'Swap grew during {name}')
    return after


def _model_gate(brains):
    """Validate the full retained graph and materialize immutable evidence."""
    from .imitation import _invariants
    records = []
    for brain in brains:
        brain.backend.materialize('causal-pilot-invariants')
        edges = brain.circuit['edges']
        if (brain.n, len(brain.weight), len(edges), len(brain.circuit['dan'])) != (166700, 25582938, 4184, 2):
            raise ValueError('Exact retained graph, plastic slots and two DAN neurons required')
        values = brain.weight[edges] / brain.baseline_plastic
        if not (values > 0).all() or not (values >= .1).all() or not (values <= 2).all():
            raise ValueError('Plastic efficacies exceed declared baseline-relative bounds')
        records.append(_invariants(brain))
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
    """Pin ignored inputs and native build identities before any simulation."""
    from doom_learning.common import GRAPH, ROOT, provenance_sources
    from .metal.benchmark import current_preflight_identity
    reference = Path(reference)
    required = ['results.json', 'protocol.json', 'provenance.json', 'initial.npz',
                'plastic/learned.npz', 'plastic/train-0-0/episode.json',
                'plastic/eval-0/episode.json', 'frozen/eval-0/episode.json']
    return {'graph': _file_pin(GRAPH),
            'annotations': _file_pin(ROOT / 'connectome_data/malecns_v1/annotations.feather'),
            'normalized_neurons': _file_pin(ROOT / 'connectome_data/malecns_v1/normalized/neurons.feather'),
            'reference': {name: _file_pin(reference / name) for name in required},
            'sources': {str(path.relative_to(ROOT)): _file_pin(path)
                        for path in provenance_sources(ROOT, ['doom', 'doom_learning', 'doom_learning_v2',
                                                              'doom_learning_v4', 'doom_learning_v5', 'doom_learning_v6'])},
            'native_build': current_preflight_identity()}


def _readouts():
    from doom_learning.common import GRAPH
    rows = json.loads((GRAPH.parent / 'manifest.json').read_text())['readouts']
    result = [row for row in rows if row['type'] in ('DNp20', 'DNpe017')]
    if len([row for row in result if row['type'] == 'DNp20']) != 2 or len([row for row in result if row['type'] == 'DNpe017']) != 2:
        raise ValueError('Require exactly two DNp20 and two DNpe017 BCI readouts')
    return result


def _check_reference(reference, train):
    root = Path(reference)
    try:
        result = json.loads((root / 'results.json').read_text())
        protocol = json.loads((root / 'protocol.json').read_text())
        trace = json.loads((root / 'plastic' / 'train-0-0' / 'episode.json').read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('Complete original reference results and protocol required') from error
    if result.get('complete') is not True or protocol.get('eta') != .001 or protocol.get('epochs') != 1 or protocol.get('diagnostic_limit') is not None:
        raise ValueError('Complete original reference results and protocol required')
    if len(trace.get('trace', [])) != train.frame_count or trace.get('complete') is not True:
        raise ValueError('Complete original plastic reference trace required')
    if trace.get('identity') != train.identity:
        raise ValueError('Reference trace does not byte-match the training artifact')
    return root, trace


def _reference_schedule(trace, count):
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
        if not isinstance(row, dict) or row.get('index') != index or row.get('teacher_index') != index:
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
                if not np.array_equal(expected[name], actual[name]):
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


def _run_causal(args, train, held, reference, readouts, out):
    import numpy as np
    from .causal_controls import dose_proof, duration_permutation, frame_ticks
    from .calibration import calibrated_brain
    from .metal.batch import MetalBatchExecutor

    _, reference_trace = _check_reference(reference, train)
    aligned, ticks = _reference_schedule(reference_trace, train.frame_count)
    shifted, mapping = duration_permutation(aligned, ticks)
    proof = dose_proof(aligned, shifted, ticks)
    brains = [calibrated_brain(.001, backend='cpu') for _ in range(4)]
    initial = Path(reference) / 'initial.npz'
    for brain in brains:
        brain.restore(initial)
    invariant = _model_gate(brains)
    pressure = {'before_training': _memory_pressure()}
    try:
        with MetalBatchExecutor(brains, window_ticks=18) as executor:
            invariant = [brain.configuration_signature() for brain in brains]
            train_result = replay_episode(brains, executor, train, readouts,
                                          np.stack([aligned, np.zeros_like(aligned), shifted, np.zeros_like(aligned)]),
                                          learning=[True, True, True, True], frozen=[False, False, False, True],
                                          directory=out / 'train')
            pressure['after_training'] = _phase_pressure(pressure['before_training'], 'training')
            if not _same_without_wall(train_result['lanes'][0]['trace'], reference_trace['trace']):
                raise ValueError('A replay differs from the preserved original plastic trace')
            learned = []
            for index, brain in enumerate(brains):
                path = out / f'lane-{index}' / 'learned.npz'; brain.checkpoint(path); learned.append(path)
            golden = _compare_checkpoint_arrays(Path(reference) / 'plastic' / 'learned.npz', learned[0])
            for brain, checkpoint in zip(brains, learned):
                brain.restore(checkpoint)
            pressure['before_evaluation'] = _memory_pressure()
            evaluation = replay_episode(brains, executor, held, readouts,
                                        np.zeros((4, held.frame_count), dtype=np.float64),
                                        learning=[False] * 4, frozen=[True] * 4,
                                        directory=out / 'evaluation')
            pressure['after_evaluation'] = _phase_pressure(pressure['before_evaluation'], 'evaluation')
            if _model_gate(brains) != invariant:
                raise ValueError('Fixed invariants changed')
    finally:
        for brain in brains:
            if getattr(brain.backend, 'name', None) != 'metal-batch':
                brain.backend.close()
    return {'mode': 'causal', 'schedule_mapping': mapping.tolist(), 'dose_proof': proof,
            'train': train_result, 'evaluation': evaluation,
            'learned_checkpoints': [str(path.relative_to(out)) for path in learned],
            'golden_checkpoint_comparison': golden, 'invariants': {'before': invariant, 'after': invariant},
            'phase_pressure': pressure}


def _run_sensitivity(train, reference, readouts, out):
    """Measure declared initial-efficacy perturbations with frozen weights."""
    import numpy as np
    from .causal_controls import initialize_efficacies, learned_direction
    from .calibration import calibrated_brain
    from .metal.batch import MetalBatchExecutor

    _check_reference(reference, train)
    initial = Path(reference) / 'initial.npz'
    probe = calibrated_brain(.001, backend='cpu')
    try:
        probe.restore(initial)
        baseline = probe.weight[probe.circuit['edges']].copy()
        with np.load(Path(reference) / 'plastic' / 'learned.npz', allow_pickle=False) as saved:
            learned = saved['weight'][probe.circuit['edges']].copy()
        direction = learned_direction(baseline, learned)
    finally:
        probe.backend.close()
    controls = _sensitivity_memory_controls(direction)
    brains = [calibrated_brain(.001, backend='cpu') for _ in range(4)]
    for brain, memory in zip(brains, controls['primary']):
        brain.restore(initial)
        initialize_efficacies(brain, memory)
    invariant = _model_gate(brains)
    pressure = {'before_training': _memory_pressure()}
    try:
        with MetalBatchExecutor(brains, window_ticks=18) as executor:
            first = replay_episode(brains, executor, train, readouts,
                                   np.zeros((len(brains), train.frame_count), dtype=np.float64),
                                   learning=[True] * len(brains), frozen=[True] * len(brains),
                                   directory=out / 'sensitivity')
            pressure['after_training'] = _phase_pressure(pressure['before_training'], 'sensitivity training')
            replica = [_without_wall_lane(first['lanes'][index]) for index in (0, 1)]
            if replica[0] != replica[1]:
                raise ValueError('Zero-memory sensitivity replicas diverged')
            actions = [[row['action'] for row in lane['trace']] for lane in first['lanes']]
            null = actions[2] == actions[0] and actions[3] == actions[0]
            if _model_gate(brains) != invariant:
                raise ValueError('Fixed invariants changed')
    finally:
        for brain in brains:
            if getattr(brain.backend, 'name', None) != 'metal-batch':
                brain.backend.close()
    additional = None
    if null:
        extra = [calibrated_brain(.001, backend='cpu') for _ in range(4)]
        for brain, memory in zip(extra, controls['fallback']):
            brain.restore(initial)
            initialize_efficacies(brain, memory)
        extra_invariant = _model_gate(extra)
        pressure['before_additional'] = _memory_pressure()
        try:
            with MetalBatchExecutor(extra, window_ticks=18) as executor:
                additional = replay_episode(extra, executor, train, readouts,
                                            np.zeros((len(extra), train.frame_count), dtype=np.float64),
                                            learning=[True] * len(extra), frozen=[True] * len(extra),
                                            directory=out / 'sensitivity-additional')
                pressure['after_additional'] = _phase_pressure(pressure['before_additional'], 'sensitivity controls')
                if _model_gate(extra) != extra_invariant:
                    raise ValueError('Fixed invariants changed')
        finally:
            for brain in extra:
                if getattr(brain.backend, 'name', None) != 'metal-batch':
                    brain.backend.close()
    baseline_actions = [row['action'] for row in first['lanes'][0]['trace']]
    fallback_comparisons = (None if additional is None else
                            [[row['action'] for row in lane['trace']] == baseline_actions
                             for lane in additional['lanes']])
    return {'mode': 'sensitivity', 'direction': direction.tolist(),
            'perturbations': controls['primary'].tolist(), 'training': first,
            'replica_exact': True, 'manual_perturbation_actions_unchanged': null,
            'additional_perturbations': (controls['fallback'].tolist() if null else None),
            'additional_training': additional, 'fallback_baseline_action_comparisons': fallback_comparisons,
            'invariants': {'before': invariant, 'after': invariant}, 'phase_pressure': pressure,
            'limitation': 'Frozen perturbations probe readout sensitivity, not learned behavior.'}


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
    return parser


def run(args):
    """Run one locked native pilot only after portable identity gates pass."""
    out = Path(args.out)
    if out.exists():
        raise ValueError('Fresh output directory required')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit) or _git_output('rev-parse', 'HEAD') != args.source_commit:
        raise ValueError('Source commit mismatch')
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
    _check_reference(args.reference, train)
    pins = _physical_pins(args.reference)
    lock = _acquire_gpu_lock()
    try:
        before = _memory_pressure()
        out.mkdir(parents=True)
        save_json(out / 'inputs.json', {'arguments': vars(args), 'train': train.identity, 'evaluation': held.identity,
                                        'reference': _file_digest(Path(args.reference) / 'results.json'),
                                        'source_commit': args.source_commit, 'lock_pid': os.getpid(),
                                        'physical_pins': pins})
        result = (_run_causal(args, train, held, args.reference, readouts, out)
                  if args.mode == 'causal' else _run_sensitivity(train, args.reference, readouts, out))
        after = _memory_pressure()
        result.update({'complete': True, 'learning_demonstrated': False, 'announcement_ready': False,
                       'memory_pressure': {'before': before, 'after': after},
                       'mechanism_diagnostic_scope': 'Controlled schedule comparison only; it does not establish fly learning.'})
        save_json(out / 'results.json', result)
        return result
    except Exception as failure:
        if out.exists():
            save_json(out / 'failure.json', {'failure': {'type': type(failure).__name__},
                                              'progress': 'incomplete'})
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    from doom_learning.common import require_single_blas_thread
    require_single_blas_thread()
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
