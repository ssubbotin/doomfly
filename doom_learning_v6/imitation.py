"""Offline turn-error reinforcement of the retained calibrated neural model.

Teacher labels are scored after fixed decoding and stimulate the following
frame only. Numerical controls and changed efficacies cannot certify learning.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from doom.engine import NeuralControls
from doom_learning.common import GRAPH, capture_provenance, digest, save_json, require_single_blas_thread
from .calibration import calibrated_brain
from .demonstrations import OfflineEpisode


DEAD_ZONE = .1
_IDENTITY_KEYS = ('dataset', 'dataset_revision', 'scenario', 'episode_index',
                  'video_sha256', 'control_sha256')


def teacher_error(decoded_turn, target_turn):
    """Declared scalar hypothesis; no label or game state selects controls."""
    if not math.isfinite(decoded_turn) or not math.isfinite(target_turn):
        raise ValueError('Finite turn values required')
    return min(abs(decoded_turn - target_turn) / 6., 1.)


def _turns(values):
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError('Finite one-dimensional turn values required') from error
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Finite nonempty one-dimensional turn values required')
    return result


def _direction(values):
    return np.where(values < -DEAD_ZONE, -1, np.where(values > DEAD_ZONE, 1, 0))


def _score(predictions, targets):
    predicted, actual = _direction(predictions), _direction(targets)
    support, recall = {}, {}
    for name, direction in [('left', -1), ('idle', 0), ('right', 1)]:
        selected = actual == direction
        support[name] = int(selected.sum())
        if support[name]:
            recall[name] = float(np.mean(predicted[selected] == direction))
    return {'mae_degrees': float(np.mean(np.abs(predictions - targets))),
            'direction_support': support, 'direction_recall': recall,
            'balanced_direction_recall': float(np.mean(list(recall.values())))}


def score_turns(predictions, targets):
    """Equal-class recall over present target classes with trivial baselines.

Directional constant magnitudes use the observed target mean absolute turn;
these descriptive baselines use labels and are never neural action policies.
"""
    predictions, targets = _turns(predictions), _turns(targets)
    if predictions.shape != targets.shape:
        raise ValueError('Prediction and target lengths must match')
    result = _score(predictions, targets)
    magnitude = max(float(np.mean(np.abs(targets))), np.nextafter(DEAD_ZONE, math.inf))
    result['dead_zone_degrees'] = DEAD_ZONE
    result['constant_baselines'] = {
        name: {'turn_degrees': value, **_score(np.full_like(targets, value), targets)}
        for name, value in [('left', -magnitude), ('idle', 0.), ('right', magnitude)]}
    result['constant_baseline_definition'] = 'Idle zero; left/right signed mean absolute target turn, above the dead zone.'
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _shift(data, offset):
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)):
        raise ValueError('Target shift must be an integer')
    return int(offset % data.frame_count)


def _targets(data):
    targets = _turns(data.turn_targets())
    if len(targets) != data.frame_count:
        raise ValueError('Episode target count differs from recorded frames')
    return targets


def _identity(data):
    return {key: data.identity[key] for key in _IDENTITY_KEYS}


def _memory(brain):
    return {**brain.memory(), **{name + '_sha256': digest(getattr(brain, name))
                                for name in ['memory_u', 'memory_w', 'rate_kc', 'rate_dan']}}


class EpisodeFailure(RuntimeError):
    """A failed episode retains its completed-frame evidence for the caller."""
    def __init__(self, summary):
        super().__init__('Episode execution failed (' + summary['failure']['type'] + ')')
        self.summary = summary


def episode(brain, data, readouts, *, training, frozen, target_shift=0,
            limit=None, warmup_ms=2000):
    """Run every requested original frame, independently resetting fast state."""
    _positive_integer(data.frame_count, 'Frame count')
    if limit is not None:
        _positive_integer(limit, 'Frame limit')
    if not math.isfinite(warmup_ms) or warmup_ms < 0 or (warmup_ms and round(warmup_ms * 10) < 1):
        raise ValueError('Finite nonnegative warmup required')
    offset = _shift(data, target_shift)
    targets = _targets(data)
    controls = NeuralControls(readouts, mode='bci')
    trace = []; pending_current = kernel_seconds = warmup_kernel = dose = 0.
    origin = None; warmup_seconds = 0.; warmup_steps = 0; before = _memory(brain)
    started = time.perf_counter(); frames = None
    failure = None; close_failure = None

    def summary(failure=None):
        steps = 0 if origin is None else brain.cursor - origin
        result = {'identity': _identity(data), 'training': bool(training),
                  'weights_frozen': bool(frozen), 'target_shift': offset,
                  'source_frame_count': data.frame_count, 'diagnostic_limit': limit,
                  'frames': len(trace), 'complete': failure is None and len(trace) == data.frame_count,
                  'brain_steps': steps, 'brain_seconds': steps * .0001,
                  'recorded_seconds': len(trace) / 35,
                  'wall_seconds': time.perf_counter() - started,
                  'kernel_seconds': kernel_seconds,
                  'warmup': {'requested_ms': warmup_ms, 'brain_steps': warmup_steps, 'brain_seconds': warmup_steps * .0001,
                             'wall_seconds': warmup_seconds, 'kernel_seconds': warmup_kernel},
                  'teacher_dose_current_ms': dose, 'pending_teacher_current': pending_current,
                  'before': before, 'after': _memory(brain), 'trace': trace,
                  'score': score_turns([r['action']['turn'] for r in trace],
                                       [r['target_turn'] for r in trace]) if trace else None}
        if failure is not None:
            # Detailed exception chains remain local; public records contain no origins.
            result['failure'] = {'type': type(failure).__name__}
            if close_failure is not None:
                result['failure']['close_type'] = type(close_failure).__name__
        return result

    try:
        brain.reset(keep_memory=True)
        brain.weights_frozen = bool(frozen)
        warmup_started = time.perf_counter()
        warmup_origin = brain.cursor
        try:
            if warmup_ms:
                _, warmup_kernel = brain.rgb_step(np.zeros((data.height, data.width, 3), dtype=np.uint8),
                                                  warmup_ms, learning=False, stimulation=None)
        finally:
            warmup_seconds = time.perf_counter() - warmup_started
            warmup_steps = brain.cursor - warmup_origin
        origin = brain.cursor; before = _memory(brain)
        frames = data.iter_frames(limit=limit)
        expected = data.frame_count if limit is None else min(limit, data.frame_count)
        for sample in frames:
            index = sample.index
            if index != len(trace) or index >= expected:
                raise ValueError('Frames must preserve contiguous original indices')
            steps = round((index + 1) * 10000 / 35) - (brain.cursor - origin)
            if steps <= 0:
                raise ValueError('Neural cursor exceeded original frame boundary')
            imposed = pending_current if training else 0.
            counts, seconds = brain.rgb_step(sample.rgb, steps * .1, learning=bool(training),
                stimulation=(brain.circuit['dan'], imposed) if imposed else None)
            kernel_seconds += seconds
            if brain.cursor - origin != round((index + 1) * 10000 / 35):
                raise ValueError('Neural integration did not reach original frame boundary')
            action = controls.decode(counts, steps * .0001)
            teacher_index = (index - offset) % data.frame_count
            target = float(targets[teacher_index])
            error = teacher_error(action['turn'], target)
            dose += imposed * steps * .1
            trace.append({'index': int(index), 'timestamp': float(sample.timestamp),
                          'raw_action': np.asarray(sample.action).tolist(),
                          'teacher_index': teacher_index, 'target_turn': target,
                          'teacher_error': error, 'teacher_current': imposed,
                          'teacher_dose_current_ms': imposed * steps * .1,
                          'neural_steps': steps, 'brain_steps': brain.cursor - origin,
                          'brain_seconds': (brain.cursor - origin) * .0001,
                          'wall_seconds': time.perf_counter() - started,
                          'action': action, 'KC_spikes': int(counts[brain.circuit['kc']].sum()),
                          'DAN_spikes': counts[brain.circuit['dan']].tolist(),
                          'MBON_spikes': counts[brain.circuit['mb']].tolist(),
                          'frame_sha256': digest(sample.rgb), 'spikes_sha256': digest(counts),
                          'memory': _memory(brain)})
            pending_current = 4 * error if training else 0.
        if len(trace) != expected:
            raise ValueError('Frame iterator ended before original boundary')
    except Exception as error:
        failure = error
    finally:
        if frames is not None:
            try:
                frames.close()
            except Exception as error:
                if failure is None:
                    failure = error
                else:
                    close_failure = error
    if failure is not None:
        raise EpisodeFailure(summary(failure)) from failure
    return summary()


def _invariants(brain):
    """Hash all nonplastic slots without retaining a second full weight array."""
    edges = np.sort(brain.circuit['edges'])
    value = hashlib.sha256()
    for start in range(0, len(brain.weight), 1000000):
        end = min(start + 1000000, len(brain.weight))
        selected = edges[(edges >= start) & (edges < end)] - start
        block = brain.weight[start:end]
        if len(selected):
            mask = np.ones(len(block), dtype=bool); mask[selected] = False
            block = block[mask]
        value.update(block.tobytes())
    return {'nonplastic_weight_sha256': value.hexdigest(), 'ptr_sha256': digest(brain.ptr),
            'post_sha256': digest(brain.post), 'ids_sha256': digest(brain.ids),
            'plastic_slots_sha256': digest(brain.circuit['edges'])}


def _cohort(brain, train, held_out, readouts, args, out, initial):
    rows = []; invariant = _invariants(brain)
    def execute(data, directory, arm, phase, *, training=False, frozen=True, shift=0, retention=None):
        directory.mkdir(parents=True)
        record = None
        try:
            if _invariants(brain) != invariant:
                raise ValueError('Immutable graph or nonplastic weights changed')
            record = episode(brain, data, readouts, training=training, frozen=frozen,
                             target_shift=shift, limit=args.max_frames)
            if _invariants(brain) != invariant:
                raise ValueError('Immutable graph or nonplastic weights changed')
        except EpisodeFailure as error:
            record = error.summary
            record.update(arm=arm, phase=phase)
            save_json(directory / 'episode.json', record)
            brain.checkpoint(directory / 'failure.npz')
            save_json(out / 'failure.json', {'arm': arm, 'phase': phase, 'failure': record['failure'],
                                           'completed_episodes': len(rows), 'episodes': rows})
            raise
        except Exception as error:
            if record is not None:
                record.update(complete=False, failure={'type': type(error).__name__},
                              arm=arm, phase=phase, invariants_unchanged=False)
                save_json(directory / 'episode.json', record)
            brain.checkpoint(directory / 'failure.npz')
            save_json(directory / 'failure.json', {'type': type(error).__name__})
            raise
        record.update(arm=arm, phase=phase, invariants_unchanged=True)
        if retention is not None:
            record['retention'] = retention
        save_json(directory / 'episode.json', record)
        slim = {k: v for k, v in record.items() if k != 'trace'}
        rows.append(slim)
        save_json(out / 'progress.json', {'completed_episodes': len(rows), 'latest': slim})
        return record

    for arm in ['plastic', 'frozen', 'shifted']:
        brain.restore(initial)
        branch = out / arm; branch.mkdir()
        for epoch in range(args.epochs):
            for index, data in enumerate(train):
                shift = (args.target_shift if args.target_shift is not None else max(1, data.frame_count // 2)) if arm == 'shifted' else 0
                execute(data, branch / f'train-{epoch}-{index}', arm, 'training',
                        training=True, frozen=arm == 'frozen', shift=shift)
        learned = branch / 'learned.npz'; brain.checkpoint(learned)
        for index, data in enumerate(held_out):
            brain.restore(learned)
            execute(data, branch / f'eval-{index}', arm, 'held_out')
        if arm == 'plastic':
            for index, data in enumerate(held_out):
                brain.restore(learned); brain.weights_frozen = False
                before = _memory(brain); clock = brain.cursor; started = time.perf_counter()
                _, kernel = brain.rgb_step(np.zeros((data.height, data.width, 3), dtype=np.uint8),
                                           5000, learning=False, stimulation=None)
                retention = {'brain_steps': brain.cursor - clock, 'brain_seconds': 5.,
                             'wall_seconds': time.perf_counter() - started, 'kernel_seconds': kernel,
                             'weights_frozen': False, 'learning': False, 'teacher_current': 0.,
                             'before': before, 'after': _memory(brain)}
                execute(data, branch / f'retention-{index}', arm, 'retention_5s', retention=retention)
                brain.restore(initial); brain.reset(keep_memory=False)
                execute(data, branch / f'erased-{index}', arm, 'memory_erased')
    if _invariants(brain) != invariant:
        raise ValueError('Immutable graph or nonplastic weights changed')
    save_json(out / 'invariants.json', {'before': invariant, 'after': _invariants(brain), 'passed': True})
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='append', required=True)
    parser.add_argument('--eval', action='append', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eta', type=float, default=.001)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--target-shift', type=int)
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='cpu')
    parser.add_argument('--metal-validation')
    return parser


def run(args):
    """Validate the entire cohort before constructing the unchanged full model."""
    _positive_integer(args.epochs, 'Epoch count')
    if args.max_frames is not None:
        _positive_integer(args.max_frames, 'Frame limit')
    if not math.isfinite(args.eta) or args.eta < 0:
        raise ValueError('Finite nonnegative eta required')
    if not args.train or not args.eval:
        raise ValueError('Nonempty training and evaluation cohorts required')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Fresh output directory required')
    train = [OfflineEpisode(path) for path in args.train]
    held_out = [OfflineEpisode(path) for path in args.eval]
    all_data = train + held_out
    for data in all_data:
        _targets(data)
    for key in ['video_sha256', 'control_sha256']:
        hashes = [data.identity[key] for data in all_data]
        if len(set(hashes)) != len(hashes):
            raise ValueError('Training/evaluation overlap or duplicate artifact hashes')
    for data in train:
        shift = args.target_shift if args.target_shift is not None else max(1, data.frame_count // 2)
        if _shift(data, shift) == 0:
            raise ValueError('Shifted-label arm requires a nonzero effective offset')
    from .metal.benchmark import (_file_digest, current_preflight_identity, model_identity,
                                  portable_backend_metadata, require_metal_validation)
    validation = expected = None
    if args.backend == 'metal':
        expected = current_preflight_identity()
        validation = require_metal_validation(args.metal_validation, expected)
    readouts = json.loads((GRAPH.parent / 'manifest.json').read_text())['readouts']
    out.mkdir(parents=True)
    # JSONL is ignored by repository policy; machine origins stay in this record.
    with (out / 'inputs.jsonl').open('w', encoding='utf-8') as stream:
        for phase, paths, data_list in [('train', args.train, train), ('eval', args.eval, held_out)]:
            for path, data in zip(paths, data_list):
                stream.write(json.dumps({'phase': phase, 'directory': str(Path(path).resolve()),
                                         'identity': _identity(data)}) + '\n')
        if args.metal_validation:
            stream.write(json.dumps({'metal_validation': str(Path(args.metal_validation).resolve())}) + '\n')
    brain = None
    try:
        capture_provenance(out, additional=['doom_learning_v2', 'doom_learning_v6'])
        brain = calibrated_brain(args.eta, backend=args.backend)
        if (brain.n, len(brain.weight), len(brain.circuit['edges'])) != (166700, 25582938, 4184):
            raise ValueError('Exact full retained MaleCNS graph and existing plastic slots required')
        if validation is not None:
            validation = require_metal_validation(args.metal_validation, {**expected, **model_identity(brain)})
        execution = {'backend': args.backend, 'backend_metadata': portable_backend_metadata(brain.backend.metadata()),
                     'metal_validation_sha256': _file_digest(args.metal_validation) if validation else None,
                     'metal_validation_horizon_ms': validation['validation_horizon_ms'] if validation else None}
        protocol = {'schema': 1, 'model': 'adaptive-centered-v6', 'connectome': 'MaleCNS v1.0',
                    'neurons': brain.n, 'retained_edges': len(brain.weight), 'plastic_slots': len(brain.circuit['edges']),
                    'model_identity': model_identity(brain), 'configuration': brain.configuration_signature(),
                    'execution': execution, 'calibration': brain.calibration,
                    'training': [_identity(data) for data in train], 'evaluation': [_identity(data) for data in held_out],
                    'epochs': args.epochs, 'eta': args.eta, 'diagnostic_limit': args.max_frames,
                    'arms': ['plastic', 'frozen', 'shifted'], 'readouts': readouts,
                    'shift_offsets': [_shift(data, args.target_shift if args.target_shift is not None else max(1, data.frame_count // 2)) for data in train],
                    'teacher': 'After decoding, min(abs(student-target)/6,1); next original frame gets 4*error at PPL101. First frame and all evaluations receive zero teacher current.',
                    'shift': 'Circular positive np.roll turn targets only. Original RGB/raw controls unchanged; feedback dose can differ and is measured.',
                    'timing': {'source_hz': 35, 'neural_dt_ms': .1, 'warmup_ms': 2000, 'retention_ms': 5000,
                               'frame_end_steps': 'round((index+1)*10000/35)'},
                    'evaluation_mode': 'Teacher-free frozen efficacies, learned full checkpoint restored independently per episode; decoder and fast state reset.',
                    'retention_mode': 'Unfrozen efficacies; no teacher or learning during five seconds dark, then frozen teacher-free evaluation.',
                    'target_calibration': 'Local ViZDoom 1.3.0: 320 yaw units for held count below six, then 640; 360/65536 degrees per unit. Original collector engine provenance unresolved.',
                    'claim_gate': 'Exploratory turn-only reinforcement. Physiological validation failed/pending. Changed weights and numeric controls establish no biological learning or held-out gameplay; hazard-arena transfer and scientific launch gates remain required.'}
        save_json(out / 'protocol.json', protocol)
        save_json(out / 'circuit.json', brain.circuit['report'])
        save_json(out / 'visual.json', brain.visual_report)
        initial = out / 'initial.npz'; brain.checkpoint(initial)
        rows = _cohort(brain, train, held_out, readouts, args, out, initial)
        result = {'complete': all(row['complete'] for row in rows), 'protocol': protocol, 'episodes': rows,
                  'learning_demonstrated': False, 'announcement_ready': False}
        save_json(out / 'results.json', result)
        return result
    except Exception as error:
        if brain is not None:
            brain.checkpoint(out / 'failure.npz')
        if not (out / 'failure.json').exists():
            save_json(out / 'failure.json', {'complete': False, 'type': type(error).__name__,
                                            'learning_demonstrated': False})
        raise
    finally:
        if brain is not None:
            brain.backend.close()


if __name__ == '__main__':
    require_single_blas_thread()
    run(build_parser().parse_args())
