"""Bounded frozen efficacy fitting with complete resident-lane evidence.

Replay, neural propagation and fixed decoding remain in causal_pilot. This
module owns only the pinned numerical schedule, accounting and provenance.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import numpy as np

from doom_learning.common import digest, save_json
from .causal_pilot import (
    _Resources, _PressureHistory, _phase_boundary, _model_gate, _readouts,
    _compare_checkpoint_arrays, _same_without_wall, _physical_pins,
    _validate_expected_pins, _source_clean, _git_output, _acquire_gpu_lock,
    _file_pin, _save_failure, _restored_brains,
)
from .causal_pilot import replay_episode
from .causal_controls import diagnostic_score
from .imitation import _invariants
from .same_slot_optimizer import (
    AttemptBudget, load_protocol, fractions, optimizer_rng, draw_direction,
    propose, training_mean, accept_proposal, _PINNED_PROTOCOL_SHA256,
)


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = _ROOT / 'docs/experiments/2026-09-13-same-slot-optimizer-protocol.json'


def _exact_protocol(value, reference):
    """Compare exact JSON protocol types and values, including object keys."""
    if type(value) is not type(reference):
        return False
    if isinstance(reference, dict):
        return (all(type(key) is str for key in value) and value.keys() == reference.keys() and
                all(_exact_protocol(value[key], item) for key, item in reference.items()))
    if isinstance(reference, list):
        return len(value) == len(reference) and all(_exact_protocol(a, b) for a, b in zip(value, reference))
    return value == reference


def _contract(train, held, readouts, protocol):
    if not _exact_protocol(protocol, load_protocol(_PROTOCOL, _PINNED_PROTOCOL_SHA256)):
        raise ValueError('Exact committed fitting protocol required')
    if len(train) != 2 or len(held) != 2:
        raise ValueError('Exactly two complete training and two held episodes required')
    for episode, row in zip([*train, *held], protocol['episodes']):
        identity = dict(dataset=protocol['dataset'], dataset_revision=protocol['dataset_revision'],
                        scenario=protocol['scenario'], episode_index=row['index'],
                        video_sha256=row['video_sha256'], control_sha256=row['control_sha256'])
        if type(episode.frame_count) is not int or episode.frame_count != row['frames'] or episode.identity != identity:
            raise ValueError('Fresh complete episode identity/count differs from protocol')
    if (len(readouts) != 14 or len({row['index'] for row in readouts}) != 14 or
            any(sum(row['type'] == kind and row.get('side') == side for row in readouts) != 1
                for kind in ('DNp20', 'DNpe017') for side in ('L', 'R'))):
        raise ValueError('Require 14 traced readouts and fixed four-neuron BCI decoder')


def _disk(out, minimum=15):
    if shutil.disk_usage(out).free < minimum * 1024 ** 3:
        raise ValueError('At least 15 GiB free disk required')


def _array(out, relative, values):
    path = out / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float64), allow_pickle=False)
    return {'path': str(path.relative_to(out)), **_file_pin(path)}


def _frozen(brain):
    brain.backend.materialize('same-slot-frozen-full-weight')
    return {'weight': digest(brain.weight), 'u': digest(brain.memory_u),
            'w': digest(brain.memory_w), 'fixed': _invariants(brain),
            'configuration': brain.configuration_signature()}


def _storage(executor):
    metadata = executor.metadata()
    if metadata.get('window_ticks') != 18 or metadata.get('lane_count') != 4:
        raise ValueError('Exact four-lane W18 resident executor required')
    return {key: metadata[key] for key in ('shared_resident_bytes', 'mutable_resident_bytes')}


def _pressure(out):
    pressure = _PressureHistory(out)
    path = Path(out) / 'pressure.json'
    if path.exists():
        pressure.observations = json.loads(path.read_text())['observations']
        accepted = [row['value'] for row in pressure.observations if row['accepted']]
        pressure.previous = accepted[-1] if accepted else None
    return pressure


def _failed(out, failure, pressure):
    # A late pin or resource-cleanup failure invalidates an already written
    # fitting result. Keep the original exception even when this writer fails.
    try:
        pressure = _pressure(out)
    except BaseException as secondary:
        failure.add_note(f'Secondary pressure evidence reload failure: {type(secondary).__name__}: {secondary}')
    if (Path(out) / 'results.json').exists():
        try:
            save_json(Path(out) / 'results.json', {'complete': False, 'learning_demonstrated': False,
                                                   'announcement_ready': False, 'failure_type': type(failure).__name__})
        except BaseException as secondary:
            failure.add_note(f'Secondary result invalidation failure: {type(secondary).__name__}: {secondary}')
    _save_failure(out, failure, pressure)


def fit(executor, train, held, readouts, protocol, out, *, replay=replay_episode):
    """Run the exact 30-wave schedule; replay is the sole injectable boundary."""
    out = Path(out)
    _contract(train, held, readouts, protocol)
    brains = tuple(executor.brains)
    if len(brains) != 4 or len({id(brain) for brain in brains}) != 4:
        raise ValueError('Exactly four distinct resident lanes required')
    if type(executor.window_ticks) is not int or executor.window_ticks != protocol['window_ticks']:
        raise ValueError('Exact protocol W18 executor required')
    if (out / 'attempt-ledger.json').exists():
        raise ValueError('Fresh fitting evidence directory required')
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    pressure = _pressure(out)
    budget, waves = AttemptBudget(protocol['budget']['maximum']), []
    names = [f'{method}-{seed}' for method in protocol['methods'] for seed in protocol['optimizer_seeds']]
    centers = {name: np.zeros(protocol['plastic_slots'], dtype=np.float64) for name in names}
    rngs = {f'{method}-{seed}': optimizer_rng(method, seed)
            for method in protocol['methods'] for seed in protocol['optimizer_seeds']}
    scores, fixed = {}, []

    def ledger():
        save_json(out / 'attempt-ledger.json', {'maximum': budget.maximum, 'used': budget.used,
                                               'remaining': budget.remaining, 'waves': waves})

    def center(name):
        _array(out, f'incumbents/{name}/theta.npy', centers[name])
        _array(out, f'incumbents/{name}/fractions.npy', fractions(centers[name]))
        save_json(out / 'incumbents' / name / 'score.json', {'training_loss': scores.get(name)})

    def wave(episode, candidates, roles, *, kind, generation=None, signs=None):
        if len(candidates) != 4 or len(roles) != 4:
            raise ValueError('Four candidate lanes required, including fillers')
        checked = [np.asarray(value, dtype=np.float64) for value in candidates]
        if any(value.shape != (4184,) or not np.isfinite(value).all() or
               np.any(value < .1) or np.any(value > 2.) for value in checked):
            raise ValueError('Invalid scheduled efficacy candidate')
        directory = out / 'waves' / f'w-{len(waves):02d}'
        _disk(out)
        entry = {'kind': kind, 'generation': generation, 'episode_index': episode.identity['episode_index'],
                 'roles': list(roles), 'signs': signs, 'state': 'charged',
                 'directory': str(directory.relative_to(out)),
                 'used': budget.charge(4, reserve_after=16 if kind != 'held' else 0)}
        waves.append(entry)
        ledger()  # Durable charge precedes every install, reset and native start.
        entry['candidate_fractions'] = [_array(out, f'candidates/{directory.name}/lane-{lane}.npy', value)
                                        for lane, value in enumerate(checked)]
        ledger()
        with _phase_boundary(pressure, directory.name):
            storage_before = _storage(executor)
            for lane, (brain, value) in enumerate(zip(brains, checked)):
                executor.install_efficacies(lane, value, brain.circuit['edges'])
            expected = [_frozen(brain) for brain in brains]
            for lane, (brain, value, state) in enumerate(zip(brains, checked, expected)):
                weights = (brain.baseline_plastic.astype(np.float64) * value).astype(np.float32)
                if (digest(brain.weight[brain.circuit['edges']]) != digest(weights) or
                        state['u'] != digest(value - 1.) or state['w'] != digest(value - 1.) or
                        any(state[key] != fixed[lane][key] for key in ('fixed', 'configuration'))):
                    raise ValueError('Installed candidate differs or fixed graph/configuration changed')
            epochs = [(brain.backend._host_weight_epoch, brain.backend._device_weight_epoch) for brain in brains]
            started = time.perf_counter()
            phase = replay(brains, executor, episode, readouts, np.zeros((4, episode.frame_count)),
                           learning=[False] * 4, frozen=[True] * 4, directory=directory,
                           warmup_ms=protocol['warmup_ms'])
            elapsed = time.perf_counter() - started
            if not phase.get('complete') or len(phase.get('lanes', [])) != 4:
                raise ValueError('Complete four-lane phase required')
            runtime = []
            targets = np.asarray(episode.turn_targets(), dtype=np.float64)
            for lane, (brain, record) in enumerate(zip(brains, phase['lanes'])):
                save_json(directory / f'lane-{lane}' / 'episode.json', record)
                if (not record.get('complete') or record.get('frames') != episode.frame_count or
                        record.get('source_frame_count') != episode.frame_count or
                        record.get('identity') != episode.identity or record.get('weights_frozen') is not True or
                        len(record['trace']) != episode.frame_count or
                        any(row['index'] != index or row['teacher_current'] != 0.
                            for index, row in enumerate(record['trace']))):
                    raise ValueError('Incomplete or nonzero-reinforcement lane record')
                if not np.array_equal([row['target_turn'] for row in record['trace']], targets):
                    raise ValueError('Scoring labels differ from original complete episode')
                observed = _frozen(brain)
                if observed != expected[lane]:
                    raise ValueError('Frozen full weights/u/w or fixed configuration changed')
                after = (brain.backend._host_weight_epoch, brain.backend._device_weight_epoch)
                # Replay has one full reset upload and one retained sparse reinstall.
                # Any additional epoch is an unauthorized runtime weight update.
                if after != (epochs[lane][0] + 2, epochs[lane][1] + 2):
                    raise ValueError('Unexpected runtime weight upload epoch delta')
                runtime.append({'frozen_before': expected[lane], 'frozen_after': observed,
                                'epochs_before_replay': epochs[lane], 'epochs_after_replay': after,
                                'required_setup_uploads': 2, 'rgb_sparse_updates': 0,
                                'phase_timing': record.get('phase_timing')})
                checkpoint = directory / f'lane-{lane}' / 'final.npz'
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if len(set(saved.files) - {'metadata'}) != 24:
                        raise ValueError('Every terminal checkpoint must retain all 24 arrays')
                    if any(digest(saved[key]) != observed[expected_key] for key, expected_key in
                           [('weight', 'weight'), ('memory_u', 'u'), ('memory_w', 'w')]):
                        raise ValueError('Saved frozen weight/u/w bytes differ from resident materialization')
            storage_after = _storage(executor)
            if storage_after != storage_before:
                raise ValueError('Native resident storage changed during a frozen wave')
            save_json(directory / 'runtime-evidence.json', {'replay_wall_seconds': elapsed, 'lanes': runtime,
                                                          'resident_storage_before': storage_before,
                                                          'resident_storage_after': storage_after,
                                                          'setup_upload_scope': 'reset plus retained-slot reinstall'})
            if kind == 'baseline':
                for lane in range(1, 4):
                    if not _same_without_wall(phase['lanes'][0]['trace'], phase['lanes'][lane]['trace']):
                        raise ValueError('Baseline canonical cross-lane trace differs')
                    _compare_checkpoint_arrays(directory / 'lane-0/final.npz', directory / f'lane-{lane}/final.npz')
            loss = [diagnostic_score([row['action']['turn'] for row in record['trace']],
                                     [row['target_turn'] for row in record['trace']])['balanced_mae_degrees']
                    for record in phase['lanes']]
            if not np.isfinite(loss).all() or min(loss) < 0:
                raise ValueError('Finite nonnegative complete-episode losses required')
            _model_gate(brains)
            _disk(out)
        entry.update(state='complete', losses=loss)
        ledger()
        return loss

    try:
        _disk(out)
        _model_gate(brains)
        fixed = [_frozen(brain) for brain in brains]
        ledger()
        save_json(out / 'protocol.json', protocol)
        save_json(out / 'inputs.json', {'training': [d.identity for d in train], 'held': [d.identity for d in held],
                                       'readouts': readouts, 'decoder_mode': 'bci', 'teacher_current': 0})
        for name in names:
            center(name)
        baseline = [wave(episode, [np.ones(4184)] * 4, ['baseline'] * 4, kind='baseline') for episode in train]
        initial_score = training_mean([loss[0] for loss in baseline])
        scores.update({name: initial_score for name in names})
        save_json(out / 'baseline-scores.json', {'losses': [loss[0] for loss in baseline], 'mean': initial_score,
                                                'cross_lane_trace_and_all24_exact': True})
        for name in names:
            center(name)
        for generation in range(protocol['generations']):
            order = names[generation:] + names[:generation]
            scale, step = protocol['probe_scales'][generation], protocol['step_lengths'][generation]
            directions = {name: draw_direction(name.rsplit('-', 1)[0], rngs[name], 4184) for name in names}
            draws = {}
            for name in names:
                draws[name] = _array(out, f'generations/g-{generation}/{name}/direction.npy', directions[name])
                _array(out, f'generations/g-{generation}/{name}/incumbent-theta.npy', centers[name])
                _array(out, f'generations/g-{generation}/{name}/incumbent-fractions.npy', fractions(centers[name]))
                for sign, multiplier in [('plus', 1), ('minus', -1)]:
                    theta = centers[name] + multiplier * scale * directions[name]
                    _array(out, f'generations/g-{generation}/{name}/{sign}-theta.npy', theta)
                    _array(out, f'generations/g-{generation}/{name}/{sign}-fractions.npy', fractions(theta))
            paired = {name: {'+': [], '-': []} for name in names}
            for pair in (order[:2], order[2:]):
                roles = [name for name in pair for _ in range(2)]
                signs = ['+', '-', '+', '-']
                candidates = [fractions(centers[name] + (scale if sign == '+' else -scale) * directions[name])
                              for name, sign in zip(roles, signs)]
                for episode in train:
                    losses = wave(episode, candidates, roles, kind='probe', generation=generation, signs=signs)
                    for name, sign, loss in zip(roles, signs, losses):
                        paired[name][sign].append(loss)
            proposals, decisions = {}, {}
            for name in names:
                proposals[name], detail = propose(centers[name], directions[name],
                    training_mean(paired[name]['+']), training_mean(paired[name]['-']), probe_scale=scale, step_length=step)
                theta = _array(out, f'generations/g-{generation}/{name}/proposal-theta.npy', proposals[name])
                values = _array(out, f'generations/g-{generation}/{name}/proposal-fractions.npy', fractions(proposals[name]))
                decisions[name] = {'run': name, 'generation': generation, 'probe_scale': scale, 'step_length': step,
                                   'paired_losses': paired[name], 'direction': draws[name], 'theta': theta,
                                   'fractions': values, 'incumbent_loss': scores[name], **detail}
            save_json(out / 'generations' / f'g-{generation}' / 'proposal-plan.json', [decisions[name] for name in names])
            proposal_losses = [wave(episode, [fractions(proposals[name]) for name in order], order,
                                    kind='proposal', generation=generation) for episode in train]
            for lane, name in enumerate(order):
                loss = training_mean([bank[lane] for bank in proposal_losses])
                accepted = accept_proposal(scores[name], loss)
                decisions[name].update(proposal_loss=loss, accepted=accepted)
                if accepted:
                    centers[name], scores[name] = proposals[name], loss
                center(name)
            save_json(out / 'generations' / f'g-{generation}' / 'decisions.json', [decisions[name] for name in names])
        finals = [{'role': 'baseline', 'values': np.ones(4184), 'training_loss': initial_score}]
        finals += [{'role': name, 'values': fractions(centers[name]), 'training_loss': scores[name]} for name in names]
        finals += [{'role': f'random-{seed}', 'seed': seed,
                    'values': np.random.Generator(np.random.PCG64(seed)).uniform(.9, 1.1, size=4184)}
                   for seed in protocol['random_control_seeds']]
        manifest = []
        for candidate in finals:
            pinned = _array(out, f'finals/{candidate["role"]}/fractions.npy', candidate['values'])
            manifest.append({key: value for key, value in candidate.items() if key != 'values'} |
                            {'fractions': pinned['path'], 'pin': pinned})
        save_json(out / 'final-candidates.json', {'frozen_before_held': True, 'candidates': manifest})
        for group in (finals[:4], finals[4:] + [{'role': 'baseline-filler', 'values': np.ones(4184)}]):
            for episode in held:
                wave(episode, [row['values'] for row in group], [row['role'] for row in group], kind='held')
        result = {'complete': True, 'learning_demonstrated': False, 'announcement_ready': False,
                  'runs': names, 'attempts': budget.used, 'planned': 120, 'reserve_unused': 8,
                  'final_candidates': 'final-candidates.json', 'phase_pressure': pressure.observations,
                  'scope': 'Frozen numerical fitting; it does not establish fly learning.'}
        save_json(out / 'results.json', result)
        return result
    except BaseException as failure:
        availability = []
        if waves:
            replay_availability = getattr(failure, '_doomfly_replay_availability', None)
            current_phase = out / waves[-1]['directory']
            current_replay = (isinstance(replay_availability, dict) and
                              replay_availability.get('directory') == str(current_phase.resolve()) and
                              replay_availability.get('lane_ids') == tuple(map(id, brains)))
            for lane, brain in enumerate(brains):
                directory = current_phase / f'lane-{lane}'
                existing = [path for path in (directory / 'failure.npz', directory / 'final.npz') if path.exists()]
                path = existing[0] if existing else directory / 'failure.npz'
                record = {'lane': lane, 'path': str(path.relative_to(out)),
                          'scope': 'already produced replay state' if existing else 'best-effort interrupted resident state'}
                decision = replay_availability['lanes'][lane] if current_replay else None
                if decision is not None and decision.get('terminal_state_available') is False:
                    record.update(decision, all24_available=False,
                                  scope='replay declared terminal state unavailable',
                                  reason=decision['terminal_state_error']['type'] + ': terminal materialization unavailable')
                    availability.append(record)
                    continue
                try:
                    if not existing:
                        # Owned checkpoint materialization rejects a poisoned
                        # executor. Never reset/restore or upload stale host state.
                        brain.checkpoint(path)
                    with np.load(path, allow_pickle=False) as checkpoint:
                        if len(set(checkpoint.files) - {'metadata'}) != 24:
                            raise ValueError('Interrupted checkpoint lacks all 24 arrays')
                    record['all24_available'] = True
                except BaseException as secondary:
                    record.update(all24_available=False, reason=f'{type(secondary).__name__}: {secondary}')
                    failure.add_note(f'Secondary lane-{lane} failure checkpoint: {type(secondary).__name__}: {secondary}')
                availability.append(record)
        try:
            ledger()
            save_json(out / 'failure-checkpoints.json', {'lanes': availability})
            save_json(out / 'interrupted-progress.json', {'attempts': budget.used,
                'last_completed_training_scores': scores,
                'incumbents': {name: f'incumbents/{name}/theta.npy' for name in names}})
        except BaseException as secondary:
            failure.add_note(f'Secondary fitting progress writer failure: {type(secondary).__name__}: {secondary}')
        _failed(out, failure, pressure)
        raise


def _initial_reference(reference, observed):
    """Validate original static model relationships without historical replay."""
    root = Path(reference)
    original = json.loads((root / 'protocol.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance['graph_sha256'] != observed['inputs']['outputs/doom/malecns_v1/graph.npz']['sha256']:
        raise ValueError('Original graph physical identity differs')
    identity = original['model_identity']
    with np.load(_ROOT / 'outputs/doom/malecns_v1/graph.npz', allow_pickle=False) as graph:
        for key, expected in [('weight', 'original_weight_sha256'), ('ids', 'ids_sha256'),
                              ('ptr', 'out_ptr_sha256'), ('post', 'out_post_sha256')]:
            if digest(graph[key]) != identity[expected]:
                raise ValueError('Original released graph array identity differs')
        if len(graph['ids']) != 166700 or len(graph['weight']) != 25582938:
            raise ValueError('Full retained graph required')
    allowed = {'doom_learning_v6/brain.py', 'doom_learning_v6/visual.py', 'doom_learning_v6/metal/api.h',
               'doom_learning_v6/metal/backend.mm', 'doom_learning_v6/metal/build.py', 'doom_learning_v6/metal/kernels.metal'}
    transitions = {}
    for name, previous in provenance['source_sha256'].items():
        current = observed['sources'].get(name, {}).get('sha256')
        if previous != current:
            if name not in allowed or current is None:
                raise ValueError(f'Original measured rule/sensory/calibration source differs: {name}')
            transitions[name] = {'original': previous, 'current': current}
    with np.load(root / 'initial.npz', allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint['metadata']))
        if len(set(checkpoint.files) - {'metadata'}) != 24:
            raise ValueError('Original initial checkpoint must contain all 24 arrays')
        for key, expected in [('model', original['model']), ('eta', .001), ('configuration_sha256', original['configuration']),
                              ('graph_ids_sha256', identity['ids_sha256']), ('graph_ptr_sha256', identity['out_ptr_sha256']),
                              ('graph_post_sha256', identity['out_post_sha256']),
                              ('plastic_edges_sha256', identity['plastic_edges_sha256'])]:
            if metadata.get(key) != expected:
                raise ValueError(f'Original initial checkpoint relationship differs: {key}')
        if (metadata['cursor'] != 0 or digest(checkpoint['weight']) != original['configuration']['initial_weight'] or
                np.any(checkpoint['memory_u']) or np.any(checkpoint['memory_w']) or
                metadata['build']['source_sha256'] != provenance['source_sha256']['doom_learning_v6/kernel.cpp'] or
                metadata['build']['flags'] != ['-O3', '-std=c++17', '-shared', '-fPIC'] or
                metadata.get('producer_backend') != 'metal' or metadata['backend']['abi_version'] != 5 or
                any(metadata['backend'][key] != original['execution']['backend_metadata'][key]
                    for key in ('sources', 'binaries', 'build_configuration'))):
            raise ValueError('Original initial state/CPU/native relationship differs')
    configuration_hash = hashlib.sha256(json.dumps(original['configuration'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if configuration_hash != identity['configuration_sha256']:
        raise ValueError('Original configuration hash differs')
    return {'protocol': original, 'initial_pin': _file_pin(root / 'initial.npz'), 'source_transitions': transitions,
            'scope': 'Initial static relationships only; historical episodes are never replayed.'}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'protocol-sha256', 'reference', 'expected-pins', 'source-commit', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--train', action='append', required=True)
    parser.add_argument('--eval', action='append', required=True)
    return parser


def run(args):
    """Preflight identities before native ownership, then retain one GPU lease."""
    out = Path(args.out)
    if out.exists():
        raise ValueError('Fresh output directory required')
    try:
        out.resolve().relative_to(_ROOT.resolve())
    except ValueError:
        pass  # Fresh external private output stays outside source Git/archives.
    else:
        try:
            ignored = _git_output('check-ignore', '--', str(out.resolve()))
        except Exception as error:
            raise ValueError('In-repository output directory must be ignored') from error
        if not ignored:
            raise ValueError('In-repository output directory must be ignored')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit) or _git_output('rev-parse', 'HEAD') != args.source_commit:
        raise ValueError('Source commit mismatch')
    if sys.platform != 'darwin' or platform.system() != 'Darwin':
        raise ValueError('Resident same-slot fitting requires macOS Metal')
    _source_clean()
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    committed = _git_output('show', args.source_commit + ':docs/experiments/2026-09-13-same-slot-optimizer-protocol.json')
    if hashlib.sha256((committed + '\n').encode()).hexdigest() != args.protocol_sha256:
        raise ValueError('Protocol is not the exact committed source protocol')
    from .demonstrations import OfflineEpisode
    train, held = [OfflineEpisode(path) for path in args.train], [OfflineEpisode(path) for path in args.eval]
    readouts = _readouts()
    _contract(train, held, readouts, protocol)
    observed = _physical_pins(args.reference)
    expected = _validate_expected_pins(args.expected_pins, observed)
    relationships = _initial_reference(args.reference, observed)
    if readouts != relationships['protocol']['readouts']:
        raise ValueError('Exact ordered original 14 readouts/BCI neurons required')
    _disk(out.parent)
    lock = _acquire_gpu_lock()
    failure = None
    pressure = _pressure(out)
    try:
        out.mkdir(parents=True, mode=0o700)
        save_json(out / 'expected-pins.json', expected)
        save_json(out / 'observed-pins.json', observed)
        save_json(out / 'reference-validation.json', relationships)
        save_json(out / 'provenance.json', {'arguments': vars(args), 'source_commit': args.source_commit,
                                           'protocol_pin': _file_pin(args.protocol), 'pid': os.getpid()})
        pressure.observe('before_native_allocation')
        from .calibration import calibrated_brain
        from .metal.batch import MetalBatchExecutor
        with _Resources() as resources:
            brains = _restored_brains(resources, lambda: calibrated_brain(.001, backend='cpu'),
                                      Path(args.reference) / 'initial.npz', 4)
            _model_gate(brains, relationships['protocol'])
            for lane, brain in enumerate(brains):
                checkpoint = out / 'initial' / f'lane-{lane}.npz'
                brain.checkpoint(checkpoint)
                _compare_checkpoint_arrays(Path(args.reference) / 'initial.npz', checkpoint)
            executor = resources.add(MetalBatchExecutor(brains, window_ticks=18))
            result = fit(executor, train, held, readouts, protocol, out)
        pressure = _pressure(out)
        pressure.observe('after_native_release')
        after = _physical_pins(args.reference)
        _validate_expected_pins(args.expected_pins, after)
        save_json(out / 'observed-pins-after.json', after)
        result.update(source_commit=args.source_commit, physical_identity_records=[
            'expected-pins.json', 'observed-pins.json', 'observed-pins-after.json', 'reference-validation.json'])
        save_json(out / 'results.json', result)
    except BaseException as error:
        failure = error
        if out.exists():
            _failed(out, failure, pressure)
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
                _failed(out, failure, pressure)
            raise failure
    return result


def main():
    from doom_learning.common import require_single_blas_thread
    require_single_blas_thread()
    print(json.dumps(run(build_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
