# Metal Long-Horizon Diagnostic Invocation

## Question and scope

Saved before execution. At source `a5d31453ae2a5eca82cb39d8a40d0643a80cb477`,
the passing 80 ms parity trace does not extend to the checked training prefix.
With identical model configuration, original decoded RGB and timing, frozen
CPU/Metal outputs differ on all 35 held-out frames. First turn is +1.04365336
on CPU and zero on Metal. The eight Metal phases finish with unchanged graph
invariants and teacher-free frozen evaluations; this is a failed longer parity
control, not learning evidence. Full Metal cohort has not been started.

This diagnostic keeps production sources unchanged. Compare artificial input
association, repeated subthreshold state evolution and identical full-graph
10 ms dark bins, then the first original recorded RGB frame. Locate first
state/spike divergence. Explicit state downloads perturb timing, so these are
diagnostic timings only. Artificial micrographs test arithmetic and provide
no circuit or biological validation.

The actual M4 CPU binary contains fused multiply-add instructions; Metal uses
the retained `-ffp-contract=off` setting. Further suspected differences are
incoming conductance association/order and
evolution on exclusively modulatory arrivals. Both CPU and Metal already
settle all fast states at each native observation boundary; adding another
settlement dispatch would duplicate existing work. CPU source, all retained
connections, plastic slots, dynamics, decoder and scientific gates stay fixed.
All machine origins and downloaded artifacts are in ignored configuration.

## Invocation

Run this code with the existing single-threaded environment on the M4 Pro in
an owned exact-source checkout after the prefix process finishes naturally.
Require a fresh output directory. Preserve all results, including failures.

```python
import gc, hashlib, json, pathlib, tempfile
import numpy as np
from doom_learning.common import GRAPH, digest, save_json
from doom_learning_v6.brain import MemoryBrain
from doom_learning_v6.calibration import calibrated_brain
from doom_learning_v6.demonstrations import OfflineEpisode
from doom_learning_v6.metal.validate import source_identity, spike_metrics
from doom.engine import NeuralControls

out = pathlib.Path('outputs/doom-learning/metal-long-horizon-diagnostic-20260913')
if out.exists():
    raise ValueError('Fresh output directory required')
out.mkdir(parents=True)
configuration = json.loads(pathlib.Path(__import__('os').environ['DOOMFLY_DIAGNOSTIC_INPUTS']).read_text())
save_json(out / 'protocol.json', {
    'schema': 1, 'source_commit': 'a5d31453ae2a5eca82cb39d8a40d0643a80cb477',
    'validation_sources': source_identity(), 'warmup_ms': 2000,
    'bin_steps': 100, 'learning': False, 'teacher_current': 0,
    'full_graph_neurons': 166700, 'full_graph_edges': 25582938,
    'timing': 'Diagnostic downloads and capture; not a throughput benchmark',
    'learning_demonstrated': False, 'announcement_ready': False})

def differences(cpu, metal):
    result = {}
    for name in ['v', 'g', 'adaptation', 'refractory', 'last', 'modulation',
                 'modulation_last', 'active_flag', 'counts']:
        a, b = getattr(cpu, name), getattr(metal, name)
        different = np.flatnonzero(a != b)
        result[name] = {'different': len(different),
            'maximum_absolute_error': float(np.abs(a.astype(np.float64) - b).max(initial=0)),
            'examples': [{'index': int(i), 'cpu': a[i].item(), 'metal': b[i].item()}
                         for i in different[:4]]}
    return result

def synthetic(path, posts, weights, modulation=False):
    np.savez(path, ptr=np.array([0, 1, 2, 2], dtype=np.int64) if posts else np.zeros(4, dtype=np.int64),
        post=np.array(posts, dtype=np.int32), weight=np.array(weights, dtype=np.float32),
        ids=np.arange(3, dtype=np.int64), retina=np.empty(0, dtype=np.int32),
        uv=np.empty((0, 2), dtype=np.float32), lamina=np.empty(0, dtype=np.int32),
        sugar=np.empty(0, dtype=np.int32), superclass=np.array(['test'] * 3))
    circuit = {'edges': np.empty(0, dtype=np.int64), 'pre': np.empty(0, dtype=np.int32),
        'kc_mask': np.zeros(3, dtype=np.uint8), 'dan_index': np.full(3, -1, dtype=np.int8),
        'gain': np.empty((0, 0), dtype=np.float32), 'kc': np.empty(0, dtype=np.int32),
        'mb': np.empty(0, dtype=np.int32), 'dan': np.empty(0, dtype=np.int32)}
    mask = np.array([1, 1, 0], dtype=np.uint8) if modulation else np.zeros(3, dtype=np.uint8)
    return [MemoryBrain(path, backend=backend, eta=0., circuit=circuit, modulation_mask=mask)
            for backend in ['cpu', 'metal']]

micro = {}
with tempfile.TemporaryDirectory(prefix='doomfly-rounding-probe.') as directory:
    for name, posts, weights, modulation, steps in [
            ('conductance_association', [2, 2], [16777216., -16777216.], False, 1),
            ('modulatory_only_settlement', [2, 2], [.275, .275], True, 2),
            ('repeated_subthreshold', [], [], False, 100)]:
        pair = synthetic(pathlib.Path(directory) / (name + '.npz'), posts, weights, modulation)
        try:
            for brain in pair:
                brain.v[:] = [-51., -52., -51.]; brain.g[:] = [0., 0., 1.]
                brain.adaptation[:] = [0., 0., .7]
                brain.active_flag.fill(0); brain.nactive.fill(0)
                if posts:
                    brain.queue[0, :2] = [0, 1]; brain.queue_count[0] = 2
                for _ in range(200 if name == 'repeated_subthreshold' else 1):
                    brain.counts.fill(0); brain.backend.advance(steps)
                brain.backend.materialize('arithmetic-diagnostic')
            micro[name] = differences(*pair)
        finally:
            for brain in pair:
                brain.backend.close()
            del pair; gc.collect()
save_json(out / 'arithmetic.json', micro)

pair = [calibrated_brain(backend=backend) for backend in ['cpu', 'metal']]
try:
    for brain in pair:
        brain.restore(configuration['initial_checkpoint'])
        brain.reset(keep_memory=True); brain.weights_frozen = True
        brain.backend.start_diagnostics()
    assert pair[0].n == pair[1].n == 166700
    assert len(pair[0].post) == len(pair[1].post) == 25582938
    total = [np.zeros(brain.n, dtype=np.int64) for brain in pair]
    bins = []; first_state = first_spikes = None
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    for index in range(200):
        before = [len(brain.backend.spike_events) for brain in pair]
        for position, brain in enumerate(pair):
            counts, _ = brain.rgb_step(black, 10., learning=False, stimulation=None)
            total[position] += counts
            brain.backend.materialize('long-horizon-diagnostic')
        delta = differences(*pair)
        cpu_events, metal_events = [set(brain.backend.spike_events[start:])
                                   for brain, start in zip(pair, before)]
        union = cpu_events | metal_events
        row = {'bin': index, 'end_steps': pair[0].cursor,
            'state_differences': {name: data['different'] for name, data in delta.items()},
            'cpu_spikes': len(cpu_events), 'metal_spikes': len(metal_events),
            'spike_jaccard': len(cpu_events & metal_events) / len(union) if union else 1.}
        if first_state is None and any(data['different'] for data in delta.values()):
            first_state = {'bin': index, 'fields': delta}
        if first_spikes is None and cpu_events != metal_events:
            first_spikes = {'bin': index, 'cpu_only': sorted(cpu_events - metal_events)[:8],
                            'metal_only': sorted(metal_events - cpu_events)[:8]}
        bins.append(row)
    data = OfflineEpisode(configuration['train_episode'])
    iterator = data.iter_frames(limit=1)
    try:
        sample = next(iterator)
    finally:
        iterator.close()
    readouts = json.loads((GRAPH.parent / 'manifest.json').read_text())['readouts']
    decisions = []
    for brain in pair:
        counts, _ = brain.rgb_step(sample.rgb, 28.6, learning=False, stimulation=None)
        action = NeuralControls(readouts, mode='bci').decode(counts, .0286)
        decisions.append({key: action[key] for key in ['turn', 'forward', 'attack']})
        brain.backend.stop_diagnostics()
    save_json(out / 'results.json', {
        'first_state_divergence': first_state, 'first_spike_divergence': first_spikes,
        'dark_bins': bins,
        'warmup_metrics': spike_metrics(
            [event for event in pair[0].backend.spike_events if event[1] < 20000],
            [event for event in pair[1].backend.spike_events if event[1] < 20000],
            total[0] / 2., total[1] / 2.),
        'first_frame_sha256': hashlib.sha256(sample.rgb.tobytes()).hexdigest(),
        'cpu_first_controls': decisions[0], 'metal_first_controls': decisions[1],
        'decoder_equal': decisions[0] == decisions[1],
        'learning_demonstrated': False, 'announcement_ready': False})
finally:
    for brain in pair:
        brain.backend.close()
```
