# Explicit CPU-Compatible Metal Arithmetic Invocation

Saved before the implementation's M4 experiment. Run on an owned exact-source
checkout with the existing single-threaded environment, conservative Metal
flags and a fresh output. `DOOMFLY_RUNTIME_COMMIT` identifies the implementation;
target paths and initial/data origins come only from ignored configuration.

The actual test-only M4 source `b9defb9` failed all eight arithmetic regressions
at the first 100-tick bin (2.40 s). The earlier checkout update was refused to
protect its untracked measured reports; that missing-test-path failure is not
arithmetic RED. Preserve both diagnostics. Build after implementation, run real
fixtures, then this numerical experiment. Keep source fixed while running.

This deliberately changes voltage operation association only: explicit FMA
matching the observed M4 CPU binary. CPU source/flags, incoming conductance
order/association, exclusively modulatory arrivals, coefficients and fallback
updates remain fixed. Full retained graph and scientific/parity gates remain
unchanged. A longer remaining failure is preserved, with no useful-learning
or biological interpretation. Diagnostic downloads invalidate speed claims.

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.metal.build --probe
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_doom_metal_arithmetic.py
```

```python
import gc, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from doom_learning.common import ROOT, capture_provenance, require_single_blas_thread, save_json
from doom_learning_v6.metal import validate
from doom_learning_v6.metal.build import probe

require_single_blas_thread()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert commit == os.environ['DOOMFLY_RUNTIME_COMMIT']
out = Path(os.environ['DOOMFLY_ARITHMETIC_OUT'])
if out.exists():
    raise ValueError('Fresh output directory required')
out.mkdir(parents=True)
capture_provenance(out, additional=['doom_learning_v2', 'doom_learning_v6', 'tests'])
metadata = probe()
assert metadata['abi_version'] == metadata['native_abi_version'] == 5
assert metadata['build_configuration']['metal_compile_flags'] == [
    '-std=macos-metal2.4', '-fno-fast-math', '-ffp-contract=off']
save_json(out / 'protocol.json', {
    'schema': 1, 'source_commit': commit, 'arithmetic_epoch': 5,
    'metal_build': validate._portable_metal_metadata(metadata),
    'neurons': 166700, 'retained_edges': 25582938, 'plastic_slots': 4184,
    'previous_micrograph_state_sha256':
        '6163f146b096db3ba7c086b2b1933977b380e3fec2cd08c6d7c7ce141b16d791',
    'long_comparison': 'Same initial checkpoint, 200 dark bins and original first RGB; source/epoch/output identities changed only',
    'learning_demonstrated': False, 'announcement_ready': False})

sys.path.insert(0, str(ROOT / 'tests'))
from test_doom_metal_parity import paired_brains, run_trace
micro = {}; brains = []
try:
    cpu, metal = paired_brains(out / 'micrograph-first')
    unused, repeated = paired_brains(out / 'micrograph-repeat')
    brains = [cpu, metal, unused, repeated]
    for name, brain in [('cpu', cpu), ('metal', metal), ('metal-repeat', repeated)]:
        brain.backend.start_diagnostics()
        counts = run_trace(brain)
        brain.backend.stop_diagnostics()
        events = np.asarray(brain.backend.spike_events, dtype=np.int64)
        micro[name] = {
            'counts_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
            'events_sha256': hashlib.sha256(events.tobytes()).hexdigest(),
            'events': events.tolist(), 'state_sha256': validate._state_digest(brain)}
    save_json(out / 'micrograph.json', {
        'schema': 1, 'source_commit': commit, 'results': micro,
        'repeat_bitwise': micro['metal'] == micro['metal-repeat'],
        'previous_state_sha256': '6163f146b096db3ba7c086b2b1933977b380e3fec2cd08c6d7c7ce141b16d791'})
    assert micro['metal'] == micro['metal-repeat']
finally:
    for brain in brains:
        brain.backend.close()
    del brains; gc.collect()

forty = validate.run(out / 'parity-40ms')
original = validate.TRACE
validate.TRACE = (('black', False), ('blue', False), ('green', False), ('white', True),
    ('left_blue', False), ('right_blue', True), ('vertical', False), ('horizontal', False))
try:
    eighty = validate.run(out / 'parity-80ms')
finally:
    validate.TRACE = original

# Reuse the exact retained diagnostic, with precisely stated identity changes.
invocation = ROOT / 'docs/experiments/2026-09-13-metal-long-horizon-diagnostic-invocation.md'
raw = invocation.read_text()
code = raw.split('```python\n', 1)[1].split('```', 1)[0]
old_out = "out = pathlib.Path('outputs/doom-learning/metal-long-horizon-diagnostic-20260913')"
old_commit = "'source_commit': 'a5d31453ae2a5eca82cb39d8a40d0643a80cb477'"
assert code.count(old_out) == code.count(old_commit) == 1
code = code.replace(old_out, 'out = pathlib.Path(' + repr(str(out / 'long-parity')) + ')')
code = code.replace(old_commit, "'source_commit': " + repr(commit))
exec(compile(code, 'retained-long-parity-invocation', 'exec'), {})
long_result = json.loads((out / 'long-parity/results.json').read_text())
save_json(out / 'results.json', {
    'schema': 1, 'complete': True, 'source_commit': commit,
    'micrograph_repeat_bitwise': micro['metal'] == micro['metal-repeat'],
    'parity_40ms_passed': forty['passed'], 'parity_80ms_passed': eighty['passed'],
    'first_state_divergence': long_result['first_state_divergence'],
    'first_spike_divergence': long_result['first_spike_divergence'],
    'warmup_metrics': long_result['warmup_metrics'],
    'decoder_equal_after_warmup': long_result['decoder_equal'],
    'baseline_invocation_sha256': hashlib.sha256(raw.encode()).hexdigest(),
    'learning_demonstrated': False, 'announcement_ready': False})
print(json.dumps(json.loads((out / 'results.json').read_text())), flush=True)
```

Return both actual matching micrograph measurements to the original worker
before changing the golden test. Run the complete M4 suite after its independently
checked update. Publish bounded portable reports with artifact checksums;
exclude game pixels, NPZ inputs/checkpoints, generated binaries and machine origins.
Continue separate incoming/arithmetic experiments if longer parity still fails.

## Original training prefix after numerical diagnostics

The exact `9d24668` diagnostic completed with passing 40/80 ms gates, exact
two-second dark spikes and equal first-original-RGB controls. Repeat the same
eight 35-frame phases as the preserved CPU/table prefixes, using this epoch's
matching 80 ms report and another fresh output. Keep original data, teacher
schedule, eta and decoder fixed. Compare all original RGB/action/spike hashes,
teacher causality, frozen controls, invariants and actual wall/brain timing.
This prefix remains incomplete relative to source episodes and establishes no
learning. Small incoming-related state differences remain despite exact dark
spikes; whole-episode parity remains untested. Do not overlap full-graph jobs.

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_ARITHMETIC_PREFIX_OUT" --epochs 1 --eta .001 \
  --backend metal --metal-validation "$DOOMFLY_METAL_PARITY_REPORT" \
  --max-frames 35
```

## Complete exploratory Metal cohort

The amended actual M4 suite at `e9e19c1` passed 371 tests (three platform skips,
50 existing dependency deprecations, 20.15 s); the independent Task 1 review
approved spec compliance and quality. Production hashes are unchanged from
the measured `9d24668` diagnostic. The prefix completed all eight phases with
unchanged invariants, teacher-free frozen evaluations and exact within-Metal
erasure/frozen repetition. All three training prefixes and the learned plastic
held-out prefix match CPU. Remaining frozen/retention/shifted held-out controls
diverge late in the one-second RGB segment; retain that failed longer parity.

Proceed with the previously planned full cohort as an explicit exploratory
experiment, using the exact passing 80 ms identity guard and the complete
original episodes. This does not promote the short guard to whole-episode
validation. Source remains fixed at `e9e19c1`; use a fresh output, preserve all
arms/checkpoints/failures and compare actual full results with the CPU cohort.
No useful-learning, physiology, speedup or launch claim follows. Do not overlap
another full-graph target job. Continue separately scoped incoming/arithmetic
work locally while this fixed-source experiment runs.

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_ARITHMETIC_FULL_OUT" --epochs 1 --eta .001 \
  --backend metal --metal-validation "$DOOMFLY_METAL_PARITY_REPORT"
```
