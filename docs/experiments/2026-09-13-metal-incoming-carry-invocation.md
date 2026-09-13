# Isolated Metal Incoming Carry Invocation

Saved before execution. Tiny controls run in a separate owned target checkout;
the complete FMA cohort stays fixed at `e9e19c1` and finishes naturally. Use the
existing single-threaded environment, absolute dependency interpreter and
bootstrap native kernel. Source/target/input origins remain ignored. No live
service, unrelated workload, CPU reference or scientific gate is changed.

## Genuine tiny controls before and after the change

Set `DOOMFLY_RUNTIME_COMMIT` to the committed test-only or implementation
revision, respectively. Assert actual HEAD matches it. Keep source immutable
during each command. Run actual paired CPU/Metal normal-valued fixtures:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_doom_metal_incoming.py
```

Expected test-only RED is a conductance mismatch in the catastrophic carry
fixture: settled initial carry1 plus arriving `16777216/-16777216`, CPU0 versus
Metal0.9801986813545227. File/compiler/configuration errors are prerequisites,
not RED. Preserve raw failure output and all seven controls. Implement only
the two accumulator assignments and arithmetic epoch5→6 after actual RED.

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.metal.build --probe
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_doom_metal_incoming.py tests/test_doom_metal_arithmetic.py \
  tests/test_doom_metal_decay.py tests/test_doom_metal_build.py \
  tests/test_doom_metal_validation.py
```

## Fresh repeated micrograph after implementation

Use a fresh `DOOMFLY_CARRY_MICRO_OUT`, exact committed implementation and
conservative flags. This tiny artificial graph measures arithmetic, with no
retained-circuit or biological interpretation. Root returns both actual state
digests before changing any golden. Preserve counts/events and oldepoch65524d57.

```python
import gc, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from doom_learning.common import ROOT, require_single_blas_thread, save_json
from doom_learning_v6.metal import validate
from doom_learning_v6.metal.build import probe

require_single_blas_thread()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==os.environ['DOOMFLY_RUNTIME_COMMIT']
out=Path(os.environ['DOOMFLY_CARRY_MICRO_OUT'])
assert not out.exists()
out.mkdir(parents=True)
metadata=probe()
assert metadata['abi_version']==metadata['native_abi_version']==6
assert metadata['build_configuration']['metal_compile_flags']==[
    '-std=macos-metal2.4','-fno-fast-math','-ffp-contract=off']
sys.path.insert(0,str(ROOT/'tests'))
from test_doom_metal_parity import paired_brains,run_trace

results={};brains=[]
try:
    cpu,metal=paired_brains(out/'first')
    unused,repeated=paired_brains(out/'repeat')
    brains=[cpu,metal,unused,repeated]
    for name,brain in [('cpu',cpu),('metal',metal),('metal-repeat',repeated)]:
        brain.backend.start_diagnostics()
        counts=run_trace(brain)
        brain.backend.stop_diagnostics()
        events=np.asarray(brain.backend.spike_events,dtype=np.int64)
        results[name]={
            'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
            'events_sha256':hashlib.sha256(events.tobytes()).hexdigest(),
            'events':events.tolist(),'state_sha256':validate._state_digest(brain)}
    result={'schema':1,'source_commit':commit,'arithmetic_epoch':6,
        'validation_sources':validate.source_identity(),
        'metal_build':validate._portable_metal_metadata(metadata),
        'results':results,'repeat_bitwise':results['metal']==results['metal-repeat'],
        'previous_state_sha256':'65524d57f34b5477f9bf61b7546bfb79248d65426b7fec6733f3d2315d53bd09',
        'learning_demonstrated':False,'announcement_ready':False}
    save_json(out/'micrograph.json',result)
    assert result['repeat_bitwise']
    print(json.dumps(result),flush=True)
finally:
    for brain in brains:brain.backend.close()
    del brains;gc.collect()
```

## Full-graph work after the original cohort finishes

Only after the complete FMA cohort ends naturally, use the reviewed carry source
for unchanged 40/80ms gates in fresh outputs. Then run all eight original
35-frame prefix phases with the exact passing epoch6 report, eta.001, two-second
dark warmup and same pinned GameWAM RGB/control artifacts. Compare real fixed
turn/forward/attack separately from observed readouts, teacher causality, hashes,
invariants, erased/frozen repetition and wall/brain timing. Save each concrete
command/configuration before execution, preserving old failed results. Neither
tiny GREEN nor short validation certifies whole-episode parity, useful learning,
speedup, living-fly physiology or public launch.

### Exact retained-graph invocation

Saved before execution. Set `DOOMFLY_RUNTIME_COMMIT` to the independently
reviewed carry implementation, `DOOMFLY_CARRY_PARITY_OUT` to a fresh output
and `DOOMFLY_FMA_COHORT_OUT` to the preserved original full cohort. Refuse
execution until that cohort reports all eight phases complete. This captures
fresh epoch-6 reports; the old epoch-5 guard cannot authorize this source.

```python
import hashlib, json, os, subprocess
from pathlib import Path
from doom_learning.common import ROOT, capture_provenance, require_single_blas_thread, save_json
from doom_learning_v6.metal import validate
from doom_learning_v6.metal.build import probe

require_single_blas_thread()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==os.environ['DOOMFLY_RUNTIME_COMMIT']
cohort=Path(os.environ['DOOMFLY_FMA_COHORT_OUT'])
finished=json.loads((cohort/'results.json').read_text())
assert finished['complete'] and len(finished['episodes'])==8
assert all(row['complete'] for row in finished['episodes'])
out=Path(os.environ['DOOMFLY_CARRY_PARITY_OUT'])
assert not out.exists()
out.mkdir(parents=True)
capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6','tests'])
metadata=probe()
assert metadata['abi_version']==metadata['native_abi_version']==6
assert metadata['build_configuration']['metal_compile_flags']==[
    '-std=macos-metal2.4','-fno-fast-math','-ffp-contract=off']
save_json(out/'protocol.json',{
    'schema':1,'source_commit':commit,'arithmetic_epoch':6,
    'metal_build':validate._portable_metal_metadata(metadata),
    'neurons':166700,'retained_edges':25582938,'plastic_slots':4184,
    'original_fma_cohort_results_sha256':hashlib.sha256((cohort/'results.json').read_bytes()).hexdigest(),
    'learning_demonstrated':False,'announcement_ready':False})
forty=validate.run(out/'parity-40ms')
original=validate.TRACE
validate.TRACE=(('black',False),('blue',False),('green',False),('white',True),
    ('left_blue',False),('right_blue',True),('vertical',False),('horizontal',False))
try:
    eighty=validate.run(out/'parity-80ms')
finally:
    validate.TRACE=original
result={'schema':1,'complete':True,'source_commit':commit,
    'parity_40ms_passed':forty['passed'],'parity_80ms_passed':eighty['passed'],
    'learning_demonstrated':False,'announcement_ready':False}
save_json(out/'results.json',result)
print(json.dumps(result),flush=True)
```

### Exact original-prefix command

Use the same reviewed source, original inputs, another fresh prefix output
and its exact epoch-6 80 ms report. Assert source identity and fresh output
before starting. Preserve eight phase traces and checkpoints, including any
remaining failure. The original episodes contain 982/988 frames, so this
35-frame diagnostic remains incomplete relative to those sources.

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_CARRY_PREFIX_OUT" --epochs 1 --eta .001 \
  --backend metal --metal-validation "$DOOMFLY_METAL_PARITY_REPORT" \
  --max-frames 35
```

### Warmup onset diagnostic after the failed prefix

The complete eight-phase prefix at the reviewed carry source has CPU/Metal
control differences at the first original RGB frame. Save and run the same
retained 200-bin dark diagnostic after prefix completion, with only exact
source/output identities changed. Preserve artificial carry/modulation controls
and first state/spike differences. State downloads perturb timing, so this
provides no throughput estimate. Production source, initial checkpoint,
inputs, decoder and teacher remain untouched.

```python
import os, re, subprocess
from pathlib import Path
from doom_learning.common import ROOT

commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==os.environ['DOOMFLY_RUNTIME_COMMIT']
raw=(ROOT/'docs/experiments/2026-09-13-metal-long-horizon-diagnostic-invocation.md').read_text()
block=re.search(r'^```python\n(.*?)^```\s*$',raw,re.S|re.M)
assert block
code=block.group(1)
old_out="out = pathlib.Path('outputs/doom-learning/metal-long-horizon-diagnostic-20260913')"
old_commit="'source_commit': 'a5d31453ae2a5eca82cb39d8a40d0643a80cb477'"
assert code.count(old_out)==code.count(old_commit)==1
code=code.replace(old_out,'out = pathlib.Path('+repr(os.environ['DOOMFLY_CARRY_DIAGNOSTIC_OUT'])+')')
code=code.replace(old_commit,"'source_commit': "+repr(commit))
exec(compile(code,'<retained-carry-onset-diagnostic>','exec'),{})
```
