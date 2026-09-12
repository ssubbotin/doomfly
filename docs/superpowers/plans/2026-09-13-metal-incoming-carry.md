# Metal Incoming Conductance Carry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the reproduced incoming carry association while retaining CPU reference, event order and all circuitry.

**Architecture:** Start the existing target-local accumulator with settled conductance, retain its deterministic incoming scan, and assign accepted fast input back once. Advance checked arithmetic identity, then measure normal-valued native controls and retained full-graph/prefix evidence separately from the running FMA cohort.

**Tech Stack:** C++17, Objective-C++, Metal 2.4, ctypes, NumPy and pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-metal-incoming-carry-design.md`

## Global Constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Preserve CPU source/flags, explicit voltage FMA, resident decay/fallbacks,
  incoming order, bitmap masking/clearing, modulation and settlement schedules.
- Preserve every retained MaleCNS v1.0 connection, 166,700 neurons, 25,582,938
  edges, dt 0.1 ms and 4,184 existing plastic slots; synthetic tests are labeled.
- Preserve live RGB mapping, reinforcement, host plasticity and fixed decoding;
  targets/observer telemetry never select actions or replace the neural model.
- Preserve Jaccard >= 0.995, spike fraction <= 0.005, rate correlation >= 0.999,
  within-one-tick fraction >= 0.999, weight rtol 1e-4, exact decoder/scientific
  decisions, structural integrity and repeated state.
- Increment checked native/builder arithmetic epoch 5→6. No C layout, binding,
  dispatch, allocation, transfer, checkpoint-schema or new feature dependency.
- Preserve failed experiments, controls and prior measured goldens. Numerical
  tests, changed weights and survival alone establish neither useful nor fly learning.
- No public deployment/launch, unrelated workload interruption, credentials or
  machine origins in tracked files, or bundled game/external research assets.

### Task 1: Paired carry regression and checked arithmetic epoch

**Files:**
- Create: `tests/test_doom_metal_incoming.py`
- Modify: `doom_learning_v6/metal/kernels.metal`, `api.h`, `build.py`
- Modify only if required by epoch: `tests/test_doom_metal_build.py`, `tests/test_doom_metal_validation.py`
- Modify `tests/test_doom_metal_parity.py` only after two actual matching measurements.

**Interfaces:**
- Consumes: `MemoryBrain(path, backend=..., eta=0., circuit=..., modulation_mask=...)`, existing incoming bitmap gather and settled `g[target]`.
- Produces: normal-valued native carry controls and arithmetic epoch 6; no new execution/transfer interface.
- Root provides genuine M4 RED/GREEN and measured micrograph digests.

- [ ] **Step 1: Write synthetic real-native paired tests first.**

Construct a three-neuron graph with sources 0/1 and target 2. Empty sensory and
plastic arrays avoid external annotations. This helper creates CPU and Metal;
test code sets v/g/refractory/queued state before the first backend advance.

```python
def carry_pair(tmp_path, pre, weights, mask=(0,0,0)):
    n=3; pre=np.asarray(pre,dtype=np.int32); path=tmp_path/'carry-graph.npz'
    np.savez(path,
        ptr=np.r_[0,np.cumsum(np.bincount(pre,minlength=n))].astype(np.int64),
        post=np.full(len(pre),2,dtype=np.int32),
        weight=np.asarray(weights,dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['synthetic-test']*n))
    circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
    return [MemoryBrain(path,backend=name,eta=0.,circuit=circuit,
                       modulation_mask=np.asarray(mask,dtype=np.uint8))
            for name in ['cpu','metal']]
```

Use a macOS skip guard for actual pairs. Parametrize these seven cases:
`pre=(0,1),weights=(16777216,-16777216),g=1,refractory=0,mask=(0,0,0)`;
duplicate `pre=(0,0)` with the same weights/g;
ordinary `weights=(2,-.5),g=1`;
zero carry `weights=(16777216,-16777216),g=0`;
refractory rejection with ordinary weights/g and refractory3;
modulatory-only `weights=(.275,.275),g=1,mask=(1,1,0)`;
empty `pre=(),weights=(),g=1`.

```python
cpu,metal=carry_pair(tmp_path,pre,weights,mask)
try:
    queued=np.unique(np.asarray(pre,dtype=np.int32))
    for brain in [cpu,metal]:
        brain.v[:]=[-51.,-52.,-51.];brain.g[:]=[0.,0.,initial_g]
        brain.refractory[2]=initial_refractory
        brain.active_flag.fill(0);brain.nactive.fill(0)
        brain.queue[0,:len(queued)]=queued;brain.queue_count[0]=len(queued)
        brain.backend.advance(1);brain.backend.materialize('incoming-carry-test')
    assert cpu.cursor==metal.cursor==1
    np.testing.assert_array_equal(cpu.counts,metal.counts)
    for name in ['v','g','adaptation','modulation']:
        np.testing.assert_array_equal(getattr(cpu,name).view(np.uint32),
                                      getattr(metal,name).view(np.uint32),err_msg=name)
    for name in ['refractory','last','modulation_last','active_flag']:
        np.testing.assert_array_equal(getattr(cpu,name),getattr(metal,name),err_msg=name)
finally:
    cpu.backend.close();metal.backend.close()
```

For the nonduplicate catastrophic case, also assert native CPU target `g==0`
and no newly fired spikes. Do not assert CPU queue order equivalence for arbitrary
production graphs; the fixture supplies equal-order sources explicitly.

- [ ] **Step 2: Commit tests and obtain genuine M4 RED.**

Run Linux collection and diff check; commit owned tests. Return exact checkpoint
to root and pause before production edits. Root uses a fresh owned target;
the existing full FMA cohort keeps its source. Actual command:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_doom_metal_incoming.py
```

Expected catastrophic target g mismatch (CPU0/Metal0.9801986813545227), with
healthy compiler/runtime. Missing files/compiler errors are prerequisites,
not arithmetic RED. Preserve unexpected control differences for diagnosis.

- [ ] **Step 3: Apply only carry association and arithmetic epoch.**

```cpp
float conductance=g[target],modulatory=0.0f;bool has_fast=false,has_modulatory=false;
// Existing scan/predicates/additions unchanged.
if(has_fast){g[target]=conductance;active[target]=1;}
```

Set `DF_METAL_ABI_VERSION` in `api.h` and `ABI_VERSION` in `build.py` to6.
Retain all code outside these four changes. No exclusively-modulatory timing,
input ordering, CPU, flags, sensory, plasticity or dispatch changes.

- [ ] **Step 4: Verify portable/native controls and measured golden.**

```sh
python -m pytest -q tests/test_doom_metal_incoming.py tests/test_doom_metal_arithmetic.py tests/test_doom_metal_decay.py tests/test_doom_metal_build.py tests/test_doom_metal_validation.py tests/test_doom_metal_parity.py
python -m pytest -q
git diff --check
```

Use supplied LLVM21/absolute Python/bootstrap environment. Root runs actual
M4 paired controls, then two fresh micrographs. Update the golden only if its
two measured hashes agree and the previous literal fails on healthy runtime;
preserve old `65524d57...` in a comment. Root completes actual M4 suite and
independent task review. Final source stays fixed while target jobs run.

- [ ] **Step 5: Commit owned source/tests and report.**

Use Sergey Subbotin <ssubbotin@gmail.com>. Stage exact owned files, preserve
others' edits and root evidence. Write full TDD commands/output, actual source
hashes, self-review, files and unresolved concerns to supplied task report.
No SSH, push, subagents or experiment-document edits by the worker.

### Task 2: Retained-graph and original-prefix evidence (controller-owned)

**Files:**
- Create: `docs/experiments/2026-09-13-metal-incoming-carry-invocation.md`
- Create: `docs/experiments/2026-09-13-metal-incoming-carry.md`
- Create: bounded portable JSON under `outputs/doom-learning/metal-incoming-carry-20260913/`
- Update: `docs/doom-metal-backend.md` for actual measurements/limits only.

**Interfaces:**
- Consumes: reviewed Task1source, actual M4 controls, fresh matching report, same pinned original data/CPU prefix evidence.
- Produces: exact-source 40/80ms results, micrograph epoch, original prefix comparison and declared remaining differences.

- [ ] **Step 1: Save invocation before execution.** Fresh output, exactsource,
  strictflags, passingreport/source identity and ignored machine/data origins.
- [ ] **Step 2: Complete actual tiny TDD controls.** Separate target checkout;
  keep the running complete FMA cohort untouched. Capture original RED/GREEN.
- [ ] **Step 3: After the cohort ends naturally, run unchanged 40/80ms gates.**

```python
forty=validate.run(out/'parity-40ms')
original=validate.TRACE
validate.TRACE=(('black',False),('blue',False),('green',False),('white',True),
    ('left_blue',False),('right_blue',True),('vertical',False),('horizontal',False))
try:
    eighty=validate.run(out/'parity-80ms')
finally:
    validate.TRACE=original
```

- [ ] **Step 4: Repeat original 35-frame control prefix.** Use fresh output,
  same source/demonstrations/eta1e-3, same2s dark warmup, all8 phases, exact
  matching80msguard, `--backend metal --max-frames 35`. Compare true fixed
  turn/forward/attack separately from observed readouts and per-frame hashes.
- [ ] **Step 5: Publish actual bounded evidence and limits.** Root verifies
  target/localSHA+bytes, portableJSON/sourceidentity, actualM4suite, independent
  whole-component review, exact-filecommit/push. Keep complete traces/checkpoints
  and raw logs ignored. Preserve remaining failures; do not claim full-episode
  equivalence/usefullearning or speed before measurements. The original FMA
  cohort's result is documented separately under its own source/protocol.
