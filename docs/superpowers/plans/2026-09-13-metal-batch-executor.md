# Resident Metal Multi-Trajectory Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run independent full-connectome RGB training trajectories in shared Metal dispatches without changing epoch-5 lane results.

**Architecture:** Extend the existing native Metal owner and kernels with lane-major mutable buffers and shared immutable topology. Preserve single-lane wrappers; add a Python batch owner that uses the unchanged preparation/R8/plasticity operations at the same observation boundaries. Prove per-lane identity before measuring complete original episodes and scaling from two to four.

**Tech Stack:** C++17, Objective-C++, Metal 2.4, ctypes, NumPy, pytest, existing ViZDoom/control/data adapters.

**Spec:** `docs/superpowers/specs/2026-09-13-metal-batch-executor-design.md`

## Global Constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Numerical parent b24249ecfee12aa84b55ed803ffe47bd09c487ad (epoch 5): preserve explicit voltage FMA, CPU-derived decay tables/fallbacks, zero-seeded incoming accumulator, ascending incoming-position gather, bitmap clearing, modulation and settlement schedules.
- Preserve CPU source/compiler flags, dt 0.1 ms, all 166,700 retained MaleCNS v1.0 neurons, all 25,582,938 retained edges and all 4,184 existing KC-to-MBON11 plastic slots.
- Preserve host float64 centered plasticity, its 50 ms filter, live RGB sensory mapping, reinforcement separation and fixed neuron-to-button decoding; targets/telemetry never select actions.
- Preserve Jaccard >= 0.995, spike fraction <= 0.005, rate correlation >= 0.999, within-one-tick fraction >= 0.999, weight rtol 1e-4, exact decoder/scientific decisions, structural integrity and repeated state.
- Preserve epoch-5 cancellation failures and long RGB mismatches, and the independently published epoch-6 carry/onset failures. No tolerance/golden changes to hide batching differences.
- Numerical tests, changed weights and survival alone establish neither useful learning nor biological validation.
- No public deployment/launch, unrelated workload interruption, tracked credentials/machine origins, bundled external research/game assets, destructive cleanup or main-branch integration.
- Git identity Sergey Subbotin <ssubbotin@gmail.com>; preserve others' changes and third-party notices; no AI author attribution in source/comments/commit messages.

---

## File responsibilities

Existing `metal/api.h`, `backend.mm` and `kernels.metal` retain one native
implementation with N=1 wrappers and additive lane API. `metal/build.py`
checks ABI 7 and fingerprints all sources. New `metal/batch.py` owns Python
lifecycle, training synchronization and lane adapters. Existing `brain.py`
and `visual.py` expose small shared preparation/update helpers, preserving
serial operations. Native tests and Python training tests stay separate.
Root owns invocation/evidence documentation and remote jobs; workers do no SSH,
push, subagents or experiment-publication edits.

### Task 1: Checked resident native lanes and raw identity controls

**Files:**

- Modify: `doom_learning_v6/metal/api.h`, `backend.mm`, `kernels.metal`, `build.py`.
- Create: `tests/test_doom_metal_batch_native.py`.
- Modify only for checked ABI: `tests/test_doom_metal_build.py`, `tests/test_doom_metal_validation.py`.
- Do not modify CPU kernels, existing arithmetic/carry associations, golden digests or scientific gate code.

**Interfaces:**

- Consumes existing `Graph`, `State`, `KCEvent`, `Timing` ctypes/C layouts and epoch-5 direct Metal stages.
- Produces these additive C symbols (existing functions remain N=1 wrappers):

```cpp
int df_metal_create_batch(const df_metal_graph *, const char *, int32_t lanes,
                         df_metal_handle *);
int df_metal_upload_lane_state(df_metal_handle, int32_t lane,
                               const df_metal_state *);
int df_metal_download_lane_state(df_metal_handle, int32_t lane, df_metal_state *);
int df_metal_upload_lane_drive(df_metal_handle, int32_t lane, const float *);
int df_metal_download_lane_observation(df_metal_handle, int32_t lane,
                                     int32_t *, int64_t *);
int df_metal_update_lane_weights(df_metal_handle, int32_t lane, int32_t count,
                                const int64_t *, const float *);
int df_metal_apply_lane_eligibility(df_metal_handle, int32_t lane,
                                  double *, int64_t *, double tau_ms);
```

- `df_metal_advance` advances all registered lanes equally; `KCEvent.reserved`
  identifies batch lane, serial reserved remains zero. `Timing` represents one
  command, with grid work multiplied by lane count; dispatch count stays
  `2*steps+2`. Add `df_metal_batch_memory_bytes(handle, uint64_t *shared,
  uint64_t *mutable)` for actual allocated resident bytes.
- Native lane cursors may be individually uploaded, but advance validates equal
  cursors before clearing counts or dispatch. Per-lane eligibility filters events
  and preserves existing expf/float64 update association.
- Root provides healthy actual M4 test-only RED and final native GREEN.

- [ ] **Step 1: Write real-native tests before adding symbols.**

Use macOS skip guards and existing fixture `test_doom_learning_v6.brain`.
Configure new ctypes symbols in a helper; test absence fails on a healthy
existing library. Build Graph from the seed's actual incoming index:

```python
from doom_learning_v6.metal.backend import Graph, State, KCEvent, Timing, _pointer
seed.backend.ensure_initialized()
incoming = seed.backend.incoming
graph = Graph(seed.n, len(seed.post), seed.queue.shape[0], seed.dt,
              seed.adaptation_jump, seed.adaptation_tau,
              _pointer(seed.ptr), _pointer(seed.post), _pointer(incoming.ptr),
              _pointer(incoming.pre), _pointer(incoming.edge),
              _pointer(seed.circuit['kc_mask']), _pointer(seed.modulation_mask))
library.df_metal_create_batch.argtypes = [
    C.POINTER(Graph), C.c_char_p, C.c_int32, C.POINTER(C.c_void_p)]
library.df_metal_create_batch.restype = C.c_int
handle = C.c_void_p()
assert library.df_metal_create_batch(
    C.byref(graph), str(DEFAULT_OUTPUT/'kernels.metallib').encode(),
    2, C.byref(handle)) == 0
```

Create two compatible serial Metal reference brains and two fresh host state
brains. Give distinct source drives `[20,0,12,0]` and `[24,0,17,0]`,
and different identified slot values 19 and 18 before initialization.
Upload every lane state using the original State pointers, then advance the
same sequence `(1,40,80,100,100,100)` on serial and batch. Download lane state
only for assertions; compare:

```python
for name in ('weight','v','g','adaptation','modulation'):
    np.testing.assert_array_equal(
        getattr(actual,name).view(np.uint32),
        getattr(expected,name).view(np.uint32), err_msg=name)
for name in ('refractory','last','modulation_last','active_flag','counts'):
    np.testing.assert_array_equal(getattr(actual,name),
                                  getattr(expected,name), err_msg=name)
assert actual.cursor == expected.cursor
```

Compare queue_count and valid queue membership, full diagnostic spike multiset
per lane, eligibility/eligibility_last after native per-lane updates, and N=1
state bits. Repeat fresh runs/reversed lane registration and vary one lane's
state/drive/identified weight; neighbors must be exact. Include zero-edge
graphs, duplicate/self-edge synthetic graphs and modulatory-only delivery.
Every native handle closes in finally.

Add tests rejecting lane -1/out-of-range, zero lane count, mismatched cursors
without state changes, null pointers, excessive allocations and bad event
capacity. Assert one command's timing/grid counts rather than invented speed.

- [ ] **Step 2: Commit test-only checkpoint and obtain M4 RED.**

Run portable collection/diff check; commit only owned tests. Report the SHA to
root and pause before production edits. Root builds that healthy source on
a new owned mini target, captures the missing native symbol/API failure and
returns the actual output. Compiler/path failures are prerequisites, not RED.

```sh
python -m pytest -q tests/test_doom_metal_batch_native.py
git diff --check
git add tests/test_doom_metal_batch_native.py
git commit -m "test: cover independent resident Metal lanes"
```

- [ ] **Step 3: Add checked lane-major native ownership and grid indexing.**

Use one native Backend with a lane count and individual uploaded cursors.
Allocate immutable buffers once and multiply mutable storage by lanes using
checked size_t products. Validate maximumBufferLength and recommended working
set before allocation; reject unsupported sizes without partial handles.
Keep every lane's full weights and all neuron/ring/touched/edge bitmap storage.

Extend Params with lane strides/count and dispatch a two-dimensional
neuron-by-lane grid. Each kernel computes lane-local source/target IDs and
offsets its mutable pointers before executing the unchanged body:

```cpp
uint neuron = position.x;
uint lane = position.y;
if (lane >= p.lanes || neuron >= p.neurons) return;
// Offset mutable v/g/ref/drive/last/adaptation/count/active by lane*neurons.
// Offset ring by lane*slots*words and edge bits by lane*edge_words.
// Keep immutable out_ptr/out_post/in_ptr/in_pre/in_edge/masks/decay shared.
```

Preserve the epoch-5 `conductance=0.0f` and `g[target]+=conductance`
association. Keep stage ordering, atomic bitmap masks and exact per-lane
clearing. Record lane in emitted events without changing neural event order.
Per-lane sparse updates operate only on that lane's full weight buffer.
Advance checks cursor equality/overflow before mutation. GPU failure/overflow
poisons owner; cleanup releases every resource. Bump api.h/build.py ABI 5→7,
record numerical parent/order in metadata without claiming epoch-6 arithmetic.

- [ ] **Step 4: Verify focused controls and unchanged serial goldens.**

```sh
python -m pytest -q tests/test_doom_metal_batch_native.py tests/test_doom_metal_state.py tests/test_doom_metal_parity.py tests/test_doom_metal_arithmetic.py tests/test_doom_metal_decay.py tests/test_doom_metal_build.py tests/test_doom_metal_validation.py
python -m pytest -q
git diff --check
```

Use supplied LLVM21/absolute Python/bootstrap environment. Root captures actual
M4 raw native GREEN, two unchanged micrographs, focused/full suite. No golden
updates: lane zero with N=1 must retain its prior digest. State transfer does
not occur in timed training loops.

- [ ] **Step 5: Commit exact owned changes, report and pass task review.**

Full report includes actual RED/GREEN commands/output, source hashes, files,
native memory accounting, self-review and concerns. Root task review gets
all commits from recorded task BASE, spec constraints, report and diff.
Fix findings via the original worker; do not claim throughput/learning.

### Task 2: Batch Python ownership and unchanged RGB/plasticity stepping

**Files:**

- Create: `doom_learning_v6/metal/batch.py`.
- Modify: `doom_learning_v6/brain.py`, `doom_learning_v6/visual.py`.
- Create: `tests/test_doom_metal_batch.py`, `tests/test_doom_metal_batch_rgb.py`.
- Modify: `tests/test_doom_learning_v6.py` only for exact serial preparation/refactor guards.
- Existing serial `metal/backend.py` may be changed only to expose shared ctypes
  setup/metadata helpers if required; no neural or plastic semantics change.

**Interfaces:**

- Consumes Task1 C symbols, Graph/State/events/timing and existing MemoryBrain/
  VisualMemoryBrain fields/configuration signature.
- Produces `MetalBatchExecutor(brains)` with `advance(steps)`,
  `step(luminances,duration_ms,learning=False,stimulations=None,lamina_bias=12.)`,
  `rgb_step(frames,duration_ms,learning=False,stimulations=None,lamina_bias=12.)`,
  context manager, metadata and close. All execution calls return
  `(list_of_copied_counts, aggregate_backend_wall_seconds)`.
- Internal helpers: `MemoryBrain._prepare_neural_input(luminance,steps,
  stimulation=None,lamina_bias=12.)`,
  `MemoryBrain._apply_centered_rule(counts,seconds,learning)`;
  `VisualMemoryBrain._prepare_rgb_input(frame,duration_ms,stimulation=None)`
  returns `(retinal_luminance,pulses)`. They retain current operation order.
- Lane backend adapter preserves existing checkpoint/reset/restore/materialize/
  sparse-update metadata interfaces. Attached direct brain.step/rgb_step fails
  before input/state changes. Only executor performs native batch advances.

- [ ] **Step 1: Write missing-owner RED and real training comparisons first.**

Portable import/API and invalid-input tests run on Linux; real native comparisons
skip only on nonmacOS. Use existing four-neuron fixture for plastic/frozen
runs, copying initial state into separate references before ownership:

```python
from doom_learning_v6.metal.batch import MetalBatchExecutor
serial = [brain(tmp_path, backend='metal') for _ in range(2)]
batched = [brain(tmp_path, backend='cpu') for _ in range(2)]
serial[1].weights_frozen = batched[1].weights_frozen = True
with MetalBatchExecutor(batched) as executor:
    for duration in (10., 28.6, 28.5, 10.):
        pulses = [([0,2],20.), ([0,2],14.)]
        expected = [b.step([], duration, learning=True, stimulation=p,
                           lamina_bias=0)[0] for b,p in zip(serial,pulses)]
        actual, seconds = executor.step([[],[]], duration, learning=True,
                                       stimulations=pulses, lamina_bias=0)
        assert seconds >= 0
        for a,e in zip(actual,expected):
            np.testing.assert_array_equal(a,e)
        for a,e in zip(batched,serial):
            a.backend.materialize('batch-test')
            for name in ['weight',*a.fields]:
                assert getattr(a,name).tobytes() == getattr(e,name).tobytes()
```

Materialize serial before full-state assertions too. Exercise scalar/per-lane
learning booleans, input validation before any lane change, immutable graph/
configuration mismatches, unsafe buffer aliases/reallocation, ownership,
reentrant close/advance, cursor mismatch, checkpoint producer metadata,
independent reset/restore and post-close safety.

RGB tests use synthetic VisualMemoryBrain-compatible R8/UV/channel arrays
and actual fixed NeuralControls decoding; no downloaded annotations are needed
for synthetic tests. Exercise different RGB frames per lane, 35 Hz durations,
R8 updates every <=100-tick subdivision and final partial bins. Compare every
serial/batch field bitwise, spike events and turn/forward/attack decisions.
Repeat/permutation/one-lane pixel or teacher perturbations test isolation.

- [ ] **Step 2: Capture RED before implementation.**

```sh
python -m pytest -q tests/test_doom_metal_batch.py tests/test_doom_metal_batch_rgb.py
```

The missing new batch module/API is expected RED. Existing compiler/runtime
must remain healthy. Preserve full failure output in report.

- [ ] **Step 3: Implement validated owner and shared preparation helpers.**

Extract the existing luminance/drive operation block verbatim into
_prepare_neural_input and existing rule/update block into _apply_centered_rule.
Serial methods call helpers at their unchanged boundaries. Extract RGB/R8
sampling/smoothing/pulse preparation verbatim; validate attached ownership
before changing serial RGB state.

The executor prevalidates every lane input and cursor, prepares all lane drives,
invokes one native batch advance, applies native eligibility separately to
lane host float64 arrays, downloads only counts/cursors, then calls the existing
centered rule independently. Preserve sim_ms, total_spikes, count accumulation,
last_rule_seconds and per-bin sparse slot update behavior. Aggregate GPU/native
timing is stored once in executor.last_timing; per-lane timing is explicitly
attributed and never silently summed as independent commands.

```python
# One observation bin after all input validation:
for b,light,pulse in zip(self.brains,luminances,stimulations):
    b._prepare_neural_input(light,ticks,stimulation=pulse,lamina_bias=lamina_bias)
counts, wall = self.advance(ticks)
for b,c,enabled in zip(self.brains,counts,learning_flags):
    b._apply_centered_rule(c,ticks*b.dt/1000,enabled)
```

Bind checked lane adapters and preserve original backends for lifecycle.
Checkpoint metadata identifies metal-batch and its numerical parent/order/
lane count. Materialize only explicitly. Misaligned restores fail on the
next advance without silently padding/resetting. Poisoned owner never falls
back automatically. Healthy close materializes before releasing ownership;
reentrant calls reject immediately and close waits safely for in-flight work.

- [ ] **Step 4: Verify exact serial/batch and serial refactor controls.**

```sh
python -m pytest -q tests/test_doom_metal_batch.py tests/test_doom_metal_batch_rgb.py tests/test_doom_learning_v6.py tests/test_doom_imitation.py tests/test_doom_metal_checkpoint.py tests/test_doom_metal_batch_native.py
python -m pytest -q
git diff --check
```

The new RGB test file also covers serial VisualMemoryBrain preparation because
the repository has no dedicated existing visual-v6 test file.
Root provides actual M4 focused/full verification. New focused output must
stay warning-free; preserve documented baseline's 50 third-party warnings.

- [ ] **Step 5: Commit and pass independent task review.**

Commit exact owned source/tests using required identity. Report RED/GREEN,
interfaces/provenance, lifecycle/alias safety, exact RGB/control comparisons
and all concerns. Root independently reviews the complete task range before
retained experiments.

### Task 3: Retained identity, original RGB timing and published evidence (root)

**Files:**

- Create: `docs/experiments/2026-09-13-metal-batch-executor-invocation.md`.
- Create: `docs/experiments/2026-09-13-metal-batch-executor.md`.
- Update: `docs/doom-metal-backend.md` with actual results and limitations.
- Publish only bounded portable JSON under `outputs/doom-learning/metal-batch-executor-20260913/`.
- Keep raw checkpoints/traces/logs, scripts and machine origins ignored; add
  specific .gitignore exceptions only for approved bounded portable evidence.

**Interfaces:**

- Consumes reviewed Task1/2 source, fresh actual M4 verification, original pinned
  train/eval episodes and existing CPU/serial-Metal evidence.
- Produces source-identified exact N=1/N=2 per-lane identity/isolation evidence,
  unchanged retained gates, complete matched RGB timing, memory/profiling
  accounting and conservative decision to scale/profile next.

- [ ] **Step 1: Save exact invocation before running owned target jobs.**
  Keep source fixed, pin all input/source/checkpoint SHA256s, require a fresh
  output and zero interference with unrelated workloads. No public stream.

- [ ] **Step 2: Obtain actual M4 native/source/full-suite evidence and retained gates.**

```python
forty = validate.run(out/'parity-40ms')
original = validate.TRACE
validate.TRACE = (('black',False),('blue',False),('green',False),('white',True),
                  ('left_blue',False),('right_blue',True),('vertical',False),
                  ('horizontal',False))
try:
    eighty = validate.run(out/'parity-80ms')
finally:
    validate.TRACE = original
```

Run N=2 complete-state/spike/control comparison against each serial epoch-5
reference through 2 s dark and the original 35-frame RGB prefixes, with
permutation and changed-neighbor controls. Verify the gate remains the same
CPU gate, and disclose its limited horizon and lane-validation scope.
Reproduce/retain two known epoch-5 carry cancellation failures using historical
test-only 71df231 against epoch5 separately, plus the unaffected controls;
do not import epoch6 arithmetic. All full retained graph/nonplastic invariants
must pass. Do not extrapolate long CPU equivalence from 40/80 ms.

- [ ] **Step 3: Measure matched complete RGB workloads, not only tiny prefixes.**

Use original center train 982 frames (28.0571 RGB brain s) and eval 988 frames
(28.2286 s); all original source indices, pixels, nine raw controls and
35 Hz cumulative tick boundaries stay exact. Run serial and N=2 matched
workloads with unchanged teacher min(abs(turn-target)/6,1), next-frame
4*error PPL101 input, eta .001 and frozen evaluation. Canonical semantic
traces/checkpoint efficacies must match serial per lane.
Record phase wall/aggregate brain seconds, initialization/warmup separately,
GPU/encode/wait/transfer/host-rule timing once, per-lane latency, memory
pressure/swap and at least three fresh timing repeats for variability.
Disable full spike/state capture for timing. Reproduce full eight-phase
controls with the validated executor before a credit-rule experiment.

Batch only matched-duration phases. Never keep a shorter/completed lane
advancing to fill a batch. Pair plastic/frozen training on the same 982 frames,
then evaluate learned/frozen on the same 988 frames; shifted training and
erased/shifted/retention evaluations use their own aligned batches and declared
duplicate controls where necessary. Five-second dark retention runs only on
its intended lane (N=1 batch owner is allowed). Duplicate controls are timing/
identity controls, not independent biological or learning replicates.

- [ ] **Step 4: Scale to N=4 only after N=2 identity and memory headroom.**
  Re-run all four per-lane identity/permutation/isolation checks and the matched
  complete RGB timing. Report measured gain or lack thereof. Lack of gain
  directs profiling/refinement with fixed lanes; it does not park Metal.

- [ ] **Step 5: Audit, independent whole-branch review, exact-file commit/push.**
  Cross-check target/local artifact byte/SHA identities, actual source/build
  metadata, original timing/teacher/frozen/erasure causality and full graph.
  Publish portable numerical evidence and failures, no raw game assets.
  Most-capable final review gets branch-start b24249e through final HEAD,
  spec/plan/report/ledger and exact diff package. Fix original-worker findings,
  scoped re-review, then push owned branch. Keep evidence/workspaces needed
  for the active training goal; no main integration or public launch.
