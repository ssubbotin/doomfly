# C++ Multi-Trajectory Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable C++17 executor that advances independent v6 neural trajectories concurrently while sharing one immutable graph and retaining exact single-lane CPU results.

**Architecture:** Python freezes one finalized graph, stores per-edge plastic-slot identifiers, and owns compact mutable lane buffers. A versioned C ABI registers stable pointers once. Persistent native workers assign one whole lane to one worker for each synchronized call, using the existing scalar integration order and a lane-local 4,184-weight overlay.

**Tech Stack:** Python 3, NumPy, `ctypes`, C++17, `clang++`, POSIX/macOS shared libraries, `pytest`, JSON evidence reports.

**Spec:** `docs/superpowers/specs/2026-09-11-cpp-multi-trajectory-executor-design.md`

## Global Constraints

- Retain all MaleCNS v1.0 neurons and released connections. The executor must not prune, merge, crop, or approximate edges.
- Keep RGB processing, reinforcement, centered plasticity, the fixed decoder, ViZDoom, checkpoints, and experiment scheduling outside this milestone.
- Treat `doom_learning_v6/kernel.cpp` and `MemoryBrain._advance_cpu()` as the numerical oracle. A lane must match every retained mutable state bit for bit.
- Freeze the fully constructed brain's baseline weights. This includes any reviewed visual-circuit adjustments made during brain construction.
- Store only the 4,184 plastic values per lane. Every immutable edge reads the shared baseline array.
- Keep the native API portable C++17. Do not add OpenMP, platform-specific thread pools, SIMD, or a new Doom renderer in this milestone.
- Preserve `memory_advance` and all current CPU, Metal, learning, and survival behavior.
- Never stage `outputs/doom-learning/metal-two-dispatch-training-smoke-m4pro/` in implementation commits.

---

## Task 1: Add the Versioned Native ABI and Reproducible Build

**Files:**
- Create: `doom_learning_v6/cpu_batch/__init__.py`
- Create: `doom_learning_v6/cpu_batch/api.h`
- Create: `doom_learning_v6/cpu_batch/executor.cpp`
- Create: `doom_learning_v6/cpu_batch/build.py`
- Create: `tests/test_doom_cpu_batch_build.py`

- [ ] **Step 1: Write failing build and probe tests**

```python
def test_build_records_source_and_binary_identity(tmp_path):
    from doom_learning_v6.cpu_batch.build import ABI_VERSION, build, library_path
    first = build(tmp_path)
    second = build(tmp_path)
    assert first == second
    assert first['abi_version'] == ABI_VERSION == 1
    assert first['sources'].keys() == {'api.h', 'executor.cpp'}
    assert first['binary_sha256']
    assert library_path(tmp_path).exists()


def test_probe_confirms_native_abi(tmp_path):
    from doom_learning_v6.cpu_batch.build import ABI_VERSION, probe
    assert probe(tmp_path)['native_abi_version'] == ABI_VERSION
```

- [ ] **Step 2: Run the tests and confirm the expected import failure**

Run: `pytest -q tests/test_doom_cpu_batch_build.py`

Expected: FAIL because `doom_learning_v6.cpu_batch` does not exist.

- [ ] **Step 3: Define the minimal C ABI**

Use opaque `df_cpu_batch_handle`, ABI version `1`, explicit status codes, and these exports:

```c
uint32_t df_cpu_batch_abi_version(void);
int df_cpu_batch_create(const df_cpu_batch_graph *graph,
    const df_cpu_batch_lane *lanes, int32_t lane_count, int32_t workers,
    df_cpu_batch_handle *handle);
int df_cpu_batch_advance(df_cpu_batch_handle handle, int32_t steps,
    df_cpu_batch_timing *timing);
const char *df_cpu_batch_error(df_cpu_batch_handle handle);
void df_cpu_batch_destroy(df_cpu_batch_handle handle);
```

`df_cpu_batch_graph` contains counts, `dt_ms`, adaptation constants, CSR arrays, shared baseline weights, signed `int16_t` plastic slots, masks, and rest potentials. `df_cpu_batch_lane` contains the mutable pointers listed in the approved design plus a scalar cursor pointer and the compact plastic-weight pointer. `df_cpu_batch_timing` contains native wall seconds, lanes advanced, worker count, and steps.

- [ ] **Step 4: Implement a build module with content-addressed reuse**

Compile with `clang++ -O3 -std=c++17 -shared -fPIC -pthread` on Linux and `clang++ -O3 -std=c++17 -dynamiclib -pthread` on macOS. Write `build.json` atomically with source digests, binary digest, compiler identity, flags, platform, architecture, and ABI version. `probe()` must load with `ctypes.CDLL` and compare the native ABI.

- [ ] **Step 5: Run the focused tests**

Run: `pytest -q tests/test_doom_cpu_batch_build.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add doom_learning_v6/cpu_batch tests/test_doom_cpu_batch_build.py
git commit -m "Add CPU Batch Native ABI"
```

---

## Task 2: Build and Validate Shared Graph and Lane Owners

**Files:**
- Create: `doom_learning_v6/cpu_batch/backend.py`
- Create: `tests/cpu_batch_helpers.py`
- Create: `tests/test_doom_cpu_batch_state.py`
- Modify: `doom_learning_v6/cpu_batch/__init__.py`

- [ ] **Step 1: Write a reusable deterministic toy-brain factory**

Create a four-neuron graph with immutable, plastic, self, and duplicate connections. Use exact NumPy dtypes from `NativeBrain`. Construct `MemoryBrain` with an injected circuit and modulation mask so tests need no external workbook or full graph.

- [ ] **Step 2: Write failing ownership and validation tests**

Cover these assertions:

```python
graph = SharedCpuGraph.from_brain(reference)
assert graph.neurons == reference.n
assert graph.edges == len(reference.post)
assert graph.plastic_edges == len(reference.circuit['edges'])
assert graph.plastic_slot.dtype == np.int16
assert np.array_equal(graph.plastic_slot[reference.circuit['edges']],
                      np.arange(graph.plastic_edges, dtype=np.int16))
assert all(not array.flags.writeable for array in graph.arrays.values())

lane = CpuBatchLane.from_brain(graph, reference)
np.testing.assert_array_equal(lane.plastic_weights,
                              reference.weight[reference.circuit['edges']])
assert not np.shares_memory(lane.v, graph.base_weight)
```

Also reject duplicate plastic edge identifiers, out-of-range identifiers, more than 32,767 plastic slots, wrong dtypes/shapes/contiguity, nonfinite values, CSR errors, a lane built from a different graph identity, and mutable-buffer aliasing across lanes.

- [ ] **Step 3: Run the tests and confirm missing-type failures**

Run: `pytest -q tests/test_doom_cpu_batch_state.py`

Expected: FAIL because graph and lane owners are absent.

- [ ] **Step 4: Implement `SharedCpuGraph.from_brain(brain)`**

Copy `brain.weight` after construction into `base_weight`. Build `plastic_slot` as `-1` followed by the circuit edge mapping. Retain exact graph/configuration arrays needed by the kernel. Make every shared array C-contiguous and read-only. Record SHA-256 identity for `ptr`, `post`, `base_weight`, plastic edges, masks, rest, and scalar dynamics. Expose:

```python
@property
def shared_bytes(self) -> int: ...

def metadata(self) -> dict: ...
```

- [ ] **Step 5: Implement `CpuBatchLane.from_brain(graph, brain)`**

Copy `v`, `g`, `refractory`, `drive`, `previous_drive`, `queue`, `queue_count`, `counts`, `active`, `active_flag`, `nactive`, `last`, `eligibility`, `eligibility_last`, `modulation`, `modulation_last`, `adaptation`, cursor, and plastic weights. Keep the cursor as a one-element contiguous `int64` array for a stable native pointer. Expose `lane_bytes`, `copy_to_brain(brain)`, and `copy_from_brain(graph, brain)` for exact test setup and observation only.

- [ ] **Step 6: Validate executor registration before native creation**

Add the Python constructor signature:

```python
MultiTrajectoryCpuExecutor(
    graph: SharedCpuGraph,
    lanes: Sequence[CpuBatchLane],
    workers: int,
)
```

Require at least one lane, `1 <= workers <= len(lanes)`, distinct mutable storage for every lane, and exact graph identity. Hold strong references to graph and lanes. Native creation can remain unimplemented behind a clear `BackendError` until Task 3.

- [ ] **Step 7: Run focused and existing state tests**

Run: `pytest -q tests/test_doom_cpu_batch_state.py tests/test_doom_learning_v6.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add doom_learning_v6/cpu_batch tests/cpu_batch_helpers.py tests/test_doom_cpu_batch_state.py
git commit -m "Add Shared CPU Batch State"
```

---

## Task 3: Implement Exact Overlay-Aware Single-Lane Advance

**Files:**
- Modify: `doom_learning_v6/cpu_batch/api.h`
- Modify: `doom_learning_v6/cpu_batch/executor.cpp`
- Modify: `doom_learning_v6/cpu_batch/backend.py`
- Create: `tests/test_doom_cpu_batch_parity.py`

- [ ] **Step 1: Write a state comparison helper**

Compare bitwise equality after every call for `counts`, cursor, voltage, conductance, refractory state, drive history, delay queue and counts, active queue and flags, last-update ticks, eligibility and its ticks, modulation and its ticks, adaptation, and plastic weights.

- [ ] **Step 2: Write failing single-lane oracle tests**

For step sequences `(1, 18, 22, 37, 100)`, start the legacy CPU brain and batch lane from identical state, apply identical changing drive before each call, clear counts on both sides, call `reference._advance_cpu(steps)` and `executor.advance(steps)`, then compare every retained state. Include:

- immutable, plastic, self, and duplicate edge delivery;
- a plastic overlay changed between calls while `graph.base_weight` remains unchanged;
- modulatory-neuron delivery and KC eligibility updates;
- refractory reset across the 19-slot delay queue;
- inactive-neuron wakeup after a drive change;
- invalid step counts `0` and `101` without any state mutation.

- [ ] **Step 3: Run parity tests and confirm advance is unavailable**

Run: `pytest -q tests/test_doom_cpu_batch_parity.py`

Expected: FAIL at executor creation or advance.

- [ ] **Step 4: Port the scalar kernel with an overlay accessor**

Preserve the statement and loop order from `memory_advance`. Replace only the delivered weight lookup:

```cpp
inline float edge_weight(const df_cpu_batch_graph &graph,
                         const df_cpu_batch_lane &lane, int64_t edge) {
  const int16_t slot = graph.plastic_slot[edge];
  return slot < 0 ? graph.base_weight[edge] : lane.plastic_weight[slot];
}
```

Use this value for ordinary synaptic conductance and modulatory magnitude. Keep all released connections, including self and duplicate edges. Keep learning disabled inside the native executor; update eligibility exactly as the oracle does. Materialize all neurons at the observation boundary.

- [ ] **Step 5: Wire native descriptors through `ctypes`**

Define structures whose field order exactly matches `api.h`, set `argtypes` and `restype`, create the native handle after Python validation, and convert every nonzero native status into `BackendError` with the handle error text. `advance(steps)` must clear every lane's counts, call native code, return a list of count-array copies, and update timing metadata.

- [ ] **Step 6: Run parity and regression tests**

Run: `pytest -q tests/test_doom_cpu_batch_parity.py tests/test_doom_learning_v6.py tests/test_doom_backend_contract.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add doom_learning_v6/cpu_batch tests/test_doom_cpu_batch_parity.py
git commit -m "Add Exact CPU Batch Lane Advance"
```

---

## Task 4: Add Persistent Multi-Lane Workers and Lifecycle Safety

**Files:**
- Modify: `doom_learning_v6/cpu_batch/executor.cpp`
- Modify: `doom_learning_v6/cpu_batch/backend.py`
- Create: `tests/test_doom_cpu_batch_executor.py`

- [ ] **Step 1: Write failing concurrency tests**

Create four lanes with distinct drives and plastic overlays. Compare each lane with an independent legacy CPU oracle. Require bitwise equality for:

- workers `1`, `2`, and `4`;
- original and reversed lane order;
- five repeated fresh runs;
- consecutive calls with steps `1`, `40`, `80`, and `100`;
- one lane changed while all neighboring lane digests remain unchanged.

Also test concurrent Python calls to the same handle, close followed by advance, idempotent close, context-manager cleanup, and construction with aliased lane storage.

- [ ] **Step 2: Run the tests and confirm concurrency/lifecycle failures**

Run: `pytest -q tests/test_doom_cpu_batch_executor.py`

Expected: FAIL because persistent workers and lifecycle guards are absent.

- [ ] **Step 3: Implement the persistent worker pool**

The handle owns copied descriptors, `std::thread` workers, one mutex, two condition variables, an atomic next-lane counter, generation number, remaining-worker count, closing flag, running flag, poison flag, and error string. Each worker must:

1. sleep until a new generation or shutdown;
2. claim lane indices atomically;
3. advance one complete lane at a time;
4. record the first exception and poison the handle;
5. signal completion after its claims finish.

The calling thread publishes the common step count, wakes workers, waits for generation completion, and returns after all lane writes are visible. Scheduling must never split one lane across workers during a call.

- [ ] **Step 4: Enforce lifecycle rules**

Reject reentrant `advance()` before touching state. Reject `advance()` after close. `destroy()` must wait for an active generation, request shutdown, join every worker, and free the handle. Native validation must finish before threads start so malformed descriptors cannot cause partial work.

- [ ] **Step 5: Report timing and memory ownership**

Return native wall seconds, lane count, worker count, and steps. Python metadata must include graph identity, `shared_graph_bytes`, `lane_bytes`, `total_lane_bytes`, ABI/build identity, and closed/poisoned state without exposing machine-specific paths.

- [ ] **Step 6: Run focused, repeated, and broad neural tests**

Run: `pytest -q tests/test_doom_cpu_batch_executor.py tests/test_doom_cpu_batch_parity.py tests/test_doom_cpu_batch_state.py tests/test_doom_learning_v6.py tests/test_doom_metal_parity.py`

Expected: PASS, with Metal-only tests skipped where unavailable.

- [ ] **Step 7: Commit**

```bash
git add doom_learning_v6/cpu_batch tests/test_doom_cpu_batch_executor.py
git commit -m "Add Persistent CPU Batch Workers"
```

---

## Task 5: Add Full-Graph Validation and Scaling Evidence Tools

**Files:**
- Create: `doom_learning_v6/cpu_batch/validate.py`
- Create: `doom_learning_v6/cpu_batch/benchmark.py`
- Create: `tests/test_doom_cpu_batch_validation.py`
- Create: `tests/test_doom_cpu_batch_benchmark.py`

- [ ] **Step 1: Write failing report-contract tests with injected toy factories**

Validation must reject an existing output directory and report:

```json
{
  "schema": 1,
  "experiment": "full-graph exact CPU batch parity",
  "structure": {"release": "MaleCNS v1.0", "neurons": 166700,
                "edges": 25582938, "plastic_edges": 4184},
  "horizons_ms": [40, 80],
  "state_bitwise_equal": true,
  "decoder_equal": true,
  "structural_identity_equal": true,
  "passed": true,
  "learning_demonstrated": false,
  "biologically_validated": false
}
```

Benchmark must require at least five repetitions and lane counts exactly `[1, 2, 4]`. It must report raw contiguous samples, medians, aggregate simulated brain-seconds per wall-second, per-lane latency, scaling efficiency relative to one lane, peak RSS, shared graph bytes, lane bytes, workers, source/build identity, and system load. It must make no fixed speedup pass claim before measurement.

- [ ] **Step 2: Run report tests and confirm missing-module failures**

Run: `pytest -q tests/test_doom_cpu_batch_validation.py tests/test_doom_cpu_batch_benchmark.py`

Expected: FAIL because validation and benchmark modules are absent.

- [ ] **Step 3: Implement exact validation**

Build one calibrated CPU reference and a shared graph from the same finalized configuration. For deterministic inputs at 40 ms and 80 ms, compare digests of every retained neural state, all counts, and fixed `NeuralControls` decisions. Repeat the four-lane batch and require identical per-lane digests. Verify exact release counts, structural/source-lock digests, and 4,184 slots. Save inputs and reports atomically in a fresh output directory.

- [ ] **Step 4: Implement the scaling benchmark**

Warm each lane count once, then collect five contiguous repetitions for 1, 2, and 4 lanes. Restore identical initial lane state before every repetition. Sample process peak RSS and record physical/logical CPU counts and load average. Keep correctness gates separate from observed performance fields.

- [ ] **Step 5: Run report tests**

Run: `pytest -q tests/test_doom_cpu_batch_validation.py tests/test_doom_cpu_batch_benchmark.py`

Expected: PASS.

- [ ] **Step 6: Run the local CPU batch suite**

Run: `pytest -q tests/test_doom_cpu_batch_build.py tests/test_doom_cpu_batch_state.py tests/test_doom_cpu_batch_parity.py tests/test_doom_cpu_batch_executor.py tests/test_doom_cpu_batch_validation.py tests/test_doom_cpu_batch_benchmark.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add doom_learning_v6/cpu_batch tests/test_doom_cpu_batch_validation.py tests/test_doom_cpu_batch_benchmark.py
git commit -m "Add CPU Batch Validation and Benchmarks"
```

---

## Task 6: Validate the Full Graph on M4 Pro and Document Measured Results

**Files:**
- Create: `docs/doom-cpu-batch.md`
- Create: `outputs/doom-learning/cpu-batch-validation-m4pro/report.json`
- Create: `outputs/doom-learning/cpu-batch-benchmark-m4pro/report.json`
- Modify: `.gitignore` only if generated native binaries are not already covered

- [ ] **Step 1: Run all local regression tests**

Run: `pytest -q`

Expected: PASS with platform-specific skips only.

- [ ] **Step 2: Prepare an isolated checkout on `ssh mini`**

Use a fresh remote worktree for this branch. Do not modify `/Users/sergey/doomfly`, which can contain user work. Link or configure the existing MaleCNS graph through ignored local configuration. Confirm M4 Pro, available memory, graph path, branch commit, and source digests before running.

- [ ] **Step 3: Run exact full-graph validation**

```bash
python -m doom_learning_v6.cpu_batch.validate \
  --out outputs/doom-learning/cpu-batch-validation-m4pro
```

Expected: process exit `0`; exact 166,700 neurons, 25,582,938 edges, 4,184 plastic slots; bitwise state parity at both horizons; fixed decoder equality; repeated batch equality.

- [ ] **Step 4: Run the scaling benchmark**

```bash
python -m doom_learning_v6.cpu_batch.benchmark \
  --validation outputs/doom-learning/cpu-batch-validation-m4pro/report.json \
  --out outputs/doom-learning/cpu-batch-benchmark-m4pro \
  --repetitions 5
```

Expected: a complete report for one, two, and four lanes. Record measured throughput and memory regardless of whether scaling is favorable.

- [ ] **Step 5: Copy only small evidence reports back and verify hashes**

Exclude graph data, native binaries, dependency checkouts, workbooks, papers, and game assets. Compare SHA-256 values on both machines before using the reports.

- [ ] **Step 6: Document architecture, use, scope, and evidence**

`docs/doom-cpu-batch.md` must explain shared immutable graph ownership, compact lane overlays, exactness guarantees, worker scheduling, build/validation commands, measured M4 Pro results, memory accounting, and current limitations. State explicitly that propagation parity and throughput do not establish learning or biological validity.

- [ ] **Step 7: Verify documentation and evidence consistency**

Run:

```bash
pytest -q
python -m doom_learning_v6.cpu_batch.build --probe
git diff --check
rg -n "TO""DO|T""BD|place""holder|learning demonstrated|biologically validated" \
  doom_learning_v6/cpu_batch tests/test_doom_cpu_batch_*.py docs/doom-cpu-batch.md
```

Expected: tests and probe pass; diff check is clean; no unfinished markers or unsupported scientific claims.

- [ ] **Step 8: Commit and push the validated implementation**

```bash
git add .gitignore doom_learning_v6/cpu_batch tests/cpu_batch_helpers.py \
  tests/test_doom_cpu_batch_*.py docs/doom-cpu-batch.md \
  outputs/doom-learning/cpu-batch-validation-m4pro/report.json \
  outputs/doom-learning/cpu-batch-benchmark-m4pro/report.json
git commit -m "Validate Shared CPU Trajectory Execution"
git push origin metal-backend
```

Before committing, inspect `git status --short` and remove the unrelated Metal training directory from the staging set if present.

---

## Completion Gates

- [ ] Every focused test was observed failing before its implementation and passing afterward.
- [ ] All existing repository tests pass on Linux and all applicable tests pass on the M4 Pro.
- [ ] One lane matches the legacy CPU oracle bit for bit after every tested call.
- [ ] Results are invariant across worker counts, lane order, neighboring lane state, and repeated execution.
- [ ] Shared arrays remain read-only; mutable buffers never alias; only 4,184 weights are stored per lane.
- [ ] Full-graph reports prove exact structural retention and fixed-decoder equality at 40 ms and 80 ms.
- [ ] Benchmark reports preserve raw samples and separate numerical correctness from throughput.
- [ ] Documentation matches the implemented and measured executor without making a learning or biological-validation claim.
- [ ] Source, reports, commits, and remote branch agree by digest and commit identifier.
