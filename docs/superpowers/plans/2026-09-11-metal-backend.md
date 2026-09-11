# Portable Metal Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic direct Objective-C++/Metal execution backend for the complete DOOMFLY v6 neural model, validate it against the CPU reference on an M4 Pro, and gate offline training on parity and performance evidence.

**Architecture:** `MemoryBrain` delegates neural advancement to an explicit backend while Python retains sensory processing, centered plasticity, controls, and experiment orchestration. The Metal backend owns shared state buffers, uses outgoing CSR for target marking and a complete incoming CSR for stable target-wise accumulation, then exposes backend-independent checkpoints and provenance.

**Tech Stack:** Python 3.11, NumPy, pytest, C++17, Objective-C++, Apple Metal compute shaders, Apple Foundation and Metal frameworks, `ctypes`, ViZDoom.

**Spec:** `docs/superpowers/specs/2026-09-11-metal-backend-design.md`

## Global Constraints

- Retain all 166,700 neurons and all 25,582,938 directed MaleCNS v1.0 connections, including weak, duplicate, self, and modulatory edges.
- Preserve `doom_learning_v6/kernel.cpp` as the CPU scientific reference and keep `cpu` as the default backend.
- Keep RGB input, neural propagation, reinforcement, centered plasticity, and fixed decoding separate and traceable.
- Target arm64 macOS, unified memory, and Apple GPU family 7 or newer; validate first on the M4 Pro host `mini`.
- Use 32-bit integer atomics only. Exclude floating-point atomics, 64-bit atomics, Metal 4-only features, ray tracing, and advanced SIMD-group features.
- Preserve the 0.1 ms neural timestep, 1.8 ms delay, 19 delay slots, 2.2 ms refractory period, and 10 ms centered-plasticity boundary.
- Fail explicitly on unsupported hardware, stale binaries, ABI mismatch, graph mismatch, GPU errors, and missing parity evidence. Never fall back silently to CPU.
- Generated `.dylib`, `.air`, `.metallib`, cached incoming indices, downloaded data, and mutable checkpoints remain untracked.
- Metal may reduce wall time; it must not change the simulated experience or establish learning by itself.

---

### Task 1: Backend Contract and CPU Reference Adapter

**Files:**
- Create: `doom_learning_v6/backend.py`
- Modify: `doom_learning_v6/brain.py`
- Modify: `tests/test_doom_learning_v6.py`
- Create: `tests/test_doom_backend_contract.py`

**Interfaces:**
- Produces: `BackendError`, `CpuBackend`, `create_backend(name: str, brain: MemoryBrain)`, and backend methods `advance(steps: int) -> float`, `sync_for_checkpoint() -> None`, `restore_from_host() -> None`, `metadata() -> dict`, and `close() -> None`.
- Preserves: `MemoryBrain(..., backend='cpu')`, `MemoryBrain.step(...) -> tuple[np.ndarray, float]`, and all existing public arrays and checkpoint methods.

- [x] **Step 1: Write failing backend-selection tests**

```python
def test_cpu_is_default_and_unknown_backend_fails(tmp_path):
    b = brain(tmp_path)
    assert b.backend.name == 'cpu'
    with pytest.raises(ValueError, match='Unknown neural backend'):
        brain(tmp_path, backend='cuda')

def test_cpu_backend_preserves_existing_trace(tmp_path):
    default = brain(tmp_path)
    explicit = brain(tmp_path, backend='cpu')
    a, _ = default.step([], 20, stimulation=([0], 20), lamina_bias=0)
    b, _ = explicit.step([], 20, stimulation=([0], 20), lamina_bias=0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(default.v, explicit.v)
```

- [x] **Step 2: Run the focused tests and confirm the new keyword fails**

Run: `python -m pytest tests/test_doom_backend_contract.py -q`

Expected: failure reporting that `MemoryBrain.__init__` does not accept `backend`.

- [x] **Step 3: Implement the backend contract and CPU adapter**

```python
class BackendError(RuntimeError):
    pass

class CpuBackend:
    name = 'cpu'
    def __init__(self, brain): self.brain = brain
    def advance(self, steps): return self.brain._advance_cpu(steps)
    def sync_for_checkpoint(self): return None
    def restore_from_host(self): return None
    def metadata(self): return {'name': self.name, 'build': self.brain.build}
    def close(self): return None

def create_backend(name, brain):
    if name == 'cpu': return CpuBackend(brain)
    if name == 'metal':
        from .metal.backend import MetalBackend
        return MetalBackend(brain)
    raise ValueError(f'Unknown neural backend: {name}')
```

Move the existing `ctypes` invocation into `MemoryBrain._advance_cpu(steps)` and
make `_neural_step` call `self.backend.advance(steps)`. Construct the selected
backend lazily on the first neural step so `VisualMemoryBrain` can finish its R8
weight corrections before Metal copies graph state. Change the shared fixture to
`def brain(tmp_path, **kwargs)` and forward `**kwargs` to `MemoryBrain`.

- [x] **Step 4: Run CPU regression tests**

Run: `python -m pytest tests/test_doom_backend_contract.py tests/test_doom_learning_v6.py tests/test_doom_live_training.py -q`

Expected: all tests pass with unchanged CPU results.

- [x] **Step 5: Commit the contract**

```bash
git add doom_learning_v6/backend.py doom_learning_v6/brain.py tests/test_doom_learning_v6.py tests/test_doom_backend_contract.py
git commit -m "Add v6 neural backend contract"
```

### Task 2: Deterministic Incoming CSR Builder

**Files:**
- Create: `doom_learning_v6/metal/__init__.py`
- Create: `doom_learning_v6/metal/graph.py`
- Create: `tests/test_doom_metal_graph.py`

**Interfaces:**
- Produces: `IncomingGraph(ptr: np.ndarray, pre: np.ndarray, edge: np.ndarray, metadata: dict)` and `load_or_build_incoming(out_ptr, out_post, cache_dir) -> IncomingGraph`.
- Consumes: contiguous outgoing `ptr:int64[n+1]` and `post:int32[e]` arrays.

- [x] **Step 1: Write failing ordering and integrity tests**

```python
def test_incoming_csr_is_stable_complete_permutation(tmp_path):
    out_ptr = np.array([0, 3, 4, 6], dtype=np.int64)
    out_post = np.array([1, 1, 0, 1, 2, 1], dtype=np.int32)
    incoming = load_or_build_incoming(out_ptr, out_post, tmp_path)
    np.testing.assert_array_equal(incoming.ptr, [0, 1, 5, 6])
    np.testing.assert_array_equal(incoming.edge, [2, 0, 1, 3, 5, 4])
    np.testing.assert_array_equal(incoming.pre, [0, 0, 0, 1, 2, 2])
    np.testing.assert_array_equal(np.sort(incoming.edge), np.arange(6))

def test_corrupt_cached_index_is_rejected(tmp_path):
    incoming = load_or_build_incoming(PTR, POST, tmp_path)
    incoming.edge[0] = incoming.edge[1]
    np.savez(tmp_path/'incoming.npz', ptr=incoming.ptr, pre=incoming.pre,
             edge=incoming.edge, metadata=json.dumps(incoming.metadata))
    with pytest.raises(ValueError, match='permutation'):
        load_or_build_incoming(PTR, POST, tmp_path)
```

- [x] **Step 2: Run the tests and confirm the import fails**

Run: `python -m pytest tests/test_doom_metal_graph.py -q`

Expected: failure because `doom_learning_v6.metal.graph` is absent.

- [x] **Step 3: Implement stable construction, validation, and atomic caching**

```python
order = np.argsort(out_post, kind='stable').astype(np.int32, copy=False)
degrees = np.diff(out_ptr).astype(np.int64, copy=False)
source = np.repeat(np.arange(len(degrees), dtype=np.int32), degrees)
incoming_ptr = np.r_[0, np.cumsum(np.bincount(out_post,
    minlength=len(degrees)), dtype=np.int64)]
incoming = IncomingGraph(incoming_ptr, source[order], order, metadata)
```

Validate shapes, dtypes, contiguity, endpoints, the exact edge permutation, and
the source/post pair for every incoming entry. Hash both outgoing inputs, builder
source, and all produced arrays. Write `incoming.npz.partial`, then rename it to
`incoming.npz`.

- [x] **Step 4: Run graph tests and a medium random-graph test**

Run: `python -m pytest tests/test_doom_metal_graph.py -q`

Expected: all tests pass, including cache reload with identical hashes.

- [x] **Step 5: Commit the graph index**

```bash
git add doom_learning_v6/metal/__init__.py doom_learning_v6/metal/graph.py tests/test_doom_metal_graph.py
git commit -m "Add deterministic incoming graph index"
```

### Task 3: Metal Build and Capability Probe

**Files:**
- Create: `doom_learning_v6/metal/api.h`
- Create: `doom_learning_v6/metal/build.py`
- Create: `doom_learning_v6/metal/backend.mm`
- Create: `doom_learning_v6/metal/kernels.metal`
- Modify: `.gitignore`
- Create: `tests/test_doom_metal_build.py`

**Interfaces:**
- Produces: `build(output_dir: Path) -> dict`, `library_path(output_dir: Path) -> Path`, and C functions `df_metal_probe`, `df_metal_last_error`, and `df_metal_destroy`.
- Metadata schema: source hashes, binary hashes, commands, compiler, SDK, macOS, architecture, and Metal language version.

- [x] **Step 1: Write failing platform and metadata tests**

```python
def test_build_rejects_non_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'platform', 'linux')
    with pytest.raises(RuntimeError, match='arm64 macOS'):
        build(tmp_path)

def test_generated_artifacts_are_ignored():
    ignored = subprocess.run(['git', 'check-ignore', '-q',
        'outputs/doom-learning/metal/libmemory-metal.dylib'], check=False)
    assert ignored.returncode == 0
    assert subprocess.run(['git', 'check-ignore', '-q',
        'outputs/doom-learning/metal/kernels.metallib']).returncode == 0
```

- [x] **Step 2: Run the tests and confirm missing modules or ignore rules fail**

Run: `python -m pytest tests/test_doom_metal_build.py -q`

Expected: failure before build support exists.

- [x] **Step 3: Define the initial C ABI and probe**

```c
typedef void *df_metal_handle;
typedef struct {
  uint32_t abi_version;
  uint32_t supported;
  uint32_t has_unified_memory;
  uint32_t supports_apple7;
  uint64_t recommended_working_set;
  char device_name[256];
} df_metal_device_info;

int df_metal_probe(df_metal_device_info *info);
const char *df_metal_last_error(void);
void df_metal_destroy(df_metal_handle handle);
```

Implement `df_metal_probe` with `MTLCreateSystemDefaultDevice`,
`hasUnifiedMemory`, and `supportsFamily:MTLGPUFamilyApple7`. Return a stable
nonzero error code and thread-local error string on failure.

- [x] **Step 4: Implement the atomic build**

Run `xcrun metal` for `kernels.metal`, `xcrun metallib` for the AIR output, and
`clang++ -std=c++17 -shared -fPIC -arch arm64 -framework Foundation -framework Metal`
for `backend.mm`. Build into `.partial` files, hash them, then rename only after
all commands succeed. Add `*.air` and `*.metallib` to `.gitignore`.

- [x] **Step 5: Run local platform tests and remote M4 probe**

Run locally: `python -m pytest tests/test_doom_metal_build.py -q`

Run on M4 Pro: `ssh mini 'cd doomfly && python3 -m doom_learning_v6.metal.build --probe'`

Expected: local guard tests pass; remote JSON reports arm64, unified memory,
Apple family 7 support, source hashes, and the M4 Pro device name.

- [x] **Step 6: Commit the build path**

```bash
git add .gitignore doom_learning_v6/metal/api.h doom_learning_v6/metal/build.py doom_learning_v6/metal/backend.mm doom_learning_v6/metal/kernels.metal tests/test_doom_metal_build.py
git commit -m "Add portable Metal build and device probe"
```

### Task 4: Metal State Ownership and C ABI

**Files:**
- Modify: `doom_learning_v6/metal/api.h`
- Modify: `doom_learning_v6/metal/backend.mm`
- Create: `doom_learning_v6/metal/backend.py`
- Create: `tests/test_doom_metal_state.py`

**Interfaces:**
- Produces: `MetalBackend(brain)`, opaque Metal handles, full state upload/export, plastic-weight updates, and `metadata()`.
- Consumes: the `IncomingGraph` from Task 2 and build artifacts from Task 3.

- [ ] **Step 1: Write a Darwin-only round-trip state test**

```python
@pytest.mark.skipif(sys.platform != 'darwin', reason='Metal requires macOS')
def test_metal_state_round_trip_preserves_all_arrays(tmp_path):
    b = brain(tmp_path, backend='metal')
    before = {name: getattr(b, name).copy() for name in b.fields}
    b.backend.ensure_initialized()
    b.backend.sync_for_checkpoint()
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(b, name), expected, err_msg=name)
    assert b.backend.metadata()['name'] == 'metal'
```

- [ ] **Step 2: Run on `mini` and confirm creation is absent**

Run: `ssh mini 'cd doomfly && python3 -m pytest tests/test_doom_metal_state.py -q'`

Expected: failure because the full Metal handle API is not implemented.

- [ ] **Step 3: Extend the C ABI with exact graph and state descriptors**

```c
typedef struct {
  int32_t neurons;
  int64_t edges;
  int32_t delay_slots;
  float dt_ms;
  const int64_t *out_ptr;
  const int32_t *out_post;
  const int64_t *in_ptr;
  const int32_t *in_pre;
  const int32_t *in_edge;
  const uint8_t *kc_mask;
  const uint8_t *modulation_mask;
} df_metal_graph;

int df_metal_create(const df_metal_graph *graph, df_metal_handle *handle);
int df_metal_upload_state(df_metal_handle handle, const df_metal_state *state);
int df_metal_download_state(df_metal_handle handle, df_metal_state *state);
int df_metal_update_weights(df_metal_handle handle, int32_t count,
    const int64_t *edge_ids, const float *values);
```

Define `df_metal_state` in `api.h` with pointers for `weight`, `v`, `g`,
`refractory`, `drive`, `previous_drive`, `queue`, `queue_count`, `counts`,
`active`, `active_flag`, `nactive`, `last`, `modulation`, `modulation_last`,
`rest`, and `adaptation`. Validate every pointer, size, and graph count before
allocating buffers.

- [ ] **Step 4: Implement Objective-C++ buffer ownership**

Allocate `MTLStorageModeShared` buffers for immutable graph arrays, incoming CSR,
mutable float32/int state, delay bitmaps, touched flags, and tick-resolved KC
events. Convert host queue lists to ring bitmaps on upload and materialize sorted
lists on download. Reject duplicate IDs within a delay slot and inconsistent
`queue_count`, `active`, or `active_flag` state.

- [ ] **Step 5: Implement the Python wrapper**

Load only binaries whose metadata hashes match `api.h`, `backend.mm`, and
`kernels.metal`. Define matching `ctypes.Structure` layouts, keep all NumPy inputs
alive for each call, translate nonzero C status into `BackendError`, and poison
the instance after command failure.

- [ ] **Step 6: Run round-trip and CPU tests**

Run on M4 Pro: `ssh mini 'cd doomfly && python3 -m pytest tests/test_doom_metal_state.py tests/test_doom_backend_contract.py -q'`

Expected: exact round-trip state and unchanged CPU behavior.

- [ ] **Step 7: Commit state ownership**

```bash
git add doom_learning_v6/metal/api.h doom_learning_v6/metal/backend.mm doom_learning_v6/metal/backend.py tests/test_doom_metal_state.py
git commit -m "Add Metal neural state bridge"
```

### Task 5: Deterministic Metal Neural Kernels

**Files:**
- Modify: `doom_learning_v6/metal/kernels.metal`
- Modify: `doom_learning_v6/metal/backend.mm`
- Modify: `doom_learning_v6/metal/api.h`
- Create: `tests/test_doom_metal_parity.py`

**Interfaces:**
- Produces: `df_metal_advance(handle, steps, kc_events, capacity, event_count, timing)` and deterministic 0.1 ms execution.
- Preserves: CPU v6 lazy evolution, threshold, reset, delay, refractory, adaptation, modulation, and complete edge delivery semantics.

- [ ] **Step 1: Write Darwin-only micrograph parity cases**

```python
@pytest.mark.skipif(sys.platform != 'darwin', reason='Metal requires macOS')
@pytest.mark.parametrize('case', [
    'self_edge', 'duplicate_edges', 'excitatory_inhibitory',
    'refractory_arrival', 'changed_drive', 'modulatory_delivery'])
def test_metal_matches_cpu_micrograph(case, tmp_path):
    cpu, metal = paired_brains(case, tmp_path)
    for drive, milliseconds in stimulus_trace(case):
        expected, _ = cpu.step([], milliseconds, stimulation=drive, lamina_bias=0)
        actual, _ = metal.step([], milliseconds, stimulation=drive, lamina_bias=0)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(metal.v, cpu.v, rtol=1e-5, atol=.002)
        np.testing.assert_allclose(metal.g, cpu.g, rtol=1e-5, atol=.002)
```

Add explicit assertions for the 18-tick delivery time, 22-tick refractory state,
stable repeated Metal results, and retained negative, zero, self, and duplicate
edge contributions.

- [ ] **Step 2: Run on `mini` and confirm advancement fails**

Run: `ssh mini 'cd doomfly && python3 -m pytest tests/test_doom_metal_parity.py -q'`

Expected: failure because `df_metal_advance` is absent.

- [ ] **Step 3: Implement the shared lazy-evolution function**

```metal
inline void evolve(uint i, long now, float current,
    device float *v, device float *g, device short *refractory,
    device long *last, device const float *rest, device float *adaptation,
    float dt, float adaptation_tau) {
  long d = now - last[i];
  if (d <= 0) return;
  int frozen = refractory[i] > 0 ? refractory[i] - 1 : 0;
  int skip = min((int)d, frozen);
  if (skip > 0 && adaptation[i] > 0.0f)
    adaptation[i] *= exp(-dt * skip / adaptation_tau);
  refractory[i] = d >= refractory[i] ? 0 : refractory[i] - (short)d;
  d -= skip;
  if (d > 0) {
    float a = exp(-dt * d / 20.0f), b = exp(-dt * d / 5.0f);
    v[i] = rest[i] + (v[i] - rest[i]) * a + current * (1.0f - a)
      + g[i] * (a - b) / 3.0f;
    g[i] *= b;
    if (adaptation[i] > 0.0f) {
      float c = exp(-dt * d / adaptation_tau);
      v[i] -= adaptation[i] * adaptation_tau
        / (adaptation_tau - 20.0f) * (c - a);
      adaptation[i] *= c;
    }
  }
  last[i] = now;
}
```

Port the CPU adaptation term exactly after the base voltage update. Use one
neuron-indexed thread for drive changes, integration, reset, and final
materialization.

- [ ] **Step 4: Implement the delayed ring and deterministic delivery**

For each tick, encode separate drive/integrate, target-mark, target-gather,
slot-clear, and future-reset dispatches. Use `atomic_fetch_or_explicit` only for
spike and touched bit flags. In target gather, traverse `in_ptr[j]:in_ptr[j+1]`
in stored order, test the current presynaptic bitmap, sum nonmodulatory weights
locally, and update the modulatory trace once plus its stable local sum. Clear
the delivered slot only after gather completes.

- [ ] **Step 5: Add command completion and timing**

Encode all ticks into one command buffer with device-wide phase ordering. Wait
for completion, check `MTLCommandBufferStatusCompleted`, copy GPU start/end time
when available, and return host elapsed time. Mark the handle poisoned on every
other terminal status.

- [ ] **Step 6: Run micrograph parity repeatedly**

Run: `for run in 1 2 3; do ssh mini 'cd doomfly && python3 -m pytest tests/test_doom_metal_parity.py -q' || exit 1; done`

Expected: exact spike events and counters, float32 state within the specified
tolerance, and bitwise-identical repeated Metal checkpoints.

- [ ] **Step 7: Commit neural execution**

```bash
git add doom_learning_v6/metal/api.h doom_learning_v6/metal/backend.mm doom_learning_v6/metal/kernels.metal tests/test_doom_metal_parity.py
git commit -m "Implement deterministic Metal neural propagation"
```

### Task 6: Host Float64 State and Cross-Backend Checkpoints

**Files:**
- Modify: `doom_learning_v6/metal/backend.mm`
- Modify: `doom_learning_v6/metal/backend.py`
- Modify: `doom_learning_v6/brain.py`
- Modify: `tests/test_doom_metal_parity.py`
- Create: `tests/test_doom_metal_checkpoint.py`

**Interfaces:**
- Consumes: tick-resolved `(tick:int64, neuron:int32)` KC events from Metal.
- Produces: exact host-owned `eligibility`, `eligibility_last`, `rate_kc`, `rate_dan`, `memory_u`, and `memory_w` state plus backend-independent checkpoints.

- [ ] **Step 1: Write failing eligibility and checkpoint migration tests**

```python
def test_cpu_checkpoint_continues_on_metal(tmp_path):
    cpu = brain(tmp_path, backend='cpu')
    cpu.step([], 20, stimulation=([0], 20), lamina_bias=0)
    path = tmp_path/'cpu.npz'; cpu.checkpoint(path)
    expected, _ = cpu.step([], 20, stimulation=([0, 2], 20), lamina_bias=0)
    metal = brain(tmp_path, backend='metal'); metal.restore(path)
    actual, _ = metal.step([], 20, stimulation=([0, 2], 20), lamina_bias=0)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(metal.eligibility, cpu.eligibility)

def test_metal_checkpoint_repeats_bitwise(tmp_path):
    first = brain(tmp_path, backend='metal')
    first.step([], 30, stimulation=([0], 20), lamina_bias=0)
    path = tmp_path/'metal.npz'; first.checkpoint(path)
    expected, _ = first.step([], 30, stimulation=([0, 2], 20), lamina_bias=0)
    second = brain(tmp_path, backend='metal'); second.restore(path)
    actual, _ = second.step([], 30, stimulation=([0, 2], 20), lamina_bias=0)
    np.testing.assert_array_equal(actual, expected)
```

- [ ] **Step 2: Run on `mini` and confirm float64 or producer checks fail**

Run: `ssh mini 'cd doomfly && python3 -m pytest tests/test_doom_metal_checkpoint.py -q'`

Expected: failure before host KC-event processing and producer-independent restore exist.

- [ ] **Step 3: Implement exact host eligibility updates**

Sort returned KC events by `(tick, neuron)` and run this C++ equation after a
successful command:

```cpp
const auto delta = tick - eligibility_last[neuron];
eligibility[neuron] *= std::exp(-dt * delta / tau_elig_ms);
eligibility[neuron] += 1.0;
eligibility_last[neuron] = tick;
```

Allocate event capacity as `kc_count * (1 + (steps - 1) / refractory_ticks)` and
reject any impossible overflow. Keep all centered-rule float64 arrays in Python,
where the existing `rule.advance` remains unchanged.

- [ ] **Step 4: Make checkpoint provenance producer-independent**

Store `producer_backend` and backend metadata in the checkpoint. Continue to
validate model, configuration, graph, rule, sensory mapping, and plastic-edge
hashes. Exclude producer binary identity from compatibility checks. Call
`sync_for_checkpoint()` before writing and `restore_from_host()` after all host
arrays validate during restore.

- [ ] **Step 5: Run checkpoint and existing recovery suites**

Run locally: `python -m pytest tests/test_doom_learning_v6.py tests/test_doom_live_training.py -q`

Run on M4 Pro: `ssh mini 'cd doomfly && python3 -m pytest tests/test_doom_metal_checkpoint.py tests/test_doom_metal_parity.py -q'`

Expected: CPU checkpoints remain exact; both migration directions pass; repeated
Metal restoration is bitwise identical.

- [ ] **Step 6: Commit checkpoint continuity**

```bash
git add doom_learning_v6/brain.py doom_learning_v6/metal/backend.mm doom_learning_v6/metal/backend.py tests/test_doom_metal_parity.py tests/test_doom_metal_checkpoint.py
git commit -m "Preserve v6 state across CPU and Metal"
```

### Task 7: Full-Graph Parity Evidence

**Files:**
- Create: `doom_learning_v6/metal/validate.py`
- Create: `tests/test_doom_metal_validation.py`
- Modify: `doom_learning_v6/metal/backend.py`

**Interfaces:**
- Produces: `python -m doom_learning_v6.metal.validate --out PATH`, a JSON report containing structural hashes, spike metrics, decoder comparison, weight comparison, device/build provenance, and pass/fail gates.
- Consumes: a verified MaleCNS graph and deterministic saved RGB/input trace.

- [ ] **Step 1: Write failing metric tests with known synthetic events**

```python
def test_parity_metrics_have_fixed_gate_boundaries():
    cpu = {(0, 1), (2, 3), (4, 5)}
    metal = {(0, 1), (2, 3), (4, 6)}
    report = spike_metrics(cpu, metal, np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert report['jaccard'] == .5
    assert report['total_spike_fraction'] == 0
    assert report['within_one_tick_fraction'] == 1
```

Add unit cases for empty spike sets, zero-rate correlation, exact decoder
decisions, maximum weight relative error, and a gate value exactly on each
accepted threshold.

- [ ] **Step 2: Run the metric tests and confirm the module is absent**

Run: `python -m pytest tests/test_doom_metal_validation.py -q`

Expected: import failure for `doom_learning_v6.metal.validate`.

- [ ] **Step 3: Implement trace capture and gate calculations**

Add a diagnostic backend option that records compressed per-tick spike pairs for
validation intervals. Run CPU and Metal from the same checkpoint and input trace.
Calculate Jaccard, total-spike fraction, per-neuron rate correlation, one-tick
matching fraction, decoder equality, scientific gate equality, and maximum
plastic-weight relative error. For one-tick matching, group events by neuron,
sort each backend's ticks, and greedily pair each CPU tick with the earliest
unused Metal tick within `[-1, +1]`; divide matched pairs by the larger event
count. Write reports atomically with `allow_nan=False`.

- [ ] **Step 4: Record full-graph evidence on M4 Pro**

Run: `ssh mini 'cd doomfly && OPENBLAS_NUM_THREADS=1 python3 -m doom_learning_v6.metal.validate --out outputs/doom-learning/metal-validation-m4pro'`

Expected gates: Jaccard `>=0.995`, total-spike difference `<=0.005`, rate
correlation `>=0.999`, one-tick match `>=0.999`, exact decoder and scientific
gate outputs, and plastic-weight `rtol<=1e-4`.

- [ ] **Step 5: Commit code and compact evidence**

```bash
git add doom_learning_v6/metal/validate.py doom_learning_v6/metal/backend.py tests/test_doom_metal_validation.py outputs/doom-learning/metal-validation-m4pro
git commit -m "Validate full-graph Metal parity"
```

### Task 8: M4 Pro Benchmark and Training Gate

**Files:**
- Create: `doom_learning_v6/metal/benchmark.py`
- Create: `tests/test_doom_metal_benchmark.py`
- Modify: `doom_learning_v6/calibration.py`
- Modify: `doom_learning_v6/survival.py`
- Modify: `doom_learning/common.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `python -m doom_learning_v6.metal.benchmark --out PATH`, `calibrated_brain(eta=.001, backend='cpu')`, and survival CLI option `--backend {cpu,metal}`.
- Requires: passing validation JSON whose exact source, binary, graph, and configuration hashes match the requested Metal run.

- [ ] **Step 1: Write failing benchmark-gate and CLI tests**

```python
def test_training_rejects_missing_or_stale_metal_validation(tmp_path):
    with pytest.raises(ValueError, match='passing Metal validation'):
        require_metal_validation(tmp_path/'missing.json', EXPECTED_HASHES)

def test_benchmark_gate_requires_speed_and_memory():
    report = gate({'cpu_median': 4.0, 'metal_median': 1.9,
                   'peak_rss_gib': 7.0, 'critical_memory_pressure': False,
                   'sustained_swap_growth': False})
    assert report['speedup'] > 2
    assert report['passed'] is True
```

- [ ] **Step 2: Run tests and confirm the benchmark API is absent**

Run: `python -m pytest tests/test_doom_metal_benchmark.py -q`

Expected: import failure for `benchmark` and missing CLI backend selection.

- [ ] **Step 3: Implement repeatable benchmark reporting**

Warm each backend, run at least five matched repetitions, and report all samples,
median, minimum, maximum, kernel GPU time, host command time, synchronization,
plasticity time, graph-build time, peak RSS, memory pressure, and swap change.
Gate on median speedup `>=2`, RSS `<8 GiB`, no critical pressure, and no sustained
swap growth. Report real-time ratio separately as a target rather than silently
changing the acceptance result.

- [ ] **Step 4: Add explicit offline training selection**

Pass `backend` through `calibrated_brain`, `VisualMemoryBrain`, and `MemoryBrain`.
Add `--backend` and `--metal-validation` to `survival.py`. For Metal, load the
validation report before model construction and require matching hashes plus all
passed gates. Record the backend, device, build, validation report hash, neural
time, wall time, and their ratio in protocol and episode output.

- [ ] **Step 5: Expand provenance capture for Metal sources**

Change `capture_provenance` to recurse and include `.py`, `.cpp`, `.h`, `.mm`,
and `.metal` sources. Keep external data, compiled artifacts, machine-specific
configuration, and dependency checkouts excluded.

- [ ] **Step 6: Run CPU tests and M4 Pro benchmark**

Run locally: `python -m pytest tests/test_doom_backend_contract.py tests/test_doom_metal_graph.py tests/test_doom_metal_build.py tests/test_doom_metal_validation.py tests/test_doom_metal_benchmark.py tests/test_doom_learning_v6.py tests/test_doom_live_training.py -q`

Run on M4 Pro: `ssh mini 'cd doomfly && OPENBLAS_NUM_THREADS=1 python3 -m doom_learning_v6.metal.benchmark --out outputs/doom-learning/metal-benchmark-m4pro'`

Expected: CPU suite passes and the benchmark records an honest pass or preserved
failure without stopping another workload.

- [ ] **Step 7: Commit benchmark and gated training support**

```bash
git add .gitignore doom_learning/common.py doom_learning_v6/calibration.py doom_learning_v6/survival.py doom_learning_v6/metal/benchmark.py tests/test_doom_metal_benchmark.py outputs/doom-learning/metal-benchmark-m4pro
git commit -m "Gate offline Metal training on validation"
```

### Task 9: End-to-End Verification and Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/doom-metal-backend.md`
- Modify: `THIRD_PARTY.md` only if implementation adds a dependency beyond Apple system frameworks
- Modify: `docs/superpowers/plans/2026-09-11-metal-backend.md`

**Interfaces:**
- Produces: documented build, parity, benchmark, checkpoint migration, and gated offline-training commands.
- Excludes: live-server selection and public launch approval.

- [ ] **Step 1: Document the exact supported state**

Document prerequisites, local build commands, `cpu` default behavior, explicit
Metal selection, graph-cache provenance, checkpoint migration, validation gates,
benchmark interpretation, and the difference between wall-clock acceleration and
learning sample efficiency. State the measured M4 Pro result and every failed
gate verbatim from committed evidence.

- [ ] **Step 2: Run the complete local regression suite**

Run: `python -m pytest -q`

Expected: all applicable tests pass; Darwin-only tests report skips on Linux.

- [ ] **Step 3: Run the complete Metal suite on `mini`**

Run: `ssh mini 'cd doomfly && OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/test_doom_backend_contract.py tests/test_doom_metal_graph.py tests/test_doom_metal_build.py tests/test_doom_metal_state.py tests/test_doom_metal_parity.py tests/test_doom_metal_checkpoint.py tests/test_doom_metal_validation.py tests/test_doom_metal_benchmark.py tests/test_doom_learning_v6.py tests/test_doom_live_training.py -q'`

Expected: all selected tests pass on the M4 Pro.

- [ ] **Step 4: Run a gated offline smoke experiment**

Run: `ssh mini 'cd doomfly && OPENBLAS_NUM_THREADS=1 python3 -m doom_learning_v6.survival --backend metal --metal-validation outputs/doom-learning/metal-validation-m4pro/report.json --out outputs/doom-learning/metal-smoke --seeds 41031 --eval-seeds 61031 --seconds 1 --train-episodes 1'`

Expected: a complete plastic/frozen/shuffled smoke run with backend provenance,
all original edges retained, fixed controls, and no public-learning claim.

- [ ] **Step 5: Mark completed plan checkboxes and inspect the final diff**

Run: `git diff --check && git status --short && git log --oneline upstream/main..HEAD`

Expected: no whitespace errors, only planned source/tests/docs/compact evidence,
and no compiled artifacts, downloaded datasets, credentials, external research
workbooks, or game assets.

- [ ] **Step 6: Commit documentation and final verification record**

```bash
git add README.md docs/doom-metal-backend.md docs/superpowers/plans/2026-09-11-metal-backend.md
git commit -m "Document validated Metal experiment workflow"
```

- [ ] **Step 7: Push the feature branch**

Run: `git push origin metal-backend`

Expected: `origin/metal-backend` points to the final verified commit. Opening a
pull request remains a separate user-authorized action.
