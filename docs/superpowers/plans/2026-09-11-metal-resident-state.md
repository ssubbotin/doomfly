# Metal Resident State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep mutable neural state resident in Metal between 10 ms learning bins, transfer only required boundary data, and collect measurements that select the next kernel optimization.

**Architecture:** Metal owns mutable propagation state between bins. Python supplies each bin's drive, receives counts, cursor, and KC events, runs the unchanged float64 plasticity rule, then mirrors all 4,184 plastic weights to Metal before the next bin. Full state upload and materialization occur only at explicit lifecycle boundaries.

**Tech Stack:** Python 3.11, NumPy, pytest, C++17, Objective-C++, Metal 2.4, `ctypes`, Apple M4 Pro.

**Spec:** `docs/superpowers/specs/2026-09-11-metal-backend-design.md`

## Global Constraints

- Retain all 166,700 neurons and all 25,582,938 MaleCNS v1.0 directed connections.
- Keep the existing sparse incoming-edge bitmap and current kernels unchanged in this slice.
- Preserve the CPU kernel as the scientific reference.
- Preserve host float64 eligibility and centered plasticity, fixed decoding, diagnostic capture, and backend-independent checkpoints.
- Use conservative `macos-metal2.4` features supported by Apple GPU family 7.
- Preserve the failing 80 ms CPU/Metal stress control. The optimized Metal backend must reproduce the retained Metal trace exactly.
- Generated binaries, caches, checkpoints, and temporary measurement artifacts remain untracked.
- Write each behavioral test first, observe its expected failure, then add the minimum implementation.

---

### Task 1: Separate Hot-Path Timing and Transfer Measurements

**Files:**
- Modify: `doom_learning_v6/metal/api.h`
- Modify: `doom_learning_v6/metal/backend.mm`
- Modify: `doom_learning_v6/metal/backend.py`
- Modify: `doom_learning_v6/metal/benchmark.py`
- Modify: `tests/test_doom_metal_benchmark.py`
- Test: `tests/test_doom_metal_state.py`

**Interfaces:**
- Extends: `df_metal_timing` with native phase durations and byte counts.
- Produces: `MetalBackend.last_timing: dict[str, int | float]` containing raw per-bin measurements.
- Preserves: current full synchronization and numerical behavior until Task 3.

- [x] **Step 1: Add a failing aggregation test**

Extend `test_sample_aggregates_metal_dispatch_work` with this timing payload and assertions:

```python
self.backend.last_timing = {
    'gpu_seconds': .1,
    'native_total_seconds': .2,
    'encode_seconds': .01,
    'commit_call_seconds': .002,
    'wait_call_seconds': .188,
    'full_upload_seconds': .03,
    'full_upload_bytes': 100,
    'materialize_seconds': .04,
    'materialize_bytes': 200,
    'drive_copy_seconds': .001,
    'drive_copy_bytes': 12,
    'counts_clear_seconds': .003,
    'counts_copy_seconds': .004,
    'counts_copy_bytes': 16,
    'native_event_copy_seconds': .005,
    'native_event_copy_bytes': 32,
    'event_conversion_sort_seconds': .006,
    'eligibility_seconds': .007,
    'sparse_weight_update_seconds': .008,
    'sparse_weight_update_bytes': 48,
    'encoder_count': 1,
    'dispatch_count': 502,
    'mark_grid_threads': 7,
    'gather_grid_threads': 9,
    'indirect_dispatch_count': 100,
    'edge_bitmap_words': 11,
}
assert result['full_upload_bytes'] == 400
assert result['counts_copy_bytes'] == 64
assert result['event_conversion_sort_seconds'] == pytest.approx(.024)
```

- [x] **Step 2: Run the focused test and confirm the missing fields fail**

Run: `.venv-neural/bin/python -m pytest tests/test_doom_metal_benchmark.py::test_sample_aggregates_metal_dispatch_work -q`

Expected: FAIL because `_sample` does not aggregate the new fields.

- [x] **Step 3: Extend the timing ABI and Python mapping**

Add native fields for `encode_seconds`, `commit_call_seconds`, `wait_call_seconds`, `counts_clear_seconds`, `native_event_copy_seconds`, and `native_event_copy_bytes`. Measure around the named operation only. Keep `gpu_seconds` from `GPUStartTime` and `GPUEndTime`.

Measure full upload, materialization, drive and counts copies, event conversion, eligibility, and sparse weight update around their Python or C ABI calls. Record explicit byte counts. Do not derive encoding time by subtracting GPU time from a wait-containing duration.

- [x] **Step 4: Aggregate raw timing fields**

Update `_sample` to sum durations and bytes across all four frames. Preserve each repetition in the benchmark report before computing minimum, median, and maximum summaries.

- [x] **Step 5: Run focused tests**

Run: `.venv-neural/bin/python -m pytest tests/test_doom_metal_benchmark.py tests/test_doom_metal_state.py -q`

Expected on Linux: benchmark tests pass and Darwin tests skip.

- [x] **Step 6: Commit instrumentation**

```bash
git add doom_learning_v6/metal/api.h doom_learning_v6/metal/backend.mm \
  doom_learning_v6/metal/backend.py doom_learning_v6/metal/benchmark.py \
  tests/test_doom_metal_benchmark.py tests/test_doom_metal_state.py
git commit -m "Measure Metal state transfer phases"
```

### Task 2: Define Narrow Boundary Transfers

**Files:**
- Modify: `doom_learning_v6/metal/api.h`
- Modify: `doom_learning_v6/metal/backend.mm`
- Modify: `doom_learning_v6/metal/backend.py`
- Modify: `tests/test_doom_metal_state.py`

**Interfaces:**
- Produces: `df_metal_upload_drive(handle, drive)`.
- Produces: `df_metal_download_observation(handle, counts, cursor)`.
- Preserves: `df_metal_upload_state` and `df_metal_download_state` for lifecycle boundaries.

- [ ] **Step 1: Write failing native boundary tests**

Add Darwin tests that upload a changed drive without altering any other state, advance one bin, and read counts and cursor without a full download. Add null-pointer tests with exact error messages.

```python
assert library.df_metal_upload_drive(handle, drive.ctypes.data_as(C.c_void_p)) == 0
assert library.df_metal_download_observation(
    handle, counts.ctypes.data_as(C.c_void_p), C.byref(cursor)) == 0
assert cursor.value == 100
```

- [ ] **Step 2: Run on M4 Pro and verify the symbols are missing**

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_state.py -q'`

Expected: FAIL because the narrow C ABI functions do not exist.

- [ ] **Step 3: Add the C ABI**

```c
int df_metal_upload_drive(df_metal_handle handle, const float *drive);
int df_metal_download_observation(df_metal_handle handle,
    int32_t *counts, int64_t *cursor);
```

Copy exactly `neurons * sizeof(float)` bytes for drive and `neurons * sizeof(int32_t)` bytes for counts. Read the backend cursor after successful command completion. Validate every pointer.

- [ ] **Step 4: Clear device counts safely**

Clear `b->counts.contents` after the previous command has completed and before encoding the next command. Keep this operation independent of `df_drive_change`, whose unchanged-drive threads return early.

- [ ] **Step 5: Bind and test the narrow calls**

Define exact `ctypes` signatures and private Python methods `_upload_drive()` and `_download_observation()`. Run the complete Darwin state and parity micrograph tests.

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_state.py tests/test_doom_metal_parity.py -q'`

Expected: PASS with current full synchronization still active.

- [ ] **Step 6: Commit the narrow ABI**

```bash
git add doom_learning_v6/metal/api.h doom_learning_v6/metal/backend.mm \
  doom_learning_v6/metal/backend.py tests/test_doom_metal_state.py
git commit -m "Add narrow Metal boundary transfers"
```

### Task 3: Make Neural State Device-Resident

**Files:**
- Modify: `doom_learning_v6/backend.py`
- Modify: `doom_learning_v6/brain.py`
- Modify: `doom_learning_v6/metal/backend.py`
- Modify: `tests/test_doom_metal_state.py`
- Modify: `tests/test_doom_metal_parity.py`

**Interfaces:**
- Produces: `MetalBackend.materialize(reason: str) -> None`.
- Produces: host validity and device/weight epoch checks.
- Changes: `restore_from_host(reason: str)` requires an explicit authoritative lifecycle reason for Metal.

- [ ] **Step 1: Write failing residency and stale-host tests**

Run two frozen-weight bins through a retained full-sync reference and a resident backend, then compare counts, events, cursor, and fully materialized state. Between resident bins, poison host `v`, `g`, `queue`, and `active_flag`; assert that the next Metal result is unchanged. Task 4 adds the learning-enabled comparison after sparse weight mirroring exists.

```python
resident.weights_frozen = True
resident.step([], 10, stimulation=([0], 20), lamina_bias=0)
resident.v.fill(np.nan)
resident.g.fill(np.nan)
resident.queue.fill(-1)
resident.active_flag.fill(255)
actual, _ = resident.step([], 10, stimulation=([3], 20), lamina_bias=0)
resident.backend.materialize('test')
np.testing.assert_array_equal(actual, expected)
```

Add a test proving a generic full upload rejects stale host neural state.

- [ ] **Step 2: Run on M4 Pro and confirm the current full upload consumes poisoned arrays**

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_state.py -q'`

Expected: FAIL because `advance` calls `restore_from_host()`.

- [ ] **Step 3: Implement explicit ownership**

After initialization or full upload, host and device are synchronized. After every Metal advance, mark host neural arrays stale while keeping copied counts, cursor, KC events, eligibility, and plastic host weights valid.

`advance()` must:

1. upload drive;
2. call `df_metal_advance`;
3. apply KC events to host float64 eligibility;
4. download counts and cursor;
5. convert and sort required events;
6. leave full neural arrays stale.

Remove hot-path `restore_from_host()` and `sync_for_checkpoint()`.

- [ ] **Step 4: Protect lifecycle boundaries**

`materialize(reason)` waits for completion, downloads full state, and marks host neural arrays valid. `checkpoint`, validation state digests, diagnostic shutdown, and explicit inspection must materialize. Reset and restore overwrite every canonical host array before a reason-tagged full upload. Reject a full upload while host neural arrays are stale unless the reason is `reset` or `restore`.

- [ ] **Step 5: Run residency and parity tests**

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_state.py tests/test_doom_metal_checkpoint.py tests/test_doom_metal_parity.py -q'`

Expected: PASS, including poisoned-host isolation and all continuation directions.

- [ ] **Step 6: Commit resident ownership**

```bash
git add doom_learning_v6/backend.py doom_learning_v6/brain.py \
  doom_learning_v6/metal/backend.py tests/test_doom_metal_state.py \
  tests/test_doom_metal_parity.py
git commit -m "Keep Metal neural state resident"
```

### Task 4: Mirror Host Plastic Weights Before the Next Bin

**Files:**
- Modify: `doom_learning_v6/backend.py`
- Modify: `doom_learning_v6/brain.py`
- Modify: `doom_learning_v6/metal/backend.py`
- Modify: `tests/test_doom_metal_checkpoint.py`
- Modify: `tests/test_doom_metal_state.py`

**Interfaces:**
- Produces: backend method `update_weights(edge_ids, values) -> None`.
- Guarantees: the Metal propagation weight epoch equals the host plastic-weight epoch before every advance.

- [ ] **Step 1: Write a failing multi-bin learning test**

Use a micrograph whose first bin changes its identified plastic edge. Assert that the second bin observes the changed efficacy on Metal. Materialize afterward and prove every unidentified weight is bitwise unchanged.

```python
before = model.weight.copy()
model.step([], 10, stimulation=([0, 3], 20), lamina_bias=0, learning=True)
changed = model.weight[model.circuit['edges']].copy()
assert not np.array_equal(changed, before[model.circuit['edges']])
model.step([], 10, stimulation=([0], 20), lamina_bias=0, learning=False)
model.backend.materialize('test')
unidentified = np.setdiff1d(
    np.arange(len(model.weight), dtype=np.int64),
    model.circuit['edges'])
np.testing.assert_array_equal(
    model.weight[unidentified], before[unidentified])
```

- [ ] **Step 2: Run on M4 Pro and confirm resident Metal misses the update**

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_state.py -q'`

Expected: FAIL because the existing `df_metal_update_weights` call is not connected to `MemoryBrain.step`.

- [ ] **Step 3: Connect the backend contract**

Add a no-op `CpuBackend.update_weights`. After each host rule update that writes `self.weight[self.circuit['edges']]`, call:

```python
self.backend.update_weights(
    self.circuit['edges'],
    self.weight[self.circuit['edges']])
```

The Metal method sends all 4,184 canonical edge IDs and contiguous float32 values, including after the final bin. Increment host and device weight epochs together and assert equality before the next advance.

- [ ] **Step 4: Run learning and checkpoint tests**

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_state.py tests/test_doom_metal_checkpoint.py tests/test_doom_learning_v6.py -q'`

Expected: PASS with exact float64 eligibility and unchanged nonplastic weights.

- [ ] **Step 5: Commit weight mirroring**

```bash
git add doom_learning_v6/backend.py doom_learning_v6/brain.py \
  doom_learning_v6/metal/backend.py tests/test_doom_metal_checkpoint.py \
  tests/test_doom_metal_state.py
git commit -m "Mirror plastic weights to resident Metal state"
```

### Task 5: Validate Lifecycle Continuity and Retained Digests

**Files:**
- Modify: `doom_learning_v6/metal/validate.py`
- Modify: `tests/test_doom_metal_checkpoint.py`
- Modify: `tests/test_doom_metal_validation.py`

**Interfaces:**
- Produces: explicit materialization before every validation digest.
- Preserves: existing validation thresholds and the retained 40 ms and 80 ms controls.

- [ ] **Step 1: Write failing explicit-materialization tests**

Add a validation helper test that poisons host mirrors after Metal advancement and proves the digest helper materializes device state before hashing. Add reset and restore tests after a stale resident interval.

- [ ] **Step 2: Verify the tests fail for stale host state**

Run: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest tests/test_doom_metal_checkpoint.py tests/test_doom_metal_validation.py -q'`

Expected: FAIL until validation requests materialization.

- [ ] **Step 3: Materialize at every observer boundary**

Update state-digest, checkpoint, diagnostic-finalization, and continuation helpers to request full materialization with a recorded reason. Keep ordinary count and decoder observation on the narrow path.

- [ ] **Step 4: Run the complete local and M4 Pro suites**

Run locally: `.venv-neural/bin/python -m pytest -q`

Run on M4 Pro: `ssh mini 'cd doomfly && .venv-neural/bin/python -m pytest -q'`

Expected: all tests pass; platform-specific skips remain explained.

- [ ] **Step 5: Commit lifecycle validation**

```bash
git add doom_learning_v6/metal/validate.py \
  tests/test_doom_metal_checkpoint.py tests/test_doom_metal_validation.py
git commit -m "Validate resident Metal lifecycle boundaries"
```

### Task 6: Produce M4 Pro Evidence and Select the Next Kernel Slice

**Files:**
- Modify: `docs/doom-metal-backend.md`
- Modify: `outputs/doom-learning/metal-optimization-study-m4pro.json`
- Create: fresh validation and benchmark report directories under `outputs/doom-learning/`

**Interfaces:**
- Produces: a source-hashed five-repetition residency benchmark and updated optimization decision.
- Selects: two-dispatch fusion or an ID-ordered block-sparse prototype from measured evidence.

- [ ] **Step 1: Push the implementation commits and update M4 Pro**

```bash
git push origin metal-backend
ssh mini 'cd doomfly && git pull --ff-only origin metal-backend'
```

- [ ] **Step 2: Run fresh validation**

```bash
ssh mini 'cd doomfly && OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python \
  -m doom_learning_v6.metal.validate \
  --out outputs/doom-learning/metal-resident-validation-m4pro'
```

Require the existing 40 ms gates and exact retained Metal digests. Rerun the preserved 80 ms stress control and require exact reproduction of its Metal-side state, counts, events, and bin digests.

The retained 40 ms values are state
`59a2739ed22f68ba13811fe4d241e8a2a0a7e8f88a90ef94576385e69d0272e6`,
counts
`89ec6db8cc45677533caf82db7f288bb68285525adf799f7e2204339a1363605`,
and 21,326 events. The retained 80 ms Metal values are state
`8fec98ef0d7636b7fd04e7a80a0349c658ef4b282adbbee2fd93452a6334a80e`,
counts
`f9c25bbd7ca544b10d08461cb91d5542d838ed8eb7ac0624034d1df961a2ca72`,
and 52,236 events. Compare every per-bin `counts_sha256` value from the
corresponding retained report.

- [ ] **Step 3: Run a fresh five-repetition benchmark**

```bash
ssh mini 'cd doomfly && OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python \
  -m doom_learning_v6.metal.benchmark \
  --validation outputs/doom-learning/metal-resident-validation-m4pro/report.json \
  --out outputs/doom-learning/metal-resident-benchmark-m4pro'
```

Promotion requires:

- no steady-state full upload or materialization;
- at most 2 MiB transferred per 10 ms bin;
- Metal neural median at most 76.00 ms per 40 ms;
- total sample wall median at most 86.13 ms;
- GPU median at most 73.25 ms;
- every reset, checkpoint, diagnostic, continuation, parity, and determinism gate passing.

- [ ] **Step 4: Record the decision**

If any correctness gate fails, roll back resident ownership and retain correct instrumentation plus the failed reports. If neural improvement is below 5 ms or wall improvement is below 8 ms, retain instrumentation and restore the full-sync hot path.

When removable clear, finalize, and full-grid boundary cost measures at least 14.22 ms per 40 ms, write the next plan for `df_integrate_mark` plus `df_gather_finalize`. Preserve a dispatch boundary between them and require 808 total dispatches and GPU median at most 56.90 ms.

When that cost is lower and the active-tick 95th-percentile occupied 128/256-neuron-block fraction is below 40%, write the next plan for an ascending-ID block-sparse prototype with a dense fallback.

- [ ] **Step 5: Update documentation and commit evidence**

Document measured timings, transfer bytes, retained digests, failures, and the chosen second slice. Do not describe the 2x CPU gate as passed unless Metal neural median is at most 17.27 ms.

```bash
git add docs/doom-metal-backend.md \
  outputs/doom-learning/metal-optimization-study-m4pro.json \
  outputs/doom-learning/metal-resident-validation-m4pro \
  outputs/doom-learning/metal-resident-benchmark-m4pro
git commit -m "Record resident Metal validation evidence"
git push origin metal-backend
```
