# Metal Temporal Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement opt-in exact temporal windows and measure complete retained-connectome RGB training throughput.

**Architecture:** Keep reference scheduling explicit and immutable per owner. Prepare old delayed arrivals into per-time scratch, then run each neuron chronologically inside a window; keep host training boundaries unchanged.

**Tech Stack:** Objective-C++17, Metal 2.4, Python/NumPy, pytest, M4 Pro via the existing ignored SSH configuration.

**Spec:** `docs/superpowers/specs/2026-09-13-metal-temporal-windows-design.md`

## Global Constraints

- Retain all 166,700 MaleCNS v1.0 neurons, all 25,582,938 released retained connections, and all 4,184 existing KC→MBON11 plastic slots; preserve duplicate, self, weak and modulatory edges.
- Preserve epoch-5 zero-seeded ascending-incoming arithmetic, explicit FMA sites, decay tables, signed-zero behavior, lazy evolution and host-double centered plasticity operation order.
- Keep RGB sensory input, modeled propagation, reinforcement, plasticity and fixed neuron-to-button decoding separate; controls and game telemetry must not select actions.
- Use direct Objective-C++/Metal with macos-metal2.4, no fast math, macOS 13 minimum, and conservative Apple Silicon features; the M4 Pro is the initial validated target.
- Preserve failed controls and original numerical gates; bitwise agreement, changed weights, survival and engineering throughput do not establish learning or biological validity.
- Preserve user edits and unrelated workloads; credentials and machine-specific origins remain ignored; AGENTS.md and third-party notices remain unchanged.
- Use Sergey Subbotin <ssubbotin@gmail.com> for commits; avoid AI attribution and reverse-engineering wording in code, comments and commit messages.

---

### Task 1: Native window scheduler and defect-sensitive tests

**Ownership / Files:**

- Modify: `doom_learning_v6/metal/api.h`, `backend.mm`, `kernels.metal`, `build.py`.
- Create: `tests/metal_window_helpers.py`, `tests/test_doom_metal_windows_native.py`.
- Existing `tests/test_doom_metal_batch_native.py` helpers are consumed unchanged; original reference assertions and numerical goldens remain unchanged.
- Do not modify Python owner or brain behavior. You are not alone in the repository; preserve others' edits. No subagents.

**Interfaces:**

- Consumes existing Graph/State/KCEvent/Timing ctypes and native lane APIs.
- Produces `int df_metal_create_batch_windowed(const df_metal_graph *graph, const char *metallib_path, int32_t lanes, int32_t window_ticks, df_metal_handle *handle)`.
- Produces ABI 8 with unchanged public layouts, numerical parent/order and original create wrappers selecting reference 0.
- Test helper `run_fixture(tmp_path, *, window_ticks, cursor, steps, case, lanes=2)` creates independent reference-0 and candidate handles, initializes identical raw states, compares all STATE_NAMES through existing assert_state, canonical events and observations, and destroys both in finally. It returns candidate Timing and actual shared/mutable bytes. Both handles own separate complete arrays; no GPU result mocks.
- Test case strings: `prefilled-reset`, `future-reuse`, `shared-word`, `fast-presence`, `zero-modulation`, `lazy-evolution`, `gather-before-reset`, `duplicate-self`, `scratch-reuse`. Their exact initialization is below.

- [ ] **Step 1: Add raw fixtures and genuine interface/behavior RED tests.**

Use existing model_state/advance/check/assert_state, but raw fixtures must keep every array alive, use the documented dtype/shape for every State field, and start with valid queue lists, counts/flags and history. Build out CSR in stable source order and incoming CSR in stable target/source/original-edge order. Keep every duplicate/self edge. At cursor c initialize v/rest=-52, g=0, last=c-1, modulation=0, modulation_last=c, refractory=0, drive=previous_drive=0, counts/queue/active flags=0 and full zero 19-slot queues.

Actual parameterized test skeleton:

```python
@pytest.mark.parametrize("window_ticks", [1, 2, 18])
@pytest.mark.parametrize("cursor", range(19))
@pytest.mark.parametrize("steps", [1, 2, 17, 18, 19, 35, 36, 37, 99, 100])
def test_window_matches_reference(tmp_path, window_ticks, cursor, steps):
    timing, memory = run_fixture(tmp_path, window_ticks=window_ticks,
        cursor=cursor, steps=steps, case="duplicate-self")
    assert timing.encoder_count == 1
    assert timing.indirect_dispatch_count == 0
    assert timing.dispatch_count == 2 + 2 * ((steps + window_ticks - 1) // window_ticks)
    assert memory["shared"] > 0 and memory["mutable"] > 0
```

Other fixtures, all valid raw states:

- prefilled-reset: edgeless neuron0 inactive, v=-52,g=.5,last=c-7; put neuron0 in queue[(c+18)%19], queue_count=1. This catches a fired-only reset.
- future-reuse: neuron1 active,v=-40,refractory=2,last=c-1, zero drive, 2-tick advance; it fires at c+1. Its new future bit physically reuses slot c%19 and must survive.
- shared-word: two edges source0→target1 (.5), source0→target2 (.75), incoming positions0/1 in one word; queue[c%19] contains source0.
- fast-presence: sources0/1/2→target3 with weights (1e20,3,-1e20), g[target3]=-0.0, inactive; all sources in current queue. Add separate zero-weight fast rows with target g=-0.0 and g=7.0. Presence activates targets even if the sum is zero; the g=7 row also detects accidentally seeding the zero accumulator from existing g.
- zero-modulation: source0 modulatory, edge0→1 weight0, modulation[1]=.5,modulation_last[1]=c-7, queued source0; zero still decays modulation and updates its timestamp.
- lazy-evolution: edgeless inactive neuron0 v=-51,g=.3,adaptation=.7,last=c-7; no queue events. Final settlement must match reference without eager per-tick rounding.
- gather-before-reset: queued source0→target1 weight.5, target1 active,v=-40,last=c-1; compare gathering followed by future membership reset.
- duplicate-self: graph edges0→0 twice,0→1 twice,1→0,2→1(modulatory source2), include zero and mixed signs; distinct lane drives and plastic edits between advances. Include zero-edge case separately.
- scratch-reuse: reuse a handle through lengths (18,1,17,19,100), then valid lane reuploads and repeat; compare with independently reuploaded reference, including unconsumed future events.

Run:
`python -m pytest -q tests/test_doom_metal_windows_native.py` on the actual M4.
Before native implementation the completed healthy build must fail the new symbol assertion. Record this as interface RED, not proof that semantic fixtures detected defects. Linux skips are not Metal test passes.

- [ ] **Step 2: Implement and commit atomic prelude independently.**

Use exact empty guards, retaining all populated arithmetic:

```metal
if (atomic_load_explicit(&touched[target], memory_order_relaxed) != 0 &&
    atomic_exchange_explicit(&touched[target], 0u, memory_order_relaxed) != 0) {
  // Existing ordered gathering body, including presence flags.
}
uint bits = 0u;
if ((atomic_load_explicit(&active_edge_bits[word], memory_order_relaxed) & mask) != 0)
  bits = atomic_fetch_and_explicit(&active_edge_bits[word], ~mask,
      memory_order_relaxed) & mask;
```

Run existing native/state/arithmetic/decay suites on M4 and save full outputs.
This correctness-neutral optimization uses baseline characterization plus exact comparisons; there is no claimed semantic RED from omitting a guard.
Commit guard-only code separately, record its exact SHA in the report, and notify the controller. Continue windows without waiting for timing results.

- [ ] **Step 3: Add immutable factory, bounded allocations and kernel parameters.**

```cpp
extern "C" int df_metal_create_batch_windowed(
    const df_metal_graph *graph, const char *path, int32_t lanes,
    int32_t window_ticks, df_metal_handle *handle) {
  if (window_ticks < 0 || window_ticks > 18)
    return fail("Metal window ticks must be in 0..18");
  // Existing checked construction body with schedule stored before allocation.
}
static_assert(sizeof(KernelParams) == 96);
static_assert(offsetof(KernelParams, window_ticks) == 88);
static_assert(offsetof(KernelParams, window_capacity) == 92);
```

Original create_batch delegates with 0. Append uint window_ticks/window_capacity identically in MSL. Set API/build ABI8. Extend Backend with configured window capacity, two new pipelines and optional two scratch buffers. Keep ordinary buffers for reference mode. Calculate W*lanes*neurons*sizeof(uint32_t) and W*lanes*edge_words*sizeof(uint32_t) with checked products before allocation; include actual allocations in resident accounting and sparse-replacement peak. Zero all new buffers at creation and each lane's entire planes on successful upload/recovery. Invalid window, huge lane count, products or allocation must return safely with no host/state mutations or leaks.

- [ ] **Step 4: Implement preparation and chronological workers.**

Preparation's x grid is words*actual W; y is lane. Its core:

```metal
uint tick_offset = position.x / p.words;
uint word = position.x % p.words;
uint slot = uint((p.clock + tick_offset) % 19);
uint members = atomic_exchange_explicit(&ring[slot*p.words + word], 0u,
    memory_order_relaxed);
while (members != 0u) {
  uint neuron = (word << 5) + ctz(members);
  if (neuron < p.neurons) {
    for (long edge=out_ptr[neuron]; edge<out_ptr[neuron+1]; ++edge) {
      atomic_store_explicit(&time_touched[out_post[edge]], 1u, memory_order_relaxed);
      uint incoming = uint(edge_to_incoming[edge]);
      atomic_fetch_or_explicit(&time_bits[incoming>>5], 1u<<(incoming&31),
          memory_order_relaxed);
    }
  }
  members &= members-1u;
}
```

Offset ring by lane*ring_stride; time_touched by (lane*window_capacity+tick_offset)*neurons; time_bits by corresponding edge_words stride. Each physical consumed word is uniquely owned. No preparation float summation.

Worker offsets all mutable neuron/weight/ring buffers by lane and loops:

```metal
for (uint offset=0; offset<p.window_ticks; ++offset) {
  long now = p.clock + offset;
  uint future = uint((now + 18) % 19);
  // Original integrate active/evolve/fire/count/event/adaptation/can_fire order.
  // Original gather with time-specific touched and masked ascending bits.
  if (ring_contains(ring, p.words, future, target)) {
    v[target]=rest[target]; g[target]=0.0f;
    refractory[target]=(short)p.refractory_ticks;
  }
}
```

Implement the two commented bodies from the existing kernels without changing expressions/order; extracting a narrowly scoped common inline integrate/gather helper is permitted if reference remains exact. Workers never clear current ring words. Future OR and reads remain atomic. No eager evolution or intermediate materialization. All-neuron coverage stays complete.

Native advance keeps its validations/counts clear/drive/final settlement; schedule windows:

```cpp
for (int32_t done=0; done<steps; ) {
  uint32_t width=uint32_t(std::min(b->window_ticks, steps-done));
  p.clock=cursor+done; p.window_ticks=width;
  p.window_capacity=uint32_t(b->window_ticks);
  encode(encoder,b->window_prepare_pipeline,prepare_buffers,p,
      uint32_t(b->words)*width,dispatch_count);
  encode(encoder,b->window_neuron_pipeline,neuron_buffers,p,
      uint32_t(b->neurons),dispatch_count);
  done+=int32_t(width);
}
```

Reference0 keeps original loop. Keep one ordinary tracked encoder and actual grid timing counters as specified. Validate uint32 counter products. Read events/cursor once per bin; failures poison exactly as before.

- [ ] **Step 5: Prove fixtures and run covering GREEN suites.**

Run parameter matrix plus lane-specific drive/weights/permutation/repetition/neighbor perturbation, diagnostics capacity/poison/recovery, cursor overflow/mismatch, invalid mode (negative/19/INT32_MAX/null graph), edgeless and memory accounting tests. Inspect scratch after completed commands through an ignored test-only bridge if needed; no new production introspection ABI.

On actual M4 create separate ignored defective builds from the production source and run the matching comparison for each: suppress live prefilled reset; clear future slot after workers; replace masked incoming consumption with whole-word clear; seed conductance with existing g; suppress zero-weight fast/modulatory presence; eagerly evolve inactive cells. Each must fail its named test. Record ineffective attempts and strengthen fixtures, never call them sensitivity successes. Keep mutation code outside tracked production and restore by building from tracked sources, not destructive Git commands.

Run portable full suite once and actual M4 native/owner/full suite once before final native commit. Full outputs, command/env (sanitized), source/build hashes and source commit must be in report. Controller serializes all M4 jobs, supplies an owned target checkout and runs the target commands if worker lacks its interpreter/resources.

Commit scoped files:
`git add doom_learning_v6/metal/api.h doom_learning_v6/metal/backend.mm doom_learning_v6/metal/kernels.metal doom_learning_v6/metal/build.py tests/metal_window_helpers.py tests/test_doom_metal_windows_native.py`;
`git commit -m "feat: add bounded Metal temporal windows"`.

### Task 2: Opt-in Python ownership and RGB integration

**Ownership / Files:**

- Modify: `doom_learning_v6/metal/batch.py`.
- Create: `tests/test_doom_metal_windows.py`.
- Modify: `tests/test_doom_metal_batch_rgb.py` only to parameterize optional schedule without weakening comparisons.
- Preserve `doom_learning_v6/brain.py` reset-before-mutation guard and every original owner validation/recovery path. You are not alone in the repository; preserve others' edits. No subagents.

**Interfaces:**

- Consumes Task1 ABI8 and exact windowed factory signature.
- Produces `MetalBatchExecutor(brains, *, window_ticks=0)`, read-only `window_ticks` property, metadata scheduler=`reference-two-dispatch` for0 or `temporal-window` for1..18.
- Existing `advance`, `rgb_step`, checkpoints, fixed controls and host learning APIs are unchanged.

- [ ] **Step 1: Add portable rejection RED and actual owner identity tests.**

```python
@pytest.mark.parametrize("bad", [True, False, np.bool_(True), -1, 19, 2**63, 1.0, None, "18"])
def test_bad_window_rejected_before_brains_are_touched(bad):
    class ReadTrap:
        def __iter__(self):
            raise AssertionError("Brains accessed before window validation")
    with pytest.raises((TypeError, ValueError), match="Window ticks must be"):
        MetalBatchExecutor(ReadTrap(), window_ticks=bad)
```

Run focused portable test before implementation; genuine RED must reflect the missing keyword/validation contract. Add tests accepting np.int64(0/1/18) at a safely denied native boundary and asserting exact chosen integer property. Test mutation of property is rejected. On M4 parameterize schedule0/1/2/18 against independent reference owners for ordinary/keep-memory resets, checkpoint restore, closed/poison CPU recovery, frozen state, eligibility, sparse weights, signed event lanes and duplicate brain rejection. Reuse existing assert_equal/full snapshots.

- [ ] **Step 2: Implement early validation and explicit factory selection.**

```python
def __init__(self, brains, *, window_ticks=0):
    if isinstance(window_ticks, (bool, np.bool_)) or not isinstance(window_ticks, (int, np.integer)):
        raise TypeError("Window ticks must be an integer in 0..18")
    if not 0 <= int(window_ticks) <= 18:
        raise ValueError("Window ticks must be in 0..18")
    self._window_ticks = int(window_ticks)
    # Existing initialization/ownership contract unchanged.

@property
def window_ticks(self):
    return self._window_ticks
```

Configure `df_metal_create_batch_windowed` ctypes signature Graph*,char*,int32,int32,Cvoid** and select it once at creation with self._window_ticks. No mutable setter. Metadata adds chosen scheduler/window_ticks; actual shared/mutable queried bytes include scratch. Store chosen value in immutable owner configuration and validate it with bindings if internal reassignment could desynchronize native state. Preserve all close/failure/freezing/restoration behavior.

- [ ] **Step 3: Parameterize live RGB tests with no timing-boundary changes.**

Modify `run_rgb(tmp_path, order=(0,1), perturb=False, window_ticks=0)` to pass the selected keyword to the owner. Add a macOS-parametrized test0/1/2/18 comparing baseline/repeat/permutation/neighbor perturbation/N1 exactly through existing helper. Keep encoder/bin/transfer/sparse-byte assertions, live pixel sampling, fixed controls and learning separate. Add invalid-neighbor/reset binding tests with window mode to demonstrate no changes before rejection.

Run:
`python -m pytest -q tests/test_doom_metal_windows.py tests/test_doom_metal_batch.py tests/test_doom_metal_batch_rgb.py tests/test_doom_metal_checkpoint.py`.
Run full portable and actual M4 suites once before commit; save all completed outputs/hashes.
Commit:
`git add doom_learning_v6/metal/batch.py tests/test_doom_metal_windows.py tests/test_doom_metal_batch_rgb.py`;
`git commit -m "feat: expose opt-in Metal window ownership"`.

### Task 3: Real retained validation, timing and portable evidence

**Ownership / Files:**

- Controller owns ignored runtime job scripts/manifests and target commands; serialized GPU work only.
- Create: `docs/experiments/2026-09-13-metal-temporal-windows.md` and bounded evidence JSON below `outputs/doom-learning/metal-temporal-windows-m4pro/`.
- Modify: `docs/doom-metal-backend.md` with exact implemented mode/evidence/status.
- No production code edits in controller; findings return to original task worker. Preserve unrelated work and all original evidence.

**Interfaces:**

- Consumes frozen Task1/2 source commits, guard-only commit, default reference0 and candidate owner API.
- Produces source/build/input-pinned results, portable raw artifact inventory (bytes/SHA256), precise throughput/memory/controls reports and explicit unresolved learning status.
- Local published summaries never expose target origins, SSH aliases, credentials, commercial assets or large external inputs.

- [ ] **Step 1: Save acceptance protocol before running jobs.**

Use fresh owned target checkouts/bundle transfer with checked hashes; retain original target/user edits. Pin exact graph and APPO pair hashes, compiler flags, source/build identities and clock. Query available memory/workloads safely without credential/process command dumps. Allocate per-mode handles one experiment at a time, release between modes, and separate initialization/dark warmup from measurement.

Use measured workload equations in runtime/evidence:
```python
brain_seconds = sum(lane["rgb_brain_seconds"] for lane in lanes)
rate = brain_seconds / measured_wall_seconds
gain = candidate_median_rate / reference_median_rate
assert all(lane["frames"] == expected_frames for lane in lanes)
assert all(lane["cursor"] == expected_cursor for lane in lanes)
```

N4 duplicates are engineering throughput lanes, not independent learning replicates. Candidate memory adds scratch; record actual MTLBuffer lengths, peak RSS, pressure/swap and replacement growth. Never infer active edge counts from dense grid counts or hardware latency from aggregate GPU timing.

- [ ] **Step 2: Run guard-only then W1/W2/W18 bounded matched diagnostics.**

Compare guard-only0 with parent0 at identical bins/drives, then newreference0 with each window mode. No arbitrary CPU multiplier gate. Select the fastest exact candidate; if none improves, retain default0 and report negative evidence while investigating measured work. Optional actual supported hardware-counter query is bounded and cannot delay implementation.

- [ ] **Step 3: Run full retained correctness before complete timing.**

Repeat unchanged40/80ms CPU gates; preserve all known long-CPU and epoch6 failures.
Six N2 and six N4 complete own-reference/permutation/repeat/neighbor trials must match all traces/scalars/weights/checkpoints and canonical events. Run all eight original complete RGB controls with original numeric checkpoints. Compare current reference against original pins honestly with CPU binary provenance; source/compiler equality does not imply binary identity.

Then run three fresh matched complete reference/candidate N4 pairs: each lane processes982 train and988 held frames at original35Hz/actionrepeat4 timing. Alternating pair order controls drift; no overlapping GPU jobs, discarded tails or padded finished lanes. Match all trajectories and checkpoint bits before counting throughput. Full eight controls establish reproduction, not learning. Preserve unsuccessful/interrupted diagnostics separately.

- [ ] **Step 4: Audit and publish exact evidence.**

Copy complete local raw artifacts and audit bytes/hashes versus target manifests. Publish only bounded portable JSON with relative evidence roles, source/build/input identities, actual rates/dispersion, canonical match counts, memory and known failures. Audit credential/private-origin/symlink/oversized-file risks and all local doc links. Update backend docs with source-specific figures, schedule/counter meanings, unchanged default0, explicit learning unresolved and no launch green light.

Run fresh full portable/M4 suites, `git diff --check`, source/build pin and document-link checks. Commit scoped evidence/docs with Sergey identity. Task reviewer gets exact protocol/report/evidence diff; final strongest whole-branch review follows, with one scoped fix wave if required. Normal push to the owned `metal-temporal-windows` branch is authorized; verify actual remote SHA after push. Preserve this plan's ignored ledger and failed artifacts for the ongoing native goal.
