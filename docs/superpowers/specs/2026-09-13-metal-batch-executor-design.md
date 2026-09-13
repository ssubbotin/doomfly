# Resident Metal Multi-Trajectory Executor Design

## Decision and scope

Implement two lockstep resident trajectories on the audited epoch-5 numerical
parent `b24249ecfee12aa84b55ed803ffe47bd09c487ad`. Extend to four only after
all two-lane identity/isolation checks pass and working-set headroom permits.
This architectural change amortizes common dispatch work while retaining
independent trajectories. Throughput improvement is a hypothesis to measure.

The user grants standing approval for routine in-scope design, implementation,
tests, reviews and owned-branch commit/push. Save and execute this design
without additional approval pauses. This is an engineering experiment and
does not authorize a public launch or assert useful fly learning.

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

## Alternatives and evidence

1. Recommended: batch epoch 5. Its completed eight-phase RGB cohort provides an
   audited reference; exact two-second dark spikes and existing short gates
   make the new change separately attributable.
2. Batch epoch 6. Its seeded carry fixes seven local cases, while dark parity
   diverges within 1.47-1.48 seconds and the first RGB turn reverses. Keep it
   reproducible on `metal-incoming-carry`; exclude its arithmetic from this step.
3. Reconstruct CPU ordered queues and per-edge modulation first. This could
   improve CPU fidelity, with unmeasured implementation/performance cost and
   no demonstrated learning benefit. Targeted causal diagnostics follow
   batching; long CPU parity remains necessary for a long-equivalence claim.

A four-member, three-stage council independently answered, anonymously
cross-critiqued and synthesized this choice. All four reviews favored the
smallest exact lane-isolation ladder. The full synthesis is recorded in
`docs/experiments/2026-09-13-metal-batch-council.md`.

## Native ownership and scheduling

Extend the existing direct Objective-C++ owner rather than create a second
kernel implementation. N=1 retains the current Python backend contract.
Add checked batch creation and lane-specific state/drive/observation/weight/
eligibility operations. Advance one shared handle with all lanes in one
grid per existing integrate/mark and gather/finalize stage.

The native owner allocates immutable outgoing/incoming adjacency, inverse
incoming map, KC/modulation masks, resting configuration and decay tables
once. It owns separate full mutable weight storage, v/g, adaptation,
refractory/last timestamps, drives, active flags, delayed rings, touched
targets and active-edge bitmaps per lane. Full mutable weights are deliberately
retained initially; no compact weight overlay or altered summation is bundled.

Use lane-major storage and checked offsets. A thread's logical neuron index
stays lane-local; all adjacency/source/target IDs remain unchanged. Only
mutable-buffer offsets depend on lane. Event records keep their current
16-byte layout, with the reserved field naming the lane for batch output;
serial events keep reserved=0. Sort/filter events independently within each
lane and retain the current native float32 exponential/host float64 eligibility
association. No event queue order is introduced into neural propagation.

All lanes advance the same integer steps (1-100) and require equal cursors
before dispatch. Lane-specific uploads may stage differing checkpoint
cursors; the next batch advance rejects misalignment before changing counts,
cursor or neural state. Do not silently pad, skip or reset a lane. Each
native advance retains one drive-change dispatch, two dispatches/tick and
one observation-settlement dispatch, independent of lane count.

Keep existing single-lane symbols as N=1 wrappers. Increment checked native
and builder ABI from 5 to 7 to identify the new layout/contract; epoch 6
stays an independently preserved arithmetic experiment. Record numerical
parent/order separately from ABI. Stale binaries must be rejected.

Validate products/offsets, lane IDs, device maximum buffer length and total
resident working set before allocation. Zero-edge graphs remain valid.
A failed GPU command or event overflow poisons the whole owner; validation
errors before dispatch are recoverable. Close is idempotent, serializes with
in-flight calls and preserves ownership until completion.

Lifecycle serialization and idempotent close are enforced by the Python owner
in Task2. As with the existing raw C API, native callers serialize operations
on one handle and destroy that handle exactly once after all calls complete;
do not add a native tombstone/handle registry as part of batching.

## Python training boundary

Add `MetalBatchExecutor(brains)` in `doom_learning_v6/metal/batch.py`.
It accepts distinct, compatible `MemoryBrain` instances with no initialized
Metal handle. Validate graph/configuration/plastic-edge identities, fixed
weights, finite state, schemas and nonaliasing mutable buffers before binding.

Attach a lane backend adapter to each brain for narrow observation transfers,
materialization, checkpoint provenance, restore, diagnostics and sparse weight
updates. A brain is owned by one executor; direct `brain.step()` while attached
fails before changing retinal/neural state. Executor close materializes healthy
lanes before releasing ownership. Poisoned lanes require explicit restore,
without silently running another backend.

Public interfaces:

```python
MetalBatchExecutor(brains: list[MemoryBrain])
executor.advance(steps: int) -> tuple[list[np.ndarray], float]
executor.step(luminances, duration_ms, *, learning=False,
              stimulations=None, lamina_bias=12.) -> tuple[list[np.ndarray], float]
executor.rgb_step(frames, duration_ms, *, learning=False,
                  stimulations=None, lamina_bias=12.) -> tuple[list[np.ndarray], float]
executor.metadata() -> dict
executor.close() -> None
```

`advance` consumes the current lane drives and returns independent copied
counts plus aggregate backend wall seconds. `step` accepts one luminance
vector per lane, a scalar duration/bias, scalar or per-lane learning booleans,
and one optional stimulation value per lane using existing pulse syntax.
Weights-frozen flags remain independently read from each brain. Validate all
lane inputs before any preparation; never partially advance a valid lane
because a neighbor's input is invalid.

Factor existing `MemoryBrain` preparation and centered-rule update into small
internal helpers used by both serial and batch stepping. Preserve operation
order and arrays exactly; serial tests and full serial/batch trace equality
guard the refactor. The executor uses the existing <=100-tick observation
bins, lane-local luminance smoothing and tonic/teacher drive, one native
batch advance per bin, then the unchanged float64 rule independently per lane.
Only the existing identified slots update. No sensory, teacher, decoder or
credit-rule changes.

For VisualMemoryBrain lanes, rgb_step accepts one RGB uint8 frame per lane and
uses the existing <=100-tick RGB subdivision, R8 coordinate/channel sampling,
linear-sRGB conversion, lane-local r8_light smoothing and R8 pulse mapping.
Factor that preparation into a helper shared with serial rgb_step. Validate
all frames before changing any lane; direct attached-brain rgb_step fails
before updating r8_light. RGB duration subdivision must match the original
VisualMemoryBrain behavior exactly, including a fractional final bin.

Checkpoint metadata reports `metal-batch`, numerical parent, ABI, order and
lane count/index; it must not claim CPU or serial Metal production. Neural
materialization occurs only for explicit checkpoints/diagnostics/close,
not each training bin. Restore preserves lane state and refuses unsafe
index schemas. Aggregate timing is counted once, not copied and summed as
though every lane paid the whole GPU command.

## Verification and measurement

Native macOS tests compare distinct lanes against serial epoch 5 at 1, 40,
80 and 100 ticks, through delayed-ring wraps, empty graphs, duplicate/self
edges, modulatory-only inputs, frozen and plastic execution. Compare float
bits (including signed zero), discrete fields, delayed membership, all spikes,
eligibility/rates/u/w, weights and actual fixed decoded outputs. Existing
fixture-only denormal handling stays confined to its original CPU comparison.

Repeat fresh N=1/N=2 runs, reverse lane order, and perturb only one lane's
pixels/state/teacher/weights. Unchanged neighbors must retain exact state and
controls. Exercise invalid lane IDs, mismatched configuration/cursors, aliases,
overflow/capacity, reentrant calls, close and checkpoint provenance.
No mocked timing result substitutes for actual native equality.

After two-lane correctness, root runs fresh unchanged retained 40/80 ms gates,
two-second dark and original RGB controls on actual M4 Pro. Verify each lane
against serial epoch 5, not merely against lane zero. Preserve the known
epoch-5 two cancellation failures separately; five unaffected carry controls
and serial/batch agreement are expected. New gate or identity failures stop
scaling until corrected, without discarding failed evidence.

Benchmark serial N=1, batched N=2, then N=4 only after correctness, using the
same original complete RGB episodes and host plasticity protocol. Include
different inputs/learning flags, warmup separately, wall phase totals,
aggregate RGB brain-seconds/wall-second, per-lane completion latency,
native GPU/encoding/wait/event/transfer/rule times, working set and variability.
Timing runs disable full spike/state diagnostics and record instrumentation
overhead separately. Compare the existing CPU measurement with explicit
concurrency; one game's latency and aggregate throughput are distinct.

No arbitrary speed target stops Metal development. A reproducible useful
gain beyond measurement noise supports scaling; absent gain directs profiling
and refinement while lane count stays fixed. The aspirational 1000x target
has no validated forecast. The CPU 97% backend fraction and 20,000 dispatches/
brain-second do not establish a measured Metal bottleneck.

## Learning follow-up and exclusions

First reproduce the existing one-pair protocol with identical teacher timing,
held-out frozen evaluation and erased/frozen controls. This verifies executor
identity, not learning. A subsequent separately versioned plan adds no-teacher
and independently trained demonstration/held-out pairs/seeds, a predeclared
primary hazard behavior and uncertainty, teacher-free frozen live transfer,
retention and shifted/frozen/erased controls. Preserve conflicting MAE/recall
and all negative CPU/Metal results.

Ordering/modulation probes should capture original within-call contribution
order without adding observation settlements. Separate replay of order and
modulation association is needed for causal attribution. They are a later
bounded experiment, not bundled with this executor.

Exclude game engine ports, policy substitution, pruning, telemetry actions,
GPU plastic-rule migration, new credit rules, public deployment and launch
claims. Update published “How it works” only as part of a separately authorized
release with exact deployed evidence.
