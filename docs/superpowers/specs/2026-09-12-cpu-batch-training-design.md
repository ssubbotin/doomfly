# Shared-Graph CPU Training Integration Design

## Approval and Purpose

User design approval on 12 September 2026 covers C++ multi-trajectory training
integration preserving sensory/learning timing, alongside bounded strict-math
Metal parity/speed testing. This file records the training architecture in
detail for written-spec review. It extends the completed
[propagation executor](2026-09-11-cpp-multi-trajectory-executor-design.md), not the
candidate's biological claims. The scalar runner remains the independent oracle.

Run complete independent RGB-to-neural-to-Doom training trajectories using one
immutable finalized connectome and persistent C++ propagation workers. Improve
measured end-to-end throughput; 1,000x real time remains an aspiration, not an
acceptance result. Preserve failures and all original controls.

## Binding Constraints

- Retain all 166,700 MaleCNS v1.0 neurons, 25,582,938 released connections and
  4,184 identified plastic slots. No cropping, pruning or substitute game policy.
- Preserve 0.1 ms neural integration, 35 Hz game observations and lane-local
  rule bins of at most 100 steps (10 ms).
- Keep live RGB mapping, propagation, reinforcement, centered plasticity and
  fixed DNp20/DNpe017 decoding separate and traceable.
- Health schedules the declared damage stimulus; coordinates and other game
  telemetry are observer-only and never select actions.
- Preserve the existing `rule.advance` arithmetic and per-lane reduction order.
- No per-lane full-edge weight array, graph reconstruction or calibrated brain.
- Preserve the serial `survival.episode` and existing uniform executor API.
- Use C++17, NumPy and existing ViZDoom dependencies. No additional runtime
  framework or Apple-specific dependency for CPU training.
- Store machine origins only in ignored configuration. Exclude graph datasets,
  source snapshots, dependency checkouts, binaries and game assets from evidence.
- Numerical equality and changed weights do not establish biological validity
  or learning. Public deployment and launch gates remain separate.

## Shared Model and Lane Ownership

`SharedTrainingModel.from_calibrated_brain(brain)` runs after visual R8 sign
corrections and tonic calibration have finalized the original baseline. It owns
the existing `SharedCpuGraph`, immutable IDs/retinal and R8 geometry, circuit
indices/gain, baseline plastic weights, tonic currents, DAN baseline, readout
metadata and a shared pristine reset template. Mutable calibration objects are
released after extraction; constructing a model does not recalibrate each lane.

Each `TrainingLane` owns an existing `CpuBatchLane` (23,854,924 native-state bytes
in the recorded full-graph release), plus luminance/R8 filters, KC/DAN rate
traces, `memory_u`, `memory_w`, frozen flag, time/spike counters and one tic-count
accumulator. Its plastic overlay is 4,184 float32 values. All registered buffers
remain stable: reset and restore copy validated values in-place between calls.
Memory snapshots carry weights, memory arrays and configuration identity.

Each episode owns its game, controller filter, frame, neural origin, game tic,
constant-US segment position, delivered-dose array, damage-pulse deadline and
observer trace. One cooperative Python coordinator handles host transitions;
C++ workers advance independent lanes. Python sensory/rule/controller work
stays scalar per lane initially. No shared mutable state between trajectories.

## Variable-Step Native Interface

Upgrade the CPU executor to ABI 2. Keep existing graph/lane descriptor layouts,
uniform `advance(steps)` behavior and uniform timing layout. Add:

```c
typedef struct {
  double native_wall_seconds;
  int32_t lanes_advanced;
  int32_t workers;
  int32_t max_steps;
  int64_t total_lane_steps;
  uint64_t generation;
  int32_t pool_threads;
} df_cpu_batch_lane_timing;

int df_cpu_batch_advance_lanes(df_cpu_batch_handle handle,
    const int32_t *steps_by_lane, int32_t lane_count,
    df_cpu_batch_lane_timing *timing);
```

The request vector exactly matches registered lane order/count and contains
0..100 steps per lane. Copy it into executor-owned storage before releasing a
generation to workers. Validate requests, all registered states (including
parked lanes), aliases and each cursor's overflow before any lane mutation.
Reject malformed input without changing counts, cursors or generation.

Only positive requests clear counts and invoke the original scalar integration
body with that lane's step count. A zero request leaves every lane byte
unchanged, including counts and drive history; do not invoke the scalar body
with zero. All-zero calls perform no dispatch, preserve generation and report
zero duration/work. Keep lifecycle synchronization and worker-failure poisoning.

Python `CpuBatchExecutor.advance_lanes(steps_by_lane)` accepts an aligned,
contiguous int32 vector, copies it and rejects overlap with registered buffers.
It retains reentrancy/address/schema/poisoning checks and returns per-lane
counts snapshots, with `None` for parked lanes. Variable timing reports active
lane count, configured workers, max/total requested steps and pool generation.
ABI-1 clients enforcing the version probe require an intentional upgrade.

## Exact Episode and Learning Schedule

1. Reset fast state while retaining memory and current efficacy overlay. Set
   the episode's frozen flag before the original 2,000 ms black warmup.
   `learning=False` still permits passive efficacy dynamics when unfrozen.
2. Create the game after warmup and record the episode neural origin.
3. For tic `k`, preserve the original neural-step expression:
   `round((k+1)*10000/35) - (cursor-origin)`. Capture one actual RGB frame.
4. Split the tic at every boolean US transition. Within each constant-US
   segment, use `min(100, segment_remaining)` independently for each lane.
   Global-minimum or globally aligned bins would alter filter/rule arithmetic.
5. `TrainingLane.prepare_bin(frame, ticks, learning, us_active)` preserves the
   existing R8 and luminance filtering, retinal sampling, tonic drive and US
   current formulas. Dispatch all ready lane-local requests together.
6. `TrainingLane.finish_bin(counts)` applies the unchanged centered rule using
   `interval=ticks*dt`, `seconds=interval/1000`. Publish
   `baseline_plastic*(1+memory_w)` through float32 overlay assignment before
   the next bin. Update traces even when frozen; then efficacy remains fixed.
7. Accumulate all bins for that tic. Decode once using `steps*.0001`, act once,
   record telemetry, then schedule observed damage until `cursor+2000`.
   Damage affects subsequent neural time only.

`EpisodeMachine.next_request()`, `accept_bin(counts)` and `finish_tic()` implement
these host transitions. `TrainingCoordinator` drains transitions, dispatches
positive requests once and completes only those lanes. It records complete
per-tic input/sensory/spike/action/health/position/clock traces.

Cohort tasks preserve dependencies: second training episodes inherit first
episode memory; shuffled training waits for its completed plastic donor and
retains the exact timing-shift seed/dose procedure; each evaluation restores
the same arm snapshot independently. Retention uses five seconds of existing
black dynamics, unfrozen and without learning, followed by fast reset and
ordinary frozen evaluation. Erasure resets memory and efficacies to baseline.

## Verification and Reporting

Native oracle tests cover heterogeneous requests `[100,86,0,1]`, all-zero calls,
parked-lane rotation, differing cursors, lane/worker ordering, uniform API
equivalence, aliases, late invalid requests/states, nonfinite parked state,
overflow and poisoning. Require all-before-mutation rejection and exact native
field/count equality against independent scalar CPU calls.

Training-bin tests cover actual RGB, 99/100/101 steps, 285/286-step game tics,
US changes around rule boundaries, frozen/passive dynamics, reset and memory
restore. Compare every native field, sensory filter, rate/memory array, overlay
and count accumulator after every bin. Keep the existing scalar implementation
independent; never define expected values through the coordinator under test.

On the full retained graph, compare complete episodes and learned-state hashes
with `survival.episode`: plastic/frozen/shuffled training, second training,
held-out evaluation, black input, retention and erasure. Preserve the known
positive `42053/62053` saved-weight case and negative controls as diagnostics;
post-hoc replay is not independent learning confirmation.

Measure at least five repeated matched end-to-end workloads at 1/2/4 lanes.
Report aggregate actual neural seconds per batch wall second, entire
episode/cohort and construction time, native/sensory/rule/game/reporting costs,
active-lane occupancy, request sizes, latency, RSS/shared/lane allocations,
awake/realtime clocks and source/build/graph/input/report identities. A shared
dispatch's duration is observed batch latency, not measured per-lane latency
divided by lane count. Preserve failed parity and throughput results; short
propagation benchmarks cannot certify complete training acceleration.
