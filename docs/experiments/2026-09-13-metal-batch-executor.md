# Resident Metal multi-trajectory executor

## Scope and fixed model

This experiment shares one immutable graph across independent resident Metal
trajectories. It retains MaleCNS v1.0's 166,700 modeled neurons and every one
of the 25,582,938 released connections between retained entries, including
duplicate, self, weak and modulatory edges. Each lane has its own weights,
neural state, delayed queues, activity membership and learning state.

Execution ABI 7 keeps the reviewed epoch-5 numerical order from
`b24249ecfee12aa84b55ed803ffe47bd09c487ad`. Epoch-6 incoming-carry arithmetic
and its failed longer controls remain a separate experiment. Batch execution
does not replace the connectome with a game policy or change its dynamics.

Live RGB conversion and sensory current preparation remain separate from
modeled propagation, reinforcement and the fixed DNp20/DNpe017 decoder.
The host float64 centered rule and 50 ms efficacy filter still update only
the existing 4,184 KC-to-MBON11 plastic slots. Sensory mapping, baseline
weights, motor decoding and all nonplastic connections stay fixed. These
mappings and dynamics are chosen hypotheses, not established fly physiology.

## Implementation and lifetime

`MetalBatchExecutor(brains)` owns synchronized lanes. It performs one command
encoding with two propagation dispatches per tick, followed by the original
per-lane eligibility and centered-rule work. Only the owner advances lanes;
direct owned-brain stepping is rejected. Checkpointing, restoration and
diagnostics validate array bindings before native pointers are used.

Invalid restore candidates leave live state unchanged. Healthy close
materializes each lane and releases the handle exactly once. Failed cleanup
still releases resources, preserves explicit recovery guards and reports the
original failure. A poisoned owner never silently switches to another backend.
See the [design](../superpowers/specs/2026-09-13-metal-batch-executor-design.md)
and [saved execution record](2026-09-13-metal-batch-executor-invocation.md).

## Completed engineering verification

At fixed source `08015d7101778e6aa3fc88103418d811db1481e5`, the actual Apple
M4 Pro run reports 173 focused passes, 86 unchanged serial passes with one
platform skip, and 508 full-suite passes with three platform skips. The full
suite retains the existing 50 dependency deprecation warnings. Native ABI 7,
strict Metal 2.4 compilation, binaries and device identity match the reviewed
build. Independent native and Python task reviews approved their final code.

Fresh serial CPU/Metal gates pass over 40 and 80 ms: exact recorded spike
events, counts, fixed decisions and plastic weights, with bitwise repeated
Metal state. CPU and Metal complete-state digests still differ. These short
gates preserve their original thresholds and do not certify long trajectories
or the batch lanes.

The original seven incoming-carry controls reproduce two failures and five
passes. Catastrophic carry and duplicate-edge cancellation fail conductance
bits under epoch 5. Their actual nonzero test exit is preserved separately;
ordinary carry, zero carry, refractory rejection, modulation-only and empty
topology pass. No new tolerance or golden masks these failures.

## Retained RGB and throughput protocol

The completed N=2 diagnostic checks each lane's complete
named state, weights, spikes, cursor and fixed decoding against its own serial
Metal reference through two seconds of dark input and 35 original RGB frames.
All six fresh baseline/repetition/reversed-order/pixel/teacher/state trials
pass exact per-lane comparisons. Repetition and lane permutation preserve both
traces; each intervention changes the tested lane's state sequence while its
neighbor remains bitwise equal. Free-memory samples remain at 80-87%, with
zero swap use. All original traces and checkpoints were copied and byte/SHA
matched. Full diagnostics are not throughput. See the bounded
[identity report](../../outputs/doom-learning/metal-batch-executor-20260913/prefix-two-identity.json).
Complete original-workload reproduction has also passed the independent
[audit](../../outputs/doom-learning/metal-batch-executor-20260913/full-two-audit.json).
All eight canonical controls match, totaling 7,886 original RGB frames and
2,253,143 original RGB ticks. All three learned checkpoints' numeric arrays
match bitwise; teacher timing, frozen memory, erasure and passive retention
match exactly. The CPU-library binary fingerprints differ across target
directories and are separately verified and recorded. CPU source and flags
agree; instruction-dump identity is not claimed.

The complete workload uses the checksum-pinned GameWAM `defend_the_center`
APPO episodes: 982 training frames and 988 held-out frames at 35 Hz, with
capture frame skip 1 and policy action repeat 4. These
are stationary agent demonstrations, not human or hazard-arena demonstrations.
Teacher error is `min(abs(turn-target)/6,1)`; the next frame receives
`4*error` at the two existing PPL101 entries, with `eta=.001`. Evaluation
uses frozen memory and zero teacher input. All original eight controls,
five-second unfrozen dark retention and exact baseline erasure are reproduced.
With learning disabled and weights unfrozen, the chosen 30-minute passive
memory decay and 50 ms weight filter continue. Frozen evaluation holds
`memory_u`, `memory_w` and weights fixed; neural activity and rate traces evolve.

Only equal-duration phases share advancing lanes. Declared duplicate controls
are execution controls, not independent learning replications. Three fresh
serial-two and batch-two complete episode pairs measure aggregate RGB brain
seconds per wall second. The completed serial baseline's three pair walls
are 746.24, 746.11 and 746.55 seconds, with median aggregate throughput
0.15085 brain-seconds per wall-second and 1.06 GB actual native allocations.
All 24 copied artifacts, 12 canonical lane traces and six learned checkpoints'
numeric arrays independently match the original reference; see its
[audit](../../outputs/doom-learning/metal-batch-executor-20260913/timing-serial-two-original-audit.json).
This baseline uses current ABI 7 serial wrappers with two interleaved handles
and one active GPU command. It is not a fresh matched measurement of the
historical ABI 5 parent. Three fresh N=2 batch pairs also reproduce all original
traces and learned numeric arrays. Pair walls are 503.81, 504.02 and 504.59 seconds;
median aggregate throughput is 0.22335 brain-seconds per wall-second, a measured
1.48058x gain over the current serial baseline. Actual native allocations fall
from 1,055,568,440 serial bytes to 642,561,724 batch bytes. Warmup and full-state
diagnostics are excluded from these RGB wall intervals. The gain exceeds the
observed ranges; it does not establish improved learning quality. See the
[batch original-reference audit](../../outputs/doom-learning/metal-batch-executor-20260913/timing-batch-two-original-audit.json)
and [paired timing audit](../../outputs/doom-learning/metal-batch-executor-20260913/timing-two-paired-audit.json).
Initialization, warmup, per-lane completion, actual
native memory, pressure, swaps and instrumented host/GPU costs are reported
separately. Overlapping GPU, native and wait timings are never added as wall.

The five completed pairs include two declared duplicate controls: 9,856
processed frames and 281.6 aggregate RGB brain seconds in 1,254.43 RGB wall
seconds. Their extra workload prevents a direct comparison with the original
eight-phase CPU wall time. Actual native allocations are about 643 MB,
including 413 MB shared topology. The first pair records 239.48 GPU seconds,
0.23 encoding seconds and 1.37 host-rule seconds; stages can overlap. These
single samples guide profiling and establish no repeated speedup.

N=2's exactness, isolated lanes, repeated gain and actual zero-swap headroom
justify N=4 validation. All six fresh four-lane trials pass exact own-serial
full-state/spike/decoder and isolation checks. All 32 copied artifacts, 24 final
checkpoint-state hashes and original N=2 records are independently verified.
Lanes 2/3 are declared copies of 0/1 for execution checks. The allocation probe
measures 872,016,316 native bytes before advancing, about 3.37 GB peak process
RSS and zero swap. Actual training allocations grow to 872,066,524 bytes,
including sparse-update workspace. See the
[four-lane identity](../../outputs/doom-learning/metal-batch-executor-20260913/prefix-four-identity.json),
[artifact audit](../../outputs/doom-learning/metal-batch-executor-20260913/prefix-four-audit.json)
and [allocation probe](../../outputs/doom-learning/metal-batch-executor-20260913/n4-resident-allocation.json).

Three complete N=4 timing pairs also finish and pass both independent audits.
All 30 copied artifacts match target bytes/SHA; all 24 canonical lane traces
and all 12 learned checkpoints' numeric arrays reproduce the original controls
and the fresh serial baseline. Pair walls are 600.90, 602.23 and 602.58 seconds,
each processing 225.1428 aggregate RGB brain seconds. Throughput ranges from
0.37363 to 0.37468, with median 0.37385. All six recorded phase samples have
82% free memory and zero swap. See the
[original-reference audit](../../outputs/doom-learning/metal-batch-executor-20260913/timing-batch-four-original-audit.json)
and [paired timing audit](../../outputs/doom-learning/metal-batch-executor-20260913/timing-four-paired-audit.json).

| Execution | Median pair wall (s) | Aggregate RGB brain s / wall s | Actual native bytes | Gain over serial |
| --- | ---: | ---: | ---: | ---: |
| ABI 7 interleaved serial, 2 lanes | 746.24 | 0.15085 | 1,055,568,440 | 1.00000x |
| Resident batch, 2 lanes | 504.02 | 0.22335 | 642,561,724 | 1.48058x |
| Resident batch, 4 lanes | 602.23 | 0.37385 | 872,066,524 | 2.47827x |

N=4 increases aggregate throughput by 1.67385x over N=2 while each complete
pair takes 19.49% longer. The measured ranges remain separated. Select four
lanes for the next fixed-lane acceleration experiments; this is a measured
choice among tested widths, not a claim of optimal width or faster single-fly
learning. Extra lanes remain execution duplicates, not independent trained
replications. A lack of gain in a subsequent change
directs profiling and refinement with fixed lanes. No arbitrary CPU multiplier
closes Metal development, and no 1000x speed forecast follows these checks.

## Scientific interpretation and publication

Original long CPU mismatches and failed/mixed learning controls remain retained.
The prior single demonstration pair worsened held-out turn MAE while improving
direction recall slightly. Changed weights, survival or numerical exactness
alone do not establish useful learning or biological validation.

Bounded, source-identified numerical reports retain the measured execution
results and limitations. Raw game pixels, traces, checkpoints, logs,
runtime scripts and machine-specific origins remain ignored. The live server
and public stream are unchanged; this experiment gives no launch green light.
