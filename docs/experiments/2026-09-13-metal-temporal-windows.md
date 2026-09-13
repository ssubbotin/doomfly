# Metal Temporal Windows: Retained Execution Study

## Scope and status

This isolated experiment accelerates modeled propagation while retaining all
166,700 MaleCNS v1.0 neurons, 25,582,938 released connections and 4,184 existing
KC-to-MBON11 plastic slots. Duplicate, self, weak and modulatory connections
remain present. Reference scheduling remains the default. The live service is
unchanged; this document supplies no launch approval.

Native implementation at `44e00de919e9eb2d38344318705bbd9228588b65` passed
independent specification and code-quality review. Python integration at
`e81d83dbaaba28e38d32cee47c333193224ed013` also passed independent review,
including restoration of the default scheduler's invalid-input rejection gate.
Fresh full-connectome control reproduction and lane-isolation checks passed.
Three complete paired throughput measurements are verified, giving a 1.7404x
ratio of median rates. Independent review gates apply before branch publication.
Useful learning remains unestablished.

The [design](../superpowers/specs/2026-09-13-metal-temporal-windows-design.md),
[implementation plan](../superpowers/plans/2026-09-13-metal-temporal-windows.md)
and [cross-model decision record](2026-09-13-metal-temporal-window-council.md)
describe the hypothesis and acceptance criteria.

## Schedule and boundaries

Execution ABI 8 preserves the epoch-5 zero-seeded ascending-incoming arithmetic,
explicit FMA sites, decay tables and host-double centered plasticity rule.
`df_metal_create_batch_windowed` fixes a handle's window width in 0..18.
Zero uses the original two-dispatch-per-tick schedule. Positive widths prepare
time-specific incoming presence, then execute each neuron's ticks in order.
Ordinary tracked dispatches provide the required preparation/worker boundary.

At delay 18 with 19 ring slots, W<=18 keeps newly emitted events outside the
logical consumed interval. Preparation exclusively clears consumed slots;
workers preserve physically reused future bits and reset from live membership.
Gathering retains ascending edge order, signed-zero and zero-weight presence
semantics. Inactive neurons retain lazy evolution. Drive application and final
settlement stay once per existing bin; eligibility and plasticity remain once
per bin on the host.

For 100 ticks, W18 changes dispatch accounting from 202 to 14. This count is
not a throughput prediction. The two additional scratch planes have checked
products and device/replacement limits. Their full-graph unaligned size is
69,564,024 bytes per lane at W18; measured resident accounting must include
actual buffer lengths and event/sparse workspace growth.

## Completed native checks

The initial validation machine is the M4 Pro, using Objective-C++ and strict
`macos-metal2.4`, no fast math, disabled FP contraction, and macOS 13 minimum.
At native implementation source `44e00de919e9eb2d38344318705bbd9228588b65`,
the full M4 suite completed with 1,146 passes and three platform skips. Its
portable suite completed with 416 passes and 733 platform skips.
Both reported 50 existing dependency warnings. Skips provide no device evidence.

The Python integration's pre-correction full suites completed with 1,179 M4
passes (three skips) and 428 portable passes (754 skips) at `8f645ea`. The
test-only correction at `e81d83d` then passed all 11 M4 RGB tests and the three
portable RGB checks (eight device skips), with source and binary identities
recorded before and after the run. Fresh final suites at `e81d83d` completed
with 1,180 M4 passes (three skips) and 428 portable passes (755 skips), both
with 50 existing warnings. Sources, native binaries and runtime scripts matched
their pre-run hashes afterward. Earlier counts retain their source identities.

The 625-test native extension includes all 19 cursor residues, ten advance
lengths and W1/W2/W18, complete raw-state/event comparisons, lane isolation,
scratch reuse, zero-edge accounting and rejection/recovery paths. Seven
separately compiled faulty candidate kernels each produced a genuine state
assertion failure against an independent healthy reference. They exercise
prefilled reset, future-slot reuse, shared-word masks, zero seeding, zero-weight
fast/modulatory presence and lazy evolution. Actual device exhaustion was not
forced. Numerical tests establish engineering behavior within their scope.

## Bounded retained diagnostics

Fresh four-lane diagnostics at `e81d83d` processed 35 training frames and 35
held-out frames per lane. Each phase advances exactly 10,000 RGB ticks per
lane, separate from its two-second dark warmup. Every candidate matched both
complete canonical traces and all eight full numeric checkpoints against
reference0. All array dtypes, shapes and bytes and fixed model/configuration
metadata were checked. CPU source/flags agreement is recorded separately from
binary identity.

| Window ticks | Aggregate RGB brain seconds / wall second | Reference-relative rate | Native buffers, decimal GB |
| --- | ---: | ---: | ---: |
| 0 | 0.583 | 1.000 | 0.872 |
| 1 | 0.446 | 0.765 | 0.888 |
| 2 | 0.582 | 0.998 | 0.903 |
| 18 | 0.923 | 1.583 | 1.150 |

W18 is the candidate for complete acceptance. Its measured additional native
allocation is 278,256,096 bytes for four lanes. Available memory remained at
least 82% with zero swap in these runs. The separate guard-only comparison matched all
traces/checkpoints but gave 0.991 times its parent's short diagnostic rate.
Negative W1 and guard measurements remain preserved.

These are single bounded diagnostics, with no claim about complete-episode
throughput or dispersion. Their eight aggregate RGB brain seconds use the
actual processed scope; the source episodes still contain 982 and 988 frames.
Native buffers exclude other process allocations. Default reference0 remains
unchanged.

Across both short phases, actual dispatches fell from 40,420 to 2,800.
Reported GPU time fell from 12.05 to 6.97 seconds; RGB wall fell from 13.72 to
8.67 seconds. CPU encoding was 0.017 and 0.003 seconds respectively; the host
plasticity rule took about 0.125 seconds in each. GPU work still accounts for
most observed wall time. These overlapping stages cannot be added as wall
time, and total GPU time per dispatch is not a hardware latency measurement.

## Full-run protocol and interpretation

The fresh 40/80 ms CPU/Metal gates passed with matching spikes, controls and
weights and bitwise Metal repeats; full CPU/Metal state digests remain different.
Both known catastrophic/duplicate incoming-carry controls still fail numerically
in the original seven-case test (five passes). All six N2 serial-reference and
isolation trials passed, comparing every state field, weight, scalar and spike
event after 200 dark 10 ms steps and 35 RGB frames per lane. Repeats/permutations
match; pixel/teacher/state perturbations change the tested lane while preserving
neighbor traces. The same six N4 trials also passed, with all 17 full
repeat/permutation/neighbor checkpoints matching bitwise and zero swap.
All eight original complete controls now match their canonical traces; all three
learned checkpoints match every numeric array (24 each). Ten processed
trajectories include two declared frozen duplicates: 9,856 RGB frames and
281.6 aggregate RGB brain seconds. Original CPU source/flags agree with current
checkpoint metadata while CPU binary hashes differ. Fifteen original physical
artifact pins and all 19 complete copied artifact hashes were independently
checked. Correctness wall times are excluded from throughput because artifact
transfer overlapped part of this validation. Matched timing ran separately.

| Held-out condition | Turn MAE, degrees | Balanced direction recall |
| --- | ---: | ---: |
| Learned | 3.2030 | 0.3391 |
| Frozen / erased | 3.1531 | 0.3259 |
| Shifted teacher | 3.2260 | 0.3106 |
| Five-second retention | 3.1425 | 0.3374 |

Training changes 1,953 plastic slots; frozen training changes zero. Learned
held-out turn error is worse than frozen, with slightly higher directional
recall. The constant-left baseline has MAE 2.1681 and balanced recall 1/3.
This single training/evaluation pair provides mixed evidence. Retention uses
50,000 dark ticks with zero teacher and disabled learning; unfrozen passive
filtering continues. Frozen evaluation keeps weights and memory u/w fixed.

Fresh guard-only/reference and W1/W2/W18 diagnostics precede unchanged 40/80 ms
CPU gates, six N2 and six N4 isolation trials, all eight original RGB controls
and three alternating-order matched N4 complete timing pairs. Timing pairs use
all 982 training and 988 held-out frames at the original 35 Hz clock. Short
diagnostics are labelled separately. Duplicate lanes are execution controls.

Throughput is aggregate completed RGB brain seconds divided by actual RGB wall
time. Initialization, dark warmup and checkpoint capture are reported separately.
All canonical traces and learned checkpoint arrays must match before a pair
counts. There is no arbitrary CPU multiplier stop gate or 1000x forecast.

All three complete N4 pairs at `e81d83d` independently match every original
982-frame training and 988-frame held-out canonical trace and all four learned
numeric checkpoints for both schedulers. Each run advances 225.1428 aggregate
RGB brain seconds. No artifact transfer or competing GPU job overlapped timing.

| Pair and order | Reference0 RGB wall, seconds | W18 RGB wall, seconds | Rate gain |
| --- | ---: | ---: | ---: |
| 1: R,C | 609.3485 | 350.0660 | 1.74067 |
| 2: C,R | 609.2647 | 349.8041 | 1.74173 |
| 3: R,C | 608.5978 | 350.1857 | 1.73793 |

Median aggregate rate increases from 0.369532 to 0.643144 RGB brain seconds
per wall second, a ratio of medians of 1.740428x. Individual paired gains span
1.737929..1.741731 (median 1.740667). These three repetitions describe this
fixed workload; they provide no uncertainty estimate for independent learning.
All 78 complete timing artifact roles were copied and independently matched by
bytes/SHA256. Local checks confirm all six original traces, 24 original learned
checkpoint comparisons and 12 direct reference/candidate comparisons, each
checking all 24 numeric arrays plus fixed metadata.

Complete-run GPU time spans 560.72..561.87 seconds for reference0 and
302.54..302.75 for W18. CPU encoding spans 0.550..0.562 and 0.083..0.087
seconds; reported host plasticity spans 3.54..3.65 and 3.57..3.64 seconds.
Dispatches fall from 1,137,534 to 78,800; encoder count stays 5,910. GPU work
remains dominant. Counters overlap and do not constitute an additive wall-time
decomposition; dense grids do not measure active edges.

Actual native storage is 872,066,524 bytes for reference0 and 1,150,322,620
for W18 in every measured phase. Available external process peak resident
readings span 3,473,571,840..3,477,766,144 bytes for reference0 (two samples)
and 3,734,716,416..3,758,161,920 for W18 (three samples), including setup and
checkpoint I/O. The reading is the maximum individual waited-process peak,
with [Darwin accounting](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_resource.c)
preserving the larger child/parent resident peak. It is separate from native
buffers and simultaneous process-tree memory. The first reference's process
peak is unavailable. All twelve phase-end pressure samples report at least
81% free with zero swap. Source, native binary and runtime hashes match their
pre-run pins after all timing jobs; five generated CPU manifests stay valid.

RGB sensory input, propagation, reinforcement, identified plasticity and fixed
DNp20/DNpe017 control decoding remain separate. The stationary APPO examples
are engineering fixtures, with independent hazard/human demonstrations still
needed. Previous long CPU differences, incoming-carry failures and mixed held-out
learning outcomes remain preserved. More speed, altered weights and numerical
agreement establish neither useful learning nor a biologically validated brain.

## Evidence

The bounded [provenance](../../outputs/doom-learning/metal-temporal-windows-m4pro/provenance.json),
[diagnostics](../../outputs/doom-learning/metal-temporal-windows-m4pro/diagnostics.json),
[correctness](../../outputs/doom-learning/metal-temporal-windows-m4pro/correctness.json)
and [artifact inventory](../../outputs/doom-learning/metal-temporal-windows-m4pro/artifacts.json),
plus the [complete paired timing](../../outputs/doom-learning/metal-temporal-windows-m4pro/timing.json),
preserve source-specific measurements and relative raw roles. Independent task
and final branch review govern publication. All 263 inventoried raw roles
(14,550,932,774 bytes) were copied and physically matched by bytes and SHA256; the raw pixels,
checkpoints, binaries and machine-specific paths remain ignored. Portable JSON
contains metadata only.
