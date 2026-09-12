# Explicit Metal FMA: Corrected Voltage, Remaining Incoming Differences

## Isolated change and numerical results

At production source `9d24668`, ABI epoch 5 adds three explicit voltage `fma`
operations matching the measured M4 CPU instruction association. CPU source,
strict Metal 2.4 flags, resident coefficients, incoming propagation, plasticity,
166,700 retained neurons, 25,582,938 edges and 4,184 plastic slots stay fixed.
The [saved invocation](2026-09-13-metal-arithmetic-invocation.md) identifies
each exact-source diagnostic and the subsequent exploratory training.

Both unchanged [40 ms](../../outputs/doom-learning/metal-arithmetic-20260913/parity-40ms/report.json)
and [80 ms](../../outputs/doom-learning/metal-arithmetic-20260913/parity-80ms/report.json)
full-graph gates pass with exact events, counts, controls and weights, and
bitwise repeated Metal state. Two fresh micrographs reproduce state `65524d57...`;
previous arithmetic epochs and unchanged count/event goldens are retained.

The [two-second dark diagnostic](../../outputs/doom-learning/metal-arithmetic-20260913/long-parity/results.json)
now has exact spikes and equal controls after the first original RGB frame.
The preceding table-only build diverged at 48 ms and decoded zero turn. Small
voltage/conductance differences still begin after incoming arrivals at the third
10 ms bin. Aggregate/spike agreement does not establish bit-identical state.

## Test-first evidence and platform limitation

The test-only M4 checkpoint fails all eight voltage regressions. Explicit FMA
passes six; two composite cases expose unchanged conductance underflow at
440 ms: CPU subnormal bits `4320706`/`5400880`, Metal zero. Apple's specification
permits subnormal flushing. [Metal specification, sections 8.1/8.5](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

The documented fixture-local ruling retains bit-exact composite checks through
43 bins, and voltage/adaptation/discrete fields through 200 bins. Only finite,
nonzero CPU subnormal conductance versus Metal zero is accepted afterward;
one positive and seven negative literal controls constrain this exception.
Production arithmetic and every
full-graph gate remain unchanged. At final fixture source `e9e19c1`, Linux
passes 313 tests (61 Apple skips); actual M4 passes 371 (three platform skips).
Both retain 50 baseline dependency deprecations. Independent task review approves
spec compliance/quality. [Verification](../../outputs/doom-learning/metal-arithmetic-20260913/software-verification.json)
and [artifact checksums](../../outputs/doom-learning/metal-arithmetic-20260913/artifact-inventory.json)
preserve the original failure/trial evidence and external specification identity.

## Original training prefix and next work

All eight 35-frame phases complete, with identical original pixels/timing,
unchanged invariants, teacher-free frozen evaluation and exact within-Metal
erasure/frozen repetition. Every training arm and the learned plastic held-out
prefix matches CPU on all action/spike/memory/teacher fields. Other held-out
controls remain different: frozen/erased fixed controls match 29/35 frames,
retention and shifted match 28/35. [Comparison](../../outputs/doom-learning/metal-arithmetic-prefix-20260913/comparison-v2.json)
separates actual controls from diagnostic readouts. Prefixes remain incomplete
relative to source episodes and establish no useful learning.

Metal phases take 8.88–9.20 wall seconds versus CPU 4.84–5.12, including two
seconds of dark brain warmup. RGB neural execution takes 4.24–4.55 wall seconds
per brain second. This is a correctness improvement, with no measured speedup.

The complete original Metal cohort runs separately at fixed `e9e19c1`, with
the matching 80 ms identity guard and all controls. It remains exploratory;
whole-episode parity and useful learning require their own evidence. Next local
work isolates incoming conductance association, followed by separately measured
ordering/modulatory settlement. Logs, complete traces, snapshots and checkpoints
remain ignored. No public service, launch status or unrelated workload changed.
