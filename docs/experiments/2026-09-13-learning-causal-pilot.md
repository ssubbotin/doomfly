# Causal-learning pilot on Metal

## Scope and result

This private offline mechanism pilot used
[`causal_pilot.py`](../../doom_learning_v6/causal_pilot.py) at source
`982475ad44bdeb9b47fe1d7e174fdf3d347dae13` and the
released MaleCNS v1.0 graph: 166,700 modeled neurons and 25,582,938 retained
connections. It tested a fixed, historically inspected pair of stationary APPO
`defend_the_center` demonstrations: 982 training frames (seed 200000065) and
988 diagnostic-evaluation frames (seed 200000071), both at 35 Hz. The latter is
not fresh validation; there are no independent learner replicas.

The primary class-balanced angular MAE was harmful or mixed across the causal
arms. Finite manual perturbations changed fixed-decoder output, so the approved
next action is a separately designed, bounded same-slot efficacy optimization
comparison on Metal (ES/SPSA). This pilot ends here. It did not implement or
test an optimizer, and it does not establish useful learning, biological
learning, general Doom learning, or readiness for a public launch.

Compact portable measurements, byte counts, hashes, and all per-arm metrics are
in [`results.json`](../../outputs/doom-learning/learning-causal-pilot-m4pro/results.json).
The [approved design](../superpowers/specs/2026-09-13-learning-causal-pilot-design.md)
and [execution plan](../superpowers/plans/2026-09-13-learning-causal-pilot.md)
define the controls and scoring before these results were known.

## What was held fixed, and what could change

Live 640x480 RGB frames were mapped to sensory input, propagated through the
retained neural graph, and read by the fixed DNp20 turn and DNpe017
forward/attack decoder. No game telemetry selected actions, aimed, recovered a
controller, or changed the decoder. The model uses chosen host dynamics and
inferred mappings. They are separate from measured released circuitry; the
retained connectome is not a literal living brain.

The sole mutable state was the efficacy fraction of 4,184 identified positive
KC-to-MBON11 slots, bounded to [0.1, 2.0] relative to baseline. RGB mapping,
graph, all nonplastic weights, neural dynamics, modulation, eta=0.001, the
plasticity rule, tonic calibration, and decoding remained fixed. During plastic
training, the predeclared signal supplied an unsigned, bounded, bilateral
additional current to the two PPL101 entries. It was delayed by one original
frame, with frame 0 at zero. This is modeled reinforcement input, not a claim
about biological dopamine or credit assignment.

## Controls and causal protocol

The sensitivity phase used the 982 training frames with frozen memory and
weights, zero teacher current, and identical fast-state initialization. It ran
baseline, an exact duplicate, and manual initial efficacy perturbations of
+0.05 and -0.05 times the preserved learned-minus-baseline direction. The
baseline and duplicate reproduced the complete trace and all 24 terminal arrays
bitwise. Their full fixed-decoder results were:

| Manual diagnostic | Balanced angular MAE, ° | Ordinary MAE, ° | Balanced recall | Initial changed slots | Maximum turn difference, ° |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 2.534691785 | 3.356719500 | 0.330359227 | 0 | 0 |
| duplicate baseline | 2.534691785 | 3.356719500 | 0.330359227 | 0 | 0 |
| +0.05 direction | 2.623370467 | 3.338437287 | 0.339837524 | 1,953 | 6.039927746 |
| -0.05 direction | 2.628995381 | 3.454617691 | 0.308531674 | 1,953 | 4.978851406 |

No ±0.20 or uniform fallback batch ran. Finite response establishes output
sensitivity for these inputs and directions only; it does not establish local
learnability or useful learning.

Each causal arm restored the same pinned baseline before 982-frame training,
then evaluated its resulting checkpoint across the 988 fixed diagnostic frames
with zero teacher current and bitwise-frozen efficacy and memory:

| Arm | Training state and additional bilateral PPL101 current | Eval balanced angular MAE, ° | Ordinary MAE, ° | Balanced recall | Changed slots / mean efficacy |
| --- | --- | ---: | ---: | ---: | ---: |
| A, aligned | plastic, preserved matched schedule | 2.610429266 | 3.202987521 | 0.339108144 | 1,953 / 0.932174493 |
| Z, zero | plastic, always zero | 2.582061701 | 3.212901494 | 0.364199483 | 1,977 / 1.031456016 |
| S, shifted | plastic, duration-stratified half-rotation of A | 2.621400954 | 3.190694615 | 0.318487658 | 2,015 / 1.038343470 |
| F, frozen | frozen, always zero | 2.511255728 | 3.153095522 | 0.325947693 | 0 / 1.000000000 |

Lower primary MAE is better. Thus A-Z = +0.028367565°, A-S = -0.010971688°,
Z-F = +0.070805974°, and S-Z = +0.039339253°. A versus Z and A versus S had
mixed signs across the four prespecified contiguous blocks. Ordinary MAE makes
A appear slightly better than Z, while the primary metric makes it worse; Z
has higher balanced recall than F while the primary metric is worse. These
negative and mixed effects are all retained. No threshold, post-hoc preferred
metric, frame-independent uncertainty estimate, or lane-as-replica inference
was used.

For S, frames 1..981 were rotated only within equal-duration groups: 281 frames
at 285 ticks and 700 frames at 286 ticks. The frame-0-inclusive assigned dose
was exact in both representations: 63,446.97635426394 current-ms requested
float64 and 63,446.97635171088 current-ms float32. It equalizes assigned pulse
dose, not effective dopamine, neural drive, or DAN firing. The corrected
aligned/shifted correlation is 0.031049707009480863 on frames 1..981 and
0.03589795362657386 on all 982 frames including the initial zero. This is a
domain-label correction, not an experimental mismatch.

## Verification, clocks, and limits

The original A trace, its 24 terminal arrays, every frozen efficacy/memory byte,
the duration permutation and dose multisets, every retained nonplastic weight,
and all target/local source, input, build, and reference pins passed their
recorded exact checks. The complete original reference set has 15 roles: three
metadata JSON files, four checkpoints, and eight episode traces. The original
eight anchors remain historical controls, not independent learners. The
independent full audit was 165,400 bytes with SHA-256
`df545aeeced42854f05d2026b95a4e73696e1bea731ca35d74185af853a30217`.
The portable record lists all 53 science artifact roles with their positive byte
counts and SHA-256 values. Recorded pressure boundaries were accepted, with at
least 81% free memory and zero swap growth.

Sensitivity recorded 198.500688334 seconds on `mach_absolute_time`, the active
monotonic counter behind fields named `wall_seconds`; its file-clock approximation
was 326.593853474 seconds and no exact epoch CLI elapsed time was captured.
Causal recorded 519.026763916 seconds epoch elapsed and 386.219725042 seconds
active monotonic time. Both pilots were interrupted by macOS Maintenance Sleep.
`caffeinate -i` was insufficient; causal training added a PID-scoped
`caffeinate -is -w` assertion. These measurements do not support a clean backend
throughput comparison, sleep subtraction, summed lane-clock timing, or a speed
forecast. The clock basis is documented by [Apple's Mach time interface](https://github.com/apple/darwin-xnu/blob/main/osfmk/mach/mach_time.h).

A three-member council used independent opinions, anonymous cross-critique, and
strong-chair synthesis before the run. It corrected duration-naive rotation,
redundant replay interpretation, held-result-derived thresholds, and automatic
parameter changes. Requested command-line model aliases are request settings,
not independently verified effective model identities. Transcripts, failures,
private evidence, raw RGB, NPZ checkpoints, and machine origins remain private.

The outcome is limited to this predeclared diagnostic. Background plasticity
changed many slots, yet that fact does not prove dopamine swamping, association,
or benefit. The next study must be separately designed and bounded, retain the
same mutable slots and fixed decoder, compare ES/SPSA on Metal, use independent
seeds, preserve negative controls, and report results without automatic changes
to baseline, current sign or gain, eta, teacher construction, rule, or decoder.
