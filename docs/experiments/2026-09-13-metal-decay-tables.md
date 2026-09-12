# Resident CPU Decay Tables on Metal

## Hypothesis and unchanged model

The prior strict-arithmetic control measured one-ULP CPU/Metal exponential
differences and failed the 80 ms full-graph gate. This experiment generates the
canonical CPU float voltage, conductance and adaptation coefficients once in
Objective-C++, then retains a 12,288-byte GPU table. All four evolution kernels
use it for intervals below 1,024 ticks. Long-interval and separate modulation
fallbacks remain unchanged. Kernel ABI advances from 3 to 4; build and validation
identities include the portable table header.

Source implementation: `a68f3c533bf65f42e7d6d1323b1ce67911514ee7`.
Same M4 Pro, macOS 26.6.2, Apple clang 21.0.0, SDK 26.5, Metal 2.4 with strict
flags. Canonical CPU kernel, 0.1 ms timestep, every retained connection,
calibrated dynamics, learning rule, sensory mapping, teacher and decoder stay
fixed. [Invocation](2026-09-13-metal-decay-tables-invocation.md) saved before
execution; full-graph jobs began after the complete offline cohort and transfer
finished naturally. Main dirty checkout and existing workloads were preserved.

## Measured arithmetic and full-graph results

All extracted voltage/conductance/adaptation/refractory-adaptation coefficient
bits match CPU at intervals 1, 2, 18, 22, 100, 286 and 1,023, for adaptation tau
200 and 137 ms. The tested 1,024 fallback case also matches; this does not
establish general fallback equality. Native header tests independently compare
all 3,072 host entries against canonical float expressions for both taus.
Synthetic sleeping states are arithmetic fixtures, not physiological or
threshold-scheduling validation.

| Full retained graph horizon | Event Jaccard | One-tick fraction | CPU/Metal spikes | Result |
|---|---:|---:|---:|---|
| 40 ms | 1.0000000000 | 1.0000000000 | 21,326 / 21,326 | Pass |
| 80 ms | 0.9999233878 | 1.0000000000 | 52,209 / 52,209 | Pass |

Both runs preserve graph integrity, exact fixed decoder decisions, exact plastic
weights, rate correlation 1.0 and bitwise repeated Metal state. Every original
threshold remains fixed. The retained strict 80 ms Jaccard was 0.9421335466.
The table experiment improves this bounded numerical discrepancy substantially;
the small residual 80 ms mismatch remains unresolved. A subsequent warmup
diagnostic fails longer trajectory gates, as reported below.

Two independently constructed Metal micrographs produce identical count/event
hashes and measured state
`6163f146b096db3ba7c086b2b1933977b380e3fec2cd08c6d7c7ce141b16d791`.
Previous strict state
`b5970b9261406420106a6289e86a9ebfc29f463a9c80ba9e724e523011e1a745`
remains in retained evidence. Counts and all three events are unchanged. The
golden test update uses these actual measurements, following its observed
old-digest failure.

Initial M4 arithmetic tests exposed eight fixture calls exceeding the existing
100-step bin limit. The original worker changed only fixture timestamps and
used one valid advance step; the same 19-test M4 suite then passed. Native limits
and dynamics were preserved. Independent task and scoped fix checks approved
the implementation. Final M4 suite at `a5d3145`: 353 passed, 3 skipped and
50 existing dependency warnings in 19.76 seconds, including the measured golden.

## Bounded timing and next execution

Five contiguous matched 40 ms samples: CPU neural median 35.812 ms; Metal neural
median 33.618 ms; Metal GPU median 30.261 ms; complete Metal sample median
38.966 ms. Neural speedup is 1.065x CPU and 1.190x real time. Peak RSS 2.382 GiB,
no swap growth, four encoders and 808 dispatches per sample. Resident full-state
uploads and materialization remain zero. One slower Metal sample is preserved
in the original report; medians are descriptive, not a confidence claim.

The historical benchmark's 2x CPU performance flag remains unmet and its exit 1
is preserved. It does not block continued exploratory Metal training, whose
existing guard checks exact passing numerical/model/build identity. These short
samples establish neither full-episode training throughput nor 1,000x real time.
Proceed with checked offline Metal execution and longer parity/throughput
measurements. The [subsequent 35-frame prefix](2026-09-13-metal-imitation-prefix.md)
has now finished and fails longer parity after its two-second warmup, with
slower measured execution than CPU. Continue the identified arithmetic
experiments before interpreting Metal training as equivalent. CPU offline
training changed weights without improving held-out
turning, and live transfer was seed-dependent. Metal speed cannot establish
better learning per frame or solve those scientific limitations.

Portable reports live in `outputs/doom-learning/metal-decay-tables-20260913/`
and `outputs/doom-learning/metal-decay-timing-20260913/`. Prior failures remain
retained. No successful fly learning, biological validation, public deployment
change or launch green light is claimed.
