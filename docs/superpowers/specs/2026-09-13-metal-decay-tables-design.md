# Metal Host Decay Tables Design

## Purpose and evidence

Continue the portable Metal implementation with one bounded numerical experiment.
The strict-arithmetic investigation measured a one-ULP difference between CPU
`std::exp(float)` and Metal `exp` for the voltage decay at one neural tick.
The retained 40 ms full-graph parity check passes; the 80 ms control fails.
These findings support testing identical decay coefficients. They do not prove
that coefficients account for the complete divergence. Incoming conductance
summation order remains a separate unresolved source.

## Selected approach

Generate the canonical CPU's three 1,024-entry float tables once in Objective-C++
when creating a backend. Store their 3,072 coefficients in one immutable shared
Metal buffer (12,288 bytes). Preserve the CPU's exact expressions:

```cpp
std::exp(-dt*i/20.f)
std::exp(-dt*i/5.f)
std::exp(-dt*i/adaptation_tau)
```

Every `evolve` caller receives the resident table. Voltage, conductance and
adaptation use table entries for elapsed intervals below 1,024 ticks. Refractory
adaptation uses the same adaptation table. Preserve the existing Metal `exp`
fallback at intervals of 1,024 ticks and above. Preserve the separate 100 ms
modulation decay. The experiment guarantees neither fallback equality nor
identical complete neural trajectories.

A small portable C++ header contains only the host table generator. It allows
real native coefficient tests on Linux as well as macOS. No new dependency,
mutable checkpoint field, per-bin table upload, public table-download API or
Python approximation is required. Alternatives using GPU double precision or
changing the canonical CPU coefficients would expand this experiment's scope.

## Interfaces and identity

`doomfly::metal::make_decay_tables(float dt, float adaptation_tau)` returns
`std::array<float, 3072>`, packed voltage, conductance, adaptation in that order.
The existing graph structure already provides both parameters. Its C fields
stay unchanged. Intentionally advance the native/kernel ABI epoch from 3 to 4
because kernel buffer bindings change. Probe, build cache and validation source
identity include the new header; stale binaries and previous validation reports
must fail identity checks. Existing strict compiler flags remain fixed.

## Global constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Retain all 166,700 neurons and 25,582,938 connections in full-graph runs.
- Preserve the canonical CPU kernel, dt 0.1 ms, every neural tick, plasticity,
  sensory mapping, reinforcement and fixed neuron-to-button decoding.
- Preserve parity thresholds: Jaccard >= 0.995, spike fraction <= 0.005,
  rate correlation >= 0.999, within-one-tick fraction >= 0.999, weight rtol 1e-4,
  exact decoder/scientific decisions, structural integrity and repeated state.
- Preserve prior failed controls and strict-arithmetic state epochs in history
  and evidence; numerical tests do not establish learning or biological validity.
- Keep machine paths and origins in ignored configuration. No public deployment
  changes or launch green light are part of this experiment.

## Validation and bounded acceptance

Use actual compiled float expressions to test every host coefficient. Exercise
voltage, conductance, adaptation and refractory cases across intervals 1, 2, 18,
22, 100, 286, 1,023 and 1,024. Zero-edge synthetic graphs isolate arithmetic and
are explicitly numerical fixtures, never replacements for the retained graph.
The fallback boundary is tested without asserting unmeasured equality.

After the running offline cohort finishes naturally, build in a separate owned
M4 Pro checkout. Repeat isolated probes and the unchanged micrograph trace twice.
If a retained golden digest changes, record both measured new digests and the
previous epoch before the original implementation worker updates that test.
Then run unchanged full-graph 40 ms and eight-frame 80 ms checks. Keep any failed
gates failed. Record exact source/build/model identity and no biological or
learning claims. Measure complete-path timing separately after correctness.

The outcome can be an arithmetic improvement with continuing full-graph parity
failure. That is useful evidence for choosing the next Metal implementation step.
