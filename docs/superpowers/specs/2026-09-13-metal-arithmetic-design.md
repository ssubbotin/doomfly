# CPU-Compatible Metal Voltage Arithmetic

## Decision and evidence

User autoapproval covers this required, isolated Metal experiment. The previous
resident-table build passes 40/80 ms parity but fails a two-second dark warmup
and original 35-frame prefix. At the first 10 ms dark bin, 7,118 voltages differ
by one float32 step while conductances, adaptation and counts match. The actual
M4 CPU `memory_advance` binary uses two fused multiply-add instructions in the
rest/current relaxation and a third in the adaptation correction. Strict Metal
currently computes these terms with separate multiply/add operations.

Recommended approach: explicitly reproduce those CPU voltage operations in
Metal. Preserve `-ffp-contract=off`, which prevents unintended contractions;
the deliberate `fma` operations become part of a checked arithmetic epoch.
Changing CPU compiler flags would change the established reference. Broad
relaxed Metal arithmetic would permit additional uncontrolled differences.
Both alternatives remain outside this experiment.

## Scope and arithmetic contract

Only change `evolve` voltage association to match the measured arm64 reference:

```cpp
float voltage = fma(v[i] - rest[i], a, rest[i]);
voltage = fma(current, 1.0f - a, voltage);
v[i] = voltage + g[i] * (a - b) / 3.0f;
// Inside the existing positive-adaptation branch:
float correction = (-adaptation[i] * adaptation_tau) / (adaptation_tau - 20.0f);
v[i] = fma(correction, c - a, v[i]);
```

Retain conductance multiply/divide association and adaptation coefficient/update
expressions. No new dispatch, transfer, buffer, feature dependency or C structure.
Increment native/builder ABI arithmetic epoch 4→5; source/build/validation hashes
must reject earlier evidence automatically. Preserve the previous measured
micrograph digest and await actual new measurements before changing its test.

Incoming event order/association, evolution on exclusively modulatory arrivals
and fallback exponential/division differences remain independent hypotheses.
Both backends already settle all fast state at each observation boundary.
This change promises no long-horizon parity or speedup before measurements.

## Binding constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Preserve canonical CPU source/flags, dt 0.1 ms, neural ticks, calibration,
  every retained MaleCNS v1.0 connection and 4,184 identified plastic slots.
- Preserve sensory mapping, reinforcement, host plasticity and fixed decoding;
  targets/observer telemetry never choose actions.
- Preserve parity thresholds: Jaccard >= 0.995, spike fraction <= 0.005,
  rate correlation >= 0.999, within-one-tick fraction >= 0.999, weight rtol 1e-4,
  exact decoder/scientific decisions, structural integrity and repeated state.
- Preserve all previous failed controls and measured epochs. Numerical tests,
  changed weights and survival alone establish neither useful nor fly learning.
- No public deployment, launch green light, unrelated workload interruption,
  credentials/origins in tracked files or bundled game/research assets.

## Acceptance and execution

Real paired M4 CPU/Metal arithmetic tests must reproduce the voltage mismatch
before implementation and pass after it. Use the existing four-neuron zero-edge
helper, nontrivial drive/conductance/adaptation/rest inputs and one or 200 valid
100-tick bins. Verify counts/cursors and float32 state bits without tolerances
that hide the measured voltage mismatch. These artificial controls provide no
biological validation. Preserve failure output if another arithmetic mechanism
also appears; diagnose it separately rather than weakening the test.

Run portable focused and full Linux tests, actual M4 fixtures, independent
micrograph repetitions and unchanged 40/80 ms full-graph gates. Re-run the
preserved warmup/first-original-frame diagnostic, recording when divergence
moves and the longer metrics. Retain any remaining long-horizon failure and
continue the next identified Metal hypothesis. Complete Metal training remains
exploratory until its actual numerical and useful-learning evidence is known.

The implementation worker owns kernel/epoch/tests. The controller owns M4
execution, portable evidence and experimental documentation. Keep source fixed
while a target job runs. Required design/review checkpoints are autoapproved.
