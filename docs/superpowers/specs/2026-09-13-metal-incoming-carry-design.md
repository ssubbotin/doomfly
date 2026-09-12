# Metal Incoming Conductance Carry Contract

## Decision and scope

This separate checked-arithmetic experiment follows the explicit voltage-FMA
epoch. User autoapproval covers required routine design/build/review steps.
The M4 FMA build has exact 40/80 ms events and two-second dark spikes, and
matches the original training prefixes. Remaining frozen/retention/shifted
held-out controls diverge late in the one-second original RGB prefix.

The retained three-neuron control already isolates conductance association:
after settling target conductance 1, incoming weights `16777216` and
`-16777216` yield CPU conductance zero and Metal `0.9801986813545227`. CPU
adds each arriving weight to the existing settled conductance. Metal sums
arrivals from zero, then adds that sum to conductance. Strict float32 addition
makes these associations different even with identical event order.

Recommended change: initialize the existing incoming accumulator from settled
`g[target]`, keep every incoming-position addition in its current order, then
assign the result when a fast arrival was accepted:

```cpp
float conductance=g[target],modulatory=0.0f;
// Existing membership scan, predicates and conductance+=value unchanged.
if(has_fast){g[target]=conductance;active[target]=1;}
```

Matching the full dynamic CPU queue order would be a larger synchronization
and ordering experiment. Changing the CPU reference would invalidate existing
comparisons. Both alternatives stay outside this isolated carry correction.

## Binding constraints

- Minimum macOS 13.0, arm64, unified memory and Apple GPU family 7.
- Metal language `macos-metal2.4`; `-fno-fast-math` and `-ffp-contract=off`.
- Preserve CPU source/flags, explicit voltage FMA, resident decay/fallbacks,
  incoming order, bitmap masking/clearing, modulation and settlement schedules.
- Preserve every retained MaleCNS v1.0 connection, 166,700 neurons, 25,582,938
  edges, dt 0.1 ms and 4,184 existing plastic slots; synthetic tests are labeled.
- Preserve live RGB mapping, reinforcement, host plasticity and fixed decoding;
  targets/observer telemetry never select actions or replace the neural model.
- Preserve Jaccard >= 0.995, spike fraction <= 0.005, rate correlation >= 0.999,
  within-one-tick fraction >= 0.999, weight rtol 1e-4, exact decoder/scientific
  decisions, structural integrity and repeated state.
- Increment checked native/builder arithmetic epoch 5→6. No C layout, binding,
  dispatch, allocation, transfer, checkpoint-schema or new feature dependency.
- Preserve failed experiments, controls and prior measured goldens. Numerical
  tests, changed weights and survival alone establish neither useful nor fly learning.
- No public deployment/launch, unrelated workload interruption, credentials or
  machine origins in tracked files, or bundled game/external research assets.

## Acceptance and experiment continuity

Real native paired tests first reproduce the catastrophic carry mismatch.
Compare CPU/Metal state bits after one valid tick with explicitly queued,
equal-order arrivals and no newly fired neurons. Test ordinary sums, zero carry,
duplicate edges, refractory rejection, exclusively modulatory arrivals and
empty topology. Close both backends in `finally`. No tolerances or subnormal
exceptions enter these normal-valued carry controls; existing fixture-local
underflow tests retain their separately documented contract.

Obtain actual M4 RED before changing source. Then apply only the two carry
assignments and two epoch declarations, verify portable tests and actual M4
controls, measure the original micrograph twice before any golden update,
and independently review source/tests. Root owns target runs and evidence.

The original complete FMA cohort continues at fixed `e9e19c1` in its existing
target checkout. Never change that checkout's source or stop it to test this
experiment. Use a separate owned target for tiny arithmetic controls. Defer
full-graph parity, prefix and performance jobs until the cohort ends naturally;
then use unchanged 40/80 ms gates and original prefix inputs with fresh outputs.
Record remaining incoming-order/modulatory differences and actual timing.
Neither the carry hypothesis nor short parity promises full-episode equivalence,
learning, speedup or 1000x real-time acceleration before measurement.
