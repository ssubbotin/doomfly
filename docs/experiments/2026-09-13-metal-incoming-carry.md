# Metal Incoming Carry: Exact Local Control, Failed Longer Parity

## Isolated arithmetic contract

Source `fba0c93112c59b5f61317b4026aee0b92978fc41`, arithmetic epoch 6,
seeds each existing incoming accumulator from settled conductance and assigns
accepted fast input once. Every incoming addition/order, CPU source/flags,
explicit voltage FMA, strict Metal 2.4 flags, resident coefficients, modulation,
dispatch, RGB mapping, plasticity and fixed decoder stay unchanged. The model
retains 166,700 MaleCNS v1.0 neurons, 25,582,938 edges and 4,184 plastic slots.
[Saved invocation](2026-09-13-metal-incoming-carry-invocation.md).

## Test-first and short retained gates

The test-only M4 source fails catastrophic and duplicate carry cases: CPU
conductance zero versus Metal 0.9801986813545227; five other controls pass.
After the two-assignment correction, all seven strict native cases pass.
Focused M4 checks give 67 passes/one platform skip without warnings; full M4
gives 378 passes/three skips. Linux gives 313 passes/68 Apple skips. Both full
suites retain 50 existing dependency warnings. Independent task review approves
the narrow correction; numerical tests provide no biological validation.

Two fresh tiny Metal micrographs retain state `65524d57...`, counts and all
three events, so the existing golden is unchanged. Fresh retained
[40 ms](../../outputs/doom-learning/metal-incoming-carry-20260913/parity/parity-40ms/report.json)
and [80 ms](../../outputs/doom-learning/metal-incoming-carry-20260913/parity/parity-80ms/report.json)
gates pass with exact events, controls, weights and repeated Metal state.

## Failed warmup and original RGB prefix

The [200-bin diagnostic](../../outputs/doom-learning/metal-incoming-carry-20260913/diagnostic/results.json)
finds state differences at the third 10 ms bin and first spike differences at
1.477 seconds. Two-second dark Jaccard is 0.7375 and within-one-tick fraction
0.8630, below unchanged gates. First RGB turn is +1.04365 on CPU and -1.04365
on Metal. The preceding FMA-only candidate had exact dark spikes.

All eight 35-frame phases finish with identical original pixels/controls/timing,
unchanged invariants, teacher-free frozen evaluation and exact erased/frozen
repetition. [Comparison](../../outputs/doom-learning/metal-incoming-carry-prefix-20260913/comparison-v2.json)
records zero exact CPU/Metal fixed-control frames in every phase. Plastic
training changes 1,493 slots versus CPU 1,687; shifted training 1,661 versus
1,701. These prefixes remain incomplete relative to the original episodes.
Phase walls are 8.28–9.19 seconds versus CPU 4.84–5.12; RGB neural execution
takes 3.63–4.55 wall seconds per brain second, without speedup.

Remaining independently identified differences include dynamic CPU queue versus
stable incoming order, exclusively modulatory settlement, and per-edge
modulation arithmetic. Their full-network causal contribution remains unresolved.
Keep this candidate isolated as a failed longer-parity experiment; pursue
instrumented ordering/timing work before promoting it for CPU-equivalent
training. Source/artifact checks, complete traces and checkpoints are preserved.
No useful-learning, public deployment or launch approval follows.
