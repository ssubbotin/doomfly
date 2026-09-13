# Learning causal pilot design

## Purpose and approval

Run a private mechanism experiment on the validated Metal temporal-window
executor. Separate background plasticity, additional teacher-current effects
and selected timing dependence before expanding the current imitation rule.
Standing user autoapproval covers this scoped experiment, implementation,
verification and ordinary owned-branch publication. Public launch remains outside
scope. The architectural training campaign is decomposed into this first
self-contained subproject; future optimization or rule changes need their own
designs.

## Evidence and council decision

The [completed backend experiment](../../experiments/2026-09-13-metal-temporal-windows.md)
measured a 1.7404× offline throughput improvement with scoped bitwise identity.
The learned diagnostic evaluation had worse ordinary turn MAE (3.20299° versus
3.15310° frozen), slightly better balanced direction recall, and 1,953 changed
plastic slots. This establishes no useful learning.

Read-only original-trace analysis found teacher-free DAN means of 108.72/99.90 Hz
for frozen baseline evaluation and 143.19/149.53 Hz for learned evaluation,
against the chosen 20.09 Hz dark baseline. Elevated activity motivates a control;
the opposing instantaneous/filtered terms can cancel at steady rates. Dopamine
swamping is an unresolved hypothesis.

A three-member cross-model council used independent opinions, anonymous
cross-critique and strongest-model synthesis. All recommended causal controls
before a population search or demonstration expansion. Reviews corrected simple
frame rotations (285/286-tick intervals differ), redundant closed-loop/replay
interpretation, held-result-derived thresholds and automatic baseline changes.
Full working transcripts and failed CLI invocations remain preserved privately.

## Fixed scientific constraints

- Retain every released connection between retained MaleCNS v1.0 entries.
- Only the 4,184 identified positive KC→MBON11 slots may change, within
  baseline-relative efficacy fractions [0.1, 2.0].
- Keep RGB sensory mapping, neural dynamics, modulation, eta, plasticity rule,
  tonic calibration and fixed neuron-to-button decoding unchanged.
- Targets may describe the preserved delayed reinforcement sequence and score
  outputs; they must never select actions or change the decoder.
- Use opt-in window_ticks=18; default window 0 and public/live runtime stay fixed.
- Serialize GPU ownership; inspect pressure; stop no unrelated workload.
- Preserve failures, controls, full traces, checkpoints and private provenance.
- Store credentials and machine-specific origins only in ignored configuration.
- No biological, general Doom-learning or public-launch claim follows this pair.

## Stage 1: training-only sensitivity

Use seed 200000065, all 982 original frames and recorded timing. From identical
fast-state initialization run baseline, duplicate baseline and ±0.05 times a
direction normalized from the preserved learned-minus-baseline efficacy-fraction
displacement. Initialize memory u/w consistently with each manual efficacy;
freeze memory and weights, impose no teacher current. Label all perturbations
manual diagnostics. Require bitwise replica agreement on neural hashes, fixed
decoded outputs and memory, excluding wall clocks.

Report continuous turn/rate/class effects. If both perturbations have exactly
identical decoded outputs to baseline, permit one additional four-lane batch:
±0.20 along that direction and uniform fractions 0.95/1.05. Compare against the
already measured baseline. A negative result concerns these directions and
inputs only; finite perturbations do not establish local learnability. Proceed
to the causal comparison regardless.

## Stage 2: four-arm causal comparison

Restore the pinned original baseline checkpoint independently in every lane.
Run all original training frames with these assignments:

| Arm | Memory/weights | Bilateral additional PPL101 current |
| --- | --- | --- |
| A: aligned | Plastic | Preserved matched-teacher schedule |
| Z: zero | Plastic | Always zero |
| S: shifted | Plastic | Duration-stratified half-rotation of A |
| F: frozen | Frozen | Always zero |

The preserved generator already includes the one-frame feedback delay. Keep
frame 0 zero; do not delay again or recompute feedback in replay arms. Its full
trace/checkpoint can replace a redundant generator rerun only after source,
configuration, graph, input and initialization checks. A must reproduce original
neural hashes, decoded outputs, memory records and all 24 terminal checkpoint
arrays bitwise; a mismatch invalidates interpretation pending investigation.

For S, partition frames 1–981 by exact neural tick count. Rotate original indices
by floor(group_size/2) inside each duration group. Preserve requested float64
and applied float32 amplitude multisets, including frame 0. Calculate assigned
dose with math.fsum of amplitude*ticks*0.1 in original destination order; require
exact equality for both representations. Actual groups are 281×285 ticks and
700×286 ticks; displacements span 490–494 frames. This preserves assigned
pulse-current dose, with remaining temporal structure and possible residual
association. It does not equalize effective dopamine, summed neural drive or
DAN firing. Report mapping and residual target/current correlations.

Save every complete learned checkpoint. Restore it before evaluating once on
all 988 frames of seed 200000071, with frozen memory/weights, zero teacher current,
and identical resets of neural, adaptation, decoder and rate-trace states.
This historically inspected pair is a fixed diagnostic evaluation sequence,
not fresh untouched validation data. Evaluation must leave u/w/weights unchanged.

## Scoring and interpretation

Primary: class-balanced angular MAE, averaging MAE equally over present target
classes using the existing 0.1° dead zone. Report ordinary MAE, balanced direction
recall, class counts, confusion matrix, per-class errors and four fixed contiguous
frame blocks. Also report paired output differences, DAN/KC/MBON and decoder
rates, efficacy distribution/bounds and assigned dose checks. Frames, blocks
and resident lanes are not independent learner replications. Constants remain
descriptive scoring baselines only.

Z−F measures background-plasticity consequences; A−Z added-current effects;
A−S selected timing dependence at matched assigned dose; S−Z misaligned-dose
effects. Timing-specific improvement motivates unchanged independent-seed
replication. Similar A/S effects leave label-specific association unsupported.
Large Z changes establish background plasticity without proving swamping.
Harmful/mixed results end expansion of this unchanged pilot. With output
sensitivity, choose a separately designed same-slot optimization or teacher-rule
diagnostic. With null probes, inspect intact-network MBON11-to-decoder response
before expensive search. No baseline/sign/gain/eta change occurs automatically.

## Implementation and acceptance

Two modules: pure schedule/score/efficacy helpers and a full-episode resident
executor runner. Add focused portable mathematical tests and actual native
synthetic RGB replay/freeze/reset tests on M4. Verify missing APIs fail before
implementation. Run the full suite on portable and M4 sources. Pin build/input
bytes before and after GPU runs; report setup/warmup and brain/wall time separately.
Publish only reviewed compact scientific metadata and this experiment's report;
raw images, NPZ, transcripts and machine paths remain ignored/private.
