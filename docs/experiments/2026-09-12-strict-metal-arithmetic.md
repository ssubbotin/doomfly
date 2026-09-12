# Strict Metal Arithmetic Diagnostic

## Scope and Build

Executed source `bf9f327dd6820dc3a34b09b1ac686ac9f0a00853` on the M4 Pro.
The Metal compiler now uses `-fno-fast-math -ffp-contract=off` with
`-std=macos-metal2.4`. Cache reuse requires the complete compilation
configuration, builder hash, source hashes and actual artifact hashes.
Objective-C++ compilation flags, shaders, CPU dynamics, fixed decoder and
learning rule are unchanged. Portable validation/timing metadata retains the
new build configuration.

The full-graph controls retain MaleCNS v1.0: 166,700 neurons, 25,582,938 released
connections and 4,184 plastic slots. These are bounded numerical diagnostics,
without ViZDoom episodes or public deployment. Physiology and learning remain
unvalidated.

## Full-Graph Parity

Both horizons start from identical calibrated baseline checkpoints, use 0.1 ms
integration and the existing 10 ms learning bins. The 40 ms trace is black,
blue, green, white; the 80 ms trace adds left blue, right blue, vertical,
horizontal. White and right blue receive the existing +4 PPL101 stimulation.
The exact frames and source identities are recorded in each report.

| Horizon | Spike Jaccard | Within one tick | CPU / Metal spikes | Numerical gates |
|---:|---:|---:|---:|---|
| 40 ms | 0.9973775405 | 1.0000000000 | 21,326 / 21,326 | Pass |
| 80 ms | 0.9421335466 | 0.9879204395 | 52,209 / 52,237 | Fail: Jaccard and timing |

At both horizons, decoder decisions, reported scientific-gate outcomes and
plastic weights match; structural integrity passes and independent Metal
repetitions are bitwise identical. Matching numerical outcome flags do not
establish biological validation or successful learning. At 80 ms, total-spike
fractional difference is
0.0005360185 and rate correlation is 0.9998518526. The timing failure remains
essentially the same as the preserved default-build result (Jaccard
0.9421871804). Strict compilation has therefore not established full parity.

## Isolated Arithmetic

A synthetic one-neuron, zero-edge control retains rest/initial voltage -52 mV,
drive 12 and zero conductance/adaptation. After one step, CPU voltage is
-51.94015121459961 and Metal is -51.940147399902344: one voltage ULP.
After 100 steps, CPU is -47.278480529785156 and Metal is
-47.2781982421875. The absolute difference, 0.00028228759765625 mV, slightly
exceeds the earlier default-build control.

A second synthetic control extracts the voltage-decay coefficient through
exact power-of-two scaling of deliberately inactive state, with rest 0 and
initial voltage -64. At a one-step elapsed interval, CPU coefficient is
0.9950124621391296 and Metal is 0.9950124025344849, one float32 coefficient ULP
lower. The same direction occurs at intervals 2, 18 and 22; interval 1023 differs
by one ULP in the other direction. Intervals 100, 286 and 1024 agree. All eight
CPU results match float64 host `math.exp` rounded to float32, which is an
observational reference rather than a formal correct-rounding guarantee.

This isolates a decay-coefficient disagreement under strict compilation.
The synthetic inactive state is expressly an arithmetic probe; larger intervals
cross a model threshold while parked and do not validate threshold scheduling.
These controls do not crop or replace the retained experiment.

## Repeatability and Golden State

The initial M4 Pro focused test run produced 50 passes, one skip and one
failure: the retained micrograph state digest still expected the old fast-math
epoch. Two fresh strict executions then produced the same new state digest:
`b5970b9261406420106a6289e86a9ebfc29f463a9c80ba9e724e523011e1a745`.
The three spike events and count/event digests remain unchanged. The test now
names and retains this measured strict-math epoch; the old digest remains in
`protocol.json` and Git history. Updating that epoch does not change any
full-graph tolerance or turn the 80 ms failure into a pass.

Micrograph CPU/Metal voltage, conductance and adaptation differ within existing
tolerances; queue/active storage ordering also differs. This is repeatability
within Metal, rather than complete CPU/Metal state identity.

## Implementation Verification

After independent review, the measured golden/evidence update was committed as
`7192e9c2d98ed923cec8b8958c80a26c1b6457ac`. An isolated, owned M4 Pro worktree
was advanced to that commit without altering the existing main checkout or
older experiment worktree. The complete suite then passed, including the new
strict-math golden:

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

- Linux: 204 passed, 35 skipped, 50 dependency warnings, 20.52 seconds.
- M4 Pro: 236 passed, three skipped, 50 dependency warnings, 15.35 seconds;
  invoked under job-scoped `caffeinate -is`.

These implementation tests do not replace the separately recorded full-graph
80 ms stress failure. No numerical or scientific acceptance threshold changed.
Both invocation blocks match their saved pre-execution scripts; portable source,
configuration, artifact and transfer hashes were independently checked.

## Observational Timing

Because 80 ms parity failed, the guarded benchmark was not invoked. A separate
profile called unchanged sampling helpers on five matched 40 ms traces per
backend: one warmup each, then contiguous CPU and Metal blocks, with the same
checkpoint restored before every sample. These observations cannot serve as a
passing training-validation report. Existing training/benchmark guards remain
unchanged.

| Median per 40 ms trace | CPU | Metal |
|---|---:|---:|
| Neural call | 35.250 ms | 33.740 ms |
| Complete sampled RGB/rule path | 37.916 ms | 39.014 ms |
| Plasticity | 0.418 ms | 3.034 ms |

Metal GPU time is 30.591 ms, with 808 kernel dispatches and four encoders.
Sparse weight publication accounts for 2.623 ms of the Metal plasticity path.
Samples have zero full-state upload/materialization bytes. Peak process RSS is
2.250 GiB; swap use is zero before and after. The small neural-call advantage
does not produce a complete-path speedup. These traces exclude construction and
the game, and do not measure complete training throughput or demonstrate 1000x
real-time execution.

The complete five-phase diagnostic took 18.616 seconds, starting after imports,
the commit guard and output creation, before provenance/build/control setup.
Job-scoped sleep assertions were released at process completion.

## Evidence and Next Metal Experiment

The [exact five-phase invocation](2026-09-12-strict-metal-invocation.md) and
[exact coefficient invocation](2026-09-12-metal-decay-coefficient-invocation.md)
were saved before execution. Untouched portable JSON evidence is under:

- `outputs/doom-learning/metal-strict-diagnostic-20260912/`: nine files,
  74,794 bytes; canonical transfer-manifest SHA256
  `00fa20fd167d3da55c5c5691707fde7f7d5ca95c454e0851944357f9271da6a1`.
- `outputs/doom-learning/metal-strict-decay-coefficients-20260912/report.json`:
  6,775 bytes; file SHA256
  `529be01505fdd67f62cd395d9ae859f080f672e1384c825ac563d35392e031ba`.

Strict metallib SHA256 is
`14cb82045b80ab668cce2659fb16477c587e79469664d43e85623d7cabae8092`;
native Metal library SHA256 is
`c6c059cceea479a19809628d4d636ffa5841f416f63a98863c1b41aa8991d44f`.
Actual configuration, CPU hashes, sources and input hashes are retained in the
reports. Machine origins, generated archives/binaries, dependency checkouts and
game assets are excluded.

The next proposed numerical experiment is to upload immutable CPU-generated
decay tables for Metal to consume, matching the CPU's existing 1,024-entry
tables. This is not implemented or established as a fix. Incoming-weight
accumulation order remains a separate discrepancy after synaptic arrivals,
as documented in the [first-divergence diagnostic](2026-09-12-metal-first-divergence.md).
No game-training or launch approval follows from these measurements.
