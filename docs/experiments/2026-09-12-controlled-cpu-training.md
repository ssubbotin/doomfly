# Controlled CPU Training Cohort

## Purpose

Run actual pixel-to-neural-to-Doom training with the unchanged
`adaptive-centered-v6` candidate, then compare its learned state with frozen and
timing-shuffled controls. This is an exploratory cohort after failed sensory and
conditioning gates. It provides behavioral and throughput evidence, including
failure; it cannot independently establish validated fly learning.

The training source is commit `2dd86b5a63d0a3ca88595ae1a1ecad4ad261c5fc`.
The runner writes its protocol and source snapshot before its first episode.
The CPU reference is selected explicitly. The validated shared-graph C++
executor is present in this release, but this cohort uses the existing serial
runner. Metal's preserved 80 ms parity failure remains unresolved.

## Fixed Protocol

```sh
OPENBLAS_NUM_THREADS=1 python -u -m doom_learning_v6.survival \
  --backend cpu --out outputs/doom-learning/cpu-controlled-training-20260912 \
  --seeds 42051,42052,42053 --eval-seeds 62051,62052,62053 \
  --seconds 30 --train-episodes 2 --eta 0.001
```

- Three training replicas, with separate plastic, frozen, and shuffled arms.
- Two training episodes per arm. The second training seed is the replica seed
  plus 1,000. All training and evaluation seeds are distinct.
- Three identical unseen evaluation starts per arm. Evaluation freezes weights
  and disables imposed reinforcement.
- Each plastic arm also receives black-input, five-second passive retention,
  and memory-erasure checks. The planned total is 54 episodes.
- Each episode resets neural state, visual filters, rate traces, and controller
  filters. Learned efficacy state persists across its two training episodes.
  Two seconds of dark equilibration precede each episode.
- The hazard is evaluated by ViZDoom. Actual damage schedules the declared
  200 ms PPL101 stimulus. Timing-shuffled controls reuse shifted donor exposure;
  actual delivered dose must be checked, including early recipient deaths.

All 166,700 neurons and 25,582,938 released edges remain. Actual RGB enters the
existing sensory mapping, propagation retains 0.1 ms steps, and centered
plasticity updates in bins of at most 10 ms on 4,184 existing connections. Fixed
DNp20/DNpe017 decoding selects actions. Health schedules reinforcement;
coordinates and other telemetry remain observer-only.

## Evidence and Decisions

Preserve `protocol.json`, `provenance.json`, complete episode traces, learned
efficacies, `results.json`, and `analysis.json`. Check frozen-weight identity,
neural/game clock alignment, actual shuffled dose, memory-erasure trace identity,
and every predefined outcome. Report censored durations and differences by
training replica. Thousands of game frames are correlated observations.

Measure complete episode wall time separately from native propagation and
one-time construction. The next acceleration milestone is a persistent
multi-trajectory training scheduler sharing the immutable graph while retaining
the same sensory, reinforcement, learning, and decoding boundaries. Further
training scale is judged against controls; weight changes alone are insufficient.

## Completed Results

All 54 episodes completed, recording 8,160 game tics. None was censored. All nine
mechanical checks passed, including six exact shuffled training doses and full
first-start memory-erasure trace identity in every replica. Analysis used
commit `8dbf9c7`; its updated sampling metadata does not alter training sources.

Mean observed held-out duration across the three starts, in seconds:

| Training replica | Plastic | Frozen | Shuffled |
|---:|---:|---:|---:|
| 42051 | 3.657 | 3.657 | 3.657 |
| 42052 | 3.657 | 3.657 | 3.657 |
| 42053 | 5.486 | 3.657 | 3.657 |

The only beneficial comparison was replica `42053`, start `62053`: plastic
survived 320 tics (9.143 seconds), both controls 128 tics (3.657 seconds).
Eight of nine comparisons showed no benefit. Black input, five-second retention,
and erasure were predeclared for start `62051`, where no benefit occurred;
they do not establish the mechanism or retention of the isolated `62053` result.
Physiological recovery and conditioning gates remain unresolved. This cohort
does not establish general survival learning or biological validation.

The first replica (`42051`) completed all 18 planned episodes. Its three unseen
evaluation starts each survived 128 game tics (3.657 seconds) in every arm.
None reached the safe sector. After two training episodes, plastic weights
changed on 1,885 connections, shuffled weights on 1,897, and frozen weights on
zero. Both shuffled training episodes delivered the donor's exact 600 ms dose.
Memory erasure restored the complete first-start frozen input, spike, action,
health, position, and neural-clock trace. This replica shows weight-dependent
movement differences without an observed survival benefit.

Replica `42052` initially survived 352 tics (10.057 seconds) during training,
entered the safe sector for 231 observed tics, then returned to the hazard and
died. Its three learned-weight held-out evaluations each died at 128 tics.
The third replica's second shuffled training episode survived 672 tics
(19.2 seconds), versus 224 tics (6.4 seconds) in its plastic arm, then died.
Longer training survival therefore does not identify the intended association.

A one-second live-process stack sample found 628 of 669 main-thread samples
inside the native `memory_advance` path. This short diagnostic supports the
episode timers identifying propagation as the dominant cost; it is not a
whole-study profile or a speedup measurement. The raw machine-specific sample
stays in ignored evidence. The experiment protocol remained unchanged.

Complete JSON traces, source/data hashes, compact learned-efficacy artifacts,
`analysis.json`, and `cohort-summary.json` are preserved under the output path
above. Dataset files, source snapshots, binaries, workbooks, papers, and game
assets are excluded from the committed cohort. Reproduce mechanical analysis:

```sh
python -m doom_learning_v6.analyze outputs/doom-learning/cpu-controlled-training-20260912
```

## Host Sleep and Timing Scope

During this cohort, the AC-powered host repeatedly entered maintenance sleep
despite its initial `caffeinate -i` assertion. A clock probe measured 99.119
seconds of real elapsed time against 30.016 seconds from `time.perf_counter()`;
the host's sleep/wake timestamps account for the gap. Recorded episode
`wall_seconds` and native timers use that suspend-excluding monotonic clock.
For this sleep-affected run, their ratios describe awake-clock processing
throughput. They must not be presented as complete elapsed-time throughput.

At 09:39:37 UTC, an additional `caffeinate -s -w <training-pid>` assertion was
attached to the existing process. It is job-scoped and releases on process exit;
permanent power settings and the neural/game protocol remain unchanged.
A subsequent probe spanning the host's 45-second maintenance-sleep boundary
measured 55.002 seconds on both clocks, with no intervening sleep. This is a
verified runtime improvement over the observed interval, not a kernel speedup.
Both training and added assertion exited normally; `PreventSystemSleep` returned
to zero afterward.

The cohort processed 341.141 brain seconds including episode warmups in
1,042.014 recorded awake seconds (0.327x); native propagation took 92.8% of that
interval. Whole-process elapsed time from launch to final results write was
1,848.551 seconds (0.185x), including construction, standalone retention and
system suspend. The summary retains the external timestamps and matched
177-file transfer digest. These measurements concern the serial training
runner, separately from the short four-lane propagation benchmark.
