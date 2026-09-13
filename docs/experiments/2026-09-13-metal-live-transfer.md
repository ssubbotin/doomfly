# Frozen Metal Doom Transfer: Private Evaluation Record

## Status and scope

This completed private study evaluated seven pre-existing frozen vector roles on
one retained model. It used 18 completed four-lane waves and charged all 72
attempts. Eight of the 80 allowed attempts remained unused. The compact,
source-identified record is
[`results.json`](../../outputs/doom-learning/metal-live-transfer-m4pro/results.json).

All three scientific status flags are false: `learning_demonstrated`,
`doom_skill_demonstrated`, and `announcement_ready`. No role was selected, no
significance claim was calculated, and this study gives no public-launch
approval.

## Exact evaluation path

The model retains MaleCNS v1.0's 166,700 modeled neurons, all 25,582,938
released connections between retained entries, and 4,184 existing KC-to-MBON11
plastic slots. It used `adaptive-centered-v6`, `eta=0.001`, Metal window width
18, and source `41d20761f5a3b6a7139a219c9fc6cb72b223b8a3`.

At each live 35 Hz engine frame, RGB pixels supplied sensory neurons. Modeled
neural propagation then ran through the retained graph and the existing fixed
DNp20/DNpe017 decoding converted activity to game controls. RGB input,
propagation, reinforcement, plasticity, and decoding remained distinct
operations. This is a frozen evaluation: weights, `memory_u`, and `memory_w`
were frozen, learning was disabled, and imposed reinforcement current was zero.
The two-second dark warmup has setup-only epochs. It did not train a vector.

ViZDoom 1.3.0 clamps requested episode tic 0 to tic 1. Gameplay clocks use
observed one-tic differences from that effective origin. The pre-control health
was 80 in `blue-floor-survival-v1` and 100 in `defend_the_center`. Each row
records its actual terminal state. After death, the game received no respawn or
further game action; its resident neural lane continued on dark RGB to the
fixed 2,100-tic horizon. This padding is explicitly excluded from gameplay
metrics.

Two four-lane all-baseline calibration waves preceded the evaluation waves.
Their frozen/runtime checks passed: blue-floor seed 202609135 used setup epoch
3 and `defend_the_center` seed 202609131 used setup epoch 6. Every lane was
frozen. The full calibration schedule, lane roles, wave clocks, asset pins, 72
trace/episode pins, and 72 all-24-array checkpoint pins are in the compact
record.

## Per-seed outcomes

All 32 blue-floor rows died at tic 128, with restricted survival
3.657142857142857 s, zero kills, 80.0 damage, and final health 0.0. They are
therefore flat relative to their same-seed baselines on each reported metric.
The evaluated role/lane schedule, including every filler, was:

| Seed | Role | Wave | Lane |
| ---: | --- | ---: | ---: |
| 202609135 | baseline | 2 | 0 |
| 202609135 | spsa-200000201 | 2 | 1 |
| 202609135 | spsa-200000202 | 2 | 2 |
| 202609135 | gaussian_es-200000201 | 2 | 3 |
| 202609135 | gaussian_es-200000202 | 3 | 0 |
| 202609135 | random-200000301 | 3 | 1 |
| 202609135 | random-200000302 | 3 | 2 |
| 202609135 | baseline-filler | 3 | 3 |
| 202609136 | spsa-200000201 | 4 | 0 |
| 202609136 | spsa-200000202 | 4 | 1 |
| 202609136 | gaussian_es-200000201 | 4 | 2 |
| 202609136 | gaussian_es-200000202 | 4 | 3 |
| 202609136 | random-200000301 | 5 | 0 |
| 202609136 | random-200000302 | 5 | 1 |
| 202609136 | baseline | 5 | 2 |
| 202609136 | baseline-filler | 5 | 3 |
| 202609137 | spsa-200000202 | 6 | 0 |
| 202609137 | gaussian_es-200000201 | 6 | 1 |
| 202609137 | gaussian_es-200000202 | 6 | 2 |
| 202609137 | random-200000301 | 6 | 3 |
| 202609137 | random-200000302 | 7 | 0 |
| 202609137 | baseline | 7 | 1 |
| 202609137 | spsa-200000201 | 7 | 2 |
| 202609137 | baseline-filler | 7 | 3 |
| 202609138 | gaussian_es-200000201 | 8 | 0 |
| 202609138 | gaussian_es-200000202 | 8 | 1 |
| 202609138 | random-200000301 | 8 | 2 |
| 202609138 | random-200000302 | 8 | 3 |
| 202609138 | baseline | 9 | 0 |
| 202609138 | spsa-200000201 | 9 | 1 |
| 202609138 | spsa-200000202 | 9 | 2 |
| 202609138 | baseline-filler | 9 | 3 |

The following `defend_the_center` table includes every evaluated lane. `S` is
restricted survival seconds, `K` is fixed-horizon kills, `D` is total damage,
and `H` is final health. Each condition died, so every survival value is an
uncensored observed terminal time. Same-seed baseline deltas for all four
metrics, including zero-delta baselines, are preserved exactly in
`paired_contrasts` in the compact record. The roles show adverse, mixed and
flat results across seeds; no aggregate selection was made.

| Seed | Role | Wave | Lane | Tics | S | K | D | H |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 202609131 | baseline | 10 | 0 | 435 | 12.428571428571429 | 2 | 114.0 | -14.0 |
| 202609131 | spsa-200000201 | 10 | 1 | 343 | 9.8 | 1 | 102.0 | -2.0 |
| 202609131 | spsa-200000202 | 10 | 2 | 303 | 8.657142857142857 | 0 | 100.0 | 0.0 |
| 202609131 | gaussian_es-200000201 | 10 | 3 | 369 | 10.542857142857143 | 1 | 100.0 | 0.0 |
| 202609131 | gaussian_es-200000202 | 11 | 0 | 311 | 8.885714285714286 | 1 | 102.0 | -2.0 |
| 202609131 | random-200000301 | 11 | 1 | 263 | 7.514285714285714 | 0 | 100.0 | 0.0 |
| 202609131 | random-200000302 | 11 | 2 | 363 | 10.371428571428572 | 2 | 102.0 | -2.0 |
| 202609131 | baseline-filler | 11 | 3 | 435 | 12.428571428571429 | 2 | 114.0 | -14.0 |
| 202609132 | spsa-200000201 | 12 | 0 | 383 | 10.942857142857143 | 2 | 132.0 | -32.0 |
| 202609132 | spsa-200000202 | 12 | 1 | 267 | 7.628571428571429 | 0 | 132.0 | -32.0 |
| 202609132 | gaussian_es-200000201 | 12 | 2 | 375 | 10.714285714285714 | 2 | 100.0 | 0.0 |
| 202609132 | gaussian_es-200000202 | 12 | 3 | 347 | 9.914285714285715 | 1 | 114.0 | -14.0 |
| 202609132 | random-200000301 | 13 | 0 | 407 | 11.628571428571428 | 2 | 100.0 | 0.0 |
| 202609132 | random-200000302 | 13 | 1 | 331 | 9.457142857142857 | 1 | 114.0 | -14.0 |
| 202609132 | baseline | 13 | 2 | 295 | 8.428571428571429 | 1 | 114.0 | -14.0 |
| 202609132 | baseline-filler | 13 | 3 | 295 | 8.428571428571429 | 1 | 114.0 | -14.0 |
| 202609133 | spsa-200000202 | 14 | 0 | 331 | 9.457142857142857 | 1 | 108.0 | -8.0 |
| 202609133 | gaussian_es-200000201 | 14 | 1 | 323 | 9.228571428571428 | 1 | 108.0 | -8.0 |
| 202609133 | gaussian_es-200000202 | 14 | 2 | 341 | 9.742857142857142 | 1 | 108.0 | -8.0 |
| 202609133 | random-200000301 | 14 | 3 | 323 | 9.228571428571428 | 1 | 108.0 | -8.0 |
| 202609133 | random-200000302 | 15 | 0 | 415 | 11.857142857142858 | 2 | 102.0 | -2.0 |
| 202609133 | baseline | 15 | 1 | 451 | 12.885714285714286 | 1 | 100.0 | 0.0 |
| 202609133 | spsa-200000201 | 15 | 2 | 295 | 8.428571428571429 | 1 | 108.0 | -8.0 |
| 202609133 | baseline-filler | 15 | 3 | 451 | 12.885714285714286 | 1 | 100.0 | 0.0 |
| 202609134 | gaussian_es-200000201 | 16 | 0 | 277 | 7.914285714285715 | 0 | 104.0 | -4.0 |
| 202609134 | gaussian_es-200000202 | 16 | 1 | 307 | 8.771428571428572 | 1 | 102.0 | -2.0 |
| 202609134 | random-200000301 | 16 | 2 | 275 | 7.857142857142857 | 0 | 104.0 | -4.0 |
| 202609134 | random-200000302 | 16 | 3 | 317 | 9.057142857142857 | 1 | 104.0 | -4.0 |
| 202609134 | baseline | 17 | 0 | 393 | 11.228571428571428 | 2 | 104.0 | -4.0 |
| 202609134 | spsa-200000201 | 17 | 1 | 287 | 8.2 | 0 | 104.0 | -4.0 |
| 202609134 | spsa-200000202 | 17 | 2 | 375 | 10.714285714285714 | 0 | 114.0 | -14.0 |
| 202609134 | baseline-filler | 17 | 3 | 393 | 11.228571428571428 | 2 | 104.0 | -4.0 |

The filler is a charged baseline control. It has no candidate contrast because
it repeats the same frozen baseline role. The public compact JSON contains all
64 evaluated rows and all 56 defined paired contrasts, including the four
numeric deltas for every non-filler row. The survival task and source-scenario
kills remain separate results and were never pooled.

A bounded, read-only control summary found differing nonzero applied controls
across blue-floor vector roles while their death metrics stayed identical. It
does not establish spatial progress, identical trajectories, an inactive
decoder, or angular success. Pixels and positions were not stored in this
evaluator.

## Vector identities and perturbations

The baseline is the unmodified frozen vector. Each tested vector pin is 33,600
bytes. RMS, mean-absolute, and maximum values are componentwise displacement
from that baseline.

| Role | RMS | Mean absolute | Maximum absolute | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.0 | 0.0 | 0.0 | 16add0823da69b593476d07e5ebe47c21fef467b73bdd16867ec7cc653d37ec6 |
| spsa-200000201 | 0.021547926242020562 | 0.019301687873840775 | 0.02913126124515908 | eeb94c077b6c7249933d03449c541448ed6baf2f8a39bc5facdf7551cec8e688 |
| spsa-200000202 | 0.019737532022490397 | 0.0197375320224904 | 0.0197375320224904 | 107420bc9df47fc2cf17087968163054990e5fabc1bdbdd5db9e235a4b227787 |
| gaussian_es-200000201 | 0.004243583494150423 | 0.003365366250029064 | 0.0197375320224904 | e34d171abc2badf4237f297f32d70ad37ed8cb4bf3267aa29882db25a05cfef1 |
| gaussian_es-200000202 | 0.007245028352855146 | 0.005792492977521764 | 0.026307141642617937 | 517452cb6852426833d98d69ccf0c4bc45a2b8320e5c91112ee6c018345ab717 |
| random-200000301 | 0.057591098341291364 | 0.049789712667099494 | 0.0999942309568399 | ce8d59979c50112ce78e9deefc4346b607baac4fc76473f44afbffb82ae7ca5a |
| random-200000302 | 0.057286263541031036 | 0.04941819621826449 | 0.09996296735131516 | 3c541d868633d9e73591b43ba419a62ce086211eb730146825b1fb331bd1911f |

The original random vectors are unselected movement controls. Their fitting
compute and perturbation magnitudes differ from the other vectors, so this
comparison makes no magnitude-matched causal claim. Replica variation is
reported by the separate `...201` and `...202` rows in each environment.

## Assets, identities, validation, and preservation

The audited model hashes cover the nonplastic weights, graph pointers,
postsynaptic indices, neuron IDs, and plastic-slot map. Three connectome inputs,
15 reference files, four native products, five CPU metadata/binary pairs, and
five game assets were independently rehashed. The compact record retains every
relative path, byte count, and SHA-256 value. It records the blue-floor WAD pin
and the ViZDoom/FreeDoom/`defend_the_center` pins per wave.

The 72 terminal all-24 checkpoints and four initial checkpoints provide all 76
artifact pins. A read-only initial audit compared every lane's 126,308,136-byte
checkpoint (SHA-256
`6c265beae5fffecc287592d4c64a9da67e7d9577afe0e2f57fc2bb9e85a658cd`) to
the pinned original and checked all 24 dtypes, shapes, finite-array conditions,
and fixed metadata. The compact record retains its full 24-array schema and
all 76 terminal/initial pins. Raw private evidence occupies 12,037,474,943
bytes; 139,511,361,536 bytes were free afterward.
Measured free memory reached at least 86%, with 0.0 MiB maximum swap use.
Each four-lane wave recorded 413,006,712 shared resident bytes and
737,265,700 mutable resident bytes.
Raw traces, checkpoints, pixels, runtime scripts, and machine-specific origins
remain ignored; no RGB video was stored.

Final native-full validation at the same source completed with exit 0, 1,647
passes, three skips, 50 warnings, no failures, and no full-connectome study
attempts. Its printed pytest time was 45.71 s. The final native-boundary proof
completed with exit 0 and two passes. The ordinary portable suite belongs to
initial source `04c42dd4b4519001a6929d4d2155afc00bcd3f93`: 868 passes, 761
skips, and 50 warnings. It is retained as a separate portable proof. The final
bounded covering record reports 473 passes and 36 skips.

The original audit helper failed with exit 1 because it compared a checkpoint
metadata dictionary to a separate reference-model identity digest. The retained
v02 helper corrected that exact dictionary comparison and independently checked
the canonical JSON SHA-256. It exited 0 without a scientific rerun or relaxed
guard. Both helper hashes and the v02 artifact pin are in `results.json`.

The lease-release proof used read-only nonblocking exclusive acquisition only
after observed process exit; its 15-byte lease pin remained unchanged. The
separate fitting preservation proof rehashed 120 previous traces and 124
previous checkpoints against published pins. Its scope is exactly those 244
artifacts and their manifest pins. It is not a manifest of every original study
file. The seven frozen candidate vectors were independently verified here.

## Clocks and execution envelope

The exact scientific CLI exited 0. Observations were recorded from
2026-09-13 15:58:42 UTC to 17:58:03 UTC, and those timestamps are observations,
not exact process boundaries. POSIX `time -p` around the CLI reported real
7,145.58 s, user 1,064.82 s, and system 164.11 s. Its scope includes preflight,
allocation, evaluation, evidence I/O, and cleanup; it excludes transport and
later independent audits.

Each of the 18 wave clock records is counted once. Across waves, RGB-loop wall
time was 6,948.429691460013 s, batch RGB wall time was 6,826.549423803415 s,
batch-native time was 6,221.885869318474 s, live-batch time was
788.0500067437388 s, dark-padding-only batch time was 5,950.1459986426635 s,
and the wave-envelope total was 7,118.9286855440005 s. GPU time
6,070.893555539238 s and wait time 6,148.595224731 s have their own shared
native scopes. These clocks overlap and must not be summed, treated as one
outer elapsed time, or described as a GPU-only speed. No 1000x multiplier is
used.

## Limits and reproducibility

Live RGB pixels were checked through their fresh source path, hashes, fixtures,
and reproducibility rather than reconstructed independently from stored pixels.
The durable final records and source contracts check setup ordering and charging;
the final ledger alone cannot replay every write. Four seeds and two fitting
replicas are exploratory.

The retained connectome, sensory mapping, dynamics, fixed decoder, reinforcement
placement, and control/angular mappings are engineering hypotheses. This
experiment contains no human or independent hazard demonstration, no spatial
trajectory or position record, and no biological validation. Longer survival,
changed weights, numerical validation, and a control difference do not establish
fly learning or useful Doom skill.

Exact private paths, credentials, and raw-output origins remain only in ignored
configuration. To reproduce on an authorized machine, supply
`DOOMFLY_METAL_LIVE_TRANSFER_COMMAND`,
`DOOMFLY_METAL_LIVE_TRANSFER_EXPECTED_PINS`, and
`DOOMFLY_METAL_LIVE_TRANSFER_OUTPUT_ROOT` from that ignored configuration, then
use the committed protocol and expected pins at source
`41d20761f5a3b6a7139a219c9fc6cb72b223b8a3`.
