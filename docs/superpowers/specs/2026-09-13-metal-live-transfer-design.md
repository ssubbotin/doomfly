# Frozen Metal live Doom transfer design

## Purpose, authority and alternatives

Test whether the seven already frozen efficacy vectors transfer from recorded
RGB fitting to closed-loop Doom. The overall goal remains learning useful Doom
behavior; this intermediate evaluation establishes neither biological learning
nor general game competence. Standing scoped autoapproval covers this design,
plan, private M4 operation and normal owned-branch commits/pushes. Shared main,
the public broadcaster/site and unrelated workloads remain outside this work.

A separate evaluator is recommended. Adapting the public broadcaster also
changes spectator integration and uses a different reinforcement path. Starting
live retraining immediately combines transfer testing with additional fitting.
Keep the present frozen comparison independently interpretable first.

## Exact model and input boundary

Retain MaleCNS v1.0: 166,700 modeled neurons, all 25,582,938 released connections
and all 4,184 existing positive KC to MBON11 slots. Reuse calibrated
`adaptive-centered-v6`, eta=0.001 and the conservative Apple Silicon Metal
backend at W18. No graph pruning, new game policy, native ABI/kernel change,
gain adjustment or mapping retuning. Only the seven saved efficacy vectors from
study `cec94940e370477bf21f0975725ecd2345bf9cba` are installed, between episodes,
using the existing checked resident installer; their SHA/bytes are in the
companion immutable protocol. No candidate is selected from held scores.

Each lane owns its own real ViZDoom instance. Its fresh RGB24 640x480 frame
enters the existing R1-R6/R8 preparation, full neural propagation and the exact
ordered fourteen readouts with fixed BCI DNp20/DNpe017 decoding. Only decoded
turn, forward and attack reach `act`, one Doom tic at a time. No health, reward,
coordinates, labels, demonstrator controls or object telemetry enter this path.
Learning=False, weights_frozen=True and no imposed stimulation apply during
the 2,000 ms dark warmup and every live/padding neural interval. Endogenous DAN
activity may remain nonzero. The existing DamageTraining adapter is bypassed:
its disabled flag currently freezes weights while still allowing damage pulses.

## Tasks, schedule and metrics

Use existing `blue-floor-survival-v1` as the primary transfer question and
`defend_the_center` as separate source-task concordance. Horizon is 2,100 tics
(60 seconds) at 35 Hz, episode start offset zero. Companion protocol fixes four
fresh seeds per task, alternating left/right hazards, all seven vector pins,
the installed ViZDoom 1.3.0 engine/API/IWAD/source-scenario asset pins, and role
rotation. Generated hazard maps and their original texture metadata are hashed
and recorded privately. Never bundle external game assets.

First run one four-lane original-baseline reproducibility wave per task, checking
identical initial pixels, canonical full traces and all 24 final checkpoint
arrays. Then run two groups of four lanes per task/seed: all seven roles and one
explicit baseline filler. Rotate role order by seed ordinal modulo seven and
retain exact role-to-lane mapping. Planned budget is 8 calibration plus 64
evaluation attempts = 72; reserve 8, absolute cap 80. Charge four durably before
every installation, reset or neural rollout, including failures and fillers.
There is no automatic retry, reserve use, adaptive seed choice or winner search.
This is a new independent budget; the completed fitting study stays immutable.

Neural frame boundaries are `round((index+1)*10000/35)`, yielding 285/286 steps
of 0.1 ms. Four lanes always advance. When a game terminates, record its exact
engine tic, death/timeout status, final health and kills. Never respawn it or
apply another control. Its remaining neural intervals use dark RGB, with
`phase=padding` and `applied_action=null`; other games continue independently.
Record requested decoder outputs separately. Wave-final all24 checkpoints are
explicitly after padding, not the brain state at gameplay termination. Padding
counts in simulated brain time and compute, and never in survival or kills.

Primary hazard metric is restricted survival seconds from actual game tics up
to the fixed horizon. Deaths are observed events; living timeout survivors are
right-censored. Damage and final health are diagnostics. Source-task primary
metric is total kills within the same fixed scheduled horizon (including zero
additional kills after death), with restricted survival and damage diagnostics.
Report every role/seed, same-seed baseline differences and variation between the
two fitted replicas per method, separately by task. Four seeds and two fitting
replicas are exploratory; correlated frames are not independent samples. No
cross-task pooling, significance claim or held-derived threshold is permitted.

## Components and failure contract

Pure `live_controls.py` owns exact typed protocol validation, deterministic
wave planning, pinned frozen candidate loading and gameplay-only metrics.
Existing Game gains optional pre-init horizon/start settings preserving defaults;
both Game and SurvivalArena must close a created engine if construction fails.
`live_transfer.py` owns checked installation/reset, independent game lifecycle,
live RGB ticking, trace/checkpoint evidence and exact-source CLI orchestration.
Reuse existing resource registry, pressure history, physical-pin/reference and
checkpoint helpers. Do not copy the recorded replay or activate its teacher path.

Validate committed protocol SHA, exact HEAD, clean model/test sources, trusted
physical source/graph/CPU/native/reference relationships, original configuration,
calibration/readouts, all24 initial state and candidate/engine pins before native
rollouts. Preserve the five known generated CPU metadata files/native products.
Register ownership immediately; constructor errors must close native games.
All evidence outputs are fresh private/ignored directories. Credentials and
machine origins remain solely in ignored configuration. Require 15 GiB free
disk and existing 25%-free/no-growing-swap guards across allocation/waves/release;
use the existing nonblocking GPU lease and AC-scoped `caffeinate -is` only.

Preserve exact primary BaseException identity through engine, checkpoint, JSON,
pressure and lease cleanup. Retain accumulated traces/charged attempts even
when incomplete. Terminal hashes require checked materialization; a failed lane
is explicitly unavailable and receives no repeated download/checkpoint attempt.
Other healthy lane evidence remains independently recoverable. No reset, stale
upload, epoch change or ownership/poison bypass is allowed as failure recovery.
Successful waves verify frozen fullweights/u/w and fixed configuration, required
setup-only upload epochs, no within-rollout sparse updates, and resident storage.

## Verification, evidence and launch boundary

Authentic TDD covers protocol/candidate rejection before work, all18 waves/72
charges/8 fillers, rotation, censoring/padding metrics, optional game settings,
constructor cleanup, divergent terminations, exact clocks, RGB-only decoding,
zero stimulation, frozen vectors, healthy checkpoints and interrupted evidence.
Portable fixtures use the real runner and CPU visual brains where integration
matters; tiny real four-lane Metal and real Doom fixtures prove their boundary.
Old scientific matrices and recorded training are preserved without replay.

After task reviews, deploy reviewed source to only the owned M4 mirror with
independent Git-derived source pins and unchanged core. Run focused native
fixtures, one ordinary full native suite for this new implemented subproject,
then the exact private 72-attempt study. Audit every trace, asset/vector role,
checkpoint schema/frozen bytes, timing, source identity, pressure and budget.
Keep raw data/private origins on durable ignored M4 storage; publish compact
audited outcomes and negative controls on the owned branch only. Separate shared
advance, input preparation, game, host evidence, warmup, live and padding clocks;
neither GPU-only nor 1000x realtime claims follow from elapsed time.

Public scientific/skill/announcement flags remain false for this exploratory
phase. No public stream, launch or How it works publication is part of this
evaluation. Future learning claims require the project's additional sensory,
conditioning, independent learned/frozen/shuffled comparisons and causal controls.

References: [ViZDoom API](https://vizdoom.farama.org/api/cpp/doom_game/),
[ViZDoom 1.3.0 release](https://github.com/Farama-Foundation/ViZDoom/releases/tag/1.3.0),
[prior factual study](../../experiments/2026-09-13-same-slot-optimizer.md),
[spectator requirements](../../doomfly-spectator-experience.md).
