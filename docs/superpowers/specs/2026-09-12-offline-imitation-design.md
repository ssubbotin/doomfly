# Offline imitation reinforcement design

## Objective and approval

Advance the full goal, training the retained MaleCNS experiment to play Doom,
by introducing verified demonstration-driven reinforcement. User autoapproval
of required goal steps applies to this design and its implementation. This
stage must produce actual full-graph training/evaluation runs, rather than
declare data acquisition or numerical tests to be learning.

## Approaches and decision

1. Use decoded expert turn error as a declared scalar reinforcement input to
   the existing centered plasticity rule. This preserves the current model and
   tests whether its existing memory connections support useful steering.
2. Optimize the same plastic slots through population search. This requires
   many complete recurrent rollouts and is a later alternative if local
   reinforcement produces no held-out effect.
3. Fit a new visual encoder or game decoder. This changes the sensory hypotheses
   or fixed interface and is outside this implementation.

Choose approach 1. Keep turn-only supervision explicitly identified as an
initial bootstrap toward the complete game task. Shooting and motion continue
through the original fixed decoder. Their coupling remains unresolved for
full-action imitation; no independent attack policy is added.

## Global constraints

- Retain all 166,700 neurons and 25,582,938 released retained edges.
- Only the 4,184 identified existing KC-to-MBON11 plastic slots may change during training.
- Keep RGB input, sensory mapping, neural propagation, teacher reinforcement, plasticity and fixed DNp20/DNpe017 decoding separate.
- Do not pass pose, health, reward, target coordinates or source controls into the neural action decoder.
- Use the existing 0.1 ms neural integration and every 35 Hz demonstration frame.
- Store downloaded videos, game imagery, credentials and machine-specific origins outside Git or in ignored configuration.
- Preserve failed experiments and third-party notices; report numerical and behavioral evidence separately from biological validation.
- Do not deploy, publish a launch green light or change the public stream in this implementation.

## Episode boundary

Create `doom_learning_v6/demonstrations.py`. Read a local artifact directory
containing `SOURCE-README.md`, `source-info.json`, `episode-metadata.json`,
`validation-report.json`, and `episode_NNNNNN.mp4/.parquet` matching metadata.
The initial source is GameWAM ViZDoom at revision
`6cc00d60462885c1d61dd480228af58c3b81b807`.

Independently verify file hashes against the recorded published checksums,
contiguous indices, a single episode, finite values, binary/continuous action
ranges, metadata, source dimensions, per-frame presentation timestamps and
float32-rounded tick timestamps. Prior passing booleans alone cannot authorize
an episode. Reject malformed data before neural training. Read action/timing
columns only; never expose pose/health through a frame sample.

Keep the nine canonical controls unchanged. `turn_targets()` accepts only
the two defend scenarios with all movement, strafe, speed and continuous-turn
dimensions zero; reject other scenarios explicitly. Convert binary left/right
to negative/positive delta targets, respectively. Increment the shared held-turn
counter while either turn is down, including direction changes; reset on idle.
Use 320 yaw units for held counts below 6 and 640 thereafter, scaled by
360/65536 degrees. This is the measured local ViZDoom 1.3.0 calibration, with
unresolved original-collector engine provenance.

Stream RGB24 frames through FFmpeg without cropping, scaling, resampling or
wall pacing. Frame iteration supports explicit diagnostic prefix limits,
maintains original indices, and reaps only its own decoder process on close.
Identity includes dataset/revision/scenario/episode and original video/control
hashes. Reject train/evaluation overlaps including renamed copies of a video.

## Reinforcement and training boundary

Create `doom_learning_v6/imitation.py`. For each original RGB frame, advance
the full neural model and decode spike counts using existing `NeuralControls`
in BCI mode. Only afterward compute
`error = min(abs(decoded_turn - target_turn) / 6, 1)`.
The next frame receives a constant `4 * error` mV-equivalent current at the
identified PPL101 cells. The first frame receives zero imposed teacher input.
Evaluation receives zero imposed teacher input throughout. The scalar feedback
and its one-tick delay are chosen engineering hypotheses, with no biological
claim. Keep the existing centered anti-Hebbian update and calibration unchanged.

Use cumulative rounding: frame t ends at `round((t+1)*10000/35)` neural
steps relative to episode origin. Reset fast state and decoder filters between
episodes; preserve learned u/w/efficacies across training epochs. Apply identical
2,000 ms dark equilibration in each arm and record its separate brain/wall time.

Run plastic, weights-frozen and circularly shifted-label arms. Shift only turn
targets with a nonzero recorded offset; preserve every RGB frame and raw
control. Label distributions match; imposed reinforcement dose can differ
because feedback depends on the student's output, and must be reported.
Restore baseline memory before each arm, save a full learned checkpoint before
evaluation, and restore it independently for each complete held-out episode.
Include teacher-free erased-memory and five-second dark-retention evaluations
for the plastic arm. Check all nonplastic weights and CSR endpoints unchanged.

## Evidence and interfaces

`OfflineEpisode(directory)` exposes `frame_count`, `width`, `height`,
`identity`, `actions`, `turn_targets()` and `iter_frames(limit=None)`.
Samples expose `index`, `timestamp`, `rgb` and original `action`.

`episode(brain, data, readouts, *, training, frozen, target_shift=0,
limit=None, warmup_ms=2000)` returns a portable summary with trace. It never
instantiates a game controller or executes expert controls. Pure helpers
`teacher_error(decoded_turn, target_turn)` and
`score_turns(predictions, targets)` return bounded error and MAE/per-class
direction recall respectively. Direction classification uses a declared
0.1-degree dead zone; include class support and constant-label baselines.

CLI accepts repeated `--train` and `--eval` directories, fresh `--out`,
`--epochs`, `--eta`, optional diagnostic `--max-frames`, and
`--backend cpu|metal`. Metal uses the existing artifact validation gate and
records its tested horizon. CPU is the numerical reference, not a replacement
of the Metal development direction. Paths belong only in ignored input records.

Record protocol, source/model signatures, causal teacher current, complete raw
label/source indices, student controls, KC/DAN/MBON counts, frame/spike hashes,
imposed dose, memory summaries and wall/brain timing. Preserve partial results
and failure state. No runner may set learning/announcement success from changed
weights or one favorable episode. Hazard-arena transfer remains a required
subsequent behavioral test toward the full goal.

## Verification

Use TDD with actual Parquet/video fixtures and existing neural components.
Cover corruption, reordered labels, invalid ranges, frame/count/PTS mismatch,
unsupported controls, early generator close, teacher causality, freeze behavior,
unaltered decoder, exact cumulative neural time and leakage rejection.
Full-graph runs use verified original data, separate complete evaluation episodes
and all three controls. Smoke prefixes are execution evidence only. Preserve
earlier physiological failures and numerical Metal limitations.
