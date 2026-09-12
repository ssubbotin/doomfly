# Online Doom demonstrations: acquisition feasibility

## Outcome and scope

Downloaded and checked two complete video/control episodes. Both passed all
frame-index, timestamp, checksum and RGB-decoding checks. `defend_the_center`
is the better candidate for initial turn-target supervision. This is data
feasibility evidence, with a separate local engine calibration. No imitation
trainer was implemented, no connectome weights or decoder changed, and no
learning or biological validation is claimed.

The source is the [GameWAM ViZDoom APPO dataset](https://huggingface.co/datasets/Yunncheng/gamewam-vizdoom),
revision `6cc00d60462885c1d61dd480228af58c3b81b807`. These are selected
agent-generated trajectories, with Apache-2.0 declared in the source card.
They are not human demonstrations or recordings of our custom hazard arena.
The original source README and metadata are preserved beside each local episode.
Downloaded videos, previews and disposable diagnostic scripts remain outside Git.
Machine-specific locations are recorded only in ignored `local-data.jsonl`.

## Measured results

| Episode | Video duration | RGB frames / control rows | Video bytes | Decision blocks |
|---|---:|---:|---:|---:|
| `battle1/000000` | 60.000 s | 2,100 / 2,100 | 55,920,610 | 525 |
| `defend_the_center/000000` | 28.057 s | 982 / 982 | 23,638,400 | 246 |

Both videos are 640×480 H.264/yuv420p at 35 FPS. Each presentation timestamp
matches `frame_index / 35` exactly. Control timestamps match float32-rounded
tick times; maximum rounding error is 1.853 microseconds and 0.927 microseconds,
respectively. Each chosen action repeats for four ticks; the final center block
contains two ticks. Preserve every intermediate RGB frame and original index.

All frames decoded to RGB24 without errors, resizing or cropping. Byte-identical
adjacent decoded frames: zero. Compression noise makes this insufficient to
prove independent simulator renders. HUD, crosshair and weapon pixels remain;
source colorimetry is unspecified, and the uncompressed source RGB is unavailable.

Portable evidence, including source and decoded-pixel hashes, is in
`outputs/doom-learning/online-demonstrations-20260912/`:

- `battle1-episode-000000.json`
- `defend_the_center-episode-000000.json`
- `native-turn-calibration.json`

## Action meaning and compatibility

The source order is `[shoot, move_forward, move_backward, strafe_left,
strafe_right, speed, turn_left, turn_right, turn_delta]`. The inspected
[collector](https://github.com/yunncheng/GameWAM/blob/e3a3d29a1eef4bc1f51ec161652bcc158f908321/scripts/datasets/vizdoom/appo/collect_appo_data.py)
stores the rendered frame and action before advancing the environment.
The original collection commit and engine version are absent from the downloaded
episode metadata. Matching timestamps establishes container alignment, rather
than an independently verified reproduction of the physical trajectories.

`battle1` contains 1,164 strafing rows, 1,660 speed-button rows, 96 backward rows
and 732 turns exceeding our ±6 degree limit. These counts overlap. Preserve this
episode as a compatibility control; no action dimensions were dropped or clamped.

The center episode contains 618 left-turn, 268 right-turn and 292 shoot rows,
with no strafing, speed or movement actions. DOOMFLY's fixed BCI decoder couples
forward motion and shooting through DNpe017 activity. Stationary expert shooting
therefore remains an unresolved full-action matching problem.

## Separate local turning calibration

A fixed-input assay used installed ViZDoom 1.3.0 and existing Freedoom `basic`
assets, with ten idle warmup ticks. Binary-turn and predetermined delta-turn
arms followed the same 47-tick schedule, including direction changes and idle
resets. Angle telemetry was recorded only as an outcome.

Left turns map to negative delta commands; right turns map to positive commands.
The first five held ticks use 1.7578125 degrees; subsequent held ticks use
3.515625 degrees. Direction changes retain the held-turn counter; idle resets it.
The [engine reference](https://github.com/Farama-Foundation/ViZDoom/blob/96e9adc1db6d8dd37fe921d96ccda1fec7ae903b/src/vizdoom/src/g_game.cpp)
supports this acceleration rule. Maximum binary-versus-delta outcome difference:
**0.0 degrees**. Maximum angle-versus-formula difference: `8.30005e-8` degrees.
All requested deltas fit the existing ±6 degree limit. This local assay does not
reproduce or validate the external collector's original engine settings.

An observer-only source-column check found every `POSITION_X`, `POSITION_Y` and
`ANGLE` entry padded in both center episodes (982 and 988 rows). Source angle
telemetry therefore cannot independently verify the original engine's turn
conversion. `observer-column-availability.json` preserves this limit; these
columns do not enter the production episode reader or learner.

## Preserved failure and next step

The initial center probe ran during its download and failed with
`moov atom not found`. After successful download completion, both files matched
published SHA-256 values; the unmodified full diagnostic passed. No repair or
source modification occurred. Local engine probes emitted PipeWire warnings
while completing successfully; those warnings are preserved locally.

Next, implement a separate offline episode reader and declared turn-target
training objective. Preserve episode continuity, raw controls and the fixed
decoder; keep pose/health columns observer-only. Use whole-episode holdouts,
frozen-weight and shuffled-label controls, then measure transfer to our hazard
arena. Human DSDA demonstrations remain a later replay/rendering source requiring
a compatible engine and legally available IWAD. Data checks alone establish no
learning advantage, scientific launch readiness or change to the public stream.

## Follow-up: checked reader and reserved evaluation episode

The production reader now checks published hashes, original action/timing
schema, policy repeats, metadata and every rational video timestamp before
training. Complete RGB iteration independently matches the original center
episode's prior pixel digest. A distinct center episode `000001` is reserved
for evaluation: 988 frames, 28.229 seconds, 21,997,229 video bytes, and different
video/Parquet hashes from the training episode. It contains 624 left, 176 idle
and 188 right target ticks. Training episode `000000` contains 618 left,
96 idle and 268 right ticks. This class imbalance requires per-direction recall
and constant-label baselines when reporting imitation scores.

`checked-reader-episodes.json` records complete production decoding and calibrated
turn summaries. `full-graph-reference-check.json` records matching source hashes,
valid CSR dimensions/ranges and the actual configured 166,700-neuron,
25,582,938-edge model with 4,184 existing plastic slots. Reference input transfer
did not change the tracked baseline manifest, graph topology or decoder.

The local hazard environment also passed its prescribed-action control:
stationary died at tick 128; crossing survived the 420-tick cap on safe ground.
This checks the environment, with no neural policy or learning tested. ViZDoom
again emitted PipeWire connection warnings. Full neural training and actual
teacher-free transfer remain subsequent evidence requirements.

The M4 Pro reader suite passed all 68 checks without warnings, and both copied
episodes passed complete RGB decoding. Default Linux FFmpeg 8 and arm64 FFmpeg 9
produce different RGB digests. On the first frame, channel differences are at
most two uint8 levels; decoded YUV bytes match exactly. Disabling CPU instruction
optimizations on both platforms makes that frame's RGB bytes identical. This
localizes the tested difference to CPU-optimized YUV-to-RGB conversion, with
no bound established for later frames. Production flags remain unchanged.
All cohort arms use one decoder and retain exact frame hashes and tool version;
cross-machine default-pixel equality and original RGB colorimetry remain
unresolved. See `cross-platform-rgb-decoder.json`.
