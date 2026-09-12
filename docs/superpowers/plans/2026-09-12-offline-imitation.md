# Offline Imitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run demonstration-driven reinforcement and teacher-free tests on the retained full MaleCNS experiment toward learning Doom.

**Architecture:** Independently checked local episodes stream pixels into the existing visual neural model. The fixed decoder's turn error schedules scalar reinforcement on the next tick. Controls compare plastic, frozen and shifted-label memory with complete held-out episodes.

**Tech Stack:** Python, NumPy, PyArrow, FFmpeg/FFprobe, existing C++ reference and Objective-C++/Metal neural backends.

**Spec:** `docs/superpowers/specs/2026-09-12-offline-imitation-design.md`

## Global Constraints

- Retain all 166,700 neurons and 25,582,938 released retained edges.
- Only the 4,184 identified existing KC-to-MBON11 plastic slots may change during training.
- Keep RGB input, sensory mapping, neural propagation, teacher reinforcement, plasticity and fixed DNp20/DNpe017 decoding separate.
- Do not pass pose, health, reward, target coordinates or source controls into the neural action decoder.
- Use the existing 0.1 ms neural integration and every 35 Hz demonstration frame.
- Store downloaded videos, game imagery, credentials and machine-specific origins outside Git or in ignored configuration.
- Preserve failed experiments and third-party notices; report numerical and behavioral evidence separately from biological validation.
- Do not deploy, publish a launch green light or change the public stream in this implementation.

### Task 1: Checked offline episode boundary

**Files:** Create `doom_learning_v6/demonstrations.py` and `tests/test_doom_demonstrations.py`.

**Interfaces:** Produce `OfflineEpisode(directory)` with integer `frame_count`,
`width`, `height`, portable `identity` dictionary, readonly float32 `actions`,
`turn_targets()` returning float64 deltas, and generator
`iter_frames(limit=None)` yielding samples with `index`, `timestamp`,
`rgb` and readonly original `action`. Consume the artifact filenames and
schema in the spec. Select original action/timing columns only.

- [ ] Write red tests using real small Parquet files and FFmpeg-generated 35-FPS videos, with declared source dimensions matching the fixture. Include hand-derived binary turn targets:

```python
def test_binary_turn_acceleration_keeps_history_on_direction_change():
    data = checked_episode_fixture(turns=[-1] * 8 + [1] * 4 + [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(data.turn_targets(), [
        -1.7578125, -1.7578125, -1.7578125, -1.7578125, -1.7578125,
        -3.515625, -3.515625, -3.515625,
        3.515625, 3.515625, 3.515625, 3.515625,
        0.0, 0.0, 0.0, 0.0,
        1.7578125, 1.7578125, 1.7578125, 1.7578125,
    ])
```

  The test-only fixture writes source schema/metadata/checksum records around
  genuine generated files; it must not construct expected turns with production
  helpers. Also cover altered file bytes, reordered action names/indices,
  invalid flags and continuous values, metadata/range/timestamp/PTS mismatches,
  unsupported battle conversion, empty data and partial final repeat blocks.

- [ ] Run red: `OPENBLAS_NUM_THREADS=1 /home/sergey/doomfly/.venv-neural/bin/python -m pytest tests/test_doom_demonstrations.py -q`. Preserve the expected missing-feature failure in the task report.
- [ ] Implement the checked boundary fresh from the tests. Validate recorded hashes dynamically, not only prior booleans. Require the declared GameWAM revision, exactly one matching episode, contiguous original indices, proper source dimensions, binary first eight action values and final value in [-1,1]. Check all decoded presentation timestamps as exact rational tick times and Parquet timestamps as their float32 rounding.
- [ ] Implement the binary target conversion with the measured rule:

```python
held = 0
targets = []
for action in actions:
    held = held + 1 if action[6] or action[7] else 0
    units = 320 if held < 6 else 640
    targets.append((action[7] - action[6]) * units * 360 / 65536)
```

  Reject movement/strafe/speed/continuous-turn actions and non-defend scenarios
  from target conversion. Preserve them in the raw reader. Check image dimensions
  against schema and cap unreasonable dimensions before allocating decoder frames.

- [ ] Implement RGB iteration via an argument-list FFmpeg subprocess, RGB24, passthrough timing and no transforms. Support positive prefix limits; close pipes and reap only the owned process on early close/error. Test a first-frame prefix against real decoding and ensure malformed counts are rejected before iteration.
- [ ] Run focused green and the complete Python suite once; record baseline dependency warnings separately from new test output.
- [ ] Commit only these two owned files as Sergey Subbotin; do not stage concurrent documentation or ignored data.

### Task 2: Causal imitation reinforcement runner

**Files:** Create `doom_learning_v6/imitation.py` and `tests/test_doom_imitation.py`; add measured experiment documentation after actual runs.

**Interfaces:** Consume Task 1's exact episode/sample API, existing
`calibrated_brain(eta, backend)`, `NeuralControls(readouts, mode='bci')`
and full-state `checkpoint/restore`. Produce
`teacher_error(decoded_turn, target_turn)`,
`score_turns(predictions, targets)`,
`episode(brain, data, readouts, *, training, frozen, target_shift=0,
limit=None, warmup_ms=2000)`, `build_parser()` and `run(args)`.

- [ ] Write red tests for bounded errors, invalid/nonfinite targets, independent direction class supports and constant baselines:

```python
def test_teacher_error_is_bounded_and_symmetric():
    assert teacher_error(0, -3) == 0.5
    assert teacher_error(0, 3) == 0.5
    assert teacher_error(-6, 6) == 1
    assert teacher_error(2, 2) == 0
```

  Use the existing numerical MemoryBrain fixture in test utilities to check
  changing only selected weights, freeze behavior, pending-current delay and
  exact 10,000 neural steps across 35 frames. For expensive full-graph boundaries,
  a narrow test double may observe the actual runner's inputs; all decoding and
  scoring remain real. Verify same student controls for different labels during
  teacher-free evaluation; labels must never modify the first frame's action.

- [ ] Run red: `OPENBLAS_NUM_THREADS=1 /home/sergey/doomfly/.venv-neural/bin/python -m pytest tests/test_doom_imitation.py -q`; record expected failure.
- [ ] Implement the pure teacher:

```python
def teacher_error(decoded_turn, target_turn):
    if not math.isfinite(decoded_turn) or not math.isfinite(target_turn):
        raise ValueError('Finite turn values required')
    return min(abs(decoded_turn - target_turn) / 6.0, 1.0)
```

  Score MAE and per-present-class direction recall with a 0.1-degree dead zone,
  including supports and constant-label baselines. Reject empty/invalid score inputs.

- [ ] Implement per-episode execution in this causal order:

```python
steps = round((index + 1) * 10000 / 35) - (brain.cursor - origin)
counts, kernel_seconds = brain.rgb_step(
    sample.rgb, steps * 0.1, learning=training,
    stimulation=(brain.circuit['dan'], pending_current) if training and pending_current else None,
)
action = controls.decode(counts, steps * 0.0001)
error = teacher_error(action['turn'], targets[index])
pending_current = 4 * error if training else 0
```

  Fast-state reset keeps learned memory; weights-frozen mode is explicit.
  Run fixed dark equilibration before the origin. Close the frame generator
  even on failure. Record raw source actions, shifted teacher indices, imposed
  input dose, spike/frame/memory hashes and separate brain/wall timing.

- [ ] Implement fresh-output CLI and cohort execution. Validate all inputs before training, reject shared video/control hashes across train/evaluation directories, and reject nonpositive epochs/limits or incompatible episode targets. Record path-free public identities in protocol and paths only in ignored `inputs.jsonl`. Use existing Metal artifact/model preflight gates for opt-in Metal, with no new precision claim.
- [ ] Run plastic/frozen/shifted-label arms, restoring initial memory per arm and learned checkpoints before each independent teacher-free held-out episode. Shift by a nonzero recorded offset, preserve raw frames and controls, and report differing feedback doses. Run erased-memory and five-second dark-retention controls for the plastic arm. Compare every nonplastic weight and CSR endpoint against the start of the run.
- [ ] Run focused green and complete suite; record counts. Use the actual 166,700-neuron graph for execution smoke and then complete-episode training with a distinct checked center episode. No tiny fixture or prefix result can establish the thread goal.
- [ ] Save measured results, limitations and exact portable protocol; keep goal active unless complete held-out gameplay, controls, retention and required launch/scientific evidence genuinely establish success.
- [ ] Commit only owned runner/tests and its measured documentation as Sergey Subbotin.
