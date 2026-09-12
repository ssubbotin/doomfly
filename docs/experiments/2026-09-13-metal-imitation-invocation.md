# Checked Offline Metal Training Invocation

Prepared before execution. Use the reviewed host-table backend and its exact
passing 80 ms numerical report. Full M4 software suite at `a5d3145` passed
353 tests (3 platform skips, 50 existing dependency warnings). Run in its
isolated target checkout; keep source/build/model identity fixed while a job runs.
Do not modify the live service, stop another workload or weaken a parity gate.

Use ignored `DOOMFLY_DEMO_TRAIN`, `DOOMFLY_DEMO_EVAL`,
`DOOMFLY_METAL_PARITY_REPORT` and fresh output path configuration. Inputs are
the same pinned original GameWAM center episodes 0/1 and M4 FFmpeg 9.0.1 used
in the CPU cohort. Keep `DOOM_KERNEL_PATH`, `OPENBLAS_NUM_THREADS=1` and a PATH
including FFmpeg/FFprobe plus standard system sbin tools. Use job-scoped
`caffeinate -is` on Apple Silicon.

## Execution prefix

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_METAL_IMITATION_PREFIX_OUT" --epochs 1 --eta .001 \
  --backend metal --metal-validation "$DOOMFLY_METAL_PARITY_REPORT" \
  --max-frames 35
```

The prefix exercises all plastic/frozen/shifted training and teacher-free
held-out, retention and erasure phases with the full retained graph. Its
`complete` flag remains false because original source episodes are longer.
Check every causal teacher/timing boundary, zero teacher in evaluation,
invariant hashes and fixed-decoder outputs against CPU reference evidence.
A full runtime reset/warmup exceeds the report's tested 80 ms horizon. Therefore
this opt-in exploratory execution does not establish long-horizon CPU/Metal
parity or useful learning. Record the actual validation horizon without extending
it through prose or a passing prefix.

## Complete cohort after prefix execution checks

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_METAL_IMITATION_FULL_OUT" --epochs 1 --eta .001 \
  --backend metal --metal-validation "$DOOMFLY_METAL_PARITY_REPORT"
```

Do not overlap the prefix with full training or performance tests. Use another
fresh output and preserve all controls/failures. Measure original brain time,
wall time and complete-episode scores independently. The CPU cohort failed its
useful turn-learning comparison; Metal acceleration does not automatically
improve sample efficiency. Neither a passing numerical report nor this invocation
certifies biological learning, successful Doom training or launch readiness.
