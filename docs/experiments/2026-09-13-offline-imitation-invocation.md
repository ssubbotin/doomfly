# Offline Imitation Reference Invocation

Prepared before full-source execution. Use a reviewed checkout and record its
exact commit, source/model hashes and decoder version in the resulting evidence.
Set input/output paths only through ignored machine configuration. Each output
directory must be fresh. Do not change an existing checkout or stop workloads.

Build the baseline native library before importing neural modules in a fresh
checkout. An ignored external output avoids modifying tracked build metadata:

```sh
python -m doom.build_kernel --output "$DOOMFLY_NATIVE_KERNEL"
export DOOM_KERNEL_PATH="$DOOMFLY_NATIVE_KERNEL"
```

## Diagnostic prefix

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_IMITATION_SMOKE_OUT" \
  --epochs 1 --eta .001 --backend cpu --max-frames 35
```

This prefix checks execution only. It cannot establish complete-episode
throughput, held-out gameplay, successful learning or biological validation.

## Complete initial cohort

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.imitation \
  --train "$DOOMFLY_DEMO_TRAIN" --eval "$DOOMFLY_DEMO_EVAL" \
  --out "$DOOMFLY_IMITATION_FULL_OUT" \
  --epochs 1 --eta .001 --backend cpu
```

Training source: pinned GameWAM `defend_the_center/000000`, 982 RGB frames.
Evaluation source: distinct `defend_the_center/000001`, 988 RGB frames.
Dataset revision: `6cc00d60462885c1d61dd480228af58c3b81b807`.
The checked artifact identities include both original video/control hashes;
the runner independently rejects overlap before training.

Keep all 166,700 neurons, 25,582,938 retained connections and existing fixed
decoding. Run plastic, frozen and shifted-label arms from independent baseline
restoration, plus teacher-free erasure and five-second dark retention. The
default training label shift is 491 frames; raw source controls remain intact.
Apply error-driven PPL101 current on the next frame only. Include identical
2,000 ms dark warmup per episode, recorded separately from demonstration time.

On Apple Silicon, prefix the commands with job-scoped `caffeinate -is` and make
FFmpeg/FFprobe available in the job's PATH. Use the same decoder for every arm
and record exact RGB hashes; cross-platform default RGB equality is unresolved.
The CPU cohort supplies numerical reference evidence for continued Metal
development. Existing Metal parity gates and physiological failures remain
unchanged. Preserve partial results/checkpoints on failure; success flags remain
false. Actual teacher-free hazard-arena transfer and replicated learning evidence
remain additional requirements toward the full goal. No public deployment or
launch green light is authorized by these commands.
