# Portable Apple Metal Backend

## Supported scope

The direct Objective-C++/Metal backend advances the v6 neural propagation step on Apple silicon. It retains all 166,700 MaleCNS v1.0 neurons and 25,582,938 directed edges. RGB input, modeled propagation, host float64 eligibility and centered plasticity, reinforcement, and the fixed neural-to-button decoder remain separate. CPU is the default scientific reference. Metal is available for offline validation, benchmarking, and explicitly selected experiments; the live server is unchanged.

The initial validated system is an Apple M4 Pro with unified memory and Apple GPU family 7 support, macOS 26.6.2, Apple clang 21.0.0, SDK 26.5, and conservative `macos-metal2.4` kernels. Generated `.air`, `.metallib`, `.dylib`, incoming-index caches, checkpoints, and downloaded connectome data remain untracked.

## Build and validation

Prepare the checksum-verified MaleCNS graph as described in the root README, then run:

```sh
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning_v6.metal.build --probe
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning_v6.metal.validate \
  --out outputs/doom-learning/metal-validation-local
```

The incoming CSR cache is a stable one-to-one permutation of every outgoing edge. Cache metadata records outgoing and incoming structural hashes. Validation starts CPU and Metal from the same checkpoint and saved deterministic RGB trace. It checks event Jaccard, spike totals, per-neuron rates, one-tick matching, fixed decoder decisions, scientific outcomes, plastic weights, graph identity, and repeatability.

The committed M4 Pro 40 ms report passed: Jaccard `0.9973775`, spike difference `0`, rate correlation `1.0`, one-tick fraction `1.0`, exact decoder decisions and weights, and bitwise repeated Metal state. This is numerical evidence over the stated horizon. The preserved 80 ms stress control failed with Jaccard `0.9421872` and one-tick fraction `0.9879394`. Target-local Metal summation and CPU active-queue summation accumulate mixed-sign inputs in different deterministic orders.

## Checkpoints and offline selection

V6 checkpoints record their producer backend while compatibility depends on model, graph, configuration, sensory mapping, and learning-rule identity. CPU-to-Metal, Metal-to-CPU, and Metal-to-Metal continuation tests pass. Metal selection refuses missing, failed, or stale evidence before an episode begins:

```sh
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning_v6.survival \
  --backend metal \
  --metal-validation outputs/doom-learning/metal-validation-m4pro/report.json \
  --out outputs/doom-learning/metal-smoke --seeds 41031 --eval-seeds 61031 \
  --seconds 1 --train-episodes 1
```

## Performance result

Run `python -m doom_learning_v6.metal.benchmark --out FRESH_PATH` for five matched repetitions. On the M4 Pro, CPU median neural time was `0.03621 s` per `0.04 s` simulated; Metal was `0.17275 s`. The measured speedup was `0.2096x`, equivalent to Metal running about 4.77 times slower. Metal reached `0.2316x` real time. Peak RSS was `2.28 GiB`, free memory remained at 66%, and swap stayed zero.

This fails the required 2x speed gate. The deterministic gather implementation needs further optimization before long Metal training is worthwhile. Backend acceleration can reduce wall time only. It does not improve learning per frame, neural tick, reinforcement event, or episode, and it does not establish biological learning.

The planned one-second offline smoke was attempted while the machine's one-minute load average was 19.87. Plastic training, held-out frozen evaluation, and the black-vision control completed with exact validation provenance. The next warmup exceeded four wall minutes, so the smoke process was stopped and its completed records preserved. Retention, erasure, frozen-arm, and shuffled-arm controls remain unrun. This partial smoke is not an end-to-end pass.
