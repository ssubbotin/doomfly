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

The incoming CSR cache is a stable one-to-one permutation of every outgoing edge. Cache metadata records outgoing and incoming structural hashes. At backend creation, Metal builds an inverse edge-to-incoming-position map. Delivered edges set bits in this incoming-order space. Each affected target scans 32-edge words and accumulates set positions in ascending incoming order. An indirect kernel clears only touched words after each tick. This retains every released edge, including duplicate, self, weak, and modulatory connections, while avoiding repeated membership checks across inactive incoming edges.

Validation starts CPU and Metal from the same checkpoint and saved deterministic RGB trace. It checks event Jaccard, spike totals, per-neuron rates, one-tick matching, fixed decoder decisions, scientific outcomes, plastic weights, graph identity, and repeatability.

The committed M4 Pro 40 ms report passed: Jaccard `0.9973775`, spike difference `0`, rate correlation `1.0`, one-tick fraction `1.0`, exact decoder decisions and weights, and bitwise repeated Metal state. This is numerical evidence over the stated horizon. The rerun 80 ms stress control failed with the same Jaccard `0.9421872` and one-tick fraction `0.9879394` as the earlier implementation. Target-local Metal summation and CPU active-queue summation accumulate mixed-sign inputs in different deterministic orders.

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

Run `python -m doom_learning_v6.metal.benchmark --out FRESH_PATH` for five matched repetitions. On the M4 Pro, CPU median neural time was `0.03454 s` per `0.04 s` simulated; Metal was `0.08462 s`, including `0.07112 s` of GPU execution. Metal improved by `2.04x` over the previous `0.17275 s` backend median and reached `0.4727x` real time. CPU-to-Metal speedup is `0.4082x`. Peak RSS was `2.37 GiB`, free memory remained at or above 66%, and swap stayed zero.

An event-work diagnostic found 846,035 delivered edges but 51,083,839 incoming-edge membership checks over the 40 ms trace, a `60.38x` scan amplification. The sparse incoming-edge bitmap removes that amplification. Earlier compact presynaptic/target queues, bitmap-word target marking, and target-word gathering were rejected after neutral or slower measurements. Their status and available measurements are recorded in `outputs/doom-learning/metal-optimization-study-m4pro.json`. The compact-queue validation and benchmark reports retain source, binary, and environment hashes; shorter exploratory checks are explicitly labeled as unverified laboratory notes because their raw samples and source patches were unavailable.

The result still fails the required 2x speed gate, so long Metal training remains closed. Backend acceleration can reduce wall time only. It does not improve learning per frame, neural tick, reinforcement event, or episode, and it does not establish biological learning.

The planned one-second offline smoke was attempted while the machine's one-minute load average was 19.87. Plastic training, held-out frozen evaluation, and the black-vision control completed with exact validation provenance. The next warmup exceeded four wall minutes, so the smoke process was stopped and its completed records preserved. Retention, erasure, frozen-arm, and shuffled-arm controls remain unrun. This partial smoke is not an end-to-end pass.
