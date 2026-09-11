# Portable Apple Metal Backend

## Supported scope

The direct Objective-C++/Metal backend advances the v6 neural propagation step on Apple silicon. It retains all 166,700 MaleCNS v1.0 neurons and 25,582,938 directed edges. RGB input, modeled propagation, host float64 eligibility and centered plasticity, reinforcement, and the fixed neural-to-button decoder remain separate. Mutable propagation state stays on the device between 10 ms bins. CPU is the default scientific reference. Metal is available for offline validation, benchmarking, and explicitly selected experiments; the live server is unchanged.

The initial validated system is an Apple M4 Pro with unified memory and Apple GPU family 7 support, macOS 26.6.2, Apple clang 21.0.0, SDK 26.5, and conservative `macos-metal2.4` kernels. Generated `.air`, `.metallib`, `.dylib`, incoming-index caches, checkpoints, and downloaded connectome data remain untracked.

## Build and validation

Prepare the checksum-verified MaleCNS graph as described in the root README, then run:

```sh
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning_v6.metal.build --probe
OPENBLAS_NUM_THREADS=1 .venv-neural/bin/python -m doom_learning_v6.metal.validate \
  --out outputs/doom-learning/metal-validation-local
```

The incoming CSR cache is a stable one-to-one permutation of every outgoing edge. Cache metadata records outgoing and incoming structural hashes. At backend creation, Metal builds an inverse edge-to-incoming-position map. Each tick uses two dispatches separated by the required propagation boundary. `df_integrate_mark` integrates active neurons and marks released outgoing edges in incoming-order space. `df_gather_finalize` atomically claims each target's exact bit range, accumulates positions in ascending incoming order, clears those bits, and finalizes delayed state. Target masks remain disjoint when several targets share one 32-bit word. This retains every released edge, including duplicate, self, weak, and modulatory connections.

Validation starts CPU and Metal from the same checkpoint and saved deterministic RGB trace. It checks event Jaccard, spike totals, per-neuron rates, one-tick matching, fixed decoder decisions, scientific outcomes, plastic weights, graph identity, and repeatability.

The two-dispatch M4 Pro 40 ms report passed: Jaccard `0.9973775`, spike difference `0`, rate correlation `1.0`, one-tick fraction `1.0`, exact decoder decisions and weights, and bitwise repeated Metal state. Its Metal state hash remains `59a2739e...d0272e6`; it recorded 21,326 events. This is numerical evidence over the stated horizon. The rerun 80 ms stress control reproduced the retained Metal state hash `8fec98ef...4a80e`, count hash, all bin hashes, and 52,236 events. It still fails with Jaccard `0.9421872` and one-tick fraction `0.9879394`. Target-local Metal summation and CPU active-queue summation accumulate mixed-sign inputs in different deterministic orders.

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

Run `python -m doom_learning_v6.metal.benchmark --out FRESH_PATH` for five repetitions. The program warms each backend once, then measures contiguous CPU and Metal blocks. This represents sustained Metal training and avoids repeatedly cooling the GPU with an intervening CPU reference run.

On the M4 Pro, the two-dispatch result used exactly 808 dispatches per 40 ms trace and no indirect dispatches. Median GPU time was `0.03010 s`, which passes the `0.05690 s` target. Median Metal neural time was `0.03328 s`; CPU was `0.03464 s`, giving `1.0408x` CPU throughput. Total sample wall time was `0.03886 s`, and Metal reached `1.2019x` real time. Relative to the retained resident-state result, neural time improved by 53.9%, GPU time by 56.8%, and wall time by 49.1%. Peak RSS was `2.35 GiB`, free memory remained at or above 85%, and swap stayed zero.

Steady-state full upload and materialization are both zero. Each 10 ms bin transfers 1,383,816 bytes: drive, counts, recorded events, and all 4,184 identified plastic edge IDs and values. A small Metal kernel applies those weights. Two preserved trials that wrote scattered values from the CPU raised GPU medians to `0.08667 s` and `0.08436 s`. Two interleaved CPU/Metal trials also retained the GPU cold-start penalty and are preserved with the final contiguous report.

An event-work diagnostic found 846,035 delivered edges but 51,083,839 incoming-edge membership checks over the 40 ms trace, a `60.38x` scan amplification. The sparse incoming-edge bitmap removes that amplification. Earlier compact presynaptic/target queues, bitmap-word target marking, and target-word gathering were rejected after neutral or slower measurements. Their status and available measurements are recorded in `outputs/doom-learning/metal-optimization-study-m4pro.json`. The compact-queue validation and benchmark reports retain source, binary, and environment hashes; shorter exploratory checks are explicitly labeled as unverified laboratory notes because their raw samples and source patches were unavailable.

The resident-state and two-dispatch targets passed. The required 2x CPU speed gate still fails, so long Metal training remains closed. Backend acceleration can reduce wall time only. It does not improve learning per frame, neural tick, reinforcement event, or episode, and it does not establish biological learning.

The dispatch lower-bound diagnostic predicted `0.01747 s` total savings. The implemented two-dispatch pipeline improved median GPU time by `0.03965 s`, from `0.06975 s` to `0.03010 s`, while preserving every retained Metal digest. The result validates the selected direction and is retained. Further acceleration should begin with separate timing of the two fused kernels and a fresh decision against the remaining `0.01727 s` neural-time target.

The planned one-second offline smoke was attempted while the machine's one-minute load average was 19.87. Plastic training, held-out frozen evaluation, and the black-vision control completed with exact validation provenance. The next warmup exceeded four wall minutes, so the smoke process was stopped and its completed records preserved. Retention, erasure, frozen-arm, and shuffled-arm controls remain unrun. This partial smoke is not an end-to-end pass.
