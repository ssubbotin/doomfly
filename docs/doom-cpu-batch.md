# Shared CPU Trajectory Executor

## Purpose and Scope

The CPU batch executor advances independent v6 neural trajectories concurrently while retaining one immutable MaleCNS v1.0 graph. It is a neural-propagation component. Live RGB conversion, reinforcement selection, centered plasticity, fixed neural decoding, ViZDoom, checkpoints, and experiment scheduling remain explicit Python stages.

Every released connection is retained, including weak, duplicate, and self edges. A plastic connection reads its lane's compact overlay value. Every other connection reads the shared finalized baseline value. The full graph contains 166,700 neurons, 25,582,938 edges, and 4,184 plastic slots.

## Memory and Execution Model

`SharedCpuGraph.from_brain()` freezes CSR endpoints, baseline weights, the signed per-edge plastic-slot map, masks, resting potentials, and dynamics. Its validated M4 Pro allocation is 258,196,660 bytes.

`CpuBatchLane.from_brain()` copies voltage, conductance, refractory state, drive history, delay queues, active sets, eligibility, modulation, adaptation, cursor, and 4,184 plastic weights. Each validated lane occupies 23,854,924 bytes.

`MultiTrajectoryCpuExecutor` registers these stable buffers once. Persistent C++17 workers claim complete lanes for each synchronized `advance(steps)` call. One lane always stays on one worker for the whole call. Calls accept 1 through 100 steps and preserve the existing 0.1 ms integration order. The Python caller may update `lane.plastic_weights` only between calls.

## Build and Verification

```bash
python -m doom_learning_v6.cpu_batch.build --probe
python -m pytest -q tests/test_doom_cpu_batch_*.py
python -m doom_learning_v6.cpu_batch.validate \
  --out outputs/doom-learning/cpu-batch-validation-m4pro
python -m doom_learning_v6.cpu_batch.benchmark \
  --validation outputs/doom-learning/cpu-batch-validation-m4pro/report.json \
  --out outputs/doom-learning/cpu-batch-benchmark-m4pro \
  --repetitions 5
```

Set `OPENBLAS_NUM_THREADS=1` for full experiment commands. Output directories must be fresh. Generated libraries and build metadata remain ignored.

## M4 Pro Results

Apple clang 21 built ABI version 1 on an M4 Pro with 24 GiB unified memory. Exact validation passed at 40 ms and 80 ms. Every retained mutable state buffer, spike-count buffer, structural identity, source lock, and fixed decoder decision matched the legacy CPU oracle bit for bit. All five repetitions were bitwise stable.

| Lanes | Median brain-s/wall-s | Median latency per lane | Scaling efficiency |
|---:|---:|---:|---:|
| 1 | 1.145 | 34.94 ms | 100.0% |
| 2 | 2.246 | 17.81 ms | 98.1% |
| 4 | 3.958 | 10.11 ms | 86.4% |

Peak process RSS was 1,403,191,296 bytes. The hot measurements exclude calibrated-brain construction, graph loading, graph freezing, and lane allocation. Those one-time stages can dominate a short command and should be amortized across a long-lived training process.

These results establish numerical parity and propagation throughput. They do not establish learned behavior, biological validity, or survival improvement. Integration with the training runner follows as a separate measured milestone.
