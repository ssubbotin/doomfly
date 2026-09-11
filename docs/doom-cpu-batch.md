# Shared CPU Trajectory Executor

## Purpose and Scope

The CPU batch executor advances independent v6 neural trajectories concurrently while retaining one immutable MaleCNS v1.0 graph. It is a neural-propagation component. Live RGB conversion, reinforcement selection, centered plasticity, fixed neural decoding, ViZDoom, checkpoints, and experiment scheduling remain explicit Python stages.

Every released connection is retained, including weak, duplicate, and self edges. A plastic connection reads its lane's compact overlay value. Every other connection reads the shared finalized baseline value. The full graph contains 166,700 neurons, 25,582,938 edges, and 4,184 plastic slots.

## Memory and Execution Model

`SharedCpuGraph.from_brain()` freezes CSR endpoints, baseline weights, the signed per-edge plastic-slot map, masks, resting potentials, and dynamics. Its validated M4 Pro allocation is 258,196,660 bytes.

`CpuBatchLane.from_brain()` copies voltage, conductance, refractory state, drive history, delay queues, active sets, eligibility, modulation, adaptation, cursor, and 4,184 plastic weights. Each validated lane occupies 23,854,924 bytes.

`MultiTrajectoryCpuExecutor` registers these stable buffers once. Persistent C++17 workers claim complete lanes for each synchronized `advance(steps)` call. One lane always stays on one worker for the whole call. Calls accept 1 through 100 steps and preserve the existing 0.1 ms integration order. The Python caller may update `lane.plastic_weights` only between calls.

Graph and lane owners can only be created from a validated brain. Registration checks every buffer's dtype, shape, layout, mutability, finiteness, identity, and memory independence before exposing pointers to C++. The executor pins the exact registered arrays and rejects changed addresses or unsafe cursor, active-set, and delay-queue indices before each call. The C ABI repeats index checks for direct native callers. `copy_from_brain()` refreshes a lane in place without changing registered addresses.

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

Apple clang 21 built ABI version 1 on an M4 Pro with 24 GiB unified memory. Exact validation passed at 40 ms and 80 ms. Each horizon used four lanes, four workers, and two fresh repeats. Every retained mutable state buffer, spike-count buffer, structural identity, source lock, and fixed decoder decision matched the legacy CPU oracle bit for bit. All five timing repetitions were bitwise stable. Reports identify Git commit `e8aef4c683d4df277515274f18bd8ac0f73a63a7`, every runtime source, the graph and manifest, and the exact validation report used by the benchmark. The committed `inputs.npz` records the deterministic validation drive; the benchmark verifies its file and array digests before measuring.

| Lanes | Median brain-s/wall-s | Batch latency | Amortized wall cost per trajectory | Scaling efficiency |
|---:|---:|---:|---:|---:|
| 1 | 1.059 | 37.76 ms | 37.76 ms | 100.0% |
| 2 | 1.892 | 42.29 ms | 21.15 ms | 89.3% |
| 4 | 2.899 | 55.18 ms | 13.80 ms | 68.4% |

Peak process RSS was 1,702,739,968 bytes. Batch latency is the time observed by every synchronized lane. Dividing it by lane count gives amortized throughput cost, not individual-lane latency. The measurements include Python and native pre-dispatch safety checks. They exclude calibrated-brain construction, graph loading, graph freezing, and lane allocation. Those one-time stages can dominate a short command and should be amortized across a long-lived training process.

These results establish numerical parity and propagation throughput. They do not establish learned behavior, biological validity, or survival improvement. Integration with the training runner follows as a separate measured milestone.
