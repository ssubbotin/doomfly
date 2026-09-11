# C++ Multi-Trajectory Executor Design

## Goal

Build a portable C++17 CPU executor that advances multiple independent v6 neural trajectories concurrently while sharing one immutable MaleCNS v1.0 graph. A lane must reproduce the existing single-lane CPU reference bit for bit. The first milestone covers neural propagation only; ViZDoom, RGB conversion, reinforcement scheduling, centered plasticity, fixed decoding, experiment control, and reports remain separate Python stages.

## Selected Approach

Use one persistent native executor per Python process. Python constructs a graph owner from one fully calibrated `MemoryBrain`, freezes its structural arrays and baseline weights, creates independent lane-state arrays, and passes stable pointers to C++ once. A persistent worker pool assigns one complete lane to one worker for each synchronized advance. Each worker runs the existing scalar operation order, so scheduling cannot change a lane's result.

The native ABI accepts a common step count from 1 through 100. Every lane has its own drive and mutable state. The caller is responsible for splitting execution at RGB, reinforcement, and learning boundaries. This keeps cross-lane synchronization outside the neural kernel and makes the first implementation independently testable.

## Alternatives Considered

### Independent Python processes

Separate processes already avoid the GIL and require little new code. They load duplicate graph and weight arrays, provide no shared native scheduler, and cannot amortize graph ownership. They remain a useful benchmark control.

### SIMD across the lane dimension

An edge-stationary SIMD kernel could reuse graph cache lines across lanes. Divergent active sets make its benefit uncertain, and changing loop structure raises a larger exactness risk. It is deferred until the lane-parallel executor measures memory-bandwidth limits.

### Persistent lane-parallel executor

One worker advances one lane with the reviewed scalar order. This approach shares graph capacity, uses multiple CPU cores, preserves lane isolation, and supplies a clean basis for later SIMD work. It is the selected first milestone.

## Native Components

`doom_learning_v6/cpu_batch/api.h` defines a versioned C ABI with graph, lane, timing, and opaque-handle structures. `doom_learning_v6/cpu_batch/executor.cpp` owns the worker pool and contains the overlay-aware propagation implementation. `doom_learning_v6/cpu_batch/build.py` produces a platform library using the repository's existing `clang++ -O3 -std=c++17` convention and records source and binary digests.

The native graph contains const pointers for outgoing CSR, baseline weights, the per-edge plastic-slot map, cell masks, rest potentials, and scalar dynamics. The executor never writes through these pointers. Creation validates counts, CSR boundaries, plastic slot bounds, delay configuration, lane count, and worker count before starting threads.

Each lane descriptor contains only mutable state: voltage, conductance, refractory counters, drive, previous drive, delayed queue, counts, active queue, activity flags, timestamps, eligibility, modulation, adaptation, cursor, and 4,184 plastic weights. Mutable buffers may not alias between lanes.

For each delivered edge, the executor reads the lane overlay when the shared signed 16-bit slot is nonnegative; otherwise it reads the shared baseline weight. Slot `-1` identifies an immutable edge. The 4,184 slots fit within signed 16-bit storage. Every released edge remains represented and delivered.

## Python Interface

`doom_learning_v6/cpu_batch/backend.py` provides three focused types:

- `SharedCpuGraph.from_brain(brain)` copies one finalized baseline weight array, reuses immutable graph/configuration arrays, builds the plastic-slot map, freezes all shared NumPy buffers, and records identity digests.
- `CpuBatchLane.from_brain(graph, brain)` copies mutable state and extracts only the plastic weight overlay.
- `MultiTrajectoryCpuExecutor(graph, lanes, workers)` registers stable buffers, exposes `advance(steps)`, reports timing and memory, and releases native workers deterministically.

The wrapper holds strong references to every NumPy buffer for the handle lifetime. `advance()` releases the GIL through `ctypes.CDLL`, clears per-call counts consistently with the reference path, invokes the native batch, updates lane cursors, and returns one count array per lane. Plasticity code may update only `lane.plastic_weights` between calls.

## Concurrency and Determinism

Workers sleep on a condition variable between generations. An advance publishes the common step count, assigns lane indices through an atomic counter, waits for every selected lane, and returns only after all writes are visible. A lane is touched by one worker for the full call. Shared graph data is read-only, and no mutable value is shared across lanes.

Lane output must remain unchanged across worker counts, lane ordering, neighboring lane contents, and repeated runs. The native executor rejects reentrant `advance()` calls and destruction during an active generation.

## Error Handling

All exported functions return explicit status codes and expose a per-handle error string. Validation finishes before any lane advances, preventing partial work for malformed descriptors. A native worker failure poisons the handle; later calls fail until the handle is destroyed. Python converts failures to `BackendError` and closes the handle through a context manager.

## Verification

Toy-graph tests first compare one batch lane with the existing CPU backend after each call, including counts, cursor, voltage, conductance, delay queues, active queues, eligibility, modulation, adaptation, and plastic weights. Tests then cover divergent inputs, plastic overlays, lane reordering, worker counts, repeated execution, alias rejection, invalid graphs, and cleanup.

A full-graph validation on `mini` compares every retained CPU state digest for at least 40 ms and 80 ms traces. It also verifies 166,700 neurons, 25,582,938 edges, 4,184 plastic slots, unchanged structural digests, and identical fixed decoder decisions. The performance report measures one, two, and four lanes with five contiguous repetitions. It reports simulated brain-seconds per wall-second, per-lane latency, peak memory, graph bytes shared, state bytes per lane, and scaling efficiency. Performance results cannot establish learning or biological validity.

## Scope Boundaries

This milestone does not move ViZDoom, RGB sampling, reinforcement selection, centered learning, decoding, checkpoint files, or experiment reports into C++. It does not prune edges, approximate time, coarsen the 0.1 ms step, share mutable state, or use telemetry for actions. Integration with the survival runner follows only after native exactness and scaling are measured.
