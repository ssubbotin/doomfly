# Portable Metal Backend Design

**Status:** Approved in discussion on 2026-09-11; awaiting review of this written specification.

## Context

DOOMFLY v6 advances 166,700 retained MaleCNS v1.0 neurons and 25,582,938
directed connections at a 0.1 ms timestep. Its current C++ kernel is serial and
limits experiment throughput. The project needs a GPU backend for offline parity
work and controlled training on Apple Silicon. The Apple M4 Pro host `mini` is
the first validation machine.

Metal is an execution backend for the existing model. It must preserve live RGB
input processing, neural propagation, reinforcement, plasticity, fixed decoding,
controls, provenance, and negative results as separate traceable stages. Faster
execution can reduce wall-clock training time and permit more controlled runs.
It must not change the amount of simulated experience required for a given
learning result.

## Goals

- Run the complete v6 neural propagation model on Apple Silicon through direct
  Objective-C++ and Metal compute kernels.
- Retain every released connection among retained neuronal entries, including
  weak, duplicate, self, and modulatory connections.
- Preserve the current CPU kernel as the scientific reference.
- Provide deterministic repeated execution on the initial M4 Pro configuration.
- Exchange checkpoints between CPU and Metal backends.
- Reach at least a 2x neural-kernel speedup on the same M4 Pro and input trace,
  with 1x real-time neural execution as the target.
- Begin with offline parity and headless training. Live serving remains a later
  gated integration.

## Non-goals

- Replacing neural dynamics with a game policy, learned controller, CatBoost
  model, or genetic algorithm.
- Changing visual input, reinforcement delivery, the centered v6 plasticity rule,
  or fixed neuron-to-button decoding.
- Supporting discrete AMD or Intel GPUs, iOS, CUDA, Metal 4-only features, or
  non-Apple-Silicon Macs in the first implementation.
- Proving biological validity or successful learning through performance,
  changed weights, or isolated long-survival episodes.
- Publishing a generic Metal graph-compute package before a second consumer and
  stable model-independent API exist.

## Repository and Git Decision

The backend belongs in the existing DOOMFLY repository because it executes the
same model and depends on the same graph preparation, checkpoints, tests,
evidence, licenses, and public explanation. A separate repository would create
duplicate model definitions and weaken traceability between CPU and Metal
results.

Development uses the public fork `ssubbotin/doomfly` and branch `metal-backend`.
The remotes are:

```text
origin    git@github.com:ssubbotin/doomfly.git
upstream  https://github.com/nftechie/doomfly.git
```

The Metal source remains in the same history as the experiment. Generated
`.dylib`, `.air`, and `.metallib` files stay ignored. A reusable library may be
extracted later once its API is independent of v6 and used by another project.

## Chosen Approach

A direct Objective-C++ library provides a C ABI consumed through Python
`ctypes`. It owns the Metal device, command queue, pipeline states, and shared
buffers. This matches the current native-library boundary and gives the project
explicit control over event delays, execution order, state export, errors, and
profiling.

MLX custom kernels remain suitable for isolated throwaway measurements. MLX is
excluded from the production backend to avoid adding lazy execution and a
fast-moving framework abstraction to the audited neural-state path. PyTorch MPS
is excluded because its tensor abstractions do not match the sparse event and
checkpoint semantics closely enough to justify the dependency.

The implementation follows Apple's compute-command model: build reusable
pipeline states, bind buffers, dispatch kernels, and validate command-buffer
completion. Relevant platform references are the Apple documentation for
[GPU calculations](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu),
[compute pipeline states](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate),
and [Metal feature limits](https://developer.apple.com/metal/limits/).

## Source Layout

```text
doom_learning_v6/
  backend.py                 # backend contract and selection
  kernel.cpp                 # retained CPU reference
  metal/
    api.h                    # stable C ABI
    backend.mm               # device, buffers, commands, errors
    kernels.metal            # neural compute kernels
    build.py                 # atomic local build and provenance
tests/
  test_doom_backend_contract.py
  test_doom_metal_parity.py
```

The first milestone changes only v6 offline execution. Baseline and live-server
integration are separate later milestones after all gates in this document pass.

## Backend Contract

`MemoryBrain` continues to own experiment orchestration, retinal drive,
stimulation, the centered plasticity rule, metrics, and checkpoint metadata. It
selects an explicit `cpu` or `metal` backend. The contract supports:

- initialization from the full graph, v6 parameters, and complete mutable state;
- advancement by an integer number of 0.1 ms ticks;
- spike-count and timing retrieval at each 10 ms learning boundary;
- updates to identified plastic weights by canonical edge ID;
- complete state export and restoration;
- build, model, device, and runtime provenance retrieval; and
- explicit shutdown.

The CPU implementation wraps the existing `kernel.cpp`. The Metal implementation
wraps the Objective-C++ C ABI. Selecting Metal on an unsupported machine, loading
a stale binary, or encountering a GPU command error raises a backend-specific
exception. There is no automatic CPU fallback.

Metal shared buffers are the authoritative mutable neural state while a command
buffer is running. The synchronous 10 ms boundary makes host reads and plastic
weight writes safe. Python-visible NumPy views may map shared buffers where the
ABI can guarantee ownership and lifetime. Delayed-event conversion remains an
explicit operation. Direct host access during an in-flight command is forbidden.

## Graph Representation

The original outgoing CSR and canonical weight array remain unchanged. Metal
adds a derived incoming CSR so each postsynaptic neuron can accumulate delivered
weights in a stable order. Every incoming entry stores its presynaptic neuron and
original edge ID; it never owns a second weight value.

The incoming index is built deterministically and cached with graph and builder
hashes. Initialization verifies that its edge IDs are an exact permutation of
`0..25,582,937`, each referenced pre/post pair matches the outgoing CSR, and all
counts and hashes match the loaded graph. Any discrepancy aborts initialization.

## Metal Execution Flow

One Python call normally advances 10 ms:

1. Python updates retinal drive, tonic drive, and external stimulation in shared
   buffers.
2. Objective-C++ records 100 ordered 0.1 ms tick sequences into one command
   buffer.
3. Each tick evolves eligible neurons with the existing lazy closed-form
   equations, records spikes, delivers the current delay slot, resets newly fired
   neurons, and advances the ring cursor.
4. The host waits for successful completion and reads per-neuron spike counts.
5. Python applies the unchanged centered plasticity rule and writes the 4,184
   identified plastic weights by canonical edge ID.

Metal scans the active flags in stable neuron-ID order rather than depending on
the CPU active-list order. The `last` timestamps retain lazy evolution semantics.
The initial implementation may scan all neuron flags on each tick; this is a
performance choice and does not crop the network or skip eligible dynamics.

The 1.8 ms delay is represented by an explicit 19-slot ring. Each slot contains
a spike bitmap and canonical sorted neuron IDs. The 2.2 ms refractory period
prevents one neuron from occurring twice in a single slot. Spike counts remain
per-neuron integers accumulated over the requested interval.

### Host-owned float64 state

The centered v6 rule already updates `rate_kc`, `rate_dan`, `memory_u`, and
`memory_w` as float64 host arrays every 10 ms. Those arrays remain on the host.
The v6 checkpoint also retains the older per-neuron float64 `eligibility` and
int64 `eligibility_last` arrays. The centered rule does not read them, but KC
spikes continue to update them and parity requires preserving that behavior.

Metal therefore records tick-resolved KC spike events during each command. At
the successful 10 ms boundary, an Objective-C++ host routine processes those
events in tick and neuron-ID order using the same double-precision equation as
the CPU kernel. Its event capacity is derived from the KC count, interval length,
and 2.2 ms refractory lower bound, so overflow is an initialization error rather
than a lossy runtime condition. This keeps double precision off the GPU without
discarding or approximating checkpoint state.

## Deterministic Delivery

Floating-point atomic accumulation is excluded because contribution order could
vary between executions. Delivery uses two stages:

1. Delayed presynaptic spikes traverse the outgoing CSR and mark affected
   postsynaptic neurons. Idempotent 32-bit integer atomics may set flags.
2. One thread per affected target traverses its incoming entries in stable edge
   order. It checks the delayed-spike bitmap and sums matching weights locally
   before updating conductance or the modeled modulatory trace.

This method can inspect inactive incoming edges of an affected target. It is the
correctness-first implementation. If it misses the performance gate, a later
optimization will compact active edge events and reduce them by target in a
fixed order. An unordered floating-point atomic scatter is outside the accepted
training design.

All 100 tick sequences are encoded in one command buffer, with separate kernel
dispatches providing device-wide synchronization between phases. Pipeline states
are created once and reused.

## Checkpoints and Continuity

The canonical checkpoint remains the existing backend-independent `.npz` format.
It includes every mutable v6 field, all weights, simulation cursor, total spike
count, active membership, and every pending delayed event. Metal materializes
active neurons and delayed slots in sorted neuron-ID order during export.

Checkpoint metadata adds the selected backend and its build/device provenance
for audit purposes. Compatibility continues to depend on model, graph, plastic
edge, rule, tonic-current, sensory-map, and configuration hashes. Backend
provenance describes the producer and does not prevent a compatible checkpoint
from loading on the other backend.

CPU-to-Metal and Metal-to-CPU continuation must meet the numerical parity gates.
Metal-to-Metal restoration on the same validated build and device must be
bitwise identical. Checkpoints are written only after successful command-buffer
completion. A failed command never advances the committed Python cursor.

## Portability and Build

The baseline requires arm64 macOS, unified memory, Apple GPU family 7 or newer,
32-bit integer atomics, and standard compute dispatch. Kernels avoid Metal 4,
64-bit atomics, floating-point atomics, ray tracing, and advanced SIMD-group
features. Threadgroup sizes are selected from pipeline and device limits rather
than assuming an M4-specific width.

`build.py` invokes the installed Apple command-line tools to compile Metal source
and link an arm64 Objective-C++ dynamic library against the system Metal and
Foundation frameworks. It writes artifacts atomically and records source,
binary, compiler, SDK, macOS, compile-flag, and Metal-language hashes or values.
The runtime also records device name, registry ID, GPU families, recommended
working-set size, and unified-memory support. Source ships in Git; target-local
binaries do not.

The M4 Pro on `mini` is the first validated machine. Portability claims initially
mean that the implementation uses this conservative Apple Silicon feature
subset. Additional machines become validated only through the same parity and
performance evidence.

## Failure Handling

Initialization fails on missing Metal support, unsupported GPU family, graph
mismatch, allocation failure, compile failure, stale metadata, or ABI mismatch.
Every command buffer is checked for completion and error status. A command error
poisons the backend instance; the experiment records the failure and must restore
the latest verified checkpoint into a new instance before continuing.

Partial GPU state is never presented as a completed interval. The implementation
does not retry with changed parameters, reduce the graph, skip neural ticks, or
fall back to CPU. Training refuses to start when required parity evidence for the
exact backend build is absent.

## Validation Gates

### Structural integrity

- Exact neuron-ID, outgoing-pointer, outgoing-post, original-weight, and
  plastic-edge hashes.
- Exact one-to-one incoming-edge permutation and matching endpoints.
- Explicit tests retaining weak, self, duplicate, and modulatory edges.

### Numerical parity

Micrographs cover delay timing, refractory writes, adaptation, conductance
decay, changing drive, modulation, self-edges, duplicate edges, stimulation,
active-set transitions, and checkpoint boundaries. Discrete events and counters
must match exactly on fixtures. Float32 voltage, conductance, drive, luminance,
modulation, adaptation, and weight state must match with `rtol=1e-5` and
`atol=0.002`. Host-owned float64 state and its timestamps must match exactly.

Full-graph CPU/Metal trials use identical initial checkpoints and input traces.
They require:

- spike-event Jaccard similarity of at least `0.995`;
- total-spike difference of at most `0.5%`;
- per-neuron firing-rate correlation of at least `0.999`;
- at least `99.9%` of matched spikes within one 0.1 ms tick;
- exact fixed-decoder decisions and scientific gate outcomes; and
- plastic weights within `rtol=1e-4` after a matched learning trace.

Repeated M4 Pro runs using the same build, checkpoint, and inputs must be bitwise
identical. CPU-to-Metal and Metal-to-CPU continuation use the full-graph parity
thresholds. Metal-to-Metal checkpoint continuation must be bitwise identical.

### M4 Pro viability

Benchmarks compare warmed CPU and Metal backends on `mini`, using the same graph,
checkpoint, input trace, interval length, and otherwise idle measurement window.
They report at least five repetitions and distributions rather than a single
best time. Acceptance requires:

- median Metal neural time at least 2x faster than median CPU neural time;
- a reported result against the target of one simulated neural second per wall
  second;
- peak process resident memory below 8 GiB;
- no sustained swap growth or critical memory pressure; and
- separate reporting of kernel, synchronization, Python plasticity, ViZDoom,
  frame conversion, and checkpoint costs.

Existing workloads are not stopped for a benchmark. Measurement waits for a
suitable load window. If deterministic gather fails the speed gate, active-event
compaction becomes a measured follow-up optimization before training proceeds.

## Offline Training Rollout

Passing structural, numerical, checkpoint, determinism, and performance gates
permits offline headless ViZDoom training. Each candidate run has matched
frozen-weight, shuffled-reinforcement, and original-baseline controls. Runs record
seeds, graph/configuration hashes, input and reward traces, backend metadata,
hardware details, wall time, neural time, checkpoints, and failures.

Metal is expected to shorten wall-clock execution for the same simulated neural
and game experience. It does not change sample efficiency. End-to-end speedup can
be lower than kernel speedup when rendering, frame conversion, Python plasticity,
or checkpoint output becomes limiting. The first release runs one sequential
recurrent simulation per process. Independent seeds may run separately after
memory and contention measurements support it.

Training results must be evaluated with frozen learned weights and the existing
visual, conditioning, survival, and control gates. The current v6 failed status
remains until new evidence passes those gates. Runtime, changed weights, and long
individual episodes do not establish successful learning or biological validity.

## Deferred Live Integration

Live-server backend selection, public streaming, and deployment configuration are
outside the first implementation milestone. They begin only after offline frozen
evaluation passes. Before any launch approval, the published "How it works" page
must be updated and checked against the exact deployed code, configuration,
connectome release, reinforcement input, plastic connections, validation results,
and run mode. Public text must distinguish training, frozen evaluation, replay,
and the original baseline.

## Completion Criteria

The Metal backend milestone is complete when its source, build provenance,
contract tests, micrograph tests, full-graph parity evidence, cross-backend
checkpoint evidence, deterministic replay evidence, and M4 Pro benchmark report
are committed. Offline training is authorized only after those artifacts pass
their gates. Live use requires the deferred evaluation and publication work.
