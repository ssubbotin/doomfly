# Metal Temporal Windows Design

Date: 2026-09-13. Status: implementation design, not validated acceleration or learning.
Parent: `6708110addebfdb6ae922ae630940b0a94e75079` (reviewed batch backend).
Standing user approval authorizes routine scoped implementation and verification.

## Purpose and choice

Reduce neural scheduling and gathering work on the M4 Pro without changing
epoch-5 dynamics, observation timing, or training. The seven-call
[council](../../experiments/2026-09-13-metal-temporal-window-council.md)
recommends empty-atomic guards followed by bounded temporal preparation.
A persistent globally synchronized kernel introduces portability/deadlock
risks. Compact queues require a larger complete-workset proof. Both remain
future experiments; this plan implements windows now regardless of guard timing.

## Global Constraints

- Retain all 166,700 MaleCNS v1.0 neurons, all 25,582,938 released retained connections, and all 4,184 existing KC→MBON11 plastic slots; preserve duplicate, self, weak and modulatory edges.
- Preserve epoch-5 zero-seeded ascending-incoming arithmetic, explicit FMA sites, decay tables, signed-zero behavior, lazy evolution and host-double centered plasticity operation order.
- Keep RGB sensory input, modeled propagation, reinforcement, plasticity and fixed neuron-to-button decoding separate; controls and game telemetry must not select actions.
- Use direct Objective-C++/Metal with macos-metal2.4, no fast math, macOS 13 minimum, and conservative Apple Silicon features; the M4 Pro is the initial validated target.
- Preserve failed controls and original numerical gates; bitwise agreement, changed weights, survival and engineering throughput do not establish learning or biological validity.
- Preserve user edits and unrelated workloads; credentials and machine-specific origins remain ignored; AGENTS.md and third-party notices remain unchanged.
- Use Sergey Subbotin <ssubbotin@gmail.com> for commits; avoid AI attribution and reverse-engineering wording in code, comments and commit messages.

## Interfaces and boundaries

Execution ABI becomes 8; numerical parent/order stay unchanged.
Public Graph, State, KCEvent and Timing layouts stay unchanged.

Add `df_metal_create_batch_windowed(const DFMetalGraph *, const char *,
int32_t lanes, int32_t window_ticks, DFMetalHandle *)`.
The original create/create_batch entry points delegate to window_ticks=0.
Reject native window_ticks outside 0..18 before traversing graph pointers.
Creation selects an immutable schedule for the handle lifetime.

Python adds `MetalBatchExecutor(brains, *, window_ticks=0)`.
Accept integral values 0..18, including NumPy integers; reject bool and
nonintegral values before touching brains or native allocation.
Mode 0 keeps the reference two-dispatch-per-tick schedule. Modes 1..18
use windows up to that capacity; advance still accepts 1..100 ticks,
including 19, split into windows. Changing mode requires a fresh owner.
Metadata records selected scheduler, configured window_ticks, ABI,
unchanged numerical order, lanes and actual native buffer accounting.
No new sensory/policy/plasticity implementation is introduced.

## Atomic prelude

In the reference gather stage, load touched first and exchange only
when nonzero. For each owned incoming word mask, load its masked bits
and fetch-and only when nonzero. Keep populated processing and float
presence flags unchanged, including zero-weight/cancelling contributions.
Completed mark dispatch precedes gathering; touched has one consumer;
incoming row masks are disjoint and other consumers only remove bits.
Preserve a separate guard-only commit for matched diagnostic timing.
Guards are behavior-preserving optimizations: baseline characterization
and exact equivalence are honest tests; missing guards cannot constitute
a semantic failing test.

## Delay proof and chronology

Chosen delay is 18 ticks, with 19 physical ring slots. At window start c
and length W<=18, arrivals at c..c+W-1 were scheduled before c.
A spike first emitted at c arrives at c+18, outside the window.
W=19 violates this proof despite using distinct physical slots.

Each window uses two ordinary serial tracked-resource dispatches:

1. Preparation visits (time offset, ring word, lane), atomically exchanges
   each consumed old ring word to zero, and enumerates every released
   outgoing edge of each old member. It marks time-specific touched
   targets and incoming-position bitmaps.
2. A neuron/lane worker iterates time offsets chronologically. At each
   tick it performs exactly the original active evolution/fire/active
   decision, deterministic ascending incoming gathering/modulation, then
   live future-ring membership reset. Inactive cells remain lazy.

Drive-change runs once before the existing advance bin. Materialization
runs once after it. Eligibility, sparse weights and host plasticity run
at their existing boundaries, never once per window.

Preparation exclusively clears consumed ring words. Workers never clear
current slots. At offset 0 the future slot is outside preparation and
valid prefilled membership must reset even an inactive nonfiring neuron.
At later offsets the future physical slot aliases an already consumed
old slot; new atomic OR bits must survive. Reset uses live membership,
not a fired-only flag. Incoming word consumption stays masked atomic
fetch-and; no plain read races with neighboring row consumers.

## Scratch and dispatch accounting

Private Params appends uint window_ticks and uint window_capacity at
offsets 88 and 92 (size 96), identically in C++ and Metal.
window_ticks is the actual remainder length; window_capacity is allocated W.
Reference mode has no additional scratch. Window mode retains ordinary
planes and adds W touched uint planes and W incoming bitmap planes per lane.
At W18 the added unaligned bytes are 69,564,024 per lane.

Check every product, actual MTLBuffer.length, maxBufferLength,
recommended working set and sparse-replacement peak before allocation.
Creation failure releases owned resources. Upload/reset/recovery clear
every scratch plane only for that lane; command failure poisons ownership
under the existing recovery contract. Each consumer restores its used
scratch to zero; remainder/unconsumed/padding bytes stay clean.

Preparation grid is words*actual_window_ticks by lanes; worker grid is
neurons by lanes. One tracked encoder/command buffer remains.
Timing records actual dispatched grids rather than reusing dense
reference counters: mark_grid_threads counts preparation work in window
mode, gather_grid_threads counts chronological worker invocations.
Reference counters keep their old meaning. dispatch_count is
2+2*ceil(steps/W), versus 2+2*steps in mode 0. At steps100,W18 this
is 14 versus 202; it is scheduling arithmetic, not a speed forecast.

## Validation and publication

Test W1/W2/W18, all 19 cursor residues and lengths
1,2,17,18,19,35,36,37,99,100 against fresh reference-mode handles.
Compare every state field bitwise, canonical lane-tagged events,
queues, cursor, counts, double eligibility and observations. Cover
prefilled inactive future membership; reused physical slots; two
neighbor incoming rows sharing a word; zero/cancelling/signed-zero fast
contributions; zero-weight modulation; lazy inactive evolution;
gather-before-reset; duplicates/self edges; dirty scratch reuse;
lane isolation; diagnostics overflow, poison and recovery.

Prove fixture sensitivity using deliberately defective candidate arms
that remove prefilled reset, clear future bits after workers, replace
masked word consumption, seed sums from existing g, suppress zero
presence, or eagerly evolve inactive cells. Run each defect separately
on the real target. Each must fail its named comparison while the
unmodified reference passes; preserve ineffective attempts honestly.
These temporary builds/tests remain ignored and never replace production
goldens. Test helpers may inspect completed scratch through a test-only
bridge compiled outside the production ABI; no unsafe public pointer API.

Run portable tests and actual M4 native/owner suites. Retain unchanged
40/80ms CPU parity gates and known failed epoch-5 long-CPU/epoch-6 controls.
Run six N2 and six N4 own-reference/isolation trials, all eight original
complete RGB controls and numeric checkpoints, then three fresh matched
reference/candidate pairs with complete 982 training and 988 held frames.
Separate warmup, use identical inputs/configuration, include every
configured lane in throughput, and reject interrupted/competing runs.
Record source/build/input/raw evidence hashes and actual memory.

Guard-only diagnostics precede W1/W2/W18 matched diagnostics. Full
acceptance measures the fastest exact candidate. Keep the default at
mode 0 until evidence and review justify a separate explicit decision.
No arbitrary 2x-CPU gate parks the work. Continue toward faster honest
aggregate neural training; 1000x real time remains aspirational.

Publish bounded portable evidence and code-matched documentation.
Do not give public launch approval. Follow-on science needs independent
demonstrations/seeds, declared outcomes/uncertainty, no-teacher/shifted/
frozen/erased/retention controls and teacher-free frozen live hazard transfer.
The existing stationary APPO demonstration pair is an engineering fixture.

## Supporting rationale

[NEST minimum-delay scheduling](https://nest-simulator.readthedocs.io/en/stable/nest_behavior/running_simulations.html)
supports delay-based partitioning in simulators. Applying it to these
Metal kernels is an inference requiring the tests above.
[Apple resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)
supports the conservative tracked-resource dispatch boundary.
Optional supported hardware-counter queries can refine later profiling;
their absence does not delay window implementation.
