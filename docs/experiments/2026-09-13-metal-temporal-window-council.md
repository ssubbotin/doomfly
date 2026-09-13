# Council: exact Metal temporal acceleration

Design deliberation completed on 13 September 2026. This report proposes the
next experiment; it contains no measurement of an implemented window kernel.
The completed resident-batch comparison provides context: aggregate RGB
brain-seconds per wall-second are 0.15085 for current ABI7 interleaved serial,
0.22335 for two resident lanes and 0.37385 for four resident lanes. These are
execution rates, not learning-quality results or a forecast toward 1000x.

## Method and recorded calls

Three independent Stage1 opinions came from Codex (gpt-5.6-sol/high), Grok CLI
(observed primary grok-4.6/high), and Claude CLI (observed primary
claude-opus-5/high). Stage2 used the complete author/model-name-free opinion
pool, with one independent critique from each of those three models. Stage3
used gpt-6-astra/high for the corrected synthesis. All seven model calls
completed. The unavailable specialized research workflow was explicitly
replaced by this manual three-stage cross-model deliberation; no empirical
invention workflow or new hardware run was fabricated.

Anonymous A/B/C ranks were A>B>C, B>A>C and C>B>A. B received 4 Borda points,
A received 3 and C received 2. Rankings reveal disagreement; they do not prove
correctness.
Grok preferred blocking first. The other two preferred an atomic prelude.
The final synthesis selects an independently attributable atomic prelude and
requires blocking to proceed regardless of its timing outcome.

Four external CLI calls recorded zero assistant tool calls. Actual primary
answer models were verified; auxiliary usage remains separately recorded
(grok-4.6-build and Claude's Haiku/Opus accounting). CLI initialization can
still list tool schemas, so zero observed calls is the supported assertion.
Two earlier CLI setup errors occurred before model invocation and remain
preserved separately: a relative prompt path and an invalid empty MCP object.
They are not counted among the seven completed model calls.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Stage1 Grok raw stream | 99,891 | 38cbb9e327d4ac6276ef2a315c651bb57fe9b508e1a1778cc387f1ee440ac766 |
| Stage1 Claude raw stream | 92,816 | d0bbae5ed421c683de3f22c0d66b9ade6b7727fb92ccec3d17d3296e4857de43 |
| Stage1 Codex report | 9,881 | 211081a286c325f85becce61cc294e6103570f65c5850ba6b9c22164a9b0c9e2 |
| Stage2 Grok raw stream | 57,816 | 0362a7faef852f1bb8bccd6a140eb964b4c93f814cdd2d88f1f059b510258c67 |
| Stage2 Claude raw stream | 104,634 | 7f060645c3f79a33bfe48442bbf663067cf9059d2c915fc91b10e4fcc00aef75 |
| Stage2 Codex report | 5,710 | 7ec4fac646f68ad48f7aa08ec4b9f8b78711e4271a1da685d9eccaded8d54eaa |
| Final synthesis | 10,463 | f9d2e059c31e31e3663f2316ec8535f0823be111ea61a29e32694c963d3925d3 |

Raw prompts, streams, model usage and local configuration remain ignored.
The initial question SHA is 1d52b063d47fff0a5223b3b261210e558298aa23f5c722eb9fc3166f822307ca;
anonymous review-question SHA is bb7db70c2e9b62d1d26368d61853421917bf4b8c61c1a3caf23ccc787471443c;
synthesis-question SHA is 08778dea43dc03dc9bcd0905d33d721dd0c38aeaaf3063bb159b6980c3e87535.

## First intervention: avoid zero-result atomic consumption

Keep nonzero exchange/fetch_and bodies unchanged. First atomically load each
touched flag and bypass its exchange only when zero. In a separately identified
arm, atomically load an incoming word and bypass masked fetch_and only when
its owned mask is zero. There are no producers during gather after the ordinary
serial tracked-resource mark dispatch; touched entries have one consumer,
while neighboring incoming rows own disjoint masks and only remove bits.
The zero case therefore cannot acquire owned work before consumption.

Keep populated masks atomic and ascending. Whole-word stores can erase a
neighbor's contribution. Cancellation, zero weights and signed-zero sums
retain event presence and never authorize skipping populated masks. Ring clear
already uses atomic_store; it is outside this intervention. Extra loads and
branches can regress performance. Short diagnostics prioritize variants;
complete-workload acceptance remains a separate gate for any retained variant.

## Substantial intervention: bounded temporal windows

The chosen uniform delay is 18 ticks (1.8 ms), with 19 physical slots and dt=0.1 ms.
For a window of W<=18 ticks, all arrivals originate before the window. Preparation
can snapshot/clear each consumed old ring word, enumerate every released edge
and populate per-time incoming/touched planes. A neuron/lane worker then runs
the original active evolution, firing, ordered gathering and membership reset
chronologically. Preserve all-neuron coverage initially and the original
drive-change/final-settlement/host-learning boundaries.

Test W1, then W2, then W18 and remainders. W19 fails causal independence:
the first new fire arrives at its last tick, although its 19 slots are distinct.
At local tick 0, the future slot lies outside preparation and any valid prefilled
membership must trigger the parent reset. At later ticks, the future physical
slot is an already-consumed old slot; new worker emissions must survive.
Workers never clear current ring words. Future OR/read stays atomic.

Consume touched flags and owned incoming masks atomically back to zero. Keep
padding clean and explicitly clear scratch on initialization/reset/recovery.
Do not mix plain whole-word reads with concurrent atomic clears. Eighteen
incoming planes plus uint touched planes total 69,564,024 bytes per lane before
alignment (57,561,624+12,002,400). This is additional storage if original planes
remain. Actual MTLBuffer allocations, maximum lengths and recommended working
set must be checked. A 100-tick call uses 14 dispatches instead of 202; this is count
arithmetic, providing no speed or compiler register-residency promise.

The minimum-delay principle has precedent in
[NEST's simulation scheduling](https://nest-simulator.readthedocs.io/en/stable/nest_behavior/running_simulations.html).
Its use here requires independent adaptation and validation. Apple's
[resource synchronization guidance](https://developer.apple.com/documentation/metal/resource-synchronization)
supports ordinary serial tracked-resource dependencies. Preserve the portable
Metal 2.4/macOS 13/arm64/unified-memory/Apple7 feature floor.

## Evidence and continuation

Require deliberately defective sensitivity controls for stale flags, shared
word destruction, cancellation presence, fired-only reset, W2 slot wiping,
reordered gather/reset, eager inactive evolution and dirty-plane reuse.
An ordinary baseline pass is GREEN. A missed defect invalidates that fixture's
sensitivity claim; strengthen it and retain the failed attempt.

Then compare every named boundary field, weight, ring membership, cursor,
canonical spike/KC event and learned array against the reference scheduler.
Cover lengths 1/2/17/18/19/35/36/37/99/100, all 19 starting residues, valid prefilled
queues, empty/self/duplicate/zero/modulatory edges, refractory transitions,
capture modes, native validation/poison/recovery/cleanup and independent lanes.
Finish unchanged retained gates, six own-reference isolation trials, all eight
original controls and fresh matched three-repeat complete 982-train/988-held pairs.
Warmup is separate; no padded/completed lane or competing measurement job.

GPU interval divided by dispatch count cannot identify launch overhead,
barrier cost or a limiting kernel. Per-tick activity and degree-weighted work
cannot be recovered from aggregate counts/KC-only events. Runtime counter
[capabilities](https://developer.apple.com/documentation/metal/confirming-which-counters-and-counter-sets-a-gpu-supports)
and [sampling boundaries](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)
are a bounded adjunct. Encoder splitting is a different diagnostic schedule.
Counter absence never parks implementation.

If guards lack gain, continue blocking. If blocking lacks gain, choose a measured
smaller window or a separately proved compact set including all potentially
firing active cells, touched targets and required future-membership resets.
Preserve all failed sources and controls. This accelerates the chosen full
MaleCNS v1.0 model without establishing useful learning, biological validation,
long CPU equivalence or a public launch. The next interface/ABI choice belongs
to the implementation specification; the council does not validate an
unimplemented public API.
