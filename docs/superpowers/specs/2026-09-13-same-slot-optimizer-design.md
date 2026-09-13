# Same-slot Metal optimizer design

## Purpose and authority

Implement a small external numerical efficacy fitter on the continuing Metal
backend. Standing user approval covers this design, its plan, owned commits and
private training. This experiment is insufficient to establish biological fly
learning, general Doom competence or public-launch readiness.

The council used independent Codex, Grok CLI and Claude CLI opinions, anonymous
cross-review and chairman synthesis. Requested CLI aliases were not independently
verified as effective model IDs. Main source inspection corrected all council
budgets for obligatory four-lane padding. GA populations and CatBoost loss
surrogates are deferred: this budget favors direct paired neural measurements.

## Fixed model and data

Retain MaleCNS v1.0:166,700 modeled neurons and all25,582,938 released connections.
Only the existing4,184 positive KC→MBON11 slots vary. Sensory mapping, calibrated
dynamics, eta=.001, modulation and fixed DNp20/DNpe017 decoding remain unchanged.
Live pixels drive neural activity; offline labels score already-decoded actions.
No telemetry or demonstrator control selects runtime actions.

Use the pinned publisher revision in the companion protocol. Complete training
indices2/3 contain948/897 frames; held indices4/5 contain1036/990. Their respective
scenario seeds are200000046/055/051/067. Mechanical selection occurred before
new control rows/model outputs were read. Publisher metadata, including quality
scores, had been inspected. Subsequent validation checked hashes, schemas,
control contracts and exact35Hz PTS without neural scoring; training videos
decoded completely. Historical indices0/1 remain historical without reruns.
These are machine APPO demonstrations, outside the custom hazard arena.

## Optimizer

Four runs: Rademacher SPSA and antithetic Gaussian ES, each at optimizer seeds
200000201/202. NumPy PCG64 with SeedSequence([replica_seed,method_index]),
method_index0=SPSA/1=ES, provides explicitly separated streams. Preserve actual
draws, hashes and software versions. Static movement controls independently use
PCG64 seeds200000301/302, uniform efficacy fractions[.9,1.1], without selection.

Start theta=0. Fractions f=1+.1*tanh(theta); this chosen numerical neighborhood
lies within the permitted[.1,2] box. Four fixed generations use probe scales
[.50,.35,.25,.20] and maximum latent steps[.20,.15,.10,.10].
One direction per update, probes theta±c*d. Loss is the equally weighted mean
of canonical whole-episode balanced angular MAE on both training episodes,
deadzone.1 degrees. Compute g=(Lplus−Lminus)*d/(2c) for both distributions.
Gaussian ES multiplies by d, never coordinate reciprocals. Propose
theta−a*g/max(abs(g)); exactly zero norm leaves theta unchanged. Reject
nonfinite arithmetic. No clipping, redraw, arbitrary floor or scale enlargement.

Score proposals on the same complete training bank. Accept strictly lower mean
against the cached incumbent; ties retain incumbent. Probes are never accepted
as incumbents. Even zero-update proposals replay cached incumbents in their
scheduled four-lane wave. Freeze baseline, all four terminal run vectors and
both random vectors before any held neural scoring. Report every vector on
both held episodes; held losses never choose updates, schedules or a winner.

## Resident execution and budget

Reuse one four-lane immutable graph allocation, independent mutable states and
existing full replay. No active-lane mask or backend rewrite. Initial baseline
waves test exact cross-lane traces and all24 checkpoint arrays on both training
episodes before using cached losses. Rotate method/replica scheduling by fixed
generation order while maintaining exact executor lane identity.

Count every started full-graph lane attempt, including darkwarm, incomplete
runs, validation and padding. Before replay, persist the four-attempt charge.
Planned:8baseline +64probes +32proposals +16held =120. Held evaluation uses two
four-lane waves per episode, with one extra baseline filler whose evidence is
preserved. Random controls have no training objective. Reserve8attempts;
absolute cap128. No automatic retries or extra search. Incomplete banks cannot
produce accepted updates. Preserve at least16held attempts before optional
validation. Every scheduled attempt includes2s frozen darkwarm.

Precheck15GiB free disk, monitor disk and existing25%-free/no-growing-swap
pressure guards at every wave. Roughly13GB checkpoint storage plus traces is
material; do not accumulate device generation buffers. Use utility/PID-scoped
caffeinate-is on AC. Report epoch wall, active counters, per-lane brain time,
warmup and sleep separately; no throughput forecast.

## Checked installation and failure handling

Keep initialize_efficacies rejecting owned brains. Add serialized
MetalBatchExecutor.install_efficacies(lane,fractions,expected_edges).
Validate lane ownership/bindings, exact eligible edge order, finite positive
baseline, fractions[.1,2], u/w shapes and positive finite float32 materialized
weights before mutation. Set u/w=f−1 and update only eligible host/native weights
between rollouts. Do not materialize stale fast state or claim it became valid.
Native installation failure poisons the executor and aborts.

Existing replay resets fast state with keep_memory=True, reinstalls candidate
weights, sets learning=False/frozen=True and uses zero reinforcement throughout
darkwarm and full RGB. The biological-proxy rule stays unchanged and inactive;
neural state evolves. Check frozen u/w/fullweight bytes and nonplastic invariance,
zero teacher input and absence of within-rollout sparse plastic writes separately
from required between-rollout/reset uploads. Preserve all14 readouts and controls.

Validate physical source/build/graph/configuration pins and all24 baseline
checkpoint arrays once before native work. Preserve complete traces/checkpoints,
candidate theta/fractions/draws, decisions, budget and partial evidence; secondary
cleanup/writer errors must not replace the original interruption. Retain last
validated incumbent files; never continue a half-installed candidate. Runtime
origins stay ignored; third-party notices stay beside private data.

## Verification and interpretation

Pure tests cover estimators, finite bounds, exact-zero/cancelled gradient,
strict acceptance, seed separation and four-lane budget accounting. Portable
and native fixtures cover rejection-before-mutation, ownership, baseline
round-trip, stale-state validity, poisoning, frozen replay and interruption
preservation. Use existing parity evidence; do not rerun old scientific matrices.

Report paired held differences, per-episode outcomes and optimizer-replica
variation without frame-independent uncertainty, significance or held-derived
thresholds. Mixed, flat and adverse outcomes remain evidence. A useful numerical
fit can motivate a separately frozen live closed-loop Doom study. Public launch
still requires exact published How it works and its separate scientific gates.

Sources: [SPSA](https://www.jhuapl.edu/SPSA/),
[Salimans etal2017](https://arxiv.org/abs/1703.03864),
[pinned dataset README](https://huggingface.co/datasets/Yunncheng/gamewam-vizdoom/blob/6cc00d60462885c1d61dd480228af58c3b81b807/README.md).
