# Same-slot Metal Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit only existing eligible efficacies with bounded SPSA/Gaussian ES, preserving full neural control and honest held evidence.

**Architecture:** Pure optimizer math is independent of native execution. A serialized resident-lane installer changes eligible efficacies between existing frozen full replays; a bounded driver owns scheduling, evidence and failures. The controller runs the reviewed private M4 study, then a fresh worker documents measured outcomes.

**Tech Stack:** Python/NumPy PCG64, existing C++/Objective-C++/Metal2.4 Apple Silicon executor, pytest, strict GameWAM OfflineEpisode.

**Spec:** `docs/superpowers/specs/2026-09-13-same-slot-optimizer-design.md`

## Global Constraints

- Retain MaleCNS v1.0:166,700 modeled neurons and all25,582,938 released connections.
- Only the existing4,184 positive KC→MBON11 slots vary. Sensory mapping, calibrated dynamics, eta=.001, modulation and fixed DNp20/DNpe017 decoding remain unchanged.
- Complete training indices2/3 contain948/897 frames; held indices4/5 contain1036/990. Historical indices0/1 remain historical without reruns.
- Publisher metadata, including quality scores, had been inspected. Selection preceded reading new control rows/model outputs; later artifact validation never tunes candidates.
- Four fixed generations use probe scales[.50,.35,.25,.20] and maximum latent steps[.20,.15,.10,.10]. Fractions f=1+.1*tanh(theta), theta=0 initially.
- NumPy PCG64 with SeedSequence([replica_seed,method_index]), method_index0=SPSA/1=ES. Optimizer seeds200000201/202; static random seeds200000301/302 uniform fractions[.9,1.1].
- Loss is the equally weighted mean of canonical whole-episode balanced angular MAE on both training episodes, deadzone.1 degrees. Strict proposal improvement; ties keep incumbent; probes never accepted directly.
- learning=False/frozen=True and zero reinforcement throughout darkwarm/fullRGB. Biological-proxy rule stays unchanged and inactive; neural state evolves.
- No active-lane mask or backend rewrite. Count all4resident lanes, validation, padding and incomplete attempts.120planned+8reserved, maximum128;16held attempts remain protected.
- Every scheduled attempt includes2s frozen darkwarm. Precheck15GiB disk; existing25%-free/no-growing-swap pressure guards at each wave.
- Preserve complete traces/all24CP, candidate theta/fractions/draws, decisions, budget and partial evidence; secondary cleanup/writer errors never replace primary interruptions.
- Runtime origins stay ignored; third-party notices stay beside private data. No public launch, shared-main mutation, broad refactor, old scientific/performance matrix rerun or learning/speed guarantee.
- Work only in the existing owned isolated worktree. Apply_patch for local edits, Sergey Subbotin <ssubbotin@gmail.com> for commits. Workers are not alone and must preserve others' edits; no worker/reviewer subagents.

## Verification environment

Controller-provided ignored runtime configuration supplies existing interpreter/kernel paths and M4 origins. Do not reinstall dependencies. For portable tests use:
`env PATH=/usr/lib/llvm-21/bin:/usr/bin:/bin OPENBLAS_NUM_THREADS=1 DOOM_KERNEL_PATH="$TASK_KERNEL" "$TASK_PYTHON" -m pytest ...`.
The controller supplies TASK_KERNEL/TASK_PYTHON per dispatch. Preserve RED/GREEN raw logs in this plan's ignored workspace. Baseline dependency warnings are recorded debt, never silently suppressed.

---

### Task 1: Pure optimizer and attempt accounting

**Files:**
- Create: `doom_learning_v6/same_slot_optimizer.py`
- Create: `tests/test_doom_learning_same_slot_optimizer.py`

**Interfaces:**
- Consumes: pinned protocol JSON in `docs/experiments/2026-09-13-same-slot-optimizer-protocol.json`, SHA256 `fae8309b51cbc24b63b23fcfaf9c4a5e5f5d141a0ce060bee3d2913821fe9c03`.
- Produces: `load_protocol(path, expected_sha256) -> dict` (fresh checked record); `fractions(theta) -> float64 ndarray`; `optimizer_rng(method, seed) -> Generator`; `draw_direction(method, rng, slots) -> float64 ndarray`.
- Produces: `propose(theta, direction, plus_loss, minus_loss, *, probe_scale, step_length) -> (float64 ndarray, dict)`; record contains contrast, gradient_norm and zero_update.
- Produces: `training_mean(losses) -> float` requiring exactly2finite episode losses; `accept_proposal(incumbent_loss, proposal_loss) -> bool`.
- Produces: `AttemptBudget(maximum=128)` with used:int, remaining:int and `charge(lanes=4, *, reserve_after=0) -> int`. Charge rejects bool/noninteger/invalid/over-cap/reserve exhaustion before changing used.

- [ ] **Step 1: Write behavioral failing tests with hand-derived expectations.**

```python
def test_gaussian_direction_is_multiplied():
    theta, record = propose(np.zeros(2), np.array([.5, 2.]), 4., 2.,
                            probe_scale=.5, step_length=.2)
    np.testing.assert_allclose(theta, [-.05, -.2])
    assert record["gradient_norm"] == 4.

def test_exact_zero_keeps_candidate():
    theta, record = propose(np.array([.1, -.2]), np.array([1., -1.]), 3., 3.,
                            probe_scale=.5, step_length=.2)
    np.testing.assert_array_equal(theta, [.1, -.2])
    assert record["zero_update"] is True

def test_budget_preserves_held_capacity():
    budget = AttemptBudget()
    budget.charge(112, reserve_after=16)
    with pytest.raises(ValueError):
        budget.charge(4, reserve_after=16)
    assert budget.used == 112
```

Also test finite mapping on large latent values; negative/nonfinite/empty/mismatched arrays and arithmetic overflow; strict loss ties; two-episode mean; independent/reproducible method streams; wrong protocol SHA/altered protocol; final120-charge capacity with8remaining and direct over-capacity rejection. Static-control generation and zero-update/no-refund behavior are tested at their actual driver consumer in Task3.

- [ ] **Step 2: Run focused tests before source creation and retain authentic RED output.**

Run portable pytest on the new test file. Missing new feature is expected; fix import/fixture mistakes before accepting RED.

- [ ] **Step 3: Implement minimal pure functions and budget class.**

```python
# Validate finite equal nonempty vectors and positive finite c/a first.
g = (plus_loss - minus_loss) * direction / (2 * probe_scale)
norm = float(np.max(np.abs(g)))
candidate = theta.copy() if norm == 0 else theta - step_length * g / norm
# Reject nonfinite intermediates/results; no clipping, floor or adaptive probe.
```

load_protocol checks exact pinned bytes/SHA and the committed protocol structure, returning a fresh record; no configuration knobs or native imports. Optimizer RNG uses the declared method index and seed derivation. Validate scalar types, including boolean rejection where integers are required.

- [ ] **Step 4: Run focused GREEN, then ordinary full portable suite once; self-review.**

Run the new test file, then `-m pytest -q`. Include command, raw paths, expected RED reason, actual GREEN counts/warnings and changed files in the ignored task report.

- [ ] **Step 5: Commit only owned production/tests.**

`git add doom_learning_v6/same_slot_optimizer.py tests/test_doom_learning_same_slot_optimizer.py`;
`git commit -m "feat: add bounded same-slot efficacy optimizer"`.

### Task 2: Serialized eligible-efficacy installation

**Files:**
- Modify: `doom_learning_v6/metal/batch.py`, add one public installer beside advance/validation methods.
- Create: `tests/test_doom_metal_candidate_install.py`

**Interfaces:**
- Consumes: existing serialized executor operation, immutable circuit edge identities, positive baseline_plastic, lane backend.update_weights.
- Produces: `MetalBatchExecutor.install_efficacies(lane, fractions, expected_edges) -> None`; fractions are baseline-relative float64 vector, expected_edges exact ordered int64 canonical circuit ids.
- Existing initialize_efficacies ownership rejection, advance/reset/restore and native ABI remain unchanged.

- [ ] **Step 1: Add real fixture rejection/round-trip tests first.**

Use existing small MemoryBrain/VisualMemoryBrain fixture patterns in tests/test_doom_metal_batch*.py; native cases skip only absent Metal. Portable tests can replace only the native call below real executor validation, with explicit argument/poison expectations.

```python
# Fixture supplies real owned executor b/ex and exact eligible ids.
before = b.memory_u.copy(), b.memory_w.copy(), b.weight.copy()
with pytest.raises(ValueError):
    ex.install_efficacies(0, [np.nan] * len(b.circuit["edges"]),
                          b.circuit["edges"].copy())
np.testing.assert_array_equal(b.memory_u, before[0])
np.testing.assert_array_equal(b.memory_w, before[1])
np.testing.assert_array_equal(b.weight, before[2])
```

Cover wrong/duplicate/reordered/ineligible slots, bool/out-of-range lane, malformed bounds/shape, baseline NaN/Inf/nonpositive, float32 overflow/underflow, changed owner/bindings, closed/poisoned/in-flight executor, exact baseline round-trip, nonplastic bytes and unchanged stale-state validity. Native failure must poison and retain primary exception; installer does not execute neural ticks.

- [ ] **Step 2: Run authentic RED focused suite before implementation.**

Run new fixture tests using portable environment; at least installer behavior must fail because missing, not fixture construction.

- [ ] **Step 3: Implement a single serialized checked boundary.**

```python
with self._operation():
    self._validate_bindings()
    # Validate exact lane, registered memory shapes, ordered expected_edges,
    # positive finite baseline, finite bounded fractions and f32 weights first.
    memory = fraction_values - 1.
    values = (brain.baseline_plastic.astype(np.float64) * fraction_values).astype(np.float32)
    brain.memory_u[:] = memory
    brain.memory_w[:] = memory
    brain.weight[edges] = values
    brain.backend.update_weights(edges, values)
```

Do not temporarily remove ownership, call unowned initializer, upload stale full state, alter _host_state_valid, release/recreate executor or modify native kernels. Checked native failure poisons; document half-installed state requires abort/explicit restore.

- [ ] **Step 4: Run focused GREEN and ordinary portable suite; retain logs and self-review.**

Native fixture validation is performed by the controller once on M4 after source commit, before study. No full-connectome episode needed for unit ownership checks.

- [ ] **Step 5: Commit owned boundary/tests.**

`git add doom_learning_v6/metal/batch.py tests/test_doom_metal_candidate_install.py`;
`git commit -m "feat: install checked efficacies between Metal rollouts"`.

### Task 3: Bounded evidence-preserving resident driver

**Files:**
- Create: `doom_learning_v6/same_slot_runner.py`
- Create: `tests/test_doom_learning_same_slot_runner.py`

**Interfaces:**
- Consumes: all Task1 APIs, Task2 installer and existing causal_pilot.replay_episode.
- Consumes: strict OfflineEpisode, calibrated_brain, _Resources/_PressureHistory/_phase_boundary, _physical_pins/_validate_expected_pins, _model_gate/_readouts, _compare_checkpoint_arrays/_same_without_wall and source/lock/failure helpers where applicable.
- Produces: `fit(executor, train, held, readouts, protocol, out, *, replay=replay_episode) -> dict`, allowing only replay injection for deterministic portable scheduling/failure tests.
- Produces CLI: `python -m doom_learning_v6.same_slot_runner --protocol FILE --protocol-sha256 SHA --train DIR --train DIR --eval DIR --eval DIR --reference DIR --expected-pins FILE --source-commit SHA --out FRESH_DIR`.
- Produces pinned protocol/inputs/provenance, persistent attempt ledger, all wave/lane traces/all24 terminalCP, per-generation theta/fractions/draws/contrasts/acceptances, frozen seven-final-candidate manifest, before/after pins and results.json. Public claim flags false.

- [ ] **Step 1: Write portable scheduling, selection and failure tests first.**

Small explicit replay double supplies complete lane records with finite hand-derived losses. Its native work is substituted; scheduling/optimizer/budget/artifact writes remain real.

```python
# Controlled complete-bank losses: baseline[3,5]=>4; proposal[2,7]=>4.5.
assert training_mean([3., 5.]) == 4.
assert accept_proposal(4., training_mean([2., 7.])) is False
# Driver test observes actual retained theta/fraction files, not only mock calls.
# Exactly30scheduled four-lane waves =>120 charged lane attempts.
# Seven distinct final roles +two baseline fillers appear on both held episodes.
```

Add tests proving held is invoked only after final-manifest creation, every probe/proposal uses BOTH training episodes, labels never affect runtime controls, generation acceptance/ties/zero_update are correct, and an actual zero-gradient driver wave retains its consumed four-attempt charge. Verify actual saved static-control vectors use the declared independent seeds and uniform bounds, randoms have no training scores, altered identities/counts/protocol/source/pins reject before GPU, and failure/KeyboardInterrupt preserves chargedbudget/lastcenter/partial records without swallowing original or cleanup errors.

- [ ] **Step 2: Run new focused RED and preserve expected failures.**

No fullgraph run; do not dispatch old pilots or alter fixture expectations to hide missing behavior.

- [ ] **Step 3: Implement the thin resident harness and preflight.**

```python
# Each wave validates four candidates, reserves budget, writes charge BEFORE
# any install/reset/start, installs in exact executor lane order, captures
# expected memory/weights, and reuses complete existing replay:
phase = replay(executor.brains, executor, episode, readouts,
               np.zeros((4, episode.frame_count)), learning=[False]*4,
               frozen=[True]*4, directory=wave_dir, warmup_ms=2000)
# Materialize/check full frozen weights +u/w and unchanged nonplastic/config.
# Required reset/init uploads precede warmup; no sparse writes inside RGB.
```

Schedule baseline4lanes×2episodes; for eachgeneration two4laneprobe groups
covering allfourruns± on eachtrainingepisode, then4proposal lanes on both.
Fixed rotate run order by generation, maintain run→lane mapping in evidence.
Baseline cross-lane fulltrace/all24CP comparisons precede cached scores.
Train-bank strict acceptance updates saved centers only after BOTH completes.
Persist actualdraws as numerical arrays plus hashes.
Draw/pin both randoms independently, then freeze baseline+4terminals+2randoms.
Held wave1 is first4roles; wave2 remaining3+baseline filler, bothepisodes.
Allattempts including fillers/no-update waves charged; no retry or masking.

CLI requires Darwin, exact clean HEAD, immutable committed protocol SHA and
trusted physical pins before native allocation. Validate source/CPU/native
relationships using existing pin helpers, graph/config/calibration and initial
all24CP baseline. No historical replay/_check_reference old episode requirement.
Verify fresh split's complete lengths/identities/source contract. Acquire existing
nonblocking GPU lease. Register resources immediately; pressure/disk checks at
everywave; preserve failures and releaselease without replacing primarycause.
Compare frozen fullweights after materialization andu/w bytes, fixed14readouts/
decoder andnonplastic/config invariants. Preserve currentrepo defaultW0; driver
explicitW18 only. CLI writes allraw/private origins only under ignored freshout.
Keep files focused; no copied replay implementation or new native buffers.

- [ ] **Step 4: Run focused GREEN, ordinary portable suite and self-review.**

Report exactcommands/rawlogs and invariants; controller deploys reviewed source
to ownedM4 and runs nativefixtures before allocating study. Do not starttraining
from worker or make public/machine configuration changes.

- [ ] **Step 5: Commit driver/tests.**

`git add doom_learning_v6/same_slot_runner.py tests/test_doom_learning_same_slot_runner.py`;
`git commit -m "feat: run bounded resident Metal efficacy fitting"`.

### Task 4: Execute reviewed study and document actual evidence

**Files:**
- Create: `docs/experiments/2026-09-13-same-slot-optimizer.md`
- Create: `outputs/doom-learning/same-slot-optimizer-m4pro/results.json` (compact audited summary only).
- Preserve: all private native raw data/logs/checkpoints/failed/control attempts and this plan's ledger.

**Interfaces:**
- Consumes: reviewed committed CLI/installer/math, pinned complete fourepisodes,
trusted graph/build/initialCP pins and ignored runtimeconfiguration.
- Produces: actual method/replica trainacceptance and allheldpairedoutcomes,
artifact-role SHA/bytes audit, exact tested source revision, validation/timing/
pressure/budget evidence and scientifically qualified public report.

- [ ] **Step 1: Controller fast-forwards only ownedM4 mirror; validate physicalpins and focused nativefixtures.**

Preserve generatedCPU manifests/nativeproducts and usercheckout/unrelatedjobs.
Use existing M4env and acquire scoped lease for anynativevalidation. Focused
pytest runs Task1/Task2/Task3newfiles plus existing causalreplay fixture tests
once; ordinary fullnative suite once before finalhandoff. Fullgraph validation
attempts, ifneeded, use fourlane waves within8reserve and do notrerun oldmatrix.

- [ ] **Step 2: Controller launches exact pinned private study using utility-scoped caffeinate-is onAC.**

Use CLI signature fromTask3 with ignoredconfigpaths/exact committedsourceSHA/
protocolSHA and controller-validated expectedpins. Poll actualprocesshandle,
neverrestart from missing/empty file ortimeout. Count allattempts. Preserve
incompleteevidence; noadaptive scale/data/method retuning. Continue monitoring
with concise updates; unrelatedworkloads remain untouched.

- [ ] **Step 3: Controller audits actual terminal outputs and supplies fresh reportworker the compact evidence file.**

Verify completeframecounts/controlseparation/all24CProles/frozenmemory/
nonplastic/config/source/physicalpins and128cap. Check all7heldvectors and2fillers,
pairdifferences tobaseline perheldepisode and replicatevariation. Retainraw
nativeCPs onM4; nevercopy13GB into localtmpfs orbundle externalassets. Compact
JSON carries semanticroles/hash/bytes, no privateorigins orfullthirdpartydata.

- [ ] **Step 4: Fresh worker writes factual report and compact publicJSON using apply_patch.**

Report actualtrainchanges/acceptance, allheldprimary contrasts, diagnostic
forward/attack, source/buildvalidation, exactclocks/pressure/budget andremaining
assumptions. Numericalfit/changedweights neverestablishbiological/Doomlearning.
Humanreport earnsnoTDDtest; use structuredJSONparse/identity/metric checks against
controller's realaudit. Preserve negativecontrols/failures. No publiclaunch.

- [ ] **Step 5: Review task; final strongest whole-subproject review, one finalfixwave/one scopedrereview iffindings; verify and commit/push onlyownedbranch.**

Controller collects everychronological Ruling withcost into finalhandoff and
preserves privateworkspace under projectfailed/control-evidence constraints.
Do notmerge sharedmain/createPR/publishsite. Keep fulltraininggoalACTIVE unless
actualgoaloutcome independentlyestablished; currentpilot alone cannotachieveit.
