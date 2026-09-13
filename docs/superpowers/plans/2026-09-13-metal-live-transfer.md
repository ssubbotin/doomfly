# Frozen Metal Live Doom Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an auditable fixed-weight closed-loop Doom comparison of every saved fitting candidate and control.

**Architecture:** Pure protocol helpers fix schedule, vector identity and gameplay-only metrics. Existing game wrappers gain safe construction and optional explicit timing; a new evaluator owns independent games over the unchanged resident Metal executor. The controller validates, executes and audits the private study before publishing qualified results.

**Tech Stack:** Existing Python/NumPy, pytest, ViZDoom 1.3.0, conservative C++/Objective-C++/Metal Apple Silicon executor W18.

**Spec:** `docs/superpowers/specs/2026-09-13-metal-live-transfer-design.md`

## Global Constraints

- Retain MaleCNS v1.0: 166,700 neurons, all 25,582,938 released connections and all 4,184 existing positive KC to MBON11 slots.
- Reuse `adaptive-centered-v6`, eta=0.001, original calibration and exact fourteen readouts/fixed BCI DNp20/DNpe017 decoder. No graph, neural rule, gains, native ABI/kernel or mapping changes.
- Immutable protocol: `docs/experiments/2026-09-13-metal-live-transfer-protocol.json`, SHA256 `757a424568027aea9852b80b75c50dc1b6ce2a66bb05d26e8ce53cf2497b1071`.
- All seven candidate SHA/bytes are fixed by the protocol. The original source-CEC fitting study and its 120 attempts/results remain immutable; no winner selection or further fitting.
- Learning=False, weights_frozen=True, zero external reinforcement during 2,000 ms dark warmup and every live/padding interval. Endogenous DAN may be nonzero. Bypass DamageTraining entirely.
- RGB24 640x480, 35 Hz, 0.1 ms neural steps; frame boundaries `round((index+1)*10000/35)`. Advance all four lanes, W18, one game action per active lane/tic.
- Horizon 2,100 tics (60 s), requested game episode start0, effective ViZDoom start1. Record actual initial engine tic/health and use actual tic differences. No action or respawn after termination; dark neural padding is labeled and excluded from gameplay metrics.
- Two four-lane calibration waves plus sixteen four-lane evaluation waves: planned 72, reserve 8, cap 80. Charge four before installation/reset/rollout, including incomplete work and eight baseline fillers. No automatic retries or reserve use.
- Primary hazard restricted survival; source-scenario fixed-horizon kills/survival separately. Preserve every role/seed/baseline contrast/replica variation, no pooling or frame-independent significance.
- Fresh private/ignored output, trusted source/graph/CPU/native/reference/candidate/engine pins, 15 GiB free disk, 25%-free/no-growing-swap pressure, nonblocking GPU lease and AC-scoped caffeinate-is.
- Preserve exact primary BaseException identity, accumulated traces, charges and independent healthy evidence through cleanup/writer/materialization failures. Known failed materialization has no repeated checkpoint/sync.
- Owned worktree/branch/mirror only; preserve user checkout, generated CPU/native products, failed/control evidence and third-party notices. No public broadcaster/site/launch, shared-main writes or external asset bundling.

## File responsibilities

- `doom_learning_v6/live_controls.py`: pure strict protocol, immutable vector loading, wave planning and gameplay metrics.
- `doom/game.py`: optional pre-init timeout/start settings and exception-safe native construction, existing defaults preserved.
- `doom_learning/survival_arena.py`: close an engine if its existing constructor fails; existing map rules/settings unchanged.
- `doom_learning_v6/live_transfer.py`: independent live-game ticking, checked frozen setup, evidence/failure/resource contracts and exact-source CLI.
- Focused new tests mirror these responsibilities; existing model/backend tests remain controls.
- `docs/experiments/2026-09-13-metal-live-transfer.md` and compact `outputs/doom-learning/metal-live-transfer-m4pro/results.json`: actual audited outcomes only after execution.

### Task 1: Exact protocol, frozen vectors, wave schedule and metrics

**Files:** Create `doom_learning_v6/live_controls.py`; create `tests/test_doom_live_controls.py`. Existing committed protocol/spec are read-only. You are not alone; preserve other edits. No children/SSH/GPU/scientific replay.

**Interfaces:**
- `load_live_protocol(path, expected_sha256) -> dict`: accept only immutable protocol bytes/SHA and return a fresh record.
- `validate_live_protocol(protocol) -> None`: recursively exact built-in JSON types/keys/values against the pinned protocol; reject type-equivalent aliases before work.
- `load_candidates(protocol, root) -> dict[str, np.ndarray]`: all seven relative files pinned by SHA/bytes, exact float64 shape(4184,), finite fractions[0.9,1.1], baseline all1; frozen read-only copies. Root is runtime private study directory.
- `plan_waves(protocol) -> list[dict]`: eighteen records with `ordinal`, `kind` (`calibration`/`evaluation`), `environment`, `seed`, `hazard_left` (bool or None), `roles` (four strings). Calibration first in environment order; each evaluation seed rotates seven roles by seed ordinal modulo7 before grouping, with `baseline-filler` as group2's fourth role. Filler resolves to baseline vector.
- `gameplay_metrics(trace, horizon_tics) -> dict`: require complete ordered horizon rows. Each row has `index`, `phase` (`live`/`padding`), `game_before`/`game_after` observer records (`tick`, `engine_tic`, `finished`, `dead`, `timeout`, `health`, `kills`), and `applied_action` (dict for live, None for padding). Validate contiguous one-tic active game advances, first terminal stops all later actions, padding retains terminal observer state. Return `live_game_tics`, `restricted_survival_seconds`, `died`, `right_censored`, `censor_reason`, `kills_at_fixed_horizon`, `damage_total`, `final_health`; use actual engine tic differences, never neural/padding duration.

- [ ] **Step 1: Write behavioral failing tests.** Load production module dynamically after a useful missing-module assertion. Cover recursively malformed protocol types/extra keys/wrong SHA or bytes; missing/altered/reordered-role candidate inputs, non-float64, wrong shape/nonfinite/bounds/baseline; eighteen exact wave records/72charges/eight fillers/both seed lists/rotation. Hand-built complete traces cover death then padding, living timeout, living horizon/other-finished censoring, damage/healing and cumulative kills, malformed ticking/postterminal actions and incomplete horizon rejection.

```python
def test_full_schedule_preserves_all_roles(protocol):
    waves = controls().plan_waves(protocol)
    assert len(waves) == 18
    assert sum(len(w['roles']) for w in waves) == 72
    assert [w['roles'] for w in waves[:2]] == [['baseline']*4]*2
    assert sum(role == 'baseline-filler' for w in waves for role in w['roles']) == 8
    for scenario in ['blue-floor-survival-v1', 'defend_the_center']:
        rows = [w for w in waves if w['kind']=='evaluation' and w['environment']==scenario]
        assert len(rows) == 8
        for seed in {w['seed'] for w in rows}:
            roles = [r for w in rows if w['seed']==seed for r in w['roles']]
            assert sorted(r for r in roles if r!='baseline-filler') == sorted(c['role'] for c in protocol['candidates'])
```

- [ ] **Step 2: Run authentic focused RED.** `python -m pytest -q tests/test_doom_live_controls.py`; preserve actual command/exit/full raw log. Missing module is acceptable initial RED, fixture errors are not.
- [ ] **Step 3: Implement the named pure interfaces.** Use hashlib/JSON/Path/NumPy only for identity/math, no native allocation. Protocol validation precedes candidate reads/planning; allow_pickle=False; reject absolute/escaping protocol file paths. Copy arrays before making them read-only. Emit the exact eighteen-wave schedule without scores or randomness. Strictly distinguish bool/int/float/container/key types.

```python
def plan_waves(protocol):
    validate_live_protocol(protocol)
    waves = []
    for env in protocol['environments']:
        seed = next(row for row in env['seeds'] if row['seed']==env['baseline_check_seed'])
        waves.append(dict(ordinal=len(waves), kind='calibration',
                          environment=env['scenario'], seed=seed['seed'],
                          hazard_left=seed.get('hazard_left'), roles=['baseline']*4))
    original = [row['role'] for row in protocol['candidates']]
    for env in protocol['environments']:
        for ordinal, seed in enumerate(env['seeds']):
            offset = ordinal % len(original)
            rotated = original[offset:]+original[:offset]
            for roles in [rotated[:4], rotated[4:]+['baseline-filler']]:
                waves.append(dict(ordinal=len(waves), kind='evaluation',
                                  environment=env['scenario'], seed=seed['seed'],
                                  hazard_left=seed.get('hazard_left'), roles=roles))
    return waves
```

- [ ] **Step 4: Run focused GREEN and self-review.** Run new file plus existing pure optimizer/control tests, no old scientific matrix/full-suite repetition. Verify data/schema/hand-derived outcomes, zero native work, immutable protocol SHA and git diff --check; retain raw output.
- [ ] **Step 5: Commit owned source/tests.** `git add doom_learning_v6/live_controls.py tests/test_doom_live_controls.py`; `git commit -m "feat: define frozen live Doom comparison contracts"`. Report actual Git-derived SHA, commands/exits/limitations under this plan's ignored workspace.

### Task 2: Safe, explicit game lifecycle

**Files:** Modify `doom/game.py`, `doom_learning/survival_arena.py`; create `tests/test_doom_game_lifecycle.py`. Own these only; not alone, no children/SSH/GPU/scientific study.

**Interfaces:** Preserve every existing positional/default Game call. Add keyword-only `episode_timeout_tics=None, episode_start_tics=None` to Game.__init__; strict nonnegative built-in ints when provided, reject before allocating DoomGame. Apply explicit values before init; original timeout/start semantics unchanged when None. Preserve SurvivalArena signature/map/settings. Both constructors close a created DoomGame upon any later BaseException, retaining exact original object if close fails. Normal close remains existing API.

- [ ] **Step 1: Write failing fake-engine lifecycle tests.** Capture calls/defaults and prove zero allocation for malformed options, explicit settings occur before init, defaults unchanged, init/first-new-episode failure closes exactly once, earlier configuration failures close too, and secondary close error adds a note while exact primary KeyboardInterrupt/Exception propagates. Test both wrappers without a real external asset dependency using existing monkeypatch patterns.

```python
def test_first_episode_failure_closes_without_masking_primary(monkeypatch, fake_engine):
    original = KeyboardInterrupt('first episode interrupted')
    fake_engine.new_episode_error = original
    with pytest.raises(KeyboardInterrupt) as caught:
        Game(episode_timeout_tics=2100, episode_start_tics=0)
    assert caught.value is original
    assert fake_engine.close_calls == 1
```

- [ ] **Step 2: Run focused RED.** `python -m pytest -q tests/test_doom_game_lifecycle.py`, preserve complete log and actual failure cause/exit.
- [ ] **Step 3: Implement minimal lifecycle changes.** Validate options before allocation; place post-allocation setup/init/new_episode under guarded cleanup. Do not change buttons, assets, hazard geometry, hidden buffers, spectator behavior or default public round timing.

```python
try:
    # Existing configuration also belongs inside this guarded boundary.
    if episode_timeout_tics is not None:
        self.game.set_episode_timeout(episode_timeout_tics)
    if episode_start_tics is not None:
        self.game.set_episode_start_time(episode_start_tics)
    self.game.init()
    self.episode=0; self.tick=0; self.episodes=[]; self.new_episode()
except BaseException as failure:
    try:
        self.game.close()
    except BaseException as secondary:
        failure.add_note('Secondary game constructor cleanup: '+type(secondary).__name__)
    raise
```

Keep all existing post-allocation configuration within the same guarded boundary.
- [ ] **Step 4: Run GREEN plus covering existing Doom/spectator/arena fixtures.** Execute real tiny Game and SurvivalArena checks where installed: request start0, effective configured/actual start1, one action advances one tic, actual final-minus-initial tic equals horizon, RGB24 dimensions/buffer isolation and deterministic same-seed initial hashes match. Include source default start10 preserved when settings are None. This is environment QA with explicitly programmed controls, zero neural attempts, no fly behavior claim. Preserve exits/assets/logs; no public server.
- [ ] **Step 5: Commit only owned files.** `git add doom/game.py doom_learning/survival_arena.py tests/test_doom_game_lifecycle.py`; `git commit -m "fix: make evaluation game timing and construction explicit"`; ignored factual report/raw logs with actual source identity.

### Task 3: Checked frozen live runner and CLI

**Files:** Create `doom_learning_v6/live_transfer.py`, `tests/test_doom_live_transfer.py`. Existing helpers are consumed, not modified; not alone, no children/deployment/scientific launch from worker.

**Interfaces:**
- Consume Task1's five exact interfaces and Task2's optional Game timing.
- `live_wave(brains, executor, games, readouts, *, horizon_tics, warmup_ms, directory) -> dict`: four ordered distinct executor brains/games, original14readouts; fresh phase; full indexed trace for every neural tic/lane; reset fast state/reinstall retained eligible slots, freeze and dark warmup; iterate active games independently/pad finished games; persist availability and all24 wave-final CPs with explicit postpadding scope.
- `evaluate(executor, candidates, readouts, protocol, out, *, game_factory) -> dict`: consumes already resident four-lane executor, seven immutable vectors, exact protocol, freshout and callable `game_factory(wave,lane,directory)` returning pixels/act/observation/close plus raw `.game` engine for authoritative tic/death/timeout observer fields. Owns durable charge, per-wave games/resources, checked installation/full frozen snapshots/epochs/resident checks, calibration canonicaltrace/all24 comparisons, complete-only metrics and all role results.
- `build_parser()`, `run(args)`, module main: require `--protocol`, `--protocol-sha256`, `--reference`, `--expected-pins`, `--source-commit`, `--candidates-root`, `--out`. Darwin/HEAD/sourceclean/committed-protocol/engine/vector/core/config/calibration/readout/all24 initial validation before native rollouts; original static reference only, never historical replay. Use existing calibrated CPU brain constructor, _restored_brains/registry and MetalBatchExecutor W18; physical before/after and pressure allocation-to-release; nonblocking scoped lease.

- [ ] **Step 1: Write failing integration/CLI tests through real functions.** Fake short games diverge at known death tics; serial CPU visual executor supplies real neural/decoder/checkpoints. Portable test protocol gates can be narrowly substituted while proving production preflight rejects altered literal/typed protocol before substitution or work. Native tiny four-lane visual fixture runs short real live_wave; real Doom smoke asserts RGB/act/tic boundary. Test all18wave schedule/charges before install, fixed/read-only vectors, zero learning/stimulation at every call, exactwarm/285-286 boundaries, no afterdeathpixels/act/respawn, requested-versus-applied/padding labels, complete-only metrics, frozenweights/u/w/epochs/+2setup-only uploads/resident bytes, calibration before metrics and failure preservation.

```python
def test_dead_game_pads_without_steering_or_masking(case):
    phase = live_wave(case.brains, case.executor, case.games, case.readouts,
                      horizon_tics=7, warmup_ms=2000, directory=case.out)
    assert all(len(row['trace']) == 7 for row in phase['lanes'])
    assert case.games[0].act_calls == 2
    assert case.games[0].pixels_calls == 2
    assert phase['lanes'][0]['trace'][2]['phase'] == 'padding'
    assert phase['lanes'][0]['trace'][2]['applied_action'] is None
    assert all(b.cursor == 20000+round(7*10000/35) for b in case.brains)
    assert all(call['learning'] == [False]*4 and call['stimulations'] == [None]*4
               for call in case.executor.rgb_calls)
```

Exercise render/act/constructor/install/terminal sync/CP/JSON/pressure/resource/lease errors, including primary KeyboardInterrupt and usable-owner terminal sync failure. Preserve traces/4charge/healthyall24/exactprimary through secondary failures; no repeated knownfailed sync/checkpoint. CLI late failures cannot leave results.complete=True or claim frozen success.
- [ ] **Step 2: Run authentic new focused RED.** `python -m pytest -q tests/test_doom_live_transfer.py`; preserve actual exits/raw logs and fixture validity.
- [ ] **Step 3: Implement the thin runner.** Use Task1 schedule plus existing AttemptBudget, `_Resources`, `_PressureHistory`, `_phase_boundary`, `_state_hashes`, `_write_phase`, `_physical_pins`, `_validate_expected_pins`, `_source_clean`, `_initial_reference`, `_restored_brains`, `_frozen`, `_storage`, `_compare_checkpoint_arrays`, `_same_without_wall`, `_readouts` where their actual interfaces fit. Never invoke recorded replay/teacher/DamageTraining. Snapshot game observers separately after decoding; controls receive only counts/time. Wave handler owns one terminal evidence attempt per lane; outer CLI writes failure metadata only, preventing a second materialization layer. For pre-wave install/constructor failure, guarded best-effort evidence occurs once without reset/upload/poison bypass.

```python
frames = [g.pixels() if active else black for g, active in zip(games, alive)]
counts, elapsed = executor.rgb_step(frames, steps*.1,
                                  learning=[False]*4, stimulations=[None]*4)
requested = [c.decode(count, steps*.0001) for c,count in zip(controls,counts)]
for lane, game in enumerate(games):
    if alive[lane]:
        game.act(requested[lane])
# Observer state determines only episode termination/padding, never requested controls.
```

Aggregate shared batch-advance/last_timing once per call, separate input/game/traceIO/materialization/checkpoint/warmup/live/padding envelope clocks. Save per-tic RGB/spikehash/readouts/memory/cursors/requested/applied/gameobserver fields and per-wave runtimefrozen/source/epoch/storage evidence; current/progress JSON is private and never a broadcast.
- [ ] **Step 4: Run focused GREEN/covering tests and ordinary portable suite once.** Include new three files plus existing causal replay/logger/installer/optimizer/control tests; preserve raw exits/knownwarning debt. Self-review exact contracts and gitdiffcheck; no old study/matrix/dependency/native rewrites. Controller later runs reviewed native fixtures before study.
- [ ] **Step 5: Commit owned runner/tests only.** `git add doom_learning_v6/live_transfer.py tests/test_doom_live_transfer.py`; `git commit -m "feat: evaluate frozen efficacies in closed-loop Metal Doom"`; actual Git-derived report/raw evidence.

### Task 4: Execute, audit and report the fixed private study

**Files:** Create `docs/experiments/2026-09-13-metal-live-transfer.md`, compact `outputs/doom-learning/metal-live-transfer-m4pro/results.json`; preserve ignored raw logs/traces/checkpoints/controls/ledger. Worker owns the two factual public files only after controller evidence is ready; not alone, no children/SSH/GPU/public launch.

**Interfaces:** Consume exact reviewed CLI and immutable protocol/vectors/model/core/engine pins. Produce actual all72charged/18waves/two calibration results, every candidate/control seed metric and paired task-specific contrasts, replica variation, role/vector/asset/checkpoint pins, exactsource/commands/exits/timing/pressure/release/remaining assumptions with all scientific claim flags false.

- [ ] **Step 1: Controller deploys reviewed stages only to ownedM4.** Independent Git-derived source SHA/bytes and original trusted core, preserve generated5CPUmetadata/nativeproducts/usercheckout. Run focused native files/newengine QA and one ordinary full native suite for this new code. No old scientific replay; native fixture fullgraph attempts0.
- [ ] **Step 2: Controller launches exact private run.** New durable ignored output and scoped AC utility, exact reviewedHEAD/committedprotocol/expectedpins/candidate source paths. Capture actual processhandle/exits and monitor same handle; charge every lane, no restart from missing output/timeout. Keep unrelated workloads untouched.
- [ ] **Step 3: Controller audits terminal evidence.** Verify18waves/72attempts/cap80/8unused/no changes to originalstudy, allframes game-neural/padding clocks/RGBcontrolseparation/zeroexternal/frozenfullbytes/setup-onlyepochs/storage, all24CP schema and baselinecalibration/filler comparisons, candidate/engine/map/core/checkpoint/trace roleSHA/bytes and sourcephysicalbeforeafter. Compute metrics independently from gameobserver/trace, pair eachseed tobaseline, no winning role selected. Preserve failures and raw assets privately, no largedata copy into localtmpfs.
- [ ] **Step 4: Fresh worker writes actual report/compactJSON.** Use supplied audited evidence; Markdown/JSONparse/publishedleafchecks replace inappropriate codeTDD for humanreport. Include censoredsurvival/killzerosafterdeath, all adverse/mixedflat results, both environments separately, independentunit limits/sourceangularmapping assumptions, clocksnotGPU-only/1000x and private storage/resource proof. No publiclaunch/Howitworks updates for an unvalidated privatecandidate.
- [ ] **Step 5: Task review and one strongest final whole-new-subproject review.** Base this subproject at b0aa5eb; at most one final bounded fixwave plus one scopedrereview. Verify covering amended code afterfixes, distinguish testedrevisions, normalpushownedbranch only. Preserveworkspace/failedcontrols perprojectconstraints, include everychronologicalRulingwithcost infinalhandoff. Keep fulltraininggoalACTIVE until actualrequired learning/Doomoutcomes independentlyestablished.
