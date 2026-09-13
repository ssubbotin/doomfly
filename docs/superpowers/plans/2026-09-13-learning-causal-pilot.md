# Learning Causal Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure background plasticity and teacher-current timing effects using the unchanged retained model on Metal.

**Architecture:** Pure mathematical helpers construct and audit assigned schedules and score fixed-decoder outputs. A resident four-lane runner records original RGB episodes and complete checkpoints. The controller runs the private experiment after task review, then publishes reviewed compact results.

**Tech Stack:** Existing Python/NumPy/ViZDoom artifact adapter, Objective-C++/Metal ABI8, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-13-learning-causal-pilot-design.md`

## Global Constraints

- Retain every released connection between retained MaleCNS v1.0 entries.
- Only the 4,184 identified positive KC→MBON11 slots may change, within baseline-relative efficacy fractions [0.1, 2.0].
- Keep RGB sensory mapping, neural dynamics, modulation, eta, plasticity rule, tonic calibration and fixed neuron-to-button decoding unchanged.
- Targets may describe the preserved delayed reinforcement sequence and score outputs; they must never select actions or change the decoder.
- Use opt-in window_ticks=18; default window 0 and public/live runtime stay fixed.
- Serialize GPU ownership; inspect pressure; stop no unrelated workload.
- Preserve failures, controls, full traces, checkpoints and private provenance.
- Store credentials and machine-specific origins only in ignored configuration.
- No biological, general Doom-learning or public-launch claim follows this pair.

## Task 1: Mathematical controls and scoring

**Files:**
- Create: `doom_learning_v6/causal_controls.py`
- Test: `tests/test_doom_learning_causal_controls.py`

**Interfaces:**
- Consumes: existing `score_turns`, `_direction`, `DEAD_ZONE`; baseline and learned plastic-weight arrays; unowned MemoryBrain-compatible state.
- Produces: `frame_ticks(count) -> int64 array`; `duration_permutation(currents,ticks) -> (float64 array,int64 source-index array)`; `dose_proof(aligned,shifted,ticks) -> dict`; `diagnostic_score(predictions,targets) -> dict`; `learned_direction(baseline,learned) -> float64 array`; `initialize_efficacies(brain,memory) -> None`.

- [ ] **Step 1: Write mathematical behavior tests before the module exists.** Use imports inside test functions so absent APIs produce test failures rather than collection failure. The following literal fixtures establish independent expected values:

```python
def test_original_boundaries():
    from doom_learning_v6.causal_controls import frame_ticks
    assert frame_ticks(7).tolist() == [286,285,286,286,286,285,286]

def test_duration_rotation_preserves_dose():
    from doom_learning_v6.causal_controls import duration_permutation, dose_proof
    ticks = np.array([286,285,286,286,286,285,286])
    source = np.array([0,.5,1,1.5,3,2.5,4])
    shifted, mapping = duration_permutation(source,ticks)
    assert mapping.tolist() == [0,5,4,6,2,1,3]
    assert shifted.tolist() == [0,2.5,3,4,1,.5,1.5]
    proof = dose_proof(source,shifted,ticks)
    assert proof['requested_current_ms'] == pytest.approx(357.2)
    assert proof['float32_current_ms'] == pytest.approx(357.2)

def test_plain_rotation_is_rejected():
    from doom_learning_v6.causal_controls import dose_proof
    with pytest.raises(ValueError):
        dose_proof([0,.5,1,1.5,3,2.5,4], [0,3,2.5,4,.5,1,1.5],
                   [286,285,286,286,286,285,286])

def test_balanced_error_is_equal_class_weighted():
    from doom_learning_v6.causal_controls import diagnostic_score
    score = diagnostic_score([0,0,0,1,0],[-2,-4,0,3,9])
    assert score['class_mae_degrees'] == {'left':3.,'idle':0.,'right':5.5}
    assert score['balanced_mae_degrees'] == pytest.approx(8.5/3)
    assert score['mae_degrees'] == pytest.approx(17/5)

def test_normalized_existing_displacement():
    from doom_learning_v6.causal_controls import learned_direction
    np.testing.assert_allclose(learned_direction([10,20,5],[9,24,5]),[-.5,1,0],atol=1e-14)
```

Add finite/schema/boundary rejection tables: count0/negative/bool/float; schedule rank2, empty, mismatched lengths, noninteger/nonpositive ticks, nonzero first current, NaN/Inf/current outside0..4; direction nonpositive baseline, nonfinite/mismatched vectors, zero displacement. Use the real tiny `brain` fixture to verify manual u/w/weights, unchanged nonplastic slots and buffers, and reject invalid or owned mutations before any state change. Add a fractional-amplitude float32 proof and four contiguous-block coverage test.

- [ ] **Step 2: Capture expected RED.** Run `OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_doom_learning_causal_controls.py -q`; preserve output in this task's ignored report/log. Missing helper imports must fail inside tests.

- [ ] **Step 3: Implement the helpers.** This is the complete reference algorithm; use four-space indentation and add ordinary validation errors, not a generalized framework:

```python
import math
import numpy as np
from .imitation import score_turns, _direction

def frame_ticks(count):
    if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or count <= 0:
        raise ValueError('Positive integer frame count required')
    boundaries = np.array([round(i*10000/35) for i in range(count+1)],dtype=np.int64)
    return np.diff(boundaries)

def _schedule(currents,ticks):
    c=np.asarray(currents,dtype=np.float64); t=np.asarray(ticks)
    if c.ndim!=1 or not len(c) or t.shape!=c.shape or t.dtype.kind not in 'iu':
        raise ValueError('Matching current/tick vectors required')
    if not np.isfinite(c).all() or np.any(c<0) or np.any(c>4) or c[0]!=0 or np.any(t<=0):
        raise ValueError('Finite bounded current, first zero, positive ticks required')
    return c.copy(),t.astype(np.int64,copy=True)

def dose_proof(aligned,shifted,ticks):
    a,t=_schedule(aligned,ticks); s,_=_schedule(shifted,ticks)
    groups={}
    for duration in np.unique(t):
        selected=t==duration
        for dtype in (np.float64,np.float32):
            unsigned=np.uint64 if dtype==np.float64 else np.uint32
            x=np.sort(a[selected].astype(dtype).view(unsigned))
            y=np.sort(s[selected].astype(dtype).view(unsigned))
            if not np.array_equal(x,y): raise ValueError('Duration-conditioned amplitude multiset differs')
        groups[str(int(duration))]=int(selected.sum())
    totals={}
    for name,dtype in [('requested',np.float64),('float32',np.float32)]:
        dose=lambda x: math.fsum(float(v)*int(dt)*.1 for v,dt in zip(x.astype(dtype),t))
        original,replayed=dose(a),dose(s)
        if original!=replayed: raise ValueError('Assigned current integral differs')
        totals[name+'_current_ms']=original
    return {'exact':True,'duration_groups':groups,**totals}

def duration_permutation(currents,ticks):
    c,t=_schedule(currents,ticks); mapping=np.arange(len(c),dtype=np.int64)
    for duration in np.unique(t[1:]):
        ix=np.flatnonzero((t==duration)&(mapping>0))
        mapping[ix]=np.roll(ix,len(ix)//2)
    shifted=c[mapping]
    if np.array_equal(mapping,np.arange(len(c))): raise ValueError('No nontrivial temporal permutation')
    dose_proof(c,shifted,t)
    return shifted,mapping

def diagnostic_score(predictions,targets):
    result=score_turns(predictions,targets)
    p=np.asarray(predictions,dtype=np.float64); t=np.asarray(targets,dtype=np.float64)
    actual,predicted=_direction(t),_direction(p)
    classes=[('left',-1),('idle',0),('right',1)]
    per={name:float(np.abs(p[actual==c]-t[actual==c]).mean())
         for name,c in classes if np.any(actual==c)}
    result.update(class_mae_degrees=per,balanced_mae_degrees=float(np.mean(list(per.values()))),
                  predicted_class_counts={name:int((predicted==c).sum()) for name,c in classes},
                  confusion_rows_target_columns_prediction_left_idle_right=
                  [[int(((actual==a)&(predicted==b)).sum()) for _,b in classes] for _,a in classes])
    blocks=[]
    for ix in np.array_split(np.arange(len(t)),4):
        if len(ix): blocks.append({'start':int(ix[0]),'end_exclusive':int(ix[-1]+1),
                                  **score_turns(p[ix],t[ix])})
    result['contiguous_blocks']=blocks
    return result

def learned_direction(baseline,learned):
    b=np.asarray(baseline,dtype=np.float64); l=np.asarray(learned,dtype=np.float64)
    if b.ndim!=1 or not len(b) or l.shape!=b.shape or not np.isfinite(b).all() or not np.isfinite(l).all() or np.any(b<=0):
        raise ValueError('Finite positive baseline and matching learned vectors required')
    displacement=l/b-1; norm=float(np.max(np.abs(displacement)))
    if not math.isfinite(norm) or norm==0: raise ValueError('Nonzero finite displacement required')
    return displacement/norm

def initialize_efficacies(brain,memory):
    m=np.asarray(memory,dtype=np.float64)
    if getattr(brain,'_metal_batch_owner',None) is not None:
        raise ValueError('Manual initialization requires an unowned brain')
    if m.shape!=brain.memory_w.shape or not np.isfinite(m).all() or np.any(m<-.9) or np.any(m>1):
        raise ValueError('Matching bounded memory required')
    weights=(brain.baseline_plastic.astype(np.float64)*(1+m)).astype(np.float32)
    if not np.isfinite(weights).all() or np.any(weights<=0): raise ValueError('Finite positive efficacy required')
    brain.memory_u[:]=m; brain.memory_w[:]=m
    brain.weight[brain.circuit['edges']]=weights
```

- [ ] **Step 4: Verify focused GREEN and full portable suite once.** Use the established ignored interpreter/kernel environment; preserve exact commands/counts/warnings. Self-review mutations: naive frame rotation, unbalanced score, incorrect decoder class counts, changing a fixed slot, accepting a nonfinite initialization. Correct defects with an additional RED case before fixing.
- [ ] **Step 5: Commit helper/test files with Sergey Subbotin identity.** Subject `feat: add causal imitation controls`. Write the full implementation/TDD/self-review report to the task report path; return concise status and commit.

## Task 2: Resident causal episode runner

**Files:**
- Create: `doom_learning_v6/causal_pilot.py`
- Test: `tests/test_doom_learning_causal_pilot.py`
- Modify: `.gitignore` (add only `outputs/doom-learning/causal-pilot-20260913/` for raw state)

**Interfaces:**
- Consumes: all Task1 helpers; existing `MetalBatchExecutor(brains,window_ticks=18)`, `calibrated_brain`, `OfflineEpisode`, `NeuralControls`, `_invariants`, `_memory`, `teacher_error`, `save_json`, `_file_digest`, `_memory_status`.
- Produces: `replay_episode(brains,executor,data,readouts,currents,*,learning,frozen,directory,warmup_ms=2000) -> dict`; CLI `python -m doom_learning_v6.causal_pilot --mode sensitivity|causal --train PATH --eval PATH --reference PATH --out FRESH_PATH --source-commit FULL_SHA`.

- [ ] **Step 1: Add actual-consumer RED tests.** Portable tests cover schedule/flags/count rejection before reset or iteration, fresh-output rejection and source-commit mismatch (subprocess CLI). M4 tests use real tiny synthetic RGB brains from `test_doom_metal_batch_rgb.visual_brain`, real `executor_type()` and BCI readouts DNp20L(index0)/R(index2), DNpe017L(index1)/R(index3). A three-frame synthetic OfflineFrame iterator replaces only expensive downloaded-video acquisition; native simulation, preparation, rule and decoder remain real. Literal intervals are 28.6,28.5,28.6ms. Compare each recorded neural hash/decoded output and terminal full state against independent actual serial Metal calls with the same literal pulses `[0,1,2]`; check frozen u/w/weights unchanged, no teachers during frozen evaluation, and reset/replay reproduces results. Keep one matched serial and one frozen lane distinct. Inject an iterator failure after frame0 and verify partial traces, typed failure, iterator close and per-lane failure checkpoints. Tests must fail from missing consumer API before implementation; main provides serialized target execution if M4 unavailable to worker. Do not fabricate GPU outputs.
- [ ] **Step 2: Capture RED, then implement the full phase loop.** The core data path is:

```python
ticks=frame_ticks(data.frame_count)
# Before mutations: validate one schedule and strict boolean learning/frozen
# flag per brain, uint8 RGB adapter dimensions, nonnegative finite warmup,
# fresh phase directory, and executor.brains identity/order.
for brain,value in zip(brains,frozen):
    brain.reset(keep_memory=True)
    brain.weights_frozen=value
black=np.zeros((data.height,data.width,3),dtype=np.uint8)
executor.rgb_step([black]*len(brains),warmup_ms,learning=False)
origin=brains[0].cursor
controls=[NeuralControls(readouts,mode='bci') for brain in brains]
targets=data.turn_targets()
traces=[[] for brain in brains]
frames=data.iter_frames()
try:
    for sample in frames:
        index=sample.index
        if index!=len(traces[0]) or index>=data.frame_count:
            raise ValueError('Original contiguous frames required')
        count=int(ticks[index])
        pulses=[(b.circuit['dan'],float(currents[lane,index]))
                if currents[lane,index] else None for lane,b in enumerate(brains)]
        counts,elapsed=executor.rgb_step([sample.rgb]*len(brains),count*.1,
                                         learning=learning,stimulations=pulses)
        if any(b.cursor-origin!=int(ticks[:index+1].sum()) for b in brains):
            raise ValueError('Original neural boundary mismatch')
        for lane,b in enumerate(brains):
            action=controls[lane].decode(counts[lane],count*.0001)
            imposed=float(currents[lane,index])
            target=float(targets[index])
            traces[lane].append({'index':index,'timestamp':float(sample.timestamp),
                'raw_action':np.asarray(sample.action).tolist(),'teacher_index':index,
                'target_turn':target,'teacher_error':teacher_error(action['turn'],target),
                'teacher_current':imposed,'teacher_float32_current':float(np.float32(imposed)),
                'teacher_dose_current_ms':imposed*count*.1,'neural_steps':count,
                'brain_steps':b.cursor-origin,'brain_seconds':(b.cursor-origin)*.0001,
                'action':action,'KC_spikes':int(counts[lane][b.circuit['kc']].sum()),
                'DAN_spikes':counts[lane][b.circuit['dan']].tolist(),
                'MBON_spikes':counts[lane][b.circuit['mb']].tolist(),
                'frame_sha256':digest(sample.rgb),'spikes_sha256':digest(counts[lane]),
                'memory':_memory(b)})
finally:
    frames.close()
if any(len(trace)!=data.frame_count for trace in traces):
    raise ValueError('Incomplete original episode')
```

Implement the consumer, not literal top-level pseudocode: wrap in the declared function, validate before mutation, handle warmup0 without a zero-duration advance, preserve failure records/checkpoints while re-raising the original error. Record source identity, original frame count, phase wall/RGB brain time, warmup requested/actual brain and wall time, resident bytes, before/after memory, full traces and `diagnostic_score` per lane. Track actual decoded-frame completion; counters overlap and must not be added as wall. Write `lane-N/episode.json`, `summary.json`, per-phase `progress.json`; frozen phases must verify u/w/plastic weights before/after. Do not monkeypatch model/rule/backend methods, crop episodes or report prefixes as complete.

- [ ] **Step 3: Implement CLI orchestration.** Validate environment OPENBLAS_NUM_THREADS=1 before NumPy work, macOS, clean expected full Git SHA, fresh output, complete original reference results/protocol, eta.001/oneepoch/nolimit, train982/eval988, distinct video/control identities,166700neurons/25582938retainededges/4184slots and two DAN neurons. Use OfflineEpisode's actual byte/metadata validation. Pin graph/annotations/normalized-neurons, every model/backend source and native build byte, all reference controls/checkpoints used, exact args privately before first simulation. Acquire a nonblocking OS file lock at an ignored parent-root `causal-pilot-gpu.lock`; record PID/ownership and release in finally. Reject pressure <25% free or swap growth; record pre/post phase pressure. Avoid existing 2×CPU gate, public launch or shared workload changes.

Construct four calibrated CPU-host brains, restore original initial checkpoint before attachment, preserve fixed invariants. For sensitivity, initialize memory0/0/+0.05d/−0.05d, attach W18 owner and run only training RGB, weights frozen/current0. Compare replicas excluding all wall fields; require neural hashes, action dictionaries and memory exact. If both manual lanes have exactly unchanged actions, reconstruct four unowned baseline brains and run additional memory+0.20d/−0.20d/uniform−.05/+ .05, comparing against the saved baseline; causal mode still proceeds.

For causal mode, read aligned requested currents and tick vector from the preserved plastic training trace; require contiguous982 indices, canonical durations, firstcurrent0/current0..4, byte-matching train identity and original delayed-feedback formula. Build shifted schedule via Task1; preserve mapping, assigned proof and residual correlations. Restore four independent original initial checkpoints, attach W18; train A/Z/S/F with learning `[True,True,True,True]`, frozen `[False,False,False,True]`, currents `[aligned,zero,shifted,zero]`. Compare A's entire original canonical frame records excluding only wall clocks and the added float32 diagnostic field, and all24 checkpoint arrays against original plastic/learned.npz. Save all four learned checkpoints before held evaluation. Restore each saved learned checkpoint independently, evaluate fullheld with all currents0, learningFalse, frozenTrue; verify u/w/weights unchanged. Materialize before invariant hashes; close owner/brains deterministically.

Write complete final `results.json` only after all expected phases/frames, original-replay comparisons, assigned-dose checks, freezes and fixed invariants pass. Include mechanism-diagnostic scope, `learning_demonstrated:false`, `announcement_ready:false`, sources/inputs/phase summaries and any sensitivity-null limitation. Failed runs retain partial progress and typed failure records. No changes to existing source or defaults beyond these two modules and raw ignore line.

- [ ] **Step 4: Verify GREEN and full suites once.** Portable worker runs focused/full tests and reports macOS skips explicitly; controller transfers exact committed code and executes actual M4 RED/GREEN/complete suite under serialized ownership before any scientific run. Review mathematical schedules, no accidental extra feedback delay, state reset/freeze, error preservation and exact replay. Preserve code hashes before/after.
- [ ] **Step 5: Commit runner/tests/ignore using Sergey Subbotin identity.** Subject `feat: run resident causal learning pilot`. Full report includes TDD, native tests actually run versus pending controller checks, and exact relevant covering commands/output.

## Task 3: Execute and publish the bounded mechanism experiment

**Files:**
- Create: `docs/experiments/2026-09-13-learning-causal-pilot.md`
- Create: `outputs/doom-learning/learning-causal-pilot-m4pro/results.json` (compact portable evidence only)
- Private: this plan's own ignored ledger, runtime protocol, raw episodes/checkpoints, source/input manifests and physically checked artifacts.

**Interfaces:**
- Consumes: reviewed Task2 CLI/results and exact physical M4 artifacts; fixed eight original control traces/three checkpoints remain source controls, not independent learners.
- Produces: a reviewed report of sensitivity and A/Z/S/F contrasts with explicit next-action decision, raw artifact hashes and measured time/pressure. Native goal remains active unless actual general learning criteria are met elsewhere.

- [ ] **Step 1: Controller completes target RED/GREEN and full suite.** Use fresh ignored tests-only staging against missing Task2 API for authentic native RED, then exact owned committed source for GREEN. Record actual target source/build/reference/input hashes and clean user checkout unchanged. Explicit target paths/hostnames belong only to ignored runtime configuration. The target worktree is owned; synchronize through non-destructive fast-forward bundle/import. Recheck GPU ownership/pressure before any job.
- [ ] **Step 2: Freeze runtime protocol, then run full training-only sensitivity.** Commands consume ignored task-specific locations:

```bash
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.causal_pilot --mode sensitivity \
  --train "$DOOMFLY_TRAIN_INPUT" --eval "$DOOMFLY_EVAL_INPUT" \
  --reference "$DOOMFLY_ORIGINAL_CONTROL" --out "$DOOMFLY_SENSITIVITY_OUT" \
  --source-commit "$DOOMFLY_MEASURED_SOURCE"
```

Verify complete982frames/lane, replica identity, bounded manual initialization and continuous decoder/readout changes; preserve any additional null-probe batch. Do not consume held sequence for sensitivity tuning.
- [ ] **Step 3: Run A/Z/S/F training and diagnostic evaluation.** Same command with `--mode causal` and a distinct fresh out path. Verify full982train+988held frames/lane, complete checkpoints, A original exact replay, current integrals/multisets, frozen held memory and retained graph/nonplastic bytes. Preserve failed attempts; investigate mismatches with systematic-debugging/TDD before interpretation. No automatic new-rule change or preferred-arm selection.
- [ ] **Step 4: Physically copy/check private evidence and calculate contrasts.** Independently stream-hash every artifact used in the scientific report on target/local; compare all named source/input/build hashes before/after. Parse actual complete traces, calculate class-balanced MAE, ordinaryMAE/recall/confusion, neural rates, weights and current checks, and per-frame/time-block A−Z/A−S/Z−F/S−Z. Report effect sizes, no frame-independent uncertainty or arbitrary threshold. Separate setup/darkwarm/RGB/brain/wall/time and pressure; no aspirational throughput forecast. Compact JSON has no rawpixels/NPZ/transcripts/credentials/privateorigins, no symlinks, <=200KiB, links to valid report/source docs, positive bytes and actual SHA fields.
- [ ] **Step 5: Write report and verify documentation/metadata.** Include council method/corrections, manual-diagnostic label, source/model/data limitations (historically inspected stationaryAPPOpair), allarms/negativeeffects, replay/freeze invariants, primary metric and mechanism interpretation, concrete next-step selection per spec. Verify links, JSON consistency, `git diff --check` and exact staged paths; publish no biological/generalDoomlearning/launch claim.
- [ ] **Step 6: Commit compact evidence/report with Sergey Subbotin identity.** Subject `docs: report causal learning pilot`. Task review and strongest whole-subproject review precede ordinary owned-branch push. Preserve ignored scientific evidence/ledger under standing preservation instruction; no cleanup of user/sibling worktrees.
