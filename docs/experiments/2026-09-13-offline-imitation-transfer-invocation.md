# Teacher-Free Hazard Transfer Invocation

Prepared before execution. This is a bounded transfer control for the first
complete offline cohort, not additional training or a launch claim. Execute
only after that cohort finishes naturally. Preserve its checkout and checkpoints.
Use the same model, full graph, fixed decoder and CPU reference backend.

Set `DOOMFLY_IMITATION_COHORT` to the completed cohort and
`DOOMFLY_IMITATION_TRANSFER_OUT` to a fresh output through ignored configuration.
Use `OPENBLAS_NUM_THREADS=1`, the bootstrapped `DOOM_KERNEL_PATH` and target-local
PATH. On Apple Silicon, run job-scoped `caffeinate -is` without concurrent
full-graph benchmarks. Execute the following Python block in the reviewed
`fb57fd1c57960fe5297c3532cd2f939cc16d0685` checkout.

## Fixed protocol

Three observer seeds: 61031, 61032, 61033. Twelve-second arena cap, identical
2,000 ms dark warmup, one live RGB frame per 35 Hz game tic and every 0.1 ms
neural tick. Independently restore each plastic/frozen/shifted learned full
checkpoint before each seed. Additional plastic controls on seed 61031: erased
memory, five-second dark retention and black sensory input.

Every evaluation freezes efficacies and passes an explicit all-zero neural-step
reinforcement schedule. This is essential: the existing survival runner's
default schedule would deliver damage-driven reinforcement. Pose and health
remain observer-only diagnostics. Map crossing, end health and capped survival
describe behavior; neither a changed route nor a single censored survival proves
learning. Preserve all twelve episodes, including failures and unchanged controls.
Checkpoint paths/origins stay in ignored JSONL; game images, IWADs and generated
WADs are excluded from public evidence.

## Execution

```python
import hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
from doom_learning.common import capture_provenance, save_json
from doom_learning_v6.calibration import calibrated_brain
from doom_learning_v6.imitation import _invariants, _memory
from doom_learning_v6.metal.benchmark import portable_backend_metadata, model_identity
from doom_learning_v6.survival import episode, horizon_steps
from doom_learning_v2.vision import frame_for

expected='fb57fd1c57960fe5297c3532cd2f939cc16d0685'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==expected
cohort=Path(os.environ['DOOMFLY_IMITATION_COHORT'])
out=Path(os.environ['DOOMFLY_IMITATION_TRANSFER_OUT'])
if out.exists():raise ValueError('Fresh output directory required')
source=json.loads((cohort/'results.json').read_text())
assert source['complete'] and len(source['episodes'])==8
assert all(r['complete'] and r['invariants_unchanged'] for r in source['episodes'])
assert not (cohort/'failure.json').exists()
assert json.loads((cohort/'invariants.json').read_text())['passed']
assert source['protocol']['diagnostic_limit'] is None
assert (source['protocol']['neurons'],source['protocol']['retained_edges'],source['protocol']['plastic_slots'])==(166700,25582938,4184)
out.mkdir(parents=True)
capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6'])
brain=calibrated_brain(source['protocol']['eta'],backend='cpu')
brain.execution_provenance={'backend':'cpu','backend_metadata':portable_backend_metadata(brain.backend.metadata()),
                            'source_commit':expected,'teacher_current':0.}
assert model_identity(brain)==source['protocol']['model_identity']
fixed=_invariants(brain);seeds=[61031,61032,61033];seconds=12
rows=[]
try:
    protocol={'schema':1,'experiment':'teacher-free transfer of the first offline cohort',
        'connectome':'MaleCNS v1.0','neurons':brain.n,'retained_edges':len(brain.weight),'plastic_slots':len(brain.circuit['edges']),
        'source_commit':expected,'source_results_sha256':hashlib.sha256((cohort/'results.json').read_bytes()).hexdigest(),
        'model_identity':model_identity(brain),'execution':brain.execution_provenance,
        'seeds':seeds,'seconds_limit':seconds,'warmup_ms':2000,'neural_dt_ms':.1,'game_hz':35,
        'evaluation':'independently restored checkpoints; frozen efficacies; explicit zero reinforcement schedule',
        'observer':'pose and health do not enter sensory mapping or fixed decoding',
        'controls':'plastic/frozen/shifted on every seed; erased, five-second dark retention and black vision on seed 61031',
        'claim':'exploratory transfer; capped survival is censored; no replicated learning or biological validation'}
    save_json(out/'protocol.json',protocol)
    with (out/'inputs.jsonl').open('w') as stream:
        stream.write(json.dumps({'cohort':str(cohort.resolve()),'checkpoints':[str((cohort/a/'learned.npz').resolve()) for a in ['plastic','frozen','shifted']]})+'\n')
    cases=[(arm,seed,'held_out') for arm in ['plastic','frozen','shifted'] for seed in seeds]
    cases += [('plastic',seeds[0],mode) for mode in ['memory_erased','retention_5s','vision_black']]
    for arm,seed,mode in cases:
        directory=out/f'{arm}-{mode}-{seed}'
        brain.restore(cohort/'initial.npz' if mode=='memory_erased' else cohort/arm/'learned.npz')
        retention=None
        if mode=='memory_erased':brain.reset(keep_memory=False)
        if mode=='retention_5s':
            before=_memory(brain);clock=brain.cursor;start=time.perf_counter();brain.weights_frozen=False
            _,kernel=brain.rgb_step(frame_for('black'),5000,learning=False,stimulation=None)
            retention={'before':before,'after':_memory(brain),'brain_steps':brain.cursor-clock,
                       'wall_seconds':time.perf_counter()-start,'kernel_seconds':kernel,'teacher_current':0.,'learning':False}
        assert _invariants(brain)==fixed
        record,delivered=episode(brain,seed,seconds,out=directory,learning=False,freeze=True,
                                  schedule=np.zeros(horizon_steps(seconds),dtype=bool),vision=mode!='vision_black',frames=False)
        assert not delivered.any() and record['US_ms']==0.
        assert record['before']['sha256']==record['after']['sha256']
        assert _invariants(brain)==fixed
        record.update(arm=arm,phase=mode,invariants_unchanged=True)
        if retention is not None:record['retention']=retention
        save_json(directory/'episode.json',record)
        slim={k:v for k,v in record.items() if k not in ['trace','assets']}
        rows.append(slim);save_json(out/'progress.json',{'completed_episodes':len(rows),'latest':slim})
        print(arm,seed,mode,record['survival_seconds'],record['dead'],record['end_health'],flush=True)
    save_json(out/'invariants.json',{'before':fixed,'after':_invariants(brain),'passed':True})
    save_json(out/'results.json',{'complete':len(rows)==12,'protocol':protocol,'episodes':rows,
                                   'learning_demonstrated':False,'announcement_ready':False})
except Exception as error:
    brain.checkpoint(out/'failure.npz')
    save_json(out/'failure.json',{'type':type(error).__name__,'completed_episodes':len(rows),
                                'episodes':rows,'learning_demonstrated':False})
    raise
finally:brain.backend.close()
```

The existing `episode` saves its trace after successful completion. A failed
current episode may leave its last `current.json`; the invocation preserves
completed episodes and a full failure checkpoint. This is a documented failure
capture limit, not a complete-frame failure trace guarantee.
