# Positive-Case Diagnostic Invocation

This is the exact one-off invocation prepared before starting the follow-up.
It calls existing APIs and changes no repository runtime source. The tested
replica and start were selected after observing the completed parent cohort;
this is a retrospective numerical/mechanism diagnostic, not independent
confirmation of learning. Saved weights are evaluated; no new training occurs.

Provide the parent cohort folder in `DOOMFLY_COHORT_SOURCE` and a fresh output
folder in `DOOMFLY_DIAGNOSTIC_OUT` through ignored local configuration. Use the
unchanged runtime source commit named below and `OPENBLAS_NUM_THREADS=1`.
On the validated Apple Silicon host, `caffeinate -is` holds job-scoped idle and
AC system-sleep assertions around the interpreter. Machine paths and generated
source snapshots/binaries/game assets are excluded from published evidence.

Timing boundary: `whole_diagnostic_realtime_seconds` starts after loading and
validating the parent results and saved states, and creating the output folder.
It includes subsequent provenance capture, construction, episodes and passive
retention. It is not process-launch-to-results elapsed time. The code below is
preserved exactly as executed; this paragraph clarifies its shorter stored
`timing_scope` description.

```python
import hashlib,json,os,time
from pathlib import Path
from doom_learning.common import ROOT,save_json,capture_provenance,require_single_blas_thread
require_single_blas_thread()
import numpy as np
from doom_learning_v6.calibration import calibrated_brain
from doom_learning_v6.survival import episode,horizon_steps
from doom_learning_v2.vision import frame_for

source=Path(os.environ['DOOMFLY_COHORT_SOURCE'])
out=Path(os.environ['DOOMFLY_DIAGNOSTIC_OUT'])
if out.exists():raise ValueError('Fresh diagnostic output directory required')
parent=json.loads((source/'results.json').read_text())
assert parent['complete'] and len(parent['episodes'])==54
replica=42053;seed=62053;seconds=30.
plans=[
    ('plastic-replay-1','plastic',True,0,False),
    ('plastic-replay-2','plastic',True,0,False),
    ('frozen-replay','frozen',True,0,False),
    ('shuffled-replay','shuffled',True,0,False),
    ('vision-black','plastic',False,0,False),
    ('retention-5s','plastic',True,5,False),
    ('memory-erased',None,True,0,True),
]
def file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
reference_files={'results.json':file_digest(source/'results.json')}
states={}
for mode in ['plastic','frozen','shuffled']:
    path=source/f'{replica}-{mode}/learned-efficacies.npz'
    reference_files[f'{replica}-{mode}/learned-efficacies.npz']=file_digest(path)
    reference_files[f'{replica}-{mode}/test-{seed}/episode.json']=file_digest(source/f'{replica}-{mode}/test-{seed}/episode.json')
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'weights','memory_u','memory_w','metadata'}
        states[mode]={k:z[k].copy() for k in ['weights','memory_u','memory_w']}
        states[mode]['configuration']=json.loads(str(z['metadata']))['configuration']
        for key,dtype in [('weights',np.float32),('memory_u',np.float64),('memory_w',np.float64)]:
            assert states[mode][key].shape==(4184,) and states[mode][key].dtype==dtype and np.isfinite(states[mode][key]).all()
out.mkdir(parents=True)
started=time.time()
provenance=capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6'])
old=json.loads((source/'provenance.json').read_text())
assert provenance['source_sha256']==old['source_sha256']
assert provenance['graph_sha256']==old['graph_sha256']
b=calibrated_brain(.001,backend='cpu')
assert b.n==166700 and len(b.post)==25582938 and len(b.circuit['edges'])==4184
assert all(s['configuration']==b.configuration_signature() for s in states.values())
b.execution_provenance=parent['protocol']['execution'].copy()
assert b.execution_provenance['backend']=='cpu'
assert b.backend.metadata()==b.execution_provenance['backend_metadata']
protocol={
    'schema':1,'experiment':'post-hoc diagnostic of one previously beneficial held-out start',
    'parent_cohort':'cpu-controlled-training-20260912',
    'parent_reference_sha256':reference_files,
    'source_commit':'2dd86b5a63d0a3ca88595ae1a1ecad4ad261c5fc',
    'selected_after_observing_parent_results':True,
    'sampling':'one existing training replica and evaluation start, selected after observing its outcome; no new independent training replica',
    'replica':replica,'evaluation_seed':seed,'episode_seconds':seconds,
    'eta':.001,'brain_step_ms':.1,'game_hz':35,'maximum_rule_bin_ms':10,
    'retained_neurons':b.n,'retained_edges':len(b.post),'plastic_slots':len(b.circuit['edges']),
    'planned_episodes':len(plans),
    'plans':[{'condition':name,'saved_state':mode,'vision':vision,'passive_retention_seconds':retention,'erase_memory':erase} for name,mode,vision,retention,erase in plans],
    'episode_boundary':'Fresh neural, sensory and controller state; two-second black warmup; episode weights frozen and zero imposed reinforcement in every condition.',
    'passive_retention':'Only retention-5s uses the original five-second black interval before episode reset, with learning disabled and passive efficacy dynamics enabled.',
    'trace_checks':'Both plastic replays and frozen/shuffled replays must match the original full trace; erasure must match the original frozen full trace.',
    'claim_scope':'Retrospective reproducibility/mechanism diagnostic only. Does not repair failed physiological/conditioning gates or establish general learning.',
    'training_enabled':False,'learning_demonstrated':False,'biologically_validated':False,'announcement_ready':False,
    'execution':b.execution_provenance,
}
save_json(out/'protocol.json',protocol)
save_json(out/'circuit.json',b.circuit['report'])
save_json(out/'visual.json',b.visual_report)
save_json(out/'progress.json',{'status':'running','completed_episodes':0,'planned_episodes':len(plans)})
print('RUN_PID',os.getpid(),flush=True)
records={};rows=[]
try:
    for name,mode,vision,retention,erase in plans:
        b.reset()
        if mode is not None:
            for key in ['memory_u','memory_w']:getattr(b,key)[:]=states[mode][key]
            b.weight[b.circuit['edges']]=states[mode]['weights']
        intervention={'before':b.memory(),'passive_retention_seconds':retention,'erased':erase}
        if retention:
            b.weights_frozen=False
            before_wall=time.perf_counter();before_real=time.time()
            b.rgb_step(frame_for('black'),retention*1000,learning=False)
            intervention.update(retention_awake_seconds=time.perf_counter()-before_wall,retention_realtime_seconds=time.time()-before_real)
        intervention['after']=b.memory()
        r,delivered=episode(b,seed,seconds,out=out/'diagnostic'/name,
            learning=False,freeze=True,vision=vision,
            schedule=np.zeros(horizon_steps(seconds),dtype=bool))
        assert r['weights_frozen'] and not r['learning_enabled'] and r['US_ms']==0 and not delivered.any()
        assert r['before']['sha256']==r['after']['sha256']
        records[name]=r
        row={k:v for k,v in r.items() if k not in ['trace','assets']}
        row.update(condition=name,intervention=intervention)
        rows.append(row)
        save_json(out/'progress.json',{'status':'running','completed_episodes':len(rows),'planned_episodes':len(plans),'latest':row})
        print('EPISODE',name,r['game_tics'],r['survival_seconds'],flush=True)
    originals={mode:json.loads((source/f'{replica}-{mode}/test-{seed}/episode.json').read_text()) for mode in ['plastic','frozen','shuffled']}
    replay_identity={
        'plastic-replay-1':records['plastic-replay-1']['trace']==originals['plastic']['trace'],
        'plastic-replay-2':records['plastic-replay-2']['trace']==originals['plastic']['trace'],
        'frozen-replay':records['frozen-replay']['trace']==originals['frozen']['trace'],
        'shuffled-replay':records['shuffled-replay']['trace']==originals['shuffled']['trace'],
        'memory-erased':records['memory-erased']['trace']==originals['frozen']['trace'],
    }
    checks={
        'complete_episode_count':len(records)==len(plans),
        'original_full_trace_reproduced':all(replay_identity.values()),
        'repeated_plastic_trace_identical':records['plastic-replay-1']['trace']==records['plastic-replay-2']['trace'],
        'neural_clock_within_half_step':all(abs(x['brain_steps']*.0001-x['seconds'])<=.000050001 for r in records.values() for x in r['trace']),
        'one_observation_per_game_tic':all(len(r['trace'])==r['game_tics'] and [x['tick'] for x in r['trace']]==list(range(1,r['game_tics']+1)) for r in records.values()),
        'episode_efficacies_frozen':all(r['before']['sha256']==r['after']['sha256'] for r in records.values()),
        'no_imposed_reinforcement':all(r['US_ms']==0 and all(x['US_steps']==0 for x in r['trace']) for r in records.values()),
        'recorded_input_and_spike_hashes':all(len(x[k])==64 for r in records.values() for x in r['trace'] for k in ['frame_sha256','sensory_sha256','spikes_sha256']),
    }
    report={'schema':1,'complete':True,'protocol':protocol,'episodes':rows,
        'technical_checks':checks,'all_technical_checks_pass':all(checks.values()),
        'original_full_trace_identity':replay_identity,
        'episode_awake_seconds':sum(r['timing']['wall_seconds'] for r in records.values()),
        'native_awake_seconds':sum(r['timing']['neural_backend_seconds'] for r in records.values()),
        'brain_seconds_including_episode_warmup':sum(r['timing']['brain_seconds_including_warmup'] for r in records.values()),
        'whole_diagnostic_realtime_seconds':time.time()-started,
        'timing_scope':'Includes construction/provenance and five-second retention in whole realtime; episode/native timers remain original awake-clock timers.',
        'learning_demonstrated':False,'biologically_validated':False,'announcement_ready':False}
    save_json(out/'results.json',report)
    save_json(out/'progress.json',{'status':'complete','completed_episodes':len(rows),'planned_episodes':len(plans),'technical_checks_pass':all(checks.values())})
    print('DIAGNOSTIC_COMPLETE',json.dumps({'technical_checks':checks,'durations':{n:r['survival_seconds'] for n,r in records.items()}}),flush=True)
    if not all(checks.values()):raise AssertionError('Post-hoc diagnostic integrity failed; preserve and inspect evidence')
finally:
    b.backend.close()
```
