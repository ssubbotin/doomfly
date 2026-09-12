# Strict Metal Diagnostic Invocation

Prepared before execution. Calls unchanged diagnostic/calibration APIs using the
strict-build source commit, checked against `DOOMFLY_RUNTIME_COMMIT`.
Provide a fresh `DOOMFLY_STRICT_DIAGNOSTIC_OUT` via ignored local configuration;
launch with `OPENBLAS_NUM_THREADS=1` and job-scoped `caffeinate -is` on the
validated M4 Pro. Synthetic graphs are arithmetic controls, never replacements
for the full MaleCNS experiment. The 80 ms phase temporarily selects the same
eight-frame stress sequence preserved in the previous failed diagnostic; its
actual frames, labels and horizon are recorded by the existing validator.

If either numerical horizon fails, timing is explicitly an observational
diagnostic. It cannot satisfy the existing passing-validation guard or authorize
game training. No gate is weakened, and speed never establishes learning.
Generated source snapshots, graph/control archives, checkpoints, compiled
artifacts and game assets remain excluded from published evidence. The Python
block below is preserved exactly as executed; timing starts at its stated
boundary and includes provenance/build/control construction.

```python
import hashlib,json,os,subprocess,time
from pathlib import Path
from doom_learning.common import ROOT,save_json,capture_provenance,require_single_blas_thread
require_single_blas_thread()
import numpy as np
from doom_learning_v6.metal import validate as validation
from doom_learning_v6.metal import benchmark
from doom_learning_v6.metal.build import probe
from doom_learning_v6.calibration import calibrated_brain
from doom_learning_v2.vision import frame_for

out=Path(os.environ['DOOMFLY_STRICT_DIAGNOSTIC_OUT'])
expected_commit=os.environ['DOOMFLY_RUNTIME_COMMIT']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==expected_commit
if out.exists():raise ValueError('Fresh diagnostic output directory required')
out.mkdir(parents=True);started=time.time()
capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6','tests'])
metadata=probe()
assert metadata['build_configuration']['metal_compile_flags']==[
    '-std=macos-metal2.4','-fno-fast-math','-ffp-contract=off']
protocol={'schema':1,'source_commit':commit,'experiment':'bounded strict Metal arithmetic diagnostic',
    'retained_neurons':166700,'retained_edges':25582938,'plastic_slots':4184,
    'brain_step_ms':.1,'maximum_rule_bin_ms':10,
    'phases':['synthetic arithmetic controls','micrograph repeatability','full-graph 40 ms parity',
        'full-graph 80 ms parity','five repeated matched 40 ms timing samples'],
    'strict_build':validation._portable_metal_metadata(metadata),
    'original_micrograph_fast_math_state_sha256':
        '0fdec182c34b38bc8aa9289d9eab39ace678df5f38d598a8f043d83bc29674e9',
    'original_default_metallib_sha256':
        '62a1f686823a2669fd2cebd000f4f918672bd41465d0b5872d7b7abf24f7ab65',
    'claim_scope':'Numerical arithmetic/parity/speed only; no game training, general learning or biological validation.',
    'timing_scope':'Whole realtime starts after imports/commit guard/output creation, before provenance and probe; timing samples exclude construction.',
    'game_training_enabled':False,'learning_demonstrated':False,'biologically_validated':False}
save_json(out/'protocol.json',protocol)
save_json(out/'progress.json',{'status':'running','completed_phases':0,'planned_phases':5})
print('RUN_PID',os.getpid(),flush=True)

from doom_learning_v6.brain import MemoryBrain
graph=out/'synthetic-zero-edge.npz'
np.savez(graph,ptr=np.array([0,0],dtype=np.int64),post=np.empty(0,dtype=np.int32),
    weight=np.empty(0,dtype=np.float32),ids=np.array([0],dtype=np.int64),
    retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
    lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
    superclass=np.array(['numerical-control']))
circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
    'kc_mask':np.zeros(1,dtype=np.uint8),'dan_index':np.full(1,-1,dtype=np.int8),
    'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
    'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
controls=[]
for steps in [1,100]:
    brains=[MemoryBrain(graph,backend=name,circuit=circuit,modulation_mask=np.zeros(1,dtype=np.uint8))
        for name in ['cpu','metal']]
    try:
        values={}
        for name,brain in zip(['cpu','metal'],brains):
            brain.drive[0]=12
            brain.backend.advance(steps);brain.backend.materialize('synthetic-arithmetic-control')
            values[name]={field:getattr(brain,field).tolist() for field in
                ['v','g','drive','previous_drive','adaptation']}
        controls.append({'steps':steps,'initial_voltage_mv':-52.,'rest_mv':-52.,
            'drive':12.,'values':values,'voltage_difference_mv':values['metal']['v'][0]-values['cpu']['v'][0]})
    finally:
        for brain in brains:brain.backend.close()
save_json(out/'arithmetic-controls.json',{'schema':1,
    'scope':'Synthetic one-neuron zero-edge numerical controls only; full retained graph is unchanged.',
    'controls':controls})
print('ARITHMETIC',json.dumps(controls),flush=True)

import sys
sys.path.insert(0,str(ROOT/'tests'))
from test_doom_metal_parity import paired_brains,run_trace
cpu,metal=paired_brains(out/'micrograph-first')
unused,repeated=paired_brains(out/'micrograph-repeat')
try:
    micro={}
    for name,brain in [('cpu',cpu),('metal',metal),('metal-repeat',repeated)]:
        brain.backend.start_diagnostics();counts=run_trace(brain);brain.backend.stop_diagnostics()
        events=np.asarray(brain.backend.spike_events,dtype=np.int64)
        micro[name]={'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
            'events_sha256':hashlib.sha256(events.tobytes()).hexdigest(),
            'events':events.tolist(),'state_sha256':validation._state_digest(brain)}
    differences={}
    for field in ['weight',*cpu.fields]:
        a=getattr(cpu,field);b=getattr(metal,field)
        differences[field]={'exact':bool(np.array_equal(a,b)),
            'maximum_absolute_difference':float(np.abs(a.astype(float)-b.astype(float)).max(initial=0))}
    assert micro['metal']==micro['metal-repeat']
    assert micro['metal']['counts_sha256']=='de7d06da6e31fe80c35eb54c19926e81db5c514483fa170ca4eccf878d0fec90'
    assert micro['metal']['events_sha256']=='00ff37a29ca8cc916a39436e1ac4f4aac1bef46291502d751bb4860e21cd6ad1'
    save_json(out/'micrograph.json',{'schema':1,'results':micro,
        'cpu_metal_field_differences':differences,'strict_repeat_bitwise':True})
    print('MICROGRAPH',json.dumps(micro),flush=True)
finally:
    for brain in [cpu,metal,unused,repeated]:brain.backend.close()
save_json(out/'progress.json',{'status':'running','completed_phases':2,'planned_phases':5})

short_trace=validation.TRACE
forty=validation.run(out/'parity-40ms')
print('PARITY_40',json.dumps(forty['comparison']),flush=True)
save_json(out/'progress.json',{'status':'running','completed_phases':3,'planned_phases':5})
validation.TRACE=(('black',False),('blue',False),('green',False),('white',True),
    ('left_blue',False),('right_blue',True),('vertical',False),('horizontal',False))
try:eighty=validation.run(out/'parity-80ms')
finally:validation.TRACE=short_trace
print('PARITY_80',json.dumps(eighty['comparison']),flush=True)
save_json(out/'progress.json',{'status':'running','completed_phases':4,'planned_phases':5})
if forty['passed'] and eighty['passed']:
    timing=benchmark.run(out/'benchmark-40ms',out/'parity-80ms'/'report.json',5)
    timing_kind='existing guarded benchmark; both numerical horizons passed'
else:
    import gc,resource,statistics
    frames=np.stack([frame_for(label) for label,_ in short_trace])
    checkpoint=out/'parity-40ms'/'initial.npz'
    cpu=calibrated_brain(backend='cpu');metal=calibrated_brain(backend='metal')
    try:
        metal.backend.ensure_initialized();before=benchmark._memory_status()
        cpu.restore(checkpoint);metal.restore(checkpoint)
        benchmark._sample(cpu,frames);benchmark._sample(metal,frames)
        cpu_samples,metal_samples=benchmark._collect_samples(cpu,metal,checkpoint,frames,5)
        after=benchmark._memory_status()
        timing={'schema':1,'experiment':'unvalidated strict Metal diagnostic timing only',
            'repetitions':5,'simulated_seconds_per_sample':.04,
            'sampling_order':'one warmup per backend, then contiguous CPU and Metal blocks',
            'cpu_samples':cpu_samples,'metal_samples':metal_samples,
            'cpu_median':statistics.median(x['neural_seconds'] for x in cpu_samples),
            'metal_median':statistics.median(x['neural_seconds'] for x in metal_samples),
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,
            'memory':{'before':before,'after':after},
            'metal':benchmark.portable_backend_metadata(metal.backend.metadata()),
            'numerically_validated':False,'game_training_enabled':False,
            'learning_demonstrated':False,
            'interpretation':'Observational profile after failed parity, unusable as a passing training-validation report. Existing training/benchmark guards remain unchanged.'}
        save_json(out/'diagnostic-timing.json',timing)
        timing_kind='unvalidated observational timing; no training gate bypass'
    finally:
        cpu.backend.close();metal.backend.close()
summary={'schema':1,'complete':True,'source_commit':commit,
    'parity_40ms_passed':forty['passed'],'parity_80ms_passed':eighty['passed'],
    'micrograph_strict_state_sha256':micro['metal']['state_sha256'],
    'timing_kind':timing_kind,'whole_diagnostic_realtime_seconds':time.time()-started,
    'game_training_enabled':False,'learning_demonstrated':False,'biologically_validated':False,'announcement_ready':False}
save_json(out/'results.json',summary)
save_json(out/'progress.json',{'status':'complete','completed_phases':5,'planned_phases':5})
print('STRICT_DIAGNOSTIC_COMPLETE',json.dumps(summary),flush=True)
```
