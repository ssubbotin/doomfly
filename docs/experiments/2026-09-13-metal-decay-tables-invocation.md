# Metal Host Decay Table Diagnostic Invocation

Prepared before execution. Use an isolated M4 Pro checkout of the exact reviewed
implementation commit given by ignored `DOOMFLY_RUNTIME_COMMIT`. Set a fresh
`DOOMFLY_DECAY_DIAGNOSTIC_OUT`, bootstrap `DOOM_KERNEL_PATH` and launch with
`OPENBLAS_NUM_THREADS=1` plus job-scoped `caffeinate -is`. Wait for the complete
offline cohort and any transfer job to finish naturally before full-graph work.
Keep strict flags, existing parity thresholds and all retained connections.

Coefficient extraction seeds sleeping zero-edge numerical fixtures explicitly.
These unusual states extract arithmetic; they do not validate threshold
scheduling, physiological behavior or replace the retained connectome. For
elapsed intervals below 1,024 ticks, compare actual CPU/Metal coefficient bits.
The long-interval and modulation fallbacks remain separate limitations.

Repeat the unchanged micrograph in two independently constructed Metal brains.
Record its new measured state alongside the previous strict epoch. Keep prior
failed full-graph evidence. A failed numerical gate remains failed; no timing
measurement, training permission or scientific launch follows from this file.
Exclude binaries, NPZ inputs/checkpoints, game imagery and source archives from
public evidence. Machine origins belong only in ignored configuration.

```python
import gc, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from doom_learning.common import ROOT, save_json, capture_provenance, require_single_blas_thread
from doom_learning_v6.brain import MemoryBrain
from doom_learning_v6.metal import validate
from doom_learning_v6.metal.build import probe

require_single_blas_thread()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit==os.environ['DOOMFLY_RUNTIME_COMMIT']
out=Path(os.environ['DOOMFLY_DECAY_DIAGNOSTIC_OUT'])
if out.exists():raise ValueError('Fresh output directory required')
out.mkdir(parents=True);started=time.perf_counter()
capture_provenance(out,additional=['doom_learning_v2','doom_learning_v6','tests'])
metadata=probe();assert metadata['abi_version']==metadata['native_abi_version']==4
assert metadata['build_configuration']['metal_compile_flags']==['-std=macos-metal2.4','-fno-fast-math','-ffp-contract=off']
assert 'decay_tables.h' in metadata['sources']
save_json(out/'protocol.json',{'schema':1,'source_commit':commit,'experiment':'resident canonical CPU decay coefficients on Metal',
    'strict_build':validate._portable_metal_metadata(metadata),'retained_neurons':166700,'retained_edges':25582938,'plastic_slots':4184,
    'table_entries':3072,'table_bytes':12288,'table_intervals':1024,'neural_dt_ms':.1,
    'previous_micrograph_strict_state_sha256':'b5970b9261406420106a6289e86a9ebfc29f463a9c80ba9e724e523011e1a745',
    'scope':'numerical arithmetic and unchanged full-graph parity gates; no learning, physiology or launch validation'})

graph=out/'coefficient-zero-edge.npz';n=4
np.savez(graph,ptr=np.zeros(n+1,dtype=np.int64),post=np.empty(0,dtype=np.int32),weight=np.empty(0,dtype=np.float32),
    ids=np.arange(n,dtype=np.int64),retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
    lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),superclass=np.array(['numerical-control']*n))
circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),'kc_mask':np.zeros(n,dtype=np.uint8),
    'dan_index':np.full(n,-1,dtype=np.int8),'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
    'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
rows=[]
for tau in [200.,137.]:
    for delta in [1,2,18,22,100,286,1023,1024]:
        values={}
        for name in ['cpu','metal']:
            brain=MemoryBrain(graph,backend=name,circuit=circuit,modulation_mask=np.zeros(n,dtype=np.uint8),adaptation_tau=tau)
            try:
                brain.rest[0]=0.;brain.v[:]=[-64.,-52.,-52.,-52.];brain.g[:]=[0.,64.,0.,0.]
                brain.adaptation[:]=[0.,0.,64.,64.];brain.refractory[:]=[0,0,0,delta+1]
                brain.drive.fill(0);brain.previous_drive.fill(0);brain.active_flag.fill(0);brain.nactive.fill(0);brain.last.fill(-delta)
                brain.backend.advance(1);brain.backend.materialize('coefficient-extraction')
                assert not brain.counts.any() and brain.cursor==1
                np.testing.assert_array_equal(brain.last,np.zeros(n,dtype=np.int64))
                coefficient=np.asarray([-brain.v[0]/64,brain.g[1]/64,brain.adaptation[2]/64,brain.adaptation[3]/64],dtype=np.float32)
                values[name]={'coefficients':coefficient.tolist(),'coefficient_float32_bits':coefficient.view(np.uint32).tolist(),
                              'v':brain.v.tolist(),'g':brain.g.tolist(),'adaptation':brain.adaptation.tolist(),'refractory':brain.refractory.tolist()}
            finally:brain.backend.close()
        equal=values['cpu']['coefficient_float32_bits']==values['metal']['coefficient_float32_bits']
        rows.append({'adaptation_tau_ms':tau,'historical_elapsed_ticks':delta,'lookup_interval':delta<1024,'values':values,'coefficients_bitwise_equal':equal})
save_json(out/'coefficients.json',{'schema':1,'scope':'four artificial sleeping zero-edge arithmetic fixtures; no threshold/physiology claim',
    'coefficient_order':['voltage','conductance','adaptation','refractory_adaptation'],'rows':rows})
save_json(out/'progress.json',{'completed_phases':1,'status':'running'})

sys.path.insert(0,str(ROOT/'tests'))
from test_doom_metal_parity import paired_brains, run_trace
micro={};brains=[]
try:
    cpu,metal=paired_brains(out/'micrograph-first');unused,repeated=paired_brains(out/'micrograph-repeat')
    brains=[cpu,metal,unused,repeated]
    for name,brain in [('cpu',cpu),('metal',metal),('metal-repeat',repeated)]:
        brain.backend.start_diagnostics();counts=run_trace(brain);brain.backend.stop_diagnostics()
        events=np.asarray(brain.backend.spike_events,dtype=np.int64)
        micro[name]={'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),'events_sha256':hashlib.sha256(events.tobytes()).hexdigest(),
                     'events':events.tolist(),'state_sha256':validate._state_digest(brain)}
    save_json(out/'micrograph.json',{'schema':1,'results':micro,'repeat_bitwise':micro['metal']==micro['metal-repeat'],
        'previous_strict_state_sha256':'b5970b9261406420106a6289e86a9ebfc29f463a9c80ba9e724e523011e1a745'})
    assert micro['metal']==micro['metal-repeat']
finally:
    for brain in brains:brain.backend.close()
    del brains;gc.collect()
save_json(out/'progress.json',{'completed_phases':2,'status':'running'})

forty=validate.run(out/'parity-40ms')
save_json(out/'progress.json',{'completed_phases':3,'status':'running','parity_40ms_passed':forty['passed']})
short_trace=validate.TRACE
validate.TRACE=(('black',False),('blue',False),('green',False),('white',True),
    ('left_blue',False),('right_blue',True),('vertical',False),('horizontal',False))
try:eighty=validate.run(out/'parity-80ms')
finally:validate.TRACE=short_trace
save_json(out/'results.json',{'schema':1,'complete':True,'source_commit':commit,
    'lookup_coefficients_bitwise_equal':all(r['coefficients_bitwise_equal'] for r in rows if r['lookup_interval']),
    'fallback_coefficients_bitwise_equal_in_tested_cases':all(r['coefficients_bitwise_equal'] for r in rows if not r['lookup_interval']),
    'micrograph_repeat_bitwise':micro['metal']==micro['metal-repeat'],'micrograph_state_sha256':micro['metal']['state_sha256'],
    'parity_40ms_passed':forty['passed'],'parity_80ms_passed':eighty['passed'],'wall_seconds':time.perf_counter()-started,
    'learning_demonstrated':False,'biologically_validated':False,'announcement_ready':False})
save_json(out/'progress.json',{'completed_phases':4,'status':'complete'})
print('DECAY_DIAGNOSTIC_COMPLETE',json.dumps(json.loads((out/'results.json').read_text())),flush=True)
```

## Separate timing after numerical checks

Use the exact passing 80 ms report from this source/build. Keep five contiguous
samples, no parallel full-graph job and a fresh output. The benchmark's retained
2x CPU target is a performance flag; it does not replace numerical evidence or
block continued exploratory Metal implementation/training. Preserve any nonzero
benchmark exit and its actual report. This is a bounded 40 ms timing sample,
not complete gameplay/training throughput or a 1,000x real-time result.

```sh
OPENBLAS_NUM_THREADS=1 python -m doom_learning_v6.metal.benchmark \
  --out "$DOOMFLY_DECAY_TIMING_OUT" \
  --validation "$DOOMFLY_DECAY_PARITY_80_REPORT" --repetitions 5
```
