# Strict Metal Decay-Coefficient Invocation

Prepared before execution with the unchanged strict runtime `bf9f327`.
Set `DOOMFLY_COEFFICIENT_OUT` to a fresh ignored output directory and
`DOOMFLY_SYNTHETIC_GRAPH` to the previous diagnostic's one-neuron zero-edge
archive. Launch with `OPENBLAS_NUM_THREADS=1` and job-scoped sleep assertions.
This extracts an arithmetic coefficient from deliberately inactive synthetic
state through exact power-of-two scaling. It tests neither physiological state
nor threshold scheduling. The complete retained connectome remains untouched.
The host reference is float64 `math.exp` rounded to float32, not a claim about
formally correctly rounded library functions.

```python
import hashlib,json,math,os,subprocess
from pathlib import Path
import numpy as np
from doom_learning.common import ROOT,save_json,require_single_blas_thread
require_single_blas_thread()
from doom_learning_v6.brain import MemoryBrain
from doom_learning_v6.metal.benchmark import portable_backend_metadata

out=Path(os.environ['DOOMFLY_COEFFICIENT_OUT'])
graph=Path(os.environ['DOOMFLY_SYNTHETIC_GRAPH'])
if out.exists():raise ValueError('Fresh output directory required')
out.mkdir(parents=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert commit=='bf9f327dd6820dc3a34b09b1ac686ac9f0a00853'
circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
    'kc_mask':np.zeros(1,dtype=np.uint8),'dan_index':np.full(1,-1,dtype=np.int8),
    'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
    'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
rows=[];metadata={}
for delta in [1,2,18,22,100,286,1023,1024]:
    values={}
    for backend in ['cpu','metal']:
        brain=MemoryBrain(graph,backend=backend,circuit=circuit,modulation_mask=np.zeros(1,dtype=np.uint8))
        try:
            # Inactive synthetic state extracts one evolve coefficient at observation.
            # It is not a threshold/scheduling or physiological control.
            brain.rest[0]=0;brain.v[0]=-64;brain.last[0]=-delta
            brain.backend.advance(1);brain.backend.materialize('decay-coefficient-extraction')
            assert brain.counts[0]==0 and brain.last[0]==0
            coefficient=np.float32(-brain.v[0]/64)
            values[backend]={'coefficient':float(coefficient),
                'float32_bits':int(coefficient.view(np.uint32)),'voltage_mv':float(brain.v[0])}
            metadata[backend]=portable_backend_metadata(brain.backend.metadata())
        finally:brain.backend.close()
    exponent=np.float32(-np.float32(.1)*np.float32(delta)/np.float32(20))
    reference=np.float32(math.exp(float(exponent)))
    rows.append({'delta_steps':delta,'exponent_float32':float(exponent),'cpu':values['cpu'],
        'metal':values['metal'],'metal_minus_cpu_float32_ulps':
            values['metal']['float32_bits']-values['cpu']['float32_bits'],
        'host_float64_exp_rounded_float32':float(reference),
        'host_reference_float32_bits':int(reference.view(np.uint32))})
report={'schema':1,'experiment':'synthetic strict Metal voltage-decay coefficient extraction',
    'source_commit':commit,'synthetic_graph_sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),
    'scope':'One inactive zero-edge synthetic neuron; rest0/v-64 and last timestamps chosen to extract exp(-dt*delta/20) through power-of-two scaling. No retained full-graph state changed; not a threshold-scheduling or physiological validation.',
    'rows':rows,'backends':metadata,'learning_demonstrated':False,'biologically_validated':False}
save_json(out/'report.json',report)
print(json.dumps(report),flush=True)
```
