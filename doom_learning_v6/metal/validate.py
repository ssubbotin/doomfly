"""Deterministic CPU/Metal parity metrics and full-graph validation."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path

import numpy as np


def _rate_correlation(cpu,metal):
    cpu=np.asarray(cpu,dtype=np.float64);metal=np.asarray(metal,dtype=np.float64)
    if cpu.shape!=metal.shape or cpu.ndim!=1 or not np.isfinite(cpu).all() or not np.isfinite(metal).all():
        raise ValueError('Invalid firing-rate arrays')
    if np.array_equal(cpu,metal):return 1.
    if len(cpu)<2 or np.ptp(cpu)==0 or np.ptp(metal)==0:return 0.
    value=float(np.corrcoef(cpu,metal)[0,1])
    return value if np.isfinite(value) else 0.


def _one_tick_fraction(cpu,metal):
    if not cpu and not metal:return 1.
    by_cpu=defaultdict(list);by_metal=defaultdict(list)
    for neuron,tick in cpu:by_cpu[neuron].append(tick)
    for neuron,tick in metal:by_metal[neuron].append(tick)
    matched=0
    for neuron,cpu_ticks in by_cpu.items():
        metal_ticks=sorted(by_metal[neuron]);position=0
        for tick in sorted(cpu_ticks):
            while position<len(metal_ticks) and metal_ticks[position]<tick-1:position+=1
            if position<len(metal_ticks) and metal_ticks[position]<=tick+1:
                matched+=1;position+=1
    return matched/max(len(cpu),len(metal),1)


def spike_metrics(cpu_events,metal_events,cpu_rates,metal_rates):
    """Compare unique ``(neuron, tick)`` spike events and firing rates."""
    cpu=set(cpu_events);metal=set(metal_events);union=cpu|metal
    return {'jaccard':len(cpu&metal)/len(union) if union else 1.,
        'total_spike_fraction':abs(len(cpu)-len(metal))/max(len(cpu),len(metal),1),
        'rate_correlation':_rate_correlation(cpu_rates,metal_rates),
        'within_one_tick_fraction':_one_tick_fraction(cpu,metal)}


def compare_decisions(cpu,metal):return cpu==metal


def weight_metrics(reference,actual,rtol=1e-4):
    reference=np.asarray(reference,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Invalid weight arrays')
    difference=np.abs(actual-reference);denominator=np.abs(reference)
    relative=np.divide(difference,denominator,out=np.full_like(difference,np.finfo(np.float64).max),
        where=denominator!=0)
    relative[(denominator==0)&(difference==0)]=0
    return {'maximum_relative_error':float(relative.max(initial=0)),
        'within_rtol':bool(np.all(difference<=rtol*denominator))}


def evaluate_gates(spikes,*,decoder_equal,scientific_gates_equal,weights,
        structural_integrity=True,metal_repeat_bitwise=True):
    gates={'spike_jaccard':spikes['jaccard']>=.995,
        'total_spike_fraction':spikes['total_spike_fraction']<=.005,
        'rate_correlation':spikes['rate_correlation']>=.999,
        'within_one_tick':spikes['within_one_tick_fraction']>=.999,
        'decoder_equal':bool(decoder_equal),'scientific_gates_equal':bool(scientific_gates_equal),
        'weights_within_rtol':bool(weights['within_rtol']) and weights['maximum_relative_error']<=1e-4,
        'structural_integrity':bool(structural_integrity),
        'metal_repeat_bitwise':bool(metal_repeat_bitwise)}
    gates['passed']=all(gates.values());return gates


TRACE=(('black',False),('blue',False),('green',False),('white',True))


def _file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def _state_digest(brain):
    brain.backend.materialize('validation-digest')
    value=hashlib.sha256()
    for name in ['weight',*brain.fields]:
        array=getattr(brain,name);value.update(name.encode());value.update(array.dtype.str.encode())
        value.update(np.asarray(array.shape,dtype=np.int64).tobytes());value.update(array.tobytes())
    return value.hexdigest()


def _portable_metal_metadata(metadata):
    keys=['schema','abi_version','sources','binaries','compiler','sdk','macos','architecture',
        'metal_language','build_configuration']
    device={k:v for k,v in metadata['device'].items() if k!='registry_id'}
    return {**{k:metadata[k] for k in keys},'device':device}


def source_identity():
    root=Path(__file__).resolve().parents[2]
    names=['doom_learning_v6/brain.py','doom_learning_v6/kernel.cpp','doom_learning_v6/rule.py',
        'doom_learning_v6/visual.py','doom_learning_v6/calibration.py','doom_learning_v6/survival.py',
        'doom_learning/common.py','doom_learning_v2/vision.py','doom/engine.py',
        'doom_learning_v6/metal/api.h','doom_learning_v6/metal/backend.mm',
        'doom_learning_v6/metal/backend.py','doom_learning_v6/metal/build.py',
        'doom_learning_v6/metal/graph.py','doom_learning_v6/metal/kernels.metal',
        'doom_learning_v6/metal/validate.py','doom_learning_v6/metal/benchmark.py']
    return {name:_file_digest(root/name) for name in names}


def _calibrated(backend):
    from doom_learning_v6.calibration import calibrated_brain
    return calibrated_brain(backend=backend)


def _scientific_outcomes(brain,counts,readouts):
    groups={'all':np.arange(brain.n),'KC':brain.circuit['kc'],'DAN':brain.circuit['dan'],
        'MBON':brain.circuit['mb'],'fixed_readout':np.asarray([r['index'] for r in readouts])}
    totals={name:int(counts[index].sum()) for name,index in groups.items()}
    outcomes={f'{name}_spikes_observed':value>0 for name,value in totals.items()}
    outcomes['finite_neural_state']=all(np.isfinite(getattr(brain,name)).all()
        for name in ['weight','v','g','drive','modulation','adaptation'])
    return outcomes,totals


def _execute(backend,checkpoint,frames,readouts):
    from doom.engine import NeuralControls
    brain=_calibrated(backend);brain.restore(checkpoint);brain.backend.start_diagnostics()
    controls=NeuralControls(readouts,mode='bci');total=np.zeros(brain.n,dtype=np.int64)
    decisions=[];bins=[];wall=0.
    try:
        for frame,(label,stimulated) in zip(frames,TRACE):
            stimulation=(brain.circuit['dan'],4.) if stimulated else None
            counts,elapsed=brain.rgb_step(frame,10,learning=True,stimulation=stimulation)
            total+=counts;wall+=elapsed
            action=controls.decode(counts,.01)
            decisions.append({k:action[k] for k in ['turn','forward','attack']})
            bins.append({'label':label,'stimulated_DAN':stimulated,'spikes':int(counts.sum()),
                'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest()})
        brain.backend.stop_diagnostics();events=brain.backend.spike_events
        if len(events)!=int(total.sum()):raise RuntimeError('Diagnostic event/count mismatch')
        outcomes,groups=_scientific_outcomes(brain,total,readouts)
        metadata=brain.backend.metadata()
        incoming=brain.backend.incoming.metadata if backend=='metal' else None
        return {'events':events,'rates':total/(len(TRACE)*.01),'decisions':decisions,
            'weights':brain.weight[brain.circuit['edges']].copy(),'state_sha256':_state_digest(brain),
            'event_count':len(events),'total_counts_sha256':hashlib.sha256(total.tobytes()).hexdigest(),
            'scientific_outcomes':outcomes,'group_spikes':groups,'bins':bins,'neural_wall_seconds':wall,
            'backend':_portable_metal_metadata(metadata) if backend=='metal' else metadata,
            'incoming':incoming}
    finally:
        brain.backend.close();del brain;gc.collect()


def _save_trace(path,frames):
    temporary=path.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,frames=frames,labels=np.asarray([x[0] for x in TRACE]),
            stimulated_DAN=np.asarray([x[1] for x in TRACE],dtype=bool),duration_ms=np.float64(10))
    temporary.replace(path)


def run(out):
    from doom_learning.common import GRAPH,ROOT,digest,save_json
    from doom_learning_v2.vision import frame_for
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    out.mkdir(parents=True)
    frames=np.stack([frame_for(label) for label,_ in TRACE]);_save_trace(out/'input-trace.npz',frames)
    manifest=json.loads((GRAPH.parent/'manifest.json').read_text())
    from .benchmark import model_identity
    initial=_calibrated('cpu');checkpoint=out/'initial.npz';initial.checkpoint(checkpoint)
    model=model_identity(initial)
    with np.load(GRAPH,allow_pickle=False) as graph:
        structure={'release':'MaleCNS v1.0','neurons':initial.n,'edges':len(initial.post),
            'graph_file_sha256':_file_digest(GRAPH),'ids_sha256':digest(initial.ids),
            'out_ptr_sha256':digest(initial.ptr),'out_post_sha256':digest(initial.post),
            'original_weight_sha256':digest(graph['weight']),
            'plastic_edges_sha256':digest(initial.circuit['edges']),
            'plastic_edges':len(initial.circuit['edges'])}
    initial.backend.close();del initial;gc.collect()
    cpu=_execute('cpu',checkpoint,frames,manifest['readouts'])
    metal=_execute('metal',checkpoint,frames,manifest['readouts'])
    repeated=_execute('metal',checkpoint,frames,manifest['readouts'])
    spikes=spike_metrics(cpu['events'],metal['events'],cpu['rates'],metal['rates'])
    weights=weight_metrics(cpu['weights'],metal['weights'])
    decoder_equal=compare_decisions(cpu['decisions'],metal['decisions'])
    scientific_equal=cpu['scientific_outcomes']==metal['scientific_outcomes']
    repeat_equal=(metal['events']==repeated['events'] and metal['decisions']==repeated['decisions']
        and metal['state_sha256']==repeated['state_sha256'])
    locked=json.loads((ROOT/'data-provenance/malecns_v1/source.lock.json').read_text())
    structural=(structure['neurons']==166700 and structure['edges']==25582938
        and manifest['source_hashes']==locked and metal['incoming']['edges']==structure['edges'])
    gates=evaluate_gates(spikes,decoder_equal=decoder_equal,scientific_gates_equal=scientific_equal,
        weights=weights,structural_integrity=structural,metal_repeat_bitwise=repeat_equal)
    identity={'validation_sources':source_identity(),'graph_file_sha256':structure['graph_file_sha256'],
        'metal_sources':metal['backend']['sources'],'metal_binaries':metal['backend']['binaries'],
        'metal_environment':{key:metal['backend'][key]
            for key in ['compiler','sdk','macos','architecture','metal_language']},
        'metal_device':metal['backend']['device'],**model}
    def compact(result):return {k:v for k,v in result.items()
        if k not in ['events','rates','weights','incoming']}
    report={'schema':1,'experiment':'full-graph CPU/Metal offline parity',
        'validation_horizon_ms':len(TRACE)*10,'validation_sources':source_identity(),
        'identity':identity,
        'structure':{**structure,'source_hashes':manifest['source_hashes'],
            'incoming':metal['incoming'],'source_lock_equal':manifest['source_hashes']==locked},
        'input_trace':{'format':'procedural RGB frames saved in ignored input-trace.npz',
            'labels':[x[0] for x in TRACE],'duration_ms_per_frame':10,
            'frames_sha256':hashlib.sha256(frames.tobytes()).hexdigest()},
        'cpu':compact(cpu),'metal':compact(metal),'metal_repeat':compact(repeated),
        'comparison':{'spikes':spikes,'decoder_equal':decoder_equal,
            'scientific_gates_equal':scientific_equal,'weights':weights,
            'metal_repeat_bitwise':repeat_equal},'gates':gates,'passed':gates['passed'],
        'biologically_validated':False,'learning_demonstrated':False,
        'interpretation':'Numerical backend parity only. Passing does not validate modeled fly physiology, vision, learning, or game performance.'}
    save_json(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    report=run(args.out);print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    from doom_learning.common import require_single_blas_thread
    require_single_blas_thread();main()
