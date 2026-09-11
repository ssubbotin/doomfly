"""Exact legacy-CPU parity validation for the shared-graph executor."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from doom_learning.common import GRAPH,ROOT,digest,save_json

from .backend import CpuBatchLane,MultiTrajectoryCpuExecutor,SharedCpuGraph


FULL_STRUCTURE={'release':'MaleCNS v1.0','neurons':166700,
    'edges':25582938,'plastic_edges':4184}
HORIZONS_MS=(40,80)


def _file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    names=['doom_learning_v6/brain.py','doom_learning_v6/kernel.cpp',
        'doom_learning_v6/cpu_batch/api.h','doom_learning_v6/cpu_batch/executor.cpp',
        'doom_learning_v6/cpu_batch/backend.py','doom_learning_v6/cpu_batch/build.py',
        'doom_learning_v6/cpu_batch/validate.py','doom_learning_v6/cpu_batch/benchmark.py']
    return {name:_file_digest(ROOT/name) for name in names if (ROOT/name).exists()}


def compare_state(brain,lane):
    fields={name:bool(np.array_equal(getattr(brain,name),lane.arrays[name]))
        for name in lane.STATE_FIELDS}
    fields['cursor']=int(brain.cursor)==int(lane.cursor[0])
    fields['plastic_weights']=bool(np.array_equal(
        brain.weight[brain.circuit['edges']],lane.plastic_weights))
    return {'equal':all(fields.values()),'fields':fields,
        'cpu_sha256':_state_digest(brain,lane,False),
        'batch_sha256':_state_digest(brain,lane,True)}


def _state_digest(brain,lane,batch):
    value=hashlib.sha256()
    for name in lane.STATE_FIELDS:
        array=lane.arrays[name] if batch else getattr(brain,name)
        value.update(name.encode());value.update(array.dtype.str.encode());value.update(array.tobytes())
    cursor=lane.cursor if batch else np.asarray([brain.cursor],dtype=np.int64)
    plastic=lane.plastic_weights if batch else brain.weight[brain.circuit['edges']]
    for name,array in [('cursor',cursor),('plastic_weights',plastic)]:
        value.update(name.encode());value.update(array.dtype.str.encode());value.update(array.tobytes())
    return value.hexdigest()


def drive_for_bin(brain,index):
    drive=np.asarray(brain.tonic,dtype=np.float32).copy()
    drive[brain.lamina]+=12.
    if len(brain.retina):
        luminance=np.float32((index%4+1)/5)
        drive[brain.retina]+=30*luminance/(np.float32(.02)+luminance)
    if index%4==3:drive[brain.circuit['dan']]+=4.
    return drive


def _default_factory():
    from doom_learning_v6.calibration import calibrated_brain
    return calibrated_brain(backend='cpu')


def _default_readouts():
    return json.loads((GRAPH.parent/'manifest.json').read_text())['readouts']


def _structure(brain,release):
    return {'release':release,'neurons':int(brain.n),'edges':len(brain.post),
        'plastic_edges':len(brain.circuit['edges'])}


def _source_lock_equal():
    manifest=json.loads((GRAPH.parent/'manifest.json').read_text())
    locked=json.loads((ROOT/'data-provenance/malecns_v1/source.lock.json').read_text())
    return manifest['source_hashes']==locked


def run(out,*,brain_factory=None,readouts=None,expected_structure=None,
        horizons_ms=HORIZONS_MS):
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    expected=FULL_STRUCTURE if expected_structure is None else dict(expected_structure)
    factory=_default_factory if brain_factory is None else brain_factory
    selected_readouts=_default_readouts() if readouts is None else list(readouts)
    brain=factory();graph=None
    try:
        actual=_structure(brain,expected['release'])
        if actual!=expected:raise ValueError(f'Unexpected graph structure: {actual}')
        graph=SharedCpuGraph.from_brain(brain)
        source_lock_equal=_source_lock_equal() if expected==FULL_STRUCTURE else True
        structure_equal=(actual==expected and graph.neurons==expected['neurons']
            and graph.edges==expected['edges'] and graph.plastic_edges==expected['plastic_edges']
            and source_lock_equal)
        from doom.engine import NeuralControls
        results=[]
        for horizon in horizons_ms:
            if horizon<10 or horizon%10:raise ValueError('Validation horizons must use 10 ms bins')
            brain.reset();lane=CpuBatchLane.from_brain(graph,brain)
            cpu_controls=NeuralControls(selected_readouts,mode='bci')
            batch_controls=NeuralControls(selected_readouts,mode='bci')
            cpu_decisions=[];batch_decisions=[];counts_equal=True
            with MultiTrajectoryCpuExecutor(graph,[lane],workers=1) as executor:
                for index in range(horizon//10):
                    drive=drive_for_bin(brain,index)
                    brain.drive[:]=drive;lane.drive[:]=drive;brain.counts.fill(0)
                    brain._advance_cpu(100);batch_counts=executor.advance(100)[0]
                    counts_equal=counts_equal and np.array_equal(brain.counts,batch_counts)
                    cpu_decisions.append(cpu_controls.decode(brain.counts,.01))
                    batch_decisions.append(batch_controls.decode(batch_counts,.01))
            state=compare_state(brain,lane)
            results.append({'horizon_ms':horizon,'counts_equal':bool(counts_equal),
                'decoder_equal':cpu_decisions==batch_decisions,'state':state,
                'cpu_decisions':cpu_decisions,'batch_decisions':batch_decisions})
        state_equal=all(result['state']['equal'] and result['counts_equal'] for result in results)
        decoder_equal=all(result['decoder_equal'] for result in results)
        passed=state_equal and decoder_equal and structure_equal
        report={'schema':1,'experiment':'full-graph exact CPU batch parity',
            'structure':actual,'horizons_ms':list(horizons_ms),
            'identity':{'graph':graph.identity,'sources':source_identity(),
                'graph_file_sha256':_file_digest(GRAPH) if GRAPH.exists() else None,
                'source_lock_equal':source_lock_equal},
            'results':results,'state_bitwise_equal':state_equal,
            'decoder_equal':decoder_equal,'structural_identity_equal':structure_equal,
            'passed':passed,'learning_demonstrated':False,
            'biologically_validated':False,
            'interpretation':'Exact neural propagation parity only. This result does not validate learning or modeled biology.'}
        out.mkdir(parents=True);save_json(out/'report.json',report);return report
    finally:
        if hasattr(brain,'backend'):brain.backend.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();report=run(args.out);print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
