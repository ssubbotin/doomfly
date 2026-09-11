"""Exact legacy-CPU parity validation for the shared-graph executor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from doom_learning.common import GRAPH,ROOT,digest,save_json

from .backend import CpuBatchLane,MultiTrajectoryCpuExecutor,SharedCpuGraph


FULL_STRUCTURE={'release':'MaleCNS v1.0','neurons':166700,
    'edges':25582938,'plastic_edges':4184}
HORIZONS_MS=(40,80)
VALIDATION_LANES=4
VALIDATION_WORKERS=4
VALIDATION_REPEATS=2
SOURCE_FILES=('doom/engine.py','doom/native.py','doom_learning/common.py',
    'doom_learning/circuit.py','doom_learning_v6/brain.py','doom_learning_v6/kernel.cpp',
    'doom_learning_v6/rule.py','doom_learning_v6/calibration.py',
    'doom_learning_v6/visual.py','doom_learning_v6/cpu_batch/api.h',
    'doom_learning_v6/cpu_batch/executor.cpp','doom_learning_v6/cpu_batch/backend.py',
    'doom_learning_v6/cpu_batch/build.py','doom_learning_v6/cpu_batch/validate.py',
    'doom_learning_v6/cpu_batch/benchmark.py')


def _file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {name:_file_digest(ROOT/name) for name in SOURCE_FILES if (ROOT/name).exists()}


def repository_identity():
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,text=True,
        stdout=subprocess.PIPE).stdout.strip()
    changed=subprocess.run(['git','status','--porcelain','--untracked-files=no','--',
        *SOURCE_FILES],cwd=ROOT,check=True,text=True,stdout=subprocess.PIPE).stdout.strip()
    return {'git_commit':commit,'source_tree_clean':not bool(changed)}


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
            runs=[]
            for repeat in range(VALIDATION_REPEATS):
                brain.reset();lanes=[CpuBatchLane.from_brain(graph,brain)
                    for _ in range(VALIDATION_LANES)]
                cpu_controls=NeuralControls(selected_readouts,mode='bci')
                batch_controls=[NeuralControls(selected_readouts,mode='bci')
                    for _ in lanes]
                cpu_decisions=[];batch_decisions=[[] for _ in lanes]
                counts_equal=[True for _ in lanes]
                with MultiTrajectoryCpuExecutor(graph,lanes,
                        workers=VALIDATION_WORKERS) as executor:
                    for index in range(horizon//10):
                        drive=drive_for_bin(brain,index);brain.drive[:]=drive
                        for lane in lanes:lane.drive[:]=drive
                        brain.counts.fill(0);brain._advance_cpu(100)
                        batch_counts=executor.advance(100)
                        cpu_decisions.append(cpu_controls.decode(brain.counts,.01))
                        for lane_index,counts in enumerate(batch_counts):
                            counts_equal[lane_index]=counts_equal[lane_index] and \
                                np.array_equal(brain.counts,counts)
                            batch_decisions[lane_index].append(
                                batch_controls[lane_index].decode(counts,.01))
                lane_results=[]
                for lane_index,lane in enumerate(lanes):
                    lane_results.append({'lane':lane_index,
                        'counts_equal':bool(counts_equal[lane_index]),
                        'decoder_equal':cpu_decisions==batch_decisions[lane_index],
                        'state':compare_state(brain,lane),
                        'batch_decisions':batch_decisions[lane_index]})
                runs.append({'repeat':repeat,'cpu_decisions':cpu_decisions,
                    'lanes':lane_results})
            lane_results=[lane for run_result in runs for lane in run_result['lanes']]
            batch_digests=[lane['state']['batch_sha256'] for lane in lane_results]
            results.append({'horizon_ms':horizon,'runs':runs,
                'counts_equal':all(lane['counts_equal'] for lane in lane_results),
                'decoder_equal':all(lane['decoder_equal'] for lane in lane_results),
                'state_equal':all(lane['state']['equal'] for lane in lane_results),
                'batch_repeat_bitwise':all(value==batch_digests[0]
                    for value in batch_digests[1:])})
        state_equal=all(result['state_equal'] and result['counts_equal']
            and result['batch_repeat_bitwise'] for result in results)
        decoder_equal=all(result['decoder_equal'] for result in results)
        passed=state_equal and decoder_equal and structure_equal
        repository=repository_identity();manifest=GRAPH.parent/'manifest.json'
        report={'schema':2,'experiment':'full-graph exact CPU batch parity',
            'structure':actual,'horizons_ms':list(horizons_ms),
            'protocol':{'lane_count':VALIDATION_LANES,'workers':VALIDATION_WORKERS,
                'repeats':VALIDATION_REPEATS,'bin_ms':10,'steps_per_bin':100},
            'identity':{'graph':dict(graph.identity),'sources':source_identity(),
                'graph_file_sha256':_file_digest(GRAPH) if GRAPH.exists() else None,
                'graph_manifest_sha256':_file_digest(manifest) if manifest.exists() else None,
                'source_lock_equal':source_lock_equal,**repository},
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
