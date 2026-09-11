"""M4 Pro timing gates and validated Metal evidence checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import statistics
import subprocess
import time

import numpy as np


METAL_TIMING_TOTALS=(
    'native_total_seconds','encode_seconds','commit_call_seconds','wait_call_seconds',
    'full_upload_seconds','full_upload_bytes','materialize_seconds','materialize_bytes',
    'drive_copy_seconds','drive_copy_bytes','counts_clear_seconds','counts_copy_seconds',
    'counts_copy_bytes','native_event_copy_seconds','native_event_copy_bytes',
    'event_conversion_sort_seconds','eligibility_seconds','sparse_weight_update_seconds',
    'sparse_weight_update_bytes')


def gate(metrics):
    cpu=float(metrics['cpu_median']);metal=float(metrics['metal_median'])
    result={**metrics,'speedup':cpu/metal,
        'real_time_ratio':float(metrics['simulated_seconds'])/metal}
    result['speed']=result['speedup']>=2
    result['memory']=float(metrics['peak_rss_gib'])<8
    result['memory_pressure']=not bool(metrics['critical_memory_pressure'])
    result['swap']=not bool(metrics['sustained_swap_growth'])
    result['passed']=all(result[k] for k in ['speed','memory','memory_pressure','swap'])
    return result


def _contains(actual,expected):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(key in actual and _contains(actual[key],value)
            for key,value in expected.items())
    return actual==expected


def require_metal_validation(path,expected_identity):
    try:report=json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):
        raise ValueError('A passing Metal validation report is required') from None
    if (report.get('passed') is not True or report.get('gates',{}).get('passed') is not True
            or not _contains(report.get('identity'),expected_identity)):
        raise ValueError('A passing Metal validation report is required')
    return report


def _file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def portable_backend_metadata(metadata):
    if metadata['name']=='cpu':return metadata
    keys=['schema','abi_version','sources','binaries','compiler','sdk','macos','architecture','metal_language']
    return {**{key:metadata[key] for key in keys},
        'device':{key:value for key,value in metadata['device'].items() if key!='registry_id'}}


def current_preflight_identity():
    from doom_learning.common import GRAPH
    from .build import probe
    from .validate import source_identity
    metadata=portable_backend_metadata({'name':'metal',**probe()})
    return {'validation_sources':source_identity(),'graph_file_sha256':_file_digest(GRAPH),
        'metal_sources':metadata['sources'],'metal_binaries':metadata['binaries'],
        'metal_environment':{key:metadata[key] for key in ['compiler','sdk','macos','architecture','metal_language']},
        'metal_device':metadata['device']}


def model_identity(brain):
    from doom_learning.common import GRAPH,digest
    with np.load(GRAPH,allow_pickle=False) as graph:original_weight=digest(graph['weight'])
    configuration=json.dumps(brain.configuration_signature(),sort_keys=True,separators=(',',':'))
    return {'ids_sha256':digest(brain.ids),'out_ptr_sha256':digest(brain.ptr),
        'out_post_sha256':digest(brain.post),'original_weight_sha256':original_weight,
        'plastic_edges_sha256':digest(brain.circuit['edges']),
        'configuration_sha256':hashlib.sha256(configuration.encode()).hexdigest()}


def _memory_status():
    pressure=subprocess.run(['memory_pressure','-Q'],check=True,text=True,
        stdout=subprocess.PIPE).stdout
    match=re.search(r'free percentage:\s*(\d+)%',pressure)
    if match is None:raise RuntimeError('Cannot parse macOS memory pressure')
    swap=subprocess.run(['sysctl','-n','vm.swapusage'],check=True,text=True,
        stdout=subprocess.PIPE).stdout
    used=re.search(r'used = ([0-9.]+)([MG])',swap)
    if used is None:raise RuntimeError('Cannot parse macOS swap usage')
    scale=1024 if used.group(2)=='G' else 1
    return {'free_percent':int(match.group(1)),'swap_used_mib':float(used.group(1))*scale}


def _sample(brain,frames):
    from .validate import TRACE
    started=time.perf_counter();neural=rule=gpu=command=0.
    encoders=dispatches=mark_threads=gather_threads=indirect_dispatches=edge_bitmap_words=0
    metal_timing={name:0 for name in METAL_TIMING_TOTALS}
    for frame,(_,stimulated) in zip(frames,TRACE):
        stimulation=(brain.circuit['dan'],4.) if stimulated else None
        _,elapsed=brain.rgb_step(frame,10,learning=True,stimulation=stimulation)
        neural+=elapsed;rule+=brain.last_rule_seconds
        if brain.backend.name=='metal':
            timing=brain.backend.last_timing
            gpu+=timing['gpu_seconds']
            native_total=timing.get('native_total_seconds',timing.get('host_seconds',0.))
            command+=native_total
            for name in METAL_TIMING_TOTALS:
                metal_timing[name]+=native_total if name=='native_total_seconds' else timing.get(name,0)
            encoders+=timing['encoder_count'];dispatches+=timing['dispatch_count']
            mark_threads+=timing['mark_grid_threads'];gather_threads+=timing['gather_grid_threads']
            indirect_dispatches+=timing['indirect_dispatch_count']
            edge_bitmap_words=max(edge_bitmap_words,timing['edge_bitmap_words'])
    wall=time.perf_counter()-started
    return {'wall_seconds':wall,'neural_seconds':neural,'gpu_seconds':gpu,
        'command_seconds':command,
        'synchronization_seconds':max(0.,neural-command) if brain.backend.name=='metal' else 0.,
        'kernel_seconds':gpu if brain.backend.name=='metal' else neural,
        'plasticity_seconds':rule,'vision_and_python_seconds':max(0.,wall-neural-rule),
        'encoder_count':encoders,'dispatch_count':dispatches,
        'mark_grid_threads':mark_threads,
        'gather_grid_threads':gather_threads,
        'indirect_dispatch_count':indirect_dispatches,
        'edge_bitmap_words':edge_bitmap_words,**metal_timing}


def run(out,validation,repetitions=5):
    from doom_learning.common import GRAPH,save_json
    from doom_learning_v2.vision import frame_for
    from doom_learning_v6.calibration import calibrated_brain
    from .validate import TRACE
    if repetitions<5:raise ValueError('At least five benchmark repetitions required')
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    expected=current_preflight_identity();validated=require_metal_validation(validation,expected)
    out.mkdir(parents=True);frames=np.stack([frame_for(label) for label,_ in TRACE])
    checkpoint=out/'initial.npz';seed=calibrated_brain();seed.checkpoint(checkpoint);seed.backend.close();del seed
    constructed=time.perf_counter();cpu=calibrated_brain(backend='cpu')
    cpu_construction=time.perf_counter()-constructed
    constructed=time.perf_counter();metal=calibrated_brain(backend='metal')
    metal_construction=time.perf_counter()-constructed
    require_metal_validation(validation,{**expected,**model_identity(metal)})
    metal.backend.ensure_initialized();metal_initialization=metal.backend.initialization_timing
    before=_memory_status();cpu.restore(checkpoint);metal.restore(checkpoint)
    _sample(cpu,frames);_sample(metal,frames)
    cpu_samples=[];metal_samples=[]
    for _ in range(repetitions):
        cpu.restore(checkpoint);cpu_samples.append(_sample(cpu,frames))
        metal.restore(checkpoint);metal_samples.append(_sample(metal,frames))
    after=_memory_status()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
    metrics={'cpu_median':statistics.median(x['neural_seconds'] for x in cpu_samples),
        'metal_median':statistics.median(x['neural_seconds'] for x in metal_samples),
        'simulated_seconds':len(TRACE)*.01,'peak_rss_gib':peak,
        'critical_memory_pressure':min(before['free_percent'],after['free_percent'])<5,
        'sustained_swap_growth':after['swap_used_mib']>before['swap_used_mib']}
    summary={name:{field:{'minimum':min(row[field] for row in samples),
            'median':statistics.median(row[field] for row in samples),
            'maximum':max(row[field] for row in samples)}
        for field in samples[0]} for name,samples in [('cpu',cpu_samples),('metal',metal_samples)]}
    result=gate(metrics)
    report={'schema':1,'experiment':'M4 Pro full-graph backend benchmark',
        'validation_report_sha256':_file_digest(validation),'validation_identity':validated['identity'],
        'graph_file_sha256':_file_digest(GRAPH),'repetitions':repetitions,
        'cpu_samples':cpu_samples,'metal_samples':metal_samples,'timing_summary':summary,
        'construction_seconds':{'cpu':cpu_construction,'metal':metal_construction},
        'metal_initialization':metal_initialization,'memory':{'before':before,'after':after},
        'system_load':{'load_average':list(os.getloadavg()),'logical_cpus':os.cpu_count()},
        'metal':portable_backend_metadata(metal.backend.metadata()),'gate':result,
        'passed':result['passed'],'interpretation':'Wall-clock viability only. Speed does not change sample efficiency or establish learning.'}
    save_json(out/'report.json',report);cpu.backend.close();metal.backend.close();return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--validation',default='outputs/doom-learning/metal-validation-m4pro/report.json')
    parser.add_argument('--repetitions',type=int,default=5);args=parser.parse_args()
    report=run(args.out,args.validation,args.repetitions);print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
