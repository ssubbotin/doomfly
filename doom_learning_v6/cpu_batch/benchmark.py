"""Contiguous one-, two-, and four-lane CPU scaling measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time

from doom_learning.common import save_json

from .backend import CpuBatchLane,MultiTrajectoryCpuExecutor,SharedCpuGraph
from .validate import _default_factory,drive_for_bin,source_identity


def summarize(values):
    values=[float(value) for value in values]
    return {'minimum':min(values),'median':statistics.median(values),'maximum':max(values)}


def _lane_digest(lane):
    value=hashlib.sha256()
    for name in sorted(lane.arrays):value.update(name.encode()+lane.arrays[name].tobytes())
    return value.hexdigest()


def _restore(lane,snapshot):
    for name,array in snapshot.items():lane.arrays[name][:]=array


def _peak_rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system()=='Darwin' else value*1024)


def _physical_cpu_count():
    if platform.system()=='Darwin':
        import subprocess
        return int(subprocess.run(['sysctl','-n','hw.physicalcpu'],check=True,text=True,
            stdout=subprocess.PIPE).stdout)
    topology=set();path=Path('/proc/cpuinfo')
    if path.exists():
        physical=core=None
        for line in [*path.read_text().splitlines(),'']:
            if line.startswith('physical id'):physical=line.split(':',1)[1].strip()
            elif line.startswith('core id'):core=line.split(':',1)[1].strip()
            elif not line and physical is not None and core is not None:
                topology.add((physical,core));physical=core=None
    return len(topology) or None


def _sample(executor,brain,lanes,horizon_ms):
    native=0.;started=time.perf_counter()
    for index in range(horizon_ms//10):
        drive=drive_for_bin(brain,index)
        for lane in lanes:lane.drive[:]=drive
        executor.advance(100);native+=executor.last_timing['native_wall_seconds']
    wall=time.perf_counter()-started;brain_seconds=horizon_ms/1000*len(lanes)
    return {'wall_seconds':wall,'native_wall_seconds':native,
        'brain_seconds':brain_seconds,
        'aggregate_brain_seconds_per_wall_second':brain_seconds/wall,
        'per_lane_latency_seconds':wall/len(lanes),
        'lane_state_sha256':[_lane_digest(lane) for lane in lanes]}


def run(out,validation,repetitions=5,lane_counts=(1,2,4),*,brain_factory=None,
        horizon_ms=40):
    if repetitions<5:raise ValueError('At least five benchmark repetitions required')
    if tuple(lane_counts)!=(1,2,4):raise ValueError('Benchmark requires 1, 2, and 4 lanes')
    if horizon_ms<10 or horizon_ms%10:raise ValueError('Benchmark horizon must use 10 ms bins')
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    try:validated=json.loads(Path(validation).read_text())
    except (OSError,ValueError,TypeError):
        raise ValueError('A passing CPU batch validation report is required') from None
    if validated.get('passed') is not True:
        raise ValueError('A passing CPU batch validation report is required')
    factory=_default_factory if brain_factory is None else brain_factory
    brain=factory()
    try:
        graph=SharedCpuGraph.from_brain(brain)
        if validated.get('identity',{}).get('graph')!=graph.identity or \
                validated.get('identity',{}).get('sources')!=source_identity():
            raise ValueError('CPU batch validation identity mismatch')
        results={}
        for lane_count in lane_counts:
            brain.reset();template=CpuBatchLane.from_brain(graph,brain)
            snapshot={name:array.copy() for name,array in template.arrays.items()}
            lanes=[CpuBatchLane.from_brain(graph,brain) for _ in range(lane_count)]
            with MultiTrajectoryCpuExecutor(graph,lanes,workers=lane_count) as executor:
                _sample(executor,brain,lanes,horizon_ms)
                samples=[]
                for _ in range(repetitions):
                    for lane in lanes:_restore(lane,snapshot)
                    samples.append(_sample(executor,brain,lanes,horizon_ms))
                metadata=executor.metadata()
            digests=[sample['lane_state_sha256'] for sample in samples]
            repeat=all(value==digests[0] for value in digests[1:])
            results[str(lane_count)]={'workers':lane_count,'samples':samples,
                'wall_seconds':summarize(sample['wall_seconds'] for sample in samples),
                'native_wall_seconds':summarize(sample['native_wall_seconds'] for sample in samples),
                'aggregate_brain_seconds_per_wall_second':summarize(
                    sample['aggregate_brain_seconds_per_wall_second'] for sample in samples),
                'per_lane_latency_seconds':summarize(
                    sample['per_lane_latency_seconds'] for sample in samples),
                'repeat_bitwise':repeat,'shared_graph_bytes':metadata['shared_graph_bytes'],
                'lane_bytes':metadata['lane_bytes'][0],
                'total_lane_bytes':metadata['total_lane_bytes']}
        baseline=results['1']['aggregate_brain_seconds_per_wall_second']['median']
        for lane_count in lane_counts:
            throughput=results[str(lane_count)]['aggregate_brain_seconds_per_wall_second']['median']
            results[str(lane_count)]['scaling_efficiency']=throughput/(baseline*lane_count)
        report={'schema':1,'experiment':'shared-graph CPU trajectory scaling',
            'lane_counts':list(lane_counts),'repetitions':repetitions,
            'horizon_ms':horizon_ms,'results':results,
            'peak_rss_bytes':_peak_rss_bytes(),
            'system':{'platform':platform.platform(),'architecture':platform.machine(),
                'logical_cpus':os.cpu_count(),'physical_cpus':_physical_cpu_count(),
                'load_average':list(os.getloadavg())},
            'identity':{'graph':graph.identity,'sources':source_identity(),
                'build':metadata['build']},
            'correctness_validation_passed':True,'complete':True,
            'learning_demonstrated':False,'biologically_validated':False,
            'interpretation':'Propagation throughput only. These measurements do not establish learning or biological validity.'}
        out.mkdir(parents=True);save_json(out/'report.json',report);return report
    finally:
        if hasattr(brain,'backend'):brain.backend.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    parser.add_argument('--validation',required=True);parser.add_argument('--repetitions',type=int,default=5)
    args=parser.parse_args();print(json.dumps(run(args.out,args.validation,args.repetitions),indent=2))


if __name__=='__main__':main()
