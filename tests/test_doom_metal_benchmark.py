import json

import numpy as np
import pytest

from doom_learning_v6.metal.benchmark import _sample,gate,require_metal_validation
from doom_learning_v6.metal.backend import Timing
from doom_learning_v6.survival import build_parser
from doom_learning.common import provenance_sources


def test_training_rejects_missing_or_stale_metal_validation(tmp_path):
    with pytest.raises(ValueError,match='passing Metal validation'):
        require_metal_validation(tmp_path/'missing.json',{'source':'current'})
    path=tmp_path/'report.json'
    path.write_text(json.dumps({'passed':True,'gates':{'passed':True},
        'identity':{'source':'old'}}))
    with pytest.raises(ValueError,match='passing Metal validation'):
        require_metal_validation(path,{'source':'current'})


def test_matching_passing_validation_is_accepted(tmp_path):
    identity={'source':'current','graph':'exact'};path=tmp_path/'report.json'
    path.write_text(json.dumps({'passed':True,'gates':{'passed':True},'identity':identity}))
    assert require_metal_validation(path,identity)['identity']==identity


def test_benchmark_gate_requires_speed_and_memory():
    report=gate({'cpu_median':4.0,'metal_median':1.9,'simulated_seconds':2.,
        'peak_rss_gib':7.0,'critical_memory_pressure':False,'sustained_swap_growth':False})
    assert report['speedup']>2
    assert report['real_time_ratio']>1
    assert report['passed'] is True
    assert not gate({**report,'metal_median':2.1})['passed']
    assert not gate({**report,'peak_rss_gib':8.0})['passed']


def test_survival_parser_exposes_explicit_backend_and_evidence():
    args=build_parser().parse_args(['--backend','metal','--metal-validation','report.json'])
    assert args.backend=='metal'
    assert args.metal_validation=='report.json'


def test_provenance_source_discovery_recurses_for_native_metal_files(tmp_path):
    folder=tmp_path/'model';nested=folder/'metal';nested.mkdir(parents=True)
    for name in ['brain.py','kernel.cpp','api.h','backend.mm','kernels.metal','ignored.txt']:
        target=folder/name if name in ['brain.py','ignored.txt'] else nested/name
        target.write_text(name)
    assert [p.relative_to(tmp_path).as_posix() for p in provenance_sources(tmp_path,['model'])]==[
        'model/brain.py','model/metal/api.h','model/metal/backend.mm',
        'model/metal/kernel.cpp','model/metal/kernels.metal']


def test_sample_aggregates_metal_dispatch_work():
    class Backend:
        name='metal'
        last_timing={}
    class Brain:
        backend=Backend()
        circuit={'dan':np.array([0],dtype=np.int32)}
        last_rule_seconds=.05
        def rgb_step(self,*args,**kwargs):
            self.backend.last_timing={'gpu_seconds':.1,'native_total_seconds':.2,
                'encode_seconds':.01,'commit_call_seconds':.002,
                'wait_call_seconds':.188,'full_upload_seconds':.03,
                'full_upload_bytes':100,'materialize_seconds':.04,
                'materialize_bytes':200,'drive_copy_seconds':.001,
                'drive_copy_bytes':12,'counts_clear_seconds':.003,
                'counts_copy_seconds':.004,'counts_copy_bytes':16,
                'native_event_copy_seconds':.005,'native_event_copy_bytes':32,
                'event_conversion_sort_seconds':.006,'eligibility_seconds':.007,
                'sparse_weight_update_seconds':.008,'sparse_weight_update_bytes':48,
                'encoder_count':1,'dispatch_count':502,
                'mark_grid_threads':7,'gather_grid_threads':9,
                'indirect_dispatch_count':100,'edge_bitmap_words':11}
            return np.zeros(1,dtype=np.int32),.3
    result=_sample(Brain(),[None]*4)
    assert result['encoder_count']==4
    assert result['dispatch_count']==2008
    assert result['mark_grid_threads']==28
    assert result['gather_grid_threads']==36
    assert result['indirect_dispatch_count']==400
    assert result['edge_bitmap_words']==11
    assert result['full_upload_bytes']==400
    assert result['materialize_bytes']==800
    assert result['counts_copy_bytes']==64
    assert result['native_event_copy_bytes']==128
    assert result['sparse_weight_update_bytes']==192
    assert result['event_conversion_sort_seconds']==pytest.approx(.024)


def test_native_timing_contract_separates_command_phases():
    names=[name for name,_ in Timing._fields_]
    assert names[:8]==[
        'native_total_seconds','gpu_seconds','counts_clear_seconds',
        'encode_seconds','commit_call_seconds','wait_call_seconds',
        'native_event_copy_seconds','native_event_copy_bytes']
