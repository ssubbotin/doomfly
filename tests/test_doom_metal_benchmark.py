import json

import pytest

from doom_learning_v6.metal.benchmark import gate,require_metal_validation
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
