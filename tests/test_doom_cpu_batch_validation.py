import json
import subprocess

import numpy as np

import pytest

from cpu_batch_helpers import toy_brain


READOUTS=[
    {'index':0,'type':'DNa02','side':'L'},
    {'index':1,'type':'DNa02','side':'R'},
    {'index':2,'type':'DNp09','side':'C'},
    {'index':3,'type':'MDN','side':'C'},
    {'index':4,'type':'MN9','side':'C'},
]


def _factory(tmp_path):return lambda:toy_brain(tmp_path,graph_name='validation-graph.npz')


def test_validation_writes_exact_toy_report(tmp_path):
    from doom_learning_v6.cpu_batch.validate import run
    out=tmp_path/'validation'
    expected={'release':'test','neurons':6,'edges':7,'plastic_edges':2}
    report=run(out,brain_factory=_factory(tmp_path),readouts=READOUTS,
        expected_structure=expected)
    assert report['schema']==2
    assert report['experiment']=='full-graph exact CPU batch parity'
    assert report['structure']==expected
    assert report['horizons_ms']==[40,80]
    assert report['protocol']=={'lane_count':4,'workers':4,'repeats':2,
        'bin_ms':10,'steps_per_bin':100}
    inputs=report['inputs']
    assert inputs['file']=='inputs.npz' and inputs['bins']==8 and inputs['neurons']==6
    assert len(inputs['drive_sha256'])==64 and len(inputs['file_sha256'])==64
    with np.load(out/'inputs.npz',allow_pickle=False) as saved:
        assert saved['drive'].shape==(8,6)
        assert saved['horizons_ms'].tolist()==[40,80]
    assert all(len(result['runs'])==2 for result in report['results'])
    assert all(len(run['lanes'])==4 for result in report['results']
        for run in result['runs'])
    assert all(result['batch_repeat_bitwise'] for result in report['results'])
    for result in report['results']:
        for run_result in result['runs']:
            assert len(run_result['cpu_decisions_sha256'])==64
            assert 'cpu_decisions' not in run_result
            for lane in run_result['lanes']:
                assert len(lane['batch_decisions_sha256'])==64
                assert 'batch_decisions' not in lane
    assert report['state_bitwise_equal'] is True
    assert report['decoder_equal'] is True
    assert report['structural_identity_equal'] is True
    assert report['passed'] is True
    assert report['learning_demonstrated'] is False
    assert report['biologically_validated'] is False
    assert json.loads((out/'report.json').read_text())==report
    with pytest.raises(ValueError,match='Fresh output'):run(out,
        brain_factory=_factory(tmp_path),readouts=READOUTS,expected_structure=expected)


def test_validation_identity_covers_runtime_dependencies_and_git_commit(tmp_path):
    from doom_learning_v6.cpu_batch.validate import run
    expected={'release':'test','neurons':6,'edges':7,'plastic_edges':2}
    report=run(tmp_path/'validation',brain_factory=_factory(tmp_path),readouts=READOUTS,
        expected_structure=expected,horizons_ms=(40,))
    sources=report['identity']['sources']
    for name in ['doom/engine.py','doom/native.py','doom_learning_v6/rule.py',
            'doom_learning_v6/calibration.py','doom_learning_v6/visual.py']:
        assert name in sources
    assert len(report['identity']['git_commit'])==40
    assert isinstance(report['identity']['source_tree_clean'],bool)
    assert len(report['identity']['graph_manifest_sha256'])==64


def test_state_comparison_names_the_first_changed_buffer(tmp_path):
    from doom_learning_v6.cpu_batch.backend import CpuBatchLane,SharedCpuGraph
    from doom_learning_v6.cpu_batch.validate import compare_state
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain);lane.v[2]+=1
    comparison=compare_state(brain,lane)
    assert comparison['equal'] is False
    assert comparison['fields']['v'] is False
    assert comparison['fields']['g'] is True


def test_validation_rejects_wrong_release_counts(tmp_path):
    from doom_learning_v6.cpu_batch.validate import run
    expected={'release':'MaleCNS v1.0','neurons':166700,
        'edges':25582938,'plastic_edges':4184}
    with pytest.raises(ValueError,match='structure'):
        run(tmp_path/'wrong',brain_factory=_factory(tmp_path),readouts=READOUTS,
            expected_structure=expected)


def test_runtime_source_commit_accepts_documentation_only_descendant():
    from pathlib import Path
    from doom_learning_v6.cpu_batch.validate import runtime_source_commit_compatible
    report=json.loads(Path(
        'outputs/doom-learning/cpu-batch-validation-m4pro/report.json').read_text())
    recorded=report['identity']['git_commit']
    head=subprocess.run(['git','rev-parse','HEAD'],check=True,text=True,
        stdout=subprocess.PIPE).stdout.strip()
    assert recorded!=head
    assert runtime_source_commit_compatible(recorded) is True


def test_saved_m4_validation_inputs_can_be_committed():
    result=subprocess.run(['git','check-ignore','-q',
        'outputs/doom-learning/cpu-batch-validation-m4pro/inputs.npz'],check=False)
    assert result.returncode==1
