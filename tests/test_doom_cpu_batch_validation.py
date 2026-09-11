import json

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
    assert report['schema']==1
    assert report['experiment']=='full-graph exact CPU batch parity'
    assert report['structure']==expected
    assert report['horizons_ms']==[40,80]
    assert report['state_bitwise_equal'] is True
    assert report['decoder_equal'] is True
    assert report['structural_identity_equal'] is True
    assert report['passed'] is True
    assert report['learning_demonstrated'] is False
    assert report['biologically_validated'] is False
    assert json.loads((out/'report.json').read_text())==report
    with pytest.raises(ValueError,match='Fresh output'):run(out,
        brain_factory=_factory(tmp_path),readouts=READOUTS,expected_structure=expected)


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
