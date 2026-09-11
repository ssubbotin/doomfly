import json

import pytest

from cpu_batch_helpers import toy_brain
from test_doom_cpu_batch_validation import READOUTS


def _factory(tmp_path):return lambda:toy_brain(tmp_path,graph_name='benchmark-graph.npz')


def _validation(tmp_path):
    from doom_learning_v6.cpu_batch.validate import run
    expected={'release':'test','neurons':6,'edges':7,'plastic_edges':2}
    out=tmp_path/'validation'
    run(out,brain_factory=_factory(tmp_path),readouts=READOUTS,expected_structure=expected)
    return out/'report.json'


def test_summary_uses_literal_median_and_extrema():
    from doom_learning_v6.cpu_batch.benchmark import summarize
    assert summarize([3.,1.,5.,2.,4.])=={'minimum':1.,'median':3.,'maximum':5.}


def test_benchmark_records_raw_scaling_and_memory_without_speed_gate(tmp_path):
    from doom_learning_v6.cpu_batch.benchmark import run
    out=tmp_path/'benchmark';validation=_validation(tmp_path)
    report=run(out,validation,repetitions=5,brain_factory=_factory(tmp_path))
    assert report['schema']==2 and report['lane_counts']==[1,2,4]
    assert report['repetitions']==5 and report['complete'] is True
    assert report['correctness_validation_passed'] is True
    assert report['identity']['build']['abi_version']==1
    assert len(report['identity']['validation_report_sha256'])==64
    assert report['learning_demonstrated'] is False
    assert report['biologically_validated'] is False
    assert 'passed' not in report
    for lanes in ['1','2','4']:
        result=report['results'][lanes]
        assert len(result['samples'])==5
        assert result['repeat_bitwise'] is True
        assert result['shared_graph_bytes']>0 and result['lane_bytes']>0
        assert result['aggregate_brain_seconds_per_wall_second']['median']>0
        assert result['batch_latency_seconds']['median']>0
        assert result['amortized_wall_seconds_per_lane']['median']>0
        assert 'per_lane_latency_seconds' not in result
    assert report['results']['1']['scaling_efficiency']==1.
    assert json.loads((out/'report.json').read_text())==report


@pytest.mark.parametrize('repetitions,lane_counts,match',[
    (4,(1,2,4),'five'),(5,(1,4),'1, 2, and 4')])
def test_benchmark_rejects_incomplete_protocol(tmp_path,repetitions,lane_counts,match):
    from doom_learning_v6.cpu_batch.benchmark import run
    with pytest.raises(ValueError,match=match):
        run(tmp_path/'benchmark',_validation(tmp_path),repetitions=repetitions,
            lane_counts=lane_counts,brain_factory=_factory(tmp_path))


def test_benchmark_rejects_tampered_validation_protocol(tmp_path):
    from doom_learning_v6.cpu_batch.benchmark import run
    validation=_validation(tmp_path)
    report=json.loads(validation.read_text());report['protocol']['lane_count']=1
    validation.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='protocol'):
        run(tmp_path/'benchmark',validation,repetitions=5,brain_factory=_factory(tmp_path))
