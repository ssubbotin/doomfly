import hashlib

import numpy as np
import pytest

from cpu_batch_helpers import toy_brain


def _assert_lane_matches_brain(lane,brain):
    for name in lane.STATE_FIELDS:
        np.testing.assert_array_equal(lane.arrays[name],getattr(brain,name),err_msg=name)
    assert int(lane.cursor[0])==brain.cursor
    np.testing.assert_array_equal(lane.plastic_weights,
        brain.weight[brain.circuit['edges']],err_msg='plastic_weights')


def _lane_digest(lane):
    value=hashlib.sha256()
    for name in sorted(lane.arrays):
        value.update(name.encode());value.update(lane.arrays[name].tobytes())
    return value.hexdigest()


def test_single_lane_matches_cpu_oracle_after_every_advance(tmp_path):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    source=toy_brain(tmp_path,graph_name='source.npz')
    reference=toy_brain(tmp_path,graph_name='reference.npz')
    graph=SharedCpuGraph.from_brain(source);lane=CpuBatchLane.from_brain(graph,source)
    baseline_digest=hashlib.sha256(graph.base_weight.tobytes()).hexdigest()
    patterns=[{0:30.},{0:30.,1:18.},{0:30.,1:24.,2:20.},
        {0:30.,1:24.,2:20.,4:30.},{0:30.,1:24.,2:20.,4:30.,5:12.}]
    with MultiTrajectoryCpuExecutor(graph,[lane],workers=1) as executor:
        for call,(steps,pattern) in enumerate(zip((1,18,22,37,100),patterns)):
            reference.drive.fill(0);lane.drive.fill(0)
            for neuron,current in pattern.items():
                reference.drive[neuron]=current;lane.drive[neuron]=current
            if call==3:
                reference.weight[3]=12.5;lane.plastic_weights[0]=12.5
            reference.counts.fill(0);expected_elapsed=reference._advance_cpu(steps)
            actual=executor.advance(steps)
            assert expected_elapsed>=0 and len(actual)==1
            np.testing.assert_array_equal(actual[0],reference.counts)
            _assert_lane_matches_brain(lane,reference)
    assert lane.eligibility[1]>0 and lane.modulation[3]>0
    assert hashlib.sha256(graph.base_weight.tobytes()).hexdigest()==baseline_digest


@pytest.mark.parametrize('steps',[0,101])
def test_invalid_step_count_preserves_lane_state(tmp_path,steps):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    with MultiTrajectoryCpuExecutor(graph,[lane],workers=1) as executor:
        before=_lane_digest(lane)
        with pytest.raises(ValueError,match='1 through 100'):executor.advance(steps)
        assert _lane_digest(lane)==before
