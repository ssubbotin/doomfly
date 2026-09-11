import numpy as np
import pytest

from cpu_batch_helpers import toy_brain


def test_shared_graph_freezes_finalized_arrays_and_builds_compact_slots(tmp_path):
    from doom_learning_v6.cpu_batch.backend import SharedCpuGraph
    brain=toy_brain(tmp_path);brain.weight[0]=4.5;brain.weight[3]=9.5
    graph=SharedCpuGraph.from_brain(brain)
    assert graph.neurons==6 and graph.edges==7 and graph.plastic_edges==2
    assert graph.plastic_slot.dtype==np.int16
    np.testing.assert_array_equal(graph.plastic_slot,np.array([-1,-1,-1,0,-1,-1,1],dtype=np.int16))
    np.testing.assert_array_equal(graph.base_weight,brain.weight)
    assert all(not array.flags.writeable for array in graph.arrays.values())
    assert graph.shared_bytes==sum(array.nbytes for array in graph.arrays.values())
    assert graph.metadata()['identity']['base_weight_sha256']


def test_lane_copies_only_mutable_state_and_plastic_overlay(tmp_path):
    from doom_learning_v6.cpu_batch.backend import CpuBatchLane,SharedCpuGraph
    brain=toy_brain(tmp_path);brain.cursor=17;brain.v[1]=-48;brain.eligibility[1]=2.5
    brain.weight[3]=8.5;brain.weight[6]=9.25
    graph=SharedCpuGraph.from_brain(brain);lane=CpuBatchLane.from_brain(graph,brain)
    np.testing.assert_array_equal(lane.plastic_weights,[8.5,9.25])
    assert lane.cursor.dtype==np.int64 and lane.cursor.tolist()==[17]
    assert lane.v[1]==-48 and lane.eligibility[1]==2.5
    assert not np.shares_memory(lane.v,brain.v)
    assert not np.shares_memory(lane.plastic_weights,graph.base_weight)
    assert lane.lane_bytes==sum(array.nbytes for array in lane.arrays.values())
    lane.v[1]=-47;lane.plastic_weights[0]=7.5;lane.copy_to_brain(brain)
    assert brain.v[1]==-47 and brain.weight[3]==7.5 and brain.cursor==17


@pytest.mark.parametrize('mutation,match',[
    (lambda b:setattr(b,'ptr',b.ptr.astype(np.int32)),'ptr'),
    (lambda b:setattr(b,'post',np.repeat(b.post,2)[::2]),'contiguous'),
    (lambda b:b.weight.__setitem__(0,np.nan),'finite'),
    (lambda b:b.ptr.__setitem__(0,1),'CSR'),
    (lambda b:b.circuit.__setitem__('edges',np.array([3,3],dtype=np.int64)),'unique'),
    (lambda b:b.circuit.__setitem__('edges',np.array([99],dtype=np.int64)),'range'),
    (lambda b:b.circuit.__setitem__('edges',np.arange(32768,dtype=np.int64)),'32,767'),
    (lambda b:b.circuit.__setitem__('kc_mask',np.zeros(5,dtype=np.uint8)),'kc_mask'),
])
def test_shared_graph_rejects_malformed_inputs(tmp_path,mutation,match):
    from doom_learning_v6.cpu_batch.backend import SharedCpuGraph
    brain=toy_brain(tmp_path);mutation(brain)
    with pytest.raises(ValueError,match=match):SharedCpuGraph.from_brain(brain)


def test_lane_rejects_a_different_graph_identity(tmp_path):
    from doom_learning_v6.cpu_batch.backend import CpuBatchLane,SharedCpuGraph
    first=toy_brain(tmp_path,graph_name='first.npz')
    second=toy_brain(tmp_path,graph_name='second.npz');second.rest[0]=-51
    graph=SharedCpuGraph.from_brain(first)
    with pytest.raises(ValueError,match='identity'):CpuBatchLane.from_brain(graph,second)


def test_executor_rejects_aliased_lanes_before_native_creation(tmp_path):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    with pytest.raises(ValueError,match='alias'):
        MultiTrajectoryCpuExecutor(graph,[lane,lane],workers=2)
    with pytest.raises(ValueError,match='workers'):
        MultiTrajectoryCpuExecutor(graph,[lane],workers=2)
    with MultiTrajectoryCpuExecutor(graph,[lane],workers=1) as executor:
        assert executor.graph is graph and executor.lanes==[lane]
