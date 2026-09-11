from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import threading

import numpy as np
import pytest

from cpu_batch_helpers import toy_brain,wide_quiet_brain


def _state_digest(lane):
    value=hashlib.sha256()
    for name in sorted(lane.arrays):value.update(name.encode()+lane.arrays[name].tobytes())
    return value.hexdigest()


def _assert_matches(lane,brain):
    for name in lane.STATE_FIELDS:
        np.testing.assert_array_equal(lane.arrays[name],getattr(brain,name),err_msg=name)
    assert lane.cursor[0]==brain.cursor
    np.testing.assert_array_equal(lane.plastic_weights,brain.weight[brain.circuit['edges']])


def _run_four_lanes(tmp_path,workers,reverse=False):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    seed=toy_brain(tmp_path,graph_name=f'seed-{workers}-{reverse}.npz')
    graph=SharedCpuGraph.from_brain(seed);pairs=[]
    for index in range(4):
        brain=toy_brain(tmp_path,graph_name=f'lane-{workers}-{reverse}-{index}.npz')
        brain.drive[[0,1,2,4]]=[30.,18.+index,20.,24.+index]
        brain.weight[3]=9.+index;brain.weight[6]=12.-index
        pairs.append((index,CpuBatchLane.from_brain(graph,brain),brain))
    if reverse:pairs.reverse()
    lanes=[lane for _,lane,_ in pairs]
    with MultiTrajectoryCpuExecutor(graph,lanes,workers=workers) as executor:
        for generation,steps in enumerate((1,40,80,100),start=1):
            for _,_,brain in pairs:brain.counts.fill(0);brain._advance_cpu(steps)
            returned=executor.advance(steps)
            assert executor.last_timing=={
                'native_wall_seconds':executor.last_timing['native_wall_seconds'],
                'lanes_advanced':4,'workers':workers,'steps':steps,
                'generation':generation,'pool_threads':workers}
            for counts,(_,lane,brain) in zip(returned,pairs):
                np.testing.assert_array_equal(counts,brain.counts);_assert_matches(lane,brain)
    return {index:_state_digest(lane) for index,lane,_ in pairs}


@pytest.mark.parametrize('workers',[1,2,4])
def test_worker_counts_and_lane_order_preserve_each_trajectory(tmp_path,workers):
    forward=_run_four_lanes(tmp_path,workers,False)
    reverse=_run_four_lanes(tmp_path,workers,True)
    assert forward==reverse


def test_five_fresh_runs_are_bitwise_repeatable(tmp_path):
    results=[_run_four_lanes(tmp_path,4,False) for _ in range(5)]
    assert all(result==results[0] for result in results[1:])


def test_changing_one_overlay_does_not_change_neighboring_lanes(tmp_path):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lanes=[CpuBatchLane.from_brain(graph,brain) for _ in range(3)]
    with MultiTrajectoryCpuExecutor(graph,lanes,workers=3) as executor:
        executor.advance(100);before=[_state_digest(lane) for lane in lanes]
        lanes[1].plastic_weights[0]+=3;executor.advance(1)
    assert lanes[0].plastic_weights[0]==lanes[2].plastic_weights[0]
    assert before[0]!=_state_digest(lanes[0]) and before[2]!=_state_digest(lanes[2])


def test_reentrant_advance_is_rejected_without_waiting_for_first_call(tmp_path):
    from doom_learning_v6.backend import BackendError
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=wide_quiet_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain);barrier=threading.Barrier(3);results=[]
    with MultiTrajectoryCpuExecutor(graph,[lane],workers=1) as executor:
        def invoke():
            barrier.wait()
            try:executor.advance(100);results.append('ok')
            except BackendError:results.append('error')
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls=[pool.submit(invoke) for _ in range(2)];barrier.wait()
            for call in calls:call.result()
    assert sorted(results)==['error','ok']


def test_close_is_idempotent_and_prevents_further_advance(tmp_path):
    from doom_learning_v6.backend import BackendError
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    executor=MultiTrajectoryCpuExecutor(graph,[CpuBatchLane.from_brain(graph,brain)],1)
    executor.close();executor.close()
    assert executor.metadata()['closed'] is True
    with pytest.raises(BackendError,match='closed'):executor.advance(1)


def test_close_waits_until_an_inflight_native_call_returns(tmp_path):
    from doom_learning_v6.cpu_batch.backend import (_NativeTiming,CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    executor=MultiTrajectoryCpuExecutor(graph,[CpuBatchLane.from_brain(graph,brain)],1)
    entered=threading.Event();release=threading.Event();closed=threading.Event()
    def blocked_advance(handle,steps,timing_pointer):
        entered.set();release.wait(5)
        timing=C.cast(timing_pointer,C.POINTER(_NativeTiming)).contents
        timing.lanes_advanced=1;timing.workers=1;timing.steps=steps
        timing.generation=1;timing.pool_threads=1
        return 0
    executor._library.df_cpu_batch_advance=blocked_advance
    def close():executor.close();closed.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        advance=pool.submit(executor.advance,1)
        assert entered.wait(1);closing=pool.submit(close)
        try:assert not closed.wait(.05)
        finally:release.set()
        advance.result();closing.result()
    assert executor.metadata()['closed'] is True
