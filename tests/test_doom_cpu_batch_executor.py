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
    control=[CpuBatchLane.from_brain(graph,brain) for _ in range(3)]
    changed=[CpuBatchLane.from_brain(graph,brain) for _ in range(3)]
    for lane in [*control,*changed]:lane.drive[1]=30
    with MultiTrajectoryCpuExecutor(graph,control,workers=3) as baseline, \
            MultiTrajectoryCpuExecutor(graph,changed,workers=3) as experiment:
        baseline.advance(100);experiment.advance(100)
        changed[1].plastic_weights[0]+=3
        baseline.advance(100);experiment.advance(100)
    assert _state_digest(changed[0])==_state_digest(control[0])
    assert _state_digest(changed[2])==_state_digest(control[2])
    assert _state_digest(changed[1])!=_state_digest(control[1])


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


@pytest.mark.parametrize('corrupt,match',[
    (lambda lane:lane.cursor.__setitem__(0,-1),'cursor'),
    (lambda lane:lane.nactive.__setitem__(0,-1),'active count'),
    (lambda lane:lane.nactive.__setitem__(0,len(lane.active)+1),'active count'),
    (lambda lane:(lane.nactive.__setitem__(0,1),lane.active.__setitem__(0,-1)),'active index'),
    (lambda lane:(lane.nactive.__setitem__(0,2),lane.active.__setitem__(slice(0,2),[1,1]),
        lane.active_flag.__setitem__(1,1)),'unique'),
    (lambda lane:(lane.nactive.__setitem__(0,1),lane.active.__setitem__(0,1),
        lane.active_flag.fill(0)),'active flags'),
    (lambda lane:lane.queue_count.__setitem__(0,-1),'queue count'),
    (lambda lane:lane.queue_count.__setitem__(0,lane.queue.shape[1]+1),'queue count'),
    (lambda lane:(lane.queue_count.__setitem__(0,1),lane.queue.__setitem__((0,0),-1)),
        'queued neuron'),
    (lambda lane:lane.queue_count.__setitem__(
        (int(lane.cursor[0])+18)%lane.queue.shape[0],1),'future queue slot'),
    (lambda lane:lane.last.__setitem__(0,-2),'timestamp'),
    (lambda lane:lane.eligibility_last.__setitem__(0,int(lane.cursor[0])+1),'timestamp'),
    (lambda lane:lane.modulation_last.__setitem__(0,-1),'timestamp'),
])
def test_advance_rejects_unsafe_mutable_index_state_before_native_call(
        tmp_path,corrupt,match):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    with MultiTrajectoryCpuExecutor(graph,[lane],1) as executor:
        corrupt(lane);called=[]
        def forbidden(*arguments):called.append(arguments);return 0
        executor._library.df_cpu_batch_advance=forbidden
        with pytest.raises(ValueError,match=match):executor.advance(1)
        assert called==[]


def test_executor_detects_registered_buffer_reallocation_before_advance(tmp_path):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    with MultiTrajectoryCpuExecutor(graph,[lane],1) as executor:
        lane.v.resize((graph.neurons*2,),refcheck=False)
        with pytest.raises(ValueError,match='registered v buffer'):
            executor.advance(1)


def test_advance_rejects_cursor_that_cannot_complete_without_overflow(tmp_path):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    with MultiTrajectoryCpuExecutor(graph,[lane],1) as executor:
        lane.cursor[0]=np.iinfo(np.int64).max
        lane.last.fill(lane.cursor[0]);lane.eligibility_last.fill(lane.cursor[0])
        lane.modulation_last.fill(lane.cursor[0]);called=[]
        executor._library.df_cpu_batch_advance=lambda *arguments:called.append(1) or 0
        with pytest.raises(ValueError,match='overflow'):executor.advance(1)
        assert called==[]


def test_executor_checks_native_abi_before_registering_buffers(tmp_path,monkeypatch):
    import doom_learning_v6.cpu_batch.backend as backend
    from doom_learning_v6.backend import BackendError
    brain=toy_brain(tmp_path);graph=backend.SharedCpuGraph.from_brain(brain)
    lane=backend.CpuBatchLane.from_brain(graph,brain);created=[]
    class Function:
        def __init__(self,result):self.result=result
        def __call__(self,*arguments):return self.result(*arguments) if callable(self.result) else self.result
    class Library:
        df_cpu_batch_abi_version=Function(999)
        df_cpu_batch_create=Function(lambda *arguments:created.append(arguments) or 0)
        df_cpu_batch_advance=Function(0)
        df_cpu_batch_error=Function(None)
        df_cpu_batch_destroy=Function(None)
    monkeypatch.setattr(backend.C,'CDLL',lambda path:Library())
    with pytest.raises(BackendError,match='ABI version'):
        backend.MultiTrajectoryCpuExecutor(graph,[lane],1)
    assert created==[]


def test_recoverable_native_failure_does_not_poison_executor(tmp_path):
    from doom_learning_v6.backend import BackendError
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lanes=[CpuBatchLane.from_brain(graph,brain) for _ in range(2)]
    with MultiTrajectoryCpuExecutor(graph,lanes,2) as executor:
        executor._library.df_cpu_batch_advance=lambda *arguments:1
        executor._library.df_cpu_batch_error=lambda handle:b'invalid state'
        with pytest.raises(BackendError,match='invalid state'):executor.advance(1)
        assert executor.metadata()['poisoned'] is False


def test_poisoned_native_handle_preserves_failure_and_rejects_later_calls(tmp_path):
    from doom_learning_v6.backend import BackendError
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    with MultiTrajectoryCpuExecutor(graph,[lane],1) as executor:
        executor._library.df_cpu_batch_advance=lambda *arguments:2
        executor._library.df_cpu_batch_error=lambda handle:b'worker failed'
        with pytest.raises(BackendError,match='worker failed'):executor.advance(1)
        assert executor.metadata()['poisoned'] is True
        called=[];executor._library.df_cpu_batch_advance=lambda *arguments:called.append(1) or 0
        with pytest.raises(BackendError,match='poisoned'):executor.advance(1)
        assert called==[]


@pytest.mark.parametrize('corrupt',[
    lambda lane:lane.cursor.__setitem__(0,-1),
    lambda lane:lane.queue_count.__setitem__(0,-1),
    lambda lane:(lane.nactive.__setitem__(0,1),lane.active.__setitem__(0,-1)),
    lambda lane:lane.queue_count.__setitem__(
        (int(lane.cursor[0])+18)%lane.queue.shape[0],lane.queue.shape[1]),
    lambda lane:lane.last.__setitem__(0,-2),
])
def test_native_registration_rejects_unsafe_index_state(tmp_path,corrupt):
    from doom_learning_v6.cpu_batch.backend import (CpuBatchLane,
        MultiTrajectoryCpuExecutor,SharedCpuGraph)
    brain=toy_brain(tmp_path);graph=SharedCpuGraph.from_brain(brain)
    lane=CpuBatchLane.from_brain(graph,brain)
    executor=MultiTrajectoryCpuExecutor(graph,[lane],1)
    executor.close();corrupt(lane);handle=C.c_void_p()
    status=executor._library.df_cpu_batch_create(C.byref(executor._native_graph),
        executor._native_lanes,1,1,C.byref(handle))
    try:assert status!=0
    finally:
        if handle.value:executor._library.df_cpu_batch_destroy(handle)


def test_native_kernel_checks_queue_capacity_and_uses_bounded_future_slot():
    from pathlib import Path
    source=Path('doom_learning_v6/cpu_batch/executor.cpp').read_text()
    assert 'if (lane.queue_count[future] >= n)' in source
    assert 'const int future = (slot + delay) % slots;' in source
