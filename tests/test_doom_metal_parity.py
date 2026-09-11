import sys

import numpy as np
import pytest

from doom_learning_v6.brain import MemoryBrain


pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def paired_brains(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True)
    n=5
    pre=np.array([0,0,0,1,2,3,4],dtype=np.int32)
    post=np.array([0,1,1,2,1,1,0],dtype=np.int32)
    weight=np.array([100,35,-10,45,-20,.275,-5],dtype=np.float32)
    path=tmp_path/'graph.npz'
    np.savez(path,ptr=np.r_[0,np.cumsum(np.bincount(pre,minlength=n))].astype(np.int64),
        post=post,weight=weight,ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['test']*n))
    circuit={'edges':np.array([1],dtype=np.int64),'pre':np.array([0],dtype=np.int32),
        'kc_mask':np.array([1,0,0,0,0],dtype=np.uint8),
        'dan_index':np.array([-1,-1,-1,0,-1],dtype=np.int8),
        'gain':np.array([[1]],dtype=np.float32),'kc':np.array([0]),
        'mb':np.array([1]),'dan':np.array([3])}
    kwargs={'eta':.001,'circuit':circuit,
        'modulation_mask':np.array([0,0,0,1,0],dtype=np.uint8),
        'kc_rest':-52.,'adaptation_jump':8.,'adaptation_tau':200.}
    return MemoryBrain(path,backend='cpu',**kwargs),MemoryBrain(path,backend='metal',**kwargs)


TRACE=[(([0],20.),10.),(None,10.),(([3],20.),10.),(([0,3],18.),20.)]


def run_trace(model):
    counts=[]
    for stimulation,duration in TRACE:
        value,_=model.step([],duration,stimulation=stimulation,lamina_bias=0)
        counts.append(value.copy())
    return np.asarray(counts)


def test_metal_matches_cpu_micrograph_events_and_state(tmp_path):
    cpu,metal=paired_brains(tmp_path)
    expected=run_trace(cpu);actual=run_trace(metal)
    np.testing.assert_array_equal(actual,expected)
    for name in ['v','g','drive','previous_drive','modulation','adaptation']:
        np.testing.assert_allclose(getattr(metal,name),getattr(cpu,name),rtol=1e-5,atol=.002,err_msg=name)
    for name in ['refractory','active_flag','last','modulation_last']:
        np.testing.assert_array_equal(getattr(metal,name),getattr(cpu,name),err_msg=name)
    assert metal.cursor==cpu.cursor==500


def test_metal_repeated_runs_are_bitwise_identical(tmp_path):
    _,first=paired_brains(tmp_path/'first')
    _,second=paired_brains(tmp_path/'second')
    np.testing.assert_array_equal(run_trace(first),run_trace(second))
    first.backend.sync_for_checkpoint();second.backend.sync_for_checkpoint()
    for name in ['weight',*first.fields]:
        np.testing.assert_array_equal(getattr(first,name),getattr(second,name),err_msg=name)


def test_diagnostic_capture_records_exact_all_neuron_spike_events(tmp_path):
    cpu,metal=paired_brains(tmp_path)
    cpu.backend.start_diagnostics();metal.backend.start_diagnostics()
    np.testing.assert_array_equal(run_trace(metal),run_trace(cpu))
    assert cpu.backend.spike_events
    assert metal.backend.spike_events==cpu.backend.spike_events
    assert any(neuron not in cpu.circuit['kc'] for neuron,_ in cpu.backend.spike_events)
