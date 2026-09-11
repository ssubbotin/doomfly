import hashlib
import sys

import numpy as np
import pytest

from doom_learning_v6.brain import MemoryBrain
from doom_learning_v6.metal.validate import _state_digest


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


def cross_word_brains(tmp_path):
    n=40
    pre=np.arange(35,dtype=np.int32)
    post=np.full(len(pre),n-1,dtype=np.int32)
    weight=np.ones(len(pre),dtype=np.float32)
    weight[31]=-0.75;weight[32]=1.25;weight[34]=-0.5
    path=tmp_path/'cross-word-graph.npz'
    np.savez(path,ptr=np.r_[0,np.cumsum(np.bincount(pre,minlength=n))].astype(np.int64),
        post=post,weight=weight,ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['test']*n))
    circuit={'edges':np.array([0],dtype=np.int64),'pre':np.array([0],dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,1),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.array([n-1]),'dan':np.empty(0,dtype=np.int32)}
    kwargs={'eta':0.,'circuit':circuit,'modulation_mask':np.zeros(n,dtype=np.uint8)}
    cpu=MemoryBrain(path,backend='cpu',**kwargs)
    metal=MemoryBrain(path,backend='metal',**kwargs)
    delayed=np.array([0,31,32,34],dtype=np.int32)
    for brain in [cpu,metal]:
        brain.queue[0,:len(delayed)]=delayed
        brain.queue_count[0]=len(delayed)
    return cpu,metal


TRACE=[(([0],20.),10.),(None,10.),(([3],20.),10.),(([0,3],18.),20.)]


def run_trace(model):
    counts=[]
    for stimulation,duration in TRACE:
        value,_=model.step([],duration,stimulation=stimulation,lamina_bias=0)
        counts.append(value.copy())
    return np.asarray(counts)


def materialize(model):
    model.backend.materialize('test')


def test_metal_matches_cpu_micrograph_events_and_state(tmp_path):
    cpu,metal=paired_brains(tmp_path)
    expected=run_trace(cpu);actual=run_trace(metal)
    np.testing.assert_array_equal(actual,expected)
    materialize(metal)
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


def test_metal_micrograph_digest_is_retained(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.backend.start_diagnostics();counts=run_trace(metal);metal.backend.stop_diagnostics()
    events=np.asarray(metal.backend.spike_events,dtype=np.int64)
    assert hashlib.sha256(counts.tobytes()).hexdigest()==(
        'de7d06da6e31fe80c35eb54c19926e81db5c514483fa170ca4eccf878d0fec90')
    assert _state_digest(metal)=='0fdec182c34b38bc8aa9289d9eab39ace678df5f38d598a8f043d83bc29674e9'
    assert hashlib.sha256(events.tobytes()).hexdigest()==(
        '00ff37a29ca8cc916a39436e1ac4f4aac1bef46291502d751bb4860e21cd6ad1')
    assert len(events)==3


def test_metal_resident_state_ignores_stale_host_arrays(tmp_path):
    _,reference=paired_brains(tmp_path/'reference')
    _,resident=paired_brains(tmp_path/'resident')
    reference.weights_frozen=resident.weights_frozen=True
    for model in [reference,resident]:
        model.step([],10,stimulation=([0],20),lamina_bias=0)
    resident.v.fill(-47);resident.g.fill(123);resident.refractory.fill(42)
    resident.previous_drive.fill(-100);resident.queue.fill(0);resident.queue_count.fill(0)
    resident.active.fill(0);resident.active_flag.fill(0);resident.nactive.fill(0)
    resident.last.fill(-999);resident.modulation.fill(77);resident.modulation_last.fill(-999)
    resident.adaptation.fill(55)
    expected,_=reference.step([],10,stimulation=([3],20),lamina_bias=0)
    actual,_=resident.step([],10,stimulation=([3],20),lamina_bias=0)
    np.testing.assert_array_equal(actual,expected)
    materialize(reference);materialize(resident)
    for name in ['v','g','refractory','previous_drive','queue','queue_count','active',
            'active_flag','nactive','last','modulation','modulation_last','adaptation']:
        np.testing.assert_array_equal(getattr(resident,name),getattr(reference,name),err_msg=name)


def test_metal_rejects_full_upload_of_stale_host_state(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.weights_frozen=True
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    with pytest.raises(RuntimeError,match='stale'):
        metal.backend.restore_from_host()


def test_metal_materialize_restores_device_state_to_host(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.weights_frozen=True
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    metal.v.fill(np.nan);metal.queue.fill(-1);metal.active_flag.fill(255)
    materialize(metal)
    assert np.isfinite(metal.v).all()
    assert np.all(metal.queue>=0)
    assert np.all(metal.active_flag<=1)


def test_metal_resident_bin_uses_only_narrow_boundary_transfers(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.weights_frozen=True
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    timing=metal.backend.last_timing
    assert timing['full_upload_bytes']==0
    assert timing['materialize_bytes']==0
    assert timing['drive_copy_bytes']==metal.drive.nbytes
    assert timing['counts_copy_bytes']==metal.counts.nbytes


def test_diagnostic_capture_records_exact_all_neuron_spike_events(tmp_path):
    cpu,metal=paired_brains(tmp_path)
    cpu.backend.start_diagnostics();metal.backend.start_diagnostics()
    np.testing.assert_array_equal(run_trace(metal),run_trace(cpu))
    assert cpu.backend.spike_events
    assert metal.backend.spike_events==cpu.backend.spike_events
    assert any(neuron not in cpu.circuit['kc'] for neuron,_ in cpu.backend.spike_events)
    cpu.backend.stop_diagnostics();metal.backend.stop_diagnostics()
    assert metal.backend._last_materialization_reason=='diagnostics'


def test_metal_advance_uses_one_compute_encoder(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    assert metal.backend.last_timing['encoder_count']==1


def test_metal_uses_two_dispatches_per_tick(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    timing=metal.backend.last_timing
    assert timing['dispatch_count']==202
    assert timing['indirect_dispatch_count']==0


def test_metal_uses_808_dispatches_per_40ms_trace(tmp_path):
    _,metal=paired_brains(tmp_path)
    dispatches=0
    for _ in range(4):
        metal.step([],10,stimulation=([0,3],20),lamina_bias=0)
        dispatches+=metal.backend.last_timing['dispatch_count']
    assert dispatches==808


def test_metal_reports_mark_and_gather_grid_sizes(tmp_path):
    _,metal=paired_brains(tmp_path)
    metal.step([],10,stimulation=([0],20),lamina_bias=0)
    timing=metal.backend.last_timing
    assert timing['mark_grid_threads']==100*metal.n
    assert timing['gather_grid_threads']==100*metal.n


def test_sparse_edge_bitmap_preserves_cross_word_delivery(tmp_path):
    cpu,metal=cross_word_brains(tmp_path)
    cpu.backend.advance(1);metal.backend.advance(1)
    np.testing.assert_array_equal(metal.counts,cpu.counts)
    materialize(metal)
    for name in ['v','g','modulation','adaptation']:
        np.testing.assert_array_equal(getattr(metal,name),getattr(cpu,name),err_msg=name)
    assert metal.g[-1]==pytest.approx(1.0)
    timing=metal.backend.last_timing
    assert timing['edge_bitmap_words']==2
    assert timing['indirect_dispatch_count']==0
    assert timing['dispatch_count']==4


def test_sparse_edge_bitmap_clears_words_between_ticks(tmp_path):
    cpu,metal=cross_word_brains(tmp_path)
    for brain in [cpu,metal]:
        brain.queue[1,0]=33
        brain.queue_count[1]=1
    cpu.backend.advance(2);metal.backend.advance(2)
    materialize(metal)
    for name in ['v','g','modulation','adaptation']:
        np.testing.assert_array_equal(getattr(metal,name),getattr(cpu,name),err_msg=name)
    assert metal.backend.last_timing['indirect_dispatch_count']==0


def test_sparse_edge_bitmap_supports_edgeless_graph(tmp_path):
    n=2;path=tmp_path/'edgeless-graph.npz'
    np.savez(path,ptr=np.zeros(n+1,dtype=np.int64),post=np.empty(0,dtype=np.int32),
        weight=np.empty(0,dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['test']*n))
    circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
    kwargs={'eta':0.,'circuit':circuit,'modulation_mask':np.zeros(n,dtype=np.uint8)}
    cpu=MemoryBrain(path,backend='cpu',**kwargs);metal=MemoryBrain(path,backend='metal',**kwargs)
    for brain in [cpu,metal]:
        brain.queue[0,0]=0;brain.queue_count[0]=1
    cpu.backend.advance(2);metal.backend.advance(2)
    materialize(metal)
    for name in ['v','g','refractory','queue_count','last']:
        np.testing.assert_array_equal(getattr(metal,name),getattr(cpu,name),err_msg=name)
    assert metal.backend.last_timing['edge_bitmap_words']==0


def test_sparse_edge_bitmap_masks_targets_sharing_a_word(tmp_path):
    n=42;pre=np.arange(35,dtype=np.int32)
    post=np.r_[np.full(10,40),np.full(25,41)].astype(np.int32)
    weight=np.ones(len(pre),dtype=np.float32)
    weight[0]=2.;weight[1]=-.5;weight[10]=3.;weight[11]=-1.
    path=tmp_path/'shared-word-graph.npz'
    np.savez(path,ptr=np.r_[0,np.cumsum(np.bincount(pre,minlength=n))].astype(np.int64),
        post=post,weight=weight,ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['test']*n))
    circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.array([40,41]),'dan':np.empty(0,dtype=np.int32)}
    kwargs={'eta':0.,'circuit':circuit,'modulation_mask':np.zeros(n,dtype=np.uint8)}
    cpu=MemoryBrain(path,backend='cpu',**kwargs);metal=MemoryBrain(path,backend='metal',**kwargs)
    for brain in [cpu,metal]:
        brain.queue[0,:4]=[0,1,10,11];brain.queue_count[0]=4
    cpu.backend.advance(1);metal.backend.advance(1)
    materialize(metal)
    np.testing.assert_array_equal(metal.g,cpu.g)
    np.testing.assert_array_equal(metal.g[-2:],[1.5,2.])
