"""Synthetic native incoming-carry controls, separate from biological validation."""
import sys

import numpy as np
import pytest

from doom_learning_v6.brain import MemoryBrain


pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def carry_pair(tmp_path,pre,weights,mask=(0,0,0)):
    n=3;pre=np.asarray(pre,dtype=np.int32);path=tmp_path/'carry-graph.npz'
    np.savez(path,
        ptr=np.r_[0,np.cumsum(np.bincount(pre,minlength=n))].astype(np.int64),
        post=np.full(len(pre),2,dtype=np.int32),
        weight=np.asarray(weights,dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['synthetic-test']*n))
    circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
    brains=[]
    try:
        for name in ['cpu','metal']:
            brains.append(MemoryBrain(path,backend=name,eta=0.,circuit=circuit,
                modulation_mask=np.asarray(mask,dtype=np.uint8)))
        return brains
    except BaseException:
        # Close any constructed native backend if its pair cannot be created.
        for brain in brains:
            brain.backend.close()
        raise


@pytest.mark.parametrize('pre,weights,initial_g,initial_refractory,mask',[
    pytest.param((0,1),(16777216,-16777216),1,0,(0,0,0),id='catastrophic-carry'),
    pytest.param((0,0),(16777216,-16777216),1,0,(0,0,0),id='duplicate-edges'),
    pytest.param((0,1),(2,-.5),1,0,(0,0,0),id='ordinary-carry'),
    pytest.param((0,1),(16777216,-16777216),0,0,(0,0,0),id='zero-carry'),
    pytest.param((0,1),(2,-.5),1,3,(0,0,0),id='refractory-rejection'),
    pytest.param((0,1),(.275,.275),1,0,(1,1,0),id='modulatory-only-one-tick'),
    pytest.param((),(),1,0,(0,0,0),id='empty-topology'),
])
def test_incoming_carry_matches_native_cpu_state_bits(
        tmp_path,pre,weights,initial_g,initial_refractory,mask):
    # Summing arrivals from zero before adding settled g breaks catastrophic
    # carry. The other fixtures protect duplicate retention, zero carry,
    # refractory rejection and unchanged modulation/settlement behavior.
    cpu,metal=carry_pair(tmp_path,pre,weights,mask)
    try:
        queued=np.unique(np.asarray(pre,dtype=np.int32))
        for brain in [cpu,metal]:
            brain.v[:]=[-51.,-52.,-51.];brain.g[:]=[0.,0.,initial_g]
            brain.refractory[2]=initial_refractory
            brain.active_flag.fill(0);brain.nactive.fill(0)
            brain.queue[0,:len(queued)]=queued;brain.queue_count[0]=len(queued)
            brain.backend.advance(1);brain.backend.materialize('incoming-carry-test')
        assert cpu.cursor==metal.cursor==1
        if pre==(0,1) and weights==(16777216,-16777216) and initial_g==1:
            # Hand-derived float32 result: (1 + 2**24) - 2**24 == 0.
            assert cpu.g[2]==0
            assert not cpu.counts.any()
            assert not metal.counts.any()
        np.testing.assert_array_equal(cpu.counts,metal.counts)
        # Report conductance first to isolate carry before voltage consequences.
        for name in ['g','v','adaptation','modulation']:
            np.testing.assert_array_equal(getattr(cpu,name).view(np.uint32),
                getattr(metal,name).view(np.uint32),err_msg=name)
        for name in ['refractory','last','modulation_last','active_flag']:
            np.testing.assert_array_equal(getattr(cpu,name),getattr(metal,name),err_msg=name)
    finally:
        try:
            cpu.backend.close()
        finally:
            metal.backend.close()
