from pathlib import Path

import numpy as np


def toy_brain(tmp_path, *, graph_name='graph.npz'):
    from doom_learning_v6.brain import MemoryBrain
    path=Path(tmp_path)/graph_name
    np.savez(path,
        ptr=np.array([0,3,4,5,6,7,7],dtype=np.int64),
        post=np.array([0,1,1,2,3,4,5],dtype=np.int32),
        weight=np.array([4.,7.,8.,10.,.275,-2.,11.],dtype=np.float32),
        ids=np.arange(6,dtype=np.int64),
        retina=np.array([0],dtype=np.int32),
        uv=np.array([[.5,.5]],dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.array(['test']*6))
    circuit={'edges':np.array([3,6],dtype=np.int64),
        'pre':np.array([1,4],dtype=np.int32),
        'kc_mask':np.array([0,1,0,0,1,0],dtype=np.uint8),
        'dan_index':np.array([-1,-1,0,-1,-1,-1],dtype=np.int8),
        'gain':np.array([[1.,.5]],dtype=np.float32),
        'kc':np.array([1,4],dtype=np.int32),
        'mb':np.array([2,5],dtype=np.int32),
        'dan':np.array([2],dtype=np.int32)}
    return MemoryBrain(path,eta=.001,circuit=circuit,
        modulation_mask=np.array([0,0,1,0,0,0],dtype=np.uint8))


def wide_quiet_brain(tmp_path,n=200_000):
    from doom_learning_v6.brain import MemoryBrain
    path=Path(tmp_path)/'wide.npz'
    np.savez(path,ptr=np.zeros(n+1,dtype=np.int64),post=np.empty(0,dtype=np.int32),
        weight=np.empty(0,dtype=np.float32),ids=np.arange(n,dtype=np.int64),
        retina=np.empty(0,dtype=np.int32),uv=np.empty((0,2),dtype=np.float32),
        lamina=np.empty(0,dtype=np.int32),sugar=np.empty(0,dtype=np.int32),
        superclass=np.full(n,'test'))
    circuit={'edges':np.empty(0,dtype=np.int64),'pre':np.empty(0,dtype=np.int32),
        'kc_mask':np.zeros(n,dtype=np.uint8),'dan_index':np.full(n,-1,dtype=np.int8),
        'gain':np.empty((0,0),dtype=np.float32),'kc':np.empty(0,dtype=np.int32),
        'mb':np.empty(0,dtype=np.int32),'dan':np.empty(0,dtype=np.int32)}
    brain=MemoryBrain(path,circuit=circuit,modulation_mask=np.zeros(n,dtype=np.uint8))
    brain.active[:]=np.arange(n,dtype=np.int32);brain.active_flag.fill(1);brain.nactive[0]=n
    brain.drive.fill(7.1);brain.previous_drive.fill(7.1)
    return brain
