import ctypes as C
import sys

import numpy as np
import pytest

from test_doom_learning_v6 import brain
from doom_learning_v6.metal.backend import Graph,State
from doom_learning_v6.metal.build import DEFAULT_OUTPUT,library_path,probe


pytestmark=pytest.mark.skipif(sys.platform!='darwin',reason='Metal requires macOS')


def raw_empty_graph(delay_slots=19):
    ptr=np.zeros(2,dtype=np.int64)
    mask=np.zeros(1,dtype=np.uint8)
    graph=Graph(1,0,delay_slots,.1,8.,200.,C.c_void_p(ptr.ctypes.data),C.c_void_p(),
        C.c_void_p(ptr.ctypes.data),C.c_void_p(),C.c_void_p(),C.c_void_p(mask.ctypes.data),
        C.c_void_p(mask.ctypes.data))
    return graph,(ptr,mask)


def raw_empty_state():
    arrays=[
        np.full(1,-52.,dtype=np.float32),np.zeros(1,dtype=np.float32),
        np.zeros(1,dtype=np.int16),np.zeros(1,dtype=np.float32),
        np.zeros(1,dtype=np.float32),np.zeros((19,1),dtype=np.int32),
        np.zeros(19,dtype=np.int32),np.zeros(1,dtype=np.int32),
        np.zeros(1,dtype=np.int32),np.zeros(1,dtype=np.uint8),
        np.zeros(1,dtype=np.int32),np.full(1,-1,dtype=np.int64),
        np.zeros(1,dtype=np.float32),np.zeros(1,dtype=np.int64),
        np.full(1,-52.,dtype=np.float32),np.zeros(1,dtype=np.float32)]
    return State(0,C.c_void_p(),*[C.c_void_p(array.ctypes.data) for array in arrays]),arrays


def metal_library():
    probe(DEFAULT_OUTPUT);library=C.CDLL(str(library_path(DEFAULT_OUTPUT)))
    library.df_metal_create.argtypes=[C.POINTER(Graph),C.c_char_p,C.POINTER(C.c_void_p)]
    library.df_metal_create.restype=C.c_int
    library.df_metal_last_error.restype=C.c_char_p
    library.df_metal_upload_state.argtypes=[C.c_void_p,C.POINTER(State)]
    library.df_metal_upload_state.restype=C.c_int
    library.df_metal_download_state.argtypes=[C.c_void_p,C.POINTER(State)]
    library.df_metal_download_state.restype=C.c_int
    library.df_metal_destroy.argtypes=[C.c_void_p]
    return library


def test_metal_abi_accepts_null_arrays_for_zero_edge_graph():
    library=metal_library();graph,graph_keepalive=raw_empty_graph();handle=C.c_void_p()
    status=library.df_metal_create(C.byref(graph),
        str(DEFAULT_OUTPUT/'kernels.metallib').encode(),C.byref(handle))
    try:
        assert status==0,library.df_metal_last_error().decode()
        state,state_keepalive=raw_empty_state()
        assert library.df_metal_upload_state(handle,C.byref(state))==0
        assert library.df_metal_download_state(handle,C.byref(state))==0
    finally:
        if handle.value:library.df_metal_destroy(handle)


def test_metal_abi_rejects_noncanonical_delay_ring():
    library=metal_library();graph,keepalive=raw_empty_graph(delay_slots=18);handle=C.c_void_p()
    status=library.df_metal_create(C.byref(graph),
        str(DEFAULT_OUTPUT/'kernels.metallib').encode(),C.byref(handle))
    try:
        assert status!=0
        assert library.df_metal_last_error().decode()=='Metal delay slot count must be 19'
    finally:
        if handle.value:library.df_metal_destroy(handle)


def test_metal_state_round_trip_preserves_all_neural_arrays(tmp_path):
    model=brain(tmp_path,backend='metal')
    before={name:getattr(model,name).copy() for name in model.fields}
    before['weight']=model.weight.copy()
    model.backend.ensure_initialized()
    model.backend.sync_for_checkpoint()
    for name,expected in before.items():
        np.testing.assert_array_equal(getattr(model,name),expected,err_msg=name)
    metadata=model.backend.metadata()
    assert metadata['name']=='metal'
    assert metadata['device']['name']=='Apple M4 Pro'


def test_metal_restore_rejects_duplicate_delayed_neuron(tmp_path):
    model=brain(tmp_path,backend='metal')
    model.queue_count[0]=2
    model.queue[0,:2]=[1,1]
    with pytest.raises(RuntimeError,match='duplicate delayed neuron'):
        model.backend.ensure_initialized()
