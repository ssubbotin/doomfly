import ctypes as C
import sys

import numpy as np
import pytest

from test_doom_learning_v6 import brain
from doom_learning_v6.metal.backend import Graph,KCEvent,State,Timing
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


def test_metal_narrow_boundary_transfers_drive_counts_and_cursor():
    library=metal_library();graph,graph_keepalive=raw_empty_graph();handle=C.c_void_p()
    assert library.df_metal_create(C.byref(graph),
        str(DEFAULT_OUTPUT/'kernels.metallib').encode(),C.byref(handle))==0
    try:
        state,state_keepalive=raw_empty_state()
        assert library.df_metal_upload_state(handle,C.byref(state))==0
        upload=library.df_metal_upload_drive
        upload.argtypes=[C.c_void_p,C.c_void_p];upload.restype=C.c_int
        download=library.df_metal_download_observation
        download.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_int64)]
        download.restype=C.c_int
        advance=library.df_metal_advance
        advance.argtypes=[C.c_void_p,C.c_int32,C.POINTER(KCEvent),C.c_int32,
            C.POINTER(C.c_int32),C.POINTER(Timing)]
        advance.restype=C.c_int
        drive=np.array([20.],dtype=np.float32)
        assert upload(handle,C.c_void_p(drive.ctypes.data))==0
        events=(KCEvent*1)();event_count=C.c_int32();timing=Timing()
        assert advance(handle,100,events,1,C.byref(event_count),C.byref(timing))==0
        counts=np.full(1,-1,dtype=np.int32);cursor=C.c_int64(-1)
        assert download(handle,C.c_void_p(counts.ctypes.data),C.byref(cursor))==0
        assert counts[0]>0
        assert cursor.value==100
    finally:
        library.df_metal_destroy(handle)


def test_metal_narrow_boundary_rejects_null_pointers():
    library=metal_library();graph,graph_keepalive=raw_empty_graph();handle=C.c_void_p()
    assert library.df_metal_create(C.byref(graph),
        str(DEFAULT_OUTPUT/'kernels.metallib').encode(),C.byref(handle))==0
    try:
        upload=library.df_metal_upload_drive
        upload.argtypes=[C.c_void_p,C.c_void_p];upload.restype=C.c_int
        assert upload(handle,None)!=0
        assert library.df_metal_last_error().decode()=='Metal drive pointer is null'
        download=library.df_metal_download_observation
        download.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_int64)]
        download.restype=C.c_int
        cursor=C.c_int64()
        assert download(handle,None,C.byref(cursor))!=0
        assert library.df_metal_last_error().decode()=='Metal observation pointer is null'
    finally:
        library.df_metal_destroy(handle)


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


def test_metal_reports_separate_native_command_phases(tmp_path):
    model=brain(tmp_path,backend='metal')
    model.weights_frozen=True
    model.step([],10,stimulation=([0],20),lamina_bias=0)
    timing=model.backend.last_timing
    expected={
        'native_total_seconds','gpu_seconds','counts_clear_seconds',
        'encode_seconds','commit_call_seconds','wait_call_seconds',
        'native_event_copy_seconds','native_event_copy_bytes',
        'full_upload_seconds','full_upload_bytes','materialize_seconds',
        'materialize_bytes','drive_copy_seconds','drive_copy_bytes',
        'counts_copy_seconds','counts_copy_bytes','event_conversion_sort_seconds',
        'eligibility_seconds'}
    assert expected<=timing.keys()
    assert all(timing[name]>=0 for name in expected)
    assert timing['native_total_seconds']>=timing['encode_seconds']
    assert timing['native_event_copy_bytes']==len(model.backend.last_kc_events)*C.sizeof(KCEvent)
    assert timing['full_upload_bytes']==timing['materialize_bytes']==0
    assert timing['drive_copy_bytes']==model.drive.nbytes
    assert timing['counts_copy_bytes']==model.counts.nbytes


def test_metal_restore_rejects_duplicate_delayed_neuron(tmp_path):
    model=brain(tmp_path,backend='metal')
    model.queue_count[0]=2
    model.queue[0,:2]=[1,1]
    with pytest.raises(RuntimeError,match='duplicate delayed neuron'):
        model.backend.ensure_initialized()
