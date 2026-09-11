"""ctypes owner for the direct Objective-C++ Metal backend."""
import ctypes as C
import hashlib
from pathlib import Path
import time

import numpy as np

from doom_learning.common import OUT
from ..backend import BackendError
from .build import ABI_VERSION,DEFAULT_OUTPUT,library_path,probe
from .graph import load_or_build_incoming


class Graph(C.Structure):
    _fields_=[('neurons',C.c_int32),('edges',C.c_int64),('delay_slots',C.c_int32),
        ('dt_ms',C.c_float),('adaptation_jump_mv',C.c_float),('adaptation_tau_ms',C.c_float),
        ('out_ptr',C.c_void_p),('out_post',C.c_void_p),('in_ptr',C.c_void_p),
        ('in_pre',C.c_void_p),('in_edge',C.c_void_p),('kc_mask',C.c_void_p),
        ('modulation_mask',C.c_void_p)]


class State(C.Structure):
    _fields_=[('cursor',C.c_int64),('weight',C.c_void_p),('v',C.c_void_p),
        ('g',C.c_void_p),('refractory',C.c_void_p),('drive',C.c_void_p),
        ('previous_drive',C.c_void_p),('queue',C.c_void_p),('queue_count',C.c_void_p),
        ('counts',C.c_void_p),('active',C.c_void_p),('active_flag',C.c_void_p),
        ('nactive',C.c_void_p),('last',C.c_void_p),('modulation',C.c_void_p),
        ('modulation_last',C.c_void_p),('rest',C.c_void_p),('adaptation',C.c_void_p)]


class KCEvent(C.Structure):
    _fields_=[('tick',C.c_int64),('neuron',C.c_int32),('reserved',C.c_int32)]


class Timing(C.Structure):
    _fields_=[('host_seconds',C.c_double),('gpu_seconds',C.c_double)]


def _pointer(array):return C.c_void_p(array.ctypes.data)


class MetalBackend:
    name='metal'

    def __init__(self,brain):
        self.brain=brain;self.handle=C.c_void_p();self.library=None
        self.incoming=None;self._metadata=None;self.poisoned=False
        self.last_kc_events=[];self.last_timing={};self.capture_spikes=False;self.spike_events=[]

    def _state(self):
        b=self.brain
        names=['weight','v','g','refractory','drive','previous_drive','queue','queue_count',
            'counts','active','active_flag','nactive','last','modulation','modulation_last','rest','adaptation']
        return State(b.cursor,*[_pointer(getattr(b,name)) for name in names])

    def _error(self,status):
        if status:
            message=self.library.df_metal_last_error()
            raise BackendError(message.decode() if message else f'Metal backend error {status}')

    def ensure_initialized(self):
        if self.handle.value:return
        self._metadata=probe(DEFAULT_OUTPUT)
        signature=hashlib.sha256(self.brain.ptr.tobytes()+self.brain.post.tobytes()).hexdigest()
        self.incoming=load_or_build_incoming(self.brain.ptr,self.brain.post,OUT/'metal'/'graphs'/signature)
        self.library=C.CDLL(str(library_path(DEFAULT_OUTPUT)))
        self.library.df_metal_last_error.restype=C.c_char_p
        self.library.df_metal_create.argtypes=[C.POINTER(Graph),C.c_char_p,C.POINTER(C.c_void_p)]
        self.library.df_metal_create.restype=C.c_int
        self.library.df_metal_upload_state.argtypes=[C.c_void_p,C.POINTER(State)]
        self.library.df_metal_upload_state.restype=C.c_int
        self.library.df_metal_download_state.argtypes=[C.c_void_p,C.POINTER(State)]
        self.library.df_metal_download_state.restype=C.c_int
        self.library.df_metal_update_weights.argtypes=[C.c_void_p,C.c_int32,C.c_void_p,C.c_void_p]
        self.library.df_metal_update_weights.restype=C.c_int
        self.library.df_metal_advance.argtypes=[C.c_void_p,C.c_int32,C.POINTER(KCEvent),
            C.c_int32,C.POINTER(C.c_int32),C.POINTER(Timing)]
        self.library.df_metal_advance.restype=C.c_int
        self.library.df_metal_apply_eligibility.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_double]
        self.library.df_metal_apply_eligibility.restype=C.c_int
        self.library.df_metal_set_diagnostics.argtypes=[C.c_void_p,C.c_int32]
        self.library.df_metal_set_diagnostics.restype=C.c_int
        self.library.df_metal_destroy.argtypes=[C.c_void_p]
        b=self.brain;i=self.incoming
        graph=Graph(b.n,len(b.post),b.queue.shape[0],b.dt,b.adaptation_jump,b.adaptation_tau,
            _pointer(b.ptr),_pointer(b.post),_pointer(i.ptr),_pointer(i.pre),_pointer(i.edge),
            _pointer(b.circuit['kc_mask']),_pointer(b.modulation_mask))
        metallib=Path(DEFAULT_OUTPUT)/'kernels.metallib'
        self._error(self.library.df_metal_create(C.byref(graph),str(metallib).encode(),C.byref(self.handle)))
        try:self._error(self.library.df_metal_upload_state(self.handle,C.byref(self._state())))
        except Exception:
            self.close();raise

    def advance(self,steps):
        self.ensure_initialized()
        if self.poisoned:raise BackendError('Metal backend is poisoned')
        self.restore_from_host()
        cells=self.brain.n if self.capture_spikes else int(np.count_nonzero(self.brain.circuit['kc_mask']))
        capacity=max(1,cells*(1+(steps-1)//22))
        events=(KCEvent*capacity)();count=C.c_int32();timing=Timing();started=time.perf_counter()
        status=self.library.df_metal_advance(self.handle,steps,events,capacity,C.byref(count),C.byref(timing))
        if status:
            self.poisoned=True;self._error(status)
        tau_ms=self.brain.rule_parameters['trace_kc_seconds']*1000
        status=self.library.df_metal_apply_eligibility(self.handle,_pointer(self.brain.eligibility),
            _pointer(self.brain.eligibility_last),tau_ms)
        if status:
            self.poisoned=True;self._error(status)
        self.sync_for_checkpoint()
        elapsed=time.perf_counter()-started
        recorded=[(int(events[i].tick),int(events[i].neuron)) for i in range(count.value)]
        self.last_kc_events=sorted((tick,neuron) for tick,neuron in recorded
            if self.brain.circuit['kc_mask'][neuron])
        if self.capture_spikes:self.spike_events.extend(sorted((neuron,tick) for tick,neuron in recorded))
        self.last_timing={'host_seconds':timing.host_seconds,'gpu_seconds':timing.gpu_seconds,
            'backend_seconds':elapsed}
        return elapsed

    def start_diagnostics(self):
        self.ensure_initialized();self._error(self.library.df_metal_set_diagnostics(self.handle,1))
        self.capture_spikes=True;self.spike_events=[]

    def stop_diagnostics(self):
        if self.handle.value:self._error(self.library.df_metal_set_diagnostics(self.handle,0))
        self.capture_spikes=False

    def sync_for_checkpoint(self):
        if not self.handle.value:return
        state=self._state();self._error(self.library.df_metal_download_state(self.handle,C.byref(state)))
        self.brain.cursor=int(state.cursor)

    def restore_from_host(self):
        if not self.handle.value:return
        self._error(self.library.df_metal_upload_state(self.handle,C.byref(self._state())))

    def update_weights(self,edge_ids,values):
        self.ensure_initialized();edge_ids=np.ascontiguousarray(edge_ids,dtype=np.int64)
        values=np.ascontiguousarray(values,dtype=np.float32)
        if edge_ids.shape!=values.shape:raise ValueError('Plastic edge/value shape mismatch')
        self._error(self.library.df_metal_update_weights(self.handle,len(edge_ids),_pointer(edge_ids),_pointer(values)))

    def metadata(self):
        if self._metadata is None:self.ensure_initialized()
        return {'name':self.name,'abi_version':ABI_VERSION,**self._metadata}

    def close(self):
        if self.library is not None and self.handle.value:
            self.library.df_metal_destroy(self.handle);self.handle=C.c_void_p()

    def __del__(self):
        try:self.close()
        except Exception:return
