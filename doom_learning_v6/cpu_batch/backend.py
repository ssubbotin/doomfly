"""Python-owned shared graph and compact mutable CPU trajectory state."""
import ctypes as C
import hashlib
import json
import threading

import numpy as np

from doom_learning.common import digest
from doom_learning_v6.backend import BackendError
from doom_learning_v6.rule import PARAMETERS

from .build import DEFAULT_OUTPUT,build,library_path


class _NativeGraph(C.Structure):
    _fields_=[('neurons',C.c_int32),('edges',C.c_int64),
        ('delay_slots',C.c_int32),('plastic_edges',C.c_int32),
        ('dt_ms',C.c_float),('eligibility_tau_ms',C.c_float),
        ('adaptation_jump_mv',C.c_float),('adaptation_tau_ms',C.c_float),
        ('out_ptr',C.c_void_p),('out_post',C.c_void_p),
        ('base_weight',C.c_void_p),('plastic_slot',C.c_void_p),
        ('kc_mask',C.c_void_p),('modulation_mask',C.c_void_p),('rest',C.c_void_p)]


class _NativeLane(C.Structure):
    _fields_=[('cursor',C.c_void_p),('plastic_weight',C.c_void_p),
        ('v',C.c_void_p),('g',C.c_void_p),('refractory',C.c_void_p),
        ('drive',C.c_void_p),('previous_drive',C.c_void_p),
        ('queue',C.c_void_p),('queue_count',C.c_void_p),('counts',C.c_void_p),
        ('active',C.c_void_p),('active_flag',C.c_void_p),('nactive',C.c_void_p),
        ('last',C.c_void_p),('eligibility',C.c_void_p),
        ('eligibility_last',C.c_void_p),('modulation',C.c_void_p),
        ('modulation_last',C.c_void_p),('adaptation',C.c_void_p)]


class _NativeTiming(C.Structure):
    _fields_=[('native_wall_seconds',C.c_double),
        ('lanes_advanced',C.c_int32),('workers',C.c_int32),('steps',C.c_int32),
        ('generation',C.c_uint64),('pool_threads',C.c_int32)]


def _pointer(array):return C.c_void_p(array.ctypes.data)


def _array(name,value,dtype,shape):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype(dtype):
        raise ValueError(f'Invalid {name} dtype')
    if value.shape!=shape:raise ValueError(f'Invalid {name} shape')
    if not value.flags.c_contiguous:raise ValueError(f'{name} must be C-contiguous')
    return value


def _finite(name,value):
    if not np.isfinite(value).all():raise ValueError(f'{name} must be finite')


def _configuration_identity(brain):
    signature=json.dumps(brain.configuration_signature(),sort_keys=True,separators=(',',':'))
    return {'ptr_sha256':digest(brain.ptr),'post_sha256':digest(brain.post),
        'initial_weight_sha256':brain.initial_weight_sha256,
        'configuration_sha256':hashlib.sha256(signature.encode()).hexdigest()}


class SharedCpuGraph:
    """Read-only v6 graph storage retained by every registered lane."""

    def __init__(self,*,neurons,edges,delay_slots,plastic_edges,dt_ms,
            eligibility_tau_ms,adaptation_jump_mv,adaptation_tau_ms,arrays,identity):
        self.neurons=neurons;self.edges=edges;self.delay_slots=delay_slots
        self.plastic_edges=plastic_edges;self.dt_ms=dt_ms
        self.eligibility_tau_ms=eligibility_tau_ms
        self.adaptation_jump_mv=adaptation_jump_mv
        self.adaptation_tau_ms=adaptation_tau_ms
        self.arrays=arrays;self.identity=identity
        for name,value in arrays.items():setattr(self,name,value)

    @classmethod
    def from_brain(cls,brain):
        n=int(brain.n);edges=len(brain.post)
        ptr=_array('ptr',brain.ptr,np.int64,(n+1,))
        post=_array('post',brain.post,np.int32,(edges,))
        weight=_array('weight',brain.weight,np.float32,(edges,))
        if ptr[0]!=0 or ptr[-1]!=edges or np.any(np.diff(ptr)<0):
            raise ValueError('Invalid CSR graph')
        if np.any(post<0) or np.any(post>=n):raise ValueError('Graph index out of range')
        _finite('weight',weight)
        circuit=brain.circuit
        plastic_edge=_array('plastic edges',circuit['edges'],np.int64,
            (len(circuit['edges']),))
        count=len(plastic_edge)
        if count>32767:raise ValueError('At most 32,767 plastic edges are supported')
        if len(np.unique(plastic_edge))!=count:raise ValueError('Plastic edges must be unique')
        if np.any(plastic_edge<0) or np.any(plastic_edge>=edges):
            raise ValueError('Plastic edge index out of range')
        kc_mask=_array('kc_mask',circuit['kc_mask'],np.uint8,(n,))
        modulation_mask=_array('modulation_mask',brain.modulation_mask,np.uint8,(n,))
        rest=_array('rest',brain.rest,np.float32,(n,))
        if np.any(kc_mask>1) or np.any(modulation_mask>1):
            raise ValueError('Masks must contain zero or one')
        _finite('rest',rest)
        dt=float(brain.dt);jump=float(brain.adaptation_jump);tau=float(brain.adaptation_tau)
        if dt!=.1:raise ValueError('CPU batch executor requires dt=0.1 ms')
        if not np.isfinite(jump) or jump<0 or not np.isfinite(tau) or tau<=20:
            raise ValueError('Invalid adaptation parameters')
        slot=np.full(edges,-1,dtype=np.int16)
        slot[plastic_edge]=np.arange(count,dtype=np.int16)
        arrays={'ptr':ptr.copy(),'post':post.copy(),'base_weight':weight.copy(),
            'plastic_slot':slot,'plastic_edge':plastic_edge.copy(),
            'kc_mask':kc_mask.copy(),'modulation_mask':modulation_mask.copy(),
            'rest':rest.copy()}
        for value in arrays.values():value.flags.writeable=False
        identity={**_configuration_identity(brain),
            'base_weight_sha256':digest(arrays['base_weight']),
            'plastic_edges_sha256':digest(arrays['plastic_edge']),
            'kc_mask_sha256':digest(arrays['kc_mask']),
            'modulation_mask_sha256':digest(arrays['modulation_mask']),
            'rest_sha256':digest(arrays['rest'])}
        return cls(neurons=n,edges=edges,delay_slots=int(round(1.8/dt))+1,
            plastic_edges=count,dt_ms=dt,
            eligibility_tau_ms=PARAMETERS['trace_kc_seconds']*1000,
            adaptation_jump_mv=jump,adaptation_tau_ms=tau,
            arrays=arrays,identity=identity)

    @property
    def shared_bytes(self):return sum(value.nbytes for value in self.arrays.values())

    def metadata(self):
        return {'neurons':self.neurons,'edges':self.edges,
            'delay_slots':self.delay_slots,'plastic_edges':self.plastic_edges,
            'dt_ms':self.dt_ms,'shared_bytes':self.shared_bytes,
            'identity':self.identity.copy()}

    def assert_compatible(self,brain):
        if _configuration_identity(brain)!={key:self.identity[key]
                for key in ['ptr_sha256','post_sha256','initial_weight_sha256','configuration_sha256']}:
            raise ValueError('Brain graph identity does not match shared graph')


class CpuBatchLane:
    """One independent set of mutable neural state and plastic efficacies."""

    STATE_FIELDS=('v','g','refractory','drive','previous_drive','queue','queue_count',
        'counts','active','active_flag','nactive','last','eligibility','eligibility_last',
        'modulation','modulation_last','adaptation')

    def __init__(self,graph_identity,arrays):
        self.graph_identity=graph_identity;self.arrays=arrays
        for name,value in arrays.items():setattr(self,name,value)

    @classmethod
    def from_brain(cls,graph,brain):
        if not isinstance(graph,SharedCpuGraph):raise TypeError('SharedCpuGraph required')
        graph.assert_compatible(brain)
        n=graph.neurons;slots=graph.delay_slots
        specifications={'v':(np.float32,(n,)),'g':(np.float32,(n,)),
            'refractory':(np.int16,(n,)),'drive':(np.float32,(n,)),
            'previous_drive':(np.float32,(n,)),'queue':(np.int32,(slots,n)),
            'queue_count':(np.int32,(slots,)),'counts':(np.int32,(n,)),
            'active':(np.int32,(n,)),'active_flag':(np.uint8,(n,)),
            'nactive':(np.int32,(1,)),'last':(np.int64,(n,)),
            'eligibility':(np.float64,(n,)),'eligibility_last':(np.int64,(n,)),
            'modulation':(np.float32,(n,)),'modulation_last':(np.int64,(n,)),
            'adaptation':(np.float32,(n,))}
        arrays={name:_array(name,getattr(brain,name),dtype,shape).copy()
            for name,(dtype,shape) in specifications.items()}
        arrays['cursor']=np.asarray([brain.cursor],dtype=np.int64)
        arrays['plastic_weights']=np.ascontiguousarray(
            brain.weight[graph.plastic_edge],dtype=np.float32)
        return cls(graph.identity.copy(),arrays)

    @property
    def lane_bytes(self):return sum(value.nbytes for value in self.arrays.values())

    def copy_to_brain(self,brain):
        identity=_configuration_identity(brain)
        expected={key:self.graph_identity[key] for key in identity}
        if identity!=expected:raise ValueError('Brain graph identity does not match lane')
        for name in self.STATE_FIELDS:getattr(brain,name)[:]=self.arrays[name]
        brain.cursor=int(self.cursor[0])
        brain.weight[np.asarray(brain.circuit['edges'],dtype=np.int64)]=self.plastic_weights


class MultiTrajectoryCpuExecutor:
    """Validated registration boundary for the native executor."""

    def __init__(self,graph,lanes,workers):
        if not isinstance(graph,SharedCpuGraph):raise TypeError('SharedCpuGraph required')
        self.graph=graph;self.lanes=list(lanes)
        if not self.lanes:raise ValueError('At least one lane is required')
        if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=len(self.lanes):
            raise ValueError('workers must be between one and the lane count')
        for lane in self.lanes:
            if not isinstance(lane,CpuBatchLane) or lane.graph_identity!=graph.identity:
                raise ValueError('Lane graph identity does not match shared graph')
        seen=[]
        for lane in self.lanes:
            for array in lane.arrays.values():
                if any(np.shares_memory(array,other) for other in seen):
                    raise ValueError('Mutable lane buffers must not alias')
                seen.append(array)
        self.workers=workers;self._closed=False;self.last_timing={}
        self._advance_lock=threading.Lock();self._lifecycle_lock=threading.Lock()
        self.build=build(DEFAULT_OUTPUT)
        self._library=C.CDLL(str(library_path(DEFAULT_OUTPUT)))
        self._library.df_cpu_batch_create.argtypes=[C.POINTER(_NativeGraph),
            C.POINTER(_NativeLane),C.c_int32,C.c_int32,C.POINTER(C.c_void_p)]
        self._library.df_cpu_batch_create.restype=C.c_int
        self._library.df_cpu_batch_advance.argtypes=[C.c_void_p,C.c_int32,
            C.POINTER(_NativeTiming)]
        self._library.df_cpu_batch_advance.restype=C.c_int
        self._library.df_cpu_batch_error.argtypes=[C.c_void_p]
        self._library.df_cpu_batch_error.restype=C.c_char_p
        self._library.df_cpu_batch_destroy.argtypes=[C.c_void_p]
        self._library.df_cpu_batch_destroy.restype=None
        self._native_graph=self._graph_descriptor()
        lane_type=_NativeLane*len(self.lanes)
        self._native_lanes=lane_type(*(self._lane_descriptor(lane) for lane in self.lanes))
        self._handle=C.c_void_p()
        status=self._library.df_cpu_batch_create(C.byref(self._native_graph),
            self._native_lanes,len(self.lanes),workers,C.byref(self._handle))
        if status:self._raise_native(status)

    def _graph_descriptor(self):
        graph=self.graph
        return _NativeGraph(graph.neurons,graph.edges,graph.delay_slots,
            graph.plastic_edges,graph.dt_ms,graph.eligibility_tau_ms,
            graph.adaptation_jump_mv,graph.adaptation_tau_ms,
            *[_pointer(graph.arrays[name]) for name in ['ptr','post','base_weight',
                'plastic_slot','kc_mask','modulation_mask','rest']])

    @staticmethod
    def _lane_descriptor(lane):
        names=['cursor','plastic_weights','v','g','refractory','drive',
            'previous_drive','queue','queue_count','counts','active','active_flag',
            'nactive','last','eligibility','eligibility_last','modulation',
            'modulation_last','adaptation']
        return _NativeLane(*[_pointer(lane.arrays[name]) for name in names])

    def _raise_native(self,status):
        message=self._library.df_cpu_batch_error(self._handle)
        raise BackendError(message.decode() if message else f'CPU batch error {status}')

    def advance(self,steps):
        if self._closed:raise BackendError('CPU batch executor is closed')
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=100:
            raise ValueError('CPU batch steps must be 1 through 100')
        if not self._advance_lock.acquire(blocking=False):
            raise BackendError('Reentrant CPU batch advance is forbidden')
        try:
            if self._closed:raise BackendError('CPU batch executor is closed')
            for lane in self.lanes:lane.counts.fill(0)
            timing=_NativeTiming()
            status=self._library.df_cpu_batch_advance(self._handle,steps,C.byref(timing))
            if status:self._raise_native(status)
            self.last_timing={'native_wall_seconds':timing.native_wall_seconds,
                'lanes_advanced':timing.lanes_advanced,'workers':timing.workers,
                'steps':timing.steps,'generation':timing.generation,
                'pool_threads':timing.pool_threads}
            return [lane.counts.copy() for lane in self.lanes]
        finally:self._advance_lock.release()

    def metadata(self):
        return {'name':'cpu-batch','build':self.build,'workers':self.workers,
            'lanes':len(self.lanes),'shared_graph_bytes':self.graph.shared_bytes,
            'lane_bytes':[lane.lane_bytes for lane in self.lanes],
            'total_lane_bytes':sum(lane.lane_bytes for lane in self.lanes),
            'graph':self.graph.metadata(),'closed':self._closed}

    def close(self):
        with self._lifecycle_lock:
            if self._closed:return
            self._closed=True
            if self._handle.value:self._library.df_cpu_batch_destroy(self._handle)
            self._handle=C.c_void_p()

    def __enter__(self):return self

    def __exit__(self,exception_type,exception,traceback):self.close()

    def __del__(self):
        if hasattr(self,'_closed'):
            try:self.close()
            except Exception:pass
