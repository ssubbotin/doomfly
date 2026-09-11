"""Python-owned shared graph and compact mutable CPU trajectory state."""
import ctypes as C
import hashlib
import json
import threading
from types import MappingProxyType

import numpy as np

from doom_learning.common import digest
from doom_learning_v6.backend import BackendError
from doom_learning_v6.rule import PARAMETERS

from .build import ABI_VERSION,DEFAULT_OUTPUT,build,library_path


_OWNER_TOKEN=object()
_STATUS_POISONED=2
_GRAPH_NATIVE_FIELDS=('ptr','post','base_weight','plastic_slot','kc_mask',
    'modulation_mask','rest')
_LANE_NATIVE_FIELDS=('cursor','plastic_weights','v','g','refractory','drive',
    'previous_drive','queue','queue_count','counts','active','active_flag',
    'nactive','last','eligibility','eligibility_last','modulation',
    'modulation_last','adaptation')


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


def _array(name,value,dtype,shape,*,writeable=None):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype(dtype):
        raise ValueError(f'Invalid {name} dtype')
    if value.shape!=shape:raise ValueError(f'Invalid {name} shape')
    if not value.flags.c_contiguous:raise ValueError(f'{name} must be C-contiguous')
    if not value.flags.aligned:raise ValueError(f'{name} must be aligned')
    if writeable is True and not value.flags.writeable:
        raise ValueError(f'{name} must be writeable')
    if writeable is False and value.flags.writeable:
        raise ValueError(f'{name} must be read-only')
    return value


def _finite(name,value):
    if not np.isfinite(value).all():raise ValueError(f'{name} must be finite')


def _configuration_identity(brain):
    signature=json.dumps(brain.configuration_signature(),sort_keys=True,separators=(',',':'))
    return {'ptr_sha256':digest(brain.ptr),'post_sha256':digest(brain.post),
        'initial_weight_sha256':brain.initial_weight_sha256,
        'configuration_sha256':hashlib.sha256(signature.encode()).hexdigest()}


def _graph_schema(graph):
    return {'ptr':(np.int64,(graph.neurons+1,)),
        'post':(np.int32,(graph.edges,)),
        'base_weight':(np.float32,(graph.edges,)),
        'plastic_slot':(np.int16,(graph.edges,)),
        'plastic_edge':(np.int64,(graph.plastic_edges,)),
        'kc_mask':(np.uint8,(graph.neurons,)),
        'modulation_mask':(np.uint8,(graph.neurons,)),
        'rest':(np.float32,(graph.neurons,))}


def _lane_schema(graph):
    n=graph.neurons;slots=graph.delay_slots
    return {'v':(np.float32,(n,)),'g':(np.float32,(n,)),
        'refractory':(np.int16,(n,)),'drive':(np.float32,(n,)),
        'previous_drive':(np.float32,(n,)),'queue':(np.int32,(slots,n)),
        'queue_count':(np.int32,(slots,)),'counts':(np.int32,(n,)),
        'active':(np.int32,(n,)),'active_flag':(np.uint8,(n,)),
        'nactive':(np.int32,(1,)),'last':(np.int64,(n,)),
        'eligibility':(np.float64,(n,)),'eligibility_last':(np.int64,(n,)),
        'modulation':(np.float32,(n,)),'modulation_last':(np.int64,(n,)),
        'adaptation':(np.float32,(n,)),'cursor':(np.int64,(1,)),
        'plastic_weights':(np.float32,(graph.plastic_edges,))}


def _validate_graph_buffer_schema(graph,*,registered=False):
    schema=_graph_schema(graph)
    if tuple(graph.arrays)!=tuple(schema):raise ValueError('Invalid shared graph buffer schema')
    for name,(dtype,shape) in schema.items():
        label=f'registered {name} buffer' if registered else name
        value=_array(label,graph.arrays[name],dtype,shape,writeable=False)
        if getattr(graph,name,None) is not value:
            raise ValueError(f'Shared graph {name} buffer binding changed')


def _validate_graph_buffers(graph):
    _validate_graph_buffer_schema(graph)
    if graph.ptr[0]!=0 or graph.ptr[-1]!=graph.edges or np.any(np.diff(graph.ptr)<0):
        raise ValueError('Invalid shared graph CSR')
    if np.any(graph.post<0) or np.any(graph.post>=graph.neurons):
        raise ValueError('Shared graph index out of range')
    if np.any(graph.plastic_edge<0) or np.any(graph.plastic_edge>=graph.edges):
        raise ValueError('Shared graph plastic edge index out of range')
    if len(np.unique(graph.plastic_edge))!=graph.plastic_edges:
        raise ValueError('Shared graph plastic edges must be unique')
    expected=np.full(graph.edges,-1,dtype=np.int16)
    expected[graph.plastic_edge]=np.arange(graph.plastic_edges,dtype=np.int16)
    if not np.array_equal(graph.plastic_slot,expected):
        raise ValueError('Invalid shared graph plastic slots')
    if np.any(graph.kc_mask>1) or np.any(graph.modulation_mask>1):
        raise ValueError('Shared graph masks must contain zero or one')
    _finite('base_weight',graph.base_weight);_finite('rest',graph.rest)


def _validate_lane_buffers(graph,lane,*,finite,registered=False):
    schema=_lane_schema(graph)
    if tuple(lane.arrays)!=tuple(schema):raise ValueError('Invalid lane buffer schema')
    for name,(dtype,shape) in schema.items():
        label=f'registered {name} buffer' if registered else name
        value=_array(label,lane.arrays[name],dtype,shape,writeable=True)
        if getattr(lane,name,None) is not value:
            raise ValueError(f'Lane {name} buffer binding changed')
        if finite and np.issubdtype(value.dtype,np.floating):_finite(name,value)


def _validate_runtime_state(graph,lane,steps=None):
    cursor=int(lane.cursor[0])
    if cursor<0:raise ValueError('Invalid CPU batch cursor')
    if steps is not None and cursor>np.iinfo(np.int64).max-steps:
        raise ValueError('CPU batch cursor would overflow')
    nactive=int(lane.nactive[0])
    if not 0<=nactive<=graph.neurons:raise ValueError('Invalid CPU batch active count')
    active=lane.active[:nactive]
    if np.any(active<0) or np.any(active>=graph.neurons):
        raise ValueError('CPU batch active index out of range')
    if len(np.unique(active))!=nactive:raise ValueError('CPU batch active indices must be unique')
    if np.any(lane.active_flag>1) or np.count_nonzero(lane.active_flag)!=nactive or \
            (nactive and not np.all(lane.active_flag[active]==1)):
        raise ValueError('CPU batch active flags do not match active indices')
    if np.any(lane.queue_count<0) or np.any(lane.queue_count>graph.neurons):
        raise ValueError('Invalid CPU batch queue count')
    for slot,count in enumerate(lane.queue_count):
        queued=lane.queue[slot,:int(count)]
        if np.any(queued<0) or np.any(queued>=graph.neurons):
            raise ValueError('CPU batch queued neuron index out of range')
    delay=graph.delay_slots-1;future=(cursor%graph.delay_slots+delay)%graph.delay_slots
    if lane.queue_count[future]!=0:
        raise ValueError('CPU batch future queue slot must be empty')
    for name,minimum in (('last',-1),('eligibility_last',0),('modulation_last',0)):
        timestamps=lane.arrays[name]
        if np.any(timestamps<minimum) or np.any(timestamps>cursor):
            raise ValueError(f'Invalid CPU batch {name} timestamp')


class SharedCpuGraph:
    """Read-only v6 graph storage retained by every registered lane."""

    def __init__(self,*,neurons,edges,delay_slots,plastic_edges,dt_ms,
            eligibility_tau_ms,adaptation_jump_mv,adaptation_tau_ms,arrays,identity,
            _token=None):
        if _token is not _OWNER_TOKEN:
            raise TypeError('Use SharedCpuGraph.from_brain()')
        self.neurons=neurons;self.edges=edges;self.delay_slots=delay_slots
        self.plastic_edges=plastic_edges;self.dt_ms=dt_ms
        self.eligibility_tau_ms=eligibility_tau_ms
        self.adaptation_jump_mv=adaptation_jump_mv
        self.adaptation_tau_ms=adaptation_tau_ms
        self.arrays=MappingProxyType(dict(arrays))
        self.identity=MappingProxyType(dict(identity))
        for name,value in self.arrays.items():setattr(self,name,value)
        self._sealed=True

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False) and not name.startswith('_'):
            raise AttributeError('Shared graph metadata and buffer bindings are read-only')
        object.__setattr__(self,name,value)

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
            arrays=arrays,identity=identity,_token=_OWNER_TOKEN)

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
        plastic=_array('plastic edges',brain.circuit['edges'],np.int64,
            (self.plastic_edges,))
        if digest(plastic)!=self.identity['plastic_edges_sha256']:
            raise ValueError('Brain plastic edge identity does not match shared graph')
        weight=_array('weight',brain.weight,np.float32,(self.edges,))
        _finite('weight',weight)
        chunk=1_000_000
        for start in range(0,self.edges,chunk):
            stop=min(start+chunk,self.edges);fixed=self.plastic_slot[start:stop]<0
            if not np.array_equal(weight[start:stop][fixed],self.base_weight[start:stop][fixed]):
                raise ValueError('Brain immutable weights do not match shared graph')


class CpuBatchLane:
    """One independent set of mutable neural state and plastic efficacies."""

    STATE_FIELDS=('v','g','refractory','drive','previous_drive','queue','queue_count',
        'counts','active','active_flag','nactive','last','eligibility','eligibility_last',
        'modulation','modulation_last','adaptation')

    def __init__(self,graph_identity,arrays,_token=None):
        if _token is not _OWNER_TOKEN:raise TypeError('Use CpuBatchLane.from_brain()')
        self.graph_identity=MappingProxyType(dict(graph_identity))
        self.arrays=MappingProxyType(dict(arrays))
        for name,value in self.arrays.items():setattr(self,name,value)
        self._sealed=True

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False) and not name.startswith('_'):
            raise AttributeError('Lane metadata and buffer bindings are read-only')
        object.__setattr__(self,name,value)

    @classmethod
    def from_brain(cls,graph,brain):
        if not isinstance(graph,SharedCpuGraph):raise TypeError('SharedCpuGraph required')
        graph.assert_compatible(brain)
        n=graph.neurons;slots=graph.delay_slots
        specifications={name:value for name,value in _lane_schema(graph).items()
            if name not in ('cursor','plastic_weights')}
        arrays={name:_array(name,getattr(brain,name),dtype,shape).copy()
            for name,(dtype,shape) in specifications.items()}
        arrays['cursor']=np.asarray([brain.cursor],dtype=np.int64)
        arrays['plastic_weights']=np.ascontiguousarray(
            brain.weight[graph.plastic_edge],dtype=np.float32)
        lane=cls(graph.identity,arrays,_token=_OWNER_TOKEN)
        _validate_lane_buffers(graph,lane,finite=True);_validate_runtime_state(graph,lane)
        return lane

    @property
    def lane_bytes(self):return sum(value.nbytes for value in self.arrays.values())

    def copy_from_brain(self,graph,brain):
        if not isinstance(graph,SharedCpuGraph) or self.graph_identity!=graph.identity:
            raise ValueError('Lane graph identity does not match shared graph')
        replacement=type(self).from_brain(graph,brain)
        for name,value in replacement.arrays.items():self.arrays[name][:]=value
        return self

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
        _validate_graph_buffers(graph);seen=list(graph.arrays.values())
        for lane in self.lanes:
            _validate_lane_buffers(graph,lane,finite=True);_validate_runtime_state(graph,lane)
            for array in lane.arrays.values():
                if any(np.shares_memory(array,other) for other in seen):
                    raise ValueError('Shared graph and mutable lane buffers must not alias')
                seen.append(array)
        self._registered_graph_buffers=tuple(graph.arrays.values())
        self._registered_lane_buffers=tuple(tuple(lane.arrays.values()) for lane in self.lanes)
        self._registered_graph_addresses=tuple(array.ctypes.data
            for array in self._registered_graph_buffers)
        self._registered_lane_addresses=tuple(tuple(array.ctypes.data for array in buffers)
            for buffers in self._registered_lane_buffers)
        self.workers=workers;self._closed=False;self._poisoned=False;self.last_timing={}
        self._advance_lock=threading.Lock();self._lifecycle_lock=threading.Lock()
        self.build=build(DEFAULT_OUTPUT)
        self._library=C.CDLL(str(library_path(DEFAULT_OUTPUT)))
        self._library.df_cpu_batch_abi_version.argtypes=[]
        self._library.df_cpu_batch_abi_version.restype=C.c_uint32
        native_abi=int(self._library.df_cpu_batch_abi_version())
        if native_abi!=ABI_VERSION:
            raise BackendError(f'Native CPU batch ABI version {native_abi} does not match {ABI_VERSION}')
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
            *[_pointer(graph.arrays[name]) for name in _GRAPH_NATIVE_FIELDS])

    @staticmethod
    def _lane_descriptor(lane):
        return _NativeLane(*[_pointer(lane.arrays[name]) for name in _LANE_NATIVE_FIELDS])

    def _validate_registered_state(self,steps):
        _validate_graph_buffer_schema(self.graph,registered=True)
        for name,array,address in zip(self.graph.arrays,self._registered_graph_buffers,
                self._registered_graph_addresses):
            if self.graph.arrays[name] is not array or array.ctypes.data!=address:
                raise ValueError(f'Invalid registered {name} buffer')
        for lane,buffers,addresses in zip(self.lanes,self._registered_lane_buffers,
                self._registered_lane_addresses):
            _validate_lane_buffers(self.graph,lane,finite=False,registered=True)
            for name,array,address in zip(lane.arrays,buffers,addresses):
                if lane.arrays[name] is not array or array.ctypes.data!=address:
                    raise ValueError(f'Invalid registered {name} buffer')
            _validate_runtime_state(self.graph,lane,steps)

    def _raise_native(self,status):
        message=self._library.df_cpu_batch_error(self._handle)
        raise BackendError(message.decode() if message else f'CPU batch error {status}')

    def advance(self,steps):
        if self._closed:raise BackendError('CPU batch executor is closed')
        if self._poisoned:raise BackendError('CPU batch executor is poisoned')
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=100:
            raise ValueError('CPU batch steps must be 1 through 100')
        if not self._advance_lock.acquire(blocking=False):
            raise BackendError('Reentrant CPU batch advance is forbidden')
        try:
            with self._lifecycle_lock:
                if self._closed:raise BackendError('CPU batch executor is closed')
                if self._poisoned:raise BackendError('CPU batch executor is poisoned')
                self._validate_registered_state(steps)
                timing=_NativeTiming()
                status=self._library.df_cpu_batch_advance(self._handle,steps,C.byref(timing))
                if status:
                    if status==_STATUS_POISONED:self._poisoned=True
                    self._raise_native(status)
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
            'poisoned':self._poisoned,
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
