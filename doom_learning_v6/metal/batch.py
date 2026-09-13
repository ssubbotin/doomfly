"""Validated, caller-serialized ownership of independent resident Metal lanes."""
from contextlib import contextmanager
import copy
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import warnings

import numpy as np

from doom_learning.common import OUT, digest
from ..backend import BackendError
from ..brain import MemoryBrain
from ..visual import VisualMemoryBrain
from .backend import Graph, State, KCEvent, Timing, _pointer
from .build import (ABI_VERSION, DEFAULT_OUTPUT, NUMERICAL_PARENT, NUMERICAL_ORDER,
                    library_path, probe)
from .graph import load_or_build_incoming


_STATE_NAMES = ('weight', 'v', 'g', 'refractory', 'drive', 'previous_drive',
                'queue', 'queue_count', 'counts', 'active', 'active_flag', 'nactive',
                'last', 'modulation', 'modulation_last', 'rest', 'adaptation')
_GRAPH_NAMES = ('ptr', 'post', 'ids', 'retina', 'uv', 'lamina', 'sugar', 'rest',
                'modulation_mask', 'tonic', 'dan_baseline_hz', 'baseline_plastic')
_STORAGE_TIMINGS = {'edge_bitmap_words', 'shared_resident_bytes', 'mutable_resident_bytes'}


def _array(name, value, dtype, shape, *, mutable=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f'Invalid {name} array schema')
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f'{name} must be contiguous and aligned')
    if mutable and not value.flags.writeable:
        raise ValueError(f'{name} must be writeable')
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite')
    return value


def _mutable_schema(b):
    n, slots, p, d = b.n, b.queue.shape[0], len(b.circuit['edges']), len(b.circuit['dan'])
    schema = {'weight': (np.float32, (len(b.post),)), 'v': (np.float32, (n,)),
        'g': (np.float32, (n,)), 'refractory': (np.int16, (n,)),
        'drive': (np.float32, (n,)), 'previous_drive': (np.float32, (n,)),
        'queue': (np.int32, (slots, n)), 'queue_count': (np.int32, (slots,)),
        'counts': (np.int32, (n,)), 'active': (np.int32, (n,)),
        'active_flag': (np.uint8, (n,)), 'nactive': (np.int32, (1,)),
        'last': (np.int64, (n,)), 'modulation': (np.float32, (n,)),
        'modulation_last': (np.int64, (n,)), 'adaptation': (np.float32, (n,)),
        'luminance': (np.float32, (len(b.retina),)), 'eligibility': (np.float64, (n,)),
        'eligibility_last': (np.int64, (n,)), 'rate_kc': (np.float64, (p,)),
        'rate_dan': (np.float64, (d,)), 'memory_u': (np.float64, (p,)),
        'memory_w': (np.float64, (p,))}
    if isinstance(b, VisualMemoryBrain): schema['r8_light'] = (np.float32, (len(b.r8),))
    if set(b.fields) != set(schema) - {'weight'}:
        raise ValueError('Unsupported brain state fields')
    return schema


def _configuration_arrays(b):
    arrays = {name: getattr(b, name) for name in _GRAPH_NAMES}
    arrays.update({f'circuit.{name}': value for name, value in b.circuit.items()
                   if isinstance(value, np.ndarray)})
    if isinstance(b, VisualMemoryBrain):
        arrays.update({name: getattr(b, name) for name in ('r8', 'r8_uv', 'r8_channel', 'corrected_edges')})
    return arrays


def _bound_array(b, name):
    return b.circuit[name[8:]] if name.startswith('circuit.') else getattr(b, name)


def _identity(b):
    return json.dumps({'configuration': b.configuration_signature(),
        'ids': digest(b.ids), 'ptr': digest(b.ptr), 'post': digest(b.post),
        'circuit': {k: digest(v) for k, v in b.circuit.items() if isinstance(v, np.ndarray)},
        'baseline_plastic': digest(b.baseline_plastic), 'dt': b.dt, 'eta': b.eta,
        'slots': b.queue.shape[0]}, sort_keys=True)


def _validate_state(b):
    cursor = b.cursor
    if isinstance(cursor, (bool, np.bool_)) or not isinstance(cursor, (int, np.integer)) or not 0 <= cursor <= np.iinfo(np.int64).max:
        raise ValueError('Invalid lane cursor')
    nactive = int(b.nactive[0])
    if not 0 <= nactive <= b.n: raise ValueError('Invalid active count')
    active = b.active[:nactive]
    if np.any(active < 0) or np.any(active >= b.n) or len(np.unique(active)) != nactive:
        raise ValueError('Invalid active neuron indices')
    if np.any(b.active_flag > 1) or np.count_nonzero(b.active_flag) != nactive or not np.all(b.active_flag[active] == 1):
        raise ValueError('Active flags and indices disagree')
    if np.any(b.queue_count < 0) or np.any(b.queue_count > b.n): raise ValueError('Invalid queue counts')
    for slot, count in enumerate(b.queue_count):
        queued = b.queue[slot, :int(count)]
        if np.any(queued < 0) or np.any(queued >= b.n) or len(np.unique(queued)) != len(queued):
            raise ValueError('Invalid delayed neuron indices')
    if np.any(b.refractory < 0) or np.any(b.counts < 0): raise ValueError('Invalid discrete state')
    if np.any(b.last < cursor - (np.iinfo(np.int32).max - 99)) or np.any(b.last > cursor):
        raise ValueError('Unsafe neural history')
    for name in ('modulation_last', 'eligibility_last'):
        value = getattr(b, name)
        if np.any(value < 0) or np.any(value > cursor): raise ValueError(f'Invalid {name} history')


def _validate_graph(b):
    n, edges = b.n, len(b.post)
    if not 0 < n <= np.iinfo(np.int32).max or edges > np.iinfo(np.int32).max or b.dt != .1:
        raise ValueError('Unsupported neural graph dimensions/dt')
    _array('ptr', b.ptr, np.int64, (n + 1,)); _array('post', b.post, np.int32, (edges,))
    if b.ptr[0] != 0 or b.ptr[-1] != edges or np.any(np.diff(b.ptr) < 0) or np.any(b.post < 0) or np.any(b.post >= n):
        raise ValueError('Invalid graph indices')
    _array('ids', b.ids, np.int64, (n,))
    for name in ('retina', 'lamina', 'sugar'):
        value = _array(name, getattr(b, name), np.int32, (len(getattr(b, name)),))
        if np.any(value < 0) or np.any(value >= n): raise ValueError('Invalid sensory indices')
    uv = _array('uv', b.uv, np.float32, (len(b.retina), 2))
    if np.any(uv < 0) or np.any(uv > 1): raise ValueError('Invalid sensory coordinates')
    for name in ('rest', 'tonic'): _array(name, getattr(b, name), np.float32, (n,))
    for name, value in (('kc_mask', b.circuit['kc_mask']), ('modulation_mask', b.modulation_mask)):
        _array(name, value, np.uint8, (n,))
        if np.any(value > 1): raise ValueError('Invalid neural mask')
    p, d = len(b.circuit['edges']), len(b.circuit['dan'])
    slots = _array('plastic edges', b.circuit['edges'], np.int64, (p,))
    if np.any(slots < 0) or np.any(slots >= edges) or len(np.unique(slots)) != p:
        raise ValueError('Invalid plastic edge indices')
    pre = _array('plastic pre', b.circuit['pre'], np.int32, (p,))
    if np.any(pre < 0) or np.any(pre >= n): raise ValueError('Invalid plastic neuron indices')
    dan = b.circuit['dan']
    if dan.ndim != 1 or not np.issubdtype(dan.dtype, np.integer) or np.any(dan < 0) or np.any(dan >= n):
        raise ValueError('Invalid DAN indices')
    _array('gain', b.circuit['gain'], np.float32, (d, p))
    _array('baseline_plastic', b.baseline_plastic, np.float32, (p,))
    _array('dan_baseline_hz', b.dan_baseline_hz, np.float64, (d,))
    if not math.isfinite(b.eta) or b.eta < 0 or not math.isfinite(b.adaptation_jump) or b.adaptation_jump < 0 or not math.isfinite(b.adaptation_tau) or b.adaptation_tau <= 20:
        raise ValueError('Invalid dynamics parameters')
    if b.queue.ndim != 2 or b.queue.shape[0] != int(round(1.8 / b.dt)) + 1:
        raise ValueError('Invalid delayed ring shape')
    if isinstance(b, VisualMemoryBrain):
        r8 = _array('r8', b.r8, np.int32, (len(b.r8),))
        uv = _array('r8_uv', b.r8_uv, np.float32, (len(r8), 2))
        channel = _array('r8_channel', b.r8_channel, np.int32, (len(r8),))
        if np.any(r8 < 0) or np.any(r8 >= n) or np.any(uv < 0) or np.any(uv > 1) or np.any(channel < 0) or np.any(channel > 2):
            raise ValueError('Invalid R8 input mapping')


def _validate_nonaliasing(records):
    spans = []
    for lane, mutable, arrays in records:
        for name, value in arrays.items():
            if value.nbytes:
                spans.append((value.ctypes.data, value.ctypes.data + value.nbytes, mutable, lane, name))
    spans.sort()
    for index, span in enumerate(spans):
        for other in spans[index + 1:]:
            if other[0] >= span[1]: break
            if span[2] or other[2]: raise ValueError(f'Unsafe mutable buffer alias: {span[3:]} / {other[3:]}')


def _accumulate(total, timing):
    for key, value in timing.items():
        total[key] = max(total.get(key, 0), value) if key in _STORAGE_TIMINGS else total.get(key, 0) + value


class _LaneBackend:
    name = 'metal-batch'

    def __init__(self, owner, lane):
        self.owner, self.lane = owner, lane
        self.brain = owner.brains[lane]
        self.capture_spikes = False; self.spike_events = []; self.last_kc_events = []
        self.last_timing = {}; self._host_state_valid = True
        self._host_weight_epoch = 0; self._device_weight_epoch = 0
        self._last_materialization_reason = None

    @property
    def poisoned(self): return self.owner.poisoned

    def advance(self, steps): raise BackendError('Brain is owned by a Metal batch executor')
    def ensure_initialized(self): self.owner._check_open()
    def metadata(self): return {**self.owner.metadata(), 'lane_index': self.lane}
    def sync_for_checkpoint(self): return self.materialize('checkpoint')

    def materialize(self, reason):
        if not reason: raise ValueError('Metal materialization reason is required')
        with self.owner._operation(adapter=True):
            self.owner._validate_bindings()
            if self.owner.poisoned: raise BackendError('Metal batch owner is poisoned; explicit restore required')
            self.owner._materialize_lane(self.lane, reason)

    def restore_from_host(self, reason=None):
        with self.owner._operation(adapter=True, allow_poison=True, allow_closed_restore=True):
            if self.lane in self.owner._released:
                raise BackendError('Metal batch lane adapter is closed')
            if reason not in ('reset', 'restore') and not self._host_state_valid:
                raise BackendError('Metal host neural state is stale')
            if self.owner.poisoned and reason not in ('reset', 'restore'):
                raise BackendError('Explicit lane reset/restore required')
            self.owner._validate_bindings()
            self.owner._validate_host_lane(self.lane)
            if self.owner.closed:
                self.owner._release_lane(self.lane)
                self.brain.backend.restore_from_host(reason=reason)
                return
            self.owner._error(self.owner.library.df_metal_upload_lane_state(
                self.owner.handle, self.lane, C.byref(self.owner._state(self.lane))))
            self.owner._device_cursors[self.lane] = self.brain.cursor
            self._host_state_valid = True
            self._host_weight_epoch += 1; self._device_weight_epoch = self._host_weight_epoch
            if self.owner.poisoned:
                self.owner._restored_lanes.add(self.lane)
                if len(self.owner._restored_lanes) == len(self.owner.brains): self.owner.poisoned = False

    def update_weights(self, edge_ids, values):
        with self.owner._operation(adapter=True):
            self.owner._validate_bindings()
            ids = np.ascontiguousarray(edge_ids, dtype=np.int64)
            values = np.ascontiguousarray(values, dtype=np.float32)
            if ids.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all() or not np.isin(ids, self.brain.circuit['edges']).all():
                raise ValueError('Invalid plastic edge/value update')
            started = time.perf_counter()
            self.owner._error(self.owner.library.df_metal_update_lane_weights(
                self.owner.handle, self.lane, len(ids), _pointer(ids), _pointer(values)), poison=True)
            elapsed = time.perf_counter() - started
            self._host_weight_epoch += 1; self._device_weight_epoch = self._host_weight_epoch
            self.last_timing['sparse_weight_update_seconds'] = self.last_timing.get('sparse_weight_update_seconds', 0.) + elapsed
            self.last_timing['sparse_weight_update_bytes'] = self.last_timing.get('sparse_weight_update_bytes', 0) + ids.nbytes + values.nbytes
            self.owner.last_timing['sparse_weight_update_seconds'] = self.owner.last_timing.get('sparse_weight_update_seconds', 0.) + elapsed
            self.owner.last_timing['sparse_weight_update_bytes'] = self.owner.last_timing.get('sparse_weight_update_bytes', 0) + ids.nbytes + values.nbytes

    def start_diagnostics(self):
        with self.owner._operation(adapter=True):
            self.owner._error(self.owner.library.df_metal_set_diagnostics(self.owner.handle, 1))
            self.capture_spikes = True; self.spike_events = []

    def stop_diagnostics(self):
        with self.owner._operation(adapter=True):
            self.owner._materialize_lane(self.lane, 'diagnostics')
            self.capture_spikes = False
            self.owner._error(self.owner.library.df_metal_set_diagnostics(
                self.owner.handle, int(any(a.capture_spikes for a in self.owner.adapters))))

    def close(self): self.owner.close()


class MetalBatchExecutor:
    """One native command stream, separate host rules and copied observations."""

    def __init__(self, brains):
        self._lock = threading.Lock(); self._thread = None
        self.closed = False; self.poisoned = False
        self.handle = C.c_void_p(); self.library = None; self.last_timing = {}
        self.brains = tuple(brains); self.adapters = []; self._frozen = []
        self._restored_lanes = set(); self._released = set()
        self._mutable = []; self._configuration = []; self._registrations = []
        if not self.brains or len(self.brains) > np.iinfo(np.int32).max:
            raise ValueError('A nonempty bounded lane collection is required')
        if len({id(b) for b in self.brains}) != len(self.brains): raise ValueError('Brains must be distinct')
        self._original = tuple(b.backend for b in self.brains if isinstance(b, MemoryBrain))
        identities, records = [], []
        for lane, b in enumerate(self.brains):
            if not isinstance(b, MemoryBrain): raise TypeError('MemoryBrain lanes required')
            if getattr(b, '_metal_batch_owner', None) is not None: raise ValueError('Brain is already owned')
            if getattr(getattr(b.backend, 'handle', None), 'value', None): raise ValueError('Initialized Metal backend cannot be attached')
            if b.backend.name not in ('cpu', 'metal'): raise ValueError('Unsupported original backend')
            _validate_graph(b)
            mutable = {name: _array(name, getattr(b, name), dtype, shape, mutable=True)
                       for name, (dtype, shape) in _mutable_schema(b).items()}
            _validate_state(b)
            config = _configuration_arrays(b)
            for name, value in config.items():
                _array(name, value, value.dtype, value.shape)
            self._mutable.append(mutable); self._configuration.append(config)
            records.extend(((lane, True, mutable), (lane, False, config)))
            identities.append(_identity(b))
        _validate_nonaliasing(records)
        if any(identity != identities[0] for identity in identities): raise ValueError('Graph/configuration/plastic identities mismatch')
        fixed = np.ones(len(self.brains[0].post), dtype=bool); fixed[self.brains[0].circuit['edges']] = False
        reference = self.brains[0].weight[fixed].tobytes()
        if any(b.weight[fixed].tobytes() != reference for b in self.brains[1:]): raise ValueError('Fixed weights mismatch')
        self._identity = identities[0]
        self._device_cursors = [b.cursor for b in self.brains]
        self._scalar_configuration = [(b.n, b.dt, b.eta, b.adaptation_jump, b.adaptation_tau,
            json.dumps(b.rule_parameters, sort_keys=True), tuple(b.fields)) for b in self.brains]
        for mutable, config in zip(self._mutable, self._configuration):
            self._registrations.append({name: (a, a.ctypes.data, a.dtype, a.shape)
                                        for name, a in {**mutable, **config}.items()})
        try:
            started = time.perf_counter()
            self._metadata = probe(DEFAULT_OUTPUT)
            self.library = C.CDLL(str(library_path(DEFAULT_OUTPUT)))
            self._setup_library()
            b = self.brains[0]
            signature = hashlib.sha256(b.ptr.tobytes() + b.post.tobytes()).hexdigest()
            self.incoming = load_or_build_incoming(b.ptr, b.post, OUT / 'metal' / 'graphs' / signature)
            i = self.incoming
            graph = Graph(b.n, len(b.post), b.queue.shape[0], b.dt, b.adaptation_jump, b.adaptation_tau,
                _pointer(b.ptr), _pointer(b.post), _pointer(i.ptr), _pointer(i.pre), _pointer(i.edge),
                _pointer(b.circuit['kc_mask']), _pointer(b.modulation_mask))
            self._error(self.library.df_metal_create_batch(C.byref(graph),
                str(Path(DEFAULT_OUTPUT) / 'kernels.metallib').encode(), len(self.brains), C.byref(self.handle)))
            for lane in range(len(self.brains)):
                self._error(self.library.df_metal_upload_lane_state(self.handle, lane, C.byref(self._state(lane))))
            shared, mutable = C.c_uint64(), C.c_uint64()
            self._error(self.library.df_metal_batch_memory_bytes(self.handle, C.byref(shared), C.byref(mutable)))
            self._memory = {'shared_resident_bytes': shared.value, 'mutable_resident_bytes': mutable.value}
            self.initialization_timing = {'total_seconds': time.perf_counter() - started}
            seen = set()
            for config in self._configuration:
                for array in config.values():
                    if id(array) not in seen:
                        seen.add(id(array)); self._frozen.append((array, array.flags.writeable))
                        array.flags.writeable = False
            self.adapters = [_LaneBackend(self, lane) for lane in range(len(self.brains))]
            for b, adapter in zip(self.brains, self.adapters):
                b.backend = adapter; b._metal_batch_owner = self
        except BaseException:
            self._destroy()
            self._unfreeze()
            self.closed = True
            raise

    def _setup_library(self):
        self.library.df_metal_last_error.restype = C.c_char_p
        signatures = {
            'df_metal_create_batch': [C.POINTER(Graph), C.c_char_p, C.c_int32, C.POINTER(C.c_void_p)],
            'df_metal_upload_lane_state': [C.c_void_p, C.c_int32, C.POINTER(State)],
            'df_metal_download_lane_state': [C.c_void_p, C.c_int32, C.POINTER(State)],
            'df_metal_upload_lane_drive': [C.c_void_p, C.c_int32, C.c_void_p],
            'df_metal_download_lane_observation': [C.c_void_p, C.c_int32, C.c_void_p, C.POINTER(C.c_int64)],
            'df_metal_update_lane_weights': [C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p],
            'df_metal_apply_lane_eligibility': [C.c_void_p, C.c_int32, C.c_void_p, C.c_void_p, C.c_double],
            'df_metal_batch_memory_bytes': [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)],
            'df_metal_advance': [C.c_void_p, C.c_int32, C.POINTER(KCEvent), C.c_int32, C.POINTER(C.c_int32), C.POINTER(Timing)],
            'df_metal_set_diagnostics': [C.c_void_p, C.c_int32]}
        for name, signature in signatures.items():
            function = getattr(self.library, name); function.argtypes = signature; function.restype = C.c_int
        self.library.df_metal_destroy.argtypes = [C.c_void_p]
        self.library.df_metal_destroy.restype = None

    def _state(self, lane): return State(self.brains[lane].cursor, *[_pointer(getattr(self.brains[lane], name)) for name in _STATE_NAMES])

    def _error(self, status, *, poison=False):
        if status:
            if poison: self.poisoned = True; self._restored_lanes.clear()
            message = self.library.df_metal_last_error()
            raise BackendError(message.decode() if message else f'Metal batch error {status}')

    def _check_open(self, *, allow_poison=False, allow_closed_restore=False):
        if self.closed and not (allow_closed_restore and self.poisoned): raise BackendError('Metal batch executor is closed')
        if self.poisoned and not allow_poison: raise BackendError('Metal batch executor is poisoned; explicit restore required')

    @contextmanager
    def _operation(self, *, adapter=False, allow_poison=False, allow_closed_restore=False):
        current = threading.get_ident()
        if self._thread == current:
            if not adapter: raise BackendError('Reentrant Metal batch execution is forbidden')
            self._check_open(allow_poison=allow_poison, allow_closed_restore=allow_closed_restore)
            yield
            return
        if not self._lock.acquire(blocking=False): raise BackendError('Metal batch operation is in-flight')
        self._thread = current
        try:
            self._check_open(allow_poison=allow_poison, allow_closed_restore=allow_closed_restore)
            yield
        finally:
            self._thread = None; self._lock.release()

    def _validate_bindings(self):
        for lane, b in enumerate(self.brains):
            if lane in self._released: continue
            if self.adapters and (b.backend is not self.adapters[lane] or getattr(b, '_metal_batch_owner', None) is not self):
                raise ValueError('Lane ownership binding changed')
            scalar = (b.n, b.dt, b.eta, b.adaptation_jump, b.adaptation_tau,
                      json.dumps(b.rule_parameters, sort_keys=True), tuple(b.fields))
            if scalar != self._scalar_configuration[lane]: raise ValueError('Lane configuration changed')
            for name, (array, address, dtype, shape) in self._registrations[lane].items():
                value = _bound_array(b, name)
                if value is not array or value.ctypes.data != address or value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous or not value.flags.aligned:
                    raise ValueError(f'Registered {name} buffer binding changed')
                mutable = name in self._mutable[lane]
                if value.flags.writeable != mutable and not (self.closed and self.poisoned):
                    raise ValueError(f'Registered {name} writeability changed')

    def _validate_host_lane(self, lane):
        b = self.brains[lane]
        for name, (dtype, shape) in _mutable_schema(b).items(): _array(name, getattr(b, name), dtype, shape, mutable=True)
        _validate_state(b)

    def _validate_restore_candidate(self, brain, archive, metadata):
        self._validate_bindings()
        if not any(brain is b for b in self.brains): raise ValueError('Unknown checkpoint lane')
        cursor, total, frozen = (metadata.get(k) for k in ('cursor', 'total_spikes', 'weights_frozen'))
        if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= np.iinfo(np.int64).max:
            raise ValueError('Invalid checkpoint cursor')
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError('Invalid checkpoint total_spikes')
        if not isinstance(frozen, bool): raise ValueError('Invalid checkpoint weights_frozen')
        candidate = copy.copy(brain)
        candidate.cursor = cursor; candidate.total_spikes = total; candidate.weights_frozen = frozen
        arrays = {}
        for name, (dtype, shape) in _mutable_schema(brain).items():
            try: value = archive[name]
            except KeyError as error: raise ValueError(f'Missing checkpoint {name} array') from error
            arrays[name] = _array(f'checkpoint {name}', value, dtype, shape, mutable=True)
            setattr(candidate, name, value)
        _validate_state(candidate)
        return arrays

    def _validate_cursors(self, steps):
        cursors = [b.cursor for b in self.brains]
        delay = round(1.8 / self.brains[0].dt)
        if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) or c < 0 or c > np.iinfo(np.int64).max - steps - delay for c in cursors):
            raise ValueError('Invalid or overflowing lane cursor')
        if any(c != cursors[0] for c in cursors): raise ValueError('Lane cursors mismatch')
        if cursors != self._device_cursors: raise ValueError('Host lane cursor changed without restore')

    def _validate_advance_inputs(self):
        for b, adapter in zip(self.brains, self.adapters):
            if adapter._host_weight_epoch != adapter._device_weight_epoch:
                raise BackendError('Metal lane weight epochs disagree')
            for name in ('drive', 'eligibility', 'luminance', 'rate_kc', 'rate_dan', 'memory_u', 'memory_w'):
                if not np.isfinite(getattr(b, name)).all(): raise ValueError(f'Invalid lane {name} input')
            if isinstance(b, VisualMemoryBrain) and not np.isfinite(b.r8_light).all():
                raise ValueError('Invalid R8 light input')
            if np.any(b.eligibility_last < 0) or np.any(b.eligibility_last > b.cursor):
                raise ValueError('Invalid lane eligibility history')

    @staticmethod
    def _steps(steps):
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= 100:
            raise ValueError('Metal batch steps must be integer 1-100')
        return int(steps)

    def advance(self, steps):
        with self._operation():
            steps = self._steps(steps); self._validate_bindings(); self._validate_cursors(steps)
            return self._advance(steps)

    def _advance(self, steps):
        self._validate_advance_inputs()
        diagnostics = any(a.capture_spikes for a in self.adapters)
        cells = sum(b.n if diagnostics else int(np.count_nonzero(b.circuit['kc_mask'])) for b in self.brains)
        capacity = max(1, cells * (1 + (steps - 1) // 22))
        if capacity > np.iinfo(np.int32).max: raise ValueError('Diagnostic event capacity would overflow')
        started = time.perf_counter()
        timing = {**self._memory, 'full_upload_seconds': 0., 'full_upload_bytes': 0,
                  'materialize_seconds': 0., 'materialize_bytes': 0,
                  'sparse_weight_update_seconds': 0., 'sparse_weight_update_bytes': 0,
                  'host_rule_seconds': 0.}
        for lane, (b, adapter) in enumerate(zip(self.brains, self.adapters)):
            copied = time.perf_counter()
            self._error(self.library.df_metal_upload_lane_drive(self.handle, lane, _pointer(b.drive)))
            adapter.last_timing = {'drive_copy_seconds': time.perf_counter() - copied, 'drive_copy_bytes': b.drive.nbytes}
            _accumulate(timing, adapter.last_timing)
        events = (KCEvent * capacity)(); count = C.c_int32(); native = Timing()
        self._error(self.library.df_metal_advance(self.handle, steps, events, capacity, C.byref(count), C.byref(native)), poison=True)
        for name, _ in Timing._fields_: timing[name] = getattr(native, name)
        recorded = [[] for _ in self.brains]
        conversion = time.perf_counter()
        for event in events[:count.value]: recorded[event.reserved].append((int(event.tick), int(event.neuron)))
        for lane, (b, adapter) in enumerate(zip(self.brains, self.adapters)):
            adapter.last_kc_events = sorted((tick, neuron) for tick, neuron in recorded[lane] if b.circuit['kc_mask'][neuron])
            if adapter.capture_spikes: adapter.spike_events.extend(sorted((neuron, tick) for tick, neuron in recorded[lane]))
        timing['event_conversion_sort_seconds'] = time.perf_counter() - conversion
        counts = []
        for lane, (b, adapter) in enumerate(zip(self.brains, self.adapters)):
            copied = time.perf_counter()
            self._error(self.library.df_metal_apply_lane_eligibility(self.handle, lane,
                _pointer(b.eligibility), _pointer(b.eligibility_last), b.rule_parameters['trace_kc_seconds'] * 1000), poison=True)
            lane_timing = {'eligibility_seconds': time.perf_counter() - copied}
            copied = time.perf_counter(); cursor = C.c_int64()
            self._error(self.library.df_metal_download_lane_observation(self.handle, lane, _pointer(b.counts), C.byref(cursor)), poison=True)
            lane_timing.update(counts_copy_seconds=time.perf_counter() - copied, counts_copy_bytes=b.counts.nbytes)
            b.cursor = int(cursor.value); self._device_cursors[lane] = b.cursor
            b.sim_ms = b.cursor * b.dt; b.total_spikes += int(b.counts.sum())
            adapter._host_state_valid = False; adapter.last_timing.update(lane_timing)
            _accumulate(timing, lane_timing); counts.append(b.counts.copy())
        elapsed = time.perf_counter() - started
        timing['backend_seconds'] = elapsed; self.last_timing = timing
        return counts, elapsed

    def _inputs(self, inputs, duration_ms, learning, stimulations, lamina_bias, *, rgb):
        self._validate_bindings()
        if not math.isfinite(duration_ms) or duration_ms <= 0 or not math.isfinite(lamina_bias): raise ValueError('Invalid duration/current')
        ticks = round(duration_ms / self.brains[0].dt)
        if ticks < 1: raise ValueError('Duration too short')
        self._validate_cursors(ticks)
        self._validate_advance_inputs()
        try:
            inputs = list(inputs)
            pulses = [None] * len(self.brains) if stimulations is None else list(stimulations)
            flags = [bool(learning)] * len(self.brains) if isinstance(learning, (bool, np.bool_)) else list(learning)
        except TypeError as error: raise ValueError('One input/learning/stimulation per lane required') from error
        if any(len(values) != len(self.brains) for values in (inputs, pulses, flags)) or any(not isinstance(f, (bool, np.bool_)) for f in flags):
            raise ValueError('One input/boolean/stimulation per lane required')
        normalized = []
        normalized_pulses = []
        for b, value, stimulation in zip(self.brains, inputs, pulses):
            value = np.asarray(value)
            if rgb:
                if not isinstance(b, VisualMemoryBrain) or value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8 or not all(value.shape[:2]): raise ValueError('RGB uint8 VisualMemoryBrain input required')
            elif value.shape != (len(b.retina),) or not np.isfinite(value).all(): raise ValueError('Invalid retinal input')
            normalized.append(value.copy())
            checked_pulses = None
            if stimulation is not None:
                entries = stimulation if isinstance(stimulation, list) else [stimulation]
                checked_pulses = []
                try:
                    for indices, current in entries:
                        ix = np.asarray(indices, dtype=np.int32); amplitude = np.asarray(current, dtype=np.float32)
                        if ix.ndim != 1 or np.any(ix < 0) or np.any(ix >= b.n) or not np.isfinite(amplitude).all() or amplitude.shape not in ((), ix.shape): raise ValueError('Invalid external stimulation')
                        checked_pulses.append((ix.copy(), amplitude.copy()))
                except (TypeError, OverflowError) as error: raise ValueError('Invalid external stimulation') from error
            normalized_pulses.append(checked_pulses)
        preflight_started = time.perf_counter()
        self._preflight_inputs(normalized, ticks, normalized_pulses, lamina_bias=lamina_bias, rgb=rgb)
        self._last_preflight_seconds = time.perf_counter() - preflight_started
        return normalized, ticks, flags, normalized_pulses

    def _preflight_inputs(self, inputs, remaining, stimulations, *, lamina_bias, rgb):
        # Shallow objects share immutable configuration only. Every buffer touched
        # by either preparation helper is private scratch; real bindings stay put.
        previews = []
        for b in self.brains:
            preview = copy.copy(b)
            preview.luminance = b.luminance.copy()
            preview.drive = np.empty_like(b.drive)
            if rgb: preview.r8_light = b.r8_light.copy()
            previews.append(preview)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                while remaining:
                    ticks = min(100, remaining)
                    for preview, value, stimulation in zip(previews, inputs, stimulations):
                        if rgb:
                            value, stimulation = VisualMemoryBrain._prepare_rgb_input(
                                preview, value, ticks * preview.dt, stimulation=stimulation)
                        MemoryBrain._prepare_neural_input(preview, value, ticks,
                            stimulation=stimulation, lamina_bias=lamina_bias)
                        if not np.isfinite(preview.drive).all(): raise ValueError('Nonfinite prepared drive')
                    remaining -= ticks
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Input preparation would overflow or become nonfinite') from error

    def step(self, luminances, duration_ms, *, learning=False, stimulations=None, lamina_bias=12.):
        with self._operation():
            args = self._inputs(luminances, duration_ms, learning, stimulations, lamina_bias, rgb=False)
            return self._step(*args, lamina_bias=lamina_bias, rgb=False)

    def rgb_step(self, frames, duration_ms, *, learning=False, stimulations=None, lamina_bias=12.):
        with self._operation():
            args = self._inputs(frames, duration_ms, learning, stimulations, lamina_bias, rgb=True)
            return self._step(*args, lamina_bias=lamina_bias, rgb=True)

    def _step(self, inputs, remaining, flags, stimulations, *, lamina_bias, rgb):
        total = [np.zeros(b.n, dtype=np.int32) for b in self.brains]
        wall = 0.; aggregate = {}; lane_aggregate = [{} for _ in self.brains]
        preparation_seconds = 0.
        for b in self.brains: b.last_rule_seconds = 0.
        while remaining:
            ticks = min(100, remaining)
            preparation_started = time.perf_counter()
            for b, value, stimulation in zip(self.brains, inputs, stimulations):
                if rgb: value, stimulation = b._prepare_rgb_input(value, ticks * b.dt, stimulation=stimulation)
                b._prepare_neural_input(value, ticks, stimulation=stimulation, lamina_bias=lamina_bias)
            preparation_seconds += time.perf_counter() - preparation_started
            counts, elapsed = self._advance(ticks)
            for lane, (b, count, enabled) in enumerate(zip(self.brains, counts, flags)):
                previous = b.last_rule_seconds
                b._apply_centered_rule(count, ticks * b.dt / 1000, enabled)
                self.last_timing['host_rule_seconds'] += b.last_rule_seconds - previous
                _accumulate(lane_aggregate[lane], self.adapters[lane].last_timing)
                total[lane] += count
            _accumulate(aggregate, self.last_timing)
            wall += elapsed; remaining -= ticks
        aggregate['host_rule_seconds'] = sum(b.last_rule_seconds for b in self.brains)
        aggregate['input_preflight_seconds'] = self._last_preflight_seconds
        aggregate['input_preparation_seconds'] = preparation_seconds
        self.last_timing = aggregate
        for b, count, adapter, timing in zip(self.brains, total, self.adapters, lane_aggregate):
            b.counts[:] = count; adapter.last_timing = timing
        return total, wall

    def metadata(self):
        self._check_open(allow_poison=True)
        return {**self._metadata, 'name': 'metal-batch', 'abi_version': ABI_VERSION,
                'numerical_parent': NUMERICAL_PARENT, 'numerical_order': NUMERICAL_ORDER,
                'lane_count': len(self.brains), **self._memory}

    def _materialize_lane(self, lane, reason):
        self._validate_bindings()
        adapter = self.adapters[lane]
        if adapter._host_weight_epoch != adapter._device_weight_epoch: raise BackendError('Metal lane weight epochs disagree')
        state = self._state(lane)
        self._error(self.library.df_metal_download_lane_state(self.handle, lane, C.byref(state)), poison=True)
        self.brains[lane].cursor = int(state.cursor)
        adapter._host_state_valid = True; adapter._last_materialization_reason = reason

    def _release_lane(self, lane):
        self.brains[lane].backend = self._original[lane]
        self.brains[lane]._metal_batch_owner = None
        self._released.add(lane)

    def _unfreeze(self):
        frozen, self._frozen = self._frozen, []
        error = None
        for array, writeable in frozen:
            try: array.flags.writeable = writeable
            except Exception as failure:
                if error is None: error = failure
        if error is not None: raise error

    def _destroy(self):
        if self.library is not None and self.handle.value:
            handle, self.handle = self.handle, C.c_void_p()
            self.library.df_metal_destroy(handle)

    def close(self):
        if self._thread == threading.get_ident(): raise BackendError('Reentrant close during in-flight Metal batch work')
        with self._lock:
            if self.closed: return
            self._thread = threading.get_ident()
            error = None
            try:
                try:
                    if self.adapters:
                        self._validate_bindings()
                        if not self.poisoned:
                            for lane in range(len(self.brains)): self._materialize_lane(lane, 'close')
                except Exception as failure: error = failure
                self.closed = True
                for cleanup in (self._destroy, self._unfreeze):
                    try: cleanup()
                    except Exception as failure:
                        if error is None: error = failure
                        else: error.add_note(f'Additional Metal cleanup failure: {failure!r}')
                if error is not None:
                    self.poisoned = True; self._restored_lanes.clear()
                    raise error
                if not self.poisoned:
                    for lane in range(len(self.brains)): self._release_lane(lane)
            finally:
                self._thread = None

    def __enter__(self): self._check_open(); return self
    def __exit__(self, *exc): self.close()
    def __del__(self):
        if not getattr(self, 'handle', C.c_void_p()).value: return
        try: self.close()
        except Exception as error:
            warnings.warn(f'Metal batch cleanup failed: {error!r}', RuntimeWarning, stacklevel=2)
