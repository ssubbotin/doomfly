"""Checked resident efficacy installation, using real small brain/owner bindings."""
import ctypes as C
import sys
import threading

import numpy as np
import pytest

from doom_learning_v6.backend import BackendError
from doom_learning_v6.brain import MemoryBrain
from doom_learning_v6.causal_controls import initialize_efficacies
from doom_learning_v6.visual import VisualMemoryBrain
from test_doom_metal_batch import executor_type, guarded_portable_owner, snapshot


def candidate_brain(tmp_path, visual=False):
    path = tmp_path / 'candidate-graph.npz'
    np.savez(path, ptr=np.array([0, 2, 2, 3, 3], dtype=np.int64),
        post=np.array([1, 1, 1], dtype=np.int32),
        weight=np.array([20., .275, 10.], dtype=np.float32),
        ids=np.arange(4, dtype=np.int64), retina=np.empty(0, dtype=np.int32),
        uv=np.empty((0, 2), dtype=np.float32), lamina=np.empty(0, dtype=np.int32),
        sugar=np.empty(0, dtype=np.int32), superclass=np.array(['test'] * 4))
    circuit = {'edges': np.array([0, 2], dtype=np.int64),
        'pre': np.array([0, 2], dtype=np.int32),
        'kc_mask': np.array([1, 0, 1, 0], dtype=np.uint8),
        'dan_index': np.array([-1, -1, -1, 0], dtype=np.int8),
        'gain': np.array([[1., 1.]], dtype=np.float32),
        'kc': np.array([0, 2]), 'mb': np.array([1]), 'dan': np.array([3])}
    b = MemoryBrain(path, circuit=circuit, modulation_mask=np.array([0, 0, 0, 1]))
    if visual:
        # The established small VisualMemoryBrain fixture pattern, without data files.
        b.__class__ = VisualMemoryBrain
        b.r8 = np.array([0, 2], dtype=np.int32)
        b.r8_uv = np.array([[0, 0], [1, 1]], dtype=np.float32)
        b.r8_channel = np.array([2, 1], dtype=np.int32)
        b.r8_light = np.zeros(2, dtype=np.float32)
        b.corrected_edges = np.empty(0, dtype=np.int64)
        b.fields.append('r8_light'); b.initial['r8_light'] = b.r8_light.copy()
    return b


@pytest.fixture(params=[False, True], ids=['memory', 'visual'])
def resident(tmp_path, monkeypatch, request):
    lanes = [candidate_brain(tmp_path, request.param) for _ in range(2)]
    with guarded_portable_owner(lanes, monkeypatch) as (owner, calls):
        device = [b.weight.copy() for b in lanes]

        def update(handle, lane, count, ids_pointer, values_pointer):
            # Substitute only the raw native sparse copy. Real adapter guards,
            # validation, epochs, and timing run above this boundary.
            assert handle.value == 1 and owner._thread is not None
            assert owner._lock.locked()
            ids = np.ctypeslib.as_array(C.cast(ids_pointer, C.POINTER(C.c_int64)),
                                       shape=(count,)).copy()
            values = np.ctypeslib.as_array(C.cast(values_pointer, C.POINTER(C.c_float)),
                                          shape=(count,)).copy()
            np.testing.assert_array_equal(ids, [0, 2])
            assert values.dtype == np.float32 and np.isfinite(values).all()
            calls.append(('sparse', lane, ids, values))
            device[lane][ids] = values
            return 0

        owner.library.df_metal_update_lane_weights = update
        owner.library.df_metal_last_error = lambda: b'Deliberate sparse failure'
        yield lanes, owner, calls, device


@pytest.mark.parametrize('stale', [False, True])
def test_install_is_baseline_relative_sparse_and_preserves_neural_state(resident, stale):
    lanes, owner, calls, device = resident
    b = lanes[1]
    b.backend._host_state_valid = not stale
    b.weights_frozen = True
    before = [snapshot(lane) for lane in lanes]
    fixed = b.weight[1:2].tobytes()
    owner.install_efficacies(np.int64(1), np.array([.9, 1.1]), np.array([0, 2], dtype=np.int64))
    np.testing.assert_allclose(b.memory_u, [-.1, .1], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(b.memory_w, b.memory_u)
    np.testing.assert_array_equal(b.weight, np.array([18., .275, 11.], dtype=np.float32))
    np.testing.assert_array_equal(device[1], b.weight)
    assert b.weight[1:2].tobytes() == fixed and snapshot(lanes[0]) == before[0]
    for name in b.fields:
        if name not in ('memory_u', 'memory_w'):
            assert getattr(b, name).tobytes() == before[1][3][name]
    assert (b.cursor, b.sim_ms, b.total_spikes) == before[1][:3]
    assert b.weights_frozen and b.backend._host_state_valid is (not stale)
    assert b.backend._host_weight_epoch == b.backend._device_weight_epoch == 1
    assert len(calls) == 1 and calls[0][0:2] == ('sparse', 1)
    assert owner.last_timing['sparse_weight_update_bytes'] == 24
    owner.install_efficacies(1, np.ones(2), b.circuit['edges'].copy())
    np.testing.assert_array_equal(b.memory_u, [0., 0.])
    np.testing.assert_array_equal(b.memory_w, [0., 0.])
    np.testing.assert_array_equal(b.weight[[0, 2]], [20., 10.])
    np.testing.assert_array_equal(device[1], b.weight)
    assert b.backend._host_state_valid is (not stale) and len(calls) == 2


BAD_ARGUMENTS = [
    ('bool-lane', True, np.ones(2), np.array([0, 2], dtype=np.int64)),
    ('numpy-bool-lane', np.bool_(False), np.ones(2), np.array([0, 2], dtype=np.int64)),
    ('float-lane', 0., np.ones(2), np.array([0, 2], dtype=np.int64)),
    ('negative-lane', -1, np.ones(2), np.array([0, 2], dtype=np.int64)),
    ('outside-lane', 2, np.ones(2), np.array([0, 2], dtype=np.int64)),
    ('nan', 0, np.array([np.nan, 1.]), np.array([0, 2], dtype=np.int64)),
    ('inf', 0, np.array([1., np.inf]), np.array([0, 2], dtype=np.int64)),
    ('low', 0, np.array([.099, 1.]), np.array([0, 2], dtype=np.int64)),
    ('high', 0, np.array([1., 2.001]), np.array([0, 2], dtype=np.int64)),
    ('fraction-shape', 0, np.ones((1, 2)), np.array([0, 2], dtype=np.int64)),
    ('fraction-length', 0, np.ones(1), np.array([0, 2], dtype=np.int64)),
    ('fraction-f32', 0, np.ones(2, dtype=np.float32), np.array([0, 2], dtype=np.int64)),
    ('fraction-list', 0, [1., 1.], np.array([0, 2], dtype=np.int64)),
    ('edge-list', 0, np.ones(2), [0, 2]),
    ('edge-f64', 0, np.ones(2), np.array([0., 2.])),
    ('edge-i32', 0, np.ones(2), np.array([0, 2], dtype=np.int32)),
    ('edge-shape', 0, np.ones(2), np.array([[0, 2]], dtype=np.int64)),
    ('edge-short', 0, np.ones(2), np.array([0], dtype=np.int64)),
    ('edge-duplicate', 0, np.ones(2), np.array([0, 0], dtype=np.int64)),
    ('edge-reordered', 0, np.ones(2), np.array([2, 0], dtype=np.int64)),
    ('edge-ineligible', 0, np.ones(2), np.array([0, 1], dtype=np.int64)),
    ('edge-negative', 0, np.ones(2), np.array([-1, 2], dtype=np.int64)),
    ('edge-outside', 0, np.ones(2), np.array([0, 3], dtype=np.int64)),
]


@pytest.mark.parametrize('_name,lane,fractions,edges', BAD_ARGUMENTS,
                         ids=[case[0] for case in BAD_ARGUMENTS])
def test_invalid_arguments_are_rejected_before_host_or_device_mutation(resident, _name, lane, fractions, edges):
    lanes, owner, calls, device = resident
    before = [snapshot(b) for b in lanes]
    with pytest.raises(ValueError): owner.install_efficacies(lane, fractions, edges)
    assert [snapshot(b) for b in lanes] == before and calls == []
    for actual, b in zip(device, lanes): np.testing.assert_array_equal(actual, b.weight)
    assert not owner.poisoned


@pytest.mark.parametrize('problem', ['owner', 'backend', 'memory-shape', 'memory-reallocated',
    'weight-readonly', 'baseline-writeable', 'baseline-nan', 'baseline-inf',
    'baseline-zero', 'baseline-negative', 'overflow', 'underflow',
    'edge-duplicate', 'edge-outside', 'memory-nan', 'memory-inf'])
def test_invalid_registered_state_is_rejected_without_mutation(resident, problem):
    lanes, owner, calls, _ = resident
    b = lanes[0]
    fractions = np.ones(2)
    if problem == 'owner': b._metal_batch_owner = object()
    elif problem == 'backend': b.backend = owner._original[0]
    elif problem == 'memory-shape': b.memory_u.shape = (1, 2)
    elif problem == 'memory-reallocated': b.memory_w = b.memory_w.copy()
    elif problem == 'weight-readonly': b.weight.flags.writeable = False
    elif problem.startswith('baseline-') or problem in ('overflow', 'underflow'):
        b.baseline_plastic.flags.writeable = True
        if problem != 'baseline-writeable':
            value = {'baseline-nan': np.nan, 'baseline-inf': np.inf,
                'baseline-zero': 0., 'baseline-negative': -1.,
                'overflow': np.finfo(np.float32).max,
                'underflow': np.nextafter(np.float32(0), np.float32(1))}[problem]
            b.baseline_plastic[0] = value
            b.baseline_plastic.flags.writeable = False
            if problem == 'overflow': fractions[0] = 2.
            if problem == 'underflow': fractions[0] = .1
    elif problem.startswith('edge-'):
        b.circuit['edges'].flags.writeable = True
        b.circuit['edges'][1] = 0 if problem == 'edge-duplicate' else 3
        b.circuit['edges'].flags.writeable = False
    elif problem == 'memory-nan': b.memory_u[0] = np.nan
    elif problem == 'memory-inf': b.memory_w[1] = np.inf
    before = [snapshot(lane) for lane in lanes]
    with pytest.raises(ValueError): owner.install_efficacies(0, fractions, b.circuit['edges'].copy())
    assert [snapshot(lane) for lane in lanes] == before and calls == []
    assert not owner.poisoned


@pytest.mark.parametrize('bounds', [(np.nan, 2.), (.1, np.inf), (0., 2.), (-.1, 2.),
    (2., .1), (1., 1.), (True, 2.), (None, 2.), ('0.1', 2.), (.1, 10 ** 1000)])
def test_malformed_registered_bounds_reject_before_mutation(tmp_path, monkeypatch, bounds):
    b = candidate_brain(tmp_path)
    b.rule_parameters.update(minimum_fraction=bounds[0], maximum_fraction=bounds[1])
    with guarded_portable_owner([b], monkeypatch) as (owner, calls):
        before = snapshot(b)
        with pytest.raises(ValueError): owner.install_efficacies(0, np.ones(2), b.circuit['edges'].copy())
        assert snapshot(b) == before and calls == [] and not owner.poisoned


def test_missing_registered_bound_rejects_before_mutation(tmp_path, monkeypatch):
    b = candidate_brain(tmp_path)
    del b.rule_parameters['minimum_fraction']
    with guarded_portable_owner([b], monkeypatch) as (owner, calls):
        before = snapshot(b)
        with pytest.raises(ValueError): owner.install_efficacies(0, np.ones(2), b.circuit['edges'].copy())
        assert snapshot(b) == before and calls == [] and not owner.poisoned


@pytest.mark.parametrize('state', ['closed', 'poisoned', 'inflight', 'reentrant', 'configuration', 'epochs'])
def test_executor_guards_precede_installation(resident, state):
    lanes, owner, calls, _ = resident
    b = lanes[0]
    if state == 'closed': owner.closed = True
    elif state == 'poisoned': owner.poisoned = True
    elif state == 'inflight': owner._lock.acquire()
    elif state == 'reentrant': owner._thread = threading.get_ident()
    elif state == 'configuration': b.rule_parameters['minimum_fraction'] = .2
    elif state == 'epochs': b.backend._host_weight_epoch += 1
    before = [snapshot(lane) for lane in lanes]
    try:
        with pytest.raises(ValueError if state == 'configuration' else BackendError):
            owner.install_efficacies(0, np.ones(2), b.circuit['edges'].copy())
        assert [snapshot(lane) for lane in lanes] == before and calls == []
    finally:
        if state == 'inflight': owner._lock.release()
        if state == 'reentrant': owner._thread = None


@pytest.mark.parametrize('failure', ['status', 'exception'])
def test_native_failure_poison_preserves_primary_and_requires_explicit_reset(resident, failure):
    lanes, owner, calls, device = resident
    b = lanes[0]
    b.backend._host_state_valid = False
    primary = BackendError('Primary native sparse exception')
    def fail(*args):
        if failure == 'exception': raise primary
        return 1
    owner.library.df_metal_update_lane_weights = fail
    with pytest.raises(BackendError) as caught:
        owner.install_efficacies(0, np.array([.9, 1.1]), b.circuit['edges'].copy())
    if failure == 'exception': assert caught.value is primary
    else: assert str(caught.value) == 'Deliberate sparse failure'
    assert owner.poisoned and owner._restored_lanes == set()
    assert not b.backend._host_state_valid
    np.testing.assert_array_equal(b.weight, np.array([18., .275, 11.], dtype=np.float32))
    np.testing.assert_array_equal(device[0], np.array([20., .275, 10.], dtype=np.float32))
    with pytest.raises(BackendError): owner.install_efficacies(0, np.ones(2), b.circuit['edges'].copy())
    with pytest.raises(BackendError): b.backend.update_weights(b.circuit['edges'], b.weight[[0, 2]])
    assert calls == []
    owner.library.df_metal_upload_lane_state = lambda *args: 0
    b.reset()
    assert owner.poisoned  # Every resident lane must be explicitly restored.
    lanes[1].reset()
    assert not owner.poisoned and b.backend._host_state_valid
    np.testing.assert_array_equal(b.weight[[0, 2]], [20., 10.])
    np.testing.assert_array_equal(b.memory_u, [0., 0.])
    np.testing.assert_array_equal(b.memory_w, [0., 0.])


def test_existing_initializer_still_rejects_owned_brain(resident):
    lanes, _, calls, _ = resident
    before = snapshot(lanes[0])
    with pytest.raises(ValueError, match='unowned'): initialize_efficacies(lanes[0], np.zeros(2))
    assert snapshot(lanes[0]) == before and calls == []


@pytest.mark.skipif(sys.platform != 'darwin', reason='Real Metal requires macOS')
@pytest.mark.parametrize('visual', [False, True])
def test_native_install_matches_device_materialization_and_baseline_round_trip(tmp_path, visual):
    lanes = [candidate_brain(tmp_path, visual) for _ in range(2)]
    with executor_type()(lanes) as owner:
        owner.advance(1)
        b = lanes[1]
        assert not b.backend._host_state_valid
        cursor = b.cursor
        owner.install_efficacies(1, np.array([.9, 1.1]), b.circuit['edges'].copy())
        assert not b.backend._host_state_valid and b.cursor == cursor
        b.backend.materialize('candidate-install-test')
        np.testing.assert_array_equal(b.weight, np.array([18., .275, 11.], dtype=np.float32))
        owner.install_efficacies(1, np.ones(2), b.circuit['edges'].copy())
        b.backend.materialize('candidate-baseline-test')
        np.testing.assert_array_equal(b.weight, np.array([20., .275, 10.], dtype=np.float32))
        np.testing.assert_array_equal(b.memory_u, [0., 0.])
        np.testing.assert_array_equal(b.memory_w, [0., 0.])
        assert b.cursor == cursor and not owner.poisoned
