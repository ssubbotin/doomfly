"""Owner validation is portable; trajectory/lifetime comparisons use real Metal."""
import sys
import threading
import ctypes as C

import numpy as np
import pytest

from doom_learning_v6.backend import BackendError
from test_doom_learning_v6 import brain


def executor_type():
    from doom_learning_v6.metal.batch import MetalBatchExecutor
    return MetalBatchExecutor


def assert_initialized_brain_rejected(b):
    with pytest.raises(ValueError, match='[Ii]nitialized'): executor_type()([b])


def assert_reentrant_calls_rejected(executor):
    with pytest.raises(BackendError, match='[Rr]eentrant|in-flight'): executor.advance(1)
    with pytest.raises(BackendError, match='[Rr]eentrant|in-flight'): executor.close()


def test_initialized_handle_guard_rejects_before_any_native_call(tmp_path):
    b = brain(tmp_path, backend='metal')
    original = b.backend.handle
    # A nonnull handle flag exercises only the pre-native ownership rejection.
    # Restore it before cleanup; this sentinel never reaches any native API.
    b.backend.handle = C.c_void_p(1)
    try: assert_initialized_brain_rejected(b)
    finally: b.backend.handle = original


def test_reentrant_guard_rejects_before_any_native_call():
    executor = executor_type().__new__(executor_type())
    executor._lock = threading.Lock(); executor._thread = None
    executor.closed = False; executor.poisoned = False
    with executor._operation(): assert_reentrant_calls_rejected(executor)


def snapshot(b):
    return (b.cursor, b.sim_ms, b.total_spikes,
            {k: getattr(b, k).tobytes() for k in ['weight', *b.fields]})


def assert_equal(a, e):
    a.backend.materialize('batch-test')
    e.backend.materialize('batch-test')
    assert snapshot(a) == snapshot(e)


def test_owner_rejects_empty_duplicate_and_nonbrain_lanes_before_loading_native(tmp_path):
    executor = executor_type()
    b = brain(tmp_path)
    original = b.backend
    for lanes in ([], [b, b], [object()]):
        with pytest.raises((ValueError, TypeError)):
            executor(lanes)
        assert b.backend is original


@pytest.mark.parametrize('field', ['g', 'eligibility', 'memory_u'])
def test_owner_rejects_cross_lane_mutable_aliases_before_binding(tmp_path, field):
    a, b = brain(tmp_path), brain(tmp_path)
    setattr(b, field, getattr(a, field))
    before = [snapshot(x) for x in (a, b)]
    with pytest.raises(ValueError, match='alias'):
        executor_type()([a, b])
    assert [snapshot(x) for x in (a, b)] == before


@pytest.mark.parametrize('change', ['graph', 'configuration', 'fixed-weight', 'plastic',
                                  'nan', 'dtype', 'history', 'queue'])
def test_owner_rejects_incompatible_or_unsafe_lanes_without_ownership(tmp_path, change):
    a, b = brain(tmp_path), brain(tmp_path)
    if change == 'graph': b.post[1] = 3
    if change == 'configuration': b.tonic[0] = 1
    if change == 'fixed-weight': b.weight[1] = 2
    if change == 'plastic': b.circuit['edges'] = np.array([1], dtype=np.int64)
    if change == 'nan': b.memory_u[0] = np.nan
    if change == 'dtype': b.drive = b.drive.astype(np.float64)
    if change == 'history': b.last[0] = -(2**31)
    if change == 'queue': b.queue_count[0] = b.n + 1
    original = [a.backend, b.backend]
    with pytest.raises(ValueError):
        executor_type()([a, b])
    assert [a.backend, b.backend] == original


def test_attached_direct_step_and_neural_step_fail_before_input_changes(tmp_path):
    b = brain(tmp_path)
    b._metal_batch_owner = object()
    before = snapshot(b)
    for call in (b.step, b._neural_step):
        with pytest.raises(BackendError, match='owned'):
            call([], 10, stimulation=([0], 20), lamina_bias=1)
        assert snapshot(b) == before


@pytest.mark.parametrize('operation', ['reset', 'restore', 'checkpoint'])
def test_owned_state_operations_reject_before_host_mutation_or_checkpoint_write(tmp_path, operation):
    b = brain(tmp_path)
    source, destination = tmp_path / 'source.npz', tmp_path / 'destination.npz'
    b.checkpoint(source)
    b.memory_u[0] = -.2
    class BusyOwner:
        def _operation(self, **kwargs): raise BackendError('Metal batch operation is in-flight')
    b._metal_batch_owner = BusyOwner()
    before = snapshot(b)
    with pytest.raises(BackendError, match='in-flight'):
        if operation == 'reset': b.reset()
        elif operation == 'restore': b.restore(source)
        else: b.checkpoint(destination)
    assert snapshot(b) == before and not destination.exists()


def test_cursor_validation_reserves_native_delay_and_rejects_unregistered_clock(tmp_path):
    b = brain(tmp_path)
    owner = executor_type().__new__(executor_type())
    owner.brains = (b,)
    b.cursor = np.iinfo(np.int64).max - 1
    owner._device_cursors = [b.cursor]
    with pytest.raises(ValueError, match='cursor'): owner._validate_cursors(1)
    b.cursor = 1
    owner._device_cursors = [0]
    with pytest.raises(ValueError, match='cursor'): owner._validate_cursors(1)


def test_graph_validation_accepts_rectangular_DAN_by_plastic_edge_gain(tmp_path):
    from doom_learning_v6.metal.batch import _validate_graph
    b = brain(tmp_path)
    b.circuit['edges'] = np.array([0, 1], dtype=np.int64)
    b.circuit['pre'] = np.array([0, 2], dtype=np.int32)
    b.circuit['gain'] = np.array([[1., .5]], dtype=np.float32)
    b.baseline_plastic = b.weight.copy()
    _validate_graph(b)
    b.circuit['gain'] = b.circuit['gain'].T.copy()
    with pytest.raises(ValueError, match='gain'): _validate_graph(b)


@pytest.mark.parametrize('combination', ['pulse', 'bias', 'rgb'])
def test_finite_overflow_preflight_leaves_all_real_lane_inputs_and_state_unchanged(tmp_path, combination):
    rgb = combination == 'rgb'
    if rgb:
        from test_doom_metal_batch_rgb import visual_brain, frame
        lanes = [visual_brain(tmp_path), visual_brain(tmp_path)]
        inputs = [frame(0), frame(1)]
    else:
        lanes = [brain(tmp_path), brain(tmp_path)]
        inputs = [[], []]
    for b in lanes: b.lamina = np.array([1], dtype=np.int32)
    huge = np.float32(3e38)
    bias = float(huge) if combination == 'bias' else 0.
    pulses = [([0], 7.), ([1], huge)] if combination == 'bias' else [([0], 7.), [([0], huge), ([0], huge)]]
    owner = executor_type().__new__(executor_type())
    owner.brains = tuple(lanes)
    before = [snapshot(b) for b in lanes]
    original_inputs = [np.asarray(value).copy() for value in inputs]
    with pytest.raises(ValueError, match='finite|overflow'):
        owner._preflight_inputs(inputs, 286, pulses, lamina_bias=bias, rgb=rgb)
    assert [snapshot(b) for b in lanes] == before
    for a, e in zip(inputs, original_inputs): np.testing.assert_array_equal(a, e)


mac = pytest.mark.skipif(sys.platform != 'darwin', reason='Real Metal requires macOS')


@mac
@pytest.mark.parametrize('flags', [True, [True, False]])
def test_training_and_frozen_lanes_are_bitwise_serial_for_every_bin(tmp_path, flags):
    serial = [brain(tmp_path, backend='metal') for _ in range(2)]
    batched = [brain(tmp_path) for _ in range(2)]
    serial[1].weights_frozen = batched[1].weights_frozen = True
    try:
        with executor_type()(batched) as executor:
            for b in serial: b.backend.start_diagnostics()
            for b in batched: b.backend.start_diagnostics()
            for duration in (10., 28.6, 28.5, 10., 100.):
                pulses = [([0, 2], 20.), ([0, 2], 14.)]
                enabled = [flags] * 2 if isinstance(flags, bool) else flags
                expected = [b.step([], duration, learning=f, stimulation=p, lamina_bias=0)[0]
                            for b, p, f in zip(serial, pulses, enabled)]
                actual, seconds = executor.step([[], []], duration, learning=flags,
                                                stimulations=pulses, lamina_bias=0)
                assert seconds >= 0
                for a, e, ca, ce in zip(batched, serial, actual, expected):
                    np.testing.assert_array_equal(ca, ce)
                    assert_equal(a, e)
                    assert a.backend.spike_events == e.backend.spike_events
                bins = (round(duration / batched[0].dt) + 99) // 100
                assert executor.last_timing['encoder_count'] == bins
                assert executor.last_timing['dispatch_count'] == 2 * round(duration / .1) + 2 * bins
                assert executor.last_timing['full_upload_bytes'] == 0
                assert executor.last_timing['materialize_bytes'] == 0
                assert executor.last_timing['sparse_weight_update_bytes'] == bins * 12
                assert sum(b.last_rule_seconds for b in batched) == executor.last_timing['host_rule_seconds']
                assert all('gpu_seconds' not in b.backend.last_timing for b in batched)
            actual[0][:] = -1
            assert not np.any(batched[0].counts < 0)
        assert all(b.backend.name == 'cpu' for b in batched)
    finally:
        for b in serial: b.backend.close()


@mac
def test_all_inputs_and_registered_buffers_are_checked_before_any_lane_change(tmp_path):
    lanes = [brain(tmp_path), brain(tmp_path)]
    with executor_type()(lanes) as executor:
        before = [snapshot(b) for b in lanes]
        invalid = [dict(luminances=[[], [1]]), dict(luminances=[[], []], learning=[True]),
                   dict(luminances=[[], []], learning=[True, 1]),
                   dict(luminances=[[], []], stimulations=[([0], 20), ([4], 20)]),
                   dict(luminances=[[], []], lamina_bias=np.nan)]
        for kwargs in invalid:
            with pytest.raises(ValueError): executor.step(duration_ms=28.6, **kwargs)
            assert [snapshot(b) for b in lanes] == before
        for steps in (0, 101, True, 1.5):
            with pytest.raises(ValueError): executor.advance(steps)
            assert [snapshot(b) for b in lanes] == before
        lanes[1].eligibility[0] = np.nan
        poisoned_input = [snapshot(b) for b in lanes]
        with pytest.raises(ValueError): executor.step([[], []], 10)
        assert [snapshot(b) for b in lanes] == poisoned_input
        lanes[1].eligibility[0] = 0
        original = lanes[1].drive
        lanes[1].drive = original.copy()
        with pytest.raises(ValueError, match='binding'): executor.advance(1)
        lanes[1].drive = original
        original = lanes[1].tonic
        lanes[1].tonic = original.copy()
        with pytest.raises(ValueError, match='binding'): executor.step([[], []], 10)
        lanes[1].tonic = original
        for b in lanes:
            with pytest.raises(BackendError, match='owned'): b.step([], 10)
        assert [snapshot(b) for b in lanes] == before


@mac
@pytest.mark.parametrize('rgb', [False, True])
def test_public_overflow_rejection_is_atomic_and_owner_remains_healthy(tmp_path, rgb):
    if rgb:
        from test_doom_metal_batch_rgb import visual_brain, frame
        lanes = [visual_brain(tmp_path), visual_brain(tmp_path)]
        inputs = [frame(0), frame(1)]
    else:
        lanes = [brain(tmp_path), brain(tmp_path)]
        inputs = [[], []]
    with executor_type()(lanes) as executor:
        before = [snapshot(b) for b in lanes]
        huge = np.float32(3e38)
        operation = executor.rgb_step if rgb else executor.step
        with pytest.raises(ValueError, match='finite|overflow'):
            operation(inputs, 28.6, stimulations=[([0], 7.), [([0], huge), ([0], huge)]])
        assert [snapshot(b) for b in lanes] == before and not executor.poisoned
        operation(inputs, 10, stimulations=[([0], 7.), ([2], 12.)])


@mac
def test_checkpoint_lane_restore_cursor_alignment_and_idempotent_close(tmp_path):
    lanes = [brain(tmp_path), brain(tmp_path)]
    originals = [b.backend for b in lanes]
    executor = executor_type()(lanes)
    adapters = [b.backend for b in lanes]
    executor.step([[], []], 28.6, stimulations=[([0], 20), ([2], 12)])
    expected = [snapshot(b) for b in lanes]
    for i, b in enumerate(lanes):
        b.checkpoint(tmp_path / f'lane-{i}.npz')
        with np.load(tmp_path / f'lane-{i}.npz') as data:
            import json
            metadata = json.loads(str(data['metadata']))
            assert metadata['producer_backend'] == 'metal-batch'
            assert metadata['backend']['lane_count'] == 2
            assert metadata['backend']['lane_index'] == i
            assert metadata['backend']['numerical_order'] == 'epoch-5-zero-seeded-ascending-incoming'
    unchanged = snapshot(lanes[1])
    lanes[0].reset()
    assert snapshot(lanes[1]) == unchanged
    before = [snapshot(b) for b in lanes]
    with pytest.raises(ValueError, match='cursor'): executor.step([[], []], 10)
    assert [snapshot(b) for b in lanes] == before
    lanes[0].restore(tmp_path / 'lane-0.npz')
    executor.advance(1)
    executor.close()
    after = [snapshot(b) for b in lanes]
    executor.close()
    assert [b.backend for b in lanes] == originals
    assert [snapshot(b) for b in lanes] == after
    for operation in (lambda: executor.advance(1), lambda: executor.step([[], []], 10),
                      lambda: adapters[0].materialize('closed')):
        with pytest.raises(BackendError, match='closed'): operation()
    assert expected[0][0] == 286


@mac
def test_initialized_or_already_owned_brains_are_rejected(tmp_path):
    b = brain(tmp_path, backend='metal')
    b.backend.ensure_initialized()
    try:
        assert_initialized_brain_rejected(b)
    finally: b.backend.close()
    b = brain(tmp_path)
    with executor_type()([b]):
        with pytest.raises(ValueError, match='owned'): executor_type()([b])


@mac
def test_reentrant_execution_rejects_and_close_waits_for_inflight_work(tmp_path, monkeypatch):
    b = brain(tmp_path)
    b.memory_u[0] = -.1
    executor = executor_type()([b])
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    ready = threading.Event()
    original = b._prepare_neural_input
    errors = []
    def prepare(*args, **kwargs):
        assert_reentrant_calls_rejected(executor)
        before = snapshot(b)
        for operation in (b.reset, lambda: b.restore(tmp_path / 'missing.npz'),
                          lambda: b.checkpoint(tmp_path / 'reentrant.npz')):
            with pytest.raises(BackendError, match='[Rr]eentrant'): operation()
            assert snapshot(b) == before
        entered.set()
        ready.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(b, '_prepare_neural_input', prepare)
    def run():
        try: executor.step([[]], 10)
        except BaseException as error: errors.append(error)
        finally: ready.set()
    def close():
        try: executor.close()
        except BaseException as error: errors.append(error)
        finally: finished.set()
    worker = threading.Thread(target=run)
    worker.start()
    closer = None
    try:
        assert ready.wait(5), 'Worker did not enter preparation or finish'
        if errors: raise errors[0]
        assert entered.is_set(), 'Worker exited before preparation checkpoint'
        before = snapshot(b)
        with pytest.raises(BackendError, match='in-flight'): b.reset()
        with pytest.raises(BackendError, match='in-flight'): b.restore(tmp_path / 'missing.npz')
        with pytest.raises(BackendError, match='in-flight'): b.checkpoint(tmp_path / 'concurrent.npz')
        assert snapshot(b) == before
        assert not (tmp_path / 'concurrent.npz').exists() and not (tmp_path / 'reentrant.npz').exists()
        with pytest.raises(BackendError, match='in-flight'): executor.advance(1)
        closer = threading.Thread(target=close)
        closer.start()
        assert not finished.wait(.05), errors
        assert b.backend.name == 'metal-batch'
        release.set()
        worker.join(5); closer.join(5)
        if errors: raise errors[0]
        assert finished.is_set() and not worker.is_alive() and not closer.is_alive()
        assert b.backend.name == 'cpu' and b.cursor == 100
    finally:
        release.set()
        worker.join(5)
        if closer is not None: closer.join(5)
        executor.close()


@mac
@pytest.mark.parametrize('close_first', [False, True])
def test_genuine_native_event_overflow_requires_all_lanes_explicitly_restored(tmp_path, monkeypatch, close_first):
    lanes = [brain(tmp_path), brain(tmp_path)]
    executor = executor_type()(lanes)
    for b in lanes:
        b.drive[[0, 2, 3]] = 30
        b.backend.start_diagnostics()
    native_advance = executor.library.df_metal_advance
    # Restrict the actual native command's output capacity to provoke overflow.
    # The kernel, status and poisoned state remain genuine native operations.
    def overflow(handle, steps, events, capacity, count, timing):
        return native_advance(handle, steps, events, 1, count, timing)
    monkeypatch.setattr(executor.library, 'df_metal_advance', overflow)
    try:
        with pytest.raises(BackendError, match='capacity'): executor.advance(100)
        assert executor.poisoned
        for b in lanes:
            with pytest.raises(BackendError, match='owned'): b.step([], 10)
        with pytest.raises(BackendError, match='poison'): executor.advance(1)
        monkeypatch.setattr(executor.library, 'df_metal_advance', native_advance)
        if close_first:
            executor.close()
            assert all(b.backend.name == 'metal-batch' for b in lanes)
        lanes[0].reset()
        if close_first:
            assert lanes[0].backend.name == 'cpu'
            assert lanes[1].backend.name == 'metal-batch'
        else:
            with pytest.raises(BackendError, match='poison'): executor.advance(1)
        lanes[1].reset()
        if close_first:
            assert all(b.backend.name == 'cpu' for b in lanes)
        else:
            assert not executor.poisoned
            executor.advance(1)
    finally: executor.close()
