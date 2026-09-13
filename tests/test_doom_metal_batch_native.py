"""Real native controls for independent resident trajectories.

These tests catch missing lane offsets, changed epoch-5 arithmetic, cross-lane
events and mutation on rejected calls. No timing or GPU operations are mocked.
"""
import ctypes as C
import sys

import numpy as np
import pytest

from test_doom_learning_v6 import brain
from test_doom_metal_parity import paired_brains
from test_doom_metal_state import metal_library, raw_empty_graph, raw_empty_state
from doom_learning_v6.metal.backend import Graph, State, KCEvent, Timing, _pointer
from doom_learning_v6.metal.build import DEFAULT_OUTPUT


pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='Metal requires macOS')
STEPS = (1, 40, 80, 100, 100, 100)
STATE_NAMES = ('weight', 'v', 'g', 'refractory', 'drive', 'previous_drive', 'queue',
               'queue_count', 'counts', 'active', 'active_flag', 'nactive', 'last',
               'modulation', 'modulation_last', 'rest', 'adaptation')


def batch_library():
    library = metal_library()
    assert hasattr(library, 'df_metal_create_batch'), 'Healthy native library lacks df_metal_create_batch lane API'
    signatures = {
        'df_metal_create_batch': [C.POINTER(Graph), C.c_char_p, C.c_int32, C.POINTER(C.c_void_p)],
        'df_metal_upload_lane_state': [C.c_void_p, C.c_int32, C.POINTER(State)],
        'df_metal_download_lane_state': [C.c_void_p, C.c_int32, C.POINTER(State)],
        'df_metal_upload_lane_drive': [C.c_void_p, C.c_int32, C.c_void_p],
        'df_metal_download_lane_observation': [C.c_void_p, C.c_int32, C.c_void_p, C.POINTER(C.c_int64)],
        'df_metal_update_lane_weights': [C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p],
        'df_metal_apply_lane_eligibility': [C.c_void_p, C.c_int32, C.c_void_p, C.c_void_p, C.c_double],
        'df_metal_batch_memory_bytes': [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)],
        'df_metal_advance': [C.c_void_p, C.c_int32, C.POINTER(KCEvent), C.c_int32,
                             C.POINTER(C.c_int32), C.POINTER(Timing)],
        'df_metal_set_diagnostics': [C.c_void_p, C.c_int32],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)  # Healthy old libraries fail here.
        function.argtypes = arguments
        function.restype = C.c_int
    return library


def check(library, status):
    assert status == 0, library.df_metal_last_error().decode()


def model_state(model):
    return State(model.cursor, *[_pointer(getattr(model, name)) for name in STATE_NAMES])


def model_graph(seed):
    seed.backend.ensure_initialized()
    incoming = seed.backend.incoming
    return Graph(seed.n, len(seed.post), seed.queue.shape[0], seed.dt,
                 seed.adaptation_jump, seed.adaptation_tau,
                 _pointer(seed.ptr), _pointer(seed.post), _pointer(incoming.ptr),
                 _pointer(incoming.pre), _pointer(incoming.edge),
                 _pointer(seed.circuit['kc_mask']), _pointer(seed.modulation_mask))


def create(library, graph, lanes):
    handle = C.c_void_p()
    check(library, library.df_metal_create_batch(C.byref(graph),
          str(DEFAULT_OUTPUT / 'kernels.metallib').encode(), lanes, C.byref(handle)))
    return handle


def advance(library, handle, steps, capacity):
    events = (KCEvent * capacity)()
    count, timing = C.c_int32(), Timing()
    check(library, library.df_metal_advance(handle, steps, events, capacity,
                                           C.byref(count), C.byref(timing)))
    return sorted((e.reserved, e.tick, e.neuron) for e in events[:count.value]), timing


def assert_state(actual, expected):
    for name in ('weight', 'v', 'g', 'adaptation', 'modulation', 'drive',
                 'previous_drive', 'rest'):
        np.testing.assert_array_equal(getattr(actual, name).view(np.uint32),
                                      getattr(expected, name).view(np.uint32), err_msg=name)
    for name in ('refractory', 'last', 'modulation_last', 'active_flag', 'counts',
                 'queue_count', 'nactive'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)
    for slot, count in enumerate(actual.queue_count):
        np.testing.assert_array_equal(np.sort(actual.queue[slot, :count]),
                                      np.sort(expected.queue[slot, :count]))
    count = int(actual.nactive[0])
    np.testing.assert_array_equal(actual.active[:count], expected.active[:count])
    assert actual.cursor == expected.cursor


def run_identity(tmp_path, order=(0, 1), perturb=False, synthetic=False, modulatory_only=False):
    library = batch_library()
    references, hosts = [], []
    handle = C.c_void_p()
    try:
        for index in range(len(order)):
            path = tmp_path / str(index)
            path.mkdir(parents=True)
            if synthetic:
                host, reference = paired_brains(path)
            else:
                reference = brain(path, backend='metal')
                host = brain(path, backend='cpu')
            references.append(reference)
            hosts.append(host)
            slot = int(reference.circuit['edges'][0])
            reference.weight[slot] = host.weight[slot] = (19., 18.)[index]
            source = 3 if synthetic else 2
            drive = np.zeros(reference.n, dtype=np.float32)
            drive[0] = (20., 24.)[index]
            drive[source] = (12., 17.)[index]
            if modulatory_only:
                drive[0] = 0
            if perturb and index == 1:
                drive[source] += 3
                reference.v[1] = host.v[1] = -51
                reference.weight[slot] = host.weight[slot] = 16
            reference.drive[:] = host.drive[:] = drive
            reference.backend.ensure_initialized()
            check(library, library.df_metal_set_diagnostics(reference.backend.handle, 1))
        graph = model_graph(references[0])
        handle = create(library, graph, len(order))
        check(library, library.df_metal_set_diagnostics(handle, 1))
        for lane, index in enumerate(order):
            check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(hosts[index]))))
            check(library, library.df_metal_upload_lane_drive(handle, lane, _pointer(hosts[index].drive)))
        all_events = [[] for _ in order]
        for iteration, steps in enumerate(STEPS):
            if iteration == 2:
                for lane, index in enumerate(order):
                    ids = hosts[index].circuit['edges']
                    values = np.array([17. + index], dtype=np.float32)
                    check(library, library.df_metal_update_lane_weights(handle, lane, 1,
                                                                       _pointer(ids), _pointer(values)))
                    check(library, references[index].backend.library.df_metal_update_weights(
                        references[index].backend.handle, 1, _pointer(ids), _pointer(values)))
            observed, timing = advance(library, handle, steps, graph.neurons * len(order) * 5)
            assert timing.encoder_count == 1
            assert timing.dispatch_count == 2 * steps + 2
            assert timing.mark_grid_threads == timing.gather_grid_threads == steps * graph.neurons * len(order)
            assert timing.indirect_dispatch_count == 0
            assert all(0 <= lane < len(order) for lane, _, _ in observed)
            for lane, index in enumerate(order):
                reference, host = references[index], hosts[index]
                expected, _ = advance(library, reference.backend.handle, steps, graph.neurons * 5)
                lane_events = [(0, tick, neuron) for registered, tick, neuron in observed if registered == lane]
                assert lane_events == expected
                all_events[index].extend(lane_events)
                check(library, library.df_metal_apply_lane_eligibility(handle, lane,
                    _pointer(host.eligibility), _pointer(host.eligibility_last), 50.))
                check(library, reference.backend.library.df_metal_apply_eligibility(reference.backend.handle,
                    _pointer(reference.eligibility), _pointer(reference.eligibility_last), 50.))
                np.testing.assert_array_equal(host.eligibility.view(np.uint64), reference.eligibility.view(np.uint64))
                np.testing.assert_array_equal(host.eligibility_last, reference.eligibility_last)
                state = model_state(host)
                check(library, library.df_metal_download_lane_state(handle, lane, C.byref(state)))
                host.cursor = state.cursor
                reference.backend.materialize('native batch assertion')
                assert_state(host, reference)
                counts, cursor = np.empty(host.n, dtype=np.int32), C.c_int64(-1)
                check(library, library.df_metal_download_lane_observation(handle, lane, _pointer(counts), C.byref(cursor)))
                np.testing.assert_array_equal(counts, host.counts)
                assert cursor.value == host.cursor
        return [({name: getattr(host, name).copy() for name in STATE_NAMES}, all_events[i])
                for i, host in enumerate(hosts)]
    finally:
        if handle.value:
            library.df_metal_destroy(handle)
        for reference in references:
            reference.backend.close()


def test_native_batch_distinct_lanes_match_serial_state_events_and_eligibility(tmp_path):
    run_identity(tmp_path)


def test_native_single_lane_keeps_serial_state_bits(tmp_path):
    run_identity(tmp_path, order=(0,))


def test_native_repeat_reverse_registration_and_neighbor_isolation(tmp_path):
    first = run_identity(tmp_path / 'first')
    for name, arguments in [('repeat', {}), ('reverse', {'order': (1, 0)}),
                            ('perturb', {'perturb': True})]:
        actual = run_identity(tmp_path / name, **arguments)
        indices = (0,) if name == 'perturb' else (0, 1)
        for index in indices:
            assert actual[index][1] == first[index][1]
            for field in STATE_NAMES:
                np.testing.assert_array_equal(actual[index][0][field], first[index][0][field], err_msg=field)


@pytest.mark.parametrize('modulatory_only', [False, True])
def test_native_duplicate_self_edges_and_modulatory_delivery(tmp_path, modulatory_only):
    run_identity(tmp_path, synthetic=True, modulatory_only=modulatory_only)


def test_native_zero_edges_and_allocated_memory_accounting():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handles = []
    try:
        accounting = []
        for lanes in (1, 2):
            handle = create(library, graph, lanes)
            handles.append(handle)
            for lane in range(lanes):
                state, arrays = raw_empty_state()
                check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(state)))
            advance(library, handle, 40, lanes)
            shared, mutable = C.c_uint64(), C.c_uint64()
            check(library, library.df_metal_batch_memory_bytes(handle, C.byref(shared), C.byref(mutable)))
            assert shared.value > 0 and mutable.value > 0
            accounting.append((shared.value, mutable.value))
        assert accounting[0][0] == accounting[1][0]
        assert accounting[1][1] > accounting[0][1]
    finally:
        for handle in handles:
            library.df_metal_destroy(handle)


def test_native_creation_rejects_null_zero_and_excessive_lanes():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    for lanes in (0, -1, 2**31 - 1):
        handle = C.c_void_p()
        assert library.df_metal_create_batch(C.byref(graph),
            str(DEFAULT_OUTPUT / 'kernels.metallib').encode(), lanes, C.byref(handle)) != 0
        assert not handle.value
    handle = C.c_void_p()
    assert library.df_metal_create_batch(None, b'bad', 2, C.byref(handle)) != 0
    assert not handle.value


def test_native_rejected_lane_pointer_cursor_and_capacity_calls_do_not_mutate():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handle = create(library, graph, 2)
    try:
        originals = []
        for lane in range(2):
            state, arrays = raw_empty_state()
            state.cursor = lane
            arrays[7][:] = 7 + lane  # Counts survive all rejected operations.
            check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(state)))
            originals.append((state, arrays, [array.copy() for array in arrays]))
        for lane in (-1, 2):
            state = originals[0][0]
            assert library.df_metal_upload_lane_state(handle, lane, C.byref(state)) != 0
            assert library.df_metal_download_lane_state(handle, lane, C.byref(state)) != 0
            assert library.df_metal_upload_lane_drive(handle, lane, state.drive) != 0
            assert library.df_metal_update_lane_weights(handle, lane, 0, None, None) != 0
            eligibility = np.zeros(1, dtype=np.float64)
            last = np.zeros(1, dtype=np.int64)
            assert library.df_metal_apply_lane_eligibility(handle, lane, _pointer(eligibility), _pointer(last), 50.) != 0
            cursor = C.c_int64()
            assert library.df_metal_download_lane_observation(handle, lane, state.counts, C.byref(cursor)) != 0
        assert library.df_metal_upload_lane_state(handle, 0, None) != 0
        assert library.df_metal_download_lane_state(handle, 0, None) != 0
        assert library.df_metal_upload_lane_drive(handle, 0, None) != 0
        assert library.df_metal_update_lane_weights(handle, 0, 1, None, None) != 0
        assert library.df_metal_apply_lane_eligibility(handle, 0, None, None, 50.) != 0
        cursor = C.c_int64()
        assert library.df_metal_download_lane_observation(handle, 0, None, C.byref(cursor)) != 0
        events, count, timing = (KCEvent * 2)(), C.c_int32(), Timing()
        assert library.df_metal_advance(handle, 40, events, 2, C.byref(count), C.byref(timing)) != 0
        assert library.df_metal_advance(handle, 40, events, -1, C.byref(count), C.byref(timing)) != 0
        for lane, (state, arrays, expected) in enumerate(originals):
            check(library, library.df_metal_download_lane_state(handle, lane, C.byref(state)))
            assert state.cursor == lane
            for actual, wanted in zip(arrays, expected):
                np.testing.assert_array_equal(actual, wanted)
        # Correcting alignment recovers from pre-dispatch validation errors.
        state = originals[1][0]
        state.cursor = 0
        check(library, library.df_metal_upload_lane_state(handle, 1, C.byref(state)))
        advance(library, handle, 1, 2)
    finally:
        library.df_metal_destroy(handle)


def test_native_cursor_overflow_rejected_before_counts_clear():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handle = create(library, graph, 2)
    try:
        for lane in range(2):
            state, arrays = raw_empty_state()
            state.cursor = 2**63 - 2
            arrays[11][:] = state.cursor
            arrays[13][:] = state.cursor
            arrays[7][:] = 9
            check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(state)))
        events, count, timing = (KCEvent * 2)(), C.c_int32(), Timing()
        assert library.df_metal_advance(handle, 100, events, 2, C.byref(count), C.byref(timing)) != 0
        for lane in range(2):
            state, arrays = raw_empty_state()
            check(library, library.df_metal_download_lane_state(handle, lane, C.byref(state)))
            assert state.cursor == 2**63 - 2
            assert arrays[7][0] == 9
    finally:
        library.df_metal_destroy(handle)


def test_native_event_overflow_poisons_whole_owner():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handle = create(library, graph, 2)
    try:
        check(library, library.df_metal_set_diagnostics(handle, 1))
        for lane in range(2):
            state, arrays = raw_empty_state()
            arrays[3][:] = 20
            check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(state)))
        count, timing = C.c_int32(), Timing()
        assert library.df_metal_advance(handle, 100, None, 0, C.byref(count), C.byref(timing)) != 0
        events = (KCEvent * 10)()
        assert library.df_metal_advance(handle, 1, events, 10, C.byref(count), C.byref(timing)) != 0
        assert 'poisoned' in library.df_metal_last_error().decode()
    finally:
        library.df_metal_destroy(handle)


def test_native_shared_rest_mismatch_rejected_without_changing_existing_lane():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handle = create(library, graph, 2)
    try:
        original, original_arrays = raw_empty_state()
        original_arrays[7][:] = 3
        check(library, library.df_metal_upload_lane_state(handle, 0, C.byref(original)))
        candidate, candidate_arrays = raw_empty_state()
        candidate_arrays[14][:] = -51
        assert library.df_metal_upload_lane_state(handle, 1, C.byref(candidate)) != 0
        downloaded, arrays = raw_empty_state()
        check(library, library.df_metal_download_lane_state(handle, 0, C.byref(downloaded)))
        for actual, expected in zip(arrays, original_arrays):
            np.testing.assert_array_equal(actual, expected)
        candidate_arrays[14][:] = -52
        check(library, library.df_metal_upload_lane_state(handle, 1, C.byref(candidate)))
        advance(library, handle, 1, 2)
    finally:
        library.df_metal_destroy(handle)


def test_native_advance_requires_validated_state_in_every_lane():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handle = create(library, graph, 2)
    try:
        state, arrays = raw_empty_state()
        arrays[7][:] = 9
        check(library, library.df_metal_upload_lane_state(handle, 0, C.byref(state)))
        events, count, timing = (KCEvent * 2)(), C.c_int32(), Timing()
        assert library.df_metal_advance(handle, 40, events, 2, C.byref(count), C.byref(timing)) != 0
        downloaded, downloaded_arrays = raw_empty_state()
        check(library, library.df_metal_download_lane_state(handle, 0, C.byref(downloaded)))
        assert downloaded.cursor == 0
        for actual, expected in zip(downloaded_arrays, arrays):
            np.testing.assert_array_equal(actual, expected)
        check(library, library.df_metal_upload_lane_state(handle, 1, C.byref(state)))
        advance(library, handle, 1, 2)
    finally:
        library.df_metal_destroy(handle)


def test_native_negative_sleeping_histories_match_serial_and_reject_unsafe_timestamps():
    library = batch_library()
    graph, keepalive = raw_empty_graph()
    handle = create(library, graph, 2)
    references = []
    try:
        for lane, history in enumerate((286, 1024)):
            reference = C.c_void_p()
            check(library, library.df_metal_create(C.byref(graph),
                str(DEFAULT_OUTPUT / 'kernels.metallib').encode(), C.byref(reference)))
            references.append(reference)
            state, arrays = raw_empty_state()
            arrays[0][:] = -51
            arrays[1][:] = 1
            arrays[11][:] = -history
            arrays[15][:] = 1
            check(library, library.df_metal_upload_state(reference, C.byref(state)))
            check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(state)))
        for steps in (1, 100):
            advance(library, handle, steps, 2)
            for lane, reference in enumerate(references):
                advance(library, reference, steps, 1)
                expected, expected_arrays = raw_empty_state()
                actual, actual_arrays = raw_empty_state()
                check(library, library.df_metal_download_state(reference, C.byref(expected)))
                check(library, library.df_metal_download_lane_state(handle, lane, C.byref(actual)))
                assert actual.cursor == expected.cursor
                for observed, wanted in zip(actual_arrays, expected_arrays):
                    np.testing.assert_array_equal(observed.view(np.uint8), wanted.view(np.uint8))
        before, before_arrays = raw_empty_state()
        check(library, library.df_metal_download_lane_state(handle, 0, C.byref(before)))
        for timestamp in (-2**63, -2**31, before.cursor + 1):
            candidate, candidate_arrays = raw_empty_state()
            candidate.cursor = before.cursor
            candidate_arrays[11][:] = timestamp
            assert library.df_metal_upload_lane_state(handle, 0, C.byref(candidate)) != 0
            after, after_arrays = raw_empty_state()
            check(library, library.df_metal_download_lane_state(handle, 0, C.byref(after)))
            assert after.cursor == before.cursor
            for observed, wanted in zip(after_arrays, before_arrays):
                np.testing.assert_array_equal(observed.view(np.uint8), wanted.view(np.uint8))
    finally:
        library.df_metal_destroy(handle)
        for reference in references:
            library.df_metal_destroy(reference)
