"""Native temporal windows: exact reference state and defect-sensitive controls."""
import ctypes as C
import sys

import pytest

from doom_learning_v6.metal.backend import KCEvent, Timing, _pointer
from doom_learning_v6.metal.build import DEFAULT_OUTPUT
from metal_window_helpers import (clone_state, compare_advance, create_window,
    download, raw_fixture, run_fixture, window_library)
from test_doom_metal_batch_native import assert_state, check, model_state

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='Metal requires macOS')


def test_native_window_interface():
    window_library()


@pytest.mark.parametrize('window_ticks', [1, 2, 18])
@pytest.mark.parametrize('cursor', range(19))
@pytest.mark.parametrize('steps', [1, 2, 17, 18, 19, 35, 36, 37, 99, 100])
def test_window_matches_reference(tmp_path, window_ticks, cursor, steps):
    timing, memory = run_fixture(tmp_path, window_ticks=window_ticks,
        cursor=cursor, steps=steps, case='duplicate-self')
    assert timing.encoder_count == 1
    assert timing.indirect_dispatch_count == 0
    assert timing.dispatch_count == 2 + 2 * ((steps + window_ticks - 1) // window_ticks)
    assert timing.mark_grid_threads == steps * 2  # one preparation word per lane
    assert timing.gather_grid_threads == ((steps + window_ticks - 1) // window_ticks) * 8 * 2
    assert memory['shared'] > 0 and memory['mutable'] > 0


@pytest.mark.parametrize('window_ticks', [1, 2, 18])
@pytest.mark.parametrize('case,steps', [
    ('prefilled-reset', 1), ('future-reuse', 2), ('shared-word', 1),
    ('fast-presence', 1), ('zero-modulation', 1), ('lazy-evolution', 18),
    ('gather-before-reset', 1), ('scratch-reuse', 100), ('zero-edge', 100)])
def test_window_raw_semantic_fixtures(tmp_path, window_ticks, case, steps):
    run_fixture(tmp_path, window_ticks=window_ticks, cursor=19, steps=steps, case=case)


@pytest.mark.parametrize('window_ticks', [1, 2, 18])
@pytest.mark.parametrize('steps', [1, 18, 19, 100])
def test_window_multiple_ring_words_and_incoming_word_boundaries(tmp_path, window_ticks, steps):
    timing, _ = run_fixture(tmp_path, window_ticks=window_ticks, cursor=37,
                           steps=steps, case='wide-rows')
    assert timing.mark_grid_threads == steps * 3 * 2
    assert timing.gather_grid_threads == ((steps + window_ticks - 1) // window_ticks) * 67 * 2


@pytest.mark.parametrize('window_ticks', [-1, 19, 2**31 - 1])
def test_window_invalid_mode_is_rejected(window_ticks):
    library = window_library()
    graph, keepalive, state = raw_fixture('zero-edge', 0)
    for graph_ptr in (C.byref(graph), None):
        handle = C.c_void_p()
        assert library.df_metal_create_batch_windowed(graph_ptr,
            str(DEFAULT_OUTPUT / 'kernels.metallib').encode(), 2, window_ticks, C.byref(handle)) != 0
        assert not handle.value


@pytest.mark.parametrize('window_ticks', [0, 1, 2, 18])
@pytest.mark.parametrize('case', ['zero-edge', 'duplicate-self'])
def test_window_factory_bounds_and_memory(window_ticks, case):
    library = window_library()
    graph, keepalive, state = raw_fixture(case, 0)
    for lanes, graph_ptr in ((0, C.byref(graph)), (-1, C.byref(graph)),
                             (2**31 - 1, C.byref(graph)), (2, None)):
        handle = C.c_void_p()
        assert library.df_metal_create_batch_windowed(graph_ptr,
            str(DEFAULT_OUTPUT / 'kernels.metallib').encode(), lanes, window_ticks,
            C.byref(handle)) != 0
        assert not handle.value
    handles = []
    try:
        allocations = []
        for mode in (0, window_ticks):
            handle = create_window(library, graph, 2, mode)
            handles.append(handle)
            shared, mutable = C.c_uint64(), C.c_uint64()
            check(library, library.df_metal_batch_memory_bytes(handle, C.byref(shared), C.byref(mutable)))
            allocations.append((shared.value, mutable.value))
        assert allocations[0][0] == allocations[1][0]
        # Zero-edge bit scratch still has the allocator's one-byte minimum.
        edge_bytes = 1 if case == 'zero-edge' else window_ticks * 2 * 4
        assert allocations[1][1] - allocations[0][1] == (window_ticks * 2 * 8 * 4 + edge_bytes if window_ticks else 0)
    finally:
        for handle in handles:
            library.df_metal_destroy(handle)


@pytest.mark.parametrize('window_ticks', [1, 2, 18])
def test_window_invalid_calls_poison_and_recovery(window_ticks):
    library = window_library()
    graph, keepalive, original = raw_fixture('duplicate-self', 19)
    handles, states = [], [[clone_state(original) for _ in range(2)] for _ in range(2)]
    try:
        for mode, lane_states in zip((0, window_ticks), states):
            handle = create_window(library, graph, 2, mode)
            handles.append(handle)
            check(library, library.df_metal_set_diagnostics(handle, 1))
            for lane, state in enumerate(lane_states):
                check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(state))))
        for handle, lane_states in zip(handles, states):
            for invalid_steps in (0, -1, 101, 2**31 - 1):
                events, count, timing = (KCEvent * 80)(), C.c_int32(-31), Timing()
                assert library.df_metal_advance(handle, invalid_steps, events, 80,
                    C.byref(count), C.byref(timing)) != 0
                assert count.value == -31
                for lane, state in enumerate(lane_states):
                    download(library, handle, lane, state)
                    assert_state(state, original)
            wrong = clone_state(original)
            wrong.cursor += 1
            check(library, library.df_metal_upload_lane_state(handle, 1, C.byref(model_state(wrong))))
            assert library.df_metal_advance(handle, 1, events, 80, C.byref(count), C.byref(timing)) != 0
            assert 'cursors differ' in library.df_metal_last_error().decode()
            for lane in range(2):
                check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(original))))
            # Capacity overflow happens after GPU mutation and poisons every lane.
            assert library.df_metal_advance(handle, 100, None, 0, C.byref(count), C.byref(timing)) != 0
            assert 'capacity exceeded' in library.df_metal_last_error().decode()
            assert library.df_metal_upload_lane_drive(handle, 0, _pointer(original.drive)) != 0
            check(library, library.df_metal_upload_lane_state(handle, 0, C.byref(model_state(original))))
            assert library.df_metal_advance(handle, 1, events, 80, C.byref(count), C.byref(timing)) != 0
            check(library, library.df_metal_upload_lane_state(handle, 1, C.byref(model_state(original))))
        compare_advance(library, handles, states, 100)
        for handle in handles:
            overflow = clone_state(original)
            overflow.cursor = 2**63 - 20
            overflow.last[:] = overflow.cursor - 1
            overflow.modulation_last[:] = overflow.cursor
            for lane in range(2):
                check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(overflow))))
            assert library.df_metal_advance(handle, 2, events, 80, C.byref(count), C.byref(timing)) != 0
            assert 'cursor overflow' in library.df_metal_last_error().decode()
            actual = clone_state(overflow)
            download(library, handle, 0, actual)
            assert_state(actual, overflow)
    finally:
        for handle in handles:
            library.df_metal_destroy(handle)


def test_window_repeat_lane_permutation_and_neighbor_isolation():
    library = window_library()
    graph, keepalive, _ = raw_fixture('duplicate-self', 19)

    def run(order, perturb=False):
        handle = create_window(library, graph, len(order), 18)
        result = {}
        try:
            lane_states = []
            for lane, identity in enumerate(order):
                state = raw_fixture('duplicate-self', 19, identity)[2]
                if perturb and identity == 1:
                    state.drive[0] += 10
                    state.weight[0] += .4
                lane_states.append(state)
                check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(state))))
            from test_doom_metal_batch_native import advance
            events, _ = advance(library, handle, 100, len(order) * 8 * 5)
            for lane, identity in enumerate(order):
                download(library, handle, lane, lane_states[lane])
                result[identity] = (lane_states[lane], [(tick, neuron) for registered, tick, neuron in events if registered == lane])
            return result
        finally:
            library.df_metal_destroy(handle)

    original = run((0, 1))
    for order, perturb in (((0, 1), False), ((1, 0), False), ((0, 1), True), ((0,), False)):
        actual = run(order, perturb)
        for identity in ((0,) if perturb else order):
            assert_state(actual[identity][0], original[identity][0])
            assert actual[identity][1] == original[identity][1]
