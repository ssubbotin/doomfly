"""Complete raw states and independent native reference/window comparisons."""
import ctypes as C
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from doom_learning_v6.metal.backend import Graph, _pointer
from doom_learning_v6.metal.build import DEFAULT_OUTPUT
from test_doom_metal_batch_native import (
    STATE_NAMES, advance, assert_state, batch_library, check, model_state,
)


@lru_cache(maxsize=1)
def window_library():
    library = batch_library()
    assert hasattr(library, 'df_metal_create_batch_windowed'), (
        'Healthy native library lacks df_metal_create_batch_windowed window API')
    library.df_metal_create_batch_windowed.argtypes = [
        C.POINTER(Graph), C.c_char_p, C.c_int32, C.c_int32, C.POINTER(C.c_void_p)]
    library.df_metal_create_batch_windowed.restype = C.c_int
    return library


def create_window(library, graph, lanes, window_ticks):
    handle = C.c_void_p()
    path = str(DEFAULT_OUTPUT / 'kernels.metallib')
    if window_ticks:
        # Mutation controls replace only the candidate's kernels. Reference 0
        # keeps the independently built healthy production metallib.
        path = os.environ.get('DOOM_METAL_WINDOW_TEST_METALLIB', path)
    check(library, library.df_metal_create_batch_windowed(C.byref(graph),
        path.encode(), lanes, window_ticks, C.byref(handle)))
    return handle


def raw_fixture(case, cursor, lane=0):
    """Keep CSR and every State backing array alive in the returned owners."""
    n = 67 if case == 'wide-rows' else 8
    edges = []
    if case == 'wide-rows':
        edges = [(source, target, (source % 7 - 3) * .125)
                 for source in range(n) for target in (31, 32, 66)]
        edges += [(32, 32, 0.), (32, 32, .3)]
    elif case in ('duplicate-self', 'scratch-reuse'):
        edges = [(0, 0, .5), (0, 0, -.25), (0, 1, 0.), (0, 1, .75),
                 (1, 0, -.125), (2, 1, .3)]
    elif case == 'shared-word':
        edges = [(0, 1, .5), (0, 2, .75)]
    elif case == 'fast-presence':
        edges = [(0, 3, 1e20), (1, 3, 3.), (2, 3, -1e20),
                 (0, 4, 0.), (0, 5, 0.)]
    elif case == 'zero-modulation':
        edges = [(0, 1, 0.)]
    elif case == 'gather-before-reset':
        edges = [(0, 1, .5)]
    elif case not in ('prefilled-reset', 'future-reuse', 'lazy-evolution', 'zero-edge'):
        raise ValueError(case)
    edges = sorted(enumerate(edges), key=lambda item: item[1][0])
    rows = [edge for _, edge in edges]
    out_ptr = np.zeros(n + 1, dtype=np.int64)
    for pre, _, _ in rows:
        out_ptr[pre + 1] += 1
    np.cumsum(out_ptr, out=out_ptr)
    out_post = np.array([post for _, post, _ in rows], dtype=np.int32)
    incoming = sorted(range(len(rows)), key=lambda edge: (rows[edge][1], rows[edge][0], edges[edge][0]))
    in_ptr = np.zeros(n + 1, dtype=np.int64)
    for _, post, _ in rows:
        in_ptr[post + 1] += 1
    np.cumsum(in_ptr, out=in_ptr)
    in_pre = np.array([rows[edge][0] for edge in incoming], dtype=np.int32)
    in_edge = np.array(incoming, dtype=np.int32)
    kc_mask = np.zeros(n, dtype=np.uint8)
    kc_mask[0] = 1
    modulation_mask = np.zeros(n, dtype=np.uint8)
    if case in ('duplicate-self', 'scratch-reuse', 'wide-rows'):
        modulation_mask[2] = 1
    if case == 'zero-modulation':
        modulation_mask[0] = 1
    keepalive = (out_ptr, out_post, in_ptr, in_pre, in_edge, kc_mask, modulation_mask)
    graph = Graph(n, len(rows), 19, .1, 8., 200., *[_pointer(a) for a in keepalive])
    state = SimpleNamespace(cursor=cursor, n=n,
        weight=np.array([value for _, _, value in rows], dtype=np.float32))
    for name in ('v', 'g', 'drive', 'previous_drive', 'modulation', 'rest', 'adaptation'):
        setattr(state, name, np.full(n, -52. if name in ('v', 'rest') else 0., dtype=np.float32))
    for name, shape, dtype in (
        ('refractory', n, np.int16), ('queue', (19, n), np.int32),
        ('queue_count', 19, np.int32), ('counts', n, np.int32),
        ('active', n, np.int32), ('active_flag', n, np.uint8), ('nactive', 1, np.int32)):
        setattr(state, name, np.zeros(shape, dtype=dtype))
    state.last = np.full(n, cursor - 1, dtype=np.int64)
    state.modulation_last = np.full(n, cursor, dtype=np.int64)

    def queued(slot, members):
        state.queue[slot % 19, :len(members)] = members
        state.queue_count[slot % 19] = len(members)

    if case == 'prefilled-reset':
        state.g[0], state.last[0] = .5, cursor - 7
        queued(cursor + 18, [0])
    elif case == 'future-reuse':
        state.active_flag[1], state.v[1], state.refractory[1] = 1, -40., 2
    elif case in ('shared-word', 'gather-before-reset', 'zero-modulation'):
        queued(cursor, [0])
        if case == 'gather-before-reset':
            state.active_flag[1], state.v[1] = 1, -40.
        elif case == 'zero-modulation':
            assert cursor >= 7, 'Modulatory histories must be nonnegative'
            state.modulation[1], state.modulation_last[1] = .5, cursor - 7
    elif case == 'fast-presence':
        queued(cursor, [0, 1, 2])
        state.g[3:5], state.g[5] = -0.0, 7.
    elif case == 'lazy-evolution':
        state.v[0], state.g[0], state.adaptation[0], state.last[0] = -51., .3, .7, cursor - 7
    elif case in ('duplicate-self', 'scratch-reuse', 'wide-rows'):
        queued(cursor, list(range(n)) if case == 'wide-rows' else [0, 1, 2])
        queued(cursor + 18, [6])
        state.drive[[0, 1, 2]] = [20. + lane * 4, 12. + lane, 17. + lane * 2]
        state.weight[:] *= np.float32(1. + lane * .125)
        if case == 'wide-rows':
            state.drive[[31, 32, 66]] = [19. + lane, 23. + lane, 27. + lane]
    active = np.flatnonzero(state.active_flag)
    state.active[:len(active)] = active
    state.nactive[0] = len(active)
    return graph, keepalive, state


def clone_state(state):
    return SimpleNamespace(cursor=state.cursor, n=state.n,
                           **{name: getattr(state, name).copy() for name in STATE_NAMES})


def download(library, handle, lane, state):
    raw = model_state(state)
    check(library, library.df_metal_download_lane_state(handle, lane, C.byref(raw)))
    state.cursor = raw.cursor


def compare_advance(library, handles, states, steps):
    all_events, timings = [], []
    for handle, lanes in zip(handles, states):
        events, timing = advance(library, handle, steps, len(lanes) * lanes[0].n * 5)
        all_events.append(events)
        timings.append(timing)
        for lane, state in enumerate(lanes):
            download(library, handle, lane, state)
            counts, cursor = np.empty(state.n, dtype=np.int32), C.c_int64(-1)
            check(library, library.df_metal_download_lane_observation(handle, lane,
                  _pointer(counts), C.byref(cursor)))
            np.testing.assert_array_equal(counts, state.counts)
            assert cursor.value == state.cursor
    assert all_events[0] == all_events[1]
    for reference, candidate in zip(*states):
        assert_state(candidate, reference)
    return timings[1]


def run_fixture(tmp_path, *, window_ticks, cursor, steps, case, lanes=2):
    library = window_library()
    graph, graph_keepalive, first = raw_fixture(case, cursor)
    originals = [first] + [raw_fixture(case, cursor, lane)[2] for lane in range(1, lanes)]
    states = [[clone_state(state) for state in originals] for _ in range(2)]
    handles = []
    try:
        for mode, lane_states in zip((0, window_ticks), states):
            handle = create_window(library, graph, lanes, mode)
            handles.append(handle)
            check(library, library.df_metal_set_diagnostics(handle, 1))
            for lane, state in enumerate(lane_states):
                check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(state))))
        lengths = (18, 1, 17, 19, 100) if case == 'scratch-reuse' else (steps,)
        for cycle in range(2 if case == 'scratch-reuse' else 1):
            if cycle:
                # Successful reuploads include every unconsumed future queue entry.
                for handle, lane_states in zip(handles, states):
                    for lane, state in enumerate(lane_states):
                        check(library, library.df_metal_upload_lane_state(handle, lane, C.byref(model_state(state))))
            for length in lengths:
                timing = compare_advance(library, handles, states, length)
                if case in ('duplicate-self', 'scratch-reuse'):
                    for handle, lane_states in zip(handles, states):
                        for lane, state in enumerate(lane_states):
                            ids = np.array([0, 2], dtype=np.int64)
                            values = np.array([.6 + lane * .2, -.05 * (lane + 1)], dtype=np.float32)
                            check(library, library.df_metal_update_lane_weights(handle, lane, 2,
                                  _pointer(ids), _pointer(values)))
            if case == 'duplicate-self':
                # Observe lane-specific plastic edits at the next bin boundary.
                compare_advance(library, handles, states, 2)
        shared, mutable = C.c_uint64(), C.c_uint64()
        check(library, library.df_metal_batch_memory_bytes(handles[1], C.byref(shared), C.byref(mutable)))
        return timing, {'shared': shared.value, 'mutable': mutable.value}
    finally:
        for handle in handles:
            library.df_metal_destroy(handle)
