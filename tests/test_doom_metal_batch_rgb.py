"""Synthetic RGB preparation and real native state/fixed-control identity."""
import math
import sys

import numpy as np
import pytest

from doom.engine import NeuralControls
from doom_learning_v6.backend import BackendError
from doom_learning_v6.visual import VisualMemoryBrain
from test_doom_learning_v6 import brain
from test_doom_metal_batch import executor_type, snapshot, assert_equal


def visual_brain(tmp_path, **kwargs):
    b = brain(tmp_path, **kwargs)
    b.__class__ = VisualMemoryBrain
    b.retina = np.array([3], dtype=np.int32)
    b.uv = np.array([[.75, .25]], dtype=np.float32)
    b.luminance = np.zeros(1, dtype=np.float32)
    b.initial['luminance'] = b.luminance.copy()
    b.r8 = np.array([0, 2], dtype=np.int32)
    b.r8_uv = np.array([[0, 0], [1, 1]], dtype=np.float32)
    b.r8_channel = np.array([2, 1], dtype=np.int32)
    b.r8_light = np.zeros(2, dtype=np.float32)
    b.corrected_edges = np.empty(0, dtype=np.int64)
    b.fields.append('r8_light'); b.initial['r8_light'] = b.r8_light.copy()
    return b


def frame(index):
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[0, 0, 2] = (255, 128)[index]
    image[2, 3, 1] = (128, 255)[index]
    image[0, 2] = (100, 120, 140)
    return image


def controls():
    return NeuralControls([{'index': 0, 'type': 'DNa02', 'side': 'L'},
                           {'index': 2, 'type': 'DNa02', 'side': 'R'},
                           {'index': 1, 'type': 'DNp09', 'side': ''},
                           {'index': 3, 'type': 'MN9', 'side': ''}])


def test_rgb_helper_preserves_channel_sampling_linearization_smoothing_and_pulse_order(tmp_path):
    b = visual_brain(tmp_path)
    extra = [([1], 7.)]
    retinal, pulses = b._prepare_rgb_input(frame(0), 8.6, stimulation=extra)
    # Array ufuncs retain float32 at each operation, unlike numpy scalar math.
    linear = ((np.array([128], dtype=np.float32) / 255 + .055) / 1.055) ** 2.4
    expected = np.array([1., linear[0]], dtype=np.float32) * (1 - math.exp(-.86))
    np.testing.assert_array_equal(b.r8_light.view(np.uint32), expected.view(np.uint32))
    assert retinal.shape == (1,)
    assert pulses[0] == extra[0] and len(extra) == 1 and len(pulses) == 2
    np.testing.assert_array_equal(pulses[1][0], [0, 2])
    np.testing.assert_array_equal(pulses[1][1], 30 * expected / (.02 + expected))


def test_attached_direct_rgb_fails_before_r8_or_neural_state_change(tmp_path):
    b = visual_brain(tmp_path)
    b._metal_batch_owner = object()
    before = snapshot(b)
    with pytest.raises(BackendError, match='owned'): b.rgb_step(frame(0), 100)
    assert snapshot(b) == before


def test_serial_rgb_subdivisions_match_explicit_observation_bins(tmp_path):
    a, e = visual_brain(tmp_path), visual_brain(tmp_path)
    total, _ = a.rgb_step(frame(0), 1000 / 35, learning=True, stimulation=([1], 3.))
    expected = np.zeros(e.n, dtype=np.int32)
    for duration in (10., 10., 8.6):
        counts, _ = e.rgb_step(frame(0), duration, learning=True, stimulation=([1], 3.))
        expected += counts
    e.counts[:] = expected
    np.testing.assert_array_equal(total, expected)
    assert_equal(a, e)


mac = pytest.mark.skipif(sys.platform != 'darwin', reason='Real Metal requires macOS')


def run_rgb(tmp_path, order=(0, 1), perturb=False, window_ticks=0):
    references = [visual_brain(tmp_path, backend='metal') for _ in range(2)]
    hosts = [visual_brain(tmp_path) for _ in range(2)]
    for b in (references[1], hosts[1]): b.weights_frozen = True
    serial_controls, batch_controls = [controls(), controls()], [controls(), controls()]
    trajectory = [[], []]
    try:
        with executor_type()([hosts[i] for i in order], window_ticks=window_ticks) as executor:
            for b in references + hosts: b.backend.start_diagnostics()
            for duration in (1000 / 35, 28.5, 10., 35., 1000 / 35):
                images = [frame(0), frame(1)]
                pulses = [([1], 3.), ([1], 5.)]
                if perturb:
                    images[1][:] = 0
                    pulses[1] = ([1], 18.)
                expected = [b.rgb_step(f, duration, learning=True, stimulation=p, lamina_bias=0)[0]
                            for b, f, p in zip(references, images, pulses)]
                actual, wall = executor.rgb_step([images[i] for i in order], duration,
                    learning=[True] * len(order), stimulations=[pulses[i] for i in order], lamina_bias=0)
                assert wall >= 0
                bins = (round(duration / .1) + 99) // 100
                assert executor.last_timing['encoder_count'] == bins
                assert executor.last_timing['sparse_weight_update_bytes'] == bins * 12
                assert executor.last_timing['materialize_bytes'] == 0
                for lane, i in enumerate(order):
                    np.testing.assert_array_equal(actual[lane], expected[i])
                    assert_equal(hosts[i], references[i])
                    assert hosts[i].backend.spike_events == references[i].backend.spike_events
                    decoded = batch_controls[i].decode(actual[lane], duration / 1000)
                    assert decoded == serial_controls[i].decode(expected[i], duration / 1000)
                    trajectory[i].append((snapshot(hosts[i]), decoded,
                                          list(hosts[i].backend.spike_events)))
        return trajectory
    finally:
        for b in references: b.backend.close()


@mac
def test_rgb_distinct_frames_repeats_permutation_and_pixel_teacher_isolation(tmp_path):
    baseline = run_rgb(tmp_path)
    assert run_rgb(tmp_path) == baseline
    assert run_rgb(tmp_path, order=(1, 0)) == baseline
    assert run_rgb(tmp_path, perturb=True)[0] == baseline[0]
    assert run_rgb(tmp_path, order=(0,))[0] == baseline[0]


@mac
@pytest.mark.parametrize('window_ticks', [0, 1, 2, 18])
def test_rgb_window_modes_preserve_repeat_permutation_and_pixel_teacher_isolation(tmp_path, window_ticks):
    baseline = run_rgb(tmp_path, window_ticks=window_ticks)
    assert run_rgb(tmp_path, window_ticks=window_ticks) == baseline
    assert run_rgb(tmp_path, order=(1, 0), window_ticks=window_ticks) == baseline
    assert run_rgb(tmp_path, perturb=True, window_ticks=window_ticks)[0] == baseline[0]
    assert run_rgb(tmp_path, order=(0,), window_ticks=window_ticks)[0] == baseline[0]


@mac
@pytest.mark.parametrize('window_ticks', [1, 18])
def test_rgb_invalid_neighbor_frame_and_stimulation_do_not_change_any_lane(tmp_path, window_ticks):
    lanes = [visual_brain(tmp_path), visual_brain(tmp_path)]
    with executor_type()(lanes, window_ticks=window_ticks) as executor:
        before = [snapshot(b) for b in lanes]
        for invalid in (np.zeros((3, 4), dtype=np.uint8), frame(0).astype(float),
                        np.zeros((0, 4, 3), dtype=np.uint8)):
            with pytest.raises(ValueError): executor.rgb_step([frame(0), invalid], 28.6)
            assert [snapshot(b) for b in lanes] == before
        with pytest.raises(ValueError):
            executor.rgb_step([frame(0), frame(1)], 28.6,
                              stimulations=[None, ([5], 20)])
        assert [snapshot(b) for b in lanes] == before
