"""Window scheduling is opt-in at the Python Metal owner boundary."""
import sys

import numpy as np
import pytest

from doom_learning_v6.metal.batch import MetalBatchExecutor
from test_doom_learning_v6 import brain
from test_doom_metal_batch import assert_equal, snapshot


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), -1, 19, 2**63, 1.0, None, "18"])
def test_bad_window_rejected_before_brains_are_touched(bad):
    class ReadTrap:
        def __iter__(self):
            raise AssertionError("Brains accessed before window validation")

    with pytest.raises((TypeError, ValueError), match="Window ticks must be"):
        MetalBatchExecutor(ReadTrap(), window_ticks=bad)


@pytest.mark.parametrize("chosen", [np.int64(0), np.int64(1), np.int64(18)])
def test_integral_window_is_stored_before_the_denied_native_boundary(tmp_path, monkeypatch, chosen):
    import doom_learning_v6.metal.batch as batch

    owner = MetalBatchExecutor.__new__(MetalBatchExecutor)

    def unavailable(output):
        assert owner.window_ticks == int(chosen)
        raise RuntimeError("Portable native boundary")

    monkeypatch.setattr(batch, "probe", unavailable)
    with pytest.raises(RuntimeError, match="Portable native boundary"):
        owner.__init__([brain(tmp_path)], window_ticks=chosen)
    assert owner.window_ticks == int(chosen)
    with pytest.raises(AttributeError):
        owner.window_ticks = 2


mac = pytest.mark.skipif(sys.platform != "darwin", reason="Real Metal requires macOS")


@mac
@pytest.mark.parametrize("window_ticks", [0, 1, 2, 18])
def test_window_owner_matches_reference_owner_through_learning_checkpoint_and_resets(tmp_path, window_ticks):
    reference_root, candidate_root = tmp_path / "reference", tmp_path / "candidate"
    reference_root.mkdir(); candidate_root.mkdir()
    reference = [brain(reference_root) for _ in range(2)]
    candidate = [brain(candidate_root) for _ in range(2)]
    reference[1].weights_frozen = candidate[1].weights_frozen = True
    reference_paths = [tmp_path / f"reference-{lane}.npz" for lane in range(2)]
    candidate_paths = [tmp_path / f"candidate-{lane}.npz" for lane in range(2)]
    with MetalBatchExecutor(reference) as expected, MetalBatchExecutor(candidate,
            window_ticks=window_ticks) as actual:
        metadata = actual.metadata()
        assert actual.window_ticks == window_ticks
        assert metadata["window_ticks"] == window_ticks
        assert metadata["scheduler"] == ("reference-two-dispatch" if window_ticks == 0 else "temporal-window")
        assert metadata["shared_resident_bytes"] > 0
        assert metadata["mutable_resident_bytes"] > 0
        for owner in (expected, actual):
            for lane in owner.brains:
                lane.backend.start_diagnostics()
        for duration in (10.0, 28.6, 35.0):
            pulses = [([0, 2], 20.0), ([0, 2], -14.0)]
            wanted, _ = expected.step([[], []], duration, learning=[True, True],
                stimulations=pulses, lamina_bias=0)
            observed, _ = actual.step([[], []], duration, learning=[True, True],
                stimulations=pulses, lamina_bias=0)
            for got, want, candidate_lane, reference_lane in zip(observed, wanted, candidate, reference):
                np.testing.assert_array_equal(got, want)
                assert_equal(candidate_lane, reference_lane)
                assert candidate_lane.backend.spike_events == reference_lane.backend.spike_events
                np.testing.assert_array_equal(candidate_lane.eligibility, reference_lane.eligibility)
                assert candidate_lane.weights_frozen == reference_lane.weights_frozen
            assert actual.last_timing["sparse_weight_update_bytes"] == expected.last_timing["sparse_weight_update_bytes"]
        for lane, path in zip(reference, reference_paths):
            lane.checkpoint(path)
        for lane, path in zip(candidate, candidate_paths):
            lane.checkpoint(path)
        for lane in reference + candidate:
            lane.memory_u[:] = -0.25
            lane.memory_w[:] = -0.5
        for lane, path in zip(reference, reference_paths):
            lane.restore(path)
        for lane, path in zip(candidate, candidate_paths):
            lane.restore(path)
        for candidate_lane, reference_lane in zip(candidate, reference):
            assert_equal(candidate_lane, reference_lane)
        for keep_memory in (False, True):
            for lane in reference + candidate:
                lane.memory_u[:] = -0.25
                lane.memory_w[:] = -0.5
            for lane in reference + candidate:
                lane.reset(keep_memory=keep_memory)
            for candidate_lane, reference_lane in zip(candidate, reference):
                assert_equal(candidate_lane, reference_lane)
                assert candidate_lane.weights_frozen == reference_lane.weights_frozen


@mac
@pytest.mark.parametrize("window_ticks", [0, 1, 2, 18])
def test_window_owner_poisoned_close_releases_each_lane_for_cpu_recovery(tmp_path, window_ticks):
    candidate_root, reference_root = tmp_path / "candidate", tmp_path / "reference"
    candidate_root.mkdir(); reference_root.mkdir()
    lanes = [brain(candidate_root) for _ in range(2)]
    reference = [brain(reference_root) for _ in range(2)]
    owner = MetalBatchExecutor(lanes, window_ticks=window_ticks)
    original = lanes[0].v
    lanes[0].v = np.empty(0, dtype=original.dtype)
    with pytest.raises(ValueError, match="binding"):
        owner.close()
    assert owner.closed and owner.poisoned
    lanes[0].v = original
    for lane in lanes:
        lane.reset()
        assert lane._metal_batch_owner is None
    for candidate_lane, reference_lane in zip(lanes, reference):
        counts, _ = candidate_lane.step([], 28.6, learning=True,
            stimulation=([0, 2], 20), lamina_bias=0)
        wanted, _ = reference_lane.step([], 28.6, learning=True,
            stimulation=([0, 2], 20), lamina_bias=0)
        np.testing.assert_array_equal(counts, wanted)
        assert snapshot(candidate_lane) == snapshot(reference_lane)


@mac
@pytest.mark.parametrize("window_ticks", [1, 18])
@pytest.mark.parametrize("keep_memory", [False, True])
def test_windowed_reset_rejects_changed_binding_before_host_mutation(tmp_path, window_ticks, keep_memory):
    lanes = [brain(tmp_path) for _ in range(2)]
    with MetalBatchExecutor(lanes, window_ticks=window_ticks) as owner:
        original = lanes[0].memory_u
        lanes[0].memory_u = lanes[1].memory_u
        before = [snapshot(lane) for lane in lanes]
        try:
            with pytest.raises(ValueError, match="binding"):
                lanes[0].reset(keep_memory=keep_memory)
            assert [snapshot(lane) for lane in lanes] == before
            assert not owner.poisoned
        finally:
            lanes[0].memory_u = original


@mac
@pytest.mark.parametrize("window_ticks", [0, 1, 2, 18])
def test_window_owner_rejects_duplicate_brain_before_native_creation(tmp_path, window_ticks):
    lane = brain(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        MetalBatchExecutor([lane, lane], window_ticks=window_ticks)
    assert getattr(lane, "_metal_batch_owner", None) is None
