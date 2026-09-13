"""Terminal hashes must describe checked, materialized checkpoint state."""
import ctypes as C
import hashlib
import json
import sys

import numpy as np
import pytest

from doom_learning_v6.backend import BackendError
from doom_learning_v6.causal_pilot import _state_hashes
from doom_learning_v6.metal.backend import State
from test_doom_learning_v6 import brain
from test_doom_metal_batch import guarded_portable_owner, executor_type
from test_doom_metal_batch_rgb import visual_brain, frame


CHECKPOINT_NAMES = (
    'weight', 'v', 'g', 'refractory', 'drive', 'previous_drive', 'queue',
    'queue_count', 'counts', 'luminance', 'active', 'active_flag', 'nactive',
    'last', 'eligibility', 'eligibility_last', 'modulation', 'modulation_last',
    'adaptation', 'rate_kc', 'rate_dan', 'memory_u', 'memory_w', 'r8_light',
)


def checkpoint_hashes(path, names=CHECKPOINT_NAMES):
    """Read persisted bytes independently of the terminal hashing helper."""
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {'metadata', *names}
        metadata = json.loads(str(saved['metadata']))
        return {
            'cursor': int(metadata['cursor']),
            'total_spikes': int(metadata['total_spikes']),
            **{name + '_sha256': hashlib.sha256(saved[name].tobytes()).hexdigest()
               for name in names},
        }


def unchanged_metadata(b):
    return (b.cursor, b.sim_ms, b.total_spikes, b.weights_frozen,
            b.weight.tobytes(), b.memory_u.tobytes(), b.memory_w.tobytes())


@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.parametrize('visual', [False, True])
def test_cpu_terminal_hashes_preserve_all_checkpoint_bytes_and_metadata(tmp_path, frozen, visual):
    b = visual_brain(tmp_path) if visual else brain(tmp_path)
    b.weights_frozen = frozen
    if visual:
        b.rgb_step(frame(0), 28.6, learning=True, stimulation=([0, 2], 20), lamina_bias=0)
    else:
        b.step([], 28.6, learning=True, stimulation=([0, 2], 20), lamina_bias=0)
    names = CHECKPOINT_NAMES if visual else CHECKPOINT_NAMES[:-1]
    before = unchanged_metadata(b)
    arrays = {name: getattr(b, name).tobytes() for name in names}
    terminal = _state_hashes(b)
    b.checkpoint(tmp_path / 'cpu.npz')
    assert terminal == checkpoint_hashes(tmp_path / 'cpu.npz', names)
    assert unchanged_metadata(b) == before
    assert {name: getattr(b, name).tobytes() for name in names} == arrays


def test_portable_terminal_hashes_download_stale_state_through_checked_lane(tmp_path, monkeypatch):
    """Removing checked synchronization leaves hashes of the initial host voltage."""
    host, current = visual_brain(tmp_path), visual_brain(tmp_path)
    current.weights_frozen = host.weights_frozen = True
    current.rgb_step(frame(0), 28.6, stimulation=([0, 2], 20), lamina_bias=0)
    current.checkpoint(tmp_path / 'authoritative.npz')
    expected = checkpoint_hashes(tmp_path / 'authoritative.npz')
    assert host.v.tobytes() != current.v.tobytes()
    device_names = {name for name, _ in State._fields_} - {'cursor'}
    # Mirror the narrow observation/host-rule boundary after device advance.
    for name in CHECKPOINT_NAMES:
        if name not in device_names or name in ('drive', 'counts'):
            getattr(host, name)[:] = getattr(current, name)
    host.cursor, host.sim_ms, host.total_spikes = current.cursor, current.sim_ms, current.total_spikes
    with guarded_portable_owner([host], monkeypatch) as (owner, calls):
        host.backend._host_state_valid = False
        before = unchanged_metadata(host)
        epochs = (host.backend._host_weight_epoch, host.backend._device_weight_epoch)

        def download(handle, lane, state_pointer):
            assert handle.value == owner.handle.value and lane == 0
            state = C.cast(state_pointer, C.POINTER(State)).contents
            for name in device_names:
                source = getattr(current, name)
                if source.nbytes:
                    C.memmove(getattr(state, name), source.ctypes.data, source.nbytes)
            state.cursor = current.cursor
            return 0

        owner.library.df_metal_download_lane_state = download
        assert _state_hashes(host) == expected
        assert host.backend._host_state_valid
        assert unchanged_metadata(host) == before
        assert (host.backend._host_weight_epoch, host.backend._device_weight_epoch) == epochs
        assert calls == []  # Existing raw upload trap stays armed.


def test_terminal_hashes_propagate_checked_materialization_error(tmp_path, monkeypatch):
    b = brain(tmp_path)
    original = BackendError('terminal download failed')
    with guarded_portable_owner([b], monkeypatch) as (owner, calls):
        b.backend._host_state_valid = False
        before = unchanged_metadata(b)
        def fail(*args):
            raise original
        owner.library.df_metal_download_lane_state = fail
        with pytest.raises(BackendError) as caught:
            _state_hashes(b)
        assert caught.value is original
        assert not b.backend._host_state_valid
        assert unchanged_metadata(b) == before and calls == []


@pytest.mark.parametrize('interrupted', [False, True])
@pytest.mark.parametrize('secondary_failure', [None, 'json', 'checkpoint'])
def test_terminal_materialization_failure_preserves_trace_and_available_checkpoints(
        tmp_path, monkeypatch, interrupted, secondary_failure):
    from doom_learning import common
    from doom_learning_v6.causal_pilot import replay_episode
    from test_doom_learning_causal_pilot import _SerialCpuExecutor, _ThreeFrames, READOUTS
    lanes = [visual_brain(tmp_path) for _ in range(3)]
    download_error = BackendError('terminal materialization failure')
    original = KeyboardInterrupt('primary frame interruption') if interrupted else download_error
    data = _ThreeFrames(fail_after=0, failure=original) if interrupted else _ThreeFrames()
    if not interrupted:
        data.frame_count = 1
        data.turn_targets = lambda: np.zeros(1)
        frames = data.iter_frames()
        def one_frame():
            try:
                yield next(frames)
            finally:
                frames.close()
        data.iter_frames = one_frame
    downloads = []
    def fail(reason):
        downloads.append(reason)
        raise download_error
    monkeypatch.setattr(lanes[0].backend, 'materialize', fail)
    checkpoints = []
    for index, b in enumerate(lanes):
        checkpoint = b.checkpoint
        def attempt(path, index=index, checkpoint=checkpoint):
            checkpoints.append(index)
            if secondary_failure == 'checkpoint' and index == 1:
                raise OSError('secondary checkpoint writer failure')
            return checkpoint(path)
        monkeypatch.setattr(b, 'checkpoint', attempt)
    phase = tmp_path / 'failed-terminal'
    if secondary_failure == 'json':
        save_json = common.save_json
        def fail_progress(path, value):
            if path == phase / 'progress.json' and 'frames' in value:
                raise OSError('secondary progress writer failure')
            return save_json(path, value)
        monkeypatch.setattr(common, 'save_json', fail_progress)
    with pytest.raises(type(original)) as caught:
        replay_episode(lanes, _SerialCpuExecutor(lanes), data, READOUTS,
                       np.zeros((3, data.frame_count)), learning=[False] * 3, frozen=[True] * 3,
                       directory=phase, warmup_ms=0)
    assert caught.value is original and data.closed
    # The failure must retain actual trace records, beyond the progress counter.
    assert all((phase / f'lane-{index}/episode.json').exists() for index in range(3))
    summary = json.loads((phase / 'summary.json').read_text())
    assert summary['complete'] is False
    assert summary['failure']['type'] == type(original).__name__
    for index in range(3):
        record = json.loads((phase / f'lane-{index}/episode.json').read_text())
        assert len(record['trace']) == record['frames'] == 1
        assert record['trace'][0]['index'] == 0
        assert record == summary['lanes'][index]
        assert not (phase / f'lane-{index}/final.npz').exists()
    unavailable = summary['lanes'][0]
    assert unavailable['complete'] is False
    assert unavailable['terminal_state'] is None
    assert unavailable['terminal_state_available'] is False
    assert unavailable['terminal_state_error'] == {'type': 'BackendError'}
    assert unavailable['checkpoint_available'] is False
    assert unavailable['checkpoint_attempted'] is False
    assert not (phase / 'lane-0/failure.npz').exists()
    assert downloads == ['checkpoint']  # Failed lane is never synchronized again.
    assert checkpoints == [1, 2]
    for index in (1, 2):
        if secondary_failure != 'checkpoint' or index != 1:
            assert summary['lanes'][index]['terminal_state'] == checkpoint_hashes(
                phase / f'lane-{index}/failure.npz')
    if interrupted:
        assert any('Secondary record failure: BackendError' in note for note in original.__notes__)
    if secondary_failure is not None:
        assert any('OSError' in note for note in original.__notes__)


@pytest.mark.skipif(sys.platform != 'darwin', reason='Metal requires macOS')
def test_native_four_lane_terminal_hashes_match_saved_24_array_checkpoint(tmp_path):
    """Small native numerical regression; this supplies no biological evidence."""
    lanes = [visual_brain(tmp_path) for _ in range(4)]
    for b in lanes:
        b.weights_frozen = True
    with executor_type()(lanes, window_ticks=18) as owner:
        initial_voltage = [b.v.tobytes() for b in lanes]
        owner.rgb_step([frame(lane % 2) for lane in range(4)], 28.6, learning=False,
                       stimulations=[([0, 2], 20 + lane) for lane in range(4)], lamina_bias=0)
        for lane, b in enumerate(lanes):
            assert not b.backend._host_state_valid
            assert b.v.tobytes() == initial_voltage[lane]
            before = unchanged_metadata(b)
            epochs = (b.backend._host_weight_epoch, b.backend._device_weight_epoch)
            terminal = _state_hashes(b)
            assert b.backend._host_state_valid
            assert b.v.tobytes() != initial_voltage[lane]
            b.checkpoint(tmp_path / f'lane-{lane}.npz')
            assert terminal == checkpoint_hashes(tmp_path / f'lane-{lane}.npz')
            assert unchanged_metadata(b) == before
            assert (b.backend._host_weight_epoch, b.backend._device_weight_epoch) == epochs
        assert not owner.poisoned
