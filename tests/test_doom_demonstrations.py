"""Behavioral checks using real source-shaped Parquet and 35 Hz video."""

import hashlib
import json
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from doom_learning_v6.demonstrations import OfflineEpisode


NAMES = ["shoot", "move_forward", "move_backward", "strafe_left",
         "strafe_right", "speed", "turn_left", "turn_right", "turn_delta"]
REVISION = "6cc00d60462885c1d61dd480228af58c3b81b807"


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def refresh_hashes(directory):
    report = json.loads((directory / "validation-report.json").read_text())
    report["files"] = {}
    for name in ["SOURCE-README.md", "source-info.json", "episode-metadata.json",
                 "episode_000000.mp4", "episode_000000.parquet"]:
        raw = (directory / name).read_bytes()
        report["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                 "bytes": len(raw)}
    report["published_sha256"] = {
        name: report["files"][name]["sha256"]
        for name in ["episode_000000.mp4", "episode_000000.parquet"]}
    write_json(directory / "validation-report.json", report)


def change_json(directory, name, change):
    data = json.loads((directory / name).read_text())
    change(data)
    write_json(directory / name, data)
    refresh_hashes(directory)


def change_column(directory, name, values):
    path = directory / "episode_000000.parquet"
    table = pq.read_table(path)
    position = table.schema.get_field_index(name)
    table = table.set_column(position, name, pa.array(values, type=table[name].type))
    pq.write_table(table, path)
    refresh_hashes(directory)


@pytest.fixture
def checked_episode_fixture(tmp_path):
    def make(turns=None, count=8, scenario="defend_the_center", fps=35):
        if turns is not None:
            count = len(turns)
        directory = tmp_path / str(len(list(tmp_path.iterdir())))
        directory.mkdir()
        actions = np.zeros((count, 9), dtype=np.float32)
        if turns is not None:
            actions[:, 6] = np.asarray(turns) < 0
            actions[:, 7] = np.asarray(turns) > 0
        source = {"fps": 35, "robot_type": "vizdoom_appo", "features": {
            "action": {"dtype": "float32", "shape": [9], "names": NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images": {"dtype": "video", "shape": [3, 16, 24],
                "names": ["channels", "height", "width"], "info": {
                    "video.height": 16, "video.width": 24, "video.channels": 3,
                    "video.fps": 35, "video.codec": "h264",
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                    "has_audio": False}}}}
        metadata = {"episode_index": 0, "length": count, "scenario": scenario,
                    "raw_file_name": f"{scenario}_000000",
                    "policy": "sample_factory_appo_rnn",
                    "capture_mode": "tick_smooth_native", "capture_frame_skip": 1,
                    "policy_action_repeat": 4, "humanized": False}
        (directory / "SOURCE-README.md").write_text("Apache-2.0 source notice\n")
        write_json(directory / "source-info.json", source)
        write_json(directory / "episode-metadata.json", metadata)
        write_json(directory / "validation-report.json", {
            "dataset": "Yunncheng/gamewam-vizdoom", "dataset_revision": REVISION,
            "scenario": scenario, "episode_index": 0, "checks": {"all": True}})
        pq.write_table(pa.table({
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 9)),
            "timestamp": pa.array(np.arange(count, dtype=np.float64) / 35,
                                   type=pa.float32()),
            "frame_index": pa.array(range(count), type=pa.int64()),
            "episode_index": pa.array([0] * count, type=pa.int64()),
            "index": pa.array(range(300, 300 + count), type=pa.int64()),
            "observation.state": pa.array([[np.nan] * 4] * count,
                                          type=pa.list_(pa.float32(), 4)),
        }), directory / "episode_000000.parquet")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"testsrc2=size=24x16:rate={fps}", "-frames:v", str(max(count, 1)),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y",
                        str(directory / "episode_000000.mp4")], check=True)
        refresh_hashes(directory)
        return directory
    return make


def test_raw_reader_preserves_indices_actions_and_portable_identity(checked_episode_fixture):
    directory = checked_episode_fixture(count=10)
    data = OfflineEpisode(directory)
    assert (data.frame_count, data.width, data.height) == (10, 24, 16)
    assert data.actions.dtype == np.float32
    assert not data.actions.flags.writeable
    with pytest.raises(ValueError):
        data.actions.setflags(write=True)
    assert data.identity["dataset"] == "Yunncheng/gamewam-vizdoom"
    assert data.identity["dataset_revision"] == REVISION
    assert data.identity["scenario"] == "defend_the_center"
    assert data.identity["episode_index"] == 0
    assert str(directory) not in json.dumps(data.identity)
    samples = list(data.iter_frames())
    assert [sample.index for sample in samples] == list(range(10))
    assert samples[-1].timestamp == float(np.float32(9 / 35))
    assert samples[0].rgb.shape == (16, 24, 3)
    assert samples[0].rgb.dtype == np.uint8
    assert not samples[0].action.flags.writeable
    assert not hasattr(samples[0], "state")
    np.testing.assert_array_equal(samples[0].action, [0] * 9)


def test_binary_turn_acceleration_keeps_history_on_direction_change(checked_episode_fixture):
    data = OfflineEpisode(checked_episode_fixture(turns=[-1] * 8 + [1] * 4 + [0] * 4 + [1] * 4))
    targets = data.turn_targets()
    assert targets.dtype == np.float64
    np.testing.assert_array_equal(targets, [
        -1.7578125, -1.7578125, -1.7578125, -1.7578125, -1.7578125,
        -3.515625, -3.515625, -3.515625,
        3.515625, 3.515625, 3.515625, 3.515625,
        0.0, 0.0, 0.0, 0.0,
        1.7578125, 1.7578125, 1.7578125, 1.7578125])


def test_partial_final_repeat_block_is_preserved(checked_episode_fixture):
    data = OfflineEpisode(checked_episode_fixture(turns=[-1] * 4 + [1] * 2))
    np.testing.assert_array_equal(data.turn_targets(),
                                 [-1.7578125] * 4 + [1.7578125, 3.515625])


@pytest.mark.parametrize("filename", ["SOURCE-README.md", "source-info.json",
    "episode-metadata.json", "episode_000000.mp4", "episode_000000.parquet"])
def test_altered_file_bytes_are_rejected_even_with_prior_passing_checks(checked_episode_fixture, filename):
    directory = checked_episode_fixture()
    with (directory / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum|hash|bytes"):
        OfflineEpisode(directory)


@pytest.mark.parametrize("field,value", [
    ("dataset_revision", "wrong"), ("dataset", "other"),
    ("scenario", "battle1"), ("episode_index", 1)])
def test_report_identity_must_match_pinned_source_and_metadata(checked_episode_fixture, field, value):
    directory = checked_episode_fixture()
    change_json(directory, "validation-report.json", lambda data: data.update({field: value}))
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


@pytest.mark.parametrize("field,value", [("length", 9), ("episode_index", 1),
    ("capture_mode", "interpolated"), ("capture_frame_skip", 4),
    ("policy_action_repeat", 0), ("humanized", True),
    ("raw_file_name", "other"), ("policy", "unknown")])
def test_metadata_mismatches_are_rejected_before_iteration(checked_episode_fixture, field, value):
    directory = checked_episode_fixture()
    change_json(directory, "episode-metadata.json", lambda data: data.update({field: value}))
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


@pytest.mark.parametrize("column,values", [
    ("frame_index", [1, 0, 2, 3, 4, 5, 6, 7]),
    ("index", [300, 301, 303, 304, 305, 306, 307, 308]),
    ("episode_index", [0, 0, 1, 0, 0, 0, 0, 0]),
    ("timestamp", [0, .02, 2/35, 3/35, 4/35, 5/35, 6/35, 7/35]),
    ("timestamp", [float("nan")] * 8)])
def test_original_indices_episode_and_float32_timing_are_checked(checked_episode_fixture, column, values):
    directory = checked_episode_fixture()
    change_column(directory, column, values)
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


def test_source_index_wrap_into_negative_int64_is_rejected(checked_episode_fixture):
    directory = checked_episode_fixture()
    change_column(directory, "index", [
        9223372036854775804, 9223372036854775805,
        9223372036854775806, 9223372036854775807,
        -9223372036854775808, -9223372036854775807,
        -9223372036854775806, -9223372036854775805])
    with pytest.raises(ValueError, match="source indices"):
        OfflineEpisode(directory)


@pytest.mark.parametrize("column,value", [(0, .5), (7, -1), (8, 1.01),
                                         (8, -1.01), (8, np.inf), (6, np.nan)])
def test_invalid_action_ranges_are_rejected(checked_episode_fixture, column, value):
    directory = checked_episode_fixture()
    actions = [[0.] * 9 for _ in range(8)]
    actions[0][column] = value
    change_column(directory, "action", actions)
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


def test_reordered_action_names_are_rejected(checked_episode_fixture):
    directory = checked_episode_fixture()
    def reorder(data):
        data["features"]["action"]["names"] = NAMES[1:] + NAMES[:1]
    change_json(directory, "source-info.json", reorder)
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


@pytest.mark.parametrize("dimension", [0, 32, 1000000])
def test_source_dimensions_are_validated_before_decode_allocation(checked_episode_fixture, dimension):
    directory = checked_episode_fixture()
    def resize(data):
        image = data["features"]["observation.images"]
        image["shape"][2] = dimension
        image["info"]["video.width"] = dimension
    change_json(directory, "source-info.json", resize)
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


def test_actual_dimension_mismatch_rejects_before_ffprobe_frame_scan(checked_episode_fixture, monkeypatch):
    directory = checked_episode_fixture()
    def change_width(data):
        image = data["features"]["observation.images"]
        image["shape"][2] = 16
        image["info"]["video.width"] = 16
    change_json(directory, "source-info.json", change_width)
    commands = []
    real_run = subprocess.run
    def observe_probe(command, **kwargs):
        if command[0] == "ffprobe":
            commands.append(list(command))
        return real_run(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", observe_probe)
    with pytest.raises(ValueError, match="dimensions"):
        OfflineEpisode(directory)
    assert commands
    assert all("-show_frames" not in command for command in commands)


def test_valid_video_stream_metadata_is_probed_before_frame_scan(checked_episode_fixture, monkeypatch):
    directory = checked_episode_fixture()
    commands = []
    real_run = subprocess.run
    def observe_probe(command, **kwargs):
        if command[0] == "ffprobe":
            commands.append(list(command))
        return real_run(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", observe_probe)
    assert OfflineEpisode(directory).frame_count == 8
    assert "-show_streams" in commands[0]
    assert "-show_frames" not in commands[0]
    assert any("-show_frames" in command for command in commands[1:])


def test_every_video_pts_is_checked_against_exact_tick_times(checked_episode_fixture):
    directory = checked_episode_fixture(fps=30)
    with pytest.raises(ValueError, match="PTS|timestamp|rate"):
        OfflineEpisode(directory)


def test_matching_video_rate_does_not_authorize_shifted_pts(checked_episode_fixture):
    directory = checked_episode_fixture()
    original = directory / "episode_000000.mp4"
    shifted = directory / "shifted.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), "-c", "copy",
                    "-output_ts_offset", "0.028571428571428571", str(shifted)], check=True)
    shifted.replace(original)
    refresh_hashes(directory)
    with pytest.raises(ValueError, match="PTS"):
        OfflineEpisode(directory)


def test_video_count_mismatch_is_rejected_before_iteration(checked_episode_fixture):
    directory = checked_episode_fixture()
    table = pq.read_table(directory / "episode_000000.parquet").slice(0, 4)
    pq.write_table(table, directory / "episode_000000.parquet")
    change_json(directory, "episode-metadata.json", lambda data: data.update(length=4))
    with pytest.raises(ValueError, match="frame count"):
        OfflineEpisode(directory)


def test_action_changes_inside_repeat_block_are_rejected(checked_episode_fixture):
    directory = checked_episode_fixture()
    actions = [[0.] * 9 for _ in range(8)]
    actions[2][7] = 1
    change_column(directory, "action", actions)
    with pytest.raises(ValueError, match="repeat"):
        OfflineEpisode(directory)


def test_published_hash_is_checked_independently_of_local_file_record(checked_episode_fixture):
    directory = checked_episode_fixture()
    report_path = directory / "validation-report.json"
    report = json.loads(report_path.read_text())
    report["published_sha256"]["episode_000000.mp4"] = "0" * 64
    write_json(report_path, report)
    with pytest.raises(ValueError, match="Published checksum"):
        OfflineEpisode(directory)


@pytest.mark.parametrize("record,field", [("validation-report.json", "files"),
    ("validation-report.json", "published_sha256"), ("source-info.json", "features")])
def test_malformed_mapping_records_are_rejected(checked_episode_fixture, record, field):
    directory = checked_episode_fixture()
    path = directory / record
    data = json.loads(path.read_text())
    data[field] = []
    write_json(path, data)
    if record != "validation-report.json":
        refresh_hashes(directory)
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


def test_malformed_image_schema_is_rejected(checked_episode_fixture):
    directory = checked_episode_fixture()
    change_json(directory, "source-info.json",
                lambda data: data["features"].update({"observation.images": []}))
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


def test_binary_both_buttons_preserve_held_turn_counter(checked_episode_fixture):
    directory = checked_episode_fixture()
    actions = [[0.] * 9 for _ in range(8)]
    for row in actions:
        row[7] = 1
    for row in actions[:4]:
        row[6] = 1
    change_column(directory, "action", actions)
    np.testing.assert_array_equal(OfflineEpisode(directory).turn_targets(),
        [0., 0., 0., 0., 1.7578125, 3.515625, 3.515625, 3.515625])


def test_corruption_after_validation_is_caught_before_frame_iteration(checked_episode_fixture):
    directory = checked_episode_fixture()
    data = OfflineEpisode(directory)
    with (directory / "episode_000000.mp4").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        next(data.iter_frames())


def test_closing_generator_reaps_its_real_owned_decoder(checked_episode_fixture, monkeypatch):
    data = OfflineEpisode(checked_episode_fixture(count=100))
    owned = []
    real_popen = subprocess.Popen
    def record_decoder(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        owned.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", record_decoder)
    frames = data.iter_frames()
    next(frames)
    assert owned[0].poll() is None
    frames.close()
    assert owned[0].poll() is not None
    assert owned[0].stdout.closed
    assert owned[0].stderr.closed


def test_rotation_metadata_does_not_transform_source_pixels(checked_episode_fixture):
    directory = checked_episode_fixture()
    original = directory / "episode_000000.mp4"
    expected = subprocess.run(["ffmpeg", "-v", "error", "-noautorotate", "-i",
        str(original), "-frames:v", "1", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, capture_output=True).stdout
    rotated = directory / "rotation.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i",
                    str(original), "-c", "copy", str(rotated)], check=True)
    rotated.replace(original)
    refresh_hashes(directory)
    sample = next(OfflineEpisode(directory).iter_frames(limit=1))
    assert sample.rgb.tobytes() == expected


def test_decoder_error_closes_pipes_and_reaps_owned_process(checked_episode_fixture, monkeypatch, tmp_path):
    data = OfflineEpisode(checked_episode_fixture())
    real_popen = subprocess.Popen
    owned = []
    def failing_decoder(command, **kwargs):
        # A real FFmpeg failure exercises cleanup without altering checked data.
        command = list(command)
        command[command.index("-i") + 1] = str(tmp_path / "absent.mp4")
        process = real_popen(command, **kwargs)
        owned.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", failing_decoder)
    with pytest.raises(ValueError, match="truncated|missing"):
        next(data.iter_frames())
    assert owned[0].poll() is not None
    assert owned[0].stdout.closed
    assert owned[0].stderr.closed


def test_empty_episode_is_rejected(checked_episode_fixture):
    with pytest.raises(ValueError):
        OfflineEpisode(checked_episode_fixture(count=0))


def test_extra_episode_is_rejected(checked_episode_fixture):
    directory = checked_episode_fixture()
    shutil.copyfile(directory / "episode_000000.parquet", directory / "episode_000001.parquet")
    with pytest.raises(ValueError):
        OfflineEpisode(directory)


@pytest.mark.parametrize("scenario", ["defend_the_center", "defend_the_line"])
def test_defend_scenarios_accept_binary_conversion(checked_episode_fixture, scenario):
    data = OfflineEpisode(checked_episode_fixture(turns=[1] * 4, scenario=scenario))
    np.testing.assert_array_equal(data.turn_targets(), [1.7578125] * 4)


@pytest.mark.parametrize("column,value", [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (8, .2)])
def test_unsupported_controls_stay_raw_and_cannot_be_converted(checked_episode_fixture, column, value):
    directory = checked_episode_fixture()
    actions = [[0.] * 9 for _ in range(8)]
    for row in actions:
        row[column] = value
    change_column(directory, "action", actions)
    data = OfflineEpisode(directory)
    assert data.actions[0, column] == np.float32(value)
    with pytest.raises(ValueError):
        data.turn_targets()


def test_battle_remains_raw_compatibility_control(checked_episode_fixture):
    data = OfflineEpisode(checked_episode_fixture(scenario="battle1"))
    assert data.frame_count == 8
    with pytest.raises(ValueError, match="scenario"):
        data.turn_targets()


def test_prefix_matches_real_rgb_decode_and_closes_cleanly(checked_episode_fixture):
    directory = checked_episode_fixture(count=100)
    data = OfflineEpisode(directory)
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i",
        str(directory / "episode_000000.mp4"), "-map", "0:v:0", "-frames:v", "1",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, capture_output=True).stdout
    first = list(data.iter_frames(limit=1))
    assert len(first) == 1
    assert first[0].rgb.tobytes() == decoded
    frames = data.iter_frames()
    assert next(frames).index == 0
    frames.close()
    assert len(list(data.iter_frames(limit=2))) == 2


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_prefix_limit_must_be_positive_integer(checked_episode_fixture, limit):
    data = OfflineEpisode(checked_episode_fixture())
    with pytest.raises(ValueError):
        list(data.iter_frames(limit=limit))
