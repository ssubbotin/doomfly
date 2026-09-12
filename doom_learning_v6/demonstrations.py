"""Checked GameWAM artifacts with RGB and original controls kept separate.

Checks establish artifact alignment, not the unknown original engine's turn
physics. Target conversion uses the separately measured local calibration.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_DATASET = "Yunncheng/gamewam-vizdoom"
_REVISION = "6cc00d60462885c1d61dd480228af58c3b81b807"
_ACTION_NAMES = ["shoot", "move_forward", "move_backward", "strafe_left",
                 "strafe_right", "speed", "turn_left", "turn_right", "turn_delta"]
_COLUMNS = ["action", "timestamp", "frame_index", "episode_index", "index"]


@dataclass(frozen=True)
class OfflineFrame:
    index: int
    timestamp: float
    rgb: np.ndarray
    action: np.ndarray


def _json(path):
    try:
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError("record must be an object")
        return result
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid artifact record {path.name}: {error}") from error


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"Cannot check artifact {path.name}: {error}") from error
    return digest.hexdigest()


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


class OfflineEpisode:
    """Validate one complete artifact before any frame can reach training."""

    def __init__(self, directory):
        self._directory = Path(directory)
        report = _json(self._directory / "validation-report.json")
        if report.get("dataset") != _DATASET or report.get("dataset_revision") != _REVISION:
            raise ValueError("Unsupported dataset or GameWAM revision")
        episode_index = report.get("episode_index")
        if not _integer(episode_index) or not 0 <= episode_index <= 999999:
            raise ValueError("Invalid episode index")
        stem = f"episode_{episode_index:06d}"
        video_name, control_name = stem + ".mp4", stem + ".parquet"
        for suffix, expected in [("mp4", video_name), ("parquet", control_name)]:
            if sorted(path.name for path in self._directory.glob(f"episode_*.{suffix}")) != [expected]:
                raise ValueError("Require exactly one matching episode video and control file")
        self._video = self._directory / video_name
        files = report.get("files", {})
        published = report.get("published_sha256", {})
        if not isinstance(files, dict) or not isinstance(published, dict):
            raise ValueError("Checksum records must be objects")
        checked_hashes = {}
        for name in ["SOURCE-README.md", "source-info.json", "episode-metadata.json",
                     video_name, control_name]:
            record = files.get(name)
            if not isinstance(record, dict):
                raise ValueError(f"Missing recorded checksum for {name}")
            path = self._directory / name
            digest = _sha256(path)
            if digest != record.get("sha256"):
                raise ValueError(f"Artifact checksum mismatch: {name}")
            if not _integer(record.get("bytes")) or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Artifact bytes mismatch: {name}")
            if name in [video_name, control_name] and digest != published.get(name):
                raise ValueError(f"Published checksum mismatch: {name}")
            checked_hashes[name] = digest

        source = _json(self._directory / "source-info.json")
        metadata = _json(self._directory / "episode-metadata.json")
        scenario = metadata.get("scenario")
        if not isinstance(scenario, str) or not scenario or scenario != report.get("scenario"):
            raise ValueError("Scenario metadata mismatch")
        if metadata.get("episode_index") != episode_index or not _integer(metadata.get("episode_index")):
            raise ValueError("Episode metadata mismatch")
        length = metadata.get("length")
        if not _integer(length) or length <= 0:
            raise ValueError("Episode must contain a positive frame count")
        expected_metadata = {
            "raw_file_name": f"{scenario}_{episode_index:06d}",
            "policy": "sample_factory_appo_rnn", "capture_mode": "tick_smooth_native",
            "capture_frame_skip": 1, "policy_action_repeat": 4, "humanized": False}
        for key, expected in expected_metadata.items():
            if type(metadata.get(key)) is not type(expected) or metadata[key] != expected:
                raise ValueError(f"Unsupported episode metadata: {key}")
        self.frame_count = length
        self._check_source(source)

        try:
            table = pq.read_table(self._directory / control_name, columns=_COLUMNS)
        except (OSError, pa.ArrowException) as error:
            raise ValueError(f"Invalid controls Parquet: {error}") from error
        if table.num_rows != length:
            raise ValueError("Control row count differs from metadata frame count")
        types = {"action": pa.list_(pa.float32(), 9), "timestamp": pa.float32(),
                 "frame_index": pa.int64(), "episode_index": pa.int64(), "index": pa.int64()}
        for name, dtype in types.items():
            if table[name].type != dtype or table[name].null_count:
                raise ValueError(f"Invalid original control column: {name}")
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        timestamps = table["timestamp"].to_numpy()
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("Nonfinite controls or timestamps")
        if not np.all((actions[:, :8] == 0) | (actions[:, :8] == 1)):
            raise ValueError("First eight actions must be binary flags")
        if not np.all(np.abs(actions[:, 8]) <= 1):
            raise ValueError("Continuous turn action outside [-1,1]")
        frame_indices = table["frame_index"].to_numpy()
        original_indices = table["index"].to_numpy()
        if not np.array_equal(frame_indices, np.arange(length)):
            raise ValueError("Noncontiguous or reordered original frame indices")
        if np.any(original_indices < 0) or not np.all(np.diff(original_indices) == 1):
            raise ValueError("Noncontiguous original source indices")
        if not np.all(table["episode_index"].to_numpy() == episode_index):
            raise ValueError("Controls contain another episode")
        expected_times = (np.arange(length, dtype=np.float64) / 35).astype(np.float32)
        if not np.array_equal(timestamps, expected_times):
            raise ValueError("Control timestamps differ from float32-rounded tick times")
        for start in range(0, length, 4):
            if not np.all(actions[start:start + 4] == actions[start]):
                raise ValueError("Actions violate recorded four-tick policy repeat")
        # A bytes-backed view cannot be made writable, including by callers.
        self.actions = np.frombuffer(actions.tobytes(), dtype=np.float32).reshape(length, 9)
        self._timestamps = np.frombuffer(timestamps.tobytes(), dtype=np.float32)
        self.identity = {"dataset": _DATASET, "dataset_revision": _REVISION,
                         "scenario": scenario, "episode_index": episode_index,
                         "video_sha256": checked_hashes[video_name],
                         "control_sha256": checked_hashes[control_name]}
        self._video_sha256 = checked_hashes[video_name]
        self._check_video()

    def _check_source(self, source):
        if source.get("fps") != 35 or source.get("robot_type") != "vizdoom_appo":
            raise ValueError("Unsupported source capture rate or robot type")
        features = source.get("features", {})
        expected = {"action": {"dtype": "float32", "shape": [9], "names": _ACTION_NAMES},
                    "timestamp": {"dtype": "float32", "shape": [1], "names": None}}
        for name in ["frame_index", "episode_index", "index"]:
            expected[name] = {"dtype": "int64", "shape": [1], "names": None}
        if not isinstance(features, dict) or any(features.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported source action names, indices or timing schema")
        image = features.get("observation.images", {})
        if not isinstance(image, dict):
            raise ValueError("Source image schema must be an object")
        shape = image.get("shape", [])
        if (not isinstance(shape, list) or len(shape) != 3 or
                any(not _integer(value) for value in shape) or shape[0] != 3):
            raise ValueError("Invalid source RGB dimensions")
        _, height, width = shape
        if not (0 < width <= 4096 and 0 < height <= 4096 and width * height <= 8388608):
            raise ValueError("Unreasonable source RGB dimensions")
        info = image.get("info", {})
        expected_info = {"video.height": height, "video.width": width,
                         "video.channels": 3, "video.fps": 35, "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                         "has_audio": False}
        if (image.get("dtype") != "video" or image.get("names") != ["channels", "height", "width"] or
                not isinstance(info, dict) or any(type(info.get(key)) is not type(value) or
                info[key] != value for key, value in expected_info.items())):
            raise ValueError("Source image schema metadata mismatch")
        self.width, self.height = width, height

    def _check_video(self):
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v", "-show_streams",
                "-show_entries",
                "stream=width,height,codec_name,pix_fmt,time_base,avg_frame_rate,nb_frames",
                "-of", "json", str(self._video)], capture_output=True, check=True)
            if result.stderr:
                raise ValueError("Video stream probing reported errors")
            probe = json.loads(result.stdout)
            streams = probe["streams"]
            if len(streams) != 1:
                raise ValueError("Require one video stream")
            stream = streams[0]
            width, height = stream["width"], stream["height"]
            if (not _integer(width) or not _integer(height) or
                    not (0 < width <= 4096 and 0 < height <= 4096 and width * height <= 8388608)):
                raise ValueError("Unreasonable actual video dimensions")
            if (width != self.width or height != self.height or
                    stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"):
                raise ValueError("Video dimensions or format differ from source schema")
            if Fraction(stream["avg_frame_rate"]) != 35:
                raise ValueError("Video rate differs from 35 Hz tick rate")
            if int(stream["nb_frames"]) != self.frame_count:
                raise ValueError("Video frame count differs from original controls")
            time_base = Fraction(stream["time_base"])
            if time_base <= 0:
                raise ValueError("Video time base must be positive")
            # Validate actual stream geometry before requesting frame decoding.
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=best_effort_timestamp", "-of", "json",
                str(self._video)], capture_output=True, check=True)
            if result.stderr:
                raise ValueError("Video decoding reported errors")
            frames = json.loads(result.stdout)["frames"]
            if len(frames) != self.frame_count:
                raise ValueError("Decoded video frame count differs from original controls")
            if any(
                Fraction(frame["best_effort_timestamp"]) * time_base != Fraction(index, 35)
                for index, frame in enumerate(frames)):
                raise ValueError("Video PTS differs from exact rational tick timestamps")
        except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ZeroDivisionError,
                json.JSONDecodeError) as error:
            raise ValueError(f"Invalid video artifact: {error}") from error

    def turn_targets(self):
        """Convert compatible defend controls using the local held-turn assay."""
        if self.identity["scenario"] not in ["defend_the_center", "defend_the_line"]:
            raise ValueError("Unsupported scenario for binary turn target conversion")
        if np.any(self.actions[:, [1, 2, 3, 4, 5, 8]]):
            raise ValueError("Movement, strafe, speed and continuous turn cannot be converted")
        targets = np.empty(self.frame_count, dtype=np.float64)
        held = 0
        for index, action in enumerate(self.actions):
            held = held + 1 if action[6] or action[7] else 0
            units = 320 if held < 6 else 640
            targets[index] = (float(action[7]) - float(action[6])) * units * 360 / 65536
        return targets

    def iter_frames(self, limit=None):
        """Yield untransformed RGB24 at original timing; own only this decoder."""
        if limit is not None and (not _integer(limit) or limit <= 0):
            raise ValueError("Frame prefix limit must be a positive integer")
        count = self.frame_count if limit is None else min(limit, self.frame_count)
        if _sha256(self._video) != self._video_sha256:
            raise ValueError("Video checksum changed since episode validation")
        command = ["ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(self._video),
                   "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "pipe:1"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = self.width * self.height * 3
        complete = False
        try:
            for index in range(count):
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise ValueError("RGB decoder returned a truncated frame or missing frames")
                yield OfflineFrame(index=index, timestamp=float(self._timestamps[index]),
                                   rgb=np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3),
                                   action=self.actions[index])
            if count == self.frame_count:
                if process.stdout.read(1):
                    raise ValueError("RGB decoder returned extra frames")
                diagnostic = process.stderr.read()
                if process.wait() != 0 or diagnostic:
                    raise ValueError("RGB decoding failed")
                complete = True
        finally:
            if not complete and process.poll() is None:
                process.terminate()
            process.stdout.close()
            process.stderr.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
