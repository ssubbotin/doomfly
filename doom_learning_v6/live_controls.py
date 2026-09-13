"""Pure contracts for the frozen Metal live Doom comparison."""
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np


LIVE_PROTOCOL_SHA256 = "757a424568027aea9852b80b75c50dc1b6ce2a66bb05d26e8ce53cf2497b1071"
_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "docs/experiments/2026-09-13-metal-live-transfer-protocol.json"
_PINNED_BYTES = _PROTOCOL_PATH.read_bytes()
if hashlib.sha256(_PINNED_BYTES).hexdigest() != LIVE_PROTOCOL_SHA256:
    raise RuntimeError("The live transfer protocol bytes do not match their immutable pin")
_PINNED_PROTOCOL = json.loads(_PINNED_BYTES.decode("utf-8"))


def _exact_json(value, reference, location="protocol"):
    if type(value) is not type(reference):
        raise ValueError(f"{location} has the wrong JSON type")
    if type(reference) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError(f"{location} has a non-builtin string key")
        if set(value) != set(reference):
            raise ValueError(f"{location} has different keys")
        for key in reference:
            _exact_json(value[key], reference[key], f"{location}.{key}")
    elif type(reference) is list:
        if len(value) != len(reference):
            raise ValueError(f"{location} has the wrong length")
        for index, (item, expected) in enumerate(zip(value, reference)):
            _exact_json(item, expected, f"{location}[{index}]")
    elif value != reference:
        raise ValueError(f"{location} differs from the immutable protocol")


def validate_live_protocol(protocol) -> None:
    """Require the complete, exact JSON record committed with this evaluator."""
    _exact_json(protocol, _PINNED_PROTOCOL)


def load_live_protocol(path, expected_sha256) -> dict:
    """Read a fresh copy only when its bytes and immutable pin agree."""
    if type(expected_sha256) is not str or expected_sha256 != LIVE_PROTOCOL_SHA256:
        raise ValueError("Unexpected live protocol SHA256")
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError, ValueError) as failure:
        raise ValueError("Unable to read the live protocol") from failure
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Live protocol bytes do not match SHA256")
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ValueError("Live protocol is not JSON") from failure
    validate_live_protocol(record)
    return record


def _candidate_path(root, relative):
    if type(relative) is not str:
        raise ValueError("Candidate path is not a string")
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("Candidate path is absolute")
    try:
        root = Path(root).resolve()
        resolved = (root / path).resolve()
        resolved.relative_to(root)
    except (OSError, TypeError, ValueError) as failure:
        raise ValueError("Candidate path escapes its root") from failure
    return resolved


def load_candidates(protocol, root) -> dict[str, np.ndarray]:
    """Load every pinned efficacy vector as an independent frozen array."""
    validate_live_protocol(protocol)
    result = {}
    for candidate in protocol["candidates"]:
        location = _candidate_path(root, candidate["path"])
        try:
            raw = location.read_bytes()
        except OSError as failure:
            raise ValueError(f"Unable to read candidate {candidate['role']}") from failure
        pin = candidate["pin"]
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError(f"Candidate {candidate['role']} does not match its pin")
        try:
            values = np.load(BytesIO(raw), allow_pickle=False)
        except (OSError, ValueError) as failure:
            raise ValueError(f"Candidate {candidate['role']} is not a valid NumPy array") from failure
        if type(values) is not np.ndarray or values.dtype != np.dtype(np.float64) or values.shape != (4184,):
            raise ValueError(f"Candidate {candidate['role']} has the wrong dtype or shape")
        if not np.isfinite(values).all() or (values < 0.9).any() or (values > 1.1).any():
            raise ValueError(f"Candidate {candidate['role']} has invalid fractions")
        if candidate["role"] == "baseline" and not np.array_equal(values, np.ones(4184, dtype=np.float64)):
            raise ValueError("The baseline candidate must contain only one")
        frozen = values.copy()
        frozen.setflags(write=False)
        result[candidate["role"]] = frozen
    if set(result) != {candidate["role"] for candidate in protocol["candidates"]}:
        raise ValueError("Candidate roles are not unique")
    result["baseline-filler"] = result["baseline"]
    return result


def plan_waves(protocol) -> list[dict]:
    """Return the fixed calibration and role-rotated evaluation sequence."""
    validate_live_protocol(protocol)
    waves = []
    for environment in protocol["environments"]:
        seed = next(row for row in environment["seeds"] if row["seed"] == environment["baseline_check_seed"])
        waves.append({"ordinal": len(waves), "kind": "calibration", "environment": environment["scenario"],
                      "seed": seed["seed"], "hazard_left": seed.get("hazard_left"), "roles": ["baseline"] * 4})
    original = [candidate["role"] for candidate in protocol["candidates"]]
    for environment in protocol["environments"]:
        for ordinal, seed in enumerate(environment["seeds"]):
            offset = ordinal % len(original)
            rotated = original[offset:] + original[:offset]
            for roles in (rotated[:4], rotated[4:] + ["baseline-filler"]):
                waves.append({"ordinal": len(waves), "kind": "evaluation", "environment": environment["scenario"],
                              "seed": seed["seed"], "hazard_left": seed.get("hazard_left"), "roles": roles})
    return waves


_OBSERVER_FIELDS = ("tick", "engine_tic", "finished", "dead", "timeout", "health", "kills")


def _integer(value):
    return type(value) is int


def _number(value):
    return type(value) in (int, float) and np.isfinite(value)


def _observer(value):
    if type(value) is not dict or not all(key in value for key in _OBSERVER_FIELDS):
        raise ValueError("Trace observer is incomplete")
    if not _integer(value["tick"]) or not _integer(value["engine_tic"]) or not _integer(value["kills"]):
        raise ValueError("Trace observer has invalid tic or kill fields")
    if any(type(value[key]) is not bool for key in ("finished", "dead", "timeout")):
        raise ValueError("Trace observer has invalid terminal fields")
    if not _number(value["health"]):
        raise ValueError("Trace observer has invalid health")
    if (value["dead"] or value["timeout"]) and not value["finished"]:
        raise ValueError("Trace observer has an unterminated terminal flag")
    return value


def _same_observer(left, right):
    return all(left[key] == right[key] for key in _OBSERVER_FIELDS)


def gameplay_metrics(trace, horizon_tics) -> dict:
    """Compute gameplay clocks and terminal labels from a complete lane trace."""
    if type(trace) is not list or not _integer(horizon_tics) or horizon_tics <= 0 or len(trace) != horizon_tics:
        raise ValueError("Trace must contain the complete positive horizon")
    previous_after = None
    terminal = None
    live_game_tics = 0
    damage_total = 0
    for index, row in enumerate(trace):
        if type(row) is not dict or set(("index", "phase", "game_before", "game_after", "applied_action")) - set(row):
            raise ValueError("Trace row is incomplete")
        if type(row["index"]) is not int or row["index"] != index or type(row["phase"]) is not str:
            raise ValueError("Trace row index or phase is invalid")
        before = _observer(row["game_before"])
        after = _observer(row["game_after"])
        if previous_after is not None and not _same_observer(before, previous_after):
            raise ValueError("Trace observers are not contiguous")
        if row["phase"] == "live":
            if terminal is not None or before["finished"] or type(row["applied_action"]) is not dict:
                raise ValueError("Live action follows game termination")
            if after["engine_tic"] - before["engine_tic"] != 1:
                raise ValueError("A live game action must advance one engine tic")
            if after["kills"] < before["kills"]:
                raise ValueError("Cumulative kills cannot decrease")
            live_game_tics += after["engine_tic"] - before["engine_tic"]
            damage_total += max(0, before["health"] - after["health"])
            if after["finished"]:
                terminal = after
        elif row["phase"] == "padding":
            if row["applied_action"] is not None:
                raise ValueError("Padding cannot apply a game action")
            if terminal is None:
                if not before["finished"] or not _same_observer(before, after):
                    raise ValueError("Padding requires a terminal observer state")
                terminal = before
            if not _same_observer(before, terminal) or not _same_observer(after, terminal):
                raise ValueError("Padding must retain terminal observer state")
        else:
            raise ValueError("Trace has an unknown phase")
        previous_after = after
    final = previous_after
    if terminal is not None and terminal["dead"]:
        died, right_censored, censor_reason = True, False, None
    elif terminal is not None and terminal["timeout"]:
        died, right_censored, censor_reason = False, True, "timeout"
    elif terminal is not None:
        died, right_censored, censor_reason = False, True, "other-finished"
    else:
        died, right_censored, censor_reason = False, True, "horizon"
    return {"live_game_tics": live_game_tics, "restricted_survival_seconds": live_game_tics / 35,
            "died": died, "right_censored": right_censored, "censor_reason": censor_reason,
            "kills_at_fixed_horizon": final["kills"], "damage_total": damage_total, "final_health": final["health"]}
