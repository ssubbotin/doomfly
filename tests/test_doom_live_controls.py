"""Contracts for the frozen Metal live Doom comparison controls."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs/experiments/2026-09-13-metal-live-transfer-protocol.json"
PROTOCOL_SHA256 = "757a424568027aea9852b80b75c50dc1b6ce2a66bb05d26e8ce53cf2497b1071"


def controls():
    """The first RED proves that the public module has not been created yet."""
    assert importlib.util.find_spec("doom_learning_v6.live_controls") is not None
    return importlib.import_module("doom_learning_v6.live_controls")


def live_protocol():
    return controls().load_live_protocol(PROTOCOL_PATH, PROTOCOL_SHA256)


def synthetic_protocol(root):
    candidates = []
    roles = ["baseline", "one", "two", "three", "four", "five", "six"]
    for offset, role in enumerate(roles):
        values = np.ones(4184, dtype=np.float64) if role == "baseline" else np.full(4184, 0.9 + offset / 100, dtype=np.float64)
        path = Path("finals") / role / "fractions.npy"
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, values)
        raw = destination.read_bytes()
        candidates.append({"role": role, "path": path.as_posix(),
                           "pin": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}})
    return {"schema": 1, "candidates": candidates}


def with_trusted_protocol(monkeypatch, value):
    module = controls()
    monkeypatch.setattr(module, "_PINNED_PROTOCOL", copy.deepcopy(value))
    return module


def observer(engine_tic, health=100, kills=0, *, finished=False, dead=False, timeout=False, tick=0):
    return {"tick": tick, "engine_tic": engine_tic, "finished": finished, "dead": dead,
            "timeout": timeout, "health": health, "kills": kills}


def live(index, before, after):
    return {"index": index, "phase": "live", "game_before": before, "game_after": after,
            "applied_action": {"turn": 0, "forward": 0, "attack": False}}


def padding(index, terminal):
    return {"index": index, "phase": "padding", "game_before": terminal.copy(),
            "game_after": terminal.copy(), "applied_action": None}


def test_loads_only_the_pinned_protocol_bytes_and_returns_fresh_records(tmp_path):
    module = controls()
    first = module.load_live_protocol(PROTOCOL_PATH, PROTOCOL_SHA256)
    first["candidates"][0]["role"] = "mutated-by-caller"
    second = module.load_live_protocol(PROTOCOL_PATH, PROTOCOL_SHA256)
    assert second["candidates"][0]["role"] == "baseline"
    with pytest.raises(ValueError):
        module.load_live_protocol(PROTOCOL_PATH, "0" * 64)

    altered = tmp_path / "protocol.json"
    altered.write_bytes(PROTOCOL_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        module.load_live_protocol(altered, PROTOCOL_SHA256)
    with pytest.raises(ValueError):
        module.load_live_protocol(altered, hashlib.sha256(altered.read_bytes()).hexdigest())


@pytest.mark.parametrize("mutate", [
    lambda value: value.__setitem__("unexpected", None),
    lambda value: value.__setitem__("learning", 1),
    lambda value: value["candidates"][0].__setitem__("role", True),
    lambda value: value["candidates"][0]["pin"].__setitem__("bytes", True),
])
def test_validation_rejects_recursive_type_aliases_and_extra_keys(mutate):
    value = live_protocol()
    mutate(value)
    with pytest.raises(ValueError):
        controls().validate_live_protocol(value)


def test_validation_accepts_the_unmodified_immutable_protocol():
    controls().validate_live_protocol(live_protocol())


class StringKeyAlias(str):
    pass


def test_validation_rejects_a_string_subclass_key_at_protocol_top_level():
    value = live_protocol()
    schema = value.pop("schema")
    value[StringKeyAlias("schema")] = schema
    with pytest.raises(ValueError):
        controls().validate_live_protocol(value)


def test_validation_rejects_a_string_subclass_key_in_a_nested_pin():
    value = live_protocol()
    pin = value["candidates"][0]["pin"]
    byte_count = pin.pop("bytes")
    pin[StringKeyAlias("bytes")] = byte_count
    with pytest.raises(ValueError):
        controls().validate_live_protocol(value)


def test_candidate_loader_checks_real_pins_and_returns_read_only_frozen_copies(tmp_path, monkeypatch):
    fixture = synthetic_protocol(tmp_path)
    module = with_trusted_protocol(monkeypatch, fixture)
    candidates = module.load_candidates(copy.deepcopy(fixture), tmp_path)
    assert set(candidates) == {"baseline", "one", "two", "three", "four", "five", "six", "baseline-filler"}
    assert candidates["baseline-filler"] is candidates["baseline"]
    assert candidates["one"].dtype == np.dtype("float64")
    assert candidates["one"].shape == (4184,)
    assert not candidates["one"].flags.writeable
    with pytest.raises(ValueError):
        candidates["one"][0] = 1.0


@pytest.mark.parametrize("break_input", ["missing", "bytes", "reordered", "dtype", "shape", "nonfinite", "bounds", "baseline", "escaping"])
def test_candidate_loader_rejects_each_invalid_vector_or_schema_input(tmp_path, monkeypatch, break_input):
    fixture = synthetic_protocol(tmp_path)
    module = controls()
    candidate = fixture["candidates"][1]
    vector_path = tmp_path / candidate["path"]
    if break_input == "missing":
        vector_path.unlink()
    elif break_input == "bytes":
        vector_path.write_bytes(vector_path.read_bytes() + b"changed")
    elif break_input == "reordered":
        fixture["candidates"][1], fixture["candidates"][2] = fixture["candidates"][2], fixture["candidates"][1]
    elif break_input == "dtype":
        np.save(vector_path, np.ones(4184, dtype=np.float32))
        raw = vector_path.read_bytes()
        candidate["pin"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    elif break_input == "shape":
        np.save(vector_path, np.ones((4184, 1), dtype=np.float64))
        raw = vector_path.read_bytes()
        candidate["pin"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    elif break_input == "nonfinite":
        np.save(vector_path, np.full(4184, np.nan, dtype=np.float64))
        raw = vector_path.read_bytes()
        candidate["pin"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    elif break_input == "bounds":
        np.save(vector_path, np.full(4184, 1.11, dtype=np.float64))
        raw = vector_path.read_bytes()
        candidate["pin"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    elif break_input == "baseline":
        baseline = fixture["candidates"][0]
        baseline_path = tmp_path / baseline["path"]
        np.save(baseline_path, np.full(4184, 1.01, dtype=np.float64))
        raw = baseline_path.read_bytes()
        baseline["pin"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    else:
        candidate["path"] = "../outside.npy"
    if break_input not in {"reordered", "escaping"}:
        monkeypatch.setattr(module, "_PINNED_PROTOCOL", copy.deepcopy(fixture))
    else:
        monkeypatch.setattr(module, "_PINNED_PROTOCOL", synthetic_protocol(tmp_path / "trusted"))
    with pytest.raises(ValueError):
        module.load_candidates(fixture, tmp_path)


def test_full_schedule_preserves_all_roles():
    waves = controls().plan_waves(live_protocol())
    assert len(waves) == 18
    assert sum(len(w["roles"]) for w in waves) == 72
    assert [w["roles"] for w in waves[:2]] == [["baseline"] * 4] * 2
    assert sum(role == "baseline-filler" for w in waves for role in w["roles"]) == 8
    assert [(wave["kind"], wave["environment"], wave["seed"], wave["hazard_left"], wave["roles"]) for wave in waves] == [
        ("calibration", "blue-floor-survival-v1", 202609135, True, ["baseline"] * 4),
        ("calibration", "defend_the_center", 202609131, None, ["baseline"] * 4),
        ("evaluation", "blue-floor-survival-v1", 202609135, True, ["baseline", "spsa-200000201", "spsa-200000202", "gaussian_es-200000201"]),
        ("evaluation", "blue-floor-survival-v1", 202609135, True, ["gaussian_es-200000202", "random-200000301", "random-200000302", "baseline-filler"]),
        ("evaluation", "blue-floor-survival-v1", 202609136, False, ["spsa-200000201", "spsa-200000202", "gaussian_es-200000201", "gaussian_es-200000202"]),
        ("evaluation", "blue-floor-survival-v1", 202609136, False, ["random-200000301", "random-200000302", "baseline", "baseline-filler"]),
        ("evaluation", "blue-floor-survival-v1", 202609137, True, ["spsa-200000202", "gaussian_es-200000201", "gaussian_es-200000202", "random-200000301"]),
        ("evaluation", "blue-floor-survival-v1", 202609137, True, ["random-200000302", "baseline", "spsa-200000201", "baseline-filler"]),
        ("evaluation", "blue-floor-survival-v1", 202609138, False, ["gaussian_es-200000201", "gaussian_es-200000202", "random-200000301", "random-200000302"]),
        ("evaluation", "blue-floor-survival-v1", 202609138, False, ["baseline", "spsa-200000201", "spsa-200000202", "baseline-filler"]),
        ("evaluation", "defend_the_center", 202609131, None, ["baseline", "spsa-200000201", "spsa-200000202", "gaussian_es-200000201"]),
        ("evaluation", "defend_the_center", 202609131, None, ["gaussian_es-200000202", "random-200000301", "random-200000302", "baseline-filler"]),
        ("evaluation", "defend_the_center", 202609132, None, ["spsa-200000201", "spsa-200000202", "gaussian_es-200000201", "gaussian_es-200000202"]),
        ("evaluation", "defend_the_center", 202609132, None, ["random-200000301", "random-200000302", "baseline", "baseline-filler"]),
        ("evaluation", "defend_the_center", 202609133, None, ["spsa-200000202", "gaussian_es-200000201", "gaussian_es-200000202", "random-200000301"]),
        ("evaluation", "defend_the_center", 202609133, None, ["random-200000302", "baseline", "spsa-200000201", "baseline-filler"]),
        ("evaluation", "defend_the_center", 202609134, None, ["gaussian_es-200000201", "gaussian_es-200000202", "random-200000301", "random-200000302"]),
        ("evaluation", "defend_the_center", 202609134, None, ["baseline", "spsa-200000201", "spsa-200000202", "baseline-filler"]),
    ]
    assert [wave["ordinal"] for wave in waves] == list(range(18))


def test_metrics_use_engine_differences_at_nonzero_origins_and_stop_at_death():
    before = observer(12, tick=4)
    middle = observer(13, health=80, kills=1, tick=5)
    terminal = observer(14, health=70, kills=2, finished=True, dead=True, tick=6)
    trace = [live(0, before, middle), live(1, middle, terminal), padding(2, terminal), padding(3, terminal)]
    result = controls().gameplay_metrics(trace, 4)
    assert result == {"live_game_tics": 2, "restricted_survival_seconds": 2 / 35, "died": True,
                      "right_censored": False, "censor_reason": None, "kills_at_fixed_horizon": 2,
                      "damage_total": 30, "final_health": 70}


@pytest.mark.parametrize("origin", [1, 301])
def test_metrics_do_not_assume_an_absolute_engine_tic_zero(origin):
    first = observer(origin, health=50, tick=20)
    second = observer(origin + 1, health=50, tick=21)
    result = controls().gameplay_metrics([live(0, first, second)], 1)
    assert result["live_game_tics"] == 1
    assert result["restricted_survival_seconds"] == 1 / 35
    assert result["right_censored"] is True
    assert result["censor_reason"] == "horizon"


@pytest.mark.parametrize(("terminal", "expected_reason"), [
    (dict(timeout=True), "timeout"),
    ({}, "other-finished"),
])
def test_metrics_mark_living_terminal_states_as_right_censored(terminal, expected_reason):
    before = observer(40, health=25)
    after = observer(41, health=25, finished=True, **terminal)
    result = controls().gameplay_metrics([live(0, before, after), padding(1, after)], 2)
    assert result["died"] is False
    assert result["right_censored"] is True
    assert result["censor_reason"] == expected_reason


def test_metrics_accumulate_damage_only_across_health_losses_and_keep_cumulative_kills():
    one = observer(70, health=100, kills=0)
    two = observer(71, health=90, kills=1)
    three = observer(72, health=96, kills=3)
    four = observer(73, health=80, kills=3)
    five = observer(74, health=80, kills=4)
    result = controls().gameplay_metrics([live(0, one, two), live(1, two, three), live(2, three, four), live(3, four, five)], 4)
    assert result["damage_total"] == 26
    assert result["kills_at_fixed_horizon"] == 4
    assert result["final_health"] == 80


@pytest.mark.parametrize("break_trace", ["incomplete", "skip_tic", "action_after_terminal", "padding_action"])
def test_metrics_reject_incomplete_or_invalid_terminal_timing(break_trace):
    before = observer(100)
    terminal = observer(101, finished=True, dead=True)
    trace = [live(0, before, terminal), padding(1, terminal)]
    if break_trace == "incomplete":
        trace = trace[:1]
    elif break_trace == "skip_tic":
        trace = [live(0, before, observer(102)), live(1, observer(102), observer(103))]
    elif break_trace == "action_after_terminal":
        trace[1] = live(1, terminal, observer(102, finished=True, dead=True))
    else:
        trace[1]["applied_action"] = {}
    with pytest.raises(ValueError):
        controls().gameplay_metrics(trace, 2)
