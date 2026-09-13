"""Pure numerical helpers for the frozen same-slot fitting protocol."""

from __future__ import annotations

import hashlib
import json
import numbers
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_PINNED_PROTOCOL_SHA256 = "fae8309b51cbc24b63b23fcfaf9c4a5e5f5d141a0ce060bee3d2913821fe9c03"
_METHOD_INDEX = {"spsa": 0, "gaussian_es": 1}


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_scalar(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _vector(value: object, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if raw.dtype.kind == "b" or result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _method_index(method: object) -> int:
    if not isinstance(method, str) or method not in _METHOD_INDEX:
        raise ValueError("method must be 'spsa' or 'gaussian_es'")
    return _METHOD_INDEX[method]


def _validate_protocol(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("protocol must be a JSON object")
    required = {
        "schema": 1,
        "kind": "frozen_same_slot_numerical_fitting",
        "connectome": "MaleCNS v1.0",
        "neurons": 166700,
        "retained_edges": 25582938,
        "plastic_slots": 4184,
        "methods": ["spsa", "gaussian_es"],
        "optimizer_seeds": [200000201, 200000202],
        "generator": "numpy.random.PCG64",
        "stream_derivation": "SeedSequence([replica_seed, method_index]); spsa=0, gaussian_es=1",
        "generations": 4,
        "probe_scales": [0.5, 0.35, 0.25, 0.2],
        "step_lengths": [0.2, 0.15, 0.1, 0.1],
        "fraction_mapping": "1 + 0.1*tanh(theta)",
        "fraction_bounds": [0.1, 2],
        "acceptance": "strict lower training bank mean; ties keep incumbent; probes not eligible",
        "learning": False,
        "frozen": True,
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("protocol does not match the committed fitting structure")
    budget = record.get("budget")
    if not isinstance(budget, dict) or budget.get("planned") != 120 or budget.get("reserved") != 8 or budget.get("maximum") != 128:
        raise ValueError("protocol budget does not match the committed fitting structure")
    episodes = record.get("episodes")
    if not isinstance(episodes, list) or [episode.get("index") if isinstance(episode, dict) else None for episode in episodes] != [2, 3, 4, 5]:
        raise ValueError("protocol episodes do not match the committed fitting structure")
    return record


def load_protocol(path: str | Path, expected_sha256: str) -> dict:
    """Read the exact, pinned optimizer protocol into a fresh JSON record."""
    if not isinstance(expected_sha256, str) or expected_sha256 != _PINNED_PROTOCOL_SHA256:
        raise ValueError("expected SHA256 must match the pinned protocol")
    try:
        payload = Path(path).read_bytes()
    except (OSError, TypeError) as error:
        raise ValueError("protocol path cannot be read") from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _PINNED_PROTOCOL_SHA256:
        raise ValueError("protocol bytes do not match the pinned SHA256")
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("protocol is not valid JSON") from error
    return _validate_protocol(record)


def fractions(theta: object) -> np.ndarray:
    """Map finite latent efficacies to the protocol's fraction neighborhood."""
    latent = _vector(theta, "theta")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.asarray(1.0 + 0.1 * np.tanh(latent), dtype=np.float64)
        except FloatingPointError as error:
            raise ValueError("fraction mapping is not finite") from error
    if not np.all(np.isfinite(result)) or np.any(result < 0.1) or np.any(result > 2.0):
        raise ValueError("fraction mapping is outside the permitted bounds")
    return result


def optimizer_rng(method: object, seed: object) -> np.random.Generator:
    """Create the declared independent PCG64 stream for one method and replica."""
    method_index = _method_index(method)
    replica_seed = _integer(seed, "seed")
    try:
        sequence = np.random.SeedSequence([replica_seed, method_index])
    except ValueError as error:
        raise ValueError("seed is outside the supported range") from error
    return np.random.Generator(np.random.PCG64(sequence))


def draw_direction(method: object, rng: object, slots: object) -> np.ndarray:
    """Draw one float64 SPSA or Gaussian-ES direction from its method stream."""
    _method_index(method)
    count = _integer(slots, "slots", positive=True)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    if method == "spsa":
        return (2.0 * rng.integers(0, 2, size=count) - 1.0).astype(np.float64)
    return np.asarray(rng.standard_normal(size=count), dtype=np.float64)


def propose(
    theta: object,
    direction: object,
    plus_loss: object,
    minus_loss: object,
    *,
    probe_scale: object,
    step_length: object,
) -> tuple[np.ndarray, dict]:
    """Form a normalized paired-loss update without clipping or adaptation."""
    current = _vector(theta, "theta")
    perturbation = _vector(direction, "direction")
    if current.shape != perturbation.shape:
        raise ValueError("theta and direction must have equal shape")
    plus = _finite_scalar(plus_loss, "plus_loss", nonnegative=True)
    minus = _finite_scalar(minus_loss, "minus_loss", nonnegative=True)
    scale = _finite_scalar(probe_scale, "probe_scale", positive=True)
    length = _finite_scalar(step_length, "step_length", positive=True)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            contrast = plus - minus
            gradient = contrast * perturbation / (2.0 * scale)
            gradient_norm = float(np.max(np.abs(gradient)))
            if not np.isfinite(gradient_norm):
                raise FloatingPointError
            zero_update = gradient_norm == 0.0
            candidate = current.copy() if zero_update else current - length * gradient / gradient_norm
    except FloatingPointError as error:
        raise ValueError("proposal arithmetic is not finite") from error
    if not np.all(np.isfinite(candidate)):
        raise ValueError("proposal arithmetic is not finite")
    return np.asarray(candidate, dtype=np.float64), {
        "contrast": float(contrast),
        "gradient_norm": gradient_norm,
        "zero_update": zero_update,
    }


def training_mean(losses: object) -> float:
    """Return the equal mean of the two required finite training losses."""
    try:
        values = tuple(losses)
    except TypeError as error:
        raise ValueError("losses must contain exactly two values") from error
    if len(values) != 2:
        raise ValueError("losses must contain exactly two values")
    first = _finite_scalar(values[0], "loss", nonnegative=True)
    second = _finite_scalar(values[1], "loss", nonnegative=True)
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean = (first + second) / 2.0
    except FloatingPointError as error:
        raise ValueError("training mean is not finite") from error
    if not np.isfinite(mean):
        raise ValueError("training mean is not finite")
    return float(mean)


def accept_proposal(incumbent_loss: object, proposal_loss: object) -> bool:
    """Accept only a strict improvement over the cached training mean."""
    incumbent = _finite_scalar(incumbent_loss, "incumbent_loss", nonnegative=True)
    proposal = _finite_scalar(proposal_loss, "proposal_loss", nonnegative=True)
    return proposal < incumbent


@dataclass
class AttemptBudget:
    """Monotonic accounting for full-graph lane attempts."""

    maximum: int = 128
    used: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.maximum = _integer(self.maximum, "maximum", positive=True)

    @property
    def remaining(self) -> int:
        return self.maximum - self.used

    def charge(self, lanes: object = 4, *, reserve_after: object = 0) -> int:
        count = _integer(lanes, "lanes", positive=True)
        reserve = _integer(reserve_after, "reserve_after")
        if count > self.remaining or reserve > self.remaining - count:
            raise ValueError("attempt charge exceeds available capacity")
        self.used += count
        return self.used
