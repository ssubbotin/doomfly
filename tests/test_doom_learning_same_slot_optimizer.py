import hashlib
from pathlib import Path

import numpy as np
import pytest

from doom_learning_v6.same_slot_optimizer import (
    AttemptBudget,
    accept_proposal,
    draw_direction,
    fractions,
    load_protocol,
    optimizer_rng,
    propose,
    training_mean,
)


PROTOCOL = Path("docs/experiments/2026-09-13-same-slot-optimizer-protocol.json")
SHA256 = "fae8309b51cbc24b63b23fcfaf9c4a5e5f5d141a0ce060bee3d2913821fe9c03"


def test_gaussian_direction_is_multiplied():
    theta, record = propose(
        np.zeros(2), np.array([0.5, 2.0]), 4.0, 2.0,
        probe_scale=0.5, step_length=0.2,
    )
    np.testing.assert_allclose(theta, [-0.05, -0.2])
    assert record["contrast"] == 2.0
    assert record["gradient_norm"] == 4.0
    assert record["zero_update"] is False


def test_exact_zero_keeps_candidate():
    theta, record = propose(
        np.array([0.1, -0.2]), np.array([1.0, -1.0]), 3.0, 3.0,
        probe_scale=0.5, step_length=0.2,
    )
    np.testing.assert_array_equal(theta, [0.1, -0.2])
    assert record["contrast"] == 0.0
    assert record["gradient_norm"] == 0.0
    assert record["zero_update"] is True


def test_fraction_mapping_is_finite_at_large_latent_values():
    actual = fractions(np.array([-1.0e300, 0.0, 1.0e300]))
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [0.9, 1.0, 1.1])
    assert np.all((0.1 <= actual) & (actual <= 2.0))


@pytest.mark.parametrize(
"theta", [np.array([]), np.array([np.nan]), np.array([np.inf]), np.array([[0.0]])],
)
def test_fraction_mapping_rejects_invalid_latent_vector(theta):
    with pytest.raises(ValueError):
        fractions(theta)


@pytest.mark.parametrize(
"theta,direction,plus_loss,minus_loss,probe_scale,step_length",
[
    (np.array([]), np.array([]), 2.0, 1.0, 0.5, 0.2),
    (np.array([0.0]), np.array([1.0, 2.0]), 2.0, 1.0, 0.5, 0.2),
    (np.array([np.nan]), np.array([1.0]), 2.0, 1.0, 0.5, 0.2),
    (np.array([0.0]), np.array([np.inf]), 2.0, 1.0, 0.5, 0.2),
    (np.array([0.0]), np.array([1.0]), -1.0, 1.0, 0.5, 0.2),
    (np.array([0.0]), np.array([1.0]), 2.0, np.nan, 0.5, 0.2),
    (np.array([0.0]), np.array([1.0]), 2.0, 1.0, 0.0, 0.2),
    (np.array([0.0]), np.array([1.0]), 2.0, 1.0, 0.5, -0.2),
],
)
def test_propose_rejects_invalid_inputs(
    theta, direction, plus_loss, minus_loss, probe_scale, step_length,
):
    with pytest.raises(ValueError):
        propose(
            theta, direction, plus_loss, minus_loss,
            probe_scale=probe_scale, step_length=step_length,
        )


def test_propose_rejects_overflowing_gradient():
    with pytest.raises(ValueError):
        propose(
            np.array([0.0]), np.array([np.finfo(np.float64).max]),
            np.finfo(np.float64).max, 0.0, probe_scale=0.5, step_length=0.2,
        )


def test_propose_rejects_overflowing_denominator():
    with pytest.raises(ValueError):
        propose(
            np.array([0.0]), np.array([1.0]), 2.0, 1.0,
            probe_scale=float(np.finfo(np.float64).max), step_length=0.2,
        )


def test_training_mean_requires_two_finite_episode_losses():
    assert training_mean([2.0, 4.0]) == 3.0
    for losses in ([], [1.0], [1.0, 2.0, 3.0], [np.nan, 1.0], [-1.0, 1.0]):
        with pytest.raises(ValueError):
            training_mean(losses)


def test_proposals_require_strict_lower_loss():
    assert accept_proposal(2.0, 1.999) is True
    assert accept_proposal(2.0, 2.0) is False
    assert accept_proposal(2.0, 2.001) is False
    for incumbent, proposal in ((np.nan, 1.0), (1.0, np.inf), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            accept_proposal(incumbent, proposal)


def test_method_rng_streams_are_reproducible_and_independent():
    spsa_first = draw_direction("spsa", optimizer_rng("spsa", 200000201), 12)
    spsa_second = draw_direction("spsa", optimizer_rng("spsa", 200000201), 12)
    gaussian = draw_direction("gaussian_es", optimizer_rng("gaussian_es", 200000201), 12)
    np.testing.assert_array_equal(spsa_first, spsa_second)
    assert set(spsa_first) == {-1.0, 1.0}
    assert gaussian.dtype == np.float64
    assert not np.array_equal(spsa_first, gaussian)


def test_rng_uses_declared_pcg64_seed_sequence():
    actual = optimizer_rng("gaussian_es", 200000202)
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([200000202, 1]))
    )
    assert type(actual.bit_generator) is np.random.PCG64
    np.testing.assert_array_equal(actual.random(6), expected.random(6))


@pytest.mark.parametrize("method", ["unknown", "SPSA", 1])
def test_direction_rejects_unknown_method(method):
    with pytest.raises(ValueError):
        optimizer_rng(method, 200000201)


@pytest.mark.parametrize("slots", [0, -1, True, 1.5])
def test_direction_rejects_invalid_slot_count(slots):
    with pytest.raises(ValueError):
        draw_direction("spsa", optimizer_rng("spsa", 200000201), slots)


def test_budget_preserves_held_capacity():
    budget = AttemptBudget()
    assert budget.charge(112, reserve_after=16) == 112
    with pytest.raises(ValueError):
        budget.charge(4, reserve_after=16)
    assert budget.used == 112
    assert budget.remaining == 16


def test_budget_keeps_final_eight_attempts_after_120_charges():
    budget = AttemptBudget()
    assert budget.charge(120, reserve_after=8) == 120
    assert budget.used == 120
    assert budget.remaining == 8
    with pytest.raises(ValueError):
        budget.charge(4, reserve_after=8)
    assert budget.used == 120


def test_budget_rejects_direct_over_capacity_without_mutating():
    budget = AttemptBudget(maximum=8)
    budget.charge(6)
    with pytest.raises(ValueError):
        budget.charge(3)
    assert budget.used == 6
    assert budget.remaining == 2


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_budget_rejects_invalid_maximum(maximum):
    with pytest.raises(ValueError):
        AttemptBudget(maximum=maximum)


@pytest.mark.parametrize("lanes,reserve_after", [
    (0, 0), (-1, 0), (True, 0), (1.5, 0), (4, -1), (4, True), (4, 1.5),
])
def test_budget_rejects_invalid_charge_without_mutating(lanes, reserve_after):
    budget = AttemptBudget()
    with pytest.raises(ValueError):
        budget.charge(lanes, reserve_after=reserve_after)
    assert budget.used == 0
    assert budget.remaining == 128


def test_protocol_has_pinned_bytes_structure_and_fresh_record(tmp_path):
    first = load_protocol(PROTOCOL, SHA256)
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == SHA256
    assert first["kind"] == "frozen_same_slot_numerical_fitting"
    assert first["budget"]["maximum"] == 128
    first["episodes"][0]["index"] = 999
    second = load_protocol(PROTOCOL, SHA256)
    assert second["episodes"][0]["index"] == 2

    altered = tmp_path / "altered-protocol.json"
    altered.write_bytes(PROTOCOL.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        load_protocol(altered, SHA256)
    with pytest.raises(ValueError):
        load_protocol(PROTOCOL, "0" * 64)
