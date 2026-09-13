"""Lifecycle boundaries using a fake engine plus a short installed-engine check."""
import hashlib
import inspect

import numpy as np
import pytest

from doom.game import Game
import doom.game as game_module
from doom_learning.survival_arena import SurvivalArena
import doom_learning.survival_arena as arena_module


class BuiltinInt(int):
    pass


class FakeEngine:
    """Records the ViZDoom boundary while leaving wrapper logic real."""

    def __init__(self, fail_at=None, failure=None, close_error=None):
        self.calls = []
        self.close_calls = 0
        self.fail_at = fail_at
        self.failure = failure
        self.close_error = close_error

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name == self.fail_at:
                raise self.failure
        return call

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def game_engine(monkeypatch):
    engines = []

    def create():
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(game_module.vzd, 'DoomGame', create)
    return engines


@pytest.fixture
def arena_engine(monkeypatch):
    engines = []

    def create():
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(arena_module, 'build_map', lambda *args, **kwargs: {'fixture': True})
    monkeypatch.setattr(arena_module.vzd, 'DoomGame', create)
    return engines


@pytest.mark.parametrize(
    ('name', 'value', 'error'),
    [
        ('episode_timeout_tics', True, TypeError),
        ('episode_timeout_tics', BuiltinInt(3), TypeError),
        ('episode_timeout_tics', -1, ValueError),
        ('episode_start_tics', np.int64(3), TypeError),
        ('episode_start_tics', -1, ValueError),
    ],
)
def test_game_rejects_malformed_timing_before_engine_allocation(game_engine, name, value, error):
    """Removing strict pre-allocation validation would allocate an engine here."""
    with pytest.raises(error):
        Game(**{name: value})
    assert game_engine == []


def test_game_applies_explicit_timing_before_initialization(game_engine):
    """Moving evaluation timing after init would make it ineffective."""
    game = Game(episode_timeout_tics=3, episode_start_tics=0)
    try:
        calls = game_engine[0].calls
        names = [name for name, _ in calls]
        timeouts = [call for call in calls if call[0] == 'set_episode_timeout']
        assert timeouts == [('set_episode_timeout', (2100,)), ('set_episode_timeout', (3,))]
        assert calls[names.index('set_episode_start_time')] == ('set_episode_start_time', (0,))
        assert max(index for index, name in enumerate(names) if name == 'set_episode_timeout') < names.index('init')
        assert names.index('set_episode_start_time') < names.index('init')
        assert names.index('init') < names.index('new_episode')
    finally:
        game.close()


def test_game_default_configuration_leaves_source_start_time_untouched(game_engine):
    """Adding a default start override would change historical public rounds."""
    game = Game()
    try:
        calls = game_engine[0].calls
        assert ('set_episode_timeout', (35 * 60,)) in calls
        assert all(name != 'set_episode_start_time' for name, _ in calls)
    finally:
        game.close()


@pytest.mark.parametrize(
    ('fail_at', 'primary'),
    [
        ('load_config', RuntimeError('configuration failed')),
        ('init', RuntimeError('initialization failed')),
        ('new_episode', KeyboardInterrupt('first episode interrupted')),
    ],
)
def test_game_constructor_failure_closes_once_and_keeps_primary(game_engine, fail_at, primary):
    """Dropping guarded cleanup would leak the engine on setup failure."""
    def create():
        engine = FakeEngine(fail_at=fail_at, failure=primary)
        game_engine.append(engine)
        return engine

    game_module.vzd.DoomGame = create
    with pytest.raises(type(primary)) as caught:
        Game(episode_timeout_tics=2100, episode_start_tics=0)
    assert caught.value is primary
    assert game_engine[0].close_calls == 1


def test_game_cleanup_error_adds_note_without_replacing_keyboard_interrupt(game_engine):
    """Letting close replace the interruption would hide the interrupted work."""
    primary = KeyboardInterrupt('first episode interrupted')
    def create():
        engine = FakeEngine('new_episode', primary, RuntimeError('close failed'))
        game_engine.append(engine)
        return engine

    game_module.vzd.DoomGame = create
    with pytest.raises(KeyboardInterrupt) as caught:
        Game(episode_timeout_tics=2100, episode_start_tics=0)
    assert caught.value is primary
    assert game_engine[0].close_calls == 1
    assert 'Secondary game constructor cleanup: RuntimeError' in caught.value.__notes__


def test_survival_arena_keeps_declared_signature_and_engine_settings(arena_engine, tmp_path):
    """Changing arena controls or timing would alter the fixed hazard environment."""
    assert str(inspect.signature(SurvivalArena)) == '(path, seed=41031, *, seconds=60, hazard_left=True, angle=None)'
    arena = SurvivalArena(tmp_path / 'hazard.wad', seconds=3)
    try:
        calls = arena_engine[0].calls
        assert ('set_episode_timeout', (105,)) in calls
        assert ('set_episode_start_time', (0,)) in calls
        assert ('set_depth_buffer_enabled', (False,)) in calls
        assert ('set_labels_buffer_enabled', (False,)) in calls
        assert ('set_automap_buffer_enabled', (False,)) in calls
        buttons = next(args for name, args in calls if name == 'set_available_buttons')
        assert len(buttons[0]) == 3
    finally:
        arena.close()


@pytest.mark.parametrize(
    ('fail_at', 'primary'),
    [
        ('set_doom_game_path', RuntimeError('configuration failed')),
        ('init', RuntimeError('initialization failed')),
        ('new_episode', KeyboardInterrupt('first episode interrupted')),
    ],
)
def test_survival_arena_constructor_failure_closes_once_and_keeps_primary(
    arena_engine, tmp_path, fail_at, primary,
):
    """Removing arena cleanup would leave an externally allocated engine open."""
    def create():
        engine = FakeEngine(fail_at=fail_at, failure=primary)
        arena_engine.append(engine)
        return engine

    arena_module.vzd.DoomGame = create
    with pytest.raises(type(primary)) as caught:
        SurvivalArena(tmp_path / 'hazard.wad')
    assert caught.value is primary
    assert arena_engine[0].close_calls == 1


def test_survival_arena_cleanup_error_adds_note_without_replacing_primary(arena_engine, tmp_path):
    """Replacing an arena setup failure with close failure loses the actual cause."""
    primary = RuntimeError('initialization failed')
    def create():
        engine = FakeEngine('init', primary, RuntimeError('close failed'))
        arena_engine.append(engine)
        return engine

    arena_module.vzd.DoomGame = create
    with pytest.raises(RuntimeError) as caught:
        SurvivalArena(tmp_path / 'hazard.wad')
    assert caught.value is primary
    assert arena_engine[0].close_calls == 1
    assert 'Secondary game constructor cleanup: RuntimeError' in caught.value.__notes__


def test_real_vizdoom_timing_rgb_and_hazard_arena_boundary(tmp_path):
    """A changed timing boundary would break fixed-horizon live evaluation."""
    first = Game(seed=41831, episode_timeout_tics=3, episode_start_tics=0)
    second = Game(seed=41831, episode_timeout_tics=3, episode_start_tics=0)
    try:
        initial = first.game.get_episode_time()
        first_pixels = first.pixels()
        initial_hash = hashlib.sha256(first_pixels.tobytes()).hexdigest()
        assert first.game.get_episode_timeout() == 3
        assert first.game.get_episode_start_time() == 1
        assert initial == 1
        assert first_pixels.shape == (480, 640, 3)
        first_pixels.fill(0)
        assert hashlib.sha256(first.pixels().tobytes()).hexdigest() == initial_hash
        assert hashlib.sha256(second.pixels().tobytes()).hexdigest() == initial_hash
        first.act({'turn': 0, 'forward': 0, 'attack': False})
        assert first.game.get_episode_time() == initial + 1
        first.act({'turn': 0, 'forward': 0, 'attack': False})
        first.act({'turn': 0, 'forward': 0, 'attack': False})
        assert first.game.get_episode_time() - initial == 3
        assert first.observation()['finished']
    finally:
        first.close()
        second.close()

    default = Game(seed=41831)
    try:
        assert default.game.get_episode_timeout() == 35 * 60
        assert default.game.get_episode_start_time() == 10
    finally:
        default.close()

    arena = SurvivalArena(tmp_path / 'hazard.wad', seed=41831, seconds=3 / 35, angle=0)
    try:
        initial = arena.game.get_episode_time()
        assert arena.game.get_episode_timeout() == 3
        assert arena.game.get_episode_start_time() == 1
        arena.act({'turn': 0, 'forward': 0, 'attack': False})
        assert arena.game.get_episode_time() == initial + 1
    finally:
        arena.close()
