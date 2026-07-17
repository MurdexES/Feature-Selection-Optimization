"""Tests for the Gymnasium environment wrapper."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment.maintenance_env import (
    OBSERVATION_SIZE,
    PumpMaintenanceEnv,
)


def make_env(episode_hours: int = 200) -> PumpMaintenanceEnv:
    return PumpMaintenanceEnv(episode_hours=episode_hours)


def test_passes_gymnasium_env_checker():
    env = make_env()
    # skip_render_check: this env has no rendering at all.
    check_env(env, skip_render_check=True)


def test_observation_shape_dtype_and_bounds():
    env = make_env()
    observation, _ = env.reset(seed=0)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)

    for _ in range(300):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)

        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if truncated:
            break


def test_hidden_health_is_not_in_observation_but_is_in_info():
    env = make_env()
    observation, info = env.reset(seed=1)

    assert "true_health" in info
    # The observation must not contain the exact hidden health value.
    assert not np.any(np.isclose(observation, info["true_health"]))


def test_episode_truncates_at_configured_length_and_never_terminates():
    env = make_env(episode_hours=50)
    env.reset(seed=2)

    for hour in range(1, 51):
        _, _, terminated, truncated, _ = env.step(0)
        assert terminated is False
        assert truncated is (hour == 50)


def test_invalid_action_is_rejected():
    env = make_env()
    env.reset(seed=3)

    with pytest.raises(ValueError):
        env.step(99)


def test_downtime_counts_down_hour_by_hour_in_observation():
    env = make_env()
    env.reset(seed=4)

    # Action 3 = preventive maintenance: 8 hours of downtime, one of
    # which is consumed by the starting hour itself.
    observation, _, _, _, info = env.step(3)
    downtime_feature = observation[6]  # remaining_downtime / 24

    expected = (env.operations.pump.preventive_downtime_hours - 1) / 24.0
    assert downtime_feature == pytest.approx(expected, abs=1e-6)

    # Each further step reduces it by exactly one hour.
    observation, _, _, _, _ = env.step(0)
    assert observation[6] == pytest.approx(expected - 1 / 24.0, abs=1e-6)


def test_exploring_starts_produce_degraded_pumps():
    env = PumpMaintenanceEnv(episode_hours=50, exploring_starts=True)

    initial_healths = []
    for seed in range(30):
        _, info = env.reset(seed=seed)
        initial_healths.append(info["true_health"])

    # Fresh pumps always start at 0.96+; exploring starts must produce a
    # wide spread that reaches down to the failure-risk zone.
    assert min(initial_healths) < 0.4
    assert max(initial_healths) > 0.85


def test_evaluation_env_starts_healthy_by_default():
    env = PumpMaintenanceEnv(episode_hours=50)

    for seed in range(10):
        _, info = env.reset(seed=seed)
        assert info["true_health"] >= 0.96


def test_same_seed_reproduces_same_episode():
    env_a = make_env()
    env_b = make_env()

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)

    for _ in range(100):
        step_a = env_a.step(0)
        step_b = env_b.step(0)
        assert np.array_equal(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]
