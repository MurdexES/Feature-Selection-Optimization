"""The Gymnasium environment: the pump problem as a formal MDP.

This wraps ``MaintenanceOperations`` in the standard Gymnasium API so any
RL algorithm can train against it. Design decisions worth understanding:

Observation (Box of 10 floats, normalised)
    Built ONLY from ``ObservableState`` -- noisy sensors, maintenance and
    inspection bookkeeping. The hidden health and true condition are
    exposed in ``info`` for debugging/evaluation, never in the
    observation. The agent must infer risk, like a real operator.
    Each feature is divided by a typical scale so everything lands
    roughly in [0, 1]; neural networks train far better when inputs
    share an order of magnitude.

Reward
    ``HourOutcome.profit * reward_scale`` -- the exact same accounting
    the baselines are measured by, scaled by 0.01 so hourly rewards sit
    around +1.5 (a good producing hour) to about -100 (a failure hour).

terminated vs truncated
    This is a *continuing* task: a failed pump is repaired and operation
    resumes, so there is no natural terminal state. ``terminated`` is
    always False; episodes end via ``truncated`` when the time limit is
    reached. Agents must bootstrap through truncation (the world keeps
    going, we just stopped watching) -- this is why the distinction
    exists in the Gymnasium API at all.

Downtime
    During maintenance/repair the environment still steps hour by hour
    and still asks for an action; the operations layer ignores it. The
    agent experiences downtime as "hours where my choice doesn't matter
    and money drains" -- which is precisely the deterrent it should
    learn about.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.actions import MaintenanceAction
from src.environment.operations import (
    MaintenanceOperations,
    ObservableState,
    OperationsConfig,
)
from src.simulation.economics import EconomicsConfig
from src.simulation.pump_model import PumpModel

# One episode defaults to 90 days of hourly decisions.
DEFAULT_EPISODE_HOURS = 2160

DEFAULT_REWARD_SCALE = 0.01

# Typical scale of each raw feature, used for normalisation. Derived from
# the sensor formulas in pump_model.py (e.g. worst-case vibration is about
# 1.5 + 2.25 + 11 + 8 ~= 23) and the bookkeeping horizons.
VIBRATION_SCALE = 20.0
TEMPERATURE_SCALE = 100.0
PRESSURE_SCALE = 10.0
FLOW_SCALE = 100.0
POWER_SCALE = 70.0
MAINTENANCE_AGE_SCALE = 3000.0
DOWNTIME_SCALE = 24.0
INSPECTION_AGE_SCALE = 500.0

# Sentinel for "never inspected since last maintenance": health estimates
# live in [0, 1], so -1 is unambiguous and linearly separable from them.
NO_INSPECTION_SENTINEL = -1.0

OBSERVATION_SIZE = 10

OBSERVATION_LOW = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, NO_INSPECTION_SENTINEL, 0],
    dtype=np.float32,
)
OBSERVATION_HIGH = np.array(
    [1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)


def build_observation(state: ObservableState) -> np.ndarray:
    """Flatten an ObservableState into the normalised feature vector.

    A module-level function (not a method) so the evaluation adapters can
    turn an ObservableState into the exact vector a trained network
    expects, without instantiating an environment.
    """

    if state.last_inspected_health is None:
        inspected_health = NO_INSPECTION_SENTINEL
        inspection_age = 1.0  # maximally stale
    else:
        inspected_health = state.last_inspected_health
        inspection_age = min(
            (state.hours_since_inspection or 0) / INSPECTION_AGE_SCALE,
            1.0,
        )

    raw = np.array(
        [
            state.sensors.vibration / VIBRATION_SCALE,
            state.sensors.temperature / TEMPERATURE_SCALE,
            state.sensors.pressure / PRESSURE_SCALE,
            state.sensors.flow_rate / FLOW_SCALE,
            state.sensors.power_consumption / POWER_SCALE,
            state.hours_since_maintenance / MAINTENANCE_AGE_SCALE,
            state.remaining_downtime / DOWNTIME_SCALE,
            state.last_load,
            inspected_health,
            inspection_age,
        ],
        dtype=np.float32,
    )

    # Sensor noise can push a reading slightly past its nominal scale;
    # clip so the observation always stays inside the declared space.
    return np.clip(raw, OBSERVATION_LOW, OBSERVATION_HIGH)


class PumpMaintenanceEnv(gym.Env):
    """Hourly ESP maintenance scheduling as a Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_hours: int = DEFAULT_EPISODE_HOURS,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        economics: EconomicsConfig | None = None,
        operations_config: OperationsConfig | None = None,
        exploring_starts: bool = False,
    ) -> None:
        super().__init__()

        self.episode_hours = episode_hours
        self.reward_scale = reward_scale
        self.economics = economics or EconomicsConfig()
        self.operations_config = operations_config or OperationsConfig()

        # Exploring starts (training only): begin episodes with the pump
        # at a random degradation level instead of always factory-fresh.
        # Without this, epsilon-greedy exploration keeps randomly
        # maintaining the pump (~1 random action in 4 is maintenance), so
        # it never survives long enough to degrade, degraded states are
        # never visited, and the agent cannot learn what to do in them.
        # Evaluation environments must leave this False.
        self.exploring_starts = exploring_starts

        self.observation_space = spaces.Box(
            low=OBSERVATION_LOW,
            high=OBSERVATION_HIGH,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(MaintenanceAction))

        self.operations: MaintenanceOperations | None = None
        self._elapsed_hours = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Gymnasium seeding: seeds self.np_random reproducibly. Each
        # episode's pump gets a fresh seed drawn from it, so one env seed
        # yields a reproducible *sequence* of varied episodes.
        super().reset(seed=seed)

        pump_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.operations = MaintenanceOperations(
            pump=PumpModel(seed=pump_seed),
            economics=self.economics,
            config=self.operations_config,
        )
        self._elapsed_hours = 0

        if self.exploring_starts:
            # The range deliberately reaches down to the failure cliff
            # (risk explodes below health ~0.25). Starting only at 0.30+
            # was tried and failed: random maintenance during exploration
            # interrupts degradation long before the cliff, so the agent
            # experienced ~20 failures in 800 episodes -- far too weak a
            # signal to learn that the cliff exists at all.
            initial_health = float(self.np_random.uniform(0.05, 1.0))
            self.operations.pump.force_degradation(initial_health)

        state = self.operations.observe()
        return build_observation(state), self._build_info(state)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.operations is None:
            raise RuntimeError("Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")

        outcome = self.operations.step(MaintenanceAction(int(action)))
        self._elapsed_hours += 1

        state = self.operations.observe()
        observation = build_observation(state)
        reward = outcome.profit * self.reward_scale

        terminated = False  # continuing task: no natural terminal state
        truncated = self._elapsed_hours >= self.episode_hours

        info = self._build_info(state)
        info["outcome"] = outcome

        return observation, reward, terminated, truncated, info

    def _build_info(self, state: ObservableState) -> dict[str, Any]:
        """Debug/evaluation extras -- includes ground truth the agent
        must never see through the observation."""

        assert self.operations is not None
        return {
            "true_health": self.operations.pump.health,
            "true_condition": self.operations.pump.condition.name,
            "observable_state": state,
        }
