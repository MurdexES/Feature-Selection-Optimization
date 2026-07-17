"""Tests for the state discretizer and the Q-learning agent's mechanics."""

import numpy as np
import pytest

from src.agents.q_learning import QLearningAgent, QLearningConfig
from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState
from src.environment.state_discretizer import StateDiscretizer
from src.simulation.pump_model import PumpSensorReadings


def make_state(
    vibration: float = 4.0,
    hours_since_maintenance: int = 100,
    remaining_downtime: int = 0,
    last_inspected_health: float | None = None,
    hours_since_inspection: int | None = None,
) -> ObservableState:
    return ObservableState(
        sensors=PumpSensorReadings(
            vibration=vibration,
            temperature=60.0,
            pressure=8.0,
            flow_rate=85.0,
            power_consumption=45.0,
        ),
        hours_since_maintenance=hours_since_maintenance,
        remaining_downtime=remaining_downtime,
        last_load=0.9,
        last_inspected_health=last_inspected_health,
        hours_since_inspection=hours_since_inspection,
    )


def test_discretized_state_is_hashable_tuple_of_ints():
    discretizer = StateDiscretizer()
    state = discretizer.discretize(make_state())

    assert isinstance(state, tuple)
    assert all(isinstance(value, int) for value in state)
    hash(state)  # must not raise


def test_vibration_bins_split_at_decision_thresholds():
    discretizer = StateDiscretizer()

    calm = discretizer.discretize(make_state(vibration=4.0))
    elevated = discretizer.discretize(make_state(vibration=8.0))
    critical = discretizer.discretize(make_state(vibration=10.0))

    assert calm[0] == 0
    assert elevated[0] == 1
    assert critical[0] == 2


def test_unknown_inspection_gets_its_own_bin():
    discretizer = StateDiscretizer()

    unknown = discretizer.discretize(make_state(last_inspected_health=None))
    known = discretizer.discretize(
        make_state(last_inspected_health=0.9, hours_since_inspection=1)
    )

    assert unknown[2] == 0
    assert known[2] > 0


def test_epsilon_never_falls_below_minimum():
    agent = QLearningAgent(
        config=QLearningConfig(epsilon_min=0.05, epsilon_decay_episodes=100)
    )

    assert agent.epsilon_for_episode(0) == pytest.approx(1.0)
    assert agent.epsilon_for_episode(100_000) == pytest.approx(0.05)


def test_q_update_moves_value_toward_target():
    agent = QLearningAgent(seed=0)
    discretizer = agent.discretizer

    state = discretizer.discretize(make_state())
    next_state = discretizer.discretize(make_state(vibration=8.0))
    action = MaintenanceAction.OPERATE_FULL

    value_before = agent.q_table[state][action]

    # A strongly negative reward must pull the value down.
    agent.update(state, action, reward=-100.0, next_state=next_state,
                 terminated=False)

    assert agent.q_table[state][action] < value_before


def test_truncation_still_bootstraps_but_termination_does_not():
    config = QLearningConfig(learning_rate=1.0, initial_q_value=0.0)

    state = (0, 0, 0, 0, 0)
    next_state = (1, 0, 0, 0, 0)
    action = MaintenanceAction.OPERATE_FULL

    # Give the next state a known value.
    agent = QLearningAgent(config=config, seed=0)
    agent.q_table[next_state][:] = 10.0

    # terminated=False (e.g. truncation): target = r + gamma * 10.
    agent.update(state, action, reward=1.0, next_state=next_state,
                 terminated=False)
    bootstrapped = agent.q_table[state][action]
    assert bootstrapped > 1.0

    # terminated=True: target = r only.
    agent_terminal = QLearningAgent(config=config, seed=0)
    agent_terminal.q_table[next_state][:] = 10.0
    agent_terminal.update(state, action, reward=1.0, next_state=next_state,
                          terminated=True)
    assert agent_terminal.q_table[state][action] == 1.0


def test_initial_q_value_applies_to_unseen_states():
    agent = QLearningAgent(
        config=QLearningConfig(initial_q_value=1300.0)
    )

    never_seen = (3, 4, 0, 0, 0)
    assert np.all(agent.q_table[never_seen] == 1300.0)
