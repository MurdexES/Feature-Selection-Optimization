"""Tabular Q-learning, written from scratch.

The algorithm in one paragraph
------------------------------
Learn a table Q(s, a): "starting from state s, taking action a, then
acting optimally forever after, how much (discounted) reward do I
expect?". After every single step, nudge the entry toward a one-step-
better estimate built from what actually happened:

    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
    target   = r + gamma * max_a' Q(s', a')

``r`` is real, observed reward; ``max_a' Q(s', a')`` is the current best
guess about the future ("bootstrapping"). alpha is how hard to nudge;
gamma is how much the future matters (0.999 here: a failure a thousand
hours away must still influence today's decision).

Three deliberate choices, each learned the hard way in this project:

1. gamma = 0.999, not 0.98. The effective planning horizon is roughly
   1/(1-gamma) steps. At 0.98 that's 50 hours -- but degradation takes
   ~2000 hours to become dangerous, so a 0.98 agent is structurally
   blind to failures. At 0.999 the horizon is ~1000 hours: enough to
   connect "skip maintenance now" with "failure eventually".

2. Bootstrap through truncation. Our episodes end because we stop
   watching (truncated), not because the world ends (terminated). The
   value target at a truncated step must still include
   gamma * max Q(s', .) -- treating truncation as terminal teaches the
   agent that consequences beyond the horizon are worth 0, which makes
   run-to-failure look artificially good.

3. Initialise Q-values at the right *scale*, not optimistically.
   We tried classic optimistic initialisation (start every value above
   the best possible return) and watched it fail here: with gamma=0.999,
   every update's target is r + 0.999 * max Q(next), so while the table
   is uniformly inflated the inflation feeds itself -- the "optimism
   bubble" deflates by only ~(1-gamma)*V per sweep and needs thousands
   of episodes to converge. Since exploration is already guaranteed by
   exploring starts plus a permanent epsilon floor, we instead
   initialise near the true steady-state value, roughly
   mean_hourly_reward / (1 - gamma) ~= 1.3 / 0.001 ~= 1300, so early
   targets are already in calibrated range and updates spend their
   budget on *differences between actions*, which is all a policy needs.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.baselines.base import Policy
from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState
from src.environment.state_discretizer import (
    DiscreteState,
    StateDiscretizer,
)

N_ACTIONS = len(MaintenanceAction)


@dataclass(frozen=True)
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.999
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_episodes: int = 800
    # See module docstring, lesson 3: calibrated to the value scale
    # (~hourly reward / (1 - gamma)), NOT optimistic.
    initial_q_value: float = 1300.0


class QLearningAgent:
    def __init__(
        self,
        config: QLearningConfig | None = None,
        discretizer: StateDiscretizer | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or QLearningConfig()
        self.discretizer = discretizer or StateDiscretizer()
        self.rng = np.random.default_rng(seed)

        initial_value = self.config.initial_q_value
        self.q_table: dict[DiscreteState, np.ndarray] = defaultdict(
            lambda: np.full(N_ACTIONS, initial_value, dtype=np.float64)
        )

    def epsilon_for_episode(self, episode: int) -> float:
        """Linear anneal from epsilon_start to epsilon_min, then hold."""

        config = self.config
        progress = min(episode / config.epsilon_decay_episodes, 1.0)

        return config.epsilon_start + progress * (
            config.epsilon_min - config.epsilon_start
        )

    def choose_action(
        self, state: DiscreteState, epsilon: float
    ) -> MaintenanceAction:
        """Epsilon-greedy: random with probability epsilon, else best known."""

        if self.rng.random() < epsilon:
            return MaintenanceAction(int(self.rng.integers(0, N_ACTIONS)))

        return MaintenanceAction(int(np.argmax(self.q_table[state])))

    def update(
        self,
        state: DiscreteState,
        action: MaintenanceAction,
        reward: float,
        next_state: DiscreteState,
        terminated: bool,
    ) -> None:
        """One Bellman update from one transition.

        Note the argument is ``terminated``, NOT "done": on truncation the
        caller passes terminated=False and we still bootstrap, because the
        pump's life continues beyond the episode window.
        """

        if terminated:
            best_next_value = 0.0
        else:
            best_next_value = float(np.max(self.q_table[next_state]))

        td_target = reward + self.config.discount_factor * best_next_value
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.config.learning_rate * td_error

    def greedy_action(self, state: DiscreteState) -> MaintenanceAction:
        return MaintenanceAction(int(np.argmax(self.q_table[state])))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(dict(self.q_table), file)

    def load(self, path: Path) -> None:
        with path.open("rb") as file:
            saved: dict[DiscreteState, np.ndarray] = pickle.load(file)
        self.q_table.update(saved)


@dataclass
class QTablePolicy:
    """Adapter: a trained Q-table exposed through the Policy protocol,
    so the baseline evaluation runner can score it on identical terms."""

    agent: QLearningAgent

    def decide(self, state: ObservableState) -> MaintenanceAction:
        discrete = self.agent.discretizer.discretize(state)
        return self.agent.greedy_action(discrete)


# Static check that the adapter satisfies the protocol.
_: type[Policy] = QTablePolicy
