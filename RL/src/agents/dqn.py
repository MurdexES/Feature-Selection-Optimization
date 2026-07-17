"""DQN and Double DQN, from scratch in PyTorch.

From table to network
---------------------
Tabular Q-learning stores one number per (state, action) cell and learns
each cell independently -- vibration 9.4 and 9.6 in different bins share
nothing. A neural network Q(s; theta) -> R^4 takes the raw continuous
observation and generalises: similar states get similar values for free.
The price is stability, paid for with two mechanisms:

Replay buffer (see replay_buffer.py)
    Decorrelates training samples and reuses rare, expensive experiences
    like failures.

Target network
    The Bellman target r + gamma * max_a' Q(s', a') would otherwise be
    computed by the very network being updated -- each gradient step
    moves its own goalposts, which oscillates or diverges. So targets
    come from a frozen copy, re-synced to the online network every
    ``target_sync_every`` training steps.

Double DQN (one flag, not a new pipeline)
    Plain DQN's max operator both *selects* and *evaluates* the next
    action with the same (target) network. Any state where noise pushed
    some action's estimate too high gets picked precisely because it is
    too high -- so the target is systematically optimistic. Double DQN
    splits the roles: the online network selects the best next action,
    the target network evaluates it. One line of math:

        DQN:        target = r + gamma * max_a' Q_target(s', a')
        Double DQN: target = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

Also here, per the project spec: gradient clipping (one bad minibatch --
say, several failure transitions at once -- cannot blow up the weights),
Huber loss instead of MSE for the same robustness reason, checkpointing,
and an evaluation-mode adapter so the trained network can be scored by
the same runner as every other policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.agents.replay_buffer import ReplayBuffer
from src.environment.actions import MaintenanceAction
from src.environment.maintenance_env import (
    OBSERVATION_SIZE,
    build_observation,
)
from src.environment.operations import ObservableState

N_ACTIONS = len(MaintenanceAction)


class QNetwork(nn.Module):
    """Observation (10 floats) -> one Q-value per action (4 floats)."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_ACTIONS),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 1e-3
    discount_factor: float = 0.999
    batch_size: int = 64
    buffer_capacity: int = 200_000
    min_buffer_size: int = 5_000
    target_sync_every: int = 1_000
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_episodes: int = 150
    double_dqn: bool = False


class DQNAgent:
    def __init__(
        self,
        config: DQNConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed if seed is not None else 0)

        self.device = torch.device("cpu")

        self.online_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self._sync_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            observation_size=OBSERVATION_SIZE,
            seed=seed,
        )

        self.train_steps = 0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def epsilon_for_episode(self, episode: int) -> float:
        config = self.config
        progress = min(episode / config.epsilon_decay_episodes, 1.0)

        return config.epsilon_start + progress * (
            config.epsilon_min - config.epsilon_start
        )

    def choose_action(
        self, observation: np.ndarray, epsilon: float
    ) -> MaintenanceAction:
        if self.rng.random() < epsilon:
            return MaintenanceAction(int(self.rng.integers(0, N_ACTIONS)))

        return self.greedy_action(observation)

    def greedy_action(self, observation: np.ndarray) -> MaintenanceAction:
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online_network(observation_tensor)

        return MaintenanceAction(int(torch.argmax(q_values, dim=1).item()))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def remember(
        self,
        observation: np.ndarray,
        action: MaintenanceAction,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.replay_buffer.push(
            observation, int(action), reward, next_observation, terminated
        )

    def train_step(self) -> float | None:
        """One gradient step on one random minibatch.

        Returns the loss, or None while the buffer is still warming up
        (training on a handful of early transitions would just overfit
        to them).
        """

        config = self.config
        if len(self.replay_buffer) < max(
            config.min_buffer_size, config.batch_size
        ):
            return None

        batch = self.replay_buffer.sample(config.batch_size)

        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        terminated = torch.as_tensor(batch.terminated, device=self.device)

        # Q(s, a) for the actions that were actually taken.
        predicted = (
            self.online_network(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            if config.double_dqn:
                # Online network selects, target network evaluates.
                best_actions = self.online_network(next_states).argmax(
                    dim=1, keepdim=True
                )
                next_values = (
                    self.target_network(next_states)
                    .gather(1, best_actions)
                    .squeeze(1)
                )
            else:
                # Target network both selects and evaluates (biased up).
                next_values = self.target_network(next_states).max(dim=1).values

            # (1 - terminated) zeroes the future for true terminal states
            # only. Truncated transitions were stored terminated=False, so
            # they correctly bootstrap -- the world outlives the episode.
            targets = rewards + (
                config.discount_factor * next_values * (1.0 - terminated)
            )

        loss = nn.functional.smooth_l1_loss(predicted, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), config.gradient_clip_norm
        )
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % config.target_sync_every == 0:
            self._sync_target_network()

        return float(loss.item())

    def _sync_target_network(self) -> None:
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(state_dict)
        self._sync_target_network()


@dataclass
class DQNPolicy:
    """Adapter: trained network -> Policy protocol, for the shared
    evaluation runner. Uses the same build_observation as the env, so
    the network sees vectors identical to the ones it trained on."""

    agent: DQNAgent

    def decide(self, state: ObservableState) -> MaintenanceAction:
        observation = build_observation(state)
        return self.agent.greedy_action(observation)
