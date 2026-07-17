"""Experience replay buffer for DQN.

Why it exists: consecutive environment steps are almost identical (hour
1500 of a pump-year looks like hour 1501), and gradient descent on a
stream of near-duplicate, time-ordered samples both overfits to the
recent past and violates the i.i.d. assumption SGD leans on. Storing
transitions in a large pool and sampling minibatches *uniformly at
random* breaks that temporal correlation, and lets each transition be
reused many times (a failure costs ~$10k to experience; we should learn
from it more than once).

Implementation: preallocated NumPy arrays used as a ring buffer -- O(1)
insert, O(batch) sample, fixed memory, and the sampled batch comes out
as ready-to-use arrays instead of a list of tuples to repack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_size: int,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

        self._states = np.zeros(
            (capacity, observation_size), dtype=np.float32
        )
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros(
            (capacity, observation_size), dtype=np.float32
        )
        self._terminated = np.zeros(capacity, dtype=np.float32)

        self._next_index = 0
        self._size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Insert one transition, overwriting the oldest when full.

        Stores ``terminated`` (world ended), NOT "done": truncated
        transitions are stored with terminated=False so the learner still
        bootstraps through them -- same rule as tabular Q-learning.
        """

        index = self._next_index

        self._states[index] = state
        self._actions[index] = action
        self._rewards[index] = reward
        self._next_states[index] = next_state
        self._terminated[index] = float(terminated)

        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        indices = self.rng.integers(0, self._size, size=batch_size)

        return TransitionBatch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminated=self._terminated[indices],
        )

    def __len__(self) -> int:
        return self._size
