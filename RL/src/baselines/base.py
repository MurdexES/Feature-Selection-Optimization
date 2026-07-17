"""The shared shape of every maintenance policy.

A policy is anything with a ``decide(state) -> MaintenanceAction`` method.
Baselines implement it with hand-written rules; later, thin adapters wrap
the trained Q-table and DQN so *all* of them can be evaluated by the same
runner and compared on identical footing.
"""

from __future__ import annotations

from typing import Protocol

from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState


class Policy(Protocol):
    def decide(self, state: ObservableState) -> MaintenanceAction: ...
