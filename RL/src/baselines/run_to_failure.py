"""Baseline A: run flat-out and let it break.

The floor every other strategy must beat. It maximises short-term
production and pays for it in failure penalties, corrective-maintenance
bills, and 24-hour outages.
"""

from __future__ import annotations

from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState


class RunToFailurePolicy:
    def decide(self, state: ObservableState) -> MaintenanceAction:
        return MaintenanceAction.OPERATE_FULL
