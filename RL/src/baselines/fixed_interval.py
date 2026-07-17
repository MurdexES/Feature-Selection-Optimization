"""Baseline B: maintain on the calendar, not on the evidence.

Classic time-based preventive maintenance: after a fixed number of
operating hours, take the pump down regardless of how it actually looks.
Safe but wasteful -- it services healthy pumps and can still miss a pump
that degrades faster than the schedule assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState


@dataclass
class FixedIntervalPolicy:
    maintenance_interval_hours: int = 1500

    def decide(self, state: ObservableState) -> MaintenanceAction:
        if state.hours_since_maintenance >= self.maintenance_interval_hours:
            return MaintenanceAction.PREVENTIVE_MAINTENANCE

        return MaintenanceAction.OPERATE_FULL
