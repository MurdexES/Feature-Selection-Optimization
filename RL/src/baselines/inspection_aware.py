"""Baseline D: buy information, then act on it.

The only baseline that uses the INSPECT action. Sensors are noisy;
an inspection returns a much more accurate health estimate (sigma 0.015
vs vibration's load- and condition-confounded signal), at the price of a
fee and one lost production hour.

Rule: keep a reasonably fresh inspection on file; once one exists, choose
load or maintenance from the *estimated health* instead of raw vibration.
Note that maintenance invalidates the previous estimate (health jumped),
so the policy naturally re-inspects after every service -- that logic
lives in the operations layer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState


@dataclass
class InspectionAwarePolicy:
    inspection_interval_hours: int = 168  # roughly weekly
    maintain_below_health: float = 0.60
    reduce_load_below_health: float = 0.78

    def decide(self, state: ObservableState) -> MaintenanceAction:
        estimate_is_missing = state.last_inspected_health is None
        estimate_is_stale = (
            state.hours_since_inspection is not None
            and state.hours_since_inspection
            >= self.inspection_interval_hours
        )

        if estimate_is_missing or estimate_is_stale:
            return MaintenanceAction.INSPECT

        if state.last_inspected_health < self.maintain_below_health:
            return MaintenanceAction.PREVENTIVE_MAINTENANCE

        if state.last_inspected_health < self.reduce_load_below_health:
            return MaintenanceAction.OPERATE_REDUCED

        return MaintenanceAction.OPERATE_FULL
