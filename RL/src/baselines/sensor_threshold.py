"""Baseline C: react to the vibration dashboard.

Condition-based maintenance in its simplest form: two thresholds on the
most informative sensor. Between them, back off the load to slow
degradation; above the critical line, take the pump down for service.

The thresholds come from the sensor model: a healthy pump at full load
vibrates around 3.5-4.5; UNBALANCE adds +2.5, RUBBING +4.5. So 7.0 says
"probably no longer NORMAL" and 9.5 says "well into degradation".
"""

from __future__ import annotations

from dataclasses import dataclass

from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState


@dataclass
class SensorThresholdPolicy:
    critical_vibration: float = 9.5
    elevated_vibration: float = 7.0

    def decide(self, state: ObservableState) -> MaintenanceAction:
        vibration = state.sensors.vibration

        if vibration >= self.critical_vibration:
            return MaintenanceAction.PREVENTIVE_MAINTENANCE

        if vibration >= self.elevated_vibration:
            return MaintenanceAction.OPERATE_REDUCED

        return MaintenanceAction.OPERATE_FULL
