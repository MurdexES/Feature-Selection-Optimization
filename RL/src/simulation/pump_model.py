"""A simplified industrial ESP (Electrical Submersible Pump) simulator.

``PumpModel`` owns the pump's hidden physical state (health, mechanical
condition, downtime, failure) and produces noisy sensor readings from it.
It knows nothing about rewards, dollars, or RL -- see ``economics.py`` for
cost/revenue configuration and (in a later phase) the Gymnasium
environment for how those combine into a reward signal.

Simulation assumption, stated explicitly per the project's modelling
notes: mechanical condition is modelled as a single ordered severity ladder
(NORMAL -> UNBALANCE -> RUBBING -> MISALIGNMENT -> FAILED) driven purely by
the hidden ``health`` scalar. Real fault categories are not proven to
progress through one shared sequence -- this is a simplifying assumption
for the first simulator, not a validated degradation path. A later
version should support independent fault modes instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .degradation import (
    DegradationConfig,
    FailureRiskConfig,
    calculate_degradation,
    calculate_failure_probability,
)


class PumpCondition(IntEnum):
    NORMAL = 0
    UNBALANCE = 1
    RUBBING = 2
    MISALIGNMENT = 3
    FAILED = 4


@dataclass
class PumpSensorReadings:
    vibration: float
    temperature: float
    pressure: float
    flow_rate: float
    power_consumption: float


@dataclass
class PumpStepResult:
    sensors: PumpSensorReadings
    condition: PumpCondition
    health: float
    failed: bool
    degradation: float
    in_downtime: bool
    remaining_downtime: int


class PumpModel:
    """Simplified industrial pump degradation simulator."""

    def __init__(
        self,
        seed: int | None = None,
        degradation_config: DegradationConfig | None = None,
        failure_risk_config: FailureRiskConfig | None = None,
        preventive_downtime_hours: int = 8,
        corrective_downtime_hours: int = 24,
        inspection_downtime_hours: int = 1,
        preventive_health_range: tuple[float, float] = (0.92, 0.98),
        corrective_health_range: tuple[float, float] = (0.86, 0.94),
    ) -> None:
        self.rng = np.random.default_rng(seed)

        self.degradation_config = degradation_config or DegradationConfig()
        self.failure_risk_config = failure_risk_config or FailureRiskConfig()

        self.preventive_downtime_hours = preventive_downtime_hours
        self.corrective_downtime_hours = corrective_downtime_hours
        self.inspection_downtime_hours = inspection_downtime_hours
        self.preventive_health_range = preventive_health_range
        self.corrective_health_range = corrective_health_range

        self.health = 1.0
        self.condition = PumpCondition.NORMAL
        self.failed = False

        self.operating_hours = 0
        self.hours_since_maintenance = 0
        self.total_failures = 0

        self.last_load = 0.0
        self.remaining_downtime = 0

    def reset(self) -> PumpSensorReadings:
        """Restore the pump to an initial healthy state."""

        self.health = self.rng.uniform(0.96, 1.0)
        self.condition = PumpCondition.NORMAL
        self.failed = False

        self.operating_hours = 0
        self.hours_since_maintenance = 0
        self.total_failures = 0

        self.last_load = 0.0
        self.remaining_downtime = 0

        return self.get_sensor_readings(load=0.0)

    def force_degradation(
        self,
        health: float,
        hours_since_maintenance: int | None = None,
    ) -> None:
        """Put the pump directly into a partially degraded state.

        Exists for "exploring starts" in RL training (episodes that begin
        with an already-worn pump so degraded states actually get visited)
        and for tests. When ``hours_since_maintenance`` is not given, a
        physically consistent value is estimated from the average
        degradation rate, so the age signal doesn't contradict the health.
        """

        if not 0.05 <= health <= 1.0:
            raise ValueError(
                f"Forced health must be between 0.05 and 1.0. "
                f"Received: {health}"
            )

        self.health = health
        self.condition = self._determine_condition()
        self.failed = False
        self.remaining_downtime = 0

        if hours_since_maintenance is None:
            typical_hourly_degradation = 0.00026
            hours_since_maintenance = min(
                int((1.0 - health) / typical_hourly_degradation),
                4000,
            )

        self.hours_since_maintenance = hours_since_maintenance

    def operate(self, load: float) -> PumpStepResult:
        """Operate the pump for one hour.

        If the pump is currently in downtime (partway through preventive or
        corrective maintenance, or an inspection), this hour is consumed by
        that downtime instead: no production happens, health is unchanged,
        and the requested load is ignored. This is deliberate -- see
        module docstring and project notes on modelling downtime hour by
        hour rather than as an instantaneous lump sum.
        """

        self._validate_load(load)

        if self.failed:
            raise RuntimeError(
                "Cannot operate a failed pump. "
                "Corrective maintenance is required."
            )

        if self.remaining_downtime > 0:
            return self._advance_downtime_hour()

        self.last_load = load
        self.operating_hours += 1
        self.hours_since_maintenance += 1

        degradation = calculate_degradation(
            self.degradation_config,
            load=load,
            hours_since_maintenance=self.hours_since_maintenance,
            condition_severity=self._condition_severity(),
            rng=self.rng,
        )
        self.health = max(0.0, self.health - degradation)

        self.condition = self._determine_condition()

        failure_probability = calculate_failure_probability(
            self.failure_risk_config,
            health=self.health,
            load=load,
            condition_severity=self._condition_severity(),
        )

        if self.rng.random() < failure_probability:
            self.failed = True
            self.condition = PumpCondition.FAILED
            self.total_failures += 1

        sensors = self.get_sensor_readings(load=load)

        return PumpStepResult(
            sensors=sensors,
            condition=self.condition,
            health=self.health,
            failed=self.failed,
            degradation=degradation,
            in_downtime=False,
            remaining_downtime=self.remaining_downtime,
        )

    def inspect(self) -> dict[str, float | str]:
        """Return a noisy but relatively accurate health inspection.

        Inspecting costs a short amount of downtime (see
        ``inspection_downtime_hours``) but never restores health -- it only
        provides information. If a longer maintenance downtime is already
        in progress, inspecting does not shorten it.
        """

        self.remaining_downtime = max(
            self.remaining_downtime, self.inspection_downtime_hours
        )

        inspection_noise = self.rng.normal(
            loc=0.0,
            scale=0.015,
        )

        estimated_health = np.clip(
            self.health + inspection_noise,
            0.0,
            1.0,
        )

        return {
            "estimated_health": float(estimated_health),
            "actual_condition": self.condition.name,
        }

    def preventive_maintenance(self) -> None:
        """Restore most of the pump's health before failure.

        Health is restored immediately and ``hours_since_maintenance``
        resets right away; the pump then spends
        ``preventive_downtime_hours`` unable to produce (see ``operate``).
        This is a simplifying assumption -- the alternative (restoring
        health only once downtime completes) would be more realistic but
        adds little for how this simulator is currently used.
        """

        if self.failed:
            raise RuntimeError(
                "Preventive maintenance cannot repair a failed pump."
            )

        if self.remaining_downtime > 0:
            raise RuntimeError(
                "Cannot start preventive maintenance while the pump is "
                "already in downtime."
            )

        restored_health = self.rng.uniform(*self.preventive_health_range)

        self.health = max(self.health, restored_health)
        self.condition = PumpCondition.NORMAL
        self.hours_since_maintenance = 0
        self.last_load = 0.0
        self.remaining_downtime = self.preventive_downtime_hours

    def corrective_maintenance(self) -> None:
        """Repair the pump after an unexpected failure."""

        if not self.failed:
            raise RuntimeError(
                "Corrective maintenance is only required after failure."
            )

        self.health = self.rng.uniform(*self.corrective_health_range)
        self.condition = PumpCondition.NORMAL
        self.failed = False
        self.hours_since_maintenance = 0
        self.last_load = 0.0
        self.remaining_downtime = self.corrective_downtime_hours

    def get_sensor_readings(
        self,
        load: float | None = None,
    ) -> PumpSensorReadings:
        """Generate noisy sensor readings from pump health and load."""

        if load is None:
            load = self.last_load

        health_loss = 1.0 - self.health

        vibration = (
            1.5
            + 2.5 * load
            + 11.0 * health_loss
            + self._condition_vibration_effect()
            + self.rng.normal(0.0, 0.35)
        )

        temperature = (
            42.0
            + 24.0 * load
            + 28.0 * health_loss
            + self.rng.normal(0.0, 1.2)
        )

        pressure_efficiency = (
            1.0
            - 0.40 * health_loss
            - 0.10 * self._condition_severity()
        )

        pressure = (
            10.0
            * load
            * max(pressure_efficiency, 0.2)
            + self.rng.normal(0.0, 0.20)
        )

        flow_efficiency = (
            1.0
            - 0.45 * health_loss
            - 0.12 * self._condition_severity()
        )

        flow_rate = (
            100.0
            * load
            * max(flow_efficiency, 0.1)
            + self.rng.normal(0.0, 1.5)
        )

        power_consumption = (
            45.0 * load
            + 22.0 * health_loss
            + 5.0 * self._condition_severity()
            + self.rng.normal(0.0, 0.8)
        )

        if self.failed:
            pressure = 0.0
            flow_rate = 0.0
            power_consumption = 0.0

        return PumpSensorReadings(
            vibration=max(0.0, float(vibration)),
            temperature=max(0.0, float(temperature)),
            pressure=max(0.0, float(pressure)),
            flow_rate=max(0.0, float(flow_rate)),
            power_consumption=max(
                0.0,
                float(power_consumption),
            ),
        )

    def _advance_downtime_hour(self) -> PumpStepResult:
        """Consume one hour of an in-progress downtime. No production."""

        self.remaining_downtime -= 1

        sensors = self.get_sensor_readings(load=0.0)

        return PumpStepResult(
            sensors=sensors,
            condition=self.condition,
            health=self.health,
            failed=self.failed,
            degradation=0.0,
            in_downtime=True,
            remaining_downtime=self.remaining_downtime,
        )

    def _determine_condition(self) -> PumpCondition:
        """Map hidden health to a simulated mechanical condition.

        See the module docstring: this single severity ladder is a
        simulation assumption, not a validated fault-progression sequence.
        """

        if self.health > 0.78:
            return PumpCondition.NORMAL

        if self.health > 0.58:
            return PumpCondition.UNBALANCE

        if self.health > 0.38:
            return PumpCondition.RUBBING

        return PumpCondition.MISALIGNMENT

    def _condition_severity(self) -> float:
        severity_mapping = {
            PumpCondition.NORMAL: 0.0,
            PumpCondition.UNBALANCE: 0.25,
            PumpCondition.RUBBING: 0.55,
            PumpCondition.MISALIGNMENT: 0.85,
            PumpCondition.FAILED: 1.0,
        }

        return severity_mapping[self.condition]

    def _condition_vibration_effect(self) -> float:
        effect_mapping = {
            PumpCondition.NORMAL: 0.0,
            PumpCondition.UNBALANCE: 2.5,
            PumpCondition.RUBBING: 4.5,
            PumpCondition.MISALIGNMENT: 6.0,
            PumpCondition.FAILED: 8.0,
        }

        return effect_mapping[self.condition]

    @staticmethod
    def _validate_load(load: float) -> None:
        if not 0.0 <= load <= 1.0:
            raise ValueError(
                f"Load must be between 0.0 and 1.0. "
                f"Received: {load}"
            )
