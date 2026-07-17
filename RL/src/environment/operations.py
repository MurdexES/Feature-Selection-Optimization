"""The operations layer: one simulated hour = one action = one cash flow.

Why this layer exists
---------------------
``PumpModel`` is pure physics (health, sensors, downtime) and
``economics.py`` is pure pricing (what a maintenance event costs). Neither
decides *when* money changes hands. That timing logic has to live exactly
once, because two different consumers need identical accounting:

  1. the baseline policies (Phase 4), evaluated in a plain Python loop;
  2. the Gymnasium environment (Phase 5), whose per-step reward is
     literally ``HourOutcome.profit`` from this class.

If baselines and the RL agent were metered by two separate pieces of
code, any accounting drift between them would silently bias the
comparison we ultimately care about ("did RL beat the baselines?").

Timing rules implemented here (each is a modelling choice, stated so it
can be challenged later):

  * Failure penalty is charged in the hour the failure happens.
  * Corrective-maintenance cost is charged when the repair *starts*
    (the hour after failure), and the pump is then down 24 hours.
  * Preventive-maintenance cost is charged when it starts; 8 down hours.
  * Inspection costs its fee plus exactly one non-producing hour.
  * Every non-producing (downtime) hour also costs the hourly downtime
    rate, and earns no revenue and pays no energy cost.
  * While the pump is failed or in downtime, the caller's action is
    ignored: repair/downtime proceeds regardless. The agent's "choice"
    that hour has no effect -- and the RL agent must learn that too.
  * Maintenance (either kind) invalidates the last inspection result:
    health has jumped, so a pre-maintenance estimate is stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.environment.actions import MaintenanceAction
from src.simulation.economics import EconomicsConfig
from src.simulation.pump_model import PumpModel, PumpSensorReadings


@dataclass(frozen=True)
class OperationsConfig:
    """Loads used for the two 'operate' actions."""

    full_load: float = 0.9
    reduced_load: float = 0.6


@dataclass
class ObservableState:
    """Everything a decision-maker is allowed to see before acting.

    This is the honest, operator's-eye view: dashboard sensor readings
    from the last hour, maintenance/inspection bookkeeping -- and nothing
    else. The hidden ``health`` float and true ``PumpCondition`` are
    deliberately absent. Baseline policies read this directly; the
    Gymnasium environment flattens it into its observation vector.
    """

    sensors: PumpSensorReadings
    hours_since_maintenance: int
    remaining_downtime: int
    last_load: float
    last_inspected_health: float | None
    hours_since_inspection: int | None


@dataclass
class HourOutcome:
    """The full cash-flow and event record of one simulated hour."""

    revenue: float = 0.0
    energy_cost: float = 0.0
    inspection_cost: float = 0.0
    maintenance_cost: float = 0.0
    downtime_cost: float = 0.0
    failure_penalty: float = 0.0

    in_downtime: bool = False
    failed_this_hour: bool = False
    preventive_started: bool = False
    corrective_started: bool = False
    inspected: bool = False
    action_ignored: bool = False

    @property
    def profit(self) -> float:
        return (
            self.revenue
            - self.energy_cost
            - self.inspection_cost
            - self.maintenance_cost
            - self.downtime_cost
            - self.failure_penalty
        )


@dataclass
class OperationsTotals:
    """Running counters, used for end-of-run reporting."""

    hours: int = 0
    producing_hours: int = 0
    downtime_hours: int = 0

    revenue: float = 0.0
    energy_cost: float = 0.0
    inspection_cost: float = 0.0
    maintenance_cost: float = 0.0
    downtime_cost: float = 0.0
    failure_penalty: float = 0.0

    failures: int = 0
    preventive_maintenance_count: int = 0
    corrective_maintenance_count: int = 0
    inspection_count: int = 0

    @property
    def total_profit(self) -> float:
        return (
            self.revenue
            - self.energy_cost
            - self.inspection_cost
            - self.maintenance_cost
            - self.downtime_cost
            - self.failure_penalty
        )

    @property
    def availability(self) -> float:
        """Fraction of hours the pump was actually producing."""

        if self.hours == 0:
            return 0.0
        return self.producing_hours / self.hours

    @property
    def mean_time_between_failures(self) -> float | None:
        """Producing hours per failure; None if no failure occurred."""

        if self.failures == 0:
            return None
        return self.producing_hours / self.failures


class MaintenanceOperations:
    """Applies one action per simulated hour and meters the money."""

    def __init__(
        self,
        pump: PumpModel,
        economics: EconomicsConfig | None = None,
        config: OperationsConfig | None = None,
    ) -> None:
        self.pump = pump
        self.economics = economics or EconomicsConfig()
        self.config = config or OperationsConfig()

        self.totals = OperationsTotals()

        self._last_sensors = self.pump.reset()
        self._last_load = 0.0
        self._last_inspected_health: float | None = None
        self._last_inspection_hour: int | None = None

    def observe(self) -> ObservableState:
        hours_since_inspection = None
        if self._last_inspection_hour is not None:
            hours_since_inspection = (
                self.totals.hours - self._last_inspection_hour
            )

        return ObservableState(
            sensors=self._last_sensors,
            hours_since_maintenance=self.pump.hours_since_maintenance,
            remaining_downtime=self.pump.remaining_downtime,
            last_load=self._last_load,
            last_inspected_health=self._last_inspected_health,
            hours_since_inspection=hours_since_inspection,
        )

    def step(self, action: MaintenanceAction | None) -> HourOutcome:
        """Advance one hour. ``action`` is ignored when the pump is failed
        or mid-downtime (repair takes priority over choice)."""

        outcome = HourOutcome()
        self.totals.hours += 1

        if self.pump.failed:
            self._start_corrective_maintenance(outcome)
            self._consume_downtime_hour(outcome)
        elif self.pump.remaining_downtime > 0:
            outcome.action_ignored = action is not None
            self._consume_downtime_hour(outcome)
        elif action is MaintenanceAction.PREVENTIVE_MAINTENANCE:
            self._start_preventive_maintenance(outcome)
            self._consume_downtime_hour(outcome)
        elif action is MaintenanceAction.INSPECT:
            self._perform_inspection(outcome)
            self._consume_downtime_hour(outcome)
        else:
            load = (
                self.config.full_load
                if action is MaintenanceAction.OPERATE_FULL
                else self.config.reduced_load
            )
            self._operate_hour(load, outcome)

        self._accumulate(outcome)
        return outcome

    # ------------------------------------------------------------------
    # The five things that can happen during an hour
    # ------------------------------------------------------------------

    def _operate_hour(self, load: float, outcome: HourOutcome) -> None:
        result = self.pump.operate(load=load)
        self._last_sensors = result.sensors
        self._last_load = load

        # A failure zeroes flow and power for the hour (see PumpModel),
        # so revenue/energy below are automatically zero in that case.
        outcome.revenue = (
            result.sensors.flow_rate * self.economics.revenue_per_flow_unit
        )
        outcome.energy_cost = (
            result.sensors.power_consumption
            * self.economics.energy_cost_per_power_unit
        )

        if result.failed:
            outcome.failed_this_hour = True
            outcome.failure_penalty = self.economics.failure_penalty
            self.totals.failures += 1
        else:
            self.totals.producing_hours += 1

    def _start_corrective_maintenance(self, outcome: HourOutcome) -> None:
        self.pump.corrective_maintenance()
        outcome.corrective_started = True
        outcome.maintenance_cost = self.economics.corrective_maintenance_cost
        self.totals.corrective_maintenance_count += 1
        self._invalidate_inspection()

    def _start_preventive_maintenance(self, outcome: HourOutcome) -> None:
        self.pump.preventive_maintenance()
        outcome.preventive_started = True
        outcome.maintenance_cost = self.economics.preventive_maintenance_cost
        self.totals.preventive_maintenance_count += 1
        self._invalidate_inspection()

    def _perform_inspection(self, outcome: HourOutcome) -> None:
        inspection = self.pump.inspect()
        outcome.inspected = True
        outcome.inspection_cost = self.economics.inspection_cost
        self.totals.inspection_count += 1

        self._last_inspected_health = float(inspection["estimated_health"])
        self._last_inspection_hour = self.totals.hours

    def _consume_downtime_hour(self, outcome: HourOutcome) -> None:
        """One non-producing hour: downtime rate applies, nothing earned."""

        result = self.pump.operate(load=0.0)
        self._last_sensors = result.sensors
        self._last_load = 0.0

        outcome.in_downtime = True
        outcome.downtime_cost = self.economics.downtime_cost_per_hour
        self.totals.downtime_hours += 1

    def _invalidate_inspection(self) -> None:
        """Maintenance changes health, so any earlier estimate is stale."""

        self._last_inspected_health = None
        self._last_inspection_hour = None

    def _accumulate(self, outcome: HourOutcome) -> None:
        self.totals.revenue += outcome.revenue
        self.totals.energy_cost += outcome.energy_cost
        self.totals.inspection_cost += outcome.inspection_cost
        self.totals.maintenance_cost += outcome.maintenance_cost
        self.totals.downtime_cost += outcome.downtime_cost
        self.totals.failure_penalty += outcome.failure_penalty
