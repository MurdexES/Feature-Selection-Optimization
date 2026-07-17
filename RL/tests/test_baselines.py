"""Tests for the four baseline policies: rule logic only, no simulation."""

from src.baselines import (
    FixedIntervalPolicy,
    InspectionAwarePolicy,
    RunToFailurePolicy,
    SensorThresholdPolicy,
)
from src.environment.actions import MaintenanceAction
from src.environment.operations import ObservableState
from src.simulation.pump_model import PumpSensorReadings


def make_state(
    vibration: float = 4.0,
    hours_since_maintenance: int = 100,
    last_inspected_health: float | None = None,
    hours_since_inspection: int | None = None,
) -> ObservableState:
    return ObservableState(
        sensors=PumpSensorReadings(
            vibration=vibration,
            temperature=60.0,
            pressure=8.0,
            flow_rate=85.0,
            power_consumption=45.0,
        ),
        hours_since_maintenance=hours_since_maintenance,
        remaining_downtime=0,
        last_load=0.9,
        last_inspected_health=last_inspected_health,
        hours_since_inspection=hours_since_inspection,
    )


def test_run_to_failure_always_operates_full():
    policy = RunToFailurePolicy()
    state = make_state(vibration=15.0, hours_since_maintenance=5000)

    assert policy.decide(state) is MaintenanceAction.OPERATE_FULL


def test_fixed_interval_maintains_exactly_on_schedule():
    policy = FixedIntervalPolicy(maintenance_interval_hours=1500)

    early = make_state(hours_since_maintenance=1499)
    due = make_state(hours_since_maintenance=1500)

    assert policy.decide(early) is MaintenanceAction.OPERATE_FULL
    assert policy.decide(due) is MaintenanceAction.PREVENTIVE_MAINTENANCE


def test_sensor_threshold_escalates_with_vibration():
    policy = SensorThresholdPolicy(
        critical_vibration=9.5, elevated_vibration=7.0
    )

    assert policy.decide(make_state(vibration=4.0)) is (
        MaintenanceAction.OPERATE_FULL
    )
    assert policy.decide(make_state(vibration=8.0)) is (
        MaintenanceAction.OPERATE_REDUCED
    )
    assert policy.decide(make_state(vibration=12.0)) is (
        MaintenanceAction.PREVENTIVE_MAINTENANCE
    )


def test_inspection_aware_inspects_when_estimate_missing_or_stale():
    policy = InspectionAwarePolicy(inspection_interval_hours=168)

    missing = make_state(last_inspected_health=None)
    stale = make_state(
        last_inspected_health=0.9, hours_since_inspection=168
    )

    assert policy.decide(missing) is MaintenanceAction.INSPECT
    assert policy.decide(stale) is MaintenanceAction.INSPECT


def test_inspection_aware_acts_on_fresh_estimate():
    policy = InspectionAwarePolicy(
        maintain_below_health=0.60, reduce_load_below_health=0.78
    )

    healthy = make_state(
        last_inspected_health=0.95, hours_since_inspection=10
    )
    degraded = make_state(
        last_inspected_health=0.70, hours_since_inspection=10
    )
    critical = make_state(
        last_inspected_health=0.50, hours_since_inspection=10
    )

    assert policy.decide(healthy) is MaintenanceAction.OPERATE_FULL
    assert policy.decide(degraded) is MaintenanceAction.OPERATE_REDUCED
    assert policy.decide(critical) is (
        MaintenanceAction.PREVENTIVE_MAINTENANCE
    )
