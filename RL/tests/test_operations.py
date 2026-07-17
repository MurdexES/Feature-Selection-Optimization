"""Tests for the operations layer: action -> cash-flow timing rules."""

import pytest

from src.environment.actions import MaintenanceAction
from src.environment.operations import MaintenanceOperations
from src.simulation.economics import EconomicsConfig
from src.simulation.pump_model import PumpModel


def make_operations(seed: int = 0) -> MaintenanceOperations:
    return MaintenanceOperations(pump=PumpModel(seed=seed))


def test_operating_hour_earns_revenue_and_pays_energy():
    operations = make_operations()

    outcome = operations.step(MaintenanceAction.OPERATE_FULL)

    assert outcome.revenue > 0
    assert outcome.energy_cost > 0
    assert outcome.maintenance_cost == 0
    assert outcome.in_downtime is False


def test_reduced_load_produces_less_than_full_load():
    full = make_operations(seed=1)
    reduced = make_operations(seed=1)

    full_revenue = sum(
        full.step(MaintenanceAction.OPERATE_FULL).revenue for _ in range(200)
    )
    reduced_revenue = sum(
        reduced.step(MaintenanceAction.OPERATE_REDUCED).revenue
        for _ in range(200)
    )

    assert reduced_revenue < full_revenue


def test_preventive_maintenance_costs_and_blocks_production():
    operations = make_operations(seed=2)
    economics = operations.economics

    outcome = operations.step(MaintenanceAction.PREVENTIVE_MAINTENANCE)

    assert outcome.preventive_started is True
    assert outcome.maintenance_cost == economics.preventive_maintenance_cost
    assert outcome.downtime_cost == economics.downtime_cost_per_hour
    assert outcome.revenue == 0

    # 8 downtime hours total: the start hour consumed one, 7 remain.
    remaining = operations.pump.preventive_downtime_hours - 1
    for _ in range(remaining):
        follow_up = operations.step(MaintenanceAction.OPERATE_FULL)
        assert follow_up.in_downtime is True
        assert follow_up.action_ignored is True
        assert follow_up.revenue == 0

    # Downtime over: production resumes.
    resumed = operations.step(MaintenanceAction.OPERATE_FULL)
    assert resumed.in_downtime is False
    assert resumed.revenue > 0

    assert operations.totals.downtime_hours == (
        operations.pump.preventive_downtime_hours
    )


def test_failure_charges_penalty_then_corrective_repair_runs():
    operations = make_operations(seed=3)
    economics = operations.economics

    # Force the pump close to death so a failure occurs quickly.
    operations.pump.health = 0.05

    failed_outcome = None
    for _ in range(2000):
        outcome = operations.step(MaintenanceAction.OPERATE_FULL)
        if outcome.failed_this_hour:
            failed_outcome = outcome
            break

    assert failed_outcome is not None, "expected a failure at health=0.05"
    assert failed_outcome.failure_penalty == economics.failure_penalty
    assert failed_outcome.revenue == 0  # failed hour produces nothing

    # The next hour must begin corrective maintenance, charging its cost.
    repair_outcome = operations.step(MaintenanceAction.OPERATE_FULL)
    assert repair_outcome.corrective_started is True
    assert repair_outcome.maintenance_cost == (
        economics.corrective_maintenance_cost
    )
    assert repair_outcome.in_downtime is True
    assert operations.pump.failed is False


def test_inspection_costs_fee_plus_one_downtime_hour():
    operations = make_operations(seed=4)
    economics = operations.economics

    outcome = operations.step(MaintenanceAction.INSPECT)

    assert outcome.inspected is True
    assert outcome.inspection_cost == economics.inspection_cost
    assert outcome.downtime_cost == economics.downtime_cost_per_hour
    assert outcome.revenue == 0

    # Exactly one hour lost: next hour operates normally.
    resumed = operations.step(MaintenanceAction.OPERATE_FULL)
    assert resumed.in_downtime is False

    state = operations.observe()
    assert state.last_inspected_health is not None
    assert 0.0 <= state.last_inspected_health <= 1.0


def test_maintenance_invalidates_previous_inspection():
    operations = make_operations(seed=5)

    operations.step(MaintenanceAction.INSPECT)
    assert operations.observe().last_inspected_health is not None

    operations.step(MaintenanceAction.PREVENTIVE_MAINTENANCE)
    assert operations.observe().last_inspected_health is None


def test_hidden_health_is_not_observable():
    operations = make_operations(seed=6)
    state = operations.observe()

    field_names = set(vars(state).keys())
    assert "health" not in field_names
    assert "condition" not in field_names


def test_profit_equals_revenue_minus_all_costs():
    operations = make_operations(seed=7)

    for hour in range(300):
        action = (
            MaintenanceAction.PREVENTIVE_MAINTENANCE
            if hour == 150
            else MaintenanceAction.OPERATE_FULL
        )
        operations.step(action)

    totals = operations.totals
    reconstructed = (
        totals.revenue
        - totals.energy_cost
        - totals.inspection_cost
        - totals.maintenance_cost
        - totals.downtime_cost
        - totals.failure_penalty
    )

    assert totals.total_profit == pytest.approx(reconstructed)
    assert totals.hours == 300
    assert totals.producing_hours + totals.downtime_hours <= totals.hours
