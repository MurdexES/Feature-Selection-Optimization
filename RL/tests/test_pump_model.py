"""Tests for PumpModel: the physical simulator, independent of any RL logic.

Run with:

    cd RL && pytest tests/test_pump_model.py -v
"""

import numpy as np
import pytest

from src.simulation.pump_model import PumpCondition, PumpModel


def make_pump(seed: int = 0) -> PumpModel:
    pump = PumpModel(seed=seed)
    pump.reset()
    return pump


def test_health_stays_within_valid_range():
    pump = make_pump(seed=1)

    for _ in range(5000):
        if pump.failed:
            pump.corrective_maintenance()
            continue

        result = pump.operate(load=0.9)
        assert 0.0 <= result.health <= 1.0


def test_load_must_be_between_zero_and_one():
    pump = make_pump(seed=2)

    with pytest.raises(ValueError):
        pump.operate(load=1.5)

    with pytest.raises(ValueError):
        pump.operate(load=-0.1)


def test_higher_load_causes_greater_average_degradation():
    """Averaged over many independent single-hour trials, running at high
    load should degrade the pump more than running at low load. Any one
    hour is noisy, so this compares means across many seeds rather than a
    single run."""

    n_trials = 500

    def mean_degradation(load: float) -> float:
        totals = []
        for seed in range(n_trials):
            pump = PumpModel(seed=seed)
            pump.reset()
            totals.append(pump.operate(load=load).degradation)
        return float(np.mean(totals))

    low_load_mean = mean_degradation(0.2)
    high_load_mean = mean_degradation(0.95)

    assert high_load_mean > low_load_mean


def test_preventive_maintenance_improves_health():
    pump = make_pump(seed=3)

    for _ in range(1500):
        if pump.failed:
            pump.corrective_maintenance()
        pump.operate(load=0.9)

    health_before = pump.health
    pump.preventive_maintenance()

    assert pump.health >= health_before
    assert pump.condition is PumpCondition.NORMAL
    assert pump.hours_since_maintenance == 0


def test_preventive_maintenance_rejected_on_failed_pump():
    pump = make_pump(seed=4)
    pump.failed = True

    with pytest.raises(RuntimeError):
        pump.preventive_maintenance()


def test_corrective_maintenance_clears_failed_state():
    pump = make_pump(seed=5)
    pump.failed = True
    pump.health = 0.05

    pump.corrective_maintenance()

    assert pump.failed is False
    assert pump.condition is PumpCondition.NORMAL
    assert pump.health > 0.05


def test_corrective_maintenance_rejected_when_not_failed():
    pump = make_pump(seed=6)

    with pytest.raises(RuntimeError):
        pump.corrective_maintenance()


def test_failed_pump_cannot_operate():
    pump = make_pump(seed=7)
    pump.failed = True

    with pytest.raises(RuntimeError):
        pump.operate(load=0.5)


def test_seeded_runs_are_reproducible():
    pump_a = make_pump(seed=42)
    pump_b = make_pump(seed=42)

    for _ in range(200):
        result_a = pump_a.operate(load=0.8)
        result_b = pump_b.operate(load=0.8)

        assert result_a.health == result_b.health
        assert result_a.sensors.vibration == result_b.sensors.vibration

        if result_a.failed:
            pump_a.corrective_maintenance()
            pump_b.corrective_maintenance()


def test_preventive_maintenance_blocks_production_during_downtime():
    pump = make_pump(seed=8)
    pump.preventive_maintenance()

    assert pump.remaining_downtime == pump.preventive_downtime_hours

    hours_since_maintenance_before = pump.hours_since_maintenance

    for expected_remaining in range(pump.preventive_downtime_hours - 1, -1, -1):
        result = pump.operate(load=0.9)

        assert result.in_downtime is True
        assert result.degradation == 0.0
        assert pump.remaining_downtime == expected_remaining
        # Downtime hours must not count as operating hours.
        assert pump.hours_since_maintenance == hours_since_maintenance_before

    # Downtime has fully elapsed: the next hour operates normally again.
    result = pump.operate(load=0.9)
    assert result.in_downtime is False
    assert pump.hours_since_maintenance == hours_since_maintenance_before + 1


def test_cannot_start_preventive_maintenance_while_already_in_downtime():
    pump = make_pump(seed=9)
    pump.preventive_maintenance()

    with pytest.raises(RuntimeError):
        pump.preventive_maintenance()


def test_inspect_does_not_change_health_but_adds_short_downtime():
    pump = make_pump(seed=10)
    health_before = pump.health

    inspection = pump.inspect()

    assert pump.health == health_before
    assert pump.remaining_downtime == pump.inspection_downtime_hours
    assert 0.0 <= inspection["estimated_health"] <= 1.0
    assert inspection["actual_condition"] == pump.condition.name


def test_inspect_does_not_shorten_an_in_progress_maintenance_downtime():
    pump = make_pump(seed=11)
    pump.preventive_maintenance()
    downtime_before = pump.remaining_downtime

    pump.inspect()

    assert pump.remaining_downtime == downtime_before


def test_force_degradation_sets_consistent_state():
    pump = make_pump(seed=12)

    pump.force_degradation(health=0.5)

    assert pump.health == 0.5
    assert pump.condition.name == "RUBBING"  # 0.38 < 0.5 <= 0.58
    assert pump.failed is False
    # Age estimate must be consistent: a half-worn pump is not fresh.
    assert pump.hours_since_maintenance > 500

    with pytest.raises(ValueError):
        pump.force_degradation(health=0.0)
