"""Run any Policy over many seeded pump-years and report the economics.

One runner for everything: baselines now, Q-learning and DQN adapters
later. Because every policy is metered by the same MaintenanceOperations
object, the resulting numbers are directly comparable -- which is the
entire question this project asks ("does the learned policy beat the
hand-written ones?").

Never judge a policy on one seed. Failures are dice rolls: a lucky seed
can make run-to-failure look clever. We run 20 independent pump-years
and report mean and standard deviation.

Run as a script to evaluate all four baselines:

    cd RL && python -m src.evaluation.evaluate_policy
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baselines import (
    FixedIntervalPolicy,
    InspectionAwarePolicy,
    Policy,
    RunToFailurePolicy,
    SensorThresholdPolicy,
)
from src.environment.operations import MaintenanceOperations
from src.simulation.pump_model import PumpModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "baselines"

HOURS_PER_YEAR = 8760
DEFAULT_SEEDS = tuple(range(20))


def run_episode(
    policy: Policy,
    seed: int,
    hours: int = HOURS_PER_YEAR,
) -> dict[str, float | int | None]:
    """Simulate one pump for ``hours`` under ``policy``; return the books."""

    operations = MaintenanceOperations(pump=PumpModel(seed=seed))

    for _ in range(hours):
        state = operations.observe()

        # During downtime or repair the action is ignored anyway; passing
        # None keeps the log honest about who chose what.
        if state.remaining_downtime > 0 or operations.pump.failed:
            operations.step(None)
        else:
            operations.step(policy.decide(state))

    totals = operations.totals

    return {
        "seed": seed,
        "total_profit": totals.total_profit,
        "revenue": totals.revenue,
        "energy_cost": totals.energy_cost,
        "inspection_cost": totals.inspection_cost,
        "maintenance_cost": totals.maintenance_cost,
        "downtime_cost": totals.downtime_cost,
        "failure_penalty": totals.failure_penalty,
        "failures": totals.failures,
        "preventive_maintenance": totals.preventive_maintenance_count,
        "corrective_maintenance": totals.corrective_maintenance_count,
        "inspections": totals.inspection_count,
        "downtime_hours": totals.downtime_hours,
        "availability": totals.availability,
        "mtbf_hours": totals.mean_time_between_failures,
    }


def evaluate_policy(
    policy: Policy,
    policy_name: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    hours: int = HOURS_PER_YEAR,
) -> pd.DataFrame:
    episodes = [
        {"policy": policy_name, **run_episode(policy, seed, hours)}
        for seed in seeds
    ]

    return pd.DataFrame(episodes)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean-and-spread table, one row per policy."""

    return (
        results.groupby("policy")
        .agg(
            mean_profit=("total_profit", "mean"),
            profit_std=("total_profit", "std"),
            mean_failures=("failures", "mean"),
            mean_preventive=("preventive_maintenance", "mean"),
            mean_inspections=("inspections", "mean"),
            mean_downtime_hours=("downtime_hours", "mean"),
            mean_availability=("availability", "mean"),
        )
        .sort_values("mean_profit", ascending=False)
        .round(2)
    )


def main() -> None:
    baselines: dict[str, Policy] = {
        "run_to_failure": RunToFailurePolicy(),
        "fixed_interval": FixedIntervalPolicy(),
        "sensor_threshold": SensorThresholdPolicy(),
        "inspection_aware": InspectionAwarePolicy(),
    }

    all_results = []
    for name, policy in baselines.items():
        print(f"Evaluating {name} over {len(DEFAULT_SEEDS)} seeds...")
        all_results.append(evaluate_policy(policy, name))

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize(results)

    print("\n--- Baseline comparison (one simulated year per seed) ---")
    print(summary.to_string())

    BASELINE_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        BASELINE_RESULTS_DIRECTORY / "baseline_episodes.csv", index=False
    )
    summary.to_csv(BASELINE_RESULTS_DIRECTORY / "baseline_summary.csv")
    print(f"\nSaved results to:\n{BASELINE_RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
