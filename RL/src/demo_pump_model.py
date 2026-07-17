from src.simulation.pump_model import PumpModel

def run_without_maintenance() -> None:
    pump = PumpModel(seed=42)
    pump.reset()

    print("Running Pump without Maint.")

    for hour in range(1, 5001):
        result = pump.operate(load=0.9)

        if hour % 250 == 0 or result.failed:
            print(
                f"Hour={hour:4d} | "
                f"Health={result.health:.3f} | "
                f"Condition={result.condition.name:12s} | "
                f"Vibration={result.sensors.vibration:.2f} | "
                f"Temperature={result.sensors.temperature:.2f} | "
                f"Flow={result.sensors.flow_rate:.2f}"
            )

        if result.failed:
            print(
                f"\nPump failed after {hour} operating hours."
            )
            break
    else:
        print("\nPump did not fail during the simulation.")


def run_with_preventive_maintenance() -> None:
    pump = PumpModel(seed=42)
    pump.reset()

    print("\n--- Preventive maintenance every 1500 hours ---")

    for hour in range(1, 5001):
        if (
            pump.hours_since_maintenance >= 1500
            and not pump.failed
        ):
            pump.preventive_maintenance()

            print(
                f"Preventive maintenance at hour {hour}. "
                f"Restored health={pump.health:.3f}"
            )

        result = pump.operate(load=0.9)

        if result.failed:
            print(
                f"Unexpected failure at hour {hour}. "
                "Performing corrective maintenance."
            )

            pump.corrective_maintenance()

    print(
        "\nSimulation finished."
        f"\nTotal failures: {pump.total_failures}"
        f"\nFinal health: {pump.health:.3f}"
    )


def main() -> None:
    run_without_maintenance()
    run_with_preventive_maintenance()


if __name__ == "__main__":
    main()