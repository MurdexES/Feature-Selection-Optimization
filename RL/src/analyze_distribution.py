from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "data_analysis"

def print_label_summary(dataframe: pd.DataFrame) -> None:
    counts = dataframe['label'].value_counts()
    percentages = dataframe['label'].value_counts(normalize=True).mul(100)

    summary = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    print("\n --- Global label distribution ---")
    print(summary)

def print_esp_label_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create a table showing fault labels available for each ESP."""

    table = pd.crosstab(
        index=dataframe["esp_id"],
        columns=dataframe["label"],
    )

    print("\n--- Label counts by ESP ---")
    print(table)

    print("\n--- Labels present for each ESP ---")
    for esp_id, row in table.iterrows():
        present_labels = row[row > 0].index.tolist()
        print(f"ESP {esp_id}: {present_labels}")

    return table


def save_label_distribution_plot(dataframe: pd.DataFrame) -> None:
    """Save a bar chart of the overall class distribution."""

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    counts = dataframe["label"].value_counts()

    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")

    plt.title("ESPset Label Distribution")
    plt.xlabel("Condition")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = RESULTS_DIRECTORY / "label_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved label-distribution plot to:\n{output_path}")


def save_esp_label_table(table: pd.DataFrame) -> None:
    """Save the ESP-by-label table for later reference."""

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIRECTORY / "esp_label_counts.csv"
    table.to_csv(output_path)

    print(f"Saved ESP-label table to:\n{output_path}")


def main() -> None:
    dataframe = load_features()

    print_label_summary(dataframe)

    esp_label_table = print_esp_label_table(dataframe)

    save_label_distribution_plot(dataframe)
    save_esp_label_table(esp_label_table)


if __name__ == "__main__":
    main()