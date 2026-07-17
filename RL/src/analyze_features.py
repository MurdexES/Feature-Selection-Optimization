from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "data_analysis"

FEATURE_COLUMNS = [
    "median(8,13)",
    "rms(98,102)",
    "median(98,102)",
    "peak1x",
    "peak2x",
    "a",
    "b",
]


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Ensure all expected feature columns exist."""

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )


def print_feature_statistics(dataframe: pd.DataFrame) -> None:
    """Print descriptive statistics for numerical model features."""

    statistics = dataframe[FEATURE_COLUMNS].describe().transpose()

    print("\n--- Feature statistics ---")
    print(statistics)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIRECTORY / "feature_statistics.csv"
    statistics.to_csv(output_path)

    print(f"\nSaved feature statistics to:\n{output_path}")


def save_feature_histograms(dataframe: pd.DataFrame) -> None:
    """Save one histogram for every model feature."""

    histogram_directory = RESULTS_DIRECTORY / "feature_histograms"
    histogram_directory.mkdir(parents=True, exist_ok=True)

    for feature in FEATURE_COLUMNS:
        plt.figure(figsize=(9, 5))
        plt.hist(
            dataframe[feature],
            bins=50,
        )

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()

        safe_name = (
            feature
            .replace("(", "_")
            .replace(")", "")
            .replace(",", "_")
        )

        output_path = histogram_directory / f"{safe_name}.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

    print(
        "\nSaved feature histograms to:\n"
        f"{histogram_directory}"
    )


def print_statistics_by_label(dataframe: pd.DataFrame) -> None:
    """Compare median feature values between conditions."""

    grouped_medians = (
        dataframe
        .groupby("label")[FEATURE_COLUMNS]
        .median()
        .round(6)
    )

    print("\n--- Median feature values by label ---")
    print(grouped_medians)

    output_path = RESULTS_DIRECTORY / "feature_medians_by_label.csv"
    grouped_medians.to_csv(output_path)

    print(f"\nSaved label feature medians to:\n{output_path}")


def main() -> None:
    dataframe = load_features()

    validate_columns(dataframe)
    print_feature_statistics(dataframe)
    print_statistics_by_label(dataframe)
    save_feature_histograms(dataframe)


if __name__ == "__main__":
    main()