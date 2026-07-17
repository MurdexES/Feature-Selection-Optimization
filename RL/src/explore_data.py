import pandas as pd

from data_loader import load_features

def display_basic_information(dataframe: pd.DataFrame) -> None:
    print("\n--- Shape ---")
    print(dataframe.shape)

    print("\n--- First five rows ---")
    print(dataframe.head())

    print("\n--- Column names ---")
    for column in dataframe.columns:
        print(f"- {column}")

    print("\n--- Data types ---")
    print(dataframe.dtypes)

    print("\n--- Missing values ---")
    print(dataframe.isna().sum())

    print("\n--- Unique values per column ---")
    print(dataframe.nunique())

    if "label" in dataframe.columns:
        print("\n--- Label distribution ---")
        print(dataframe["label"].value_counts(dropna=False))

        print("\n--- Label percentages ---")
        percentages = (
            dataframe["label"]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
        )
        print(percentages)

    if "esp_id" in dataframe.columns:
        print("\n--- Samples per ESP ---")
        print(dataframe["esp_id"].value_counts())


def main() -> None:
    features = load_features()
    display_basic_information(features)


if __name__ == "__main__":
    main()