import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from src.data_loader import load_features
from src.train_condition_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_pipeline,
)


def evaluate_one_pump(
    dataframe: pd.DataFrame,
    test_esp_id: int,
) -> dict[str, float | int]:
    """Train on all other pumps and test on one unseen pump."""

    train_data = dataframe[
        dataframe["esp_id"] != test_esp_id
    ].copy()

    test_data = dataframe[
        dataframe["esp_id"] == test_esp_id
    ].copy()

    train_labels = set(train_data[TARGET_COLUMN].unique())
    test_labels = set(test_data[TARGET_COLUMN].unique())

    unseen_labels = test_labels - train_labels

    if unseen_labels:
        print(
            f"\nESP {test_esp_id} contains labels absent "
            f"from training: {sorted(unseen_labels)}"
        )

    model = create_pipeline()

    model.fit(
        train_data[FEATURE_COLUMNS],
        train_data[TARGET_COLUMN],
    )

    predictions = model.predict(
        test_data[FEATURE_COLUMNS]
    )

    macro_f1 = f1_score(
        test_data[TARGET_COLUMN],
        predictions,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        test_data[TARGET_COLUMN],
        predictions,
    )

    print(f"\n=== Test ESP {test_esp_id} ===")
    print(f"Test samples: {len(test_data)}")
    print(
        classification_report(
            test_data[TARGET_COLUMN],
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    return {
        "test_esp_id": test_esp_id,
        "test_samples": len(test_data),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }


def main() -> None:
    dataframe = load_features()

    all_results = []

    for esp_id in sorted(dataframe["esp_id"].unique()):
        result = evaluate_one_pump(
            dataframe=dataframe,
            test_esp_id=int(esp_id),
        )

        all_results.append(result)

    result_dataframe = pd.DataFrame(all_results)

    print("\n=== Leave-one-pump-out summary ===")
    print(result_dataframe.round(4))

    print("\n=== Mean performance ===")
    print(
        result_dataframe[
            ["macro_f1", "balanced_accuracy"]
        ].mean()
    )


if __name__ == "__main__":
    main()