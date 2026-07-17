from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "models"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "condition_model"

FEATURE_COLUMNS = [
    "median(8,13)",
    "rms(98,102)",
    "median(98,102)",
    "peak1x",
    "peak2x",
    "a",
    "b",
]

TARGET_COLUMN = "label"

def prepare_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    
    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target

def create_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

def evaluate_model(
    model: Pipeline,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(test_features)

    macro_f1 = f1_score(
        test_target,
        predictions,
        average="macro",
    )

    balanced_accuracy = balanced_accuracy_score(
        test_target,
        predictions,
    )

    report = classification_report(
        test_target,
        predictions,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        test_target,
        predictions,
        labels=model.classes_,
    )

    print("\n--- Classification report ---")
    print(report)

    print("\n--- Confusion matrix ---")
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_,
    )
    print(matrix_dataframe)

    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIRECTORY / "classification_report.txt"
    report_path.write_text(report)

    matrix_path = RESULTS_DIRECTORY / "confusion_matrix.csv"
    matrix_dataframe.to_csv(matrix_path)

    metrics = {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
    }

    metrics_path = RESULTS_DIRECTORY / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    return metrics


def main() -> None:
    dataframe = load_features()

    features, target = prepare_data(dataframe)

    train_features, test_features, train_target, test_target = (
        train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=42,
            stratify=target,
        )
    )

    print("--- Dataset split ---")
    print(f"Training samples: {len(train_features)}")
    print(f"Testing samples: {len(test_features)}")

    print("\n--- Training label distribution ---")
    print(train_target.value_counts(normalize=True).round(4))

    print("\n--- Testing label distribution ---")
    print(test_target.value_counts(normalize=True).round(4))

    model = create_pipeline()

    print("\nTraining Random Forest...")
    model.fit(train_features, train_target)

    evaluate_model(
        model=model,
        test_features=test_features,
        test_target=test_target,
    )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIRECTORY / "esp_condition_model.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved trained model to:\n{model_path}")


if __name__ == "__main__":
    main()