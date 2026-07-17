from src.condition_predictor import (
    ConditionPredictor,
    FEATURE_COLUMNS,
)
from src.data_loader import load_features


def main() -> None:
    dataframe = load_features()
    predictor = ConditionPredictor()

    sample_rows = dataframe.sample(
        n=10,
        random_state=42,
    )

    for _, row in sample_rows.iterrows():
        feature_values = {
            feature: float(row[feature])
            for feature in FEATURE_COLUMNS
        }

        predicted_label = predictor.predict(feature_values)
        probabilities = predictor.predict_probabilities(
            feature_values
        )

        print("\n------------------------------")
        print(f"ESP ID: {row['esp_id']}")
        print(f"Actual condition: {row['label']}")
        print(f"Predicted condition: {predicted_label}")

        sorted_probabilities = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        print("Probabilities:")

        for label, probability in sorted_probabilities:
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()