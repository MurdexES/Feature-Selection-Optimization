from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "esp_condition_model.joblib"
)

FEATURE_COLUMNS = [
    "median(8,13)",
    "rms(98,102)",
    "median(98,102)",
    "peak1x",
    "peak2x",
    "a",
    "b",
]


class ConditionPredictor:
    """Predict ESP operating condition from vibration features."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train the condition model first."
            )

        self.model = joblib.load(model_path)

    def predict(
        self,
        features: dict[str, float],
    ) -> str:
        """Predict the most likely ESP condition."""

        dataframe = self._to_dataframe(features)
        prediction = self.model.predict(dataframe)

        return str(prediction[0])

    def predict_probabilities(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Return the predicted probability for every condition."""

        dataframe = self._to_dataframe(features)
        probabilities = self.model.predict_proba(dataframe)[0]

        return {
            str(label): float(probability)
            for label, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }

    def _to_dataframe(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:
        """Validate and convert one observation into a DataFrame."""

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in features
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        ordered_features = {
            feature: features[feature]
            for feature in FEATURE_COLUMNS
        }

        return pd.DataFrame([ordered_features])