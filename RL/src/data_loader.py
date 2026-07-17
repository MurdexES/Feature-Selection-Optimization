from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "raw" / "features.csv"

def load_features(path: Path = DEFAULT_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place features.csv inside data/raw/."
        )
    
    dataframe = pd.read_csv(path, sep=";")

    if dataframe.empty:
        raise ValueError("The loaded dataset is empty")
    
    return dataframe

if __name__ == "__main__":
    features = load_features()

    print("Dataset loaded successfully")
    print(f"Shape: {features.shape}")
    print(f"Column: {features.columns.tolist()}")
