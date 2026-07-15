"""Optional standalone feature-screening stage for the genetic algorithm."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .feature_eliminator import FeatureEliminator
except ImportError:
    from feature_eliminator import FeatureEliminator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Screen candidate features before GA")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "masked_data.parquet")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--min-features", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--penalty-threshold", type=int, default=5)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--tree-max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=HERE / "outputs" / "surviving_features.npy")
    parser.add_argument("--output-csv", type=Path,
                        default=HERE / "outputs" / "surviving_features.csv")
    parser.add_argument("--diagnostics", type=Path,
                        default=HERE / "outputs" / "screening_diagnostics.png")
    return parser.parse_args(argv)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    df = pd.read_parquet(args.data)
    required = {"Timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    feature_frame = df.drop(columns=["Timestamp", "target"])
    X, y = feature_frame.to_numpy(), df["target"].to_numpy()
    if not 1 <= args.min_features <= X.shape[1]:
        raise ValueError("min-features must be between 1 and the feature count")

    eliminator = FeatureEliminator(
        task="regression",
        batch_size=args.batch_size,
        n_iterations=args.iterations,
        penalty_threshold=args.penalty_threshold,
        n_splits=args.cv_splits,
        tree_max_depth=args.tree_max_depth,
        min_features=args.min_features,
        shuffle=False,
        relative_thresholds=True,
        verbose=True,
    )
    surviving = np.asarray(eliminator.fit(X, y), dtype=int)
    names = feature_frame.columns[surviving].astype(str).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, surviving)
    pd.DataFrame({"feature_idx": surviving, "feature_name": names}).to_csv(
        args.output_csv, index=False
    )
    figure = eliminator.get_score_history()
    figure.savefig(args.diagnostics, dpi=140, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {len(surviving)} feature indices -> {args.output}")
    print(f"Saved feature names -> {args.output_csv}")
    print(f"Saved diagnostics -> {args.diagnostics}")
    return surviving


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
