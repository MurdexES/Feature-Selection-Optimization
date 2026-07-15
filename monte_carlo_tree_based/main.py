from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib_cache"))

try:
    from .feature_selection import make_initial_state
    from .MCTS import mcts_search
    from .tree_visualizer import save_tree_snapshot
except ImportError:
    from feature_selection import make_initial_state
    from MCTS import mcts_search
    from tree_visualizer import save_tree_snapshot

try:
    from .data_prep import fmt_duration, prepare_data
except ImportError:
    from data_prep import fmt_duration, prepare_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MCTS fixed-size feature selection for masked_data.parquet"
    )
    parser.add_argument("--data", type=Path, default=HERE / "data" / "masked_data.parquet")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--subset-size", type=int, default=6)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--report-every", type=int, default=0,
                        help="0 prints about 20 updates")
    parser.add_argument("--snapshot-every", type=int, default=100,
                        help="0 disables periodic PNG snapshots")
    parser.add_argument("--snapshot-dir", type=Path, default=HERE / "outputs" / "tree_snapshots")
    parser.add_argument(
        "--max-snapshot-nodes",
        type=int,
        default=180,
        help="maximum high-visit nodes drawn per snapshot (default: 180)",
    )
    parser.add_argument("--candidate-indices", type=Path, default=None,
                        help="optional .npy of original feature indices to search")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--exploration", type=float, default=math.sqrt(2.0))
    parser.add_argument("--widening-constant", type=float, default=1.5)
    parser.add_argument("--widening-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "mcts_selected_features.npy",
    )
    return parser.parse_args(argv)


class SubsetEvaluator:
    """Cached TimeSeriesSplit MAE evaluator with a bounded MCTS reward."""

    def __init__(self, X, y, cv_splits, model_params, persistence_mae):
        self.X = X
        self.y = y
        self.cv = TimeSeriesSplit(n_splits=cv_splits)
        self.model_params = model_params
        self.persistence_mae = max(float(persistence_mae), np.finfo(float).eps)
        self.cache = {}

    def mae(self, state):
        key = state.get_state()
        if key not in self.cache:
            model = GradientBoostingRegressor(**self.model_params)
            neg_mae = cross_val_score(
                model,
                self.X[:, list(key)],
                self.y,
                cv=self.cv,
                scoring="neg_mean_absolute_error",
            ).mean()
            self.cache[key] = float(-neg_mae) # is needed to store cache of feature combinations and their MAE score, to optimize search
        return self.cache[key]

    def __call__(self, state):
        # Bounded (0, 1] and monotonic: a smaller MAE gives a larger reward.
        return self.mae_to_reward(self.mae(state))

    def mae_to_reward(self, mae):
        return 1.0 / (1.0 + max(float(mae), 0.0) / self.persistence_mae)

    def reward_to_mae(self, reward):
        if reward <= 0:
            return float("inf")
        return self.persistence_mae * (1.0 / reward - 1.0)


def load_data(path: Path, candidate_indices: Optional[Path]):
    df = pd.read_parquet(path).sort_values("Timestamp").reset_index(drop=True)
    required = {"Timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    feature_frame = df.drop(columns=["Timestamp", "target"])
    names = feature_frame.columns.to_numpy(dtype=str)
    X, y, names = prepare_data(
        feature_frame.to_numpy(),
        df["target"].to_numpy(),
        names,
        target_transform="difference",
    )
    original_indices = np.arange(X.shape[1], dtype=int)

    if candidate_indices is not None:
        chosen = np.asarray(np.load(candidate_indices), dtype=int).reshape(-1)
        if len(chosen) == 0 or len(set(chosen.tolist())) != len(chosen):
            raise ValueError("candidate-indices must contain unique indices")
        if chosen.min() < 0 or chosen.max() >= X.shape[1]:
            raise ValueError("candidate-indices contains an out-of-range index")
        X = X[:, chosen]
        names = names[chosen]
        original_indices = chosen
    return X, y, names, original_indices


def run(args):
    np.random.seed(args.seed)
    X, y, names, original_indices = load_data(args.data, args.candidate_indices)
    if not 0 < args.subset_size <= X.shape[1]:
        raise ValueError("subset-size must be between 1 and the candidate feature count")

    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    persistence_mae = float(np.mean([np.abs(y[val]).mean() for _, val in cv.split(X)]))
    evaluator = SubsetEvaluator(
        X,
        y,
        args.cv_splits,
        dict(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=args.seed,
        ),
        persistence_mae,
    )
    initial_state = make_initial_state(names, args.subset_size)
    report_every = args.report_every or max(1, args.iterations // 20)
    started = time.time()
    snapshot_paths = []

    print("=" * 72)
    print("MCTS FEATURE SELECTION")
    print("=" * 72)
    print(f"Data: {args.data} | rows: {X.shape[0]} | candidates: {X.shape[1]}")
    print(f"Subset size: {args.subset_size} | iterations: {args.iterations} | "
          f"TimeSeriesSplit folds: {args.cv_splits}")
    print(f"Persistence baseline MAE: {persistence_mae:.6f}")
    print(f"Tree snapshots: {'disabled' if not args.snapshot_every else args.snapshot_dir}")

    def callback(step, total, node, state, reward, best_state, best_reward):
        current_mae = evaluator.mae(state)
        best_mae = evaluator.mae(best_state)
        if step % report_every == 0 or step == 1 or step == total:
            elapsed = time.time() - started
            eta = elapsed / step * (total - step)
            print(
                f"[{step:>5}/{total}] {100 * step / total:6.1f}% | "
                f"elapsed {fmt_duration(elapsed):>7} | ETA {fmt_duration(eta):>7} | "
                f"MAE now {current_mae:.6f} | best {best_mae:.6f} | "
                f"depth {len(node.game_state.selected)} | evals {len(evaluator.cache)}",
                flush=True,
            )
        if args.snapshot_every and (step % args.snapshot_every == 0 or step == total):
            root = node
            while root.parent is not None:
                root = root.parent
            path = save_tree_snapshot(
                root,
                step,
                args.snapshot_dir,
                evaluator.reward_to_mae,
                best_state.get_state(),
                args.max_snapshot_nodes,
            )
            snapshot_paths.append(path)
            print(f"  snapshot -> {path}", flush=True)

    result = mcts_search(
        initial_state,
        evaluator,
        n_iterations=args.iterations,
        exploration_constant=args.exploration,
        widening_constant=args.widening_constant,
        widening_alpha=args.widening_alpha,
        random_state=args.seed,
        on_iteration=callback,
    )

    local_indices = np.array(result.best_state.get_state(), dtype=int)
    selected_indices = original_indices[local_indices]
    selected_names = names[local_indices]
    best_mae = evaluator.mae(result.best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, selected_indices)

    print("\n" + "=" * 72)
    print("BEST SUBSET FOUND")
    print("=" * 72)
    print(f"CV MAE: {best_mae:.6f} (persistence: {persistence_mae:.6f})")
    print(f"Original indices: {selected_indices.tolist()}")
    print(f"Feature names: {selected_names.tolist()}")
    print(f"Saved indices -> {args.output}")
    if snapshot_paths:
        print(f"Latest tree image -> {snapshot_paths[-1]}")
    return result


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
