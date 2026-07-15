"""Standalone Metropolis-Hastings feature selection CLI."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

try:
    from .data_prep import prepare_data
    from .feature_eliminator import fmt_duration
    from .search_graph import save_search_graph
except ImportError:
    from data_prep import prepare_data
    from feature_eliminator import fmt_duration
    from search_graph import save_search_graph


def mcmc_updater(curr_state, curr_likelihood, likelihood, proposal_distribution):
    proposal_state = proposal_distribution(curr_state)
    proposal_likelihood = likelihood(proposal_state)
    acceptance_ratio = proposal_likelihood / curr_likelihood
    if acceptance_ratio > np.random.uniform(0, 1):
        return proposal_state, proposal_likelihood, True
    return curr_state, curr_likelihood, False


def metropolis_hastings(
    likelihood,
    proposal_distribution,
    initial_state,
    num_samples,
    burnin=0.2,
    on_step=None,
):
    chain, accepted = [], []
    burnin_idx = int(burnin * num_samples)
    current_state = initial_state
    current_likelihood = likelihood(current_state)

    for index in range(num_samples):
        current_state, current_likelihood, was_accepted = mcmc_updater(
            current_state,
            current_likelihood,
            likelihood,
            proposal_distribution,
        )
        chain.append(current_state)
        accepted.append(was_accepted)
        if on_step is not None:
            on_step(index, num_samples, current_state, current_likelihood, was_accepted)

    chain = np.array(chain, dtype=object)
    accepted = np.asarray(accepted, dtype=bool)
    return chain, chain[burnin_idx:], accepted, burnin_idx


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Metropolis-Hastings feature selection for time-series regression"
    )
    parser.add_argument("--data", type=Path, default=HERE / "data" / "masked_data.parquet")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--subset-size", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--burnin", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=0,
                        help="0 prints approximately 20 progress updates")
    parser.add_argument("--graph-every", type=int, default=50,
                        help="0 disables search-graph snapshots")
    parser.add_argument("--graph-dir", type=Path, default=HERE / "outputs" / "search_snapshots")
    parser.add_argument("--graph-window", type=int, default=None)
    parser.add_argument("--candidate-indices", type=Path, default=None,
                        help="optional .npy containing original feature indices")
    parser.add_argument("--output", type=Path,
                        default=HERE / "outputs" / "mcmc_selected_features.npy")
    parser.add_argument("--diagnostics", type=Path,
                        default=HERE / "outputs" / "mcmc_diagnostics.png")
    return parser.parse_args(argv)


def load_data(path: Path, candidate_indices: Path | None):
    df = pd.read_parquet(path)
    required = {"Timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    df = df.sort_values("Timestamp").reset_index(drop=True)

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
        X, names, original_indices = X[:, chosen], names[chosen], chosen
    return X, y, names, original_indices


def save_diagnostics(
    path,
    mae_trace,
    accepted,
    burnin_idx,
    inclusion,
    feature_names,
    persistence_mae,
):
    running_best = np.minimum.accumulate(mae_trace)
    kept_mae = mae_trace[burnin_idx:]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(mae_trace, color="#37474f", lw=0.7, label="Current MAE")
    axes[0, 0].plot(running_best, color="#e53935", lw=1.5, label="Best so far")
    axes[0, 0].axhline(persistence_mae, color="#1e88e5", ls=":", label="Persistence")
    axes[0, 0].axvspan(0, burnin_idx, color="#ff9800", alpha=0.2)
    axes[0, 0].set_title(f"MAE trace (acceptance {accepted.mean():.1%})")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("MAE (lower is better)")
    axes[0, 0].legend(fontsize=8)

    top = np.argsort(inclusion)[::-1][:15][::-1]
    axes[0, 1].barh(range(len(top)), inclusion[top], color="#1e88e5")
    axes[0, 1].set_yticks(range(len(top)))
    axes[0, 1].set_yticklabels([feature_names[i] for i in top], fontsize=7)
    axes[0, 1].set_title("Post-burn-in inclusion frequency")
    axes[0, 1].set_xlim(0, 1)

    axes[1, 0].plot(running_best, color="#e53935", lw=1.5)
    axes[1, 0].axhline(persistence_mae, color="#1e88e5", ls=":")
    axes[1, 0].set_title("Best MAE convergence")
    axes[1, 0].set_xlabel("Iteration")

    axes[1, 1].hist(kept_mae, bins=min(30, max(5, len(kept_mae))), color="#43a047", alpha=0.7)
    axes[1, 1].axvline(kept_mae.min(), color="#e53935", label=f"best {kept_mae.min():.4f}")
    axes[1, 1].set_title("Post-burn-in MAE distribution")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Metropolis-Hastings feature selection")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(args):
    if args.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not 0 <= args.burnin < 1:
        raise ValueError("burnin must be in [0, 1)")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    X, y, names, original_indices = load_data(args.data, args.candidate_indices)
    n_features = X.shape[1]
    if not 0 < args.subset_size < n_features:
        raise ValueError("subset-size must be between 1 and candidate_count - 1")

    model_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    fitness_cache = {}

    def fitness(subset):
        key = frozenset(subset)
        if key not in fitness_cache:
            model = GradientBoostingRegressor(**model_params)
            score = cross_val_score(
                model,
                X[:, sorted(key)],
                y,
                cv=TimeSeriesSplit(n_splits=args.cv_splits),
                scoring="neg_mean_absolute_error",
            ).mean()
            fitness_cache[key] = float(score)
        return fitness_cache[key]

    def likelihood(subset):
        return float(np.exp(np.clip(fitness(subset) / args.temperature, -300, 300)))

    pending = {}

    def proposal_distribution(subset):
        drop = random.choice(tuple(subset))
        inactive = [index for index in range(n_features) if index not in subset]
        add = random.choice(inactive)
        proposal = frozenset((set(subset) - {drop}) | {add})
        pending["from"], pending["proposal"] = subset, proposal
        return proposal

    cv = TimeSeriesSplit(n_splits=args.cv_splits)
    persistence_mae = float(np.mean([np.abs(y[val]).mean() for _, val in cv.split(X)]))
    initial_state = frozenset(random.sample(range(n_features), args.subset_size))
    report_every = args.report_every or max(1, args.iterations // 20)
    edges, node_mae = [], {}
    progress = {"accepts": 0, "best": float("inf")}
    started = time.time()

    print("=" * 72)
    print("METROPOLIS-HASTINGS FEATURE SELECTION")
    print("=" * 72)
    print(f"Data: {args.data} | rows: {X.shape[0]} | candidates: {n_features}")
    print(f"Subset: {args.subset_size} | iterations: {args.iterations} | T: {args.temperature}")
    print(f"Persistence baseline MAE: {persistence_mae:.6f}")

    def on_step(index, total, state, _likelihood, was_accepted):
        step = index + 1
        progress["accepts"] += int(was_accepted)
        current_mae = -fitness(state)
        progress["best"] = min(progress["best"], current_mae)
        source, proposal = pending["from"], pending["proposal"]
        edges.append((source, proposal, was_accepted))
        node_mae[source], node_mae[proposal] = -fitness(source), -fitness(proposal)

        if step == 1 or step % report_every == 0 or step == total:
            elapsed = time.time() - started
            eta = elapsed / step * (total - step)
            print(
                f"[{step:>5}/{total}] {100 * step / total:6.1f}% | "
                f"elapsed {fmt_duration(elapsed):>7} | ETA {fmt_duration(eta):>7} | "
                f"MAE now {current_mae:.6f} | best {progress['best']:.6f} | "
                f"accept {progress['accepts'] / step:.1%} | evals {len(fitness_cache)}",
                flush=True,
            )
        if args.graph_every and (step % args.graph_every == 0 or step == total):
            path = save_search_graph(
                edges,
                node_mae,
                step,
                args.graph_dir,
                names=names,
                window=args.graph_window,
            )
            print(f"  snapshot -> {path}", flush=True)

    chain, samples, accepted, burnin_idx = metropolis_hastings(
        likelihood,
        proposal_distribution,
        initial_state,
        args.iterations,
        burnin=args.burnin,
        on_step=on_step,
    )
    mae_trace = np.asarray([-fitness(state) for state in chain])
    best_index = int(np.argmin(mae_trace))
    best_subset = sorted(chain[best_index])
    inclusion = np.zeros(n_features)
    for sample in samples:
        for feature in sample:
            inclusion[feature] += 1
    inclusion /= max(len(samples), 1)

    selected_indices = original_indices[np.asarray(best_subset, dtype=int)]
    selected_names = names[np.asarray(best_subset, dtype=int)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, selected_indices)
    save_diagnostics(
        args.diagnostics,
        mae_trace,
        accepted,
        burnin_idx,
        inclusion,
        names,
        persistence_mae,
    )

    print("\n" + "=" * 72)
    print("BEST SUBSET FOUND")
    print("=" * 72)
    print(f"CV MAE: {mae_trace[best_index]:.6f} (persistence: {persistence_mae:.6f})")
    print(f"Original indices: {selected_indices.tolist()}")
    print(f"Feature names: {selected_names.tolist()}")
    print(f"Saved indices -> {args.output}")
    print(f"Saved diagnostics -> {args.diagnostics}")
    return selected_indices


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
