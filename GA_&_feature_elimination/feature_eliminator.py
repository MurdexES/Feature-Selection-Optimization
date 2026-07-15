from collections import defaultdict

import pandas as pd
import random
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.tree import DecisionTreeRegressor

class FeatureEliminator:
    def __init__(
            self,
            task = "regression",
            batch_size = 3,                 # small batches so all-noise batches occur & score low
            n_iterations = 300,
            penalty_threshold = 5,
            bad_score_threshold = 0.0,      # only used when relative_thresholds=False
            good_score_threshold = 0.6,     # only used when relative_thresholds=False
            n_splits = 3,
            tree_max_depth = 5,
            min_features = 30,
            shuffle = False,                # False -> TimeSeriesSplit (time-aware, no leakage)
            relative_thresholds = True,     # judge each batch vs the recent score distribution
            warmup = 50,                    # iterations before relative thresholds kick in
            window = 200,                   # rolling window of recent scores for the percentiles
            low_pct = 25,                   # batch is BAD  if score below this percentile
            high_pct = 75,                  # batch is GOOD if score above this percentile
            verbose = True
        ):
            self.task = task
            self.batch_size = batch_size
            self.n_iterations = n_iterations
            self.penalty_threshold = penalty_threshold
            self.bad_score_threshold = bad_score_threshold
            self.good_score_threshold = good_score_threshold
            self.n_splits = n_splits
            self.tree_max_depth = tree_max_depth
            self.min_features = min_features
            self.shuffle = shuffle
            self.relative_thresholds = relative_thresholds
            self.warmup = warmup
            self.window = window
            self.low_pct = low_pct
            self.high_pct = high_pct
            self.verbose = verbose

            self.penalty_scores  = None
            self.eliminated      = set()
            self.active_features = None
            self.iteration_log   = []

    def _cv(self):
        if self.shuffle:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return TimeSeriesSplit(n_splits=self.n_splits)

    def _evaluate_batch(self, X, y, feature_indices):
        X_batch = X[:, feature_indices]

        tree = DecisionTreeRegressor(
            max_depth=self.tree_max_depth,
            random_state=42
        )

        cv = self._cv()
        scores = []

        for train_idx, val_idx in cv.split(X_batch):
            X_train = X_batch[train_idx]
            X_val = X_batch[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            tree.fit(X_train, y_train)

            preds = tree.predict(X_val)
            score = r2_score(y_val, preds)
            score = max(score, -1.0)

            scores.append(score)

        return np.mean(scores)

    def fit(self, X, y):
        n_features = X.shape[1]

        self.penalty_scores = defaultdict(int)
        self.eliminated = set()
        self.active_features = list(range(n_features))

        metric_name = "R²" if self.task == 'regression' else "AUC"

        if self.verbose:
            print("=" * 70)
            print("TIME SERIES FEATURE SCREENER")
            print("=" * 70)
            print(f"Task: {self.task}")
            print(f"Evaluation: TimeSeriesSplit (n_splits={self.n_splits})")
            print(f"Tree: Decision Tree (max_depth={self.tree_max_depth})")
            print(f"Total features: {n_features}")
            print(f"Batch size: {self.batch_size}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Penalty threshold: {self.penalty_threshold}")
            print(f"Bad {metric_name} threshold:  < {self.bad_score_threshold}")
            print(f"Good {metric_name} threshold: >= {self.good_score_threshold}")
            print(f"Min features: {self.min_features}")
            print("=" * 70)
            print(f"\n{'Iter':>5} {'Active':>7} {metric_name:>10} "
                  f"{'Result':>8} {'Newly Eliminated'}")
            print("-" * 70)

        for iteration in range(self.n_iterations):

            # Sample batch from active features only
            if len(self.active_features) <= self.batch_size:
                batch = self.active_features.copy()
            else:
                batch = random.sample(self.active_features, self.batch_size)

            score = self._evaluate_batch(X, y, batch)
            result = "ok"
            newly_elim = []

            # Adaptive cutoffs: since KFold scores cluster high (e.g. 0.9-0.99),
            # fixed 0.0/0.6 thresholds never separate anything. Instead compare each
            # batch against the recent score distribution — the worst batches (bottom
            # percentile) are the most noise-heavy and get penalized.
            if self.relative_thresholds and len(self.iteration_log) >= self.warmup:
                recent = [log['score'] for log in self.iteration_log[-self.window:]]
                bad_cut = np.percentile(recent, self.low_pct)
                good_cut = np.percentile(recent, self.high_pct)
            else:
                bad_cut = self.bad_score_threshold
                good_cut = self.good_score_threshold

            if score < bad_cut:
                # Bad batch — all features in it get penalized
                result = "BAD"

                for feat_idx in batch:
                    self.penalty_scores[feat_idx] += 1

                    # Eliminate if penalty threshold reached
                    if (self.penalty_scores[feat_idx] >= self.penalty_threshold
                            and feat_idx not in self.eliminated
                            and len(self.active_features) > self.min_features):

                        self.eliminated.add(feat_idx)
                        self.active_features.remove(feat_idx)
                        newly_elim.append(feat_idx)

            elif score >= good_cut:
                # Good batch — reward features by reducing their penalty
                result = "GOOD"
                for feat_idx in batch:
                    if self.penalty_scores[feat_idx] > 0:
                        self.penalty_scores[feat_idx] -= 1

            self.iteration_log.append({
                'iteration': iteration + 1,
                'score': score,
                'result': result,
                'n_active': len(self.active_features),
                'batch': batch
            })

            if self.verbose:
                elim_str = str(newly_elim) if newly_elim else "-"
                print(f"{iteration+1:>5} {len(self.active_features):>7} "
                      f"{score:>10.4f} {result:>8}  {elim_str}")

            # Stop early if enough features eliminated
            if len(self.active_features) <= self.min_features:
                if self.verbose:
                    print(f"\nReached minimum features "
                          f"({self.min_features}). Stopping early.")
                break

        if self.verbose:
            self._print_summary(n_features, metric_name)

        return self.active_features

    def _print_summary(self, original_n, metric_name):
        print("\n" + "=" * 70)
        print("SCREENING COMPLETE")
        print("=" * 70)
        print(f"  Original features: {original_n}")
        print(f"  Eliminated: {len(self.eliminated)}")
        print(f"  Surviving: {len(self.active_features)}")
        print(f"  Reduction: {len(self.eliminated)/original_n:.1%} of features removed")

        # Score distribution from log
        scores = [log['score'] for log in self.iteration_log]
        print(f"\n  {metric_name} score distribution across batches:")
        print(f"    Mean: {np.mean(scores):.4f}")
        print(f"    Median: {np.median(scores):.4f}")
        print(f"    Min: {np.min(scores):.4f}")
        print(f"    Max: {np.max(scores):.4f}")
        print(f"    Bad batches: {sum(1 for s in scores if s < self.bad_score_threshold)}"
              f" / {len(scores)} "
              f"({sum(1 for s in scores if s < self.bad_score_threshold)/len(scores):.1%})")

        # Most penalized features
        if self.penalty_scores:
            worst = sorted(self.penalty_scores.items(),
                          key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Most penalized features:")
            print(f"  {'Feature':>8} {'Penalty':>8} {'Status':>12}")
            print(f"  {'-'*30}")
            for feat_idx, score in worst:
                status = "ELIMINATED" if feat_idx in self.eliminated else "survived"
                print(f"  {feat_idx:>8} {score:>8} {status:>12}")
        print("=" * 70)

    def get_score_history(self):
        """Plot score over iterations to see screening progress."""
        import matplotlib.pyplot as plt

        scores   = [log['score'] for log in self.iteration_log]
        n_active = [log['n_active'] for log in self.iteration_log]
        results  = [log['result'] for log in self.iteration_log]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: score over iterations
        colors = ['red' if r == 'BAD' else
                  'green' if r == 'GOOD' else
                  'steelblue' for r in results]

        axes[0].scatter(range(len(scores)), scores,
                       c=colors, s=10, alpha=0.6)
        axes[0].axhline(self.bad_score_threshold,
                       color='red', linestyle='--',
                       label=f'Bad threshold ({self.bad_score_threshold})')
        axes[0].axhline(self.good_score_threshold,
                       color='green', linestyle='--',
                       label=f'Good threshold ({self.good_score_threshold})')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Batch Score Per Iteration')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: active features over time
        axes[1].plot(n_active, color='steelblue', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Active Features')
        axes[1].set_title('Feature Count Over Iterations')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
