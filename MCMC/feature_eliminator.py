import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold


def fmt_duration(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class FeatureEliminator:
    def __init__(self, task="regression", batch_size=5, n_iterations=3000,
                 penalty_threshold=5, n_splits=3, n_estimators=100, forest_max_depth=8,
                 min_features=60, shuffle=False, relative_thresholds=True,
                 warmup=50, window=200, low_pct=25, high_pct=75,
                 verbose=True, print_every=50):
        self.task = task
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.penalty_threshold = penalty_threshold
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.forest_max_depth = forest_max_depth
        self.min_features = min_features
        self.shuffle = shuffle
        self.relative_thresholds = relative_thresholds
        self.warmup = warmup
        self.window = window
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.verbose = verbose
        self.print_every = print_every

        self.penalty_scores = None
        self.eliminated = set()
        self.active_features = None
        self.iteration_log = []

    def _cv(self):
        if self.shuffle:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return TimeSeriesSplit(n_splits=self.n_splits)

    def _evaluate_batch(self, X, y, feature_indices):
        X_batch = X[:, feature_indices]

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.forest_max_depth,
            random_state=42,
            n_jobs=-1,
        )

        cv = self._cv()
        scores = []
        for train_idx, val_idx in cv.split(X_batch):
            model.fit(X_batch[train_idx], y[train_idx])
            preds = model.predict(X_batch[val_idx])
            scores.append(max(r2_score(y[val_idx], preds), -1.0))

        return np.mean(scores)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.penalty_scores = defaultdict(int)
        self.eliminated = set()
        self.active_features = list(range(n_features))

        if self.verbose:
            print("=" * 70)
            print("RANDOM FOREST FEATURE SCREENER (TimeSeriesSplit)")
            print("=" * 70)
            print(f"Model: RandomForest(n_estimators={self.n_estimators}, "
                  f"max_depth={self.forest_max_depth})")
            print(f"Features: {n_features} | batch: {self.batch_size} | "
                  f"iters: {self.n_iterations} | target survivors: {self.min_features}")
            print("-" * 70)
            print(f"{'Iter':>6} {'Active':>7} {'R2':>9} {'Result':>8} {'elapsed<ETA':>14}  Newly eliminated")

        t0 = time.time()
        for iteration in range(self.n_iterations):
            # Sample a small batch from the still-active features.
            if len(self.active_features) <= self.batch_size:
                batch = self.active_features.copy()
            else:
                batch = random.sample(self.active_features, self.batch_size)

            score = self._evaluate_batch(X, y, batch)
            result = "ok"
            newly_elim = []

            # Judge each batch against the recent score distribution (works even when
            # scores are negative). Bottom-percentile batches are the noise-heavy ones.
            if self.relative_thresholds and len(self.iteration_log) >= self.warmup:
                recent = [log['score'] for log in self.iteration_log[-self.window:]]
                bad_cut = np.percentile(recent, self.low_pct)
                good_cut = np.percentile(recent, self.high_pct)
            else:
                bad_cut = 0.0
                good_cut = 0.6

            if score < bad_cut:
                result = "BAD"
                for feat_idx in batch:
                    self.penalty_scores[feat_idx] += 1
                    if (self.penalty_scores[feat_idx] >= self.penalty_threshold
                            and feat_idx not in self.eliminated
                            and len(self.active_features) > self.min_features):
                        self.eliminated.add(feat_idx)
                        self.active_features.remove(feat_idx)
                        newly_elim.append(feat_idx)
            elif score >= good_cut:
                result = "GOOD"
                for feat_idx in batch:
                    if self.penalty_scores[feat_idx] > 0:
                        self.penalty_scores[feat_idx] -= 1

            self.iteration_log.append({
                'iteration': iteration + 1, 'score': score,
                'result': result, 'n_active': len(self.active_features), 'batch': batch,
            })

            if self.verbose and ((iteration + 1) % self.print_every == 0 or newly_elim):
                step = iteration + 1
                elapsed = time.time() - t0
                eta = elapsed / step * (self.n_iterations - step)   # upper bound (may early-stop)
                clock = f"{fmt_duration(elapsed)}<{fmt_duration(eta)}"
                elim_str = str(newly_elim) if newly_elim else "-"
                print(f"{step:>6} {len(self.active_features):>7} "
                      f"{score:>9.4f} {result:>8} {clock:>14}  {elim_str}", flush=True)

            # Stop once we have screened down to the target number of features.
            if len(self.active_features) <= self.min_features:
                if self.verbose:
                    print(f"\nReached target of {self.min_features} features. Stopping early.")
                break

        if self.verbose:
            print(f"\nScreening done: {n_features} -> {len(self.active_features)} active "
                  f"({len(self.eliminated)} eliminated)")
        return self.active_features

    def get_score_history(self):
        scores = [log['score'] for log in self.iteration_log]
        n_active = [log['n_active'] for log in self.iteration_log]
        results = [log['result'] for log in self.iteration_log]

        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['#e53935' if r == 'BAD' else '#43a047' if r == 'GOOD' else '#1e88e5'
                  for r in results]
        axes[0].scatter(range(len(scores)), scores, c=colors, s=10, alpha=0.6)
        axes[0].axhline(0.0, color='#e53935', ls='--', label='R2 = 0 (guessing the mean)')
        axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Batch R2')
        axes[0].set_title('Batch score per iteration (RandomForest)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(n_active, color='#1e88e5', lw=2)
        axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Active features')
        axes[1].set_title('Feature count over iterations')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()


def two_phase_screen(X_full, y, feature_names=None,
                     phase1_iters=5000, phase1_survivors=150,
                     phase2_iters=2000, phase2_survivors=60,
                     save_npy=None, save_csv=None, plot=True):
    # 2 Phase feature elimination, first bulk cut, second refinement of the rest features
    print(f"\n### Phase 1/2 - 30-tree forest, bulk cut {X_full.shape[1]} -> ~{phase1_survivors}")
    elim1 = FeatureEliminator(
        batch_size=5, n_iterations=phase1_iters, penalty_threshold=10,
        min_features=phase1_survivors, n_estimators=30, forest_max_depth=6,
        shuffle=False, relative_thresholds=True, verbose=True, print_every=200,
    )
    surv1 = np.array(sorted(elim1.fit(X_full, y)))          # indices in full space

    print(f"\n### Phase 2/2 - 100-tree forest, refine {len(surv1)} -> ~{phase2_survivors}")
    X_phase2 = X_full[:, surv1]
    elim2 = FeatureEliminator(
        batch_size=5, n_iterations=phase2_iters, penalty_threshold=10,
        min_features=phase2_survivors, n_estimators=100, forest_max_depth=8,
        shuffle=False, relative_thresholds=True, verbose=True, print_every=50,
    )
    surv2 = np.array(sorted(elim2.fit(X_phase2, y)))        # indices into X_phase2 columns
    final = np.array(sorted(surv1[surv2]))                  # map back to full space

    print(f"\nTwo-phase screening: {X_full.shape[1]} -> {len(surv1)} -> {len(final)}")

    if save_npy is not None:
        np.save(save_npy, final)
    if save_csv is not None:
        names = ([str(feature_names[j]) for j in final]
                 if feature_names is not None else [str(int(j)) for j in final])
        pd.DataFrame({"feature_idx": final, "feature_name": names}).to_csv(save_csv, index=False)
    if save_npy is not None or save_csv is not None:
        print(f"Saved {len(final)} survivors"
              + (f" -> {Path(save_npy).name}" if save_npy is not None else ""))

    if plot:
        elim2.get_score_history()   # progress of the refinement phase

    return final
