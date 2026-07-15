"""Standalone genetic hyperparameter and feature-selection CLI."""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

try:
    from .checkpoint_manager import CheckpointManager
    from .fitness_cache import FitnessCache
    from .performance_tracker import PerformanceTracker
except ImportError:
    from checkpoint_manager import CheckpointManager
    from fitness_cache import FitnessCache
    from performance_tracker import PerformanceTracker

HERE = Path(__file__).resolve().parent

PARAM_SPACE = {
    "n_estimators": [50, 100, 150, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
    "min_samples_leaf": [1, 2, 5, 10, 15, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Genetic hyperparameter optimization and feature selection"
    )
    parser.add_argument("--data", type=Path, default=HERE / "data" / "masked_data.parquet")
    parser.add_argument("--candidate-indices", type=Path, default=None,
                        help="optional .npy containing original feature indices")
    parser.add_argument("--population-size", type=int, default=150)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=7)
    parser.add_argument("--elitism", type=int, default=3)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--diversity-threshold", type=float, default=0.15)
    parser.add_argument("--feature-penalty", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=HERE / "outputs" / "checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest trusted local checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=HERE / "outputs" / "ga_selected_features.npy")
    parser.add_argument("--result-json", type=Path,
                        default=HERE / "outputs" / "ga_result.json")
    return parser.parse_args(argv)


def load_data(path: Path, candidate_indices: Path | None):
    df = pd.read_parquet(path)
    required = {"Timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    feature_frame = df.drop(columns=["Timestamp", "target"])
    X_full = feature_frame.to_numpy()
    y = df["target"].to_numpy()
    names = feature_frame.columns.to_numpy(dtype=str)
    original_indices = np.arange(X_full.shape[1], dtype=int)

    if candidate_indices is not None:
        chosen = np.asarray(np.load(candidate_indices), dtype=int).reshape(-1)
        if len(chosen) == 0 or len(set(chosen.tolist())) != len(chosen):
            raise ValueError("candidate-indices must contain unique indices")
        if chosen.min() < 0 or chosen.max() >= X_full.shape[1]:
            raise ValueError("candidate-indices contains an out-of-range index")
        return X_full[:, chosen], y, names[chosen], chosen, X_full.shape[1]
    return X_full, y, names, original_indices, X_full.shape[1]


class GeneticFeatureSelector:
    def __init__(self, X, y, original_indices, feature_names, args):
        self.X = X
        self.y = y
        self.original_indices = original_indices
        self.feature_names = feature_names
        self.args = args
        self.n_features = X.shape[1]
        self.param_names = list(PARAM_SPACE)
        self.param_values = list(PARAM_SPACE.values())
        self.hyperparameter_length = len(self.param_values)
        self.fitness_cache = FitnessCache()
        self.stop_requested = False

    def create_individual(self):
        genes = [random.randrange(len(values)) for values in self.param_values]
        feature_mask = [0] * self.n_features
        count = random.randint(self.args.min_features, self.args.max_features)
        for index in random.sample(range(self.n_features), count):
            feature_mask[index] = 1
        return genes + feature_mask

    def decode(self, individual):
        genes = individual[: self.hyperparameter_length]
        params = {
            name: self.param_values[index][gene]
            for index, (name, gene) in enumerate(zip(self.param_names, genes))
        }
        mask = individual[self.hyperparameter_length :]
        selected = [index for index, bit in enumerate(mask) if bit]
        return params, selected

    def fitness(self, individual):
        cached = self.fitness_cache.get(individual)
        if cached is not None:
            return cached
        params, selected = self.decode(individual)
        model = GradientBoostingRegressor(**params, random_state=self.args.seed)
        cv = KFold(n_splits=self.args.cv_splits, shuffle=True, random_state=self.args.seed)
        score = float(
            cross_val_score(
                model,
                self.X[:, selected],
                self.y,
                cv=cv,
                scoring="r2",
            ).mean()
        )
        score -= self.args.feature_penalty * len(selected) / self.n_features
        self.fitness_cache.set(individual, score)
        return score

    def tournament_selection(self, population, fitnesses):
        competitors = random.sample(
            list(zip(population, fitnesses)), self.args.tournament_size
        )
        return max(competitors, key=lambda item: item[1])[0]

    @staticmethod
    def uniform_crossover(parent1, parent2):
        child1 = [a if random.random() < 0.5 else b for a, b in zip(parent1, parent2)]
        child2 = [b if random.random() < 0.5 else a for a, b in zip(parent1, parent2)]
        return child1, child2

    def mutate(self, individual, mutation_rate):
        mutated = []
        for index, gene in enumerate(individual):
            if random.random() >= mutation_rate:
                mutated.append(gene)
            elif index < self.hyperparameter_length:
                mutated.append(random.randrange(len(self.param_values[index])))
            else:
                mutated.append(1 - gene)

        feature_offset = self.hyperparameter_length
        active = [
            feature_offset + index
            for index, bit in enumerate(mutated[feature_offset:])
            if bit
        ]
        if len(active) > self.args.max_features:
            for index in random.sample(active, len(active) - self.args.max_features):
                mutated[index] = 0
        elif len(active) < self.args.min_features:
            inactive = [
                feature_offset + index
                for index, bit in enumerate(mutated[feature_offset:])
                if not bit
            ]
            for index in random.sample(inactive, self.args.min_features - len(active)):
                mutated[index] = 1
        return mutated

    def diversity(self, population):
        offset = self.hyperparameter_length
        feature_sets = [
            frozenset(index for index, bit in enumerate(individual[offset:]) if bit)
            for individual in population
        ]
        total, comparisons = 0.0, 0
        for left in range(len(feature_sets)):
            for right in range(left + 1, len(feature_sets)):
                union = feature_sets[left] | feature_sets[right]
                distance = 1 - len(feature_sets[left] & feature_sets[right]) / len(union)
                total += distance
                comparisons += 1
        return total / comparisons if comparisons else 0.0

    def request_stop(self, _signum, _frame):
        self.stop_requested = True
        print("\nStop requested; the current generation will finish and checkpoint.")

    def run(self):
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.args.checkpoint_dir),
            save_interval=self.args.checkpoint_every,
            max_checkpoints=self.args.max_checkpoints,
        )
        state = checkpoint_manager.load_latest() if self.args.resume else None
        if state:
            population = state["population"]
            expected_length = self.hyperparameter_length + self.n_features
            if not population or len(population[0]) != expected_length:
                raise ValueError("checkpoint is incompatible with the current candidate pool")
            best_overall = state["best_overall"]
            best_overall_fit = state["best_overall_fit"]
            history = state["history"]
            no_improvement = state["no_improvement"]
            mutation_rate = state["mutation_rate"]
            start_generation = state["generation"] + 1
            self.fitness_cache = state["fitness_cache"]
        else:
            population = [self.create_individual() for _ in range(self.args.population_size)]
            best_overall = None
            best_overall_fit = float("-inf")
            history = []
            no_improvement = 0
            mutation_rate = self.args.mutation_rate
            start_generation = 0

        tracker = PerformanceTracker()
        tracker.start()
        signal.signal(signal.SIGINT, self.request_stop)
        print("=" * 80)
        print("GENETIC HYPERPARAMETER AND FEATURE SELECTION")
        print("=" * 80)
        print(f"Rows: {self.X.shape[0]} | candidates: {self.n_features} | "
              f"population: {self.args.population_size} | generations: {self.args.generations}")

        def save_checkpoint(generation):
            checkpoint_manager.save(
                generation,
                {
                    "population": population,
                    "best_overall": best_overall,
                    "best_overall_fit": best_overall_fit,
                    "history": history,
                    "mutation_rate": mutation_rate,
                    "no_improvement": no_improvement,
                    "generation": generation,
                    "fitness_cache": self.fitness_cache,
                },
            )

        completed_generations = start_generation
        for generation in range(start_generation, self.args.generations):
            tracker.start_generation()
            fitnesses = [self.fitness(individual) for individual in population]
            best_index = int(np.argmax(fitnesses))
            best_fit = fitnesses[best_index]
            history.append(best_fit)
            _, generation_features = self.decode(population[best_index])

            if best_fit > best_overall_fit + 0.0001:
                best_overall_fit = best_fit
                best_overall = population[best_index].copy()
                no_improvement = 0
                mutation_rate = max(0.10, mutation_rate * 0.9)
                status = "improved"
            else:
                no_improvement += 1
                mutation_rate = min(0.45, mutation_rate * 1.2)
                status = f"no improvement {no_improvement}/{self.args.patience}"

            generation_time, ram_mb = tracker.end_generation()
            print(
                f"{generation + 1:>4}/{self.args.generations} | "
                f"current {best_fit:>8.4f} | best {best_overall_fit:>8.4f} | "
                f"features {len(generation_features):>3} | mutation {mutation_rate:.3f} | "
                f"{generation_time:.1f}s | {ram_mb:.1f}MB | {status}",
                flush=True,
            )

            if self.diversity(population) < self.args.diversity_threshold:
                for index in np.argsort(fitnesses)[: min(5, len(population))]:
                    population[int(index)] = self.create_individual()
                print("     low diversity: injected fresh individuals", flush=True)

            if no_improvement >= self.args.patience:
                population = [
                    self.create_individual() for _ in range(self.args.population_size - 1)
                ] + [best_overall.copy()]
                no_improvement = 0
                mutation_rate = self.args.mutation_rate
                print("     patience reached: restarted population", flush=True)
            else:
                elite_count = min(self.args.elitism, len(population))
                elite_indices = np.argsort(fitnesses)[-elite_count:]
                next_population = (
                    [population[int(index)].copy() for index in elite_indices]
                    if elite_count else []
                )
                while len(next_population) < self.args.population_size:
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                    next_population.extend(
                        [self.mutate(child1, mutation_rate), self.mutate(child2, mutation_rate)]
                    )
                population = next_population[: self.args.population_size]

            completed_generations = generation + 1
            if checkpoint_manager.should_save(generation) or self.stop_requested:
                save_checkpoint(generation)
            if self.stop_requested:
                print(f"Stopped after generation {completed_generations}.")
                break

        if best_overall is None:
            raise RuntimeError("no generation was evaluated; checkpoint may already be complete")
        best_params, best_features = self.decode(best_overall)
        original = self.original_indices[np.asarray(best_features, dtype=int)]
        names = self.feature_names[np.asarray(best_features, dtype=int)]
        clean_model = GradientBoostingRegressor(**best_params, random_state=self.args.seed)
        clean_scores = cross_val_score(
            clean_model,
            self.X[:, best_features],
            self.y,
            cv=KFold(n_splits=self.args.cv_splits, shuffle=True, random_state=self.args.seed),
            scoring="r2",
        )

        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.args.result_json.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.args.output, original)
        result = {
            "original_indices": original.tolist(),
            "feature_names": names.tolist(),
            "best_parameters": best_params,
            "penalized_cv_r2": float(best_overall_fit),
            "clean_cv_r2_mean": float(clean_scores.mean()),
            "clean_cv_r2_std": float(clean_scores.std()),
            "completed_generations": completed_generations,
            "seed": self.args.seed,
        }
        self.args.result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        print(f"Original indices: {result['original_indices']}")
        print(f"Feature names: {result['feature_names']}")
        print(f"Best parameters: {best_params}")
        print(f"Penalized CV R2: {best_overall_fit:.6f}")
        print(f"Clean CV R2: {clean_scores.mean():.6f} ± {clean_scores.std():.6f}")
        print(f"Saved indices -> {self.args.output}")
        print(f"Saved result -> {self.args.result_json}")
        self.fitness_cache.stats()
        tracker.print_summary(completed_generations)
        return result


def validate_args(args, candidate_count):
    if args.population_size < 2:
        raise ValueError("population-size must be at least 2")
    if args.generations < 1:
        raise ValueError("generations must be at least 1")
    if not 1 <= args.min_features <= args.max_features <= candidate_count:
        raise ValueError("feature limits must satisfy 1 <= min <= max <= candidate count")
    if not 0 <= args.elitism < args.population_size:
        raise ValueError("elitism must be between 0 and population-size - 1")
    if not 2 <= args.tournament_size <= args.population_size:
        raise ValueError("tournament-size must be between 2 and population-size")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be at least 1")


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    X, y, names, original_indices, original_count = load_data(
        args.data, args.candidate_indices
    )
    validate_args(args, X.shape[1])
    source = str(args.candidate_indices) if args.candidate_indices else "all features"
    print(f"Candidate pool: {X.shape[1]} of {original_count} ({source})")
    return GeneticFeatureSelector(X, y, original_indices, names, args).run()


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
