"""Train DQN (and optionally Double DQN) and evaluate against baselines.

    cd RL && python -m src.training.train_dqn            # standard DQN
    cd RL && python -m src.training.train_dqn --double   # Double DQN

Same loop shape as tabular training, with two differences:

  * the agent consumes the env's normalised Box observation directly --
    no discretizer, the network handles continuous inputs;
  * learning happens on replayed minibatches (agent.train_step() once per
    environment step), not on the live transition.

Same exploring starts, same gamma, same evaluation protocol, so any
performance difference against tabular Q-learning is attributable to
function approximation, not to a different training regime.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.agents.dqn import DQNAgent, DQNConfig, DQNPolicy
from src.environment.maintenance_env import PumpMaintenanceEnv
from src.evaluation.evaluate_policy import evaluate_policy, summarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

N_EPISODES = 250
TRAINING_SEED = 0


def train(
    config: DQNConfig,
    n_episodes: int = N_EPISODES,
    seed: int = TRAINING_SEED,
) -> tuple[DQNAgent, pd.DataFrame]:
    env = PumpMaintenanceEnv(exploring_starts=True)
    agent = DQNAgent(config=config, seed=seed)

    history = []

    for episode in range(n_episodes):
        epsilon = agent.epsilon_for_episode(episode)

        observation, info = env.reset(seed=seed + episode)

        episode_reward = 0.0
        failures = 0
        preventive = 0
        losses = []

        while True:
            action = agent.choose_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, info = env.step(
                int(action)
            )

            agent.remember(
                observation, action, reward, next_observation, terminated
            )
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            observation = next_observation
            episode_reward += reward

            outcome = info["outcome"]
            failures += outcome.failed_this_hour
            preventive += outcome.preventive_started

            if terminated or truncated:
                break

        history.append(
            {
                "episode": episode,
                "epsilon": epsilon,
                "total_reward": episode_reward,
                "failures": failures,
                "preventive_maintenance": preventive,
                "mean_loss": float(np.mean(losses)) if losses else None,
            }
        )

        if (episode + 1) % 25 == 0:
            recent = pd.DataFrame(history[-25:])
            print(
                f"Episode {episode + 1:4d}/{n_episodes} | "
                f"eps={epsilon:.3f} | "
                f"reward(25)={recent['total_reward'].mean():8.1f} | "
                f"failures(25)={recent['failures'].mean():.2f} | "
                f"maint(25)={recent['preventive_maintenance'].mean():.1f}"
            )

    return agent, pd.DataFrame(history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--double",
        action="store_true",
        help="Use Double DQN targets (online selects, target evaluates).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EPISODES,
        help="Training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAINING_SEED,
        help="Training seed. Deep RL runs are seed-sensitive; comparing "
        "algorithm variants on a single seed each is unreliable.",
    )
    arguments = parser.parse_args()

    variant = "double_dqn" if arguments.double else "dqn"
    results_directory = (
        PROJECT_ROOT / "results" / "reinforcement_learning" / variant
    )

    config = DQNConfig(double_dqn=arguments.double)

    print(f"--- Phase 7/8: training {variant} ---")
    agent, history = train(
        config, n_episodes=arguments.episodes, seed=arguments.seed
    )

    results_directory.mkdir(parents=True, exist_ok=True)
    history.to_csv(results_directory / "training_history.csv", index=False)
    agent.save(results_directory / "network.pt")

    with (results_directory / "config.json").open("w") as file:
        json.dump(asdict(config), file, indent=2)

    print("\n--- Evaluating greedy policy on 20 unseen seeds ---")
    results = evaluate_policy(DQNPolicy(agent), variant)
    results.to_csv(results_directory / "evaluation.csv", index=False)

    print(summarize(results).to_string())
    print(f"\nSaved everything to:\n{results_directory}")


if __name__ == "__main__":
    main()
