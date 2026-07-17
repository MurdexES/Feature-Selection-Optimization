"""Train the tabular Q-learning agent and evaluate it against baselines.

    cd RL && python -m src.training.train_q_learning

Training loop shape: for each episode, reset the env (fresh random pump),
then for each hour choose epsilon-greedily, step, and update the Q-table
from the observed transition. The discretizer reads the *named*
ObservableState (passed through info) rather than the normalised Box
vector -- bins on named physical quantities are meaningful and auditable;
the Box vector exists for the neural-network agents.

After training, the greedy policy is wrapped in QTablePolicy and scored
by the exact same evaluator as the baselines: 20 unseen seeds, 8760 hours
each, exploration off.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.agents.q_learning import (
    QLearningAgent,
    QLearningConfig,
    QTablePolicy,
)
from src.environment.maintenance_env import PumpMaintenanceEnv
from src.evaluation.evaluate_policy import evaluate_policy, summarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIRECTORY = (
    PROJECT_ROOT / "results" / "reinforcement_learning" / "q_learning"
)

N_EPISODES = 1500
TRAINING_SEED = 0


def train(
    n_episodes: int = N_EPISODES,
    seed: int = TRAINING_SEED,
) -> tuple[QLearningAgent, pd.DataFrame]:
    # exploring_starts: some episodes begin with an already-degraded pump,
    # otherwise random maintenance during exploration keeps the pump young
    # forever and degraded states are never learned about (we watched this
    # exact failure happen: 3 of 320 states visited in 800 episodes).
    env = PumpMaintenanceEnv(exploring_starts=True)
    agent = QLearningAgent(config=QLearningConfig(), seed=seed)

    history = []

    for episode in range(n_episodes):
        epsilon = agent.epsilon_for_episode(episode)

        _, info = env.reset(seed=seed + episode)
        state = agent.discretizer.discretize(info["observable_state"])

        episode_reward = 0.0
        failures = 0
        preventive = 0
        inspections = 0

        while True:
            action = agent.choose_action(state, epsilon=epsilon)
            _, reward, terminated, truncated, info = env.step(int(action))

            next_state = agent.discretizer.discretize(
                info["observable_state"]
            )

            # terminated (not truncated!) controls bootstrapping --
            # see QLearningAgent.update.
            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward

            outcome = info["outcome"]
            failures += outcome.failed_this_hour
            preventive += outcome.preventive_started
            inspections += outcome.inspected

            if terminated or truncated:
                break

        history.append(
            {
                "episode": episode,
                "epsilon": epsilon,
                "total_reward": episode_reward,
                "failures": failures,
                "preventive_maintenance": preventive,
                "inspections": inspections,
                "q_table_states": len(agent.q_table),
            }
        )

        if (episode + 1) % 50 == 0:
            recent = pd.DataFrame(history[-50:])
            print(
                f"Episode {episode + 1:4d}/{n_episodes} | "
                f"eps={epsilon:.3f} | "
                f"reward(50)={recent['total_reward'].mean():8.1f} | "
                f"failures(50)={recent['failures'].mean():.2f} | "
                f"maint(50)={recent['preventive_maintenance'].mean():.1f} | "
                f"states={len(agent.q_table)}"
            )

    return agent, pd.DataFrame(history)


def main() -> None:
    print("--- Phase 6: training tabular Q-learning ---")
    agent, history = train()

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    history.to_csv(RESULTS_DIRECTORY / "training_history.csv", index=False)
    agent.save(RESULTS_DIRECTORY / "q_table.pkl")

    with (RESULTS_DIRECTORY / "config.json").open("w") as file:
        json.dump(asdict(agent.config), file, indent=2)

    print("\n--- Evaluating greedy policy on 20 unseen seeds ---")
    results = evaluate_policy(QTablePolicy(agent), "q_learning")
    results.to_csv(RESULTS_DIRECTORY / "evaluation.csv", index=False)

    print(summarize(results).to_string())
    print(f"\nSaved everything to:\n{RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
