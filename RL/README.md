# Data-Driven Reinforcement Learning for ESP Maintenance Scheduling

> **This project is an educational simulation and is not intended for
> direct operation of industrial equipment.** All economic figures and
> degradation dynamics are stated simulation assumptions, not measured
> SOCAR or industry data.

## The problem

An Electrical Submersible Pump (ESP) in oil production earns revenue
every hour it runs — and wears out while doing so. The operator's hourly
choice is a genuine sequential decision problem:

* running at **full load** earns the most now but degrades the pump fastest
  (degradation grows with load *squared*);
* **reduced load** trades revenue for slower wear;
* **inspection** costs money and an hour of production, but returns an
  accurate estimate of the pump's hidden health;
* **preventive maintenance** costs money and 8 hours of downtime, but
  resets degradation;
* doing none of these risks an **unplanned failure**: a large penalty,
  an expensive corrective repair, and 24 hours of downtime.

Because today's action changes tomorrow's failure risk, single-shot
prediction is not enough — the problem is sequential, which is why
reinforcement learning applies.

## Two ML components — kept strictly separate

**Component A: condition classification (supervised).** The ESPset
dataset (`data/raw/features.csv`: 6032 vibration-feature rows from 11
physical ESPs, 5 condition labels) trains a Random Forest that answers
*"what condition does this vibration signature look like?"*. Evaluated
with a stratified split **and** leave-one-pump-out (the honest test:
generalising to an unseen physical pump). Class imbalance (79.6% Normal,
1.2% Misalignment) means macro-F1 and balanced accuracy are the metrics,
never plain accuracy.

**Component B: maintenance scheduling (reinforcement learning).**
ESPset **cannot** train the RL agent: it is fault-diagnosis data, not a
chronological run-to-failure record, and it contains no counterfactual
maintenance decisions ("what would have happened had we serviced
earlier?"). The RL agent therefore trains inside a simulator that
models degradation, failure risk, maintenance effects, downtime, and
economics. The real dataset's role is to ground the condition
classifier, which can later feed condition probabilities into the RL
observation.

## Architecture

```text
src/
├── simulation/          # physics only, no RL, no money
│   ├── pump_model.py    #   hidden health, sensors, downtime, failure
│   ├── degradation.py   #   health-loss + failure-risk formulas (configs)
│   └── economics.py     #   cost/revenue configuration (no timing logic)
├── environment/         # the decision process
│   ├── actions.py       #   OPERATE_FULL / OPERATE_REDUCED / INSPECT / MAINTAIN
│   ├── operations.py    #   "one cash register": action -> hour -> cash flow
│   ├── maintenance_env.py  # Gymnasium wrapper (Box obs, reward, truncation)
│   └── state_discretizer.py# continuous state -> hashable bins (tabular only)
├── baselines/           # hand-written policies to beat
├── agents/              # Q-learning, replay buffer, DQN / Double DQN
├── training/            # training loops + result saving
├── evaluation/          # one shared multi-seed evaluator for ALL policies
└── (top level)          # Component A: classifier scripts
tests/                   # pytest suite
```

Two structural rules do most of the work:

1. **One cash register.** Every policy — baseline or learned — is metered
   by the same `MaintenanceOperations` layer. The env's reward is
   literally that layer's hourly profit. No accounting drift between
   what baselines report and what the agent optimises.
2. **Partial observability is enforced by type.** Policies receive an
   `ObservableState` (noisy sensors + bookkeeping). True health exists
   only inside the simulator and in the env's `info` dict for debugging.

## Setup

```bash
cd RL
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `data/raw/features.csv` (semicolon-separated).

## Commands

```bash
pytest tests/ -v                                # full test suite

python -m src.train_condition_model             # Component A: train classifier
python -m src.evaluate_by_pump                  # Component A: leave-one-pump-out

python -m src.evaluation.evaluate_policy        # baselines, 20 seeded years
python -m src.training.train_q_learning         # Phase 6: tabular Q-learning
python -m src.training.train_dqn                # Phase 7: DQN
python -m src.training.train_dqn --double       # Phase 8: Double DQN
```

Results land under `results/` (baselines, condition model, RL runs each
in their own folder, with training history, config, and evaluation CSVs).

## Evaluation philosophy

Zero failures is not the goal — a policy that maintains constantly has
no failures and terrible economics. Every policy is scored on 20 unseen
seeded pump-years by mean profit, spread, failures, maintenance and
inspection counts, downtime, and availability. A learned policy is only
interesting if it beats the best hand-written baseline, not just the
worst one.

## Results (20 unseen seeded pump-years each)

| policy | mean profit | σ | failures | preventive | notes |
| --- | --- | --- | --- | --- | --- |
| **q_learning** | **$1,357,636** | 4,192 | 0.00 | 17.0 | best; discovered ~500 h service cycle |
| **dqn** | $1,349,336 | 4,705 | 0.00 | 12.0 | close second on far fewer episodes |
| fixed_interval | $1,271,979 | 12,335 | 0.15 | 4.9 | best hand-written baseline |
| sensor_threshold | $1,183,694 | 14,138 | 0.00 | 7.4 | |
| double_dqn | $1,132,180 | 13,129 | 4.05 | 0.0 | collapsed to run-to-failure (see below) |
| run_to_failure | $1,132,180 | 13,129 | 4.05 | 0.0 | |
| inspection_aware | $1,038,184 | 22,107 | 0.20 | 4.65 | 57 inspections/yr don't pay for themselves |

Both successful RL agents beat every baseline (+6.7% over the best one)
and cut profit variance to a third — and both independently learned that
*no policy should inspect*: vibration+age already time maintenance well,
so paying for extra information isn't worth it in this simulator.

**The Double DQN finding** (2/2 seeds, network probing in the training
history): the maintain-vs-operate value gap here is under 1% of the value
scale. Plain DQN's overestimation bias acts as useful optimism about
rarely-taken actions and tips those marginal decisions correctly; Double
DQN's bias correction removes exactly that nudge and lands MAINTAIN just
below OPERATE everywhere, collapsing to run-to-failure at this training
budget. Bias corrections are trade-offs, not free wins.

## Hard-won training lessons (kept because they generalise)

* **γ must cover the failure horizon.** γ=0.98 gives an effective
  ~50-hour horizon; degradation plays out over ~2000 hours. Agents with
  short horizons rationally ignore failures. We use γ=0.999.
* **Bootstrap through truncation.** Episodes end because we stop
  watching, not because the world ends. Treating truncation as terminal
  teaches the agent that consequences beyond the horizon are worth zero.
* **ε-greedy exploration can starve the state space.** A random action
  is maintenance 25% of the time, so the exploring pump never degrades
  and the agent never visits the states that matter. Fix: **exploring
  starts** — training episodes begin at a random degradation level,
  all the way down to the failure cliff.
* **Optimistic initialisation breaks at γ≈1.** Bootstrapped targets
  recycle the optimism (`r + γ·maxQ` stays inflated while everything is
  inflated), and the bubble deflates at rate (1−γ) per sweep.
  Initialise at the right value *scale* (`r̄/(1−γ)`) instead and let
  updates spend their budget on differences between actions.

## Known limitations / future work

* The condition ladder (NORMAL → UNBALANCE → RUBBING → MISALIGNMENT) is
  a simulation assumption driven by one hidden health scalar — real
  faults are independent modes. A later simulator version should model
  them independently, and treat `Faulty sensor` as an
  observation-quality problem rather than mechanical degradation.
* The condition classifier is not yet wired into the RL observation;
  the planned pipeline is: hidden simulated condition → ESPset-like
  synthetic feature vector → classifier probabilities → observation.
* Economic parameters are placeholders for experimentation.
