# Feature Selection Optimization Suite

A monorepo containing three stochastic feature-selection implementations for
time-series regression:

| Algorithm | Directory | Entrypoint |
|---|---|---|
| Monte Carlo Tree Search | `monte_carlo_tree_based/` | `main.py` |
| Metropolis-Hastings MCMC | `MCMC/` | `metropolis_hastings.py` |
| Genetic Algorithm | `GA_&_feature_elimination/` | `genetic.py` |

Each algorithm is independently reproducible: it has its own `pyproject.toml`,
`uv.lock`, `requirements.txt`, ignored `.venv`, and detailed README. This avoids
forcing all three algorithms to share one environment while keeping their source
and deployment documentation in the same GitHub repository.

## Repository name

Recommended GitHub name: **`feature-selection-optimization-suite`**.

Other suitable names:

- `stochastic-feature-selection`
- `feature-selection-search-suite`
- `time-series-feature-selection`

## Installation

Install only the algorithm needed on a given server worker:

```bash
cd monte_carlo_tree_based
uv sync --frozen
```

Use `MCMC` or `GA_&_feature_elimination` instead for the other environments.

## Example runs

```bash
# MCTS
cd monte_carlo_tree_based
uv run python main.py --data /secure/data/masked_data.parquet --iterations 5000

# MCMC
cd ../MCMC
uv run python metropolis_hastings.py \
  --data /secure/data/masked_data.parquet --iterations 5000

# Genetic Algorithm
cd '../GA_&_feature_elimination'
uv run python genetic.py \
  --data /secure/data/masked_data.parquet \
  --population-size 500 --generations 250
```

See each algorithm README for output paths, snapshots, candidate screening, and
checkpoint/resume options.

## Data safety

Corporate data must remain outside the checkout and be passed through `--data`.
The root and algorithm-specific `.gitignore` files exclude:

- Parquet and CSV data
- NumPy selections
- Pickle checkpoints
- Generated plots and output directories
- Local environments, caches, secrets, and `.env` files

Before every push, inspect exactly what will be uploaded:

```bash
git status --short
git diff --cached --name-only
```

Never use `git add -f` for ignored data or output files.

## Create the GitHub repository

After reviewing the initial commit, create a **private** corporate repository named
`feature-selection-optimization-suite`, add it as the remote, and push `main`:

```bash
git add .
git status --short
git commit -m "Initial feature-selection optimization suite"
git remote add origin <corporate-github-repository-url>
git push -u origin main
```
