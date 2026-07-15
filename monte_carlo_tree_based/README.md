# MCTS Feature Selection

Standalone Monte Carlo Tree Search feature selection for time-series regression.
No dataset, generated feature list, model output, checkpoint, or snapshot is intended
to be committed to this repository.

## Requirements

- Python 3.10–3.13
- [`uv`](https://docs.astral.sh/uv/) on the deployment server
- A Parquet dataset containing `Timestamp`, `target`, and candidate feature columns

## Install

```bash
uv sync --frozen
```

If `uv.lock` is intentionally being refreshed, run `uv sync` instead. A pip fallback
is available with `python -m pip install -r requirements.txt`.

## Data safety

Keep corporate data outside the Git checkout and pass its absolute path with
`--data`. The repository `.gitignore` also blocks Parquet, CSV, NumPy, pickle,
checkpoint, output, and local environment files.

Before pushing, verify staged files with:

```bash
git status --short
git diff --cached --name-only
```

## Run

```bash
uv run python main.py \
  --data /secure/data/masked_data.parquet \
  --iterations 5000 \
  --subset-size 6 \
  --cv-splits 4 \
  --snapshot-every 250 \
  --snapshot-dir /secure/outputs/mcts/tree_snapshots \
  --output /secure/outputs/mcts/mcts_selected_features.npy
```

Use `uv run python main.py --help` for all tuning parameters. For long corporate
server runs, launch through the platform's scheduler/service and direct outputs to
persistent storage outside the Git checkout.
