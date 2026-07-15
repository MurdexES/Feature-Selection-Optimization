# Genetic Feature Selection

Standalone genetic hyperparameter optimization and feature selection. Dataset,
candidate lists, checkpoints, and generated results are excluded from Git.

## Install

```bash
uv sync --frozen
```

The pip fallback is `python -m pip install -r requirements.txt`.

Optional pre-screening can be run first:

```bash
uv run python feature_elimination.py \
  --data /secure/data/masked_data.parquet \
  --iterations 20000 \
  --min-features 60 \
  --output /secure/outputs/ga/surviving_features.npy
```

## Run

```bash
uv run python genetic.py \
  --data /secure/data/masked_data.parquet \
  --candidate-indices /secure/data/surviving_features.npy \
  --population-size 500 \
  --generations 250 \
  --min-features 5 \
  --max-features 10 \
  --checkpoint-every 1 \
  --checkpoint-dir /secure/outputs/ga/checkpoints \
  --output /secure/outputs/ga/selected_features.npy \
  --result-json /secure/outputs/ga/result.json
```

Resume a trusted checkpoint with the same candidate pool and configuration by adding
`--resume`. Checkpoints use Python pickle and must never be loaded from an untrusted
source.

Keep corporate data outside the checkout. Before pushing, inspect
`git status --short` and `git diff --cached --name-only`.
