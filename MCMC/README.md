# MCMC Feature Selection

Standalone Metropolis-Hastings feature selection for time-series regression.
Input data and generated outputs are intentionally excluded from Git.

## Install

```bash
uv sync --frozen
```

The pip fallback is `python -m pip install -r requirements.txt`.

## Run

Keep corporate data outside the checkout and use an absolute path:

```bash
uv run python metropolis_hastings.py \
  --data /secure/data/masked_data.parquet \
  --iterations 5000 \
  --subset-size 6 \
  --temperature 0.001 \
  --cv-splits 4 \
  --graph-every 250 \
  --graph-dir /secure/outputs/mcmc/search_snapshots \
  --diagnostics /secure/outputs/mcmc/diagnostics.png \
  --output /secure/outputs/mcmc/selected_features.npy
```

Use `uv run python metropolis_hastings.py --help` for all parameters. An optional
`--candidate-indices /secure/path/surviving_features.npy` restricts the search to a
pre-screened pool while preserving original dataset indices in the final output.

Before pushing, inspect `git status --short` and `git diff --cached --name-only`.
