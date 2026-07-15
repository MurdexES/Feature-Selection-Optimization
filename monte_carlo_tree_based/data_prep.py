"""Local data-preparation helpers; kept here so this repository is standalone."""

from __future__ import annotations

import numpy as np


def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def prepare_data(
    X,
    y,
    feature_names=None,
    impute="median",
    winsorize_features=None,
    winsorize_target=None,
    target_transform=None,
    verbose=True,
):
    """Clean features/target and optionally first-difference the target."""
    X = np.asarray(X, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    report = []

    n_inf = int(np.isinf(X).sum())
    if n_inf:
        X[np.isinf(X)] = np.nan
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        if impute == "median":
            medians = np.nanmedian(X, axis=0)
            rows, columns = np.where(np.isnan(X))
            X[rows, columns] = medians[columns]
        else:
            X = np.nan_to_num(X, nan=0.0)
    report.append(f"features: {n_inf} inf + {n_nan} NaN cells cleaned ({impute})")

    if winsorize_features is not None:
        low, high = winsorize_features
        lower = np.quantile(X, low, axis=0)
        upper = np.quantile(X, high, axis=0)
        clipped = int(((X < lower) | (X > upper)).sum())
        X = np.clip(X, lower, upper)
        report.append(
            f"features: winsorized to [{low:.1%}, {high:.1%}] "
            f"({clipped} cells clipped)"
        )

    finite_target = np.isfinite(y)
    if not finite_target.all():
        X, y = X[finite_target], y[finite_target]
        report.append(f"dropped {int((~finite_target).sum())} rows with non-finite target")

    if winsorize_target is not None:
        low, high = winsorize_target
        lower, upper = np.quantile(y, low), np.quantile(y, high)
        clipped = int(((y < lower) | (y > upper)).sum())
        y = np.clip(y, lower, upper)
        report.append(
            f"target: winsorized to [{low:.1%}, {high:.1%}] "
            f"({clipped} values clipped)"
        )

    if target_transform == "difference":
        y = np.diff(y)
        X = X[1:]
        report.append("target: first-differenced (predicts the change)")
    elif target_transform is not None:
        raise ValueError(f"unknown target_transform: {target_transform!r}")

    if verbose:
        print("Data preparation:")
        for item in report:
            print("  -", item)
        print(f"  -> X {X.shape}, y {y.shape}")

    if feature_names is not None:
        feature_names = np.asarray(feature_names)
    return X, y, feature_names
