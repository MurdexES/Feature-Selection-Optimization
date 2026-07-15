import numpy as np


def prepare_data(X, y, feature_names=None,
                 impute="median",
                 winsorize_features=None,
                 winsorize_target=None,
                 target_transform=None,
                 verbose=True):
    X = np.asarray(X, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    report = []

    n_inf = int(np.isinf(X).sum())
    if n_inf:
        X[np.isinf(X)] = np.nan
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        if impute == "median":
            med = np.nanmedian(X, axis=0)
            rows, cols = np.where(np.isnan(X))
            X[rows, cols] = med[cols]
        else:
            X = np.nan_to_num(X, nan=0.0)
    report.append(f"features: {n_inf} inf + {n_nan} NaN cells cleaned ({impute})")

    if winsorize_features is not None:
        lo, hi = winsorize_features
        qlo = np.quantile(X, lo, axis=0)
        qhi = np.quantile(X, hi, axis=0)
        n_clip = int(((X < qlo) | (X > qhi)).sum())
        X = np.clip(X, qlo, qhi)
        report.append(f"features: winsorized to [{lo:.1%}, {hi:.1%}] ({n_clip} cells clipped)")

    good = np.isfinite(y)
    if not good.all():
        X, y = X[good], y[good]
        report.append(f"dropped {int((~good).sum())} rows with non-finite target")

    if winsorize_target is not None:
        lo, hi = winsorize_target
        tlo, thi = np.quantile(y, lo), np.quantile(y, hi)
        n_clip = int(((y < tlo) | (y > thi)).sum())
        y = np.clip(y, tlo, thi)
        report.append(f"target: winsorized to [{lo:.1%}, {hi:.1%}] ({n_clip} values clipped)")

    if target_transform == "difference":
        y = np.diff(y)
        X = X[1:]
        report.append("target: first-differenced (stationary; predicts the change)")
    elif target_transform is not None:
        raise ValueError(f"unknown target_transform: {target_transform!r}")

    if verbose:
        print("Stage 0 - data preparation:")
        for r in report:
            print("  -", r)
        print(f"  -> X {X.shape}, y {y.shape}")

    if feature_names is not None:
        feature_names = np.asarray(feature_names)
    return X, y, feature_names
