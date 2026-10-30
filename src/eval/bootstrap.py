"""
src/eval/bootstrap.py
---------------------
Bootstrap confidence intervals for per-condition metrics.

Usage
-----
    from src.eval.bootstrap import bootstrap_ci, paired_diff_ci

    # CI for a single metric column
    mean, lo, hi = bootstrap_ci(df["lane_f1"].values, n=1000, ci=0.95, seed=42)

    # Paired difference between two method columns on the same frames
    mean_diff, lo, hi = paired_diff_ci(
        df["lane_f1_proposed"].values,
        df["lane_f1_baseline"].values,
        n=1000, ci=0.95, seed=42
    )
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n: int   = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : np.ndarray
        1-D array of per-frame metric values.
    n : int
        Number of bootstrap resamples.
    ci : float
        Confidence level, e.g. 0.95 for 95 %.
    seed : int

    Returns
    -------
    (mean, lower, upper) : tuple[float, float, float]
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means  = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n)
    ])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1.0 - alpha)))
    return float(values.mean()), lo, hi


def paired_diff_ci(
    method_a: np.ndarray,
    method_b: np.ndarray,
    n: int    = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the paired difference mean(A) − mean(B).

    Parameters
    ----------
    method_a, method_b : np.ndarray
        Per-frame metric values for method A and B respectively.
        Must be the same length (same frames, same order).

    Returns
    -------
    (mean_diff, lower, upper)
    """
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length: {len(a)} vs {len(b)}")

    diff = a - b
    mean_diff, lo, hi = bootstrap_ci(diff, n=n, ci=ci, seed=seed)
    return mean_diff, lo, hi


def aggregate_by_condition(
    df,
    metric_cols: list[str],
    condition_col: str = "condition",
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> "pd.DataFrame":
    """Aggregate metrics per condition with bootstrap CIs.

    Parameters
    ----------
    df : pd.DataFrame
        Per-frame results CSV (output of run_eval.py).
    metric_cols : list[str]
        Column names to aggregate.
    condition_col : str
    n, ci, seed : bootstrap parameters

    Returns
    -------
    pd.DataFrame with columns:
        condition, <metric>_mean, <metric>_lo, <metric>_hi for each metric.
    """
    import pandas as pd

    rows = []
    for cond, grp in df.groupby(condition_col):
        rec: dict = {"condition": cond, "n_frames": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                rec[f"{col}_mean"] = float("nan")
                rec[f"{col}_lo"]   = float("nan")
                rec[f"{col}_hi"]   = float("nan")
            else:
                mean, lo, hi = bootstrap_ci(vals, n=n, ci=ci, seed=seed)
                rec[f"{col}_mean"] = mean
                rec[f"{col}_lo"]   = lo
                rec[f"{col}_hi"]   = hi
        rows.append(rec)

    return pd.DataFrame(rows)
