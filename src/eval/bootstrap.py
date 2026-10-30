"""
src/eval/bootstrap.py
---------------------
Bootstrap confidence intervals for per-condition metrics (1,000 resamples, paired option).

Usage
-----
    from src.eval.bootstrap import bootstrap_ci, paired_diff_ci

    # 1. CI for a single metric column (1,000 resamples, 95% CI)
    mean, lo, hi = bootstrap_ci(df["lane_f1"].values, n=1000, ci=0.95, seed=42)

    # 2. Paired difference CI between two methods on the same frames
    mean_diff, lo, hi = bootstrap_ci(
        df["lane_f1_proposed"].values,
        paired_with=df["lane_f1_baseline"].values,
        n=1000, ci=0.95, seed=42
    )

    # Or via paired_diff_ci directly
    mean_diff, lo, hi = paired_diff_ci(
        df["lane_f1_proposed"].values,
        df["lane_f1_baseline"].values,
        n=1000, ci=0.95, seed=42
    )
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray | Sequence[float],
    paired_with: Optional[np.ndarray | Sequence[float]] = None,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean, with an optional paired difference.

    Parameters
    ----------
    values : np.ndarray | Sequence[float]
        1-D array of per-frame metric values (or method A for paired test).
    paired_with : np.ndarray | Sequence[float] | None
        Optional 1-D array of method B per-frame metric values for paired difference.
        Must be the same length and align frame-by-frame with `values`.
    n : int
        Number of bootstrap resamples (default: 1,000).
    ci : float
        Confidence level, e.g. 0.95 for 95%.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (mean, lower, upper) : tuple[float, float, float]
        Mean (or paired difference mean) and the bootstrap lower and upper percentiles.
    """
    values = np.asarray(values, dtype=np.float64)

    if paired_with is not None:
        paired_with = np.asarray(paired_with, dtype=np.float64)
        if len(values) != len(paired_with):
            raise ValueError(
                f"Array lengths differ for paired CI: {len(values)} vs {len(paired_with)}"
            )
        values = values - paired_with

    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    # Resample with replacement n times
    indices = rng.integers(0, len(values), size=(n, len(values)))
    resamples = values[indices]
    means = resamples.mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1.0 - alpha)))
    return float(values.mean()), lo, hi


def paired_diff_ci(
    method_a: np.ndarray | Sequence[float],
    method_b: np.ndarray | Sequence[float],
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the paired difference mean(A) − mean(B) with 1,000 resamples.

    Parameters
    ----------
    method_a, method_b : np.ndarray | Sequence[float]
        Per-frame metric values for method A and B respectively.
        Must have same length and correspond to identical frames.
    n : int
        Bootstrap resamples (default: 1,000).
    ci : float
        Confidence level (default: 0.95).
    seed : int
        Random seed.

    Returns
    -------
    (mean_diff, lower, upper) : tuple[float, float, float]
    """
    return bootstrap_ci(method_a, paired_with=method_b, n=n, ci=ci, seed=seed)


def aggregate_by_condition(
    df: pd.DataFrame,
    metric_cols: list[str],
    condition_col: str = "condition",
    n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate metrics per condition with bootstrap CIs (1,000 resamples).

    Parameters
    ----------
    df : pd.DataFrame
        Per-frame results DataFrame (output of evaluation).
    metric_cols : list[str]
        Column names to aggregate (e.g. ['road_iou', 'lane_f1', 'miou']).
    condition_col : str
        Name of condition column.
    n, ci, seed : bootstrap parameters

    Returns
    -------
    pd.DataFrame with columns:
        condition, n_frames, and for each metric: <metric>_mean, <metric>_lo, <metric>_hi
    """
    rows = []
    for cond, grp in df.groupby(condition_col):
        rec: dict = {"condition": cond, "n_frames": len(grp)}
        for col in metric_cols:
            if col not in grp.columns:
                continue
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
