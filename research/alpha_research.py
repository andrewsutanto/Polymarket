"""Alpha research utilities for signal quality analysis.

Provides functions for evaluating signal predictive power without
modifying core code. Works with any pandas Series.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """Compute Information Coefficient (rank correlation).

    Args:
        signal: Signal values aligned by index.
        forward_returns: Forward return values aligned by index.
        method: "spearman" or "pearson".

    Returns:
        IC value between -1 and 1.
    """
    aligned = pd.concat([signal, forward_returns], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method))


def signal_decay(
    signal: pd.Series,
    forward_returns: pd.Series,
    lags: list[int] | None = None,
) -> dict[int, float]:
    """Compute IC at multiple forward lags to measure signal persistence.

    Args:
        signal: Signal values.
        forward_returns: Base forward returns (1-period).
        lags: List of lag periods to test. Defaults to [1, 2, 5, 10, 20].

    Returns:
        {lag: IC_at_lag}
    """
    lags = lags or [1, 2, 5, 10, 20]
    results: dict[int, float] = {}

    for lag in lags:
        shifted_returns = forward_returns.shift(-lag)
        ic = compute_ic(signal, shifted_returns)
        results[lag] = ic

    return results


def turnover(signal: pd.Series) -> float:
    """Compute signal turnover (fraction of periods where signal flips direction).

    Args:
        signal: Signal values.

    Returns:
        Turnover rate between 0 and 1.
    """
    s = signal.dropna()
    if len(s) < 2:
        return 0.0

    signs = np.sign(s.values)
    flips = np.sum(signs[1:] != signs[:-1])
    return float(flips / (len(signs) - 1))


def seasonal_stability(
    signal: pd.Series,
    forward_returns: pd.Series,
    freq: str = "M",
) -> pd.DataFrame:
    """Check if signal works across seasons/time periods.

    Args:
        signal: Signal values with datetime index.
        forward_returns: Forward returns with datetime index.
        freq: Grouping frequency ("M" for monthly, "Q" for quarterly).

    Returns:
        DataFrame with IC per period.
    """
    aligned = pd.concat(
        [signal.rename("signal"), forward_returns.rename("returns")],
        axis=1, join="inner",
    ).dropna()

    if len(aligned) < 20:
        return pd.DataFrame()

    grouped = aligned.groupby(pd.Grouper(freq=freq))
    results = []
    for period, group in grouped:
        if len(group) < 5:
            continue
        ic = float(group["signal"].corr(group["returns"], method="spearman"))
        results.append({"period": period, "ic": ic, "n_obs": len(group)})

    return pd.DataFrame(results)


def signal_quantile_returns(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Bin signal into quantiles and compute average return per bin.

    Args:
        signal: Signal values.
        forward_returns: Forward returns.
        n_quantiles: Number of bins.

    Returns:
        DataFrame with quantile, mean_return, n_obs.
    """
    aligned = pd.concat(
        [signal.rename("signal"), forward_returns.rename("returns")],
        axis=1, join="inner",
    ).dropna()

    if len(aligned) < n_quantiles * 3:
        return pd.DataFrame()

    aligned["quantile"] = pd.qcut(aligned["signal"], n_quantiles, labels=False, duplicates="drop")
    summary = aligned.groupby("quantile")["returns"].agg(["mean", "count"]).reset_index()
    summary.columns = ["quantile", "mean_return", "n_obs"]
    return summary
