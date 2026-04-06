"""Cross-signal and cross-city correlation analysis.

Helps determine if strategies are diversified or redundant,
and whether city-level signals are independent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cross_signal_correlation(
    signal_returns: dict[str, pd.Series],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix between strategy return streams.

    Args:
        signal_returns: {strategy_name: pd.Series of returns}
        method: "pearson" or "spearman".

    Returns:
        Correlation matrix as DataFrame.
    """
    if len(signal_returns) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(signal_returns)
    return df.corr(method=method)


def cross_city_correlation(
    city_spreads: dict[str, pd.Series],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation of model-vs-market spreads across cities.

    Args:
        city_spreads: {city: pd.Series of spread values}
        method: Correlation method.

    Returns:
        Correlation matrix.
    """
    if len(city_spreads) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(city_spreads)
    return df.corr(method=method)


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Compute rolling correlation between two series.

    Args:
        series_a: First series.
        series_b: Second series.
        window: Rolling window size.

    Returns:
        Rolling correlation series.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def diversification_ratio(
    strategy_returns: dict[str, pd.Series],
) -> float:
    """Compute portfolio diversification ratio.

    DR = sum(individual vols) / portfolio vol.
    DR > 1 means strategies provide diversification benefit.

    Args:
        strategy_returns: {strategy_name: pd.Series of returns}

    Returns:
        Diversification ratio. Higher is better.
    """
    if len(strategy_returns) < 2:
        return 1.0

    df = pd.DataFrame(strategy_returns).dropna()
    if len(df) < 10:
        return 1.0

    individual_vols = df.std()
    weights = np.ones(len(individual_vols)) / len(individual_vols)  # Equal weight
    cov = df.cov()

    weighted_sum_vol = float(np.dot(weights, individual_vols))
    port_var = float(np.dot(weights, np.dot(cov, weights)))
    port_vol = np.sqrt(max(port_var, 1e-10))

    return weighted_sum_vol / port_vol if port_vol > 1e-10 else 1.0


def cointegration_test(
    series_a: pd.Series,
    series_b: pd.Series,
) -> dict[str, float]:
    """Simplified cointegration test between two series.

    Uses Engle-Granger approach: regress A on B, test residuals for stationarity.

    Args:
        series_a: First price/probability series.
        series_b: Second price/probability series.

    Returns:
        Dict with 'beta', 'residual_adf_stat', 'mean_reversion_half_life'.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"beta": 0.0, "residual_adf_stat": 0.0, "mean_reversion_half_life": float("inf")}

    a = aligned.iloc[:, 0].values
    b = aligned.iloc[:, 1].values

    # OLS regression
    beta = float(np.cov(a, b)[0, 1] / max(np.var(b), 1e-10))
    residuals = a - beta * b

    # ADF-like test: autocorrelation of residuals
    if len(residuals) < 5:
        return {"beta": beta, "residual_adf_stat": 0.0, "mean_reversion_half_life": float("inf")}

    diffs = np.diff(residuals)
    if np.std(residuals[:-1]) < 1e-10:
        adf_stat = 0.0
    else:
        adf_stat = float(np.corrcoef(residuals[:-1], diffs)[0, 1])

    # Half-life estimate
    if adf_stat < -0.01:
        half_life = -np.log(2) / np.log(1 + adf_stat)
    else:
        half_life = float("inf")

    return {
        "beta": beta,
        "residual_adf_stat": adf_stat,
        "mean_reversion_half_life": max(half_life, 0),
    }
