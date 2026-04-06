"""Performance and risk analytics for backtest results.

Computes 30+ metrics including returns, risk, risk-adjusted, trade-level,
strategy attribution, regime-conditional, weather-specific, and overfitting
diagnostics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestResult, BacktestTrade

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """Comprehensive metrics report."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    daily_mean: float = 0.0
    daily_median: float = 0.0
    daily_std: float = 0.0
    daily_skew: float = 0.0
    daily_kurtosis: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Trade-level
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_holding_bars: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0

    # Strategy attribution
    strategy_pnl: dict[str, float] = field(default_factory=dict)
    strategy_trades: dict[str, int] = field(default_factory=dict)
    strategy_win_rates: dict[str, float] = field(default_factory=dict)

    # Regime-conditional
    regime_sharpe: dict[str, float] = field(default_factory=dict)
    regime_pnl: dict[str, float] = field(default_factory=dict)
    regime_time_pct: dict[str, float] = field(default_factory=dict)

    # Weather-specific
    city_pnl: dict[str, float] = field(default_factory=dict)
    city_win_rates: dict[str, float] = field(default_factory=dict)

    # Overfitting
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    sharpe_delta: float = 0.0
    deflated_sharpe: float = 0.0
    overfit_flag: bool = False

    # Metadata
    label: str = ""


def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.05,
    periods_per_year: float = 365 * 48,
    label: str = "",
) -> MetricsReport:
    """Compute all metrics from a backtest result.

    Args:
        result: BacktestResult from the engine.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        periods_per_year: Evaluation periods per year (for annualization).
        label: Label for this report (e.g., "IS" or "OOS").

    Returns:
        MetricsReport with all computed metrics.
    """
    report = MetricsReport(label=label)
    ec = np.array(result.equity_curve)

    if len(ec) < 2:
        return report

    # Returns
    returns = np.diff(ec) / ec[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return report

    report.total_return = (ec[-1] - ec[0]) / ec[0] if ec[0] > 0 else 0.0
    report.daily_mean = float(np.mean(returns))
    report.daily_median = float(np.median(returns))
    report.daily_std = float(np.std(returns))
    report.daily_skew = float(_safe_skew(returns))
    report.daily_kurtosis = float(_safe_kurtosis(returns))

    n_periods = len(ec) - 1
    years = n_periods / periods_per_year if periods_per_year > 0 else 1.0
    if years > 0 and ec[0] > 0:
        report.annualized_return = report.total_return / max(years, 0.01)
        report.cagr = (ec[-1] / ec[0]) ** (1.0 / max(years, 0.01)) - 1.0

    # Risk
    dd, dd_dur = _max_drawdown(ec)
    report.max_drawdown = dd
    report.max_drawdown_duration_bars = dd_dur
    report.annualized_volatility = float(np.std(returns) * np.sqrt(min(periods_per_year, 100000)))
    report.downside_deviation = float(_downside_dev(returns, risk_free_rate / periods_per_year))
    report.var_95 = float(np.percentile(returns, 5)) if len(returns) > 20 else 0.0
    report.var_99 = float(np.percentile(returns, 1)) if len(returns) > 100 else 0.0
    report.cvar_95 = float(np.mean(returns[returns <= np.percentile(returns, 5)])) if len(returns) > 20 else 0.0

    # Risk-adjusted
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    report.sharpe_ratio = _sharpe(excess, periods_per_year)
    report.sortino_ratio = _sortino(excess, periods_per_year)
    report.calmar_ratio = report.annualized_return / max(abs(report.max_drawdown), 1e-8)
    report.omega_ratio = _omega(returns, rf_per_period)

    # Trade-level
    trades = result.trades
    report.n_trades = len(trades)
    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        report.win_rate = len(wins) / len(pnls) if pnls else 0.0
        report.avg_win = float(np.mean(wins)) if wins else 0.0
        report.avg_loss = float(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        report.profit_factor = gross_profit / max(gross_loss, 1e-8)
        report.avg_holding_bars = float(np.mean([t.holding_bars for t in trades]))
        report.max_consecutive_wins, report.max_consecutive_losses = _consecutive(pnls)
        report.expectancy = report.avg_win * report.win_rate + report.avg_loss * (1 - report.win_rate)

        # Strategy attribution
        strat_trades: dict[str, list[float]] = {}
        for t in trades:
            strat_trades.setdefault(t.strategy_name, []).append(t.pnl)
        for s, pnl_list in strat_trades.items():
            report.strategy_pnl[s] = sum(pnl_list)
            report.strategy_trades[s] = len(pnl_list)
            report.strategy_win_rates[s] = sum(1 for p in pnl_list if p > 0) / len(pnl_list)

        # City attribution
        city_trades: dict[str, list[float]] = {}
        for t in trades:
            city_trades.setdefault(t.city, []).append(t.pnl)
        for c, pnl_list in city_trades.items():
            report.city_pnl[c] = sum(pnl_list)
            report.city_win_rates[c] = sum(1 for p in pnl_list if p > 0) / len(pnl_list)

    # Regime-conditional
    if result.regime_history:
        regimes = result.regime_history
        unique_regimes = set(regimes)
        total_bars = len(regimes)
        for r in unique_regimes:
            r_mask = [i for i, reg in enumerate(regimes) if reg == r]
            report.regime_time_pct[r] = len(r_mask) / total_bars
            r_trades = [t for t in trades if t.regime_at_entry == r]
            report.regime_pnl[r] = sum(t.pnl for t in r_trades)
            if len(r_mask) > 10:
                r_returns = returns[np.array(r_mask[:-1]) if r_mask[-1] >= len(returns) else np.array(r_mask)]
                r_returns = r_returns[r_returns < len(returns)]
                if len(r_returns) > 2:
                    report.regime_sharpe[r] = _sharpe(r_returns - rf_per_period, periods_per_year)

    return report


def compute_overfit_diagnostics(
    is_report: MetricsReport,
    oos_report: MetricsReport,
    n_combos_tested: int = 1,
) -> MetricsReport:
    """Add overfitting diagnostics to the OOS report.

    Args:
        is_report: In-sample metrics.
        oos_report: Out-of-sample metrics (will be modified in-place).
        n_combos_tested: Number of parameter combinations tested.

    Returns:
        Updated OOS report with overfitting fields.
    """
    oos_report.is_sharpe = is_report.sharpe_ratio
    oos_report.oos_sharpe = oos_report.sharpe_ratio
    oos_report.sharpe_delta = is_report.sharpe_ratio - oos_report.sharpe_ratio

    # Deflated Sharpe Ratio
    if n_combos_tested > 1 and is_report.n_trades > 0:
        oos_report.deflated_sharpe = _deflated_sharpe(
            is_report.sharpe_ratio,
            n_combos_tested,
            is_report.n_trades,
            is_report.daily_skew,
            is_report.daily_kurtosis,
        )

    # Overfit flag
    if is_report.sharpe_ratio > 0:
        pct_degradation = oos_report.sharpe_delta / is_report.sharpe_ratio
        oos_report.overfit_flag = pct_degradation > 0.30
    else:
        oos_report.overfit_flag = oos_report.sharpe_ratio < is_report.sharpe_ratio

    return oos_report


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # Duration
    in_dd = dd > 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dd, max_dur


def _downside_dev(returns: np.ndarray, threshold: float = 0.0) -> float:
    below = returns[returns < threshold] - threshold
    if len(below) == 0:
        return 0.0
    return float(np.sqrt(np.mean(below ** 2)))


def _sharpe(excess_returns: np.ndarray, periods_per_year: float) -> float:
    if len(excess_returns) < 2:
        return 0.0
    std = np.std(excess_returns)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(min(periods_per_year, 100000)))


def _sortino(excess_returns: np.ndarray, periods_per_year: float) -> float:
    if len(excess_returns) < 2:
        return 0.0
    dd = _downside_dev(excess_returns, 0.0)
    if dd < 1e-10:
        return 0.0
    return float(np.mean(excess_returns) / dd * np.sqrt(min(periods_per_year, 100000)))


def _omega(returns: np.ndarray, threshold: float = 0.0) -> float:
    gains = np.sum(np.maximum(returns - threshold, 0))
    losses = np.sum(np.maximum(threshold - returns, 0))
    if losses < 1e-10:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def _consecutive(pnls: list[float]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m3 = np.mean((arr - np.mean(arr)) ** 3)
    s3 = np.std(arr) ** 3
    return m3 / s3 if s3 > 1e-10 else 0.0


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    m4 = np.mean((arr - np.mean(arr)) ** 4)
    s4 = np.std(arr) ** 4
    return (m4 / s4 - 3.0) if s4 > 1e-10 else 0.0


def _deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado)."""
    if n_trials <= 1 or n_obs < 2:
        return sharpe
    e_max_sharpe = _expected_max_sharpe(n_trials, n_obs)
    sr_std = np.sqrt((1.0 + 0.5 * sharpe ** 2 - skew * sharpe + (kurtosis / 4.0) * sharpe ** 2) / max(n_obs - 1, 1))
    if sr_std < 1e-10:
        return sharpe
    from scipy.stats import norm
    return float(norm.cdf((sharpe - e_max_sharpe) / sr_std))


def _expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """E[max SR] approximation."""
    from scipy.stats import norm
    gamma = 0.5772  # Euler-Mascheroni
    z = norm.ppf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0
    return z * (1.0 - gamma) + gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
