"""Walk-forward analysis with regime-aware window reporting.

For each window: optimize on IS, select best params, run on OOS, collect
results. Concatenates OOS segments into the most realistic equity estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strategies.base import BaseStrategy
from backtesting.backtest_engine import BacktestEngine, BacktestResult
from backtesting.splitter import walk_forward_split, SplitWindow
from backtesting.data_loader import BacktestDataset
from backtesting.metrics import compute_metrics, MetricsReport
from backtesting.optimizer import random_search, grid_search

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindowResult:
    """Results for a single walk-forward window."""

    fold: int
    best_params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_return: float
    oos_n_trades: int
    oos_equity_segment: list[float]
    regime_distribution: dict[str, float]
    overfit_warning: bool  # IS Sharpe > 2x OOS Sharpe


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward analysis results."""

    windows: list[WalkForwardWindowResult]
    concatenated_equity: list[float]
    aggregate_sharpe: float
    aggregate_return: float
    aggregate_n_trades: int
    regime_stability: bool  # Whether regime distribution was stable


def run_walk_forward(
    strategy: BaseStrategy,
    dataset: BacktestDataset,
    train_days: int = 60,
    test_days: int = 15,
    param_grid: dict[str, list[Any]] | None = None,
    param_ranges: dict[str, tuple[float, float]] | None = None,
    n_random_trials: int = 30,
    objective: str = "sharpe",
    seed: int = 42,
    engine_kwargs: dict[str, Any] | None = None,
) -> WalkForwardResult:
    """Execute full walk-forward analysis.

    Args:
        strategy: Strategy to test.
        dataset: Full dataset (will be split into walk-forward windows).
        train_days: Days per training window.
        test_days: Days per test window.
        param_grid: Grid search params (mutually exclusive with param_ranges).
        param_ranges: Random search params.
        n_random_trials: Trials per window for random search.
        objective: Optimization objective.
        seed: Random seed.
        engine_kwargs: Additional BacktestEngine kwargs.

    Returns:
        WalkForwardResult with per-window and aggregate stats.
    """
    splits = walk_forward_split(dataset, train_days, test_days)
    ek = engine_kwargs or {}

    if not splits:
        logger.warning("No walk-forward windows generated")
        return WalkForwardResult([], [], 0.0, 0.0, 0, True)

    window_results: list[WalkForwardWindowResult] = []
    all_oos_equity: list[float] = []
    all_oos_trades = 0
    prev_regime_dist: dict[str, float] | None = None
    regime_stable = True

    for split in splits:
        logger.info("Walk-forward fold %d: train %s→%s, test %s→%s",
                     split.fold, split.train_start, split.train_end,
                     split.test_start, split.test_end)

        # Optimize on IS
        if param_grid:
            opt_result = grid_search(
                strategy, param_grid, split.train_data, objective=objective,
                top_n=1, engine_kwargs=ek,
            )
        elif param_ranges:
            opt_result = random_search(
                strategy, param_ranges, split.train_data, n_trials=n_random_trials,
                objective=objective, top_n=1, seed=seed + split.fold, engine_kwargs=ek,
            )
        else:
            # No optimization — use current params
            opt_result = None

        if opt_result:
            best_params = opt_result.best_params
            is_metric = opt_result.best_is_metric
            strategy.set_parameters(best_params)
        else:
            best_params = strategy.get_parameters()
            engine = BacktestEngine(strategy=strategy, **ek)
            is_result = engine.run(split.train_data)
            is_report = compute_metrics(is_result, label="IS")
            is_metric = is_report.sharpe_ratio

        # Run OOS
        engine = BacktestEngine(strategy=strategy, **ek)
        oos_result = engine.run(split.test_data)
        oos_report = compute_metrics(oos_result, label="OOS")

        # Regime distribution
        if oos_result.regime_history:
            regimes = oos_result.regime_history
            total = len(regimes)
            regime_dist = {}
            for r in set(regimes):
                regime_dist[r] = regimes.count(r) / total

            # Check stability
            if prev_regime_dist:
                max_shift = max(
                    abs(regime_dist.get(r, 0) - prev_regime_dist.get(r, 0))
                    for r in set(list(regime_dist.keys()) + list(prev_regime_dist.keys()))
                )
                if max_shift > 0.4:
                    regime_stable = False
                    logger.warning("Regime instability in fold %d: max shift %.2f", split.fold, max_shift)
            prev_regime_dist = regime_dist
        else:
            regime_dist = {}

        # Overfit check
        overfit = is_metric > 2 * oos_report.sharpe_ratio if oos_report.sharpe_ratio > 0 else is_metric > 1.0

        window_results.append(WalkForwardWindowResult(
            fold=split.fold,
            best_params=best_params,
            is_sharpe=is_metric,
            oos_sharpe=oos_report.sharpe_ratio,
            oos_return=oos_report.total_return,
            oos_n_trades=oos_report.n_trades,
            oos_equity_segment=oos_result.equity_curve,
            regime_distribution=regime_dist,
            overfit_warning=overfit,
        ))

        # Concatenate OOS equity
        if oos_result.equity_curve:
            if all_oos_equity:
                scale = all_oos_equity[-1] / oos_result.equity_curve[0] if oos_result.equity_curve[0] > 0 else 1.0
                scaled = [v * scale for v in oos_result.equity_curve[1:]]
                all_oos_equity.extend(scaled)
            else:
                all_oos_equity.extend(oos_result.equity_curve)

        all_oos_trades += oos_report.n_trades

    # Aggregate
    agg_return = (all_oos_equity[-1] / all_oos_equity[0] - 1.0) if all_oos_equity and all_oos_equity[0] > 0 else 0.0
    agg_sharpe = 0.0
    if len(all_oos_equity) > 2:
        ec = np.array(all_oos_equity)
        rets = np.diff(ec) / ec[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 0 and np.std(rets) > 1e-10:
            agg_sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(365 * 48))

    return WalkForwardResult(
        windows=window_results,
        concatenated_equity=all_oos_equity,
        aggregate_sharpe=agg_sharpe,
        aggregate_return=agg_return,
        aggregate_n_trades=all_oos_trades,
        regime_stability=regime_stable,
    )
