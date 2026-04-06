"""Parameter grid/random search optimizer (in-sample only).

Runs optimization exclusively on training data, then validates the
best parameter sets on held-out test data with degradation analysis.
"""

from __future__ import annotations

import csv
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from strategies.base import BaseStrategy
from backtesting.backtest_engine import BacktestEngine, BacktestResult
from backtesting.data_loader import BacktestDataset
from backtesting.metrics import compute_metrics, compute_overfit_diagnostics, MetricsReport

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


@dataclass
class OptimizationResult:
    """Result from parameter search."""

    best_params: dict[str, Any]
    best_is_metric: float
    best_oos_metric: float | None
    all_results: list[dict[str, Any]]
    top_n_comparison: list[dict[str, Any]]
    objective: str
    n_combos: int


def grid_search(
    strategy: BaseStrategy,
    param_grid: dict[str, list[Any]],
    train_data: BacktestDataset,
    test_data: BacktestDataset | None = None,
    objective: str = "sharpe",
    top_n: int = 5,
    engine_kwargs: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Exhaustive grid search over parameter combinations.

    Args:
        strategy: Strategy to optimize.
        param_grid: {param_name: [values]} to search over.
        train_data: In-sample data only.
        test_data: Out-of-sample data (optional, for validation).
        objective: Metric to maximize: "sharpe", "sortino", "calmar", "total_return".
        top_n: Number of best parameter sets to validate on OOS.
        engine_kwargs: Additional kwargs for BacktestEngine.

    Returns:
        OptimizationResult with all tested combos and degradation analysis.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    logger.info("Grid search: %d combinations over %s", len(combos), keys)
    return _run_search(strategy, keys, combos, train_data, test_data, objective, top_n, engine_kwargs)


def random_search(
    strategy: BaseStrategy,
    param_ranges: dict[str, tuple[float, float]],
    train_data: BacktestDataset,
    test_data: BacktestDataset | None = None,
    n_trials: int = 50,
    objective: str = "sharpe",
    top_n: int = 5,
    seed: int = 42,
    engine_kwargs: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Random search over continuous parameter ranges.

    Args:
        strategy: Strategy to optimize.
        param_ranges: {param_name: (min, max)} to sample from.
        train_data: In-sample data.
        test_data: Out-of-sample data (optional).
        n_trials: Number of random samples.
        objective: Metric to maximize.
        top_n: Number of best to validate on OOS.
        seed: Random seed.
        engine_kwargs: Additional kwargs for BacktestEngine.

    Returns:
        OptimizationResult.
    """
    rng = np.random.default_rng(seed)
    keys = list(param_ranges.keys())
    combos = []
    for _ in range(n_trials):
        combo = []
        for k in keys:
            lo, hi = param_ranges[k]
            combo.append(float(rng.uniform(lo, hi)))
        combos.append(tuple(combo))

    logger.info("Random search: %d trials over %s", n_trials, keys)
    return _run_search(strategy, keys, combos, train_data, test_data, objective, top_n, engine_kwargs)


def _run_search(
    strategy: BaseStrategy,
    keys: list[str],
    combos: list[tuple],
    train_data: BacktestDataset,
    test_data: BacktestDataset | None,
    objective: str,
    top_n: int,
    engine_kwargs: dict[str, Any] | None,
) -> OptimizationResult:
    """Core search loop shared by grid and random search."""
    ek = engine_kwargs or {}
    all_results: list[dict[str, Any]] = []
    original_params = strategy.get_parameters()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        strategy.set_parameters(params)

        engine = BacktestEngine(strategy=strategy, **ek)
        result = engine.run(train_data)
        report = compute_metrics(result, label="IS")
        metric_val = _get_metric(report, objective)

        entry = {
            "combo_idx": i,
            "params": dict(params),
            "is_metric": metric_val,
            "is_sharpe": report.sharpe_ratio,
            "is_total_return": report.total_return,
            "is_n_trades": report.n_trades,
            "is_win_rate": report.win_rate,
            "is_max_dd": report.max_drawdown,
        }
        all_results.append(entry)

        if (i + 1) % 10 == 0:
            logger.info("Search progress: %d/%d combos tested", i + 1, len(combos))

    # Sort by IS metric
    all_results.sort(key=lambda x: x["is_metric"], reverse=True)

    # Log to CSV
    _save_optimization_log(all_results)

    # Validate top N on OOS
    top_comparison: list[dict[str, Any]] = []
    if test_data and test_data.n_bars > 0:
        for entry in all_results[:top_n]:
            strategy.set_parameters(entry["params"])
            engine = BacktestEngine(strategy=strategy, **ek)
            oos_result = engine.run(test_data)
            is_report = MetricsReport(sharpe_ratio=entry["is_sharpe"], total_return=entry["is_total_return"])
            oos_report = compute_metrics(oos_result, label="OOS")
            oos_report = compute_overfit_diagnostics(is_report, oos_report, len(combos))

            oos_metric = _get_metric(oos_report, objective)
            delta = entry["is_metric"] - oos_metric
            pct_delta = delta / max(abs(entry["is_metric"]), 1e-8) * 100

            top_comparison.append({
                "params": entry["params"],
                "is_metric": entry["is_metric"],
                "oos_metric": oos_metric,
                "delta": delta,
                "pct_degradation": pct_delta,
                "overfit_flag": pct_delta > 30,
                "oos_sharpe": oos_report.sharpe_ratio,
                "oos_win_rate": oos_report.win_rate,
                "deflated_sharpe": oos_report.deflated_sharpe,
            })

    # Restore original params
    strategy.set_parameters(original_params)

    best = all_results[0] if all_results else {"params": {}, "is_metric": 0}
    return OptimizationResult(
        best_params=best["params"],
        best_is_metric=best["is_metric"],
        best_oos_metric=top_comparison[0]["oos_metric"] if top_comparison else None,
        all_results=all_results,
        top_n_comparison=top_comparison,
        objective=objective,
        n_combos=len(combos),
    )


def _get_metric(report: MetricsReport, objective: str) -> float:
    mapping = {
        "sharpe": report.sharpe_ratio,
        "sortino": report.sortino_ratio,
        "calmar": report.calmar_ratio,
        "total_return": report.total_return,
    }
    return mapping.get(objective, report.sharpe_ratio)


def _save_optimization_log(results: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "optimization_log.csv"
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["params"] = str(row["params"])
            writer.writerow(row)
    logger.info("Saved optimization log to %s", path)
