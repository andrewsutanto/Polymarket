"""CLI entry point for the backtesting framework.

Supports single strategy, ensemble, head-to-head comparison, walk-forward,
parameter optimization, and HTML/terminal reporting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from backtesting.data_loader import load_dataset
from backtesting.splitter import get_splits
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import compute_metrics, compute_overfit_diagnostics
from backtesting.optimizer import grid_search, random_search
from backtesting.walk_forward import run_walk_forward
from backtesting.report import print_terminal_report, generate_html_report
from strategies.forecast_arb import ForecastArbStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.forecast_momentum import ForecastMomentumStrategy
from strategies.cross_city_arb import CrossCityArbStrategy
from strategies.ensemble import EnsembleStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "forecast_arb": ForecastArbStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": ForecastMomentumStrategy,
    "cross_city": CrossCityArbStrategy,
}

# Default parameter ranges for optimization
DEFAULT_PARAM_RANGES = {
    "forecast_arb": {
        "entry_threshold": [0.10, 0.12, 0.15, 0.18, 0.20],
        "min_value_ratio": [2.0, 2.5, 3.0, 3.5, 4.0],
        "min_edge": [0.05, 0.08, 0.10, 0.12, 0.15],
    },
    "mean_reversion": {
        "zscore_entry": [1.5, 2.0, 2.5, 3.0],
        "zscore_exit": [0.3, 0.5, 0.7],
        "lookback": [30, 50, 75, 100],
    },
    "momentum": {
        "drift_threshold": [0.4, 0.5, 0.6, 0.7, 0.8],
        "entry_edge": [0.03, 0.05, 0.07, 0.10],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weather Arbitrage Backtesting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--city", type=str, default="all", help="City or comma-separated list (default: all)")
    parser.add_argument("--days", type=int, default=180, help="Historical days (default: 180)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Strategy
    parser.add_argument("--strategy", type=str, default="forecast_arb",
                        help="Strategy: forecast_arb, mean_reversion, momentum, cross_city, ensemble, or comma-separated for --compare")
    parser.add_argument("--regime-adaptive", action="store_true", help="Use regime-adaptive ensemble weights")
    parser.add_argument("--compare", action="store_true", help="Run strategies head-to-head")

    # Splitting
    parser.add_argument("--split", type=str, default="simple", help="Split: simple, embargo, walkforward, kfold")
    parser.add_argument("--ratio", type=float, default=0.7, help="Train/test ratio (default: 0.7)")
    parser.add_argument("--embargo-days", type=int, default=3, help="Embargo gap days (default: 3)")
    parser.add_argument("--train-days", type=int, default=60, help="Walk-forward train window (default: 60)")
    parser.add_argument("--test-days", type=int, default=15, help="Walk-forward test window (default: 15)")
    parser.add_argument("--n-folds", type=int, default=5, help="K-fold folds (default: 5)")

    # Optimization
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimization on IS")
    parser.add_argument("--objective", type=str, default="sharpe", help="Optimization objective (default: sharpe)")
    parser.add_argument("--n-trials", type=int, default=30, help="Random search trials (default: 30)")

    # Output
    parser.add_argument("--report", type=str, default="terminal", help="Report: terminal, html, both")
    parser.add_argument("--capital", type=float, default=50.0, help="Starting capital (default: 50)")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    # Parse cities
    if args.city.lower() == "all":
        cities = None  # All configured
    else:
        cities = [c.strip() for c in args.city.split(",")]

    # Load data
    logger.info("Loading dataset: %d days, cities=%s", args.days, cities or "all")
    dataset = load_dataset(cities=cities, days=args.days, seed=args.seed)
    logger.info("Dataset: %d bars, %s to %s", dataset.n_bars, dataset.start_date, dataset.end_date)

    if dataset.odds_synthetic:
        logger.warning("*** USING SYNTHETIC ODDS DATA — results are indicative only ***")

    engine_kwargs = {"starting_capital": args.capital}

    # Build strategy
    strategy_names = [s.strip() for s in args.strategy.split(",")]

    if args.compare:
        # Head-to-head comparison
        _run_comparison(strategy_names, dataset, args, engine_kwargs)
        return

    if args.split == "walkforward" and args.optimize:
        # Walk-forward with optimization
        strategy = _build_strategy(strategy_names[0], args.regime_adaptive)
        param_grid = DEFAULT_PARAM_RANGES.get(strategy_names[0])
        wf_result = run_walk_forward(
            strategy, dataset, args.train_days, args.test_days,
            param_grid=param_grid, objective=args.objective,
            seed=args.seed, engine_kwargs=engine_kwargs,
        )
        if "html" in args.report or args.report == "both":
            generate_html_report(wf_result=wf_result)
        print_terminal_report(wf_result=wf_result)
        return

    # Standard split
    strategy = _build_strategy(strategy_names[0], args.regime_adaptive)
    splits = get_splits(
        dataset, args.split, args.ratio, args.embargo_days,
        args.train_days, args.test_days, args.n_folds,
    )

    if not splits:
        logger.error("No valid splits generated")
        return

    split = splits[0]  # For simple/embargo, use first split

    # Optimization
    n_combos = 1
    if args.optimize:
        param_grid = DEFAULT_PARAM_RANGES.get(strategy_names[0])
        if param_grid:
            opt_result = grid_search(
                strategy, param_grid, split.train_data, split.test_data,
                objective=args.objective, engine_kwargs=engine_kwargs,
            )
            n_combos = opt_result.n_combos
            strategy.set_parameters(opt_result.best_params)
            logger.info("Best params: %s (IS %s=%.3f)", opt_result.best_params,
                        args.objective, opt_result.best_is_metric)

            if opt_result.top_n_comparison:
                logger.info("Degradation table:")
                for c in opt_result.top_n_comparison:
                    flag = " *** OVERFIT ***" if c["overfit_flag"] else ""
                    logger.info("  IS=%.3f OOS=%.3f delta=%.1f%%%s",
                                c["is_metric"], c["oos_metric"], c["pct_degradation"], flag)

    # Run IS backtest
    engine = BacktestEngine(strategy=strategy, **engine_kwargs)
    is_result = engine.run(split.train_data)
    is_report = compute_metrics(is_result, label="IS")

    # Run OOS backtest
    engine = BacktestEngine(strategy=strategy, **engine_kwargs)
    oos_result = engine.run(split.test_data)
    oos_report = compute_metrics(oos_result, label="OOS")
    oos_report = compute_overfit_diagnostics(is_report, oos_report, n_combos)

    # Report
    if args.report in ("terminal", "both"):
        print_terminal_report(is_report, oos_report, oos_result)
    if args.report in ("html", "both"):
        generate_html_report(is_report, oos_report, oos_result)


def _build_strategy(name: str, regime_adaptive: bool = False) -> "BaseStrategy":
    if name == "ensemble":
        strats = [cls() for cls in STRATEGY_MAP.values()]
        ens = EnsembleStrategy(strategies=strats)
        if regime_adaptive:
            from regime.detector import DEFAULT_REGIME_WEIGHTS
            ens.set_regime_weights(DEFAULT_REGIME_WEIGHTS)
        return ens

    cls = STRATEGY_MAP.get(name)
    if cls is None:
        logger.error("Unknown strategy: %s", name)
        sys.exit(1)
    return cls()


def _run_comparison(
    strategy_names: list[str],
    dataset,
    args,
    engine_kwargs: dict,
) -> None:
    """Run multiple strategies and compare."""
    from backtesting.splitter import simple_split
    split = simple_split(dataset, args.ratio)

    results: dict[str, tuple] = {}
    for name in strategy_names:
        strategy = _build_strategy(name)
        engine = BacktestEngine(strategy=strategy, **engine_kwargs)
        is_result = engine.run(split.train_data)
        is_report = compute_metrics(is_result, label=f"IS-{name}")

        engine = BacktestEngine(strategy=strategy, **engine_kwargs)
        oos_result = engine.run(split.test_data)
        oos_report = compute_metrics(oos_result, label=f"OOS-{name}")
        results[name] = (is_report, oos_report, oos_result)

    # Print comparison
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="Strategy Comparison (OOS)", header_style="bold cyan")
    table.add_column("Strategy", width=18)
    table.add_column("Return", justify="right", width=10)
    table.add_column("Sharpe", justify="right", width=8)
    table.add_column("Win Rate", justify="right", width=10)
    table.add_column("Max DD", justify="right", width=10)
    table.add_column("Trades", justify="right", width=8)
    table.add_column("P/F", justify="right", width=8)

    for name, (is_r, oos_r, _) in results.items():
        table.add_row(
            name, f"{oos_r.total_return:.2%}", f"{oos_r.sharpe_ratio:.2f}",
            f"{oos_r.win_rate:.0%}", f"{oos_r.max_drawdown:.2%}",
            str(oos_r.n_trades), f"{oos_r.profit_factor:.2f}",
        )
    console.print(table)

    if args.report in ("html", "both"):
        first = list(results.values())[0]
        generate_html_report(first[0], first[1], first[2])


if __name__ == "__main__":
    main()
