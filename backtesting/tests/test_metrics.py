"""Tests for performance metrics — verify against hand-calculated examples."""

import pytest
import numpy as np
from datetime import datetime, timezone

from backtesting.metrics import compute_metrics, _max_drawdown, _sharpe, _omega, _consecutive
from backtesting.backtest_engine import BacktestResult, BacktestTrade


def _make_result(equity: list[float], trades: list[BacktestTrade] | None = None) -> BacktestResult:
    return BacktestResult(
        trades=trades or [],
        equity_curve=equity,
        timestamps=[datetime(2026, 1, 1, tzinfo=timezone.utc)] * len(equity),
        regime_history=["NEUTRAL"] * len(equity),
        final_equity=equity[-1] if equity else 50.0,
        starting_capital=equity[0] if equity else 50.0,
        n_bars_processed=len(equity),
        config={},
        dataset_info={},
    )


def _make_trade(pnl: float, city: str = "NYC", strategy: str = "test", regime: str = "NEUTRAL") -> BacktestTrade:
    return BacktestTrade(
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
        city=city, market_id="test", strategy_name=strategy,
        direction="BUY", entry_price=0.10, exit_price=0.15 if pnl > 0 else 0.05,
        size_usd=1.0, pnl=pnl, edge_at_entry=0.05, regime_at_entry=regime,
        holding_bars=10, slippage_cost=0.005, fee_cost=0.02,
    )


class TestMaxDrawdown:
    def test_no_drawdown(self):
        dd, dur = _max_drawdown(np.array([100, 101, 102, 103]))
        assert dd == 0.0
        assert dur == 0

    def test_simple_drawdown(self):
        dd, dur = _max_drawdown(np.array([100, 90, 80, 90, 100]))
        assert abs(dd - 0.20) < 0.01  # 20% drawdown
        assert dur == 3  # 3 bars in drawdown

    def test_deeper_second_drawdown(self):
        dd, _ = _max_drawdown(np.array([100, 95, 100, 70, 80]))
        assert abs(dd - 0.30) < 0.01  # 30% from 100 to 70


class TestSharpe:
    def test_positive_sharpe(self):
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.005] * 20)
        sr = _sharpe(returns, 365)
        assert sr > 0

    def test_zero_returns(self):
        returns = np.zeros(50)
        sr = _sharpe(returns, 365)
        assert sr == 0.0

    def test_negative_returns_negative_sharpe(self):
        returns = np.array([-0.01, -0.02, -0.005, -0.015] * 20)
        sr = _sharpe(returns, 365)
        assert sr < 0


class TestOmega:
    def test_all_positive(self):
        returns = np.array([0.01, 0.02, 0.03])
        assert _omega(returns) == float("inf") or _omega(returns) > 10

    def test_balanced_returns(self):
        returns = np.array([0.01, -0.01, 0.01, -0.01])
        omega = _omega(returns)
        assert abs(omega - 1.0) < 0.5


class TestConsecutive:
    def test_basic_streak(self):
        pnls = [1, 1, 1, -1, -1, 1]
        wins, losses = _consecutive(pnls)
        assert wins == 3
        assert losses == 2

    def test_all_wins(self):
        pnls = [1, 1, 1, 1]
        wins, losses = _consecutive(pnls)
        assert wins == 4
        assert losses == 0


class TestComputeMetrics:
    def test_win_rate(self):
        trades = [_make_trade(0.5), _make_trade(0.3), _make_trade(-0.2)]
        result = _make_result([50, 50.5, 50.8, 50.6], trades)
        report = compute_metrics(result)
        assert abs(report.win_rate - 2.0 / 3.0) < 0.01

    def test_strategy_attribution(self):
        trades = [
            _make_trade(0.5, strategy="A"),
            _make_trade(-0.2, strategy="A"),
            _make_trade(0.3, strategy="B"),
        ]
        result = _make_result([50, 50.5, 50.3, 50.6], trades)
        report = compute_metrics(result)
        assert "A" in report.strategy_pnl
        assert "B" in report.strategy_pnl
        assert abs(report.strategy_pnl["A"] - 0.3) < 0.01
        assert abs(report.strategy_pnl["B"] - 0.3) < 0.01

    def test_city_attribution(self):
        trades = [_make_trade(0.5, city="NYC"), _make_trade(0.3, city="CHI")]
        result = _make_result([50, 50.5, 50.8], trades)
        report = compute_metrics(result)
        assert "NYC" in report.city_pnl
        assert "CHI" in report.city_pnl

    def test_empty_result(self):
        result = _make_result([50])
        report = compute_metrics(result)
        assert report.n_trades == 0
        assert report.total_return == 0.0
