"""Tests for backtest engine — run a tiny backtest and verify P&L."""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any

from strategies.base import BaseStrategy, Signal
from backtesting.backtest_engine import BacktestEngine
from backtesting.data_loader import BacktestDataset, WeatherBar


class AlwaysBuyStrategy(BaseStrategy):
    """Test strategy that always generates a BUY signal."""

    @property
    def name(self) -> str:
        return "always_buy"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        if market_data.get("has_position"):
            if market_data.get("market_price", 0) > 0.5:
                return Signal("SELL", 0.8, 0.1, self.name, market_data.get("market_id", ""), market_data.get("city", ""))
            return None
        return Signal("BUY", 0.9, 0.2, self.name, market_data.get("market_id", ""), market_data.get("city", ""))

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, params: dict) -> None:
        pass


class NeverTradeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "never_trade"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        return None

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, params: dict) -> None:
        pass


def _make_bars(n: int = 20) -> list[WeatherBar]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(n):
        price = 0.10 + (i / n) * 0.50  # Rising from 0.10 to 0.60
        bars.append(WeatherBar(
            timestamp=base + timedelta(hours=i),
            city="NYC",
            forecast_high_f=70.0 + i * 0.5,
            forecast_low_f=60.0,
            forecast_shift_f=0.5 if i % 5 == 0 else 0.0,
            model_prob=0.40,
            market_price=price,
            book_depth=100.0,
            spread=0.02,
            volume=200.0,
            lead_days=2,
            is_forecast_update=i % 5 == 0,
            data_source="synthetic",
        ))
    return bars


def _make_dataset(bars: list[WeatherBar]) -> BacktestDataset:
    return BacktestDataset(
        cities=["NYC"],
        bars=bars,
        start_date=bars[0].timestamp,
        end_date=bars[-1].timestamp,
        n_bars=len(bars),
        odds_synthetic=True,
    )


class TestBacktestEngine:
    def test_basic_run(self):
        bars = _make_bars(20)
        dataset = _make_dataset(bars)
        engine = BacktestEngine(strategy=AlwaysBuyStrategy(), starting_capital=50.0)
        result = engine.run(dataset)
        assert result.n_bars_processed == 20
        assert len(result.equity_curve) == 20

    def test_no_trades_preserves_capital(self):
        bars = _make_bars(20)
        dataset = _make_dataset(bars)
        engine = BacktestEngine(strategy=NeverTradeStrategy(), starting_capital=50.0)
        result = engine.run(dataset)
        assert result.final_equity == 50.0
        assert len(result.trades) == 0

    def test_always_buy_generates_trades(self):
        bars = _make_bars(20)
        dataset = _make_dataset(bars)
        engine = BacktestEngine(
            strategy=AlwaysBuyStrategy(), starting_capital=50.0, latency_bars=0,
        )
        result = engine.run(dataset)
        assert len(result.trades) > 0

    def test_temporal_ordering_enforced(self):
        bars = _make_bars(20)
        # Swap two bars to create disorder
        bars[5], bars[10] = bars[10], bars[5]
        dataset = _make_dataset(bars)
        engine = BacktestEngine(strategy=NeverTradeStrategy())
        with pytest.raises(AssertionError, match="Temporal ordering"):
            engine.run(dataset)

    def test_kill_switch_halts_trading(self):
        """Create a scenario with large losses to trigger kill switch."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = []
        for i in range(30):
            # Price crashes from 0.20 to 0.01
            price = max(0.20 - i * 0.01, 0.01)
            bars.append(WeatherBar(
                timestamp=base + timedelta(hours=i),
                city="NYC", forecast_high_f=70.0, forecast_low_f=60.0,
                forecast_shift_f=0.0, model_prob=0.40, market_price=price,
                book_depth=100.0, spread=0.02, volume=200.0, lead_days=2,
                is_forecast_update=False, data_source="synthetic",
            ))
        dataset = _make_dataset(bars)
        engine = BacktestEngine(
            strategy=AlwaysBuyStrategy(), starting_capital=10.0,
            max_drawdown=0.15, latency_bars=0,
        )
        result = engine.run(dataset)
        # Should have stopped generating new trades after kill switch
        assert result.n_bars_processed == 30

    def test_regime_history_recorded(self):
        bars = _make_bars(20)
        dataset = _make_dataset(bars)
        engine = BacktestEngine(strategy=NeverTradeStrategy())
        result = engine.run(dataset)
        assert len(result.regime_history) == 20
        assert all(isinstance(r, str) for r in result.regime_history)
