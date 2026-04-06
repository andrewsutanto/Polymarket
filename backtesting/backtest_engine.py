"""Event-driven backtest engine.

Iterates bar-by-bar through test data, runs regime detection, feeds
data to strategies, passes signals through risk management, simulates
fills, and tracks portfolio equity curve with full trade attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal
from regime.detector import RegimeDetector, Regime
from backtesting.data_loader import BacktestDataset, WeatherBar

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""

    entry_time: datetime
    exit_time: datetime | None
    city: str
    market_id: str
    strategy_name: str
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    edge_at_entry: float
    regime_at_entry: str
    holding_bars: int
    slippage_cost: float
    fee_cost: float


@dataclass
class BacktestResult:
    """Complete result from a backtest run."""

    trades: list[BacktestTrade]
    equity_curve: list[float]
    timestamps: list[datetime]
    regime_history: list[str]
    final_equity: float
    starting_capital: float
    n_bars_processed: int
    config: dict[str, Any]
    dataset_info: dict[str, Any]


@dataclass
class OpenPosition:
    """An open position being tracked."""

    entry_time: datetime
    city: str
    market_id: str
    strategy_name: str
    direction: str
    entry_price: float
    size_usd: float
    edge_at_entry: float
    regime_at_entry: str
    bars_held: int = 0


class BacktestEngine:
    """Event-driven backtesting engine that reuses live trading logic."""

    def __init__(
        self,
        strategy: BaseStrategy,
        starting_capital: float = 50.0,
        slippage_pct: float = 0.005,
        fee_pct: float = 0.02,
        latency_bars: int = 1,
        max_position_usd: float = 5.0,
        max_open_positions: int = 15,
        max_drawdown: float = 0.20,
        kelly_fraction: float = 0.5,
        min_trade_size: float = 0.50,
        max_trade_size: float = 5.00,
        regime_adaptive: bool = False,
    ) -> None:
        self._strategy = strategy
        self._starting_capital = starting_capital
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct
        self._latency_bars = latency_bars
        self._max_position_usd = max_position_usd
        self._max_open = max_open_positions
        self._max_drawdown = max_drawdown
        self._kelly_fraction = kelly_fraction
        self._min_trade_size = min_trade_size
        self._max_trade_size = max_trade_size
        self._regime_adaptive = regime_adaptive

        # Runtime state
        self._equity = starting_capital
        self._cash = starting_capital
        self._peak_equity = starting_capital
        self._positions: dict[str, OpenPosition] = {}
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[float] = []
        self._timestamps: list[datetime] = []
        self._regime_history: list[str] = []
        self._regime_detector = RegimeDetector()
        self._pending_signals: list[tuple[int, Signal]] = []  # (execute_at_bar, signal)
        self._killed = False
        self._bar_idx = 0

    def run(self, dataset: BacktestDataset) -> BacktestResult:
        """Execute the backtest over the given dataset.

        Args:
            dataset: The test-period dataset to backtest on.

        Returns:
            BacktestResult with full trade log, equity curve, and metadata.
        """
        self._reset()
        bars = dataset.bars

        logger.info(
            "Starting backtest: %d bars, %s strategy, $%.2f capital",
            len(bars), self._strategy.name, self._starting_capital,
        )

        for i, bar in enumerate(bars):
            self._bar_idx = i

            # Anti-lookahead assertion
            if i > 0:
                assert bar.timestamp >= bars[i - 1].timestamp, (
                    f"Temporal ordering violation at bar {i}: "
                    f"{bar.timestamp} < {bars[i-1].timestamp}"
                )

            # Update regime
            regime = self._regime_detector.update({
                "city": bar.city,
                "model_prob": bar.model_prob,
                "market_price": bar.market_price,
                "forecast_shift_f": bar.forecast_shift_f,
                "is_forecast_update": bar.is_forecast_update,
            })

            # Execute pending signals (latency model)
            self._execute_pending(i, bar)

            # Mark-to-market open positions
            self._mark_to_market(bar)

            # Check kill switch
            if not self._killed:
                drawdown = self._get_drawdown()
                if drawdown >= self._max_drawdown:
                    self._killed = True
                    logger.warning("Kill switch at bar %d, drawdown=%.2f%%", i, drawdown * 100)

            # Generate signal
            if not self._killed:
                market_data = self._build_market_data(bar, regime)
                signal = self._strategy.generate_signal(market_data)

                if signal is not None:
                    execute_at = i + self._latency_bars
                    self._pending_signals.append((execute_at, signal))

            # Record state
            self._equity_curve.append(self._get_equity(bar))
            self._timestamps.append(bar.timestamp)
            self._regime_history.append(regime.value)

        # Close all remaining positions at last bar
        if bars:
            self._close_all(bars[-1])

        return BacktestResult(
            trades=self._trades,
            equity_curve=self._equity_curve,
            timestamps=self._timestamps,
            regime_history=self._regime_history,
            final_equity=self._equity_curve[-1] if self._equity_curve else self._starting_capital,
            starting_capital=self._starting_capital,
            n_bars_processed=len(bars),
            config=self._get_config(),
            dataset_info={
                "cities": dataset.cities,
                "n_bars": dataset.n_bars,
                "start": str(dataset.start_date),
                "end": str(dataset.end_date),
                "synthetic": dataset.odds_synthetic,
            },
        )

    def _reset(self) -> None:
        self._equity = self._starting_capital
        self._cash = self._starting_capital
        self._peak_equity = self._starting_capital
        self._positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        self._timestamps.clear()
        self._regime_history.clear()
        self._pending_signals.clear()
        self._regime_detector.reset()
        self._strategy.reset()
        self._killed = False
        self._bar_idx = 0

    def _build_market_data(self, bar: WeatherBar, regime: Regime) -> dict[str, Any]:
        pos_key = f"{bar.city}_{bar.market_price:.4f}"
        has_pos = any(p.city == bar.city for p in self._positions.values())
        return {
            "timestamp": bar.timestamp,
            "city": bar.city,
            "market_id": f"{bar.city}_{bar.lead_days}",
            "market_price": bar.market_price,
            "model_prob": bar.model_prob,
            "forecast_high_f": bar.forecast_high_f,
            "forecast_shift_f": bar.forecast_shift_f,
            "forecast_update_time": bar.timestamp if bar.is_forecast_update else None,
            "book_depth": bar.book_depth,
            "spread": bar.spread,
            "volume_24h": bar.volume,
            "lead_days": bar.lead_days,
            "confidence": min(bar.model_prob * 1.2, 1.0),
            "has_position": has_pos,
            "regime": regime.value,
            "time_to_resolution_hrs": bar.lead_days * 24,
            "is_forecast_update": bar.is_forecast_update,
        }

    def _execute_pending(self, current_bar: int, bar: WeatherBar) -> None:
        remaining: list[tuple[int, Signal]] = []
        for exec_bar, signal in self._pending_signals:
            if current_bar >= exec_bar:
                self._execute_signal(signal, bar)
            else:
                remaining.append((exec_bar, signal))
        self._pending_signals = remaining

    def _execute_signal(self, signal: Signal, bar: WeatherBar) -> None:
        pos_key = f"{signal.city}_{signal.market_id}"

        if signal.direction == "SELL":
            if pos_key in self._positions:
                self._close_position(pos_key, bar)
            return

        # BUY
        if pos_key in self._positions:
            return
        if len(self._positions) >= self._max_open:
            return

        # Half-Kelly sizing
        size = self._compute_kelly_size(signal, bar)
        if size < self._min_trade_size:
            return
        size = min(size, self._max_trade_size, self._cash)
        if size < self._min_trade_size:
            return

        # Apply slippage
        fill_price = bar.market_price * (1.0 + self._slippage_pct)
        fee = size * self._fee_pct
        total_cost = size + fee

        if total_cost > self._cash:
            return

        self._cash -= total_cost
        self._positions[pos_key] = OpenPosition(
            entry_time=bar.timestamp,
            city=signal.city,
            market_id=signal.market_id,
            strategy_name=signal.strategy_name,
            direction=signal.direction,
            entry_price=fill_price,
            size_usd=size,
            edge_at_entry=signal.edge,
            regime_at_entry=self._regime_detector.current_regime.value,
        )

    def _close_position(self, pos_key: str, bar: WeatherBar) -> None:
        pos = self._positions.pop(pos_key, None)
        if not pos:
            return

        exit_price = bar.market_price * (1.0 - self._slippage_pct)
        fee = pos.size_usd * self._fee_pct
        slippage = pos.size_usd * self._slippage_pct * 2  # Entry + exit

        qty = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0
        proceeds = qty * exit_price
        pnl = proceeds - pos.size_usd - fee

        self._cash += proceeds - fee

        self._trades.append(BacktestTrade(
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            city=pos.city,
            market_id=pos.market_id,
            strategy_name=pos.strategy_name,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            pnl=pnl,
            edge_at_entry=pos.edge_at_entry,
            regime_at_entry=pos.regime_at_entry,
            holding_bars=pos.bars_held,
            slippage_cost=slippage,
            fee_cost=fee,
        ))

    def _close_all(self, bar: WeatherBar) -> None:
        for key in list(self._positions.keys()):
            self._close_position(key, bar)

    def _mark_to_market(self, bar: WeatherBar) -> None:
        for pos in self._positions.values():
            if pos.city == bar.city:
                pos.bars_held += 1

    def _get_equity(self, bar: WeatherBar) -> float:
        unrealized = 0.0
        for pos in self._positions.values():
            qty = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0
            unrealized += qty * bar.market_price - pos.size_usd
        equity = self._cash + sum(p.size_usd for p in self._positions.values()) + unrealized
        self._peak_equity = max(self._peak_equity, equity)
        return equity

    def _get_drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        current = self._equity_curve[-1] if self._equity_curve else self._starting_capital
        return (self._peak_equity - current) / self._peak_equity

    def _compute_kelly_size(self, signal: Signal, bar: WeatherBar) -> float:
        p = signal.strength
        if bar.market_price <= 0 or bar.market_price >= 1:
            return 0.0
        odds = (1.0 / bar.market_price) - 1.0
        if odds <= 0:
            return 0.0
        kelly = (p * odds - (1.0 - p)) / odds
        kelly *= self._kelly_fraction
        if kelly <= 0:
            return 0.0
        return kelly * self._cash

    def _get_config(self) -> dict[str, Any]:
        return {
            "starting_capital": self._starting_capital,
            "slippage_pct": self._slippage_pct,
            "fee_pct": self._fee_pct,
            "latency_bars": self._latency_bars,
            "max_position_usd": self._max_position_usd,
            "max_open_positions": self._max_open,
            "max_drawdown": self._max_drawdown,
            "kelly_fraction": self._kelly_fraction,
            "strategy": self._strategy.name,
            "strategy_params": self._strategy.get_parameters(),
        }
