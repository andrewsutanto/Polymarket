"""Risk management: Kelly sizing, position limits, and kill switch.

All trade proposals pass through the risk manager before execution.
Works with the strategies.base.Signal type (market-agnostic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Any

from config import settings
from strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class TradeProposal:
    signal: Signal
    size_usd: float
    kelly_fraction: float
    approved: bool
    reject_reason: str


class RiskManager:
    """Validates trade proposals against risk limits and sizes positions."""

    def __init__(self, starting_capital: float) -> None:
        self._starting_capital = starting_capital
        self._available_cash: float = starting_capital
        self._open_positions: dict[str, float] = {}
        self._trades_this_cycle: int = 0
        self._session_pnl: float = 0.0
        self._peak_value: float = starting_capital
        self._kill_switch_active = False
        self._drawdown_alerts_sent: set[float] = set()
        self._kill_callbacks: list[Callable[[], Any]] = []
        self._drawdown_callbacks: list[Callable[[float], Any]] = []

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def on_kill_switch(self, cb: Callable[[], Any]) -> None:
        self._kill_callbacks.append(cb)

    def on_drawdown_warning(self, cb: Callable[[float], Any]) -> None:
        self._drawdown_callbacks.append(cb)

    def activate_kill_switch(self) -> None:
        if not self._kill_switch_active:
            self._kill_switch_active = True
            logger.critical("KILL SWITCH ACTIVATED — all trading halted")
            for cb in self._kill_callbacks:
                try:
                    cb()
                except Exception:
                    logger.exception("Kill switch callback error")

    def reset_cycle(self) -> None:
        self._trades_this_cycle = 0

    def update_state(
        self,
        available_cash: float,
        open_positions: dict[str, float],
        session_pnl: float,
        portfolio_value: float,
    ) -> None:
        """Sync risk state with portfolio tracker."""
        self._available_cash = available_cash
        self._open_positions = dict(open_positions)
        self._session_pnl = session_pnl
        self._peak_value = max(self._peak_value, portfolio_value)
        self._check_drawdown(portfolio_value)

    def evaluate_signal(self, signal: Signal) -> TradeProposal | None:
        """Evaluate a strategies.base.Signal and return a sized TradeProposal.

        Args:
            signal: Signal from any strategy.

        Returns:
            TradeProposal if approved, None if rejected.
        """
        if self._kill_switch_active:
            return None

        token_id = signal.token_id or signal.market_id

        if signal.direction == "SELL":
            size = self._open_positions.get(token_id, 0.0)
            if size <= 0:
                return None
            return TradeProposal(
                signal=signal,
                size_usd=size,
                kelly_fraction=0.0,
                approved=True,
                reject_reason="",
            )

        # BUY entry filters
        if signal.strength < settings.MIN_CONFIDENCE:
            return None

        if signal.edge < settings.MIN_EDGE:
            return None

        existing = self._open_positions.get(token_id, 0.0)
        if existing >= settings.MAX_POSITION_USD:
            return None

        if len(self._open_positions) >= settings.MAX_OPEN_POSITIONS:
            return None

        if self._trades_this_cycle >= settings.MAX_TRADES_PER_RUN:
            return None

        # Half-Kelly sizing
        kelly, size = self._compute_kelly_size(signal)
        if kelly <= 0:
            return None

        size = min(size, self._available_cash)
        size = min(size, settings.MAX_POSITION_USD - existing)

        if size < settings.MIN_TRADE_SIZE:
            return None

        self._trades_this_cycle += 1
        return TradeProposal(
            signal=signal,
            size_usd=round(size, 2),
            kelly_fraction=kelly,
            approved=True,
            reject_reason="",
        )

    # Keep backwards-compatible evaluate() for backtest engine
    def evaluate(self, signal: Any) -> TradeProposal:
        """Legacy evaluate for backtest engine compatibility."""
        result = self.evaluate_signal(signal) if isinstance(signal, Signal) else None
        if result:
            return result
        return TradeProposal(
            signal=signal,
            size_usd=0.0,
            kelly_fraction=0.0,
            approved=False,
            reject_reason="Rejected",
        )

    def _compute_kelly_size(self, signal: Signal) -> tuple[float, float]:
        """Compute half-Kelly position size from signal edge + strength."""
        # Use signal strength as probability estimate, edge as information
        p = signal.strength
        market_price = 1.0 - signal.edge  # Approximate market price from edge
        if market_price <= 0 or market_price >= 1:
            # Fall back to using metadata if available
            market_price = signal.metadata.get("mid_price", signal.metadata.get("yes_price", 0.5))

        if market_price <= 0 or market_price >= 1:
            return 0.0, 0.0

        odds = (1.0 / market_price) - 1.0
        if odds <= 0:
            return 0.0, 0.0

        q = 1.0 - p
        kelly = (p * odds - q) / odds
        kelly *= settings.KELLY_FRACTION

        if kelly <= 0:
            return kelly, 0.0

        size = kelly * self._available_cash
        size = max(settings.MIN_TRADE_SIZE, min(size, settings.MAX_TRADE_SIZE))
        return round(kelly, 4), round(size, 2)

    def _check_drawdown(self, portfolio_value: float) -> None:
        if self._peak_value <= 0:
            return
        drawdown = (self._peak_value - portfolio_value) / self._peak_value

        for level in settings.DRAWDOWN_WARNING_LEVELS:
            if drawdown >= level and level not in self._drawdown_alerts_sent:
                self._drawdown_alerts_sent.add(level)
                logger.warning("Drawdown warning: %.1f%%", drawdown * 100)
                for cb in self._drawdown_callbacks:
                    try:
                        cb(drawdown)
                    except Exception:
                        logger.exception("Drawdown callback error")

        if drawdown >= settings.MAX_DAILY_DRAWDOWN:
            self.activate_kill_switch()
