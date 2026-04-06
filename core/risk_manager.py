"""Risk management: Kelly sizing, position limits, and kill switch.

All trade proposals pass through the risk manager before execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Any

from config import settings
from core.signal_engine import Signal

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
        self._open_positions: dict[str, float] = {}  # contract_id -> size_usd
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

    def evaluate(self, signal: Signal) -> TradeProposal:
        """Evaluate a signal and return a sized trade proposal.

        Args:
            signal: The BUY or SELL signal to evaluate.

        Returns:
            TradeProposal with approved=True/False and sizing.
        """
        if self._kill_switch_active:
            return self._reject(signal, "Kill switch active")

        if signal.signal_type == "SELL":
            size = self._open_positions.get(signal.contract_id, 0.0)
            return TradeProposal(
                signal=signal,
                size_usd=size,
                kelly_fraction=0.0,
                approved=size > 0,
                reject_reason="" if size > 0 else "No open position",
            )

        # BUY entry filters
        if signal.confidence < settings.MIN_CONFIDENCE:
            return self._reject(signal, f"Confidence {signal.confidence:.2f} < {settings.MIN_CONFIDENCE}")

        existing_size = self._open_positions.get(signal.contract_id, 0.0)
        if existing_size >= settings.MAX_POSITION_USD:
            return self._reject(signal, f"Position limit reached: ${existing_size:.2f}")

        if len(self._open_positions) >= settings.MAX_OPEN_POSITIONS:
            return self._reject(signal, f"Max open positions ({settings.MAX_OPEN_POSITIONS}) reached")

        if self._trades_this_cycle >= settings.MAX_TRADES_PER_RUN:
            return self._reject(signal, f"Max trades per cycle ({settings.MAX_TRADES_PER_RUN}) reached")

        if signal.signal_type == "BUY":
            ttl_hrs = 0.0
            snap_data = getattr(signal, "time_to_resolution_hrs", None)
            if snap_data is None:
                from datetime import timedelta
                ttl_hrs = (
                    datetime.combine(signal.target_date, datetime.max.time()).replace(tzinfo=timezone.utc)
                    - datetime.now(timezone.utc)
                ).total_seconds() / 3600.0
            if ttl_hrs < settings.MIN_TTL_HOURS:
                return self._reject(signal, f"TTL {ttl_hrs:.1f}h < {settings.MIN_TTL_HOURS}h")

        # Half-Kelly sizing
        kelly, size = self._compute_kelly_size(signal)
        if kelly <= 0:
            return self._reject(signal, f"Kelly fraction non-positive: {kelly:.4f}")

        size = min(size, self._available_cash)
        size = min(size, settings.MAX_POSITION_USD - existing_size)

        if size < settings.MIN_TRADE_SIZE:
            return self._reject(signal, f"Size ${size:.2f} below minimum ${settings.MIN_TRADE_SIZE}")

        self._trades_this_cycle += 1
        return TradeProposal(
            signal=signal,
            size_usd=round(size, 2),
            kelly_fraction=kelly,
            approved=True,
            reject_reason="",
        )

    def _compute_kelly_size(self, signal: Signal) -> tuple[float, float]:
        """Compute half-Kelly position size.

        Returns:
            (kelly_fraction, dollar_size)
        """
        if signal.market_price <= 0 or signal.market_price >= 1:
            return 0.0, 0.0

        odds = (1.0 / signal.market_price) - 1.0
        if odds <= 0:
            return 0.0, 0.0

        p = signal.model_prob
        q = 1.0 - p
        kelly = (p * odds - q) / odds
        kelly *= settings.KELLY_FRACTION  # half-Kelly

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
                logger.warning("Drawdown warning: %.1f%% (threshold: %.1f%%)", drawdown * 100, level * 100)
                for cb in self._drawdown_callbacks:
                    try:
                        cb(drawdown)
                    except Exception:
                        logger.exception("Drawdown callback error")

        if drawdown >= settings.MAX_DAILY_DRAWDOWN:
            self.activate_kill_switch()

    @staticmethod
    def _reject(signal: Signal, reason: str) -> TradeProposal:
        logger.debug("REJECT %s %s %s: %s", signal.signal_type, signal.location, signal.bucket_label, reason)
        return TradeProposal(
            signal=signal,
            size_usd=0.0,
            kelly_fraction=0.0,
            approved=False,
            reject_reason=reason,
        )
