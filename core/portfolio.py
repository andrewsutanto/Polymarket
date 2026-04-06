"""Portfolio state tracking, P&L calculation, and resolution outcomes.

In-memory state with SQLite persistence via the database module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import settings
from core.executor import TradeResult

logger = logging.getLogger(__name__)


@dataclass
class Position:
    contract_id: str
    location: str
    target_date: str
    bucket_label: str
    side: str
    qty: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    mode: str


@dataclass
class ClosedPosition:
    contract_id: str
    location: str
    target_date: str
    bucket_label: str
    entry_price: float
    exit_price: float
    size_usd: float
    realized_pnl: float
    resolution: str  # WON / LOST / SOLD
    closed_at: datetime


class Portfolio:
    """Tracks portfolio state, open positions, and P&L."""

    def __init__(self, starting_capital: float | None = None) -> None:
        self._starting_capital = starting_capital or settings.STARTING_CAPITAL
        self._cash = self._starting_capital
        self._positions: dict[str, Position] = {}
        self._closed: list[ClosedPosition] = []
        self._start_time = datetime.now(timezone.utc)
        self._peak_value = self._starting_capital
        self._daily_start_value = self._starting_capital

    @property
    def cash(self) -> float:
        return self._cash

    def get_portfolio_value(self) -> float:
        unrealized = sum(p.qty * p.current_price for p in self._positions.values())
        return self._cash + unrealized

    def get_daily_pnl(self) -> float:
        return self.get_portfolio_value() - self._daily_start_value

    def get_drawdown(self) -> float:
        value = self.get_portfolio_value()
        self._peak_value = max(self._peak_value, value)
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - value) / self._peak_value

    def get_position_usd(self, contract_id: str) -> float:
        pos = self._positions.get(contract_id)
        if not pos:
            return 0.0
        return pos.qty * pos.current_price

    def get_win_rate(self) -> float:
        if not self._closed:
            return 0.0
        wins = sum(1 for c in self._closed if c.realized_pnl > 0)
        return wins / len(self._closed)

    def get_resolution_accuracy(self) -> float:
        resolved = [c for c in self._closed if c.resolution in ("WON", "LOST")]
        if not resolved:
            return 0.0
        won = sum(1 for c in resolved if c.resolution == "WON")
        return won / len(resolved)

    def get_open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_open_position_map(self) -> dict[str, float]:
        """Return {contract_id: entry_price} for the signal engine."""
        return {cid: p.entry_price for cid, p in self._positions.items()}

    def get_position_sizes(self) -> dict[str, float]:
        """Return {contract_id: size_usd} for the risk manager."""
        return {cid: p.qty * p.entry_price for cid, p in self._positions.items()}

    def get_uptime(self) -> str:
        delta = datetime.now(timezone.utc) - self._start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours}h {minutes}m"

    def get_stats(self) -> dict[str, Any]:
        return {
            "portfolio_value": round(self.get_portfolio_value(), 2),
            "cash": round(self._cash, 2),
            "daily_pnl": round(self.get_daily_pnl(), 2),
            "drawdown": round(self.get_drawdown(), 4),
            "open_positions": len(self._positions),
            "win_rate": round(self.get_win_rate(), 4),
            "resolution_accuracy": round(self.get_resolution_accuracy(), 4),
            "uptime": self.get_uptime(),
            "total_trades": len(self._closed),
        }

    # ------------------------------------------------------------------
    # Trade processing
    # ------------------------------------------------------------------

    def process_trade(self, result: TradeResult) -> None:
        """Update portfolio state after a trade execution."""
        if result.status == "CANCELLED":
            return

        if result.side == "BUY":
            self._process_buy(result)
        elif result.side == "SELL":
            self._process_sell(result)

    def _process_buy(self, result: TradeResult) -> None:
        qty = result.size_usd / result.price if result.price > 0 else 0
        cost = result.size_usd

        if self._cash < cost:
            logger.warning("Insufficient cash: $%.2f < $%.2f", self._cash, cost)
            cost = self._cash
            qty = cost / result.price if result.price > 0 else 0

        self._cash -= cost

        existing = self._positions.get(result.contract_id)
        if existing:
            total_qty = existing.qty + qty
            avg_price = (
                (existing.qty * existing.entry_price + qty * result.price) / total_qty
                if total_qty > 0 else result.price
            )
            existing.qty = total_qty
            existing.entry_price = avg_price
        else:
            self._positions[result.contract_id] = Position(
                contract_id=result.contract_id,
                location=result.location,
                target_date=result.target_date,
                bucket_label=result.bucket_label,
                side="YES",
                qty=qty,
                entry_price=result.price,
                entry_time=result.timestamp,
                current_price=result.price,
                unrealized_pnl=0.0,
                mode=result.mode,
            )

        logger.info(
            "Portfolio BUY: %s %s | qty=%.2f @ $%.4f | cash=$%.2f",
            result.location, result.bucket_label, qty, result.price, self._cash,
        )

    def _process_sell(self, result: TradeResult) -> None:
        pos = self._positions.pop(result.contract_id, None)
        if not pos:
            logger.warning("Sell for unknown position: %s", result.contract_id)
            return

        proceeds = pos.qty * result.price
        cost_basis = pos.qty * pos.entry_price
        pnl = proceeds - cost_basis
        self._cash += proceeds

        self._closed.append(ClosedPosition(
            contract_id=result.contract_id,
            location=result.location,
            target_date=result.target_date,
            bucket_label=result.bucket_label,
            entry_price=pos.entry_price,
            exit_price=result.price,
            size_usd=cost_basis,
            realized_pnl=pnl,
            resolution="SOLD",
            closed_at=result.timestamp,
        ))

        logger.info(
            "Portfolio SELL: %s %s | pnl=$%.2f | cash=$%.2f",
            result.location, result.bucket_label, pnl, self._cash,
        )

    # ------------------------------------------------------------------
    # Mark-to-market + resolution
    # ------------------------------------------------------------------

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update current prices for all open positions."""
        for cid, pos in self._positions.items():
            if cid in prices:
                pos.current_price = prices[cid]
                cost_basis = pos.qty * pos.entry_price
                current_val = pos.qty * pos.current_price
                pos.unrealized_pnl = current_val - cost_basis

    def resolve_contract(self, contract_id: str, won: bool) -> float:
        """Handle contract resolution. Returns realized P&L."""
        pos = self._positions.pop(contract_id, None)
        if not pos:
            return 0.0

        payout = pos.qty * 1.0 if won else 0.0
        cost_basis = pos.qty * pos.entry_price
        pnl = payout - cost_basis
        self._cash += payout

        self._closed.append(ClosedPosition(
            contract_id=contract_id,
            location=pos.location,
            target_date=pos.target_date,
            bucket_label=pos.bucket_label,
            entry_price=pos.entry_price,
            exit_price=1.0 if won else 0.0,
            size_usd=cost_basis,
            realized_pnl=pnl,
            resolution="WON" if won else "LOST",
            closed_at=datetime.now(timezone.utc),
        ))

        logger.info(
            "Resolution %s: %s %s | pnl=$%.2f",
            "WON" if won else "LOST", pos.location, pos.bucket_label, pnl,
        )
        return pnl

    def reset_daily(self) -> None:
        """Reset daily tracking at start of new day."""
        self._daily_start_value = self.get_portfolio_value()
