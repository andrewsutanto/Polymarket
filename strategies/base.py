"""Abstract strategy interface for the multi-strategy framework.

All strategies conform to this interface so the backtester, ensemble,
and execution layer treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Signal:
    """Trading signal emitted by a strategy."""

    direction: str  # "BUY" or "SELL"
    strength: float  # 0.0–1.0 raw signal confidence
    edge: float  # estimated edge in percentage points
    strategy_name: str
    market_id: str  # Polymarket contract/token identifier
    city: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all weather arbitrage strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @abstractmethod
    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze current market state and return a Signal or None.

        Args:
            market_data: Dict containing at minimum:
                - timestamp: current evaluation time
                - city: target city key
                - market_id: Polymarket contract id
                - market_price: current CLOB mid price (0-1)
                - model_prob: model-derived fair probability (0-1)
                - forecast_high_f: NOAA forecasted high
                - forecast_shift_f: change since last model run
                - book_depth: USD depth at top of book
                - spread: bid-ask spread
                - volume_24h: 24h volume
                - lead_days: days until resolution
                Plus strategy-specific fields documented per implementation.

        Returns:
            Signal or None if no trade opportunity detected.
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return current parameter values for logging and optimization."""
        ...

    @abstractmethod
    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update parameters. Used by optimizer during backtesting."""
        ...

    def reset(self) -> None:
        """Reset any internal state. Called between backtest windows."""
        pass
