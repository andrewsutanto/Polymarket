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
    market_id: str  # Polymarket condition_id
    token_id: str  # Polymarket token_id for the specific outcome
    market_slug: str  # Human-readable market identifier
    category: str  # "politics", "sports", "crypto", "macro", "other"
    outcome: str  # The outcome being traded (e.g., "Yes", "Trump", etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all arbitrage strategies."""

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
                - market_id: Polymarket condition_id
                - token_id: token for the specific outcome
                - market_slug: human-readable slug
                - category: market category
                - question: full market question text
                - outcome: outcome label
                - yes_price: current YES price (0-1)
                - no_price: current NO price (0-1)
                - mid_price: (yes_price + no_price) / 2 or best estimate
                - spread: bid-ask spread
                - book_depth: USD depth at top of book
                - volume_24h: 24h volume in USD
                - liquidity: total liquidity
                - time_to_resolution_hrs: hours until expiry
                - has_position: whether we hold this token
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
