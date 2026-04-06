"""Stale market detection strategy.

Detects markets where price hasn't moved meaningfully despite
approaching resolution. As expiry nears, prices should converge
toward 0 or 1 — if they haven't, there may be mispricing.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class StaleMarketStrategy(BaseStrategy):
    """Trade markets that are stale near expiry."""

    def __init__(
        self,
        stale_hours: int = 48,
        max_ttl_hours: float = 168.0,
        min_distance_from_boundary: float = 0.15,
        convergence_threshold: float = 0.85,
        min_lookback: int = 20,
    ) -> None:
        self._stale_hours = stale_hours
        self._max_ttl = max_ttl_hours
        self._min_distance = min_distance_from_boundary
        self._convergence_threshold = convergence_threshold
        self._min_lookback = min_lookback

        # Track price history per market
        self._price_history: dict[str, deque[float]] = {}
        self._first_seen: dict[str, int] = {}
        self._tick_count: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "stale_market"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Detect stale markets near expiry.

        A market is "stale" if:
        1. Price hasn't moved > min_distance from its range midpoint
        2. It's within max_ttl_hours of resolution
        3. Price is far from 0 or 1 (hasn't converged)

        Trade toward the nearer boundary (0 or 1).
        """
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")
        mid_price = market_data.get("mid_price", 0.0)
        ttl = market_data.get("time_to_resolution_hrs", 9999.0)

        if mid_price <= 0 or mid_price >= 1:
            return None
        if ttl > self._max_ttl or ttl <= 0:
            return None

        key = f"{market_id}_{outcome}"

        if key not in self._price_history:
            self._price_history[key] = deque(maxlen=200)
            self._tick_count[key] = 0

        self._price_history[key].append(mid_price)
        self._tick_count[key] += 1

        history = self._price_history[key]
        if len(history) < self._min_lookback:
            return None

        # Check staleness: price range over lookback
        arr = np.array(history)
        price_range = float(np.max(arr) - np.min(arr))
        price_std = float(np.std(arr))

        # Market is stale if price barely moved
        is_stale = price_range < 0.05 and price_std < 0.02

        if not is_stale:
            return None

        # Check distance from boundaries
        dist_from_zero = mid_price
        dist_from_one = 1.0 - mid_price

        if min(dist_from_zero, dist_from_one) < self._min_distance:
            return None  # Already near a boundary

        # As expiry approaches, price should converge
        # The closer to expiry, the more edge there is in trading toward a boundary
        urgency = max(0, 1.0 - ttl / self._max_ttl)

        # Exit if holding and price starts moving
        if market_data.get("has_position"):
            if price_range > 0.10 or price_std > 0.04:
                return Signal(
                    direction="SELL",
                    strength=0.6,
                    edge=price_range,
                    strategy_name=self.name,
                    market_id=market_id,
                    token_id=token_id,
                    market_slug=slug,
                    category=category,
                    outcome=outcome,
                    metadata={"exit_reason": "price_moving", "price_range": price_range},
                )
            return None

        # Trade toward the nearer boundary
        if mid_price > 0.5:
            # More likely to resolve YES — buy YES
            direction = "BUY"
            edge = (mid_price - 0.5) * urgency * 0.3
        else:
            # Price below 0.5, but stale — could go either way
            # In stale markets, the status quo tends to hold
            # If price is 0.3 and stale, "No" is more likely
            direction = "SELL"
            edge = (0.5 - mid_price) * urgency * 0.3

        if edge < 0.02:
            return None

        return Signal(
            direction=direction,
            strength=min(urgency * 0.8, 1.0),
            edge=edge,
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id,
            market_slug=slug,
            category=category,
            outcome=outcome,
            metadata={
                "price_range": price_range,
                "price_std": price_std,
                "ttl_hours": ttl,
                "urgency": urgency,
                "dist_from_zero": dist_from_zero,
                "dist_from_one": dist_from_one,
            },
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "stale_hours": self._stale_hours,
            "max_ttl_hours": self._max_ttl,
            "min_distance_from_boundary": self._min_distance,
            "convergence_threshold": self._convergence_threshold,
            "min_lookback": self._min_lookback,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            if hasattr(self, f"_{k}"):
                setattr(self, f"_{k}", v)

    def reset(self) -> None:
        self._price_history.clear()
        self._tick_count.clear()
