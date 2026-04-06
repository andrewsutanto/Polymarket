"""Volume-price divergence strategy.

Detects when volume spikes without proportional price movement,
indicating informed trading that hasn't moved the price yet.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class VolumeDivergenceStrategy(BaseStrategy):
    """Trade when volume spikes diverge from price movement."""

    def __init__(
        self,
        volume_zscore_threshold: float = 2.5,
        price_move_threshold: float = 0.03,
        lookback: int = 50,
        min_volume: float = 100.0,
    ) -> None:
        self._vol_zscore = volume_zscore_threshold
        self._price_threshold = price_move_threshold
        self._lookback = lookback
        self._min_volume = min_volume

        # Rolling history per market
        self._volume_history: dict[str, deque[float]] = {}
        self._price_history: dict[str, deque[float]] = {}

    @property
    def name(self) -> str:
        return "volume_divergence"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Detect volume spike without proportional price move.

        When volume is anomalously high but price barely moved, it suggests
        informed participants are accumulating. Trade in the direction
        of the price drift (even if small).
        """
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")
        volume = market_data.get("volume_24h", 0.0)
        mid_price = market_data.get("mid_price", 0.0)

        if volume < self._min_volume or mid_price <= 0:
            return None

        key = f"{market_id}_{outcome}"

        # Track history
        if key not in self._volume_history:
            self._volume_history[key] = deque(maxlen=self._lookback)
            self._price_history[key] = deque(maxlen=self._lookback)

        self._volume_history[key].append(volume)
        self._price_history[key].append(mid_price)

        vol_hist = self._volume_history[key]
        price_hist = self._price_history[key]

        if len(vol_hist) < 10:
            return None

        # Volume Z-score
        vol_arr = np.array(vol_hist)
        vol_mean = np.mean(vol_arr)
        vol_std = np.std(vol_arr)
        if vol_std < 1e-8:
            return None
        vol_zscore = (volume - vol_mean) / vol_std

        # Price change over last N bars
        price_arr = np.array(price_hist)
        if len(price_arr) >= 5:
            recent_price_change = price_arr[-1] - price_arr[-5]
        else:
            recent_price_change = 0.0

        # Divergence: volume spiked but price barely moved
        if vol_zscore < self._vol_zscore:
            return None
        if abs(recent_price_change) > self._price_threshold:
            return None  # Price already moved — no divergence

        # Direction: follow the small drift
        if recent_price_change > 0.001:
            direction = "BUY"
        elif recent_price_change < -0.001:
            direction = "SELL"
        else:
            # No drift at all — skip (can't determine direction)
            return None

        edge = vol_zscore * 0.01  # Heuristic: higher Z-score = more edge

        # Exit existing position if volume normalizes
        if market_data.get("has_position"):
            if vol_zscore < 1.0:
                return Signal(
                    direction="SELL",
                    strength=0.5,
                    edge=edge,
                    strategy_name=self.name,
                    market_id=market_id,
                    token_id=token_id,
                    market_slug=slug,
                    category=category,
                    outcome=outcome,
                    metadata={"exit_reason": "volume_normalized", "vol_zscore": vol_zscore},
                )
            return None

        return Signal(
            direction=direction,
            strength=min(vol_zscore / 5.0, 1.0),
            edge=edge,
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id,
            market_slug=slug,
            category=category,
            outcome=outcome,
            metadata={
                "vol_zscore": vol_zscore,
                "volume": volume,
                "vol_mean": vol_mean,
                "price_drift": recent_price_change,
            },
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "volume_zscore_threshold": self._vol_zscore,
            "price_move_threshold": self._price_threshold,
            "lookback": self._lookback,
            "min_volume": self._min_volume,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "volume_zscore_threshold" in params:
            self._vol_zscore = params["volume_zscore_threshold"]
        if "price_move_threshold" in params:
            self._price_threshold = params["price_move_threshold"]
        if "lookback" in params:
            self._lookback = params["lookback"]
        if "min_volume" in params:
            self._min_volume = params["min_volume"]

    def reset(self) -> None:
        self._volume_history.clear()
        self._price_history.clear()
