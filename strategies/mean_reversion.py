"""Mean reversion strategy — fade post-forecast-update overshoots.

Hypothesis: After a NOAA forecast update, Polymarket odds overshoot in
the direction of the forecast change then revert toward fair value as
the market digests the information.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Fade post-update overshoots when Z-score is extreme."""

    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        post_update_window_min: int = 30,
        lookback: int = 50,
        min_reversion_rate: float = 0.60,
        max_holding_periods: int = 20,
    ) -> None:
        self._zscore_entry = zscore_entry
        self._zscore_exit = zscore_exit
        self._post_update_window_min = post_update_window_min
        self._lookback = lookback
        self._min_reversion_rate = min_reversion_rate
        self._max_holding_periods = max_holding_periods

        # Internal state
        self._spread_history: dict[str, deque[float]] = {}
        self._last_update_times: dict[str, datetime] = {}
        self._reversion_counts: dict[str, tuple[int, int]] = {}  # (reverted, total)

    @property
    def name(self) -> str:
        return "mean_reversion"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Generate a mean-reversion signal if post-update overshoot detected.

        Additional market_data keys:
            forecast_update_time: datetime of most recent NOAA update
            forecast_shift_f: magnitude of forecast change
        """
        market_id = market_data.get("market_id", "")
        city = market_data.get("city", "")
        market_price = market_data.get("market_price", 0.0)
        model_prob = market_data.get("model_prob", 0.0)
        timestamp = market_data.get("timestamp", datetime.now(timezone.utc))
        update_time = market_data.get("forecast_update_time")
        shift = abs(market_data.get("forecast_shift_f", 0.0))

        if market_price <= 0 or model_prob <= 0:
            return None

        spread = market_price - model_prob
        key = f"{city}_{market_id}"

        # Track spread history
        if key not in self._spread_history:
            self._spread_history[key] = deque(maxlen=self._lookback)
        self._spread_history[key].append(spread)

        history = self._spread_history[key]
        if len(history) < 10:
            return None

        # Compute Z-score
        arr = np.array(history)
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-8:
            return None
        zscore = (spread - mean) / std

        # Check if we're within post-update window
        in_window = False
        if update_time and shift > 0.5:
            if isinstance(update_time, str):
                try:
                    update_time = datetime.fromisoformat(update_time)
                except ValueError:
                    update_time = None
            if update_time:
                elapsed = (timestamp - update_time).total_seconds() / 60.0
                in_window = 0 < elapsed <= self._post_update_window_min
                # Track for reversion rate
                prev_update = self._last_update_times.get(key)
                if prev_update and prev_update != update_time:
                    self._last_update_times[key] = update_time
                elif not prev_update:
                    self._last_update_times[key] = update_time

        # Check historical reversion rate
        rev, total = self._reversion_counts.get(key, (0, 0))
        reversion_rate = rev / total if total > 5 else self._min_reversion_rate

        # Filter: skip if second update during window
        if market_data.get("second_update_during_hold", False):
            return None

        # SELL signal if existing position and Z-score reverted
        if market_data.get("has_position"):
            if abs(zscore) <= self._zscore_exit:
                # Record reversion success
                self._reversion_counts[key] = (rev + 1, total + 1)
                return Signal(
                    direction="SELL",
                    strength=0.7,
                    edge=abs(spread),
                    strategy_name=self.name,
                    market_id=market_id,
                    city=city,
                    metadata={"exit_reason": "zscore_reverted", "zscore": zscore},
                )
            holding = market_data.get("holding_periods", 0)
            if holding >= self._max_holding_periods:
                self._reversion_counts[key] = (rev, total + 1)
                return Signal(
                    direction="SELL",
                    strength=0.5,
                    edge=abs(spread),
                    strategy_name=self.name,
                    market_id=market_id,
                    city=city,
                    metadata={"exit_reason": "max_holding", "zscore": zscore},
                )
            return None

        # BUY entry: Z-score extreme, within window, reversion rate ok
        if not in_window:
            return None
        if reversion_rate < self._min_reversion_rate and total > 5:
            return None

        if zscore > self._zscore_entry:
            # Market overshooting high — fade by selling / shorting
            return Signal(
                direction="SELL",
                strength=min(abs(zscore) / 4.0, 1.0),
                edge=abs(spread),
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={
                    "zscore": zscore,
                    "spread": spread,
                    "reversion_rate": reversion_rate,
                    "post_update_elapsed_min": (timestamp - update_time).total_seconds() / 60 if update_time else 0,
                },
            )
        elif zscore < -self._zscore_entry:
            # Market overshooting low — buy
            return Signal(
                direction="BUY",
                strength=min(abs(zscore) / 4.0, 1.0),
                edge=abs(spread),
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={
                    "zscore": zscore,
                    "spread": spread,
                    "reversion_rate": reversion_rate,
                },
            )

        return None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "zscore_entry": self._zscore_entry,
            "zscore_exit": self._zscore_exit,
            "post_update_window_min": self._post_update_window_min,
            "lookback": self._lookback,
            "min_reversion_rate": self._min_reversion_rate,
            "max_holding_periods": self._max_holding_periods,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            if hasattr(self, f"_{k}"):
                setattr(self, f"_{k}", v)

    def reset(self) -> None:
        self._spread_history.clear()
        self._last_update_times.clear()
        self._reversion_counts.clear()
