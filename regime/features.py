"""Regime feature engineering from weather and market data.

Computes rolling features used by the regime classifier to determine
current market conditions.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeFeatureSet:
    """Feature vector for regime classification."""

    forecast_volatility: float  # Std of forecast changes over last N updates
    market_volatility: float  # Realized vol of odds changes
    forecast_agreement: float  # Correlation between model fair prob and market prob
    update_frequency_ratio: float  # Recent update count vs 30-day average
    spread_autocorrelation: float  # Autocorr of model-vs-market spread at lag 1
    cross_city_dispersion: float  # Std of spreads across cities


class RegimeFeatures:
    """Rolling feature computation for regime detection."""

    def __init__(
        self,
        forecast_lookback: int = 10,
        market_lookback: int = 20,
        update_window_hours: float = 6.0,
        update_avg_days: int = 30,
    ) -> None:
        self._fc_lookback = forecast_lookback
        self._mkt_lookback = market_lookback
        self._update_window_hrs = update_window_hours
        self._update_avg_days = update_avg_days

        # Rolling buffers
        self._forecast_changes: dict[str, deque[float]] = {}
        self._market_changes: deque[float] = deque(maxlen=market_lookback)
        self._model_probs: deque[float] = deque(maxlen=market_lookback)
        self._market_probs: deque[float] = deque(maxlen=market_lookback)
        self._spreads: deque[float] = deque(maxlen=market_lookback)
        self._city_spreads: dict[str, float] = {}
        self._update_counts: deque[float] = deque(maxlen=update_avg_days * 4)
        self._recent_updates: int = 0

    def update(self, market_data: dict[str, Any]) -> None:
        """Ingest a new data point and update all rolling features.

        Args:
            market_data: Dict with keys: city, model_prob, market_price,
                forecast_shift_f, is_forecast_update.
        """
        city = market_data.get("city", "unknown")
        model_prob = market_data.get("model_prob", 0.5)
        market_price = market_data.get("market_price", 0.5)
        shift = market_data.get("forecast_shift_f", 0.0)
        is_update = market_data.get("is_forecast_update", False)

        # Forecast changes per city
        if city not in self._forecast_changes:
            self._forecast_changes[city] = deque(maxlen=self._fc_lookback)
        if abs(shift) > 0.001:
            self._forecast_changes[city].append(shift)

        # Market changes
        if self._market_probs:
            prev = self._market_probs[-1]
            self._market_changes.append(market_price - prev)

        self._model_probs.append(model_prob)
        self._market_probs.append(market_price)

        # Spread
        spread = model_prob - market_price
        self._spreads.append(spread)
        self._city_spreads[city] = spread

        # Update frequency
        if is_update:
            self._recent_updates += 1

    def compute(self) -> RegimeFeatureSet:
        """Compute the current feature set from accumulated data."""
        return RegimeFeatureSet(
            forecast_volatility=self._calc_forecast_vol(),
            market_volatility=self._calc_market_vol(),
            forecast_agreement=self._calc_agreement(),
            update_frequency_ratio=self._calc_update_freq_ratio(),
            spread_autocorrelation=self._calc_spread_autocorr(),
            cross_city_dispersion=self._calc_city_dispersion(),
        )

    def _calc_forecast_vol(self) -> float:
        all_changes: list[float] = []
        for changes in self._forecast_changes.values():
            all_changes.extend(changes)
        if len(all_changes) < 3:
            return 0.0
        return float(np.std(all_changes))

    def _calc_market_vol(self) -> float:
        if len(self._market_changes) < 5:
            return 0.0
        return float(np.std(list(self._market_changes)))

    def _calc_agreement(self) -> float:
        if len(self._model_probs) < 5:
            return 0.5
        m = np.array(list(self._model_probs))
        p = np.array(list(self._market_probs))
        if np.std(m) < 1e-8 or np.std(p) < 1e-8:
            return 0.5
        return float(np.corrcoef(m, p)[0, 1])

    def _calc_update_freq_ratio(self) -> float:
        if not self._update_counts:
            return 1.0
        avg = np.mean(list(self._update_counts)) if self._update_counts else 1.0
        if avg < 0.1:
            return 1.0
        return self._recent_updates / avg

    def _calc_spread_autocorr(self) -> float:
        s = list(self._spreads)
        if len(s) < 5:
            return 0.0
        arr = np.array(s)
        if np.std(arr) < 1e-8:
            return 0.0
        lag1 = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        return float(lag1) if np.isfinite(lag1) else 0.0

    def _calc_city_dispersion(self) -> float:
        if len(self._city_spreads) < 2:
            return 0.0
        return float(np.std(list(self._city_spreads.values())))

    def tick_period(self) -> None:
        """Call at the end of each period to rotate update counts."""
        self._update_counts.append(float(self._recent_updates))
        self._recent_updates = 0

    def reset(self) -> None:
        """Reset all state."""
        self._forecast_changes.clear()
        self._market_changes.clear()
        self._model_probs.clear()
        self._market_probs.clear()
        self._spreads.clear()
        self._city_spreads.clear()
        self._update_counts.clear()
        self._recent_updates = 0
