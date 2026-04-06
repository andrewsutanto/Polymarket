"""Forecast momentum strategy — ride consistent forecast drift.

Hypothesis: When NOAA forecasts for a city drift consistently in one
direction across multiple update cycles, Polymarket odds adjust with a
lag. There is momentum in the odds adjustment process.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class ForecastMomentumStrategy(BaseStrategy):
    """Trade momentum when forecasts drift consistently in one direction."""

    def __init__(
        self,
        n_updates: int = 5,
        drift_threshold: float = 0.6,
        entry_edge: float = 0.05,
        min_consistent_updates: int = 3,
        min_ttl_hours: float = 2.0,
    ) -> None:
        self._n_updates = n_updates
        self._drift_threshold = drift_threshold
        self._entry_edge = entry_edge
        self._min_consistent = min_consistent_updates
        self._min_ttl_hours = min_ttl_hours

        # Internal: track forecast changes per city/date
        self._forecast_changes: dict[str, deque[float]] = {}

    @property
    def name(self) -> str:
        return "forecast_momentum"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Generate a momentum signal if forecast drifts consistently.

        Additional market_data keys:
            forecast_shift_f: change in forecast high since last update
            time_to_resolution_hrs: hours until contract resolves
        """
        market_id = market_data.get("market_id", "")
        city = market_data.get("city", "")
        market_price = market_data.get("market_price", 0.0)
        model_prob = market_data.get("model_prob", 0.0)
        shift = market_data.get("forecast_shift_f", 0.0)
        ttl = market_data.get("time_to_resolution_hrs", 24.0)

        if market_price <= 0 or model_prob <= 0:
            return None

        key = f"{city}_{market_id}"

        # Record forecast change
        if key not in self._forecast_changes:
            self._forecast_changes[key] = deque(maxlen=self._n_updates)
        if abs(shift) > 0.01:
            self._forecast_changes[key].append(shift)

        changes = self._forecast_changes[key]
        if len(changes) < self._min_consistent:
            return None

        # Compute drift score: sign-consistency weighted by magnitude
        signs = np.sign(list(changes))
        magnitudes = np.abs(list(changes))
        if np.sum(magnitudes) < 1e-8:
            return None

        consistent_dir = np.sign(np.sum(signs))
        consistency = abs(np.sum(signs)) / len(signs)
        weighted_drift = np.sum(signs * magnitudes) / np.sum(magnitudes)
        drift_score = consistency * abs(weighted_drift)

        # SELL exit if drift decays
        if market_data.get("has_position"):
            if drift_score < self._drift_threshold * 0.5:
                return Signal(
                    direction="SELL",
                    strength=0.6,
                    edge=abs(model_prob - market_price),
                    strategy_name=self.name,
                    market_id=market_id,
                    city=city,
                    metadata={"exit_reason": "drift_decayed", "drift_score": drift_score},
                )
            edge = model_prob - market_price
            if abs(edge) < self._entry_edge * 0.3:
                return Signal(
                    direction="SELL",
                    strength=0.5,
                    edge=abs(edge),
                    strategy_name=self.name,
                    market_id=market_id,
                    city=city,
                    metadata={"exit_reason": "spread_closed", "drift_score": drift_score},
                )
            return None

        # Entry filters
        if drift_score < self._drift_threshold:
            return None
        if ttl < self._min_ttl_hours:
            return None

        edge = model_prob - market_price
        if consistent_dir > 0 and edge >= self._entry_edge:
            return Signal(
                direction="BUY",
                strength=min(drift_score, 1.0),
                edge=edge,
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={
                    "drift_score": drift_score,
                    "consistency": consistency,
                    "n_updates": len(changes),
                    "cumulative_shift": sum(changes),
                },
            )
        elif consistent_dir < 0 and edge <= -self._entry_edge:
            return Signal(
                direction="SELL",
                strength=min(drift_score, 1.0),
                edge=abs(edge),
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={
                    "drift_score": drift_score,
                    "consistency": consistency,
                    "n_updates": len(changes),
                    "cumulative_shift": sum(changes),
                },
            )

        return None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "n_updates": self._n_updates,
            "drift_threshold": self._drift_threshold,
            "entry_edge": self._entry_edge,
            "min_consistent_updates": self._min_consistent,
            "min_ttl_hours": self._min_ttl_hours,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "n_updates" in params:
            self._n_updates = params["n_updates"]
        if "drift_threshold" in params:
            self._drift_threshold = params["drift_threshold"]
        if "entry_edge" in params:
            self._entry_edge = params["entry_edge"]
        if "min_consistent_updates" in params:
            self._min_consistent = params["min_consistent_updates"]
        if "min_ttl_hours" in params:
            self._min_ttl_hours = params["min_ttl_hours"]

    def reset(self) -> None:
        self._forecast_changes.clear()
