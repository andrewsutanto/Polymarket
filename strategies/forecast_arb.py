"""Forecast arbitrage strategy — refactored from core/signal_engine.py.

Compares NOAA model-derived fair probability (normal CDF over temperature
buckets) against Polymarket CLOB odds. When the spread exceeds a threshold,
generates a trade signal. Wraps existing core logic without duplicating it.
"""

from __future__ import annotations

import logging
from typing import Any

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class ForecastArbStrategy(BaseStrategy):
    """Forecast-vs-odds arbitrage. The original strategy, now as a plugin."""

    def __init__(
        self,
        entry_threshold: float = 0.15,
        exit_threshold: float = 0.45,
        min_value_ratio: float = 3.0,
        min_edge: float = 0.10,
        min_confidence: float = 0.85,
    ) -> None:
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold
        self._min_value_ratio = min_value_ratio
        self._min_edge = min_edge
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "forecast_arb"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Detect mispricing between model probability and market price.

        Required market_data keys:
            market_price, model_prob, market_id, city, confidence,
            forecast_high_f, book_depth, spread, lead_days.
        """
        market_price = market_data.get("market_price", 0.0)
        model_prob = market_data.get("model_prob", 0.0)
        confidence = market_data.get("confidence", 0.0)
        market_id = market_data.get("market_id", "")
        city = market_data.get("city", "")

        if market_price <= 0 or model_prob <= 0:
            return None

        edge = model_prob - market_price
        value_ratio = model_prob / market_price

        # BUY signal
        if (
            market_price < self._entry_threshold
            and value_ratio >= self._min_value_ratio
            and edge >= self._min_edge
            and confidence >= self._min_confidence
        ):
            return Signal(
                direction="BUY",
                strength=min(confidence, 1.0),
                edge=edge,
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={
                    "model_prob": model_prob,
                    "market_price": market_price,
                    "value_ratio": value_ratio,
                    "forecast_high_f": market_data.get("forecast_high_f", 0),
                    "lead_days": market_data.get("lead_days", 0),
                },
            )

        # SELL signal (for existing positions)
        if market_data.get("has_position") and market_price >= self._exit_threshold:
            return Signal(
                direction="SELL",
                strength=0.8,
                edge=market_price - model_prob,
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={"exit_reason": "price_above_exit_threshold"},
            )

        return None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "entry_threshold": self._entry_threshold,
            "exit_threshold": self._exit_threshold,
            "min_value_ratio": self._min_value_ratio,
            "min_edge": self._min_edge,
            "min_confidence": self._min_confidence,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "entry_threshold" in params:
            self._entry_threshold = params["entry_threshold"]
        if "exit_threshold" in params:
            self._exit_threshold = params["exit_threshold"]
        if "min_value_ratio" in params:
            self._min_value_ratio = params["min_value_ratio"]
        if "min_edge" in params:
            self._min_edge = params["min_edge"]
        if "min_confidence" in params:
            self._min_confidence = params["min_confidence"]
