"""Rule-based weather regime classifier.

Classifies current market conditions into one of five regimes to inform
strategy weight allocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from regime.features import RegimeFeatures, RegimeFeatureSet

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    ACTIVE_WEATHER = "ACTIVE_WEATHER"
    STABLE = "STABLE"
    CONSENSUS = "CONSENSUS"
    DISAGREEMENT = "DISAGREEMENT"
    NEUTRAL = "NEUTRAL"


# Default strategy weights per regime
DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    Regime.ACTIVE_WEATHER: {
        "forecast_arb": 0.4, "forecast_momentum": 0.3,
        "mean_reversion": 0.2, "cross_city_arb": 0.1,
    },
    Regime.STABLE: {
        "cross_city_arb": 0.4, "mean_reversion": 0.3,
        "forecast_arb": 0.2, "forecast_momentum": 0.1,
    },
    Regime.CONSENSUS: {
        "cross_city_arb": 0.5, "mean_reversion": 0.3,
        "forecast_arb": 0.1, "forecast_momentum": 0.1,
    },
    Regime.DISAGREEMENT: {
        "forecast_arb": 0.5, "forecast_momentum": 0.3,
        "mean_reversion": 0.1, "cross_city_arb": 0.1,
    },
    Regime.NEUTRAL: {
        "forecast_arb": 0.25, "forecast_momentum": 0.25,
        "mean_reversion": 0.25, "cross_city_arb": 0.25,
    },
}


@dataclass
class RegimeTransition:
    timestamp: datetime
    old_regime: Regime
    new_regime: Regime
    features: RegimeFeatureSet


class RegimeDetector:
    """Classify current conditions into a weather/market regime."""

    def __init__(
        self,
        eval_every_n: int = 20,
        forecast_vol_high_pct: float = 0.75,
        forecast_vol_low_pct: float = 0.25,
        market_vol_low_pct: float = 0.25,
        update_freq_high: float = 1.5,
        agreement_high: float = 0.8,
        agreement_low: float = 0.4,
    ) -> None:
        self._eval_every = eval_every_n
        self._fc_vol_high = forecast_vol_high_pct
        self._fc_vol_low = forecast_vol_low_pct
        self._mkt_vol_low = market_vol_low_pct
        self._update_freq_high = update_freq_high
        self._agree_high = agreement_high
        self._agree_low = agreement_low

        self._features = RegimeFeatures()
        self._current = Regime.NEUTRAL
        self._step_count = 0
        self._history: list[RegimeTransition] = []

        # For percentile calculations
        self._fc_vol_history: list[float] = []
        self._mkt_vol_history: list[float] = []

    @property
    def current_regime(self) -> Regime:
        return self._current

    @property
    def transitions(self) -> list[RegimeTransition]:
        return list(self._history)

    @property
    def features(self) -> RegimeFeatures:
        return self._features

    def update(self, market_data: dict[str, Any]) -> Regime:
        """Ingest data and optionally re-evaluate regime.

        Args:
            market_data: Passed through to RegimeFeatures.update().

        Returns:
            Current regime.
        """
        self._features.update(market_data)
        self._step_count += 1

        if self._step_count % self._eval_every == 0:
            self._classify()
            self._features.tick_period()

        return self._current

    def force_classify(self) -> Regime:
        """Force immediate re-evaluation."""
        self._classify()
        return self._current

    def _classify(self) -> None:
        feat = self._features.compute()

        # Track history for percentiles
        self._fc_vol_history.append(feat.forecast_volatility)
        self._mkt_vol_history.append(feat.market_volatility)

        fc_pct = self._percentile(feat.forecast_volatility, self._fc_vol_history)
        mkt_pct = self._percentile(feat.market_volatility, self._mkt_vol_history)

        old = self._current

        # Rule-based classification
        if fc_pct > self._fc_vol_high and feat.update_frequency_ratio > self._update_freq_high:
            new = Regime.ACTIVE_WEATHER
        elif fc_pct < self._fc_vol_low and mkt_pct < self._mkt_vol_low:
            new = Regime.STABLE
        elif feat.forecast_agreement > self._agree_high:
            new = Regime.CONSENSUS
        elif feat.forecast_agreement < self._agree_low:
            new = Regime.DISAGREEMENT
        else:
            new = Regime.NEUTRAL

        if new != old:
            transition = RegimeTransition(
                timestamp=datetime.now(timezone.utc),
                old_regime=old,
                new_regime=new,
                features=feat,
            )
            self._history.append(transition)
            logger.info("Regime transition: %s -> %s", old.value, new.value)

        self._current = new

    @staticmethod
    def _percentile(value: float, history: list[float]) -> float:
        if len(history) < 5:
            return 0.5
        import numpy as np
        arr = np.array(history)
        return float(np.sum(arr <= value) / len(arr))

    def get_strategy_weights(
        self, custom_mapping: dict[str, dict[str, float]] | None = None
    ) -> dict[str, float]:
        """Return strategy weights for the current regime."""
        mapping = custom_mapping or DEFAULT_REGIME_WEIGHTS
        return mapping.get(self._current, mapping.get(Regime.NEUTRAL, {}))

    def reset(self) -> None:
        self._current = Regime.NEUTRAL
        self._step_count = 0
        self._history.clear()
        self._features.reset()
        self._fc_vol_history.clear()
        self._mkt_vol_history.clear()
