"""Cross-city statistical arbitrage strategy.

Hypothesis: Weather contracts for geographically correlated cities should
maintain a stable probability relationship. When the spread between their
implied probabilities deviates from the historical norm, trade the spread.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

DEFAULT_PAIRS = [
    ("NYC", "Chicago"),
    ("NYC", "Atlanta"),
    ("Chicago", "Dallas"),
    ("Atlanta", "Dallas"),
    ("Seattle", "Chicago"),
]


class CrossCityArbStrategy(BaseStrategy):
    """Trade divergences between correlated city pairs."""

    def __init__(
        self,
        city_pairs: list[tuple[str, str]] | None = None,
        lookback: int = 100,
        entry_sigma: float = 2.0,
        exit_sigma: float = 0.5,
        min_coint_pvalue: float = 0.05,
        forecast_diverge_override: float = 5.0,
    ) -> None:
        self._city_pairs = city_pairs or DEFAULT_PAIRS
        self._lookback = lookback
        self._entry_sigma = entry_sigma
        self._exit_sigma = exit_sigma
        self._min_coint_pvalue = min_coint_pvalue
        self._forecast_diverge_override = forecast_diverge_override

        # Spread history per pair
        self._spread_history: dict[str, deque[float]] = {}
        self._coint_valid: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "cross_city_arb"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Generate cross-city arb signal from spread deviation.

        Additional market_data keys:
            city_prices: dict[str, float] — {city: market_price}
            city_probs: dict[str, float] — {city: model_prob}
            city_forecasts: dict[str, float] — {city: forecast_high_f}
        """
        city_prices = market_data.get("city_prices", {})
        city_probs = market_data.get("city_probs", {})
        city_forecasts = market_data.get("city_forecasts", {})
        market_id = market_data.get("market_id", "")

        best_signal: Signal | None = None
        best_zscore = 0.0

        for city_a, city_b in self._city_pairs:
            if city_a not in city_prices or city_b not in city_prices:
                continue

            price_a = city_prices[city_a]
            price_b = city_prices[city_b]
            if price_a <= 0 or price_b <= 0:
                continue

            spread = price_a - price_b
            pair_key = f"{city_a}_{city_b}"

            if pair_key not in self._spread_history:
                self._spread_history[pair_key] = deque(maxlen=self._lookback)
            self._spread_history[pair_key].append(spread)

            history = list(self._spread_history[pair_key])
            if len(history) < 20:
                continue

            # Check cointegration (simplified: stationarity of spread)
            if not self._check_cointegration(pair_key, history):
                continue

            # Check forecast divergence override
            fc_a = city_forecasts.get(city_a, 0)
            fc_b = city_forecasts.get(city_b, 0)
            if abs(fc_a - fc_b) > self._forecast_diverge_override:
                continue

            # Z-score of current spread
            arr = np.array(history)
            mean = np.mean(arr)
            std = np.std(arr)
            if std < 1e-8:
                continue
            zscore = (spread - mean) / std

            # Exit check
            if market_data.get("has_position"):
                if abs(zscore) <= self._exit_sigma:
                    return Signal(
                        direction="SELL",
                        strength=0.7,
                        edge=abs(spread),
                        strategy_name=self.name,
                        market_id=market_id,
                        city=f"{city_a}_{city_b}",
                        metadata={"exit_reason": "spread_reverted", "zscore": zscore},
                    )

            # Entry: spread exceeds threshold
            if abs(zscore) > self._entry_sigma and abs(zscore) > abs(best_zscore):
                if zscore > self._entry_sigma:
                    # A overpriced vs B — sell A, buy B
                    direction = "BUY"
                    edge = abs(spread - mean) / max(price_b, 0.01)
                    target_city = city_b
                else:
                    # B overpriced vs A — buy A, sell B
                    direction = "BUY"
                    edge = abs(spread - mean) / max(price_a, 0.01)
                    target_city = city_a

                best_zscore = zscore
                best_signal = Signal(
                    direction=direction,
                    strength=min(abs(zscore) / 4.0, 1.0),
                    edge=edge,
                    strategy_name=self.name,
                    market_id=market_id,
                    city=target_city,
                    metadata={
                        "pair": f"{city_a}/{city_b}",
                        "zscore": zscore,
                        "spread": spread,
                        "spread_mean": mean,
                        "spread_std": std,
                        "forecast_diff": fc_a - fc_b,
                    },
                )

        return best_signal

    def _check_cointegration(self, pair_key: str, history: list[float]) -> bool:
        """Simplified cointegration check via ADF-like stationarity test."""
        if pair_key in self._coint_valid and len(history) < self._lookback:
            return self._coint_valid[pair_key]

        arr = np.array(history)
        if len(arr) < 30:
            return False

        # Simple stationarity heuristic: mean-reversion tendency
        diffs = np.diff(arr)
        autocorr = np.corrcoef(arr[:-1], diffs)[0, 1] if len(arr) > 2 else 0
        # Negative autocorrelation = mean-reverting = likely cointegrated
        is_valid = autocorr < -0.1
        self._coint_valid[pair_key] = is_valid

        if not is_valid:
            logger.debug("Pair %s not cointegrated (autocorr=%.3f)", pair_key, autocorr)

        return is_valid

    def get_parameters(self) -> dict[str, Any]:
        return {
            "lookback": self._lookback,
            "entry_sigma": self._entry_sigma,
            "exit_sigma": self._exit_sigma,
            "min_coint_pvalue": self._min_coint_pvalue,
            "forecast_diverge_override": self._forecast_diverge_override,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            if hasattr(self, f"_{k}"):
                setattr(self, f"_{k}", v)

    def reset(self) -> None:
        self._spread_history.clear()
        self._coint_valid.clear()
