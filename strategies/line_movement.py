"""Line movement tracking strategy.

Detects when market odds are trending in a direction but haven't
fully arrived at equilibrium. Trades momentum in price movement.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class LineMovementStrategy(BaseStrategy):
    """Trade price trends that haven't fully played out."""

    def __init__(
        self,
        lookback: int = 100,
        trend_threshold: float = 0.03,
        momentum_min_r2: float = 0.3,
        entry_edge: float = 0.03,
        exit_reversal: float = 0.02,
        min_data_points: int = 20,
    ) -> None:
        self._lookback = lookback
        self._trend_threshold = trend_threshold
        self._momentum_r2 = momentum_min_r2
        self._entry_edge = entry_edge
        self._exit_reversal = exit_reversal
        self._min_points = min_data_points

        self._price_history: dict[str, deque[float]] = {}

    @property
    def name(self) -> str:
        return "line_movement"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Detect trending price movement.

        Fits a linear regression to recent price history. If the trend
        is strong (high R²) and the price hasn't reached the projected
        level yet, trade in the trend direction.
        """
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")
        mid_price = market_data.get("mid_price", 0.0)

        if mid_price <= 0 or mid_price >= 1:
            return None

        key = f"{market_id}_{outcome}"

        if key not in self._price_history:
            self._price_history[key] = deque(maxlen=self._lookback)

        self._price_history[key].append(mid_price)
        history = list(self._price_history[key])

        if len(history) < self._min_points:
            return None

        # Linear regression on recent prices
        y = np.array(history[-self._min_points:])
        x = np.arange(len(y), dtype=float)

        slope, intercept, r_squared = self._linear_regression(x, y)

        # Total price move over the window
        total_move = slope * len(y)

        # Exit if trend reversed
        if market_data.get("has_position"):
            recent_reversal = y[-1] - y[-3] if len(y) >= 3 else 0
            if abs(recent_reversal) > self._exit_reversal and np.sign(recent_reversal) != np.sign(slope):
                return Signal(
                    direction="SELL",
                    strength=0.6,
                    edge=abs(recent_reversal),
                    strategy_name=self.name,
                    market_id=market_id,
                    token_id=token_id,
                    market_slug=slug,
                    category=category,
                    outcome=outcome,
                    metadata={"exit_reason": "trend_reversed", "slope": slope, "r2": r_squared},
                )
            return None

        # Entry: strong trend that hasn't fully played out
        if r_squared < self._momentum_r2:
            return None
        if abs(total_move) < self._trend_threshold:
            return None

        # Project where price is heading
        projected = intercept + slope * (len(y) + 5)
        projected = max(0.01, min(0.99, projected))
        edge = abs(projected - mid_price)

        if edge < self._entry_edge:
            return None

        direction = "BUY" if slope > 0 else "SELL"

        return Signal(
            direction=direction,
            strength=min(r_squared * abs(total_move) / 0.05, 1.0),
            edge=edge,
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id,
            market_slug=slug,
            category=category,
            outcome=outcome,
            metadata={
                "slope": slope,
                "r_squared": r_squared,
                "total_move": total_move,
                "projected_5bar": projected,
                "data_points": len(y),
            },
        )

    @staticmethod
    def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """Simple linear regression returning (slope, intercept, r_squared)."""
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_yy = np.sum((y - y_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        if ss_xx < 1e-10:
            return 0.0, float(y_mean), 0.0

        slope = float(ss_xy / ss_xx)
        intercept = float(y_mean - slope * x_mean)

        if ss_yy < 1e-10:
            r_squared = 0.0
        else:
            r_squared = float((ss_xy ** 2) / (ss_xx * ss_yy))

        return slope, intercept, r_squared

    def get_parameters(self) -> dict[str, Any]:
        return {
            "lookback": self._lookback,
            "trend_threshold": self._trend_threshold,
            "momentum_min_r2": self._momentum_r2,
            "entry_edge": self._entry_edge,
            "exit_reversal": self._exit_reversal,
            "min_data_points": self._min_points,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "lookback" in params:
            self._lookback = params["lookback"]
        if "trend_threshold" in params:
            self._trend_threshold = params["trend_threshold"]
        if "momentum_min_r2" in params:
            self._momentum_r2 = params["momentum_min_r2"]
        if "entry_edge" in params:
            self._entry_edge = params["entry_edge"]
        if "exit_reversal" in params:
            self._exit_reversal = params["exit_reversal"]
        if "min_data_points" in params:
            self._min_points = params["min_data_points"]

    def reset(self) -> None:
        self._price_history.clear()
