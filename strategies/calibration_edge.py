"""Calibration Edge strategy — exploit longshot bias as the primary alpha.

Instead of trying to predict price movements (Markov), this strategy:
1. Takes the MARKET PRICE as the best probability estimate
2. Looks up the EMPIRICAL win rate from the 72M-trade calibration table
3. If empirical_win_rate - market_price > threshold (after fees), trades

The edge is real, persistent, and well-documented:
- Cheap YES contracts are systematically overpriced (longshot bias)
- Expensive YES contracts are slightly underpriced (favorite bias)
- NO outperforms YES at 69/99 price levels

This is NOT a prediction model. It exploits a proven market inefficiency
with minimal model complexity. The calibration table IS the strategy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.bias_calibrator import (
    BiasCalibrator,
    EMPIRICAL_WIN_RATES,
    TAKER_COST_PER_TRADE,
)
from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


# Default taker fee on Polymarket (2% round-trip: 1% entry + 1% exit)
DEFAULT_TAKER_FEE = 0.02

# Maximum edge we consider credible (anything above is likely data error)
MAX_CREDIBLE_EDGE = 0.25


class CalibrationEdgeStrategy(BaseStrategy):
    """Trade the gap between market price and empirical win probability.

    The simplest possible strategy with real information advantage:
    the market systematically misprices contracts at extreme levels.

    Parameters:
        min_edge: Minimum raw edge (empirical - market) to trade (default 4%)
        taker_fee: Round-trip taker fee to subtract (default 2%)
        min_fee_adjusted_edge: Minimum edge AFTER fees (default 2%)
        use_feature_adjustment: Whether to apply feature model adjustment
        max_spread: Maximum bid-ask spread to trade (wider = more slippage)
        min_volume_24h: Minimum 24h volume in USD
        min_time_to_expiry_hrs: Don't trade markets expiring too soon
        category_scaling: Whether to scale edge by category inefficiency
    """

    def __init__(
        self,
        min_edge: float = 0.04,
        taker_fee: float = DEFAULT_TAKER_FEE,
        min_fee_adjusted_edge: float = 0.02,
        use_feature_adjustment: bool = True,
        max_spread: float = 0.08,
        min_volume_24h: float = 0.0,
        min_time_to_expiry_hrs: float = 1.0,
        category_scaling: bool = True,
    ) -> None:
        self._min_edge = min_edge
        self._taker_fee = taker_fee
        self._min_fee_adjusted_edge = min_fee_adjusted_edge
        self._use_feature_adjustment = use_feature_adjustment
        self._max_spread = max_spread
        self._min_volume_24h = min_volume_24h
        self._min_time_to_expiry_hrs = min_time_to_expiry_hrs
        self._category_scaling = category_scaling

        self._calibrator = BiasCalibrator(
            use_longshot_correction=True,
            use_category_adjustment=category_scaling,
            use_no_premium=True,
        )

        # Optional feature model (lazy import to avoid circular deps)
        self._feature_model = None
        if use_feature_adjustment:
            try:
                from core.feature_model import FeatureModel
                self._feature_model = FeatureModel()
            except ImportError:
                logger.warning("FeatureModel not available, running without feature adjustment")

    @property
    def name(self) -> str:
        return "calibration_edge"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Compute calibration edge and generate trade signal.

        The core logic:
        1. Get market YES price
        2. Look up empirical win rate at that price level
        3. Compute raw edge = empirical_win_rate - market_price
        4. Subtract taker fees
        5. If fee-adjusted edge > threshold, signal a trade

        Works on BOTH sides:
        - Longshot bias (cheap contracts): empirical < market -> BUY NO
        - Favorite bias (expensive contracts): empirical > market -> BUY YES
        """
        yes_price = market_data.get("yes_price", 0.0)
        no_price = market_data.get("no_price", 0.0)
        spread = market_data.get("spread", 1.0)
        category = market_data.get("category", "other")
        volume_24h = market_data.get("volume_24h", 0.0)
        time_to_expiry = market_data.get("time_to_resolution_hrs", 999.0)

        # ── Basic filters ──
        if spread > self._max_spread:
            return None
        if volume_24h < self._min_volume_24h:
            return None
        if time_to_expiry < self._min_time_to_expiry_hrs:
            return None
        if yes_price <= 0.01 or yes_price >= 0.99:
            return None

        # ── Core: look up empirical win rate ──
        empirical_yes = self._calibrator.calibrate(
            raw_probability=yes_price,
            market_price=yes_price,
            category=category,
            is_yes_side=True,
        )

        # Also compute NO side empirical rate for the complement
        empirical_no = 1.0 - self._calibrator.calibrate(
            raw_probability=1.0 - no_price if no_price > 0 else 1.0 - yes_price,
            market_price=1.0 - (no_price if no_price > 0 else yes_price),
            category=category,
            is_yes_side=True,
        )

        # ── Compute edges for both sides ──
        yes_edge = empirical_yes - yes_price  # Positive = YES underpriced
        no_edge = (1.0 - empirical_yes) - no_price  # Positive = NO underpriced

        # ── Apply feature model adjustment if available ──
        feature_adj = 0.0
        if self._feature_model is not None:
            features = self._extract_features(market_data)
            feature_adj = self._feature_model.predict_adjustment(features)
            # Feature model shifts the empirical probability
            yes_edge += feature_adj
            no_edge -= feature_adj  # Opposite direction for NO

        # ── Category scaling ──
        cat_mult = 1.0
        if self._category_scaling:
            cat_mult = self._calibrator.get_category_multiplier(category)

        # ── Pick the better side ──
        scaled_yes_edge = yes_edge * cat_mult
        scaled_no_edge = no_edge * cat_mult

        if scaled_yes_edge >= scaled_no_edge and scaled_yes_edge > 0:
            raw_edge = scaled_yes_edge
            direction = "BUY"
            outcome = "Yes"
            trade_price = yes_price
        elif scaled_no_edge > 0:
            raw_edge = scaled_no_edge
            direction = "BUY"
            outcome = "No"
            trade_price = no_price if no_price > 0 else 1.0 - yes_price
        else:
            return None

        # ── Fee adjustment ──
        fee_adjusted_edge = raw_edge - self._taker_fee

        # ── Threshold checks ──
        if raw_edge < self._min_edge:
            return None
        if fee_adjusted_edge < self._min_fee_adjusted_edge:
            return None
        if raw_edge > MAX_CREDIBLE_EDGE:
            # Suspiciously large edge — likely stale data or error
            return None

        # ── Signal strength proportional to edge ──
        # Normalize: 2% fee-adjusted edge = 0.3 strength, 10% = 1.0
        strength = min(1.0, max(0.1, fee_adjusted_edge / 0.10))

        return Signal(
            direction=direction,
            strength=round(strength, 3),
            edge=round(fee_adjusted_edge, 4),
            strategy_name=self.name,
            market_id=market_data.get("market_id", ""),
            token_id=market_data.get("token_id", ""),
            market_slug=market_data.get("market_slug", ""),
            category=category,
            outcome=outcome,
            metadata={
                "raw_edge": round(raw_edge, 4),
                "fee_adjusted_edge": round(fee_adjusted_edge, 4),
                "empirical_yes_prob": round(empirical_yes, 4),
                "market_yes_price": round(yes_price, 4),
                "category_multiplier": round(cat_mult, 2),
                "feature_adjustment": round(feature_adj, 4),
                "trade_price": round(trade_price, 4),
                "taker_fee": self._taker_fee,
            },
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_edge": self._min_edge,
            "taker_fee": self._taker_fee,
            "min_fee_adjusted_edge": self._min_fee_adjusted_edge,
            "use_feature_adjustment": self._use_feature_adjustment,
            "max_spread": self._max_spread,
            "min_volume_24h": self._min_volume_24h,
            "min_time_to_expiry_hrs": self._min_time_to_expiry_hrs,
            "category_scaling": self._category_scaling,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "min_edge" in params:
            self._min_edge = params["min_edge"]
        if "taker_fee" in params:
            self._taker_fee = params["taker_fee"]
        if "min_fee_adjusted_edge" in params:
            self._min_fee_adjusted_edge = params["min_fee_adjusted_edge"]
        if "max_spread" in params:
            self._max_spread = params["max_spread"]
        if "min_volume_24h" in params:
            self._min_volume_24h = params["min_volume_24h"]
        if "min_time_to_expiry_hrs" in params:
            self._min_time_to_expiry_hrs = params["min_time_to_expiry_hrs"]
        if "category_scaling" in params:
            self._category_scaling = params["category_scaling"]

    def reset(self) -> None:
        """No internal state to reset — the calibration table is static."""
        pass

    def _extract_features(self, market_data: dict[str, Any]) -> dict[str, float]:
        """Extract features for the feature model from market data.

        Maps the market_data dict into the feature vector the FeatureModel expects.
        Returns 0.0 for any missing features (graceful degradation).
        """
        history = market_data.get("price_history", [])

        # Momentum features
        momentum_5m = 0.0
        momentum_1h = 0.0
        if len(history) >= 2:
            momentum_5m = history[-1] - history[-2]
        if len(history) >= 12:
            momentum_1h = history[-1] - history[-12]

        # Volatility
        volatility_20 = 0.0
        if len(history) >= 20:
            volatility_20 = float(np.std(history[-20:]))

        # Book features
        spread = market_data.get("spread", 0.0)
        best_bid = market_data.get("best_bid", 0.0)
        best_ask = market_data.get("best_ask", 1.0)
        bid_depth = market_data.get("book_depth", 0.0)
        # Book imbalance: positive = more buying pressure
        if best_bid + best_ask > 0:
            book_imbalance = (best_bid - (1.0 - best_ask)) / max(best_bid + best_ask, 0.01)
        else:
            book_imbalance = 0.0

        return {
            "momentum_5m": momentum_5m,
            "momentum_1h": momentum_1h,
            "volatility_20": volatility_20,
            "spread": spread,
            "book_imbalance": book_imbalance,
            "time_to_expiry_hrs": market_data.get("time_to_resolution_hrs", 999.0),
            "volume_24h": market_data.get("volume_24h", 0.0),
        }
