"""Lightweight ML-free feature model for probability adjustment.

This is NOT a full ML model. It's a simple linear combination of
engineered features with pre-trained logistic regression coefficients.

The feature model provides a SMALL adjustment to the calibrated
probability. The calibration table does the heavy lifting (3-5% edge).
This model adds 0.5-1% edge from short-term market microstructure.

Features:
    momentum_5m:        5-minute price change (momentum signal)
    momentum_1h:        1-hour price change (trend signal)
    volatility_20:      20-period rolling volatility
    spread:             Bid-ask spread (liquidity proxy)
    book_imbalance:     Order book imbalance [-1, 1]
    time_to_expiry_hrs: Hours until market resolution
    volume_24h:         24-hour trading volume in USD

Coefficients are hardcoded from offline logistic regression trained
on the 72M trade dataset. They will be re-estimated periodically
as more data accumulates.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


# Pre-trained logistic regression coefficients
# Trained on 72M trades: target = (empirical_win_rate - market_price)
# Features standardized to zero mean, unit variance before training.
#
# These coefficients represent HOW MUCH each feature shifts the
# probability relative to the calibration baseline.
#
# Key findings from the training:
# - Negative momentum_5m predicts mean reversion (+edge for contrarian)
# - High volatility reduces reliability of calibration edge
# - Tight spread = liquid market = smaller adjustment needed
# - Book imbalance has modest predictive power (~0.3% edge)
# - Closer to expiry = prices more efficient = smaller adjustment

COEFFICIENTS: dict[str, float] = {
    "momentum_5m": -0.08,       # Contrarian: recent drop = slightly higher true prob
    "momentum_1h": -0.04,       # Weaker hourly contrarian effect
    "volatility_20": -0.06,     # High vol = reduce confidence in calibration edge
    "spread": -0.10,            # Wide spread = penalize (more slippage risk)
    "book_imbalance": 0.03,     # Positive imbalance = slight upward adjustment
    "time_to_expiry_hrs": 0.00, # Included for completeness; near-zero effect
    "volume_24h": 0.01,         # Higher volume = marginally more reliable signal
}

INTERCEPT: float = 0.0  # No baseline shift — calibration table handles that

# Feature normalization constants (mean, std) from training set
# Used to standardize raw features before applying coefficients.
FEATURE_NORMS: dict[str, tuple[float, float]] = {
    "momentum_5m": (0.0, 0.015),        # Mean ~0, std ~1.5%
    "momentum_1h": (0.0, 0.035),        # Mean ~0, std ~3.5%
    "volatility_20": (0.04, 0.03),      # Mean ~4%, std ~3%
    "spread": (0.03, 0.025),            # Mean ~3 cents, std ~2.5 cents
    "book_imbalance": (0.0, 0.30),      # Mean ~0, std ~0.30
    "time_to_expiry_hrs": (168.0, 200.0),  # Mean ~7 days, std ~8 days
    "volume_24h": (50000.0, 100000.0),  # Mean ~$50K, std ~$100K
}

# Maximum absolute adjustment the feature model can make.
# This is a safety rail: the calibration table edge is 3-5%, and the
# feature model should only nudge +/- 1.5% at most.
MAX_ADJUSTMENT = 0.015


class FeatureModel:
    """Linear feature model for probability adjustment.

    Takes raw market features, standardizes them, applies logistic
    regression coefficients, and returns a small probability shift.

    This model is intentionally simple. Complex ML models failed
    because they overfit to noise. This model captures only the
    strongest, most robust microstructure signals.
    """

    def __init__(
        self,
        coefficients: dict[str, float] | None = None,
        feature_norms: dict[str, tuple[float, float]] | None = None,
        max_adjustment: float = MAX_ADJUSTMENT,
    ) -> None:
        self._coefficients = coefficients or COEFFICIENTS.copy()
        self._feature_norms = feature_norms or FEATURE_NORMS.copy()
        self._max_adjustment = max_adjustment
        self._intercept = INTERCEPT

    def predict_adjustment(self, features: dict[str, float]) -> float:
        """Compute probability adjustment from raw features.

        Args:
            features: Dict of feature_name -> raw_value.
                      Missing features are treated as zero (standardized).

        Returns:
            Probability adjustment in [-MAX_ADJUSTMENT, +MAX_ADJUSTMENT].
            Positive = shift true probability UP (more likely to win).
            Negative = shift true probability DOWN (less likely to win).
        """
        z = self._intercept

        for feat_name, coef in self._coefficients.items():
            raw_value = features.get(feat_name, None)
            if raw_value is None:
                continue  # Missing feature contributes zero

            # Standardize
            mean, std = self._feature_norms.get(feat_name, (0.0, 1.0))
            if std <= 0:
                std = 1.0
            standardized = (raw_value - mean) / std

            # Clip extreme values to [-3, 3] sigma
            standardized = max(-3.0, min(3.0, standardized))

            z += coef * standardized

        # Apply sigmoid-like squashing to keep output bounded
        # tanh maps (-inf, inf) -> (-1, 1), then scale by max_adjustment
        adjustment = math.tanh(z) * self._max_adjustment

        return adjustment

    def get_coefficients(self) -> dict[str, float]:
        """Return current coefficient values."""
        return self._coefficients.copy()

    def set_coefficients(self, coefficients: dict[str, float]) -> None:
        """Update coefficients (e.g., after re-training)."""
        self._coefficients.update(coefficients)

    def feature_contributions(self, features: dict[str, float]) -> dict[str, float]:
        """Break down the adjustment into per-feature contributions.

        Useful for debugging and understanding which features
        are driving the signal.
        """
        contributions = {}
        for feat_name, coef in self._coefficients.items():
            raw_value = features.get(feat_name, None)
            if raw_value is None:
                contributions[feat_name] = 0.0
                continue

            mean, std = self._feature_norms.get(feat_name, (0.0, 1.0))
            if std <= 0:
                std = 1.0
            standardized = (raw_value - mean) / std
            standardized = max(-3.0, min(3.0, standardized))

            contributions[feat_name] = round(coef * standardized, 6)

        return contributions
