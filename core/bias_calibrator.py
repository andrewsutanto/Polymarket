"""Longshot bias calibration from empirical Polymarket data.

Based on analysis of 72.1 million trades showing systematic
mispricing at extreme price levels. Cheap YES contracts are
overpriced; NO consistently outperforms YES.

Key findings (Becker 2026):
- 1¢ contracts return 43¢ per dollar (not 100¢)
- 5¢ contracts win 4.18% of the time (not 5%)
- Takers lose -1.12% per trade, makers earn +1.12%
- NO outperforms YES at 69 of 99 price levels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of bias calibration."""

    raw_probability: float
    calibrated_probability: float
    bias_adjustment: float
    maker_taker_edge: float
    no_side_premium: float
    category_multiplier: float


# Empirical mispricing table from 72.1M trade analysis
# Format: {market_price_cents: actual_win_rate}
# Derived from Becker (2026) prediction market microstructure study
EMPIRICAL_WIN_RATES = {
    1: 0.0043,   # 1¢ → wins 0.43% (not 1%)
    2: 0.0112,   # 2¢ → wins 1.12%
    3: 0.0185,   # 3¢ → wins 1.85%
    5: 0.0418,   # 5¢ → wins 4.18%
    10: 0.0892,  # 10¢ → wins 8.92%
    15: 0.1340,  # 15¢ → wins 13.4%
    20: 0.1810,  # 20¢ → wins 18.1%
    25: 0.2295,  # 25¢ → wins 22.95%
    30: 0.2790,  # 30¢ → wins 27.9%
    35: 0.3280,  # 35¢ → wins 32.8%
    40: 0.3820,  # 40¢ → wins 38.2%
    45: 0.4350,  # 45¢ → wins 43.5%
    50: 0.4950,  # 50¢ → wins 49.5%
    55: 0.5550,  # 55¢ → wins 55.5%
    60: 0.6150,  # 60¢ → wins 61.5%
    65: 0.6720,  # 65¢ → wins 67.2%
    70: 0.7250,  # 70¢ → wins 72.5%
    75: 0.7770,  # 75¢ → wins 77.7%
    80: 0.8280,  # 80¢ → wins 82.8%
    85: 0.8750,  # 85¢ → wins 87.5%
    90: 0.9180,  # 90¢ → wins 91.8%
    95: 0.9580,  # 95¢ → wins 95.8%
    99: 0.9920,  # 99¢ → wins 99.2%
}

# Category inefficiency multipliers (from maker-taker spread data)
# Higher = more inefficient = more edge for makers
CATEGORY_EDGE_MULTIPLIERS = {
    "entertainment": 1.8,  # 7.32pp maker-taker gap
    "sports": 1.5,         # ~5pp gap, emotional betting
    "crypto": 1.4,         # High vol, narrative-driven
    "politics": 1.2,       # Tribal bias (YES on "my candidate")
    "science": 1.1,        # Moderate inefficiency
    "macro": 0.8,          # Near-efficient (quant traders)
    "other": 1.0,          # Baseline
}

# Maker vs taker edge (from 72M trade analysis)
MAKER_EDGE_PER_TRADE = 0.0112   # +1.12%
TAKER_COST_PER_TRADE = -0.0112  # -1.12%

# NO outperforms YES at 69/99 price levels
NO_SIDE_PREMIUM = 0.015  # ~1.5% average NO overperformance


class BiasCalibrator:
    """Calibrates raw probability estimates against empirical biases.

    Applies three corrections:
    1. Longshot bias: cheap contracts are systematically overpriced
    2. Category adjustment: some categories are more inefficient
    3. NO-side premium: YES is systematically overpriced by retail
    """

    def __init__(
        self,
        use_longshot_correction: bool = True,
        use_category_adjustment: bool = True,
        use_no_premium: bool = True,
    ) -> None:
        self._use_longshot = use_longshot_correction
        self._use_category = use_category_adjustment
        self._use_no = use_no_premium

        # Build interpolation table
        self._price_points = sorted(EMPIRICAL_WIN_RATES.keys())
        self._win_rates = [EMPIRICAL_WIN_RATES[p] for p in self._price_points]

    def calibrate(
        self,
        raw_probability: float,
        market_price: float,
        category: str = "other",
        is_yes_side: bool = True,
    ) -> float:
        """Apply all bias corrections to a raw probability estimate.

        Args:
            raw_probability: Model's raw probability estimate [0, 1].
            market_price: Current market price [0, 1].
            category: Market category for edge multiplier.
            is_yes_side: Whether we're pricing the YES outcome.

        Returns:
            Calibrated probability.
        """
        prob = raw_probability

        # 1. Longshot bias correction
        if self._use_longshot:
            prob = self._correct_longshot(prob)

        # 2. NO-side premium
        if self._use_no and is_yes_side:
            # YES is systematically overpriced — adjust down slightly
            if market_price < 0.30:
                prob -= NO_SIDE_PREMIUM * (0.30 - market_price) / 0.30
            elif market_price > 0.70:
                prob += NO_SIDE_PREMIUM * (market_price - 0.70) / 0.30

        return float(np.clip(prob, 0.001, 0.999))

    def get_category_multiplier(self, category: str) -> float:
        """Get edge multiplier for a market category.

        Higher multiplier = more inefficient = size positions larger.
        """
        if not self._use_category:
            return 1.0
        return CATEGORY_EDGE_MULTIPLIERS.get(category, 1.0)

    def get_maker_edge(self) -> float:
        """Return expected maker edge per trade."""
        return MAKER_EDGE_PER_TRADE

    def get_no_side_edge(self, market_price: float) -> float:
        """Return expected NO-side edge at a given price level.

        Below 30¢, NO side has significant systematic edge.
        """
        if market_price < 0.05:
            return 0.57  # 1¢ YES returns only 43¢ → 57% NO edge
        elif market_price < 0.10:
            return 0.20
        elif market_price < 0.30:
            return 0.08
        elif market_price > 0.70:
            return -0.02  # Slight YES edge at high prices
        return 0.0

    def full_calibration(
        self,
        raw_probability: float,
        market_price: float,
        category: str = "other",
        is_yes_side: bool = True,
        is_maker: bool = True,
    ) -> CalibrationResult:
        """Full calibration with all adjustments broken out."""
        calibrated = self.calibrate(raw_probability, market_price, category, is_yes_side)
        category_mult = self.get_category_multiplier(category)
        maker_edge = MAKER_EDGE_PER_TRADE if is_maker else TAKER_COST_PER_TRADE
        no_premium = self.get_no_side_edge(market_price) if not is_yes_side else 0.0

        return CalibrationResult(
            raw_probability=raw_probability,
            calibrated_probability=calibrated,
            bias_adjustment=calibrated - raw_probability,
            maker_taker_edge=maker_edge,
            no_side_premium=no_premium,
            category_multiplier=category_mult,
        )

    def _correct_longshot(self, prob: float) -> float:
        """Apply longshot bias correction via interpolation.

        Maps naive probability to empirically observed win rate.
        """
        price_cents = prob * 100.0

        if price_cents <= self._price_points[0]:
            # Below lowest data point — extrapolate conservatively
            ratio = EMPIRICAL_WIN_RATES[self._price_points[0]] / (self._price_points[0] / 100.0)
            return prob * ratio

        if price_cents >= self._price_points[-1]:
            # Above highest — near 1.0, minimal correction
            ratio = EMPIRICAL_WIN_RATES[self._price_points[-1]] / (self._price_points[-1] / 100.0)
            return prob * ratio

        # Linear interpolation between nearest data points
        for i in range(len(self._price_points) - 1):
            lo = self._price_points[i]
            hi = self._price_points[i + 1]
            if lo <= price_cents <= hi:
                frac = (price_cents - lo) / (hi - lo)
                win_lo = EMPIRICAL_WIN_RATES[lo]
                win_hi = EMPIRICAL_WIN_RATES[hi]
                return win_lo + frac * (win_hi - win_lo)

        return prob
