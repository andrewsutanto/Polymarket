"""Temporal bias strategy — exploit time-dependent structural patterns.

Three systematic biases in prediction markets related to time:

1. **Expiry convergence**: Markets < 24hrs from resolution converge to
   true value faster. Mispricings near expiry are smaller but higher
   conviction — the market "knows" the answer and stragglers get picked off.

2. **Time-of-day effects**: Asian session (00:00-08:00 UTC) has wider
   spreads and thinner books due to low US participation. US session
   (13:00-21:00 UTC) has tighter spreads and more efficient pricing.
   European overlap (08:00-13:00 UTC) is transitional.

3. **Weekend patterns**: Saturday/Sunday have lower volume, wider spreads,
   and slower price discovery. Monday openings often see mean-reversion
   as weekend drift corrects.

These effects are structural and persistent — they arise from the
composition of the participant base, not from any particular market
fundamental.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

# Session definitions (UTC hours)
_ASIAN_SESSION = (0, 8)      # 00:00–08:00 UTC
_EURO_SESSION = (8, 13)      # 08:00–13:00 UTC
_US_SESSION = (13, 21)       # 13:00–21:00 UTC
_OFF_HOURS = (21, 24)        # 21:00–00:00 UTC

# Spread multipliers by session (relative to US = 1.0)
# Asian hours have ~60% wider spreads on average
SESSION_SPREAD_MULTIPLIERS = {
    "asian": 1.60,
    "european": 1.20,
    "us": 1.00,
    "off_hours": 1.35,
}

# Weekend spread widening factor
WEEKEND_SPREAD_MULTIPLIER = 1.40


class TemporalBiasStrategy(BaseStrategy):
    """Exploit time-dependent structural inefficiencies.

    Parameters:
        min_edge: Minimum estimated edge to emit a signal.
        expiry_hours_threshold: Hours-to-expiry below which convergence
            logic activates (default 24).
        expiry_edge_boost: Multiplier for edge confidence near expiry.
        spread_ratio_threshold: Ratio of current spread to expected
            session spread — above this, the market is unusually wide.
        weekend_drift_threshold: Price drift from Friday close beyond
            which we expect Monday mean-reversion.
        min_liquidity: Minimum USD liquidity to consider the market.
        off_session_min_spread: Minimum spread (cents) to consider
            an off-session opportunity.
    """

    def __init__(
        self,
        min_edge: float = 0.02,
        expiry_hours_threshold: float = 24.0,
        expiry_edge_boost: float = 1.5,
        spread_ratio_threshold: float = 1.5,
        weekend_drift_threshold: float = 0.03,
        min_liquidity: float = 200.0,
        off_session_min_spread: float = 0.03,
    ) -> None:
        self._min_edge = min_edge
        self._expiry_hours_threshold = expiry_hours_threshold
        self._expiry_edge_boost = expiry_edge_boost
        self._spread_ratio_threshold = spread_ratio_threshold
        self._weekend_drift_threshold = weekend_drift_threshold
        self._min_liquidity = min_liquidity
        self._off_session_min_spread = off_session_min_spread

    @property
    def name(self) -> str:
        return "temporal_bias"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Evaluate temporal biases and emit signal if actionable.

        Checks (in priority order):
        1. Near-expiry convergence plays
        2. Off-session spread capture
        3. Weekend drift mean-reversion
        """
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")
        yes_price = market_data.get("yes_price", 0.0)
        no_price = market_data.get("no_price", 0.0)
        mid_price = market_data.get("mid_price", (yes_price + no_price) / 2.0)
        spread = market_data.get("spread", abs(yes_price - (1.0 - no_price)))
        liquidity = market_data.get("liquidity", 0.0)
        time_to_resolution_hrs = market_data.get("time_to_resolution_hrs", float("inf"))
        timestamp = market_data.get("timestamp")

        if liquidity < self._min_liquidity:
            return None

        # Parse timestamp
        now_utc = self._parse_timestamp(timestamp)
        session = self._classify_session(now_utc)
        is_weekend = now_utc.weekday() >= 5  # Saturday=5, Sunday=6

        # --- Signal 1: Near-expiry convergence ---
        signal = self._check_expiry_convergence(
            time_to_resolution_hrs, mid_price, spread, session,
            market_id, token_id, slug, category, outcome, market_data,
        )
        if signal is not None:
            return signal

        # --- Signal 2: Off-session spread capture ---
        signal = self._check_session_spread(
            session, is_weekend, spread, mid_price, yes_price, no_price,
            market_id, token_id, slug, category, outcome, market_data,
        )
        if signal is not None:
            return signal

        # --- Signal 3: Weekend drift mean-reversion ---
        signal = self._check_weekend_reversion(
            is_weekend, now_utc, mid_price, market_data,
            market_id, token_id, slug, category, outcome,
        )
        if signal is not None:
            return signal

        return None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_edge": self._min_edge,
            "expiry_hours_threshold": self._expiry_hours_threshold,
            "expiry_edge_boost": self._expiry_edge_boost,
            "spread_ratio_threshold": self._spread_ratio_threshold,
            "weekend_drift_threshold": self._weekend_drift_threshold,
            "min_liquidity": self._min_liquidity,
            "off_session_min_spread": self._off_session_min_spread,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            attr = f"_{k}"
            if hasattr(self, attr):
                setattr(self, attr, v)

    # ------------------------------------------------------------------ #
    # Sub-signal checkers
    # ------------------------------------------------------------------ #

    def _check_expiry_convergence(
        self,
        time_to_resolution_hrs: float,
        mid_price: float,
        spread: float,
        session: str,
        market_id: str,
        token_id: str,
        slug: str,
        category: str,
        outcome: str,
        market_data: dict[str, Any],
    ) -> Signal | None:
        """Near-expiry markets converge — mispricings are high-conviction.

        If the market is near expiry and the price is far from 0 or 1,
        it should be converging. A wide spread near expiry is a strong
        signal that the market hasn't fully priced in the resolution.
        """
        if time_to_resolution_hrs > self._expiry_hours_threshold:
            return None

        # Convergence strength: how close to expiry (exponential urgency)
        # At 1hr: urgency = 1.0. At 24hr: urgency ~= 0.04
        urgency = math.exp(-0.15 * time_to_resolution_hrs)

        # Price should be converging to 0 or 1. The "stuck in the middle"
        # region (0.3-0.7) near expiry suggests mispricing.
        distance_from_edge = min(mid_price, 1.0 - mid_price)

        # If price is already near 0 or 1, convergence is done — no edge
        if distance_from_edge < 0.10:
            return None

        # Spread near expiry = opportunity. Normally spreads tighten near
        # expiry; if they haven't, someone is stale.
        expected_spread = spread / max(
            SESSION_SPREAD_MULTIPLIERS.get(session, 1.0), 0.5
        )

        # Edge estimate: urgency * distance * spread-based component
        edge = urgency * distance_from_edge * 0.15 + expected_spread * 0.5
        edge *= self._expiry_edge_boost

        if edge < self._min_edge:
            return None

        # Direction: near expiry, lean toward the side closer to resolution
        # If price > 0.5, likely resolving YES; if < 0.5, likely NO
        if mid_price >= 0.5:
            direction = "BUY"
            signal_outcome = outcome or "Yes"
        else:
            direction = "SELL"
            signal_outcome = outcome or "Yes"

        strength = min(urgency * (1.0 + distance_from_edge), 1.0)

        return Signal(
            direction=direction,
            strength=round(strength, 4),
            edge=round(edge, 4),
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id,
            market_slug=slug,
            category=category,
            outcome=signal_outcome,
            metadata={
                "sub_signal": "expiry_convergence",
                "time_to_resolution_hrs": round(time_to_resolution_hrs, 2),
                "urgency": round(urgency, 4),
                "distance_from_edge": round(distance_from_edge, 4),
                "session": session,
            },
        )

    def _check_session_spread(
        self,
        session: str,
        is_weekend: bool,
        spread: float,
        mid_price: float,
        yes_price: float,
        no_price: float,
        market_id: str,
        token_id: str,
        slug: str,
        category: str,
        outcome: str,
        market_data: dict[str, Any],
    ) -> Signal | None:
        """Capture wider spreads during off-peak sessions.

        During Asian hours or weekends, spreads widen because US-based
        market makers are less active. We can provide liquidity (act as
        maker) and capture the wider spread.
        """
        if spread < self._off_session_min_spread:
            return None

        # Expected spread for this session
        session_mult = SESSION_SPREAD_MULTIPLIERS.get(session, 1.0)
        if is_weekend:
            session_mult *= WEEKEND_SPREAD_MULTIPLIER

        # Is spread unusually wide for the session?
        # We compare to US session as baseline
        us_expected_spread = spread / session_mult
        spread_ratio = spread / max(us_expected_spread, 0.001)

        if spread_ratio < self._spread_ratio_threshold:
            return None

        # Edge is half the excess spread (we capture it by making on the
        # better side and waiting for US hours to tighten)
        excess_spread = spread - us_expected_spread
        edge = excess_spread * 0.5

        if edge < self._min_edge:
            return None

        # Direction: buy the cheaper side (provide liquidity)
        if yes_price < no_price:
            direction = "BUY"
            signal_outcome = outcome or "Yes"
        else:
            direction = "BUY"
            signal_outcome = outcome or "No"

        strength = min(spread_ratio / (self._spread_ratio_threshold * 2), 1.0)

        return Signal(
            direction=direction,
            strength=round(strength, 4),
            edge=round(edge, 4),
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id,
            market_slug=slug,
            category=category,
            outcome=signal_outcome,
            metadata={
                "sub_signal": "session_spread",
                "session": session,
                "is_weekend": is_weekend,
                "spread": round(spread, 4),
                "spread_ratio": round(spread_ratio, 4),
                "session_multiplier": session_mult,
                "excess_spread": round(excess_spread, 4),
            },
        )

    def _check_weekend_reversion(
        self,
        is_weekend: bool,
        now_utc: datetime,
        mid_price: float,
        market_data: dict[str, Any],
        market_id: str,
        token_id: str,
        slug: str,
        category: str,
        outcome: str,
    ) -> Signal | None:
        """Monday mean-reversion of weekend price drift.

        On Monday (especially early hours), prices that drifted over the
        weekend tend to revert. We need a reference price (Friday close)
        to detect drift.
        """
        # Only fire on Monday before US session opens
        if now_utc.weekday() != 0:  # Monday = 0
            return None
        if now_utc.hour >= 13:  # After US session opens, drift is corrected
            return None

        friday_close = market_data.get("friday_close_price")
        if friday_close is None:
            return None

        drift = mid_price - friday_close

        if abs(drift) < self._weekend_drift_threshold:
            return None

        # Expect mean-reversion: if price drifted up, sell; if down, buy
        edge = abs(drift) * 0.6  # Expect ~60% reversion on average

        if edge < self._min_edge:
            return None

        if drift > 0:
            direction = "SELL"
        else:
            direction = "BUY"

        strength = min(abs(drift) / 0.10, 1.0)

        return Signal(
            direction=direction,
            strength=round(strength, 4),
            edge=round(edge, 4),
            strategy_name=self.name,
            market_id=market_id,
            token_id=token_id,
            market_slug=slug,
            category=category,
            outcome=outcome or "Yes",
            metadata={
                "sub_signal": "weekend_reversion",
                "friday_close": round(friday_close, 4),
                "current_mid": round(mid_price, 4),
                "drift": round(drift, 4),
                "expected_reversion": round(edge, 4),
            },
        )

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_timestamp(timestamp: Any) -> datetime:
        """Parse timestamp to UTC datetime."""
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _classify_session(dt: datetime) -> str:
        """Classify a UTC datetime into a trading session."""
        hour = dt.hour
        if _ASIAN_SESSION[0] <= hour < _ASIAN_SESSION[1]:
            return "asian"
        if _EURO_SESSION[0] <= hour < _EURO_SESSION[1]:
            return "european"
        if _US_SESSION[0] <= hour < _US_SESSION[1]:
            return "us"
        return "off_hours"
