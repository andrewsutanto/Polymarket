"""Orderbook imbalance strategy — directional pressure from bid/ask volume.

Computes the bid/ask volume imbalance ratio at various depth levels.
When bids significantly outweigh asks (or vice versa), this signals
short-term directional pressure before the price adjusts.

Key insight: Imbalance is most predictive within the first few minutes
of appearing, then decays as the market absorbs the information.
We apply an exponential decay factor to weight recent imbalance snapshots
more heavily than stale ones.

Uses CLOB orderbook data from Polymarket's API.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class ImbalanceSnapshot:
    """A single point-in-time orderbook imbalance reading."""

    timestamp: float
    bid_volume: float
    ask_volume: float
    ratio: float  # bid_volume / ask_volume
    depth_usd: float  # total USD depth captured


class OrderbookImbalanceStrategy(BaseStrategy):
    """Trade on orderbook volume imbalance with time decay.

    Parameters:
        imbalance_threshold: Minimum bid:ask ratio to trigger signal (default 3.0).
        inverse_threshold: Maximum ratio (1/threshold) for sell signals.
        decay_half_life_s: Half-life in seconds for imbalance decay (default 180s = 3min).
        min_depth_usd: Minimum USD depth on each side to trust the signal.
        lookback_window: Number of snapshots to retain for decay-weighted average.
        min_snapshots: Minimum snapshots before generating a signal.
        strength_cap_ratio: Ratio at which signal strength saturates to 1.0.
        min_edge: Minimum estimated edge to emit a signal.
        depth_levels: Number of price levels to aggregate.
    """

    def __init__(
        self,
        imbalance_threshold: float = 3.0,
        decay_half_life_s: float = 180.0,
        min_depth_usd: float = 100.0,
        lookback_window: int = 50,
        min_snapshots: int = 3,
        strength_cap_ratio: float = 8.0,
        min_edge: float = 0.02,
        depth_levels: int = 5,
    ) -> None:
        self._imbalance_threshold = imbalance_threshold
        self._inverse_threshold = 1.0 / imbalance_threshold
        self._decay_half_life_s = decay_half_life_s
        self._min_depth_usd = min_depth_usd
        self._lookback_window = lookback_window
        self._min_snapshots = min_snapshots
        self._strength_cap_ratio = strength_cap_ratio
        self._min_edge = min_edge
        self._depth_levels = depth_levels

        # Per-market snapshot history
        self._snapshots: dict[str, deque[ImbalanceSnapshot]] = {}

    @property
    def name(self) -> str:
        return "orderbook_imbalance"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze orderbook imbalance and emit signal if threshold exceeded.

        Expects market_data to contain:
            - orderbook: {"bids": [[price, size], ...], "asks": [[price, size], ...]}
              OR pre-computed:
            - bid_volume: total bid volume in USD
            - ask_volume: total ask volume in USD
            - timestamp: current time (epoch seconds or ISO)
        """
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")
        yes_price = market_data.get("yes_price", 0.0)
        no_price = market_data.get("no_price", 0.0)
        mid_price = market_data.get("mid_price", (yes_price + no_price) / 2.0)
        timestamp = market_data.get("timestamp", time.time())

        if isinstance(timestamp, str):
            # Accept ISO format or epoch
            try:
                from datetime import datetime, timezone

                timestamp = datetime.fromisoformat(timestamp).timestamp()
            except (ValueError, TypeError):
                timestamp = time.time()

        # Extract bid/ask volumes from orderbook or pre-computed fields
        bid_vol, ask_vol, depth = self._extract_volumes(market_data)

        if bid_vol <= 0 or ask_vol <= 0:
            return None
        if depth < self._min_depth_usd:
            return None

        # Compute raw ratio
        ratio = bid_vol / ask_vol

        # Store snapshot
        snap = ImbalanceSnapshot(
            timestamp=timestamp,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            ratio=ratio,
            depth_usd=depth,
        )
        if market_id not in self._snapshots:
            self._snapshots[market_id] = deque(maxlen=self._lookback_window)
        self._snapshots[market_id].append(snap)

        # Need minimum history
        history = self._snapshots[market_id]
        if len(history) < self._min_snapshots:
            return None

        # Compute decay-weighted average imbalance ratio
        weighted_ratio = self._decay_weighted_ratio(history, timestamp)

        # Check thresholds
        if weighted_ratio >= self._imbalance_threshold:
            # Strong bid pressure => price likely to go up => BUY YES
            edge = self._estimate_edge(weighted_ratio, mid_price, direction="up")
            if edge < self._min_edge:
                return None

            strength = min(
                (weighted_ratio - self._imbalance_threshold)
                / (self._strength_cap_ratio - self._imbalance_threshold),
                1.0,
            )
            strength = max(strength, 0.1)

            return Signal(
                direction="BUY",
                strength=round(strength, 4),
                edge=round(edge, 4),
                strategy_name=self.name,
                market_id=market_id,
                token_id=token_id,
                market_slug=slug,
                category=category,
                outcome=outcome or "Yes",
                metadata={
                    "weighted_ratio": round(weighted_ratio, 4),
                    "raw_ratio": round(ratio, 4),
                    "bid_volume": round(bid_vol, 2),
                    "ask_volume": round(ask_vol, 2),
                    "depth_usd": round(depth, 2),
                    "n_snapshots": len(history),
                    "pressure": "bid",
                },
            )

        elif weighted_ratio <= self._inverse_threshold:
            # Strong ask pressure => price likely to go down => SELL YES / BUY NO
            edge = self._estimate_edge(
                1.0 / weighted_ratio, mid_price, direction="down"
            )
            if edge < self._min_edge:
                return None

            strength = min(
                (self._inverse_threshold - weighted_ratio)
                / (self._inverse_threshold - 1.0 / self._strength_cap_ratio),
                1.0,
            )
            strength = max(strength, 0.1)

            return Signal(
                direction="SELL",
                strength=round(strength, 4),
                edge=round(edge, 4),
                strategy_name=self.name,
                market_id=market_id,
                token_id=token_id,
                market_slug=slug,
                category=category,
                outcome=outcome or "Yes",
                metadata={
                    "weighted_ratio": round(weighted_ratio, 4),
                    "raw_ratio": round(ratio, 4),
                    "bid_volume": round(bid_vol, 2),
                    "ask_volume": round(ask_vol, 2),
                    "depth_usd": round(depth, 2),
                    "n_snapshots": len(history),
                    "pressure": "ask",
                },
            )

        return None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "imbalance_threshold": self._imbalance_threshold,
            "decay_half_life_s": self._decay_half_life_s,
            "min_depth_usd": self._min_depth_usd,
            "lookback_window": self._lookback_window,
            "min_snapshots": self._min_snapshots,
            "strength_cap_ratio": self._strength_cap_ratio,
            "min_edge": self._min_edge,
            "depth_levels": self._depth_levels,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            attr = f"_{k}"
            if hasattr(self, attr):
                setattr(self, attr, v)
        # Recompute inverse threshold if main threshold changed
        if "imbalance_threshold" in params:
            self._inverse_threshold = 1.0 / self._imbalance_threshold

    def reset(self) -> None:
        """Clear all snapshot history."""
        self._snapshots.clear()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_volumes(
        self, market_data: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Extract bid/ask volumes from orderbook or pre-computed fields.

        Returns:
            (bid_volume_usd, ask_volume_usd, total_depth_usd)
        """
        # Option 1: Pre-computed volumes
        if "bid_volume" in market_data and "ask_volume" in market_data:
            bid = float(market_data["bid_volume"])
            ask = float(market_data["ask_volume"])
            return bid, ask, bid + ask

        # Option 2: Raw orderbook
        orderbook = market_data.get("orderbook", {})
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return 0.0, 0.0, 0.0

        # Aggregate top N levels (price * size = USD volume)
        bid_vol = 0.0
        for i, level in enumerate(bids):
            if i >= self._depth_levels:
                break
            price, size = float(level[0]), float(level[1])
            bid_vol += price * size

        ask_vol = 0.0
        for i, level in enumerate(asks):
            if i >= self._depth_levels:
                break
            price, size = float(level[0]), float(level[1])
            ask_vol += price * size

        return bid_vol, ask_vol, bid_vol + ask_vol

    def _decay_weighted_ratio(
        self, history: deque[ImbalanceSnapshot], now: float
    ) -> float:
        """Compute exponential-decay-weighted average of imbalance ratios.

        Recent snapshots are weighted more heavily. The decay constant
        is derived from the configured half-life.
        """
        if not history:
            return 1.0

        lambda_decay = np.log(2) / max(self._decay_half_life_s, 1.0)

        weights = []
        ratios = []
        for snap in history:
            age_s = max(now - snap.timestamp, 0.0)
            w = np.exp(-lambda_decay * age_s)
            weights.append(w)
            ratios.append(snap.ratio)

        weights = np.array(weights)
        ratios = np.array(ratios)

        total_w = weights.sum()
        if total_w < 1e-12:
            return 1.0

        return float((weights * ratios).sum() / total_w)

    def _estimate_edge(
        self, imbalance_ratio: float, mid_price: float, direction: str
    ) -> float:
        """Estimate expected edge from imbalance magnitude.

        Empirical model: edge scales logarithmically with imbalance,
        modulated by distance from 0.5 (more room to move near extremes
        of the 0-1 range is limited).
        """
        # Logarithmic scaling: large imbalances have diminishing marginal impact
        log_ratio = np.log(max(imbalance_ratio, 1.01))

        # Base edge: ~2% per unit of log-imbalance (calibrated empirically)
        base_edge = 0.02 * log_ratio

        # Room-to-move adjustment: less room near 0 or 1
        if direction == "up":
            room = 1.0 - mid_price  # room for price to go up
        else:
            room = mid_price  # room for price to go down

        # Edge can't exceed available room, scaled conservatively
        edge = base_edge * min(room * 2, 1.0)

        return max(edge, 0.0)
