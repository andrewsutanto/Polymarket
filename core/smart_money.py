"""Smart money detector — track large orders and institutional flow.

Monitors trade data from the Polymarket CLOB API to identify large
orders (> configurable USD threshold) and computes the net directional
pressure from these "smart money" participants.

Key insight: Large orders on Polymarket are disproportionately placed
by informed traders (market makers, quant funds, political insiders).
When smart money flow disagrees with the current price, this is a
strong alpha signal.

Signals:
- Net large-order flow is heavily BUY-side but price is low => BUY
- Net large-order flow is heavily SELL-side but price is high => SELL
- Flow agrees with price => no signal (already priced in)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LargeOrder:
    """A single large order observation."""

    timestamp: float
    side: str          # "BUY" or "SELL"
    size_usd: float    # Order size in USD
    price: float       # Execution price
    market_id: str
    token_id: str


@dataclass
class SmartMoneyFlow:
    """Aggregated smart money metrics for a market."""

    market_id: str
    net_flow_usd: float         # Positive = net buying, negative = net selling
    buy_volume_usd: float
    sell_volume_usd: float
    n_large_buys: int
    n_large_sells: int
    flow_imbalance: float       # -1 to 1 (normalized directional pressure)
    vwap_buy: float             # Volume-weighted average buy price
    vwap_sell: float            # Volume-weighted average sell price
    avg_order_size: float
    last_updated: float


class SmartMoneyDetector:
    """Detect and track smart money flow from CLOB trade data.

    Parameters:
        large_order_threshold_usd: Minimum USD size to classify as "large" (default $1000).
        flow_window_s: Lookback window in seconds for flow computation (default 3600 = 1hr).
        max_orders_tracked: Maximum large orders retained per market.
        min_large_orders: Minimum large orders needed to compute a signal.
        flow_disagreement_threshold: Minimum |flow_imbalance| to consider
            disagreement with price (default 0.4).
        price_disagreement_threshold: Minimum |flow_direction - price_direction|
            for a signal (default 0.15).
        decay_half_life_s: Half-life for time-weighting recent orders more heavily.
        min_edge: Minimum estimated edge to emit.
    """

    def __init__(
        self,
        large_order_threshold_usd: float = 1000.0,
        flow_window_s: float = 3600.0,
        max_orders_tracked: int = 200,
        min_large_orders: int = 3,
        flow_disagreement_threshold: float = 0.4,
        price_disagreement_threshold: float = 0.15,
        decay_half_life_s: float = 1800.0,
        min_edge: float = 0.02,
    ) -> None:
        self._threshold_usd = large_order_threshold_usd
        self._flow_window_s = flow_window_s
        self._max_orders = max_orders_tracked
        self._min_large_orders = min_large_orders
        self._flow_disagree_thresh = flow_disagreement_threshold
        self._price_disagree_thresh = price_disagreement_threshold
        self._decay_half_life_s = decay_half_life_s
        self._min_edge = min_edge

        # Per-market order history
        self._orders: dict[str, deque[LargeOrder]] = {}

    def ingest_trade(
        self,
        market_id: str,
        token_id: str,
        side: str,
        size_usd: float,
        price: float,
        timestamp: float | None = None,
    ) -> bool:
        """Ingest a single trade and classify it.

        Returns True if the trade was classified as a large order.
        """
        if size_usd < self._threshold_usd:
            return False

        if timestamp is None:
            timestamp = time.time()

        order = LargeOrder(
            timestamp=timestamp,
            side=side.upper(),
            size_usd=size_usd,
            price=price,
            market_id=market_id,
            token_id=token_id,
        )

        if market_id not in self._orders:
            self._orders[market_id] = deque(maxlen=self._max_orders)

        self._orders[market_id].append(order)

        logger.debug(
            "Large %s order: $%.0f at %.3f on %s",
            side,
            size_usd,
            price,
            market_id,
        )
        return True

    def ingest_trades_batch(
        self,
        market_id: str,
        token_id: str,
        trades: list[dict[str, Any]],
    ) -> int:
        """Ingest a batch of trades from the CLOB API.

        Each trade dict should have: side, size (USD), price, timestamp.
        Returns count of large orders detected.
        """
        count = 0
        for trade in trades:
            was_large = self.ingest_trade(
                market_id=market_id,
                token_id=token_id,
                side=trade.get("side", "BUY"),
                size_usd=float(trade.get("size", 0)),
                price=float(trade.get("price", 0)),
                timestamp=trade.get("timestamp"),
            )
            if was_large:
                count += 1
        return count

    def compute_flow(
        self,
        market_id: str,
        now: float | None = None,
    ) -> SmartMoneyFlow | None:
        """Compute aggregated smart money flow for a market.

        Returns None if insufficient data.
        """
        if market_id not in self._orders:
            return None

        if now is None:
            now = time.time()

        orders = self._orders[market_id]
        cutoff = now - self._flow_window_s
        lambda_decay = np.log(2) / max(self._decay_half_life_s, 1.0)

        buy_vol = 0.0
        sell_vol = 0.0
        buy_price_vol = 0.0
        sell_price_vol = 0.0
        n_buys = 0
        n_sells = 0

        for order in orders:
            if order.timestamp < cutoff:
                continue

            age = max(now - order.timestamp, 0.0)
            weight = np.exp(-lambda_decay * age)
            weighted_size = order.size_usd * weight

            if order.side == "BUY":
                buy_vol += weighted_size
                buy_price_vol += order.price * weighted_size
                n_buys += 1
            else:
                sell_vol += weighted_size
                sell_price_vol += order.price * weighted_size
                n_sells += 1

        total_orders = n_buys + n_sells
        if total_orders < self._min_large_orders:
            return None

        net_flow = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol

        # Flow imbalance: -1 (all sells) to +1 (all buys)
        flow_imbalance = net_flow / max(total_vol, 1e-8)

        vwap_buy = buy_price_vol / max(buy_vol, 1e-8) if n_buys > 0 else 0.0
        vwap_sell = sell_price_vol / max(sell_vol, 1e-8) if n_sells > 0 else 0.0

        return SmartMoneyFlow(
            market_id=market_id,
            net_flow_usd=round(net_flow, 2),
            buy_volume_usd=round(buy_vol, 2),
            sell_volume_usd=round(sell_vol, 2),
            n_large_buys=n_buys,
            n_large_sells=n_sells,
            flow_imbalance=round(flow_imbalance, 4),
            vwap_buy=round(vwap_buy, 4),
            vwap_sell=round(vwap_sell, 4),
            avg_order_size=round(total_vol / max(total_orders, 1), 2),
            last_updated=now,
        )

    def get_signal(
        self,
        market_id: str,
        current_price: float,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Generate a trading signal when smart money disagrees with price.

        Returns a signal dict compatible with AlphaCombiner, or None.
        """
        flow = self.compute_flow(market_id, now)
        if flow is None:
            return None

        imbalance = flow.flow_imbalance

        # Determine if flow disagrees with price
        # Price > 0.5 => market expects YES. Flow < 0 => smart money sells.
        price_lean = current_price - 0.5  # positive = market expects YES
        flow_lean = imbalance              # positive = smart money buying

        # Disagreement: flow and price lean in opposite directions
        # AND both are significant enough
        if abs(imbalance) < self._flow_disagree_thresh:
            return None

        disagreement = -price_lean * flow_lean
        if disagreement <= 0:
            # Flow agrees with price — no edge
            return None

        # Edge estimate: proportional to flow strength and price distance
        # from 0.5 (more room for correction near extremes is limited)
        edge = abs(imbalance) * abs(price_lean) * 0.4

        # VWAP-based edge: if smart money is buying at prices below current,
        # they see value. If selling above current, they see overpricing.
        if flow_lean > 0 and flow.vwap_buy > 0:
            vwap_edge = current_price - flow.vwap_buy
            edge += max(vwap_edge, 0) * 0.3
        elif flow_lean < 0 and flow.vwap_sell > 0:
            vwap_edge = flow.vwap_sell - current_price
            edge += max(vwap_edge, 0) * 0.3

        if edge < self._min_edge:
            return None

        # Direction follows smart money
        direction = "BUY" if flow_lean > 0 else "SELL"
        strength = min(abs(imbalance), 1.0)

        return {
            "direction": direction,
            "edge": round(edge, 4),
            "strength": round(strength, 4),
            "flow": {
                "net_flow_usd": flow.net_flow_usd,
                "buy_volume_usd": flow.buy_volume_usd,
                "sell_volume_usd": flow.sell_volume_usd,
                "flow_imbalance": flow.flow_imbalance,
                "n_large_buys": flow.n_large_buys,
                "n_large_sells": flow.n_large_sells,
                "vwap_buy": flow.vwap_buy,
                "vwap_sell": flow.vwap_sell,
            },
            "disagreement": round(disagreement, 4),
            "current_price": current_price,
        }

    def get_all_flows(self, now: float | None = None) -> dict[str, SmartMoneyFlow]:
        """Compute flow for all tracked markets."""
        flows = {}
        for market_id in self._orders:
            flow = self.compute_flow(market_id, now)
            if flow is not None:
                flows[market_id] = flow
        return flows

    def clear_market(self, market_id: str) -> None:
        """Clear order history for a resolved market."""
        self._orders.pop(market_id, None)

    def reset(self) -> None:
        """Clear all tracked data."""
        self._orders.clear()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "large_order_threshold_usd": self._threshold_usd,
            "flow_window_s": self._flow_window_s,
            "max_orders_tracked": self._max_orders,
            "min_large_orders": self._min_large_orders,
            "flow_disagreement_threshold": self._flow_disagree_thresh,
            "price_disagreement_threshold": self._price_disagree_thresh,
            "decay_half_life_s": self._decay_half_life_s,
            "min_edge": self._min_edge,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        mapping = {
            "large_order_threshold_usd": "_threshold_usd",
            "flow_window_s": "_flow_window_s",
            "max_orders_tracked": "_max_orders",
            "min_large_orders": "_min_large_orders",
            "flow_disagreement_threshold": "_flow_disagree_thresh",
            "price_disagreement_threshold": "_price_disagree_thresh",
            "decay_half_life_s": "_decay_half_life_s",
            "min_edge": "_min_edge",
        }
        for k, v in params.items():
            attr = mapping.get(k)
            if attr and hasattr(self, attr):
                setattr(self, attr, v)
