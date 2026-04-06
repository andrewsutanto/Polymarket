"""Cross-market consistency arbitrage.

Detects logically related markets with contradictory pricing. E.g.,
"Trump wins" at 40% but "Republican wins" at 35% is a logical
impossibility — Trump winning implies Republican winning.
"""

from __future__ import annotations

import logging
from typing import Any

from strategies.base import BaseStrategy, Signal
from core.relationship_graph import MarketRelationship, detect_contradiction

logger = logging.getLogger(__name__)


class CrossMarketArbStrategy(BaseStrategy):
    """Trade contradictions between logically related markets."""

    def __init__(
        self,
        min_gap: float = 0.05,
        min_confidence: float = 0.5,
        exit_gap: float = 0.02,
    ) -> None:
        self._min_gap = min_gap
        self._min_confidence = min_confidence
        self._exit_gap = exit_gap

    @property
    def name(self) -> str:
        return "cross_market_arb"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Detect pricing contradictions between related markets.

        Additional market_data keys:
            relationships: list[MarketRelationship] for this market
            outcome_prices: list of prices for all outcomes
        """
        relationships: list[MarketRelationship] = market_data.get("relationships", [])
        if not relationships:
            return None

        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")
        mid_price = market_data.get("mid_price", 0.0)
        yes_price = market_data.get("yes_price", 0.0)

        # Exit if holding and gap has closed
        if market_data.get("has_position"):
            for rel in relationships:
                is_contradiction, gap = detect_contradiction(rel, yes_price, 0.5)
                if gap <= self._exit_gap:
                    return Signal(
                        direction="SELL",
                        strength=0.6,
                        edge=gap,
                        strategy_name=self.name,
                        market_id=market_id,
                        token_id=token_id,
                        market_slug=slug,
                        category=category,
                        outcome=outcome,
                        metadata={"exit_reason": "gap_closed"},
                    )
            return None

        # Check each relationship for contradictions
        best_signal: Signal | None = None
        best_gap = 0.0

        for rel in relationships:
            if rel.confidence < self._min_confidence:
                continue

            # For subset relationships: the subset should be <= superset
            if rel.relationship_type == "subset":
                # If this market is the subset and it's priced higher than expected
                if rel.market_a_id == market_id:
                    # We are the subset — should be cheaper or equal
                    is_contradiction, gap = detect_contradiction(rel, yes_price, 0.5)
                    if is_contradiction and gap >= self._min_gap and gap > best_gap:
                        best_gap = gap
                        best_signal = Signal(
                            direction="SELL",
                            strength=min(gap / 0.15, 1.0),
                            edge=gap,
                            strategy_name=self.name,
                            market_id=market_id,
                            token_id=token_id,
                            market_slug=slug,
                            category=category,
                            outcome=outcome,
                            metadata={
                                "relationship": rel.relationship_type,
                                "related_market": rel.market_b_id,
                                "gap": gap,
                                "description": rel.description,
                            },
                        )

            elif rel.relationship_type == "temporal":
                # Later deadline should have higher probability
                is_contradiction, gap = detect_contradiction(rel, yes_price, 0.5)
                if is_contradiction and gap >= self._min_gap and gap > best_gap:
                    best_gap = gap
                    best_signal = Signal(
                        direction="BUY",
                        strength=min(gap / 0.15, 1.0),
                        edge=gap,
                        strategy_name=self.name,
                        market_id=market_id,
                        token_id=token_id,
                        market_slug=slug,
                        category=category,
                        outcome=outcome,
                        metadata={
                            "relationship": rel.relationship_type,
                            "related_market": rel.market_b_id if rel.market_a_id == market_id else rel.market_a_id,
                            "gap": gap,
                        },
                    )

        return best_signal

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_gap": self._min_gap,
            "min_confidence": self._min_confidence,
            "exit_gap": self._exit_gap,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            if hasattr(self, f"_{k}"):
                setattr(self, f"_{k}", v)
