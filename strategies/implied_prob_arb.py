"""Implied probability arbitrage — structural mispricing detection.

When YES + NO prices don't sum to ~1.0, there's either excessive vig
(overpriced) or a structural discount (underpriced). The simplest and
most reliable strategy since it requires no external oracle.
"""

from __future__ import annotations

import logging
from typing import Any

from strategies.base import BaseStrategy, Signal
from core.fair_value import structural_fair_value

logger = logging.getLogger(__name__)


class ImpliedProbArbStrategy(BaseStrategy):
    """Trade when implied probabilities are structurally mispriced."""

    def __init__(
        self,
        min_edge: float = 0.03,
        max_vig: float = 0.04,
        min_liquidity: float = 200.0,
        exit_edge: float = 0.01,
    ) -> None:
        self._min_edge = min_edge
        self._max_vig = max_vig
        self._min_liquidity = min_liquidity
        self._exit_edge = exit_edge

    @property
    def name(self) -> str:
        return "implied_prob_arb"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Detect structural mispricing from YES/NO price imbalance.

        Trades when one side is significantly cheaper than its fair value
        implied by the complement.
        """
        yes_price = market_data.get("yes_price", 0.0)
        no_price = market_data.get("no_price", 0.0)
        liquidity = market_data.get("liquidity", 0.0)
        mid_price = market_data.get("mid_price", 0.0)
        market_id = market_data.get("market_id", "")
        token_id = market_data.get("token_id", "")
        slug = market_data.get("market_slug", "")
        category = market_data.get("category", "other")
        outcome = market_data.get("outcome", "")

        if yes_price <= 0 or no_price <= 0:
            return None
        if liquidity < self._min_liquidity:
            return None

        fv = structural_fair_value(yes_price, no_price)
        vig = fv["vig"]
        fair_yes = fv["fair_yes"]
        fair_no = fv["fair_no"]

        # Exit if holding and edge has collapsed
        if market_data.get("has_position"):
            if abs(vig) <= self._exit_edge:
                return Signal(
                    direction="SELL",
                    strength=0.6,
                    edge=abs(vig),
                    strategy_name=self.name,
                    market_id=market_id,
                    token_id=token_id,
                    market_slug=slug,
                    category=category,
                    outcome=outcome,
                    metadata={"exit_reason": "vig_collapsed", "vig": vig},
                )
            return None

        # If total > 1 + max_vig, both sides are overpriced — no trade
        if vig > self._max_vig:
            return None

        # If total < 1 - min_edge, both sides are underpriced — buy the cheaper one
        if vig < -self._min_edge:
            # Buy whichever side has more edge
            yes_edge = fair_yes - yes_price
            no_edge = fair_no - no_price

            if yes_edge > no_edge and yes_edge >= self._min_edge:
                return Signal(
                    direction="BUY",
                    strength=min(yes_edge / 0.10, 1.0),
                    edge=yes_edge,
                    strategy_name=self.name,
                    market_id=market_id,
                    token_id=token_id,
                    market_slug=slug,
                    category=category,
                    outcome="Yes",
                    metadata={
                        "vig": vig,
                        "fair_yes": fair_yes,
                        "fair_no": fair_no,
                        "yes_edge": yes_edge,
                        "no_edge": no_edge,
                    },
                )
            elif no_edge >= self._min_edge:
                return Signal(
                    direction="BUY",
                    strength=min(no_edge / 0.10, 1.0),
                    edge=no_edge,
                    strategy_name=self.name,
                    market_id=market_id,
                    token_id=token_id,
                    market_slug=slug,
                    category=category,
                    outcome="No",
                    metadata={
                        "vig": vig,
                        "fair_yes": fair_yes,
                        "fair_no": fair_no,
                        "yes_edge": yes_edge,
                        "no_edge": no_edge,
                    },
                )

        # Check individual side mispricing (one side cheaper than complement implies)
        complement = market_data.get("complement_price", 1.0 - mid_price)
        implied_fair = 1.0 - complement
        edge = implied_fair - mid_price

        if edge >= self._min_edge and mid_price < 0.5:
            return Signal(
                direction="BUY",
                strength=min(edge / 0.10, 1.0),
                edge=edge,
                strategy_name=self.name,
                market_id=market_id,
                token_id=token_id,
                market_slug=slug,
                category=category,
                outcome=outcome,
                metadata={"implied_fair": implied_fair, "complement": complement},
            )

        return None

    def get_parameters(self) -> dict[str, Any]:
        return {
            "min_edge": self._min_edge,
            "max_vig": self._max_vig,
            "min_liquidity": self._min_liquidity,
            "exit_edge": self._exit_edge,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            if hasattr(self, f"_{k}"):
                setattr(self, f"_{k}", v)
