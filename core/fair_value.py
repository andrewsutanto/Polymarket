"""Fair value providers for market-agnostic pricing.

Provides simple fair-value estimates from structural market data alone.
No external oracle required for the primary strategies.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def structural_fair_value(yes_price: float, no_price: float) -> dict[str, float]:
    """Compute fair values from the structural relationship YES + NO = 1.

    If YES + NO > 1, there's overpricing (vig).
    If YES + NO < 1, there's underpricing (rare but tradeable).

    Args:
        yes_price: Current YES token price.
        no_price: Current NO token price.

    Returns:
        Dict with fair_yes, fair_no, vig, and mispricing.
    """
    total = yes_price + no_price
    if total <= 0:
        return {"fair_yes": 0.5, "fair_no": 0.5, "vig": 0.0, "mispricing": 0.0}

    fair_yes = yes_price / total
    fair_no = no_price / total
    vig = total - 1.0
    mispricing = abs(vig)

    return {
        "fair_yes": fair_yes,
        "fair_no": fair_no,
        "vig": vig,
        "mispricing": mispricing,
    }


def vwap_fair_value(
    bid_levels: list[tuple[float, float]],
    ask_levels: list[tuple[float, float]],
    depth_usd: float = 100.0,
) -> float:
    """Estimate fair value from VWAP across order book levels.

    Computes volume-weighted average price up to a given depth.

    Args:
        bid_levels: [(price, size)] from best to worst.
        ask_levels: [(price, size)] from best to worst.
        depth_usd: USD depth to consider for VWAP.

    Returns:
        VWAP fair value estimate.
    """
    total_value = 0.0
    total_size = 0.0

    for levels in [bid_levels, ask_levels]:
        remaining = depth_usd
        for price, size in levels:
            if remaining <= 0:
                break
            take = min(size, remaining / max(price, 0.001))
            total_value += price * take
            total_size += take
            remaining -= take * price

    if total_size <= 0:
        return 0.5
    return total_value / total_size


def complement_fair_value(
    own_price: float,
    complement_price: float,
) -> float:
    """Derive fair value from the complement token's price.

    In a binary market: fair_yes = 1 - no_price (and vice versa).
    If own_price diverges from (1 - complement), there's an opportunity.

    Args:
        own_price: This token's current price.
        complement_price: The other outcome's current price.

    Returns:
        Fair value implied by the complement.
    """
    return max(0.0, min(1.0, 1.0 - complement_price))
