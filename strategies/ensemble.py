"""Ensemble strategy — blends signals with regime-adaptive weights.

Combines signals from all active strategies using configurable weights.
Boosts confidence when strategies agree, reduces when they conflict.
"""

from __future__ import annotations

import logging
from typing import Any

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "forecast_arb": 0.35,
    "mean_reversion": 0.25,
    "forecast_momentum": 0.25,
    "cross_city_arb": 0.15,
}


class EnsembleStrategy(BaseStrategy):
    """Blend multiple strategy signals with configurable weights."""

    def __init__(
        self,
        strategies: list[BaseStrategy] | None = None,
        weights: dict[str, float] | None = None,
        min_agreement: int = 2,
        solo_edge_threshold: float = 0.08,
    ) -> None:
        self._strategies = strategies or []
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._min_agreement = min_agreement
        self._solo_edge_threshold = solo_edge_threshold
        self._regime_weights: dict[str, dict[str, float]] | None = None

    @property
    def name(self) -> str:
        return "ensemble"

    def set_regime_weights(self, mapping: dict[str, dict[str, float]]) -> None:
        """Set regime-to-weight mapping for adaptive weighting.

        Args:
            mapping: {regime_name: {strategy_name: weight}}
        """
        self._regime_weights = mapping

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self._strategies.append(strategy)

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        """Collect signals from all strategies and blend.

        Additional market_data keys:
            regime: current regime name (for adaptive weights)
        """
        if not self._strategies:
            return None

        regime = market_data.get("regime", "NEUTRAL")
        weights = self._get_weights(regime)

        # Collect signals
        votes: list[tuple[Signal, float]] = []
        for strat in self._strategies:
            try:
                sig = strat.generate_signal(market_data)
                if sig is not None:
                    w = weights.get(strat.name, 0.25)
                    votes.append((sig, w))
            except Exception as exc:
                logger.warning("Strategy %s error: %s", strat.name, exc)

        if not votes:
            return None

        market_id = market_data.get("market_id", "")
        city = market_data.get("city", "")

        # Count direction agreement
        buy_votes = [(s, w) for s, w in votes if s.direction == "BUY"]
        sell_votes = [(s, w) for s, w in votes if s.direction == "SELL"]

        # Determine direction
        buy_weight = sum(w * s.strength for s, w in buy_votes)
        sell_weight = sum(w * s.strength for s, w in sell_votes)

        if buy_weight > sell_weight:
            direction = "BUY"
            aligned = buy_votes
            opposing = sell_votes
        elif sell_weight > buy_weight:
            direction = "SELL"
            aligned = sell_votes
            opposing = buy_votes
        else:
            return None

        n_aligned = len(aligned)

        # Meta-filter: require agreement or high-conviction solo
        if n_aligned < self._min_agreement:
            # Check for high-conviction solo signal
            best = max(aligned, key=lambda x: x[0].edge)
            if best[0].edge < self._solo_edge_threshold:
                return None
            # Allow solo high-conviction through
            return Signal(
                direction=direction,
                strength=best[0].strength * 0.7,
                edge=best[0].edge,
                strategy_name=self.name,
                market_id=market_id,
                city=city,
                metadata={
                    "source": "solo_high_conviction",
                    "contributing_strategy": best[0].strategy_name,
                    "votes": {s.strategy_name: s.direction for s, _ in votes},
                },
            )

        # Compute blended signal
        total_weight = sum(w for _, w in aligned)
        if total_weight <= 0:
            return None

        blended_edge = sum(s.edge * w for s, w in aligned) / total_weight
        blended_strength = sum(s.strength * w for s, w in aligned) / total_weight

        # Boost if multiple agree, penalize if opposition
        agreement_boost = min(n_aligned / len(self._strategies), 1.0) * 0.2
        opposition_penalty = len(opposing) / max(len(votes), 1) * 0.15
        blended_strength = min(blended_strength + agreement_boost - opposition_penalty, 1.0)

        return Signal(
            direction=direction,
            strength=max(blended_strength, 0.0),
            edge=blended_edge,
            strategy_name=self.name,
            market_id=market_id,
            city=city,
            metadata={
                "n_aligned": n_aligned,
                "n_opposing": len(opposing),
                "regime": regime,
                "votes": {s.strategy_name: s.direction for s, _ in votes},
                "vote_edges": {s.strategy_name: s.edge for s, _ in votes},
                "weights_used": weights,
            },
        )

    def _get_weights(self, regime: str) -> dict[str, float]:
        if self._regime_weights and regime in self._regime_weights:
            return self._regime_weights[regime]
        return self._weights

    def get_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "weights": dict(self._weights),
            "min_agreement": self._min_agreement,
            "solo_edge_threshold": self._solo_edge_threshold,
        }
        for strat in self._strategies:
            params[f"sub_{strat.name}"] = strat.get_parameters()
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        if "weights" in params:
            self._weights.update(params["weights"])
        if "min_agreement" in params:
            self._min_agreement = params["min_agreement"]
        if "solo_edge_threshold" in params:
            self._solo_edge_threshold = params["solo_edge_threshold"]
        for strat in self._strategies:
            sub_key = f"sub_{strat.name}"
            if sub_key in params:
                strat.set_parameters(params[sub_key])

    def reset(self) -> None:
        for strat in self._strategies:
            strat.reset()
