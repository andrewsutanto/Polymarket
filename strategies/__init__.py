"""Multi-strategy Polymarket arbitrage framework.

Market-agnostic strategies that scan the full Polymarket universe
for structural mispricing, cross-market contradictions, volume
divergences, stale markets, and line movement.
"""

from strategies.base import BaseStrategy, Signal
from strategies.implied_prob_arb import ImpliedProbArbStrategy
from strategies.cross_market_arb import CrossMarketArbStrategy
from strategies.volume_divergence import VolumeDivergenceStrategy
from strategies.stale_market import StaleMarketStrategy
from strategies.line_movement import LineMovementStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.orderbook_imbalance import OrderbookImbalanceStrategy
from strategies.temporal_bias import TemporalBiasStrategy

ALL_STRATEGIES = {
    "implied_prob_arb": ImpliedProbArbStrategy,
    "cross_market_arb": CrossMarketArbStrategy,
    "volume_divergence": VolumeDivergenceStrategy,
    "stale_market": StaleMarketStrategy,
    "line_movement": LineMovementStrategy,
    "mean_reversion": MeanReversionStrategy,
    "orderbook_imbalance": OrderbookImbalanceStrategy,
    "temporal_bias": TemporalBiasStrategy,
}

__all__ = [
    "BaseStrategy",
    "Signal",
    "ImpliedProbArbStrategy",
    "CrossMarketArbStrategy",
    "VolumeDivergenceStrategy",
    "StaleMarketStrategy",
    "LineMovementStrategy",
    "MeanReversionStrategy",
    "EnsembleStrategy",
    "OrderbookImbalanceStrategy",
    "TemporalBiasStrategy",
    "ALL_STRATEGIES",
]
