"""Multi-strategy weather arbitrage framework.

Provides a BaseStrategy interface and four weather-specific implementations:
forecast arbitrage, mean reversion, forecast momentum, and cross-city arb.
An ensemble combiner blends signals with regime-adaptive weights.
"""

from strategies.base import BaseStrategy, Signal
from strategies.forecast_arb import ForecastArbStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.forecast_momentum import ForecastMomentumStrategy
from strategies.cross_city_arb import CrossCityArbStrategy
from strategies.ensemble import EnsembleStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "ForecastArbStrategy",
    "MeanReversionStrategy",
    "ForecastMomentumStrategy",
    "CrossCityArbStrategy",
    "EnsembleStrategy",
]
