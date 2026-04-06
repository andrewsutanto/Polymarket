"""Tests for ensemble strategy — verify agreement/conflict handling."""

import pytest
from typing import Any

from strategies.base import BaseStrategy, Signal
from strategies.ensemble import EnsembleStrategy


class StubBuyStrategy(BaseStrategy):
    def __init__(self, edge: float = 0.10):
        self._edge = edge

    @property
    def name(self) -> str:
        return "stub_buy"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        return Signal("BUY", 0.8, self._edge, self.name, "test", "NYC")

    def get_parameters(self) -> dict:
        return {"edge": self._edge}

    def set_parameters(self, params: dict) -> None:
        if "edge" in params:
            self._edge = params["edge"]


class StubSellStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "stub_sell"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        return Signal("SELL", 0.8, 0.10, self.name, "test", "NYC")

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, params: dict) -> None:
        pass


class StubNoneStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "stub_none"

    def generate_signal(self, market_data: dict[str, Any]) -> Signal | None:
        return None

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, params: dict) -> None:
        pass


class TestEnsemble:
    def test_agreement_boosts_signal(self):
        ens = EnsembleStrategy(
            strategies=[StubBuyStrategy(), StubBuyStrategy()],
            min_agreement=2,
        )
        sig = ens.generate_signal({"market_id": "test", "city": "NYC"})
        assert sig is not None
        assert sig.direction == "BUY"
        assert sig.metadata["n_aligned"] == 2

    def test_conflict_reduces_or_skips(self):
        ens = EnsembleStrategy(
            strategies=[StubBuyStrategy(edge=0.03), StubSellStrategy()],
            min_agreement=2,
            solo_edge_threshold=0.08,
        )
        sig = ens.generate_signal({"market_id": "test", "city": "NYC"})
        # With low edge and conflict, should be None or solo
        assert sig is None  # Both have same strength, neither > solo threshold

    def test_solo_high_conviction_passes(self):
        ens = EnsembleStrategy(
            strategies=[StubBuyStrategy(edge=0.15), StubNoneStrategy()],
            min_agreement=2,
            solo_edge_threshold=0.08,
        )
        sig = ens.generate_signal({"market_id": "test", "city": "NYC"})
        assert sig is not None
        assert sig.metadata.get("source") == "solo_high_conviction"

    def test_no_strategies_returns_none(self):
        ens = EnsembleStrategy(strategies=[])
        assert ens.generate_signal({"market_id": "test", "city": "NYC"}) is None

    def test_all_none_returns_none(self):
        ens = EnsembleStrategy(strategies=[StubNoneStrategy(), StubNoneStrategy()])
        assert ens.generate_signal({"market_id": "test", "city": "NYC"}) is None

    def test_regime_adaptive_weights(self):
        ens = EnsembleStrategy(strategies=[StubBuyStrategy(), StubBuyStrategy()])
        ens.set_regime_weights({
            "ACTIVE_WEATHER": {"stub_buy": 0.8},
            "STABLE": {"stub_buy": 0.2},
        })
        sig = ens.generate_signal({
            "market_id": "test", "city": "NYC", "regime": "ACTIVE_WEATHER",
        })
        assert sig is not None
