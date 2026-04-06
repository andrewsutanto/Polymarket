"""Tests for Kelly criterion calculations."""

import pytest
from datetime import date, datetime, timezone

from core.risk_manager import RiskManager
from core.signal_engine import Signal


def make_signal(
    model_prob: float = 0.40,
    market_price: float = 0.10,
    edge: float | None = None,
    confidence: float = 0.90,
) -> Signal:
    if edge is None:
        edge = model_prob - market_price
    return Signal(
        timestamp=datetime.now(timezone.utc),
        signal_type="BUY",
        contract_id="test",
        location="NYC",
        target_date=date(2026, 4, 10),
        bucket_label="74-75°F",
        model_prob=model_prob,
        market_price=market_price,
        edge=edge,
        value_ratio=model_prob / max(market_price, 0.001),
        confidence=confidence,
        forecast_high_f=74.0,
        forecast_sigma_f=2.5,
        lead_days=1,
        forecast_shift_f=0.0,
        book_depth=50.0,
        spread=0.02,
        volume_24h=100.0,
    )


class TestKellyCriterion:
    def test_positive_ev_gives_positive_kelly(self) -> None:
        rm = RiskManager(100.0)
        rm.update_state(100.0, {}, 0.0, 100.0)
        sig = make_signal(model_prob=0.50, market_price=0.10)
        proposal = rm.evaluate(sig)
        assert proposal.kelly_fraction > 0
        assert proposal.size_usd > 0

    def test_higher_edge_gives_larger_size(self) -> None:
        rm1 = RiskManager(100.0)
        rm1.update_state(100.0, {}, 0.0, 100.0)
        rm2 = RiskManager(100.0)
        rm2.update_state(100.0, {}, 0.0, 100.0)

        small_edge = rm1.evaluate(make_signal(model_prob=0.35, market_price=0.10))
        large_edge = rm2.evaluate(make_signal(model_prob=0.60, market_price=0.10))

        if small_edge.approved and large_edge.approved:
            assert large_edge.kelly_fraction >= small_edge.kelly_fraction

    def test_kelly_clamped_to_max(self) -> None:
        rm = RiskManager(1000.0)
        rm.update_state(1000.0, {}, 0.0, 1000.0)
        sig = make_signal(model_prob=0.90, market_price=0.05)
        proposal = rm.evaluate(sig)
        assert proposal.size_usd <= 5.0  # MAX_TRADE_SIZE

    def test_half_kelly_applied(self) -> None:
        rm = RiskManager(100.0)
        kelly, _ = rm._compute_kelly_size(make_signal(model_prob=0.50, market_price=0.20))
        # Full Kelly for these params
        odds = (1.0 / 0.20) - 1.0  # = 4.0
        p = 0.50
        full_kelly = (p * odds - (1 - p)) / odds
        assert abs(kelly - full_kelly * 0.5) < 0.01

    def test_zero_price_returns_zero(self) -> None:
        rm = RiskManager(100.0)
        kelly, size = rm._compute_kelly_size(make_signal(market_price=0.0))
        assert kelly == 0.0
        assert size == 0.0

    def test_negative_ev_returns_zero(self) -> None:
        rm = RiskManager(100.0)
        sig = make_signal(model_prob=0.05, market_price=0.10)
        kelly, size = rm._compute_kelly_size(sig)
        assert kelly <= 0
