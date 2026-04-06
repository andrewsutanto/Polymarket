"""Tests for the risk manager: entry filters, sizing, kill switch."""

import pytest
from datetime import date, datetime, timezone

from config import settings
from core.risk_manager import RiskManager
from core.signal_engine import Signal


def make_signal(
    edge: float = 0.30,
    model_prob: float = 0.40,
    market_price: float = 0.10,
    confidence: float = 0.90,
    contract_id: str = "c74",
    signal_type: str = "BUY",
) -> Signal:
    return Signal(
        timestamp=datetime.now(timezone.utc),
        signal_type=signal_type,
        contract_id=contract_id,
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


@pytest.fixture
def risk() -> RiskManager:
    rm = RiskManager(starting_capital=50.0)
    rm.update_state(
        available_cash=50.0,
        open_positions={},
        session_pnl=0.0,
        portfolio_value=50.0,
    )
    return rm


class TestEntryFilters:
    def test_approved_good_signal(self, risk: RiskManager) -> None:
        sig = make_signal()
        proposal = risk.evaluate(sig)
        assert proposal.approved
        assert proposal.size_usd >= settings.MIN_TRADE_SIZE

    def test_reject_low_confidence(self, risk: RiskManager) -> None:
        sig = make_signal(confidence=0.50)
        proposal = risk.evaluate(sig)
        assert not proposal.approved
        assert "Confidence" in proposal.reject_reason

    def test_reject_max_positions(self, risk: RiskManager) -> None:
        positions = {f"c{i}": 3.0 for i in range(settings.MAX_OPEN_POSITIONS)}
        risk.update_state(50.0, positions, 0.0, 50.0)
        sig = make_signal(contract_id="new_contract")
        proposal = risk.evaluate(sig)
        assert not proposal.approved
        assert "Max open positions" in proposal.reject_reason

    def test_reject_max_trades_per_cycle(self, risk: RiskManager) -> None:
        for i in range(settings.MAX_TRADES_PER_RUN):
            sig = make_signal(contract_id=f"c{i}")
            risk.evaluate(sig)

        sig = make_signal(contract_id="one_more")
        proposal = risk.evaluate(sig)
        assert not proposal.approved
        assert "Max trades per cycle" in proposal.reject_reason

    def test_cycle_reset_allows_new_trades(self, risk: RiskManager) -> None:
        for i in range(settings.MAX_TRADES_PER_RUN):
            risk.evaluate(make_signal(contract_id=f"c{i}"))

        risk.reset_cycle()
        proposal = risk.evaluate(make_signal(contract_id="fresh"))
        assert proposal.approved


class TestKillSwitch:
    def test_kill_switch_blocks_all_trades(self, risk: RiskManager) -> None:
        risk.activate_kill_switch()
        sig = make_signal()
        proposal = risk.evaluate(sig)
        assert not proposal.approved
        assert "Kill switch" in proposal.reject_reason

    def test_drawdown_triggers_kill_switch(self, risk: RiskManager) -> None:
        triggered = []
        risk.on_kill_switch(lambda: triggered.append(True))
        risk.update_state(50.0, {}, 0.0, 50.0)
        risk.update_state(50.0, {}, -15.0, 35.0)  # 30% drawdown
        assert risk.kill_switch_active
        assert triggered

    def test_drawdown_warning_callbacks(self, risk: RiskManager) -> None:
        warnings = []
        risk.on_drawdown_warning(lambda d: warnings.append(d))
        risk.update_state(50.0, {}, 0.0, 50.0)
        risk.update_state(50.0, {}, -6.0, 44.0)  # 12% drawdown
        assert len(warnings) >= 1


class TestPositionSizing:
    def test_size_within_bounds(self, risk: RiskManager) -> None:
        sig = make_signal()
        proposal = risk.evaluate(sig)
        assert proposal.approved
        assert settings.MIN_TRADE_SIZE <= proposal.size_usd <= settings.MAX_TRADE_SIZE

    def test_sell_approved_for_open_position(self, risk: RiskManager) -> None:
        risk.update_state(40.0, {"c74": 5.0}, 0.0, 45.0)
        sig = make_signal(signal_type="SELL")
        proposal = risk.evaluate(sig)
        assert proposal.approved
        assert proposal.size_usd == 5.0

    def test_sell_rejected_no_position(self, risk: RiskManager) -> None:
        sig = make_signal(signal_type="SELL", contract_id="nonexistent")
        proposal = risk.evaluate(sig)
        assert not proposal.approved
