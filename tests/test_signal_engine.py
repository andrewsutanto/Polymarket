"""Tests for the signal engine mispricing detection."""

import pytest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from core.signal_engine import SignalEngine, Signal
from core.weather_model import WeatherModel, BucketDef, BucketProbabilities
from core.polymarket_feed import PolymarketFeed, MarketSnapshot
from core.falcon_intel import FalconFeed, neutral_intel
from core.noaa_feed import NOAAFeed, WeatherForecast


def make_forecast(high: float = 74.0, agreement: bool = True) -> WeatherForecast:
    return WeatherForecast(
        timestamp=datetime.now(timezone.utc),
        location="NYC",
        target_date=date(2026, 4, 10),
        forecasted_high_f=high,
        forecasted_low_f=60.0,
        hourly_temps=[70, 72, 74, 73, 71],
        model_run_time=datetime.now(timezone.utc),
        forecast_shift_f=0.0,
        nws_confidence="",
        open_meteo_high_f=high if agreement else high + 5,
        model_agreement=agreement,
        update_count=1,
    )


def make_snapshot(
    contract_id: str = "c74",
    mid_price: float = 0.10,
    depth: float = 50.0,
    spread: float = 0.02,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=datetime.now(timezone.utc),
        contract_id=contract_id,
        location="NYC",
        target_date=date(2026, 4, 10),
        bucket_label="74-75°F",
        bucket_low_f=74,
        bucket_high_f=75,
        best_bid=mid_price - spread / 2,
        best_ask=mid_price + spread / 2,
        mid_price=mid_price,
        spread=spread,
        depth_usd=depth,
        volume_24h=100.0,
        time_to_resolution_hrs=48.0,
    )


@pytest.fixture
def engine():
    noaa = MagicMock(spec=NOAAFeed)
    model = WeatherModel()
    polymarket = MagicMock(spec=PolymarketFeed)
    falcon = MagicMock(spec=FalconFeed)
    falcon.is_enabled = False
    falcon.get_intel.return_value = neutral_intel("c74")

    buckets = [
        BucketDef("c72", "NYC", date(2026, 4, 10), 72, 73, "72-73°F"),
        BucketDef("c74", "NYC", date(2026, 4, 10), 74, 75, "74-75°F"),
        BucketDef("c76", "NYC", date(2026, 4, 10), 76, 77, "76-77°F"),
    ]
    model.register_buckets(buckets)

    eng = SignalEngine(noaa, model, polymarket, falcon)
    return eng, noaa, polymarket


class TestSignalDetection:
    def test_buy_signal_when_underpriced(self, engine):
        eng, noaa, pm = engine
        noaa.get_latest.return_value = make_forecast(high=74.5)
        pm.get_active_location_dates.return_value = {("NYC", date(2026, 4, 10))}
        pm.get_snapshot.return_value = make_snapshot(mid_price=0.08, depth=80.0)
        pm.get_all_snapshots.return_value = {}

        signals = eng.scan()
        buy_signals = [s for s in signals if s.signal_type == "BUY"]
        assert len(buy_signals) >= 1
        assert buy_signals[0].contract_id == "c74"
        assert buy_signals[0].edge > 0

    def test_no_signal_when_fairly_priced(self, engine):
        eng, noaa, pm = engine
        noaa.get_latest.return_value = make_forecast(high=74.5)
        pm.get_active_location_dates.return_value = {("NYC", date(2026, 4, 10))}
        pm.get_snapshot.return_value = make_snapshot(mid_price=0.40)

        signals = eng.scan()
        buy_signals = [s for s in signals if s.signal_type == "BUY"]
        assert len(buy_signals) == 0

    def test_sell_signal_when_above_exit(self, engine):
        eng, noaa, pm = engine
        eng.set_open_positions({"c74": 0.10})
        noaa.get_latest.return_value = make_forecast(high=74.5)
        pm.get_active_location_dates.return_value = {("NYC", date(2026, 4, 10))}
        pm.get_snapshot.return_value = make_snapshot(mid_price=0.50)

        signals = eng.scan()
        sell_signals = [s for s in signals if s.signal_type == "SELL"]
        assert len(sell_signals) >= 1

    def test_no_duplicate_buy_for_open_position(self, engine):
        eng, noaa, pm = engine
        eng.set_open_positions({"c74": 0.10})
        noaa.get_latest.return_value = make_forecast(high=74.5)
        pm.get_active_location_dates.return_value = {("NYC", date(2026, 4, 10))}
        pm.get_snapshot.return_value = make_snapshot(mid_price=0.08)

        signals = eng.scan()
        buy_signals = [s for s in signals if s.signal_type == "BUY" and s.contract_id == "c74"]
        assert len(buy_signals) == 0
