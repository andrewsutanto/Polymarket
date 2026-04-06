"""Mispricing detection and edge calculation engine.

Compares NOAA-derived bucket probabilities against Polymarket prices.
Emits BUY/SELL signals when thresholds are met.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Any

from config import settings
from core.weather_model import WeatherModel, BucketProbabilities
from core.polymarket_feed import PolymarketFeed, MarketSnapshot
from core.falcon_intel import FalconFeed, FalconIntel, neutral_intel
from core.noaa_feed import NOAAFeed

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    timestamp: datetime
    signal_type: str
    contract_id: str
    location: str
    target_date: date
    bucket_label: str
    model_prob: float
    market_price: float
    edge: float
    value_ratio: float
    confidence: float
    forecast_high_f: float
    forecast_sigma_f: float
    lead_days: int
    forecast_shift_f: float
    book_depth: float
    spread: float
    volume_24h: float
    smart_money_bias: float | None = None
    whale_activity: bool | None = None
    kalshi_confirms: bool | None = None
    cross_market_gap: float | None = None


class SignalEngine:
    """Detects mispricing between weather forecasts and Polymarket odds."""

    def __init__(
        self,
        noaa: NOAAFeed,
        model: WeatherModel,
        polymarket: PolymarketFeed,
        falcon: FalconFeed,
    ) -> None:
        self._noaa = noaa
        self._model = model
        self._polymarket = polymarket
        self._falcon = falcon
        self._callbacks: list[Callable[[Signal], Any]] = []
        self._open_positions: dict[str, float] = {}

    def on_signal(self, cb: Callable[[Signal], Any]) -> None:
        self._callbacks.append(cb)

    def set_open_positions(self, positions: dict[str, float]) -> None:
        """Update known open positions {contract_id: entry_price}."""
        self._open_positions = dict(positions)

    def scan(self) -> list[Signal]:
        """Run a full scan cycle across all active contracts.

        Returns:
            List of BUY/SELL signals detected this cycle.
        """
        signals: list[Signal] = []
        today = datetime.now(timezone.utc).date()

        for loc_key, target_date in self._polymarket.get_active_location_dates():
            lead_days = (target_date - today).days
            if lead_days < 0 or lead_days > settings.MAX_LEAD_DAYS:
                continue

            forecast = self._noaa.get_latest(loc_key, target_date)
            if not forecast:
                continue

            probs = self._model.compute_probabilities(
                location=loc_key,
                target_date=target_date,
                forecast_high_f=forecast.forecasted_high_f,
                lead_days=lead_days,
                model_agreement=forecast.model_agreement,
            )
            if not probs:
                continue

            for contract_id, model_prob in probs.buckets.items():
                snap = self._polymarket.get_snapshot(contract_id)
                if not snap:
                    continue

                falcon = self._falcon.get_intel(contract_id) if self._falcon.is_enabled else neutral_intel(contract_id)

                buy_signals = self._check_buy(
                    contract_id, model_prob, snap, probs, forecast, lead_days, falcon
                )
                if buy_signals:
                    signals.append(buy_signals)

                sell_signal = self._check_sell(contract_id, model_prob, snap, probs, forecast, lead_days)
                if sell_signal:
                    signals.append(sell_signal)

        for sig in signals:
            for cb in self._callbacks:
                try:
                    cb(sig)
                except Exception:
                    logger.exception("Signal callback error")

        return signals

    def _check_buy(
        self,
        contract_id: str,
        model_prob: float,
        snap: MarketSnapshot,
        probs: BucketProbabilities,
        forecast: Any,
        lead_days: int,
        falcon: FalconIntel,
    ) -> Signal | None:
        if contract_id in self._open_positions:
            return None

        market_price = snap.mid_price
        if market_price <= 0:
            return None

        edge = model_prob - market_price
        value_ratio = model_prob / market_price

        if market_price >= settings.ENTRY_THRESHOLD:
            return None
        if value_ratio < settings.MIN_VALUE_RATIO:
            return None
        if edge < settings.MIN_EDGE:
            return None

        confidence = self._compute_confidence(probs, snap, falcon)
        if confidence < settings.MIN_CONFIDENCE:
            logger.debug(
                "SKIP %s %s: confidence %.2f < %.2f",
                snap.location, snap.bucket_label, confidence, settings.MIN_CONFIDENCE,
            )
            return None

        return Signal(
            timestamp=datetime.now(timezone.utc),
            signal_type="BUY",
            contract_id=contract_id,
            location=snap.location,
            target_date=snap.target_date,
            bucket_label=snap.bucket_label,
            model_prob=model_prob,
            market_price=market_price,
            edge=edge,
            value_ratio=value_ratio,
            confidence=confidence,
            forecast_high_f=forecast.forecasted_high_f,
            forecast_sigma_f=probs.sigma_f,
            lead_days=lead_days,
            forecast_shift_f=forecast.forecast_shift_f,
            book_depth=snap.depth_usd,
            spread=snap.spread,
            volume_24h=snap.volume_24h,
            smart_money_bias=falcon.smart_money_bias if self._falcon.is_enabled else None,
            whale_activity=falcon.whale_activity if self._falcon.is_enabled else None,
            kalshi_confirms=falcon.kalshi_confirms_direction if self._falcon.is_enabled else None,
            cross_market_gap=falcon.cross_market_gap if self._falcon.is_enabled else None,
        )

    def _check_sell(
        self,
        contract_id: str,
        model_prob: float,
        snap: MarketSnapshot,
        probs: BucketProbabilities,
        forecast: Any,
        lead_days: int,
    ) -> Signal | None:
        entry_price = self._open_positions.get(contract_id)
        if entry_price is None:
            return None

        should_exit = False
        if snap.mid_price >= settings.EXIT_THRESHOLD:
            should_exit = True
        if model_prob < settings.STOP_LOSS_RATIO * entry_price:
            should_exit = True

        if not should_exit:
            return None

        return Signal(
            timestamp=datetime.now(timezone.utc),
            signal_type="SELL",
            contract_id=contract_id,
            location=snap.location,
            target_date=snap.target_date,
            bucket_label=snap.bucket_label,
            model_prob=model_prob,
            market_price=snap.mid_price,
            edge=model_prob - snap.mid_price,
            value_ratio=model_prob / max(snap.mid_price, 0.001),
            confidence=0.0,
            forecast_high_f=forecast.forecasted_high_f,
            forecast_sigma_f=probs.sigma_f,
            lead_days=lead_days,
            forecast_shift_f=forecast.forecast_shift_f,
            book_depth=snap.depth_usd,
            spread=snap.spread,
            volume_24h=snap.volume_24h,
        )

    def _compute_confidence(
        self,
        probs: BucketProbabilities,
        snap: MarketSnapshot,
        falcon: FalconIntel,
    ) -> float:
        """Weighted confidence score 0-1."""
        scores: list[tuple[float, float]] = []

        # Forecast stability (0.25)
        scores.append((probs.forecast_stability, 0.25))

        # Model agreement (0.20)
        scores.append((1.0 if probs.model_agreement else 0.3, 0.20))

        # Lead time (0.20)
        if probs.lead_days <= 1:
            lead_score = 1.0
        elif probs.lead_days == 2:
            lead_score = 0.7
        elif probs.lead_days == 3:
            lead_score = 0.5
        else:
            lead_score = 0.2
        scores.append((lead_score, 0.20))

        # Book depth (0.10)
        depth_score = min(snap.depth_usd / 100.0, 1.0)
        scores.append((depth_score, 0.10))

        # Spread tightness (0.05)
        spread_score = max(1.0 - snap.spread * 5, 0.0)
        scores.append((spread_score, 0.05))

        if self._falcon.is_enabled:
            # Smart money alignment (0.10)
            sm_score = (falcon.smart_money_bias + 1.0) / 2.0
            scores.append((sm_score, 0.10))

            # Cross-market confirmation (0.10)
            cm_score = 1.0 if falcon.kalshi_confirms_direction else 0.3
            scores.append((cm_score, 0.10))
        else:
            # Redistribute Falcon weight to stability and agreement
            scores.append((probs.forecast_stability, 0.10))
            scores.append((1.0 if probs.model_agreement else 0.3, 0.10))

        return sum(s * w for s, w in scores)
