"""Forecast-to-probability distribution engine.

Converts NOAA point forecasts + uncertainty into bucket probabilities
that map directly to Polymarket temperature contract outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from scipy.stats import norm

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class BucketProbabilities:
    timestamp: datetime
    location: str
    target_date: date
    forecast_high_f: float
    sigma_f: float
    buckets: dict[str, float]
    lead_days: int
    model_agreement: bool
    forecast_stability: float


@dataclass
class BucketDef:
    contract_id: str
    location: str
    target_date: date
    temp_low_f: int
    temp_high_f: int
    label: str


class WeatherModel:
    """Builds probability distributions across temperature buckets."""

    def __init__(self) -> None:
        self._buckets: dict[str, list[BucketDef]] = {}
        self._recent_shifts: dict[tuple[str, date], list[float]] = {}

    def register_buckets(self, buckets: list[BucketDef]) -> None:
        """Register temperature buckets parsed from Polymarket contracts."""
        self._buckets.clear()
        for b in buckets:
            key = f"{b.location}_{b.target_date}"
            self._buckets.setdefault(key, []).append(b)
        logger.info("Registered %d buckets across %d market groups", len(buckets), len(self._buckets))

    def record_shift(self, location: str, target_date: date, shift_f: float) -> None:
        """Record a forecast shift for stability tracking."""
        key = (location, target_date)
        history = self._recent_shifts.setdefault(key, [])
        history.append(abs(shift_f))
        if len(history) > 10:
            self._recent_shifts[key] = history[-10:]

    def compute_probabilities(
        self,
        location: str,
        target_date: date,
        forecast_high_f: float,
        lead_days: int,
        model_agreement: bool,
    ) -> BucketProbabilities | None:
        """Compute bucket probabilities for a single location + date.

        Args:
            location: City key (e.g. "NYC").
            target_date: The date the contract resolves.
            forecast_high_f: Best-estimate high temperature in Fahrenheit.
            lead_days: Days until target_date.
            model_agreement: Whether NWS and Open-Meteo agree within 2F.

        Returns:
            BucketProbabilities or None if no buckets registered.
        """
        key = f"{location}_{target_date}"
        bucket_list = self._buckets.get(key)
        if not bucket_list:
            return None

        sigma = self._compute_sigma(location, target_date, lead_days, model_agreement)
        stability = self._compute_stability(location, target_date)

        raw_probs: dict[str, float] = {}
        total = 0.0
        for b in bucket_list:
            p = self._bucket_probability(forecast_high_f, sigma, b.temp_low_f, b.temp_high_f)
            raw_probs[b.contract_id] = p
            total += p

        if total > 0:
            normalized = {cid: p / total for cid, p in raw_probs.items()}
        else:
            n = len(raw_probs)
            normalized = {cid: 1.0 / n for cid in raw_probs} if n > 0 else {}

        return BucketProbabilities(
            timestamp=datetime.now(timezone.utc),
            location=location,
            target_date=target_date,
            forecast_high_f=forecast_high_f,
            sigma_f=sigma,
            buckets=normalized,
            lead_days=lead_days,
            model_agreement=model_agreement,
            forecast_stability=stability,
        )

    def _compute_sigma(
        self,
        location: str,
        target_date: date,
        lead_days: int,
        model_agreement: bool,
    ) -> float:
        """Determine forecast standard deviation based on lead time and conditions."""
        sigma = settings.FORECAST_SIGMA.get(lead_days, settings.FORECAST_SIGMA_DEFAULT)

        if not model_agreement:
            sigma += settings.SIGMA_DISAGREE_PENALTY

        stability = self._compute_stability(location, target_date)
        if stability > 0.8:
            sigma -= settings.SIGMA_STABILITY_BONUS

        return max(sigma, 0.5)

    def _compute_stability(self, location: str, target_date: date) -> float:
        """Return 0-1 stability score. Higher = forecast has been stable."""
        key = (location, target_date)
        shifts = self._recent_shifts.get(key, [])
        if len(shifts) < 2:
            return 0.5
        last_three = shifts[-3:]
        max_shift = max(last_three) if last_three else 0.0
        if max_shift <= 1.0:
            return 1.0
        if max_shift >= 4.0:
            return 0.0
        return 1.0 - (max_shift - 1.0) / 3.0

    @staticmethod
    def _bucket_probability(mean: float, sigma: float, low_f: int, high_f: int) -> float:
        """P(low - 0.5 <= X <= high + 0.5) under N(mean, sigma)."""
        return float(norm.cdf(high_f + 0.5, loc=mean, scale=sigma) - norm.cdf(low_f - 0.5, loc=mean, scale=sigma))
